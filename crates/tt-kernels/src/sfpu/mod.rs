//! SFPU programs: a builder whose output is correct by construction, and an
//! interpreter that says what any program computes (`hardware-coverage.md` F2,
//! F5).
//!
//! The builder ([`Program`]) is where the Vector Unit's hazards live, as code:
//!
//! - registers are [`LReg`]s, so a write to a constant or to `LReg[11..15]`,
//!   which the hardware drops without a word, does not compile into a
//!   program -- it panics while building it;
//! - conditional execution is a scope ([`Program::if_`], [`Program::if_else`]):
//!   `SFPPUSHC`, `SFPSETCC`, `SFPCOMPC` and `SFPPOPC` are emitted balanced, the
//!   flag stack's depth is tracked, and a ninth level is refused, as is a
//!   complex `SFPPOPC` mode on a full stack (Tier 2; `SFPPOPC.md` says
//!   Blackhole fixed it, and in the same page's summary that it did not);
//! - an `SFPNOP` goes exactly where `SFPMAD`'s automatic stalling misses a
//!   dependency (`Instruction::stalls_automatically_after_mad`) and nowhere
//!   else;
//! - a loop over a tile's row groups ([`Program::for_each_row_group`]) is
//!   written once, recorded into the replay buffer and replayed (X1), the
//!   `Dst` row stepped by an address modifier on its last `Dst` access -- or,
//!   when the body does not fit the buffer, unrolled, and the program says
//!   which ([`Program::loops`]).
//!
//! The interpreter ([`interp`]) runs a program's instruction stream -- the
//! same words the device runs, `REPLAY` expanded by its own model of the
//! buffer -- over a model of the Vector Unit and `Dst` built from the
//! specification's functional models, so a kernel's expected output is
//! computed, never hand-derived. It is checked instruction by instruction
//! against ttsim and both cards (`step26_sfpu_isa`).

pub mod interp;
pub mod kernel;
pub mod ops;

use tt_isa::frontend;
use tt_isa::isa::generated::{defs, encode};
use tt_isa::isa::Instruction;
use tt_isa::sfpu::{self as enc, mad_mod1, mod0_fmt};
pub use tt_isa::sfpu::{ConfigLReg, LReg};

use crate::datapath::thread_entry;

/// Depth of the per-lane flag stack (`SFPPUSHC.md`).
pub const FLAG_STACK: u8 = 8;

/// `Dst` row groups and column halves in one FP32 tile: sixty-four rows, four
/// to a group, two halves each.
pub const TILE_ITERATIONS: u32 = 32;

/// The address modifier the row loop steps `Dst` by: entry 0 is kept at no
/// increment for every other `SFPLOAD`/`SFPSTORE`, and this one adds two
/// rows' worth of address -- the next column half, or after the odd half the
/// next group of four rows.
pub const ROW_LOOP_ADDR_MOD: u32 = 7;

/// How a `Dst` datum is moved between `Dst` and an `LReg` (`SFPLOAD.md`'s
/// `MOD0_FMT_*`). Only the formats some kernel uses are offered.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Format {
    /// FP32 in `Dst`; a store flushes denormals to signed zero.
    Fp32,
    /// The 32 bits as they are, either way.
    Int32,
}

impl Format {
    fn mod0(self) -> u32 {
        match self {
            Format::Fp32 => mod0_fmt::FP32,
            Format::Int32 => mod0_fmt::INT32,
        }
    }
}

/// A lane condition, on an `LReg`'s 32 bits read as a signed integer
/// (`SFPSETCC.md`). For FP32 that is a test of the sign bit and of all-zero
/// bits: [`Cond::Lt0`] holds for `-0.0` and for a negative NaN, [`Cond::Eq0`]
/// only for `+0.0`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Cond {
    Lt0(LReg),
    Ne0(LReg),
    Gte0(LReg),
    Eq0(LReg),
    /// `a < b`, the 32 bits of each read as sign-magnitude integers: for FP32
    /// the total order `-NaN < -inf < ... < -0 < +0 < ... < +inf < +NaN`
    /// (Blackhole's `SFPGT`, `SFPGT_MOD1_SET_CC`).
    Less(LReg, LReg),
}

impl Cond {
    fn setcc(self) -> Instruction {
        // `SFPSETCC_MOD1_LREG_*`.
        let (vc, mod1) = match self {
            Cond::Lt0(r) => (r, 0),
            Cond::Ne0(r) => (r, 2),
            Cond::Gte0(r) => (r, 4),
            Cond::Eq0(r) => (r, 6),
            Cond::Less(a, b) => return encode::sfpgt(a.index(), b.index(), 1).unwrap(),
        };
        encode::sfpsetcc(0, vc.index(), 0, mod1).unwrap()
    }
}

/// How a program's row loops came out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoopForm {
    /// Recorded once and replayed: one `REPLAY` per further iteration.
    Replayed { body: usize },
    /// Written out, every iteration: the body did not fit the buffer, or the
    /// program asked for it.
    Unrolled { body: usize },
}

/// Which form a row loop should take.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoopPolicy {
    /// Replayed when the body fits, unrolled otherwise.
    Replay,
    /// Always unrolled: immediate addresses, no address modifiers, no replay.
    Unrolled,
}

/// An SFPU program for the math thread, under construction. See the module
/// documentation for what it guarantees.
#[derive(Clone, Debug)]
pub struct Program {
    ins: Vec<Instruction>,
    /// Flag-stack entries pushed and not popped.
    depth: u8,
    /// The last instruction was one whose result `SFPMAD`'s stalling covers
    /// (the MAD sub-unit's), so the next may need an `SFPNOP`.
    after_mad: bool,
    /// Inside a row loop's body: no nested loop, no `REPLAY`.
    in_loop: bool,
    loops: Vec<LoopForm>,
    policy: LoopPolicy,
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Program {
    /// A program whose lanes are all enabled, driven by `LaneFlags` from the
    /// start, as `VectorUnit.md` recommends: `SFPENCC` setting both
    /// `UseLaneFlagsForLaneEnable` and `LaneFlags`.
    pub fn new() -> Self {
        Program::with_policy(LoopPolicy::Replay)
    }

    pub fn with_policy(policy: LoopPolicy) -> Self {
        let mut p = Program {
            ins: Vec::new(),
            depth: 0,
            after_mad: false,
            in_loop: false,
            loops: Vec::new(),
            policy,
        };
        // `SFPENCC_MOD1_EI | SFPENCC_MOD1_RI`, both immediate bits set.
        p.push(encode::sfpencc(3, 0, 2 | 8).unwrap());
        // The thread's address modifier 0 steps nothing and its `Dst` row
        // counter is zero, so an `SFPLOAD`/`SFPSTORE` address is absolute --
        // whatever the kernel before left (a matmul steps the counter).
        p.ins
            .push(thread_entry(crate::datapath::addr_mod_entry(0).dst_incr, 0));
        p.ins.push(clear_dst_rwc());
        p
    }

    /// Append `i`, first inserting the `SFPNOP` automatic stalling would miss.
    fn push(&mut self, i: Instruction) {
        if self.after_mad && !i.stalls_automatically_after_mad() {
            self.ins.push(enc::nop());
        }
        self.after_mad = matches!(
            i.def().mnemonic(),
            "SFPMAD"
                | "SFPMUL"
                | "SFPADD"
                | "SFPMULI"
                | "SFPADDI"
                | "SFPMUL24"
                | "SFPLUT"
                | "SFPLUTFP32"
        );
        self.ins.push(i);
    }

    fn dst(d: LReg) -> u32 {
        assert!(
            d.writable(),
            "LReg[{}] cannot be written: the hardware would drop it",
            d.index()
        );
        d.index()
    }

    /// `d = Dst[at]` for this lane's datum: `at` is a `Dst` address, its high
    /// eight bits a four-row group and bit 1 the odd columns
    /// (`SFPLOAD.md`).
    pub fn load(&mut self, d: LReg, fmt: Format, at: u32) {
        self.push(enc::load(Self::dst(d), fmt.mod0(), 0, at).unwrap());
    }

    /// `Dst[at] = s`, as [`Program::load`] addresses it.
    pub fn store(&mut self, s: LReg, fmt: Format, at: u32) {
        self.push(enc::store(s.index(), fmt.mod0(), 0, at).unwrap());
    }

    /// `d = v` in every enabled lane: one `SFPLOADI` when `v`'s low 16 bits
    /// are zero (a BF16 immediate), two otherwise.
    pub fn loadi(&mut self, d: LReg, v: f32) {
        self.loadi_bits(d, v.to_bits());
    }

    /// `d = bits` in every enabled lane.
    pub fn loadi_bits(&mut self, d: LReg, bits: u32) {
        let vd = Self::dst(d);
        if bits & 0xffff == 0 {
            self.push(enc::loadi(vd, enc::loadi_mode::FLOATB, bits >> 16).unwrap());
        } else {
            for i in enc::load_f32(vd, bits).unwrap() {
                self.push(i);
            }
        }
    }

    /// `d = a * b + c`, `fma_bh`'s rounding (`SFPMAD.md`).
    pub fn mad(&mut self, a: LReg, b: LReg, c: LReg, d: LReg) {
        self.push(enc::mad(a.index(), b.index(), c.index(), Self::dst(d), 0).unwrap());
    }

    /// `d = a * b`, keeping the sign of a zero product (`-0` addend, numerics
    /// row C).
    pub fn mul(&mut self, a: LReg, b: LReg, d: LReg) {
        self.push(enc::mul(a.index(), b.index(), Self::dst(d)).unwrap());
    }

    /// `d = a + b`.
    pub fn add(&mut self, a: LReg, b: LReg, d: LReg) {
        self.push(enc::add(a.index(), b.index(), Self::dst(d)).unwrap());
    }

    /// `d = a - b`.
    pub fn sub(&mut self, a: LReg, b: LReg, d: LReg) {
        self.push(enc::sub(a.index(), b.index(), Self::dst(d)).unwrap());
    }

    /// `d = -(a * b) + c`.
    pub fn nmad(&mut self, a: LReg, b: LReg, c: LReg, d: LReg) {
        self.push(
            enc::mad(
                a.index(),
                b.index(),
                c.index(),
                Self::dst(d),
                mad_mod1::NEGATE_VA,
            )
            .unwrap(),
        );
    }

    /// `d = s`.
    pub fn mov(&mut self, s: LReg, d: LReg) {
        self.push(encode::sfpmov(s.index(), Self::dst(d), 0).unwrap());
    }

    /// `d = s` with its sign bit flipped: FP32 negation, NaNs included.
    pub fn neg(&mut self, s: LReg, d: LReg) {
        // `SFPMOV_MOD1_NEGATE`.
        self.push(encode::sfpmov(s.index(), Self::dst(d), 1).unwrap());
    }

    /// `d = |s|` as FP32: the sign cleared, except that a negative NaN stays
    /// one (`SFPABS.md`, `SFPABS_MOD1_FLOAT`).
    pub fn abs(&mut self, s: LReg, d: LReg) {
        self.push(encode::sfpabs(s.index(), Self::dst(d), 1).unwrap());
    }

    /// `d = s` with sign bit `negative` (`SFPSETSGN_MOD1_ARG_IMM`).
    pub fn set_sign(&mut self, s: LReg, negative: bool, d: LReg) {
        self.push(encode::sfpsetsgn(u32::from(negative), s.index(), Self::dst(d), 1).unwrap());
    }

    /// `d = s` with the sign bit of `sign` (`SFPSETSGN` taking it from `VD`,
    /// which is why `sign` is moved into `d` first).
    pub fn copy_sign(&mut self, s: LReg, sign: LReg, d: LReg) {
        if sign != d {
            self.mov(sign, d);
        }
        self.push(encode::sfpsetsgn(0, s.index(), Self::dst(d), 0).unwrap());
    }

    /// `d = ApproxRecip(|x|)` with `x`'s sign (`SFPARECIP`, Blackhole only):
    /// within 0.56% of `1/x` for `2^-126 <= |x| < 2^126`, infinite below and
    /// zero above.
    pub fn approx_recip(&mut self, x: LReg, d: LReg) {
        let mod1 = tt_isa::numerics::sfpu::arecip_mod1::RECIP;
        self.push(encode::sfparecip(0, x.index(), Self::dst(d), mod1).unwrap());
    }

    /// `d = 1/x`, within one ulp of the correctly rounded reciprocal for
    /// every normal `x` whose reciprocal is normal; `1/±0 = ±inf`, `1/±inf =
    /// ±0`, a NaN stays one; denormals in or out flush to zero, as all of the
    /// SFPU's arithmetic does. Uses `t0`, `t1` as scratch, and needs `one`
    /// holding `1.0` and `max` holding `f32::MAX` (loaded once, outside a
    /// loop).
    ///
    /// The seed ([`Program::approx_recip`]) is within `e0 < 0.0056`; each
    /// Newton step `y += y * (1 - x*y)` squares the error and adds at most
    /// two roundings (`e1 <= e0^2 + 2^-23 < 3.2e-5`), and the last step's
    /// result is `1/x * (1 - e1^2)` before its single rounding, `e1^2 < 1.1e-9`
    /// -- so within half an ulp plus 0.02 of one of `1/x`, hence at most one
    /// ulp from its correct rounding. (`step28_recip` holds the device to
    /// this program bit for bit, and the program to the bound.)
    pub fn recip(&mut self, x: LReg, d: LReg, t0: LReg, t1: LReg, max: LReg) {
        assert!(d != x && t0 != x && t1 != x && d != t0 && d != t1 && t0 != t1);
        self.approx_recip(x, d);
        for _ in 0..2 {
            self.nmad(x, d, LReg::ONE, t0);
            self.mad(t0, d, d, d);
        }
        // `1/±0`, and a denormal, which the arithmetic flushes to a zero: the
        // seed's infinity met `0 * inf` in the steps.
        self.abs(x, t1);
        self.loadi_bits(t0, 0x0080_0000);
        self.if_(Cond::Less(t1, t0), |p| {
            p.loadi_bits(t0, 0x7f80_0000);
            p.copy_sign(t0, x, d);
        });
        // `1/±inf`: the seed's zero met `inf * 0`.
        self.if_(Cond::Less(max, t1), |p| {
            p.loadi_bits(t0, 0x7f80_0001);
            p.if_(Cond::Less(t1, t0), |p| {
                p.copy_sign(LReg::ZERO, x, d);
            });
        });
    }

    fn push_flags(&mut self) {
        assert!(
            self.depth < FLAG_STACK,
            "conditional execution nested deeper than the {FLAG_STACK}-entry flag stack"
        );
        self.push(encode::sfppushc(0, 0).unwrap());
        self.depth += 1;
    }

    fn pop_flags(&mut self) {
        assert!(self.depth > 0, "a flag-stack pop with nothing pushed");
        self.push(encode::sfppopc(0, 0).unwrap());
        self.depth -= 1;
    }

    /// Run `then` only in the lanes where `cond` holds (among those enabled
    /// now). The lanes' predication is saved first and restored after, so a
    /// scope leaves the program as it found it.
    pub fn if_(&mut self, cond: Cond, then: impl FnOnce(&mut Self)) {
        self.push_flags();
        self.push(cond.setcc());
        then(self);
        self.pop_flags();
    }

    /// `then` where `cond` holds, `otherwise` in the other enabled lanes.
    /// `SFPCOMPC` inverts `LaneFlags` against the saved state, so a lane that
    /// was disabled before the scope runs neither branch.
    pub fn if_else(
        &mut self,
        cond: Cond,
        then: impl FnOnce(&mut Self),
        otherwise: impl FnOnce(&mut Self),
    ) {
        self.push_flags();
        self.push(cond.setcc());
        then(self);
        self.push(encode::sfpcompc(0).unwrap());
        otherwise(self);
        self.pop_flags();
    }

    /// `body`, once per row group and column half of the `rows` `Dst` rows
    /// (64 per FP32 tile): `body` is handed the address offset to add to
    /// every `Dst` address it uses (`0` when replayed, where the row counter
    /// does the stepping instead).
    ///
    /// Replayed (X1) when the policy allows and the body fits the 32-entry
    /// buffer: the row counter `RWCs.Dst` is cleared, address modifier
    /// [`ROW_LOOP_ADDR_MOD`] set to step it by two and entry 0 to step it by
    /// nothing, the body recorded while it runs its first iteration -- its
    /// last `Dst` access carrying the stepping modifier -- and replayed for
    /// the rest; the counter is cleared again after, so later addresses are
    /// absolute. Unrolled otherwise, with the offset in each address.
    pub fn for_each_row_group(&mut self, rows: u32, body: impl Fn(&mut Self, u32)) {
        assert!(!self.in_loop, "a row loop inside a row loop");
        assert!(
            rows % 4 == 0 && rows > 0,
            "{rows} rows are not whole groups of four"
        );
        let iterations = rows / 2;
        // The body once, at offset 0, to measure and patch.
        let mut b = Program {
            ins: Vec::new(),
            depth: self.depth,
            after_mad: self.after_mad,
            in_loop: true,
            loops: Vec::new(),
            policy: self.policy,
        };
        body(&mut b, 0);
        assert_eq!(
            b.depth, self.depth,
            "a row loop's body leaves the flag stack as it found it"
        );
        let last_dst = b
            .ins
            .iter()
            .rposition(|i| {
                core::ptr::eq(i.def(), &defs::SFPLOAD) || core::ptr::eq(i.def(), &defs::SFPSTORE)
            })
            .expect("a row loop's body reads or writes Dst");
        // The body was built following what precedes the loop, so a NOP it
        // needs at its start is already in it. Replayed, it also follows its
        // own end: a body ending in a multiply-add whose first instruction
        // stalling would miss needs one more, at its end.
        let wraps_nop = b.after_mad
            && b.ins
                .first()
                .is_some_and(|i| !i.stalls_automatically_after_mad());
        let fits = b.ins.len() + usize::from(wraps_nop) <= frontend::REPLAY_BUFFER as usize;
        if self.policy == LoopPolicy::Unrolled || !fits {
            for k in 0..iterations {
                body(self, 2 * k);
            }
            self.loops.push(LoopForm::Unrolled { body: b.ins.len() });
            return;
        }
        let mut ins = b.ins;
        let step = ins[last_dst];
        ins[last_dst] = Instruction::new(step.word() | (ROW_LOOP_ADDR_MOD << 13), step.def());
        if wraps_nop {
            ins.push(enc::nop());
        }
        // Not SFPU instructions: they leave the multiply-add bookkeeping as
        // it was, and the body already allowed for it.
        let row = crate::datapath::addr_mod_entry(0);
        let stepping = crate::datapath::addr_mod_entry(ROW_LOOP_ADDR_MOD as usize);
        self.ins.push(thread_entry(row.dst_incr, 0));
        self.ins.push(thread_entry(stepping.dst_incr, 2));
        self.ins.push(clear_dst_rwc());
        frontend::record(0, &ins, true, &mut self.ins).expect("fits, and a body has no REPLAY");
        let r = frontend::replay(0, ins.len()).unwrap();
        for _ in 1..iterations {
            self.ins.push(r);
        }
        self.ins.push(clear_dst_rwc());
        self.after_mad = ins.last().is_some_and(is_mad_unit);
        self.loops.push(LoopForm::Replayed { body: ins.len() });
    }

    /// The forms the program's row loops took, in order.
    pub fn loops(&self) -> &[LoopForm] {
        &self.loops
    }

    /// The instructions, for the math thread. Refuses a program that leaves
    /// anything on the flag stack.
    pub fn finish(self) -> Vec<Instruction> {
        assert_eq!(self.depth, 0, "a program ends with the flag stack empty");
        self.ins
    }
}

fn is_mad_unit(i: &Instruction) -> bool {
    matches!(
        i.def().mnemonic(),
        "SFPMAD"
            | "SFPMUL"
            | "SFPADD"
            | "SFPMULI"
            | "SFPADDI"
            | "SFPMUL24"
            | "SFPLUT"
            | "SFPLUTFP32"
    )
}

/// `SETRWC` clearing the issuing thread's `Dst` row counter and nothing else.
fn clear_dst_rwc() -> Instruction {
    encode::Setrwc::ZERO.dst(1).encode().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mnemonics(p: &[Instruction]) -> Vec<&'static str> {
        p.iter().map(|i| i.def().mnemonic()).collect()
    }

    #[test]
    #[should_panic(expected = "cannot be written")]
    fn a_write_to_a_constant_is_refused() {
        Program::new().mov(LReg::L0, LReg::ONE);
    }

    #[test]
    fn scopes_balance_and_nest() {
        let mut p = Program::new();
        p.if_else(
            Cond::Lt0(LReg::L0),
            |p| p.if_(Cond::Eq0(LReg::L1), |p| p.mov(LReg::ZERO, LReg::L0)),
            |p| p.neg(LReg::L0, LReg::L0),
        );
        let m = mnemonics(&p.finish());
        let push = m.iter().filter(|&&x| x == "SFPPUSHC").count();
        let pop = m.iter().filter(|&&x| x == "SFPPOPC").count();
        assert_eq!((push, pop), (2, 2));
        assert_eq!(m.iter().filter(|&&x| x == "SFPCOMPC").count(), 1);
    }

    #[test]
    #[should_panic(expected = "deeper than")]
    fn a_ninth_level_is_refused() {
        fn nest(p: &mut Program, n: u32) {
            if n > 0 {
                p.if_(Cond::Lt0(LReg::L0), |p| nest(p, n - 1));
            }
        }
        nest(&mut Program::new(), 9);
    }

    /// A NOP exactly where automatic stalling misses: after a multiply-add
    /// and before the cases `SFPMAD.md` lists, and nowhere else.
    #[test]
    fn nops_go_where_stalling_misses_and_nowhere_else() {
        let mut p = Program::new();
        p.mul(LReg::L0, LReg::L1, LReg::L2);
        p.store(LReg::L2, Format::Fp32, 0);
        p.add(LReg::L0, LReg::L1, LReg::L2);
        p.push(encode::sfpiadd(0, 2, 2, 0).unwrap());
        let m = mnemonics(&p.finish());
        let nops: Vec<usize> = m
            .iter()
            .enumerate()
            .filter(|(_, x)| **x == "SFPNOP")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(nops.len(), 1, "{m:?}");
        assert_eq!(m[nops[0] + 1], "SFPIADD");
        assert!(matches!(m[nops[0] - 1], "SFPMAD" | "SFPADD"));
    }

    #[test]
    fn a_row_loop_replays_when_it_fits_and_unrolls_when_it_does_not() {
        let mut p = Program::new();
        p.for_each_row_group(64, |p, o| {
            p.load(LReg::L0, Format::Fp32, o);
            p.load(LReg::L1, Format::Fp32, 64 + o);
            p.add(LReg::L0, LReg::L1, LReg::L0);
            p.store(LReg::L0, Format::Fp32, o);
        });
        assert_eq!(p.loops(), &[LoopForm::Replayed { body: 4 }]);
        let ins = p.finish();
        let m = mnemonics(&ins);
        assert_eq!(
            m.iter().filter(|&&x| x == "REPLAY").count(),
            32,
            "one record, 31 replays"
        );
        let store = ins
            .iter()
            .find(|i| i.def().mnemonic() == "SFPSTORE")
            .unwrap();
        assert_eq!(
            store.operand("AddrMod"),
            Some(ROW_LOOP_ADDR_MOD),
            "the last Dst access steps"
        );
        let load = ins
            .iter()
            .find(|i| i.def().mnemonic() == "SFPLOAD")
            .unwrap();
        assert_eq!(load.operand("AddrMod"), Some(0));

        let mut p = Program::new();
        p.for_each_row_group(64, |p, o| {
            for k in 0..40 {
                p.load(LReg::L0, Format::Fp32, o + 4 * (k % 2));
            }
        });
        assert_eq!(p.loops(), &[LoopForm::Unrolled { body: 40 }]);
        assert_eq!(
            p.finish().len(),
            3 + 32 * 40,
            "the prologue and every iteration"
        );

        let mut p = Program::with_policy(LoopPolicy::Unrolled);
        p.for_each_row_group(4, |p, o| p.store(LReg::L0, Format::Fp32, o));
        let ins = p.finish();
        let addrs: Vec<_> = ins
            .iter()
            .skip(3)
            .map(|i| i.operand("Imm10").unwrap())
            .collect();
        assert_eq!(addrs, [0, 2], "group 0, both halves");
    }
}
