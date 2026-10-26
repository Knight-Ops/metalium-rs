//! The accumulation order of a matmul kernel, read off the instruction streams its unpack and
//! math roles hand the coprocessor (`docs/plans/prefill-operand-reuse.md` P4, step B1).
//!
//! A reordered pair is bit for bit the old one if and only if every `Dst` 8x16 block takes the
//! same terms in the same order: each `MVMUL` adds one rounded FP32 term to its block, so the
//! block's value is the fold of its terms in issue order, and nothing else reaches it. This
//! module interprets the two streams symbolically -- the unpackers' `Src` writes and bank
//! handovers, the Matrix Unit's RWCs, address modifiers (`RWCs.md`, `ApplyAddrMod`, eight entries
//! on Blackhole), fidelity phase and bank releases (`MVMUL.md`, `SETRWC.md`) -- and yields, per
//! output tile and `Dst` block, the ordered list of `(A face, half, B face, phase)` terms each
//! `MVMUL` adds. It shares nothing with `pair_code` but the instruction definitions: it reads the
//! words, not the builder, so it is the oracle a reordering is held to, and [`canonical`] states
//! the order every kernel must produce from first principles.
//!
//! The two roles run concurrently on silicon, handing `Src` banks over through ownership
//! (`SrcASrcB.md`); the interpreter steps whichever role can proceed, which yields the same terms
//! whatever the interleaving, since a bank's rows are only read while the Matrix Unit owns it.

use std::collections::BTreeMap;

use tt_isa::cfg::generated::{thcon, ALL_THREAD_CONFIG_FIELDS};
use tt_isa::isa::Instruction;

/// One `MVMUL`'s term in a `Dst` block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term {
    /// `A`'s tile (`REG3_Base_address` of unpacker 1), its face (`Z`) and the face's 8-row half.
    pub a: u32,
    pub a_face: u32,
    pub a_half: u32,
    /// `B`'s tile (unpacker 0's base), the face its slot holds (`Z`), and whether the unpacker
    /// transposed it (`Haloize_mode`).
    pub b: u32,
    pub b_face: u32,
    pub b_transposed: bool,
    /// The fidelity phase the term was multiplied at.
    pub phase: u32,
}

/// Each output tile's `Dst` blocks (row / 8) and their terms in order, output by output: an output
/// ends where `Dst` is cleared whole (`ZEROACC` mode 3).
pub type Order = Vec<BTreeMap<u32, Vec<Term>>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SrcRow {
    base: u32,
    face: u32,
    row: u32,
    transposed: bool,
}

struct Bank {
    matrix_owns: bool,
    rows: [Option<SrcRow>; 64],
}

impl Bank {
    fn new() -> Self {
        Bank {
            matrix_owns: false,
            rows: [None; 64],
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Adc {
    x0: u32,
    x1: u32,
    z: u32,
}

struct Machine {
    /// `[SrcA, SrcB][bank]`.
    src: [[Bank; 2]; 2],
    /// The unpackers' bank pointers (0 writes `SrcA`, 1 `SrcB`) and the Matrix Unit's.
    unpacker_bank: [usize; 2],
    matrix_bank: [usize; 2],
    // Unpack role.
    gpr: [u32; 64],
    config: BTreeMap<u16, u32>,
    adc: [Adc; 2],
    // Math role.
    thread: BTreeMap<u16, u16>,
    rwc: Rwc,
    blocks: BTreeMap<u32, Vec<Term>>,
    outputs: Order,
}

#[derive(Default)]
struct Rwc {
    src_a: u32,
    src_a_cr: u32,
    src_b: u32,
    src_b_cr: u32,
    dst: u32,
    dst_cr: u32,
    fidelity: u32,
}

fn op(i: Instruction, name: &str) -> u32 {
    i.operand(name)
        .unwrap_or_else(|| panic!("{} has no {name}", i.def().key()))
}

impl Machine {
    fn new() -> Self {
        Machine {
            src: [[Bank::new(), Bank::new()], [Bank::new(), Bank::new()]],
            unpacker_bank: [0; 2],
            matrix_bank: [0; 2],
            gpr: [0; 64],
            config: BTreeMap::new(),
            adc: [Adc::default(); 2],
            thread: BTreeMap::new(),
            rwc: Rwc::default(),
            blocks: BTreeMap::new(),
            outputs: Vec::new(),
        }
    }

    fn cfg(&self, f: tt_isa::cfg::ConfigField) -> u32 {
        f.extract(self.config.get(&f.addr32()).copied().unwrap_or(0))
    }

    fn thread_field(&self, name: &str) -> u32 {
        let f = ALL_THREAD_CONFIG_FIELDS
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("no thread field {name}"))
            .1;
        u32::from(f.extract(self.thread.get(&f.addr32()).copied().unwrap_or(0)))
    }

    /// One unpack-role instruction; `false` if it must wait for a bank.
    fn unpack(&mut self, i: Instruction) -> bool {
        match i.def().key() {
            "SETDMAREG_Immediate" => {
                let (half, v) = (op(i, "ResultHalfReg"), op(i, "NewValue"));
                let (r, shift) = ((half / 2) as usize, 16 * (half % 2));
                self.gpr[r] = self.gpr[r] & !(0xFFFF << shift) | v << shift;
            }
            "ADDDMAREG" => {
                let (l, r) = (op(i, "LeftReg") as usize, op(i, "RightReg") as usize);
                self.gpr[op(i, "ResultReg") as usize] = self.gpr[l].wrapping_add(self.gpr[r]);
            }
            "WRCFG" => {
                assert_eq!(op(i, "Is128Bit"), 0);
                self.config.insert(
                    op(i, "CfgIndex") as u16,
                    self.gpr[op(i, "InputReg") as usize],
                );
            }
            "SETADCXX" => {
                for u in 0..2 {
                    if op(i, ["U0", "U1"][u]) == 1 {
                        self.adc[u].x0 = op(i, "X0Val");
                        self.adc[u].x1 = op(i, "X1Val");
                    }
                }
            }
            "SETADCZW" => {
                assert_eq!(
                    op(i, "Z1") | op(i, "W0") | op(i, "W1"),
                    0,
                    "only Z0 is modelled"
                );
                for u in 0..2 {
                    if op(i, ["U0", "U1"][u]) == 1 && op(i, "Z0") == 1 {
                        self.adc[u].z = op(i, "Z0Val");
                    }
                }
            }
            "UNPACR_Regular" => return self.unpacr(i),
            other => match i.def().mnemonic() {
                "SETC16" | "STALLWAIT" | "SEMWAIT" | "SEMPOST" | "SEMGET" | "SEMINIT" | "NOP"
                | "SFPNOP" => {}
                _ => panic!("the unpack role's {other} is not modelled"),
            },
        }
        true
    }

    fn unpacr(&mut self, i: Instruction) -> bool {
        let u = op(i, "WhichUnpacker") as usize;
        assert_eq!(
            op(i, "Ch0YInc") | op(i, "Ch1YInc") | op(i, "Ch1ZInc"),
            0,
            "only Ch0ZInc is modelled"
        );
        let to_dst = u == 0 && self.cfg(thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0) != 0;
        let bank = &mut self.src[u][self.unpacker_bank[u]];
        if bank.matrix_owns {
            return false;
        }
        if to_dst {
            // `UnpackToDst` leaves `SrcA[Bank]` unpredictable (`UNPACR_Regular.md`).
            bank.rows = [None; 64];
            return true;
        }
        if u == 0 {
            // `SrcA` row 0 is `OutAddr / 16 - 4`, the base replacing the address.
            assert_eq!(
                self.cfg(thcon::THCON_SEC0_REG5_Dest_cntx0_address),
                64,
                "SrcA rows from row 0"
            );
        }
        let base = self.cfg(if u == 0 {
            thcon::THCON_SEC0_REG3_Base_address
        } else {
            thcon::THCON_SEC1_REG3_Base_address
        });
        let transposed = u == 0 && self.cfg(thcon::THCON_SEC0_REG2_Haloize_mode) != 0;
        let adc = self.adc[u];
        let first = adc.z * 256 + adc.x0;
        let n = adc.x1 + 1 - adc.x0;
        assert!(
            first % 16 == 0 && n % 16 == 0 && n / 16 <= 64,
            "whole rows: {n} datums from {first}"
        );
        let bank = &mut self.src[u][self.unpacker_bank[u]];
        for r in 0..n / 16 {
            let d = first / 16 + r;
            bank.rows[r as usize] = Some(SrcRow {
                base,
                face: d / 16,
                row: d % 16,
                transposed,
            });
        }
        if op(i, "FlipSrc") == 1 {
            bank.matrix_owns = true;
            self.unpacker_bank[u] ^= 1;
        }
        self.adc[u].z += op(i, "Ch0ZInc");
        true
    }

    /// One math-role instruction; `false` if it must wait for a bank.
    fn math(&mut self, i: Instruction) -> bool {
        match i.def().mnemonic() {
            "MVMUL" => return self.mvmul(i),
            "SETRWC" => {
                let r = &mut self.rwc;
                if op(i, "SrcA") == 1 {
                    let v = op(i, "SrcAVal") + if op(i, "SrcACr") == 1 { r.src_a_cr } else { 0 };
                    (r.src_a, r.src_a_cr) = (v & 63, v & 63);
                }
                if op(i, "SrcB") == 1 {
                    let v = op(i, "SrcBVal") + if op(i, "SrcBCr") == 1 { r.src_b_cr } else { 0 };
                    (r.src_b, r.src_b_cr) = (v & 63, v & 63);
                }
                if op(i, "Dst") == 1 || op(i, "DstCtoCr") == 1 {
                    let mut v = op(i, "DstVal");
                    if op(i, "DstCtoCr") == 1 {
                        v += r.dst;
                    } else if op(i, "DstCr") == 1 {
                        v += r.dst_cr;
                    }
                    (r.dst, r.dst_cr) = (v & 1023, v & 1023);
                }
                if op(i, "Fidelity") == 1 {
                    r.fidelity = 0;
                }
                self.release(op(i, "FlipSrcA") == 1, op(i, "FlipSrcB") == 1);
            }
            "SETC16" => {
                self.thread
                    .insert(op(i, "CfgIndex") as u16, op(i, "NewValue") as u16);
            }
            "ZEROACC" => {
                assert_eq!(op(i, "Mode"), 3, "only a whole-Dst clear is modelled");
                let blocks = std::mem::take(&mut self.blocks);
                if !blocks.is_empty() {
                    self.outputs.push(blocks);
                }
            }
            "STALLWAIT" | "SEMWAIT" | "SEMPOST" | "SEMGET" | "SEMINIT" | "NOP" | "SFPNOP" => {}
            other => panic!("the math role's {other} is not modelled"),
        }
        true
    }

    fn release(&mut self, a: bool, b: bool) {
        for (s, flip) in [(0, a), (1, b)] {
            if flip {
                let bank = &mut self.src[s][self.matrix_bank[s]];
                assert!(
                    bank.matrix_owns,
                    "a release of a bank the Matrix Unit does not own"
                );
                bank.matrix_owns = false;
                bank.rows = [None; 64];
                self.matrix_bank[s] ^= 1;
            }
        }
    }

    fn mvmul(&mut self, i: Instruction) -> bool {
        let (a, b) = (
            &self.src[0][self.matrix_bank[0]],
            &self.src[1][self.matrix_bank[1]],
        );
        if !a.matrix_owns || !b.matrix_owns {
            return false;
        }
        assert_eq!(op(i, "BroadcastSrcBRow"), 0);
        // `MVMUL.md`: 16 rows of `SrcA` and 8 of `SrcB` from their RWCs, 8-aligned; ttsim asks
        // 16-aligned `SrcA` (divergence row 41), and a face is 16 rows.
        let ra = (self.rwc.src_a & 0x38) as usize;
        let rb = (self.rwc.src_b & 0x38) as usize;
        let face_a = a.rows[ra].unwrap_or_else(|| panic!("SrcA row {ra} holds nothing"));
        assert_eq!(face_a.row, 0, "SrcA rows {ra}.. start a face");
        for k in 0..16 {
            assert_eq!(
                a.rows[ra + k],
                Some(SrcRow {
                    row: k as u32,
                    ..face_a
                }),
                "SrcA rows {ra}.. hold one whole face"
            );
        }
        let face_b = b.rows[rb].unwrap_or_else(|| panic!("SrcB row {rb} holds nothing"));
        assert_eq!(face_b.row % 8, 0, "SrcB rows {rb}.. start a half");
        for k in 0..8 {
            assert_eq!(
                b.rows[rb + k],
                Some(SrcRow {
                    row: face_b.row + k as u32,
                    ..face_b
                }),
                "SrcB rows {rb}.. hold one half face"
            );
        }
        let dst =
            (op(i, "DstRow") + self.thread_field("DEST_TARGET_REG_CFG_MATH_Offset") + self.rwc.dst)
                & (0x400 - 8);
        let phase = (self.rwc.fidelity + self.thread_field("FIDELITY_BASE_Phase")) & 3;
        self.blocks.entry(dst / 8).or_default().push(Term {
            a: face_b.base,
            a_face: face_b.face,
            a_half: face_b.row / 8,
            b: face_a.base,
            b_face: face_a.face,
            b_transposed: face_a.transposed,
            phase,
        });
        self.release(op(i, "FlipSrcA") == 1, op(i, "FlipSrcB") == 1);
        self.addr_mod(op(i, "AddrMod"));
        true
    }

    /// `RWCs.md`'s `ApplyAddrMod`, with Blackhole's three-bit index.
    fn addr_mod(&mut self, index: u32) {
        let ab = |f: &str| self.thread_field(&format!("ADDR_MOD_AB_SEC{index}_{f}"));
        let ds = |f: &str| self.thread_field(&format!("ADDR_MOD_DST_SEC{index}_{f}"));
        let bias = |f: &str| self.thread_field(&format!("ADDR_MOD_BIAS_SEC{index}_{f}"));
        let ab2 = |f: &str| self.thread_field(&format!("ADDR_MOD_AB2_SEC{index}_{f}"));
        assert_eq!(
            bias("BiasClear") | bias("BiasIncr") | ab2("SrcAIncr") | ab2("SrcBIncr"),
            0,
            "the bias and the increments' extra bit are not modelled"
        );
        let (a_incr, a_cr, a_clear) = (ab("SrcAIncr"), ab("SrcACR"), ab("SrcAClear"));
        let (b_incr, b_cr, b_clear) = (ab("SrcBIncr"), ab("SrcBCR"), ab("SrcBClear"));
        let (d_incr, d_cr, d_c2cr, d_clear) = (
            ds("DestIncr"),
            ds("DestCR"),
            ds("DestCToCR"),
            ds("DestClear"),
        );
        let (f_incr, f_clear) = (ds("FidelityIncr"), ds("FidelityClear"));
        let r = &mut self.rwc;
        if a_clear == 1 {
            (r.src_a, r.src_a_cr) = (0, 0);
        } else if a_cr == 1 {
            r.src_a_cr = (r.src_a_cr + a_incr) & 63;
            r.src_a = r.src_a_cr;
        } else {
            r.src_a = (r.src_a + a_incr) & 63;
        }
        if b_clear == 1 {
            (r.src_b, r.src_b_cr) = (0, 0);
        } else if b_cr == 1 {
            r.src_b_cr = (r.src_b_cr + b_incr) & 63;
            r.src_b = r.src_b_cr;
        } else {
            r.src_b = (r.src_b + b_incr) & 63;
        }
        if d_clear == 1 {
            (r.dst, r.dst_cr) = (0, 0);
        } else if d_c2cr == 1 {
            r.dst = (r.dst + d_incr) & 1023;
            r.dst_cr = r.dst;
        } else if d_cr == 1 {
            r.dst_cr = (r.dst_cr + d_incr) & 1023;
            r.dst = r.dst_cr;
        } else {
            r.dst = (r.dst + d_incr) & 1023;
        }
        if f_clear == 1 {
            r.fidelity = 0;
        } else {
            r.fidelity = (r.fidelity + f_incr) & 3;
        }
    }
}

/// The terms each output's `Dst` blocks take, from the streams the unpack and math roles push
/// (each after the frontend's expanders: [`crate::loops::frontend_stream`]).
pub fn interpret(unpack: &[Instruction], math: &[Instruction]) -> Order {
    let mut m = Machine::new();
    let (mut u, mut t) = (0, 0);
    while u < unpack.len() || t < math.len() {
        let mut moved = false;
        while u < unpack.len() && m.unpack(unpack[u]) {
            u += 1;
            moved = true;
        }
        while t < math.len() && m.math(math[t]) {
            t += 1;
            moved = true;
        }
        assert!(
            moved,
            "deadlock: unpack at {u} ({:?}), math at {t} ({:?})",
            unpack.get(u).map(|i| i.def().key()),
            math.get(t).map(|i| i.def().key())
        );
    }
    let blocks = std::mem::take(&mut m.blocks);
    if !blocks.is_empty() {
        m.outputs.push(blocks);
    }
    m.outputs
}

/// The order every kernel must produce: for each output, its pairs in order; for each pair, `k'`
/// then the phases; and into each live block `(fi, fj, half)`, `A`'s face `(fi, k')` half against
/// `B`'s face `(k', fj)` (the slot's `(fj, k')` when `b_transposed`). Each block's `Dst` row is
/// `16 (2 fi + fj) + 8 half`.
pub fn canonical(
    outputs: &[super::OutputTile],
    phases: u32,
    live: super::LiveRows,
    b_transposed: bool,
) -> Order {
    let units = crate::datapath::tile_base_units;
    outputs
        .iter()
        .map(|o| {
            let mut blocks: BTreeMap<u32, Vec<Term>> = BTreeMap::new();
            for &(a, b) in &o.pairs {
                for k in 0..2 {
                    for phase in 0..phases {
                        for fi in 0..live.face_rows() {
                            for fj in 0..2 {
                                for half in 0..live.halves() {
                                    let row = 16 * (2 * fi + fj) + 8 * half;
                                    blocks.entry(row / 8).or_default().push(Term {
                                        a: units(a),
                                        a_face: 2 * fi + k,
                                        a_half: half,
                                        b: units(b),
                                        b_face: if b_transposed { 2 * fj + k } else { 2 * k + fj },
                                        b_transposed,
                                        phase,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            blocks
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::loops::{frontend_stream, lower_with};
    use tt_isa::cfg::generated::thread;

    const FIDELITIES: [Fidelity; 4] = [
        Fidelity::Lo,
        Fidelity::HiFi2,
        Fidelity::HiFi3,
        Fidelity::HiFi4,
    ];
    const LIVE: [LiveRows; 3] = [LiveRows::All, LiveRows::Sixteen, LiveRows::Eight];
    const ROUTES: [SrcRoute; 2] = [SrcRoute::Tf32FromFp32, SrcRoute::Tf32WithBf16B];

    /// Output tiles over `pairs` pairs each: evenly spaced (the GPR-stepped loop), in runs (a
    /// packed `A`'s shape), or scattered (retargeted every pair).
    fn outputs(shape: &str, n: usize, pairs: usize) -> Vec<OutputTile> {
        use tt_isa::dm::TILE_SLOT;
        let a0 = 0x2_0000u64;
        let b0 = 0x4_0000u64;
        (0..n)
            .map(|o| OutputTile {
                pairs: (0..pairs)
                    .map(|q| {
                        let q2 = match shape {
                            "even" => q,
                            "runs" => q + q / 4,
                            _ => (q * 7 + o) % (pairs + 5),
                        } as u64;
                        (
                            a0 + (o as u64 * 16 + q2) * TILE_SLOT,
                            b0 + (o as u64 * 16 + q2) * TILE_SLOT,
                        )
                    })
                    .collect(),
                out: 0x6_0000 + o as u64 * TILE_SLOT,
            })
            .collect()
    }

    fn streams(items: &[Vec<Item>; 3]) -> (Vec<Instruction>, Vec<Instruction>) {
        let s = |items: &[Item]| {
            let l = lower_with(items, false);
            frontend_stream(&l.words, l.mop.as_ref()).expect("a stream the frontend runs")
        };
        (s(&items[0]), s(&items[1]))
    }

    fn items(
        outs: &[OutputTile],
        route: SrcRoute,
        fidelity: Fidelity,
        b_transposed: bool,
        live: LiveRows,
    ) -> [Vec<Item>; 3] {
        let (sems, _) = MatmulSemaphores::alone();
        matmul_items_continuing(
            outs,
            sems,
            route.src_formats(),
            fidelity,
            None,
            b_transposed,
            live,
        )
    }

    #[test]
    fn the_matmul_kernels_accumulate_in_the_canonical_order() {
        for shape in ["even", "runs", "scattered"] {
            for fidelity in FIDELITIES {
                for live in LIVE {
                    for b_transposed in [false, true] {
                        for route in ROUTES {
                            let outs = outputs(shape, 3, 9);
                            let (u, m) =
                                streams(&items(&outs, route, fidelity, b_transposed, live));
                            assert_eq!(
                                interpret(&u, &m),
                                canonical(&outs, fidelity.phases(), live, b_transposed),
                                "{shape}, {fidelity:?}, {live:?}, transposed {b_transposed}, {route:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The looped blocked kernel (`block_stream_kernel_looped`) a first chunk: its role programs
    /// with their block repeats expanded as the runner pushes them, then the frontend's.
    #[test]
    fn the_looped_block_kernel_accumulates_in_the_canonical_order() {
        for fidelity in FIDELITIES {
            for b_transposed in [false, true] {
                for route in ROUTES {
                    let shape = [2, 3, 2];
                    let layout = block_stream_layout(shape, route.b_slot()).unwrap();
                    let (plain, _) = layout.chunk(0, shape);
                    let (code, loops) = block_stream_kernel_looped(
                        &layout,
                        0,
                        shape,
                        route,
                        fidelity,
                        b_transposed,
                        LiveRows::All,
                    )
                    .expect("fits");
                    let stream = |t: usize| {
                        let c = crate::code::Code {
                            ins: code[t].clone(),
                            loops: loops[t].clone(),
                        };
                        frontend_stream(&c.expand(), None).expect("a stream the frontend runs")
                    };
                    assert_eq!(
                        interpret(&stream(0), &stream(1)),
                        canonical(
                            &plain.outputs,
                            fidelity.phases(),
                            LiveRows::All,
                            b_transposed
                        ),
                        "{fidelity:?}, transposed {b_transposed}, {route:?}"
                    );
                }
            }
        }
    }

    /// `k'` reversed in every pair: each `A` face `(fi, k)` and `B` face `(k, fj)` unpacked as
    /// `k`'s other, by rewriting the `Z` each face unpack names.
    fn k_reversed(unpack: &[Instruction], b_transposed: bool) -> Vec<Instruction> {
        use tt_isa::isa::generated::encode;
        unpack
            .iter()
            .map(|&i| {
                if i.def().key() != "SETADCZW" {
                    return i;
                }
                let z = op(i, "Z0Val");
                let (u0, u1) = (op(i, "U0"), op(i, "U1"));
                // `A`'s (fi, k) is 2 fi + k; `B`'s (k, fj) 2 k + fj, or 2 fj + k transposed.
                let z = if u1 == 1 || b_transposed {
                    z ^ 1
                } else {
                    z ^ 2
                };
                encode::Setadczw::ZERO
                    .u0(u0)
                    .u1(u1)
                    .z0(1)
                    .z0_val(z)
                    .encode()
                    .unwrap()
            })
            .collect()
    }

    /// Phases 0 and 1 swapped in the first block a stream's first `MVMUL` writes: the phase base
    /// moved for its first two `MVMUL`s on that block (`MVMUL.md`: the phase is the RWC's plus
    /// `FIDELITY_BASE_Phase`).
    fn phases_swapped(math: &[Instruction]) -> Vec<Instruction> {
        let base = |v: u16| crate::datapath::thread_entry(thread::FIDELITY_BASE_Phase, v);
        let mut out = Vec::new();
        let mut first_block = None;
        let mut seen = 0;
        for &i in math {
            if i.def().mnemonic() == "MVMUL" && seen < 2 {
                let row = op(i, "DstRow");
                if *first_block.get_or_insert(row) == row {
                    // RWC phase 0 read as 1, then 1 read as 0.
                    out.push(base(if seen == 0 { 1 } else { 3 }));
                    out.push(i);
                    out.push(base(0));
                    seen += 1;
                    continue;
                }
            }
            out.push(i);
        }
        assert_eq!(seen, 2, "two MVMULs on the first block");
        out
    }

    #[test]
    fn the_order_check_sees_k_reversed_and_phases_swapped() {
        for b_transposed in [false, true] {
            let outs = outputs("even", 2, 4);
            let want = canonical(&outs, 4, LiveRows::All, b_transposed);
            let (u, m) = streams(&items(
                &outs,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                b_transposed,
                LiveRows::All,
            ));
            assert_eq!(interpret(&u, &m), want);
            // The k'-reversed program computes every block's terms, in another order.
            let reversed = interpret(&k_reversed(&u, b_transposed), &m);
            assert_ne!(reversed, want, "k' reversed");
            for (r, w) in reversed.iter().zip(&want) {
                for (block, terms) in w {
                    let mut a = r[block].clone();
                    let mut b = terms.clone();
                    a.sort();
                    b.sort();
                    assert_eq!(a, b, "the same terms, reordered");
                }
            }
            let swapped = interpret(&u, &phases_swapped(&m));
            assert_ne!(swapped, want, "phases swapped");
        }
    }
}
