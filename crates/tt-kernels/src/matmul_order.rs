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
    /// [`interpret_roles`]: the semaphores, the pack role, and `Dst` as rows the packer reads
    /// (`blocks` keyed by absolute 8-row block, cleared block by block).
    roles: bool,
    sems: [u8; 8],
    pack: Pack,
}

/// The pack role, as [`interpret_roles`] models it: its GPRs and `ThreadConfig`, the input ADC's
/// `Y`, the tile it is reading and the tiles it has read.
#[derive(Default)]
struct Pack {
    gpr: Vec<u32>,
    thread: BTreeMap<u16, u16>,
    y: u32,
    tile: BTreeMap<u32, Vec<Term>>,
    out: Order,
}

impl Pack {
    fn thread_field(&self, name: &str) -> u32 {
        let f = ALL_THREAD_CONFIG_FIELDS
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("no thread field {name}"))
            .1;
        u32::from(f.extract(self.thread.get(&f.addr32()).copied().unwrap_or(0)))
    }
}

/// The term a reload's prior partial puts in each block of its tile ([`interpret_roles`]): `a` is
/// the prior's tile (`tile_base_units`), `a_face` this marker.
pub const PRIOR: u32 = u32::MAX;

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
            roles: false,
            sems: [0; 8],
            pack: Pack {
                gpr: vec![0; 64],
                ..Pack::default()
            },
        }
    }

    /// A semaphore instruction of any role, under [`interpret_roles`]; `None` for another.
    /// `Some(false)` where a `SEMWAIT` must wait.
    fn semaphore(&mut self, i: Instruction) -> Option<bool> {
        let selected = |mask: u32| (0..8).filter(move |s| mask >> s & 1 == 1);
        match i.def().mnemonic() {
            "SEMWAIT" => {
                let (m, cond) = (op(i, "SemaphoreMask"), op(i, "ConditionMask"));
                let wait = selected(m).any(|s| {
                    cond & 1 == 1 && self.sems[s] == 0
                        || cond & 2 == 2 && self.sems[s] >= tt_isa::sync::MAX_VALUE
                });
                Some(!wait)
            }
            "SEMGET" => {
                for s in selected(op(i, "SemaphoreMask")) {
                    self.sems[s] = self.sems[s].saturating_sub(1);
                }
                Some(true)
            }
            "SEMPOST" => {
                for s in selected(op(i, "SemaphoreMask")) {
                    self.sems[s] = (self.sems[s] + 1).min(tt_isa::sync::MAX_VALUE);
                }
                Some(true)
            }
            _ => None,
        }
    }

    /// One pack-role instruction under [`interpret_roles`]; `false` if it must wait.
    fn pack(&mut self, i: Instruction) -> bool {
        if let Some(go) = self.semaphore(i) {
            return go;
        }
        match i.def().key() {
            "SETDMAREG_Immediate" => {
                let (half, v) = (op(i, "ResultHalfReg"), op(i, "NewValue"));
                let (r, shift) = ((half / 2) as usize, 16 * (half % 2));
                self.pack.gpr[r] = self.pack.gpr[r] & !(0xFFFF << shift) | v << shift;
            }
            "ADDDMAREG" => {
                let (l, r) = (op(i, "LeftReg") as usize, op(i, "RightReg") as usize);
                self.pack.gpr[op(i, "ResultReg") as usize] =
                    self.pack.gpr[l].wrapping_add(self.pack.gpr[r]);
            }
            "WRCFG" => {
                assert_eq!(op(i, "Is128Bit"), 0);
                self.config.insert(
                    op(i, "CfgIndex") as u16,
                    self.pack.gpr[op(i, "InputReg") as usize],
                );
            }
            "SETADCXY" => {
                if op(i, "PK") == 1 && op(i, "Y0") == 1 {
                    self.pack.y = op(i, "Y0Val");
                }
            }
            "PACR_BH" => {
                let offset =
                    self.cfg(tt_isa::cfg::generated::global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset);
                let mask = op(i, "ReadIntfSel");
                for k in 0..4 {
                    if mask >> k & 1 == 1 {
                        let row = offset + self.pack.y + k;
                        if let Some(terms) = self.blocks.get(&(row / 8)) {
                            self.pack.tile.insert((row - offset) / 8, terms.clone());
                        }
                    }
                }
                let incr = self
                    .pack
                    .thread_field(&format!("ADDR_MOD_PACK_SEC{}_YsrcIncr", op(i, "AddrMod")));
                self.pack.y += incr;
                if op(i, "Last") == 1 {
                    let tile = std::mem::take(&mut self.pack.tile);
                    self.pack.out.push(tile);
                }
            }
            other => match i.def().mnemonic() {
                "SETC16" => {
                    self.pack
                        .thread
                        .insert(op(i, "CfgIndex") as u16, op(i, "NewValue") as u16);
                }
                "SETADCXX" | "STALLWAIT" | "NOP" | "SFPNOP" => {}
                _ => panic!("the pack role's {other} is not modelled"),
            },
        }
        true
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
        if self.roles {
            if let Some(go) = self.semaphore(i) {
                return go;
            }
        }
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
                // Channel 1 and `W` only ever back to zero (`clear_unpacker0_adcs`).
                assert_eq!(
                    (op(i, "Z1") * op(i, "Z1Val"))
                        | (op(i, "W0") * op(i, "W0Val"))
                        | (op(i, "W1") * op(i, "W1Val")),
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
                // The counters' positions, which the ranges here stand for (`SETADCXX`).
                "SETADCXY"
                    if op(i, "X0Val") | op(i, "Y0Val") | op(i, "X1Val") | op(i, "Y1Val") == 0 => {}
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
            if self.roles {
                // The prior's rows: `Dst` row `OutAddr / 16 - 4`, a row each sixteen datums.
                let row = self.cfg(thcon::THCON_SEC0_REG5_Dest_cntx0_address) / 16 - 4;
                let adc = self.adc[0];
                let n = adc.x1 + 1 - adc.x0;
                assert!(
                    row % 8 == 0 && n % 128 == 0,
                    "whole blocks: {n} datums at row {row}"
                );
                let base = self.cfg(thcon::THCON_SEC0_REG3_Base_address);
                for b in row / 8..row / 8 + n / 128 {
                    self.blocks.insert(
                        b,
                        vec![Term {
                            a: base,
                            a_face: PRIOR,
                            a_half: 0,
                            b: 0,
                            b_face: 0,
                            b_transposed: false,
                            phase: 0,
                        }],
                    );
                }
            }
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
        if self.roles {
            if let Some(go) = self.semaphore(i) {
                return go;
            }
        }
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
            "ZEROACC" => match (op(i, "Mode"), op(i, "UseDst32b")) {
                (3, _) => {
                    let blocks = std::mem::take(&mut self.blocks);
                    if !blocks.is_empty() && !self.roles {
                        self.outputs.push(blocks);
                    }
                }
                // Sixteen 16-bit rows: eight FP32 rows, one block.
                (1, 0) if self.roles => {
                    let block = op(i, "Imm10");
                    assert!(block < 64, "a sixteen-row block of Dst");
                    self.blocks.remove(&block);
                }
                (mode, wide) => panic!("ZEROACC mode {mode} (UseDst32b {wide}) is not modelled"),
            },
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

/// The terms each output tile holds when the pack role reads it, tile by tile in pack order, from
/// the three roles' streams (each after the frontend's expanders) and the semaphores' starting
/// values (`init`): the observable a `Dst` plan is held to. `Dst` is modelled as 8-row blocks of
/// terms, cleared block by block (`ZEROACC` modes 1 and 3) and written by `MVMUL`s and by a
/// reload's `UnpackToDst` (a [`PRIOR`] term), and the roles are scheduled adversarially: the pack
/// role steps only while the other two cannot, so a section the math role clears or overwrites
/// before the packer has read it shows in what the packer reads.
pub fn interpret_roles(
    unpack: &[Instruction],
    math: &[Instruction],
    pack: &[Instruction],
    init: &[crate::runtime::SemaphoreInit],
) -> Order {
    let mut m = Machine::new();
    m.roles = true;
    for &(s, v, _) in init {
        m.sems[s.index() as usize] = v;
    }
    let (mut u, mut t, mut p) = (0, 0, 0);
    while u < unpack.len() || t < math.len() || p < pack.len() {
        let mut moved = false;
        while t < math.len() && m.math(math[t]) {
            t += 1;
            moved = true;
        }
        while u < unpack.len() && m.unpack(unpack[u]) {
            u += 1;
            moved = true;
        }
        if !moved && p < pack.len() && m.pack(pack[p]) {
            p += 1;
            moved = true;
        }
        assert!(
            moved,
            "deadlock: unpack at {u} ({:?}), math at {t} ({:?}), pack at {p} ({:?})",
            unpack.get(u).map(|i| i.def().key()),
            math.get(t).map(|i| i.def().key()),
            pack.get(p).map(|i| i.def().key())
        );
    }
    m.pack.out
}

/// [`canonical`] for a chunk that continues from priors: each output's blocks (all eight) take
/// its prior first ([`PRIOR`], `priors[o]`'s tile), then its pairs.
pub fn canonical_continuing(
    outputs: &[super::OutputTile],
    priors: &[u64],
    phases: u32,
    b_transposed: bool,
) -> Order {
    let units = crate::datapath::tile_base_units;
    canonical(outputs, phases, super::LiveRows::All, b_transposed)
        .into_iter()
        .zip(priors)
        .map(|(mut blocks, &prior)| {
            for b in 0..8 {
                blocks.entry(b).or_default().insert(
                    0,
                    Term {
                        a: units(prior),
                        a_face: PRIOR,
                        a_half: 0,
                        b: 0,
                        b_face: 0,
                        b_transposed: false,
                        phase: 0,
                    },
                );
            }
            blocks
        })
        .collect()
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
    const ROUTES: [SrcRoute; 2] = [SrcRoute::Tf32FromFp32, SrcRoute::TF32_WITH_BF16_B];

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

    const ORDERS: [PairOrder; 6] = PairOrder::ALL;

    #[test]
    fn the_matmul_kernels_accumulate_in_the_canonical_order() {
        for order in ORDERS {
            ORDER_HERE.set(Some(order));
            kernels_accumulate_in_the_canonical_order(order);
        }
    }

    fn kernels_accumulate_in_the_canonical_order(order: PairOrder) {
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
                                "{order:?}, {shape}, {fidelity:?}, {live:?}, transposed {b_transposed}, {route:?}"
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
        for order in ORDERS {
            ORDER_HERE.set(Some(order));
            looped_block_kernel_accumulates_in_the_canonical_order(order);
        }
    }

    fn looped_block_kernel_accumulates_in_the_canonical_order(order: PairOrder) {
        for fidelity in FIDELITIES {
            for b_transposed in [false, true] {
                for route in ROUTES {
                    let shape = [2, 3, 2];
                    let layout = block_stream_layout(shape, route.slots()).unwrap();
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
                        "{order:?}, {fidelity:?}, transposed {b_transposed}, {route:?}"
                    );
                }
            }
        }
    }

    /// Each order's math role is other words than the faces order's (but at one phase or one
    /// half, where there is nothing to interleave), and the looped block kernel's math role
    /// still has its fused `.ttinsn` form (`code::Code::fused`; the runner pushes it otherwise,
    /// silently).
    #[test]
    fn each_order_is_its_own_words_and_keeps_the_fused_form() {
        for fidelity in FIDELITIES {
            for live in LIVE {
                let outs = outputs("even", 2, 5);
                let math = |order| {
                    ORDER_HERE.set(Some(order));
                    streams(&items(&outs, SrcRoute::Tf32FromFp32, fidelity, false, live)).1
                };
                let faces = math(PairOrder::Faces);
                for order in ORDERS {
                    let same = math(order) == faces;
                    let alike = order == PairOrder::Faces
                        || order == PairOrder::PhasesOuter
                            && (fidelity == Fidelity::Lo || live == LiveRows::Eight);
                    assert_eq!(same, alike, "{order:?}, {fidelity:?}, {live:?}");
                }
            }
            for order in ORDERS {
                ORDER_HERE.set(Some(order));
                for route in ROUTES {
                    let shape = [2, 3, 2];
                    let layout = block_stream_layout(shape, route.slots()).unwrap();
                    for c in [0, 1] {
                        let (code, loops) = block_stream_kernel_looped(
                            &layout,
                            c,
                            shape,
                            route,
                            fidelity,
                            false,
                            LiveRows::All,
                        )
                        .expect("fits");
                        let math = crate::code::Code {
                            ins: code[1].clone(),
                            loops: loops[1].clone(),
                        };
                        assert!(
                            math.fused().is_ok(),
                            "{order:?}, {fidelity:?}, {route:?}, chunk {c}: {:?}",
                            math.fused().err()
                        );
                    }
                }
            }
        }
    }

    /// What `silicon_probe_pair_cost`'s products' math roles push and whether they take the fused
    /// form (`session::fused_form`: a fused form, `tt_isa::ttinsn::validate`, and at least
    /// `FUSE_MIN_WORDS` pushed), for each order, `B` format and fidelity: printed (`--nocapture`).
    #[test]
    #[ignore = "a report: cargo test -p tt-kernels --lib fused_forms_of_the_pair_cost_products -- --ignored --nocapture"]
    fn fused_forms_of_the_pair_cost_products() {
        use crate::tensor::{block_stream_choice, BlockStream};
        for tiles in [[32usize, 64, 256], [32, 256, 64]] {
            for route in ROUTES {
                let shape = block_stream_choice(
                    BlockStream::Auto,
                    tiles,
                    120,
                    f64::INFINITY,
                    route,
                    Fidelity::HiFi4,
                )
                .expect("blocked");
                for order in ORDERS {
                    ORDER_HERE.set(Some(order));
                    for fidelity in [Fidelity::Lo, Fidelity::HiFi4] {
                        let layout = block_stream_layout(shape.tiles, route.slots()).unwrap();
                        let mut line = format!(
                            "{tiles:?} {route:?} {:?} {order:?} {fidelity:?}:",
                            shape.tiles
                        );
                        for c in [0, 1] {
                            let (code, loops) = block_stream_kernel_looped(
                                &layout,
                                c,
                                shape.tiles,
                                route,
                                fidelity,
                                false,
                                LiveRows::All,
                            )
                            .expect("fits");
                            let math = crate::code::Code {
                                ins: code[1].clone(),
                                loops: loops[1].clone(),
                            };
                            let fused = math.fused().ok();
                            let pushed = fused
                                .as_ref()
                                .and_then(|f| tt_isa::ttinsn::validate(f).ok());
                            line += &format!(
                                " chunk {c}: {} words pushed, {} stored, fused {}",
                                math.expand().len(),
                                math.ins.len(),
                                match pushed {
                                    Some(p) if p >= crate::session::FUSE_MIN_WORDS as u64 =>
                                        "yes".to_string(),
                                    Some(p) => format!("no (pushes {p} < FUSE_MIN_WORDS)"),
                                    None => "no (no fused form)".to_string(),
                                }
                            );
                        }
                        println!("{line}");
                    }
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
        for order in ORDERS {
            ORDER_HERE.set(Some(order));
            sees_k_reversed_and_phases_swapped(order);
        }
    }

    /// Whether `run` gives other terms than `want`, or the checker refuses the streams.
    fn caught(want: &Order, run: impl FnOnce() -> Order + std::panic::UnwindSafe) -> bool {
        std::panic::catch_unwind(run).map_or(true, |got| got != *want)
    }

    /// Address modifier 1, the next `SrcB` half in every whole order, moving `SrcB` nowhere.
    fn half_step_lost(math: &[Instruction]) -> Vec<Instruction> {
        let ab1 = thread::ADDR_MOD_AB_SEC1_SrcBIncr;
        math.iter()
            .map(|&i| {
                if i.def().mnemonic() == "SETC16" && op(i, "CfgIndex") == u32::from(ab1.addr32()) {
                    let v = ab1.insert(op(i, "NewValue") as u16, 0);
                    tt_isa::backend::set_thread_entry(ab1.addr32(), v).unwrap()
                } else {
                    i
                }
            })
            .collect()
    }

    fn sees_k_reversed_and_phases_swapped(order: PairOrder) {
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
            if order
                .resolve(SrcRoute::Tf32FromFp32.src_formats())
                .whole()
                .is_none()
            {
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
            } else {
                // `k'` is in the RWCs here: a lost `SrcB` step reads one half twice.
                let (u, m) = (u.clone(), half_step_lost(&m));
                assert!(
                    caught(&want, move || interpret(&u, &m)),
                    "{order:?}: half step lost"
                );
            }
            let swapped = interpret(&u, &phases_swapped(&m));
            assert_ne!(swapped, want, "{order:?}: phases swapped");
        }
    }

    /// A blocked chunk's three role streams as the runner and the frontend push them, and the
    /// semaphores its run starts: the subblocked kernel's (`plan` given) or the looped one's.
    fn block_roles(
        layout: &BlockStreamLayout,
        c: usize,
        tiles: [usize; 3],
        route: SrcRoute,
        fidelity: Fidelity,
        b_transposed: bool,
        plan: Option<&crate::tile_op::dst::RowPlan>,
    ) -> ([Vec<Instruction>; 3], Vec<crate::runtime::SemaphoreInit>) {
        let (code, loops) = match plan {
            Some(plan) => block_stream_kernel_subblocked(
                layout,
                c,
                tiles,
                route,
                fidelity,
                b_transposed,
                LiveRows::All,
                plan,
            )
            .expect("fits"),
            None => block_stream_kernel_looped(
                layout,
                c,
                tiles,
                route,
                fidelity,
                b_transposed,
                LiveRows::All,
            )
            .expect("fits"),
        };
        let stream = |t: usize| {
            let c = crate::code::Code {
                ins: code[t].clone(),
                loops: loops[t].clone(),
            };
            frontend_stream(&c.expand(), None).expect("a stream the frontend runs")
        };
        let init = match plan {
            Some(p) => subblock_init(&layout.init, layout.sems, p.plan),
            None => layout.init.clone(),
        };
        ([stream(0), stream(1), stream(2)], init)
    }

    /// What a chunk's packer must read: each output's pairs, after its prior where it continues.
    fn want(
        layout: &BlockStreamLayout,
        c: usize,
        tiles: [usize; 3],
        phases: u32,
        bt: bool,
    ) -> Order {
        let (plain, priors) = layout.chunk(c, tiles);
        match priors {
            None => canonical(&plain.outputs, phases, LiveRows::All, bt),
            Some(priors) => canonical_continuing(&plain.outputs, &priors, phases, bt),
        }
    }

    /// Block shapes the subblock gates run: odd and even rows, columns that four divides, that no
    /// width divides (3, 5, 7, 9: the FLUX.2 VAE's 96 channels are 3 tiles), one column, rows a
    /// period of two.
    const SUBBLOCK_SHAPES: [[usize; 3]; 9] = [
        [2, 2, 4],
        [3, 2, 3],
        [1, 2, 5],
        [2, 1, 9],
        [3, 2, 1],
        [2, 3, 7],
        [1, 1, 8],
        [3, 1, 6],
        [2, 2, 2],
    ];

    /// The subblocked kernel (`block_stream_kernel_subblocked`): what its packer reads, under an
    /// adversarial schedule of the three roles (`interpret_roles`), is each output's canonical
    /// order -- a first chunk's pairs, a continuing chunk's prior then pairs -- and the looped
    /// kernel's, for every shape, fidelity, route and `B` transposed or not.
    #[test]
    fn the_subblocked_kernel_packs_the_canonical_order() {
        ORDER_HERE.set(Some(PairOrder::Auto));
        subblocked_kernel_packs_the_canonical_order();
    }

    fn subblocked_kernel_packs_the_canonical_order() {
        for tiles in SUBBLOCK_SHAPES {
            for fidelity in FIDELITIES {
                for b_transposed in [false, true] {
                    for route in ROUTES {
                        let layout = block_stream_layout(tiles, route.slots()).unwrap();
                        let plan = crate::tile_op::dst::RowPlan::choose(
                            tiles[2],
                            crate::tile_op::dst::DstPlan::HALF,
                        );
                        for c in [0, 1, 2] {
                            let what = format!(
                                "{tiles:?} {fidelity:?} {route:?} bt {b_transposed} chunk {c}"
                            );
                            let want = want(&layout, c, tiles, fidelity.phases(), b_transposed);
                            let ([u, m, p], init) = block_roles(
                                &layout,
                                c,
                                tiles,
                                route,
                                fidelity,
                                b_transposed,
                                Some(&plan),
                            );
                            assert_eq!(
                                interpret_roles(&u, &m, &p, &init),
                                want,
                                "subblocked, {what}"
                            );
                            let ([u, m, p], init) =
                                block_roles(&layout, c, tiles, route, fidelity, b_transposed, None);
                            assert_eq!(interpret_roles(&u, &m, &p, &init), want, "looped, {what}");
                        }
                    }
                }
            }
        }
    }

    /// The negative controls the subblock gates watch fail: each subblock clearing all of `Dst`
    /// (the half the packer drains too), the second section overlapping the first, the looped
    /// kernel run with `free` at two (its whole-`Dst` clear under a packer still reading), and the
    /// subblocked kernel's `A` tile released after every `B` tile.
    #[test]
    fn the_subblock_negative_controls_fail() {
        ORDER_HERE.set(Some(PairOrder::Auto));
        subblock_negative_controls_fail();
    }

    fn subblock_negative_controls_fail() {
        let route = SrcRoute::Tf32FromFp32;
        let fidelity = Fidelity::HiFi4;
        for tiles in [[2, 2, 4], [3, 2, 3], [1, 2, 5], [2, 1, 9]] {
            let layout = block_stream_layout(tiles, route.slots()).unwrap();
            let plan =
                crate::tile_op::dst::RowPlan::choose(tiles[2], crate::tile_op::dst::DstPlan::HALF);
            for c in [0, 1] {
                let want = want(&layout, c, tiles, fidelity.phases(), false);
                for mutation in [DstMutation::ClearBoth, DstMutation::Overlap] {
                    MUTATION_HERE.set(Some(mutation));
                    let ([u, m, p], init) =
                        block_roles(&layout, c, tiles, route, fidelity, false, Some(&plan));
                    MUTATION_HERE.set(None);
                    // What the packer read, not a refusal: the mutation runs, and reads wrong.
                    assert_ne!(
                        interpret_roles(&u, &m, &p, &init),
                        want,
                        "{mutation:?} {tiles:?} chunk {c} not caught"
                    );
                }
                let ([u, m, p], init) =
                    block_roles(&layout, c, tiles, route, fidelity, false, None);
                let two = subblock_init(&init, layout.sems, crate::tile_op::dst::DstPlan::HALF);
                assert_ne!(
                    interpret_roles(&u, &m, &p, &two),
                    want,
                    "the looped kernel at two sections, {tiles:?} chunk {c}, not caught"
                );
            }
        }
    }

    /// The subblocked kernel's math role has its fused `.ttinsn` form for every shape, and every
    /// role's program fits its slot.
    #[test]
    fn the_subblocked_kernel_keeps_the_fused_form() {
        ORDER_HERE.set(Some(PairOrder::Auto));
        for tiles in
            SUBBLOCK_SHAPES
                .into_iter()
                .chain([[8, 2, 9], [6, 4, 12], [4, 2, 18], [8, 4, 6]])
        {
            for route in ROUTES {
                let Ok(layout) = block_stream_layout(tiles, route.slots()) else {
                    continue;
                };
                let plan = crate::tile_op::dst::RowPlan::choose(
                    tiles[2],
                    crate::tile_op::dst::DstPlan::HALF,
                );
                for c in [0, 1] {
                    let (code, loops) = block_stream_kernel_subblocked(
                        &layout,
                        c,
                        tiles,
                        route,
                        Fidelity::HiFi4,
                        false,
                        LiveRows::All,
                        &plan,
                    )
                    .unwrap_or_else(|| {
                        panic!("{tiles:?} {route:?} chunk {c}: no subblocked programs")
                    });
                    let math = crate::code::Code {
                        ins: code[1].clone(),
                        loops: loops[1].clone(),
                    };
                    assert!(
                        math.fused().is_ok(),
                        "{tiles:?} {route:?} chunk {c}: {:?}",
                        math.fused().err()
                    );
                    let sizes: Vec<usize> = code.iter().map(|p| p.len()).collect();
                    println!(
                        "{tiles:?} {route:?} chunk {c}: words stored {sizes:?}, pushed {}",
                        math.expand().len()
                    );
                }
            }
        }
    }
}
