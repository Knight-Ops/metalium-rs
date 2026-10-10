//! Hazards as data: which unit runs each instruction, which waits hold it, and a
//! checker that replays a thread's instruction stream against those facts
//! (`implementation-checklist.md`, "Hazards as data"; `hardware-coverage-closeout.md`, P9).
//!
//! Every wait in the repository's Tensix programs is placed by hand through
//! `backend::Before` and `sync::post_after`. This module states the *reasons*
//! those waits exist, as a table, and checks a program against them. It does not
//! insert anything: a planner is only worth writing if the checker finds misses,
//! and over today's builders it finds none (`step121_wait_planner`).
//!
//! # The model
//!
//! A `STALLWAIT` does not wait *for* anything by itself: its condition mask says
//! what must be quiet (`cond`), its block mask which *following* instructions are
//! held until it is (`block`; `STALLWAIT.md`). An instruction is held when the
//! wait's block mask and the instruction's own row of the specification's
//! block table share a bit ([`blocked_by`]; the table is cross-checked against
//! `STALLWAIT.md` by `step121_wait_planner`). The checker therefore tracks, for
//! each backend [`Unit`] this thread has used, the block bits of the waits seen
//! since the unit last had an instruction issued to it. A consumer is covered when
//! one of those bits holds it back.
//!
//! The pairs checked ([`Kind`]) are the ones the documentation names or a silicon
//! failure taught, each with its source:
//!
//! * [`Kind::PublishBeforeDrain`] -- a `SEMPOST` announces finished work, so every
//!   unit the thread drove must have drained under a wait that blocks the Sync Unit
//!   (`SEMPOST.md`, "Instruction scheduling"; `sync::post_after`).
//! * [`Kind::TakeNotHeld`] -- the wait of a semaphore *take* (`SEMWAIT` then
//!   `SEMGET`) must hold back the first instruction of a unit that reads what was
//!   handed over (`backend::Before`: the block bit names the consumer, the first
//!   silicon run of the elementwise gate read `Dst` before the unpacker wrote it).
//! * [`Kind::DstUnpackerToSfpu`], [`Kind::DstMatrixToSfpu`] -- `SFPLOAD` reads
//!   `Dst` that an `UNPACR` or a Matrix Unit instruction wrote earlier in the same
//!   thread: the hardware stalls only Matrix Unit and `PACR` readers (`Dst.md`,
//!   "Instruction scheduling"), so the load needs a wait (B8) or, after a Matrix
//!   Unit write, three unrelated instructions (`SFPLOAD.md`).
//! * [`Kind::ConfigNotLanded`] -- an `UNPACR` or `PACR` reads `Config` when it
//!   issues, so the instruction right after a `Config` write must not be one
//!   (`WRCFG.md`, `CFGSHIFTMASK.md`: one intervening instruction, a `NOP`
//!   suffices), or a wait for C12 blocking it must intervene
//!   (`ConfigurationUnit.md`; `datapath::unpack_tile_to_dst` does the latter, the
//!   matmul roles the former).
//! * [`Kind::ConfigWhileBusy`] -- a `Config` write while an unpacker or the packer
//!   still has an instruction in flight would change what it reads: drain first
//!   with B7 (`Before::CONFIG`).
//! * [`Kind::MadMissedStall`] -- on the stream the backend receives, an `SFPMAD`
//!   immediately followed by one of the cases its automatic stalling misses
//!   (`SFPMAD.md`; [`Instruction::stalls_automatically_after_mad`]). Checked on the
//!   *expanded* stream so a replay or runner-loop seam is covered.
//! * [`Kind::ConfigShiftMaskAdjacent`] -- the instruction after `CFGSHIFTMASK`
//!   must not be a consumer outside the Configuration Unit (`CFGSHIFTMASK.md`).
//! * [`Kind::EndsBusy`] -- a closed program leaves nothing in flight.
//!
//! Not modelled (and so neither flagged nor excused): `Dst` block timing of a few
//! cycles that hardware stalls for itself, the `SrcA`/`SrcB` bank hand-over the
//! Matrix Unit waits for by itself, address overlap between producers and
//! consumers, and anything between threads (semaphores are the interface there,
//! and [`Kind::TakeNotHeld`] / [`Kind::PublishBeforeDrain`] check both ends).
//!
//! Input is the stream *as the backend receives it*: `MOP` and `REPLAY` already
//! expanded (`tt_kernels::loops::frontend_stream`) and any block repeats
//! (`tt_kernels::code::Code::expand`) unrolled.

use crate::backend::{block, cond};
use crate::isa::Instruction;

/// The backend units whose in-flight work a wait can drain.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Unit {
    Unpacker0,
    Unpacker1,
    Packer,
    Matrix,
    Sfpu,
    Config,
    Scalar,
    Mover,
}

impl Unit {
    const COUNT: usize = 8;
    const ALL: [Unit; Self::COUNT] = [
        Unit::Unpacker0,
        Unit::Unpacker1,
        Unit::Packer,
        Unit::Matrix,
        Unit::Sfpu,
        Unit::Config,
        Unit::Scalar,
        Unit::Mover,
    ];

    const fn index(self) -> usize {
        self as usize
    }

    /// The `STALLWAIT` condition bit that reports this unit busy.
    pub const fn condition(self) -> u32 {
        match self {
            Unit::Unpacker0 => cond::UNPACKER0_BUSY,
            Unit::Unpacker1 => cond::UNPACKER1_BUSY,
            Unit::Packer => cond::PACKER_BUSY,
            Unit::Matrix => cond::MATRIX_BUSY,
            Unit::Sfpu => cond::SFPU_BUSY,
            Unit::Config => cond::CONFIG_BUSY,
            Unit::Scalar => cond::SCALAR_OUTSTANDING,
            Unit::Mover => cond::MOVER_OUTSTANDING,
        }
    }

    /// Does a wait on this unit's condition speak only of this thread's work?
    /// (`Config` and `Mover` conditions cover every thread, so a single stream
    /// cannot call a wait on them redundant.)
    const fn thread_local(self) -> bool {
        !matches!(self, Unit::Config | Unit::Mover)
    }
}

/// What kind of instruction this is, for hazard purposes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Class {
    Unpack,
    Pack,
    Matrix,
    Sfpu,
    Config,
    Scalar,
    Mover,
    /// `SEMPOST`, `SEMGET`, `SEMINIT`, `ATGETM`, `ATRELM`.
    Sync,
    /// `STALLWAIT`, `SEMWAIT`, `STREAMWAIT`: blocked by everything.
    WaitGate,
    /// B0 only: ADC and `Dst` valid bookkeeping.
    Misc,
    /// Handled by the frontend or held by no bit: `NOP`, `MOP`, `MOP_CFG`, `REPLAY`.
    Free,
}

/// The class of `i`, by the mnemonic the specification's block table names.
pub fn class_of(i: Instruction) -> Class {
    let m = i.def().mnemonic();
    if m.starts_with("SFP") {
        return Class::Sfpu;
    }
    if m.starts_with("UNPACR") {
        return Class::Unpack;
    }
    if m.starts_with("RMWCIB") {
        return Class::Config;
    }
    if m.starts_with("STOREIND") {
        return Class::Scalar;
    }
    match m {
        "PACR" | "PACR_SETREG" => Class::Pack,
        "MVMUL" | "DOTPV" | "ELWADD" | "ELWSUB" | "ELWMUL" | "GMPOOL" | "GAPOOL" | "MOVA2D"
        | "MOVB2A" | "MOVB2D" | "MOVD2A" | "MOVD2B" | "MOVDBGA2D" | "ZEROACC" | "ZEROSRC"
        | "SHIFTXA" | "SHIFTXB" | "TRNSPSRCB" | "SETRWC" | "INCRWC" | "CLEARDVALID"
        | "CLREXPHIST" | "GATESRCRST" => Class::Matrix,
        "CFGSHIFTMASK" | "SETC16" | "WRCFG" | "STREAMWRCFG" | "RDCFG" => Class::Config,
        "ADDDMAREG" | "ATCAS" | "ATINCGET" | "ATINCGETPTR" | "ATSWAP" | "BITWOPDMAREG"
        | "CMPDMAREG" | "DMANOP" | "FLUSHDMA" | "LOADIND" | "LOADREG" | "MULDMAREG"
        | "REG2FLOP" | "SETDMAREG" | "SHIFTDMAREG" | "STOREREG" | "SUBDMAREG" => Class::Scalar,
        "XMOV" => Class::Mover,
        "ATGETM" | "ATRELM" | "SEMGET" | "SEMINIT" | "SEMPOST" => Class::Sync,
        "STALLWAIT" | "SEMWAIT" | "STREAMWAIT" => Class::WaitGate,
        "ADDRCRXY" | "ADDRCRZW" | "INCADCXY" | "INCADCZW" | "SETADC" | "SETADCXX" | "SETADCXY"
        | "SETADCZW" | "SETDVALID" | "RSTDMA" | "REG2FLOP_ADC" => Class::Misc,
        _ => Class::Free,
    }
}

/// The `STALLWAIT` block bits that hold `i` back: its row of the specification's
/// block table (`STALLWAIT.md`, "The exact set of instructions blocked from
/// starting by each bit").
pub fn blocked_by(i: Instruction) -> u32 {
    use block::*;
    match class_of(i) {
        Class::Sfpu => SFPU,
        Class::Unpack => ANY_DMA | UNPACKER,
        Class::Pack => ANY_DMA | PACKER,
        Class::Matrix => MATRIX,
        Class::Config => CONFIG,
        Class::Scalar => ANY_DMA | SCALAR,
        Class::Mover => ANY_DMA | MOVER,
        Class::Sync => SYNC,
        Class::WaitGate => {
            ANY_DMA | SYNC | PACKER | UNPACKER | MOVER | SCALAR | MATRIX | CONFIG | SFPU
        }
        Class::Misc => ANY_DMA,
        Class::Free => 0,
    }
}

/// The unit(s) an instruction gives work to, as a bit set over [`Unit`].
fn issues_to(i: Instruction, class: Class) -> u8 {
    let bit = |u: Unit| 1u8 << u.index();
    match class {
        Class::Unpack => match i.operand("WhichUnpacker") {
            Some(0) => bit(Unit::Unpacker0),
            Some(_) => bit(Unit::Unpacker1),
            None => bit(Unit::Unpacker0) | bit(Unit::Unpacker1),
        },
        Class::Pack => bit(Unit::Packer),
        // Only what writes `Dst` is work to wait for: `SETRWC` and the source
        // bank bookkeeping run in the same pipeline but leave nothing for a
        // reader or a poster to wait on.
        Class::Matrix if writes_dst(i.def().mnemonic()) => bit(Unit::Matrix),
        // `SFPNOP` is padding: a program that uses it as a separator in a thread
        // that never runs the SFPU has no SFPU work to drain.
        Class::Sfpu if i.def().mnemonic() != "SFPNOP" => bit(Unit::Sfpu),
        // `SETC16` is not in the Configuration Unit's pipeline (its own IPC group,
        // one cycle: `ConfigurationUnit.md`). No page documents how it is ordered
        // against a later `UNPACR` or `PACR`; the builders rely on in-order issue
        // and every gate on silicon has held, so it is not modelled as a write
        // that must land.
        Class::Config if i.def().mnemonic() != "SETC16" => bit(Unit::Config),
        // Only the Scalar Unit's *memory* requests are work another thread or a
        // core can observe (C0); `SETDMAREG` and the other register-to-register
        // arithmetic complete in order and publish nothing.
        Class::Scalar if touches_memory(i.def().mnemonic()) => bit(Unit::Scalar),
        Class::Mover => bit(Unit::Mover),
        _ => 0,
    }
}

/// Matrix Unit instructions that write `Dst`.
fn writes_dst(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        "MVMUL"
            | "DOTPV"
            | "ELWADD"
            | "ELWSUB"
            | "ELWMUL"
            | "GMPOOL"
            | "GAPOOL"
            | "MOVA2D"
            | "MOVB2D"
            | "MOVDBGA2D"
            | "ZEROACC"
    )
}

/// Scalar Unit instructions that issue memory requests (`STALLWAIT.md`, C0).
fn touches_memory(mnemonic: &str) -> bool {
    mnemonic.starts_with("STOREIND")
        || mnemonic.starts_with("LOADIND")
        || matches!(
            mnemonic,
            "STOREREG" | "LOADREG" | "ATCAS" | "ATINCGET" | "ATINCGETPTR" | "ATSWAP" | "FLUSHDMA"
        )
}

/// What the checker found.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Kind {
    // Misses: a documented hazard with no adequate wait.
    PublishBeforeDrain,
    TakeNotHeld,
    DstUnpackerToSfpu,
    DstMatrixToSfpu,
    ConfigNotLanded,
    ConfigWhileBusy,
    MadMissedStall,
    ConfigShiftMaskAdjacent,
    EndsBusy,
    /// A `STALLWAIT` with an empty block or condition mask: the hardware reads
    /// zero as "B6" and "C0..C3" (`STALLWAIT.md`, functional model), which is
    /// rarely what was meant.
    EmptyMask,
    // Lint, not a miss.
    /// A wait on this thread's own units whose condition already held for
    /// everything it blocks: nothing was issued to the unit since an earlier
    /// wait that blocked the same consumers.
    RedundantWait,
}

impl Kind {
    /// A hazard with no adequate wait (everything but the lint).
    pub const fn is_miss(self) -> bool {
        !matches!(self, Kind::RedundantWait)
    }
}

/// One finding: the position in the stream, what it is, the unit whose work was
/// unprotected (for a miss) and the instruction that needed the protection.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Finding {
    pub index: usize,
    pub kind: Kind,
    pub producer: Option<Unit>,
    pub consumer: &'static str,
}

/// `SEMWAIT`'s mask is its block mask only for the load-bearing bits.
fn masks(i: Instruction) -> (u32, u32) {
    (
        i.operand("BlockMask").unwrap_or(0),
        i.operand("ConditionMask").unwrap_or(0),
    )
}

fn missed_by_stalling(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
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

/// Check `stream` and report every finding through `found`. `closed`: the stream
/// is a whole role program, so it must leave nothing in flight.
pub fn check(
    stream: impl IntoIterator<Item = Instruction>,
    closed: bool,
    mut found: impl FnMut(Finding),
) {
    // Block bits of the waits since each unit last had work issued; everything
    // is protected at the start (a program begins with the unit quiet).
    let mut protected = [!0u32; Unit::COUNT];
    // Instructions issued since each unit last had work issued.
    let mut gap = [u32::MAX; Unit::COUNT];
    let mut last_semwait: Option<u32> = None;
    let mut armed_take: Option<u32> = None;
    let mut after_mad = false;
    let mut after_shiftmask = false;
    let mut n = 0usize;

    for (index, i) in stream.into_iter().enumerate() {
        n = index + 1;
        let class = class_of(i);
        let held = blocked_by(i);
        let name = i.def().mnemonic();
        let mut flag = |kind, producer: Option<Unit>| {
            found(Finding {
                index,
                kind,
                producer,
                consumer: name,
            })
        };

        // Rules whose consumer is this instruction.
        let covered = |p: Unit, protected: &[u32; Unit::COUNT]| protected[p.index()] & held != 0;
        match class {
            Class::Sync if name == "SEMPOST" => {
                for p in [
                    Unit::Unpacker0,
                    Unit::Unpacker1,
                    Unit::Packer,
                    Unit::Matrix,
                    Unit::Sfpu,
                    Unit::Scalar,
                    Unit::Mover,
                ] {
                    if !covered(p, &protected) {
                        flag(Kind::PublishBeforeDrain, Some(p));
                    }
                }
            }
            Class::Unpack | Class::Pack => {
                // The documented rule is separation by one instruction
                // (`WRCFG.md`, `CFGSHIFTMASK.md`); a wait for C12 that blocks
                // the consumer is the stronger form.
                if !covered(Unit::Config, &protected) && gap[Unit::Config.index()] < 1 {
                    flag(Kind::ConfigNotLanded, Some(Unit::Config));
                }
            }
            Class::Sfpu if matches!(name, "SFPLOAD" | "SFPLOADMACRO") => {
                for p in [Unit::Unpacker0, Unit::Unpacker1] {
                    if !covered(p, &protected) {
                        flag(Kind::DstUnpackerToSfpu, Some(p));
                    }
                }
                if !covered(Unit::Matrix, &protected) && gap[Unit::Matrix.index()] < 3 {
                    flag(Kind::DstMatrixToSfpu, Some(Unit::Matrix));
                }
            }
            Class::Config if name != "RDCFG" => {
                for p in [Unit::Unpacker0, Unit::Unpacker1, Unit::Packer] {
                    if !covered(p, &protected) {
                        flag(Kind::ConfigWhileBusy, Some(p));
                    }
                }
            }
            _ => {}
        }

        // A take's wait must hold the first data consumer after it.
        if matches!(
            class,
            Class::Unpack | Class::Pack | Class::Matrix | Class::Sfpu
        ) {
            if let Some(mask) = armed_take.take() {
                if mask & held == 0 {
                    flag(Kind::TakeNotHeld, None);
                }
            }
        }
        if after_mad && class == Class::Sfpu && !i.stalls_automatically_after_mad() {
            flag(Kind::MadMissedStall, Some(Unit::Sfpu));
        }
        if after_shiftmask
            && matches!(
                class,
                Class::Unpack
                    | Class::Pack
                    | Class::Matrix
                    | Class::Sfpu
                    | Class::Scalar
                    | Class::Mover
                    | Class::Misc
            )
        {
            flag(Kind::ConfigShiftMaskAdjacent, Some(Unit::Config));
        }

        // Effects of this instruction.
        for g in gap.iter_mut() {
            *g = g.saturating_add(1);
        }
        let issued = issues_to(i, class);
        for u in Unit::ALL {
            if issued & (1 << u.index()) != 0 {
                protected[u.index()] = 0;
                gap[u.index()] = 0;
            }
        }
        match name {
            "STALLWAIT" => {
                let (bm, cm) = masks(i);
                if bm == 0 || cm == 0 {
                    found(Finding {
                        index,
                        kind: Kind::EmptyMask,
                        producer: None,
                        consumer: name,
                    });
                }
                let bm = if bm == 0 { block::MATRIX } else { bm };
                let cm = if cm == 0 { 0x0f } else { cm };
                let mut redundant = true;
                let mut any_unit = false;
                for u in Unit::ALL {
                    if cm & u.condition() != 0 {
                        any_unit = true;
                        if !u.thread_local() || protected[u.index()] & bm != bm {
                            redundant = false;
                        }
                        protected[u.index()] |= bm;
                    }
                }
                // A condition on a source bank or the RISC-V core is not one of
                // this thread's units: not judged.
                let units_only = cm & !(Unit::ALL.iter().fold(0, |a, u| a | u.condition())) == 0;
                if redundant && any_unit && units_only {
                    found(Finding {
                        index,
                        kind: Kind::RedundantWait,
                        producer: None,
                        consumer: name,
                    });
                }
            }
            "SEMWAIT" => last_semwait = Some(masks(i).0),
            "SEMGET" => armed_take = last_semwait.take(),
            _ => {}
        }
        after_mad = class == Class::Sfpu && missed_by_stalling(name);
        after_shiftmask = name == "CFGSHIFTMASK";
    }

    if closed {
        for p in [
            Unit::Unpacker0,
            Unit::Unpacker1,
            Unit::Packer,
            Unit::Matrix,
            Unit::Sfpu,
            Unit::Scalar,
            Unit::Mover,
        ] {
            if protected[p.index()] == 0 {
                found(Finding {
                    index: n,
                    kind: Kind::EndsBusy,
                    producer: Some(p),
                    consumer: "end of program",
                });
            }
        }
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend;
    use crate::sync::{self, Semaphore, Unit as SyncUnit};
    use std::vec::Vec;

    fn run(stream: &[Instruction], closed: bool) -> Vec<Finding> {
        let mut out = Vec::new();
        check(stream.iter().copied(), closed, |f| out.push(f));
        out
    }

    fn misses(stream: &[Instruction], closed: bool) -> Vec<Finding> {
        run(stream, closed)
            .into_iter()
            .filter(|f| f.kind.is_miss())
            .collect()
    }

    fn unpacr() -> Instruction {
        crate::isa::generated::encode::UnpacrRegular::ZERO
            .encode()
            .unwrap()
    }

    #[test]
    fn a_post_after_its_unit_is_covered_and_a_bare_post_is_not() {
        let s = Semaphore::new(3).unwrap();
        let mut good = std::vec![unpacr()];
        good.extend(sync::post_after(SyncUnit::Unpacker0, s));
        assert!(misses(&good, true).is_empty());

        let bare = std::vec![unpacr(), sync::post(s)];
        let f = misses(&bare, true);
        assert!(f
            .iter()
            .any(|f| f.kind == Kind::PublishBeforeDrain && f.producer == Some(Unit::Unpacker0)));
        assert!(f.iter().any(|f| f.kind == Kind::EndsBusy));
    }

    #[test]
    fn a_wait_that_does_not_block_the_consumer_does_not_cover_it() {
        let s = Semaphore::new(1).unwrap();
        // Drained, but blocking only the unpackers: the post still races.
        let weak = std::vec![
            unpacr(),
            backend::stallwait(block::UNPACKER, cond::UNPACKER0_BUSY).unwrap(),
            sync::post(s),
        ];
        assert!(misses(&weak, false)
            .iter()
            .any(|f| f.kind == Kind::PublishBeforeDrain));
    }
}
