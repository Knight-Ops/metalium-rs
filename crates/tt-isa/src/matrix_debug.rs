//! Matrix-unit diagnostics: `MOVDBGA2D` and `GATESRCRST`.
//!
//! Both are *diagnostic* surfaces. Normal kernels read `SrcA` with `MOVA2D`,
//! which waits for the Matrix Unit to own the bank. Nothing here is on a tensor
//! path, and nothing here is a production kernel primitive.
//!
//! # `MOVDBGA2D`
//!
//! `MOVDBGA2D` is `MOVA2D` without the wait for
//! `SrcA[MatrixUnit.SrcABank].AllowedClient == MatrixUnit`
//! (`WormholeB0/.../MOVDBGA2D.md`; Blackhole has no page, only the measured
//! `AddrMod` position, `MOVDBGA2D_BH`). Its value is that it can read a bank the
//! unpacker still owns -- staged rows that were never handed over -- where
//! `MOVA2D` would wait forever. The price is that the hardware no longer
//! orders the read against the unpacker, so this module does it:
//!
//! * [`Banks::debug_move_a`] exists only for [`Filling`] and [`Loaded`] `SrcA`.
//!   In the lockstep discipline of [`crate::matrix`] the Matrix Unit's bank
//!   pointer equals the unpacker's while the operand is [`Filling`], so the
//!   move reads the bank being filled; [`Empty`](crate::matrix::Empty) holds nothing
//!   to read and is refused at compile time.
//! * The helper emits `STALLWAIT unpacker 0 idle` before the move, so a
//!   partial unpack has retired when it is read.
//!
//! The format the page leaves implicit is established explicitly by
//! [`DebugFormat::setup`]: `DISABLE_IMPLIED_SRCA_FMT_Base` is set (the page
//! "strongly recommends" it, as the implied format of a bank that is not
//! valid is `NonContractualBehavior`) and `SrcA`'s format comes from the
//! override field. The page's undefined combinations are rejected at
//! construction, not at run time:
//!
//! * `UseDst32b || UseDst32bLo` must equal `Fp32_enabled || INT8_math_enabled`
//!   ("`DstRowValid` update addressing is incorrect").
//! * `UseDst32b && UseDst32bLo` corrupts the write (HW erratum TEN-4245).
//! * With `Fp32_enabled`, Blackhole forces `SrcAFmt = TF32`, so the override
//!   value does not select the format there ([`DebugMode::Dst32Tf32`]).
//!
//! The one-row/eight-row forms mask their row arguments (`SrcRow &= 0x38`,
//! `DstRow &= 0x3f8` for eight rows), which would move a different block than
//! the one asked for, so an unaligned eight-row argument is an error here.
//! [`model`] is an independent port of the page's functional model.
//!
//! # `GATESRCRST`
//!
//! "There is a one-slot operand cache between `SrcB` and the Matrix Unit. This
//! instruction will forcibly invalidate the cache. It should only be required if
//! there are hardware bugs in the cache invalidation logic"
//! (`WormholeB0/.../GATESRCRST.md`; `UNVERIFIED` on Blackhole: the encoding is a
//! Wormhole page's, and the page defines no observable cache state). The helper
//! [`invalidate_src_b_cache`] only encodes. Whether it *does* anything is the
//! open question; [`src_b_cache_probe`] is the controlled experiment, with a
//! pre-declared decision rule on [`CacheOutcome`].

use crate::backend::{self, Before, ConfigWords, ThreadConfigEntry};
use crate::cfg::generated::{alu, thread};
use crate::isa::generated::encode;
use crate::isa::{EncodeError, Instruction};
use crate::matrix::{Banks, Filling, Loaded};

/// Why a diagnostic could not be built.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DebugError {
    /// A row argument is out of range or would be masked into another block.
    Row {
        which: &'static str,
        value: u32,
        reason: &'static str,
    },
    /// A format combination the page calls undefined.
    Format(&'static str),
    /// A field did not fit.
    Encode(EncodeError),
    /// A configuration write could not be staged.
    Config(backend::EncodeError),
}

impl From<EncodeError> for DebugError {
    fn from(e: EncodeError) -> Self {
        DebugError::Encode(e)
    }
}

impl From<backend::EncodeError> for DebugError {
    fn from(e: backend::EncodeError) -> Self {
        DebugError::Config(e)
    }
}

impl core::fmt::Display for DebugError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DebugError::Row {
                which,
                value,
                reason,
            } => write!(f, "MOVDBGA2D {which} row {value}: {reason}"),
            DebugError::Format(why) => write!(f, "MOVDBGA2D format: {why}"),
            DebugError::Encode(e) => write!(f, "{e}"),
            DebugError::Config(e) => write!(f, "{e:?}"),
        }
    }
}

/// What `SrcA` holds, by the data-format code the unpacker and the
/// `ALU_FORMAT_SPEC_REG_SrcA_val` field use.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SrcAFormat {
    Fp32,
    Fp16,
    Tf32,
    Bf16,
}

impl SrcAFormat {
    pub const fn code(self) -> u32 {
        match self {
            SrcAFormat::Fp32 => 0,
            SrcAFormat::Fp16 => 1,
            SrcAFormat::Tf32 => 4,
            SrcAFormat::Bf16 => 5,
        }
    }

    /// The page's `Use8bExponent` set, for these formats.
    pub const fn eight_bit_exponent(self) -> bool {
        !matches!(self, SrcAFormat::Fp16)
    }
}

/// How the move writes `Dst`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DebugMode {
    /// `ALU_ACC_CTRL_Fp32_enabled`: 32-bit `Dst`. Blackhole forces `SrcAFmt` to
    /// TF32, so the move writes a TF32/FP32 datum whatever the override says.
    /// `UseDst32bLo` is clear (set is erratum TEN-4245).
    Dst32Tf32,
    /// Neither `Fp32_enabled` nor `INT8_math_enabled`: 16-bit `Dst`, the
    /// format being BF16 or FP16 as `SrcAFmt` says. `UseDst32bLo` is clear.
    Dst16,
    /// `INT8_math_enabled` without `Fp32_enabled`: the low half of a 32-bit
    /// `Dst` datum, `UseDst32bLo` set, `SrcAFmt` not TF32. The only form of
    /// `UseDst32bLo` the page does not call undefined.
    Dst32Low,
}

/// A `MOVDBGA2D` format, established explicitly.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DebugFormat {
    mode: DebugMode,
    src_format: SrcAFormat,
    flush_denormals: bool,
}

impl DebugFormat {
    /// `src_format` is the value the override field is given. In
    /// [`DebugMode::Dst32Tf32`] it does not select the format (TF32 is forced),
    /// so any value is accepted there; in the other modes it must not be TF32,
    /// which would make the page's `UseDst32b` disagree with the configuration.
    /// `flush_denormals` is `!ALU_ACC_CTRL_Zero_Flag_disabled_src`.
    pub const fn new(
        mode: DebugMode,
        src_format: SrcAFormat,
        flush_denormals: bool,
    ) -> Result<Self, DebugError> {
        if !matches!(mode, DebugMode::Dst32Tf32) && matches!(src_format, SrcAFormat::Tf32) {
            return Err(DebugError::Format(
                "TF32 SrcA without Fp32_enabled: UseDst32b disagrees with the configuration",
            ));
        }
        Ok(DebugFormat {
            mode,
            src_format,
            flush_denormals,
        })
    }

    pub const fn mode(self) -> DebugMode {
        self.mode
    }

    /// The format the move actually uses (`SrcAFmt` after Blackhole's TF32 forcing).
    pub const fn effective_format(self) -> SrcAFormat {
        match self.mode {
            DebugMode::Dst32Tf32 => SrcAFormat::Tf32,
            _ => self.src_format,
        }
    }

    pub const fn flush_denormals(self) -> bool {
        self.flush_denormals
    }

    /// The `UseDst32bLo` operand this format requires.
    pub const fn use_dst32b_lo(self) -> bool {
        matches!(self.mode, DebugMode::Dst32Low)
    }

    /// The configuration that makes this the format of every `MOVDBGA2D` that
    /// follows: one `SETC16` for `DISABLE_IMPLIED_SRCA_FMT_Base`, and `Config`
    /// words for the override, the accumulation mode and the denormal flush.
    ///
    /// [`ConfigWords`] composes whole words from zero: every other field of the
    /// words it touches is written as zero, as for every `ConfigWords` user. A
    /// caller whose program configures other fields of those words must set them
    /// in the same value (as `tt_kernels::matmul::unpack_prelude` does for
    /// `Fp32_enabled`).
    pub fn setup(self) -> Result<(Instruction, ConfigWords), DebugError> {
        let disable = ThreadConfigEntry::zeroed(thread::DISABLE_IMPLIED_SRCA_FMT_Base.addr32())
            .set(thread::DISABLE_IMPLIED_SRCA_FMT_Base, 1)?
            .encode()?;
        let mut words = ConfigWords::new();
        words.set(alu::ALU_FORMAT_SPEC_REG_SrcA_override, 1)?;
        words.set(alu::ALU_FORMAT_SPEC_REG_SrcA_val, self.src_format.code())?;
        words.set(
            alu::ALU_ACC_CTRL_Fp32_enabled,
            u32::from(matches!(self.mode, DebugMode::Dst32Tf32)),
        )?;
        words.set(
            alu::ALU_ACC_CTRL_INT8_math_enabled,
            u32::from(matches!(self.mode, DebugMode::Dst32Low)),
        )?;
        words.set(
            alu::ALU_ACC_CTRL_Zero_Flag_disabled_src,
            u32::from(!self.flush_denormals),
        )?;
        Ok((disable, words))
    }
}

/// The rows one `MOVDBGA2D` moves.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Rows {
    One {
        src_row: u32,
        dst_row: u32,
    },
    /// An aligned block; both rows are multiples of eight.
    Eight {
        src_row: u32,
        dst_row: u32,
    },
}

/// A checked `MOVDBGA2D`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DebugMove {
    rows: Rows,
    addr_mod: u32,
    format: DebugFormat,
}

impl DebugMove {
    /// `src_row` is 0..64 and `dst_row` 0..1024 (the operand widths, before
    /// the RWC and base offsets the hardware adds); `addr_mod` is the
    /// Blackhole three-bit selector (divergence row 42). Eight-row arguments
    /// must be multiples of eight, because the hardware masks them.
    pub fn new(rows: Rows, addr_mod: u32, format: DebugFormat) -> Result<Self, DebugError> {
        let (src, dst, align) = match rows {
            Rows::One { src_row, dst_row } => (src_row, dst_row, 1),
            Rows::Eight { src_row, dst_row } => (src_row, dst_row, 8),
        };
        if src >= 64 {
            return Err(DebugError::Row {
                which: "source",
                value: src,
                reason: "SrcRow is six bits",
            });
        }
        if dst >= 1024 {
            return Err(DebugError::Row {
                which: "destination",
                value: dst,
                reason: "DstRow is ten bits",
            });
        }
        if src % align != 0 {
            return Err(DebugError::Row {
                which: "source",
                value: src,
                reason: "eight-row moves mask SrcRow to a multiple of eight",
            });
        }
        if dst % align != 0 {
            return Err(DebugError::Row {
                which: "destination",
                value: dst,
                reason: "eight-row moves mask DstRow to a multiple of eight",
            });
        }
        let m = DebugMove {
            rows,
            addr_mod,
            format,
        };
        m.encode()?;
        Ok(m)
    }

    pub const fn rows(self) -> Rows {
        self.rows
    }

    pub const fn format(self) -> DebugFormat {
        self.format
    }

    /// The `MOVDBGA2D` word, with `UseDst32bLo` as the format requires.
    pub fn encode(self) -> Result<Instruction, EncodeError> {
        let (src, dst, eight) = match self.rows {
            Rows::One { src_row, dst_row } => (src_row, dst_row, 0),
            Rows::Eight { src_row, dst_row } => (src_row, dst_row, 1),
        };
        encode::Movdbga2D::ZERO
            .use_dst32b_lo(u32::from(self.format.use_dst32b_lo()))
            .src_row(src)
            .addr_mod(self.addr_mod)
            .move8_rows(eight)
            .dst_row(dst)
            .encode()
    }
}

/// `SrcA` states in which a debug move has something to read: the operand
/// holds staged or handed-over rows. Sealed.
pub trait DebugReadable: private::Sealed {}
mod private {
    pub trait Sealed {}
    impl Sealed for super::Filling {}
    impl Sealed for super::Loaded {}
}
impl DebugReadable for Filling {}
impl DebugReadable for Loaded {}

impl<A: DebugReadable, B> Banks<A, B> {
    /// Wait for unpacker 0 to go idle, then `MOVDBGA2D`. The state does not
    /// change: neither instruction moves a bank pointer.
    ///
    /// The wait drains unpack work issued by *this thread*; an unpack another
    /// thread issued needs the semaphore that thread posts when it retires.
    ///
    /// The format must already be in effect ([`DebugFormat::setup`]); it is
    /// part of `mv` so the `UseDst32bLo` operand cannot disagree with it.
    pub fn debug_move_a(self, mv: DebugMove) -> Result<([Instruction; 2], Self), DebugError> {
        let wait = backend::wait_for_unpacker0(Before::EVERYTHING)?;
        Ok(([wait, mv.encode()?], self))
    }
}

/// `GATESRCRST` with `InvalidateSrcBCache` set. `UNVERIFIED` on Blackhole.
pub fn invalidate_src_b_cache() -> Instruction {
    encode::gatesrcrst(1).expect("InvalidateSrcBCache is one bit")
}

/// `GATESRCRST` with `InvalidateSrcBCache` clear: the page gives the operand no
/// other meaning, so this is the experiment's control, not a feature.
pub fn clear_src_b_cache_operand() -> Instruction {
    encode::gatesrcrst(0).expect("InvalidateSrcBCache is one bit")
}

impl<A> Banks<A, Loaded> {
    /// [`invalidate_src_b_cache`], only while the Matrix Unit owns `SrcB`, so
    /// what is invalidated is a cache of data the caller holds.
    pub fn gatesrcrst(self) -> Result<(Instruction, Self), EncodeError> {
        Ok((encode::gatesrcrst(1)?, self))
    }
}

/// An arm of the stale-versus-fresh `SrcB` experiment.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CacheArm {
    /// `SrcB` is rewritten through `MOVD2B`, then the second `MVMUL` runs.
    NoGate,
    /// As [`CacheArm::NoGate`] plus `GATESRCRST` with the operand clear.
    GateOperandClear,
    /// As [`CacheArm::NoGate`] plus `GATESRCRST` with `InvalidateSrcBCache` set.
    GateInvalidate,
    /// Detector control: no rewrite at all, so the second `MVMUL` must give the
    /// stale prediction. It shows the two predictions are separable.
    NoRewrite,
}

/// The experiment's instructions (`tt-isa` has no allocator).
#[derive(Copy, Clone, Debug)]
pub struct CacheProgram {
    words: [Instruction; CacheProgram::CAPACITY],
    len: usize,
}

impl CacheProgram {
    const CAPACITY: usize = 10;

    fn new() -> Self {
        CacheProgram {
            words: [backend::nop(); Self::CAPACITY],
            len: 0,
        }
    }

    fn push(&mut self, i: Instruction) {
        self.words[self.len] = i;
        self.len += 1;
    }

    pub fn as_slice(&self) -> &[Instruction] {
        &self.words[..self.len]
    }
}

/// First `Dst` row of the staging block: `SrcA` rows 8..16, copied by `MOVA2D`.
pub const PROBE_STAGE_ROWS: u32 = 8;

/// The experiment's matrix-unit program, for a bank pair loaded with `SrcA` = `M`
/// (16x16) and `SrcB` = `B0` (8x16, at row 0), `RWC.SrcA = RWC.SrcB = RWC.Dst = 0`,
/// `Dst` zeroed:
///
/// 1. `MVMUL` at `Dst` 0: `Dst[0..8] = B0 @ M`. This reads `SrcB`, so a cache holds it.
/// 2. `MOVA2D` eight rows, `SrcA` 8..16 into `Dst` 8..16, and (unless
///    [`CacheArm::NoRewrite`]) `MOVD2B` four rows twice, `Dst` 8..16 into `SrcB`
///    0..8: `SrcB` becomes `B1 = M[8..16]`, written by a path that is not an unpack.
/// 3. The arm's `GATESRCRST`, if any, after the matrix queue drains.
/// 4. `MVMUL` at `Dst` 0 again, releasing both banks.
///
/// Fresh result: `Dst[0..8] = B0 @ M + B1 @ M`. Stale result: `2 * (B0 @ M)`.
/// `Dst[8..16] = M[8..16]` in every arm. See [`CacheOutcome`] for the decision rule.
pub fn src_b_cache_probe(
    banks: Banks<Loaded, Loaded>,
    arm: CacheArm,
) -> Result<CacheProgram, DebugError> {
    let mut p = CacheProgram::new();
    let (i, banks) = banks.mvmul(encode::Mvmul::ZERO.dst_row(0))?;
    p.push(i);
    p.push(backend::wait_for_matrix(Before::EVERYTHING)?);
    let (i, mut banks) = banks.mova2d(
        encode::Mova2D::ZERO
            .move8_rows(1)
            .src_row(8)
            .dst_row(PROBE_STAGE_ROWS),
    )?;
    p.push(i);
    p.push(backend::wait_for_matrix(Before::EVERYTHING)?);
    if arm != CacheArm::NoRewrite {
        for half in [0, 4] {
            let (i, next) = banks.movd2b(
                encode::Movd2B::ZERO
                    .move4_rows(1)
                    .src_row(half)
                    .dst_row(PROBE_STAGE_ROWS + half),
            )?;
            p.push(i);
            banks = next;
        }
        p.push(backend::wait_for_matrix(Before::EVERYTHING)?);
    }
    match arm {
        CacheArm::GateOperandClear => p.push(clear_src_b_cache_operand()),
        CacheArm::GateInvalidate => {
            let (i, next) = banks.gatesrcrst()?;
            p.push(i);
            banks = next;
        }
        CacheArm::NoGate | CacheArm::NoRewrite => {}
    }
    p.push(backend::wait_for_matrix(Before::EVERYTHING)?);
    let (i, _) = banks.mvmul_release_both(encode::Mvmul::ZERO.dst_row(0))?;
    p.push(i);
    Ok(p)
}

/// How a run of the experiment's second `MVMUL` is classified.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CacheOutcome {
    /// `B0 @ M + B1 @ M`.
    Fresh,
    /// `2 * (B0 @ M)`: the `MVMUL` used the `SrcB` it had cached.
    Stale,
    /// Neither: something other than the cache is wrong; the experiment says nothing.
    Neither,
}

/// What the experiment concludes about `GATESRCRST`, from the outcomes of the
/// [`CacheArm::NoGate`] and [`CacheArm::GateInvalidate`] arms. Fixed before any run.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CacheVerdict {
    /// No gate: stale. Gate: fresh. The instruction has an observable effect:
    /// semantic completion is available, with this as its gate.
    Observable,
    /// Both arms fresh. No observable effect: the page says the instruction
    /// is needed only against hardware bugs, and the experiment found none. The
    /// instruction is `[-]` "no observable oracle"; a no-effect comparison is
    /// not evidence of invalidation.
    NoObservableOracle,
    /// Any other pair, including a stale gate arm. Investigate; claim nothing.
    Inconclusive,
}

pub const fn cache_verdict(no_gate: CacheOutcome, gate: CacheOutcome) -> CacheVerdict {
    match (no_gate, gate) {
        (CacheOutcome::Stale, CacheOutcome::Fresh) => CacheVerdict::Observable,
        (CacheOutcome::Fresh, CacheOutcome::Fresh) => CacheVerdict::NoObservableOracle,
        _ => CacheVerdict::Inconclusive,
    }
}

/// An independent port of the `MOVA2D` functional model (`MOVDBGA2D` is identical
/// but for the wait), for `MOVDBGA2D`'s diagnostic use.
pub mod model {
    use super::{DebugFormat, DebugMode, SrcAFormat};

    /// The configuration and counters the model reads.
    #[derive(Copy, Clone, Debug)]
    pub struct State {
        pub format: DebugFormat,
        /// `RWCs.SrcA`.
        pub src_rwc: u32,
        /// `RWCs.Dst + DEST_REGW_BASE_Base + DEST_TARGET_REG_CFG_MATH_Offset`.
        pub dst_base: u32,
        /// `LaneConfig[c / 2].BLOCK_DEST_MOV` bit `c & 1`: columns not written.
        pub block_columns: [u8; 8],
    }

    /// One datum the move wrote.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum Write {
        /// `Dst32b[row][column]`, physical layout (`Sign, HiMan(7), Exp(8), LoMan(3), 0(13)`).
        Dst32 {
            row: usize,
            column: usize,
            value: u32,
        },
        /// `Dst16b[row][column]`, physical (`BF16: Sign, Man(7), Exp(8)`;
        /// `FP16: Sign, Man(10), Exp(5)`).
        Dst16 {
            row: usize,
            column: usize,
            value: u16,
        },
        /// The low half of `Dst32b[row][column]`, the high half untouched.
        Dst32Low {
            row: usize,
            column: usize,
            value: u16,
        },
    }

    /// The page's `RemoveLowMantissa`, on a 19-bit `Src` datum.
    pub const fn remove_low_mantissa(x: u32) -> u16 {
        (((x & (1 << 18)) >> 3) | ((x & (0x7f << 11)) >> 3) | (x & 0xff)) as u16
    }

    /// The page's `RemoveHighExponent`.
    pub const fn remove_high_exponent(x: u32) -> u16 {
        (((x & (1 << 18)) >> 3) | ((x & (0x3ff << 8)) >> 3) | (x & 0x1f)) as u16
    }

    /// The IEEE-754 bits of a physical `Dst32b` TF32/FP32 datum, as the
    /// harness's FP32 dump reads it: `Sign, HiMan, Exp, LoMan` to
    /// `Sign, Exp, Man`.
    pub const fn dst32_to_ieee(x: u32) -> u32 {
        (x & 0x8000_0000) | (((x >> 16) & 0xff) << 23) | (((x >> 24) & 0x7f) << 16) | (x & 0xe000)
    }

    /// The `Src` datum (`Sign, Man(10), Exp(8)`) of an IEEE FP32 value after the
    /// unpacker's TF32 truncation: the independent route from L1 data to `SrcA`.
    pub const fn tf32_src_datum(bits: u32) -> u32 {
        ((bits >> 31) << 18) | (((bits >> 13) & 0x3ff) << 8) | ((bits >> 23) & 0xff)
    }

    /// Apply one move to the 64-row `SrcA` bank, as the page says.
    ///
    /// `Err` carries the page's `UndefinedBehavior`: the combinations
    /// [`DebugFormat::new`] already refuses cannot arise, so this fires only
    /// when `Fp32` and `UseDst32bLo` are combined by hand.
    pub fn move_rows(
        src: &[[u32; 16]; 64],
        state: State,
        src_row: u32,
        dst_row: u32,
        move8_rows: bool,
        use_dst32b_lo: bool,
        mut sink: impl FnMut(Write),
    ) -> Result<(), &'static str> {
        let fp32 = matches!(state.format.mode(), DebugMode::Dst32Tf32);
        let int8 = matches!(state.format.mode(), DebugMode::Dst32Low);
        let fmt = if fp32 {
            SrcAFormat::Tf32
        } else {
            state.format.effective_format()
        };
        let use_8b_exp = fp32 || int8 || fmt.eight_bit_exponent();
        let use_dst32b = fmt == SrcAFormat::Tf32;
        if (use_dst32b || use_dst32b_lo) != (fp32 || int8) {
            return Err("DstRowValid update addressing is incorrect");
        }
        let mut dst = dst_row + state.dst_base;
        let mut src_r = src_row + state.src_rwc;
        let n = if move8_rows {
            dst &= 0x3f8;
            src_r &= 0x38;
            8
        } else {
            dst &= 0x3ff;
            src_r &= 0x3f;
            1
        };
        for k in 0..n {
            let (d, s) = ((dst + k) as usize, ((src_r + k) & 0x3f) as usize);
            for (c, &datum) in src[s].iter().enumerate() {
                if state.block_columns[c / 2] & (1 << (c & 1)) != 0 {
                    continue;
                }
                let mut v = datum;
                if state.format.flush_denormals() && v & 0xff == 0 {
                    v = 0;
                }
                let v16 = if use_8b_exp {
                    remove_low_mantissa(v)
                } else {
                    remove_high_exponent(v)
                };
                if use_dst32b {
                    if use_dst32b_lo {
                        return Err("write data corrupted (HW erratum TEN-4245)");
                    }
                    let low = ((v >> 8) & 7) << 13;
                    sink(Write::Dst32 {
                        row: d,
                        column: c,
                        value: (u32::from(v16) << 16) | low,
                    });
                } else if use_dst32b_lo {
                    sink(Write::Dst32Low {
                        row: d,
                        column: c,
                        value: v16,
                    });
                } else {
                    sink(Write::Dst16 {
                        row: d,
                        column: c,
                        value: v16,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use model::Write;
    use std::vec::Vec;

    /// `model::move_rows`, collected.
    fn moved(
        src: &[[u32; 16]; 64],
        state: model::State,
        src_row: u32,
        dst_row: u32,
        eight: bool,
        lo: bool,
    ) -> Result<Vec<Write>, &'static str> {
        let mut out = Vec::new();
        model::move_rows(src, state, src_row, dst_row, eight, lo, |w| out.push(w))?;
        Ok(out)
    }

    fn fp32() -> DebugFormat {
        DebugFormat::new(DebugMode::Dst32Tf32, SrcAFormat::Tf32, true).unwrap()
    }

    #[test]
    fn row_forms_encode_at_the_measured_blackhole_positions() {
        let one = DebugMove::new(
            Rows::One {
                src_row: 5,
                dst_row: 9,
            },
            1,
            fp32(),
        )
        .unwrap();
        // DstRow 0..9, Move8Rows 13, AddrMod 14..16, SrcRow 17..22, UseDst32bLo 23.
        assert_eq!(
            one.encode().unwrap().word() & 0x00ff_ffff,
            9 | (1 << 14) | (5 << 17)
        );
        let eight = DebugMove::new(
            Rows::Eight {
                src_row: 8,
                dst_row: 16,
            },
            4,
            fp32(),
        )
        .unwrap();
        assert_eq!(
            eight.encode().unwrap().word() & 0x00ff_ffff,
            16 | (1 << 13) | (4 << 14) | (8 << 17)
        );
        assert_eq!(eight.encode().unwrap().word() >> 24, 0x09);
    }

    #[test]
    fn unchecked_arguments_are_refused() {
        let ok = |rows, am| DebugMove::new(rows, am, fp32());
        assert!(ok(
            Rows::One {
                src_row: 64,
                dst_row: 0
            },
            0
        )
        .is_err());
        assert!(ok(
            Rows::One {
                src_row: 0,
                dst_row: 1024
            },
            0
        )
        .is_err());
        assert!(ok(
            Rows::One {
                src_row: 63,
                dst_row: 1023
            },
            7
        )
        .is_ok());
        assert!(ok(
            Rows::One {
                src_row: 0,
                dst_row: 0
            },
            8
        )
        .is_err());
        assert!(ok(
            Rows::Eight {
                src_row: 4,
                dst_row: 0
            },
            0
        )
        .is_err());
        assert!(ok(
            Rows::Eight {
                src_row: 0,
                dst_row: 12
            },
            0
        )
        .is_err());
        assert!(ok(
            Rows::Eight {
                src_row: 56,
                dst_row: 1016
            },
            0
        )
        .is_ok());
    }

    #[test]
    fn page_undefined_format_combinations_cannot_be_built() {
        let f = SrcAFormat::Tf32;
        assert!(DebugFormat::new(DebugMode::Dst16, f, true).is_err());
        assert!(DebugFormat::new(DebugMode::Dst32Low, f, true).is_err());
        // The override is ignored under Fp32_enabled (TF32 forced), so it is accepted.
        let f = DebugFormat::new(DebugMode::Dst32Tf32, SrcAFormat::Bf16, true).unwrap();
        assert_eq!(f.effective_format(), SrcAFormat::Tf32);
        assert!(!f.use_dst32b_lo());
        let low = DebugFormat::new(DebugMode::Dst32Low, SrcAFormat::Bf16, true).unwrap();
        let m = DebugMove::new(
            Rows::One {
                src_row: 0,
                dst_row: 0,
            },
            0,
            low,
        )
        .unwrap();
        assert_eq!(m.encode().unwrap().word() >> 23 & 1, 1);
    }

    #[test]
    fn setup_disables_the_implied_format_and_names_every_field() {
        let (i, words) = fp32().setup().unwrap();
        assert_eq!(i.def().mnemonic(), "SETC16");
        let get = |f: crate::cfg::ConfigField| {
            let (_, w) = words
                .words()
                .iter()
                .copied()
                .find(|&(a, _)| a == f.addr32())
                .unwrap();
            f.extract(w)
        };
        assert_eq!(get(alu::ALU_FORMAT_SPEC_REG_SrcA_override), 1);
        assert_eq!(get(alu::ALU_FORMAT_SPEC_REG_SrcA_val), 4);
        assert_eq!(get(alu::ALU_ACC_CTRL_Fp32_enabled), 1);
        assert_eq!(get(alu::ALU_ACC_CTRL_INT8_math_enabled), 0);
        assert_eq!(get(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src), 0);
    }

    #[test]
    fn a_debug_move_waits_for_the_unpacker_and_keeps_the_state() {
        use crate::matrix::Banks;
        let (_, b) = Banks::after_reset()
            .unpack_a_partial(encode::UnpacrRegular::ZERO)
            .unwrap();
        let mv = DebugMove::new(
            Rows::Eight {
                src_row: 0,
                dst_row: 0,
            },
            0,
            fp32(),
        )
        .unwrap();
        let ([wait, mov], b) = b.debug_move_a(mv).unwrap();
        assert_eq!(wait.def().mnemonic(), "STALLWAIT");
        assert_eq!(mov.def().key(), "MOVDBGA2D_BH");
        // Still Filling: the unpacker can continue.
        let _ = b.unpack_a_partial(encode::UnpacrRegular::ZERO).unwrap();
    }

    #[test]
    fn gatesrcrst_encodes_the_wormhole_operand_position() {
        assert_eq!(invalidate_src_b_cache().word(), 0x3500_0002);
        assert_eq!(clear_src_b_cache_operand().word(), 0x3500_0000);
        assert_eq!(invalidate_src_b_cache().def().mnemonic(), "GATESRCRST");
    }

    #[test]
    fn probe_arms_differ_exactly_where_they_claim() {
        let loaded = || {
            let (_, b) = Banks::after_reset()
                .unpack_a(encode::UnpacrRegular::ZERO)
                .unwrap();
            let (_, b) = b.unpack_b(encode::UnpacrRegular::ZERO).unwrap();
            b
        };
        let keys = |arm| -> Vec<&'static str> {
            src_b_cache_probe(loaded(), arm)
                .unwrap()
                .as_slice()
                .iter()
                .map(|i| i.def().mnemonic())
                .collect()
        };
        let base = keys(CacheArm::NoGate);
        assert_eq!(base.iter().filter(|k| **k == "MOVD2B").count(), 2);
        assert!(!base.contains(&"GATESRCRST"));
        assert_eq!(keys(CacheArm::GateInvalidate).len(), base.len() + 1);
        assert_eq!(keys(CacheArm::GateOperandClear).len(), base.len() + 1);
        let norewrite = keys(CacheArm::NoRewrite);
        assert!(!norewrite.contains(&"MOVD2B"));
        assert_eq!(norewrite.iter().filter(|k| **k == "MVMUL").count(), 2);
    }

    #[test]
    fn verdict_is_fixed_before_the_run() {
        use CacheOutcome::*;
        assert_eq!(cache_verdict(Stale, Fresh), CacheVerdict::Observable);
        assert_eq!(
            cache_verdict(Fresh, Fresh),
            CacheVerdict::NoObservableOracle
        );
        assert_eq!(cache_verdict(Stale, Stale), CacheVerdict::Inconclusive);
        assert_eq!(cache_verdict(Fresh, Stale), CacheVerdict::Inconclusive);
        assert_eq!(cache_verdict(Neither, Fresh), CacheVerdict::Inconclusive);
    }

    #[test]
    fn model_matches_the_page_for_each_mode() {
        use model::*;
        let mut src = [[0u32; 16]; 64];
        // Sign 1, Man 0b10_1010_1101 (10b), Exp 0x81.
        let datum = (1 << 18) | (0b10_1010_1101 << 8) | 0x81;
        src[3][5] = datum;
        src[3][6] = 0x7f00; // exponent zero: flushed
        let state = |f| State {
            format: f,
            src_rwc: 0,
            dst_base: 0,
            block_columns: [0; 8],
        };
        let w = moved(&src, state(fp32()), 3, 9, false, false).unwrap();
        // HiMan = 0b1010101, LoMan = 0b101; physical Sign,HiMan,Exp,LoMan,0.
        let want = (1u32 << 31) | (0b101_0101 << 24) | (0x81 << 16) | (0b101 << 13);
        assert_eq!(
            w[5],
            Write::Dst32 {
                row: 9,
                column: 5,
                value: want
            }
        );
        assert_eq!(
            w[6],
            Write::Dst32 {
                row: 9,
                column: 6,
                value: 0
            }
        );
        // IEEE view: sign, exp 0x81, mantissa 0b10_1010_1101 << 13.
        assert_eq!(
            dst32_to_ieee(want),
            (1 << 31) | (0x81 << 23) | (0b10_1010_1101 << 13)
        );
        assert_eq!(tf32_src_datum(dst32_to_ieee(want)), datum);

        let bf16 = DebugFormat::new(DebugMode::Dst16, SrcAFormat::Bf16, true).unwrap();
        let w = moved(&src, state(bf16), 3, 9, false, false).unwrap();
        assert_eq!(
            w[5],
            Write::Dst16 {
                row: 9,
                column: 5,
                value: ((1 << 15) | (0b101_0101 << 8) | 0x81) as u16
            }
        );
        let fp16 = DebugFormat::new(DebugMode::Dst16, SrcAFormat::Fp16, true).unwrap();
        let d16 = (1 << 18) | (0b11_0000_0001 << 8) | 0x1d;
        src[3][5] = d16;
        let w = moved(&src, state(fp16), 3, 9, false, false).unwrap();
        assert_eq!(
            w[5],
            Write::Dst16 {
                row: 9,
                column: 5,
                value: ((1 << 15) | (0b11_0000_0001 << 5) | 0x1d) as u16
            }
        );
        let low = DebugFormat::new(DebugMode::Dst32Low, SrcAFormat::Bf16, false).unwrap();
        let w = moved(&src, state(low), 3, 9, false, true).unwrap();
        assert!(matches!(
            w[5],
            Write::Dst32Low {
                row: 9,
                column: 5,
                ..
            }
        ));
        // Flush disabled keeps the zero-exponent datum.
        assert_eq!(
            w[6],
            Write::Dst32Low {
                row: 9,
                column: 6,
                value: remove_low_mantissa(0x7f00)
            }
        );
        // Undefined combinations the model itself refuses.
        assert!(moved(&src, state(fp32()), 3, 9, false, true).is_err());
        assert!(moved(&src, state(bf16), 3, 9, false, true).is_err());
    }

    #[test]
    fn eight_row_moves_mask_and_one_row_moves_wrap_the_source() {
        use model::*;
        let mut src = [[0u32; 16]; 64];
        for (r, row) in src.iter_mut().enumerate() {
            row[0] = 0x100 | (r as u32 + 1);
        }
        let st = State {
            format: fp32(),
            src_rwc: 0,
            dst_base: 0,
            block_columns: [0; 8],
        };
        // RWC.SrcA 4 + SrcRow 8 = 12, masked to 8 for eight rows.
        let w = moved(&src, State { src_rwc: 4, ..st }, 8, 16, true, false).unwrap();
        let col0: Vec<_> = w
            .iter()
            .filter_map(|w| match w {
                Write::Dst32 {
                    row,
                    column: 0,
                    value,
                } => Some((*row, *value)),
                _ => None,
            })
            .collect();
        assert_eq!(col0.len(), 8);
        assert_eq!(col0[0].0, 16);
        assert_eq!(col0[0].1 >> 16 & 0xff, 9); // source row 8, exponent 9
                                               // One row: SrcRow 63 + RWC 2 wraps to 1.
        let w = moved(&src, State { src_rwc: 2, ..st }, 63, 0, false, false).unwrap();
        assert!(
            matches!(w[0], Write::Dst32 { row: 0, column: 0, value } if value >> 16 & 0xff == 2)
        );
        // Blocked columns are skipped.
        let mut blocked = st;
        blocked.block_columns[0] = 0b01;
        let w = moved(&src, blocked, 0, 0, false, false).unwrap();
        assert!(!w
            .iter()
            .any(|w| matches!(w, Write::Dst32 { column: 0, .. })));
        assert_eq!(w.len(), 15);
    }
}
