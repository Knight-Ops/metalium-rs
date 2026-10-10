//! Physical FP16 tiles: the same 2112-byte slot as BF16, a different element
//! format. Conversions run on Tensix; B/NC only copy the packed payload.
//!
//! The stored bits are **IEEE binary16**. The coprocessor's FP16 is not: it has
//! no NaN, reads `Exp == 31` as the finite value `(1 + m/1024) * 2^16`, and
//! treats denormals as zero (`FloatBitPatterns.md#fp16`, Wormhole
//! documentation; the Blackhole packer/unpacker behaviour is UNMEASURED and is
//! the subject of the T7 probe, divergence row 101). The shipped casts
//! therefore never ask the packer or unpacker for an FP16 conversion:
//!
//! * **widening** ([`Widening::Integer`]): B copies each stored halfword into
//!   the low half of a 32-bit datum of an F32 tile, the unpacker moves it to
//!   `Dst` as raw bits (the D3 pass-through), and an SFPU integer program
//!   builds the exact binary16-to-FP32 value ([`widen`]), subnormals, infinities
//!   and NaN payloads included;
//! * **narrowing** ([`Narrowing::Integer`]): an SFPU integer program computes
//!   the exact ties-even binary16 bits ([`narrow`]) in the low half of each
//!   datum, the packer writes them as raw FP32-coded words, and B compacts the
//!   halfwords into the slot.
//!
//! Both are bit-exact against the IEEE definition on ttsim and silicon. The
//! coprocessor's own conversions stay available as measurement variants
//! ([`Narrowing::Rounded`], [`Narrowing::RawPacker`], [`Widening::SrcRaw`]) so
//! the probe can record what they do.
use std::sync::Arc;

use tt_device::Transport;
use tt_isa::{
    backend::{self, Before, ConfigWords},
    cfg::generated::{global, thcon},
    dm::{op, TILE_DATA},
    dram::DramRange,
    isa::Instruction,
    sync::{self, Unit},
};

use crate::{
    bf16::Bf16Tensor,
    datapath,
    sfpu::{kernel, Cond, Format, LReg, Program},
    tensor::{
        DramAlloc, DramTensor, Elem, Job, Pad, Placement, Result, Step, TensorError, TransferBatch,
        Work,
    },
};

/// 1024 two-byte datums, a 16-byte header and 64-byte slot alignment.
pub const TILE_SLOT: u64 = tt_isa::dm::BF16_TILE_SLOT;

/// The unpacker/packer code for FP16 (`L1Format::Fp16`, measured by
/// `probe_src::survey_src_out_formats`).
pub const FP16_CODE: u32 = 1;

/// A resident IEEE binary16 matrix with BF16's physical layout. Intentionally
/// distinct from `Bf16Tensor` so a value cannot be read in the wrong format;
/// the raw movement operations reinterpret it as two-byte slots, which is all
/// they look at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fp16Tensor {
    pub rows: usize,
    pub cols: usize,
    pub(crate) placement: Placement,
}

impl Fp16Tensor {
    /// The same slots as raw two-byte storage, for the format-blind movers.
    pub fn as_raw(&self) -> Bf16Tensor {
        Bf16Tensor {
            rows: self.rows,
            cols: self.cols,
            placement: self.placement.clone(),
        }
    }
    /// Raw two-byte storage whose elements are FP16.
    pub fn from_raw(raw: Bf16Tensor) -> Self {
        Self {
            rows: raw.rows,
            cols: raw.cols,
            placement: raw.placement,
        }
    }
    pub fn rows_view(&self, first_row: usize, rows: usize) -> Result<Self> {
        Ok(Self::from_raw(self.as_raw().rows_view(first_row, rows)?))
    }
    pub fn tile_count(&self) -> usize {
        self.placement.tiles()
    }
    /// Physical slot range, useful for inspecting actual format bytes.
    pub fn slot(&self, tile: usize) -> DramRange {
        self.placement.slot(tile)
    }
}

/// How F32 becomes FP16.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Narrowing {
    /// Exact binary16 by SFPU integer arithmetic and a halfword compaction.
    /// The shipped path.
    Integer,
    /// SFPU ties-even rounding to the FP16 grid ([`quantize`]), then the
    /// packer's FP16 encoding of a value it should represent exactly.
    /// Measurement only.
    Rounded,
    /// The packer's own FP32-to-FP16 conversion, no SFPU step. Measurement only.
    RawPacker,
}

/// How FP16 becomes F32.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Widening {
    /// Exact binary16 by halfword expansion and SFPU integer arithmetic.
    /// The shipped path.
    Integer,
    /// The unpacker's FP16 read into `SrcA` and `MOVA2D` to `Dst`, unmodified.
    /// Measurement only: ttsim moves the 5-bit exponent unconverted.
    SrcRaw,
}

use crate::bf16::transfer;
/// FP32 bits (in `Dst` rows, read as `Int32`) to the FP32 value whose FP16
/// encoding is the binary16 rounding of the input, ties to even; a measurement
/// variant, see [`Narrowing::Rounded`]:
///
/// | input `a = |x|` | result |
/// |---|---|
/// | NaN | `1.5 * 2^16` (FP16 `0x7e00`) |
/// | `a >= 65520` (rounds to 2^16) or infinity | `2^16` (FP16 `0x7c00`) |
/// | `a < 2^-14 - 2^-25` | zero (the binary16 result is a subnormal or zero) |
/// | `2^-14 - 2^-25 <= a < 2^-14` | `2^-14` (rounds up to the smallest normal) |
/// | otherwise | `a` rounded to 11 significant bits, ties even |
///
/// The sign is kept, including on zero.
pub fn quantize() -> Vec<Instruction> {
    let mut p = Program::new();
    p.for_each_row_group(64, |p, row| {
        p.load(LReg::L0, Format::Int32, row);
        p.loadi_bits(LReg::L3, 0x7fffffff);
        p.and(LReg::L0, LReg::L3, LReg::L1); // L1 = |x| bits
        p.loadi_bits(LReg::L3, 0x80000000);
        p.and(LReg::L0, LReg::L3, LReg::L2); // L2 = sign
        p.loadi_bits(LReg::L3, 0x2000);
        p.and(LReg::L1, LReg::L3, LReg::L0);
        p.loadi_bits(LReg::L3, (-13i32) as u32);
        p.shr_by(LReg::L3, LReg::L0); // L0 = lsb of the kept mantissa
        p.loadi_bits(LReg::L3, 0xfff);
        p.iadd(LReg::L3, LReg::L0);
        p.iadd(LReg::L1, LReg::L0);
        p.loadi_bits(LReg::L3, 0xffffe000);
        p.and(LReg::L0, LReg::L3, LReg::L0);
        p.loadi_bits(LReg::L3, 0x38800000);
        p.if_(Cond::Less(LReg::L1, LReg::L3), |p| {
            p.loadi_bits(LReg::L0, 0);
            p.loadi_bits(LReg::L3, 0x387fe000);
            p.if_(Cond::LessEq(LReg::L3, LReg::L1), |p| {
                p.loadi_bits(LReg::L0, 0x38800000);
            });
        });
        p.loadi_bits(LReg::L3, 0x477ff000);
        p.if_(Cond::LessEq(LReg::L3, LReg::L1), |p| {
            p.loadi_bits(LReg::L0, 0x47800000);
        });
        p.loadi_bits(LReg::L3, 0x7f800000);
        p.if_(Cond::Less(LReg::L3, LReg::L1), |p| {
            p.loadi_bits(LReg::L0, 0x47c00000);
        });
        p.or(LReg::L0, LReg::L2, LReg::L0);
        p.store(LReg::L0, Format::Int32, row);
    });
    p.finish()
}

/// Exact IEEE binary16 from FP32 bits, ties to even, in the low 16 bits of
/// each datum (the high 16 are zero). Follows the `half` crate's
/// `f32::to_f16` semantics: overflow (`|x| >= 65520`) and infinity give
/// infinity, a NaN keeps its sign and top ten payload bits and is made quiet,
/// subnormal results round on the 2^-24 grid (and `|x| <= 2^-25` gives zero).
pub fn narrow() -> Vec<Instruction> {
    narrow_with(true)
}

/// [`narrow`], or with `ties_even` false the truncating mutant the model
/// gate's negative control uses (every rounding increment dropped).
fn narrow_with(ties_even: bool) -> Vec<Instruction> {
    let mut p = Program::new();
    p.for_each_row_group(64, |p, row| {
        let (l0, l1, l2, l3, l4, l5, l6, l7) = (
            LReg::L0,
            LReg::L1,
            LReg::L2,
            LReg::L3,
            LReg::L4,
            LReg::L5,
            LReg::L6,
            LReg::L7,
        );
        p.load(l0, Format::Int32, row);
        p.loadi_bits(l3, 0x7fffffff);
        p.and(l0, l3, l1); // l1 = |x| bits
        p.loadi_bits(l3, 0x80000000);
        p.and(l0, l3, l2);
        p.loadi_bits(l3, (-16i32) as u32);
        p.shr_by(l3, l2); // l2 = sign in bit 15
                          // Normal range: ((a + 0xfff + lsb) >> 13) - (112 << 10).
        if ties_even {
            p.loadi_bits(l3, 0x2000);
            p.and(l1, l3, l0);
            p.loadi_bits(l3, (-13i32) as u32);
            p.shr_by(l3, l0);
            p.loadi_bits(l3, 0xfff);
            p.iadd(l3, l0);
            p.iadd(l1, l0);
        } else {
            p.mov(l1, l0);
        }
        p.loadi_bits(l3, (-13i32) as u32);
        p.shr_by(l3, l0);
        p.loadi_bits(l3, (-0x1c000i32) as u32);
        p.iadd(l3, l0);
        // Below 2^-14: round on the 2^-24 grid, `n = round(m >> (126 - e))`.
        p.loadi_bits(l3, 0x38800000);
        p.if_(Cond::Less(l1, l3), |p| {
            p.loadi_bits(l0, 0);
            p.loadi_bits(l3, 0x33000000);
            // Above 2^-25; the tie at 2^-25 itself rounds to even, zero.
            p.if_(Cond::Less(l3, l1), |p| {
                p.mov(l1, l4);
                p.loadi_bits(l3, (-23i32) as u32);
                p.shr_by(l3, l4); // e
                p.loadi_bits(l5, 126);
                p.isub_from(l5, l4); // l4 = shift = 126 - e
                p.loadi_bits(l3, 0x7fffff);
                p.and(l1, l3, l0);
                p.loadi_bits(l3, 0x800000);
                p.or(l0, l3, l0); // l0 = m
                p.mov(l4, l5);
                p.loadi_bits(l3, 0);
                p.isub_from(l3, l5); // l5 = -shift
                if ties_even {
                    p.mov(l0, l3);
                    p.shr_by(l5, l3);
                    p.loadi_bits(l6, 1);
                    p.and(l3, l6, l3); // l3 = lsb of n
                    p.iadd_imm(l4, -1, l7);
                    p.shl_by(l7, l6); // l6 = 1 << (shift - 1)
                    p.iadd_imm(l6, -1, l6);
                    p.iadd(l3, l6);
                    p.iadd(l6, l0); // l0 = m + (half - 1) + lsb
                }
                p.shr_by(l5, l0);
            });
        });
        p.loadi_bits(l3, 0x477ff000);
        p.if_(Cond::LessEq(l3, l1), |p| {
            p.loadi_bits(l0, 0x7c00);
        });
        p.loadi_bits(l3, 0x7f800000);
        p.if_(Cond::Less(l3, l1), |p| {
            p.mov(l1, l0);
            p.loadi_bits(l3, (-13i32) as u32);
            p.shr_by(l3, l0);
            p.loadi_bits(l3, 0x3ff);
            p.and(l0, l3, l0);
            p.loadi_bits(l3, 0x7e00);
            p.or(l0, l3, l0);
        });
        p.or(l0, l2, l0);
        p.store(l0, Format::Int32, row);
    });
    p.finish()
}

/// Exact FP32 from IEEE binary16 held in the low 16 bits of each datum (the
/// high bits are ignored): normals rebias the exponent, subnormals are
/// `m * 2^-24` (an exact conversion and power-of-two product), infinity and NaN
/// keep their sign and payload with the quiet bit set on a NaN.
pub fn widen() -> Vec<Instruction> {
    let mut p = Program::new();
    p.for_each_row_group(64, |p, row| {
        let (l0, l1, l2, l3) = (LReg::L0, LReg::L1, LReg::L2, LReg::L3);
        p.load(l0, Format::Int32, row);
        p.loadi_bits(l3, 0xffff);
        p.and(l0, l3, l0);
        p.loadi_bits(l3, 0x7fff);
        p.and(l0, l3, l1); // l1 = magnitude
        p.loadi_bits(l3, 0x8000);
        p.and(l0, l3, l2);
        p.shl(l2, 16, l2); // l2 = sign
        p.shl(l1, 13, l0);
        p.loadi_bits(l3, 0x38000000);
        p.iadd(l3, l0); // normal: rebias 15 -> 127
        p.loadi_bits(l3, 0x7c00);
        p.if_(Cond::LessEq(l3, l1), |p| {
            p.shl(l1, 13, l0);
            p.loadi_bits(l3, 0x70000000);
            p.or(l0, l3, l0); // exponent 255, payload kept
            p.loadi_bits(l3, 0x7c00);
            p.if_(Cond::Less(l3, l1), |p| {
                p.loadi_bits(l3, 0x400000);
                p.or(l0, l3, l0);
            });
        });
        p.loadi_bits(l3, 0x400);
        p.if_(Cond::Less(l1, l3), |p| {
            p.mov(l1, l0);
            p.sm32_to_float(l0, l0);
            p.muli(f32::from_bits(0x33800000), l0); // * 2^-24
        });
        p.or(l0, l2, l0);
        p.store(l0, Format::Int32, row);
    });
    p.finish()
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Mode {
    Narrow(Narrowing),
    Widen(Widening),
}

fn roles(layout: &kernel::Layout, mode: Mode) -> [Vec<Instruction>; 3] {
    let s = layout.sems;
    let mut unpack = datapath::thread_config();
    unpack.extend(datapath::clear_unpacker0_adcs());
    let mut math = crate::matmul::math_prelude();
    if mode == Mode::Widen(Widening::SrcRaw) {
        // Eight 128-datum chunks through SrcA and MOVA2D, as BF16 does.
        unpack = datapath::src_thread_config();
        let mut empty = tt_isa::matrix::Banks::after_reset();
        for chunk in 0..8 {
            unpack.extend(sync::take(s.free, Before::UNPACKER));
            let mut words = ConfigWords::new();
            datapath::unpack_src_config(
                &mut words,
                datapath::Unpacker::SrcA,
                datapath::flat_descriptor(128).with_in_data_format_raw(FP16_CODE),
                layout.a_at + chunk * 256,
                FP16_CODE,
            );
            words
                .set(tt_isa::cfg::generated::alu::ALU_ACC_CTRL_Fp32_enabled, 1)
                .unwrap();
            words
                .set(
                    tt_isa::cfg::generated::alu::ALU_ACC_CTRL_Zero_Flag_disabled_src,
                    1,
                )
                .unwrap();
            unpack.extend(datapath::config_program(&words));
            unpack.push(datapath::set_adc_x(datapath::Unpacker::SrcA, 0, 127));
            let (i, banks) = empty
                .unpack_a(tt_isa::isa::generated::encode::UnpacrRegular::ZERO.multi_context_mode(1))
                .unwrap();
            unpack.push(i);
            unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
            math.extend(sync::take(s.unpacked, Before::MATRIX));
            let (i, banks) = banks
                .mova2d(
                    tt_isa::isa::generated::encode::Mova2D::ZERO
                        .move8_rows(1)
                        .dst_row(chunk as u32 * 8),
                )
                .unwrap();
            math.push(i);
            math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
            let (i, next) = banks.release_a().unwrap();
            empty = next;
            math.push(i);
            math.extend(sync::post_after(
                Unit::Matrix,
                if chunk == 7 { s.computed } else { s.free },
            ));
        }
    } else {
        // An F32 tile of raw words through `Dst`, an SFPU program over it.
        unpack.extend(sync::take(s.free, Before::UNPACKER));
        let mut words = ConfigWords::new();
        datapath::tile_unpack_config(&mut words, layout.a_at);
        unpack.extend(datapath::config_program(&words));
        unpack.extend(datapath::unpack_tile_to_dst(layout.a_at, 0));
        unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
        math.extend(sync::take(s.unpacked, Before::SFPU));
        match mode {
            Mode::Narrow(Narrowing::Integer) => math.extend(narrow()),
            Mode::Narrow(Narrowing::Rounded) => math.extend(quantize()),
            Mode::Narrow(Narrowing::RawPacker) => {}
            Mode::Widen(_) => math.extend(widen()),
        }
        math.extend(sync::post_after(Unit::Sfpu, s.computed));
    }
    let mut pack = vec![datapath::state_id()];
    pack.extend(sync::take(s.computed, Before::PACKER));
    pack.push(backend::wait_for_packer(Before::CONFIG).unwrap());
    let mut words = ConfigWords::new();
    datapath::pack_config(&mut words, layout.out_at + TILE_DATA);
    if matches!(
        mode,
        Mode::Narrow(Narrowing::Rounded | Narrowing::RawPacker)
    ) {
        words
            .set(thcon::THCON_SEC0_REG1_Out_data_format, FP16_CODE)
            .unwrap();
    }
    words
        .set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, 0)
        .unwrap();
    pack.extend(datapath::config_program(&words));
    pack.extend(datapath::pack_rows(64));
    pack.extend(sync::post_after(Unit::Packer, s.free));
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    [unpack, math, pack]
}

fn jobs(input: &Placement, output: &Placement, mode: Mode) -> Result<Vec<Job>> {
    let layout = kernel::plan_layout(1, kernel::Operands::Unary)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let roles = Arc::new(roles(&layout, mode));
    let (a, out) = (layout.a_at, layout.out_at);
    let halfwords = |src: u64, dst: u64, src_stride: u32, dst_stride: u32| {
        [
            op::COPY_HALFWORDS,
            (src + TILE_DATA) as u32,
            (dst + TILE_DATA) as u32,
            1024,
            src_stride,
            dst_stride,
            0,
            0,
        ]
    };
    Ok((0..input.tiles())
        .map(|t| {
            let (gather, scatter) = match mode {
                // Stage one: raw words (a halfword in each datum's low half) to
                // an F32 staging tensor, as any F32 kernel writes its result.
                Mode::Narrow(Narrowing::Integer) => (
                    vec![transfer(
                        input.slot(t),
                        true,
                        a,
                        tt_isa::dm::TILE_SLOT as u32,
                    )],
                    vec![transfer(output.slot(t), false, out, 4096)],
                ),
                Mode::Narrow(_) => (
                    vec![transfer(
                        input.slot(t),
                        true,
                        a,
                        tt_isa::dm::TILE_SLOT as u32,
                    )],
                    vec![transfer(output.slot(t), false, out, 2048)],
                ),
                Mode::Widen(Widening::Integer) => (
                    // The slot is staged in the (still unused) output region
                    // and widened into the input tile.
                    vec![
                        transfer(input.slot(t), true, out, TILE_SLOT as u32),
                        halfwords(out, a, 2, 4),
                    ],
                    vec![transfer(output.slot(t), false, out, 4096)],
                ),
                Mode::Widen(Widening::SrcRaw) => (
                    vec![transfer(input.slot(t), true, a, TILE_SLOT as u32)],
                    vec![transfer(output.slot(t), false, out, 4096)],
                ),
            };
            vec![
                Step::List {
                    what: "FP16 gather",
                    entries: gather,
                },
                Step::Kernel {
                    roles: roles.clone(),
                    init: layout.init.clone(),
                    mop: Box::new([None; 3]),
                    loops: Default::default(),
                    half: None,
                },
                Step::List {
                    what: "FP16 scatter",
                    entries: scatter,
                },
            ]
        })
        .collect())
}

/// Stage two of integer narrowing: B reads each staged F32 tile of raw words,
/// compacts the low halfwords into a BF16-layout slot, and NC writes it. The
/// copy is B-only, so it cannot follow the kernel in a scatter list; it is a
/// standalone transfer, as the bit repacks are.
fn compact_jobs(staged: &Placement, output: &Placement) -> Result<Vec<Job>> {
    let mut req = crate::l1::Requirements::new(1);
    let src = req.scratch("FP16 staged words", tt_isa::dm::TILE_SLOT, 64, 0..1);
    let dst = req.scratch("FP16 compacted", TILE_SLOT, 64, 0..1);
    let layout = req
        .plan(tt_isa::l1::DATA)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let (src, dst) = (layout.addr(src), layout.addr(dst));
    Ok((0..staged.tiles())
        .map(|t| {
            vec![Step::Transfer {
                what: "FP16 compact",
                depth: 1,
                batches: vec![TransferBatch {
                    read: vec![
                        transfer(staged.slot(t), true, src, tt_isa::dm::TILE_SLOT as u32),
                        [
                            op::COPY_HALFWORDS,
                            (src + TILE_DATA) as u32,
                            (dst + TILE_DATA) as u32,
                            1024,
                            4,
                            2,
                            0,
                            0,
                        ],
                    ],
                    write: vec![transfer(output.slot(t), false, dst, 2048)],
                }],
            }]
        })
        .collect())
}

/// Narrow `input`; the third value is the staging tensor integer narrowing
/// reads in its second stage, which the caller frees once the jobs are queued.
pub(crate) fn compress(
    alloc: &mut DramAlloc,
    input: &DramTensor,
    narrowing: Narrowing,
) -> Result<(Fp16Tensor, Vec<Job>, Option<DramTensor>)> {
    input.expect("FP16 conversion", Elem::F32)?;
    let output = Fp16Tensor {
        rows: input.rows,
        cols: input.cols,
        placement: alloc.alloc_slots(input.placement.tiles(), TILE_SLOT)?,
    };
    let built = (|| {
        if narrowing != Narrowing::Integer {
            let jobs = jobs(&input.placement, &output.placement, Mode::Narrow(narrowing))?;
            return Ok((jobs, None));
        }
        let staged = DramTensor::alloc(alloc, input.rows, input.cols)?;
        staged.set_pad(Pad::Undefined);
        let built = jobs(&input.placement, &staged.placement, Mode::Narrow(narrowing)).and_then(
            |mut jobs| {
                jobs.extend(compact_jobs(&staged.placement, &output.placement)?);
                Ok(jobs)
            },
        );
        match built {
            Ok(jobs) => Ok((jobs, Some(staged))),
            Err(error) => {
                alloc.free(&staged.placement);
                Err(error)
            }
        }
    })();
    match built {
        Ok((jobs, staged)) => Ok((output, jobs, staged)),
        Err(error) => {
            alloc.free(&output.placement);
            Err(error)
        }
    }
}

pub(crate) fn expand(
    alloc: &mut DramAlloc,
    input: &Fp16Tensor,
    widening: Widening,
) -> Result<Work> {
    let out = DramTensor::alloc(alloc, input.rows, input.cols)?;
    let jobs = match jobs(&input.placement, &out.placement, Mode::Widen(widening)) {
        Ok(jobs) => jobs,
        Err(error) => {
            alloc.free(&out.placement);
            return Err(error);
        }
    };
    out.set_pad(Pad::Undefined);
    Ok(Work { out, jobs })
}

const RESET_BUDGET: u64 = crate::session::RESET_BUDGET;

impl<T: Transport> crate::session::Session<T> {
    /// Narrow resident F32 to physically packed IEEE binary16: ties-even,
    /// overflow and infinity to infinity, NaN to a quiet NaN, exact subnormals.
    pub fn fp16_from_f32(&mut self, input: &DramTensor) -> Result<Fp16Tensor> {
        self.fp16_narrow(input, Narrowing::Integer)
    }

    /// The coprocessor's own FP32-to-FP16 conversions, for the T7 measurement:
    /// `Rounded` (SFPU rounds, packer encodes) or `RawPacker` (packer only).
    pub fn fp16_from_f32_with(
        &mut self,
        input: &DramTensor,
        narrowing: Narrowing,
    ) -> Result<Fp16Tensor> {
        self.fp16_narrow(input, narrowing)
    }

    fn fp16_narrow(&mut self, input: &DramTensor, narrowing: Narrowing) -> Result<Fp16Tensor> {
        let (out, jobs, staged) = compress(self.dram_alloc()?, input, narrowing)?;
        let submitted = self.submit_jobs(jobs, RESET_BUDGET);
        if let Some(staged) = staged {
            // Freed after the jobs that read it; the scheduler defers the free.
            let _ = self.free(staged);
        }
        if let Err(error) = submitted {
            self.dram_alloc()?.free(&out.placement);
            return Err(error);
        }
        Ok(out)
    }

    /// Widen packed binary16 to F32 exactly.
    pub fn fp16_to_f32(&mut self, input: &Fp16Tensor) -> Result<DramTensor> {
        self.fp16_to_f32_with(input, Widening::Integer)
    }

    /// Widen through the chosen route; `SrcRaw` is the unpacker's reading, for
    /// the T7 measurement.
    pub fn fp16_to_f32_with(
        &mut self,
        input: &Fp16Tensor,
        widening: Widening,
    ) -> Result<DramTensor> {
        let Work { out, jobs } = expand(self.dram_alloc()?, input, widening)?;
        if let Err(error) = self.submit_jobs(jobs, RESET_BUDGET) {
            self.dram_alloc()?.free(&out.placement);
            return Err(error);
        }
        Ok(out)
    }

    pub fn free_fp16(&mut self, tensor: Fp16Tensor) -> Result<()> {
        self.free_bf16(tensor.as_raw())
    }

    /// Upload binary16 storage bits without converting their values.
    pub fn upload_fp16(&mut self, bits: &[u16], rows: usize, cols: usize) -> Result<Fp16Tensor> {
        Ok(Fp16Tensor::from_raw(self.upload_bf16(bits, rows, cols)?))
    }

    /// Explicit raw readback of binary16 bits. Never used for native arithmetic.
    pub fn download_fp16(&mut self, tensor: &Fp16Tensor) -> Result<Vec<u16>> {
        self.download_bf16(&tensor.as_raw())
    }

    /// Bit-preserving gather of elements into a new matrix.
    pub fn repack_fp16(
        &mut self,
        input: &Fp16Tensor,
        sources: &[[usize; 2]],
        dims: [usize; 2],
    ) -> Result<Fp16Tensor> {
        Ok(Fp16Tensor::from_raw(self.repack_bf16(
            &input.as_raw(),
            sources,
            dims,
        )?))
    }

    /// Positive-zero FP16 storage (`0x0000`).
    pub fn zeros_fp16(&mut self, dims: [usize; 2]) -> Result<Fp16Tensor> {
        Ok(Fp16Tensor::from_raw(self.zeros_bf16(dims)?))
    }

    /// Copy raw FP16 storage bits through Tensix, with physically zero padding.
    pub fn copy_fp16_xmov(&mut self, source: &Fp16Tensor) -> Result<Fp16Tensor> {
        Ok(Fp16Tensor::from_raw(self.copy_bf16_xmov(&source.as_raw())?))
    }
}

/// Independent host definitions of the conversions, written from the IEEE 754
/// binary16 definition (and the `half` crate's NaN rule) with exact integer and
/// `f64` arithmetic, not from the SFPU programs. Public so the gates in
/// `tt-tests` use the same oracle as the model tests.
pub mod host {
    /// IEEE binary16 bits of an FP32, round to nearest, ties to even.
    pub fn f32_to_f16(bits: u32) -> u16 {
        let sign = ((bits >> 16) & 0x8000) as u16;
        let a = bits & 0x7fff_ffff;
        if a > 0x7f80_0000 {
            return sign | 0x7c00 | ((a >> 13) & 0x3ff) as u16 | 0x200;
        }
        if a == 0x7f80_0000 {
            return sign | 0x7c00;
        }
        if a == 0 {
            return sign;
        }
        // value = m * 2^e with m an integer.
        let (m, e): (u128, i32) = if a >> 23 == 0 {
            (a as u128, -149)
        } else {
            (
                ((a & 0x7f_ffff) | 0x80_0000) as u128,
                (a >> 23) as i32 - 150,
            )
        };
        let log2 = 127 - m.leading_zeros() as i32 + e;
        // The binary16 quantum: 2^-24 below 2^-14, else 2^(floor(log2 v) - 10).
        let q = if log2 < -14 { -24 } else { log2 - 10 };
        let shift = (q - e) as u32;
        let whole = m >> shift;
        let rem = m & ((1u128 << shift) - 1);
        let half = 1u128 << (shift - 1);
        let n = whole + u128::from(rem > half || (rem == half && whole & 1 == 1));
        let value = n as f64 * 2f64.powi(q);
        if value >= 65520.0 {
            return sign | 0x7c00;
        }
        if value < 2f64.powi(-14) {
            return sign | n as u16; // subnormal (or zero); 1024 is the smallest normal
        }
        let exp = ((value.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        sign | (((exp + 15) as u16) << 10) | ((value.to_bits() >> 42) & 0x3ff) as u16
    }

    /// FP32 bits of an IEEE binary16 (NaN keeps sign and payload, made quiet).
    pub fn f16_to_f32(h: u16) -> u32 {
        let sign = (h as u32 & 0x8000) << 16;
        let e = (h >> 10) & 0x1f;
        let m = h as u32 & 0x3ff;
        match e {
            // m * 2^-24 is exact in FP32 (a normal number, or zero).
            0 => sign | (m as f32 * f32::from_bits(0x3380_0000)).to_bits(),
            31 if m == 0 => sign | 0x7f80_0000,
            31 => sign | 0x7fc0_0000 | (m << 13),
            _ => sign | (((e as u32) + 112) << 23) | (m << 13),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::host::{f16_to_f32, f32_to_f16};
    use super::*;

    fn run(program: &[Instruction], values: &[u32]) -> Vec<u32> {
        let mut model = crate::sfpu::interp::Vector::new();
        model.put_tile(0, values);
        model.run(program).unwrap();
        model.tile(0)
    }

    /// Every input class the narrowing distinguishes: ties on both sides of
    /// even, carries into the exponent, the 65520 overflow tie, the subnormal
    /// boundaries and ties, signed zeros, infinities and NaN payloads, then
    /// random words half raw and half inside the binary16 exponent range.
    pub(crate) fn corpus() -> Vec<u32> {
        let mut v = vec![
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0xff80_0001,
            0x7fc1_2345,
            0xffc5_4321,
            0x7fff_ffff,
            0x7f7f_ffff,
            0xff7f_ffff,
            // 1 + 2^-11 (tie, even is down), 1 + 3*2^-11 (tie, even is up).
            0x3f80_1000,
            0x3f80_3000,
            0x3f80_0fff,
            0x3f80_1001,
            // 65504, 65519.99, 65520 (ties to infinity), 65520+ulp, 65536, 70000.
            0x477f_e000,
            0x477f_efff,
            0x477f_f000,
            0x477f_f001,
            0x4780_0000,
            0x4788_b800,
            // 2^-14, 2^-14 - 2^-25 (tie up to 2^-14), just below it, 2^-15,
            // 2^-24, 2^-25 (tie to zero), 2^-25 + ulp.
            0x3880_0000,
            0x387f_e000,
            0x387f_dfff,
            0x3800_0000,
            0x3380_0000,
            0x3300_0000,
            0x3300_0001,
            // 1.5 * 2^-24 (tie, even is 2), 2.5 * 2^-24 (tie, even is 2).
            0x33c0_0000,
            0x3420_0000,
        ];
        let mut s = 0x1234_5678u64;
        while v.len() < 1024 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let w = (s >> 32) as u32;
            v.push(if v.len() % 2 == 0 {
                w
            } else {
                (w & 0x807f_ffff) | ((100 + w % 50) << 23)
            });
        }
        v
    }

    #[test]
    fn narrow_matches_independent_binary16_conversion() {
        let values = corpus();
        for (i, (got, &x)) in run(&narrow(), &values).into_iter().zip(&values).enumerate() {
            assert_eq!(got, f32_to_f16(x) as u32, "{i}: input {x:#010x}");
        }
    }

    #[test]
    fn narrow_spot_checks() {
        let r = |x: u32| run(&narrow(), &vec![x; 1024])[0];
        assert_eq!(r(0x3f80_1000), 0x3c00, "tie to even, down");
        assert_eq!(r(0x3f80_3000), 0x3c02, "tie to even, up");
        assert_eq!(r(0x477f_efff), 0x7bff);
        assert_eq!(r(0x477f_f000), 0x7c00, "65520 ties to infinity");
        assert_eq!(r(0xff80_0000), 0xfc00);
        assert_eq!(r(0x7fc1_2345), 0x7e00 | ((0x7fc1_2345 >> 13) & 0x3ff));
        assert_eq!(r(0x3880_0000), 0x0400);
        assert_eq!(r(0x387f_e000), 0x0400, "rounds up to the smallest normal");
        assert_eq!(r(0x3380_0000), 0x0001, "2^-24 is the smallest subnormal");
        assert_eq!(r(0x3300_0000), 0, "2^-25 ties to zero");
        assert_eq!(r(0x3300_0001), 1);
        assert_eq!(r(0x33c0_0000), 2, "1.5 * 2^-24 ties to even");
        assert_eq!(r(0x8000_0001), 0x8000);
    }

    /// Negative control, against the executable program: the truncating mutant
    /// must disagree with the independent reference on the tie cases (and not
    /// where nothing rounds up), and the full gate above must reject it.
    #[test]
    fn truncating_narrowing_fails_the_tie_cases() {
        let values = corpus();
        let got = run(&narrow_with(false), &values);
        for t in [0x3f80_3000u32, 0x3f80_1001, 0x33c0_0000, 0x3300_0001] {
            let at = values.iter().position(|&x| x == t).unwrap();
            assert_ne!(got[at], f32_to_f16(t) as u32, "{t:#010x} must differ");
        }
        for t in [0x3f80_0fffu32, 0x3f80_1000] {
            let at = values.iter().position(|&x| x == t).unwrap();
            assert_eq!(
                got[at],
                f32_to_f16(t) as u32,
                "{t:#010x} rounds down anyway"
            );
        }
        let wrong = got
            .iter()
            .zip(&values)
            .filter(|(g, &x)| **g != f32_to_f16(x) as u32)
            .count();
        assert!(wrong > values.len() / 8, "{wrong}");
    }

    #[test]
    fn widen_matches_independent_binary16_reading_for_every_pattern() {
        for base in (0..65536u32).step_by(1024) {
            let values: Vec<u32> = (0..1024)
                .map(|i| (base + i) | if i % 3 == 0 { 0xdead_0000 } else { 0 })
                .collect();
            for (i, got) in run(&widen(), &values).into_iter().enumerate() {
                let h = (values[i] & 0xffff) as u16;
                assert_eq!(got, f16_to_f32(h), "pattern {h:#06x}");
            }
        }
    }

    #[test]
    fn host_conversions_round_trip_every_binary16_pattern() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            let back = f32_to_f16(f);
            if (h >> 10) & 0x1f == 31 && h & 0x3ff != 0 {
                assert_eq!(back, h | 0x200, "NaN {h:#06x} comes back quiet");
            } else {
                assert_eq!(back, h, "{h:#06x} via {f:#010x}");
            }
        }
    }

    /// The measurement variant's model: ties-even to the FP16 grid, specials
    /// mapped to the exponent-31 row, subnormal results flushed.
    #[test]
    fn quantize_agrees_with_narrow_where_the_grid_is_ieee() {
        let values = corpus();
        let q = run(&quantize(), &values);
        for (i, (&got, &x)) in q.iter().zip(&values).enumerate() {
            let h = f32_to_f16(x);
            let expected = match (h >> 10) & 0x1f {
                0 if h & 0x3ff != 0 => (h & 0x8000) as u32 * 0x10000, // flushed subnormal
                31 if h & 0x3ff == 0 => ((h as u32 & 0x8000) << 16) | 0x4780_0000,
                31 => ((h as u32 & 0x8000) << 16) | 0x47c0_0000,
                _ => f16_to_f32(h),
            };
            // Subnormals that round up to 2^-14 are already normal in `h`.
            assert_eq!(got, expected, "{i}: input {x:#010x}");
        }
    }
}
