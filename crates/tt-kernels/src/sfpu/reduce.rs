//! Reductions over one dimension on the SFPU (`hardware-coverage.md` R1).
//!
//! A `[rows, cols]` tensor reduced over its columns (`Axis::Cols`, Burn's
//! `dim = 1`) gives `[rows, 1]`; over its rows (`Axis::Rows`, `dim = 0`),
//! `[1, cols]`. Each output tile is made from a line of input tiles -- a tile
//! row, or a tile column -- in three steps on the math thread:
//!
//! 1. **accumulate**: each input tile, unpacked to `Dst` rows 0..64, is
//!    combined lanewise into an accumulator tile at rows 128..192 (the first
//!    copied there). The last tile along the reduced dimension of a ragged
//!    tensor has its padding lanes replaced by the operation's identity first
//!    (`-inf` for a maximum, `0` for a sum), by the lane's column (or row)
//!    index from `LReg[15]` -- so padding never needs filling.
//! 2. **finish**: the accumulator folded within the tile. Over columns: each
//!    tile row's 32 values -- two faces, two column halves, eight lanes each --
//!    combined lanewise, then across the eight lanes by seven rotations
//!    (`SFPSHFT2`), and the row's result stored into every one of its
//!    columns: an output tile whose every column is the answer, ready to be
//!    broadcast. Over rows: each column's 32 values -- eight row groups --
//!    combined lanewise, then across the four lane rows by `SFPTRANSP`, the
//!    result in row 0.
//! 3. the packer writes the accumulator out.
//!
//! A maximum is exact whatever the order: the total order of `SFPGT`, in which
//! a positive NaN is above everything (so it propagates, and every NaN the
//! SFPU's arithmetic makes is one) and a negative NaN below everything. A sum
//! over columns is computed in the tree order above, not Flex's left to right:
//! its error is the standard `(n - 1) u sum |x|` bound, stated against Flex in
//! the gate.
//!
//! A **sum over rows** is Flex's exactly, so it is not a tree: each column
//! from `+0.0`, adding its rows one at a time, top to bottom, tile after tile
//! ([`accumulate_in_order`]). A load holds four rows of eight columns; one
//! `SFPTRANSP` puts each of the four rows into row 0 of its own register, and
//! they are added into the running sum in order. The sum lives in row 0 of
//! `L4`, which is on the transpose's diagonal and so survives it. Rows past
//! the data are not added at all -- adding `+0.0` would turn a `-0.0` sum
//! positive -- and there is no finishing fold.

use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::dm::TILE_SLOT;
use tt_isa::isa::Instruction;
use tt_isa::sync::{self, Semaphore, Unit};

use super::kernel::{A_ROW, OUT_ROW};
use super::{Cond, Format, LReg, LoopPolicy, Program};
use crate::datapath::{
    clear_unpacker0_adcs, config_program, pack_tile_from_dst, state_id, thread_config,
    tile_unpack_config, unpack_tile_to_dst,
};
use crate::l1::{PlanError, Requirements};
use crate::runtime::SemaphoreInit;

/// What a reduction computes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    /// Direct multiplication, with one final fold after all input chunks.
    Prod,
    /// Conjunction of canonical Boolean words.
    All,
    /// Disjunction of canonical Boolean words.
    Any,
    /// OR the raw bits of F32 storage, with zero-bit padding. Used after
    /// masking to one selected datum per row, preserving zeros/NaN payloads.
    /// This is a bit fold, not a floating-point arithmetic reduction.
    BitOr,
    SumI32,
    ProdI32,
    MinI32,
    MaxI32,
}

impl ReduceOp {
    /// The identity a padding lane is replaced by.
    fn identity(self) -> u32 {
        match self {
            ReduceOp::Sum => 0,
            ReduceOp::Max => 0xff80_0000,
            ReduceOp::Prod => 0x3f80_0000,
            ReduceOp::All => 1,
            ReduceOp::Any | ReduceOp::BitOr => 0,
            ReduceOp::SumI32 => 0,
            ReduceOp::ProdI32 => 1,
            ReduceOp::MinI32 => i32::MAX as u32,
            ReduceOp::MaxI32 => i32::MIN as u32,
        }
    }

    /// Physical element type consumed and produced by this reduction.
    pub fn elem(self) -> crate::tensor::Elem {
        match self {
            Self::All | Self::Any => crate::tensor::Elem::Bool,
            Self::SumI32 | Self::ProdI32 | Self::MinI32 | Self::MaxI32 => crate::tensor::Elem::I32,
            _ => crate::tensor::Elem::F32,
        }
    }

    pub(crate) fn format(self) -> Format {
        match self {
            Self::Max
            | Self::All
            | Self::Any
            | Self::BitOr
            | Self::SumI32
            | Self::ProdI32
            | Self::MinI32
            | Self::MaxI32 => Format::Int32,
            _ => Format::Fp32,
        }
    }
}

/// Which dimension is reduced.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Over rows (Burn's `dim = 0`): `[rows, cols] -> [1, cols]`.
    Rows,
    /// Over columns (`dim = 1`): `[rows, cols] -> [rows, 1]`.
    Cols,
}

/// `into = op(other, into)`, lanewise.
pub(crate) fn combine(p: &mut Program, op: ReduceOp, other: LReg, into: LReg) {
    match op {
        ReduceOp::Sum => p.add(other, into, into),
        ReduceOp::Max => p.if_(Cond::Less(into, other), |p| p.mov(other, into)),
        ReduceOp::Prod => p.mul(other, into, into),
        ReduceOp::All => p.and(other, into, into),
        ReduceOp::Any | ReduceOp::BitOr => p.or(other, into, into),
        ReduceOp::SumI32 => p.iadd(other, into),
        ReduceOp::ProdI32 => {
            // Preserve L0..L3: finishing a row fold still needs the other
            // three transposed row registers. Reconstruct a full low-32-bit
            // product from 16-bit limbs and the low/upper MUL24 modes.
            p.loadi_bits(LReg::L6, 0xffff);
            p.and(other, LReg::L6, LReg::L4);
            p.and(into, LReg::L6, LReg::L5);
            p.mov(into, LReg::L7);
            p.loadi_bits(LReg::L6, (-16i32) as u32);
            p.shr_by(LReg::L6, LReg::L7);
            p.mul24(LReg::L4, LReg::L7, false, LReg::L7);
            p.shl(LReg::L7, 16, LReg::L7);
            p.mul24(LReg::L4, LReg::L5, false, into);
            p.mul24(LReg::L4, LReg::L5, true, LReg::L6);
            p.shl(LReg::L6, 23, LReg::L6);
            p.or(LReg::L6, into, into);
            p.iadd(LReg::L7, into);
            p.mov(other, LReg::L4);
            p.loadi_bits(LReg::L6, (-16i32) as u32);
            p.shr_by(LReg::L6, LReg::L4);
            p.mul24(LReg::L4, LReg::L5, false, LReg::L7);
            p.shl(LReg::L7, 16, LReg::L7);
            p.iadd(LReg::L7, into);
        }
        ReduceOp::MinI32 | ReduceOp::MaxI32 => {
            p.loadi_bits(LReg::L4, 0x7fffffff);
            p.mov(other, LReg::L5);
            p.mov(into, LReg::L7);
            p.if_(Cond::Lt0(LReg::L5), |p| p.xor(LReg::L4, LReg::L5));
            p.if_(Cond::Lt0(LReg::L7), |p| p.xor(LReg::L4, LReg::L7));
            let cond = if op == ReduceOp::MinI32 {
                Cond::Less(LReg::L5, LReg::L7)
            } else {
                Cond::Less(LReg::L7, LReg::L5)
            };
            p.if_(cond, |p| p.mov(other, into));
        }
    }
}

/// The accumulate program for one input tile: `first` copies it to the
/// accumulator, otherwise it is combined in. `valid`: the input is the last
/// along the reduced dimension and only its first `valid` columns (or rows,
/// for [`Axis::Rows`]) are data -- the rest are replaced by the identity.
/// A sum over rows is [`accumulate_in_order`] instead.
pub fn accumulate(op: ReduceOp, axis: Axis, first: bool, valid: Option<u32>) -> Vec<Instruction> {
    if (op, axis) == (ReduceOp::Sum, Axis::Rows) {
        return accumulate_in_order(first, valid.unwrap_or(32));
    }
    accumulate_with(op, axis, first, valid, false, 32)
}

/// [`accumulate`] of each datum's square (`x * x` as `MUL`'s program forms it, `SFPMAD(x, x,
/// -0)`) for a sum over columns: what accumulating `MUL(x, x)`'s output tile does, bit for bit,
/// since a product is never a denormal and so its FP32 store and reload change nothing. The sum
/// of squares of `sfpu::norm`, over the tile's first `rows` rows only ([`row_offsets`]).
pub fn accumulate_squares(first: bool, valid: Option<u32>, rows: u32) -> Vec<Instruction> {
    accumulate_with(ReduceOp::Sum, Axis::Cols, first, valid, true, rows)
}

/// The `Dst` offsets, from a tile's first row and as `for_each_row_group(64, ..)` hands them,
/// of the row groups that hold the tile's first `rows` (1..=32) tensor rows. Tensor rows 0..16
/// are faces 0 (offsets 0..16) and 1 (16..32), rows 16..32 faces 2 and 3; each face is four
/// groups of four rows, a group's even columns at `4g` and its odd at `4g | 2`. A load holds
/// four rows of eight columns, and a reduction over columns never moves a value between rows
/// (its fold rotates within a lane row), so the groups past `rows` -- padding -- need not run:
/// a decode row (`rows == 1`) is 4 of the 32.
pub fn row_offsets(rows: u32) -> Vec<u32> {
    assert!((1..=32).contains(&rows), "{rows} rows of a tile");
    (0..64)
        .step_by(2)
        .filter(|o| o / 32 * 16 + o % 16 / 4 * 4 < rows)
        .collect()
}

fn accumulate_with(
    op: ReduceOp,
    axis: Axis,
    first: bool,
    valid: Option<u32>,
    square: bool,
    rows: u32,
) -> Vec<Instruction> {
    assert!(
        rows == 32 || axis == Axis::Cols,
        "rows limit a sum over columns"
    );
    let fmt = op.format();
    let masked = valid.is_some_and(|v| v < 32);
    let mut p = Program::with_policy(if masked || rows < 32 {
        LoopPolicy::Unrolled
    } else {
        LoopPolicy::Replay
    });
    let mask_setup = |p: &mut Program| {
        // `L6` = this lane's column within its half (`2 * (lane & 7)`), or
        // its row within its group (`lane >> 3`), from `LReg[15] = 2 * lane`.
        match axis {
            Axis::Cols => {
                p.loadi_bits(LReg::L7, 14);
                p.and(LReg::LANE_X2, LReg::L7, LReg::L6);
            }
            Axis::Rows => {
                p.loadi_bits(LReg::L7, (-4i32) as u32);
                p.mov(LReg::LANE_X2, LReg::L6);
                p.shr_by(LReg::L7, LReg::L6);
            }
        }
        p.loadi_bits(LReg::L4, op.identity());
    };
    if masked {
        mask_setup(&mut p);
    }
    let v = valid.unwrap_or(32) as i32;
    let body = |p: &mut Program, o: u32| {
        if masked && matches!(op, ReduceOp::ProdI32 | ReduceOp::MinI32 | ReduceOp::MaxI32) {
            mask_setup(p);
        }
        p.load(LReg::L0, fmt, A_ROW + o);
        if square {
            p.mul(LReg::L0, LReg::L0, LReg::L0);
        }
        if masked {
            let g = o / 4;
            // What the lane index in `L6` must stay below for the lane to
            // hold data in this row group and half.
            let t = match axis {
                Axis::Cols => v - (o & 2 != 0) as i32 - 16 * ((g / 4) % 2) as i32,
                Axis::Rows => v - 4 * (g % 4) as i32 - 16 * (g / 8) as i32,
            };
            if t <= 0 {
                p.mov(LReg::L4, LReg::L0);
            } else if t < 16 {
                p.loadi_bits(LReg::L5, t as u32);
                p.if_else(
                    Cond::Less(LReg::L6, LReg::L5),
                    |_| {},
                    |p| p.mov(LReg::L4, LReg::L0),
                );
            }
        }
        if first {
            p.store(LReg::L0, fmt, OUT_ROW + o);
        } else {
            p.load(LReg::L1, fmt, OUT_ROW + o);
            combine(p, op, LReg::L0, LReg::L1);
            p.store(LReg::L1, fmt, OUT_ROW + o);
        }
    };
    if rows == 32 {
        p.for_each_row_group(64, body);
    } else {
        for o in row_offsets(rows) {
            body(&mut p, o);
        }
    }
    p.finish()
}

/// The column sums of one input tile's first `valid` rows, added in order into
/// the running sums in row 0 of the accumulator tile -- from `+0.0` for the
/// `first` tile, which also zeroes the accumulator's other rows (as the
/// result's padding). Flex's `sum` over `dim = 0`, bit for bit.
pub fn accumulate_in_order(first: bool, valid: u32) -> Vec<Instruction> {
    in_order(first, valid, false)
}

/// The column sums of squares of one input tile's first `valid` rows, in order, into row 0 of
/// the accumulator tile: what [`accumulate_squares`] adds a lane at a time over a row's tiles,
/// for that row laid out one of its tiles' data rows per row (`tensor::pack_row`: row `kk` is
/// the row's tile `kk`). Column `c` then takes the same squares in the same order as lane `c`
/// of the row's sum -- the first as it is (not added to `+0.0`), each later one added as
/// `square + sum` -- and [`finish_rows`]`(.., 1)` folds row 0 as it folds the row's.
pub fn accumulate_squares_in_order(first: bool, valid: u32) -> Vec<Instruction> {
    in_order(first, valid, true)
}

fn in_order(first: bool, valid: u32, square: bool) -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    let acc = |r: u32| OUT_ROW + r;
    if first {
        p.for_each_row_group(64, |p, o| p.store(LReg::ZERO, Format::Fp32, acc(o)));
    }
    // Columns 0..16 are faces 0 (rows 0..16) and 2 (16..32), columns 16..32
    // faces 1 and 3; each column half of a face is eight lanes, a load four
    // rows of them.
    for (top, bottom) in [(0, 32), (16, 48)] {
        for half in [0, 2] {
            if first {
                p.mov(LReg::ZERO, LReg::L4);
            } else {
                p.load(LReg::L4, Format::Fp32, acc(top + half));
            }
            // The first square is the sum, as `accumulate_squares` stores the first tile's.
            let mut fresh = first && square;
            for g in 0..8u32 {
                let rows = valid.saturating_sub(4 * g).min(4);
                if rows == 0 {
                    break;
                }
                let r = if g < 4 {
                    top + 4 * g
                } else {
                    bottom + 4 * (g - 4)
                };
                p.load(LReg::L0, Format::Fp32, A_ROW + r + half);
                if square {
                    p.mul(LReg::L0, LReg::L0, LReg::L0);
                }
                // Row `i` of the four into row 0 of `L[i]`; row 0 of `L4`
                // stays where it is.
                p.transpose4();
                for i in 0..rows {
                    let row = [LReg::L0, LReg::L1, LReg::L2, LReg::L3][i as usize];
                    if fresh {
                        p.mov(row, LReg::L4);
                        fresh = false;
                    } else if square {
                        // `combine`'s order: the new value, then the sum.
                        p.add(row, LReg::L4, LReg::L4);
                    } else {
                        p.add(LReg::L4, row, LReg::L4);
                    }
                }
            }
            // Row 0 the sums, rows 1..4 zero: `L5..L8` cleared and transposed
            // into them.
            for l in [LReg::L5, LReg::L6, LReg::L7] {
                p.mov(LReg::ZERO, l);
            }
            p.transpose4();
            p.store(LReg::L4, Format::Fp32, acc(top + half));
        }
    }
    p.finish()
}

/// The finishing program: the accumulator folded within the tile (see the
/// module documentation). Nothing for a sum over rows, already in row 0.
pub fn finish(op: ReduceOp, axis: Axis) -> Vec<Instruction> {
    finish_rows(op, axis, 32)
}

/// [`finish`] of a sum over columns folding only the row groups of the tile's first `rows`
/// rows ([`row_offsets`]); the others are left as they were, padding.
pub fn finish_rows(op: ReduceOp, axis: Axis, rows: u32) -> Vec<Instruction> {
    assert!(
        rows == 32 || axis == Axis::Cols,
        "rows limit a sum over columns"
    );
    let fmt = op.format();
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    if (op, axis) == (ReduceOp::Sum, Axis::Rows) {
        return p.finish();
    }
    let acc = |r: u32| OUT_ROW + r;
    match axis {
        Axis::Cols => {
            // Tensor rows 0..16 are in faces 0 (left) and 1 (right), 16..32 in
            // faces 2 and 3; a group of four rows in each.
            for lower in [0, 32] {
                for g in 0..4 {
                    if lower / 2 + 4 * g >= rows {
                        continue;
                    }
                    let (left, right) = (lower + 4 * g, lower + 16 + 4 * g);
                    let at = [left, left | 2, right, right | 2];
                    p.load(LReg::L0, fmt, acc(at[0]));
                    for &a in &at[1..] {
                        p.load(LReg::L1, fmt, acc(a));
                        combine(&mut p, op, LReg::L1, LReg::L0);
                    }
                    // Across the eight lanes of each lane row.
                    p.mov(LReg::L0, LReg::L2);
                    for _ in 0..7 {
                        p.rotate_row(LReg::L2, LReg::L2);
                        combine(&mut p, op, LReg::L2, LReg::L0);
                    }
                    for &a in &at {
                        p.store(LReg::L0, fmt, acc(a));
                    }
                }
            }
        }
        Axis::Rows => {
            for (top, bottom) in [(0, 32), (16, 48)] {
                for half in [0, 2] {
                    p.load(LReg::L0, fmt, acc(top + half));
                    for g in 0..8 {
                        let r = if g < 4 {
                            top + 4 * g
                        } else {
                            bottom + 4 * (g - 4)
                        };
                        if r != top {
                            p.load(LReg::L1, fmt, acc(r + half));
                            combine(&mut p, op, LReg::L1, LReg::L0);
                        }
                    }
                    // Lane row `i` of `L0` into lane row 0 of `L[i]`.
                    p.transpose4();
                    combine(&mut p, op, LReg::L1, LReg::L0);
                    combine(&mut p, op, LReg::L3, LReg::L2);
                    combine(&mut p, op, LReg::L2, LReg::L0);
                    p.store(LReg::L0, fmt, acc(top + half));
                }
            }
        }
    }
    p.finish()
}

/// The reduce kernel's semaphores, numbered as a matmul's and an element-wise
/// kernel's are on the ones they share (zero, one, zero), so all three
/// alternate with no setup run.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReduceSemaphores {
    /// T0 -> T1: an input tile is in `Dst`.
    pub unpacked: Semaphore,
    /// T1 -> T0: the math has read it; the next may be unpacked over it.
    pub consumed: Semaphore,
    /// T1 -> T2: an output tile is finished.
    pub computed: Semaphore,
    /// T2 -> T1: the packer has read the accumulator.
    pub packed: Semaphore,
}

/// Where a run of `outputs` output tiles, each from `per` input tiles, lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub outputs: usize,
    pub per: usize,
    pub in_at: u64,
    pub out_at: u64,
    pub sems: ReduceSemaphores,
    pub init: Vec<SemaphoreInit>,
}

impl Layout {
    /// This layout with its slots `by` bytes on, its semaphores where they
    /// were: the second half of a double-buffered pair.
    pub fn shifted(&self, by: u64) -> Layout {
        Layout {
            in_at: self.in_at + by,
            out_at: self.out_at + by,
            ..self.clone()
        }
    }
}

pub fn plan_layout(outputs: usize, per: usize) -> Result<Layout, PlanError> {
    plan_layout_in(outputs, per, crate::l1::arena())
}

/// [`plan_layout`] in `arena`: half the data arena, for a run double-buffered
/// with the next (`crate::matmul::half_arena`, then [`Layout::shifted`]).
pub fn plan_layout_in(
    outputs: usize,
    per: usize,
    arena: tt_isa::l1::Region,
) -> Result<Layout, PlanError> {
    let mut req = Requirements::new(1);
    let align = tt_isa::dram::ALIGN;
    let input = req.scratch(
        "reduce inputs",
        (outputs * per) as u64 * TILE_SLOT,
        align,
        0..1,
    );
    let out = req.scratch("reduce outputs", outputs as u64 * TILE_SLOT, align, 0..1);
    let unpacked = req.semaphore("reduce unpacked", 0, 0..1);
    let consumed = req.semaphore("reduce consumed", 1, 0..1);
    let computed = req.semaphore("reduce computed", 0, 0..1);
    let packed = req.semaphore("reduce packed", 1, 0..1);
    let plan = req.plan(arena)?;
    Ok(Layout {
        outputs,
        per,
        in_at: plan.addr(input),
        out_at: plan.addr(out),
        sems: ReduceSemaphores {
            unpacked: plan.semaphore(unpacked),
            consumed: plan.semaphore(consumed),
            computed: plan.semaphore(computed),
            packed: plan.semaphore(packed),
        },
        init: plan.semaphore_init(),
    })
}

/// The math programs of one output tile: per input, in order, then the
/// finishing one.
pub fn math_programs(
    op: ReduceOp,
    axis: Axis,
    per: usize,
    last_valid: u32,
) -> (Vec<Vec<Instruction>>, Vec<Instruction>) {
    let inputs = (0..per)
        .map(|n| {
            let valid = (n + 1 == per && last_valid < 32).then_some(last_valid);
            accumulate(op, axis, n == 0, valid)
        })
        .collect();
    (inputs, finish(op, axis))
}

/// [`math_programs`] of a reduction over columns computing only the row groups of each tile's
/// first `rows` rows ([`row_offsets`]): a decode head's score row is 4 of the 32. A reduction
/// over columns never moves a value between rows, so those rows keep their bits; the others
/// are left as they were, padding.
pub fn math_programs_rows(
    op: ReduceOp,
    per: usize,
    last_valid: u32,
    rows: u32,
) -> (Vec<Vec<Instruction>>, Vec<Instruction>) {
    let inputs = (0..per)
        .map(|n| {
            let valid = (n + 1 == per && last_valid < 32).then_some(last_valid);
            accumulate_with(op, Axis::Cols, n == 0, valid, false, rows)
        })
        .collect();
    (inputs, finish_rows(op, Axis::Cols, rows))
}

/// The three role programs of a run over `layout`.
pub fn roles(
    layout: &Layout,
    inputs: &[Vec<Instruction>],
    finish: &[Instruction],
) -> [Vec<Instruction>; 3] {
    assert_eq!(inputs.len(), layout.per);
    roles_at(layout, inputs, finish, |k, n| {
        layout.in_at + (k * layout.per + n) as u64 * TILE_SLOT
    })
}

/// [`roles`], with output `k`'s input `n` unpacked from `input(k, n)`.
pub fn roles_at(
    layout: &Layout,
    inputs: &[Vec<Instruction>],
    finish: &[Instruction],
    input: impl Fn(usize, usize) -> u64,
) -> [Vec<Instruction>; 3] {
    let s = layout.sems;
    let slot = |base: u64, n: usize| base + n as u64 * TILE_SLOT;
    let mut unpack = thread_config();
    unpack.extend(clear_unpacker0_adcs());
    let mut words = ConfigWords::new();
    tile_unpack_config(&mut words, layout.in_at);
    unpack.extend(config_program(&words));
    let mut math = vec![state_id()];
    let mut pack = vec![state_id()];
    for k in 0..layout.outputs {
        for (n, program) in inputs.iter().enumerate() {
            unpack.extend(sync::take(s.consumed, Before::UNPACKER));
            unpack.extend(unpack_tile_to_dst(input(k, n), A_ROW));
            unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));

            math.extend(sync::take(s.unpacked, Before::SFPU));
            if n == 0 {
                math.extend(sync::take(s.packed, Before::SFPU));
            }
            math.extend_from_slice(program);
            math.extend(sync::post_after(Unit::Sfpu, s.consumed));
        }
        math.extend_from_slice(finish);
        math.extend(sync::post_after(Unit::Sfpu, s.computed));

        pack.extend(sync::take(s.computed, Before::PACKER));
        pack.extend(pack_tile_from_dst(
            slot(layout.out_at, k) + tt_isa::dm::TILE_DATA,
            OUT_ROW,
        ));
        pack.extend(sync::post_after(Unit::Packer, s.packed));
    }
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    [unpack, math, pack]
}

/// [`roles`] as the role slots hold them, with each role's block repeats
/// ([`roles_code_at`]).
pub fn roles_code(
    layout: &Layout,
    inputs: &[Vec<Instruction>],
    finish: &[Instruction],
) -> ([Vec<Instruction>; 3], [Vec<crate::code::Loop>; 3]) {
    assert_eq!(inputs.len(), layout.per);
    roles_code_at(layout, inputs, finish, |k, n| {
        layout.in_at + (k * layout.per + n) as u64 * TILE_SLOT
    })
}

/// [`roles_at`], with an output's middle inputs -- the same math program,
/// read from consecutive slots -- one block repeated, and the outputs one
/// block repeated around it, their addresses stepped in GPRs
/// (`docs/plans/long-context.md` 1.A): a program's size does not grow with
/// the inputs an output reduces, or the outputs. The unpacker and packer see
/// the same addresses, in the same order, as [`roles_at`] gives them, and the
/// math runs the same programs. [`roles_at`]'s programs where the inputs do not
/// have that shape.
pub fn roles_code_at(
    layout: &Layout,
    inputs: &[Vec<Instruction>],
    finish: &[Instruction],
    input: impl Fn(usize, usize) -> u64,
) -> ([Vec<Instruction>; 3], [Vec<crate::code::Loop>; 3]) {
    roles_looped(layout, inputs, finish, &input)
        .unwrap_or_else(|| (roles_at(layout, inputs, finish, input), Default::default()))
}

/// GPRs [`roles_looped`] keeps its addresses in: clear of the ones
/// `sfpu::kernel`'s notes name, as its own are.
const FIRST_GPR: u32 = 32;
const FIRST_STRIDE_GPR: u32 = 33;
const MID_GPR: u32 = 34;
const GAP_GPR: u32 = 35;
const SLOT_GPR: u32 = 36;
const OUT_GPR: u32 = 37;
const PACK_SLOT_GPR: u32 = 39;

fn roles_looped(
    layout: &Layout,
    inputs: &[Vec<Instruction>],
    finish: &[Instruction],
    input: &impl Fn(usize, usize) -> u64,
) -> Option<crate::code::LoopedRoles> {
    use crate::code::Loop;
    use crate::datapath::{
        pack_dest_units, pack_tile_from_dst_gpr, tile_base_units, unpack_datums_to_dst_gpr,
        TILE_DATUMS,
    };
    use tt_isa::backend::set_gpr;
    use tt_isa::isa::generated::encode;
    let (outputs, per) = (layout.outputs, inputs.len());
    // A first input, a middle of at least one, a last.
    if per < 3 || inputs[1..per - 1].iter().any(|p| *p != inputs[1]) {
        return None;
    }
    // Each output's inputs after its first are consecutive slots; its first, and its
    // middle's start, a fixed distance on from the last output's.
    let aligned = |a: u64| a % 16 == 0;
    let consecutive = (0..outputs).all(|k| {
        (1..per - 1).all(|n| input(k, n + 1) == input(k, n) + TILE_SLOT)
            && (0..per).all(|n| aligned(input(k, n)))
    });
    let distance = |f: &dyn Fn(usize) -> u64| -> Option<u64> {
        let d = f(1).checked_sub(f(0))?;
        (1..outputs.saturating_sub(1))
            .all(|k| f(k + 1).checked_sub(f(k)) == Some(d))
            .then_some(d)
    };
    let (first_stride, gap) = if outputs > 1 {
        (
            distance(&|k| input(k, 0))?,
            // From the last input read to the next output's first middle one, a slot back.
            distance(&|k| input(k, 1))?.checked_sub(per as u64 * TILE_SLOT - TILE_SLOT)?,
        )
    } else {
        (0, 0)
    };
    if !consecutive {
        return None;
    }
    let units = |a: u64| tile_base_units(a);
    let slot = units(TILE_SLOT);
    let set = |p: &mut Vec<Instruction>, gpr: u32, v: u32| p.extend(set_gpr(gpr, v).unwrap());
    let add = |gpr: u32, by: u32| encode::adddmareg(gpr, by, gpr).unwrap();
    let s = layout.sems;

    let mut unpack = thread_config();
    unpack.extend(clear_unpacker0_adcs());
    let mut words = ConfigWords::new();
    tile_unpack_config(&mut words, layout.in_at);
    unpack.extend(config_program(&words));
    set(&mut unpack, SLOT_GPR, slot);
    set(&mut unpack, FIRST_GPR, units(input(0, 0)));
    set(&mut unpack, FIRST_STRIDE_GPR, units(first_stride));
    // The middle steps before it reads: a slot back.
    set(&mut unpack, MID_GPR, units(input(0, 1)) - slot);
    set(&mut unpack, GAP_GPR, units(gap));
    let mut unpack_loops = Vec::new();
    let output_block = unpack.len() as u32;
    let read = |p: &mut Vec<Instruction>, gpr: u32, step: Option<u32>| {
        p.extend(sync::take(s.consumed, Before::UNPACKER));
        p.extend(unpack_datums_to_dst_gpr(gpr, step, TILE_DATUMS, A_ROW));
        p.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
    };
    read(&mut unpack, FIRST_GPR, None);
    for count in middle_counts(per) {
        let middle = unpack.len() as u32;
        read(&mut unpack, MID_GPR, Some(SLOT_GPR));
        unpack_loops.push(Loop {
            start: middle,
            len: unpack.len() as u32 - middle,
            count,
        });
    }
    read(&mut unpack, MID_GPR, Some(SLOT_GPR));
    // On to the next output: written, so stepped after.
    unpack.push(add(FIRST_GPR, FIRST_STRIDE_GPR));
    unpack.push(add(MID_GPR, GAP_GPR));
    if outputs > 1 {
        unpack_loops.insert(
            0,
            Loop {
                start: output_block,
                len: unpack.len() as u32 - output_block,
                count: outputs as u32,
            },
        );
    }
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());

    let mut math = vec![state_id()];
    let mut math_loops = Vec::new();
    let output_block = math.len() as u32;
    let compute = |m: &mut Vec<Instruction>, n: usize| {
        m.extend(sync::take(s.unpacked, Before::SFPU));
        if n == 0 {
            m.extend(sync::take(s.packed, Before::SFPU));
        }
        m.extend_from_slice(&inputs[n]);
        m.extend(sync::post_after(Unit::Sfpu, s.consumed));
    };
    compute(&mut math, 0);
    for count in middle_counts(per) {
        let middle = math.len() as u32;
        compute(&mut math, 1);
        math_loops.push(Loop {
            start: middle,
            len: math.len() as u32 - middle,
            count,
        });
    }
    compute(&mut math, per - 1);
    math.extend_from_slice(finish);
    math.extend(sync::post_after(Unit::Sfpu, s.computed));
    if outputs > 1 {
        math_loops.insert(
            0,
            Loop {
                start: output_block,
                len: math.len() as u32 - output_block,
                count: outputs as u32,
            },
        );
    }

    let mut pack = vec![state_id()];
    set(&mut pack, PACK_SLOT_GPR, slot);
    set(
        &mut pack,
        OUT_GPR,
        pack_dest_units(layout.out_at + tt_isa::dm::TILE_DATA) - slot,
    );
    let output_block = pack.len() as u32;
    pack.extend(sync::take(s.computed, Before::PACKER));
    pack.extend(pack_tile_from_dst_gpr(
        OUT_GPR,
        Some(PACK_SLOT_GPR),
        OUT_ROW,
    ));
    pack.extend(sync::post_after(Unit::Packer, s.packed));
    let pack_loops = if outputs > 1 {
        vec![Loop {
            start: output_block,
            len: pack.len() as u32 - output_block,
            count: outputs as u32,
        }]
    } else {
        Vec::new()
    };
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());

    let code = [unpack, math, pack];
    let loops = [unpack_loops, math_loops, pack_loops];
    let fits = (0..3).all(|t| crate::code::check(&loops[t], code[t].len() as u32).is_ok());
    fits.then_some((code, loops))
}

/// The middle inputs of [`roles_looped`]'s output, `per - 2`, as the counts of
/// sibling loops each within what one loop counts (`mailbox::loops`).
fn middle_counts(per: usize) -> Vec<u32> {
    let most = 1 << tt_isa::mailbox::loops::COUNT_BITS;
    let mut left = per as u32 - 2;
    let mut counts = Vec::new();
    while left > 0 {
        let n = left.min(most);
        counts.push(n);
        left -= n;
    }
    counts
}

/// A sum over rows too long for one run, in chunks of [`ROW_CHUNK`] row
/// tiles: each chunk after the first starts from the last chunk's sums, read
/// back as an input of one valid row ahead of its own tiles. Exactly Flex's
/// order still -- from `+0.0`, rows in order -- because a running sum from
/// `+0.0` is never `-0.0`, so `+0.0` plus it is it.
pub const ROW_CHUNK: usize = 16;

/// Where a chunk of a long sum over rows lives: the last chunk's sums, one
/// tile per output at `prior_at`, then [`ROW_CHUNK`] input tiles per output at
/// `in_at`, and the outputs at `out_at`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkLayout {
    pub layout: Layout,
    pub prior_at: u64,
}

pub fn plan_chunk_layout(outputs: usize) -> Result<ChunkLayout, PlanError> {
    let mut req = Requirements::new(1);
    let align = tt_isa::dram::ALIGN;
    let prior = req.scratch("sum prior", outputs as u64 * TILE_SLOT, align, 0..1);
    let input = req.scratch(
        "sum inputs",
        (outputs * ROW_CHUNK) as u64 * TILE_SLOT,
        align,
        0..1,
    );
    let out = req.scratch("sum outputs", outputs as u64 * TILE_SLOT, align, 0..1);
    let unpacked = req.semaphore("reduce unpacked", 0, 0..1);
    let consumed = req.semaphore("reduce consumed", 1, 0..1);
    let computed = req.semaphore("reduce computed", 0, 0..1);
    let packed = req.semaphore("reduce packed", 1, 0..1);
    let plan = req.plan(crate::l1::arena())?;
    Ok(ChunkLayout {
        layout: Layout {
            outputs,
            per: ROW_CHUNK,
            in_at: plan.addr(input),
            out_at: plan.addr(out),
            sems: ReduceSemaphores {
                unpacked: plan.semaphore(unpacked),
                consumed: plan.semaphore(consumed),
                computed: plan.semaphore(computed),
                packed: plan.semaphore(packed),
            },
            init: plan.semaphore_init(),
        },
        prior_at: plan.addr(prior),
    })
}

/// The role programs of one chunk of a long sum over rows: `tiles` input
/// tiles per output (at most [`ROW_CHUNK`]), the last with `last_valid` rows,
/// after the prior chunk's sums when `prior`.
pub fn chunk_roles(
    c: &ChunkLayout,
    prior: bool,
    tiles: usize,
    last_valid: u32,
) -> [Vec<Instruction>; 3] {
    assert!((1..=ROW_CHUNK).contains(&tiles));
    let mut inputs = Vec::new();
    if prior {
        inputs.push(accumulate_in_order(true, 1));
    }
    for n in 0..tiles {
        let valid = if n + 1 == tiles { last_valid } else { 32 };
        inputs.push(accumulate_in_order(!prior && n == 0, valid));
    }
    let fin = finish(ReduceOp::Sum, Axis::Rows);
    let l = &c.layout;
    roles_at(l, &inputs, &fin, |k, n| match (prior, n) {
        (true, 0) => c.prior_at + k as u64 * TILE_SLOT,
        (true, n) => l.in_at + (k * ROW_CHUNK + n - 1) as u64 * TILE_SLOT,
        (false, n) => l.in_at + (k * ROW_CHUNK + n) as u64 * TILE_SLOT,
    })
}

/// Continuation of a bounded reduction, preserving the unfolded accumulator.
#[allow(clippy::too_many_arguments)]
pub fn continuation_roles(
    c: &ChunkLayout,
    op: ReduceOp,
    axis: Axis,
    prior: bool,
    tiles: usize,
    valid: u32,
    last: bool,
) -> ([Vec<Instruction>; 3], [Vec<crate::code::Loop>; 3]) {
    let mut inputs = Vec::new();
    if prior {
        let mut p = Program::new();
        p.for_each_row_group(64, |p, o| {
            p.load(LReg::L0, Format::Int32, A_ROW + o);
            p.store(LReg::L0, Format::Int32, OUT_ROW + o);
        });
        inputs.push(p.finish());
    }
    for n in 0..tiles {
        inputs.push(accumulate(
            op,
            axis,
            !prior && n == 0,
            (n + 1 == tiles && valid < 32).then_some(valid),
        ));
    }
    let fin = if last { finish(op, axis) } else { Vec::new() };
    roles_code_at(&c.layout, &inputs, &fin, |k, n| match (prior, n) {
        (true, 0) => c.prior_at + k as u64 * TILE_SLOT,
        (true, n) => c.layout.in_at + (k * ROW_CHUNK + n - 1) as u64 * TILE_SLOT,
        (false, n) => c.layout.in_at + (k * ROW_CHUNK + n) as u64 * TILE_SLOT,
    })
}

/// What the device computes for a reduction of `a` (`[rows, cols]`): the
/// same programs run by the interpreter over each line of input tiles. The
/// result is `[rows, 1]` for [`Axis::Cols`], `[1, cols]` for [`Axis::Rows`].
pub fn reference(op: ReduceOp, axis: Axis, a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    use super::interp::Vector;
    use tt_isa::dm::face_index;
    let (rt, ct) = (rows.div_ceil(32), cols.div_ceil(32));
    let (outs, per, valid) = match axis {
        Axis::Cols => (rt, ct, (cols % 32) as u32),
        Axis::Rows => (ct, rt, (rows % 32) as u32),
    };
    let valid = if valid == 0 { 32 } else { valid };
    let (inputs, fin) = math_programs(op, axis, per, valid);
    let tile = |i: usize, j: usize| -> Vec<u32> {
        let mut t = vec![0u32; 1024];
        for r in 0..32 {
            for c in 0..32 {
                let (gr, gc) = (32 * i + r, 32 * j + c);
                if gr < rows && gc < cols {
                    t[face_index(r, c)] = a[gr * cols + gc].to_bits();
                }
            }
        }
        t
    };
    let mut out = Vec::new();
    for k in 0..outs {
        let mut v = Vector::new();
        for (n, program) in inputs.iter().enumerate() {
            let (i, j) = match axis {
                Axis::Cols => (k, n),
                Axis::Rows => (n, k),
            };
            v.put_tile(A_ROW as usize, &tile(i, j));
            v.run(program).unwrap_or_else(|e| panic!("accumulate: {e}"));
        }
        v.run(&fin).unwrap_or_else(|e| panic!("finish: {e}"));
        let t = v.tile(OUT_ROW as usize);
        match axis {
            Axis::Cols => {
                for r in 0..32 {
                    if 32 * k + r < rows {
                        out.push(f32::from_bits(t[face_index(r, 0)]));
                    }
                }
            }
            Axis::Rows => {
                for c in 0..32 {
                    if 32 * k + c < cols {
                        out.push(f32::from_bits(t[face_index(0, c)]));
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The looped reduce roles present every unpack and pack the configuration
    /// [`roles_at`]'s do, and are otherwise the same stream: whole runs, and a
    /// continuation's chunks after a prior; and their size does not grow with
    /// the inputs an output reduces.
    #[test]
    fn looped_reduce_roles_unpack_and_pack_what_the_unrolled_ones_do() {
        let expand = |code: &[Instruction], loops: &[crate::code::Loop]| {
            crate::code::Code {
                ins: code.to_vec(),
                loops: loops.to_vec(),
            }
            .expand()
        };
        let compare = |what: &str,
                       layout: &Layout,
                       inputs: &[Vec<Instruction>],
                       fin: &[Instruction],
                       input: &dyn Fn(usize, usize) -> u64|
         -> [usize; 3] {
            let (looped, loops) = roles_looped(layout, inputs, fin, &input)
                .unwrap_or_else(|| panic!("{what}: looped"));
            let unrolled = roles_at(layout, inputs, fin, input);
            for t in 0..3 {
                let a = crate::code::transfers(&expand(&looped[t], &loops[t]));
                let b = crate::code::transfers(&unrolled[t]);
                assert_eq!(a.0.len(), b.0.len(), "{what}, role {t}: transfers");
                for (n, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
                    for (addr, v) in y {
                        assert_eq!(
                            x.get(addr),
                            Some(v),
                            "{what}, role {t}: transfer {n}, word {addr}"
                        );
                    }
                }
                assert_eq!(
                    a.1, b.1,
                    "{what}, role {t}: the stream outside the addresses"
                );
            }
            std::array::from_fn(|t| {
                crate::code::Code {
                    ins: looped[t].clone(),
                    loops: loops[t].clone(),
                }
                .stored()
                .unwrap()
                .0
                .len()
            })
        };
        for (op, axis) in [
            (ReduceOp::Sum, Axis::Cols),
            (ReduceOp::Max, Axis::Cols),
            (ReduceOp::Sum, Axis::Rows),
        ] {
            for outputs in [1, 2, 5] {
                let mut sizes = Vec::new();
                for per in [3, 4, 17, 130, 131, 200] {
                    let Ok(layout) = plan_layout(outputs, per) else {
                        continue;
                    };
                    let (inputs, fin) = math_programs(op, axis, per, 7);
                    let what = format!("{op:?} {axis:?}, {outputs} outputs of {per}");
                    let input =
                        |k: usize, n: usize| layout.in_at + (k * per + n) as u64 * TILE_SLOT;
                    sizes.push(compare(&what, &layout, &inputs, &fin, &input));
                }
                // One loop counts 128; past it the middle is two sibling loops.
                let (short, long): (Vec<_>, Vec<_>) =
                    sizes.iter().enumerate().partition(|&(i, _)| i < 4);
                for set in [short, long] {
                    assert!(
                        set.windows(2).all(|w| w[0].1 == w[1].1),
                        "{op:?} {axis:?} {outputs}: stored words {sizes:?}"
                    );
                }
            }
        }
        // A continuation's chunk: the prior's copy first, then its tiles, outputs
        // `ROW_CHUNK` slots apart.
        let c = plan_chunk_layout(3).unwrap();
        for tiles in [2, 5, ROW_CHUNK] {
            let mut copy = Program::new();
            copy.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                p.store(LReg::L0, Format::Int32, OUT_ROW + o);
            });
            let mut inputs = vec![copy.finish()];
            for n in 0..tiles {
                inputs.push(accumulate(
                    ReduceOp::Max,
                    Axis::Cols,
                    false,
                    (n + 1 == tiles).then_some(9),
                ));
            }
            let fin = finish(ReduceOp::Max, Axis::Cols);
            let input = |k: usize, n: usize| match n {
                0 => c.prior_at + k as u64 * TILE_SLOT,
                n => c.layout.in_at + (k * ROW_CHUNK + n - 1) as u64 * TILE_SLOT,
            };
            compare(
                &format!("continuation of {tiles}"),
                &c.layout,
                &inputs,
                &fin,
                &input,
            );
        }
    }

    /// Where the middle inputs' programs differ, or their slots are not
    /// consecutive, there is nothing to repeat.
    #[test]
    fn roles_that_do_not_repeat_are_not_looped() {
        let layout = plan_layout(1, 4).unwrap();
        let (mut inputs, fin) = math_programs(ReduceOp::Sum, Axis::Cols, 4, 32);
        let input = |_: usize, n: usize| layout.in_at + n as u64 * TILE_SLOT;
        assert!(roles_looped(&layout, &inputs, &fin, &input).is_some());
        let backwards = |_: usize, n: usize| layout.in_at + (3 - n) as u64 * TILE_SLOT;
        assert!(roles_looped(&layout, &inputs, &fin, &backwards).is_none());
        inputs[2].push(tt_isa::sfpu::nop());
        assert!(roles_looped(&layout, &inputs, &fin, &input).is_none());
    }

    fn data(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i * 37 + 11) % 101) as f32 - 50.0)
            .collect()
    }

    /// A sum over rows is Flex's to the bit, on values whose sums round at
    /// every step and whose order matters: `+0.0` then each row in turn,
    /// ragged rows left out (a `-0.0` column stays `-0.0`).
    #[test]
    fn a_sum_over_rows_is_flex_order_bit_for_bit() {
        for (rows, cols) in [
            (37, 70),
            (32, 32),
            (5, 3),
            (64, 100),
            (100, 33),
            (1, 40),
            (97, 1),
        ] {
            let mut a: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let x = ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32;
                    (x - 0.5) * 10f32.powi((i % 7) as i32 - 3)
                })
                .collect();
            // A column of negative zeros.
            for r in 0..rows {
                a[r * cols] = -0.0;
            }
            let flex: Vec<f32> = (0..cols)
                .map(|c| (0..rows).fold(0.0f32, |s, r| s + a[r * cols + c]))
                .collect();
            let got = reference(ReduceOp::Sum, Axis::Rows, &a, rows, cols);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&flex), "[{rows}, {cols}]");
        }
    }

    /// Small integers: every sum exact whatever the order, so the model must
    /// equal the plain reduction, ragged edges and all.
    #[test]
    fn the_programs_reduce_every_shape() {
        for (rows, cols) in [(37, 70), (32, 32), (5, 3), (64, 100), (100, 33)] {
            let a = data(rows, cols);
            for op in [ReduceOp::Sum, ReduceOp::Max] {
                let f = |x: f32, y: f32| if op == ReduceOp::Sum { x + y } else { x.max(y) };
                let init = if op == ReduceOp::Sum {
                    0.0
                } else {
                    f32::NEG_INFINITY
                };
                let by_row: Vec<f32> = (0..rows)
                    .map(|r| {
                        a[r * cols..(r + 1) * cols]
                            .iter()
                            .fold(init, |s, &x| f(s, x))
                    })
                    .collect();
                let by_col: Vec<f32> = (0..cols)
                    .map(|c| (0..rows).fold(init, |s, r| f(s, a[r * cols + c])))
                    .collect();
                assert_eq!(
                    reference(op, Axis::Cols, &a, rows, cols),
                    by_row,
                    "{op:?} over columns [{rows}, {cols}]"
                );
                assert_eq!(
                    reference(op, Axis::Rows, &a, rows, cols),
                    by_col,
                    "{op:?} over rows [{rows}, {cols}]"
                );
            }
        }
    }
}
