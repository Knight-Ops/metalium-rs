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
//! the gate. (A sum over rows stays on the mover, which adds in Flex's order
//! exactly: `tensor::sum_rows`.)

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
}

impl ReduceOp {
    /// The identity a padding lane is replaced by.
    fn identity(self) -> u32 {
        match self {
            ReduceOp::Sum => 0,
            ReduceOp::Max => 0xff80_0000,
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
fn combine(p: &mut Program, op: ReduceOp, other: LReg, into: LReg) {
    match op {
        ReduceOp::Sum => p.add(other, into, into),
        ReduceOp::Max => p.if_(Cond::Less(into, other), |p| p.mov(other, into)),
    }
}

/// The accumulate program for one input tile: `first` copies it to the
/// accumulator, otherwise it is combined in. `valid`: the input is the last
/// along the reduced dimension and only its first `valid` columns (or rows,
/// for [`Axis::Rows`]) are data -- the rest are replaced by the identity.
pub fn accumulate(op: ReduceOp, axis: Axis, first: bool, valid: Option<u32>) -> Vec<Instruction> {
    let masked = valid.is_some_and(|v| v < 32);
    let mut p = Program::with_policy(if masked {
        LoopPolicy::Unrolled
    } else {
        LoopPolicy::Replay
    });
    if masked {
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
    }
    let v = valid.unwrap_or(32) as i32;
    p.for_each_row_group(64, |p, o| {
        p.load(LReg::L0, Format::Fp32, A_ROW + o);
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
            p.store(LReg::L0, Format::Fp32, OUT_ROW + o);
        } else {
            p.load(LReg::L1, Format::Fp32, OUT_ROW + o);
            combine(p, op, LReg::L0, LReg::L1);
            p.store(LReg::L1, Format::Fp32, OUT_ROW + o);
        }
    });
    p.finish()
}

/// The finishing program: the accumulator folded within the tile (see the
/// module documentation).
pub fn finish(op: ReduceOp, axis: Axis) -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    let acc = |r: u32| OUT_ROW + r;
    match axis {
        Axis::Cols => {
            // Tensor rows 0..16 are in faces 0 (left) and 1 (right), 16..32 in
            // faces 2 and 3; a group of four rows in each.
            for lower in [0, 32] {
                for g in 0..4 {
                    let (left, right) = (lower + 4 * g, lower + 16 + 4 * g);
                    let at = [left, left | 2, right, right | 2];
                    p.load(LReg::L0, Format::Fp32, acc(at[0]));
                    for &a in &at[1..] {
                        p.load(LReg::L1, Format::Fp32, acc(a));
                        combine(&mut p, op, LReg::L1, LReg::L0);
                    }
                    // Across the eight lanes of each lane row.
                    p.mov(LReg::L0, LReg::L2);
                    for _ in 0..7 {
                        p.rotate_row(LReg::L2, LReg::L2);
                        combine(&mut p, op, LReg::L2, LReg::L0);
                    }
                    for &a in &at {
                        p.store(LReg::L0, Format::Fp32, acc(a));
                    }
                }
            }
        }
        Axis::Rows => {
            for (top, bottom) in [(0, 32), (16, 48)] {
                for half in [0, 2] {
                    p.load(LReg::L0, Format::Fp32, acc(top + half));
                    for g in 0..8 {
                        let r = if g < 4 {
                            top + 4 * g
                        } else {
                            bottom + 4 * (g - 4)
                        };
                        if r != top {
                            p.load(LReg::L1, Format::Fp32, acc(r + half));
                            combine(&mut p, op, LReg::L1, LReg::L0);
                        }
                    }
                    // Lane row `i` of `L0` into lane row 0 of `L[i]`.
                    p.transpose4();
                    combine(&mut p, op, LReg::L1, LReg::L0);
                    combine(&mut p, op, LReg::L3, LReg::L2);
                    combine(&mut p, op, LReg::L2, LReg::L0);
                    p.store(LReg::L0, Format::Fp32, acc(top + half));
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

pub fn plan_layout(outputs: usize, per: usize) -> Result<Layout, PlanError> {
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
    let plan = req.plan(tt_isa::l1::DATA)?;
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

/// The three role programs of a run over `layout`.
pub fn roles(
    layout: &Layout,
    inputs: &[Vec<Instruction>],
    finish: &[Instruction],
) -> [Vec<Instruction>; 3] {
    assert_eq!(inputs.len(), layout.per);
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
            unpack.extend(unpack_tile_to_dst(
                slot(layout.in_at, k * layout.per + n),
                A_ROW,
            ));
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

    fn data(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i * 37 + 11) % 101) as f32 - 50.0)
            .collect()
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
