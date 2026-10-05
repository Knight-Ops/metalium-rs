//! Inclusive scans in increasing logical row order. A continuation reads the
//! preceding output tile's last row, never a folded partial reduction.
use super::kernel::{A_ROW, B_ROW, OUT_ROW};
use super::{Cond, Format, LReg, Program};
use tt_isa::isa::Instruction;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScanOp {
    Sum,
    Prod,
    /// Raw FP32 total order: -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN.
    Min,
    Max,
}

/// One tile, with an optional preceding output tile in B. The four rows
/// loaded together are transposed into L0..L3, visited in order and then
/// transposed back. L4's row zero survives both transposes (the diagonal).
pub fn program(op: ScanOp, first: bool) -> Vec<Instruction> {
    let mut p = Program::new();
    for (top, bottom) in [(0, 32), (16, 48)] {
        for half in [0, 2] {
            if first {
                p.loadi_bits(LReg::L4, identity(op));
            } else {
                p.load(LReg::L0, Format::Int32, B_ROW + bottom + 12 + half);
                p.transpose4();
                p.mov(LReg::L3, LReg::L4);
            }
            for g in 0..8 {
                let row = if g < 4 {
                    top + 4 * g
                } else {
                    bottom + 4 * (g - 4)
                } + half;
                p.load(LReg::L0, Format::Int32, A_ROW + row);
                p.transpose4();
                for reg in [LReg::L0, LReg::L1, LReg::L2, LReg::L3] {
                    match op {
                        ScanOp::Sum => p.add(LReg::L4, reg, LReg::L4),
                        ScanOp::Prod => p.mul(LReg::L4, reg, LReg::L4),
                        ScanOp::Min => p.if_(Cond::Less(reg, LReg::L4), |p| p.mov(reg, LReg::L4)),
                        ScanOp::Max => p.if_(Cond::Less(LReg::L4, reg), |p| p.mov(reg, LReg::L4)),
                    }
                    p.mov(LReg::L4, reg);
                }
                p.transpose4();
                p.store(LReg::L0, Format::Int32, OUT_ROW + row);
            }
        }
    }
    p.finish()
}

fn identity(op: ScanOp) -> u32 {
    match op {
        ScanOp::Sum => 0,
        ScanOp::Prod => 0x3f800000,
        ScanOp::Min => 0x7fffffff,
        ScanOp::Max => 0xffffffff,
    }
}

/// Independent scalar execution order, using the specified Blackhole arithmetic.
pub fn reference(op: ScanOp, input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = input.to_vec();
    for c in 0..cols {
        let mut acc = f32::from_bits(identity(op));
        for r in 0..rows {
            let x = input[r * cols + c].to_bits();
            let bits = match op {
                ScanOp::Sum => tt_isa::numerics::fma_bh(1.0f32.to_bits(), x, acc.to_bits()),
                ScanOp::Prod => tt_isa::numerics::fma_bh(acc.to_bits(), x, 0x80000000),
                ScanOp::Min | ScanOp::Max => {
                    let order = f32::from_bits(x).total_cmp(&acc);
                    if (op == ScanOp::Min && order.is_lt()) || (op == ScanOp::Max && order.is_gt())
                    {
                        x
                    } else {
                        acc.to_bits()
                    }
                }
            };
            acc = f32::from_bits(bits);
            out[r * cols + c] = acc;
        }
    }
    out
}
