//! Element-wise ops as SFPU programs (`hardware-coverage.md` S1): today's
//! data-mover kinds (`tt_isa::dm::kind`), bit for bit what the mover's FP32
//! unit computes.
//!
//! Each program reads its operands from the kernel's rows
//! (`super::kernel::A_ROW`, `B_ROW`) and writes `OUT_ROW`. The mover's
//! `fadd.s`/`fsub.s`/`fmul.s` round to nearest even and flush denormals, and
//! so does `SFPMAD` (`fma_bh`, which `fma_oracle` holds to `fma.c`); the
//! integer tests of `RELU` are the same predicate in `SFPGT`'s total order.
//! `step19_eltwise` runs both units over the same operands against `burn-flex`.

use tt_isa::dm::kind;
use tt_isa::isa::Instruction;

use super::kernel::{bias_row, Operands, A_ROW, B_ROW, OUT_ROW};
use super::{Cond, Format, LReg, Program};

/// `+inf`'s bits plus one: the first positive NaN.
const FIRST_NAN: u32 = 0x7f80_0001;

/// The program for `kind` (with its scalar), and what operands it takes, or
/// `None` for a kind with no SFPU program.
pub fn program(kind: u32, scalar: f32) -> Option<(Operands, Vec<Instruction>)> {
    let mut p = Program::new();
    let operands = match kind {
        kind::ADD | kind::SUB | kind::MUL => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                p.load(LReg::L1, Format::Fp32, B_ROW + o);
                match kind {
                    kind::ADD => p.add(LReg::L0, LReg::L1, LReg::L2),
                    kind::SUB => p.sub(LReg::L0, LReg::L1, LReg::L2),
                    _ => p.mul(LReg::L0, LReg::L1, LReg::L2),
                }
                p.store(LReg::L2, Format::Fp32, OUT_ROW + o);
            });
            Operands::Binary
        }
        kind::MUL_SCALAR | kind::ADD_SCALAR => {
            p.loadi(LReg::L3, scalar);
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                if kind == kind::MUL_SCALAR {
                    p.mul(LReg::L0, LReg::L3, LReg::L2);
                } else {
                    p.add(LReg::L0, LReg::L3, LReg::L2);
                }
                p.store(LReg::L2, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        // `max(x, 0)` as the mover has it: `x` where it is positive and not
        // NaN (`+0 < x <= +inf` in the total order), `+0` everywhere else --
        // and `RELU_BACKWARD`'s gradient on the same lanes.
        kind::RELU | kind::RELU_BACKWARD => {
            let backward = kind == kind::RELU_BACKWARD;
            p.loadi_bits(LReg::L5, FIRST_NAN);
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                if backward {
                    p.load(LReg::L1, Format::Fp32, B_ROW + o);
                }
                p.mov(LReg::ZERO, LReg::L2);
                p.if_(Cond::Less(LReg::ZERO, LReg::L0), |p| {
                    p.if_(Cond::Less(LReg::L0, LReg::L5), |p| {
                        p.mov(if backward { LReg::L1 } else { LReg::L0 }, LReg::L2)
                    })
                });
                p.store(LReg::L2, Format::Fp32, OUT_ROW + o);
            });
            if backward {
                Operands::Binary
            } else {
                Operands::Unary
            }
        }
        // Each row group reads its column half of the broadcast row from the
        // rows the kernel laid it in, which depend on the group: unrolled.
        kind::ADD_ROW => {
            let mut p = Program::with_policy(super::LoopPolicy::Unrolled);
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                p.load(LReg::L1, Format::Fp32, bias_row(o / 4) + (o & 2));
                p.add(LReg::L0, LReg::L1, LReg::L2);
                p.store(LReg::L2, Format::Fp32, OUT_ROW + o);
            });
            return Some((Operands::RowBroadcast, p.finish()));
        }
        _ => return None,
    };
    Some((operands, p.finish()))
}

#[cfg(test)]
mod tests {
    use super::super::interp::Vector;
    use super::*;

    fn tile(seed: u32) -> Vec<u32> {
        let mut s = seed | 1;
        (0..1024)
            .map(|i| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                match i % 13 {
                    0 => 0x7fc0_0000,
                    1 => 0x8000_0000,
                    2 => 0x7f80_0000,
                    3 => 0xff80_0000,
                    4 => 0xffc0_0001,
                    _ => (s & 0x8000_0000) | (((s >> 8) % 40 + 107) << 23) | (s & 0x7f_ffff),
                }
            })
            .collect()
    }

    /// The mover's per-datum arithmetic (`dm_b.rs`), in IEEE terms with its
    /// denormal flush -- the reference the programs must equal.
    fn mover(kind: u32, s: f32, a: u32, b: u32, bias: u32) -> u32 {
        let f = f32::from_bits;
        let positive = (a as i32) > 0 && a <= 0x7f80_0000;
        let r = match kind {
            kind::ADD => (f(a) + f(b)).to_bits(),
            kind::SUB => (f(a) - f(b)).to_bits(),
            kind::MUL => (f(a) * f(b)).to_bits(),
            kind::MUL_SCALAR => (f(a) * s).to_bits(),
            kind::ADD_SCALAR => (f(a) + s).to_bits(),
            kind::RELU => {
                if positive {
                    a
                } else {
                    0
                }
            }
            kind::RELU_BACKWARD => {
                if positive {
                    b
                } else {
                    0
                }
            }
            kind::ADD_ROW => (f(a) + f(bias)).to_bits(),
            _ => unreachable!(),
        };
        if f(r).is_nan() && !matches!(kind, kind::RELU | kind::RELU_BACKWARD) {
            0x7fc0_0000
        } else {
            r
        }
    }

    #[test]
    fn every_program_computes_what_the_mover_does() {
        let (a, b) = (tile(3), tile(5));
        for (k, s) in [
            (kind::ADD, 0.0),
            (kind::SUB, 0.0),
            (kind::MUL, 0.0),
            (kind::MUL_SCALAR, -1.75),
            (kind::MUL_SCALAR, 3.0e-3),
            (kind::ADD_SCALAR, -0.0),
            (kind::ADD_SCALAR, 2.7),
            (kind::RELU, 0.0),
            (kind::RELU_BACKWARD, 0.0),
            (kind::ADD_ROW, 0.0),
        ] {
            let (operands, math) = program(k, s).unwrap();
            let mut v = Vector::new();
            v.put_tile(A_ROW as usize, &a);
            if operands == Operands::RowBroadcast {
                // As the kernel lays it: row 0's two faces, four times each.
                for r in 0..4 {
                    v.dst[(B_ROW + r) as usize] = std::array::from_fn(|c| b[c]);
                    v.dst[(B_ROW + 4 + r) as usize] = std::array::from_fn(|c| b[256 + c]);
                }
            } else {
                v.put_tile(B_ROW as usize, &b);
            }
            v.run(&math).unwrap_or_else(|e| panic!("kind {k}: {e}"));
            let got = v.tile(OUT_ROW as usize);
            for i in 0..1024 {
                // Datum i's column, for the broadcast row.
                let col = (i / 256 % 2) * 16 + i % 16;
                let bias = if col < 16 { b[col] } else { b[256 + col - 16] };
                assert_eq!(
                    got[i],
                    mover(k, s, a[i], b[i], bias),
                    "kind {k}: datum {i}: a {:#010x} b {:#010x}",
                    a[i],
                    b[i]
                );
            }
        }
    }
}

#[cfg(test)]
mod fit {
    use super::*;
    use crate::sfpu::kernel::{plan_layout, roles};

    /// Every op's longest run (`tensor::sfpu_eltwise`'s group) has role
    /// programs that fit a program slot, and is no shorter than it must be.
    #[test]
    fn every_op_s_longest_run_fits_a_program_slot() {
        for k in [
            kind::ADD,
            kind::SUB,
            kind::MUL,
            kind::MUL_SCALAR,
            kind::ADD_SCALAR,
            kind::RELU,
            kind::RELU_BACKWARD,
            kind::ADD_ROW,
        ] {
            let (operands, math) = program(k, 0.5).unwrap();
            let n = crate::tensor::sfpu_group_for_tests(k, 0.5, operands);
            let fits = |n: usize| {
                let layout = plan_layout(n, operands).unwrap();
                roles(&layout, operands, &math)
                    .iter()
                    .all(|p| p.len() <= tt_isa::mailbox::PROGRAM_MAX as usize)
            };
            assert!(fits(n), "kind {k}: {n} tiles");
            assert!(
                n == 64 || !fits(n + 1),
                "kind {k}: {n} tiles, but {} would fit",
                n + 1
            );
        }
    }
}
