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

/// Ops only the SFPU has: numbered above the mover's kinds
/// (`tt_isa::dm::kind::LAST`), so the session sends them to the SFPU whatever
/// unit it is set to.
pub mod kind_sfpu {
    /// `1 / a`, within one ulp of the correctly rounded reciprocal
    /// (`Program::recip`).
    pub const RECIP: u32 = 0x100;
    /// `a / b`, within one ulp of the correctly rounded quotient.
    pub const DIV: u32 = 0x101;
    /// `a / s`.
    pub const DIV_SCALAR: u32 = 0x102;
}

/// Does the data mover implement `kind`?
pub fn mover_has(kind: u32) -> bool {
    (1..=kind::LAST).contains(&kind)
}

/// What operands `kind` takes, if the SFPU has it.
pub fn operands(kind: u32) -> Option<Operands> {
    Some(match kind {
        kind::ADD | kind::SUB | kind::MUL | kind::RELU_BACKWARD | kind_sfpu::DIV => {
            Operands::Binary
        }
        kind::MUL_SCALAR
        | kind::ADD_SCALAR
        | kind::RELU
        | kind_sfpu::RECIP
        | kind_sfpu::DIV_SCALAR => Operands::Unary,
        kind::ADD_ROW => Operands::RowBroadcast,
        _ => return None,
    })
}

/// `q = a / b` into `q`, from `a`, `b` and `y = 1/b` (`Program::recip`): the
/// product, then one correction `q += (a - b*q) * y` in the lanes where the
/// product is finite and not zero. The product is within `e_y + 2^-24`
/// (`< 1.8e-7`) of `a/b`; the remainder `a - b*q` is computed with one
/// rounding (`SFPMAD`) and is that error times `a`; adding `remainder * y`
/// leaves `a/b * (1 + O(e_y^2))` before the last rounding -- so within half an
/// ulp plus `~1e-13` relative, at most one ulp from the correct rounding.
/// `r`, `t` are scratch; `inf` holds `+inf`.
#[allow(clippy::too_many_arguments)]
fn divide(p: &mut Program, a: LReg, b: LReg, y: LReg, q: LReg, r: LReg, t: LReg, inf: LReg) {
    p.mul(a, y, q);
    p.abs(q, t);
    p.if_(Cond::Less(LReg::ZERO, t), |p| {
        p.if_(Cond::Less(t, inf), |p| {
            p.nmad(b, q, a, r);
            p.mad(r, y, q, q);
        })
    });
}

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
        kind_sfpu::RECIP => {
            p.loadi_bits(LReg::L6, f32::MAX.to_bits());
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                p.recip(LReg::L0, LReg::L2, LReg::L3, LReg::L4, LReg::L6);
                p.store(LReg::L2, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::DIV | kind_sfpu::DIV_SCALAR => {
            let scalar_b = kind == kind_sfpu::DIV_SCALAR;
            p.loadi_bits(LReg::L6, f32::MAX.to_bits());
            p.loadi_bits(LReg::L7, 0x7f80_0000);
            if scalar_b {
                p.loadi(LReg::L1, scalar);
            }
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                if !scalar_b {
                    p.load(LReg::L1, Format::Fp32, B_ROW + o);
                }
                p.recip(LReg::L1, LReg::L2, LReg::L3, LReg::L4, LReg::L6);
                divide(
                    p,
                    LReg::L0,
                    LReg::L1,
                    LReg::L2,
                    LReg::L5,
                    LReg::L3,
                    LReg::L4,
                    LReg::L7,
                );
                p.store(LReg::L5, Format::Fp32, OUT_ROW + o);
            });
            if scalar_b {
                Operands::Unary
            } else {
                Operands::Binary
            }
        }
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

/// What the device computes for `kind` over a whole `[rows, cols]` tensor:
/// the op's own program run by the interpreter (`super::interp`) over each
/// tile, laid in `Dst` as the kernel lays it -- so a gate can hold the device
/// to it bit for bit, and hold it to `burn-flex` within the op's derived
/// bound. `b` is `[rows, cols]`, or `[1, cols]` for a row broadcast. Padding
/// is zero, as an upload leaves it.
pub fn reference(
    kind: u32,
    scalar: f32,
    a: &[f32],
    b: Option<&[f32]>,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    use super::interp::Vector;
    use tt_isa::dm::face_index;
    let (operands, math) = program(kind, scalar).expect("an SFPU op");
    let (rt, ct) = (rows.div_ceil(32), cols.div_ceil(32));
    let mut out = vec![0.0f32; rows * cols];
    let tile = |x: &[f32], xr: usize, i: usize, j: usize| -> Vec<u32> {
        let mut t = vec![0u32; 1024];
        for r in 0..32 {
            for c in 0..32 {
                let (gr, gc) = (32 * i + r, 32 * j + c);
                if gr < xr && gc < cols {
                    t[face_index(r, c)] = x[gr * cols + gc].to_bits();
                }
            }
        }
        t
    };
    for i in 0..rt {
        for j in 0..ct {
            let mut v = Vector::new();
            v.put_tile(A_ROW as usize, &tile(a, rows, i, j));
            match (operands, b) {
                (Operands::Binary, Some(b)) => v.put_tile(B_ROW as usize, &tile(b, rows, i, j)),
                (Operands::RowBroadcast, Some(b)) => {
                    let t = tile(b, 1, 0, j);
                    for r in 0..4 {
                        v.dst[(B_ROW + r) as usize] = std::array::from_fn(|c| t[c]);
                        v.dst[(B_ROW + 4 + r) as usize] = std::array::from_fn(|c| t[256 + c]);
                    }
                }
                (Operands::Unary, _) => {}
                _ => panic!("kind {kind:#x} takes a second operand"),
            }
            v.run(&math)
                .unwrap_or_else(|e| panic!("kind {kind:#x}: {e}"));
            let t = v.tile(OUT_ROW as usize);
            for r in 0..32 {
                for c in 0..32 {
                    let (gr, gc) = (32 * i + r, 32 * j + c);
                    if gr < rows && gc < cols {
                        out[gr * cols + gc] = f32::from_bits(t[face_index(r, c)]);
                    }
                }
            }
        }
    }
    out
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

#[cfg(test)]
mod division {
    use super::super::interp::Vector;
    use super::*;

    /// Distance in ulps between two finite floats of the same sign.
    fn ulps(a: f32, b: f32) -> u32 {
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
    }

    fn normals(seed: u32) -> Vec<u32> {
        let mut s = seed | 1;
        (0..1024)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Exponents well inside the range, so quotients and
                // reciprocals stay normal.
                (s & 0x8000_0000) | (((s >> 8) % 100 + 77) << 23) | (s & 0x7f_ffff)
            })
            .collect()
    }

    fn run(k: u32, s: f32, a: &[u32], b: &[u32]) -> Vec<u32> {
        let (_, math) = program(k, s).unwrap();
        let mut v = Vector::new();
        v.put_tile(A_ROW as usize, a);
        v.put_tile(B_ROW as usize, b);
        v.run(&math).unwrap_or_else(|e| panic!("kind {k:#x}: {e}"));
        v.tile(OUT_ROW as usize)
    }

    /// Within one ulp of the correctly rounded result (the bound derived on
    /// `Program::recip` and `divide`), and the reported worst case.
    #[test]
    fn reciprocal_and_quotient_are_within_one_ulp() {
        let (a, b) = (normals(7), normals(11));
        let f = f32::from_bits;
        let recip = run(kind_sfpu::RECIP, 0.0, &a, &b);
        let div = run(kind_sfpu::DIV, 0.0, &a, &b);
        let div_s = run(kind_sfpu::DIV_SCALAR, -3.7, &a, &b);
        let (mut worst, mut exact) = (0, 0);
        for i in 0..1024 {
            for (got, want) in [
                (f(recip[i]), 1.0 / f(a[i])),
                (f(div[i]), f(a[i]) / f(b[i])),
                (f(div_s[i]), f(a[i]) / -3.7),
            ] {
                let d = ulps(got, want);
                assert!(
                    d <= 1,
                    "{got:e} vs {want:e}: {d} ulps (a {:#x} b {:#x})",
                    a[i],
                    b[i]
                );
                worst = worst.max(d);
                exact += usize::from(d == 0);
            }
        }
        println!("worst {worst} ulp; {exact} of 3072 correctly rounded");
    }

    /// The special cases, as IEEE has them (denormals aside: they flush).
    #[test]
    fn the_special_cases() {
        let f = f32::from_bits;
        let cases: [(f32, f32); 10] = [
            (1.0, 0.0),
            (1.0, -0.0),
            (-2.0, f32::INFINITY),
            (0.0, 0.0),
            (f32::INFINITY, f32::INFINITY),
            (f32::INFINITY, 2.0),
            (5.0, f32::NAN),
            (f32::NAN, 5.0),
            (0.0, 3.0),
            (1.0e-30, 1.0e20),
        ];
        let mut a = vec![1.0f32.to_bits(); 1024];
        let mut b = vec![1.0f32.to_bits(); 1024];
        for (i, (x, y)) in cases.iter().enumerate() {
            a[i] = x.to_bits();
            b[i] = y.to_bits();
        }
        let recip = run(kind_sfpu::RECIP, 0.0, &b, &a);
        let div = run(kind_sfpu::DIV, 0.0, &a, &b);
        for (i, (x, y)) in cases.iter().enumerate() {
            for (got, want, what) in [(f(recip[i]), 1.0 / y, "recip"), (f(div[i]), x / y, "div")] {
                // 1e-30 / 1e20 is denormal in IEEE and flushes here.
                let want = if want != 0.0 && want.abs() < f32::MIN_POSITIVE {
                    0.0
                } else {
                    want
                };
                if want.is_nan() {
                    assert!(got.is_nan(), "{what} {x} / {y}: {got}");
                } else {
                    assert_eq!(got, want, "{what} {x} / {y}");
                    assert_eq!(
                        got.is_sign_negative(),
                        want.is_sign_negative(),
                        "{what} {x} / {y}: sign"
                    );
                }
            }
        }
    }
}
