//! The Matrix Unit's group adder, as measured (`step158_matmul_accumulation`, silicon operating
//! notes "Matmul accumulation: the Matrix Unit's group adder"): a host model of what a tile matmul's
//! `Dst` chain computes, shared by the gates whose oracles predict a product exactly (step158, and
//! flash-decode's split order, `docs/plans/long-context.md` Part 2).
//!
//! One `MVMUL` is a 16-term dot product run once a fidelity phase, its terms two groups of eight.
//! A group is aligned once, to a quantum of `2^(max term exponent - 10)` (a term's exponent the sum
//! of its operands', a low slice's taken from its position), each product rounded half-up on its
//! magnitude to the quantum and summed exactly; the group's sum is added to the F32 `Dst`, to
//! nearest with ties toward +infinity. Measured over HiFi4 with TF32 operands; not for subnormals,
//! NaN or infinity.

use tt_isa::numerics::{src_a_fidelity_bits, src_b_fidelity_bits};

/// The measured rules, with the three knobs the negative controls turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    pub fraction_bits: i32,
    pub zero_slice_sets_exponent: bool,
    pub ties_toward_zero: bool,
}
pub const MEASURED: Rules = Rules {
    fraction_bits: 10,
    zero_slice_sets_exponent: true,
    ties_toward_zero: false,
};

fn exp2(x: f32) -> i32 {
    f64::from(x.abs()).log2().floor() as i32
}

/// `x` to a multiple of `q`, half-up on the magnitude (or toward zero on a tie, as a control).
fn align(x: f64, q: f64, rules: Rules) -> f64 {
    let r = x.abs() / q;
    let n = if rules.ties_toward_zero {
        (r - 0.5).ceil().max(0.0)
    } else {
        (r + 0.5).floor()
    };
    x.signum() * n * q
}

/// The group quantum of one phase, and the products, of eight terms; `None` if every term is zero.
fn group(a: &[f32], b: &[f32], phase: u32, rules: Rules) -> Option<(f64, Vec<f64>)> {
    let mut e_max = i32::MIN;
    let mut products = Vec::new();
    for k in 0..8 {
        let (x, y) = (
            src_b_fidelity_bits(a[k], phase),
            src_a_fidelity_bits(b[k], phase),
        );
        products.push(f64::from(x) * f64::from(y));
        let live = if rules.zero_slice_sets_exponent {
            a[k] != 0.0 && b[k] != 0.0
        } else {
            x != 0.0 && y != 0.0
        };
        if live {
            let ea = exp2(a[k]) - if phase & 2 != 0 { 7 } else { 0 };
            let eb = exp2(b[k]) - if phase & 1 != 0 { 5 } else { 0 };
            e_max = e_max.max(ea + eb);
        }
    }
    (e_max != i32::MIN).then(|| (2f64.powi(e_max - rules.fraction_bits), products))
}

/// The next F32 above `x` (finite, not the largest).
fn above(x: f32) -> f32 {
    if x == 0.0 {
        f32::from_bits(1)
    } else if x > 0.0 {
        f32::from_bits(x.to_bits() + 1)
    } else {
        f32::from_bits(x.to_bits() - 1)
    }
}

/// The next F32 below `x`.
fn below(x: f32) -> f32 {
    -above(-x)
}

/// Round an exact value to F32: nearest, ties toward +infinity.
pub fn f32_ties_up(x: f64) -> f32 {
    let near = x as f32;
    let (lo, hi) = if f64::from(near) <= x {
        (near, above(near))
    } else {
        (below(near), near)
    };
    if x - f64::from(lo) == f64::from(hi) - x {
        hi
    } else {
        near
    }
}

fn ulp(x: f32) -> f64 {
    2f64.powi(exp2(x.max(f32::MIN_POSITIVE)) - 23)
}

/// `a . b` as the tile computes it, and the error bound the rules imply (against the exact dot of the
/// operands, valid where they are exact in the slices): half a quantum per rounded term, plus an ulp of
/// `Dst` per add.
pub fn dot(a: &[f32], b: &[f32], rules: Rules) -> (f32, f64) {
    dot_from(0.0, a, b, rules)
}

/// [`dot`] continuing a `Dst` chain from `start` (a K block's reload, or a product's previous
/// tiles): the same groups, added to `start` as they would be to an `Dst` holding it.
pub fn dot_from(start: f32, a: &[f32], b: &[f32], rules: Rules) -> (f32, f64) {
    let k = a.len().div_ceil(16) * 16;
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    a.resize(k, 0.0);
    b.resize(k, 0.0);
    let (mut dst, mut bound) = (start, 0f64);
    for face in 0..k / 16 {
        for phase in 0..4 {
            for g in 0..2 {
                let at = face * 16 + g * 8;
                let sum =
                    group(&a[at..at + 8], &b[at..at + 8], phase, rules).map_or(0.0, |(q, p)| {
                        bound += p.iter().filter(|&&x| x != 0.0).count() as f64 * q / 2.0;
                        p.iter().map(|&x| align(x, q, rules)).sum::<f64>()
                    });
                let next = f32_ties_up(f64::from(dst) + sum);
                bound += ulp(next).max(ulp(dst));
                dst = next;
            }
        }
    }
    (dst, bound)
}
