//! `MatmulMode::Accurate`: a matmul as integer slices with shared exponents, on the Matrix Unit.
//!
//! The fast matmul is inexact because the unit's group adder aligns each group of eight products to
//! the largest term's exponent and rounds the rest to that quantum (`docs/learnings/
//! silicon-operating-notes.md`, "Matmul accumulation: the Matrix Unit's group adder"). The loss
//! happens only where a term is not a multiple of the quantum, so it is avoided by arranging that
//! every term is: if the left operand is an integer `|x| <= 127` (the seven significant bits of the
//! `SrcB` slice at `Fidelity::Lo`) and the right one an integer `|y| <= 31` (the five of `SrcA`),
//! the largest operand-exponent sum is 6 + 4 = 10, the quantum `2^(10 - 10)` is 1, and every product
//! and every sum of eight is exact (`step158_matmul_accumulation`, `lofi_is_exact_on_...`). This is
//! the Ozaki scheme, fitted to that measured rule.
//!
//! Per row of the left operand and column of the right one a power of two is taken from the largest
//! magnitude (`float_pow2_floor`, an exponent-field mask), the operand is scaled by it into
//! `[2^k, 2^(k+1))` at its largest, and cut into integer slices: `trunc`, then the remainder times
//! the radix, repeated. Slice `s` of the left operand carries `2^-(7 (s - 1))`, slice `t` of the right
//! one `2^-(5 (t - 1))`, both exact. One `Fidelity::Lo` matmul per kept pair `(s, t)` follows, and
//! the partial products are added from the smallest to the largest, in F32. The sum is scaled back
//! by the two powers of two, which are exact.
//!
//! **Domain.** Operands are clamped to `|x| <= 2^64` and a row or column whose largest magnitude is
//! below `2^-64` is scaled as if it were `2^-64`, so every scale factor is a normal float and no
//! intermediate overflows or flushes. Inside it the result is within the bound `step163` derives;
//! outside, use `MatmulMode::Exact`.
//!
//! **NaN and infinity** are IEEE's and cost no precision: they are taken out of the operands before
//! the slicing, and three matmuls of 0 and +-1 (exact at any fidelity) say what the sum does with
//! them. A NaN anywhere in a row of the left operand or a column of the right one makes every entry
//! of that row or column NaN. A product of an infinity and a zero is NaN. A product of an infinity
//! and any other nonzero value is an infinity of the sign of the product; an entry that has both
//! signs is NaN; one that has one is that infinity, whatever the finite terms add up to.
use super::*;
use crate::{Fidelity, MatmulMode};

/// Magnitudes the slicing is specified for: `2^-DOMAIN ..= 2^DOMAIN`.
const DOMAIN: i32 = 64;
/// Significant bits of the left operand's slice (`SrcB`, `0xfffe_0000`) and the right one's
/// (`SrcA`, `0xfff8_0000`) at `Fidelity::Lo`.
const LHS_BITS: u32 = 7;
const RHS_BITS: u32 = 5;
/// Slices of each operand and the largest total weight `7 (s - 1) + 5 (t - 1)` of a kept pair.
/// Four slices hold 28 bits of the left operand and five 25 of the right one; the pairs kept
/// reach 23 bits below the leading product, past an F32's 24.
pub const LHS_SLICES: usize = 4;
pub const RHS_SLICES: usize = 5;
pub const MAX_OFFSET: u32 = 23;

static OFFSET: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(MAX_OFFSET);
static LEAD_SHIFT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Tune the slicing for the process: the largest total weight of a kept slice pair (fewer pairs are
/// fewer matmuls and a larger error) and a shift of the leading slice's scale, in bits (0 is the
/// design: the leading slices have 7 and 5 significant bits, which the unit keeps whole). A nonzero
/// shift breaks that on purpose and is how the gate shows the rule is needed. Not for use otherwise.
#[doc(hidden)]
pub fn set_slicing(max_offset: u32, lead_shift: i32) {
    OFFSET.store(max_offset, std::sync::atomic::Ordering::Relaxed);
    LEAD_SHIFT.store(lead_shift, std::sync::atomic::Ordering::Relaxed);
}

fn scalar(x: f32) -> burn_backend::Scalar {
    x.into()
}

fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// The slice pairs `(s, t)`, zero based, from the smallest weight to the largest: the order the
/// partial products are added in.
pub fn pairs() -> Vec<(usize, usize)> {
    let mut kept: Vec<(usize, usize)> = (0..LHS_SLICES)
        .flat_map(|s| (0..RHS_SLICES).map(move |t| (s, t)))
        .filter(|&(s, t)| {
            (LHS_BITS as usize * s + RHS_BITS as usize * t) as u32
                <= OFFSET.load(std::sync::atomic::Ordering::Relaxed)
        })
        .collect();
    kept.sort_by_key(|&(s, t)| std::cmp::Reverse(LHS_BITS as usize * s + RHS_BITS as usize * t));
    kept
}

/// `x` cut into `count` integer slices of `bits` bits, slice `s` already carrying `2^-(bits s)`.
/// `x` is scaled so that `|x| < 2^bits`; every step is exact.
fn slices(x: TtTensor, bits: u32, count: usize) -> Vec<TtTensor> {
    let radix = pow2(bits as i32);
    let mut out = Vec::with_capacity(count);
    let mut rest = x;
    for s in 0..count {
        let top = float::float_trunc(rest.clone());
        out.push(if s == 0 {
            top.clone()
        } else {
            float::float_mul_scalar(top.clone(), scalar(pow2(-(bits as i32) * s as i32)))
        });
        if s + 1 < count {
            rest = float::float_mul_scalar(float::float_sub(rest, top), scalar(radix));
        }
    }
    out
}

/// The product at `Fidelity::Lo`, on the Matrix Unit, bypassing this mode's own hook.
fn lo_matmul(a: TtTensor, b: TtTensor) -> TtTensor {
    crate::matmul_mode::with_matmul_mode(MatmulMode::Fast, || {
        crate::matmul_mode::with_fidelity(Fidelity::Lo, || float::float_matmul(a, b))
    })
}

/// What an operand holds that is not an ordinary finite number, as matrices of 0 and +-1 (any
/// matmul of which is exact), and the part that is.
struct Classes {
    /// 1 where the entry is an infinity.
    infinite: TtTensor,
    zero: TtTensor,
    nonzero: TtTensor,
    /// +-1 at an infinity, 0 elsewhere.
    inf_sign: TtTensor,
    /// +-1 at an infinity or a finite nonzero, 0 at a zero.
    sign: TtTensor,
    /// 1 where the entry is finite and nonzero.
    finite_nonzero: TtTensor,
    /// +-1 at a finite nonzero.
    finite_sign: TtTensor,
    /// NaNs along the reduced axis, per row (left operand) or column (right one).
    nans: TtTensor,
    /// The operand with NaN and infinity replaced by 0, clamped into the domain.
    finite: TtTensor,
}

fn classes(t: TtTensor, axis: usize, bool_dtype: burn_backend::BoolDType) -> Classes {
    let as_float = |x| bool::bool_into_float(x, burn_backend::FloatDType::F32);
    let nan = float::float_is_nan(t.clone(), bool_dtype);
    let inf = float::float_is_inf(t.clone(), bool_dtype);
    let zero = float::float_equal_elem(t.clone(), scalar(0.0), bool_dtype);
    let infinite = as_float(inf.clone());
    // A NaN is no sign and no infinity: it is counted in `nans`, and overrides its entries.
    let clean = float::float_mask_fill(t.clone(), nan.clone(), scalar(0.0));
    let sign = float::float_sign(clean.clone());
    let inf_sign = float::float_mul(sign.clone(), infinite.clone());
    let nonzero = as_float(bool::bool_not(zero.clone()));
    let bound = pow2(DOMAIN);
    let finite = float::float_clamp(
        float::float_mask_fill(clean, inf, scalar(0.0)),
        (-bound).into(),
        bound.into(),
    );
    Classes {
        finite_nonzero: float::float_sub(nonzero.clone(), infinite.clone()),
        finite_sign: float::float_sub(sign.clone(), inf_sign.clone()),
        nans: float::float_sum_dim(as_float(nan), axis),
        zero: as_float(zero),
        infinite,
        nonzero,
        inf_sign,
        sign,
        finite,
    }
}

/// The accurate product if the matmul is to run accurately, `None` if it is not.
pub(crate) fn accurate_matmul(lhs: &TtTensor, rhs: &TtTensor) -> Option<TtTensor> {
    let device = lhs.device;
    if crate::matmul_mode::effective(device) != MatmulMode::Accurate {
        return None;
    }
    let refuse = |why: &str| -> ! {
        fail(
            "float_matmul",
            format_args!(
                "accurate matmul: {why}: lhs=({}), rhs=({})",
                context(lhs),
                context(rhs)
            ),
        )
    };
    // The slicing composes element-wise ops and reductions, which a mesh runs on chip 0 only.
    if !crate::server::supports_exact_matmul(device) {
        refuse("this engine does not support the accurate mode");
    }
    let (l, r) = (lhs.shape().to_vec(), rhs.shape().to_vec());
    if l.len() < 2 || r.len() < 2 {
        refuse("operands need rank >= 2");
    }
    let (m, k) = (l[l.len() - 2], l[l.len() - 1]);
    let (k2, n) = (r[r.len() - 2], r[r.len() - 1]);
    if k != k2 {
        refuse("inner dimensions differ");
    }
    let rank = l.len().max(r.len());
    let lead = |s: &[usize]| -> Vec<usize> {
        let mut v = vec![1; rank - s.len()];
        v.extend_from_slice(s);
        v
    };
    let (ls, rs) = (lead(&l), lead(&r));
    let mut batches = 1usize;
    for d in 0..rank - 2 {
        match (ls[d], rs[d]) {
            (a, b) if a == b || b == 1 => batches *= a,
            (1, b) => batches *= b,
            _ => refuse("batch axes do not broadcast"),
        }
    }
    // No elements, no arithmetic: the ordinary path answers with the right shape.
    if batches == 0 || m == 0 || n == 0 || k == 0 {
        return None;
    }
    if !lhs.is_storable() || !rhs.is_storable() || !crate::server::supports_dram(device) {
        refuse("operands must be resident-storable with GDDR");
    }
    let bool_dtype = burn_backend::get_device_settings::<TtBackend>(&device).bool_dtype;

    // Equal ranks, so the scales broadcast.
    let a = float::float_reshape(lhs.clone(), ls.clone().into());
    let b = float::float_reshape(rhs.clone(), rs.clone().into());
    let (a_axis, b_axis) = (rank - 1, rank - 2);

    let (ca, cb) = (
        classes(a, a_axis, bool_dtype),
        classes(b, b_axis, bool_dtype),
    );
    let (a, b) = (ca.finite.clone(), cb.finite.clone());

    // One power of two per row of `a` and column of `b`, from its largest magnitude.
    let largest = |t: &TtTensor, axis: usize| {
        float::float_clamp_min(
            float::float_max_dim(float::float_abs(t.clone()), axis),
            scalar(pow2(-DOMAIN)),
        )
    };
    let (ra, rb) = (largest(&a, a_axis), largest(&b, b_axis));
    let (ua, ub) = (
        float::float_pow2_floor_inv(ra.clone()),
        float::float_pow2_floor_inv(rb.clone()),
    );
    let (va, vb) = (float::float_pow2_floor(ra), float::float_pow2_floor(rb));

    // Scaled into `[2^6, 2^7)` and `[2^4, 2^5)` at the largest entry, so the first slice has
    // `LHS_BITS` and `RHS_BITS` significant bits.
    let shift = LEAD_SHIFT.load(std::sync::atomic::Ordering::Relaxed);
    let xa = float::float_mul_scalar(
        float::float_mul(a, ua),
        scalar(pow2(LHS_BITS as i32 - 1 + shift)),
    );
    let yb = float::float_mul_scalar(
        float::float_mul(b, ub),
        scalar(pow2(RHS_BITS as i32 - 1 + shift)),
    );
    let a_slices = slices(xa, LHS_BITS, LHS_SLICES);
    let b_slices = slices(yb, RHS_BITS, RHS_SLICES);

    let mut sum: Option<TtTensor> = None;
    for (s, t) in pairs() {
        let part = lo_matmul(a_slices[s].clone(), b_slices[t].clone());
        sum = Some(match sum {
            None => part,
            Some(acc) => float::float_add(acc, part),
        });
    }
    let sum = sum.expect("at least one slice pair");

    // Back by the two powers of two: `a = xa va / 2^6`, `b = yb vb / 2^4`.
    let scaled = float::float_mul(
        float::float_mul(
            sum,
            float::float_mul_scalar(va, scalar(pow2(1 - LHS_BITS as i32 - shift))),
        ),
        float::float_mul_scalar(vb, scalar(pow2(1 - RHS_BITS as i32 - shift))),
    );

    // What the NaNs and infinities do to the sum, by three matmuls of 0 and +-1.
    let mm = |x: &TtTensor, y: &TtTensor| lo_matmul(x.clone(), y.clone());
    let add = |x, y| float::float_add(x, y);
    // `signed`: +1 for each product that is +inf, -1 for each that is -inf. `infinite`: how many are.
    // `undefined`: inf times 0, each a NaN.
    let signed = add(
        mm(&ca.inf_sign, &cb.sign),
        mm(&ca.finite_sign, &cb.inf_sign),
    );
    let infinite = add(
        mm(&ca.infinite, &cb.nonzero),
        mm(&ca.finite_nonzero, &cb.infinite),
    );
    let undefined = add(mm(&ca.infinite, &cb.zero), mm(&ca.zero, &cb.infinite));
    // Both signs among the infinities: `infinite - |signed|` is twice the smaller count.
    let both = float::float_sub(infinite.clone(), float::float_abs(signed.clone()));
    let nan_total = add(add(add(ca.nans, cb.nans), undefined), both);
    let infinity = float::float_mul_scalar(float::float_sign(signed), scalar(f32::INFINITY));
    let has_inf = float::float_greater_elem(infinite, scalar(0.0), bool_dtype);
    let is_nan = float::float_greater_elem(nan_total, scalar(0.0), bool_dtype);
    let out = float::float_mask_fill(
        float::float_mask_where(scaled, has_inf, infinity),
        is_nan,
        scalar(f32::NAN),
    );
    let mut shape: Vec<usize> = ls[..rank - 2]
        .iter()
        .zip(&rs[..rank - 2])
        .map(|(&a, &b)| a.max(b))
        .collect();
    shape.extend([m, n]);
    debug_assert_eq!(out.shape().to_vec(), shape);
    Some(out)
}
