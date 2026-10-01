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
use crate::tensor::Elem;

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
    /// `e^a` (`exp_program`).
    pub const EXP: u32 = 0x103;
    /// `ln a` (`log_program`).
    pub const LOG: u32 = 0x104;
    /// `!a` of a `Bool` tensor (`hardware-coverage.md` D3).
    pub const BOOL_NOT: u32 = 0x105;
    /// `a && b`, `Bool`.
    pub const BOOL_AND: u32 = 0x106;
    /// `a || b`, `Bool`.
    pub const BOOL_OR: u32 = 0x107;
    /// `a != b`, `Bool`.
    pub const BOOL_XOR: u32 = 0x108;
    /// `-a`: the sign bit flipped, NaNs included (S2; 10.2c).
    pub const NEG: u32 = 0x109;
    /// `|a|`: the sign bit cleared, NaNs included.
    pub const ABS: u32 = 0x10a;
    /// `1`, `-1`, `+0` for `a > 0`, `a < 0`, `±0`; a NaN itself.
    pub const SIGN: u32 = 0x10b;
    /// `a.clamp(scalar, scalar2)`, as `f32::clamp`: bounds neither NaN nor
    /// crossed (a program is refused otherwise, as `f32::clamp` panics).
    pub const CLAMP: u32 = 0x10c;
    /// `a.max(scalar)` as Flex's `float_clamp_min`: a NaN on either side gives
    /// the other, and equal values (`±0`) give the scalar.
    pub const CLAMP_MIN: u32 = 0x10d;
    /// `a.min(scalar)`, as [`CLAMP_MIN`] mirrored.
    pub const CLAMP_MAX: u32 = 0x10e;
    /// `a >= 0 ? a : scalar * a`.
    pub const LEAKY_RELU: u32 = 0x10f;
    /// `(scalar * a + scalar2).clamp(0, 1)`, two roundings.
    pub const HARD_SIGMOID: u32 = 0x110;
    /// IEEE comparisons, `F32` with `F32` to `Bool`: `-0 == +0`, a NaN
    /// unordered (every one false but `NE`).
    pub const EQ: u32 = 0x111;
    pub const NE: u32 = 0x112;
    pub const GT: u32 = 0x113;
    pub const GE: u32 = 0x114;
    pub const LT: u32 = 0x115;
    pub const LE: u32 = 0x116;
    /// The same against `scalar`.
    pub const EQ_S: u32 = 0x117;
    pub const NE_S: u32 = 0x118;
    pub const GT_S: u32 = 0x119;
    pub const GE_S: u32 = 0x11a;
    pub const LT_S: u32 = 0x11b;
    pub const LE_S: u32 = 0x11c;
    /// `F32` to `Bool`.
    pub const IS_NAN: u32 = 0x11d;
    pub const IS_INF: u32 = 0x11e;
    /// `mask ? scalar : a`, `a` `F32`, `mask` `Bool` (as `B`, which may be a
    /// row or a column).
    pub const MASK_FILL: u32 = 0x11f;
    /// `mask ? c : a`: `F32`, `Bool`, `F32`, all one shape (ternary).
    pub const MASK_WHERE: u32 = 0x120;
    /// `a >= 0 ? a : b * a`, Flex's `prelu`: `LEAKY_RELU` with the slope a
    /// tensor -- a row of per-channel slopes, broadcast.
    pub const PRELU: u32 = 0x121;
    /// `scalar` everywhere, exact (Flex's `ones` for `powi_scalar(x, 0)`).
    pub const FILL: u32 = 0x122;
    /// `sqrt(a)`, within one ulp of the correctly rounded root (S4, 10.2d).
    pub const SQRT: u32 = 0x123;
    /// `1 / sqrt(a)`, within [`super::RSQRT_BOUND`].
    pub const RSQRT: u32 = 0x124;
    /// `ln(1 + a)`, within [`super::LOG1P_BOUND`].
    pub const LOG1P: u32 = 0x125;
    /// `ln |a|`: the first stage of `pow` (`crate::session::Session::pow`),
    /// within `LOG_BOUND`.
    pub const LOG_ABS: u32 = 0x126;
    /// `pow`'s last stage, ternary: `x`, `y` and `e = e^(y ln|x|)` to `powf`'s
    /// value (`pow_fix`): `e` where nothing is special.
    pub const POW_FIX: u32 = 0x127;
    /// The same with `y` the scalar: `x` and `e`.
    pub const POW_FIX_S: u32 = 0x128;
    /// An `I32` as `F32`, `as f32`'s rounding (Flex's `int_into_float`).
    pub const I32_TO_F32: u32 = 0x129;
}

/// The IEEE comparisons, tensor with tensor, with their scalar forms.
const COMPARES: [(u32, u32); 6] = [
    (kind_sfpu::EQ, kind_sfpu::EQ_S),
    (kind_sfpu::NE, kind_sfpu::NE_S),
    (kind_sfpu::GT, kind_sfpu::GT_S),
    (kind_sfpu::GE, kind_sfpu::GE_S),
    (kind_sfpu::LT, kind_sfpu::LT_S),
    (kind_sfpu::LE, kind_sfpu::LE_S),
];

/// `kind`'s tensor-with-tensor comparison, if it is one of either form.
fn compare_of(kind: u32) -> Option<u32> {
    COMPARES
        .iter()
        .find(|(t, s)| *t == kind || *s == kind)
        .map(|(t, _)| *t)
}

/// What `x (cmp) y` is for two FP32 values by IEEE, as Flex compares.
pub fn ieee_compare(cmp: u32, x: f32, y: f32) -> bool {
    match compare_of(cmp).expect("a comparison") {
        kind_sfpu::EQ => x == y,
        kind_sfpu::NE => x != y,
        kind_sfpu::GT => x > y,
        kind_sfpu::GE => x >= y,
        kind_sfpu::LT => x < y,
        _ => x <= y,
    }
}

/// An op's element types: one per operand, and its output's.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Sig {
    pub inputs: &'static [Elem],
    pub out: Elem,
}

/// What `kind` computes on and what it produces (`crate::tensor::Elem`); `None`
/// for a kind that moves datums whatever they are (`COPY`), whose output is
/// its input's. The session refuses any other operand
/// (`TensorError::Elem`) before choosing a unit.
pub fn elems(kind: u32) -> Option<Sig> {
    use Elem::{Bool, F32};
    let sig = |inputs, out| Some(Sig { inputs, out });
    match kind {
        kind::COPY => None,
        kind_sfpu::BOOL_NOT => sig(&[Bool], Bool),
        kind_sfpu::BOOL_AND | kind_sfpu::BOOL_OR | kind_sfpu::BOOL_XOR => sig(&[Bool, Bool], Bool),
        kind_sfpu::EQ..=kind_sfpu::LE => sig(&[F32, F32], Bool),
        kind_sfpu::EQ_S..=kind_sfpu::IS_INF => sig(&[F32], Bool),
        kind_sfpu::MASK_FILL => sig(&[F32, Bool], F32),
        kind_sfpu::MASK_WHERE => sig(&[F32, Bool, F32], F32),
        kind_sfpu::POW_FIX => sig(&[F32, F32, F32], F32),
        kind_sfpu::I32_TO_F32 => sig(&[Elem::I32], F32),
        _ => sig(&[F32, F32], F32),
    }
}

/// Whether `kind` gives the host's bits exactly or an approximation held to a
/// derived bound -- which decides whether burn-tt's exact mode may run it on
/// the device (`burn-backend-parity.md` 4.3a).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Accuracy {
    /// Flex's bits, every input.
    Exact,
    /// Within a derived bound: `Program::recip`'s ulp, `EXP_BOUND`, ...
    Approximate,
}

/// [`Accuracy`] of `kind`.
pub fn accuracy(kind: u32) -> Accuracy {
    match kind {
        kind_sfpu::RECIP
        | kind_sfpu::DIV
        | kind_sfpu::DIV_SCALAR
        | kind_sfpu::EXP
        | kind_sfpu::LOG
        | kind_sfpu::SQRT..=kind_sfpu::LOG_ABS => Accuracy::Approximate,
        _ => Accuracy::Exact,
    }
}

/// `e^x` of `x` into `d`, every register but `x` and `d` scratch.
///
/// `n = round(x log2 e)` by the magic-number rounding (adding `1.5 * 2^23`
/// leaves `n` in the low mantissa bits, round to nearest even), `r = x - n ln 2`
/// by Cody and Waite's two-part `ln 2` (the high part has nine significant
/// bits, so `n * ln2_hi` is exact for `|n| <= 128`), `|r| <= ln 2 / 2`; then
/// `e^r` by its degree-7 Taylor polynomial in Horner form, and `2^n` added to
/// the exponent field as an integer.
///
/// Error, as a fraction of `e^x`: the polynomial's remainder `r^8/8! * e^|r|`
/// is below `7.4e-9`; Horner's seven roundings, with `|r| <= 0.347` damping
/// all but the last two, contribute below `1.7 * 2^-24` (`1.0e-7`); the
/// reduction's rounding of `r` (one rounding of the low product, `|n ln2_lo|
/// < 2.8e-2`) moves `r` by under `2^-24 * |r| + 2^-30`, which `e^r` carries
/// one for one (`2.3e-8`). In all under [`EXP_BOUND`] `= 1.3e-7` of `e^x`
/// (about two ulps where the result's mantissa is near 2, one near 1).
/// Outside `[-87.34, 88.72]` the result is `+0`
/// (it would be denormal, which flushes) or `+inf`; a NaN stays one.
pub fn exp_program(p: &mut Program, x: LReg, d: LReg) {
    use LReg as R;
    let regs: Vec<LReg> = [R::L0, R::L1, R::L2, R::L3, R::L4, R::L5, R::L6, R::L7]
        .into_iter()
        .filter(|r| *r != x && *r != d)
        .collect();
    let (k, magic, t, nf, r, c) = (regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
    p.loadi(k, std::f32::consts::LOG2_E);
    p.loadi_bits(magic, 0x4b40_0000);
    p.mad(x, k, magic, t);
    p.sub(t, magic, nf);
    p.loadi_bits(k, 0x3f31_8000); // ln2_hi = 0.693359375
    p.nmad(nf, k, x, r);
    p.loadi_bits(k, 0xb95e_8083); // ln2_lo = -2.12194440e-4
    p.nmad(nf, k, r, r);
    // e^r = 1 + r(1 + r(1/2 + r(1/6 + r(1/24 + r(1/120 + r(1/720 + r/5040))))))
    p.loadi(d, 1.0 / 5040.0);
    for coeff in [1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5] {
        p.loadi(c, coeff);
        p.mad(d, r, c, d);
    }
    p.mad(d, r, LReg::ONE, d);
    p.mad(d, r, LReg::ONE, d);
    // `n = bits(t) - bits(magic)`, into the exponent field.
    p.isub_from(t, magic);
    p.shl(magic, 23, magic);
    p.iadd(magic, d);
    // The range: below `ln 2^-126` a zero, above `ln f32::MAX` infinity
    // (which a positive NaN also passes), and a NaN restored.
    p.loadi_bits(k, 0xc2ae_ac50); // -87.336544
    p.if_(Cond::Less(x, k), |p| p.mov(LReg::ZERO, d));
    // `0x42b1_7218` (88.72284) is the first float above `ln f32::MAX`: from it
    // on, `n = 128` would carry the exponent field into 255 -- an infinity's,
    // or with mantissa bits a NaN's. (It was `>` alone until 10.2d's `pow`
    // met `e^(ln f32::MAX)`, whose `z` rounds to exactly this.)
    p.loadi_bits(k, 0x42b1_7218); // 88.72284
    p.if_(Cond::LessEq(k, x), |p| p.loadi_bits(d, 0x7f80_0000));
    p.abs(x, t);
    p.loadi_bits(k, 0x7f80_0000);
    p.if_(Cond::Less(k, t), |p| p.loadi_bits(d, 0x7fc0_0000));
}

/// [`exp_program`]'s derived bound, relative to `e^x`.
pub const EXP_BOUND: f64 = 1.3e-7;

/// [`log_program`]'s derived bound, relative to `ln x`.
pub const LOG_BOUND: f64 = 7.12 / 16_777_216.0;

/// `ln x` of `x` into `d`, every register but `x` and `d` scratch.
///
/// `x = 2^e m` with `m` in `[sqrt(2)/2, sqrt(2))` (`SFPEXEXP`, `SFPSETEXP`, and
/// a halving where `m` comes out above `sqrt 2`); `f = m - 1` exactly
/// (Sterbenz); `ln m = 2 atanh(s)`, `s = f / (2 + f)`, `|s| <= 0.1716`, by
/// `2s(1 + s^2/3 + s^4/5 + s^6/7 + s^8/9)` -- the next term, `s^10/11`, is
/// below `2.0e-9` of the sum -- and `ln x = e ln2_hi + (e ln2_lo + ln m)`.
///
/// Error, relative, in units of `u = 2^-24`: `fl(2 + f)` is within `u` of
/// `2 + f`, and `s` within 1.5 ulps (`3u`) of `f / fl(2 + f)` (`divide`), so
/// `4u` of `f/(2+f)`; `ln m` is `2s (1 + O(s^2))` and carries that one for one
/// (`4.12u` with the `s^2 < 0.03` term), the series' own roundings damped by
/// `s^2` to under `0.04u`, plus `u` for the fma that adds them; the two final
/// fmas round once each, and `e ln 2` and `ln m` never cancel to less than
/// `|ln m|` (they share a sign unless `|e| = 1`, where `|ln x| >= 0.34 >=
/// |ln m|`). Under [`LOG_BOUND`] `= 7.12u` (`4.3e-7`) of `ln x` in all. `ln(±0)` (and
/// a denormal, which flushes) is `-inf`, `ln` of a negative number NaN,
/// `ln(+inf) = +inf`, a NaN stays one.
pub fn log_program(p: &mut Program, x: LReg, d: LReg) {
    use LReg as R;
    let regs: Vec<LReg> = [R::L0, R::L1, R::L2, R::L3, R::L4, R::L5, R::L6, R::L7]
        .into_iter()
        .filter(|r| *r != x && *r != d)
        .collect();
    let (e, m, f, den, y, q) = (regs[0], regs[1], regs[2], regs[3], regs[4], regs[5]);
    p.exponent(x, true, e);
    p.set_exponent(x, 127, m);
    // Positive: `SFPSETEXP` keeps the sign, and a negative `x` is a NaN below.
    p.set_sign(m, false, m);
    p.loadi_bits(f, 0x3fb5_04f3); // sqrt(2)
    p.if_(Cond::Less(f, m), |p| {
        p.set_exponent(m, 126, m);
        p.iadd_imm(e, 1, e);
    });
    p.sub(m, LReg::ONE, f);
    p.loadi(den, 2.0);
    p.add(f, den, den);
    // `s = f / (2 + f)`: `y = 1/den` uses `m` and `q` as scratch, `d` holds
    // `f32::MAX`; then the quotient into `q`, `m` and `d` scratch, `inf` in
    // `den` once it is spent.
    p.loadi_bits(d, f32::MAX.to_bits());
    p.recip(den, y, m, q, d);
    p.mul(f, y, q);
    p.abs(q, m);
    p.if_(Cond::Less(LReg::ZERO, m), |p| {
        p.nmad(den, q, f, m);
        p.mad(m, y, q, q);
    });
    // `s` in `q`; `s^2` into `f`; the series into `d`.
    p.mul(q, q, f);
    p.loadi(d, 1.0 / 9.0);
    for coeff in [1.0 / 7.0, 1.0 / 5.0, 1.0 / 3.0] {
        p.loadi(m, coeff);
        p.mad(d, f, m, d);
    }
    // ln m = 2(s + s * (s^2 * series)).
    p.mul(d, f, d);
    p.mad(q, d, q, d);
    p.add(d, d, d);
    // `e` as a float: `bits(1.5 * 2^23) + e` is `1.5 * 2^23 + e` exactly.
    p.loadi_bits(m, 0x4b40_0000);
    p.iadd(m, e);
    p.sub(e, m, e);
    p.loadi_bits(m, 0xb95e_8083); // ln2_lo
    p.mad(e, m, d, d);
    p.loadi_bits(m, 0x3f31_8000); // ln2_hi
    p.mad(e, m, d, d);
    // The special cases, each its own lanes (the total order of `SFPGT`).
    p.abs(x, m);
    // A zero or a denormal (which the arithmetic flushes): `-inf`.
    p.loadi_bits(f, 0x0080_0000);
    p.if_(Cond::Less(m, f), |p| p.loadi_bits(d, 0xff80_0000));
    // Anything else negative -- below minus the largest denormal in the total
    // order, `-inf` and negative NaNs included (whose `SFPABS` stays negative
    // and so took the branch above first): NaN.
    p.loadi_bits(f, 0x807f_ffff);
    p.if_(Cond::Less(x, f), |p| p.loadi_bits(d, 0x7fc0_0000));
    // `+inf`: `+inf`; a positive NaN: NaN.
    p.loadi_bits(f, f32::MAX.to_bits());
    p.if_(Cond::Less(f, x), |p| p.loadi_bits(d, 0x7f80_0000));
    p.loadi_bits(f, 0x7f80_0000);
    p.if_(Cond::Less(f, x), |p| p.loadi_bits(d, 0x7fc0_0000));
}

/// [`sqrt_program`]'s bound for the reciprocal root, relative.
pub const RSQRT_BOUND: f64 = 4.1 / 16_777_216.0;

/// `sqrt x` (or, `inverse`, `1/sqrt x`) of `x` into `d`; `x` kept, every
/// other register scratch.
///
/// The seed is the bit trick `0x5f3759df - bits(x) / 2` (within `e0 <=
/// 0.0344` of `1/sqrt x` for every positive normal `x`); three Newton steps
/// `y += y/2 (1 - x y^2)`, each taking `e` to `1.5 e^2` plus its four
/// roundings: `e1 <= 1.78e-3`, `e2 <= 4.8e-6`, `e3 <= 3.5e-11 + 4u` (`u =
/// 2^-24`) -- `x y` is formed before `y` multiplies it again, so `x y^2` never
/// leaves the normal range. So `1/sqrt x` within [`RSQRT_BOUND`] `= 4.1u`. The
/// root is `s = x y`, then one correction `s += (y/2)(x - s^2)` with the
/// residual in one rounding (`SFPMAD`): `s` within `5.1u` before it, so within
/// half an ulp plus `O(25u^2)` after -- at most one ulp from the correctly
/// rounded root, as `Program::recip` is from its reciprocal.
///
/// `sqrt(±0) = ±0`, `sqrt(+inf) = +inf`, anything negative (a negative
/// denormal included, as Flex has it) NaN, a NaN stays one; a positive
/// denormal flushes, so its root is `+0` (numerics row D). `1/sqrt`: `±inf`
/// for `±0`, `+0` for `+inf`, NaN as `sqrt`.
pub fn sqrt_program(p: &mut Program, x: LReg, d: LReg, inverse: bool) {
    use LReg as R;
    let regs: Vec<LReg> = [R::L0, R::L1, R::L2, R::L3, R::L4, R::L5, R::L6, R::L7]
        .into_iter()
        .filter(|r| *r != x && *r != d)
        .collect();
    let (y, t, h, c) = (regs[0], regs[1], regs[2], regs[3]);
    // The seed: `magic - (bits(x) >> 1)`.
    p.mov(x, y);
    p.loadi_bits(c, (-1i32) as u32);
    p.shr_by(c, y);
    p.loadi_bits(c, 0x5f37_59df);
    p.isub_from(c, y);
    p.loadi(c, 0.5);
    for _ in 0..3 {
        // `h = x y`, `t = 1 - h y`, `y += (y/2) t`.
        p.mul(x, y, h);
        p.nmad(h, y, LReg::ONE, t);
        p.mul(y, c, h);
        p.mad(h, t, y, y);
    }
    if inverse {
        p.mov(y, d);
    } else {
        // `s = x y`; `s += (y/2)(x - s^2)`.
        p.mul(x, y, d);
        p.nmad(d, d, x, t);
        p.mul(y, c, h);
        p.mad(h, t, d, d);
    }
    // The special lanes, in the total order: `±0` and a positive denormal
    // (`|x| < 2^-126`), `+inf`, then everything negative but `-0`, and NaNs.
    p.abs(x, t);
    p.loadi_bits(c, 0x0080_0000);
    p.if_(Cond::Less(t, c), |p| {
        if inverse {
            p.loadi_bits(h, 0x7f80_0000);
            p.copy_sign(h, x, d);
        } else {
            // `±0` (a positive denormal flushes to `+0`).
            p.loadi_bits(h, 0x8000_0000);
            p.and(x, h, d);
        }
    });
    p.loadi_bits(c, 0x7f80_0000);
    p.if_(Cond::LessEq(c, x), |p| {
        if inverse {
            p.mov(LReg::ZERO, d);
        } else {
            p.mov(c, d);
        }
    });
    // Negative and not `-0`: below `-0` in the total order; a NaN above
    // `+inf` or below `-inf`.
    p.loadi_bits(c, 0x8000_0000);
    p.if_(Cond::Less(x, c), |p| p.loadi_bits(d, 0x7fc0_0000));
    p.loadi_bits(c, 0x7f80_0000);
    p.if_(Cond::Less(c, x), |p| p.loadi_bits(d, 0x7fc0_0000));
}

/// [`log1p_program`]'s derived bound, relative to `ln(1 + x)`.
pub const LOG1P_BOUND: f64 = 11.12 / 16_777_216.0;

/// `ln(1 + x)` of `x` into `d`, `x` read as raw bits and spilled to `Dst` at
/// `spill` (it outlives [`log_program`], which takes every register).
///
/// Kahan's: `u = fl(1 + x)`; where `u = 1` (`|x| <= 2^-24`, where
/// `ln(1 + x) = x (1 - x/2 + ...)` is `x` to under `2^-25` relative) the
/// result is `x` itself, bits and all; elsewhere `ln(u) * x / (u - 1)`. The
/// quotient `g(u) = ln(u)/(u - 1)` varies slowly enough that evaluating it at
/// `u` instead of `1 + x` costs at most `|g'/g| |u - (1 + x)| <= u` (`u =
/// 2^-24`, the rounding of `1 + x` being at most half an ulp of `u`), and `u -
/// 1` is exact (Sterbenz below 2; `1` a multiple of `u`'s ulp up to `2^24`).
/// Error, in units of `u`: `ln u` within `7.12` (`LOG_BOUND`), the quotient
/// `x / (u - 1)` within `2` (`divide`, one ulp), the product `1`, the
/// evaluation point `1`: under [`LOG1P_BOUND`] `= 11.12u` (`6.6e-7`). From `x =
/// 2^24` the result is `ln u` alone (`u = x`, and `ln(1 + x)` is `ln x` to
/// within `1/(x ln x) < 2^-27` relative).
/// `log1p(-1) = -inf`, below `-1` NaN, `log1p(+inf) = +inf`, a NaN stays one.
pub fn log1p_program(p: &mut Program, x: LReg, d: LReg, spill: u32) {
    use LReg as R;
    assert!(
        x == R::L0 && d == R::L7,
        "log1p_program's registers are fixed"
    );
    p.store(x, Format::Int32, spill);
    let u = R::L1;
    p.add(x, LReg::ONE, u);
    log_program(p, u, d);
    p.load(x, Format::Int32, spill);
    // `den = u - 1`; `q = x / den`; `d = ln(u) q`.
    let (den, y, q) = (R::L2, R::L3, R::L4);
    p.sub(u, LReg::ONE, den);
    p.loadi_bits(R::L6, f32::MAX.to_bits());
    p.recip(den, y, R::L4, R::L5, R::L6);
    p.loadi_bits(R::L6, 0x7f80_0000);
    divide(p, x, den, y, q, R::L5, u, R::L6);
    // `ln u` in `d` survives: `divide` and `recip` use none of it. From
    // `x = 2^24` on, `u = x` and `ln(1 + x) = ln(x) (1 + O(1/x))`: `ln u` as it
    // is -- the quotient, `x / x`, would need `1/x`, which flushes from `2^126`.
    p.loadi_bits(y, 0x4b80_0000); // 2^24
    p.if_(Cond::Less(x, y), |p| p.mul(d, q, d));
    // Where `u = 1`: `x`. `u - 1` is then exactly `+0`.
    p.if_(Cond::Eq0(den), |p| p.mov(x, d));
    // `+inf`: `+inf` (the quotient `inf/inf` would be NaN).
    p.loadi_bits(y, 0x7f80_0000);
    p.if_(Cond::LessEq(y, x), |p| {
        p.if_(Cond::LessEq(x, y), |p| p.mov(y, d))
    });
}

/// `n`, a two's-complement `i32`, as FP32 in place, rounding to nearest even
/// as `as f32` does: the magnitude made sign-magnitude for `SFPCAST`, the sign
/// put back; `i32::MIN`, whose magnitude has no 31-bit form, by name. `t`, `c`
/// scratch.
fn i32_to_float(p: &mut Program, n: LReg, t: LReg, c: LReg) {
    p.mov(n, t);
    p.if_(Cond::Lt0(n), |p| p.isub_from(LReg::ZERO, t));
    p.sm32_to_float(t, t);
    p.if_(Cond::Lt0(n), |p| p.neg(t, t));
    p.loadi_bits(c, 0x8000_0000);
    p.xor(n, c);
    p.if_(Cond::Eq0(c), |p| p.loadi(t, -2_147_483_648.0));
    p.mov(t, n);
}

/// `pow`'s bound at `(x, y)`, relative: the chain computes `e^z`,
/// `z = y ln|x|`, and `ln|x|` (within `LOG_BOUND`) and the product's rounding
/// (`2^-24`) move `z` by `|z| (LOG_BOUND + 2^-24)` -- which `e^z` turns into
/// that relative error -- on top of `exp`'s own `EXP_BOUND`; the factor
/// `1.01` covers the products of the small terms. So a large `|z|` (a result
/// near the range's ends) carries a large bound: `4.3e-5` at `|z| = 88`.
pub fn pow_bound(x: f32, y: f32) -> f64 {
    // `0 ln 0` and the like: a special value, exact.
    let z = (y as f64 * (x.abs() as f64).ln()).abs();
    let z = if z.is_finite() { z } else { 0.0 };
    z * (LOG_BOUND + 1.0 / 16_777_216.0) * 1.01 + EXP_BOUND
}

/// `powf`'s special values over `e = e^(y ln|x|)`, the chain's general case
/// (`crate::session::Session::pow`: `LOG_ABS`, a multiply, `EXP`, then this):
/// `x` in `L0`, `y` in `L1`, `e` in `L7` (raw bits), the result in `L7`. One
/// program would not fit a role's program slot (`mailbox::PROGRAM_MAX`), so
/// the stages are ops; the arithmetic, and so [`pow_bound`], are the same.
///
/// Each scope over the last: a negative finite non-zero base gives NaN for a
/// non-integer `y` and is negated for an odd one (`-inf` and `-0` take only
/// the sign: their magnitudes, `inf` and `0` from `e^(±inf)`, are already
/// right); `y = ±inf` is `1` at `|x| = 1`, else `+inf` where `|x| > 1` meets
/// `y > 0` or `|x| < 1` meets `y < 0`, else `+0`; a NaN anywhere NaN; `x = 1`
/// gives `1`, and `y = ±0` gives `1`, whatever the other. `y` is an integer
/// where `|y| >= 2^23`, or where `|y| + 2^23` rounded back is `|y|`; odd where
/// the integer's last bit is (`|y| < 2^24`).
pub fn pow_fix(p: &mut Program) {
    use LReg as R;
    let (x, y, res, ax, ay, flags, t, c) = (R::L0, R::L1, R::L7, R::L2, R::L3, R::L4, R::L5, R::L6);
    p.loadi_bits(c, 0x7fff_ffff);
    p.and(x, c, ax);
    p.and(y, c, ay);
    // `flags`: 0 not an integer, 1 an even integer, 3 an odd one.
    p.mov(R::ZERO, flags);
    p.loadi_bits(c, 0x4b00_0000); // 2^23
    p.if_(Cond::Less(ay, c), |p| {
        p.add(ay, c, t);
        p.sub(t, c, c);
        p.if_(Cond::LessEq(c, ay), |p| {
            p.if_(Cond::LessEq(ay, c), |p| {
                p.loadi_bits(flags, 1);
                p.loadi_bits(c, 1);
                p.and(t, c, c);
                p.if_(Cond::Ne0(c), |p| p.loadi_bits(flags, 3));
            })
        });
    });
    p.loadi_bits(c, 0x4b00_0000);
    p.if_(Cond::LessEq(c, ay), |p| {
        p.loadi_bits(t, 0x7f80_0000);
        p.if_(Cond::Less(ay, t), |p| {
            p.loadi_bits(flags, 1);
            p.loadi_bits(t, 0x4b80_0000); // 2^24
            p.if_(Cond::Less(ay, t), |p| {
                p.loadi_bits(c, 1);
                p.and(ay, c, c);
                p.if_(Cond::Ne0(c), |p| p.loadi_bits(flags, 3));
            });
        });
    });
    // A negative base.
    p.loadi_bits(c, 0x7f80_0000);
    p.if_(Cond::Lt0(x), |p| {
        p.if_else(
            Cond::Eq0(flags),
            // Not an integer: NaN, unless the base is `-0` or `-inf`.
            |p| {
                p.if_(Cond::Ne0(ax), |p| {
                    p.if_(Cond::Less(ax, c), |p| p.loadi_bits(res, 0x7fc0_0000))
                })
            },
            |p| {
                p.loadi_bits(t, 2);
                p.and(flags, t, t);
                p.if_(Cond::Ne0(t), |p| p.neg(res, res));
            },
        )
    });
    // `y = ±inf`.
    p.if_(Cond::LessEq(c, ay), |p| {
        p.if_(Cond::LessEq(ay, c), |p| {
            p.mov(R::ZERO, res);
            p.if_(Cond::Lt0(y), |p| {
                p.if_(Cond::Less(ax, R::ONE), |p| p.mov(c, res))
            });
            p.if_(Cond::Gte0(y), |p| {
                p.if_(Cond::Less(R::ONE, ax), |p| p.mov(c, res))
            });
            p.if_(Cond::LessEq(ax, R::ONE), |p| {
                p.if_(Cond::LessEq(R::ONE, ax), |p| p.mov(R::ONE, res))
            });
        })
    });
    // NaNs.
    p.if_(Cond::Less(c, ax), |p| p.loadi_bits(res, 0x7fc0_0000));
    p.if_(Cond::Less(c, ay), |p| p.loadi_bits(res, 0x7fc0_0000));
    // `1^y` and `x^0`.
    p.if_(Cond::LessEq(x, R::ONE), |p| {
        p.if_(Cond::LessEq(R::ONE, x), |p| p.mov(R::ONE, res))
    });
    p.if_(Cond::Eq0(ay), |p| p.mov(R::ONE, res));
}

/// Does the data mover implement `kind`?
pub fn mover_has(kind: u32) -> bool {
    (1..=kind::LAST).contains(&kind)
}

/// What operands `kind` takes, if the SFPU has it.
pub fn operands(kind: u32) -> Option<Operands> {
    Some(match kind {
        kind::ADD
        | kind::SUB
        | kind::MUL
        | kind::RELU_BACKWARD
        | kind_sfpu::DIV
        | kind_sfpu::BOOL_AND
        | kind_sfpu::BOOL_OR
        | kind_sfpu::BOOL_XOR
        | kind_sfpu::EQ..=kind_sfpu::LE
        | kind_sfpu::MASK_FILL
        | kind_sfpu::PRELU
        | kind_sfpu::POW_FIX_S => Operands::Binary,
        kind_sfpu::POW_FIX => Operands::Ternary,
        kind_sfpu::MASK_WHERE => Operands::Ternary,
        kind::MUL_SCALAR
        | kind::ADD_SCALAR
        | kind::RELU
        | kind_sfpu::RECIP
        | kind_sfpu::DIV_SCALAR
        | kind_sfpu::EXP
        | kind_sfpu::LOG
        | kind_sfpu::BOOL_NOT
        | kind_sfpu::NEG..=kind_sfpu::HARD_SIGMOID
        | kind_sfpu::EQ_S..=kind_sfpu::IS_INF
        | kind_sfpu::FILL..=kind_sfpu::LOG_ABS
        | kind_sfpu::I32_TO_F32 => Operands::Unary,
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

/// Where `|b| > 2^100`, `a` and `b` both times `2^-64` (exact; the quotient
/// unchanged): the reciprocal and the correction then stay in range. An `a`
/// that flushes doing so had a quotient under `2^-162`, which flushes anyway.
/// `t`, `c` scratch.
fn scale_large_divisor(p: &mut Program, a: LReg, b: LReg, t: LReg, c: LReg) {
    p.abs(b, t);
    p.loadi_bits(c, 0x7180_0000); // 2^100
    p.if_(Cond::Less(c, t), |p| {
        p.loadi_bits(c, 0x1f80_0000); // 2^-64
        p.mul(a, c, a);
        p.mul(b, c, b);
    });
}

/// How a binary op's second operand meets the first.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Broadcast {
    /// The same shape.
    None,
    /// `[1, cols]`: its one row added (or subtracted, ...) to every row.
    Row,
    /// `[rows, 1]`: its one column to every column.
    Col,
}

/// The kinds that take a broadcast second operand.
pub fn broadcasts(kind: u32) -> bool {
    matches!(
        kind,
        kind::ADD
            | kind::SUB
            | kind::MUL
            | kind_sfpu::DIV
            | kind_sfpu::BOOL_AND
            | kind_sfpu::BOOL_OR
            | kind_sfpu::BOOL_XOR
            | kind_sfpu::EQ..=kind_sfpu::LE | kind_sfpu::MASK_FILL | kind_sfpu::PRELU
    )
}

/// `out = a (kind) b` for one row group, `a` in `L0`, `b` from `Dst` at `b_at`,
/// the result stored at `out_at`. `DIV` needs `f32::MAX` in `L6` and `+inf`
/// in `L7` ([`binary_constants`]).
fn binary_body(p: &mut Program, kind: u32, a_at: u32, b_at: u32, out_at: u32) {
    // A `Bool` is `0`/`1` as an integer -- a denormal to FP32's load and
    // store, which would flush it -- so the logic ops move raw bits.
    // The exact ops of S2 move raw bits too: an FP32 store would flush a
    // denormal Flex keeps.
    let fmt = match kind {
        kind::ADD | kind::SUB | kind::MUL | kind_sfpu::DIV => Format::Fp32,
        _ => Format::Int32,
    };
    p.load(LReg::L0, fmt, a_at);
    p.load(LReg::L1, fmt, b_at);
    let out = match kind {
        kind_sfpu::BOOL_AND => {
            p.and(LReg::L0, LReg::L1, LReg::L2);
            LReg::L2
        }
        kind_sfpu::BOOL_OR => {
            p.or(LReg::L0, LReg::L1, LReg::L2);
            LReg::L2
        }
        kind_sfpu::BOOL_XOR => {
            p.mov(LReg::L0, LReg::L2);
            p.xor(LReg::L1, LReg::L2);
            LReg::L2
        }
        k if compare_of(k).is_some() => {
            compare_body(p, k, true);
            LReg::L2
        }
        // As `LEAKY_RELU`, the slope `b`'s lane.
        kind_sfpu::PRELU => {
            p.mul(LReg::L0, LReg::L1, LReg::L2);
            p.and(LReg::L0, LReg::L4, LReg::L3);
            p.if_(Cond::Gte0(LReg::L0), |p| {
                p.if_(Cond::LessEq(LReg::L3, LReg::L5), |p| {
                    p.mov(LReg::L0, LReg::L2)
                })
            });
            p.if_(Cond::Eq0(LReg::L3), |p| p.mov(LReg::L0, LReg::L2));
            LReg::L2
        }
        // `mask ? value : a`, the value in `L3` (`binary_constants`).
        kind_sfpu::MASK_FILL => {
            p.mov(LReg::L0, LReg::L2);
            p.if_(Cond::Ne0(LReg::L1), |p| p.mov(LReg::L3, LReg::L2));
            LReg::L2
        }
        kind::ADD => {
            p.add(LReg::L0, LReg::L1, LReg::L2);
            LReg::L2
        }
        kind::SUB => {
            p.sub(LReg::L0, LReg::L1, LReg::L2);
            LReg::L2
        }
        kind::MUL => {
            p.mul(LReg::L0, LReg::L1, LReg::L2);
            LReg::L2
        }
        kind_sfpu::DIV => {
            scale_large_divisor(p, LReg::L0, LReg::L1, LReg::L3, LReg::L4);
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
            LReg::L5
        }
        _ => unreachable!("kind {kind:#x} is not a broadcastable binary op"),
    };
    p.store(out, fmt, out_at);
}

/// The constants [`binary_body`] needs for `kind`, loaded once.
fn binary_constants(p: &mut Program, kind: u32, scalars: [f32; 2]) {
    if kind == kind_sfpu::DIV {
        p.loadi_bits(LReg::L6, f32::MAX.to_bits());
        p.loadi_bits(LReg::L7, 0x7f80_0000);
    }
    if compare_of(kind).is_some() || kind == kind_sfpu::PRELU {
        compare_constants(p);
    }
    if kind == kind_sfpu::MASK_FILL {
        p.loadi_bits(LReg::L3, scalars[0].to_bits());
    }
}

/// `L4 = 0x7fff_ffff` (the magnitude mask), `L5 = +inf`, `L6 = 1` (a `Bool`'s
/// true): what the IEEE comparisons use.
fn compare_constants(p: &mut Program) {
    p.loadi_bits(LReg::L4, 0x7fff_ffff);
    p.loadi_bits(LReg::L5, 0x7f80_0000);
    p.loadi_bits(LReg::L6, 1);
}

/// `L2 = x (cmp) y` as a `Bool`, by IEEE, `x` in `L0` and `y` in `L1` as raw
/// bits (clobbered), with [`compare_constants`] loaded. `SFPGT`/`SFPLE` order
/// sign-magnitude (`-0 < +0`, NaNs ranked), so both are first made canonical
/// -- a zero of either sign `+0` -- and a lane where either is a NaN takes the
/// unordered answer (false; true for `NE`). `y_may_be_nan` is false where the
/// caller has settled `y` (a scalar, on the host). Bit and flag operations
/// only: no arithmetic, so a denormal compares as itself, as on the host.
fn compare_body(p: &mut Program, kind: u32, y_may_be_nan: bool) {
    let cmp = compare_of(kind).expect("a comparison");
    let (x, y, out, ax, ay, one) = (LReg::L0, LReg::L1, LReg::L2, LReg::L3, LReg::L7, LReg::L6);
    let (mask, inf) = (LReg::L4, LReg::L5);
    p.and(x, mask, ax);
    p.if_(Cond::Eq0(ax), |p| p.mov(LReg::ZERO, x));
    p.and(y, mask, ay);
    p.if_(Cond::Eq0(ay), |p| p.mov(LReg::ZERO, y));
    let (unordered, holds) = if cmp == kind_sfpu::NE {
        (one, LReg::ZERO)
    } else {
        (LReg::ZERO, one)
    };
    p.mov(unordered, out);
    let ordered = move |p: &mut Program| match cmp {
        kind_sfpu::EQ | kind_sfpu::NE => p.if_(Cond::LessEq(x, y), |p| {
            p.if_(Cond::LessEq(y, x), |p| p.mov(holds, out))
        }),
        kind_sfpu::GT => p.if_(Cond::Less(y, x), |p| p.mov(holds, out)),
        kind_sfpu::GE => p.if_(Cond::LessEq(y, x), |p| p.mov(holds, out)),
        kind_sfpu::LT => p.if_(Cond::Less(x, y), |p| p.mov(holds, out)),
        _ => p.if_(Cond::LessEq(x, y), |p| p.mov(holds, out)),
    };
    p.if_(Cond::LessEq(ax, inf), |p| {
        if y_may_be_nan {
            p.if_(Cond::LessEq(ay, inf), ordered);
        } else {
            ordered(p);
        }
    });
}

/// `bits` with a zero of either sign made `+0`: how a scalar meets the
/// canonical lanes of [`compare_body`].
fn canonical(bits: u32) -> u32 {
    if bits & 0x7fff_ffff == 0 {
        0
    } else {
        bits
    }
}

/// The program for an exact S2 op (10.2c) over a tile, or `None` for a kind
/// that is not one -- or a `CLAMP` whose bounds `f32::clamp` would refuse.
fn exact_program(p: &mut Program, kind: u32, [s, s2]: [f32; 2]) -> Option<Operands> {
    use kind_sfpu::*;
    let (x, out, ax) = (LReg::L0, LReg::L2, LReg::L3);
    let (mask, inf) = (LReg::L4, LReg::L5);
    let (load, store) = (
        move |p: &mut Program, o: u32| p.load(x, Format::Int32, A_ROW + o),
        move |p: &mut Program, o: u32| p.store(out, Format::Int32, OUT_ROW + o),
    );
    match kind {
        NEG | ABS => {
            p.loadi_bits(mask, 0x7fff_ffff);
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                if kind == NEG {
                    p.neg(x, out);
                } else {
                    p.and(x, mask, out);
                }
                store(p, o);
            });
        }
        SIGN => {
            p.loadi_bits(mask, 0x7fff_ffff);
            p.loadi_bits(inf, 0x7f80_0000);
            p.loadi(LReg::L6, 1.0);
            p.loadi(LReg::L7, -1.0);
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                // A NaN is itself.
                p.mov(x, out);
                p.and(x, mask, ax);
                p.if_(Cond::LessEq(ax, inf), |p| {
                    p.if_else(
                        Cond::Eq0(ax),
                        |p| p.mov(LReg::ZERO, out),
                        |p| {
                            p.if_else(
                                Cond::Lt0(x),
                                |p| p.mov(LReg::L7, out),
                                |p| p.mov(LReg::L6, out),
                            )
                        },
                    )
                });
                store(p, o);
            });
        }
        CLAMP_MIN | CLAMP_MAX => {
            // A NaN scalar: the other side, `x`, everywhere.
            if s.is_nan() {
                p.for_each_row_group(64, |p, o| {
                    load(p, o);
                    p.mov(x, out);
                    store(p, o);
                });
                return Some(Operands::Unary);
            }
            p.loadi_bits(mask, 0x7fff_ffff);
            p.loadi_bits(inf, 0x7f80_0000);
            p.loadi_bits(LReg::L1, s.to_bits());
            p.loadi_bits(LReg::L7, canonical(s.to_bits()));
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                // The scalar, unless `x` is not NaN and beyond it.
                p.mov(LReg::L1, out);
                p.and(x, mask, ax);
                p.mov(x, LReg::L6);
                p.if_(Cond::Eq0(ax), |p| p.mov(LReg::ZERO, LReg::L6));
                p.if_(Cond::LessEq(ax, inf), |p| {
                    let beyond = if kind == CLAMP_MIN {
                        Cond::Less(LReg::L7, LReg::L6)
                    } else {
                        Cond::Less(LReg::L6, LReg::L7)
                    };
                    p.if_(beyond, |p| p.mov(x, out));
                });
                store(p, o);
            });
        }
        CLAMP => {
            if s.is_nan() || s2.is_nan() || s > s2 {
                return None;
            }
            use super::ConfigLReg;
            // `f32::clamp`: below the low bound it, above the high one it,
            // else `x` -- a NaN, and a zero equal to a bound, included.
            p.constant(ConfigLReg::L11, s2.to_bits());
            p.constant(ConfigLReg::L12, canonical(s2.to_bits()));
            p.loadi_bits(mask, 0x7fff_ffff);
            p.loadi_bits(inf, 0x7f80_0000);
            p.loadi_bits(LReg::L1, s.to_bits());
            p.loadi_bits(LReg::L7, canonical(s.to_bits()));
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                p.mov(x, out);
                p.and(x, mask, ax);
                p.mov(x, LReg::L6);
                p.if_(Cond::Eq0(ax), |p| p.mov(LReg::ZERO, LReg::L6));
                p.if_(Cond::LessEq(ax, inf), |p| {
                    p.if_(Cond::Less(LReg::L6, LReg::L7), |p| p.mov(LReg::L1, out));
                    p.if_(Cond::Less(ConfigLReg::L12.lreg(), LReg::L6), |p| {
                        p.mov(ConfigLReg::L11.lreg(), out)
                    });
                });
                store(p, o);
            });
        }
        LEAKY_RELU => {
            // `x >= 0` holds for `-0` and for every non-negative non-NaN; the
            // rest -- negatives and NaNs -- take the product.
            p.loadi_bits(mask, 0x7fff_ffff);
            p.loadi_bits(inf, 0x7f80_0000);
            p.loadi_bits(LReg::L6, s.to_bits());
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                p.mul(x, LReg::L6, out);
                p.and(x, mask, ax);
                p.if_(Cond::Gte0(x), |p| {
                    p.if_(Cond::LessEq(ax, inf), |p| p.mov(x, out))
                });
                p.if_(Cond::Eq0(ax), |p| p.mov(x, out));
                store(p, o);
            });
        }
        HARD_SIGMOID => {
            // Flex's `alpha * x + beta`: two roundings, as two instructions.
            p.loadi_bits(mask, 0x7fff_ffff);
            p.loadi_bits(inf, 0x7f80_0000);
            p.loadi_bits(LReg::L6, s.to_bits());
            p.loadi_bits(LReg::L7, s2.to_bits());
            p.loadi(LReg::L1, 1.0);
            p.for_each_row_group(64, |p, o| {
                p.load(x, Format::Fp32, A_ROW + o);
                p.mul(LReg::L6, x, out);
                p.add(out, LReg::L7, out);
                // `.clamp(0, 1)`: a NaN stays, `-0` stays.
                p.and(out, mask, ax);
                p.if_(Cond::LessEq(ax, inf), |p| {
                    p.if_(Cond::Lt0(out), |p| {
                        p.if_(Cond::Ne0(ax), |p| p.mov(LReg::ZERO, out))
                    });
                    p.if_(Cond::Less(LReg::L1, out), |p| p.mov(LReg::L1, out));
                });
                store(p, o);
            });
        }
        k if (EQ_S..=LE_S).contains(&k) => {
            // Against a scalar: its zero made `+0` and its NaN settled here,
            // where every lane gets the unordered answer.
            compare_constants(p);
            if s.is_nan() {
                let v = if compare_of(k) == Some(NE) {
                    LReg::L6
                } else {
                    LReg::ZERO
                };
                p.for_each_row_group(64, |p, o| {
                    p.mov(v, out);
                    store(p, o);
                });
                return Some(Operands::Unary);
            }
            p.loadi_bits(LReg::L1, canonical(s.to_bits()));
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                compare_body(p, k, false);
                store(p, o);
            });
        }
        IS_NAN | IS_INF => {
            compare_constants(p);
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                p.mov(LReg::ZERO, out);
                p.and(x, mask, ax);
                if kind == IS_NAN {
                    p.if_(Cond::Less(inf, ax), |p| p.mov(LReg::L6, out));
                } else {
                    p.if_(Cond::LessEq(inf, ax), |p| {
                        p.if_(Cond::LessEq(ax, inf), |p| p.mov(LReg::L6, out))
                    });
                }
                store(p, o);
            });
        }
        MASK_WHERE => {
            p.for_each_row_group(64, |p, o| {
                load(p, o);
                p.load(LReg::L1, Format::Int32, super::kernel::B_ROW + o);
                p.load(LReg::L3, Format::Int32, super::kernel::C_ROW + o);
                p.mov(x, out);
                p.if_(Cond::Ne0(LReg::L1), |p| p.mov(LReg::L3, out));
                store(p, o);
            });
            return Some(Operands::Ternary);
        }
        _ => return None,
    }
    Some(Operands::Unary)
}

/// The program for `kind` with its second operand broadcast as `bcast`, and
/// what operands it takes; `None` where there is none. A row broadcast reads
/// the row's two faces from where the kernel lays them, which depends on the
/// row group, so its loop is unrolled; a column broadcast reads a whole tile
/// the mover has made of the column (`READ_BROADCAST_COL`), so its program is
/// the plain binary one.
pub fn program_for(
    kind: u32,
    scalars: [f32; 2],
    bcast: Broadcast,
) -> Option<(Operands, Vec<Instruction>)> {
    match bcast {
        Broadcast::None => program2(kind, scalars),
        _ if !broadcasts(kind) => None,
        Broadcast::Col => program2(kind, scalars).map(|(_, p)| (Operands::ColBroadcast, p)),
        Broadcast::Row => {
            let mut p = Program::with_policy(super::LoopPolicy::Unrolled);
            binary_constants(&mut p, kind, scalars);
            p.for_each_row_group(64, |p, o| {
                binary_body(p, kind, A_ROW + o, bias_row(o / 4) + (o & 2), OUT_ROW + o)
            });
            Some((Operands::RowBroadcast, p.finish()))
        }
    }
}

/// The program for `kind` (with its scalar), and what operands it takes, or
/// `None` for a kind with no SFPU program. `ADD_ROW` is `ADD` with a row
/// broadcast ([`program_for`]).
pub fn program(kind: u32, scalar: f32) -> Option<(Operands, Vec<Instruction>)> {
    program2(kind, [scalar, 0.0])
}

/// [`program`] for a kind with two scalars ([`kind_sfpu::CLAMP`],
/// [`kind_sfpu::HARD_SIGMOID`]).
pub fn program2(kind: u32, scalars: [f32; 2]) -> Option<(Operands, Vec<Instruction>)> {
    let scalar = scalars[0];
    let mut p = Program::new();
    if let Some(o) = exact_program(&mut p, kind, scalars) {
        return Some((o, p.finish()));
    }
    let mut p = Program::new();
    let operands = match kind {
        kind::ADD
        | kind::SUB
        | kind::MUL
        | kind_sfpu::DIV
        | kind_sfpu::BOOL_AND
        | kind_sfpu::BOOL_OR
        | kind_sfpu::BOOL_XOR
        | kind_sfpu::EQ..=kind_sfpu::LE
        | kind_sfpu::MASK_FILL
        | kind_sfpu::PRELU => {
            binary_constants(&mut p, kind, scalars);
            p.for_each_row_group(64, |p, o| {
                binary_body(p, kind, A_ROW + o, B_ROW + o, OUT_ROW + o)
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
        kind_sfpu::DIV_SCALAR => {
            p.loadi_bits(LReg::L6, f32::MAX.to_bits());
            p.loadi_bits(LReg::L7, 0x7f80_0000);
            // A divisor above `2^100`: both sides times `2^-64`, as
            // `scale_large_divisor` does, the scalar's here.
            let large = scalar.abs() > f32::from_bits(0x7180_0000);
            p.loadi(
                LReg::L1,
                if large {
                    scalar * f32::from_bits(0x1f80_0000)
                } else {
                    scalar
                },
            );
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                if large {
                    p.loadi_bits(LReg::L3, 0x1f80_0000);
                    p.mul(LReg::L0, LReg::L3, LReg::L0);
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
            Operands::Unary
        }
        kind_sfpu::EXP | kind_sfpu::LOG => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                if kind == kind_sfpu::EXP {
                    exp_program(p, LReg::L0, LReg::L7);
                } else {
                    log_program(p, LReg::L0, LReg::L7);
                }
                p.store(LReg::L7, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        // `0`/`1` either way: `x ^ 1`, on raw bits.
        kind_sfpu::BOOL_NOT => {
            p.loadi_bits(LReg::L3, 1);
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L2, Format::Int32, A_ROW + o);
                p.xor(LReg::L3, LReg::L2);
                p.store(LReg::L2, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::FILL => {
            p.loadi_bits(LReg::L2, scalar.to_bits());
            p.for_each_row_group(64, |p, o| p.store(LReg::L2, Format::Int32, OUT_ROW + o));
            Operands::Unary
        }
        kind_sfpu::SQRT | kind_sfpu::RSQRT => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                sqrt_program(p, LReg::L0, LReg::L7, kind == kind_sfpu::RSQRT);
                p.store(LReg::L7, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::LOG1P => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                log1p_program(p, LReg::L0, LReg::L7, super::kernel::SPILL_ROW + o);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::LOG_ABS => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
                p.abs(LReg::L0, LReg::L0);
                log_program(p, LReg::L0, LReg::L7);
                p.store(LReg::L7, Format::Fp32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind_sfpu::POW_FIX | kind_sfpu::POW_FIX_S => {
            let ternary = kind == kind_sfpu::POW_FIX;
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Int32, A_ROW + o);
                if ternary {
                    p.load(LReg::L1, Format::Int32, B_ROW + o);
                    p.load(LReg::L7, Format::Int32, super::kernel::C_ROW + o);
                } else {
                    p.loadi_bits(LReg::L1, scalar.to_bits());
                    p.load(LReg::L7, Format::Int32, B_ROW + o);
                }
                pow_fix(p);
                p.store(LReg::L7, Format::Int32, OUT_ROW + o);
            });
            if ternary {
                Operands::Ternary
            } else {
                Operands::Binary
            }
        }
        kind_sfpu::I32_TO_F32 => {
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L1, Format::Int32, A_ROW + o);
                i32_to_float(p, LReg::L1, LReg::L2, LReg::L3);
                p.store(LReg::L1, Format::Int32, OUT_ROW + o);
            });
            Operands::Unary
        }
        kind::ADD_ROW => return program_for(kind::ADD, scalars, Broadcast::Row),
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
    let bcast = match (kind, b.map(<[f32]>::len)) {
        (kind::ADD_ROW, _) => Broadcast::Row,
        (_, Some(n)) if n == cols && rows > 1 && broadcasts(kind) => Broadcast::Row,
        (_, Some(n)) if n == rows && cols > 1 && broadcasts(kind) => Broadcast::Col,
        _ => Broadcast::None,
    };
    let kind = if kind == kind::ADD_ROW {
        kind::ADD
    } else {
        kind
    };
    reference_for(kind, scalar, bcast, a, b, rows, cols)
}

/// [`reference`] with the broadcast said rather than read from `b`'s length.
pub fn reference_for(
    kind: u32,
    scalar: f32,
    bcast: Broadcast,
    a: &[f32],
    b: Option<&[f32]>,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let inputs: Vec<&[f32]> = std::iter::once(a).chain(b).collect();
    reference_op(kind, [scalar, 0.0], bcast, &inputs, rows, cols)
}

/// What [`crate::session::Session::pow`] computes, by the same chain's
/// programs: `y` an `F32` tensor of `x`'s shape or (`y` `None`) the scalar `s`.
pub fn pow_reference(x: &[f32], y: Option<&[f32]>, s: f32, rows: usize, cols: usize) -> Vec<f32> {
    let n = Broadcast::None;
    let l = reference_op(kind_sfpu::LOG_ABS, [0.0; 2], n, &[x], rows, cols);
    let z = match y {
        Some(y) => reference_op(kind::MUL, [0.0; 2], n, &[&l, y], rows, cols),
        None => reference_op(kind::MUL_SCALAR, [s, 0.0], n, &[&l], rows, cols),
    };
    let e = reference_op(kind_sfpu::EXP, [0.0; 2], n, &[&z], rows, cols);
    match y {
        Some(y) => reference_op(kind_sfpu::POW_FIX, [0.0; 2], n, &[x, y, &e], rows, cols),
        None => reference_op(kind_sfpu::POW_FIX_S, [s, 0.0], n, &[x, &e], rows, cols),
    }
}

/// [`reference_for`] with both scalars and any number of operands: `B` as
/// `bcast` says, a third (`Operands::Ternary`) the same shape as `A`.
pub fn reference_op(
    kind: u32,
    scalars: [f32; 2],
    bcast: Broadcast,
    inputs: &[&[f32]],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    use super::interp::Vector;
    use tt_isa::dm::face_index;
    let (a, b, c) = (inputs[0], inputs.get(1).copied(), inputs.get(2).copied());
    let (operands, math) = program_for(kind, scalars, bcast).expect("an SFPU op");
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
                (Operands::Ternary, Some(b)) => {
                    v.put_tile(B_ROW as usize, &tile(b, rows, i, j));
                    let c = c.expect("a ternary op's third operand");
                    v.put_tile(super::kernel::C_ROW as usize, &tile(c, rows, i, j));
                }
                (Operands::RowBroadcast, Some(b)) => {
                    let t = tile(b, 1, 0, j);
                    for r in 0..4 {
                        v.dst[(B_ROW + r) as usize] = std::array::from_fn(|c| t[c]);
                        v.dst[(B_ROW + 4 + r) as usize] = std::array::from_fn(|c| t[256 + c]);
                    }
                }
                (Operands::ColBroadcast, Some(b)) => {
                    // As `READ_BROADCAST_COL` makes it: row `r`'s value in
                    // every column.
                    let mut t = vec![0u32; 1024];
                    for r in 0..32 {
                        let gr = 32 * i + r;
                        let v = if gr < rows { b[gr].to_bits() } else { 0 };
                        for c in 0..32 {
                            t[face_index(r, c)] = v;
                        }
                    }
                    v.put_tile(B_ROW as usize, &t);
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

#[cfg(test)]
mod transcendental {
    use super::*;

    /// Ulps between `got` and the exact value `want` (an f64), in units of
    /// `got`'s binade.
    fn ulps(got: f32, want: f64) -> f64 {
        let ulp = f32::from_bits((got.abs().to_bits() & 0x7f80_0000).max(0x0080_0000)) as f64
            * f64::powi(2.0, -23);
        (got as f64 - want).abs() / ulp
    }

    /// Every input's result within `bound` of the exact value, relative (the
    /// bound derived on the program); the worst case is reported in ulps.
    fn sweep(kind: u32, xs: impl Iterator<Item = f32>, exact: fn(f64) -> f64, bound: f64) {
        let xs: Vec<f32> = xs.collect();
        let mut worst = (0.0, 0.0f32);
        for chunk in xs.chunks(1024) {
            let mut a = chunk.to_vec();
            a.resize(1024, 1.0);
            let got = reference(kind, 0.0, &a, None, 32, 32);
            for (x, g) in chunk.iter().zip(&got) {
                let w = exact(*x as f64);
                if w.is_nan() {
                    assert!(g.is_nan(), "kind {kind:#x}({x}): {g}");
                } else if w.is_infinite()
                    || w.abs() < f32::MIN_POSITIVE as f64
                    || w.abs() > f32::MAX as f64
                {
                    let want = if w.abs() < f32::MIN_POSITIVE as f64 {
                        0.0
                    } else {
                        w.signum() * f64::INFINITY
                    };
                    assert_eq!(*g as f64, want, "kind {kind:#x}({x:e})");
                } else {
                    let u = ulps(*g, w);
                    if u > worst.0 {
                        worst = (u, *x);
                    }
                    let rel = (*g as f64 - w).abs() / w.abs();
                    assert!(
                        rel <= bound,
                        "kind {kind:#x}({x:e}) = {g:e}, exact {w:e}: {rel:e} ({u:.2} ulps)"
                    );
                }
            }
        }
        println!(
            "kind {kind:#x}: worst {:.3} ulps at {:e} over {} inputs",
            worst.0,
            worst.1,
            xs.len()
        );
    }

    fn grid(lo: f32, hi: f32, n: usize) -> impl Iterator<Item = f32> {
        (0..n).map(move |i| lo + (hi - lo) * (i as f32 / n as f32))
    }

    #[test]
    fn exp_is_within_its_derived_bound() {
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            88.7,
            88.73,
            f32::from_bits(0x42b1_7217),
            f32::from_bits(0x42b1_7218),
            f32::from_bits(0x42b1_7219),
            f32::from_bits(0xc2ae_ac4f),
            f32::from_bits(0xc2ae_ac50),
            f32::from_bits(0xc2ae_ac51),
            -87.3,
            -87.4,
            -100.0,
            100.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        sweep(
            kind_sfpu::EXP,
            grid(-87.3, 88.7, 60_000)
                .chain(grid(-0.01, 0.01, 4096))
                .chain(specials),
            f64::exp,
            EXP_BOUND,
        );
    }

    #[test]
    fn log_is_within_its_derived_bound() {
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            0.5,
            1.0e-40,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
        ];
        let ln = |x: f64| {
            if x == 0.0 || x.abs() < f32::MIN_POSITIVE as f64 {
                f64::NEG_INFINITY
            } else {
                x.ln()
            }
        };
        sweep(
            kind_sfpu::LOG,
            grid(1.0e-6, 3.0, 40_000)
                .chain(grid(0.7, 1.45, 20_000))
                .chain((0..20_000).map(|i| f32::from_bits(0x0080_0000 + i * 106_000)))
                .chain(specials),
            ln,
            LOG_BOUND,
        );
    }

    /// Every binade of positive normals, `n` mantissas each.
    fn binades(n: u32) -> impl Iterator<Item = f32> {
        (1..255u32).flat_map(move |e| {
            (0..n).map(move |k| f32::from_bits((e << 23) | (k * (0x7f_ffff / n.max(1)))))
        })
    }

    /// Every binade, so the large inputs whose Newton steps once flushed
    /// (`Program::recip`, `|x| > 2^100`) are held as the rest are.
    #[test]
    fn recip_is_within_one_ulp_in_every_binade() {
        let r = |x: f64| {
            if (1.0 / x).abs() < f32::MIN_POSITIVE as f64 {
                0.0
            } else {
                1.0 / x
            }
        };
        sweep(
            kind_sfpu::RECIP,
            binades(256).chain(binades(64).map(|x| -x)),
            r,
            1.5 / 8_388_608.0,
        );
    }

    /// Quotients across the whole range, the smallest normal ones included:
    /// `divide`'s correction `(a - b q) y` is the quotient's error times `a/b`,
    /// so it too could fall below `2^-126`.
    #[test]
    fn division_is_within_one_ulp_down_to_the_smallest_quotients() {
        let mut worst = 0.0f64;
        for (ai, bi) in [(0u32, 0u32), (0x3f80_0000, 0), (0x0100_0000, 0x3f80_0000)] {
            let b: Vec<f32> = binades(4).skip(bi as usize % 7).take(1024).collect();
            let mut b = b;
            b.resize(1024, 3.0);
            let a: Vec<f32> = (0..1024)
                .map(|i| f32::from_bits(ai.max(0x0080_0000) + (i as u32) * 0x0007_ffff))
                .collect();
            let got = reference(kind_sfpu::DIV, 0.0, &a, Some(&b), 32, 32);
            for i in 0..1024 {
                let q = a[i] as f64 / b[i] as f64;
                if q.abs() < f32::MIN_POSITIVE as f64 || q.abs() > f32::MAX as f64 {
                    continue;
                }
                let rel = (got[i] as f64 - q).abs() / q.abs();
                worst = worst.max(rel);
                assert!(
                    rel <= 1.5 / 8_388_608.0,
                    "{:e} / {:e} = {:e}, exact {q:e}: {rel:e}",
                    a[i],
                    b[i],
                    got[i]
                );
            }
        }
        println!("division: worst {:.3} ulps", worst * 8_388_608.0);
    }

    /// `powf`'s special values, every pairing of special bases and exponents,
    /// exactly as the host's `powf` has them (the sign of a zero or infinity
    /// included); finite results within [`pow_bound`] of the exact power.
    /// Random pairs over the ranges where the result is normal too. A denormal
    /// base is flushed first (numerics row D), as the device's arithmetic does.
    #[test]
    fn pow_has_powf_s_special_values_and_is_within_its_bound() {
        let ftz = |x: f32| {
            if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
                f32::from_bits(x.to_bits() & 0x8000_0000)
            } else {
                x
            }
        };
        let xs = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.0,
            0.5,
            -0.5,
            3.0,
            -3.0,
            10.0,
            -10.0,
            1.0e-30,
            -1.0e-30,
            1.0e30,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MAX,
            f32::MIN_POSITIVE,
            0.9999,
            1.0001,
        ];
        let ys = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            2.0,
            3.0,
            -2.0,
            -3.0,
            0.5,
            -0.5,
            2.5,
            -2.5,
            1.0e10,
            8_388_609.0,
            16_777_216.0,
            -8_388_609.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.1,
            100.0,
            -100.0,
        ];
        let mut pairs: Vec<(f32, f32)> = xs
            .iter()
            .flat_map(|&x| ys.iter().map(move |&y| (x, y)))
            .collect();
        let mut st = 0x2545_f491u32;
        let mut rnd = || {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            st as f32 / u32::MAX as f32
        };
        for _ in 0..20_000 {
            let x = rnd() * 20.0 - 10.0;
            let y = rnd() * 16.0 - 8.0;
            pairs.push((x, y));
            pairs.push((x, y.round()));
            pairs.push((rnd() * 1.0e3, rnd() * 25.0 - 12.5));
        }
        let mut worst = 0.0f64;
        for chunk in pairs.chunks(1024) {
            let mut x: Vec<f32> = chunk.iter().map(|p| p.0).collect();
            let mut y: Vec<f32> = chunk.iter().map(|p| p.1).collect();
            x.resize(1024, 1.0);
            y.resize(1024, 1.0);
            let got = pow_reference(&x, Some(&y), 0.0, 32, 32);
            for (i, &(xi, yi)) in chunk.iter().enumerate() {
                let host = ftz(xi).powf(yi);
                let g = got[i];
                let what = format!(
                    "pow({xi:e}, {yi:e}) = {g:e} ({:#010x}), powf {host:e}",
                    g.to_bits()
                );
                if host.is_nan() {
                    assert!(g.is_nan(), "{what}");
                } else if g.is_nan() {
                    panic!("{what}: NaN where the host has a value");
                } else if host.is_infinite() || host == 0.0 || host.abs() < f32::MIN_POSITIVE {
                    // The exact power is infinite or flushes: the device's
                    // must be too, with its sign -- except within the bound of
                    // the range's ends, where either side may round over.
                    let exact = (ftz(xi) as f64).powf(yi as f64);
                    let edge = exact.is_finite()
                        && exact != 0.0
                        && ((exact.abs() / f32::MAX as f64 - 1.0).abs() < 2.0 * pow_bound(xi, yi)
                            || (exact.abs() / f32::MIN_POSITIVE as f64 - 1.0).abs()
                                < 2.0 * pow_bound(xi, yi)
                            || exact.abs() < f32::MIN_POSITIVE as f64);
                    if !edge {
                        let want = if host.is_infinite() {
                            host
                        } else {
                            f32::from_bits(host.to_bits() & 0x8000_0000)
                        };
                        assert_eq!(g.to_bits(), want.to_bits(), "{what}");
                    }
                } else {
                    let exact = (ftz(xi) as f64).powf(yi as f64);
                    let rel = (g as f64 - exact).abs() / exact.abs();
                    // Within the bound of the range's ends either side may
                    // round over: `e^z` saturates at the threshold `exp`
                    // draws.
                    let b2 = 1.0 + 2.0 * pow_bound(xi, yi);
                    if exact.abs() * b2 > f32::MAX as f64
                        || exact.abs() < f32::MIN_POSITIVE as f64 * b2
                    {
                        continue;
                    }
                    worst = worst.max(rel / pow_bound(xi, yi));
                    assert!(
                        rel <= pow_bound(xi, yi),
                        "{what}: {rel:e} against {:e}",
                        pow_bound(xi, yi)
                    );
                    assert_eq!(
                        g.is_sign_negative(),
                        host.is_sign_negative(),
                        "{what}: sign"
                    );
                }
            }
        }
        println!("pow: worst {worst:.3} of the bound");
        // A scalar exponent, by the same stages.
        let x: Vec<f32> = (0..1024)
            .map(|i| xs[i % xs.len()] * (1.0 + i as f32 / 1024.0))
            .collect();
        for s in [2.5f32, -0.5, 3.0, 0.0, f32::NAN, f32::INFINITY] {
            let got = pow_reference(&x, None, s, 32, 32);
            let viat = pow_reference(&x, Some(&vec![s; 1024]), 0.0, 32, 32);
            let b = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(b(&got), b(&viat), "a scalar exponent {s} is a tensor of it");
        }
    }

    #[test]
    fn sqrt_is_within_one_ulp() {
        let specials = [
            0.0,
            -0.0,
            1.0,
            4.0,
            -1.0,
            1.0e-40,
            -1.0e-40,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        // Flushed: a positive denormal's root is `+0`; a negative one is NaN.
        let root = |x: f64| {
            if x > 0.0 && x < f32::MIN_POSITIVE as f64 {
                0.0
            } else {
                x.sqrt()
            }
        };
        // One ulp of the correctly rounded root is at most 1.5 ulps of the
        // exact one: `1.5 * 2^-23` relative.
        sweep(
            kind_sfpu::SQRT,
            binades(256).chain(grid(0.0, 16.0, 20_000)).chain(specials),
            root,
            1.5 / 8_388_608.0,
        );
    }

    #[test]
    fn rsqrt_is_within_its_derived_bound() {
        let specials = [0.0, -0.0, 1.0, 4.0, f32::MAX, f32::INFINITY, f32::NAN, -2.0];
        let r = |x: f64| {
            if x == 0.0 || (x > 0.0 && x < f32::MIN_POSITIVE as f64) {
                f64::INFINITY.copysign(x)
            } else {
                1.0 / x.sqrt()
            }
        };
        sweep(
            kind_sfpu::RSQRT,
            binades(256).chain(grid(0.01, 16.0, 20_000)).chain(specials),
            r,
            RSQRT_BOUND,
        );
    }

    #[test]
    fn log1p_is_within_its_derived_bound() {
        // Denormal inputs come back as themselves (`log1p(x) = x` there, bits
        // and all), which the sweep's flush-to-zero rule would refuse; the
        // smallest normals are in.
        let specials = [
            0.0,
            -0.0,
            -1.0,
            -2.0,
            1.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            5.9604645e-8,
            -5.9604645e-8,
            1.0e-7,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        sweep(
            kind_sfpu::LOG1P,
            grid(-0.999, 3.0, 40_000)
                .chain(grid(-1.0e-3, 1.0e-3, 20_000))
                .chain(binades(64).filter(|x| *x > 1.0e-30))
                .chain(
                    binades(64)
                        .filter(|x| *x > 1.0e-30 && *x < 0.99)
                        .map(|x| -x),
                )
                .chain(specials),
            f64::ln_1p,
            LOG1P_BOUND,
        );
    }
}

#[cfg(test)]
mod arity {
    use super::*;

    /// `operands` says what `program` builds, for every kind there is: the
    /// shape check and the kernel cannot disagree.
    #[test]
    fn operands_agrees_with_program() {
        for k in (1..=kind::LAST).chain(0x100..0x140) {
            // Scalars any kind takes: `CLAMP`'s bounds uncrossed.
            let got = program2(k, [-0.5, 0.5]).map(|(o, _)| o);
            assert_eq!(operands(k), got, "kind {k:#x}");
        }
    }
}

#[cfg(test)]
mod s2 {
    use super::kind_sfpu::*;
    use super::*;

    /// Both zeros, both infinities, NaNs of either sign with payloads,
    /// denormals, the extremes, and ordinary values: every pairing of them
    /// lands in one 32x32 tile.
    const SPECIALS: [u32; 16] = [
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0xffc0_0001,
        0x0000_0001,
        0x8040_0000,
        0x7f7f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x4000_0000,
        0xbf00_0000,
        0x3e4c_cccd,
        0xc2c8_0000,
    ];

    fn pairs() -> (Vec<f32>, Vec<f32>) {
        let a = (0..1024)
            .map(|i| f32::from_bits(SPECIALS[i % 16]))
            .collect();
        let b = (0..1024)
            .map(|i| f32::from_bits(SPECIALS[(i / 16) % 16]))
            .collect();
        (a, b)
    }

    fn b(x: bool) -> f32 {
        f32::from_bits(u32::from(x))
    }

    /// The SFPU's arithmetic flushes a denormal operand to a signed zero
    /// before it computes (numerics row D): the host's arithmetic, so.
    fn ftz(x: f32) -> f32 {
        if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
            f32::from_bits(x.to_bits() & 0x8000_0000)
        } else {
            x
        }
    }

    /// The device's bits for the host's `want`: the same bits, or -- where
    /// `products` says the op multiplies -- a flushed denormal's signed zero,
    /// or any NaN for a NaN.
    fn same(got: f32, want: f32, products: bool) -> bool {
        let (g, w) = (got.to_bits(), want.to_bits());
        g == w
            || (products && want.is_nan() && got.is_nan())
            || (products && want != 0.0 && want.abs() < f32::MIN_POSITIVE && g == w & 0x8000_0000)
    }

    fn check(kind: u32, scalars: [f32; 2], host: impl Fn(f32, f32) -> f32, products: bool) {
        let (a, bv) = pairs();
        let inputs: Vec<&[f32]> = match operands(kind).unwrap() {
            Operands::Unary => vec![&a],
            _ => vec![&a, &bv],
        };
        let got = reference_op(kind, scalars, Broadcast::None, &inputs, 32, 32);
        for i in 0..1024 {
            let want = host(a[i], bv[i]);
            assert!(
                same(got[i], want, products),
                "kind {kind:#x} {scalars:?}: ({:#010x}, {:#010x}) gave {:#010x}, the host {:#010x}",
                a[i].to_bits(),
                bv[i].to_bits(),
                got[i].to_bits(),
                want.to_bits()
            );
        }
    }

    #[test]
    fn the_unary_ops_are_the_host_s_bit_for_bit() {
        check(NEG, [0.0; 2], |x, _| -x, false);
        check(ABS, [0.0; 2], |x, _| x.abs(), false);
        let sign = |x: f32| {
            if x.is_nan() {
                x
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        };
        check(SIGN, [0.0; 2], move |x, _| sign(x), false);
        for (lo, hi) in [(-1.0f32, 1.0f32), (0.0, 0.5), (-0.0, 0.0), (-2.0, -2.0)] {
            check(CLAMP, [lo, hi], move |x, _| x.clamp(lo, hi), false);
        }
        // Flex's `x.max(s)`, measured: a NaN on either side gives the other,
        // and equal values the scalar.
        let max = |x: f32, s: f32| {
            if s.is_nan() || x > s {
                x
            } else {
                s
            }
        };
        let min = |x: f32, s: f32| {
            if s.is_nan() || x < s {
                x
            } else {
                s
            }
        };
        for s in [0.0f32, -0.0, 1.0, -0.5, f32::NAN, f32::INFINITY] {
            check(CLAMP_MIN, [s, 0.0], move |x, _| max(x, s), false);
            check(CLAMP_MAX, [s, 0.0], move |x, _| min(x, s), false);
        }
        for ns in [0.01f32, -3.0, 0.0] {
            // The product of a flushed operand; the pass-through keeps `x`.
            let leaky = move |x: f32, _| if x >= 0.0 { x } else { ns * ftz(x) };
            check(LEAKY_RELU, [ns, 0.0], leaky, true);
        }
        for (al, be) in [(0.2f32, 0.5f32), (1.0 / 6.0, 0.5), (-1.0, 0.0)] {
            check(
                HARD_SIGMOID,
                [al, be],
                move |x, _| ftz(ftz(al * ftz(x)) + be).clamp(0.0, 1.0),
                true,
            );
        }
        let prelu = |x: f32, a: f32| if x >= 0.0 { x } else { ftz(a) * ftz(x) };
        check(PRELU, [0.0; 2], prelu, true);
        check(IS_NAN, [0.0; 2], |x, _| b(x.is_nan()), false);
        check(IS_INF, [0.0; 2], |x, _| b(x.is_infinite()), false);
    }

    #[test]
    fn the_comparisons_are_ieee_s() {
        for (t, sk) in COMPARES {
            check(t, [0.0; 2], move |x, y| b(ieee_compare(t, x, y)), false);
            for s in [
                0.0f32,
                -0.0,
                1.0,
                -0.5,
                f32::NAN,
                f32::NEG_INFINITY,
                f32::from_bits(1),
            ] {
                check(sk, [s, 0.0], move |x, _| b(ieee_compare(t, x, s)), false);
            }
        }
    }

    #[test]
    fn the_masks_pass_bits_through() {
        let (a, v) = pairs();
        let m: Vec<f32> = (0..1024).map(|i| b((i * 7) % 3 == 0)).collect();
        for value in [2.5f32, -0.0, f32::NAN] {
            let got = reference_op(MASK_FILL, [value, 0.0], Broadcast::None, &[&a, &m], 32, 32);
            for i in 0..1024 {
                let want = if m[i].to_bits() != 0 { value } else { a[i] };
                assert_eq!(got[i].to_bits(), want.to_bits(), "fill {i}");
            }
        }
        let got = reference_op(MASK_WHERE, [0.0; 2], Broadcast::None, &[&a, &m, &v], 32, 32);
        for i in 0..1024 {
            let want = if m[i].to_bits() != 0 { v[i] } else { a[i] };
            assert_eq!(got[i].to_bits(), want.to_bits(), "where {i}");
        }
    }
}
