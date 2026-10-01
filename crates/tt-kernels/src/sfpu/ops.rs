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
}

/// What `kind` computes on and what it produces (`crate::tensor::Elem`); `None`
/// for a kind that moves datums whatever they are (`COPY`), whose output is
/// its input's. The session refuses any other operand
/// (`TensorError::Elem`) before choosing a unit.
pub fn elems(kind: u32) -> Option<(Elem, Elem)> {
    match kind {
        kind::COPY => None,
        kind_sfpu::BOOL_NOT | kind_sfpu::BOOL_AND | kind_sfpu::BOOL_OR | kind_sfpu::BOOL_XOR => {
            Some((Elem::Bool, Elem::Bool))
        }
        _ => Some((Elem::F32, Elem::F32)),
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
        | kind_sfpu::LOG => Accuracy::Approximate,
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
    p.loadi_bits(k, 0x42b1_7218); // 88.72284
    p.if_(Cond::Less(k, x), |p| p.loadi_bits(d, 0x7f80_0000));
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
        | kind_sfpu::BOOL_XOR => Operands::Binary,
        kind::MUL_SCALAR
        | kind::ADD_SCALAR
        | kind::RELU
        | kind_sfpu::RECIP
        | kind_sfpu::DIV_SCALAR
        | kind_sfpu::EXP
        | kind_sfpu::LOG
        | kind_sfpu::BOOL_NOT => Operands::Unary,
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
    )
}

/// `out = a (kind) b` for one row group, `a` in `L0`, `b` from `Dst` at `b_at`,
/// the result stored at `out_at`. `DIV` needs `f32::MAX` in `L6` and `+inf`
/// in `L7` ([`binary_constants`]).
fn binary_body(p: &mut Program, kind: u32, a_at: u32, b_at: u32, out_at: u32) {
    // A `Bool` is `0`/`1` as an integer -- a denormal to FP32's load and
    // store, which would flush it -- so the logic ops move raw bits.
    let fmt = match elems(kind) {
        Some((Elem::F32, _)) => Format::Fp32,
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
fn binary_constants(p: &mut Program, kind: u32) {
    if kind == kind_sfpu::DIV {
        p.loadi_bits(LReg::L6, f32::MAX.to_bits());
        p.loadi_bits(LReg::L7, 0x7f80_0000);
    }
}

/// The program for `kind` with its second operand broadcast as `bcast`, and
/// what operands it takes; `None` where there is none. A row broadcast reads
/// the row's two faces from where the kernel lays them, which depends on the
/// row group, so its loop is unrolled; a column broadcast reads a whole tile
/// the mover has made of the column (`READ_BROADCAST_COL`), so its program is
/// the plain binary one.
pub fn program_for(
    kind: u32,
    scalar: f32,
    bcast: Broadcast,
) -> Option<(Operands, Vec<Instruction>)> {
    match bcast {
        Broadcast::None => program(kind, scalar),
        _ if !broadcasts(kind) => None,
        Broadcast::Col => program(kind, scalar).map(|(_, p)| (Operands::ColBroadcast, p)),
        Broadcast::Row => {
            let mut p = Program::with_policy(super::LoopPolicy::Unrolled);
            binary_constants(&mut p, kind);
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
    let mut p = Program::new();
    let operands = match kind {
        kind::ADD
        | kind::SUB
        | kind::MUL
        | kind_sfpu::DIV
        | kind_sfpu::BOOL_AND
        | kind_sfpu::BOOL_OR
        | kind_sfpu::BOOL_XOR => {
            binary_constants(&mut p, kind);
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
            p.loadi(LReg::L1, scalar);
            p.for_each_row_group(64, |p, o| {
                p.load(LReg::L0, Format::Fp32, A_ROW + o);
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
        kind::ADD_ROW => return program_for(kind::ADD, scalar, Broadcast::Row),
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
    use super::interp::Vector;
    use tt_isa::dm::face_index;
    let (operands, math) = program_for(kind, scalar, bcast).expect("an SFPU op");
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
}

#[cfg(test)]
mod arity {
    use super::*;

    /// `operands` says what `program` builds, for every kind there is: the
    /// shape check and the kernel cannot disagree.
    #[test]
    fn operands_agrees_with_program() {
        for k in (1..=kind::LAST).chain(0x100..0x140) {
            assert_eq!(operands(k), program(k, 0.5).map(|(o, _)| o), "kind {k:#x}");
        }
    }
}
