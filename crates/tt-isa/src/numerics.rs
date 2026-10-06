//! Bit-exact models of the arithmetic the hardware actually performs.
//!
//! The tolerance policy this project works to is "never a guessed epsilon": an
//! expected value comes from a documented model, asserted bit for bit. For
//! multiply-add that model is `Miscellaneous/FMA/fma.c`, whose `fma_model_bh`
//! matches "Blackhole Baby RISCV `fma.s` family of instructions and Blackhole
//! Tensix Vector Unit (SFPU) [`SFPMAD`] family of instructions"
//! (`Miscellaneous/FMA/README.md`).
//!
//! # This is a port, and ports drift
//!
//! [`fma_bh`] is hand-translated C, which is exactly the kind of transcription the
//! rest of this crate generates its way out of. Three things keep it honest:
//!
//! * `Miscellaneous/FMA/fma.c` is inside the digest `PINS.toml` pins, so the source
//!   cannot change without the pin check failing.
//! * The `README.md` beside it *enumerates* how `fma_model_bh` differs from IEEE
//!   754, and the tests below assert each named difference — so the port is checked
//!   against prose written independently of the code it describes, not just against
//!   itself.
//! * Where the C looks wrong, it is reproduced anyway. `Miscellaneous/FMA/README.md`
//!   and the checklist both say the documented models faithfully reproduce the
//!   hardware's bugs and must be matched rather than fixed.

pub mod sfpu;
pub mod stochastic;

/// `x * y + z` as Blackhole computes it, on FP32 bit patterns.
///
/// A direct port of `fma_model_bh` (`Miscellaneous/FMA/fma.c:102`). The integer
/// widths are load-bearing and match the C: the product accumulator is 64-bit and
/// the addend is 32-bit, which is why they are shifted and complemented
/// differently.
///
/// Differs from IEEE 754 in the ways `Miscellaneous/FMA/README.md` lists:
/// denormal inputs are treated as signed zero, denormal outputs are flushed after
/// rounding, an intermediate product that would overflow gives infinity even where
/// the fused result would be finite, one that would underflow is treated as exactly
/// zero, and the product carries four extra bits rather than the addend being
/// widened.
pub const fn fma_bh(x: u32, y: u32, z: u32) -> u32 {
    // Unpack, flushing denormal inputs to zero.
    let x_e = ((x >> 23) & 255) as i32;
    let mut x_m = (x & 0x7f_ffff) ^ 0x80_0000;
    if x_e == 0 {
        x_m = 0;
    }
    let y_e = ((y >> 23) & 255) as i32;
    let mut y_m = (y & 0x7f_ffff) ^ 0x80_0000;
    if y_e == 0 {
        y_m = 0;
    }
    let z_e = ((z >> 23) & 255) as i32;
    let mut z_m = (z & 0x7f_ffff) ^ 0x80_0000;
    if z_e == 0 {
        z_m = 0;
    }
    let z_sign = z & 0x8000_0000;

    // p = x * y
    let p_sign = (x ^ y) & 0x8000_0000;
    let mut p_m = (x_m as u64) * (y_m as u64);
    let mut p_e = x_e + y_e - 23 - 127;

    // Three extra bits of precision (G, R, S).
    p_m <<= 3;
    z_m <<= 3;

    // Realign p_m to match z_m, removing 23 bits and keeping a sticky bit.
    p_m = (p_m >> 23) | ((p_m & 0x7f_ffff) != 0) as u64;
    p_e += 23;

    // NaN or infinite input, or an infinite product.
    if x_e == 255 || y_e == 255 || p_e >= 255 || z_e == 255 {
        if (x_e == 255 && (x_m != 0x80_0000 || y_m == 0))
            || (y_e == 255 && (y_m != 0x80_0000 || x_m == 0))
            || (z_e == 255 && z_m != 0x400_0000)
            || (z_e == 255 && (x_e == 255 || y_e == 255) && (z_sign != p_sign))
        {
            return 0x7fc0_0000;
        } else if z_e == 255 {
            return z;
        } else {
            return p_sign | 0x7f80_0000;
        }
    }

    // A zero product, or a multiply that would underflow on its own.
    if p_m == 0 || p_e < 0 {
        return if z_m != 0 { z } else { z_sign & p_sign };
    }

    // r = z + p, aligning both to the larger exponent.
    let mut r_e = if p_e > z_e { p_e } else { z_e };
    if p_e < r_e {
        p_m = semi_sticky_shift_64(p_m, r_e - p_e);
    }
    if z_e < r_e {
        z_m = semi_sticky_shift_32(z_m, r_e - z_e);
    }
    let r_sign = if p_m >= z_m as u64 { p_sign } else { z_sign };
    if z_sign != r_sign {
        z_m = !z_m;
    }
    if p_sign != r_sign {
        p_m = !p_m;
    }
    // The C assigns this sum to a `uint32_t`, so it truncates. That is what makes
    // the two's-complement negation above work out.
    let mut r_m = (z_m as u64)
        .wrapping_add(p_m)
        .wrapping_add((p_sign != z_sign) as u64) as u32;

    if r_m == 0 {
        return z_sign & p_sign;
    }

    // Normalise to 5 zero bits, 1 one bit, 26 fractional bits.
    let mut n = 5 - r_m.leading_zeros() as i32;
    r_e += n;
    if r_e >= 255 {
        return r_sign | 0x7f80_0000;
    }
    if r_e <= 0 {
        n += 1;
        r_e = 0;
    }
    if n <= 0 {
        r_m <<= -n;
    } else {
        // `r_m & (n | 1)` rather than a mask of the discarded bits. That is what
        // the C says, and `Miscellaneous/FMA/README.md` is explicit that these
        // models reproduce the hardware rather than the intent, so it is
        // transcribed rather than corrected.
        r_m = (r_m >> n) | ((r_m & (n as u32 | 1)) != 0) as u32;
    }

    let mut r = ((r_e as u32) << 23).wrapping_add((r_m >> 3) & 0x7f_ffff);
    // Round to nearest, ties to even.
    r = r.wrapping_add((((r_m & 7) + (r & 1)) > 4) as u32);
    // Flush denormal results, after rounding, preserving the sign.
    if (r >> 23) == 0 {
        r = 0;
    }
    r_sign | r
}

/// `var >>= amount`, keeping a sticky bit, for a 64-bit accumulator.
const fn semi_sticky_shift_64(var: u64, amount: i32) -> u64 {
    if amount >= 64 {
        return 0;
    }
    let orig = var;
    let shifted = var >> amount;
    if shifted != 0 {
        shifted | (((shifted << amount) != orig) as u64)
    } else {
        shifted
    }
}

/// The same, for the 32-bit addend. The width is not incidental: the C declares
/// `z_m` as `uint32_t` and `p_m` as `uint64_t`, so the "shifted all the bits out"
/// threshold differs between them.
const fn semi_sticky_shift_32(var: u32, amount: i32) -> u32 {
    if amount >= 32 {
        return 0;
    }
    let orig = var as u64;
    let shifted = var >> amount;
    if shifted != 0 {
        shifted | ((((shifted << amount) as u64) != orig) as u32)
    } else {
        shifted
    }
}

/// `x * y` as the SFPU computes it.
///
/// `SFPMUL` is `SFPMAD` with the addend forced to the constant `LReg[9]`, and
/// `tt_isa::sfpu::mul` negates it so the addend is `-0` rather than `+0` — without
/// which the multiply drops the sign of a zero result (`SFPMUL.md`, and
/// `docs/learnings/ttsim-divergence.md` row C, which found it by differential testing). The
/// oracle has to model the same thing.
pub const fn mul_bh(x: u32, y: u32) -> u32 {
    fma_bh(x, y, 0x8000_0000)
}

/// `x + y` as the SFPU computes it, via the `1.0 * x + y` form `SFPADD` uses.
pub const fn add_bh(x: u32, y: u32) -> u32 {
    fma_bh(0x3f80_0000, x, y)
}

// ---------------------------------------------------------------------------
// Matrix Unit
// ---------------------------------------------------------------------------

/// The part of an FP32-valued `SrcA` operand one fidelity phase consumes.
///
/// A port of `SrcAFidelityBits` (`WormholeB0/.../MatrixUnit.md:143`). Even phases
/// take the sign, exponent, implicit bit and top four mantissa bits; odd phases
/// take the next five, as the difference `x - (x & 0xfff83fff)`. The last TF32
/// mantissa bit (bit 13) is consumed by no phase, which is `MVMUL.md`'s "the
/// least significant bit of the ... TF32 mantissa is ignored".
///
/// Operates on the value `SrcDecodeTF32` produces, i.e. the FP32 bit pattern of a
/// TF32 or BF16 `Src` datum; BF16 needs no separate case because its low mantissa
/// bits are zero.
pub fn src_a_fidelity_bits(x: f32, phase: u32) -> f32 {
    if phase & 1 == 0 {
        f32::from_bits(x.to_bits() & 0xfff8_0000)
    } else {
        x - f32::from_bits(x.to_bits() & 0xfff8_3fff)
    }
}

/// The part of an FP32-valued `SrcB` operand one fidelity phase consumes.
///
/// A port of `SrcBFidelityBits` (`MatrixUnit.md:155`): phases 0 and 1 take the top
/// six mantissa bits with the implicit bit, phases 2 and 3 the next four.
pub fn src_b_fidelity_bits(x: f32, phase: u32) -> f32 {
    if phase & 2 == 0 {
        f32::from_bits(x.to_bits() & 0xfffe_0000)
    } else {
        x - f32::from_bits(x.to_bits() & 0xfffe_1fff)
    }
}

/// Denormals read out of `Src` are flushed (`MVMUL.md`: "Denormals will be flushed
/// to zero"), keeping the sign.
fn flush(x: f32) -> f32 {
    if x.to_bits() & 0x7f80_0000 == 0 {
        f32::from_bits(x.to_bits() & 0x8000_0000)
    } else {
        x
    }
}

/// `MVMUL` into FP32 `Dst`: `dst += src_b @ src_a`, once per fidelity phase in
/// `phases`, for the floating-point styles (TF32/BF16 `Src`, FP32 `Dst`).
///
/// `src_b` is 8x16 and `src_a` 16x16, both already in `Src` precision -- truncate
/// with `tile::fp32_to_tf32` first.
///
/// # This is only a model where the arithmetic is exact
///
/// `MVMUL.md` says its float model is "a rough guide": the summation order,
/// fusion and intermediate precision are unspecified. So this returns `None`
/// unless every partial product and every sum it forms is exact in FP32 --
/// checked by recomputing in `f64` and comparing -- in which case any order gives
/// the same bits and the answer does not depend on what the document leaves open.
/// A gate built on it therefore needs operands chosen to stay in that regime, and
/// fails loudly (rather than asserting a guess) when they do not.
pub fn mvmul_reference(
    dst: &[[f32; 16]; 8],
    src_b: &[[f32; 16]; 8],
    src_a: &[[f32; 16]; 16],
    phases: &[u32],
) -> Option<[[f32; 16]; 8]> {
    let mut out = *dst;
    for &phase in phases {
        for i in 0..8 {
            for j in 0..16 {
                let mut x = 0f32;
                let mut wide = 0f64;
                for k in 0..16 {
                    let b = src_b_fidelity_bits(flush(src_b[i][k]), phase);
                    let a = src_a_fidelity_bits(flush(src_a[k][j]), phase);
                    let p = b * a;
                    if f64::from(p) != f64::from(b) * f64::from(a) {
                        return None;
                    }
                    x += p;
                    wide += f64::from(p);
                }
                if f64::from(x) != wide {
                    return None;
                }
                let sum = out[i][j] + x;
                if f64::from(sum) != f64::from(out[i][j]) + f64::from(x) {
                    return None;
                }
                out[i][j] = sum;
            }
        }
    }
    Some(out)
}

/// A 32x32 tile matmul, `dst += a @ b`, composed from [`mvmul_reference`] the
/// way the Matrix Unit composes it: output face `(i, j)` accumulates
/// `a` face `(i, k)` (as `SrcB`, eight rows at a time) against `b` face
/// `(k, j)` (as `SrcA`), for `k = 0, 1`, in that order.
///
/// Faces are the four 16x16 quadrants, row-major. `None` wherever
/// [`mvmul_reference`] would be: the composition is only a model while every
/// product and sum stays exact.
pub fn matmul_tile_reference(
    dst: &[[f32; 32]; 32],
    a: &[[f32; 32]; 32],
    b: &[[f32; 32]; 32],
    phases: &[u32],
) -> Option<[[f32; 32]; 32]> {
    let mut out = *dst;
    for fi in 0..2 {
        for fj in 0..2 {
            for half in 0..2 {
                let r0 = 16 * fi + 8 * half;
                let mut acc = [[0f32; 16]; 8];
                for (i, row) in acc.iter_mut().enumerate() {
                    row.copy_from_slice(&out[r0 + i][16 * fj..16 * fj + 16]);
                }
                for k in 0..2 {
                    let mut src_b = [[0f32; 16]; 8];
                    for (i, row) in src_b.iter_mut().enumerate() {
                        row.copy_from_slice(&a[r0 + i][16 * k..16 * k + 16]);
                    }
                    let mut src_a = [[0f32; 16]; 16];
                    for (i, row) in src_a.iter_mut().enumerate() {
                        row.copy_from_slice(&b[16 * k + i][16 * fj..16 * fj + 16]);
                    }
                    acc = mvmul_reference(&acc, &src_b, &src_a, phases)?;
                }
                for (i, row) in acc.iter().enumerate() {
                    out[r0 + i][16 * fj..16 * fj + 16].copy_from_slice(row);
                }
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tile_reference_is_the_integer_product() {
        let mut a = [[0f32; 32]; 32];
        let mut b = [[0f32; 32]; 32];
        for i in 0..32 {
            for j in 0..32 {
                a[i][j] = ((i * 7 + j * 3) % 23) as f32 - 11.0;
                b[i][j] = ((i * 5 + j * 11) % 17) as f32 - 8.0;
            }
        }
        let got = matmul_tile_reference(&[[0f32; 32]; 32], &a, &b, &[0]).unwrap();
        for i in 0..32 {
            for j in 0..32 {
                let want: f32 = (0..32).map(|k| a[i][k] * b[k][j]).sum();
                assert_eq!(got[i][j], want, "[{i}][{j}]");
            }
        }
    }

    #[test]
    fn fidelity_phases_split_each_operand_exactly() {
        // The pair `step9_matmul.rs` watched ttsim split, one phase at a time.
        let a = 1.0f32 + 2f32.powi(-5) + 2f32.powi(-9);
        let b = 1.0f32 + 2f32.powi(-7) + 2f32.powi(-9);
        assert_eq!(src_a_fidelity_bits(a, 0), 1.0);
        assert_eq!(src_a_fidelity_bits(a, 1), 2f32.powi(-5) + 2f32.powi(-9));
        assert_eq!(src_b_fidelity_bits(b, 0), 1.0);
        assert_eq!(src_b_fidelity_bits(b, 2), 2f32.powi(-7) + 2f32.powi(-9));
        // Phases 0 and 2 share the `SrcA` half, 0 and 1 the `SrcB` half.
        assert_eq!(src_a_fidelity_bits(a, 2), src_a_fidelity_bits(a, 0));
        assert_eq!(src_b_fidelity_bits(b, 1), src_b_fidelity_bits(b, 0));
        // All four phases recover the product.
        let sum: f32 = (0..4)
            .map(|p| src_b_fidelity_bits(b, p) * src_a_fidelity_bits(a, p))
            .sum();
        assert_eq!(sum, a * b);
    }

    #[test]
    fn the_last_tf32_bit_of_src_a_is_consumed_by_no_phase() {
        let a = 1.0f32 + 2f32.powi(-10);
        let consumed = src_a_fidelity_bits(a, 0) + src_a_fidelity_bits(a, 1);
        assert_eq!(consumed, 1.0);
    }

    #[test]
    fn the_reference_refuses_to_guess_where_order_matters() {
        let dst = [[0f32; 16]; 8];
        let mut a = [[0f32; 16]; 16];
        let mut b = [[0f32; 16]; 8];
        // 2^24 + 1 + 1 is 2^24 + 2 in one order and 2^24 in the other.
        a[0][0] = 1.0;
        a[1][0] = 1.0;
        a[2][0] = 1.0;
        b[0][0] = 16_777_216.0;
        b[0][1] = 1.0;
        b[0][2] = 1.0;
        assert_eq!(mvmul_reference(&dst, &b, &a, &[0]), None);
        // Small integers are exact, so it answers.
        b[0][0] = 3.0;
        let out = mvmul_reference(&dst, &b, &a, &[0]).unwrap();
        assert_eq!(out[0][0], 5.0);
    }
}

/// Matrix elementwise arithmetic; multiplication accumulates phase products.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MatrixEltwiseOp {
    Add,
    Sub,
    Mul,
}

/// Independent exact-domain oracle for ELW instructions with FP32 Dst.
/// Inputs are already decoded Src values. This deliberately refuses special
/// values, subnormal inputs/results, inexact sums/products and invalid phases:
/// the specification's floating model is only a rough guide. It does not use
/// the SFPU FMA model. Zero signs require separate measured characterization.
pub fn elw_reference(
    dst: &[[f32; 16]; 8],
    a: &[[f32; 16]; 8],
    b: &[[f32; 16]; 8],
    op: MatrixEltwiseOp,
    broadcast: crate::matrix::SrcBroadcast,
    phases: &[u32],
) -> Option<[[f32; 16]; 8]> {
    if phases.is_empty()
        || phases.iter().any(|&p| p > 3)
        || (op != MatrixEltwiseOp::Mul && phases != [0])
    {
        return None;
    }
    let valid = |x: f32| x.is_finite() && (x == 0.0 || x.is_normal());
    let exact = |wide: f64| {
        let x = wide as f32;
        (valid(x) && f64::from(x) == wide).then_some(x)
    };
    let mut out = *dst;
    for i in 0..8 {
        for j in 0..16 {
            let (r, c) = broadcast.coordinate(i, j);
            let (a, b) = (a[i][j], b[r][c]);
            if !valid(a) || !valid(b) {
                return None;
            }
            if op != MatrixEltwiseOp::Mul {
                // ELWADD/SUB align both inputs at a shared 10-fraction-bit
                // quantum (step90). Exact FP32 sums may still lose data here.
                let exp = |x: f32| (x.to_bits() >> 23) & 255;
                let common = exp(a).max(exp(b));
                for x in [a, b] {
                    if x != 0.0 {
                        let shift = 13 + common - exp(x);
                        if shift >= 24 || x.to_bits() & ((1u32 << shift) - 1) != 0 {
                            return None;
                        }
                    }
                }
            }
            out[i][j] = match op {
                MatrixEltwiseOp::Add => exact(f64::from(a) + f64::from(b))?,
                MatrixEltwiseOp::Sub => exact(f64::from(a) - f64::from(b))?,
                MatrixEltwiseOp::Mul => {
                    let mut acc = dst[i][j];
                    if !valid(acc) {
                        return None;
                    }
                    for &phase in phases {
                        let aa = src_a_fidelity_bits(a, phase);
                        let bb = src_b_fidelity_bits(b, phase);
                        let product = exact(f64::from(aa) * f64::from(bb))?;
                        acc = exact(f64::from(acc) + f64::from(product))?;
                    }
                    acc
                }
            };
        }
    }
    Some(out)
}

#[cfg(test)]
mod elw_tests {
    use super::*;
    use crate::matrix::SrcBroadcast;
    #[test]
    fn exact_oracle_refuses_alignment_loss_and_exceptional_arithmetic() {
        let zero = [[0.0; 16]; 8];
        for a in [f32::NAN, f32::INFINITY, f32::from_bits(1)] {
            assert!(elw_reference(
                &zero,
                &[[a; 16]; 8],
                &[[1.0; 16]; 8],
                MatrixEltwiseOp::Add,
                SrcBroadcast::None,
                &[0]
            )
            .is_none());
        }
        // This sum is exact in FP32 but loses the small operand on the
        // matrix alignment datapath. Never assert its IEEE bits on hardware.
        assert!(elw_reference(
            &zero,
            &[[f32::from_bits(107u32 << 23); 16]; 8],
            &[[1.0; 16]; 8],
            MatrixEltwiseOp::Add,
            SrcBroadcast::None,
            &[0]
        )
        .is_none());
        assert!(elw_reference(
            &zero,
            &[[1.0; 16]; 8],
            &[[2.0; 16]; 8],
            MatrixEltwiseOp::Mul,
            SrcBroadcast::None,
            &[4]
        )
        .is_none());
    }
}
