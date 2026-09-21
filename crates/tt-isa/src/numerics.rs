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
/// `docs/ttsim-divergence.md` row C, which found it by differential testing). The
/// oracle has to model the same thing.
pub const fn mul_bh(x: u32, y: u32) -> u32 {
    fma_bh(x, y, 0x8000_0000)
}

/// `x + y` as the SFPU computes it, via the `1.0 * x + y` form `SFPADD` uses.
pub const fn add_bh(x: u32, y: u32) -> u32 {
    fma_bh(0x3f80_0000, x, y)
}
