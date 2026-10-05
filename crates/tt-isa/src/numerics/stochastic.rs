//! Blackhole SFPSTOCHRND_FloatFloat and VectorUnit PRNG functional models.
//! These reproduce documented rounding defects; they are not IEEE conversions.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Precision {
    Tf32 = 0,
    Bf16 = 1,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Rounding {
    Nearest = 0,
    Stochastic = 1,
    TowardZero = 2,
}

/// State after AdvancePRNG returns the supplied old state.
pub const fn advance(state: u32) -> u32 {
    let taps = (state & 0x80200003).count_ones();
    ((!taps & 1) << 31) | (state >> 1)
}

/// The instruction consumes the old PRNG state even in deterministic modes.
/// Nearest ties increase magnitude. All NaNs become signed infinity; zeros
/// and subnormals become positive zero. TowardZero has the documented >= bug.
pub const fn round(x: u32, prng: u32, precision: Precision, rounding: Rounding) -> u32 {
    let exp = (x >> 23) & 255;
    if exp == 0 {
        return 0;
    }
    if exp == 255 {
        return x & 0xff800000;
    }
    let random = match rounding {
        Rounding::Nearest => 0x400000,
        Rounding::TowardZero => 0x7fffff,
        Rounding::Stochastic => prng & 0x7fffff,
    };
    let discard = match precision {
        Precision::Tf32 => 13,
        Precision::Bf16 => 16,
    };
    let mask = (1 << discard) - 1;
    let remainder = x & mask;
    let truncated = x - remainder;
    if remainder >= random >> (23 - discard) {
        truncated.wrapping_add(1 << discard)
    } else {
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn documented_defects_and_prng_boundaries_are_preserved() {
        assert_eq!(round(0x80000000, 0, Precision::Bf16, Rounding::Nearest), 0);
        assert_eq!(
            round(0xffc12345, 0, Precision::Bf16, Rounding::Nearest),
            0xff800000
        );
        assert_eq!(
            round(0x3f808000, 0, Precision::Bf16, Rounding::Nearest),
            0x3f810000
        );
        assert_eq!(
            round(0x3f80ffff, 0, Precision::Bf16, Rounding::TowardZero),
            0x3f810000
        );
        assert_eq!(
            round(0x3f800000, 0, Precision::Bf16, Rounding::Stochastic),
            0x3f810000
        );
        assert_eq!(advance(0), 0x80000000);
        assert_eq!(advance(1), 0);
        assert_eq!(advance(3), 0x80000001);
    }
}
