//! Independent binary16 oracles shared by the FP16 gates (`step143`, `step144`).
//! Written from the IEEE 754 definition with `f64` arithmetic and
//! `round_ties_even`: independent of the SFPU programs and of
//! `tt_kernels::fp16::host`.
#![allow(dead_code)]

/// IEEE binary16 bits of an FP32, round to nearest even, by scaling in `f64`.
pub fn oracle_to_f16(bits: u32) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let x = f32::from_bits(bits);
    if x.is_nan() {
        // The `half` crate's rule: keep the top ten payload bits, force quiet.
        return sign | 0x7c00 | ((bits >> 13) & 0x3ff) as u16 | 0x200;
    }
    if x.is_infinite() {
        return sign | 0x7c00;
    }
    let a = (x.abs()) as f64;
    if a == 0.0 {
        return sign;
    }
    let exponent = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let quantum = 2f64.powi(exponent.max(-14) - 10);
    let n = (a / quantum).round_ties_even();
    let value = n * quantum;
    if value >= 65520.0 {
        return sign | 0x7c00;
    }
    if value < 2f64.powi(-14) {
        return sign | n as u16;
    }
    let e = ((value.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let m = ((value / 2f64.powi(e) - 1.0) * 1024.0) as u16;
    sign | (((e + 15) as u16) << 10) | m
}

/// FP32 bits of an IEEE binary16, from the value formula.
pub fn oracle_to_f32(h: u16) -> u32 {
    let sign = (h as u32 & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    if e == 31 {
        return if m == 0 {
            sign | 0x7f80_0000
        } else {
            sign | 0x7fc0_0000 | (m << 13)
        };
    }
    let value = if e == 0 {
        m as f64 * 2f64.powi(-24)
    } else {
        (1.0 + m as f64 / 1024.0) * 2f64.powi(e - 15)
    };
    sign | (value as f32).to_bits()
}

/// The truncating mutant of [`oracle_to_f16`], for sensitivity checks.
pub fn truncating_to_f16(bits: u32) -> u16 {
    let a = bits & 0x7fff_ffff;
    let r = oracle_to_f16(bits);
    if (0x3880_0000..0x477f_f000).contains(&a) {
        // Normal range: drop the low thirteen mantissa bits and rebias.
        (((bits >> 16) & 0x8000) as u16) | ((((a >> 13) as i32) - (112 << 10)) as u16)
    } else {
        r
    }
}

pub fn lcg(s: &mut u64) -> u32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 32) as u32
}

/// Every class narrowing distinguishes, then ties at many exponents (halfway
/// points with even and odd neighbours, and one ulp either side), then random
/// words half raw and half inside the binary16 exponent range.
pub fn f32_corpus(n: usize) -> Vec<u32> {
    let mut v = vec![
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fc1_2345,
        0xffc5_4321,
        0x7fff_ffff,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x477f_e000,
        0x477f_efff,
        0x477f_f000,
        0x477f_f001,
        0x4780_0000,
        0x4788_b800,
        0xc77f_f000,
        0x3880_0000,
        0x387f_e000,
        0x387f_dfff,
        0x3800_0000,
        0x3380_0000,
        0x3300_0000,
        0x3300_0001,
        0x33c0_0000,
        0x3420_0000,
        0xb3c0_0000,
    ];
    for exponent in 100..=142u32 {
        for k in [0u32, 1, 2, 3, 0x3ff] {
            let tie = (exponent << 23) | (k << 13) | 0x1000;
            v.extend([tie, tie - 1, tie + 1, tie | 0x8000_0000]);
        }
    }
    let mut s = 0x1234_5678u64;
    while v.len() < n {
        let w = lcg(&mut s);
        v.push(if v.len() % 2 == 0 {
            w
        } else {
            (w & 0x807f_ffff) | ((100 + w % 50) << 23)
        });
    }
    v.truncate(n);
    v
}
