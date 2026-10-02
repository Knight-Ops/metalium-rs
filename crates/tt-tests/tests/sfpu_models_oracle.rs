//! Are `tt_isa::numerics::sfpu`'s ports really the pages' models?
//!
//! `build.rs` extracts each ported function's C from the pinned instruction
//! page and compiles it; this compares the two over every input that can reach
//! a lookup table or a branch (the functions depend on the high sixteen bits
//! and pass the low sixteen through or ignore them). The same reason as
//! `fma_oracle`: the device gates are asserted against these ports bit for
//! bit, so an error here would look like a hardware finding.

#![cfg(have_sfpu_models)]

use tt_isa::numerics::sfpu::{
    approx_exp, approx_recip, lut16_to_fp32, lut8_to_fp32, sign_mag_is_smaller,
};

#[link(name = "sfpumodels", kind = "static")]
extern "C" {
    #[link_name = "ApproxRecip"]
    fn c_approx_recip(x: u32) -> u32;
    #[link_name = "ApproxExp"]
    fn c_approx_exp(x: u32) -> u32;
    #[link_name = "SignMagIsSmaller"]
    fn c_sign_mag_is_smaller(c: u32, d: u32) -> bool;
    #[link_name = "Lut8ToFp32"]
    fn c_lut8_to_fp32(x: u8) -> u32;
    #[link_name = "Lut16ToFp32"]
    fn c_lut16_to_fp32(x: u16) -> f32;
}

#[test]
fn approx_recip_and_approx_exp_are_the_page_s() {
    for hi in 0..=0xffffu32 {
        for lo in [0u32, 1, 0x7fff, 0x8000, 0xfffe, 0xffff] {
            let x = (hi << 16) | lo;
            // Safe: pure functions of one integer, from the pinned pages.
            let (r, e) = unsafe { (c_approx_recip(x), c_approx_exp(x)) };
            assert_eq!(approx_recip(x), r, "ApproxRecip({x:#010x})");
            assert_eq!(approx_exp(x), e, "ApproxExp({x:#010x})");
        }
    }
}

/// Every LUT code, both widths: the decodings are tables in all but name.
#[test]
fn the_lut_decodings_are_the_page_s() {
    for x in 0..=u8::MAX {
        // Safe: pure functions of one integer, from the pinned pages.
        assert_eq!(
            lut8_to_fp32(x),
            unsafe { c_lut8_to_fp32(x) },
            "Lut8ToFp32({x:#04x})"
        );
    }
    for x in 0..=u16::MAX {
        let c = unsafe { c_lut16_to_fp32(x) }.to_bits();
        assert_eq!(lut16_to_fp32(x), c, "Lut16ToFp32({x:#06x})");
    }
}

/// The order every compare, `SFPSWAP` and the interpreter's `SFPGT`/`SFPLE`
/// rest on, over each pair of a set reaching every sign, exponent class and
/// the remap's boundary bits (`>> 30`).
#[test]
fn sign_mag_is_smaller_is_the_page_s() {
    let mut v: Vec<u32> = vec![
        0,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0x3fff_ffff,
        0x4000_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0x7f80_0001,
        0x7fc0_0000,
        0x7fff_ffff,
    ];
    let neg: Vec<u32> = v.iter().map(|x| x | 0x8000_0000).collect();
    v.extend(neg);
    let mut x = 0x9e37_79b9u32;
    for _ in 0..200 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        v.push(x);
    }
    for &c in &v {
        for &d in &v {
            let want = unsafe { c_sign_mag_is_smaller(c, d) };
            assert_eq!(sign_mag_is_smaller(c, d), want, "({c:#x}, {d:#x})");
        }
    }
}
