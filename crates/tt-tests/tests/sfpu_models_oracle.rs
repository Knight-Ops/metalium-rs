//! Are `tt_isa::numerics::sfpu`'s ports really the pages' models?
//!
//! `build.rs` extracts each ported function's C from the pinned instruction
//! page and compiles it; this compares the two over every input that can reach
//! a lookup table or a branch (the functions depend on the high sixteen bits
//! and pass the low sixteen through or ignore them). The same reason as
//! `fma_oracle`: the device gates are asserted against these ports bit for
//! bit, so an error here would look like a hardware finding.

#![cfg(have_sfpu_models)]

use tt_isa::numerics::sfpu::{approx_exp, approx_recip};

#[link(name = "sfpumodels", kind = "static")]
extern "C" {
    #[link_name = "ApproxRecip"]
    fn c_approx_recip(x: u32) -> u32;
    #[link_name = "ApproxExp"]
    fn c_approx_exp(x: u32) -> u32;
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
