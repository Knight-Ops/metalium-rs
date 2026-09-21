//! Is `tt_isa::numerics::fma_bh` really the specification's model?
//!
//! `fma_bh` is a hand port of `Miscellaneous/FMA/fma.c:102`, and hand ports drift.
//! This links the **actual C** — compiled by `build.rs` from the tree `PINS.toml`
//! pins — and compares the two over a large deterministic sample plus every edge
//! case the accompanying `README.md` names.
//!
//! That makes the oracle the document rather than a copy of it, which is what the
//! tolerance policy needs: Phase 5's gates assert device results against `fma_bh`
//! bit for bit, so an error here would be invisible and would look like a hardware
//! finding.

#![cfg(have_fma_oracle)]

use tt_isa::numerics::fma_bh;

// Named explicitly rather than relying on `build.rs`'s `cargo:rustc-link-lib` to
// reach this target: that directive is emitted for the package, and an integration
// test is a separate crate that did not pick it up.
#[link(name = "fmaoracle", kind = "static")]
extern "C" {
    fn fma_model_bh(x: u32, y: u32, z: u32) -> u32;
    fn fma_model_ieee(x: u32, y: u32, z: u32) -> u32;
}

fn c_bh(x: u32, y: u32, z: u32) -> u32 {
    // Safe: a pure function over three integers, from the specification's own
    // sources, with no state and no allocation.
    unsafe { fma_model_bh(x, y, z) }
}

fn c_ieee(x: u32, y: u32, z: u32) -> u32 {
    unsafe { fma_model_ieee(x, y, z) }
}

/// A deterministic RNG. CI is flake-free on purpose, so the sample is the same on
/// every run and a failure is reproducible from the seed alone.
struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        // SplitMix64, which is short enough to carry rather than depend on.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 16) as u32
    }

    /// A value biased towards the interesting parts of the FP32 space: exponents
    /// near zero, near the denormal boundary, and near infinity.
    fn next_float_bits(&mut self) -> u32 {
        let r = self.next_u32();
        match r % 8 {
            // Fully random bit patterns, which land mostly in ordinary exponents.
            0..=3 => self.next_u32(),
            // Small exponents, so denormals and underflow get exercised.
            4 => (r & 0x8000_0000) | ((self.next_u32() % 8) << 23) | (self.next_u32() & 0x7f_ffff),
            // Large exponents, so overflow and infinity get exercised.
            5 => {
                (r & 0x8000_0000)
                    | ((248 + self.next_u32() % 8) << 23)
                    | (self.next_u32() & 0x7f_ffff)
            }
            // Exact small integers, where the arithmetic is easy to reason about.
            6 => ((self.next_u32() % 64) as f32 - 32.0).to_bits(),
            // Named edge cases.
            _ => {
                const EDGES: [u32; 10] = [
                    0x0000_0000, // +0
                    0x8000_0000, // -0
                    0x3f80_0000, // 1.0
                    0xbf80_0000, // -1.0
                    0x7f80_0000, // +inf
                    0xff80_0000, // -inf
                    0x7fc0_0000, // canonical NaN
                    0x7f80_0001, // signalling NaN
                    0x0000_0001, // smallest denormal
                    0x007f_ffff, // largest denormal
                ];
                EDGES[(self.next_u32() as usize) % EDGES.len()]
            }
        }
    }
}

#[test]
fn the_port_matches_the_specifications_c_over_a_large_sample() {
    let mut rng = Rng(0x5EED_1234_5678_9ABC);
    let mut checked = 0u32;
    for _ in 0..200_000 {
        let x = rng.next_float_bits();
        let y = rng.next_float_bits();
        let z = rng.next_float_bits();
        let want = c_bh(x, y, z);
        let got = fma_bh(x, y, z);
        assert_eq!(
            got, want,
            "fma_bh({x:#010x}, {y:#010x}, {z:#010x}) = {got:#010x}, the C says {want:#010x}"
        );
        checked += 1;
    }
    // A sample that silently became empty would pass every assertion in it.
    assert_eq!(checked, 200_000);
}

#[test]
fn the_port_matches_the_c_on_every_named_edge_case() {
    const VALUES: [u32; 14] = [
        0x0000_0000,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x4000_0000,
        0xc000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x7f80_0001,
        0xffc0_0000,
        0x0000_0001,
        0x007f_ffff,
        0x7f7f_ffff,
    ];
    for &x in &VALUES {
        for &y in &VALUES {
            for &z in &VALUES {
                assert_eq!(
                    fma_bh(x, y, z),
                    c_bh(x, y, z),
                    "fma_bh({x:#010x}, {y:#010x}, {z:#010x})"
                );
            }
        }
    }
}

/// The port is not merely an expensive `f32::mul_add`.
///
/// Without this the differential above would pass on a port that had quietly
/// become the host's own arithmetic, which is precisely the failure that would make
/// every Phase 5 gate vacuous.
///
/// Each case asserts the *divergence* rather than a hand-written expected value.
/// Writing the expectation out by hand is how the first version of this test got it
/// wrong: `fma_bh(2^127, 2^127, -inf)` looked like it should be NaN, because the
/// product overflows and the addend is the opposite infinity — but the model's NaN
/// clause needs `x` or `y` to *be* infinite, not merely to overflow, so the answer
/// is `-inf`. The C and the port agreed; only the expectation was wrong.
#[test]
fn the_model_diverges_from_the_host_where_the_readme_says_it_does() {
    /// Assert `fma_bh` matches the C, and that both differ from the host's fused
    /// multiply-add. Returns what the model produced, for cases worth naming.
    #[track_caller]
    fn diverges(x: u32, y: u32, z: u32) -> u32 {
        let model = fma_bh(x, y, z);
        assert_eq!(model, c_bh(x, y, z), "the port must match the C");
        let host = f32::from_bits(x)
            .mul_add(f32::from_bits(y), f32::from_bits(z))
            .to_bits();
        assert_ne!(
            model, host,
            "fma_bh({x:#010x}, {y:#010x}, {z:#010x}) = {model:#010x}, and the host \
             agrees; this case does not demonstrate a divergence"
        );
        model
    }

    // Denormal inputs are treated as signed zero; the host keeps them.
    assert_eq!(
        diverges(0x0000_0001, 2.0f32.to_bits(), 0),
        0,
        "a denormal operand flushes, so the product is zero"
    );

    // Denormal outputs are flushed after rounding, preserving sign.
    assert_eq!(
        diverges(0x0080_0000, 0.5f32.to_bits(), 0),
        0,
        "a denormal result flushes to zero"
    );

    // If `x * y` on its own would overflow, the result is infinity even where the
    // fused operation would have been finite. `2^64 * 2^64 - FLT_MAX` is the
    // smallest such case that fits: the product is `2^128`, which overflows FP32,
    // but subtracting `FLT_MAX` from it leaves `2^104`, which does not. The host
    // and the IEEE model in the same file both give `2^104`; Blackhole gives
    // infinity because it has already discarded the product.
    assert_eq!(
        diverges(0x5f80_0000, 0x5f80_0000, 0xff7f_ffff),
        0x7f80_0000,
        "an overflowing product gives infinity rather than a fused finite result"
    );
    assert_eq!(
        c_ieee(0x5f80_0000, 0x5f80_0000, 0xff7f_ffff),
        0x7380_0000,
        "the IEEE model in the same file keeps the finite result"
    );

    // And the sign of a zero result: `-1.0 * 0.0` is `+0` unless the addend is
    // `-0`, which is why `tt_isa::sfpu::mul` passes `-0` (divergence row C).
    assert_eq!(fma_bh(0xbf80_0000, 0x0000_0000, 0x0000_0000), 0x0000_0000);
    assert_eq!(
        tt_isa::numerics::mul_bh(0xbf80_0000, 0x0000_0000),
        0x8000_0000,
        "mul_bh passes -0 as the addend, which is what preserves the sign"
    );
}

/// The IEEE model in the same file disagrees with the Blackhole one, in the ways
/// the `README.md` enumerates.
///
/// This is the cross-check that the *pair* of models is what it claims to be. If
/// both were the same function, `fma_bh` matching the C would prove nothing about
/// Blackhole specifically.
#[test]
fn the_blackhole_model_is_not_the_ieee_one() {
    let denormal = 0x0000_0001u32;
    let two = 2.0f32.to_bits();
    assert_ne!(
        c_bh(denormal, two, 0),
        c_ieee(denormal, two, 0),
        "the two models must differ on denormal inputs"
    );

    // And where they agree, our port agrees too -- so the divergence above is the
    // model's, not the port's.
    let three = 3.0f32.to_bits();
    assert_eq!(c_bh(three, two, 0), c_ieee(three, two, 0));
    assert_eq!(fma_bh(three, two, 0), 6.0f32.to_bits());
}
