//! A product whose `B` is BF16 in L1 and whose `A` is FP32: is it the FP32 product over the same
//! values, bit for bit? (`docs/plans/bf16-weight-storage.md` A0.)
//!
//! A BF16 value is an FP32 one with its low sixteen bits clear, and the unpacker's TF32 keeps the
//! top ten mantissa bits, so in `Src` the two should be the same datum. If the Matrix Unit's
//! phases slice `SrcA` by its bit positions alone, a weight stored BF16 multiplies exactly as the
//! same weight widened to FP32 -- at half the bytes. `SrcRoute::Tf32WithBf16B` (`B` moved as BF16,
//! `A` truncated to TF32) is checked against `SrcRoute::Tf32FromFp32` over the same values at
//! every fidelity, specials included; the control stages `B` values with mantissa bits below
//! BF16's, which the BF16 staging must lose. (BF16 in with TF32 out is the other way to write
//! it; the specification calls a mismatched pair undefined and ttsim refuses it.)
use tt_kernels::matmul::{self, Fidelity, SrcRoute};
use tt_tests::harness;

/// A deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
}

/// FP32 values with every mantissa bit live, exponents over about 2^-20..2^20.
fn activations(rng: &mut Lcg, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| {
            let r = rng.next();
            let exp = 127 - 20 + (r >> 24) % 41;
            f32::from_bits((r & 0x8000_0000) | (exp << 23) | (rng.next() & 0x7f_ffff))
        })
        .collect()
}

/// BF16-valued weights (low sixteen bits clear), exponents over about 2^-12..2^12, with the
/// specials BF16 can hold planted at the front: signed zeros, subnormals, the largest finite,
/// infinities and NaN payloads.
fn bf16_weights(rng: &mut Lcg, count: usize, specials: bool) -> Vec<f32> {
    let mut w: Vec<f32> = (0..count)
        .map(|_| {
            let r = rng.next();
            let exp = 127 - 12 + (r >> 24) % 25;
            f32::from_bits((r & 0x8000_0000) | (exp << 23) | (r & 0x7f_0000))
        })
        .collect();
    if specials {
        let planted: [u32; 10] = [
            0x0000_0000,
            0x8000_0000,
            0x0001_0000,
            0x8040_0000,
            0x7f7f_0000,
            0xff7f_0000,
            0x0080_0000,
            0x3f80_0000,
            0x7f80_0000,
            0x7fc1_0000,
        ];
        // Spread over rows and columns: one column per special, rows 0..k so every output
        // column meets at most one infinity or NaN.
        for (i, bits) in planted.into_iter().enumerate() {
            w[i * 37 % count] = f32::from_bits(bits);
        }
    }
    w
}

fn product(a: &[f32], b: &[f32], mkn: [usize; 3], route: SrcRoute, fidelity: Fidelity) -> Vec<u32> {
    let path = std::env::temp_dir().join(format!(
        "tt232-{}-{:?}.bin",
        std::process::id(),
        std::thread::current().id()
    ));
    harness::in_device(|dev| {
        let c = matmul::matmul(
            dev,
            harness::tensix_tile(),
            &tt_tests::firmware::ROLES,
            a,
            b,
            mkn,
            route,
            fidelity,
            harness::BUDGET,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let bytes: Vec<u8> = c.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
    });
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn first_difference(got: &[u32], want: &[u32]) -> Option<(usize, u32, u32)> {
    got.iter()
        .zip(want)
        .enumerate()
        .find(|(_, (g, w))| g != w)
        .map(|(i, (&g, &w))| (i, g, w))
}

#[test]
fn bf16_b_is_the_fp32_product_bit_for_bit() {
    let mkn = [64, 96, 64];
    let [m, k, n] = mkn;
    let mut rng = Lcg(232);
    let a = activations(&mut rng, m * k);
    for specials in [false, true] {
        let b = bf16_weights(&mut rng, k * n, specials);
        for fidelity in [Fidelity::Lo, Fidelity::HiFi2, Fidelity::HiFi4] {
            let want = product(&a, &b, mkn, SrcRoute::Tf32FromFp32, fidelity);
            assert!(
                want.iter().any(|&w| w != 0),
                "the reference must not be empty"
            );
            let got = product(&a, &b, mkn, SrcRoute::Tf32WithBf16B, fidelity);
            if let Some((i, g, w)) = first_difference(&got, &want) {
                let count = got.iter().zip(&want).filter(|(g, w)| g != w).count();
                panic!(
                    "{fidelity:?}, specials {specials}: {count} of {} differ, \
                     first output {i}: {g:08x} against {w:08x}",
                    got.len()
                );
            }
        }
    }
    // The control: weights with mantissa bits below BF16's. Staged as BF16 they are not the
    // values the FP32 route multiplies, so the products must differ.
    let b: Vec<f32> = bf16_weights(&mut rng, k * n, false)
        .into_iter()
        .map(|v| f32::from_bits(v.to_bits() | 0x0000_7000))
        .collect();
    let want = product(&a, &b, mkn, SrcRoute::Tf32FromFp32, Fidelity::HiFi4);
    let got = product(&a, &b, mkn, SrcRoute::Tf32WithBf16B, Fidelity::HiFi4);
    assert!(
        first_difference(&got, &want).is_some(),
        "BF16 staging must lose the low bits the FP32 route keeps"
    );
}
