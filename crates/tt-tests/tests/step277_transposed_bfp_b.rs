//! A product with a transposed `B` stored block-float (`docs/plans/block-float-kv.md` K2: a BFP8
//! or BFP4 KV cache's `q k^T`): `record::GATHER_B_UNPACK_TRANSPOSE` with a BFP `B` has the mover
//! read each of `B`'s tiles whole from its slot, as it lies, and the kernel's unpacker decodes it
//! into `SrcA` and transposes it within each face (`Haloize_mode`) as it does an FP32 or BF16 one
//! (`step201`, `step272`). Before it, a transposed BFP `B` was refused by the kernels.
//!
//! The claims, every one bit for bit against the same product with `B` decoded to FP32 on the
//! card (`Session::bfp_to_f32`, exact: a block float's values have at most six fraction bits, which
//! TF32 holds) and transposed by the unpacker:
//! - **every path a product takes:** whole tiles, a ragged `B` (zero padding, as a cast leaves),
//!   a thin `A` (`m = 1`, a decode step's query), `K` in blocks;
//! - BFP8 and BFP4;
//! - **the transpose is real:** a square `B` read untransposed gives another product.

use tt_kernels::bfp::BfpFormat;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Elem};
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        use tt_tests::backend::GATE_TILE;
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let tiles = TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1);
        let mut s = Session::open(dev, tt_firmware_images::ROLES, tiles, |_, _| Ok(None)).unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(4),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Values of several magnitudes a row (so a face-row's shared exponent truncates the small ones),
/// signed zeros among them.
fn f32_values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            match i % 53 {
                3 => 0.0,
                29 => -0.0,
                _ => x * f32::powi(2.0, (s >> 20) as i32 % 9 - 4),
            }
        })
        .collect()
}

fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
}

/// `A [m, k] @ B^T`, `B` stored `[n, k]` as `format`, against the same `B` decoded.
fn check<T: tt_device::Transport>(
    s: &mut Session<T>,
    what: &str,
    format: BfpFormat,
    [m, k, n]: [usize; 3],
) {
    let what = format!("{what}, {format:?}");
    let a = s
        .upload(&f32_values(m as u64 * 7 + k as u64, m * k), m, k)
        .unwrap();
    let f = s
        .upload(&f32_values(n as u64 * 31 + k as u64, n * k), n, k)
        .unwrap();
    let packed = s.bfp_from_f32(&f, format).unwrap();
    let wide = s.bfp_to_f32(&packed).unwrap();
    let b = DramTensor::from_bfp(packed);
    let product = |s: &mut Session<T>, b: &DramTensor| {
        s.matmul_dram_as(
            &a,
            false,
            b,
            true,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            1 << 40,
            Elem::F32,
        )
        .unwrap_or_else(|e| panic!("{what}: {e}"))
    };
    let want = product(s, &wide);
    let got = product(s, &b);
    assert_bits(
        &what,
        &s.download(&got).unwrap(),
        &s.download(&want).unwrap(),
    );
    for t in [a, f, b, wide, want, got] {
        s.free(t).unwrap();
    }
}

#[test]
fn a_transposed_bfp_b_is_its_decoded_product_on_every_path() {
    with_session(|s| {
        for format in [BfpFormat::Bfp8, BfpFormat::Bfp4] {
            check(s, "whole tiles", format, [64, 96, 64]);
            check(s, "a ragged B", format, [40, 70, 33]);
            check(s, "a thin A", format, [1, 256, 96]);
            s.set_matmul_k_block_limit(std::num::NonZeroUsize::new(2));
            check(s, "K in blocks", format, [32, 192, 64]);
            s.set_matmul_k_block_limit(None);
        }
    });
}

#[test]
fn a_transposed_bfp_b_is_not_read_untransposed() {
    with_session(|s| {
        let k = 64;
        let a = s.upload(&f32_values(3, 32 * k), 32, k).unwrap();
        let f = s.upload(&f32_values(5, k * k), k, k).unwrap();
        let b = DramTensor::from_bfp(s.bfp_from_f32(&f, BfpFormat::Bfp8).unwrap());
        let product = |s: &mut Session<_>, transposed| {
            let c = s
                .matmul_dram_as(
                    &a,
                    false,
                    &b,
                    transposed,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    1 << 40,
                    Elem::F32,
                )
                .unwrap();
            let v = s.download(&c).unwrap();
            s.free(c).unwrap();
            v
        };
        let (t, plain) = (product(s, true), product(s, false));
        assert_ne!(
            t.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "a square B's transpose gives another product"
        );
        for x in [a, f, b] {
            s.free(x).unwrap();
        }
    });
}
