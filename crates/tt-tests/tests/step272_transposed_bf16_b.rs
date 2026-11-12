//! A product with a transposed `B` stored BF16 (roadmap W1): `record::GATHER_B_UNPACK_TRANSPOSE`
//! with a BF16 `B` has the mover read each of `B`'s tiles whole from its own 2,112-byte slot, as it
//! lies, and the kernel's unpacker transposes it within each face (`Haloize_mode`) as it moves the
//! BF16 datums to `SrcA` as they are (`SrcRoute::Edges`). Before it, a transposed BF16 `B` was
//! refused by the kernels, and burn-tt copied it plain through a per-element host map: Qwen3.5's
//! tied head `last @ table^T` (248,320 x 2,560) passed 40 GB of host memory on that copy.
//!
//! The claims, every one bit for bit against the same product with `B` widened to FP32 (exactly)
//! and transposed by the unpacker (`step201`), on values with BF16's special patterns among them:
//! - **every path a product takes:** whole tiles, a ragged `B` whose padding is undefined (its
//!   `K` padding met as an FP32 one's is), a thin `A` (`m = 1`), `K` in blocks, and (on silicon) a
//!   BF16 result;
//! - **the transpose is real:** a square `B` read untransposed gives another product.

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Elem, Pad};
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

/// BF16 patterns of several magnitudes, with BF16's special values among them: signed zeros, a
/// subnormal (flushed in `Src` either way), infinities, a NaN, the largest finite.
fn bf16_bits(seed: u64, n: usize) -> Vec<u16> {
    const SPECIAL: [u16; 8] = [
        0x0000, 0x8000, 0x0001, 0x807f, 0x7f80, 0xff80, 0x7fc1, 0x7f7f,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let h = (s >> 33) as u32;
            if i % 97 == 13 {
                SPECIAL[h as usize % SPECIAL.len()]
            } else {
                // Exponents 2^-8 .. 2^7, every mantissa bit live.
                ((h >> 31) as u16) << 15 | ((0x77 + (h >> 8) % 16) as u16) << 7 | (h & 0x7f) as u16
            }
        })
        .collect()
}

fn f32_values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn widened(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect()
}

fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
}

/// `A [m, k] @ B^T`, `B` stored `[n, k]`: in BF16 slots, against the same `B` widened.
/// With `bf16_result`, a BF16 result too (none under `K` blocks, which a BF16 result has no
/// schedule for: its partials would pass through it).
fn check<T: tt_device::Transport>(
    s: &mut Session<T>,
    what: &str,
    [m, k, n]: [usize; 3],
    bf16_result: bool,
) {
    // ttsim packs no BF16 result (below).
    #[cfg(not(feature = "silicon"))]
    let _ = bf16_result;
    let a = s
        .upload(&f32_values(m as u64 * 7 + k as u64, m * k), m, k)
        .unwrap();
    let bits = bf16_bits(n as u64 * 31 + k as u64, n * k);
    // Padding undefined, as a BF16 activation's is: `K`'s is met before the product.
    let b = DramTensor::from_bf16(s.upload_bf16(&bits, n, k).unwrap(), Pad::Undefined);
    let wide = s.upload(&widened(&bits), n, k).unwrap();
    let product = |s: &mut Session<T>, b: &DramTensor, out: Elem| {
        s.matmul_dram_as(
            &a,
            false,
            b,
            true,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            1 << 40,
            out,
        )
        .unwrap_or_else(|e| panic!("{what}: {e}"))
    };
    let want = product(s, &wide, Elem::F32);
    let got = product(s, &b, Elem::F32);
    assert_bits(
        what,
        &s.download(&got).unwrap(),
        &s.download(&want).unwrap(),
    );
    // A BF16 result: the same product rounded once at its pack. Silicon only: ttsim does not pack
    // FP32 to BF16 (`ttsim-divergence.md`, "BF16 late narrowing").
    #[cfg(feature = "silicon")]
    if bf16_result {
        let want16 = product(s, &wide, Elem::BF16);
        let got16 = product(s, &b, Elem::BF16);
        let as_f32 = |s: &mut Session<T>, t: DramTensor| {
            let wide = s.bf16_to_f32(&t.as_bf16().expect("a BF16 result")).unwrap();
            let v = s.download(&wide).unwrap();
            s.free(wide).unwrap();
            v
        };
        let (g, w) = (as_f32(s, got16), as_f32(s, want16));
        assert_bits(&format!("{what}, a BF16 result"), &g, &w);
    }
    for t in [a, b, wide, want, got] {
        s.free(t).unwrap();
    }
}

#[test]
fn a_transposed_bf16_b_is_its_widened_product_on_every_path() {
    with_session(|s| {
        check(s, "whole tiles", [64, 96, 64], true);
        check(s, "a ragged B", [40, 70, 33], true);
        check(s, "a thin A", [1, 160, 96], true);
        s.set_matmul_k_block_limit(std::num::NonZeroUsize::new(2));
        check(s, "K in blocks", [32, 192, 64], false);
        s.set_matmul_k_block_limit(None);
    });
}

#[test]
fn a_transposed_bf16_b_is_not_read_untransposed() {
    with_session(|s| {
        let k = 64;
        let a = s.upload(&f32_values(3, 32 * k), 32, k).unwrap();
        let bits = bf16_bits(5, k * k);
        let b = DramTensor::from_bf16(s.upload_bf16(&bits, k, k).unwrap(), Pad::Zero);
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
        s.free(a).unwrap();
        s.free(b).unwrap();
    });
}
