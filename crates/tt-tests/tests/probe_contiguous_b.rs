//! The contiguous-weight probe (`tt_kernels::tensor::probe_contiguous_b`,
//! `docs/plans/decode-bytes-in-flight.md`, roadmap Track 2 follow-ups item 2b) on ttsim, before
//! the card times it: a one-row product's weight stream, moved to plain reads of a scratch region
//! in large entries, runs to its end with nothing refused, and its `B` comes from the scratch.
//!
//! The claims, for `down`'s schedule (`K` streamed in chunks, a ring of two) and `gate|up`'s (`K`
//! whole, a ring of one), at tile-sized and 16 KiB entries:
//! - **It runs.** The mover accepts each read (one channel, congruent mod 64, inside the listed
//!   L1) and the product finishes.
//! - **Every byte of `B` moved.** The probe's count of moved bytes is the weight's, and nothing is
//!   left read ahead (one tile, no prefetching units).
//! - **`B` is the scratch's.** The scratch is fresh GDDR, which ttsim holds at zero: every output
//!   is `+0.0`, where the same product with the probe off (the negative control) is not.
#![cfg(not(feature = "silicon"))]

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{probe_contiguous_b, probe_contiguous_b_bytes};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

fn floats(seed: u64, n: usize) -> Vec<f32> {
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

/// `[1, k] @ [k, n]`, `K` streamed or whole, its output downloaded.
fn product<T: tt_device::Transport>(
    s: &mut Session<T>,
    k: usize,
    n: usize,
    streamed: bool,
) -> Vec<f32> {
    let a = s.upload(&floats(k as u64, k), 1, k).unwrap();
    let b = s.upload(&floats(n as u64, k * n), k, n).unwrap();
    s.set_streamed_k(streamed);
    let out = s
        .matmul_dram(
            &a,
            false,
            &b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET,
        )
        .unwrap();
    let got = s.download(&out).unwrap();
    s.set_streamed_k(true);
    for t in [a, b, out] {
        s.free(t).unwrap();
    }
    got
}

#[test]
fn a_weight_stream_reads_its_scratch_in_large_entries() {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        for (k, n, streamed) in [(8192, 32, true), (1024, 256, false)] {
            let what = format!("[1, {k}] @ [{k}, {n}], K streamed {streamed}");
            probe_contiguous_b(None);
            let control = product(&mut s, k, n, streamed);
            assert!(
                control.iter().any(|&v| v != 0.0),
                "{what}: the control computes a product"
            );
            for entry in [2112, 16_384] {
                let at = format!("{what}, {entry}-byte reads");
                probe_contiguous_b(Some(entry));
                let got = product(&mut s, k, n, streamed);
                let [moved, ahead] = probe_contiguous_b_bytes();
                probe_contiguous_b(None);
                println!("{at}: {moved} bytes moved, {ahead} read ahead");
                // Every tile of `B` in its FP32 slot (the tile and its header).
                let slots = (k / 32 * (n / 32)) as u64 * tt_isa::dm::TILE_SLOT;
                assert_eq!(moved, slots, "{at}: every slot of `B` moved");
                assert_eq!(ahead, 0, "{at}: nothing read ahead on one tile");
                assert!(
                    got.iter().all(|&v| v.to_bits() == 0),
                    "{at}: `B` read from the zeroed scratch, not the weight"
                );
            }
        }
    }) {
        panic!("{e}");
    }
}
