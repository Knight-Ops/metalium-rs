//! A matmul's transposed `B` transposed by the kernel's unpacker (`docs/plans/decode-attention.md`
//! A3): `record::GATHER_B_UNPACK_TRANSPOSE` has the mover read each of `B`'s tiles as it lies, and
//! unpacker 0 transposes it within each face (`Haloize_mode`) while the kernel unpacks `B`'s face
//! `(k, fj)` from the slot's face `(fj, k)`. The mover's own transpose (`READ_TRANSPOSED`) moves
//! a tile one datum at a time, ~10 us on silicon: a decode step's `q @ K^T` read the cache at
//! ~30 GB/s.
//!
//! The claims, every one bit for bit:
//! - **the same product as the mover's transpose** (`Session::set_unpack_transposed_b(false)`),
//!   and as the plain product of `B^T` uploaded transposed by the host, on data with every
//!   special value (signed zeros, subnormals, infinities, NaN payloads, the largest finite);
//! - **every path a matmul takes:** whole tiles, a ragged `B` (rows and columns not whole
//!   tiles), a thin `A` (`m = 1`), `K` in blocks (`set_matmul_k_block_limit`), `A` transposed too,
//!   and a batched product (`matmul_dram_batched`, an attention's `q k^T`).
//!
//! Watched failing: with the faces unpacked in the untransposed order, and with `Haloize_mode`
//! left off (the products differ from the mover's).

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{Block, DramTensor};
use tt_tests::harness::BUDGET;
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

/// Ordinary values of several magnitudes, with every special value among them.
fn values(seed: u64, n: usize) -> Vec<f32> {
    const SPECIAL: [u32; 10] = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x807f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0x7f7f_ffff,
        0x0080_0000,
        0x6080_0000,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if i % 37 == 11 {
                return f32::from_bits(SPECIAL[(s >> 33) as usize % SPECIAL.len()]);
            }
            let scale = [1.0e-3f32, 1.0, 37.0][i % 3];
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| v[(i % rows) * cols + i / rows])
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn mm<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    at: bool,
    b: &DramTensor,
    bt: bool,
) -> Vec<f32> {
    let c = s
        .matmul_dram(
            a,
            at,
            b,
            bt,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET,
        )
        .unwrap();
    let v = s.download(&c).unwrap();
    s.free(c).unwrap();
    v
}

#[test]
fn an_unpacker_transposed_b_is_the_movers_product_bit_for_bit() {
    with_session(|s| {
        // `[m, k] @ [n, k]^T`, with the K-block limit (tiles) each case runs under.
        let cases: [(usize, usize, usize, Option<usize>, &str); 6] = [
            (64, 64, 96, None, "whole tiles"),
            (37, 70, 45, None, "ragged"),
            (1, 128, 64, None, "thin A (m = 1)"),
            (32, 256, 32, Some(2), "K in blocks of 2 tiles"),
            (33, 96, 70, Some(1), "ragged, K in blocks of 1 tile"),
            (5, 2048, 64, None, "a decode row, K = 2048"),
        ];
        for (m, k, n, limit, what) in cases {
            for a_transposed in [false, true] {
                let label = format!("[{m}, {k}] @ [{n}, {k}]^T, {what}, A^T {a_transposed}");
                let av = values((m * 1000 + k) as u64, m * k);
                let bv = values((n * 1000 + k + 7) as u64, n * k);
                let a = if a_transposed {
                    s.upload(&transpose(&av, m, k), k, m).unwrap()
                } else {
                    s.upload(&av, m, k).unwrap()
                };
                let b = s.upload(&bv, n, k).unwrap();
                let b_plain = s.upload(&transpose(&bv, n, k), k, n).unwrap();
                s.set_matmul_k_block_limit(limit.and_then(std::num::NonZeroUsize::new));
                s.set_unpack_transposed_b(false);
                let mover = mm(s, &a, a_transposed, &b, true);
                s.set_unpack_transposed_b(true);
                let unpacker = mm(s, &a, a_transposed, &b, true);
                let plain = mm(s, &a, a_transposed, &b_plain, false);
                s.set_matmul_k_block_limit(None);
                assert_eq!(bits(&unpacker), bits(&mover), "{label}: against the mover");
                assert_eq!(bits(&unpacker), bits(&plain), "{label}: against B^T plain");
                for t in [a, b, b_plain] {
                    s.free(t).unwrap();
                }
            }
        }
    });
}

/// An attention's `q k^T` through the batched product: four heads' `[3, 64] @ [80, 64]^T`, each
/// a block of one buffer, as the unpacker and as the mover transpose them.
#[test]
fn a_batched_product_with_b_transposed_is_the_movers_bit_for_bit() {
    with_session(|s| {
        let (heads, m, k, n) = (4usize, 3usize, 64usize, 80usize);
        let qv = values(1, heads * 32 * k);
        let kv = values(2, heads * n * k);
        let q = s.upload(&qv, heads * 32, k).unwrap();
        let kk = s.upload(&kv, heads * n, k).unwrap();
        let items: Vec<(Block, Block)> = (0..heads)
            .map(|h| {
                (
                    Block {
                        at: [h * 32, 0],
                        transposed: false,
                    },
                    Block {
                        at: [h * n, 0],
                        transposed: true,
                    },
                )
            })
            .collect();
        let run = |s: &mut Session<_>, unpack: bool| {
            s.set_unpack_transposed_b(unpack);
            let c = s
                .matmul_dram_batched(
                    &q,
                    &kk,
                    &items,
                    [m, k, n],
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    BUDGET,
                )
                .unwrap();
            let v = s.download(&c).unwrap();
            s.free(c).unwrap();
            v
        };
        let mover = run(s, false);
        let unpacker = run(s, true);
        s.set_unpack_transposed_b(true);
        assert_eq!(bits(&unpacker), bits(&mover));
        assert!(mover.iter().any(|v| v.is_finite() && *v != 0.0));
        for t in [q, kk] {
            s.free(t).unwrap();
        }
    });
}
