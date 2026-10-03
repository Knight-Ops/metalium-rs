//! Checklist 9.15: a unit's mover overlaps its moves with its kernels.
//!
//! With `Session::set_pipeline`, a GDDR matmul's blocks are staged in two
//! halves of the data arena, a unit's consecutive blocks alternating halves,
//! and its list runs `LAUNCH k, scatter k-1, gather k+1, KERNEL_WAIT k`: block
//! k+1 moves in, and block k-1 out, while block k computes. The claims:
//!
//! * the results are the plain path's bits, on one tile and on three, with
//!   either operand transposed;
//! * the overlapped path ran (`Session::pipelined_blocks`), so the bits are
//!   not the plain path's by default.
//!
//! The plain path's own accuracy is `step18_dram_matmul`'s.

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_ttsim::fork_scope;

const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;
use tt_tests::harness::BUDGET;

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

#[cfg(not(feature = "silicon"))]
fn with_tiles(n: usize, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(n),
            |_, _| Ok(None),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{n} tiles: {e}");
    }
}

#[cfg(feature = "silicon")]
fn with_tiles(n: usize, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(n),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{n} tiles: {e}");
    }
}

#[test]
fn pipelined_matmuls_are_the_plain_paths_bits() {
    // [m, k, n] and the transposes: each takes several blocks in half the
    // arena, so a unit has a run to overlap.
    let cases: [([usize; 3], bool, bool); 3] = [
        ([256, 256, 256], false, false),
        ([192, 320, 224], true, false),
        ([160, 256, 576], false, true),
    ];
    for n in [1usize, 3] {
        with_tiles(n, |s| {
            for (k, ([m, kk, nn], ta, tb)) in cases.into_iter().enumerate() {
                let av = floats(k as u64 * 2 + 1, m * kk);
                let bv = floats(k as u64 * 2 + 2, kk * nn);
                let (ar, ac) = if ta { (kk, m) } else { (m, kk) };
                let (br, bc) = if tb { (nn, kk) } else { (kk, nn) };
                let a = s.upload(&av, ar, ac).unwrap();
                let b = s.upload(&bv, br, bc).unwrap();
                let run = |s: &mut _, on: bool| {
                    let s: &mut Session<_> = s;
                    s.set_pipeline(on);
                    let c = s
                        .matmul_dram(&a, ta, &b, tb, ROUTE, Fidelity::HiFi4, BUDGET)
                        .unwrap_or_else(|e| panic!("{e}"));
                    let v = s.download(&c).unwrap();
                    s.free(c).unwrap();
                    v
                };
                let plain = run(s, false);
                let before = s.pipelined_blocks();
                let piped = run(s, true);
                let overlapped = s.pipelined_blocks() - before;
                assert!(overlapped > 0, "{n} tiles, case {k}: nothing overlapped");
                assert_eq!(plain.len(), piped.len());
                for (i, (p, q)) in plain.iter().zip(&piped).enumerate() {
                    assert_eq!(
                        p.to_bits(),
                        q.to_bits(),
                        "{n} tiles, case {k}: element {i}: {p} vs {q}"
                    );
                }
                println!("{n} tiles, case {k}: {overlapped} blocks overlapped, bits equal");
                s.free(a).unwrap();
                s.free(b).unwrap();
            }
        });
    }
}
