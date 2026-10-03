//! Checklist 9.15: a unit's mover overlaps its moves with its kernels.
//!
//! With `Session::set_pipeline`, a GDDR matmul's blocks are staged in two
//! halves of the data arena, a unit's consecutive blocks alternating halves,
//! and its list runs `LAUNCH k, scatter k-1, gather k+1, KERNEL_WAIT k`: block
//! k+1 moves in, and block k-1 out, while block k computes. The claims:
//!
//! * the results are the plain path's bits, on one tile and on three, with
//!   either operand transposed;
//! * the overlapped path ran exactly where `tensor::pipelining_pays` says it
//!   pays (`Session::pipelined_blocks`), and on each tile count for at least
//!   one case, so the bits are not the plain path's by default.
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
    // [m, k, n] and the transposes: each takes at least two blocks a unit in
    // half the arena, on one tile and on three, so pipelining pays
    // (`tensor::pipelining_pays`) and a unit has a run to overlap.
    let cases: [([usize; 3], bool, bool); 3] = [
        ([512, 512, 512], false, false),
        ([384, 512, 384], true, false),
        ([512, 512, 512], false, true),
    ];
    for n in [1usize, 3] {
        with_tiles(n, |s| {
            let mut piped_any = false;
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
                let pays =
                    tt_kernels::tensor::pipelining_pays([m, kk, nn], ROUTE, Fidelity::HiFi4, n);
                assert_eq!(
                    overlapped > 0,
                    pays,
                    "{n} tiles, case {k}: overlapped {overlapped} blocks, but pipelining pays: {pays}"
                );
                piped_any |= overlapped > 0;
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
            assert!(piped_any, "{n} tiles: no case pipelined");
        });
    }
}
