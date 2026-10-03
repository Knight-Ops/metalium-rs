//! Checklist 9.15, the reader / writer split: with
//! `Session::set_scatter_mover`, a pipelined group's outputs are written out
//! by the tile's RISCV NC while B gathers and runs the kernels, the two
//! ordered by `SIGNAL` / `WAIT_PEER`. The claims:
//!
//! * a pipelined matmul, element-wise op and reduction give B-only's bits, on
//!   one tile and on two, with ops queued back to back (nothing synced
//!   between them, so one op's NC scatters and the next op's B gathers are in
//!   flight together);
//! * it switches off and on again mid-session, B's and NC's progress counts
//!   carrying on.

use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
use tt_kernels::tensor::{DramTensor, Eltwise};
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

/// A matmul, an add of its product and a max over its rows, queued back to
/// back, then downloaded: the three results.
fn chain<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
) -> [Vec<f32>; 3] {
    let c = s
        .matmul_dram(
            a,
            false,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let d = s
        .eltwise(
            Eltwise {
                kind: kind::ADD,
                scalar: 0.0,
                scalar2: 0.0,
            },
            &c,
            Some(&c),
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let e = s
        .reduce(&d, ReduceOp::Max, Axis::Rows)
        .unwrap_or_else(|e| panic!("{e}"));
    let out = [&c, &d, &e].map(|t| s.download(t).unwrap());
    for t in [c, d, e] {
        s.free(t).unwrap();
    }
    out
}

#[test]
fn scatters_on_nc_are_b_onlys_bits() {
    for units in [1usize, 2] {
        with_tiles(units, |s| {
            // 1024 x 512 @ 512 x 1024: a pipelined matmul on both tile
            // counts, and its 1024-tile product enough for the add and the
            // max to pipeline too.
            let (m, k, n) = (1024, 512, 1024);
            let a = s.upload(&floats(1, m * k), m, k).unwrap();
            let b = s.upload(&floats(2, k * n), k, n).unwrap();
            let want = chain(s, &a, &b);
            for round in 0..2 {
                s.set_scatter_mover(Some(tt_firmware_images::DM_NC.1));
                let before = s.pipelined_blocks();
                let got = chain(s, &a, &b);
                let overlapped = s.pipelined_blocks() - before;
                assert!(overlapped > 0, "{units} tiles: nothing pipelined");
                for (what, (w, g)) in ["product", "sum", "max"].iter().zip(want.iter().zip(&got)) {
                    assert_eq!(w.len(), g.len());
                    for (i, (x, y)) in w.iter().zip(g).enumerate() {
                        assert_eq!(
                            x.to_bits(),
                            y.to_bits(),
                            "{units} tiles, round {round}, {what}: element {i}: {x} vs {y}"
                        );
                    }
                }
                println!(
                    "{units} tiles, round {round}: {overlapped} blocks, NC scattering, bits equal"
                );
                // B alone between rounds: NC idles, and its count carries on.
                s.set_scatter_mover(None);
                let again = chain(s, &a, &b);
                assert!(
                    again == want,
                    "{units} tiles, round {round}: B alone after NC"
                );
            }
            s.free(a).unwrap();
            s.free(b).unwrap();
        });
    }
}
