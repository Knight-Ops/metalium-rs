//! Stress: pipelined matmuls (`Session::set_pipeline`, on by default) queued
//! back to back with element-wise ops and reductions between them, each
//! pipelined where it pays, for as long as
//! `STRESS_SECS` says (default 60 s on silicon, two rounds on the simulator),
//! every result checked against the plain path's bits.
//!
//! What a pipelined list could get wrong under load: a block's scatter still
//! in flight when a later op on the same unit gathers it, a half of the arena
//! overwritten while its kernel still reads it, a `KERNEL_WAIT` naming the
//! wrong generation once the ring wraps. Nothing is synced between a round's
//! ops, so each queues behind the last; the round's downloads are its only
//! waits. `STRESS_TILES` picks the tile counts (default 1 and 8, where the
//! shapes below pipeline); `STRESS_NC=1` puts the scatters on NC.

use std::time::{Duration, Instant};

use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
use tt_kernels::tensor::{pipelining_pays, Eltwise};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;
const SHAPES: [[usize; 3]; 4] = [
    [512, 512, 512],
    [1024, 256, 1024],
    [256, 1024, 256],
    [384, 512, 384],
];

fn secs() -> Duration {
    let default = if tt_tests::backend::ON_SILICON { 60 } else { 0 };
    Duration::from_secs(
        std::env::var("STRESS_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default),
    )
}

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

fn same_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
}

fn soak<T: tt_device::Transport>(s: &mut Session<T>, tiles: usize) {
    let add = Eltwise {
        kind: kind::ADD,
        scalar: 0.0,
        scalar2: 0.0,
    };
    // Per shape: its operands, and the plain path's product and product + b'
    // (b' an operand of the product's shape), the bits every round must match.
    let mut cases = Vec::new();
    s.set_pipeline(false);
    for (k, &[m, kk, n]) in SHAPES.iter().enumerate() {
        let a = s.upload(&floats(k as u64 * 3 + 1, m * kk), m, kk).unwrap();
        let b = s.upload(&floats(k as u64 * 3 + 2, kk * n), kk, n).unwrap();
        let bias = s.upload(&floats(k as u64 * 3 + 3, m * n), m, n).unwrap();
        let c = s
            .matmul_dram(&a, false, &b, false, ROUTE, Fidelity::HiFi4, BUDGET)
            .unwrap();
        let d = s.eltwise(add, &c, Some(&bias)).unwrap();
        let e = s.reduce(&d, ReduceOp::Max, Axis::Rows).unwrap();
        let (cv, dv) = (s.download(&c).unwrap(), s.download(&d).unwrap());
        let ev = s.download(&e).unwrap();
        for t in [c, d, e] {
            s.free(t).unwrap();
        }
        cases.push(([m, kk, n], a, b, bias, cv, (dv, ev)));
    }
    assert!(
        SHAPES.iter().any(|&sh| pipelining_pays(
            sh,
            ROUTE,
            Fidelity::HiFi4,
            s.lists_per_tile().len()
        )),
        "{tiles} tiles: no shape pipelines, so this stresses nothing"
    );
    s.set_pipeline(true);
    // `STRESS_NC=1`: the scatters on NC (`Session::set_scatter_mover`).
    if std::env::var_os("STRESS_NC").is_some() {
        s.set_scatter_mover(Some(tt_firmware_images::DM_NC.1));
    }
    let deadline = Instant::now() + secs();
    let mut round = 0usize;
    let before = s.pipelined_blocks();
    while round < 2 || Instant::now() < deadline {
        // Every shape's matmul and add queued, in a rotating order, then all
        // downloaded.
        let mut outs = Vec::new();
        for j in 0..cases.len() {
            let i = (round + j) % cases.len();
            let (_, a, b, bias, _, _) = &cases[i];
            let c = s
                .matmul_dram(a, false, b, false, ROUTE, Fidelity::HiFi4, BUDGET)
                .unwrap_or_else(|e| panic!("{tiles} tiles, round {round}: {e}"));
            let d = s
                .eltwise(add, &c, Some(bias))
                .unwrap_or_else(|e| panic!("{tiles} tiles, round {round}: {e}"));
            let e = s
                .reduce(&d, ReduceOp::Max, Axis::Rows)
                .unwrap_or_else(|e| panic!("{tiles} tiles, round {round}: {e}"));
            outs.push((i, c, d, e));
        }
        for (i, c, d, e) in outs {
            let (shape, _, _, _, cv, (dv, ev)) = &cases[i];
            let what = format!("{tiles} tiles, round {round}, {shape:?}");
            same_bits(&s.download(&c).unwrap(), cv, &format!("{what}, product"));
            same_bits(&s.download(&d).unwrap(), dv, &format!("{what}, sum"));
            same_bits(&s.download(&e).unwrap(), ev, &format!("{what}, max"));
            s.free(e).unwrap();
            s.free(c).unwrap();
            s.free(d).unwrap();
        }
        round += 1;
    }
    println!(
        "{tiles} tiles: {round} rounds, {} blocks overlapped, all bits equal",
        s.pipelined_blocks() - before
    );
    for (_, a, b, bias, _, _) in cases {
        for t in [a, b, bias] {
            s.free(t).unwrap();
        }
    }
}

#[test]
#[ignore = "stress: long-running"]
fn pipelined_matmuls_queued_back_to_back_keep_the_plain_paths_bits() {
    let counts: Vec<usize> = match std::env::var("STRESS_TILES") {
        Ok(v) => v.split(',').map(|n| n.trim().parse().unwrap()).collect(),
        Err(_) => vec![1, 8],
    };
    for tiles in counts {
        if let Err(e) = fork_scope(|| {
            #[cfg(feature = "silicon")]
            let mut s = Session::open_card(
                tt_tests::backend::device_index(),
                tt_firmware_images::ROLES,
                TileChoice::Count(tiles),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            #[cfg(not(feature = "silicon"))]
            let mut sim = tt_ttsim::Simulator::open().unwrap();
            #[cfg(not(feature = "silicon"))]
            let mut s = Session::open(
                tt_device::Device::open(sim.transport()).unwrap(),
                tt_firmware_images::ROLES,
                TileChoice::Count(tiles),
                |_, _| Ok(None),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
            soak(&mut s, tiles);
        }) {
            panic!("{tiles} tiles: {e}");
        }
    }
}
