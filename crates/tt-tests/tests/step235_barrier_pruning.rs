//! A capture drops the barriers its data does not need (`Session::set_barrier_prune`,
//! `tt_kernels::barriers`, `docs/plans/decode-small-ops.md` S5 A1).
//!
//! The case: three element-wise ops dealt over the same eight units (`x + y`, `(x + y) y`,
//! `(x + y) y + x`), each unit reading only the tiles it wrote, then a product whose every unit
//! reads rows of the last one's output that other units wrote. The two barriers inside the chain
//! are unneeded; the one before the product is not. The oracle is the host's arithmetic on binary
//! operands, exact in FP32 and TF32 alike, so a stale read is a wrong bit, not a tolerance, and
//! the eager run of the same ops on the same operands gives the same bits.
//!
//! Claims:
//! 1. **Pruned where unneeded, kept where not:** the trace holds two ops (the chain as one, labelled
//!    with its three ops, and the product), and each unit's stream two arrivals; with pruning off,
//!    four ops and four arrivals a unit.
//! 2. **Replays are the eager run, bit for bit,** over three sets of fresh operands written in, and
//!    an eager op after each replay that reads its results on other units sees them.
//! 3. **An element-wise op follows its operand's runs** (`Session::set_follow_runs`): RMSNorm's
//!    division deals a `[1, 1024]` row over 16 of 32 units (two tiles a unit); the product by
//!    `gamma` after it takes the same 16, so the barrier between them is unneeded and pruned. With
//!    following off it spreads over 32 and the barrier stays. Replays equal the eager run either way.
//! 4. **The negative control:** with every barrier pruned (`barriers::probe_prune_every_barrier`),
//!    `step208`'s slow op then its readers: the readers' streams no longer wait for it. (A replay's
//!    units did not race on ttsim or card 0, so its bits stay right: see the test.) The
//!    classifier's own controls are `tt_kernels::barriers`' unit tests.

use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const UNITS: usize = 8;

#[cfg(not(feature = "silicon"))]
fn with_tiles(test: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    with_n_tiles(UNITS, test)
}

#[cfg(not(feature = "silicon"))]
fn with_n_tiles(units: usize, test: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    fork_scope(|| {
        let mut simulator = tt_ttsim::Simulator::open().unwrap();
        let device = tt_device::Device::open(simulator.transport()).unwrap();
        let mut session = Session::open(
            device,
            tt_firmware_images::ROLES,
            TileChoice::Count(units),
            |_, _| Ok(None),
        )
        .unwrap();
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        test(&mut session);
    })
    .unwrap();
}

#[cfg(feature = "silicon")]
fn with_tiles(test: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    with_n_tiles(UNITS, test)
}

#[cfg(feature = "silicon")]
fn with_n_tiles(units: usize, test: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    fork_scope(|| {
        let mut session = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(units),
        )
        .unwrap();
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        test(&mut session);
    })
    .unwrap();
}

/// 0 or 1 by a hash: every sum and product below is an integer under 2^15, exact.
fn bits(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i.wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)) >> 7 & 1) as f32)
        .collect()
}

fn host_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[i * k + kk] as f64 * b[kk * n + j] as f64;
            }
            c[i * n + j] = acc as f32;
        }
    }
    c
}

fn mm<T: tt_device::Transport>(s: &mut Session<T>, a: &DramTensor, b: &DramTensor) -> DramTensor {
    s.matmul_dram(
        a,
        false,
        b,
        false,
        SrcRoute::Tf32FromFp32,
        Fidelity::HiFi4,
        BUDGET,
    )
    .unwrap()
}

fn ew<T: tt_device::Transport>(
    s: &mut Session<T>,
    kind: u32,
    a: &DramTensor,
    b: &DramTensor,
) -> DramTensor {
    s.eltwise(
        Eltwise {
            scalar2: 0.0,
            kind,
            scalar: 0.0,
        },
        a,
        Some(b),
    )
    .unwrap()
}

/// The chain's operands, `[M, K]`, 32 tiles over the 8 units; the product's `B`, `[K, N]`.
const M: usize = 64;
const K: usize = 512;
const N: usize = 256;

/// The chain and the product: `(chain's last, product)`, and the intermediates to free.
fn ops<T: tt_device::Transport>(
    s: &mut Session<T>,
    x: &DramTensor,
    y: &DramTensor,
    w: &DramTensor,
) -> (DramTensor, DramTensor, Vec<DramTensor>) {
    let t1 = ew(s, kind::ADD, x, y);
    let t2 = ew(s, kind::MUL, &t1, y);
    let t3 = ew(s, kind::ADD, &t2, x);
    // `[M, K] @ [K, N]`: each unit reads whole rows of `t3`, written by other units.
    let p = mm(s, &t3, w);
    (t3, p, vec![t1, t2])
}

fn host(x: &[f32], y: &[f32], w: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let t3: Vec<f32> = x.iter().zip(y).map(|(&x, &y)| (x + y) * y + x).collect();
    let p = host_matmul(&t3, w, M, K, N);
    (t3, p)
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// The trace of [`ops`] over operands at seed 0, with pruning `on`: its id and tensors.
fn capture<T: tt_device::Transport>(
    s: &mut Session<T>,
    on: bool,
) -> (
    tt_kernels::trace::TraceId,
    [DramTensor; 3],
    (DramTensor, DramTensor, Vec<DramTensor>),
) {
    s.set_barrier_prune(on);
    let x = s.upload(&bits(M * K, 1), M, K).unwrap();
    let y = s.upload(&bits(M * K, 2), M, K).unwrap();
    let w = s.upload(&bits(K * N, 3), K, N).unwrap();
    s.sync().unwrap();
    s.begin_trace().unwrap();
    let out = ops(s, &x, &y, &w);
    let id = s.end_trace().unwrap();
    (id, [x, y, w], out)
}

/// Claims one and two.
#[test]
fn a_same_deal_chain_is_pruned_and_replays_bit_for_bit() {
    with_tiles(|s| {
        // Pruning off: every op keeps its barrier.
        let (id, inputs, (t3, p, rest)) = capture(s, false);
        assert_eq!(s.last_barrier_prune(), None, "nothing pruned with it off");
        assert_eq!(s.trace_ops(id).unwrap().len(), 4);
        let unpruned = s.trace_barrier_entries(id).unwrap();
        s.release_trace(id).unwrap();
        for t in inputs.into_iter().chain([t3, p]).chain(rest) {
            s.free(t).unwrap();
        }

        let (id, [x, y, w], (t3, p, rest)) = capture(s, true);
        let pruned = match s.last_barrier_prune() {
            Some(Ok(p)) => p,
            other => panic!("the capture was not pruned: {other:?}"),
        };
        assert_eq!(
            pruned.ops,
            [4, 2],
            "the chain is one op, the product another"
        );
        let labels: Vec<&str> = s.trace_ops(id).unwrap().iter().map(|o| o.what).collect();
        assert_eq!(labels.len(), 2);
        assert_eq!(
            labels[0].matches(" + ").count(),
            2,
            "the run is named by its three ops: {labels:?}"
        );
        // Each unit: one arrival a run where unpruned it had one an op.
        let entries = s.trace_barrier_entries(id).unwrap();
        for (u, (e, before)) in entries.iter().zip(&unpruned).enumerate() {
            let (e, before) = (
                e.expect("every unit runs"),
                before.expect("every unit runs"),
            );
            assert_eq!(before[0], 4, "unit {u} unpruned: {before:?}");
            assert_eq!(e[0], 2, "unit {u} pruned: {e:?}");
            assert_eq!(e[2], 2, "unit {u}: its count ends at the runs");
        }
        assert_eq!(s.trace_arrivals(id).unwrap(), 2 * UNITS as u32);
        for seed in 1..4 {
            let (hx, hy, hw) = (
                bits(M * K, 10 + seed),
                bits(M * K, 20 + seed),
                bits(K * N, 30 + seed),
            );
            // The eager run first, on the same operands.
            s.write(&x, &hx).unwrap();
            s.write(&y, &hy).unwrap();
            s.write(&w, &hw).unwrap();
            let (e3, ep, erest) = ops(s, &x, &y, &w);
            let (eager3, eagerp) = (s.download(&e3).unwrap(), s.download(&ep).unwrap());
            for t in [e3, ep].into_iter().chain(erest) {
                s.free(t).unwrap();
            }
            s.replay(id).unwrap();
            // An eager op after the replay, on every unit, reading what other units wrote.
            let after = mm(s, &t3, &w);
            let (got3, gotp, got_after) = (
                s.download(&t3).unwrap(),
                s.download(&p).unwrap(),
                s.download(&after).unwrap(),
            );
            s.free(after).unwrap();
            let (want3, wantp) = host(&hx, &hy, &hw);
            assert!(same_bits(&eager3, &want3), "seed {seed}: the eager chain");
            assert!(same_bits(&eagerp, &wantp), "seed {seed}: the eager product");
            assert!(same_bits(&got3, &eager3), "seed {seed}: the replayed chain");
            assert!(
                same_bits(&gotp, &eagerp),
                "seed {seed}: the replayed product"
            );
            assert!(
                same_bits(&got_after, &wantp),
                "seed {seed}: the op after the replay"
            );
        }
        s.release_trace(id).unwrap();
        for t in [x, y, w, t3, p].into_iter().chain(rest) {
            s.free(t).unwrap();
        }
    });
}

/// `step208`'s case: the slow op (one output tile, a long `K`: one job, on unit 0), then its
/// readers (a job on each unit, each reading the slow op's whole output as its `B`).
const SLOW: (usize, usize, usize) = (32, 8192, 32);
const READER: (usize, usize) = (256, 32);

/// Claim four: with every barrier pruned, the readers' streams no longer wait for the slow op.
/// Their bits are reported, not held: a replay's units did not race on either target, so this
/// pruned trace gave the right bits on ttsim (also with `step208`'s release broken) and on card 0
/// (also with the slow op at `K` 65536, run `1793038688`), as `step176`'s dropped waits did
/// (`docs/plans/replay-fixed-cost.md` P1). The stream counts are this control, and the classifier's
/// are `tt_kernels::barriers`' unit tests (a dependency of each kind dropped is watched failing).
#[test]
fn pruning_a_needed_barrier_gives_other_bits() {
    with_tiles(|s| {
        let ((m1, k1, n1), (m2, k2)) = (SLOW, READER);
        let x = s.upload(&bits(m1 * k1, 10), m1, k1).unwrap();
        let wt = s.upload(&bits(k1 * n1, 20), k1, n1).unwrap();
        let z = s.upload(&bits(m2 * k2, 30), m2, k2).unwrap();
        s.sync().unwrap();
        let run = |s: &mut Session<_>, every: bool| {
            tt_kernels::barriers::probe_prune_every_barrier(every);
            s.begin_trace().unwrap();
            let slow = mm(s, &x, &wt);
            let out = mm(s, &z, &slow);
            let id = s.end_trace().unwrap();
            tt_kernels::barriers::probe_prune_every_barrier(false);
            let pruned = s.last_barrier_prune();
            let entries = s.trace_barrier_entries(id).unwrap();
            let mut wrong = 0;
            for seed in 1..9 {
                let (a1, b1, a2) = (
                    bits(m1 * k1, 10 + seed),
                    bits(k1 * n1, 20 + seed),
                    bits(m2 * k2, 30 + seed),
                );
                s.write(&x, &a1).unwrap();
                s.write(&wt, &b1).unwrap();
                s.write(&z, &a2).unwrap();
                s.replay(id).unwrap();
                let want = host_matmul(&a2, &host_matmul(&a1, &b1, m1, k1, n1), m2, k2, n1);
                wrong += usize::from(!same_bits(&s.download(&out).unwrap(), &want));
            }
            s.release_trace(id).unwrap();
            s.free(slow).unwrap();
            s.free(out).unwrap();
            (pruned, entries, wrong)
        };
        let (pruned, entries, wrong) = run(s, false);
        assert_eq!(
            pruned, None,
            "the readers' barrier is needed: nothing pruned"
        );
        for (u, e) in entries.iter().enumerate().skip(1) {
            assert_eq!(
                *e,
                Some([1, 1, 2]),
                "unit {u} waits for the slow op, then arrives"
            );
        }
        assert_eq!(wrong, 0, "the case holds before the control");
        let (pruned, entries, wrong) = run(s, true);
        assert!(
            matches!(pruned, Some(Ok(p)) if p.ops == [2, 1]),
            "{pruned:?}"
        );
        for (u, e) in entries.iter().enumerate() {
            assert_eq!(*e, Some([1, 0, 1]), "unit {u} runs straight to its arrival");
        }
        eprintln!("negative control: {wrong} of 8 replays gave other bits");
        for t in [x, wt, z] {
            s.free(t).unwrap();
        }
    });
}

/// Claim three.
#[test]
fn a_norm_s_scale_follows_its_runs_and_its_barrier_goes() {
    with_n_tiles(32, |s| {
        let k = 1024;
        let add = Eltwise {
            kind: kind::ADD,
            scalar: 0.0,
            scalar2: 0.0,
        };
        let row = |seed: usize| -> Vec<f32> {
            bits(k, seed)
                .iter()
                .enumerate()
                .map(|(i, b)| b + (i % 7) as f32 * 0.25)
                .collect()
        };
        let x = s.upload(&row(1), 1, k).unwrap();
        let dz = s.upload(&row(2), 1, k).unwrap();
        let gamma = s.upload(&row(3), 1, k).unwrap();
        let w = s.upload(&bits(k * 64, 4), k, 64).unwrap();
        let step = |s: &mut Session<_>| {
            let y = s.eltwise(add, &x, Some(&dz)).unwrap();
            let n = s.rms_norm(&y, 1e-5).unwrap();
            let g = ew(s, kind::MUL, &n, &gamma);
            let p = mm(s, &g, &w);
            (p, vec![y, n, g])
        };
        for follow in [false, true] {
            s.set_follow_runs(follow);
            let mut eager = Vec::new();
            for seed in [10usize, 20, 21] {
                s.write(&x, &row(seed)).unwrap();
                let (p, rest) = step(s);
                eager.push(s.download(&p).unwrap());
                for t in rest.into_iter().chain([p]) {
                    s.free(t).unwrap();
                }
            }
            s.write(&x, &row(10)).unwrap();
            s.begin_trace().unwrap();
            let (p, rest) = step(s);
            let id = s.end_trace().unwrap();
            let pruned = s.last_barrier_prune();
            if follow {
                assert!(
                    matches!(pruned, Some(Ok(q)) if q.ops == [4, 3]),
                    "{pruned:?}"
                );
            } else {
                assert_eq!(
                    pruned, None,
                    "the norm and its scale deal apart: nothing pruned"
                );
            }
            let mut got = vec![s.download(&p).unwrap()];
            for seed in [20usize, 21] {
                s.write(&x, &row(seed)).unwrap();
                s.replay(id).unwrap();
                got.push(s.download(&p).unwrap());
            }
            for (i, (g, e)) in got.iter().zip(&eager).enumerate() {
                assert!(
                    same_bits(g, e),
                    "follow {follow}, run {i}: the replay is the eager run"
                );
            }
            s.release_trace(id).unwrap();
            for t in rest.into_iter().chain([p]) {
                s.free(t).unwrap();
            }
        }
        for t in [x, dz, gamma, w] {
            s.free(t).unwrap();
        }
    });
}
