//! decode-to-bandwidth.md Phase 8, the feasibility probe's two switches, checked before they are
//! used on silicon (`silicon_bench_models::traced_generation_stream_ceiling`):
//! - `tensor::probe_skip_weight_reads`: a product with a packed `A` gathers no `B` and nothing
//!   prefetches, so a step's weight stream is free (its products are wrong; only time counts);
//! - `session::probe_background_stream`: every capture's streams on the units named begin with
//!   reads of a scratch tensor into the top of `DATA`, unbarriered, so a replay reads GDDR there
//!   while its ops run.
//!
//! On eight units with every op on four (`Spread::Fixed`), a chain of three one-row products and
//! two element-wise ops between them. The claims:
//! - **Skipping.** Captured with the switch on, before any product has run, the trace holds no
//!   `GATHER` on any unit and its replay completes; the products then come out other than the
//!   ops run fresh (they read no weights: the negative control that the switch did something).
//! - **The background.** Captured with both switches, units 4-7 hold the background's `READ_RUN`s
//!   and nothing else of the ops; the replay completes.
//! - **After the first op.** With an op on all eight units first, the background's reads follow
//!   that op's entries on units 4-7 (`Session::end_trace` moves them), so the replay runs the ops
//!   first and its products, reading weights as usual, are the ops' bits.
//! - **Nothing left behind.** Off again, a capture is the ops run fresh bit for bit, with the
//!   records of a capture taken before the probe (its products prefetching on units 4-7).

use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{probe_background_stream, ProbeBackground, Session, Spread, TileChoice};
use tt_kernels::tensor::{probe_skip_weight_reads, DramTensor, Eltwise};
use tt_ttsim::fork_scope;

const K: usize = 1024;

fn values(seed: u64, n: usize) -> Vec<f32> {
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

fn same_bits(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| g.to_bits() == w.to_bits())
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(8),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_batching(true).unwrap();
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
            TileChoice::Count(8),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_batching(true).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// `x @ w0`, a scale, `@ w1`, a relu, `@ w2`, every op on four units.
fn forward<T: tt_device::Transport>(
    s: &mut Session<T>,
    w: &[DramTensor; 3],
    x: &DramTensor,
) -> DramTensor {
    let mm = |s: &mut Session<T>, a: &DramTensor, b: &DramTensor| {
        s.matmul_dram(
            a,
            false,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            1 << 40,
        )
        .unwrap()
    };
    let ew = |s: &mut Session<T>, kind, scalar, a: &DramTensor| {
        s.eltwise(
            Eltwise {
                kind,
                scalar,
                scalar2: 0.0,
            },
            a,
            None,
        )
        .unwrap()
    };
    let h = mm(s, x, &w[0]);
    let hs = ew(s, kind::MUL_SCALAR, 0.5, &h);
    let g = mm(s, &hs, &w[1]);
    let gr = ew(s, kind::RELU, 0.0, &g);
    let y = mm(s, &gr, &w[2]);
    for t in [h, hs, g, gr] {
        s.free(t).unwrap();
    }
    y
}

/// Each unit's `(GATHER, READ_RUN)` records in trace `id` (`None` for a unit it does not use).
fn records<T: tt_device::Transport>(
    s: &Session<T>,
    id: tt_kernels::trace::TraceId,
) -> Vec<Option<(u32, u32)>> {
    use tt_isa::dm::record;
    s.trace_footprint(id)
        .unwrap()
        .into_iter()
        .map(|u| {
            u.map(|u| {
                let n = |op| u.stream.heads.get(&op).copied().unwrap_or(0);
                (n(record::GATHER), n(record::READ_RUN))
            })
        })
        .collect()
}

/// Capture [`forward`] under the switches as they are set, replay it once, and return its records
/// and the replay's result.
fn capture_and_replay<T: tt_device::Transport>(
    s: &mut Session<T>,
    w: &[DramTensor; 3],
    x: &DramTensor,
) -> (Vec<Option<(u32, u32)>>, Vec<f32>) {
    s.begin_trace().unwrap();
    let y = forward(s, w, x);
    let id = s.end_trace().unwrap();
    s.replay(id).unwrap();
    s.sync().unwrap();
    let got = s.download(&y).unwrap();
    let recs = records(s, id);
    s.release_trace(id).unwrap();
    s.free(y).unwrap();
    (recs, got)
}

#[test]
fn the_probe_switches_skip_weights_and_stream_on_idle_units() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        let w = [(1, K), (2, K), (3, 256)]
            .map(|(seed, n)| s.upload(&values(seed, K * n), K, n).unwrap());
        let x = s.upload(&values(10, K), 1, K).unwrap();

        // Skipping: no `GATHER` anywhere, and the products read no weights. First, before any
        // product has run, so no `B` slot can hold a weight from an earlier read.
        probe_skip_weight_reads(true);
        let (recs, got) = capture_and_replay(s, &w, &x);
        probe_skip_weight_reads(false);
        let fresh = forward(s, &w, &x);
        let want = s.download(&fresh).unwrap();
        s.free(fresh).unwrap();
        assert!(
            recs.iter().flatten().all(|&(gathers, _)| gathers == 0),
            "no GATHER under the probe: {recs:?}"
        );
        assert!(!same_bits(&got, &want), "the products read no weights");
        // Today's capture, for the last claim (its products prefetch on units 4-7).
        let (plain, got) = capture_and_replay(s, &w, &x);
        assert!(
            same_bits(&got, &want),
            "a plain capture is the ops run fresh"
        );

        // The background: units 4-7 read and run nothing else.
        let tiles = 300;
        probe_skip_weight_reads(true);
        probe_background_stream(Some(ProbeBackground {
            first: 4,
            units: 4,
            tiles_per_unit: tiles,
        }));
        let (recs, _) = capture_and_replay(s, &w, &x);
        probe_skip_weight_reads(false);
        probe_background_stream(None);
        let runs = tiles.div_ceil(128) as u32;
        for (u, r) in recs.iter().enumerate().skip(4) {
            assert_eq!(
                *r,
                Some((0, runs)),
                "unit {u}: the background's reads alone"
            );
        }

        // The background after a unit's last op: a first op on all eight units (as a decode
        // step's mask is) puts entries on units 4-7 behind the background's; the capture moves the
        // background after them, so the replay runs the op first and the bits are the ops'.
        probe_background_stream(Some(ProbeBackground {
            first: 4,
            units: 4,
            tiles_per_unit: tiles,
        }));
        s.begin_trace().unwrap();
        s.set_spread(Spread::All);
        let lead = s
            .eltwise(
                Eltwise {
                    kind: kind::MUL_SCALAR,
                    scalar: 1.0,
                    scalar2: 0.0,
                },
                &x,
                None,
            )
            .unwrap();
        s.set_spread(Spread::Fixed(4));
        let y = forward(s, &w, &lead);
        let id = s.end_trace().unwrap();
        probe_background_stream(None);
        s.replay(id).unwrap();
        s.sync().unwrap();
        assert!(
            same_bits(&s.download(&y).unwrap(), &want),
            "the ops under a background moved past the first op"
        );
        for (u, r) in records(s, id).iter().enumerate().skip(4) {
            assert!(
                r.is_some_and(|(_, reads)| reads >= runs),
                "unit {u}: the background's reads, after the first op's: {r:?}"
            );
        }
        s.release_trace(id).unwrap();
        for t in [lead, y] {
            s.free(t).unwrap();
        }

        // Off again: the ops run fresh, bit for bit, captured as before the probe.
        let (recs, got) = capture_and_replay(s, &w, &x);
        assert!(same_bits(&got, &want), "the switches leave nothing behind");
        assert_eq!(recs, plain, "the records of a capture before the probe");
        for t in w.into_iter().chain([x]) {
            s.free(t).unwrap();
        }
    });
}
