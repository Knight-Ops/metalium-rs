//! An op's barrier involves only the units it ran on (`Session::catch_up`): a tile an op does
//! not use enqueues nothing for it, and orders itself after it with one wait-only `BARRIER`
//! before its next list.
//!
//! The case that would go wrong without the catch-up: a slow op on unit 0 alone (a matmul
//! with one output tile and a long `K`), then an op on units 0-3 that reads its output (a
//! matmul whose `B` it is). Units 1-3 sat the first op out; unless they wait for its barrier
//! they gather its output before unit 0 has written it. The oracle is the host's product on
//! binary operands, exact in TF32 and FP32 alike, so a stale read is a wrong bit, not a
//! tolerance. The controls: the first op ran on one unit (or the case tests nothing), the
//! others enqueued nothing for it, and the same pair, repeated with fresh operands, is right
//! every time.
//!
//! A capture keeps the rule (`docs/plans/replay-fixed-cost.md` P1): a captured op's barrier
//! counts the units it ran on, a unit that sat ops out waits (a wait-only `BARRIER` in its
//! stream) before its next op and at the trace's end, and a replay adds the arrivals the
//! capture counted. Held to the host's products on fresh operands at every replay, with ops
//! after the replay that read its results on other units.

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::DramTensor;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_tiles(count: usize, test: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    fork_scope(|| {
        let mut simulator = tt_ttsim::Simulator::open().unwrap();
        let device = tt_device::Device::open(simulator.transport()).unwrap();
        let mut session = Session::open(
            device,
            tt_firmware_images::ROLES,
            TileChoice::Count(count),
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
fn with_tiles(count: usize, test: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    fork_scope(|| {
        let mut session = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(count),
        )
        .unwrap();
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        test(&mut session);
    })
    .unwrap();
}

/// 0 or 1 by a hash of the index and seed: every product and sum below is an integer under
/// 2^15, exact in TF32 operands and FP32 accumulation.
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

#[test]
fn a_unit_that_sat_an_op_out_waits_for_it_before_reading_its_output() {
    const UNITS: usize = 4;
    // The slow op: one output tile, so one job, on unit 0; K long enough that it is still
    // running when the next op's units start.
    let (m1, k1, n1) = (32, 1024, 32);
    // The reader: four output tiles, so a job on each of the four units, each gathering the
    // whole of the slow op's output as its `B`.
    let (m2, k2) = (128, 32);
    with_tiles(UNITS, |s| {
        assert_eq!(s.lists_per_tile().len(), UNITS);
        for round in 0..3 {
            let a1 = bits(m1 * k1, 10 + round);
            let b1 = bits(k1 * n1, 20 + round);
            let a2 = bits(m2 * k2, 30 + round);
            let x = s.upload(&a1, m1, k1).unwrap();
            let w = s.upload(&b1, k1, n1).unwrap();
            let z = s.upload(&a2, m2, k2).unwrap();
            s.sync().unwrap();

            let before = s.lists_per_tile();
            let slow = mm(s, &x, &w);
            let after_slow = s.lists_per_tile();
            let out = mm(s, &z, &slow);
            let after_both = s.lists_per_tile();

            // The case is only a test if the slow op ran on one unit and the reader on all.
            let ran_slow: Vec<u64> = after_slow.iter().zip(&before).map(|(a, b)| a - b).collect();
            assert!(
                ran_slow[0] > 0,
                "round {round}: unit 0 ran nothing: {ran_slow:?}"
            );
            assert!(
                ran_slow[1..].iter().all(|&l| l == 0),
                "round {round}: the units the slow op did not run on enqueued for it: {ran_slow:?}"
            );
            let ran_reader: Vec<u64> = after_both
                .iter()
                .zip(&after_slow)
                .map(|(a, b)| a - b)
                .collect();
            assert!(
                ran_reader.iter().all(|&l| l > 0),
                "round {round}: the reader did not run on every unit: {ran_reader:?}"
            );
            // Units 1-3 caught up, unit 0 did not need to, and each has one job: the same
            // lists on all four, because the wait rides in the unit's first list for the
            // op rather than in a list of its own.
            assert!(
                ran_reader.iter().all(|&l| l == ran_reader[0]),
                "round {round}: a catch-up cost a list of its own: {ran_reader:?}"
            );

            let want_slow = host_matmul(&a1, &b1, m1, k1, n1);
            let want = host_matmul(&a2, &want_slow, m2, k2, n1);
            let got = s.download(&out).unwrap();
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "round {round}: the reader saw something other than the slow op's output"
            );
            for t in [x, w, z, slow, out] {
                s.free(t).unwrap();
            }
        }
    });
}

/// A capture's barriers count only the units each op ran on: the slow op on unit 0 adds one
/// arrival, the reader four. Every replay, with fresh operands copied in, gives the host's
/// products: units 1-3 wait for the slow op inside the trace. After each replay an op on units
/// 1-3's side reads its result, and the counter the replay left is the one every unit reached
/// (or the next op's barrier would never complete).
#[test]
fn a_captured_op_s_barrier_counts_the_units_it_ran_on() {
    const UNITS: usize = 4;
    with_tiles(UNITS, |s| {
        // K long enough (256 tiles on one unit) that the slow op is still running when units 1-3,
        // were they not to wait, would read its output.
        let (m1, k1, n1) = (32, 8192, 32);
        let (m2, k2) = (128, 32);
        let x = s.upload(&bits(m1 * k1, 1), m1, k1).unwrap();
        let w = s.upload(&bits(k1 * n1, 2), k1, n1).unwrap();
        let z = s.upload(&bits(m2 * k2, 3), m2, k2).unwrap();
        let q = s.upload(&bits(m1 * m2, 4), m1, m2).unwrap();
        s.begin_trace().unwrap();
        // Unit 0 alone, then all four reading it, then unit 0 alone again reading what all four
        // wrote (one output tile), then all four again reading that: units 1-3 sit out two ops.
        let slow = mm(s, &x, &w);
        let out = mm(s, &z, &slow);
        let again = mm(s, &q, &out);
        let wide = mm(s, &z, &again);
        let trace = s.end_trace().unwrap();
        let ops = s.trace_ops(trace).unwrap().to_vec();
        let ran: Vec<usize> = ops
            .iter()
            .map(|o| o.entries.iter().filter(|r| !r.is_empty()).count())
            .collect();
        assert!(
            ran.contains(&1) && ran.contains(&UNITS),
            "the case needs ops on one unit and on all: {ran:?}"
        );
        let arrivals = s.trace_arrivals(trace).unwrap();
        assert_eq!(
            arrivals as usize,
            ran.iter().filter(|&&u| u > 0).sum::<usize>(),
            "a captured barrier counts its op's units: {ran:?}"
        );
        assert!((arrivals as usize) < ops.len() * UNITS);
        // Unit 0 arrives at all four barriers; units 1-3 at two, and wait before each op after one
        // they sat out (`out`, `wide`): the waits a replay needs, in the streams. Each wait stands
        // for the one barrier its unit sat out, so every stream accounts for all four (the
        // profile mode's window counts ops by them).
        let entries = s.trace_barrier_entries(trace).unwrap();
        assert_eq!(entries[0], Some([4, 0, 4]), "unit 0's stream");
        for (u, e) in entries.iter().enumerate().skip(1) {
            assert_eq!(*e, Some([2, 2, 4]), "unit {u}'s stream");
        }
        for round in 0..3 {
            let (a1, b1, a2) = (
                bits(m1 * k1, 10 + round),
                bits(k1 * n1, 20 + round),
                bits(m2 * k2, 30 + round),
            );
            for (t, v, r, c) in [(&x, &a1, m1, k1), (&w, &b1, k1, n1), (&z, &a2, m2, k2)] {
                let fresh = s.upload(v, r, c).unwrap();
                s.copy_into(&fresh, t).unwrap();
                s.free(fresh).unwrap();
            }
            s.replay(trace).unwrap();
            // Read the replay's result on units 1-3's side, eagerly.
            let after = mm(s, &z, &again);
            let want_slow = host_matmul(&a1, &b1, m1, k1, n1);
            let want_out = host_matmul(&a2, &want_slow, m2, k2, n1);
            let want_again = host_matmul(&bits(m1 * m2, 4), &want_out, m1, m2, n1);
            let want_wide = host_matmul(&a2, &want_again, m2, k2, n1);
            let as_bits = |v: Vec<f32>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                as_bits(s.download(&wide).unwrap()),
                as_bits(want_wide.clone()),
                "round {round}: the replay's last product"
            );
            assert_eq!(
                as_bits(s.download(&after).unwrap()),
                as_bits(want_wide),
                "round {round}: an eager op after the replay"
            );
            s.free(after).unwrap();
        }
        s.release_trace(trace).unwrap();
        for t in [x, w, z, q, slow, out, again, wide] {
            s.free(t).unwrap();
        }
    });
}

/// A capture whose run syncs between its ops (as the profile mode's does after every op) has the
/// same waits in its streams: a sync catches every unit up on the card, not in a replay. Watched
/// failing with the waits taken from the session's catch-up state (no wait at all, a garbage
/// generation under `--profile`).
#[test]
fn a_capture_that_syncs_between_its_ops_keeps_its_waits() {
    const UNITS: usize = 4;
    with_tiles(UNITS, |s| {
        let (m1, k1, n1) = (32, 256, 32);
        let (m2, k2) = (128, 32);
        let x = s.upload(&bits(m1 * k1, 1), m1, k1).unwrap();
        let w = s.upload(&bits(k1 * n1, 2), k1, n1).unwrap();
        let z = s.upload(&bits(m2 * k2, 3), m2, k2).unwrap();
        let q = s.upload(&bits(m1 * m2, 4), m1, m2).unwrap();
        s.begin_trace().unwrap();
        let slow = mm(s, &x, &w);
        s.sync().unwrap();
        let out = mm(s, &z, &slow);
        s.sync().unwrap();
        let again = mm(s, &q, &out);
        s.sync().unwrap();
        let wide = mm(s, &z, &again);
        let trace = s.end_trace().unwrap();
        let entries = s.trace_barrier_entries(trace).unwrap();
        assert_eq!(entries[0], Some([4, 0, 4]), "unit 0's stream");
        for (u, e) in entries.iter().enumerate().skip(1) {
            assert_eq!(*e, Some([2, 2, 4]), "unit {u}'s stream");
        }
        let want_slow = host_matmul(&bits(m1 * k1, 1), &bits(k1 * n1, 2), m1, k1, n1);
        let want_out = host_matmul(&bits(m2 * k2, 3), &want_slow, m2, k2, n1);
        let want_again = host_matmul(&bits(m1 * m2, 4), &want_out, m1, m2, n1);
        let want_wide = host_matmul(&bits(m2 * k2, 3), &want_again, m2, k2, n1);
        for _ in 0..2 {
            s.replay(trace).unwrap();
            let got = s.download(&wide).unwrap();
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want_wide.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        s.release_trace(trace).unwrap();
        for t in [x, w, z, q, slow, out, again, wide] {
            s.free(t).unwrap();
        }
    });
}

/// The barrier count: an op on one unit of four adds one arrival, not four, and a trace's
/// barriers still involve every unit (its replay is held to the fresh result).
#[test]
fn a_trace_replays_with_every_unit_in_its_barriers_after_partial_ops() {
    const UNITS: usize = 4;
    with_tiles(UNITS, |s| {
        let (m1, k1, n1) = (32, 256, 32);
        let (m2, k2) = (128, 32);
        let a1 = bits(m1 * k1, 1);
        let b1 = bits(k1 * n1, 2);
        let a2 = bits(m2 * k2, 3);
        let x = s.upload(&a1, m1, k1).unwrap();
        let w = s.upload(&b1, k1, n1).unwrap();
        let z = s.upload(&a2, m2, k2).unwrap();
        // A partial op before the capture leaves units 1-3 behind the counter; the capture's
        // own sync catches them up.
        let warm = mm(s, &x, &w);
        s.begin_trace().unwrap();
        let slow = mm(s, &x, &w);
        let out = mm(s, &z, &slow);
        let trace = s.end_trace().unwrap();
        // A partial op between replays leaves them behind again; the replay catches up.
        let mut extra = Vec::new();
        for _ in 0..3 {
            extra.push(mm(s, &x, &w));
            s.replay(trace).unwrap();
        }
        let want = host_matmul(&a2, &host_matmul(&a1, &b1, m1, k1, n1), m2, k2, n1);
        let got = s.download(&out).unwrap();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        s.release_trace(trace).unwrap();
        for t in [x, w, z, warm, slow, out].into_iter().chain(extra) {
            s.free(t).unwrap();
        }
    });
}
