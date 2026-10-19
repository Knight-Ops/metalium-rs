//! Released barriers (`docs/plans/decode-to-bandwidth.md` Phase 3, `Session::set_barrier_release`,
//! `dm::op::BARRIER` word 6): unit 0 waits on the barrier counter in its own L1, then multicasts
//! each target it arrives at to every other unit's `dm::BARRIER_RELEASE`, and each unit waits on
//! its own word, with no NoC traffic while it waits -- tt-metal's coordinator-broadcast barrier,
//! where every unit used to poll unit 0's counter over the NoC.
//!
//! On eight units, the case that needs a barrier (`step176`'s): a slow op on unit 0 alone, then
//! an op on every unit that reads its output. The claims:
//! - **Order.** The reader gives the host's products, exactly, every round, eager and in a
//!   replayed trace.
//! - **Every unit is released.** After each sync, every unit's release word is the count of
//!   arrivals: the multicast reached all seven, and unit 0 arrived at every barrier (it releases
//!   only what it arrives at).
//! - **The words are the release's.** With release off, the same rounds leave the words where
//!   they were and give the same products.
//! - **They are what orders the units** (negative control, ttsim): with unit 0 releasing nothing
//!   and every word far past any target, the reader reads the slow op's output before it is
//!   written, in some round.

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::DramTensor;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const UNITS: usize = 8;

#[cfg(not(feature = "silicon"))]
fn with_tiles(test: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut session = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(UNITS),
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
    fork_scope(|| {
        let mut session = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(UNITS),
        )
        .unwrap();
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        test(&mut session);
    })
    .unwrap();
}

/// 0 or 1 by a hash: every product and sum below is an integer under 2^15, exact.
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

/// The slow op (one output tile, `K` 1024: one job, on unit 0) then the reader (eight output
/// tiles, a job on each unit, each reading the slow op's whole output as its `B`).
const SLOW: (usize, usize, usize) = (32, 1024, 32);
const READER: (usize, usize) = (256, 32);
/// The negative control's slow op's `K`: slow enough on ttsim for the readers to overtake it.
const SLOW_K: usize = 8192;

/// One round on fresh operands: whether the reader gave the host's products.
fn round<T: tt_device::Transport>(s: &mut Session<T>, seed: usize) -> bool {
    round_k(s, seed, SLOW.1)
}

/// [`round`] with the slow op's `K` at `k1`.
fn round_k<T: tt_device::Transport>(s: &mut Session<T>, seed: usize, k1: usize) -> bool {
    let ((m1, _, n1), (m2, k2)) = (SLOW, READER);
    let (a1, b1, a2) = (
        bits(m1 * k1, 10 + seed),
        bits(k1 * n1, 20 + seed),
        bits(m2 * k2, 30 + seed),
    );
    let x = s.upload(&a1, m1, k1).unwrap();
    let w = s.upload(&b1, k1, n1).unwrap();
    let z = s.upload(&a2, m2, k2).unwrap();
    s.sync().unwrap();
    let slow = mm(s, &x, &w);
    let out = mm(s, &z, &slow);
    let want = host_matmul(&a2, &host_matmul(&a1, &b1, m1, k1, n1), m2, k2, n1);
    let got = s.download(&out).unwrap();
    for t in [x, w, z, slow, out] {
        s.free(t).unwrap();
    }
    got.iter()
        .map(|v| v.to_bits())
        .eq(want.iter().map(|v| v.to_bits()))
}

/// Claims one and two, eager.
#[test]
fn released_barriers_order_the_units_and_reach_every_one() {
    with_tiles(|s| {
        for seed in 0..4 {
            // In odd rounds, the negative control's slow op.
            let k1 = if seed % 2 == 1 { SLOW_K } else { SLOW.1 };
            assert!(
                round_k(s, seed, k1),
                "round {seed}: the reader saw another output"
            );
            let (words, count) = s.barrier_release_words().unwrap();
            assert!(count > 0, "barriers were counted");
            assert_eq!(
                words[1..],
                vec![count; UNITS - 1],
                "round {seed}: every unit released"
            );
        }
    });
}

/// Claim three: polled barriers order the units too, and leave the release words alone.
#[test]
fn polled_barriers_leave_the_release_words_alone() {
    with_tiles(|s| {
        s.set_barrier_release(false).unwrap();
        // The words are zeroed with the counter when unit 0's mover starts (L1 keeps an
        // earlier session's until then): one round first.
        assert!(round(s, 9), "the first round");
        let (before, _) = s.barrier_release_words().unwrap();
        for seed in 0..3 {
            assert!(
                round(s, seed),
                "round {seed}: the reader saw another output"
            );
        }
        let (after, count) = s.barrier_release_words().unwrap();
        assert!(count > 0);
        assert_eq!(after, before, "polling released nothing");
        // Turned on again, every word starts at the count every unit has passed.
        s.set_barrier_release(true).unwrap();
        let (words, count) = s.barrier_release_words().unwrap();
        assert_eq!(words, vec![count; UNITS]);
        assert!(round(s, 7), "after turning release back on");
    });
}

/// Claim one in a trace: the pair captured, replayed over fresh operands written in.
#[test]
fn a_replayed_trace_s_barriers_are_released() {
    with_tiles(|s| {
        let ((m1, k1, n1), (m2, k2)) = (SLOW, READER);
        let x = s.upload(&bits(m1 * k1, 10), m1, k1).unwrap();
        let w = s.upload(&bits(k1 * n1, 20), k1, n1).unwrap();
        let z = s.upload(&bits(m2 * k2, 30), m2, k2).unwrap();
        s.sync().unwrap();
        s.begin_trace().unwrap();
        let slow = mm(s, &x, &w);
        let out = mm(s, &z, &slow);
        let id = s.end_trace().unwrap();
        for seed in 1..4 {
            let (a1, b1, a2) = (
                bits(m1 * k1, 10 + seed),
                bits(k1 * n1, 20 + seed),
                bits(m2 * k2, 30 + seed),
            );
            s.write(&x, &a1).unwrap();
            s.write(&w, &b1).unwrap();
            s.write(&z, &a2).unwrap();
            s.replay(id).unwrap();
            let want = host_matmul(&a2, &host_matmul(&a1, &b1, m1, k1, n1), m2, k2, n1);
            let got = s.download(&out).unwrap();
            assert!(
                got.iter()
                    .map(|v| v.to_bits())
                    .eq(want.iter().map(|v| v.to_bits())),
                "replay {seed}: the reader saw another output"
            );
            let (words, count) = s.barrier_release_words().unwrap();
            assert_eq!(
                words[1..],
                vec![count; UNITS - 1],
                "replay {seed}: every unit released"
            );
        }
        s.release_trace(id).unwrap();
        for t in [x, w, z, slow, out] {
            s.free(t).unwrap();
        }
    });
}

/// Claim four, the negative control. ttsim only: on card 0 the readers never overtook the slow
/// op even at `K` 8192 and 32768 (runs `1792383789`, `1792383833`): the host queues the reader
/// on all eight units more slowly than unit 0 finishes, so the order holds by timing there.
#[test]
#[cfg_attr(feature = "silicon", ignore)]
fn without_the_release_the_reader_runs_ahead() {
    with_tiles(|s| {
        assert!(round(s, 0), "the case holds before the control");
        s.break_barrier_release().unwrap();
        // A slower slow op for the readers to overtake.
        let wrong = (1..9).filter(|&seed| !round_k(s, seed, SLOW_K)).count();
        assert!(
            wrong > 0,
            "with no barrier, every reader still saw the slow op's output"
        );
    });
}
