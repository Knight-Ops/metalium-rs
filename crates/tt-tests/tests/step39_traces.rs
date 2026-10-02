//! Phase 10 gate (X4d): a captured trace replays as the ops it captured.
//!
//! A layer's forward pass -- `[m, 784] @ [784, 128]`, a bias row, `relu`,
//! `@ [128, 10]` -- is captured once, then replayed with new inputs written
//! between replays: each replay's output is the same ops run fresh on that
//! input, bit for bit, and a replay sends the card a `CALL` per unit rather
//! than the ops' lists. And each of the footguns `crate::trace` closes is
//! provoked, to see it refused rather than corrupt anything.

use tt_isa::dm::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise, TensorError};
use tt_kernels::trace::TraceError;
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_tiles(choice: TileChoice, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(dev, tt_firmware_images::ROLES, choice, |_, _| Ok(None)).unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_tiles(choice: TileChoice, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            choice,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn gate_tile() -> TileChoice {
    TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1)
}

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// The layer's weights, resident.
struct Layer {
    w1: DramTensor,
    b1: DramTensor,
    w2: DramTensor,
}

impl Layer {
    fn upload<T: tt_device::Transport>(s: &mut Session<T>) -> Self {
        Layer {
            w1: s.upload(&values(2, 784 * 128), 784, 128).unwrap(),
            b1: s.upload(&values(3, 128), 1, 128).unwrap(),
            w2: s.upload(&values(4, 128 * 10), 128, 10).unwrap(),
        }
    }

    /// The forward pass on `x`; every intermediate freed.
    fn forward<T: tt_device::Transport>(&self, s: &mut Session<T>, x: &DramTensor) -> DramTensor {
        let mm = |s: &mut Session<T>, a, b| {
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
        let ew = |s: &mut Session<T>, kind, a, b| {
            s.eltwise(Eltwise { kind, scalar: 0.0 }, a, b).unwrap()
        };
        let h = mm(s, x, &self.w1);
        let hb = ew(s, kind::ADD_ROW, &h, Some(&self.b1));
        let r = ew(s, kind::RELU, &hb, None);
        let y = mm(s, &r, &self.w2);
        for t in [h, hb, r] {
            s.free(t).unwrap();
        }
        y
    }
}

fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}, element {i}: {g:e} against {w:e}"
        );
    }
}

fn err<T>(r: Result<T, TensorError>) -> TraceError {
    match r {
        Ok(_) => panic!("not refused"),
        Err(e) => trace_error(e),
    }
}

fn trace_error(e: TensorError) -> TraceError {
    match e {
        TensorError::Trace(t) => t,
        other => panic!("not a trace error: {other}"),
    }
}

/// Capture, then replay over new inputs, each replay checked against the ops
/// run fresh; and what a replay sends the card, against what the capture did.
fn replays_are_the_fresh_ops(choice: TileChoice, m: usize) {
    with_tiles(choice, |s| {
        let layer = Layer::upload(s);
        let x = s.upload(&values(1, m * 784), m, 784).unwrap();
        // Warm: the programs resident, so the capture's traffic is the ops'.
        let warm = layer.forward(s, &x);
        s.free(warm).unwrap();

        let before = s.device().traffic();
        s.begin_trace().unwrap();
        let y = layer.forward(s, &x);
        let id = s.end_trace().unwrap();
        let captured = s.device().traffic().bytes_written - before.bytes_written;
        assert!(
            !s.trace_ops(id).unwrap().is_empty(),
            "the capture recorded its ops"
        );
        let first = s.download(&y).unwrap();
        let fresh = layer.forward(s, &x);
        assert_bits(
            "the capture's own run",
            &first,
            &s.download(&fresh).unwrap(),
        );
        s.free(fresh).unwrap();

        for round in 0..3u64 {
            let input = values(10 + round, m * 784);
            s.write(&x, &input).unwrap();
            let before = s.device().traffic();
            s.replay(id).unwrap();
            s.sync().unwrap();
            let sent = s.device().traffic().bytes_written - before.bytes_written;
            let replayed = s.download(&y).unwrap();
            let fresh = layer.forward(s, &x);
            assert_bits(
                &format!("replay {round}"),
                &replayed,
                &s.download(&fresh).unwrap(),
            );
            s.free(fresh).unwrap();
            // A `CALL` per unit and the queue's bookkeeping, not the lists.
            assert!(
                sent * 20 < captured,
                "replay {round} wrote {sent} bytes; the capture {captured}"
            );
            eprintln!("MEASURE trace replay {round}: {sent} bytes written, capture {captured}");
        }
        s.release_trace(id).unwrap();
        for t in [x, y, layer.w1, layer.b1, layer.w2] {
            s.free(t).unwrap();
        }
    });
}

#[test]
fn a_replay_is_the_ops_run_fresh_on_one_tile() {
    replays_are_the_fresh_ops(gate_tile(), 64);
}

/// With barriers between ops, each replay's relative to the session's.
#[test]
fn a_replay_is_the_ops_run_fresh_on_two_tiles() {
    replays_are_the_fresh_ops(TileChoice::Count(2), 96);
}

/// Freeing what a trace reads is deferred to its release: the slots are not
/// handed to the next allocation, so a replay still reads the weights.
#[test]
fn a_trace_holds_what_it_reads_until_released() {
    with_tiles(gate_tile(), |s| {
        let layer = Layer::upload(s);
        let x = s.upload(&values(1, 32 * 784), 32, 784).unwrap();
        s.begin_trace().unwrap();
        let y = layer.forward(s, &x);
        let id = s.end_trace().unwrap();
        let want = s.download(&y).unwrap();

        let free_before = s.dram_free_bytes();
        let Layer { w1, b1, w2 } = layer;
        let w1_tiles = w1.placement.tiles();
        s.free(w1).unwrap();
        assert_eq!(s.dram_free_bytes(), free_before, "the free was deferred");
        // An allocation the size of `w1`, filled with garbage: it must land
        // elsewhere.
        let garbage = s.upload(&vec![1e30; 784 * 128], 784, 128).unwrap();
        assert_eq!(garbage.placement.tiles(), w1_tiles);
        s.replay(id).unwrap();
        assert_bits("replayed after the free", &s.download(&y).unwrap(), &want);

        // Released: the deferred free happens.
        let free_before = s.dram_free_bytes();
        s.release_trace(id).unwrap();
        assert!(
            s.dram_free_bytes() > free_before,
            "the release gave the slots back"
        );
        assert_eq!(
            trace_error(s.replay(id).unwrap_err()),
            TraceError::Unknown(0),
            "a released trace"
        );
        // Allocated after the capture: freed at once.
        let free_before = s.dram_free_bytes();
        s.free(garbage).unwrap();
        assert!(s.dram_free_bytes() > free_before);
        for t in [x, y, b1, w2] {
            s.free(t).unwrap();
        }
    });
}

/// Each refusal, provoked.
#[test]
fn what_a_replay_could_not_repeat_is_refused() {
    with_tiles(gate_tile(), |s| {
        let a = s.upload(&values(1, 32 * 32), 32, 32).unwrap();
        let relu = |s: &mut Session<_>, a| {
            s.eltwise(
                Eltwise {
                    kind: kind::RELU,
                    scalar: 0.0,
                },
                a,
                None,
            )
            .unwrap()
        };

        assert_eq!(err(s.end_trace()), TraceError::NotCapturing);
        s.begin_trace().unwrap();
        assert_eq!(err(s.begin_trace()), TraceError::Capturing);
        assert_eq!(err(s.end_trace()), TraceError::Empty);

        s.begin_trace().unwrap();
        let y = relu(s, &a);
        assert_eq!(err(s.download(&y)), TraceError::HostTransfer("download"));
        assert_eq!(
            err(s.write(&a, &[0.0; 1024])),
            TraceError::HostTransfer("write")
        );
        assert_eq!(err(s.set_batching(false)), TraceError::Capturing);
        // A host-run path (here the tile reset), which the capture would miss.
        assert!(s.prepare().is_err(), "a host-run kernel during a capture");
        let id = s.end_trace().unwrap();
        s.begin_trace().unwrap();
        assert_eq!(err(s.replay(id)), TraceError::Capturing);
        assert_eq!(err(s.end_trace()), TraceError::Empty);

        // The tiles reset: everything the trace named is gone.
        s.prepare().unwrap();
        assert_eq!(err(s.replay(id)), TraceError::Stale);
        s.release_trace(id).unwrap();

        s.set_batching(false).unwrap();
        assert_eq!(err(s.begin_trace()), TraceError::NotBatching);
        s.set_batching(true).unwrap();
        for t in [a, y] {
            s.free(t).unwrap();
        }
    });
}
