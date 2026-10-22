//! decode-to-bandwidth.md Phase 5, the weight stream under the small ops: in a capture, a one-row
//! product with a packed `A` and its `K` whole puts its first columns on units that ran nothing
//! since the last such product (`Session::set_weight_prefetch`). Each of those units reads its
//! columns' `B` (`tensor::PREFETCH`) as soon as that product is done, under the ops between the
//! two, and the product then reads only `A` there.
//!
//! On eight units with every op on at most four (`Spread::Fixed`), units 4-7 sit out the
//! element-wise ops between three chained products. The claims:
//! - **Same bits.** The capture's own run, and replays over new inputs, are the ops run fresh
//!   (no capture, so no prefetch), bit for bit; and the same as a capture without prefetch.
//! - **The prefetch is taken.** The second and third products run on units 4-7 too (their
//!   records' entries there), and with prefetch off they do not.
//! - **Only idle units.** With the element-wise ops between the products on all eight units,
//!   units 4-7 are not idle since the product before, and no product takes them; the bits are
//!   the same. (A prefetch is safe on any unit: its list is appended to the unit's stream when
//!   the product is captured, after everything the unit did before and with nothing but the
//!   product's own wait after it. Taking only idle units is a policy: a busy unit's reads would
//!   start no earlier than its last op ends, and its columns could finish after the others'.)
//!
//! - **The prefetch keeps its units** (`Session::set_prefetch_keeps_units`, on by default): once a
//!   product has streamed, ops that would spread over all eight units in the capture run on the
//!   four it ran on, so units 4-7 stay idle and the next products prefetch there; the bits are the
//!   same. Claim three turns it off. (A decode step's attention at long context took the units
//!   `o`'s weights stream on.)
//!
//! - **The counts are the session's** (`Session::set_prefetch_units`, `set_thin_stream_count`,
//!   `TT_PREFETCH_UNITS` and `TT_THIN_STREAM_UNITS` at open): a prefetch takes at most its count
//!   of the idle units, a weight stream runs on at most its count, and the bits are the same.
//!
//! - **A streamed product prefetches too** (`Session::set_stream_prefetch`): under a K-block limit
//!   every product streams its `K` in chunks (a decode step's `down`), and an idle unit takes a
//!   column, reading its first two chunks' `B` (the ring's two halves) ahead; the bits are the
//!   same. Watched failing with three chunks read ahead (the third overwrites the first's half
//!   before its kernel reads it).
//!
//! Watched failing: the prefetched columns not left out of the other units' deal (the rest
//! dealt from column 0), which leaves the last columns uncomputed.

use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, Spread, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise};
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

fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
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

struct Chain {
    w: [DramTensor; 3],
}

impl Chain {
    fn upload<T: tt_device::Transport>(s: &mut Session<T>) -> Self {
        let w = [(1, K), (2, K), (3, 256)]
            .map(|(seed, n)| s.upload(&values(seed, K * n), K, n).unwrap());
        Chain { w }
    }

    /// `x @ w0`, a scale, `@ w1`, a relu, `@ w2`: the element-wise ops on `between` units.
    fn forward<T: tt_device::Transport>(
        &self,
        s: &mut Session<T>,
        x: &DramTensor,
        between: Spread,
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
            s.set_spread(between);
            let out = s
                .eltwise(
                    Eltwise {
                        kind,
                        scalar,
                        scalar2: 0.0,
                    },
                    a,
                    None,
                )
                .unwrap();
            s.set_spread(Spread::Fixed(4));
            out
        };
        let h = mm(s, x, &self.w[0]);
        let hs = ew(s, kind::MUL_SCALAR, 0.5, &h);
        let g = mm(s, &hs, &self.w[1]);
        let gr = ew(s, kind::RELU, 0.0, &g);
        let y = mm(s, &gr, &self.w[2]);
        for t in [h, hs, g, gr] {
            s.free(t).unwrap();
        }
        y
    }
}

/// The units 4-7 entries of each product a capture recorded (`float_matmul` ops), in order.
fn on_idle_units<T: tt_device::Transport>(
    s: &Session<T>,
    id: tt_kernels::trace::TraceId,
) -> Vec<bool> {
    s.trace_ops(id)
        .unwrap()
        .iter()
        .filter(|op| op.what.contains("matmul") || op.what == tt_kernels::tensor::PREFETCH)
        .map(|op| op.entries[4..8].iter().any(|r| !r.is_empty()))
        .collect()
}

/// Claims one and two.
#[test]
fn prefetched_products_replay_as_the_ops_run_fresh() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        let chain = Chain::upload(s);
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let warm = chain.forward(s, &x, Spread::Fixed(4));
        s.free(warm).unwrap();
        let mut runs = Vec::new();
        for prefetch in [false, true] {
            // The capture's own run on the first input (the last pass's replays wrote others).
            s.write(&x, &values(10, K)).unwrap();
            s.set_weight_prefetch(prefetch);
            s.begin_trace().unwrap();
            let y = chain.forward(s, &x, Spread::Fixed(4));
            let id = s.end_trace().unwrap();
            s.set_weight_prefetch(true);
            let used = on_idle_units(s, id);
            let first = s.download(&y).unwrap();
            let mut replays = vec![first];
            for round in 0..2u64 {
                s.write(&x, &values(20 + round, K)).unwrap();
                s.replay(id).unwrap();
                s.sync().unwrap();
                replays.push(s.download(&y).unwrap());
            }
            s.release_trace(id).unwrap();
            s.free(y).unwrap();
            runs.push((used, replays));
        }
        // The ops run fresh on each input, no capture and so no prefetch.
        let mut fresh = Vec::new();
        for seed in [10, 20, 21] {
            s.write(&x, &values(seed, K)).unwrap();
            let y = chain.forward(s, &x, Spread::Fixed(4));
            fresh.push(s.download(&y).unwrap());
            s.free(y).unwrap();
        }
        let (off, on) = (&runs[0], &runs[1]);
        assert_eq!(
            off.0,
            [false, false, false],
            "the control: no product on units 4-7"
        );
        assert_eq!(
            on.0,
            [false, true, true],
            "the second and third products prefetch on units 4-7"
        );
        for (i, want) in fresh.iter().enumerate() {
            assert_bits(&format!("unprefetched, run {i}"), &off.1[i], want);
            assert_bits(&format!("prefetched, run {i}"), &on.1[i], want);
        }
        for t in chain.w.into_iter().chain([x]) {
            s.free(t).unwrap();
        }
    });
}

/// Claim three.
#[test]
fn a_unit_that_ran_an_op_since_the_last_product_does_not_prefetch() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        let chain = Chain::upload(s);
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let warm = chain.forward(s, &x, Spread::Fixed(4));
        let want = s.download(&warm).unwrap();
        s.free(warm).unwrap();
        // The ops may take every unit (claim four keeps them off the prefetch's).
        s.set_prefetch_keeps_units(false);
        s.begin_trace().unwrap();
        let y = chain.forward(s, &x, Spread::All);
        s.set_prefetch_keeps_units(true);
        let id = s.end_trace().unwrap();
        assert_bits("the capture's own run", &s.download(&y).unwrap(), &want);
        s.replay(id).unwrap();
        s.sync().unwrap();
        assert_bits("the replay", &s.download(&y).unwrap(), &want);
        assert_eq!(
            on_idle_units(s, id),
            [false, false, false],
            "units 4-7 ran the ops between the products"
        );
        s.release_trace(id).unwrap();
        for t in chain.w.into_iter().chain([x, y]) {
            s.free(t).unwrap();
        }
    });
}

/// Claim four: with the prefetch keeping its units, element-wise ops asked to spread over all
/// eight units in the capture run on the product's four, and the next products prefetch on 4-7.
/// Watched failing with the cap left out of `Session::spread_for`.
#[test]
fn ops_after_a_streaming_product_keep_off_the_prefetch_s_units() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        let chain = Chain::upload(s);
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let warm = chain.forward(s, &x, Spread::Fixed(4));
        let want = s.download(&warm).unwrap();
        s.free(warm).unwrap();
        s.begin_trace().unwrap();
        let y = chain.forward(s, &x, Spread::All);
        let id = s.end_trace().unwrap();
        assert_bits("the capture's own run", &s.download(&y).unwrap(), &want);
        s.replay(id).unwrap();
        s.sync().unwrap();
        assert_bits("the replay", &s.download(&y).unwrap(), &want);
        let ops = s.trace_ops(id).unwrap();
        for op in ops.iter().filter(|op| op.what.contains("sfpu")) {
            assert!(
                op.entries[4..8].iter().all(|r| r.is_empty()),
                "{}: on units 4-7",
                op.what
            );
        }
        assert_eq!(
            on_idle_units(s, id),
            [false, true, true],
            "the second and third products prefetch on units 4-7"
        );
        s.release_trace(id).unwrap();
        for t in chain.w.into_iter().chain([x, y]) {
            s.free(t).unwrap();
        }
    });
}

/// Which of units 0-7 each product a capture recorded has entries on (`float_matmul` ops and
/// their prefetch lists), in order.
fn units_of_products<T: tt_device::Transport>(
    s: &Session<T>,
    id: tt_kernels::trace::TraceId,
) -> Vec<Vec<bool>> {
    s.trace_ops(id)
        .unwrap()
        .iter()
        .filter(|op| op.what.contains("matmul") || op.what == tt_kernels::tensor::PREFETCH)
        .map(|op| op.entries[..8].iter().map(|r| !r.is_empty()).collect())
        .collect()
}

/// Claim five: with the prefetch held to two units, the second and third products take units
/// 4-5 and not 6-7; with a weight stream held to two units and no prefetch, every product runs
/// on units 0-1. Each capture's run and replay are the ops run fresh, bit for bit.
#[test]
fn the_prefetch_and_the_stream_take_the_session_s_unit_counts() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        let chain = Chain::upload(s);
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let warm = chain.forward(s, &x, Spread::Fixed(4));
        let want = s.download(&warm).unwrap();
        s.free(warm).unwrap();
        let mut seen = Vec::new();
        for (prefetch, stream) in [(2, None), (0, Some(2))] {
            s.set_prefetch_units(prefetch);
            if let Some(n) = stream {
                s.set_thin_stream_count(n);
            }
            s.begin_trace().unwrap();
            let y = chain.forward(s, &x, Spread::Fixed(4));
            let id = s.end_trace().unwrap();
            s.set_prefetch_units(tt_kernels::session::PREFETCH_UNITS);
            s.set_thin_stream_count(tt_kernels::session::THIN_STREAM_UNITS);
            assert_bits("the capture's own run", &s.download(&y).unwrap(), &want);
            s.replay(id).unwrap();
            s.sync().unwrap();
            assert_bits("the replay", &s.download(&y).unwrap(), &want);
            seen.push(units_of_products(s, id));
            s.release_trace(id).unwrap();
            s.free(y).unwrap();
        }
        let on = |units: &[usize]| -> Vec<bool> { (0..8).map(|u| units.contains(&u)).collect() };
        let idle: Vec<_> = seen[0].iter().map(|p| p[4..].to_vec()).collect();
        assert_eq!(
            idle,
            [
                [false; 4],
                [true, true, false, false],
                [true, true, false, false]
            ],
            "a prefetch of two units takes units 4-5 only"
        );
        let two = on(&[0, 1]);
        assert_eq!(
            seen[1],
            [two.clone(), two.clone(), two],
            "a weight stream of two units runs on units 0-1"
        );
        for t in chain.w.into_iter().chain([x]) {
            s.free(t).unwrap();
        }
    });
}

/// Claim six: under a K-block limit of 8 tiles and unpipelined (a pipelined product of many
/// columns a unit blocks its `K` instead), every product streams its `K` (1,024) in four chunks; with stream prefetch on, the second and third take units 4-7, reading their first two
/// chunks ahead, and every run is the ops run fresh, bit for bit.
#[test]
fn streamed_products_prefetch_their_first_chunks_and_replay_as_the_ops_run_fresh() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        s.set_matmul_k_block_limit(std::num::NonZeroUsize::new(8));
        s.set_pipeline(false);
        let chain = Chain::upload(s);
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let warm = chain.forward(s, &x, Spread::Fixed(4));
        s.free(warm).unwrap();
        let mut runs = Vec::new();
        for on in [false, true] {
            s.write(&x, &values(10, K)).unwrap();
            s.set_stream_prefetch(on);
            s.begin_trace().unwrap();
            let y = chain.forward(s, &x, Spread::Fixed(4));
            let id = s.end_trace().unwrap();
            s.set_stream_prefetch(true);
            let ops = s.trace_ops(id).unwrap();
            assert_eq!(
                ops.iter()
                    .filter(|op| op.what == "streamed matmul gather")
                    .count(),
                3,
                "every product streams its K"
            );
            let used = on_idle_units(s, id);
            let mut replays = vec![s.download(&y).unwrap()];
            for round in 0..2u64 {
                s.write(&x, &values(20 + round, K)).unwrap();
                s.replay(id).unwrap();
                s.sync().unwrap();
                replays.push(s.download(&y).unwrap());
            }
            s.release_trace(id).unwrap();
            s.free(y).unwrap();
            runs.push((used, replays));
        }
        let mut fresh = Vec::new();
        for seed in [10, 20, 21] {
            s.write(&x, &values(seed, K)).unwrap();
            let y = chain.forward(s, &x, Spread::Fixed(4));
            fresh.push(s.download(&y).unwrap());
            s.free(y).unwrap();
        }
        s.set_matmul_k_block_limit(None);
        s.set_pipeline(true);
        let (off, on) = (&runs[0], &runs[1]);
        assert_eq!(
            off.0,
            [false, false, false],
            "the control: no product on units 4-7"
        );
        assert_eq!(
            on.0,
            [false, true, true],
            "the second and third products prefetch on units 4-7"
        );
        for (i, want) in fresh.iter().enumerate() {
            assert_bits(&format!("unprefetched, run {i}"), &off.1[i], want);
            assert_bits(&format!("prefetched, run {i}"), &on.1[i], want);
        }
        for t in chain.w.into_iter().chain([x]) {
            s.free(t).unwrap();
        }
    });
}
