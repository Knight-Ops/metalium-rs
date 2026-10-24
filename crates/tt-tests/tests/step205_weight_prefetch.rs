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
//! - **A streamed column is handed between two units** (`Session::set_stream_handoff`): with the
//!   stream prefetch held to units 4-5, units 6-7 have no job in a product, and each takes the
//!   rest of a column one of units 0-3 would read whole -- reading it ahead in the product, then
//!   reloading the partial the column's unit wrote and computing the rest in an op of its own
//!   after the product's barrier. Handed at chunk 1 and at half (2 of 4); the bits are the same.
//!   Watched failing with the partial reloaded into the other half's slot, and with its reload
//!   dropped.
//!
//! - **A unit's own anchor** (`Session::set_prefetch_anchor`; decode-to-bandwidth.md Phase 8,
//!   "The GDDR idle before `o`"): with the prefetch held to units 4-5, units 6-7 sit out the
//!   second product, and under `OwnOldest` the third takes them, each list waiting for the first
//!   product's barrier, not the second's; `Own` keeps index order (4-5), `Last` today's waits.
//!   The bits are the same. Two negative controls, each watched failing through a probe:
//!   - a `B` the capture makes between the first and second products (an element-wise result of
//!     an input rewritten before each replay) is read ahead by no unit; with the guard probed
//!     off, `OwnOldest`'s replays read it before it is made, and `Last`'s do not;
//!   - a busy unit (2-3) that sat out the second product reads the third's `B` from the first
//!     product on, but not across an op between (a 64-row product) that planned into its ring;
//!     with `l1_top` probed away, the list moves across it and the bits differ.
//!
//! Watched failing: the prefetched columns not left out of the other units' deal (the rest
//! dealt from column 0), which leaves the last columns uncomputed.

use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{PrefetchAnchor, Session, Spread, TileChoice};
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

/// Each idle unit's second job (`Session::set_prefetch_second_job`, in the trace arena,
/// `docs/plans/decode-to-bandwidth.md` Phase 8, R2): a column more a prefetching unit, its `B` in a
/// ring below the first job's and read ahead with it. A product of 20 columns at `K` = 2048 after
/// another (whose barrier anchors the prefetch), `B` against its ring's end: it replays as it runs fresh,
/// bit for bit, and its prefetching units run more of it (the second job's entries). Watched
/// failing with the second ring placed on the first: the bits differ.
#[test]
fn second_jobs_in_the_trace_arena_replay_as_the_ops_run_fresh() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        s.set_trace_arena(true);
        let w0 = s.upload(&values(1, K * 2048), K, 2048).unwrap();
        let w1 = s.upload(&values(2, 2048 * 640), 2048, 640).unwrap();
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let forward = |s: &mut Session<_>, x: &DramTensor| {
            let h = mm(s, x, &w0);
            let y = mm(s, &h, &w1);
            s.free(h).unwrap();
            y
        };
        let warm = forward(s, &x);
        s.free(warm).unwrap();
        let mut fresh = Vec::new();
        for seed in [10, 20, 21] {
            s.write(&x, &values(seed, K)).unwrap();
            let y = forward(s, &x);
            fresh.push(s.download(&y).unwrap());
            s.free(y).unwrap();
        }
        let mut entries = Vec::new();
        for second in [false, true] {
            s.set_prefetch_second_job(second);
            s.write(&x, &values(10, K)).unwrap();
            s.begin_trace().unwrap();
            let y = forward(s, &x);
            let id = s.end_trace().unwrap();
            // The prefetching units' entries for each product, in order.
            let ops = s.trace_ops(id).unwrap();
            entries.push(
                ops.iter()
                    .filter(|op| op.what.contains("matmul"))
                    .map(|op| op.entries[4..8].iter().map(|r| r.len()).sum::<usize>())
                    .collect::<Vec<_>>(),
            );
            let mut got = vec![s.download(&y).unwrap()];
            for seed in [20u64, 21] {
                s.write(&x, &values(seed, K)).unwrap();
                s.replay(id).unwrap();
                s.sync().unwrap();
                got.push(s.download(&y).unwrap());
            }
            for (i, (g, w)) in got.iter().zip(&fresh).enumerate() {
                assert_bits(&format!("second jobs {second}, run {i}"), g, w);
            }
            s.release_trace(id).unwrap();
            s.free(y).unwrap();
        }
        s.set_prefetch_second_job(false);
        s.set_trace_arena(false);
        eprintln!("prefetching units' entries a product, without and with: {entries:?}");
        assert!(
            entries[1][1] > entries[0][1],
            "the prefetching units take second jobs: {entries:?}"
        );
        for t in [w0, w1, x] {
            s.free(t).unwrap();
        }
    });
}

/// A balanced deal (`Session::set_balanced_deal`, `docs/plans/decode-to-bandwidth.md` Phase 8, "a
/// product's longest unit"): a product of 40 columns at `K` = 2048 after another, whose own units
/// 0-3 would run five rounds of two columns while units 4-7 take one job read ahead. With it on,
/// the 24 columns after the first round are dealt over all eight units, so units 4-7 run three
/// more each in the same ring after their job read ahead. It replays as it runs fresh, bit for
/// bit, and each unit's columns read in the span are as dealt. Watched failing with each unit's later columns
/// shifted one back (a column computed twice, the last never): the bits differ.
#[test]
fn balanced_deals_replay_as_the_ops_run_fresh() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        let w0 = s.upload(&values(1, K * 2048), K, 2048).unwrap();
        let w1 = s.upload(&values(2, 2048 * 1280), 2048, 1280).unwrap();
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let forward = |s: &mut Session<_>, x: &DramTensor| {
            let h = mm(s, x, &w0);
            let y = mm(s, &h, &w1);
            s.free(h).unwrap();
            y
        };
        let warm = forward(s, &x);
        s.free(warm).unwrap();
        let mut fresh = Vec::new();
        for seed in [10, 20, 21] {
            s.write(&x, &values(seed, K)).unwrap();
            let y = forward(s, &x);
            fresh.push(s.download(&y).unwrap());
            s.free(y).unwrap();
        }
        s.set_count_trace_bytes(true);
        // Per unit, the second product's `B` read in its span (`OpRecord::unit_bytes`).
        let mut in_span = Vec::new();
        for balanced in [false, true] {
            s.set_balanced_deal(balanced);
            s.write(&x, &values(10, K)).unwrap();
            s.begin_trace().unwrap();
            let y = forward(s, &x);
            let id = s.end_trace().unwrap();
            let ops = s.trace_ops(id).unwrap();
            let second = ops
                .iter()
                .filter(|op| op.what.contains("matmul"))
                .nth(1)
                .expect("two products");
            let mut b = [0u64; 8];
            for (u, bytes) in &second.unit_bytes {
                b[*u as usize] = bytes.b - bytes.ahead.min(bytes.b);
            }
            in_span.push(b.map(|v| v / (64 * tt_isa::dm::TILE_SLOT)));
            let mut got = vec![s.download(&y).unwrap()];
            for seed in [20u64, 21] {
                s.write(&x, &values(seed, K)).unwrap();
                s.replay(id).unwrap();
                s.sync().unwrap();
                got.push(s.download(&y).unwrap());
            }
            for (i, (g, w)) in got.iter().zip(&fresh).enumerate() {
                assert_bits(&format!("balanced {balanced}, run {i}"), g, w);
            }
            s.release_trace(id).unwrap();
            s.free(y).unwrap();
        }
        s.set_balanced_deal(false);
        s.set_count_trace_bytes(false);
        // Columns of `B` (64 tiles each) read in the span, by unit. Units 4-7 read their job's
        // two ahead and units 0-3 their first round's (busy prefetch), so 24 columns are left:
        // without, six each on units 0-3 and none on 4-7; with, three on every unit.
        eprintln!("columns read in the span by unit, without and with: {in_span:?}");
        assert_eq!(in_span[0], [6, 6, 6, 6, 0, 0, 0, 0], "without");
        assert_eq!(in_span[1], [3; 8], "with");
        for t in [w0, w1, x] {
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
/// 4-5 and not 6-7 (anchored at the last product: claim eight takes 6-7 by their own); with a
/// weight stream held to two units and no prefetch, every product runs on units 0-1. Each
/// capture's run and replay are the ops run fresh, bit for bit.
#[test]
fn the_prefetch_and_the_stream_take_the_session_s_unit_counts() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        s.set_prefetch_anchor(PrefetchAnchor::Last);
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

/// Claim seven: with the stream prefetch on units 4-5, the second and third products hand two
/// columns each to units 6-7, whose rest runs as an op of its own on those units alone; every run
/// is the ops run fresh, bit for bit, handed at chunk 1 and at half.
#[test]
fn handed_columns_replay_as_the_ops_run_fresh() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        s.set_matmul_k_block_limit(std::num::NonZeroUsize::new(8));
        s.set_pipeline(false);
        s.set_stream_prefetch_units(2);
        let chain = Chain::upload(s);
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let warm = chain.forward(s, &x, Spread::Fixed(4));
        s.free(warm).unwrap();
        let mut runs = Vec::new();
        for at in [None, Some(1), Some(0)] {
            s.write(&x, &values(10, K)).unwrap();
            s.set_stream_handoff(at);
            s.begin_trace().unwrap();
            let y = chain.forward(s, &x, Spread::Fixed(4));
            let id = s.end_trace().unwrap();
            s.set_stream_handoff(None);
            // The handed columns' second ops: entries on units 6-7 and on none of 1-5 (unit 0
            // coordinates every barrier), each planning the L1 its product did (a capture refuses
            // an op whose layout it cannot place against a persistent region).
            let ops = s.trace_ops(id).unwrap();
            let rests: Vec<usize> = (1..ops.len())
                .filter(|&k| {
                    ops[k].entries[1..6].iter().all(|r| r.is_empty())
                        && ops[k].entries[6..8].iter().all(|r| !r.is_empty())
                })
                .collect();
            for &k in &rests {
                assert!(
                    ops[k].l1_top != 0,
                    "handed at {at:?}: the rest's L1 planned"
                );
                assert_eq!(
                    ops[k].l1_top,
                    ops[k - 1].l1_top,
                    "handed at {at:?}: its product's"
                );
            }
            let rests = rests.len();
            let mut replays = vec![s.download(&y).unwrap()];
            for round in 0..2u64 {
                s.write(&x, &values(20 + round, K)).unwrap();
                s.replay(id).unwrap();
                s.sync().unwrap();
                replays.push(s.download(&y).unwrap());
            }
            s.release_trace(id).unwrap();
            s.free(y).unwrap();
            runs.push((at, rests, replays));
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
        s.set_stream_prefetch_units(tt_kernels::session::PREFETCH_UNITS);
        for (at, rests, replays) in &runs {
            let want = if at.is_some() { 2 } else { 0 };
            assert_eq!(
                *rests, want,
                "handed at {at:?}: the second and third products' rest"
            );
            for (i, w) in fresh.iter().enumerate() {
                assert_bits(&format!("handed at {at:?}, run {i}"), &replays[i], w);
            }
        }
        for t in chain.w.into_iter().chain([x]) {
            s.free(t).unwrap();
        }
    });
}

fn mm<T: tt_device::Transport>(s: &mut Session<T>, a: &DramTensor, b: &DramTensor) -> DramTensor {
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
}

fn ew<T: tt_device::Transport>(
    s: &mut Session<T>,
    kind: u32,
    scalar: f32,
    a: &DramTensor,
) -> DramTensor {
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
}

/// Each product's `PREFETCH` lists' anchors (`OpRecord::ahead_waits`), by unit, in order.
fn waits_of_products<T: tt_device::Transport>(
    s: &Session<T>,
    id: tt_kernels::trace::TraceId,
) -> Vec<Vec<(usize, u32)>> {
    s.trace_ops(id)
        .unwrap()
        .iter()
        .filter(|op| op.what.contains("matmul"))
        .map(|op| op.ahead_waits.clone())
        .collect()
}

/// Claim eight: under `Last` and `Own` the prefetch of two units takes units 4-5 for both later
/// products, the third's lists waiting for the second product; under `OwnOldest` the third takes
/// units 6-7, idle since the first, and waits for the first product's barrier, as the second's
/// lists do. Each capture's run and replays are the ops run fresh, bit for bit.
#[test]
fn own_anchors_start_a_prefetch_under_the_products_its_units_sat_out() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        let chain = Chain::upload(s);
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let mut fresh = Vec::new();
        for seed in [10, 20, 21] {
            s.write(&x, &values(seed, K)).unwrap();
            let y = chain.forward(s, &x, Spread::Fixed(4));
            fresh.push(s.download(&y).unwrap());
            s.free(y).unwrap();
        }
        s.set_prefetch_units(2);
        let modes = [
            PrefetchAnchor::Last,
            PrefetchAnchor::Own,
            PrefetchAnchor::OwnOldest,
        ];
        let mut seen = Vec::new();
        for mode in modes {
            s.set_prefetch_anchor(mode);
            s.write(&x, &values(10, K)).unwrap();
            s.begin_trace().unwrap();
            let y = chain.forward(s, &x, Spread::Fixed(4));
            let id = s.end_trace().unwrap();
            let mut runs = vec![s.download(&y).unwrap()];
            for round in 0..2u64 {
                s.write(&x, &values(20 + round, K)).unwrap();
                s.replay(id).unwrap();
                s.sync().unwrap();
                runs.push(s.download(&y).unwrap());
            }
            for (i, want) in fresh.iter().enumerate() {
                assert_bits(&format!("{mode:?}, run {i}"), &runs[i], want);
            }
            let idle: Vec<Vec<bool>> = units_of_products(s, id)
                .iter()
                .map(|p| p[4..].to_vec())
                .collect();
            seen.push((mode, idle, waits_of_products(s, id)));
            s.release_trace(id).unwrap();
            s.free(y).unwrap();
        }
        s.set_prefetch_anchor(PrefetchAnchor::Last);
        s.set_prefetch_units(tt_kernels::session::PREFETCH_UNITS);
        for (mode, idle, waits) in &seen {
            let third = if *mode == PrefetchAnchor::OwnOldest {
                [false, false, true, true]
            } else {
                [true, true, false, false]
            };
            assert_eq!(
                idle,
                &[
                    vec![false; 4],
                    vec![true, true, false, false],
                    third.to_vec()
                ],
                "{mode:?}: the idle units each product takes"
            );
            assert!(
                waits[0].is_empty(),
                "{mode:?}: the first product reads nothing ahead"
            );
            let first: Vec<u32> = waits[1].iter().map(|&(_, w)| w).collect();
            let then: Vec<u32> = waits[2].iter().map(|&(_, w)| w).collect();
            assert!(
                first.len() == 2 && first.iter().all(|&w| w == first[0]),
                "{mode:?}: the second product's lists wait for the first: {waits:?}"
            );
            assert_eq!(then.len(), 2, "{mode:?}: {waits:?}");
            if *mode == PrefetchAnchor::OwnOldest {
                assert!(
                    then.iter().all(|&w| w == first[0]),
                    "{mode:?}: units 6-7 wait for the first product: {waits:?}"
                );
            } else {
                assert!(
                    then.iter().all(|&w| w > first[0]),
                    "{mode:?}: units 4-5 wait for the second product: {waits:?}"
                );
            }
        }
        for t in chain.w.into_iter().chain([x]) {
            s.free(t).unwrap();
        }
    });
}

/// `x @ w0` (on units 0-3), a scale, `B = relu(src)` (made in the capture, between the first two
/// products), `@ w1`, a relu, `@ B`.
fn forward_made_b<T: tt_device::Transport>(
    s: &mut Session<T>,
    chain: &Chain,
    x: &DramTensor,
    src: &DramTensor,
) -> DramTensor {
    let h = mm(s, x, &chain.w[0]);
    let hs = ew(s, kind::MUL_SCALAR, 0.5, &h);
    let b = ew(s, kind::RELU, 0.0, src);
    let g = mm(s, &hs, &chain.w[1]);
    let gr = ew(s, kind::RELU, 0.0, &g);
    let y = mm(s, &gr, &b);
    for t in [h, hs, b, g, gr] {
        s.free(t).unwrap();
    }
    y
}

/// Claim nine, the first control: a `B` the capture made is read ahead by no unit, and the
/// replays over new sources are the ops run fresh; with the guard probed off, `OwnOldest`'s third
/// product reads it on units 6-7 from the first product's barrier, before it is made, and a replay
/// differs, while `Last`'s (waiting for the second product, after it) does not.
#[test]
fn a_b_the_capture_makes_is_not_read_ahead() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        s.set_prefetch_units(2);
        let chain = Chain::upload(s);
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let src = s.upload(&values(30, K * 256), K, 256).unwrap();
        let mut fresh = Vec::new();
        for seed in [30, 40, 41] {
            s.write(&src, &values(seed, K * 256)).unwrap();
            let y = forward_made_b(s, &chain, &x, &src);
            fresh.push(s.download(&y).unwrap());
            s.free(y).unwrap();
        }
        let mut seen = Vec::new();
        for (mode, stale) in [
            (PrefetchAnchor::OwnOldest, false),
            (PrefetchAnchor::Last, true),
            (PrefetchAnchor::OwnOldest, true),
        ] {
            s.set_prefetch_anchor(mode);
            tt_kernels::session::probe_stale_b(stale);
            s.write(&src, &values(30, K * 256)).unwrap();
            s.begin_trace().unwrap();
            let y = forward_made_b(s, &chain, &x, &src);
            let id = s.end_trace().unwrap();
            tt_kernels::session::probe_stale_b(false);
            assert_bits(
                &format!("{mode:?}, stale {stale}: the capture's own run"),
                &s.download(&y).unwrap(),
                &fresh[0],
            );
            let mut same = true;
            for round in 0..2usize {
                s.write(&src, &values(40 + round as u64, K * 256)).unwrap();
                s.replay(id).unwrap();
                s.sync().unwrap();
                let got = s.download(&y).unwrap();
                same &= got
                    .iter()
                    .zip(&fresh[1 + round])
                    .all(|(g, w)| g.to_bits() == w.to_bits());
            }
            let third = units_of_products(s, id)[2][4..].to_vec();
            seen.push((mode, stale, third, same));
            s.release_trace(id).unwrap();
            s.free(y).unwrap();
        }
        s.set_prefetch_anchor(PrefetchAnchor::Last);
        s.set_prefetch_units(tt_kernels::session::PREFETCH_UNITS);
        assert_eq!(
            seen,
            [
                (PrefetchAnchor::OwnOldest, false, vec![false; 4], true),
                (
                    PrefetchAnchor::Last,
                    true,
                    vec![true, true, false, false],
                    true
                ),
                (
                    PrefetchAnchor::OwnOldest,
                    true,
                    vec![false, false, true, true],
                    false
                ),
            ],
            "(anchor, guard probed off, the third product's idle units, replays as fresh)"
        );
        for t in chain.w.into_iter().chain([x, src]) {
            s.free(t).unwrap();
        }
    });
}

/// `x @ w0` (units 0-3), a scale, with `x64` a 64-row `x64 @ w2` on units 0-3 (an op that
/// plans most of `DATA`), `@ w1` on units 0-1 alone, a relu, `@ w2` on units 0-3: units 2-3 sit
/// out the second product. Returns the result and the 64-row product's.
fn forward_skip<T: tt_device::Transport>(
    s: &mut Session<T>,
    chain: &Chain,
    x: &DramTensor,
    x64: Option<&DramTensor>,
) -> (DramTensor, Option<DramTensor>) {
    let h = mm(s, x, &chain.w[0]);
    let hs = ew(s, kind::MUL_SCALAR, 0.5, &h);
    let wide = x64.map(|a| mm(s, a, &chain.w[2]));
    s.set_spread(Spread::Fixed(2));
    let g = mm(s, &hs, &chain.w[1]);
    s.set_spread(Spread::Fixed(4));
    let gr = ew(s, kind::RELU, 0.0, &g);
    let y = mm(s, &gr, &chain.w[2]);
    for t in [h, hs, g, gr] {
        s.free(t).unwrap();
    }
    (y, wide)
}

/// Claim nine, the second control: under `Own`, the third product's busy units 2-3, which sat
/// out the second, read their `B` from just after the first product (before the scale) -- unless
/// an op between planned into their ring (the 64-row product), when it stays after that op. With
/// `l1_top` probed away it moves across that op too, and the bits differ.
#[test]
fn a_busy_unit_reads_ahead_from_its_own_anchor_but_not_across_an_op_in_its_ring() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        s.set_weight_prefetch(false);
        let chain = Chain::upload(s);
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let x64 = s.upload(&values(50, 64 * K), 64, K).unwrap();
        let (y, wide) = forward_skip(s, &chain, &x, Some(&x64));
        let want = [
            s.download(&y).unwrap(),
            s.download(wide.as_ref().unwrap()).unwrap(),
        ];
        s.free(y).unwrap();
        s.free(wide.unwrap()).unwrap();
        let (y, _) = forward_skip(s, &chain, &x, None);
        let want_plain = s.download(&y).unwrap();
        s.free(y).unwrap();
        s.set_prefetch_anchor(PrefetchAnchor::Own);
        let mut seen = Vec::new();
        for (with_wide, ignore) in [(false, false), (true, false), (true, true)] {
            tt_kernels::session::probe_busy_ignore_tops(ignore);
            s.begin_trace().unwrap();
            let (y, wide) = forward_skip(s, &chain, &x, with_wide.then_some(&x64));
            let id = s.end_trace().unwrap();
            tt_kernels::session::probe_busy_ignore_tops(false);
            // Where unit 2's read ahead of the third product sits, against the scale's and the
            // 64-row product's entries there.
            let ops = s.trace_ops(id).unwrap();
            let products: Vec<_> = ops.iter().filter(|op| op.what.contains("matmul")).collect();
            let third = products.last().unwrap();
            let ahead = third
                .ahead
                .iter()
                .find(|(u, _)| *u == 2)
                .map(|(_, r)| r.start)
                .expect("unit 2 reads the third product's B ahead");
            // The first op on unit 2 after the first product: the scale.
            let first = ops
                .iter()
                .position(|op| op.what.contains("matmul"))
                .unwrap();
            let scale = ops[first + 1..]
                .iter()
                .find(|op| !op.entries[2].is_empty())
                .map(|op| op.entries[2].start)
                .unwrap();
            let wide_end = with_wide.then(|| products[1].entries[2].end);
            let mut same = true;
            for _ in 0..2 {
                s.replay(id).unwrap();
                s.sync().unwrap();
                let w = if with_wide { &want[0] } else { &want_plain };
                same &= s
                    .download(&y)
                    .unwrap()
                    .iter()
                    .zip(w)
                    .all(|(g, w)| g.to_bits() == w.to_bits());
                if let Some(t) = &wide {
                    same &= s
                        .download(t)
                        .unwrap()
                        .iter()
                        .zip(&want[1])
                        .all(|(g, w)| g.to_bits() == w.to_bits());
                }
            }
            seen.push((
                with_wide,
                ignore,
                ahead <= scale,
                wide_end.is_some_and(|e| ahead >= e),
                same,
            ));
            s.release_trace(id).unwrap();
            s.free(y).unwrap();
            if let Some(t) = wide {
                s.free(t).unwrap();
            }
        }
        s.set_prefetch_anchor(PrefetchAnchor::Last);
        s.set_weight_prefetch(true);
        assert_eq!(
            seen,
            [
                (false, false, true, false, true),
                (true, false, false, true, true),
                (true, true, true, false, false),
            ],
            "(the 64-row product between, l1_top ignored, read ahead from before the scale, \
             after the 64-row product, replays as fresh)"
        );
        for t in chain.w.into_iter().chain([x, x64]) {
            s.free(t).unwrap();
        }
    });
}
