//! decode-to-bandwidth.md Phase 8, Step 5: in a capture, the units a one-row product with a packed
//! `A` runs on read their first job's `B` ahead, as soon as the last such product is done and under
//! the ops between (`Session::set_busy_prefetch`, `tensor::BUSY_PREFETCH`). The list is queued with
//! the product, and the capture moves it back in the unit's stream to just after the last product;
//! its reads are a prefetch (`record::GATHER_PREFETCH`), which the ops' drains leave in flight, and
//! the job waits for them before its kernel. `B`'s ring sits at the top of `DATA`
//! (`matmul::plan_layout_packed_a`), above what the element-wise ops between plan.
//!
//! Three products chained on four units of eight, an element-wise op between each two on the same
//! four units; the prefetch on idle units off, so only the busy one is seen. The claims:
//! - **Same bits.** With it on, the capture's own run and replays over new inputs are the ops run
//!   fresh, bit for bit, as with it off.
//! - **Taken, and moved back.** With it on, the element-wise op before each later product starts at
//!   least five entries later on each of units 0-3 (the `GATHER` moved in ahead of it) with its
//!   own entries unchanged, and the product holds four fewer there (the `GATHER` gone, its `WAIT`
//!   added).
//!
//! - **Counted as the product's.** What a product's units read ahead, busy and idle alike, is the
//!   product's in the trace's byte count (`OpBytes::ahead`): each product's `B` is its weight's
//!   tiles, once, the trace moves as many bytes with the prefetch as without, and the products
//!   after the first read part of their `B` ahead -- the part the count missed before, when a
//!   moved list was no op's (the control: the bytes in the products' own spans fall short).
//!
//! Whether a unit qualifies (the ops before planned below `B`'s ring, each known) is
//! `tensor`'s unit test `a_units_first_job_reads_its_b_ahead_where_the_ops_before_plan_below_it`.

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

/// `x @ w0`, a scale, `@ w1`, a relu, `@ w2`: every op on four units.
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

/// Each captured op's ranges on units 0-3.
type Ranges = Vec<Vec<std::ops::Range<usize>>>;

/// Capture [`forward`] with busy prefetch `on`, replay it over `inputs`, and return the replays'
/// results (the capture's own run first) and each op's ranges on units 0-3.
fn captured<T: tt_device::Transport>(
    s: &mut Session<T>,
    w: &[DramTensor; 3],
    x: &DramTensor,
    on: bool,
    inputs: &[u64],
) -> (Vec<Vec<f32>>, Ranges) {
    s.write(x, &values(inputs[0], K)).unwrap();
    s.set_busy_prefetch(on);
    s.begin_trace().unwrap();
    let y = forward(s, w, x);
    let id = s.end_trace().unwrap();
    s.set_busy_prefetch(true);
    let mut runs = vec![s.download(&y).unwrap()];
    for &seed in &inputs[1..] {
        s.write(x, &values(seed, K)).unwrap();
        s.replay(id).unwrap();
        s.sync().unwrap();
        runs.push(s.download(&y).unwrap());
    }
    let ranges = s
        .trace_ops(id)
        .unwrap()
        .iter()
        .map(|op| op.entries[..4].to_vec())
        .collect();
    s.release_trace(id).unwrap();
    s.free(y).unwrap();
    (runs, ranges)
}

#[test]
fn a_products_units_read_their_b_under_the_ops_before_it() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        s.set_weight_prefetch(false);
        let w = [(1, K), (2, K), (3, 256)]
            .map(|(seed, n)| s.upload(&values(seed, K * n), K, n).unwrap());
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let warm = forward(s, &w, &x);
        s.free(warm).unwrap();
        let inputs = [10, 20, 21];
        let (off, off_ranges) = captured(s, &w, &x, false, &inputs);
        let (on, on_ranges) = captured(s, &w, &x, true, &inputs);
        // The ops run fresh on each input: no capture, so no prefetch.
        for (i, &seed) in inputs.iter().enumerate() {
            s.write(&x, &values(seed, K)).unwrap();
            let y = forward(s, &w, &x);
            let want = s.download(&y).unwrap();
            s.free(y).unwrap();
            assert_bits(&format!("busy prefetch off, run {i}"), &off[i], &want);
            assert_bits(&format!("busy prefetch on, run {i}"), &on[i], &want);
        }
        // The ops in capture order: the products (the packed-A packing ops among them) and
        // the element-wise ops. The two element-wise ops are the ones whose range differs.
        assert_eq!(off_ranges.len(), on_ranges.len(), "the same ops");
        let mut moved = 0;
        for (k, (a, b)) in off_ranges.iter().zip(&on_ranges).enumerate() {
            for u in 0..4 {
                let (a, b) = (&a[u], &b[u]);
                if a.start == a.end {
                    continue;
                }
                if b.len() == a.len() && b.start >= a.start + 5 {
                    // An op the prefetch moved in ahead of (earlier moves add to the shift).
                    moved += 1;
                } else if b.len() + 4 != a.len() {
                    // Not a product whose first job read ahead (its `GATHER` gone, its `WAIT`
                    // added): the op's own entries are the same.
                    assert_eq!(b.len(), a.len(), "op {k}, unit {u}: its own entries");
                }
            }
        }
        println!("ops moved down by a prefetch moved in ahead of them, unit by unit: {moved}");
        assert!(
            moved >= 8,
            "each of the two later products' prefetches moved ahead of the op before it on four units"
        );
        for t in w.into_iter().chain([x]) {
            s.free(t).unwrap();
        }
    });
}

/// Each captured op's bytes, with busy prefetch `on` (and the idle units' as the session has it).
fn counted<T: tt_device::Transport>(
    s: &mut Session<T>,
    w: &[DramTensor; 3],
    x: &DramTensor,
    on: bool,
) -> Vec<tt_kernels::profile_mode::OpBytes> {
    s.set_busy_prefetch(on);
    s.begin_trace().unwrap();
    let y = forward(s, w, x);
    let id = s.end_trace().unwrap();
    s.set_busy_prefetch(true);
    let bytes = s.trace_ops(id).unwrap().iter().map(|op| op.bytes).collect();
    s.release_trace(id).unwrap();
    s.free(y).unwrap();
    bytes
}

#[test]
fn what_a_product_reads_ahead_is_counted_as_its_own() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        s.set_count_trace_bytes(true);
        let ns = [K, K, 256];
        let w = [(1, ns[0]), (2, ns[1]), (3, ns[2])]
            .map(|(seed, n)| s.upload(&values(seed, K * n), K, n).unwrap());
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let warm = forward(s, &w, &x);
        s.free(warm).unwrap();
        let weights = ns.map(|n| ((K / 32) * n.div_ceil(32)) as u64 * tt_isa::dm::TILE_SLOT);
        for idle in [false, true] {
            s.set_weight_prefetch(idle);
            let off = counted(s, &w, &x, false);
            let on = counted(s, &w, &x, true);
            for (what, ops) in [("busy prefetch off", &off), ("busy prefetch on", &on)] {
                let products: Vec<_> = ops.iter().filter(|b| b.b > 0).collect();
                let b: Vec<u64> = products.iter().map(|b| b.b).collect();
                let ahead: Vec<u64> = products.iter().map(|b| b.ahead).collect();
                println!("idle prefetch {idle}, {what}: products' B {b:?}, read ahead {ahead:?}");
                assert_eq!(b, weights, "idle {idle}, {what}: each product's B, once");
                assert_eq!(ahead[0], 0, "idle {idle}, {what}: nothing before the first");
            }
            let total = |ops: &[tt_kernels::profile_mode::OpBytes]| -> (u64, u64) {
                ops.iter()
                    .fold((0, 0), |(r, w), b| (r + b.read, w + b.write))
            };
            assert_eq!(
                total(&on),
                total(&off),
                "idle {idle}: the same bytes, moved or not"
            );
            // The last product's units read its whole `B` ahead (one job a unit); the middle
            // one's run four jobs pipelined, which reads nothing ahead on a busy unit.
            let products: Vec<_> = on.iter().filter(|b| b.b > 0).collect();
            assert!(
                products[2].ahead > 0,
                "idle {idle}: the last product read nothing ahead"
            );
            for (k, p) in products.iter().enumerate().filter(|(_, p)| p.ahead > 0) {
                // The control: what the count gave when a moved list was no op's.
                assert!(
                    p.b - p.ahead < weights[k],
                    "idle {idle}: product {k}'s own span reads all of its B"
                );
            }
        }
        s.set_count_trace_bytes(false);
        s.set_weight_prefetch(true);
        for t in w.into_iter().chain([x]) {
            s.free(t).unwrap();
        }
    });
}
