//! F4a.1 (`docs/plans/dataflow-compiler.md`, H4): a capture's kept barriers lowered to per-edge
//! credits (`Session::set_edge_credits`, `tt_kernels::barriers::credit`). Where a barrier's next
//! run's units each depend on a few producers, each producer credits each consumer's word
//! (`op::CREDIT`, after its own requests) where its arrival was, and each consumer waits for its
//! count (`op::WAIT_CREDIT`) before its first op of the run; at most two such barriers in a row
//! (one credit word each), then a barrier.
//!
//! The claims:
//! - **Same bits.** Element-wise ops dealt over eight units, then four, then eight (each of the
//!   four reads tiles two of the eight wrote: a sparse barrier, two producers a consumer), captured
//!   with credits and without, replay over new inputs as the ops run fresh, bit for bit; and the
//!   credits were lowered (`Session::last_edge_credits`).
//! - **Mutations are refused before anything runs.** A dropped edge (one producer's credit left
//!   out) is refused by the coverage check, and an unbalanced credit (a consumer taking two a
//!   firing from a producer that sends one) by `channel::Graph::check`; the trace then keeps its
//!   barriers and still gives the same bits. The interleavings themselves are `barriers`' unit
//!   tests (`sparse_barriers_lower_to_credits_that_order_every_dependency`, whose control prunes
//!   the same barriers with no credits and finds a race).
//! - **A credited decode is the barriered one** (silicon, a model workload): SmolLM2-135M's traced
//!   generation with credits equals its eager generation (every barrier kept) token for token and
//!   bit for bit, and its captures lowered barriers.

use tt_kernels::kind;
use tt_kernels::session::{Session, Spread, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_ttsim::fork_scope;

#[cfg(feature = "silicon")]
mod ladder_support;

/// Elements of the operand, `[32, 16384]`: 512 tiles, in jobs over every unit.
const N: usize = 32 * 16384;

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

/// `x` scaled on eight units, then a relu on four, a scale on eight, a relu on four, a scale on
/// eight: each change of deal a sparse barrier.
fn forward<T: tt_device::Transport>(s: &mut Session<T>, x: &DramTensor) -> DramTensor {
    let mut t = None::<DramTensor>;
    for (i, spread) in [8, 4, 8, 4, 8].into_iter().enumerate() {
        s.set_spread(Spread::Fixed(spread));
        let (kind, scalar) = match i % 2 {
            0 => (kind::MUL_SCALAR, 0.5 + i as f32),
            _ => (kind::RELU, 0.0),
        };
        let next = s
            .eltwise(
                Eltwise {
                    kind,
                    scalar,
                    scalar2: 0.0,
                },
                t.as_ref().unwrap_or(x),
                None,
            )
            .unwrap();
        if let Some(t) = t.replace(next) {
            s.free(t).unwrap();
        }
    }
    s.set_spread(Spread::All);
    t.unwrap()
}

/// Capture [`forward`] with credits at `fan` (`None`: barriers), replay it over `inputs`, and
/// return the replays' results (the capture's own run first) and what the credits did.
fn captured<T: tt_device::Transport>(
    s: &mut Session<T>,
    x: &DramTensor,
    fan: Option<usize>,
    inputs: &[u64],
) -> (
    Vec<Vec<f32>>,
    Option<Result<tt_kernels::barriers::Credited, String>>,
) {
    s.write(x, &values(inputs[0], N)).unwrap();
    s.set_edge_credits(fan);
    s.begin_trace().unwrap();
    let y = forward(s, x);
    let id = s.end_trace().unwrap();
    let credited = s.last_edge_credits();
    s.set_edge_credits(None);
    let mut runs = vec![s.download(&y).unwrap()];
    for &seed in &inputs[1..] {
        s.write(x, &values(seed, N)).unwrap();
        s.replay(id).unwrap();
        s.sync().unwrap();
        runs.push(s.download(&y).unwrap());
    }
    s.release_trace(id).unwrap();
    s.free(y).unwrap();
    (runs, credited)
}

#[test]
fn sparse_ops_replay_with_edge_credits_bit_for_bit() {
    with_session(|s| {
        let x = s.upload(&values(10, N), 32, N / 32).unwrap();
        let warm = forward(s, &x);
        s.free(warm).unwrap();
        let inputs = [10, 20, 21, 22];
        let mut want = Vec::new();
        for &seed in &inputs {
            s.write(&x, &values(seed, N)).unwrap();
            let y = forward(s, &x);
            want.push(s.download(&y).unwrap());
            s.free(y).unwrap();
        }
        let (bare, none) = captured(s, &x, None, &inputs);
        assert!(none.is_none(), "credits off: nothing lowered");
        let (credited, done) = captured(s, &x, Some(8), &inputs);
        let done = done.expect("credits on").expect("lowered");
        println!("credits: {done:?}");
        assert!(done.barriers[1] > 0, "a barrier lowered: {done:?}");
        assert!(done.edges > 0 && done.waits > 0, "{done:?}");
        for (i, w) in want.iter().enumerate() {
            assert_bits(&format!("barriers, run {i}"), &bare[i], w);
            assert_bits(&format!("credits, run {i}"), &credited[i], w);
        }
        // The mutations: refused at the capture's end, which then keeps its barriers.
        type Mutation = (&'static str, fn(bool), &'static str);
        let mutations: [Mutation; 2] = [
            (
                "a dropped edge",
                tt_kernels::barriers::probe_drop_credit_edge,
                "neither a credit nor a barrier",
            ),
            (
                "an unbalanced credit",
                tt_kernels::barriers::probe_unbalance_credit,
                "as a graph",
            ),
        ];
        for (what, probe, why) in mutations {
            probe(true);
            let (runs, done) = captured(s, &x, Some(8), &inputs);
            probe(false);
            let e = done.expect("credits on").expect_err(what);
            println!("{what}: {e}");
            assert!(e.contains(why), "{what}: {e}");
            for (i, w) in want.iter().enumerate() {
                assert_bits(&format!("{what}, barriers kept, run {i}"), &runs[i], w);
            }
        }
        s.free(x).unwrap();
    });
}

/// SmolLM2-135M's traced generation with credits (`TT_EDGE_CREDITS`, read as each session opens)
/// against its eager generation, every barrier kept: `step187`'s comparison.
#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask silicon --release --filter step265_edge_credits"
)]
fn a_credited_decode_equals_the_eager_one() {
    #[cfg(feature = "silicon")]
    {
        use tt_models::generate::Shapes;
        use tt_models::llama::{Llama, Precision, SMOLLM2_135M};
        use tt_tests::burn_device::{with_device, Config};
        // SAFETY: each silicon test runs in a process of its own, before any session opens.
        unsafe { std::env::set_var("TT_EDGE_CREDITS", "4") };
        let before = tt_kernels::barriers::lowered_barriers();
        let config = Config {
            tiles: Some(burn_tt::TileChoice::All),
            ..Config::default()
        };
        with_device(config, |d| {
            let (model, _) =
                Llama::<burn_tt::TtBackend>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &d)
                    .unwrap();
            ladder_support::traced_equals_eager(
                &model,
                &Shapes::new(16, &[4, 8]),
                &[1, 5, 12],
                3,
                "SmolLM2 with edge credits",
                &d,
            );
            let lowered = tt_kernels::barriers::lowered_barriers() - before;
            println!("barriers lowered to credits: {lowered}");
            assert!(lowered > 0, "the captures lowered no barrier");
        });
    }
}
