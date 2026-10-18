//! F0 of `docs/plans/decode-small-ops.md`: the decode step captured while the model runs on
//! `burn_tt::Tt` (burn-fusion over `TtBackend`) rather than on `TtBackend` itself.
//!
//! The claims:
//! - **A decoder on `Tt` captures and replays.** `TracedGenerator<_, Tt>` resolves each capture's
//!   outputs and new state inside the capture, so the fusion stream runs to its end there and the
//!   trace records it. Its traced generation equals the same model's eager fixed-shape schedule on
//!   `Tt`, token for token and bit for bit, unfused and with `Llama::fused` projections. The
//!   captures run natively and the generations run no Burn op.
//! - **On `Tt` the step is `TtBackend`'s, bit for bit.** burn-fusion 0.21 does not forward
//!   `ActivationOps` (`impl ActivationOps<Self> for Fusion<B> {}`), so SiLU's `sigmoid` reaches the
//!   stream as Burn's default composition -- a cast, `-x`, `exp`, `+ 1`, `log`, a negation, `exp` and
//!   a cast. The chain fuser (`burn_tt::fusion::chain`) runs `silu(gate) * up` as `TtBackend`'s ops
//!   instead, and each RMSNorm up to its scale as one kernel: asserted from the ops each capture
//!   ran (a fused `silu(gate) * up` for each of `TtBackend`'s sigmoids, a fused norm for each
//!   `mean_dim`, no `exp`, `log` or cast) and by the traced logits, equal to `TtBackend`'s bit for
//!   bit.
//! - **Negative control:** one weight scaled by `1 + 2^-8` gives a different traced generation.
//!
//! Timings are printed (capture with its warm-up, the prefills, the mean decode replay). On the
//! simulator they are its time, not the card's.
//!
//! What burn-fusion's stream holds is printed by `BURN_FUSION_LOG=full` (burn-std's logger) with
//! `--nocapture`.

mod ladder_support;

use std::collections::HashMap;
use std::time::Instant;

use burn::module::{Module, ModuleMapper, Param};
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_tt::{Tt, TtBackend, TtDevice};
use ladder_support::{assert_same, bits, prompt};
use tt_models::generate::{generate_fixed, Generation, Shapes, TracedBackend, TracedGenerator};
use tt_models::llama::{Llama, LlamaConfig};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

const MAX_LEN: usize = 16;
const BUCKETS: [usize; 2] = [4, 8];
const STEPS: usize = 4;
/// One chunk padded, and a chunk of 8 then the rest through the 4.
const LENGTHS: [usize; 2] = [3, 9];

fn shapes() -> Shapes {
    Shapes::new(MAX_LEN, &BUCKETS)
}

/// Two of step187's synthetic Llamas: multi-head with head width 32, and grouped-query 4:1.
fn configs() -> Vec<(&'static str, LlamaConfig)> {
    let base = LlamaConfig {
        hidden: 64,
        heads: 2,
        kv_heads: 2,
        head_dim: 32,
        ff: 96,
        vocab: 100,
        layers: 2,
        rope_theta: 10_000.0,
        eps: 1e-5,
        tied_head: true,
    };
    vec![
        ("multi-head, tied", base),
        (
            "grouped-query 4:1, untied",
            LlamaConfig {
                hidden: 128,
                heads: 4,
                kv_heads: 1,
                ff: 160,
                vocab: 64,
                rope_theta: 100_000.0,
                tied_head: false,
                ..base
            },
        ),
        (
            "head width 64, multi-head",
            LlamaConfig {
                hidden: 128,
                heads: 2,
                kv_heads: 2,
                head_dim: 64,
                ..base
            },
        ),
    ]
}

/// `model`'s weights in a model of the same architecture on `B2`.
fn moved<B2: Backend<Device = TtDevice>>(
    model: &Llama<TtBackend>,
    config: &LlamaConfig,
    d: &TtDevice,
) -> Llama<B2> {
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = recorder
        .record(model.clone().into_record(), ())
        .expect("record the weights");
    let record = recorder.load(bytes, d).expect("load the weights");
    Llama::<B2>::new(config, d).load_record(record)
}

/// Scales the first rank-1 parameter (an RMSNorm's gamma) by `1 + 2^-8`.
struct Nudge(bool);

impl<B: Backend> ModuleMapper<B> for Nudge {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if D != 1 || self.0 {
            return param;
        }
        self.0 = true;
        param.map(|t| t.mul_scalar(1.0 + 1.0 / 256.0))
    }
}

/// What every thread's Burn ops did during `f`. Under `Tt`, burn-fusion runs the ops on the thread
/// that drains its stream, not always the caller's, so `burn_tt::with_report` (one thread's) would
/// miss them: this reads the process-wide table, which is this gate's alone (it runs in a fork).
fn reported<R>(f: impl FnOnce() -> R) -> (R, burn_tt::Report) {
    burn_tt::report_reset();
    let r = f();
    (r, burn_tt::report())
}

/// A traced generation of every prompt length.
struct Run {
    generations: Vec<(usize, Generation)>,
    /// The Burn ops the warm-up and captures ran, by calls.
    ops: HashMap<&'static str, u64>,
}

/// A traced generation of every prompt length on `B`, with its timings printed. The captures and
/// warm-up run natively (`assert_native_model`; the trace inputs made resident are the only
/// transfers outside an op), and the generations run no Burn op and stage nothing.
fn traced_run<B: TracedBackend>(model: &Llama<B>, what: &str, d: &TtDevice) -> Run {
    let t0 = Instant::now();
    let (g, report) = reported(|| TracedGenerator::<_, B>::new(model, &shapes(), d));
    let capture = t0.elapsed();
    let g = g.unwrap();
    let ops = report.0.iter().map(|s| (s.op, s.calls)).collect();
    assert_native_model(&burn_tt::Report(
        report
            .0
            .into_iter()
            .filter(|s| !(s.op == "(outside an op)" && s.downloaded == 0 && s.staged == 0))
            .collect(),
    ));
    let generations: Vec<_> = LENGTHS
        .iter()
        .map(|&len| {
            let p = prompt(len, model.config().vocab);
            let (out, report) = reported(|| g.generate(&p, STEPS));
            for s in &report.0 {
                assert_eq!(
                    s.op, "(outside an op)",
                    "{what}: a traced generation ran a Burn op"
                );
                assert_eq!(s.staged, 0, "{what}: a traced generation staged");
            }
            (len, out.unwrap())
        })
        .collect();
    let prefills: Vec<_> = generations.iter().map(|(_, g)| g.times[0]).collect();
    let steps: Vec<_> = generations
        .iter()
        .flat_map(|(_, g)| g.times[1..].to_vec())
        .collect();
    let mean = steps.iter().sum::<std::time::Duration>() / steps.len() as u32;
    eprintln!(
        "{what}: capture and warm-up {capture:.2?}, prefills {prefills:.2?}, decode replay mean \
         {mean:.2?} over {}",
        steps.len()
    );
    Run { generations, ops }
}

#[test]
fn a_decoder_on_tt_captures_and_replays_as_eagerly() {
    with_device(Config::default(), |d| {
        for (what, config) in configs() {
            let reference = Llama::<TtBackend>::new(&config, &d);
            for fused in [false, true] {
                let model = moved::<Tt>(&reference, &config, &d);
                let model = if fused { model.fused() } else { model };
                let what = format!("{what}{}", if fused { ", fused" } else { "" });
                burn_tt::reset_fusion_counters();
                let run = traced_run(&model, &format!("Tt, {what}"), &d);
                for (len, got) in &run.generations {
                    let p = prompt(*len, config.vocab);
                    let want = generate_fixed(&model, &p, STEPS, &shapes(), &d).unwrap();
                    assert_same(got, &want, &format!("{what}, prompt {len}, eager on Tt"));
                }
                eprintln!(
                    "{what}: fusers that ran: add+relu {}, relu {}",
                    burn_tt::fused_add_relu_count(),
                    burn_tt::fused_relu_count()
                );
            }
        }
    });
}

#[test]
fn tt_and_tt_backend_compared() {
    with_device(Config::default(), |d| {
        for (what, config) in configs() {
            let reference = Llama::<TtBackend>::new(&config, &d);
            let native = traced_run(&reference, &format!("TtBackend, {what}"), &d);
            let model = moved::<Tt>(&reference, &config, &d);
            let on_tt = traced_run(&model, &format!("Tt, {what}"), &d);
            let calls = |run: &Run, op: &str| run.ops.get(op).copied().unwrap_or(0);
            // A SiLU a layer, each warm-up and capture.
            assert!(calls(&native, "sigmoid") > 0, "{what}: TtBackend's sigmoid");
            for op in ["float_exp", "float_log", "float_cast"] {
                assert_eq!(calls(&native, op), 0, "{what}: TtBackend ran {op}");
                assert_eq!(calls(&on_tt, op), 0, "{what}: Tt ran {op}");
            }
            // Each SiLU and its `* up` one fused op on `Tt`, and each RMSNorm (Llama's only
            // `mean_dim`) one up to its scale.
            assert_eq!(
                (calls(&on_tt, "tt_silu_mul"), calls(&on_tt, "sigmoid")),
                (calls(&native, "sigmoid"), 0),
                "{what}: every silu(gate) * up fused"
            );
            assert!(
                calls(&native, "float_mean_dim") > 0,
                "{what}: TtBackend's norms"
            );
            assert_eq!(
                (
                    calls(&on_tt, "tt_rms_norm"),
                    calls(&on_tt, "float_mean_dim")
                ),
                (calls(&native, "float_mean_dim"), 0),
                "{what}: every RMSNorm fused"
            );
            // Each RoPE application's `x cos + rotate(x) sin` is one `MUL_ADD` on `Tt` where its
            // halves are whole tiles (`head_dim` 64): an `add` fewer for each. At 32 the table does
            // not lie over the heads by tile copies, and the sum runs as itself. The generations
            // are `TtBackend`'s bit for bit either way (below).
            // (A single key head's table needs no relaying: its sums fuse at 32 too.)
            let mul_adds = calls(&on_tt, "tt_mul_add");
            assert!(mul_adds > 0, "{what}: RoPE's sums matched");
            let saved = calls(&native, "float_add") - calls(&on_tt, "float_add");
            assert!(
                saved <= mul_adds,
                "{what}: {saved} adds saved by {mul_adds} sums"
            );
            if config.head_dim % 64 == 0 {
                assert_eq!(saved, mul_adds, "{what}: every sum fused");
            }
            for ((len, a), (_, b)) in native.generations.iter().zip(&on_tt.generations) {
                assert_same(b, a, &format!("{what}, prompt {len}, Tt against TtBackend"));
            }
        }
    });
}

#[test]
fn a_nudged_weight_on_tt_diverges() {
    with_device(Config::default(), |d| {
        let (what, config) = configs()[0];
        let reference = Llama::<TtBackend>::new(&config, &d);
        let model = moved::<Tt>(&reference, &config, &d);
        let nudged = moved::<Tt>(&reference, &config, &d).map(&mut Nudge(false));
        let a = traced_run(&model, what, &d);
        let b = traced_run(&nudged, &format!("{what}, nudged"), &d);
        let differs = a
            .generations
            .iter()
            .zip(&b.generations)
            .any(|((_, x), (_, y))| {
                x.logits
                    .iter()
                    .zip(&y.logits)
                    .any(|(x, y)| bits(x) != bits(y))
            });
        assert!(differs, "a nudged weight must change the traced logits");
    });
}

/// SmolLM2-135M's first layer, its checkpoint's weights and fused projections, on `Tt`: as the
/// synthetic models.
#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-tests/weights"
)]
fn smollm2_on_tt_captures_and_replays_as_eagerly() {
    use tt_models::llama::Precision;
    with_device(Config::default(), |d| {
        let (model, _) = Llama::<Tt>::smollm2_135m(1, Precision::F32, &d).unwrap();
        let model = model.fused();
        let run = traced_run(&model, "Tt, SmolLM2-135M (1 layer), fused", &d);
        for (len, got) in &run.generations {
            let p = prompt(*len, model.config().vocab);
            let want = generate_fixed(&model, &p, STEPS, &shapes(), &d).unwrap();
            assert_same(got, &want, &format!("SmolLM2, prompt {len}, eager on Tt"));
        }
    });
}
