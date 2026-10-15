//! SmolLM2's decode step with fixed shapes, captured once and replayed at every position.
//!
//! `Llama::forward_embedded` grows its attention by a position a step and writes its cache with
//! `slice_assign` at a host range, so no two steps have the same shapes and a trace of one replays
//! only that one. `Llama::step_fixed` takes the position as a tensor:
//! - the rotary rows are selected from a table;
//! - the key and value land through `mask_where` on a one-hot of the position;
//! - attention reads the whole cache with the later positions masked.
//!
//! `burn_tt::TracedInference::capture_with_state` writes the cache back in place after each replay.
//!
//! The claims, on the checkpoint's weights (the simulator runs `SMALL_LAYERS`):
//! - **every fixed step's logits equal the growing cache's, bit for bit.** A masked position adds an
//!   exact zero to the softmax's sum and to `P V`, and the rotary rows are the same products;
//! - **one captured step, replayed at each later position** with that token's embedding and the
//!   position as inputs, gives the eager fixed steps' logits bit for bit, the cache carried from
//!   replay to replay on the card;
//! - **without the write-back** (`capture`, no state) the replays diverge: the second replay reads a
//!   cache that misses the first replay's position (negative control).
//!
//! Everything runs natively (`assert_native_model`).

use burn::tensor::{Int, Tensor, TensorData, TensorPrimitive};
use burn_tt::{InputPayload, OutputPayload, TracedInference, TtBackend, TtTensor};
use tt_models::llama::{Llama, Precision, FULL_LAYERS, HIDDEN};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// "The capital of France is Paris", tokenised with the model's own tokenizer.
const TOKENS: [i32; 6] = [504, 3575, 282, 4649, 314, 7042];
const SMALL_LAYERS: usize = 1;
const MAX_LEN: usize = 8;

fn layers() -> usize {
    if cfg!(feature = "silicon") {
        FULL_LAYERS
    } else {
        SMALL_LAYERS
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn float(t: Tensor<TtBackend, 4>) -> TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!(),
    }
}

fn float2(t: Tensor<TtBackend, 2>) -> TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!(),
    }
}

fn logits(out: &[OutputPayload]) -> Vec<f32> {
    match &out[0] {
        OutputPayload::F32(v) => v.clone(),
        other => panic!("logits as {other:?}"),
    }
}

/// Each token's embedding, `[HIDDEN]` values read back: a replay's input.
fn embeddings(m: &Llama<TtBackend>, d: &burn_tt::TtDevice) -> Vec<Vec<f32>> {
    TOKENS
        .iter()
        .map(|&t| {
            let ids = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(vec![t], [1, 1]), d);
            m.embed(ids).into_data().to_vec::<f32>().unwrap()
        })
        .collect()
}

fn embedded(row: &[f32], d: &burn_tt::TtDevice) -> Tensor<TtBackend, 3> {
    Tensor::from_data(TensorData::new(row.to_vec(), [1, 1, HIDDEN]), d)
}

fn position(p: usize, d: &burn_tt::TtDevice) -> Tensor<TtBackend, 1, Int> {
    Tensor::from_data(TensorData::new(vec![p as i32], [1]), d)
}

/// The fixed step's logits at every position, eagerly.
fn fixed_run(m: &Llama<TtBackend>, rows: &[Vec<f32>], d: &burn_tt::TtDevice) -> Vec<Vec<f32>> {
    let dtype = Precision::F32.dtype();
    let mut cache = m.fixed_cache(MAX_LEN, dtype, d);
    rows.iter()
        .enumerate()
        .map(|(p, row)| {
            m.step_fixed(embedded(row, d), position(p, d), &mut cache, dtype)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        })
        .collect()
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-tests/weights"
)]
fn fixed_steps_equal_the_growing_cache() {
    with_device(Config::default(), |d| {
        let dtype = Precision::F32.dtype();
        let (m, _) = Llama::<TtBackend>::pretrained(layers(), Precision::F32, &d).unwrap();
        let rows = embeddings(&m, &d);
        let mut cache = m.cache(MAX_LEN, dtype, &d);
        let growing: Vec<Vec<f32>> = rows
            .iter()
            .map(|row| {
                m.forward_embedded(embedded(row, &d), &mut cache, dtype)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
            })
            .collect();
        let (fixed, report) = burn_tt::with_report(|| fixed_run(&m, &rows, &d));
        assert_native_model(&report);
        for (p, (f, g)) in fixed.iter().zip(&growing).enumerate() {
            assert_eq!(bits(f), bits(g), "position {p}");
        }
    });
}

/// One fixed step captured at position 1 (after an eager step at 0 made the cache resident), then
/// replayed at 2.. with `stateful` write-back or not; the logits of every replay. The eager step and
/// the capture are checked to run natively; a replay's own writes of its inputs are uploads by design.
fn replays(
    m: &Llama<TtBackend>,
    rows: &[Vec<f32>],
    d: &burn_tt::TtDevice,
    stateful: bool,
) -> Vec<Vec<f32>> {
    let ((trace, first), report) = burn_tt::with_report(|| capture(m, rows, d, stateful));
    // The trace's input, made resident by the caller (`ensure_resident`), is the one transfer
    // outside an op: an upload, nothing read back or staged.
    let report = burn_tt::Report(
        report
            .0
            .into_iter()
            .filter(|s| !(s.op == "(outside an op)" && s.downloaded == 0 && s.staged == 0))
            .collect(),
    );
    assert_native_model(&report);
    let mut out = vec![logits(&first)];
    for (p, row) in rows.iter().enumerate().skip(2) {
        let o = trace
            .run(vec![
                InputPayload::F32(row.clone()),
                InputPayload::Bits(vec![p as u32]),
            ])
            .unwrap();
        out.push(logits(&o));
    }
    out
}

fn capture(
    m: &Llama<TtBackend>,
    rows: &[Vec<f32>],
    d: &burn_tt::TtDevice,
    stateful: bool,
) -> (TracedInference, Vec<OutputPayload>) {
    let dtype = Precision::F32.dtype();
    let mut cache = m.fixed_cache(MAX_LEN, dtype, d);
    let _ = m
        .step_fixed(embedded(&rows[0], d), position(0, d), &mut cache, dtype)
        .into_data();
    let x = embedded(&rows[1], d);
    let pos = position(1, d);
    let TensorPrimitive::Float(xp) = x.clone().into_primitive() else {
        unreachable!()
    };
    xp.ensure_resident();
    let pp = pos.clone().into_primitive();
    pp.ensure_resident();
    let old: Vec<TtTensor> = cache.state().into_iter().map(float).collect();
    TracedInference::capture_with_state(&[&xp, &pp], || {
        let y = float2(m.step_fixed(x.clone(), pos.clone(), &mut cache, dtype));
        let new: Vec<TtTensor> = cache.state().into_iter().map(float).collect();
        let updates = if stateful {
            new.into_iter().zip(old.iter().cloned()).collect()
        } else {
            Vec::new()
        };
        (vec![y], updates)
    })
    .unwrap()
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-tests/weights"
)]
fn a_traced_step_replays_every_position() {
    with_device(Config::default(), |d| {
        let (m, _) = Llama::<TtBackend>::pretrained(layers(), Precision::F32, &d).unwrap();
        let rows = embeddings(&m, &d);
        let eager = fixed_run(&m, &rows, &d);
        let traced = replays(&m, &rows, &d, true);
        for (i, t) in traced.iter().enumerate() {
            assert_eq!(bits(t), bits(&eager[i + 1]), "position {}", i + 1);
        }
        // Negative control: the cache is not carried, so a later replay misses earlier positions.
        let stale = replays(&m, &rows, &d, false);
        assert_eq!(bits(&stale[0]), bits(&eager[1]), "the capture's own run");
        assert!(
            stale
                .iter()
                .zip(&eager[1..])
                .skip(2)
                .any(|(s, e)| bits(s) != bits(e)),
            "a trace without the state write-back must diverge"
        );
    });
}
