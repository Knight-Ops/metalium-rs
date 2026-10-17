//! A traced step's state written in place (`docs/plans/in-place-cache-write.md` D1-D3).
//!
//! A [`SharedStateTraces`] capture names its state before the step runs (a decoder's key/value
//! cache), so a row scatter into a state tensor (`scatter_nd` with `Assign`, the cache write)
//! writes the state's own buffer, and the trace needs neither the scatter's tile copy of the
//! cache nor the write-back after the step. Burn's tensors are values, so the claims are about
//! what that may never change:
//! - **the same results:** a step written in place gives the copying step's outputs and state,
//!   bit for bit, at the capture and on every replay, the state carried from replay to replay;
//! - **writes chain:** a second write into the first's result is in place too;
//! - **the copies are gone:** the trace has no `copy list` and no `copy_into list`, where the
//!   copying capture has one of each (the negative control, `set_state_in_place(false)`);
//! - **a read of the cache after its write is refused by name,** not answered from the step's
//!   future: the old handle, and a view of it taken before the write;
//! - **nothing kept from before is reused:** a transpose kept beside the state (read twice as a
//!   matmul's `B` eagerly) is not read after the capture writes the state;
//! - **an error ends the donation:** after a refused capture, an eager scatter into the same
//!   tensor copies it and leaves it as it was;
//! - **a decoder's generation:** a synthetic Llama's traced decode steps write their caches in
//!   place, and its traced generation equals the copying one's token for token and bit for bit.
//!
//! Watched failing: with `DramRef::current`'s check removed (the stale reads return the new
//! rows), and with the donation's `written_in_place` removed (the kept transpose is read).

use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData, TensorPrimitive};
use burn_tt::{InputPayload, OutputPayload, SharedStateTraces, TtBackend, TtDevice, TtTensor};
use tt_models::generate::{Shapes, TracedGenerator};
use tt_models::llama::{Llama, LlamaConfig};
use tt_tests::burn_device::{with_device, Config};

const HEADS: usize = 4;
const LEN: usize = 64;
const DIM: usize = 32;

fn float2(t: Tensor<TtBackend, 2>) -> TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!(),
    }
}

fn tensor2(t: &TtTensor) -> Tensor<TtBackend, 2> {
    Tensor::from_primitive(TensorPrimitive::Float(t.clone()))
}

fn values(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32 / 7.0)
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn payload(out: &[OutputPayload], i: usize) -> Vec<f32> {
    match &out[i] {
        OutputPayload::F32(v) => v.clone(),
        other => panic!("output {i} as {other:?}"),
    }
}

/// `[HEADS * LEN, DIM]` on the card, resident.
fn cache(d: &TtDevice) -> TtTensor {
    let t = float2(
        Tensor::<TtBackend, 2>::from_data(
            TensorData::new(values(1, HEADS * LEN * DIM), [HEADS * LEN, DIM]),
            d,
        )
        .add_scalar(0.0),
    );
    t.ensure_resident();
    t
}

/// The rows a step at `position` writes: one a head.
fn rows(position: Tensor<TtBackend, 1, Int>, d: &TtDevice) -> Tensor<TtBackend, 2, Int> {
    (Tensor::<TtBackend, 1, Int>::arange(0..HEADS as i64, d).mul_scalar(LEN as i64)
        + position.expand([HEADS]))
    .reshape([HEADS, 1])
}

/// The new rows, from the position, so each replay writes other values: `[HEADS, DIM]`.
fn new_rows(position: Tensor<TtBackend, 1, Int>, d: &TtDevice) -> Tensor<TtBackend, 2> {
    let base =
        Tensor::<TtBackend, 2>::from_data(TensorData::new(values(2, HEADS * DIM), [HEADS, DIM]), d);
    base + position.float().reshape([1, 1]).expand([HEADS, DIM])
}

/// The host's step: `cache` with the step's rows written at `position`.
fn host_step(cache: &mut [f32], position: usize) {
    let base = values(2, HEADS * DIM);
    for h in 0..HEADS {
        for c in 0..DIM {
            cache[(h * LEN + position) * DIM + c] = base[h * DIM + c] + position as f32;
        }
    }
}

fn position(p: i32, d: &TtDevice) -> TtTensor {
    let t = Tensor::<TtBackend, 1, Int>::from_data([p], d).add_scalar(0);
    let t = t.into_primitive();
    t.ensure_resident();
    t
}

/// What each op of a trace queued first, its name (`tensor::stats`).
fn whats(set: &SharedStateTraces<usize>) -> Vec<&'static str> {
    set.get(0)
        .unwrap()
        .ops()
        .unwrap()
        .into_iter()
        .map(|op| op.what)
        .collect()
}

/// One capture of the step over a fresh cache, then replays at 9, 40 and 63: the outputs (the
/// step's sum over the new cache's columns, and the new cache), and what the trace's ops are.
fn run_step(d: &TtDevice) -> (Vec<Vec<f32>>, Vec<&'static str>) {
    let state = cache(d);
    let mut set = SharedStateTraces::<usize>::new(vec![state.clone()]).unwrap();
    let pp = position(5, d);
    let (s, p) = (state.clone(), pp.clone());
    let first = set
        .capture(0, &[&pp], || {
            let at = Tensor::<TtBackend, 1, Int>::from_primitive(p.clone());
            let new = tensor2(&s).scatter_nd(
                rows(at.clone(), d),
                new_rows(at, d),
                IndexingUpdateOp::Assign,
            );
            let sum = new.clone().sum_dim(1);
            (
                vec![float2(sum), float2(new.clone().mul_scalar(1.0))],
                vec![float2(new)],
            )
        })
        .unwrap();
    let mut out = vec![payload(&first, 0), payload(&first, 1)];
    for p in [9u32, 40, 63] {
        let o = set
            .get(0)
            .unwrap()
            .run(vec![InputPayload::Bits(vec![p])])
            .unwrap();
        out.push(payload(&o, 0));
        out.push(payload(&o, 1));
    }
    (out, whats(&set))
}

#[test]
fn a_step_writes_its_state_in_place_with_the_copying_step_s_bits() {
    with_device(Config::default(), |d| {
        burn_tt::set_state_in_place(false);
        let (copying, copying_ops) = run_step(&d);
        burn_tt::set_state_in_place(true);
        let (in_place, in_place_ops) = run_step(&d);
        // The host's cache, each position written in turn: the state carried across replays.
        let mut host = values(1, HEADS * LEN * DIM);
        for (k, p) in [5usize, 9, 40, 63].into_iter().enumerate() {
            host_step(&mut host, p);
            assert_eq!(
                bits(&in_place[2 * k + 1]),
                bits(&host),
                "the cache after position {p}"
            );
        }
        assert_eq!(copying.len(), in_place.len());
        for (k, (a, b)) in copying.iter().zip(&in_place).enumerate() {
            assert_eq!(bits(a), bits(b), "output {k}: in place against the copy");
        }
        let count = |ops: &[&str], what: &str| ops.iter().filter(|w| **w == what).count();
        // The negative control: the copying trace copies the cache and writes it back.
        assert_eq!(count(&copying_ops, "copy list"), 1, "{copying_ops:?}");
        assert_eq!(count(&copying_ops, "copy_into list"), 1, "{copying_ops:?}");
        assert_eq!(count(&in_place_ops, "copy list"), 0, "{in_place_ops:?}");
        assert_eq!(
            count(&in_place_ops, "copy_into list"),
            0,
            "{in_place_ops:?}"
        );
        assert_eq!(in_place_ops.len() + 2, copying_ops.len());
    });
}

/// Two writes to one state tensor in a step: the second scatters into the first's result, both in
/// place, and the state after the step holds both rows, at the capture and on every replay.
#[test]
fn two_writes_to_one_state_chain_in_place() {
    with_device(Config::default(), |d| {
        let state = cache(&d);
        let mut set = SharedStateTraces::<usize>::new(vec![state.clone()]).unwrap();
        let pp = position(5, &d);
        let (s, p) = (state.clone(), pp.clone());
        let first = set
            .capture(0, &[&pp], || {
                let at = Tensor::<TtBackend, 1, Int>::from_primitive(p.clone());
                let next = at.clone().add_scalar(1);
                let once = tensor2(&s).scatter_nd(
                    rows(at.clone(), &d),
                    new_rows(at, &d),
                    IndexingUpdateOp::Assign,
                );
                let twice = once.scatter_nd(
                    rows(next.clone(), &d),
                    new_rows(next, &d),
                    IndexingUpdateOp::Assign,
                );
                (
                    vec![float2(twice.clone().mul_scalar(1.0))],
                    vec![float2(twice)],
                )
            })
            .unwrap();
        let ops = whats(&set);
        assert!(
            !ops.contains(&"copy list") && !ops.contains(&"copy_into list"),
            "{ops:?}"
        );
        let mut host = values(1, HEADS * LEN * DIM);
        host_step(&mut host, 5);
        host_step(&mut host, 6);
        assert_eq!(bits(&payload(&first, 0)), bits(&host), "the capture's run");
        for p in [20u32, 40] {
            let o = set
                .get(0)
                .unwrap()
                .run(vec![InputPayload::Bits(vec![p])])
                .unwrap();
            host_step(&mut host, p as usize);
            host_step(&mut host, p as usize + 1);
            assert_eq!(bits(&payload(&o, 0)), bits(&host), "the replay at {p}");
        }
    });
}

/// A capture whose step reads `read` (made from the cache and the write's result) after the
/// write: refused, with the message naming what happened.
fn refused(
    d: &TtDevice,
    read: fn(
        Tensor<TtBackend, 2>,
        Tensor<TtBackend, 2>,
        Tensor<TtBackend, 2>,
    ) -> Tensor<TtBackend, 2>,
) {
    let state = cache(d);
    let before = tensor2(&state).into_data().to_vec::<f32>().unwrap();
    let mut set = SharedStateTraces::<usize>::new(vec![state.clone()]).unwrap();
    let pp = position(5, d);
    let (s, p) = (state.clone(), pp.clone());
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        set.capture(0, &[&pp], || {
            let old = tensor2(&s);
            // A view taken before the write: the first head's rows.
            let view = old.clone().slice([0..LEN, 0..DIM]);
            let at = Tensor::<TtBackend, 1, Int>::from_primitive(p.clone());
            let new = old.clone().scatter_nd(
                rows(at.clone(), d),
                new_rows(at, d),
                IndexingUpdateOp::Assign,
            );
            let y = read(old, view, new.clone());
            (vec![float2(y)], vec![float2(new)])
        })
    }));
    let message = match caught {
        Ok(r) => panic!("the stale read was not refused: {r:?}"),
        Err(e) => e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    };
    assert!(
        message.contains("after this capture wrote it in place"),
        "refused, but not by name: {message}"
    );
    // The donation ended with the failed capture: an eager scatter into the cache copies it.
    let at = Tensor::<TtBackend, 1, Int>::from_data([7], d).add_scalar(0);
    let written = tensor2(&state).scatter_nd(
        rows(at.clone(), d),
        new_rows(at, d),
        IndexingUpdateOp::Assign,
    );
    let _ = written.into_data();
    let after = tensor2(&state).into_data().to_vec::<f32>().unwrap();
    assert_eq!(
        bits(&after),
        bits(&before),
        "an eager scatter after the failed capture wrote its table"
    );
}

#[test]
fn a_read_of_the_cache_after_its_write_is_refused_by_name() {
    with_device(Config::default(), |d| {
        refused(&d, |old, _, new| old + new);
    });
}

#[test]
fn a_view_taken_before_the_write_is_refused_by_name() {
    with_device(Config::default(), |d| {
        refused(&d, |_, view, new| {
            view.sum_dim(1) + new.slice([0..LEN, 0..DIM]).sum_dim(1)
        });
    });
}

/// A transpose kept beside the state (`DramRef::plain_for_matmul`, made on its second eager use
/// as a matmul's transposed `B`) is dropped when the capture is given the state: the step's
/// product reads the rows the step wrote.
#[test]
fn a_transpose_kept_from_before_the_capture_is_not_read() {
    with_device(Config::default(), |d| {
        let state = cache(&d);
        let x = Tensor::<TtBackend, 2>::from_data(TensorData::new(values(3, DIM), [1, DIM]), &d);
        for _ in 0..3 {
            let _ = x.clone().matmul(tensor2(&state).transpose()).into_data();
        }
        let mut host = tensor2(&state).into_data().to_vec::<f32>().unwrap();
        let mut set = SharedStateTraces::<usize>::new(vec![state.clone()]).unwrap();
        let pp = position(5, &d);
        let (s, p, xc) = (state.clone(), pp.clone(), x.clone());
        let first = set
            .capture(0, &[&pp], || {
                let at = Tensor::<TtBackend, 1, Int>::from_primitive(p.clone());
                let new = tensor2(&s).scatter_nd(
                    rows(at.clone(), &d),
                    new_rows(at, &d),
                    IndexingUpdateOp::Assign,
                );
                let y = xc.matmul(new.clone().transpose());
                (vec![float2(y)], vec![float2(new)])
            })
            .unwrap();
        // The same product on the card over the host's written cache, uploaded afresh.
        host_step(&mut host, 5);
        let w = Tensor::<TtBackend, 2>::from_data(TensorData::new(host, [HEADS * LEN, DIM]), &d);
        let want = x.matmul(w.transpose()).into_data().to_vec::<f32>().unwrap();
        assert_eq!(
            bits(&payload(&first, 0)),
            bits(&want),
            "the product read a stale transpose"
        );
    });
}

/// A synthetic Llama's traced decode steps write their caches in place, and its generation is
/// the copying generation's, token for token and bit for bit.
#[test]
fn a_decoder_s_steps_write_their_caches_in_place() {
    with_device(Config::default(), |d| {
        let config = LlamaConfig {
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
        let model = Llama::<TtBackend>::new(&config, &d);
        let shapes = Shapes::new(16, &[4, 8]);
        let prompt = [3, 17, 42, 5, 9];
        let mut runs = Vec::new();
        for in_place in [false, true] {
            burn_tt::set_state_in_place(in_place);
            let g = TracedGenerator::<_, TtBackend>::new(&model, &shapes, &d).unwrap();
            let (mut copies, mut index_copies) = (0, 0);
            for (key, ops) in g.trace_ops().unwrap() {
                if key.starts_with("Step") {
                    copies += ops
                        .iter()
                        .filter(|op| op.what == "copy_into list" || op.what == "copy list")
                        .count();
                    // The write's index column, a `[1, n]` buffer read transposed, is read as
                    // the row it is, not copied.
                    index_copies += ops
                        .iter()
                        .filter(|op| op.label == "float_scatter_nd" && op.what == "block copy list")
                        .count();
                }
            }
            assert_eq!(index_copies, 0, "the write's indices copied");
            let generation = g.generate(&prompt, 4).unwrap();
            let logits: Vec<Vec<u32>> = generation.logits.iter().map(|l| bits(l)).collect();
            runs.push((copies, logits, generation.tokens));
        }
        let (copying, in_place) = (&runs[0], &runs[1]);
        // Two caches a layer, each copied by its scatter and written back: four copies a layer.
        assert_eq!(
            copying.0,
            4 * config.layers * shapes.steps.len(),
            "the negative control"
        );
        assert_eq!(in_place.0, 0, "copies left in the decode steps");
        assert_eq!(copying.1, in_place.1, "every step's logits");
        assert_eq!(copying.2, in_place.2, "the generated tokens");
    });
}
