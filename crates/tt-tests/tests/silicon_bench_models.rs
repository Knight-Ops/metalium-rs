//! What a forward pass of each model of the ladder costs on a card, to set the order of the
//! performance work (`docs/plans/roadmap.md`, Track 2).
//!
//! Each benchmark loads a pretrained model, runs two warm-ups and nine timed forward passes, and
//! reports the median in milliseconds with its spread, timed from host dispatch through the final
//! readback (the convention of the other silicon benchmarks). One more pass runs under
//! `with_report`: its reported device operations (views included), the jobs it queued on the server
//! thread, and the server thread's busy time, with a table of that time by the Burn op that queued
//! each job (`burn_tt::server_report`). Outputs are validated outside the timing
//! (the same finite logits and the same top-1 as burn-flex).
//!
//! `smollm2_decode_without_the_embedding_lookup` feeds the model a precomputed embedding row. The
//! lookup cost 15 s a token when it was measured (`probe_embedding_cost`, 2026-10-12: `select`
//! staged the whole table for every element), and the point was to see what the other three
//! thousand operations of a decoded token cost. `smollm2_decode_with_the_embedding_lookup` is the
//! whole step from a token id, which the row select (`step171_select_rows`) made affordable.
//!
//! Silicon only, `#[ignore]`d (run with `cargo xtask bench --filter silicon_bench_models`), and
//! independent of each other, so two can run at once on two cards.

#![cfg(feature = "silicon")]

use std::time::Instant;

use burn::tensor::{DType, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_models::image_input::{load_chw, HALF_MEAN, HALF_STD, IMAGENET_MEAN, IMAGENET_STD};
use tt_models::llama::{Llama, Precision, SMOLLM2_135M, SMOLLM2_1_7B_INSTRUCT};
use tt_models::resnet::ResNet18;
use tt_models::vit::{VitTiny, FULL_DEPTH};
use tt_tests::bench::{git_sha, report, Stats};
use tt_tests::burn_device::{with_device, Config};

const WARMUPS: usize = 2;
const SAMPLES: usize = 9;

fn conditions(card: usize) {
    println!(
        "BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"card\":{card},\"warmups\":{WARMUPS},\"samples\":{SAMPLES},\"timing\":\"host dispatch through final readback; validation outside timing\",\"model_inputs\":\"cats.jpg, pretrained weights, batch 1\"}}",
        git_sha(),
        !cfg!(debug_assertions)
    );
}

fn card() -> usize {
    std::env::var("TT_SILICON_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Time `forward` (milliseconds), with its operation count from one reported pass.
fn measure(key: &str, mut forward: impl FnMut() -> Vec<f32>) -> Vec<f32> {
    for _ in 0..WARMUPS {
        forward();
    }
    let mut last = Vec::new();
    let mut ms = Vec::new();
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        last = forward();
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    report(key, "ms", "host", Stats::of(ms));
    burn_tt::server_report_reset();
    let t0 = Instant::now();
    let (_, ops) = burn_tt::with_report(&mut forward);
    let wall = t0.elapsed();
    eprintln!("{ops}");
    // Operations the report lists as run on the device (a view counts, though it submits nothing).
    let reported: u64 = ops.0.iter().map(|s| s.on_device).sum();
    let calls: u64 = ops.0.iter().map(|s| s.calls).sum();
    // Jobs queued on the server thread, each charged to the innermost Burn op that queued it.
    let jobs: u64 = ops.0.iter().map(|s| s.jobs).sum();
    report(
        &format!("{key} reported device ops"),
        "ops",
        "count",
        Stats::of([reported as f64]),
    );
    report(
        &format!("{key} server jobs"),
        "jobs",
        "count",
        Stats::of([jobs as f64]),
    );
    // Where the server thread's time went, by the op that queued the work. The forward's readback
    // waited for every job it depends on; frees queued after it may still be running.
    let server = burn_tt::server_report();
    let busy: std::time::Duration = server.iter().map(|s| s.time).sum();
    eprintln!(
        "server thread: {:.1} ms busy of a {:.1} ms pass",
        busy.as_secs_f64() * 1e3,
        wall.as_secs_f64() * 1e3
    );
    eprintln!("{:<34} {:>8} {:>10} {:>10}", "op", "jobs", "ms", "us/job");
    for s in &server {
        let ms = s.time.as_secs_f64() * 1e3;
        eprintln!(
            "{:<34} {:>8} {:>10.2} {:>10.1}",
            s.op,
            s.jobs,
            ms,
            ms * 1e3 / s.jobs.max(1) as f64
        );
    }
    report(
        &format!("{key} server thread busy"),
        "ms",
        "host",
        Stats::of([busy.as_secs_f64() * 1e3]),
    );
    report(
        &format!("{key} backend calls"),
        "calls",
        "count",
        Stats::of([calls as f64]),
    );
    last
}

/// What a captured trace is made of (`docs/plans/replay-fixed-cost.md`, P0): its ops, the
/// barriers among them, how many units each op ran on (a unit whose share of the op is only the
/// barrier entry sat it out), and by the Burn op that queued each (`OpRecord::label`). Reported as
/// `BENCH` counts under `key`; the table goes to stderr, and the whole list, one op a line, to the
/// file `TT_TRACE_OPS` names if set.
fn trace_shape(key: &str, trace: &burn_tt::TracedInference) {
    let ops = trace.ops().unwrap();
    let ran = |o: &tt_kernels::trace::OpRecord| {
        let fixed = o.barrier as usize;
        o.entries.iter().filter(|r| r.len() > fixed).count()
    };
    let barriers = ops.iter().filter(|o| o.barrier).count();
    let units = ops.first().map_or(0, |o| o.entries.len());
    let entries: Vec<usize> = (0..units)
        .map(|u| ops.iter().map(|o| o.entries[u].len()).sum())
        .collect();
    report(
        &format!("{key} trace ops"),
        "ops",
        "count",
        Stats::of([ops.len() as f64]),
    );
    report(
        &format!("{key} trace barriers"),
        "barriers",
        "count",
        Stats::of([barriers as f64]),
    );
    report(
        &format!("{key} trace units an op ran on"),
        "units",
        "count",
        Stats::of(ops.iter().map(|o| ran(o) as f64).collect::<Vec<_>>()),
    );
    eprintln!(
        "trace: {} ops, {barriers} barriers, {units} units; stream entries on unit 0 {}, most {}",
        ops.len(),
        entries.first().copied().unwrap_or(0),
        entries.iter().max().copied().unwrap_or(0)
    );
    // By label: ops, the units they ran on, and their entries summed over every unit.
    let mut by: std::collections::BTreeMap<&str, (usize, usize, usize)> = Default::default();
    for o in &ops {
        let r = by.entry(o.label).or_default();
        r.0 += 1;
        r.1 += ran(o);
        r.2 += o.entries.iter().map(|e| e.len()).sum::<usize>();
    }
    let mut rows: Vec<_> = by.into_iter().collect();
    rows.sort_by_key(|&(label, (n, _, _))| (std::cmp::Reverse(n), label));
    eprintln!(
        "{:<34} {:>6} {:>10} {:>10}",
        "label", "ops", "units/op", "entries"
    );
    for (label, (n, units, entries)) in rows {
        eprintln!(
            "{:<34} {:>6} {:>10.1} {:>10}",
            label,
            n,
            units as f64 / n as f64,
            entries
        );
    }
    if let Ok(path) = std::env::var("TT_TRACE_OPS") {
        let mut out = String::from("index\tlabel\twhat\tunits_ran\tunit0_entries\tbarrier\n");
        for (i, o) in ops.iter().enumerate() {
            out += &format!(
                "{i}\t{}\t{}\t{}\t{}\t{}\n",
                o.label,
                o.what,
                ran(o),
                o.entries[0].len(),
                o.barrier
            );
        }
        std::fs::write(&path, out).unwrap();
        eprintln!("trace ops written to {path}");
    }
}

/// Medians of a trace's replays split as `run_timed` measures them (`GenericTraceRun`): the input
/// write, the replay to its end, the output read.
fn report_split(key: &str, runs: &[burn_tt::GenericTraceRun]) {
    let ms = |f: fn(&burn_tt::GenericTraceRun) -> std::time::Duration| {
        Stats::of(
            runs.iter()
                .map(|r| f(r).as_secs_f64() * 1e3)
                .collect::<Vec<_>>(),
        )
    };
    report(
        &format!("{key}, input write"),
        "ms",
        "host",
        ms(|r| r.write),
    );
    report(&format!("{key}, replay"), "ms", "host", ms(|r| r.replay));
    report(&format!("{key}, output read"), "ms", "host", ms(|r| r.read));
}

/// The same forward pass on burn-flex (the host), timed the same way: the baseline a device number
/// is held against.
fn baseline(key: &str, mut forward: impl FnMut() -> Vec<f32>) -> Vec<f32> {
    for _ in 0..WARMUPS {
        forward();
    }
    let mut last = Vec::new();
    let mut ms = Vec::new();
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        last = forward();
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    report(
        &format!("{key} (burn-flex, host)"),
        "ms",
        "host",
        Stats::of(ms),
    );
    last
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

#[test]
#[ignore = "a benchmark: cargo xtask bench --filter silicon_bench_models"]
fn resnet18_forward() {
    let x = load_chw("cats.jpg", 224, IMAGENET_MEAN, IMAGENET_STD).unwrap();
    let host = {
        let (m, _) = ResNet18::<Flex>::pretrained(&FlexDevice).unwrap();
        baseline("resnet18 224x224 batch 1 forward", || {
            let t = Tensor::<Flex, 4>::from_data(
                TensorData::new(x.clone(), [1, 3, 224, 224]),
                &FlexDevice,
            );
            m.forward(t).into_data().to_vec::<f32>().unwrap()
        })
    };
    with_device(Config::default(), |d| {
        conditions(card());
        let (m, _) = ResNet18::<TtBackend>::pretrained(&d).unwrap();
        let got = measure("resnet18 224x224 batch 1 forward", || {
            let t =
                Tensor::<TtBackend, 4>::from_data(TensorData::new(x.clone(), [1, 3, 224, 224]), &d);
            m.forward(t).into_data().to_vec::<f32>().unwrap()
        });
        assert_eq!(argmax(&got), argmax(&host));
    });
}

/// `forward` on a `[1, 3, 224, 224]` image as a trace (`burn_tt::Trace`): one eager forward first (what
/// a model makes and keeps on first use -- relaid weights, tap indices, a causal mask -- is made
/// then), one captured, then each timed pass a replay that writes the image and reads the logits
/// back, with no op submitted by the host. Its logits equal the eager forward's bit for bit (the same
/// ops), and its top-1 burn-flex's (`host`). Until traces are segmented the capture fits only where
/// the programs are spread over enough tiles; otherwise it is skipped with the reason printed.
fn traced_image_forward(
    key: &str,
    d: &burn_tt::TtDevice,
    x: &[f32],
    host: &[f32],
    forward: impl Fn(Tensor<TtBackend, 4>) -> Tensor<TtBackend, 2>,
) {
    use burn::tensor::TensorPrimitive;
    let input = Tensor::<TtBackend, 4>::from_data(TensorData::new(x.to_vec(), [1, 3, 224, 224]), d);
    let eager = forward(input.clone()).into_data().to_vec::<f32>().unwrap();
    let TensorPrimitive::Float(image) = input.clone().into_primitive() else {
        unreachable!()
    };
    let captured = burn_tt::Trace::capture(&image, || {
        let TensorPrimitive::Float(y) = forward(input.clone()).into_primitive() else {
            unreachable!()
        };
        y
    });
    let (trace, first) = match captured {
        Ok(t) => t,
        Err(e) => {
            // A forward's programs fit the program cache only when spread over enough tiles
            // (every tile with the default spread does; one tile does not) until traces are
            // segmented (roadmap Track 2, "Trace segmentation"): that is reported, not timed.
            let cause = format!(
                "{:?}",
                <TtBackend as burn::tensor::backend::Backend>::sync(d)
            );
            assert!(cause.contains("do not fit"), "capture failed: {e}; {cause}");
            eprintln!(
                "SKIP {key}: the forward's programs do not fit one tile's program cache (run \
                 with TT_TILES=all until traces are segmented): {cause}"
            );
            return;
        }
    };
    assert_eq!(first, eager, "the capture's run is the eager forward");
    for _ in 0..WARMUPS {
        trace.run(x.to_vec()).unwrap();
    }
    let mut ms = Vec::new();
    let mut got = Vec::new();
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        got = trace.run(x.to_vec()).unwrap();
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    report(key, "ms", "host", Stats::of(ms));
    assert_eq!(got, eager, "a replay is the eager forward");
    assert_eq!(argmax(&got), argmax(host));
}

/// `resnet18_forward` as a trace ([`traced_image_forward`]).
#[test]
#[ignore = "a benchmark: cargo xtask bench --filter silicon_bench_models"]
fn resnet18_forward_traced() {
    let x = load_chw("cats.jpg", 224, IMAGENET_MEAN, IMAGENET_STD).unwrap();
    let host = {
        let (m, _) = ResNet18::<Flex>::pretrained(&FlexDevice).unwrap();
        baseline("resnet18 224x224 batch 1 forward, traced", || {
            let t = Tensor::<Flex, 4>::from_data(
                TensorData::new(x.clone(), [1, 3, 224, 224]),
                &FlexDevice,
            );
            m.forward(t).into_data().to_vec::<f32>().unwrap()
        })
    };
    with_device(Config::default(), |d| {
        conditions(card());
        let (m, _) = ResNet18::<TtBackend>::pretrained(&d).unwrap();
        traced_image_forward(
            "resnet18 224x224 batch 1 forward, traced",
            &d,
            &x,
            &host,
            |t| m.forward(t),
        );
    });
}

/// `vit_tiny_forward` as a trace ([`traced_image_forward`]).
#[test]
#[ignore = "a benchmark: cargo xtask bench --filter silicon_bench_models"]
fn vit_tiny_forward_traced() {
    let x = load_chw("cats.jpg", 224, HALF_MEAN, HALF_STD).unwrap();
    let host = {
        let (m, _) = VitTiny::<Flex>::pretrained(FULL_DEPTH, &FlexDevice).unwrap();
        baseline("vit-tiny 224x224 batch 1 forward, traced", || {
            let t = Tensor::<Flex, 4>::from_data(
                TensorData::new(x.clone(), [1, 3, 224, 224]),
                &FlexDevice,
            );
            m.forward(t).into_data().to_vec::<f32>().unwrap()
        })
    };
    with_device(Config::default(), |d| {
        conditions(card());
        let (m, _) = VitTiny::<TtBackend>::pretrained(FULL_DEPTH, &d).unwrap();
        traced_image_forward(
            "vit-tiny 224x224 batch 1 forward, traced",
            &d,
            &x,
            &host,
            |t| m.forward(t),
        );
    });
}

#[test]
#[ignore = "a benchmark: cargo xtask bench --filter silicon_bench_models"]
fn vit_tiny_forward() {
    let x = load_chw("cats.jpg", 224, HALF_MEAN, HALF_STD).unwrap();
    let host = {
        let (m, _) = VitTiny::<Flex>::pretrained(FULL_DEPTH, &FlexDevice).unwrap();
        baseline("vit-tiny 224x224 batch 1 forward", || {
            let t = Tensor::<Flex, 4>::from_data(
                TensorData::new(x.clone(), [1, 3, 224, 224]),
                &FlexDevice,
            );
            m.forward(t).into_data().to_vec::<f32>().unwrap()
        })
    };
    with_device(Config::default(), |d| {
        conditions(card());
        let (m, _) = VitTiny::<TtBackend>::pretrained(FULL_DEPTH, &d).unwrap();
        let got = measure("vit-tiny 224x224 batch 1 forward", || {
            let t =
                Tensor::<TtBackend, 4>::from_data(TensorData::new(x.clone(), [1, 3, 224, 224]), &d);
            m.forward(t).into_data().to_vec::<f32>().unwrap()
        });
        assert_eq!(argmax(&got), argmax(&host));
    });
}

#[test]
#[ignore = "a benchmark: cargo xtask bench --filter silicon_bench_models"]
fn smollm2_decode_without_the_embedding_lookup() {
    // The embedding of " Paris" (token 7042), read from the host model: the input of a decode step.
    let row: Vec<f32> = {
        let (m, _) = Llama::<Flex>::smollm2_135m(1, Precision::F32, &FlexDevice).unwrap();
        let ids = Tensor::<Flex, 2, burn::tensor::Int>::from_data(
            TensorData::new(vec![7042i32], [1, 1]),
            &FlexDevice,
        );
        m.embed(ids).into_data().to_vec::<f32>().unwrap()
    };
    assert_eq!(row.len(), SMOLLM2_135M.hidden);
    // The host, doing the same decode steps (a fresh cache after sixty, as below).
    {
        let (m, _) =
            Llama::<Flex>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &FlexDevice).unwrap();
        let mut cache = m.cache(64, DType::F32, &FlexDevice);
        let mut step = 0usize;
        baseline(
            "smollm2-135m decode step without the embedding lookup",
            || {
                let x = Tensor::<Flex, 3>::from_data(
                    TensorData::new(row.clone(), [1, 1, SMOLLM2_135M.hidden]),
                    &FlexDevice,
                );
                step += 1;
                if step > 60 {
                    cache = m.cache(64, DType::F32, &FlexDevice);
                    step = 1;
                }
                m.forward_embedded(x, &mut cache, DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
            },
        );
    }
    with_device(Config::default(), |d| {
        conditions(card());
        let (m, _) =
            Llama::<TtBackend>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &d).unwrap();
        // A cache holding a five-token prompt's worth of positions would need the prompt run; a
        // decode step at position 0 has the same operations, with a one-position attention.
        let mut cache = m.cache(64, DType::F32, &d);
        let mut step = 0usize;
        let got = measure(
            "smollm2-135m decode step without the embedding lookup",
            || {
                let x = Tensor::<TtBackend, 3>::from_data(
                    TensorData::new(row.clone(), [1, 1, SMOLLM2_135M.hidden]),
                    &d,
                );
                step += 1;
                if step > 60 {
                    cache = m.cache(64, DType::F32, &d);
                    step = 1;
                }
                m.forward_embedded(x, &mut cache, DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
            },
        );
        assert!(got.iter().all(|v| v.is_finite()));
    });
}

/// `smollm2_decode_without_the_embedding_lookup` with fixed shapes (`Llama::step_fixed`: the position
/// a tensor, attention over a 64-position cache with the later positions masked), first eagerly,
/// then as a trace: one step captured at position 1 with the cache as state written back in place
/// (`TracedInference::capture_with_state`), each timed pass a replay that writes the next
/// position's embedding and position and reads the logits back, with no op submitted by the host.
/// Every replay's logits equal the eager fixed step's at that position bit for bit. Skipped, with
/// the reason, where the step's programs do not fit the program cache.
#[test]
#[ignore = "a benchmark: cargo xtask bench --filter silicon_bench_models"]
fn smollm2_decode_traced() {
    use burn::tensor::{Int, TensorPrimitive};
    use burn_tt::{InputPayload, OutputPayload, TracedInference, TtTensor};
    const MAX_LEN: usize = 64;
    let row: Vec<f32> = {
        let (m, _) = Llama::<Flex>::smollm2_135m(1, Precision::F32, &FlexDevice).unwrap();
        let ids = Tensor::<Flex, 2, burn::tensor::Int>::from_data(
            TensorData::new(vec![7042i32], [1, 1]),
            &FlexDevice,
        );
        m.embed(ids).into_data().to_vec::<f32>().unwrap()
    };
    {
        let (m, _) =
            Llama::<Flex>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &FlexDevice).unwrap();
        let mut cache = m.cache(MAX_LEN, DType::F32, &FlexDevice);
        baseline("smollm2-135m decode step, traced", || {
            let x = Tensor::<Flex, 3>::from_data(
                TensorData::new(row.clone(), [1, 1, SMOLLM2_135M.hidden]),
                &FlexDevice,
            );
            m.forward_embedded(x, &mut cache, DType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        });
    }
    with_device(Config::default(), |d| {
        conditions(card());
        let dtype = DType::F32;
        let (m, _) =
            Llama::<TtBackend>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &d).unwrap();
        let x = || {
            Tensor::<TtBackend, 3>::from_data(
                TensorData::new(row.clone(), [1, 1, SMOLLM2_135M.hidden]),
                &d,
            )
        };
        let at = |p: usize| {
            Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(vec![p as i32], [1]), &d)
        };
        let positions = 2 + WARMUPS + SAMPLES;
        // Eager fixed steps: the reference logits at every position, and their cost.
        let mut cache = m.fixed_cache(MAX_LEN, dtype, &d);
        let mut eager = Vec::new();
        let mut ms = Vec::new();
        for p in 0..positions {
            let t0 = Instant::now();
            eager.push(
                m.step_fixed(x(), at(p), &mut cache, dtype)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            );
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        report(
            "smollm2-135m decode step, fixed shapes, eager",
            "ms",
            "host",
            Stats::of(ms[2 + WARMUPS..].to_vec()),
        );
        // The trace: an eager step at 0 makes the cache resident, the step at 1 is captured.
        let mut cache = m.fixed_cache(MAX_LEN, dtype, &d);
        let _ = m.step_fixed(x(), at(0), &mut cache, dtype).into_data();
        let (xin, pin) = (x(), at(1));
        let TensorPrimitive::Float(xp) = xin.clone().into_primitive() else {
            unreachable!()
        };
        xp.ensure_resident();
        let pp = pin.clone().into_primitive();
        pp.ensure_resident();
        let float = |t: Tensor<TtBackend, 4>| match t.into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!(),
        };
        let old: Vec<TtTensor> = cache.state().into_iter().map(float).collect();
        let captured = TracedInference::capture_with_state(&[&xp, &pp], || {
            let TensorPrimitive::Float(y) = m
                .step_fixed(xin.clone(), pin.clone(), &mut cache, dtype)
                .into_primitive()
            else {
                unreachable!()
            };
            let new = cache.state().into_iter().map(float);
            (vec![y], new.zip(old.iter().cloned()).collect())
        });
        let (trace, first) = match captured {
            Ok(t) => t,
            Err(e) => {
                let cause = format!(
                    "{:?}",
                    <TtBackend as burn::tensor::backend::Backend>::sync(&d)
                );
                assert!(cause.contains("do not fit"), "capture failed: {e}; {cause}");
                eprintln!(
                    "SKIP smollm2-135m decode step, traced: the step's programs do not fit one \
                     tile's program cache (run with TT_TILES=all until traces are segmented): {cause}"
                );
                return;
            }
        };
        let logits = |o: &[OutputPayload]| match &o[0] {
            OutputPayload::F32(v) => v.clone(),
            other => panic!("logits as {other:?}"),
        };
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&logits(&first)), bits(&eager[1]), "the capture's run");
        trace_shape("smollm2-135m decode step, traced", &trace);
        let mut ms = Vec::new();
        let mut split = Vec::new();
        for (p, want) in eager.iter().enumerate().skip(2) {
            let t0 = Instant::now();
            let got = trace
                .run_timed(vec![
                    InputPayload::F32(row.clone()),
                    InputPayload::Bits(vec![p as u32]),
                ])
                .unwrap();
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
            assert_eq!(
                bits(&logits(&got.outputs)),
                bits(want),
                "replay at position {p}"
            );
            split.push(got);
        }
        report(
            "smollm2-135m decode step, traced",
            "ms",
            "host",
            Stats::of(ms[WARMUPS..].to_vec()),
        );
        report_split("smollm2-135m decode step, traced", &split[WARMUPS..]);
    });
}

/// The chunk size of a traced prefill (`Llama::step_chunk`: `c` prompt tokens from a start position,
/// both tensors): for each `c` of `PREFILL_CHUNKS`, one chunk captured on a fresh 512-position cache
/// with the cache as state, then replays timed. A prompt of `len` tokens costs `ceil(len / c)`
/// replays, so the replay's time and its time a token decide the chunk. One replay at the second
/// chunk's start is checked bit for bit against the eager chunk there. A chunk whose programs do not
/// fit the program cache is reported and skipped.
#[test]
#[ignore = "a benchmark: cargo xtask bench --filter silicon_bench_models"]
fn smollm2_prefill_chunk_sweep() {
    use burn::tensor::{Int, TensorPrimitive};
    use burn_tt::{InputPayload, OutputPayload, TracedInference, TtTensor};
    const MAX_LEN: usize = 512;
    const PREFILL_CHUNKS: [usize; 5] = [1, 32, 64, 128, 256];
    let row: Vec<f32> = {
        let (m, _) = Llama::<Flex>::smollm2_135m(1, Precision::F32, &FlexDevice).unwrap();
        let ids = Tensor::<Flex, 2, burn::tensor::Int>::from_data(
            TensorData::new(vec![7042i32], [1, 1]),
            &FlexDevice,
        );
        m.embed(ids).into_data().to_vec::<f32>().unwrap()
    };
    // burn-flex's whole prefill of a prompt of each length, for the cost a token.
    {
        let (m, _) =
            Llama::<Flex>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &FlexDevice).unwrap();
        for len in [32, 128, 256] {
            let values: Vec<f32> = (0..len).flat_map(|_| row.iter().copied()).collect();
            baseline(&format!("smollm2-135m prefill of {len} tokens"), || {
                let mut cache = m.cache(len, DType::F32, &FlexDevice);
                let x = Tensor::<Flex, 3>::from_data(
                    TensorData::new(values.clone(), [1, len, SMOLLM2_135M.hidden]),
                    &FlexDevice,
                );
                m.forward_embedded(x, &mut cache, DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
            });
        }
    }
    with_device(Config::default(), |d| {
        conditions(card());
        let dtype = DType::F32;
        let (m, _) =
            Llama::<TtBackend>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &d).unwrap();
        let at = |p: usize| {
            Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(vec![p as i32], [1]), &d)
        };
        let float = |t: Tensor<TtBackend, 4>| match t.into_primitive() {
            TensorPrimitive::Float(t) => t,
            _ => unreachable!(),
        };
        let logits = |o: &[OutputPayload]| match &o[0] {
            OutputPayload::F32(v) => v.clone(),
            other => panic!("logits as {other:?}"),
        };
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for c in PREFILL_CHUNKS {
            let values: Vec<f32> = (0..c).flat_map(|_| row.iter().copied()).collect();
            let x = || {
                Tensor::<TtBackend, 3>::from_data(
                    TensorData::new(values.clone(), [1, c, SMOLLM2_135M.hidden]),
                    &d,
                )
            };
            // The eager chunks at 0 and `c`: the reference for the replay at `c`.
            let want = {
                let mut cache = m.fixed_cache(MAX_LEN, dtype, &d);
                let _ = m
                    .step_chunk(x(), at(0), at(c - 1), &mut cache, dtype)
                    .into_data();
                m.step_chunk(x(), at(c), at(c - 1), &mut cache, dtype)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap()
            };
            let mut cache = m.fixed_cache(MAX_LEN, dtype, &d);
            for t in cache.state() {
                float(t).ensure_resident();
            }
            let (xin, sin, lin) = (x(), at(0), at(c - 1));
            let TensorPrimitive::Float(xp) = xin.clone().into_primitive() else {
                unreachable!()
            };
            xp.ensure_resident();
            let (sp, lp) = (sin.clone().into_primitive(), lin.clone().into_primitive());
            sp.ensure_resident();
            lp.ensure_resident();
            let old: Vec<TtTensor> = cache.state().into_iter().map(float).collect();
            let captured = TracedInference::capture_with_state(&[&xp, &sp, &lp], || {
                let TensorPrimitive::Float(y) = m
                    .step_chunk(xin.clone(), sin.clone(), lin.clone(), &mut cache, dtype)
                    .into_primitive()
                else {
                    unreachable!()
                };
                let new = cache.state().into_iter().map(float);
                (vec![y], new.zip(old.iter().cloned()).collect())
            });
            let (trace, _) = match captured {
                Ok(t) => t,
                Err(e) => {
                    let cause = format!(
                        "{:?}",
                        <TtBackend as burn::tensor::backend::Backend>::sync(&d)
                    );
                    assert!(cause.contains("do not fit"), "capture failed: {e}; {cause}");
                    // The attachment's error is sticky, and a larger chunk has more programs.
                    eprintln!(
                        "SKIP smollm2 prefill chunks of {c} and up: do not fit the program \
                         cache: {cause}"
                    );
                    break;
                }
            };
            let run = |start: usize| {
                trace
                    .run_timed(vec![
                        InputPayload::F32(values.clone()),
                        InputPayload::Bits(vec![start as u32]),
                        InputPayload::Bits(vec![c as u32 - 1]),
                    ])
                    .unwrap()
            };
            assert_eq!(
                bits(&logits(&run(c).outputs)),
                bits(&want),
                "chunk {c}: the replay at {c}"
            );
            trace_shape(
                &format!("smollm2-135m prefill chunk of {c}, traced"),
                &trace,
            );
            for _ in 0..WARMUPS {
                run(0);
            }
            let mut ms = Vec::new();
            let mut split = Vec::new();
            for _ in 0..SAMPLES {
                let t0 = Instant::now();
                split.push(run(0));
                ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            report_split(
                &format!("smollm2-135m prefill chunk of {c}, traced"),
                &split,
            );
            let per_token: Vec<f64> = ms.iter().map(|t| t / c as f64).collect();
            report(
                &format!("smollm2-135m prefill chunk of {c}, traced replay"),
                "ms",
                "host",
                Stats::of(ms),
            );
            report(
                &format!("smollm2-135m prefill chunk of {c}, traced, a token"),
                "ms",
                "host",
                Stats::of(per_token),
            );
            drop(trace);
        }
    });
}

#[test]
#[ignore = "a benchmark: cargo xtask bench --filter silicon_bench_models"]
fn smollm2_decode_with_the_embedding_lookup() {
    use burn::tensor::Int;
    // A whole decode step from a token id: the lookup, the thirty layers and the output head.
    let ids = |data: Vec<i32>| TensorData::new(data, [1, 1]);
    {
        let (m, _) =
            Llama::<Flex>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &FlexDevice).unwrap();
        let mut cache = m.cache(64, DType::F32, &FlexDevice);
        let mut step = 0usize;
        baseline("smollm2-135m decode step from a token id", || {
            let t = Tensor::<Flex, 2, Int>::from_data(ids(vec![7042]), &FlexDevice);
            step += 1;
            if step > 60 {
                cache = m.cache(64, DType::F32, &FlexDevice);
                step = 1;
            }
            m.forward(t, &mut cache, DType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        });
    }
    with_device(Config::default(), |d| {
        conditions(card());
        let (m, _) =
            Llama::<TtBackend>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &d).unwrap();
        let mut cache = m.cache(64, DType::F32, &d);
        let mut step = 0usize;
        let got = measure("smollm2-135m decode step from a token id", || {
            let t = Tensor::<TtBackend, 2, Int>::from_data(ids(vec![7042]), &d);
            step += 1;
            if step > 60 {
                cache = m.cache(64, DType::F32, &d);
                step = 1;
            }
            m.forward(t, &mut cache, DType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        });
        assert!(got.iter().all(|v| v.is_finite()));
    });
}

/// Traced generation end to end (`tt_models::generate::TracedGenerator`, gate
/// `step187_traced_generation`): SmolLM2, 30 layers, a 512-position cache. Token ids go in and
/// logits come out; the lookup runs inside the trace and the cache stays on the card.
///
/// Two things are measured:
/// 1. **What the traces cost together:** the step and the chunks (`TT_GEN_BUCKETS`, default
///    `64,128,256`) over a cache of `TT_GEN_MAX_LEN` positions (default 512), captured in one
///    attachment. The report gives the program-cache bytes held on the fullest unit and the trace
///    streams' GDDR bytes. A capture that does not fit fails its attachment for good, so the bench
///    then stops and names the sizes.
/// 2. **A generation's time:** prompts of 32, 128 and 256 tokens, each followed by `NEW` tokens.
///    Reported are:
///    - the driver's build time (the eager warm-up and the captures);
///    - the time to the first token (the prefill and its logits read);
///    - the median decode step;
///    - the whole generation.
///
///    burn-flex's eager growing-cache generation of the same prompts is timed in the same run.
///
/// Validated outside the timing:
/// - a 32-token prompt's first 4 tokens, bit for bit against the eager fixed-shape schedule on
///   the card (`generate_fixed`);
/// - every timed run produces the same tokens.
#[test]
#[ignore = "a benchmark: cargo xtask bench --filter silicon_bench_models"]
fn traced_generation() {
    use tt_models::generate::{generate_fixed, Generation, Shapes, TracedGenerator};
    const PROMPTS: [usize; 3] = [32, 128, 256];
    let max_len: usize = std::env::var("TT_GEN_MAX_LEN")
        .map(|v| v.parse().unwrap())
        .unwrap_or(512);
    // The model (`TT_GEN_MODEL`): its preset and the weights `cargo xtask fetch-weights` puts in
    // place. burn-flex's baseline takes `TT_GEN_FLEX_SAMPLES` samples (SmolLM2-1.7B on the host
    // takes about a minute a sample), and each generation `TT_GEN_NEW` tokens.
    let name = std::env::var("TT_GEN_MODEL").unwrap_or_else(|_| "smollm2-135m".into());
    let config = match name.as_str() {
        "smollm2-135m" => SMOLLM2_135M,
        "smollm2-1.7b-instruct" => SMOLLM2_1_7B_INSTRUCT,
        other => panic!("TT_GEN_MODEL={other}: smollm2-135m or smollm2-1.7b-instruct"),
    };
    let env = |key: &str, default: usize| -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let flex_samples = env("TT_GEN_FLEX_SAMPLES", SAMPLES);
    let new = env("TT_GEN_NEW", 64);
    let vocab = config.vocab;
    // Tokens a second, from a median time.
    let median = |v: &[f64]| {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let prompt = |len: usize| -> Vec<i32> {
        (0..len)
            .map(|i| ((7 + 7919 * i + 104_729 * i * i) % vocab) as i32)
            .collect()
    };
    let decode_ms = |g: &Generation| -> f64 {
        let mut d: Vec<f64> = g.times[1..].iter().map(|t| t.as_secs_f64() * 1e3).collect();
        d.sort_by(f64::total_cmp);
        d[d.len() / 2]
    };
    let ms = |t: std::time::Duration| t.as_secs_f64() * 1e3;
    // burn-flex, eager, growing cache.
    {
        let (m, _) =
            Llama::<Flex>::pretrained(&config, &name, Precision::F32, &FlexDevice).unwrap();
        for len in PROMPTS {
            let p = prompt(len);
            let (mut first, mut step, mut total) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..WARMUPS.min(flex_samples) + flex_samples {
                let t0 = Instant::now();
                let g = m.generate(&p, new, Precision::F32, &FlexDevice);
                let all = t0.elapsed();
                if i >= WARMUPS.min(flex_samples) {
                    first.push(ms(g.times[0]));
                    step.push(decode_ms(&g));
                    total.push(ms(all));
                }
            }
            let key = format!("{name} generation, {len}-token prompt, {new} tokens");
            report(
                &format!("{key}, prefill (burn-flex, host)"),
                "tokens/s",
                "host",
                Stats::of([len as f64 * 1e3 / median(&first)]),
            );
            report(
                &format!("{key}, decode (burn-flex, host)"),
                "tokens/s",
                "host",
                Stats::of([1e3 / median(&step)]),
            );
            report(
                &format!("{key}, first token (burn-flex, host)"),
                "ms",
                "host",
                Stats::of(first),
            );
            report(
                &format!("{key}, decode step (burn-flex, host)"),
                "ms",
                "host",
                Stats::of(step),
            );
            report(
                &format!("{key}, total (burn-flex, host)"),
                "ms",
                "host",
                Stats::of(total),
            );
        }
    }
    let buckets: Vec<usize> = std::env::var("TT_GEN_BUCKETS")
        .map(|v| v.split(',').map(|b| b.trim().parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![64, 128, 256]);
    with_device(Config::default(), |d| {
        conditions(card());
        let (m, _) = Llama::<TtBackend>::pretrained(&config, &name, Precision::F32, &d).unwrap();
        // What a decode step reads at least: every F32 parameter once (counted before fusing, whose
        // views would count the fused matrices twice).
        let weight_bytes = burn::module::Module::num_params(&m) as f64 * 4.0;
        // `q|k|v` and `gate|up` one product each (`Llama::fused`, bit for bit the unfused model;
        // `TT_GEN_FUSED=0` keeps them apart).
        let fused = std::env::var("TT_GEN_FUSED").map_or(true, |v| v != "0");
        println!("BENCH {{\"kind\":\"conditions\",\"fused_projections\":{fused}}}");
        let m = if fused { m.fused() } else { m };
        let t0 = Instant::now();
        // Decode steps attending to 64, 128, ... positions (`TT_GEN_STEPS`, "64,128,256"; empty
        // for the whole cache at every step).
        let mut shapes = Shapes::new(max_len, &buckets);
        if let Ok(v) = std::env::var("TT_GEN_STEPS") {
            let steps: Vec<usize> = v
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse().unwrap())
                .collect();
            shapes = shapes.with_steps(&steps);
        }
        let g = match TracedGenerator::new(&m, &shapes, &d) {
            Ok(g) => g,
            Err(e) => {
                // A capture that does not fit fails the attachment for good: rerun with fewer.
                eprintln!("SKIP chunks {buckets:?} and the step do not capture together: {e} (set TT_GEN_BUCKETS)");
                return;
            }
        };
        let traces = format!(
            "chunks {buckets:?}, steps {:?}, {max_len}-position cache",
            shapes.steps
        );
        report(
            &format!("{name} traced generation, build ({traces})"),
            "ms",
            "host",
            Stats::of([ms(t0.elapsed())]),
        );
        // What the traces hold together: the program cache bounds which chunk sizes fit.
        for set in g.footprint().unwrap() {
            let (held, size) = set.cache_held_bytes();
            eprintln!("{traces}: program cache {held} of {size} bytes on the fullest unit");
            for (key, units) in &set.traces {
                let most = units
                    .iter()
                    .flatten()
                    .map(|u| u.program_bytes)
                    .max()
                    .unwrap_or(0);
                let gddr: u64 = units.iter().flatten().map(|u| u.gddr_bytes).sum();
                // What the streams hold: every entry is 32 bytes (`chunked`'s padding included), and
                // each `PAIR_CALL` names a reader and a writer section, each stored on its own.
                let real: u64 = units
                    .iter()
                    .flatten()
                    .map(|u| u.stream.entries as u64 * 32)
                    .sum();
                let sections: u64 = units
                    .iter()
                    .flatten()
                    .map(|u| {
                        2 * *u.stream.heads.get(&tt_isa::dm::op::PAIR_CALL).unwrap_or(&0) as u64
                    })
                    .sum();
                let unit_traces = units.iter().flatten().count();
                eprintln!(
                    "  {key}: at most {most} program bytes a unit, {gddr} stream bytes in all, \
                     {real} bytes of entries ({:.1}x), {sections} sections over {unit_traces} units \
                     ({:.0} bytes a section allocated)",
                    gddr as f64 / real.max(1) as f64,
                    gddr as f64 / (sections + unit_traces as u64).max(1) as f64,
                );
                report(
                    &format!("{name} traced generation, {key} entry bytes ({traces})"),
                    "bytes",
                    "count",
                    Stats::of([real as f64]),
                );
                report(
                    &format!("{name} traced generation, {key} stream bytes ({traces})"),
                    "bytes",
                    "count",
                    Stats::of([gddr as f64]),
                );
            }
            report(
                &format!("{name} traced generation, program cache held ({traces})"),
                "bytes",
                "count",
                Stats::of([held as f64]),
            );
            report(
                &format!("{name} traced generation, trace streams ({traces})"),
                "bytes",
                "count",
                Stats::of([set.gddr_bytes() as f64]),
            );
        }
        // Validation: the eager schedule on the card, bit for bit.
        let p = prompt(PROMPTS[0]);
        let want = generate_fixed(&m, &p, 4, &shapes, &d).unwrap();
        let got = g.generate(&p, 4).unwrap();
        assert_eq!(got.tokens, want.tokens, "tokens against the eager schedule");
        for (i, (a, b)) in got.logits.iter().zip(&want.logits).enumerate() {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "logits {i} against the eager schedule"
            );
        }
        for len in PROMPTS {
            let p = prompt(len);
            let (mut first, mut step, mut total) = (Vec::new(), Vec::new(), Vec::new());
            let mut tokens = None;
            for i in 0..WARMUPS + SAMPLES {
                let t0 = Instant::now();
                let out = g.generate(&p, new).unwrap();
                let all = t0.elapsed();
                match &tokens {
                    None => tokens = Some(out.tokens.clone()),
                    Some(t) => assert_eq!(&out.tokens, t, "a run's tokens"),
                }
                if i >= WARMUPS {
                    first.push(ms(out.times[0]));
                    step.push(decode_ms(&out));
                    total.push(ms(all));
                }
            }
            let key = format!("{name} generation, {len}-token prompt, {new} tokens ({traces})");
            let step_ms = median(&step);
            report(
                &format!("{key}, prefill, traced"),
                "tokens/s",
                "host",
                Stats::of([len as f64 * 1e3 / median(&first)]),
            );
            report(
                &format!("{key}, decode, traced"),
                "tokens/s",
                "host",
                Stats::of([1e3 / step_ms]),
            );
            // The F32 weights read a decode step, over its time: how near decode is to GDDR's bound.
            report(
                &format!("{key}, decode weight bandwidth, traced"),
                "GB/s",
                "host",
                Stats::of([weight_bytes / step_ms / 1e6]),
            );
            report(
                &format!("{key}, first token, traced"),
                "ms",
                "host",
                Stats::of(first),
            );
            report(
                &format!("{key}, decode step, traced"),
                "ms",
                "host",
                Stats::of(step),
            );
            report(
                &format!("{key}, total, traced"),
                "ms",
                "host",
                Stats::of(total),
            );
        }
    });
}
