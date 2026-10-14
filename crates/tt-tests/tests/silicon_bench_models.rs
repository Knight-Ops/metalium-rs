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
use tt_models::llama::{Llama, Precision, FULL_LAYERS, HIDDEN};
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

/// `resnet18_forward` as a trace (`burn_tt::Trace`): one eager forward first (the relaid weights
/// and tap indices are made and kept then), one captured, then each timed pass a replay that writes
/// the image and reads the logits back, with no op submitted by the host. Its logits equal the eager
/// forward's bit for bit (the same ops), and its top-1 burn-flex's. Until traces are segmented the
/// capture fits only with the programs spread over many tiles (`TT_TILES=all`); on fewer it is
/// skipped with the reason printed.
#[test]
#[ignore = "a benchmark: cargo xtask bench --filter silicon_bench_models"]
fn resnet18_forward_traced() {
    use burn::tensor::TensorPrimitive;
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
        let input =
            Tensor::<TtBackend, 4>::from_data(TensorData::new(x.clone(), [1, 3, 224, 224]), &d);
        let eager = m
            .forward(input.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let TensorPrimitive::Float(image) = input.clone().into_primitive() else {
            unreachable!()
        };
        let captured = burn_tt::Trace::capture(&image, || {
            let TensorPrimitive::Float(y) = m.forward(input.clone()).into_primitive() else {
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
                    <TtBackend as burn::tensor::backend::Backend>::sync(&d)
                );
                assert!(cause.contains("do not fit"), "capture failed: {e}; {cause}");
                eprintln!(
                    "SKIP resnet18 traced: the forward's programs do not fit one tile's program \
                     cache (run with TT_TILES=all until traces are segmented): {cause}"
                );
                return;
            }
        };
        assert_eq!(first, eager, "the capture's run is the eager forward");
        for _ in 0..WARMUPS {
            trace.run(x.clone()).unwrap();
        }
        let mut ms = Vec::new();
        let mut got = Vec::new();
        for _ in 0..SAMPLES {
            let t0 = Instant::now();
            got = trace.run(x.clone()).unwrap();
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        report(
            "resnet18 224x224 batch 1 forward, traced",
            "ms",
            "host",
            Stats::of(ms),
        );
        assert_eq!(got, eager, "a replay is the eager forward");
        assert_eq!(argmax(&got), argmax(&host));
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
        let (m, _) = Llama::<Flex>::pretrained(1, Precision::F32, &FlexDevice).unwrap();
        let ids = Tensor::<Flex, 2, burn::tensor::Int>::from_data(
            TensorData::new(vec![7042i32], [1, 1]),
            &FlexDevice,
        );
        m.embed(ids).into_data().to_vec::<f32>().unwrap()
    };
    assert_eq!(row.len(), HIDDEN);
    // The host, doing the same decode steps (a fresh cache after sixty, as below).
    {
        let (m, _) = Llama::<Flex>::pretrained(FULL_LAYERS, Precision::F32, &FlexDevice).unwrap();
        let mut cache = m.cache(64, DType::F32, &FlexDevice);
        let mut step = 0usize;
        baseline(
            "smollm2-135m decode step without the embedding lookup",
            || {
                let x = Tensor::<Flex, 3>::from_data(
                    TensorData::new(row.clone(), [1, 1, HIDDEN]),
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
        let (m, _) = Llama::<TtBackend>::pretrained(FULL_LAYERS, Precision::F32, &d).unwrap();
        // A cache holding a five-token prompt's worth of positions would need the prompt run; a
        // decode step at position 0 has the same operations, with a one-position attention.
        let mut cache = m.cache(64, DType::F32, &d);
        let mut step = 0usize;
        let got = measure(
            "smollm2-135m decode step without the embedding lookup",
            || {
                let x = Tensor::<TtBackend, 3>::from_data(
                    TensorData::new(row.clone(), [1, 1, HIDDEN]),
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

#[test]
#[ignore = "a benchmark: cargo xtask bench --filter silicon_bench_models"]
fn smollm2_decode_with_the_embedding_lookup() {
    use burn::tensor::Int;
    // A whole decode step from a token id: the lookup, the thirty layers and the output head.
    let ids = |data: Vec<i32>| TensorData::new(data, [1, 1]);
    {
        let (m, _) = Llama::<Flex>::pretrained(FULL_LAYERS, Precision::F32, &FlexDevice).unwrap();
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
        let (m, _) = Llama::<TtBackend>::pretrained(FULL_LAYERS, Precision::F32, &d).unwrap();
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
