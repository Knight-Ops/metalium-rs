//! What a forward pass of each model of the ladder costs on a card, to set the order of the
//! performance work (`docs/plans/roadmap.md`, Track 2).
//!
//! Each benchmark loads a pretrained model, runs two warm-ups and nine timed forward passes, and
//! reports the median in milliseconds with its spread, timed from host dispatch through the final
//! readback (the convention of the other silicon benchmarks). One more pass runs under
//! `with_report` and its count of reported device operations is reported with it (an indication of what the host submitted;
//! convolution and pooling submit many more than they report). Outputs are validated outside the timing
//! (the same finite logits and the same top-1 as burn-flex).
//!
//! `smollm2_decode_without_the_embedding_lookup` feeds the model a precomputed embedding row: the
//! lookup itself costs 15 s a token (`probe_embedding_cost`), and the point is to see what the
//! other three thousand operations of a decoded token cost.
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
    let (_, ops) = burn_tt::with_report(&mut forward);
    eprintln!("{ops}");
    // Operations the report lists as run on the device. One reported `conv2d` or `max_pool2d` hides many
    // more submissions (a chunk of 32 positions or windows each), so this undercounts them.
    let jobs: u64 = ops.0.iter().map(|s| s.on_device).sum();
    let calls: u64 = ops.0.iter().map(|s| s.calls).sum();
    report(
        &format!("{key} reported device ops"),
        "ops",
        "count",
        Stats::of([jobs as f64]),
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
