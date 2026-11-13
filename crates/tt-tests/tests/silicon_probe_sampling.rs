//! What choosing a decode step's token on the card costs, against downloading its logits
//! (decode-to-bandwidth Phase 7): a `[1, 49152]` row of logits, traced on every tile, replayed
//! with the logits downloaded or with only the chosen token.
//!
//! `cargo xtask bench --device 0 --filter silicon_probe_sampling`
#![cfg(feature = "silicon")]

use burn::tensor::{Tensor, TensorData, TensorPrimitive};
use burn_tt::{TileChoice, TracedInference, TtBackend};
use tt_tests::burn_device::{with_device, Config};

const SAMPLES: usize = 15;

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 40.0 - 20.0
        })
        .collect()
}

fn native<const D: usize>(t: Tensor<TtBackend, D>) -> burn_tt::TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(t) => t,
        TensorPrimitive::QFloat(_) => unreachable!(),
    }
}

/// The median run of `trace` (write, replay, read) in microseconds.
fn time(trace: &TracedInference, values: &[f32]) -> [f64; 4] {
    let mut runs: Vec<[f64; 4]> = (0..SAMPLES + 2)
        .map(|_| {
            let t = std::time::Instant::now();
            let r = trace
                .run_timed(vec![burn_tt::InputPayload::F32(values.to_vec())])
                .unwrap();
            [
                t.elapsed().as_secs_f64() * 1e6,
                r.write.as_secs_f64() * 1e6,
                r.replay.as_secs_f64() * 1e6,
                r.read.as_secs_f64() * 1e6,
            ]
        })
        .skip(2)
        .collect();
    runs.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
    runs[runs.len() / 2]
}

#[test]
#[ignore = "benchmark"]
fn the_token_on_the_card_against_the_logits_downloaded() {
    let config = Config {
        tiles: Some(TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        let v = 49152;
        let values = floats(1, v);
        let x = Tensor::<TtBackend, 2>::from_data(TensorData::new(values.clone(), [1, v]), &d);
        let _ = (x.clone() * 1.0).sum().into_data();
        let xi = native(x.clone());
        let (logits, _) =
            TracedInference::capture(&[&xi], || vec![native(x.clone() * 1.0)]).unwrap();
        let (token, _) =
            TracedInference::capture(&[&xi], || vec![x.clone().argmax(1).into_primitive()])
                .unwrap();
        for (name, trace) in [("logits", &logits), ("argmax", &token)] {
            let [all, write, replay, read] = time(trace, &values);
            println!(
                "MEASURE sampling [1, {v}]: {name} {all:.1} us (write {write:.1}, replay {replay:.1}, read {read:.1})"
            );
        }
        // The secondary path: temperature, top-k and top-p (`tt_models::generate::choose`), run
        // on the card over logits already there, as `TracedGenerator` samples: its sort's
        // programs do not fit a capture beside a model's traces.
        use tt_models::generate::{choose, Sampler};
        let u = Tensor::<TtBackend, 2>::from_data(TensorData::new(vec![0.37f32], [1, 1]), &d);
        for (name, sampler) in [
            (
                "sample top-k 50, top-p 0.9",
                Sampler::Sample {
                    temperature: 0.8,
                    top_k: Some(50),
                    top_p: Some(0.9),
                    seed: 0,
                },
            ),
            (
                "sample top-p 0.9 alone (the 1024 largest)",
                Sampler::Sample {
                    temperature: 0.8,
                    top_k: None,
                    top_p: Some(0.9),
                    seed: 0,
                },
            ),
        ] {
            let mut runs: Vec<(f64, i32)> = (0..7)
                .map(|_| {
                    let t = std::time::Instant::now();
                    let tok = choose(x.clone(), &sampler, Some(u.clone()))
                        .into_data()
                        .to_vec::<i32>()
                        .unwrap()[0];
                    (t.elapsed().as_secs_f64() * 1e6, tok)
                })
                .collect();
            runs.sort_by(|a, b| a.0.total_cmp(&b.0));
            println!(
                "MEASURE sampling [1, {v}]: {name}: {:.1} us on the card, eager (token {})",
                runs[3].0, runs[3].1
            );
        }
        let want = (0..v)
            .max_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap().then(b.cmp(&a)))
            .unwrap();
        let got = token
            .run(vec![burn_tt::InputPayload::F32(values.clone())])
            .unwrap();
        println!("MEASURE sampling: argmax {got:?}, host first maximum {want}");
    });
}

/// Roadmap W4's diagnosis: where Qwen3.5-4B's ~2.7 ms a greedy part goes. Traced on every tile,
/// each alone: one 31,040-column part's `argmax` on a row of its own; the same part sliced from
/// a 248,320-column row; and `generate::greedy` over the whole row (eight such parts).
#[test]
#[ignore = "benchmark"]
fn a_wide_vocabulary_s_greedy_pick() {
    let config = Config {
        tiles: Some(TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        let (part, vocab) = (31_040usize, 248_320usize);
        let wide = floats(3, vocab);
        let x = Tensor::<TtBackend, 2>::from_data(TensorData::new(wide.clone(), [1, vocab]), &d);
        let _ = (x.clone() * 1.0).sum().into_data();
        let xi = native(x.clone());
        let narrow = wide[..part].to_vec();
        let y = Tensor::<TtBackend, 2>::from_data(TensorData::new(narrow.clone(), [1, part]), &d);
        let yi = native(y.clone());
        let (own, _) =
            TracedInference::capture(&[&yi], || vec![y.clone().argmax(1).into_primitive()])
                .unwrap();
        let (first, _) = TracedInference::capture(&[&xi], || {
            vec![x.clone().slice([0..1, 0..part]).argmax(1).into_primitive()]
        })
        .unwrap();
        let (last, _) = TracedInference::capture(&[&xi], || {
            vec![x
                .clone()
                .slice([0..1, vocab - part..vocab])
                .argmax(1)
                .into_primitive()]
        })
        .unwrap();
        let (slice, _) = TracedInference::capture(&[&xi], || {
            vec![native(x.clone().slice([0..1, vocab - part..vocab]) * 1.0)]
        })
        .unwrap();
        let (greedy, _) = TracedInference::capture(&[&xi], || {
            vec![tt_models::generate::greedy(x.clone()).into_primitive()]
        })
        .unwrap();
        for (name, trace, values) in [
            ("one part, its own row", &own, &narrow),
            ("the first part, sliced", &first, &wide),
            ("the last part, sliced", &last, &wide),
            ("the last part's slice alone, x 1", &slice, &wide),
            ("greedy (one argmax) over the whole row", &greedy, &wide),
        ] {
            let [all, write, replay, read] = time(trace, values);
            println!(
                "MEASURE greedy pick: {name}: {all:.1} us (write {write:.1}, replay {replay:.1}, read {read:.1})"
            );
        }
        let want = (0..vocab)
            .max_by(|&a, &b| wide[a].partial_cmp(&wide[b]).unwrap().then(b.cmp(&a)))
            .unwrap();
        let got = greedy
            .run(vec![burn_tt::InputPayload::F32(wide.clone())])
            .unwrap();
        println!("MEASURE greedy pick: {got:?}, host first maximum {want}");
    });
}
