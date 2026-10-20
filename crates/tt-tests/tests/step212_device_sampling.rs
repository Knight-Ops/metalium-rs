//! A decode step's next token chosen on the card (`tt_models::generate::choose`,
//! `TracedGenerator::generate_tokens`; decode-to-bandwidth Phase 7): each step and chunk trace
//! carries its token as a second output, and only the token is read back, not the `[1, vocab]`
//! logits (~0.45 ms a 1.7B step on card 1).
//!
//! The claims:
//! - **Greedy, the token path is the logits path's:** `generate_tokens` equals `generate` (the
//!   host's first maximum over the downloaded logits) on three synthetic Llamas and every prompt
//!   layout the schedule feeds.
//! - **Sampling is burn-flex's choice:** temperature, top-k and top-p, drawn by the same uniform
//!   `u`, choose the same token on the card as on burn-flex for every draw that is not within
//!   1e-5 of a boundary of burn-flex's cumulative distribution (where two correct roundings may
//!   fall either side). Top-k of one is the greedy token, and every draw is one of the `k`
//!   largest.
//! - **A traced sampled generation is the same choices run eagerly** on the card, draw for draw.
//!
//! Negative control: other draws give other tokens.

mod ladder_support;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{TtBackend, TtDevice};
use ladder_support::prompt;
use tt_models::generate::{choose, Draws, FixedShapeDecoder, Sampler, Shapes, TracedGenerator};
use tt_models::llama::{Llama, LlamaConfig};
use tt_tests::burn_device::{with_device, Config};

const MAX_LEN: usize = 16;
const BUCKETS: [usize; 2] = [4, 8];
const LENGTHS: [usize; 5] = [1, 3, 4, 8, 9];
const STEPS: usize = 4;

fn shapes() -> Shapes {
    Shapes::new(MAX_LEN, &BUCKETS)
}

fn configs() -> Vec<LlamaConfig> {
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
        base,
        LlamaConfig {
            hidden: 128,
            heads: 4,
            kv_heads: 1,
            ff: 160,
            vocab: 64,
            tied_head: false,
            ..base
        },
        LlamaConfig {
            hidden: 96,
            heads: 2,
            kv_heads: 1,
            head_dim: 128,
            ff: 200,
            vocab: 77,
            tied_head: false,
            ..base
        },
    ]
}

#[test]
fn greedy_tokens_on_the_card_are_the_logits_path_s() {
    with_device(Config::default(), |d| {
        for (i, config) in configs().iter().enumerate() {
            let model = Llama::<TtBackend>::new(config, &d);
            let g = TracedGenerator::<_, TtBackend>::new(&model, &shapes(), &d).unwrap();
            for len in LENGTHS {
                let p = prompt(len, config.vocab);
                let want = g.generate(&p, STEPS).unwrap();
                let got = g.generate_tokens(&p, STEPS).unwrap();
                assert_eq!(got.tokens, want.tokens, "config {i}, prompt {len}");
            }
        }
    });
}

fn logits(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
        })
        .collect()
}

fn chosen<B: Backend>(v: &[f32], sampler: &Sampler, u: f32, d: &B::Device) -> i32 {
    let x = Tensor::<B, 2>::from_data(TensorData::new(v.to_vec(), [1, v.len()]), d);
    let u = Tensor::<B, 2>::from_data(TensorData::new(vec![u], [1, 1]), d);
    let t: Tensor<B, 2, Int> = choose(x, sampler, Some(u));
    t.into_data().convert::<i32>().to_vec::<i32>().unwrap()[0]
}

#[test]
fn sampled_tokens_are_burn_flex_s() {
    with_device(Config::default(), |d| {
        let sample = |temperature, top_k, top_p| Sampler::Sample {
            temperature,
            top_k,
            top_p,
            seed: 0,
        };
        let samplers = [
            sample(0.7, Some(5), None),
            sample(1.0, Some(20), Some(0.9)),
            sample(1.3, None, Some(0.5)),
            sample(1.0, None, None),
        ];
        let (mut compared, mut near) = (0, 0);
        // 3,000: past one sort, so the top-k is found in rounds of rows.
        for (n, seed) in [(100usize, 3u64), (1000, 5), (3000, 7)] {
            let v = logits(seed, n);
            let mut sorted: Vec<usize> = (0..n).collect();
            sorted.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
            let mut draws = Draws::new(seed);
            for _ in 0..12 {
                let u = draws.draw();
                for s in &samplers {
                    let flex = chosen::<Flex>(&v, s, u, &FlexDevice);
                    let edge = [u - 1e-5, u + 1e-5].iter().any(|&w| {
                        chosen::<Flex>(&v, s, w.clamp(0.0, 0.999_999), &FlexDevice) != flex
                    });
                    if edge {
                        near += 1;
                        continue;
                    }
                    let tt = chosen::<TtBackend>(&v, s, u, &d);
                    assert_eq!(tt, flex, "[1, {n}] {s:?}, u {u}");
                    if let Sampler::Sample { top_k: Some(k), .. } = s {
                        assert!(
                            sorted[..*k].contains(&(tt as usize)),
                            "{s:?}: in the top {k}"
                        );
                    }
                    compared += 1;
                }
                // Top-k of one is the greedy choice.
                let one = chosen::<TtBackend>(&v, &sample(0.8, Some(1), None), u, &d);
                assert_eq!(one, chosen::<TtBackend>(&v, &Sampler::Greedy, 0.0, &d));
            }
        }
        println!("{compared} draws compared, {near} within 1e-5 of a boundary left out");
        assert!(compared > 120, "too few draws compared: {compared}");
    });
}

/// `steps` tokens chosen by `sampler` with the same draws as `generate_tokens`, each step run
/// eagerly on the card.
fn eager_tokens(
    model: &Llama<TtBackend>,
    p: &[i32],
    steps: usize,
    sampler: &Sampler,
    seed: u64,
    d: &TtDevice,
) -> Vec<i32> {
    let ids = |t: &[i32]| {
        Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(t.to_vec(), [1, t.len()]), d)
    };
    let index =
        |i: usize| Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(vec![i as i32], [1]), d);
    let draw = |u: f32| Tensor::<TtBackend, 2>::from_data(TensorData::new(vec![u], [1, 1]), d);
    let token = |t: Tensor<TtBackend, 2, Int>| t.into_data().to_vec::<i32>().unwrap()[0];
    let shapes = shapes();
    let mut draws = Draws::new(seed);
    let mut cache = model.new_cache(MAX_LEN, d);
    let mut logits = None;
    for chunk in tt_models::generate::schedule(p.len(), &shapes.chunks) {
        let mut t = p[chunk.start..chunk.start + chunk.n].to_vec();
        t.resize(chunk.c, 0);
        let (l, state) = model.chunk(ids(&t), index(chunk.start), index(chunk.n - 1), &cache);
        cache.set_state(state);
        logits = Some(l);
    }
    let mut out = vec![token(choose(
        logits.unwrap(),
        sampler,
        Some(draw(draws.draw())),
    ))];
    for step in 1..steps {
        let position = p.len() + step - 1;
        let (l, state) = model.step(
            ids(&[*out.last().unwrap()]),
            index(position),
            shapes.step_len(position),
            &cache,
        );
        cache.set_state(state);
        out.push(token(choose(l, sampler, Some(draw(draws.draw())))));
    }
    out
}

#[test]
fn a_traced_sampled_generation_is_the_same_choices_eagerly() {
    with_device(Config::default(), |d| {
        let config = configs()[1];
        let model = Llama::<TtBackend>::new(&config, &d);
        let sampler = |seed| Sampler::Sample {
            temperature: 1.5,
            top_k: Some(16),
            top_p: Some(0.95),
            seed,
        };
        let mut seen = Vec::new();
        for seed in [1u64, 2, 3] {
            let g =
                TracedGenerator::<_, TtBackend>::with_sampler(&model, &shapes(), &d, sampler(seed))
                    .unwrap();
            for len in [3usize, 9] {
                let p = prompt(len, config.vocab);
                let got = g.generate_tokens(&p, 6).unwrap().tokens;
                let want = eager_tokens(&model, &p, 6, &sampler(seed), seed, &d);
                assert_eq!(got, want, "seed {seed}, prompt {len}");
                seen.push(got);
            }
        }
        // Other draws, other tokens: the draws reach the choice.
        seen.dedup();
        assert!(seen.len() > 2, "every seed chose the same tokens: {seen:?}");
    });
}
