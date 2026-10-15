//! Traced generation for any fixed-shape decoder (`tt_models::generate`): token ids in, logits
//! out, a decoder's step and its prefill chunks captured once as traces over one key/value cache
//! on the card (`burn_tt::SharedStateTraces`), then replayed for any prompt.
//!
//! The claims:
//! - **a traced generation equals the eager one, token for token and bit for bit.** The eager one
//!   is the same schedule run op by op (`generate_fixed`), and for a Llama-architecture model also
//!   the growing cache (`Llama::generate`).
//! - **The driver is not one model's.** It is checked on Llama shapes SmolLM2 does not have:
//!   - multi-head attention, and grouped-query attention 4:1;
//!   - head widths 32 and 128;
//!   - an untied head;
//!   - other rotary bases;
//!   - a width and a vocabulary that are not whole tiles.
//!
//!   It is also checked on a decoder that is not a Llama at all: GPT-2 style, with LayerNorm,
//!   learned positions, a fused QKV projection with biases and GELU. SmolLM2's checkpoint runs too,
//!   with the `weights` feature.
//! - **Prompts are covered in every way the schedule feeds them:** one chunk exactly, one chunk
//!   padded, several chunks with a remainder through either size.
//! - **One driver serves many prompts:** cache rows left by a longer earlier generation (and by the
//!   captures' own runs) never reach a later one.
//! - **Only the logits come back.** The lookup runs inside the trace, the host builds no op while
//!   generating, and the captures run natively.
//! - **Negative control:** the chunks captured over a cache the step never reads
//!   (`TracedGenerator::with_private_prefill`) give a different generation. The prefill's logits
//!   still agree; the decode steps do not.

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Int, Tensor};
use burn_tt::{TtBackend, TtDevice};
use tt_models::generate::{
    generate_fixed, FixedShapeCache, FixedShapeDecoder, GenerateError, Generation, TracedGenerator,
};
use tt_models::kv_cache::FixedCache;
use tt_models::llama::{Llama, LlamaConfig, Precision, SMOLLM2_135M};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

const MAX_LEN: usize = 16;
const BUCKETS: [usize; 2] = [4, 8];
/// Prompt lengths: one chunk exactly (4, 8), one padded (1, 3, 5), a chunk of 8 and the rest
/// through the 4 (9, 12).
const LENGTHS: [usize; 7] = [1, 3, 4, 5, 8, 9, 12];
/// Decode steps per prompt (the tokens chosen; one fewer are fed back).
const STEPS: usize = 3;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A prompt of `len` ids below `vocab`, not all alike.
fn prompt(len: usize, vocab: usize) -> Vec<i32> {
    (0..len)
        .map(|i| ((7 + 31 * i * i + 13 * i) % vocab) as i32)
        .collect()
}

fn assert_same(got: &Generation, want: &Generation, what: &str) {
    assert_eq!(got.tokens, want.tokens, "{what}: tokens");
    assert_eq!(got.logits.len(), want.logits.len(), "{what}: predictions");
    for (i, (g, w)) in got.logits.iter().zip(&want.logits).enumerate() {
        assert_eq!(bits(g), bits(w), "{what}: logits {i}");
    }
}

/// Build a generator, checking that its captures and warm-up ran natively: the trace inputs it
/// makes resident are the only transfers outside an op (uploads, nothing read back or staged).
fn traced<'m, M: FixedShapeDecoder<TtBackend>>(
    model: &'m M,
    max_len: usize,
    buckets: &[usize],
    d: &TtDevice,
) -> TracedGenerator<'m, M> {
    let (g, report) = burn_tt::with_report(|| TracedGenerator::new(model, max_len, buckets, d));
    let report = burn_tt::Report(
        report
            .0
            .into_iter()
            .filter(|s| !(s.op == "(outside an op)" && s.downloaded == 0 && s.staged == 0))
            .collect(),
    );
    assert_native_model(&report);
    g.unwrap()
}

/// A traced generation, checking that the host built no Burn op for it: what runs is replays,
/// their inputs written and their logits read.
fn generate(
    g: &TracedGenerator<'_, impl FixedShapeDecoder<TtBackend>>,
    p: &[i32],
    steps: usize,
) -> Generation {
    let (out, report) = burn_tt::with_report(|| g.generate(p, steps));
    for s in &report.0 {
        assert_eq!(s.op, "(outside an op)", "a traced generation ran a Burn op");
        assert_eq!(s.staged, 0, "a traced generation staged");
    }
    out.unwrap()
}

/// Shapes SmolLM2 does not have, each two layers.
fn synthetic_configs() -> Vec<(&'static str, LlamaConfig)> {
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
        ("multi-head, tied, ragged vocabulary", base),
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
            "head width 128, ragged width, untied",
            LlamaConfig {
                hidden: 96,
                heads: 2,
                kv_heads: 1,
                head_dim: 128,
                ff: 200,
                vocab: 77,
                rope_theta: 500_000.0,
                tied_head: false,
                ..base
            },
        ),
    ]
}

/// A Llama `model`'s traced generation of every prompt length against the eager fixed-shape
/// schedule and the growing cache.
fn check_llama(model: &Llama<TtBackend>, what: &str, d: &TtDevice) {
    let vocab = model.config().vocab;
    let g = traced(model, MAX_LEN, &BUCKETS, d);
    for len in LENGTHS {
        let p = prompt(len, vocab);
        let got = generate(&g, &p, STEPS);
        let fixed = generate_fixed(model, &p, STEPS, MAX_LEN, &BUCKETS, d).unwrap();
        assert_same(
            &got,
            &fixed,
            &format!("{what}, prompt {len}, eager fixed schedule"),
        );
        let growing = model.generate(&p, STEPS, Precision::F32, d);
        assert_same(
            &got,
            &growing,
            &format!("{what}, prompt {len}, growing cache"),
        );
    }
}

#[test]
fn synthetic_llamas_generate_as_eagerly() {
    with_device(Config::default(), |d| {
        for (what, config) in synthetic_configs() {
            let model = Llama::<TtBackend>::new(&config, &d);
            check_llama(&model, what, &d);
        }
    });
}

#[test]
fn one_driver_serves_many_prompts() {
    with_device(Config::default(), |d| {
        let (_, config) = synthetic_configs()[1];
        let model = Llama::<TtBackend>::new(&config, &d);
        let g = traced(&model, MAX_LEN, &BUCKETS, &d);
        // A generation filling the cache, then shorter ones over what it left.
        for (len, steps) in [(12, 5), (3, 4), (9, 2), (1, MAX_LEN)] {
            let p = prompt(len, config.vocab);
            let got = generate(&g, &p, steps);
            let want = model.generate(&p, steps, Precision::F32, &d);
            assert_same(&got, &want, &format!("prompt {len}, {steps} steps"));
        }
    });
}

#[test]
fn chunks_over_an_unshared_cache_diverge() {
    with_device(Config::default(), |d| {
        let (_, config) = synthetic_configs()[0];
        let model = Llama::<TtBackend>::new(&config, &d);
        let g = TracedGenerator::with_private_prefill(&model, MAX_LEN, &BUCKETS, &d).unwrap();
        let p = prompt(5, config.vocab);
        let got = g.generate(&p, STEPS).unwrap();
        let want = model.generate(&p, STEPS, Precision::F32, &d);
        assert_eq!(
            bits(&got.logits[0]),
            bits(&want.logits[0]),
            "the prefill itself"
        );
        assert!(
            got.logits[1..]
                .iter()
                .zip(&want.logits[1..])
                .any(|(g, w)| bits(g) != bits(w)),
            "a step that never reads the prefill's cache must diverge"
        );
    });
}

#[test]
fn requests_that_do_not_fit_are_refused_by_name() {
    with_device(Config::default(), |d| {
        let (_, config) = synthetic_configs()[0];
        let model = Llama::<TtBackend>::new(&config, &d);
        assert!(matches!(
            TracedGenerator::new(&model, MAX_LEN, &[4, MAX_LEN + 1], &d),
            Err(GenerateError::Refused(_))
        ));
        let g = TracedGenerator::new(&model, MAX_LEN, &BUCKETS, &d).unwrap();
        assert!(matches!(
            g.generate(&prompt(12, config.vocab), 6),
            Err(GenerateError::Capacity {
                needed: 17,
                max_len: MAX_LEN
            })
        ));
        assert!(matches!(
            g.decode(0, MAX_LEN),
            Err(GenerateError::Capacity { .. })
        ));
    });
}

/// A decoder that is not a Llama: GPT-2's block (LayerNorm, learned positions, a fused QKV
/// projection with biases, multi-head attention, a GELU MLP, an untied head), over the same
/// architecture-neutral cache.
#[derive(Module, Debug)]
struct Gpt<B: Backend> {
    wte: Embedding<B>,
    wpe: Embedding<B>,
    blocks: Vec<GptBlock<B>>,
    ln_f: LayerNorm<B>,
    head: Linear<B>,
    heads: usize,
}

#[derive(Module, Debug)]
struct GptBlock<B: Backend> {
    ln_1: LayerNorm<B>,
    qkv: Linear<B>,
    proj: Linear<B>,
    ln_2: LayerNorm<B>,
    fc: Linear<B>,
    out: Linear<B>,
}

const GPT_HIDDEN: usize = 48;
const GPT_HEADS: usize = 3;
const GPT_VOCAB: usize = 90;

impl<B: Backend> Gpt<B> {
    fn new(layers: usize, device: &B::Device) -> Self {
        let h = GPT_HIDDEN;
        Self {
            wte: EmbeddingConfig::new(GPT_VOCAB, h).init(device),
            wpe: EmbeddingConfig::new(MAX_LEN, h).init(device),
            blocks: (0..layers)
                .map(|_| GptBlock {
                    ln_1: LayerNormConfig::new(h).init(device),
                    qkv: LinearConfig::new(h, 3 * h).init(device),
                    proj: LinearConfig::new(h, h).init(device),
                    ln_2: LayerNormConfig::new(h).init(device),
                    fc: LinearConfig::new(h, 4 * h).init(device),
                    out: LinearConfig::new(4 * h, h).init(device),
                })
                .collect(),
            ln_f: LayerNormConfig::new(h).init(device),
            head: LinearConfig::new(h, GPT_VOCAB)
                .with_bias(false)
                .init(device),
            heads: GPT_HEADS,
        }
    }

    /// `c` tokens at `positions` (`[c]`, clamped into the table) through every block: the last
    /// block's output and the new state.
    fn blocks_fixed(
        &self,
        tokens: Tensor<B, 2, Int>,
        positions: Tensor<B, 1, Int>,
        cache: &GptCache<B>,
        write: &tt_models::kv_cache::FixedWrite<B>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 4>>) {
        let [_, c] = tokens.dims();
        let (h, heads) = (GPT_HIDDEN, self.heads);
        let d = h / heads;
        let mut x = self.wte.forward(tokens) + self.wpe.forward(positions.reshape([1, c]));
        let mut state = Vec::with_capacity(2 * self.blocks.len());
        for (i, b) in self.blocks.iter().enumerate() {
            let qkv = b.qkv.forward(b.ln_1.forward(x.clone()));
            let part = |j: usize| {
                qkv.clone()
                    .slice([0..1, 0..c, j * h..(j + 1) * h])
                    .reshape([1, c, heads, d])
                    .swap_dims(1, 2)
            };
            let (k_cache, v_cache) = cache.kv.layer(i);
            let k = write.write(k_cache, part(1));
            let v = write.write(v_cache, part(2));
            let ctx = attention(
                part(0),
                k.clone(),
                v.clone(),
                Some(write.later()),
                None,
                AttentionModuleOptions {
                    scale: None,
                    softcap: None,
                    is_causal: false,
                },
            );
            x = x + b.proj.forward(ctx.swap_dims(1, 2).reshape([1, c, h]));
            x = x.clone() + b.out.forward(gelu(b.fc.forward(b.ln_2.forward(x))));
            state.extend([k, v]);
        }
        (x, state)
    }
}

#[derive(Clone, Debug)]
struct GptCache<B: Backend> {
    kv: FixedCache<B>,
}

impl<B: Backend> FixedShapeCache<B> for GptCache<B> {
    fn max_len(&self) -> usize {
        self.kv.max_len()
    }
    fn state(&self) -> Vec<Tensor<B, 4>> {
        self.kv.state()
    }
    fn set_state(&mut self, state: Vec<Tensor<B, 4>>) {
        self.kv.set_state(state)
    }
}

impl<B: Backend> FixedShapeDecoder<B> for Gpt<B> {
    type Cache = GptCache<B>;

    fn new_cache(&self, max_len: usize, device: &B::Device) -> GptCache<B> {
        assert!(max_len <= MAX_LEN, "learned positions for {MAX_LEN}");
        let dtype = self.head.weight.val().dtype();
        GptCache {
            kv: FixedCache::zeros(
                self.blocks.len(),
                self.heads,
                GPT_HIDDEN / self.heads,
                max_len,
                dtype,
                device,
            ),
        }
    }

    fn chunk(
        &self,
        tokens: Tensor<B, 2, Int>,
        start: Tensor<B, 1, Int>,
        last: Tensor<B, 1, Int>,
        cache: &GptCache<B>,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        let [_, c] = tokens.dims();
        let (write, own) = cache.kv.chunk_write(start, c);
        let positions = own.clamp_max(cache.kv.max_len() as i64 - 1);
        let (x, state) = self.blocks_fixed(tokens, positions, cache, &write);
        let row = self
            .ln_f
            .forward(x)
            .reshape([c, GPT_HIDDEN])
            .select(0, last)
            .reshape([1, GPT_HIDDEN]);
        (self.head.forward(row), state)
    }

    fn step(
        &self,
        token: Tensor<B, 2, Int>,
        position: Tensor<B, 1, Int>,
        cache: &GptCache<B>,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        let write = cache.kv.step_write(position.clone());
        let (x, state) = self.blocks_fixed(token, position, cache, &write);
        let row = self.ln_f.forward(x).reshape([1, GPT_HIDDEN]);
        (self.head.forward(row), state)
    }
}

#[test]
fn a_decoder_that_is_not_a_llama_generates_as_eagerly() {
    with_device(Config::default(), |d| {
        let model = Gpt::<TtBackend>::new(2, &d);
        let g = traced(&model, MAX_LEN, &BUCKETS, &d);
        for len in LENGTHS {
            let p = prompt(len, GPT_VOCAB);
            let got = generate(&g, &p, STEPS);
            let want = generate_fixed(&model, &p, STEPS, MAX_LEN, &BUCKETS, &d).unwrap();
            assert_same(&got, &want, &format!("GPT, prompt {len}"));
        }
    });
}

/// SmolLM2's checkpoint: one layer on the simulator; on silicon all thirty, over every tile.
#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-tests/weights"
)]
fn smollm2_generates_as_eagerly() {
    let (layers, config) = if cfg!(feature = "silicon") {
        (
            SMOLLM2_135M.layers,
            Config {
                tiles: Some(burn_tt::TileChoice::All),
                ..Config::default()
            },
        )
    } else {
        (1, Config::default())
    };
    with_device(config, |d| {
        let (model, _) = Llama::<TtBackend>::smollm2_135m(layers, Precision::F32, &d).unwrap();
        check_llama(&model, "SmolLM2", &d);
    });
}
