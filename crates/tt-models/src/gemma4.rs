//! The Gemma 4 text model (the "E" models' shape): a decoder of sliding-window and full attention
//! layers, interleaved, each fed a **per-layer embedding** of the token besides the residual
//! stream, its last layers **sharing** the keys and values of an earlier layer instead of computing
//! their own, and its logits soft-capped. `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16`
//! ([`NEMOTRON_3_5_BOLT_E4B`], a fine-tune of `google/gemma-4-E4B-it`) is the checkpoint the
//! gates load; the code is the architecture's.
//!
//! The reference is HuggingFace `transformers`' `modeling_gemma4.py` (5.19.0), and these are its
//! details the configuration does not state:
//!
//! * Every RMSNorm scales by `weight` (not `1 + weight`), in F32; the values' norm has no scale.
//!   Each layer norms the attention's and the MLP's outputs as well as their inputs, and the
//!   per-layer branch's output.
//! * The embedding is scaled by `sqrt(hidden)`. The per-layer input of layer `i` is
//!   `(rms(proj(embedded) / sqrt(hidden))[i] + sqrt(per_layer) table[token][i]) / sqrt(2)`, the
//!   projection normed per layer; a layer gates it with `gelu_tanh(gate(x))`, projects it back to
//!   the stream's width, norms it and adds it to the stream. Every layer ends by scaling the
//!   stream by its `layer_scalar`.
//! * Attention scales scores by 1 (the q/k norms do the work). Sliding layers attend to the
//!   `sliding_window` latest positions, their own included, with the ordinary rotary embedding;
//!   full layers attend to every position with wider heads and a "proportional" embedding: a
//!   quarter of the pairs rotate, their frequencies `theta^(-2i / head_dim)` over the whole head's
//!   width, the other pairs at frequency zero ([`crate::rope::frequencies`]).
//! * The last `kv_shared_layers` layers compute only queries: each reads the keys and values of
//!   the last earlier layer of its own kind. (The checkpoint's
//!   `model-frozen-shared-kv.safetensors` holds those layers' unused key/value weights; it is not
//!   fetched.)
//! * The logits are `cap tanh(logits / cap)`.
//!
//! Text only: the checkpoint's vision and audio towers are left in the file, by name
//! ([`Gemma4::pretrained`]). Under [`Precision::F32`] the two embedding tables stay in the
//! checkpoint's BF16 and each looked-up row is widened (exactly), so the 2.8-billion-entry
//! per-layer table is not held at twice its size; the output head is the token table widened.

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Bool, DType, Int, Tensor, TensorData};

use crate::generate::{eager, EagerDecoder, Generation};
use crate::llama::{project, repeat_kv, Precision};
use crate::load::{load_safetensors, weights_files, Layout, LoadError, Loaded};
use crate::qwen35::rms;
use crate::rope::{frequencies, Rope};

/// One kind of attention layer's shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionShape {
    pub head_dim: usize,
    pub kv_heads: usize,
    pub rope_theta: f64,
    /// Rotary pairs that turn (of `head_dim / 2`).
    pub rotated: usize,
}

/// The shape of a Gemma 4 text model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4Config {
    /// The residual stream's width.
    pub hidden: usize,
    /// Query heads, in every layer.
    pub heads: usize,
    /// Sliding-window layers, and full-attention ones.
    pub sliding: AttentionShape,
    pub full: AttentionShape,
    /// Layer `i` is full attention when `(i + 1) % full_every == 0`.
    pub full_every: usize,
    /// How many latest positions a sliding layer sees, its own included.
    pub sliding_window: usize,
    /// The last this many layers read an earlier layer's keys and values.
    pub kv_shared_layers: usize,
    /// The MLP's inner width.
    pub ff: usize,
    pub vocab: usize,
    pub layers: usize,
    /// Each layer's per-layer input width (`hidden_size_per_layer_input`).
    pub per_layer: usize,
    pub eps: f64,
    /// The logits' soft cap (`final_logit_softcapping`).
    pub logit_cap: Option<f64>,
}

/// NVIDIA-Nemotron-3.5-Bolt-E4B (Gemma 4 E4B, text): 42 layers of width 2560, 8 query heads; five
/// sliding-window layers (2 key/value heads of 256, a window of 512) to each full one (2 of 512,
/// 64 of 256 pairs rotating); the last 18 layers sharing keys and values; an MLP of 10,240; a
/// 262,144-token vocabulary, the head tied to it; per-layer inputs of 256; logits capped at 30.
/// Checked against the checkpoint's `config.json` by
/// `crates/tt-models/reference/gemma4_reference.py` (`tests/gemma4_host.rs`).
pub const NEMOTRON_3_5_BOLT_E4B: Gemma4Config = Gemma4Config {
    hidden: 2560,
    heads: 8,
    sliding: AttentionShape {
        head_dim: 256,
        kv_heads: 2,
        rope_theta: 10_000.0,
        rotated: 128,
    },
    full: AttentionShape {
        head_dim: 512,
        kv_heads: 2,
        rope_theta: 1_000_000.0,
        rotated: 64,
    },
    full_every: 6,
    sliding_window: 512,
    kv_shared_layers: 18,
    ff: 10240,
    vocab: 262_144,
    layers: 42,
    per_layer: 256,
    eps: 1e-6,
    logit_cap: Some(30.0),
};

impl Gemma4Config {
    /// Whether layer `index` is full attention (else sliding-window).
    pub fn is_full(&self, index: usize) -> bool {
        (index + 1) % self.full_every == 0
    }

    fn shape(&self, index: usize) -> &AttentionShape {
        if self.is_full(index) {
            &self.full
        } else {
            &self.sliding
        }
    }

    /// The first layer that reads another's keys and values (`layers` when none does).
    pub fn first_shared(&self) -> usize {
        self.layers.saturating_sub(self.kv_shared_layers)
    }

    /// The layer whose keys and values layer `index` attends with: its own, or, for a sharing
    /// layer, the last layer of its kind before the sharing ones.
    pub fn kv_source(&self, index: usize) -> usize {
        let first = self.first_shared();
        if index < first {
            return index;
        }
        let full = self.is_full(index);
        (0..first)
            .rev()
            .find(|&i| self.is_full(i) == full)
            .expect("a sharing layer has an earlier layer of its kind")
    }

    fn check(&self) {
        assert!(self.layers >= 1 && self.full_every >= 1, "{self:?}");
        for s in [&self.sliding, &self.full] {
            assert!(
                self.heads % s.kv_heads == 0 && s.head_dim % 2 == 0 && s.rotated <= s.head_dim / 2,
                "{s:?}"
            );
        }
        for i in self.first_shared()..self.layers {
            self.kv_source(i);
        }
    }
}

fn linear<B: Backend>(inp: usize, out: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(inp, out).with_bias(false).init(device)
}

/// An RMSNorm scaling by `weight` (none when absent), in F32, returned in `x`'s dtype.
#[derive(Module, Debug)]
pub struct Norm<B: Backend> {
    weight: Option<Param<Tensor<B, 1>>>,
    eps: f64,
}

impl<B: Backend> Norm<B> {
    fn new(width: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Some(Param::from_tensor(Tensor::ones([width], device))),
            eps,
        }
    }

    fn unscaled(eps: f64) -> Self {
        Self { weight: None, eps }
    }

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = x.dtype();
        let normed = rms(x, self.eps);
        match &self.weight {
            Some(w) => normed * w.val().cast(DType::F32).unsqueeze::<D>(),
            None => normed,
        }
        .cast(dtype)
    }
}

/// `gelu` with the tanh approximation, as PyTorch computes it:
/// `x / 2 (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))`.
fn gelu_tanh<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    const BETA: f64 = std::f64::consts::SQRT_2 * std::f64::consts::FRAC_2_SQRT_PI * 0.5;
    let cube = x.clone() * x.clone() * x.clone();
    let inner = (x.clone() + cube.mul_scalar(0.044715)).mul_scalar(BETA);
    x.mul_scalar(0.5) * inner.tanh().add_scalar(1.0)
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: Linear<B>,
    q_norm: Norm<B>,
    /// Absent in a layer that reads another's keys and values.
    k_proj: Option<Linear<B>>,
    v_proj: Option<Linear<B>>,
    k_norm: Option<Norm<B>>,
    v_norm: Norm<B>,
    o_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Layer<B: Backend> {
    input_layernorm: Norm<B>,
    self_attn: Attention<B>,
    post_attention_layernorm: Norm<B>,
    pre_feedforward_layernorm: Norm<B>,
    mlp: Mlp<B>,
    post_feedforward_layernorm: Norm<B>,
    per_layer_input_gate: Linear<B>,
    per_layer_projection: Linear<B>,
    post_per_layer_input_norm: Norm<B>,
    /// The stream's scale at the layer's end, `[1]`.
    layer_scalar: Param<Tensor<B, 1>>,
}

/// One layer's keys and values, `[1, kv_heads, max_len, head_dim]`, filled front to back (none
/// for a layer that reads another's).
#[derive(Clone, Debug)]
pub struct LayerCache<B: Backend> {
    pub k: Tensor<B, 4>,
    pub v: Tensor<B, 4>,
}

/// The keys and values of every position a model has seen.
pub struct Cache<B: Backend> {
    layers: Vec<Option<LayerCache<B>>>,
    len: usize,
    max_len: usize,
}

impl<B: Backend> Cache<B> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Where the queries `start..start + t` may not look among keys `0..start + t`: later positions,
/// and, with a window, positions `window` or more before. `None` when nothing is masked (one
/// query that sees everything).
fn mask<B: Backend>(
    start: usize,
    t: usize,
    window: Option<usize>,
    device: &B::Device,
) -> Option<Tensor<B, 4, Bool>> {
    let total = start + t;
    let hidden = |q: usize, k: usize| k > q || window.is_some_and(|w| q - k >= w);
    let values: Vec<bool> = (start..total)
        .flat_map(|q| (0..total).map(move |k| hidden(q, k)))
        .collect();
    values
        .iter()
        .any(|&m| m)
        .then(|| Tensor::from_data(TensorData::new(values, [1, 1, t, total]), device))
}

impl<B: Backend> Attention<B> {
    fn new(cfg: &Gemma4Config, index: usize, device: &B::Device) -> Self {
        let s = cfg.shape(index);
        let own = cfg.kv_source(index) == index;
        let (q, kv) = (cfg.heads * s.head_dim, s.kv_heads * s.head_dim);
        Self {
            q_proj: linear(cfg.hidden, q, device),
            q_norm: Norm::new(s.head_dim, cfg.eps, device),
            k_proj: own.then(|| linear(cfg.hidden, kv, device)),
            v_proj: own.then(|| linear(cfg.hidden, kv, device)),
            k_norm: own.then(|| Norm::new(s.head_dim, cfg.eps, device)),
            v_norm: Norm::unscaled(cfg.eps),
            o_proj: linear(q, cfg.hidden, device),
        }
    }

    /// The attention's output for the normed `x`. A layer with its own keys and values writes
    /// this step's into `cache` at `start..`; a sharing layer reads `shared`'s.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        cfg: &Gemma4Config,
        index: usize,
        x: Tensor<B, 3>,
        rope: &Rope<B>,
        start: usize,
        cache: Option<&mut LayerCache<B>>,
        shared: Option<&LayerCache<B>>,
    ) -> Tensor<B, 3> {
        let [n, t, _] = x.dims();
        let s = cfg.shape(index);
        let (heads, d) = (cfg.heads, s.head_dim);
        let total = start + t;
        let q = project(x.clone(), &self.q_proj).reshape([n, t, heads, d]);
        let q = rope.apply(self.q_norm.forward(q).swap_dims(1, 2));
        let (keys, values) = match (cache, shared) {
            (Some(cache), None) => {
                let (k_proj, v_proj, k_norm) = (
                    self.k_proj.as_ref().expect("a layer with its own keys"),
                    self.v_proj.as_ref().expect("a layer with its own values"),
                    self.k_norm.as_ref().expect("a layer with its own keys"),
                );
                let k = project(x.clone(), k_proj).reshape([n, t, s.kv_heads, d]);
                let v = project(x, v_proj).reshape([n, t, s.kv_heads, d]);
                let k = rope.apply(k_norm.forward(k).swap_dims(1, 2));
                let v = self.v_norm.forward(v).swap_dims(1, 2);
                let at = [0..n, 0..s.kv_heads, start..total, 0..d];
                cache.k = cache.k.clone().slice_assign(at.clone(), k);
                cache.v = cache.v.clone().slice_assign(at, v);
                (cache.k.clone(), cache.v.clone())
            }
            (None, Some(source)) => (source.k.clone(), source.v.clone()),
            _ => unreachable!("a layer either keeps keys and values or reads another's"),
        };
        let seen = [0..n, 0..s.kv_heads, 0..total, 0..d];
        let groups = heads / s.kv_heads;
        let keys = repeat_kv(keys.slice(seen.clone()), groups);
        let values = repeat_kv(values.slice(seen), groups);
        let window = (!cfg.is_full(index)).then_some(cfg.sliding_window);
        let out = attention(
            q,
            keys,
            values,
            mask(start, t, window, &x_device(&self.o_proj)),
            None,
            AttentionModuleOptions {
                scale: Some(1.0),
                softcap: None,
                is_causal: false,
            },
        );
        let context = out.swap_dims(1, 2).reshape([n, t, heads * d]);
        project(context, &self.o_proj)
    }
}

fn x_device<B: Backend>(l: &Linear<B>) -> B::Device {
    l.weight.val().device()
}

impl<B: Backend> Layer<B> {
    fn new(cfg: &Gemma4Config, index: usize, device: &B::Device) -> Self {
        let norm = || Norm::new(cfg.hidden, cfg.eps, device);
        Self {
            input_layernorm: norm(),
            self_attn: Attention::new(cfg, index, device),
            post_attention_layernorm: norm(),
            pre_feedforward_layernorm: norm(),
            mlp: Mlp {
                gate_proj: linear(cfg.hidden, cfg.ff, device),
                up_proj: linear(cfg.hidden, cfg.ff, device),
                down_proj: linear(cfg.ff, cfg.hidden, device),
            },
            post_feedforward_layernorm: norm(),
            per_layer_input_gate: linear(cfg.hidden, cfg.per_layer, device),
            per_layer_projection: linear(cfg.per_layer, cfg.hidden, device),
            post_per_layer_input_norm: norm(),
            layer_scalar: Param::from_tensor(Tensor::ones([1], device)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        cfg: &Gemma4Config,
        index: usize,
        x: Tensor<B, 3>,
        per_layer_input: Tensor<B, 3>,
        rope: &Rope<B>,
        start: usize,
        cache: Option<&mut LayerCache<B>>,
        shared: Option<&LayerCache<B>>,
    ) -> Tensor<B, 3> {
        let h = self.input_layernorm.forward(x.clone());
        let h = self
            .self_attn
            .forward(cfg, index, h, rope, start, cache, shared);
        let x = x + self.post_attention_layernorm.forward(h);

        let h = self.pre_feedforward_layernorm.forward(x.clone());
        let m = &self.mlp;
        let h = gelu_tanh(project(h.clone(), &m.gate_proj)) * project(h, &m.up_proj);
        let x = x + self
            .post_feedforward_layernorm
            .forward(project(h, &m.down_proj));

        let h = gelu_tanh(project(x.clone(), &self.per_layer_input_gate)) * per_layer_input;
        let h = project(h, &self.per_layer_projection);
        let x = x + self.post_per_layer_input_norm.forward(h);
        let scalar = self.layer_scalar.val().cast(x.dtype()).unsqueeze::<3>();
        x * scalar
    }
}

/// The Gemma 4 text model.
#[derive(Module, Debug)]
pub struct Gemma4<B: Backend> {
    embed_tokens: Embedding<B>,
    /// Every layer's per-layer embedding of a token, `[vocab, layers per_layer]`.
    embed_tokens_per_layer: Embedding<B>,
    /// The stream's projection to every layer's per-layer input, `[hidden, layers per_layer]`.
    per_layer_model_projection: Linear<B>,
    per_layer_projection_norm: Norm<B>,
    layers: Vec<Layer<B>>,
    norm: Norm<B>,
    /// Not a parameter: the module derive skips it.
    config: Gemma4Config,
}

/// Widens every float parameter to F32, on its data (exact): a freshly loaded burn-tt tensor is
/// still on the host, and widening it there uploads only the F32 tensor, where a cast on the
/// device would first upload the BF16 one (in an F32 tile's slot) and free it after, leaving GDDR
/// in holes the 11.3 GB per-layer table then finds no run in (`step261_gemma4`, 2026-10-30).
struct ToF32;

fn widen<B: Backend, const D: usize>(t: Tensor<B, D>) -> Tensor<B, D> {
    if t.dtype() == DType::F32 {
        return t;
    }
    let device = t.device();
    Tensor::from_data(t.into_data().convert::<f32>(), (&device, DType::F32))
}

impl<B: Backend> ModuleMapper<B> for ToF32 {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        param.map(widen)
    }
}

impl<B: Backend> Gemma4<B> {
    /// The architecture `config` describes, with its initialiser's weights.
    pub fn new(config: &Gemma4Config, device: &B::Device) -> Self {
        config.check();
        let packed = config.layers * config.per_layer;
        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab, config.hidden).init(device),
            embed_tokens_per_layer: EmbeddingConfig::new(config.vocab, packed).init(device),
            per_layer_model_projection: linear(config.hidden, packed, device),
            per_layer_projection_norm: Norm::new(config.per_layer, config.eps, device),
            layers: (0..config.layers)
                .map(|i| Layer::new(config, i, device))
                .collect(),
            norm: Norm::new(config.hidden, config.eps, device),
            config: *config,
        }
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    /// A pretrained Gemma 4 checkpoint (`vendor/weights/<weights>/<file>`) whose text shape
    /// `config` gives, computing in `precision`. The language model's tensors live under
    /// `model.language_model.`; the vision and audio towers (`model.vision_tower.`,
    /// `model.audio_tower.`, `model.embed_vision.`, `model.embed_audio.`) are the only tensors the
    /// file may hold besides.
    pub fn pretrained(
        config: &Gemma4Config,
        weights: &str,
        file: &str,
        precision: Precision,
        device: &B::Device,
    ) -> Result<(Self, Loaded), LoadError> {
        let mut model = Self::new(config, device);
        let ignore = |name: &str| {
            name.strip_prefix("model.")
                .is_some_and(|rest| rest.contains("vision") || rest.contains("audio"))
        };
        let loaded = load_safetensors(
            &mut model,
            &weights_files(weights, &[file])[0],
            &Layout {
                remaps: &[(r"^model\.language_model\.", "")],
                pytorch: true,
                ignore: &ignore,
            },
        )?;
        let model = match precision {
            Precision::Bf16 => model,
            Precision::F32 => model.widened(),
        };
        Ok((model, loaded))
    }

    /// Every parameter widened to F32 but the two embedding tables, whose looked-up rows are
    /// widened instead (the same values).
    fn widened(self) -> Self {
        Self {
            per_layer_model_projection: self.per_layer_model_projection.map(&mut ToF32),
            per_layer_projection_norm: self.per_layer_projection_norm.map(&mut ToF32),
            layers: self.layers.into_iter().map(|l| l.map(&mut ToF32)).collect(),
            norm: self.norm.map(&mut ToF32),
            ..self
        }
    }

    /// The same model with its two embedding tables widened to F32 as well. On burn-tt a BF16
    /// tensor takes an F32 tile's slot in GDDR, so the BF16 per-layer table would hold 11.3 GB
    /// anyway, and an F32 table is looked up by a row select and multiplied by the head as it lies.
    /// Widened on their data, as [`Precision::F32`] widens the rest.
    pub fn with_f32_tables(self) -> Self {
        Self {
            embed_tokens: self.embed_tokens.map(&mut ToF32),
            embed_tokens_per_layer: self.embed_tokens_per_layer.map(&mut ToF32),
            ..self
        }
    }

    /// NVIDIA-Nemotron-3.5-Bolt-E4B ([`NEMOTRON_3_5_BOLT_E4B`], `cargo xtask fetch-weights --model
    /// nemotron-3.5-bolt-e4b`).
    pub fn nemotron_3_5_bolt_e4b(
        precision: Precision,
        device: &B::Device,
    ) -> Result<(Self, Loaded), LoadError> {
        Self::pretrained(
            &NEMOTRON_3_5_BOLT_E4B,
            "nemotron-3.5-bolt-e4b",
            "model-00001-of-00001.safetensors",
            precision,
            device,
        )
    }

    /// An empty cache for `max_len` positions of one sequence, in `dtype`.
    pub fn cache(&self, max_len: usize, dtype: DType, device: &B::Device) -> Cache<B> {
        let cfg = &self.config;
        let layers = (0..cfg.layers)
            .map(|i| {
                (cfg.kv_source(i) == i).then(|| {
                    let s = cfg.shape(i);
                    let shape = [1, s.kv_heads, max_len, s.head_dim];
                    LayerCache {
                        k: Tensor::zeros(shape, (device, dtype)),
                        v: Tensor::zeros(shape, (device, dtype)),
                    }
                })
            })
            .collect();
        Cache {
            layers,
            len: 0,
            max_len,
        }
    }

    /// The final hidden states `[1, t, hidden]` (after the last norm) of `tokens` (`[1, t]`),
    /// whose positions follow `cache`'s, which they are appended to.
    pub fn hidden(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Cache<B>,
        dtype: DType,
    ) -> Tensor<B, 3> {
        let cfg = &self.config;
        let [n, t] = tokens.dims();
        assert_eq!(n, 1, "one sequence at a time");
        let start = cache.len;
        assert!(start + t <= cache.max_len, "the cache is full");
        let device = tokens.device();
        let (layers, per) = (cfg.layers, cfg.per_layer);

        let x = self
            .embed_tokens
            .forward(tokens.clone())
            .cast(dtype)
            .mul_scalar((cfg.hidden as f64).sqrt());
        let identity = self
            .embed_tokens_per_layer
            .forward(tokens)
            .cast(dtype)
            .mul_scalar((per as f64).sqrt())
            .reshape([n, t, layers, per]);
        let context = project(x.clone(), &self.per_layer_model_projection)
            .mul_scalar((cfg.hidden as f64).powf(-0.5))
            .reshape([n, t, layers, per]);
        let per_layer = (self.per_layer_projection_norm.forward(context) + identity)
            .mul_scalar(2f64.powf(-0.5));

        let rope_of = |s: &AttentionShape| {
            Rope::with_frequencies(
                start,
                t,
                &frequencies(s.head_dim, s.rotated, s.rope_theta),
                dtype,
                &device,
            )
        };
        let (sliding_rope, full_rope) = (rope_of(&cfg.sliding), rope_of(&cfg.full));
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            let rope = if cfg.is_full(i) {
                &full_rope
            } else {
                &sliding_rope
            };
            let input = per_layer
                .clone()
                .slice([0..n, 0..t, i..i + 1, 0..per])
                .reshape([n, t, per]);
            let source = cfg.kv_source(i);
            x = if source == i {
                let own = cache.layers[i].as_mut();
                layer.forward(cfg, i, x, input, rope, start, own, None)
            } else {
                let shared = cache.layers[source].as_ref();
                layer.forward(cfg, i, x, input, rope, start, None, shared)
            };
        }
        cache.len += t;
        self.norm.forward(x)
    }

    /// The logits `[1, vocab]` after the last of `tokens` (`[1, t]`), which are appended to
    /// `cache`.
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut Cache<B>,
        dtype: DType,
    ) -> Tensor<B, 2> {
        let [_, t] = tokens.dims();
        let hidden = self.config.hidden;
        let x = self.hidden(tokens, cache, dtype);
        let last = x.slice([0..1, t - 1..t, 0..hidden]).reshape([1, hidden]);
        let table = self.embed_tokens.weight.val().cast(last.dtype());
        let logits = last.matmul(table.transpose());
        match self.config.logit_cap {
            Some(cap) => logits.div_scalar(cap).tanh().mul_scalar(cap),
            None => logits,
        }
    }
}

impl<B: burn_tt::storage::StorageBackend> Gemma4<B> {
    /// Every matrix but the two embedding tables stored BF16
    /// ([`crate::quant::Plan::bf16_weights`]), for an F32 model: half the bytes and the GDDR of
    /// those matrices, the arithmetic F32 on the checkpoint's own BF16 values (bit for bit the F32
    /// model's logits is shown for the Llama, `step240`, not yet for this model).
    pub fn with_bf16_weights(self) -> Self {
        crate::quant::apply(
            self,
            &crate::quant::Plan::bf16_weights(),
            &crate::quant::Native,
        )
    }
}

impl<B: Backend> EagerDecoder<B> for Gemma4<B> {
    fn generate(
        &self,
        prompt: &[i32],
        steps: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Generation {
        let dtype = precision.dtype();
        let len = prompt.len() + steps;
        let mut cache = self.cache(len, dtype, device);
        eager(prompt, steps, None, device, |ids| {
            self.forward(ids, &mut cache, dtype)
        })
    }

    fn forced(
        &self,
        prompt: &[i32],
        continuation: &[i32],
        precision: Precision,
        device: &B::Device,
    ) -> Generation {
        let dtype = precision.dtype();
        let len = prompt.len() + continuation.len() + 1;
        let mut cache = self.cache(len, dtype, device);
        eager(prompt, 0, Some(continuation), device, |ids| {
            self.forward(ids, &mut cache, dtype)
        })
    }
}

#[cfg(test)]
fn ids<B: Backend>(tokens: &[i32], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(tokens.to_vec(), [1, tokens.len()]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_flex::{Flex, FlexDevice};

    #[test]
    fn sharing_layers_read_the_last_earlier_layer_of_their_kind() {
        let cfg = NEMOTRON_3_5_BOLT_E4B;
        assert_eq!(cfg.first_shared(), 24);
        assert_eq!(cfg.kv_source(3), 3);
        // 22 is the last sliding layer before 24, 23 the last full one.
        assert_eq!(cfg.kv_source(24), 22);
        assert_eq!(cfg.kv_source(29), 23);
        assert_eq!(cfg.kv_source(41), 23);
        assert_eq!(cfg.kv_source(40), 22);
    }

    #[test]
    fn the_window_hides_what_is_later_or_too_old() {
        let m = mask::<Flex>(2, 3, Some(2), &FlexDevice)
            .unwrap()
            .into_data()
            .to_vec::<bool>()
            .unwrap();
        // Queries 2, 3, 4 over keys 0..5: each sees itself and the one before.
        #[rustfmt::skip]
        let want = [
            true, false, false, true, true,
            true, true, false, false, true,
            true, true, true, false, false,
        ];
        assert_eq!(m, want);
        assert!(mask::<Flex>(7, 1, None, &FlexDevice).is_none());
        assert!(mask::<Flex>(7, 1, Some(8), &FlexDevice).is_none());
        assert!(mask::<Flex>(7, 1, Some(7), &FlexDevice).is_some());
    }

    /// Two sliding layers and a full one with their own keys and values, then one of each kind
    /// sharing them, a window shorter than the prompt.
    const TINY: Gemma4Config = Gemma4Config {
        hidden: 32,
        heads: 4,
        sliding: AttentionShape {
            head_dim: 8,
            kv_heads: 2,
            rope_theta: 10_000.0,
            rotated: 4,
        },
        full: AttentionShape {
            head_dim: 16,
            kv_heads: 1,
            rope_theta: 1_000_000.0,
            rotated: 2,
        },
        full_every: 3,
        sliding_window: 3,
        kv_shared_layers: 3,
        ff: 48,
        vocab: 64,
        layers: 6,
        per_layer: 8,
        eps: 1e-6,
        logit_cap: Some(30.0),
    };

    fn logits(model: &Gemma4<Flex>, cache: &mut Cache<Flex>, ids: &[i32]) -> Vec<f32> {
        model
            .forward(super::ids(ids, &FlexDevice), cache, DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    /// The cache a prompt leaves (every keeping layer's keys and values, read by the sharing
    /// layers) is what feeding its tokens one at a time leaves, with the window shorter than the
    /// prompt: the last token's logits agree to the rounding of sums over other shapes (measured
    /// below 1e-6), where a wrongly placed window or a wrong source layer moves them by tenths.
    #[test]
    fn a_prompt_at_once_leaves_the_cache_its_tokens_one_at_a_time_do() {
        let model = Gemma4::<Flex>::new(&TINY, &FlexDevice);
        let prompt = [3, 17, 42, 5, 9, 60, 1];
        let mut whole = model.cache(prompt.len(), DType::F32, &FlexDevice);
        let at_once = logits(&model, &mut whole, &prompt);
        let mut stepped = model.cache(prompt.len(), DType::F32, &FlexDevice);
        let mut one_at_a_time = Vec::new();
        for &token in &prompt {
            one_at_a_time = logits(&model, &mut stepped, &[token]);
        }
        let worst = at_once
            .iter()
            .zip(&one_at_a_time)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("largest logit difference: {worst:e}");
        assert!(worst < 1e-4, "{worst}");

        // The control: a token the window has passed still reaches the logits through the full
        // layers, so changing it moves them.
        let mut changed = prompt;
        changed[0] = 4;
        let mut other = model.cache(prompt.len(), DType::F32, &FlexDevice);
        let moved = logits(&model, &mut other, &changed)
            .iter()
            .zip(&at_once)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(moved > 1e-4, "{moved}");
    }
}
