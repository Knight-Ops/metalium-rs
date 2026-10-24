//! A decoder-only Llama-architecture language model, its shape a [`LlamaConfig`]: pre-norm layers
//! of grouped-query attention with rotary position embeddings and a SwiGLU MLP, the output head
//! either the embedding table (`tie_word_embeddings`) or an `lm_head`. SmolLM2-135M
//! ([`SMOLLM2_135M`]) is the checkpoint the gates load; the code is the architecture's, not that
//! model's.
//!
//! Written with Burn's own `Embedding`, `Linear` and `RmsNorm`, so it exercises what a real model
//! uses: RMSNorm, SiLU, RoPE (`sin` and `cos` computed on the device from the positions),
//! `cat`/`slice` for the rotation, and a key/value cache kept in preallocated tensors. Two caches:
//! a growing one filled with `slice_assign` at host positions ([`Llama::forward`], the eager
//! reference), and a [`FixedCache`] whose steps have the same shapes at every position
//! ([`Llama::step_fixed`], [`Llama::step_chunk`]), so they can be traced
//! ([`crate::generate::TracedGenerator`]).
//!
//! Field names follow the HuggingFace checkpoints, so only the `model.` prefix and the norms'
//! `weight` need remapping on load (`Layout::remaps`: Burn's `RmsNorm` calls its scale `gamma`, and
//! `burn-store`'s PyTorch adapter renames only batch, layer and group norms).
//!
//! A checkpoint in BF16 stays so under [`Precision::Bf16`], and every activation follows;
//! [`Precision::F32`] widens each parameter once after loading (lossless), which is what makes a
//! comparison with burn-flex a comparison of the same arithmetic on the same values.

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, Int, Tensor, TensorData};

pub use crate::generate::Generation;
use crate::generate::{argmax, FixedShapeCache, FixedShapeDecoder};
use crate::kv_cache::{CacheLayout, FixedCache, FixedWrite};
use crate::load::{load_safetensors, weights_path, Layout, LoadError, Loaded};
use crate::rope::Rope;

/// The shape of a Llama-architecture model: what differs between SmolLM2, Llama 2 and 3,
/// Mistral (bar its sliding window) and their kin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LlamaConfig {
    /// The residual stream's width.
    pub hidden: usize,
    /// Query heads, key/value heads (grouped-query attention when fewer), and each head's width.
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// The MLP's inner width.
    pub ff: usize,
    pub vocab: usize,
    pub layers: usize,
    pub rope_theta: f64,
    /// RMSNorm's epsilon.
    pub eps: f64,
    /// The output projection is the embedding table (`tie_word_embeddings`), not an `lm_head`.
    pub tied_head: bool,
}

/// SmolLM2-135M (`HuggingFaceTB/SmolLM2-135M`): 30 layers of 9 query and 3 key/value heads of 64,
/// a 49,152-token vocabulary, the head tied to the embedding.
pub const SMOLLM2_135M: LlamaConfig = LlamaConfig {
    hidden: 576,
    heads: 9,
    kv_heads: 3,
    head_dim: 64,
    ff: 1536,
    vocab: 49_152,
    layers: 30,
    rope_theta: 100_000.0,
    eps: 1e-5,
    tied_head: true,
};

/// SmolLM2-1.7B-Instruct (`HuggingFaceTB/SmolLM2-1.7B-Instruct`): 24 layers of 32 query and 32
/// key/value heads of 64 (multi-head attention), width 2048, an MLP of 8,192, SmolLM2's
/// 49,152-token vocabulary, the head tied to the embedding. Checked against the checkpoint's
/// `config.json` by `crates/tt-models/reference/llama_reference.py`
/// (`tests/llama_1_7b_host.rs`).
pub const SMOLLM2_1_7B_INSTRUCT: LlamaConfig = LlamaConfig {
    hidden: 2048,
    heads: 32,
    kv_heads: 32,
    head_dim: 64,
    ff: 8192,
    vocab: 49_152,
    layers: 24,
    rope_theta: 130_000.0,
    eps: 1e-5,
    tied_head: true,
};

impl LlamaConfig {
    /// The same architecture with its first `layers` layers.
    pub fn with_layers(self, layers: usize) -> Self {
        Self { layers, ..self }
    }

    /// Query heads that share a key/value head.
    pub fn groups(&self) -> usize {
        self.heads / self.kv_heads
    }

    fn check(&self) {
        assert!(
            self.layers >= 1 && self.heads >= 1 && self.kv_heads >= 1,
            "{self:?}"
        );
        assert!(
            self.heads % self.kv_heads == 0,
            "{} query heads do not group over {} key/value heads",
            self.heads,
            self.kv_heads
        );
        assert!(
            self.head_dim % 2 == 0,
            "a rotary head of odd width {}",
            self.head_dim
        );
    }
}

/// The dtype the model computes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// The checkpoint's own BF16: parameters, activations and the cache.
    Bf16,
    /// Parameters widened to F32 after loading.
    F32,
}

impl Precision {
    pub fn dtype(self) -> DType {
        match self {
            Precision::Bf16 => DType::BF16,
            Precision::F32 => DType::F32,
        }
    }
}

fn linear<B: Backend>(inp: usize, out: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(inp, out).with_bias(false).init(device)
}

fn rms_norm<B: Backend>(cfg: &LlamaConfig, device: &B::Device) -> RmsNorm<B> {
    RmsNormConfig::new(cfg.hidden)
        .with_epsilon(cfg.eps)
        .init(device)
}

#[derive(Module, Debug)]
pub struct SelfAttn<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    /// `q|k|v` as one `[hidden, q + 2 kv]` matrix ([`Llama::fused`]), whose column blocks the three
    /// projections above are then views of. `None` as loaded.
    qkv: Option<Linear<B>>,
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
    /// `gate|up` as one `[hidden, 2 ff]` matrix ([`Llama::fused`]), as `qkv` is.
    gate_up: Option<Linear<B>>,
}

/// The projections `parts` as one matrix, their weights concatenated along the outputs, and each
/// part's weight replaced by its column block of it (a view where the backend makes slices views,
/// so the parts and the whole share storage). `None`, and `parts` untouched, unless every part is
/// the same dtype and without a bias.
fn fuse<B: Backend>(parts: &mut [&mut Linear<B>]) -> Option<Linear<B>> {
    let dtype = parts[0].weight.val().dtype();
    if parts
        .iter()
        .any(|p| p.bias.is_some() || p.weight.val().dtype() != dtype)
    {
        return None;
    }
    let whole = Tensor::cat(parts.iter().map(|p| p.weight.val()).collect(), 1);
    let [rows, _] = whole.dims();
    let mut at = 0;
    for p in parts.iter_mut() {
        let [_, cols] = p.weight.val().dims();
        p.weight = Param::from_tensor(whole.clone().slice([0..rows, at..at + cols]));
        at += cols;
    }
    Some(Linear {
        weight: Param::from_tensor(whole),
        bias: None,
    })
}

/// `x @ W` for the fused `whole`, split back into the parts' outputs, `widths` columns each.
fn project_split<B: Backend, const N: usize>(
    x: Tensor<B, 3>,
    whole: &Linear<B>,
    widths: [usize; N],
) -> [Tensor<B, 3>; N] {
    let y = project(x, whole);
    let [n, t, _] = y.dims();
    let mut at = 0;
    widths.map(|w| {
        let part = y.clone().slice([0..n, 0..t, at..at + w]);
        at += w;
        part
    })
}

#[derive(Module, Debug)]
pub struct Layer<B: Backend> {
    input_layernorm: RmsNorm<B>,
    self_attn: SelfAttn<B>,
    post_attention_layernorm: RmsNorm<B>,
    mlp: Mlp<B>,
}

/// Each key/value head repeated for its group of query heads (`repeat_interleave`).
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, groups: usize) -> Tensor<B, 4> {
    let [n, kv, l, d] = x.dims();
    let mut heads = Vec::with_capacity(kv * groups);
    for h in 0..kv {
        let one = x.clone().slice([0..n, h..h + 1, 0..l, 0..d]);
        heads.extend((0..groups).map(|_| one.clone()));
    }
    Tensor::cat(heads, 1)
}

/// A named activation boundary: `"layers.3.mlp_hidden"` and `"layers.3.attn_ctx"` pass through it,
/// so a precision plan can hold an activation in another format ([`crate::quant`]).
pub type ActFn<B> = std::sync::Arc<dyn Fn(&str, Tensor<B, 3>) -> Tensor<B, 3> + Send + Sync>;

/// `x @ W` for a projection, computed in the weight's own dtype: the activation is cast in and the
/// result cast back to the residual stream's dtype, so matrices held in different formats (F32,
/// BF16, F16, block floating point) share one stream.
fn project<B: Backend>(x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 3> {
    let stream = x.dtype();
    let weight = linear.weight.val().dtype();
    if weight == stream {
        linear.forward(x)
    } else {
        linear.forward(x.cast(weight)).cast(stream)
    }
}

impl<B: Backend> Layer<B> {
    fn new(cfg: &LlamaConfig, device: &B::Device) -> Self {
        let (h, q, kv) = (
            cfg.hidden,
            cfg.heads * cfg.head_dim,
            cfg.kv_heads * cfg.head_dim,
        );
        Self {
            input_layernorm: rms_norm(cfg, device),
            self_attn: SelfAttn {
                q_proj: linear(h, q, device),
                k_proj: linear(h, kv, device),
                v_proj: linear(h, kv, device),
                o_proj: linear(q, h, device),
                qkv: None,
            },
            post_attention_layernorm: rms_norm(cfg, device),
            mlp: Mlp {
                gate_proj: linear(h, cfg.ff, device),
                up_proj: linear(h, cfg.ff, device),
                down_proj: linear(cfg.ff, h, device),
                gate_up: None,
            },
        }
    }

    /// [`Llama::fused`] for one layer.
    fn fuse(&mut self) {
        let a = &mut self.self_attn;
        if a.qkv.is_none() {
            a.qkv = fuse(&mut [&mut a.q_proj, &mut a.k_proj, &mut a.v_proj]);
        }
        let m = &mut self.mlp;
        if m.gate_up.is_none() {
            m.gate_up = fuse(&mut [&mut m.gate_proj, &mut m.up_proj]);
        }
    }

    /// The layer's output, and its cache's keys and values (`cache_k`, `cache_v`: `[1, kv_heads,
    /// max_len, head_dim]` for a growing cache, as a [`FixedCache`] stores them for a fixed one)
    /// with this step's written as `write` says.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        cfg: &LlamaConfig,
        x: Tensor<B, 3>,
        rope: &Rope<B>,
        cache_k: &Tensor<B, 4>,
        cache_v: &Tensor<B, 4>,
        write: &Write<'_, B>,
        index: usize,
        act: Option<&ActFn<B>>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let hook = |name: &str, t: Tensor<B, 3>| match act {
            Some(f) => f(&format!("layers.{index}.{name}"), t),
            None => t,
        };
        let (kv_heads, head_dim, groups) = (cfg.kv_heads, cfg.head_dim, cfg.groups());
        let [n, t, _] = x.dims();
        let h = self.input_layernorm.forward(x.clone());
        let a = &self.self_attn;
        let heads =
            |p: Tensor<B, 3>, count: usize| p.reshape([n, t, count, head_dim]).swap_dims(1, 2);
        let (qw, kvw) = (cfg.heads * head_dim, kv_heads * head_dim);
        let [q, k, v] = match &a.qkv {
            Some(qkv) => project_split(h, qkv, [qw, kvw, kvw]),
            None => [
                project(h.clone(), &a.q_proj),
                project(h.clone(), &a.k_proj),
                project(h, &a.v_proj),
            ],
        };
        let q = rope.apply(heads(q, cfg.heads));
        let k = rope.apply(heads(k, kv_heads));
        let v = heads(v, kv_heads);

        let (q, new_k, new_v, keys, values, mask, is_causal) = match write {
            Write::At(start) => {
                let total = start + t;
                let new_k = cache_k
                    .clone()
                    .slice_assign([0..n, 0..kv_heads, *start..total, 0..head_dim], k);
                let new_v = cache_v
                    .clone()
                    .slice_assign([0..n, 0..kv_heads, *start..total, 0..head_dim], v);
                let keys = repeat_kv(
                    new_k
                        .clone()
                        .slice([0..n, 0..kv_heads, 0..total, 0..head_dim]),
                    groups,
                );
                let values = repeat_kv(
                    new_v
                        .clone()
                        .slice([0..n, 0..kv_heads, 0..total, 0..head_dim]),
                    groups,
                );
                // A prompt attends causally to itself; one new token attends to everything cached.
                (q, new_k, new_v, keys, values, None, t > 1)
            }
            Write::Fixed { write, mask } => {
                // The new keys and values land where the step writes, every other position kept;
                // each query attends to every position not later than its own. Grouped-query
                // attention reads the cache's own key/value heads: each group of query heads is
                // folded into the rows (query head `h groups + g` is `repeat_kv`'s order, its row
                // `i` is row `g t + i`), so nothing repeats the cache, and each score and each sum
                // is the one `repeat_kv`'s attention computes.
                let new_k = write.write(cache_k.clone(), k);
                let new_v = write.write(cache_v.clone(), v);
                let q = q.reshape([n, kv_heads, groups * t, head_dim]);
                let (keys, values) = (write.attended(new_k.clone()), write.attended(new_v.clone()));
                (q, new_k, new_v, keys, values, Some(mask.clone()), false)
            }
        };
        let out = attention(
            q,
            keys,
            values,
            mask,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal,
            },
        );
        // Unfolded (a no-op unless the query heads were folded into rows above).
        let out = out.reshape([n, cfg.heads, t, head_dim]);
        let context = hook(
            "attn_ctx",
            out.swap_dims(1, 2).reshape([n, t, cfg.heads * head_dim]),
        );
        let x = x + project(context, &a.o_proj);

        let h = self.post_attention_layernorm.forward(x.clone());
        let m = &self.mlp;
        let [gate, up] = match &m.gate_up {
            Some(gate_up) => project_split(h, gate_up, [cfg.ff, cfg.ff]),
            None => [project(h.clone(), &m.gate_proj), project(h, &m.up_proj)],
        };
        let hidden = hook("mlp_hidden", silu(gate) * up);
        (x + project(hidden, &m.down_proj), new_k, new_v)
    }
}

/// One layer's keys and values, `[batch, kv_heads, max_len, head_dim]`, filled front to back.
struct LayerCache<B: Backend> {
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
}

/// How a step puts its keys and values into a layer's cache, and which of them it attends to.
enum Write<'a, B: Backend> {
    /// At positions `start..start + t`, known on the host; the query attends to positions
    /// `..start + t` (causally within the new ones).
    At(usize),
    /// Where a [`FixedCache`] step writes: positions given as tensors, so the step's shapes do not
    /// depend on them.
    Fixed {
        write: &'a FixedWrite<B>,
        /// `write`'s attention mask with each row repeated for the query heads folded into it
        /// (`[1, 1, groups t, max_len]`): made once a step.
        mask: Tensor<B, 4, burn::tensor::Bool>,
    },
}

/// A [`FixedCache`] and the rotary table its steps select their rows from: what
/// [`Llama::step_fixed`] and [`Llama::step_chunk`] read.
#[derive(Clone, Debug)]
pub struct LlamaCache<B: Backend> {
    kv: FixedCache<B>,
    rope: Rope<B>,
}

impl<B: Backend> LlamaCache<B> {
    pub fn max_len(&self) -> usize {
        self.kv.max_len()
    }

    /// Every layer's keys and values, in layer order: the state a traced step writes back.
    pub fn state(&self) -> Vec<Tensor<B, 4>> {
        self.kv.state()
    }

    /// Replace the state with what a step returned.
    pub fn set_state(&mut self, state: Vec<Tensor<B, 4>>) {
        self.kv.set_state(state)
    }

    pub fn kv(&self) -> &FixedCache<B> {
        &self.kv
    }
}

impl<B: Backend> FixedShapeCache<B> for LlamaCache<B> {
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

/// The keys and values of every position a model has seen.
pub struct KvCache<B: Backend> {
    layers: Vec<LayerCache<B>>,
    len: usize,
    max_len: usize,
    act: Option<ActFn<B>>,
}

impl<B: Backend> KvCache<B> {
    /// Pass every named activation of the run through `act`.
    pub fn with_activations(mut self, act: ActFn<B>) -> Self {
        self.act = Some(act);
        self
    }

    /// Positions already cached.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Module, Debug)]
pub struct Llama<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<Layer<B>>,
    norm: RmsNorm<B>,
    /// The output projection, absent when it is the embedding table ([`LlamaConfig::tied_head`]).
    lm_head: Option<Linear<B>>,
    /// Not a parameter: the module derive skips it.
    config: LlamaConfig,
}

/// Widens every float parameter to F32.
struct ToF32;

impl<B: Backend> ModuleMapper<B> for ToF32 {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        param.map(|t| t.cast(DType::F32))
    }
}

impl<B: Backend> Llama<B> {
    /// The architecture `config` describes, with its initialiser's weights.
    pub fn new(config: &LlamaConfig, device: &B::Device) -> Self {
        config.check();
        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab, config.hidden).init(device),
            layers: (0..config.layers)
                .map(|_| Layer::new(config, device))
                .collect(),
            norm: rms_norm(config, device),
            lm_head: (!config.tied_head).then(|| linear(config.hidden, config.vocab, device)),
            config: *config,
        }
    }

    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// The same model with each layer's `q|k|v` and `gate|up` projections one matrix each, their
    /// outputs split back by column views: three products and two become one each, so a decode
    /// step's narrow `m = 1` products spread over more of the card. Every output column keeps its
    /// own accumulation over the hidden dimension, so the logits are the unfused model's **bit for
    /// bit** (`step191_fused_projections`). The separate weights become views of the fused one,
    /// which keeps parameter paths (and a later save) as loaded. A group whose matrices differ in
    /// dtype stays unfused. Not for a model a [`crate::quant`] plan rewrites, before or after: its
    /// block-float storage is invisible here (the dtype is F32), and its rules name the separate
    /// matrices.
    pub fn fused(mut self) -> Self {
        for layer in &mut self.layers {
            layer.fuse();
        }
        self
    }

    /// A pretrained Llama-architecture checkpoint (`cargo xtask fetch-weights --model <weights>`)
    /// whose shape `config` gives, keeping its first `config.layers` layers, computing in
    /// `precision`. Every tensor the kept layers need must be in the file, and the file must hold
    /// nothing else but the dropped layers.
    pub fn pretrained(
        config: &LlamaConfig,
        weights: &str,
        precision: Precision,
        device: &B::Device,
    ) -> Result<(Self, Loaded), LoadError> {
        let mut model = Self::new(config, device);
        let layers = config.layers;
        let dropped = move |name: &str| {
            name.strip_prefix("layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|i| i.parse::<usize>().ok())
                .is_some_and(|i| i >= layers)
        };
        let loaded = load_safetensors(
            &mut model,
            &weights_path(weights),
            &Layout {
                remaps: &[
                    (
                        r"^model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight$",
                        "layers.$1.$2.gamma",
                    ),
                    (r"^model\.norm\.weight$", "norm.gamma"),
                    (r"^model\.", ""),
                ],
                pytorch: true,
                ignore: &dropped,
            },
        )?;
        let model = match precision {
            Precision::Bf16 => model,
            Precision::F32 => model.map(&mut ToF32),
        };
        Ok((model, loaded))
    }

    /// SmolLM2-1.7B-Instruct ([`SMOLLM2_1_7B_INSTRUCT`], `cargo xtask fetch-weights --model
    /// smollm2-1.7b-instruct`), its first `layers` layers of twenty-four.
    pub fn smollm2_1_7b_instruct(
        layers: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Result<(Self, Loaded), LoadError> {
        Self::pretrained(
            &SMOLLM2_1_7B_INSTRUCT.with_layers(layers),
            "smollm2-1.7b-instruct",
            precision,
            device,
        )
    }

    /// SmolLM2-135M ([`SMOLLM2_135M`], `cargo xtask fetch-weights --model smollm2-135m`), its
    /// first `layers` layers of thirty.
    pub fn smollm2_135m(
        layers: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Result<(Self, Loaded), LoadError> {
        Self::pretrained(
            &SMOLLM2_135M.with_layers(layers),
            "smollm2-135m",
            precision,
            device,
        )
    }

    /// The embedding rows of `tokens` `[1, t]`, as `[1, t, hidden]`.
    pub fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed_tokens.forward(tokens)
    }

    /// An empty cache for `max_len` positions of one sequence.
    pub fn cache(&self, max_len: usize, dtype: DType, device: &B::Device) -> KvCache<B> {
        let shape = [1, self.config.kv_heads, max_len, self.config.head_dim];
        KvCache {
            layers: self
                .layers
                .iter()
                .map(|_| LayerCache {
                    k: Tensor::zeros(shape, (device, dtype)),
                    v: Tensor::zeros(shape, (device, dtype)),
                })
                .collect(),
            len: 0,
            max_len,
            act: None,
        }
    }

    /// The logits `[1, vocab]` after the last of `tokens` (`[1, t]`), which are appended to
    /// `cache`. Either a prompt into an empty cache, or one token into a filled one.
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: &mut KvCache<B>,
        dtype: DType,
    ) -> Tensor<B, 2> {
        let x = self.embed_tokens.forward(tokens);
        self.forward_embedded(x, cache, dtype)
    }

    /// [`Self::forward`] from the embedded tokens `[1, t, hidden]`: everything after the table
    /// lookup. It exists to time the model apart from the lookup (`silicon_bench_models`).
    pub fn forward_embedded(
        &self,
        mut x: Tensor<B, 3>,
        cache: &mut KvCache<B>,
        dtype: DType,
    ) -> Tensor<B, 2> {
        let [n, t, _] = x.dims();
        assert_eq!(n, 1, "one sequence at a time");
        let start = cache.len;
        assert!(
            start == 0 || t == 1,
            "a cache holding {start} positions takes one token, not {t}"
        );
        assert!(start + t <= cache.max_len, "the cache is full");
        let device = x.device();

        if x.dtype() != dtype {
            x = x.cast(dtype);
        }
        let cfg = &self.config;
        let rope = Rope::new(start, t, cfg.head_dim, cfg.rope_theta, dtype, &device);
        let act = cache.act.clone();
        let write = Write::At(start);
        for (index, (layer, c)) in self.layers.iter().zip(cache.layers.iter_mut()).enumerate() {
            let (y, k, v) = layer.forward(cfg, x, &rope, &c.k, &c.v, &write, index, act.as_ref());
            (x, c.k, c.v) = (y, k, v);
        }
        cache.len += t;
        self.head(x, t)
    }

    /// An empty [`LlamaCache`] for `max_len` positions, in the layout the heads take
    /// ([`CacheLayout::for_heads`]).
    pub fn fixed_cache(&self, max_len: usize, dtype: DType, device: &B::Device) -> LlamaCache<B> {
        let cfg = &self.config;
        let layout = CacheLayout::for_heads(cfg.kv_heads, cfg.head_dim);
        self.fixed_cache_in(max_len, layout, dtype, device)
    }

    /// An empty [`LlamaCache`] for `max_len` positions, laid out as `layout` says (of the model's
    /// key/value heads).
    pub fn fixed_cache_in(
        &self,
        max_len: usize,
        layout: CacheLayout,
        dtype: DType,
        device: &B::Device,
    ) -> LlamaCache<B> {
        let cfg = &self.config;
        assert_eq!(
            layout.heads(),
            (cfg.kv_heads, cfg.head_dim),
            "a layout of the model's key/value heads"
        );
        LlamaCache {
            kv: FixedCache::zeros_in(cfg.layers, layout, max_len, dtype, device),
            rope: Rope::table(max_len, cfg.head_dim, cfg.rope_theta, dtype, device),
        }
    }

    /// One decode step whose shapes do not depend on its position: the embedded token `x`
    /// (`[1, 1, hidden]`) at `position` (`[1]`, below `cache.max_len()`), its keys and values
    /// written into `cache` there, the logits `[1, vocab]` returned. The position is a tensor,
    /// so the step can be captured once and replayed at every position
    /// ([`crate::generate::TracedGenerator`]).
    pub fn step_fixed(
        &self,
        x: Tensor<B, 3>,
        position: Tensor<B, 1, Int>,
        cache: &mut LlamaCache<B>,
        dtype: DType,
    ) -> Tensor<B, 2> {
        let (logits, state) = self.fixed_step(x, position, cache, dtype);
        cache.set_state(state);
        logits
    }

    /// [`Self::step_fixed`] that returns the new state instead of writing it.
    pub fn fixed_step(
        &self,
        x: Tensor<B, 3>,
        position: Tensor<B, 1, Int>,
        cache: &LlamaCache<B>,
        dtype: DType,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        self.fixed_step_within(x, position, cache.max_len(), cache, dtype)
    }

    /// [`Self::fixed_step`] attending only to the cache's first `len` positions (`position` below
    /// `len`): the same logits, at a cost that follows `len` (`FixedCache::step_write_within`).
    pub fn fixed_step_within(
        &self,
        mut x: Tensor<B, 3>,
        position: Tensor<B, 1, Int>,
        len: usize,
        cache: &LlamaCache<B>,
        dtype: DType,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        assert_eq!(x.dims(), [1, 1, self.config.hidden], "one token a step");
        if x.dtype() != dtype {
            x = x.cast(dtype);
        }
        let write = cache.kv.step_write_within(position.clone(), len);
        let rope = cache.rope.at(position);
        let (x, state) = self.layers_fixed(x, &rope, cache, &write);
        (self.head(x, 1), state)
    }

    /// `c` tokens of a prompt (`x`, `[1, c, hidden]`) from position `start` (`[1]`) on, with shapes
    /// that depend on neither: their keys and values written into `cache` at `start..start + c`,
    /// the logits `[1, vocab]` of chunk row `last` (`[1]`) returned. A prompt is its chunks in
    /// order, the last one padded (padding tokens write cache rows past the prompt, which a decode
    /// step overwrites before it reads them, and their rotary rows are clamped into the table);
    /// `last` is the prompt's last token within it. Like [`Self::step_fixed`], it can be captured
    /// once for a chunk size and replayed at every start, and each row's arithmetic is
    /// [`Self::forward_embedded`]'s.
    pub fn step_chunk(
        &self,
        x: Tensor<B, 3>,
        start: Tensor<B, 1, Int>,
        last: Tensor<B, 1, Int>,
        cache: &mut LlamaCache<B>,
        dtype: DType,
    ) -> Tensor<B, 2> {
        let (logits, state) = self.fixed_chunk(x, start, last, cache, dtype);
        cache.set_state(state);
        logits
    }

    /// [`Self::step_chunk`] that returns the new state instead of writing it.
    pub fn fixed_chunk(
        &self,
        x: Tensor<B, 3>,
        start: Tensor<B, 1, Int>,
        last: Tensor<B, 1, Int>,
        cache: &LlamaCache<B>,
        dtype: DType,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        self.fixed_chunk_within(x, start, last, cache.max_len(), cache, dtype)
    }

    /// [`Self::fixed_chunk`] attending only to the cache's first `len` positions (every row of
    /// the chunk below `len`): the same logits, at a cost that follows `len`
    /// (`FixedCache::chunk_write_within`).
    pub fn fixed_chunk_within(
        &self,
        mut x: Tensor<B, 3>,
        start: Tensor<B, 1, Int>,
        last: Tensor<B, 1, Int>,
        len: usize,
        cache: &LlamaCache<B>,
        dtype: DType,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        let [n, c, h] = x.dims();
        let hidden = self.config.hidden;
        let max_len = cache.max_len();
        assert!(
            n == 1 && h == hidden && c <= len && len <= max_len,
            "a chunk of {c} tokens attending to {len} of {max_len} positions"
        );
        if x.dtype() != dtype {
            x = x.cast(dtype);
        }
        let (write, own) = cache.kv.chunk_write_within(start, c, len);
        let rope = cache.rope.at(own.clamp_max(max_len as i64 - 1));
        let (x, state) = self.layers_fixed(x, &rope, cache, &write);
        let row = self
            .norm
            .forward(x)
            .reshape([c, hidden])
            .select(0, last)
            .reshape([1, hidden]);
        (self.project_out(row), state)
    }

    /// Every layer over a [`LlamaCache`] step: the last layer's output and the new state.
    fn layers_fixed(
        &self,
        mut x: Tensor<B, 3>,
        rope: &Rope<B>,
        cache: &LlamaCache<B>,
        write: &FixedWrite<B>,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 4>>) {
        let groups = self.config.groups();
        let later = write.later();
        // A query row of every folded head shares its position's mask row (`Layer::forward`).
        let mask = if groups == 1 || later.dims()[2] == 1 {
            later
        } else {
            later.repeat_dim(2, groups)
        };
        let write = Write::Fixed { write, mask };
        let mut state = Vec::with_capacity(2 * self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let (k, v) = cache.kv.layer(index);
            let (y, k, v) = layer.forward(&self.config, x, rope, &k, &v, &write, index, None);
            x = y;
            state.extend([k, v]);
        }
        (x, state)
    }

    /// The logits `[1, vocab]` of the last of `t` positions of `x`.
    fn head(&self, x: Tensor<B, 3>, t: usize) -> Tensor<B, 2> {
        let hidden = self.config.hidden;
        let last = self
            .norm
            .forward(x)
            .slice([0..1, t - 1..t, 0..hidden])
            .reshape([1, hidden]);
        self.project_out(last)
    }

    /// The logits `[1, vocab]` of a normed hidden row `[1, hidden]`.
    fn project_out(&self, last: Tensor<B, 2>) -> Tensor<B, 2> {
        if let Some(head) = &self.lm_head {
            let weight = head.weight.val().dtype();
            let x = if weight == last.dtype() {
                last
            } else {
                last.cast(weight)
            };
            return head.forward(x);
        }
        // The output projection is the embedding table, transposed, computed in its dtype.
        let table = self.embed_tokens.weight.val();
        let last = if table.dtype() == last.dtype() {
            last
        } else {
            last.cast(table.dtype())
        };
        last.matmul(table.transpose())
    }
}

impl<B: Backend> FixedShapeDecoder<B> for Llama<B> {
    type Cache = LlamaCache<B>;

    /// A cache in the model's own dtype.
    fn new_cache(&self, max_len: usize, device: &B::Device) -> LlamaCache<B> {
        self.fixed_cache(max_len, self.embed_tokens.weight.val().dtype(), device)
    }

    fn chunk(
        &self,
        tokens: Tensor<B, 2, Int>,
        start: Tensor<B, 1, Int>,
        last: Tensor<B, 1, Int>,
        len: usize,
        cache: &LlamaCache<B>,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        let x = self.embed(tokens);
        self.fixed_chunk_within(x, start, last, len, cache, cache.kv.dtype())
    }

    fn step(
        &self,
        token: Tensor<B, 2, Int>,
        position: Tensor<B, 1, Int>,
        len: usize,
        cache: &LlamaCache<B>,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        let x = self.embed(token);
        self.fixed_step_within(x, position, len, cache, cache.kv.dtype())
    }
}

fn ids<B: Backend>(tokens: &[i32], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(tokens.to_vec(), [1, tokens.len()]), device)
}

impl<B: Backend> Llama<B> {
    /// Greedy decoding: `steps` tokens after `prompt`, each chosen from the previous step's logits.
    pub fn generate(
        &self,
        prompt: &[i32],
        steps: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Generation {
        self.generate_with(prompt, steps, precision, None, device)
    }

    /// [`Self::generate`] with named activations passed through `act`.
    pub fn generate_with(
        &self,
        prompt: &[i32],
        steps: usize,
        precision: Precision,
        act: Option<ActFn<B>>,
        device: &B::Device,
    ) -> Generation {
        assert!(steps >= 1);
        let dtype = precision.dtype();
        let mut cache = self.cache(prompt.len() + steps, dtype, device);
        if let Some(act) = act {
            cache = cache.with_activations(act);
        }
        let mut out = Generation {
            times: Vec::new(),
            tokens: Vec::with_capacity(steps),
            logits: Vec::with_capacity(steps),
        };
        out.step(|| self.forward(ids(prompt, device), &mut cache, dtype));
        for step in 0..steps {
            let next = argmax(out.logits.last().expect("one prediction per step")) as i32;
            out.tokens.push(next);
            if step + 1 < steps {
                out.step(|| self.forward(ids(&[next], device), &mut cache, dtype));
            }
        }
        out
    }

    /// The predictions after `prompt` and after each prefix of `continuation`, feeding the
    /// continuation's own tokens, so two backends are compared on the same inputs at every step
    /// however their own argmax would have gone.
    pub fn forced(
        &self,
        prompt: &[i32],
        continuation: &[i32],
        precision: Precision,
        device: &B::Device,
    ) -> Generation {
        self.forced_with(prompt, continuation, precision, None, device)
    }

    /// [`Self::forced`] with named activations passed through `act`.
    pub fn forced_with(
        &self,
        prompt: &[i32],
        continuation: &[i32],
        precision: Precision,
        act: Option<ActFn<B>>,
        device: &B::Device,
    ) -> Generation {
        let dtype = precision.dtype();
        let mut cache = self.cache(prompt.len() + continuation.len() + 1, dtype, device);
        if let Some(act) = act {
            cache = cache.with_activations(act);
        }
        let mut out = Generation {
            times: Vec::new(),
            tokens: continuation.to_vec(),
            logits: Vec::with_capacity(continuation.len() + 1),
        };
        out.step(|| self.forward(ids(prompt, device), &mut cache, dtype));
        for &token in continuation {
            out.step(|| self.forward(ids(&[token], device), &mut cache, dtype));
        }
        out
    }
}
