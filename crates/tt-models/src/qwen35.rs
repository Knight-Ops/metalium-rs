//! The Qwen3.5 text model: a decoder whose layers alternate two token mixers. Most layers are a
//! **Gated DeltaNet** (linear attention): a short causal convolution, then a recurrent state per
//! head, `[k_dim, v_dim]`, decayed and corrected by the delta rule at every token, so a layer
//! carries a fixed-size state rather than a cache that grows with the sequence. Every
//! [`Qwen35Config::full_every`]th layer is full softmax attention with q/k RMSNorm, rotary
//! embeddings on part of each head, and a sigmoid **output gate** that the query projection
//! computes beside the queries. The MLP is SwiGLU. `Qwen/Qwen3.5-4B` ([`QWEN3_5_4B`]) is the
//! checkpoint the gates load; the code is the architecture's, which the MOSS-Image text encoder
//! shares at another width.
//!
//! The reference is HuggingFace `transformers`' `modeling_qwen3_5.py` (5.19.0), and these are its
//! details the configuration does not state:
//!
//! * The layer norms are zero-centred: the scale is `1 + weight`, applied in F32
//!   ([`CenteredNorm`]). The DeltaNet's output norm is not: its scale is `weight`, and its result
//!   is gated by `silu(z)` ([`GatedNorm`]).
//! * `q_proj` emits `2 head_dim` per head, the query then the gate, interleaved by head.
//! * The rotary embedding is multimodal (mRoPE), but for text the three position streams are equal,
//!   so it is the ordinary rotate-half embedding over the first `rotary_dim` of each head's
//!   dimensions with frequencies `theta^(-2i / rotary_dim)`, the rest passing through.
//! * The DeltaNet's queries and keys are L2-normalised in F32 (`x rsqrt(sum x^2 + 1e-6)`), the
//!   queries then scaled by `k_dim^-0.5`. The decay is `g = -exp(A_log) softplus(a + dt_bias)`
//!   (log space) and the write strength `beta = sigmoid(b)`. Per token, per head:
//!   `S <- exp(g) S`, `S <- S + k (beta (v - k^T S))^T`, `o = q^T S`. The state is F32
//!   (`mamba_ssm_dtype`).
//! * The key heads are repeated (`repeat_interleave`) to the value heads' count.
//!
//! The recurrence is computed a token at a time, for a prompt as for a decode step. The reference
//! computes a prompt in chunks of 64 (the WY form, `torch_chunk_gated_delta_rule`), which is the
//! same function summed in another order.
//!
//! Text only: the checkpoint's vision tower (`model.visual.*`) and its multi-token-prediction head
//! (`mtp.*`) are left in the file, by name ([`Qwen35::pretrained`]).

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, silu};
use burn::tensor::backend::Backend;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Bool, DType, Distribution, Int, Tensor, TensorData};

use crate::generate::{eager, EagerDecoder, FixedShapeCache, FixedShapeDecoder, Generation};
use crate::kv_cache::{CacheLayout, FixedCache, FixedWrite};
use crate::llama::{project, repeat_kv, Precision};
use crate::load::{load_safetensors_shards, weights_files, Layout, LoadError, Loaded};
use crate::rope::Rope;

/// The shape of a Qwen3.5 text model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35Config {
    /// The residual stream's width.
    pub hidden: usize,
    /// Full attention: query heads, key/value heads, each head's width, and how many of its
    /// leading dimensions are rotated.
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f64,
    /// Gated DeltaNet: key heads and their width, value heads and theirs, and the convolution's
    /// width in tokens.
    pub linear_k_heads: usize,
    pub linear_k_dim: usize,
    pub linear_v_heads: usize,
    pub linear_v_dim: usize,
    pub conv_kernel: usize,
    /// Layer `i` is full attention when `(i + 1) % full_every == 0`, else a DeltaNet.
    pub full_every: usize,
    /// The MLP's inner width.
    pub ff: usize,
    pub vocab: usize,
    pub layers: usize,
    /// Every RMSNorm's epsilon.
    pub eps: f64,
    /// The output projection is the embedding table (`tie_word_embeddings`).
    pub tied_head: bool,
}

/// Qwen3.5-4B (`Qwen/Qwen3.5-4B`, text): 32 layers of width 2560, every fourth full attention
/// (16 query and 4 key/value heads of 256, 64 of them rotated), the others DeltaNets (16 key heads
/// and 32 value heads of 128, a 4-token convolution), an MLP of 9,216, a 248,320-token vocabulary,
/// the head tied to the embedding. Checked against the checkpoint's `config.json` by
/// `crates/tt-models/reference/qwen35_reference.py` (`tests/qwen35_host.rs`).
pub const QWEN3_5_4B: Qwen35Config = Qwen35Config {
    hidden: 2560,
    heads: 16,
    kv_heads: 4,
    head_dim: 256,
    rotary_dim: 64,
    rope_theta: 10_000_000.0,
    linear_k_heads: 16,
    linear_k_dim: 128,
    linear_v_heads: 32,
    linear_v_dim: 128,
    conv_kernel: 4,
    full_every: 4,
    ff: 9216,
    vocab: 248_320,
    layers: 32,
    eps: 1e-6,
    tied_head: true,
};

/// MOSS-Image-Base's text encoder (`text_encoder/` of `OpenMOSS-Team/MOSS-Image-Base`, Qwen3.5-9B's
/// shape): [`QWEN3_5_4B`]'s heads and layer pattern at width 4096 with an MLP of 12,288. It is read
/// for its last hidden states, so it has no output head ([`Qwen35::text_encoder`]); `tied_head` is
/// the checkpoint's `false`. Checked against its `config.json` by
/// `crates/tt-models/reference/moss_image_reference.py` (`tests/moss_image_host.rs`).
pub const MOSS_TEXT_ENCODER: Qwen35Config = Qwen35Config {
    hidden: 4096,
    ff: 12288,
    tied_head: false,
    ..QWEN3_5_4B
};

impl Qwen35Config {
    /// The same architecture with its first `layers` layers.
    pub fn with_layers(self, layers: usize) -> Self {
        Self { layers, ..self }
    }

    /// Whether layer `index` is full attention (else a Gated DeltaNet).
    pub fn is_full(&self, index: usize) -> bool {
        (index + 1) % self.full_every == 0
    }

    /// The DeltaNet's query and key width, and its value width (all heads).
    fn key_dim(&self) -> usize {
        self.linear_k_heads * self.linear_k_dim
    }

    fn value_dim(&self) -> usize {
        self.linear_v_heads * self.linear_v_dim
    }

    /// The convolution's channels: queries, keys and values.
    fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    fn check(&self) {
        assert!(self.layers >= 1 && self.full_every >= 1, "{self:?}");
        assert!(
            self.heads % self.kv_heads == 0,
            "{} query heads do not group over {} key/value heads",
            self.heads,
            self.kv_heads
        );
        assert!(
            self.linear_v_heads % self.linear_k_heads == 0,
            "{} value heads do not group over {} key heads",
            self.linear_v_heads,
            self.linear_k_heads
        );
        assert!(
            self.rotary_dim % 2 == 0 && self.rotary_dim <= self.head_dim,
            "rotating {} of {} dimensions",
            self.rotary_dim,
            self.head_dim
        );
        assert!(self.conv_kernel >= 1, "{self:?}");
    }
}

fn linear<B: Backend>(inp: usize, out: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(inp, out).with_bias(false).init(device)
}

/// `x rsqrt(mean(x^2) + eps)` over the last dimension, in F32.
pub(crate) fn rms<B: Backend, const D: usize>(x: Tensor<B, D>, eps: f64) -> Tensor<B, D> {
    let x = x.cast(DType::F32);
    let mean_square = (x.clone() * x.clone()).mean_dim(D - 1);
    x * mean_square.add_scalar(eps).sqrt().recip()
}

/// A zero-centred RMSNorm: `rms(x) (1 + weight)` in F32, returned in `x`'s dtype. On the card
/// one op (`burn_tt::rms_norm`, `x / sqrt(mean(x x) + eps) (1 + weight)`) where it takes the row.
#[derive(Module, Debug)]
pub struct CenteredNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    eps: f64,
    /// `weight` already holds `1 + weight` in F32 ([`Qwen35::prepared`]): the sum made once, the
    /// same F32 addition a step would make.
    shifted: bool,
}

impl<B: Backend> CenteredNorm<B> {
    fn new(width: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([width], device)),
            eps,
            shifted: false,
        }
    }

    /// `1 + weight`, in F32.
    fn scale(&self) -> Tensor<B, 1> {
        let w = self.weight.val().cast(DType::F32);
        if self.shifted {
            w
        } else {
            w.add_scalar(1.0)
        }
    }

    fn shift(&mut self) {
        if !self.shifted {
            self.weight = Param::from_tensor(self.scale());
            self.shifted = true;
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = x.dtype();
        let scale = self.scale();
        if let Some(y) = burn_tt::rms_norm(&x, &scale, self.eps) {
            return y;
        }
        (rms(x, self.eps) * scale.unsqueeze::<D>()).cast(dtype)
    }
}

/// The DeltaNet's output norm: `rms(x) weight silu(z)`, the RMS in F32, the scale in `x`'s dtype.
#[derive(Module, Debug)]
pub struct GatedNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> GatedNorm<B> {
    fn new(width: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([width], device)),
            eps,
        }
    }

    fn forward(&self, x: Tensor<B, 4>, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let dtype = x.dtype();
        let weight = self.weight.val();
        // On the card: the scaled norm as one op (`burn_tt::rms_norm`), then the gate as
        // `silu(z) * normed`, the chain `Tt`'s fuser runs as one op.
        if dtype == DType::F32 && weight.dtype() == DType::F32 {
            if let Some(normed) = burn_tt::rms_norm(&x, &weight, self.eps) {
                return silu(z.cast(DType::F32)) * normed;
            }
        }
        let normed = rms(x, self.eps).cast(dtype) * weight.cast(dtype).unsqueeze::<4>();
        (normed.cast(DType::F32) * silu(z.cast(DType::F32))).cast(dtype)
    }
}

/// Full attention with q/k norms, partial rotary embeddings and a sigmoid output gate.
#[derive(Module, Debug)]
pub struct GatedAttention<B: Backend> {
    /// `[hidden, heads 2 head_dim]`: each head's query, then its gate.
    q_proj: Option<Linear<B>>,
    k_proj: Option<Linear<B>>,
    v_proj: Option<Linear<B>>,
    /// `q|k|v` as one `[hidden, 2 q + 2 kv]` matrix, the three then `None` ([`Qwen35::prepared`]).
    qkv: Option<Linear<B>>,
    o_proj: Linear<B>,
    q_norm: CenteredNorm<B>,
    k_norm: CenteredNorm<B>,
}

/// A Gated DeltaNet: linear attention with a recurrent state per value head.
#[derive(Module, Debug)]
pub struct GatedDeltaNet<B: Backend> {
    /// `[hidden, 2 key_dim + value_dim]`: queries, keys and values, before the convolution.
    in_proj_qkv: Option<Linear<B>>,
    /// The output gate, `[hidden, value_dim]`.
    in_proj_z: Option<Linear<B>>,
    /// The write strength and the decay, one per value head.
    in_proj_b: Option<Linear<B>>,
    in_proj_a: Option<Linear<B>>,
    /// `qkv|z|b|a` as one `[hidden, conv_dim + value_dim + 2 v_heads]` matrix, the four parts then
    /// `None` ([`Qwen35::prepared`]): one product where there were four.
    in_proj: Option<Linear<B>>,
    /// The depthwise causal convolution's taps, `[conv_dim, 1, conv_kernel]` (no bias).
    conv1d: Param<Tensor<B, 3>>,
    /// The taps as rows, `[1, conv_kernel, conv_dim]` ([`Qwen35::prepared`]).
    taps: Option<Param<Tensor<B, 3>>>,
    /// The decay's rate and bias per value head (`A_log`, `dt_bias`).
    a_log: Param<Tensor<B, 1>>,
    dt_bias: Param<Tensor<B, 1>>,
    /// `-exp(A_log)` in F32 ([`Qwen35::prepared`]): the same bits as each step's `exp` then the
    /// product's negation.
    neg_rate: Option<Param<Tensor<B, 1>>>,
    /// The scales that make an RMS norm over `k_dim` (with `eps / k_dim`) the L2 norm the queries
    /// and keys take, `1 / k_dim` and `k_dim^-0.5` ([`Qwen35::prepared`]): one op on the card.
    q_scale: Option<Param<Tensor<B, 1>>>,
    k_scale: Option<Param<Tensor<B, 1>>>,
    norm: GatedNorm<B>,
    out_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    gate_proj: Option<Linear<B>>,
    up_proj: Option<Linear<B>>,
    down_proj: Linear<B>,
    /// `gate|up` as one `[hidden, 2 ff]` matrix, the two then `None` ([`Qwen35::prepared`]).
    gate_up: Option<Linear<B>>,
}

/// One decoder layer: exactly one of `linear_attn` and `self_attn` is present
/// ([`Qwen35Config::is_full`]).
#[derive(Module, Debug)]
pub struct Layer<B: Backend> {
    input_layernorm: CenteredNorm<B>,
    linear_attn: Option<GatedDeltaNet<B>>,
    self_attn: Option<GatedAttention<B>>,
    post_attention_layernorm: CenteredNorm<B>,
    mlp: Mlp<B>,
}

/// What one layer keeps between steps.
#[derive(Clone, Debug)]
pub enum LayerState<B: Backend> {
    /// Full attention: keys and values, `[1, kv_heads, max_len, head_dim]`, filled front to back.
    Attention { k: Tensor<B, 4>, v: Tensor<B, 4> },
    /// A DeltaNet: the convolution's last `conv_kernel - 1` inputs, `[1, conv_kernel - 1,
    /// conv_dim]` (zeros before the first token), and the recurrent state, `[1, v_heads, k_dim,
    /// v_dim]`, F32.
    Recurrent { conv: Tensor<B, 3>, s: Tensor<B, 4> },
}

/// The state of every layer after the positions a model has seen.
pub struct State<B: Backend> {
    layers: Vec<LayerState<B>>,
    len: usize,
    max_len: usize,
}

impl<B: Backend> State<B> {
    /// Positions already seen.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn layers(&self) -> &[LayerState<B>] {
        &self.layers
    }
}

impl<B: Backend> GatedAttention<B> {
    fn new(cfg: &Qwen35Config, device: &B::Device) -> Self {
        let (h, q, kv) = (
            cfg.hidden,
            cfg.heads * cfg.head_dim,
            cfg.kv_heads * cfg.head_dim,
        );
        Self {
            q_proj: Some(linear(h, 2 * q, device)),
            k_proj: Some(linear(h, kv, device)),
            v_proj: Some(linear(h, kv, device)),
            qkv: None,
            o_proj: linear(q, h, device),
            q_norm: CenteredNorm::new(cfg.head_dim, cfg.eps, device),
            k_norm: CenteredNorm::new(cfg.head_dim, cfg.eps, device),
        }
    }

    /// The queries with their gates, the keys and the values of `x`.
    fn project_in(&self, cfg: &Qwen35Config, x: Tensor<B, 3>) -> [Tensor<B, 3>; 3] {
        match (&self.qkv, &self.q_proj, &self.k_proj, &self.v_proj) {
            (Some(w), ..) => {
                let (q, kv) = (2 * cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
                split_columns(project(x, w), [0, q, q + kv])
            }
            (None, Some(q), Some(k), Some(v)) => {
                [project(x.clone(), q), project(x.clone(), k), project(x, v)]
            }
            _ => unreachable!("the attention's products, fused or apart"),
        }
    }

    fn fuse(&mut self) {
        if let (Some(q), Some(k), Some(v)) =
            (self.q_proj.take(), self.k_proj.take(), self.v_proj.take())
        {
            self.qkv = Some(concat(&[q, k, v]));
        }
    }

    /// `x` (`[1, t, heads, head_dim]`, transposed to `[1, heads, t, head_dim]`) with its first
    /// `rotary_dim` dimensions rotated.
    fn rotate(cfg: &Qwen35Config, x: Tensor<B, 4>, rope: &Rope<B>) -> Tensor<B, 4> {
        let [n, h, t, d] = x.dims();
        if cfg.rotary_dim == d {
            return rope.apply(x);
        }
        let rotated = rope.apply(x.clone().slice([0..n, 0..h, 0..t, 0..cfg.rotary_dim]));
        let passed = x.slice([0..n, 0..h, 0..t, cfg.rotary_dim..d]);
        Tensor::cat(vec![rotated, passed], 3)
    }

    /// The mixer's output `[1, t, hidden]` for the normed `x`, the step's keys and values written
    /// into `k`/`v` at `start..start + t`.
    fn forward(
        &self,
        cfg: &Qwen35Config,
        x: Tensor<B, 3>,
        rope: &Rope<B>,
        start: usize,
        k_cache: &mut Tensor<B, 4>,
        v_cache: &mut Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [n, t, _] = x.dims();
        let (heads, kv_heads, d) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
        let [qg, k, v] = self.project_in(cfg, x);
        let qg = qg.reshape([n, t, heads, 2 * d]);
        let q = qg.clone().slice([0..n, 0..t, 0..heads, 0..d]);
        let gate = qg
            .slice([0..n, 0..t, 0..heads, d..2 * d])
            .reshape([n, t, heads * d]);
        let k = k.reshape([n, t, kv_heads, d]);
        let v = v.reshape([n, t, kv_heads, d]);
        let q = Self::rotate(cfg, self.q_norm.forward(q).swap_dims(1, 2), rope);
        let k = Self::rotate(cfg, self.k_norm.forward(k).swap_dims(1, 2), rope);
        let v = v.swap_dims(1, 2);

        let total = start + t;
        *k_cache = k_cache
            .clone()
            .slice_assign([0..n, 0..kv_heads, start..total, 0..d], k);
        *v_cache = v_cache
            .clone()
            .slice_assign([0..n, 0..kv_heads, start..total, 0..d], v);
        let groups = heads / kv_heads;
        let keys = repeat_kv(
            k_cache.clone().slice([0..n, 0..kv_heads, 0..total, 0..d]),
            groups,
        );
        let values = repeat_kv(
            v_cache.clone().slice([0..n, 0..kv_heads, 0..total, 0..d]),
            groups,
        );
        // A prompt attends causally to itself; one new token attends to everything cached.
        let out = attention(
            q,
            keys,
            values,
            None,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: t > 1,
            },
        );
        let context = out.swap_dims(1, 2).reshape([n, t, heads * d]);
        project(context * sigmoid(gate), &self.o_proj)
    }

    /// [`Self::forward`] over a [`FixedCache`] layer (`cache_k`, `cache_v`, as it stores them):
    /// the keys and values written where `write` says, each query attending to the positions
    /// `write` reads but those `mask` (`[1, 1, groups t, len]`, or `[1, 1, 1, len]` for one
    /// token) sets, with each group of query heads folded into the rows, as
    /// `crate::llama::Layer::forward` does. The output and the layer's new keys and values.
    #[allow(clippy::too_many_arguments)]
    fn forward_fixed(
        &self,
        cfg: &Qwen35Config,
        x: Tensor<B, 3>,
        rope: &Rope<B>,
        cache_k: Tensor<B, 4>,
        cache_v: Tensor<B, 4>,
        write: &FixedWrite<B>,
        mask: &Tensor<B, 4, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let [n, t, _] = x.dims();
        let (heads, kv_heads, d) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
        let groups = heads / kv_heads;
        let [qg, k, v] = self.project_in(cfg, x);
        let qg = qg.reshape([n, t, heads, 2 * d]);
        let q = qg.clone().slice([0..n, 0..t, 0..heads, 0..d]);
        let gate = qg
            .slice([0..n, 0..t, 0..heads, d..2 * d])
            .reshape([n, t, heads * d]);
        let k = k.reshape([n, t, kv_heads, d]);
        let v = v.reshape([n, t, kv_heads, d]);
        let q = Self::rotate(cfg, self.q_norm.forward(q).swap_dims(1, 2), rope);
        let k = Self::rotate(cfg, self.k_norm.forward(k).swap_dims(1, 2), rope);
        let v = v.swap_dims(1, 2);

        let new_k = write.write(cache_k, k);
        let new_v = write.write(cache_v, v);
        let q = q.reshape([n, kv_heads, groups * t, d]);
        let (keys, values) = (write.attended(new_k.clone()), write.attended(new_v.clone()));
        let out = attention(
            q,
            keys,
            values,
            Some(mask.clone()),
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: false,
            },
        );
        let context = out
            .reshape([n, heads, t, d])
            .swap_dims(1, 2)
            .reshape([n, t, heads * d]);
        (project(context * sigmoid(gate), &self.o_proj), new_k, new_v)
    }
}

/// `softplus(x) = log(1 + exp(x))`, and `x` itself above 20, as `torch.nn.functional.softplus`
/// computes it (its default threshold).
fn softplus<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let above = x.clone().greater_elem(20.0);
    x.clone().exp().log1p().mask_where(above, x)
}

/// `x rsqrt(sum(x^2) + 1e-6)` over the last dimension (F32 in, F32 out).
fn l2norm<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let sum_square = (x.clone() * x.clone()).sum_dim(D - 1);
    x * sum_square.add_scalar(1e-6).sqrt().recip()
}

impl<B: Backend> GatedDeltaNet<B> {
    fn new(cfg: &Qwen35Config, device: &B::Device) -> Self {
        let (h, vh) = (cfg.hidden, cfg.linear_v_heads);
        Self {
            in_proj_qkv: Some(linear(h, cfg.conv_dim(), device)),
            in_proj_z: Some(linear(h, cfg.value_dim(), device)),
            in_proj_b: Some(linear(h, vh, device)),
            in_proj_a: Some(linear(h, vh, device)),
            in_proj: None,
            taps: None,
            neg_rate: None,
            q_scale: None,
            k_scale: None,
            // As PyTorch initialises them: `Conv1d`'s uniform bound for `conv_kernel` inputs a
            // channel, and `A` uniform in `[0.01, 16]` (here its logarithm).
            conv1d: Param::from_tensor(Tensor::random(
                [cfg.conv_dim(), 1, cfg.conv_kernel],
                Distribution::Uniform(
                    -(cfg.conv_kernel as f64).powf(-0.5),
                    (cfg.conv_kernel as f64).powf(-0.5),
                ),
                device,
            )),
            a_log: Param::from_tensor(Tensor::random(
                [vh],
                Distribution::Uniform(0.01f64.ln(), 16f64.ln()),
                device,
            )),
            dt_bias: Param::from_tensor(Tensor::ones([vh], device)),
            norm: GatedNorm::new(cfg.linear_v_dim, cfg.eps, device),
            out_proj: linear(cfg.value_dim(), h, device),
        }
    }

    /// The depthwise causal convolution of `x` (`[1, t, conv_dim]`) after the `conv_kernel - 1`
    /// inputs in `state`, then SiLU; `state` becomes the last `conv_kernel - 1` inputs, or, for a
    /// padded chunk ([`Padding`]), the last `conv_kernel - 1` before its padding rows.
    fn convolve(
        &self,
        x: Tensor<B, 3>,
        state: &mut Tensor<B, 3>,
        padding: Option<&Padding<B>>,
    ) -> Tensor<B, 3> {
        let [n, t, c] = x.dims();
        let taps = self.taps();
        let [_, kernel, _] = taps.dims();
        let dtype = taps.dtype();
        let padded = Tensor::cat(vec![state.clone(), x.cast(dtype)], 1);
        let past = kernel - 1;
        *state = match padding {
            None => padded.clone().slice([0..n, t..t + past, 0..c]),
            // Rows `last + 1 ..= last + past` of `[state, x]`: the inputs that end at the
            // prompt's last token, selected by a tensor so the shapes do not depend on it.
            Some(p) => padded
                .clone()
                .reshape([past + t, c])
                .select(0, p.conv_rows.clone())
                .reshape([n, past, c]),
        };
        // `out[i] = sum_j taps[j] padded[i + j]` (a cross-correlation, as `conv1d`): for one token,
        // its window times the taps summed over the window, two ops; for more, in tap order.
        if t == 1 {
            return silu((padded * taps).sum_dim(1));
        }
        let tap = |j: usize| taps.clone().slice([0..1, j..j + 1, 0..c]);
        let mut out = padded.clone().slice([0..n, 0..t, 0..c]) * tap(0);
        for j in 1..kernel {
            out = out + padded.clone().slice([0..n, j..j + t, 0..c]) * tap(j);
        }
        silu(out)
    }

    /// The taps as rows, `[1, conv_kernel, conv_dim]`.
    fn taps(&self) -> Tensor<B, 3> {
        match &self.taps {
            Some(taps) => taps.val(),
            None => {
                let taps = self.conv1d.val();
                let [c, _, kernel] = taps.dims();
                taps.reshape([c, kernel])
                    .swap_dims(0, 1)
                    .reshape([1, kernel, c])
            }
        }
    }

    /// The products of `x` (`[1, t, hidden]`): the convolution's inputs, the output gate, the
    /// write strength's and the decay's logits.
    fn project_in(&self, cfg: &Qwen35Config, x: Tensor<B, 3>) -> [Tensor<B, 3>; 4] {
        fn part<B: Backend>(p: &Option<Linear<B>>) -> &Linear<B> {
            p.as_ref().expect("the in-projections, fused or apart")
        }
        match &self.in_proj {
            Some(w) => {
                let (c, v) = (cfg.conv_dim(), cfg.value_dim());
                let vh = cfg.linear_v_heads;
                split_columns(project(x, w), [0, c, c + v, c + v + vh])
            }
            None => [
                project(x.clone(), part(&self.in_proj_qkv)),
                project(x.clone(), part(&self.in_proj_z)),
                project(x.clone(), part(&self.in_proj_b)),
                project(x, part(&self.in_proj_a)),
            ],
        }
    }

    /// `x` (`[1, t, heads, k_dim]`, F32) L2-normed over `k_dim` and times `scale`: one op on the
    /// card where `gamma` (`scale / k_dim^0.5` as a row) is made (an RMS norm with `1e-6 / k_dim`),
    /// else the composition.
    fn unit(x: Tensor<B, 4>, gamma: &Option<Param<Tensor<B, 1>>>, scale: f64) -> Tensor<B, 4> {
        let width = x.dims()[3];
        if let Some(gamma) = gamma {
            if let Some(y) = burn_tt::rms_norm(&x, &gamma.val(), 1e-6 / width as f64) {
                return y;
            }
        }
        let y = l2norm(x);
        if scale == 1.0 {
            y
        } else {
            y.mul_scalar(scale)
        }
    }

    /// The constants a step would otherwise make again ([`Qwen35::prepared`]).
    fn prepare(&mut self, cfg: &Qwen35Config) {
        if let (Some(qkv), Some(z), Some(b), Some(a)) = (
            self.in_proj_qkv.take(),
            self.in_proj_z.take(),
            self.in_proj_b.take(),
            self.in_proj_a.take(),
        ) {
            self.in_proj = Some(concat(&[qkv, z, b, a]));
        }
        if self.taps.is_none() {
            self.taps = Some(Param::from_tensor(self.taps()));
        }
        if self.neg_rate.is_none() {
            let rate = self.a_log.val().cast(DType::F32).exp().neg();
            self.neg_rate = Some(Param::from_tensor(rate));
        }
        let kd = cfg.linear_k_dim;
        let device = self.a_log.val().device();
        let row = |v: f64| Param::from_tensor(Tensor::full([kd], v, &device));
        if self.q_scale.is_none() {
            self.q_scale = Some(row(1.0 / kd as f64));
            self.k_scale = Some(row((kd as f64).powf(-0.5)));
        }
    }

    /// The mixer's output `[1, t, hidden]` for the normed `x`, the convolution's inputs and the
    /// recurrent state carried in `conv` and `s`.
    /// A chunk's padding rows (`padding`) leave `conv` and `s` as the prompt's last row left them:
    /// their decay is 1 and their write strength 0, so the recurrence carries the state through
    /// them unchanged, and the convolution keeps the inputs that end at the prompt's last token.
    fn forward(
        &self,
        cfg: &Qwen35Config,
        x: Tensor<B, 3>,
        conv: &mut Tensor<B, 3>,
        s: &mut Tensor<B, 4>,
        padding: Option<&Padding<B>>,
    ) -> Tensor<B, 3> {
        let [n, t, _] = x.dims();
        let stream = x.dtype();
        let (kh, kd, vh, vd) = (
            cfg.linear_k_heads,
            cfg.linear_k_dim,
            cfg.linear_v_heads,
            cfg.linear_v_dim,
        );
        let (key_dim, value_dim) = (cfg.key_dim(), cfg.value_dim());
        let [qkv, z, b, a] = self.project_in(cfg, x);
        let mixed = self.convolve(qkv, conv, padding);
        let z = z.reshape([n, t, vh, vd]);
        let beta = sigmoid(b).cast(DType::F32);
        let a = a.cast(DType::F32);
        let neg_rate = match &self.neg_rate {
            Some(r) => r.val(),
            None => self.a_log.val().cast(DType::F32).exp().neg(),
        }
        .unsqueeze::<3>();
        let bias = self.dt_bias.val().cast(DType::F32).unsqueeze::<3>();
        // The decay, in log space, `[1, t, v_heads]`: `-(exp(A_log) softplus)`, the negation on
        // the rate (the same bits).
        let g = neg_rate * softplus(a + bias);
        // Per token and head, `[1, t, v_heads]`: the decay factor and the write strength, a
        // padding row's 1 and 0 (exact: `s 1 = s`, and the update it adds is zeros).
        let (decay, beta) = match padding {
            None => (g.exp(), beta),
            Some(p) => (
                g.exp().mask_fill(p.rows.clone(), 1.0),
                beta.mask_fill(p.rows.clone(), 0.0),
            ),
        };

        let split = |from: usize, to: usize, heads: usize, width: usize| {
            mixed
                .clone()
                .slice([0..n, 0..t, from..to])
                .reshape([n, t, heads, width])
                .cast(DType::F32)
        };
        // Each key head repeated for its value heads, `repeat_interleave` order.
        let groups = vh / kh;
        let widen = |x: Tensor<B, 4>| {
            if groups == 1 {
                x
            } else {
                x.reshape([n, t, kh, 1, kd])
                    .repeat_dim(3, groups)
                    .reshape([n, t, vh, kd])
            }
        };
        let q = widen(Self::unit(
            split(0, key_dim, kh, kd),
            &self.q_scale,
            (kd as f64).powf(-0.5),
        ));
        let k = widen(Self::unit(
            split(key_dim, 2 * key_dim, kh, kd),
            &self.k_scale,
            1.0,
        ));
        let v = split(2 * key_dim, 2 * key_dim + value_dim, vh, vd);
        // `[1, v_heads, t, width]`, and the per-token scalars `[1, v_heads, t, 1]`.
        let (q, k, v) = (q.swap_dims(1, 2), k.swap_dims(1, 2), v.swap_dims(1, 2));
        // Each token's decay as a column of the state's `[v_heads k_dim, v_dim]` matrix, `[1, t,
        // v_heads, k_dim]`: the card multiplies the state by a column where it lies, where a
        // scalar a head would first be expanded to the state's whole size.
        let decay = decay.unsqueeze_dim::<4>(3).expand([n, t, vh, kd]);
        let beta = beta.swap_dims(1, 2).unsqueeze_dim::<4>(3);

        let mut outs = Vec::with_capacity(t);
        for i in 0..t {
            let row =
                |x: &Tensor<B, 4>, width: usize| x.clone().slice([0..n, 0..vh, i..i + 1, 0..width]);
            let (q_i, k_i, v_i) = (row(&q, kd), row(&k, kd), row(&v, vd));
            let decay_i = decay
                .clone()
                .slice([0..n, i..i + 1, 0..vh, 0..kd])
                .reshape([n, vh, kd, 1]);
            let state = s.clone() * decay_i;
            // What the state already predicts for this key, and the correction written for it
            // (the key as a column: `k^T`, the same values).
            let predicted = k_i.clone().matmul(state.clone());
            let delta = (v_i - predicted) * row(&beta, 1);
            let state = state + k_i.reshape([n, vh, kd, 1]).matmul(delta);
            outs.push(q_i.matmul(state.clone()));
            *s = state;
        }
        let out = Tensor::cat(outs, 2).swap_dims(1, 2).cast(stream);
        let out = self.norm.forward(out, z).reshape([n, t, value_dim]);
        project(out, &self.out_proj)
    }
}

impl<B: Backend> Mlp<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (gate, up) = match (&self.gate_up, &self.gate_proj, &self.up_proj) {
            (Some(gate_up), _, _) => {
                let y = project(x, gate_up);
                let ff = y.dims()[2] / 2;
                let [gate, up] = split_columns(y, [0, ff]);
                (gate, up)
            }
            (None, Some(gate), Some(up)) => (project(x.clone(), gate), project(x, up)),
            _ => unreachable!("the MLP's products, fused or apart"),
        };
        project(silu(gate) * up, &self.down_proj)
    }

    /// `gate|up` as one product.
    fn fuse(&mut self) {
        if let (Some(gate), Some(up)) = (self.gate_proj.take(), self.up_proj.take()) {
            self.gate_up = Some(concat(&[gate, up]));
        }
    }
}

/// The projections `parts` (no bias) as one, their weights side by side along the outputs.
fn concat<B: Backend>(parts: &[Linear<B>]) -> Linear<B> {
    assert!(
        parts.iter().all(|p| p.bias.is_none()),
        "projections without a bias"
    );
    // On the host's data, one matrix at a time: a freshly loaded weight is still host data on the
    // card's backend, and a device `cat` would upload each part first.
    let first = parts[0].weight.val();
    let (device, dtype, rows) = (first.device(), first.dtype(), first.dims()[0]);
    let datas: Vec<(Vec<f32>, usize)> = parts
        .iter()
        .map(|p| {
            let w = p.weight.val();
            let [r, cols] = w.dims();
            assert_eq!(r, rows, "projections of one input width");
            let v = w.into_data().convert::<f32>().into_vec::<f32>();
            (v.expect("a float weight"), cols)
        })
        .collect();
    let width: usize = datas.iter().map(|(_, c)| c).sum();
    let mut out = Vec::with_capacity(rows * width);
    for r in 0..rows {
        for (v, cols) in &datas {
            out.extend_from_slice(&v[r * cols..(r + 1) * cols]);
        }
    }
    drop(datas);
    let whole = Tensor::<B, 2>::from_data(TensorData::new(out, [rows, width]), &device);
    Linear {
        weight: Param::from_tensor(if dtype == DType::F32 {
            whole
        } else {
            whole.cast(dtype)
        }),
        bias: None,
    }
}

/// `y`'s last dimension cut at the column offsets `at` (ascending, the first 0) into `N` parts.
fn split_columns<B: Backend, const N: usize>(y: Tensor<B, 3>, at: [usize; N]) -> [Tensor<B, 3>; N] {
    let [n, t, w] = y.dims();
    std::array::from_fn(|i| {
        let end = if i + 1 < N { at[i + 1] } else { w };
        y.clone().slice([0..n, 0..t, at[i]..end])
    })
}

impl<B: Backend> Layer<B> {
    fn new(cfg: &Qwen35Config, index: usize, device: &B::Device) -> Self {
        let full = cfg.is_full(index);
        Self {
            input_layernorm: CenteredNorm::new(cfg.hidden, cfg.eps, device),
            linear_attn: (!full).then(|| GatedDeltaNet::new(cfg, device)),
            self_attn: full.then(|| GatedAttention::new(cfg, device)),
            post_attention_layernorm: CenteredNorm::new(cfg.hidden, cfg.eps, device),
            mlp: Mlp {
                gate_proj: Some(linear(cfg.hidden, cfg.ff, device)),
                up_proj: Some(linear(cfg.hidden, cfg.ff, device)),
                down_proj: linear(cfg.ff, cfg.hidden, device),
                gate_up: None,
            },
        }
    }

    fn forward(
        &self,
        cfg: &Qwen35Config,
        x: Tensor<B, 3>,
        rope: &Rope<B>,
        start: usize,
        state: &mut LayerState<B>,
    ) -> Tensor<B, 3> {
        let h = self.input_layernorm.forward(x.clone());
        let mixed = match (state, &self.linear_attn, &self.self_attn) {
            (LayerState::Recurrent { conv, s }, Some(net), None) => {
                net.forward(cfg, h, conv, s, None)
            }
            (LayerState::Attention { k, v }, None, Some(attn)) => {
                attn.forward(cfg, h, rope, start, k, v)
            }
            _ => unreachable!("a layer's state is of its mixer's kind"),
        };
        let x = x + mixed;
        let h = self.post_attention_layernorm.forward(x.clone());
        x + self.mlp.forward(h)
    }

    /// [`Self::forward`] over a [`HybridCache`]: the layer's state tensors in (`state`, its two:
    /// keys and values, or convolution inputs `[1, 1, conv_kernel - 1, conv_dim]` and recurrent
    /// state), the output and the new state out.
    fn forward_fixed(
        &self,
        cfg: &Qwen35Config,
        x: Tensor<B, 3>,
        state: [Tensor<B, 4>; 2],
        at: &FixedAt<B>,
    ) -> (Tensor<B, 3>, [Tensor<B, 4>; 2]) {
        let h = self.input_layernorm.forward(x.clone());
        let [a, b] = state;
        let (mixed, state) = match (&self.linear_attn, &self.self_attn) {
            (Some(net), None) => {
                let [_, _, past, c] = a.dims();
                let mut conv = a.reshape([1, past, c]);
                let mut s = b;
                let y = net.forward(cfg, h, &mut conv, &mut s, at.padding.as_ref());
                (y, [conv.reshape([1, 1, past, c]), s])
            }
            (None, Some(attn)) => {
                let (write, mask) = at.attend.as_ref().expect("a write for an attention layer");
                let (y, k, v) = attn.forward_fixed(cfg, h, &at.rope, a, b, write, mask);
                (y, [k, v])
            }
            _ => unreachable!("a layer has one mixer"),
        };
        let x = x + mixed;
        let h = self.post_attention_layernorm.forward(x.clone());
        (x + self.mlp.forward(h), state)
    }
}

/// A padded chunk's rows past the prompt, for a DeltaNet ([`GatedDeltaNet::forward`]).
#[derive(Clone, Debug)]
struct Padding<B: Backend> {
    /// `[1, c, v_heads]`: set on the rows past the prompt's last.
    rows: Tensor<B, 3, Bool>,
    /// The `conv_kernel - 1` rows of `[carried inputs, chunk]` that end at the prompt's last token.
    conv_rows: Tensor<B, 1, Int>,
}

/// What every layer of a fixed-shape step shares: where an attention layer writes and what it
/// attends to, the rotary rows of the step's positions, and a chunk's padding.
struct FixedAt<B: Backend> {
    /// Where the keys and values go, and that write's mask with each row repeated for the query
    /// heads folded into it (`None` for a model without attention layers).
    attend: Option<(FixedWrite<B>, Tensor<B, 4, Bool>)>,
    rope: Rope<B>,
    padding: Option<Padding<B>>,
}

/// The Qwen3.5 text model.
#[derive(Module, Debug)]
pub struct Qwen35<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<Layer<B>>,
    norm: CenteredNorm<B>,
    /// The output projection, absent when it is the embedding table ([`Qwen35Config::tied_head`]).
    lm_head: Option<Linear<B>>,
    /// Not a parameter: the module derive skips it.
    config: Qwen35Config,
    /// The numeric contract the model was made for ([`Qwen35::for_inference`]), or `None`. Not a
    /// parameter.
    numerics: Option<burn_tt::Numerics>,
}

/// Widens every float parameter to F32.
struct ToF32;

impl<B: Backend> ModuleMapper<B> for ToF32 {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        param.map(|t| t.cast(DType::F32))
    }
}

/// The shards of `Qwen/Qwen3.5-4B` (`cargo xtask fetch-weights --model qwen3.5-4b`).
const QWEN3_5_4B_FILES: [&str; 2] = [
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
];

impl<B: Backend> Qwen35<B> {
    /// The architecture `config` describes, with its initialiser's weights.
    pub fn new(config: &Qwen35Config, device: &B::Device) -> Self {
        config.check();
        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab, config.hidden).init(device),
            layers: (0..config.layers)
                .map(|i| Layer::new(config, i, device))
                .collect(),
            norm: CenteredNorm::new(config.hidden, config.eps, device),
            lm_head: (!config.tied_head).then(|| linear(config.hidden, config.vocab, device)),
            config: *config,
            numerics: None,
        }
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    /// The embedding table, `[vocab, hidden]` (also the output head when it is tied).
    pub fn embed_table(&self) -> Tensor<B, 2> {
        self.embed_tokens.weight.val()
    }

    /// The device the parameters are on.
    pub fn device(&self) -> B::Device {
        self.embed_tokens.weight.val().device()
    }

    /// A pretrained Qwen3.5 checkpoint in `files` of `vendor/weights/<weights>/` whose text shape
    /// `config` gives, keeping its first `config.layers` layers, computing in `precision`. The
    /// language model's tensors live under `model.language_model.` (or `language_model.`, a
    /// checkpoint of the base model alone); the vision tower (`model.visual.`, `visual.`), the
    /// multi-token-prediction head (`mtp.`) and the dropped layers are the only tensors the file may
    /// hold besides.
    pub fn pretrained(
        config: &Qwen35Config,
        weights: &str,
        files: &[&str],
        precision: Precision,
        device: &B::Device,
    ) -> Result<(Self, Loaded), LoadError> {
        Self::load(Self::new(config, device), weights, files, precision)
    }

    /// A pretrained Qwen3.5 base model without an output head, read for its last hidden states
    /// ([`Self::hidden`]): a text encoder, such as [`MOSS_TEXT_ENCODER`].
    pub fn text_encoder(
        config: &Qwen35Config,
        weights: &str,
        files: &[&str],
        precision: Precision,
        device: &B::Device,
    ) -> Result<(Self, Loaded), LoadError> {
        let mut model = Self::new(config, device);
        model.lm_head = None;
        Self::load(model, weights, files, precision)
    }

    fn load(
        mut model: Self,
        weights: &str,
        files: &[&str],
        precision: Precision,
    ) -> Result<(Self, Loaded), LoadError> {
        let layers = model.config.layers;
        let ignore = move |name: &str| {
            name.starts_with("model.visual.")
                || name.starts_with("visual.")
                || name.starts_with("mtp.")
                || name
                    .strip_prefix("layers.")
                    .and_then(|rest| rest.split('.').next())
                    .and_then(|i| i.parse::<usize>().ok())
                    .is_some_and(|i| i >= layers)
        };
        let loaded = load_safetensors_shards(
            &mut model,
            &weights_files(weights, files),
            &Layout {
                remaps: &[
                    (r"^(model\.)?language_model\.", ""),
                    (r"\.linear_attn\.A_log$", ".linear_attn.a_log"),
                    (r"\.linear_attn\.conv1d\.weight$", ".linear_attn.conv1d"),
                ],
                pytorch: true,
                ignore: &ignore,
            },
        )?;
        let model = match precision {
            Precision::Bf16 => model,
            Precision::F32 => model.map(&mut ToF32),
        };
        Ok((model.prepared(), loaded))
    }

    /// The same model with what each step would otherwise make again made once, the same
    /// arithmetic: the zero-centred norms' `1 + weight`, each DeltaNet's four in-projections and
    /// each MLP's two as one product, the convolution's taps as rows, the decay's `-exp(A_log)`
    /// and the scales that make the queries' and keys' L2 norms one op on the card. A pretrained
    /// model is loaded prepared; apply before storing weights in another format
    /// ([`Self::with_bf16_weights`]), so the fused matrices are what is stored.
    pub fn prepared(mut self) -> Self {
        let cfg = self.config;
        for layer in &mut self.layers {
            layer.input_layernorm.shift();
            layer.post_attention_layernorm.shift();
            if let Some(attn) = &mut layer.self_attn {
                attn.q_norm.shift();
                attn.k_norm.shift();
                attn.fuse();
            }
            if let Some(net) = &mut layer.linear_attn {
                net.prepare(&cfg);
            }
            layer.mlp.fuse();
        }
        self.norm.shift();
        self
    }

    /// Qwen3.5-4B ([`QWEN3_5_4B`], `cargo xtask fetch-weights --model qwen3.5-4b`), its first
    /// `layers` layers of thirty-two.
    pub fn qwen3_5_4b(
        layers: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Result<(Self, Loaded), LoadError> {
        Self::pretrained(
            &QWEN3_5_4B.with_layers(layers),
            "qwen3.5-4b",
            &QWEN3_5_4B_FILES,
            precision,
            device,
        )
    }

    /// The embedding rows of `tokens` `[1, t]`, as `[1, t, hidden]`.
    pub fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed_tokens.forward(tokens)
    }

    /// An empty state for `max_len` positions of one sequence: keys and values in `dtype`, the
    /// convolutions' inputs in their taps' dtype, the recurrent states in F32.
    pub fn state(&self, max_len: usize, dtype: DType, device: &B::Device) -> State<B> {
        let cfg = &self.config;
        let layers = self
            .layers
            .iter()
            .map(|layer| match &layer.linear_attn {
                Some(net) => LayerState::Recurrent {
                    conv: Tensor::zeros(
                        [1, cfg.conv_kernel - 1, cfg.conv_dim()],
                        (device, net.conv1d.val().dtype()),
                    ),
                    s: Tensor::zeros(
                        [1, cfg.linear_v_heads, cfg.linear_k_dim, cfg.linear_v_dim],
                        (device, DType::F32),
                    ),
                },
                None => {
                    let shape = [1, cfg.kv_heads, max_len, cfg.head_dim];
                    LayerState::Attention {
                        k: Tensor::zeros(shape, (device, dtype)),
                        v: Tensor::zeros(shape, (device, dtype)),
                    }
                }
            })
            .collect();
        State {
            layers,
            len: 0,
            max_len,
        }
    }

    /// The final hidden states `[1, t, hidden]` (after the last norm) of `x` (`[1, t, hidden]`,
    /// embedded), whose positions follow `state`'s, which they are appended to.
    pub fn hidden(&self, mut x: Tensor<B, 3>, state: &mut State<B>, dtype: DType) -> Tensor<B, 3> {
        let [n, t, _] = x.dims();
        assert_eq!(n, 1, "one sequence at a time");
        let start = state.len;
        assert!(start + t <= state.max_len, "the state is full");
        if x.dtype() != dtype {
            x = x.cast(dtype);
        }
        let cfg = &self.config;
        let rope = Rope::new(start, t, cfg.rotary_dim, cfg.rope_theta, dtype, &x.device());
        for (layer, s) in self.layers.iter().zip(state.layers.iter_mut()) {
            x = layer.forward(cfg, x, &rope, start, s);
        }
        state.len += t;
        self.norm.forward(x)
    }

    /// The logits `[1, vocab]` after the last of `tokens` (`[1, t]`), which are appended to
    /// `state`.
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut State<B>,
        dtype: DType,
    ) -> Tensor<B, 2> {
        let [_, t] = tokens.dims();
        let hidden = self.config.hidden;
        let x = self.hidden(self.embed_tokens.forward(tokens), state, dtype);
        let last = x.slice([0..1, t - 1..t, 0..hidden]).reshape([1, hidden]);
        self.project_out(last)
    }

    /// The logits `[1, vocab]` of a normed hidden row `[1, hidden]`, in the head's dtype. A tied
    /// head with a BF16 table is computed as `table @ last^T` (then laid out as a row), the table a
    /// plain left operand: `last @ table^T` would hand burn-tt a transposed BF16 right operand,
    /// which its BF16 product first copies plain, and for a 248,320 x 2,560 table that copy's host
    /// bookkeeping exceeds 40 GB (`silicon_probe_qwen35_bf16_memory`, 2026-10-30). An F32 table
    /// (the F32 model, and the BF16-stored weights', whose table stays F32) is `last @ table^T`,
    /// whose F32 product reads the transposed table where it lies: its row is a plain one, which a
    /// trace can return (the row laid out from `[vocab, 1]` is a transposed view, which it cannot).
    fn project_out(&self, last: Tensor<B, 2>) -> Tensor<B, 2> {
        let to = |x: Tensor<B, 2>, dtype: DType| if x.dtype() == dtype { x } else { x.cast(dtype) };
        match &self.lm_head {
            Some(head) => {
                let weight = head.weight.val();
                to(last, weight.dtype()).matmul(weight)
            }
            None => {
                let table = self.embed_tokens.weight.val();
                let [vocab, _] = table.dims();
                let last = to(last, table.dtype());
                if table.dtype() == DType::F32 {
                    last.matmul(table.transpose())
                } else {
                    table.matmul(last.transpose()).reshape([1, vocab])
                }
            }
        }
    }
}

impl<B: burn_tt::storage::StorageBackend> Qwen35<B> {
    /// The model as inference runs it under the numeric contract `numerics` (ADR 0006): every
    /// matrix but the embedding table in the contract's format ([`crate::quant::Plan::for_numerics`]:
    /// BF16 under the reference contract, BFP8 under the performance one, BFP4 `gate|up` with
    /// [`WeightOptions::mlp_bfp4`]). Every matrix here is a product's weight, read where it lies.
    /// A tied head stays the F32 table: a head of its own would be a second copy of the table
    /// (roadmap, "a default holds a weight's bytes once"), until the table itself can be stored
    /// packed and still looked up. For a loaded (prepared) model of BF16 values.
    ///
    /// [`WeightOptions::mlp_bfp4`]: crate::quant::WeightOptions::mlp_bfp4
    pub fn for_inference(
        self,
        numerics: burn_tt::Numerics,
        options: crate::quant::WeightOptions,
    ) -> Self {
        let mut model = crate::quant::apply(
            self,
            &crate::quant::Plan::for_numerics(numerics, options),
            &crate::quant::Native,
        );
        model.numerics = Some(numerics);
        model
    }

    /// Every matrix but the embedding table stored BF16 ([`crate::quant::Plan::bf16_weights`]),
    /// for an F32 model ([`Precision::F32`]): half the bytes a decode step reads, the arithmetic
    /// F32 on the checkpoint's own BF16 values. That this gives the F32 model's logits bit for bit
    /// is shown for the Llama (`step240`), not yet for this model; `step260_qwen35` checks it
    /// against burn-flex as it checks the F32 model.
    pub fn with_bf16_weights(self) -> Self {
        crate::quant::apply(
            self,
            &crate::quant::Plan::bf16_weights(),
            &crate::quant::Native,
        )
    }
}

impl<B: Backend> EagerDecoder<B> for Qwen35<B> {
    fn generate(
        &self,
        prompt: &[i32],
        steps: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Generation {
        let dtype = precision.dtype();
        let len = prompt.len() + steps;
        let mut state = self.state(len, dtype, device);
        eager(prompt, steps, None, device, |ids| {
            self.forward(ids, &mut state, dtype)
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
        let mut state = self.state(len, dtype, device);
        eager(prompt, 0, Some(continuation), device, |ids| {
            self.forward(ids, &mut state, dtype)
        })
    }
}

/// Qwen3.5's state for fixed-shape steps ([`crate::generate::FixedShapeDecoder`]): a
/// [`FixedCache`] of the full-attention layers' keys and values, each DeltaNet's convolution
/// inputs (`[1, 1, conv_kernel - 1, conv_dim]`) and recurrent state (`[1, v_heads, k_dim, v_dim]`,
/// F32), and the rotary table the steps select their rows from.
///
/// Unlike keys and values, a recurrent state cannot be overwritten later, so a chunk's padding rows
/// must leave it as the prompt's last token did: their decay is 1 and their write strength 0, and
/// the carried convolution inputs are the ones that end at the prompt's last token ([`Padding`]).
/// A chunk from position 0 starts from zeros whatever the state held, as a prompt does.
#[derive(Clone, Debug)]
pub struct HybridCache<B: Backend> {
    /// The full-attention layers' keys and values, in layer order (`None` without any).
    kv: Option<FixedCache<B>>,
    /// `[conv0, s0, conv1, s1, ...]`, in the DeltaNets' layer order.
    recurrent: Vec<Tensor<B, 4>>,
    rope: Rope<B>,
    /// `0..max_len`: a chunk's rows, compared with its last.
    positions: Tensor<B, 1, Int>,
    max_len: usize,
}

impl<B: Backend> HybridCache<B> {
    pub fn kv(&self) -> Option<&FixedCache<B>> {
        self.kv.as_ref()
    }

    /// The DeltaNets' convolution inputs and recurrent states, `[conv0, s0, conv1, s1, ...]`.
    pub fn recurrent(&self) -> &[Tensor<B, 4>] {
        &self.recurrent
    }

    fn kv_tensors(&self) -> usize {
        self.kv.as_ref().map_or(0, |kv| 2 * kv.layers())
    }
}

impl<B: Backend> FixedShapeCache<B> for HybridCache<B> {
    fn max_len(&self) -> usize {
        self.max_len
    }

    /// The keys and values, `[k0, v0, ...]` of the attention layers, then
    /// [`HybridCache::recurrent`].
    fn state(&self) -> Vec<Tensor<B, 4>> {
        let mut state = self.kv.as_ref().map_or_else(Vec::new, |kv| kv.state());
        state.extend(self.recurrent.iter().cloned());
        state
    }

    fn set_state(&mut self, mut state: Vec<Tensor<B, 4>>) {
        let recurrent = state.split_off(self.kv_tensors());
        assert_eq!(
            recurrent.len(),
            self.recurrent.len(),
            "a state tensor a layer side"
        );
        for (new, old) in recurrent.iter().zip(&self.recurrent) {
            assert_eq!(new.dims(), old.dims(), "the recurrent state's shape");
        }
        self.recurrent = recurrent;
        if let Some(kv) = &mut self.kv {
            kv.set_state(state);
        }
    }
}

/// Where a fixed-shape pass's tokens are.
enum At<B: Backend> {
    /// One token at this position (`[1]`).
    Step(Tensor<B, 1, Int>),
    /// A chunk from `start` (`[1]`) whose prompt ends at row `last` (`[1]`); with `mask` false, its
    /// padding rows run through the recurrence as tokens (wrong by construction: the negative
    /// control of [`Padding`]).
    Chunk {
        start: Tensor<B, 1, Int>,
        last: Tensor<B, 1, Int>,
        mask: bool,
    },
}

impl<B: Backend> Qwen35<B> {
    /// An empty [`HybridCache`] for `max_len` positions: keys and values in `dtype` (laid out as
    /// [`CacheLayout::for_heads`] says), the convolutions' inputs in their taps' dtype, the
    /// recurrent states in F32.
    pub fn hybrid_cache(&self, max_len: usize, dtype: DType, device: &B::Device) -> HybridCache<B> {
        let cfg = &self.config;
        let full = (0..cfg.layers).filter(|&i| cfg.is_full(i)).count();
        assert!(max_len >= cfg.conv_kernel, "a cache of {max_len} positions");
        let kv = (full > 0).then(|| {
            FixedCache::zeros_in(
                full,
                CacheLayout::for_heads(cfg.kv_heads, cfg.head_dim),
                max_len,
                dtype,
                device,
            )
        });
        let recurrent = self
            .layers
            .iter()
            .filter_map(|layer| layer.linear_attn.as_ref())
            .flat_map(|net| {
                [
                    Tensor::zeros(
                        [1, 1, cfg.conv_kernel - 1, cfg.conv_dim()],
                        (device, net.conv1d.val().dtype()),
                    ),
                    Tensor::zeros(
                        [1, cfg.linear_v_heads, cfg.linear_k_dim, cfg.linear_v_dim],
                        (device, DType::F32),
                    ),
                ]
            })
            .collect();
        HybridCache {
            kv,
            recurrent,
            rope: Rope::table(max_len, cfg.rotary_dim, cfg.rope_theta, dtype, device),
            positions: Tensor::arange(0..max_len as i64, device),
            max_len,
        }
    }

    /// One fixed-shape pass over `cache`: the embedded `x` (`[1, t, hidden]`) at `at`, attending
    /// to the cache's first `len` positions. The logits `[1, vocab]` of the step's token or the
    /// chunk's row `last`, and the new state. Each row's arithmetic is [`Self::forward`]'s.
    fn fixed(
        &self,
        mut x: Tensor<B, 3>,
        at: At<B>,
        len: usize,
        cache: &HybridCache<B>,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        let cfg = &self.config;
        let [n, t, hidden] = x.dims();
        let max_len = cache.max_len;
        assert!(
            n == 1 && hidden == cfg.hidden && t <= len && len <= max_len,
            "{t} tokens attending to {len} of {max_len} positions"
        );
        let dtype = cache.rope.tables().0.dtype();
        if x.dtype() != dtype {
            x = x.cast(dtype);
        }
        let rows = || cache.positions.clone().narrow(0, 0, t);
        // Each token's position, where its keys and values go, and what each query attends to.
        let (own, write) = match &at {
            At::Step(position) => (
                position.clone(),
                cache
                    .kv
                    .as_ref()
                    .map(|kv| kv.step_write_within(position.clone(), len)),
            ),
            At::Chunk { start, .. } => match &cache.kv {
                Some(kv) => {
                    let (write, own) = kv.chunk_write_within(start.clone(), t, len);
                    (own, Some(write))
                }
                None => (start.clone().expand([t]) + rows(), None),
            },
        };
        let groups = cfg.heads / cfg.kv_heads;
        let attend = write.map(|write| {
            let later = write.later();
            let mask = if groups == 1 || t == 1 {
                later
            } else {
                later.repeat_dim(2, groups)
            };
            (write, mask)
        });
        // A padding row's rotary row is clamped into the table (its keys land past the prompt and
        // are overwritten before anything reads them).
        let rope = cache.rope.at(own.clamp_max(max_len as i64 - 1));

        let mut recurrent = cache.recurrent.clone();
        let padding = match &at {
            At::Step(_) => None,
            At::Chunk { start, last, mask } => {
                // A prompt's first chunk starts from zeros, whatever the state held.
                let fresh = start.clone().equal_elem(0).reshape([1, 1, 1, 1]);
                for s in &mut recurrent {
                    let shape = s.dims();
                    *s = s.clone().mask_fill(fresh.clone().expand(shape), 0.0);
                }
                mask.then(|| {
                    let vh = cfg.linear_v_heads;
                    let past = cfg.conv_kernel - 1;
                    Padding {
                        rows: rows()
                            .greater(last.clone().expand([t]))
                            .reshape([1, t, 1])
                            .expand([1, t, vh]),
                        conv_rows: cache.positions.clone().narrow(0, 0, past)
                            + last.clone().add_scalar(1).expand([past]),
                    }
                })
            }
        };
        let at_all = FixedAt {
            attend,
            rope,
            padding,
        };

        let kv = cache.kv.as_ref().map_or_else(Vec::new, |kv| kv.state());
        let (mut kv_new, mut rec_new) = (Vec::with_capacity(kv.len()), Vec::new());
        let (mut kv_at, mut rec_at) = (0, 0);
        for layer in &self.layers {
            let (a, b) = if layer.self_attn.is_some() {
                kv_at += 2;
                (kv[kv_at - 2].clone(), kv[kv_at - 1].clone())
            } else {
                rec_at += 2;
                (recurrent[rec_at - 2].clone(), recurrent[rec_at - 1].clone())
            };
            let (y, [a, b]) = layer.forward_fixed(cfg, x, [a, b], &at_all);
            x = y;
            if layer.self_attn.is_some() {
                kv_new.extend([a, b]);
            } else {
                rec_new.extend([a, b]);
            }
        }
        let normed = self.norm.forward(x);
        let row = match at {
            At::Step(_) => normed.reshape([1, hidden]),
            At::Chunk { last, .. } => normed
                .reshape([t, hidden])
                .select(0, last)
                .reshape([1, hidden]),
        };
        kv_new.extend(rec_new);
        (self.project_out(row), kv_new)
    }

    /// [`crate::generate::FixedShapeDecoder::chunk`] with the padding rows run through the
    /// recurrence as tokens when `mask` is false: wrong by construction for a padded chunk, the
    /// negative control of the padding mask ([`UnmaskedPadding`]).
    fn fixed_chunk(
        &self,
        tokens: Tensor<B, 2, Int>,
        start: Tensor<B, 1, Int>,
        last: Tensor<B, 1, Int>,
        len: usize,
        cache: &HybridCache<B>,
        mask: bool,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        let x = self.embed(tokens);
        self.fixed(x, At::Chunk { start, last, mask }, len, cache)
    }
}

impl<B: Backend> FixedShapeDecoder<B> for Qwen35<B> {
    type Cache = HybridCache<B>;

    fn numerics(&self) -> Option<burn_tt::Numerics> {
        self.numerics
    }

    /// A cache whose keys and values are in the embedding table's dtype.
    fn new_cache(&self, max_len: usize, device: &B::Device) -> HybridCache<B> {
        self.hybrid_cache(max_len, self.embed_tokens.weight.val().dtype(), device)
    }

    fn chunk(
        &self,
        tokens: Tensor<B, 2, Int>,
        start: Tensor<B, 1, Int>,
        last: Tensor<B, 1, Int>,
        len: usize,
        cache: &HybridCache<B>,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        self.fixed_chunk(tokens, start, last, len, cache, true)
    }

    fn step(
        &self,
        token: Tensor<B, 2, Int>,
        position: Tensor<B, 1, Int>,
        len: usize,
        cache: &HybridCache<B>,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        let x = self.embed(token);
        self.fixed(x, At::Step(position), len, cache)
    }
}

/// A [`Qwen35`] whose chunks run their padding rows through the DeltaNets' recurrence and
/// convolution as if they were prompt tokens: what a hybrid cache without the padding mask would
/// compute, wrong by construction for a prompt that pads a chunk. The negative control of the mask
/// (`step263_qwen35_traced`).
#[doc(hidden)]
pub struct UnmaskedPadding<'a, B: Backend>(pub &'a Qwen35<B>);

impl<B: Backend> FixedShapeDecoder<B> for UnmaskedPadding<'_, B> {
    type Cache = HybridCache<B>;

    fn new_cache(&self, max_len: usize, device: &B::Device) -> HybridCache<B> {
        self.0.new_cache(max_len, device)
    }

    fn chunk(
        &self,
        tokens: Tensor<B, 2, Int>,
        start: Tensor<B, 1, Int>,
        last: Tensor<B, 1, Int>,
        len: usize,
        cache: &HybridCache<B>,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        self.0.fixed_chunk(tokens, start, last, len, cache, false)
    }

    fn step(
        &self,
        token: Tensor<B, 2, Int>,
        position: Tensor<B, 1, Int>,
        len: usize,
        cache: &HybridCache<B>,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        self.0.step(token, position, len, cache)
    }
}

#[cfg(test)]
fn ids<B: Backend>(tokens: &[i32], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(
        burn::tensor::TensorData::new(tokens.to_vec(), [1, tokens.len()]),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_flex::{Flex, FlexDevice};

    /// Every part at a few dimensions: two DeltaNets with grouped key heads and two full-attention
    /// layers with grouped key/value heads and half of each head rotated.
    const TINY: Qwen35Config = Qwen35Config {
        hidden: 32,
        heads: 4,
        kv_heads: 2,
        head_dim: 8,
        rotary_dim: 4,
        rope_theta: 10_000.0,
        linear_k_heads: 2,
        linear_k_dim: 8,
        linear_v_heads: 4,
        linear_v_dim: 4,
        conv_kernel: 4,
        full_every: 2,
        ff: 48,
        vocab: 64,
        layers: 4,
        eps: 1e-6,
        tied_head: true,
    };

    fn logits(model: &Qwen35<Flex>, state: &mut State<Flex>, ids: &[i32]) -> Vec<f32> {
        model
            .forward(super::ids(ids, &FlexDevice), state, DType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    /// The state a prompt leaves (every DeltaNet's convolution inputs and recurrence, every
    /// attention layer's keys and values) is what feeding its tokens one at a time leaves: the
    /// last token's logits agree. They are the same F32 operations on the same values, the
    /// attention's and the projections' sums taken over other shapes, so they agree to a few
    /// rounding errors of the logits' size (measured: below 1e-6 here); a state carried wrongly
    /// (a convolution tap off by a token, a decay applied twice) moves them by tenths.
    #[test]
    fn a_prompt_at_once_leaves_the_state_its_tokens_one_at_a_time_do() {
        let model = Qwen35::<Flex>::new(&TINY, &FlexDevice);
        let prompt = [3, 17, 42, 5, 9, 60, 1];

        let mut whole = model.state(prompt.len(), DType::F32, &FlexDevice);
        let at_once = logits(&model, &mut whole, &prompt);

        let mut stepped = model.state(prompt.len(), DType::F32, &FlexDevice);
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

        // The control: a prompt with one token changed predicts something else, so the
        // comparison above can fail.
        let mut other = model.state(prompt.len(), DType::F32, &FlexDevice);
        let mut changed = prompt;
        changed[2] = 43;
        let moved = logits(&model, &mut other, &changed)
            .iter()
            .zip(&at_once)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(moved > 1e-3, "{moved}");
    }

    fn worst(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// The fixed-shape schedule over a [`HybridCache`] (chunks of 4 and 8, the last one padded,
    /// then a step a token) predicts what the growing state does, at prompt lengths that pad a
    /// chunk by 1, 3 and 3 rows and one that fills it. The same operations on the same values
    /// a row, summed over other shapes: agreement to rounding (measured below 1e-6).
    ///
    /// The control: without the padding mask, a padded chunk's padding rows run through the
    /// recurrence and the convolution as tokens, and every decode step after it is wrong by tenths
    /// (the prompt's own logits are not: no row reads a later one).
    #[test]
    fn the_hybrid_cache_steps_as_the_growing_state_does() {
        use crate::generate::{generate_fixed, Shapes};
        let model = Qwen35::<Flex>::new(&TINY, &FlexDevice);
        let prepared = model.clone().prepared();
        let shapes = Shapes::new(16, &[4, 8]);
        let steps = 4;
        let pairs: Vec<(&Qwen35<Flex>, i32)> = [&model, &prepared]
            .into_iter()
            .flat_map(|m| [3, 4, 5, 9].map(move |len| (m, len)))
            .collect();
        for (model, len) in pairs {
            let prompt: Vec<i32> = (0..len).map(|i| (7 + 13 * i * i) % 64).collect();
            let growing = model.generate(&prompt, steps, Precision::F32, &FlexDevice);
            let fixed = generate_fixed(model, &prompt, steps, &shapes, &FlexDevice).unwrap();
            let w = worst(&growing.logits, &fixed.logits);
            eprintln!("prompt {len}: largest logit difference {w:e}");
            assert!(w < 1e-4, "prompt {len}: {w}");
            assert_eq!(growing.tokens, fixed.tokens, "prompt {len}");
            if len != 4 {
                let unmasked = generate_fixed(
                    &UnmaskedPadding(model),
                    &prompt,
                    steps,
                    &shapes,
                    &FlexDevice,
                )
                .unwrap();
                let prefill = worst(&growing.logits[..1], &unmasked.logits[..1]);
                let decode = worst(&growing.logits[1..], &unmasked.logits[1..]);
                eprintln!("prompt {len} unmasked: prefill {prefill:e}, decode {decode:e}");
                assert!(prefill < 1e-4 && decode > 1e-3, "{prefill} {decode}");
            }
        }
    }

    /// A chunk from position 0 starts from zeros whatever the cache held: a prompt fed after
    /// another, on the same cache, predicts what it does on a fresh one, bit for bit.
    #[test]
    fn a_prompt_from_position_zero_forgets_the_last() {
        let model = Qwen35::<Flex>::new(&TINY, &FlexDevice);
        let int = |v: i32| Tensor::<Flex, 1, Int>::from_data([v], &FlexDevice);
        let run = |cache: &mut HybridCache<Flex>, prompt: &[i32]| -> Vec<f32> {
            let (logits, state) = model.chunk(
                super::ids(prompt, &FlexDevice),
                int(0),
                int(prompt.len() as i32 - 1),
                16,
                cache,
            );
            cache.set_state(state);
            let (logits2, state) = model.step(
                super::ids(&[5], &FlexDevice),
                int(prompt.len() as i32),
                16,
                cache,
            );
            cache.set_state(state);
            let mut v = logits.into_data().to_vec::<f32>().unwrap();
            v.extend(logits2.into_data().to_vec::<f32>().unwrap());
            v
        };
        let mut fresh = model.new_cache(16, &FlexDevice);
        let want = run(&mut fresh, &[9, 8, 7, 6]);
        let mut used = model.new_cache(16, &FlexDevice);
        let other = run(&mut used, &[1, 2, 3, 4]);
        let got = run(&mut used, &[9, 8, 7, 6]);
        assert_eq!(got, want);
        assert_ne!(other, want);
    }

    /// [`Qwen35::prepared`] (the constants made once, the products fused) predicts what the model
    /// did: on burn-flex the same arithmetic (measured: identical), the products' columns computed
    /// apart or side by side.
    #[test]
    fn a_prepared_model_predicts_what_it_did() {
        let model = Qwen35::<Flex>::new(&TINY, &FlexDevice);
        let prompt = [3, 17, 42, 5, 9];
        let want = model.generate(&prompt, 4, Precision::F32, &FlexDevice);
        let prepared = model.clone().prepared();
        assert!(prepared.layers.iter().all(|l| l.mlp.gate_up.is_some()
            && l.self_attn.as_ref().is_none_or(|a| a.qkv.is_some())
            && l.linear_attn.as_ref().is_none_or(|n| n.in_proj.is_some())));
        let got = prepared.generate(&prompt, 4, Precision::F32, &FlexDevice);
        let w = worst(&want.logits, &got.logits);
        eprintln!("prepared: largest logit difference {w:e}");
        assert!(w < 1e-5, "{w}");
        assert_eq!(want.tokens, got.tokens);
    }

    #[test]
    fn the_layers_alternate_as_the_configuration_says() {
        let model = Qwen35::<Flex>::new(&QWEN3_5_4B.with_layers(8), &FlexDevice);
        let full: Vec<usize> = (0..8)
            .filter(|&i| model.layers[i].self_attn.is_some())
            .collect();
        assert_eq!(full, [3, 7]);
        assert!(model
            .layers
            .iter()
            .all(|l| l.self_attn.is_some() != l.linear_attn.is_some()));
    }
}
