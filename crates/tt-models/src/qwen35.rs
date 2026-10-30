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
use burn::tensor::{DType, Distribution, Int, Tensor};

use crate::generate::{eager, EagerDecoder, Generation};
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

/// A zero-centred RMSNorm: `rms(x) (1 + weight)` in F32, returned in `x`'s dtype.
#[derive(Module, Debug)]
pub struct CenteredNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> CenteredNorm<B> {
    fn new(width: usize, eps: f64, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([width], device)),
            eps,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = x.dtype();
        let scale = self
            .weight
            .val()
            .cast(DType::F32)
            .add_scalar(1.0)
            .unsqueeze::<D>();
        (rms(x, self.eps) * scale).cast(dtype)
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
        let normed = rms(x, self.eps).cast(dtype) * weight.cast(dtype).unsqueeze::<4>();
        (normed.cast(DType::F32) * silu(z.cast(DType::F32))).cast(dtype)
    }
}

/// Full attention with q/k norms, partial rotary embeddings and a sigmoid output gate.
#[derive(Module, Debug)]
pub struct GatedAttention<B: Backend> {
    /// `[hidden, heads 2 head_dim]`: each head's query, then its gate.
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_norm: CenteredNorm<B>,
    k_norm: CenteredNorm<B>,
}

/// A Gated DeltaNet: linear attention with a recurrent state per value head.
#[derive(Module, Debug)]
pub struct GatedDeltaNet<B: Backend> {
    /// `[hidden, 2 key_dim + value_dim]`: queries, keys and values, before the convolution.
    in_proj_qkv: Linear<B>,
    /// The output gate, `[hidden, value_dim]`.
    in_proj_z: Linear<B>,
    /// The write strength and the decay, one per value head.
    in_proj_b: Linear<B>,
    in_proj_a: Linear<B>,
    /// The depthwise causal convolution's taps, `[conv_dim, 1, conv_kernel]` (no bias).
    conv1d: Param<Tensor<B, 3>>,
    /// The decay's rate and bias per value head (`A_log`, `dt_bias`).
    a_log: Param<Tensor<B, 1>>,
    dt_bias: Param<Tensor<B, 1>>,
    norm: GatedNorm<B>,
    out_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
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
            q_proj: linear(h, 2 * q, device),
            k_proj: linear(h, kv, device),
            v_proj: linear(h, kv, device),
            o_proj: linear(q, h, device),
            q_norm: CenteredNorm::new(cfg.head_dim, cfg.eps, device),
            k_norm: CenteredNorm::new(cfg.head_dim, cfg.eps, device),
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
        let qg = project(x.clone(), &self.q_proj).reshape([n, t, heads, 2 * d]);
        let q = qg.clone().slice([0..n, 0..t, 0..heads, 0..d]);
        let gate = qg
            .slice([0..n, 0..t, 0..heads, d..2 * d])
            .reshape([n, t, heads * d]);
        let k = project(x.clone(), &self.k_proj).reshape([n, t, kv_heads, d]);
        let v = project(x, &self.v_proj).reshape([n, t, kv_heads, d]);
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
            in_proj_qkv: linear(h, cfg.conv_dim(), device),
            in_proj_z: linear(h, cfg.value_dim(), device),
            in_proj_b: linear(h, vh, device),
            in_proj_a: linear(h, vh, device),
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
    /// inputs in `state`, then SiLU; `state` becomes the last `conv_kernel - 1` inputs.
    fn convolve(&self, x: Tensor<B, 3>, state: &mut Tensor<B, 3>) -> Tensor<B, 3> {
        let [n, t, c] = x.dims();
        let taps = self.conv1d.val();
        let [_, _, kernel] = taps.dims();
        let dtype = taps.dtype();
        let padded = Tensor::cat(vec![state.clone(), x.cast(dtype)], 1);
        let past = kernel - 1;
        *state = padded.clone().slice([0..n, t..t + past, 0..c]);
        // `out[i] = sum_j taps[j] padded[i + j]`, in tap order (a cross-correlation, as `conv1d`).
        let tap = |j: usize| {
            taps.clone()
                .slice([0..c, 0..1, j..j + 1])
                .reshape([1, 1, c])
        };
        let mut out = padded.clone().slice([0..n, 0..t, 0..c]) * tap(0);
        for j in 1..kernel {
            out = out + padded.clone().slice([0..n, j..j + t, 0..c]) * tap(j);
        }
        silu(out)
    }

    /// The mixer's output `[1, t, hidden]` for the normed `x`, the convolution's inputs and the
    /// recurrent state carried in `conv` and `s`.
    fn forward(
        &self,
        cfg: &Qwen35Config,
        x: Tensor<B, 3>,
        conv: &mut Tensor<B, 3>,
        s: &mut Tensor<B, 4>,
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
        let mixed = self.convolve(project(x.clone(), &self.in_proj_qkv), conv);
        let z = project(x.clone(), &self.in_proj_z).reshape([n, t, vh, vd]);
        let beta = sigmoid(project(x.clone(), &self.in_proj_b)).cast(DType::F32);
        let a = project(x, &self.in_proj_a).cast(DType::F32);
        let rate = self.a_log.val().cast(DType::F32).exp().unsqueeze::<3>();
        let bias = self.dt_bias.val().cast(DType::F32).unsqueeze::<3>();
        // The decay, in log space, `[1, t, v_heads]`.
        let g = (rate * softplus(a + bias)).neg();

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
        let q = widen(l2norm(split(0, key_dim, kh, kd))).mul_scalar((kd as f64).powf(-0.5));
        let k = widen(l2norm(split(key_dim, 2 * key_dim, kh, kd)));
        let v = split(2 * key_dim, 2 * key_dim + value_dim, vh, vd);
        // `[1, v_heads, t, width]`, and the per-token scalars `[1, v_heads, t, 1]`.
        let (q, k, v) = (q.swap_dims(1, 2), k.swap_dims(1, 2), v.swap_dims(1, 2));
        let decay = g.swap_dims(1, 2).exp().unsqueeze_dim::<4>(3);
        let beta = beta.swap_dims(1, 2).unsqueeze_dim::<4>(3);

        let mut outs = Vec::with_capacity(t);
        for i in 0..t {
            let row =
                |x: &Tensor<B, 4>, width: usize| x.clone().slice([0..n, 0..vh, i..i + 1, 0..width]);
            let (q_i, k_i, v_i) = (row(&q, kd), row(&k, kd), row(&v, vd));
            let state = s.clone() * row(&decay, 1);
            // What the state already predicts for this key, and the correction written for it.
            let predicted = k_i.clone().matmul(state.clone());
            let delta = (v_i - predicted) * row(&beta, 1);
            let state = state + k_i.swap_dims(2, 3).matmul(delta);
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
        let hidden = silu(project(x.clone(), &self.gate_proj)) * project(x, &self.up_proj);
        project(hidden, &self.down_proj)
    }
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
                gate_proj: linear(cfg.hidden, cfg.ff, device),
                up_proj: linear(cfg.hidden, cfg.ff, device),
                down_proj: linear(cfg.ff, cfg.hidden, device),
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
            (LayerState::Recurrent { conv, s }, Some(net), None) => net.forward(cfg, h, conv, s),
            (LayerState::Attention { k, v }, None, Some(attn)) => {
                attn.forward(cfg, h, rope, start, k, v)
            }
            _ => unreachable!("a layer's state is of its mixer's kind"),
        };
        let x = x + mixed;
        let h = self.post_attention_layernorm.forward(x.clone());
        x + self.mlp.forward(h)
    }
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
        Ok((model, loaded))
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
    /// head is computed as `table @ last^T` (then laid out as a row), the table a plain left
    /// operand: `last @ table^T` would hand burn-tt a transposed BF16 right operand, which its BF16
    /// product first copies plain, and for a 248,320 x 2,560 table that copy's host bookkeeping
    /// exceeds 40 GB (`silicon_probe_qwen35_bf16_memory`, 2026-10-30). The products and their order
    /// over `hidden` are the same.
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
                table.matmul(last.transpose()).reshape([1, vocab])
            }
        }
    }
}

impl<B: burn_tt::storage::StorageBackend> Qwen35<B> {
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
