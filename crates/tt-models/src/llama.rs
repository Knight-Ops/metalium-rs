//! A decoder-only Llama-architecture language model: SmolLM2-135M.
//!
//! 30 pre-norm layers of grouped-query attention (9 query heads, 3 key/value heads, head size 64)
//! with rotary position embeddings, and a SwiGLU MLP, over a 49,152-token vocabulary whose
//! embedding table is also the output projection (`tie_word_embeddings`). Written with Burn's own
//! `Embedding`, `Linear` and `RmsNorm`, so it exercises what a real model uses: RMSNorm, SiLU, RoPE
//! (`sin` and `cos` computed on the device from the positions), `cat`/`slice` for the rotation,
//! and a key/value cache kept in preallocated tensors and filled with `slice_assign`.
//!
//! Field names follow the checkpoint, so only the `model.` prefix and the norms' `weight` need
//! remapping on load (`Layout::remaps`: Burn's `RmsNorm` calls its scale `gamma`, and `burn-store`'s
//! PyTorch adapter renames only batch, layer and group norms).
//!
//! The checkpoint is BF16. [`Precision::Bf16`] keeps it so, and every activation follows;
//! [`Precision::F32`] widens each parameter once after loading (lossless), which is what makes a
//! comparison with burn-flex a comparison of the same arithmetic on the same values.

use burn::module::{Module, ModuleMapper, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, Int, Tensor, TensorData};

use crate::load::{load_safetensors, weights_path, Layout, LoadError, Loaded};

pub const HIDDEN: usize = 576;
pub const HEADS: usize = 9;
pub const KV_HEADS: usize = 3;
pub const HEAD_DIM: usize = HIDDEN / HEADS;
pub const GROUPS: usize = HEADS / KV_HEADS;
pub const FF: usize = 1536;
pub const VOCAB: usize = 49_152;
pub const FULL_LAYERS: usize = 30;
const ROPE_THETA: f64 = 100_000.0;
const EPS: f64 = 1e-5;

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

fn rms_norm<B: Backend>(device: &B::Device) -> RmsNorm<B> {
    RmsNormConfig::new(HIDDEN).with_epsilon(EPS).init(device)
}

#[derive(Module, Debug)]
pub struct SelfAttn<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
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
    input_layernorm: RmsNorm<B>,
    self_attn: SelfAttn<B>,
    post_attention_layernorm: RmsNorm<B>,
    mlp: Mlp<B>,
}

/// Rotary tables for a run of positions: `[1, 1, t, HEAD_DIM]` each. `sin` holds its first half
/// negated, the sign `rotate_half` would put on `x`'s second half ([`Rope::apply`]).
struct Rope<B: Backend> {
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
}

impl<B: Backend> Rope<B> {
    /// `angle = position * theta^(-2i / HEAD_DIM)`, the first half of the row repeated as the
    /// second (the rotate-half layout), `sin`'s first half negated. The positions and the
    /// frequencies are host data that the shapes fix; the products and the `sin` and `cos` run on
    /// the device.
    fn new(start: usize, t: usize, dtype: DType, device: &B::Device) -> Self {
        let positions: Vec<f32> = (start..start + t).map(|p| p as f32).collect();
        let freqs: Vec<f32> = (0..HEAD_DIM / 2)
            .map(|i| ROPE_THETA.powf(-(2.0 * i as f64) / HEAD_DIM as f64) as f32)
            .collect();
        let pos = Tensor::<B, 2>::from_data(TensorData::new(positions, [t, 1]), device);
        let freq = Tensor::<B, 2>::from_data(TensorData::new(freqs, [1, HEAD_DIM / 2]), device);
        let angles = pos * freq;
        let sin = angles.clone().sin();
        let sin = Tensor::cat(vec![sin.clone().neg(), sin], 1).reshape([1, 1, t, HEAD_DIM]);
        let angles = Tensor::cat(vec![angles.clone(), angles], 1).reshape([1, 1, t, HEAD_DIM]);
        Self {
            cos: angles.cos().cast(dtype),
            sin: sin.cast(dtype),
        }
    }

    /// The tables for every position below `max_len`, `[max_len, HEAD_DIM]` each: what
    /// [`Rope::at`] selects a position's row from. Each row is [`Rope::new`]'s for that position,
    /// the same products in the same order.
    fn table(max_len: usize, dtype: DType, device: &B::Device) -> Self {
        let r = Self::new(0, max_len, dtype, device);
        Self {
            cos: r.cos.reshape([1, 1, max_len, HEAD_DIM]),
            sin: r.sin.reshape([1, 1, max_len, HEAD_DIM]),
        }
    }

    /// The rows of a [`Rope::table`] at `positions` (`[n]`, on the device): a row select, so a
    /// traced step replays it with new positions.
    fn at(&self, positions: Tensor<B, 1, Int>) -> Self {
        let [_, _, max_len, d] = self.cos.dims();
        let [n] = positions.dims();
        let row = |t: &Tensor<B, 4>| {
            t.clone()
                .reshape([max_len, d])
                .select(0, positions.clone())
                .reshape([1, 1, n, d])
        };
        Self {
            cos: row(&self.cos),
            sin: row(&self.sin),
        }
    }

    fn apply(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, h, t, d] = x.dims();
        let half = d / 2;
        let x1 = x.clone().slice([0..n, 0..h, 0..t, 0..half]);
        let x2 = x.clone().slice([0..n, 0..h, 0..t, half..d]);
        // `rotate_half(x) * sin`, `cat(-x2, x1) * [s, s]`, as `cat(x2, x1) * [-s, s]`: the same bits
        // (a product's sign is its operands' product, its magnitude theirs), without the
        // negation an application.
        let swapped = Tensor::cat(vec![x2, x1], 3);
        x * self.cos.clone() + swapped * self.sin.clone()
    }
}

/// Each key/value head repeated for its group of query heads (`repeat_interleave`).
fn repeat_kv<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, kv, l, d] = x.dims();
    let mut heads = Vec::with_capacity(kv * GROUPS);
    for h in 0..kv {
        let one = x.clone().slice([0..n, h..h + 1, 0..l, 0..d]);
        heads.extend((0..GROUPS).map(|_| one.clone()));
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
    fn new(device: &B::Device) -> Self {
        Self {
            input_layernorm: rms_norm(device),
            self_attn: SelfAttn {
                q_proj: linear(HIDDEN, HEADS * HEAD_DIM, device),
                k_proj: linear(HIDDEN, KV_HEADS * HEAD_DIM, device),
                v_proj: linear(HIDDEN, KV_HEADS * HEAD_DIM, device),
                o_proj: linear(HEADS * HEAD_DIM, HIDDEN, device),
            },
            post_attention_layernorm: rms_norm(device),
            mlp: Mlp {
                gate_proj: linear(HIDDEN, FF, device),
                up_proj: linear(HIDDEN, FF, device),
                down_proj: linear(FF, HIDDEN, device),
            },
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &Rope<B>,
        cache: &mut LayerCache<B>,
        write: &Write<B>,
        index: usize,
        act: Option<&ActFn<B>>,
    ) -> Tensor<B, 3> {
        let hook = |name: &str, t: Tensor<B, 3>| match act {
            Some(f) => f(&format!("layers.{index}.{name}"), t),
            None => t,
        };
        let [n, t, _] = x.dims();
        let h = self.input_layernorm.forward(x.clone());
        let a = &self.self_attn;
        let heads =
            |p: Tensor<B, 3>, count: usize| p.reshape([n, t, count, HEAD_DIM]).swap_dims(1, 2);
        let q = rope.apply(heads(project(h.clone(), &a.q_proj), HEADS));
        let k = rope.apply(heads(project(h.clone(), &a.k_proj), KV_HEADS));
        let v = heads(project(h, &a.v_proj), KV_HEADS);

        let (keys, values, mask, is_causal) =
            match write {
                Write::At(start) => {
                    let total = start + t;
                    cache.k = cache
                        .k
                        .clone()
                        .slice_assign([0..n, 0..KV_HEADS, *start..total, 0..HEAD_DIM], k);
                    cache.v = cache
                        .v
                        .clone()
                        .slice_assign([0..n, 0..KV_HEADS, *start..total, 0..HEAD_DIM], v);
                    let keys = repeat_kv(cache.k.clone().slice([
                        0..n,
                        0..KV_HEADS,
                        0..total,
                        0..HEAD_DIM,
                    ]));
                    let values = repeat_kv(cache.v.clone().slice([
                        0..n,
                        0..KV_HEADS,
                        0..total,
                        0..HEAD_DIM,
                    ]));
                    // A prompt attends causally to itself; one new token attends to everything cached.
                    (keys, values, None, t > 1)
                }
                Write::Masked { here, later } => {
                    // The new key and value land where `here` is set, every other position kept;
                    // the query attends to every position not `later` than its own.
                    let shape = cache.k.dims();
                    cache.k = cache.k.clone().mask_where(here.clone(), k.expand(shape));
                    cache.v = cache.v.clone().mask_where(here.clone(), v.expand(shape));
                    (
                        repeat_kv(cache.k.clone()),
                        repeat_kv(cache.v.clone()),
                        Some(later.clone()),
                        false,
                    )
                }
                Write::Chunk { here, later, index } => {
                    // The chunk's keys and values laid over the whole cache by a row gather (row
                    // `index` of the chunk's `[KV_HEADS c, HEAD_DIM]` for each cache row), kept
                    // where `here` is set; each query attends to every position not `later` than
                    // its own.
                    let shape = cache.k.dims();
                    let place = |x: Tensor<B, 4>| {
                        let [_, kv, c, d] = x.dims();
                        x.reshape([kv * c, d])
                            .select(0, index.clone())
                            .reshape(shape)
                    };
                    cache.k = cache.k.clone().mask_where(here.clone(), place(k));
                    cache.v = cache.v.clone().mask_where(here.clone(), place(v));
                    (
                        repeat_kv(cache.k.clone()),
                        repeat_kv(cache.v.clone()),
                        Some(later.clone()),
                        false,
                    )
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
        let context = hook(
            "attn_ctx",
            out.swap_dims(1, 2).reshape([n, t, HEADS * HEAD_DIM]),
        );
        let x = x + project(context, &a.o_proj);

        let h = self.post_attention_layernorm.forward(x.clone());
        let m = &self.mlp;
        let hidden = hook(
            "mlp_hidden",
            silu(project(h.clone(), &m.gate_proj)) * project(h, &m.up_proj),
        );
        x + project(hidden, &m.down_proj)
    }
}

/// One layer's keys and values, `[batch, KV_HEADS, max_len, HEAD_DIM]`, filled front to back.
struct LayerCache<B: Backend> {
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
}

/// How a step puts its keys and values into a layer's cache, and which of them it attends to.
enum Write<B: Backend> {
    /// At positions `start..start + t`, known on the host; the query attends to positions
    /// `..start + t` (causally within the new ones).
    At(usize),
    /// At the one position set in `here` (the cache's shape, `[1, KV_HEADS, max_len, HEAD_DIM]`:
    /// made once a step, not once a layer); the query attends to every position but those set in
    /// `later` (`[1, 1, 1, max_len]`). Both are tensors, so the step's shapes do not depend on the
    /// position ([`FixedCache`]).
    Masked {
        here: Tensor<B, 4, burn::tensor::Bool>,
        later: Tensor<B, 4, burn::tensor::Bool>,
    },
    /// A chunk of `c` tokens at the positions set in `here` (the cache's shape, a run from the
    /// chunk's start): cache row `j` of head `h` takes the chunk's row `index[h max_len + j]` of
    /// its `[KV_HEADS c, HEAD_DIM]` keys and values; query `i` attends to every position but those
    /// set in row `i` of `later` (`[1, 1, c, max_len]`).
    Chunk {
        here: Tensor<B, 4, burn::tensor::Bool>,
        later: Tensor<B, 4, burn::tensor::Bool>,
        index: Tensor<B, 1, Int>,
    },
}

/// A key/value cache whose step has the same shapes at every position, so it can be traced: one
/// token a step, its position a tensor, attention over all `max_len` positions with the ones not
/// yet written masked. [`Llama::step_fixed`] is [`Llama::forward_embedded`]'s decode step with
/// the same arithmetic on every position it attends to.
pub struct FixedCache<B: Backend> {
    layers: Vec<LayerCache<B>>,
    max_len: usize,
    rope: Rope<B>,
    /// `0..max_len`, to compare a position with.
    positions: Tensor<B, 1, Int>,
}

impl<B: Backend> FixedCache<B> {
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Every layer's keys and values, in layer order: the state a traced step writes back.
    pub fn state(&self) -> Vec<Tensor<B, 4>> {
        self.layers
            .iter()
            .flat_map(|c| [c.k.clone(), c.v.clone()])
            .collect()
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
}

/// Widens every float parameter to F32.
struct ToF32;

impl<B: Backend> ModuleMapper<B> for ToF32 {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        param.map(|t| t.cast(DType::F32))
    }
}

impl<B: Backend> Llama<B> {
    /// The architecture with `layers` layers and its initialiser's weights.
    pub fn new(layers: usize, device: &B::Device) -> Self {
        assert!((1..=FULL_LAYERS).contains(&layers), "layers {layers}");
        Self {
            embed_tokens: EmbeddingConfig::new(VOCAB, HIDDEN).init(device),
            layers: (0..layers).map(|_| Layer::new(device)).collect(),
            norm: rms_norm(device),
        }
    }

    /// The pretrained model (`cargo xtask fetch-weights --model smollm2-135m`), keeping the first
    /// `layers` of its thirty, computing in `precision`.
    pub fn pretrained(
        layers: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Result<(Self, Loaded), LoadError> {
        let mut model = Self::new(layers, device);
        let dropped = move |name: &str| {
            name.strip_prefix("layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|i| i.parse::<usize>().ok())
                .is_some_and(|i| i >= layers)
        };
        let loaded = load_safetensors(
            &mut model,
            &weights_path("smollm2-135m"),
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

    /// The embedding rows of `tokens` `[1, t]`, as `[1, t, HIDDEN]`.
    pub fn embed(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed_tokens.forward(tokens)
    }

    /// An empty cache for `max_len` positions of one sequence.
    pub fn cache(&self, max_len: usize, dtype: DType, device: &B::Device) -> KvCache<B> {
        let shape = [1, KV_HEADS, max_len, HEAD_DIM];
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

    /// The logits `[1, VOCAB]` after the last of `tokens` (`[1, t]`), which are appended to
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

    /// [`Self::forward`] from the embedded tokens `[1, t, HIDDEN]`: everything after the table
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
        let rope = Rope::new(start, t, dtype, &device);
        let act = cache.act.clone();
        let write = Write::At(start);
        for (index, (layer, c)) in self.layers.iter().zip(cache.layers.iter_mut()).enumerate() {
            x = layer.forward(x, &rope, c, &write, index, act.as_ref());
        }
        cache.len += t;
        self.head(x, t)
    }

    /// An empty [`FixedCache`] for `max_len` positions.
    pub fn fixed_cache(&self, max_len: usize, dtype: DType, device: &B::Device) -> FixedCache<B> {
        let shape = [1, KV_HEADS, max_len, HEAD_DIM];
        FixedCache {
            layers: self
                .layers
                .iter()
                .map(|_| LayerCache {
                    k: Tensor::zeros(shape, (device, dtype)),
                    v: Tensor::zeros(shape, (device, dtype)),
                })
                .collect(),
            max_len,
            rope: Rope::table(max_len, dtype, device),
            positions: Tensor::arange(0..max_len as i64, device),
        }
    }

    /// One decode step whose shapes do not depend on its position: the embedded token `x`
    /// (`[1, 1, HIDDEN]`) at `position` (`[1]`, below `cache.max_len()`), its keys and values
    /// written into `cache` there, the logits `[1, VOCAB]` returned. The position is a tensor,
    /// so the step can be captured once and replayed at every position
    /// (`burn_tt::TracedInference::capture_with_state`, with [`FixedCache::state`]).
    pub fn step_fixed(
        &self,
        mut x: Tensor<B, 3>,
        position: Tensor<B, 1, Int>,
        cache: &mut FixedCache<B>,
        dtype: DType,
    ) -> Tensor<B, 2> {
        assert_eq!(x.dims(), [1, 1, HIDDEN], "one token a step");
        if x.dtype() != dtype {
            x = x.cast(dtype);
        }
        let max_len = cache.max_len;
        let at = position.clone().expand([max_len]);
        let write = Write::Masked {
            // Expanded to the cache's shape here, once, rather than in each layer.
            here: cache
                .positions
                .clone()
                .equal(at.clone())
                .reshape([1, 1, max_len, 1])
                .expand([1, KV_HEADS, max_len, HEAD_DIM]),
            later: cache
                .positions
                .clone()
                .greater(at)
                .reshape([1, 1, 1, max_len]),
        };
        let rope = cache.rope.at(position);
        for (index, (layer, c)) in self.layers.iter().zip(cache.layers.iter_mut()).enumerate() {
            x = layer.forward(x, &rope, c, &write, index, None);
        }
        self.head(x, 1)
    }

    /// `c` tokens of a prompt (`x`, `[1, c, HIDDEN]`) from position `start` (`[1]`) on, with shapes
    /// that depend on neither: their keys and values written into `cache` at `start..start + c`,
    /// the logits `[1, VOCAB]` of chunk row `last` (`[1]`) returned. A prompt is its chunks in
    /// order, the last one padded (padding tokens write cache rows past the prompt, which a decode
    /// step overwrites before it reads them, and their rotary rows are clamped into the table);
    /// `last` is the prompt's last token within it. Like [`Self::step_fixed`], it can be captured
    /// once for a chunk size and replayed at every start (the prefill of `attention-jobs`' traced
    /// decode), and each row's arithmetic is [`Self::forward_embedded`]'s.
    pub fn step_chunk(
        &self,
        mut x: Tensor<B, 3>,
        start: Tensor<B, 1, Int>,
        last: Tensor<B, 1, Int>,
        cache: &mut FixedCache<B>,
        dtype: DType,
    ) -> Tensor<B, 2> {
        let [n, c, h] = x.dims();
        let max_len = cache.max_len;
        assert!(
            n == 1 && h == HIDDEN && c <= max_len,
            "a chunk of at most {max_len} tokens"
        );
        if x.dtype() != dtype {
            x = x.cast(dtype);
        }
        let positions = cache.positions.clone();
        let from = start.clone().expand([max_len]);
        let here = positions
            .clone()
            .greater_equal(from.clone())
            .bool_and(positions.clone().lower(from.clone().add_scalar(c as i64)))
            .reshape([1, 1, max_len, 1])
            .expand([1, KV_HEADS, max_len, HEAD_DIM]);
        // Each token's own position.
        let own = start.expand([c]) + positions.clone().narrow(0, 0, c);
        let later = positions
            .clone()
            .reshape([1, max_len])
            .expand([c, max_len])
            .greater(own.clone().reshape([c, 1]).expand([c, max_len]))
            .reshape([1, 1, c, max_len]);
        let offset = (positions.clone() - from).clamp(0, c as i64 - 1);
        let base = positions
            .narrow(0, 0, KV_HEADS)
            .mul_scalar(c as i64)
            .reshape([KV_HEADS, 1])
            .expand([KV_HEADS, max_len]);
        let index = (base + offset.reshape([1, max_len]).expand([KV_HEADS, max_len]))
            .reshape([KV_HEADS * max_len]);
        let write = Write::Chunk { here, later, index };
        let rope = cache.rope.at(own.clamp_max(max_len as i64 - 1));
        for (index, (layer, cl)) in self.layers.iter().zip(cache.layers.iter_mut()).enumerate() {
            x = layer.forward(x, &rope, cl, &write, index, None);
        }
        let row = self
            .norm
            .forward(x)
            .reshape([c, HIDDEN])
            .select(0, last)
            .reshape([1, HIDDEN]);
        self.project_out(row)
    }

    /// The logits `[1, VOCAB]` of the last of `t` positions of `x`.
    fn head(&self, x: Tensor<B, 3>, t: usize) -> Tensor<B, 2> {
        let last = self
            .norm
            .forward(x)
            .slice([0..1, t - 1..t, 0..HIDDEN])
            .reshape([1, HIDDEN]);
        self.project_out(last)
    }

    /// The logits `[1, VOCAB]` of a normed hidden row `[1, HIDDEN]`.
    fn project_out(&self, last: Tensor<B, 2>) -> Tensor<B, 2> {
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

/// The logits a run of next-token predictions produced.
#[derive(Debug)]
pub struct Generation {
    /// Wall time of each forward pass, the prompt's first, including reading its logits back.
    pub times: Vec<std::time::Duration>,
    /// The tokens chosen (greedy), or fed (teacher-forced), after the prompt.
    pub tokens: Vec<i32>,
    /// `logits[i]` is the prediction for the position after `prompt + tokens[..i]`, `[VOCAB]`.
    pub logits: Vec<Vec<f32>>,
}

impl Generation {
    /// Run one forward pass, read its logits back and record the wall time of both.
    fn step<B: Backend>(&mut self, forward: impl FnOnce() -> Tensor<B, 2>) {
        let t0 = std::time::Instant::now();
        self.logits.push(read(forward()));
        self.times.push(t0.elapsed());
        if std::env::var_os("TT_MODELS_TRACE").is_some() {
            eprintln!("forward {}: {:.2?}", self.times.len() - 1, t0.elapsed());
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

fn read<B: Backend>(logits: Tensor<B, 2>) -> Vec<f32> {
    // BF16 logits widen exactly; F32 ones are unchanged.
    logits
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("F32 logits")
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
