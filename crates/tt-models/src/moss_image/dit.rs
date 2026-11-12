//! MOSS-Image's diffusion transformer: one stream of image and text tokens through 42 blocks of
//! full (non-causal) attention and a bounded gated MLP, every block modulated (adaLN: a shift, a
//! scale and a gate for each half) by one projection of the timestep that all blocks share.
//!
//! The reference is `transformer/pipeline_moss_image.py` at the pinned revision, and these are its
//! details:
//!
//! * Its RMSNorm is `rms(x)` in F32, cast back, times `weight`; `eps` is `1e-6` throughout.
//! * The MLP's activation is "situ", `4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25)` of the gate `g`
//!   and the hidden `h` (`ffn_up_gate` emits `h` then `g`).
//! * The rotary embedding has four axes (an unused one, the text position, the image row and
//!   column) of `head_dim / 8` bands each, `base^(-k / bands)`; an image token sits at
//!   `(0, text_len, row - (H - 1) / 2, col - (W - 1) / 2)`, text token `j` at `(0, j, 0, 0)`. Pair
//!   `a bands + k` of a head turns by axis `a`'s coordinate times band `k`, in the rotate-half
//!   layout.
//! * The text is first projected to the stream's width and refined by two blocks of its own (with
//!   biases, no modulation), at its own rotary rows.
//! * The timestep's sinusoid (`cos` then `sin`, `10000^(-k / half)`) is computed in F64.

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, silu};
use burn::tensor::backend::Backend;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, Tensor, TensorData};

use crate::llama::project;
use crate::qwen35::rms;
use crate::rope::Rope;

/// The shape of a MOSS-Image transformer (`transformer/config.json`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitConfig {
    pub dim: usize,
    pub ffn_dim: usize,
    pub heads: usize,
    pub layers: usize,
    /// The text encoder's width.
    pub text_dim: usize,
    /// The timestep sinusoid's width.
    pub freq_dim: usize,
    /// Channels of a latent token (the VAE's 32 latents in 2 x 2 patches).
    pub latent_channels: usize,
    pub rope_base: f64,
}

/// MOSS-Image-Base: width 3072, 24 heads of 128, 42 layers, an MLP of 8,192, text of width 4,096.
pub const MOSS_IMAGE_BASE: DitConfig = DitConfig {
    dim: 3072,
    ffn_dim: 8192,
    heads: 24,
    layers: 42,
    text_dim: 4096,
    freq_dim: 256,
    latent_channels: 128,
    rope_base: 10000.0,
};

impl DitConfig {
    fn head_dim(&self) -> usize {
        self.dim / self.heads
    }

    /// Rotary bands an axis.
    fn bands(&self) -> usize {
        self.head_dim() / 8
    }
}

const EPS: f64 = 1e-6;

/// `rms(x)` in F32, cast back to `x`'s dtype, times `weight`.
#[derive(Module, Debug)]
pub struct Norm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
}

impl<B: Backend> Norm<B> {
    fn new(width: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([width], device)),
        }
    }

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = x.dtype();
        rms(x, EPS).cast(dtype) * self.weight.val().unsqueeze::<D>()
    }
}

/// `4 tanh(gate / 4) sigmoid(gate) * 25 tanh(hidden / 25)`.
fn situ<B: Backend>(hidden: Tensor<B, 3>, gate: Tensor<B, 3>) -> Tensor<B, 3> {
    let g = gate.clone().div_scalar(4.0).tanh().mul_scalar(4.0) * sigmoid(gate);
    g * hidden.div_scalar(25.0).tanh().mul_scalar(25.0)
}

fn linear<B: Backend>(inp: usize, out: usize, bias: bool, device: &B::Device) -> Linear<B> {
    LinearConfig::new(inp, out).with_bias(bias).init(device)
}

/// Full attention over `[1, n, heads head_dim]` queries, keys and values, all rotated by `rope`.
fn attend<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    rope: &Rope<B>,
    heads: usize,
) -> Tensor<B, 3> {
    let [n, t, width] = q.dims();
    let d = width / heads;
    let split = |x: Tensor<B, 3>| x.reshape([n, t, heads, d]).swap_dims(1, 2);
    let out = attention(
        rope.apply(split(q)),
        rope.apply(split(k)),
        split(v),
        None,
        None,
        AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: false,
        },
    );
    out.swap_dims(1, 2).reshape([n, t, width])
}

/// Each head's norm of `x` (`[1, n, heads head_dim]`).
fn head_norm<B: Backend>(norm: &Norm<B>, x: Tensor<B, 3>, heads: usize) -> Tensor<B, 3> {
    let [n, t, width] = x.dims();
    norm.forward(x.reshape([n, t, heads, width / heads]))
        .reshape([n, t, width])
}

/// A text-refinement block: attention and the situ MLP, with biases, unmodulated.
#[derive(Module, Debug)]
pub struct TextBlock<B: Backend> {
    norm1: Norm<B>,
    norm2: Norm<B>,
    q_norm: Norm<B>,
    k_norm: Norm<B>,
    qkv: Linear<B>,
    proj: Linear<B>,
    ffn_gate: Linear<B>,
    ffn_up: Linear<B>,
    ffn_down: Linear<B>,
}

impl<B: Backend> TextBlock<B> {
    fn new(cfg: &DitConfig, device: &B::Device) -> Self {
        let (dim, ff) = (cfg.dim, cfg.ffn_dim);
        Self {
            norm1: Norm::new(dim, device),
            norm2: Norm::new(dim, device),
            q_norm: Norm::new(cfg.head_dim(), device),
            k_norm: Norm::new(cfg.head_dim(), device),
            qkv: linear(dim, 3 * dim, true, device),
            proj: linear(dim, dim, true, device),
            ffn_gate: linear(dim, ff, true, device),
            ffn_up: linear(dim, ff, true, device),
            ffn_down: linear(ff, dim, true, device),
        }
    }

    fn forward(&self, cfg: &DitConfig, x: Tensor<B, 3>, rope: &Rope<B>) -> Tensor<B, 3> {
        let [n, t, dim] = x.dims();
        let qkv = project(self.norm1.forward(x.clone()), &self.qkv);
        let part = |i: usize| qkv.clone().slice([0..n, 0..t, i * dim..(i + 1) * dim]);
        let q = head_norm(&self.q_norm, part(0), cfg.heads);
        let k = head_norm(&self.k_norm, part(1), cfg.heads);
        let x = x + project(attend(q, k, part(2), rope, cfg.heads), &self.proj);
        let y = self.norm2.forward(x.clone());
        let hidden = situ(project(y.clone(), &self.ffn_up), project(y, &self.ffn_gate));
        x + project(hidden, &self.ffn_down)
    }
}

#[derive(Module, Debug)]
pub struct TextEmbedder<B: Backend> {
    proj: Linear<B>,
    blocks: Vec<TextBlock<B>>,
}

/// A main block's projections (`proj.q`, `proj.k`, ...), without biases.
#[derive(Module, Debug)]
pub struct BlockProj<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    /// `[dim, 2 ffn_dim]`: the hidden half, then the gate half.
    ffn_up_gate: Linear<B>,
    ffn_down: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    norm1: Norm<B>,
    norm2: Norm<B>,
    q_norm: Norm<B>,
    k_norm: Norm<B>,
    proj: BlockProj<B>,
}

/// The six modulation rows every block shares, each `[1, 1, dim]`: shift, scale and gate of the
/// attention half, then of the MLP half.
struct Modulation<B: Backend> {
    rows: [Tensor<B, 3>; 6],
}

impl<B: Backend> Block<B> {
    fn new(cfg: &DitConfig, device: &B::Device) -> Self {
        let (dim, ff) = (cfg.dim, cfg.ffn_dim);
        Self {
            norm1: Norm::new(dim, device),
            norm2: Norm::new(dim, device),
            q_norm: Norm::new(cfg.head_dim(), device),
            k_norm: Norm::new(cfg.head_dim(), device),
            proj: BlockProj {
                q: linear(dim, dim, false, device),
                k: linear(dim, dim, false, device),
                v: linear(dim, dim, false, device),
                o: linear(dim, dim, false, device),
                ffn_up_gate: linear(dim, 2 * ff, false, device),
                ffn_down: linear(ff, dim, false, device),
            },
        }
    }

    fn forward(
        &self,
        cfg: &DitConfig,
        x: Tensor<B, 3>,
        rope: &Rope<B>,
        m: &Modulation<B>,
    ) -> Tensor<B, 3> {
        let [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f] = m.rows.clone();
        let p = &self.proj;
        let y = self.norm1.forward(x.clone()) * scale_a.add_scalar(1.0) + shift_a;
        let q = head_norm(&self.q_norm, project(y.clone(), &p.q), cfg.heads);
        let k = head_norm(&self.k_norm, project(y.clone(), &p.k), cfg.heads);
        let attended = attend(q, k, project(y, &p.v), rope, cfg.heads);
        let x = x + project(attended, &p.o) * gate_a;
        let y = self.norm2.forward(x.clone()) * scale_f.add_scalar(1.0) + shift_f;
        let both = project(y, &p.ffn_up_gate);
        let [n, t, _] = both.dims();
        let ff = cfg.ffn_dim;
        let hidden = both.clone().slice([0..n, 0..t, 0..ff]);
        let gate = both.slice([0..n, 0..t, ff..2 * ff]);
        x + project(situ(hidden, gate), &p.ffn_down) * gate_f
    }
}

/// The MOSS-Image diffusion transformer.
#[derive(Module, Debug)]
pub struct Dit<B: Backend> {
    video_embedder: Linear<B>,
    text_embedder: TextEmbedder<B>,
    /// `time_embedding.0` and `time_embedding.2` (a SiLU between).
    time_in: Linear<B>,
    time_out: Linear<B>,
    shared_block_modulation: Linear<B>,
    layers: Vec<Block<B>>,
    final_norm_video: Norm<B>,
    final_linear_video: Linear<B>,
    video_head_modulation: Linear<B>,
    /// Not a parameter: the module derive skips it.
    config: DitConfig,
}

impl<B: Backend> Dit<B> {
    /// The architecture `config` describes, with its initialiser's weights.
    pub fn new(config: &DitConfig, device: &B::Device) -> Self {
        let (dim, c) = (config.dim, config.latent_channels);
        Self {
            video_embedder: linear(c, dim, true, device),
            text_embedder: TextEmbedder {
                proj: linear(config.text_dim, dim, true, device),
                blocks: (0..2).map(|_| TextBlock::new(config, device)).collect(),
            },
            time_in: linear(config.freq_dim, dim, true, device),
            time_out: linear(dim, dim, true, device),
            shared_block_modulation: linear(dim, 6 * dim, true, device),
            layers: (0..config.layers)
                .map(|_| Block::new(config, device))
                .collect(),
            final_norm_video: Norm::new(dim, device),
            final_linear_video: linear(dim, c, false, device),
            video_head_modulation: linear(dim, 2 * dim, true, device),
            config: *config,
        }
    }

    pub fn config(&self) -> &DitConfig {
        &self.config
    }

    /// The rotary tables of `height x width` image tokens then `text` text tokens.
    fn rope(&self, height: usize, width: usize, text: usize, device: &B::Device) -> Rope<B> {
        let bands = self.config.bands();
        // As the reference builds them: in F32, `1 / base^(k / bands)`.
        let freqs: Vec<f32> = (0..bands)
            .map(|k| 1.0 / (self.config.rope_base as f32).powf(k as f32 / bands as f32))
            .collect();
        let (mid_h, mid_w) = ((height as f32 - 1.0) / 2.0, (width as f32 - 1.0) / 2.0);
        let image = (0..height).flat_map(|r| {
            (0..width).map(move |c| [0.0, text as f32, r as f32 - mid_h, c as f32 - mid_w])
        });
        let words = (0..text).map(|j| [0.0, j as f32, 0.0, 0.0]);
        let angles: Vec<f32> = image
            .chain(words)
            .flat_map(|axes| {
                let freqs = &freqs;
                axes.into_iter()
                    .flat_map(move |coord| freqs.iter().map(move |f| coord * f))
            })
            .collect();
        let t = height * width + text;
        Rope::from_angles(angles, t, 4 * bands, DType::F32, device)
    }

    /// The timestep's sinusoid `[1, freq_dim]`: `cos` then `sin` of `t 10000^(-k / half)`, in F64.
    fn sinusoid(&self, timestep: f32, device: &B::Device) -> Tensor<B, 2> {
        let half = self.config.freq_dim / 2;
        let phase: Vec<f64> = (0..half)
            .map(|k| timestep as f64 * 10000f64.powf(-(k as f64) / half as f64))
            .collect();
        let values: Vec<f32> = phase
            .iter()
            .map(|p| p.cos() as f32)
            .chain(phase.iter().map(|p| p.sin() as f32))
            .collect();
        Tensor::from_data(TensorData::new(values, [1, 2 * half]), device)
    }

    /// The predicted velocity `[1, latent_channels, H, W]` for `latents` (same shape) at
    /// `timestep` (the scheduler's, `sigma * 1000`), conditioned on `context` (`[1, S,
    /// text_dim]`, the text encoder's last hidden states), with the blocks in `skip` left out.
    pub fn forward(
        &self,
        latents: Tensor<B, 4>,
        timestep: f32,
        context: Tensor<B, 3>,
        skip: &[usize],
    ) -> Tensor<B, 4> {
        let cfg = &self.config;
        let [n, c, height, width] = latents.dims();
        assert_eq!((n, c), (1, cfg.latent_channels), "one latent at a time");
        let [_, text, _] = context.dims();
        let device = latents.device();
        let (dim, image) = (cfg.dim, height * width);

        let time = project(
            silu(self.time_in.forward(self.sinusoid(timestep, &device))).unsqueeze_dim(0),
            &self.time_out,
        );
        let rows = project(time.clone(), &self.shared_block_modulation);
        let row =
            |m: &Tensor<B, 3>, i: usize| m.clone().slice([0..1, 0..1, i * dim..(i + 1) * dim]);
        let m = Modulation {
            rows: std::array::from_fn(|i| row(&rows, i)),
        };
        let head = project(time, &self.video_head_modulation);

        let rope = self.rope(height, width, text, &device);
        let video = latents.reshape([1, c, image]).swap_dims(1, 2);
        let video = project(video, &self.video_embedder);
        let text_rope = rope.rows(image, image + text);
        let mut words = project(context, &self.text_embedder.proj);
        for block in &self.text_embedder.blocks {
            words = block.forward(cfg, words, &text_rope);
        }
        let mut x = Tensor::cat(vec![video, words], 1);
        for (i, layer) in self.layers.iter().enumerate() {
            if !skip.contains(&i) {
                x = layer.forward(cfg, x, &rope, &m);
            }
        }
        let x = self
            .final_norm_video
            .forward(x.slice([0..1, 0..image, 0..dim]));
        let x = x * row(&head, 1).add_scalar(1.0) + row(&head, 0);
        project(x, &self.final_linear_video)
            .swap_dims(1, 2)
            .reshape([1, c, height, width])
    }
}
