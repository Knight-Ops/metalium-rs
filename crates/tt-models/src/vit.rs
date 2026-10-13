//! ViT-Tiny/16 at 224, in the layout of timm's `vit_tiny_patch16_224`.
//!
//! The one network of the ladder that exercises attention, layer norm and GELU end to end:
//! a 16x16 patch convolution, a class token and learned position embeddings, twelve pre-norm
//! blocks of multi-head self-attention and an MLP, a final norm, and the head on the class token.
//! Field names follow the checkpoint (`patch_embed.proj`, `blocks.N.attn.qkv`, `cls_token`), so
//! it loads with the PyTorch layout conversion and no renames.
//!
//! `depth` lets the simulator lane run the first few blocks of the real network (the unused
//! blocks' tensors are ignored on load); silicon runs all twelve.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Distribution, Tensor};

use crate::load::{load_safetensors, weights_path, Layout, LoadError, Loaded};

pub const DIM: usize = 192;
pub const HEADS: usize = 3;
pub const HEAD_DIM: usize = DIM / HEADS;
pub const MLP: usize = 4 * DIM;
pub const PATCH: usize = 16;
pub const IMAGE: usize = 224;
pub const TOKENS: usize = (IMAGE / PATCH) * (IMAGE / PATCH) + 1;
pub const FULL_DEPTH: usize = 12;
pub const CLASSES: usize = 1000;
/// timm builds the transformer's norms with this epsilon.
const EPS: f64 = 1e-6;

fn norm<B: Backend>(device: &B::Device) -> LayerNorm<B> {
    LayerNormConfig::new(DIM).with_epsilon(EPS).init(device)
}

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    proj: Conv2d<B>,
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    qkv: Linear<B>,
    proj: Linear<B>,
}

impl<B: Backend> Attention<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [n, t, _] = x.dims();
        let qkv = self.qkv.forward(x);
        // timm lays the fused projection out as (q, k, v), each (heads, head_dim).
        let part = |i: usize| {
            qkv.clone()
                .slice([0..n, 0..t, i * DIM..(i + 1) * DIM])
                .reshape([n, t, HEADS, HEAD_DIM])
                .swap_dims(1, 2)
        };
        let out = attention(
            part(0),
            part(1),
            part(2),
            None,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: false,
            },
        );
        self.proj.forward(out.swap_dims(1, 2).reshape([n, t, DIM]))
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attn: Attention<B>,
    norm2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            norm1: norm(device),
            attn: Attention {
                qkv: LinearConfig::new(DIM, 3 * DIM).init(device),
                proj: LinearConfig::new(DIM, DIM).init(device),
            },
            norm2: norm(device),
            mlp: Mlp {
                fc1: LinearConfig::new(DIM, MLP).init(device),
                fc2: LinearConfig::new(MLP, DIM).init(device),
            },
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.norm1.forward(x));
        let h = gelu(self.mlp.fc1.forward(self.norm2.forward(x.clone())));
        x + self.mlp.fc2.forward(h)
    }
}

#[derive(Module, Debug)]
pub struct VitTiny<B: Backend> {
    patch_embed: PatchEmbed<B>,
    cls_token: Param<Tensor<B, 3>>,
    pos_embed: Param<Tensor<B, 3>>,
    blocks: Vec<Block<B>>,
    norm: LayerNorm<B>,
    head: Linear<B>,
}

impl<B: Backend> VitTiny<B> {
    /// The architecture with `depth` blocks and its initialiser's weights.
    pub fn new(depth: usize, device: &B::Device) -> Self {
        assert!((1..=FULL_DEPTH).contains(&depth), "depth {depth}");
        Self {
            patch_embed: PatchEmbed {
                proj: Conv2dConfig::new([3, DIM], [PATCH, PATCH])
                    .with_stride([PATCH, PATCH])
                    .init(device),
            },
            cls_token: Param::from_tensor(Tensor::random(
                [1, 1, DIM],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            pos_embed: Param::from_tensor(Tensor::random(
                [1, TOKENS, DIM],
                Distribution::Normal(0.0, 0.02),
                device,
            )),
            blocks: (0..depth).map(|_| Block::new(device)).collect(),
            norm: norm(device),
            head: LinearConfig::new(DIM, CLASSES).init(device),
        }
    }

    /// The pretrained ImageNet network (`cargo xtask fetch-weights --model vit-tiny`), keeping the
    /// first `depth` of its twelve blocks.
    pub fn pretrained(depth: usize, device: &B::Device) -> Result<(Self, Loaded), LoadError> {
        let mut model = Self::new(depth, device);
        let dropped = move |name: &str| {
            name.strip_prefix("blocks.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|i| i.parse::<usize>().ok())
                .is_some_and(|i| i >= depth)
        };
        let loaded = load_safetensors(
            &mut model,
            &weights_path("vit-tiny"),
            &Layout {
                remaps: &[],
                pytorch: true,
                ignore: &dropped,
            },
        )?;
        Ok((model, loaded))
    }

    /// Logits `[batch, 1000]` for a normalised `[batch, 3, 224, 224]` image.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.patch_embed.proj.forward(x);
        let [n, c, h, w] = x.dims();
        let patches = x.reshape([n, c, h * w]).swap_dims(1, 2);
        let cls = self.cls_token.val().expand([n, 1, DIM]);
        let mut x = Tensor::cat(vec![cls, patches], 1) + self.pos_embed.val();
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = self.norm.forward(x);
        let cls = x.slice([0..n, 0..1, 0..DIM]).reshape([n, DIM]);
        self.head.forward(cls)
    }
}
