//! The FLUX.2 VAE's decoder (`diffusers`' `AutoencoderKLFlux2`, 0.38.0), and the step MOSS-Image
//! takes from its latent tokens to the decoder's input.
//!
//! The decoder is diffusers' `Decoder`: a 3 x 3 convolution in, a middle block (a residual block,
//! single-head self-attention over the pixels, a residual block), four up blocks of three
//! residual blocks each (a nearest 2x upsampling and a convolution after each but the last), then
//! group norm, SiLU and a convolution out. Every group norm has 32 groups and `eps = 1e-6`. A
//! residual block is `x + conv2(silu(norm2(conv1(silu(norm1(x))))))`, through a 1 x 1 convolution
//! when the width changes.
//!
//! The nearest upsampling is a broadcast and a reshape, not `interpolate`, so the device runs it
//! as element moves.
//!
//! Before decoding, MOSS-Image un-normalises each of the 128 latent channels by the VAE's batch
//! statistics (`z sqrt(var + 1e-4) + mean`, `bn` in the checkpoint) and unfolds each 2 x 2 patch:
//! channel `4 c + 2 i + j` of token `(h, w)` is channel `c` of pixel `(2 h + i, 2 w + j)`.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::Tensor;

use crate::load::{load_safetensors, weights_files, Layout, LoadError, Loaded};

/// The decoder's shape (`vae/config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct VaeConfig {
    pub latent_channels: usize,
    /// The up blocks' widths, from the image's side (`decoder_block_out_channels`).
    pub block_out_channels: [usize; 4],
    pub layers_per_block: usize,
    pub groups: usize,
    pub out_channels: usize,
    /// The latent statistics' `eps` (`batch_norm_eps`).
    pub batch_norm_eps: f64,
}

pub const FLUX2_VAE: VaeConfig = VaeConfig {
    latent_channels: 32,
    block_out_channels: [96, 192, 384, 384],
    layers_per_block: 2,
    groups: 32,
    out_channels: 3,
    batch_norm_eps: 1e-4,
};

const EPS: f64 = 1e-6;

fn conv<B: Backend>(inp: usize, out: usize, kernel: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([inp, out], [kernel, kernel])
        .with_padding(PaddingConfig2d::Explicit(
            kernel / 2,
            kernel / 2,
            kernel / 2,
            kernel / 2,
        ))
        .init(device)
}

fn norm<B: Backend>(groups: usize, channels: usize, device: &B::Device) -> GroupNorm<B> {
    GroupNormConfig::new(groups, channels)
        .with_epsilon(EPS)
        .init(device)
}

#[derive(Module, Debug)]
pub struct Resnet<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    conv_shortcut: Option<Conv2d<B>>,
}

impl<B: Backend> Resnet<B> {
    fn new(groups: usize, inp: usize, out: usize, device: &B::Device) -> Self {
        Self {
            norm1: norm(groups, inp, device),
            conv1: conv(inp, out, 3, device),
            norm2: norm(groups, out, device),
            conv2: conv(out, out, 3, device),
            conv_shortcut: (inp != out).then(|| conv(inp, out, 1, device)),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.conv1.forward(silu(self.norm1.forward(x.clone())));
        let h = self.conv2.forward(silu(self.norm2.forward(h)));
        let x = match &self.conv_shortcut {
            Some(c) => c.forward(x),
            None => x,
        };
        x + h
    }
}

/// Single-head self-attention over the pixels, with a residual connection.
#[derive(Module, Debug)]
pub struct PixelAttention<B: Backend> {
    group_norm: GroupNorm<B>,
    to_q: Linear<B>,
    to_k: Linear<B>,
    to_v: Linear<B>,
    /// `to_out.0`.
    to_out: Vec<Linear<B>>,
}

impl<B: Backend> PixelAttention<B> {
    fn new(groups: usize, channels: usize, device: &B::Device) -> Self {
        let lin = || LinearConfig::new(channels, channels).init(device);
        Self {
            group_norm: norm(groups, channels, device),
            to_q: lin(),
            to_k: lin(),
            to_v: lin(),
            to_out: vec![lin()],
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, c, h, w] = x.dims();
        let pixels = self
            .group_norm
            .forward(x.clone())
            .reshape([n, c, h * w])
            .swap_dims(1, 2);
        let head = |l: &Linear<B>| l.forward(pixels.clone()).unsqueeze_dim::<4>(1);
        let out = attention(
            head(&self.to_q),
            head(&self.to_k),
            head(&self.to_v),
            None,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: false,
            },
        )
        .reshape([n, h * w, c]);
        let out = self.to_out[0]
            .forward(out)
            .swap_dims(1, 2)
            .reshape([n, c, h, w]);
        out + x
    }
}

#[derive(Module, Debug)]
pub struct MidBlock<B: Backend> {
    resnets: Vec<Resnet<B>>,
    attentions: Vec<PixelAttention<B>>,
}

#[derive(Module, Debug)]
pub struct Upsampler<B: Backend> {
    conv: Conv2d<B>,
}

/// `x` (`[n, c, h, w]`) at twice the resolution, each pixel repeated 2 x 2 (nearest): each row
/// repeated, then each column as a row of the transpose. Whole rows move as blocks; broadcasting a
/// trailing axis of one instead (`[.., w, 1] -> [.., w, 2]`) interleaves single elements, which
/// the card's block copier refuses past ~400 K elements (`silicon_probe_moss_image_ops`,
/// 2026-10-30).
pub fn upsample_nearest<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let rows = |x: Tensor<B, 4>| {
        let [n, c, h, w] = x.dims();
        x.reshape([n, c, h, 1, w])
            .expand([n, c, h, 2, w])
            .reshape([n, c, 2 * h, w])
    };
    rows(rows(x).swap_dims(2, 3)).swap_dims(2, 3)
}

#[derive(Module, Debug)]
pub struct UpBlock<B: Backend> {
    resnets: Vec<Resnet<B>>,
    upsamplers: Vec<Upsampler<B>>,
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    conv_in: Conv2d<B>,
    mid_block: MidBlock<B>,
    up_blocks: Vec<UpBlock<B>>,
    conv_norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
}

/// The latent statistics MOSS-Image un-normalises by (`bn`, a batch norm's running state).
#[derive(Module, Debug)]
pub struct LatentStats<B: Backend> {
    running_mean: Param<Tensor<B, 1>>,
    running_var: Param<Tensor<B, 1>>,
}

/// The FLUX.2 VAE's decoding half.
#[derive(Module, Debug)]
pub struct Vae<B: Backend> {
    bn: LatentStats<B>,
    post_quant_conv: Conv2d<B>,
    decoder: Decoder<B>,
    /// Not a parameter: the module derive skips it.
    batch_norm_eps: f64,
}

impl<B: Backend> Vae<B> {
    pub fn new(cfg: &VaeConfig, device: &B::Device) -> Self {
        let g = cfg.groups;
        let widths: Vec<usize> = cfg.block_out_channels.iter().rev().copied().collect();
        let top = widths[0];
        let mut prev = top;
        let up_blocks = widths
            .iter()
            .enumerate()
            .map(|(i, &out)| {
                let block = UpBlock {
                    resnets: (0..cfg.layers_per_block + 1)
                        .map(|j| Resnet::new(g, if j == 0 { prev } else { out }, out, device))
                        .collect(),
                    upsamplers: (i + 1 < widths.len())
                        .then(|| Upsampler {
                            conv: conv(out, out, 3, device),
                        })
                        .into_iter()
                        .collect(),
                };
                prev = out;
                block
            })
            .collect();
        let patched = 4 * cfg.latent_channels;
        Self {
            bn: LatentStats {
                running_mean: Param::from_tensor(Tensor::zeros([patched], device)),
                running_var: Param::from_tensor(Tensor::ones([patched], device)),
            },
            post_quant_conv: conv(cfg.latent_channels, cfg.latent_channels, 1, device),
            decoder: Decoder {
                conv_in: conv(cfg.latent_channels, top, 3, device),
                mid_block: MidBlock {
                    resnets: vec![
                        Resnet::new(g, top, top, device),
                        Resnet::new(g, top, top, device),
                    ],
                    attentions: vec![PixelAttention::new(g, top, device)],
                },
                up_blocks,
                conv_norm_out: norm(g, cfg.block_out_channels[0], device),
                conv_out: conv(cfg.block_out_channels[0], cfg.out_channels, 3, device),
            },
            batch_norm_eps: cfg.batch_norm_eps,
        }
    }

    /// The VAE in `vae/` of `vendor/weights/<weights>/`: its decoder, `post_quant_conv` and latent
    /// statistics (the encoder and `quant_conv` are left in the file, by name).
    pub fn pretrained(
        cfg: &VaeConfig,
        weights: &str,
        device: &B::Device,
    ) -> Result<(Self, Loaded), LoadError> {
        let mut model = Self::new(cfg, device);
        let ignore = |name: &str| {
            name.starts_with("encoder.")
                || name.starts_with("quant_conv.")
                || name == "bn.num_batches_tracked"
        };
        let loaded = load_safetensors(
            &mut model,
            &weights_files(weights, &["vae/diffusion_pytorch_model.safetensors"])[0],
            &Layout {
                remaps: &[],
                pytorch: true,
                ignore: &ignore,
            },
        )?;
        Ok((model, loaded))
    }

    /// The decoder's input for MOSS-Image's latent tokens (`[1, 4 latent_channels, H, W]`):
    /// un-normalised by the latent statistics, each 2 x 2 patch unfolded into pixels
    /// (`[1, latent_channels, 2 H, 2 W]`).
    pub fn unpatch(&self, latents: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, c, h, w] = latents.dims();
        let std = self
            .bn
            .running_var
            .val()
            .add_scalar(self.batch_norm_eps)
            .sqrt()
            .reshape([1, c, 1, 1]);
        let mean = self.bn.running_mean.val().reshape([1, c, 1, 1]);
        let z = latents * std + mean;
        z.reshape([n, c / 4, 2, 2, h, w])
            .permute([0, 1, 4, 2, 5, 3])
            .reshape([n, c / 4, 2 * h, 2 * w])
    }

    /// The image `[1, 3, 8 h, 8 w]` (in `[-1, 1]`) of the decoder input `z` (`[1, latent_channels,
    /// h, w]`).
    pub fn decode(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let d = &self.decoder;
        let mut x = d.conv_in.forward(self.post_quant_conv.forward(z));
        let mid = &d.mid_block;
        x = mid.resnets[0].forward(x);
        x = mid.attentions[0].forward(x);
        x = mid.resnets[1].forward(x);
        for block in &d.up_blocks {
            for r in &block.resnets {
                x = r.forward(x);
            }
            for u in &block.upsamplers {
                x = u.conv.forward(upsample_nearest(x));
            }
        }
        let x = d.conv_out.forward(silu(d.conv_norm_out.forward(x)));
        x.clamp(-1.0, 1.0)
    }
}
