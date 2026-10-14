//! ResNet-18, in the layout of timm's `resnet18` (the same network as torchvision's).
//!
//! Written with Burn's own layers, generic over the backend, so one definition runs on burn-tt
//! and on burn-flex. Field names follow the checkpoint's tensor names (`conv1`, `bn1`,
//! `layer1.0.conv2`, `fc`), so loading needs one rename: PyTorch's `downsample` is a
//! `Sequential(conv, bn)`, keys `downsample.0` and `downsample.1`.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::load::{load_safetensors, weights_path, Layout, LoadError, Loaded};

pub const CLASSES: usize = 1000;

fn conv<B: Backend>(
    inp: usize,
    out: usize,
    k: usize,
    stride: usize,
    pad: usize,
    device: &B::Device,
) -> Conv2d<B> {
    Conv2dConfig::new([inp, out], [k, k])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
        .with_bias(false)
        .init(device)
}

fn bn<B: Backend>(channels: usize, device: &B::Device) -> BatchNorm<B> {
    // PyTorch's default, which the checkpoint was trained with.
    BatchNormConfig::new(channels)
        .with_epsilon(1e-5)
        .init(device)
}

/// The 1x1 convolution and norm on a block's shortcut when its shape changes.
#[derive(Module, Debug)]
pub struct Downsample<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
}

#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    downsample: Option<Downsample<B>>,
}

impl<B: Backend> BasicBlock<B> {
    fn new(inp: usize, out: usize, stride: usize, device: &B::Device) -> Self {
        let downsample = (stride != 1 || inp != out).then(|| Downsample {
            conv: conv(inp, out, 1, stride, 0, device),
            bn: bn(out, device),
        });
        Self {
            conv1: conv(inp, out, 3, stride, 1, device),
            bn1: bn(out, device),
            conv2: conv(out, out, 3, 1, 1, device),
            bn2: bn(out, device),
            downsample,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = relu(self.bn1.forward(self.conv1.forward(x.clone())));
        let out = self.bn2.forward(self.conv2.forward(out));
        let shortcut = match &self.downsample {
            Some(d) => d.bn.forward(d.conv.forward(x)),
            None => x,
        };
        relu(out + shortcut)
    }
}

#[derive(Module, Debug)]
pub struct ResNet18<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    maxpool: MaxPool2d,
    layer1: Vec<BasicBlock<B>>,
    layer2: Vec<BasicBlock<B>>,
    layer3: Vec<BasicBlock<B>>,
    layer4: Vec<BasicBlock<B>>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,
}

impl<B: Backend> ResNet18<B> {
    /// The architecture with its initialiser's weights; [`Self::pretrained`] loads the real ones.
    pub fn new(device: &B::Device) -> Self {
        let stage = |inp, out, stride| {
            vec![
                BasicBlock::new(inp, out, stride, device),
                BasicBlock::new(out, out, 1, device),
            ]
        };
        Self {
            conv1: conv(3, 64, 7, 2, 3, device),
            bn1: bn(64, device),
            maxpool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(),
            layer1: stage(64, 64, 1),
            layer2: stage(64, 128, 2),
            layer3: stage(128, 256, 2),
            layer4: stage(256, 512, 2),
            avgpool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc: LinearConfig::new(512, CLASSES).init(device),
        }
    }

    /// The pretrained ImageNet network (`cargo xtask fetch-weights --model resnet18`).
    pub fn pretrained(device: &B::Device) -> Result<(Self, Loaded), LoadError> {
        let mut model = Self::new(device);
        let loaded = load_safetensors(
            &mut model,
            &weights_path("resnet18"),
            &Layout {
                remaps: &[
                    (r"\.downsample\.0\.", ".downsample.conv."),
                    (r"\.downsample\.1\.", ".downsample.bn."),
                ],
                pytorch: true,
                // PyTorch's step counters for the running statistics: training state, unused here.
                ignore: &|name| name.ends_with(".num_batches_tracked"),
            },
        )?;
        Ok((model, loaded))
    }

    /// Logits `[batch, 1000]` for a normalised `[batch, 3, h, w]` image.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = self
            .maxpool
            .forward(relu(self.bn1.forward(self.conv1.forward(x))));
        for layer in [&self.layer1, &self.layer2, &self.layer3, &self.layer4] {
            for block in layer {
                x = block.forward(x);
            }
        }
        let x = self.avgpool.forward(x);
        let [n, c, _, _] = x.dims();
        self.fc.forward(x.reshape([n, c]))
    }
}
