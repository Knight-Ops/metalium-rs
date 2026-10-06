//! A small deterministic MNIST CNN shared by the application and native gates.
use burn::module::{Module, Param};
use burn::nn::conv::Conv2d;
use burn::nn::pool::{AvgPool2d, AvgPool2dConfig};
use burn::nn::{Linear, PaddingConfig2d};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor, TensorData};

pub const CHANNELS: usize = 8;
pub const FEATURES: usize = CHANNELS * 3 * 3;
pub const PARAMETERS: usize = CHANNELS * 25 + CHANNELS + FEATURES * 10 + 10;

/// Initial values are generated on the host once, never inside a forward pass.
pub struct Init {
    pub conv_weight: TensorData,
    pub conv_bias: TensorData,
    pub linear_weight: TensorData,
    pub linear_bias: TensorData,
}

impl Default for Init {
    fn default() -> Self {
        let mut state = 0x3a15u64;
        let mut values = |n: usize, fan_in: usize| {
            let scale = 1.0 / (fan_in as f32).sqrt();
            (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f64 / (1u64 << 30) as f64 - 1.0) as f32 * scale
                })
                .collect::<Vec<_>>()
        };
        Self {
            conv_weight: TensorData::new(values(CHANNELS * 25, 25), [CHANNELS, 1, 5, 5]),
            conv_bias: TensorData::new(values(CHANNELS, 25), [CHANNELS]),
            linear_weight: TensorData::new(values(FEATURES * 10, FEATURES), [FEATURES, 10]),
            linear_bias: TensorData::new(values(10, FEATURES), [10]),
        }
    }
}

#[derive(Module, Debug)]
pub struct Cnn<B: Backend> {
    pub conv: Conv2d<B>,
    pub pool: AvgPool2d,
    pub head: Linear<B>,
}

impl<B: Backend> Cnn<B> {
    pub fn new(init: &Init, dtype: DType, device: &B::Device) -> Self {
        Self {
            conv: Conv2d {
                weight: Param::from_tensor(Tensor::from_data(
                    init.conv_weight.clone(),
                    (device, dtype),
                )),
                bias: Some(Param::from_tensor(Tensor::from_data(
                    init.conv_bias.clone(),
                    (device, dtype),
                ))),
                stride: [4, 4],
                kernel_size: [5, 5],
                dilation: [1, 1],
                groups: 1,
                padding: PaddingConfig2d::Valid,
            },
            pool: AvgPool2dConfig::new([2, 2]).init(),
            head: Linear {
                weight: Param::from_tensor(Tensor::from_data(
                    init.linear_weight.clone(),
                    (device, dtype),
                )),
                bias: Some(Param::from_tensor(Tensor::from_data(
                    init.linear_bias.clone(),
                    (device, dtype),
                ))),
            },
        }
    }

    /// Named storage boundaries retain F32 master parameters. F32 biases and
    /// pooling reductions promote results; compress activations explicitly.
    pub fn forward_with_precision<P: burn_tt::storage::PrecisionPolicy<B>>(
        &self,
        images: Tensor<B, 2>,
        policy: &P,
    ) -> Tensor<B, 2> {
        use burn::tensor::module::conv2d;
        use burn::tensor::ops::ConvOptions;
        let [batch, pixels] = images.dims();
        assert_eq!(pixels, 28 * 28);
        let x = policy.apply("cnn.input", images.reshape([batch, 1, 28, 28]));
        let weight = policy.apply("cnn.conv.weight", self.conv.weight.val());
        let bias = self
            .conv
            .bias
            .as_ref()
            .map(|b| policy.apply("cnn.conv.bias", b.val()));
        let x = conv2d(
            x,
            weight,
            bias,
            ConvOptions::new(
                self.conv.stride,
                [0, 0],
                self.conv.dilation,
                self.conv.groups,
            ),
        );
        let x = policy.apply("cnn.relu", relu(x));
        let features = self.pool.forward(x).reshape([batch, FEATURES]);
        let weight = policy.apply("cnn.head.weight", self.head.weight.val());
        let mut out = features.matmul(weight);
        if let Some(bias) = &self.head.bias {
            out = out + policy.apply("cnn.head.bias", bias.val()).unsqueeze();
        }
        policy.apply("cnn.logits", out)
    }

    /// Flattened dataset rows become NCHW; pooling leaves 8×3×3 features.
    pub fn forward(&self, images: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, pixels] = images.dims();
        assert_eq!(pixels, 28 * 28);
        let features = self
            .pool
            .forward(relu(self.conv.forward(images.reshape([batch, 1, 28, 28]))));
        self.head.forward(features.reshape([batch, FEATURES]))
    }
}
