//! A small transformer, made of Burn's own modules: the general-model
//! benchmark next to MNIST's MLP, and the model `tt-tests`'
//! `step59_burn_transformer` gate trains.
//!
//! It exercises what a model past MNIST is made of: an embedding, rank-3 and
//! rank-4 activations, batched matmuls inside attention, the heads' reshapes
//! and `swap_dims`, softmax, layer norm, GELU, and Linears over a
//! `[batch, seq, d]` input. `burn::nn`'s `Embedding`, a pre-norm
//! `TransformerEncoder` and a `Linear` head, `CrossEntropyLoss`, `Sgd`; the
//! task is next-token prediction on arithmetic progressions mod the
//! vocabulary, which is learnable from the current token alone.

use std::time::{Duration, Instant};

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput,
};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};

pub const VOCAB: usize = 64;
pub const D_MODEL: usize = 64;
pub const D_FF: usize = 128;
pub const HEADS: usize = 2;
pub const LAYERS: usize = 1;
pub const BATCH: usize = 4;
pub const SEQ: usize = 32;
pub const LR: f64 = 0.05;

#[derive(Module, Debug)]
pub struct TinyTransformer<B: Backend> {
    embed: Embedding<B>,
    encoder: TransformerEncoder<B>,
    head: Linear<B>,
}

impl<B: Backend> TinyTransformer<B> {
    /// A model with Burn's initialisation; [`load`] replaces its weights.
    pub fn new(device: &B::Device) -> Self {
        TinyTransformer {
            embed: EmbeddingConfig::new(VOCAB, D_MODEL).init(device),
            encoder: TransformerEncoderConfig::new(D_MODEL, D_FF, HEADS, LAYERS)
                .with_norm_first(true)
                .with_dropout(0.0)
                .init(device),
            head: LinearConfig::new(D_MODEL, VOCAB).init(device),
        }
    }

    /// Logits for every position, as `[batch * seq, vocab]`.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let x = self.embed.forward(tokens);
        let x = self.encoder.forward(TransformerEncoderInput::new(x));
        self.head.forward(x).reshape([BATCH * SEQ, VOCAB])
    }
}

/// Every float parameter's data, in visiting order.
struct Collect(Vec<TensorData>);

impl<B: Backend> ModuleVisitor<B> for Collect {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.0.push(param.val().into_data());
    }
}

/// Replace every float parameter, in visiting order, by the next of `0`.
struct Load<I>(I);

impl<B: Backend, I: Iterator<Item = TensorData>> ModuleMapper<B> for Load<I> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, _) = param.consume();
        let data = self.0.next().expect("as many parameters as were collected");
        assert_eq!(data.shape.to_vec(), tensor.dims().to_vec());
        Param::initialized(id, Tensor::from_data(data, &tensor.device()).require_grad())
    }
}

/// The weights every run starts from: Flex's initialiser, seeded, so two
/// backends start from the same bits without drawing random numbers alike.
pub fn init(seed: u64) -> Vec<TensorData> {
    Flex::seed(&FlexDevice, seed);
    let mut c = Collect(Vec::new());
    TinyTransformer::<Flex>::new(&FlexDevice).visit(&mut c);
    c.0
}

/// A model on `device` holding `weights` ([`init`]).
pub fn load<B: Backend>(weights: &[TensorData], device: &B::Device) -> TinyTransformer<B> {
    TinyTransformer::<B>::new(device).map(&mut Load(weights.iter().cloned()))
}

/// Step `step`'s batch: sequence `b` is an arithmetic progression mod
/// [`VOCAB`]. The inputs, and the targets (the inputs shifted by one).
pub fn batch<B: Backend>(
    step: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 1, Int>) {
    let mut x = Vec::with_capacity(BATCH * SEQ);
    let mut y = Vec::with_capacity(BATCH * SEQ);
    for b in 0..BATCH {
        let stride = 1 + (b + step) % 3;
        let start = (7 * b + 3 * step) % VOCAB;
        for i in 0..SEQ {
            x.push(((start + stride * i) % VOCAB) as i32);
            y.push(((start + stride * (i + 1)) % VOCAB) as i32);
        }
    }
    (
        Tensor::from_data(TensorData::new(x, [BATCH, SEQ]), device),
        Tensor::from_data(TensorData::new(y, [BATCH * SEQ]), device),
    )
}

/// A training run: each step's loss and wall-clock time (the loss read back
/// inside it, as a training loop that logs it does).
pub struct Run {
    pub losses: Vec<f32>,
    pub times: Vec<Duration>,
}

impl Run {
    /// Mean time per step after the first, which pays one-time costs
    /// (programs built and placed, parameters uploaded).
    pub fn steady(&self) -> Option<Duration> {
        let rest = self.times.get(1..).filter(|r| !r.is_empty())?;
        Some(rest.iter().sum::<Duration>() / rest.len() as u32)
    }
}

/// Train `steps` steps from `weights`.
pub fn train<B: AutodiffBackend>(weights: &[TensorData], steps: usize, device: &B::Device) -> Run {
    let mut model = load::<B>(weights, device);
    let mut optim = SgdConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let mut run = Run {
        losses: Vec::with_capacity(steps),
        times: Vec::with_capacity(steps),
    };
    for step in 0..steps {
        let t0 = Instant::now();
        let (x, y) = batch::<B>(step, device);
        let loss = loss_fn.forward(model.forward(x), y);
        run.losses.push(loss.clone().into_scalar().elem::<f32>());
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(LR, model, grads);
        run.times.push(t0.elapsed());
    }
    run
}
