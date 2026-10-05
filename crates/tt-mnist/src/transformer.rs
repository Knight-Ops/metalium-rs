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
/// Raw host data for step `step`: sequence `b` is an arithmetic progression mod [`VOCAB`].
pub fn batch_data(step: usize) -> (Vec<i32>, Vec<i32>) {
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
    (x, y)
}

/// Step `step`'s batch: sequence `b` is an arithmetic progression mod
/// [`VOCAB`]. The inputs, and the targets (the inputs shifted by one).
pub fn batch<B: Backend>(
    step: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 1, Int>) {
    let (x, y) = batch_data(step);
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

/// Train `steps` steps from `weights` using hardware-traced execution (`TracedTrainingStep`).
///
/// The forward pass, loss calculation, backward pass, optimizer parameter updates,
/// and in-place GDDR weight buffer updates are captured into a single hardware command stream.
/// Subsequent steps replay the stream in hardware without host op construction.
pub fn train_traced(
    weights: &[TensorData],
    steps: usize,
    device: &burn_tt::TtDevice,
) -> Result<Run, String> {
    use crate::trace::{
        collect_parameter_updates, ensure_resident, primitive_float, primitive_int,
    };
    use burn::backend::Autodiff;
    use burn_tt::{InputPayload, TracedTrainingStep, TtBackend};

    let model = load::<Autodiff<TtBackend>>(weights, device);
    let mut optim = SgdConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);

    let mut run = Run {
        losses: Vec::with_capacity(steps),
        times: Vec::with_capacity(steps),
    };

    if steps == 0 {
        return Ok(run);
    }

    // Ensure parameters are resident in GDDR before capture
    ensure_resident(&model);

    // Step 0: Capture the entire training step into a hardware trace
    let t0 = Instant::now();
    let (x0, y0) = batch::<Autodiff<TtBackend>>(0, device);
    let xp = primitive_int(x0.clone());
    let yp = primitive_int(y0.clone());

    let (traced_step, first_loss) = TracedTrainingStep::capture(&[&xp, &yp], || {
        let loss = loss_fn.forward(model.forward(x0), y0);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let new_model = optim.step(LR, model.clone(), grads);
        let updates = collect_parameter_updates(&model, &new_model);
        (primitive_float(loss), updates)
    })
    .map_err(|e| format!("capturing transformer training trace: {e}"))?;

    run.losses.push(first_loss);
    run.times.push(t0.elapsed());

    // Steps 1..steps: Stream new batch data directly to device and replay the hardware trace
    for step in 1..steps {
        let (x_raw, y_raw) = batch_data(step);
        let x_payload: Vec<u32> = x_raw.into_iter().map(|v| v as u32).collect();
        let y_payload: Vec<u32> = y_raw.into_iter().map(|v| v as u32).collect();

        let t = Instant::now();
        let timing = traced_step
            .step(vec![
                InputPayload::Bits(x_payload),
                InputPayload::Bits(y_payload),
            ])
            .map_err(|e| format!("replaying transformer training step {step}: {e}"))?;

        run.losses.push(timing.loss);
        run.times.push(t.elapsed());
    }

    Ok(run)
}
