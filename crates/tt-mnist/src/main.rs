//! `tt-mnist`: train a small neural network on MNIST, in pure Rust, on a
//! Tenstorrent Blackhole card -- one self-contained binary.
//!
//! Everything it needs is inside it: the dataset (compressed, `build.rs`), the
//! firmware for the card's RISC-V cores (`tt-firmware-images`), and the whole
//! stack between Burn and the hardware (`burn-tt` -> `tt-kernels` ->
//! `tt-device` -> `tt-kmd`). What the machine needs is the card and
//! Tenstorrent's kernel driver (`tt-kmd`, ioctl API 2); no Python, no C++, no
//! TT-Metalium.
//!
//! The model is Burn's own `nn::Linear` twice with a ReLU between, trained by
//! Burn's autodiff and SGD. On the card: every matmul (forward and backward),
//! the bias adds, the ReLU and its gradient, the bias-gradient sums and the
//! SGD updates, with the dataset, weights and activations resident in the
//! card's GDDR6. Cross-entropy and its backward pass use native kernels.
//!
//! ```text
//! tt-mnist [--card N | --cards 0,1] [--tiles T] [--epochs E] [--steps S] [--host]
//! ```

use std::time::{Duration, Instant};

use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::Linear;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::activation;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{DType, ElementConversion, FloatDType, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{attach_topology, Fidelity, SrcRoute, Topology, TtBackend, TtDevice};

const PIXELS: usize = 28 * 28;
const CLASSES: usize = 10;
const BATCH: usize = 64;
const LR: f64 = 0.1;

// --- The data, embedded -------------------------------------------------------

struct Split {
    images: Vec<f32>,
    labels: Vec<u8>,
    n: usize,
}

fn inflate(packed: &[u8]) -> Vec<u8> {
    miniz_oxide::inflate::decompress_to_vec(packed).expect("the embedded data inflates")
}

fn idx(images: &[u8], labels: &[u8]) -> Split {
    let word = |b: &[u8], i: usize| u32::from_be_bytes(b[4 * i..4 * i + 4].try_into().unwrap());
    assert_eq!(word(images, 0), 0x803, "not an IDX image file");
    assert_eq!(word(labels, 0), 0x801, "not an IDX label file");
    let n = word(images, 1) as usize;
    assert_eq!(n, word(labels, 1) as usize);
    Split {
        images: images[16..].iter().map(|&p| f32::from(p) / 255.0).collect(),
        labels: labels[8..].to_vec(),
        n,
    }
}

fn mnist() -> (Split, Split) {
    macro_rules! embedded {
        ($name:literal) => {
            inflate(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/",
                $name,
                ".deflate"
            )))
        };
    }
    (
        idx(
            &embedded!("train-images-idx3-ubyte"),
            &embedded!("train-labels-idx1-ubyte"),
        ),
        idx(
            &embedded!("t10k-images-idx3-ubyte"),
            &embedded!("t10k-labels-idx1-ubyte"),
        ),
    )
}

// --- The model ----------------------------------------------------------------

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
    #[module(skip)]
    act: Act,
}

/// The hidden layer's activation (`--activation`): Burn's own functions, so
/// the example exercises whatever burn-tt runs of them on the card -- and
/// shows native execution and transfer costs in its timings.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Act {
    Relu,
    LeakyRelu,
    Gelu,
    Tanh,
    Sigmoid,
    Silu,
    HardSigmoid,
}

impl Act {
    const ALL: [(&'static str, Act); 7] = [
        ("relu", Act::Relu),
        ("leaky-relu", Act::LeakyRelu),
        ("gelu", Act::Gelu),
        ("tanh", Act::Tanh),
        ("sigmoid", Act::Sigmoid),
        ("silu", Act::Silu),
        ("hard-sigmoid", Act::HardSigmoid),
    ];

    fn name(self) -> &'static str {
        Act::ALL.iter().find(|(_, a)| *a == self).unwrap().0
    }

    fn apply<B: Backend>(self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Act::Relu => activation::relu(x),
            Act::LeakyRelu => activation::leaky_relu(x, 0.01),
            Act::Gelu => activation::gelu(x),
            Act::Tanh => activation::tanh(x),
            Act::Sigmoid => activation::sigmoid(x),
            Act::Silu => activation::silu(x),
            Act::HardSigmoid => activation::hard_sigmoid(x, 0.2, 0.5),
        }
    }
}

/// The initial weights, drawn once so the card and the host start the same,
/// and the activation between the layers.
struct Init {
    layers: Vec<(TensorData, TensorData)>,
    act: Act,
    dtype: DType,
    cnn: Option<tt_mnist::cnn::Init>,
}

fn init(hidden: &[usize]) -> Init {
    let mut s: u64 = 0x3a15;
    let mut unit = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 33) as f64 / (1u64 << 30) as f64 - 1.0) as f32
    };
    let mut layer = |d_in: usize, d_out: usize| {
        let k = 1.0 / (d_in as f32).sqrt();
        let w: Vec<f32> = (0..d_in * d_out).map(|_| unit() * k).collect();
        let b: Vec<f32> = (0..d_out).map(|_| unit() * k).collect();
        (
            TensorData::new(w, [d_in, d_out]),
            TensorData::new(b, [d_out]),
        )
    };
    let mut layers = Vec::new();
    let mut prev = PIXELS;
    for &h in hidden {
        layers.push(layer(prev, h));
        prev = h;
    }
    layers.push(layer(prev, CLASSES));
    Init {
        layers,
        act: Act::Relu,
        dtype: DType::F32,
        cnn: None,
    }
}

impl<B: Backend> Mlp<B> {
    fn new(init: &Init, device: &B::Device) -> Self {
        let linear = |(w, b): &(TensorData, TensorData)| Linear {
            weight: Param::from_tensor(Tensor::from_data(w.clone(), (device, init.dtype))),
            bias: Some(Param::from_tensor(Tensor::from_data(
                b.clone(),
                (device, init.dtype),
            ))),
        };
        Mlp {
            layers: init.layers.iter().map(linear).collect(),
            act: init.act,
        }
    }

    fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i + 1 < n {
                x = self.act.apply(x);
            }
        }
        x
    }
}

/// Both classifiers use the same optimizer, resident dataset and trace paths.
#[derive(Module, Debug)]
struct Classifier<B: Backend> {
    mlp: Option<Mlp<B>>,
    cnn: Option<tt_mnist::cnn::Cnn<B>>,
}

impl<B: Backend> Classifier<B> {
    fn new(init: &Init, device: &B::Device) -> Self {
        match &init.cnn {
            Some(cnn) => Self {
                mlp: None,
                cnn: Some(tt_mnist::cnn::Cnn::new(cnn, init.dtype, device)),
            },
            None => Self {
                mlp: Some(Mlp::new(init, device)),
                cnn: None,
            },
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match &self.cnn {
            Some(cnn) => cnn.forward(x),
            None => self.mlp.as_ref().expect("MLP classifier").forward(x),
        }
    }
}

// --- Training -----------------------------------------------------------------

struct Run {
    /// burn-tt's per-call timers once the dataset is resident.
    calls: Vec<(&'static str, u64, Duration)>,
    losses: Vec<(usize, f32)>,
    accuracy: f64,
    preload: Duration,
    train: Duration,
    steps: usize,
}

fn labels<B: Backend>(
    split: &Split,
    from: usize,
    n: usize,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let y: Vec<i32> = split.labels[from..from + n]
        .iter()
        .map(|&l| i32::from(l))
        .collect();
    Tensor::from_data(TensorData::new(y, [n]), device)
}

fn train<B: AutodiffBackend>(
    train: &Split,
    test: &Split,
    init: &Init,
    epochs: usize,
    max_steps: usize,
    device: &B::Device,
    report: impl Fn(usize, f32),
) -> Run {
    let t0 = Instant::now();
    let mut model = Classifier::<B>::new(init, device);
    let mut optim = SgdConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let samples = train.n - train.n % BATCH;
    // Uploaded once; each batch is a view of it (on the card: no copy).
    let images: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(train.images[..samples * PIXELS].to_vec(), [samples, PIXELS]),
        (device, init.dtype),
    )
    .to_device(device);
    let preload = t0.elapsed();
    let calls = burn_tt::device_time();

    let t0 = Instant::now();
    let mut losses = Vec::new();
    let mut step = 0;
    'epochs: for _ in 0..epochs {
        for from in (0..samples).step_by(BATCH) {
            if step == max_steps {
                break 'epochs;
            }
            let x = images.clone().slice([from..from + BATCH, 0..PIXELS]);
            let y = labels::<B>(train, from, BATCH, device);
            // Keep cross-entropy in F32; cast gradients return to BF16 parameters.
            let loss = loss_fn.forward(model.forward(x).cast(FloatDType::F32), y);
            if step % 100 == 0 {
                let l = loss.clone().into_scalar().elem::<f32>();
                report(step, l);
                losses.push((step, l));
            }
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(LR, model, grads);
            step += 1;
        }
    }
    let train_time = t0.elapsed();

    let model = model.valid();
    let mut right = 0usize;
    let eval_batch = if init.cnn.is_some() { BATCH } else { 1000 };
    for from in (0..test.n).step_by(eval_batch) {
        let m = eval_batch.min(test.n - from);
        let x = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(
                test.images[from * PIXELS..(from + m) * PIXELS].to_vec(),
                [m, PIXELS],
            ),
            (device, init.dtype),
        );
        // Predictions are an explicit application result readback. Accuracy
        // bookkeeping compares them with the dataset's labels on the host.
        let pred = model
            .forward(x)
            .argmax(1)
            .reshape([m])
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("predictions");
        right += pred
            .iter()
            .zip(&test.labels[from..from + m])
            .filter(|(p, l)| **p == i32::from(**l))
            .count();
    }
    Run {
        calls,
        losses,
        accuracy: right as f64 / test.n as f64,
        preload,
        train: train_time,
        steps: step,
    }
}

/// Train MNIST using hardware-traced execution (`TracedTrainingStep`).
///
/// The forward pass, cross-entropy loss, backward pass, optimizer parameter updates,
/// and in-place GDDR weight buffer updates are captured into a single hardware command stream.
/// Subsequent steps replay the stream in hardware without host op construction.
fn train_traced(
    train_split: &Split,
    test_split: &Split,
    init: &Init,
    epochs: usize,
    max_steps: usize,
    device: &TtDevice,
    report: impl Fn(usize, f32),
) -> Run {
    use burn_tt::{InputPayload, TracedTrainingStep};
    use tt_mnist::trace::{
        collect_parameter_updates, ensure_resident, primitive_float, primitive_int,
    };

    let t0 = Instant::now();
    let model = Classifier::<Autodiff<TtBackend>>::new(init, device);
    let mut optim = SgdConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let samples = train_split.n - train_split.n % BATCH;
    let preload = t0.elapsed();
    let calls = burn_tt::device_time();

    let mut losses = Vec::new();
    let mut step = 0;

    // Ensure parameters are resident in GDDR before capture
    ensure_resident(&model);

    // Step 0: Capture step
    let x0: Tensor<Autodiff<TtBackend>, 2> = Tensor::from_data(
        TensorData::new(
            train_split.images[..BATCH * PIXELS].to_vec(),
            [BATCH, PIXELS],
        ),
        device,
    );
    let y0 = labels::<Autodiff<TtBackend>>(train_split, 0, BATCH, device);
    let xp = primitive_float(x0.clone());
    let yp = primitive_int(y0.clone());

    let t0_train = Instant::now();
    let (traced_step, first_loss) = TracedTrainingStep::capture(&[&xp, &yp], || {
        let loss = loss_fn.forward(model.forward(x0.cast(init.dtype)).cast(FloatDType::F32), y0);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let new_model = optim.step(LR, model.clone(), grads);
        let updates = collect_parameter_updates(&model, &new_model);
        (primitive_float(loss), updates)
    })
    .unwrap_or_else(|e| panic!("capturing training step: {e}"));

    report(0, first_loss);
    losses.push((0, first_loss));
    step += 1;

    // Replay remaining steps
    'epochs: for _ in 0..epochs {
        for from in (0..samples).step_by(BATCH) {
            if step >= max_steps {
                break 'epochs;
            }
            if step == 1 && from == 0 {
                continue;
            }
            let x_data = train_split.images[from * PIXELS..(from + BATCH) * PIXELS].to_vec();
            let y_data: Vec<u32> = train_split.labels[from..from + BATCH]
                .iter()
                .map(|&l| i32::from(l) as u32)
                .collect();

            let timing = traced_step
                .step(vec![InputPayload::F32(x_data), InputPayload::Bits(y_data)])
                .unwrap_or_else(|e| panic!("replaying training step {step}: {e}"));

            if step % 100 == 0 || step == max_steps - 1 {
                report(step, timing.loss);
                losses.push((step, timing.loss));
            }
            step += 1;
        }
    }
    let train_time = t0_train.elapsed();

    // Accuracy evaluation on test set using the in-place trained model
    let valid_model = model.valid();
    let mut right = 0usize;
    let eval_batch = if init.cnn.is_some() { BATCH } else { 1000 };
    for from in (0..test_split.n).step_by(eval_batch) {
        let m = eval_batch.min(test_split.n - from);
        let x = Tensor::<TtBackend, 2>::from_data(
            TensorData::new(
                test_split.images[from * PIXELS..(from + m) * PIXELS].to_vec(),
                [m, PIXELS],
            ),
            (device, init.dtype),
        );
        let pred = valid_model
            .forward(x)
            .argmax(1)
            .reshape([m])
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("predictions");
        right += pred
            .iter()
            .zip(&test_split.labels[from..from + m])
            .filter(|(p, l)| **p == i32::from(**l))
            .count();
    }

    Run {
        calls,
        losses,
        accuracy: right as f64 / test_split.n as f64,
        preload,
        train: train_time,
        steps: step,
    }
}

/// What an inference run measured.
struct Infer {
    /// burn-tt's per-call timers once the test set is resident.
    calls: Vec<(&'static str, u64, Duration)>,
    /// The test images to the device, once.
    preload: Duration,
    /// The first batch alone: program caches and the session cold.
    first: Duration,
    /// Every batch after the first.
    rest: Duration,
    batches: usize,
    accuracy: f64,
    /// With `--trace`, every batch after the first: the input's write from
    /// the host, the replay, and the logits' read, summed -- PCIe's time and
    /// the card's, reported apart.
    traced: Option<TraceParts>,
}

#[derive(Default)]
struct TraceParts {
    write: Duration,
    replay: Duration,
    read: Duration,
}

/// The forward pass alone, no autodiff: the test set uploaded once, then
/// `passes` rounds over it in batches of `batch`, each batch's predictions
/// downloaded (`argmax` on the host, as a caller would read them). Untrained
/// weights unless the caller trains first; accuracy is printed only as a
/// check that the right pictures went through.
fn infer<B: Backend>(
    test: &Split,
    init: &Init,
    batch: usize,
    passes: usize,
    device: &B::Device,
) -> Infer {
    let model = Classifier::<B>::new(init, device);
    let t0 = Instant::now();
    let n = test.n - test.n % batch;
    let images: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(test.images[..n * PIXELS].to_vec(), [n, PIXELS]),
        (device, init.dtype),
    )
    .to_device(device);
    let preload = t0.elapsed();
    let calls = burn_tt::device_time();
    let (mut first, mut rest) = (Duration::ZERO, Duration::ZERO);
    let (mut batches, mut right) = (0, 0usize);
    for _ in 0..passes {
        for from in (0..n).step_by(batch) {
            let t = Instant::now();
            let x = images.clone().slice([from..from + batch, 0..PIXELS]);
            let pred: Vec<i64> = model
                .forward(x)
                .argmax(1)
                .reshape([batch])
                .into_data()
                .convert::<i64>()
                .to_vec()
                .expect("integer predictions");
            if batches == 0 {
                first = t.elapsed();
            } else {
                rest += t.elapsed();
            }
            batches += 1;
            right += pred
                .iter()
                .zip(&test.labels[from..from + batch])
                .filter(|(p, &l)| **p == i64::from(l))
                .count();
        }
    }
    Infer {
        calls,
        preload,
        first,
        rest,
        batches,
        accuracy: right as f64 / (batches * batch).max(1) as f64,
        traced: None,
    }
}

/// [`infer`], each batch run by a trace (`burn_tt::Trace`, X4d): the forward
/// pass captured once on a batch-sized input, then each batch written into
/// that input and replayed -- the logits come back, and `argmax` runs on the
/// host. A batch is written from the host each time, as a server's inputs
/// would be, where `infer` slices a test set already on the card.
fn infer_traced(
    test: &Split,
    init: &Init,
    batch: usize,
    passes: usize,
    device: &TtDevice,
    fusion: bool,
) -> Infer {
    use burn::tensor::TensorPrimitive;
    let t0 = Instant::now();
    let n = test.n - test.n % batch;
    let (trace, _) = if fusion {
        let prim = |t: Tensor<burn_tt::Tt, 2>| match t.into_primitive() {
            TensorPrimitive::Float(p) => burn_tt::resolve_float_tensor(&p),
            _ => unreachable!("a float tensor"),
        };
        let model = Classifier::<burn_tt::Tt>::new(init, device);
        let x: Tensor<burn_tt::Tt, 2> = Tensor::from_data(
            TensorData::new(test.images[..batch * PIXELS].to_vec(), [batch, PIXELS]),
            device,
        );
        let xp = prim(x.clone());
        let _ = prim(model.forward(x.clone().cast(init.dtype)));
        burn_tt::Trace::capture(&xp, || {
            prim(
                model
                    .forward(x.clone().cast(init.dtype))
                    .cast(FloatDType::F32),
            )
        })
        .unwrap_or_else(|e| panic!("capturing the forward pass with fusion: {e}"))
    } else {
        let prim = |t: Tensor<TtBackend, 2>| match t.into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!("a float tensor"),
        };
        let model = Classifier::<TtBackend>::new(init, device);
        let x: Tensor<TtBackend, 2> = Tensor::from_data(
            TensorData::new(test.images[..batch * PIXELS].to_vec(), [batch, PIXELS]),
            device,
        );
        let xp = prim(x.clone());
        // Pre-run forward pass to ensure weights/biases are resident in GDDR before capture
        let _ = model.forward(x.clone().cast(init.dtype));
        burn_tt::Trace::capture(&xp, || {
            prim(
                model
                    .forward(x.clone().cast(init.dtype))
                    .cast(FloatDType::F32),
            )
        })
        .unwrap_or_else(|e| panic!("capturing the forward pass: {e}"))
    };
    let classes = trace.output_dims()[1];
    let preload = t0.elapsed();
    let calls = burn_tt::device_time();
    let (mut first, mut rest) = (Duration::ZERO, Duration::ZERO);
    let (mut batches, mut right) = (0, 0usize);
    let mut parts = TraceParts::default();
    for _ in 0..passes {
        for from in (0..n).step_by(batch) {
            let t = Instant::now();
            let input = test.images[from * PIXELS..(from + batch) * PIXELS].to_vec();
            let run = trace
                .run_timed(input)
                .unwrap_or_else(|e| panic!("replaying the forward pass: {e}"));
            if batches > 0 {
                parts.write += run.write;
                parts.replay += run.replay;
                parts.read += run.read;
            }
            let logits = run.output;
            let pred: Vec<usize> = logits
                .chunks_exact(classes)
                .map(|row| {
                    (0..classes)
                        .max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a)))
                        .expect("classes")
                })
                .collect();
            if batches == 0 {
                first = t.elapsed();
            } else {
                rest += t.elapsed();
            }
            batches += 1;
            right += pred
                .iter()
                .zip(&test.labels[from..from + batch])
                .filter(|(p, &l)| **p == usize::from(l))
                .count();
        }
    }
    Infer {
        calls,
        preload,
        first,
        rest,
        batches,
        accuracy: right as f64 / (batches * batch).max(1) as f64,
        traced: Some(parts),
    }
}

fn print_infer(r: &Infer, batch: usize) {
    let steady = r.rest.as_secs_f64() / (r.batches.saturating_sub(1)).max(1) as f64;
    println!(
        "  test images uploaded once                  {:.2?}",
        r.preload
    );
    println!(
        "  first batch (cold)                         {:.2?}",
        r.first
    );
    println!(
        "  {} more batches of {batch}                   {:.2?}  ({:.3} ms/batch, {:.0} images/s)",
        r.batches.saturating_sub(1),
        r.rest,
        steady * 1e3,
        batch as f64 / steady
    );
    if let Some(p) = &r.traced {
        let per = |d: Duration| d.as_secs_f64() * 1e3 / (r.batches.saturating_sub(1)).max(1) as f64;
        let replay = per(p.replay);
        println!(
            "    input written from the host            {:.3} ms/batch  (PCIe)",
            per(p.write)
        );
        println!(
            "    replay, to its end                     {replay:.3} ms/batch  ({:.0} images/s on the card alone)",
            batch as f64 / (replay / 1e3)
        );
        println!(
            "    logits read back                       {:.3} ms/batch  (PCIe)",
            per(p.read)
        );
        println!(
            "    the rest (argmax, server round trip)   {:.3} ms/batch",
            steady * 1e3 - per(p.write) - replay - per(p.read)
        );
    }
    println!(
        "  accuracy (untrained weights: a check, not a result)  {:.2}%",
        r.accuracy * 100.0
    );
}

/// Where the host's time went: each kind of device call burn-tt made, timed
/// on the caller's side (the server's queue, the session's work and the wait
/// for the card included), per `per` units of work, and what was left outside
/// every call -- Burn itself and the host's own ops.
fn print_calls(before: &[(&'static str, u64, Duration)], wall: Duration, per: usize, unit: &str) {
    let now = burn_tt::device_time();
    let mut inside = Duration::ZERO;
    println!("  where the time went, per {unit} (caller's side):");
    for (k, n, d) in &now {
        let (n0, d0) = before
            .iter()
            .find(|(b, ..)| b == k)
            .map_or((0, Duration::ZERO), |(_, n, d)| (*n, *d));
        let (n, d) = (n - n0, *d - d0);
        if n == 0 {
            continue;
        }
        inside += d;
        println!(
            "    {k:<14} {:>6.1} calls {:>8.1} us  ({:.1} us a call)",
            n as f64 / per as f64,
            d.as_secs_f64() * 1e6 / per as f64,
            d.as_secs_f64() * 1e6 / n as f64
        );
    }
    println!(
        "    {:<14} {:>15.1} us",
        "outside calls",
        wall.saturating_sub(inside).as_secs_f64() * 1e6 / per as f64
    );
}

// --- The command line ---------------------------------------------------------

struct Args {
    topology: Topology,
    epochs: usize,
    steps: usize,
    host: bool,
    /// `--infer`: the forward pass alone, no training.
    infer: bool,
    /// `--trace`: with `--infer`, each batch a replay of a captured trace.
    trace: bool,
    batch: usize,
    passes: usize,
    activation: Act,
    /// `--model transformer`: the general-model benchmark instead of MNIST.
    transformer: bool,
    cnn: bool,
    bf16: bool,
    /// `--train-trace`: train with hardware trace (capture step 0 and replay subsequent steps).
    train_trace: bool,
    /// `--fusion`: enable Burn kernel fusion on device.
    fusion: bool,
    /// `--hidden H1,H2,...`: hidden layer sizes.
    hidden: Vec<usize>,
}

const USAGE: &str = "\
usage: tt-mnist [--card N | --cards 0,1] [--tiles T] [--epochs E] [--steps S] [--host] [--activation A]
       tt-mnist --infer [--trace] [--batch B] [--passes P] [--card N | --cards 0,1] [--tiles T] [--host]
       tt-mnist --model transformer [--steps S] [--card N] [--tiles T] [--train-trace]

  --card N      train on /dev/tenstorrent/N (default 0)
  --cards 0,1   several cabled cards, matmuls sharded over Ethernet
  --tiles T     compute on T Tensix tiles of the card, or `all` (default 1)
  --epochs E    passes over the 60 000 training images (default 1)
  --steps S     stop after S steps
  --host        also train on the host CPU (burn-flex), for comparison
  --bf16        store MNIST weights and activations in BF16 (single card);
                matrix accumulation, loss and trace input/output remain F32
  --hidden H    hidden layer sizes, comma-separated (default 512,256: 784-512-256-10, 535k params)
  --train-trace train with hardware trace (capture step 0 and replay subsequent steps)
  --fusion      enable Burn kernel fusion on device
  --infer       benchmark inference alone: no training, the forward pass over
                the test set (untrained weights), so it profiles on its own
                (`TT_PROFILE`)
  --trace       with --infer: capture the forward pass once as a trace and
                replay it for every batch, each batch written from the host
  --batch B     inference batch size (default 64)
  --passes P    rounds over the test set (default 3)
  --activation A  the hidden layer's activation: relu (default), leaky-relu,
                gelu, tanh, sigmoid, silu, hard-sigmoid
  --model M     mnist (default), cnn (Conv2D/ReLU/AvgPool/Linear), or transformer: a small Burn transformer
                (embedding, pre-norm encoder layer, Linear head) trained S
                steps (default 50) on the card and on burn-flex, with the
                time per step of each and burn-tt's per-op report";

fn args() -> Result<Args, String> {
    let mut a = Args {
        topology: Topology::single(0),
        epochs: 1,
        steps: usize::MAX,
        host: false,
        infer: false,
        trace: false,
        batch: BATCH,
        passes: 3,
        activation: Act::Relu,
        transformer: false,
        cnn: false,
        bf16: false,
        train_trace: false,
        fusion: false,
        hidden: vec![512, 256],
    };
    let mut tiles = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().ok_or(format!("{arg} needs a value\n\n{USAGE}"));
        let number = |v: String| {
            v.parse::<usize>()
                .map_err(|_| format!("{arg} {v}: not a number"))
        };
        match arg.as_str() {
            "--card" => {
                let v = value()?;
                a.topology =
                    Topology::single(v.parse().map_err(|_| format!("--card {v}: not a number"))?);
            }
            "--cards" => a.topology = Topology::parse(&value()?).map_err(|e| e.to_string())?,
            "--tiles" => tiles = Some(burn_tt::parse_tiles(&value()?).map_err(|e| e.to_string())?),
            "--epochs" => a.epochs = number(value()?)?,
            "--steps" => a.steps = number(value()?)?,
            "--host" => a.host = true,
            "--bf16" => a.bf16 = true,
            "--hidden" => {
                let v = value()?;
                a.hidden = v
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|_| format!("--hidden {s}: not a number\n\n{USAGE}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if a.hidden.is_empty() {
                    return Err(format!("--hidden needs at least one layer size\n\n{USAGE}"));
                }
            }
            "--train-trace" => a.train_trace = true,
            "--fusion" => a.fusion = true,
            "--infer" => a.infer = true,
            "--trace" => a.trace = true,
            "--batch" => a.batch = number(value()?)?,
            "--passes" => a.passes = number(value()?)?,
            "--activation" => {
                let v = value()?;
                a.activation = Act::ALL
                    .iter()
                    .find(|(n, _)| *n == v)
                    .map(|(_, act)| *act)
                    .ok_or(format!(
                        "--activation {v}: not one of the choices\n\n{USAGE}"
                    ))?;
            }
            "--model" => match value()?.as_str() {
                "mnist" => {
                    a.transformer = false;
                    a.cnn = false;
                }
                "cnn" => {
                    a.transformer = false;
                    a.cnn = true;
                }
                "transformer" => {
                    a.transformer = true;
                    a.cnn = false;
                }
                v => {
                    return Err(format!(
                        "--model {v}: not mnist, cnn or transformer\n\n{USAGE}"
                    ))
                }
            },
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n\n{USAGE}")),
        }
    }
    if a.batch == 0 || a.batch > 10_000 || a.passes == 0 {
        return Err(format!(
            "--batch must be 1..=10000 and --passes at least 1\n\n{USAGE}"
        ));
    }
    // After the loop, so `--tiles` applies whichever order it came in.
    if let Some(t) = tiles {
        a.topology = a.topology.on_tiles(t).map_err(|e| e.to_string())?;
    }
    if a.bf16 && (a.transformer || matches!(a.topology, Topology::Cards { .. })) {
        return Err("--bf16 currently requires MNIST execution on one card".into());
    }
    Ok(a)
}

fn main() {
    let a = match args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if a.transformer {
        return transformer_benchmark(&a);
    }
    let cards = match &a.topology {
        Topology::Single { card, tile } => {
            let tiles = match tile {
                burn_tt::TileChoice::Count(n) => format!("{n} Tensix tiles"),
                burn_tt::TileChoice::All => "every Tensix tile".into(),
                _ => "one Tensix tile".into(),
            };
            format!("/dev/tenstorrent/{card}, {tiles}, tensors resident in GDDR6")
        }
        Topology::Cards { cards, .. } => format!(
            "/dev/tenstorrent/{{{}}}, matmuls sharded across the cards over Ethernet",
            cards
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ),
    };
    let layers_str = std::iter::once(PIXELS)
        .chain(a.hidden.iter().copied())
        .chain(std::iter::once(CLASSES))
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let param_count: usize = {
        let mut total = 0;
        let mut prev = PIXELS;
        for &h in &a.hidden {
            total += prev * h + h;
            prev = h;
        }
        total += prev * CLASSES + CLASSES;
        total
    };
    if a.cnn {
        println!("tt-mnist: CNN, Conv2D 1→8 (5×5, stride 4), ReLU, AvgPool 2×2, Linear 72→10 ({} parameters), SGD lr {LR}, batch {BATCH}", tt_mnist::cnn::PARAMETERS);
        println!("  device   {cards}");
    } else {
        println!("tt-mnist: a {layers_str} network ({param_count} parameters) learning MNIST, in Rust, on Tenstorrent Blackhole");
        println!("  device   {cards}");
        println!(
            "  model    Burn nn::Linear x{} + {}, cross-entropy, SGD lr {LR}, batch {BATCH}",
            a.hidden.len() + 1,
            a.activation.name()
        );
    }

    let t0 = Instant::now();
    let (train_split, test_split) = mnist();
    println!(
        "  data     {} training / {} test images, embedded ({:.2?} to unpack)\n",
        train_split.n,
        test_split.n,
        t0.elapsed()
    );
    let init = Init {
        act: a.activation,
        cnn: a.cnn.then(tt_mnist::cnn::Init::default),
        dtype: if a.bf16 { DType::BF16 } else { DType::F32 },
        ..init(&a.hidden)
    };
    println!(
        "  storage  {:?}; matrix accumulation and loss F32",
        init.dtype
    );

    let device = TtDevice::new(0);
    let guard = match attach_topology(
        device,
        a.topology.clone(),
        SrcRoute::Tf32FromFp32,
        Fidelity::HiFi4,
    ) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("could not open the card: {e}");
            eprintln!(
                "(is the tenstorrent kernel driver loaded, and /dev/tenstorrent/* readable?)"
            );
            std::process::exit(1);
        }
    };
    if a.infer {
        println!(
            "inference only, batch {}, {} passes over the test set:",
            a.batch, a.passes
        );
        let before = burn_tt::tensor_traffic();
        let r = if a.trace {
            infer_traced(&test_split, &init, a.batch, a.passes, &device, a.fusion)
        } else if a.fusion {
            infer::<burn_tt::Tt>(&test_split, &init, a.batch, a.passes, &device)
        } else {
            infer::<TtBackend>(&test_split, &init, a.batch, a.passes, &device)
        };
        let moved = burn_tt::tensor_traffic() - before;
        drop(guard);
        println!("\non the card:");
        print_infer(&r, a.batch);
        print_calls(&r.calls, r.first + r.rest, r.batches, "batch");
        if a.fusion {
            let fused_add_relu = burn_tt::fused_add_relu_count();
            let fused_relu = burn_tt::fused_relu_count();
            println!(
                "  kernel fusion:                            {} add_relu, {} relu executed on device",
                fused_add_relu, fused_relu
            );
        }
        println!(
            "  tensor data over PCIe, whole run          {:.1} MB up (incl. the test set), {:.2} MB down",
            moved.uploaded as f64 / 1e6,
            moved.downloaded as f64 / 1e6
        );
        if a.host {
            println!("\non the host CPU (burn-flex), same model:");
            print_infer(
                &infer::<Flex>(&test_split, &init, a.batch, a.passes, &FlexDevice),
                a.batch,
            );
        }
        return;
    }
    let before = burn_tt::tensor_traffic();
    let card = if a.train_trace {
        println!("training with hardware trace (step 0 capture, subsequent steps replayed):");
        train_traced(
            &train_split,
            &test_split,
            &init,
            a.epochs,
            a.steps,
            &device,
            |step, loss| println!("  step {step:>5}   loss {loss:.4}"),
        )
    } else if a.fusion {
        train::<Autodiff<burn_tt::Tt>>(
            &train_split,
            &test_split,
            &init,
            a.epochs,
            a.steps,
            &device,
            |step, loss| println!("  step {step:>5}   loss {loss:.4}"),
        )
    } else {
        train::<Autodiff<TtBackend>>(
            &train_split,
            &test_split,
            &init,
            a.epochs,
            a.steps,
            &device,
            |step, loss| println!("  step {step:>5}   loss {loss:.4}"),
        )
    };
    let moved = burn_tt::tensor_traffic() - before;
    drop(guard);

    let per = |d: Duration| d.as_secs_f64() * 1e3 / card.steps.max(1) as f64;
    println!();
    println!("on the card:");
    if a.fusion {
        let fused_add_relu = burn_tt::fused_add_relu_count();
        let fused_relu = burn_tt::fused_relu_count();
        println!(
            "  kernel fusion:                            {} add_relu, {} relu executed on device",
            fused_add_relu, fused_relu
        );
    }
    println!(
        "  model and {} images uploaded once       {:.2?}",
        train_split.n - train_split.n % BATCH,
        card.preload
    );
    println!(
        "  {} training steps                        {:.2?}  ({:.1} ms/step)",
        card.steps,
        card.train,
        per(card.train)
    );
    print_calls(&card.calls, card.train, card.steps.max(1), "step");
    println!(
        "  loss                                      {:.4} -> {:.4}",
        card.losses.first().map_or(f32::NAN, |l| l.1),
        card.losses.last().map_or(f32::NAN, |l| l.1)
    );
    println!(
        "  test accuracy                             {:.2}%",
        card.accuracy * 100.0
    );
    println!(
        "  tensor data over PCIe, whole run          {:.1} MB up (incl. the dataset), {:.2} MB down",
        moved.uploaded as f64 / 1e6,
        moved.downloaded as f64 / 1e6
    );

    if a.host {
        println!("\non the host CPU (burn-flex), same model, same initial weights:");
        let host = train::<Autodiff<Flex>>(
            &train_split,
            &test_split,
            &init,
            a.epochs,
            a.steps,
            &FlexDevice,
            |_, _| {},
        );
        println!(
            "  {} training steps                        {:.2?}  ({:.1} ms/step)",
            host.steps,
            host.train,
            host.train.as_secs_f64() * 1e3 / host.steps.max(1) as f64
        );
        println!(
            "  test accuracy                             {:.2}%",
            host.accuracy * 100.0
        );
        let same = card
            .losses
            .iter()
            .zip(&host.losses)
            .filter(|(c, h)| (c.1 - h.1).abs() <= 1e-3 * h.1.abs().max(1.0))
            .count();
        println!(
            "  loss curves agree to 0.1% at {same} of {} checkpoints",
            card.losses.len().min(host.losses.len())
        );
    }
}

/// `--model transformer`: the same steps on the card and on burn-flex, from
/// the same weights, and what burn-tt ran where.
fn transformer_benchmark(a: &Args) {
    use tt_mnist::transformer as tf;
    let steps = if a.steps == usize::MAX {
        50
    } else {
        a.steps.max(2)
    };
    println!(
        "tt-mnist --model transformer: vocab {}, d_model {}, d_ff {}, {} heads, {} layer, \
         batch {} x seq {}, SGD lr {}",
        tf::VOCAB,
        tf::D_MODEL,
        tf::D_FF,
        tf::HEADS,
        tf::LAYERS,
        tf::BATCH,
        tf::SEQ,
        tf::LR
    );
    let weights = tf::init(59);
    let device = TtDevice::new(0);
    let guard = match attach_topology(
        device,
        a.topology.clone(),
        SrcRoute::Tf32FromFp32,
        Fidelity::HiFi4,
    ) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("could not open the card: {e}");
            std::process::exit(1);
        }
    };
    let before = burn_tt::tensor_traffic();
    let calls = burn_tt::device_time();
    let (card, report) = if a.train_trace {
        println!("training with hardware trace (step 0 capture, subsequent steps replayed):");
        let r = tf::train_traced(&weights, steps, &device)
            .unwrap_or_else(|e| panic!("traced transformer training failed: {e}"));
        (r, burn_tt::Report::default())
    } else {
        burn_tt::with_report(|| tf::train::<Autodiff<TtBackend>>(&weights, steps, &device))
    };
    let card_wall: Duration = card.times.iter().sum();
    let moved = burn_tt::tensor_traffic() - before;
    drop(guard);
    let host = tf::train::<Autodiff<Flex>>(&weights, steps, &FlexDevice);

    let ms = |d: Option<Duration>| d.map_or(f64::NAN, |d| d.as_secs_f64() * 1e3);
    let (c, h) = (ms(card.steady()), ms(host.steady()));
    println!("\n{steps} training steps, time per step after the first:");
    println!(
        "  burn-tt (card)     {c:8.2} ms/step   first step {:.2} ms",
        ms(card.times.first().copied())
    );
    println!(
        "  burn-flex (host)   {h:8.2} ms/step   first step {:.2} ms",
        ms(host.times.first().copied())
    );
    println!("  card / host        {:8.2}x", c / h);
    print_calls(&calls, card_wall, steps, "step");
    println!(
        "  loss               card {:.4} -> {:.4}, host {:.4} -> {:.4}",
        card.losses[0],
        card.losses[steps - 1],
        host.losses[0],
        host.losses[steps - 1]
    );
    println!(
        "  tensor data over PCIe   {:.2} MB up, {:.2} MB down ({:.1} KB a step)",
        moved.uploaded as f64 / 1e6,
        moved.downloaded as f64 / 1e6,
        (moved.uploaded + moved.downloaded) as f64 / 1e3 / steps as f64
    );
    println!("\nburn-tt per op, whole run (tt: native implementation; unsupported: required method without an implementation):");
    print!("{report}");
}
