//! Phase 7, step 12: the milestone -- a Burn MLP trains on MNIST through
//! `burn-autodiff`, with every matmul of the forward and backward passes on a
//! Tensix tile.
//!
//! The model is Burn's own: two `burn::nn::Linear` layers with a ReLU between
//! them, `CrossEntropyLoss`, and `Sgd`, in a `#[derive(Module)]` struct. Weights
//! come from a fixed generator rather than Burn's initialiser, so the device run
//! and the host run (`Autodiff<Flex>`) start from the same bits without relying
//! on the two backends drawing random numbers identically.
//!
//! What is claimed, and how strongly:
//!
//! * **The first forward pass is within a derived bound of Flex's**
//!   (`the_first_forward_pass_is_within_the_derived_bound`), carried through the
//!   network from `step11_burn`'s per-matmul bound.
//! * **Training works**: the loss falls by a stated factor, and the host run of
//!   the same setup meets the same factor, so the factor is a property of the
//!   setup rather than a number chosen to pass.
//! * **The run is deterministic, and silicon computes what ttsim computes**: the
//!   reduced run's loss curve is pinned bit for bit in
//!   `tests/golden/mnist_reduced.txt`, which the simulator run wrote and the
//!   silicon run must reproduce.
//! * **Not claimed**: that later steps stay within a bound of the host run.
//!   Two runs whose first steps differ by rounding diverge along chaotic
//!   trajectories, and a bound that honestly covered that would say nothing.
//!   The curves are printed side by side instead.

use burn::backend::Autodiff;
use burn::module::{Module, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, Relu};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_tests::burn_device::{with_device, Config};
use tt_tests::mnist::{self, Split, PIXELS};

const HIDDEN: usize = 128;
const CLASSES: usize = 10;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    relu: Relu,
}

/// A deterministic generator, so both backends start from the same weights.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f64 / (1u64 << 30) as f64 - 1.0) as f32
    }
}

/// `U(-k, k)` with `k = 1 / sqrt(d_in)`, Burn's own `Linear` default.
fn layer_data(rng: &mut Lcg, d_in: usize, d_out: usize) -> (TensorData, TensorData) {
    let k = 1.0 / (d_in as f32).sqrt();
    let w: Vec<f32> = (0..d_in * d_out).map(|_| rng.unit() * k).collect();
    // Symmetric about zero, or every ReLU is dead and nothing trains -- which
    // is what a generator confined to [-1, 0) once did here.
    assert!(w.iter().any(|x| *x > k / 2.0) && w.iter().any(|x| *x < -k / 2.0));
    let b: Vec<f32> = (0..d_out).map(|_| rng.unit() * k).collect();
    (
        TensorData::new(w, [d_in, d_out]),
        TensorData::new(b, [d_out]),
    )
}

fn linear<B: Backend>((w, b): &(TensorData, TensorData), device: &B::Device) -> Linear<B> {
    Linear {
        weight: Param::from_tensor(Tensor::from_data(w.clone(), device)),
        bias: Some(Param::from_tensor(Tensor::from_data(b.clone(), device))),
    }
}

struct Init {
    l1: (TensorData, TensorData),
    l2: (TensorData, TensorData),
}

fn init() -> Init {
    let mut rng = Lcg(0x3a15);
    Init {
        l1: layer_data(&mut rng, PIXELS, HIDDEN),
        l2: layer_data(&mut rng, HIDDEN, CLASSES),
    }
}

impl<B: Backend> Mlp<B> {
    fn new(init: &Init, device: &B::Device) -> Self {
        Mlp {
            l1: linear(&init.l1, device),
            l2: linear(&init.l2, device),
            relu: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.l2.forward(self.relu.forward(self.l1.forward(x)))
    }
}

/// Images `[from, from + n)` of `split` as a batch.
fn batch<B: Backend>(
    split: &Split,
    from: usize,
    n: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
    let x = TensorData::new(
        split.images[from * PIXELS..(from + n) * PIXELS].to_vec(),
        [n, PIXELS],
    );
    let y = TensorData::new(
        split.labels[from..from + n]
            .iter()
            .map(|&l| i32::from(l))
            .collect::<Vec<i32>>(),
        [n],
    );
    (Tensor::from_data(x, device), Tensor::from_data(y, device))
}

struct Setup {
    samples: usize,
    batch: usize,
    epochs: usize,
    lr: f64,
}

/// The reduced run the simulator gate uses, and silicon repeats bit for bit.
const REDUCED: Setup = Setup {
    samples: 512,
    batch: 64,
    epochs: 4,
    lr: 0.5,
};

/// Train from `init`, in order, returning the loss of every step and the model.
fn train<B: AutodiffBackend>(
    split: &Split,
    setup: &Setup,
    init: &Init,
    device: &B::Device,
) -> (Vec<f32>, Mlp<B>) {
    let mut model = Mlp::<B>::new(init, device);
    let mut optim = SgdConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let mut losses = Vec::new();
    for _ in 0..setup.epochs {
        for from in (0..setup.samples).step_by(setup.batch) {
            let (x, y) = batch::<B>(split, from, setup.batch, device);
            let loss = loss_fn.forward(model.forward(x), y);
            losses.push(loss.clone().into_scalar().elem::<f32>());
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(setup.lr, model, grads);
        }
    }
    (losses, model)
}

use burn::tensor::ElementConversion;

/// Fraction of `split`'s first `n` images classified correctly. Used by the
/// silicon-only full run.
#[cfg_attr(not(feature = "silicon"), allow(dead_code))]
fn accuracy<B: Backend>(model: &Mlp<B>, split: &Split, n: usize, device: &B::Device) -> f64 {
    let mut right = 0usize;
    for from in (0..n).step_by(1000) {
        let m = 1000.min(n - from);
        let (x, y) = batch::<B>(split, from, m, device);
        let pred = model.forward(x).argmax(1).reshape([m]);
        right += pred.equal(y).int().sum().into_scalar().elem::<i64>() as usize;
    }
    right as f64 / n as f64
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len() as f32
}

/// The loss must fall to at most this fraction of its first value, averaged
/// over the last epoch. The host run of the same setup is held to it too.
const DESCENT: f32 = 0.5;

fn golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/mnist_reduced.txt")
}

fn render(losses: &[f32]) -> String {
    let mut s = String::from(
        "# The reduced MNIST run's loss per step, as f32 bits. Written by the ttsim run\n\
         # of step12_mnist::the_mlp_trains_on_a_reduced_dataset with TT_BLESS=1; the\n\
         # silicon run must reproduce it exactly.\n",
    );
    for l in losses {
        s.push_str(&format!("{:08x} {l}\n", l.to_bits()));
    }
    s
}

fn golden() -> Option<Vec<u32>> {
    let text = std::fs::read_to_string(golden_path()).ok()?;
    Some(
        text.lines()
            .filter(|l| !l.starts_with('#'))
            .map(|l| u32::from_str_radix(l.split_whitespace().next().unwrap(), 16).unwrap())
            .collect(),
    )
}

#[test]
fn the_mlp_trains_on_a_reduced_dataset() {
    let split = mnist::load(true);
    let init = init();
    let (host, _) = train::<Autodiff<Flex>>(&split, &REDUCED, &init, &FlexDevice);
    let steps = REDUCED.samples / REDUCED.batch;
    assert!(
        mean(&host[host.len() - steps..]) <= DESCENT * host[0],
        "the setup itself must train: host {host:?}"
    );
    let bless = std::env::var_os("TT_BLESS").is_some();
    let want = golden();
    with_device(Config::default(), |d| {
        let (tt, _) = train::<Autodiff<TtBackend>>(&split, &REDUCED, &init, &d);
        for (i, (t, h)) in tt.iter().zip(&host).enumerate() {
            eprintln!("step {i:2}: device {t:.6}  host {h:.6}");
        }
        assert!(
            mean(&tt[tt.len() - steps..]) <= DESCENT * tt[0],
            "the loss must fall to {DESCENT} of its first value: {tt:?}"
        );
        assert_ne!(tt, host, "the matmuls must have run on the device");
        if bless && !tt_tests::backend::ON_SILICON {
            std::fs::create_dir_all(golden_path().parent().unwrap()).unwrap();
            std::fs::write(golden_path(), render(&tt)).unwrap();
            eprintln!("wrote {}", golden_path().display());
            return;
        }
        let want = want.unwrap_or_else(|| {
            panic!(
                "{} is missing: run this gate on the simulator with TT_BLESS=1",
                golden_path().display()
            )
        });
        let got: Vec<u32> = tt.iter().map(|l| l.to_bits()).collect();
        assert_eq!(
            got,
            want,
            "the loss curve differs from the one pinned in {}",
            golden_path().display()
        );
    });
}

/// The first forward pass from identical weights, carried through the network
/// from `step11_burn`'s per-matmul bound: `h = relu(x W1 + b1)` is within
/// `beta1` of the host's (ReLU does not widen a difference), and the logits
/// `h W2 + b2` within `bound(h, W2) + |W2|^T beta1`. Bias additions round once
/// on each side: `2^-24` of each operand, twice.
#[test]
fn the_first_forward_pass_is_within_the_derived_bound() {
    let split = mnist::load(true);
    let init = init();
    let n = 64;
    let x = split.images[..n * PIXELS].to_vec();
    let host_model = Mlp::<Flex>::new(&init, &FlexDevice);
    let (xt, _) = batch::<Flex>(&split, 0, n, &FlexDevice);
    let h: Vec<f32> = host_model
        .relu
        .forward(host_model.l1.forward(xt.clone()))
        .into_data()
        .to_vec()
        .unwrap();
    let logits: Vec<f32> = host_model.forward(xt).into_data().to_vec().unwrap();

    let w1 = init.l1.0.to_vec::<f32>().unwrap();
    let b1 = init.l1.1.to_vec::<f32>().unwrap();
    let w2 = init.l2.0.to_vec::<f32>().unwrap();
    let b2 = init.l2.1.to_vec::<f32>().unwrap();
    let per =
        |k: usize| 2f64.powi(-9) + 5.0 * k as f64 * 2f64.powi(-23) + k as f64 * 2f64.powi(-24);
    let bias_round = |v: f64, b: f32| 2.0 * 2f64.powi(-24) * (v + f64::from(b.abs()));
    // beta1: the first layer, after the bias.
    let beta1: Vec<f64> = (0..n * HIDDEN)
        .map(|e| {
            let (i, j) = (e / HIDDEN, e % HIDDEN);
            let s: f64 = (0..PIXELS)
                .map(|q| f64::from(x[i * PIXELS + q].abs()) * f64::from(w1[q * HIDDEN + j].abs()))
                .sum();
            s * per(PIXELS) + bias_round(s, b1[j])
        })
        .collect();

    with_device(Config::default(), |d| {
        let model = Mlp::<TtBackend>::new(&init, &d);
        let (xt, _) = batch::<TtBackend>(&split, 0, n, &d);
        let h_tt: Vec<f32> = model
            .relu
            .forward(model.l1.forward(xt.clone()))
            .into_data()
            .to_vec()
            .unwrap();
        let logits_tt: Vec<f32> = model.forward(xt).into_data().to_vec().unwrap();
        for e in 0..n * HIDDEN {
            let err = (f64::from(h_tt[e]) - f64::from(h[e])).abs();
            assert!(err <= beta1[e], "h[{e}]: {err} > {}", beta1[e]);
        }
        let mut worst = 0f64;
        for e in 0..n * CLASSES {
            let (i, c) = (e / CLASSES, e % CLASSES);
            let s: f64 = (0..HIDDEN)
                .map(|j| {
                    f64::from(h_tt[i * HIDDEN + j].abs()) * f64::from(w2[j * CLASSES + c].abs())
                })
                .sum();
            let carried: f64 = (0..HIDDEN)
                .map(|j| beta1[i * HIDDEN + j] * f64::from(w2[j * CLASSES + c].abs()))
                .sum();
            let limit = s * per(HIDDEN) + carried + bias_round(s, b2[c]);
            let err = (f64::from(logits_tt[e]) - f64::from(logits[e])).abs();
            assert!(err <= limit, "logits[{e}]: {err} > {limit}");
            worst = worst.max(err / limit);
        }
        eprintln!("first forward pass: worst logit error {worst:.3} of the bound");
        assert_ne!(
            logits_tt, logits,
            "the forward pass must have run on the device"
        );
    });
}

/// The whole training set, one epoch, then the test set -- silicon only; the
/// simulator gate above is the reduced run.
#[test]
#[cfg(feature = "silicon")]
fn the_mlp_trains_on_full_mnist() {
    let train_split = mnist::load(true);
    let test_split = mnist::load(false);
    let init = init();
    let full = Setup {
        samples: train_split.n,
        batch: 64,
        epochs: 1,
        lr: 0.1,
    };
    let full = Setup {
        samples: full.samples - full.samples % full.batch,
        ..full
    };
    let steps = full.samples / full.batch;
    let t0 = std::time::Instant::now();
    let (host, host_model) = train::<Autodiff<Flex>>(&train_split, &full, &init, &FlexDevice);
    let host_time = t0.elapsed();
    let host_acc = accuracy(&host_model, &test_split, test_split.n, &FlexDevice);
    with_device(Config::default(), |d| {
        let t0 = std::time::Instant::now();
        let (tt, model) = train::<Autodiff<TtBackend>>(&train_split, &full, &init, &d);
        let tt_time = t0.elapsed();
        let acc = accuracy(&model, &test_split, test_split.n, &d);
        for i in (0..steps).step_by(50) {
            eprintln!("step {i:4}: device {:.4}  host {:.4}", tt[i], host[i]);
        }
        eprintln!(
            "test accuracy: device {:.4}, host {:.4}; {:.1} ms/step on the device, \
             {:.1} ms/step on the host",
            acc,
            host_acc,
            tt_time.as_secs_f64() * 1e3 / steps as f64,
            host_time.as_secs_f64() * 1e3 / steps as f64,
        );
        let tail = steps / 10;
        assert!(
            mean(&tt[steps - tail..]) <= DESCENT * tt[0],
            "the loss must fall to {DESCENT} of its first value"
        );
        // A claim about training, not about arithmetic: one epoch of this
        // network reaches 90% on the test set on the host, and must on the
        // device.
        assert!(
            host_acc >= 0.9,
            "the setup itself must reach 90%: {host_acc}"
        );
        assert!(acc >= 0.9, "the device-trained model must reach 90%: {acc}");
    });
}
