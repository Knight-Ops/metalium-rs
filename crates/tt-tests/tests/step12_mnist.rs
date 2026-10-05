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
//!
//! The training runs are the end-to-end tier (`tt-tests`'s `e2e` feature): on
//! ttsim they run only with it, on silicon always. The first-forward-pass
//! bound is cheap and stays in the default tier.

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
// Host work in the parent goes through it, so no test forks mid-Burn.
use tt_ttsim::outside_fork;

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
    (Tensor::from_data(x, device), labels(split, from, n, device))
}

/// Labels `[from, from + n)` of `split`.
fn labels<B: Backend>(
    split: &Split,
    from: usize,
    n: usize,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let y = TensorData::new(
        split.labels[from..from + n]
            .iter()
            .map(|&l| i32::from(l))
            .collect::<Vec<i32>>(),
        [n],
    );
    Tensor::from_data(y, device)
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
    train_timed(split, setup, init, device, &mut |_| {}).0
}

/// What [`train_timed`] returns: the losses and model, the dataset's upload
/// time, the steps' time without it, and `burn_tt::device_time` when the steps
/// began.
type Timed<B> = (
    (Vec<f32>, Mlp<B>),
    std::time::Duration,
    std::time::Duration,
    Vec<(&'static str, u64, std::time::Duration)>,
);

/// [`train`], and how long the steps took, without the dataset's upload.
/// `after_step` is called with each step's index once its optimizer step is
/// done, so a gate can sample counters per step.
fn train_timed<B: AutodiffBackend>(
    split: &Split,
    setup: &Setup,
    init: &Init,
    device: &B::Device,
    after_step: &mut dyn FnMut(usize),
) -> Timed<B> {
    let t0 = std::time::Instant::now();
    let mut model = Mlp::<B>::new(init, device);
    let mut optim = SgdConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let mut losses = Vec::new();
    // The images go to the device once; each batch is a row slice of them. On
    // `burn-tt`, `to_device` makes the tensor resident in GDDR and a slice of
    // whole tile rows is a view, so no image crosses PCIe after this.
    let images: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(
            split.images[..setup.samples * PIXELS].to_vec(),
            [setup.samples, PIXELS],
        ),
        device,
    )
    .to_device(device);
    let preload = t0.elapsed();
    let device_before = burn_tt::device_time();
    let t0 = std::time::Instant::now();
    for _ in 0..setup.epochs {
        for from in (0..setup.samples).step_by(setup.batch) {
            let y = labels::<B>(split, from, setup.batch, device);
            let x = images.clone().slice([from..from + setup.batch, 0..PIXELS]);
            let loss = loss_fn.forward(model.forward(x), y);
            losses.push(loss.clone().into_scalar().elem::<f32>());
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(setup.lr, model, grads);
            after_step(losses.len() - 1);
        }
    }
    ((losses, model), preload, t0.elapsed(), device_before)
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
        let pred = model
            .forward(x)
            .argmax(1)
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .unwrap();
        let labels = y.into_data().convert::<i32>().to_vec::<i32>().unwrap();
        right += pred.iter().zip(&labels).filter(|(p, l)| p == l).count();
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

/// What one step may move between host and device once the dataset and
/// weights are resident (Phase 9.5), each with why. Everything else stays in
/// GDDR; only labels, autodiff initial values, and scalar readback cross.
/// Gather/scatter index grids are replayable mover metadata, not tensor uploads.
fn steady_state_transfers(batch: usize) -> Vec<(burn_tt::Transfer, &'static str)> {
    use burn_tt::{Direction::*, Transfer};
    let t = |direction, shape| Transfer { direction, shape };
    vec![
        (t(Up, [batch, 1]), "the target indices"),
        (t(Down, [1, 1]), "the scalar loss readback"),
        (t(Up, [1, 1]), "autodiff's scalar seed"),
        (t(Up, [1, batch]), "the full mean's backward initial values"),
        (t(Up, [batch, CLASSES]), "the scatter's initial zeros"),
        (
            t(Up, [batch, CLASSES]),
            "the log-softmax backward initial values",
        ),
    ]
}

/// The Phase 9.5 claim, from counters sampled after every step: after the
/// first step, each step moves exactly [`steady_state_transfers`] as tensor
/// data, and writes the same number of bytes to the device -- descriptors,
/// programs and tensors together. Reads are printed, not asserted: they are
/// mostly completion polls, whose count depends on timing on silicon.
fn assert_steady_state_traffic(
    per_step: &[(
        burn_tt::TensorTraffic,
        tt_device::Traffic,
        Vec<burn_tt::Transfer>,
    )],
) {
    let want = steady_state_transfers(REDUCED.batch);
    let named = |t: &burn_tt::Transfer| {
        want.iter()
            .find(|(w, _)| w == t)
            .map_or("NOT EXPECTED", |(_, why)| *why)
    };
    let want_bytes: u64 = want
        .iter()
        .map(|(t, _)| (t.shape[0] * t.shape[1] * 4) as u64)
        .sum();
    let writes = per_step[1].1.bytes_written - per_step[0].1.bytes_written;
    for (i, w) in per_step.windows(2).enumerate() {
        let step = i + 1;
        let got: Vec<_> = w[1].2.clone();
        assert!(
            got == want.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            "step {step} moved other tensors than the steady-state budget:\n  got  {:#?}\n  want {:#?}",
            got.iter().map(|t| (t, named(t))).collect::<Vec<_>>(),
            want,
        );
        let (t, dv) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
        assert_eq!(t.uploaded + t.downloaded, want_bytes, "step {step}");
        assert_eq!(
            dv.bytes_written, writes,
            "step {step} wrote a different number of bytes to the device"
        );
    }
    let last = per_step[per_step.len() - 1].1 - per_step[per_step.len() - 2].1;
    println!(
        "MEASURE steady-state step: {want_bytes} B of tensors; device {writes} B written in {} \
         writes ({} retargets), {} B read in {} reads",
        last.write_calls, last.retargets, last.bytes_read, last.read_calls
    );
}

#[test]
#[cfg_attr(
    not(feature = "e2e"),
    ignore = "end-to-end training on ttsim: run with --features tt-tests/e2e"
)]
fn the_mlp_trains_on_a_reduced_dataset() {
    reduced_run_matches_the_golden(Config {
        ..Config::default()
    });
}

/// Phase 9.6: the same run with every GDDR op dealt over four Tensix tiles.
/// Splitting a matmul by output blocks with `K` whole, element-wise ops by
/// tile and column sums by column changes no accumulation, so the claim is
/// the single-tile golden bit for bit, and the same steady-state traffic.
#[test]
#[cfg_attr(
    not(feature = "e2e"),
    ignore = "end-to-end training on ttsim: run with --features tt-tests/e2e"
)]
fn the_mlp_trains_on_four_tiles_matching_the_golden() {
    reduced_run_matches_the_golden(Config {
        tiles: Some(burn_tt::TileChoice::Count(4)),

        ..Config::default()
    });
}

/// The reduced run on a device configured by `config`, held to the golden
/// and to [`assert_steady_state_traffic`]. Only the single-tile default
/// blesses the golden.
fn reduced_run_matches_the_golden(config: Config) {
    let split = mnist::load(true);
    let init = init();
    let (host, _) = outside_fork(|| train::<Autodiff<Flex>>(&split, &REDUCED, &init, &FlexDevice));
    let steps = REDUCED.samples / REDUCED.batch;
    assert!(
        mean(&host[host.len() - steps..]) <= DESCENT * host[0],
        "the setup itself must train: host {host:?}"
    );
    let bless = std::env::var_os("TT_BLESS").is_some() && config.tiles.is_none();
    let want = golden();
    with_device(config, |d| {
        let before = burn_tt::tensor_traffic();
        // Counters after every step, so steady state can be told from the
        // first step's uploads.
        burn_tt::record_transfers(true);
        let mut per_step = Vec::new();
        let (training, report) = burn_tt::with_report(|| {
            train_timed::<Autodiff<TtBackend>>(&split, &REDUCED, &init, &d, &mut |_| {
                per_step.push((
                    burn_tt::tensor_traffic(),
                    burn_tt::device_traffic(d).expect("the engine reports its traffic"),
                    burn_tt::take_transfers(),
                ))
            })
        });
        let ((tt, _), ..) = training;
        tt_tests::burn_device::assert_native_model(&report);
        eprintln!("{report}");
        burn_tt::record_transfers(false);
        let moved = burn_tt::tensor_traffic() - before;
        assert_steady_state_traffic(&per_step);
        let n = tt.len() as u64;
        println!(
            "MEASURE tensor traffic per step, averaged over the run with the preload: {} B up in {} uploads, {} B down in {} downloads",
            moved.uploaded / n,
            moved.uploads / n,
            moved.downloaded / n,
            moved.downloads / n
        );
        assert!(
            moved.uploads > 0,
            "the matmuls must have run on device-resident tensors"
        );
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
    let (h, logits) = outside_fork(|| {
        let host_model = Mlp::<Flex>::new(&init, &FlexDevice);
        let (xt, _) = batch::<Flex>(&split, 0, n, &FlexDevice);
        let h: Vec<f32> = host_model
            .relu
            .forward(host_model.l1.forward(xt.clone()))
            .into_data()
            .to_vec()
            .unwrap();
        let logits: Vec<f32> = host_model.forward(xt).into_data().to_vec().unwrap();
        (h, logits)
    });

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
    let ((host, host_model), _, host_time, _) = outside_fork(|| {
        train_timed::<Autodiff<Flex>>(&train_split, &full, &init, &FlexDevice, &mut |_| {})
    });
    let host_acc = outside_fork(|| accuracy(&host_model, &test_split, test_split.n, &FlexDevice));
    with_device(Config::default(), |d| {
        let t0 = std::time::Instant::now();
        let ((tt, model), preload, tt_time, before) =
            train_timed::<Autodiff<TtBackend>>(&train_split, &full, &init, &d, &mut |_| {});
        let before: std::collections::HashMap<_, _> =
            before.into_iter().map(|(k, n, t)| (k, (n, t))).collect();
        let _ = t0;
        eprintln!(
            "MEASURE preload (model and {} images to the device): {preload:.2?}",
            full.samples
        );
        let mut device = std::time::Duration::ZERO;
        for (k, n, t) in burn_tt::device_time() {
            let (n0, t0) = before.get(k).copied().unwrap_or_default();
            let (n, t) = (n - n0, t - t0);
            device += t;
            eprintln!(
                "MEASURE per step {k:<12} {:>5.2} calls {:>7.3} ms",
                n as f64 / steps as f64,
                t.as_secs_f64() * 1e3 / steps as f64
            );
        }
        eprintln!(
            "MEASURE per step host (Burn, autodiff, loss) {:.3} ms",
            (tt_time - device).as_secs_f64() * 1e3 / steps as f64
        );
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

/// Phase 8: the reduced run with every matmul split along `N` across `chips`
/// chips joined by Ethernet (`tt_kernels::shard`), operands entering and
/// results leaving through chip 0. The claim is the strong one: the loss curve
/// is the single-chip golden, bit for bit -- sharding along `N` with `K` whole
/// changes no accumulation order.
fn sharded_training_matches_the_golden(chips: usize) {
    let split = mnist::load(true);
    let init = init();
    let want = golden().unwrap_or_else(|| {
        panic!(
            "{} is missing: run the_mlp_trains_on_a_reduced_dataset with TT_BLESS=1",
            golden_path().display()
        )
    });
    tt_tests::burn_device::with_mesh_device(Config::default(), chips, |d| {
        let ((tt, _), report) =
            burn_tt::with_report(|| train::<Autodiff<TtBackend>>(&split, &REDUCED, &init, &d));
        tt_tests::burn_device::assert_native_model(&report);
        let got: Vec<u32> = tt.iter().map(|l| l.to_bits()).collect();
        let first = got.iter().zip(&want).position(|(g, w)| g != w);
        assert_eq!(
            got, want,
            "the {chips}-chip loss curve differs from the golden, first at step {first:?}"
        );
    });
}

#[test]
#[cfg_attr(
    not(feature = "e2e"),
    ignore = "end-to-end training on ttsim: run with --features tt-tests/e2e"
)]
fn the_mlp_trains_sharded_over_two_chips_matching_the_golden() {
    sharded_training_matches_the_golden(2);
}

#[cfg(not(feature = "silicon"))]
#[test]
#[cfg_attr(
    not(feature = "e2e"),
    ignore = "end-to-end training on ttsim: run with --features tt-tests/e2e"
)]
fn the_mlp_trains_sharded_round_a_four_chip_ring_matching_the_golden() {
    sharded_training_matches_the_golden(4);
}
