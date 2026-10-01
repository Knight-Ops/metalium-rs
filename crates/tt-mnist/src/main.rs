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
//! card's GDDR6. The cross-entropy loss is computed on the host.
//!
//! ```text
//! tt-mnist [--card N | --cards 0,1] [--tiles T] [--epochs E] [--steps S] [--host]
//! ```

use std::time::{Duration, Instant};

use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, Relu};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{attach_topology, Fidelity, SrcRoute, Topology, TtBackend, TtDevice};

const PIXELS: usize = 28 * 28;
const HIDDEN: usize = 128;
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
    l1: Linear<B>,
    l2: Linear<B>,
    relu: Relu,
}

/// The initial weights, drawn once so the card and the host start the same.
struct Init {
    l1: (TensorData, TensorData),
    l2: (TensorData, TensorData),
}

fn init() -> Init {
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
    Init {
        l1: layer(PIXELS, HIDDEN),
        l2: layer(HIDDEN, CLASSES),
    }
}

impl<B: Backend> Mlp<B> {
    fn new(init: &Init, device: &B::Device) -> Self {
        let linear = |(w, b): &(TensorData, TensorData)| Linear {
            weight: Param::from_tensor(Tensor::from_data(w.clone(), device)),
            bias: Some(Param::from_tensor(Tensor::from_data(b.clone(), device))),
        };
        Mlp {
            l1: linear(&init.l1),
            l2: linear(&init.l2),
            relu: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.l2.forward(self.relu.forward(self.l1.forward(x)))
    }
}

// --- Training -----------------------------------------------------------------

struct Run {
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
    let mut model = Mlp::<B>::new(init, device);
    let mut optim = SgdConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let samples = train.n - train.n % BATCH;
    // Uploaded once; each batch is a view of it (on the card: no copy).
    let images: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(train.images[..samples * PIXELS].to_vec(), [samples, PIXELS]),
        device,
    )
    .to_device(device);
    let preload = t0.elapsed();

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
            let loss = loss_fn.forward(model.forward(x), y);
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
    for from in (0..test.n).step_by(1000) {
        let m = 1000.min(test.n - from);
        let x = Tensor::<B::InnerBackend, 2>::from_data(
            TensorData::new(
                test.images[from * PIXELS..(from + m) * PIXELS].to_vec(),
                [m, PIXELS],
            ),
            device,
        );
        let pred = model.forward(x).argmax(1).reshape([m]);
        let y = labels::<B::InnerBackend>(test, from, m, device);
        right += pred.equal(y).int().sum().into_scalar().elem::<i64>() as usize;
    }
    Run {
        losses,
        accuracy: right as f64 / test.n as f64,
        preload,
        train: train_time,
        steps: step,
    }
}

// --- The command line ---------------------------------------------------------

struct Args {
    topology: Topology,
    epochs: usize,
    steps: usize,
    host: bool,
}

const USAGE: &str = "\
usage: tt-mnist [--card N | --cards 0,1] [--tiles T] [--epochs E] [--steps S] [--host]

  --card N      train on /dev/tenstorrent/N (default 0)
  --cards 0,1   several cabled cards, matmuls sharded over Ethernet
  --tiles T     compute on T Tensix tiles of the card, or `all` (default 1)
  --epochs E    passes over the 60 000 training images (default 1)
  --steps S     stop after S steps
  --host        also train on the host CPU (burn-flex), for comparison";

fn args() -> Result<Args, String> {
    let mut a = Args {
        topology: Topology::single(0),
        epochs: 1,
        steps: usize::MAX,
        host: false,
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
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n\n{USAGE}")),
        }
    }
    // After the loop, so `--tiles` applies whichever order it came in.
    if let Some(t) = tiles {
        a.topology = a.topology.on_tiles(t).map_err(|e| e.to_string())?;
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
    println!("tt-mnist: a {PIXELS}-{HIDDEN}-{CLASSES} network learning MNIST, in Rust, on Tenstorrent Blackhole");
    println!("  device   {cards}");
    println!("  model    Burn nn::Linear x2 + ReLU, cross-entropy, SGD lr {LR}, batch {BATCH}");

    let t0 = Instant::now();
    let (train_split, test_split) = mnist();
    println!(
        "  data     {} training / {} test images, embedded ({:.2?} to unpack)\n",
        train_split.n,
        test_split.n,
        t0.elapsed()
    );
    let init = init();

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
    let before = burn_tt::tensor_traffic();
    let card = train::<Autodiff<TtBackend>>(
        &train_split,
        &test_split,
        &init,
        a.epochs,
        a.steps,
        &device,
        |step, loss| println!("  step {step:>5}   loss {loss:.4}"),
    );
    let moved = burn_tt::tensor_traffic() - before;
    drop(guard);

    let per = |d: Duration| d.as_secs_f64() * 1e3 / card.steps.max(1) as f64;
    println!();
    println!("on the card:");
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
