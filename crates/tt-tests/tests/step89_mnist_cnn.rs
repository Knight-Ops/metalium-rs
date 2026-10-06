//! The application CNN learns actual MNIST with native convolution gradients.
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{DType, ElementConversion, FloatDType, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_mnist::cnn::{Cnn, Init};
use tt_tests::burn_device::{assert_native_model, with_device, Config};
use tt_tests::mnist::{self, Split, PIXELS};

struct Setup {
    samples: usize,
    batch: usize,
    epochs: usize,
    lr: f64,
}
const REDUCED: Setup = Setup {
    samples: 16,
    batch: 4,
    epochs: 8,
    lr: 0.5,
};

fn train<B: AutodiffBackend>(
    split: &Split,
    init: &Init,
    setup: &Setup,
    dtype: DType,
    device: &B::Device,
    after_step: impl Fn(),
) -> (Vec<f32>, Cnn<B>) {
    let mut model = Cnn::new(init, dtype, device);
    let mut optimizer = SgdConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    let images = Tensor::<B, 2>::from_data(
        TensorData::new(
            split.images[..setup.samples * PIXELS].to_vec(),
            [setup.samples, PIXELS],
        ),
        (device, dtype),
    )
    .to_device(device);
    let labels = Tensor::<B, 1, Int>::from_data(
        TensorData::new(
            split.labels[..setup.samples]
                .iter()
                .map(|&l| i32::from(l))
                .collect::<Vec<_>>(),
            [setup.samples],
        ),
        device,
    )
    .to_device(device);
    let mut losses = Vec::new();
    for _ in 0..setup.epochs {
        for from in (0..setup.samples).step_by(setup.batch) {
            let loss = loss_fn.forward(
                model
                    .forward(images.clone().slice([from..from + setup.batch, 0..PIXELS]))
                    .cast(FloatDType::F32),
                labels.clone().slice_dim(0, from..from + setup.batch),
            );
            losses.push(loss.clone().into_scalar().elem::<f32>());
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(setup.lr, model, grads);
            after_step();
        }
    }
    (losses, model)
}

fn cnn_training(dtype: DType) {
    let split = mnist::load(true);
    let init = Init::default();
    let initial = init
        .conv_weight
        .clone()
        .convert_dtype(dtype)
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    let host = tt_ttsim::outside_fork(|| {
        train::<Autodiff<Flex>>(&split, &init, &REDUCED, dtype, &FlexDevice, || {}).0
    });
    let tail = |losses: &[f32]| losses[losses.len() - 4..].iter().sum::<f32>() / 4.0;
    assert!(
        tail(&host) < 0.8 * host[0],
        "host setup must learn: {host:?}"
    );
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            burn_tt::record_transfers(true);
            let ((losses, model), report) = burn_tt::with_report(|| {
                train::<Autodiff<TtBackend>>(&split, &init, &REDUCED, dtype, &d, || {
                    let downloads: Vec<_> = burn_tt::take_transfers()
                        .into_iter()
                        .filter(|t| t.direction == burn_tt::Direction::Down)
                        .collect();
                    assert_eq!(
                        downloads.len(),
                        1,
                        "only scalar loss may be downloaded per step"
                    );
                    assert_eq!(downloads[0].shape, [1, 1]);
                })
            });
            burn_tt::record_transfers(false);
            assert_native_model(&report);
            assert!(
                tail(&losses) < 0.8 * losses[0],
                "native CNN must learn: {losses:?}"
            );
            let valid = model.valid();
            let trained = valid
                .conv
                .weight
                .val()
                .cast(FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_ne!(
                trained, initial,
                "convolution parameters must receive native gradients and SGD updates"
            );
            eprintln!(
                "MNIST CNN {dtype:?}: loss {:.6} → {:.6}; host {:.6} → {:.6}",
                losses[0],
                tail(&losses),
                host[0],
                tail(&host)
            );
        },
    );
}

#[test]
#[cfg_attr(
    not(feature = "e2e"),
    ignore = "MNIST CNN training: use --features tt-tests/e2e"
)]
fn convolutional_mnist_learns_with_resident_gradients() {
    cnn_training(DType::F32);
}

#[test]
#[cfg(feature = "silicon")]
fn bf16_convolutional_mnist_learns_with_f32_loss() {
    cnn_training(DType::BF16);
}

fn cnn_trace(dtype: DType) {
    let split = mnist::load(true);
    for tiles in [1, 2] {
        with_device(
            Config {
                tiles: Some(burn_tt::TileChoice::Count(tiles)),
                ..Config::default()
            },
            |d| {
                use burn_tt::{InputPayload, TracedTrainingStep};
                use tt_mnist::trace::{
                    collect_parameter_updates, ensure_resident, primitive_float, primitive_int,
                };
                type AD = Autodiff<TtBackend>;
                let init = Init::default();
                let model = Cnn::<AD>::new(&init, dtype, &d);
                let mut reference = Cnn::<AD>::new(&init, dtype, &d);
                let mut optimizer = SgdConfig::new().init();
                let mut reference_optimizer = SgdConfig::new().init();
                let loss_fn = CrossEntropyLossConfig::new().init(&d);
                ensure_resident(&model);
                ensure_resident(&reference);
                let image = |i: usize| split.images[i * PIXELS..(i + 1) * PIXELS].to_vec();
                let x = Tensor::<AD, 2>::from_data(TensorData::new(image(0), [1, PIXELS]), &d);
                let y = Tensor::<AD, 1, Int>::from_data([i32::from(split.labels[0])], &d);
                let xp = primitive_float(x.clone());
                let yp = primitive_int(y.clone());
                let (trace, initial_loss) = TracedTrainingStep::capture(&[&xp, &yp], || {
                    let loss =
                        loss_fn.forward(model.forward(x.cast(dtype)).cast(FloatDType::F32), y);
                    let grads = GradientsParams::from_grads(loss.clone().backward(), &model);
                    let updated = optimizer.step(0.1, model.clone(), grads);
                    (
                        primitive_float(loss),
                        collect_parameter_updates(&model, &updated),
                    )
                })
                .unwrap();
                for i in 0..3 {
                    let x = Tensor::<AD, 2>::from_data(
                        TensorData::new(image(i), [1, PIXELS]),
                        (&d, dtype),
                    );
                    let y = Tensor::<AD, 1, Int>::from_data([i32::from(split.labels[i])], &d);
                    let loss = loss_fn.forward(reference.forward(x).cast(FloatDType::F32), y);
                    let expected = loss.clone().into_scalar().elem::<f32>();
                    let grads = GradientsParams::from_grads(loss.backward(), &reference);
                    reference = reference_optimizer.step(0.1, reference, grads);
                    let actual = if i == 0 {
                        initial_loss
                    } else {
                        trace
                            .step(vec![
                                InputPayload::F32(image(i)),
                                InputPayload::Bits(vec![u32::from(split.labels[i])]),
                            ])
                            .unwrap()
                            .loss
                    };
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{dtype:?}, {tiles} tiles, changed-image step {i}"
                    );
                }
                let weights = |m: Cnn<AD>| {
                    // Read mutable resident buffers, not original creation data.
                    primitive_float(m.conv.weight.val())
                        .download_device()
                        .convert::<f32>()
                        .to_vec::<f32>()
                        .unwrap()
                };
                assert_eq!(
                    weights(model),
                    weights(reference),
                    "traced convolution SGD updates must equal fresh updates"
                );
            },
        );
    }
}

#[test]
fn convolutional_mnist_trace_matches_fresh_changed_batches() {
    cnn_trace(DType::F32);
}

#[test]
#[cfg(feature = "silicon")]
fn bf16_convolutional_mnist_trace_matches_fresh_changed_batches() {
    cnn_trace(DType::BF16);
}

/// Full application model baseline, kept out of smoke due to its dataset size.
#[test]
#[cfg(feature = "silicon")]
#[ignore = "full MNIST CNN epoch and accuracy benchmark"]
fn convolutional_mnist_full_epoch_accuracy_and_timing() {
    let split = mnist::load(true);
    let test = mnist::load(false);
    let init = Init::default();
    let setup = Setup {
        samples: split.n / 64 * 64,
        batch: 64,
        epochs: 1,
        lr: 0.1,
    };
    let accuracy = |model: &Cnn<Flex>| {
        let mut correct = 0;
        for first in (0..test.n).step_by(64) {
            let batch = 64.min(test.n - first);
            let x = Tensor::<Flex, 2>::from_data(
                TensorData::new(
                    test.images[first * PIXELS..(first + batch) * PIXELS].to_vec(),
                    [batch, PIXELS],
                ),
                &FlexDevice,
            );
            let predicted = model
                .forward(x)
                .argmax(1)
                .into_data()
                .to_vec::<i32>()
                .unwrap();
            correct += predicted
                .iter()
                .zip(&test.labels[first..first + batch])
                .filter(|(p, l)| **p == i32::from(**l))
                .count();
        }
        correct as f64 / test.n as f64
    };
    let (host_losses, host_accuracy, host_time) = tt_ttsim::outside_fork(|| {
        let start = std::time::Instant::now();
        let (losses, model) =
            train::<Autodiff<Flex>>(&split, &init, &setup, DType::F32, &FlexDevice, || {});
        let elapsed = start.elapsed();
        (losses, accuracy(&model.valid()), elapsed)
    });
    with_device(Config::default(), |d| {
        let start = std::time::Instant::now();
        let ((losses, model), report) = burn_tt::with_report(|| {
            train::<Autodiff<TtBackend>>(&split, &init, &setup, DType::F32, &d, || {})
        });
        let elapsed = start.elapsed();
        assert_native_model(&report);
        let model = model.valid();
        let mut correct = 0;
        for first in (0..test.n).step_by(64) {
            let batch = 64.min(test.n - first);
            let x = Tensor::<TtBackend, 2>::from_data(
                TensorData::new(
                    test.images[first * PIXELS..(first + batch) * PIXELS].to_vec(),
                    [batch, PIXELS],
                ),
                &d,
            );
            let predicted = model
                .forward(x)
                .argmax(1)
                .into_data()
                .to_vec::<i32>()
                .unwrap();
            correct += predicted
                .iter()
                .zip(&test.labels[first..first + batch])
                .filter(|(p, l)| **p == i32::from(**l))
                .count();
        }
        let acc = correct as f64 / test.n as f64;
        let tail = |v: &[f32]| v[v.len() - 100..].iter().sum::<f32>() / 100.0;
        assert!(tail(&losses) < losses[0] / 2.0);
        assert!(tail(&host_losses) < host_losses[0] / 2.0);
        // Six standard deviations above a uniform random ten-class classifier.
        let chance_bound = 0.1 + 6.0 * (0.1f64 * 0.9 / test.n as f64).sqrt();
        assert!(acc > chance_bound && host_accuracy > chance_bound);
        eprintln!("CNN FULL: {} resident samples, {} steps, device loss {:.6} → {:.6}, accuracy {:.2}%, {:.3} ms/step; Flex loss {:.6} → {:.6}, accuracy {:.2}%, {:.3} ms/step; timing includes preload and scalar losses", setup.samples, losses.len(), losses[0], tail(&losses), acc*100.0, elapsed.as_secs_f64()*1000.0/losses.len() as f64, host_losses[0], tail(&host_losses), host_accuracy*100.0, host_time.as_secs_f64()*1000.0/host_losses.len() as f64);
    });
}
