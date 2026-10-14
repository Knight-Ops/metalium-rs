//! Rung 1 of the model ladder: pretrained ResNet-18 on burn-tt, against burn-flex.
//!
//! The same weights (`cargo xtask fetch-weights`, loaded strictly by `tt_models::load`) and the
//! same preprocessed photograph go through the network on the host (Flex) and on the device
//! (`TtBackend`). The host run is itself checked against the world in `tt-models`'s
//! `resnet18_host` (the cat photo is a cat), so agreement here is agreement on a working network,
//! not on nonsense.
//!
//! On ttsim the image is `SMALL` pixels square, because the simulator cannot afford 224; the
//! network is fully convolutional up to its global pool, so every layer still runs, only with less
//! spatial extent. On silicon (`--features silicon`) it is 224, the size the network was trained at.
//!
//! What is claimed:
//!
//! * **The model runs on the device**: `assert_native_model` (no staged compute, no host arithmetic,
//!   no downloads of operands), so the comparison cannot be a host fallback agreeing with itself.
//! * **The classification agrees**: the top-1 class equals Flex's. The error against Flex is
//!   measured and printed, per logit, so a change in it is visible; it is not turned into a
//!   tolerance here, because the Matrix Unit's rounding has a derived bound per matmul
//!   (`step158_matmul_accumulation`) and not one for a 20-layer network.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_models::image_input::{load_chw, IMAGENET_MEAN, IMAGENET_STD};
use tt_models::resnet::ResNet18;
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// Spatial size of the simulator lane. Silicon runs the training resolution.
const SMALL: u32 = 64;

fn size() -> u32 {
    if cfg!(feature = "silicon") {
        224
    } else {
        SMALL
    }
}

fn chw(image: &str) -> Vec<f32> {
    load_chw(image, size(), IMAGENET_MEAN, IMAGENET_STD).unwrap()
}

fn flex_logits(image: &str) -> Vec<f32> {
    let s = size() as usize;
    let (model, _) = ResNet18::<Flex>::pretrained(&FlexDevice).unwrap();
    let x = Tensor::<Flex, 4>::from_data(TensorData::new(chw(image), [1, 3, s, s]), &FlexDevice);
    model.forward(x).into_data().to_vec::<f32>().unwrap()
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

fn top(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx.truncate(k);
    idx
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-tests/weights"
)]
fn resnet18_on_the_device_classifies_like_flex() {
    let image = "cats.jpg";
    let s = size() as usize;
    let host = flex_logits(image);
    with_device(Config::default(), |d| {
        let (model, loaded) = ResNet18::<TtBackend>::pretrained(&d).unwrap();
        assert_eq!(loaded.applied, 102);
        let x = Tensor::<TtBackend, 4>::from_data(TensorData::new(chw(image), [1, 3, s, s]), &d);
        let (device, report) =
            burn_tt::with_report(|| model.forward(x).into_data().to_vec::<f32>().unwrap());
        eprintln!("{report}");
        assert_native_model(&report);

        let max_abs = host
            .iter()
            .zip(&device)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = host.iter().map(|v| v.abs()).fold(0f32, f32::max);
        eprintln!(
            "{image} at {s}x{s}: flex top-5 {:?}, device top-5 {:?}; max |flex - device| {max_abs:.4e} \
             against a largest logit of {scale:.4}",
            top(&host, 5),
            top(&device, 5)
        );
        assert_eq!(device.len(), 1000);
        assert!(device.iter().all(|v| v.is_finite()), "a non-finite logit");
        assert_eq!(argmax(&device), argmax(&host), "the top-1 class differs");
    });
}
