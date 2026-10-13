//! Rung 2 of the model ladder: pretrained ViT-Tiny on burn-tt, against burn-flex.
//!
//! The same shape of gate as `step165_resnet18`: the same strictly loaded weights and the same
//! photograph through the network on the host (Flex) and on the device. The host run is checked
//! against the world in `tt-models`'s `vit_host` (the cat photo is a cat).
//!
//! The image is always 224x224, because the learned position embedding is a fixed 197 tokens. The
//! simulator lane therefore cannot shrink the image; it runs the first `SMALL_DEPTH` of the twelve
//! blocks of the real network instead (the other blocks' tensors are named and ignored on load).
//! Silicon (`--features silicon`) runs all twelve.
//!
//! What is claimed:
//!
//! * **The model runs on the device**: `assert_native_model` (no staged compute, no host
//!   arithmetic, no downloads of operands), so the comparison cannot be a host fallback agreeing
//!   with itself. Layer norm, GELU, softmax and Burn's `attention` all run through `burn-tt`.
//! * **The classification agrees**: top-1 equals Flex's. The error against Flex is measured and
//!   printed, not turned into a tolerance (see `step165_resnet18` for why).

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_models::image_input::{load_chw, HALF_MEAN, HALF_STD};
use tt_models::vit::{VitTiny, FULL_DEPTH, IMAGE};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// Blocks the simulator lane runs.
const SMALL_DEPTH: usize = 1;

fn depth() -> usize {
    if cfg!(feature = "silicon") {
        FULL_DEPTH
    } else {
        SMALL_DEPTH
    }
}

fn chw(image: &str) -> Vec<f32> {
    load_chw(image, IMAGE as u32, HALF_MEAN, HALF_STD).unwrap()
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
fn vit_tiny_on_the_device_classifies_like_flex() {
    let image = "cats.jpg";
    let host = {
        let (model, _) = VitTiny::<Flex>::pretrained(depth(), &FlexDevice).unwrap();
        let x = Tensor::<Flex, 4>::from_data(
            TensorData::new(chw(image), [1, 3, IMAGE, IMAGE]),
            &FlexDevice,
        );
        model.forward(x).into_data().to_vec::<f32>().unwrap()
    };
    with_device(Config::default(), |d| {
        let (model, _) = VitTiny::<TtBackend>::pretrained(depth(), &d).unwrap();
        let x = Tensor::<TtBackend, 4>::from_data(
            TensorData::new(chw(image), [1, 3, IMAGE, IMAGE]),
            &d,
        );
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
            "{image}, {} of {FULL_DEPTH} blocks: flex top-5 {:?}, device top-5 {:?}; \
             max |flex - device| {max_abs:.4e} against a largest logit of {scale:.4}",
            depth(),
            top(&host, 5),
            top(&device, 5)
        );
        assert_eq!(device.len(), 1000);
        assert!(device.iter().all(|v| v.is_finite()), "a non-finite logit");
        assert_eq!(argmax(&device), argmax(&host), "the top-1 class differs");
    });
}
