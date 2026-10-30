//! MOSS-Image-Base's diffusion transformer and VAE decoder on burn-tt against burn-flex.
//!
//! The same strictly loaded weights and the same synthetic inputs (`tt-models`'s
//! `moss_image_host`, whose host parts are checked against the reference pipeline) go through one
//! transformer step and one VAE decode on the host (Flex) and on the device, the parts one at a
//! time. What is claimed: both run on the device (`assert_native_model`: the four-axis rotary
//! embedding, the shared modulation, non-causal attention over 256 image and 12 text tokens, the
//! situ MLP; the VAE's convolutions, group norms, pixel attention and nearest upsampling as
//! element moves), and the device's outputs are those of the host to within a stated fraction of
//! their RMS, printed. A model workload: silicon only.

use burn::tensor::{DType, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{TtBackend, TtDevice};
use tt_models::llama::Precision;
use tt_models::moss_image;
use tt_models::moss_image::vae::{Vae, FLUX2_VAE};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn synth(n: usize, a: f64, b: f64, c: f64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let i = i as f64;
            ((a * i).sin() + 0.7 * (b * i + 1.3).sin() + 0.5 * (c * i).cos()) as f32
        })
        .collect()
}

fn inputs<B: burn::tensor::backend::Backend>(d: &B::Device) -> (Tensor<B, 4>, Tensor<B, 3>) {
    let latent = synth(128 * 16 * 16, 0.91, 2.17, 0.123);
    let context: Vec<f32> = synth(12 * 4096, 0.37, 1.61, 0.071)
        .into_iter()
        .map(|v| 0.5 * v)
        .collect();
    (
        Tensor::from_data(TensorData::new(latent, [1, 128, 16, 16]), d),
        Tensor::from_data(TensorData::new(context, [1, 12, 4096]), d),
    )
}

fn values<B: burn::tensor::backend::Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.cast(DType::F32).into_data().to_vec::<f32>().unwrap()
}

/// The largest difference between `device` and `host`, and the root mean square of the
/// difference, each as a fraction of the host's RMS.
fn relative(host: &[f32], device: &[f32]) -> (f64, f64) {
    assert_eq!(host.len(), device.len());
    assert!(device.iter().all(|v| v.is_finite()), "a non-finite output");
    let n = host.len() as f64;
    let rms = (host.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n).sqrt();
    let diffs = host
        .iter()
        .zip(device)
        .map(|(&a, &b)| (a as f64 - b as f64).abs());
    let worst = diffs.clone().fold(0f64, f64::max);
    let spread = (diffs.map(|d| d * d).sum::<f64>() / n).sqrt();
    (worst / rms, spread / rms)
}

/// The transformer's blocks run: the first `k` (the rest skipped), so the error's growth with
/// depth is measured on the real weights.
const DEPTHS: [usize; 3] = [1, 8, 42];

fn skip(k: usize) -> Vec<usize> {
    (k..42).collect()
}

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask fetch-weights --model moss-image-base, then cargo xtask silicon --release --filter step262_moss_image"
)]
fn a_transformer_step_and_a_decode_on_the_device_agree_with_flex() {
    let (host_steps, host_image) = {
        let (dit, _) = moss_image::transformer::<Flex>(Precision::F32, &FlexDevice).unwrap();
        let (latent, context) = inputs::<Flex>(&FlexDevice);
        let steps: Vec<Vec<f32>> = DEPTHS
            .iter()
            .map(|&k| values(dit.forward(latent.clone(), 750.0, context.clone(), &skip(k))))
            .collect();
        drop(dit);
        let (vae, _) =
            Vae::<Flex>::pretrained(&FLUX2_VAE, moss_image::WEIGHTS, &FlexDevice).unwrap();
        (steps, values(vae.decode(vae.unpatch(latent))))
    };

    with_device(Config::default(), |d: TtDevice| {
        let ((steps, image), report) = burn_tt::with_report(|| {
            let (latent, context) = inputs::<TtBackend>(&d);
            let steps: Vec<Vec<f32>> = {
                let (dit, _) = moss_image::transformer::<TtBackend>(Precision::F32, &d).unwrap();
                DEPTHS
                    .iter()
                    .map(|&k| values(dit.forward(latent.clone(), 750.0, context.clone(), &skip(k))))
                    .collect()
            };
            let (vae, _) =
                Vae::<TtBackend>::pretrained(&FLUX2_VAE, moss_image::WEIGHTS, &d).unwrap();
            (steps, values(vae.decode(vae.unpatch(latent))))
        });
        eprintln!("{report}");
        assert_native_model(&report);
        let mut worst = 0f64;
        for ((k, host), device) in DEPTHS.iter().zip(&host_steps).zip(&steps) {
            let (max, spread) = relative(host, device);
            eprintln!("{k} blocks: |flex - device| at most {max:.3e} of the rms, rms {spread:.3e}");
            worst = worst.max(max);
        }
        let (max, spread) = relative(&host_image, &image);
        eprintln!("the image: |flex - device| at most {max:.3e} of the rms, rms {spread:.3e}");
        // A guard, not an error bound (none is derived yet): the matrix unit computes these F32
        // products in TF32, and the measured difference grows with depth (printed); a wrong
        // block or a wrong constant moves an output by its whole size.
        assert!(worst < 0.25 && max < 0.25, "step {worst:e}, image {max:e}");
    });
}
