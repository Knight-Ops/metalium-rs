//! MOSS-Image-Base on the host, part by part against its reference pipeline: the text encoder's
//! encodings of a prompt and of the empty negative prompt, one transformer step on a synthetic
//! latent and context (with and without the block the negative branch skips), the Euler schedule,
//! a short guided denoising of synthetic noise, and the VAE's images.
//!
//! Every reference value here was printed by `crates/tt-models/reference/moss_image_reference.py`
//! (the checkpoint's own `pipeline_moss_image.py` at the pinned revision, `diffusers` 0.38.0,
//! `transformers` 5.19.0, torch 2.14.1+cpu, `float32`):
//!
//! ```text
//! uv run crates/tt-models/reference/moss_image_reference.py \
//!     --revision 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3 \
//!     --weights vendor/weights/moss-image-base \
//!     --prompt "A red apple on a wooden table" --steps 4
//! ```
//!
//! A tensor is compared by its mean, its root mean square and sixteen evenly spaced elements
//! ([`Stats`]); the synthetic inputs are [`synth`], computed identically there. Needs `cargo xtask
//! fetch-weights --model moss-image-base` (27.2 GB) and about 34 GB of memory: the parts are loaded
//! one at a time.

use burn::tensor::{DType, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_models::llama::Precision;
use tt_models::moss_image::vae::{Vae, FLUX2_VAE};
use tt_models::moss_image::{self, denoise, encode, Schedule, GUIDANCE, UNCOND_SKIP};
use tt_models::qwen35::{Qwen35Config, MOSS_TEXT_ENCODER, QWEN3_5_4B};

// The reference's values (see the module docs).
const ENCODER_HIDDEN: usize = 4096;
const ENCODER_FF: usize = 12288;
const PROMPT_IDS: [i32; 17] = [
    248045, 846, 198, 32, 2438, 22544, 383, 264, 21664, 1898, 248046, 198, 248045, 74455, 198,
    248068, 198,
];
const NEGATIVE_IDS: [i32; 10] = [
    248045, 846, 198, 248046, 198, 248045, 74455, 198, 248068, 198,
];
const PROMPT_ENCODING: Stats = Stats {
    n: 69632,
    mean: 2.023296e-02,
    rms: 2.018319e+00,
    samples: [
        1.220819,
        -8.102286e-1,
        -8.862202e-1,
        -9.329239e-1,
        2.31066e-1,
        -2.982407,
        -1.199211,
        -9.102671,
        -3.399361e-1,
        -9.315763e-2,
        1.636794,
        -3.099399,
        1.916645,
        6.201366e-1,
        1.857455,
        2.984564,
    ],
};
const NEGATIVE_ENCODING: Stats = Stats {
    n: 40960,
    mean: 1.363632e-02,
    rms: 1.971176e+00,
    samples: [
        -2.633674e-1,
        2.664504,
        -2.036747,
        1.15677,
        4.775617e-1,
        3.844198,
        2.014042,
        -7.145818e-1,
        1.629284,
        1.959585e-2,
        -3.079499e-1,
        -7.008327e-1,
        -4.143081e-1,
        3.106847,
        4.535743,
        1.49362,
    ],
};
const STEP: Stats = Stats {
    n: 32768,
    mean: -2.038074e-02,
    rms: 8.121727e-01,
    samples: [
        6.656645e-1,
        -4.229979e-1,
        6.225282e-1,
        -1.260537,
        -7.688735e-1,
        1.404889,
        -1.572609,
        7.062246e-2,
        2.725957e-1,
        -6.33741e-1,
        1.407966,
        -1.090906,
        -1.335538e-1,
        -2.252009e-1,
        2.019694e-1,
        -2.300125e-1,
    ],
};
const STEP_WITHOUT_22: Stats = Stats {
    n: 32768,
    mean: -1.283388e-02,
    rms: 7.905637e-01,
    samples: [
        1.018712,
        -4.155064e-1,
        5.171416e-1,
        -1.244446,
        -9.04629e-1,
        1.474336,
        -1.701756,
        1.832017e-1,
        1.715323e-1,
        -3.842868e-1,
        1.150925,
        -9.249331e-1,
        6.141454e-3,
        -1.614496e-2,
        3.576943e-1,
        -2.360483e-1,
    ],
};
const STEPS: usize = 4;
const SIGMAS: [f32; 5] = [1.0, 0.90999997, 0.717, 0.024, 0.0];
const TIMESTEPS: [f32; 4] = [1000.0, 909.99994, 717.0, 24.0];
const DENOISED: Stats = Stats {
    n: 32768,
    mean: -7.559610e-02,
    rms: 1.288191e+00,
    samples: [
        9.72281e-1,
        -1.926622,
        -1.395107,
        1.193774,
        -1.502698e-1,
        -2.130284,
        1.926607,
        6.533908e-1,
        1.275973e-1,
        1.062199,
        -2.235676e-1,
        -1.398669,
        4.015114e-1,
        4.264854e-1,
        -7.651698e-1,
        1.222305,
    ],
};
const DECODED_LATENT: Stats = Stats {
    n: 196608,
    mean: -1.218295e-01,
    rms: 3.429445e-01,
    samples: [
        -4.840445e-2,
        -8.137001e-2,
        1.347888e-1,
        2.66938e-1,
        1.701104e-1,
        1.909474e-1,
        8.943567e-3,
        -5.534959e-2,
        -2.261192e-1,
        -1.716665e-1,
        -5.075861e-1,
        -6.367085e-1,
        -4.126395e-1,
        -6.41982e-1,
        -5.04531e-1,
        -3.460235e-1,
    ],
};
const IMAGE: Stats = Stats {
    n: 196608,
    mean: -2.880299e-01,
    rms: 7.133746e-01,
    samples: [
        -7.927896e-1,
        3.069395e-1,
        -7.287376e-1,
        -6.217057e-1,
        5.452662e-1,
        -1.0,
        -2.478333e-1,
        -4.959867e-2,
        -1.0,
        -8.661873e-3,
        -7.36598e-1,
        3.536775e-1,
        -8.105134e-1,
        -4.692186e-1,
        -1.127742e-1,
        -1.0,
    ],
};

/// `sin(a i) + 0.7 sin(b i + 1.3) + 0.5 cos(c i)` in F64, rounded to F32: the reference's
/// synthetic inputs.
fn synth(n: usize, a: f64, b: f64, c: f64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let i = i as f64;
            ((a * i).sin() + 0.7 * (b * i + 1.3).sin() + 0.5 * (c * i).cos()) as f32
        })
        .collect()
}

fn tensor<const D: usize>(values: Vec<f32>, shape: [usize; D]) -> Tensor<Flex, D> {
    Tensor::from_data(TensorData::new(values, shape), &FlexDevice)
}

fn values<const D: usize>(t: Tensor<Flex, D>) -> Vec<f32> {
    t.cast(DType::F32).into_data().to_vec::<f32>().unwrap()
}

/// What the reference prints of a tensor.
struct Stats {
    n: usize,
    mean: f64,
    rms: f64,
    samples: [f32; 16],
}

/// `got` against the reference's `want`: the same element count, and the mean, the root mean
/// square and each sample within `guard` of the reference's root mean square. Measured: 2.7e-6
/// to 2.4e-5 of the RMS across the encodings, the steps, the denoised latents and the images. The
/// guard is far looser than that, because summation order differs between CPUs
/// and libraries; it exists to catch a wrong block or a wrong constant, which moves these by
/// whole fractions of the RMS, not to state an error bound.
fn matches(got: &[f32], want: &Stats, guard: f64, what: &str) {
    assert_eq!(got.len(), want.n, "{what}: element count");
    let n = got.len();
    let mean = got.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let rms = (got.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    let picks: Vec<f64> = (0..16).map(|k| got[k * n / 16 + n / 32] as f64).collect();
    let worst = picks
        .iter()
        .zip(want.samples)
        .map(|(g, w)| (g - w as f64).abs())
        .chain([(mean - want.mean).abs(), (rms - want.rms).abs()])
        .fold(0f64, f64::max)
        / want.rms;
    eprintln!(
        "{what}: rms {rms:.6e} (reference {:.6e}); largest difference {worst:.2e} of the rms",
        want.rms
    );
    assert!(
        worst < guard,
        "{what}: {worst:e} of the rms; samples {picks:?}"
    );
}

#[test]
fn the_encoder_preset_is_the_checkpoints_shape() {
    // The reference prints the encoder's `text_config`: Qwen3.5-4B's heads, linear heads,
    // convolution, layer pattern, rotary embedding, epsilon and vocabulary at width 4096 with an
    // MLP of 12,288 and no tied head.
    assert_eq!(
        MOSS_TEXT_ENCODER,
        Qwen35Config {
            hidden: ENCODER_HIDDEN,
            ff: ENCODER_FF,
            tied_head: false,
            ..QWEN3_5_4B
        }
    );
}

#[test]
fn the_schedule_is_the_references_bit_for_bit() {
    let s = Schedule::new(STEPS, 5.0);
    assert_eq!(s.sigmas, SIGMAS, "sigmas");
    assert_eq!(s.timesteps, TIMESTEPS, "timesteps");
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights --model moss-image-base, then --features tt-models/weights"
)]
fn the_vae_decodes_a_latent_as_the_reference_does() {
    let (vae, loaded) =
        Vae::<Flex>::pretrained(&FLUX2_VAE, moss_image::WEIGHTS, &FlexDevice).unwrap();
    assert!(loaded.ignored.iter().all(|n| n.starts_with("encoder.")
        || n.starts_with("quant_conv.")
        || n == "bn.num_batches_tracked"));
    let latent = tensor(synth(128 * 16 * 16, 0.91, 2.17, 0.123), [1, 128, 16, 16]);
    let image = vae.decode(vae.unpatch(latent));
    assert_eq!(image.dims(), [1, 3, 256, 256]);
    matches(&values(image), &DECODED_LATENT, 1e-3, "the decoded latent");
}

/// One test, so the parts are in memory one at a time: the encoder (then dropped), the
/// transformer (then dropped), the VAE.
#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights --model moss-image-base, then --features tt-models/weights"
)]
fn the_pipeline_reproduces_the_reference_part_by_part() {
    let (cond, uncond) = {
        let (encoder, loaded) =
            moss_image::text_encoder::<Flex>(Precision::F32, &FlexDevice).unwrap();
        assert!(
            loaded.ignored.iter().all(|n| n.starts_with("visual.")),
            "{:?}",
            loaded.ignored
        );
        let cond = encode(&encoder, &PROMPT_IDS, DType::F32);
        let uncond = encode(&encoder, &NEGATIVE_IDS, DType::F32);
        matches(
            &values(cond.clone()),
            &PROMPT_ENCODING,
            1e-3,
            "the prompt's encoding",
        );
        matches(
            &values(uncond.clone()),
            &NEGATIVE_ENCODING,
            1e-3,
            "the negative prompt's encoding",
        );
        (cond, uncond)
    };

    let latents = {
        let (dit, loaded) = moss_image::transformer::<Flex>(Precision::F32, &FlexDevice).unwrap();
        assert_eq!(loaded.ignored, ["rope.bands"]);
        let latent = tensor(synth(128 * 16 * 16, 0.91, 2.17, 0.123), [1, 128, 16, 16]);
        let context = tensor(
            synth(12 * 4096, 0.37, 1.61, 0.071)
                .into_iter()
                .map(|v| 0.5 * v)
                .collect(),
            [1, 12, 4096],
        );
        let step = dit.forward(latent.clone(), 750.0, context.clone(), &[]);
        matches(&values(step), &STEP, 1e-3, "one step");
        let skipped = dit.forward(latent, 750.0, context, &UNCOND_SKIP);
        matches(
            &values(skipped),
            &STEP_WITHOUT_22,
            1e-3,
            "one step without block 22",
        );

        let noise = tensor(synth(128 * 16 * 16, 1.13, 0.59, 0.287), [1, 128, 16, 16]);
        let schedule = Schedule::new(STEPS, 5.0);
        let latents = denoise(&dit, noise, cond, uncond, &schedule, GUIDANCE, &UNCOND_SKIP);
        matches(
            &values(latents.clone()),
            &DENOISED,
            1e-3,
            "the denoised latents",
        );
        latents
    };

    let (vae, _) = Vae::<Flex>::pretrained(&FLUX2_VAE, moss_image::WEIGHTS, &FlexDevice).unwrap();
    let image = vae.decode(vae.unpatch(latents));
    matches(&values(image), &IMAGE, 1e-3, "the image");
}
