//! Probe: which of MOSS-Image's constructs the card refuses. `step262_moss_image` fails with
//! "block copy ... descriptor refused before submission (code 2)"; each stage here runs in its own
//! session on small tensors, and prints whether it completes (and, where it is deterministic,
//! whether it matches Flex bit for bit).

#![cfg(feature = "silicon")]

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{TtBackend, TtDevice};
use tt_models::moss_image::dit::{Dit, DitConfig, MOSS_IMAGE_BASE};
use tt_models::moss_image::vae::{upsample_nearest, Vae, VaeConfig, FLUX2_VAE};
use tt_tests::burn_device::{with_device, Config};

fn ramp<B: Backend, const D: usize>(shape: [usize; D], d: &B::Device) -> Tensor<B, D> {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|i| ((i * 37) % 101) as f32 / 50.0 - 1.0)
        .collect();
    Tensor::from_data(TensorData::new(v, shape), d)
}

fn upsample<B: Backend>(d: &B::Device) -> Tensor<B, 4> {
    upsample_nearest(ramp::<B, 4>([1, 64, 16, 16], d))
}

fn unfold<B: Backend>(d: &B::Device) -> Tensor<B, 4> {
    let (n, c, h, w) = (1, 128, 16, 16);
    ramp::<B, 4>([n, c, h, w], d)
        .reshape([n, c / 4, 2, 2, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([n, c / 4, 2 * h, 2 * w])
}

const TINY_DIT: DitConfig = DitConfig {
    dim: 64,
    ffn_dim: 96,
    heads: 2,
    layers: 2,
    text_dim: 48,
    freq_dim: 16,
    latent_channels: 128,
    rope_base: 10000.0,
};

const TINY_VAE: VaeConfig = VaeConfig {
    latent_channels: 32,
    block_out_channels: [32, 32, 64, 64],
    layers_per_block: 1,
    groups: 8,
    out_channels: 3,
    batch_norm_eps: 1e-4,
};

fn stage(name: &str, run: impl Fn(&TtDevice) -> Vec<f32> + Send, flex: Option<Vec<f32>>) {
    with_device(Config::default(), |d: TtDevice| {
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&d)));
        match (got, &flex) {
            (Ok(v), Some(want)) => eprintln!(
                "{name}: completes, {} flex",
                if &v == want {
                    "bit for bit"
                } else {
                    "differs from"
                }
            ),
            (Ok(_), None) => eprintln!("{name}: completes"),
            (Err(e), _) => eprintln!(
                "{name}: FAILS: {}",
                e.downcast_ref::<String>().cloned().unwrap_or_default()
            ),
        }
    });
}

fn values<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn which_construct_the_card_refuses() {
    stage(
        "nearest upsampling (6-D broadcast)",
        |d| values(upsample::<TtBackend>(d)),
        Some(values(upsample::<Flex>(&FlexDevice))),
    );
    stage(
        "the 2 x 2 patch unfold (6-D permute)",
        |d| values(unfold::<TtBackend>(d)),
        Some(values(unfold::<Flex>(&FlexDevice))),
    );
    stage(
        "a tiny DiT forward (8 x 8 latent, 5 text tokens)",
        |d| {
            let dit = Dit::<TtBackend>::new(&TINY_DIT, d);
            values(dit.forward(ramp([1, 128, 8, 8], d), 500.0, ramp([1, 5, 48], d), &[1]))
        },
        None,
    );
    stage(
        "a tiny VAE decode (16 x 16 latent tokens)",
        |d| {
            let vae = Vae::<TtBackend>::new(&TINY_VAE, d);
            values(vae.decode(vae.unpatch(ramp([1, 128, 8, 8], d))))
        },
        None,
    );
}

/// The same at the real widths, random weights: one DiT block at width 3072 over 256 image and 12
/// text tokens, and the real VAE decoder to a 256 x 256 image.
#[test]
fn which_construct_the_card_refuses_at_full_width() {
    stage(
        "a one-block DiT at width 3072 (16 x 16 latent, 12 text tokens)",
        |d| {
            let cfg = DitConfig {
                layers: 1,
                ..MOSS_IMAGE_BASE
            };
            let dit = Dit::<TtBackend>::new(&cfg, d);
            values(dit.forward(
                ramp([1, 128, 16, 16], d),
                750.0,
                ramp([1, 12, 4096], d),
                &[],
            ))
        },
        None,
    );
    stage(
        "the FLUX.2 decoder to 256 x 256",
        |d| {
            let vae = Vae::<TtBackend>::new(&FLUX2_VAE, d);
            values(vae.decode(vae.unpatch(ramp([1, 128, 16, 16], d))))
        },
        None,
    );
}

/// The decoder's operations one at a time at their real shapes.
#[test]
fn which_decoder_operation_the_card_refuses() {
    use burn::nn::conv::Conv2dConfig;
    use burn::nn::{GroupNormConfig, PaddingConfig2d};
    use burn::tensor::module::attention;
    use burn::tensor::ops::AttentionModuleOptions;
    for (c, side) in [(384, 32), (384, 64), (384, 128), (192, 128)] {
        stage(
            &format!("nearest upsampling of [1, {c}, {side}, {side}]"),
            move |d| {
                values(upsample_nearest(ramp::<TtBackend, 4>(
                    [1, c, side, side],
                    d,
                )))
            },
            Some(values(upsample_nearest(ramp::<Flex, 4>(
                [1, c, side, side],
                &FlexDevice,
            )))),
        );
    }
    for (c, side) in [(384, 32), (384, 256), (96, 256)] {
        stage(
            &format!("group norm (32 groups) of [1, {c}, {side}, {side}]"),
            move |d| {
                let n = GroupNormConfig::new(32, c)
                    .with_epsilon(1e-6)
                    .init::<TtBackend>(d);
                values(n.forward(ramp([1, c, side, side], d)))
            },
            None,
        );
        stage(
            &format!("a 3 x 3 convolution of [1, {c}, {side}, {side}]"),
            move |d| {
                let k = Conv2dConfig::new([c, c], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                    .init::<TtBackend>(d);
                values(k.forward(ramp([1, c, side, side], d)))
            },
            None,
        );
    }
    stage(
        "single-head attention over 1,024 pixels of 384",
        |d| {
            let x = ramp::<TtBackend, 4>([1, 1, 1024, 384], d);
            let opts = AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: false,
            };
            values(attention(x.clone(), x.clone(), x, None, None, opts))
        },
        None,
    );
    stage(
        "the pixels as rows: [1, 384, 32, 32] -> [1, 1024, 384] and back",
        |d| {
            let x = ramp::<TtBackend, 4>([1, 384, 32, 32], d);
            values(
                x.reshape([1, 384, 1024])
                    .swap_dims(1, 2)
                    .swap_dims(1, 2)
                    .reshape([1, 384, 32, 32]),
            )
        },
        Some(values(ramp::<Flex, 4>([1, 384, 32, 32], &FlexDevice))),
    );
}
