//! Where ResNet-18's forward goes: each of its op shapes, alone, on the card.
//!
//! `silicon_bench_models` showed 211 device jobs taking 16 s (ViT-Tiny's 709 take 0.8 s), so the
//! time was host work inside a few ops. This times the shapes ResNet-18 uses, one op at a time,
//! after a warm-up, with the weights resident, two ways: from a plain NCHW input with the result
//! read back (which, since convolution and max pooling run on NHWC tables, includes relaying the
//! input in and the output out, a datum a mover entry each), and as in the model, from an NHWC
//! view a previous convolution made, the result left on the card (to a `sync`). A measurement,
//! not a gate. Silicon only.

#![cfg(feature = "silicon")]

use std::time::Instant;

use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::{AdaptiveAvgPool2dConfig, MaxPool2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::{Distribution, Tensor};
use burn_tt::TtBackend;
use tt_tests::burn_device::{with_device, Config};

type B = TtBackend;

fn time<T>(label: &str, mut f: impl FnMut() -> T) {
    f();
    let mut ms = Vec::new();
    for _ in 0..3 {
        let t0 = Instant::now();
        let _ = f();
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("PROBE {label:<58} {:>10.1} ms (median of 3)", ms[1]);
}

#[test]
#[ignore = "a measurement, not a gate: cargo xtask silicon --include-ignored"]
fn resnet18_ops_one_at_a_time() {
    with_device(Config::default(), |d| {
        let conv = |inp, out, k, stride, pad| {
            Conv2dConfig::new([inp, out], [k, k])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
                .with_bias(false)
                .init::<B>(&d)
        };
        let input = |c, hw| Tensor::<B, 4>::random([1, c, hw, hw], Distribution::Default, &d);

        let x = input(3, 224);
        let c1 = conv(3, 64, 7, 2, 3);
        time("conv 7x7/2  3->64   on 224x224 (conv1)", || {
            c1.forward(x.clone()).into_data()
        });
        let x = input(64, 112);
        let pool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init();
        time("maxpool 3x3/2 on 64x112x112", || {
            pool.forward(x.clone()).into_data()
        });
        let x = input(64, 56);
        let c = conv(64, 64, 3, 1, 1);
        time("conv 3x3/1  64->64  on 56x56 (layer1)", || {
            c.forward(x.clone()).into_data()
        });
        let x = input(128, 28);
        let c = conv(128, 128, 3, 1, 1);
        time("conv 3x3/1  128->128 on 28x28 (layer2)", || {
            c.forward(x.clone()).into_data()
        });
        let x = input(256, 14);
        let c = conv(256, 256, 3, 1, 1);
        time("conv 3x3/1  256->256 on 14x14 (layer3)", || {
            c.forward(x.clone()).into_data()
        });
        let x = input(512, 7);
        let c = conv(512, 512, 3, 1, 1);
        time("conv 3x3/1  512->512 on 7x7 (layer4)", || {
            c.forward(x.clone()).into_data()
        });
        let gap = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        time("adaptive avg pool 512x7x7 -> 1x1", || {
            gap.forward(x.clone()).into_data()
        });
        let y = input(64, 56);
        time("elementwise add + relu 64x56x56", || {
            burn::tensor::activation::relu(y.clone() + y.clone()).into_data()
        });

        // As in the model: the input an NHWC view a 1x1 convolution made (outside the timing),
        // the output left on the card.
        let view = |c, hw| conv(c, c, 1, 1, 0).forward(input(c, hw));
        let sync = || <B as burn::tensor::backend::Backend>::sync(&d).unwrap();
        let x = view(64, 112);
        time(
            "maxpool 3x3/2 on 64x112x112, NHWC view in, on the card",
            || {
                let y = pool.forward(x.clone());
                sync();
                y
            },
        );
        for (ch, hw) in [(64, 56), (128, 28), (256, 14), (512, 7)] {
            let x = view(ch, hw);
            let c = conv(ch, ch, 3, 1, 1);
            time(
                &format!("conv 3x3/1  {ch}->{ch} on {hw}x{hw}, NHWC view in, on the card"),
                || {
                    let y = c.forward(x.clone());
                    sync();
                    y
                },
            );
        }
        let y = view(64, 56);
        time("add + relu 64x56x56, NHWC views, on the card", || {
            let z = burn::tensor::activation::relu(y.clone() + y.clone());
            sync();
            z
        });
    });
}
