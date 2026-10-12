//! `conv3d` and `conv_transpose3d`, forward and all three gradients.
//!
//! Found by Burn's own backend tests (`burn-tt-conformance`: the `conv3d` and `conv_transpose3d`
//! groups, 20 tests). Both were refused by name, a deferral. No kernel is new: the host builds the
//! im2col patch and overlap-sum tables from shapes (`ops/conv.rs`, `Geometry<D>`, now generic over the
//! number of spatial dimensions), and the device runs the same gather, matmul and in-order fold the
//! 2-D operations use.
//!
//! The oracle is Flex's, exactly: the data are small integers, so the TF32 operands and every sum are
//! exact, and a convolution is a sum of products with no rounding to bound. The loss is `sum(y * m)`
//! with a non-uniform `m`, so the x and weight gradients are not tested against an all-ones upstream
//! gradient (which a transposed index would not disturb). Cases cover groups, stride, padding,
//! dilation, output padding below, at and past `max(stride, dilation)`, a batch of two, ragged sizes,
//! and more output positions than one 32-row patch chunk. The negative control mirrors the kernel along
//! one axis: the output keeps its shape and must change.

use burn::backend::Autodiff;
use burn::tensor::module::{conv3d, conv_transpose3d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_tests::burn_device::{with_device, Config};

type TT = Autodiff<TtBackend>;
type FX = Autodiff<Flex>;

fn small(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + seed * 3) % 9) as f32 - 4.0)
        .collect()
}

fn tensor<B: burn::tensor::backend::Backend>(
    device: &B::Device,
    shape: [usize; 5],
    seed: usize,
) -> Tensor<B, 5> {
    Tensor::from_data(
        TensorData::new(small(shape.iter().product(), seed), shape),
        device,
    )
}

/// Output, and the gradients of `sum(y * m)`: x, weight, bias.
type Out = (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn finish<B: burn::tensor::backend::AutodiffBackend>(
    y: Tensor<B, 5>,
    x: Tensor<B, 5>,
    w: Tensor<B, 5>,
    b: Tensor<B, 1>,
    device: &B::Device,
) -> Out {
    let shape = y.dims();
    let values = y.clone().inner().into_data().to_vec::<f32>().unwrap();
    let m = tensor::<B>(device, shape, 5);
    let grads = (y * m).sum().backward();
    let get = |t: Option<Tensor<B::InnerBackend, 5>>| t.unwrap().into_data().to_vec().unwrap();
    (
        shape.to_vec(),
        values,
        get(x.grad(&grads)),
        get(w.grad(&grads)),
        b.grad(&grads).unwrap().into_data().to_vec().unwrap(),
    )
}

#[derive(Clone, Copy, Debug)]
struct Conv {
    batch: usize,
    channels: [usize; 2], // input, output
    size: [usize; 3],
    kernel: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    padding_out: [usize; 3], // transposed only
    dilation: [usize; 3],
    groups: usize,
}

fn run<B: burn::tensor::backend::AutodiffBackend>(device: &B::Device, c: Conv) -> Out {
    let [cin, cout] = c.channels;
    let x = tensor::<B>(device, [c.batch, cin, c.size[0], c.size[1], c.size[2]], 1).require_grad();
    let w = tensor::<B>(
        device,
        [cout, cin / c.groups, c.kernel[0], c.kernel[1], c.kernel[2]],
        2,
    )
    .require_grad();
    let b =
        Tensor::<B, 1>::from_data(TensorData::new(small(cout, 3), [cout]), device).require_grad();
    let y = conv3d(
        x.clone(),
        w.clone(),
        Some(b.clone()),
        ConvOptions::new(c.stride, c.padding, c.dilation, c.groups),
    );
    finish(y, x, w, b, device)
}

fn run_transpose<B: burn::tensor::backend::AutodiffBackend>(device: &B::Device, c: Conv) -> Out {
    let [cin, cout] = c.channels;
    let x = tensor::<B>(device, [c.batch, cin, c.size[0], c.size[1], c.size[2]], 1).require_grad();
    let w = tensor::<B>(
        device,
        [cin, cout / c.groups, c.kernel[0], c.kernel[1], c.kernel[2]],
        2,
    )
    .require_grad();
    let b =
        Tensor::<B, 1>::from_data(TensorData::new(small(cout, 3), [cout]), device).require_grad();
    let y = conv_transpose3d(
        x.clone(),
        w.clone(),
        Some(b.clone()),
        ConvTransposeOptions::new(c.stride, c.padding, c.padding_out, c.dilation, c.groups),
    );
    finish(y, x, w, b, device)
}

fn same(ours: &Out, want: &Out, what: &str) {
    assert_eq!(ours.0, want.0, "{what}: shape");
    assert_eq!(ours.1, want.1, "{what}: output");
    assert_eq!(ours.2, want.2, "{what}: x gradient");
    assert_eq!(ours.3, want.3, "{what}: weight gradient");
    assert_eq!(ours.4, want.4, "{what}: bias gradient");
}

#[test]
fn conv3d_matches_flex() {
    with_device(Config::default(), |d| {
        let base = Conv {
            batch: 1,
            channels: [2, 3],
            size: [3, 3, 3],
            kernel: [1, 1, 1],
            stride: [1; 3],
            padding: [0; 3],
            padding_out: [0; 3],
            dilation: [1; 3],
            groups: 1,
        };
        let cases = [
            base,
            // Groups, a batch of two, per-axis stride, padding and dilation, ragged sizes.
            Conv {
                batch: 2,
                channels: [4, 6],
                size: [4, 5, 3],
                kernel: [2, 3, 2],
                stride: [1, 2, 1],
                padding: [1, 0, 1],
                dilation: [1, 1, 2],
                groups: 2,
                ..base
            },
            // More output positions than one 32-row chunk (64), with padding on every side.
            Conv {
                size: [4, 4, 4],
                kernel: [3, 3, 3],
                padding: [1; 3],
                ..base
            },
            // Dilation and stride on the leading axis only.
            Conv {
                size: [6, 3, 3],
                kernel: [2, 2, 2],
                stride: [2, 1, 1],
                dilation: [2, 1, 1],
                ..base
            },
        ];
        for c in cases {
            same(
                &run::<TT>(&d, c),
                &run::<FX>(&FlexDevice, c),
                &format!("{c:?}"),
            );
        }
    });
}

#[test]
fn conv_transpose3d_matches_flex() {
    with_device(Config::default(), |d| {
        let base = Conv {
            batch: 1,
            channels: [2, 3],
            size: [2, 2, 2],
            kernel: [2, 2, 2],
            stride: [1; 3],
            padding: [0; 3],
            padding_out: [0; 3],
            dilation: [1; 3],
            groups: 1,
        };
        let cases = [
            base,
            // Groups, a batch of two, per-axis stride, padding, dilation, and an output padding
            // past `max(stride, dilation)` on the last axis.
            Conv {
                batch: 2,
                channels: [4, 6],
                size: [3, 2, 3],
                stride: [2, 1, 2],
                padding: [1, 0, 0],
                padding_out: [1, 0, 3],
                dilation: [1, 2, 1],
                groups: 2,
                ..base
            },
            Conv {
                stride: [2; 3],
                padding_out: [1; 3],
                ..base
            },
            Conv {
                size: [3, 3, 2],
                kernel: [3, 1, 2],
                stride: [3, 2, 1],
                padding: [1, 0, 1],
                padding_out: [2, 1, 0],
                dilation: [2, 1, 1],
                ..base
            },
        ];
        for c in cases {
            same(
                &run_transpose::<TT>(&d, c),
                &run_transpose::<FX>(&FlexDevice, c),
                &format!("{c:?}"),
            );
        }
    });
}

/// BF16 inputs widen once and the result narrows once; with values in {-1, 0, 1} every sum is exact.
///
/// Silicon-only, like every BF16 narrowing gate: the result's F32 -> BF16 pack is PACR mode `0x105`,
/// which ttsim refuses (`ttsim-divergence.md`, "BF16 late narrowing").
#[cfg(feature = "silicon")]
#[test]
fn bf16_conv3d_matches_flex() {
    use burn::tensor::FloatDType;
    fn forward<B: burn::tensor::backend::Backend>(device: &B::Device) -> Vec<f32> {
        let tri = |n: usize, seed: usize| -> Vec<f32> {
            small(n, seed)
                .iter()
                .map(|v| (v % 3.0).clamp(-1.0, 1.0))
                .collect()
        };
        let x = Tensor::<B, 5>::from_data(
            TensorData::new(tri(2 * 4 * 4 * 4, 1), [1, 2, 4, 4, 4]),
            device,
        )
        .cast(FloatDType::BF16);
        let w =
            Tensor::<B, 5>::from_data(TensorData::new(tri(3 * 2 * 8, 2), [3, 2, 2, 2, 2]), device)
                .cast(FloatDType::BF16);
        conv3d(
            x,
            w,
            None,
            ConvOptions::new([1, 2, 1], [1, 0, 1], [1; 3], 1),
        )
        .cast(FloatDType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
    }
    with_device(Config::default(), |d| {
        let ours = forward::<TtBackend>(&d);
        assert!(ours.iter().any(|&v| v != 0.0));
        assert_eq!(ours, forward::<Flex>(&FlexDevice));
    });
}

/// The comparison tells a mirrored kernel axis apart: the output keeps its shape and must change.
#[test]
fn negative_control_a_mirrored_kernel_axis_is_another_result() {
    with_device(Config::default(), |d| {
        type B = TtBackend;
        let x = tensor::<B>(&d, [1, 2, 3, 3, 3], 1);
        let w = tensor::<B>(&d, [3, 2, 2, 2, 2], 2);
        let go = |w: Tensor<B, 5>| {
            conv3d(
                x.clone(),
                w,
                None,
                ConvOptions::new([1; 3], [0; 3], [1; 3], 1),
            )
            .into_data()
            .to_vec::<f32>()
            .unwrap()
        };
        let a = go(w.clone());
        let b = go(w.flip([4]));
        assert_eq!(a.len(), b.len());
        assert_ne!(a, b);
    });
}
