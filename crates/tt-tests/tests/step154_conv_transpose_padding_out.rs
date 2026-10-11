//! `conv_transpose` with an output padding at or past `max(stride, dilation)`.
//!
//! Found by Burn's own backend tests (`burn-tt-conformance`:
//! `conv_transpose1d_padding_out_stride1_backward_shape`, stride 1 with `padding_out = 1`): the op
//! asserted PyTorch's limit, `padding_out < max(stride, dilation)`, which Burn does not impose: its
//! reference accepts any output padding, and its own test (a regression for a backward pass that
//! "did not account for the trailing `padding_out` cells") uses one the limit forbids. The cells
//! no input reaches are zero, plus the bias, and the backward pass drops `padding_out / stride` of
//! them (`conv_transpose2d_x_backward`), which is what makes the gradient `x`'s shape again.
//!
//! The oracle is Flex's, exactly: the data are small integers, so the TF32 operands and the sums are
//! exact, and a transposed convolution is a sum of products with no rounding to bound. It covers the
//! output, and `x`'s, the weight's and the bias's gradients, in one and two dimensions, over output
//! paddings below, at and past the old limit, strides and dilations. The negative control is the same
//! convolution with a different output padding, which has another shape and so cannot equal.

use burn::backend::Autodiff;
use burn::tensor::module::{conv_transpose1d, conv_transpose2d};
use burn::tensor::ops::ConvTransposeOptions;
use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_tests::burn_device::{with_device, Config};

fn small(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + seed * 3) % 9) as f32 - 4.0)
        .collect()
}

type Out = (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Output, and the gradients of its sum, of a 1-D transposed convolution.
fn run1d<B: burn::tensor::backend::AutodiffBackend>(
    device: &B::Device,
    batch: usize,
    channels: [usize; 2],
    kernel: usize,
    size: usize,
    o: [usize; 4], // stride, padding, padding_out, dilation
) -> Out {
    let x = Tensor::<B, 3>::from_data(
        TensorData::new(
            small(batch * channels[0] * size, 1),
            [batch, channels[0], size],
        ),
        device,
    )
    .require_grad();
    let w = Tensor::<B, 3>::from_data(
        TensorData::new(
            small(channels[0] * channels[1] * kernel, 2),
            [channels[0], channels[1], kernel],
        ),
        device,
    )
    .require_grad();
    let b = Tensor::<B, 1>::from_data(
        TensorData::new(small(channels[1], 3), [channels[1]]),
        device,
    )
    .require_grad();
    let y = conv_transpose1d(
        x.clone(),
        w.clone(),
        Some(b.clone()),
        ConvTransposeOptions::new([o[0]], [o[1]], [o[2]], [o[3]], 1),
    );
    let shape = y.dims().to_vec();
    let values = y.clone().inner().into_data().to_vec::<f32>().unwrap();
    let grads = y.sum().backward();
    let get =
        |t: Option<Tensor<B::InnerBackend, 3>>| t.unwrap().into_data().to_vec::<f32>().unwrap();
    (
        shape,
        values,
        get(x.grad(&grads)),
        get(w.grad(&grads)),
        b.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap(),
    )
}

#[test]
fn output_padding_at_and_past_the_pytorch_limit_matches_flex_in_one_dimension() {
    with_device(Config::default(), |d| {
        type TT = Autodiff<TtBackend>;
        type FX = Autodiff<Flex>;
        // (stride, padding, padding_out, dilation): below, at and past `max(stride, dilation)`.
        let cases: [[usize; 4]; 9] = [
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 3, 1],
            [2, 0, 1, 1],
            [2, 1, 2, 1],
            [2, 0, 5, 1],
            [3, 1, 4, 2],
            [1, 1, 2, 3],
            [2, 2, 3, 2],
        ];
        for o in cases {
            let ours = run1d::<TT>(&d, 2, [2, 3], 3, 5, o);
            let want = run1d::<FX>(&FlexDevice, 2, [2, 3], 3, 5, o);
            assert_eq!(ours.0, want.0, "{o:?}: shape");
            assert_eq!(ours.1, want.1, "{o:?}: output");
            assert_eq!(ours.2, want.2, "{o:?}: x gradient");
            assert_eq!(ours.3, want.3, "{o:?}: weight gradient");
            assert_eq!(ours.4, want.4, "{o:?}: bias gradient");
        }
    });
}

#[test]
fn output_padding_past_the_limit_matches_flex_in_two_dimensions() {
    with_device(Config::default(), |d| {
        fn run2d<B: burn::tensor::backend::AutodiffBackend>(
            device: &B::Device,
            o: ConvTransposeOptions<2>,
        ) -> (Vec<usize>, Vec<f32>, Vec<f32>) {
            let x = Tensor::<B, 4>::from_data(
                TensorData::new(small(2 * 2 * 3 * 4, 1), [2, 2, 3, 4]),
                device,
            )
            .require_grad();
            let w = Tensor::<B, 4>::from_data(
                TensorData::new(small(2 * 2 * 2 * 3, 2), [2, 2, 2, 3]),
                device,
            );
            let y = conv_transpose2d(x.clone(), w, None, o);
            let shape = y.dims().to_vec();
            let values = y.clone().inner().into_data().to_vec::<f32>().unwrap();
            let grads = y.sum().backward();
            (
                shape,
                values,
                x.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap(),
            )
        }
        for (stride, padding, out, dilation) in [
            ([1, 1], [0, 0], [1, 2], [1, 1]),
            ([2, 1], [1, 0], [2, 1], [1, 1]),
            ([1, 2], [0, 1], [3, 5], [2, 1]),
        ] {
            let o = || ConvTransposeOptions::new(stride, padding, out, dilation, 1);
            let ours = run2d::<Autodiff<TtBackend>>(&d, o());
            let want = run2d::<Autodiff<Flex>>(&FlexDevice, o());
            assert_eq!(
                ours, want,
                "stride {stride:?} padding {padding:?} out {out:?} dilation {dilation:?}"
            );
        }
    });
}

/// The comparison tells output paddings apart: the same convolution with another one has another shape.
#[test]
fn negative_control_another_output_padding_is_another_result() {
    with_device(Config::default(), |d| {
        type TT = Autodiff<TtBackend>;
        let a = run1d::<TT>(&d, 1, [1, 1], 3, 4, [1, 0, 1, 1]);
        let b = run1d::<TT>(&d, 1, [1, 1], 3, 4, [1, 0, 2, 1]);
        assert_ne!(a.0, b.0);
        assert_ne!(a.1, b.1);
    });
}
