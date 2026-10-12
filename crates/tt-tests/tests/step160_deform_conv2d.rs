//! `deform_conv2d`, forward and the gradients of x, offset, weight, mask and bias.
//!
//! Found by Burn's own backend tests (`burn-tt-conformance`: the `deform_conv2d` group, 17 tests),
//! refused by name until now. No kernel is new (`ops/deform_conv.rs`): sample positions, bilinear
//! corners and the zero row for corners off the plane are device arithmetic on the offsets; the corner
//! values are four resident gathers; the product with the weights, the bias and the weight gradient
//! are the 2-D convolution's own; the x gradient is a one-hot matrix times the column gradient.
//!
//! The oracle is Flex's, exactly. The vendored tests are loose (`relative(0.5)` on every gradient), so
//! passing them does not show the arithmetic is right; this gate does. The data are chosen so that
//! every product and sum is exact in f32 and survives the Matrix Unit: x, weights and the loss weights
//! are small integers, offsets are multiples of 1/4 (so the bilinear weights are multiples of 1/16),
//! masks multiples of 1/4, and the largest product stays below the group adder's quantum for these
//! granularities (`step158_matmul_accumulation`). With nothing to round there is no tolerance to
//! derive: the results must be equal, in a different summation order from Flex's.
//!
//! The offsets run from -2 to 2 against base positions from -pad to the far edge, so samples fall on
//! integer positions, between pixels, exactly on the `-1` and `H` borders, and well off the plane.
//! The loss is `sum(y * m)` with a non-uniform `m`, so no gradient is tested only against all ones.
//!
//! Negative controls change an input of ours so that it computes a specific wrong function (the
//! offset's h and w channels swapped; the mask dropped; every offset shifted a quarter pixel) and
//! require the comparison to fail; each case also asserts that its gradients are not all zero, so an
//! equality cannot hold vacuously.

use burn::backend::Autodiff;
use burn::tensor::module::deform_conv2d;
use burn::tensor::ops::DeformConvOptions;
use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_tests::burn_device::{with_device, Config};

type TT = Autodiff<TtBackend>;
type FX = Autodiff<Flex>;

fn integers(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + seed * 3) % 5) as f32 - 2.0)
        .collect()
}

/// Multiples of 1/4 in [-2, 2].
fn quarters(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 5 + seed) % 17) as f32 / 4.0 - 2.0)
        .collect()
}

/// {1/4, 1/2, 3/4, 1}.
fn masks(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 3 + seed) % 4 + 1) as f32 / 4.0)
        .collect()
}

fn loss_weights(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + 5) % 3) as f32 - 1.0).collect()
}

#[derive(Clone, Copy, Debug)]
enum Offsets {
    Grid,
    /// Every offset the same.
    Constant(f32),
}

#[derive(Clone, Copy, Debug)]
struct Case {
    batch: usize,
    channels: [usize; 2], // input, output
    size: [usize; 2],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    weight_groups: usize,
    offset_groups: usize,
    mask: bool,
    bias: bool,
    offsets: Offsets,
}

impl Case {
    fn out(&self) -> [usize; 2] {
        let one = |a: usize| {
            (self.size[a] + 2 * self.padding[a] - self.dilation[a] * (self.kernel[a] - 1) - 1)
                / self.stride[a]
                + 1
        };
        [one(0), one(1)]
    }
    fn offset_shape(&self) -> [usize; 4] {
        let [oh, ow] = self.out();
        [
            self.batch,
            self.offset_groups * self.kernel[0] * self.kernel[1] * 2,
            oh,
            ow,
        ]
    }
    fn mask_shape(&self) -> [usize; 4] {
        let mut s = self.offset_shape();
        s[1] /= 2;
        s
    }
}

/// Everything one run reports.
#[derive(Debug)]
struct Out {
    shape: Vec<usize>,
    y: Vec<f32>,
    x_grad: Vec<f32>,
    offset_grad: Vec<f32>,
    weight_grad: Vec<f32>,
    mask_grad: Option<Vec<f32>>,
    bias_grad: Option<Vec<f32>>,
}

/// How a run's inputs differ from the case's (only ours is ever changed, by a control).
#[derive(Clone, Copy, PartialEq)]
enum Tweak {
    None,
    /// Swap the h and w channel of every offset pair.
    SwapAxes,
    /// Add a quarter pixel to every offset.
    Shift,
    /// Drop the mask.
    NoMask,
}

fn data<B: burn::tensor::backend::Backend, const D: usize>(
    device: &B::Device,
    shape: [usize; D],
    values: Vec<f32>,
) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

fn run<B: burn::tensor::backend::AutodiffBackend>(
    device: &B::Device,
    c: Case,
    x_values: Option<Vec<f32>>,
    tweak: Tweak,
) -> Out {
    let [cin, cout] = c.channels;
    let x_shape = [c.batch, cin, c.size[0], c.size[1]];
    let w_shape = [cout, cin / c.weight_groups, c.kernel[0], c.kernel[1]];
    let n_off = c.offset_shape().iter().product();
    let mut offsets = match c.offsets {
        Offsets::Grid => quarters(n_off, 1),
        Offsets::Constant(v) => vec![v; n_off],
    };
    match tweak {
        Tweak::SwapAxes => {
            let plane = c.offset_shape()[2] * c.offset_shape()[3];
            let per_batch = n_off / c.batch;
            for chunk in offsets.chunks_mut(per_batch) {
                for pair in chunk.chunks_mut(2 * plane) {
                    let (h, w) = pair.split_at_mut(plane);
                    h.swap_with_slice(w);
                }
            }
        }
        Tweak::Shift => offsets.iter_mut().for_each(|v| *v += 0.25),
        Tweak::None | Tweak::NoMask => {}
    }
    let x = data::<B, 4>(
        device,
        x_shape,
        x_values.unwrap_or_else(|| integers(x_shape.iter().product(), 1)),
    )
    .require_grad();
    let offset = data::<B, 4>(device, c.offset_shape(), offsets).require_grad();
    let w = data::<B, 4>(device, w_shape, integers(w_shape.iter().product(), 2)).require_grad();
    let mask = (c.mask && tweak != Tweak::NoMask).then(|| {
        data::<B, 4>(
            device,
            c.mask_shape(),
            masks(c.mask_shape().iter().product(), 1),
        )
        .require_grad()
    });
    let bias = c
        .bias
        .then(|| data::<B, 1>(device, [cout], integers(cout, 3)).require_grad());
    let y = deform_conv2d(
        x.clone(),
        offset.clone(),
        w.clone(),
        mask.clone(),
        bias.clone(),
        DeformConvOptions::new(
            c.stride,
            c.padding,
            c.dilation,
            c.weight_groups,
            c.offset_groups,
        ),
    );
    let shape = y.dims();
    let values = y.clone().inner().into_data().to_vec::<f32>().unwrap();
    let m = data::<B, 4>(device, shape, loss_weights(shape.iter().product()));
    let grads = (y * m).sum().backward();
    let get = |g: Option<Tensor<B::InnerBackend, 4>>| g.unwrap().into_data().to_vec().unwrap();
    Out {
        shape: shape.to_vec(),
        y: values,
        x_grad: get(x.grad(&grads)),
        offset_grad: get(offset.grad(&grads)),
        weight_grad: get(w.grad(&grads)),
        mask_grad: mask.map(|t| get(t.grad(&grads))),
        bias_grad: bias.map(|t| t.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap()),
    }
}

/// Equal as numbers, NaN equal to NaN.
fn equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
}

fn differences(ours: &Out, want: &Out) -> Vec<&'static str> {
    let optional = |a: &Option<Vec<f32>>, b: &Option<Vec<f32>>| match (a, b) {
        (Some(a), Some(b)) => equal(a, b),
        (None, None) => true,
        _ => false,
    };
    [
        ("shape", ours.shape == want.shape),
        ("output", equal(&ours.y, &want.y)),
        ("x gradient", equal(&ours.x_grad, &want.x_grad)),
        (
            "offset gradient",
            equal(&ours.offset_grad, &want.offset_grad),
        ),
        (
            "weight gradient",
            equal(&ours.weight_grad, &want.weight_grad),
        ),
        ("mask gradient", optional(&ours.mask_grad, &want.mask_grad)),
        ("bias gradient", optional(&ours.bias_grad, &want.bias_grad)),
    ]
    .into_iter()
    .filter(|&(_, ok)| !ok)
    .map(|(name, _)| name)
    .collect()
}

fn same(ours: &Out, want: &Out, what: &str) {
    let found = differences(ours, want);
    assert!(found.is_empty(), "{what}: differs in {found:?}");
}

/// The comparison cannot be vacuous: the reference has something to disagree about.
fn assert_live(want: &Out, what: &str) {
    let live = |v: &[f32]| v.iter().any(|&x| x != 0.0);
    assert!(live(&want.y), "{what}: output is all zero");
    assert!(live(&want.x_grad), "{what}: x gradient is all zero");
    assert!(
        live(&want.weight_grad),
        "{what}: weight gradient is all zero"
    );
    if let Some(m) = &want.mask_grad {
        assert!(live(m), "{what}: mask gradient is all zero");
    }
}

fn base() -> Case {
    Case {
        batch: 1,
        channels: [2, 3],
        size: [4, 4],
        kernel: [3, 3],
        stride: [1; 2],
        padding: [0; 2],
        dilation: [1; 2],
        weight_groups: 1,
        offset_groups: 1,
        mask: true,
        bias: true,
        offsets: Offsets::Grid,
    }
}

#[test]
fn deform_conv2d_matches_flex() {
    with_device(Config::default(), |d| {
        let b = base();
        let cases = [
            b,
            // Weight groups and offset groups that differ, a batch of two, ragged extents.
            Case {
                batch: 2,
                channels: [4, 4],
                size: [5, 4],
                kernel: [2, 3],
                weight_groups: 2,
                offset_groups: 4,
                ..b
            },
            // Offset groups finer than weight groups; the other way round.
            Case {
                channels: [4, 2],
                weight_groups: 1,
                offset_groups: 2,
                ..b
            },
            Case {
                channels: [4, 4],
                weight_groups: 4,
                offset_groups: 1,
                ..b
            },
            // Per-axis stride, padding and dilation, and more output positions than one 32-row chunk.
            Case {
                batch: 2,
                size: [6, 7],
                kernel: [2, 3],
                stride: [2, 1],
                padding: [1, 2],
                dilation: [2, 1],
                ..b
            },
            // No mask, no bias.
            Case {
                mask: false,
                bias: false,
                ..b
            },
            Case {
                padding: [2, 3],
                ..b
            },
        ];
        for c in cases {
            let want = run::<FX>(&FlexDevice, c, None, Tweak::None);
            assert_live(&want, &format!("{c:?}"));
            same(
                &run::<TT>(&d, c, None, Tweak::None),
                &want,
                &format!("{c:?}"),
            );
        }
    });
}

/// Samples exactly on the borders, on integer positions, and just inside: Flex's rule is that a
/// coordinate must be above -1 and below the extent, and the corners are then clipped to the plane.
#[test]
fn border_samples_match_flex() {
    with_device(Config::default(), |d| {
        let b = Case {
            padding: [1; 2],
            ..base()
        };
        for off in [-1.0, -0.75, -0.25, 0.0, 0.5, 1.0, 2.0, 3.0, 3.25, 4.0, 5.0] {
            let c = Case {
                offsets: Offsets::Constant(off),
                ..b
            };
            let want = run::<FX>(&FlexDevice, c, None, Tweak::None);
            same(
                &run::<TT>(&d, c, None, Tweak::None),
                &want,
                &format!("constant offset {off}"),
            );
        }
    });
}

/// A NaN or an infinity in x reaches the samples that read it, and no others: a sample off the
/// plane is zero however large its neighbours, and an invalid corner reads zero, not the pixel the
/// index would have named. Forward only is compared (the gradients of a poisoned input are not defined).
///
/// In `MatmulMode::Exact` the products are the SFPU's, so the result must equal Flex's NaN for NaN:
/// that isolates the gathers and the bilinear arithmetic. In the default mode the Matrix Unit's
/// unpack turns a NaN into an infinity and `inf * 0` into zero (silicon-operating-notes, the 17-pair
/// special corpus), so the poisoned cells take other non-finite values; what must still hold is that
/// every cell Flex leaves finite is equal, so the poison went no further.
#[test]
fn non_finite_x_reaches_only_its_samples() {
    with_device(Config::default(), |d| {
        let c = Case {
            padding: [1; 2],
            mask: false,
            bias: false,
            ..base()
        };
        let n: usize = [1, 2, 4, 4].iter().product();
        for (name, value) in [("NaN", f32::NAN), ("+inf", f32::INFINITY)] {
            let mut x = integers(n, 1);
            x[5] = value; // channel 0, row 1, column 1
            let want = run::<FX>(&FlexDevice, c, Some(x.clone()), Tweak::None);
            assert!(
                want.y.iter().any(|v| !v.is_finite()),
                "{name}: reaches no sample"
            );
            assert!(
                want.y.iter().any(|v| v.is_finite()),
                "{name}: reaches every sample"
            );

            let exact = burn_tt::exact_matmul(|| run::<TT>(&d, c, Some(x.clone()), Tweak::None));
            assert!(
                equal(&exact.y, &want.y),
                "{name}, exact: output differs\nours {:?}\nflex {:?}",
                exact.y,
                want.y
            );

            let fast = run::<TT>(&d, c, Some(x), Tweak::None);
            for (i, (a, b)) in fast.y.iter().zip(&want.y).enumerate() {
                assert!(
                    !b.is_finite() || a == b,
                    "{name}, fast: cell {i} is {a}, Flex's finite {b}"
                );
            }
            assert!(
                fast.y.iter().any(|v| !v.is_finite()),
                "{name}, fast: nothing poisoned"
            );
        }
    });
}

#[test]
fn negative_controls_are_other_results() {
    with_device(Config::default(), |d| {
        let c = Case {
            batch: 2,
            padding: [1; 2],
            ..base()
        };
        let want = run::<FX>(&FlexDevice, c, None, Tweak::None);
        assert_live(&want, "control case");
        same(
            &run::<TT>(&d, c, None, Tweak::None),
            &want,
            "control case itself",
        );
        for (name, tweak) in [
            ("swapped h and w offsets", Tweak::SwapAxes),
            ("every offset shifted a quarter pixel", Tweak::Shift),
            ("mask dropped", Tweak::NoMask),
        ] {
            // The numbers, not which gradients exist: dropping the mask removes one.
            let wrong = run::<TT>(&d, c, None, tweak);
            assert!(
                !equal(&wrong.y, &want.y),
                "{name}: the output is the real one"
            );
            assert!(
                !equal(&wrong.x_grad, &want.x_grad)
                    || !equal(&wrong.offset_grad, &want.offset_grad),
                "{name}: the gradients are the real ones"
            );
        }
    });
}

/// BF16 inputs widen once and the result narrows once. With x, weights in {-1, 0, 1} and offsets in
/// {-1, -1/2, 0, 1/2, 1}, every bilinear weight is a multiple of 1/4 and every sum has at most seven
/// significant bits, so BF16 holds the result exactly and Flex's F32 run of the same values is the oracle.
///
/// Silicon-only, like every BF16 narrowing gate: the result's F32 -> BF16 pack is PACR mode `0x105`,
/// which ttsim refuses (`ttsim-divergence.md`, "BF16 late narrowing").
#[cfg(feature = "silicon")]
#[test]
fn bf16_deform_conv2d_matches_flex() {
    use burn::tensor::FloatDType;
    let c = Case {
        padding: [1; 2],
        mask: false,
        bias: false,
        ..base()
    };
    let tri = |n: usize, seed: usize| -> Vec<f32> {
        integers(n, seed)
            .iter()
            .map(|v| v.clamp(-1.0, 1.0))
            .collect()
    };
    let halves = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 5 + 1) % 5) as f32 / 2.0 - 1.0)
            .collect()
    };
    fn forward<B: burn::tensor::backend::Backend>(
        device: &B::Device,
        c: Case,
        x: &[f32],
        w: &[f32],
        offset: &[f32],
        bf16: bool,
    ) -> Vec<f32> {
        let cast = |t: Tensor<B, 4>| if bf16 { t.cast(FloatDType::BF16) } else { t };
        let x = cast(data::<B, 4>(device, [1, 2, 4, 4], x.to_vec()));
        let w = cast(data::<B, 4>(device, [3, 2, 3, 3], w.to_vec()));
        let offset = cast(data::<B, 4>(device, c.offset_shape(), offset.to_vec()));
        deform_conv2d(
            x,
            offset,
            w,
            None,
            None,
            DeformConvOptions::new(c.stride, c.padding, c.dilation, 1, 1),
        )
        .cast(FloatDType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
    }
    let x = tri(2 * 4 * 4, 1);
    let w = tri(3 * 2 * 9, 2);
    let offset = halves(c.offset_shape().iter().product());
    with_device(Config::default(), |d| {
        let ours = forward::<TtBackend>(&d, c, &x, &w, &offset, true);
        assert!(ours.iter().any(|&v| v != 0.0));
        assert_eq!(
            ours,
            forward::<Flex>(&FlexDevice, c, &x, &w, &offset, false)
        );
    });
}
