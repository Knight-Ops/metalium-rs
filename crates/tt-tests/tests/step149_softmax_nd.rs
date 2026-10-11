//! `softmax`, `log_softmax` and `softmin` of any rank and any axis, on the device.
//!
//! Found by Burn's own backend tests (`burn-tt-conformance`: `softmax::test_softmax_d1`,
//! `_d3_last_axis`, `_non_last_axis_d3`, `_non_last_axis_d4_dim0`, `_non_contiguous_input`, and the CTC
//! loss's `log_softmax` over `[T, N, C]`): the device composition took a plain rank-2 matrix and
//! refused everything else, though an attention score is `[batch, heads, t, t]`. The axis is now moved
//! last and the rest flattened into rows, so the *same* composition runs (`step32_burn_softmax`), and
//! the result is put back; each step is a device op, a view where the strides allow.
//!
//! The claim is the one `step32` makes, with `n` the length of the axis the softmax runs over: the
//! device computes Burn's composition, Flex a fused one, so the distance between them is a derived
//! bound -- the SFPU's `exp` twice, the sum's order, the division's ulp, and Flex's own counterparts
//! (see that file's header). Beyond it:
//!
//! * nothing is uploaded or downloaded between the input being resident and the result being read;
//! * the results are computed on the device (`computed_on_device`, which a cached host copy cannot fake);
//! * negative control: the same tensors softmaxed over the *wrong* axis, compared with the same bound,
//!   must break it, so the bound can tell a moved axis from a correct one.

use burn::tensor::backend::Backend;
use burn::tensor::{activation, Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend, TtDevice};
use tt_kernels::sfpu::ops::{EXP_BOUND, LOG_BOUND};
use tt_tests::burn_device::{with_device, Config};

const U: f64 = 1.0 / 16_777_216.0;

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 20.0
        })
        .collect()
}

fn floats<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

fn resident<const D: usize>(t: &Tensor<TtBackend, D>) -> bool {
    match t.clone().into_primitive() {
        TensorPrimitive::Float(p) => p.computed_on_device(),
        _ => false,
    }
}

/// The softmax's relative bound, for an axis of length `n` (`step32`'s).
fn softmax_bound(n: f64) -> f64 {
    2.0 * EXP_BOUND + (n - 1.0) * U + 2.0 * U // ours
        + 2.0 * 2.0 * U + (n - 1.0) * U + 2.0 * U // Flex's
}

/// The log-softmax's absolute bound for one element whose Flex value is `w` (`step32`'s).
fn log_softmax_bound(n: f64, w: f32) -> f64 {
    LOG_BOUND * n.ln().max(1.0)
        + 2.0 * ((n - 1.0) * U + EXP_BOUND)
        + 2.0 * U * (w as f64).abs().max(1.0) * 2.0
}

fn within_softmax(ours: &[f32], flex: &[f32], bound: f64) -> bool {
    ours.iter().zip(flex).all(|(s, w)| {
        let rel = (*s as f64 - *w as f64).abs() / (*w as f64).abs().max(f32::MIN_POSITIVE as f64);
        rel <= bound || (*s == 0.0 && w.abs() < f32::MIN_POSITIVE * 4.0)
    })
}

fn within_log_softmax(ours: &[f32], flex: &[f32], n: f64) -> bool {
    ours.iter()
        .zip(flex)
        .all(|(s, w)| (*s as f64 - *w as f64).abs() <= log_softmax_bound(n, *w))
}

/// Softmax, log-softmax and softmin of a `shape` tensor over `dim`, on the device with no host
/// traffic, within the bound of Flex's.
fn check<const D: usize>(device: &TtDevice, shape: [usize; D], dim: usize) {
    let n = shape[dim] as f64;
    let count: usize = shape.iter().product();
    let v = values(count as u64 + dim as u64, count);
    let t = Tensor::<TtBackend, D>::from_data(TensorData::new(v.clone(), shape), device)
        .to_device(device);
    let f = Tensor::<Flex, D>::from_data(TensorData::new(v, shape), &FlexDevice);

    let before = tensor_traffic();
    let soft = activation::softmax(t.clone(), dim);
    let logsoft = activation::log_softmax(t.clone(), dim);
    let softmin = activation::softmin(t.clone(), dim);
    let during = tensor_traffic() - before;
    assert_eq!(
        (during.uploads, during.downloads),
        (0, 0),
        "{shape:?} dim {dim}: {during:?}"
    );
    for (name, x) in [
        ("softmax", &soft),
        ("log_softmax", &logsoft),
        ("softmin", &softmin),
    ] {
        assert!(resident(x), "{shape:?} dim {dim}: {name} ran on the host");
        assert_eq!(x.dims(), shape, "{name} changed the shape");
    }

    let bound = softmax_bound(n);
    let flex_soft = floats(activation::softmax(f.clone(), dim));
    assert!(
        within_softmax(&floats(soft), &flex_soft, bound),
        "{shape:?} dim {dim}: softmax is outside {bound:e} of Flex's"
    );
    assert!(
        within_softmax(
            &floats(softmin),
            &floats(activation::softmin(f.clone(), dim)),
            bound
        ),
        "{shape:?} dim {dim}: softmin is outside {bound:e} of Flex's"
    );
    assert!(
        within_log_softmax(
            &floats(logsoft),
            &floats(activation::log_softmax(f, dim)),
            n
        ),
        "{shape:?} dim {dim}: log_softmax is outside its bound of Flex's"
    );
}

#[test]
fn rank_one_three_and_four_softmax_on_every_axis_matches_flex_on_the_device() {
    with_device(Config::default(), |d| {
        check(&d, [37], 0);
        // A rank-3 tensor, the axes the CTC loss and sequence models use.
        for dim in 0..3 {
            check(&d, [2, 3, 70], dim);
        }
        // An attention score `[batch, heads, t, t]`, ragged against the 32-wide tile.
        for dim in 0..4 {
            check(&d, [2, 2, 5, 33], dim);
        }
        // Rows and columns that are whole tiles, and a unit axis.
        check(&d, [2, 1, 32, 64], 3);
        check(&d, [3, 1, 4], 1);
    });
}

/// A transposed matrix is not a plain one: the axis is the same, the strides are not.
#[test]
fn a_transposed_matrix_softmaxes_like_its_values() {
    with_device(Config::default(), |d| {
        let (r, c) = (6usize, 40usize);
        let v = values(9, r * c);
        let t = Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), [r, c]), &d)
            .to_device(&d)
            .transpose();
        let f = Tensor::<Flex, 2>::from_data(TensorData::new(v, [r, c]), &FlexDevice).transpose();
        for dim in [0usize, 1] {
            let n = if dim == 0 { c } else { r } as f64;
            let before = tensor_traffic();
            let soft = activation::softmax(t.clone(), dim);
            let during = tensor_traffic() - before;
            assert_eq!(
                (during.uploads, during.downloads),
                (0, 0),
                "dim {dim}: {during:?}"
            );
            assert!(resident(&soft), "dim {dim}: softmax ran on the host");
            assert!(
                within_softmax(
                    &floats(soft),
                    &floats(activation::softmax(f.clone(), dim)),
                    softmax_bound(n)
                ),
                "transposed dim {dim}"
            );
        }
    });
}

/// The bound is not so loose that a moved axis passes it: the device's softmax over axis `dim` set
/// against Flex's over another axis of the same tensor must break the bound. If the implementation
/// moved the axis and did not move it back, or took the wrong one, the checks above would be the
/// ones to fail; this shows they can.
#[test]
fn negative_control_the_wrong_axis_breaks_the_bound() {
    with_device(Config::default(), |d| {
        let shape = [2usize, 3, 70];
        let count: usize = shape.iter().product();
        let v = values(5, count);
        let t =
            Tensor::<TtBackend, 3>::from_data(TensorData::new(v.clone(), shape), &d).to_device(&d);
        let f = Tensor::<Flex, 3>::from_data(TensorData::new(v, shape), &FlexDevice);
        let ours = floats(activation::softmax(t, 2));
        // Flex over the same axis passes; over axis 1 it must not.
        let right = floats(activation::softmax(f.clone(), 2));
        let wrong = floats(activation::softmax(f, 1));
        assert!(within_softmax(&ours, &right, softmax_bound(70.0)));
        assert!(
            !within_softmax(&ours, &wrong, softmax_bound(70.0)),
            "the bound accepts a softmax over the wrong axis"
        );
    });
}
