//! Phase 10 gate (G2, minimal): tensors of any rank on the device.
//!
//! A tensor is stored as the matrix `[product of the leading dimensions, last
//! dimension]` (`burn_tt::tensor::stored_dims`, rank 1 as one row), so:
//! element-wise ops between tensors of one shape run on the device whatever
//! the rank; a broadcast that is one row or one column of that matrix does
//! too; and a reshape that keeps the matrix is a view. Each against
//! `burn-flex` bit for bit, downloading nothing. A broadcast the matrices
//! would misread -- `[6, 1, 4] + [6, 1]`, a column of `[6, 4]` by the
//! matrices but `[6, 6, 4]` by the rule -- still gets Flex's answer.
//!
//! What it buys: a linear layer's bias is rank 1, and its gradient comes out
//! of `linear_bias_backward` through a reshape, so before this every training
//! step downloaded both biases' gradients and uploaded both biases.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{with_device, Config};

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            match i % 13 {
                0 => -0.0,
                1 => f32::INFINITY,
                _ => ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0,
            }
        })
        .collect()
}

fn bits<const D: usize, B: burn::tensor::backend::Backend>(t: Tensor<B, D>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

/// Bit for bit, but a NaN by class: the device canonicalises NaNs to
/// `0x7FC0_0000` and the host's `inf - inf` is `0xFFC0_0000` (numerics row D).
fn same(got: Vec<u32>, want: Vec<u32>) -> bool {
    let nan = |b: u32| f32::from_bits(b).is_nan();
    got.len() == want.len()
        && got
            .iter()
            .zip(&want)
            .all(|(&g, &w)| g == w || (nan(g) && nan(w)))
}

fn computed_on_device<const D: usize>(t: &Tensor<TtBackend, D>) -> bool {
    match t.clone().into_primitive() {
        burn::tensor::TensorPrimitive::Float(p) => p.computed_on_device(),
        _ => false,
    }
}

/// `f` on the device's tensors and `g` on Flex's: the same bits, and -- if
/// `resident` -- computed on the device with nothing downloaded.
fn check<const D: usize>(
    what: &str,
    resident: bool,
    f: impl FnOnce() -> Tensor<TtBackend, D>,
    g: impl FnOnce() -> Tensor<Flex, D>,
) {
    let before = tensor_traffic();
    let got = f();
    let during = tensor_traffic() - before;
    if resident {
        assert!(
            computed_on_device(&got),
            "{what}: not computed on the device"
        );
        assert_eq!(during.downloads, 0, "{what}: downloaded {during:?}");
    }
    let want = g();
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    assert!(same(bits(got), bits(want)), "{what}: not Flex's bits");
}

#[test]
fn tensors_of_any_rank_stay_on_the_device() {
    with_device(Config::default(), |d| {
        // Rank 1: a bias-sized vector, element-wise with itself and a scalar.
        let n = 128;
        let (pv, gv) = (values(1, n), values(2, n));
        let p =
            Tensor::<TtBackend, 1>::from_data(TensorData::new(pv.clone(), [n]), &d).to_device(&d);
        let g =
            Tensor::<TtBackend, 1>::from_data(TensorData::new(gv.clone(), [n]), &d).to_device(&d);
        let fp = Tensor::<Flex, 1>::from_data(TensorData::new(pv.clone(), [n]), &FlexDevice);
        let fg = Tensor::<Flex, 1>::from_data(TensorData::new(gv.clone(), [n]), &FlexDevice);
        let before = tensor_traffic();
        let step = p.clone() - g.clone().mul_scalar(0.1);
        let during = tensor_traffic() - before;
        assert!(
            computed_on_device(&step),
            "an SGD step on a bias: on the device"
        );
        assert_eq!(during.downloads, 0, "an SGD step on a bias: {during:?}");
        assert_eq!(step.dims(), [n]);
        assert!(same(
            bits(step),
            bits(fp.clone() - fg.clone().mul_scalar(0.1))
        ));

        // Reshapes that keep the stored matrix are views: [1, n] <-> [n].
        let before = tensor_traffic();
        let row: Tensor<TtBackend, 2> = p.clone().reshape([1, n]);
        let back: Tensor<TtBackend, 1> = (row.clone() + row.clone()).reshape([n]);
        assert!(computed_on_device(&back), "a row added and reshaped back");
        assert_eq!((tensor_traffic() - before).downloads, 0);
        let fback: Tensor<Flex, 1> =
            (fp.clone().reshape([1, n]) + fp.clone().reshape([1, n])).reshape([n]);
        assert!(same(bits(back), bits(fback)));

        // Uploaded at their own shape: a reshape on the way would change the
        // stored matrix, which is the host's job.
        fn dev<const D: usize>(
            v: &[f32],
            s: [usize; D],
            d: &burn_tt::TtDevice,
        ) -> Tensor<TtBackend, D> {
            Tensor::<TtBackend, D>::from_data(TensorData::new(v.to_vec(), s), d).to_device(d)
        }
        fn fl<const D: usize>(v: &[f32], s: [usize; D]) -> Tensor<Flex, D> {
            Tensor::<Flex, D>::from_data(TensorData::new(v.to_vec(), s), &FlexDevice)
        }
        // Rank 3, ragged: [3, 5, 70] is stored as [15, 70].
        let s3 = [3, 5, 70];
        let m: usize = s3.iter().product();
        let (av, bv) = (values(3, m), values(4, m));
        let (a3, b3, fa3, fb3) = (dev(&av, s3, &d), dev(&bv, s3, &d), fl(&av, s3), fl(&bv, s3));
        check(
            "rank 3: a * b",
            true,
            || a3.clone() * b3.clone(),
            || fa3.clone() * fb3.clone(),
        );
        check(
            "rank 3: relu",
            true,
            || burn::tensor::activation::relu(a3.clone()),
            || burn::tensor::activation::relu(fa3.clone()),
        );
        // A row of the stored matrix: [1, 1, 70] broadcast over [3, 5, 70].
        let rv = values(5, 70);
        let (r, fr) = (dev(&rv, [1, 1, 70], &d), fl(&rv, [1, 1, 70]));
        check(
            "rank 3 + a row",
            true,
            || a3.clone() + r.clone(),
            || fa3.clone() + fr.clone(),
        );
        // A column of the stored matrix: [3, 5, 1] broadcast over [3, 5, 70].
        let cv = values(6, 15);
        let (c, fc) = (dev(&cv, [3, 5, 1], &d), fl(&cv, [3, 5, 1]));
        check(
            "rank 3 * a column",
            true,
            || a3.clone() * c.clone(),
            || fa3.clone() * fc.clone(),
        );
        // Rank 3 to the matrix it is stored as, and back: views.
        check(
            "rank 3 as its matrix",
            true,
            || {
                let flat: Tensor<TtBackend, 2> = (a3.clone() + b3.clone()).reshape([15, 70]);
                flat.reshape(s3)
            },
            || {
                let flat: Tensor<Flex, 2> = (fa3.clone() + fb3.clone()).reshape([15, 70]);
                flat.reshape(s3)
            },
        );

        // The trap: [6, 1, 4] + [1, 6, 1] broadcasts to [6, 6, 4]; by the
        // matrices, [6, 4] and [6, 1], it would look like a column.
        let (xv, yv) = (values(7, 24), values(8, 6));
        let (x, y, fx, fy) = (
            dev(&xv, [6, 1, 4], &d),
            dev(&yv, [1, 6, 1], &d),
            fl(&xv, [6, 1, 4]),
            fl(&yv, [1, 6, 1]),
        );
        check(
            "the broadcast the matrices misread",
            false,
            || x.clone() + y.clone(),
            || fx.clone() + fy.clone(),
        );
    });
}
