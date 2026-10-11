//! `int_reshape` and `bool_reshape` of a tensor that is already on the device, for every reshape --
//! a view where the strides express it, else a native copy into the new matrix.
//!
//! Found by Burn's own backend tests (`burn-tt-conformance`: the CTC loss's `[T, N]` Boolean mask
//! reshaped to `[T, N, 1]`, `det`'s `[3, 4, 1]` to `[3, 4]`, `grid_sample`, `one_hot`, `cartesian_grid`
//! and `stack`'s `[4, 6, 2]` to `[4, 12]`): the integer and Boolean reshapes only kept a stored
//! matrix as it was, and failed for a tensor computed on the device when the new shape is stored as
//! another. Float did these natively. The copy moves stored words and never reads them, so the
//! float's `reshaped_strided` now takes I32 and Bool too.
//!
//! The oracle is not an epsilon: a reshape is data movement, so the result must equal Flex's **bit for
//! bit**, over values chosen to expose a word that is read as something else (`i32::MIN`, `i32::MAX`,
//! `-1` for I32; both Boolean values in a pattern that is not a tile's). Beyond it:
//!
//! * nothing is uploaded or downloaded between the tensor being resident and the result being read, and
//!   the result is computed on the device (`computed_on_device`);
//! * the shapes are ragged against the 32 x 32 tile as well as whole tiles, and a reshape of a
//!   transposed, permuted and sliced view is covered, so the strided and the copying paths both run;
//! * padding: a reshaped ragged tensor feeds an integer reduction, which must not see the old tile's
//!   padding in its new place (`OpPadding`, the ragged-producer-then-reduction rule);
//! * negative control: the same data reshaped in a different element order must *not* equal the answer,
//!   so the comparison can tell a wrong order from a right one.

use burn::tensor::{Bool, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend, TtDevice};
use tt_tests::burn_device::{with_device, Config};

/// Full-range words, with the extremes first: any copy that treats a word as a float or a half, or
/// narrows it, changes one of them.
fn words(n: usize) -> Vec<i32> {
    let mut s = 0x2545_F491u32;
    let mut v: Vec<i32> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s as i32
        })
        .collect();
    for (slot, x) in v
        .iter_mut()
        .zip([i32::MIN, i32::MAX, -1, 0, 1, i32::MIN + 1])
    {
        *slot = x;
    }
    v
}

fn flags(n: usize) -> Vec<bool> {
    (0..n).map(|i| (i * 7 + i / 3) % 5 < 2).collect()
}

fn ints<B: burn::tensor::backend::Backend, const D: usize>(t: Tensor<B, D, Int>) -> Vec<i32> {
    t.into_data().convert::<i32>().to_vec::<i32>().unwrap()
}

fn bools<B: burn::tensor::backend::Backend, const D: usize>(t: Tensor<B, D, Bool>) -> Vec<bool> {
    t.into_data().to_vec::<bool>().unwrap()
}

fn resident_int<const D: usize>(t: &Tensor<TtBackend, D, Int>) -> bool {
    t.clone().into_primitive().computed_on_device()
}

fn resident_bool<const D: usize>(t: &Tensor<TtBackend, D, Bool>) -> bool {
    t.clone().into_primitive().computed_on_device()
}

/// Reshape `from` to `to` for I32 and Bool, resident, exactly Flex's.
fn check<const A: usize, const B: usize>(d: &TtDevice, from: [usize; A], to: [usize; B]) {
    let n: usize = from.iter().product();
    assert_eq!(n, to.iter().product::<usize>(), "{from:?} to {to:?}");

    let v = words(n);
    let t =
        Tensor::<TtBackend, A, Int>::from_data(TensorData::new(v.clone(), from), d).to_device(d);
    let f = Tensor::<Flex, A, Int>::from_data(TensorData::new(v, from), &FlexDevice);
    let before = tensor_traffic();
    let r = t.reshape(to);
    let during = tensor_traffic() - before;
    assert_eq!(
        (during.uploads, during.downloads),
        (0, 0),
        "int {from:?} to {to:?}: {during:?}"
    );
    assert!(resident_int(&r), "int {from:?} to {to:?} ran on the host");
    assert_eq!(r.dims(), to);
    assert_eq!(ints(r), ints(f.reshape(to)), "int {from:?} to {to:?}");

    let v = flags(n);
    let t =
        Tensor::<TtBackend, A, Bool>::from_data(TensorData::new(v.clone(), from), d).to_device(d);
    let f = Tensor::<Flex, A, Bool>::from_data(TensorData::new(v, from), &FlexDevice);
    let before = tensor_traffic();
    let r = t.reshape(to);
    let during = tensor_traffic() - before;
    assert_eq!(
        (during.uploads, during.downloads),
        (0, 0),
        "bool {from:?} to {to:?}: {during:?}"
    );
    assert!(resident_bool(&r), "bool {from:?} to {to:?} ran on the host");
    assert_eq!(bools(r), bools(f.reshape(to)), "bool {from:?} to {to:?}");
}

#[test]
fn reshapes_that_change_the_stored_matrix_are_exactly_flexs_on_the_device() {
    with_device(Config::default(), |d| {
        // The shapes Burn's tests failed on.
        check(&d, [5, 3], [5, 3, 1]);
        check(&d, [3, 4, 1], [3, 4]);
        check(&d, [4, 6, 2], [4, 12]);
        check(&d, [2, 2], [2, 2, 1]);
        check(&d, [1, 1, 2, 2], [1, 1, 4]);
        // Ranks one to five, ragged against the tile.
        check(&d, [37], [1, 37]);
        check(&d, [37], [37, 1]);
        check(&d, [7, 9], [63]);
        check(&d, [63], [3, 21]);
        check(&d, [2, 3, 4, 5], [6, 20]);
        check(&d, [6, 20], [2, 3, 4, 5]);
        check(&d, [1, 2, 3, 4, 5], [120]);
        check(&d, [3, 35], [105]);
        check(&d, [105], [5, 21]);
        // Whole tiles, and a reshape that moves a tile boundary.
        check(&d, [32, 64], [64, 32]);
        check(&d, [2, 32, 32], [64, 32]);
        check(&d, [64, 32], [2, 32, 32]);
        check(&d, [4, 32, 8], [32, 32]);
        // A shape whose stored matrix is unchanged is still a plain view.
        check(&d, [2, 3, 4], [6, 4]);
    });
}

/// A view -- transposed, permuted or sliced -- is reshaped through its strides, not its buffer's order.
#[test]
fn reshaping_a_transposed_permuted_or_sliced_view_follows_its_elements() {
    with_device(Config::default(), |d| {
        let v = words(5 * 7);
        let t = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(v.clone(), [5, 7]), &d)
            .to_device(&d);
        let f = Tensor::<Flex, 2, Int>::from_data(TensorData::new(v, [5, 7]), &FlexDevice);
        let before = tensor_traffic();
        let r = t.transpose().reshape([7, 5, 1]);
        assert_eq!((tensor_traffic() - before).downloads, 0);
        assert!(resident_int(&r));
        assert_eq!(
            ints(r),
            ints(f.transpose().reshape([7, 5, 1])),
            "transposed"
        );

        let v = words(2 * 3 * 40);
        let t = Tensor::<TtBackend, 3, Int>::from_data(TensorData::new(v.clone(), [2, 3, 40]), &d)
            .to_device(&d);
        let f = Tensor::<Flex, 3, Int>::from_data(TensorData::new(v, [2, 3, 40]), &FlexDevice);
        let r = t.clone().permute([2, 0, 1]).reshape([40, 6]);
        assert!(resident_int(&r));
        assert_eq!(
            ints(r),
            ints(f.clone().permute([2, 0, 1]).reshape([40, 6])),
            "permuted"
        );
        let r = t.slice([0..2, 1..3, 5..37]).reshape([4, 32]);
        assert!(resident_int(&r));
        assert_eq!(
            ints(r),
            ints(f.slice([0..2, 1..3, 5..37]).reshape([4, 32])),
            "sliced"
        );

        let v = flags(6 * 20);
        let t = Tensor::<TtBackend, 2, Bool>::from_data(TensorData::new(v.clone(), [6, 20]), &d)
            .to_device(&d);
        let f = Tensor::<Flex, 2, Bool>::from_data(TensorData::new(v, [6, 20]), &FlexDevice);
        let r = t.transpose().reshape([20, 6, 1]);
        assert!(resident_bool(&r));
        assert_eq!(
            bools(r),
            bools(f.transpose().reshape([20, 6, 1])),
            "bool transposed"
        );
    });
}

/// The reshaped tensor's padding is not data: an integer reduction over a ragged reshape sees exactly
/// its elements.
#[test]
fn a_reshaped_ragged_tensor_feeds_a_reduction_without_its_padding() {
    with_device(Config::default(), |d| {
        // Small values, so the sums do not wrap and the two backends agree on what they add up to.
        let small = |n: usize| -> Vec<i32> { (0..n).map(|i| (i % 11) as i32 - 5).collect() };
        for (from, to) in [
            ([3usize, 35usize], [5usize, 21usize]),
            ([7, 9], [3, 21]),
            ([2, 5], [10, 1]),
        ] {
            let n = from[0] * from[1];
            let t = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(small(n), from), &d)
                .to_device(&d);
            let f = Tensor::<Flex, 2, Int>::from_data(TensorData::new(small(n), from), &FlexDevice);
            let r = t.reshape(to);
            assert!(resident_int(&r));
            for dim in [0usize, 1] {
                assert_eq!(
                    ints(r.clone().sum_dim(dim)),
                    ints(f.clone().reshape(to).sum_dim(dim)),
                    "{from:?} to {to:?} sum over {dim}"
                );
            }
            assert_eq!(
                ints(r.sum()),
                ints(f.reshape(to).sum()),
                "{from:?} to {to:?} sum"
            );
        }
    });
}

/// The comparison can tell a wrong element order from the right one: the same words reshaped after a
/// transpose are not the words reshaped as they were.
#[test]
fn negative_control_a_different_element_order_is_not_equal() {
    with_device(Config::default(), |d| {
        let v = words(4 * 6);
        let t = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(v.clone(), [4, 6]), &d)
            .to_device(&d);
        let f = Tensor::<Flex, 2, Int>::from_data(TensorData::new(v, [4, 6]), &FlexDevice);
        let ours = ints(t.reshape([2, 12]));
        assert_eq!(ours, ints(f.clone().reshape([2, 12])));
        assert_ne!(
            ours,
            ints(f.transpose().reshape([2, 12])),
            "the comparison accepts a different element order"
        );
    });
}
