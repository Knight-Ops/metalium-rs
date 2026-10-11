//! Tensors with a zero-length dimension -- made, concatenated, read -- on the device.
//!
//! Found by Burn's own backend tests (`burn-tt-conformance`: `cat::should_support_cat_with_empty_tensor`,
//! `_first`, `_multiple_empty_tensors`, and the int and Boolean ones): `cat` is Burn's own composition
//! (allocate the result, `slice_assign` each input), and an empty input is made by `empty([1, 0])`,
//! which tried to make a device buffer with a zero dimension. The device refused at execution, and the
//! failure surfaced only when something was read back ("an earlier device op failed: native
//! initialization: invalid metadata dimensions"). `native_zeros` now declines an empty shape, as
//! `native_random` always did, so the caller builds the empty tensor on the host, which is all an empty
//! tensor needs.
//!
//! The claim is that an empty tensor behaves as Burn defines it: it has its shape, no elements, and
//! `cat` skips it. `cat` is data movement, so the result is **bit for bit** Flex's. The failure was
//! sticky and late, so every read here is `try_into_data` and must be `Ok`: a device op that failed
//! quietly would be found by it. Beyond that: the non-empty operands stay on the device (nothing
//! downloaded, the result `computed_on_device`), and the negative control is an element order the
//! comparison must reject.

use burn::tensor::{Bool, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend, TtDevice};
use tt_tests::burn_device::{with_device, Config};

fn floats(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| seed + i as f32 * 0.5 - 3.0).collect()
}

fn ints(n: usize) -> Vec<i32> {
    (0..n)
        .map(|i| (i as i32).wrapping_mul(0x9E3779B1u32 as i32))
        .collect()
}

fn flags(n: usize) -> Vec<bool> {
    (0..n).map(|i| (i + i / 3) % 2 == 0).collect()
}

/// A tensor of `shape` on the card (empty ones are made by `empty`, as Burn's tests make them).
fn float_on<const D: usize>(d: &TtDevice, shape: [usize; D], seed: f32) -> Tensor<TtBackend, D> {
    let n: usize = shape.iter().product();
    if n == 0 {
        return Tensor::empty(shape, d);
    }
    Tensor::from_data(TensorData::new(floats(n, seed), shape), d).to_device(d)
}

fn float_flex<const D: usize>(shape: [usize; D], seed: f32) -> Tensor<Flex, D> {
    let n: usize = shape.iter().product();
    Tensor::from_data(TensorData::new(floats(n, seed), shape), &FlexDevice)
}

fn read<B: burn::tensor::backend::Backend, const D: usize>(
    t: Tensor<B, D>,
) -> (Vec<usize>, Vec<f32>) {
    let data = t
        .try_into_data()
        .expect("the read failed: a device op failed earlier");
    (data.shape.to_vec(), data.to_vec::<f32>().unwrap())
}

fn bits(shape_and_values: (Vec<usize>, Vec<f32>)) -> (Vec<usize>, Vec<u32>) {
    (
        shape_and_values.0,
        shape_and_values.1.into_iter().map(f32::to_bits).collect(),
    )
}

/// `cat` of `shapes` (some with a zero extent) along `dim`, equal to Flex's, the non-empty operands
/// never downloaded.
fn cat_case<const D: usize>(d: &TtDevice, shapes: &[[usize; D]], dim: usize) {
    let tt: Vec<_> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| float_on(d, *s, i as f32))
        .collect();
    let fx: Vec<_> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| float_flex(*s, i as f32))
        .collect();
    let before = tensor_traffic();
    let out = Tensor::cat(tt, dim);
    let during = tensor_traffic() - before;
    assert_eq!(during.downloads, 0, "{shapes:?} dim {dim}: {during:?}");
    let want = bits(read(Tensor::cat(fx, dim)));
    assert_eq!(bits(read(out)), want, "{shapes:?} dim {dim}");
}

#[test]
fn cat_skips_empty_tensors_wherever_they_are_and_matches_flex() {
    with_device(Config::default(), |d| {
        // Burn's own shapes: empty first, last, repeated, and all empty.
        cat_case(&d, &[[1, 3], [1, 0]], 1);
        cat_case(&d, &[[1, 0], [1, 3]], 1);
        cat_case(&d, &[[2, 0], [2, 2], [2, 0], [2, 1]], 1);
        cat_case(&d, &[[2, 0], [2, 0]], 1);
        // The other axis, other ranks, ragged against the tile.
        cat_case(&d, &[[0, 3], [2, 3], [0, 3]], 0);
        cat_case(&d, &[[3], [0], [4]], 0);
        cat_case(&d, &[[2, 0, 3], [2, 5, 3], [2, 0, 3]], 1);
        cat_case(&d, &[[2, 3, 0], [2, 3, 33]], 2);
        cat_case(&d, &[[0, 33], [7, 33], [1, 33]], 0);
    });
}

/// Integers and Booleans take the same path through `empty` and `cat`.
#[test]
fn empty_int_and_bool_tensors_concatenate_exactly() {
    with_device(Config::default(), |d| {
        let a = ints(6);
        let t = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(a.clone(), [2, 3]), &d)
            .to_device(&d);
        let e = Tensor::<TtBackend, 2, Int>::empty([2, 0], &d);
        let out = Tensor::cat(vec![e.clone(), t, e], 1)
            .try_into_data()
            .expect("int cat read");
        assert_eq!(out.shape.as_slice(), [2, 3]);
        assert_eq!(out.convert::<i32>().to_vec::<i32>().unwrap(), a);

        let b = flags(6);
        let t = Tensor::<TtBackend, 2, Bool>::from_data(TensorData::new(b.clone(), [2, 3]), &d)
            .to_device(&d);
        let e = Tensor::<TtBackend, 2, Bool>::empty([2, 0], &d);
        let out = Tensor::cat(vec![e.clone(), t, e], 1)
            .try_into_data()
            .expect("bool cat read");
        assert_eq!(out.shape.as_slice(), [2, 3]);
        assert_eq!(out.to_vec::<bool>().unwrap(), b);
    });
}

/// An empty tensor is its shape and nothing else: made three ways, read without error.
#[test]
fn an_empty_tensor_keeps_its_shape_and_has_no_elements() {
    with_device(Config::default(), |d| {
        for shape in [[0usize, 3], [3, 0], [1, 0], [0, 0]] {
            for t in [
                Tensor::<TtBackend, 2>::empty(shape, &d),
                Tensor::zeros(shape, &d),
                Tensor::ones(shape, &d),
            ] {
                assert_eq!(t.dims(), shape);
                let data = t.try_into_data().expect("an empty tensor reads back");
                assert_eq!(data.shape.as_slice(), shape);
                assert_eq!(data.num_elements(), 0);
            }
            let i = Tensor::<TtBackend, 2, Int>::zeros(shape, &d)
                .try_into_data()
                .expect("empty int reads back");
            assert_eq!(i.num_elements(), 0);
            let b = Tensor::<TtBackend, 2, Bool>::empty(shape, &d)
                .try_into_data()
                .expect("empty bool reads back");
            assert_eq!(b.num_elements(), 0);
        }
    });
}

/// The comparison tells an order from its reverse: `cat` of empties around `a` then `b` is `[a, b]`,
/// not `[b, a]`.
#[test]
fn negative_control_the_element_order_is_checked() {
    with_device(Config::default(), |d| {
        let tt = vec![
            float_on(&d, [1, 0], 0.0),
            float_on(&d, [1, 2], 1.0),
            float_on(&d, [1, 0], 0.0),
            float_on(&d, [1, 2], 2.0),
        ];
        let ours = bits(read(Tensor::cat(tt, 1)));
        let (a, b) = (float_flex([1, 2], 1.0), float_flex([1, 2], 2.0));
        assert_eq!(ours, bits(read(Tensor::cat(vec![a.clone(), b.clone()], 1))));
        assert_ne!(
            ours,
            bits(read(Tensor::cat(vec![b, a], 1))),
            "the comparison accepts a reversed order"
        );
    });
}
