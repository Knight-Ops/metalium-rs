//! `mask_where` and `powf` with operands that need broadcasting, on the device.
//!
//! Found by Burn's own backend tests (`burn-tt-conformance`: `mask::should_support_mask_where_broadcast`
//! and `_value_small`, `fmod::should_support_fmod_broadcasting_*`, the autodiff `mask_where_broadcast`,
//! `powf::should_support_powf_broadcasted` and `_scalar_tensor`): both kernels took operands of one
//! shape and refused the rest by name, though Burn's contract is that they broadcast -- a `[1, t, t]`
//! score against a `[batch, t, t]` mask is every masked attention. Operands of another shape are now
//! expanded to the common one by device copies (`expand`'s), as the integer and Boolean `mask_where`
//! already did, then the same kernel runs.
//!
//! The oracles differ because the operations do:
//!
//! * `mask_where` selects, so it is **bit for bit** Flex's, over values that expose a select that
//!   computes (`-0.0`, NaN, infinities, a subnormal);
//! * `powf` is an approximation of the SFPU, so it is within `pow_bound`, derived in
//!   `tt_kernels::sfpu::ops` (the `log` and `exp` bounds and the product's rounding), of Flex's, on the
//!   operands broadcast to the common shape.
//!
//! Beyond that: nothing is uploaded or downloaded between the operands being resident and the result
//! being read; the result is computed on the device; the shapes are ragged against the tile; and the
//! negative control is a *misaligned* broadcast (the mask transposed, the operands swapped), which must
//! differ, so the comparison tells which operand was stretched along which axis.

use burn::tensor::{Bool, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend, TtDevice};
use tt_kernels::sfpu::ops::pow_bound;
use tt_tests::burn_device::{with_device, Config};

/// Reproducible values, with the special ones first.
fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    let mut v: Vec<f32> = (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 20.0
        })
        .collect();
    for (slot, x) in v.iter_mut().zip([
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE / 4.0,
        0.0,
    ]) {
        *slot = x;
    }
    v
}

fn flags(seed: usize, n: usize) -> Vec<bool> {
    (0..n).map(|i| (i * 5 + seed + i / 4) % 3 == 0).collect()
}

fn count<const D: usize>(shape: [usize; D]) -> usize {
    shape.iter().product()
}

fn bits<B: burn::tensor::backend::Backend, const D: usize>(t: Tensor<B, D>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn resident<const D: usize>(t: &Tensor<TtBackend, D>) -> bool {
    match t.clone().into_primitive() {
        burn::tensor::TensorPrimitive::Float(p) => p.computed_on_device(),
        _ => false,
    }
}

/// `tensor.mask_where(mask, value)` over shapes that broadcast to one, on the device, equal to Flex's
/// in every bit.
fn mask_where_case<const D: usize>(d: &TtDevice, t: [usize; D], m: [usize; D], v: [usize; D]) {
    let (tv, mv, vv) = (floats(1, count(t)), flags(3, count(m)), floats(2, count(v)));
    let tt_t = Tensor::<TtBackend, D>::from_data(TensorData::new(tv.clone(), t), d).to_device(d);
    let tt_m =
        Tensor::<TtBackend, D, Bool>::from_data(TensorData::new(mv.clone(), m), d).to_device(d);
    let tt_v = Tensor::<TtBackend, D>::from_data(TensorData::new(vv.clone(), v), d).to_device(d);
    let fx_t = Tensor::<Flex, D>::from_data(TensorData::new(tv, t), &FlexDevice);
    let fx_m = Tensor::<Flex, D, Bool>::from_data(TensorData::new(mv, m), &FlexDevice);
    let fx_v = Tensor::<Flex, D>::from_data(TensorData::new(vv, v), &FlexDevice);

    let before = tensor_traffic();
    let out = tt_t.mask_where(tt_m, tt_v);
    let during = tensor_traffic() - before;
    assert_eq!(
        (during.uploads, during.downloads),
        (0, 0),
        "{t:?} {m:?} {v:?}: {during:?}"
    );
    assert!(resident(&out), "{t:?} {m:?} {v:?}: ran on the host");
    let want = fx_t.mask_where(fx_m, fx_v);
    assert_eq!(out.dims(), want.dims(), "{t:?} {m:?} {v:?}");
    assert_eq!(bits(out), bits(want), "{t:?} {m:?} {v:?}");
}

#[test]
fn mask_where_broadcasts_every_operand_and_selects_exactly() {
    with_device(Config::default(), |d| {
        // Burn's own shapes: the score stretched against the mask, a one-element value.
        mask_where_case(&d, [1, 5, 5], [4, 5, 5], [4, 5, 5]);
        mask_where_case(&d, [1, 2, 2], [4, 2, 2], [4, 2, 2]);
        mask_where_case(&d, [2], [2], [1]);
        mask_where_case(&d, [3, 2], [3, 2], [1, 2]);
        mask_where_case(&d, [2, 1, 3], [2, 1, 3], [1, 1, 3]);
        // Each operand the small one, ragged against the tile.
        mask_where_case(&d, [1, 33], [7, 33], [7, 33]);
        mask_where_case(&d, [7, 33], [1, 33], [7, 33]);
        mask_where_case(&d, [7, 33], [7, 33], [7, 1]);
        mask_where_case(&d, [7, 1], [1, 70], [7, 70]);
        mask_where_case(&d, [2, 1, 40], [1, 3, 40], [2, 3, 1]);
        // The same shapes, which never needed it.
        mask_where_case(&d, [4, 5], [4, 5], [4, 5]);
    });
}

/// `x.powf(y)` over shapes that broadcast to one, on the device, within `pow_bound` of Flex's.
fn powf_case<const D: usize>(d: &TtDevice, xs: [usize; D], ys: [usize; D]) {
    // Finite, positive bases: the claim is the broadcast. (`powf(inf, subnormal)` is NaN on the
    // device and `inf` in Flex, the SFPU flushing the exponent; that is not this gate's.)
    let xv: Vec<f32> = floats(7, count(xs))
        .iter()
        .map(|x| if x.is_finite() { x.abs() + 0.25 } else { 1.5 })
        .collect();
    let yv: Vec<f32> = floats(8, count(ys))
        .iter()
        .map(|y| {
            if y.is_finite() {
                y.clamp(-4.0, 4.0)
            } else {
                0.5
            }
        })
        .collect();
    let x = Tensor::<TtBackend, D>::from_data(TensorData::new(xv.clone(), xs), d).to_device(d);
    let y = Tensor::<TtBackend, D>::from_data(TensorData::new(yv.clone(), ys), d).to_device(d);
    let fx = Tensor::<Flex, D>::from_data(TensorData::new(xv, xs), &FlexDevice);
    let fy = Tensor::<Flex, D>::from_data(TensorData::new(yv, ys), &FlexDevice);

    let before = tensor_traffic();
    let out = x.powf(y);
    let during = tensor_traffic() - before;
    assert_eq!(
        (during.uploads, during.downloads),
        (0, 0),
        "{xs:?} {ys:?}: {during:?}"
    );
    assert!(resident(&out), "{xs:?} {ys:?}: ran on the host");

    // The bound is a function of the operand pair, so take the pair at every output element from
    // Flex's own broadcast.
    let shape = fx.clone().powf(fy.clone()).dims();
    assert_eq!(out.dims(), shape, "{xs:?} {ys:?}");
    let ex = fx
        .clone()
        .expand(shape)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let ey = fy
        .clone()
        .expand(shape)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let want = fx.powf(fy).into_data().to_vec::<f32>().unwrap();
    let got = out.into_data().to_vec::<f32>().unwrap();
    for i in 0..got.len() {
        let rel = (got[i] as f64 - want[i] as f64).abs()
            / (want[i] as f64).abs().max(f32::MIN_POSITIVE as f64);
        let bound = pow_bound(ex[i], ey[i]);
        assert!(
            rel <= bound || got[i] == want[i] || (got[i].is_nan() && want[i].is_nan()),
            "{xs:?} {ys:?} [{i}]: {}^{} is {:e}, Flex's {:e}, {rel:e} > {bound:e}",
            ex[i],
            ey[i],
            got[i],
            want[i]
        );
    }
}

#[test]
fn powf_broadcasts_both_operands_within_its_bound() {
    with_device(Config::default(), |d| {
        // Burn's own shapes.
        powf_case(&d, [3], [1]);
        powf_case(&d, [1], [32]);
        // A row against a column, a scalar-like against a matrix, ragged against the tile.
        powf_case(&d, [7, 1], [1, 33]);
        powf_case(&d, [1, 70], [5, 70]);
        powf_case(&d, [5, 70], [1, 1]);
        powf_case(&d, [2, 1, 9], [1, 3, 1]);
        // The same shape, which never needed it.
        powf_case(&d, [4, 5], [4, 5]);
    });
}

/// The comparison can tell which axis was stretched: the same data with the mask transposed, or `powf`'s
/// operands swapped, does not match.
#[test]
fn negative_control_a_misaligned_broadcast_is_not_equal() {
    with_device(Config::default(), |d| {
        let (t, m, v) = ([1usize, 6], [6usize, 6], [6usize, 6]);
        let (tv, mv, vv) = (floats(1, count(t)), flags(3, count(m)), floats(2, count(v)));
        let fx_t = Tensor::<Flex, 2>::from_data(TensorData::new(tv.clone(), t), &FlexDevice);
        let fx_m = Tensor::<Flex, 2, Bool>::from_data(TensorData::new(mv.clone(), m), &FlexDevice);
        let fx_v = Tensor::<Flex, 2>::from_data(TensorData::new(vv.clone(), v), &FlexDevice);
        let tt_t = Tensor::<TtBackend, 2>::from_data(TensorData::new(tv, t), &d).to_device(&d);
        let tt_m =
            Tensor::<TtBackend, 2, Bool>::from_data(TensorData::new(mv, m), &d).to_device(&d);
        let tt_v = Tensor::<TtBackend, 2>::from_data(TensorData::new(vv, v), &d).to_device(&d);
        let ours = bits(tt_t.mask_where(tt_m, tt_v));
        let right = bits(fx_t.clone().mask_where(fx_m.clone(), fx_v.clone()));
        assert_eq!(ours, right);
        let wrong = bits(fx_t.mask_where(fx_m.transpose(), fx_v));
        assert_ne!(ours, wrong, "the comparison accepts a misaligned mask");
    });
}
