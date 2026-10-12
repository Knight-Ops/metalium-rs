//! The element-wise fuser must not take another scalar for zero.
//!
//! Found by Burn's own backend tests on `Fusion<TtBackend>` (`burn-tt-conformance --fusion`:
//! `thresholded_relu_d2`, `thresholded_relu_d2_alpha`, `mask_fill_swap_dims`). Burn lowers `relu(x)` to
//! `mask_fill(x, lower_equal_elem(x, 0), 0)`, and the fuser (`fusion/eltwise.rs`) turned that shape into
//! one `relu` dispatch. But burn-fusion hands a fuser every scalar as an id (`ScalarIr::UInt`), its value
//! bound only when the plan executes, and the fuser took every id for zero: `thresholded_relu(x, 0.5)`
//! and `mask_fill(x, x <= t, -5)` ran as `relu`. A cached plan is also reused for the same graph with
//! other scalars, so the check belongs to execution (`TtOptimization::execute` reads
//! `Context::scalars` and runs the original operations when a scalar is not `+0.0`).
//!
//! The oracle is the host's definition of each function over data where every value is exact. Each
//! case is run after a real `relu` of the same shape, so the other scalars meet a plan that was built
//! for zero, and the fused counters show that a real `relu` and a real `add` + `relu` still run fused.
//! A control requires that the expected values differ from `relu`'s, so a fused `relu` could not pass.

use burn::tensor::activation::{relu, thresholded_relu};
use burn::tensor::{Tensor, TensorData};
use burn_tt::Tt;
use tt_tests::burn_device::{with_device, Config};

const SHAPE: [usize; 2] = [2, 4];

/// Negative, zero, positive and values on both sides of 0.5.
fn values(seed: usize) -> Vec<f32> {
    const V: [f32; 8] = [-3.0, -1.0, 0.0, 0.25, 0.5, 0.75, 2.0, 4.0];
    (0..8).map(|i| V[(i * 3 + seed) % 8]).collect()
}

fn tensor(v: &[f32], d: &burn_tt::TtDevice) -> Tensor<Tt, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), SHAPE), d)
}

fn host(v: &[f32], f: impl Fn(f32) -> f32) -> Vec<f32> {
    v.iter().map(|&x| f(x)).collect()
}

fn read(t: Tensor<Tt, 2>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

#[test]
fn another_scalar_is_not_relu() {
    with_device(Config::default(), |d| {
        let x = values(0);
        let y = values(5);
        let relu_of = |v: f32| if v <= 0.0 { 0.0 } else { v };
        burn_tt::reset_fusion_counters();

        // A real relu: fused, and exact.
        assert_eq!(read(relu(tensor(&x, &d))), host(&x, relu_of));
        assert_eq!(
            burn_tt::fused_relu_count(),
            1,
            "a real relu must stay fused"
        );

        // The same graph with another threshold, after the plan for zero exists.
        let above = |v: f32| if v <= 0.5 { 0.0 } else { v };
        assert_ne!(
            host(&x, above),
            host(&x, relu_of),
            "control: the oracle must tell thresholded_relu from relu"
        );
        assert_eq!(read(thresholded_relu(tensor(&x, &d), 0.5)), host(&x, above));
        assert_eq!(read(relu(tensor(&x, &d))), host(&x, relu_of));
        assert_eq!(
            burn_tt::fused_relu_count(),
            2,
            "only the two real relus execute fused"
        );

        // `mask_fill` with a fill value that is not zero, under a zero threshold.
        let fill = |v: f32| if v <= 0.0 { -5.0 } else { v };
        let masked = |t: Tensor<Tt, 2>| t.clone().mask_fill(t.lower_equal_elem(0.0), -5.0);
        assert_eq!(read(masked(tensor(&x, &d))), host(&x, fill));
        // A zero fill value under another threshold.
        let low = |v: f32| if v <= 0.25 { 0.0 } else { v };
        let thresholded = |t: Tensor<Tt, 2>| t.clone().mask_fill(t.lower_equal_elem(0.25), 0.0);
        assert_eq!(read(thresholded(tensor(&x, &d))), host(&x, low));
        assert_eq!(burn_tt::fused_relu_count(), 2);

        // The same with an add first (the `add_relu` shape).
        burn_tt::reset_fusion_counters();
        let sum = |a: f32, b: f32| a + b;
        let add_relu = |a: Tensor<Tt, 2>, b: Tensor<Tt, 2>| relu(a + b);
        let expect = |f: &dyn Fn(f32) -> f32| -> Vec<f32> {
            x.iter().zip(&y).map(|(&a, &b)| f(sum(a, b))).collect()
        };
        assert_eq!(
            read(add_relu(tensor(&x, &d), tensor(&y, &d))),
            expect(&relu_of)
        );
        assert_eq!(
            burn_tt::fused_add_relu_count(),
            1,
            "a real add + relu must stay fused"
        );
        let add_threshold = |a: Tensor<Tt, 2>, b: Tensor<Tt, 2>| {
            let s = a + b;
            s.clone().mask_fill(s.lower_equal_elem(0.5), 0.0)
        };
        assert_eq!(
            read(add_threshold(tensor(&x, &d), tensor(&y, &d))),
            expect(&above)
        );
        let add_fill = |a: Tensor<Tt, 2>, b: Tensor<Tt, 2>| {
            let s = a + b;
            s.clone().mask_fill(s.lower_equal_elem(0.0), -5.0)
        };
        assert_eq!(
            read(add_fill(tensor(&x, &d), tensor(&y, &d))),
            expect(&fill)
        );
        assert_eq!(burn_tt::fused_add_relu_count(), 1);
    });
}
