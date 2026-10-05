//! Burn's normalization compositions are native; no fused kernel is required.
use burn::{
    backend::Autodiff,
    module::Param,
    nn::{LayerNormConfig, RmsNormConfig},
    tensor::{Tensor, TensorData},
};
use burn_tt::{tensor_traffic, TtBackend};
use tt_kernels::{
    kind,
    sfpu::{
        ops::{kind_sfpu, reference},
        reduce::{reference as reduce_reference, Axis, ReduceOp},
    },
};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn same(got: &[f32], want: &[f32]) {
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.to_bits(), w.to_bits(), "{g} vs {w}");
    }
    assert_eq!(got.len(), want.len());
}

fn model(x: &[f32], rows: usize, cols: usize, epsilon: f32, rms: bool) -> Vec<f32> {
    let mean = |x: &[f32]| {
        let sum = reduce_reference(ReduceOp::Sum, Axis::Cols, x, rows, cols);
        reference(kind::MUL_SCALAR, 1.0 / cols as f32, &sum, None, rows, 1)
    };
    let centered = if rms {
        x.to_vec()
    } else {
        reference(kind::SUB, 0.0, x, Some(&mean(x)), rows, cols)
    };
    let squared = reference(kind::MUL, 0.0, &centered, Some(&centered), rows, cols);
    let var = mean(&squared);
    let var = reference(kind::ADD_SCALAR, epsilon, &var, None, rows, 1);
    let denom = reference(kind_sfpu::SQRT, 0.0, &var, None, rows, 1);
    reference(kind_sfpu::DIV, 0.0, &centered, Some(&denom), rows, cols)
}

#[test]
fn layer_and_rms_norm_ragged_wide_and_constant_rows_match_programs() {
    with_device(Config::default(), |d| {
        for (rows, cols) in [(3, 37), (2, 8193), (33, 7)] {
            let values: Vec<_> = (0..rows * cols)
                .map(|i| {
                    if i / cols == 0 {
                        1.0
                    } else {
                        ((i * 13) % 31) as f32 / 16.0 - 1.0
                    }
                })
                .collect();
            let input = Tensor::<TtBackend, 2>::from_data(
                TensorData::new(values.clone(), [rows, cols]),
                &d,
            )
            .to_device(&d);
            let layer = LayerNormConfig::new(cols)
                .with_epsilon(0.25)
                .init::<TtBackend>(&d);
            let nobias = LayerNormConfig::new(cols)
                .with_epsilon(0.25)
                .with_bias(false)
                .init::<TtBackend>(&d);
            let rms = RmsNormConfig::new(cols)
                .with_epsilon(0.25)
                .init::<TtBackend>(&d);
            let (outputs, report) = burn_tt::with_report(|| {
                let before = tensor_traffic();
                let outputs = [
                    layer.forward(input.clone()),
                    nobias.forward(input.clone()),
                    rms.forward(input),
                ];
                assert_eq!(tensor_traffic().downloads, before.downloads);
                outputs
            });
            assert_native_model(&report);
            same(
                &outputs[0].clone().into_data().to_vec::<f32>().unwrap(),
                &model(&values, rows, cols, 0.25, false),
            );
            same(
                &outputs[1].clone().into_data().to_vec::<f32>().unwrap(),
                &model(&values, rows, cols, 0.25, false),
            );
            same(
                &outputs[2].clone().into_data().to_vec::<f32>().unwrap(),
                &model(&values, rows, cols, 0.25, true),
            );
        }
    });
}

#[test]
fn rank_four_permuted_normalization_and_parameters_stay_resident() {
    with_device(Config::default(), |d| {
        let shape = [2, 3, 5, 7];
        let values: Vec<_> = (0..210).map(|i| (i % 17) as f32 - 8.0).collect();
        let input = Tensor::<TtBackend, 4>::from_data(TensorData::new(values.clone(), shape), &d)
            .to_device(&d)
            .swap_dims(1, 3);
        let mut layer = LayerNormConfig::new(3)
            .with_epsilon(0.25)
            .init::<TtBackend>(&d);
        layer.gamma = Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![2.0, -1.0, 0.5], [3]),
            &d,
        ));
        layer.beta = Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0], [3]),
            &d,
        )));
        let rms = RmsNormConfig::new(3)
            .with_epsilon(0.25)
            .init::<TtBackend>(&d);
        let (outputs, report) =
            burn_tt::with_report(|| [layer.forward(input.clone()), rms.forward(input)]);
        assert_native_model(&report);
        let mut packed = Vec::new();
        for b in 0..2 {
            for c in 0..7 {
                for s in 0..5 {
                    for a in 0..3 {
                        packed.push(values[((b * 3 + a) * 5 + s) * 7 + c]);
                    }
                }
            }
        }
        let normalized = model(&packed, 70, 3, 0.25, false);
        let scaled = reference(kind::MUL, 0.0, &normalized, Some(&[2.0, -1.0, 0.5]), 70, 3);
        let shifted = reference(kind::ADD, 0.0, &scaled, Some(&[1.0, 2.0, 3.0]), 70, 3);
        same(
            &outputs[0].clone().into_data().to_vec::<f32>().unwrap(),
            &shifted,
        );
        same(
            &outputs[1].clone().into_data().to_vec::<f32>().unwrap(),
            &model(&packed, 70, 3, 0.25, true),
        );
    });
}

#[test]
fn layer_and_rms_backwards_match_independent_analytic_derivatives() {
    with_device(Config::default(), |d| {
        for rms in [false, true] {
            let x = Tensor::<Autodiff<TtBackend>, 2>::from_data([[-1.0, 1.0]], &d).require_grad();
            let gamma = Tensor::<Autodiff<TtBackend>, 1>::from_data([1.0, 2.0], &d).require_grad();
            let beta = Tensor::<Autodiff<TtBackend>, 1>::from_data([0.0, 0.0], &d).require_grad();
            let g = Tensor::<Autodiff<TtBackend>, 2>::from_data([[2.0, -1.0]], &d);
            let (gradients, report) = burn_tt::with_report(|| {
                let out = if rms {
                    let mut module = RmsNormConfig::new(2)
                        .with_epsilon(1.0)
                        .init::<Autodiff<TtBackend>>(&d);
                    module.gamma = Param::from_tensor(gamma.clone());
                    module.forward(x.clone())
                } else {
                    let mut module = LayerNormConfig::new(2)
                        .with_epsilon(1.0)
                        .init::<Autodiff<TtBackend>>(&d);
                    module.gamma = Param::from_tensor(gamma.clone());
                    module.beta = Some(Param::from_tensor(beta.clone()));
                    module.forward(x.clone())
                };
                let grads = (out * g).sum().backward();
                let dx = x.grad(&grads).unwrap();
                let dg = gamma.grad(&grads).unwrap();
                let db = if rms {
                    None
                } else {
                    Some(beta.grad(&grads).unwrap())
                };
                (dx, dg, db)
            });
            assert_native_model(&report);
            // h = upstream*gamma = [2,-2], mean(h*x)=-2, denom=sqrt(2).
            // dx=(h - x*mean(h*x)/2)/sqrt(2); LayerNorm also subtracts
            // mean(h)=0. dgamma=upstream*x/sqrt(2), dbeta=upstream.
            // The graph has fewer than 32 rounding equivalents per path
            // (sqrt/div count twice), over terms with absolute sum <=4.
            // Sqrt backward additionally evaluates pow(2,-0.5), whose
            // approximation error contributes at most four times pow_bound.
            // All intermediates are normal and the denominator is >=1.
            let u = 2f64.powi(-24);
            let bound = 4.0 * 32.0 * u / (1.0 - 32.0 * u)
                + 4.0 * tt_kernels::sfpu::ops::pow_bound(2.0, -0.5);
            for (got, want) in [
                (
                    gradients.0.into_data().to_vec::<f32>().unwrap(),
                    vec![1.0 / 2f64.sqrt(), -1.0 / 2f64.sqrt()],
                ),
                (
                    gradients.1.into_data().to_vec::<f32>().unwrap(),
                    vec![-2.0 / 2f64.sqrt(), -1.0 / 2f64.sqrt()],
                ),
            ] {
                for (got, want) in got.iter().zip(want) {
                    assert!((*got as f64 - want).abs() <= bound, "{got} vs {want}");
                }
            }
            if let Some(db) = gradients.2 {
                same(&db.into_data().to_vec::<f32>().unwrap(), &[2.0, -1.0]);
            }
        }
    });
}
