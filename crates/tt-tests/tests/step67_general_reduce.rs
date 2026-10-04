//! General axes and ragged repacks remain native. Integer-valued inputs
//! make the Flex comparison exact independently of the SFPU program model.
use burn::tensor::{Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend};
use tt_kernels::sfpu::reduce::{reference, Axis, ReduceOp};
use tt_tests::burn_device::{with_device, Config};

fn resident<const D: usize>(t: Tensor<TtBackend, D>) -> bool {
    match t.into_primitive() {
        TensorPrimitive::Float(p) => p.computed_on_device(),
        _ => false,
    }
}
fn same(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.to_bits(), w.to_bits(), "{g} vs {w}");
    }
}

#[test]
fn every_axis_of_ragged_rank_n_and_permuted_views_is_native() {
    with_device(Config::default(), |d| {
        for shape in [[3, 5, 37], [2, 33, 7], [1, 3, 5]] {
            let values: Vec<_> = (0..shape.iter().product())
                .map(|i| (i % 17) as f32 - 8.)
                .collect();
            for permuted in [false, true] {
                let mut t =
                    Tensor::<TtBackend, 3>::from_data(TensorData::new(values.clone(), shape), &d)
                        .to_device(&d);
                let mut f = Tensor::<Flex, 3>::from_data(
                    TensorData::new(values.clone(), shape),
                    &FlexDevice,
                );
                if permuted {
                    t = t.swap_dims(0, 2);
                    f = f.swap_dims(0, 2);
                }
                for axis in 0..3 {
                    let before = tensor_traffic();
                    let sum = t.clone().sum_dim(axis);
                    let max = t.clone().max_dim(axis);
                    let mean = t.clone().mean_dim(axis);
                    assert!(
                        resident(sum.clone()) && resident(max.clone()) && resident(mean.clone())
                    );
                    assert_eq!(tensor_traffic().downloads, before.downloads);
                    same(
                        &sum.into_data().to_vec::<f32>().unwrap(),
                        &f.clone().sum_dim(axis).into_data().to_vec::<f32>().unwrap(),
                    );
                    same(
                        &max.into_data().to_vec::<f32>().unwrap(),
                        &f.clone().max_dim(axis).into_data().to_vec::<f32>().unwrap(),
                    );
                    // mean_dim multiplies by RN(1/n), versus Flex division.
                    // With an exact integer sum, RN reciprocal and product
                    // contribute (1+u)^2; Flex's RN division contributes u.
                    // In terms of its rounded result the bound is
                    // ((1+u)^2/(1-u)-1)|mean| = (3u+u^2)/(1-u)|mean|.
                    let got = mean.into_data().to_vec::<f32>().unwrap();
                    let want = f
                        .clone()
                        .mean_dim(axis)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                    for (g, w) in got.iter().zip(want) {
                        assert!(
                            (*g as f64 - w as f64).abs()
                                <= ((3.0 + 1.0 / 16_777_216.0)
                                    / 16_777_216.0
                                    / (1.0 - 1.0 / 16_777_216.0))
                                    * (w as f64).abs()
                        );
                    }
                }
            }
        }
    });
}

#[test]
fn long_column_sum_and_max_preserve_the_unfolded_accumulator() {
    with_device(Config::default(), |d| {
        let (rows, cols) = (3, 8193);
        let values: Vec<_> = (0..rows * cols)
            .map(|i| -1. - (i % 31) as f32 / 13.)
            .collect();
        let t =
            Tensor::<TtBackend, 2>::from_data(TensorData::new(values.clone(), [rows, cols]), &d)
                .to_device(&d);
        for op in [ReduceOp::Sum, ReduceOp::Max] {
            let got = if op == ReduceOp::Sum {
                t.clone().sum_dim(1)
            } else {
                t.clone().max_dim(1)
            };
            same(
                &got.into_data().to_vec::<f32>().unwrap(),
                &reference(op, Axis::Cols, &values, rows, cols),
            );
        }
        let t = t.transpose();
        same(
            &t.max_dim(0).into_data().to_vec::<f32>().unwrap(),
            &reference(ReduceOp::Max, Axis::Cols, &values, rows, cols),
        );
    });
}

#[test]
fn ragged_reshape_and_permutation_preserve_all_bits() {
    with_device(Config::default(), |d| {
        let specials = [
            0u32, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc00001, 0xffc00001,
        ];
        let values: Vec<_> = (0..3 * 5 * 37)
            .map(|i| f32::from_bits(specials[i % specials.len()]))
            .collect();
        let t = Tensor::<TtBackend, 3>::from_data(TensorData::new(values.clone(), [3, 5, 37]), &d)
            .to_device(&d)
            .swap_dims(0, 2)
            .reshape([37, 15]);
        assert!(resident(t.clone()));
        let got = t.into_data().to_vec::<f32>().unwrap();
        let want: Vec<_> = (0..37)
            .flat_map(|c| {
                let values = &values;
                (0..5).flat_map(move |b| (0..3).map(move |a| values[(a * 5 + b) * 37 + c]))
            })
            .collect();
        same(&got, &want);
    });
}

#[test]
fn rank_four_reductions_and_autodiff_do_not_download_intermediates() {
    use burn::backend::Autodiff;
    use tt_tests::burn_device::assert_native_model;
    with_device(Config::default(), |d| {
        let shape = [2, 3, 5, 7];
        let values: Vec<_> = (0..210).map(|i| (i % 11) as f32 - 5.).collect();
        let t = Tensor::<TtBackend, 4>::from_data(TensorData::new(values.clone(), shape), &d);
        let f = Tensor::<Flex, 4>::from_data(TensorData::new(values.clone(), shape), &FlexDevice);
        for axis in 0..4 {
            same(
                &t.clone().sum_dim(axis).into_data().to_vec::<f32>().unwrap(),
                &f.clone().sum_dim(axis).into_data().to_vec::<f32>().unwrap(),
            );
        }
        let (gradient, report) = burn_tt::with_report(|| {
            let x = Tensor::<Autodiff<TtBackend>, 4>::from_data(TensorData::new(values, shape), &d)
                .require_grad();
            let y = x.clone().sum_dim(1).sum();
            let grads = y.backward();
            x.grad(&grads).unwrap().into_data()
        });
        same(&gradient.to_vec::<f32>().unwrap(), &vec![1.; 210]);
        assert_native_model(&report);
    });
}

#[test]
fn general_axis_trace_replays_changed_inputs_and_preserves_padding() {
    use burn_tt::Trace;
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            let shape = [3, 5, 7];
            let t = Tensor::<TtBackend, 3>::from_data(TensorData::new(vec![1.; 105], shape), &d)
                .to_device(&d);
            let TensorPrimitive::Float(input) = t.clone().into_primitive() else {
                unreachable!()
            };
            let (trace, first) = Trace::capture(&input, || {
                // Repacked output enters another reduction, checking ragged pad.
                let TensorPrimitive::Float(output) =
                    t.clone().sum_dim(1).sum_dim(2).into_primitive()
                else {
                    unreachable!()
                };
                output
            })
            .unwrap();
            same(&first, &[35.; 3]);
            for value in [2., -1., 0.] {
                same(&trace.run(vec![value; 105]).unwrap(), &[35. * value; 3]);
            }
            drop(trace);
            let got = t.sum_dim(1).sum_dim(2).into_data().to_vec::<f32>().unwrap();
            same(&got, &[0.; 3]);
        },
    );
}

#[test]
fn general_axis_special_values_match_the_instruction_model() {
    with_device(Config::default(), |d| {
        let shape = [3, 37, 5];
        let specials = [
            0.,
            -0.,
            f32::from_bits(1),
            -f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc01234),
            f32::from_bits(0xffc01234),
            f32::MAX,
            f32::MIN_POSITIVE,
            -7.,
        ];
        let values: Vec<_> = (0..3 * 37 * 5)
            .map(|i| specials[i % specials.len()])
            .collect();
        let t = Tensor::<TtBackend, 3>::from_data(TensorData::new(values.clone(), shape), &d);
        let matrix: Vec<_> = (0..15)
            .flat_map(|o| {
                let values = &values;
                (0..37).map(move |k| values[(o / 5 * 37 + k) * 5 + o % 5])
            })
            .collect();
        for op in [ReduceOp::Sum, ReduceOp::Max] {
            let got = if op == ReduceOp::Sum {
                t.clone().sum_dim(1)
            } else {
                t.clone().max_dim(1)
            };
            same(
                &got.into_data().to_vec::<f32>().unwrap(),
                &reference(op, Axis::Cols, &matrix, 15, 37),
            );
        }
    });
}
