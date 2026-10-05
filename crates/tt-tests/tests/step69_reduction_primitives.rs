//! Direct products and Boolean reductions use identities, never log/exp or host data.
use burn::tensor::{Bool, Int, Tensor, TensorData, TensorPrimitive};
use burn_tt::{tensor_traffic, TtBackend};
use tt_kernels::sfpu::reduce::{reference, Axis, ReduceOp};
use tt_tests::burn_device::{with_device, Config};

fn same(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.to_bits(), w.to_bits(), "{g} vs {w}");
    }
}
fn resident<const D: usize>(t: Tensor<TtBackend, D>) -> bool {
    matches!(t.into_primitive(), TensorPrimitive::Float(p) if p.computed_on_device())
}

#[test]
fn negative_products_all_axes_and_full_products_are_native() {
    with_device(Config::default(), |d| {
        let shape = [3, 5, 37];
        // Powers of two and signs give an independent exact oracle. Products
        // stay normal, so no approximation bound or FTZ allowance is needed.
        let data: Vec<_> = (0..shape.iter().product())
            .map(|i| if i % 7 == 0 { -1.0 } else { 1.0 })
            .collect();
        let t = Tensor::<TtBackend, 3>::from_data(TensorData::new(data.clone(), shape), &d)
            .to_device(&d);
        for axis in 0..3 {
            let before = tensor_traffic();
            let got = t.clone().prod_dim(axis);
            assert!(resident(got.clone()));
            assert_eq!(tensor_traffic().downloads, before.downloads);
            let mut out = shape;
            out[axis] = 1;
            let want: Vec<_> = (0..out.iter().product())
                .map(|i| {
                    let mut coords = [i / (out[1] * out[2]), (i / out[2]) % out[1], i % out[2]];
                    (0..shape[axis]).fold(1.0, |v, k| {
                        coords[axis] = k;
                        v * data[(coords[0] * shape[1] + coords[1]) * shape[2] + coords[2]]
                    })
                })
                .collect();
            same(&got.into_data().to_vec::<f32>().unwrap(), &want);
        }
        same(
            &t.clone().prod().into_data().to_vec::<f32>().unwrap(),
            &[data.iter().product()],
        );
        same(
            &t.swap_dims(0, 2)
                .prod()
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            &[data.iter().product()],
        );
    });
}

#[test]
fn products_match_the_program_on_long_axes_and_special_values() {
    with_device(Config::default(), |d| {
        for (rows, cols) in [(3, 8193), (8193, 3), (37, 70)] {
            let specials = [
                0.0f32,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::from_bits(1),
                f32::from_bits(0x80000001),
                f32::from_bits(0x7fc00001),
                f32::from_bits(0xffc00001),
            ];
            let data: Vec<_> = (0..rows * cols)
                .map(|i| {
                    if i < specials.len() {
                        specials[i]
                    } else if i % 13 == 0 {
                        -1.0
                    } else {
                        1.0
                    }
                })
                .collect();
            let t =
                Tensor::<TtBackend, 2>::from_data(TensorData::new(data.clone(), [rows, cols]), &d)
                    .to_device(&d);
            for (dim, axis) in [(0, Axis::Rows), (1, Axis::Cols)] {
                same(
                    &t.clone().prod_dim(dim).into_data().to_vec::<f32>().unwrap(),
                    &reference(ReduceOp::Prod, axis, &data, rows, cols),
                );
            }
        }
        // Independent finite products: gamma_(n-1) bounds RN multiply error,
        // for normal nonoverflowing intermediates. Each order against truth
        // contributes gamma, so the comparison uses 2 gamma |exact product|.
        let n = 70;
        let data: Vec<_> = (0..n)
            .map(|i| if i % 3 == 0 { -1.003f32 } else { 0.999 })
            .collect();
        let exact = data.iter().fold(1.0f64, |p, &x| p * x as f64);
        let gamma = (n - 1) as f64 * 2f64.powi(-24) / (1.0 - (n - 1) as f64 * 2f64.powi(-24));
        let t = Tensor::<TtBackend, 1>::from_data(TensorData::new(data, [n]), &d);
        let got = t.prod().into_scalar() as f64;
        assert!((got - exact).abs() <= 2.0 * gamma * exact.abs());
    });
}

#[test]
fn boolean_reductions_preserve_dtype_and_mask_ragged_padding() {
    with_device(Config::default(), |d| {
        let shape = [3, 5, 37];
        let data: Vec<_> = (0..555).map(|i| i % 11 != 0).collect();
        let t = Tensor::<TtBackend, 3, Bool>::from_data(TensorData::new(data.clone(), shape), &d)
            .to_device(&d);
        for axis in 0..3 {
            let before = tensor_traffic();
            let all = t.clone().all_dim(axis);
            let any = t.clone().any_dim(axis);
            assert!(all.clone().into_primitive().computed_on_device());
            assert!(any.clone().into_primitive().computed_on_device());
            assert_eq!(tensor_traffic().downloads, before.downloads);
            let mut out = shape;
            out[axis] = 1;
            let want: Vec<_> = (0..out.iter().product())
                .map(|i| {
                    let mut coords = [i / (out[1] * out[2]), (i / out[2]) % out[1], i % out[2]];
                    let xs: Vec<_> = (0..shape[axis])
                        .map(|k| {
                            coords[axis] = k;
                            data[(coords[0] * shape[1] + coords[1]) * shape[2] + coords[2]]
                        })
                        .collect();
                    (xs.iter().all(|&v| v), xs.iter().any(|&v| v))
                })
                .collect();
            assert_eq!(
                all.into_data().to_vec::<bool>().unwrap(),
                want.iter().map(|v| v.0).collect::<Vec<_>>()
            );
            assert_eq!(
                any.into_data().to_vec::<bool>().unwrap(),
                want.iter().map(|v| v.1).collect::<Vec<_>>()
            );
        }
        assert!(!t.clone().all().into_scalar());
        assert!(t.any().into_scalar());
        let all = Tensor::<TtBackend, 2, Bool>::from_data(
            TensorData::new(vec![true; 3 * 8193], [3, 8193]),
            &d,
        );
        assert!(all.all().into_scalar(), "zero padding must not enter AND");
    });
}

#[test]
fn arg_reductions_support_ragged_rank_n_and_permuted_views() {
    with_device(Config::default(), |d| {
        let data: Vec<_> = (0..3 * 5 * 37)
            .map(|i| ((i * 13) % 31) as f32 - 15.0)
            .collect();
        let t = Tensor::<TtBackend, 3>::from_data(TensorData::new(data.clone(), [3, 5, 37]), &d)
            .to_device(&d);
        for axis in 0..3 {
            let got: Tensor<TtBackend, 3, Int> = t.clone().argmax(axis);
            assert!(got.clone().into_primitive().computed_on_device());
            let mut shape = [3, 5, 37];
            let n = shape[axis];
            shape[axis] = 1;
            let want: Vec<i32> = (0..shape.iter().product())
                .map(|i| {
                    let mut coords = [
                        i / (shape[1] * shape[2]),
                        (i / shape[2]) % shape[1],
                        i % shape[2],
                    ];
                    let mut best = f32::NEG_INFINITY;
                    let mut index = 0;
                    for k in 0..n {
                        coords[axis] = k;
                        let x = data[(coords[0] * 5 + coords[1]) * 37 + coords[2]];
                        if x > best {
                            best = x;
                            index = k;
                        }
                    }
                    index as i32
                })
                .collect();
            assert_eq!(got.into_data().to_vec::<i32>().unwrap(), want);
        }
        let nan = Tensor::<TtBackend, 3>::from_data(
            TensorData::new(vec![f32::NAN, 2.0, f32::NAN, 2.0, 1.0, 1.0], [1, 2, 3]),
            &d,
        )
        .swap_dims(0, 2);
        assert_eq!(
            nan.argmax(0).into_data().to_vec::<i32>().unwrap(),
            vec![0, 0]
        );
    });
}

#[test]
fn inclusive_scans_keep_logical_order_across_tiles_and_views() {
    use tt_kernels::sfpu::scan::{reference as scan_reference, ScanOp};
    with_device(Config::default(), |d| {
        for shape in [[2, 37, 3], [3, 5, 70]] {
            let data: Vec<_> = (0..shape.iter().product())
                .map(|i| if i % 7 == 0 { -1.0 } else { 1.0 })
                .collect();
            let t = Tensor::<TtBackend, 3>::from_data(TensorData::new(data.clone(), shape), &d)
                .to_device(&d);
            for axis in 0..3 {
                for op in [ScanOp::Sum, ScanOp::Prod] {
                    let before = tensor_traffic();
                    let got = if op == ScanOp::Sum {
                        t.clone().cumsum(axis)
                    } else {
                        t.clone().cumprod(axis)
                    };
                    assert!(resident(got.clone()));
                    assert_eq!(tensor_traffic().downloads, before.downloads);
                    let want: Vec<_> = (0..data.len())
                        .map(|i| {
                            let mut coords = [
                                i / (shape[1] * shape[2]),
                                (i / shape[2]) % shape[1],
                                i % shape[2],
                            ];
                            let n = coords[axis];
                            (0..=n).fold(if op == ScanOp::Sum { 0.0 } else { 1.0 }, |v, k| {
                                coords[axis] = k;
                                let x =
                                    data[(coords[0] * shape[1] + coords[1]) * shape[2] + coords[2]];
                                if op == ScanOp::Sum {
                                    v + x
                                } else {
                                    v * x
                                }
                            })
                        })
                        .collect();
                    same(&got.into_data().to_vec::<f32>().unwrap(), &want);
                }
            }
            let swapped = t.swap_dims(0, 2);
            let scanned = swapped.cumsum(2).swap_dims(0, 2);
            same(
                &scanned.into_data().to_vec::<f32>().unwrap(),
                &scan_reference(ScanOp::Sum, &data, shape[0], shape[1] * shape[2]),
            );
        }
        let rows = 97;
        let cols = 3;
        let values: Vec<_> = (0..rows * cols)
            .map(|i| {
                [
                    1.0e10,
                    1.0,
                    -1.0e10,
                    -0.0,
                    f32::from_bits(1),
                    f32::INFINITY,
                    f32::NAN,
                ][i % 7]
            })
            .collect();
        let t =
            Tensor::<TtBackend, 2>::from_data(TensorData::new(values.clone(), [rows, cols]), &d);
        for op in [ScanOp::Sum, ScanOp::Prod] {
            let got = if op == ScanOp::Sum {
                t.clone().cumsum(0)
            } else {
                t.clone().cumprod(0)
            };
            same(
                &got.into_data().to_vec::<f32>().unwrap(),
                &scan_reference(op, &values, rows, cols),
            );
        }
    });
}

#[test]
fn scans_replay_changed_inputs_and_feed_reductions_without_padding_leaks() {
    use burn_tt::Trace;
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            let shape = [2, 37, 3];
            let t = Tensor::<TtBackend, 3>::from_data(TensorData::new(vec![1.0; 222], shape), &d)
                .to_device(&d);
            let TensorPrimitive::Float(input) = t.clone().into_primitive() else {
                unreachable!()
            };
            let (trace, first) = Trace::capture(&input, || {
                let TensorPrimitive::Float(out) =
                    t.clone().cumsum(1).sum_dim(1).sum_dim(2).into_primitive()
                else {
                    unreachable!()
                };
                out
            })
            .unwrap();
            // sum prefixes 1..37 = 703, three columns each, independent oracle.
            same(&first, &[2109.0; 2]);
            for value in [-1.0, 2.0, 0.0] {
                same(&trace.run(vec![value; 222]).unwrap(), &[2109.0 * value; 2]);
            }
            drop(trace);
            same(&t.sum().into_data().to_vec::<f32>().unwrap(), &[0.0]);
        },
    );
}

#[test]
fn scan_backwards_remain_native() {
    use burn::backend::Autodiff;
    use tt_tests::burn_device::assert_native_model;
    with_device(Config::default(), |d| {
        let (grad, report) = burn_tt::with_report(|| {
            let x = Tensor::<Autodiff<TtBackend>, 2>::from_data(
                TensorData::new(vec![1.0; 74], [2, 37]),
                &d,
            )
            .require_grad();
            let y = x.clone().cumsum(1).sum();
            x.grad(&y.backward()).unwrap()
        });
        assert_native_model(&report);
        let want: Vec<f32> = (0..74).map(|i| (37 - i % 37) as f32).collect();
        same(&grad.into_data().to_vec::<f32>().unwrap(), &want);
        let (grad, report) = burn_tt::with_report(|| {
            let x = Tensor::<Autodiff<TtBackend>, 2>::from_data(
                TensorData::new(vec![1.0; 74], [2, 37]),
                &d,
            )
            .require_grad();
            let y = x.clone().cumprod(1).sum();
            x.grad(&y.backward()).unwrap()
        });
        assert_native_model(&report);
        same(&grad.into_data().to_vec::<f32>().unwrap(), &want);
    });
}

#[test]
fn flipped_and_stepped_permuted_views_preserve_all_bits() {
    use burn::tensor::Slice;
    with_device(Config::default(), |d| {
        let shape = [3, 5, 37];
        let values: Vec<_> = (0..555)
            .map(|i| {
                f32::from_bits(match i % 7 {
                    0 => 0x80000000,
                    1 => 1,
                    2 => 0x80000001,
                    3 => 0x7fc00000 | i as u32,
                    4 => 0xffc00000 | i as u32,
                    5 => 0x7f800000,
                    _ => (i as f32).to_bits(),
                })
            })
            .collect();
        let t = Tensor::<TtBackend, 3>::from_data(TensorData::new(values.clone(), shape), &d)
            .to_device(&d)
            .swap_dims(0, 2);
        let before = tensor_traffic();
        let flipped = t.clone().flip([0, 2]);
        let stepped = t.slice([
            Slice::with_step(1, Some(36), 3),
            Slice::with_step(0, None, -2),
            Slice::full(),
        ]);
        assert!(resident(flipped.clone()) && resident(stepped.clone()));
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let mut want = Vec::new();
        for c in (0..37).rev() {
            for b in 0..5 {
                for a in (0..3).rev() {
                    want.push(values[(a * 5 + b) * 37 + c]);
                }
            }
        }
        same(&flipped.into_data().to_vec::<f32>().unwrap(), &want);
        want.clear();
        for c in (1..36).step_by(3) {
            for b in [4, 2, 0] {
                for a in 0..3 {
                    want.push(values[(a * 5 + b) * 37 + c]);
                }
            }
        }
        assert_eq!(stepped.dims(), [12, 3, 3]);
        same(&stepped.into_data().to_vec::<f32>().unwrap(), &want);
    });
}
