//! Small native compositions, typed Boolean conversions, and movement ops.
use burn_flex::{Flex, FlexDevice};
use burn_tensor::{Bool, Int, Tensor, TensorData};
use burn_tt::TtBackend;
use tt_tests::burn_device::{assert_native_model, with_device, Config};

#[test]
fn argmin_preserves_ties_nan_priority_and_special_values() {
    with_device(Config::default(), |d| {
        for cols in [1, 10, 37, 65] {
            let specials = [
                f32::INFINITY,
                f32::NEG_INFINITY,
                0.0,
                -0.0,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::NAN,
                -f32::NAN,
                -3.0,
            ];
            let mut values: Vec<_> = (0..35 * cols)
                .map(|i| specials[i % specials.len()])
                .collect();
            values[..cols].fill(f32::INFINITY);
            values[cols..2 * cols].fill(-0.0);
            let data = TensorData::new(values, [35, cols]);
            for axis in [0, 1] {
                let want = Tensor::<Flex, 2>::from_data(data.clone(), &FlexDevice)
                    .argmin(axis)
                    .into_data()
                    .convert::<i32>();
                let (got, report) = burn_tt::with_report(|| {
                    burn_tt::strictly(|| {
                        Tensor::<TtBackend, 2>::from_data(data.clone(), &d)
                            .argmin(axis)
                            .into_data()
                    })
                });
                assert_eq!(got, want, "cols={cols}, axis={axis}");
                assert_native_model(&report);
            }
        }
        let (got, report) = burn_tt::with_report(|| {
            Tensor::<TtBackend, 1>::from_data([3.0, -7.0, -7.0], &d)
                .argmin(0)
                .into_data()
        });
        assert_eq!(got.to_vec::<i32>().unwrap(), [1]);
        assert_native_model(&report);
    });
}

#[test]
fn boolean_equality_conversions_and_float_truth_reductions_are_native() {
    with_device(Config::default(), |d| {
        let data = TensorData::new(
            (0..35 * 37).map(|i| i % 3 != 0).collect::<Vec<_>>(),
            [35, 37],
        );
        let ((equal, scalar_true, scalar_false, floats, integers, any, all), report) =
            burn_tt::with_report(|| {
                burn_tt::strictly(|| {
                    let b = Tensor::<TtBackend, 2, Bool>::from_data(data.clone(), &d).to_device(&d);
                    let row = Tensor::<TtBackend, 2, Bool>::from_data(
                        TensorData::new((0..37).map(|i| i % 2 == 0).collect::<Vec<_>>(), [1, 37]),
                        &d,
                    );
                    let f = b.clone().float();
                    // Arithmetic consumes the converted buffer to check its physical element type.
                    let floats = (f.clone() + 2.0).into_data();
                    let integers = b.clone().int().float().into_data();
                    (
                        b.clone().equal(row).into_data(),
                        b.clone().equal_elem(true).into_data(),
                        b.equal_elem(false).into_data(),
                        floats,
                        integers,
                        f.clone().any_dim(1).into_data(),
                        f.all_dim(1).into_data(),
                    )
                })
            });
        let b = Tensor::<Flex, 2, Bool>::from_data(data.clone(), &FlexDevice);
        let row = Tensor::<Flex, 2, Bool>::from_data(
            TensorData::new((0..37).map(|i| i % 2 == 0).collect::<Vec<_>>(), [1, 37]),
            &FlexDevice,
        );
        assert_eq!(equal, b.clone().equal(row).into_data());
        assert_eq!(scalar_true, b.clone().equal_elem(true).into_data());
        assert_eq!(scalar_false, b.clone().equal_elem(false).into_data());
        assert_eq!(floats, (b.clone().float() + 2.0).into_data());
        assert_eq!(integers, b.clone().int().float().into_data());
        assert_eq!(any, b.clone().float().any_dim(1).into_data());
        assert_eq!(all, b.float().all_dim(1).into_data());
        assert_native_model(&report);
        let (got, report) = burn_tt::with_report(|| {
            burn_tt::strictly(|| {
                let f = Tensor::<TtBackend, 1>::from_data([0.0, -0.0, f32::NAN, f32::INFINITY], &d);
                (f.clone().any().into_data(), f.all().into_data())
            })
        });
        assert_eq!(got.0.to_vec::<bool>().unwrap(), [true]);
        assert_eq!(got.1.to_vec::<bool>().unwrap(), [false]);
        assert_native_model(&report);
    });
}

#[test]
fn integer_and_boolean_expansion_preserves_bits_and_residency() {
    with_device(Config::default(), |d| {
        for (from, to) in [([1, 1], [35, 37]), ([1, 37], [35, 37]), ([35, 1], [35, 37])] {
            let ints = TensorData::new(
                (0..from[0] * from[1])
                    .map(|i| [i32::MIN, i32::MAX, -1, 0][i % 4])
                    .collect::<Vec<_>>(),
                from,
            );
            let bools = TensorData::new(
                (0..from[0] * from[1])
                    .map(|i| i % 2 == 0)
                    .collect::<Vec<_>>(),
                from,
            );
            let ((i, b), report) = burn_tt::with_report(|| {
                burn_tt::strictly(|| {
                    let i = Tensor::<TtBackend, 2, Int>::from_data(ints.clone(), &d)
                        .to_device(&d)
                        .expand(to);
                    let b = Tensor::<TtBackend, 2, Bool>::from_data(bools.clone(), &d)
                        .to_device(&d)
                        .expand(to);
                    assert!(i.clone().into_primitive().computed_on_device());
                    assert!(b.clone().into_primitive().computed_on_device());
                    (i.into_data(), b.into_data())
                })
            });
            assert_eq!(
                i,
                Tensor::<Flex, 2, Int>::from_data(ints, &FlexDevice)
                    .expand(to)
                    .into_data()
                    .convert::<i32>()
            );
            assert_eq!(
                b,
                Tensor::<Flex, 2, Bool>::from_data(bools, &FlexDevice)
                    .expand(to)
                    .into_data()
            );
            assert_native_model(&report);
        }
    });
}

#[test]
fn permutations_share_resident_storage_and_feed_native_compute() {
    with_device(Config::default(), |d| {
        let data = TensorData::new(
            (0..32 * 32 * 32)
                .map(|i| (i % 11) as f32 - 5.0)
                .collect::<Vec<_>>(),
            [32, 32, 32],
        );
        // Every permutation is a resident view; explicit readback can gather layouts
        // that the native whole-tile materializer does not support.
        for axes in [[2, 0, 1], [2, 1, 0], [1, 2, 0], [1, 0, 2]] {
            let want = Tensor::<Flex, 3>::from_data(data.clone(), &FlexDevice)
                .permute(axes)
                .into_data();
            let (got, report) = burn_tt::with_report(|| {
                burn_tt::strictly(|| {
                    let input = Tensor::<TtBackend, 3>::from_data(data.clone(), &d) + 0.0;
                    let view = input.permute(axes);
                    assert!(view.clone().into_primitive().tensor().computed_on_device());
                    view.into_data()
                })
            });
            assert_eq!(got, want);
            assert_native_model(&report);
        }
        for axes in [[0, 1, 2], [0, 2, 1]] {
            let want = (Tensor::<Flex, 3>::from_data(data.clone(), &FlexDevice).permute(axes)
                + 1.0)
                .into_data();
            let (got, report) = burn_tt::with_report(|| {
                burn_tt::strictly(|| {
                    let input = Tensor::<TtBackend, 3>::from_data(data.clone(), &d) + 0.0;
                    let view = input.permute(axes);
                    assert!(view.clone().into_primitive().tensor().computed_on_device());
                    (view + 1.0).into_data()
                })
            });
            assert_eq!(got, want, "axes={axes:?}");
            assert_native_model(&report);
            if axes != [0, 1, 2] {
                let op = report.op("float_permute").unwrap();
                assert_eq!((op.downloads, op.uploads, op.staged), (0, 0, 0));
            }
        }
    });
}

#[test]
fn boolean_stores_and_column_broadcasts_preserve_logical_values() {
    use burn_tensor::{BoolStore, DType};
    with_device(Config::default(), |d| {
        for store in [BoolStore::Native, BoolStore::U8, BoolStore::U32] {
            let dtype = DType::Bool(store);
            let data = TensorData::new(
                (0..35 * 37).map(|i| i % 2 == 0).collect::<Vec<_>>(),
                [35, 37],
            )
            .convert_dtype(dtype);
            let column = TensorData::new((0..35).map(|i| i % 3 == 0).collect::<Vec<_>>(), [35, 1])
                .convert_dtype(dtype);
            let ((eq, floats, ints), report) = burn_tt::with_report(|| {
                burn_tt::strictly(|| {
                    let b = Tensor::<TtBackend, 2, Bool>::from_data(data.clone(), (&d, dtype));
                    let c = Tensor::<TtBackend, 2, Bool>::from_data(column.clone(), (&d, dtype));
                    (
                        b.clone().equal(c).into_data(),
                        b.clone().float().into_data(),
                        b.int().into_data(),
                    )
                })
            });
            let expected = data.clone().convert::<bool>().to_vec::<bool>().unwrap();
            let col = column.convert::<bool>().to_vec::<bool>().unwrap();
            assert_eq!(eq.dtype, dtype);
            assert_eq!(
                eq.convert::<bool>().to_vec::<bool>().unwrap(),
                expected
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v == col[i / 37])
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                floats.to_vec::<f32>().unwrap(),
                expected
                    .iter()
                    .map(|&v| if v { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>()
            );
            assert_eq!(ints.dtype, DType::I32);
            assert_eq!(
                ints.to_vec::<i32>().unwrap(),
                expected.iter().map(|&v| i32::from(v)).collect::<Vec<_>>()
            );
            assert_native_model(&report);
        }
    });
}
