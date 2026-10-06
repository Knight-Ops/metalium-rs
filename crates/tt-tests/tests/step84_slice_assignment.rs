//! Multi-source copies, composed cat/repeat and slice gradients stay resident.
use burn::tensor::{Bool, DType, Int, Tensor, TensorData, TensorPrimitive};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn raw_bf16(bits: Vec<u16>, shape: [usize; 2]) -> TensorData {
    let mut data = TensorData::new(bits, shape);
    data.dtype = DType::BF16;
    data
}

#[test]
fn assignment_and_composed_copies_preserve_float_payloads() {
    with_device(Config::default(), |d| {
        for bf16 in [false, true] {
            let patterns = [
                0x80000000, 0x00010000, 0x7fc10000, 0xff810000, 0x7f800000, 0x3f800000, 0xbf800000,
            ];
            let original: Vec<_> = (0..37 * 35).map(|i| patterns[i % patterns.len()]).collect();
            let replacement: Vec<_> = (0..12)
                .map(|i| patterns[(i + 2) % patterns.len()])
                .collect();
            let data = |bits: &[u32], shape| {
                if bf16 {
                    raw_bf16(bits.iter().map(|b| (b >> 16) as u16).collect(), shape)
                } else {
                    TensorData::new(
                        bits.iter().copied().map(f32::from_bits).collect::<Vec<_>>(),
                        shape,
                    )
                }
            };
            let dtype = if bf16 { DType::BF16 } else { DType::F32 };
            let parent = Tensor::<TtBackend, 2>::from_data(data(&original, [37, 35]), (&d, dtype))
                .to_device(&d);
            let value = Tensor::<TtBackend, 2>::from_data(data(&replacement, [3, 4]), (&d, dtype))
                .to_device(&d);
            let before = tensor_traffic();
            let ((assigned, repeated), report) = burn_tt::with_report(|| {
                // Both inputs are views; output crosses faces and ragged tiles.
                let assigned = parent
                    .clone()
                    .transpose()
                    .slice_assign([30..34, 31..34], value.clone().transpose());
                let repeated = Tensor::cat(vec![value.clone(), value.clone()], 1).repeat_dim(0, 2);
                for t in [&assigned, &repeated] {
                    let TensorPrimitive::Float(t) = t.clone().into_primitive() else {
                        unreachable!()
                    };
                    assert!(t.computed_on_device());
                }
                assert_eq!(tensor_traffic().downloads, before.downloads);
                (assigned, repeated)
            });
            assert_native_model(&report);
            let mut expected: Vec<_> = (0..35)
                .flat_map(|c| {
                    let original = &original;
                    (0..37).map(move |r| original[r * 35 + c])
                })
                .collect();
            for r in 0..4 {
                for c in 0..3 {
                    expected[(30 + r) * 37 + 31 + c] = replacement[c * 4 + r];
                }
            }
            assert!(
                (0..4).any(|r| (0..3)
                    .any(|c| expected[(30 + r) * 37 + 31 + c] != original[(31 + c) * 35 + 30 + r])),
                "omitting assignment must change the oracle"
            );
            let bits = |mut data: TensorData| -> Vec<u32> {
                if bf16 {
                    data.dtype = DType::U16;
                    data.to_vec::<u16>()
                        .unwrap()
                        .into_iter()
                        .map(|b| (b as u32) << 16)
                        .collect()
                } else {
                    data.to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect()
                }
            };
            assert_eq!(bits(assigned.into_data()), expected);
            let repeated_expected: Vec<_> = (0..6)
                .flat_map(|r| {
                    let replacement = &replacement;
                    (0..8).map(move |c| replacement[(r % 3) * 4 + c % 4])
                })
                .collect();
            assert_eq!(bits(repeated.into_data()), repeated_expected);
            assert_eq!(
                bits(parent.into_data()),
                original,
                "copy must preserve parent"
            );
        }
    });
}

#[test]
fn integer_boolean_cat_and_repeat_use_native_assignment() {
    with_device(Config::default(), |d| {
        let ints = Tensor::<TtBackend, 2, Int>::from_data([[i32::MIN, i32::MAX], [-1, 0]], &d)
            .to_device(&d);
        let bools = Tensor::<TtBackend, 2, Bool>::from_data([[true, false], [false, true]], &d)
            .to_device(&d);
        let before = tensor_traffic();
        let ((i, b), report) = burn_tt::with_report(|| {
            let i = Tensor::cat(vec![ints.clone(), ints], 0).repeat_dim(1, 2);
            let b = Tensor::cat(vec![bools.clone(), bools], 1).repeat_dim(0, 2);
            assert!(i.clone().into_primitive().computed_on_device());
            assert!(b.clone().into_primitive().computed_on_device());
            assert_eq!(tensor_traffic().downloads, before.downloads);
            (i, b)
        });
        assert_native_model(&report);
        assert_eq!(
            i.into_data().to_vec::<i32>().unwrap(),
            vec![
                i32::MIN,
                i32::MAX,
                i32::MIN,
                i32::MAX,
                -1,
                0,
                -1,
                0,
                i32::MIN,
                i32::MAX,
                i32::MIN,
                i32::MAX,
                -1,
                0,
                -1,
                0
            ]
        );
        assert_eq!(
            b.into_data().to_vec::<bool>().unwrap(),
            vec![
                true, false, true, false, false, true, false, true, true, false, true, false,
                false, true, false, true
            ]
        );
    });
}

#[test]
fn slice_assignment_backward_routes_native_gradients() {
    use burn::backend::Autodiff;
    with_device(Config::default(), |d| {
        type AD = Autodiff<TtBackend>;
        let x = Tensor::<AD, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &d).require_grad();
        let v = Tensor::<AD, 2>::from_data([[8.0, 9.0]], &d).require_grad();
        let ((gx, gv), report) = burn_tt::with_report(|| {
            let grads = x
                .clone()
                .slice_assign([1..2, 1..3], v.clone())
                .sum()
                .backward();
            (x.grad(&grads).unwrap(), v.grad(&grads).unwrap())
        });
        assert_native_model(&report);
        assert_eq!(
            gx.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(gv.into_data().to_vec::<f32>().unwrap(), vec![1.0, 1.0]);
    });
}

#[test]
fn integer_and_boolean_permuted_unaligned_slices_are_native() {
    with_device(Config::default(), |d| {
        let values: Vec<i32> = (0..24)
            .map(|i| {
                if i % 2 == 0 {
                    i32::MIN.wrapping_add(i)
                } else {
                    i32::MAX.wrapping_sub(i)
                }
            })
            .collect();
        let bits: Vec<bool> = (0..24).map(|i| i % 3 == 0).collect();
        let input =
            Tensor::<TtBackend, 3, Int>::from_data(TensorData::new(values.clone(), [2, 4, 3]), &d)
                .add_scalar(0);
        let flags =
            Tensor::<TtBackend, 3, Bool>::from_data(TensorData::new(bits.clone(), [2, 4, 3]), &d)
                .bool_not()
                .bool_not();
        let before = tensor_traffic();
        let ((ints, bools), report) = burn_tt::with_report(|| {
            let ints = input.swap_dims(0, 1).slice([1..3, 0..2, 1..3]);
            let bools = flags.swap_dims(0, 1).slice([1..3, 0..2, 1..3]);
            assert!(ints.clone().into_primitive().computed_on_device());
            assert!(bools.clone().into_primitive().computed_on_device());
            assert_eq!(tensor_traffic().downloads, before.downloads);
            (ints, bools)
        });
        assert_native_model(&report);
        let indices: Vec<_> = (1..3)
            .flat_map(|c| (0..2).flat_map(move |n| (1..3).map(move |p| (n * 4 + c) * 3 + p)))
            .collect();
        assert_eq!(
            ints.into_data().to_vec::<i32>().unwrap(),
            indices.iter().map(|&i| values[i]).collect::<Vec<_>>()
        );
        assert_eq!(
            bools.into_data().to_vec::<bool>().unwrap(),
            indices.iter().map(|&i| bits[i]).collect::<Vec<_>>()
        );
    });
}
