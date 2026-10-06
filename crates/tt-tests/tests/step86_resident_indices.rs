//! Resident dynamic indices: raw copies, deterministic additions and domains.
use burn::tensor::{module, DType, Int, Tensor, TensorData, TensorPrimitive};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

#[test]
fn arbitrary_axis_gather_preserves_raw_payloads_and_resident_indices() {
    with_device(Config::default(), |d| {
        for bf16 in [false, true] {
            let patterns = [
                0x80000000, 0x00010000, 0x7fc10000, 0xff810000, 0x7f800000, 0x3f800000,
            ];
            let bits: Vec<u32> = (0..210).map(|i| patterns[(i + i / 3) % 6]).collect();
            let data = if bf16 {
                let mut data = TensorData::new(
                    bits.iter().map(|b| (b >> 16) as u16).collect::<Vec<_>>(),
                    [2, 35, 3],
                );
                data.dtype = DType::BF16;
                data
            } else {
                TensorData::new(
                    bits.iter().copied().map(f32::from_bits).collect::<Vec<_>>(),
                    [2, 35, 3],
                )
            };
            let dtype = if bf16 { DType::BF16 } else { DType::F32 };
            let x = Tensor::<TtBackend, 3>::from_data(data, (&d, dtype)).to_device(&d);
            let iv: Vec<i32> = (0..24).map(|i| [0, 15, 16, 31, 32, 34][i % 6]).collect();
            let indices =
                Tensor::<TtBackend, 3, Int>::from_data(TensorData::new(iv.clone(), [2, 4, 3]), &d)
                    .add_scalar(0);
            assert!(indices.clone().into_primitive().computed_on_device());
            let before = tensor_traffic();
            let (out, report) = burn_tt::with_report(|| x.clone().gather(1, indices));
            assert_native_model(&report);
            let TensorPrimitive::Float(p) = out.clone().into_primitive() else {
                unreachable!()
            };
            assert!(p.computed_on_device());
            assert_eq!(tensor_traffic().downloads, before.downloads);
            let expected: Vec<_> = (0..24)
                .map(|i| bits[(i / 12 * 35 + iv[i] as usize) * 3 + i % 3])
                .collect();
            let mut data = out.into_data();
            let got: Vec<_> = if bf16 {
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
            };
            assert_eq!(got, expected);
        }
        let x =
            Tensor::<TtBackend, 2, Int>::from_data([[i32::MIN, i32::MAX], [-1, 0], [31, 32]], &d)
                .to_device(&d);
        let indices = Tensor::<TtBackend, 1, Int>::from_data([2, 0, 2], &d).add_scalar(0);
        let before = tensor_traffic();
        let (out, report) = burn_tt::with_report(|| x.select(0, indices));
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        assert_eq!(
            out.into_data().to_vec::<i32>().unwrap(),
            vec![31, 32, i32::MIN, i32::MAX, 31, 32]
        );
    });
}

#[test]
fn duplicate_scatter_add_and_embedding_gradients_follow_logical_order() {
    use burn::backend::Autodiff;
    with_device(Config::default(), |d| {
        let x = Tensor::<TtBackend, 2>::zeros([4, 2], &d);
        let idx =
            Tensor::<TtBackend, 2, Int>::from_data([[2, 2], [2, 2], [2, 2]], &d).add_scalar(0);
        let value =
            Tensor::<TtBackend, 2>::from_data([[1e20, 1e20], [-1e20, -1e20], [3.0, 9.0]], &d)
                .to_device(&d);
        let before = tensor_traffic();
        let (out, report) =
            burn_tt::with_report(|| x.scatter(0, idx, value, burn::tensor::IndexingUpdateOp::Add));
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        assert_eq!(
            out.into_data().to_vec::<f32>().unwrap(),
            vec![0.0, 0.0, 0.0, 0.0, 3.0, 9.0, 0.0, 0.0]
        );
        type AD = Autodiff<TtBackend>;
        let weights =
            Tensor::<AD, 2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &d).require_grad();
        let indices = Tensor::<AD, 2, Int>::from_data([[1, 1, 2]], &d).add_scalar(0);
        let before = tensor_traffic();
        let ((out, grad), report) = burn_tt::with_report(|| {
            let out = module::embedding(weights.clone(), indices);
            let grads = out.clone().sum().backward();
            (out.inner(), weights.grad(&grads).unwrap())
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        assert_eq!(
            out.into_data().to_vec::<f32>().unwrap(),
            vec![3.0, 4.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            grad.into_data().to_vec::<f32>().unwrap(),
            vec![0.0, 0.0, 2.0, 2.0, 1.0, 1.0]
        );
    });
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_duplicate_add_narrows_only_after_all_indices() {
    use burn::tensor::FloatDType;
    with_device(Config::default(), |d| {
        let x = Tensor::<TtBackend, 1>::zeros([2], &d).cast(FloatDType::BF16);
        let value =
            Tensor::<TtBackend, 1>::from_data([256.0, 1.0, -256.0], &d).cast(FloatDType::BF16);
        let idx = Tensor::<TtBackend, 1, Int>::from_data([1, 1, 1], &d).add_scalar(0);
        let (out, report) = burn_tt::with_report(|| {
            x.select_assign(0, idx, value, burn::tensor::IndexingUpdateOp::Add)
        });
        assert_native_model(&report);
        assert_eq!(
            out.into_data().convert::<f32>().to_vec::<f32>().unwrap(),
            vec![0.0, 1.0]
        );
    });
}

#[test]
fn resident_indices_reject_negative_and_logical_end_before_copying() {
    use tt_kernels::{
        session::{Session, TileChoice},
        tensor::Elem,
    };
    for invalid in [u32::MAX, 35] {
        tt_ttsim::fork_scope(|| {
            #[cfg(not(feature = "silicon"))]
            let mut sim = tt_ttsim::Simulator::open().unwrap();
            #[cfg(not(feature = "silicon"))]
            let dev = tt_device::Device::open(sim.transport()).unwrap();
            #[cfg(not(feature = "silicon"))]
            let mut s = Session::open(
                dev,
                tt_firmware_images::ROLES,
                TileChoice::Count(2),
                |_, _| Ok(None),
            )
            .unwrap();
            #[cfg(feature = "silicon")]
            let mut s = Session::open_card(
                tt_tests::backend::device_index(),
                tt_firmware_images::ROLES,
                TileChoice::Count(2),
            )
            .unwrap();
            s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
            let x = s
                .upload(&(0..3 * 35).map(|i| i as f32).collect::<Vec<_>>(), 3, 35)
                .unwrap();
            let valid = s.upload_bits(&[0, 16, 34], 3, 1, Elem::I32).unwrap();
            let out = s.gather_indexed(&x, &valid).unwrap();
            assert_eq!(s.download(&out).unwrap(), vec![0.0, 51.0, 104.0]);
            s.free(out).unwrap();
            let bad = s.upload_bits(&[0, invalid, 34], 3, 1, Elem::I32).unwrap();
            let _pending = s.gather_indexed(&x, &bad).unwrap();
            let err = s.sync().unwrap_err();
            assert!(
                err.to_string().contains("code 11"),
                "logical index DOMAIN: {err}"
            );
        })
        .unwrap();
    }
}

#[test]
fn dynamic_indices_trace_replays_device_produced_changes() {
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            let x = Tensor::<TtBackend, 2>::from_data([[1.0, 4.0, 2.0], [9.0, 3.0, 8.0]], &d)
                .mul_scalar(1.0);
            let TensorPrimitive::Float(input) = x.clone().into_primitive() else {
                unreachable!()
            };
            let before = tensor_traffic();
            let ((trace, first), report) = burn_tt::with_report(|| {
                burn_tt::Trace::capture(&input, || {
                    // argmax changes between replays, so cached host indices cannot pass.
                    let index = x.clone().argmax(1);
                    let out = x.clone().gather(1, index);
                    let TensorPrimitive::Float(out) = out.into_primitive() else {
                        unreachable!()
                    };
                    out
                })
                .unwrap()
            });
            assert_native_model(&report);
            assert_eq!(tensor_traffic().uploads, before.uploads);
            assert_eq!(first, vec![4.0, 9.0]);
            assert_eq!(
                trace.run(vec![8.0, 2.0, 1.0, 0.0, 5.0, 7.0]).unwrap(),
                vec![8.0, 7.0]
            );
            assert_eq!(
                trace.run(vec![-1.0, -2.0, 3.0, 4.0, 6.0, 5.0]).unwrap(),
                vec![3.0, 6.0]
            );
            drop(trace);
            assert_eq!(x.sum().into_data().to_vec::<f32>().unwrap(), vec![15.0]);
        },
    );
}

#[test]
fn integer_and_boolean_scatter_and_select_updates_are_resident() {
    use burn::tensor::Bool;
    use burn_tensor::ops::{BoolTensorOps, IntTensorOps};
    with_device(Config::default(), |d| {
        let base: Vec<i32> = (0..12)
            .map(|i| [i32::MIN, i32::MAX, 17, -33][i % 4])
            .collect();
        let values: Vec<i32> = (0..16)
            .map(|i| [1, -1, i32::MIN, i32::MAX, 7, -7][i % 6])
            .collect();
        let indices: Vec<i32> = (0..16).map(|i| [0, 2, 0, 1][i / 2 % 4]).collect();
        let t =
            Tensor::<TtBackend, 3, Int>::from_data(TensorData::new(base.clone(), [2, 3, 2]), &d)
                .add_scalar(0);
        let v =
            Tensor::<TtBackend, 3, Int>::from_data(TensorData::new(values.clone(), [2, 4, 2]), &d)
                .add_scalar(0);
        let idx =
            Tensor::<TtBackend, 3, Int>::from_data(TensorData::new(indices.clone(), [2, 4, 2]), &d)
                .add_scalar(0);
        let one = Tensor::<TtBackend, 1, Int>::from_data([0, 2, 0, 1], &d).add_scalar(0);
        let bool_base: Vec<_> = (0..12).map(|i| i % 5 == 0).collect();
        let bool_values: Vec<_> = (0..16).map(|i| i % 3 == 1).collect();
        let bt = Tensor::<TtBackend, 3, Bool>::from_data(
            TensorData::new(bool_base.clone(), [2, 3, 2]),
            &d,
        )
        .to_device(&d);
        let bv = Tensor::<TtBackend, 3, Bool>::from_data(
            TensorData::new(bool_values.clone(), [2, 4, 2]),
            &d,
        )
        .to_device(&d);
        let before = tensor_traffic();
        let ((a, b, c, e), report) = burn_tt::with_report(|| {
            let a = <TtBackend as IntTensorOps<TtBackend>>::int_scatter_add(
                1,
                t.clone().into_primitive(),
                idx.clone().into_primitive(),
                v.clone().into_primitive(),
            );
            let b = <TtBackend as IntTensorOps<TtBackend>>::int_select_add(
                t.into_primitive(),
                1,
                one.clone().into_primitive(),
                v.into_primitive(),
            );
            let c = <TtBackend as BoolTensorOps<TtBackend>>::bool_scatter_or(
                1,
                bt.clone().into_primitive(),
                idx.into_primitive(),
                bv.clone().into_primitive(),
            );
            let e = <TtBackend as BoolTensorOps<TtBackend>>::bool_select_or(
                bt.into_primitive(),
                1,
                one.into_primitive(),
                bv.into_primitive(),
            );
            for p in [&a, &b, &c, &e] {
                assert!(p.computed_on_device());
            }
            (a, b, c, e)
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let mut expected = base;
        let mut bool_expected = bool_base;
        for i in 0..16 {
            let at = (i / 8 * 3 + indices[i] as usize) * 2 + i % 2;
            expected[at] = expected[at].wrapping_add(values[i]);
            bool_expected[at] |= bool_values[i];
        }
        assert_eq!(
            Tensor::<TtBackend, 3, Int>::from_primitive(a)
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            expected
        );
        assert_eq!(
            Tensor::<TtBackend, 3, Int>::from_primitive(b)
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            expected
        );
        assert_eq!(
            Tensor::<TtBackend, 3, Bool>::from_primitive(c)
                .into_data()
                .to_vec::<bool>()
                .unwrap(),
            bool_expected
        );
        assert_eq!(
            Tensor::<TtBackend, 3, Bool>::from_primitive(e)
                .into_data()
                .to_vec::<bool>()
                .unwrap(),
            bool_expected
        );
    });
}

#[test]
fn integer_and_boolean_updates_replay_device_produced_indices() {
    use burn::tensor::Bool;
    use burn_tensor::ops::{BoolTensorOps, IntTensorOps};
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            let input = Tensor::<TtBackend, 2>::from_data([[0.0]], &d).mul_scalar(1.0);
            let TensorPrimitive::Float(primitive) = input.clone().into_primitive() else {
                unreachable!()
            };
            let base = Tensor::<TtBackend, 2, Int>::from_data(
                TensorData::new(vec![i32::MAX; 35], [1, 35]),
                &d,
            )
            .add_scalar(0);
            let value = Tensor::<TtBackend, 2, Int>::ones([1, 1], &d).add_scalar(0);
            let bits = Tensor::<TtBackend, 2, Bool>::from_data(
                TensorData::new(vec![false; 35], [1, 35]),
                &d,
            )
            .to_device(&d);
            let truth = Tensor::<TtBackend, 2, Bool>::from_data([[true]], &d).to_device(&d);
            let ((trace, first), report) = burn_tt::with_report(|| {
                burn_tt::Trace::capture(&primitive, || {
                    let index = input.clone().int().into_primitive();
                    let ints = <TtBackend as IntTensorOps<TtBackend>>::int_scatter_add(
                        1,
                        base.clone().into_primitive(),
                        index.clone(),
                        value.clone().into_primitive(),
                    );
                    let bools = <TtBackend as BoolTensorOps<TtBackend>>::bool_scatter_or(
                        1,
                        bits.clone().into_primitive(),
                        index,
                        truth.clone().into_primitive(),
                    );
                    let output = Tensor::cat(
                        vec![
                            Tensor::<TtBackend, 2, Int>::from_primitive(ints).float(),
                            Tensor::<TtBackend, 2, Bool>::from_primitive(bools).float(),
                        ],
                        1,
                    );
                    let TensorPrimitive::Float(output) = output.into_primitive() else {
                        unreachable!()
                    };
                    output
                })
                .unwrap()
            });
            assert_native_model(&report);
            for column in [0, 15, 16, 31, 32, 34] {
                let mut expected = vec![2147483648.0; 35];
                expected[column] = -2147483648.0;
                expected.extend((0..35).map(|c| if c == column { 1.0 } else { 0.0 }));
                let output = if column == 0 {
                    first.clone()
                } else {
                    trace.run(vec![column as f32]).unwrap()
                };
                assert_eq!(output, expected, "device-produced index {column}");
            }
        },
    );
}
