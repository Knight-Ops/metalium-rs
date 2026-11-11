//! Native BF16 storage/views work in ttsim; narrowing arithmetic needs silicon.
#[cfg(feature = "silicon")]
use burn::tensor::FloatDType;
use burn::tensor::{DType, Tensor, TensorData, TensorPrimitive};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn data(bits: Vec<u16>, shape: impl Into<burn::tensor::Shape>) -> TensorData {
    let mut data = TensorData::new(bits, shape);
    data.dtype = DType::BF16;
    data
}

fn bits(data: TensorData) -> Vec<u16> {
    assert_eq!(data.dtype, DType::BF16);
    let mut data = data;
    data.dtype = DType::U16;
    data.to_vec().unwrap()
}

fn resident<const D: usize>(t: &Tensor<TtBackend, D>) {
    match t.clone().into_primitive() {
        TensorPrimitive::Float(t) => assert!(t.computed_on_device()),
        _ => panic!("unexpected quantized tensor"),
    }
}

#[test]
fn raw_bf16_views_preserve_all_payloads_and_use_two_byte_transfers() {
    with_device(Config::default(), |d| {
        let (rows, cols) = (37, 70);
        let values: Vec<_> = (0..rows * cols)
            .map(|i| (i as u16).wrapping_mul(113))
            .collect();
        let before = tensor_traffic();
        let input = Tensor::<TtBackend, 2>::from_data(
            data(values.clone(), [rows, cols]),
            (&d, DType::BF16),
        )
        .to_device(&d);
        assert_eq!(
            (tensor_traffic() - before).uploaded,
            (rows * cols * 2) as u64
        );
        let (output, report) = burn_tt::with_report(|| {
            let output = input.clone().transpose().reshape([1, rows * cols]);
            resident(&output);
            let flipped = input.clone().flip([0, 1]);
            resident(&flipped);
            let tail = input.clone().slice([32..37, 0..cols]);
            resident(&tail);
            assert_eq!(tensor_traffic().downloads, before.downloads);
            (output, flipped, tail)
        });
        assert_native_model(&report);
        let want: Vec<_> = (0..cols)
            .flat_map(|c| {
                let values = &values;
                (0..rows).map(move |r| values[r * cols + c])
            })
            .collect();
        assert_eq!(bits(output.0.into_data()), want);
        assert_eq!(
            bits(output.1.into_data()),
            values.iter().rev().copied().collect::<Vec<_>>()
        );
        assert_eq!(bits(output.2.into_data()), values[32 * cols..]);
        assert_eq!(
            bits(input.into_data()),
            values,
            "views must not modify their parent"
        );
    });
}

#[cfg(feature = "silicon")]
fn round(value: f32) -> f32 {
    let bits = value.to_bits();
    let sign = bits & 0x80000000;
    let magnitude = bits & 0x7fffffff;
    if magnitude > 0x7f800000 {
        return f32::from_bits((bits | 0x00400000) & 0xffff0000);
    }
    // Independent f64 grid quantization rather than the SFPU's bit bias.
    let exponent = ((bits >> 23) & 255) as i32 - 127;
    let quantum = 2f64.powi(exponent.max(-126) - 7);
    let rounded = (value.abs() as f64 / quantum).round_ties_even() * quantum;
    let bits = if rounded < 2f64.powi(-126) {
        sign
    } else {
        sign | (rounded as f32).to_bits()
    };
    f32::from_bits(bits)
}

/// Roadmap W1, native BF16 arithmetic: a BF16 element-wise op and a BF16 product run on their BF16
/// operands as they lie and round their results once (`Session::eltwise3`,
/// `Session::matmul_dram_as`) -- no cast job -- bit for bit Burn's explicit composition
/// `op(x.cast(F32)).cast(BF16)`, over special values, ragged shapes and a column broadcast. A row
/// broadcast, which the kernel has no BF16 form of, widens and narrows, and says so in its casts.
#[cfg(feature = "silicon")]
#[test]
fn bf16_ops_run_natively_bit_for_bit_their_widened_composition() {
    use burn::tensor::activation::relu;
    with_device(Config::default(), |d| {
        let (m, k, n) = (67, 300, 70);
        let pattern = |len: usize, salt: u32| -> Vec<u16> {
            const SPECIAL: [u16; 10] = [
                0x0000, 0x8000, 0x0001, 0x807f, 0x7f80, 0xff80, 0x7fc0, 0x7f7f, 0x3f80, 0xbf81,
            ];
            (0..len as u32)
                .map(|i| {
                    let h = (i ^ salt.wrapping_mul(0x9e37_79b9)).wrapping_mul(2_654_435_761);
                    if i % 11 == 0 {
                        SPECIAL[(h >> 7) as usize % SPECIAL.len()]
                    } else {
                        ((h >> 31) as u16) << 15
                            | ((0x70 + (h >> 8) % 0x18) as u16) << 7
                            | (h & 0x7f) as u16
                    }
                })
                .collect()
        };
        let tensor = |bits: Vec<u16>, shape: [usize; 2]| {
            Tensor::<TtBackend, 2>::from_data(data(bits, shape), (&d, DType::BF16))
        };
        let (x, y) = (
            tensor(pattern(m * k, 1), [m, k]),
            tensor(pattern(m * k, 2), [m, k]),
        );
        let col = tensor(pattern(m, 3), [m, 1]);
        let row = tensor(pattern(k, 4), [1, k]);
        let w = tensor(
            pattern(k * n, 5)
                .iter()
                .map(|&b| b & 0x807f | 0x3c00)
                .collect(),
            [k, n],
        );
        type Op = fn(&[Tensor<TtBackend, 2>]) -> Tensor<TtBackend, 2>;
        // (what, op, the device jobs it queues: one an op natively, a cast more a widened operand
        // and a narrowed result)
        let cases: [(&str, Op, u64); 13] = [
            ("add", |t| t[0].clone() + t[1].clone(), 1),
            ("sub", |t| t[0].clone() - t[1].clone(), 1),
            ("mul", |t| t[0].clone() * t[1].clone(), 1),
            ("div", |t| t[0].clone() / t[1].clone(), 1),
            ("mul_scalar", |t| t[0].clone().mul_scalar(0.3), 1),
            ("add_scalar", |t| t[0].clone().add_scalar(1.5), 1),
            ("exp", |t| t[0].clone().exp(), 1),
            ("sqrt", |t| t[0].clone().abs().sqrt(), 2),
            ("relu", |t| relu(t[0].clone()), 1),
            (
                "sigmoid",
                |t| burn::tensor::activation::sigmoid(t[0].clone()),
                1,
            ),
            ("add a column", |t| t[0].clone() + t[2].clone(), 1),
            // Widened: two inputs and the result, three casts.
            ("mul a row", |t| t[0].clone() * t[3].clone(), 4),
            ("matmul", |t| t[0].clone().matmul(t[4].clone()), 1),
        ];
        let inputs = [x, y, col, row, w];
        let wide: Vec<_> = inputs
            .iter()
            .map(|t| t.clone().cast(FloatDType::F32))
            .collect();
        for (what, op, jobs) in cases {
            let (native, report) = burn_tt::with_report(|| op(&inputs));
            resident(&native);
            assert_native_model(&report);
            let queued: u64 = report.0.iter().map(|s| s.jobs).sum();
            assert_eq!(queued, jobs, "{what}: device jobs ({report:?})");
            assert_eq!(native.dtype(), DType::BF16, "{what}");
            let composed = op(&wide).cast(FloatDType::BF16);
            let (got, want) = (bits(native.into_data()), bits(composed.into_data()));
            let differ = (0..got.len()).filter(|&i| got[i] != want[i]).count();
            assert_eq!(differ, 0, "{what}: {differ} of {} differ", got.len());
        }
    });
}

/// Roadmap W1: RMSNorm on a BF16 `x` and scale (`burn_tt::rms_norm`) is Burn's own module bit for
/// bit -- the statistic in F32, cast to BF16, then `x / rms` and `* gamma` each rounded once -- at a
/// decode row and a ragged block of rows, in fewer jobs than the module's composition.
#[cfg(feature = "silicon")]
#[test]
fn bf16_rms_norm_is_the_module_bit_for_bit() {
    use burn::nn::RmsNormConfig;
    with_device(Config::default(), |d| {
        let cols = 2048;
        for rows in [1usize, 67] {
            let values: Vec<f32> = (0..rows * cols)
                .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 9) as f32 / 4.0e6 - 1.0)
                .collect();
            let x = Tensor::<TtBackend, 3>::from_data(TensorData::new(values, [1, rows, cols]), &d)
                .cast(FloatDType::BF16);
            let mut norm = RmsNormConfig::new(cols)
                .with_epsilon(1e-5)
                .init::<TtBackend>(&d);
            let scale: Vec<f32> = (0..cols).map(|i| 0.5 + (i % 37) as f32 / 64.0).collect();
            norm.gamma = burn::module::Param::from_tensor(
                Tensor::<TtBackend, 1>::from_data(TensorData::new(scale, [cols]), &d)
                    .cast(FloatDType::BF16),
            );
            let (fused, report) =
                burn_tt::with_report(|| burn_tt::rms_norm(&x, &norm.gamma.val(), 1e-5).unwrap());
            assert_native_model(&report);
            let jobs: u64 = report.0.iter().map(|s| s.jobs).sum();
            let (module, composed) = burn_tt::with_report(|| norm.forward(x.clone()));
            let module_jobs: u64 = composed.0.iter().map(|s| s.jobs).sum();
            println!(
                "MEASURE bf16 rms_norm [{rows}, {cols}]: {jobs} jobs, the module's {module_jobs}"
            );
            assert!(
                jobs < module_jobs,
                "[{rows}, {cols}]: {jobs} jobs against {module_jobs}"
            );
            assert_eq!(fused.dtype(), DType::BF16);
            assert_eq!(
                bits(fused.into_data()),
                bits(module.into_data()),
                "[{rows}, {cols}]"
            );
        }
    });
}

/// Roadmap W1: a logical BF16 weight stored BFP8 (`with_storage`) decodes back to BF16 values
/// exactly -- the values of the F32 tensor stored BFP8 -- and a BF16 activation's product with it
/// is one job with a BF16 result, bit for bit the widened product narrowed.
#[cfg(feature = "silicon")]
#[test]
fn a_bf16_weight_stored_bfp8_meets_a_bf16_activation() {
    use burn_tt::storage::{StorageFormat, TensorStorageExt};
    with_device(Config::default(), |d| {
        let (m, k, n) = (1, 2048, 384);
        let values = |len: usize, salt: u32| -> Vec<f32> {
            (0..len as u32)
                .map(|i| {
                    let h = (i ^ salt.wrapping_mul(0x9e37_79b9)).wrapping_mul(2_654_435_761);
                    ((h >> 8) % 2001) as f32 / 1000.0 - 1.0
                })
                .collect()
        };
        let f32s = |v: Vec<f32>, shape: [usize; 2]| {
            Tensor::<TtBackend, 2>::from_data(TensorData::new(v, shape), &d)
        };
        let x = f32s(values(m * k, 1), [m, k]).cast(FloatDType::BF16);
        let w = f32s(values(k * n, 2), [k, n]).cast(FloatDType::BF16);
        let stored = w.clone().with_storage(StorageFormat::Bfp8);
        assert_eq!(stored.dtype(), DType::BF16);
        assert_eq!(stored.storage_format(), StorageFormat::Bfp8);
        // Decoded, the BF16 tensor holds the values the F32 one stored BFP8 decodes to.
        let decoded = bits(stored.clone().with_storage(StorageFormat::F32).into_data());
        let via_f32: Vec<f32> = w
            .clone()
            .cast(FloatDType::F32)
            .with_storage(StorageFormat::Bfp8)
            .with_storage(StorageFormat::F32)
            .into_data()
            .to_vec()
            .unwrap();
        let exact = via_f32
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        assert!(
            via_f32.iter().all(|v| v.to_bits() & 0xffff == 0),
            "BFP8 values are BF16 values"
        );
        assert_eq!(decoded, exact);
        let (native, report) = burn_tt::with_report(|| x.clone().matmul(stored.clone()));
        resident(&native);
        assert_native_model(&report);
        let queued: u64 = report.0.iter().map(|s| s.jobs).sum();
        assert!(queued <= 2, "one product (and a packed row): {report:?}");
        assert_eq!(native.dtype(), DType::BF16);
        let widened = f32s(via_f32, [k, n]);
        let want = bits(
            x.cast(FloatDType::F32)
                .matmul(widened)
                .cast(FloatDType::BF16)
                .into_data(),
        );
        assert_eq!(bits(native.into_data()), want);
    });
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_arithmetic_matmul_reductions_and_gradients_stay_resident() {
    use burn::backend::Autodiff;
    use burn::tensor::{Bool, Int};
    with_device(Config::default(), |d| {
        let ints = Tensor::<TtBackend, 1, Int>::from_data([3, -2, 0], &d);
        let bools = Tensor::<TtBackend, 1, Bool>::from_data([true, false, true], &d);
        let ((ints, bools), report) =
            burn_tt::with_report(|| (ints.cast(FloatDType::BF16), bools.cast(FloatDType::BF16)));
        assert_native_model(&report);
        resident(&ints);
        resident(&bools);
        assert_eq!(bits(ints.into_data()), vec![0x4040, 0xc000, 0]);
        assert_eq!(bits(bools.into_data()), vec![0x3f80, 0, 0x3f80]);
        let values: Vec<_> = (0..37 * 35)
            .map(|i| ((i * 13) % 31) as f32 / 16.0 - 1.0)
            .collect();
        let x = Tensor::<TtBackend, 2>::from_data(TensorData::new(values.clone(), [37, 35]), &d)
            .cast(FloatDType::BF16);
        resident(&x);
        let before = tensor_traffic();
        let (outputs, report) = burn_tt::with_report(|| {
            let outputs = [
                x.clone().mul_scalar(1.25),
                x.clone().sum_dim(1),
                x.clone().mean_dim(0),
            ];
            for output in &outputs {
                resident(output);
            }
            assert_eq!(tensor_traffic().downloads, before.downloads);
            outputs
        });
        assert_native_model(&report);
        let expected: Vec<_> = values
            .iter()
            .map(|v| (round(round(*v) * 1.25).to_bits() >> 16) as u16)
            .collect();
        assert_eq!(bits(outputs[0].clone().into_data()), expected);
        // Values and all intermediate sums are binary rationals exactly
        // representable in F32, so the independent scalar order is exact.
        let sums: Vec<_> = values
            .chunks(35)
            .map(|row| (round(row.iter().sum()).to_bits() >> 16) as u16)
            .collect();
        assert_eq!(bits(outputs[1].clone().into_data()), sums);
        let expected: Vec<_> = (0..35)
            .map(|c| {
                let sum: f32 = (0..37).map(|r| values[r * 35 + c]).sum();
                (round(sum / 37.0).to_bits() >> 16) as u16
            })
            .collect();
        assert_eq!(bits(outputs[2].clone().into_data()), expected);

        let (m, k, n) = (37, 65, 35);
        let a: Vec<_> = (0..m * k).map(|i| ((i * 3) % 7) as f32 - 3.0).collect();
        let b: Vec<_> = (0..k * n).map(|i| ((i * 5) % 7) as f32 - 3.0).collect();
        let lhs = Tensor::<TtBackend, 2>::from_data(TensorData::new(a.clone(), [m, k]), &d)
            .cast(FloatDType::BF16);
        let rhs = Tensor::<TtBackend, 2>::from_data(TensorData::new(b.clone(), [k, n]), &d)
            .cast(FloatDType::BF16);
        let (out, report) = burn_tt::with_report(|| lhs.matmul(rhs));
        resident(&out);
        assert_native_model(&report);
        let want: Vec<_> = (0..m)
            .flat_map(|r| {
                let (a, b) = (&a, &b);
                (0..n).map(move |c| {
                    (round((0..k).map(|q| a[r * k + q] * b[q * n + c]).sum()).to_bits() >> 16)
                        as u16
                })
            })
            .collect();
        assert_eq!(bits(out.into_data()), want);

        type AD = Autodiff<TtBackend>;
        let input = Tensor::<AD, 2>::from_data([[1.0, -2.0], [3.0, -4.0]], &d)
            .cast(FloatDType::BF16)
            .require_grad();
        let (grad, report) = burn_tt::with_report(|| {
            let grads = (input.clone() * input.clone()).sum().backward();
            input.grad(&grads).unwrap()
        });
        resident(&grad);
        assert_native_model(&report);
        assert_eq!(
            bits(grad.into_data()),
            [2.0f32, -4.0, 6.0, -8.0].map(|v| (v.to_bits() >> 16) as u16)
        );
    });
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_layer_and_rms_norm_forward_and_backward_are_native() {
    use burn::{
        backend::Autodiff,
        module::Param,
        nn::{LayerNormConfig, RmsNormConfig},
    };
    type AD = Autodiff<TtBackend>;
    with_device(Config::default(), |d| {
        let input = Tensor::<AD, 2>::from_data([[-1.0, 1.0], [-1.0, 1.0]], (&d, DType::BF16))
            .require_grad();
        let weights = Tensor::<AD, 2>::from_data([[2.0, 0.0], [2.0, 0.0]], (&d, DType::BF16));
        let mut layer = LayerNormConfig::new(2).with_epsilon(3.0).init::<AD>(&d);
        layer.gamma = Param::from_tensor(Tensor::from_data([1.0, 1.0], (&d, DType::BF16)));
        layer.beta = Some(Param::from_tensor(Tensor::from_data(
            [0.0, 0.0],
            (&d, DType::BF16),
        )));
        let mut rms = RmsNormConfig::new(2).with_epsilon(3.0).init::<AD>(&d);
        rms.gamma = Param::from_tensor(Tensor::from_data([1.0, 1.0], (&d, DType::BF16)));
        for rms_mode in [false, true] {
            let ((out, grad), report) = burn_tt::with_report(|| {
                let out = if rms_mode {
                    rms.forward(input.clone())
                } else {
                    layer.forward(input.clone())
                };
                let grads = (out.clone() * weights.clone()).sum().backward();
                (out, input.grad(&grads).unwrap())
            });
            assert_native_model(&report);
            resident(&out.clone().inner());
            resident(&grad);
            assert_eq!(
                bits(out.into_data()),
                [-0.5f32, 0.5, -0.5, 0.5].map(|v| (v.to_bits() >> 16) as u16)
            );
            // Analytic derivatives: variance=1, epsilon=3, denominator=2.
            // Layer: (dy-mean(dy))/2 - x*mean(dy*x)/8.
            // RMS: dy/2 - x*mean(dy*x)/8. All are exact binary rationals.
            let want = if rms_mode {
                [0.875f32, 0.125, 0.875, 0.125]
            } else {
                [0.375f32, -0.375, 0.375, -0.375]
            };
            assert_eq!(
                bits(grad.into_data()),
                want.map(|v| (v.to_bits() >> 16) as u16)
            );
        }
    });
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_mlp_sgd_step_with_f32_loss_stays_resident() {
    use burn::{
        backend::Autodiff,
        module::Param,
        nn::Linear,
        optim::{GradientsParams, Optimizer, SgdConfig},
    };
    type AD = Autodiff<TtBackend>;
    with_device(Config::default(), |d| {
        let mut model = Linear::<AD> {
            weight: Param::from_tensor(Tensor::from_data([[1.0], [2.0]], (&d, DType::BF16))),
            bias: Some(Param::from_tensor(Tensor::from_data(
                [0.0],
                (&d, DType::BF16),
            ))),
        };
        let input = Tensor::<AD, 2>::from_data([[1.0, 1.0], [1.0, 1.0]], (&d, DType::BF16));
        let mut optimizer = SgdConfig::new().init();
        let ((output, weight, bias), report) = burn_tt::with_report(|| {
            let output = model.forward(input);
            let loss = output.clone().cast(FloatDType::F32).square().mean();
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(0.125, model, gradients);
            (
                output,
                model.weight.val().inner(),
                model.bias.unwrap().val().inner(),
            )
        });
        assert_native_model(&report);
        resident(&weight);
        resident(&bias);
        // y=3, mean(y^2)=9, dw=db=6. SGD subtracts 0.75 exactly.
        assert_eq!(bits(output.into_data()), vec![0x4040; 2]);
        assert_eq!(bits(weight.into_data()), vec![0x3e80, 0x3fa0]);
        assert_eq!(bits(bias.into_data()), vec![0xbf40]);
    });
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_trace_replays_changed_inputs_and_holds_temporary_storage() {
    use burn_tt::Trace;
    with_device(Config::default(), |d| {
        let input = Tensor::<TtBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &d).to_device(&d);
        let weight =
            Tensor::<TtBackend, 2>::from_data([[1.0], [2.0]], (&d, DType::BF16)).to_device(&d);
        let bias = Tensor::<TtBackend, 2>::from_data([[0.0]], (&d, DType::BF16)).to_device(&d);
        let TensorPrimitive::Float(primitive) = input.clone().into_primitive() else {
            unreachable!()
        };
        let (trace, first) = Trace::capture(&primitive, || {
            let linear = input.clone().cast(FloatDType::BF16).matmul(weight.clone()) + bias.clone();
            let out = burn::tensor::activation::relu(linear)
                .mul_scalar(0.5)
                .cast(FloatDType::F32);
            let TensorPrimitive::Float(out) = out.into_primitive() else {
                unreachable!()
            };
            out
        })
        .unwrap();
        assert_eq!(first, vec![2.5, 5.5]);
        drop(weight);
        drop(bias);
        for value in [-2.0, 0.0, 4.0] {
            assert_eq!(
                trace.run(vec![value; 4]).unwrap(),
                vec![value.max(0.0) * 1.5; 2]
            );
        }
        drop(trace);
        // Compute a fresh result to observe the device buffer, rather than the
        // input constructor's cached host copy.
        assert_eq!(
            input.mul_scalar(2.0).into_data().to_vec::<f32>().unwrap(),
            vec![8.0; 4]
        );
    });
}

#[cfg(feature = "silicon")]
#[test]
fn packed_bf16_batched_matmul_handles_broadcast_ragged_and_transposed_views() {
    with_device(Config::default(), |d| {
        for (batch, m, k, n) in [(2, 3, 7, 5), (2, 33, 35, 3)] {
            let a: Vec<_> = (0..batch * m * k).map(|i| (i % 5) as f32 - 2.0).collect();
            let b: Vec<_> = (0..batch * k * n).map(|i| (i % 7) as f32 - 3.0).collect();
            let lhs = Tensor::<TtBackend, 3>::from_data(
                TensorData::new(a.clone(), [batch, m, k]),
                (&d, DType::BF16),
            );
            let rhs = Tensor::<TtBackend, 3>::from_data(
                TensorData::new(b.clone(), [batch, k, n]),
                (&d, DType::BF16),
            );
            let before = tensor_traffic();
            let (out, report) = burn_tt::with_report(|| lhs.matmul(rhs));
            resident(&out);
            assert_native_model(&report);
            assert_eq!(tensor_traffic().downloads, before.downloads);
            let mut want = Vec::new();
            for q in 0..batch {
                for r in 0..m {
                    for c in 0..n {
                        let sum: f32 = (0..k)
                            .map(|i| a[(q * m + r) * k + i] * b[(q * k + i) * n + c])
                            .sum();
                        want.push((sum.to_bits() >> 16) as u16);
                    }
                }
            }
            assert_eq!(bits(out.into_data()), want);
        }
        // Rank-four broadcast, with physically transposed matrix operands.
        let a: Vec<_> = (0..2 * 3 * 5 * 7).map(|i| (i % 3) as f32 - 1.0).collect();
        let lhs = Tensor::<TtBackend, 4>::from_data(
            TensorData::new(a.clone(), [2, 3, 5, 7]),
            (&d, DType::BF16),
        )
        .swap_dims(2, 3);
        let weight = Tensor::<TtBackend, 4>::from_data(
            TensorData::new(vec![1.0f32; 20], [1, 1, 4, 5]),
            (&d, DType::BF16),
        )
        .swap_dims(2, 3);
        let (out, report) = burn_tt::with_report(|| lhs.matmul(weight));
        resident(&out);
        assert_native_model(&report);
        assert_eq!(out.dims(), [2, 3, 7, 4]);
        let want: Vec<_> = (0..6)
            .flat_map(|q| {
                let a = &a;
                (0..7).flat_map(move |r| {
                    let sum: f32 = (0..5).map(|i| a[(q * 5 + i) * 7 + r]).sum();
                    [(sum.to_bits() >> 16) as u16; 4]
                })
            })
            .collect();
        assert_eq!(bits(out.into_data()), want);
    });
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_mesh_product_uses_resident_ethernet_execution_across_two_cards() {
    tt_tests::burn_device::with_mesh_device(Config::default(), 2, |d| {
        let (m, k, n) = (35, 37, 225);
        let a: Vec<_> = (0..m * k).map(|i| (i % 5) as f32 - 2.0).collect();
        let b: Vec<_> = (0..k * n).map(|i| (i % 7) as f32 - 3.0).collect();
        let lhs = Tensor::<TtBackend, 2>::from_data(
            TensorData::new(a.clone(), [m, k]),
            (&d, DType::BF16),
        );
        let rhs = Tensor::<TtBackend, 2>::from_data(
            TensorData::new(b.clone(), [k, n]),
            (&d, DType::BF16),
        );
        let before = tensor_traffic();
        let (out, report) = burn_tt::with_report(|| lhs.matmul(rhs));
        resident(&out);
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let want: Vec<_> = (0..m)
            .flat_map(|r| {
                let (a, b) = (&a, &b);
                (0..n).map(move |c| {
                    let sum: f32 = (0..k).map(|i| a[r * k + i] * b[i * n + c]).sum();
                    (sum.to_bits() >> 16) as u16
                })
            })
            .collect();
        assert_eq!(bits(out.into_data()), want);
    });
}

/// Roadmap W1's BF16 activations: the ops a model's products read their BF16 `A` from, each with
/// its result's dtype named (`burn_tt::rms_norm_as`, `silu_mul_as`, `matmul_as`), are bit for bit
/// their compositions -- the F32 norm (`burn_tt::rms_norm`) then the cast, `silu(gate) * up` then
/// the cast, and the product of a BF16 `x` cast to F32 with an F32 weight stored BFP8 -- on
/// `TtBackend` and on `Tt` (each a custom operation in the fusion stream), at a decode row and at
/// 67 rows; on `TtBackend` in fewer jobs than the composition (no cast job). Silicon: a BF16 pack.
#[cfg(feature = "silicon")]
#[test]
fn ops_with_a_named_result_are_their_compositions_bit_for_bit() {
    use burn::tensor::activation::silu;
    use burn::tensor::backend::Backend;
    use burn_tt::storage::{StorageFormat, TensorStorageExt};
    use burn_tt::TtDevice;

    fn values(len: usize, salt: u32) -> Vec<f32> {
        (0..len as u32)
            .map(|i| {
                let h = (i ^ salt.wrapping_mul(0x9e37_79b9)).wrapping_mul(2_654_435_761);
                ((h >> 8) % 200_001) as f32 / 100_000.0 - 1.0 + (h & 0xff) as f32 * 1e-9
            })
            .collect()
    }
    fn f32_bits<const D: usize, B: Backend>(t: Tensor<B, D>) -> Vec<u32> {
        let data = t.into_data();
        assert_eq!(data.dtype, DType::F32);
        data.to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }
    /// The three ops and their compositions on `B`, `(rows, what, jobs named, jobs composed)` for
    /// each when `count`.
    fn check<B: burn_tt::storage::StorageBackend<Device = TtDevice>>(
        d: &TtDevice,
        count: bool,
    ) -> Vec<(usize, &'static str, u64, u64)> {
        let (cols, n) = (2048, 384);
        let mut jobs = Vec::new();
        let measured = |f: &mut dyn FnMut()| -> u64 {
            let ((), report) = burn_tt::with_report(&mut *f);
            report.0.iter().map(|s| s.jobs).sum()
        };
        let gamma = Tensor::<B, 1>::from_data(TensorData::new(values(cols, 9), [cols]), d);
        let w = Tensor::<B, 2>::from_data(TensorData::new(values(cols * n, 8), [cols, n]), d)
            .with_storage(StorageFormat::Bfp8);
        for rows in [1usize, 67] {
            let f32s = |salt| {
                Tensor::<B, 3>::from_data(
                    TensorData::new(values(rows * cols, salt), [1, rows, cols]),
                    d,
                )
            };
            let (x, gate, up) = (f32s(1), f32s(2), f32s(3));

            let named = burn_tt::rms_norm_as(&x, &gamma, 1e-5, DType::BF16).unwrap();
            let composed = burn_tt::rms_norm(&x, &gamma, 1e-5)
                .unwrap()
                .cast(FloatDType::BF16);
            assert_eq!(named.dtype(), DType::BF16);
            assert_eq!(
                bits(named.clone().into_data()),
                bits(composed.into_data()),
                "rms_norm_as, {rows} rows"
            );
            if count {
                let a = measured(&mut || {
                    burn_tt::rms_norm_as(&x, &gamma, 1e-5, DType::BF16)
                        .unwrap()
                        .into_data();
                });
                let b = measured(&mut || {
                    burn_tt::rms_norm(&x, &gamma, 1e-5)
                        .unwrap()
                        .cast(FloatDType::BF16)
                        .into_data();
                });
                jobs.push((rows, "rms_norm_as", a, b));
            }

            let named = burn_tt::silu_mul_as(&gate, &up, DType::BF16).unwrap();
            let composed = (silu(gate.clone()) * up.clone()).cast(FloatDType::BF16);
            assert_eq!(named.dtype(), DType::BF16);
            assert_eq!(
                bits(named.into_data()),
                bits(composed.into_data()),
                "silu_mul_as, {rows} rows"
            );
            if count {
                let a = measured(&mut || {
                    burn_tt::silu_mul_as(&gate, &up, DType::BF16)
                        .unwrap()
                        .into_data();
                });
                let b = measured(&mut || {
                    (silu(gate.clone()) * up.clone())
                        .cast(FloatDType::BF16)
                        .into_data();
                });
                jobs.push((rows, "silu_mul_as", a, b));
            }

            let h = burn_tt::rms_norm_as(&x, &gamma, 1e-5, DType::BF16).unwrap();
            let named = burn_tt::matmul_as(&h, &w, DType::F32).unwrap();
            let composed = h
                .clone()
                .cast(FloatDType::F32)
                .matmul(w.clone().unsqueeze());
            assert_eq!(named.dims(), [1, rows, n]);
            assert_eq!(
                f32_bits(named),
                f32_bits(composed),
                "matmul_as, {rows} rows"
            );
            if count {
                let a = measured(&mut || {
                    burn_tt::matmul_as(&h, &w, DType::F32).unwrap().into_data();
                });
                let b = measured(&mut || {
                    h.clone()
                        .cast(FloatDType::F32)
                        .matmul(w.clone().unsqueeze())
                        .into_data();
                });
                jobs.push((rows, "matmul_as", a, b));
            }
        }
        jobs
    }
    with_device(Config::default(), |d| {
        let jobs = check::<TtBackend>(&d, true);
        for (rows, what, a, b) in &jobs {
            println!("MEASURE {what}, {rows} rows: {a} jobs, the composition's {b}");
            assert!(
                a < b,
                "{what}, {rows} rows: {a} jobs against the composition's {b}"
            );
        }
        check::<burn_tt::Tt>(&d, false);
    });
}
