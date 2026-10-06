//! Analytic resident attention cases; no intermediate tensor readback.
use burn::{
    backend::Autodiff,
    tensor::{module::attention, Bool, Tensor, TensorData},
};
use burn_tensor::{ops::AttentionModuleOptions, TensorPrimitive};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn resident(t: Tensor<TtBackend, 4>) -> bool {
    match t.into_primitive() {
        TensorPrimitive::Float(t) => t.computed_on_device(),
        _ => false,
    }
}

fn options(causal: bool) -> AttentionModuleOptions {
    AttentionModuleOptions {
        scale: Some(1.0),
        softcap: None,
        is_causal: causal,
    }
}

#[test]
fn ragged_attention_has_bottom_right_causality_and_zero_masked_rows() {
    with_device(Config::default(), |d| {
        // Zero logits yield exact uniform probabilities over each allowed set.
        let q = Tensor::<TtBackend, 4>::zeros([1, 2, 3, 5], &d);
        let k = Tensor::<TtBackend, 4>::zeros([1, 2, 5, 5], &d);
        let values: Vec<f32> = (0..2)
            .flat_map(|h| {
                (0..5).flat_map(move |r| (0..3).map(move |c| (12 * r + 3 * c + 60 * h) as f32))
            })
            .collect();
        let v = Tensor::<TtBackend, 4>::from_data(TensorData::new(values, [1, 2, 5, 3]), &d);
        let mask = Tensor::<TtBackend, 4, Bool>::from_data(
            TensorData::new(
                (0..15).map(|i| i / 5 == 1).collect::<Vec<_>>(),
                [1, 1, 3, 5],
            ),
            &d,
        );
        let before = tensor_traffic();
        let (out, report) =
            burn_tt::with_report(|| attention(q, k, v, Some(mask), None, options(true)));
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        assert!(resident(out.clone()));
        let expected: Vec<f32> = (0..2)
            .flat_map(|h| {
                (0..3).flat_map(move |r| {
                    (0..3).map(move |c| {
                        if r == 1 {
                            0.0
                        } else {
                            (6 * (r + 2) + 3 * c + 60 * h) as f32
                        }
                    })
                })
            })
            .collect();
        assert_ne!(expected, vec![0.0; expected.len()], "all-zero mutant");
        // exp(0) and the integer V entries are exact. Division is within
        // one reciprocal ulp; TF32 truncation of the probability adds <2^-10
        // relative error. Positive accumulation adds at most four F32 ulps.
        for (got, want) in out
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .zip(expected)
        {
            assert!(
                (got - want).abs() <= want.abs() * (1.0 / 1024.0 + 4.0 * f32::EPSILON),
                "{got} vs {want}"
            );
        }
    });
}

#[test]
fn attention_autodiff_keeps_q_k_v_and_bias_gradients_resident() {
    type AD = Autodiff<TtBackend>;
    with_device(Config::default(), |d| {
        let q = Tensor::<AD, 4>::zeros([1, 1, 1, 1], &d).require_grad();
        let k = Tensor::<AD, 4>::from_data([[[[1.0], [3.0]]]], &d).require_grad();
        let v = Tensor::<AD, 4>::from_data([[[[2.0], [6.0]]]], &d).require_grad();
        let bias = Tensor::<AD, 4>::zeros([1, 1, 1, 2], &d).require_grad();
        let before = tensor_traffic();
        let ((out, grads), report) = burn_tt::with_report(|| {
            let out = attention(
                q.clone(),
                k.clone(),
                v.clone(),
                None,
                Some(bias.clone()),
                options(false),
            );
            let grads = out.clone().sum().backward();
            (out, grads)
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let dq = q.grad(&grads).unwrap();
        let dk = k.grad(&grads).unwrap();
        let dv = v.grad(&grads).unwrap();
        let db = bias.grad(&grads).unwrap();
        for t in [&dq, &dk, &dv, &db] {
            assert!(resident(t.clone()));
        }
        assert_eq!(out.into_data().to_vec::<f32>().unwrap(), [4.0]);
        assert_eq!(dq.into_data().to_vec::<f32>().unwrap(), [2.0]);
        assert_eq!(dk.into_data().to_vec::<f32>().unwrap(), [0.0, 0.0]);
        assert_eq!(dv.into_data().to_vec::<f32>().unwrap(), [0.5, 0.5]);
        assert_eq!(db.into_data().to_vec::<f32>().unwrap(), [-1.0, 1.0]);
    });
}

#[test]
fn causal_attention_trace_replays_changed_values_after_temporary_frees() {
    with_device(Config::default(), |d| {
        let q = Tensor::<TtBackend, 4>::zeros([1, 1, 2, 3], &d).mul_scalar(1.0);
        let k = q.clone();
        let v = Tensor::<TtBackend, 4>::ones([1, 1, 2, 3], &d).mul_scalar(1.0);
        let TensorPrimitive::Float(input) = v.clone().into_primitive() else {
            unreachable!()
        };
        let before = tensor_traffic();
        let ((trace, first), report) = burn_tt::with_report(|| {
            burn_tt::Trace::capture(&input, || {
                let out = attention(q.clone(), k.clone(), v.clone(), None, None, options(true));
                let TensorPrimitive::Float(out) = out.into_primitive() else {
                    unreachable!()
                };
                out
            })
            .unwrap()
        });
        assert_native_model(&report);
        assert_eq!(
            tensor_traffic().uploads,
            before.uploads,
            "causal geometry was uploaded"
        );
        assert_eq!(first, [1.0; 6]);
        for value in [-2.0, 0.0, 4.0] {
            assert_eq!(trace.run(vec![value; 6]).unwrap(), [value; 6]);
        }
    });
}

#[test]
fn mesh_attention_dispatches_batched_and_ragged_products() {
    tt_tests::burn_device::with_mesh_device(Config::default(), 2, |d| {
        for sq in [32, 3] {
            let q = Tensor::<TtBackend, 4>::zeros([1, 2, sq, 32], &d);
            let k = Tensor::<TtBackend, 4>::zeros([1, 2, 64, 32], &d);
            let values: Vec<f32> = (0..2)
                .flat_map(|h| {
                    (0..64).flat_map(move |r| (0..64).map(move |c| (2 * r + c + 100 * h) as f32))
                })
                .collect();
            let v = Tensor::<TtBackend, 4>::from_data(TensorData::new(values, [1, 2, 64, 64]), &d);
            let before = tensor_traffic();
            let execution = burn_tt::mesh_execution(d).unwrap();
            let (out, report) =
                burn_tt::with_report(|| attention(q, k, v, None, None, options(false)));
            assert_native_model(&report);
            assert_eq!(tensor_traffic().downloads, before.downloads);
            assert!(resident(out.clone()));
            let after = burn_tt::mesh_execution(d).unwrap();
            for card in 0..2 {
                assert!(after.completed_matmuls[card] > execution.completed_matmuls[card]);
            }
            assert!(after.acknowledged_ethernet_bytes > execution.acknowledged_ethernet_bytes);
            let expected: Vec<f32> = (0..2)
                .flat_map(|h| {
                    (0..sq).flat_map(move |_| (0..64).map(move |c| (63 + c + 100 * h) as f32))
                })
                .collect();
            assert_eq!(out.into_data().to_vec::<f32>().unwrap(), expected);
        }
    });
}

fn mesh_attention_gradients(dtype: burn::tensor::FloatDType) {
    tt_tests::burn_device::with_mesh_device(Config::default(), 2, |d| {
        type AD = Autodiff<TtBackend>;
        let q = Tensor::<AD, 4>::zeros([1, 1, 3, 64], &d)
            .cast(dtype)
            .require_grad();
        let k = Tensor::<AD, 4>::zeros([1, 1, 64, 64], &d)
            .cast(dtype)
            .require_grad();
        let v = Tensor::<AD, 4>::ones([1, 1, 64, 64], &d)
            .cast(dtype)
            .require_grad();
        let bias = Tensor::<AD, 4>::zeros([1, 1, 3, 64], &d)
            .cast(dtype)
            .require_grad();
        let before = tensor_traffic();
        let execution = burn_tt::mesh_execution(d).unwrap();
        let ((out, gq, gk, gv, gb), report) = burn_tt::with_report(|| {
            let out = attention(
                q.clone(),
                k.clone(),
                v.clone(),
                None,
                Some(bias.clone()),
                options(false),
            );
            let grads = out.clone().sum().backward();
            (
                out.inner(),
                q.grad(&grads).unwrap(),
                k.grad(&grads).unwrap(),
                v.grad(&grads).unwrap(),
                bias.grad(&grads).unwrap(),
            )
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let after = burn_tt::mesh_execution(d).unwrap();
        for card in 0..2 {
            assert!(after.completed_matmuls[card] >= execution.completed_matmuls[card] + 6);
        }
        assert!(after.acknowledged_ethernet_packets > execution.acknowledged_ethernet_packets);
        assert!(after.acknowledged_ethernet_bytes > execution.acknowledged_ethernet_bytes);
        assert_eq!(
            out.cast(burn::tensor::FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            vec![1.0; 192]
        );
        assert_eq!(
            gq.cast(burn::tensor::FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            vec![0.0; 192]
        );
        assert_eq!(
            gk.cast(burn::tensor::FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            vec![0.0; 4096]
        );
        assert_eq!(
            gv.cast(burn::tensor::FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            vec![3.0 / 64.0; 4096]
        );
        assert_eq!(
            gb.cast(burn::tensor::FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            vec![0.0; 192]
        );
    });
}

#[test]
fn mesh_attention_backward_computes_on_both_cards() {
    mesh_attention_gradients(burn::tensor::FloatDType::F32);
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_mesh_attention_backward_computes_on_both_cards() {
    mesh_attention_gradients(burn::tensor::FloatDType::BF16);
}

#[test]
fn attention_custom_scale_softcap_and_bias_follow_pinned_order() {
    with_device(Config::default(), |d| {
        let q = Tensor::<TtBackend, 4>::from_data([[[[1.0]]]], &d);
        let k = Tensor::<TtBackend, 4>::from_data([[[[-1.0], [1.0]]]], &d);
        let v = Tensor::<TtBackend, 4>::from_data([[[[2.0], [6.0]]]], &d);
        let bias = Tensor::<TtBackend, 4>::from_data([[[[0.25, -0.25]]]], &d);
        let opts = AttentionModuleOptions {
            scale: Some(0.5),
            softcap: Some(1.0),
            is_causal: false,
        };
        let (out, report) = burn_tt::with_report(|| attention(q, k, v, None, Some(bias), opts));
        assert_native_model(&report);
        let flex = attention(
            Tensor::<burn_flex::Flex, 4>::from_data([[[[1.0]]]], &burn_flex::FlexDevice),
            Tensor::<burn_flex::Flex, 4>::from_data([[[[-1.0], [1.0]]]], &burn_flex::FlexDevice),
            Tensor::<burn_flex::Flex, 4>::from_data([[[[2.0], [6.0]]]], &burn_flex::FlexDevice),
            None,
            Some(Tensor::<burn_flex::Flex, 4>::from_data(
                [[[[0.25, -0.25]]]],
                &burn_flex::FlexDevice,
            )),
            AttentionModuleOptions {
                scale: Some(0.5),
                softcap: Some(1.0),
                is_causal: false,
            },
        )
        .into_data()
        .to_vec::<f32>()
        .unwrap()[0] as f64;
        let logit = 0.5f64.tanh() - 0.25;
        let p = logit.exp() / (logit.exp() + (-logit).exp());
        let want = 2.0 * (1.0 - p) + 6.0 * p;
        // tanh's absolute error at |x|<=1 is bounded by TANH_BOUND.
        // Softmax sensitivity is <=1/2; exp/reciprocal contribute the stated
        // relative bounds. TF32 truncates each probability by <2^-10; the
        // positive two-term product sum adds two F32 rounding errors.
        let bound = 6.0
            * (1.0 / 1024.0
                + tt_kernels::sfpu::ops::TANH_BOUND
                + 2.0 * tt_kernels::sfpu::ops::EXP_BOUND
                + 4.0 * f32::EPSILON as f64);
        let got = out.into_data().to_vec::<f32>().unwrap()[0] as f64;
        assert!(
            (got - want).abs() <= bound,
            "{got} vs {want}, bound={bound}"
        );
        // Charge sixteen extra F32 roundings for the Flex tanh/exp/divide
        // and final weighted sum; all intermediates here have magnitude <=6.
        let flex_gamma = 16.0 * 2f64.powi(-24) / (1.0 - 16.0 * 2f64.powi(-24));
        assert!((got - flex).abs() <= bound + 6.0 * flex_gamma);
        assert!(
            (want - 4.0).abs() > bound,
            "ignoring softcap/bias must be detectable"
        );
    });
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_attention_bias_and_resident_training_use_f32_loss() {
    use burn::tensor::{DType, FloatDType};
    type AD = Autodiff<TtBackend>;
    with_device(Config::default(), |d| {
        let q = Tensor::<TtBackend, 4>::zeros([1, 1, 1, 3], &d).cast(FloatDType::BF16);
        let k = Tensor::<TtBackend, 4>::zeros([1, 1, 2, 3], &d).cast(FloatDType::BF16);
        let v = Tensor::<TtBackend, 4>::from_data([[[[2.0], [6.0]]]], &d).cast(FloatDType::BF16);
        let bias = Tensor::<TtBackend, 4>::zeros([1, 1, 1, 2], &d).cast(FloatDType::BF16);
        let before = tensor_traffic();
        let (out, report) =
            burn_tt::with_report(|| attention(q, k, v, None, Some(bias), options(false)));
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        assert!(resident(out.clone()));
        assert_eq!(
            out.into_data()
                .convert_dtype(DType::F32)
                .to_vec::<f32>()
                .unwrap(),
            [4.0]
        );

        let q = Tensor::<AD, 4>::zeros([1, 1, 1, 3], &d)
            .cast(FloatDType::BF16)
            .require_grad();
        let k = Tensor::<AD, 4>::zeros([1, 1, 2, 3], &d)
            .cast(FloatDType::BF16)
            .require_grad();
        let v = Tensor::<AD, 4>::from_data([[[[2.0], [6.0]]]], &d)
            .cast(FloatDType::BF16)
            .require_grad();
        let before = tensor_traffic();
        let (updated, report) = burn_tt::with_report(|| {
            let loss = attention(q, k, v.clone(), None, None, options(false))
                .cast(FloatDType::F32)
                .sum();
            let grads = loss.backward();
            let grad = v.grad(&grads).unwrap();
            v.inner() - grad.mul_scalar(0.5)
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        assert!(resident(updated.clone()));
        assert_eq!(
            updated
                .into_data()
                .convert_dtype(DType::F32)
                .to_vec::<f32>()
                .unwrap(),
            [1.75, 5.75]
        );
    });
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_attention_trace_replays_ragged_heads_and_sequences() {
    use burn::tensor::FloatDType;
    for tiles in [1, 2] {
        with_device(
            Config {
                tiles: Some(burn_tt::TileChoice::Count(tiles)),
                ..Config::default()
            },
            |d| {
                let q = Tensor::<TtBackend, 4>::zeros([1, 2, 3, 5], &d).cast(FloatDType::BF16);
                let k = Tensor::<TtBackend, 4>::zeros([1, 2, 64, 5], &d).cast(FloatDType::BF16);
                let v = Tensor::<TtBackend, 4>::ones([1, 2, 64, 64], &d).mul_scalar(1.0);
                let TensorPrimitive::Float(input) = v.clone().into_primitive() else {
                    unreachable!()
                };
                let before = tensor_traffic();
                let ((trace, first), report) = burn_tt::with_report(|| {
                    burn_tt::Trace::capture(&input, || {
                        let out = attention(
                            q.clone(),
                            k.clone(),
                            v.clone().cast(FloatDType::BF16),
                            None,
                            None,
                            options(false),
                        )
                        .cast(FloatDType::F32);
                        let TensorPrimitive::Float(out) = out.into_primitive() else {
                            unreachable!()
                        };
                        out
                    })
                    .unwrap()
                });
                assert_native_model(&report);
                assert_eq!(tensor_traffic().uploads, before.uploads);
                assert_eq!(first, vec![1.0; 384]);
                for value in [-2.0, 4.0] {
                    assert_eq!(trace.run(vec![value; 8192]).unwrap(), vec![value; 384]);
                }
            },
        );
    }
}

#[test]
fn attention_large_k_permuted_views_and_masked_rows_match_analytic_bound() {
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            let width = 3609;
            let q = Tensor::<TtBackend, 4>::ones([1, 2, width, 3], &d)
                .to_device(&d)
                .swap_dims(2, 3);
            let kv: Vec<f32> = (0..2 * width * 3)
                .map(|i| [0.0, 1.0 / 4096.0, -1.0 / 4096.0][i % 3])
                .collect();
            let k = Tensor::<TtBackend, 4>::from_data(TensorData::new(kv, [1, 2, width, 3]), &d)
                .to_device(&d)
                .swap_dims(2, 3);
            let vv: Vec<f32> = (0..2 * 3 * 35)
                .map(|i| (2 * (i / 35 % 3) + 2 + 4 * (i % 35)) as f32)
                .collect();
            let v = Tensor::<TtBackend, 4>::from_data(TensorData::new(vv, [1, 2, 3, 35]), &d)
                .to_device(&d);
            let mask = Tensor::<TtBackend, 4, Bool>::from_data(
                TensorData::new((0..9).map(|i| i / 3 == 1).collect::<Vec<_>>(), [1, 1, 3, 3]),
                &d,
            )
            .to_device(&d);
            let before = tensor_traffic();
            let (out, report) =
                burn_tt::with_report(|| attention(q, k, v, Some(mask), None, options(true)));
            assert_native_model(&report);
            assert!(resident(out.clone()));
            assert_eq!(tensor_traffic().downloads, before.downloads);
            let logits = [0.0, (width as f64) / 4096.0, -(width as f64) / 4096.0];
            let z: f64 = logits.iter().map(|x| x.exp()).sum();
            let p = logits.map(|x| x.exp() / z);
            let got = out.into_data().to_vec::<f32>().unwrap();
            for h in 0..2 {
                for row in 0..3 {
                    for c in 0..35 {
                        let expected = match row {
                            0 => (2 + 4 * c) as f64,
                            1 => 0.0,
                            _ => (0..3).map(|r| p[r] * (2 * r + 2 + 4 * c) as f64).sum(),
                        };
                        // Unit/dyadic QK products and every K continuation sum are exact
                        // in F32. Softmax exp/reciprocal and TF32 probability conversion
                        // contribute relative error; charge twelve final phase additions.
                        let maximum = (6 + 4 * c) as f64;
                        let bound = maximum
                            * (2.0 * tt_kernels::sfpu::ops::EXP_BOUND
                                + 1.0 / 1024.0
                                + 16.0 * f32::EPSILON as f64);
                        assert!(
                            (got[(h * 3 + row) * 35 + c] as f64 - expected).abs() <= bound,
                            "head {h},row {row},column {c}: {} vs {expected},bound {bound}",
                            got[(h * 3 + row) * 35 + c]
                        );
                        if row == 1 {
                            assert_eq!(got[(h * 3 + row) * 35 + c], 0.0);
                        }
                    }
                }
            }
        },
    );
}
