//! Independent scalar convolution oracles and resident analytic gradients.
use burn::backend::Autodiff;
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{module, DType, Tensor, TensorData, TensorPrimitive};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn resident<const D: usize>(x: &Tensor<TtBackend, D>) {
    let TensorPrimitive::Float(x) = x.clone().into_primitive() else {
        unreachable!()
    };
    assert!(x.computed_on_device());
}

fn convolution_case(xs: [usize; 4], ws: [usize; 4], options: ConvOptions<2>) {
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            type AD = Autodiff<TtBackend>;
            let xv: Vec<_> = (0..xs.iter().product())
                .map(|i| (i % 5) as f32 - 2.0)
                .collect();
            let wv: Vec<_> = (0..ws.iter().product())
                .map(|i| (i % 3) as f32 - 1.0)
                .collect();
            let bv: Vec<_> = (0..ws[0]).map(|i| (i as f32) - 2.0).collect();
            let x = Tensor::<AD, 4>::from_data(TensorData::new(xv.clone(), xs), &d).require_grad();
            let w = Tensor::<AD, 4>::from_data(TensorData::new(wv.clone(), ws), &d).require_grad();
            let b =
                Tensor::<AD, 1>::from_data(TensorData::new(bv.clone(), [ws[0]]), &d).require_grad();
            let before = tensor_traffic();
            let ((out, gx, gw, gb), report) = burn_tt::with_report(|| {
                let y = module::conv2d(x.clone(), w.clone(), Some(b.clone()), options.clone());
                let grads = y.clone().sum().backward();
                let result = (
                    y.inner(),
                    x.grad(&grads).unwrap(),
                    w.grad(&grads).unwrap(),
                    b.grad(&grads).unwrap(),
                );
                resident(&result.0);
                resident(&result.1);
                resident(&result.2);
                resident(&result.3);
                assert_eq!(tensor_traffic().downloads, before.downloads);
                result
            });
            assert_native_model(&report);
            let ys = out.dims();
            let mut want = vec![0.0; ys.iter().product()];
            let mut dx = vec![0.0; xv.len()];
            let mut dw = vec![0.0; wv.len()];
            let mut db = vec![0.0; bv.len()];
            let ocg = ws[0] / options.groups;
            for n in 0..xs[0] {
                for oc in 0..ws[0] {
                    for r in 0..ys[2] {
                        for c in 0..ys[3] {
                            let y = ((n * ws[0] + oc) * ys[2] + r) * ys[3] + c;
                            want[y] = bv[oc];
                            db[oc] += 1.0;
                            for ic in 0..ws[1] {
                                for kr in 0..ws[2] {
                                    for kc in 0..ws[3] {
                                        let ir = (r * options.stride[0] + kr * options.dilation[0])
                                            as isize
                                            - options.padding[0] as isize;
                                        let jc = (c * options.stride[1] + kc * options.dilation[1])
                                            as isize
                                            - options.padding[1] as isize;
                                        if ir < 0
                                            || ir >= xs[2] as isize
                                            || jc < 0
                                            || jc >= xs[3] as isize
                                        {
                                            continue;
                                        }
                                        let xi = ((n * xs[1] + oc / ocg * ws[1] + ic) * xs[2]
                                            + ir as usize)
                                            * xs[3]
                                            + jc as usize;
                                        let wi = ((oc * ws[1] + ic) * ws[2] + kr) * ws[3] + kc;
                                        want[y] += xv[xi] * wv[wi];
                                        dx[xi] += wv[wi];
                                        dw[wi] += xv[xi];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Small integers make every product, partial and sum exactly representable.
            assert!(want.iter().any(|&v| v != 0.0), "cleared output cannot pass");
            assert_eq!(out.into_data().to_vec::<f32>().unwrap(), want);
            assert_eq!(gx.into_data().to_vec::<f32>().unwrap(), dx);
            assert_eq!(gw.into_data().to_vec::<f32>().unwrap(), dw);
            assert_eq!(gb.into_data().to_vec::<f32>().unwrap(), db);
        },
    );
}

#[test]
fn grouped_dilated_ragged_convolution_and_gradients() {
    convolution_case(
        [1, 4, 5, 7],
        [6, 2, 2, 3],
        ConvOptions::new([2, 2], [1, 2], [2, 1], 2),
    );
}

#[test]
fn depthwise_batches_cross_patch_boundaries() {
    convolution_case(
        [2, 3, 7, 9],
        [3, 1, 3, 2],
        ConvOptions::new([1, 1], [1, 1], [1, 1], 3),
    );
}

#[test]
fn convolution_large_ragged_k_and_all_gradients() {
    // K=3609 spans 113 tiles: two operands exceed the bounded staging arena,
    // forcing matmul continuation; the final K tile is ragged.
    let plan = tt_ttsim::outside_fork(|| {
        tt_kernels::matmul::plan_in(
            [1, 3609, 2],
            burn_tt::SrcRoute::Tf32FromFp32,
            burn_tt::Fidelity::HiFi4,
            tt_kernels::matmul::Staging::Slots,
        )
        .unwrap()
    });
    assert!(plan.tiles[1] < 3609usize.div_ceil(32));
    convolution_case(
        [1, 401, 3, 3],
        [2, 401, 3, 3],
        ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
    );
}

fn tiny(dtype: DType) {
    with_device(Config::default(), |d| {
        type AD = Autodiff<TtBackend>;
        let x = Tensor::<AD, 4>::from_data(
            TensorData::new((1..10).map(|i| i as f32).collect::<Vec<_>>(), [1, 1, 3, 3]),
            (&d, dtype),
        )
        .require_grad();
        let w =
            Tensor::<AD, 4>::from_data(TensorData::new(vec![1.0f32; 4], [1, 1, 2, 2]), (&d, dtype))
                .require_grad();
        let b = Tensor::<AD, 1>::from_data([0.0], (&d, dtype)).require_grad();
        let ((y, gx, gw, gb), report) = burn_tt::with_report(|| {
            let y = module::conv2d(
                x.clone(),
                w.clone(),
                Some(b.clone()),
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            );
            let grads = y.clone().sum().backward();
            (
                y.inner(),
                x.grad(&grads).unwrap(),
                w.grad(&grads).unwrap(),
                b.grad(&grads).unwrap(),
            )
        });
        assert_native_model(&report);
        resident(&y);
        resident(&gx);
        resident(&gw);
        resident(&gb);
        let values = |data: TensorData| data.convert::<f32>().to_vec::<f32>().unwrap();
        assert_eq!(values(y.into_data()), vec![12.0, 16.0, 24.0, 28.0]);
        assert_eq!(
            values(gx.into_data()),
            vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0]
        );
        assert_eq!(values(gw.into_data()), vec![12.0, 16.0, 24.0, 28.0]);
        assert_eq!(values(gb.into_data()), vec![4.0]);
    });
}

#[test]
fn tiny_analytic_convolution_gradients() {
    tiny(DType::F32);
}

#[test]
#[cfg(feature = "silicon")]
fn bf16_convolution_has_f32_accumulation_and_native_gradients() {
    tiny(DType::BF16);
}

#[test]
fn transposed_convolution_output_padding_and_gradients() {
    with_device(Config::default(), |d| {
        type AD = Autodiff<TtBackend>;
        let x = Tensor::<AD, 4>::from_data([[[[1.0, 2.0], [3.0, 4.0]]]], &d).require_grad();
        let w = Tensor::<AD, 4>::from_data([[[[1.0, 2.0], [3.0, 4.0]]]], &d).require_grad();
        let b = Tensor::<AD, 1>::from_data([1.0], &d).require_grad();
        let ((out, gx, gw, gb), report) = burn_tt::with_report(|| {
            let y = module::conv_transpose2d(
                x.clone(),
                w.clone(),
                Some(b.clone()),
                ConvTransposeOptions::new([2, 2], [1, 1], [1, 1], [1, 1], 1),
            );
            let grads = y.clone().sum().backward();
            (
                y.inner(),
                x.grad(&grads).unwrap(),
                w.grad(&grads).unwrap(),
                b.grad(&grads).unwrap(),
            )
        });
        assert_native_model(&report);
        assert_eq!(out.dims(), [1, 1, 3, 3]);
        assert_eq!(
            out.into_data().to_vec::<f32>().unwrap(),
            vec![5.0, 7.0, 9.0, 7.0, 5.0, 9.0, 13.0, 13.0, 17.0]
        );
        assert_eq!(
            gx.into_data().to_vec::<f32>().unwrap(),
            vec![4.0, 7.0, 6.0, 10.0]
        );
        assert_eq!(
            gw.into_data().to_vec::<f32>().unwrap(),
            vec![4.0, 7.0, 6.0, 10.0]
        );
        assert_eq!(gb.into_data().to_vec::<f32>().unwrap(), vec![9.0]);
    });
}

#[test]
fn conv1d_and_unfold_defaults_compose_native_primitives() {
    with_device(Config::default(), |d| {
        type AD = Autodiff<TtBackend>;
        let x = Tensor::<AD, 3>::from_data([[[1.0, 2.0, 3.0, 4.0]]], &d).require_grad();
        let w = Tensor::<AD, 3>::from_data([[[1.0, 1.0]]], &d).require_grad();
        let ((y, gx, gw), report) = burn_tt::with_report(|| {
            let y = module::conv1d(
                x.clone(),
                w.clone(),
                None,
                ConvOptions::new([1], [0], [1], 1),
            );
            let grads = y.clone().sum().backward();
            (y.inner(), x.grad(&grads).unwrap(), w.grad(&grads).unwrap())
        });
        assert_native_model(&report);
        assert_eq!(y.into_data().to_vec::<f32>().unwrap(), vec![3.0, 5.0, 7.0]);
        assert_eq!(
            gx.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 2.0, 1.0]
        );
        assert_eq!(gw.into_data().to_vec::<f32>().unwrap(), vec![6.0, 9.0]);
        let x =
            Tensor::<AD, 4>::from_data([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]], &d).require_grad();
        let ((y, gx), report) = burn_tt::with_report(|| {
            let y = module::unfold4d(
                x.clone(),
                [2, 2],
                burn::tensor::ops::UnfoldOptions::new([1, 1], [0, 0], [1, 1]),
            );
            let grads = y.clone().sum().backward();
            (y.inner(), x.grad(&grads).unwrap())
        });
        assert_native_model(&report);
        assert_eq!(y.dims(), [1, 4, 2]);
        assert_eq!(
            y.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0]
        );
        assert_eq!(
            gx.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 1.0, 1.0, 2.0, 1.0]
        );
    });
}

fn mesh_convolution_gradients(dtype: burn::tensor::FloatDType) {
    tt_tests::burn_device::with_mesh_device(Config::default(), 2, |d| {
        type AD = Autodiff<TtBackend>;
        // All three products have 64 output columns: forward Cout, dW Cout,
        // and dX patch K. Each therefore needs both card partitions.
        let x = Tensor::<AD, 4>::ones([1, 64, 2, 3], &d)
            .cast(dtype)
            .require_grad();
        let w = Tensor::<AD, 4>::ones([64, 64, 1, 1], &d)
            .cast(dtype)
            .require_grad();
        let b = Tensor::<AD, 1>::zeros([64], &d).cast(dtype).require_grad();
        let before = tensor_traffic();
        let execution = burn_tt::mesh_execution(d).unwrap();
        let ((y, gx, gw, gb), report) = burn_tt::with_report(|| {
            let y = module::conv2d(
                x.clone(),
                w.clone(),
                Some(b.clone()),
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            );
            let grads = y.clone().sum().backward();
            (
                y.inner(),
                x.grad(&grads).unwrap(),
                w.grad(&grads).unwrap(),
                b.grad(&grads).unwrap(),
            )
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let after = burn_tt::mesh_execution(d).unwrap();
        for card in 0..2 {
            assert!(
                after.completed_matmuls[card] >= execution.completed_matmuls[card] + 3,
                "card {card} must complete forward, dX and dW"
            );
        }
        assert!(after.acknowledged_ethernet_packets > execution.acknowledged_ethernet_packets);
        assert!(after.acknowledged_ethernet_bytes > execution.acknowledged_ethernet_bytes);
        assert_eq!(
            y.cast(burn::tensor::FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            vec![64.0; 384]
        );
        assert_eq!(
            gx.cast(burn::tensor::FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            vec![64.0; 384]
        );
        assert_eq!(
            gw.cast(burn::tensor::FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            vec![6.0; 4096]
        );
        assert_eq!(
            gb.cast(burn::tensor::FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            vec![6.0; 64]
        );
    });
}

#[test]
fn mesh_convolution_forward_and_all_gradients_compute_on_both_cards() {
    mesh_convolution_gradients(burn::tensor::FloatDType::F32);
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_mesh_convolution_forward_and_all_gradients_compute_on_both_cards() {
    mesh_convolution_gradients(burn::tensor::FloatDType::BF16);
}

fn convolution_trace(bf16: bool) {
    use burn::tensor::FloatDType;
    for tiles in [1, 2] {
        with_device(
            Config {
                tiles: Some(burn_tt::TileChoice::Count(tiles)),
                ..Config::default()
            },
            |d| {
                let x = Tensor::<TtBackend, 4>::ones([1, 1, 9, 10], &d).mul_scalar(1.0);
                let TensorPrimitive::Float(input) = x.clone().into_primitive() else {
                    unreachable!()
                };
                let w = Tensor::<TtBackend, 4>::ones([64, 1, 2, 2], &d).to_device(&d);
                let before = tensor_traffic();
                let ((trace, first), report) = burn_tt::with_report(|| {
                    burn_tt::Trace::capture(&input, || {
                        let (x, w) = if bf16 {
                            (
                                x.clone().cast(FloatDType::BF16),
                                w.clone().cast(FloatDType::BF16),
                            )
                        } else {
                            (x.clone(), w.clone())
                        };
                        let out =
                            module::conv2d(x, w, None, ConvOptions::new([1, 1], [0, 0], [1, 1], 1))
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
                assert_eq!(first, vec![4.0; 4608]);
                for value in [-2.0, 3.0] {
                    assert_eq!(trace.run(vec![value; 90]).unwrap(), vec![4.0 * value; 4608]);
                }
                drop(trace);
                assert_eq!(x.sum().into_data().to_vec::<f32>().unwrap(), vec![270.0]);
            },
        );
    }
}

#[test]
fn convolution_trace_replays_bounded_patch_batches() {
    convolution_trace(false);
}

#[test]
#[cfg(feature = "silicon")]
fn bf16_convolution_trace_replays_with_one_output_narrowing() {
    convolution_trace(true);
}

#[test]
#[cfg(feature = "silicon")]
fn bf16_convolution_resident_training_uses_f32_loss() {
    use burn::tensor::FloatDType;
    with_device(Config::default(), |d| {
        type AD = Autodiff<TtBackend>;
        let x = Tensor::<AD, 4>::ones([1, 1, 3, 3], &d).cast(FloatDType::BF16);
        let w = Tensor::<AD, 4>::ones([1, 1, 2, 2], &d)
            .cast(FloatDType::BF16)
            .require_grad();
        let before = tensor_traffic();
        let (updated, report) = burn_tt::with_report(|| {
            let loss = module::conv2d(
                x,
                w.clone(),
                None,
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            )
            .cast(FloatDType::F32)
            .sum();
            let grads = loss.backward();
            w.clone().inner() - w.grad(&grads).unwrap().mul_scalar(0.125)
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        resident(&updated);
        assert_eq!(
            updated
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap(),
            vec![0.5; 4]
        );
    });
}

#[test]
fn permuted_fractional_convolution_matches_flex_with_derived_bound() {
    use burn_flex::{Flex, FlexDevice};
    with_device(Config::default(), |d| {
        let base: Vec<f32> = (0..70)
            .map(|i| ((i * 17 % 29) as f32 - 14.0) / 19.0)
            .collect();
        let weights: Vec<f32> = (0..24)
            .map(|i| ((i * 11 % 17) as f32 - 8.0) / 13.0)
            .collect();
        let x = Tensor::<TtBackend, 4>::from_data(TensorData::new(base.clone(), [1, 2, 7, 5]), &d)
            .to_device(&d)
            .swap_dims(2, 3);
        let w =
            Tensor::<TtBackend, 4>::from_data(TensorData::new(weights.clone(), [2, 2, 2, 3]), &d);
        let options = ConvOptions::new([2, 1], [1, 1], [1, 2], 1);
        let before = tensor_traffic();
        let (out, report) = burn_tt::with_report(|| module::conv2d(x, w, None, options.clone()));
        assert_native_model(&report);
        resident(&out);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let ys = out.dims();
        let got = out.into_data().to_vec::<f32>().unwrap();
        let fx =
            Tensor::<Flex, 4>::from_data(TensorData::new(base.clone(), [1, 2, 7, 5]), &FlexDevice)
                .swap_dims(2, 3);
        let fw = Tensor::<Flex, 4>::from_data(
            TensorData::new(weights.clone(), [2, 2, 2, 3]),
            &FlexDevice,
        );
        let expected = module::conv2d(fx, fw, None, options)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        // Same operand conversion/phase bound as step68: TF32 truncation
        // contributes <2^-9 sum|xw|; charge 4K device additions at 2^-23
        // and K Flex additions at 2^-24. Copies and zero padding are exact.
        let k = 12.0;
        let dg = 4.0 * k * 2f64.powi(-23) / (1.0 - 4.0 * k * 2f64.powi(-23));
        let fg = k * 2f64.powi(-24) / (1.0 - k * 2f64.powi(-24));
        for oc in 0..2 {
            for r in 0..ys[2] {
                for c in 0..ys[3] {
                    let mut magnitude = 0.0;
                    for ic in 0..2 {
                        for kr in 0..2 {
                            for kc in 0..3 {
                                let ir = (r * 2 + kr) as isize - 1;
                                let jc = (c + kc * 2) as isize - 1;
                                if (0..5).contains(&ir) && (0..7).contains(&jc) {
                                    let a = base[(ic * 7 + jc as usize) * 5 + ir as usize] as f64;
                                    let b = weights[((oc * 2 + ic) * 2 + kr) * 3 + kc] as f64;
                                    magnitude += (a * b).abs();
                                }
                            }
                        }
                    }
                    let at = (oc * ys[2] + r) * ys[3] + c;
                    let bound = (2f64.powi(-9) + (1.0 + 2f64.powi(-9)) * dg + fg) * magnitude;
                    assert!(
                        (got[at] as f64 - expected[at] as f64).abs() <= bound,
                        "output {at}: {} vs {}, bound {bound}",
                        got[at],
                        expected[at]
                    );
                }
            }
        }
    });
}

#[test]
fn transposed_convolution_output_padding_uses_dilation_limit() {
    with_device(Config::default(), |d| {
        type AD = Autodiff<TtBackend>;
        let xv: Vec<f32> = (1..=8).map(|n| n as f32).collect();
        let wv = vec![1.0, 2.0, 3.0, 4.0, -1.0, 2.0, -3.0, 4.0];
        let x = Tensor::<AD, 4>::from_data(TensorData::new(xv.clone(), [1, 2, 2, 2]), &d)
            .require_grad();
        let w = Tensor::<AD, 4>::from_data(TensorData::new(wv.clone(), [2, 1, 2, 2]), &d)
            .require_grad();
        let before = tensor_traffic();
        let ((out, gx, gw), report) = burn_tt::with_report(|| {
            let y = module::conv_transpose2d(
                x.clone(),
                w.clone(),
                None,
                ConvTransposeOptions::new([1, 2], [1, 1], [2, 2], [3, 3], 2),
            );
            let grads = y.clone().sum().backward();
            (y.inner(), x.grad(&grads).unwrap(), w.grad(&grads).unwrap())
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        resident(&out);
        resident(&gx);
        resident(&gw);
        assert_eq!(out.dims(), [1, 2, 5, 6]);
        let mut expected = vec![0.0; 60];
        let mut dx = vec![0.0; 8];
        let mut dw = vec![0.0; 8];
        for ch in 0..2 {
            for r in 0..2 {
                for c in 0..2 {
                    for kr in 0..2 {
                        for kc in 0..2 {
                            let yr = (r + kr * 3) as isize - 1;
                            let yc = (c * 2 + kc * 3) as isize - 1;
                            if (0..5).contains(&yr) && (0..6).contains(&yc) {
                                let xi = (ch * 2 + r) * 2 + c;
                                let wi = (ch * 2 + kr) * 2 + kc;
                                expected[(ch * 5 + yr as usize) * 6 + yc as usize] +=
                                    xv[xi] * wv[wi];
                                dx[xi] += wv[wi];
                                dw[wi] += xv[xi];
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(out.into_data().to_vec::<f32>().unwrap(), expected);
        assert_eq!(gx.into_data().to_vec::<f32>().unwrap(), dx);
        assert_eq!(gw.into_data().to_vec::<f32>().unwrap(), dw);
    });
}

#[test]
fn transposed_conv1d_and_unfold_gradients_compose_for_storage_formats() {
    use burn::tensor::FloatDType;
    with_device(Config::default(), |d| {
        type AD = Autodiff<TtBackend>;
        let types = if cfg!(feature = "silicon") {
            vec![FloatDType::F32, FloatDType::BF16]
        } else {
            vec![FloatDType::F32]
        };
        for dtype in types {
            let x = Tensor::<AD, 3>::from_data([[[1.0, 2.0]]], &d)
                .cast(dtype)
                .require_grad();
            let w = Tensor::<AD, 3>::from_data([[[1.0, 3.0]]], &d)
                .cast(dtype)
                .require_grad();
            let b = Tensor::<AD, 1>::ones([1], &d).cast(dtype).require_grad();
            let image = Tensor::<AD, 4>::ones([1, 1, 3, 3], &d)
                .cast(dtype)
                .require_grad();
            let before = tensor_traffic();
            let ((y, dx, dw, db, patches, di), report) = burn_tt::with_report(|| {
                let y = module::conv_transpose1d(
                    x.clone(),
                    w.clone(),
                    Some(b.clone()),
                    ConvTransposeOptions::new([2], [1], [1], [1], 1),
                );
                let grad = y.clone().cast(FloatDType::F32).sum().backward();
                let patches = module::unfold4d(
                    image.clone(),
                    [2, 2],
                    burn::tensor::ops::UnfoldOptions::new([1, 1], [0, 0], [1, 1]),
                );
                let ig = patches.clone().cast(FloatDType::F32).sum().backward();
                (
                    y.inner().cast(FloatDType::F32),
                    x.grad(&grad).unwrap().cast(FloatDType::F32),
                    w.grad(&grad).unwrap().cast(FloatDType::F32),
                    b.grad(&grad).unwrap().cast(FloatDType::F32),
                    patches.inner().cast(FloatDType::F32),
                    image.grad(&ig).unwrap().cast(FloatDType::F32),
                )
            });
            assert_native_model(&report);
            assert_eq!(tensor_traffic().downloads, before.downloads);
            for t in [&y, &dx, &dw] {
                resident(t);
            }
            resident(&db);
            resident(&patches);
            resident(&di);
            assert_eq!(y.into_data().to_vec::<f32>().unwrap(), vec![4.0, 3.0, 7.0]);
            assert_eq!(dx.into_data().to_vec::<f32>().unwrap(), vec![3.0, 4.0]);
            assert_eq!(dw.into_data().to_vec::<f32>().unwrap(), vec![2.0, 3.0]);
            assert_eq!(db.into_data().to_vec::<f32>().unwrap(), vec![3.0]);
            assert_eq!(patches.into_data().to_vec::<f32>().unwrap(), vec![1.0; 16]);
            assert_eq!(
                di.into_data().to_vec::<f32>().unwrap(),
                vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0]
            );
        }
    });
}
