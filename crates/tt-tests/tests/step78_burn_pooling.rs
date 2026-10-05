//! Resident NCHW pooling and overlapping gradients, checked independently.
use burn::backend::Autodiff;
use burn::tensor::{module, DType, Tensor, TensorData, TensorPrimitive};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn resident<const D: usize>(x: &Tensor<TtBackend, D>) {
    match x.clone().into_primitive() {
        TensorPrimitive::Float(t) => assert!(t.computed_on_device()),
        _ => panic!("unexpected quantized output"),
    }
}

fn check(dtype: DType) {
    with_device(Config::default(), |d| {
        let x = Tensor::<TtBackend, 4>::from_data(
            TensorData::new((0..20).map(|i| i as f32).collect::<Vec<_>>(), [1, 1, 4, 5]),
            (&d, dtype),
        );
        let before = tensor_traffic();
        let ((avg, max, indices), report) = burn_tt::with_report(|| {
            let avg = module::avg_pool2d(x.clone(), [2, 2], [1, 1], [0, 0], false, false);
            let (max, indices) =
                module::max_pool2d_with_indices(x, [2, 2], [1, 1], [0, 0], [1, 1], false);
            resident(&avg);
            resident(&max);
            assert_eq!(tensor_traffic().downloads, before.downloads);
            (avg, max, indices)
        });
        assert_native_model(&report);
        assert_eq!(avg.dims(), [1, 1, 3, 4]);
        let starts: Vec<_> = (0..3)
            .flat_map(|r| (0..4).map(move |c| r * 5 + c))
            .collect();
        let f32_values = |data: TensorData| data.convert::<f32>().to_vec::<f32>().unwrap();
        assert_eq!(
            f32_values(avg.into_data()),
            starts.iter().map(|i| *i as f32 + 3.0).collect::<Vec<_>>()
        );
        assert_eq!(
            f32_values(max.into_data()),
            starts.iter().map(|i| *i as f32 + 6.0).collect::<Vec<_>>()
        );
        assert_eq!(
            indices.into_data().to_vec::<i32>().unwrap(),
            starts.iter().map(|i| *i + 6).collect::<Vec<_>>()
        );

        type AD = Autodiff<TtBackend>;
        let x = Tensor::<AD, 4>::from_data(
            TensorData::new(vec![1.0f32; 20], [1, 1, 4, 5]),
            (&d, dtype),
        )
        .require_grad();
        let (grad, report) = burn_tt::with_report(|| {
            let loss = module::avg_pool2d(x.clone(), [2, 2], [1, 1], [0, 0], false, false).sum();
            let grads = loss.backward();
            x.grad(&grads).unwrap()
        });
        resident(&grad);
        assert_native_model(&report);
        let expected: Vec<_> = [1, 2, 2, 1]
            .into_iter()
            .flat_map(|r| [1, 2, 2, 2, 1].map(|c| (r * c) as f32 / 4.0))
            .collect();
        assert_eq!(f32_values(grad.into_data()), expected);

        // A 35-element window exercises more than one 16-lane GAPOOL chunk.
        let x = Tensor::<TtBackend, 4>::from_data(
            TensorData::new(vec![2.0f32; 35], [1, 1, 5, 7]),
            (&d, dtype),
        );
        let (out, report) = burn_tt::with_report(|| module::adaptive_avg_pool2d(x, [1, 1]));
        resident(&out);
        assert_native_model(&report);
        assert_eq!(f32_values(out.into_data()), vec![2.0]);
    });
}

#[test]
fn f32_pooling_and_overlap_gradients_are_native() {
    check(DType::F32);
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_fpu_pooling_and_overlap_gradients_are_native() {
    check(DType::BF16);
}

#[test]
fn pooling_padding_dilation_ceil_adaptive_and_max_gradients() {
    with_device(Config::default(), |d| {
        // Ceil mode permits one partial window when the kernel exceeds the
        // unpadded extent by less than a stride.
        let one = Tensor::<TtBackend, 4>::from_data([[[[4.0]]]], &d);
        assert_eq!(
            module::avg_pool2d(one, [2, 2], [2, 2], [0, 0], true, true)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            vec![4.0]
        );
        let x =
            Tensor::<TtBackend, 4>::from_data(TensorData::new(vec![4.0f32; 15], [1, 1, 3, 5]), &d);
        let ((pad, exclude, adaptive, max), report) = burn_tt::with_report(|| {
            let pad = module::avg_pool2d(x.clone(), [2, 2], [2, 2], [1, 1], true, true);
            let exclude = module::avg_pool2d(x.clone(), [2, 2], [2, 2], [1, 1], false, true);
            let adaptive = module::adaptive_avg_pool2d(x.clone(), [5, 7]);
            let max = module::max_pool2d_with_indices(x, [2, 2], [2, 2], [1, 1], [2, 2], true);
            (pad, exclude, adaptive, max)
        });
        assert_native_model(&report);
        assert_eq!(pad.dims(), [1, 1, 2, 3]);
        assert_eq!(
            pad.into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 2.0, 2.0, 4.0, 4.0]
        );
        assert_eq!(exclude.into_data().to_vec::<f32>().unwrap(), vec![4.0; 6]);
        assert_eq!(adaptive.into_data().to_vec::<f32>().unwrap(), vec![4.0; 35]);
        assert_eq!(max.0.into_data().to_vec::<f32>().unwrap(), vec![4.0; 6]);
        // Equal maxima select the first valid spatial index in scan order.
        assert_eq!(
            max.1.into_data().to_vec::<i32>().unwrap(),
            vec![6, 6, 8, 6, 6, 8]
        );
        type AD = Autodiff<TtBackend>;
        let input = Tensor::<AD, 4>::from_data(
            TensorData::new((0..20).map(|i| i as f32).collect::<Vec<_>>(), [1, 1, 4, 5]),
            &d,
        )
        .require_grad();
        let (grad, report) = burn_tt::with_report(|| {
            let (out, _) = module::max_pool2d_with_indices(
                input.clone(),
                [2, 2],
                [1, 1],
                [0, 0],
                [1, 1],
                false,
            );
            input.grad(&out.sum().backward()).unwrap()
        });
        assert_native_model(&report);
        let want: Vec<_> = (0..20)
            .map(|i| if i / 5 >= 1 && i % 5 >= 1 { 1.0 } else { 0.0 })
            .collect();
        assert_eq!(grad.into_data().to_vec::<f32>().unwrap(), want);
    });
}

#[test]
fn max_pool_values_and_indices_agree_for_zeros_nans_and_extremes() {
    with_device(Config::default(), |d| {
        let patterns = [
            [0x80000000, 0, 0x80000000, 0],
            [0xffc12345, 0x3f800000, 0x7fc23456, 0x7f800000],
            [0xff800000, 0x80000001, 1, 0x7f7fffff],
            [0x3f800000; 4],
        ];
        let input = Tensor::<TtBackend, 4>::from_data(
            TensorData::new(
                patterns
                    .into_iter()
                    .flatten()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>(),
                [4, 1, 2, 2],
            ),
            &d,
        );
        let ((values, indices), report) = burn_tt::with_report(|| {
            module::max_pool2d_with_indices(input, [2, 2], [1, 1], [0, 0], [1, 1], false)
        });
        assert_native_model(&report);
        let got: Vec<_> = values
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(got, vec![0x80000000, 0xffc12345, 0x7f7fffff, 0x3f800000]);
        assert_eq!(
            indices.into_data().to_vec::<i32>().unwrap(),
            vec![0, 0, 3, 0]
        );
    });
}

fn trace_pooling(bf16: bool) {
    use burn::tensor::FloatDType;
    use burn_tt::Trace;
    with_device(Config::default(), |d| {
        let input = Tensor::<TtBackend, 4>::from_data(
            TensorData::new((0..35).map(|i| i as f32).collect::<Vec<_>>(), [1, 1, 5, 7]),
            &d,
        )
        .to_device(&d)
        .mul_scalar(1.0);
        let TensorPrimitive::Float(primitive) = input.clone().into_primitive() else {
            unreachable!()
        };
        let traffic = tensor_traffic();
        let ((trace, first), report) = burn_tt::with_report(|| {
            Trace::capture(&primitive, || {
                let x = if bf16 {
                    input.clone().cast(FloatDType::BF16)
                } else {
                    input.clone()
                };
                // A long adaptive window, overlapping average windows and max
                // indices all create constants during capture. Replays must also
                // hold intermediate metadata after its temporary handle is freed.
                let avg = module::avg_pool2d(x.clone(), [2, 2], [1, 1], [0, 0], false, false);
                let max = module::max_pool2d_with_indices(
                    x.clone(),
                    [2, 2],
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    false,
                )
                .0;
                let out = module::adaptive_avg_pool2d(avg + max, [1, 1])
                    + module::adaptive_avg_pool2d(x, [1, 1]);
                let TensorPrimitive::Float(out) = out.cast(FloatDType::F32).into_primitive() else {
                    unreachable!()
                };
                out
            })
            .unwrap()
        });
        assert_native_model(&report);
        assert_eq!(
            tensor_traffic().uploads,
            traffic.uploads,
            "capture uploaded pool metadata"
        );
        assert_eq!(first, vec![55.0]); // mean(x)=17, mean(avg)=17, mean(max)=21
        for value in [-2.0, 0.0, 4.0] {
            assert_eq!(trace.run(vec![value; 35]).unwrap(), vec![3.0 * value]);
        }
        drop(trace);
        assert_eq!(
            input.mul_scalar(2.0).into_data().to_vec::<f32>().unwrap(),
            vec![8.0; 35]
        );
    });
}

#[test]
fn f32_pool_trace_replays_changed_inputs_and_retains_geometry() {
    trace_pooling(false);
}

#[cfg(feature = "silicon")]
#[test]
fn bf16_pool_trace_replays_changed_inputs_and_retains_geometry() {
    trace_pooling(true);
}
