//! Native storage and failure behavior need no hardware or reference backend.
use burn_tensor::{
    backend::Backend, Bool, DType, Distribution, Int, Tensor, TensorData, Transaction,
};
use burn_tt::{TtBackend, TtDevice};

#[test]
fn dtype_capabilities_do_not_advertise_missing_integer_or_reduced_float_compute() {
    use burn_backend::DTypeUsage;
    let d = TtDevice::new(304);
    let float = TtBackend::dtype_usage(&d, DType::F32);
    assert!(float.contains(DTypeUsage::Arithmetic) && float.contains(DTypeUsage::Accelerated));
    let int = TtBackend::dtype_usage(&d, DType::I32);
    assert!(int.contains(DTypeUsage::Storage) && !int.contains(DTypeUsage::Arithmetic));
    // F16 is raw storage with exact device casts, not compute.
    let half = TtBackend::dtype_usage(&d, DType::F16);
    assert!(
        half.contains(DTypeUsage::Storage)
            && !half.contains(DTypeUsage::Arithmetic)
            && !half.contains(DTypeUsage::Accelerated)
    );
    assert!(TtBackend::dtype_usage(&d, DType::BF16).contains(DTypeUsage::Accelerated));
    assert!(TtBackend::dtype_usage(&d, DType::F64).is_empty());
}

#[test]
fn owned_buffers_and_transactions_preserve_types_shapes_and_order() {
    let d = TtDevice::new(300);
    let f = Tensor::<TtBackend, 2>::from_data([[-0.0, 2.5], [3.0, 4.0]], &d);
    let i = Tensor::<TtBackend, 1, Int>::from_data([-7, 0, i32::MAX], &d);
    let b = Tensor::<TtBackend, 1, Bool>::from_data([true, false], &d);
    let copies = f.clone().transpose().reshape([4, 1]).into_data();
    assert_eq!(copies.to_vec::<f32>().unwrap(), [-0.0, 3.0, 2.5, 4.0]);
    let data = Transaction::default()
        .register(i)
        .register(f.clone())
        .register(b)
        .register(f)
        .execute();
    assert_eq!(data[0].dtype, DType::I32);
    assert_eq!(data[0].to_vec::<i32>().unwrap(), [-7, 0, i32::MAX]);
    assert_eq!(data[1].shape.to_vec(), [2, 2]);
    assert_eq!(
        data[1].to_vec::<f32>().unwrap()[0].to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(data[2].to_vec::<bool>().unwrap(), [true, false]);
    assert_eq!(data[1], data[3]);
}

#[test]
fn seeded_streams_repeat_and_are_independent_between_devices() {
    let (a, b) = (TtDevice::new(301), TtDevice::new(302));
    let draw = |device: &TtDevice| {
        Tensor::<TtBackend, 1>::random([64], Distribution::Normal(0.0, 1.0), device).into_data()
    };
    TtBackend::seed(&a, 19);
    let first = draw(&a);
    let second = draw(&a);
    TtBackend::seed(&b, 19);
    assert_eq!(first, draw(&b));
    TtBackend::seed(&b, 77);
    assert_ne!(first, second);
    TtBackend::seed(&a, 19);
    assert_eq!(first, draw(&a));
    assert_eq!(second, draw(&a));
}

#[test]
fn unsupported_methods_name_operation_shape_and_dtype() {
    let d = TtDevice::new(303);
    // `rfft` is out of scope by design (`out_of_scope.rs`): a stable example.
    let signal = match Tensor::<TtBackend, 2>::ones([2, 3], &d).into_primitive() {
        burn_backend::TensorPrimitive::Float(t) => t,
        burn_backend::TensorPrimitive::QFloat(_) => unreachable!("ones is a float tensor"),
    };
    let message = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        <TtBackend as burn_backend::ops::ModuleOps<TtBackend>>::rfft(signal, 1, None)
    }))
    .unwrap_err();
    let message = message.downcast_ref::<String>().unwrap();
    assert!(message.contains("unsupported operation rfft"), "{message}");
    assert!(message.contains("[2, 3]"), "{message}");
    assert!(message.contains("F32"), "{message}");
    // Dtype rejection must report the input metadata, before attaching hardware.
    let message = std::panic::catch_unwind(|| {
        <TtBackend as burn_backend::ops::FloatTensorOps<TtBackend>>::float_from_data(
            TensorData::new(vec![1.0f64], [1]),
            &d,
        )
    })
    .unwrap_err();
    let message = message.downcast_ref::<String>().unwrap();
    assert!(
        message.contains("float_from_data") && message.contains("F64") && message.contains("[1]")
    );
}

#[test]
fn host_permutation_and_backend_axis_validation() {
    use burn_backend::{ops::FloatTensorOps, Shape};
    let d = TtDevice::new(305);
    let data = TensorData::new((0..24).map(|i| i as f32).collect::<Vec<_>>(), [2, 3, 4]);
    let result = Tensor::<TtBackend, 3>::from_data(data, &d)
        .permute([2, 0, 1])
        .into_data();
    assert_eq!(result.shape, Shape::new([4, 2, 3]));
    let expected: Vec<_> = (0..4)
        .flat_map(|k| (0..2).flat_map(move |i| (0..3).map(move |j| (i * 12 + j * 4 + k) as f32)))
        .collect();
    assert_eq!(result.to_vec::<f32>().unwrap(), expected);
    for axes in [vec![0, 0, 2], vec![0, 1], vec![0, 1, 3]] {
        let tensor = <TtBackend as FloatTensorOps<TtBackend>>::float_from_data(
            TensorData::new(vec![1.0f32; 24], [2, 3, 4]),
            &d,
        );
        let message = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <TtBackend as FloatTensorOps<TtBackend>>::float_permute(tensor, &axes)
        }))
        .unwrap_err();
        let message = message.downcast_ref::<String>().unwrap();
        assert!(message.contains("float_permute") && message.contains("axes="));
    }
}

#[test]
fn unsupported_boolean_conversion_dtypes_name_operation_and_input() {
    use burn_backend::{ops::BoolTensorOps, FloatDType, IntDType};
    let d = TtDevice::new(306);
    for float in [true, false] {
        let tensor = <TtBackend as BoolTensorOps<TtBackend>>::bool_from_data(
            TensorData::new(vec![true, false], [2]),
            &d,
        );
        let message = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if float {
                <TtBackend as BoolTensorOps<TtBackend>>::bool_into_float(tensor, FloatDType::F64)
            } else {
                <TtBackend as BoolTensorOps<TtBackend>>::bool_into_int(tensor, IntDType::I64)
            }
        }))
        .unwrap_err();
        let message = message.downcast_ref::<String>().unwrap();
        assert!(message.contains(if float {
            "bool_into_float"
        } else {
            "bool_into_int"
        }));
        assert!(
            message.contains("Bool") && message.contains("[2]") && message.contains("out_dtype=")
        );
    }
}

#[test]
fn invalid_reduction_axes_fail_with_metadata_before_device_access() {
    use burn_backend::ops::FloatTensorOps;
    use burn_tensor::TensorPrimitive;
    type Reduce = fn(burn_tt::TtTensor, usize) -> burn_tt::TtTensor;
    let d = TtDevice::new(305);
    let TensorPrimitive::Float(input) = Tensor::<TtBackend, 2>::ones([2, 3], &d).into_primitive()
    else {
        unreachable!()
    };
    let cases: [(&str, Reduce); 6] = [
        ("float_sum_dim", TtBackend::float_sum_dim),
        ("float_mean_dim", TtBackend::float_mean_dim),
        ("float_max_dim", TtBackend::float_max_dim),
        ("float_prod_dim", TtBackend::float_prod_dim),
        ("float_cumsum", TtBackend::float_cumsum),
        ("float_cumprod", TtBackend::float_cumprod),
    ];
    for (name, reduce) in cases {
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            reduce(input.clone(), usize::MAX)
        }))
        .unwrap_err();
        let message = panic.downcast_ref::<String>().unwrap();
        assert!(
            message.contains(name)
                && message.contains("[2, 3]")
                && message.contains("F32")
                && message.contains("axis="),
            "{message}"
        );
    }
    use burn_backend::ops::BoolTensorOps;
    let input = Tensor::<TtBackend, 2, burn_tensor::Bool>::from_data(
        [[true, false, true], [false, true, false]],
        &d,
    )
    .into_primitive();
    let cases: [(&str, Reduce); 2] = [
        ("bool_any_dim", TtBackend::bool_any_dim),
        ("bool_all_dim", TtBackend::bool_all_dim),
    ];
    for (name, reduce) in cases {
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            reduce(input.clone(), usize::MAX)
        }))
        .unwrap_err();
        let message = panic.downcast_ref::<String>().unwrap();
        assert!(
            message.contains(name)
                && message.contains("[2, 3]")
                && message.contains("Bool")
                && message.contains("axis="),
            "{message}"
        );
    }
}
