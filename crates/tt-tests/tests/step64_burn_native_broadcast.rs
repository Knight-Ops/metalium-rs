//! Device broadcast and transpose copies preserve logical bytes without host compute.
use burn_flex::{Flex, FlexDevice};
use burn_tensor::{Tensor, TensorData};
use burn_tt::TtBackend;
use tt_tests::burn_device::{with_device, Config};

#[test]
fn expanded_scalars_rows_and_columns_preserve_bits_on_the_device() {
    with_device(Config::default(), |d| {
        for (values, from, to) in [
            (vec![-0.0f32], [1, 1], [35, 37]),
            (vec![f32::from_bits(1)], [1, 1], [3, 5]),
            (vec![f32::INFINITY, -0.0, f32::NAN], [1, 3], [5, 3]),
            (vec![1.0, -2.0, 7.0], [3, 1], [3, 5]),
        ] {
            let data = TensorData::new(values, from);
            let want = Tensor::<Flex, 2>::from_data(data.clone(), &FlexDevice)
                .expand(to)
                .into_data();
            let ((got, resident), report) = burn_tt::with_report(|| {
                let result = Tensor::<TtBackend, 2>::from_data(data, &d).expand(to);
                let resident = result
                    .clone()
                    .into_primitive()
                    .tensor()
                    .computed_on_device();
                (result.into_data(), resident)
            });
            assert!(resident);
            assert_eq!(got.as_bytes(), want.as_bytes());
            let op = report.op("float_expand").unwrap();
            assert_eq!((op.on_host, op.downloads, op.staged), (0, 0, 0));
            assert!(op.on_device > 0);
            assert_eq!(report.op("float_into_data").unwrap().downloads, 1);
        }
    });
}

#[test]
fn scalar_broadcast_gradients_and_ragged_transposes_compute_natively() {
    with_device(Config::default(), |d| {
        let data = TensorData::new(
            (0..35 * 37)
                .map(|i| (i % 9) as f32 - 4.0)
                .collect::<Vec<_>>(),
            [35, 37],
        );
        let want = (Tensor::<Flex, 2>::from_data(data.clone(), &FlexDevice).transpose()
            * Tensor::<Flex, 2>::from_data([[3.0]], &FlexDevice))
        .into_data();
        let (got, report) = burn_tt::with_report(|| {
            (Tensor::<TtBackend, 2>::from_data(data, &d).transpose()
                * Tensor::<TtBackend, 2>::from_data([[3.0]], &d))
            .into_data()
        });
        assert_eq!(got.as_bytes(), want.as_bytes());
        let op = report.op("float_mul").unwrap();
        assert_eq!((op.on_host, op.downloads, op.staged), (0, 0, 0));
        assert!(op.on_device > 0);
    });
}
