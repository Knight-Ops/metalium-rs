//! Exact full-width division/remainder, including signed extremes and broadcasts.
use burn::tensor::{Int, Tensor, TensorData};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

#[test]
#[ignore = "WIP: packed domain flags falsely reject valid divisors; Session route disabled"]
fn integer_division_and_python_remainder_are_native_and_full_width() {
    with_device(Config::default(), |d| {
        let shape = [35, 37];
        let count = shape.iter().product();
        let cases = [
            i32::MIN,
            i32::MAX,
            -1,
            0,
            1,
            -3,
            3,
            0x7fffffff,
            0x40000000,
            0x7fffff,
            0x1fffffff,
        ];
        let a: Vec<_> = (0..count)
            .map(|i| {
                if i < cases.len() * cases.len() {
                    cases[i / cases.len()]
                } else {
                    (i as i32).wrapping_mul(123456789)
                }
            })
            .collect();
        let b: Vec<_> = (0..count)
            .map(|i| {
                let v = if i < cases.len() * cases.len() {
                    cases[i % cases.len()]
                } else {
                    (i as i32).wrapping_mul(0x61c88647)
                };
                if v == 0 {
                    1
                } else {
                    v
                }
            })
            .collect();
        let x = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(a.clone(), shape), &d);
        let y = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(b.clone(), shape), &d);
        let before = tensor_traffic();
        let ((q, r), report) =
            burn_tt::with_report(|| (x.clone() / y.clone(), x.clone().remainder(y)));
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        assert!(q.clone().into_primitive().computed_on_device());
        let div = |a: i32, b: i32| (a as i64 / b as i64) as i32;
        let rem = |a: i32, b: i32| {
            let (a, b) = (a as i64, b as i64);
            (((a % b) + b) % b) as i32
        };
        assert_eq!(
            q.into_data().to_vec::<i32>().unwrap(),
            a.iter()
                .zip(&b)
                .map(|(&a, &b)| div(a, b))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            r.into_data().to_vec::<i32>().unwrap(),
            a.iter()
                .zip(&b)
                .map(|(&a, &b)| rem(a, b))
                .collect::<Vec<_>>()
        );
        for divisor in [i32::MIN, -7, -1, 1, 7, i32::MAX] {
            let q = x.clone().div_scalar(divisor);
            let r = x.clone().remainder_scalar(divisor);
            assert_eq!(
                q.into_data().to_vec::<i32>().unwrap(),
                a.iter().map(|&a| div(a, divisor)).collect::<Vec<_>>()
            );
            assert_eq!(
                r.into_data().to_vec::<i32>().unwrap(),
                a.iter().map(|&a| rem(a, divisor)).collect::<Vec<_>>()
            );
        }
        let row: Vec<_> = (0..37)
            .map(|i| cases[i % cases.len()])
            .map(|i| if i == 0 { 1 } else { i })
            .collect();
        let rhs = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(row.clone(), [1, 37]), &d);
        assert_eq!(
            (x.clone() / rhs.clone())
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            a.iter()
                .enumerate()
                .map(|(i, &a)| div(a, row[i % 37]))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            x.remainder(rhs).into_data().to_vec::<i32>().unwrap(),
            a.iter()
                .enumerate()
                .map(|(i, &a)| rem(a, row[i % 37]))
                .collect::<Vec<_>>()
        );
    });
}

#[test]
#[ignore = "WIP: packed domain flags falsely reject valid divisors; Session route disabled"]
fn zero_divisors_report_a_device_domain_error_without_host_tensor_download() {
    use tt_kernels::{
        session::{Session, TileChoice},
        sfpu::ops::kind_sfpu,
        tensor::{Elem, Eltwise},
    };
    tt_ttsim::fork_scope(|| {
        #[cfg(not(feature = "silicon"))]
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        #[cfg(not(feature = "silicon"))]
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        #[cfg(not(feature = "silicon"))]
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(1),
            |_, _| Ok(None),
        )
        .unwrap();
        #[cfg(feature = "silicon")]
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(1),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        let a = s.upload_bits(&[7; 35], 1, 35, Elem::I32).unwrap();
        let mut divisors = vec![1u32; 35];
        divisors[34] = 0;
        let b = s.upload_bits(&divisors, 1, 35, Elem::I32).unwrap();
        let result = s.eltwise(
            Eltwise {
                kind: kind_sfpu::INT_DIV,
                scalar: 0.0,
                scalar2: 0.0,
            },
            &a,
            Some(&b),
        );
        let failed = match result {
            Err(e) => e.to_string(),
            Ok(_) => s.sync().unwrap_err().to_string(),
        };
        assert!(failed.contains("11"), "expected DOMAIN (11), got {failed}");
    })
    .unwrap();
}
