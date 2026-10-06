//! Exact full-width division/remainder, including signed extremes and broadcasts.
use burn::tensor::{Int, Tensor, TensorData};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

#[test]
fn packed_domain_flags_match_each_physical_face() {
    use tt_isa::{
        backend::{self, Before},
        tile::{L1Format, TileImage},
    };
    use tt_kernels::{
        datapath,
        runtime::{self, Kernel, Schedule},
        sfpu::{kernel, ops, Cond, Format, LReg, Program},
    };
    use tt_tests::harness;
    harness::in_device(|dev| {
        let layout = kernel::plan_layout(1, kernel::Operands::Binary).unwrap();
        let image = TileImage::new(datapath::tile_descriptor(), L1Format::Fp32).unwrap();
        let words: Vec<u32> = (0..1024)
            .map(|i| {
                if [0, 15, 16, 255, 256, 511, 512, 767, 768, 1023].contains(&i) {
                    0
                } else if i % 5 == 0 {
                    i32::MIN as u32
                } else {
                    7
                }
            })
            .collect();
        let encode = |words: &[u32]| {
            let mut bytes = vec![0; image.total_bytes()];
            for (i, word) in words.iter().enumerate() {
                let at = image.datum_bit_offset(i) / 8;
                bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
            bytes
        };
        let a = encode(&vec![21; 1024]);
        let b = encode(&words);
        let mut flags = Program::new();
        flags.for_each_row_group(64, |p, o| {
            p.load(LReg::L1, Format::Int32, kernel::B_ROW + o);
            p.loadi_bits(LReg::L3, 1);
            p.if_(Cond::Eq0(LReg::L1), |p| p.mov(LReg::ZERO, LReg::L3));
            p.store(LReg::L3, Format::Int32, kernel::C_ROW + o);
            p.store(LReg::L1, Format::Int32, kernel::OUT_ROW + o);
        });
        for code in [
            flags.finish_code(),
            ops::code2(ops::kind_sfpu::INT_DIV, [0.0; 2]).unwrap().1,
        ] {
            let (mut roles, loops) = kernel::roles_code(&layout, kernel::Operands::Binary, &code);
            roles[2].extend(datapath::pack_tile_from_dst(
                layout.b_at.unwrap() + 16,
                kernel::C_ROW,
            ));
            roles[2].push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
            let stage = [
                (layout.a_at, a.as_slice()),
                (layout.b_at.unwrap(), b.as_slice()),
            ];
            let read_back = [(layout.b_at.unwrap() + 16, 4096)];
            let run = Kernel {
                stage: &stage,
                read_back: &read_back,
                loops: [&loops[0], &loops[1], &loops[2]],
                ..Kernel::new(
                    [&roles[0], &roles[1], &roles[2]],
                    Schedule::Concurrent(&layout.init),
                )
            };
            let out = runtime::run(
                dev,
                harness::tensix_tile(),
                &tt_tests::firmware::ROLES,
                &run,
                harness::BUDGET,
            )
            .unwrap();
            let got: Vec<_> = out.l1[0]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            for (i, (&got, &divisor)) in got.iter().zip(&words).enumerate() {
                assert_eq!(got, u32::from(divisor != 0), "physical lane {i}");
            }
        }
    });
}

#[test]
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
            x.clone()
                .remainder(rhs)
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            a.iter()
                .enumerate()
                .map(|(i, &a)| rem(a, row[i % 37]))
                .collect::<Vec<_>>()
        );
        let col: Vec<_> = (0..35).map(|i| if i % 2 == 0 { -7 } else { 3 }).collect();
        let rhs = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(col.clone(), [35, 1]), &d);
        assert_eq!(
            (x.clone() / rhs.clone())
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            a.iter()
                .enumerate()
                .map(|(i, &a)| div(a, col[i / 37]))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            x.remainder(rhs).into_data().to_vec::<i32>().unwrap(),
            a.iter()
                .enumerate()
                .map(|(i, &a)| rem(a, col[i / 37]))
                .collect::<Vec<_>>()
        );
    });
}

#[test]
fn integer_mean_wraps_before_division_on_permuted_axes() {
    with_device(Config::default(), |d| {
        let values: Vec<i32> = (0usize..3 * 5 * 37)
            .map(|i| (i as i32).wrapping_mul(123456789).wrapping_add(i32::MAX))
            .collect();
        let x =
            Tensor::<TtBackend, 3, Int>::from_data(TensorData::new(values.clone(), [3, 5, 37]), &d)
                .swap_dims(0, 2);
        let before = tensor_traffic();
        let ((a, b, c, full), report) = burn_tt::with_report(|| {
            (
                x.clone().mean_dim(0),
                x.clone().mean_dim(1),
                x.clone().mean_dim(2),
                x.mean(),
            )
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        for (axis, result) in [a, b, c].into_iter().enumerate() {
            assert!(result.clone().into_primitive().computed_on_device());
            let mut shape = [37, 5, 3];
            let count = shape[axis];
            shape[axis] = 1;
            let mut expected = Vec::new();
            for c in 0..shape[0] {
                for b in 0..shape[1] {
                    for a in 0..shape[2] {
                        let mut sum = 0i32;
                        for k in 0..count {
                            let mut at = [c, b, a];
                            at[axis] = k;
                            sum = sum.wrapping_add(values[(at[2] * 5 + at[1]) * 37 + at[0]]);
                        }
                        expected.push(sum / count as i32);
                    }
                }
            }
            assert_eq!(result.into_data().to_vec::<i32>().unwrap(), expected);
        }
        assert_eq!(
            full.into_data().to_vec::<i32>().unwrap(),
            vec![values.iter().fold(0i32, |a, &b| a.wrapping_add(b)) / values.len() as i32]
        );
    });
}

#[test]
fn zero_divisors_report_a_device_domain_error_without_host_tensor_download() {
    use tt_kernels::{
        session::{Session, TileChoice},
        sfpu::ops::kind_sfpu,
        tensor::{Elem, Eltwise},
    };
    for zero in [
        0,
        15,
        16,
        31,
        32,
        15 * 37,
        16 * 37,
        31 * 37,
        32 * 37,
        35 * 37 - 1,
        35 * 37, // Scalar zero, without a tensor divisor.
    ] {
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
            let a = s.upload_bits(&[7; 35 * 37], 35, 37, Elem::I32).unwrap();
            let valid = s.upload_bits(&[1; 35 * 37], 35, 37, Elem::I32).unwrap();
            let out = s
                .eltwise(
                    Eltwise {
                        kind: kind_sfpu::INT_DIV,
                        scalar: 0.0,
                        scalar2: 0.0,
                    },
                    &a,
                    Some(&valid),
                )
                .unwrap();
            assert_eq!(s.download_bits(&out).unwrap(), vec![7; 35 * 37]);
            let mut divisors = vec![1u32; 35 * 37];
            if zero < divisors.len() {
                divisors[zero] = 0;
            }
            let b = s.upload_bits(&divisors, 35, 37, Elem::I32).unwrap();
            let scalar_zero = zero == 35 * 37;
            let result = s.eltwise(
                Eltwise {
                    kind: if scalar_zero {
                        kind_sfpu::INT_DIV_S
                    } else {
                        kind_sfpu::INT_DIV
                    },
                    scalar: 0.0,
                    scalar2: 0.0,
                },
                &a,
                if scalar_zero { None } else { Some(&b) },
            );
            let failed = match result {
                Err(e) => e.to_string(),
                Ok(_) => s.sync().unwrap_err().to_string(),
            };
            assert!(failed.contains("11"), "expected DOMAIN (11), got {failed}");
        })
        .unwrap();
    }
}

#[test]
fn checked_integer_trace_replays_changed_divisors_on_two_tiles() {
    use burn::tensor::TensorPrimitive;
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            let divisors = Tensor::<TtBackend, 2>::ones([35, 37], &d).mul_scalar(1.0);
            let numerator = Tensor::<TtBackend, 2, Int>::from_data(
                TensorData::new(vec![7i32; 35 * 37], [35, 37]),
                &d,
            )
            .add_scalar(0);
            let TensorPrimitive::Float(input) = divisors.clone().into_primitive() else {
                unreachable!()
            };
            let before = tensor_traffic();
            let ((trace, first), report) = burn_tt::with_report(|| {
                burn_tt::Trace::capture(&input, || {
                    let out = (numerator.clone() / divisors.clone().int()).float();
                    let TensorPrimitive::Float(out) = out.into_primitive() else {
                        unreachable!()
                    };
                    out
                })
                .unwrap()
            });
            assert_native_model(&report);
            assert_eq!(tensor_traffic().uploads, before.uploads);
            assert_eq!(first, vec![7.0; 35 * 37]);
            for value in [2.0, -2.0, 4.0] {
                assert_eq!(
                    trace.run(vec![value; 35 * 37]).unwrap(),
                    vec![(7i32 / value as i32) as f32; 35 * 37]
                );
            }
            assert!(trace
                .run(vec![0.0; 35 * 37])
                .unwrap_err()
                .to_string()
                .contains("11"));
        },
    );
}
