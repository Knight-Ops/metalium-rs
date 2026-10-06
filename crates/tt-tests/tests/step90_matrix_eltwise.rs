//! Matrix-unit elementwise arithmetic, resident lifetimes and Burn opt-in.
use tt_kernels::{
    matmul::Fidelity,
    session::{Session, TileChoice},
};
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
type Transport<'a> = tt_ttsim::LibTtsim<'a>;
#[cfg(feature = "silicon")]
type Transport<'a> = tt_kmd::Kmd;
fn with_session(tiles: usize, f: impl FnOnce(&mut Session<Transport<'_>>)) {
    if let Err(e) = fork_scope(|| {
        #[cfg(not(feature = "silicon"))]
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        #[cfg(not(feature = "silicon"))]
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        #[cfg(not(feature = "silicon"))]
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(tiles),
            |_, _| Ok(None),
        )
        .unwrap();
        #[cfg(feature = "silicon")]
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(tiles),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[test]
fn resident_matrix_arithmetic_and_rhs_broadcasts() {
    with_session(2, |s| {
        use tt_kernels::matrix_eltwise::{MatrixEltwiseOp as Op, SrcPrecision};
        for (rows, cols) in [(16, 16), (32, 32), (37, 65)] {
            let a: Vec<_> = (0..rows * cols).map(|i| (i % 13 + 1) as f32).collect();
            let ta = s.upload(&a, rows, cols).unwrap();
            let abits: Vec<_> = a.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let pa = s.upload_bf16(&abits, rows, cols).unwrap();
            for dims in [[rows, cols], [1, cols], [rows, 1], [1, 1]] {
                let b: Vec<_> = (0..dims[0] * dims[1]).map(|i| (i % 7 + 1) as f32).collect();
                let tb = s.upload(&b, dims[0], dims[1]).unwrap();
                let bbits: Vec<_> = b.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
                let pb = s.upload_bf16(&bbits, dims[0], dims[1]).unwrap();
                for op in [Op::Add, Op::Sub, Op::Mul] {
                    for (precision, packed) in [
                        (SrcPrecision::Tf32, false),
                        (SrcPrecision::Bf16, false),
                        (SrcPrecision::Bf16, true),
                    ] {
                        let before = s.dataflow_stats().transfer_packets;
                        let out = if packed {
                            s.matrix_eltwise_bf16(op, &pa, &pb, Fidelity::HiFi4)
                        } else {
                            s.matrix_eltwise(op, &ta, &tb, precision, Fidelity::HiFi4)
                        }
                        .unwrap();
                        assert_eq!(
                            s.dataflow_stats().transfer_packets - before,
                            0,
                            "matrix RHS {dims:?} must not stage an expanded tensor"
                        );
                        let got = s.download(&out).unwrap();
                        for r in 0..rows {
                            for c in 0..cols {
                                let (a, b) =
                                    (a[r * cols + c], b[(r % dims[0]) * dims[1] + c % dims[1]]);
                                let want = match op {
                                    Op::Add => a + b,
                                    Op::Sub => a - b,
                                    Op::Mul => a * b,
                                };
                                assert_eq!(
                                    got[r * cols + c],
                                    want,
                                    "{op:?} {precision:?} {rows}x{cols} RHS {dims:?} at {r},{c}"
                                );
                            }
                        }
                        s.free(out).unwrap();
                    }
                }
                s.free(tb).unwrap();
                s.free_bf16(pb).unwrap();
            }
            s.free(ta).unwrap();
            s.free_bf16(pa).unwrap();
        }
    });
}

#[test]
fn packed_bf16_and_fidelity_phases_match_an_independent_oracle() {
    use tt_isa::numerics::{elw_reference, MatrixEltwiseOp as Op};
    use tt_kernels::matrix_eltwise::{SrcBroadcast, SrcPrecision};
    with_session(2, |s| {
        let (rows, cols) = (37, 65);
        let a: Vec<_> = (0..rows * cols)
            .map(|i| 1.03125 + (i % 3) as f32 / 8.0)
            .collect();
        let b: Vec<_> = (0..rows * cols)
            .map(|i| 1.0078125 + (i % 5) as f32 / 16.0)
            .collect();
        let aa: Vec<_> = a.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let bb: Vec<_> = b.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let pa = s.upload_bf16(&aa, rows, cols).unwrap();
        let pb = s.upload_bf16(&bb, rows, cols).unwrap();
        let ta = s.upload(&a, rows, cols).unwrap();
        let tb = s.upload(&b, rows, cols).unwrap();
        for fidelity in [
            Fidelity::Lo,
            Fidelity::HiFi2,
            Fidelity::HiFi3,
            Fidelity::HiFi4,
        ] {
            let phases: Vec<_> = (0..fidelity.phases()).collect();
            let want: Vec<_> = a
                .iter()
                .zip(&b)
                .map(|(&a, &b)| {
                    elw_reference(
                        &[[0.0; 16]; 8],
                        &[[a; 16]; 8],
                        &[[b; 16]; 8],
                        Op::Mul,
                        SrcBroadcast::None,
                        &phases,
                    )
                    .unwrap()[0][0]
                })
                .collect();
            for packed in [false, true] {
                let out = if packed {
                    s.matrix_eltwise_bf16(Op::Mul, &pa, &pb, fidelity)
                } else {
                    s.matrix_eltwise(Op::Mul, &ta, &tb, SrcPrecision::Tf32, fidelity)
                }
                .unwrap();
                assert_eq!(
                    s.download(&out).unwrap(),
                    want,
                    "{fidelity:?} packed={packed}"
                );
                s.free(out).unwrap();
            }
        }
        s.free(ta).unwrap();
        s.free(tb).unwrap();
        s.free_bf16(pa).unwrap();
        s.free_bf16(pb).unwrap();
    });
}

#[test]
fn changed_input_traces_hold_freed_operands_and_preserve_padding() {
    use tt_kernels::{
        matrix_eltwise::{MatrixEltwiseOp as Op, SrcPrecision},
        tensor::Pad,
    };
    for tiles in [1, 2] {
        with_session(tiles, |s| {
            let dims = [37, 65];
            let a = s
                .upload(&vec![2.0; dims[0] * dims[1]], dims[0], dims[1])
                .unwrap();
            let b = s.upload(&vec![3.0; dims[1]], 1, dims[1]).unwrap();
            let before = (a.pad(), b.pad());
            s.begin_trace().unwrap();
            let product = s
                .matrix_eltwise(Op::Mul, &a, &b, SrcPrecision::Tf32, Fidelity::HiFi4)
                .unwrap();
            assert_eq!(product.pad(), Pad::Undefined);
            let sum = s.sum_rows(&product).unwrap();
            s.free(product).unwrap();
            let trace = s.end_trace().unwrap();
            assert_eq!((a.pad(), b.pad()), before);
            s.free(b).unwrap();
            for value in [2.0, 5.0, -3.0] {
                s.write(&a, &vec![value; dims[0] * dims[1]]).unwrap();
                s.replay(trace).unwrap();
                assert_eq!(
                    s.download(&sum).unwrap(),
                    vec![value * 3.0 * dims[0] as f32; dims[1]]
                );
                assert_eq!(a.pad(), before.0);
            }
            s.release_trace(trace).unwrap();
            s.free(sum).unwrap();
            s.free(a).unwrap();
        });
    }
}

#[test]
fn burn_opt_in_preserves_packed_storage_and_analytic_gradients() {
    use burn::{
        backend::Autodiff,
        tensor::{FloatDType, Tensor, TensorData},
    };
    use burn_tt::{ElementwiseMode, SrcPrecision, TtBackend};
    use tt_tests::burn_device::{assert_native_model, with_device, Config};
    with_device(
        Config {
            elementwise: ElementwiseMode::Matrix {
                precision: SrcPrecision::Tf32,
                fidelity: Fidelity::HiFi4,
            },
            tiles: Some(TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            for dtype in if cfg!(feature = "silicon") {
                vec![FloatDType::F32, FloatDType::BF16]
            } else {
                vec![FloatDType::F32]
            } {
                let a = Tensor::<TtBackend, 2>::from_data(
                    TensorData::new(vec![2.0; 37 * 65], [37, 65]),
                    &d,
                )
                .cast(dtype)
                .to_device(&d);
                let b =
                    Tensor::<TtBackend, 2>::from_data(TensorData::new(vec![3.0; 65], [1, 65]), &d)
                        .cast(dtype)
                        .to_device(&d);
                let (out, report) =
                    burn_tt::with_report(|| (a.clone() + b.clone() - b.clone()) * b.clone());
                assert_native_model(&report);
                assert_eq!(report.0.iter().map(|op| op.matrix_eltwise).sum::<u64>(), 3);
                assert_eq!(
                    out.into_data().convert::<f32>().to_vec::<f32>().unwrap(),
                    vec![6.0; 37 * 65]
                );
                let (out, report) =
                    burn_tt::with_report(|| a.add_scalar(4.0).sub_scalar(1.0).mul_scalar(2.0));
                assert_native_model(&report);
                assert_eq!(report.0.iter().map(|op| op.matrix_eltwise).sum::<u64>(), 3);
                assert_eq!(
                    out.into_data().convert::<f32>().to_vec::<f32>().unwrap(),
                    vec![10.0; 37 * 65]
                );
            }
            type AD = Autodiff<TtBackend>;
            for dtype in if cfg!(feature = "silicon") {
                vec![FloatDType::F32, FloatDType::BF16]
            } else {
                vec![FloatDType::F32]
            } {
                let a =
                    Tensor::<AD, 2>::from_data(TensorData::new(vec![2.0; 37 * 65], [37, 65]), &d)
                        .cast(dtype)
                        .to_device(&d)
                        .require_grad();
                let b = Tensor::<AD, 2>::from_data(TensorData::new(vec![3.0; 65], [1, 65]), &d)
                    .cast(dtype)
                    .to_device(&d)
                    .require_grad();
                let (grad, report) = burn_tt::with_report(|| {
                    ((a.clone() + b.clone() - b.clone()) * b.clone())
                        .sum()
                        .backward()
                });
                assert_native_model(&report);
                assert_eq!(
                    a.grad(&grad)
                        .unwrap()
                        .into_data()
                        .convert::<f32>()
                        .to_vec::<f32>()
                        .unwrap(),
                    vec![3.0; 37 * 65]
                );
                assert_eq!(
                    b.grad(&grad)
                        .unwrap()
                        .into_data()
                        .convert::<f32>()
                        .to_vec::<f32>()
                        .unwrap(),
                    vec![74.0; 65]
                );
            }
        },
    );
}

#[test]
fn matrix_special_value_characterization() {
    use tt_kernels::matrix_eltwise::{MatrixEltwiseOp as Op, SrcPrecision};
    let pairs = [
        (0, 0),
        (0x80000000, 0),
        (0x80000000, 0x80000000),
        (1, 0),
        (0x80000001, 0),
        (0x007fffff, 0),
        (0x807fffff, 0),
        (0x7f800000, 0x3f800000),
        (0xff800000, 0x3f800000),
        (0x7f800000, 0xff800000),
        (0x7f800000, 0),
        (0x7fc12345, 0x3f800000),
        (0xffc12345, 0x3f800000),
        (0x7f7fffff, 0x7f7fffff),
        (0xff7fffff, 0x7f7fffff),
        (0x00800000, 0x3f000000),
        (0x00800000, 0x00800000),
    ];
    with_session(1, |s| {
        let a: Vec<_> = pairs.iter().map(|p| f32::from_bits(p.0)).collect();
        let b: Vec<_> = pairs.iter().map(|p| f32::from_bits(p.1)).collect();
        let ta = s.upload(&a, 1, a.len()).unwrap();
        let tb = s.upload(&b, 1, b.len()).unwrap();
        let abits: Vec<_> = a.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let bbits: Vec<_> = b.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let pa = s.upload_bf16(&abits, 1, a.len()).unwrap();
        let pb = s.upload_bf16(&bbits, 1, b.len()).unwrap();
        for (precision, packed) in [
            (SrcPrecision::Tf32, false),
            (SrcPrecision::Bf16, false),
            (SrcPrecision::Bf16, true),
        ] {
            for op in [Op::Add, Op::Sub, Op::Mul] {
                let out = if packed {
                    s.matrix_eltwise_bf16(op, &pa, &pb, Fidelity::HiFi4)
                } else {
                    s.matrix_eltwise(op, &ta, &tb, precision, Fidelity::HiFi4)
                }
                .unwrap();
                let bits: Vec<_> = s
                    .download(&out)
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect();
                // Both-card measured special-value contract, run 1791254100.
                let expected = match op {
                    Op::Add => [
                        0, 0, 0, 0, 0, 0, 0, 0x7f800000, 0xff800000, 0, 0x7f800000, 0x7f800000,
                        0xff800000, 0x7f800000, 0, 0x3f000000, 0x01000000,
                    ],
                    Op::Sub => [
                        0, 0, 0, 0, 0, 0, 0, 0x7f800000, 0xff800000, 0x7f800000, 0x7f800000,
                        0x7f800000, 0xff800000, 0, 0xff800000, 0xbf000000, 0,
                    ],
                    Op::Mul => [
                        0, 0, 0, 0, 0, 0, 0, 0x7f800000, 0xff800000, 0, 0, 0x7f800000, 0xff800000,
                        0x7f800000, 0xff800000, 0, 0,
                    ],
                };
                assert_eq!(bits, expected, "{precision:?} {op:?}");
                s.free(out).unwrap();
            }
        }
        s.free(ta).unwrap();
        s.free(tb).unwrap();
        s.free_bf16(pa).unwrap();
        s.free_bf16(pb).unwrap();
    });
}

/// Absolute error budget, derived from Src truncation and FP32 accumulation.
/// Normal operands/products only; special-value behavior has a separate gate.
#[test]
fn finite_precision_bounds_hold_across_exponents() {
    use tt_kernels::matrix_eltwise::{MatrixEltwiseOp as Op, SrcPrecision};
    with_session(2, |s| {
        let n: usize = 37 * 65;
        let a: Vec<_> = (0..n)
            .map(|i| {
                (1.0 + (i % 1021) as f32 / 1024.0)
                    * 2f32.powi((i % 41) as i32 - 20)
                    * if i % 3 == 0 { -1.0 } else { 1.0 }
            })
            .collect();
        let b: Vec<_> = (0..n)
            .map(|i| {
                (1.0 + (i * 7 % 997) as f32 / 1024.0)
                    * 2f32.powi((i * 11 % 41) as i32 - 20)
                    * if i % 5 == 0 { -1.0 } else { 1.0 }
            })
            .collect();
        let ta = s.upload(&a, 37, 65).unwrap();
        let tb = s.upload(&b, 37, 65).unwrap();
        for precision in [SrcPrecision::Tf32, SrcPrecision::Bf16] {
            for op in [Op::Add, Op::Sub, Op::Mul] {
                for fidelity in [
                    Fidelity::Lo,
                    Fidelity::HiFi2,
                    Fidelity::HiFi3,
                    Fidelity::HiFi4,
                ] {
                    let out = s.matrix_eltwise(op, &ta, &tb, precision, fidelity).unwrap();
                    let got = s.download(&out).unwrap();
                    for (i, (&a, &b)) in a.iter().zip(&b).enumerate() {
                        let (a, b) = (f64::from(a), f64::from(b));
                        // Independent f64 grid truncation to Src precision.
                        let truncate = |x: f64, bits: i32| {
                            let exp = x.abs().log2().floor() as i32;
                            let quantum = 2f64.powi(exp - bits);
                            (x / quantum).trunc() * quantum
                        };
                        let bits = if precision == SrcPrecision::Tf32 {
                            10
                        } else {
                            7
                        };
                        let (qa, qb) = (truncate(a, bits), truncate(b, bits));
                        let (expected, budget) = if op == Op::Mul {
                            // SrcA's final TF32 fraction bit is consumed by no
                            // phase. Each phase product has at most 12 significant
                            // bits. Its product is exact in this normal domain.
                            let qa = truncate(qa, if bits == 10 { 9 } else { 7 });
                            let (ah, bh) = (truncate(qa, 4), truncate(qb, 6));
                            let (al, bl) = (qa - ah, qb - bh);
                            let products = [ah * bh, al * bh, ah * bl, al * bl];
                            let kept: f64 = products[..fidelity.phases() as usize].iter().sum();
                            let omitted = (qa * qb - kept).abs();
                            // Triangle bound for conversion and omitted phases;
                            // gamma_8 covers four multiplies/four accumulations
                            // at FP32 unit roundoff. This is a domain gate, not
                            // a promise about exceptional/underflowing arithmetic.
                            let u = 2f64.powi(-23);
                            let gamma = 8.0 * u / (1.0 - 8.0 * u);
                            let mag: f64 = products.iter().map(|x| x.abs()).sum();
                            (a * b, (a * b - qa * qb).abs() + omitted + gamma * mag)
                        } else {
                            let expected = if op == Op::Add { a + b } else { a - b };
                            // Shared 10-fraction-bit alignment (step90 probe),
                            // even for BF16. Two aligned operands lose at most
                            // one common quantum each; the sum then fits FP32.
                            let exp = qa.abs().max(qb.abs()).log2().floor() as i32;
                            let q = 2f64.powi(exp - 10);
                            (expected, (a - qa).abs() + (b - qb).abs() + 2.0 * q)
                        };
                        assert!((f64::from(got[i])-expected).abs()<=budget,"{op:?} {precision:?} {fidelity:?} at {i}: {} vs {expected}, bound {budget}",got[i]);
                    }
                    s.free(out).unwrap();
                }
            }
        }
        s.free(ta).unwrap();
        s.free(tb).unwrap();
    });
}

#[test]
#[ignore = "benchmark"]
#[cfg(feature = "silicon")]
fn matrix_vs_sfpu_release_baseline() {
    use std::time::Instant;
    use tt_kernels::{
        kind,
        matrix_eltwise::{MatrixEltwiseOp as Op, SrcPrecision},
        tensor::Eltwise,
    };
    use tt_tests::bench::{report, Stats};
    let release = !cfg!(debug_assertions);
    with_session(2, |s| {
        println!("BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{release},\"tiles\":2,\"warmups\":2,\"samples\":7,\"timing\":\"host dispatch through sync, validation outside timing\",\"fidelity\":\"HiFi4\"}}",tt_tests::bench::git_sha());
        for (rows, cols) in [(64, 64), (65, 70)] {
            let a = vec![2.0; rows * cols];
            let ta = s.upload(&a, rows, cols).unwrap();
            let pa = s
                .upload_bf16(&vec![0x4000; rows * cols], rows, cols)
                .unwrap();
            for dims in [[rows, cols], [1, cols], [rows, 1], [1, 1]] {
                let tb = s
                    .upload(&vec![3.0; dims[0] * dims[1]], dims[0], dims[1])
                    .unwrap();
                let pb = s
                    .upload_bf16(&vec![0x4040; dims[0] * dims[1]], dims[0], dims[1])
                    .unwrap();
                for (op, kind, want) in [
                    (Op::Add, kind::ADD, 5.0f32),
                    (Op::Sub, kind::SUB, -1.0),
                    (Op::Mul, kind::MUL, 6.0),
                ] {
                    for packed in [false, true] {
                        for matrix in [false, true] {
                            let before = s.dataflow_stats().clone();
                            let mut samples = Vec::new();
                            for rep in 0..9 {
                                let start = Instant::now();
                                let mut temps = Vec::new();
                                let out = if matrix {
                                    if packed {
                                        s.matrix_eltwise_bf16(op, &pa, &pb, Fidelity::HiFi4)
                                    } else {
                                        s.matrix_eltwise(
                                            op,
                                            &ta,
                                            &tb,
                                            SrcPrecision::Tf32,
                                            Fidelity::HiFi4,
                                        )
                                    }
                                } else {
                                    let scalar = dims == [1, 1];
                                    let operation = Eltwise {
                                        kind: if scalar {
                                            match op {
                                                Op::Add | Op::Sub => kind::ADD_SCALAR,
                                                Op::Mul => kind::MUL_SCALAR,
                                            }
                                        } else {
                                            kind
                                        },
                                        scalar: if scalar {
                                            if op == Op::Sub {
                                                -3.0
                                            } else {
                                                3.0
                                            }
                                        } else {
                                            0.0
                                        },
                                        scalar2: 0.0,
                                    };
                                    if packed {
                                        let a = s.bf16_to_f32(&pa).unwrap();
                                        let b = if scalar {
                                            None
                                        } else {
                                            Some(s.bf16_to_f32(&pb).unwrap())
                                        };
                                        let out = s.eltwise(operation, &a, b.as_ref());
                                        temps.push(a);
                                        temps.extend(b);
                                        out
                                    } else {
                                        s.eltwise(
                                            operation,
                                            &ta,
                                            if scalar { None } else { Some(&tb) },
                                        )
                                    }
                                }
                                .unwrap();
                                let narrowed = if packed {
                                    Some(s.bf16_from_f32(&out).unwrap())
                                } else {
                                    None
                                };
                                s.sync().unwrap();
                                let elapsed = start.elapsed().as_secs_f64() * 1e6;
                                if rep >= 2 {
                                    samples.push(elapsed);
                                }
                                if let Some(t) = narrowed {
                                    assert_eq!(
                                        s.download_bf16(&t).unwrap(),
                                        vec![(want.to_bits() >> 16) as u16; rows * cols]
                                    );
                                    s.free_bf16(t).unwrap();
                                } else {
                                    assert_eq!(s.download(&out).unwrap(), vec![want; rows * cols]);
                                }
                                s.free(out).unwrap();
                                for t in temps {
                                    s.free(t).unwrap();
                                }
                            }
                            let after = s.dataflow_stats();
                            let key = format!(
                                "ELW {op:?} {} {} [{rows},{cols}] RHS{dims:?}",
                                if packed { "BF16" } else { "F32" },
                                if matrix { "matrix" } else { "SFPU" }
                            );
                            report(&key, "us", "host", Stats::of(samples));
                            println!("BENCH {{\"kind\":\"dataflow\",\"key\":\"{key}\",\"regions\":{},\"batches\":{},\"pack_waits\":{},\"release_waits\":{},\"transfer_packets\":{}}}",after.regions-before.regions,after.batches-before.batches,after.pack_waits-before.pack_waits,after.release_waits-before.release_waits,after.transfer_packets-before.transfer_packets);
                        }
                    }
                }
                s.free(tb).unwrap();
                s.free_bf16(pb).unwrap();
            }
            s.free(ta).unwrap();
            s.free_bf16(pa).unwrap();
        }
    });
}

#[test]
fn addition_alignment_quantum_probe() {
    use tt_kernels::matrix_eltwise::{MatrixEltwiseOp as Op, SrcPrecision};
    with_session(1, |s| {
        for precision in [SrcPrecision::Tf32, SrcPrecision::Bf16] {
            let a: Vec<_> = (0..41)
                .map(|i| (1.0 + 1.0 / 1024.0) * 2f32.powi(i - 20))
                .collect();
            let b = vec![1.0 + 7.0 / 1024.0; 41];
            let ta = s.upload(&a, 1, 41).unwrap();
            let tb = s.upload(&b, 1, 41).unwrap();
            for op in [Op::Add, Op::Sub] {
                let out = s
                    .matrix_eltwise(op, &ta, &tb, precision, Fidelity::Lo)
                    .unwrap();
                let got = s.download(&out).unwrap();
                let truncate = |x: f64| {
                    let exp = x.abs().log2().floor() as i32;
                    let q = 2f64.powi(
                        exp - if precision == SrcPrecision::Tf32 {
                            10
                        } else {
                            7
                        },
                    );
                    (x / q).trunc() * q
                };
                for (i, (&a, &b)) in a.iter().zip(&b).enumerate() {
                    let (a, b) = (truncate(f64::from(a)), truncate(f64::from(b)));
                    let exp = a.abs().max(b.abs()).log2().floor() as i32;
                    let q = 2f64.powi(exp - 10);
                    let align = |x: f64| x.signum() * (x.abs() / q + 0.5).floor() * q;
                    let want = if op == Op::Add {
                        align(a) + align(b)
                    } else {
                        align(a) - align(b)
                    };
                    assert_eq!(
                        f64::from(got[i]),
                        want,
                        "{precision:?} {op:?} exponent difference {}",
                        i as i32 - 20
                    );
                }
                s.free(out).unwrap();
            }
            s.free(ta).unwrap();
            s.free(tb).unwrap();
        }
    });
}

#[test]
fn ragged_views_and_matmul_consumers_preserve_the_parent() {
    use burn::tensor::{FloatDType, Tensor, TensorData};
    use burn_tt::{ElementwiseMode, SrcPrecision, TtBackend};
    use tt_tests::burn_device::{assert_native_model, with_device, Config};
    with_device(
        Config {
            elementwise: ElementwiseMode::Matrix {
                precision: SrcPrecision::Tf32,
                fidelity: Fidelity::HiFi4,
            },
            tiles: Some(TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            for dtype in if cfg!(feature = "silicon") {
                vec![FloatDType::F32, FloatDType::BF16]
            } else {
                vec![FloatDType::F32]
            } {
                let parent = Tensor::<TtBackend, 2>::from_data(
                    TensorData::new(vec![2.0; 41 * 71], [41, 71]),
                    &d,
                )
                .cast(dtype)
                .to_device(&d);
                let view = parent.clone().slice([2..39, 3..68]).transpose();
                let b =
                    Tensor::<TtBackend, 2>::from_data(TensorData::new(vec![3.0; 65], [65, 1]), &d)
                        .cast(dtype)
                        .to_device(&d);
                let w = Tensor::<TtBackend, 2>::from_data(
                    TensorData::new(vec![1.0; 37 * 3], [37, 3]),
                    &d,
                )
                .cast(dtype)
                .to_device(&d);
                let (out, report) = burn_tt::with_report(|| (view * b).matmul(w));
                assert_native_model(&report);
                assert_eq!(report.0.iter().map(|op| op.matrix_eltwise).sum::<u64>(), 1);
                // BF16 222 is exact, so one narrowing introduces no rounding here.
                assert_eq!(
                    out.into_data().convert::<f32>().to_vec::<f32>().unwrap(),
                    vec![222.0; 65 * 3]
                );
                assert_eq!(
                    parent.into_data().convert::<f32>().to_vec::<f32>().unwrap(),
                    vec![2.0; 41 * 71]
                );
            }
        },
    );
}

#[test]
fn packed_changed_input_trace_and_deferred_frees() {
    use tt_kernels::matrix_eltwise::MatrixEltwiseOp as Op;
    for tiles in [1, 2] {
        with_session(tiles, |s| {
            let a = s.upload_bf16(&vec![0x4000; 37 * 65], 37, 65).unwrap();
            let b = s.upload_bf16(&[0x4040; 65], 1, 65).unwrap();
            s.begin_trace().unwrap();
            let out = s
                .matrix_eltwise_bf16(Op::Mul, &a, &b, Fidelity::HiFi4)
                .unwrap();
            let trace = s.end_trace().unwrap();
            s.free_bf16(b).unwrap();
            for value in [2.0f32, 5.0, -3.0] {
                let temp = s
                    .upload_bf16(&vec![(value.to_bits() >> 16) as u16; 37 * 65], 37, 65)
                    .unwrap();
                s.copy_into_bf16(&temp, &a).unwrap();
                s.free_bf16(temp).unwrap();
                s.replay(trace).unwrap();
                assert_eq!(s.download(&out).unwrap(), vec![value * 3.0; 37 * 65]);
            }
            s.release_trace(trace).unwrap();
            s.free(out).unwrap();
            s.free_bf16(a).unwrap();
        });
    }
}

#[test]
fn poisoned_ragged_lanes_never_modify_parent_storage_or_claims() {
    use tt_kernels::{
        matrix_eltwise::{MatrixEltwiseOp as Op, SrcPrecision},
        tensor::Pad,
    };
    with_session(2, |s| {
        let (rows, cols) = (37, 65);
        for dims in [[rows, cols], [1, cols], [rows, 1], [1, 1]] {
            let a = s.upload(&vec![2.0; rows * cols], rows, cols).unwrap();
            let b = s
                .upload(&vec![3.0; dims[0] * dims[1]], dims[0], dims[1])
                .unwrap();
            let pa = s
                .upload_bf16(&vec![0x4000; rows * cols], rows, cols)
                .unwrap();
            let pb = s
                .upload_bf16(&vec![0x4040; dims[0] * dims[1]], dims[0], dims[1])
                .unwrap();
            a.set_pad(Pad::Undefined);
            b.set_pad(Pad::Undefined);
            s.sync().unwrap();
            let w = s
                .device()
                .alloc_window(tt_device::tlb::WindowKind::FourGib)
                .unwrap();
            let mut snapshots = Vec::new();
            let slots = [
                (
                    (0..a.placement.tiles())
                        .map(|t| a.placement.slot(t))
                        .collect::<Vec<_>>(),
                    false,
                    [rows, cols],
                ),
                (
                    (0..b.placement.tiles())
                        .map(|t| b.placement.slot(t))
                        .collect(),
                    false,
                    dims,
                ),
                (
                    (0..pa.tile_count()).map(|t| pa.slot(t)).collect(),
                    true,
                    [rows, cols],
                ),
                (
                    (0..pb.tile_count()).map(|t| pb.slot(t)).collect(),
                    true,
                    dims,
                ),
            ];
            for (slots, packed, [source_rows, source_cols]) in slots {
                for (tile, slot) in slots.into_iter().enumerate() {
                    let mut bytes = vec![0; slot.len() as usize];
                    s.device().dram_read(&w, slot, &mut bytes).unwrap();
                    for r in 0..32 {
                        for c in 0..32 {
                            if tile / source_cols.div_ceil(32) * 32 + r >= source_rows
                                || tile % source_cols.div_ceil(32) * 32 + c >= source_cols
                            {
                                let at =
                                    16 + tt_isa::dm::face_index(r, c) * if packed { 2 } else { 4 };
                                if packed {
                                    bytes[at..at + 2].copy_from_slice(&0x7fc1u16.to_le_bytes());
                                } else {
                                    bytes[at..at + 4].copy_from_slice(&0x7fc12345u32.to_le_bytes());
                                }
                            }
                        }
                    }
                    s.device().dram_write(&w, slot, &bytes).unwrap();
                    snapshots.push((slot, bytes));
                }
            }
            for op in [Op::Add, Op::Sub, Op::Mul] {
                for packed in [false, true] {
                    let out = if packed {
                        s.matrix_eltwise_bf16(op, &pa, &pb, Fidelity::HiFi4)
                    } else {
                        s.matrix_eltwise(op, &a, &b, SrcPrecision::Tf32, Fidelity::HiFi4)
                    }
                    .unwrap();
                    assert_eq!(out.pad(), Pad::Undefined);
                    let sum = s.sum_rows(&out).unwrap();
                    let value = match op {
                        Op::Add => 5.0,
                        Op::Sub => -1.0,
                        Op::Mul => 6.0,
                    };
                    assert_eq!(s.download(&sum).unwrap(), vec![value * rows as f32; cols]);
                    s.free(sum).unwrap();
                    s.free(out).unwrap();
                }
            }
            assert_eq!((a.pad(), b.pad()), (Pad::Undefined, Pad::Undefined));
            for (slot, want) in snapshots {
                let mut bytes = vec![0; slot.len() as usize];
                s.device().dram_read(&w, slot, &mut bytes).unwrap();
                assert_eq!(bytes, want, "arithmetic/repair modified operand padding");
            }
            let bad = s.upload(&vec![1.0; 2 * cols], 2, cols).unwrap();
            assert!(s
                .matrix_eltwise(Op::Add, &a, &bad, SrcPrecision::Tf32, Fidelity::HiFi4)
                .is_err());
            s.free(bad).unwrap();
            s.free(a).unwrap();
            s.free(b).unwrap();
            s.free_bf16(pa).unwrap();
            s.free_bf16(pb).unwrap();
        }
    });
}

/// Direct RHS addressing must see replay-time data, including nonconstant
/// row/column values across both face boundaries and physical tile boundaries.
#[test]
fn broadcast_traces_read_changed_rhs_without_materialization() {
    use tt_kernels::matrix_eltwise::{MatrixEltwiseOp as Op, SrcPrecision};
    for tiles in [1, 2] {
        with_session(tiles, |s| {
            let (rows, cols) = (37, 65);
            let a = s.upload(&vec![32.0; rows * cols], rows, cols).unwrap();
            let pa = s
                .upload_bf16(&vec![0x4200; rows * cols], rows, cols)
                .unwrap();
            for dims in [[1, cols], [rows, 1], [1, 1]] {
                let b = s
                    .upload(&vec![1.0; dims[0] * dims[1]], dims[0], dims[1])
                    .unwrap();
                let pb = s
                    .upload_bf16(&vec![0x3f80; dims[0] * dims[1]], dims[0], dims[1])
                    .unwrap();
                let padding = b.pad();
                for packed in [false, true] {
                    s.begin_trace().unwrap();
                    let out = if packed {
                        s.matrix_eltwise_bf16(Op::Sub, &pa, &pb, Fidelity::HiFi4)
                    } else {
                        s.matrix_eltwise(Op::Sub, &a, &b, SrcPrecision::Tf32, Fidelity::HiFi4)
                    }
                    .unwrap();
                    let trace = s.end_trace().unwrap();
                    for pass in [0, 1, 2] {
                        let values: Vec<_> = (0..dims[0] * dims[1])
                            .map(|i| ((i * 3 + pass * 5) % 23 + 1) as f32)
                            .collect();
                        if packed {
                            let temp = s
                                .upload_bf16(
                                    &values
                                        .iter()
                                        .map(|v| (v.to_bits() >> 16) as u16)
                                        .collect::<Vec<_>>(),
                                    dims[0],
                                    dims[1],
                                )
                                .unwrap();
                            s.copy_into_bf16(&temp, &pb).unwrap();
                            s.free_bf16(temp).unwrap();
                        } else {
                            s.write(&b, &values).unwrap();
                        }
                        s.replay(trace).unwrap();
                        let want: Vec<_> = (0..rows)
                            .flat_map(|r| {
                                let values = &values;
                                (0..cols).map(move |c| {
                                    32.0 - values[(r % dims[0]) * dims[1] + c % dims[1]]
                                })
                            })
                            .collect();
                        assert_eq!(
                            s.download(&out).unwrap(),
                            want,
                            "RHS {dims:?}, packed={packed}, tiles={tiles}, pass={pass}"
                        );
                        assert_eq!(b.pad(), padding);
                    }
                    s.release_trace(trace).unwrap();
                    s.free(out).unwrap();
                }
                s.free(b).unwrap();
                s.free_bf16(pb).unwrap();
            }
            s.free(a).unwrap();
            s.free_bf16(pa).unwrap();
        });
    }
}
