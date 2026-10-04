//! K blocking continues the same Matrix Unit product sequence.
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;
#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 73 + seed) % 257) as f32 / 131.0 - 1.0)
        .collect()
}
fn same(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} vs {b}");
    }
}
#[test]
fn forced_k_blocks_match_unsplit_for_ragged_products_and_fidelities() {
    with_session(|s| {
        s.set_pipeline(false);
        for fidelity in [
            Fidelity::Lo,
            Fidelity::HiFi2,
            Fidelity::HiFi3,
            Fidelity::HiFi4,
        ] {
            let (m, k, n) = (37, 97, 35);
            let a = s.upload(&values(m * k, 1), m, k).unwrap();
            let b = s.upload(&values(k * n, 11), k, n).unwrap();
            let baseline = s
                .matmul_dram(
                    &a,
                    false,
                    &b,
                    false,
                    SrcRoute::Tf32FromFp32,
                    fidelity,
                    BUDGET * 20,
                )
                .unwrap();
            let want = s.download(&baseline).unwrap();
            for block in [1, 2, 3] {
                s.set_matmul_k_block_limit(std::num::NonZeroUsize::new(block));
                let out = s
                    .matmul_dram(
                        &a,
                        false,
                        &b,
                        false,
                        SrcRoute::Tf32FromFp32,
                        fidelity,
                        BUDGET * 20,
                    )
                    .unwrap();
                same(&s.download(&out).unwrap(), &want);
                s.free(out).unwrap();
            }
            s.set_matmul_k_block_limit(None);
            s.free(baseline).unwrap();
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
    });
}

#[test]
fn large_k_and_batched_products_are_native() {
    with_session(|s| {
        use tt_kernels::tensor::Block;
        // Every product and addition is exact: a sparse identity-like B
        // independently predicts outputs without sharing the kernel builder.
        let (m, k, n) = (64, 8192, 64);
        let av: Vec<_> = (0..m * k).map(|i| (i % 13) as f32 - 6.).collect();
        let bv: Vec<_> = (0..k * n)
            .map(|i| if i / n == i % n { 1. } else { 0. })
            .collect();
        let a = s.upload(&av, m, k).unwrap();
        let b = s.upload(&bv, k, n).unwrap();
        let out = s
            .matmul_dram(
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                BUDGET * 40,
            )
            .unwrap();
        let want: Vec<_> = (0..m)
            .flat_map(|r| av[r * k..r * k + n].iter().copied())
            .collect();
        same(&s.download(&out).unwrap(), &want);
        s.free(out).unwrap();
        let items = [(
            Block {
                at: [0, 0],
                transposed: false,
            },
            Block {
                at: [0, 0],
                transposed: false,
            },
        ); 2];
        let out = s
            .matmul_dram_batched(
                &a,
                &b,
                &items,
                [m, k, n],
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                BUDGET * 80,
            )
            .unwrap();
        same(&s.download(&out).unwrap(), &[want.clone(), want].concat());
        s.free(out).unwrap();
        s.free(a).unwrap();
        s.free(b).unwrap();
    });
}

#[test]
fn transposed_k_blocks_and_special_values_match_unsplit() {
    with_session(|s| {
        for route in [SrcRoute::Tf32FromFp32, SrcRoute::Bf16FromFp32] {
            for flags in [[true, false], [false, true], [true, true]] {
                let (m, k, n) = (5, 65, 7);
                let [ar, ac] = if flags[0] { [k, m] } else { [m, k] };
                let [br, bc] = if flags[1] { [n, k] } else { [k, n] };
                let mut av = values(ar * ac, 7);
                let mut bv = values(br * bc, 17);
                for (i, v) in [
                    0.,
                    -0.,
                    f32::from_bits(1),
                    -f32::from_bits(1),
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                    -f32::NAN,
                    f32::MAX,
                    f32::MIN_POSITIVE,
                ]
                .into_iter()
                .enumerate()
                {
                    av[i] = v;
                    bv[i] = v;
                }
                let a = s.upload(&av, ar, ac).unwrap();
                let b = s.upload(&bv, br, bc).unwrap();
                let plain = s
                    .matmul_dram(
                        &a,
                        flags[0],
                        &b,
                        flags[1],
                        route,
                        Fidelity::HiFi4,
                        BUDGET * 20,
                    )
                    .unwrap();
                let want = s.download(&plain).unwrap();
                s.set_matmul_k_block_limit(std::num::NonZeroUsize::new(1));
                let blocked = s
                    .matmul_dram(
                        &a,
                        flags[0],
                        &b,
                        flags[1],
                        route,
                        Fidelity::HiFi4,
                        BUDGET * 20,
                    )
                    .unwrap();
                same(&s.download(&blocked).unwrap(), &want);
                s.set_matmul_k_block_limit(None);
                for out in [a, b, plain, blocked] {
                    s.free(out).unwrap();
                }
            }
        }
    });
}

#[test]
fn continuation_traces_replay_changed_inputs_and_hold_freed_operands() {
    with_session(|s| {
        let (m, k, n) = (37, 97, 35);
        let a = s.upload(&values(m * k, 1), m, k).unwrap();
        let b = s.upload(&values(k * n, 3), k, n).unwrap();
        s.set_matmul_k_block_limit(std::num::NonZeroUsize::new(1));
        s.begin_trace().unwrap();
        let out = s
            .matmul_dram(
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                BUDGET * 20,
            )
            .unwrap();
        let trace = s.end_trace().unwrap();
        for seed in [7, 13, 19] {
            s.write(&a, &values(m * k, seed)).unwrap();
            s.replay(trace).unwrap();
            s.sync().unwrap();
            let got = s.download(&out).unwrap();
            s.set_matmul_k_block_limit(None);
            let fresh = s
                .matmul_dram(
                    &a,
                    false,
                    &b,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    BUDGET * 20,
                )
                .unwrap();
            same(&got, &s.download(&fresh).unwrap());
            s.free(fresh).unwrap();
        }
        s.free(a).unwrap();
        s.free(b).unwrap();
        s.replay(trace).unwrap();
        s.sync().unwrap();
        s.release_trace(trace).unwrap();
        s.free(out).unwrap();
    });
}

#[test]
fn burn_large_k_stays_resident_within_the_derived_bound() {
    use burn::tensor::{Tensor, TensorData, TensorPrimitive};
    use burn_flex::{Flex, FlexDevice};
    use burn_tt::{tensor_traffic, TtBackend};
    use tt_tests::burn_device::{assert_native_model, with_device, Config};
    with_device(
        Config {
            tiles: Some(TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            let (m, k, n) = (3, 8193, 5);
            let av = values(m * k, 37);
            let bv = values(k * n, 59);
            let a = Tensor::<TtBackend, 2>::from_data(TensorData::new(av.clone(), [m, k]), &d);
            let b = Tensor::<TtBackend, 2>::from_data(TensorData::new(bv.clone(), [k, n]), &d);
            let before = tensor_traffic();
            let (out, report) = burn_tt::with_report(|| a.matmul(b));
            let TensorPrimitive::Float(p) = out.clone().into_primitive() else {
                unreachable!()
            };
            assert!(p.computed_on_device());
            assert_eq!(tensor_traffic().downloads, before.downloads);
            assert_native_model(&report);
            let got = out.into_data().to_vec::<f32>().unwrap();
            let want =
                Tensor::<Flex, 2>::from_data(TensorData::new(av.clone(), [m, k]), &FlexDevice)
                    .matmul(Tensor::from_data(
                        TensorData::new(bv.clone(), [k, n]),
                        &FlexDevice,
                    ))
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
            // TF32 input truncation contributes <2^-9 sum|ab|. Charge each
            // device phase addition 2^-23 and each Flex addition 2^-24;
            // gamma covers repeated rounding, including cancellation. Reloads
            // add no arithmetic. These normal finite inputs cannot overflow.
            let device_gamma =
                (4 * k) as f64 * 2f64.powi(-23) / (1. - (4 * k) as f64 * 2f64.powi(-23));
            let flex_gamma = k as f64 * 2f64.powi(-24) / (1. - k as f64 * 2f64.powi(-24));
            for index in 0..m * n {
                let abs: f64 = (0..k)
                    .map(|q| (av[index / n * k + q] as f64 * bv[q * n + index % n] as f64).abs())
                    .sum();
                let bound =
                    (2f64.powi(-9) + (1. + 2f64.powi(-9)) * device_gamma + flex_gamma) * abs;
                assert!((got[index] as f64 - want[index] as f64).abs() <= bound);
            }
        },
    );
}

#[test]
fn underflow_keeps_the_unsplit_matrix_unit_behavior() {
    with_session(|s| {
        let k = 128;
        let a = s.upload(&vec![2f32.powi(-70); k], 1, k).unwrap();
        let b = s.upload(&vec![2f32.powi(-62); k], k, 1).unwrap();
        let plain = s
            .matmul_dram(
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                BUDGET * 20,
            )
            .unwrap();
        let want = s.download(&plain).unwrap();
        // In exact arithmetic the result would be normal 2^-125. ttsim
        // drops these subnormal products before they can accumulate. This
        // checks that blocking preserves that behavior; it does not claim
        // the Matrix Unit accumulates subnormals or establish silicon's
        // underflow policy. The silicon gate compares with its unsplit run.
        #[cfg(not(feature = "silicon"))]
        assert_eq!(want[0].to_bits(), 0);
        s.set_matmul_k_block_limit(std::num::NonZeroUsize::new(1));
        let blocked = s
            .matmul_dram(
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                BUDGET * 20,
            )
            .unwrap();
        same(&s.download(&blocked).unwrap(), &want);
        for out in [a, b, plain, blocked] {
            s.free(out).unwrap();
        }
    });
}
