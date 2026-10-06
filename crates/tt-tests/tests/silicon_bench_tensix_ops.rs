//! Validated release Burn operation baselines, including final result readback.
#![cfg(feature = "silicon")]
use burn::tensor::{
    module,
    ops::{AttentionModuleOptions, ConvOptions},
    Int, Tensor, TensorData,
};
use burn_tt::TtBackend;
use std::time::Instant;
use tt_tests::{
    bench::{report, Stats},
    burn_device::{with_device, Config},
};

fn measure(key: &str, mut run: impl FnMut() -> TensorData, expected: &[f32]) {
    let mut samples = Vec::new();
    for repetition in 0..9 {
        let start = Instant::now();
        let output = run();
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        assert_eq!(output.convert::<f32>().to_vec::<f32>().unwrap(), expected);
        if repetition >= 2 {
            samples.push(elapsed);
        }
    }
    report(key, "us", "host", Stats::of(samples));
}

#[test]
#[ignore = "benchmark"]
fn resident_operation_baselines() {
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            println!("BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"tiles\":2,\"warmups\":2,\"samples\":7,\"inputs\":\"resident\",\"timing\":\"host dispatch through final output readback; validation outside timing\",\"dtype\":\"F32 unless key says I32\"}}", tt_tests::bench::git_sha(), !cfg!(debug_assertions));
            let ints = Tensor::<TtBackend, 2, Int>::from_data(
                TensorData::new(vec![12345; 37 * 70], [37, 70]),
                &d,
            )
            .add_scalar(0);
            measure(
                "I32 division [37,70] / 3",
                || ints.clone().div_scalar(3).into_data(),
                &vec![4115.0; 37 * 70],
            );
            let x = Tensor::<TtBackend, 4>::ones([1, 1, 9, 10], &d).to_device(&d);
            measure(
                "average pool [1,1,9,10] kernel4x8",
                || module::avg_pool2d(x.clone(), [4, 8], [1, 1], [0, 0], false, false).into_data(),
                &[1.0; 18],
            );
            let w = Tensor::<TtBackend, 4>::ones([2, 1, 2, 2], &d).to_device(&d);
            measure(
                "conv2d [1,1,9,10] weight[2,1,2,2]",
                || {
                    module::conv2d(
                        x.clone(),
                        w.clone(),
                        None,
                        ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
                    )
                    .into_data()
                },
                &[4.0; 144],
            );
            let q = Tensor::<TtBackend, 4>::zeros([1, 1, 3, 32], &d).to_device(&d);
            let k = Tensor::<TtBackend, 4>::zeros([1, 1, 64, 32], &d).to_device(&d);
            let v = Tensor::<TtBackend, 4>::ones([1, 1, 64, 32], &d).to_device(&d);
            measure(
                "attention Q[1,1,3,32] KVseq64",
                || {
                    module::attention(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        None,
                        None,
                        AttentionModuleOptions {
                            scale: None,
                            softcap: None,
                            is_causal: false,
                        },
                    )
                    .into_data()
                },
                &[1.0; 96],
            );
        },
    );
}

#[test]
#[ignore = "benchmark"]
fn bf16_and_mesh_operation_baselines() {
    use burn::tensor::FloatDType;
    let run = |d: burn_tt::TtDevice, mesh: bool| {
        println!("BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"cards\":{},\"warmups\":2,\"samples\":7,\"timing\":\"host dispatch through final readback; validation outside timing\"}}",tt_tests::bench::git_sha(),!cfg!(debug_assertions),if mesh {2} else {1});
        for dtype in [FloatDType::F32, FloatDType::BF16] {
            if !mesh && dtype == FloatDType::F32 {
                continue;
            }
            let x = Tensor::<TtBackend, 4>::ones([1, 64, 2, 3], &d)
                .cast(dtype)
                .to_device(&d);
            let w = Tensor::<TtBackend, 4>::ones([64, 64, 1, 1], &d)
                .cast(dtype)
                .to_device(&d);
            let execution = burn_tt::mesh_execution(d);
            measure(
                &format!("{dtype:?} mesh={mesh} conv2d x[1,64,2,3] w[64,64,1,1]"),
                || {
                    module::conv2d(
                        x.clone(),
                        w.clone(),
                        None,
                        ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
                    )
                    .into_data()
                },
                &[64.0; 384],
            );
            let q = Tensor::<TtBackend, 4>::zeros([1, 1, 3, 64], &d)
                .cast(dtype)
                .to_device(&d);
            let k = Tensor::<TtBackend, 4>::zeros([1, 1, 64, 64], &d)
                .cast(dtype)
                .to_device(&d);
            let v = Tensor::<TtBackend, 4>::ones([1, 1, 64, 64], &d)
                .cast(dtype)
                .to_device(&d);
            measure(
                &format!("{dtype:?} mesh={mesh} attention Q[1,1,3,64] KVseq64"),
                || {
                    module::attention(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        None,
                        None,
                        AttentionModuleOptions {
                            scale: None,
                            softcap: None,
                            is_causal: false,
                        },
                    )
                    .into_data()
                },
                &[1.0; 192],
            );
            if let Some(before) = execution {
                let after = burn_tt::mesh_execution(d).unwrap();
                for card in 0..2 {
                    assert!(after.completed_matmuls[card] > before.completed_matmuls[card]);
                }
                assert!(after.acknowledged_ethernet_bytes > before.acknowledged_ethernet_bytes);
            }
        }
    };
    with_device(Config::default(), |d| run(d, false));
    tt_tests::burn_device::with_mesh_device(Config::default(), 2, |d| run(d, true));
}
