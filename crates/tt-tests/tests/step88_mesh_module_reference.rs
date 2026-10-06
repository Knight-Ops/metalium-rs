//! Compare actual two-card module forward/backward with the single-card path.
#![cfg(feature = "silicon")]
use burn::{
    backend::Autodiff,
    tensor::{
        module,
        ops::{AttentionModuleOptions, ConvOptions},
        FloatDType, Tensor, TensorData,
    },
};
use burn_tt::{tensor_traffic, TtBackend, TtDevice};
use tt_tests::burn_device::assert_native_model;

fn run(d: TtDevice, dtype: FloatDType) -> Vec<Vec<f32>> {
    type AD = Autodiff<TtBackend>;
    let x = Tensor::<AD, 4>::from_data(
        TensorData::new(
            (0..384).map(|i| (i % 5) as f32).collect::<Vec<_>>(),
            [1, 64, 2, 3],
        ),
        &d,
    )
    .cast(dtype)
    .require_grad();
    let w = Tensor::<AD, 4>::from_data(
        TensorData::new(
            (0..4096).map(|i| (i % 3) as f32).collect::<Vec<_>>(),
            [64, 64, 1, 1],
        ),
        &d,
    )
    .cast(dtype)
    .require_grad();
    let b = Tensor::<AD, 1>::zeros([64], &d).cast(dtype).require_grad();
    // Q uses columns 0 and 32; K is zero there and varies in every other
    // column. Logits remain exactly zero, while dQ and dK have nonzero
    // columns in both device partitions.
    let q = Tensor::<AD, 4>::from_data(
        TensorData::new(
            (0..192)
                .map(|i| {
                    if i % 64 == 0 || i % 64 == 32 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
            [1, 1, 3, 64],
        ),
        &d,
    )
    .cast(dtype)
    .require_grad();
    let k = Tensor::<AD, 4>::from_data(
        TensorData::new(
            (0..4096)
                .map(|i| {
                    if i % 64 == 0 || i % 64 == 32 {
                        0.0
                    } else {
                        (i / 64 % 4) as f32
                    }
                })
                .collect::<Vec<_>>(),
            [1, 1, 64, 64],
        ),
        &d,
    )
    .cast(dtype)
    .require_grad();
    let v = Tensor::<AD, 4>::from_data(
        TensorData::new(
            (0..4096).map(|i| ((i / 64) % 4) as f32).collect::<Vec<_>>(),
            [1, 1, 64, 64],
        ),
        &d,
    )
    .cast(dtype)
    .require_grad();
    let bias = Tensor::<AD, 4>::zeros([1, 1, 3, 64], &d)
        .cast(dtype)
        .require_grad();
    let before = tensor_traffic();
    let execution = burn_tt::mesh_execution(d);
    let ((cy, dx, dw, db, ay, dq, dk, dv, da), report) = burn_tt::with_report(|| {
        let cy = module::conv2d(
            x.clone(),
            w.clone(),
            Some(b.clone()),
            ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        );
        let cg = cy.clone().cast(FloatDType::F32).sum().backward();
        let ay = module::attention(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            Some(bias.clone()),
            AttentionModuleOptions {
                scale: Some(1.0),
                softcap: None,
                is_causal: false,
            },
        );
        let ag = ay.clone().cast(FloatDType::F32).sum().backward();
        (
            cy.inner().cast(FloatDType::F32),
            x.grad(&cg).unwrap().cast(FloatDType::F32),
            w.grad(&cg).unwrap().cast(FloatDType::F32),
            b.grad(&cg).unwrap().cast(FloatDType::F32),
            ay.inner().cast(FloatDType::F32),
            q.grad(&ag).unwrap().cast(FloatDType::F32),
            k.grad(&ag).unwrap().cast(FloatDType::F32),
            v.grad(&ag).unwrap().cast(FloatDType::F32),
            bias.grad(&ag).unwrap().cast(FloatDType::F32),
        )
    });
    assert_native_model(&report);
    assert_eq!(tensor_traffic().downloads, before.downloads);
    if let Some(before) = execution {
        let after = burn_tt::mesh_execution(d).unwrap();
        for card in 0..2 {
            assert!(after.completed_matmuls[card] >= before.completed_matmuls[card] + 9);
        }
        assert!(after.acknowledged_ethernet_bytes > before.acknowledged_ethernet_bytes);
    }
    let result: Vec<Vec<f32>> = vec![
        cy.into_data().to_vec().unwrap(),
        dx.into_data().to_vec().unwrap(),
        dw.into_data().to_vec().unwrap(),
        db.into_data().to_vec().unwrap(),
        ay.into_data().to_vec().unwrap(),
        dq.into_data().to_vec().unwrap(),
        dk.into_data().to_vec().unwrap(),
        dv.into_data().to_vec().unwrap(),
        da.into_data().to_vec().unwrap(),
    ];
    assert_eq!(result[4], vec![1.5; 192]);
    let dq: Vec<f32> = (0..192)
        .map(|i| {
            if i % 64 == 0 || i % 64 == 32 {
                0.0
            } else {
                80.0
            }
        })
        .collect();
    let dk: Vec<f32> = (0..4096)
        .map(|i| {
            if i % 64 == 0 || i % 64 == 32 {
                3.0 * ((i / 64 % 4) as f32 - 1.5)
            } else {
                0.0
            }
        })
        .collect();
    let db: Vec<f32> = (0..192).map(|i| (i % 64 % 4) as f32 - 1.5).collect();
    assert_eq!(result[5], dq);
    assert_eq!(result[6], dk);
    assert_eq!(result[7], vec![3.0 / 64.0; 4096]);
    assert_eq!(result[8], db);
    result
}

#[test]
fn mesh_convolution_and_attention_gradients_match_single_card() {
    tt_ttsim::fork_scope(|| {
        let device = TtDevice::new(0);
        let config = tt_tests::burn_device::Config::default();
        for dtype in [FloatDType::F32, FloatDType::BF16] {
            let single = {
                let _guard = burn_tt::attach_topology(
                    device,
                    burn_tt::Topology::Single {
                        card: 0,
                        tile: burn_tt::TileChoice::Count(1),
                    },
                    config.route,
                    config.fidelity,
                )
                .unwrap();
                run(device, dtype)
            };
            let mesh = {
                let _guard = burn_tt::attach(
                    device,
                    burn_tt::kmd_mesh_engine(
                        vec![0, 1],
                        tt_tests::backend::GATE_TILE,
                        tt_tests::backend::RELAY_TILE,
                        config.route,
                        config.fidelity,
                    ),
                )
                .unwrap();
                run(device, dtype)
            };
            assert_eq!(
                mesh, single,
                "storage {dtype:?}, forwards and every gradient"
            );
        }
    })
    .unwrap();
}
