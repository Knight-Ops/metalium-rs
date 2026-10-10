//! Mesh trace capture and replay (`hardware-coverage.md` R4, lane T9).
//!
//! A two-chip [`burn_tt::MeshEngine`] captures what a distributed product or
//! attention forward enqueues on every chip, with the Ethernet transfers
//! between them recorded as steps of their own (`tt_kernels::mesh_trace`), and
//! replays it on changed inputs. Every replay must equal a fresh run of the same
//! Burn ops bit for bit and the independent oracle (the host's `Flex` product,
//! or an analytic attention result), with no tensor uploaded and only the
//! outputs downloaded, and with real device work shown by the fabric's
//! completion evidence (peer matmuls on both chips, acknowledged Ethernet
//! bytes).
//!
//! Simulator and silicon run the same Burn gates (`with_mesh_device`: `bh_x2`,
//! or cards 0 and 1 with `--features silicon`); the Fabric-level refusal gates
//! need a simulator-built fabric and are simulator-only.

use burn::tensor::{module::attention, Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tensor::ops::AttentionModuleOptions;
use burn_tt::{tensor_traffic, InputPayload, OutputPayload, TracedInference, TtBackend, TtTensor};
use tt_tests::burn_device::{assert_native_model, with_mesh_device, Config};

type T2 = Tensor<TtBackend, 2>;
type T4 = Tensor<TtBackend, 4>;

fn float2(t: T2) -> TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!("a float tensor"),
    }
}

fn float4(t: T4) -> TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!("a float tensor"),
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn f32_output(p: &OutputPayload) -> Vec<u32> {
    bits(p.as_f32().expect("an F32 output"))
}

/// Small integers: every product and sum below is exact in TF32 operands and
/// FP32 accumulation, so the host's `Flex` product is an exact oracle.
fn integers(round: u32, len: usize, modulus: u32, offset: i32) -> Vec<f32> {
    let mut s = round.wrapping_mul(2_654_435_761) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) % modulus) as i32 as f32 + offset as f32
        })
        .collect()
}

/// Fractional values, so TF32 truncation and accumulation order matter: only
/// a fresh run on the same engine can be the reference.
fn fractions(round: u32, len: usize) -> Vec<f32> {
    let mut s = round.wrapping_mul(2_246_822_519) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

#[test]
fn product_trace_replays_changed_inputs_equal_to_fresh_runs() {
    with_mesh_device(Config::default(), 2, |d| {
        // Ragged in every dimension, and a product of a ragged product: the
        // second one's operand has 35 live rows of 225 columns.
        let (m, k, n, p) = (35, 37, 225, 70);
        let wv = integers(7, k * n, 7, -3);
        let w2v = integers(8, n * p, 3, -1);
        let w: T2 = Tensor::from_data(TensorData::new(wv.clone(), [k, n]), &d);
        let w2: T2 = Tensor::from_data(TensorData::new(w2v.clone(), [n, p]), &d);
        let x: T2 = Tensor::from_data(TensorData::new(integers(0, m * k, 5, -2), [m, k]), &d);
        let xp = float2(x.clone());
        // Resident before the capture: their own uploads are not the capture's.
        xp.ensure_resident();
        float2(w.clone()).ensure_resident();
        float2(w2.clone()).ensure_resident();
        let flex = |xv: &[f32]| -> Vec<u32> {
            let x = Tensor::<Flex, 2>::from_data(TensorData::new(xv.to_vec(), [m, k]), &FlexDevice);
            let w = Tensor::<Flex, 2>::from_data(TensorData::new(wv.clone(), [k, n]), &FlexDevice);
            let w2 =
                Tensor::<Flex, 2>::from_data(TensorData::new(w2v.clone(), [n, p]), &FlexDevice);
            bits(&x.matmul(w).matmul(w2).into_data().to_vec::<f32>().unwrap())
        };
        let fresh = |xv: Vec<f32>| -> Vec<u32> {
            let x: T2 = Tensor::from_data(TensorData::new(xv, [m, k]), &d);
            bits(
                &x.matmul(w.clone())
                    .matmul(w2.clone())
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            )
        };

        let before = tensor_traffic();
        let execution = burn_tt::mesh_execution(d).unwrap();
        let ((trace, first), report) = burn_tt::with_report(|| {
            TracedInference::capture(&[&xp], || {
                vec![float2(x.clone().matmul(w.clone()).matmul(w2.clone()))]
            })
            .unwrap()
        });
        assert_native_model(&report);
        let after = tensor_traffic();
        assert_eq!(after.uploads, before.uploads, "the capture uploaded");
        // Its first outputs come back through the trace's own readback, which
        // the Burn tensor counters do not see: no tensor was downloaded.
        assert_eq!(after.downloads, before.downloads, "the capture downloaded");
        assert_eq!(trace.output_shapes(), [[m, p]]);
        assert_eq!(f32_output(&first[0]), flex(&integers(0, m * k, 5, -2)));
        // The capture ran both products on both chips.
        let captured = burn_tt::mesh_execution(d).unwrap();
        for chip in 0..2 {
            assert!(captured.completed_matmuls[chip] >= execution.completed_matmuls[chip] + 2);
        }

        let mut last = f32_output(&first[0]);
        for round in 1..=4u32 {
            let xv = integers(round, m * k, 5, -2);
            let up = tensor_traffic();
            let at = burn_tt::mesh_execution(d).unwrap();
            let out = trace.run(vec![InputPayload::F32(xv.clone())]).unwrap();
            let now = tensor_traffic();
            assert_eq!(now.uploads, up.uploads, "replay {round} uploaded");
            assert_eq!(now.downloads, up.downloads, "replay {round} downloaded");
            // The device did the work: both products on both chips, with
            // data crossing Ethernet.
            let done = burn_tt::mesh_execution(d).unwrap();
            for chip in 0..2 {
                assert_eq!(
                    done.completed_matmuls[chip],
                    at.completed_matmuls[chip] + 2,
                    "replay {round}: chip {chip}'s matmuls"
                );
            }
            assert!(done.acknowledged_ethernet_bytes > at.acknowledged_ethernet_bytes);
            let got = f32_output(&out[0]);
            assert_eq!(got, flex(&xv), "replay {round} against the host oracle");
            assert_ne!(got, last, "replay {round} repeated its predecessor");
            last = got.clone();
            // And what a fresh run of the same ops gives. The fresh run also
            // allocates and frees on both chips between replays, which
            // reuses any slot a replay's capture did not keep held.
            assert_eq!(fresh(xv), got, "replay {round} against a fresh run");
        }

        // Fractional data: only the fresh run is the reference.
        let xv = fractions(11, m * k);
        let out = trace.run(vec![InputPayload::F32(xv.clone())]).unwrap();
        assert_eq!(f32_output(&out[0]), fresh(xv), "fractional replay");

        // The trace is released and a replay of fresh work still agrees.
        drop(trace);
        let xv = integers(99, m * k, 5, -2);
        assert_eq!(fresh(xv.clone()), flex(&xv), "after the release");
    });
}

fn attention_options() -> AttentionModuleOptions {
    AttentionModuleOptions {
        scale: Some(1.0),
        softcap: None,
        is_causal: false,
    }
}

#[test]
fn attention_trace_replays_changed_values_equal_to_fresh_runs() {
    with_mesh_device(Config::default(), 2, |d| {
        // Zero logits make every probability exactly 1/64, so the result is
        // the mean of the value rows: an exact analytic oracle for integer V.
        let (heads, sq, keys, dim) = (2usize, 3usize, 64usize, 64usize);
        let q = T4::zeros([1, heads, sq, 32], &d);
        let k = T4::zeros([1, heads, keys, 32], &d);
        let vv = |round: u32| integers(round, heads * keys * dim, 9, -4);
        let v: T4 = Tensor::from_data(TensorData::new(vv(0), [1, heads, keys, dim]), &d);
        let vp = float4(v.clone());
        vp.ensure_resident();
        // The zero operands are lazily uploaded too.
        float4(q.clone()).ensure_resident();
        float4(k.clone()).ensure_resident();
        let analytic = |values: &[f32]| -> Vec<u32> {
            let mut out = Vec::new();
            for h in 0..heads {
                for _ in 0..sq {
                    for c in 0..dim {
                        let sum: f32 = (0..keys).map(|j| values[(h * keys + j) * dim + c]).sum();
                        out.push(sum / keys as f32);
                    }
                }
            }
            bits(&out)
        };
        let fresh = |values: Vec<f32>| -> Vec<u32> {
            let v: T4 = Tensor::from_data(TensorData::new(values, [1, heads, keys, dim]), &d);
            bits(
                &attention(q.clone(), k.clone(), v, None, None, attention_options())
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            )
        };

        let before = tensor_traffic();
        let ((trace, first), report) = burn_tt::with_report(|| {
            TracedInference::capture(&[&vp], || {
                vec![float4(attention(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    None,
                    None,
                    attention_options(),
                ))]
            })
            .unwrap()
        });
        assert_native_model(&report);
        let after = tensor_traffic();
        assert_eq!(after.uploads, before.uploads, "the capture uploaded");
        assert_eq!(after.downloads, before.downloads, "the capture downloaded");
        assert_eq!(f32_output(&first[0]), analytic(&vv(0)), "capture run");

        let mut last = f32_output(&first[0]);
        for round in 1..=3u32 {
            let values = vv(round);
            let up = tensor_traffic();
            let at = burn_tt::mesh_execution(d).unwrap();
            let out = trace.run(vec![InputPayload::F32(values.clone())]).unwrap();
            let now = tensor_traffic();
            assert_eq!(now.uploads, up.uploads, "replay {round} uploaded");
            assert_eq!(now.downloads, up.downloads, "replay {round} downloaded");
            let done = burn_tt::mesh_execution(d).unwrap();
            for chip in 0..2 {
                assert!(
                    done.completed_matmuls[chip] > at.completed_matmuls[chip],
                    "replay {round}: chip {chip} ran no matmul"
                );
            }
            assert!(done.acknowledged_ethernet_bytes > at.acknowledged_ethernet_bytes);
            let got = f32_output(&out[0]);
            assert_eq!(got, analytic(&values), "replay {round} against the oracle");
            assert_ne!(got, last, "replay {round} repeated its predecessor");
            last = got.clone();
            assert_eq!(fresh(values), got, "replay {round} against a fresh run");
        }
    });
}

#[test]
fn a_second_capture_is_refused_and_leaves_the_first_usable() {
    with_mesh_device(Config::default(), 2, |d| {
        let (m, k, n) = (3, 5, 33);
        let wv = integers(3, k * n, 5, -2);
        let w: T2 = Tensor::from_data(TensorData::new(wv.clone(), [k, n]), &d);
        let x: T2 = Tensor::from_data(TensorData::new(integers(1, m * k, 5, -2), [m, k]), &d);
        let xp = float2(x.clone());
        xp.ensure_resident();
        let mut nested = None;
        let (trace, _) = TracedInference::capture(&[&xp], || {
            nested = Some(TracedInference::capture(&[&xp], Vec::new).err());
            vec![float2(x.clone().matmul(w.clone()))]
        })
        .unwrap();
        let message = nested.unwrap().expect("the nested capture was refused");
        assert!(
            message.to_string().contains("already open"),
            "refusal: {message}"
        );
        let xv = integers(2, m * k, 5, -2);
        let out = trace.run(vec![InputPayload::F32(xv.clone())]).unwrap();
        let want = Tensor::<Flex, 2>::from_data(TensorData::new(xv, [m, k]), &FlexDevice)
            .matmul(Tensor::from_data(TensorData::new(wv, [k, n]), &FlexDevice))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(f32_output(&out[0]), bits(&want));
    });
}

#[cfg(not(feature = "silicon"))]
mod fabric {
    //! The Fabric-level contracts, on a simulator-built two-chip fabric.
    use super::*;
    use tt_device::Device;
    use tt_kernels::mesh_trace::MeshTraceError;
    use tt_kernels::shard::{Chip, Fabric};
    use tt_kernels::tensor::TensorError;
    use tt_kernels::trace::TraceError;

    fn with_fabric(f: impl FnOnce(&mut Fabric<tt_ttsim::LibTtsim<'_>>)) {
        tt_ttsim::fork_scope(|| {
            let mut sim = tt_ttsim::Simulator::open_path(tt_ttsim::x2_lib_path()).unwrap();
            let chips = sim
                .transports()
                .into_iter()
                .map(|t| {
                    Chip::new(
                        Device::open(t).unwrap(),
                        tt_tests::harness::tensix_tile(),
                        tt_tests::harness::relay_tile(),
                    )
                    .unwrap()
                })
                .collect();
            let mut fabric = Fabric::new(
                chips,
                &tt_tests::topology::links(&tt_tests::topology::BH_X2),
                tt_firmware_images::ROLES,
                tt_firmware_images::ETH_E1,
            )
            .unwrap();
            fabric
                .enable_resident(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
            f(&mut fabric);
        })
        .unwrap();
    }

    #[test]
    fn a_transfer_whose_slots_no_trace_holds_is_refused_by_name() {
        with_fabric(|fabric| {
            let t = fabric.chips[0]
                .session()
                .upload(&integers(1, 40 * 33, 9, -4), 40, 33)
                .unwrap();
            fabric.begin_mesh_trace().unwrap();
            // Chip 0 computes, so its capture is stored; chip 1 only
            // receives, and runs nothing after: nothing holds the slots the
            // replay's transfer would write.
            let doubled = fabric.chips[0]
                .session()
                .eltwise(
                    tt_kernels::tensor::Eltwise {
                        kind: tt_kernels::kind::ADD,
                        scalar: 0.0,
                        scalar2: 0.0,
                    },
                    &t,
                    Some(&t),
                )
                .unwrap();
            let moved = fabric.transfer_tensor(0, &doubled, 1).unwrap();
            match fabric.end_mesh_trace() {
                Err(MeshTraceError::UnheldTransfer {
                    transfer: 0,
                    chip: 1,
                    end: "destination",
                }) => {}
                other => panic!("expected the unheld-transfer refusal, got {other:?}"),
            }
            assert!(!fabric.mesh_capturing(), "a refused capture stays open");
            fabric.chips[1].session().free(moved).unwrap();
        });
    }

    #[test]
    fn host_transfers_nested_captures_and_unknown_traces_are_refused() {
        with_fabric(|fabric| {
            let t = fabric.chips[0]
                .session()
                .upload(&integers(1, 33 * 33, 9, -4), 33, 33)
                .unwrap();
            fabric.begin_mesh_trace().unwrap();
            assert!(matches!(
                fabric.begin_mesh_trace(),
                Err(MeshTraceError::Capturing)
            ));
            // A download on any chip is a host transfer a replay could not
            // repeat.
            assert!(matches!(
                fabric.chips[0].session().download(&t),
                Err(TensorError::Trace(TraceError::HostTransfer("download")))
            ));
            // With nothing run, there is no trace.
            assert!(matches!(
                fabric.end_mesh_trace(),
                Err(MeshTraceError::Empty)
            ));
            assert!(matches!(
                fabric.end_mesh_trace(),
                Err(MeshTraceError::NotCapturing)
            ));
            assert!(matches!(
                fabric.replay_mesh_trace(9),
                Err(MeshTraceError::Unknown(9))
            ));
            assert!(matches!(
                fabric.release_mesh_trace(9),
                Err(MeshTraceError::Unknown(9))
            ));
        });
    }
}
