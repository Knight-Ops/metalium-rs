//! Sorts and top-k captured in a trace and replayed on changed inputs: every
//! replay is what a fresh run of the same Burn ops (and the independent host
//! oracle) gives, with no tensor uploaded or downloaded by the capture and no
//! host work between replays.
use burn::tensor::{Int, Tensor, TensorData, TensorPrimitive};
use burn_tt::{tensor_traffic, InputPayload, TracedInference, TtBackend, TtTensor};
use tt_kernels::sfpu::sort::reference;
use tt_kernels::tensor::Elem;
use tt_tests::burn_device::{with_device, Config};

type T2 = Tensor<TtBackend, 2>;

fn float(t: T2) -> TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!("a float tensor"),
    }
}

fn int(t: Tensor<TtBackend, 2, Int>) -> TtTensor {
    t.into_primitive()
}

/// Quiet NaNs, both zeros, infinities, subnormals, duplicates: no signalling
/// NaNs, whose payload the host's float copies may quiet.
fn inputs(round: u64, n: usize) -> Vec<f32> {
    let specials = [
        f32::NAN,
        -f32::NAN,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0e-40,
        -1.0e-40,
        3.5,
        3.5,
        -2.0,
    ];
    let mut s = round.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (s >> 33) as usize;
            if r % 4 == 0 {
                specials[(r / 4 + i) % specials.len()]
            } else {
                ((r % 9) as f32 - 4.0) * 0.75
            }
        })
        .collect()
}

fn host(v: &[f32], rows: usize, cols: usize, dim: usize, descending: bool) -> (Vec<u32>, Vec<u32>) {
    let bits: Vec<u32> = v.iter().map(|x| x.to_bits()).collect();
    let (n, lanes) = if dim == 1 { (cols, rows) } else { (rows, cols) };
    let (mut values, mut indices) = (bits.clone(), vec![0; bits.len()]);
    for l in 0..lanes {
        let at = |e: usize| if dim == 1 { l * cols + e } else { e * cols + l };
        let lane: Vec<u32> = (0..n).map(|e| bits[at(e)]).collect();
        let (v, i) = reference(Elem::F32, &lane, descending);
        for e in 0..n {
            values[at(e)] = v[e];
            indices[at(e)] = i[e];
        }
    }
    (values, indices)
}

#[test]
fn sort_and_topk_traces_replay_changed_inputs_without_uploads() {
    with_device(Config::default(), |d| {
        let (rows, cols, k) = (9, 20, 4);
        let x: T2 = Tensor::from_data(TensorData::new(inputs(0, rows * cols), [rows, cols]), &d);
        let xp = float(x.clone());
        // Resident before the capture: its own upload is not the capture's.
        xp.ensure_resident();
        let before = tensor_traffic();
        let (trace, first) = TracedInference::capture(&[&xp], || {
            vec![
                float(x.clone().sort(1)),
                int(x.clone().argsort_descending(1)),
                float(x.clone().topk(k, 1)),
                int(x.clone().argtopk(k, 0)),
                float(x.clone().sort_descending(0)),
            ]
        })
        .unwrap();
        let after = tensor_traffic();
        // The capture itself moved nothing but what it downloaded to report
        // its first outputs (five).
        assert_eq!(after.uploads, before.uploads, "the capture uploaded");
        assert!(first.len() == 5);
        let want = |v: &[f32]| -> Vec<Vec<u32>> {
            let (sorted, _) = host(v, rows, cols, 1, false);
            let (desc_values, desc_indices) = host(v, rows, cols, 1, true);
            let (desc0_values, desc0_indices) = host(v, rows, cols, 0, true);
            vec![
                sorted,
                desc_indices,
                // topk(k, 1): the first k of each row's descending order.
                (0..rows)
                    .flat_map(|r| (0..k).map(move |e| r * cols + e))
                    .map(|at| desc_values[at])
                    .collect(),
                // argtopk(k, 0): the first k rows of the descending columns.
                desc0_indices[..k * cols].to_vec(),
                desc0_values,
            ]
        };
        let outputs = |o: &[burn_tt::OutputPayload]| -> Vec<Vec<u32>> {
            o.iter()
                .map(|p| match p {
                    burn_tt::OutputPayload::F32(v) => v.iter().map(|x| x.to_bits()).collect(),
                    burn_tt::OutputPayload::Bits(b) => b.clone(),
                })
                .collect()
        };
        assert_eq!(
            outputs(&first),
            want(&inputs(0, rows * cols)),
            "capture run"
        );
        let mut last = outputs(&first);
        for round in 1..=4u64 {
            let v = inputs(round, rows * cols);
            let up = tensor_traffic().uploads;
            let out = trace.run(vec![InputPayload::F32(v.clone())]).unwrap();
            assert_eq!(tensor_traffic().uploads, up, "round {round} uploaded");
            let got = outputs(&out);
            assert_eq!(got, want(&v), "replay {round}");
            // The replay computed from the new input: it is not the last one.
            assert_ne!(got, last, "round {round} repeated its predecessor");
            last = got.clone();
            // And it is what fresh Burn ops give.
            let fresh: T2 = Tensor::from_data(TensorData::new(v, [rows, cols]), &d);
            let fresh = fresh.sort(1).into_data().to_vec::<f32>().unwrap();
            assert_eq!(
                fresh.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                got[0],
                "fresh run, round {round}"
            );
        }
    });
}
