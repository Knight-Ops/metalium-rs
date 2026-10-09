//! The sort across tiles: ties broken by original index through the merge
//! network's tiles, many problems, BF16 storage, and multi-pass traces.
use burn::tensor::{DType, Int, Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{InputPayload, OutputPayload, TracedInference, TtBackend};
use tt_kernels::sfpu::sort::reference;
use tt_kernels::tensor::Elem;
use tt_tests::burn_device::{with_device, Config};

fn bits_of(v: Vec<f32>) -> Vec<u32> {
    v.into_iter().map(f32::to_bits).collect()
}

#[test]
fn equal_keys_keep_original_order_across_tiles_in_both_directions() {
    with_device(Config::default(), |d| {
        // 200 positions over seven tiles' worth of merge passes; three
        // distinct values, so almost every comparison is a tie.
        let n = 200;
        let values: Vec<f32> = (0..n)
            .map(|i| [1.5, -0.0, 1.5][(i * 7 + i / 5) % 3])
            .collect();
        let x = Tensor::<TtBackend, 1>::from_data(TensorData::new(values.clone(), [n]), &d);
        for descending in [false, true] {
            let (v, i) = if descending {
                x.clone().sort_descending_with_indices(0)
            } else {
                x.clone().sort_with_indices(0)
            };
            let got_v = bits_of(v.into_data().to_vec::<f32>().unwrap());
            let got_i: Vec<u32> = i
                .into_data()
                .to_vec::<i32>()
                .unwrap()
                .into_iter()
                .map(|x| x as u32)
                .collect();
            let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
            let (want_v, want_i) = reference(Elem::F32, &bits, descending);
            assert_eq!(got_v, want_v, "values, descending {descending}");
            assert_eq!(got_i, want_i, "indices, descending {descending}");
            // The contract itself, stated without the oracle: within a run
            // of equal values the indices ascend, in either direction.
            for w in got_v.windows(2).zip(got_i.windows(2)) {
                if w.0[0] == w.0[1] {
                    assert!(w.1[0] < w.1[1], "tie out of order, descending {descending}");
                }
            }
        }
        // An all-equal axis is the identity permutation, both ways.
        let same = Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(vec![7; 300], [300]), &d);
        for descending in [false, true] {
            let a = if descending {
                same.clone().argsort_descending(0)
            } else {
                same.clone().argsort(0)
            };
            assert_eq!(
                a.into_data().to_vec::<i32>().unwrap(),
                (0..300).collect::<Vec<i32>>()
            );
        }
        // NaNs of both signs and payloads sort to the two ends by sign.
        let nans: Vec<f32> = (0..70)
            .map(|i| match i % 4 {
                0 => f32::NAN,
                1 => -f32::NAN,
                2 => f32::from_bits(0x7fc0_0001),
                _ => f32::from_bits(0xffc0_0002),
            })
            .collect();
        let x = Tensor::<TtBackend, 1>::from_data(TensorData::new(nans.clone(), [70]), &d);
        let flex = Tensor::<Flex, 1>::from_data(TensorData::new(nans, [70]), &FlexDevice);
        assert_eq!(
            bits_of(x.sort(0).into_data().to_vec::<f32>().unwrap()),
            bits_of(flex.sort(0).into_data().to_vec::<f32>().unwrap())
        );
    });
}

#[test]
fn many_problems_over_several_column_blocks_and_axes() {
    with_device(Config::default(), |d| {
        let cases: [(&[usize], usize); 3] = [(&[70, 40], 1), (&[40, 70], 0), (&[3, 75, 5], 1)];
        for (shape, dim) in cases {
            let count: usize = shape.iter().product();
            let values: Vec<f32> = (0..count)
                .map(|i| ((i * 2654435761usize) >> 7) as f32 % 11.0 - 5.0)
                .collect();
            if shape.len() == 2 {
                let x = Tensor::<TtBackend, 2>::from_data(
                    TensorData::new(values.clone(), [shape[0], shape[1]]),
                    &d,
                );
                let flex = Tensor::<Flex, 2>::from_data(
                    TensorData::new(values.clone(), [shape[0], shape[1]]),
                    &FlexDevice,
                );
                let (v, i) = x.sort_descending_with_indices(dim);
                let (fv, _) = flex.sort_descending_with_indices(dim);
                assert_eq!(
                    bits_of(v.into_data().to_vec::<f32>().unwrap()),
                    bits_of(fv.into_data().to_vec::<f32>().unwrap()),
                    "{shape:?} dim {dim}"
                );
                // Indices: each lane's are a permutation, ties ascending.
                let idx = i.into_data().to_vec::<i32>().unwrap();
                let n = shape[dim];
                let lanes = count / n;
                for l in 0..lanes {
                    let at = |e: usize| {
                        if dim == 1 {
                            l * n + e
                        } else {
                            e * shape[1] + l
                        }
                    };
                    let mut seen = vec![false; n];
                    for e in 0..n {
                        let ix = idx[at(e)] as usize;
                        assert!(!seen[ix], "{shape:?}: index repeated");
                        seen[ix] = true;
                        if e > 0 {
                            let (a, b) = (idx[at(e - 1)] as usize, ix);
                            let lane = |k: usize| {
                                values[if dim == 1 {
                                    l * n + k
                                } else {
                                    k * shape[1] + l
                                }]
                            };
                            if lane(a) == lane(b) {
                                assert!(a < b, "{shape:?}: unstable");
                            }
                        }
                    }
                }
            } else {
                let x = Tensor::<TtBackend, 3>::from_data(
                    TensorData::new(values.clone(), [shape[0], shape[1], shape[2]]),
                    &d,
                );
                let flex = Tensor::<Flex, 3>::from_data(
                    TensorData::new(values.clone(), [shape[0], shape[1], shape[2]]),
                    &FlexDevice,
                );
                assert_eq!(
                    bits_of(x.sort(dim).into_data().to_vec::<f32>().unwrap()),
                    bits_of(flex.sort(dim).into_data().to_vec::<f32>().unwrap()),
                    "{shape:?} dim {dim}"
                );
            }
        }
    });
}

fn bf16_words(rows: usize, cols: usize) -> Vec<u16> {
    let patterns: [u16; 10] = [
        0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1, 0xffc2, 0x0001, 0x4000,
    ];
    (0..rows * cols)
        .map(|i| patterns[(i * 7 + i / 3) % patterns.len()])
        .collect()
}

fn bf16_tensor(
    words: &[u16],
    rows: usize,
    cols: usize,
    d: &burn_tt::TtDevice,
) -> Tensor<TtBackend, 2> {
    let mut data = TensorData::new(words.to_vec(), [rows, cols]);
    data.dtype = DType::BF16;
    Tensor::<TtBackend, 2>::from_data(data, (d, DType::BF16))
}

/// BF16 tensors sort through an exact F32 widening, which is monotone in the
/// total order: the indices are the stable ones. (Widening runs on ttsim.)
#[test]
fn bf16_argsorts_are_the_stable_ones() {
    with_device(Config::default(), |d| {
        let (rows, cols) = (4, 45);
        let words = bf16_words(rows, cols);
        let idx = bf16_tensor(&words, rows, cols, &d)
            .argsort(1)
            .into_data()
            .to_vec::<i32>()
            .unwrap();
        for r in 0..rows {
            let lane: Vec<u32> = (0..cols)
                .map(|c| (words[r * cols + c] as u32) << 16)
                .collect();
            let (_, want_i) = reference(Elem::F32, &lane, false);
            let gi: Vec<u32> = (0..cols).map(|c| idx[r * cols + c] as u32).collect();
            assert_eq!(gi, want_i, "row {r} indices");
        }
    });
}

/// The values come back BF16 -- bit-exact -- which needs the narrowing cast:
/// silicon only (ttsim refuses PACR mode 0x105, divergence log).
#[cfg(feature = "silicon")]
#[test]
fn bf16_sorts_return_bf16_values() {
    with_device(Config::default(), |d| {
        let (rows, cols) = (4, 45);
        let words = bf16_words(rows, cols);
        let (v, i) = bf16_tensor(&words, rows, cols, &d).sort_with_indices(1);
        assert_eq!(v.dtype(), DType::BF16);
        let mut got = v.into_data();
        got.dtype = DType::U16;
        let got = got.to_vec::<u16>().unwrap();
        let idx = i.into_data().to_vec::<i32>().unwrap();
        for r in 0..rows {
            let lane: Vec<u32> = (0..cols)
                .map(|c| (words[r * cols + c] as u32) << 16)
                .collect();
            let (want_v, want_i) = reference(Elem::F32, &lane, false);
            // The F32 -> BF16 narrowing flushes BF16 subnormals to signed zero
            // (`docs/plans/hardware-coverage.md`, 2026-10-05): the sorted F32
            // values, flushed, are what comes back.
            let want_v: Vec<u32> = want_v
                .into_iter()
                .map(|w| {
                    if w & 0x7f80_0000 == 0 {
                        w & 0x8000_0000
                    } else {
                        w
                    }
                })
                .collect();
            let g: Vec<u32> = (0..cols)
                .map(|c| (got[r * cols + c] as u32) << 16)
                .collect();
            assert_eq!(g, want_v, "row {r} values");
            let gi: Vec<u32> = (0..cols).map(|c| idx[r * cols + c] as u32).collect();
            assert_eq!(gi, want_i, "row {r} indices");
        }
    });
}

/// A sort across tiles, its argsort and a top-k in one trace: the replay of a
/// changed input is the fresh run, with no upload by the replay.
#[test]
fn multi_pass_sorts_replay_in_a_trace() {
    with_device(Config::default(), |d| {
        let (rows, cols, k) = (3, 100, 7);
        let make = |round: u64| -> Vec<f32> {
            let mut s = round.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            (0..rows * cols)
                .map(|i| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let r = (s >> 33) as usize;
                    match r % 8 {
                        0 => f32::NAN,
                        1 => -0.0,
                        _ => ((r % 13) as f32 - 6.0) + (i % 2) as f32 * 0.0,
                    }
                })
                .collect()
        };
        let x = Tensor::<TtBackend, 2>::from_data(TensorData::new(make(0), [rows, cols]), &d);
        let TensorPrimitive::Float(xp) = x.clone().into_primitive() else {
            unreachable!()
        };
        xp.ensure_resident();
        let up = burn_tt::tensor_traffic().uploads;
        let (trace, first) = TracedInference::capture(&[&xp], || {
            let TensorPrimitive::Float(sorted) = x.clone().sort(1).into_primitive() else {
                unreachable!()
            };
            let TensorPrimitive::Float(top) = x.clone().topk(k, 1).into_primitive() else {
                unreachable!()
            };
            vec![
                sorted,
                x.clone().argsort_descending(1).into_primitive(),
                top,
            ]
        })
        .unwrap();
        assert_eq!(
            burn_tt::tensor_traffic().uploads,
            up,
            "the capture uploaded"
        );
        let want = |v: &[f32]| -> [Vec<u32>; 3] {
            let mut sorted = Vec::new();
            let mut arg = Vec::new();
            let mut top = Vec::new();
            for r in 0..rows {
                let lane: Vec<u32> = v[r * cols..(r + 1) * cols]
                    .iter()
                    .map(|x| x.to_bits())
                    .collect();
                sorted.extend(reference(Elem::F32, &lane, false).0);
                let (dv, di) = reference(Elem::F32, &lane, true);
                arg.extend(di);
                top.extend_from_slice(&dv[..k]);
            }
            [sorted, arg, top]
        };
        let outputs = |o: &[OutputPayload]| -> [Vec<u32>; 3] {
            let word = |p: &OutputPayload| match p {
                OutputPayload::F32(v) => v.iter().map(|x| x.to_bits()).collect(),
                OutputPayload::Bits(b) => b.clone(),
            };
            [word(&o[0]), word(&o[1]), word(&o[2])]
        };
        assert_eq!(outputs(&first), want(&make(0)), "capture run");
        for round in 1..=3u64 {
            let v = make(round);
            let up = burn_tt::tensor_traffic().uploads;
            let out = trace.run(vec![InputPayload::F32(v.clone())]).unwrap();
            assert_eq!(burn_tt::tensor_traffic().uploads, up, "round {round}");
            assert_eq!(outputs(&out), want(&v), "round {round}");
        }
    });
}
