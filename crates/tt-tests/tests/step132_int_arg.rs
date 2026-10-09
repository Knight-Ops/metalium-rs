//! `int_argmax`/`int_argmin` (first index of the signed extreme) and Burn's composed
//! `int_max_dim_with_indices`/`int_min_dim_with_indices` over them, on the device.
//!
//! The oracle is plain Rust (`max_by`/`min_by` scanning for the first extreme) and
//! Flex. The data carries what a float image of an `i32` cannot: words above 2^24
//! that differ by one (a float comparison ties them), the signed extremes, ties, and
//! all-equal columns.
use burn::tensor::{Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, InputPayload, TracedInference, TtBackend, TtTensor};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// First index of the extreme of `column`, signed.
fn first_extreme(column: &[i32], minimum: bool) -> usize {
    let mut best = 0;
    for (i, &v) in column.iter().enumerate() {
        let better = if minimum {
            v < column[best]
        } else {
            v > column[best]
        };
        if better {
            best = i;
        }
    }
    best
}

fn arg_along(values: &[i32], shape: &[usize], dim: usize, minimum: bool) -> Vec<i32> {
    let inner: usize = shape[dim + 1..].iter().product();
    let n = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let mut out = Vec::new();
    for o in 0..outer {
        for i in 0..inner {
            let column: Vec<i32> = (0..n).map(|k| values[(o * n + k) * inner + i]).collect();
            out.push(first_extreme(&column, minimum) as i32);
        }
    }
    out
}

/// Words with ties and with neighbours a float image cannot tell apart.
fn words(n: usize, seed: u64) -> Vec<i32> {
    let table = [
        1 << 24,
        (1 << 24) + 1,
        (1 << 24) + 2,
        -(1 << 24),
        -(1 << 24) - 1,
        i32::MIN,
        i32::MAX,
        i32::MAX - 1,
        i32::MIN + 1,
        0,
        -1,
        1,
        0x7fff_ff00,
        -0x7fff_ff00,
        0x3fff_ffff,
        0x4000_0000,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let w = (s >> 33) as usize;
            if i % 5 == 4 {
                (w as u32).wrapping_mul(0x9e37_79b9) as i32
            } else {
                table[w % table.len()]
            }
        })
        .collect()
}

fn check_rank<const D: usize>(d: &burn_tt::TtDevice, shape: [usize; D], values: Vec<i32>) {
    let data = TensorData::new(values.clone(), shape.to_vec());
    for dim in 0..D {
        for minimum in [false, true] {
            let want = arg_along(&values, &shape, dim, minimum);
            let flex = {
                let t = Tensor::<Flex, D, Int>::from_data(data.clone(), &FlexDevice);
                let r = if minimum {
                    t.argmin(dim)
                } else {
                    t.argmax(dim)
                };
                r.into_data().convert::<i32>().to_vec::<i32>().unwrap()
            };
            assert_eq!(flex, want, "the oracle is Flex's: {shape:?} dim {dim}");
            let x = Tensor::<TtBackend, D, Int>::from_data(data.clone(), d);
            let before = tensor_traffic();
            let (y, report) = burn_tt::with_report(|| {
                if minimum {
                    x.clone().argmin(dim)
                } else {
                    x.clone().argmax(dim)
                }
            });
            assert_native_model(&report);
            let name = if minimum { "int_argmin" } else { "int_argmax" };
            let op = report.op(name).unwrap();
            assert_eq!((op.downloads, op.staged, op.on_host), (0, 0, 0), "{name}");
            assert_eq!(tensor_traffic().downloads, before.downloads);
            assert!(y.clone().into_primitive().computed_on_device());
            let mut out_shape = shape;
            out_shape[dim] = 1;
            assert_eq!(y.dims(), out_shape);
            assert_eq!(
                y.into_data().to_vec::<i32>().unwrap(),
                want,
                "{name} {shape:?} dim {dim}"
            );

            // Burn's composed value-and-index methods, over the native pair.
            let (values_out, indices_out) = {
                let (r, report) = burn_tt::with_report(|| {
                    if minimum {
                        x.clone().min_dim_with_indices(dim)
                    } else {
                        x.clone().max_dim_with_indices(dim)
                    }
                });
                assert_native_model(&report);
                r
            };
            assert_eq!(indices_out.into_data().to_vec::<i32>().unwrap(), want);
            let extreme: Vec<i32> = {
                let inner: usize = shape[dim + 1..].iter().product();
                let n = shape[dim];
                let outer: usize = shape[..dim].iter().product();
                let mut e = Vec::new();
                for o in 0..outer {
                    for i in 0..inner {
                        let c = (0..n).map(|k| values[(o * n + k) * inner + i]);
                        e.push(if minimum { c.min() } else { c.max() }.unwrap());
                    }
                }
                e
            };
            assert_eq!(values_out.into_data().to_vec::<i32>().unwrap(), extreme);
        }
    }
}

#[test]
fn int_arg_extremes_are_first_signed_on_every_axis_and_rank() {
    with_device(Config::default(), |d| {
        check_rank::<1>(&d, [67], words(67, 1));
        check_rank::<2>(&d, [67, 35], words(67 * 35, 2));
        check_rank::<2>(&d, [3, 129], words(3 * 129, 3));
        check_rank::<3>(&d, [3, 35, 5], words(3 * 35 * 5, 4));
        check_rank::<4>(&d, [2, 3, 34, 4], words(2 * 3 * 34 * 4, 5));
    });
}

#[test]
fn neighbours_above_2_pow_24_ties_and_extremes_are_exact() {
    with_device(Config::default(), |d| {
        // Rows: [2^24, 2^24+1, 2^24, 2^24+1] etc. as columns of a [4, 6] matrix.
        let columns: [[i32; 4]; 6] = [
            [1 << 24, (1 << 24) + 1, 1 << 24, (1 << 24) + 1],
            [-(1 << 24) - 1, -(1 << 24), -(1 << 24) - 1, -(1 << 24)],
            [i32::MIN; 4],
            [i32::MAX; 4],
            [i32::MIN, i32::MAX, i32::MAX, i32::MIN],
            [0, -1, 0, -1],
        ];
        let mut values = vec![0; 24];
        for (c, col) in columns.iter().enumerate() {
            for (r, &v) in col.iter().enumerate() {
                values[r * 6 + c] = v;
            }
        }
        let x = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(values, [4, 6]), &d);
        let max = x.clone().argmax(0).into_data().to_vec::<i32>().unwrap();
        let min = x.argmin(0).into_data().to_vec::<i32>().unwrap();
        assert_eq!(max, [1, 1, 0, 0, 1, 0]);
        assert_eq!(min, [0, 0, 0, 0, 0, 1]);
    });
}

fn primitive<const D: usize>(t: Tensor<TtBackend, D, Int>) -> TtTensor {
    t.into_primitive()
}

#[test]
fn ragged_arg_extremes_feed_reductions_and_replay_a_trace_on_new_inputs() {
    with_device(Config::default(), |d| {
        for (rows, cols) in [(32, 32), (40, 33)] {
            let forward = |x: Tensor<TtBackend, 2, Int>| {
                // An index vector into an integer reduction, and the picked values
                // into another.
                let (v, i) = x.clone().max_dim_with_indices(0);
                (i.clone().sum_dim(1), v.sum_dim(1), x.argmin(1).max_dim(0))
            };
            let want = |values: &[i32]| -> [Vec<i32>; 3] {
                let imax = arg_along(values, &[rows, cols], 0, false);
                let vmax: Vec<i32> = (0..cols)
                    .map(|c| (0..rows).map(|r| values[r * cols + c]).max().unwrap())
                    .collect();
                let imin = arg_along(values, &[rows, cols], 1, true);
                [
                    vec![imax.iter().fold(0i32, |a, &b| a.wrapping_add(b))],
                    vec![vmax.iter().fold(0i32, |a, &b| a.wrapping_add(b))],
                    vec![*imin.iter().max().unwrap()],
                ]
            };
            let first = words(rows * cols, 7);
            let x = Tensor::<TtBackend, 2, Int>::from_data(
                TensorData::new(first.clone(), [rows, cols]),
                &d,
            );
            let xp = primitive(x.clone());
            let (outs, report) = burn_tt::with_report(|| forward(x.clone()));
            assert_native_model(&report);
            let read = |t: Tensor<TtBackend, 2, Int>| t.into_data().to_vec::<i32>().unwrap();
            let w = want(&first);
            assert_eq!(read(outs.0), w[0], "{rows}x{cols}");
            assert_eq!(read(outs.1), w[1], "{rows}x{cols}");
            assert_eq!(read(outs.2), w[2], "{rows}x{cols}");

            let (trace, _) = TracedInference::capture(&[&xp], || {
                let (a, b, c) = forward(x.clone());
                vec![primitive(a), primitive(b), primitive(c)]
            })
            .unwrap();
            for seed in [8u64, 9] {
                let next = words(rows * cols, seed);
                let payload = next.iter().map(|&v| v as u32).collect();
                let out = trace.run(vec![InputPayload::Bits(payload)]).unwrap();
                let w = want(&next);
                for k in 0..3 {
                    let got: Vec<i32> = out[k]
                        .as_bits()
                        .unwrap()
                        .iter()
                        .map(|&b| b as i32)
                        .collect();
                    assert_eq!(got, w[k], "{rows}x{cols} replay {seed} output {k}");
                }
            }
        }
    });
}
