//! A native `cat` and `repeat_dim`: one device job, not a zero-fill and a `slice_assign` a piece
//! (`docs/plans/decode-host-cost.md`, P1).
//!
//! Burn's default `cat` fills an empty tensor and writes each piece with `slice_assign`, which maps
//! every element of the whole destination on the host and copies all of it: a SmolLM2 decode step
//! spent 61% of its server thread there on every tile (`performance-log.md`, "Decode step: where
//! the host time goes"). Now (`burn-tt` `ops/region.rs`) a `cat` along any dimension but the last,
//! of pieces that are whole rows of their buffers, is one row gather, and any other is one element
//! repack over every piece; `repeat_dim` is a `cat` of the same tensor.
//!
//! The claims, each against burn-flex on the same data (a `cat` is a copy: bit for bit, never a
//! tolerance):
//! - F32 data with signed zeros, subnormals, NaN payloads, infinities and `f32::MAX` comes through
//!   unchanged, along every dimension of ranks 1 to 4, at last dimensions aligned (32, 64) and not
//!   (5, 7, 33), with pieces across tile rows, a 9-way `repeat_kv` and RoPE's two halves;
//! - pieces that are views (a swap, a transpose, a reshape, a row slice) are read where they lie;
//! - I32 and Bool `cat`, and `repeat_dim` of F32, I32 and Bool along every dimension;
//! - BF16 and F16 `cat` (one element repack of the packed buffers);
//! - a ragged producer, then a `cat`, then a row sum and a matmul (exact on small integers);
//! - a trace of a `cat` replays changed inputs;
//! - a 9-way `cat` off the last dimension is **one** job, and neither fills nor `slice_assign`s; a
//!   `cat` downloads nothing and its result is on the device.
//!
//! Watched failing (2026-10-14, ttsim): the pieces taken in reverse order for each outer index, a
//! piece's rows one short (the last row of each run repeated), the element map's column off by one,
//! and a dropped piece (its rows taken from the next).

use burn::tensor::{Bool, Shape};
use burn::tensor::{DType, Int, Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{device_time, tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// `n` datums of raw F32 bits: every fifth one a value a copy must not touch.
fn raw_bits(seed: u64, n: usize) -> Vec<f32> {
    const SPECIAL: [u32; 8] = [
        0x8000_0000, // -0.0
        0x0000_0001, // the least subnormal
        0x7fc1_2345, // a quiet NaN with a payload
        0xff81_0000, // a negative signalling NaN
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7f7f_ffff, // f32::MAX
        0x007f_ffff, // the greatest subnormal
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f32::from_bits(if i % 5 == 0 {
                SPECIAL[(i / 5) % SPECIAL.len()]
            } else {
                (s >> 32) as u32
            })
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn resident<const D: usize>(x: &Tensor<TtBackend, D>) -> bool {
    let TensorPrimitive::Float(x) = x.clone().into_primitive() else {
        unreachable!()
    };
    x.computed_on_device()
}

/// `x` made resident on the card as it is: no arithmetic, which could quiet a NaN or flush a
/// subnormal.
fn on_card<const D: usize>(x: Tensor<TtBackend, D>) -> Tensor<TtBackend, D> {
    let TensorPrimitive::Float(p) = x.clone().into_primitive() else {
        unreachable!()
    };
    p.ensure_resident();
    x
}

/// How many times the server ran a kind of job in this process so far.
fn ran(kind: &str) -> u64 {
    device_time()
        .iter()
        .find(|(k, _, _)| *k == kind)
        .map_or(0, |(_, n, _)| *n)
}

/// The jobs `op` queued in a report.
fn jobs(report: &burn_tt::Report, op: &str) -> u64 {
    report.op(op).map_or(0, |s| s.jobs)
}

/// `cat` of F32 pieces of `shapes` along `dim` on the card and on burn-flex, bit for bit; the
/// card's pieces are made resident first, so the `cat` is the only job.
fn check_cat<const D: usize>(d: &burn_tt::TtDevice, shapes: &[[usize; D]], dim: usize, seed: u64) {
    let datas: Vec<TensorData> = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            TensorData::new(
                raw_bits(seed * 31 + i as u64, s.iter().product()),
                Shape::from(s.to_vec()),
            )
        })
        .collect();
    let want: Vec<f32> = Tensor::<Flex, D>::cat(
        datas
            .iter()
            .map(|t| Tensor::<Flex, D>::from_data(t.clone(), &FlexDevice))
            .collect(),
        dim,
    )
    .into_data()
    .to_vec()
    .unwrap();
    let pieces: Vec<Tensor<TtBackend, D>> = datas
        .iter()
        .map(|t| on_card(Tensor::<TtBackend, D>::from_data(t.clone(), d)))
        .collect();
    let before = tensor_traffic();
    let (gathers, repacks) = (ran("gather_rows"), ran("repack_many"));
    let (y, report) = burn_tt::with_report(|| Tensor::cat(pieces, dim));
    // Off the last dimension, whole rows: one row gather; along it, one element repack.
    let by_rows = dim + 1 < D;
    assert_eq!(
        (ran("gather_rows") - gathers, ran("repack_many") - repacks),
        if by_rows { (1, 0) } else { (0, 1) },
        "{shapes:?} along {dim}: the lowering"
    );
    assert!(resident(&y), "{shapes:?} along {dim}: not on the device");
    assert_eq!(
        jobs(&report, "float_cat"),
        1,
        "{shapes:?} along {dim}: one job\n{report}"
    );
    assert!(report.op("float_empty").is_none() && report.op("float_slice_assign").is_none());
    assert_native_model(&report);
    assert_eq!(tensor_traffic().downloads, before.downloads);
    let got: Vec<f32> = y.into_data().to_vec().unwrap();
    assert_eq!(bits(&got), bits(&want), "{shapes:?} along {dim}");
}

#[test]
fn cats_are_burn_flex_bit_for_bit_in_one_job() {
    with_device(Config::default(), |d| {
        // The attachment's first op asks its engine once whether it keeps tensors on the device (a
        // server round trip): ask before counting.
        let settle = on_card(Tensor::<TtBackend, 1>::from_data([1.0f32, 2.0], &d));
        let _ = Tensor::cat(vec![settle.clone(), settle], 0).into_data();
        check_cat(&d, &[[5], [7], [33]], 0, 1);
        check_cat(&d, &[[3, 5], [4, 5], [40, 5]], 0, 2);
        check_cat(&d, &[[3, 5], [3, 40], [3, 1]], 1, 3);
        check_cat(&d, &[[40, 33], [70, 33]], 0, 4);
        check_cat(&d, &[[1, 3, 7], [2, 3, 7]], 0, 5);
        check_cat(&d, &[[2, 3, 7], [2, 1, 7], [2, 4, 7]], 1, 6);
        check_cat(&d, &[[2, 3, 32], [2, 3, 32]], 2, 7);
        check_cat(&d, &[[2, 3, 4, 5], [1, 3, 4, 5]], 0, 8);
        check_cat(&d, &[[2, 3, 4, 5], [2, 5, 4, 5]], 1, 9);
        check_cat(&d, &[[2, 3, 4, 5], [2, 3, 1, 5]], 2, 10);
        check_cat(&d, &[[2, 3, 4, 5], [2, 3, 4, 7]], 3, 11);
        // SmolLM2's: the 9-way `repeat_kv` of a 5-position cache, and RoPE's two halves.
        check_cat(&d, &[[1, 1, 5, 64]; 9], 1, 12);
        check_cat(&d, &[[1, 9, 1, 32], [1, 9, 1, 32]], 3, 13);
        check_cat(&d, &[[1, 9, 40, 32], [1, 9, 40, 32]], 3, 14);
    });
}

#[test]
fn view_pieces_are_read_where_they_lie() {
    with_device(Config::default(), |d| {
        let a = raw_bits(21, 2 * 4 * 6);
        let b = raw_bits(22, 5 * 7);
        let c = raw_bits(23, 64 * 6);
        // Three `[4, 2, 6]` pieces: a swap of leading dimensions (whole rows of its buffer), a
        // slice of a reshape, and a reshape of a row slice of a resident `[64, 6]` (a row view).
        fn pieces<B: burn::tensor::backend::Backend>(
            a: &[f32],
            c: &[f32],
            d: &B::Device,
            keep3: impl Fn(Tensor<B, 3>) -> Tensor<B, 3>,
            keep2: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
        ) -> Vec<Tensor<B, 3>> {
            let t3 = |v: &[f32], s: [usize; 3]| {
                keep3(Tensor::from_data(TensorData::new(v.to_vec(), s), d))
            };
            let c2 = keep2(Tensor::from_data(TensorData::new(c.to_vec(), [64, 6]), d));
            vec![
                t3(a, [2, 4, 6]).swap_dims(0, 1),
                t3(c, [4, 16, 6]).narrow(1, 3, 2),
                c2.narrow(0, 32, 8).reshape([4, 2, 6]),
            ]
        }
        let pf = pieces::<Flex>(&a, &c, &FlexDevice, |x| x, |x| x);
        let pt = pieces::<TtBackend>(&a, &c, &d, on_card, on_card);
        for dim in 0..3 {
            let want: Vec<f32> = Tensor::cat(pf.clone(), dim).into_data().to_vec().unwrap();
            let got: Vec<f32> = Tensor::cat(pt.clone(), dim).into_data().to_vec().unwrap();
            assert_eq!(bits(&got), bits(&want), "view pieces along {dim}");
        }

        // A transposed rank-2 view, along both dimensions.
        let fb = Tensor::<Flex, 2>::from_data(TensorData::new(b.clone(), [7, 5]), &FlexDevice);
        let tb = on_card(Tensor::<TtBackend, 2>::from_data(
            TensorData::new(b.clone(), [7, 5]),
            &d,
        ));
        for dim in 0..2 {
            let want: Vec<f32> =
                Tensor::cat(vec![fb.clone().transpose(), fb.clone().transpose()], dim)
                    .into_data()
                    .to_vec()
                    .unwrap();
            let got: Vec<f32> =
                Tensor::cat(vec![tb.clone().transpose(), tb.clone().transpose()], dim)
                    .into_data()
                    .to_vec()
                    .unwrap();
            assert_eq!(bits(&got), bits(&want), "transposed views along {dim}");
        }
    });
}

#[test]
fn int_bool_and_repeat_dim_are_burn_flex_exactly() {
    with_device(Config::default(), |d| {
        let ints: Vec<i32> = (0..2 * 3 * 33usize)
            .map(|i| match i % 7 {
                0 => i32::MIN,
                1 => i32::MAX,
                2 => -1,
                3 => 0,
                _ => (i as i32).wrapping_mul(-104_729),
            })
            .collect();
        let bools: Vec<bool> = (0..2 * 3 * 33).map(|i| (i * 7 + 3) % 5 < 2).collect();
        let fi = Tensor::<Flex, 3, Int>::from_data(
            TensorData::new(ints.clone(), [2, 3, 33]),
            &FlexDevice,
        );
        let ti =
            Tensor::<TtBackend, 3, Int>::from_data(TensorData::new(ints.clone(), [2, 3, 33]), &d);
        let fbool = Tensor::<Flex, 3, Bool>::from_data(
            TensorData::new(bools.clone(), [2, 3, 33]),
            &FlexDevice,
        );
        let tbool =
            Tensor::<TtBackend, 3, Bool>::from_data(TensorData::new(bools.clone(), [2, 3, 33]), &d);
        let ff = Tensor::<Flex, 3>::from_data(
            TensorData::new(raw_bits(31, 2 * 3 * 33), [2, 3, 33]),
            &FlexDevice,
        );
        let tf = Tensor::<TtBackend, 3>::from_data(
            TensorData::new(raw_bits(31, 2 * 3 * 33), [2, 3, 33]),
            &d,
        );
        for dim in 0..3 {
            let want: Vec<i32> =
                Tensor::cat(vec![fi.clone(), fi.clone().flip([dim as isize])], dim)
                    .into_data()
                    .convert::<i32>()
                    .to_vec()
                    .unwrap();
            let got: Vec<i32> = Tensor::cat(vec![ti.clone(), ti.clone().flip([dim as isize])], dim)
                .into_data()
                .convert::<i32>()
                .to_vec()
                .unwrap();
            assert_eq!(got, want, "int cat along {dim}");
            let want: Vec<bool> = Tensor::cat(vec![fbool.clone(), fbool.clone().bool_not()], dim)
                .into_data()
                .to_vec()
                .unwrap();
            let got: Vec<bool> = Tensor::cat(vec![tbool.clone(), tbool.clone().bool_not()], dim)
                .into_data()
                .to_vec()
                .unwrap();
            assert_eq!(got, want, "bool cat along {dim}");
            for times in [1, 3] {
                let want: Vec<f32> = ff
                    .clone()
                    .repeat_dim(dim, times)
                    .into_data()
                    .to_vec()
                    .unwrap();
                let (y, report) = burn_tt::with_report(|| tf.clone().repeat_dim(dim, times));
                assert!(report.op("float_slice_assign").is_none(), "{report}");
                let got: Vec<f32> = y.into_data().to_vec().unwrap();
                assert_eq!(bits(&got), bits(&want), "float repeat_dim {dim} x{times}");
                let want: Vec<i32> = fi
                    .clone()
                    .repeat_dim(dim, times)
                    .into_data()
                    .convert::<i32>()
                    .to_vec()
                    .unwrap();
                let got: Vec<i32> = ti
                    .clone()
                    .repeat_dim(dim, times)
                    .into_data()
                    .convert::<i32>()
                    .to_vec()
                    .unwrap();
                assert_eq!(got, want, "int repeat_dim {dim} x{times}");
                let want: Vec<bool> = fbool
                    .clone()
                    .repeat_dim(dim, times)
                    .into_data()
                    .to_vec()
                    .unwrap();
                let got: Vec<bool> = tbool
                    .clone()
                    .repeat_dim(dim, times)
                    .into_data()
                    .to_vec()
                    .unwrap();
                assert_eq!(got, want, "bool repeat_dim {dim} x{times}");
            }
        }
    });
}

/// BF16 and F16: one element repack of the packed buffers, every bit kept.
#[test]
fn half_precision_cats_keep_every_bit() {
    with_device(Config::default(), |d| {
        for dtype in [DType::BF16, DType::F16] {
            let half = |seed: u64, n: usize| -> Vec<u16> {
                raw_bits(seed, n)
                    .iter()
                    .map(|v| (v.to_bits() >> 16) as u16)
                    .collect()
            };
            let data = |bits: Vec<u16>, shape: [usize; 3]| {
                let mut t = TensorData::new(bits, shape);
                t.dtype = dtype;
                t
            };
            let (a, b) = (half(41, 2 * 3 * 40), half(42, 2 * 5 * 40));
            for (shape_b, dim) in [([2, 5, 40], 1), ([2, 3, 40], 2), ([2, 3, 40], 0)] {
                let b = b[..shape_b.iter().product::<usize>()].to_vec();
                let want = Tensor::<Flex, 3>::cat(
                    vec![
                        Tensor::from_data(data(a.clone(), [2, 3, 40]), (&FlexDevice, dtype)),
                        Tensor::from_data(data(b.clone(), shape_b), (&FlexDevice, dtype)),
                    ],
                    dim,
                )
                .into_data();
                let got = Tensor::<TtBackend, 3>::cat(
                    vec![
                        Tensor::from_data(data(a.clone(), [2, 3, 40]), (&d, dtype)),
                        Tensor::from_data(data(b.clone(), shape_b), (&d, dtype)),
                    ],
                    dim,
                )
                .into_data();
                assert_eq!(got.dtype, dtype);
                assert_eq!(got.as_bytes(), want.as_bytes(), "{dtype:?} along {dim}");
            }
        }
    });
}

/// A ragged device result, a `cat`, then a row sum and a matmul: the `cat`'s padding claims hold
/// where an accumulation reads past the data (small integers: every sum is exact).
#[test]
fn a_ragged_cat_feeds_a_reduction_and_a_matmul() {
    with_device(Config::default(), |d| {
        let ints = |n: usize, k: i32| {
            (0..n)
                .map(|i| ((i as i32 * k) % 7 - 3) as f32)
                .collect::<Vec<_>>()
        };
        let (a, b) = (ints(33 * 17, 5), ints(7 * 17, 3));
        let f = |v: &[f32], s: [usize; 2]| {
            Tensor::<Flex, 2>::from_data(TensorData::new(v.to_vec(), s), &FlexDevice)
        };
        let t = |v: &[f32], s: [usize; 2]| {
            Tensor::<TtBackend, 2>::from_data(TensorData::new(v.to_vec(), s), &d)
        };
        for dim in 0..2 {
            let (bf, bt, ws) = if dim == 0 {
                (f(&b, [7, 17]), t(&b, [7, 17]), [17, 9])
            } else {
                (
                    f(&ints(33 * 7, 3), [33, 7]),
                    t(&ints(33 * 7, 3), [33, 7]),
                    [24, 9],
                )
            };
            let wv = ints(ws[0] * ws[1], 2);
            // `x * 2 - x` is exact on these integers and leaves a ragged device result as the
            // producer.
            let cf = Tensor::cat(
                vec![f(&a, [33, 17]).mul_scalar(2.0) - f(&a, [33, 17]), bf],
                dim,
            );
            let ct = Tensor::cat(
                vec![t(&a, [33, 17]).mul_scalar(2.0) - t(&a, [33, 17]), bt],
                dim,
            );
            for k in 0..2 {
                let want: Vec<f32> = cf.clone().sum_dim(k).into_data().to_vec().unwrap();
                let got: Vec<f32> = ct.clone().sum_dim(k).into_data().to_vec().unwrap();
                assert_eq!(got, want, "cat along {dim}, then sum_dim({k})");
            }
            let want: Vec<f32> = cf.matmul(f(&wv, ws)).into_data().to_vec().unwrap();
            let got: Vec<f32> = ct.matmul(t(&wv, ws)).into_data().to_vec().unwrap();
            assert_eq!(got, want, "cat along {dim}, then matmul");
        }
    });
}

/// A trace of `cat`s off and along the last dimension replays changed inputs.
#[test]
fn a_trace_of_cats_replays_changed_inputs() {
    with_device(Config::default(), |d| {
        let shape = [3, 4, 64];
        let n: usize = shape.iter().product();
        // Integers: `* 2` is exact on them, so the oracle is burn-flex's own `* 2`.
        let values = |seed: u64| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as u64 * seed) % 1000) as f32 - 500.0)
                .collect()
        };
        let x = Tensor::<TtBackend, 3>::from_data(TensorData::new(values(51), shape), &d)
            .mul_scalar(1.0);
        let TensorPrimitive::Float(input) = x.clone().into_primitive() else {
            unreachable!()
        };
        let forward = |x: Tensor<TtBackend, 3>| {
            let rows = Tensor::cat(vec![x.clone(), x.clone().mul_scalar(2.0)], 1);
            let halves = Tensor::cat(
                vec![x.clone().narrow(2, 32, 32), x.clone().narrow(2, 0, 32)],
                2,
            );
            Tensor::cat(vec![rows, halves], 1)
        };
        let host = |v: &[f32]| -> Vec<u32> {
            let fx = Tensor::<Flex, 3>::from_data(TensorData::new(v.to_vec(), shape), &FlexDevice);
            let rows = Tensor::cat(vec![fx.clone(), fx.clone().mul_scalar(2.0)], 1);
            let halves = Tensor::cat(
                vec![fx.clone().narrow(2, 32, 32), fx.clone().narrow(2, 0, 32)],
                2,
            );
            bits(
                &Tensor::cat(vec![rows, halves], 1)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            )
        };
        let (trace, first) = burn_tt::Trace::capture(&input, || {
            let TensorPrimitive::Float(y) = forward(x.clone()).into_primitive() else {
                unreachable!()
            };
            y
        })
        .unwrap();
        assert_eq!(bits(&first), host(&values(51)));
        for seed in [52, 53] {
            assert_eq!(
                bits(&trace.run(values(seed)).unwrap()),
                host(&values(seed)),
                "replay {seed}"
            );
        }
    });
}
