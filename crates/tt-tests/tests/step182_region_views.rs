//! Slices and swaps as views, and ragged views copied by tiles (`docs/completed-plans/decode-host-cost.md`, P2).
//!
//! A slice of rank three or more was an element repack of everything it selected, and `swap_dims`
//! materialised a view before swapping it: ViT-Tiny spent 83 of its 392 ms of server time on them
//! (37 slices of its fused projection and 37 swaps into heads). Now a slice whose steps are all
//! positive is a view of the same buffer (`sliced_native`: the base moved, the strides scaled), a
//! swap of a view stays a view, and a view made plain is a tile copy wherever each output tile is one
//! source block, a ragged last tile row or column included where the source ends there too
//! (`Strided::tile_moves`). A batched matmul reads a head of the projection in place.
//!
//! The claims, each against burn-flex on the same data, bit for bit (copies, and arithmetic on
//! small integers where every result is exact):
//! - slices of ranks 2 to 4 along every dimension, with steps and column offsets, queue **no job**;
//!   so does a swap of one; a negative step is still one copy;
//! - each consumer reads such a view correctly: an element-wise op, a matmul, a reduction, `cat`,
//!   `slice_assign`, a reshape, a read back;
//! - a column block of a `[197, 576]` projection is made plain by **one tile copy**, not a repack;
//! - ViT's heads (a slice of the projection, a reshape, a swap) feed `q k^T` with no slice or swap
//!   job;
//! - I32, Bool, BF16 and F16 slices are views and read back exactly;
//! - a ragged producer, then a slice, then a reduction; a trace replay through views.
//!
//! Watched failing (2026-10-14, ttsim): a slice's base one row down, the column base applied as rows,
//! a step's stride left unscaled, and a ragged tile copy taken where the source does not end.

use burn::tensor::{Bool, Slice};
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

/// Small integers, so sums and products are exact in any order.
fn ints(seed: u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as u64 * seed + 3) % 9) as f32 - 4.0)
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `x` made resident on the card as it is: no arithmetic.
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

/// The attachment's first op asks its engine once whether it keeps tensors on the device (a server
/// round trip): ask before counting jobs.
fn settle(d: &burn_tt::TtDevice) {
    let x = on_card(Tensor::<TtBackend, 2>::from_data([[1.0f32, 2.0]], d));
    let _ = x.slice(&slices(&[(0, 1, 1), (1, 2, 1)])).into_data();
}

fn slices(r: &[(usize, usize, isize)]) -> Vec<Slice> {
    r.iter()
        .map(|&(a, b, step)| Slice::new(a as isize, Some(b as isize), step))
        .collect()
}

/// A slice of F32 raw bits on the card and on burn-flex: no job when every step is positive, one
/// copy otherwise, and the same bits read back.
fn check_slice<const D: usize>(
    d: &burn_tt::TtDevice,
    shape: [usize; D],
    region: &[(usize, usize, isize)],
    seed: u64,
) {
    let data = TensorData::new(raw_bits(seed, shape.iter().product()), shape);
    let s = slices(region);
    let want: Vec<f32> = Tensor::<Flex, D>::from_data(data.clone(), &FlexDevice)
        .slice(&s)
        .into_data()
        .to_vec()
        .unwrap();
    let t = on_card(Tensor::<TtBackend, D>::from_data(data, d));
    let (y, report) = burn_tt::with_report(|| t.slice(&s));
    let forward = region.iter().all(|&(_, _, step)| step > 0);
    let jobs = report.op("float_slice").map_or(0, |s| s.jobs);
    assert_eq!(jobs, u64::from(!forward), "{shape:?} {region:?}\n{report}");
    assert_native_model(&report);
    let got: Vec<f32> = y.into_data().to_vec().unwrap();
    assert_eq!(bits(&got), bits(&want), "{shape:?} {region:?}");
}

#[test]
fn slices_are_views_and_read_back_exactly() {
    with_device(Config::default(), |d| {
        settle(&d);
        check_slice(&d, [70, 66], &[(3, 60, 1), (32, 64, 1)], 1);
        check_slice(&d, [70, 66], &[(1, 70, 3), (0, 66, 2)], 2);
        check_slice(
            &d,
            [3, 197, 576],
            &[(0, 3, 1), (0, 197, 1), (192, 384, 1)],
            3,
        );
        check_slice(&d, [3, 197, 576], &[(1, 3, 1), (5, 197, 7), (3, 576, 5)], 4);
        check_slice(
            &d,
            [2, 3, 9, 64],
            &[(0, 2, 1), (1, 3, 1), (0, 5, 1), (0, 32, 1)],
            5,
        );
        check_slice(
            &d,
            [2, 3, 9, 64],
            &[(0, 2, 1), (0, 3, 1), (4, 9, 2), (32, 64, 1)],
            6,
        );
        // A negative step: one copy.
        check_slice(&d, [4, 5, 7], &[(0, 4, 1), (0, 5, -1), (0, 7, 1)], 7);
        check_slice(&d, [4, 5, 7], &[(0, 4, -2), (0, 5, 1), (1, 7, 1)], 8);
    });
}

/// Each consumer of a slice view reads it as the slice it is (small integers: exact arithmetic).
#[test]
fn every_consumer_reads_a_view_correctly() {
    with_device(Config::default(), |d| {
        let shape = [2, 40, 96];
        let x = ints(5, shape.iter().product());
        let w = ints(7, 32 * 9);
        let f = Tensor::<Flex, 3>::from_data(TensorData::new(x.clone(), shape), &FlexDevice);
        let t = on_card(Tensor::<TtBackend, 3>::from_data(
            TensorData::new(x, shape),
            &d,
        ));
        let fw = Tensor::<Flex, 2>::from_data(TensorData::new(w.clone(), [32, 9]), &FlexDevice);
        let tw = on_card(Tensor::<TtBackend, 2>::from_data(
            TensorData::new(w, [32, 9]),
            &d,
        ));
        // Rows 3..37 step 2 of columns 32..64: neither whole rows nor whole tiles.
        let region = slices(&[(0, 2, 1), (3, 37, 2), (32, 64, 1)]);
        let (fv, tv) = (f.clone().slice(&region), t.clone().slice(&region));
        let read = |a: Tensor<Flex, 3>, b: Tensor<TtBackend, 3>, what: &str| {
            let want: Vec<f32> = a.into_data().to_vec().unwrap();
            let got: Vec<f32> = b.into_data().to_vec().unwrap();
            assert_eq!(bits(&got), bits(&want), "{what}");
        };
        read(fv.clone(), tv.clone(), "read back");
        read(
            fv.clone() + fv.clone().mul_scalar(3.0),
            tv.clone() + tv.clone().mul_scalar(3.0),
            "add",
        );
        read(fv.clone().sum_dim(1), tv.clone().sum_dim(1), "sum_dim(1)");
        read(fv.clone().sum_dim(2), tv.clone().sum_dim(2), "sum_dim(2)");
        read(
            fv.clone().matmul(fw.clone().unsqueeze()),
            tv.clone().matmul(tw.clone().unsqueeze()),
            "matmul",
        );
        read(
            Tensor::cat(vec![fv.clone(), fv.clone()], 1),
            Tensor::cat(vec![tv.clone(), tv.clone()], 1),
            "cat",
        );
        read(
            f.clone().slice_assign(&region, fv.clone().mul_scalar(-1.0)),
            t.clone().slice_assign(&region, tv.clone().mul_scalar(-1.0)),
            "slice_assign",
        );
        read(
            fv.clone().swap_dims(0, 2).swap_dims(1, 2),
            tv.clone().swap_dims(0, 2).swap_dims(1, 2),
            "swaps",
        );
        read(
            fv.clone().reshape([1, 17, 64]),
            tv.clone().reshape([1, 17, 64]),
            "reshape",
        );
        // A swap of a view queues nothing.
        let (_, report) = burn_tt::with_report(|| tv.clone().swap_dims(1, 2));
        assert_eq!(
            report.op("float_swap_dims").map_or(0, |s| s.jobs),
            0,
            "{report}"
        );
    });
}

/// A column block of a `[197, 576]` projection made plain is one tile copy: its last tile row is
/// ragged, and so is the source's, there.
#[test]
fn a_ragged_column_block_is_copied_by_tiles() {
    with_device(Config::default(), |d| {
        let data = raw_bits(21, 197 * 576);
        let f =
            Tensor::<Flex, 2>::from_data(TensorData::new(data.clone(), [197, 576]), &FlexDevice);
        let t = on_card(Tensor::<TtBackend, 2>::from_data(
            TensorData::new(data, [197, 576]),
            &d,
        ));
        let region = slices(&[(0, 197, 1), (192, 384, 1)]);
        let want: Vec<f32> = f.slice(&region).into_data().to_vec().unwrap();
        let view = t.slice(&region);
        let (blocks, repacks) = (ran("copy_blocks"), ran("repack") + ran("repack_many"));
        let TensorPrimitive::Float(p) = view.clone().into_primitive() else {
            unreachable!()
        };
        p.ensure_resident();
        assert_eq!(ran("copy_blocks") - blocks, 1, "one tile copy");
        assert_eq!(ran("repack") + ran("repack_many") - repacks, 0, "no repack");
        let got: Vec<f32> = view.into_data().to_vec().unwrap();
        assert_eq!(bits(&got), bits(&want));

        // Rows 0..197 of a 300-row buffer (rank 3, so a slice view, not a row view of tile rows):
        // the last tile row is ragged where the source is not, so no tile copy can take it; it is
        // repacked, and still exact.
        let data = raw_bits(22, 300 * 64);
        let f =
            Tensor::<Flex, 3>::from_data(TensorData::new(data.clone(), [1, 300, 64]), &FlexDevice);
        let t = on_card(Tensor::<TtBackend, 3>::from_data(
            TensorData::new(data, [1, 300, 64]),
            &d,
        ));
        let region = slices(&[(0, 1, 1), (0, 197, 1), (0, 64, 1)]);
        let want: Vec<f32> = f.slice(&region).into_data().to_vec().unwrap();
        let view = t.slice(&region);
        let (blocks, repacks) = (ran("copy_blocks"), ran("repack"));
        let TensorPrimitive::Float(p) = view.clone().into_primitive() else {
            unreachable!()
        };
        p.ensure_resident();
        assert_eq!(
            ran("copy_blocks") - blocks,
            0,
            "no tile copy inside the source"
        );
        assert_eq!(ran("repack") - repacks, 1, "a repack instead");
        let got: Vec<f32> = view.into_data().to_vec().unwrap();
        assert_eq!(bits(&got), bits(&want));
    });
}

/// ViT's heads: the fused projection sliced into q and k, each reshaped to heads and swapped, then
/// `q k^T`. No slice or swap queues a job; the product reads the heads where they lie.
#[test]
fn vit_heads_feed_a_product_in_place() {
    with_device(Config::default(), |d| {
        settle(&d);
        let (t, heads, dh) = (197, 3, 64);
        let dim = heads * dh;
        let x = ints(3, t * 3 * dim);
        let f =
            Tensor::<Flex, 3>::from_data(TensorData::new(x.clone(), [1, t, 3 * dim]), &FlexDevice);
        let c = on_card(Tensor::<TtBackend, 3>::from_data(
            TensorData::new(x, [1, t, 3 * dim]),
            &d,
        ));
        fn part<B: burn::tensor::backend::Backend>(
            qkv: &Tensor<B, 3>,
            i: usize,
            [t, heads, dh]: [usize; 3],
        ) -> Tensor<B, 4> {
            let dim = heads * dh;
            qkv.clone()
                .slice([0..1, 0..t, i * dim..(i + 1) * dim])
                .reshape([1, t, heads, dh])
                .swap_dims(1, 2)
        }
        let want: Vec<f32> = part(&f, 0, [t, heads, dh])
            .matmul(part(&f, 1, [t, heads, dh]).transpose())
            .into_data()
            .to_vec()
            .unwrap();
        let before = tensor_traffic();
        let (scores, report) = burn_tt::with_report(|| {
            part(&c, 0, [t, heads, dh]).matmul(part(&c, 1, [t, heads, dh]).transpose())
        });
        for op in [
            "float_slice",
            "float_swap_dims",
            "float_reshape",
            "float_transpose",
        ] {
            assert_eq!(report.op(op).map_or(0, |s| s.jobs), 0, "{op}\n{report}");
        }
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let got: Vec<f32> = scores.into_data().to_vec().unwrap();
        assert_eq!(got, want);
    });
}

#[test]
fn int_bool_and_half_precision_slices_are_views() {
    with_device(Config::default(), |d| {
        let region = [0..3, 2..9, 32..64];
        let v: Vec<i32> = (0..3 * 9 * 70usize)
            .map(|i| (i as i32).wrapping_mul(-104_729))
            .collect();
        let want: Vec<i32> =
            Tensor::<Flex, 3, Int>::from_data(TensorData::new(v.clone(), [3, 9, 70]), &FlexDevice)
                .slice(region.clone())
                .into_data()
                .convert::<i32>()
                .to_vec()
                .unwrap();
        let got: Vec<i32> =
            Tensor::<TtBackend, 3, Int>::from_data(TensorData::new(v, [3, 9, 70]), &d)
                .slice(region.clone())
                .into_data()
                .convert::<i32>()
                .to_vec()
                .unwrap();
        assert_eq!(got, want, "I32");
        let b: Vec<bool> = (0..3 * 9 * 70).map(|i| i % 3 == 1).collect();
        let want: Vec<bool> =
            Tensor::<Flex, 3, Bool>::from_data(TensorData::new(b.clone(), [3, 9, 70]), &FlexDevice)
                .slice(region.clone())
                .into_data()
                .to_vec()
                .unwrap();
        let got: Vec<bool> =
            Tensor::<TtBackend, 3, Bool>::from_data(TensorData::new(b, [3, 9, 70]), &d)
                .slice(region.clone())
                .into_data()
                .to_vec()
                .unwrap();
        assert_eq!(got, want, "Bool");
        for dtype in [DType::BF16, DType::F16] {
            let bits16: Vec<u16> = raw_bits(31, 3 * 9 * 70)
                .iter()
                .map(|x| (x.to_bits() >> 16) as u16)
                .collect();
            let data = || {
                let mut t = TensorData::new(bits16.clone(), [3, 9, 70]);
                t.dtype = dtype;
                t
            };
            let want = Tensor::<Flex, 3>::from_data(data(), (&FlexDevice, dtype))
                .slice(region.clone())
                .into_data();
            let got = Tensor::<TtBackend, 3>::from_data(data(), (&d, dtype))
                .slice(region.clone())
                .into_data();
            assert_eq!(got.dtype, dtype);
            assert_eq!(got.as_bytes(), want.as_bytes(), "{dtype:?}");
        }
    });
}

/// A ragged producer, a slice of it, then reductions; and a trace whose ops read slice views.
#[test]
fn ragged_producers_and_a_trace_through_views() {
    with_device(Config::default(), |d| {
        let x = ints(11, 33 * 70);
        let f = Tensor::<Flex, 2>::from_data(TensorData::new(x.clone(), [33, 70]), &FlexDevice);
        let t = Tensor::<TtBackend, 2>::from_data(TensorData::new(x, [33, 70]), &d);
        // `x * 2 - x` is exact and leaves a ragged device result as the producer.
        let fv = (f.clone().mul_scalar(2.0) - f).slice([1..33, 5..70]);
        let tv = (t.clone().mul_scalar(2.0) - t).slice([1..33, 5..70]);
        for k in 0..2 {
            let want: Vec<f32> = fv.clone().sum_dim(k).into_data().to_vec().unwrap();
            let got: Vec<f32> = tv.clone().sum_dim(k).into_data().to_vec().unwrap();
            assert_eq!(got, want, "sum_dim({k}) of a slice of a ragged result");
        }

        let shape = [2, 33, 96];
        let n: usize = shape.iter().product();
        let values = |seed: u64| ints(seed, n);
        let input = Tensor::<TtBackend, 3>::from_data(TensorData::new(values(41), shape), &d)
            .mul_scalar(1.0);
        let TensorPrimitive::Float(p) = input.clone().into_primitive() else {
            unreachable!()
        };
        let forward = |x: Tensor<TtBackend, 3>| {
            let a = x.clone().slice([0..2, 0..33, 0..32]);
            let b = x.slice([0..2, 1..33, 64..96]).swap_dims(0, 1);
            a.sum_dim(1) + b.sum_dim(0).swap_dims(0, 1)
        };
        let host = |v: Vec<f32>| -> Vec<f32> {
            let x = Tensor::<Flex, 3>::from_data(TensorData::new(v, shape), &FlexDevice);
            let a = x.clone().slice([0..2, 0..33, 0..32]);
            let b = x.slice([0..2, 1..33, 64..96]).swap_dims(0, 1);
            (a.sum_dim(1) + b.sum_dim(0).swap_dims(0, 1))
                .into_data()
                .to_vec()
                .unwrap()
        };
        let (trace, first) = burn_tt::Trace::capture(&p, || {
            let TensorPrimitive::Float(y) = forward(input.clone()).into_primitive() else {
                unreachable!()
            };
            y
        })
        .unwrap();
        assert_eq!(first, host(values(41)));
        for seed in [42, 43] {
            assert_eq!(
                trace.run(values(seed)).unwrap(),
                host(values(seed)),
                "replay {seed}"
            );
        }
    });
}
