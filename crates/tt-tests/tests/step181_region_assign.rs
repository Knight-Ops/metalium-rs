//! `slice_assign` of whole rows as one row gather (`docs/completed-plans/decode-host-cost.md`, P3).
//!
//! `slice_assign` mapped every element of the whole destination on the host and repacked all of
//! it: a SmolLM2 cache write of one position into a `[1, 3, 64, 64]` cache cost 254 us of server
//! time, 60 a decode step. Now (`burn-tt` `ops/region.rs`) a region that writes the last dimension
//! whole and in order is a tile copy of the destination with the value's rows written over it
//! (`Session::assign_rows`: a few run records and entries for the written rows only), or, for a
//! destination that is a view, one row gather over the destination and the value; any other region
//! keeps the element map. The result is a new buffer every way, so a tensor that still holds the
//! destination keeps its contents.
//!
//! The claims, each against burn-flex on the same data (a copy: bit for bit):
//! - F32 data with every special value, for regions along every dimension of ranks 2 to 4: a
//!   cache's positions, stepped and reversed rows, a head, the whole tensor; a region of half the
//!   last dimension (the element path);
//! - **the destination is not written**: SmolLM2's `cache.clone().slice_assign(..)`, step after
//!   step, leaves every earlier version as it was;
//! - destinations and values that are views (a swap, a transpose);
//! - I32, Bool, BF16 and F16;
//! - a ragged producer, then `slice_assign`, then a row sum and a matmul (exact on integers);
//! - a trace of a cache write replays changed inputs;
//! - a whole-row region is **one** copy-and-write job (a view destination one row gather) and no
//!   repack, and nothing is downloaded.
//!
//! Watched failing (2026-10-14, ttsim): the region's rows one row down, the value's rows taken in
//! reverse, the two sources swapped, and a region of half the last dimension let through by rows.

use burn::tensor::{Bool, Shape};
use burn::tensor::{DType, Int, Slice, Tensor, TensorData, TensorPrimitive};
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

fn slices(r: &[(usize, usize, isize)]) -> Vec<Slice> {
    r.iter()
        .map(|&(a, b, step)| Slice::new(a as isize, Some(b as isize), step))
        .collect()
}

/// The value's shape for `region` (`(start, end, step)` per dimension).
fn region_shape(region: &[(usize, usize, isize)]) -> Vec<usize> {
    region
        .iter()
        .map(|&(a, b, step)| (b - a).div_ceil(step.unsigned_abs()))
        .collect()
}

/// `slice_assign` of F32 raw bits on the card and on burn-flex, bit for bit; the card's operands
/// are resident first, so the assignment is the only job: the value's rows written over a tile copy
/// (`assign_rows`) if `by_rows`, else one element repack.
fn check_assign<const D: usize>(
    d: &burn_tt::TtDevice,
    shape: [usize; D],
    region: &[(usize, usize, isize)],
    by_rows: bool,
    seed: u64,
) {
    let vs = region_shape(region);
    let dest = TensorData::new(raw_bits(seed, shape.iter().product()), shape);
    let value = TensorData::new(
        raw_bits(seed + 1000, vs.iter().product()),
        Shape::from(vs.clone()),
    );
    let s = slices(region);
    let want: Vec<f32> = Tensor::<Flex, D>::from_data(dest.clone(), &FlexDevice)
        .slice_assign(&s, Tensor::from_data(value.clone(), &FlexDevice))
        .into_data()
        .to_vec()
        .unwrap();
    let t = on_card(Tensor::<TtBackend, D>::from_data(dest, d));
    let v = on_card(Tensor::<TtBackend, D>::from_data(value, d));
    let before = tensor_traffic();
    let (writes, gathers, repacks) = (ran("assign_rows"), ran("gather_rows"), ran("repack_many"));
    let (y, report) = burn_tt::with_report(|| t.slice_assign(&s, v));
    assert_eq!(
        (
            ran("assign_rows") - writes,
            ran("gather_rows") - gathers,
            ran("repack_many") - repacks
        ),
        if by_rows { (1, 0, 0) } else { (0, 0, 1) },
        "{shape:?} <- {region:?}: the lowering"
    );
    assert_eq!(
        report.op("float_slice_assign").map(|s| s.jobs),
        Some(1),
        "{report}"
    );
    assert_native_model(&report);
    assert_eq!(tensor_traffic().downloads, before.downloads);
    let got: Vec<f32> = y.into_data().to_vec().unwrap();
    assert_eq!(bits(&got), bits(&want), "{shape:?} <- {region:?}");
}

#[test]
fn slice_assign_is_burn_flex_bit_for_bit_in_one_job() {
    with_device(Config::default(), |d| {
        // The attachment's first op asks its engine once whether it keeps tensors on the device (a
        // server round trip): ask before counting.
        let settle = on_card(Tensor::<TtBackend, 1>::from_data([1.0f32, 2.0], &d));
        let _ = settle
            .clone()
            .slice_assign(&slices(&[(0, 1, 1)]), settle.slice(&slices(&[(1, 2, 1)])))
            .into_data();

        // A key/value cache's positions.
        check_assign(
            &d,
            [1, 3, 8, 64],
            &[(0, 1, 1), (0, 3, 1), (5, 7, 1), (0, 64, 1)],
            true,
            1,
        );
        check_assign(
            &d,
            [1, 3, 64, 64],
            &[(0, 1, 1), (0, 3, 1), (63, 64, 1), (0, 64, 1)],
            true,
            2,
        );
        // Rows of a matrix: a run across a tile row, stepped, reversed.
        check_assign(&d, [70, 33], &[(20, 45, 1), (0, 33, 1)], true, 3);
        check_assign(&d, [70, 33], &[(1, 70, 3), (0, 33, 1)], true, 4);
        check_assign(&d, [70, 33], &[(5, 40, -2), (0, 33, 1)], true, 5);
        // A head, a middle dimension, the whole tensor.
        check_assign(&d, [4, 5, 7], &[(2, 3, 1), (0, 5, 1), (0, 7, 1)], true, 6);
        check_assign(&d, [4, 5, 7], &[(0, 4, 1), (1, 4, 1), (0, 7, 1)], true, 7);
        check_assign(&d, [2, 6], &[(0, 2, 1), (0, 6, 1)], true, 8);
        // Not whole rows: half the last dimension, and the last dimension reversed.
        check_assign(&d, [4, 64], &[(0, 4, 1), (32, 64, 1)], false, 9);
        check_assign(
            &d,
            [4, 2, 64],
            &[(1, 3, 1), (0, 2, 1), (0, 64, -1)],
            false,
            10,
        );
    });
}

/// SmolLM2's value: a projection's `[1, 192]` row split into heads (`reshape`, then `swap_dims`), a
/// view whose rows are parts of a buffer row, written at a cache position. The value is made plain
/// first (a repack of its own elements), then written by rows: never a map of the whole cache.
#[test]
fn a_value_split_into_heads_is_made_plain_then_written_by_rows() {
    with_device(Config::default(), |d| {
        let (cache, k) = (raw_bits(61, 3 * 64 * 64), raw_bits(62, 192));
        let want: Vec<f32> = Tensor::<Flex, 4>::from_data(
            TensorData::new(cache.clone(), [1, 3, 64, 64]),
            &FlexDevice,
        )
        .slice_assign(
            [0..1, 0..3, 9..10, 0..64],
            Tensor::<Flex, 3>::from_data(TensorData::new(k.clone(), [1, 1, 192]), &FlexDevice)
                .reshape([1, 1, 3, 64])
                .swap_dims(1, 2),
        )
        .into_data()
        .to_vec()
        .unwrap();
        let c = on_card(Tensor::<TtBackend, 4>::from_data(
            TensorData::new(cache, [1, 3, 64, 64]),
            &d,
        ));
        let heads = on_card(Tensor::<TtBackend, 3>::from_data(
            TensorData::new(k, [1, 1, 192]),
            &d,
        ))
        .reshape([1, 1, 3, 64])
        .swap_dims(1, 2);
        let (writes, repacks) = (ran("assign_rows"), ran("repack_many") + ran("repack"));
        let got: Vec<f32> = c
            .slice_assign([0..1, 0..3, 9..10, 0..64], heads)
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(bits(&got), bits(&want));
        assert_eq!(ran("assign_rows") - writes, 1, "written by rows");
        assert!(
            ran("repack_many") + ran("repack") - repacks <= 1,
            "at most the value's own repack"
        );
    });
}

/// SmolLM2's cache: `cache = cache.clone().slice_assign(position, k)`, step after step. Every
/// earlier version is kept and compared at the end, so a write into the destination's buffer
/// would show in one of them.
#[test]
fn the_destination_keeps_its_contents() {
    with_device(Config::default(), |d| {
        let shape = [1, 3, 16, 64];
        let start = raw_bits(11, shape.iter().product());
        let mut flex =
            Tensor::<Flex, 4>::from_data(TensorData::new(start.clone(), shape), &FlexDevice);
        let mut card = on_card(Tensor::<TtBackend, 4>::from_data(
            TensorData::new(start, shape),
            &d,
        ));
        let (mut flex_versions, mut card_versions) = (vec![flex.clone()], vec![card.clone()]);
        for p in 0..16 {
            let k = raw_bits(100 + p as u64, 3 * 64);
            let region = [0..1, 0..3, p..p + 1, 0..64];
            flex = flex.clone().slice_assign(
                region.clone(),
                Tensor::from_data(TensorData::new(k.clone(), [1, 3, 1, 64]), &FlexDevice),
            );
            card = card.clone().slice_assign(
                region,
                on_card(Tensor::from_data(TensorData::new(k, [1, 3, 1, 64]), &d)),
            );
            flex_versions.push(flex.clone());
            card_versions.push(card.clone());
        }
        for (i, (f, c)) in flex_versions.into_iter().zip(card_versions).enumerate() {
            let want: Vec<f32> = f.into_data().to_vec().unwrap();
            let got: Vec<f32> = c.into_data().to_vec().unwrap();
            assert_eq!(bits(&got), bits(&want), "version {i}");
        }
    });
}

#[test]
fn views_int_bool_and_half_precision() {
    with_device(Config::default(), |d| {
        // A destination that is a swap of leading dimensions (whole rows) and a value that is one
        // too; then a transposed value (not whole rows).
        let a = raw_bits(21, 2 * 4 * 6);
        let b = raw_bits(22, 2 * 2 * 6);
        let fa = Tensor::<Flex, 3>::from_data(TensorData::new(a.clone(), [2, 4, 6]), &FlexDevice);
        let fb = Tensor::<Flex, 3>::from_data(TensorData::new(b.clone(), [2, 2, 6]), &FlexDevice);
        let ta = on_card(Tensor::<TtBackend, 3>::from_data(
            TensorData::new(a, [2, 4, 6]),
            &d,
        ));
        let tb = on_card(Tensor::<TtBackend, 3>::from_data(
            TensorData::new(b, [2, 2, 6]),
            &d,
        ));
        let want: Vec<f32> = fa
            .clone()
            .swap_dims(0, 1)
            .slice_assign([1..3, 0..2, 0..6], fb.clone().swap_dims(0, 1))
            .into_data()
            .to_vec()
            .unwrap();
        // A destination that is a view is one row gather over it and the value.
        let gathers = ran("gather_rows");
        let got: Vec<f32> = ta
            .clone()
            .swap_dims(0, 1)
            .slice_assign([1..3, 0..2, 0..6], tb.clone().swap_dims(0, 1))
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(bits(&got), bits(&want), "views");
        assert_eq!(
            ran("gather_rows") - gathers,
            1,
            "a view destination by rows"
        );
        let m = raw_bits(23, 6 * 5);
        let fm = Tensor::<Flex, 2>::from_data(TensorData::new(m.clone(), [6, 5]), &FlexDevice);
        let tm = on_card(Tensor::<TtBackend, 2>::from_data(
            TensorData::new(m, [6, 5]),
            &d,
        ));
        let want: Vec<f32> = fa
            .reshape([8, 6])
            .slice_assign([1..6, 0..6], fm.transpose())
            .into_data()
            .to_vec()
            .unwrap();
        let got: Vec<f32> = ta
            .reshape([8, 6])
            .slice_assign([1..6, 0..6], tm.transpose())
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(bits(&got), bits(&want), "a transposed value");

        // I32 and Bool, by rows.
        let ints: Vec<i32> = (0..5 * 33usize)
            .map(|i| match i % 5 {
                0 => i32::MIN,
                1 => i32::MAX,
                _ => (i as i32).wrapping_mul(-7919),
            })
            .collect();
        let vals: Vec<i32> = (0..2 * 33).map(|i| -i - 1).collect();
        let want: Vec<i32> =
            Tensor::<Flex, 2, Int>::from_data(TensorData::new(ints.clone(), [5, 33]), &FlexDevice)
                .slice_assign(
                    [3..5, 0..33],
                    Tensor::from_data(TensorData::new(vals.clone(), [2, 33]), &FlexDevice),
                )
                .into_data()
                .convert::<i32>()
                .to_vec()
                .unwrap();
        let got: Vec<i32> =
            Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(ints, [5, 33]), &d)
                .slice_assign(
                    [3..5, 0..33],
                    Tensor::from_data(TensorData::new(vals, [2, 33]), &d),
                )
                .into_data()
                .convert::<i32>()
                .to_vec()
                .unwrap();
        assert_eq!(got, want, "I32");
        let bools: Vec<bool> = (0..5 * 33).map(|i| i % 3 == 0).collect();
        let bv: Vec<bool> = (0..2 * 33).map(|i| i % 2 == 0).collect();
        let want: Vec<bool> = Tensor::<Flex, 2, Bool>::from_data(
            TensorData::new(bools.clone(), [5, 33]),
            &FlexDevice,
        )
        .slice_assign(
            [0..2, 0..33],
            Tensor::from_data(TensorData::new(bv.clone(), [2, 33]), &FlexDevice),
        )
        .into_data()
        .to_vec()
        .unwrap();
        let got: Vec<bool> =
            Tensor::<TtBackend, 2, Bool>::from_data(TensorData::new(bools, [5, 33]), &d)
                .slice_assign(
                    [0..2, 0..33],
                    Tensor::from_data(TensorData::new(bv, [2, 33]), &d),
                )
                .into_data()
                .to_vec()
                .unwrap();
        assert_eq!(got, want, "Bool");

        // BF16 and F16: the element path over the packed buffers.
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
            let (x, v) = (half(41, 2 * 8 * 40), half(42, 2 * 3 * 40));
            let want =
                Tensor::<Flex, 3>::from_data(data(x.clone(), [2, 8, 40]), (&FlexDevice, dtype))
                    .slice_assign(
                        [0..2, 4..7, 0..40],
                        Tensor::from_data(data(v.clone(), [2, 3, 40]), (&FlexDevice, dtype)),
                    )
                    .into_data();
            let got = Tensor::<TtBackend, 3>::from_data(data(x, [2, 8, 40]), (&d, dtype))
                .slice_assign(
                    [0..2, 4..7, 0..40],
                    Tensor::from_data(data(v, [2, 3, 40]), (&d, dtype)),
                )
                .into_data();
            assert_eq!(got.dtype, dtype);
            assert_eq!(got.as_bytes(), want.as_bytes(), "{dtype:?}");
        }
    });
}

/// A ragged device result, a `slice_assign` of whole rows, then a row sum and a matmul (small
/// integers: every sum is exact).
#[test]
fn a_ragged_assignment_feeds_a_reduction_and_a_matmul() {
    with_device(Config::default(), |d| {
        let ints = |n: usize, k: i32| {
            (0..n)
                .map(|i| ((i as i32 * k) % 7 - 3) as f32)
                .collect::<Vec<_>>()
        };
        let (a, v, w) = (ints(33 * 17, 5), ints(4 * 17, 3), ints(17 * 9, 2));
        let f = |x: &[f32], s: [usize; 2]| {
            Tensor::<Flex, 2>::from_data(TensorData::new(x.to_vec(), s), &FlexDevice)
        };
        let t = |x: &[f32], s: [usize; 2]| {
            Tensor::<TtBackend, 2>::from_data(TensorData::new(x.to_vec(), s), &d)
        };
        // `x * 2 - x` is exact on these integers and leaves a ragged device result as the producer.
        let cf = (f(&a, [33, 17]).mul_scalar(2.0) - f(&a, [33, 17]))
            .slice_assign([29..33, 0..17], f(&v, [4, 17]));
        let ct = (t(&a, [33, 17]).mul_scalar(2.0) - t(&a, [33, 17]))
            .slice_assign([29..33, 0..17], t(&v, [4, 17]));
        for k in 0..2 {
            let want: Vec<f32> = cf.clone().sum_dim(k).into_data().to_vec().unwrap();
            let got: Vec<f32> = ct.clone().sum_dim(k).into_data().to_vec().unwrap();
            assert_eq!(got, want, "sum_dim({k})");
        }
        let want: Vec<f32> = cf.matmul(f(&w, [17, 9])).into_data().to_vec().unwrap();
        let got: Vec<f32> = ct.matmul(t(&w, [17, 9])).into_data().to_vec().unwrap();
        assert_eq!(got, want, "matmul");
    });
}

/// A trace of a cache write: the input is the new key, written at a position of a resident cache,
/// then the cache summed over positions. Replays with changed keys.
#[test]
fn a_trace_of_a_cache_write_replays_changed_inputs() {
    with_device(Config::default(), |d| {
        let values = |seed: u64, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as u64 * seed) % 1000) as f32 - 500.0)
                .collect()
        };
        let cache_data = values(7, 3 * 8 * 64);
        let cache =
            Tensor::<TtBackend, 3>::from_data(TensorData::new(cache_data.clone(), [3, 8, 64]), &d)
                .mul_scalar(1.0);
        let k =
            Tensor::<TtBackend, 3>::from_data(TensorData::new(values(51, 3 * 64), [3, 1, 64]), &d)
                .mul_scalar(1.0);
        let TensorPrimitive::Float(input) = k.clone().into_primitive() else {
            unreachable!()
        };
        let host = |kv: &[f32]| -> Vec<u32> {
            let c = Tensor::<Flex, 3>::from_data(
                TensorData::new(cache_data.clone(), [3, 8, 64]),
                &FlexDevice,
            );
            let k =
                Tensor::<Flex, 3>::from_data(TensorData::new(kv.to_vec(), [3, 1, 64]), &FlexDevice);
            bits(
                &c.slice_assign([0..3, 5..6, 0..64], k)
                    .sum_dim(1)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            )
        };
        let (trace, first) = burn_tt::Trace::capture(&input, || {
            let y = cache
                .clone()
                .slice_assign([0..3, 5..6, 0..64], k.clone())
                .sum_dim(1);
            let TensorPrimitive::Float(y) = y.into_primitive() else {
                unreachable!()
            };
            y
        })
        .unwrap();
        assert_eq!(bits(&first), host(&values(51, 3 * 64)));
        for seed in [52, 53] {
            let v = values(seed, 3 * 64);
            assert_eq!(
                bits(&trace.run(v.clone()).unwrap()),
                host(&v),
                "replay {seed}"
            );
        }
    });
}
