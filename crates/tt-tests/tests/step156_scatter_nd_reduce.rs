//! `float/int_scatter_nd` with a `Mul`, `Min` or `Max` reduction, on the device.
//!
//! Found by Burn's own backend tests (`burn-tt-conformance`: 26 tests, the `scatter_nd_{mul,min,max}`
//! family and their gradients under both autodiff binaries): these three were refused by name as
//! "duplicates undefined". The refusal was a deferral. The `Assign` path already applies the index
//! tuples one at a time in index order, so the same loop folds a duplicate row the way Flex's
//! sequential update does: `Mul` replaces the row by `current * value`, `Min` / `Max` replace an
//! element only when `value < current` (`value > current`) holds.
//!
//! Oracles, independent of the implementation:
//!
//! * the host definition, applied tuple by tuple in index order over raw words;
//! * Flex for the same inputs. `Min`/`Max` only compare and move, so Flex equals the device bit for
//!   bit over every special (signed zeros, subnormals, infinities, NaN payloads, extremes). Float
//!   `Mul` is held to the specification's `SFPMAD` multiply (`tt_isa::numerics::mul_bh`) chained in
//!   index order, because that is the arithmetic the device runs; on ordinary normals Flex's IEEE
//!   `*=` agrees bit for bit, and the gate says so. I32 `Mul` wraps modulo 2^32: the host reference
//!   uses `wrapping_mul`, Flex is compared on products that cannot overflow.
//!
//! Negative controls, each asserted to differ from the device's result on the gate's data: `Min`/`Max`
//! with `<=`/`>=` (a `-0.0` over `+0.0` replaces), with a NaN allowed to replace, and with the first
//! writer winning; `Mul` with the first tuple only and with a skipped duplicate.
use burn::tensor::{backend::Backend, IndexingUpdateOp, Int, Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend, TtDevice};
use tt_isa::numerics;
use tt_tests::burn_device::{assert_native_model, with_device, Config};
use tt_tests::data::{float_data, lcg};

// ---------------------------------------------------------------- data

const SPECIALS: [u32; 14] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc1_2345,
    0xffc5_4321,
    0x7f80_0001,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x3f80_0000,
];

/// Every third word a special (signed zero, subnormal, infinity, NaN payload,
/// extreme), the rest random normals.
fn float_words(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            let w = lcg(&mut s);
            if i % 3 == 0 {
                SPECIALS[w as usize % SPECIALS.len()]
            } else {
                normal(w)
            }
        })
        .collect()
}

/// A normal number with exponent in 2^-7 .. 2^8 and a random sign and mantissa.
fn normal(w: u64) -> u32 {
    let exponent = 120 + (w >> 24) as u32 % 16;
    (((w >> 9) as u32 & 1) << 31) | (exponent << 23) | (w as u32 & 0x7f_ffff)
}

fn normal_words(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n).map(|_| normal(lcg(&mut s))).collect()
}

/// Words with the signed extremes and neighbours a float image cannot separate.
fn int_words(n: usize, seed: u64) -> Vec<i32> {
    let table = [
        i32::MIN,
        i32::MAX,
        1 << 24,
        (1 << 24) + 1,
        -(1 << 24) - 1,
        0,
        -1,
        1,
        0x7fff_ff00,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            let w = lcg(&mut s);
            if i % 2 == 0 {
                table[w as usize % table.len()]
            } else {
                (w as u32).wrapping_mul(0x9e37_79b9) as i32
            }
        })
        .collect()
}

/// Small integers whose sums never overflow (Flex's integer `+=` panics on overflow
/// in debug builds).
fn small_ints(n: usize, seed: u64) -> Vec<i32> {
    let mut s = seed | 1;
    (0..n).map(|_| lcg(&mut s) as i32 % 100 - 50).collect()
}

/// `n` index tuples of `k` coordinates in range: the all-zero tuple, the last
/// element, then random ones; with `duplicates`, every odd tuple from the third on
/// repeats an earlier one.
fn tuples(shape: &[usize], k: usize, n: usize, seed: u64, duplicates: bool) -> Vec<i32> {
    let mut s = seed | 1;
    let mut out: Vec<i32> = Vec::new();
    for t in 0..n {
        let tuple: Vec<i32> = if t == 0 {
            vec![0; k]
        } else if t == 1 {
            (0..k).map(|j| shape[j] as i32 - 1).collect()
        } else if duplicates && t % 2 == 1 {
            let e = lcg(&mut s) as usize % t;
            out[e * k..e * k + k].to_vec()
        } else {
            (0..k)
                .map(|j| (lcg(&mut s) as usize % shape[j]) as i32)
                .collect()
        };
        out.extend(tuple);
    }
    out
}

// ---------------------------------------------------------------- host reference

/// The `[P]` row of every tuple: `sum_j i_j * stride_j`, strides of the first `k`
/// axes. With `swapped` the strides are in the wrong order (the negative control).
fn rows_of(shape: &[usize], k: usize, tuples: &[i32], swapped: bool) -> Vec<usize> {
    let mut stride = vec![1usize; k];
    for j in (0..k.saturating_sub(1)).rev() {
        stride[j] = stride[j + 1] * shape[j + 1];
    }
    if swapped {
        stride.reverse();
    }
    let p: usize = shape[..k].iter().product();
    tuples
        .chunks(k)
        .map(|t| {
            t.iter()
                .zip(&stride)
                .map(|(&i, &s)| i as usize * s)
                .sum::<usize>()
                % p
        })
        .collect()
}

/// `data` with `values[t]` combined into row `rows[t]` for `t` in index order.
/// `first_wins` keeps the first write of a row (the negative control), `reverse`
/// applies the tuples last to first.
fn ref_scatter<T: Copy>(
    data: &[T],
    rest: usize,
    rows: &[usize],
    values: &[T],
    first_wins: bool,
    reverse: bool,
    f: impl Fn(T, T) -> T,
) -> Vec<T> {
    let mut out = data.to_vec();
    let mut written = vec![false; data.len() / rest];
    let order: Vec<usize> = if reverse {
        (0..rows.len()).rev().collect()
    } else {
        (0..rows.len()).collect()
    };
    for t in order {
        let r = rows[t];
        if first_wins && written[r] {
            continue;
        }
        written[r] = true;
        for c in 0..rest {
            out[r * rest + c] = f(out[r * rest + c], values[t * rest + c]);
        }
    }
    out
}

fn has_duplicates(rows: &[usize]) -> bool {
    let mut sorted = rows.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).any(|w| w[0] == w[1])
}

// ---------------------------------------------------------------- tensors

fn read_bits<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn tt_float<const D: usize>(d: &TtDevice, bits: &[u32], shape: [usize; D]) -> Tensor<TtBackend, D> {
    Tensor::<TtBackend, D>::from_data(TensorData::new(float_data(bits), shape), d).to_device(d)
}

fn tt_int<const D: usize>(
    d: &TtDevice,
    values: &[i32],
    shape: [usize; D],
) -> Tensor<TtBackend, D, Int> {
    Tensor::<TtBackend, D, Int>::from_data(TensorData::new(values.to_vec(), shape), d).to_device(d)
}

/// Indices computed on the device, as a model's would be.
fn tt_indices<const M: usize>(
    d: &TtDevice,
    values: &[i32],
    shape: [usize; M],
) -> Tensor<TtBackend, M, Int> {
    let t = Tensor::<TtBackend, M, Int>::from_data(TensorData::new(values.to_vec(), shape), d)
        .add_scalar(0);
    assert!(t.clone().into_primitive().computed_on_device());
    t
}

fn is_resident<const D: usize>(t: &Tensor<TtBackend, D>) -> bool {
    let TensorPrimitive::Float(p) = t.clone().into_primitive() else {
        unreachable!()
    };
    p.computed_on_device()
}

/// The op ran natively, on the card, moving nothing across PCIe.
fn assert_resident(report: &burn_tt::Report, name: &str) {
    assert_native_model(report);
    let op = report
        .op(name)
        .unwrap_or_else(|| panic!("{name} was not reported"));
    assert_eq!((op.downloads, op.staged, op.on_host), (0, 0, 0), "{name}");
    assert!(op.on_device > 0, "{name} did no device work");
}

fn out_shape(shape: &[usize], idx_shape: &[usize]) -> Vec<usize> {
    let k = idx_shape[idx_shape.len() - 1];
    let mut out = idx_shape[..idx_shape.len() - 1].to_vec();
    out.extend_from_slice(&shape[k..]);
    out
}

// ---------------------------------------------------------------- the reductions

#[derive(Clone, Copy, Debug)]
enum Op {
    Mul,
    Min,
    Max,
}

impl Op {
    fn burn(self) -> IndexingUpdateOp {
        match self {
            Op::Mul => IndexingUpdateOp::Mul,
            Op::Min => IndexingUpdateOp::Min,
            Op::Max => IndexingUpdateOp::Max,
        }
    }
}

/// What the device does to one float element, as the specification defines it.
fn float_model(op: Op, cur: u32, v: u32) -> u32 {
    let (c, x) = (f32::from_bits(cur), f32::from_bits(v));
    match op {
        Op::Mul => numerics::mul_bh(cur, v),
        Op::Min => {
            if x < c {
                v
            } else {
                cur
            }
        }
        Op::Max => {
            if x > c {
                v
            } else {
                cur
            }
        }
    }
}

/// Plausible wrong `Min`/`Max`, which the ties test must tell from the right one: `0` lets a tie
/// replace (`<=`, `>=`), `1` lets a NaN value replace (`!(x >= c)`, `!(x <= c)`). The negated
/// comparisons are the point: they differ from `x < c` exactly when a NaN is involved.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn float_mutation(op: Op, which: usize, cur: u32, v: u32) -> u32 {
    let (c, x) = (f32::from_bits(cur), f32::from_bits(v));
    let replaces = match (op, which) {
        (Op::Min, 0) => x <= c,
        (Op::Max, 0) => x >= c,
        (Op::Min, _) => !(x >= c),
        (Op::Max, _) => !(x <= c),
        (Op::Mul, _) => unreachable!("only Min and Max have these mutations"),
    };
    if replaces {
        v
    } else {
        cur
    }
}

fn int_model(op: Op, cur: i32, v: i32) -> i32 {
    match op {
        Op::Mul => cur.wrapping_mul(v),
        Op::Min => cur.min(v),
        Op::Max => cur.max(v),
    }
}

fn run_float<const D: usize, const M: usize, const DV: usize>(
    d: &TtDevice,
    shape: [usize; D],
    idx_shape: [usize; M],
    seed: u64,
    op: Op,
) {
    let k = idx_shape[M - 1];
    let n: usize = idx_shape[..M - 1].iter().product();
    let rest: usize = shape[k..].iter().product();
    let count: usize = shape.iter().product();
    let vshape: [usize; DV] = out_shape(&shape, &idx_shape).try_into().unwrap();
    for ordinary in [false, true] {
        let (bits, vbits) = if ordinary {
            (
                normal_words(count, seed),
                normal_words(n * rest, seed + 100),
            )
        } else {
            (float_words(count, seed), float_words(n * rest, seed + 100))
        };
        let tup = tuples(&shape, k, n, seed, true);
        let rows = rows_of(&shape, k, &tup, false);
        assert!(has_duplicates(&rows), "the gate needs duplicate rows");
        let want = ref_scatter(&bits, rest, &rows, &vbits, false, false, |c, v| {
            float_model(op, c, v)
        });
        // Controls. `Assign` ignores the current value; `Mul` also differs from a fold that
        // skips the duplicates (the first writer of a row only). `Min`/`Max` do not depend on
        // the order of non-tied values, so the duplicate control for them is in the ties test.
        // (On the special-laden pass a row may hold only infinities and NaNs, which nothing
        // replaces; there Flex is the oracle, and the controls are asserted on the normals.)
        if ordinary {
            let assign = ref_scatter(&bits, rest, &rows, &vbits, false, false, |_, v| v);
            assert_ne!(want, assign, "negative control: Assign, {op:?}");
            assert_ne!(want, bits, "negative control: nothing applied, {op:?}");
        }
        if matches!(op, Op::Mul) {
            let first = ref_scatter(&bits, rest, &rows, &vbits, true, false, |c, v| {
                float_model(op, c, v)
            });
            assert_ne!(want, first, "negative control: first writer only, {op:?}");
        }
        let x = tt_float(d, &bits, shape);
        let v = tt_float(d, &vbits, vshape);
        let i = tt_indices(d, &tup, idx_shape);
        let before = tensor_traffic();
        let (y, report) = burn_tt::with_report(|| {
            x.clone()
                .scatter_nd::<M, DV>(i.clone(), v.clone(), op.burn())
        });
        assert_resident(&report, "float_scatter_nd");
        assert_eq!(tensor_traffic().downloads, before.downloads);
        assert!(is_resident(&y));
        assert_eq!(y.dims(), shape);
        let got = read_bits(y);
        assert_eq!(got, want, "float {op:?} {shape:?} ordinary={ordinary}");
        // Flex: bit for bit for the comparisons over every special; for Mul on ordinary
        // normals, where the device's multiply is the IEEE one.
        if ordinary || !matches!(op, Op::Mul) {
            let flex = Tensor::<Flex, D>::from_data(
                TensorData::new(float_data(&bits), shape),
                &FlexDevice,
            )
            .scatter_nd::<M, DV>(
                Tensor::<Flex, M, Int>::from_data(
                    TensorData::new(tup.clone(), idx_shape),
                    &FlexDevice,
                ),
                Tensor::<Flex, DV>::from_data(
                    TensorData::new(float_data(&vbits), vshape),
                    &FlexDevice,
                ),
                op.burn(),
            );
            assert_eq!(
                read_bits(flex),
                got,
                "Flex is the oracle: {op:?} {shape:?} ordinary={ordinary}"
            );
        }
    }
}

fn run_int<const D: usize, const M: usize, const DV: usize>(
    d: &TtDevice,
    shape: [usize; D],
    idx_shape: [usize; M],
    seed: u64,
    op: Op,
) {
    let k = idx_shape[M - 1];
    let n: usize = idx_shape[..M - 1].iter().product();
    let rest: usize = shape[k..].iter().product();
    let count: usize = shape.iter().product();
    let vshape: [usize; DV] = out_shape(&shape, &idx_shape).try_into().unwrap();
    // Full-width words for the device (a product wraps modulo 2^32); Flex only on small ones.
    for small in [false, true] {
        let (values, vv) = if small {
            (small_ints(count, seed), small_ints(n * rest, seed + 100))
        } else {
            (int_words(count, seed), int_words(n * rest, seed + 100))
        };
        let tup = tuples(&shape, k, n, seed, true);
        let rows = rows_of(&shape, k, &tup, false);
        assert!(has_duplicates(&rows), "the gate needs duplicate rows");
        let want = ref_scatter(&values, rest, &rows, &vv, false, false, |c, v| {
            int_model(op, c, v)
        });
        if matches!(op, Op::Mul) {
            let first = ref_scatter(&values, rest, &rows, &vv, true, false, |c, v| {
                int_model(op, c, v)
            });
            assert_ne!(want, first, "negative control: first writer only, {op:?}");
        }
        let assign = ref_scatter(&values, rest, &rows, &vv, false, false, |_, v| v);
        assert_ne!(want, assign, "negative control: Assign, {op:?}");
        assert_ne!(want, values, "negative control: nothing applied, {op:?}");
        let (x, v) = (tt_int(d, &values, shape), tt_int(d, &vv, vshape));
        let i = tt_indices(d, &tup, idx_shape);
        let before = tensor_traffic();
        let (y, report) = burn_tt::with_report(|| {
            x.clone()
                .scatter_nd::<M, DV>(i.clone(), v.clone(), op.burn())
        });
        assert_resident(&report, "int_scatter_nd");
        assert_eq!(tensor_traffic().downloads, before.downloads);
        assert!(y.clone().into_primitive().computed_on_device());
        let got = y.into_data().to_vec::<i32>().unwrap();
        assert_eq!(got, want, "int {op:?} {shape:?} small={small}");
        if small || !matches!(op, Op::Mul) {
            let flex = Tensor::<Flex, D, Int>::from_data(
                TensorData::new(values.clone(), shape),
                &FlexDevice,
            )
            .scatter_nd::<M, DV>(
                Tensor::<Flex, M, Int>::from_data(TensorData::new(tup, idx_shape), &FlexDevice),
                Tensor::<Flex, DV, Int>::from_data(TensorData::new(vv, vshape), &FlexDevice),
                op.burn(),
            );
            assert_eq!(
                flex.into_data().convert::<i32>().to_vec::<i32>().unwrap(),
                got,
                "Flex is the oracle: {op:?} {shape:?}"
            );
        }
    }
}

const OPS: [Op; 3] = [Op::Mul, Op::Min, Op::Max];

#[test]
fn float_scatter_nd_reductions_fold_duplicates_in_index_order() {
    with_device(Config::default(), |d| {
        for (s, op) in OPS.into_iter().enumerate() {
            let s = s as u64 * 10;
            run_float::<3, 3, 3>(&d, [3, 4, 5], [2, 3, 2], 61 + s, op);
            run_float::<2, 2, 2>(&d, [35, 3], [6, 1], 62 + s, op);
            run_float::<1, 2, 1>(&d, [37], [9, 1], 63 + s, op);
            run_float::<4, 2, 2>(&d, [2, 3, 5, 7], [4, 3], 64 + s, op);
        }
    });
}

#[test]
fn int_scatter_nd_reductions_fold_duplicates_in_index_order() {
    with_device(Config::default(), |d| {
        for (s, op) in OPS.into_iter().enumerate() {
            let s = s as u64 * 10;
            run_int::<3, 3, 3>(&d, [3, 4, 5], [2, 3, 2], 71 + s, op);
            run_int::<2, 2, 2>(&d, [35, 3], [6, 1], 72 + s, op);
            run_int::<1, 2, 1>(&d, [37], [9, 1], 73 + s, op);
        }
    });
}

/// The cases random data may not reach: a tie between `-0.0` and `+0.0`, NaN on either side, a
/// subnormal against zero, infinities. One row of four elements, written by one tuple at a time.
#[test]
fn min_max_ties_nans_and_zeros_follow_the_comparison() {
    const Z: u32 = 0x0000_0000;
    const NZ: u32 = 0x8000_0000;
    const SUB: u32 = 0x0000_0001;
    const NAN: u32 = 0x7fc0_0001;
    const INF: u32 = 0x7f80_0000;
    const ONE: u32 = 0x3f80_0000;
    // (data, value) pairs per element; every pair is one tuple into the same row.
    let data = [Z, NZ, NAN, SUB, ONE, INF, NAN, Z];
    let first = [NZ, Z, ONE, Z, NAN, NAN, NAN, SUB];
    let second = [NZ, NZ, NAN, NZ, INF, ONE, INF, NZ];
    with_device(Config::default(), |d| {
        for op in [Op::Min, Op::Max] {
            let mut want = data.to_vec();
            let mut tie = data.to_vec();
            let mut nan_replaces = data.to_vec();
            let mut first_only = data.to_vec();
            for c in 0..8 {
                first_only[c] = float_model(op, first_only[c], first[c]);
            }
            for pass in [&first, &second] {
                for c in 0..8 {
                    want[c] = float_model(op, want[c], pass[c]);
                    tie[c] = float_mutation(op, 0, tie[c], pass[c]);
                    nan_replaces[c] = float_mutation(op, 1, nan_replaces[c], pass[c]);
                }
            }
            assert_ne!(want, tie, "negative control: a tie replaces, {op:?}");
            assert_ne!(want, nan_replaces, "negative control: NaN replaces, {op:?}");
            assert_ne!(
                want, first_only,
                "negative control: first writer only, {op:?}"
            );
            let x = tt_float(&d, &data, [1, 8]);
            let v = tt_float(
                &d,
                &first.iter().chain(&second).copied().collect::<Vec<_>>(),
                [2, 8],
            );
            let i = tt_indices(&d, &[0, 0], [2, 1]);
            let y = x.scatter_nd::<2, 2>(i, v, op.burn());
            assert_eq!(read_bits(y), want, "{op:?}");
        }
    });
}
