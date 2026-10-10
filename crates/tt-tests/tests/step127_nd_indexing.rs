//! `float/int_gather_nd` and `float/int_scatter_nd` on the device (lane T2).
//!
//! The oracles are independent of the implementation: a host reference built from
//! the definition (row `sum_j i_j * stride_j` of the `[P, R]` view; sequential
//! application of the update in index order) and Flex for the same inputs. Float
//! `Add` is held to the specification's `SFPMAD` add model (`tt_isa::numerics::add_bh`)
//! chained in index order, because that is the arithmetic the device runs; where the
//! data are ordinary normal numbers Flex's IEEE `+=` agrees bit for bit, and the
//! gate says so. Gathers and `Assign` move raw datums, so signed zeros, subnormals
//! and NaN payloads must survive.
//!
//! Duplicate-index contract (stated in `ops_index.rs`): `Add` folds in index order,
//! `Assign` is last-writer-wins in index order, `Mul`/`Min`/`Max` are refused naming
//! the variant. Every coordinate is bounds-checked against its own axis on the device
//! (code 11 DOMAIN), not only the flat row.
//!
//! Negative controls, each asserted below to differ from the device on the gate's
//! data (and each watched failing against a mutated implementation, see the lane
//! report): a gather with transposed strides; a scatter `Assign` with first-writer-wins;
//! a scatter `Add` in reverse index order; a flat-row-only domain check.
use burn::tensor::{backend::Backend, IndexingUpdateOp, Int, Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TileChoice, TtBackend, TtDevice};
use tt_isa::numerics;
use tt_tests::burn_device::{assert_native_model, with_device, Config};

// ---------------------------------------------------------------- data

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

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

fn ref_gather<T: Copy>(data: &[T], rest: usize, rows: &[usize]) -> Vec<T> {
    rows.iter()
        .flat_map(|&r| data[r * rest..(r + 1) * rest].iter().copied())
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

fn float_data(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

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

// ---------------------------------------------------------------- gather_nd

fn gather_float<const D: usize, const M: usize, const DV: usize>(
    d: &TtDevice,
    shape: [usize; D],
    idx_shape: [usize; M],
    seed: u64,
) {
    let k = idx_shape[M - 1];
    let n: usize = idx_shape[..M - 1].iter().product();
    let rest: usize = shape[k..].iter().product();
    let bits = float_words(shape.iter().product(), seed);
    let tup = tuples(&shape, k, n, seed, true);
    let want = ref_gather(&bits, rest, &rows_of(&shape, k, &tup, false));
    if k > 1 {
        let wrong = ref_gather(&bits, rest, &rows_of(&shape, k, &tup, true));
        assert_ne!(want, wrong, "transposed strides must be detectable here");
    }
    let x = tt_float(d, &bits, shape);
    let i = tt_indices(d, &tup, idx_shape);
    let before = tensor_traffic();
    let (y, report) = burn_tt::with_report(|| x.clone().gather_nd::<M, DV>(i.clone()));
    assert_resident(&report, "float_gather_nd");
    assert_eq!(tensor_traffic().downloads, before.downloads);
    assert!(is_resident(&y));
    assert_eq!(y.dims().to_vec(), out_shape(&shape, &idx_shape));
    assert_eq!(
        read_bits(y),
        want,
        "float gather_nd {shape:?} {idx_shape:?}"
    );
    let flex = Tensor::<Flex, D>::from_data(TensorData::new(float_data(&bits), shape), &FlexDevice)
        .gather_nd::<M, DV>(Tensor::<Flex, M, Int>::from_data(
            TensorData::new(tup, idx_shape),
            &FlexDevice,
        ));
    assert_eq!(read_bits(flex), want, "Flex is the oracle: {shape:?}");
}

fn gather_int<const D: usize, const M: usize, const DV: usize>(
    d: &TtDevice,
    shape: [usize; D],
    idx_shape: [usize; M],
    seed: u64,
) {
    let k = idx_shape[M - 1];
    let n: usize = idx_shape[..M - 1].iter().product();
    let rest: usize = shape[k..].iter().product();
    let values = int_words(shape.iter().product(), seed);
    let tup = tuples(&shape, k, n, seed, true);
    let want = ref_gather(&values, rest, &rows_of(&shape, k, &tup, false));
    if k > 1 {
        let wrong = ref_gather(&values, rest, &rows_of(&shape, k, &tup, true));
        assert_ne!(want, wrong, "transposed strides must be detectable here");
    }
    let x = tt_int(d, &values, shape);
    let i = tt_indices(d, &tup, idx_shape);
    let before = tensor_traffic();
    let (y, report) = burn_tt::with_report(|| x.clone().gather_nd::<M, DV>(i.clone()));
    assert_resident(&report, "int_gather_nd");
    assert_eq!(tensor_traffic().downloads, before.downloads);
    assert!(y.clone().into_primitive().computed_on_device());
    assert_eq!(y.dims().to_vec(), out_shape(&shape, &idx_shape));
    assert_eq!(
        y.into_data().to_vec::<i32>().unwrap(),
        want,
        "int gather_nd {shape:?} {idx_shape:?}"
    );
    let flex = Tensor::<Flex, D, Int>::from_data(TensorData::new(values, shape), &FlexDevice)
        .gather_nd::<M, DV>(Tensor::<Flex, M, Int>::from_data(
            TensorData::new(tup, idx_shape),
            &FlexDevice,
        ));
    assert_eq!(
        flex.into_data().convert::<i32>().to_vec::<i32>().unwrap(),
        want
    );
}

#[test]
fn float_gather_nd_is_raw_exact_resident_and_checked_per_axis() {
    with_device(Config::default(), |d| {
        // K = 2 of 3 axes, a batch of 2x3 tuples, tail [5].
        gather_float::<3, 3, 3>(&d, [3, 4, 5], [2, 3, 2], 1);
        // K = 1: whole rows; ragged 35 x 3.
        gather_float::<2, 3, 3>(&d, [35, 3], [4, 3, 1], 2);
        // K = D: single datums; a 1-D result.
        gather_float::<3, 2, 1>(&d, [3, 4, 5], [7, 3], 3);
        // 1-D data.
        gather_float::<1, 2, 1>(&d, [37], [5, 1], 4);
        // K = 2 of 4 axes, tail [5, 7].
        gather_float::<4, 2, 3>(&d, [2, 3, 5, 7], [3, 2], 5);
        // More tuples than 32: the resident gather's chunking.
        gather_float::<2, 2, 2>(&d, [9, 4], [70, 1], 6);
    });
}

#[test]
fn int_gather_nd_is_exact_for_words_above_2_pow_24() {
    with_device(Config::default(), |d| {
        gather_int::<3, 3, 3>(&d, [3, 4, 5], [2, 3, 2], 11);
        gather_int::<2, 3, 3>(&d, [35, 3], [4, 3, 1], 12);
        gather_int::<3, 2, 1>(&d, [3, 4, 5], [7, 3], 13);
        gather_int::<1, 2, 1>(&d, [37], [5, 1], 14);
        gather_int::<4, 2, 3>(&d, [2, 3, 5, 7], [3, 2], 15);
    });
}

/// Silicon only: ttsim refuses the BF16 pack (PACR `0x105`, divergence log).
#[cfg(feature = "silicon")]
#[test]
fn bf16_gather_nd_copies_raw_datums() {
    use burn::tensor::DType;
    with_device(Config::default(), |d| {
        let shape = [3, 4, 5];
        let words: Vec<u16> = float_words(60, 21)
            .iter()
            .map(|b| (b >> 16) as u16)
            .collect();
        let mut data = TensorData::new(words.clone(), shape);
        data.dtype = DType::BF16;
        let x = Tensor::<TtBackend, 3>::from_data(data, (&d, DType::BF16)).to_device(&d);
        let tup = tuples(&shape, 2, 5, 22, true);
        let i = tt_indices(&d, &tup, [5, 2]);
        let (y, report) = burn_tt::with_report(|| x.gather_nd::<2, 2>(i));
        assert_resident(&report, "float_gather_nd");
        let want = ref_gather(&words, 5, &rows_of(&shape, 2, &tup, false));
        let mut got = y.into_data();
        got.dtype = DType::U16;
        assert_eq!(got.to_vec::<u16>().unwrap(), want);
    });
}

// ---------------------------------------------------------------- scatter_nd

/// Float `Add`: the device folds duplicates in index order with `SFPMAD` add.
fn scatter_float_add<const D: usize, const M: usize, const DV: usize>(
    d: &TtDevice,
    shape: [usize; D],
    idx_shape: [usize; M],
    seed: u64,
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
        assert!(
            has_duplicates(&rows) || n < 4,
            "the gate needs duplicate rows"
        );
        let add = |a: u32, b: u32| numerics::add_bh(a, b);
        let want = ref_scatter(&bits, rest, &rows, &vbits, false, false, add);
        let x = tt_float(d, &bits, shape);
        let v = tt_float(d, &vbits, vshape);
        let i = tt_indices(d, &tup, idx_shape);
        let before = tensor_traffic();
        let (y, report) = burn_tt::with_report(|| {
            x.clone()
                .scatter_nd::<M, DV>(i.clone(), v.clone(), IndexingUpdateOp::Add)
        });
        assert_resident(&report, "float_scatter_nd");
        assert_eq!(tensor_traffic().downloads, before.downloads);
        assert!(is_resident(&y));
        assert_eq!(y.dims(), shape);
        let got = read_bits(y);
        assert_eq!(
            got, want,
            "float scatter_nd Add {shape:?} ordinary={ordinary}"
        );
        if ordinary {
            // Flex's IEEE `+=`, in the same order, on ordinary normals.
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
                IndexingUpdateOp::Add,
            );
            let flex = read_bits(flex);
            let differ = flex.iter().zip(&got).filter(|(a, b)| a != b).count();
            assert_eq!(
                differ, 0,
                "device add and Flex's IEEE add differ on ordinary normals {shape:?}"
            );
        }
    }
}

fn scatter_int_add<const D: usize, const M: usize, const DV: usize>(
    d: &TtDevice,
    shape: [usize; D],
    idx_shape: [usize; M],
    seed: u64,
) {
    let k = idx_shape[M - 1];
    let n: usize = idx_shape[..M - 1].iter().product();
    let rest: usize = shape[k..].iter().product();
    let count: usize = shape.iter().product();
    let vshape: [usize; DV] = out_shape(&shape, &idx_shape).try_into().unwrap();
    let tup = tuples(&shape, k, n, seed, true);
    let rows = rows_of(&shape, k, &tup, false);
    // Wrapping words: the host oracle only (Flex's `+=` panics on overflow).
    let (values, vv) = (int_words(count, seed), int_words(n * rest, seed + 100));
    let want = ref_scatter(&values, rest, &rows, &vv, false, false, i32::wrapping_add);
    let (x, v) = (tt_int(d, &values, shape), tt_int(d, &vv, vshape));
    let i = tt_indices(d, &tup, idx_shape);
    let before = tensor_traffic();
    let (y, report) = burn_tt::with_report(|| {
        x.clone()
            .scatter_nd::<M, DV>(i.clone(), v.clone(), IndexingUpdateOp::Add)
    });
    assert_resident(&report, "int_scatter_nd");
    assert_eq!(tensor_traffic().downloads, before.downloads);
    assert!(y.clone().into_primitive().computed_on_device());
    assert_eq!(y.into_data().to_vec::<i32>().unwrap(), want, "{shape:?}");
    // Flex on sums that cannot overflow.
    let (values, vv) = (small_ints(count, seed), small_ints(n * rest, seed + 100));
    let want = ref_scatter(&values, rest, &rows, &vv, false, false, |a, b| a + b);
    let y = tt_int(d, &values, shape).scatter_nd::<M, DV>(
        tt_indices(d, &tup, idx_shape),
        tt_int(d, &vv, vshape),
        IndexingUpdateOp::Add,
    );
    assert_eq!(y.into_data().to_vec::<i32>().unwrap(), want);
    let flex = Tensor::<Flex, D, Int>::from_data(TensorData::new(values, shape), &FlexDevice)
        .scatter_nd::<M, DV>(
            Tensor::<Flex, M, Int>::from_data(TensorData::new(tup, idx_shape), &FlexDevice),
            Tensor::<Flex, DV, Int>::from_data(TensorData::new(vv, vshape), &FlexDevice),
            IndexingUpdateOp::Add,
        );
    assert_eq!(
        flex.into_data().convert::<i32>().to_vec::<i32>().unwrap(),
        want
    );
}

#[test]
fn scatter_nd_add_folds_duplicates_in_index_order() {
    with_device(Config::default(), |d| {
        scatter_float_add::<3, 3, 3>(&d, [3, 4, 5], [2, 3, 2], 31);
        scatter_float_add::<2, 2, 2>(&d, [35, 3], [6, 1], 32);
        scatter_float_add::<1, 2, 1>(&d, [37], [9, 1], 33);
        scatter_float_add::<4, 2, 2>(&d, [2, 3, 5, 7], [4, 3], 34);
        scatter_int_add::<3, 3, 3>(&d, [3, 4, 5], [2, 3, 2], 35);
        scatter_int_add::<2, 2, 2>(&d, [35, 3], [6, 1], 36);
        scatter_int_add::<1, 2, 1>(&d, [37], [9, 1], 37);
    });
}

/// Order matters: `((0 + 1e20) + 3) + -1e20` is 0 but `((0 + 1e20) + -1e20) + 3`
/// is 3, so a reordered fold (reverse, tree, sorted) shows.
#[test]
fn scatter_nd_add_order_is_the_index_order() {
    with_device(Config::default(), |d| {
        for values in [[3.0f32, 1e20, -1e20, 7.0], [1e20, -1e20, 3.0, 7.0]] {
            let vbits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
            let rows = [0usize, 0, 0, 1];
            let bits = [0.5f32.to_bits(), 0.25f32.to_bits()];
            let add = |a: u32, b: u32| numerics::add_bh(a, b);
            let want = ref_scatter(&bits, 1, &rows, &vbits, false, false, add);
            let reversed = ref_scatter(&bits, 1, &rows, &vbits, false, true, add);
            assert_ne!(
                want, reversed,
                "negative control: reverse order must differ"
            );
            let x = tt_float(&d, &bits, [2]);
            let v = tt_float(&d, &vbits, [4]);
            let i = tt_indices(&d, &[0, 0, 0, 1], [4, 1]);
            let y = x.scatter_nd::<2, 1>(i, v, IndexingUpdateOp::Add);
            assert_eq!(read_bits(y), want, "{values:?}");
            let flex =
                Tensor::<Flex, 1>::from_data(TensorData::new(float_data(&bits), [2]), &FlexDevice)
                    .scatter_nd::<2, 1>(
                        Tensor::<Flex, 2, Int>::from_data(
                            TensorData::new(vec![0, 0, 0, 1], [4, 1]),
                            &FlexDevice,
                        ),
                        Tensor::<Flex, 1>::from_data(
                            TensorData::new(values.to_vec(), [4]),
                            &FlexDevice,
                        ),
                        IndexingUpdateOp::Add,
                    );
            assert_eq!(read_bits(flex), want, "Flex is the oracle: {values:?}");
        }
    });
}

fn scatter_float_assign<const D: usize, const M: usize, const DV: usize>(
    d: &TtDevice,
    shape: [usize; D],
    idx_shape: [usize; M],
    seed: u64,
) {
    let k = idx_shape[M - 1];
    let n: usize = idx_shape[..M - 1].iter().product();
    let rest: usize = shape[k..].iter().product();
    let vshape: [usize; DV] = out_shape(&shape, &idx_shape).try_into().unwrap();
    let bits = float_words(shape.iter().product(), seed);
    let vbits = float_words(n * rest, seed + 100);
    let tup = tuples(&shape, k, n, seed, true);
    let rows = rows_of(&shape, k, &tup, false);
    assert!(has_duplicates(&rows), "the gate needs duplicate rows");
    let want = ref_scatter(&bits, rest, &rows, &vbits, false, false, |_, v| v);
    let first = ref_scatter(&bits, rest, &rows, &vbits, true, false, |_, v| v);
    assert_ne!(
        want, first,
        "negative control: first-writer-wins must differ"
    );
    let x = tt_float(d, &bits, shape);
    let v = tt_float(d, &vbits, vshape);
    let i = tt_indices(d, &tup, idx_shape);
    let before = tensor_traffic();
    let (y, report) = burn_tt::with_report(|| {
        x.clone()
            .scatter_nd::<M, DV>(i.clone(), v.clone(), IndexingUpdateOp::Assign)
    });
    assert_resident(&report, "float_scatter_nd");
    assert_eq!(tensor_traffic().downloads, before.downloads);
    assert!(is_resident(&y));
    assert_eq!(read_bits(y), want, "float Assign {shape:?}");
    let flex = Tensor::<Flex, D>::from_data(TensorData::new(float_data(&bits), shape), &FlexDevice)
        .scatter_nd::<M, DV>(
            Tensor::<Flex, M, Int>::from_data(TensorData::new(tup, idx_shape), &FlexDevice),
            Tensor::<Flex, DV>::from_data(TensorData::new(float_data(&vbits), vshape), &FlexDevice),
            IndexingUpdateOp::Assign,
        );
    assert_eq!(read_bits(flex), want, "Flex is the oracle: {shape:?}");
}

fn scatter_int_assign<const D: usize, const M: usize, const DV: usize>(
    d: &TtDevice,
    shape: [usize; D],
    idx_shape: [usize; M],
    seed: u64,
) {
    let k = idx_shape[M - 1];
    let n: usize = idx_shape[..M - 1].iter().product();
    let rest: usize = shape[k..].iter().product();
    let vshape: [usize; DV] = out_shape(&shape, &idx_shape).try_into().unwrap();
    let values = int_words(shape.iter().product(), seed);
    let vv = int_words(n * rest, seed + 100);
    let tup = tuples(&shape, k, n, seed, true);
    let rows = rows_of(&shape, k, &tup, false);
    assert!(has_duplicates(&rows), "the gate needs duplicate rows");
    let want = ref_scatter(&values, rest, &rows, &vv, false, false, |_, v| v);
    let first = ref_scatter(&values, rest, &rows, &vv, true, false, |_, v| v);
    assert_ne!(
        want, first,
        "negative control: first-writer-wins must differ"
    );
    let (x, v) = (tt_int(d, &values, shape), tt_int(d, &vv, vshape));
    let i = tt_indices(d, &tup, idx_shape);
    let before = tensor_traffic();
    let (y, report) = burn_tt::with_report(|| {
        x.clone()
            .scatter_nd::<M, DV>(i.clone(), v.clone(), IndexingUpdateOp::Assign)
    });
    assert_resident(&report, "int_scatter_nd");
    assert_eq!(tensor_traffic().downloads, before.downloads);
    assert!(y.clone().into_primitive().computed_on_device());
    assert_eq!(y.into_data().to_vec::<i32>().unwrap(), want, "{shape:?}");
    let flex = Tensor::<Flex, D, Int>::from_data(TensorData::new(values, shape), &FlexDevice)
        .scatter_nd::<M, DV>(
            Tensor::<Flex, M, Int>::from_data(TensorData::new(tup, idx_shape), &FlexDevice),
            Tensor::<Flex, DV, Int>::from_data(TensorData::new(vv, vshape), &FlexDevice),
            IndexingUpdateOp::Assign,
        );
    assert_eq!(
        flex.into_data().convert::<i32>().to_vec::<i32>().unwrap(),
        want
    );
}

#[test]
fn scatter_nd_assign_is_last_writer_in_index_order_and_raw_exact() {
    with_device(Config::default(), |d| {
        scatter_float_assign::<3, 3, 3>(&d, [3, 4, 5], [2, 3, 2], 41);
        scatter_float_assign::<2, 2, 2>(&d, [35, 3], [6, 1], 42);
        scatter_float_assign::<1, 2, 1>(&d, [37], [9, 1], 43);
        scatter_float_assign::<4, 2, 2>(&d, [2, 3, 5, 7], [4, 3], 44);
        scatter_int_assign::<3, 3, 3>(&d, [3, 4, 5], [2, 3, 2], 45);
        scatter_int_assign::<2, 2, 2>(&d, [35, 3], [6, 1], 46);
        scatter_int_assign::<1, 2, 1>(&d, [37], [9, 1], 47);
    });
}

fn panic_message(f: impl FnOnce()) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .expect_err("the operation must refuse");
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn unsupported_update_variants_fail_naming_the_op_and_the_variant() {
    with_device(Config::default(), |d| {
        let x = tt_float(&d, &float_words(12, 51), [3, 4]);
        let v = tt_float(&d, &float_words(8, 52), [2, 4]);
        let xi = tt_int(&d, &int_words(12, 53), [3, 4]);
        let vi = tt_int(&d, &int_words(8, 54), [2, 4]);
        let i = tt_indices(&d, &[2, 0], [2, 1]);
        for variant in [
            IndexingUpdateOp::Mul,
            IndexingUpdateOp::Min,
            IndexingUpdateOp::Max,
        ] {
            let m = panic_message(|| {
                let _ = x.clone().scatter_nd::<2, 2>(i.clone(), v.clone(), variant);
            });
            assert!(
                m.contains("float_scatter_nd") && m.contains(&format!("{variant:?}")),
                "{m}"
            );
            let m = panic_message(|| {
                let _ = xi
                    .clone()
                    .scatter_nd::<2, 2>(i.clone(), vi.clone(), variant);
            });
            assert!(
                m.contains("int_scatter_nd") && m.contains(&format!("{variant:?}")),
                "{m}"
            );
        }
        // The supported variants still work after a refusal.
        let y = x.clone().scatter_nd::<2, 2>(i, v, IndexingUpdateOp::Assign);
        assert_eq!(y.dims(), [3, 4]);
    });
}

// ---------------------------------------------------------------- domain

/// Out-of-range coordinates fail with DOMAIN (code 11) on the device. `[0, 4]`
/// into a `[3, 4]` tensor has the flat row 4, inside `[12]`, and `[1, -1]` has the
/// flat row 3: a check on the flat row alone would let both through and read or
/// write the neighbouring row, so these are the control for the per-axis check.
#[test]
fn each_coordinate_is_checked_against_its_own_axis() {
    #[derive(Clone, Copy, Debug)]
    enum Kind {
        GatherFloat,
        GatherInt,
        ScatterAddFloat,
        ScatterAssignInt,
    }
    let bad: [[i32; 2]; 5] = [[0, 4], [1, -1], [3, 0], [0, i32::MAX], [-1, 0]];
    for kind in [
        Kind::GatherFloat,
        Kind::GatherInt,
        Kind::ScatterAddFloat,
        Kind::ScatterAssignInt,
    ] {
        for tuple in bad {
            with_device(Config::default(), |d| {
                let run = |tuple: [i32; 2]| -> Vec<u32> {
                    let i = tt_indices(&d, &[2, 3, tuple[0], tuple[1]], [2, 2]);
                    match kind {
                        Kind::GatherFloat => {
                            let x = tt_float(
                                &d,
                                &(0..12).map(|v| (v as f32).to_bits()).collect::<Vec<_>>(),
                                [3, 4],
                            );
                            read_bits(x.gather_nd::<2, 1>(i))
                        }
                        Kind::GatherInt => {
                            let x = tt_int(&d, &(0..12).collect::<Vec<_>>(), [3, 4]);
                            x.gather_nd::<2, 1>(i)
                                .into_data()
                                .to_vec::<i32>()
                                .unwrap()
                                .into_iter()
                                .map(|v| v as u32)
                                .collect()
                        }
                        Kind::ScatterAddFloat => {
                            let x = tt_float(&d, &[0; 12], [3, 4]);
                            let v = tt_float(&d, &[1f32.to_bits(), 2f32.to_bits()], [2]);
                            read_bits(x.scatter_nd::<2, 1>(i, v, IndexingUpdateOp::Add))
                        }
                        Kind::ScatterAssignInt => {
                            let x = tt_int(&d, &[0; 12], [3, 4]);
                            let v = tt_int(&d, &[1, 2], [2]);
                            x.scatter_nd::<2, 1>(i, v, IndexingUpdateOp::Assign)
                                .into_data()
                                .to_vec::<i32>()
                                .unwrap()
                                .into_iter()
                                .map(|v| v as u32)
                                .collect()
                        }
                    }
                };
                // The in-range pair works in this very device, so the failure is the index.
                let good = run([1, 1]);
                assert!(!good.is_empty());
                let m = panic_message(|| {
                    let _ = run(tuple);
                });
                assert!(
                    m.contains("code 11"),
                    "{kind:?} {tuple:?}: expected a DOMAIN fault, got {m}"
                );
            });
        }
    }
}

// ---------------------------------------------------------------- padding and frees

/// Ragged results (37 x 5, 9 x 5 against 32 x 32 tiles) feed reductions and a
/// matmul; the producers' inputs are dropped before the consumers run (deferred
/// frees). Integer-valued data, so every sum is exact whatever its order.
#[test]
fn ragged_results_feed_reductions_and_matmul_and_inputs_free_late() {
    with_device(Config::default(), |d| {
        let small: Vec<u32> = (0..37 * 5)
            .map(|i| ((i * 7 % 13) as f32 - 6.0).to_bits())
            .collect();
        let tup: Vec<i32> = vec![36, 0, 5, 5, 36, 17, 0, 0, 30];
        let rows: Vec<usize> = tup.iter().map(|&r| r as usize).collect();
        let host_out = ref_gather(&small, 5, &rows);
        let to_f = |bits: &[u32]| bits.iter().map(|&b| f32::from_bits(b)).collect::<Vec<_>>();
        let expect_sum = |v: &[u32], r: usize, c: usize| {
            let f = to_f(v);
            let cols: Vec<f32> = (0..c).map(|j| (0..r).map(|i| f[i * c + j]).sum()).collect();
            let rowsum: Vec<f32> = (0..r).map(|i| (0..c).map(|j| f[i * c + j]).sum()).collect();
            (cols, rowsum)
        };
        let (cols, rowsum) = expect_sum(&host_out, 9, 5);

        let x = tt_float(&d, &small, [37, 5]);
        let i = tt_indices(&d, &tup, [9, 1]);
        let before = tensor_traffic();
        let (g, report) = burn_tt::with_report(|| x.clone().gather_nd::<2, 2>(i.clone()));
        assert_resident(&report, "float_gather_nd");
        drop(i);
        drop(x);
        let ones = Tensor::<TtBackend, 2>::ones([5, 3], &d);
        let ((s0, s1, m), report) = burn_tt::with_report(|| {
            (
                g.clone().sum_dim(0),
                g.clone().sum_dim(1),
                g.clone().matmul(ones.clone()),
            )
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        assert_eq!(s0.into_data().to_vec::<f32>().unwrap(), cols);
        assert_eq!(s1.into_data().to_vec::<f32>().unwrap(), rowsum);
        let mm: Vec<f32> = rowsum.iter().flat_map(|&v| [v, v, v]).collect();
        assert_eq!(m.into_data().to_vec::<f32>().unwrap(), mm);

        // The scatter result, the same way.
        let v = tt_float(&d, &host_out, [9, 5]);
        let z = tt_float(&d, &[0; 37 * 5], [37, 5]);
        let i = tt_indices(&d, &tup, [9, 1]);
        let (y, report) = burn_tt::with_report(|| {
            z.clone()
                .scatter_nd::<2, 2>(i.clone(), v.clone(), IndexingUpdateOp::Add)
        });
        assert_resident(&report, "float_scatter_nd");
        drop((z, v, i));
        let want = ref_scatter(
            &[0.0f32.to_bits(); 37 * 5],
            5,
            &rows,
            &host_out,
            false,
            false,
            numerics::add_bh,
        );
        let (cols, rowsum) = expect_sum(&want, 37, 5);
        assert_eq!(
            y.clone().sum_dim(0).into_data().to_vec::<f32>().unwrap(),
            cols
        );
        assert_eq!(
            y.clone().sum_dim(1).into_data().to_vec::<f32>().unwrap(),
            rowsum
        );
        assert_eq!(read_bits(y), want);

        // Integers: an Int gather into an Int reduction.
        let iv: Vec<i32> = (0..37 * 5).map(|i| (i * 977) ^ 0x1234_5678).collect();
        let xi = tt_int(&d, &iv, [37, 5]);
        let ii = tt_indices(&d, &tup, [9, 1]);
        let gi = xi.gather_nd::<2, 2>(ii);
        let sum = gi.sum_dim(0).into_data().to_vec::<i32>().unwrap();
        let want: Vec<i32> = (0..5)
            .map(|c| {
                rows.iter()
                    .fold(0i32, |a, &r| a.wrapping_add(iv[r * 5 + c]))
            })
            .collect();
        assert_eq!(sum, want);
    });
}

// ---------------------------------------------------------------- traces

/// A trace whose index tuples are produced on the device from the input, so a
/// host-cached index cannot pass: argmax/argmin of each input row choose the data
/// row and column. Run on one tile and on two, with the inputs freed before the
/// replays and the trace released before its tensors are used again.
#[test]
fn changed_index_traces_replay_on_one_and_two_tiles() {
    for tiles in [1usize, 2] {
        with_device(
            Config {
                tiles: Some(TileChoice::Count(tiles)),
                ..Config::default()
            },
            |d| {
                let data_bits: Vec<u32> =
                    (0..20).map(|i| (i as f32 * 10.0 + 0.5).to_bits()).collect();
                let data_ints: Vec<i32> = (0..20).map(|i| i * 1000 - 7).collect();
                let data = tt_float(&d, &data_bits, [4, 5]);
                let ints = tt_int(&d, &data_ints, [4, 5]);
                let zeros = Tensor::<TtBackend, 2>::zeros([4, 5], &d).to_device(&d);
                let bumps = tt_float(&d, &[1f32.to_bits(), 2f32.to_bits(), 4f32.to_bits()], [3]);
                let input = Tensor::<TtBackend, 2>::from_data(
                    TensorData::new(
                        vec![
                            1.0f32, 4.0, 2.0, 0.0, 9.0, 3.0, 8.0, 5.0, 6.0, 7.0, 0.5, 1.5,
                        ],
                        [3, 4],
                    ),
                    &d,
                )
                .mul_scalar(1.0);
                let TensorPrimitive::Float(primitive) = input.clone().into_primitive() else {
                    unreachable!()
                };
                // Host reference for any input values.
                let reference = |x: &[f32]| -> Vec<f32> {
                    let mut tup = Vec::new();
                    for r in 0..3 {
                        let row = &x[r * 4..r * 4 + 4];
                        let mx = (0..4).fold(0, |b, i| if row[i] > row[b] { i } else { b });
                        let mn = (0..4).fold(0, |b, i| if row[i] < row[b] { i } else { b });
                        tup.push((mx, mn));
                    }
                    let gathered: Vec<f32> = tup
                        .iter()
                        .map(|&(a, b)| f32::from_bits(data_bits[a * 5 + b]))
                        .collect();
                    let gathered_i: Vec<f32> = tup
                        .iter()
                        .map(|&(a, b)| data_ints[a * 5 + b] as f32)
                        .collect();
                    let mut added = vec![0.0f32; 20];
                    let mut assigned: Vec<f32> =
                        data_bits.iter().map(|&b| f32::from_bits(b)).collect();
                    for (t, &(a, b)) in tup.iter().enumerate() {
                        added[a * 5 + b] += [1.0, 2.0, 4.0][t];
                        assigned[a * 5 + b] = [1.0, 2.0, 4.0][t];
                    }
                    let mut out = gathered;
                    out.extend(gathered_i);
                    out.extend(added);
                    out.extend(assigned);
                    out
                };
                let before = tensor_traffic();
                let ((trace, first), report) = burn_tt::with_report(|| {
                    burn_tt::Trace::capture(&primitive, || {
                        let row = input.clone().argmax(1);
                        let col = input.clone().argmin(1);
                        let idx = Tensor::cat(vec![row, col], 1);
                        let g = data.clone().gather_nd::<2, 1>(idx.clone());
                        let gi = ints.clone().gather_nd::<2, 1>(idx.clone()).float();
                        let added = zeros
                            .clone()
                            .scatter_nd::<2, 1>(idx.clone(), bumps.clone(), IndexingUpdateOp::Add)
                            .reshape([20]);
                        let assigned = data
                            .clone()
                            .scatter_nd::<2, 1>(idx, bumps.clone(), IndexingUpdateOp::Assign)
                            .reshape([20]);
                        let out = Tensor::cat(vec![g, gi, added, assigned], 0);
                        let TensorPrimitive::Float(out) = out.into_primitive() else {
                            unreachable!()
                        };
                        out
                    })
                    .unwrap()
                });
                assert_native_model(&report);
                assert_eq!(tensor_traffic().uploads, before.uploads, "capture uploaded");
                let x0 = vec![
                    1.0f32, 4.0, 2.0, 0.0, 9.0, 3.0, 8.0, 5.0, 6.0, 7.0, 0.5, 1.5,
                ];
                assert_eq!(first, reference(&x0), "{tiles} tiles: capture");
                // The inputs are gone before the replays: the trace holds what it reads.
                drop(input);
                for x in [
                    vec![
                        8.0f32, 2.0, 1.0, 0.0, 5.0, 7.0, 3.0, 9.0, 0.5, 4.0, 6.0, 2.5,
                    ],
                    vec![
                        -1.0, -2.0, 3.0, 4.0, 6.0, 5.0, 0.0, 1.0, 2.0, 2.5, -9.0, 1.5,
                    ],
                    vec![0.0, 0.5, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 9.0, 8.0, 7.0, 6.0],
                ] {
                    let got = trace.run(x.clone()).unwrap();
                    assert_eq!(got, reference(&x), "{tiles} tiles: replay {x:?}");
                }
                drop(trace);
                // Released: the tensors it held are still intact.
                assert_eq!(
                    data.sum().into_data().to_vec::<f32>().unwrap(),
                    vec![(0..20).map(|i| i as f32 * 10.0 + 0.5).sum::<f32>()]
                );
                assert_eq!(ints.into_data().to_vec::<i32>().unwrap(), data_ints);
            },
        );
    }
}
