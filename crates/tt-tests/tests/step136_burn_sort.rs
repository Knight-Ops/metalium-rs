//! Burn's sort family on the device: values and indices exact under the stated
//! tie contract (stable, ties by ascending original index, `total_cmp` for F32),
//! every axis of rank 1..=3, both directions, resident, with no host fallback.
//!
//! The oracle is `tt_kernels::sfpu::sort::reference` -- a host stable sort on
//! `(key, original index)` that shares no code with the kernel -- applied lane
//! by lane here. Burn's own sorts are unstable, so burn-flex is the oracle only
//! where its answer is determined: every sorted *value* (equal keys are equal
//! bits), and the indices of inputs without ties.
use burn::tensor::{Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_kernels::sfpu::sort::reference;
use tt_kernels::tensor::Elem;
use tt_tests::burn_device::{with_device, Config};

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}

const F32_SPECIALS: [u32; 22] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x3f80_0000,
    0xbf80_0000,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0000,
    0x7f80_0001,
    0xff80_0001,
    0x7fff_ffff,
    0xffff_ffff,
    0x7fc1_2345,
    0xffc5_4321,
    0x4000_0000,
    0xc000_0000,
];

const I32_SPECIALS: [i32; 10] = [
    0,
    1,
    -1,
    i32::MIN,
    i32::MAX,
    i32::MIN + 1,
    i32::MAX - 1,
    0x0000_ffff,
    -0x0001_0000,
    16_777_217,
];

/// Words with duplicates and specials throughout.
fn words(elem: Elem, count: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..count)
        .map(|i| match (i % 5, elem) {
            (0, Elem::I32) => I32_SPECIALS[lcg(&mut s) as usize % I32_SPECIALS.len()] as u32,
            (0, _) => F32_SPECIALS[lcg(&mut s) as usize % F32_SPECIALS.len()],
            (1 | 2, Elem::I32) => ((lcg(&mut s) % 7) as i32 - 3) as u32,
            (1 | 2, _) => ((lcg(&mut s) % 7) as f32 - 3.0).to_bits(),
            (_, _) => lcg(&mut s),
        })
        .collect()
}

/// `reference` applied to every lane along `dim` of a row-major tensor:
/// the sorted words and the original indices.
fn lanewise(
    elem: Elem,
    shape: &[usize],
    dim: usize,
    bits: &[u32],
    descending: bool,
) -> (Vec<u32>, Vec<u32>) {
    let n = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    let mut values = bits.to_vec();
    let mut indices = vec![0u32; bits.len()];
    for o in 0..outer {
        for i in 0..inner {
            let at = |e: usize| (o * n + e) * inner + i;
            let lane: Vec<u32> = (0..n).map(|e| bits[at(e)]).collect();
            let (v, ix) = reference(elem, &lane, descending);
            for e in 0..n {
                values[at(e)] = v[e];
                indices[at(e)] = ix[e];
            }
        }
    }
    (values, indices)
}

fn f32_tensor<const D: usize>(
    bits: &[u32],
    shape: [usize; D],
    d: &burn_tt::TtDevice,
) -> Tensor<TtBackend, D> {
    let v: Vec<f32> = bits.iter().copied().map(f32::from_bits).collect();
    Tensor::from_data(TensorData::new(v, shape), d)
}

fn bits_of<const D: usize>(t: Tensor<TtBackend, D>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn ints_of<const D: usize>(t: Tensor<TtBackend, D, Int>) -> Vec<u32> {
    t.into_data()
        .to_vec::<i32>()
        .unwrap()
        .into_iter()
        .map(|x| x as u32)
        .collect()
}

fn float_case<const D: usize>(
    d: &burn_tt::TtDevice,
    shape: [usize; D],
    dim: usize,
    bits: &[u32],
    descending: bool,
) {
    let (want_v, want_i) = lanewise(Elem::F32, &shape, dim, bits, descending);
    let make = || f32_tensor(bits, shape, d);
    let tag = format!("{shape:?} dim {dim} descending {descending}");
    let sorted = if descending {
        make().sort_descending(dim)
    } else {
        make().sort(dim)
    };
    assert_eq!(bits_of(sorted), want_v, "sort {tag}");
    // Burn-flex: every sorted value is determined.
    let flex = Tensor::<Flex, D>::from_data(
        TensorData::new(
            bits.iter().copied().map(f32::from_bits).collect::<Vec<_>>(),
            shape,
        ),
        &FlexDevice,
    );
    let flex_sorted = if descending {
        flex.sort_descending(dim)
    } else {
        flex.sort(dim)
    };
    let flex_bits: Vec<u32> = flex_sorted
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect();
    assert_eq!(flex_bits, want_v, "burn-flex values {tag}");
    {
        let (v, i) = if descending {
            make().sort_descending_with_indices(dim)
        } else {
            make().sort_with_indices(dim)
        };
        assert_eq!(bits_of(v), want_v, "sort_with_indices values {tag}");
        assert_eq!(ints_of(i), want_i, "sort_with_indices indices {tag}");
        let a = if descending {
            make().argsort_descending(dim)
        } else {
            make().argsort(dim)
        };
        assert_eq!(ints_of(a), want_i, "argsort {tag}");
    }
}

#[test]
fn float_sorts_are_stable_total_order_sorts_on_every_axis() {
    with_device(Config::default(), |d| {
        let mut seed = 136;
        // Rank 1, 2, 3; axes up to the values-only bound of 64 (with
        // indices, 32).
        for descending in [false, true] {
            let bits = words(Elem::F32, 29, lcg(&mut seed) as u64);
            float_case(&d, [29], 0, &bits, descending);
            let bits = words(Elem::F32, 7 * 20, lcg(&mut seed) as u64);
            float_case(&d, [7, 20], 0, &bits, descending);
            float_case(&d, [7, 20], 1, &bits, descending);
            let bits = words(Elem::F32, 35 * 9, lcg(&mut seed) as u64);
            float_case(&d, [35, 9], 0, &bits, descending);
            float_case(&d, [35, 9], 1, &bits, descending);
            let bits = words(Elem::F32, 3 * 4 * 19, lcg(&mut seed) as u64);
            float_case(&d, [3, 4, 19], 0, &bits, descending);
            float_case(&d, [3, 4, 19], 1, &bits, descending);
            float_case(&d, [3, 4, 19], 2, &bits, descending);
            let bits = words(Elem::F32, 2 * 64 * 3, lcg(&mut seed) as u64);
            float_case(&d, [2, 64, 3], 1, &bits, descending);
            let bits = words(Elem::F32, 2 * 32 * 3, lcg(&mut seed) as u64);
            float_case(&d, [2, 32, 3], 1, &bits, descending);
            // Several tiles of axis: a padded power of two of four, and 1024.
            let bits = words(Elem::F32, 2 * 100 * 3, lcg(&mut seed) as u64);
            float_case(&d, [2, 100, 3], 1, &bits, descending);
            let bits = words(Elem::F32, 130, lcg(&mut seed) as u64);
            float_case(&d, [130], 0, &bits, descending);
            let bits = words(Elem::F32, 1024 * 2, lcg(&mut seed) as u64);
            float_case(&d, [1024, 2], 0, &bits, descending);
        }
        // A single element and a unit axis.
        let bits = words(Elem::F32, 5, 9);
        float_case(&d, [5, 1], 1, &bits, false);
        float_case(&d, [1], 0, &bits[..1], true);
    });
}

fn i32_tensor<const D: usize>(
    bits: &[u32],
    shape: [usize; D],
    d: &burn_tt::TtDevice,
) -> Tensor<TtBackend, D, Int> {
    let v: Vec<i32> = bits.iter().map(|&b| b as i32).collect();
    Tensor::from_data(TensorData::new(v, shape), d)
}

fn int_case<const D: usize>(
    d: &burn_tt::TtDevice,
    shape: [usize; D],
    dim: usize,
    bits: &[u32],
    descending: bool,
) {
    let (want_v, want_i) = lanewise(Elem::I32, &shape, dim, bits, descending);
    let make = || i32_tensor(bits, shape, d);
    let tag = format!("{shape:?} dim {dim} descending {descending}");
    let sorted = if descending {
        make().sort_descending(dim)
    } else {
        make().sort(dim)
    };
    assert_eq!(ints_of(sorted), want_v, "int sort {tag}");
    let flex = Tensor::<Flex, D, Int>::from_data(
        TensorData::new(bits.iter().map(|&b| b as i32).collect::<Vec<_>>(), shape),
        &FlexDevice,
    );
    let flex_sorted = if descending {
        flex.sort_descending(dim)
    } else {
        flex.sort(dim)
    };
    let flex_bits: Vec<u32> = flex_sorted
        .into_data()
        .to_vec::<i32>()
        .unwrap()
        .into_iter()
        .map(|x| x as u32)
        .collect();
    assert_eq!(flex_bits, want_v, "burn-flex int values {tag}");
    {
        let (v, i) = if descending {
            make().sort_descending_with_indices(dim)
        } else {
            make().sort_with_indices(dim)
        };
        assert_eq!(ints_of(v), want_v, "int sort_with_indices values {tag}");
        assert_eq!(ints_of(i), want_i, "int sort_with_indices indices {tag}");
        let a = if descending {
            make().argsort_descending(dim)
        } else {
            make().argsort(dim)
        };
        assert_eq!(ints_of(a), want_i, "int argsort {tag}");
    }
}

#[test]
fn int_sorts_are_stable_two_complement_sorts_on_every_axis() {
    with_device(Config::default(), |d| {
        let mut seed = 137;
        for descending in [false, true] {
            let bits = words(Elem::I32, 31, lcg(&mut seed) as u64);
            int_case(&d, [31], 0, &bits, descending);
            let bits = words(Elem::I32, 6 * 21, lcg(&mut seed) as u64);
            int_case(&d, [6, 21], 0, &bits, descending);
            int_case(&d, [6, 21], 1, &bits, descending);
            let bits = words(Elem::I32, 40 * 5, lcg(&mut seed) as u64);
            int_case(&d, [40, 5], 0, &bits, descending);
            int_case(&d, [40, 5], 1, &bits, descending);
            let bits = words(Elem::I32, 3 * 5 * 17, lcg(&mut seed) as u64);
            int_case(&d, [3, 5, 17], 0, &bits, descending);
            int_case(&d, [3, 5, 17], 1, &bits, descending);
            int_case(&d, [3, 5, 17], 2, &bits, descending);
            // Several tiles of axis.
            let bits = words(Elem::I32, 64 * 2, lcg(&mut seed) as u64);
            int_case(&d, [64, 2], 0, &bits, descending);
            let bits = words(Elem::I32, 3 * 77, lcg(&mut seed) as u64);
            int_case(&d, [3, 77], 1, &bits, descending);
        }
    });
}

/// Inputs with no ties: Burn's answer is determined, indices included.
#[test]
fn without_ties_burn_flex_agrees_on_indices_exactly() {
    with_device(Config::default(), |d| {
        let mut seed = 138;
        let shape = [5, 24];
        let mut order: Vec<u32> = (0..120).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, lcg(&mut seed) as usize % (i + 1));
        }
        // Distinct values: signed, with both zeros' neighbours and a NaN.
        let vals: Vec<f32> = order
            .iter()
            .map(|&k| match k {
                0 => f32::NAN,
                1 => f32::NEG_INFINITY,
                2 => f32::INFINITY,
                k => (k as f32 - 60.0) * 0.37,
            })
            .collect();
        let flex = Tensor::<Flex, 2>::from_data(TensorData::new(vals.clone(), shape), &FlexDevice);
        let ours = Tensor::<TtBackend, 2>::from_data(TensorData::new(vals, shape), &d);
        for dim in [0, 1] {
            for descending in [false, true] {
                let (fv, fi) = if descending {
                    flex.clone().sort_descending_with_indices(dim)
                } else {
                    flex.clone().sort_with_indices(dim)
                };
                let (ov, oi) = if descending {
                    ours.clone().sort_descending_with_indices(dim)
                } else {
                    ours.clone().sort_with_indices(dim)
                };
                let bits = |t: Vec<f32>| t.into_iter().map(f32::to_bits).collect::<Vec<_>>();
                assert_eq!(
                    bits(ov.into_data().to_vec::<f32>().unwrap()),
                    bits(fv.into_data().to_vec::<f32>().unwrap())
                );
                assert_eq!(
                    oi.into_data().to_vec::<i32>().unwrap(),
                    fi.into_data().to_vec::<i32>().unwrap(),
                    "dim {dim} descending {descending}"
                );
            }
        }
    });
}
