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
use tt_tests::data::lcg_word;

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
            (0, Elem::I32) => I32_SPECIALS[lcg_word(&mut s) as usize % I32_SPECIALS.len()] as u32,
            (0, _) => F32_SPECIALS[lcg_word(&mut s) as usize % F32_SPECIALS.len()],
            (1 | 2, Elem::I32) => ((lcg_word(&mut s) % 7) as i32 - 3) as u32,
            (1 | 2, _) => ((lcg_word(&mut s) % 7) as f32 - 3.0).to_bits(),
            (_, _) => lcg_word(&mut s),
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
            let bits = words(Elem::F32, 29, lcg_word(&mut seed) as u64);
            float_case(&d, [29], 0, &bits, descending);
            let bits = words(Elem::F32, 7 * 20, lcg_word(&mut seed) as u64);
            float_case(&d, [7, 20], 0, &bits, descending);
            float_case(&d, [7, 20], 1, &bits, descending);
            let bits = words(Elem::F32, 35 * 9, lcg_word(&mut seed) as u64);
            float_case(&d, [35, 9], 0, &bits, descending);
            float_case(&d, [35, 9], 1, &bits, descending);
            let bits = words(Elem::F32, 3 * 4 * 19, lcg_word(&mut seed) as u64);
            float_case(&d, [3, 4, 19], 0, &bits, descending);
            float_case(&d, [3, 4, 19], 1, &bits, descending);
            float_case(&d, [3, 4, 19], 2, &bits, descending);
            let bits = words(Elem::F32, 2 * 64 * 3, lcg_word(&mut seed) as u64);
            float_case(&d, [2, 64, 3], 1, &bits, descending);
            let bits = words(Elem::F32, 2 * 32 * 3, lcg_word(&mut seed) as u64);
            float_case(&d, [2, 32, 3], 1, &bits, descending);
            // Several tiles of axis: a padded power of two of four, and 1024.
            let bits = words(Elem::F32, 2 * 100 * 3, lcg_word(&mut seed) as u64);
            float_case(&d, [2, 100, 3], 1, &bits, descending);
            let bits = words(Elem::F32, 130, lcg_word(&mut seed) as u64);
            float_case(&d, [130], 0, &bits, descending);
            let bits = words(Elem::F32, 1024 * 2, lcg_word(&mut seed) as u64);
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
            let bits = words(Elem::I32, 31, lcg_word(&mut seed) as u64);
            int_case(&d, [31], 0, &bits, descending);
            let bits = words(Elem::I32, 6 * 21, lcg_word(&mut seed) as u64);
            int_case(&d, [6, 21], 0, &bits, descending);
            int_case(&d, [6, 21], 1, &bits, descending);
            let bits = words(Elem::I32, 40 * 5, lcg_word(&mut seed) as u64);
            int_case(&d, [40, 5], 0, &bits, descending);
            int_case(&d, [40, 5], 1, &bits, descending);
            let bits = words(Elem::I32, 3 * 5 * 17, lcg_word(&mut seed) as u64);
            int_case(&d, [3, 5, 17], 0, &bits, descending);
            int_case(&d, [3, 5, 17], 1, &bits, descending);
            int_case(&d, [3, 5, 17], 2, &bits, descending);
            // Several tiles of axis.
            let bits = words(Elem::I32, 64 * 2, lcg_word(&mut seed) as u64);
            int_case(&d, [64, 2], 0, &bits, descending);
            let bits = words(Elem::I32, 3 * 77, lcg_word(&mut seed) as u64);
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
            order.swap(i, lcg_word(&mut seed) as usize % (i + 1));
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

mod residency {
    //! The sort family is resident and native: nothing is downloaded, nothing is
    //! computed or staged on the host, results are device-computed tensors; `topk`
    //! composes natively from the sort; ragged producers feed sorts that feed
    //! reductions; what the device cannot sort is refused by name.
    use burn::tensor::{Int, Tensor, TensorData, TensorPrimitive};
    use burn_flex::{Flex, FlexDevice};
    use burn_tt::{tensor_traffic, TtBackend};
    use tt_kernels::sfpu::sort::reference;
    use tt_kernels::tensor::Elem;
    use tt_tests::burn_device::{assert_native_model, with_device, Config};

    fn float_bits(t: Vec<f32>) -> Vec<u32> {
        t.into_iter().map(f32::to_bits).collect()
    }

    fn lane_oracle(
        elem: Elem,
        rows: usize,
        cols: usize,
        bits: &[u32],
        dim: usize,
        descending: bool,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut values = bits.to_vec();
        let mut indices = vec![0u32; bits.len()];
        let (n, lanes) = if dim == 1 { (cols, rows) } else { (rows, cols) };
        for l in 0..lanes {
            let at = |e: usize| if dim == 1 { l * cols + e } else { e * cols + l };
            let lane: Vec<u32> = (0..n).map(|e| bits[at(e)]).collect();
            let (v, i) = reference(elem, &lane, descending);
            for e in 0..n {
                values[at(e)] = v[e];
                indices[at(e)] = i[e];
            }
        }
        (values, indices)
    }

    const SPECIAL: [f32; 10] = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        1.0,
        -1.0,
        1.0e-40,
        -1.0e-40,
    ];

    fn pattern(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| {
                if i % 3 == 0 {
                    SPECIAL[(i / 3 + i / 11) % SPECIAL.len()]
                } else {
                    ((i * 7 % 13) as f32 - 6.0) * 0.5
                }
            })
            .collect()
    }

    #[test]
    fn sorts_of_resident_tensors_download_and_compute_nothing_on_the_host() {
        with_device(Config::default(), |d| {
            let (rows, cols) = (11, 19);
            let values = pattern(rows * cols);
            let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
            let make = || {
                // A device-computed input, not an upload.
                Tensor::<TtBackend, 2>::from_data(TensorData::new(values.clone(), [rows, cols]), &d)
                    .add_scalar(0.0)
            };
            let resident = |t: &burn_tt::TtTensor| assert!(t.computed_on_device());
            let input = make();
            let TensorPrimitive::Float(p) = input.clone().into_primitive() else {
                unreachable!()
            };
            resident(&p);
            for dim in [0, 1] {
                // The input's -0 is a +0 after `add_scalar(0.0)`, as the device's
                // float add has it: the oracle reads what the device holds.
                let held = bits_after_add(&input);
                let before = tensor_traffic();
                let ((v, i), report) =
                    burn_tt::with_report(|| input.clone().sort_with_indices(dim));
                assert_eq!(tensor_traffic().downloads, before.downloads, "dim {dim}");
                assert_native_model(&report);
                let op = report.op("float_sort_with_indices").unwrap();
                assert_eq!((op.downloads, op.staged, op.on_host), (0, 0, 0));
                assert!(op.on_device > 0, "the sort ran on the device");
                let TensorPrimitive::Float(vp) = v.clone().into_primitive() else {
                    unreachable!()
                };
                resident(&vp);
                resident(&i.clone().into_primitive());
                let (want_v, want_i) = lane_oracle(Elem::F32, rows, cols, &held, dim, false);
                assert_eq!(float_bits(v.into_data().to_vec::<f32>().unwrap()), want_v);
                assert_eq!(
                    i.into_data()
                        .to_vec::<i32>()
                        .unwrap()
                        .into_iter()
                        .map(|x| x as u32)
                        .collect::<Vec<_>>(),
                    want_i
                );
                let (a, report) = burn_tt::with_report(|| input.clone().argsort_descending(dim));
                assert_native_model(&report);
                resident(&a.clone().into_primitive());
                let (_, want_i) = lane_oracle(Elem::F32, rows, cols, &held, dim, true);
                assert_eq!(
                    a.into_data()
                        .to_vec::<i32>()
                        .unwrap()
                        .into_iter()
                        .map(|x| x as u32)
                        .collect::<Vec<_>>(),
                    want_i
                );
            }
            let _ = bits;
        });
    }

    /// What the device holds after `add_scalar(0.0)`: read back, once, outside
    /// any measured region.
    fn bits_after_add(t: &Tensor<TtBackend, 2>) -> Vec<u32> {
        float_bits(t.clone().into_data().to_vec::<f32>().unwrap())
    }

    #[test]
    fn topk_composes_natively_from_the_sort() {
        with_device(Config::default(), |d| {
            let (rows, cols) = (6, 21);
            let values = pattern(rows * cols);
            let x = Tensor::<TtBackend, 2>::from_data(TensorData::new(values, [rows, cols]), &d)
                .add_scalar(0.0);
            let held = bits_after_add(&x);
            for dim in [0, 1] {
                let n = if dim == 1 { cols } else { rows };
                let (want_v, want_i) = lane_oracle(Elem::F32, rows, cols, &held, dim, true);
                for k in [1, 3, n - 1] {
                    // Expected: the first k of each lane of the descending order.
                    let take = |all: &[u32]| -> Vec<u32> {
                        let (lanes, stride_e, stride_l) = if dim == 1 {
                            (rows, 1, cols)
                        } else {
                            (cols, cols, 1)
                        };
                        let mut out = Vec::new();
                        if dim == 1 {
                            for l in 0..lanes {
                                for e in 0..k {
                                    out.push(all[l * stride_l + e * stride_e]);
                                }
                            }
                        } else {
                            for e in 0..k {
                                for l in 0..lanes {
                                    out.push(all[l * stride_l + e * stride_e]);
                                }
                            }
                        }
                        out
                    };
                    let before = tensor_traffic();
                    let ((top, idx), report) =
                        burn_tt::with_report(|| x.clone().topk_with_indices(k, dim));
                    assert_eq!(
                        tensor_traffic().downloads,
                        before.downloads,
                        "k={k} dim={dim}"
                    );
                    assert_native_model(&report);
                    assert_eq!(
                        float_bits(top.into_data().to_vec::<f32>().unwrap()),
                        take(&want_v),
                        "topk values k={k} dim={dim}"
                    );
                    assert_eq!(
                        idx.into_data()
                            .to_vec::<i32>()
                            .unwrap()
                            .into_iter()
                            .map(|x| x as u32)
                            .collect::<Vec<_>>(),
                        take(&want_i),
                        "topk indices k={k} dim={dim}"
                    );
                    let (arg, report) = burn_tt::with_report(|| x.clone().argtopk(k, dim));
                    assert_native_model(&report);
                    assert!(report.op("float_argtopk").unwrap().on_device > 0);
                    assert_eq!(
                        arg.into_data()
                            .to_vec::<i32>()
                            .unwrap()
                            .into_iter()
                            .map(|x| x as u32)
                            .collect::<Vec<_>>(),
                        take(&want_i),
                        "argtopk k={k} dim={dim}"
                    );
                    let (plain, report) = burn_tt::with_report(|| x.clone().topk(k, dim));
                    assert_native_model(&report);
                    assert_eq!(
                        float_bits(plain.into_data().to_vec::<f32>().unwrap()),
                        take(&want_v),
                        "topk k={k} dim={dim}"
                    );
                }
            }
            // Integer top-k.
            let ints: Vec<i32> = (0..rows * cols)
                .map(|i| [i32::MIN, 5, -7, i32::MAX, 5, 0][(i * 5 + i / 7) % 6])
                .collect();
            let xi = Tensor::<TtBackend, 2, Int>::from_data(
                TensorData::new(ints.clone(), [rows, cols]),
                &d,
            )
            .add_scalar(0);
            let bits: Vec<u32> = ints.iter().map(|&x| x as u32).collect();
            let (_, want_i) = lane_oracle(Elem::I32, rows, cols, &bits, 1, true);
            let (arg, report) = burn_tt::with_report(|| xi.clone().argtopk(4, 1));
            assert_native_model(&report);
            let want: Vec<i32> = (0..rows)
                .flat_map(|l| (0..4).map(move |e| (l, e)))
                .map(|(l, e)| want_i[l * cols + e] as i32)
                .collect();
            assert_eq!(arg.into_data().to_vec::<i32>().unwrap(), want);
        });
    }

    /// A ragged producer (a matmul result with a ragged edge) feeds a sort, whose
    /// output feeds reductions and a matmul: padding is declared, never leaked.
    /// `[35, 9]` ragged in both directions, through sorts of either axis.
    fn chain<B: burn::tensor::backend::Backend>(
        a: Tensor<B, 2>,
        b: Tensor<B, 2>,
        w: Tensor<B, 2>,
    ) -> [Tensor<B, 2>; 5] {
        let p = a.matmul(b);
        let s = p.clone().sort(1);
        let s0 = p.sort_descending(0);
        [
            s.clone().sum_dim(0),
            s.clone().max_dim(0),
            s0.clone().sum_dim(1),
            s.matmul(w.clone()),
            s0.matmul(w),
        ]
    }

    #[test]
    fn ragged_producers_feed_sorts_that_feed_reductions_and_matmuls() {
        with_device(Config::default(), |d| {
            let (m, k, n) = (35, 5, 9);
            let a: Vec<f32> = (0..m * k).map(|i| ((i * 3) % 7) as f32 - 3.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i * 5) % 11) as f32 - 5.0).collect();
            let w: Vec<f32> = (0..n * 4).map(|i| (i % 3) as f32 - 1.0).collect();
            let flex = |v: &Vec<f32>, shape: [usize; 2]| {
                Tensor::<Flex, 2>::from_data(TensorData::new(v.clone(), shape), &FlexDevice)
            };
            let ours = |v: &Vec<f32>, shape: [usize; 2]| {
                Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), shape), &d)
            };
            let want = chain(flex(&a, [m, k]), flex(&b, [k, n]), flex(&w, [n, 4]));
            let (got, report) = burn_tt::with_report(|| {
                chain(ours(&a, [m, k]), ours(&b, [k, n]), ours(&w, [n, 4]))
            });
            assert_native_model(&report);
            fn data<B: burn::tensor::backend::Backend>(t: Tensor<B, 2>) -> Vec<f32> {
                t.into_data().to_vec::<f32>().unwrap()
            }
            for (i, (g, w)) in got.into_iter().zip(want).enumerate() {
                assert_eq!(float_bits(data(g)), float_bits(data(w)), "output {i}");
            }
        });
    }

    #[test]
    fn what_the_device_cannot_sort_is_refused_by_name_and_metadata() {
        with_device(Config::default(), |d| {
            let refused = |f: Box<dyn FnOnce()>| -> String {
                let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
                    .expect_err("must refuse");
                e.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap()
            };
            let long = Tensor::<TtBackend, 2>::from_data(
                TensorData::new(vec![1.0; 2 * 1025], [2, 1025]),
                &d,
            );
            let l = long.clone();
            let m = refused(Box::new(move || {
                let _ = l.sort(1);
            }));
            assert!(
                m.contains("float_sort")
                    && m.contains("1025")
                    && m.contains("1024")
                    && m.contains("[2, 1025]"),
                "{m}"
            );
            let l = long.clone();
            let m = refused(Box::new(move || {
                let _ = l.argsort(1);
            }));
            assert!(m.contains("float_argsort") && m.contains("1025"), "{m}");
            let l = long.clone();
            let m = refused(Box::new(move || {
                let _ = l.topk(2, 1);
            }));
            assert!(m.contains("float_topk") && m.contains("k=2"), "{m}");
            let l = long;
            let m = refused(Box::new(move || {
                let _ = l.sort_with_indices(1);
            }));
            assert!(
                m.contains("float_sort_with_indices") && m.contains("1025"),
                "{m}"
            );
            // Index dtypes other than the device's I32.
            let small = Tensor::<TtBackend, 1>::from_data([3.0, 1.0, 2.0], &d);
            let m = refused(Box::new(move || {
                let p = match small.into_primitive() {
                    TensorPrimitive::Float(p) => p,
                    _ => unreachable!(),
                };
                let _ = <TtBackend as burn::tensor::ops::FloatTensorOps<TtBackend>>::float_argsort(
                    p,
                    0,
                    false,
                    burn::tensor::IntDType::I64,
                );
            }));
            assert!(m.contains("float_argsort") && m.contains("I64"), "{m}");
            let ints =
                Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(vec![1; 2000], [2000]), &d);
            let m = refused(Box::new(move || {
                let _ = ints.argsort(0);
            }));
            assert!(m.contains("int_argsort") && m.contains("2000"), "{m}");
        });
    }

    /// Autodiff's sort routes the gradient by the indices the device sort made:
    /// each weight returns to the element it was sorted from, ties by original
    /// index, all of it native.
    #[test]
    fn sort_gradients_follow_the_stable_indices_natively() {
        use burn::backend::Autodiff;
        with_device(Config::default(), |d| {
            let (rows, cols) = (3, 20);
            let values: Vec<f32> = (0..rows * cols)
                .map(|i| ((i * 5 + i / 4) % 7) as f32 - 3.0)
                .collect();
            let weights: Vec<f32> = (0..rows * cols).map(|i| (i % 9) as f32 + 1.0).collect();
            let (grad, report) = burn_tt::with_report(|| {
                let x = Tensor::<Autodiff<TtBackend>, 2>::from_data(
                    TensorData::new(values.clone(), [rows, cols]),
                    &d,
                )
                .require_grad();
                let w = Tensor::<Autodiff<TtBackend>, 2>::from_data(
                    TensorData::new(weights.clone(), [rows, cols]),
                    &d,
                );
                let loss = (x.clone().sort_descending(1) * w).sum();
                x.grad(&loss.backward()).unwrap()
            });
            assert_native_model(&report);
            let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
            let (_, idx) = lane_oracle(Elem::F32, rows, cols, &bits, 1, true);
            let mut want = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for e in 0..cols {
                    want[r * cols + idx[r * cols + e] as usize] += weights[r * cols + e];
                }
            }
            assert_eq!(
                float_bits(grad.into_data().to_vec::<f32>().unwrap()),
                float_bits(want)
            );
        });
    }
}

mod traces {
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

    fn host(
        v: &[f32],
        rows: usize,
        cols: usize,
        dim: usize,
        descending: bool,
    ) -> (Vec<u32>, Vec<u32>) {
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
            let x: T2 =
                Tensor::from_data(TensorData::new(inputs(0, rows * cols), [rows, cols]), &d);
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
}

mod long_axes {
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
            let same =
                Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(vec![7; 300], [300]), &d);
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
}
