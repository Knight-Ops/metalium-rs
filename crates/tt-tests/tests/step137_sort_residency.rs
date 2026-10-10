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
            let ((v, i), report) = burn_tt::with_report(|| input.clone().sort_with_indices(dim));
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
        let xi =
            Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(ints.clone(), [rows, cols]), &d)
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
        let (got, report) =
            burn_tt::with_report(|| chain(ours(&a, [m, k]), ours(&b, [k, n]), ours(&w, [n, 4])));
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
            let e =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("must refuse");
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap()
        };
        let long =
            Tensor::<TtBackend, 2>::from_data(TensorData::new(vec![1.0; 2 * 1025], [2, 1025]), &d);
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
