//! Lane T1 gate (`hardware-coverage-closeout.md`): `int/bool_{permute,flip,unfold}`.
//!
//! Integer and boolean views are the float views' dtype-generic bodies: strided views of the
//! device buffer and native copies that move raw words. Held here, against an independent host
//! index oracle (explicit multi-index arithmetic, not any Burn backend) and exact bit equality:
//!
//! - every word survives -- `i32::MIN`, `i32::MAX`, `2^24 + 1`, FP32 NaN and denormal patterns --
//!   at ragged shapes (`[3, 33, 65]`) whose permuted, flipped and unfolded results are not tile
//!   multiples;
//! - nothing is downloaded and every result is device-resident (a view or a device copy);
//! - the ragged results feed `int_sum_dim` and a float matmul, which read through the result's
//!   declared padding, so a wrong padding claim shows as a wrong sum;
//! - a view outlives its dropped parent (deferred frees), and bools stay canonical `0`/`1`.
//!
//! Negative control: a wrong permutation (`int_permute` given the reversed axes) must fail
//! the permute gate.

use burn::tensor::{Bool, Int, Tensor, TensorData};
use burn_tt::{tensor_traffic, TtBackend, TtDevice};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// Integers whose bit patterns an `f32` path would damage, then pseudo-random words.
fn ints(n: usize, seed: u32) -> Vec<i32> {
    let specials = [
        i32::MIN,
        i32::MAX,
        -1,
        0,
        1,
        16_777_217,
        -16_777_217,
        0x7fc0_0001,
        0x8000_0001u32 as i32,
        0x007f_ffff,
        0x7f80_0000,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if i % 4 == 0 {
                specials[(i / 4 + seed as usize) % specials.len()]
            } else {
                s as i32
            }
        })
        .collect()
}

fn bools(n: usize, seed: u32) -> Vec<bool> {
    ints(n, seed)
        .iter()
        .map(|x| x.count_ones() & 1 == 1)
        .collect()
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// `out[o] = data[source(o)]` over every multi-index `o` of `out_shape`.
fn gather<T: Copy>(
    data: &[T],
    shape: &[usize],
    out_shape: &[usize],
    source: impl Fn(&[usize]) -> Vec<usize>,
) -> Vec<T> {
    let (n, st) = (out_shape.iter().product::<usize>(), strides(shape));
    (0..n)
        .map(|mut flat| {
            let mut o = vec![0; out_shape.len()];
            for d in (0..out_shape.len()).rev() {
                o[d] = flat % out_shape[d];
                flat /= out_shape[d];
            }
            data[source(&o)
                .iter()
                .zip(&st)
                .map(|(i, s)| i * s)
                .sum::<usize>()]
        })
        .collect()
}

fn permute_host<T: Copy>(data: &[T], shape: &[usize], axes: &[usize]) -> (Vec<T>, Vec<usize>) {
    let out: Vec<usize> = axes.iter().map(|&a| shape[a]).collect();
    let v = gather(data, shape, &out, |o| {
        let mut s = vec![0; shape.len()];
        for (i, &a) in axes.iter().enumerate() {
            s[a] = o[i];
        }
        s
    });
    (v, out)
}

fn flip_host<T: Copy>(data: &[T], shape: &[usize], axes: &[usize]) -> Vec<T> {
    gather(data, shape, shape, |o| {
        (0..shape.len())
            .map(|d| {
                if axes.contains(&d) {
                    shape[d] - 1 - o[d]
                } else {
                    o[d]
                }
            })
            .collect()
    })
}

fn unfold_host<T: Copy>(
    data: &[T],
    shape: &[usize],
    dim: usize,
    size: usize,
    step: usize,
) -> (Vec<T>, Vec<usize>) {
    let mut out = shape.to_vec();
    out[dim] = (shape[dim] - size) / step + 1;
    out.push(size);
    let v = gather(data, shape, &out, |o| {
        let mut s = o[..shape.len()].to_vec();
        s[dim] = o[dim] * step + o[shape.len()];
        s
    });
    (v, out)
}

/// Wrapping sum of `data` (of `shape`) along `dim`, shape with `dim` kept as 1.
fn sum_dim_host(data: &[i32], shape: &[usize], dim: usize) -> Vec<i32> {
    let mut out = shape.to_vec();
    out[dim] = 1;
    let st = strides(shape);
    let n: usize = out.iter().product();
    (0..n)
        .map(|mut flat| {
            let mut o = vec![0; shape.len()];
            for d in (0..shape.len()).rev() {
                o[d] = flat % out[d];
                flat /= out[d];
            }
            (0..shape[dim]).fold(0i32, |acc, k| {
                o[dim] = k;
                acc.wrapping_add(data[o.iter().zip(&st).map(|(i, s)| i * s).sum::<usize>()])
            })
        })
        .collect()
}

fn int_tensor<const D: usize>(
    d: &TtDevice,
    v: &[i32],
    shape: [usize; D],
) -> Tensor<TtBackend, D, Int> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), d).to_device(d)
}

fn bool_tensor<const D: usize>(
    d: &TtDevice,
    v: &[bool],
    shape: [usize; D],
) -> Tensor<TtBackend, D, Bool> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), d).to_device(d)
}

fn resident<const D: usize>(t: &Tensor<TtBackend, D, Int>) -> bool {
    t.clone().into_primitive().computed_on_device()
}

fn resident_bool<const D: usize>(t: &Tensor<TtBackend, D, Bool>) -> bool {
    t.clone().into_primitive().computed_on_device()
}

fn read_ints<const D: usize>(t: Tensor<TtBackend, D, Int>) -> Vec<i32> {
    t.into_data().to_vec::<i32>().unwrap()
}

/// Every permutation worth distinguishing at rank 3, ragged and not.
const PERMS3: [[usize; 3]; 5] = [[2, 1, 0], [1, 0, 2], [0, 2, 1], [1, 2, 0], [2, 0, 1]];

#[test]
fn int_views_preserve_every_word_and_feed_reductions() {
    with_device(Config::default(), |d| {
        for shape in [[3usize, 33, 65], [2, 32, 32], [4, 1, 7]] {
            let n: usize = shape.iter().product();
            let data = ints(n, 7 + n as u32);
            let x = int_tensor(&d, &data, shape);
            let before = tensor_traffic();
            let ((permuted, flipped, unfolded), report) = burn_tt::with_report(|| {
                let permuted: Vec<_> = PERMS3.iter().map(|&a| x.clone().permute(a)).collect();
                let flipped: Vec<_> = [[0isize].as_slice(), &[1], &[2], &[0, 2], &[0, 1, 2]]
                    .iter()
                    .map(|a| {
                        // `flip` takes a const-length array: build each by length.
                        match a.len() {
                            1 => x.clone().flip([a[0]]),
                            2 => x.clone().flip([a[0], a[1]]),
                            _ => x.clone().flip([a[0], a[1], a[2]]),
                        }
                    })
                    .collect();
                let mut unfolded = Vec::new();
                for (dim, size, step) in [(2usize, 7usize, 3usize), (1, shape[1], 1), (0, 1, 1)] {
                    if size <= shape[dim] {
                        unfolded.push((dim, size, step, x.clone().unfold::<4, _>(dim, size, step)));
                    }
                }
                // Every result reads through its declared padding.
                for p in &permuted {
                    let _ = p.clone().sum_dim(1);
                }
                (permuted, flipped, unfolded)
            });
            assert_native_model(&report);
            assert_eq!(tensor_traffic().downloads, before.downloads, "{shape:?}");
            for (axes, got) in PERMS3.iter().zip(permuted) {
                assert!(resident(&got), "permute {axes:?} of {shape:?}");
                let (want, out) = permute_host(&data, &shape, axes);
                assert_eq!(got.dims().to_vec(), out);
                for dim in 0..3 {
                    assert_eq!(
                        read_ints(got.clone().sum_dim(dim)),
                        sum_dim_host(&want, &out, dim),
                        "sum_dim({dim}) of permute {axes:?} of {shape:?}"
                    );
                }
                assert_eq!(read_ints(got), want, "permute {axes:?} of {shape:?}");
            }
            let flips: [&[usize]; 5] = [&[0], &[1], &[2], &[0, 2], &[0, 1, 2]];
            for (axes, got) in flips.iter().zip(flipped) {
                assert!(resident(&got), "flip {axes:?} of {shape:?}");
                let want = flip_host(&data, &shape, axes);
                for dim in 0..3 {
                    assert_eq!(
                        read_ints(got.clone().sum_dim(dim)),
                        sum_dim_host(&want, &shape, dim),
                        "sum_dim({dim}) of flip {axes:?} of {shape:?}"
                    );
                }
                assert_eq!(read_ints(got), want, "flip {axes:?} of {shape:?}");
            }
            for (dim, size, step, got) in unfolded {
                assert!(resident(&got), "unfold {dim},{size},{step} of {shape:?}");
                let (want, out) = unfold_host(&data, &shape, dim, size, step);
                assert_eq!(got.dims().to_vec(), out);
                for sum in 0..4 {
                    assert_eq!(
                        read_ints(got.clone().sum_dim(sum)),
                        sum_dim_host(&want, &out, sum),
                        "sum_dim({sum}) of unfold {dim},{size},{step} of {shape:?}"
                    );
                }
                assert_eq!(
                    read_ints(got),
                    want,
                    "unfold {dim},{size},{step} of {shape:?}"
                );
            }
        }
    });
}

#[test]
fn int_views_of_rank_four_and_two_and_repeated_use() {
    with_device(Config::default(), |d| {
        let shape = [2usize, 5, 7, 3];
        let data = ints(210, 3);
        let x = int_tensor(&d, &data, shape);
        let axes = [3usize, 1, 0, 2];
        let got = x.clone().permute(axes).flip([0isize, 2]);
        assert!(resident(&got));
        let (p, out) = permute_host(&data, &shape, &axes);
        assert_eq!(read_ints(got), flip_host(&p, &out, &[0, 2]));
        // A matrix: permute is a transpose, flip both axes a copy.
        let m = ints(33 * 65, 5);
        let x = int_tensor(&d, &m, [33, 65]);
        assert_eq!(
            read_ints(x.clone().permute([1, 0])),
            permute_host(&m, &[33, 65], &[1, 0]).0
        );
        assert_eq!(
            read_ints(x.clone().flip([0isize, 1])),
            flip_host(&m, &[33, 65], &[0, 1])
        );
        // The identity permute and an empty flip are copies, not errors.
        assert_eq!(read_ints(x.clone().permute([0, 1])), m);
        assert_eq!(read_ints(x.flip([0isize; 0])), m);
    });
}

/// A view keeps its parent alive: the parent is dropped (its host copy never made, its device
/// buffer owned by the view) before the view is read, in every order of a chain.
#[test]
fn views_outlive_their_dropped_parents() {
    with_device(Config::default(), |d| {
        let shape = [3usize, 33, 65];
        let data = ints(3 * 33 * 65, 11);
        // Device-computed parent: no host copy to fall back on.
        let parent = int_tensor(&d, &data, shape).add_scalar(0);
        assert!(resident(&parent));
        let view = parent.clone().permute([2, 0, 1]);
        let copy = view.clone().flip([1isize]);
        let windows = copy.clone().unfold::<4, _>(0, 5, 4);
        drop(parent);
        drop(view);
        drop(copy);
        let (p, ps) = permute_host(&data, &shape, &[2, 0, 1]);
        let f = flip_host(&p, &ps, &[1]);
        let (w, ws) = unfold_host(&f, &ps, 0, 5, 4);
        assert_eq!(windows.dims().to_vec(), ws);
        assert_eq!(read_ints(windows), w);
        // Churn: many short-lived views, freed immediately, then a fresh read of an old one.
        let keep = int_tensor(&d, &data, shape).add_scalar(0);
        for k in 0..8 {
            let t = keep
                .clone()
                .permute(PERMS3[k % PERMS3.len()])
                .flip([0isize]);
            let _ = t.sum_dim(1);
        }
        assert_eq!(
            read_ints(keep.permute([1, 0, 2])),
            permute_host(&data, &shape, &[1, 0, 2]).0
        );
    });
}

/// A ragged flipped integer tensor, widened to float, through a matmul: the products and sums of
/// small integers are exact in FP32, so the float result is the host's integer matmul. The
/// matmul reads the flipped copy's padding as zero or the gate fails.
#[test]
fn ragged_int_views_feed_a_float_matmul() {
    with_device(Config::default(), |d| {
        let small: Vec<i32> = ints(33 * 65, 9).iter().map(|x| (x % 11) - 5).collect();
        let w: Vec<i32> = ints(65 * 7, 13).iter().map(|x| (x % 7) - 3).collect();
        let x = int_tensor(&d, &small, [33, 65]);
        let weights = int_tensor(&d, &w, [65, 7]).float();
        let before = tensor_traffic();
        let (out, report) = burn_tt::with_report(|| {
            let flipped = x.clone().flip([1isize]).permute([1, 0]).permute([1, 0]);
            flipped.float().matmul(weights.clone())
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let f = flip_host(&small, &[33, 65], &[1]);
        let want: Vec<f32> = (0..33 * 7)
            .map(|i| {
                let (r, c) = (i / 7, i % 7);
                (0..65).map(|k| f[r * 65 + k] * w[k * 7 + c]).sum::<i32>() as f32
            })
            .collect();
        assert_eq!(out.into_data().to_vec::<f32>().unwrap(), want);
    });
}

#[test]
fn bool_views_preserve_canonical_words_and_feed_reductions() {
    with_device(Config::default(), |d| {
        for shape in [[3usize, 33, 65], [2, 32, 32], [4, 1, 7]] {
            let n: usize = shape.iter().product();
            let data = bools(n, 5 + n as u32);
            let x = bool_tensor(&d, &data, shape);
            let before = tensor_traffic();
            let (results, report) = burn_tt::with_report(|| {
                let mut r = Vec::new();
                for a in PERMS3 {
                    r.push((
                        format!("permute {a:?}"),
                        permute_host(&data, &shape, &a),
                        x.clone().permute(a),
                    ));
                }
                for axes in [&[0usize][..], &[1], &[2], &[0, 2], &[0, 1, 2]] {
                    let t = match axes.len() {
                        1 => x.clone().flip([axes[0] as isize]),
                        2 => x.clone().flip([axes[0] as isize, axes[1] as isize]),
                        _ => x.clone().flip([0isize, 1, 2]),
                    };
                    r.push((
                        format!("flip {axes:?}"),
                        (flip_host(&data, &shape, axes), shape.to_vec()),
                        t,
                    ));
                }
                r
            });
            assert_native_model(&report);
            assert_eq!(tensor_traffic().downloads, before.downloads, "{shape:?}");
            for (what, (want, out), got) in results {
                assert!(resident_bool(&got), "{what} of {shape:?}");
                assert_eq!(got.dims().to_vec(), out);
                // A bool reduction reads the result's padding.
                let counts = got.clone().int().sum_dim(2);
                let as_int: Vec<i32> = want.iter().map(|&b| i32::from(b)).collect();
                assert_eq!(
                    read_ints(counts),
                    sum_dim_host(&as_int, &out, 2),
                    "{what} of {shape:?}"
                );
                assert_eq!(
                    got.into_data().to_vec::<bool>().unwrap(),
                    want,
                    "{what} of {shape:?}"
                );
            }
            let (want, out) = unfold_host(&data, &shape, 2, shape[2].min(7), 2);
            let got = x.clone().unfold::<4, _>(2, shape[2].min(7), 2);
            assert!(resident_bool(&got));
            assert_eq!(got.dims().to_vec(), out);
            assert_eq!(got.clone().into_data().to_vec::<bool>().unwrap(), want);
            let as_int: Vec<i32> = want.iter().map(|&b| i32::from(b)).collect();
            assert_eq!(
                read_ints(got.int().sum_dim(3)),
                sum_dim_host(&as_int, &out, 3)
            );
        }
    });
}
