//! `float_cross` and `int_matmul` on the device.
//!
//! # `float_cross`
//!
//! Flex's composition (three slices of each operand, six products, three
//! differences, a concatenation) over the device's `SFPMAD` mul and add. The oracle
//! is the specification's functional model (`tt_isa::numerics::{mul_bh, add_bh}`, a
//! difference being `add_bh(x, y ^ sign)`): not fused, denormals flushed, NaN
//! canonicalised. The device must match it bit for bit on every element. Flex's
//! IEEE result must also match wherever the model's per-operation rounding equals
//! IEEE's for that element's two products and difference; the elements where it
//! does not (the denormal and NaN classes the model flushes and canonicalises) are
//! counted, and the count is asserted nonzero on the special-value data so the
//! Flex comparison there is not vacuous in the other direction.
//!
//! # `int_matmul`
//!
//! Exact modulo 2^32: wrapping `i32` multiplication and addition are associative
//! and commutative, so the order of the device's reduction cannot matter. The
//! oracle is a plain wrapping triple loop and Flex; the negative control is an
//! `f32` matmul of the same operands (what a device routing through `f32` would
//! compute), which must differ from the exact result on the gate's data above 2^24.
use burn::tensor::{Int, Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, InputPayload, TileChoice, TracedInference, TtBackend, TtDevice};
use tt_isa::numerics;
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn assert_resident(report: &burn_tt::Report, name: &str) {
    assert_native_model(report);
    let op = report
        .op(name)
        .unwrap_or_else(|| panic!("{name} was not reported"));
    assert_eq!((op.downloads, op.staged, op.on_host), (0, 0, 0), "{name}");
    // A composition reports its device work under the operations it calls.
    let device_work: u64 = report.0.iter().map(|o| o.on_device).sum();
    assert!(device_work > 0, "{name} did no device work");
}

/// The arithmetic of a composition ran on the device: each named part reported
/// device work and no host compute.
fn assert_parts_on_device(report: &burn_tt::Report, parts: &[&str]) {
    for part in parts {
        let op = report
            .op(part)
            .unwrap_or_else(|| panic!("{part} was not reported"));
        assert!(op.on_device > 0, "{part} did no device work");
        assert_eq!((op.downloads, op.staged, op.on_host), (0, 0, 0), "{part}");
    }
}

fn panic_message(f: impl FnOnce()) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .expect_err("the operation must refuse");
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

// ---------------------------------------------------------------- cross

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

fn normal(w: u64) -> u32 {
    let exponent = 120 + (w >> 24) as u32 % 16;
    (((w >> 9) as u32 & 1) << 31) | (exponent << 23) | (w as u32 & 0x7f_ffff)
}

fn words(n: usize, seed: u64, specials: bool) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            let w = lcg(&mut s);
            if specials && i % 3 == 0 {
                SPECIALS[w as usize % SPECIALS.len()]
            } else {
                normal(w)
            }
        })
        .collect()
}

fn float_data(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn read_bits<B: burn::tensor::backend::Backend, const D: usize>(t: Tensor<B, D>) -> Vec<u32> {
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

/// `data` of shape `from` as the broadcast `to`.
fn broadcast<T: Copy>(data: &[T], from: &[usize], to: &[usize]) -> Vec<T> {
    let count: usize = to.iter().product();
    (0..count)
        .map(|mut flat| {
            let mut at = 0;
            let mut stride = 1;
            for d in (0..to.len()).rev() {
                let c = flat % to[d];
                flat /= to[d];
                if from[d] != 1 {
                    at += c * stride;
                }
                stride *= from[d];
            }
            data[at]
        })
        .collect()
}

fn dev_sub(a: u32, b: u32) -> u32 {
    numerics::add_bh(a, b ^ 0x8000_0000)
}

/// The cross product of two broadcast arrays along `dim` with the device's
/// arithmetic, and (second result) whether every one of an element's two products
/// and its difference rounds as IEEE does (the Flex-comparable elements).
fn model_cross(a: &[u32], b: &[u32], shape: &[usize], dim: usize) -> (Vec<u32>, Vec<bool>) {
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    let at = |o: usize, c: usize, i: usize| (o * 3 + c) * inner + i;
    let mut out = vec![0; a.len()];
    let mut ieee = vec![true; a.len()];
    let f = f32::from_bits;
    for o in 0..outer {
        for i in 0..inner {
            let x = |c| a[at(o, c, i)];
            let y = |c| b[at(o, c, i)];
            for c in 0..3 {
                let (p, q) = ((c + 1) % 3, (c + 2) % 3);
                let m1 = numerics::mul_bh(x(p), y(q));
                let m2 = numerics::mul_bh(x(q), y(p));
                let r = dev_sub(m1, m2);
                out[at(o, c, i)] = r;
                let i1 = (f(x(p)) * f(y(q))).to_bits();
                let i2 = (f(x(q)) * f(y(p))).to_bits();
                let ir = (f(i1) - f(i2)).to_bits();
                // Equal bits, or both NaN: a NaN's payload is not a rounding.
                let same = |u: u32, v: u32| u == v || (f(u).is_nan() && f(v).is_nan());
                ieee[at(o, c, i)] = same(m1, i1) && same(m2, i2) && same(r, ir);
            }
        }
    }
    (out, ieee)
}

fn cross_case<const D: usize>(
    d: &TtDevice,
    ashape: [usize; D],
    bshape: [usize; D],
    dim: usize,
    seed: u64,
    specials: bool,
) -> (usize, usize) {
    let (abits, bbits) = (
        words(ashape.iter().product(), seed, specials),
        words(bshape.iter().product(), seed + 7, specials),
    );
    let out: Vec<usize> = (0..D).map(|i| ashape[i].max(bshape[i])).collect();
    let (ea, eb) = (
        broadcast(&abits, &ashape, &out),
        broadcast(&bbits, &bshape, &out),
    );
    let (want, ieee) = model_cross(&ea, &eb, &out, dim);
    let (swapped, _) = model_cross(&eb, &ea, &out, dim);
    assert_ne!(
        want, swapped,
        "negative control: swapped operands must be detectable"
    );

    let (a, b) = (tt_float(d, &abits, ashape), tt_float(d, &bbits, bshape));
    let before = tensor_traffic();
    let (y, report) = burn_tt::with_report(|| a.clone().cross(b.clone(), dim));
    assert_resident(&report, "float_cross");
    assert_parts_on_device(&report, &["float_slice", "float_mul", "float_sub"]);
    assert_eq!(tensor_traffic().downloads, before.downloads);
    let TensorPrimitive::Float(p) = y.clone().into_primitive() else {
        unreachable!()
    };
    assert!(p.computed_on_device());
    assert_eq!(y.dims().to_vec(), out);
    let got = read_bits(y);
    assert_eq!(
        got, want,
        "cross {ashape:?} x {bshape:?} dim {dim} (specials {specials})"
    );
    let flex =
        Tensor::<Flex, D>::from_data(TensorData::new(float_data(&abits), ashape), &FlexDevice)
            .cross(
                Tensor::<Flex, D>::from_data(
                    TensorData::new(float_data(&bbits), bshape),
                    &FlexDevice,
                ),
                dim,
            );
    let flex = read_bits(flex);
    let mut compared = 0;
    let mut exceptions = 0;
    for (e, &agree) in ieee.iter().enumerate() {
        if agree {
            compared += 1;
            // A NaN is compared as a class: the device canonicalises its payload
            // (the model's and the hardware's rule), Flex propagates one.
            let both_nan = f32::from_bits(flex[e]).is_nan() && f32::from_bits(got[e]).is_nan();
            assert!(
                flex[e] == got[e] || both_nan,
                "Flex and the device round alike here but differ: element {e}: {:#x} vs {:#x}",
                flex[e],
                got[e]
            );
        } else {
            exceptions += 1;
        }
    }
    (compared, exceptions)
}

#[test]
fn float_cross_is_the_devices_mul_and_sub_bit_for_bit_and_flexs_where_they_round_alike() {
    with_device(Config::default(), |d| {
        let mut compared = 0;
        let mut exceptions = 0;
        for specials in [false, true] {
            let mut add = |r: (usize, usize)| {
                compared += r.0;
                exceptions += r.1;
            };
            add(cross_case::<1>(&d, [3], [3], 0, 1, specials));
            add(cross_case::<2>(&d, [5, 3], [5, 3], 1, 2, specials));
            add(cross_case::<2>(&d, [3, 5], [3, 5], 0, 3, specials));
            add(cross_case::<3>(&d, [2, 3, 4], [2, 3, 4], 1, 4, specials));
            // Broadcast other axes.
            add(cross_case::<2>(&d, [4, 3], [1, 3], 1, 5, specials));
            add(cross_case::<3>(&d, [1, 3, 4], [2, 3, 1], 1, 6, specials));
            // Ragged against the 32-element tile.
            add(cross_case::<2>(&d, [37, 3], [37, 3], 1, 7, specials));
            if !specials {
                assert_eq!(
                    exceptions, 0,
                    "ordinary normals: the device's mul and sub round as IEEE does"
                );
            }
        }
        assert!(compared > 100, "the Flex comparison covered {compared}");
        assert!(
            exceptions > 0,
            "the special-value data must reach the denormal and NaN classes"
        );
        eprintln!("cross: {compared} elements compared with Flex, {exceptions} special-class");
    });
}

#[test]
fn cross_refuses_what_it_does_not_gate() {
    use burn_tensor::ops::FloatTensorOps;
    with_device(Config::default(), |d| {
        let prim = |shape: [usize; 2]| {
            let TensorPrimitive::Float(p) =
                tt_float(&d, &words(shape[0] * shape[1], 1, false), shape).into_primitive()
            else {
                unreachable!()
            };
            p
        };
        let cross = |a, b, dim| {
            let _ = <TtBackend as FloatTensorOps<TtBackend>>::float_cross(a, b, dim);
        };
        // Size 4 along dim (Burn's own check sits above the backend; this is the
        // primitive's contract), a mismatched axis, and an axis out of range.
        for (a, b, dim) in [
            (prim([2, 4]), prim([2, 4]), 1),
            (prim([2, 3]), prim([3, 3]), 1),
            (prim([2, 3]), prim([2, 3]), 2),
        ] {
            let m = panic_message(|| cross(a, b, dim));
            assert!(m.contains("float_cross"), "{m}");
        }
    });
}

/// Ragged `[5, 3]` and `[37, 3]` cross products feed reductions and a matmul; integer
/// valued operands, so products and sums are exact whatever their order. The
/// operands are dropped before the consumers run.
#[test]
fn ragged_cross_feeds_reductions_and_traces_changed_inputs() {
    for tiles in [1usize, 2] {
        with_device(
            Config {
                tiles: Some(TileChoice::Count(tiles)),
                ..Config::default()
            },
            |d| {
                for rows in [5usize, 37] {
                    let ints = |seed: u64| -> Vec<f32> {
                        let mut s = seed | 1;
                        (0..rows * 3)
                            .map(|_| (lcg(&mut s) % 17) as f32 - 8.0)
                            .collect()
                    };
                    let cross_of = |a: &[f32], b: &[f32]| -> Vec<f32> {
                        let mut out = vec![0.0; rows * 3];
                        for r in 0..rows {
                            let (x, y) = (&a[r * 3..r * 3 + 3], &b[r * 3..r * 3 + 3]);
                            out[r * 3] = x[1] * y[2] - x[2] * y[1];
                            out[r * 3 + 1] = x[2] * y[0] - x[0] * y[2];
                            out[r * 3 + 2] = x[0] * y[1] - x[1] * y[0];
                        }
                        out
                    };
                    let wanted = |a: &[f32], b: &[f32]| -> Vec<f32> {
                        let c = cross_of(a, b);
                        let sums: Vec<f32> = (0..3)
                            .map(|j| (0..rows).map(|r| c[r * 3 + j]).sum())
                            .collect();
                        let mut out = c;
                        out.extend(sums);
                        out
                    };
                    let (a0, b0) = (ints(1), ints(2));
                    let b = Tensor::<TtBackend, 2>::from_data(
                        TensorData::new(b0.clone(), [rows, 3]),
                        &d,
                    )
                    .to_device(&d);
                    let input = Tensor::<TtBackend, 2>::from_data(
                        TensorData::new(a0.clone(), [rows, 3]),
                        &d,
                    )
                    .mul_scalar(1.0);
                    let TensorPrimitive::Float(primitive) = input.clone().into_primitive() else {
                        unreachable!()
                    };
                    let before = tensor_traffic();
                    let ((trace, first), report) = burn_tt::with_report(|| {
                        burn_tt::Trace::capture(&primitive, || {
                            let c = input.clone().cross(b.clone(), 1);
                            let sums = c.clone().sum_dim(0).reshape([3]);
                            let out = Tensor::cat(vec![c.reshape([rows * 3]), sums], 0);
                            let TensorPrimitive::Float(out) = out.into_primitive() else {
                                unreachable!()
                            };
                            out
                        })
                        .unwrap()
                    });
                    assert_native_model(&report);
                    assert_eq!(tensor_traffic().uploads, before.uploads);
                    assert_eq!(first, wanted(&a0, &b0), "{rows} rows, {tiles} tiles");
                    drop(input);
                    for seed in [3u64, 4, 5] {
                        let a = ints(seed);
                        assert_eq!(
                            trace.run(a.clone()).unwrap(),
                            wanted(&a, &b0),
                            "{rows} rows, {tiles} tiles, replay {seed}"
                        );
                    }
                    drop(trace);
                    // The matmul of a cross product, outside the trace.
                    let w = Tensor::<TtBackend, 2>::ones([3, 2], &d);
                    let c = Tensor::<TtBackend, 2>::from_data(
                        TensorData::new(a0.clone(), [rows, 3]),
                        &d,
                    )
                    .to_device(&d)
                    .cross(b.clone(), 1);
                    let got = c.matmul(w).into_data().to_vec::<f32>().unwrap();
                    let cc = cross_of(&a0, &b0);
                    let want: Vec<f32> = (0..rows)
                        .flat_map(|r| {
                            let s = cc[r * 3] + cc[r * 3 + 1] + cc[r * 3 + 2];
                            [s, s]
                        })
                        .collect();
                    assert_eq!(got, want);
                }
            },
        );
    }
}

// ---------------------------------------------------------------- int_matmul

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
        46341,
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

/// A plain wrapping triple loop over the broadcast batch dimensions.
fn host_matmul(a: &[i32], ash: &[usize], b: &[i32], bsh: &[usize], f32_path: bool) -> Vec<i32> {
    let rank = ash.len();
    let (m, k, n) = (ash[rank - 2], ash[rank - 1], bsh[rank - 1]);
    let batch: Vec<usize> = (0..rank - 2).map(|d| ash[d].max(bsh[d])).collect();
    let batches: usize = batch.iter().product();
    let mut out = Vec::with_capacity(batches * m * n);
    for flat in 0..batches {
        let (mut ia, mut ib) = (0, 0);
        let (mut sa, mut sb) = (1, 1);
        let mut rem = flat;
        for d in (0..rank - 2).rev() {
            let c = rem % batch[d];
            rem /= batch[d];
            if ash[d] != 1 {
                ia += c * sa;
            }
            if bsh[d] != 1 {
                ib += c * sb;
            }
            sa *= ash[d];
            sb *= bsh[d];
        }
        for i in 0..m {
            for j in 0..n {
                if f32_path {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc += a[ia * m * k + i * k + p] as f32 * b[ib * k * n + p * n + j] as f32;
                    }
                    out.push(acc as i32);
                } else {
                    let mut acc = 0i32;
                    for p in 0..k {
                        acc = acc.wrapping_add(
                            a[ia * m * k + i * k + p].wrapping_mul(b[ib * k * n + p * n + j]),
                        );
                    }
                    out.push(acc);
                }
            }
        }
    }
    out
}

fn tt_int<const D: usize>(
    d: &TtDevice,
    values: &[i32],
    shape: [usize; D],
) -> Tensor<TtBackend, D, Int> {
    Tensor::<TtBackend, D, Int>::from_data(TensorData::new(values.to_vec(), shape), d).to_device(d)
}

fn matmul_case<const D: usize>(d: &TtDevice, ash: [usize; D], bsh: [usize; D], seed: u64) {
    let (a, b) = (
        int_words(ash.iter().product(), seed),
        int_words(bsh.iter().product(), seed + 50),
    );
    let want = host_matmul(&a, &ash, &b, &bsh, false);
    let float_path = host_matmul(&a, &ash, &b, &bsh, true);
    assert_ne!(
        want, float_path,
        "negative control: an f32 matmul must differ from the exact one on these words"
    );
    let (x, y) = (tt_int(d, &a, ash), tt_int(d, &b, bsh));
    let before = tensor_traffic();
    let (z, report) = burn_tt::with_report(|| x.clone().matmul(y.clone()));
    assert_resident(&report, "int_matmul");
    assert_eq!(tensor_traffic().downloads, before.downloads);
    assert!(z.clone().into_primitive().computed_on_device());
    let mut shape: Vec<usize> = (0..D - 2).map(|i| ash[i].max(bsh[i])).collect();
    shape.extend([ash[D - 2], bsh[D - 1]]);
    assert_eq!(z.dims().to_vec(), shape);
    let got = z.into_data().to_vec::<i32>().unwrap();
    assert_eq!(got, want, "{ash:?} x {bsh:?}");
    assert_ne!(got, float_path, "the device must not be an f32 matmul");
    let flex =
        Tensor::<Flex, D, Int>::from_data(TensorData::new(a, ash), &FlexDevice).matmul(Tensor::<
            Flex,
            D,
            Int,
        >::from_data(
            TensorData::new(b, bsh),
            &FlexDevice,
        ));
    assert_eq!(
        flex.into_data().convert::<i32>().to_vec::<i32>().unwrap(),
        want,
        "Flex is the oracle"
    );
}

#[test]
fn int_matmul_is_exact_modulo_2_pow_32_and_never_an_f32_matmul() {
    with_device(Config::default(), |d| {
        matmul_case::<2>(&d, [3, 5], [5, 4], 1);
        matmul_case::<2>(&d, [7, 9], [9, 5], 2);
        matmul_case::<3>(&d, [2, 3, 4], [2, 4, 5], 3);
        // Broadcast batch axes, on either side and both.
        matmul_case::<3>(&d, [2, 3, 4], [1, 4, 5], 4);
        matmul_case::<3>(&d, [1, 3, 4], [2, 4, 5], 5);
        matmul_case::<4>(&d, [2, 1, 3, 4], [1, 3, 4, 2], 6);
        // Ragged against the 32-element tile.
        matmul_case::<2>(&d, [33, 35], [35, 3], 7);
        // A single inner element.
        matmul_case::<2>(&d, [4, 1], [1, 6], 8);
    });
}

#[test]
fn int_matmul_neighbours_above_2_pow_24_are_not_tied() {
    with_device(Config::default(), |d| {
        // 2^24 + 1 against 1: a float image rounds the first to 2^24.
        let a = [(1 << 24) + 1, (1 << 24) + 3, i32::MAX, i32::MIN];
        let b = [1, 1, 1, 1];
        let z = tt_int(&d, &a, [2, 2]).matmul(tt_int(&d, &b, [2, 2]));
        assert_eq!(
            z.into_data().to_vec::<i32>().unwrap(),
            vec![
                (1 << 24) + 1 + (1 << 24) + 3,
                (1 << 24) + 1 + (1 << 24) + 3,
                i32::MAX.wrapping_add(i32::MIN),
                i32::MAX.wrapping_add(i32::MIN)
            ]
        );
        // i32::MAX * i32::MAX wraps to 1; i32::MIN * -1 wraps to i32::MIN.
        let z =
            tt_int(&d, &[i32::MAX, i32::MIN], [1, 2]).matmul(tt_int(&d, &[i32::MAX, -1], [2, 1]));
        assert_eq!(
            z.into_data().to_vec::<i32>().unwrap(),
            vec![1i32.wrapping_add(i32::MIN)]
        );
    });
}

#[test]
fn int_matmul_refuses_above_its_budget_naming_the_shapes() {
    with_device(Config::default(), |d| {
        // 64 * 128 * 600 = 4_915_200 > 2^22.
        let a = tt_int(&d, &vec![1; 64 * 128], [64, 128]);
        let b = tt_int(&d, &vec![1; 128 * 600], [128, 600]);
        let m = panic_message(|| {
            let _ = a.clone().matmul(b.clone());
        });
        assert!(
            m.contains("int_matmul")
                && m.contains("budget")
                && m.contains("[64, 128]")
                && m.contains("[128, 600]"),
            "{m}"
        );
    });
}

/// Ragged integer products feed an integer reduction and a second product, and a
/// trace replays changed inputs (extreme words) with nothing uploaded.
#[test]
fn ragged_int_matmul_feeds_reductions_and_replays_changed_inputs() {
    for tiles in [1usize, 2] {
        with_device(
            Config {
                tiles: Some(TileChoice::Count(tiles)),
                ..Config::default()
            },
            |d| {
                let (rows, cols) = (5usize, 7usize);
                let w = int_words(cols * 3, 90);
                let v = int_words(3 * 2, 91);
                let wt = tt_int(&d, &w, [cols, 3]);
                let vt = tt_int(&d, &v, [3, 2]);
                let want = |x: &[i32]| -> [Vec<i32>; 3] {
                    let y = host_matmul(x, &[rows, cols], &w, &[cols, 3], false);
                    let sums: Vec<i32> = (0..3)
                        .map(|j| (0..rows).fold(0i32, |a, r| a.wrapping_add(y[r * 3 + j])))
                        .collect();
                    let z = host_matmul(&y, &[rows, 3], &v, &[3, 2], false);
                    [y, sums, z]
                };
                let first = int_words(rows * cols, 92);
                let x = tt_int(&d, &first, [rows, cols]);
                let xp = x.clone().into_primitive();
                let forward = |x: Tensor<TtBackend, 2, Int>| {
                    let y = x.matmul(wt.clone());
                    (y.clone(), y.clone().sum_dim(0), y.matmul(vt.clone()))
                };
                let before = tensor_traffic();
                let (outs, report) = burn_tt::with_report(|| forward(x.clone()));
                assert_native_model(&report);
                assert_eq!(tensor_traffic().downloads, before.downloads);
                let w0 = want(&first);
                let read = |t: Tensor<TtBackend, 2, Int>| t.into_data().to_vec::<i32>().unwrap();
                assert_eq!(read(outs.0), w0[0]);
                assert_eq!(read(outs.1), w0[1]);
                assert_eq!(read(outs.2), w0[2]);
                let (trace, _) = TracedInference::capture(&[&xp], || {
                    let (a, b, c) = forward(x.clone());
                    vec![a.into_primitive(), b.into_primitive(), c.into_primitive()]
                })
                .unwrap();
                drop(x);
                for seed in [93u64, 94] {
                    let next = int_words(rows * cols, seed);
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
                        assert_eq!(got, w[k], "{tiles} tiles, replay {seed}, output {k}");
                    }
                }
            },
        );
    }
}
