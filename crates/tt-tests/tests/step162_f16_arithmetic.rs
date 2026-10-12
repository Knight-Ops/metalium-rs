//! F16 arithmetic through Burn: widen to F32, compute, round once to binary16 (D1).
//!
//! Found by Burn's own `tensor_f16` and `autodiff_f16` suites (`burn-tt-conformance`, 1,187 tests),
//! which could not start while `dtype_usage(F16)` was `Storage`. `DType::F16` is IEEE binary16 in
//! BF16's two-byte slots; an operation on it widens its operands to F32 on the card (the generated
//! `float_*` wrappers, `float_compute_input`), runs the F32 operation, and narrows the result once
//! (`cast_native`), as BF16 does. It found three defects the BF16 path shared: `expand` of a half-
//! precision tensor (the row gather reads F32 buffers only), an optional operand (a convolution's
//! bias) left in its own dtype beside widened ones, and pooling, `interpolate`, `float_empty` and
//! `float_random` refusing F16 by name.
//!
//! The oracle restates no operation's numerics. For each operation the F16 result must equal the
//! host's binary16 rounding (`oracle_to_f16`: from the IEEE definition, in f64, independent of the
//! device) of the **same operation run on the same operands widened on the host** (`oracle_to_f32`)
//! in F32 on the card. The F32 arithmetic has its own gates; what is checked here is the plumbing and
//! the single rounding: bit for bit, NaN against NaN class (a NaN's payload is the cast's business,
//! `step143`), signed zeros included.
//!
//! Negative controls: a truncating narrowing (`truncating_to_f16`) differs from the device on the
//! `mul` and `exp` corpora, so a result narrowed any other way than to nearest even fails; and every
//! case counts the elements whose F32 result was not already a binary16 value, which must be
//! nonzero, so the comparison cannot pass by an F32 result that needed no rounding. Residency: each
//! result is on the device and nothing was downloaded before the explicit readback.
//!
//! Also the CTC loss gradient against burn-flex, in F16 on both (the conformance entries
//! `ctc_loss_grad_mixed_input_lengths` and `_repeated_labels` are `precision`): see its test.

use burn::backend::Autodiff;
use burn::tensor::activation::{log_softmax, relu, softmax};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::module::{
    adaptive_avg_pool2d, avg_pool2d, conv2d, conv3d, ctc_loss, interpolate, max_pool2d,
};
use burn::tensor::ops::{ConvOptions, InterpolateMode, InterpolateOptions};
use burn::tensor::{DType, Distribution, FloatDType, Int, Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend, TtDevice};
use tt_tests::burn_device::{with_device, Config};

mod fp16_support;
use fp16_support::{lcg, oracle_to_f16, oracle_to_f32, truncating_to_f16};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dt {
    F16,
    F32,
}

/// binary16 bit patterns, a fixed function of `seed`. `wild`: every payload class (NaNs, infinities,
/// subnormals, signed zeros, overflow neighbours) among raw random words; otherwise finite values
/// in [-4, 4] that are not all on a coarse grid.
fn bits(seed: u64, n: usize, wild: bool) -> Vec<u16> {
    let mut s = seed;
    let specials: [u16; 16] = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e01,
        0xfe00, 0x3c00, 0xbc00, 0x3555, 0x4248,
    ];
    (0..n)
        .map(|i| {
            if wild {
                if i < specials.len() {
                    specials[(i + seed as usize) % specials.len()]
                } else {
                    (lcg(&mut s) >> 8) as u16
                }
            } else {
                let unit = (lcg(&mut s) >> 8) as f32 / (1 << 24) as f32;
                oracle_to_f16((unit * 8.0 - 4.0).to_bits())
            }
        })
        .collect()
}

/// The operand `seed` names, as F16, or as the host's F32 widening of the same patterns.
fn input<const D: usize>(
    d: &TtDevice,
    dt: Dt,
    seed: u64,
    shape: [usize; D],
    wild: bool,
) -> Tensor<TtBackend, D> {
    let n: usize = shape.iter().product();
    let b = bits(seed, n, wild);
    match dt {
        Dt::F16 => {
            let mut data = TensorData::new(b, shape);
            data.dtype = DType::F16;
            Tensor::from_data(data, (d, DType::F16))
        }
        Dt::F32 => {
            let v: Vec<f32> = b
                .iter()
                .map(|&h| f32::from_bits(oracle_to_f32(h)))
                .collect();
            Tensor::from_data(TensorData::new(v, shape), d)
        }
    }
}

fn is_nan16(h: u16) -> bool {
    h & 0x7c00 == 0x7c00 && h & 0x03ff != 0
}

/// Compare one operation in F16 with the oracle; returns the elements the rounding changed and the
/// elements a truncating narrowing would have got wrong.
fn check<const R: usize>(
    name: &str,
    d: &TtDevice,
    op: impl Fn(Dt) -> Tensor<TtBackend, R>,
) -> (usize, usize) {
    let before = tensor_traffic();
    let half = op(Dt::F16);
    assert_eq!(
        half.dtype(),
        DType::F16,
        "{name}: the result keeps the dtype"
    );
    match half.clone().into_primitive() {
        TensorPrimitive::Float(t) => assert!(t.computed_on_device(), "{name}: on the device"),
        _ => panic!("{name}: unexpected quantized tensor"),
    }
    assert_eq!(
        tensor_traffic().downloads,
        before.downloads,
        "{name}: nothing is read back before the explicit readback"
    );
    let wide = op(Dt::F32);
    let mut data = half.into_data();
    data.dtype = DType::U16;
    let got: Vec<u16> = data.to_vec().unwrap();
    let wide: Vec<u32> = wide
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|f| f.to_bits())
        .collect();
    assert_eq!(got.len(), wide.len(), "{name}: shape");
    let _ = d;
    let (mut rounded, mut truncated) = (0, 0);
    for (i, (&g, &w)) in got.iter().zip(&wide).enumerate() {
        let expected = oracle_to_f16(w);
        if is_nan16(expected) {
            assert!(
                is_nan16(g),
                "{name}[{i}]: {g:#06x} is not a NaN (F32 {w:#010x})"
            );
            continue;
        }
        assert_eq!(g, expected, "{name}[{i}]: F32 result {w:#010x}");
        rounded += usize::from(oracle_to_f32(expected) != w);
        truncated += usize::from(truncating_to_f16(w) != expected);
    }
    (rounded, truncated)
}

#[test]
fn elementwise_and_reductions_round_once() {
    with_device(Config::default(), |d| {
        let m = |dt, seed| input::<2>(&d, dt, seed, [64, 64], true);
        let f = |dt, seed| input::<2>(&d, dt, seed, [64, 64], false);
        let mut totals = std::collections::BTreeMap::new();
        // An operation that computes a new value: its F32 result must have needed rounding
        // somewhere, or the case proves nothing about the narrowing.
        let mut note = |name: &str, r: (usize, usize)| {
            assert!(
                r.0 > 0,
                "{name}: the rounding changed nothing, so the case proves nothing"
            );
            totals.insert(name.to_string(), r);
        };
        // A selection or movement: the result is operand values, so nothing may need rounding.
        let exact = |name: &str, r: (usize, usize)| {
            assert_eq!(
                r.0, 0,
                "{name}: a selection must return operand values unrounded"
            );
        };
        note("add", check("add", &d, |dt| m(dt, 1) + m(dt, 2)));
        note("sub", check("sub", &d, |dt| m(dt, 3) - m(dt, 4)));
        note("mul", check("mul", &d, |dt| m(dt, 5) * m(dt, 6)));
        note("div", check("div", &d, |dt| f(dt, 7) / f(dt, 8)));
        note("exp", check("exp", &d, |dt| f(dt, 9).exp()));
        note("log", check("log", &d, |dt| f(dt, 10).abs().log()));
        note("sqrt", check("sqrt", &d, |dt| f(dt, 11).abs().sqrt()));
        note("tanh", check("tanh", &d, |dt| f(dt, 12).tanh()));
        note(
            "mul_scalar",
            check("mul_scalar", &d, |dt| f(dt, 13).mul_scalar(0.3)),
        );
        note(
            "add_scalar",
            check("add_scalar", &d, |dt| f(dt, 14).add_scalar(0.1)),
        );
        exact("neg_abs", check("neg_abs", &d, |dt| m(dt, 15).neg().abs()));
        exact("relu", check("relu", &d, |dt| relu(m(dt, 16))));
        note("softmax", check("softmax", &d, |dt| softmax(f(dt, 17), 1)));
        // A ragged shape: the widened tensor's padding must not reach a reduction.
        let r = |dt, seed| input::<2>(&d, dt, seed, [37, 70], false);
        note("sum_dim", check("sum_dim", &d, |dt| r(dt, 18).sum_dim(1)));
        note(
            "mean_dim",
            check("mean_dim", &d, |dt| r(dt, 19).mean_dim(0)),
        );
        exact("max_dim", check("max_dim", &d, |dt| r(dt, 20).max_dim(1)));
        // Products: the Matrix Unit's own rounding is the F32 path's, equal on both sides.
        note(
            "matmul",
            check("matmul", &d, |dt| {
                input::<2>(&d, dt, 21, [33, 40], false).matmul(input::<2>(
                    &d,
                    dt,
                    22,
                    [40, 29],
                    false,
                ))
            }),
        );
        note(
            "matmul_batched",
            check("matmul_batched", &d, |dt| {
                input::<3>(&d, dt, 23, [2, 17, 23], false).matmul(input::<3>(
                    &d,
                    dt,
                    24,
                    [2, 23, 19],
                    false,
                ))
            }),
        );
        // Broadcast: the case the row gather could not read.
        exact(
            "expand",
            check("expand", &d, |dt| {
                input::<2>(&d, dt, 25, [1, 70], false).expand([37, 70])
            }),
        );
        exact(
            "expand_3d",
            check("expand_3d", &d, |dt| {
                input::<2>(&d, dt, 26, [2, 5], false).expand([3, 2, 5])
            }),
        );
        // Control: a truncating narrowing is told apart on the corpora where rounding is exercised.
        for name in ["mul", "exp"] {
            assert!(
                totals[name].1 > 0,
                "{name}: a truncating narrowing must differ from the device somewhere"
            );
        }
    });
}

#[test]
fn modules_round_once_and_take_a_half_precision_bias() {
    with_device(Config::default(), |d| {
        let x4 = |dt, seed| input::<4>(&d, dt, seed, [1, 2, 6, 6], false);
        let mut seen = 0;
        let mut note = |name: &str, r: (usize, usize)| {
            assert!(
                r.0 > 0,
                "{name}: the rounding changed nothing, so the case proves nothing"
            );
            seen += 1;
        };
        let exact = |name: &str, r: (usize, usize)| {
            assert_eq!(
                r.0, 0,
                "{name}: a selection must return operand values unrounded"
            );
        };
        // The bias is an optional operand; it computes in F32 with the rest.
        note(
            "conv2d",
            check("conv2d", &d, |dt| {
                conv2d(
                    x4(dt, 30),
                    input::<4>(&d, dt, 31, [3, 2, 3, 3], false),
                    Some(input::<1>(&d, dt, 32, [3], false)),
                    ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
                )
            }),
        );
        note(
            "conv3d_groups",
            check("conv3d_groups", &d, |dt| {
                conv3d(
                    input::<5>(&d, dt, 33, [1, 2, 4, 4, 4], false),
                    input::<5>(&d, dt, 34, [2, 1, 2, 2, 2], false),
                    Some(input::<1>(&d, dt, 35, [2], false)),
                    ConvOptions::new([1; 3], [0; 3], [1; 3], 2),
                )
            }),
        );
        note(
            "avg_pool2d",
            check("avg_pool2d", &d, |dt| {
                avg_pool2d(x4(dt, 36), [3, 3], [2, 2], [1, 1], true, false)
            }),
        );
        note(
            "adaptive_avg_pool2d",
            check("adaptive_avg_pool2d", &d, |dt| {
                adaptive_avg_pool2d(x4(dt, 37), [4, 3])
            }),
        );
        exact(
            "max_pool2d",
            check("max_pool2d", &d, |dt| {
                max_pool2d(x4(dt, 38), [2, 2], [2, 2], [0, 0], [1, 1], false)
            }),
        );
        note(
            "interpolate",
            check("interpolate", &d, |dt| {
                interpolate(
                    input::<4>(&d, dt, 39, [1, 2, 4, 4], false),
                    [7, 5],
                    InterpolateOptions::new(InterpolateMode::Bilinear),
                )
            }),
        );
        assert_eq!(seen, 5);
    });
}

#[test]
fn empty_and_random_make_half_precision_tensors() {
    with_device(Config::default(), |d| {
        let e = Tensor::<TtBackend, 2>::empty([3, 4], (&d, DType::F16));
        assert_eq!((e.dtype(), e.dims()), (DType::F16, [3, 4]));
        let empty = Tensor::<TtBackend, 2>::empty([0, 3], (&d, DType::F16));
        assert_eq!((empty.dtype(), empty.dims()), (DType::F16, [0, 3]));
        let r = Tensor::<TtBackend, 2>::random(
            [16, 16],
            Distribution::Uniform(0.0, 1.0),
            (&d, DType::F16),
        );
        assert_eq!(r.dtype(), DType::F16);
        let mut data = r.into_data();
        data.dtype = DType::U16;
        let got: Vec<u16> = data.to_vec().unwrap();
        for &h in &got {
            let v = f32::from_bits(oracle_to_f32(h));
            // A draw in [0, 1) held as binary16: a finite value that round-trips, below one
            // unless the narrowing of a value just under one rounded up to it.
            assert!((0.0..=1.0).contains(&v), "{h:#06x}");
            assert_eq!(oracle_to_f16(v.to_bits()), h);
        }
        let distinct: std::collections::BTreeSet<_> = got.iter().collect();
        assert!(
            distinct.len() > 100,
            "the draw must not be constant: {}",
            distinct.len()
        );
    });
}

fn logits(t: usize, n: usize, c: usize) -> Vec<f32> {
    let mut v = Vec::new();
    for ti in 0..t {
        for ni in 0..n {
            for ci in 0..c {
                v.push(((ti * 7 + ni * 13 + ci * 3) as f32 * 0.1).sin());
            }
        }
    }
    v
}

struct Ctc {
    t: usize,
    n: usize,
    c: usize,
    targets: Vec<i64>,
    shape: [usize; 2],
    input_lengths: Vec<i64>,
    target_lengths: Vec<i64>,
}

/// The gradient of the CTC loss of `log_softmax(logits)`, the loss computed in F16 or F32.
fn ctc_grad<B: AutodiffBackend>(d: &B::Device, half: bool, k: &Ctc) -> Vec<f32> {
    let x = Tensor::<B, 3>::from_data(TensorData::new(logits(k.t, k.n, k.c), [k.t, k.n, k.c]), d)
        .require_grad();
    let y = if half {
        x.clone().cast(FloatDType::F16)
    } else {
        x.clone()
    };
    let targets = Tensor::<B, 2, Int>::from_data(TensorData::new(k.targets.clone(), k.shape), d);
    let input_lengths =
        Tensor::<B, 1, Int>::from_data(TensorData::new(k.input_lengths.clone(), [k.n]), d);
    let target_lengths =
        Tensor::<B, 1, Int>::from_data(TensorData::new(k.target_lengths.clone(), [k.n]), d);
    let loss = ctc_loss(log_softmax(y, 2), targets, input_lengths, target_lengths, 0).sum();
    let grads = loss.backward();
    x.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Burn's own `autodiff_f16` CTC gradient tests (`mixed_input_lengths`, `repeated_labels`) compare the
/// F16 gradient with a PyTorch F32 reference at `rel_abs(1e-3, 1e-3)`, and burn-tt misses them in
/// both matmul profiles. The cause is not burn-tt: Burn's default `ctc_loss` composes elementwise
/// operations, each of which rounds to binary16, and burn-flex, running the same composition in F16 on
/// the CPU, misses the same reference by the same amount while agreeing with burn-tt to rounding noise.
/// The test states that: both F16 gradients are further from the F32 gradient than the vendored
/// tolerance, and burn-tt's differs from Flex's by under 1% of that shortfall.
#[test]
fn ctc_gradient_in_f16_is_flexs_to_rounding_noise() {
    let cases = [
        Ctc {
            t: 12,
            n: 3,
            c: 5,
            targets: vec![1, 4, 2, 0, 3, 1, 0, 0, 2, 4, 1, 3],
            shape: [3, 4],
            input_lengths: vec![12, 7, 10],
            target_lengths: vec![3, 2, 4],
        },
        Ctc {
            t: 8,
            n: 4,
            c: 6,
            targets: vec![1, 1, 2, 0, 2, 3, 2, 1, 5, 0, 0, 0, 1, 2, 3, 4],
            shape: [4, 4],
            input_lengths: vec![8; 4],
            target_lengths: vec![3, 4, 1, 4],
        },
    ];
    let fd = FlexDevice;
    // Host reference work in the parent: no other test of this binary may fork meanwhile
    // (`tt_ttsim::outside_fork`), or its child can wait on a lock this thread holds.
    let flex: Vec<(Vec<f32>, Vec<f32>)> = tt_ttsim::outside_fork(|| {
        cases
            .iter()
            .map(|k| {
                (
                    ctc_grad::<Autodiff<Flex>>(&fd, false, k),
                    ctc_grad::<Autodiff<Flex>>(&fd, true, k),
                )
            })
            .collect()
    });
    with_device(Config::default(), |d| {
        for (k, (f32_ref, flex16)) in cases.iter().zip(&flex) {
            // Control: the F32 gradient is the reference's (Flex's), so a shortfall is F16's.
            let tt32 = ctc_grad::<Autodiff<TtBackend>>(&d, false, k);
            assert!(
                max_diff(&tt32, f32_ref) < 1e-5,
                "{}",
                max_diff(&tt32, f32_ref)
            );
            let tt16 = ctc_grad::<Autodiff<TtBackend>>(&d, true, k);
            let shortfall = max_diff(flex16, f32_ref);
            assert!(
                shortfall > 1e-3,
                "Flex F16 meets the tolerance: {shortfall}"
            );
            assert!(max_diff(&tt16, f32_ref) > 1e-3);
            let apart = max_diff(&tt16, flex16);
            assert!(
                apart * 100.0 <= shortfall,
                "burn-tt F16 and Flex F16 differ by {apart}, the shortfall is {shortfall}"
            );
        }
    });
}
