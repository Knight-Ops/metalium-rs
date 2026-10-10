//! Gate, Burn level: `float_remainder` and `float_remainder_scalar`
//! are `burn-flex`'s `((a % b) + b) % b` bit for bit, native on the device.
//!
//! Against Flex itself (the external oracle; the kernel-level oracles are in
//! `step133_remainder`): every special value, denormals and signed zeros,
//! row/column/scalar/rank-3 broadcasts, a ragged producer chained into a
//! reduction and a matmul (the tensor form's padding is `0 % 0 = NaN`, declared
//! `Undefined`), the scalar form (padding `0 % 3 = 0`, declared zero), nothing
//! downloaded and the result computed on the device, and a one- and a two-tile
//! trace replayed with a changed divisor.
//!
//! A NaN is compared by class: the device stores the canonical `0x7fc00000`,
//! the host's libm its own payload.

use burn::tensor::{Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, InputPayload, TracedInference, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};
use tt_tests::data::xorshift;

/// Bits from every region the algorithm treats differently.
fn corpus(seed: u64, n: usize) -> Vec<f32> {
    let mut next = xorshift(seed);
    (0..n)
        .map(|_| {
            let (x, k) = (next(), next() >> 40);
            let sign = (x as u32) & 0x8000_0000;
            let frac = (x as u32) & 0x7f_ffff;
            let e = |lo: u32, span: u32| (((x >> 40) as u32) % span + lo) << 23;
            f32::from_bits(match k % 6 {
                0 => x as u32,
                1 => sign | frac,
                2 => sign | e(100, 40) | frac,
                3 => sign | e(1, 254) | frac,
                4 => sign | e(0, 12) | frac,
                _ => sign | e(243, 12) | frac,
            })
        })
        .collect()
}

const SPECIALS: [f32; 24] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    -3.0,
    4.0,
    0.5,
    f32::MAX,
    -f32::MAX,
    f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE,
    f32::from_bits(1),
    f32::from_bits(0x8000_0001),
    f32::from_bits(0x7f_ffff),
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::from_bits(0xffc0_0000),
    1e30,
    -1e30,
    1e-20,
    3.0e38,
    3.3e38,
];

/// The specials crossed, then the corpus: `n` pairs.
fn pairs(seed: u64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    'cross: for &x in &SPECIALS {
        for &y in &SPECIALS {
            if a.len() == n {
                break 'cross;
            }
            a.push(x);
            b.push(y);
        }
    }
    let rest = n - a.len();
    a.extend(corpus(seed, rest));
    b.extend(corpus(seed + 1000, rest));
    (a, b)
}

/// Finite operands, `b` never zero: a remainder whose sums are finite.
fn finite(seed: u64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let (a, b) = pairs(seed, n);
    let clean = |v: Vec<f32>| -> Vec<f32> {
        v.into_iter()
            .map(|x| if x.is_finite() && x != 0.0 { x } else { 1.5 })
            .collect()
    };
    let a: Vec<f32> = a
        .into_iter()
        .map(|x| {
            if x.is_finite() {
                x.clamp(-1e6, 1e6)
            } else {
                1.0
            }
        })
        .collect();
    (
        a,
        clean(b).into_iter().map(|x| x.clamp(-1e3, 1e3)).collect(),
    )
}

fn tt<const D: usize>(d: &burn_tt::TtDevice, v: &[f32], shape: [usize; D]) -> Tensor<TtBackend, D> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), d)
}

fn fx<const D: usize>(v: &[f32], shape: [usize; D]) -> Tensor<Flex, D> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), &FlexDevice)
}

fn values<B: burn::tensor::backend::Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[track_caller]
fn assert_flex(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = if w.is_nan() {
            g.is_nan()
        } else {
            g.to_bits() == w.to_bits()
        };
        assert!(
            ok,
            "{what}: element {i}: device {g:e} ({:08x}), flex {w:e} ({:08x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

fn primitive<const D: usize>(t: Tensor<TtBackend, D>) -> burn_tt::TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        TensorPrimitive::QFloat(_) => unreachable!("a float tensor"),
    }
}

const SCALARS: [f32; 12] = [
    3.0,
    -2.0,
    0.1,
    1.0e-40,
    f32::MAX,
    -f32::MIN_POSITIVE,
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NAN,
    7.0,
    f32::from_bits(1),
];

#[test]
fn remainder_is_flexs_bits_resident_and_nothing_downloaded() {
    with_device(Config::default(), |d| {
        let shape = [35, 37];
        let (a, b) = pairs(5, 35 * 37);
        let (x, y) = (tt(&d, &a, shape), tt(&d, &b, shape));
        let before = tensor_traffic();
        let (out, report) = burn_tt::with_report(|| {
            let r = x.clone().remainder(y.clone());
            assert!(
                primitive(r.clone()).computed_on_device(),
                "computed on the device"
            );
            let scalars: Vec<_> = SCALARS
                .iter()
                .map(|&s| {
                    let r = x.clone().remainder_scalar(s);
                    assert!(primitive(r.clone()).computed_on_device());
                    r
                })
                .collect();
            assert_eq!(
                tensor_traffic().downloads,
                before.downloads,
                "nothing downloaded"
            );
            (r, scalars)
        });
        assert_native_model(&report);
        for name in ["float_remainder", "float_remainder_scalar"] {
            let op = report.op(name).unwrap_or_else(|| panic!("{name} not run"));
            assert_eq!((op.on_host, op.downloads, op.staged), (0, 0, 0), "{name}");
            assert!(op.on_device > 0, "{name} ran on the device");
        }
        let (fa, fb) = (fx(&a, shape), fx(&b, shape));
        assert_flex(
            &values(out.0),
            &values(fa.clone().remainder(fb)),
            "tensor % tensor",
        );
        for (r, &s) in out.1.into_iter().zip(&SCALARS) {
            assert_flex(
                &values(r),
                &values(fa.clone().remainder_scalar(s)),
                &format!("tensor % {s:e}"),
            );
        }
    });
}

#[test]
fn broadcasts_are_flexs_bits_and_computed_on_the_device() {
    with_device(Config::default(), |d| {
        let (a, b) = pairs(9, 35 * 37);
        let (fa, ta) = (fx(&a, [35, 37]), tt(&d, &a, [35, 37]));
        let same_shape = |shape: &[usize]| shape.iter().product::<usize>();
        // (rhs shape): a row, a column, a scalar, a vector, and operands both broadcast.
        for rhs in [[1usize, 37], [35, 1], [1, 1]] {
            let n = same_shape(&rhs);
            let v = &b[100..100 + n];
            let got = ta.clone().remainder(tt(&d, v, rhs));
            assert!(primitive(got.clone()).computed_on_device(), "{rhs:?}");
            assert_flex(
                &values(got),
                &values(fa.clone().remainder(fx(v, rhs))),
                &format!("[35, 37] % {rhs:?}"),
            );
        }
        // The dividend the one that broadcasts.
        let row = &a[7..7 + 37];
        let got = tt(&d, row, [1, 37]).remainder(tt(&d, &b, [35, 37]));
        assert!(primitive(got.clone()).computed_on_device());
        assert_flex(
            &values(got),
            &values(fx(row, [1, 37]).remainder(fx(&b, [35, 37]))),
            "[1, 37] % [35, 37]",
        );
        // Both sides broadcast: `[4, 1, 5] % [1, 3, 1]` is `[4, 3, 5]`.
        let (p, q) = (&a[..20], &b[50..53]);
        let got = tt(&d, p, [4, 1, 5]).remainder(tt(&d, q, [1, 3, 1]));
        assert!(primitive(got.clone()).computed_on_device());
        assert_flex(
            &values(got),
            &values(fx(p, [4, 1, 5]).remainder(fx(q, [1, 3, 1]))),
            "[4, 1, 5] % [1, 3, 1]",
        );
    });
}

/// A ragged producer's padding must not reach a reduction or a matmul: the
/// tensor form's padding is `0 % 0 = NaN`, which a sum over the padded tile
/// would carry into every row. Held against the same values uploaded fresh
/// (zero padding) through the same device reduction -- the same accumulation
/// order, so bit for bit.
#[test]
fn a_ragged_remainder_chains_into_reductions_and_matmuls() {
    with_device(Config::default(), |d| {
        let (rows, cols) = (37, 35);
        let (a, b) = finite(3, rows * cols);
        // The matmul's other operand is a ragged remainder too (`[35, 5]`, its
        // padding rows along `K` NaN): a clean zero there would hide this
        // side's dirt (`NaN * 0`).
        let wa: Vec<f32> = (0..cols * 5)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.5)
            .collect();
        let wb: Vec<f32> = (0..cols * 5)
            .map(|i| ((i % 5) as f32 + 1.0) * if i % 3 == 0 { -1.5 } else { 1.0 })
            .collect();
        let (ta, tb) = (tt(&d, &a, [rows, cols]), tt(&d, &b, [rows, cols]));
        let flex_r = values(fx(&a, [rows, cols]).remainder(fx(&b, [rows, cols])));
        let fresh = tt(&d, &flex_r, [rows, cols]);
        let r = ta.clone().remainder(tb.clone());
        for (name, got, want) in [
            ("sum_dim(1)", r.clone().sum_dim(1), fresh.clone().sum_dim(1)),
            ("sum_dim(0)", r.clone().sum_dim(0), fresh.clone().sum_dim(0)),
        ] {
            assert_eq!(values(got), values(want), "tensor form, {name}");
        }
        let wt = tt(&d, &wa, [cols, 5]).remainder(tt(&d, &wb, [cols, 5]));
        let wt_fresh = tt(
            &d,
            &values(fx(&wa, [cols, 5]).remainder(fx(&wb, [cols, 5]))),
            [cols, 5],
        );
        assert_eq!(
            values(r.clone().matmul(wt.clone())),
            values(fresh.matmul(wt_fresh.clone())),
            "tensor form, matmul"
        );
        // The scalar form's padding is `0 % 3 = 0`, declared zero.
        let s = ta.clone().remainder_scalar(3.0);
        let flex_s = values(fx(&a, [rows, cols]).remainder_scalar(3.0));
        let fresh_s = tt(&d, &flex_s, [rows, cols]);
        assert_eq!(
            values(s.clone().sum_dim(1)),
            values(fresh_s.clone().sum_dim(1))
        );
        assert_eq!(values(s.matmul(wt)), values(fresh_s.matmul(wt_fresh)));
    });
}

#[test]
fn a_traced_remainder_is_the_fresh_one_with_a_changed_divisor() {
    with_device(Config::default(), |d| {
        // One tile and two tiles.
        for cols in [32usize, 64] {
            let n = 32 * cols;
            let (a0, b0) = finite(1, n);
            let (a, b) = (tt(&d, &a0, [32, cols]), tt(&d, &b0, [32, cols]));
            let (pa, pb) = (primitive(a.clone()), primitive(b.clone()));
            let (trace, first) = TracedInference::capture(&[&pa, &pb], || {
                vec![primitive(a.clone().remainder(b.clone()))]
            })
            .unwrap();
            let want = values(fx(&a0, [32, cols]).remainder(fx(&b0, [32, cols])));
            assert_flex(first[0].as_f32().unwrap(), &want, "the capture's own run");
            for round in 0..3u64 {
                // New dividends *and* new divisors, signs and magnitudes changed.
                let (na, nb) = finite(20 + round, n);
                let nb: Vec<f32> = nb.iter().map(|v| v * (round as f32 * 7.0 - 3.0)).collect();
                let out = trace
                    .run(vec![
                        InputPayload::F32(na.clone()),
                        InputPayload::F32(nb.clone()),
                    ])
                    .unwrap();
                let want = values(fx(&na, [32, cols]).remainder(fx(&nb, [32, cols])));
                assert_flex(
                    out[0].as_f32().unwrap(),
                    &want,
                    &format!("[32, {cols}] replay {round}"),
                );
            }
        }
        // The scalar form in a trace: the dividend changes.
        let a0 = finite(2, 32 * 32).0;
        let a = tt(&d, &a0, [32, 32]);
        let pa = primitive(a.clone());
        let (trace, _) =
            TracedInference::capture(&[&pa], || vec![primitive(a.clone().remainder_scalar(-3.0))])
                .unwrap();
        let na = finite(3, 32 * 32).0;
        let out = trace.run(vec![InputPayload::F32(na.clone())]).unwrap();
        assert_flex(
            out[0].as_f32().unwrap(),
            &values(fx(&na, [32, 32]).remainder_scalar(-3.0)),
            "scalar replay",
        );
    });
}

#[test]
fn shapes_that_do_not_broadcast_are_refused_by_name() {
    with_device(Config::default(), |d| {
        let (x, y) = (tt(&d, &[1.0; 6], [2, 3]), tt(&d, &[1.0; 20], [4, 5]));
        let message =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| x.remainder(y))).unwrap_err();
        let message = message
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| message.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap();
        assert!(
            message.contains("float_remainder")
                && message.contains("[2, 3]")
                && message.contains("[4, 5]"),
            "{message}"
        );
    });
}
