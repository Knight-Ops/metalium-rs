//! Lane T8 gate (S10) through Burn: `TtDevice::set_math_mode`.
//!
//! `burn_tt::TtDevice::set_math_mode(MathMode::Approx)` makes `exp`, `log`,
//! `recip`, `sigmoid`, `tanh` and `gelu` run the fast programs
//! (`tt_kernels::sfpu::approx`) from the next op on; `Precise` is the default.
//! Held here, through Burn tensors:
//!
//! * the result is the interpreter's program for the kind the mode runs, bit
//!   for bit, and within the bound derived beside it of an f64 oracle;
//! * residency: nothing is uploaded or downloaded by the op, the result was
//!   computed on the device, every reported op is native;
//! * the mode is switchable on a live device, and each mode's bits are its own
//!   (Precise and Approx differ; switching back gives Precise's again);
//! * a ragged `[37, 70]` Approx result feeds a reduction and a matmul and
//!   neither sees the padding: both equal the same values uploaded clean, bit
//!   for bit.
//!
//! The fused programs (softmax, log-softmax, norms) are Precise in both modes,
//! and autodiff's backward ops keep their Precise programs (their formulas are
//! exact in the forward output). `TT_MATH` is read by `Session::open`, the one
//! reader, and held in `step145_approx_math`.

use burn::tensor::{activation, Tensor, TensorData, TensorPrimitive};
use burn_tt::{tensor_traffic, MathMode, TtBackend, TtDevice};
use tt_kernels::sfpu::approx::{
    gelu_approx_bound, kind, EXP_APPROX_BOUND, LOG_APPROX_BOUND, RECIP_APPROX_BOUND,
    SIGMOID_APPROX_BOUND, TANH_APPROX_BOUND,
};
use tt_kernels::sfpu::ops::{kind_sfpu, reference};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

type T2 = Tensor<TtBackend, 2>;

struct Op {
    name: &'static str,
    precise: u32,
    approx: u32,
    run: fn(T2) -> T2,
    /// Inputs for the bound check: finite, in range.
    range: (f64, f64),
    positive: bool,
    /// The f64 oracle and its allowed error at `x`.
    check: fn(f64, f64) -> (f64, f64),
}

fn rel(want: f64, bound: f64) -> (f64, f64) {
    (want, bound * want.abs())
}

const OPS: [Op; 6] = [
    Op {
        name: "exp",
        precise: kind_sfpu::EXP,
        approx: kind::EXP,
        run: |x| x.exp(),
        range: (-86.0, 87.0),
        positive: false,
        check: |x, _| rel(x.exp(), EXP_APPROX_BOUND),
    },
    Op {
        name: "log",
        precise: kind_sfpu::LOG,
        approx: kind::LOG,
        run: |x| x.log(),
        range: (1e-30, 1e30),
        positive: true,
        check: |x, _| {
            // `ln 1 = +0` exactly; elsewhere relative.
            let w = x.ln();
            (w, LOG_APPROX_BOUND * w.abs())
        },
    },
    Op {
        name: "recip",
        precise: kind_sfpu::RECIP,
        approx: kind::RECIP,
        run: |x| x.recip(),
        range: (1e-30, 1e30),
        positive: true,
        check: |x, _| rel(1.0 / x, RECIP_APPROX_BOUND),
    },
    Op {
        name: "sigmoid",
        precise: kind_sfpu::SIGMOID,
        approx: kind::SIGMOID,
        run: activation::sigmoid,
        range: (-86.0, 86.0),
        positive: false,
        check: |x, _| rel(1.0 / (1.0 + (-x).exp()), SIGMOID_APPROX_BOUND),
    },
    Op {
        name: "tanh",
        precise: kind_sfpu::TANH,
        approx: kind::TANH,
        run: |x| x.tanh(),
        range: (-8.9, 8.9),
        positive: false,
        check: |x, _| rel(x.tanh(), TANH_APPROX_BOUND),
    },
    Op {
        name: "gelu",
        precise: kind_sfpu::GELU,
        approx: kind::GELU,
        run: activation::gelu,
        range: (-12.0, 12.0),
        positive: false,
        check: |x, _| {
            let w = 0.5 * x * libm::erfc(-x / std::f64::consts::SQRT_2);
            (w, gelu_approx_bound(x, w) + f32::MIN_POSITIVE as f64)
        },
    },
];

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

/// `n` finite inputs of `op`'s range (a log-uniform draw for the positive ops).
fn inputs(op: &Op, seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            let u = (lcg(&mut s) & 0xff_ffff) as f64 / (1u64 << 24) as f64;
            if op.positive {
                (op.range.0.ln() + u * (op.range.1.ln() - op.range.0.ln())).exp() as f32
            } else {
                (op.range.0 + u * (op.range.1 - op.range.0)) as f32
            }
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn values(t: T2) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

fn tensor(d: &TtDevice, v: &[f32], shape: [usize; 2]) -> T2 {
    T2::from_data(TensorData::new(v.to_vec(), shape), d).to_device(d)
}

/// One op in one mode through Burn: resident, native, computed on the device.
fn resident(op: &Op, d: &TtDevice, x: &T2) -> Vec<f32> {
    let before = tensor_traffic();
    let (y, report) = burn_tt::with_report(|| (op.run)(x.clone()));
    let during = tensor_traffic() - before;
    assert_eq!(
        (during.uploads, during.downloads),
        (0, 0),
        "{}: {during:?}",
        op.name
    );
    assert_native_model(&report);
    let TensorPrimitive::Float(p) = y.clone().into_primitive() else {
        unreachable!()
    };
    assert!(p.computed_on_device(), "{}", op.name);
    let _ = d;
    values(y)
}

#[test]
fn approx_through_burn_is_the_program_within_its_bound_and_switchable() {
    let [r, c] = [37, 70];
    with_device(Config::default(), |d| {
        // `TT_MATH` unset: Precise.
        assert_eq!(d.math_mode(), MathMode::Precise);
        for op in &OPS {
            let v = inputs(op, 11, r * c);
            let x = tensor(&d, &v, [r, c]);

            // Precise first: the default, the Precise program's bits.
            let precise = resident(op, &d, &x);
            assert_eq!(
                bits(&precise),
                bits(&reference(op.precise, 0.0, &v, None, r, c)),
                "{} precise",
                op.name
            );

            d.set_math_mode(MathMode::Approx).unwrap();
            assert_eq!(d.math_mode(), MathMode::Approx);
            let approx = resident(op, &d, &x);
            assert_eq!(
                bits(&approx),
                bits(&reference(op.approx, 0.0, &v, None, r, c)),
                "{} approx",
                op.name
            );
            assert_ne!(
                bits(&approx),
                bits(&precise),
                "{}: Approx gave Precise's bits",
                op.name
            );
            // The f64 oracle, on every element.
            let mut worst = 0.0f64;
            for (xi, g) in v.iter().zip(&approx) {
                let (want, bound) = (op.check)(*xi as f64, 0.0);
                let err = (*g as f64 - want).abs();
                assert!(
                    err <= bound || (want.abs() < 4.0 * f32::MIN_POSITIVE as f64 && *g == 0.0),
                    "{} approx({xi:e}) = {g:e} vs {want:e}: {err:e} > {bound:e}",
                    op.name
                );
                if bound > 0.0 {
                    worst = worst.max(err / bound);
                }
            }
            println!("{}: worst {:.1}% of its bound", op.name, 100.0 * worst);

            // Back to Precise on the live device: Precise's bits again (the
            // programs are cached per mode, neither replaces the other).
            d.set_math_mode(MathMode::Precise).unwrap();
            assert_eq!(bits(&resident(op, &d, &x)), bits(&precise), "{}", op.name);
        }
    });
}

/// Ragged Approx results into a reduction and a matmul: both read the real
/// datums only, bit for bit what the same values uploaded clean give.
#[test]
fn ragged_approx_results_feed_reductions_and_matmuls() {
    let [r, c, n] = [37, 70, 24];
    with_device(Config::default(), |d| {
        d.set_math_mode(MathMode::Approx).unwrap();
        let wv: Vec<f32> = (0..c * n).map(|i| ((i * 7 + 3) % 5) as f32 - 2.0).collect();
        let w = tensor(&d, &wv, [c, n]);
        for op in &OPS {
            // Moderate values: no sum or product leaves the range.
            let v: Vec<f32> = inputs(op, 3, r * c)
                .into_iter()
                .map(|x| {
                    if x.abs() < 8.0 && x.abs() > 1e-6 {
                        x
                    } else {
                        0.5
                    }
                })
                .collect();
            let x = tensor(&d, &v, [r, c]);
            let y = (op.run)(x);
            let clean = tensor(&d, &values(y.clone()), [r, c]);
            for dim in [0usize, 1] {
                assert_eq!(
                    bits(&values(y.clone().sum_dim(dim))),
                    bits(&values(clean.clone().sum_dim(dim))),
                    "{} sum over dim {dim}: padding reached a real datum",
                    op.name
                );
            }
            assert_eq!(
                bits(&values(y.clone().matmul(w.clone()))),
                bits(&values(clean.clone().matmul(w.clone()))),
                "{} matmul: padding reached a real datum",
                op.name
            );
        }
    });
}

/// Ops with no Approx program run Precise in either mode: `sqrt` here.
#[test]
fn ops_without_an_approx_twin_are_unchanged() {
    let [r, c] = [5, 33];
    with_device(Config::default(), |d| {
        let v: Vec<f32> = (0..r * c).map(|i| 0.5 + i as f32 / 7.0).collect();
        let x = tensor(&d, &v, [r, c]);
        let precise = values(x.clone().sqrt());
        d.set_math_mode(MathMode::Approx).unwrap();
        assert_eq!(bits(&values(x.sqrt())), bits(&precise));
    });
}
