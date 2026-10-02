//! Phase 10 gate (S4, milestone 10.2e): the exponential family and the
//! activations built on it, on the SFPU.
//!
//! As `step29_exp_log` and `step44_algebraic`: the device **bit for bit** to its
//! programs (`tt_kernels::sfpu::ops::reference_op`, `gelu_reference`), and the
//! programs to `burn-flex` within
//! the bounds derived on them (`EXPM1_BOUND`, `SIGMOID_BOUND`, `TANH_BOUND`,
//! `ERF_BOUND`, `GELU_BOUND`, `gelu_backward_bound`, `SINH_BOUND`, `COSH_BOUND`,
//! `ASINH_BOUND`, `ACOSH_BOUND`, `ATANH_BOUND`) plus Flex's own error --
//! an ulp, and for `gelu` the cancellation of Flex's own `1 + erf` on the
//! negative side (`|x| 2^-24` absolute), which the device's `erfc` does not
//! suffer (`ops::transcendental` holds the device to the exact value there).
//! `sigmoid_backward` is exact, Flex's order of roundings.

use burn::tensor::backend::ops::ActivationOps;
use burn::tensor::{activation, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::{
    gelu_backward_bound, gelu_reference, kind_sfpu::*, reference_op, Broadcast, ACOSH_BOUND,
    ASINH_BOUND, ATANH_BOUND, COSH_BOUND, ERF_BOUND, EXPM1_BOUND, GELU_BOUND, SIGMOID_BOUND,
    SINH_BOUND, TANH_BOUND,
};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Values across the activations' whole ranges, both tails, near zero, and
/// the specials (NaNs of both signs, both zeros, infinities).
fn values(seed: u64, n: usize) -> Vec<f32> {
    let specials = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        1.0,
        -1.0,
        1.0e-6,
        -1.0e-6,
        9.01,
        -13.0,
        88.7,
        -87.0,
        88.0,
        -89.41,
        89.42,
        2.44e-4,
        0.999_999_94,
        -1.000_000_1,
        4096.0,
        f32::MAX,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 40) as f32 / (1u64 << 24) as f32;
            match i % 5 {
                0 => specials[(s >> 33) as usize % specials.len()],
                1 => u * 28.0 - 14.0,
                2 => u * 2.0 - 1.0,
                3 => u * 0.02 - 0.01,
                _ => u * 180.0 - 90.0,
            }
        })
        .collect()
}

fn flex(v: &[f32], r: usize, c: usize) -> Tensor<Flex, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), &FlexDevice)
}

fn host(t: Tensor<Flex, 2>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

fn same(got: &[f32], model: &[f32], what: &str) {
    for (i, (g, m)) in got.iter().zip(model).enumerate() {
        assert_eq!(
            g.to_bits(),
            m.to_bits(),
            "{what}: element {i}: device {g:e}, program {m:e}"
        );
    }
}

/// `got` within `rel` (relative) plus `abs` of Flex's `want`, plus Flex's ulp;
/// a NaN by class; an infinity exactly; a denormal or zero answer as a zero,
/// either side of the range's end within the bound.
fn close(got: f32, want: f32, rel: f64, abs: f64, what: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: {got:e} vs NaN");
        return;
    }
    assert!(!got.is_nan(), "{what}: NaN vs {want:e}");
    if want.is_infinite() {
        assert_eq!(got, want, "{what}");
        return;
    }
    let (g, w) = (got as f64, want as f64);
    let tol = (rel + 1.2e-7) * w.abs() + abs + f32::MIN_POSITIVE as f64;
    assert!(
        (g - w).abs() <= tol,
        "{what}: {got:e} vs Flex {want:e}: {:e} > {tol:e}",
        (g - w).abs()
    );
}

/// Flex's own error in `atanh x`, relative: std's `0.5 * log1p(2x/(1 - x))`
/// on the *signed* `x`, so for `x < 0` the argument nears `-1`, where `log1p`
/// amplifies its two roundings (`1 - x`, the quotient) by `k = |w/((1 + w)
/// ln(1 + w))|` -- 42 at `x = -0.99` -- and adds its own ulp. The device works
/// on `|x|` (odd symmetry), where `k <= 1`.
fn flex_atanh_error(x: f32) -> f64 {
    let x = x as f64;
    let w = 2.0 * x / (1.0 - x);
    let k = (w / ((1.0 + w) * w.ln_1p())).abs();
    let k = if k.is_finite() { k } else { 1.0 };
    (2.0 * k + 2.0) / 16_777_216.0
}

fn op(kind: u32) -> Eltwise {
    Eltwise {
        kind,
        scalar: 0.0,
        scalar2: 0.0,
    }
}

fn run<T: tt_device::Transport>(s: &mut Session<T>, kind: u32, t: &[&DramTensor]) -> DramTensor {
    s.eltwise3(op(kind), t[0], t.get(1).copied(), t.get(2).copied())
        .unwrap()
}

#[test]
fn the_unary_kinds_are_their_programs_within_their_bounds() {
    with_session(|s| {
        for (r, c) in [(37, 70), (64, 128)] {
            let av = values(1, r * c);
            let a = s.upload(&av, r, c).unwrap();
            let fa = flex(&av, r, c);
            let none = Broadcast::None;
            for (kind, want, rel, what) in [
                (EXPM1, host(fa.clone().exp() - 1.0), EXPM1_BOUND, "expm1"),
                (
                    SIGMOID,
                    host(activation::sigmoid(fa.clone())),
                    SIGMOID_BOUND,
                    "sigmoid",
                ),
                (TANH, host(fa.clone().tanh()), TANH_BOUND, "tanh"),
                (ERF, host(fa.clone().erf()), ERF_BOUND, "erf"),
                (SINH, host(fa.clone().sinh()), SINH_BOUND, "sinh"),
                (COSH, host(fa.clone().cosh()), COSH_BOUND, "cosh"),
                (ASINH, host(fa.clone().asinh()), ASINH_BOUND, "asinh"),
                (ACOSH, host(fa.clone().acosh()), ACOSH_BOUND, "acosh"),
                (ATANH, host(fa.clone().atanh()), ATANH_BOUND, "atanh"),
            ] {
                let out = run(s, kind, &[&a]);
                let got = s.download(&out).unwrap();
                s.free(out).unwrap();
                let model = reference_op(kind, [0.0; 2], none, &[&av], r, c);
                same(&got, &model, what);
                for i in 0..r * c {
                    // Flex's `exp(x) - 1` cancels near zero (the device's does
                    // not): its error there is an ulp of 1.
                    let abs = match kind {
                        EXPM1 => 1.2e-7,
                        ATANH => flex_atanh_error(av[i]) * (want[i] as f64).abs(),
                        _ => 0.0,
                    };
                    close(
                        model[i],
                        want[i],
                        rel,
                        abs,
                        &format!("{what}({:e}) [{r}, {c}]", av[i]),
                    );
                }
            }
            s.free(a).unwrap();
        }
    });
}

#[test]
fn gelu_and_the_backwards_are_their_programs_within_their_bounds() {
    with_session(|s| {
        for (r, c) in [(37, 70), (64, 64)] {
            let (xv, gv) = (values(2, r * c), values(3, r * c));
            let gv: Vec<f32> = gv
                .iter()
                .map(|g| {
                    if g.is_finite() {
                        g.clamp(-3.0, 3.0)
                    } else {
                        1.0
                    }
                })
                .collect();
            let (x, g) = (s.upload(&xv, r, c).unwrap(), s.upload(&gv, r, c).unwrap());
            let (fx, fg) = (flex(&xv, r, c), flex(&gv, r, c));
            let out = run(s, GELU, &[&x]);
            let got = s.download(&out).unwrap();
            same(&got, &gelu_reference(&xv, None, r, c), "gelu");
            let want = host(activation::gelu(fx.clone()));
            for i in 0..r * c {
                let abs = (xv[i] as f64).abs() / 16_777_216.0;
                close(
                    got[i],
                    want[i],
                    GELU_BOUND,
                    abs,
                    &format!("gelu({:e})", xv[i]),
                );
            }
            s.free(out).unwrap();
            let out = run(s, GELU_BACKWARD, &[&x, &g]);
            let got = s.download(&out).unwrap();
            same(&got, &gelu_reference(&xv, Some(&gv), r, c), "gelu_backward");
            let want = host(Tensor::from_primitive(
                burn::tensor::TensorPrimitive::Float(<Flex as ActivationOps<Flex>>::gelu_backward(
                    fx.clone().into_primitive().tensor(),
                    fg.clone().into_primitive().tensor(),
                )),
            ));
            for i in 0..r * c {
                let abs = gelu_backward_bound(xv[i], gv[i]) + (gv[i] as f64).abs() / 8_388_608.0;
                close(
                    got[i],
                    want[i],
                    0.0,
                    abs,
                    &format!("gelu'({:e}) * {}", xv[i], gv[i]),
                );
            }
            s.free(out).unwrap();
            // `sigmoid_backward`, exact: Flex's `g * s * (1 - s)`.
            let sv = host(activation::sigmoid(fx.clone()));
            let sd = s.upload(&sv, r, c).unwrap();
            let out = run(s, SIGMOID_BACKWARD, &[&sd, &g]);
            let got = s.download(&out).unwrap();
            same(
                &got,
                &reference_op(
                    SIGMOID_BACKWARD,
                    [0.0; 2],
                    Broadcast::None,
                    &[&sv, &gv],
                    r,
                    c,
                ),
                "sigmoid_backward",
            );
            let want = host(Tensor::from_primitive(
                burn::tensor::TensorPrimitive::Float(
                    <Flex as ActivationOps<Flex>>::sigmoid_backward(
                        flex(&sv, r, c).into_primitive().tensor(),
                        fg.clone().into_primitive().tensor(),
                    ),
                ),
            ));
            for i in 0..r * c {
                let (g_, w) = (got[i], want[i]);
                // A denormal result flushes; so does a denormal operand (Flex's
                // own sigmoid gives some), whose product is then a zero.
                let tiny = |v: f32| v != 0.0 && v.abs() < f32::MIN_POSITIVE;
                let flushed = (tiny(w) || tiny(sv[i])) && g_ == 0.0;
                assert!(
                    g_.to_bits() == w.to_bits() || (g_.is_nan() && w.is_nan()) || flushed,
                    "sigmoid_backward({:e}, {}): {g_:e} vs Flex {w:e}",
                    sv[i],
                    gv[i]
                );
            }
        }
    });
}
