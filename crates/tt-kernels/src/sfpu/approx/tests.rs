//! `approx.rs`'s interpreter gates: the fits against a Remez derivation, every
//! program against an f64 reference over its range, special values, the mode.
#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use super::*;
use crate::sfpu::ops::{self, kind_sfpu, reference};

/// The program for `kind` over `a`, by the interpreter, `256` columns wide.
fn run(kind: u32, a: &[f32]) -> Vec<f32> {
    let cols = 256;
    let rows = a.len().div_ceil(cols);
    let mut padded = a.to_vec();
    padded.resize(rows * cols, 1.0);
    let mut out = reference(kind, 0.0, &padded, None, rows, cols);
    out.truncate(a.len());
    out
}

/// A weighted Remez exchange, independent of the programs, so the hard-coded
/// fits are checked against a derivation.
mod remez {
    fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
        let n = b.len();
        for c in 0..n {
            let p = (c..n)
                .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
                .unwrap();
            a.swap(c, p);
            b.swap(c, p);
            for r in c + 1..n {
                let f = a[r][c] / a[c][c];
                for k in c..n {
                    a[r][k] -= f * a[c][k];
                }
                b[r] -= f * b[c];
            }
        }
        let mut x = vec![0.0; n];
        for c in (0..n).rev() {
            let mut s = b[c];
            for k in c + 1..n {
                s -= a[c][k] * x[k];
            }
            x[c] = s / a[c][c];
        }
        x
    }

    fn horner(c: &[f64], x: f64) -> f64 {
        c.iter().rev().fold(0.0, |a, &k| a * x + k)
    }

    /// Minimise `max |(P(x) - f(x)) w(x)|` over `[lo, hi]`, `P` of degree `n`.
    pub fn fit(
        f: &dyn Fn(f64) -> f64,
        w: &dyn Fn(f64) -> f64,
        lo: f64,
        hi: f64,
        n: usize,
    ) -> (Vec<f64>, f64) {
        let m = n + 2;
        let mut xs: Vec<f64> = (0..m)
            .map(|i| {
                (lo + hi) / 2.0
                    - (hi - lo) / 2.0 * (std::f64::consts::PI * i as f64 / (m - 1) as f64).cos()
            })
            .collect();
        let (mut coef, mut level) = (vec![0.0; n + 1], 0.0);
        for _ in 0..60 {
            let mut a = vec![vec![0.0; m]; m];
            let mut b = vec![0.0; m];
            for i in 0..m {
                let mut p = 1.0;
                for j in 0..=n {
                    a[i][j] = p * w(xs[i]);
                    p *= xs[i];
                }
                a[i][n + 1] = if i % 2 == 0 { -1.0 } else { 1.0 };
                b[i] = f(xs[i]) * w(xs[i]);
            }
            let sol = solve(a, b);
            coef = sol[..=n].to_vec();
            level = sol[n + 1].abs();
            let err = |x: f64| (horner(&coef, x) - f(x)) * w(x);
            let grid = 50_000;
            let g: Vec<f64> = (0..=grid)
                .map(|i| lo + (hi - lo) * i as f64 / grid as f64)
                .collect();
            let ev: Vec<f64> = g.iter().map(|&x| err(x)).collect();
            let mut new = Vec::new();
            let mut i = 0;
            while i <= grid {
                let s = ev[i] >= 0.0;
                let (mut j, mut best) = (i, i);
                while j <= grid && (ev[j] >= 0.0) == s {
                    if ev[j].abs() > ev[best].abs() {
                        best = j;
                    }
                    j += 1;
                }
                let (mut a_, mut b_) = (g[best.saturating_sub(1)], g[(best + 1).min(grid)]);
                for _ in 0..80 {
                    let c = b_ - 0.618_033_988_749_895 * (b_ - a_);
                    let d = a_ + 0.618_033_988_749_895 * (b_ - a_);
                    if err(c).abs() > err(d).abs() {
                        b_ = d;
                    } else {
                        a_ = c;
                    }
                }
                new.push((a_ + b_) / 2.0);
                i = j;
            }
            while new.len() > m {
                if err(new[0]).abs() < err(*new.last().unwrap()).abs() {
                    new.remove(0);
                } else {
                    new.pop();
                }
            }
            assert_eq!(new.len(), m, "the error did not alternate {m} times");
            let top = new.iter().map(|&x| err(x).abs()).fold(0.0, f64::max);
            xs = new;
            if (top - level).abs() / level < 1e-9 {
                level = top;
                break;
            }
        }
        (coef, level)
    }
}

#[test]
fn the_fits_are_the_remez_fits() {
    let h = std::f64::consts::LN_2 / 2.0;
    let g = |r: f64| {
        if r.abs() < 1e-8 {
            1.0 + r / 2.0
        } else {
            r.exp_m1() / r
        }
    };
    let q = |f: f64| {
        if f.abs() < 1e-8 {
            1.0 - f / 2.0
        } else {
            f.ln_1p() / f
        }
    };
    let (lo, hi) = (
        std::f64::consts::FRAC_1_SQRT_2 - 1.0,
        std::f64::consts::SQRT_2 - 1.0,
    );
    let cases: [(&str, (Vec<f64>, f64), &[f64], f64); 3] = [
        (
            "exp",
            remez::fit(&|r: f64| r.exp(), &|r: f64| (-r).exp(), -h, h, 3),
            &EXP_COEFFS,
            EXP_FIT_ERROR,
        ),
        (
            "expm1/r",
            remez::fit(&g, &|r: f64| 1.0 / g(r), -h, h, 3),
            &EXPM1_COEFFS,
            EXPM1_FIT_ERROR,
        ),
        (
            "ln(1+f)/f",
            remez::fit(&q, &|f: f64| 1.0 / q(f), lo, hi, 4),
            &LOG_COEFFS,
            LOG_FIT_ERROR,
        ),
    ];
    for (name, (coef, level), hard, stated) in cases {
        for (a, b) in coef.iter().zip(hard) {
            assert!((a - b).abs() < 1e-9, "{name}: {a} vs {b}");
        }
        // The stated error is the level, rounded up in the fifth digit.
        assert!(
            level <= stated && stated <= level * 1.0005,
            "{name}: {level:e} vs {stated:e}"
        );
    }
}

/// `sigmoid(c1 x + c3 x^3)` against `Phi`, over `[0, 8]` on a 1e-6 grid: the
/// derivative of the difference is below 0.8 (`sigma' z' <= 0.4`, `phi <=
/// 0.399`), so the grid's maximum plus `0.4e-6` bounds the supremum.
#[test]
fn gelu_fit_error_is_the_supremum() {
    let (c1, c3) = (GELU_C1 as f32 as f64, GELU_C3 as f32 as f64);
    let mut worst = 0.0f64;
    for i in 0..=8_000_000u32 {
        let x = i as f64 * 1e-6;
        let fit = 1.0 / (1.0 + (-(c1 * x + c3 * x * x * x)).exp());
        let phi = 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
        worst = worst.max((fit - phi).abs());
    }
    println!("gelu fit supremum {worst:e}");
    assert!(worst + 0.4e-6 <= GELU_FIT_ERROR, "{worst:e}");
    assert!(worst > 1.3e-4, "{worst:e}: a better fit exists?");
}

fn rel(got: f32, want: f64) -> f64 {
    (got as f64 - want).abs() / want.abs()
}

/// Hold `kind` to `want` within `bound` (relative) on every input where `want`
/// is a normal number, and print the worst case.
fn sweep(name: &str, kind: u32, xs: &[f32], want: &dyn Fn(f64) -> f64, bound: f64) -> f64 {
    let got = run(kind, xs);
    let mut worst = (0.0f64, 0.0f32);
    for (&x, &g) in xs.iter().zip(&got) {
        let w = want(x as f64);
        if !(w.is_finite() && w.abs() >= f32::MIN_POSITIVE as f64) {
            continue;
        }
        let e = rel(g, w);
        assert!(
            e <= bound,
            "{name}({x:e} = {:#010x}) = {g:e} vs {w:e}: {e:e} > {bound:e}",
            x.to_bits()
        );
        if e > worst.0 {
            worst = (e, x);
        }
    }
    println!(
        "{name}: worst {:.3e} ({:.1}% of {bound:e}) at {:e}",
        worst.0,
        100.0 * worst.0 / bound,
        worst.1
    );
    worst.0
}

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f32> {
    (0..=n)
        .map(|i| (lo + (hi - lo) * i as f64 / n as f64) as f32)
        .collect()
}

/// `m` positions in every binade `lo..=hi`, each times every sign.
fn binades(lo: i32, hi: i32, m: usize, signs: &[f32]) -> Vec<f32> {
    let mut v = Vec::new();
    for e in lo..=hi {
        for i in 0..m {
            let mant = 1.0 + (i as f64 + 0.37) / m as f64;
            for s in signs {
                v.push(s * (mant * 2f64.powi(e)) as f32);
            }
        }
    }
    v
}

#[test]
fn exp_is_within_its_bound_on_its_range() {
    let mut xs = linspace(-87.0, 88.0 - 1e-5, 80_000);
    xs.extend(linspace(-1.0, 1.0, 20_000));
    let worst = sweep("exp", kind::EXP, &xs, &|x| x.exp(), EXP_APPROX_BOUND);
    // The bound is not slack by an order of magnitude.
    assert!(worst > EXP_APPROX_BOUND / 4.0, "{worst:e}");
}

#[test]
fn log_is_within_its_bound_for_every_positive_normal() {
    let mut xs = binades(-126, 127, 64, &[1.0]);
    xs.extend(linspace(0.5, 1.5, 20_000));
    xs.extend(linspace(1.4, 1.43, 3_000));
    xs.extend([f32::MIN_POSITIVE, f32::MAX, 1.0, 2.0, 0.5]);
    let worst = sweep("log", kind::LOG, &xs, &|x| x.ln(), LOG_APPROX_BOUND);
    assert!(worst > LOG_APPROX_BOUND / 4.0, "{worst:e}");
}

#[test]
fn recip_is_within_its_bound_and_its_seed_beyond() {
    let xs = binades(-126, 109, 48, &[1.0, -1.0]);
    let worst = sweep("recip", kind::RECIP, &xs, &|x| 1.0 / x, RECIP_APPROX_BOUND);
    assert!(worst > RECIP_APPROX_BOUND / 4.0, "{worst:e}");
    // Where the Newton step flushes: the seed's.
    let far = binades(110, 125, 48, &[1.0, -1.0]);
    sweep(
        "recip far",
        kind::RECIP,
        &far,
        &|x| 1.0 / x,
        RECIP_SEED_BOUND,
    );
    // From `2^126` the seed is zero and so is the result.
    for x in [f32::from_bits(0x7e80_0000), 3.0e38, f32::MAX] {
        let got = run(kind::RECIP, &[x, -x]);
        assert_eq!([got[0].to_bits(), got[1].to_bits()], [0, 0x8000_0000]);
    }
}

#[test]
fn sigmoid_is_within_its_bound() {
    let mut xs = linspace(-87.0, 87.0, 80_000);
    xs.extend(linspace(-1.0, 1.0, 20_000));
    let w = |x: f64| 1.0 / (1.0 + (-x).exp());
    let worst = sweep("sigmoid", kind::SIGMOID, &xs, &w, SIGMOID_APPROX_BOUND);
    assert!(worst > SIGMOID_APPROX_BOUND / 4.0, "{worst:e}");
}

#[test]
fn tanh_is_within_its_bound_including_near_zero() {
    let mut xs = linspace(-9.5, 9.5, 80_000);
    xs.extend(linspace(-0.01, 0.01, 20_000));
    xs.extend(binades(-40, -1, 16, &[1.0, -1.0]));
    let worst = sweep("tanh", kind::TANH, &xs, &|x| x.tanh(), TANH_APPROX_BOUND);
    assert!(worst > TANH_APPROX_BOUND / 4.0, "{worst:e}");
}

#[test]
fn gelu_is_within_its_absolute_bound() {
    let mut xs = linspace(-12.0, 12.0, 80_000);
    xs.extend(linspace(-1.0, 1.0, 20_000));
    xs.extend([1e-20, -1e-20, 1e10, -1e10, 3e38, -3e38]);
    let got = run(kind::GELU, &xs);
    let mut worst = 0.0f64;
    for (&x, &g) in xs.iter().zip(&got) {
        let x = x as f64;
        let want = 0.5 * x * libm::erfc(-x / std::f64::consts::SQRT_2);
        let bound = gelu_approx_bound(x, want);
        let e = (g as f64 - want).abs();
        assert!(
            e <= bound,
            "gelu({x:e}) = {g:e} vs {want:e}: {e:e} > {bound:e}"
        );
        worst = worst.max(e / bound);
    }
    println!("gelu: worst {:.1}% of its bound", 100.0 * worst);
    assert!(worst > 0.25, "{worst}");
}

/// Every special value an ML input meets, per op.
#[test]
fn special_values() {
    let nans = [
        0x7fc0_0000u32,
        0xffc0_0000,
        0x7f80_0001,
        0xff80_0001,
        0xffff_ffff,
    ];
    let inf = f32::INFINITY;
    let denorm = f32::from_bits(0x0000_1234);
    let is = |kind: u32, x: f32| run(kind, &[x])[0];
    let bits = |kind: u32, x: f32| is(kind, x).to_bits();
    for k in kind::FIRST..=kind::LAST {
        for n in nans {
            assert!(is(k, f32::from_bits(n)).is_nan(), "kind {k:#x}({n:#x})");
        }
    }
    // exp.
    assert_eq!(is(kind::EXP, inf), inf);
    assert_eq!(bits(kind::EXP, -inf), 0);
    assert_eq!(is(kind::EXP, 88.0), inf);
    assert_eq!(is(kind::EXP, 1e30), inf);
    assert_eq!(bits(kind::EXP, -1e30), 0);
    assert_eq!(bits(kind::EXP, -87.5), 0);
    for z in [0.0f32, -0.0, denorm, -denorm] {
        assert!(rel(is(kind::EXP, z), 1.0) <= EXP_APPROX_BOUND, "exp({z:e})");
    }
    // log.
    for z in [0.0f32, -0.0, denorm, -denorm] {
        assert_eq!(is(kind::LOG, z), -inf, "log({z:e})");
    }
    assert_eq!(is(kind::LOG, inf), inf);
    assert!(is(kind::LOG, -inf).is_nan() && is(kind::LOG, -1.0).is_nan());
    assert_eq!(bits(kind::LOG, 1.0), 0);
    // recip.
    assert_eq!(is(kind::RECIP, 0.0), inf);
    assert_eq!(is(kind::RECIP, -0.0), -inf);
    assert_eq!(is(kind::RECIP, denorm), inf);
    assert_eq!(bits(kind::RECIP, inf), 0);
    assert_eq!(bits(kind::RECIP, -inf), 0x8000_0000);
    // sigmoid.
    // Saturates to within the bound of 1 (the Newton step on `1/1`), not to 1.
    for big in [inf, 1e30] {
        assert!(rel(is(kind::SIGMOID, big), 1.0) <= SIGMOID_APPROX_BOUND);
    }
    assert_eq!(bits(kind::SIGMOID, -inf), 0);
    assert_eq!(bits(kind::SIGMOID, -1e30), 0);
    for z in [0.0f32, -0.0] {
        assert!(rel(is(kind::SIGMOID, z), 0.5) <= SIGMOID_APPROX_BOUND);
    }
    // tanh.
    assert_eq!(is(kind::TANH, inf), 1.0);
    assert_eq!(is(kind::TANH, -inf), -1.0);
    assert_eq!(is(kind::TANH, 100.0), 1.0);
    assert_eq!(bits(kind::TANH, 0.0), 0);
    assert_eq!(bits(kind::TANH, -0.0), 0x8000_0000);
    // gelu.
    assert_eq!(is(kind::GELU, inf), inf);
    assert!(is(kind::GELU, -inf).is_nan(), "Flex's `-inf * 0`");
    assert_eq!(bits(kind::GELU, 0.0), 0);
    assert_eq!(bits(kind::GELU, -0.0), 0x8000_0000);
    assert!(rel(is(kind::GELU, 1e30), 1e30) <= SIGMOID_APPROX_BOUND);
}

/// The mode switches the program: some input is outside Precise's bound under
/// Approx (a negative control: it fails if `lower` ever returned the Precise
/// kind), and Precise is the default and its own kind.
#[test]
fn approx_breaks_the_precise_bound_somewhere() {
    assert_eq!(MathMode::default(), MathMode::Precise);
    let xs = linspace(-3.0, 3.0, 4000);
    let cases: [(u32, &dyn Fn(f64) -> f64, f64); 5] = [
        (kind_sfpu::EXP, &|x| x.exp(), ops::EXP_BOUND),
        (
            kind_sfpu::SIGMOID,
            &|x| 1.0 / (1.0 + (-x).exp()),
            ops::SIGMOID_BOUND,
        ),
        (kind_sfpu::TANH, &|x| x.tanh(), ops::TANH_BOUND),
        (kind_sfpu::LOG, &|x| x.ln(), ops::LOG_BOUND),
        (kind_sfpu::RECIP, &|x| 1.0 / x, 1.0 / 8_388_608.0),
    ];
    for (precise, want, bound) in cases {
        assert_eq!(MathMode::Precise.lower(precise), precise);
        let approx = MathMode::Approx.lower(precise);
        assert_ne!(approx, precise);
        assert_eq!(MathMode::of(approx), MathMode::Approx);
        assert_eq!(MathMode::of(precise), MathMode::Precise);
        assert_eq!(precise_of(approx), Some(precise));
        let xs: Vec<f32> = if precise == kind_sfpu::LOG || precise == kind_sfpu::RECIP {
            xs.iter().map(|x| x.abs() + 0.01).collect()
        } else {
            xs.clone()
        };
        let (p, a) = (run(precise, &xs), run(approx, &xs));
        let worst = |got: &[f32]| {
            xs.iter()
                .zip(got)
                .map(|(&x, &g)| {
                    let w = want(x as f64);
                    if w == 0.0 {
                        0.0
                    } else {
                        rel(g, w)
                    }
                })
                .fold(0.0, f64::max)
        };
        assert!(worst(&p) <= bound, "precise {precise:#x}: {:e}", worst(&p));
        assert!(
            worst(&a) > bound,
            "approx {approx:#x} stays within Precise's bound {bound:e}: {:e}",
            worst(&a)
        );
        assert_ne!(p, a, "{precise:#x}: the modes gave the same bits");
    }
    // An op with no Approx twin is itself in either mode.
    for k in [
        kind_sfpu::DIV,
        kind_sfpu::POW,
        kind_sfpu::SIN,
        kind_sfpu::ERF,
    ] {
        assert_eq!(MathMode::Approx.lower(k), k);
    }
}

#[test]
fn tt_math_is_parsed_and_refused() {
    assert_eq!(MathMode::parse(None), Ok(MathMode::Precise));
    assert_eq!(MathMode::parse(Some("precise")), Ok(MathMode::Precise));
    assert_eq!(MathMode::parse(Some("approx")), Ok(MathMode::Approx));
    let e = MathMode::parse(Some("fast")).unwrap_err();
    assert!(e.contains("TT_MATH=fast"), "{e}");
}

/// The instruction counts per tile, Precise against Approx.
#[test]
fn approx_programs_are_shorter() {
    println!(
        "{:<8} {:>9} {:>9} {:>7}",
        "op", "precise", "approx", "ratio"
    );
    for (precise, approx) in TWINS {
        let count = |k: u32| {
            let (_, code) = ops::program(k, 0.0).expect("a program");
            instructions_per_tile(&code)
        };
        let (p, a) = (count(precise), count(approx));
        println!(
            "{precise:#06x}   {p:>9} {a:>9} {:>6.2}x",
            p as f64 / a as f64
        );
        assert!(a < p, "{precise:#x}: approx {a} >= precise {p}");
    }
}
