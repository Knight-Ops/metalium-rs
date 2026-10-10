//! Shared by `step140`-`step142`: the pinned statistical tests and
//! the special functions their p-values need. Nothing here touches the code
//! under test.
//!
//! # Critical values (derived, not tuned)
//!
//! Every test reports a p-value and passes when `p >= ALPHA = 1e-6`.
//! * chi-square: `p = Q(df/2, x/2)`, the regularised upper incomplete gamma
//!   (series for `x < a + 1`, Lentz continued fraction otherwise), `df` the
//!   number of cells minus one; expected counts are exact (`n / cells`) and at
//!   least 20 per cell, where the chi-square approximation holds.
//! * Kolmogorov-Smirnov: `p = 2 sum_{j>=1} (-1)^(j-1) exp(-2 j^2 l^2)`,
//!   `l = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) D` (Stephens), the asymptotic
//!   distribution of `sqrt(n) D`; `n` here is above 10^5.
//! * mean/variance z-scores: the sample mean has standard deviation
//!   `sigma / sqrt(n)`, the sample variance `sqrt((mu4 - sigma^4) / n)`; the
//!   two-sided tolerance is `Z_ALPHA = 4.8916` standard deviations
//!   (`P(|N(0,1)| > 4.8916) = 1e-6`).
#![allow(dead_code)]

use tt_kernels::prng::*;
use tt_kernels::sfpu::ops::{COS_BOUND, LOG_BOUND};

/// The pinned significance level of every gate.
pub const ALPHA: f64 = 1e-6;
/// Two-sided normal quantile for [`ALPHA`]: `P(|Z| > 4.891638) = 1e-6`.
pub const Z_ALPHA: f64 = 4.891638;

fn ln_gamma(x: f64) -> f64 {
    // Lanczos, g = 7, n = 9 (relative error below 1e-15 for x > 0.5).
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularised upper incomplete gamma `Q(a, x)`.
fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        let (mut sum, mut term, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..100_000 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        1.0 - sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..100_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        (-x + a * x.ln() - ln_gamma(a)).exp() * h
    }
}

/// Upper-tail p-value of a chi-square statistic with `df` degrees of freedom.
pub fn chi2_p(statistic: f64, df: usize) -> f64 {
    gamma_q(df as f64 / 2.0, statistic / 2.0)
}

/// Chi-square statistic of `counts` against equal expected counts.
pub fn chi2_uniform(counts: &[u64]) -> f64 {
    let n: u64 = counts.iter().sum();
    let expected = n as f64 / counts.len() as f64;
    assert!(expected >= 20.0, "expected count {expected} < 20");
    counts
        .iter()
        .map(|&c| (c as f64 - expected).powi(2) / expected)
        .sum()
}

/// p-value of the uniformity of `values` (each in `[0, 1)`) over `cells`
/// equal bins of the integer `(v * 2^23) >> shift` truncation, i.e. bins of the
/// top bits when `shift > 0` -- or of the LOW bits when `low` (`k mod cells`).
pub fn binned_p(values: &[f32], cells: usize, low: bool) -> f64 {
    let mut counts = vec![0u64; cells];
    for &v in values {
        let k = (f64::from(v) * 8_388_608.0) as u64;
        let bin = if low {
            k as usize % cells
        } else {
            ((k * cells as u64) >> 23) as usize
        };
        counts[bin] += 1;
    }
    chi2_p(chi2_uniform(&counts), cells - 1)
}

/// Joint chi-square p-value of disjoint pairs `(a, b)` of 23-bit uniforms over
/// a `bins x bins` grid of their top bits.
pub fn pair_p(pairs: &[(f32, f32)], bins: usize) -> f64 {
    let mut counts = vec![0u64; bins * bins];
    for &(a, b) in pairs {
        let bin = |v: f32| ((f64::from(v) * bins as f64) as usize).min(bins - 1);
        counts[bin(a) * bins + bin(b)] += 1;
    }
    chi2_p(chi2_uniform(&counts), bins * bins - 1)
}

/// Kolmogorov-Smirnov p-value of `sample` against the CDF `cdf`.
pub fn ks_p(sample: &mut [f64], cdf: impl Fn(f64) -> f64) -> f64 {
    sample.sort_by(f64::total_cmp);
    let n = sample.len() as f64;
    let mut d = 0.0f64;
    for (i, &x) in sample.iter().enumerate() {
        let f = cdf(x);
        d = d.max(f - i as f64 / n).max((i as f64 + 1.0) / n - f);
    }
    let l = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * d;
    let mut p = 0.0;
    for j in 1..=100 {
        let term = (-2.0 * (j * j) as f64 * l * l).exp();
        p += if j % 2 == 1 { term } else { -term };
        if term < 1e-18 {
            break;
        }
    }
    (2.0 * p).clamp(0.0, 1.0)
}

/// The standard normal CDF.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
}

/// Two-sided z-score of `observed` against `expected` with deviation `sd`.
pub fn z_score(observed: f64, expected: f64, sd: f64) -> f64 {
    (observed - expected) / sd
}

/// Sample mean and (population) variance.
pub fn mean_var(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var)
}

/// Tile-image index of in-tile `(row, column)`.
pub fn inverse() -> [[usize; 32]; 32] {
    let mut inv = [[usize::MAX; 32]; 32];
    for i in 0..1024 {
        let (r, c) = tile_coord(i);
        inv[r][c] = i;
    }
    assert!(inv.iter().flatten().all(|&i| i != usize::MAX));
    inv
}

/// The model's `[rows, cols]` logical matrix of a draw `(base, role)`, each
/// element mapped by `f` from the tile's model words or units.
pub fn expected(dims: [usize; 2], base: u64, role: u32, target: Target, units: bool) -> Vec<u32> {
    let ct = dims[1].div_ceil(32);
    let inv = inverse();
    let tiles: Vec<Vec<u32>> = (0..dims[0].div_ceil(32) * ct)
        .map(|t| {
            let seed = tile_seed(base, t as u64, role);
            if units {
                tile_units(seed, target)
            } else {
                tile_words(seed, target)
            }
        })
        .collect();
    (0..dims[0] * dims[1])
        .map(|e| {
            let (r, c) = (e / dims[1], e % dims[1]);
            tiles[(r / 32) * ct + c / 32][inv[r % 32][c % 32]]
        })
        .collect()
}

pub fn units_of(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

/// `|device - reference|` allowed for `mean + std z`, derived term by term from
/// the programs' own bounds. With `x = 1 - u1` exact, `l = ln x` within
/// `LOG_BOUND` (relative), `r = sqrt(-2 l)` within one ulp (`2^-23` relative;
/// `-2 l` is exact), `c = cos(theta)` for the F32 `theta = fl(u2 * TWO_PI)`
/// within `COS_BOUND` (relative to `cos theta`), the product `r c`, `std *` and
/// `mean +` each one correct F32 rounding (`2^-24` relative).
pub fn normal_bound(z_ref: f64, out_ref: f64, std: f64) -> f64 {
    let u = 2f64.powi(-24);
    let r1 =
        (1.0 + LOG_BOUND / 2.0 + 1e-12) * (1.0 + 2.0 * u) * (1.0 + COS_BOUND) * (1.0 + u) - 1.0;
    z_ref.abs() * std.abs() * r1 + u * (std * z_ref).abs() * (1.0 + r1) + u * out_ref.abs() + 1e-30
}
