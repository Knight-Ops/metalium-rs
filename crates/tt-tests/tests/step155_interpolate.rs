//! `interpolate` and its gradient, nearest, bilinear, bicubic and Lanczos3, native on the card.
//!
//! Found by Burn's own backend tests (`burn-tt-conformance`): 22 tests were listed as refused by
//! name. The claim: the op is a fixed sparse linear map whose indices and weights depend only on the
//! shapes, so it is composed from device primitives (a word-copy gather of every output's taps, one
//! SFPU multiply per weight factor, the in-order SFPU row sum; `ops/interpolate.rs`), with no new
//! kernel, no host arithmetic and no readback.
//!
//! # The oracles
//!
//! None is an epsilon. Each bound is derived below from the arithmetic the card runs.
//!
//! * **Nearest** moves words: equal to Flex's, bit for bit, forward (NaN payloads, `-0.0`, a
//!   subnormal included) and, summing in the same order from `+0.0`, backward.
//! * **Bilinear, bicubic, Lanczos3 forward**, against Flex. The card rounds each of an output's
//!   (up to) `n` weight factors to f32, multiplies, and sums the `n` products in tap order from
//!   `+0.0`. With `u = 2^-24` and `γ_k = k u / (1 - k u)`, a term with `f` factors is within
//!   `γ_{2f}` of its exact value (each factor is rounded once and multiplied once), the sum adds
//!   `γ_{n-1}` of `S = Σ |x_i a_i|`, so the card is within `(γ_{2f} + γ_{n-1}(1 + γ_{2f})) S` of the
//!   exact result. `a` is the exact weight row, read off Flex itself (Flex run on unit impulses, so
//!   the oracle does not depend on `burn-tt`'s tables), and `n` counts its nonzero entries (a zero
//!   weight adds an exact zero). Flex rounds its f64 sum once (`u |y|`), except bilinear, which it
//!   evaluates in f32 like the card: that mode's bound is doubled.
//! * **Lanczos3 forward, independently of Flex**: the `image` crate's `resize(.., Lanczos3)`.
//!   It agrees with Burn only where its filter does (it widens its support when shrinking, and it
//!   samples at pixel centres): upsampling or equal size, `align_corners = false`. There, with the
//!   data inside (0, 1) (it clamps `f32` images to [0, 1]; the gate asserts nothing was clamped), its
//!   error against the exact filter is bounded as derived at [`image_error`].
//! * **Backward, all four modes**: the adjoint. The gradient is `Aᵀ g`, with `A` the forward map
//!   read off Flex as above; the oracle is that product in f64, and the card is within the same
//!   `(γ_{2f} + γ_{n-1}(1 + γ_{2f})) S_i` with `S_i = Σ_p |a_{pi} g_p|` over the `n` outputs that read
//!   input `i`. Flex has no Lanczos3 gradient, so the adjoint is its only oracle; for nearest,
//!   bilinear and bicubic Flex's own gradient is also compared, within both implementations' errors.
//!
//! # The negative controls
//!
//! Each comparison is run again against a wrong answer, which must fail it: the opposite
//! `align_corners` (where it changes the result), the neighbouring mode, a reversed row.
//!
//! Residency: no download and exactly one upload (the input), the result computed on the device.

use burn::tensor::backend::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, ModuleOps};
use burn::tensor::{DType, Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, with_report, TtBackend, TtDevice};
use image::imageops::FilterType;
use tt_tests::burn_device::{assert_native_model, with_device, Config};

const U: f64 = 1.0 / 16_777_216.0;

fn gamma(k: usize) -> f64 {
    k as f64 * U / (1.0 - k as f64 * U)
}

type Dims = [usize; 4];

/// `(input [n, c, h, w], output [h, w])`. The first five are Burn's own tests' shapes; the rest
/// are ragged against the 32 x 32 tile and whole, with unit extents, equal sizes and mixed
/// directions (up in one axis, down in the other).
const CASES: &[(Dims, [usize; 2])] = &[
    ([2, 1, 7, 5], [8, 7]),
    ([1, 1, 45, 14], [4, 6]),
    ([1, 1, 1, 6], [1, 9]),
    ([1, 1, 4, 4], [8, 8]),
    ([1, 1, 28, 4], [24, 2]),
    ([1, 3, 5, 33], [7, 33]),
    ([2, 2, 34, 3], [17, 6]),
    ([1, 1, 32, 32], [64, 32]),
    ([1, 1, 5, 1], [3, 1]),
    ([1, 1, 3, 3], [3, 3]),
    ([3, 1, 1, 1], [4, 5]),
    ([1, 1, 6, 6], [1, 1]),
];

fn all_modes() -> [InterpolateMode; 4] {
    [
        InterpolateMode::Nearest,
        InterpolateMode::Bilinear,
        InterpolateMode::Bicubic,
        InterpolateMode::Lanczos3,
    ]
}

fn options(mode: &InterpolateMode, align_corners: bool) -> InterpolateOptions {
    InterpolateOptions::new(mode.clone()).with_align_corners(align_corners)
}

/// Multiplies applied to each tap of a mode.
fn factors(mode: &InterpolateMode) -> usize {
    match mode {
        InterpolateMode::Nearest => 0,
        InterpolateMode::Bilinear | InterpolateMode::Bicubic => 2,
        InterpolateMode::Lanczos3 => 1,
    }
}

/// Reproducible values in [-4, 4) with all 24 mantissa bits in use.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005) ^ 0x9E37_79B9;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 16_777_216.0) * 8.0 - 4.0
        })
        .collect()
}

fn tt(d: &TtDevice, x: &[f32], dims: Dims, dtype: DType) -> Tensor<TtBackend, 4> {
    Tensor::from_data(TensorData::new(x.to_vec(), dims), (d, dtype))
}

fn flex(x: &[f32], dims: Dims) -> Tensor<Flex, 4> {
    Tensor::from_data(TensorData::new(x.to_vec(), dims), &FlexDevice)
}

fn host<B: Backend>(t: Tensor<B, 4>) -> Vec<f32> {
    t.into_data().convert::<f32>().to_vec::<f32>().unwrap()
}

/// `B::interpolate_backward` through the primitive API: `x` supplies only its shape.
fn gradient<B: Backend>(
    x: Tensor<B, 4>,
    g: Tensor<B, 4>,
    size: [usize; 2],
    o: InterpolateOptions,
) -> Tensor<B, 4> {
    let (TensorPrimitive::Float(x), TensorPrimitive::Float(g)) =
        (x.into_primitive(), g.into_primitive())
    else {
        unreachable!("float tensors")
    };
    Tensor::from_primitive(TensorPrimitive::Float(
        <B as ModuleOps<B>>::interpolate_backward(x, g, size, o),
    ))
}

fn computed_on_device(x: &Tensor<TtBackend, 4>) -> bool {
    match x.clone().into_primitive() {
        TensorPrimitive::Float(t) => t.computed_on_device(),
        TensorPrimitive::QFloat(_) => false,
    }
}

#[cfg(feature = "silicon")]
fn dtype_tt(d: &TtDevice, x: &[f32], dims: Dims, dtype: DType) -> Tensor<TtBackend, 4> {
    tt(d, x, dims, dtype)
}

#[cfg(feature = "silicon")]
fn flex_dtype(x: &[f32], dims: Dims, dtype: DType) -> Tensor<Flex, 4> {
    Tensor::from_data(TensorData::new(x.to_vec(), dims), (&FlexDevice, dtype))
}

/// The forward map as a dense `[outputs, inputs]` matrix of one plane, read off Flex: row `p`
/// column `i` is output `p` for a unit impulse at input `i` (all impulses in one batch).
fn matrix(h: usize, w: usize, size: [usize; 2], o: &InterpolateOptions) -> Vec<f64> {
    let n = h * w;
    let mut x = vec![0f32; n * n];
    for i in 0..n {
        x[i * n + i] = 1.0;
    }
    let y = host(interpolate(flex(&x, [n, 1, h, w]), size, o.clone()));
    let s = size[0] * size[1];
    let mut a = vec![0f64; s * n];
    for i in 0..n {
        for p in 0..s {
            a[p * n + i] = y[i * s + p] as f64;
        }
    }
    a
}

/// The error of the card's tap sum: `n` taps of `f` factors each, over magnitudes summing to `s`.
fn card_error(f: usize, n: usize, s: f64) -> f64 {
    let gf = gamma(2 * f);
    (gf + gamma(n.saturating_sub(1)) * (1.0 + gf)) * s * (1.0 + U)
}

/// Per output: the bound on `|card - Flex|` of a forward pass (see the header).
fn forward_bound(
    mode: &InterpolateMode,
    a: &[f64],
    n_in: usize,
    x: &[f32],
    want: &[f32],
) -> Vec<f64> {
    let s_out = a.len() / n_in;
    let planes = x.len() / n_in;
    let mut out = Vec::with_capacity(want.len());
    for j in 0..planes {
        for p in 0..s_out {
            let (mut s, mut n) = (0.0f64, 0);
            for (i, &w) in a[p * n_in..(p + 1) * n_in].iter().enumerate() {
                if w != 0.0 {
                    n += 1;
                    s += (w * x[j * n_in + i] as f64).abs();
                }
            }
            let y = want[j * s_out + p] as f64;
            let mut b = card_error(factors(mode), n, s) + 2f64.powi(-48) * s + U * y.abs();
            if matches!(mode, InterpolateMode::Bilinear) {
                b *= 2.0; // Flex evaluates bilinear in f32, as the card does
            }
            out.push(b);
        }
    }
    out
}

fn within(got: &[f32], want: &[f32], bound: &[f64]) -> Option<usize> {
    (0..got.len()).find(|&i| {
        if want[i].is_nan() {
            !got[i].is_nan()
        } else if want[i].is_infinite() {
            got[i] != want[i]
        } else {
            (got[i] as f64 - want[i] as f64).abs() > bound[i]
        }
    })
}

#[track_caller]
fn assert_within(what: &str, got: &[f32], want: &[f32], bound: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = within(got, want, bound) {
        panic!(
            "{what}: element {i}: {} against {}, |difference| {:e} over the bound {:e}",
            got[i],
            want[i],
            (got[i] as f64 - want[i] as f64).abs(),
            bound[i]
        );
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn case_name(mode: &InterpolateMode, align: bool, dims: Dims, size: [usize; 2]) -> String {
    format!("{mode:?} align_corners={align} {dims:?} -> {size:?}")
}

/// Modes whose card arithmetic is Flex's, operation for operation: a move (nearest) or f32 products and
/// sums in the same tap order (bilinear). The card is held to equality there, not to a bound.
fn is_flexs_arithmetic(mode: &InterpolateMode) -> bool {
    matches!(mode, InterpolateMode::Nearest | InterpolateMode::Bilinear)
}

/// Every row of `x` (rows of `width`) reversed: a wrong answer of the right shape and values.
fn reversed_rows(x: &[f32], width: usize) -> Vec<f32> {
    x.chunks(width)
        .flat_map(|r| r.iter().rev().copied())
        .collect()
}

/// The card against Flex, for one mode over every case and both coordinate maps.
fn check_forward(mode: InterpolateMode) {
    with_device(Config::default(), |d| {
        let (mut controls, mut cases) = (0, 0);
        for &(dims, size) in CASES {
            let [n, c, h, w] = dims;
            for align in [true, false] {
                let o = options(&mode, align);
                let name = case_name(&mode, align, dims, size);
                let x = values(n * c * h * w, (h * 31 + w) as u64);
                let want = host(interpolate(flex(&x, dims), size, o.clone()));
                let got = host(interpolate(tt(&d, &x, dims, DType::F32), size, o.clone()));
                let a = matrix(h, w, size, &o);
                let bound = if is_flexs_arithmetic(&mode) {
                    vec![0.0; want.len()]
                } else {
                    forward_bound(&mode, &a, h * w, &x, &want)
                };
                if matches!(mode, InterpolateMode::Nearest) {
                    assert_eq!(bits(&got), bits(&want), "{name}");
                }
                assert_within(&name, &got, &want, &bound);
                // Controls: a wrong answer, where it differs from Flex's by more than the bound,
                // must differ from the card's by more too. The opposite coordinate map (nearest
                // ignores it) and every row reversed.
                let mut wrongs = vec![reversed_rows(&want, size[1])];
                if !matches!(mode, InterpolateMode::Nearest) {
                    wrongs.push(host(interpolate(
                        flex(&x, dims),
                        size,
                        options(&mode, !align),
                    )));
                }
                for wrong in wrongs {
                    if within(&wrong, &want, &bound).is_some() {
                        assert!(within(&got, &wrong, &bound).is_some(), "{name}: control");
                        controls += 1;
                    }
                }
                cases += 1;
            }
        }
        assert_eq!(cases, 2 * CASES.len());
        assert!(
            controls >= CASES.len(),
            "{mode:?}: only {controls} controls exercised"
        );
    });
}

#[test]
fn nearest_forward_is_flexs_bit_for_bit() {
    check_forward(InterpolateMode::Nearest);
}

#[test]
fn bilinear_forward_matches_flex_within_the_derived_bound() {
    check_forward(InterpolateMode::Bilinear);
}

#[test]
fn bicubic_forward_matches_flex_within_the_derived_bound() {
    check_forward(InterpolateMode::Bicubic);
}

#[test]
fn lanczos3_forward_matches_flex_within_the_derived_bound() {
    check_forward(InterpolateMode::Lanczos3);
}

/// Per input element: the adjoint `Aᵀ g` in f64, the sum `S_i` of the magnitudes and the number of
/// outputs that read it (nonzero weights).
fn adjoint(a: &[f64], n_in: usize, g: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let s_out = a.len() / n_in;
    let planes = g.len() / s_out;
    let (mut r, mut s, mut n) = (
        vec![0.0; planes * n_in],
        vec![0.0; planes * n_in],
        vec![0; planes * n_in],
    );
    for j in 0..planes {
        for p in 0..s_out {
            let gp = g[j * s_out + p] as f64;
            for i in 0..n_in {
                let w = a[p * n_in + i];
                if w != 0.0 {
                    r[j * n_in + i] += w * gp;
                    s[j * n_in + i] += (w * gp).abs();
                    n[j * n_in + i] += 1;
                }
            }
        }
    }
    (r, s, n)
}

/// The card's gradient against the adjoint (all modes) and against Flex's own (the modes it has).
fn check_backward(mode: InterpolateMode) {
    with_device(Config::default(), |d| {
        let mut controls = 0;
        for &(dims, size) in CASES {
            let [n, c, h, w] = dims;
            let gdims = [n, c, size[0], size[1]];
            for align in [true, false] {
                let o = options(&mode, align);
                let name = format!("backward {}", case_name(&mode, align, dims, size));
                let x = values(n * c * h * w, 7);
                let g = values(n * c * size[0] * size[1], (size[0] * 17 + size[1]) as u64);
                let got = host(gradient(
                    tt(&d, &x, dims, DType::F32),
                    tt(&d, &g, gdims, DType::F32),
                    size,
                    o.clone(),
                ));
                let a = matrix(h, w, size, &o);
                let (reference, sums, counts) = adjoint(&a, h * w, &g);
                let f = factors(&mode);
                let bound: Vec<f64> = (0..got.len())
                    .map(|i| card_error(f, counts[i], sums[i]) + 2f64.powi(-48) * sums[i])
                    .collect();
                // Against the exact adjoint (an f64 value, so measured in f64).
                for i in 0..got.len() {
                    assert!(
                        (got[i] as f64 - reference[i]).abs() <= bound[i],
                        "{name}: element {i}: {} against {}, bound {:e}",
                        got[i],
                        reference[i],
                        bound[i]
                    );
                }
                // Controls: the adjoint of the opposite coordinate map (nearest ignores it), and of
                // the gradient with every row reversed.
                let mut wrongs = vec![adjoint(&a, h * w, &reversed_rows(&g, size[1])).0];
                if !matches!(mode, InterpolateMode::Nearest) {
                    wrongs.push(adjoint(&matrix(h, w, size, &options(&mode, !align)), h * w, &g).0);
                }
                for wrong in wrongs {
                    if (0..got.len()).any(|i| (wrong[i] - reference[i]).abs() > bound[i]) {
                        assert!(
                            (0..got.len()).any(|i| (got[i] as f64 - wrong[i]).abs() > bound[i]),
                            "{name}: control"
                        );
                        controls += 1;
                    }
                }
                // Flex's gradient, for the modes it has: equal where the arithmetic is the same,
                // else within both implementations' errors (Flex's bicubic rounds each f64 tap
                // product to f32 before its f32 sum: `u` a term and `γ_{n-1}` the sum).
                if !matches!(mode, InterpolateMode::Lanczos3) {
                    let want = host(gradient(flex(&x, dims), flex(&g, gdims), size, o.clone()));
                    let both: Vec<f64> = (0..got.len())
                        .map(|i| match mode {
                            InterpolateMode::Bicubic => {
                                bound[i]
                                    + (U + gamma(counts[i].saturating_sub(1)) * (1.0 + U))
                                        * sums[i]
                                        * (1.0 + U)
                            }
                            _ => 0.0,
                        })
                        .collect();
                    assert_within(&format!("{name} against Flex's"), &got, &want, &both);
                }
            }
        }
        assert!(
            controls >= CASES.len(),
            "{mode:?}: only {controls} controls exercised"
        );
    });
}

#[test]
fn nearest_gradient_matches_the_adjoint_and_flex() {
    check_backward(InterpolateMode::Nearest);
}

#[test]
fn bilinear_gradient_matches_the_adjoint_and_flex() {
    check_backward(InterpolateMode::Bilinear);
}

#[test]
fn bicubic_gradient_matches_the_adjoint_and_flex() {
    check_backward(InterpolateMode::Bicubic);
}

#[test]
fn lanczos3_gradient_matches_the_adjoint() {
    check_backward(InterpolateMode::Lanczos3);
}

// ---- Lanczos3 against the `image` crate ---------------------------------------------------------

/// One plane resized by the `image` crate (`f32` luma, the vertical pass then the horizontal).
fn image_resize(
    x: &[f32],
    [h, w]: [usize; 2],
    [oh, ow]: [usize; 2],
    filter: FilterType,
) -> Vec<f32> {
    let img =
        image::ImageBuffer::<image::Luma<f32>, Vec<f32>>::from_raw(w as u32, h as u32, x.to_vec())
            .expect("a plane");
    image::imageops::resize(&img, ow as u32, oh as u32, filter).into_raw()
}

/// The unnormalized Lanczos3 weights' sum over an axis position's in-range taps, in f64: the divisor
/// both Burn's and `image`'s normalization uses (the taps are the six at `floor(c) - 2 ..= floor(c) + 3`,
/// `c = (o + 1/2) in/out - 1/2`; `image` adds taps whose weight is zero).
fn lanczos3_raw_sum(o: usize, in_size: usize, out_size: usize) -> f64 {
    let sinc = |t: f64| {
        if t == 0.0 {
            1.0
        } else {
            (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)
        }
    };
    let c = (o as f64 + 0.5) * in_size as f64 / out_size as f64 - 0.5;
    let c0 = c.floor() as isize;
    (c0 - 2..=c0 + 3)
        .filter(|i| (0..in_size as isize).contains(i))
        .map(|i| {
            let x = c - i as f64;
            if x.abs() >= 3.0 {
                0.0
            } else {
                sinc(x) * sinc(x / 3.0)
            }
        })
        .sum()
}

/// The error of `image`'s f32 Lanczos3 against the exact filter, per output, for data within `xmax`
/// and an output whose weights have magnitudes summing to `r`.
///
/// `image` computes each raw weight as `sinc(x) sinc(x / 3)` in f32: `sinc(t) = sin(a) / a` with
/// `a = fl(π t)` (relative error `u`), whose `sin` (taken within one ulp, `u`) moves by at most `a u`
/// and which is then divided (`u`): at most `3u` absolute for `|sinc| <= 1` once `a >= 1`, and a
/// relative `3u` below it. Two sincs and their product: `4u + 4u + u = 9u` per raw weight. The six
/// raw weights (each at most 1, so magnitudes summing to at most 6) are summed in f32: the divisor
/// is off by `6 · 9u + 6 · 6u = 90u`, and is at least `SIGMA_MIN`, which the gate checks over every
/// axis position it uses. A normalized weight `w / Σ`, of magnitude at most `r` (an axis's sum is
/// at most the 2D sum, the other axis's normalized weights summing to 1), is off by at most
/// `(9u + r · 90u) / SIGMA_MIN + r u` (the divisor's error scaled by the weight, and the
/// division's rounding). A 2D weight is a product of two, so the output is off by `2 ε r xmax` for
/// the weights, and `2 γ_6 r xmax` for the two f32 passes' sums.
fn image_error(r: f64, xmax: f64) -> f64 {
    let eps = (9.0 * U + r * 90.0 * U) / SIGMA_MIN + r * U;
    (2.0 * eps + 2.0 * gamma(6)) * r * xmax
}

const SIGMA_MIN: f64 = 0.4;

#[test]
fn lanczos3_forward_matches_the_image_crate_where_their_filters_coincide() {
    let o = options(&InterpolateMode::Lanczos3, false);
    let used: Vec<_> = CASES
        .iter()
        .filter(|(dims, size)| size[0] >= dims[2] && size[1] >= dims[3])
        .collect();
    assert!(used.len() >= 6, "too few cases in the domain");
    // The error bound's one assumption: the divisor never shrinks below SIGMA_MIN.
    for &&(dims, size) in &used {
        for (len, out) in [(dims[2], size[0]), (dims[3], size[1])] {
            for p in 0..out {
                let raw = lanczos3_raw_sum(p, len, out);
                assert!(
                    raw >= SIGMA_MIN,
                    "raw weight sum {raw} at {p} of {len} -> {out}"
                );
            }
        }
    }
    with_device(Config::default(), |d| {
        let (mut controls, mut worst, mut worst_r) = (0, 0.0f64, 0.0f64);
        for &&(dims, size) in &used {
            let [n, c, h, w] = dims;
            // Inside (0, 1) with room for the filter's overshoot: `image` clamps f32 images to [0, 1].
            let x: Vec<f32> = values(n * c * h * w, 3)
                .iter()
                .map(|v| 0.5 + v / 20.0)
                .collect();
            let xmax = 0.7;
            let got = host(interpolate(tt(&d, &x, dims, DType::F32), size, o.clone()));
            let a = matrix(h, w, size, &o);
            let s_out = size[0] * size[1];
            for j in 0..n * c {
                let plane = &x[j * h * w..(j + 1) * h * w];
                let theirs = image_resize(plane, [h, w], size, FilterType::Lanczos3);
                assert!(
                    theirs.iter().all(|v| *v > 0.0 && *v < 1.0),
                    "image clamped {dims:?} -> {size:?}"
                );
                let wrong = image_resize(plane, [h, w], size, FilterType::CatmullRom);
                let mut rejected = false;
                for p in 0..s_out {
                    let row = &a[p * h * w..(p + 1) * h * w];
                    let (mut s, mut r, mut nz) = (0.0f64, 0.0f64, 0);
                    for (i, &wt) in row.iter().enumerate() {
                        if wt != 0.0 {
                            nz += 1;
                            r += wt.abs();
                            s += (wt * plane[i] as f64).abs();
                        }
                    }
                    worst_r = worst_r.max(r);
                    // The card against the exact filter (the weights read off Flex carry `u` too),
                    // and `image` against it.
                    let bound =
                        card_error(1, nz, s) + (2f64.powi(-48) + U) * s + image_error(r, xmax);
                    let got = got[j * s_out + p] as f64;
                    let diff = (got - theirs[p] as f64).abs();
                    assert!(
                        diff <= bound,
                        "{dims:?} -> {size:?} plane {j} output {p}: {got} against {}",
                        theirs[p]
                    );
                    worst = worst.max(diff / bound);
                    rejected |= (got - wrong[p] as f64).abs() > bound;
                }
                // Control: another filter's resize must not satisfy the bound. (Not where every
                // normalized filter agrees: the same size, which copies, or a single input pixel.)
                if size != [h, w] && h * w > 1 {
                    assert!(rejected, "{dims:?} -> {size:?}: control");
                    controls += 1;
                }
            }
        }
        assert!(controls >= 5, "{controls} controls");
        eprintln!("Lanczos3 against image: worst |difference| / bound = {worst:.3}, largest weight magnitude sum {worst_r:.3}");
    });
}

// ---- residency -----------------------------------------------------------------------------------

#[test]
fn interpolation_and_its_gradient_compute_on_the_device_and_download_nothing() {
    with_device(Config::default(), |d| {
        let (dims, size) = ([2, 3, 7, 5], [9, 12]);
        let [n, c, h, w] = dims;
        let gdims = [n, c, size[0], size[1]];
        for mode in all_modes() {
            let o = options(&mode, false);
            let x = values(n * c * h * w, 5);
            let g = values(n * c * size[0] * size[1], 6);
            let x_t = tt(&d, &x, dims, DType::F32);
            let before = tensor_traffic();
            let (y, report) = with_report(|| interpolate(x_t.clone(), size, o.clone()));
            let after = tensor_traffic();
            assert_native_model(&report);
            assert!(
                computed_on_device(&y),
                "{mode:?}: the result was not computed on the device"
            );
            assert_eq!(after.downloads, before.downloads, "{mode:?}: a download");
            // The input crosses once, lazily on first use, and nothing else does: the tables are
            // built on the card from metadata (`server::metadata`), never uploaded.
            assert_eq!(after.uploads - before.uploads, 1, "{mode:?}: uploads");
            assert_eq!(
                after.uploaded - before.uploaded,
                (x.len() * 4) as u64,
                "{mode:?}: upload bytes"
            );

            let g_t = tt(&d, &g, gdims, DType::F32);
            let before = tensor_traffic();
            let (dx, report) = with_report(|| gradient(x_t.clone(), g_t.clone(), size, o.clone()));
            let after = tensor_traffic();
            assert_native_model(&report);
            assert!(
                computed_on_device(&dx),
                "{mode:?}: the gradient was not computed on the device"
            );
            assert_eq!(
                after.downloads, before.downloads,
                "{mode:?}: a gradient download"
            );
            // Only the gradient crosses (the input supplies its shape).
            assert_eq!(
                after.uploaded - before.uploaded,
                (g.len() * 4) as u64,
                "{mode:?}: gradient upload bytes"
            );
            // Reading the results is the only download.
            let before = tensor_traffic();
            let _ = (host(y), host(dx));
            assert_eq!(tensor_traffic().downloads - before.downloads, 2, "{mode:?}");
        }
    });
}

// ---- views, padding, specials, dtypes, empties ---------------------------------------------------

/// The card against Flex on a tensor made by a view: a permuted NHWC input and a sliced one, so the
/// strided path (no repack of the input first) runs.
#[test]
fn views_of_the_input_interpolate_like_their_contents() {
    with_device(Config::default(), |d| {
        let (n, c, h, w, size) = (2, 3, 5, 6, [8, 9]);
        for mode in all_modes() {
            let o = options(&mode, false);
            // NHWC data, permuted to NCHW on both backends.
            let nhwc = values(n * h * w * c, 11);
            let mut nchw = vec![0f32; nhwc.len()];
            for b in 0..n {
                for y in 0..h {
                    for x in 0..w {
                        for k in 0..c {
                            nchw[((b * c + k) * h + y) * w + x] =
                                nhwc[((b * h + y) * w + x) * c + k];
                        }
                    }
                }
            }
            let got = host(interpolate(
                tt(&d, &nhwc, [n, h, w, c], DType::F32).permute([0, 3, 1, 2]),
                size,
                o.clone(),
            ));
            let want = host(interpolate(
                flex(&nhwc, [n, h, w, c]).permute([0, 3, 1, 2]),
                size,
                o.clone(),
            ));
            let a = matrix(h, w, size, &o);
            let bound = if is_flexs_arithmetic(&mode) {
                vec![0.0; want.len()]
            } else {
                forward_bound(&mode, &a, h * w, &nchw, &want)
            };
            assert_within(&format!("{mode:?} permuted"), &got, &want, &bound);

            // A window of a larger tensor.
            let big = values(n * c * (h + 2) * (w + 3), 12);
            let got = host(interpolate(
                tt(&d, &big, [n, c, h + 2, w + 3], DType::F32).slice([
                    0..n,
                    0..c,
                    1..h + 1,
                    2..w + 2,
                ]),
                size,
                o.clone(),
            ));
            let want = host(interpolate(
                flex(&big, [n, c, h + 2, w + 3]).slice([0..n, 0..c, 1..h + 1, 2..w + 2]),
                size,
                o.clone(),
            ));
            let mut cropped = Vec::new();
            for plane in 0..n * c {
                for y in 1..h + 1 {
                    for x in 2..w + 2 {
                        cropped.push(big[(plane * (h + 2) + y) * (w + 3) + x]);
                    }
                }
            }
            let bound = if is_flexs_arithmetic(&mode) {
                vec![0.0; want.len()]
            } else {
                forward_bound(&mode, &a, h * w, &cropped, &want)
            };
            assert_within(&format!("{mode:?} sliced"), &got, &want, &bound);
        }
    });
}

/// A result feeding a reduction and a matmul over a ragged shape: the tile's padding must not appear.
#[test]
fn a_resized_result_feeds_ragged_reductions_and_matmuls() {
    with_device(Config::default(), |d| {
        for (dims, size) in [([1, 2, 7, 5], [9, 6]), ([1, 1, 5, 33], [7, 45])] {
            let [n, c, h, w] = dims;
            let (oh, ow) = (size[0], size[1]);
            // Small integers: nearest copies them, so the row sums and the matmul (TF32 holds them)
            // are exact whatever their order.
            let x: Vec<f32> = (0..n * c * h * w).map(|i| (i % 7) as f32 - 3.0).collect();
            let b: Vec<f32> = (0..ow * 5).map(|i| (i % 5) as f32 - 2.0).collect();
            let run = |y: Tensor<TtBackend, 4>| {
                let rows = y.clone().sum_dim(3);
                let product =
                    y.reshape([n * c * oh, ow])
                        .matmul(Tensor::<TtBackend, 2>::from_data(
                            TensorData::new(b.clone(), [ow, 5]),
                            &d,
                        ));
                (host(rows), product.into_data().to_vec::<f32>().unwrap())
            };
            let o = options(&InterpolateMode::Nearest, true);
            let got = run(interpolate(tt(&d, &x, dims, DType::F32), size, o.clone()));
            let y = flex(&x, dims);
            let wy = interpolate(y, size, o);
            let rows = host(wy.clone().sum_dim(3));
            let product = wy
                .reshape([n * c * oh, ow])
                .matmul(Tensor::<Flex, 2>::from_data(
                    TensorData::new(b.clone(), [ow, 5]),
                    &FlexDevice,
                ))
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(got.0, rows, "{dims:?}: row sums");
            assert_eq!(got.1, product, "{dims:?}: matmul");

            // Bilinear equals Flex's values, so only the reduction's order differs: either side is
            // within `γ_{ow-1} Σ|y|` of the exact row sum.
            let o = options(&InterpolateMode::Bilinear, false);
            let x = values(n * c * h * w, 21);
            let y = host(interpolate(flex(&x, dims), size, o.clone()));
            let got = host(interpolate(tt(&d, &x, dims, DType::F32), size, o.clone()).sum_dim(3));
            let want = host(interpolate(flex(&x, dims), size, o).sum_dim(3));
            let bound: Vec<f64> = y
                .chunks(ow)
                .map(|row| 2.0 * gamma(ow - 1) * row.iter().map(|v| v.abs() as f64).sum::<f64>())
                .collect();
            assert_within(&format!("{dims:?} bilinear row sums"), &got, &want, &bound);
        }
    });
}

/// Values that expose a misread word: nearest moves them untouched, and bilinear, which does the
/// same f32 arithmetic as Flex, propagates them identically (a zero weight still multiplies a NaN).
#[test]
fn special_values_move_and_propagate_as_flex_does() {
    with_device(Config::default(), |d| {
        let dims = [1, 2, 5, 6];
        let mut x = values(2 * 5 * 6, 31);
        let subnormal = f32::from_bits(0x0000_1234);
        for (i, v) in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            subnormal,
            -subnormal,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            f32::from_bits(0x7fc0_0001),
            0.0,
        ]
        .into_iter()
        .enumerate()
        {
            x[i * 5 + 1] = v;
        }
        // A gradient sums from +0.0, so a -0.0 contribution gathers to +0.0 (Flex's scatter does the
        // same), including where each input has a single contributor and a plain move would keep it.
        for (size, name) in [([5, 6], "one contributor each"), ([10, 12], "several")] {
            let g: Vec<f32> = (0..2 * size[0] * size[1])
                .map(|i| {
                    if i % 3 == 0 {
                        -0.0
                    } else {
                        (i % 5) as f32 - 2.0
                    }
                })
                .collect();
            let o = options(&InterpolateMode::Nearest, true);
            let gd = [1, 2, size[0], size[1]];
            let got = host(gradient(
                tt(&d, &x, dims, DType::F32),
                tt(&d, &g, gd, DType::F32),
                size,
                o.clone(),
            ));
            let want = host(gradient(flex(&x, dims), flex(&g, gd), size, o));
            assert_eq!(
                bits(&got),
                bits(&want),
                "nearest gradient of signed zeros, {name}"
            );
        }
        for align in [true, false] {
            let o = options(&InterpolateMode::Nearest, align);
            let got = host(interpolate(tt(&d, &x, dims, DType::F32), [7, 9], o.clone()));
            let want = host(interpolate(flex(&x, dims), [7, 9], o));
            assert_eq!(bits(&got), bits(&want), "nearest, align_corners={align}");
            // Without NaN, infinities and subnormals (the SFPU flushes those operands), bilinear's
            // special values are the f32 ones; with them, NaN and infinities are class-equal.
            let o = options(&InterpolateMode::Bilinear, align);
            let finite: Vec<f32> = x
                .iter()
                .map(|v| {
                    if v.is_finite() && v.abs() >= f32::MIN_POSITIVE {
                        *v
                    } else {
                        1.5
                    }
                })
                .collect();
            let got = host(interpolate(
                tt(&d, &finite, dims, DType::F32),
                [7, 9],
                o.clone(),
            ));
            let want = host(interpolate(flex(&finite, dims), [7, 9], o.clone()));
            assert_eq!(got, want, "bilinear extremes, align_corners={align}");
            let mut poisoned = values(2 * 5 * 6, 32);
            poisoned[7] = f32::NAN;
            poisoned[19] = f32::INFINITY;
            poisoned[40] = f32::NEG_INFINITY;
            let got = host(interpolate(
                tt(&d, &poisoned, dims, DType::F32),
                [7, 9],
                o.clone(),
            ));
            let want = host(interpolate(flex(&poisoned, dims), [7, 9], o));
            for i in 0..got.len() {
                assert_eq!(
                    got[i].is_nan(),
                    want[i].is_nan(),
                    "bilinear element {i}: {} against {}",
                    got[i],
                    want[i]
                );
                assert!(
                    got[i].is_nan() || got[i] == want[i],
                    "bilinear element {i}: {} against {}",
                    got[i],
                    want[i]
                );
            }
        }
    });
}

/// Silicon-only, like every BF16 narrowing gate: the result's F32 -> BF16 pack is PACR mode `0x105`, which
/// ttsim refuses (`ttsim-divergence.md`, "BF16 late narrowing").
///
/// BF16 widens, computes in f32 and rounds once, as Flex does: nearest and bilinear equal Flex's
/// values (the same f32 arithmetic, rounded the same way), the others within the f32 bound plus the
/// one bf16 rounding of a result that the two f32 values straddle (`2^-8 |y|`).
#[cfg(feature = "silicon")]
#[test]
fn bf16_widens_computes_in_f32_and_rounds_once() {
    with_device(Config::default(), |d| {
        let (dims, size) = ([2, 2, 7, 5], [8, 9]);
        let [n, c, h, w] = dims;
        let x = values(n * c * h * w, 41);
        let quantized = host(flex_dtype(&x, dims, DType::BF16));
        let g = values(n * c * size[0] * size[1], 42);
        let gq = host(flex_dtype(&g, [n, c, size[0], size[1]], DType::BF16));
        for mode in all_modes() {
            for align in [true, false] {
                let o = options(&mode, align);
                let name = case_name(&mode, align, dims, size);
                let got = host(interpolate(
                    dtype_tt(&d, &x, dims, DType::BF16),
                    size,
                    o.clone(),
                ));
                let want = host(interpolate(
                    flex_dtype(&x, dims, DType::BF16),
                    size,
                    o.clone(),
                ));
                let a = matrix(h, w, size, &o);
                let exact = is_flexs_arithmetic(&mode);
                let bound: Vec<f64> = if exact {
                    vec![0.0; want.len()]
                } else {
                    forward_bound(&mode, &a, h * w, &quantized, &want)
                        .iter()
                        .zip(&want)
                        .map(|(b, y)| b + 2f64.powi(-8) * (*y as f64).abs())
                        .collect()
                };
                assert_within(&format!("bf16 {name}"), &got, &want, &bound);

                if !matches!(mode, InterpolateMode::Lanczos3) {
                    let gdims = [n, c, size[0], size[1]];
                    let got = host(gradient(
                        dtype_tt(&d, &x, dims, DType::BF16),
                        dtype_tt(&d, &g, gdims, DType::BF16),
                        size,
                        o.clone(),
                    ));
                    let want = host(gradient(
                        flex_dtype(&x, dims, DType::BF16),
                        flex_dtype(&g, gdims, DType::BF16),
                        size,
                        o,
                    ));
                    let (_, sums, counts) = adjoint(&a, h * w, &gq);
                    let bound: Vec<f64> = (0..got.len())
                        .map(|i| {
                            if exact {
                                0.0
                            } else {
                                2.0 * card_error(factors(&mode), counts[i], sums[i])
                                    + 2f64.powi(-7) * (want[i] as f64).abs()
                            }
                        })
                        .collect();
                    assert_within(&format!("bf16 gradient {name}"), &got, &want, &bound);
                }
            }
        }
    });
}

/// Empty batches and outputs give empty results of the right shape, and a gradient of nothing is zeros.
#[test]
fn empty_extents_give_results_of_the_right_shape() {
    with_device(Config::default(), |d| {
        for mode in all_modes() {
            let o = options(&mode, false);
            let y = interpolate(tt(&d, &[], [0, 2, 4, 4], DType::F32), [8, 8], o.clone());
            assert_eq!(y.dims(), [0, 2, 8, 8], "{mode:?}");
            let x = values(16, 51);
            let y = interpolate(tt(&d, &x, [1, 1, 4, 4], DType::F32), [0, 5], o.clone());
            assert_eq!(y.dims(), [1, 1, 0, 5], "{mode:?}");
            let dx = gradient(
                tt(&d, &x, [1, 1, 4, 4], DType::F32),
                tt(&d, &[], [1, 1, 0, 5], DType::F32),
                [0, 5],
                o.clone(),
            );
            assert_eq!(host(dx), vec![0.0; 16], "{mode:?}: a gradient of nothing");
            let dx = gradient(
                tt(&d, &[], [0, 2, 4, 4], DType::F32),
                tt(&d, &[], [0, 2, 8, 8], DType::F32),
                [8, 8],
                o,
            );
            assert_eq!(dx.dims(), [0, 2, 4, 4], "{mode:?}");
        }
    });
}
