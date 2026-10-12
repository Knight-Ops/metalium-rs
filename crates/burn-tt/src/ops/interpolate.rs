//! `interpolate` and its gradient as a fixed sparse linear map, composed from device primitives.
//!
//! Every source index and weight depends only on the input and output extents, the mode and
//! `align_corners`: never on the data. The host computes those tables (metadata, as the pooling
//! windows are) and the card does the rest:
//!
//! 1. one device word copy (`mapped_native`) gathers every output's taps into a `[taps, outputs]`
//!    matrix, padding slots reading a one-element zero tensor;
//! 2. one SFPU multiply per weight factor (bilinear and bicubic: the x factor, then the y factor,
//!    as burn-flex applies them; Lanczos3: one combined, renormalized factor; nearest: none);
//! 3. the SFPU row sum, which adds from `+0.0`, top to bottom, tile after tile: the order in which
//!    burn-flex's gradient scatter adds, and the order the taps are listed here.
//!
//! Nearest forward is the copy alone, so it moves words (NaN payloads, `-0.0`, subnormals) exactly.
//! The gradient is the transpose of the forward map: each input element lists the output elements
//! that read it, in the order burn-flex's scatter visits them. Lanczos3 has no burn-flex gradient;
//! this is its adjoint (`docs/plans/burn-op-notes.md`).
//!
//! What it does not do: it does not matmul by an interpolation matrix (F32 matmul truncates its
//! operands to TF32), and it makes one pass of up to 36 taps rather than a separable pair, so the
//! host stages `taps * N * C * outputs` addresses ([`MAX_SOURCES`]); larger resizes are refused by
//! name.
use super::pool::constant;
use super::*;
use burn_backend::ops::{InterpolateMode, InterpolateOptions};

/// The most gathered words one call stages on the host (`taps * N * C * positions`).
const MAX_SOURCES: usize = 1 << 24;

/// Keys' cubic convolution coefficient, as burn-flex.
const CUBIC_A: f64 = -0.75;

/// One tap of one position: where it reads in the plane (`None`: a padding slot reading zero) and
/// the factors it is multiplied by, in order.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Tap {
    src: Option<usize>,
    factors: [f32; 2],
}

const PAD: Tap = Tap {
    src: None,
    factors: [0.0; 2],
};

/// The taps of every position of one plane. `taps[k * positions + p]` is tap `k` of position `p`.
#[derive(Debug)]
struct Plan {
    /// Positions of the result plane.
    positions: usize,
    /// Taps per position.
    k: usize,
    /// Multiplies applied to each tap (0, 1 or 2).
    factors: usize,
    /// Elements in the plane the taps read.
    plane: usize,
    taps: Vec<Tap>,
}

fn coord_ratio(in_size: usize, out_size: usize, align_corners: bool) -> f64 {
    if align_corners {
        (in_size as f64 - 1.0) / (out_size.max(1) - 1).max(1) as f64
    } else {
        in_size as f64 / out_size as f64
    }
}

fn map_coord(out_coord: usize, ratio: f64, align_corners: bool) -> f64 {
    if align_corners {
        out_coord as f64 * ratio
    } else {
        (out_coord as f64 + 0.5) * ratio - 0.5
    }
}

fn cubic_weight(t: f64, a: f64) -> f64 {
    let t_abs = t.abs();
    if t_abs < 1.0 {
        ((a + 2.0) * t_abs - (a + 3.0)) * t_abs * t_abs + 1.0
    } else if t_abs < 2.0 {
        ((a * t_abs - 5.0 * a) * t_abs + 8.0 * a) * t_abs - 4.0 * a
    } else {
        0.0
    }
}

fn lanczos3_weight(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    if x.abs() >= 3.0 {
        return 0.0;
    }
    let pi = std::f64::consts::PI;
    let pi_x = pi * x;
    let pi_x_over_3 = pi_x / 3.0;
    (pi_x.sin() * pi_x_over_3.sin()) / (pi_x * pi_x_over_3)
}

/// `min(floor(o * in / out), in - 1)`; `align_corners` plays no part (burn-flex's legacy nearest).
fn nearest_map(in_size: usize, out_size: usize) -> Vec<usize> {
    let ratio = in_size as f64 / out_size as f64;
    (0..out_size)
        .map(|o| ((o as f64 * ratio).floor() as usize).min(in_size - 1))
        .collect()
}

/// Taps per position and multiplies per tap of a mode.
fn shape_of(mode: &InterpolateMode) -> (usize, usize) {
    match mode {
        InterpolateMode::Nearest => (1, 0),
        InterpolateMode::Bilinear => (4, 2),
        InterpolateMode::Bicubic => (16, 2),
        InterpolateMode::Lanczos3 => (36, 1),
    }
}

/// Refuse, before any table is built, a resize whose forward gather alone would stage more than
/// [`MAX_SOURCES`] host addresses (the gradient's is of the same order).
fn check_size(x: &TtTensor, op: &str, planes: usize, output: [usize; 2], mode: &InterpolateMode) {
    let staged = [planes, output[0], output[1], shape_of(mode).0]
        .into_iter()
        .try_fold(1usize, usize::checked_mul);
    if staged.is_none_or(|n| n > MAX_SOURCES) {
        fail(
            op,
            format_args!(
                "stages {planes} planes x {output:?} outputs x {} taps of host addresses, over \
                 the {MAX_SOURCES} limit (a separable two-pass is not implemented); {}",
                shape_of(mode).0,
                context(x)
            ),
        );
    }
}

/// The forward taps for an `[h, w]` plane resized to `[oh, ow]`, in burn-flex's term order.
fn forward(
    [h, w]: [usize; 2],
    [oh, ow]: [usize; 2],
    mode: &InterpolateMode,
    align_corners: bool,
) -> Plan {
    let (yr, xr) = (
        coord_ratio(h, oh, align_corners),
        coord_ratio(w, ow, align_corners),
    );
    let positions = oh * ow;
    let (k, factors) = shape_of(mode);
    let mut taps = vec![PAD; k * positions];
    let mut set = |tap: usize, p: usize, src: Option<usize>, factors: [f32; 2]| {
        taps[tap * positions + p] = Tap { src, factors };
    };
    match mode {
        InterpolateMode::Nearest => {
            let (ys, xs) = (nearest_map(h, oh), nearest_map(w, ow));
            for (y, &iy) in ys.iter().enumerate() {
                for (x, &ix) in xs.iter().enumerate() {
                    set(0, y * ow + x, Some(iy * w + ix), [0.0; 2]);
                }
            }
        }
        InterpolateMode::Bilinear => {
            // (low, high, weight) per coordinate; the weight rounds to f32 once, as `T::from`.
            let axis = |o: usize, size: usize, ratio: f64| {
                let c = map_coord(o, ratio, align_corners);
                let low = c.floor().max(0.0) as usize;
                assert!(low < size, "bilinear source out of range");
                (
                    low,
                    (low + 1).min(size - 1),
                    (c - low as f64).max(0.0) as f32,
                )
            };
            for y in 0..oh {
                let (y_low, y_high, wy) = axis(y, h, yr);
                for x in 0..ow {
                    let (x_low, x_high, wx) = axis(x, w, xr);
                    let (nx, ny) = (1.0f32 - wx, 1.0f32 - wy);
                    let p = y * ow + x;
                    set(0, p, Some(y_low * w + x_low), [nx, ny]);
                    set(1, p, Some(y_low * w + x_high), [wx, ny]);
                    set(2, p, Some(y_high * w + x_low), [nx, wy]);
                    set(3, p, Some(y_high * w + x_high), [wx, wy]);
                }
            }
        }
        InterpolateMode::Bicubic => {
            // Four (clamped index, weight) pairs per coordinate, offsets -1..=2 around its floor.
            let axis = |o: usize, size: usize, ratio: f64| {
                let c = map_coord(o, ratio, align_corners);
                let c0 = c.floor() as isize;
                let mut taps = [(0usize, 0.0f32); 4];
                for (slot, d) in (-1..=2isize).enumerate() {
                    let index = (c0 + d).clamp(0, size as isize - 1) as usize;
                    let t = (c - c0 as f64) - d as f64;
                    taps[slot] = (index, cubic_weight(t, CUBIC_A) as f32);
                }
                taps
            };
            let xs: Vec<_> = (0..ow).map(|x| axis(x, w, xr)).collect();
            for y in 0..oh {
                let ys = axis(y, h, yr);
                for (x, xt) in xs.iter().enumerate() {
                    for (dy, &(iy, wy)) in ys.iter().enumerate() {
                        for (dx, &(ix, wx)) in xt.iter().enumerate() {
                            set(dy * 4 + dx, y * ow + x, Some(iy * w + ix), [wx, wy]);
                        }
                    }
                }
            }
        }
        InterpolateMode::Lanczos3 => {
            // Six (index if in range, f64 weight) per coordinate, offsets -2..=3 around its floor.
            let axis = |o: usize, size: usize, ratio: f64| {
                let c = map_coord(o, ratio, align_corners);
                let c0 = c.floor();
                let mut taps = [(None, 0.0f64); 6];
                for (slot, d) in (-2..=3isize).enumerate() {
                    let index = c0 as isize + d;
                    if (0..size as isize).contains(&index) {
                        taps[slot] = (Some(index as usize), lanczos3_weight(c - (c0 + d as f64)));
                    }
                }
                taps
            };
            let xs: Vec<_> = (0..ow).map(|x| axis(x, w, xr)).collect();
            for y in 0..oh {
                let ys = axis(y, h, yr);
                for (x, xt) in xs.iter().enumerate() {
                    // The 2D weights and their sum, accumulated as burn-flex does (rows outer).
                    let mut sum = 0.0f64;
                    for &(iy, wy) in &ys {
                        for &(ix, wx) in xt {
                            if iy.is_some() && ix.is_some() {
                                sum += wy * wx;
                            }
                        }
                    }
                    for (ky, &(iy, wy)) in ys.iter().enumerate() {
                        for (kx, &(ix, wx)) in xt.iter().enumerate() {
                            if let (Some(iy), Some(ix)) = (iy, ix) {
                                let weight = wy * wx;
                                let weight = if sum != 0.0 { weight / sum } else { weight };
                                set(
                                    ky * 6 + kx,
                                    y * ow + x,
                                    Some(iy * w + ix),
                                    [weight as f32, 0.0],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Plan {
        positions,
        k,
        factors,
        plane: h * w,
        taps,
    }
}

/// The transpose of `fwd` for an input plane of `inputs` elements: each input element lists the
/// output positions that read it, in the order burn-flex's scatter visits them (position, then tap).
/// Short lists are padded with zero-source slots; their `+0.0` leaves a sum from `+0.0` unchanged.
fn transpose(fwd: &Plan, inputs: usize) -> Plan {
    let mut lists = vec![Vec::new(); inputs];
    for p in 0..fwd.positions {
        for k in 0..fwd.k {
            let tap = fwd.taps[k * fwd.positions + p];
            if let Some(i) = tap.src {
                lists[i].push(Tap {
                    src: Some(p),
                    factors: tap.factors,
                });
            }
        }
    }
    let k = lists.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let mut taps = vec![PAD; k * inputs];
    for (i, list) in lists.iter().enumerate() {
        for (slot, &tap) in list.iter().enumerate() {
            taps[slot * inputs + i] = tap;
        }
    }
    Plan {
        positions: inputs,
        k,
        factors: fwd.factors,
        plane: fwd.positions,
        taps,
    }
}

/// `plan` applied to every one of `planes` planes of `x` (F32 on the card), giving `out_shape`.
/// `forward`: a plan of one untouched tap per output is then the gather alone, moving words as they
/// are; a gradient always sums, from `+0.0` as Flex's scatter does (`-0.0` gathers to `+0.0`).
fn apply(
    x: &TtTensor,
    plan: &Plan,
    planes: usize,
    out_shape: [usize; 4],
    op: &str,
    forward: bool,
) -> TtTensor {
    let device = x.device;
    let (k, s) = (plan.k, plan.positions);
    let m = planes * s;
    if k.checked_mul(m).is_none_or(|n| n > MAX_SOURCES) {
        fail(
            op,
            format_args!(
                "stages {k} taps x {m} outputs of host addresses, over the {MAX_SOURCES} limit \
                 (a separable two-pass is not implemented); {}",
                context(x)
            ),
        );
    }
    let padded = plan.taps.iter().any(|t| t.src.is_none());
    let zero = padded.then(|| native_zeros([1, 1].into(), &device, DType::F32).expect("zeros"));
    let inputs: Vec<&TtTensor> = std::iter::once(x).chain(zero.as_ref()).collect();
    let mut sources = Vec::with_capacity(k * m);
    for tap in 0..k {
        for plane in 0..planes {
            for p in 0..s {
                sources.push(match plan.taps[tap * s + p].src {
                    Some(i) => (0, plane * plan.plane + i),
                    None => (1, 0),
                });
            }
        }
    }
    if forward && plan.factors == 0 && k == 1 {
        return mapped_native(&inputs, sources, out_shape.to_vec());
    }
    let mut packed = mapped_native(&inputs, sources, vec![k, m]);
    for f in 0..plan.factors {
        let table = constant(
            device,
            plan.taps.iter().map(|t| t.factors[f]).collect(),
            [k, s],
        );
        // The weights repeat across planes: copy them out on the card, never per plane on the host.
        let weights = if planes == 1 {
            table
        } else {
            let sources = (0..k)
                .flat_map(|tap| {
                    (0..planes).flat_map(move |_| (0..s).map(move |p| (0, tap * s + p)))
                })
                .collect();
            mapped_native(&[&table], sources, vec![k, m])
        };
        packed = float::float_mul(packed, weights);
    }
    let sum = float::float_sum_dim(packed, 0);
    repack_shape(&sum, out_shape.to_vec())
}

/// Validate an NCHW operand and return its extents; refuse by name what cannot be done. An empty
/// tensor has nothing to store, so only a nonempty one must be resident.
fn nchw(x: &TtTensor, op: &str) -> [usize; 4] {
    let shape = x.shape().to_vec();
    let nonempty = !shape.contains(&0);
    if shape.len() != 4
        || !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
        || (nonempty && !x.is_storable())
    {
        fail(
            op,
            format_args!(
                "requires a resident F32, BF16 or F16 NCHW tensor: {}",
                context(x)
            ),
        );
    }
    shape.try_into().unwrap()
}

/// The input plane of a resize must have something to sample (burn-flex asserts the same).
fn sampled([n, c, h, w]: [usize; 4], x: &TtTensor, op: &str) -> [usize; 4] {
    if h == 0 || w == 0 {
        fail(
            op,
            format_args!("input spatial dimensions must be > 0: {}", context(x)),
        );
    }
    [n, c, h, w]
}

fn empty(x: &TtTensor, shape: [usize; 4]) -> TtTensor {
    let dtype = match x.dtype() {
        DType::BF16 => burn_backend::FloatDType::BF16,
        DType::F16 => burn_backend::FloatDType::F16,
        _ => burn_backend::FloatDType::F32,
    };
    float::float_empty(shape.into(), &x.device, dtype)
}

/// Resize the spatial extents of an NCHW tensor to `output_size`.
///
/// Refuses by name: F16 or other dtypes, a non-NCHW tensor, an empty input plane, and a resize that
/// would stage more than [`MAX_SOURCES`] host addresses. BF16 widens, computes in F32 and rounds
/// once at the end, as burn-flex does.
pub fn interpolate(x: TtTensor, output_size: [usize; 2], options: InterpolateOptions) -> TtTensor {
    let [n, c, h, w] = sampled(nchw(&x, "interpolate"), &x, "interpolate");
    let out_shape = [n, c, output_size[0], output_size[1]];
    if out_shape.contains(&0) {
        return empty(&x, out_shape);
    }
    let dtype = x.dtype();
    check_size(&x, "interpolate", n * c, output_size, &options.mode);
    let plan = forward([h, w], output_size, &options.mode, options.align_corners);
    let out = apply(
        &float_compute_input(x),
        &plan,
        n * c,
        out_shape,
        "interpolate",
        true,
    );
    cast_native(out, dtype)
}

/// The gradient of [`interpolate`] with respect to `x`, of which only the shape is read.
///
/// Every mode, including Lanczos3 (burn-flex has none; this is the adjoint of the forward map).
pub fn interpolate_backward(
    x: TtTensor,
    grad: TtTensor,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> TtTensor {
    let [n, c, h, w] = sampled(nchw(&x, "interpolate_backward"), &x, "interpolate_backward");
    let g = nchw(&grad, "interpolate_backward");
    if g != [n, c, output_size[0], output_size[1]]
        || x.device != grad.device
        || x.dtype() != grad.dtype()
    {
        fail(
            "interpolate_backward",
            format_args!(
                "x=({}), grad=({}), output_size={output_size:?}: the gradient must be the \
                 resized shape, on the same device and dtype",
                context(&x),
                context(&grad)
            ),
        );
    }
    if output_size.contains(&0) || n == 0 || c == 0 {
        return native_zeros([n, c, h, w].into(), &x.device, x.dtype())
            .unwrap_or_else(|| empty(&x, [n, c, h, w]));
    }
    let dtype = grad.dtype();
    check_size(
        &x,
        "interpolate_backward",
        n * c,
        output_size,
        &options.mode,
    );
    let fwd = forward([h, w], output_size, &options.mode, options.align_corners);
    let plan = transpose(&fwd, h * w);
    let out = apply(
        &float_compute_input(grad),
        &plan,
        n * c,
        [n, c, h, w],
        "interpolate_backward",
        false,
    );
    cast_native(out, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODES: [InterpolateMode; 4] = [
        InterpolateMode::Nearest,
        InterpolateMode::Bilinear,
        InterpolateMode::Bicubic,
        InterpolateMode::Lanczos3,
    ];

    /// `plan` on one plane in f64: the sum of every tap's source times its factors.
    fn host_apply(plan: &Plan, x: &[f64]) -> Vec<f64> {
        (0..plan.positions)
            .map(|p| {
                (0..plan.k)
                    .map(|k| {
                        let tap = plan.taps[k * plan.positions + p];
                        tap.src.map_or(0.0, |i| {
                            (0..plan.factors).fold(x[i], |v, f| v * tap.factors[f] as f64)
                        })
                    })
                    .sum()
            })
            .collect()
    }

    fn sample(n: usize, salt: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (((i * 2654435761 + salt * 40503) % 1000) as f64 - 500.0) / 37.0)
            .collect()
    }

    #[test]
    fn nearest_floors_and_ignores_align_corners() {
        // Burn's 1d test: six values to nine take 1.541, 1.541, -0.2934, ...
        assert_eq!(nearest_map(6, 9), [0, 0, 1, 2, 2, 3, 4, 4, 5]);
        let a = forward([1, 6], [1, 9], &InterpolateMode::Nearest, true);
        let b = forward([1, 6], [1, 9], &InterpolateMode::Nearest, false);
        assert_eq!(a.taps, b.taps);
        // Downsampling past the edge never reads out of range.
        assert_eq!(nearest_map(14, 6), [0, 2, 4, 7, 9, 11]);
    }

    #[test]
    fn corner_aligned_coordinates_span_the_input() {
        // 5 -> 7 columns: 4/6 per step, so the second output is 2/3 of the way to the second input.
        let plan = forward([1, 5], [1, 7], &InterpolateMode::Bilinear, true);
        let tap = |k: usize, p: usize| plan.taps[k * plan.positions + p];
        assert_eq!(tap(0, 1).src, Some(0));
        assert_eq!(tap(1, 1).src, Some(1));
        assert_eq!(tap(1, 1).factors[0], (2.0f64 / 3.0) as f32);
        // The last output lands exactly on the last input: both taps read it, the weight is 1.
        assert_eq!(tap(0, 6).src, Some(4));
        assert_eq!(tap(1, 6).src, Some(4));
        assert_eq!(tap(1, 6).factors[0], 0.0);
        // One output: coordinate 0, whatever the input extent.
        let one = forward([1, 5], [1, 1], &InterpolateMode::Bilinear, true);
        assert_eq!(one.taps[0].src, Some(0));
        assert_eq!(one.taps[1].factors[0], 0.0);
    }

    #[test]
    fn half_pixel_coordinates_below_zero_read_the_edge_with_zero_weight() {
        // 4 -> 8: output 0 is at -0.25, so its weight is clamped to 0 and both taps are input 0 / 1.
        let plan = forward([1, 4], [1, 8], &InterpolateMode::Bilinear, false);
        let (a, b) = (plan.taps[0], plan.taps[plan.positions]);
        assert_eq!((a.src, b.src), (Some(0), Some(1)));
        assert_eq!((a.factors[0], b.factors[0]), (1.0, 0.0));
        // Bicubic clamps the indices instead: floor(-0.25) is -1, so the taps are at -2, -1, 0, 1
        // and the first three all read input 0.
        let cubic = forward([1, 4], [1, 8], &InterpolateMode::Bicubic, false);
        let srcs: Vec<_> = (0..4).map(|dx| cubic.taps[dx * 8].src).collect();
        assert_eq!(srcs, [Some(0), Some(0), Some(0), Some(1)]);
    }

    #[test]
    fn every_mode_preserves_a_constant_up_to_f32_weight_rounding() {
        for mode in MODES {
            for align in [true, false] {
                for (i, o) in [([7, 5], [8, 7]), ([45, 14], [4, 6]), ([4, 4], [8, 8])] {
                    let plan = forward(i, o, &mode, align);
                    let out = host_apply(&plan, &vec![1.0; plan.plane]);
                    for v in out {
                        // Nearest picks one tap; the others sum their weights to 1 (bicubic and
                        // bilinear by construction, Lanczos3 by renormalization).
                        assert!((v - 1.0).abs() < 1e-5, "{mode:?} {align} {i:?}->{o:?}: {v}");
                    }
                }
            }
        }
    }

    #[test]
    fn lanczos3_skips_out_of_range_taps_and_renormalizes() {
        // Output 0 of a corner-aligned resize is input 0: only taps at offsets 0..=3 are in range.
        let plan = forward([1, 8], [1, 16], &InterpolateMode::Lanczos3, true);
        let live: Vec<_> = (0..36)
            .filter(|&k| plan.taps[k * plan.positions].src.is_some())
            .collect();
        // Rows: only ky = 2 (the single row, offset 0) is in range; columns kx = 2..=5.
        assert_eq!(live, [14, 15, 16, 17]);
        let total: f32 = live
            .iter()
            .map(|&k| plan.taps[k * plan.positions].factors[0])
            .sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_gradient_plan_is_the_transpose_of_the_forward_plan() {
        // <A x, g> = <x, A^T g>, in f64 over the very tables the device is given.
        for mode in MODES {
            for align in [true, false] {
                for (i, o) in [
                    ([7, 5], [8, 7]),
                    ([45, 14], [4, 6]),
                    ([4, 4], [8, 8]),
                    ([1, 6], [1, 9]),
                    ([5, 1], [3, 1]),
                    ([3, 3], [3, 3]),
                ] {
                    let fwd = forward(i, o, &mode, align);
                    let bwd = transpose(&fwd, i[0] * i[1]);
                    assert_eq!((bwd.positions, bwd.plane), (i[0] * i[1], o[0] * o[1]));
                    let (x, g) = (sample(fwd.plane, 1), sample(fwd.positions, 2));
                    let lhs: f64 = host_apply(&fwd, &x)
                        .iter()
                        .zip(&g)
                        .map(|(a, b)| a * b)
                        .sum();
                    let rhs: f64 = host_apply(&bwd, &g)
                        .iter()
                        .zip(&x)
                        .map(|(a, b)| a * b)
                        .sum();
                    assert!(
                        (lhs - rhs).abs() <= 1e-9 * (1.0 + lhs.abs()),
                        "{mode:?} {align} {i:?}->{o:?}: {lhs} != {rhs}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_gradient_lists_contributors_in_scatter_order() {
        // Nearest 2 -> 4 columns: input 0 is read by outputs 0 and 1, input 1 by 2 and 3.
        let bwd = transpose(&forward([1, 2], [1, 4], &InterpolateMode::Nearest, true), 2);
        assert_eq!(bwd.k, 2);
        let src = |k: usize, i: usize| bwd.taps[k * 2 + i].src;
        assert_eq!(
            [src(0, 0), src(1, 0), src(0, 1), src(1, 1)],
            [Some(0), Some(1), Some(2), Some(3)]
        );
        // With several taps the order is Flex's scatter order: output by output, a tap at a
        // time. Bilinear 3 -> 5 columns (coordinates 0, .5, 1, 1.5, 2): input 1 is read by
        // outputs 0 and 1 through taps b and d (high), then by outputs 2 and 3 through a and c.
        let bwd = transpose(
            &forward([1, 3], [1, 5], &InterpolateMode::Bilinear, true),
            3,
        );
        assert_eq!(bwd.k, 8);
        let order: Vec<_> = (0..8).map(|slot| bwd.taps[slot * 3 + 1].src).collect();
        let want = [0, 0, 1, 1, 2, 2, 3, 3].map(Some);
        assert_eq!(order, want);
        // An input nobody reads (4 -> 2 columns skips input 1) is one padding row.
        let bwd = transpose(&forward([1, 4], [1, 2], &InterpolateMode::Nearest, true), 4);
        assert_eq!(bwd.taps[1].src, None);
    }
}
