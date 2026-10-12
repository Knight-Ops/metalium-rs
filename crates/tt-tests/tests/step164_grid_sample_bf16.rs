//! BF16 `grid_sample_2d` rounds once (D2, found by `tensor_bf16`'s
//! `should_pad_reflection_grid_sample_2d`).
//!
//! Burn's default bilinear `grid_sample_2d` is a composition of element-wise ops. burn-tt did not
//! override it, so on a BF16 tensor every step (coordinates, reflection, weights, the weighted sum)
//! rounded to BF16, and BF16's 8 significant bits put the result more than one step from the
//! correctly rounded one: `1.3047` where the BF16 grid (`0.2001953`, `-0.8007813`) gives `1.3003`,
//! which is `1.296875` as BF16. `float_grid_sample_2d` is now overridden, so the generated wrapper
//! widens the operands to F32, composes there and rounds the result once, as every other half
//! precision operation does.
//!
//! Silicon only: ttsim refuses the PACR mode every BF16 result goes through (divergence row 31).
//!
//! The oracle is independent of burn-tt: bilinear sampling restated in f64 from the PyTorch
//! definition (zeros, border and reflection padding, `align_corners = false`), applied to the BF16
//! operands as the device holds them, then rounded to BF16 (nearest even) on the host. The F32
//! composition on the card is not f64, so a result may differ from the oracle by one BF16 step only
//! where the f64 value lies within `1e-5` of a rounding boundary; the gate counts those and bounds
//! them. Negative control: the default composition run in BF16 on the same operands
//! (`float_grid_sample_2d_ref`, bypassing the override) must differ from the oracle on the pinned
//! case, so a gate that a per-operation-rounded result could pass fails here.
#![cfg(feature = "silicon")]

use burn::tensor::ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode};
use burn::tensor::{DType, Tensor, TensorData, TensorPrimitive};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{with_device, Config};

/// f32 -> BF16 bits, nearest even; the value is rounded to f32 first, as the card holds it.
fn to_bf16(value: f64) -> u16 {
    let bits = (value as f32).to_bits();
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7fff + lsb) >> 16) as u16
}

fn from_bf16(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

/// Distance, relative to the value, from `v` to the nearest BF16 rounding boundary (a midpoint).
fn boundary_distance(v: f64) -> f64 {
    let r = from_bf16(to_bf16(v));
    let ulp = {
        let e = (v.abs().max(f64::MIN_POSITIVE)).log2().floor();
        2f64.powf(e - 7.0)
    };
    let below = (v - (r - ulp / 2.0)).abs();
    let above = (v - (r + ulp / 2.0)).abs();
    below.min(above) / v.abs().max(f64::MIN_POSITIVE)
}

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / ((1u64 << 31) as f64)
}

fn bf16_data(values: &[f64], shape: [usize; 4]) -> (Vec<f64>, TensorData) {
    let bits: Vec<u16> = values.iter().map(|v| to_bf16(*v)).collect();
    let held: Vec<f64> = bits.iter().map(|b| from_bf16(*b)).collect();
    let mut data = TensorData::new(bits, shape);
    data.dtype = DType::BF16;
    (held, data)
}

/// PyTorch's `grid_sample`, bilinear, `align_corners = false`, in f64. `image` is `[n, c, h, w]`,
/// `grid` is `[n, ho, wo, 2]`.
fn oracle(
    image: &[f64],
    grid: &[f64],
    [n, c, h, w]: [usize; 4],
    [ho, wo]: [usize; 2],
    padding: GridSamplePaddingMode,
) -> Vec<f64> {
    let reflect = |x: f64, size: usize| -> f64 {
        let (lo, hi) = (-0.5, size as f64 - 0.5);
        let span = hi - lo;
        let t = (x - lo).abs() % (2.0 * span);
        (if t <= span { t } else { 2.0 * span - t }) + lo
    };
    let mut out = Vec::with_capacity(n * c * ho * wo);
    for b in 0..n {
        for ch in 0..c {
            for i in 0..ho {
                for j in 0..wo {
                    let g = ((b * ho + i) * wo + j) * 2;
                    let mut x = ((grid[g] + 1.0) * w as f64 - 1.0) / 2.0;
                    let mut y = ((grid[g + 1] + 1.0) * h as f64 - 1.0) / 2.0;
                    match padding {
                        GridSamplePaddingMode::Zeros => {}
                        GridSamplePaddingMode::Border => {
                            x = x.clamp(0.0, w as f64 - 1.0);
                            y = y.clamp(0.0, h as f64 - 1.0);
                        }
                        GridSamplePaddingMode::Reflection => {
                            x = reflect(x, w).clamp(0.0, w as f64 - 1.0);
                            y = reflect(y, h).clamp(0.0, h as f64 - 1.0);
                        }
                    }
                    let (x0, y0) = (x.floor(), y.floor());
                    let (fx, fy) = (x - x0, y - y0);
                    let tap = |yy: f64, xx: f64| -> f64 {
                        if yy < 0.0 || xx < 0.0 || yy > h as f64 - 1.0 || xx > w as f64 - 1.0 {
                            return 0.0;
                        }
                        image[((b * c + ch) * h + yy as usize) * w + xx as usize]
                    };
                    out.push(
                        tap(y0, x0) * (1.0 - fx) * (1.0 - fy)
                            + tap(y0, x0 + 1.0) * fx * (1.0 - fy)
                            + tap(y0 + 1.0, x0) * (1.0 - fx) * fy
                            + tap(y0 + 1.0, x0 + 1.0) * fx * fy,
                    );
                }
            }
        }
    }
    out
}

fn options(padding: GridSamplePaddingMode) -> GridSampleOptions {
    GridSampleOptions::new(InterpolateMode::Bilinear).with_padding_mode(padding)
}

fn bits_of(tensor: Tensor<TtBackend, 4>) -> Vec<u16> {
    let mut data = tensor.into_data();
    assert_eq!(data.dtype, DType::BF16);
    data.dtype = DType::U16;
    data.to_vec().unwrap()
}

/// Burn's own test case: the pinned value, and the negative control.
#[test]
fn reflection_case_from_burns_suite_rounds_once_and_the_default_composition_does_not() {
    with_device(Config::default(), |d| {
        let (image, image_data) =
            bf16_data(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 1, 3, 3]);
        let (grid, grid_data) =
            bf16_data(&[0.0, 0.0, -1.0, 0.25, 1.0, 1.0, 0.2, -0.8], [1, 2, 2, 2]);
        let expected: Vec<u16> = oracle(
            &image,
            &grid,
            [1, 1, 3, 3],
            [2, 2],
            GridSamplePaddingMode::Reflection,
        )
        .into_iter()
        .map(to_bf16)
        .collect();
        // The pinned element: 1.3003 on the BF16 grid, which is 1.296875.
        assert_eq!(from_bf16(expected[3]), 1.296875);

        let tensor = Tensor::<TtBackend, 4>::from_data(image_data, (&d, DType::BF16));
        let grid = Tensor::<TtBackend, 4>::from_data(grid_data, (&d, DType::BF16));
        let before = tensor_traffic();
        let got = tensor
            .clone()
            .grid_sample_2d(grid.clone(), options(GridSamplePaddingMode::Reflection));
        match got.clone().into_primitive() {
            TensorPrimitive::Float(t) => assert!(t.computed_on_device()),
            _ => panic!("unexpected quantized tensor"),
        }
        assert_eq!(
            tensor_traffic().downloads,
            before.downloads,
            "the result was read back before the explicit readback"
        );
        assert_eq!(bits_of(got), expected);

        // Negative control: the default composition in BF16, on the same operands.
        let (TensorPrimitive::Float(image), TensorPrimitive::Float(grid)) =
            (tensor.into_primitive(), grid.into_primitive())
        else {
            panic!("unexpected quantized tensor")
        };
        let default = burn::tensor::ops::grid_sample::float_grid_sample_2d_ref::<TtBackend>(
            image,
            grid,
            options(GridSamplePaddingMode::Reflection),
        );
        let default = bits_of(Tensor::<TtBackend, 4>::from_primitive(
            TensorPrimitive::Float(default),
        ));
        assert_ne!(
            default, expected,
            "the default BF16 composition now matches the oracle: the control no longer controls"
        );
        assert_eq!(from_bf16(default[3]), 1.3046875);
    });
}

/// Every padding mode, random operands, in range and out of range.
#[test]
fn every_padding_mode_equals_the_f64_oracle_rounded_once() {
    with_device(Config::default(), |d| {
        let mut seed = 0x5eed_u64;
        let shape = [2, 3, 5, 6];
        let (ho, wo) = (4, 5);
        let image: Vec<f64> = (0..shape.iter().product::<usize>())
            .map(|_| lcg(&mut seed) * 8.0 - 4.0)
            .collect();
        let grid: Vec<f64> = (0..shape[0] * ho * wo * 2)
            .map(|_| lcg(&mut seed) * 3.2 - 1.6)
            .collect();
        let (image, image_data) = bf16_data(&image, shape);
        let (grid, grid_data) = bf16_data(&grid, [shape[0], ho, wo, 2]);
        let tensor = Tensor::<TtBackend, 4>::from_data(image_data, (&d, DType::BF16));
        let grid_t = Tensor::<TtBackend, 4>::from_data(grid_data, (&d, DType::BF16));
        let mut near = 0;
        let mut total = 0;
        for padding in [
            GridSamplePaddingMode::Zeros,
            GridSamplePaddingMode::Border,
            GridSamplePaddingMode::Reflection,
        ] {
            let exact = oracle(&image, &grid, shape, [ho, wo], padding);
            let got = bits_of(
                tensor
                    .clone()
                    .grid_sample_2d(grid_t.clone(), options(padding)),
            );
            assert_eq!(got.len(), exact.len());
            for (i, (g, e)) in got.iter().zip(&exact).enumerate() {
                total += 1;
                let want = to_bf16(*e);
                if *g != want {
                    // One step, and only where the oracle sits on a rounding boundary.
                    assert!(
                        boundary_distance(*e) < 1e-5 && (*g as i32 - want as i32).abs() == 1,
                        "{padding:?} element {i}: got {} want {} (f64 {e})",
                        from_bf16(*g),
                        from_bf16(want)
                    );
                    near += 1;
                }
            }
        }
        assert!(
            near * 50 <= total,
            "{near} of {total} elements sat on a boundary: the oracle is not agreeing"
        );
    });
}
