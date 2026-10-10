//! `prelu` with a one-element weight stays on the device.
//!
//! `burn::tensor::activation::prelu` (and `burn::nn::PRelu`, whose default is one
//! shared slope) reshapes a weight of shape `[1]` to `[1; D]` before it reaches the
//! backend's `prelu`. This gate holds that the whole path is resident for tensors of
//! every rank 1..=4, including shapes ragged against the 32-element tile: nothing is
//! uploaded or downloaded by the op, the result is computed on the device, and a
//! reduction or matmul of it afterwards reads the right padding.
//!
//! The oracle is Flex's `if x >= 0 { x } else { a * x }` with the device's multiply
//! (`tt_isa::numerics::mul_bh`: not fused, denormals flushed, NaN canonicalised).
//! Flex's IEEE result is also compared wherever the model rounds as IEEE does.
//! Negative controls: the branch swapped (the slope on the non-negative side), and
//! the identity slope, must both differ from the device on the gate's data.
use burn::tensor::{activation, Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend};
use tt_isa::numerics;
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

const SPECIALS: [u32; 12] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x807f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc1_2345,
    0xffc5_4321,
    0x7f7f_ffff,
    0xff7f_ffff,
];

fn words(n: usize, seed: u64, specials: bool) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            let w = lcg(&mut s);
            if specials && i % 4 == 0 {
                SPECIALS[w as usize % SPECIALS.len()]
            } else {
                let exponent = 120 + (w >> 24) as u32 % 16;
                (((w >> 9) as u32 & 1) << 31) | (exponent << 23) | (w as u32 & 0x7f_ffff)
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

/// Flex's definition with the device's multiply.
fn model(x: &[u32], alpha: u32, swapped: bool) -> Vec<u32> {
    x.iter()
        .map(|&b| {
            let keep = f32::from_bits(b) >= 0.0;
            if keep != swapped {
                b
            } else {
                numerics::mul_bh(alpha, b)
            }
        })
        .collect()
}

fn case<const D: usize>(shape: [usize; D], alpha: f32, seed: u64, specials: bool) {
    let n: usize = shape.iter().product();
    let bits = words(n, seed, specials);
    let want = model(&bits, alpha.to_bits(), false);
    assert_ne!(
        want,
        model(&bits, alpha.to_bits(), true),
        "negative control: the branch swapped must differ"
    );
    if alpha != 1.0 {
        assert_ne!(want, bits, "negative control: the identity must differ");
    }
    with_device(Config::default(), |d| {
        let x = Tensor::<TtBackend, D>::from_data(TensorData::new(float_data(&bits), shape), &d)
            .to_device(&d);
        let a = Tensor::<TtBackend, 1>::from_data([alpha], &d).to_device(&d);
        let before = tensor_traffic();
        let (y, report) = burn_tt::with_report(|| activation::prelu(x.clone(), a.clone()));
        let during = tensor_traffic() - before;
        assert_eq!(
            (during.uploads, during.downloads),
            (0, 0),
            "{shape:?}: {during:?}"
        );
        assert_native_model(&report);
        let op = report.op("prelu").expect("prelu was not reported");
        assert!(op.on_device > 0);
        assert_eq!((op.downloads, op.staged, op.on_host), (0, 0, 0));
        let TensorPrimitive::Float(p) = y.clone().into_primitive() else {
            unreachable!()
        };
        assert!(p.computed_on_device(), "{shape:?}");
        assert_eq!(y.dims(), shape);
        let got = read_bits(y);
        let mismatches: Vec<_> = (0..n).filter(|&i| got[i] != want[i]).collect();
        assert!(
            mismatches.is_empty(),
            "{shape:?} alpha {alpha} specials {specials}: device vs model at {:?}: {:?} vs {:?} (inputs {:?})",
            mismatches.iter().take(4).collect::<Vec<_>>(),
            mismatches.iter().take(4).map(|&i| got[i]).collect::<Vec<_>>(),
            mismatches.iter().take(4).map(|&i| want[i]).collect::<Vec<_>>(),
            mismatches.iter().take(4).map(|&i| bits[i]).collect::<Vec<_>>(),
        );
        // Flex, where the model's multiply is IEEE's (a NaN as a class).
        let fx =
            Tensor::<Flex, D>::from_data(TensorData::new(float_data(&bits), shape), &FlexDevice);
        let fa = Tensor::<Flex, 1>::from_data([alpha], &FlexDevice);
        let flex = read_bits(activation::prelu(fx, fa));
        let mut compared = 0;
        for i in 0..n {
            let ieee = (alpha * f32::from_bits(bits[i])).to_bits();
            let model_mul = numerics::mul_bh(alpha.to_bits(), bits[i]);
            let nan = |u: u32| f32::from_bits(u).is_nan();
            let rounds_alike = model_mul == ieee || (nan(model_mul) && nan(ieee));
            if rounds_alike {
                compared += 1;
                assert!(
                    flex[i] == got[i] || (nan(flex[i]) && nan(got[i])),
                    "{shape:?} element {i}: Flex {:#x}, device {:#x}, input {:#x}",
                    flex[i],
                    got[i],
                    bits[i]
                );
            }
        }
        assert!(
            compared > n / 2,
            "the Flex comparison covered {compared}/{n}"
        );
    });
}

#[test]
fn one_element_weight_prelu_is_resident_and_exact_at_every_rank() {
    for specials in [false, true] {
        case::<1>([37], 0.25, 1, specials);
        case::<2>([5, 3], -1.5, 2, specials);
        case::<2>([33, 35], 0.25, 3, specials);
        case::<3>([2, 3, 5], 0.0, 4, specials);
        case::<4>([2, 3, 4, 5], 2.0, 5, specials);
    }
}

/// `burn::nn::PRelu`'s default is one shared slope, a `[1]` parameter: the module's
/// forward is the same path.
#[test]
fn prelu_module_default_slope_is_resident() {
    use burn::nn::PReluConfig;
    with_device(Config::default(), |d| {
        let layer = PReluConfig::new().init::<TtBackend>(&d);
        let bits = words(7 * 35, 9, false);
        let x = Tensor::<TtBackend, 2>::from_data(TensorData::new(float_data(&bits), [7, 35]), &d)
            .to_device(&d);
        let before = tensor_traffic();
        let (y, report) = burn_tt::with_report(|| layer.forward(x.clone()));
        let during = tensor_traffic() - before;
        assert_eq!(during.downloads, 0, "{during:?}");
        assert_native_model(&report);
        assert!(y.clone().into_primitive().tensor().computed_on_device());
        assert_eq!(read_bits(y), model(&bits, 0.25f32.to_bits(), false));
    });
}

/// A ragged result feeds a reduction and a matmul, and a trace replays changed
/// inputs. Integer-valued data and a power-of-two slope, so every product and sum is
/// exact whatever its order.
#[test]
fn ragged_prelu_feeds_reductions_and_replays_changed_inputs() {
    with_device(Config::default(), |d| {
        let (rows, cols) = (5usize, 7usize);
        let ints = |seed: u64| -> Vec<f32> {
            let mut s = seed | 1;
            (0..rows * cols)
                .map(|_| (lcg(&mut s) % 33) as f32 - 16.0)
                .collect()
        };
        let alpha = 0.5f32;
        let activated = |x: &[f32]| -> Vec<f32> {
            x.iter()
                .map(|&v| if v >= 0.0 { v } else { alpha * v })
                .collect()
        };
        let want = |x: &[f32]| -> Vec<f32> {
            let y = activated(x);
            let sums: Vec<f32> = (0..cols)
                .map(|c| (0..rows).map(|r| y[r * cols + c]).sum())
                .collect();
            let mut mm = Vec::new();
            for r in 0..rows {
                let s: f32 = (0..cols).map(|c| y[r * cols + c]).sum();
                mm.extend([s, s]);
            }
            let mut out = sums;
            out.extend(mm);
            out
        };
        let first = ints(1);
        let a = Tensor::<TtBackend, 1>::from_data([alpha], &d).to_device(&d);
        let ones = Tensor::<TtBackend, 2>::ones([cols, 2], &d).to_device(&d);
        let input =
            Tensor::<TtBackend, 2>::from_data(TensorData::new(first.clone(), [rows, cols]), &d)
                .mul_scalar(1.0);
        let TensorPrimitive::Float(primitive) = input.clone().into_primitive() else {
            unreachable!()
        };
        let before = tensor_traffic();
        let ((trace, got), report) = burn_tt::with_report(|| {
            burn_tt::Trace::capture(&primitive, || {
                let y = activation::prelu(input.clone(), a.clone());
                let sums = y.clone().sum_dim(0).reshape([cols]);
                let mm = y.matmul(ones.clone()).reshape([rows * 2]);
                let out = Tensor::cat(vec![sums, mm], 0);
                let TensorPrimitive::Float(out) = out.into_primitive() else {
                    unreachable!()
                };
                out
            })
            .unwrap()
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().uploads, before.uploads);
        assert_eq!(got, want(&first));
        drop(input);
        for seed in [2u64, 3] {
            let next = ints(seed);
            assert_eq!(
                trace.run(next.clone()).unwrap(),
                want(&next),
                "replay {seed}"
            );
        }
    });
}
