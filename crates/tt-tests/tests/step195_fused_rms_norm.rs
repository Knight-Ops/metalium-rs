//! RMSNorm as one op on `Tt` (`docs/plans/decode-small-ops.md` S3, `docs/plans/kernel-fusion.md`
//! Phase 4): burn-fusion's chain fuser recognises Burn's `RmsNorm` up to its scale -- `x * x`,
//! the mean over the last dimension, `+ eps`, `sqrt`, `x / rms` -- and runs it as one kernel
//! (`tt_kernels::sfpu::norm`) in place of five device ops and their intermediate tensors.
//!
//! The claims:
//! - **Bit for bit `TtBackend`'s ops**, as Burn's `RmsNorm` module (its scale applied after, as
//!   its own op) and as the bare normalisation: decode's `[1, 1, d]` at SmolLM2's widths, a
//!   ragged width, several tile rows, rank 3, `eps` 1e-5 and 1e-6 (both SmolLM2 presets), over
//!   signed zeros, subnormals, and a NaN or an infinity in one element of a row. The kernel's
//!   own unit gate (`sfpu::norm`) holds its programs to the composed ops on the specification's
//!   model, with a negative control on the sum's order.
//! - **One op:** the normalisation runs as `tt_rms_norm` and runs no `float_mean_dim`,
//!   `float_sqrt` or `float_div`.
//! - **Negative controls**, each correct and not fused: a mean over another dimension, and a
//!   normalisation whose `x * x` is read again afterwards. A second `eps` after the plan for the
//!   first exists is bound at execution and used.

use std::collections::HashMap;

use burn::nn::RmsNormConfig;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_tt::{Tt, TtBackend, TtDevice};
use tt_tests::burn_device::{with_device, Config};

fn values(n: usize, seed: u32, specials: bool) -> Vec<f32> {
    const SPECIAL: [u32; 6] = [
        0x8000_0000, // -0
        0x0000_0001, // a subnormal
        0x807f_ffff, // a negative subnormal
        0x7f80_0000, // +inf
        0x7fc1_2345, // a NaN
        0x0000_0000, // +0
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if specials && i % 97 == 13 {
                return f32::from_bits(SPECIAL[(i / 97) % SPECIAL.len()]);
            }
            let scale = [1.0e-3f32, 1.0, 37.0][(s >> 3) as usize % 3];
            ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

fn tensor<B: Backend<Device = TtDevice>, const D: usize>(
    v: &[f32],
    shape: [usize; D],
    d: &TtDevice,
) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), d)
}

fn bits<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Burn's `RmsNorm::forward` up to its scale, as written there.
fn normalise<B: Backend, const D: usize>(x: Tensor<B, D>, eps: f64) -> Tensor<B, D> {
    let rms = (x.clone().square().mean_dim(D - 1) + eps).sqrt();
    x / rms
}

/// `(calls, jobs)` of each Burn op `f` ran, on every thread (burn-fusion runs its stream on the
/// thread that drains it; the table is this gate's alone, as it runs in a fork).
fn reported<R>(f: impl FnOnce() -> R) -> (R, HashMap<&'static str, (u64, u64)>) {
    burn_tt::report_reset();
    let r = f();
    let ops = burn_tt::report()
        .0
        .iter()
        .map(|s| (s.op, (s.calls, s.jobs)))
        .collect();
    (r, ops)
}

fn assert_bits(what: &str, got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let differ = got.iter().zip(want).filter(|(a, b)| a != b).count();
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{what}: {differ} of {} differ; first at {i}: {:#010x} for {:#010x}",
            got.len(),
            got[i],
            want[i]
        );
    }
}

#[test]
fn rms_norm_is_tt_backends_ops_bit_for_bit() {
    with_device(Config::default(), |d| {
        let check = |what: &str, v: &[f32], shape: [usize; 3], eps: f64| {
            let want = bits(normalise(tensor::<TtBackend, 3>(v, shape, &d), eps));
            burn_tt::reset_fusion_counters();
            let (got, ops) = reported(|| bits(normalise(tensor::<Tt, 3>(v, shape, &d), eps)));
            assert_bits(&format!("{what} {shape:?}, eps {eps:e}"), &got, &want);
            assert_eq!(burn_tt::fused_count("tt_rms_norm"), 1, "{what}: fused");
            let calls = |op: &str| ops.get(op).map_or(0, |s| s.0);
            assert_eq!(calls("tt_rms_norm"), 1, "{what}: one op");
            for op in [
                "float_mul",
                "float_mean_dim",
                "float_sum_dim",
                "float_sqrt",
                "float_div",
            ] {
                assert_eq!(calls(op), 0, "{what}: ran {op}");
            }
        };
        for (shape, specials) in [
            ([1, 1, 2048], false),
            ([1, 1, 576], true),
            ([1, 3, 37], true),
            ([2, 17, 96], true),
        ] {
            let n = shape.iter().product();
            for eps in [1e-5, 1e-6] {
                check("normalised", &values(n, n as u32, specials), shape, eps);
            }
        }
    });
}

/// Burn's `RmsNorm` module: the normalisation fused, its scale its own op after.
#[test]
fn burns_rms_norm_module_is_tt_backends() {
    fn module<B: Backend<Device = TtDevice>>(gamma: &[f32], x: &[f32], d: &TtDevice) -> Vec<u32> {
        let width = gamma.len();
        let mut m = RmsNormConfig::new(width).with_epsilon(1e-5).init::<B>(d);
        m.gamma = m.gamma.map(|_| tensor(gamma, [width], d));
        bits(m.forward(tensor::<B, 3>(x, [1, 5, width], d)))
    }
    with_device(Config::default(), |d| {
        let width = 576;
        let (gamma, x) = (values(width, 3, false), values(5 * width, 11, true));
        let want = module::<TtBackend>(&gamma, &x, &d);
        burn_tt::reset_fusion_counters();
        assert_bits("RmsNorm", &module::<Tt>(&gamma, &x, &d), &want);
        assert_eq!(burn_tt::fused_count("tt_rms_norm"), 1);
    });
}

/// The mean over dimension 1, not the last.
fn other<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let rms = (x.clone().square().mean_dim(1) + 1e-5).sqrt();
    x / rms
}

/// The normalisation, and its `x * x` read again afterwards.
fn kept<B: Backend>(x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let squares = x.clone().square();
    let rms = (squares.clone().mean_dim(2) + 1e-5).sqrt();
    (x / rms, squares)
}

#[test]
fn near_misses_run_as_their_operations() {
    with_device(Config::default(), |d| {
        let shape = [3, 40, 64];
        let x = values(shape.iter().product(), 5, true);
        burn_tt::reset_fusion_counters();

        // A second `eps` after the plan for the first: bound at execution.
        for eps in [1e-5, 1e-6, 1e-5] {
            assert_bits(
                &format!("eps {eps:e}"),
                &bits(normalise(tensor::<Tt, 3>(&x, shape, &d), eps)),
                &bits(normalise(tensor::<TtBackend, 3>(&x, shape, &d), eps)),
            );
        }
        assert_eq!(burn_tt::fused_count("tt_rms_norm"), 3);

        assert_bits(
            "mean over dimension 1",
            &bits(other(tensor::<Tt, 3>(&x, shape, &d))),
            &bits(other(tensor::<TtBackend, 3>(&x, shape, &d))),
        );
        let (got, got_squares) = kept(tensor::<Tt, 3>(&x, shape, &d));
        let (want, want_squares) = kept(tensor::<TtBackend, 3>(&x, shape, &d));
        assert_bits("beside its kept squares", &bits(got), &bits(want));
        assert_bits("the kept squares", &bits(got_squares), &bits(want_squares));
        assert_eq!(burn_tt::fused_count("tt_rms_norm"), 3, "no control fuses");
    });
}

/// `burn_tt::rms_norm` (decode-to-bandwidth Phase 7): Burn's `RmsNorm` with its scale as one op,
/// called by a model where burn-fusion's blocks keep the scale from the fuser. Bit for bit the
/// module on `TtBackend`, on `TtBackend` and on `Tt`, at a decode row of SmolLM2's widths, several
/// rows and rank 3, over specials; a single row is the native call and runs no `float_mul`, more
/// rows the norm's op and the product. A BF16 input is declined (the module casts its statistic). Negative control: a
/// gamma of ones gives other bits.
#[test]
fn the_scaled_norm_is_one_op_and_the_module_bit_for_bit() {
    fn module<B: Backend<Device = TtDevice>>(
        gamma: &[f32],
        x: &[f32],
        shape: [usize; 3],
        d: &TtDevice,
    ) -> Vec<u32> {
        let mut m = RmsNormConfig::new(gamma.len())
            .with_epsilon(1e-5)
            .init::<B>(d);
        m.gamma = m.gamma.map(|_| tensor(gamma, [gamma.len()], d));
        bits(m.forward(tensor::<B, 3>(x, shape, d)))
    }
    fn scaled<B: Backend<Device = TtDevice>>(
        gamma: &[f32],
        x: &[f32],
        shape: [usize; 3],
        d: &TtDevice,
    ) -> Vec<u32> {
        let g = tensor::<B, 1>(gamma, [gamma.len()], d);
        bits(burn_tt::rms_norm(&tensor::<B, 3>(x, shape, d), &g, 1e-5).expect("taken"))
    }
    with_device(Config::default(), |d| {
        for (shape, specials, native) in [
            ([1, 1, 2048], false, true),
            ([1, 1, 576], true, true),
            ([1, 5, 576], true, false),
            ([2, 17, 96], true, false),
        ] {
            let width = shape[2];
            let n = shape.iter().product();
            let (gamma, x) = (values(width, 7, specials), values(n, n as u32, specials));
            let want = module::<TtBackend>(&gamma, &x, shape, &d);
            let before = burn_tt::rms_norm_count();
            let (got, ops) = reported(|| scaled::<TtBackend>(&gamma, &x, shape, &d));
            assert_bits(&format!("TtBackend {shape:?}"), &got, &want);
            if native {
                assert_eq!(burn_tt::rms_norm_count(), before + 1, "{shape:?}: native");
                if shape[1] == 1 {
                    let calls = |op: &str| ops.get(op).map_or(0, |s| s.0);
                    assert_eq!(calls("float_mul"), 0, "{shape:?}: the scale inside the op");
                }
            }
            assert_bits(
                &format!("Tt {shape:?}"),
                &scaled::<Tt>(&gamma, &x, shape, &d),
                &want,
            );
            // On a row without specials (a NaN's row is NaN whatever the scale).
            let (ones, plain) = (vec![1.0f32; width], values(n, n as u32, false));
            assert_ne!(
                scaled::<TtBackend>(&ones, &plain, shape, &d),
                scaled::<TtBackend>(&gamma, &plain, shape, &d),
                "{shape:?}: the scale reaches the outputs"
            );
        }
        let x = tensor::<TtBackend, 3>(&values(64, 1, false), [1, 1, 64], &d)
            .cast(burn::tensor::DType::BF16);
        let g = tensor::<TtBackend, 1>(&values(64, 2, false), [64], &d);
        assert!(burn_tt::rms_norm(&x, &g, 1e-5).is_none(), "BF16 declined");
    });
}

/// A residual sum and the norm after it on `Tt` (decode-to-bandwidth Phase 7): `s = x + h W`
/// then `burn_tt::rms_norm(s)`, and `s` then Burn's composition up to the scale, at decode rows of
/// 2048 and 576 (the second below the packed rows' width: the two ops). Each matches its chain
/// (`tt_add_rms_norm_scaled`, `tt_add_rms_norm`), the sum is read afterwards, and both the sum and
/// the norm are `TtBackend`'s bit for bit, through the native call (one op at 2048, where `x`
/// (an element-wise result) and `h W` (a one-row product's) keep packed rows; the session's two
/// at 576). Control: the norm of `x` alone differs.
#[test]
fn a_residual_sum_and_its_norm_are_one_op_on_tt() {
    fn run<B: Backend<Device = TtDevice>>(
        width: usize,
        scaled: bool,
        specials: bool,
        d: &TtDevice,
    ) -> (Vec<u32>, Vec<u32>) {
        let x = tensor::<B, 3>(&values(width, 3, specials), [1, 1, width], d).mul_scalar(1.0);
        let h = tensor::<B, 3>(&values(64, 5, false), [1, 1, 64], d);
        let w = tensor::<B, 2>(&values(64 * width, 7, false), [64, width], d);
        let gamma = tensor::<B, 1>(&values(width, 9, false), [width], d);
        let s = x + h.matmul(w.unsqueeze());
        let n = if scaled {
            burn_tt::rms_norm(&s, &gamma, 1e-5).expect("taken")
        } else {
            normalise(s.clone(), 1e-5)
        };
        (bits(n), bits(s))
    }
    with_device(Config::default(), |d| {
        // Opt-in (slower on silicon): on for this gate, whose process is its own fork.
        burn_tt::set_add_rms_norm(true);
        for (width, scaled, specials) in [2048, 576]
            .into_iter()
            .flat_map(|w| [(w, true), (w, false)])
            .flat_map(|(w, s)| [(w, s, true), (w, s, false)])
        {
            {
                let what = format!("width {width}, scaled {scaled}, specials {specials}");
                let want = run::<TtBackend>(width, scaled, specials, &d);
                burn_tt::reset_fusion_counters();
                let before = burn_tt::add_rms_norm_count();
                let got = run::<Tt>(width, scaled, specials, &d);
                assert_bits(&format!("{what}: the norm"), &got.0, &want.0);
                assert_bits(&format!("{what}: the sum"), &got.1, &want.1);
                let chain = if scaled {
                    "tt_add_rms_norm_scaled"
                } else {
                    "tt_add_rms_norm"
                };
                assert_eq!(burn_tt::fused_count(chain), 1, "{what}: {chain} matched");
                assert_eq!(
                    burn_tt::add_rms_norm_count() - before,
                    1,
                    "{what}: the native call"
                );
            }
        }
        let x = tensor::<Tt, 3>(&values(2048, 3, false), [1, 1, 2048], &d).mul_scalar(1.0);
        let (alone, _) = run::<Tt>(2048, false, false, &d);
        assert_ne!(bits(normalise(x, 1e-5)), alone, "the sum reaches the norm");
    });
}
