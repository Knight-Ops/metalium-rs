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
