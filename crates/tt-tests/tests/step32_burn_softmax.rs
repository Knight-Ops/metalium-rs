//! Phase 10 gate (R1, R2 through Burn): reductions, softmax and log-softmax
//! on device-resident tensors, computed on the device end to end.
//!
//! `float_max_dim` is exactly Flex's value; `float_sum_dim` over columns is
//! within the order bound `step31_reduce` states. `softmax` and `log_softmax`
//! run Burn's own composition through this backend's device ops (max,
//! broadcast subtract, `exp`, sum, broadcast divide or `log` and subtract),
//! so the claim against Flex's fused implementation is a derived bound, as a
//! fraction of the result: the SFPU's `exp` twice (`EXP_BOUND`, numerator
//! and inside the sum), the sum's order (`(n - 1) u`), the division's ulp
//! (`2^-23`), and Flex's own counterparts of each (`exp` and division to an
//! ulp, its sum to `(n - 1) u`). For `log_softmax`, an absolute bound: `log`'s
//! relative error of `ln(sum)` plus the sum's relative error, for each side.

use burn::tensor::{activation, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend};
use tt_kernels::sfpu::ops::{EXP_BOUND, LOG_BOUND};
use tt_tests::burn_device::{with_device, Config};

const U: f64 = 1.0 / 16_777_216.0;

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 20.0
        })
        .collect()
}

fn floats<B: burn::tensor::backend::Backend>(t: Tensor<B, 2>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

fn resident(t: &Tensor<TtBackend, 2>) -> bool {
    match t.clone().into_primitive() {
        burn::tensor::TensorPrimitive::Float(p) => p.computed_on_device(),
        _ => false,
    }
}

#[test]
fn reductions_and_softmax_run_on_the_device_within_their_bounds() {
    with_device(Config::default(), |d| {
        // Any size: approximations follow their data (MNIST's two-tile
        // `[64, 10]` logits too).
        for [r, c] in [[256, 10], [70, 120], [40, 400], [64, 10], [5, 7]] {
            let v = values((r * c) as u64, r * c);
            let t = Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), [r, c]), &d)
                .to_device(&d);
            let f = Tensor::<Flex, 2>::from_data(TensorData::new(v.clone(), [r, c]), &FlexDevice);
            for dim in [0usize, 1] {
                let n = if dim == 1 { c } else { r } as f64;
                let before = tensor_traffic();
                let max = t.clone().max_dim(dim);
                let sum = t.clone().sum_dim(dim);
                let soft = activation::softmax(t.clone(), dim);
                let logsoft = activation::log_softmax(t.clone(), dim);
                let softmin = activation::softmin(t.clone(), dim);
                let during = tensor_traffic() - before;
                assert_eq!(
                    (during.uploads, during.downloads),
                    (0, 0),
                    "[{r}, {c}] dim {dim}: {during:?}"
                );
                for (name, x) in [
                    ("max", &max),
                    ("sum", &sum),
                    ("softmax", &soft),
                    ("log_softmax", &logsoft),
                    ("softmin", &softmin),
                ] {
                    assert!(resident(x), "[{r}, {c}] dim {dim}: {name} ran on the host");
                }
                for (m, w) in floats(max).iter().zip(floats(f.clone().max_dim(dim))) {
                    assert!(*m == w, "[{r}, {c}] dim {dim} max: {m} vs {w}");
                }
                // A sum over rows is the mover's, in Flex's order: exact.
                let sum_bound = if dim == 0 {
                    0.0
                } else {
                    2.0 * (n - 1.0) * U * 10.0 * n
                };
                for (s, w) in floats(sum).iter().zip(floats(f.clone().sum_dim(dim))) {
                    assert!(
                        (*s as f64 - w as f64).abs() <= sum_bound,
                        "[{r}, {c}] dim {dim} sum: {s} vs {w}"
                    );
                }
                let sm_bound = 2.0 * EXP_BOUND + (n - 1.0) * U + 2.0 * U  // ours
                    + 2.0 * 2.0 * U + (n - 1.0) * U + 2.0 * U; // Flex's
                let mut worst = 0.0f64;
                for (s, w) in floats(soft)
                    .iter()
                    .zip(floats(activation::softmax(f.clone(), dim)))
                {
                    let rel = (*s as f64 - w as f64).abs()
                        / (w as f64).abs().max(f32::MIN_POSITIVE as f64);
                    worst = worst.max(rel);
                    assert!(
                        rel <= sm_bound || (*s == 0.0 && w.abs() < f32::MIN_POSITIVE * 4.0),
                        "[{r}, {c}] dim {dim} softmax: {s:e} vs {w:e}, {rel:e} > {sm_bound:e}"
                    );
                }
                // Softmin is the softmax of `-x`, the negation exact on both
                // sides: the softmax's bound.
                for (s, w) in floats(softmin)
                    .iter()
                    .zip(floats(activation::softmin(f.clone(), dim)))
                {
                    let rel = (*s as f64 - w as f64).abs()
                        / (w as f64).abs().max(f32::MIN_POSITIVE as f64);
                    assert!(
                        rel <= sm_bound || (*s == 0.0 && w.abs() < f32::MIN_POSITIVE * 4.0),
                        "[{r}, {c}] dim {dim} softmin: {s:e} vs {w:e}, {rel:e} > {sm_bound:e}"
                    );
                }
                let flex_ls = floats(activation::log_softmax(f.clone(), dim));
                let mut worst_ls = 0.0f64;
                for (s, w) in floats(logsoft).iter().zip(&flex_ls) {
                    // `ln(sum)` is at most `ln n` (the max's term is 1, every
                    // other at most 1); its error `LOG_BOUND ln n`, the sum's
                    // relative error carried one for one, and a rounding of
                    // the result on each side.
                    let bound = LOG_BOUND * n.ln().max(1.0)
                        + 2.0 * ((n - 1.0) * U + EXP_BOUND)
                        + 2.0 * U * (*w as f64).abs().max(1.0) * 2.0;
                    let diff = (*s as f64 - *w as f64).abs();
                    worst_ls = worst_ls.max(diff / bound);
                    assert!(
                        diff <= bound,
                        "[{r}, {c}] dim {dim} log_softmax: {s:e} vs {w:e}"
                    );
                }
                println!("[{r}, {c}] dim {dim}: softmax worst {worst:.2e} (bound {sm_bound:.2e}); log_softmax at {worst_ls:.2} of its bound");
            }
        }
    });
}

/// Supported host inputs upload for native softmax instead of choosing a CPU path.
#[test]
fn host_inputs_upload_for_native_softmax() {
    with_device(Config::default(), |d| {
        let [r, c] = [64, 10];
        let v = values(3, r * c);
        let (soft, report) = burn_tt::with_report(|| {
            activation::softmax(
                Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), [r, c]), &d),
                1,
            )
        });
        assert!(resident(&soft));
        let op = report.op("softmax").unwrap();
        assert_eq!((op.on_host, op.downloads, op.staged), (0, 0, 0));
        assert_eq!(op.uploads, 1);
        let want = floats(activation::softmax(
            Tensor::<Flex, 2>::from_data(TensorData::new(v, [r, c]), &FlexDevice),
            1,
        ));
        for (got, want) in floats(soft).into_iter().zip(want) {
            assert!((got - want).abs() < 5e-6);
        }
    });
}
