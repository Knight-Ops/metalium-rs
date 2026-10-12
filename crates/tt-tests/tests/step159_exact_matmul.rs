//! `MatmulMode::Exact`: F32 products summed in increasing `k` order on the SFPU, instead of the
//! Matrix Unit's group adder (`step158_matmul_accumulation`).
//!
//! The oracle is a host loop that is not the implementation: for each output, start from `+0.0`, and
//! for `k = 0, 1, ...` add the product, each product and each sum an F32 operation with denormal inputs
//! and results flushed to zero and NaN canonical, which is how the SFPU `mul` and `add` programs are
//! specified (`tt-kernels/src/sfpu/ops.rs`, `mover`). The device must equal it bit for bit, on
//! random operands of wide exponent spread, on every special value, and through every route a matmul
//! takes (batch broadcast, transposed and ragged views, `linear`, `conv2d`, autodiff, a trace). Nothing
//! may be downloaded or computed on the host.
//!
//! The default mode is untouched (the hook returns without acting), which the controls watch: the same
//! data in the default mode is not the oracle's.

use burn::backend::Autodiff;
use burn::tensor::module::{conv2d, linear};
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};
use burn_tt::{exact_matmul, with_matmul_mode, MatmulMode, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn exact() -> Config {
    Config {
        exact_matmul: true,
        ..Config::default()
    }
}

fn fast() -> Config {
    Config {
        exact_matmul: false,
        ..Config::default()
    }
}

// ---- the oracle ----

fn flush(x: f32) -> f32 {
    if x.is_subnormal() {
        0.0f32.copysign(x)
    } else {
        x
    }
}

fn mul(a: f32, b: f32) -> f32 {
    flush(flush(a) * flush(b))
}

fn add(a: f32, b: f32) -> f32 {
    flush(flush(a) + flush(b))
}

/// `sum_k a[k] b[k]`, from `+0.0`, in increasing `k`.
fn dot(a: impl Iterator<Item = f32>, b: impl Iterator<Item = f32>) -> f32 {
    a.zip(b).fold(0.0, |acc, (x, y)| add(acc, mul(x, y)))
}

/// `[m, k] @ [k, n]`, row-major.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    (0..m * n)
        .map(|x| {
            let (i, j) = (x / n, x % n);
            dot((0..k).map(|p| a[i * k + p]), (0..k).map(|p| b[p * n + j]))
        })
        .collect()
}

/// Equal bits, or both NaN (a NaN's payload is not inferred).
fn same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
            "{what}: element {i}: device {g} ({:#010x}), oracle {w} ({:#010x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
    /// A float of random sign and mantissa, with an exponent in `[-20, 20]`.
    fn wide(&mut self) -> f32 {
        let m = 1.0 + self.below(1 << 23) as f32 / (1 << 23) as f32;
        let v = m * 2f32.powi(self.below(41) as i32 - 20);
        if self.below(2) == 0 {
            v
        } else {
            -v
        }
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.wide()).collect()
    }
}

fn t2(d: &burn_tt::TtDevice, v: &[f32], r: usize, c: usize) -> Tensor<TtBackend, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), d)
}

fn host<const D: usize>(t: Tensor<TtBackend, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

// ---- the gate ----

#[test]
fn random_operands_equal_the_sequential_sfpu_oracle_and_stay_within_its_bound() {
    with_device(exact(), |d| {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for (m, k, n) in [
            (1, 1, 1),
            (5, 7, 3),
            (32, 16, 1),
            (33, 27, 9),
            (4, 64, 5),
            (9, 100, 2),
        ] {
            let (a, b) = (rng.vec(m * k), rng.vec(k * n));
            let got = host(t2(&d, &a, m, k).matmul(t2(&d, &b, k, n)));
            let want = matmul(&a, &b, m, k, n);
            same(&got, &want, &format!("[{m},{k}] @ [{k},{n}]"));
            // And against the true value: k products and k adds each round by at most 2^-24.
            let u = 2f64.powi(-24);
            let gamma = (k as f64 + 1.0) * u / (1.0 - (k as f64 + 1.0) * u);
            for (x, &y) in got.iter().enumerate() {
                let (i, j) = (x / n, x % n);
                let (mut exact, mut mag) = (0f64, 0f64);
                for p in 0..k {
                    let t = f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
                    exact += t;
                    mag += t.abs();
                }
                assert!(
                    (f64::from(y) - exact).abs() <= gamma * mag,
                    "[{m},{k}] @ [{k},{n}] element {x}: error over gamma_{{k+1}} sum|ab|"
                );
            }
        }
    });
}

#[test]
fn special_values_follow_the_sfpu_reference() {
    with_device(exact(), |d| {
        let specials = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1e-40,
            -1e-40,
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            3.0e30,
            -3.0e30,
            1.0e-30,
            2.0,
        ];
        let mut rng = Rng(0x1357_9bdf_2468_ace0);
        let (m, k, n) = (64, 5, 9);
        let pick = |rng: &mut Rng, count: usize| -> Vec<f32> {
            (0..count)
                .map(|_| specials[rng.below(specials.len() as u64) as usize])
                .collect()
        };
        let (a, b) = (pick(&mut rng, m * k), pick(&mut rng, k * n));
        let got = host(t2(&d, &a, m, k).matmul(t2(&d, &b, k, n)));
        same(&got, &matmul(&a, &b, m, k, n), "specials");
        // The control: the data does exercise the special cases.
        let want = matmul(&a, &b, m, k, n);
        assert!(want.iter().any(|x| x.is_nan()));
        assert!(want.iter().any(|x| x.is_infinite()));
        assert!(want.contains(&0.0));
    });
}

#[test]
fn every_route_a_matmul_takes_is_exact_and_resident() {
    with_device(exact(), |d| {
        let mut rng = Rng(0x2468_ace0_1357_9bdf);
        let before = burn_tt::tensor_traffic();

        // Batch axes that broadcast: [2, 1, m, k] @ [1, 3, k, n] -> [2, 3, m, n].
        let (m, k, n) = (3, 11, 4);
        let (a, b) = (rng.vec(2 * m * k), rng.vec(3 * k * n));
        let (y, report) = burn_tt::with_report(|| {
            let lhs =
                Tensor::<TtBackend, 4>::from_data(TensorData::new(a.clone(), [2, 1, m, k]), &d);
            let rhs =
                Tensor::<TtBackend, 4>::from_data(TensorData::new(b.clone(), [1, 3, k, n]), &d);
            lhs.matmul(rhs)
        });
        assert_native_model(&report);
        assert_eq!(y.dims(), [2, 3, m, n]);
        let got = host(y);
        let mut want = Vec::new();
        for b0 in 0..2 {
            for b1 in 0..3 {
                want.extend(matmul(
                    &a[b0 * m * k..][..m * k],
                    &b[b1 * k * n..][..k * n],
                    m,
                    k,
                    n,
                ));
            }
        }
        same(&got, &want, "batch broadcast");

        // A transposed left operand and a ragged slice of the right.
        let (m, k, n) = (6, 13, 5);
        let (at, big) = (rng.vec(k * m), rng.vec((k + 2) * (n + 3)));
        let lhs = t2(&d, &at, k, m).transpose();
        let rhs = t2(&d, &big, k + 2, n + 3).slice([1..k + 1, 2..n + 2]);
        let a: Vec<f32> = (0..m * k).map(|x| at[(x % k) * m + x / k]).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|x| big[(x / n + 1) * (n + 3) + x % n + 2])
            .collect();
        same(
            &host(lhs.matmul(rhs)),
            &matmul(&a, &b, m, k, n),
            "transposed and sliced views",
        );

        // `linear`: the matmul, then the bias add.
        let (rows, d_in, d_out) = (4, 9, 3);
        let (x, w, bias) = (rng.vec(rows * d_in), rng.vec(d_in * d_out), rng.vec(d_out));
        let (y, report) = burn_tt::with_report(|| {
            linear(
                t2(&d, &x, rows, d_in),
                t2(&d, &w, d_in, d_out),
                Some(Tensor::<TtBackend, 1>::from_data(
                    TensorData::new(bias.clone(), [d_out]),
                    &d,
                )),
            )
        });
        assert_native_model(&report);
        let want: Vec<f32> = matmul(&x, &w, rows, d_in, d_out)
            .iter()
            .enumerate()
            .map(|(i, &v)| add(v, bias[i % d_out]))
            .collect();
        same(&host(y), &want, "linear");

        // `conv2d`: patches @ weights, the patch order (channel, row, column), then the bias.
        let (ci, co, h, wd, kh, kw) = (2usize, 3usize, 5usize, 6usize, 2usize, 3usize);
        let (x, w, bias) = (
            rng.vec(ci * h * wd),
            rng.vec(co * ci * kh * kw),
            rng.vec(co),
        );
        let (y, report) = burn_tt::with_report(|| {
            conv2d(
                Tensor::<TtBackend, 4>::from_data(TensorData::new(x.clone(), [1, ci, h, wd]), &d),
                Tensor::<TtBackend, 4>::from_data(TensorData::new(w.clone(), [co, ci, kh, kw]), &d),
                Some(Tensor::<TtBackend, 1>::from_data(
                    TensorData::new(bias.clone(), [co]),
                    &d,
                )),
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
            )
        });
        assert_native_model(&report);
        let (oh, ow) = (h - kh + 1, wd - kw + 1);
        let mut want = Vec::new();
        for oc in 0..co {
            for r in 0..oh {
                for c in 0..ow {
                    let patch = (0..ci * kh * kw).map(|q| {
                        let (ic, kr, kc) = (q / (kh * kw), q / kw % kh, q % kw);
                        x[(ic * h + r + kr) * wd + c + kc]
                    });
                    let weights = (0..ci * kh * kw).map(|q| w[oc * ci * kh * kw + q]);
                    want.push(add(dot(patch, weights), bias[oc]));
                }
            }
        }
        same(&host(y), &want, "conv2d");

        assert_eq!(
            burn_tt::tensor_traffic().downloads,
            before.downloads + 4,
            "only the four result reads may download"
        );
    });
}

/// More outputs than one chunk holds, with the chunk made small (`set_exact_matmul_chunk_products`):
/// three chunks, the last ragged, joined into one result in output order. At the default `2^20`
/// products a chunk costs about 70 s on ttsim, which is why the size is a setting.
#[test]
fn a_product_of_several_chunks_is_joined_in_order() {
    with_device(exact(), |d| {
        burn_tt::set_exact_matmul_chunk_products(64 * 5);
        let mut rng = Rng(0x6a09_e667_f3bc_c908);
        // 5 outputs a chunk at k = 64: 12 outputs are three chunks of 5, 5 and 2.
        let (m, k, n) = (3, 64, 4);
        let (a, b) = (rng.vec(m * k), rng.vec(k * n));
        let got = host(t2(&d, &a, m, k).matmul(t2(&d, &b, k, n)));
        same(&got, &matmul(&a, &b, m, k, n), "three chunks");
        // A batch broadcast across chunk boundaries, too.
        let (a4, b4) = (rng.vec(2 * m * k), rng.vec(2 * k * n));
        let lhs = Tensor::<TtBackend, 3>::from_data(TensorData::new(a4.clone(), [2, m, k]), &d);
        let rhs = Tensor::<TtBackend, 3>::from_data(TensorData::new(b4.clone(), [2, k, n]), &d);
        let mut want = matmul(&a4[..m * k], &b4[..k * n], m, k, n);
        want.extend(matmul(&a4[m * k..], &b4[k * n..], m, k, n));
        same(&host(lhs.matmul(rhs)), &want, "a batch over chunks");
    });
}

/// What the exact path will not do, it refuses by name rather than falling back to the fast one: a
/// `k` that does not fit one chunk (a sum is never split across `k`), and a product needing more
/// than the limit of 1024 chunks.
#[test]
fn what_does_not_fit_is_refused_by_name() {
    with_device(exact(), |d| {
        let refusal = |f: &dyn Fn()| -> String {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
                .expect_err("expected a refusal");
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default()
        };
        let ones = |r: usize, c: usize| t2(&d, &vec![1.0; r * c], r, c);
        burn_tt::set_exact_matmul_chunk_products(63);
        let message = refusal(&|| {
            let _ = ones(2, 64).matmul(ones(64, 2));
        });
        assert!(
            message.contains("exact matmul") && message.contains("k = 64"),
            "{message}"
        );
        // One output a chunk: 1025 outputs are 1025 chunks.
        burn_tt::set_exact_matmul_chunk_products(1);
        let message = refusal(&|| {
            let _ = ones(1025, 1).matmul(ones(1, 1));
        });
        assert!(
            message.contains("1025 chunks") && message.contains("limit 1024"),
            "{message}"
        );
        // Within both limits it runs.
        burn_tt::set_exact_matmul_chunk_products(1 << 20);
        assert_eq!(host(ones(2, 3).matmul(ones(3, 2))), vec![3.0; 4]);
    });
}

/// The default is the fast matmul, a scoped override changes it for its closure only, and a device
/// attached exact is exact unless the closure says otherwise. 125..249 times 1..53 summed over 64:
/// every product and sum is exact in F32, so the true value is the oracle's, and the Matrix Unit's
/// group adder (`step158`) misses it.
#[test]
fn the_default_is_fast_and_the_config_and_the_override_choose() {
    let (m, k) = (32usize, 64usize);
    let a: Vec<f32> = (0..m * k).map(|i| (125 + i % 125) as f32).collect();
    let b: Vec<f32> = (0..k).map(|i| (1 + i % 53) as f32).collect();
    let truth: Vec<f32> = (0..m)
        .map(|r| {
            (0..k)
                .map(|p| f64::from(a[r * k + p]) * f64::from(b[p]))
                .sum::<f64>() as f32
        })
        .collect();
    assert_eq!(truth, matmul(&a, &b, m, k, 1), "the data is exact in F32");
    let run = |d: &burn_tt::TtDevice| host(t2(d, &a, m, k).matmul(t2(d, &b, k, 1)));

    with_device(fast(), |d| {
        assert_ne!(run(&d), truth, "the default is the fast path");
        assert_eq!(exact_matmul(|| run(&d)), truth);
        assert_eq!(with_matmul_mode(MatmulMode::Exact, || run(&d)), truth);
        assert_ne!(run(&d), truth, "the scope ended");
        // Scopes nest and the innermost wins.
        exact_matmul(|| {
            assert_ne!(with_matmul_mode(MatmulMode::Fast, || run(&d)), truth);
            assert_eq!(run(&d), truth);
        });
        // A panic inside a scope does not leave it on.
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            exact_matmul(|| panic!("inside the scope"))
        }));
        assert!(caught.is_err());
        assert_ne!(run(&d), truth, "a panic must not leak the override");
    });
    with_device(exact(), |d| {
        assert_eq!(run(&d), truth, "an exact device is exact");
        assert_ne!(with_matmul_mode(MatmulMode::Fast, || run(&d)), truth);
        assert_eq!(run(&d), truth);
    });
}

/// Gradients are matmuls (`g @ w^T`, `x^T @ g`) through the same entry point: wrapped in the scope,
/// `backward()` is exact, and the left operand of each is a transposed view.
#[test]
fn backward_under_the_scope_is_exact() {
    with_device(fast(), |d| {
        type AD = Autodiff<TtBackend>;
        let mut rng = Rng(0x0bad_cafe_1234_5678);
        let (m, k, n) = (5, 9, 4);
        let (xv, wv, gv) = (rng.vec(m * k), rng.vec(k * n), rng.vec(m * n));
        let x = Tensor::<AD, 2>::from_data(TensorData::new(xv.clone(), [m, k]), &d).require_grad();
        let w = Tensor::<AD, 2>::from_data(TensorData::new(wv.clone(), [k, n]), &d).require_grad();
        let g = Tensor::<AD, 2>::from_data(TensorData::new(gv.clone(), [m, n]), &d);
        let loss = (x.clone().matmul(w.clone()) * g).sum();
        let grads = exact_matmul(|| loss.backward());
        let gx = x.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();
        let gw = w.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();
        // d loss / dx = g @ w^T, d loss / dw = x^T @ g.
        let wt: Vec<f32> = (0..n * k).map(|q| wv[(q % k) * n + q / k]).collect();
        let xt: Vec<f32> = (0..k * m).map(|q| xv[(q % m) * k + q / m]).collect();
        same(&gx, &matmul(&gv, &wt, m, n, k), "x gradient");
        same(&gw, &matmul(&xt, &gv, k, m, n), "w gradient");
    });
}

/// A trace freezes the mode in effect when it was captured, and replays what it captured.
#[test]
fn a_trace_captured_exact_replays_exact() {
    with_device(exact(), |d| {
        let mut rng = Rng(0x5555_aaaa_1111_2222);
        let (m, k, n) = (6, 10, 3);
        let (xv, wv) = (rng.vec(m * k), rng.vec(k * n));
        let x = t2(&d, &xv, m, k).mul_scalar(1.0);
        let TensorPrimitive::Float(input) = x.clone().into_primitive() else {
            unreachable!()
        };
        let w = t2(&d, &wv, k, n).to_device(&d);
        let (trace, first) = burn_tt::Trace::capture(&input, || {
            let TensorPrimitive::Float(out) = x.clone().matmul(w.clone()).into_primitive() else {
                unreachable!()
            };
            out
        })
        .unwrap();
        same(&first, &matmul(&xv, &wv, m, k, n), "the capture's own run");
        for _ in 0..2 {
            let again = rng.vec(m * k);
            same(
                &trace.run(again.clone()).unwrap(),
                &matmul(&again, &wv, m, k, n),
                "a replay",
            );
        }
    });
}

/// The gate can fail: the oracle's order matters, and the default mode is not the oracle.
#[test]
fn controls_the_order_matters_and_the_default_is_not_the_oracle() {
    let mut rng = Rng(0x7777_1111_9999_3333);
    let (m, k, n) = (16, 48, 4);
    let (a, b) = (rng.vec(m * k), rng.vec(k * n));
    let want = matmul(&a, &b, m, k, n);
    let reversed: Vec<f32> = (0..m * n)
        .map(|x| {
            let (i, j) = (x / n, x % n);
            dot(
                (0..k).rev().map(|p| a[i * k + p]),
                (0..k).rev().map(|p| b[p * n + j]),
            )
        })
        .collect();
    assert_ne!(
        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        reversed.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "summing in the other order must give other bits, or the gate cannot see an order error"
    );
    with_device(fast(), |d| {
        let got = host(t2(&d, &a, m, k).matmul(t2(&d, &b, k, n)));
        assert!(
            got.iter()
                .zip(&want)
                .any(|(g, w)| g.to_bits() != w.to_bits()),
            "the default (fast) mode must differ from the oracle on this data"
        );
        let exact = exact_matmul(|| host(t2(&d, &a, m, k).matmul(t2(&d, &b, k, n))));
        same(&exact, &want, "the same data, scoped exact");
    });
}

/// BF16 operands widen, take the exact path and narrow once. Silicon only: the narrowing pack is PACR
/// `0x105`, which ttsim refuses.
#[cfg(feature = "silicon")]
#[test]
fn bf16_operands_widen_compute_exactly_and_round_once() {
    use burn::tensor::FloatDType;
    with_device(exact(), |d| {
        let mut rng = Rng(0x4242_4242_4242_4242);
        let (m, k, n) = (7, 12, 5);
        let (a, b) = (rng.vec(m * k), rng.vec(k * n));
        let bf = |t: Tensor<TtBackend, 2>| t.cast(FloatDType::BF16);
        let (lhs, rhs) = (bf(t2(&d, &a, m, k)), bf(t2(&d, &b, k, n)));
        let widen = |t: &Tensor<TtBackend, 2>| host(t.clone().cast(FloatDType::F32));
        let (aw, bw) = (widen(&lhs), widen(&rhs));
        let got = host(lhs.matmul(rhs).cast(FloatDType::F32));
        // Round to nearest even to BF16, from the oracle's F32 result.
        let narrow = |x: f32| {
            let bits = x.to_bits();
            let round = 0x7fff + ((bits >> 16) & 1);
            f32::from_bits(bits.wrapping_add(round) & 0xffff_0000)
        };
        let want: Vec<f32> = matmul(&aw, &bw, m, k, n).into_iter().map(narrow).collect();
        same(&got, &want, "bf16");
    });
}

/// What the exact mode costs, on silicon: host-timed medians of the fast Matrix Unit matmul and of the
/// exact one, over shapes from one chunk to sixteen, with the outputs validated on every sample (the
/// exact one bit for bit against the oracle, the fast one against it within a loose sanity bound, not
/// a tolerance claim: the TF32 truncation and the group alignment together stay far below it).
/// `cargo xtask bench --device 0 --filter step159_exact_matmul::benchmark`.
#[cfg(feature = "silicon")]
#[test]
#[ignore = "release benchmark"]
fn benchmark_exact_against_fast_matmul() {
    use std::time::Instant;
    use tt_tests::bench::{report, Stats, REPS};
    with_device(fast(), |d| {
        let mut rng = Rng(0x5eed_0000_0000_0159);
        // (m, k, n): one chunk, a conv-like product, four chunks, sixteen.
        for (m, k, n) in [
            (64, 32, 64),
            (1024, 27, 32),
            (128, 256, 128),
            (256, 256, 256),
        ] {
            let a: Vec<f32> = (0..m * k)
                .map(|_| 0.5 + rng.below(1 << 20) as f32 / (1 << 20) as f32)
                .collect();
            let a: Vec<f32> = a
                .iter()
                .map(|&v| if rng.below(2) == 0 { v } else { -v })
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|_| 0.5 + rng.below(1 << 20) as f32 / (1 << 20) as f32)
                .collect();
            let (lhs, rhs) = (t2(&d, &a, m, k), t2(&d, &b, k, n));
            let want = matmul(&a, &b, m, k, n);
            for (name, mode) in [("fast", MatmulMode::Fast), ("exact", MatmulMode::Exact)] {
                let mut micros = Vec::new();
                for i in 0..REPS + 2 {
                    let t = Instant::now();
                    let got = with_matmul_mode(mode, || host(lhs.clone().matmul(rhs.clone())));
                    let elapsed = t.elapsed();
                    if mode == MatmulMode::Exact {
                        same(&got, &want, "benchmark output");
                    } else {
                        for (x, (&g, &w)) in got.iter().zip(&want).enumerate() {
                            let mag: f64 = (0..k)
                                .map(|p| {
                                    (f64::from(a[x / n * k + p]) * f64::from(b[p * n + x % n]))
                                        .abs()
                                })
                                .sum();
                            assert!(
                                (f64::from(g) - f64::from(w)).abs() <= mag / 64.0,
                                "the fast result is implausible at {x}"
                            );
                        }
                    }
                    if i >= 2 {
                        micros.push(elapsed.as_secs_f64() * 1e6);
                    }
                }
                report(
                    &format!("matmul_{name}_{m}x{k}x{n}"),
                    "us",
                    "host",
                    Stats::of(micros),
                );
            }
        }
    });
}
