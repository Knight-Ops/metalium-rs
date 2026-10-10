//! S7, step 142: Burn's `float_random` / `int_random` are drawn on the
//! device by the seeded tile kernel, resident, with no upload; dropout composes
//! native Bernoulli with the mask multiply; a failed or unsupported draw is an
//! explicit failure, never a host retry; a trace refuses to capture a draw.
//!
//! Oracles: `tt_kernels::prng`'s pure-Rust stream model (`step140`/`step141`),
//! f32 host emulation of the affine maps, and an f64 Box-Muller with the bound
//! `prng_support::normal_bound` derives from the programs' own bounds.
//!
//! A device draw takes the device's next base `splitmix64(seed ^
//! splitmix64(n))` (`n` the draws since `Backend::seed`), so the expected bits
//! are a function of `(seed, n)` the test recomputes.
//!
//! Negative controls (watched to fail, recorded in the close-out record): (1) the
//! host `StdRng` path substituted for an attached device draw would upload
//! (`uploads` rises) and fail the model comparison; (2) a draw that ignores the
//! per-call counter repeats the previous tensor (`consecutive draws differ`).

mod prng_support;

use burn::backend::Autodiff;
use burn::tensor::{backend::Backend, Distribution, Int, Tensor, TensorPrimitive};
use burn_tt::{tensor_traffic, with_report, TtBackend};
use prng_support::{expected, normal_bound, units_of};
use tt_kernels::prng::{next_down, role, splitmix64, uniform_bound, Target, TWO_PI};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

const SEED: u64 = 5;

fn target() -> Target {
    Target::from_simulated(!cfg!(feature = "silicon"))
}

/// The base of the `n`-th (1-based) draw after `Backend::seed(.., seed)`.
fn base(seed: u64, n: u64) -> u64 {
    splitmix64(seed ^ splitmix64(n))
}

fn float_primitive(t: Tensor<TtBackend, 2>) -> burn_tt::TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        TensorPrimitive::QFloat(_) => unreachable!("a float tensor"),
    }
}

/// A draw, measured: no upload, nothing staged or computed on the host, the
/// result device-computed. Returns the tensor.
fn resident<R>(what: &str, f: impl FnOnce() -> R) -> R {
    let before = tensor_traffic();
    let (r, report) = with_report(f);
    assert_eq!(tensor_traffic().uploads, before.uploads, "{what} uploaded");
    assert_eq!(
        tensor_traffic().downloads,
        before.downloads,
        "{what} downloaded"
    );
    assert_native_model(&report);
    let op = report
        .op("float_random")
        .or_else(|| report.op("int_random"))
        .unwrap_or_else(|| panic!("{what}: no random op in {report:?}"));
    assert_eq!(
        (op.on_host, op.staged, op.downloads),
        (0, 0, 0),
        "{what}: {op:?}"
    );
    assert!(op.on_device > 0, "{what}: not drawn on the device: {op:?}");
    r
}

#[test]
fn draws_are_resident_seeded_and_equal_the_model() {
    with_device(Config::default(), |d| {
        let dims = [70, 50];
        TtBackend::seed(&d, SEED);
        let t: Tensor<TtBackend, 2> = resident("default", || {
            Tensor::random(dims, Distribution::Default, &d)
        });
        let p = float_primitive(t.clone());
        assert!(p.computed_on_device(), "a device draw is not host data");
        let first = t.into_data().to_vec::<f32>().unwrap();
        let want = units_of(&expected(dims, base(SEED, 1), role::UNIT, target(), true));
        assert_eq!(
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the first draw is the model's tile stream for (seed, 1)"
        );
        assert!(first.iter().all(|&u| (0.0..1.0).contains(&u)));
        // The next draw is the next base: different bits, the model's again.
        let second = Tensor::<TtBackend, 2>::random(dims, Distribution::Default, &d)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_ne!(first, second, "consecutive draws differ");
        assert_eq!(
            second,
            units_of(&expected(dims, base(SEED, 2), role::UNIT, target(), true))
        );
        // Re-seeding repeats the whole sequence.
        TtBackend::seed(&d, SEED);
        let again = Tensor::<TtBackend, 2>::random(dims, Distribution::Default, &d)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(again, first, "same seed, same first draw");
        // A different seed is a different stream.
        TtBackend::seed(&d, SEED + 1);
        let other = Tensor::<TtBackend, 2>::random(dims, Distribution::Default, &d)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_ne!(other, first);
    });
}

#[test]
fn every_distribution_is_its_documented_function_of_the_model_stream() {
    with_device(Config::default(), |d| {
        let dims = [37, 45]; // ragged: padding is random and never read back
        TtBackend::seed(&d, SEED);
        let mut n = 0u64;
        let mut draw = || {
            n += 1;
            base(SEED, n)
        };
        let units = |role, b| units_of(&expected(dims, b, role, target(), true));
        // Uniform [lo, hi).
        let (lo, hi) = (-2.0f32, 3.0f32);
        let b = draw();
        let u = units(role::UNIFORM, b);
        let t: Tensor<TtBackend, 2> = resident("uniform", || {
            Tensor::random(
                dims,
                Distribution::Uniform(f64::from(lo), f64::from(hi)),
                &d,
            )
        });
        let got = t.into_data().to_vec::<f32>().unwrap();
        for (i, (&g, &x)) in got.iter().zip(&u).enumerate() {
            let host = ((x * (hi - lo)) + lo).min(next_down(hi));
            assert_eq!(g.to_bits(), host.to_bits(), "uniform element {i}");
            assert!(g >= lo && g < hi);
            let real = f64::from(lo) + f64::from(x) * f64::from(hi - lo);
            assert!((f64::from(g) - real).abs() <= uniform_bound(lo, hi));
        }
        // Bernoulli, float and int.
        let b = draw();
        let u = units(role::BERNOULLI, b);
        let p = 0.3f64;
        let t: Tensor<TtBackend, 2> = resident("bernoulli", || {
            Tensor::random(dims, Distribution::Bernoulli(p), &d)
        });
        let got = t.into_data().to_vec::<f32>().unwrap();
        for (i, (&g, &x)) in got.iter().zip(&u).enumerate() {
            assert_eq!(
                g,
                f32::from(u8::from(x < p as f32)),
                "bernoulli element {i}"
            );
        }
        let b = draw();
        let u = units(role::BERNOULLI, b);
        let t: Tensor<TtBackend, 2, Int> = resident("int bernoulli", || {
            Tensor::random(dims, Distribution::Bernoulli(p), &d)
        });
        let got = t.into_data().to_vec::<i32>().unwrap();
        for (i, (&g, &x)) in got.iter().zip(&u).enumerate() {
            assert_eq!(g, i32::from(x < p as f32), "int bernoulli element {i}");
        }
        // Normal(mean, std): the f64 Box-Muller within the derived bound.
        let (mean, std) = (1.5f32, 0.5f32);
        let b = draw();
        let (u1, u2) = (units(role::NORMAL_RADIUS, b), units(role::NORMAL_ANGLE, b));
        let t: Tensor<TtBackend, 2> = resident("normal", || {
            Tensor::random(
                dims,
                Distribution::Normal(f64::from(mean), f64::from(std)),
                &d,
            )
        });
        let got = t.into_data().to_vec::<f32>().unwrap();
        for (i, ((&g, &a), &c)) in got.iter().zip(&u1).zip(&u2).enumerate() {
            let x = 1.0 - f64::from(a);
            let theta = f64::from(c * TWO_PI);
            let z = (-2.0 * x.ln()).sqrt() * theta.cos();
            let want = f64::from(mean) + f64::from(std) * z;
            let bound = normal_bound(z, want, f64::from(std));
            assert!(
                (f64::from(g) - want).abs() <= bound,
                "normal element {i}: {g} vs {want}"
            );
        }
        // Integers: every 32 bits (Default) and a range.
        let b = draw();
        let words = prng_support::expected(dims, b, role::INT_WORD, target(), false);
        let t: Tensor<TtBackend, 2, Int> = resident("int default", || {
            Tensor::random(dims, Distribution::Default, &d)
        });
        let got = t.into_data().to_vec::<i32>().unwrap();
        assert_eq!(
            got.iter().map(|&v| v as u32).collect::<Vec<_>>(),
            words,
            "int Default is the raw mixed words"
        );
        let b = draw();
        let u = units(role::INT_RANGE, b);
        let t: Tensor<TtBackend, 2, Int> = resident("int uniform", || {
            Tensor::random(dims, Distribution::Uniform(-7.0, 1000.0), &d)
        });
        let got = t.into_data().to_vec::<i32>().unwrap();
        for (i, (&g, &x)) in got.iter().zip(&u).enumerate() {
            assert_eq!(g, -7 + (x * 1007.0) as i32, "int uniform element {i}");
        }
    });
}

#[test]
fn dropout_composes_native_bernoulli_and_the_mask_multiply() {
    type A = Autodiff<TtBackend>;
    with_device(Config::default(), |d| {
        let dims = [33, 40];
        A::seed(&d, SEED);
        // A device-computed input (a draw), so nothing is uploaded at all.
        let x: Tensor<A, 2> = Tensor::random(dims, Distribution::Default, &d);
        let x_host = x.clone().into_data().to_vec::<f32>().unwrap();
        let before = tensor_traffic();
        let (y, report) =
            with_report(|| burn::nn::DropoutConfig::new(0.3).init().forward(x.clone()));
        assert_eq!(tensor_traffic().uploads, before.uploads, "dropout uploaded");
        assert_eq!(
            tensor_traffic().downloads,
            before.downloads,
            "dropout downloaded"
        );
        assert_native_model(&report);
        let random = report
            .op("float_random")
            .expect("the mask is a device draw");
        assert!(random.on_device > 0 && random.on_host == 0, "{random:?}");
        assert!(
            report.op("float_mul").is_some(),
            "x * mask is a device multiply"
        );
        // Oracle: the mask is `u < 0.7` of the model stream of the draw that
        // followed the input's (draw 2); the output `(x * mask) * (1 / 0.7)`.
        let u = units_of(&expected(
            dims,
            base(SEED, 2),
            role::BERNOULLI,
            target(),
            true,
        ));
        let keep = (1.0 - 0.3f64) as f32;
        let scale = (1.0 / (1.0 - 0.3f64)) as f32;
        let got = y.into_data().to_vec::<f32>().unwrap();
        let mut kept = 0usize;
        for (i, ((&g, &x), &m)) in got.iter().zip(&x_host).zip(&u).enumerate() {
            let mask = f32::from(u8::from(m < keep));
            kept += usize::from(mask == 1.0);
            assert_eq!(g.to_bits(), ((x * mask) * scale).to_bits(), "element {i}");
        }
        // About 70% kept: |kept - 0.7 n| within 4.9 sd.
        let n = got.len() as f64;
        let z = (kept as f64 - 0.7 * n) / (n * 0.7 * 0.3).sqrt();
        assert!(z.abs() < 4.9, "kept {kept} of {n}: z = {z}");
    });
}

/// The panic message of `f`.
fn panic_text(f: impl FnOnce()) -> String {
    let p = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).expect_err("must fail");
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

// A device failure is sticky (`ids.failed`): one failing draw per device.

#[test]
fn a_bad_probability_fails_explicitly_naming_the_draw() {
    with_device(Config::default(), |d| {
        let bad: Tensor<TtBackend, 2> = Tensor::random([8, 8], Distribution::Bernoulli(1.5), &d);
        let message = panic_text(|| drop(bad.into_data()));
        assert!(
            message.contains("native random") && message.contains("Bernoulli probability 1.5"),
            "{message}"
        );
    });
}

#[test]
fn an_integer_range_beyond_the_multiply_shift_fails_explicitly() {
    with_device(Config::default(), |d| {
        let wide: Tensor<TtBackend, 2, Int> =
            Tensor::random([8, 8], Distribution::Uniform(0.0, 1.0e9), &d);
        let message = panic_text(|| drop(wide.into_data()));
        assert!(message.contains("integer uniform"), "{message}");
    });
}

#[test]
fn an_empty_interval_fails_explicitly() {
    with_device(Config::default(), |d| {
        let empty: Tensor<TtBackend, 2> =
            Tensor::random([8, 8], Distribution::Uniform(1.0, 1.0), &d);
        let message = panic_text(|| drop(empty.into_data()));
        assert!(
            message.contains("not a finite, nonempty interval"),
            "{message}"
        );
    });
}

#[test]
fn an_unsupported_dtype_fails_naming_shape_and_dtype() {
    use burn::tensor::ops::FloatTensorOps;
    with_device(Config::default(), |d| {
        let message = panic_text(|| {
            drop(<TtBackend as FloatTensorOps<TtBackend>>::float_random(
                [4, 4].into(),
                Distribution::Default,
                &d,
                burn::tensor::FloatDType::F64,
            ))
        });
        assert!(
            message.contains("float_random") && message.contains("F64"),
            "{message}"
        );
    });
}

/// A trace replays its jobs verbatim, so a draw captured in it would repeat the
/// same bits every replay. The draw is refused; the capture itself ends empty
/// (`TraceError::Empty`) and the device reports the refusal on its next use.
#[test]
fn a_trace_refuses_to_capture_a_draw() {
    use burn_tt::TracedInference;
    with_device(Config::default(), |d| {
        let x: Tensor<TtBackend, 2> = Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &d);
        let xp = float_primitive(x.clone());
        xp.ensure_resident();
        let mut product = None;
        let captured = TracedInference::capture(&[&xp], || {
            let r: Tensor<TtBackend, 2> = Tensor::random([2, 2], Distribution::Default, &d);
            let y = x.clone() * r;
            product = Some(y.clone());
            vec![float_primitive(y)]
        });
        assert!(captured.is_err(), "a trace captured a random draw");
        let message = panic_text(|| drop(product.unwrap().into_data()));
        assert!(
            message.contains("random inside a trace is refused"),
            "{message}"
        );
    });
}
