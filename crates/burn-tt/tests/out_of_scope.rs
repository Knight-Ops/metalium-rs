//! Operations hardware coverage deliberately does not provide (the `[-]` dispositions in
//! `docs/plans/burn-op-coverage.md`). Each must fail explicitly with
//! its name and the input metadata, never fall back to a host computation.
//! Needs no hardware: refusal inspects metadata only.
use burn_backend::ops::{
    ConvOptions, ConvTransposeOptions, InterpolateMode, InterpolateOptions, ModuleOps,
};
use burn_backend::tensor::FloatTensor;
use burn_backend::TensorPrimitive;
use burn_tensor::Tensor;
use burn_tt::{TtBackend, TtDevice};

fn primitive<const D: usize>(shape: [usize; D], d: &TtDevice) -> FloatTensor<TtBackend> {
    match Tensor::<TtBackend, D>::ones(shape, d).into_primitive() {
        TensorPrimitive::Float(t) => t,
        TensorPrimitive::QFloat(_) => unreachable!("ones is a float tensor"),
    }
}

fn refusal(f: impl FnOnce()) -> String {
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_err();
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a refusal carries a message")
}

#[track_caller]
fn assert_refused(message: &str, op: &str, shape: &str) {
    assert!(
        message.contains(&format!("unsupported operation {op}")),
        "{message}"
    );
    assert!(message.contains(shape), "{message}");
    assert!(message.contains("F32"), "{message}");
}

#[test]
fn volumetric_convolutions_are_refused_by_name() {
    let d = TtDevice::new(310);
    let (x, w) = (
        primitive([1, 1, 2, 2, 2], &d),
        primitive([1, 1, 1, 1, 1], &d),
    );
    let message = refusal(move || {
        let _ = TtBackend::conv3d(x, w, None, ConvOptions::new([1; 3], [0; 3], [1; 3], 1));
    });
    assert_refused(&message, "conv3d", "[1, 1, 2, 2, 2]");

    let (x, w) = (
        primitive([1, 1, 2, 2, 2], &d),
        primitive([1, 1, 1, 1, 1], &d),
    );
    let message = refusal(move || {
        let _ = TtBackend::conv_transpose3d(
            x,
            w,
            None,
            ConvTransposeOptions::new([1; 3], [0; 3], [0; 3], [1; 3], 1),
        );
    });
    assert_refused(&message, "conv_transpose3d", "[1, 1, 2, 2, 2]");
}

/// `interpolate` is native (`step155_interpolate`); what it still refuses, it refuses by name.
#[test]
fn interpolation_beyond_its_limits_is_refused_by_name() {
    let d = TtDevice::new(311);
    // Lanczos3's 36 taps over 700 x 700 outputs: more host addresses than one gather stages.
    let x = primitive([1, 1, 2, 2], &d);
    let message = refusal(move || {
        let _ = TtBackend::interpolate(
            x,
            [700, 700],
            InterpolateOptions::new(InterpolateMode::Lanczos3),
        );
    });
    assert_refused(&message, "interpolate", "[1, 1, 2, 2]");
    assert!(message.contains("limit"), "{message}");

    // The gradient has the same limit, and names itself.
    let (x, grad) = (primitive([1, 1, 2, 2], &d), primitive([1, 1, 700, 700], &d));
    let message = refusal(move || {
        let _ = TtBackend::interpolate_backward(
            x,
            grad,
            [700, 700],
            InterpolateOptions::new(InterpolateMode::Lanczos3),
        );
    });
    assert_refused(&message, "interpolate_backward", "[1, 1, 2, 2]");

    // An empty input plane has nothing to interpolate (burn-flex asserts the same).
    let x = primitive([1, 1, 0, 2], &d);
    let message = refusal(move || {
        let _ =
            TtBackend::interpolate(x, [4, 4], InterpolateOptions::new(InterpolateMode::Nearest));
    });
    assert_refused(&message, "interpolate", "[1, 1, 0, 2]");
    assert!(message.contains("spatial dimensions"), "{message}");
}

#[test]
fn fourier_transforms_are_refused_by_name() {
    let d = TtDevice::new(312);
    let signal = primitive([2, 8], &d);
    let message = refusal(move || {
        let _ = TtBackend::rfft(signal, 1, None);
    });
    assert_refused(&message, "rfft", "[2, 8]");

    let (re, im) = (primitive([2, 5], &d), primitive([2, 5], &d));
    let message = refusal(move || {
        let _ = TtBackend::irfft(re, im, 1, None);
    });
    assert_refused(&message, "irfft", "[2, 5]");
}
