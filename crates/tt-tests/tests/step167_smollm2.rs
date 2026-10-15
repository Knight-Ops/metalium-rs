//! Rung 3 of the model ladder: SmolLM2-135M, a decoder-only language model with a key/value
//! cache, on burn-tt against burn-flex.
//!
//! The same strictly loaded weights and the same prompt go through the model on the host (Flex)
//! and on the device. The host model is checked against HuggingFace `transformers` in
//! `tt-models`'s `llama_host` (identical top-10 and logits to 5e-5), so agreement here is agreement
//! on a model that is right.
//!
//! The simulator lane runs `SMALL_LAYERS` of the thirty layers and two decoded tokens, because the
//! simulator cannot afford more. Silicon (`--features silicon`) runs the whole model in F32, and the
//! checkpoint's own BF16 (BF16 parameters, activations and cache) at six layers and four tokens.
//!
//! What is claimed:
//!
//! * **The model runs on the device**: `assert_native_model` over the whole generation (prefill,
//!   rotary embeddings, grouped-query attention against a cache filled by `slice_assign`, the
//!   tied output projection), so the comparison cannot be a host fallback agreeing with itself.
//! * **Teacher-forced agreement**: both backends are fed the *same* tokens, and the top-1
//!   prediction at every position must agree. Free-running greedy decoding is also run and printed
//!   with the position of the first divergence, if any: two backends that differ in the last bits
//!   can part at a near-tie, which says nothing about either being wrong, so the free-running
//!   comparison is reported and the forced one is the claim.

use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_models::llama::{Generation, Llama, Precision, SMOLLM2_135M};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// "The capital of France is", tokenised with the model's own tokenizer.
const PROMPT: [i32; 5] = [504, 3575, 282, 4649, 314];
/// Layers the simulator lane runs.
const SMALL_LAYERS: usize = 1;

fn layers() -> usize {
    if cfg!(feature = "silicon") {
        SMOLLM2_135M.layers
    } else {
        SMALL_LAYERS
    }
}

fn steps() -> usize {
    if cfg!(feature = "silicon") {
        16
    } else {
        2
    }
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

fn gap(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| b.total_cmp(a));
    s[0] - s[1]
}

fn compare(label: &str, host: &Generation, device: &Generation) {
    let mut worst = 0f32;
    let mut scale = 0f32;
    for (h, d) in host.logits.iter().zip(&device.logits) {
        for (a, b) in h.iter().zip(d) {
            worst = worst.max((a - b).abs());
            scale = scale.max(a.abs());
        }
    }
    eprintln!(
        "{label}: max |flex - device| {worst:.4e} against a largest logit of {scale:.4}; \
         top-1 margin of the closest position {:.4}",
        host.logits.iter().map(|l| gap(l)).fold(f32::MAX, f32::min)
    );
    assert!(
        device.logits.iter().flatten().all(|v| v.is_finite()),
        "a non-finite logit"
    );
    for (i, (h, d)) in host.logits.iter().zip(&device.logits).enumerate() {
        assert_eq!(
            argmax(d),
            argmax(h),
            "{label}: the top-1 prediction at position {i} differs (flex margin {:.4})",
            gap(h)
        );
    }
}

fn run(precision: Precision, layers: usize, steps: usize) {
    let host_model = Llama::<Flex>::smollm2_135m(layers, precision, &FlexDevice)
        .unwrap()
        .0;
    let free_host = host_model.generate(&PROMPT, steps, precision, &FlexDevice);
    let forced_host = host_model.forced(&PROMPT, &free_host.tokens, precision, &FlexDevice);

    with_device(Config::default(), |d| {
        let (model, loaded) = Llama::<TtBackend>::smollm2_135m(layers, precision, &d).unwrap();
        eprintln!(
            "loaded {} tensors, {} layers, {precision:?}",
            loaded.applied, layers
        );
        let ((free, forced), report) = burn_tt::with_report(|| {
            (
                model.generate(&PROMPT, steps, precision, &d),
                model.forced(&PROMPT, &free_host.tokens, precision, &d),
            )
        });
        eprintln!("{report}");
        for (label, g) in [("free-running", &free), ("teacher-forced", &forced)] {
            eprintln!(
                "{label}: prompt {:.2?}, then per token {:?}",
                g.times[0],
                g.times[1..]
                    .iter()
                    .map(|t| format!("{:.0?}", t))
                    .collect::<Vec<_>>()
            );
        }
        assert_native_model(&report);

        compare("teacher-forced", &forced_host, &forced);
        match free
            .tokens
            .iter()
            .zip(&free_host.tokens)
            .position(|(a, b)| a != b)
        {
            None => eprintln!(
                "free-running: {} tokens, identical: {:?}",
                free.tokens.len(),
                free.tokens
            ),
            Some(i) => eprintln!(
                "free-running: first divergence at token {i}: flex {:?}, device {:?}",
                free_host.tokens, free.tokens
            ),
        }
    });
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-tests/weights"
)]
fn smollm2_in_f32_on_the_device_agrees_with_flex() {
    run(Precision::F32, layers(), steps());
}

/// BF16 is the checkpoint's own dtype and the half type this hardware prefers; ttsim refuses the
/// packer mode every BF16 result goes through (`ttsim-divergence.md`, "BF16 late narrowing").
#[cfg(feature = "silicon")]
#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-tests/weights"
)]
fn smollm2_in_bf16_on_the_device_agrees_with_flex() {
    // Six layers and four tokens: through the embedding lookup (15 s a token, more in BF16) the
    // whole model takes over 25 minutes (run `1791847160` timed out at 1,500 s).
    run(Precision::Bf16, 6, 4);
}
