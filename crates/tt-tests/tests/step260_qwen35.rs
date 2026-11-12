//! Qwen3.5-4B, a hybrid of Gated DeltaNet (linear attention with a recurrent state) and gated full
//! attention, on burn-tt against burn-flex.
//!
//! The same strictly loaded weights and the same prompt go through the model on the host (Flex)
//! and on the device. The host model is checked against HuggingFace `transformers` in
//! `tt-models`'s `qwen35_host` (identical top-10, logits within 2.7e-5 as measured, the reference's
//! 24 greedy tokens), so agreement here is agreement on a model that is right.
//!
//! What is claimed, as `step167_smollm2` claims it for a Llama: the model runs on the device
//! (`assert_native_model` over the whole generation: the depthwise convolution and its carried
//! inputs, the delta-rule recurrence a token at a time, the gated norms, the partial rotary
//! embedding and the output-gated attention against a growing cache, the tied head), and the
//! teacher-forced top-1 agrees at every position. A model workload: silicon only.

mod ladder_support;

use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_models::llama::Precision;
use tt_models::qwen35::{Qwen35, QWEN3_5_4B};
use tt_tests::burn_device::Config;

/// "What is the capital of France?" in the model's chat template (`tests/qwen35_host.rs`).
const PROMPT: [i32; 17] = [
    248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068,
    198,
];

/// The device model: the checkpoint in `precision`, with its matrices stored BF16 when `stored`.
fn run(precision: Precision, stored: bool, steps: usize) {
    ladder_support::flex_then_device(
        Config::default(),
        precision,
        &PROMPT,
        steps,
        || {
            Qwen35::<Flex>::qwen3_5_4b(QWEN3_5_4B.layers, precision, &FlexDevice)
                .unwrap()
                .0
        },
        |d| {
            let model = Qwen35::<TtBackend>::qwen3_5_4b(QWEN3_5_4B.layers, precision, d)
                .unwrap()
                .0;
            if stored {
                model.with_bf16_weights()
            } else {
                model
            }
        },
    );
}

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask fetch-weights --model qwen3.5-4b, then cargo xtask silicon --release --filter step260_qwen35"
)]
fn qwen35_in_f32_on_the_device_agrees_with_flex() {
    run(Precision::F32, false, 8);
}

/// The checkpoint's BF16 weights stored as they are, the arithmetic F32 (what the 1.7B ships). `Precision::Bf16` (BF16 activations) is not gated: on the
/// card it holds over 40 GB of host memory (`docs/plans/new-architectures.md`, phase B).
#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask fetch-weights --model qwen3.5-4b, then cargo xtask silicon --release --filter step260_qwen35"
)]
fn qwen35_with_bf16_weights_on_the_device_agrees_with_flex() {
    run(Precision::F32, true, 8);
}
