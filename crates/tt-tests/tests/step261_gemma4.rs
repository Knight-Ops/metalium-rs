//! NVIDIA-Nemotron-3.5-Bolt-E4B (Gemma 4 E4B's text model: per-layer embeddings, sliding-window
//! and full attention interleaved, layers sharing keys and values, soft-capped logits) on burn-tt
//! against burn-flex.
//!
//! The same strictly loaded weights and the same prompt go through the model on the host (Flex)
//! and on the device. The host model is checked against HuggingFace `transformers` in
//! `tt-models`'s `gemma4_host`, so agreement here is agreement on a model that is right.
//!
//! What is claimed, as `step167_smollm2` claims it for a Llama: the model runs on the device
//! (`assert_native_model` over the whole generation: both embedding lookups (F32 tables; every other
//! matrix stored BF16, F32 arithmetic), the per-layer inputs, both attention kinds with their masks and rotary
//! tables, the shared keys and values, the capped head), and the teacher-forced top-1 agrees at
//! every position. A model workload: silicon only.

mod ladder_support;

use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_models::gemma4::Gemma4;
use tt_models::llama::Precision;
use tt_tests::burn_device::Config;

/// "What is the capital of France?" in the model's chat template (`tests/gemma4_host.rs`).
const PROMPT: [i32; 16] = [
    2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107,
];

fn run(precision: Precision, steps: usize) {
    ladder_support::flex_then_device(
        Config::default(),
        precision,
        &PROMPT,
        steps,
        || {
            Gemma4::<Flex>::nemotron_3_5_bolt_e4b(precision, &FlexDevice)
                .unwrap()
                .0
        },
        |d| {
            // On the card: both tables F32 and every other matrix stored BF16 (F32 arithmetic on
            // the checkpoint's own values). Held as the host holds them, the BF16 tables take F32 tile slots in GDDR
            // and the model does not fit.
            Gemma4::<TtBackend>::nemotron_3_5_bolt_e4b(precision, d)
                .unwrap()
                .0
                .with_f32_tables()
                .with_bf16_weights()
        },
    );
}

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask fetch-weights --model nemotron-3.5-bolt-e4b, then cargo xtask silicon --release --filter step261_gemma4"
)]
fn gemma4_in_f32_on_the_device_agrees_with_flex() {
    run(Precision::F32, 8);
}
