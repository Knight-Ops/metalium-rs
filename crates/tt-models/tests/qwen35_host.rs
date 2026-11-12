//! Qwen3.5-4B on the host: the preset is the checkpoint's text shape, the pretrained language model
//! loads completely, and it reproduces the reference implementation (HuggingFace `transformers`)
//! on a raw prompt, on a chat-templated one, and in a greedy continuation.
//!
//! Every reference value here was printed by `crates/tt-models/reference/qwen35_reference.py`
//! (`transformers` 5.19.0, torch 2.14.1+cpu, `float32`, pinned in the script), on the pinned
//! shards and the revision's `config.json` and tokenizer:
//!
//! ```text
//! uv run crates/tt-models/reference/qwen35_reference.py \
//!     --repo Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
//!     --weights vendor/weights/qwen3.5-4b/model.safetensors-00001-of-00002.safetensors \
//!     --weights vendor/weights/qwen3.5-4b/model.safetensors-00002-of-00002.safetensors \
//!     --prompt "The capital of France is" --chat "What is the capital of France?" --generate 24
//! ```
//!
//! The token ids are fixed here, so nothing in the build tokenises. Needs
//! `cargo xtask fetch-weights --model qwen3.5-4b` (9.3 GB) and about 26 GB of memory while the BF16
//! checkpoint is widened to the F32 model (17 GB).

use burn::tensor::{DType, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_models::generate::EagerDecoder;
use tt_models::llama::Precision;
use tt_models::qwen35::{Qwen35, Qwen35Config, QWEN3_5_4B};

/// The checkpoint's `text_config`, as the reference script prints it (partial rotary factor 0.25,
/// interleaved mRoPE, no attention bias, the output gate on).
const CHECKPOINT: Qwen35Config = Qwen35Config {
    hidden: 2560,
    heads: 16,
    kv_heads: 4,
    head_dim: 256,
    rotary_dim: 64,
    rope_theta: 10000000.0,
    linear_k_heads: 16,
    linear_k_dim: 128,
    linear_v_heads: 32,
    linear_v_dim: 128,
    conv_kernel: 4,
    full_every: 4,
    ff: 9216,
    vocab: 248320,
    layers: 32,
    eps: 1e-06,
    tied_head: true,
};

/// "The capital of France is"
const FRANCE: [i32; 5] = [760, 6511, 314, 9338, 369];
const FRANCE_TOP10: [usize; 10] = [11751, 7172, 264, 3750, 279, 303, 31586, 524, 198, 1259];
const FRANCE_LOGITS: [f32; 10] = [
    18.358, 16.21519, 15.81695, 15.55742, 15.38628, 15.19623, 15.13532, 15.01013, 15.00149,
    14.91174,
];

/// "What is the capital of France?" in the model's chat template (the user turn, then the
/// assistant's opened with its thinking tag), as `apply_chat_template` lays it out.
const CHAT: [i32; 17] = [
    248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068,
    198,
];
const CHAT_TOP10: [usize; 10] = [90700, 31248, 40, 8160, 760, 37405, 79420, 5170, 81266, 12];
const CHAT_LOGITS: [f32; 10] = [
    25.35214, 16.62602, 16.40565, 16.28804, 15.93601, 15.31908, 13.94371, 13.88698, 13.80452,
    13.78062,
];
/// The greedy continuation: "Thinking Process:\n\n1.  **Identify the core question:** The user is
/// asking \"What is the capital of".
const CHAT_ANSWER: [i32; 24] = [
    90700, 8340, 25, 271, 16, 13, 220, 2972, 27382, 1386, 279, 6007, 3296, 64700, 561, 1156, 369,
    9859, 328, 3710, 369, 279, 6511, 314,
];

fn tokens(ids: &[i32]) -> Tensor<Flex, 2, Int> {
    Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), &FlexDevice)
}

fn top10(logits: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    order.truncate(10);
    order
}

fn run(model: &Qwen35<Flex>, ids: &[i32]) -> Vec<f32> {
    let mut state = model.state(64, DType::F32, &FlexDevice);
    model
        .forward(tokens(ids), &mut state, DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

/// The ten most likely next tokens equal the reference's, in order, and their logits are close.
/// Measured: 2.6e-5 and 2.7e-5 at most on the two prompts. The guard is far looser than that,
/// because summation order differs between CPUs and libraries, and here also between the
/// reference's chunked prompt (`torch_chunk_gated_delta_rule`, chunks of 64) and this model's
/// token-by-token recurrence; it exists to catch a wrong layer or a wrong constant, which moves a
/// logit by whole units, not to state an error bound.
fn reproduces(logits: &[f32], want_top: [usize; 10], want_logits: [f32; 10], what: &str) {
    let top = top10(logits);
    assert_eq!(top, want_top, "{what}: the ten most likely next tokens");
    let worst = top
        .iter()
        .zip(want_logits)
        .map(|(&i, want)| (logits[i] - want).abs())
        .fold(0f32, f32::max);
    eprintln!("{what}: largest logit difference from the reference: {worst:e}");
    assert!(worst < 1e-3, "{what}: {worst}");
}

#[test]
fn the_preset_is_the_checkpoints_shape() {
    assert_eq!(QWEN3_5_4B, CHECKPOINT);
}

/// One test, so the 17 GB model is loaded once: loading it completely, then reproducing the
/// reference with it.
#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights --model qwen3.5-4b, then --features tt-models/weights"
)]
fn the_pretrained_model_loads_completely_and_reproduces_the_reference_implementation() {
    let (model, loaded) =
        Qwen35::<Flex>::qwen3_5_4b(QWEN3_5_4B.layers, Precision::F32, &FlexDevice).unwrap();
    // 24 DeltaNets of 9 tensors and 8 attention layers of 6, each layer's two norms and three
    // MLP matrices, the embedding table and the final norm; the output head is tied. Left in the
    // file: the vision tower's 297 tensors and the multi-token-prediction head's 15.
    assert_eq!(loaded.applied, 24 * (9 + 5) + 8 * (6 + 5) + 2, "{loaded:?}");
    assert_eq!(loaded.ignored.len(), 297 + 15, "{:?}", loaded.ignored);
    assert!(
        loaded
            .ignored
            .iter()
            .all(|n| n.starts_with("model.visual.") || n.starts_with("mtp.")),
        "{:?}",
        loaded.ignored
    );

    let france = run(&model, &FRANCE);
    assert_eq!(france.len(), QWEN3_5_4B.vocab);
    reproduces(
        &france,
        FRANCE_TOP10,
        FRANCE_LOGITS,
        "\"The capital of France is\"",
    );
    reproduces(
        &run(&model, &CHAT),
        CHAT_TOP10,
        CHAT_LOGITS,
        "the chat prompt",
    );
    // And it continues: greedy decoding, each step a single token through the carried state,
    // gives the reference's 24 tokens.
    let answer = model.generate(&CHAT, CHAT_ANSWER.len(), Precision::F32, &FlexDevice);
    assert_eq!(answer.tokens, CHAT_ANSWER, "the greedy continuation");
}
