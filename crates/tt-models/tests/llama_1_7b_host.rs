//! SmolLM2-1.7B-Instruct on the host: the preset is the checkpoint's shape, the pretrained model
//! loads completely, and it reproduces the reference implementation (HuggingFace `transformers`) on
//! a raw prompt and on a chat-templated one.
//!
//! Every reference value here was printed by `crates/tt-models/reference/llama_reference.py`
//! (`transformers` 5.19.0, torch 2.14.1+cpu, `float32`, pinned in the script), on the pinned
//! `model.safetensors` and the revision's `config.json` and tokenizer:
//!
//! ```text
//! uv run crates/tt-models/reference/llama_reference.py \
//!     --repo HuggingFaceTB/SmolLM2-1.7B-Instruct \
//!     --revision 31b70e2e869a7173562077fd711b654946d38674 \
//!     --weights vendor/weights/smollm2-1.7b-instruct/model.safetensors \
//!     --prompt "The capital of France is" --chat "What is the capital of France?" --generate 16
//! ```
//!
//! The token ids are fixed here, so nothing in the build tokenises. Needs
//! `cargo xtask fetch-weights --model smollm2-1.7b-instruct` (3.4 GB) and about 7 GB of memory for
//! the F32 model.

use burn::tensor::{DType, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_models::llama::{Llama, LlamaConfig, Precision, SMOLLM2_1_7B_INSTRUCT};

/// The checkpoint's `config.json`, as the reference script prints it (`rope_scaling` default, no
/// attention or MLP bias).
const CHECKPOINT: LlamaConfig = LlamaConfig {
    hidden: 2048,
    heads: 32,
    kv_heads: 32,
    head_dim: 64,
    ff: 8192,
    vocab: 49152,
    layers: 24,
    rope_theta: 130000.0,
    eps: 1e-05,
    tied_head: true,
};

/// "The capital of France is"
const FRANCE: [i32; 5] = [504, 3575, 282, 4649, 314];
const FRANCE_TOP10: [usize; 10] = [7042, 253, 260, 2483, 1129, 933, 1811, 1343, 42, 2007];
const FRANCE_LOGITS: [f32; 10] = [
    19.13401, 15.17946, 14.28281, 13.96916, 13.62437, 13.59968, 13.47444, 13.46691, 13.43538,
    13.31552,
];

/// "What is the capital of France?" in the model's chat template (its default system message, the
/// user turn, then the assistant's turn opened), as `apply_chat_template` lays it out.
const CHAT: [i32; 37] = [
    1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407,
    19712, 8182, 2, 198, 1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47, 2, 198, 1, 520, 9531,
    198,
];
const CHAT_TOP10: [usize; 10] = [
    504, 34355, 38634, 10539, 1653, 47302, 42686, 57, 15635, 5195,
];
const CHAT_LOGITS: [f32; 10] = [
    26.42105, 20.75684, 18.42353, 18.34732, 17.38144, 17.00735, 16.40621, 15.87775, 15.49902,
    15.4255,
];
/// The greedy answer: "The capital of France is Paris.<|im_end|>".
const CHAT_ANSWER: [i32; 8] = [504, 3575, 282, 4649, 314, 7042, 30, 2];

fn tokens(ids: &[i32]) -> Tensor<Flex, 2, Int> {
    Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), &FlexDevice)
}

fn top10(logits: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    order.truncate(10);
    order
}

fn run(model: &Llama<Flex>, ids: &[i32]) -> Vec<f32> {
    let mut cache = model.cache(64, DType::F32, &FlexDevice);
    model
        .forward(tokens(ids), &mut cache, DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

/// The ten most likely next tokens equal the reference's, in order, and their logits are close.
/// Measured: 3.3e-5 and 2.7e-5 at most on the two prompts. The guard is far looser than that,
/// because summation order differs
/// between CPUs and libraries; it exists to catch a wrong layer or a wrong constant, which moves a
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
    assert_eq!(SMOLLM2_1_7B_INSTRUCT, CHECKPOINT);
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-models/weights"
)]
fn the_pretrained_model_loads_completely() {
    let (_, loaded) = Llama::<Flex>::smollm2_1_7b_instruct(
        SMOLLM2_1_7B_INSTRUCT.layers,
        Precision::F32,
        &FlexDevice,
    )
    .unwrap();
    // 24 layers of 9 tensors, the embedding table and the final norm; the output head is tied.
    assert_eq!(loaded.applied, 24 * 9 + 2, "{loaded:?}");
    assert!(loaded.ignored.is_empty(), "{:?}", loaded.ignored);
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-models/weights"
)]
fn the_host_model_reproduces_the_reference_implementation() {
    let (model, _) = Llama::<Flex>::smollm2_1_7b_instruct(
        SMOLLM2_1_7B_INSTRUCT.layers,
        Precision::F32,
        &FlexDevice,
    )
    .unwrap();
    let france = run(&model, &FRANCE);
    assert_eq!(france.len(), SMOLLM2_1_7B_INSTRUCT.vocab);
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
    // And it answers: greedy decoding gives the reference's whole answer, up to its end token.
    let answer = model.generate(&CHAT, CHAT_ANSWER.len(), Precision::F32, &FlexDevice);
    assert_eq!(answer.tokens, CHAT_ANSWER, "the greedy answer");
}
