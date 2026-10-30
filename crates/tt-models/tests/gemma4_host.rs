//! NVIDIA-Nemotron-3.5-Bolt-E4B (Gemma 4 E4B's text model) on the host: the preset is the
//! checkpoint's text shape, the pretrained language model loads completely, and it reproduces the
//! reference implementation (HuggingFace `transformers`) on a raw prompt, on a prompt longer than
//! the sliding window, on a chat-templated one, and in a greedy answer.
//!
//! Every reference value here was printed by `crates/tt-models/reference/gemma4_reference.py`
//! (`transformers` 5.19.0, torch 2.14.1+cpu, `float32`, eager attention, pinned in the script), on
//! the pinned `model-00001-of-00001.safetensors` and the revision's `config.json` and tokenizer:
//!
//! ```text
//! uv run crates/tt-models/reference/gemma4_reference.py \
//!     --repo nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 \
//!     --revision 49843eed8db1f1c616c47af75cfd36e07aec0667 \
//!     --weights vendor/weights/nemotron-3.5-bolt-e4b/model-00001-of-00001.safetensors \
//!     --prompt "The capital of France is" \
//!     --repeat "The quick brown fox jumps over the lazy dog. " 64 \
//!     --chat "What is the capital of France?" --generate 16
//! ```
//!
//! The token ids are fixed here, so nothing in the build tokenises (the tokenizer puts no
//! beginning-of-text token before raw text; the chat template does). Needs
//! `cargo xtask fetch-weights --model nemotron-3.5-bolt-e4b` (15.9 GB) and about 32 GB of memory.

use burn::tensor::{DType, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_models::gemma4::{AttentionShape, Gemma4, Gemma4Config, NEMOTRON_3_5_BOLT_E4B};
use tt_models::generate::EagerDecoder;
use tt_models::llama::Precision;

/// The checkpoint's `text_config`, as the reference script prints it ("proportional" rotary
/// embedding with a factor of 0.25 on the full layers, the default on the sliding ones; gelu-tanh;
/// the head tied; no MoE block, no double-wide MLP, keys and values separate).
const CHECKPOINT: Gemma4Config = Gemma4Config {
    hidden: 2560,
    heads: 8,
    sliding: AttentionShape {
        head_dim: 256,
        kv_heads: 2,
        rope_theta: 10000.0,
        rotated: 128,
    },
    full: AttentionShape {
        head_dim: 512,
        kv_heads: 2,
        rope_theta: 1000000.0,
        rotated: 64,
    },
    full_every: 6,
    sliding_window: 512,
    kv_shared_layers: 18,
    ff: 10240,
    vocab: 262144,
    layers: 42,
    per_layer: 256,
    eps: 1e-06,
    logit_cap: Some(30.0),
};

/// "The capital of France is"
const FRANCE: [i32; 5] = [818, 5279, 529, 7001, 563];
const FRANCE_TOP10: [usize; 10] = [7001, 1390, 7830, 496, 669, 5213, 506, 711, 808, 107];
const FRANCE_LOGITS: [f32; 10] = [
    9.64537, 1.37648, 1.19109, 1.15058, 0.93218, 0.92861, 0.84352, 0.78136, 0.0111, -0.6575,
];

/// "The quick brown fox jumps over the lazy dog. " 64 times: 704 tokens, past the 512-position
/// window, so the sliding layers' masks hide the start while the full layers still see it.
const FOX: [i32; 11] = [
    818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761, 236743,
];
const FOX_TIMES: usize = 64;
const FOX_TOP10: [usize; 10] = [107, 108, 818, 236770, 2819, 109, 236778, 432, 236819, 106];
const FOX_LOGITS: [f32; 10] = [
    8.27969, 6.892, 6.84464, 3.43107, 2.75291, 2.5585, 2.35923, 2.32047, 1.36659, 1.24285,
];

/// "What is the capital of France?" in the model's chat template (the beginning-of-text token,
/// the user turn, then the model's turn opened), as `apply_chat_template` lays it out.
const CHAT: [i32; 16] = [
    2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107,
];
const CHAT_TOP10: [usize; 10] = [
    818, 1018, 50429, 669, 198, 1437, 89946, 236797, 236780, 1509,
];
const CHAT_LOGITS: [f32; 10] = [
    28.0373, 14.21151, 12.17391, 11.16323, 7.95487, 7.32609, 7.10704, 6.70965, 5.58582, 5.44908,
];
/// The greedy answer, up to the end-of-text token: "The capital of France is **Paris**." and the
/// end-of-turn markers.
const CHAT_ANSWER: [i32; 14] = [
    818, 5279, 529, 7001, 563, 5213, 50429, 84750, 106, 106, 106, 106, 107, 1,
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

fn run(model: &Gemma4<Flex>, ids: &[i32]) -> Vec<f32> {
    let mut cache = model.cache(ids.len(), DType::F32, &FlexDevice);
    model
        .forward(tokens(ids), &mut cache, DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

/// The ten most likely next tokens equal the reference's, in order, and their logits are close.
/// Measured: 4.2e-5, 1.3e-4 (704 positions) and 3.0e-5 at most. The guard is far looser than
/// that, because summation order differs between CPUs and libraries; it exists to catch a wrong layer or a wrong constant, which
/// moves a logit by whole units, not to state an error bound.
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
    assert_eq!(NEMOTRON_3_5_BOLT_E4B, CHECKPOINT);
}

/// One test, so the model is loaded once: loading it completely, then reproducing the reference.
#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights --model nemotron-3.5-bolt-e4b, then --features tt-models/weights"
)]
fn the_pretrained_model_loads_completely_and_reproduces_the_reference_implementation() {
    let (model, loaded) =
        Gemma4::<Flex>::nemotron_3_5_bolt_e4b(Precision::F32, &FlexDevice).unwrap();
    // 42 layers of 17 tensors, less the 18 sharing layers' key and value projections and key
    // norms; the two tables, the per-layer projection and its norm, and the final norm. The head
    // is tied. Left in the file: the vision and audio towers.
    assert_eq!(loaded.applied, 42 * 17 - 18 * 3 + 5, "{loaded:?}");
    assert!(!loaded.ignored.is_empty());
    assert!(
        loaded
            .ignored
            .iter()
            .all(|n| n.contains("vision") || n.contains("audio")),
        "{:?}",
        loaded.ignored
    );

    reproduces(
        &run(&model, &FRANCE),
        FRANCE_TOP10,
        FRANCE_LOGITS,
        "\"The capital of France is\"",
    );
    let fox = FOX.repeat(FOX_TIMES);
    assert!(fox.len() > NEMOTRON_3_5_BOLT_E4B.sliding_window);
    reproduces(
        &run(&model, &fox),
        FOX_TOP10,
        FOX_LOGITS,
        "704 tokens, past the window",
    );
    reproduces(
        &run(&model, &CHAT),
        CHAT_TOP10,
        CHAT_LOGITS,
        "the chat prompt",
    );
    let answer = model.generate(&CHAT, CHAT_ANSWER.len(), Precision::F32, &FlexDevice);
    assert_eq!(answer.tokens, CHAT_ANSWER, "the greedy answer");
}
