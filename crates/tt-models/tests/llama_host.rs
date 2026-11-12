//! SmolLM2-135M on the host: the pretrained model loads completely and reproduces the reference
//! implementation (HuggingFace `transformers`) on real prompts.
//!
//! The host run has to be right before the device run is compared with it (see `resnet18_host`).
//! The prompt "The capital of France is" was tokenised with the model's own tokenizer
//! (`tokenizer.json` at the pinned revision, sha256
//! 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c) and the ids are fixed here, so
//! nothing in the build tokenises. Needs `cargo xtask fetch-weights`.

use burn::tensor::{DType, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_models::llama::{Llama, Precision, SMOLLM2_135M};

/// "The capital of France is"
const PROMPT: [i32; 5] = [504, 3575, 282, 4649, 314];
/// " Paris"
const PARIS: usize = 7042;

fn tokens(ids: &[i32]) -> Tensor<Flex, 2, Int> {
    Tensor::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), &FlexDevice)
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-models/weights"
)]
fn the_pretrained_model_loads_completely() {
    let (_, loaded) =
        Llama::<Flex>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &FlexDevice).unwrap();
    // 30 layers of 9 tensors, the embedding table and the final norm; the output head is tied.
    assert_eq!(loaded.applied, 272, "{loaded:?}");
    assert!(loaded.ignored.is_empty(), "{:?}", loaded.ignored);
}

/// What HuggingFace `transformers` 5.19.0 (torch 2.14.1, CPU, `float32`) gives for the pinned
/// checkpoint: the ten largest logits after each prompt, in order. This is the independent oracle;
/// our own run on the host must reproduce it before any comparison with the device means anything.
const FRANCE_TOP10: [usize; 10] = [260, 7042, 3807, 1217, 253, 13010, 281, 216, 1343, 441];
const FRANCE_LOGITS: [f32; 10] = [
    14.02282, 12.99731, 12.74723, 12.23693, 12.12465, 12.03612, 11.24277, 10.9279, 10.78081,
    10.69593,
];
const FIB_TOP10: [usize; 10] = [472, 198, 3136, 448, 3805, 1004, 2738, 907, 2367, 7116];

fn top10(logits: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    order.truncate(10);
    order
}

fn run(model: &Llama<Flex>, ids: &[i32]) -> Vec<f32> {
    let mut cache = model.cache(32, DType::F32, &FlexDevice);
    model
        .forward(tokens(ids), &mut cache, DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-models/weights"
)]
fn the_host_model_reproduces_the_reference_implementation() {
    let (model, _) =
        Llama::<Flex>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &FlexDevice).unwrap();
    let logits = run(&model, &PROMPT);
    assert_eq!(logits.len(), SMOLLM2_135M.vocab);
    let top = top10(&logits);
    assert_eq!(top, FRANCE_TOP10, "the ten most likely next tokens");
    // Greedy decoding says " the" (260), with " Paris" (7042) second: a base model this small.
    assert_eq!(top[0], 260);
    assert_eq!(top[1], PARIS);

    // Measured against the reference: 3e-5 at most. The guard is far looser than the
    // measurement, because summation order differs between CPUs and libraries; it exists to catch a
    // wrong layer, which moves a logit by whole units, not to state an error bound.
    let worst = top
        .iter()
        .zip(FRANCE_LOGITS)
        .map(|(&i, want)| (logits[i] - want).abs())
        .fold(0f32, f32::max);
    eprintln!("largest logit difference from the reference: {worst:e}");
    assert!(worst < 1e-3, "{worst}");

    let fib = run(&model, &[1604, 3987, 46477, 24, 94, 727]);
    assert_eq!(top10(&fib), FIB_TOP10, "\"def fibonacci(n):\"");
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-models/weights"
)]
fn a_cached_decode_equals_a_full_prefill() {
    // The same six tokens, once as one prompt and once as five plus one through the cache.
    // Both are the same arithmetic in a different order of work, so they agree on the host
    // to the last bits that attention's summation order can move.
    let (model, _) = Llama::<Flex>::smollm2_135m(4, Precision::F32, &FlexDevice).unwrap();
    let ids = [504, 3575, 282, 4649, 314, 7042];
    let mut full = model.cache(16, DType::F32, &FlexDevice);
    let whole = model
        .forward(tokens(&ids), &mut full, DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let mut step = model.cache(16, DType::F32, &FlexDevice);
    model.forward(tokens(&ids[..5]), &mut step, DType::F32);
    let cached = model
        .forward(tokens(&ids[5..]), &mut step, DType::F32)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let max = whole
        .iter()
        .zip(&cached)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = whole.iter().map(|v| v.abs()).fold(0f32, f32::max);
    eprintln!("prefill vs cache: max diff {max:e} of {scale}");
    assert!(max <= 1e-4 * scale, "{max:e} of {scale}");
    assert_eq!(argmax(&whole), argmax(&cached));
}
