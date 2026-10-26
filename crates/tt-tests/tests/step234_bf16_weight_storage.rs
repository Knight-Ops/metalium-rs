//! A model whose weights are stored BF16 (`Llama::with_bf16_weights`: half the bytes a decode step
//! reads) generates the F32 model's tokens and logits **bit for bit**, traced, wherever its weights
//! hold BF16 values -- as a BF16 checkpoint widened to F32 does (`docs/plans/bf16-weight-storage.md`
//! A3). Checked on Llama shapes SmolLM2 lacks (grouped-query attention, an untied head, head width
//! 128, ragged widths and vocabularies) and on SmolLM2's own checkpoint (`weights` feature).
//!
//! - The weights are rounded to BF16 first (`quant::Emulated`, every matrix, the embedding table
//!   too, since a tied head is the table): then the F32 model and the stored one hold the same
//!   values, and the claim is that the card multiplies them alike.
//! - The stored model's matrices really are BF16 storage, its head untied, and no op widens one
//!   (a product reads it as it lies).
//! - **Negative control:** the same storage of weights that were not rounded first gives other
//!   logits (the storage rounds them; the F32 model keeps their low bits).
//!
//! Silicon only: storing narrows through the packer's BF16 mode, which ttsim refuses (divergence
//! "BF16 late narrowing"); `step233` checks the products on ttsim from host-made BF16 tiles.

mod ladder_support;

use burn_tt::storage::{StorageFormat, TensorStorageExt};
use burn_tt::{TtBackend, TtDevice};
use ladder_support::{assert_same, generate, prompt, traced};
use tt_models::generate::{generate_fixed, Shapes};
use tt_models::llama::{Llama, LlamaConfig, Precision, SMOLLM2_135M};
use tt_models::quant::{apply, Emulated, Format, Plan};
use tt_tests::burn_device::{with_device, Config};

const MAX_LEN: usize = 16;
const BUCKETS: [usize; 2] = [4, 8];
const LENGTHS: [usize; 4] = [1, 5, 8, 12];
const STEPS: usize = 3;

fn shapes() -> Shapes {
    Shapes::new(MAX_LEN, &BUCKETS)
}

fn synthetic_configs() -> Vec<(&'static str, LlamaConfig)> {
    let base = LlamaConfig {
        hidden: 64,
        heads: 2,
        kv_heads: 2,
        head_dim: 32,
        ff: 96,
        vocab: 100,
        layers: 2,
        rope_theta: 10_000.0,
        eps: 1e-5,
        tied_head: true,
    };
    vec![
        ("multi-head, tied, ragged vocabulary", base),
        (
            "grouped-query 4:1, untied",
            LlamaConfig {
                hidden: 128,
                heads: 4,
                kv_heads: 1,
                ff: 160,
                vocab: 64,
                tied_head: false,
                ..base
            },
        ),
        (
            "head width 128, ragged width, untied",
            LlamaConfig {
                hidden: 96,
                heads: 2,
                kv_heads: 1,
                head_dim: 128,
                ff: 200,
                vocab: 77,
                tied_head: false,
                ..base
            },
        ),
    ]
}

/// `model` with every matrix's values rounded to BF16, held in F32.
fn rounded(model: Llama<TtBackend>) -> Llama<TtBackend> {
    apply(model, &Plan::new().rule("", Format::Bf16Stored), &Emulated)
}

/// The F32 model and the stored one, traced, over every prompt length: the same generation.
fn assert_stored_is_f32(
    f32_model: &Llama<TtBackend>,
    stored: &Llama<TtBackend>,
    what: &str,
    d: &TtDevice,
) {
    let vocab = f32_model.config().vocab;
    // The products read the stored weights as they are: nothing widens one to F32 first.
    let (_, report) = burn_tt::with_report(|| {
        generate_fixed(stored, &prompt(5, vocab), 1, &shapes(), d).unwrap()
    });
    assert!(
        report.0.iter().all(|s| s.op != "with_storage"),
        "{what}: a stored weight was widened"
    );
    let (a, b) = (
        traced(f32_model, &shapes(), d),
        traced(stored, &shapes(), d),
    );
    for len in LENGTHS {
        let p = prompt(len, vocab);
        let want = generate(&a, &p, STEPS);
        let got = generate(&b, &p, STEPS);
        assert_same(&got, &want, &format!("{what}, prompt {len}"));
    }
}

/// Every matrix of `model` but the embedding table is BF16 storage.
fn assert_stored(model: &Llama<TtBackend>) {
    use burn::module::{Module, ModuleVisitor, Param};
    use burn::tensor::Tensor;
    struct Check(Vec<String>, usize);
    impl ModuleVisitor<TtBackend> for Check {
        fn enter_module(&mut self, name: &str, _: &str) {
            self.0.push(name.into());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.0.pop();
        }
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<TtBackend, D>>) {
            let path = self.0.join(".");
            if D == 2 && !path.contains("embed_tokens") {
                assert_eq!(p.val().storage_format(), StorageFormat::Bf16, "{path}");
                self.1 += 1;
            }
        }
    }
    let mut c = Check(Vec::new(), 0);
    model.visit(&mut c);
    assert!(c.1 > 0, "a model with matrices");
}

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "storing narrows through the BF16 packer mode, which ttsim refuses"
)]
fn synthetic_llamas_with_bf16_weights_generate_bit_for_bit() {
    with_device(Config::default(), |d| {
        for (what, config) in synthetic_configs() {
            let model = rounded(Llama::<TtBackend>::new(&config, &d)).fused();
            let stored = model.clone().with_bf16_weights();
            assert_stored(&stored);
            assert_stored_is_f32(&model, &stored, what, &d);
        }
    });
}

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "storing narrows through the BF16 packer mode, which ttsim refuses"
)]
fn weights_that_are_not_bf16_values_do_not_agree() {
    with_device(Config::default(), |d| {
        let (_, config) = synthetic_configs()[1];
        let model = Llama::<TtBackend>::new(&config, &d).fused();
        let stored = model.clone().with_bf16_weights();
        let p = prompt(5, config.vocab);
        let (a, b) = (
            traced(&model, &shapes(), &d),
            traced(&stored, &shapes(), &d),
        );
        let (want, got) = (generate(&a, &p, STEPS), generate(&b, &p, STEPS));
        assert_ne!(
            got.logits, want.logits,
            "storage rounds the low bits the F32 model keeps"
        );
    });
}

/// SmolLM2-135M's own checkpoint is BF16, widened at load: stored BF16 it is the F32 model.
#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-tests/weights"
)]
#[cfg_attr(
    all(feature = "weights", not(feature = "silicon")),
    ignore = "storing narrows through the BF16 packer mode, which ttsim refuses"
)]
fn smollm2_with_bf16_weights_generates_bit_for_bit() {
    let config = Config {
        tiles: Some(burn_tt::TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        let (model, _) =
            Llama::<TtBackend>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &d).unwrap();
        let model = model.fused();
        let stored = model.clone().with_bf16_weights();
        assert_stored(&stored);
        assert_stored_is_f32(&model, &stored, "SmolLM2-135M", &d);
    });
}
