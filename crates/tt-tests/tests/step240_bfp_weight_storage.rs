//! A model whose weights are stored in a block-floating-point format
//! (`Llama::with_stored_weights`: BFP8, BFP4 or BFP2, a quarter to a sixteenth of F32's bytes)
//! generates, traced and eager, the tokens and logits of the F32 model whose weights hold the
//! values that storage decodes to, **bit for bit** (`docs/completed-plans/dataflow-compiler.md` F1c: a
//! packed weight is a product's `B` edge, read where it lies, `SrcRoute::Tf32WithB`). The generic
//! form of `step234`, whose BF16 weights are one stored format.
//!
//! - **The oracle is independent of the kernels.** The F32 model's matrices are rounded on the host
//!   by `quant::bfp_round_trip` (the packer's rule and `tt_isa`'s unpacker model). The stored
//!   model's matrices, read back, must be exactly those values: the card's compression of values
//!   it can hold is exact, and its decode is the model's.
//! - Rounding follows the layout the products read: the model is fused and its head untied first,
//!   so a fused `gate|up` or `q|k|v`, and the head (the table transposed), are rounded as the
//!   matrices they are, not part by part (a block of 16 columns may straddle two parts).
//! - The stored model's matrices really are BFP storage, and no op widens one: each product reads
//!   its weight as it lies.
//! - **Negative control:** the same storage of weights that were not rounded first gives other
//!   logits (the storage rounds them; the F32 model keeps their low bits).
//!
//! BFP8 and BFP4 run on ttsim and silicon; BFP2 on silicon only (ttsim refuses its `Src` format).

mod ladder_support;

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::{Tensor, TensorData};
use burn_tt::storage::{StorageFormat, TensorStorageExt};
use burn_tt::{TtBackend, TtDevice};
use ladder_support::{assert_same, generate, prompt, traced};
use std::collections::BTreeMap;
use tt_models::generate::{generate_fixed, Shapes};
use tt_models::llama::{Llama, LlamaConfig, Precision, SMOLLM2_135M};
use tt_models::quant::{apply, Emulated, Format, Plan};
use tt_tests::burn_device::{with_device, Config};

const MAX_LEN: usize = 16;
const BUCKETS: [usize; 2] = [4, 8];
const LENGTHS: [usize; 3] = [1, 5, 12];
const STEPS: usize = 3;

fn shapes() -> Shapes {
    Shapes::new(MAX_LEN, &BUCKETS)
}

/// The formats this target can store and multiply.
fn formats() -> Vec<Format> {
    let mut v = vec![Format::Bfp8, Format::Bfp4];
    if cfg!(feature = "silicon") {
        v.push(Format::Bfp2);
    }
    v
}

fn storage(format: Format) -> StorageFormat {
    match format {
        Format::Bfp8 => StorageFormat::Bfp8,
        Format::Bfp4 => StorageFormat::Bfp4,
        Format::Bfp2 => StorageFormat::Bfp2,
        other => panic!("{other:?} is not a BFP format"),
    }
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
            "grouped-query 4:1, untied, ragged width",
            LlamaConfig {
                hidden: 96,
                heads: 4,
                kv_heads: 1,
                head_dim: 32,
                ff: 200,
                vocab: 77,
                tied_head: false,
                ..base
            },
        ),
    ]
}

/// `model` fused and untied, every matrix's values rounded to `format` on the host, held in F32:
/// the reference. The stored model is made from it.
fn rounded(model: Llama<TtBackend>, format: Format) -> Llama<TtBackend> {
    apply(
        model.fused().untied(),
        &Plan::new().rule("", format),
        &Emulated,
    )
}

/// Every matrix of `model` but the embedding table, by parameter path, as the values it reads back
/// as (a stored matrix decodes on the card).
fn matrices(model: &Llama<TtBackend>) -> BTreeMap<String, (StorageFormat, Vec<u32>)> {
    struct Collect(Vec<String>, BTreeMap<String, (StorageFormat, Vec<u32>)>);
    impl ModuleVisitor<TtBackend> for Collect {
        fn enter_module(&mut self, name: &str, _: &str) {
            self.0.push(name.into());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.0.pop();
        }
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<TtBackend, D>>) {
            let path = self.0.join(".");
            if D == 2 && !path.contains("embed_tokens") {
                let t = p.val();
                let format = t.storage_format();
                let bits = t
                    .into_data()
                    .to_vec::<f32>()
                    .expect("F32 values")
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                self.1.insert(path, (format, bits));
            }
        }
    }
    let mut c = Collect(Vec::new(), BTreeMap::new());
    model.visit(&mut c);
    assert!(!c.1.is_empty(), "a model with matrices");
    c.1
}

/// Every matrix of `stored` is `format` storage and reads back as `reference`'s host-rounded
/// values, bit for bit.
fn assert_stored_as_rounded(
    reference: &Llama<TtBackend>,
    stored: &Llama<TtBackend>,
    format: Format,
    what: &str,
) {
    let (want, got) = (matrices(reference), matrices(stored));
    assert_eq!(
        want.keys().collect::<Vec<_>>(),
        got.keys().collect::<Vec<_>>(),
        "{what}"
    );
    for (path, (stored_as, bits)) in &got {
        assert_eq!(*stored_as, storage(format), "{what}: {path}");
        let (_, want) = &want[path];
        let differ = bits.iter().zip(want).filter(|(g, w)| g != w).count();
        assert_eq!(
            differ, 0,
            "{what}: {path} ({format:?}): {differ} values differ from the host's rounding"
        );
    }
}

/// The F32 model and the stored one, eager and traced, over every prompt length: the same
/// generation, and nothing widens a stored weight.
fn assert_stored_is_f32(
    reference: &Llama<TtBackend>,
    stored: &Llama<TtBackend>,
    what: &str,
    d: &TtDevice,
) {
    let vocab = reference.config().vocab;
    let p = prompt(5, vocab);
    let (eager, report) =
        burn_tt::with_report(|| generate_fixed(stored, &p, STEPS, &shapes(), d).unwrap());
    assert!(
        report.0.iter().all(|s| s.op != "with_storage"),
        "{what}: a stored weight was widened"
    );
    let want_eager = generate_fixed(reference, &p, STEPS, &shapes(), d).unwrap();
    assert_same(&eager, &want_eager, &format!("{what}, eager"));
    let (a, b) = (
        traced(reference, &shapes(), d),
        traced(stored, &shapes(), d),
    );
    for len in LENGTHS {
        let p = prompt(len, vocab);
        let want = generate(&a, &p, STEPS);
        let got = generate(&b, &p, STEPS);
        assert_same(&got, &want, &format!("{what}, traced, prompt {len}"));
    }
}

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask silicon --release --filter step240_bfp_weight_storage"
)]
fn synthetic_llamas_with_bfp_weights_generate_bit_for_bit() {
    with_device(Config::default(), |d| {
        for format in formats() {
            for (what, config) in synthetic_configs() {
                let what = format!("{what}, {format:?}");
                let reference = rounded(Llama::<TtBackend>::new(&config, &d), format);
                let stored = reference.clone().with_stored_weights(format);
                assert_stored_as_rounded(&reference, &stored, format, &what);
                assert_stored_is_f32(&reference, &stored, &what, &d);
            }
        }
    });
}

/// `model` with its embedding table's values `±1/16` (by sign): every block of every BFP format
/// holds them exactly, along the table's rows or its columns.
fn with_signed_table(model: Llama<TtBackend>) -> Llama<TtBackend> {
    struct Signed(Vec<String>);
    impl ModuleMapper<TtBackend> for Signed {
        fn enter_module(&mut self, name: &str, _: &str) {
            self.0.push(name.into());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.0.pop();
        }
        fn map_float<const D: usize>(
            &mut self,
            p: Param<Tensor<TtBackend, D>>,
        ) -> Param<Tensor<TtBackend, D>> {
            if !self.0.join(".").contains("embed_tokens") {
                return p;
            }
            p.map(|t| {
                let (shape, device) = (t.shape(), t.device());
                let v: Vec<f32> = t
                    .into_data()
                    .to_vec::<f32>()
                    .expect("F32 values")
                    .iter()
                    .map(|x| if *x < 0.0 { -0.0625 } else { 0.0625 })
                    .collect();
                Tensor::from_data(TensorData::new(v, shape), &device)
            })
        }
    }
    model.map(&mut Signed(Vec::new()))
}

/// A tied model stored as a checkpoint is (`with_stored_weights` on the model itself, whose head
/// is then a transposed view of the table): the head is stored as the product reads it, `[hidden,
/// vocab]` in blocks along the vocabulary, and no step widens it (a transposed packed weight
/// would be decoded to F32 at every call).
#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask silicon --release --filter step240_bfp_weight_storage"
)]
fn a_tied_head_is_stored_as_its_product_reads_it() {
    with_device(Config::default(), |d| {
        let (_, config) = synthetic_configs()[0];
        assert!(config.tied_head);
        for format in formats() {
            let what = format!("tied head, {format:?}");
            // Every matrix but the table rounded first; the table's values are held exactly by
            // every blocking, so its transpose (the head) is too.
            let model = apply(
                with_signed_table(Llama::<TtBackend>::new(&config, &d)).fused(),
                &Plan::stored_weights(format),
                &Emulated,
            );
            let reference = model.clone().untied();
            let stored = model.with_stored_weights(format);
            assert_stored_as_rounded(&reference, &stored, format, &what);
            assert_stored_is_f32(&reference, &stored, &what, &d);
        }
    });
}

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask silicon --release --filter step240_bfp_weight_storage"
)]
fn weights_that_are_not_bfp_values_do_not_agree() {
    with_device(Config::default(), |d| {
        let (_, config) = synthetic_configs()[1];
        let model = Llama::<TtBackend>::new(&config, &d).fused().untied();
        let stored = model.clone().with_stored_weights(Format::Bfp8);
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

/// SmolLM2-135M's checkpoint with every weight but the table stored BFP8 and BFP4: the model whose
/// weights hold the same values, bit for bit.
#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "pretrained model, silicon tier: cargo xtask fetch-weights, then cargo xtask silicon --release --filter step240_bfp_weight_storage"
)]
fn smollm2_with_bfp_weights_generates_bit_for_bit() {
    let config = Config {
        tiles: Some(burn_tt::TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        for format in [Format::Bfp8, Format::Bfp4] {
            let (model, _) =
                Llama::<TtBackend>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &d).unwrap();
            let what = format!("SmolLM2-135M, {format:?}");
            let reference = rounded(model, format);
            let stored = reference.clone().with_stored_weights(format);
            assert_stored_as_rounded(&reference, &stored, format, &what);
            assert_stored_is_f32(&reference, &stored, &what, &d);
        }
    });
}

/// "A default holds a weight's bytes once" (roadmap, standing constraints): SmolLM2-135M fused
/// (`Llama::fused`: `q|k|v` and `gate|up` one matrix each, the separate ones dropped) and then
/// stored holds no more GDDR than the same model stored unfused. A fused matrix pads no more than
/// its parts together, so fusing may only save bytes. Each footprint is the GDDR the stored model
/// holds: what was free before it was loaded less what is free while it lives (the F32 checkpoint
/// it was made from is dropped by then). Negative control: before the parts were dropped
/// (2026-11-10, card 0, `1794286677`) the fused model held 543 MB against 399 MB in BF16, its
/// `q`, `k`, `v`, `gate` and `up` stored a second time beside the fused matrices.
#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "pretrained model, silicon tier: cargo xtask fetch-weights, then cargo xtask silicon --release --filter step240_bfp_weight_storage"
)]
fn a_fused_stored_model_holds_each_weight_once() {
    let config = Config {
        tiles: Some(burn_tt::TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        let free = || burn_tt::gddr_free_bytes(d).unwrap().1;
        for format in [Format::Bf16Stored, Format::Bfp8] {
            let held = |fused: bool| {
                let before = free();
                let (model, _) =
                    Llama::<TtBackend>::smollm2_135m(SMOLLM2_135M.layers, Precision::F32, &d)
                        .unwrap();
                let model = if fused { model.fused() } else { model };
                let stored = model.with_stored_weights(format);
                let held = before - free();
                drop(stored);
                held
            };
            let (unfused, fused) = (held(false), held(true));
            println!(
                "BENCH {{\"kind\":\"footprint\",\"model\":\"smollm2-135m\",\"format\":\"{format:?}\",\
                 \"unfused_bytes\":{unfused},\"fused_bytes\":{fused}}}"
            );
            assert!(
                fused <= unfused,
                "{format:?}: the fused model holds {fused} bytes, the unfused one {unfused}: \
                 {} bytes held twice",
                fused - unfused
            );
        }
    });
}
