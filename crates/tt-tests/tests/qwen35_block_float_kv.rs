//! A Qwen3.5 with its KV cache stored block-float (`WeightOptions::kv_block_float`,
//! `docs/plans/block-float-kv.md` K4): BFP8 or BFP4 keys and values, written as encoded rows and
//! read where they lie.
//!
//! The claims, for BFP8 and BFP4:
//! - the new cache's keys and values are stored in the format (not BF16, not F32);
//! - the traced generation is the eager fixed-shape schedule's, bit for bit: a capture writes the
//!   cache in place as an eager step writes a copy, and both attend over the same stored values;
//! - **negative control:** the logits differ from the same model's with a BF16 cache, so the
//!   format is in effect.
//!
//! The model is Qwen3.5-4B's shape at four layers (three DeltaNets and a full-attention layer)
//! with a vocabulary of 4,096, its weights drawn, under the performance contract. A model
//! workload: silicon only.

mod ladder_support;

use burn_tt::storage::{BfpFormat, StorageFormat, TensorStorageExt};
use burn_tt::{Numerics, TtBackend};
use tt_models::generate::{generate_fixed, FixedShapeDecoder, Shapes};
use tt_models::quant::WeightOptions;
use tt_models::qwen35::{Qwen35, Qwen35Config, QWEN3_5_4B};
use tt_tests::burn_device::{with_device, Config};

const CONFIG: Qwen35Config = Qwen35Config {
    vocab: 4096,
    layers: 4,
    ..QWEN3_5_4B
};

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask silicon --release --filter qwen35_block_float_kv"
)]
fn a_block_float_cache_generates_traced_as_eager() {
    let prompts = [5, 70].map(|n| ladder_support::prompt(n, CONFIG.vocab));
    let shapes = Shapes::new(128, &[8, 16]).with_prefixes(&[]);
    let steps = 6;
    let config = Config {
        tiles: Some(burn_tt::TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        let model = Qwen35::<TtBackend>::new(&CONFIG, &d);
        let bf16 = model
            .clone()
            .for_inference(Numerics::Performance, WeightOptions::default());
        let bf16_logits: Vec<_> = prompts
            .iter()
            .map(|p| generate_fixed(&bf16, p, steps, &shapes, &d).unwrap())
            .collect();
        for (format, storage) in [
            (BfpFormat::Bfp8, StorageFormat::Bfp8),
            (BfpFormat::Bfp4, StorageFormat::Bfp4),
        ] {
            let m = model.clone().for_inference(
                Numerics::Performance,
                WeightOptions {
                    kv_block_float: Some(format),
                    ..WeightOptions::default()
                },
            );
            assert_eq!(m.kv_block_float(), Some(format));
            let cache = m.new_cache(shapes.max_len, &d);
            let kv = cache.kv().expect("a full-attention layer").state();
            assert!(!kv.is_empty());
            for t in kv {
                assert_eq!(t.storage_format(), storage, "{format:?}: the cache's keys and values");
            }
            let g = ladder_support::traced(&m, &shapes, &d);
            for (p, bf16) in prompts.iter().zip(&bf16_logits) {
                let what = format!("{format:?}, prompt {}", p.len());
                let got = ladder_support::generate(&g, p, steps);
                let eager = generate_fixed(&m, p, steps, &shapes, &d).unwrap();
                ladder_support::assert_same(&got, &eager, &format!("{what}: traced against eager"));
                let moved = got
                    .logits
                    .iter()
                    .zip(&bf16.logits)
                    .filter(|(a, b)| ladder_support::bits(a) != ladder_support::bits(b))
                    .count();
                assert!(moved > 0, "{what}: the logits are the BF16 cache's");
                eprintln!(
                    "{what}: traced = eager; tokens {:?} (BF16 cache {:?}), {moved} of {} predictions moved",
                    got.tokens,
                    bf16.tokens,
                    got.logits.len()
                );
            }
        }
    });
}

/// The same model through the fusion backend (`burn_tt::Tt`, what `codeki-tensix` runs): traced
/// against eager, a BF16 cache and a BFP8 one, each with chunks of 16 into a cache of 128.
#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask silicon --release --filter qwen35_block_float_kv"
)]
fn through_fusion_a_block_float_cache_generates_traced_as_eager() {
    use tt_models::generate::TracedGenerator;
    let prompt = ladder_support::prompt(70, CONFIG.vocab);
    let shapes = Shapes::new(128, &[16]).with_prefixes(&[]);
    let config = Config {
        tiles: Some(burn_tt::TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        let model = Qwen35::<burn_tt::Tt>::new(&CONFIG, &d);
        for kv in [None, Some(BfpFormat::Bfp8)] {
            let m = model.clone().for_inference(
                Numerics::Performance,
                WeightOptions {
                    kv_block_float: kv,
                    ..WeightOptions::default()
                },
            );
            let g = TracedGenerator::<_, burn_tt::Tt>::new(&m, &shapes, &d).unwrap();
            let got = g.generate(&prompt, 4).unwrap();
            let eager = generate_fixed(&m, &prompt, 4, &shapes, &d).unwrap();
            assert_eq!(got.tokens, eager.tokens, "{kv:?}: tokens");
            for (a, b) in got.logits.iter().zip(&eager.logits) {
                assert_eq!(ladder_support::bits(a), ladder_support::bits(b), "{kv:?}: logits");
            }
            eprintln!("fusion, {kv:?}: traced = eager, tokens {:?}", got.tokens);
        }
    });
}
