//! `Qwen35::streamed` (one layer at a time) gives the model `pretrained` -> `for_inference` gives,
//! bit for bit: the same teacher-forced logits at every position, under the performance contract
//! with BFP4 `gate|up`. Streaming is what lets a checkpoint larger than GDDR in F32 reach the card;
//! this shows it changes nothing else. A model workload: silicon only.

mod ladder_support;

use burn_tt::{Numerics, TtBackend};
use tt_models::generate::EagerDecoder;
use tt_models::llama::Precision;
use tt_models::load::weights_files;
use tt_models::quant::WeightOptions;
use tt_models::qwen35::{Qwen35, QWEN3_5_4B};
use tt_tests::burn_device::{with_device, Config};

const FILES: [&str; 2] = [
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
];

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask fetch-weights --model qwen3.5-4b, then cargo xtask silicon --release --filter streamed_load"
)]
fn streamed_load_equals_the_whole_model_load_bit_for_bit() {
    let layers = 8;
    let config = QWEN3_5_4B.with_layers(layers);
    let options = WeightOptions {
        mlp_bfp4: true,
        ..WeightOptions::default()
    };
    let prompt = ladder_support::prompt(17, config.vocab);
    let continuation: Vec<i32> = ladder_support::prompt(9, config.vocab)[1..].to_vec();
    with_device(Config::default(), |d| {
        let whole = {
            let model = Qwen35::<TtBackend>::qwen3_5_4b(layers, Precision::F32, &d)
                .unwrap()
                .0
                .for_inference(Numerics::Performance, options);
            model.forced(&prompt, &continuation, Precision::F32, &d)
        };
        let streamed = {
            let paths = weights_files("qwen3.5-4b", &FILES);
            let (model, loaded) =
                Qwen35::<TtBackend>::streamed(&config, &paths, Numerics::Performance, options, &d)
                    .unwrap();
            eprintln!("streamed: {} parameters applied", loaded.applied);
            // Skipped by the first pass, loaded by its own: not reported as ignored.
            assert!(
                !loaded.ignored.iter().any(|n| n.contains("layers.0.")),
                "a loaded layer is listed as ignored"
            );
            model.forced(&prompt, &continuation, Precision::F32, &d)
        };
        ladder_support::assert_same(&streamed, &whole, "streamed against whole-model load");
    });
}
