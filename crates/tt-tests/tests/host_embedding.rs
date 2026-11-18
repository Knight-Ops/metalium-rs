//! A Qwen3.5 whose embedding table stays in host memory (`Qwen35::with_host_embedding`,
//! `tt_models::host_embed`) generates on the card as the same model with its table in GDDR does,
//! bit for bit: the traced steps take each token's row as their input instead of its id
//! (`FixedShapeDecoder::step_rows`, `chunk_rows`), and the rows are the device lookup's.
//!
//! The model is Qwen3.5-4B's shape at four layers (three DeltaNets and a full-attention layer)
//! with an untied head and a vocabulary of 4,096, its weights drawn, under the performance
//! contract. No checkpoint: the table is written to a file from the device model's own values.
//! A model workload: silicon only.

mod ladder_support;

use burn_tt::{Numerics, TtBackend};
use tt_models::generate::Shapes;
use tt_models::host_embed::{write_table, HostEmbedding};
use tt_models::quant::WeightOptions;
use tt_models::qwen35::{is_embed_table, Qwen35, Qwen35Config, QWEN3_5_4B};
use tt_tests::burn_device::{with_device, Config};

const CONFIG: Qwen35Config = Qwen35Config {
    vocab: 4096,
    layers: 4,
    tied_head: false,
    ..QWEN3_5_4B
};

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask silicon --release --filter host_embedding"
)]
fn a_host_table_generates_what_the_device_table_does() {
    // Prompts that pad a chunk (5), fill one (16), and run three chunks into a second step length.
    let prompts = [5, 16, 70].map(|n| ladder_support::prompt(n, CONFIG.vocab));
    let shapes = Shapes::new(128, &[8, 16]).with_prefixes(&[]);
    let steps = 6;
    with_device(Config::default(), |d| {
        // Every parameter is made by `for_inference`'s pass, so the clone below shares them.
        let device_table = Qwen35::<TtBackend>::new(&CONFIG, &d)
            .for_inference(Numerics::Performance, WeightOptions::default());
        let values = device_table
            .embed_table()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let dir = std::env::temp_dir().join(format!("host-embedding-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("embed.safetensors");
        write_table(
            &path,
            "embed_tokens.weight",
            &values,
            CONFIG.vocab,
            CONFIG.hidden,
        )
        .unwrap();
        let host_table = device_table
            .clone()
            .with_host_embedding(HostEmbedding::from_shards(&[path], &is_embed_table).unwrap())
            .unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        let want: Vec<_> = {
            let g = ladder_support::traced(&device_table, &shapes, &d);
            prompts
                .iter()
                .map(|p| g.generate(p, steps).unwrap())
                .collect()
        };
        let g = ladder_support::traced(&host_table, &shapes, &d);
        for (p, want) in prompts.iter().zip(&want) {
            let got = g.generate(p, steps).unwrap();
            eprintln!("prompt {}: tokens {:?}", p.len(), got.tokens);
            ladder_support::assert_same(&got, want, &format!("prompt {}", p.len()));
        }
    });
}
