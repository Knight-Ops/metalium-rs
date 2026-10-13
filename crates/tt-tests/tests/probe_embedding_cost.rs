//! How long an embedding lookup takes on a real vocabulary, on its own.
//!
//! SmolLM2's per-token cost on the card is a flat ~14 s (`step167_smollm2`), and `gather_native`
//! stages `32 x rows` host addresses per chunk of outputs. This isolates the lookup: a 49,152 x 576
//! table, one token and five tokens, through Burn's own `Embedding`. Silicon only (the simulator
//! cannot afford it), one test per process.

#![cfg(feature = "silicon")]

use std::time::Instant;

use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Int, Tensor, TensorData};
use burn_tt::TtBackend;
use tt_tests::burn_device::{with_device, Config};

const VOCAB: usize = 49_152;
const DIM: usize = 576;

#[test]
#[ignore = "a measurement, not a gate: cargo xtask silicon --include-ignored"]
fn embedding_lookup_on_a_real_vocabulary() {
    with_device(Config::default(), |d| {
        let table: Embedding<TtBackend> = EmbeddingConfig::new(VOCAB, DIM).init(&d);
        for tokens in [1usize, 5] {
            let ids: Vec<i32> = (0..tokens as i32).map(|i| 7042 + 13 * i).collect();
            let x = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(ids, [1, tokens]), &d);
            let t0 = Instant::now();
            let out = table.forward(x);
            let _ = out.into_data();
            eprintln!(
                "embedding of {tokens} token(s) from [{VOCAB}, {DIM}]: {:.2?}",
                t0.elapsed()
            );
        }
    });
}
