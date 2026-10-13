//! How long an embedding lookup takes on a real vocabulary, on its own.
//!
//! SmolLM2's per-token cost on the card was a flat ~14 s (`step167_smollm2`) because `select`
//! staged every row of the table for every element it returned (`gather_native`: 576 x 49,152 =
//! 28 M host addresses a token, 15.9 s, 2026-10-12 run `1791845218`). `Session::select_rows`
//! (`step171_select_rows`) reads only the rows it returns. This isolates the lookup: a
//! 49,152 x 576 table through Burn's own `Embedding`, a cold first lookup (the table's first
//! materialisation on the card and the programs' first use) and then, warm, the median of seven
//! lookups of 1, 5 and 32 tokens, each validated bit for bit against the host's copy of the
//! weights. Silicon only (the simulator cannot afford it), one test per process.

#![cfg(feature = "silicon")]

use std::time::{Duration, Instant};

use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::{Int, Tensor, TensorData};
use burn_tt::TtBackend;
use tt_tests::burn_device::{with_device, Config};

const VOCAB: usize = 49_152;
const DIM: usize = 576;
const RUNS: usize = 7;

#[test]
#[ignore = "a measurement, not a gate: cargo xtask silicon --include-ignored"]
fn embedding_lookup_on_a_real_vocabulary() {
    with_device(Config::default(), |d| {
        let table: Embedding<TtBackend> = EmbeddingConfig::new(VOCAB, DIM).init(&d);
        // The host's copy of the weights, to validate every lookup against.
        let weights = table
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .expect("the table's weights");
        let lookup = |ids: &[i32]| -> (Vec<f32>, Duration) {
            let x = Tensor::<TtBackend, 2, Int>::from_data(
                TensorData::new(ids.to_vec(), [1, ids.len()]),
                &d,
            );
            let t0 = Instant::now();
            let out = table.forward(x).into_data().to_vec::<f32>().unwrap();
            (out, t0.elapsed())
        };
        let check = |ids: &[i32], got: &[f32]| {
            for (k, &id) in ids.iter().enumerate() {
                let want = &weights[id as usize * DIM..(id as usize + 1) * DIM];
                let row = &got[k * DIM..(k + 1) * DIM];
                assert!(
                    row.iter()
                        .zip(want)
                        .all(|(g, w)| g.to_bits() == w.to_bits()),
                    "token {k} (row {id}) is not the host's row"
                );
            }
        };
        let (_, cold) = lookup(&[7042]);
        eprintln!("embedding of 1 token from [{VOCAB}, {DIM}], cold (first use): {cold:.2?}");
        for tokens in [1usize, 5, 32] {
            let mut times = Vec::new();
            for run in 0..=RUNS {
                // Different rows each run, so no cache can hold the answer.
                let ids: Vec<i32> = (0..tokens as i32)
                    .map(|i| ((7042 + 13 * i + 977 * run as i32) as usize % VOCAB) as i32)
                    .collect();
                let (out, took) = lookup(&ids);
                check(&ids, &out);
                if run > 0 {
                    times.push(took); // run 0 is the warm-up
                }
            }
            times.sort();
            eprintln!(
                "embedding of {tokens} token(s) from [{VOCAB}, {DIM}], warm: median {:.2?} \
                 (min {:.2?}, max {:.2?}, {RUNS} runs, each validated against the host's rows)",
                times[RUNS / 2],
                times[0],
                times[RUNS - 1]
            );
        }
    });
}
