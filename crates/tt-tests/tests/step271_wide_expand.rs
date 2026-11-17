//! Roadmap W1, defect 5: a broadcast of a trailing axis of one over more rows than a tensor
//! reference names column tiles (`tt_isa::dm::record::MAX_EXTENT`, 65,536 tiles: 2,097,152 rows).
//!
//! `expand` repeats such an axis through `[1, rows]`: past 2,097,152 rows the copy was refused
//! ("descriptor refused before submission (code 2)", seen at the next download), the nearest
//! upsampling of MOSS-Image's VAE from `[1, 384, 64, 64]` on (`new-architectures.md`). It now goes
//! in chunks, joined by one row gather. What is claimed: the expand runs on the card and every
//! element is its source's, on both sides of the limit. Silicon only: 3.1 million rows are a
//! model's tensor, not a simulator's.

#![cfg(feature = "silicon")]

use burn::tensor::{Tensor, TensorData};
use burn_tt::TtBackend;
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// `[1, c, h, 1, w, 1] -> [1, c, h, 2, w, 2]`, a nearest upsampling by two written as one
/// broadcast, checked element by element.
fn upsample(c: usize, h: usize, w: usize) {
    with_device(Config::default(), |d| {
        // Distinct, exact in F32 (below 2^24).
        let values: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let x = Tensor::<TtBackend, 6>::from_data(
            TensorData::new(values.clone(), [1, c, h, 1, w, 1]),
            &d,
        );
        let (y, report) = burn_tt::with_report(|| {
            let y = x.expand([1, c, h, 2, w, 2]);
            let _ = y
                .clone()
                .slice([0..1, 0..1, 0..1, 0..1, 0..1, 0..1])
                .into_data();
            y
        });
        assert_native_model(&report);
        let got: Vec<f32> = y.into_data().to_vec().unwrap();
        assert_eq!(got.len(), c * h * 2 * w * 2);
        let mut at = 0;
        for ch in 0..c {
            for r in 0..h {
                for _ in 0..2 {
                    for col in 0..w {
                        let want = values[(ch * h + r) * w + col];
                        assert_eq!(
                            [got[at], got[at + 1]],
                            [want; 2],
                            "[{ch}, {r}, _, {col}, _] of a [1, {c}, {h}, 2, {w}, 2] expand"
                        );
                        at += 2;
                    }
                }
            }
        }
    });
}

/// 786,432 rows: one chunk, as before.
#[test]
fn a_trailing_axis_of_one_expands_inside_the_extent() {
    upsample(384, 32, 32);
}

/// 3,145,728 rows (98,304 column tiles wide as one `[1, rows]`): refused before the chunks.
#[test]
fn a_trailing_axis_of_one_expands_past_the_extent() {
    upsample(384, 64, 64);
}
