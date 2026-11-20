//! Attention a block of keys at a time (`burn_tt::set_attention_block`): the scores of one block
//! held at once, the blocks combined by an online softmax, for a prefill chunk at long context,
//! whose row of scores over every position otherwise takes `heads t len` F32 twice in GDDR.
//!
//! The claims, for F32 keys and values and a BFP8 cache read where it lies:
//! - **as close to the exact attention as the row's softmax:** against an f64 attention of the
//!   values the card holds, at most twice the row softmax's error (both round the products'
//!   operands through TF32, about 5e-4 of a value; the blocks round other values, each `exp`
//!   taken twice and the division moved after `P V`), for blocks that divide the length and one
//!   that leaves a ragged tail, and for rows whose first blocks are all masked (a prefill chunk's
//!   early rows at a long cache);
//! - **deterministic:** the same bits twice;
//! - **negative control:** its bits are not the row softmax's, so the blocked path ran;
//! - **a model:** Qwen3.5's shape at four layers through the fusion backend, a BF16 and a BFP8
//!   cache, chunks of 32 into a cache of 128 attended in blocks of 32: traced equal to eager bit
//!   for bit, and the tokens the unblocked model's.
//!
//! Silicon only.

mod ladder_support;

use burn::tensor::{module::attention, Bool, Tensor, TensorData};
use burn_tensor::ops::AttentionModuleOptions;
use burn_tt::storage::{StorageFormat, TensorStorageExt};
use burn_tt::TtBackend;
use tt_tests::burn_device::{with_device, Config};

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

/// A prefill chunk's mask at the cache's end: row `r` sees positions up to `len - rows + r`; rows
/// past 32 also lose the first 160 positions (blocks masked whole for them).
fn mask_of(r: usize, p: usize, rows: usize, len: usize) -> bool {
    p > len - rows + r || (r >= 32 && p < 160)
}

fn floats(t: Tensor<TtBackend, 4>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "silicon tier: cargo xtask silicon --release --filter step279_blocked_attention"
)]
fn attention_in_blocks_is_the_row_softmax_within_rounding() {
    let config = Config {
        tiles: Some(burn_tt::TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        let (kv, rows, len, dim) = (2usize, 64usize, 480usize, 128usize);
        let q = Tensor::<TtBackend, 4>::from_data(
            TensorData::new(values(1, kv * rows * dim), [1, kv, rows, dim]),
            &d,
        );
        let k = Tensor::<TtBackend, 4>::from_data(
            TensorData::new(values(2, kv * len * dim), [1, kv, len, dim]),
            &d,
        );
        let v = Tensor::<TtBackend, 4>::from_data(
            TensorData::new(values(3, kv * len * dim), [1, kv, len, dim]),
            &d,
        );
        // A prefill chunk's mask at the cache's end: row `r` sees positions up to
        // `len - rows + r`; rows past 32 also lose the first 160 positions (blocks masked whole).
        let mask: Vec<bool> = (0..rows)
            .flat_map(|r| (0..len).map(move |p| mask_of(r, p, rows, len)))
            .collect();
        let mask =
            Tensor::<TtBackend, 4, Bool>::from_data(TensorData::new(mask, [1, 1, rows, len]), &d);
        let options = || AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: false,
        };
        for (what, k, v) in [
            ("F32", k.clone(), v.clone()),
            (
                "BFP8",
                k.clone().with_row_storage(StorageFormat::Bfp8),
                v.clone().with_row_storage(StorageFormat::Bfp8),
            ),
        ] {
            let run = |block: Option<usize>| {
                burn_tt::set_attention_block(block);
                let out = floats(attention(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    Some(mask.clone()),
                    None,
                    options(),
                ));
                burn_tt::set_attention_block(None);
                out
            };
            let rows_softmax = run(None);
            // The exact attention (f64 on the host) of the values the card holds: the keys and
            // values as they decode, so a BFP8 cache's rounding is in both.
            let host = |t: Tensor<TtBackend, 4>| -> Vec<f64> {
                t.into_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(f64::from)
                    .collect()
            };
            let (hq, hk, hv) = (host(q.clone()), host(k.clone()), host(v.clone()));
            let scale = 1.0 / (dim as f64).sqrt();
            let mut exact = vec![0f64; kv * rows * dim];
            for h in 0..kv {
                for r in 0..rows {
                    let qr = &hq[(h * rows + r) * dim..][..dim];
                    let seen: Vec<usize> =
                        (0..len).filter(|&p| !mask_of(r, p, rows, len)).collect();
                    let s: Vec<f64> = seen
                        .iter()
                        .map(|&p| {
                            let kp = &hk[(h * len + p) * dim..][..dim];
                            qr.iter().zip(kp).map(|(a, b)| a * b).sum::<f64>() * scale
                        })
                        .collect();
                    let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                    let sum: f64 = e.iter().sum();
                    for (w, &p) in e.iter().zip(&seen) {
                        let vp = &hv[(h * len + p) * dim..][..dim];
                        for c in 0..dim {
                            exact[(h * rows + r) * dim + c] += w / sum * vp[c];
                        }
                    }
                }
            }
            let off = |got: &[f32]| {
                got.iter()
                    .zip(&exact)
                    .map(|(a, b)| (*a as f64 - b).abs())
                    .fold(0f64, f64::max)
            };
            let rows_off = off(&rows_softmax);
            let largest = exact.iter().fold(0f64, |m, x| m.max(x.abs()));
            for block in [96, 160, 128] {
                let got = run(Some(block));
                assert_eq!(got.len(), rows_softmax.len());
                let blocks_off = off(&got);
                // The products round through TF32 in both (about 5e-4 of a value); the blocks'
                // rounding falls elsewhere, not further.
                assert!(
                    blocks_off <= 2.0 * rows_off + 1e-6 && blocks_off <= 4e-3 * largest,
                    "{what}, blocks of {block}: off by {blocks_off:e} from the exact attention, \
                     the row softmax by {rows_off:e} (largest {largest})"
                );
                let again = run(Some(block));
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    bits(&got),
                    bits(&again),
                    "{what}, blocks of {block}: deterministic"
                );
                assert_ne!(
                    bits(&got),
                    bits(&rows_softmax),
                    "{what}, blocks of {block}: the blocked path ran"
                );
                eprintln!(
                    "{what}, blocks of {block}: off the exact attention by {blocks_off:e}, the row \
                     softmax by {rows_off:e} (largest {largest:.3})"
                );
            }
        }
    });
}

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask silicon --release --filter step279_blocked_attention"
)]
fn a_model_attending_in_blocks_generates_traced_as_eager() {
    use burn_tt::storage::BfpFormat;
    use tt_models::generate::{generate_fixed, Shapes, TracedGenerator};
    use tt_models::quant::WeightOptions;
    use tt_models::qwen35::{Qwen35, Qwen35Config, QWEN3_5_4B};
    const CONFIG: Qwen35Config = Qwen35Config {
        vocab: 4096,
        layers: 4,
        ..QWEN3_5_4B
    };
    let prompt = ladder_support::prompt(70, CONFIG.vocab);
    let shapes = Shapes::new(128, &[32]).with_prefixes(&[]);
    let config = Config {
        tiles: Some(burn_tt::TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        let model = Qwen35::<burn_tt::Tt>::new(&CONFIG, &d);
        for kv in [None, Some(BfpFormat::Bfp8)] {
            let m = model.clone().for_inference(
                burn_tt::Numerics::Performance,
                WeightOptions {
                    kv_block_float: kv,
                    ..WeightOptions::default()
                },
            );
            let plain = generate_fixed(&m, &prompt, 4, &shapes, &d).unwrap();
            burn_tt::set_attention_block(Some(32));
            let g = TracedGenerator::<_, burn_tt::Tt>::new(&m, &shapes, &d).unwrap();
            let got = g.generate(&prompt, 4).unwrap();
            let eager = generate_fixed(&m, &prompt, 4, &shapes, &d).unwrap();
            burn_tt::set_attention_block(None);
            assert_eq!(got.tokens, eager.tokens, "{kv:?}: tokens");
            for (a, b) in got.logits.iter().zip(&eager.logits) {
                assert_eq!(
                    ladder_support::bits(a),
                    ladder_support::bits(b),
                    "{kv:?}: logits"
                );
            }
            assert_eq!(
                got.tokens, plain.tokens,
                "{kv:?}: the unblocked model's tokens"
            );
            eprintln!(
                "{kv:?}: blocks of 32, traced = eager, tokens {:?} (unblocked {:?})",
                got.tokens, plain.tokens
            );
        }
    });
}
