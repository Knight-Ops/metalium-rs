//! Where attention's device jobs go, and how few there are (`docs/completed-plans/attention-jobs.md`).
//!
//! `module::attention` charges each stage's jobs to a part of its own (`report::part`):
//! `attention/qk`, `/scale`, `/mask`, `/softmax`, `/pv`, `/widen` and `/narrow`. This gate runs
//! the shapes the model ladder runs -- a SmolLM2 decode step (nine heads of one query over a cache
//! stacked by rows), ViT-Tiny (three heads of 197 tokens, read in place from one projection) and a
//! causal prefill -- and holds each product to **one job** (P1, P2; it was `3 heads + 1`: 28 of a
//! decode call's 64 jobs each). Run with `--nocapture` to see the table.
//!
//! Each call's result is bit for bit the same attention run one head at a time: every product is
//! `matmul_dram`'s on its head's blocks either way (`step184`), and the softmax is per row. Where
//! padding costs at most a quarter more rows (ViT's 197 tokens, not a decode's one query) the
//! scores stay a head per whole tile rows from `Q K^T` to `P V` (`padded_attention`); a single
//! head never pads, so the reference took the stacked layout's arithmetic, row for row.
//!
//! Watched failing (2026-10-15, ttsim): a causal mask shifted one column (`step83`'s analytic
//! cases); the kept mask's blocks after the first left unmasked; `P V` reading head `i` from row
//! `i m` of the padded scores; the padded causal mask laid out at the stacked stride.

use burn::tensor::{module::attention, Tensor, TensorData};
use burn_tensor::ops::AttentionModuleOptions;
use burn_tt::TtBackend;
use tt_tests::burn_device::{assert_native_model, with_device, Config};

const PARTS: [&str; 7] = [
    "attention/widen",
    "attention/qk",
    "attention/scale",
    "attention/mask",
    "attention/softmax",
    "attention/pv",
    "attention/narrow",
];

/// Small integers, so every product is exact.
fn ints(seed: u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as u64 * seed + 3) % 9) as f32 - 4.0)
        .collect()
}

fn tensor<const D: usize>(
    d: &burn_tt::TtDevice,
    shape: [usize; D],
    seed: u64,
) -> Tensor<TtBackend, D> {
    let x = Tensor::<TtBackend, D>::from_data(
        TensorData::new(ints(seed, shape.iter().product()), shape),
        d,
    );
    // Resident before the count starts.
    let _ = (x.clone() * 1.0).sum().into_data();
    x
}

/// Jobs per part of one attention call (and the total charged to `attention` itself, which the
/// parts leave at zero), after checking the call against the same attention a head at a time.
fn jobs(
    what: &str,
    q: Tensor<TtBackend, 4>,
    k: Tensor<TtBackend, 4>,
    v: Tensor<TtBackend, 4>,
    causal: bool,
) -> Vec<(&'static str, u64)> {
    let options = || AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: causal,
    };
    let heads = q.dims()[1];
    let one = |t: &Tensor<TtBackend, 4>, h: usize| {
        let [b, _, s, d] = t.dims();
        t.clone().slice([0..b, h..h + 1, 0..s, 0..d])
    };
    let want: Vec<u32> = (0..heads)
        .flat_map(|h| {
            attention(one(&q, h), one(&k, h), one(&v, h), None, None, options())
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        })
        .map(f32::to_bits)
        .collect();
    let bits = |out: Tensor<TtBackend, 4>| -> Vec<u32> {
        out.into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    };
    // The first call makes what an attachment keeps (a causal mask); the second is counted.
    let first = attention(q.clone(), k.clone(), v.clone(), None, None, options());
    assert_eq!(bits(first), want, "{what}: against one head at a time");
    let (out, report) = burn_tt::with_report(|| attention(q, k, v, None, None, options()));
    assert_native_model(&report);
    assert_eq!(bits(out), want, "{what}: a second call");
    let mut rows: Vec<_> = PARTS
        .iter()
        .map(|&p| (p, report.op(p).map_or(0, |s| s.jobs)))
        .collect();
    rows.push(("attention", report.op("attention").map_or(0, |s| s.jobs)));
    let total: u64 = rows.iter().map(|(_, n)| n).sum();
    println!("{what}: {total} jobs");
    for (p, n) in &rows {
        println!("  {p:<20} {n}");
    }
    let of = |part: &str| rows.iter().find(|(p, _)| *p == part).unwrap().1;
    assert_eq!(of("attention"), 0, "{what}: every job is charged to a part");
    assert_eq!(of("attention/qk"), 1, "{what}: q k^T is one job");
    assert_eq!(of("attention/pv"), 1, "{what}: p v is one job");
    assert_eq!(
        of("attention/mask"),
        u64::from(causal),
        "{what}: a kept causal mask is one mask_fill"
    );
    rows
}

#[test]
fn attention_jobs_by_part() {
    with_device(Config::default(), |d| {
        // SmolLM2 decode: 9 heads, one query, a cache of 37 positions (a plain [9 * 37, 64]
        // after `repeat_kv`'s cat).
        let q = tensor(&d, [1, 9, 1, 64], 3);
        let k = tensor(&d, [1, 9, 37, 64], 5);
        let v = tensor(&d, [1, 9, 37, 64], 7);
        jobs("decode [1,9,1,64] over 37", q, k, v, false);

        // ViT-Tiny: 3 heads of 197 tokens, each a column block of one [197, 576] projection.
        let qkv = tensor(&d, [1, 197, 576], 11);
        let heads = |i: usize| {
            qkv.clone()
                .slice([0..1, 0..197, i * 192..(i + 1) * 192])
                .reshape([1, 197, 3, 64])
                .swap_dims(1, 2)
        };
        jobs("vit [1,3,197,64]", heads(0), heads(1), heads(2), false);
        // The same heads, causal: the scores stay a head per whole tile rows (`padded_attention`),
        // so the kept mask is laid out the same way.
        jobs(
            "vit-shaped causal [1,3,197,64]",
            heads(0),
            heads(1),
            heads(2),
            true,
        );

        // SmolLM2 prefill: a causal prompt of five tokens.
        let q = tensor(&d, [1, 9, 5, 64], 13);
        let k = tensor(&d, [1, 9, 5, 64], 17);
        let v = tensor(&d, [1, 9, 5, 64], 19);
        jobs("prefill [1,9,5,64] causal", q, k, v, true);
    });
}
