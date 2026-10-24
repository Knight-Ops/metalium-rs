//! What a decode step's attention as one op (`Session::attention_rows`) spends its time on, kernel
//! by kernel: decode-to-bandwidth Phase 1.
//!
//! `cargo xtask silicon --release --device 0 --include-ignored --filter silicon_probe_attention_rows`
//!
//! A SmolLM2-1.7B layer's attention (32 heads of `head_dim` 64, one query row each, keys and
//! values column blocks of position-major `[l, 2048]` caches) on every tile of the card, traced
//! (so the time is the card's): the whole op, then the op with each of its seven kernels left out
//! in turn (`attention::probe_stages`; the result is then wrong, its time is the point), and the
//! op with only its first kernel. A trace holds `REPS` copies of the op; the op's time is the
//! difference from a trace of one, over `REPS - 1`.
//!
//! Validated outside the timing: the whole op's replay equals its eager run bit for bit.
#![cfg(feature = "silicon")]

use std::time::Instant;

use tt_kernels::attention::{probe_stages, Item};
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Elem};
use tt_kernels::trace::TraceId;
use tt_tests::backend::device_index;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

type Card = Session<tt_kmd::Kmd>;

const WARMUPS: usize = 2;
const SAMPLES: usize = 9;
const REPS: usize = 8;
const KERNELS: [&str; 7] = ["qk", "scale+mask", "max", "sub_exp", "sum", "div", "pv"];

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn replay_us(s: &mut Card, trace: TraceId) -> f64 {
    let mut v: Vec<f64> = (0..WARMUPS + SAMPLES)
        .map(|_| {
            let t = Instant::now();
            s.replay(trace).unwrap();
            s.sync().unwrap();
            t.elapsed().as_secs_f64() * 1e6
        })
        .skip(WARMUPS)
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn op_us(s: &mut Card, mut op: impl FnMut(&mut Card) -> DramTensor) -> (f64, DramTensor) {
    let mut last = Vec::new();
    let mut trace = |s: &mut Card, n: usize, last: &mut Vec<DramTensor>| {
        s.begin_trace().unwrap();
        for _ in 0..n {
            last.push(op(s));
        }
        s.end_trace().unwrap()
    };
    let one = trace(s, 1, &mut last);
    let many = trace(s, REPS, &mut last);
    let (t1, tn) = (replay_us(s, one), replay_us(s, many));
    let out = last.pop().unwrap();
    for t in last {
        s.free(t).unwrap();
    }
    ((tn - t1) / (REPS - 1) as f64, out)
}

#[test]
#[ignore = "benchmark"]
fn attention_rows_kernel_by_kernel() {
    let card = device_index();
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::All)
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        let (heads, d) = (32usize, 64usize);
        for l in [64usize, 512] {
            let q = s.upload(&floats(1, heads * d), heads, d).unwrap();
            let k = s.upload(&floats(2, l * heads * d), l, heads * d).unwrap();
            let v = s.upload(&floats(3, l * heads * d), l, heads * d).unwrap();
            let later: Vec<u32> = (0..l).map(|p| u32::from(p + 1 == l)).collect();
            let mask = s.upload_bits(&later, 1, l, Elem::Bool).unwrap();
            let items: Vec<Item> = (0..heads)
                .map(|h| Item {
                    q: [h, 0],
                    k: [0, h * d],
                    v: [0, h * d],
                })
                .collect();
            let mut op = |s: &mut Card| {
                s.attention_rows(
                    &q,
                    &k,
                    &v,
                    &mask,
                    &items,
                    [1, d, l, d],
                    0.125,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    BUDGET,
                )
                .unwrap()
            };
            let eager = op(&mut s);
            let want = s.download(&eager).unwrap();
            s.free(eager).unwrap();
            let (whole, out) = op_us(&mut s, &mut op);
            let got = s.download(&out).unwrap();
            s.free(out).unwrap();
            assert_eq!(
                got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "l {l}: the replay is the eager op"
            );
            println!("MEASURE attention-rows l {l}: whole op {whole:.1} us");
            for (n, name) in KERNELS.iter().enumerate().skip(1) {
                probe_stages(!(1 << n));
                let (us, out) = op_us(&mut s, &mut op);
                s.free(out).unwrap();
                println!(
                    "MEASURE attention-rows l {l}: without {name} {us:.1} us ({:.1} us its own)",
                    whole - us
                );
            }
            probe_stages(1);
            let (us, out) = op_us(&mut s, &mut op);
            s.free(out).unwrap();
            println!("MEASURE attention-rows l {l}: reads, qk and writes only {us:.1} us");
            // Which kernel before `SUB_EXP` or `DIV` makes it slow, if any.
            for set in [
                &[3usize][..],
                &[5],
                &[1, 5],
                &[4, 5],
                &[2, 3],
                &[1, 3],
                &[3, 5],
            ] {
                probe_stages(set.iter().map(|&n| 1 << n).sum());
                let (with, out) = op_us(&mut s, &mut op);
                s.free(out).unwrap();
                let names: Vec<_> = set.iter().map(|&n| KERNELS[n]).collect();
                println!(
                    "MEASURE attention-rows l {l}: qk, {names:?}: {with:.1} us ({:.1} over qk alone)",
                    with - us
                );
            }
            probe_stages(u32::MAX);
            // The same kinds as ordinary element-wise ops on one unit: `[1, l]` by a `[1, 1]`
            // column, row-limited, as RMSNorm's `DIV` runs.
            s.set_spread(tt_kernels::session::Spread::Fixed(1));
            let row = s.upload(&floats(4, l), 1, l).unwrap();
            let col = s.upload(&[3.0], 1, 1).unwrap();
            for (name, kind) in [
                ("mul", tt_kernels::kind::MUL),
                ("div", tt_kernels::sfpu::ops::kind_sfpu::DIV),
                ("sub_exp", tt_kernels::sfpu::ops::kind_sfpu::SUB_EXP),
            ] {
                let e = tt_kernels::tensor::Eltwise {
                    kind,
                    scalar: 0.0,
                    scalar2: 0.0,
                };
                let (us, out) = op_us(&mut s, |s| s.eltwise(e, &row, Some(&col)).unwrap());
                s.free(out).unwrap();
                println!(
                    "MEASURE attention-rows l {l}: element-wise {name} [1, {l}] by [1, 1] on one unit {us:.1} us"
                );
            }
            // `DIV` by special divisors, the column's other rows as the upload leaves them.
            for (name, d) in [
                ("0", 0.0f32),
                ("denormal", f32::from_bits(0x0000_1234)),
                ("nan", f32::NAN),
                ("inf", f32::INFINITY),
                ("2^-140 normalish", 1e-38),
            ] {
                let dcol = s.upload(&[d], 1, 1).unwrap();
                let e = tt_kernels::tensor::Eltwise {
                    kind: tt_kernels::sfpu::ops::kind_sfpu::DIV,
                    scalar: 0.0,
                    scalar2: 0.0,
                };
                let (us, out) = op_us(&mut s, |s| s.eltwise(e, &row, Some(&dcol)).unwrap());
                s.free(out).unwrap();
                s.free(dcol).unwrap();
                println!(
                    "MEASURE attention-rows l {l}: element-wise div [1, {l}] by {name} on one unit {us:.1} us"
                );
            }
            s.set_spread(tt_kernels::session::Spread::All);
            s.free(row).unwrap();
            s.free(col).unwrap();
            for t in [q, k, v, mask] {
                s.free(t).unwrap();
            }
        }
    }) {
        panic!("{e}");
    }
}

/// A decode row's RMSNorm as one op (`Session::rms_norm`) against the statistic op and its `DIV`,
/// at `K` 2048 on 64 units, traced: the time a norm takes each way.
#[test]
#[ignore = "benchmark"]
fn rms_norm_one_op_against_two() {
    let card = device_index();
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::All)
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_spread(tt_kernels::session::Spread::Fixed(64));
        let k = 2048;
        let raw = s.upload(&floats(5, k), 1, k).unwrap();
        let one = tt_kernels::tensor::Eltwise {
            kind: tt_kernels::kind::MUL_SCALAR,
            scalar: 1.0,
            scalar2: 0.0,
        };
        let x = s.eltwise(one, &raw, None).unwrap();
        for fused in [true, false] {
            s.set_rms_divide(fused);
            for units in [8usize, 16, 32, 64] {
                s.set_spread(tt_kernels::session::Spread::Fixed(units));
                let (us, out) = op_us(&mut s, |s| s.rms_norm(&x, 1e-5).unwrap());
                s.free(out).unwrap();
                println!("MEASURE rms-norm K {k} on {units} units, one op {fused}: {us:.1} us");
            }
        }
        s.set_rms_divide(true);
        s.set_spread(tt_kernels::session::Spread::Fixed(64));
        let (us, out) = op_us(&mut s, |s| s.rms(&x, 1e-5).unwrap());
        s.free(out).unwrap();
        println!("MEASURE rms-norm K {k}: the statistic op alone {us:.1} us");
        let (us, out) = op_us(&mut s, |s| s.eltwise(one, &x, None).unwrap());
        s.free(out).unwrap();
        println!("MEASURE rms-norm K {k}: an element-wise op on 64 units {us:.1} us");
    }) {
        panic!("{e}");
    }
}

/// Flash-decode's probe (P0, no order change): the reads and kernels of a head's positions split
/// in two, each half its own item, so `2 x heads` units each attend `l / 2` positions. The result
/// is not attention (each half normalizes alone); its time is the point. Against today's op, one
/// head a unit over all `l`, and the same halves on only `heads` units (a unit's rate alone).
/// Isolated: nothing else reads GDDR beside it, unlike a replayed step.
#[test]
#[ignore = "benchmark"]
fn attention_rows_split_in_half() {
    let card = device_index();
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::All)
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        let (heads, d) = (32usize, 64usize);
        for l in [128usize, 256, 512, 1024, 2048] {
            let q = s.upload(&floats(1, heads * d), heads, d).unwrap();
            let k = s.upload(&floats(2, l * heads * d), l, heads * d).unwrap();
            let v = s.upload(&floats(3, l * heads * d), l, heads * d).unwrap();
            let time = |s: &mut Card, items: &[Item], len: usize, units: usize| {
                let later: Vec<u32> = (0..len).map(|p| u32::from(p + 1 == len)).collect();
                let mask = s.upload_bits(&later, 1, len, Elem::Bool).unwrap();
                s.set_spread(tt_kernels::session::Spread::Fixed(units));
                let fits = tt_kernels::attention::fits([1, d, len, d], 0);
                let us = fits.then(|| {
                    let (us, out) = op_us(s, |s| {
                        s.attention_rows(
                            &q,
                            &k,
                            &v,
                            &mask,
                            items,
                            [1, d, len, d],
                            0.125,
                            SrcRoute::Tf32FromFp32,
                            Fidelity::HiFi4,
                            BUDGET,
                        )
                        .unwrap()
                    });
                    s.free(out).unwrap();
                    us
                });
                s.free(mask).unwrap();
                s.set_spread(tt_kernels::session::Spread::All);
                us
            };
            let whole: Vec<Item> = (0..heads)
                .map(|h| Item {
                    q: [h, 0],
                    k: [0, h * d],
                    v: [0, h * d],
                })
                .collect();
            let halves: Vec<Item> = (0..2 * heads)
                .map(|j| {
                    let (h, r) = (j % heads, (j / heads) * (l / 2));
                    Item {
                        q: [h, 0],
                        k: [r, h * d],
                        v: [r, h * d],
                    }
                })
                .collect();
            let today = time(&mut s, &whole, l, heads);
            let split = time(&mut s, &halves, l / 2, 2 * heads);
            let one_rate = time(&mut s, &halves[..heads], l / 2, heads);
            let mb = (2 * l * heads * d * 4) as f64 / 1e6;
            let show = |t: Option<f64>| t.map_or("does not fit".into(), |t| format!("{t:.1} us"));
            println!(
                "MEASURE attention-split l {l} ({mb:.2} MB): today {} on {heads} units, \
                 halves {} on {} units, one half a head on {heads} units {}",
                show(today),
                show(split),
                2 * heads,
                show(one_rate)
            );
            for t in [q, k, v] {
                s.free(t).unwrap();
            }
        }
    }) {
        panic!("{e}");
    }
}
