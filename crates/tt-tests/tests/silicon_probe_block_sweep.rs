//! The time of every block on a coarse grid, with and without `A` pulled, for a blocked prefill
//! product and each `B` format (`docs/completed-plans/dataflow-compiler.md` F4b.2, H9): the landscape a
//! measured block choice (`tensor::BlockStream::Measured`) picks from, so its candidates
//! (`tensor::block_stream_candidates`) can be judged against the best there is, not against the
//! model that proposes them.
//!
//! Each block is timed as `Measured` times a candidate (`Session::time_product`: traces of one and
//! of two copies of the product, each replayed twice to warm and the median of 5 taken, the
//! difference the product's time), on every tile, at HiFi4. The sweeps recorded in the log
//! (`docs/learnings/performance-log.md`, F4b.2) were taken before the difference, a trace of the
//! product alone: their ranks hold, their rates are lower. The grid's blocks are
//! those the product can run as here (`tensor::block_stream_fits`). Each product's `Auto` block and
//! the fastest found are run again and checked against exact integer arithmetic on the host.
//!
//! `TT_PAIR_SHAPES` (`m,k,n;...`, default the four F4b shapes) and `TT_SWEEP_FORMATS` (default
//! every format) choose what a run sweeps; each block is captured twice, mostly host time (one
//! capture a block took ~0.6 s, a shape and its five formats ~40 minutes). Every line is in the run's output file as it
//! is measured (`target/silicon/out/`), a run cut short by its limit included. Silicon only, a probe: `cargo xtask bench --device 1 --filter
//! silicon_probe_block_sweep`. Lines: `SWEEP <shape> B <format> <block> <reuse>: <us> us, <TFLOPS>`.

#![cfg(feature = "silicon")]

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{self, BlockStream, DramTensor, Reuse};
use tt_kernels::tile_op::reuse::AUTO_HOLDERS;
use tt_tests::harness::BUDGET;

fn shapes() -> Vec<[usize; 3]> {
    std::env::var("TT_PAIR_SHAPES").map_or(
        vec![
            [2048, 2048, 16384],
            [1024, 2048, 8192],
            [1024, 8192, 2048],
            [2048, 8192, 2048],
        ],
        |v| {
            v.split(';')
                .map(|s| {
                    let d: Vec<usize> = s.split(',').map(|x| x.trim().parse().unwrap()).collect();
                    [d[0], d[1], d[2]]
                })
                .collect()
        },
    )
}

/// Integers whose products and sums are exact: the products may be checked against each other.
fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + seed) % 5) as f32 - 2.0).collect()
}

/// The grid. At H1's shape every format's fastest blocks streamed `K` 4 to 8 tiles a chunk with
/// sides of 5 to 12, and no block with `kq = 1` or a side of 1 or 2 came within 30% of them
/// (`1793285475`); the grid keeps a margin around that.
const MBS: [usize; 10] = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16];
const KQS: [usize; 7] = [2, 3, 4, 6, 8, 12, 16];
const NBS: [usize; 13] = [3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24];

/// `TT_SWEEP_FORMATS` (`fp32,bf16,bfp8,bfp4,bfp2`, any of them; default all): the `B` formats a
/// run sweeps, so one format a run stays inside a bench's time limit.
fn formats() -> Vec<String> {
    std::env::var("TT_SWEEP_FORMATS").map_or(
        ["FP32", "BF16", "BFP8", "BFP4", "BFP2"]
            .map(String::from)
            .to_vec(),
        |v| v.split(',').map(|f| f.trim().to_uppercase()).collect(),
    )
}

/// `a @ b` as `block` with `reuse`, checked against exact arithmetic at a lattice of entries.
#[allow(clippy::too_many_arguments)]
fn check(
    s: &mut Session<tt_kmd::Kmd>,
    (a, av): (&DramTensor, &[f32]),
    (b, bv): (&DramTensor, &[f32]),
    [m, k, n]: [usize; 3],
    block: BlockStream,
    reuse: Reuse,
    what: &str,
) {
    s.set_block_stream(block);
    s.set_operand_reuse(reuse).unwrap();
    let c = s
        .matmul_dram(
            a,
            false,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET * 400,
        )
        .unwrap();
    let out = s.download(&c).unwrap();
    s.free(c).unwrap();
    for i in (0..m).step_by(97).chain([m - 1]) {
        for j in (0..n).step_by(101).chain([n - 1]) {
            let exact: f32 = (0..k).map(|q| av[i * k + q] * bv[q * n + j]).sum();
            assert_eq!(out[i * n + j], exact, "{what}: ({i}, {j})");
        }
    }
    s.set_block_stream(BlockStream::Auto);
    s.set_operand_reuse(Reuse::Off).unwrap();
}

#[test]
#[ignore = "a probe: cargo xtask bench --device 1 --filter silicon_probe_block_sweep"]
fn block_sweep() {
    tt_ttsim::fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::All,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_batching(true).unwrap();
        let pull = Reuse::Pull {
            holders: AUTO_HOLDERS,
            b_holders: None,
        };
        for [m, k, n] in shapes() {
            let (av, bv) = (values(m * k, 1), values(k * n, 3));
            let a = s.upload(&av, m, k).unwrap();
            let b = s.upload(&bv, k, n).unwrap();
            let bb = s.store_bf16(&b).unwrap();
            let mut weights: Vec<(&str, DramTensor, Vec<f32>)> = Vec::new();
            for (name, f) in [
                ("BFP8", tt_kernels::bfp::BfpFormat::Bfp8),
                ("BFP4", tt_kernels::bfp::BfpFormat::Bfp4),
                ("BFP2", tt_kernels::bfp::BfpFormat::Bfp2),
            ] {
                let p = s.bfp_from_f32(&b, f).unwrap();
                let decoded = s.bfp_to_f32(&p).unwrap();
                let values = s.download(&decoded).unwrap();
                s.free(decoded).unwrap();
                weights.push((name, DramTensor::from_bfp(p), values));
            }
            let flops = 2.0 * (m * k * n) as f64;
            let shape = format!("[{m},{k}]@[{k},{n}]");
            let mut cases: Vec<(&str, &DramTensor, &[f32])> = vec![("FP32", &b, &bv), ("BF16", &bb, &bv)];
            cases.extend(weights.iter().map(|(name, t, v)| (*name, t, v.as_slice())));
            let wanted = formats();
            for (format, bt, bvals) in cases {
                if !wanted.iter().any(|w| w == format) {
                    continue;
                }
                let q = s
                    .block_question_of((&a, false), (bt, false), Fidelity::HiFi4)
                    .unwrap()
                    .expect("a blocked product");
                let candidates: Vec<[usize; 3]> = tensor::block_stream_candidates(&q, None)
                    .into_iter()
                    .map(|c| c.block)
                    .collect();
                if candidates.is_empty() {
                    // `Auto` runs the other paths here, so `Measured` would measure nothing.
                    println!("MEASURE sweep {shape} B {format}: Auto does not block it");
                    continue;
                }
                let mut grid: Vec<[usize; 3]> = candidates.clone();
                for mb in MBS {
                    for kq in KQS {
                        for nb in NBS {
                            let block = [mb, kq, nb];
                            if !grid.contains(&block) && tensor::block_stream_fits(&q, block) {
                                grid.push(block);
                            }
                        }
                    }
                }
                println!(
                    "MEASURE sweep {shape} B {format}: {} blocks, candidates {candidates:?}, {q:?}",
                    grid.len()
                );
                let mut best: Option<(f64, [usize; 3], Reuse)> = None;
                let mut auto_us = None;
                for &block in &grid {
                    for reuse in [Reuse::Off, pull] {
                        let us = s
                            .time_product(
                                (&a, false),
                                (bt, false),
                                SrcRoute::Tf32FromFp32,
                                Fidelity::HiFi4,
                                BUDGET * 400,
                                BlockStream::Fixed(block),
                                reuse,
                                1,
                            )
                            .unwrap_or_else(|e| panic!("{shape} {format} {block:?} {reuse:?}: {e}"));
                        let tflops = flops / us / 1e6;
                        let tag = match reuse {
                            Reuse::Off => "off",
                            _ => "pull",
                        };
                        println!("SWEEP {shape} B {format} {block:?} {tag}: {us:.1} us, {tflops:.2} TFLOPS");
                        if block == candidates[0] && reuse == Reuse::Off {
                            auto_us = Some(us);
                        }
                        if best.is_none_or(|(b, ..)| us < b) {
                            best = Some((us, block, reuse));
                        }
                    }
                }
                let (best_us, block, reuse) = best.expect("a block");
                let auto_us = auto_us.expect("Auto's block timed");
                check(
                    &mut s,
                    (&a, &av),
                    (bt, bvals),
                    [m, k, n],
                    BlockStream::Fixed(block),
                    reuse,
                    &format!("{shape} B {format} best {block:?} {reuse:?}"),
                );
                check(
                    &mut s,
                    (&a, &av),
                    (bt, bvals),
                    [m, k, n],
                    BlockStream::Auto,
                    Reuse::Off,
                    &format!("{shape} B {format} Auto"),
                );
                for (what, us) in [("auto", auto_us), ("best", best_us)] {
                    let tflops = flops / us / 1e6;
                    println!(
                        "BENCH {{\"kind\":\"result\",\"key\":\"block sweep {shape} B {format} {what}\",\"unit\":\"TFLOPS\",\"timed\":\"host\",\"median\":{tflops},\"p10\":{tflops},\"p90\":{tflops},\"n\":5}}"
                    );
                }
                println!(
                    "MEASURE sweep {shape} B {format}: best {block:?} {reuse:?} {:.2} TFLOPS against Auto's {:?} {:.2}",
                    flops / best_us / 1e6,
                    candidates[0],
                    flops / auto_us / 1e6
                );
            }
            for t in [a, b, bb].into_iter().chain(weights.into_iter().map(|(_, t, _)| t)) {
                s.free(t).unwrap();
            }
        }
    })
    .unwrap();
}

/// `TT_SWEEP_BLOCKS` (`mb,kq,nb,pull|off;...`): the blocks [`block_confirm`] times again.
fn listed() -> Vec<([usize; 3], bool)> {
    let v = std::env::var("TT_SWEEP_BLOCKS").expect("TT_SWEEP_BLOCKS: mb,kq,nb,pull|off;...");
    v.split(';')
        .map(|b| {
            let p: Vec<&str> = b.split(',').map(str::trim).collect();
            let n = |i: usize| p[i].parse::<usize>().unwrap();
            ([n(0), n(1), n(2)], p[3] == "pull")
        })
        .collect()
}

/// The sweep's best blocks timed again, to tell them apart from its noise: every block of
/// `TT_SWEEP_BLOCKS`, and `Auto`'s, for the one shape of `TT_PAIR_SHAPES` and each format of
/// `TT_SWEEP_FORMATS`, timed in rounds (`TT_SWEEP_ROUNDS`, default 5), each round every block once
/// in turn, so drift falls on all of them alike; the median of the rounds, with their least and
/// most. Each block's product is checked against exact arithmetic once. Lines: `CONFIRM <shape> B
/// <format> <block> <reuse>: median <TFLOPS> (<least>-<most>)`.
#[test]
#[ignore = "a probe: cargo xtask bench --device 1 --filter silicon_probe_block_sweep::block_confirm"]
fn block_confirm() {
    let rounds: usize = std::env::var("TT_SWEEP_ROUNDS").map_or(5, |v| v.parse().unwrap());
    let blocks = listed();
    tt_ttsim::fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::All,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_batching(true).unwrap();
        let pull = Reuse::Pull {
            holders: AUTO_HOLDERS,
            b_holders: None,
        };
        let [m, k, n] = shapes()[0];
        let (av, bv) = (values(m * k, 1), values(k * n, 3));
        let a = s.upload(&av, m, k).unwrap();
        let b = s.upload(&bv, k, n).unwrap();
        let shape = format!("[{m},{k}]@[{k},{n}]");
        let flops = 2.0 * (m * k * n) as f64;
        for format in formats() {
            let (bt, bvals) = match format.as_str() {
                "FP32" => (b.clone(), bv.clone()),
                "BF16" => (s.store_bf16(&b).unwrap(), bv.clone()),
                f => {
                    let f = match f {
                        "BFP8" => tt_kernels::bfp::BfpFormat::Bfp8,
                        "BFP4" => tt_kernels::bfp::BfpFormat::Bfp4,
                        "BFP2" => tt_kernels::bfp::BfpFormat::Bfp2,
                        other => panic!("TT_SWEEP_FORMATS: {other}"),
                    };
                    let p = s.bfp_from_f32(&b, f).unwrap();
                    let decoded = s.bfp_to_f32(&p).unwrap();
                    let values = s.download(&decoded).unwrap();
                    s.free(decoded).unwrap();
                    (DramTensor::from_bfp(p), values)
                }
            };
            let q = s
                .block_question_of((&a, false), (&bt, false), Fidelity::HiFi4)
                .unwrap()
                .expect("a blocked product");
            let Some(auto) = tensor::block_stream_candidates(&q, None)
                .first()
                .map(|c| c.block)
            else {
                println!("MEASURE confirm {shape} B {format}: Auto does not block it");
                continue;
            };
            let mut runs: Vec<([usize; 3], Reuse)> = vec![(auto, Reuse::Off)];
            for &(block, pulled) in &blocks {
                if tensor::block_stream_fits(&q, block) {
                    runs.push((block, if pulled { pull } else { Reuse::Off }));
                } else {
                    println!("MEASURE confirm {shape} B {format}: {block:?} does not fit");
                }
            }
            for &(block, reuse) in &runs {
                check(
                    &mut s,
                    (&a, &av),
                    (&bt, &bvals),
                    [m, k, n],
                    BlockStream::Fixed(block),
                    reuse,
                    &format!("{shape} B {format} {block:?} {reuse:?}"),
                );
            }
            let mut times: Vec<Vec<f64>> = vec![Vec::new(); runs.len()];
            for _ in 0..rounds {
                for (i, &(block, reuse)) in runs.iter().enumerate() {
                    let us = s
                        .time_product(
                            (&a, false),
                            (&bt, false),
                            SrcRoute::Tf32FromFp32,
                            Fidelity::HiFi4,
                            BUDGET * 400,
                            BlockStream::Fixed(block),
                            reuse,
                            1,
                        )
                        .unwrap_or_else(|e| panic!("{shape} {format} {block:?}: {e}"));
                    times[i].push(flops / us / 1e6);
                }
            }
            for (i, &(block, reuse)) in runs.iter().enumerate() {
                let t = &mut times[i];
                t.sort_by(f64::total_cmp);
                let (median, least, most) = (t[t.len() / 2], t[0], t[t.len() - 1]);
                let tag = match (i, reuse) {
                    (0, _) => "auto",
                    (_, Reuse::Off) => "off",
                    _ => "pull",
                };
                println!(
                    "CONFIRM {shape} B {format} {block:?} {tag}: median {median:.2} ({least:.2}-{most:.2}) TFLOPS"
                );
                println!(
                    "BENCH {{\"kind\":\"result\",\"key\":\"block confirm {shape} B {format} {block:?} {tag}\",\"unit\":\"TFLOPS\",\"timed\":\"host\",\"median\":{median},\"p10\":{least},\"p90\":{most},\"n\":{rounds}}}"
                );
            }
            if format != "FP32" {
                s.free(bt).unwrap();
            }
        }
        s.free(a).unwrap();
        s.free(b).unwrap();
    })
    .unwrap();
}

/// The measured choice as a session makes it for a trace (`BlockStream::Measured`, `Reuse::Auto`,
/// real times, the product captured),
/// for each shape of `TT_PAIR_SHAPES` and format of `TT_SWEEP_FORMATS`: how many it timed and how
/// long its first product took, then its choice and `Auto`'s timed again in rounds
/// (`TT_SWEEP_ROUNDS`, default 5) as [`block_confirm`] times them, and the choice's product
/// checked against exact arithmetic. Lines: `MEASURED <shape> B <format>: <choice>, <n> timed in
/// <s> s; median <TFLOPS> (<least>-<most>) against Auto's <block> <TFLOPS>`.
#[test]
#[ignore = "a probe: cargo xtask bench --device 1 --filter silicon_probe_block_sweep::block_measured"]
fn block_measured() {
    let rounds: usize = std::env::var("TT_SWEEP_ROUNDS").map_or(5, |v| v.parse().unwrap());
    tt_ttsim::fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::All,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_batching(true).unwrap();
        for [m, k, n] in shapes() {
            let (av, bv) = (values(m * k, 1), values(k * n, 3));
            let a = s.upload(&av, m, k).unwrap();
            let b = s.upload(&bv, k, n).unwrap();
            let shape = format!("[{m},{k}]@[{k},{n}]");
            let flops = 2.0 * (m * k * n) as f64;
            for format in formats() {
                let (bt, bvals) = match format.as_str() {
                    "FP32" => (b.clone(), bv.clone()),
                    "BF16" => (s.store_bf16(&b).unwrap(), bv.clone()),
                    f => {
                        let f = match f {
                            "BFP8" => tt_kernels::bfp::BfpFormat::Bfp8,
                            "BFP4" => tt_kernels::bfp::BfpFormat::Bfp4,
                            "BFP2" => tt_kernels::bfp::BfpFormat::Bfp2,
                            other => panic!("TT_SWEEP_FORMATS: {other}"),
                        };
                        let p = s.bfp_from_f32(&b, f).unwrap();
                        let decoded = s.bfp_to_f32(&p).unwrap();
                        let values = s.download(&decoded).unwrap();
                        s.free(decoded).unwrap();
                        (DramTensor::from_bfp(p), values)
                    }
                };
                let q = s
                    .block_question_of((&a, false), (&bt, false), Fidelity::HiFi4)
                    .unwrap()
                    .expect("a blocked product");
                let Some(auto) = tensor::block_stream_candidates(&q, None).first().map(|c| c.block)
                else {
                    println!("MEASURED {shape} B {format}: Auto does not block it");
                    continue;
                };
                tensor::forget_measured();
                static TIMED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                TIMED.store(0, std::sync::atomic::Ordering::SeqCst);
                fn count(_: [usize; 3], _: Reuse, us: f64) -> f64 {
                    TIMED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    us
                }
                s.probe_block_scores(Some(count));
                s.set_block_stream(BlockStream::Measured);
                s.set_operand_reuse(Reuse::Auto).unwrap();
                // Measured as a trace's product is: in a capture, paused for it.
                let start = std::time::Instant::now();
                s.begin_trace().unwrap();
                let c = s
                    .matmul_dram(&a, false, &bt, false, SrcRoute::Tf32FromFp32, Fidelity::HiFi4, BUDGET * 400)
                    .unwrap();
                let id = s.end_trace().unwrap();
                s.sync().unwrap();
                let first_s = start.elapsed().as_secs_f64();
                s.free(c).unwrap();
                s.release_trace(id).unwrap();
                s.probe_block_scores(None);
                let choice = tensor::measured_choice(&q)
                    .flatten()
                    .expect("a measured choice");
                check(
                    &mut s,
                    (&a, &av),
                    (&bt, &bvals),
                    [m, k, n],
                    BlockStream::Fixed(choice.block),
                    choice.reuse,
                    &format!("{shape} B {format} measured {choice:?}"),
                );
                let runs = [(choice.block, choice.reuse), (auto, Reuse::Off)];
                let mut times: Vec<Vec<f64>> = vec![Vec::new(), Vec::new()];
                for _ in 0..rounds {
                    for (i, &(block, reuse)) in runs.iter().enumerate() {
                        let us = s
                            .time_product(
                                (&a, false),
                                (&bt, false),
                                SrcRoute::Tf32FromFp32,
                                Fidelity::HiFi4,
                                BUDGET * 400,
                                BlockStream::Fixed(block),
                                reuse,
                                1,
                            )
                            .unwrap();
                        times[i].push(flops / us / 1e6);
                    }
                }
                for t in &mut times {
                    t.sort_by(f64::total_cmp);
                }
                let med = |t: &[f64]| t[t.len() / 2];
                println!(
                    "MEASURED {shape} B {format}: {choice:?}, {} timed in {first_s:.1} s; median {:.2} ({:.2}-{:.2}) against Auto's {auto:?} {:.2}",
                    TIMED.load(std::sync::atomic::Ordering::SeqCst),
                    med(&times[0]),
                    times[0][0],
                    times[0][rounds - 1],
                    med(&times[1]),
                );
                for (what, t) in [("measured", &times[0]), ("auto", &times[1])] {
                    let (median, least, most) = (med(t), t[0], t[rounds - 1]);
                    println!(
                        "BENCH {{\"kind\":\"result\",\"key\":\"block measured {shape} B {format} {what}\",\"unit\":\"TFLOPS\",\"timed\":\"host\",\"median\":{median},\"p10\":{least},\"p90\":{most},\"n\":{rounds}}}"
                    );
                }
                s.set_block_stream(BlockStream::Auto);
                s.set_operand_reuse(Reuse::Off).unwrap();
                if format != "FP32" {
                    s.free(bt).unwrap();
                }
            }
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
    })
    .unwrap();
}
