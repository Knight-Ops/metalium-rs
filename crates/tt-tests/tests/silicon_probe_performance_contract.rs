//! Roadmap W1, "Bound it first" (`docs/plans/tt-metal-parity.md` §5 W1; ADR 0006): what the
//! performance contract's products would cost on this stack before any Burn plumbing. SmolLM2-1.7B's
//! products at its decode and prefill shapes, on every tile, release, traced, through
//! `Session::matmul_dram` (the path a model's products take), over:
//! - `A` (the activation) in FP32 or BF16 (`Session::upload_bf16`, interleaved, as an activation
//!   would lie);
//! - `B` (the weight) stored BF16, BFP8 or BFP4, laid as a model's weights are
//!   (`Session::upload_bf16_weight`, `Session::store_bfp`);
//! - HiFi4, HiFi2 and LoFi.
//!
//! The reference contract's row is `A` FP32, `B` BF16, HiFi4: what the 1.7B decodes with today
//! (9.90 ms a step, card 1). The performance contract's row is `A` BF16, `B` BFP8, at the derived
//! fidelity (`SrcRoute::full_precision`) capped at HiFi2; BFP4 at LoFi is its named option.
//!
//! - **`decode_products`:** qkv, o, gate|up, down and the head at `m` = 1 (the 1.7B's batch-1 step)
//!   and 32 (tt-metal's `decode_mm.py` rows; W3's first look). Each op is timed as a trace of
//!   `DECODE_REPS` copies less a trace of one (`tt_tests::bench::trace_slope_us`), and each
//!   configuration's products summed as a step's (24 layers and the head) and turned into a
//!   predicted step: `TT_CONTRACT_STEP_MS` (9.90) less the reference row's sum plus its own.
//! - **`prefill_products`:** tt-metal's `prefill_mm.py` shapes and H13's 512-row chunk of the 1.7B,
//!   in TFLOPS, under a shorter list of configurations (`PREFILL_CONFIGS`), and each again with
//!   `A`'s gathers skipped (`tensor::probe_skip_a_reads`: what `A`'s reads cost).
//! - **`prefill_products_without_the_reload`** (its own process: the hook is read when programs
//!   are built): each continuing K chunk without reloading its priors into `Dst`
//!   (`matmul::probe_skip_reload`): the most packer L1 accumulation could save.
//!
//! **Validated outside the timing.** The operands are small integers (every product and sum exact
//! at every fidelity: at most two significant bits), and every timed product but the hooks' equals
//! the exact sums over the values its operands decode to, bit for bit. Then, a reading and not a
//! gate (ADR 0006's gate is W1's per-op specification models), each configuration once more on
//! values in [-1, 1]: its PCC against an f64 product of the source values (what tt-metal's
//! `ttnn_parity` scripts report) and its largest error against an f64 product of the decoded
//! values, over the largest such product (the fidelity's own loss).
//!
//! Knobs: `TT_CONTRACT_ROWS` (`1,32`), `TT_CONTRACT_A` (`fp32,bf16`), `TT_CONTRACT_B`
//! (`bf16,bfp8,bfp4`), `TT_CONTRACT_FIDELITIES` (`hifi4,hifi2,lo`), `TT_CONTRACT_STEP_MS`
//! (`9.90`), `TT_CONTRACT_PREFILL` (`m,k,n;...`). Silicon only, a probe:
//! `cargo xtask bench --device 1 --filter silicon_probe_performance_contract`.
#![cfg(feature = "silicon")]

use tt_isa::format::TileFormat;
use tt_kernels::bfp::BfpFormat;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Pad};
use tt_tests::backend::device_index;
use tt_tests::bench::trace_slope_us;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

type Card = Session<tt_kmd::Kmd>;

const WARMUPS: usize = 2;
const SAMPLES: usize = 7;
/// Copies of a decode product in the long trace.
const DECODE_REPS: usize = 8;
/// Copies of a prefill product in the long trace (each is milliseconds).
const PREFILL_REPS: usize = 3;
/// SmolLM2-1.7B's layers (`tt_models::llama::SMOLLM2_1_7B_INSTRUCT`).
const LAYERS: f64 = 24.0;

/// A layer's products and the head, `(name, k, n)`, hidden 2048, ff 8192, vocabulary 49,152.
const DECODE_SHAPES: [(&str, usize, usize); 5] = [
    ("qkv", 2048, 6144),
    ("o", 2048, 2048),
    ("gate|up", 2048, 16384),
    ("down", 8192, 2048),
    ("head", 2048, 49152),
];

/// tt-metal's `prefill_mm.py` shapes and H13's 512-row chunk of the 1.7B.
const PREFILL_SHAPES: [[usize; 3]; 7] = [
    [2048, 2048, 16384],
    [1024, 8192, 2048],
    [2048, 8192, 2048],
    [4096, 4096, 4096],
    [512, 2048, 6144],
    [512, 2048, 16384],
    [512, 8192, 2048],
];

/// The prefill configurations, `(A, B, fidelity)`: the reference row, P7's LoFi with a BF16 `B`,
/// BF16 `A` at each fidelity, and the contract's BF16 x BFP8 (tt-metal's `prefill_mm.py` modes
/// `bf16:bf16:HiFi4`, `bf16:bf16:HiFi2`, `bf16:bfp8:LoFi` among them).
const PREFILL_CONFIGS: [(TileFormat, TileFormat, Fidelity); 7] = [
    (TileFormat::FP32, TileFormat::BF16, Fidelity::HiFi4),
    (TileFormat::FP32, TileFormat::BF16, Fidelity::Lo),
    (TileFormat::BF16, TileFormat::BF16, Fidelity::HiFi4),
    (TileFormat::BF16, TileFormat::BF16, Fidelity::HiFi2),
    (TileFormat::BF16, TileFormat::BF16, Fidelity::Lo),
    (TileFormat::BF16, TileFormat::BFP8, Fidelity::HiFi2),
    (TileFormat::BF16, TileFormat::BFP8, Fidelity::Lo),
];

fn env_list<T>(key: &str, default: &str, parse: impl Fn(&str) -> T) -> Vec<T> {
    std::env::var(key)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(|v| parse(v.trim()))
        .collect()
}

fn format_named(v: &str) -> TileFormat {
    match v {
        "fp32" => TileFormat::FP32,
        "bf16" => TileFormat::BF16,
        "bfp8" => TileFormat::BFP8,
        "bfp4" => TileFormat::BFP4,
        other => panic!("a format: fp32, bf16, bfp8 or bfp4, not {other}"),
    }
}

fn fidelity_named(v: &str) -> Fidelity {
    match v {
        "lo" => Fidelity::Lo,
        "hifi2" => Fidelity::HiFi2,
        "hifi3" => Fidelity::HiFi3,
        "hifi4" => Fidelity::HiFi4,
        other => panic!("a fidelity: lo, hifi2, hifi3 or hifi4, not {other}"),
    }
}

fn name(f: TileFormat) -> &'static str {
    match f {
        TileFormat::FP32 => "FP32",
        TileFormat::BF16 => "BF16",
        TileFormat::BFP8 => "BFP8",
        TileFormat::BFP4 => "BFP4",
        TileFormat::BFP2 => "BFP2",
        _ => "?",
    }
}

fn fid(f: Fidelity) -> &'static str {
    match f {
        Fidelity::Lo => "LoFi",
        Fidelity::HiFi2 => "HiFi2",
        Fidelity::HiFi3 => "HiFi3",
        Fidelity::HiFi4 => "HiFi4",
    }
}

/// The contract's fidelity for a pair: the derived one (every mantissa bit the edges bring),
/// capped at HiFi2, and LoFi wherever an edge is BFP4 or narrower (ADR 0006).
fn contract_fidelity(a: TileFormat, b: TileFormat) -> Fidelity {
    let derived = SrcRoute::edges(a, b)
        .expect("a pair a gather reads")
        .full_precision();
    if [a, b]
        .iter()
        .any(|&f| f == TileFormat::BFP4 || f == TileFormat::BFP2)
    {
        return Fidelity::Lo;
    }
    Fidelity::from_phases(derived.phases().min(2)).unwrap()
}

/// Integers in [-2, 2]: every product and sum exact in FP32 at any fidelity, in any format here.
fn integers(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + seed) % 5) as f32 - 2.0).collect()
}

/// Values in about [-1, 1] with every FP32 mantissa bit live.
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

/// `v` rounded to BF16 (ties to even), held in F32.
fn bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

/// `vals` as a `[rows, cols]` operand in `format` (an activation interleaved, a weight laid as
/// stored weights are), and the values it decodes to.
fn operand(
    s: &mut Card,
    format: TileFormat,
    weight: bool,
    vals: &[f32],
    rows: usize,
    cols: usize,
) -> (DramTensor, Vec<f32>) {
    match format {
        TileFormat::FP32 => (s.upload(vals, rows, cols).unwrap(), vals.to_vec()),
        TileFormat::BF16 => {
            let rounded: Vec<f32> = vals.iter().map(|&v| bf16(v)).collect();
            let bits: Vec<u16> = rounded.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let t = if weight {
                s.upload_bf16_weight(&bits, rows, cols).unwrap()
            } else {
                DramTensor::from_bf16(s.upload_bf16(&bits, rows, cols).unwrap(), Pad::Zero)
            };
            (t, rounded)
        }
        f => {
            let bfp = BfpFormat::of(f).expect("a BFP format");
            assert!(weight, "a BFP activation is not in the contract");
            let w = s.upload(vals, rows, cols).unwrap();
            let t = s.store_bfp(&w, bfp).unwrap();
            s.free(w).unwrap();
            let decoded = s.bfp_to_f32(&t.as_bfp().unwrap()).unwrap();
            let v = s.download(&decoded).unwrap();
            s.free(decoded).unwrap();
            (t, v)
        }
    }
}

fn product(s: &mut Card, a: &DramTensor, b: &DramTensor, fidelity: Fidelity) -> DramTensor {
    s.matmul_dram(a, false, b, false, SrcRoute::Tf32FromFp32, fidelity, BUDGET)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// The output elements a check reads: every row up to 32, then a stride; columns a stride, and
/// the last of each.
fn samples(m: usize, n: usize) -> Vec<(usize, usize)> {
    let rows: Vec<usize> = if m <= 32 {
        (0..m).collect()
    } else {
        (0..m).step_by(97).chain([m - 1]).collect()
    };
    let step = (n / 64).max(1) | 1;
    let cols: Vec<usize> = (0..n).step_by(step).chain([n - 1]).collect();
    rows.iter()
        .flat_map(|&i| cols.iter().map(move |&j| (i, j)))
        .collect()
}

/// Every sampled element of `out` equals the exact sum over the decoded operands.
fn check_exact(what: &str, out: &[f32], a: &[f32], b: &[f32], [m, k, n]: [usize; 3]) {
    for (i, j) in samples(m, n) {
        let exact: f32 = (0..k).map(|q| a[i * k + q] * b[q * n + j]).sum();
        assert_eq!(
            out[i * n + j].to_bits(),
            exact.to_bits(),
            "{what}: ({i}, {j}): {} against {exact}",
            out[i * n + j]
        );
    }
}

/// `(PCC against the source values' f64 product, largest error against the decoded values' f64
/// product over that product's largest magnitude)`, over the sampled elements.
fn accuracy(
    out: &[f32],
    (a, a_dec): (&[f32], &[f32]),
    (b, b_dec): (&[f32], &[f32]),
    [m, k, n]: [usize; 3],
) -> (f64, f64) {
    let dot = |x: &[f32], y: &[f32], i: usize, j: usize| -> f64 {
        (0..k)
            .map(|q| x[i * k + q] as f64 * y[q * n + j] as f64)
            .sum()
    };
    let (mut got, mut src, mut dec) = (Vec::new(), Vec::new(), Vec::new());
    for (i, j) in samples(m, n) {
        got.push(out[i * n + j] as f64);
        src.push(dot(a, b, i, j));
        dec.push(dot(a_dec, b_dec, i, j));
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (mg, ms) = (mean(&got), mean(&src));
    let cov: f64 = got.iter().zip(&src).map(|(g, s)| (g - mg) * (s - ms)).sum();
    let vg: f64 = got.iter().map(|g| (g - mg).powi(2)).sum();
    let vs: f64 = src.iter().map(|s| (s - ms).powi(2)).sum();
    let pcc = cov / (vg * vs).sqrt();
    let scale = dec.iter().fold(0f64, |a, d| a.max(d.abs()));
    let err = got
        .iter()
        .zip(&dec)
        .fold(0f64, |a, (g, d)| a.max((g - d).abs()));
    (pcc, err / scale)
}

fn open() -> Card {
    let mut s = Session::open_card(device_index(), tt_firmware_images::ROLES, TileChoice::All)
        .unwrap_or_else(|e| panic!("{e}"));
    s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
        .unwrap();
    s.set_batching(true).unwrap();
    s
}

fn conditions(what: &str, reps: usize) {
    println!(
        "BENCH {{\"kind\":\"conditions\",\"probe\":\"{what}\",\"git\":\"{}\",\"release\":{},\"warmups\":{WARMUPS},\"samples\":{SAMPLES},\"reps\":{reps},\"timing\":\"replay of {reps} ops less replay of 1, over {}\"}}",
        tt_tests::bench::git_sha(),
        !cfg!(debug_assertions),
        reps - 1
    );
}

fn bench(key: &str, unit: &str, v: f64) {
    println!(
        "BENCH {{\"kind\":\"result\",\"key\":\"{key}\",\"unit\":\"{unit}\",\"timed\":\"host\",\"median\":{v},\"p10\":{v},\"p90\":{v},\"n\":{SAMPLES}}}"
    );
}

/// One configuration's accuracy reading on values in [-1, 1], eager.
fn read_accuracy(
    s: &mut Card,
    (fa, fb, fidelity): (TileFormat, TileFormat, Fidelity),
    [m, k, n]: [usize; 3],
) -> (f64, f64) {
    let (av, bv) = (floats(11, m * k), floats(13, k * n));
    let (a, a_dec) = operand(s, fa, false, &av, m, k);
    let (b, b_dec) = operand(s, fb, true, &bv, k, n);
    let c = product(s, &a, &b, fidelity);
    let out = s.download(&c).unwrap();
    for t in [a, b, c] {
        s.free(t).unwrap();
    }
    accuracy(&out, (&av, &a_dec), (&bv, &b_dec), [m, k, n])
}

#[test]
#[ignore = "a probe: cargo xtask bench --filter silicon_probe_performance_contract"]
fn decode_products() {
    let rows = env_list("TT_CONTRACT_ROWS", "1,32", |v| v.parse::<usize>().unwrap());
    let a_formats = env_list("TT_CONTRACT_A", "fp32,bf16", format_named);
    let b_formats = env_list("TT_CONTRACT_B", "bf16,bfp8,bfp4", format_named);
    let fidelities = env_list("TT_CONTRACT_FIDELITIES", "hifi4,hifi2,lo", fidelity_named);
    let step_ms: f64 = std::env::var("TT_CONTRACT_STEP_MS").map_or(9.90, |v| {
        v.parse().expect("TT_CONTRACT_STEP_MS: milliseconds")
    });
    let reference = (TileFormat::FP32, TileFormat::BF16, Fidelity::HiFi4);
    let mut configs = vec![reference];
    for &fa in &a_formats {
        for &fb in &b_formats {
            for &f in &fidelities {
                if !configs.contains(&(fa, fb, f)) {
                    configs.push((fa, fb, f));
                }
            }
        }
    }
    fork_scope(|| {
        let mut s = open();
        conditions("decode_products", DECODE_REPS);
        let mut pairs: Vec<(TileFormat, TileFormat)> = Vec::new();
        for &(fa, fb, _) in &configs {
            if !pairs.contains(&(fa, fb)) {
                pairs.push((fa, fb));
            }
        }
        for &(fa, fb) in &pairs {
            let derived = SrcRoute::edges(fa, fb).unwrap().full_precision();
            println!(
                "MEASURE contract fidelity A {} B {}: derived {}, the contract's {}",
                name(fa),
                name(fb),
                fid(derived),
                fid(contract_fidelity(fa, fb))
            );
        }
        for &m in &rows {
            // Per configuration: each op's microseconds, in DECODE_SHAPES' order.
            let mut times: Vec<Vec<f64>> = vec![Vec::new(); configs.len()];
            for &(op, k, n) in &DECODE_SHAPES {
                let (av, bv) = (integers(m * k, 1), integers(k * n, 3));
                for (c, &(fa, fb, fidelity)) in configs.iter().enumerate() {
                    let what = format!(
                        "decode m={m} {op} [{k},{n}] A {} B {} {}",
                        name(fa),
                        name(fb),
                        fid(fidelity)
                    );
                    let (a, a_dec) = operand(&mut s, fa, false, &av, m, k);
                    let (b, b_dec) = operand(&mut s, fb, true, &bv, k, n);
                    let mut outs = Vec::new();
                    let us = trace_slope_us(&mut s, DECODE_REPS, WARMUPS, SAMPLES, |s| {
                        outs.push(product(s, &a, &b, fidelity))
                    });
                    let got = s.download(outs.last().unwrap()).unwrap();
                    check_exact(&what, &got, &a_dec, &b_dec, [m, k, n]);
                    for t in outs {
                        s.free(t).unwrap();
                    }
                    let bytes = (k.div_ceil(32) * n.div_ceil(32)) as f64 * fb.slot_bytes() as f64;
                    println!(
                        "MEASURE contract {what}: {us:.1} us, B {:.0} GB/s",
                        bytes / us / 1e3
                    );
                    bench(&format!("contract {what}"), "us", us);
                    times[c].push(us);
                    for t in [a, b] {
                        s.free(t).unwrap();
                    }
                }
            }
            let step = |t: &[f64]| (LAYERS * (t[0] + t[1] + t[2] + t[3]) + t[4]) / 1e3;
            let reference_ms = step(&times[0]);
            for (c, &(fa, fb, fidelity)) in configs.iter().enumerate() {
                let ms = step(&times[c]);
                let marks = match (fa, fb, fidelity) {
                    _ if c == 0 => " (the reference contract)",
                    (TileFormat::BF16, TileFormat::BFP8, f) if f == contract_fidelity(fa, fb) => {
                        " (the performance contract)"
                    }
                    _ => "",
                };
                let what = format!(
                    "decode m={m} A {} B {} {}",
                    name(fa),
                    name(fb),
                    fid(fidelity)
                );
                println!(
                    "MEASURE contract {what}{marks}: products {ms:.3} ms a step (24 layers + head), \
                     predicted step {:.2} ms ({step_ms} - {reference_ms:.3} + {ms:.3})",
                    step_ms - reference_ms + ms
                );
                bench(&format!("contract {what}, products a step"), "ms", ms);
            }
        }
        // The accuracy reading: one row, the 1.7B's gate|up and down (each of K's two sizes).
        for &(op, k, n) in &DECODE_SHAPES[2..4] {
            for &config in &configs {
                let (pcc, err) = read_accuracy(&mut s, config, [1, k, n]);
                println!(
                    "MEASURE contract accuracy decode m=1 {op} [{k},{n}] A {} B {} {}: PCC {pcc:.7} \
                     against the source, largest error {err:.3e} of the decoded product's largest",
                    name(config.0),
                    name(config.1),
                    fid(config.2)
                );
            }
        }
    })
    .unwrap_or_else(|e| panic!("{e}"));
}

fn prefill_shapes() -> Vec<[usize; 3]> {
    std::env::var("TT_CONTRACT_PREFILL").map_or(PREFILL_SHAPES.to_vec(), |v| {
        v.split(';')
            .map(|s| {
                let d: Vec<usize> = s.split(',').map(|x| x.trim().parse().unwrap()).collect();
                [d[0], d[1], d[2]]
            })
            .collect()
    })
}

fn prefill(skip_reload: bool) {
    fork_scope(|| {
        // In the child, before its session builds any program.
        tt_kernels::matmul::probe_skip_reload(skip_reload);
        let mut s = open();
        conditions(
            if skip_reload {
                "prefill_products_without_the_reload"
            } else {
                "prefill_products"
            },
            PREFILL_REPS,
        );
        for [m, k, n] in prefill_shapes() {
            let (av, bv) = (integers(m * k, 1), integers(k * n, 3));
            let flops = 2.0 * (m * k * n) as f64;
            for &(fa, fb, fidelity) in &PREFILL_CONFIGS {
                let what = format!(
                    "prefill [{m},{k}]@[{k},{n}] A {} B {} {}",
                    name(fa),
                    name(fb),
                    fid(fidelity)
                );
                let (a, a_dec) = operand(&mut s, fa, false, &av, m, k);
                let (b, b_dec) = operand(&mut s, fb, true, &bv, k, n);
                let variants: &[(bool, &str)] = if skip_reload {
                    &[(false, ", without the reload")]
                } else {
                    &[(false, ""), (true, ", without A's gathers")]
                };
                for &(skip_a, variant) in variants {
                    tt_kernels::tensor::probe_skip_a_reads(skip_a);
                    let mut outs = Vec::new();
                    let us = trace_slope_us(&mut s, PREFILL_REPS, 1, SAMPLES, |s| {
                        outs.push(product(s, &a, &b, fidelity))
                    });
                    tt_kernels::tensor::probe_skip_a_reads(false);
                    if !skip_a && !skip_reload {
                        let got = s.download(outs.last().unwrap()).unwrap();
                        check_exact(&what, &got, &a_dec, &b_dec, [m, k, n]);
                    }
                    for t in outs {
                        s.free(t).unwrap();
                    }
                    let tflops = flops / us / 1e6;
                    println!("MEASURE contract {what}{variant}: {us:.1} us, {tflops:.2} TFLOPS");
                    bench(&format!("contract {what}{variant}"), "TFLOPS", tflops);
                }
                for t in [a, b] {
                    s.free(t).unwrap();
                }
                if !skip_reload {
                    let (pcc, err) = read_accuracy(&mut s, (fa, fb, fidelity), [m, k, n]);
                    println!(
                        "MEASURE contract accuracy {what}: PCC {pcc:.7} against the source, \
                         largest error {err:.3e} of the decoded product's largest"
                    );
                }
            }
        }
    })
    .unwrap_or_else(|e| panic!("{e}"));
}

#[test]
#[ignore = "a probe: cargo xtask bench --filter silicon_probe_performance_contract"]
fn prefill_products() {
    prefill(false);
}

#[test]
#[ignore = "a probe: cargo xtask bench --filter silicon_probe_performance_contract"]
fn prefill_products_without_the_reload() {
    prefill(true);
}
