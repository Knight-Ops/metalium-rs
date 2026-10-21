//! 2h-i of `docs/plans/planned-l1-residency.md`: where a persistent region starts to cost the
//! one-row products, and what predicts it.
//!
//! `cargo xtask silicon --release --device 0 --include-ignored --filter silicon_probe_region_cliff`
//!
//! A region kept at the top of each unit's `DATA` (`Session::set_l1_region`) shrinks the arena the
//! captured ops plan in. Phase 2a measured the 1.7B's step: free up to 96 KiB, +1.6 ms past ~128
//! KiB, where its products take narrower blocks. The rule this probes (`matmul::plan_floor`): a
//! product's block is the widest whose probe layout fits, with `A` unpacked, so a `K`-whole
//! product keeps its layout while the region is at most `DATA.end` less that probe's top, 110,464
//! bytes at `K` = 2048; a streamed one (`K` blocked) while its own chunk layout fits.
//!
//! For each product, on every tile, release, the decode path's own deal (`THIN_STREAM_UNITS`),
//! traced (a trace of `REPS` against a trace of one, warm-ups and medians), under regions from 0
//! to past the predicted limit:
//! - its time, its read entries a unit (`OpBytes::reads`, `Session::set_count_trace_bytes`) and
//!   its stream's entries a unit (`OpRecord::entries`), its block (`matmul::probe_plan_at`) and the need its trace records (`OpRecord::l1_need`);
//! - the predicted limit, from the trace with no region: `DATA.end` less its highest need.
//!
//! Shapes: the 1.7B's (and Llama-3.2-1B's) `qkv`, `gate|up` and head at `K` = 2048, Qwen2.5-1.5B's
//! `gate|up` at 1536, an 8B's `qkv` at 4096, and the `down` products at 8192, 8960 and 14,336.
//! Validated outside the timing: every region's replayed product equals the eager product bit for
//! bit.
#![cfg(feature = "silicon")]

use std::time::Instant;

use tt_isa::l1::DATA;
use tt_kernels::matmul::{Fidelity, SrcRoute, Staging};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::DramTensor;
use tt_kernels::trace::TraceId;
use tt_tests::backend::device_index;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

type Card = Session<tt_kmd::Kmd>;

const WARMUPS: usize = 2;
const SAMPLES: usize = 9;
/// Copies of the op in the long trace.
const REPS: usize = 8;
const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;
const FIDELITY: Fidelity = Fidelity::HiFi4;

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The median of `SAMPLES` host-timed `replay` + `sync`, after `WARMUPS`, in microseconds.
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

fn product(s: &mut Card, x: &DramTensor, w: &DramTensor) -> DramTensor {
    s.matmul_dram(x, false, w, false, ROUTE, FIDELITY, BUDGET)
        .unwrap()
}

/// What one region gives a product: its time, its read entries a unit, its highest need, and the
/// bits of a replay.
struct Point {
    us: f64,
    entries_a_unit: f64,
    stream_a_unit: f64,
    need: u64,
    bits: Vec<u32>,
}

fn measure(s: &mut Card, x: &DramTensor, w: &DramTensor) -> Point {
    let mut outs = Vec::new();
    let mut trace = |s: &mut Card, n: usize| {
        s.begin_trace().unwrap();
        for _ in 0..n {
            outs.push(product(s, x, w));
        }
        s.end_trace().unwrap()
    };
    let one = trace(s, 1);
    let many = trace(s, REPS);
    let ops = s.trace_ops(one).unwrap().to_vec();
    let need = ops.iter().map(|o| o.l1_need).max().unwrap_or(0);
    // The product's own op: the one that reads most.
    let op = ops.iter().max_by_key(|o| o.bytes.reads).unwrap();
    let units = op.entries.iter().filter(|r| !r.is_empty()).count().max(1);
    let entries_a_unit = op.bytes.reads as f64 / units as f64;
    // Its stream's entries a unit: a narrower block is more jobs, each its own entries.
    let stream_a_unit = op.entries.iter().map(|r| r.len()).sum::<usize>() as f64 / units as f64;
    let (t1, tn) = (replay_us(s, one), replay_us(s, many));
    s.replay(one).unwrap();
    s.sync().unwrap();
    let got = bits(&s.download(&outs[0]).unwrap());
    for id in [one, many] {
        s.release_trace(id).unwrap();
    }
    for t in outs {
        s.free(t).unwrap();
    }
    Point {
        us: (tn - t1) / (REPS - 1) as f64,
        entries_a_unit,
        stream_a_unit,
        need,
        bits: got,
    }
}

#[test]
#[ignore = "benchmark"]
fn region_cliff_of_the_one_row_products() {
    let card = device_index();
    fork_scope(|| {
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::All)
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        // The region alone: no residency plan, which would size its own.
        s.set_residency(0);
        s.set_count_trace_bytes(true);
        println!(
            "BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"tiles\":{},\"warmups\":{WARMUPS},\"samples\":{SAMPLES},\"reps\":{REPS},\"route\":\"Tf32FromFp32\",\"fidelity\":\"HiFi4\",\"timing\":\"replay of {REPS} ops less replay of 1, over {}\"}}",
            tt_tests::bench::git_sha(),
            !cfg!(debug_assertions),
            s.tiles().len(),
            REPS - 1
        );
        for (name, k, n) in [
            ("1.7B qkv", 2048usize, 6144usize),
            ("1.7B gate|up", 2048, 16384),
            ("1.7B head", 2048, 49152),
            ("Qwen2.5-1.5B gate|up", 1536, 17920),
            ("8B qkv", 4096, 6144),
            ("1.7B down", 8192, 2048),
            ("Qwen2.5-1.5B down", 8960, 1536),
            ("8B down", 14336, 4096),
        ] {
            let x = s.upload(&floats(1, k), 1, k).unwrap();
            let w = s.upload(&floats(2, k * n), k, n).unwrap();
            let reference = product(&mut s, &x, &w);
            let want = bits(&s.download(&reference).unwrap());
            s.free(reference).unwrap();
            s.set_l1_region(0);
            let base = measure(&mut s, &x, &w);
            assert_eq!(base.bits, want, "{name}: no region");
            let limit = (DATA.end - base.need) / 64 * 64;
            let mut regions: Vec<u64> = [0u64, 32 << 10, 64 << 10, 96 << 10, 128 << 10]
                .into_iter()
                .chain([192 << 10, 256 << 10, 320 << 10, 384 << 10])
                .filter(|&r| r <= (256 << 10).max(limit + (64 << 10)))
                .chain([limit - 64, limit, limit + 64])
                .collect();
            regions.sort_unstable();
            regions.dedup();
            for region in regions {
                s.set_l1_region(region);
                let p = if region == 0 {
                    Point {
                        bits: base.bits.clone(),
                        ..base
                    }
                } else {
                    measure(&mut s, &x, &w)
                };
                assert_eq!(p.bits, want, "{name} under a region of {region}: a replay");
                let (chunk, _) = tt_kernels::matmul::probe_plan_at(
                    DATA.end - region.next_multiple_of(64),
                    [1, k, n],
                    ROUTE,
                    FIDELITY,
                    Staging::Slots,
                );
                let [_, kc, nc] = chunk.map_or([0; 3], |c| c.tiles);
                println!(
                    "REGIONCLIFF {{\"product\":\"{name}\",\"k\":{k},\"n\":{n},\"region\":{region},\
                     \"limit\":{limit},\"us\":{:.1},\"entries_a_unit\":{:.1},\
                     \"stream_a_unit\":{:.1},\"kc\":{kc},\"kt\":{},\"nc\":{nc},\"need\":{}}}",
                    p.us,
                    p.entries_a_unit,
                    p.stream_a_unit,
                    k / 32,
                    p.need
                );
            }
            s.set_l1_region(0);
            s.free(w).unwrap();
            s.free(x).unwrap();
        }
    })
    .unwrap_or_else(|e| panic!("{e}"));
}
