//! Phase 0 of `docs/plans/decode-to-bandwidth.md`: what bounds a decode product's weight read,
//! against the card's read ceiling (469 GB/s, `silicon_bench_memory`, 120 tiles).
//!
//! `cargo xtask silicon --release --device 0 --filter silicon_probe_decode_stream`
//!
//! For each of a SmolLM2-1.7B layer's products at `m = 1` (`qkv` `[2048, 6144]`, `o`
//! `[2048, 2048]`, `gate|up` `[2048, 16384]`, `down` `[8192, 2048]`), on every tile of the card,
//! release, traced (so the time is the card's), warm-ups and medians:
//! - **the product** (`Session::matmul_dram`, the decode path: thin `A`, a column a job), on 64
//!   units and on every unit;
//! - **its weights alone** (`Session::probe_stream`): every tile of `B` read once into L1 as
//!   `READ_RUN`s, on the same 64 and every unit, dealt as the product deals them (a column a
//!   unit, down its `K` tiles) and as consecutive tiles (row-major, which walks the channels).
//!
//! A trace holds `REPS` copies of the op, and the op's time is the difference between that and a
//! trace of one, over `REPS - 1`: a replay's start and the host's sync drop out. Rates are slot
//! bytes (4,160 a tile, as the profile's step budget counts them) over that time.
//!
//! Validated outside the timing: each replayed product equals the eager product bit for bit, and
//! the tiles a stream left in the first and last unit's slots equal `B`'s in GDDR.
#![cfg(feature = "silicon")]

use std::time::Instant;

use tt_device::tlb::WindowKind;
use tt_isa::dm::TILE_SLOT;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, Spread, StreamOrder, TileChoice};
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
/// Slots a stream's runs land in, at the top of `DATA`.
const WINDOW: usize = 64;

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

/// One op's time on the card: a trace of `REPS` against a trace of one, each built by `op`.
fn op_us(s: &mut Card, mut op: impl FnMut(&mut Card)) -> f64 {
    let mut trace = |s: &mut Card, n: usize| {
        s.begin_trace().unwrap();
        for _ in 0..n {
            op(s);
        }
        s.end_trace().unwrap()
    };
    let one = trace(s, 1);
    let many = trace(s, REPS);
    let (t1, tn) = (replay_us(s, one), replay_us(s, many));
    (tn - t1) / (REPS - 1) as f64
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Every tile a stream left in unit `u`'s slots equals `w`'s in GDDR.
fn landed(s: &mut Card, w: &DramTensor, u: usize, tiles: &[[usize; 2]]) {
    let tile = s.tiles()[u];
    let window = s.device().alloc_window(WindowKind::FourGib).unwrap();
    let at = tt_isa::l1::DATA.end - WINDOW as u64 * TILE_SLOT;
    let mut l1 = vec![0u8; tiles.len() * TILE_SLOT as usize];
    s.device().l1_read(&window, tile, at, &mut l1).unwrap();
    for (n, &[i, j]) in tiles.iter().enumerate() {
        let mut gddr = vec![0u8; TILE_SLOT as usize];
        s.device()
            .dram_read(&window, w.tile(i, j), &mut gddr)
            .unwrap();
        let slot = &l1[n * TILE_SLOT as usize..(n + 1) * TILE_SLOT as usize];
        assert!(
            slot == &gddr[..],
            "unit {u}: slot {n} is not tile ({i}, {j})"
        );
    }
}

#[test]
#[ignore = "benchmark"]
fn decode_products_against_their_weight_stream() {
    let card = device_index();
    fork_scope(|| {
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::All)
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        let all = s.tiles().len();
        println!(
            "BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"tiles\":{all},\"warmups\":{WARMUPS},\"samples\":{SAMPLES},\"reps\":{REPS},\"window\":{WINDOW},\"route\":\"Tf32FromFp32\",\"fidelity\":\"HiFi4\",\"timing\":\"replay of {REPS} ops less replay of 1, over {}\"}}",
            tt_tests::bench::git_sha(),
            !cfg!(debug_assertions),
            REPS - 1
        );
        let product = |s: &mut Card, x: &DramTensor, w: &DramTensor| {
            s.matmul_dram(
                x,
                false,
                w,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                BUDGET,
            )
            .unwrap()
        };
        for (name, k, n) in [
            ("qkv", 2048usize, 6144usize),
            ("o", 2048, 2048),
            ("gate|up", 2048, 16384),
            ("down", 8192, 2048),
            ("head", 2048, 49152),
            ("135M head", 576, 49152),
        ] {
            let x = s.upload(&floats(1, k), 1, k).unwrap();
            let w = s.upload(&floats(2, k * n), k, n).unwrap();
            let [rt, ct] = w.grid();
            let bytes = (rt * ct) as f64 * TILE_SLOT as f64;
            let reference = product(&mut s, &x, &w);
            let want = s.download(&reference).unwrap();
            s.free(reference).unwrap();
            // The product's deal: how its time moves with the units it runs on.
            for units in [32usize, 48, 64, 80, 96, 104, all] {
                s.set_spread(Spread::Fixed(units));
                let mut last = None;
                let us = op_us(&mut s, |s| last = Some(product(s, &x, &w)));
                let got = s.download(last.as_ref().unwrap()).unwrap();
                assert_eq!(bits(&got), bits(&want), "{name} on {units}: a replay");
                println!(
                    "MEASURE decode-stream {name} [{k}, {n}] product, spread {units}: {us:.1} us, \
                     B {:.0} GB/s",
                    bytes / us / 1e3
                );
            }
            for units in [64usize, all] {
                if name.contains("head") {
                    break;
                }
                for order in [StreamOrder::Columns, StreamOrder::Rows] {
                    let mut tiles = Vec::new();
                    let us = op_us(&mut s, |s| {
                        tiles = s.probe_stream(&w, units, order, WINDOW).unwrap();
                    });
                    let used: Vec<usize> = (0..units).filter(|&u| !tiles[u].is_empty()).collect();
                    for &u in [used[0], *used.last().unwrap()].iter() {
                        landed(&mut s, &w, u, &tiles[u]);
                    }
                    println!(
                        "MEASURE decode-stream {name} [{k}, {n}] B alone {order:?} on {} units: \
                         {us:.1} us, {:.0} GB/s",
                        used.len(),
                        bytes / us / 1e3
                    );
                }
            }
            s.set_spread(Spread::All);
            s.free(w).unwrap();
            s.free(x).unwrap();
        }
    })
    .unwrap_or_else(|e| panic!("{e}"));
}
