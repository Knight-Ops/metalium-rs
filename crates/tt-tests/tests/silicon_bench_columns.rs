//! The eager half of Phase C1's bound (`docs/plans/decode-bytes-in-flight.md`, rule 1): products
//! over a weight laid by column (`Session::set_column_weights`) against the same weight
//! interleaved, in one process, ABBA (interleaved / by column / by column / interleaved).
//!
//! `cargo xtask bench --device 1 --filter silicon_bench_columns` (every tile). Shapes: the
//! SmolLM2-1.7B's four weights, stored BF16 (`qkv` `[2048, 6144]`, `o` `[2048, 2048]`, `gate|up`
//! `[2048, 16384]`, `down` `[8192, 2048]`), at `m` = 1 (the decode product, eager), 64, 256 and
//! 512 (`TT_COLUMNS_ROWS` replaces them). Each layout is its own weight with the same values; the
//! GDDR each took is printed beside its interleaved twin's.
//!
//! Timed: host dispatch through `sync`, 2 warmups and 7 samples a configuration; validated outside
//! the timing, bit for bit against the first interleaved product.

#![cfg(feature = "silicon")]

use std::time::Instant;

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Layout};
use tt_tests::bench::{report, Stats};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const WARMUPS: usize = 2;
const SAMPLES: usize = 7;

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

#[test]
#[ignore = "benchmark: cargo xtask bench --filter silicon_bench_columns"]
fn eager_products_over_a_weight_laid_by_column() {
    let rows: Vec<usize> = std::env::var("TT_COLUMNS_ROWS").map_or(vec![1, 64, 256, 512], |v| {
        v.split(',').map(|x| x.trim().parse().unwrap()).collect()
    });
    println!(
        "BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"card\":{},\"tiles\":\"all\",\"warmups\":{WARMUPS},\"samples\":{SAMPLES},\"weights\":\"bf16\",\"timing\":\"host dispatch through sync, ABBA interleaved/by column/by column/interleaved; validation outside timing\"}}",
        tt_tests::bench::git_sha(),
        !cfg!(debug_assertions),
        tt_tests::backend::device_index()
    );
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::All,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        let route = SrcRoute::TF32_WITH_BF16_B;
        for (name, k, n) in [
            ("qkv", 2048, 6144),
            ("o", 2048, 2048),
            ("gate|up", 2048, 16384),
            ("down", 8192, 2048),
        ] {
            let half: Vec<u16> = floats(n as u64, k * n)
                .iter()
                .map(|x| (x.to_bits() >> 16) as u16)
                .collect();
            let store = |s: &mut Session<tt_kmd::Kmd>, by_column: bool| {
                s.set_column_weights(by_column).unwrap();
                let before = s.dram_free_bytes();
                let w = s.upload_bf16_weight(&half, k, n).unwrap();
                s.set_column_weights(false).unwrap();
                (w, before - s.dram_free_bytes())
            };
            let (flat, flat_bytes) = store(&mut s, false);
            let (cols, cols_bytes) = store(&mut s, true);
            assert_eq!(cols.placement.layout(), Layout::ByColumn);
            for (tag, bytes) in [("interleaved", flat_bytes), ("by column", cols_bytes)] {
                report(
                    &format!("columns/gddr {name} [{k},{n}] BF16, B {tag}"),
                    "bytes",
                    "allocator",
                    Stats::of(vec![bytes as f64]),
                );
            }
            let weights: [&DramTensor; 2] = [&flat, &cols];
            for &m in &rows {
                let a = s.upload(&floats((m * k) as u64, m * k), m, k).unwrap();
                let run = |s: &mut Session<tt_kmd::Kmd>, b: &DramTensor| {
                    let start = Instant::now();
                    let c = s
                        .matmul_dram(&a, false, b, false, route, Fidelity::HiFi4, BUDGET)
                        .unwrap();
                    s.sync().unwrap();
                    let us = start.elapsed().as_secs_f64() * 1e6;
                    let got: Vec<u32> = s
                        .download(&c)
                        .unwrap()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect();
                    s.free(c).unwrap();
                    (us, got)
                };
                let want = run(&mut s, &flat).1;
                let mut times = [Vec::new(), Vec::new()];
                for (round, which) in [0usize, 1, 1, 0].into_iter().enumerate() {
                    for i in 0..WARMUPS + SAMPLES {
                        let (us, got) = run(&mut s, weights[which]);
                        assert_eq!(
                            got, want,
                            "{name} m {m}: layout {which}, round {round}, run {i}"
                        );
                        if i >= WARMUPS {
                            times[which].push(us);
                        }
                    }
                }
                for (which, t) in times.into_iter().enumerate() {
                    let tag = ["interleaved", "by column"][which];
                    report(
                        &format!("columns/eager {name} [{m},{k}]@[{k},{n}] BF16, B {tag}"),
                        "us",
                        "host",
                        Stats::of(t),
                    );
                }
                s.free(a).unwrap();
            }
            s.free(flat).unwrap();
            s.free(cols).unwrap();
        }
    }) {
        panic!("{e}");
    }
}
