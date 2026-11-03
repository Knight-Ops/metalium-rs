//! The bound on one copy of a relaid weight instead of two (`docs/plans/decode-bytes-in-flight.md`,
//! Phase C0): eager products of more rows over a weight X9 relaid, reading it as it lies against
//! reading it from its regions (`tt_kernels::tensor::probe_relaid_blocks`), in one process over the
//! same weight and the same regions, ABBA.
//!
//! `cargo xtask bench --device 1 --filter silicon_bench_relaid` (every tile). Shapes: the
//! SmolLM2-1.7B's four weights, stored BF16 (`qkv` `[2048, 6144]`, `o` `[2048, 2048]`, `gate|up`
//! `[2048, 16384]`, `down` `[8192, 2048]`), at `m` = 64, 256 and 512 (`TT_RELAID_ROWS` replaces
//! them). Each weight is relaid by one decode capture (`[1, k] @ B`, the layout on), as the
//! generator's step captures relay it.
//!
//! Timed: host dispatch through `sync`, 2 warmups and 7 samples a configuration; validated outside
//! the timing, bit for bit against the first eager product read as it lies.

#![cfg(feature = "silicon")]

use std::time::Instant;

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{
    probe_relaid_audit, probe_relaid_audit_reset, probe_relaid_blocks, probe_relaid_blocks_counts,
};
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
#[ignore = "benchmark: cargo xtask bench --filter silicon_bench_relaid"]
fn eager_products_over_a_relaid_weight() {
    let rows: Vec<usize> = std::env::var("TT_RELAID_ROWS").map_or(vec![64, 256, 512], |v| {
        v.split(',').map(|x| x.trim().parse().unwrap()).collect()
    });
    println!(
        "BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"card\":{},\"tiles\":\"all\",\"warmups\":{WARMUPS},\"samples\":{SAMPLES},\"weights\":\"bf16\",\"timing\":\"host dispatch through sync, ABBA probe off/on/on/off; validation outside timing\"}}",
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
        s.set_weight_layout(true).unwrap();
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
            let b = s.upload_bf16_weight(&half, k, n).unwrap();
            probe_relaid_audit_reset();
            // The decode capture that relays `B`.
            let row = s.upload(&floats(k as u64, k), 1, k).unwrap();
            s.begin_trace().unwrap();
            let decode_out = s
                .matmul_dram(&row, false, &b, false, route, Fidelity::HiFi4, BUDGET)
                .unwrap();
            let decode = s.end_trace().unwrap();
            println!("{name}: {} bytes relaid in all", s.relaid_bytes());
            for &m in &rows {
                let a = s.upload(&floats((m * k) as u64, m * k), m, k).unwrap();
                let run = |s: &mut Session<tt_kmd::Kmd>| {
                    let start = Instant::now();
                    let c = s
                        .matmul_dram(&a, false, &b, false, route, Fidelity::HiFi4, BUDGET)
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
                probe_relaid_blocks(false);
                let want = run(&mut s).1;
                let mut times = [Vec::new(), Vec::new()];
                for (round, on) in [false, true, true, false].into_iter().enumerate() {
                    probe_relaid_blocks(on);
                    for i in 0..WARMUPS + SAMPLES {
                        let (us, got) = run(&mut s);
                        assert_eq!(
                            got, want,
                            "{name} m {m}: probe {on}, round {round}, run {i}"
                        );
                        if i >= WARMUPS {
                            times[usize::from(on)].push(us);
                        }
                    }
                    if on {
                        let [split, entries, regions, lies] = probe_relaid_blocks_counts();
                        println!(
                            "{name} m {m} round {round}: {split} gathers became {entries} entries; \
                             {regions} B bytes from regions, {lies} as they lie"
                        );
                    }
                }
                probe_relaid_blocks(false);
                for (on, t) in times.into_iter().enumerate() {
                    let tag = if on == 1 {
                        "from its regions"
                    } else {
                        "as it lies"
                    };
                    report(
                        &format!("relaid/eager {name} [{m},{k}]@[{k},{n}] BF16, B read {tag}"),
                        "us",
                        "host",
                        Stats::of(t),
                    );
                }
                s.free(a).unwrap();
            }
            let ([blocks, stream], widths) = probe_relaid_audit();
            println!("{name}: coverage {blocks:?} / {stream:?}, run widths {widths:?}");
            s.release_trace(decode).unwrap();
            for t in [row, decode_out, b] {
                s.free(t).unwrap();
            }
        }
    }) {
        panic!("{e}");
    }
}
