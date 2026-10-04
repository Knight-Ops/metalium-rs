//! How fast the card moves host memory itself (`tt_isa::dm::op::HOST_READ`,
//! `HOST_WRITE`), against the host's own copies through a BAR, which this
//! machine's VM passthrough leaves uncached (`docs/ttsim-divergence.md` row
//! M): 1 to 32 tiles each streaming 16 KiB requests between their L1 and
//! their own part of one pinned 1 GiB buffer, host-timed from the first
//! enqueue to the last list's end.
//!
//! `cargo xtask bench --filter silicon_bench_host_dma`

#![cfg(feature = "silicon")]

use std::time::Instant;

use tt_device::tlb::WindowKind;
use tt_device::Transport;
use tt_isa::dm::op;
use tt_isa::noc::Noc0;
use tt_kernels::dm::DataMover;
use tt_tests::backend::tensix_grid;
use tt_tests::bench::{on_card, report_rate, Stats, REPS};
use tt_tests::harness::tile;

const L1_AT: u32 = 0x2_0000;
const ENTRY: u32 = 16 << 10;
/// Entries a list: 512 KiB of L1.
const ENTRIES: u32 = 32;
/// Lists a tile a run: 2 MiB a tile.
const LISTS: usize = 4;
/// Each tile's share of the host buffer.
const SHARE: u64 = (ENTRY * ENTRIES) as u64 * LISTS as u64;

#[test]
#[ignore = "benchmark"]
fn host_dma_bandwidth() {
    on_card(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let grid = tensix_grid(d);
        let mut host = d.transport().host_memory(1 << 30).unwrap();
        let base = host.noc_address();
        println!("MEASURE host memory: 1 GiB at NoC address {base:#x}");
        let coords: Vec<_> = grid.tiles::<Noc0>().take(32).collect();
        let mut movers: Vec<DataMover<Noc0>> = coords
            .iter()
            .map(|c| {
                let t = tile(d, c.x(), c.y());
                DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap()
            })
            .collect();
        // Every tile's share a known pattern, for the integrity check.
        let pattern: Vec<u8> = (0..SHARE as usize * 32)
            .map(|i| (i * 7 + i / 4096) as u8)
            .collect();
        host.write(0, &pattern);
        for n in [1usize, 2, 4, 8, 16, 32] {
            for (name, kind) in [("read", op::HOST_READ), ("write", op::HOST_WRITE)] {
                let lists: Vec<Vec<Vec<[u32; 8]>>> = (0..n)
                    .map(|u| {
                        (0..LISTS)
                            .map(|l| {
                                (0..ENTRIES)
                                    .map(|e| {
                                        let host = base
                                            + u as u64 * SHARE
                                            + (l as u64 * ENTRIES as u64 + e as u64) * ENTRY as u64;
                                        [
                                            kind,
                                            host as u32,
                                            (host >> 32) as u32,
                                            0,
                                            L1_AT + e * ENTRY,
                                            ENTRY,
                                            0,
                                            0,
                                        ]
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                let mut run = || {
                    let t0 = Instant::now();
                    let mut last = vec![0u32; n];
                    // A list on every tile in turn, so all of them stream at
                    // once.
                    for l in 0..LISTS {
                        for ((m, list), at) in movers.iter_mut().zip(&lists).zip(&mut last) {
                            *at = m.enqueue(d, &w, &list[l]).unwrap();
                        }
                    }
                    for (m, &at) in movers.iter_mut().zip(&last) {
                        m.wait_for(d, &w, at).unwrap();
                    }
                    t0.elapsed().as_secs_f64() * 1e6
                };
                run();
                let us = Stats::of((0..REPS).map(|_| run()));
                report_rate(
                    &format!("host dma {name} {n:>2} tiles"),
                    "host",
                    (n as u64 * SHARE) as f64,
                    us,
                    None,
                );
                if kind == op::HOST_READ {
                    // The last list's 512 KiB, in tile 0's L1.
                    let mut back = vec![0u8; (ENTRY * ENTRIES) as usize];
                    d.l1_read(&w, movers[0].tile(), L1_AT as u64, &mut back)
                        .unwrap();
                    let from = ((LISTS - 1) as u64 * (ENTRY * ENTRIES) as u64) as usize;
                    assert!(
                        back == pattern[from..from + back.len()],
                        "{n} tiles: read corrupt"
                    );
                }
            }
        }
        // The host's own copies through the BAR, for the comparison.
        let t = movers[0].tile();
        let data = vec![0x5Au8; 2 << 20];
        let us = Stats::of((0..REPS).map(|_| {
            let t0 = Instant::now();
            d.l1_write(&w, t, L1_AT as u64, &data[..512 << 10]).unwrap();
            t0.elapsed().as_secs_f64() * 1e6
        }));
        report_rate(
            "host BAR write 1 tile",
            "host",
            (512 << 10) as f64,
            us,
            None,
        );
        let us = Stats::of((0..REPS).map(|_| {
            let mut back = vec![0u8; 512 << 10];
            let t0 = Instant::now();
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            t0.elapsed().as_secs_f64() * 1e6
        }));
        report_rate("host BAR read 1 tile", "host", (512 << 10) as f64, us, None);
        for m in movers {
            m.stop(d, &w).unwrap();
        }
    });
}

/// Where a `Session` upload's and download's time goes, by size, with the
/// card's DMA (`Session::set_host_dma`) and through the BAR.
#[test]
#[ignore = "benchmark"]
fn session_transfers() {
    use tt_kernels::session::{Session, TileChoice, Tilize};
    let card = tt_tests::backend::device_index();
    for tiles in [1usize, 8, 32] {
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(tiles))
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        for (rows, cols) in [(64usize, 10usize), (64, 784), (1024, 1024), (8192, 1024)] {
            let v: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
            for (how, dma, at) in [
                ("card", true, Tilize::Card),
                ("host", true, Tilize::Host),
                ("bar", false, Tilize::Host),
            ] {
                s.set_host_dma(dma);
                s.set_tilize(at);
                let t = s.upload(&v, rows, cols).unwrap();
                s.free(t).unwrap();
                // An upload is queued: timed to its end on the card.
                let up = Stats::of((0..REPS).map(|_| {
                    let t0 = Instant::now();
                    let t = s.upload(&v, rows, cols).unwrap();
                    s.sync().unwrap();
                    let us = t0.elapsed().as_secs_f64() * 1e6;
                    s.free(t).unwrap();
                    us
                }));
                let t = s.upload(&v, rows, cols).unwrap();
                let down = Stats::of((0..REPS).map(|_| {
                    let t0 = Instant::now();
                    let back = s.download(&t).unwrap();
                    let us = t0.elapsed().as_secs_f64() * 1e6;
                    assert!(back == v, "{how}: {rows}x{cols} did not round-trip");
                    us
                }));
                s.free(t).unwrap();
                let bytes = (rows * cols * 4) as f64;
                report_rate(
                    &format!("session upload {rows}x{cols} {tiles} tiles {how}"),
                    "host",
                    bytes,
                    up,
                    None,
                );
                report_rate(
                    &format!("session download {rows}x{cols} {tiles} tiles {how}"),
                    "host",
                    bytes,
                    down,
                    None,
                );
            }
            s.set_host_dma(true);
            s.set_tilize(Tilize::Host);
        }
    }
}

/// A training-shaped loop that streams its inputs: each step uploads a batch
/// and multiplies it by resident weights, nothing synced until the end. With
/// uploads queued behind the ops (the card's DMA, `Session::set_host_dma`),
/// the host tilizes and queues the next batch while the card computes; with
/// a sync after each upload (what an upload did before it was queued), the
/// two take turns; through the BAR, the host's stores are the step.
#[test]
#[ignore = "benchmark"]
fn streamed_steps() {
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_kernels::session::{Session, TileChoice};
    let card = tt_tests::backend::device_index();
    const STEPS: usize = 100;
    for tiles in [1usize, 8] {
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(tiles))
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        for (batch, input, hidden) in [(64usize, 784usize, 128usize), (512, 1024, 1024)] {
            let w: Vec<f32> = (0..input * hidden).map(|i| (i % 7) as f32 * 0.01).collect();
            let w = s.upload(&w, input, hidden).unwrap();
            let x: Vec<f32> = (0..batch * input).map(|i| (i % 13) as f32 * 0.1).collect();
            for (how, dma, sync_each) in [
                ("queued", true, false),
                ("synced", true, true),
                ("bar", false, false),
            ] {
                s.set_host_dma(dma);
                let mut run = || {
                    let t0 = Instant::now();
                    for _ in 0..STEPS {
                        let xt = s.upload(&x, batch, input).unwrap();
                        if sync_each {
                            s.sync().unwrap();
                        }
                        let y = s
                            .matmul_dram(
                                &xt,
                                false,
                                &w,
                                false,
                                SrcRoute::Tf32FromFp32,
                                Fidelity::HiFi4,
                                1 << 30,
                            )
                            .unwrap();
                        s.free(xt).unwrap();
                        s.free(y).unwrap();
                    }
                    s.sync().unwrap();
                    t0.elapsed().as_secs_f64() * 1e6 / STEPS as f64
                };
                run();
                let us = Stats::of((0..3).map(|_| run()));
                println!(
                    "MEASURE streamed {batch}x{input}x{hidden} on {tiles} tiles, {how}: {:.1} us/step (p10 {:.1}, p90 {:.1})",
                    us.median, us.p10, us.p90
                );
            }
            s.set_host_dma(true);
            s.free(w).unwrap();
        }
    }
}
