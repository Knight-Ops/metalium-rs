//! Phase 9 benchmarks: every performance number in the docs comes from here.
//!
//! `cargo xtask silicon --release --include-ignored --filter silicon_perf`
//!
//! Each prints `MEASURE` lines and asserts only that the data arrived intact:
//! a benchmark that moves the wrong bytes quickly measures nothing.

#![cfg(feature = "silicon")]

use std::time::{Duration, Instant};

use tt_device::tlb::WindowKind;
use tt_tests::backend::{open_card, scrub};
use tt_tests::harness::{tile, Dev};
use tt_ttsim::fork_scope;

const REPS: usize = 9;

fn mbps(bytes: usize, t: Duration) -> f64 {
    bytes as f64 / t.as_secs_f64() / 1e6
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as u8
        })
        .collect()
}

fn on_card(f: impl FnOnce(&mut Dev<'_>)) {
    let card = std::env::var("TT_SILICON_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if let Err(e) = fork_scope(|| {
        let mut d = open_card(card);
        f(&mut d);
        scrub(&mut d);
    }) {
        panic!("{e}");
    }
}

/// Time `write` then a read-back fence, and `read`, over `REPS`; check the data.
fn bench(
    d: &mut Dev<'_>,
    label: &str,
    size: usize,
    mut write: impl FnMut(&mut Dev<'_>, &[u8]),
    mut fence: impl FnMut(&mut Dev<'_>),
    mut read: impl FnMut(&mut Dev<'_>, &mut [u8]),
) {
    let data = pattern(size, size as u32);
    let mut back = vec![0u8; size];
    let (mut w, mut r) = (Vec::new(), Vec::new());
    for _ in 0..REPS {
        let t0 = Instant::now();
        write(d, &data);
        fence(d);
        w.push(t0.elapsed());
        let t0 = Instant::now();
        read(d, &mut back);
        r.push(t0.elapsed());
        assert!(back == data, "{label} {size}: data did not round-trip");
        back.fill(0);
    }
    let (w, r) = (median(w), median(r));
    println!(
        "MEASURE {label:<10} {:>6} KiB: write {:>9.1?} {:>7.0} MB/s | read {:>9.1?} {:>7.0} MB/s",
        size >> 10,
        w,
        mbps(size, w),
        r,
        mbps(size, r)
    );
}

/// Host <-> Tensix L1: the UC dword path against the bulk (WC) path.
#[test]
#[ignore = "benchmark"]
fn pcie_l1() {
    on_card(|d| {
        println!(
            "MEASURE wc mappings: bar0 {} bar4 {}",
            d.transport().has_wc(tt_device::Bar::Bar0),
            d.transport().has_wc(tt_device::Bar::Bar4)
        );
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let t = tile(d, 4, 4);
        const AT: u64 = 0x8_0000;
        for size in [4 << 10, 64 << 10, 512 << 10] {
            bench(
                d,
                "l1 uc",
                size,
                |d, data| d.write(&w, t, AT, data).unwrap(),
                |d| {
                    let _ = d.read32(&w, t, AT).unwrap();
                },
                |d, out| d.read(&w, t, AT, out).unwrap(),
            );
            bench(
                d,
                "l1 bulk",
                size,
                |d, data| d.l1_write(&w, t, AT, data).unwrap(),
                |d| {
                    let _ = d.read32(&w, t, AT).unwrap();
                },
                |d, out| d.l1_read(&w, t, AT, out).unwrap(),
            );
        }
    });
}

/// Host <-> GDDR through a 4 GiB window, bulk path: startup population speed.
#[test]
#[ignore = "benchmark"]
fn pcie_dram() {
    on_card(|d| {
        let w2 = d.alloc_window(WindowKind::TwoMib).unwrap();
        let ch = d.dram_grid(&w2).unwrap().channel(0).unwrap();
        let w = d.alloc_window(WindowKind::FourGib).unwrap();
        for size in [64 << 10, 1 << 20, 16 << 20] {
            let r = ch.range(0x100_0000, size as u64).unwrap();
            let last = ch.range(0x100_0000 + size as u64 - 4, 4).unwrap();
            bench(
                d,
                "dram bulk",
                size,
                |d, data| d.dram_write(&w, r, data).unwrap(),
                |d| d.dram_read(&w, last, &mut [0; 4]).unwrap(),
                |d, out| d.dram_read(&w, r, out).unwrap(),
            );
        }
    });
}
