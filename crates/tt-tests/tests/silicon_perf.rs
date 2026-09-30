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

/// One MNIST training step's matmuls on a resident `Session`: where the time
/// goes, split into host preparation (planning, program building, tilizing)
/// and each run phase, with the bytes each moved.
#[test]
#[ignore = "benchmark"]
fn mnist_step_breakdown() {
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_kernels::runtime::Phase;
    use tt_kernels::session::{Session, TileChoice};
    let card = std::env::var("TT_SILICON_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if let Err(e) = fork_scope(|| {
        let (x, y) = tt_tests::backend::GATE_TILE;
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Exactly(x, y))
            .unwrap_or_else(|e| panic!("{e}"));
        // Forward: x@W1, h@W2. Backward: g2@W2^T, h^T@g2, x^T@g1.
        let shapes = [
            ("fwd x@W1", [64, 784, 128]),
            ("fwd h@W2", [64, 128, 10]),
            ("bwd g2@W2t", [64, 10, 128]),
            ("bwd ht@g2", [128, 64, 10]),
            ("bwd xt@g1", [784, 64, 128]),
        ];
        let mut total = Duration::ZERO;
        for (label, [m, k, n]) in shapes {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
            let _ = s.matmul(
                &a,
                &b,
                [m, k, n],
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                400_000,
            );
            let _ = s.take_profile();
            let before = s.device().traffic();
            let t0 = Instant::now();
            s.matmul(
                &a,
                &b,
                [m, k, n],
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                400_000,
            )
            .unwrap_or_else(|e| panic!("{e}"));
            let wall = t0.elapsed();
            total += wall;
            let traffic = s.device().traffic() - before;
            let p = s.take_profile();
            let runs = p.phases.iter().filter(|x| x.0 == Phase::Stage).count();
            let mut line = format!(
                "MEASURE step {label:<11} {wall:>9.2?} {runs:>2} runs, {:>8} B out {:>7} B in |",
                traffic.bytes_written, traffic.bytes_read
            );
            let mut device = Duration::ZERO;
            for ph in [
                Phase::Stage,
                Phase::Setup,
                Phase::Programs,
                Phase::Launch,
                Phase::Wait,
                Phase::ReadBack,
            ] {
                let (d, t) = p.of(ph);
                device += d;
                line += &format!(" {ph:?} {d:.2?}/{}B", t.bytes_written + t.bytes_read);
            }
            line += &format!(" | host {:.2?}", wall.saturating_sub(device));
            println!("{line}");
        }
        println!("MEASURE step total {total:.2?}");
    }) {
        panic!("{e}");
    }
}
