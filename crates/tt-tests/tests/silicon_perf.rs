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

/// The DRAM-resident matmuls of one MNIST step, split into the mover's gather,
/// the Tensix run and the mover's scatter.
#[test]
#[ignore = "benchmark"]
fn dram_matmul_breakdown() {
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_kernels::session::{Session, TileChoice};
    let card = std::env::var("TT_SILICON_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if let Err(e) = fork_scope(|| {
        let (x, y) = tt_tests::backend::GATE_TILE;
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Exactly(x, y))
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        let cases = [
            ("x@W1", [64, 784], false, [784, 128], false),
            ("h@W2", [64, 128], false, [128, 10], false),
            ("g2@W2t", [64, 10], false, [128, 10], true),
            ("ht@g2", [64, 128], true, [64, 10], false),
            ("xt@g1", [64, 784], true, [64, 128], false),
        ];
        for (label, [ar, ac], ta, [br, bc], tb) in cases {
            let a = s.upload(&vec![0.5; ar * ac], ar, ac).unwrap();
            let b = s.upload(&vec![0.25; br * bc], br, bc).unwrap();
            let c = s
                .matmul_dram(
                    &a,
                    ta,
                    &b,
                    tb,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    400_000,
                )
                .unwrap();
            s.free(c).unwrap();
            let _ = tt_kernels::tensor::stats::take();
            let _ = s.take_profile();
            let t0 = Instant::now();
            let c = s
                .matmul_dram(
                    &a,
                    ta,
                    &b,
                    tb,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    400_000,
                )
                .unwrap();
            let wall = t0.elapsed();
            let mut line = format!("MEASURE dram {label:<7} {wall:>9.2?} |");
            for (k, n, d) in tt_kernels::tensor::stats::take() {
                line += &format!(" {k} {n}x {d:.2?};");
            }
            let p = s.take_profile();
            for ph in [
                tt_kernels::runtime::Phase::Setup,
                tt_kernels::runtime::Phase::Programs,
                tt_kernels::runtime::Phase::Launch,
                tt_kernels::runtime::Phase::Wait,
                tt_kernels::runtime::Phase::ReadBack,
            ] {
                let (d, t) = p.of(ph);
                line += &format!(" {ph:?} {d:.2?}/{}w{}r;", t.write_calls, t.read_calls);
            }
            println!("{line}");
            for t in [a, b, c] {
                s.free(t).unwrap();
            }
        }
    }) {
        panic!("{e}");
    }
}

/// Phase 9.6: how GDDR ops scale with the number of Tensix tiles a session
/// deals them over. A matmul bigger than MNIST's (`[512, 512] @ [512, 512]`,
/// 256 output tiles), an element-wise add over 2048 tiles and a column sum
/// over 32 columns, each the median of five after a warm-up, at every tile
/// count from 1 to the whole chip. Also how long opening the session took,
/// since every tile is reset and has its roles loaded in turn.
#[test]
#[ignore = "benchmark"]
fn many_tiles_sweep() {
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_kernels::session::{Session, TileChoice};
    use tt_kernels::tensor::Eltwise;
    let card = std::env::var("TT_SILICON_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    for choice in [1, 2, 4, 8, 16, 32, 64]
        .map(TileChoice::Count)
        .into_iter()
        .chain([TileChoice::All])
    {
        if let Err(e) = fork_scope(|| {
            let t0 = Instant::now();
            let mut s = Session::open_card(card, tt_firmware_images::ROLES, choice)
                .unwrap_or_else(|e| panic!("{e}"));
            s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
            let open = t0.elapsed();
            let tiles = s.tiles().len();
            let n = 512;
            let a = s.upload(&pattern_f32(n * n, 1), n, n).unwrap();
            let b = s.upload(&pattern_f32(n * n, 2), n, n).unwrap();
            let (er, ec) = (2048, 1024);
            let x = s.upload(&pattern_f32(er * ec, 3), er, ec).unwrap();
            let y = s.upload(&pattern_f32(er * ec, 4), er, ec).unwrap();
            let time = |label: &str,
                        s: &mut Session<tt_kmd::Kmd>,
                        op: &mut dyn FnMut(&mut Session<tt_kmd::Kmd>)| {
                op(s);
                let mut v = Vec::new();
                for _ in 0..5 {
                    let before = s.device().traffic();
                    let t = Instant::now();
                    op(s);
                    v.push((t.elapsed(), s.device().traffic() - before));
                }
                let calls = v[0].1.write_calls + v[0].1.read_calls;
                let med = median(v.into_iter().map(|p| p.0).collect());
                println!(
                    "MEASURE {tiles:>3} tiles {label:<22} {med:>10.2?}  {calls:>6} PCIe calls"
                );
            };
            time("matmul 512^3 HiFi4", &mut s, &mut |s| {
                let c = s
                    .matmul_dram(
                        &a,
                        false,
                        &b,
                        false,
                        SrcRoute::Tf32FromFp32,
                        Fidelity::HiFi4,
                        400_000,
                    )
                    .unwrap_or_else(|e| panic!("{e}"));
                s.free(c).unwrap();
            });
            time("add [2048, 1024]", &mut s, &mut |s| {
                let o = s
                    .eltwise(
                        Eltwise {
                            kind: tt_isa::dm::kind::ADD,
                            scalar: 0.0,
                        },
                        &x,
                        Some(&y),
                    )
                    .unwrap_or_else(|e| panic!("{e}"));
                s.free(o).unwrap();
            });
            time("sum rows [2048, 1024]", &mut s, &mut |s| {
                let o = s.sum_rows(&x).unwrap_or_else(|e| panic!("{e}"));
                s.free(o).unwrap();
            });
            println!("MEASURE {tiles:>3} tiles open                   {open:>10.2?}");
        }) {
            panic!("{choice:?}: {e}");
        }
    }
}

fn pattern_f32(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 2001) as f32 / 1000.0 - 1.0
        })
        .collect()
}

/// Where opening a session's time goes, per tile: the extra driver file
/// descriptor that holds the tile's cleanup write, the tile reset, the
/// per-thread state reset and the resident roles' start.
#[test]
#[ignore = "benchmark"]
fn session_open_breakdown() {
    use tt_kernels::runtime::Resident;
    use tt_kernels::session::{reset_thread_state, reset_tile};
    let card = std::env::var("TT_SILICON_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if let Err(e) = fork_scope(|| {
        let t = Instant::now();
        let extra: Vec<_> = (0..4).map(|_| tt_kmd::Kmd::open(card).unwrap()).collect();
        println!("MEASURE open: Kmd::open {:>10.2?} each", t.elapsed() / 4);
        drop(extra);
        let (x, y) = tt_tests::backend::GATE_TILE;
        let t = Instant::now();
        let held: Vec<_> = (0..4)
            .map(|_| {
                tt_kmd::CleanupWrite::register(
                    card,
                    x,
                    y,
                    0,
                    tt_isa::tensix::SOFT_RESET_0,
                    tt_kernels::session::ALL_BABIES_HELD,
                )
                .unwrap()
            })
            .collect();
        println!(
            "MEASURE open: CleanupWrite::register {:>10.2?} each",
            t.elapsed() / 4
        );
        drop(held);
        let mut dev = open_card(card);
        let images = tt_firmware_images::ROLES;
        let tile = tile(
            &mut dev,
            tt_tests::backend::GATE_TILE.0,
            tt_tests::backend::GATE_TILE.1,
        );
        for _ in 0..3 {
            let t = Instant::now();
            reset_tile(&mut dev, tile).unwrap();
            let a = t.elapsed();
            reset_thread_state(&mut dev, tile, &images).unwrap();
            let b = t.elapsed();
            let r = Resident::start(&mut dev, tile, &images, 400_000).unwrap();
            let c = t.elapsed();
            r.stop(&mut dev, &images).unwrap();
            println!(
                "MEASURE open: reset_tile {a:>9.2?}  reset_thread_state {:>9.2?}  Resident::start {:>9.2?}",
                b - a,
                c - b
            );
        }
        scrub(&mut dev);
    }) {
        panic!("{e}");
    }
}
