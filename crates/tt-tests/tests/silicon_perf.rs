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
                let (written, read) = (v[0].1.bytes_written, v[0].1.bytes_read);
                let med = median(v.into_iter().map(|p| p.0).collect());
                println!(
                    "MEASURE {tiles:>3} tiles {label:<22} {med:>10.2?}  {calls:>6} PCIe calls, {written:>8} B written, {read:>6} B read"
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
                            scalar2: 0.0,
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

/// Element-wise on the SFPU against the data mover's FP32 unit: one op's wall
/// time, by tiles per unit, on 1 and 8 units -- the measurement that decides
/// when a run goes to the SFPU (`tt_kernels::tensor::EltwiseUnit`). Each
/// figure the median of seven after a warm-up; also the PCIe writes per op.
#[test]
#[ignore = "benchmark"]
fn eltwise_unit_sweep() {
    use tt_isa::dm::kind;
    use tt_kernels::session::{Session, TileChoice};
    use tt_kernels::tensor::{Eltwise, EltwiseUnit};
    let card = std::env::var("TT_SILICON_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    for units in [1, 8] {
        if let Err(e) = fork_scope(|| {
            let mut s =
                Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(units))
                    .unwrap_or_else(|e| panic!("{e}"));
            s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
            for per_unit in [1, 2, 4, 8, 16, 32, 64] {
                let tiles = per_unit * units;
                let (r, c) = (32 * tiles, 32);
                let x = s.upload(&pattern_f32(r * c, 3), r, c).unwrap();
                let y = s.upload(&pattern_f32(r * c, 4), r, c).unwrap();
                let mut line = format!("{units} units, {per_unit:>2} tiles/unit:");
                for unit in [EltwiseUnit::Mover, EltwiseUnit::Sfpu] {
                    s.set_eltwise_unit(unit);
                    for (name, k) in [("add", kind::ADD), ("relu", kind::RELU)] {
                        let op = Eltwise {
                            scalar2: 0.0,
                            kind: k,
                            scalar: 0.0,
                        };
                        let other = (k == kind::ADD).then_some(&y);
                        let o = s.eltwise(op, &x, other).unwrap();
                        s.free(o).unwrap();
                        let mut v = Vec::new();
                        let mut writes = 0;
                        for _ in 0..7 {
                            let before = s.device().traffic();
                            let t = Instant::now();
                            let o = s.eltwise(op, &x, other).unwrap();
                            v.push(t.elapsed());
                            writes = (s.device().traffic() - before).bytes_written;
                            s.free(o).unwrap();
                        }
                        let med = median(v);
                        line += &format!(
                            "  {unit:?} {name} {:>7.1} us ({writes} B)",
                            med.as_secs_f64() * 1e6
                        );
                    }
                }
                println!("{line}");
                s.free(x).unwrap();
                s.free(y).unwrap();
            }
        }) {
            panic!("{e}");
        }
    }
}

/// Softmax through Burn on a device-resident `[rows, cols]` tensor: the
/// device composition (max, subtract, exp, sum, divide on the device) against
/// the host's fused softmax after a download, by size, then the result read
/// back either way. Decides when `burn-tt` composes on the device.
#[test]
#[ignore = "benchmark"]
fn softmax_placement_sweep() {
    use burn::tensor::{activation, Tensor, TensorData};
    use burn_tt::TtBackend;
    use tt_tests::burn_device::{with_device, Config};
    let modes: &[bool] = if std::env::var_os("SOFTMAX_DEVICE_ONLY").is_some() {
        &[false]
    } else {
        &[false, true]
    };
    for &exact in modes {
        with_device(
            Config {
                exact,
                ..Config::default()
            },
            |d| {
                let sizes: Vec<(usize, usize)> = match std::env::var("SOFTMAX_SIZE") {
                    Ok(v) => {
                        let (r, c) = v.split_once('x').expect("RxC");
                        vec![(r.parse().unwrap(), c.parse().unwrap())]
                    }
                    Err(_) => vec![
                        (64, 10),
                        (256, 10),
                        (64, 128),
                        (512, 128),
                        (1024, 256),
                        (4096, 512),
                    ],
                };
                for (r, c) in sizes {
                    let v: Vec<f32> = (0..r * c).map(|i| (i % 97) as f32 * 0.01).collect();
                    // Computed on the device, so neither path finds a host copy to
                    // start from: the host path pays the download, as it would
                    // after any device op.
                    let t = Tensor::<TtBackend, 2>::from_data(TensorData::new(v, [r, c]), &d)
                        .to_device(&d)
                        * 1.0;
                    let _ = activation::softmax(t.clone(), 1).into_data();
                    let mut times = Vec::new();
                    for _ in 0..5 {
                        let t0 = Instant::now();
                        let s = activation::softmax(t.clone(), 1);
                        let _ = s.into_data();
                        times.push(t0.elapsed());
                    }
                    println!(
                        "softmax [{r}, {c}] ({} tiles) {}: {:.1} us",
                        r.div_ceil(32) * c.div_ceil(32),
                        if exact { "host" } else { "device" },
                        median(times).as_secs_f64() * 1e6
                    );
                }
            },
        );
    }
}

/// Each op of a device softmax on one `[512, 128]` tensor, alone, through the
/// session: where a composition's time goes.
#[test]
#[ignore = "benchmark"]
fn softmax_parts() {
    use tt_kernels::session::{Session, TileChoice};
    use tt_kernels::sfpu::ops::kind_sfpu;
    use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
    use tt_kernels::tensor::Eltwise;
    let card = std::env::var("TT_SILICON_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if let Err(e) = fork_scope(|| {
        let mut s =
            Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(1)).unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        let (r, c) = (512, 128);
        let x = s.upload(&pattern_f32(r * c, 3), r, c).unwrap();
        let col = s.upload(&pattern_f32(r, 4), r, 1).unwrap();
        let time =
            |label: &str,
             s: &mut Session<tt_kmd::Kmd>,
             f: &dyn Fn(&mut Session<tt_kmd::Kmd>) -> tt_kernels::tensor::DramTensor| {
                let o = f(s);
                s.free(o).unwrap();
                let mut v = Vec::new();
                for _ in 0..5 {
                    let before = s.device().traffic();
                    let t = Instant::now();
                    let o = f(s);
                    v.push((t.elapsed(), (s.device().traffic() - before).bytes_written));
                    s.free(o).unwrap();
                }
                let w = v[0].1;
                println!(
                    "{label:>14}: {:>8.1} us, {w} B written",
                    median(v.into_iter().map(|p| p.0).collect()).as_secs_f64() * 1e6
                );
            };
        let e = |k| Eltwise {
            scalar2: 0.0,
            kind: k,
            scalar: 0.0,
        };
        time("max over cols", &mut s, &|s| {
            s.reduce(&x, ReduceOp::Max, Axis::Cols).unwrap()
        });
        time("sub col", &mut s, &|s| {
            s.eltwise(e(tt_isa::dm::kind::SUB), &x, Some(&col)).unwrap()
        });
        time("exp", &mut s, &|s| {
            s.eltwise(e(kind_sfpu::EXP), &x, None).unwrap()
        });
        time("sum over cols", &mut s, &|s| {
            s.reduce(&x, ReduceOp::Sum, Axis::Cols).unwrap()
        });
        time("div col", &mut s, &|s| {
            s.eltwise(e(kind_sfpu::DIV), &x, Some(&col)).unwrap()
        });
        time("mul (ref)", &mut s, &|s| {
            s.eltwise(e(tt_isa::dm::kind::MUL), &x, Some(&x)).unwrap()
        });
    }) {
        panic!("{e}");
    }
}

/// What one data mover's GDDR -> L1 reads cost, by request shape: the same
/// 1 MiB as 4 KiB requests on one channel and port, rotating the channel's
/// ports, rotating channels, and 16 KiB (the NIU's largest request) and
/// 64 KiB entries. One list each, timed from the host around
/// `submit_list`/`wait` and less an empty list's time; medians of nine. The
/// question it answers: is a matmul gather's ~1 us a tile the NoC and GDDR,
/// or the mover's per-request work?
#[test]
#[ignore = "benchmark"]
fn mover_read_shapes() {
    use tt_isa::dm::op;
    use tt_kernels::dm::DataMover;
    on_card(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tt_tests::harness::tensix_tile();
        let (_, image, _) = tt_firmware_images::DM_B;
        let mut m = DataMover::start(d, &w, t, &dram, image).unwrap();
        const L1: u32 = 0x2_0000;
        const L1_SPAN: u32 = 0x4_0000; // destinations cycle through 256 KiB
        const TOTAL: u32 = 1 << 20;
        let chans: Vec<_> = dram.channels().collect();
        // Each channel's first 1 MiB at 64 MiB holds a known pattern.
        let base = 64u64 << 20;
        let data = pattern(TOTAL as usize, 7);
        for ch in &chans {
            d.dram_write(&w4, ch.range(base, TOTAL as u64).unwrap(), &data)
                .unwrap();
        }
        let entry = |ch: usize, port: u32, off: u32, l1: u32, len: u32| {
            [
                op::READ,
                chans[ch].index() as u32,
                port,
                (base as u32) + off,
                l1,
                len,
                0,
                0,
            ]
        };
        let shape = |len: u32, ch_of: &dyn Fn(u32) -> usize, port_of: &dyn Fn(u32) -> u32| {
            (0..TOTAL / len)
                .map(|i| entry(ch_of(i), port_of(i), i * len, L1 + (i * len) % L1_SPAN, len))
                .collect::<Vec<_>>()
        };
        let time = |d: &mut Dev<'_>, m: &mut DataMover<tt_isa::noc::Noc0>, list: &[[u32; 8]]| {
            let mut v = Vec::new();
            for _ in 0..REPS {
                let t0 = Instant::now();
                m.submit_list(d, &w, list).unwrap();
                m.wait(d, &w).unwrap();
                v.push(t0.elapsed());
            }
            median(v)
        };
        let empty = time(d, &mut m, &[[op::WAIT, 0, 0, 0, 0, 0, 0, 0]]);
        println!("MEASURE mover_read empty list {empty:?}");
        let n = chans.len();
        let cases: Vec<(&str, Vec<[u32; 8]>)> = vec![
            ("4K one port", shape(4096, &|_| 0, &|_| 0)),
            ("4K 3 ports", shape(4096, &|_| 0, &|i| i % 3)),
            ("4K all chans", shape(4096, &|i| i as usize % n, &|_| 0)),
            ("16K one port", shape(16384, &|_| 0, &|_| 0)),
            ("64K one port", shape(65536, &|_| 0, &|_| 0)),
            ("64K all chans", shape(65536, &|i| i as usize % n, &|_| 0)),
        ];
        // The parts of an entry's cost: the loop and decode alone, then the
        // whole path for a request with almost no data.
        let waits = vec![[op::WAIT, 0, 0, 0, 0, 0, 0, 0]; 256];
        let tiny: Vec<[u32; 8]> = (0..256u32)
            .map(|i| entry(0, 0, i * 64, L1 + i * 64, 64))
            .collect();
        for (name, list) in [("256 waits", &waits), ("256 x 64 B", &tiny)] {
            let dt = time(d, &mut m, list).saturating_sub(empty);
            println!(
                "MEASURE mover_read {name:<14} {:>4} entries {:>9.1?} {:>6.2} us/entry",
                list.len(),
                dt,
                dt.as_secs_f64() * 1e6 / list.len() as f64
            );
        }
        for (name, list) in &cases {
            let dt = time(d, &mut m, list).saturating_sub(empty);
            println!(
                "MEASURE mover_read {name:<14} {:>4} entries {:>9.1?} {:>7.0} MB/s {:>6.2} us/4KiB",
                list.len(),
                dt,
                mbps(TOTAL as usize, dt),
                dt.as_secs_f64() * 1e6 / (TOTAL / 4096) as f64
            );
        }
        // The bytes arrived: the last 256 KiB of the 4K-one-port case.
        m.submit_list(d, &w, &cases[0].1).unwrap();
        m.wait(d, &w).unwrap();
        let mut back = vec![0u8; L1_SPAN as usize];
        d.l1_read(&w, t, L1 as u64, &mut back).unwrap();
        let tail = &data[(TOTAL - L1_SPAN) as usize..];
        assert!(back == tail, "the reads did not land");
        m.stop(d, &w).unwrap();
    });
}

/// Is a kernel bound by its runners pushing instructions, or by the backend
/// executing them? Each role's `START -> PUSHED` and `PUSHED -> RETIRED`, in
/// cycles, for a program of NOPs (the runner's push rate: the backend takes a
/// NOP a cycle) and for one matmul tile at LoFi and HiFi4. A role that pushes
/// at the NOP rate and retires right after is push-bound, and fewer pushed
/// words -- `REPLAY`, `MOP` -- make it faster; one that pushes slower than the
/// NOP rate is held back by its FIFO, the backend's pace (X2b).
#[test]
#[ignore = "benchmark"]
fn role_push_rate() {
    use tt_isa::mailbox::trace as ev;
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    let spans = |trace: &[tt_device::trace::TraceEvent]| {
        let mut out = [[0u64; 3]; 3];
        for e in trace {
            let (src, what) = ev::split(e.token);
            if src < 3 && (1..=3).contains(&what) {
                out[src as usize][what as usize - 1] = e.cycles;
            }
        }
        out.map(|[s, p, r]| (p.saturating_sub(s), r.saturating_sub(p)))
    };
    tt_tests::harness::in_device(|dev| {
        let nops = vec![tt_isa::sfpu::nop(); 4000];
        let out = tt_tests::harness::run(
            dev,
            &tt_tests::harness::Run::roles(tt_tests::harness::Roles {
                unpack: &[],
                math: &nops,
                pack: &[],
            })
            .dump_rows(0)
            .traced(),
        );
        let (push, retire) = spans(&out.trace)[1];
        println!(
            "MEASURE role_push nops: {} words, push {push} cycles ({:.2}/word), retire +{retire}",
            nops.len(),
            push as f64 / nops.len() as f64
        );
        let tile = tt_tests::harness::tensix_tile();
        for (name, fidelity) in [("LoFi", Fidelity::Lo), ("HiFi4", Fidelity::HiFi4)] {
            let kt = 8;
            let (m, k, n) = (32, 32 * kt, 32);
            let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32).collect();
            let mut words = [0usize; 3];
            let mut trace = Vec::new();
            tt_kernels::matmul::matmul_with(
                &a,
                &b,
                [m, k, n],
                SrcRoute::Tf32FromFp32,
                fidelity,
                |kern| {
                    for (t, r) in kern.roles.iter().enumerate() {
                        words[t] = r.len();
                    }
                    let traced = tt_kernels::runtime::Kernel {
                        trace: true,
                        ..*kern
                    };
                    let o = tt_kernels::runtime::run(
                        dev,
                        tile,
                        &tt_firmware_images::ROLES,
                        &traced,
                        1 << 30,
                    )?;
                    trace = o.trace.clone();
                    Ok(o)
                },
            )
            .unwrap();
            let s = spans(&trace);
            for (t, role) in ["unpack", "math", "pack"].iter().enumerate() {
                let (push, retire) = s[t];
                println!(
                    "MEASURE role_push matmul {name} 1x{kt}x1 {role}: {} words, push {push} cycles ({:.2}/word), retire +{retire}",
                    words[t],
                    push as f64 / words[t].max(1) as f64
                );
            }
        }
    });
}
