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
use tt_tests::bench::{mbps, median, on_card, pattern, REPS};
use tt_tests::harness::{tile, Dev};
use tt_ttsim::fork_scope;

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
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
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
            s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
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
                            kind: tt_kernels::kind::ADD,
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

/// The SFPU's transcendentals by cost: one op's wall time on one unit, by
/// tiles, and the slope -- what a tile of each program costs on the card
/// (10.2f: the trig reduction against `exp`). Each figure the median of
/// seven after a warm-up.
#[test]
#[ignore = "benchmark"]
fn sfpu_transcendental_cost() {
    use tt_kernels::session::{Session, TileChoice};
    use tt_kernels::sfpu::ops::kind_sfpu::*;
    use tt_kernels::tensor::Eltwise;
    let card = std::env::var("TT_SILICON_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let kinds = [
        ("exp", EXP),
        ("tanh", TANH),
        ("gelu", GELU),
        ("sin", SIN),
        ("cos", COS),
        ("tan", TAN),
        ("atan", ATAN),
        ("asin", ASIN),
        ("acos", ACOS),
    ];
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(1))
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        for (name, kind) in kinds {
            let op = Eltwise {
                kind,
                scalar: 0.0,
                scalar2: 0.0,
            };
            let mut at = Vec::new();
            for tiles in [1usize, 2, 4, 8, 16, 32, 64] {
                let (r, c) = (32 * tiles, 32);
                let x = s.upload(&pattern_f32(r * c, 3), r, c).unwrap();
                let o = s.eltwise(op, &x, None).unwrap();
                s.sync().unwrap();
                s.free(o).unwrap();
                let mut v = Vec::new();
                let mut writes = 0;
                for _ in 0..7 {
                    let before = s.device().traffic();
                    let t = Instant::now();
                    let o = s.eltwise(op, &x, None).unwrap();
                    // The op is queued (X4): its time is to its completion.
                    s.sync().unwrap();
                    v.push(t.elapsed());
                    writes = (s.device().traffic() - before).bytes_written;
                    s.free(o).unwrap();
                }
                s.free(x).unwrap();
                at.push((tiles, median(v).as_secs_f64() * 1e6, writes));
            }
            let line: Vec<String> = at
                .iter()
                .map(|(t, us, w)| format!("{t}: {us:.1} us ({w} B)"))
                .collect();
            println!("MEASURE sfpu {name:>5}: {}", line.join(", "));
        }
    }) {
        panic!("{e}");
    }
}

/// Native resident softmax with readback, compared with an external Flex tensor.
#[test]
#[ignore = "benchmark"]
fn softmax_placement_sweep() {
    use burn::tensor::{activation, Tensor, TensorData};
    use burn_flex::{Flex, FlexDevice};
    use burn_tt::TtBackend;
    use tt_tests::burn_device::{with_device, Config};
    let device_only = std::env::var_os("SOFTMAX_DEVICE_ONLY").is_some();
    with_device(Config::default(), |d| {
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
            let data = TensorData::new(
                (0..r * c)
                    .map(|i| (i % 97) as f32 * 0.01)
                    .collect::<Vec<_>>(),
                [r, c],
            );
            let t = Tensor::<TtBackend, 2>::from_data(data.clone(), &d).to_device(&d);
            let _ = activation::softmax(t.clone(), 1).into_data();
            let mut times = Vec::new();
            for _ in 0..5 {
                let t0 = Instant::now();
                let _ = activation::softmax(t.clone(), 1).into_data();
                times.push(t0.elapsed());
            }
            println!(
                "softmax [{r}, {c}] ({} tiles) native: {:.1} us",
                r.div_ceil(32) * c.div_ceil(32),
                median(times).as_secs_f64() * 1e6
            );
            if !device_only {
                let reference = Tensor::<Flex, 2>::from_data(data, &FlexDevice);
                let _ = activation::softmax(reference.clone(), 1).into_data();
                let mut times = Vec::new();
                for _ in 0..5 {
                    let t0 = Instant::now();
                    let _ = activation::softmax(reference.clone(), 1).into_data();
                    times.push(t0.elapsed());
                }
                println!(
                    "softmax [{r}, {c}] external Flex: {:.1} us",
                    median(times).as_secs_f64() * 1e6
                );
            }
        }
    });
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
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
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
            s.eltwise(e(tt_kernels::kind::SUB), &x, Some(&col)).unwrap()
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
            s.eltwise(e(tt_kernels::kind::MUL), &x, Some(&x)).unwrap()
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

/// X6: the mover's fast read path (`dm::READ_FAST`, request initiator 1 of NoC
/// #0 writing only the words that change) against the path that writes every
/// register of initiator 0, by the shapes of `mover_read_shapes`, plus a
/// 2 KiB-entry gather. Each case is run on both paths in alternation (ABAB), after
/// three untimed warm-up lists on each, and timed as `mover_read_shapes` times:
/// host clock around `submit_list`/`wait`, less an empty list's time, median of
/// `REPS`. Every timed list's bytes are checked against the host's copy of the
/// channels after the last repetition, so a path that is fast because it moves
/// the wrong bytes fails instead of winning.
///
/// Run only after `step118_mover_fast_path`'s silicon persistence probes
/// have passed on the card: it writes request initiator 1, which no code
/// here has driven on silicon before.
#[test]
#[ignore = "benchmark"]
fn mover_read_fast_path() {
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
        const L1_SPAN: u32 = 0x4_0000;
        const TOTAL: u32 = 1 << 20;
        let chans: Vec<_> = dram.channels().collect();
        let base = 64u64 << 20;
        let datas: Vec<Vec<u8>> = chans
            .iter()
            .map(|ch| {
                let data = pattern(TOTAL as usize, 7 + ch.index() as u32);
                d.dram_write(&w4, ch.range(base, TOTAL as u64).unwrap(), &data)
                    .unwrap();
                data
            })
            .collect();
        let n = chans.len();
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
            let t0 = Instant::now();
            m.submit_list(d, &w, list).unwrap();
            m.wait(d, &w).unwrap();
            t0.elapsed()
        };
        m.set_read_fast(d, &w, false).unwrap();
        let empty = median(
            (0..REPS)
                .map(|_| time(d, &mut m, &[[op::WAIT, 0, 0, 0, 0, 0, 0, 0]]))
                .collect(),
        );
        println!("MEASURE mover_read_fast empty list {empty:?}");
        let cases: Vec<(&str, Vec<[u32; 8]>)> = vec![
            (
                "64 B x 256",
                (0..256u32)
                    .map(|i| entry(0, 0, i * 64, L1 + i * 64, 64))
                    .collect(),
            ),
            ("2 KiB one port", shape(2048, &|_| 0, &|_| 0)),
            ("2 KiB all chans", shape(2048, &|i| i as usize % n, &|_| 0)),
            ("4 KiB one port", shape(4096, &|_| 0, &|_| 0)),
            ("4 KiB 3 ports", shape(4096, &|_| 0, &|i| i % 3)),
            ("4 KiB all chans", shape(4096, &|i| i as usize % n, &|_| 0)),
            ("16 KiB one port", shape(16384, &|_| 0, &|_| 0)),
        ];
        for (name, list) in &cases {
            let mut slow = vec![];
            let mut fast = vec![];
            for path in [false, true] {
                m.set_read_fast(d, &w, path).unwrap();
                for _ in 0..3 {
                    time(d, &mut m, list);
                }
            }
            for _ in 0..REPS {
                for path in [false, true] {
                    m.set_read_fast(d, &w, path).unwrap();
                    let dt = time(d, &mut m, list).saturating_sub(empty);
                    if path { &mut fast } else { &mut slow }.push(dt);
                }
            }
            // The bytes of the last list's last L1_SPAN (the later entries overwrite
            // the earlier in the cycling destination), from the host's copy.
            let mut back = vec![0u8; L1_SPAN as usize];
            d.l1_read(&w, t, L1 as u64, &mut back).unwrap();
            let landed = list.iter().rev().take(24).all(|e| {
                let (ch, off, at, len) = (
                    chans.iter().position(|c| c.index() as u32 == e[1]).unwrap(),
                    e[3] - base as u32,
                    e[4],
                    e[5] as usize,
                );
                let tail = &back[(at - L1) as usize..][..len.min((L1 + L1_SPAN - at) as usize)];
                tail == &datas[ch][off as usize..][..tail.len()]
            });
            assert!(landed, "{name}: the fast path's reads did not land");
            let (slow, fast) = (median(slow), median(fast));
            println!(
                "MEASURE mover_read_fast {name:<16} {:>4} entries slow {:>9.1?} {:>6.3} us/entry, fast {:>9.1?} {:>6.3} us/entry, fast/slow {:.3}",
                list.len(),
                slow,
                slow.as_secs_f64() * 1e6 / list.len() as f64,
                fast,
                fast.as_secs_f64() * 1e6 / list.len() as f64,
                fast.as_secs_f64() / slow.as_secs_f64()
            );
        }
        m.set_read_fast(d, &w, false).unwrap();
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

/// What `MathMode::Approx` saves per tile (S10; lane T8): each transcendental
/// with an Approx twin, in both modes, one op's wall time on one unit by tiles
/// and the slope, the median of seven after a warm-up -- beside the
/// instruction counts `step145_approx_math` prints. The data is checked: each
/// mode's output is the interpreter's program for the kind it runs, bit for
/// bit, so a number is never for the wrong program.
#[test]
#[ignore = "benchmark"]
fn approx_per_tile_saving() {
    use tt_kernels::session::{Session, TileChoice};
    use tt_kernels::sfpu::approx::{MathMode, TWINS};
    use tt_kernels::sfpu::ops::reference;
    use tt_kernels::tensor::Eltwise;
    let card = std::env::var("TT_SILICON_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(1))
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        for (precise, approx) in TWINS {
            let op = Eltwise {
                kind: precise,
                scalar: 0.0,
                scalar2: 0.0,
            };
            // Positive and moderate, in every op's range.
            let data = |n: usize| -> Vec<f32> {
                pattern_f32(n, 3)
                    .into_iter()
                    .map(|v| v.abs() + 0.05)
                    .collect()
            };
            let mut slope = [0.0f64; 2];
            for (k, mode) in [MathMode::Precise, MathMode::Approx]
                .into_iter()
                .enumerate()
            {
                s.set_math_mode(mode);
                let mut at = Vec::new();
                for tiles in [1usize, 4, 16, 64] {
                    let (r, c) = (32 * tiles, 32);
                    let v = data(r * c);
                    let x = s.upload(&v, r, c).unwrap();
                    let o = s.eltwise(op, &x, None).unwrap();
                    s.sync().unwrap();
                    // The program the mode runs, bit for bit.
                    let run_kind = if mode == MathMode::Approx {
                        approx
                    } else {
                        precise
                    };
                    let want = reference(run_kind, 0.0, &v, None, r, c);
                    let got = s.download(&o).unwrap();
                    assert!(
                        got.iter()
                            .zip(&want)
                            .all(|(g, w)| g.to_bits() == w.to_bits()),
                        "{precise:#x} {mode:?} {tiles} tiles: not the program's bits"
                    );
                    s.free(o).unwrap();
                    let mut t = Vec::new();
                    for _ in 0..7 {
                        let start = Instant::now();
                        let o = s.eltwise(op, &x, None).unwrap();
                        s.sync().unwrap();
                        t.push(start.elapsed());
                        s.free(o).unwrap();
                    }
                    s.free(x).unwrap();
                    at.push((tiles, median(t).as_secs_f64() * 1e6));
                }
                // The slope between the 4- and 64-tile runs: per tile.
                slope[k] = (at[3].1 - at[1].1) / 60.0;
                let line: Vec<String> = at
                    .iter()
                    .map(|(t, us)| format!("{t}: {us:.1} us"))
                    .collect();
                println!(
                    "MEASURE approx {precise:#06x} {mode:?}: {}",
                    line.join(", ")
                );
            }
            println!(
                "MEASURE approx {precise:#06x} per tile: precise {:.2} us, approx {:.2} us, saved {:.2}x",
                slope[0],
                slope[1],
                slope[0] / slope[1]
            );
        }
        s.set_math_mode(MathMode::Precise);
    }) {
        panic!("{e}");
    }
}
