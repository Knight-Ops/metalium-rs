//! The path through a tile's cores, device-timed: B's data mover, then the
//! three role runners T0 (unpack), T1 (math) and T2 (pack), as a normal op
//! takes it.
//!
//! `cargo xtask bench --filter silicon_bench_path`
//!
//! Every hop is a gap between two events on one tile's timestamper, so B and
//! the three roles are stamped by one counter:
//!
//! ```text
//! host submit -> B LIST_BEGIN -> ENTRY_BEGIN(KERNEL) -> KICK
//!   -> Tn WOKE -> START -> PUSHED -> RETIRED -> ACKED     (each of T0..T2)
//!   -> B ROLES_DONE -> ENTRY_END -> LIST_END -> host sees done
//! ```
//!
//! `null_kernel` runs that path with nothing in it (empty programs, then NOP
//! streams), so what it measures is the path's own cost. `workflow_ops` runs
//! real ops through a `Session` and splits each one's device time into data
//! movement, hand-offs and the roles' programs.

#![cfg(feature = "silicon")]

use std::collections::BTreeMap;
use std::time::Instant;

use tt_device::tlb::WindowKind;
use tt_device::trace::TraceEvent;
use tt_isa::dm::{self, op};
use tt_isa::mailbox::{trace as ev, TRACE_BUFFER, TRACE_BUFFER_BYTES};
use tt_kernels::dm::DataMover;
use tt_kernels::profile::DeviceProfile;
use tt_kernels::runtime::{Kernel, Resident, Schedule};
use tt_kernels::session::{Session, TileChoice};
use tt_tests::backend::device_index;
use tt_tests::bench::{on_card, report, Conditions, Stats, REPS};
use tt_tests::harness::tensix_tile;
use tt_ttsim::fork_scope;

const BUDGET: u64 = 1 << 30;

/// The cycle of the first event `(source, event)` in `trace`.
fn at(trace: &[TraceEvent], source: u32, event: u32) -> u64 {
    trace
        .iter()
        .find(|e| ev::split(e.token) == (source, event))
        .unwrap_or_else(|| panic!("no event ({source}, {event}) in the trace"))
        .cycles
}

/// One kernel's hops, in cycles, keyed by name in path order.
fn hops(trace: &[TraceEvent], host_before: u64) -> Vec<(String, f64)> {
    let m = ev::MOVER;
    let list_begin = at(trace, m, ev::LIST_BEGIN);
    let entry_begin = at(trace, m, ev::ENTRY_BEGIN);
    let kick = at(trace, m, ev::KICK);
    let done = at(trace, m, ev::ROLES_DONE);
    let list_end = at(trace, m, ev::LIST_END);
    let mut v = vec![
        (
            "0 host write -> B list begin".to_string(),
            list_begin.saturating_sub(host_before) as f64,
        ),
        (
            "1 B list begin -> KERNEL entry".into(),
            (entry_begin - list_begin) as f64,
        ),
        (
            "2 B KERNEL entry -> kick".into(),
            (kick - entry_begin) as f64,
        ),
    ];
    let mut last_ack = 0;
    for t in 0..3u32 {
        let woke = at(trace, t, ev::WOKE);
        let start = at(trace, t, ev::START);
        let pushed = at(trace, t, ev::PUSHED);
        let retired = at(trace, t, ev::RETIRED);
        let acked = at(trace, t, ev::ACKED);
        last_ack = last_ack.max(acked);
        v.extend([
            (
                format!("3 T{t} kick -> woke"),
                woke.saturating_sub(kick) as f64,
            ),
            (format!("4 T{t} woke -> start"), (start - woke) as f64),
            (format!("5 T{t} start -> pushed"), (pushed - start) as f64),
            (
                format!("6 T{t} pushed -> retired"),
                (retired - pushed) as f64,
            ),
            (format!("7 T{t} retired -> acked"), (acked - retired) as f64),
        ]);
    }
    v.extend([
        (
            "8 B last ack -> roles done".into(),
            done.saturating_sub(last_ack) as f64,
        ),
        (
            "9 B roles done -> list end".into(),
            (list_end - done) as f64,
        ),
        (
            "A B list begin -> list end".into(),
            (list_end - list_begin) as f64,
        ),
    ]);
    v
}

/// The path with nothing on it: the resident roles run an empty program,
/// then `n` NOPs each, from a one-entry `KERNEL` list on the tile's mover.
/// Every hop of every run, median over runs; and the host's round trip.
#[test]
#[ignore = "benchmark"]
fn null_kernel() {
    on_card(|d| {
        let t = tensix_tile();
        let c = Conditions::measure(d, device_index(), t);
        c.print();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let roles = tt_firmware_images::ROLES;
        let mut r = Resident::start(d, t, &roles, BUDGET).unwrap();
        let mut m = DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap();
        r.set_profiling(true);
        d.write32(&w, t, dm::TRACE, 1).unwrap();
        for n in [0usize, 1, 64, 1024] {
            let nops = vec![tt_isa::backend::nop(); n];
            let mut kernel = Kernel::new([&nops, &nops, &nops], Schedule::Concurrent(&[]));
            kernel.restores_semaphores = true;
            let mut by: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            let mut host = Vec::new();
            for rep in 0..=REPS {
                let g = r.reserve(d, &roles, &kernel, BUDGET, 1, false).unwrap();
                d.configure_trace(&w, t, TRACE_BUFFER, TRACE_BUFFER_BYTES)
                    .unwrap();
                let before = d.wall_clock(&w, t).unwrap();
                let h0 = Instant::now();
                m.submit_list(d, &w, &[[op::KERNEL, g.start, 0, 0, 0, 0, 0, 0]])
                    .unwrap();
                m.wait(d, &w).unwrap();
                let rt = h0.elapsed();
                r.reserved_done(d, true).unwrap();
                let trace = d.read_trace(&w, t, TRACE_BUFFER).unwrap();
                if rep == 0 {
                    continue;
                }
                host.push(rt);
                for (k, v) in hops(&trace, before) {
                    by.entry(k).or_default().push(v);
                }
            }
            for (k, v) in &by {
                let label = &k[2..];
                report(
                    &format!("path {n:>4} NOPs/role: {label}"),
                    "cycles",
                    "device",
                    Stats::of(v.iter().copied()),
                );
            }
            if n > 0 {
                for t in 0..3 {
                    let push = Stats::of(by[&format!("5 T{t} start -> pushed")].iter().copied());
                    report(
                        &format!("path {n:>4} NOPs/role: T{t} push rate"),
                        "cycles/word",
                        "device",
                        push.map(|c| c / n as f64),
                    );
                }
            }
            report(
                &format!("path {n:>4} NOPs/role: host submit -> done seen"),
                "us",
                "host",
                Stats::of_durations(host),
            );
        }
        d.write32(&w, t, dm::TRACE, 0).unwrap();
        m.stop(d, &w).unwrap();
    });
}

/// A session on the gate tile alone, with GDDR, and its conditions.
fn one_unit(card: u16) -> (Session<tt_kmd::Kmd>, Conditions) {
    let gate = tt_tests::backend::GATE_TILE;
    let mut s = Session::open_card(
        card,
        tt_firmware_images::ROLES,
        TileChoice::Exactly(gate.0, gate.1),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
    // `PIPELINE=1`: GDDR matmuls overlap their moves with their kernels
    // (`Session::set_pipeline`, checklist 9.15).
    let pipeline = std::env::var("PIPELINE").is_ok_and(|v| v == "1");
    s.set_pipeline(pipeline);
    // A pipelined profile is the mover's alone: with the mover and the roles
    // storing to the timestamper at once, card 0's streams lost events.
    s.set_profile_roles(!pipeline);
    println!("MEASURE pipeline: {pipeline}");
    let t = s.tile();
    let c = Conditions::measure(s.device(), card, t);
    c.print();
    (s, c)
}

/// What one profiled op's device time went to, on one unit: the union of
/// its lists, each entry kind's share of it, the kernels' hand-offs, and
/// each role's program, wake and acknowledgement time.
#[derive(Default)]
struct Split {
    lists: usize,
    device: f64,
    parts: BTreeMap<String, f64>,
}

fn split(p: &DeviceProfile) -> Split {
    let mut s = Split::default();
    let (mut first, mut last) = (u64::MAX, 0);
    for u in &p.units {
        for sp in u.spans().unwrap_or_else(|e| panic!("{e}")) {
            let len = (sp.end - sp.begin) as f64;
            if sp.name == "list" {
                s.lists += 1;
                first = first.min(sp.begin);
                last = last.max(sp.end);
            } else {
                *s.parts
                    .entry(format!("{} {}", sp.track, sp.name))
                    .or_default() += len;
            }
        }
    }
    s.device = last.saturating_sub(first) as f64;
    s
}

/// Real ops through a one-unit `Session`, profiled: per op, the device time
/// (first list begin to last list end), the host's time to the op's sync,
/// and the device time each mover entry kind, the kernels' kicks, and each
/// role's program took -- with the fraction of the device time the busiest
/// role's programs were running (the rest is the path's overhead and data
/// movement the roles waited on).
#[test]
#[ignore = "benchmark"]
fn workflow_ops() {
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_kernels::sfpu::ops::kind_sfpu;
    use tt_kernels::tensor::Eltwise;
    let card = device_index();
    if let Err(e) = fork_scope(|| {
        let (mut s, c) = one_unit(card);
        let f = |n: usize, seed: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 17) as f32 / 8.0 - 1.0
                })
                .collect()
        };
        type Op<'a> = Box<dyn FnMut(&mut Session<tt_kmd::Kmd>) + 'a>;
        let mut cases: Vec<(String, Op)> = Vec::new();
        for (m, k, n) in [
            (32, 32, 32),
            (32, 256, 32),
            (128, 128, 128),
            (512, 512, 512),
        ] {
            for (fname, fid) in [("LoFi", Fidelity::Lo), ("HiFi4", Fidelity::HiFi4)] {
                if fid == Fidelity::Lo && m > 32 {
                    continue;
                }
                let a = s.upload(&f(m * k, 1), m, k).unwrap();
                let b = s.upload(&f(k * n, 2), k, n).unwrap();
                cases.push((
                    format!("matmul {m}x{k}x{n} {fname}"),
                    Box::new(move |s| {
                        let o = s
                            .matmul_dram(&a, false, &b, false, SrcRoute::Tf32FromFp32, fid, 400_000)
                            .unwrap_or_else(|e| panic!("{e}"));
                        s.sync().unwrap();
                        s.free(o).unwrap();
                    }),
                ));
            }
        }
        for tiles in [1usize, 64] {
            let (r, cols) = (32 * tiles, 32);
            let x = std::rc::Rc::new(s.upload(&f(r * cols, 3), r, cols).unwrap());
            let y = std::rc::Rc::new(s.upload(&f(r * cols, 4), r, cols).unwrap());
            {
                let op = Eltwise {
                    kind: tt_kernels::kind::ADD,
                    scalar: 0.0,
                    scalar2: 0.0,
                };
                let (x, y) = (x.clone(), y.clone());
                cases.push((
                    format!("add {tiles} tiles on sfpu"),
                    Box::new(move |s| {
                        let o = s.eltwise(op, &x, Some(&*y)).unwrap();
                        s.sync().unwrap();
                        s.free(o).unwrap();
                    }),
                ));
            }
            {
                // The sum over rows, in Flex's order on the SFPU.
                let x = x.clone();
                cases.push((
                    format!("sum rows {tiles} tiles on sfpu"),
                    Box::new(move |s| {
                        let o = s.sum_rows(&x).unwrap();
                        s.sync().unwrap();
                        s.free(o).unwrap();
                    }),
                ));
            }
            let exp = Eltwise {
                kind: kind_sfpu::EXP,
                scalar: 0.0,
                scalar2: 0.0,
            };
            cases.push((
                format!("exp {tiles} tiles on sfpu"),
                Box::new(move |s| {
                    let o = s.eltwise(exp, &x, None).unwrap();
                    s.sync().unwrap();
                    s.free(o).unwrap();
                }),
            ));
        }
        for (name, op) in &mut cases {
            op(&mut s);
            let mut device = Vec::new();
            let mut host = Vec::new();
            let mut lists = 0;
            let mut parts: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            for _ in 0..REPS {
                s.profile_start().unwrap();
                let h0 = Instant::now();
                op(&mut s);
                host.push(h0.elapsed());
                let p = s.profile_stop().unwrap();
                let sp = split(&p);
                lists = sp.lists;
                device.push(sp.device);
                for (k, v) in sp.parts {
                    parts.entry(k).or_default().push(v);
                }
            }
            let dev = Stats::of(device.iter().copied());
            report(
                &format!("op {name}: host op -> sync"),
                "us",
                "host",
                Stats::of_durations(host),
            );
            report(
                &format!("op {name}: device ({lists} lists)"),
                "us",
                "device",
                dev.map(|x| c.cycles_to_us(x)),
            );
            let busiest = (0..3)
                .filter_map(|t| parts.get(&format!("T{t} program")))
                .map(|v| Stats::of(v.iter().copied()).median)
                .fold(0.0, f64::max);
            for (k, v) in &parts {
                report(
                    &format!("op {name}: {k}"),
                    "us",
                    "device",
                    Stats::of(v.iter().copied()).map(|x| c.cycles_to_us(x)),
                );
            }
            report(
                &format!("op {name}: busiest role's programs / device time"),
                "%",
                "device",
                Stats::of([100.0 * busiest / dev.median]),
            );
        }
    }) {
        panic!("{e}");
    }
}

/// Back to back: `n` ops queued and one sync, against one op and its sync.
/// The device's busy time (the union of its lists) against the host's wall
/// time says how much of the host's time the device sat idle -- what queueing
/// hides of the per-op path, and what it does not.
#[test]
#[ignore = "benchmark"]
fn queue_steady_state() {
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    let card = device_index();
    if let Err(e) = fork_scope(|| {
        let (mut s, c) = one_unit(card);
        let (m, k, n) = (32, 256, 32);
        let v = |len: usize, seed: usize| -> Vec<f32> {
            (0..len).map(|i| ((i + seed) % 7) as f32 - 3.0).collect()
        };
        let a = s.upload(&v(m * k, 1), m, k).unwrap();
        let b = s.upload(&v(k * n, 2), k, n).unwrap();
        let run = |s: &mut Session<tt_kmd::Kmd>, count: usize| {
            let mut outs = Vec::new();
            for _ in 0..count {
                outs.push(
                    s.matmul_dram(
                        &a,
                        false,
                        &b,
                        false,
                        SrcRoute::Tf32FromFp32,
                        Fidelity::HiFi4,
                        400_000,
                    )
                    .unwrap_or_else(|e| panic!("{e}")),
                );
            }
            s.sync().unwrap();
            for o in outs {
                s.free(o).unwrap();
            }
        };
        run(&mut s, 1);
        for count in [1usize, 4, 16] {
            let mut host = Vec::new();
            let mut busy = Vec::new();
            let mut span = Vec::new();
            for _ in 0..REPS {
                s.profile_start().unwrap();
                let h0 = Instant::now();
                run(&mut s, count);
                let wall = h0.elapsed();
                let p = s.profile_stop().unwrap();
                let mut lists: Vec<(u64, u64)> = p
                    .units
                    .iter()
                    .flat_map(|u| u.spans().unwrap())
                    .filter(|sp| sp.name == "list")
                    .map(|sp| (sp.begin, sp.end))
                    .collect();
                lists.sort_unstable();
                let mut covered = 0u64;
                let mut reach = 0u64;
                for (b, e) in &lists {
                    let b = (*b).max(reach);
                    if *e > b {
                        covered += e - b;
                        reach = *e;
                    }
                }
                let first = lists.first().map_or(0, |l| l.0);
                host.push(wall.as_secs_f64() * 1e6 / count as f64);
                busy.push(c.cycles_to_us(covered as f64) / count as f64);
                span.push(c.cycles_to_us((reach - first) as f64) / count as f64);
            }
            let (h, b) = (Stats::of(host), Stats::of(busy));
            let label = format!("queue {count:>2} x matmul {m}x{k}x{n} HiFi4");
            report(&format!("{label}: host per op"), "us", "host", h);
            report(&format!("{label}: device busy per op"), "us", "device", b);
            report(
                &format!("{label}: device first list -> last per op"),
                "us",
                "device",
                Stats::of(span),
            );
            report(
                &format!("{label}: device busy / host wall"),
                "%",
                "device",
                Stats::of([100.0 * b.median / h.median]),
            );
        }
    }) {
        panic!("{e}");
    }
}

/// Whether pipelined matmuls (`Session::set_pipeline`) are ever slower:
/// shapes from MNIST's up to 1024s, on 1, 8 and 32 tiles, each with
/// pipelining off and on, device time from the mover's events alone in both
/// (the same measurement either way). A pipelined block fits half the arena,
/// so an op may gather the same operand tiles more often; this is where that
/// would show. `SWEEP_TILES` picks the tile counts.
#[test]
#[ignore = "benchmark"]
fn matmul_pipeline_sweep() {
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    let card = device_index();
    let counts: Vec<usize> = match std::env::var("SWEEP_TILES") {
        Ok(s) => s.split(',').map(|n| n.trim().parse().unwrap()).collect(),
        Err(_) => vec![1, 8, 32],
    };
    let shapes: [[usize; 3]; 9] = [
        [64, 784, 128],
        [64, 128, 10],
        [128, 128, 128],
        [256, 256, 256],
        [256, 1024, 256],
        [512, 512, 512],
        [1024, 256, 1024],
        [64, 2048, 512],
        [1024, 1024, 1024],
    ];
    for tiles in counts {
        if let Err(e) = fork_scope(|| {
            let mut s =
                Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(tiles))
                    .unwrap_or_else(|e| panic!("{e}"));
            s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
            s.set_profile_roles(false);
            let t = s.tile();
            let c = Conditions::measure(s.device(), card, t);
            let f = |n: usize, seed: u32| -> Vec<f32> {
                (0..n)
                    .map(|i| {
                        ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 17) as f32 / 8.0
                            - 1.0
                    })
                    .collect()
            };
            for [m, k, n] in shapes {
                let a = s.upload(&f(m * k, 1), m, k).unwrap();
                let b = s.upload(&f(k * n, 2), k, n).unwrap();
                let mut median = [0f64; 2];
                let mut overlapped = 0;
                let mut bits: [Vec<f32>; 2] = Default::default();
                for (on, slot) in [(false, 0usize), (true, 1)] {
                    s.set_pipeline(on);
                    let run = |s: &mut Session<tt_kmd::Kmd>| {
                        let o = s
                            .matmul_dram(
                                &a,
                                false,
                                &b,
                                false,
                                SrcRoute::Tf32FromFp32,
                                Fidelity::HiFi4,
                                BUDGET,
                            )
                            .unwrap_or_else(|e| panic!("{e}"));
                        s.sync().unwrap();
                        o
                    };
                    let lists_before: u64 = s.lists_per_tile().iter().sum();
                    let drains_before = s.drains();
                    let o = run(&mut s);
                    let lists = s.lists_per_tile().iter().sum::<u64>() - lists_before;
                    let drains = s.drains() - drains_before;
                    println!(
                        "MEASURE sweep {m}x{k}x{n} on {tiles} tiles, pipeline {on}: {lists} lists, {drains} drains"
                    );
                    bits[slot] = s.download(&o).unwrap();
                    s.free(o).unwrap();
                    let before = s.pipelined_blocks();
                    // Device time from the profile; for an op whose events
                    // overflow the timestamper's buffer, the host's time from
                    // the op to its sync instead (marked `host`).
                    let mut device = Vec::new();
                    let mut host = Vec::new();
                    let mut overflowed = false;
                    // Past ~2^29 multiply-adds a profile's events overflow
                    // the buffer (and the op's sync reports it): time those
                    // from the host.
                    let profiled = m * k * n < 1 << 29;
                    overflowed |= !profiled;
                    for _ in 0..REPS {
                        if profiled {
                            s.profile_start().unwrap();
                        }
                        let h0 = Instant::now();
                        let o = run(&mut s);
                        host.push(h0.elapsed().as_secs_f64() * 1e6);
                        if profiled {
                            match s.profile_stop() {
                                Ok(p) => device.push(split(&p).device),
                                Err(_) => overflowed = true,
                            }
                        }
                        s.free(o).unwrap();
                    }
                    if on {
                        overlapped = (s.pipelined_blocks() - before) / REPS as u64;
                    }
                    let (st, timed) = if overflowed {
                        (Stats::of(host.iter().copied()), "host")
                    } else {
                        (
                            Stats::of(device.iter().copied()).map(|x| c.cycles_to_us(x)),
                            "device",
                        )
                    };
                    report(
                        &format!(
                            "matmul {m}x{k}x{n} on {tiles} tiles, pipeline {}",
                            if on { "on " } else { "off" }
                        ),
                        "us",
                        timed,
                        st,
                    );
                    median[slot] = st.median;
                }
                assert!(
                    bits[0]
                        .iter()
                        .zip(&bits[1])
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{m}x{k}x{n} on {tiles} tiles: pipelining changed the bits"
                );
                println!(
                    "MEASURE sweep {m}x{k}x{n} on {tiles} tiles: on/off {:.3} ({overlapped} blocks overlapped per op)",
                    median[1] / median[0]
                );
                s.free(a).unwrap();
                s.free(b).unwrap();
            }
        }) {
            panic!("{tiles} tiles: {e}");
        }
    }
}
