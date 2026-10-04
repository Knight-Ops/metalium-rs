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
    s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
        .unwrap();
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

/// Compare serialized and overlapped ownership, with NC as the fixed writer.
fn sweep_arm(s: &mut Session<tt_kmd::Kmd>, on: bool) {
    s.set_pipeline(on);
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
            s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
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
                    sweep_arm(&mut s, on);
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
                    // With NC scattering, B's timestamper events come out
                    // garbled (an entry's end repeated): host time instead.
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

/// Whether pipelined element-wise ops and reductions are ever slower: an
/// add, an `exp`, a sum over columns and a max over rows, from MNIST's sizes
/// up to 2048², on 1, 8 and 32 tiles (`SWEEP_TILES`), each with pipelining off
/// and on, device time from the mover's events alone in both. Prints the
/// tiles a pipelined run held, the number `tensor::MIN_PIPELINED_RUN` is set
/// from.
#[test]
#[ignore = "benchmark"]
fn sfpu_pipeline_sweep() {
    use tt_kernels::kind;
    use tt_kernels::sfpu::ops::kind_sfpu;
    use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
    use tt_kernels::tensor::{DramTensor, Eltwise};
    let card = device_index();
    let counts: Vec<usize> = match std::env::var("SWEEP_TILES") {
        Ok(s) => s.split(',').map(|n| n.trim().parse().unwrap()).collect(),
        Err(_) => vec![1, 8, 32],
    };
    let shapes: [[usize; 2]; 6] = [
        [64, 784],
        [256, 256],
        [512, 512],
        [1024, 1024],
        [2048, 1024],
        [2048, 2048],
    ];
    type Op<T> = fn(&mut Session<T>, &DramTensor, &DramTensor) -> DramTensor;
    let ops: [(&str, Op<tt_kmd::Kmd>); 4] = [
        ("add", |s, a, b| {
            s.eltwise(
                Eltwise {
                    kind: kind::ADD,
                    scalar: 0.0,
                    scalar2: 0.0,
                },
                a,
                Some(b),
            )
            .unwrap()
        }),
        ("exp", |s, a, _| {
            s.eltwise(
                Eltwise {
                    kind: kind_sfpu::EXP,
                    scalar: 0.0,
                    scalar2: 0.0,
                },
                a,
                None,
            )
            .unwrap()
        }),
        ("sum over cols", |s, a, _| {
            s.reduce(a, ReduceOp::Sum, Axis::Cols).unwrap()
        }),
        ("max over rows", |s, a, _| {
            s.reduce(a, ReduceOp::Max, Axis::Rows).unwrap()
        }),
    ];
    for tiles in counts {
        if let Err(e) = fork_scope(|| {
            let mut s =
                Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(tiles))
                    .unwrap_or_else(|e| panic!("{e}"));
            s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
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
            for [r, cols] in shapes {
                let a = s.upload(&f(r * cols, 1), r, cols).unwrap();
                let b = s.upload(&f(r * cols, 2), r, cols).unwrap();
                for (name, op) in ops {
                    let mut median = [0f64; 2];
                    let mut host_median = [0f64; 2];
                    let mut bits: [Vec<f32>; 2] = Default::default();
                    let mut overlapped = 0;
                    for (on, slot) in [(false, 0usize), (true, 1)] {
                        sweep_arm(&mut s, on);
                        let lists_before: u64 = s.lists_per_tile().iter().sum();
                        let drains_before = s.drains();
                        let t0 = Instant::now();
                        let o = op(&mut s, &a, &b);
                        let t1 = Instant::now();
                        s.sync().unwrap();
                        println!(
                            "MEASURE sfpu sweep {name} {r}x{cols} on {tiles} tiles, pipeline {on}: {} lists, {} drains, call {:?}, sync {:?}",
                            s.lists_per_tile().iter().sum::<u64>() - lists_before,
                            s.drains() - drains_before,
                            t1 - t0,
                            t1.elapsed()
                        );
                        bits[slot] = s.download(&o).unwrap();
                        s.free(o).unwrap();
                        let before = s.pipelined_blocks();
                        let cache = |s: &Session<tt_kmd::Kmd>| {
                            s.program_cache_stats().iter().fold([0u64; 3], |a, c| {
                                [a[0] + c.misses, a[1] + c.bytes_uploaded, a[2] + c.bypassed]
                            })
                        };
                        let cache_before = cache(&s);
                        // Past 2048 tiles a profile's events overflow the
                        // timestamper's buffer: time those from the host.
                        let profiled = r * cols <= 2048 * 1024;
                        let (mut device, mut host) = (Vec::new(), Vec::new());
                        let mut overflowed = !profiled;
                        for _ in 0..REPS {
                            if profiled {
                                s.profile_start().unwrap();
                            }
                            let h0 = Instant::now();
                            let o = op(&mut s, &a, &b);
                            s.sync().unwrap();
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
                        // The host's time unprofiled: a profile syncs and
                        // drains the trace inside every op's call.
                        let mut plain_host = Vec::new();
                        for _ in 0..REPS {
                            let h0 = Instant::now();
                            let o = op(&mut s, &a, &b);
                            s.sync().unwrap();
                            plain_host.push(h0.elapsed().as_secs_f64() * 1e6);
                            s.free(o).unwrap();
                        }
                        host = plain_host;
                        let cache_after = cache(&s);
                        println!(
                            "MEASURE sfpu sweep {name} {r}x{cols} on {tiles} tiles, pipeline {on}: per op {} program misses, {} B uploaded, {} bypassed",
                            (cache_after[0] - cache_before[0]) / REPS as u64,
                            (cache_after[1] - cache_before[1]) / REPS as u64,
                            (cache_after[2] - cache_before[2]) / REPS as u64,
                        );
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
                                "{name} {r}x{cols} on {tiles} tiles, pipeline {}",
                                if on { "on " } else { "off" }
                            ),
                            "us",
                            timed,
                            st,
                        );
                        median[slot] = st.median;
                        host_median[slot] = Stats::of(host.iter().copied()).median;
                    }
                    assert!(
                        bits[0]
                            .iter()
                            .zip(&bits[1])
                            .all(|(p, q)| p.to_bits() == q.to_bits()),
                        "{name} {r}x{cols} on {tiles} tiles: pipelined bits differ"
                    );
                    println!(
                        "MEASURE sfpu sweep {name} {r}x{cols} on {tiles} tiles: on/off {:.3}, host {:.3} ({overlapped} runs overlapped)",
                        median[1] / median[0],
                        host_median[1] / host_median[0],
                    );
                }
                s.free(a).unwrap();
                s.free(b).unwrap();
            }
        }) {
            panic!("{tiles} tiles: {e}");
        }
    }
}

/// Where the host's time goes queueing an op (checklist 9.17): an add of
/// MNIST's size and of 512², and a 256³ matmul, 50 of each queued back to
/// back on 1, 8 and 32 tiles (`SWEEP_TILES`), with `Session::host_times`'s
/// stages, PCIe traffic and TLB retargets per op, and the call's wall time.
#[test]
#[ignore = "benchmark"]
fn host_time_per_op() {
    use tt_kernels::kind;
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_kernels::tensor::Eltwise;
    let card = device_index();
    let counts: Vec<usize> = match std::env::var("SWEEP_TILES") {
        Ok(s) => s.split(',').map(|n| n.trim().parse().unwrap()).collect(),
        Err(_) => vec![1, 8, 32],
    };
    const OPS: usize = 50;
    for tiles in counts {
        if let Err(e) = fork_scope(|| {
            let mut s =
                Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(tiles))
                    .unwrap_or_else(|e| panic!("{e}"));
            s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
            let add = Eltwise {
                kind: kind::ADD,
                scalar: 0.0,
                scalar2: 0.0,
            };
            for (name, r, c) in [("add", 64, 784), ("add", 512, 512), ("matmul", 256, 256)] {
                let a = s.upload(&vec![0.5; r * c], r, c).unwrap();
                let b = s.upload(&vec![0.25; r * c], r, c).unwrap();
                let op = |s: &mut Session<tt_kmd::Kmd>| {
                    if name == "add" {
                        s.eltwise(add, &a, Some(&b)).unwrap()
                    } else {
                        s.matmul_dram(
                            &a,
                            false,
                            &b,
                            false,
                            SrcRoute::Tf32FromFp32,
                            Fidelity::HiFi4,
                            BUDGET,
                        )
                        .unwrap()
                    }
                };
                // Warm: programs resident, movers started.
                for _ in 0..3 {
                    let o = op(&mut s);
                    s.sync().unwrap();
                    s.free(o).unwrap();
                }
                s.reset_host_times();
                let t0 = Instant::now();
                let outs: Vec<_> = (0..OPS).map(|_| op(&mut s)).collect();
                let calls = t0.elapsed();
                s.sync().unwrap();
                let total = t0.elapsed();
                for o in outs {
                    s.free(o).unwrap();
                }
                let h = s.host_times().clone();
                let per = |d: std::time::Duration| d.as_secs_f64() * 1e6 / OPS as f64;
                println!(
                    "MEASURE host {name} {r}x{c} on {tiles} tiles: call {:.1} us/op, with sync {:.1} us/op",
                    per(calls),
                    per(total)
                );
                for (stage, t) in h.stages.iter() {
                    if t.count == 0 {
                        continue;
                    }
                    println!(
                        "MEASURE host {name} {r}x{c} on {tiles} tiles:   {:<10} {:>7.1} us/op  x{:<5.1} reads {:>5.1} writes {:>6.1} retargets {:>5.1} /op",
                        format!("{stage:?}"),
                        per(t.time),
                        t.count as f64 / OPS as f64,
                        t.traffic.read_calls as f64 / OPS as f64,
                        t.traffic.write_calls as f64 / OPS as f64,
                        t.traffic.retargets as f64 / OPS as f64,
                    );
                }
                s.free(a).unwrap();
                s.free(b).unwrap();
            }
        }) {
            panic!("{tiles} tiles: {e}");
        }
    }
}

/// Trace replay against queueing ops fresh (checklist 9.17): a layer's
/// forward pass -- matmul, add a row, relu, matmul -- at MNIST's size and at
/// 512 x 1024, 50 times back to back, queued fresh (pipelined and plain: a
/// capture runs plain) and replayed from one capture, on 1, 8 and 32 tiles
/// (`SWEEP_TILES`). Host time is the calls' alone; end to end includes the
/// sync, so it shows what streaming a trace from GDDR costs the device.
#[test]
#[ignore = "benchmark"]
fn trace_replay_vs_fresh() {
    use tt_kernels::kind;
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_kernels::tensor::{DramTensor, Eltwise};
    let card = device_index();
    let counts: Vec<usize> = match std::env::var("SWEEP_TILES") {
        Ok(s) => s.split(',').map(|n| n.trim().parse().unwrap()).collect(),
        Err(_) => vec![1, 8, 32],
    };
    const PASSES: usize = 50;
    const OPS: usize = 4;
    fn forward(
        s: &mut Session<tt_kmd::Kmd>,
        x: &DramTensor,
        w1: &DramTensor,
        b1: &DramTensor,
        w2: &DramTensor,
    ) -> DramTensor {
        let mm = |s: &mut Session<tt_kmd::Kmd>, a, b| {
            s.matmul_dram(
                a,
                false,
                b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                BUDGET,
            )
            .unwrap()
        };
        let ew = |s: &mut Session<tt_kmd::Kmd>, kind, a, b| {
            s.eltwise(
                Eltwise {
                    scalar2: 0.0,
                    kind,
                    scalar: 0.0,
                },
                a,
                b,
            )
            .unwrap()
        };
        let h = mm(s, x, w1);
        let hb = ew(s, kind::ADD_ROW, &h, Some(b1));
        let r = ew(s, kind::RELU, &hb, None);
        let y = mm(s, &r, w2);
        for t in [h, hb, r] {
            s.free(t).unwrap();
        }
        y
    }
    for tiles in counts {
        if let Err(e) = fork_scope(|| {
            let mut s =
                Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(tiles))
                    .unwrap_or_else(|e| panic!("{e}"));
            s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
            for (batch, input, hidden, out) in [(64, 784, 128, 10), (512, 1024, 1024, 1024)] {
                let x = s.upload(&vec![0.5; batch * input], batch, input).unwrap();
                let w1 = s
                    .upload(&vec![0.01; input * hidden], input, hidden)
                    .unwrap();
                let b1 = s.upload(&vec![0.1; hidden], 1, hidden).unwrap();
                let w2 = s.upload(&vec![0.02; hidden * out], hidden, out).unwrap();
                let what = format!("{batch}x{input}x{hidden}x{out} on {tiles} tiles");
                let per_op = |d: std::time::Duration| d.as_secs_f64() * 1e6 / (PASSES * OPS) as f64;
                let fresh = |s: &mut Session<tt_kmd::Kmd>, pipeline: bool| {
                    s.set_pipeline(pipeline);
                    for _ in 0..3 {
                        let y = forward(s, &x, &w1, &b1, &w2);
                        s.sync().unwrap();
                        s.free(y).unwrap();
                    }
                    let tr0 = s.device().traffic();
                    let t0 = Instant::now();
                    let ys: Vec<_> = (0..PASSES).map(|_| forward(s, &x, &w1, &b1, &w2)).collect();
                    let calls = t0.elapsed();
                    s.sync().unwrap();
                    let total = t0.elapsed();
                    let tr = s.device().traffic();
                    for y in ys {
                        s.free(y).unwrap();
                    }
                    println!(
                        "MEASURE replay {what}, fresh {}: host {:.2} us/op, end to end {:.2} us/op, {:.1} writes {:.1} reads /op",
                        if pipeline { "pipelined" } else { "plain    " },
                        per_op(calls),
                        per_op(total),
                        (tr.write_calls - tr0.write_calls) as f64 / (PASSES * OPS) as f64,
                        (tr.read_calls - tr0.read_calls) as f64 / (PASSES * OPS) as f64,
                    );
                };
                fresh(&mut s, true);
                fresh(&mut s, false);
                s.set_pipeline(true);
                s.begin_trace().unwrap();
                let y = forward(&mut s, &x, &w1, &b1, &w2);
                let id = s.end_trace().unwrap();
                for _ in 0..3 {
                    s.replay(id).unwrap();
                    s.sync().unwrap();
                }
                let tr0 = s.device().traffic();
                let t0 = Instant::now();
                for _ in 0..PASSES {
                    s.replay(id).unwrap();
                }
                let calls = t0.elapsed();
                s.sync().unwrap();
                let total = t0.elapsed();
                let tr = s.device().traffic();
                println!(
                    "MEASURE replay {what}, replayed       : host {:.2} us/op, end to end {:.2} us/op, {:.1} writes {:.1} reads /op",
                    per_op(calls),
                    per_op(total),
                    (tr.write_calls - tr0.write_calls) as f64 / (PASSES * OPS) as f64,
                    (tr.read_calls - tr0.read_calls) as f64 / (PASSES * OPS) as f64,
                );
                // A server's batch: the input written from the host, the
                // trace replayed, the output read back.
                let values = vec![0.25f32; batch * input];
                let (mut wr, mut rp, mut dl) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..20 {
                    let t0 = Instant::now();
                    s.write(&x, &values).unwrap();
                    let t1 = Instant::now();
                    s.replay(id).unwrap();
                    s.sync().unwrap();
                    let t2 = Instant::now();
                    let _ = s.download(&y).unwrap();
                    wr.push((t1 - t0).as_secs_f64() * 1e6);
                    rp.push((t2 - t1).as_secs_f64() * 1e6);
                    dl.push(t2.elapsed().as_secs_f64() * 1e6);
                }
                println!(
                    "MEASURE replay {what}, a served batch: write {} B {:.0} us, replay {:.0} us, read {} B {:.0} us",
                    batch * input * 4,
                    Stats::of(wr.into_iter()).median,
                    Stats::of(rp.into_iter()).median,
                    batch * out * 4,
                    Stats::of(dl.into_iter()).median,
                );
                s.release_trace(id).unwrap();
                for t in [y, x, w1, b1, w2] {
                    s.free(t).unwrap();
                }
            }
        }) {
            panic!("{tiles} tiles: {e}");
        }
    }
}
