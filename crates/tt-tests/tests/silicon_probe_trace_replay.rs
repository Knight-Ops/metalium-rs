//! Probe: where the device time of one replayed trace goes, on one tile's mover (B).
//!
//! `cargo xtask bench --device 0 --filter silicon_probe_trace_replay`
//!
//! A trace replay is one `CALL` per unit. B fetches the stream a chunk at a time, patches it, runs
//! the chunk, and waits for the NoC before the next; each streaming op is a `PAIR_CALL` whose reader
//! section is a nested `call`. The timestamper stamps every entry begin and end (`ENTRY_*`), every
//! kick of the roles and their completion (`KICK`, `ROLES_DONE`); the roles do not stamp
//! (`set_profile_roles(false)`: with both, a streaming export loses events). So the gaps between
//! one entry's end and the next's begin are the loop's own work (chunk fetch, patch, drain, entry
//! decode), the entries' spans are what each op costs, and `KICK` to `ROLES_DONE` is compute.
//!
//! The probe is a baseline for the trace hot path (`docs/plans`: single-digit microseconds an op):
//! what the control overhead of a replay is, against the work it carries. Each replay is stamped
//! (~12 cycles an event); the host-timed replay without stamps is printed beside it so the
//! perturbation shows.

#![cfg(feature = "silicon")]

use std::time::Instant;

use tt_isa::dm::op;
use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::profile::{op_name, DeviceProfile, Event};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_kernels::trace::TraceId;
use tt_tests::backend::device_index;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

type Card = Session<tt_kmd::Kmd>;

/// A layer's forward pass at MNIST's size: matmul, add a row, relu, matmul.
fn layer(s: &mut Card) -> TraceId {
    let (batch, input, hidden, out) = (64usize, 784usize, 128usize, 10usize);
    let x = s.upload(&vec![0.5; batch * input], batch, input).unwrap();
    let w1 = s
        .upload(&vec![0.01; input * hidden], input, hidden)
        .unwrap();
    let b1 = s.upload(&vec![0.1; hidden], 1, hidden).unwrap();
    let w2 = s.upload(&vec![0.02; hidden * out], hidden, out).unwrap();
    s.begin_trace().unwrap();
    let mm = |s: &mut Card, a: &DramTensor, b: &DramTensor| {
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
    let ew = |s: &mut Card, kind, a: &DramTensor, b: Option<&DramTensor>| {
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
    let h = mm(s, &x, &w1);
    let hb = ew(s, kind::ADD_ROW, &h, Some(&b1));
    let r = ew(s, kind::RELU, &hb, None);
    let _y = mm(s, &r, &w2);
    s.end_trace().unwrap()
}

/// `n` relus of one 64x64 tensor, one after another: ops with next to no compute, so a replay is
/// its own control cost.
fn chain(s: &mut Card, n: usize) -> TraceId {
    let x = s.upload(&vec![0.5; 64 * 64], 64, 64).unwrap();
    s.begin_trace().unwrap();
    let mut t = x;
    for _ in 0..n {
        t = s
            .eltwise(
                Eltwise {
                    scalar2: 0.0,
                    kind: kind::RELU,
                    scalar: 0.0,
                },
                &t,
                None,
            )
            .unwrap();
    }
    s.end_trace().unwrap()
}

fn replay_profiled(s: &mut Card, trace: TraceId) -> DeviceProfile {
    s.profile_start().unwrap();
    s.replay(trace).unwrap();
    s.sync().unwrap();
    s.profile_stop().unwrap()
}

/// Host-timed replays, `replay` then `sync` each, in microseconds: the median of `reps`.
fn host_replay_us(s: &mut Card, trace: TraceId, reps: usize) -> f64 {
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            s.replay(trace).unwrap();
            s.sync().unwrap();
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn dump(profile: &DeviceProfile, unit: usize) {
    let ev = &profile.units[unit].events;
    let t0 = ev.first().map_or(0, |e| e.cycles);
    let mut last = t0;
    println!(
        "DUMP unit {unit}: {} events, {:.1} ticks/us",
        ev.len(),
        profile.ticks_per_us
    );
    for e in ev {
        let what = match Event::decode(e.token) {
            Event::List { begin, entries } => {
                format!("LIST_{} {entries}", if begin { "BEGIN" } else { "END" })
            }
            Event::Entry { begin, op } => {
                format!(
                    "ENTRY_{} {:#x} {}",
                    if begin { "BEGIN" } else { "END" },
                    op,
                    op_name(op)
                )
            }
            Event::Kick { done, generation } => {
                format!("{} {generation}", if done { "ROLES_DONE" } else { "KICK" })
            }
            other => format!("{other:?}"),
        };
        println!("DUMP {:>8} +{:>6}  {what}", e.cycles - t0, e.cycles - last);
        last = e.cycles;
    }
}

#[test]
#[ignore = "benchmark"]
fn replay_dump() {
    let card = device_index();
    let tiles: Vec<usize> = std::env::var("SWEEP_TILES")
        .map(|s| s.split(',').map(|n| n.trim().parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![1]);
    for tiles in tiles {
        fork_scope(|| {
            let mut s =
                Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(tiles))
                    .unwrap_or_else(|e| panic!("{e}"));
            s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
            s.set_profile_roles(false);
            for (name, trace) in [("layer", layer(&mut s)), ("chain8", chain(&mut s, 8))] {
                for _ in 0..3 {
                    s.replay(trace).unwrap();
                    s.sync().unwrap();
                }
                let host = host_replay_us(&mut s, trace, 20);
                println!("DUMP {name} on {tiles} tiles: host replay {host:.1} us unstamped");
                let profile = replay_profiled(&mut s, trace);
                dump(&profile, 0);
                s.release_trace(trace).unwrap();
            }
        })
        .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// What an interval between two consecutive stamps of the mover is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Part {
    /// `LIST_BEGIN` to the first entry.
    CallStart,
    /// One chunk's `LIST_END` to the next chunk's `LIST_BEGIN`: the progress write, the NoC fetch
    /// of the next 64 entries and the patch pass over them. A replay with a stream of more than
    /// 64 entries has one of these per extra chunk.
    ChunkGap,
    /// The trace's `POKE`s: role descriptor words written before the ops (entry spans and the gaps between).
    Pokes,
    /// The trace's setup `KERNEL` entry (the host-run semaphore setup), span and gaps.
    SetupKernel,
    /// From the end of one op's last entry to the next op's `LAUNCH` (finishing the previous
    /// `PAIR_CALL`, which includes waiting for NC's writes, and starting the next: its mailbox
    /// handshakes and its reader section's fetch).
    PairGap,
    /// The first op's gap: starting the `PAIR_CALL` only, nothing to finish before it.
    PairBegin,
    /// A `LAUNCH` entry: posting the generation to the roles.
    Launch,
    /// A `CB` credit entry (reserve or release: partly waits for NC and the roles).
    Credit,
    /// A data record or read (`READ_RUN`, `GATHER`, ...).
    Data,
    /// A `KERNEL_WAIT` entry: the roles' run, wake to acknowledge.
    KernelWait,
    /// A `BARRIER`.
    Barrier,
    /// The loop's own gap between two entries inside a section.
    LoopGap,
    /// The last entry's end to `LIST_END`: the NoC drain, and for the last chunk finishing the
    /// last `PAIR_CALL`.
    ListEnd,
    /// Anything else.
    Other,
}

fn class(word: u32) -> Part {
    use tt_isa::dm::record;
    match word {
        op::POKE => Part::Pokes,
        op::KERNEL => Part::SetupKernel,
        op::LAUNCH => Part::Launch,
        op::KERNEL_WAIT => Part::KernelWait,
        op::CB => Part::Credit,
        op::BARRIER => Part::Barrier,
        record::GATHER
        | record::SCATTER
        | record::READ_RUN
        | record::WRITE_RUN
        | record::FILL_PAD
        | record::PAD_WRITE
        | op::READ
        | op::WRITE => Part::Data,
        _ => Part::Other,
    }
}

/// One replay's cycles by [`Part`], from unit `unit`'s stamps; and how many ops (`KERNEL_WAIT`s)
/// and stamps it held. Every interval between two stamps lands in exactly one part: an entry's
/// span (begin to end, with any kick or completion stamps inside it) is its class's, the gap
/// before it is the loop's, a pair's or the prologue's, and anything unforeseen is `Other`.
fn account(
    profile: &DeviceProfile,
    unit: usize,
) -> (std::collections::BTreeMap<Part, u64>, usize, usize) {
    let ev: Vec<(Event, u64)> = profile.units[unit]
        .events
        .iter()
        .map(|e| (Event::decode(e.token), e.cycles))
        .collect();
    let mut by: std::collections::BTreeMap<Part, u64> = Default::default();
    let mut ops = 0;
    let mut open: Option<(u32, u64)> = None;
    // The entry that last ended, and whether a pair has started: until one has, gaps after the
    // trace's own entries are theirs.
    let mut ended: Option<u32> = None;
    let mut first_pair = true;
    let mut last = ev[0].1;
    let mut prev_was_list_begin = false;
    // A list has ended: the next `LIST_BEGIN` is another chunk's.
    let mut listed = false;
    for &(e, t) in &ev {
        let gap = t - last;
        match e {
            Event::List { begin: true, .. } => {
                if listed {
                    *by.entry(Part::ChunkGap).or_default() += gap;
                }
                prev_was_list_begin = true;
            }
            Event::List { begin: false, .. } => {
                listed = true;
                let part = if ended.is_some() {
                    Part::ListEnd
                } else {
                    Part::Other
                };
                *by.entry(part).or_default() += gap;
            }
            Event::Entry { begin: true, op: y } => {
                let part = match (prev_was_list_begin, ended) {
                    (true, _) => Part::CallStart,
                    (false, Some(x)) if y == op::LAUNCH => {
                        let part = if x == op::KERNEL_WAIT {
                            Part::PairGap
                        } else if first_pair {
                            Part::PairBegin
                        } else {
                            // A barrier or another entry ended the last op's section.
                            Part::PairGap
                        };
                        first_pair = false;
                        part
                    }
                    (false, Some(x)) if first_pair && class(x) == Part::Pokes => Part::Pokes,
                    (false, Some(x)) if first_pair && class(x) == Part::SetupKernel => {
                        Part::SetupKernel
                    }
                    (false, Some(_)) => Part::LoopGap,
                    (false, None) => Part::Other,
                };
                *by.entry(part).or_default() += gap;
                prev_was_list_begin = false;
                open = Some((y, t));
            }
            Event::Entry {
                begin: false,
                op: x,
            } => {
                let Some((o, t0)) = open.take() else {
                    *by.entry(Part::Other).or_default() += gap;
                    last = t;
                    continue;
                };
                debug_assert_eq!(o, x);
                *by.entry(class(x)).or_default() += t - t0;
                if x == op::KERNEL_WAIT {
                    ops += 1;
                }
                ended = Some(x);
            }
            // Inside an entry they are part of its span; outside one they are unforeseen.
            Event::Kick { .. }
            | Event::Throttle { .. }
            | Event::Role { .. }
            | Event::Unknown(_) => {
                if open.is_none() {
                    *by.entry(Part::Other).or_default() += gap;
                }
            }
        }
        last = t;
    }
    (by, ops, ev.len())
}

fn median_u64(mut v: Vec<u64>) -> u64 {
    v.sort_unstable();
    v[v.len() / 2]
}

/// Cycles by part for one replay of `trace`, medians over `reps` stamped replays, printed per
/// replay and per op. Checked against the stamps' own span: the parts partition it exactly.
fn breakdown(s: &mut Card, name: &str, tiles: usize, trace: TraceId, reps: usize) {
    for _ in 0..3 {
        s.replay(trace).unwrap();
        s.sync().unwrap();
    }
    let host = host_replay_us(s, trace, 20);
    let mut rows: Vec<std::collections::BTreeMap<Part, u64>> = Vec::new();
    let (mut ops, mut stamps, mut spans, mut ticks) = (0, 0, Vec::new(), 0.0);
    for _ in 0..reps {
        let profile = replay_profiled(s, trace);
        ticks = profile.ticks_per_us;
        let (by, n, stamped) = account(&profile, 0);
        let ev = &profile.units[0].events;
        assert!(
            matches!(Event::decode(ev[0].token), Event::List { begin: true, .. })
                && matches!(
                    Event::decode(ev.last().unwrap().token),
                    Event::List { begin: false, .. }
                ),
            "{name}: a replay is one list from LIST_BEGIN to LIST_END"
        );
        let span = ev.last().unwrap().cycles - ev[0].cycles;
        if by.values().sum::<u64>() != span {
            dump(&profile, 0);
            panic!(
                "{name}: the parts sum to {} of a span of {span}: {by:?}",
                by.values().sum::<u64>()
            );
        }
        spans.push(span);
        ops = n;
        stamps = stamped;
        rows.push(by);
    }
    let total = median_u64(spans);
    let us = |c: u64| c as f64 / ticks;
    println!(
        "MEASURE trace-replay {name} on {tiles} tiles: {} ops, {} stamps; device {:.1} us ({} cycles, median of {reps}); \
         host unstamped {host:.1} us (host - device = {:.1} us); {:.2} us per op",
        ops,
        stamps,
        us(total),
        total,
        host - us(total),
        us(total) / ops as f64
    );
    let parts: std::collections::BTreeSet<Part> =
        rows.iter().flat_map(|r| r.keys().copied()).collect();
    for part in parts {
        let c = median_u64(
            rows.iter()
                .map(|r| r.get(&part).copied().unwrap_or(0))
                .collect(),
        );
        println!(
            "MEASURE trace-replay {name} on {tiles} tiles:   {:<12} {:>7} cycles {:>6.2} us {:>5.1}%  {:>7.0} cycles/op",
            format!("{part:?}"),
            c,
            us(c),
            100.0 * c as f64 / total as f64,
            c as f64 / ops as f64
        );
    }
}

#[test]
#[ignore = "benchmark"]
fn replay_breakdown() {
    let card = device_index();
    let tiles: Vec<usize> = std::env::var("SWEEP_TILES")
        .map(|s| s.split(',').map(|n| n.trim().parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![1, 8]);
    for tiles in tiles {
        fork_scope(|| {
            let mut s =
                Session::open_card(card, tt_firmware_images::ROLES, TileChoice::Count(tiles))
                    .unwrap_or_else(|e| panic!("{e}"));
            s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
            s.set_profile_roles(false);
            let layer_trace = layer(&mut s);
            let chain_trace = chain(&mut s, 8);
            breakdown(&mut s, "layer", tiles, layer_trace, 7);
            breakdown(&mut s, "chain8", tiles, chain_trace, 7);
        })
        .unwrap_or_else(|e| panic!("{e}"));
    }
}
