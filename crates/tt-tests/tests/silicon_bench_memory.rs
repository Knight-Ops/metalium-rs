//! Memory throughput: what RISCV B's data mover gets out of GDDR6, device-timed.
//!
//! `cargo xtask bench --filter silicon_bench_memory`
//!
//! Every span is the mover's own `LIST_BEGIN -> LIST_END` (or barrier exit ->
//! `LIST_END`) on its tile's counter: from its first look at the list to every
//! move in it having landed, with none of the host's submit or PCIe polling in
//! it. One tile's mover is held against its NIU's link (or, on one channel,
//! the channel); every tile's together against the card's GDDR6.
//!
//! No tile-to-tile L1 copy: the mover has no op for one (`dm::op`), so the NoC
//! alone cannot be separated from GDDR here yet.

#![cfg(feature = "silicon")]

use std::time::Instant;

use tt_device::tlb::WindowKind;
use tt_device::trace::TraceEvent;
use tt_device::Window;
use tt_isa::dm::{self, op};
use tt_isa::dram::Dram;
use tt_isa::mailbox::{trace as ev, TRACE_BUFFER, TRACE_BUFFER_BYTES};
use tt_isa::noc::{Noc0, NocCoord};
use tt_kernels::dm::DataMover;
use tt_tests::backend::{device_index, tensix_grid};
use tt_tests::bench::{on_card, pattern, report, report_rate, Conditions, Peak, Stats, REPS};
use tt_tests::harness::{tensix_tile, tile, Dev};

/// L1 the moves land in (or leave from): 256 KiB, clear of the mover's list,
/// scratch and trace chunk below it and the role mailboxes and trace buffer
/// above.
const L1: u32 = 0x2_0000;
const L1_SPAN: u32 = 0x4_0000;
/// Where in each channel the benchmarks' data sits: clear of the low GDDR the
/// sessions use, and far from each channel's top 16 MiB (divergence row 63).
const BASE: u64 = 64 << 20;

const _: () = assert!(
    (L1 + L1_SPAN) as u64 <= TRACE_BUFFER || L1 as u64 >= TRACE_BUFFER + TRACE_BUFFER_BYTES
);
const _: () =
    assert!(L1 as u64 >= dm::TRACE_CHUNK + dm::TRACE_CHUNK_ENTRIES as u64 * dm::ENTRY_BYTES);

type Entry = [u32; 8];

fn entry(kind: u32, ch: u8, port: u32, dram_off: u64, l1: u32, len: u32) -> Entry {
    [kind, ch as u32, port, dram_off as u32, l1, len, 0, 0]
}

/// The cycles between the first `begin` and the last `end` of the mover's
/// events.
fn span(trace: &[TraceEvent], begin: u32, end: u32) -> u64 {
    let at = |want: u32| {
        trace
            .iter()
            .filter(|e| ev::split(e.token) == (ev::MOVER, want))
            .map(|e| e.cycles)
            .collect::<Vec<_>>()
    };
    let (b, e) = (at(begin), at(end));
    let (b, e) = (
        *b.first().expect("no begin event"),
        *e.last().expect("no end event"),
    );
    e.checked_sub(b).expect("end before begin")
}

/// Run `list` on `m` with the timestamper on, and return its events.
fn traced(d: &mut Dev<'_>, w: &Window, m: &mut DataMover<Noc0>, list: &[Entry]) -> Vec<TraceEvent> {
    let t = m.tile();
    d.configure_trace(w, t, TRACE_BUFFER, TRACE_BUFFER_BYTES)
        .unwrap();
    m.submit_list(d, w, list).unwrap();
    m.wait(d, w).unwrap();
    d.read_trace(w, t, TRACE_BUFFER).unwrap()
}

/// One traced run's list span, warm-up first, `REPS` times.
fn list_cycles(d: &mut Dev<'_>, w: &Window, m: &mut DataMover<Noc0>, list: &[Entry]) -> Stats {
    traced(d, w, m, list);
    Stats::of((0..REPS).map(|_| span(&traced(d, w, m, list), ev::LIST_BEGIN, ev::LIST_END) as f64))
}

struct One<'a> {
    d: &'a mut Dev<'static>,
    w: Window,
    w4: Window,
    dram: Dram,
    m: DataMover<Noc0>,
    c: Conditions,
}

/// The gate tile's mover, traced, with each channel's first MiB at [`BASE`]
/// holding a known pattern.
fn one_tile(d: &mut Dev<'static>, f: impl FnOnce(&mut One<'_>)) {
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
    let dram = d.dram_grid(&w).unwrap();
    let t = tensix_tile();
    let c = Conditions::measure(d, device_index(), t);
    c.print();
    let m = DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap();
    d.write32(&w, t, dm::TRACE, 1).unwrap();
    let mut one = One {
        d,
        w,
        w4,
        dram,
        m,
        c,
    };
    f(&mut one);
    let One { d, w, m, .. } = one;
    d.write32(&w, t, dm::TRACE, 0).unwrap();
    m.stop(d, &w).unwrap();
}

/// Entry sizes swept.
const SIZES: [u32; 7] = [64, 256, 1024, 4096, 16384, 65536, 131072];

/// NoC requests the sweeps put in one list. Not a limit any more -- the mover
/// keeps its requests in flight under `tt_isa::noc::niu::MAX_IN_FLIGHT`, so a
/// list of any length completes exactly (`gddr_in_flight` runs 300 a tile) --
/// but the size of the baseline's lists (`docs/firmware-performance.md`), so
/// the numbers stay comparable.
const MAX_REQUESTS: u32 = 240;

/// How many entries of `len` a list carries: up to a MiB, and no more than
/// [`MAX_REQUESTS`] 16 KiB packets (which also keeps it inside the 1024-event
/// trace buffer: two events an entry, two a list).
fn entries_of(len: u32) -> u32 {
    ((1u32 << 20) / len)
        .min(MAX_REQUESTS / len.div_ceil(16384))
        .max(1)
}

/// Which channel index and port list entry `i` uses.
type Spread = Box<dyn Fn(u32) -> (usize, u32)>;

/// Reads and writes of each size: one channel through one port, one channel
/// through all three of its ports, and every channel in turn.
fn sweep(one: &mut One<'_>, kind: u32) {
    let name = if kind == op::READ { "read" } else { "write" };
    let chans: Vec<u8> = one.dram.channels().map(|c| c.index()).collect();
    let n = chans.len();
    if kind == op::WRITE {
        let src = pattern(L1_SPAN as usize, 3);
        let t = one.m.tile();
        one.d.write(&one.w, t, L1 as u64, &src).unwrap();
        let _ = one.d.read32(&one.w, t, (L1 + L1_SPAN - 4) as u64).unwrap();
    }
    let link = one.c.noc_link();
    let chan = one.c.gddr_channel();
    for len in SIZES {
        let count = entries_of(len);
        let bytes = (count * len) as f64;
        let shapes: [(&str, Spread, Option<&Peak>); 3] = [
            ("1ch 1port", Box::new(|_| (0, 0)), chan.as_ref()),
            ("1ch 3ports", Box::new(|i| (0, i % 3)), chan.as_ref()),
            (
                "all ch",
                Box::new(move |i| (i as usize % n, 0)),
                Some(&link),
            ),
        ];
        for (shape, of, peak) in &shapes {
            let list: Vec<Entry> = (0..count)
                .map(|i| {
                    let (c, port) = of(i);
                    // Each channel's run of entries is contiguous in it.
                    let k = if *shape == "all ch" { i / n as u32 } else { i };
                    entry(
                        kind,
                        chans[c],
                        port,
                        BASE + (k * len) as u64,
                        L1 + (i * len) % L1_SPAN,
                        len,
                    )
                })
                .collect();
            let cycles = list_cycles(one.d, &one.w, &mut one.m, &list);
            let us = cycles.map(|c| one.c.cycles_to_us(c));
            report_rate(
                &format!("gddr {name} 1 tile {shape:<10} {len:>6} B x{count}"),
                "device",
                bytes,
                us,
                *peak,
            );
            report(
                &format!("gddr {name} 1 tile {shape:<10} {len:>6} B cycles/entry"),
                "cycles",
                "device",
                cycles.map(|c| c / count as f64),
            );
        }
    }
}

/// One tile's mover reading GDDR6 into its L1: sizes x channel spread.
#[test]
#[ignore = "benchmark"]
fn mover_read_sweep() {
    on_card(|d| {
        one_tile(d, |one| {
            let data = pattern(1 << 20, 7);
            let chans: Vec<_> = one.dram.channels().collect();
            for ch in &chans {
                one.d
                    .dram_write(&one.w4, ch.range(BASE, 1 << 20).unwrap(), &data)
                    .unwrap();
            }
            sweep(one, op::READ);
            // The bytes arrived: the last 256 KiB of a 1 MiB read of channel 0
            // in 64 KiB entries.
            let list: Vec<Entry> = (0..16)
                .map(|i| {
                    entry(
                        op::READ,
                        chans[0].index(),
                        0,
                        BASE + i * 65536,
                        L1 + (i as u32 * 65536) % L1_SPAN,
                        65536,
                    )
                })
                .collect();
            one.m.run_list(one.d, &one.w, &list).unwrap();
            let mut back = vec![0u8; L1_SPAN as usize];
            one.d
                .l1_read(&one.w, one.m.tile(), L1 as u64, &mut back)
                .unwrap();
            assert!(
                back[..] == data[(1 << 20) - L1_SPAN as usize..],
                "the reads did not land"
            );
        });
    });
}

/// One tile's mover writing its L1 to GDDR6. A write's list ends when every
/// write is acknowledged by the DRAM tile.
#[test]
#[ignore = "benchmark"]
fn mover_write_sweep() {
    on_card(|d| {
        one_tile(d, |one| {
            sweep(one, op::WRITE);
            // The bytes arrived: 256 KiB of L1 written to channel 1, read back.
            let ch = one.dram.channels().nth(1).unwrap();
            let at = BASE + (32 << 20);
            let list: Vec<Entry> = (0..4)
                .map(|i| {
                    entry(
                        op::WRITE,
                        ch.index(),
                        0,
                        at + i * 65536,
                        L1 + i as u32 * 65536,
                        65536,
                    )
                })
                .collect();
            one.m.run_list(one.d, &one.w, &list).unwrap();
            let mut back = vec![0u8; L1_SPAN as usize];
            one.d
                .dram_read(&one.w4, ch.range(at, L1_SPAN as u64).unwrap(), &mut back)
                .unwrap();
            assert!(
                back == pattern(L1_SPAN as usize, 3),
                "the writes did not land"
            );
        });
    });
}

/// The mover's fixed costs: a list of `WAIT`s (loop, decode, a NoC drain
/// check), and the list itself (an empty-of-data one-entry list). Traced and
/// untraced from the host, too: what tracing itself costs the numbers above.
#[test]
#[ignore = "benchmark"]
fn mover_overhead() {
    on_card(|d| {
        one_tile(d, |one| {
            let wait = [op::WAIT, 0, 0, 0, 0, 0, 0, 0];
            for count in [1u32, 16, 256] {
                let list = vec![wait; count as usize];
                let cycles = list_cycles(one.d, &one.w, &mut one.m, &list);
                report(
                    &format!("mover {count:>3} WAITs list span"),
                    "cycles",
                    "device",
                    cycles,
                );
                report(
                    &format!("mover {count:>3} WAITs per entry"),
                    "cycles",
                    "device",
                    cycles.map(|c| c / count as f64),
                );
            }
            // Tracing's own cost: the same 256 x 4 KiB read, host-timed, with
            // and without the timestamper stores.
            let ch = one.dram.channels().next().unwrap().index();
            let list: Vec<Entry> = (0..256)
                .map(|i| {
                    entry(
                        op::READ,
                        ch,
                        0,
                        BASE + i as u64 * 4096,
                        L1 + (i * 4096) % L1_SPAN,
                        4096,
                    )
                })
                .collect();
            let t = one.m.tile();
            for (label, on) in [("traced", 1), ("untraced", 0)] {
                one.d.write32(&one.w, t, dm::TRACE, on).unwrap();
                one.d
                    .configure_trace(&one.w, t, TRACE_BUFFER, TRACE_BUFFER_BYTES)
                    .unwrap();
                let mut v = Vec::new();
                for _ in 0..=REPS {
                    one.d
                        .configure_trace(&one.w, t, TRACE_BUFFER, TRACE_BUFFER_BYTES)
                        .unwrap();
                    let t0 = Instant::now();
                    one.m.submit_list(one.d, &one.w, &list).unwrap();
                    one.m.wait(one.d, &one.w).unwrap();
                    v.push(t0.elapsed());
                }
                report(
                    &format!("mover 256 x 4096 B read, {label}"),
                    "us",
                    "host",
                    Stats::of_durations(v.into_iter().skip(1)),
                );
            }
            one.d.write32(&one.w, t, dm::TRACE, 1).unwrap();
        });
    });
}

/// Every usable tile with a running mover, in the grid's order.
struct Fleet {
    w: Window,
    w4: Window,
    dram: Dram,
    movers: Vec<DataMover<Noc0>>,
    c: Conditions,
    /// Each tile's counter less tile 0's, in cycles, and the uncertainty of it.
    offset: Vec<i64>,
    skew_bound: Vec<i64>,
}

fn fleet(d: &mut Dev<'static>) -> Fleet {
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
    let dram = d.dram_grid(&w).unwrap();
    let grid = tensix_grid(d);
    let coords: Vec<NocCoord<Noc0>> = grid.tiles::<Noc0>().collect();
    let c = Conditions::measure(d, device_index(), tensix_tile());
    c.print();
    let mut movers = Vec::new();
    for t in coords {
        let t = tile(d, t.x(), t.y());
        match DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1) {
            Ok(m) => {
                d.write32(&w, t, dm::TRACE, 1).unwrap();
                movers.push(m);
            }
            Err(e) => println!(
                "MEASURE skipping ({},{}): the mover did not start: {e}",
                t.x(),
                t.y()
            ),
        }
    }
    let (offset, skew_bound) = skew(d, &w, &movers, &c);
    Fleet {
        w,
        w4,
        dram,
        movers,
        c,
        offset,
        skew_bound,
    }
}

/// Each tile's counter against tile 0's, estimated from the host: every
/// counter read twice, in a forward and then a reverse sweep, each read
/// placed in host time. The two estimates bracket the read's own PCIe time,
/// so their mean is the offset and half their difference bounds its error.
fn skew(
    d: &mut Dev<'_>,
    w: &Window,
    movers: &[DataMover<Noc0>],
    c: &Conditions,
) -> (Vec<i64>, Vec<i64>) {
    let t0 = Instant::now();
    let sample = |d: &mut Dev<'_>, t| {
        let h = t0.elapsed().as_secs_f64() * 1e6;
        let v = d.wall_clock(w, t).unwrap();
        let h2 = t0.elapsed().as_secs_f64() * 1e6;
        // The counter, less where the host's clock says tile 0's would be.
        v as f64 - (h + h2) / 2.0 * c.cycles_per_us()
    };
    let tiles: Vec<_> = movers.iter().map(|m| m.tile()).collect();
    let fwd: Vec<f64> = tiles.iter().map(|&t| sample(d, t)).collect();
    let mut rev: Vec<f64> = tiles.iter().rev().map(|&t| sample(d, t)).collect();
    rev.reverse();
    let base = (fwd[0] + rev[0]) / 2.0;
    let offset = fwd
        .iter()
        .zip(&rev)
        .map(|(a, b)| ((a + b) / 2.0 - base) as i64)
        .collect::<Vec<_>>();
    let bound = fwd
        .iter()
        .zip(&rev)
        .map(|(a, b)| ((a - b).abs() / 2.0) as i64)
        .collect::<Vec<_>>();
    report(
        &format!("tile counter offset vs tile 0, {} tiles", tiles.len()),
        "cycles",
        "host",
        Stats::of(offset.iter().map(|&o| o as f64)),
    );
    report(
        "tile counter offset uncertainty",
        "cycles",
        "host",
        Stats::of(bound.iter().map(|&o| o as f64)),
    );
    (offset, bound)
}

/// What every tile's list does, by tile and entry index.
#[derive(Copy, Clone, Debug, PartialEq)]
enum Mix {
    Read,
    Write,
    /// Reads and writes alternating.
    Half,
    /// Reads, every tile from channel 0 (through its three ports).
    OneChannel,
}

/// Entries per tile per run, of [`AGG_LEN`] each: 3.75 MiB a tile, in
/// [`MAX_REQUESTS`] packets.
const AGG_LEN: u32 = 65536;
const AGG_ENTRIES: u32 = MAX_REQUESTS / (AGG_LEN / 16384);

/// What each tile's list moves: `entries` of `len` bytes.
#[derive(Copy, Clone)]
struct Shape {
    entries: u32,
    len: u32,
}

impl Shape {
    const AGG: Shape = Shape {
        entries: AGG_ENTRIES,
        len: AGG_LEN,
    };

    fn bytes(self) -> f64 {
        (self.entries * self.len) as f64
    }
}

/// A tile's share of a channel: each tile reads (or writes) its own MiB of
/// every channel, so no two tiles touch the same bytes.
fn agg_list(fleet: &Fleet, tile: usize, n: usize, mix: Mix, shape: Shape) -> Vec<Entry> {
    let Shape { entries, len } = shape;
    let chans: Vec<u8> = fleet.dram.channels().map(|c| c.index()).collect();
    let nc = chans.len() as u32;
    let coord = fleet.movers[0].tile();
    let mut list = vec![[
        op::BARRIER,
        n as u32,
        coord.x() as u32,
        coord.y() as u32,
        0,
        0,
        0,
        0,
    ]];
    for i in 0..entries {
        let (c, k) = match mix {
            Mix::OneChannel => (0, i),
            _ => ((i + tile as u32) % nc, i / nc),
        };
        let kind = match mix {
            Mix::Write => op::WRITE,
            Mix::Half if i % 2 == 1 => op::WRITE,
            _ => op::READ,
        };
        let per_tile = if mix == Mix::OneChannel {
            (entries * len) as u64
        } else {
            1 << 20
        };
        let off = BASE + tile as u64 * per_tile + (k * len) as u64;
        list.push(entry(
            kind,
            chans[c as usize],
            tile as u32 % 3,
            off,
            L1 + (i * len) % L1_SPAN,
            len,
        ));
    }
    list
}

/// One run of `n` tiles' lists at once: per tile, its rate from barrier exit
/// to list end; together, all bytes over the earliest exit to the latest end
/// on tile 0's clock (each tile's events less its offset); and the host's
/// time from the last list submitted to the last seen done. Also how many of
/// the tiles' requests waited for room under the in-flight cap.
fn agg_run(d: &mut Dev<'_>, f: &mut Fleet, n: usize, mix: Mix, shape: Shape) -> AggRun {
    let coord = f.movers[0].tile();
    d.write32(&f.w, coord, dm::BARRIER_COUNTER, 0).unwrap();
    let lists: Vec<_> = (0..n).map(|t| agg_list(f, t, n, mix, shape)).collect();
    let throttle = |d: &mut Dev<'_>, f: &Fleet| {
        f.movers[..n]
            .iter()
            .map(|m| m.throttle(d, &f.w).unwrap().stalls as u64)
            .sum::<u64>()
    };
    let stalls_before = throttle(d, f);
    for m in &f.movers[..n] {
        d.configure_trace(&f.w, m.tile(), TRACE_BUFFER, TRACE_BUFFER_BYTES)
            .unwrap();
    }
    // The barrier holds every list until the last is submitted.
    for (m, list) in f.movers[..n].iter_mut().zip(&lists) {
        m.submit_list(d, &f.w, list).unwrap();
    }
    let h0 = Instant::now();
    for m in &f.movers[..n] {
        m.wait(d, &f.w).unwrap();
    }
    let host_us = h0.elapsed().as_secs_f64() * 1e6;
    let bytes = shape.bytes();
    let mut rates = Vec::new();
    let (mut first, mut last) = (i64::MAX, i64::MIN);
    for (i, m) in f.movers[..n].iter().enumerate() {
        let trace = d.read_trace(&f.w, m.tile(), TRACE_BUFFER).unwrap();
        // The barrier is the list's first entry: its end is the release.
        let released = trace
            .iter()
            .find(|e| ev::split(e.token) == (ev::MOVER, ev::ENTRY_END))
            .expect("no barrier end")
            .cycles;
        let end = trace
            .iter()
            .rfind(|e| ev::split(e.token) == (ev::MOVER, ev::LIST_END))
            .expect("no list end")
            .cycles;
        rates.push(f.c.rate(bytes, (end - released) as f64));
        first = first.min(released as i64 - f.offset[i]);
        last = last.max(end as i64 - f.offset[i]);
    }
    let together = f.c.rate(bytes * n as f64, (last - first) as f64);
    let stalls = throttle(d, f) - stalls_before;
    AggRun {
        rates,
        together,
        host_us,
        stalls,
    }
}

struct AggRun {
    /// Each tile's rate, bytes per second.
    rates: Vec<f64>,
    /// All tiles' bytes over the earliest start to the latest end.
    together: f64,
    host_us: f64,
    /// Requests, over every tile, that waited under the in-flight cap.
    stalls: u64,
}

/// [`agg_run`] after a warm-up, [`REPS`] times, reported as `label`: the
/// tiles together against `peak`, the same from the host, each tile against
/// its NoC link, and the requests that waited under the in-flight cap.
#[allow(clippy::too_many_arguments)]
fn measure(
    d: &mut Dev<'_>,
    f: &mut Fleet,
    n: usize,
    mix: Mix,
    shape: Shape,
    peak: Option<&Peak>,
    label: &str,
) -> Vec<AggRun> {
    agg_run(d, f, n, mix, shape);
    let runs: Vec<AggRun> = (0..REPS).map(|_| agg_run(d, f, n, mix, shape)).collect();
    let bytes = shape.bytes() * n as f64;
    // As microseconds for the whole card, so the stats are a time and the
    // rate is the median's.
    let us = Stats::of(runs.iter().map(|r| bytes / r.together * 1e6));
    report_rate(&format!("{label} together"), "device", bytes, us, peak);
    report_rate(
        &format!("{label} together (host-timed)"),
        "host",
        bytes,
        Stats::of(runs.iter().map(|r| r.host_us)),
        peak,
    );
    let link = if mix == Mix::Half {
        f.c.noc_link_both()
    } else {
        f.c.noc_link()
    };
    let per = Stats::of(
        runs.iter()
            .flat_map(|r| r.rates.iter().map(|rate| shape.bytes() / rate * 1e6)),
    );
    report_rate(
        &format!("{label}, each tile"),
        "device",
        shape.bytes(),
        per,
        Some(&link),
    );
    report(
        &format!("{label}: requests that waited under the in-flight cap, per run"),
        "requests",
        "device",
        Stats::of(runs.iter().map(|r| r.stalls as f64)),
    );
    runs
}

/// Every tile's mover at once, against the card's GDDR6: how many movers it
/// takes to reach the ceiling, and what the ceiling is. Reads, writes, a mix,
/// and every tile on one channel.
#[test]
#[ignore = "benchmark"]
fn gddr_aggregate() {
    on_card(|d| {
        let t0 = Instant::now();
        let mut f = fleet(d);
        let have = f.movers.len();
        println!(
            "MEASURE gddr_aggregate: {have} movers running, started in {:.1?}",
            t0.elapsed()
        );
        // Two tiles' ranges hold a pattern, checked after the read runs.
        let check = [0usize, have - 1];
        for &t in &check {
            for ch in f.dram.channels() {
                let data = pattern(1 << 20, t as u32 * 16 + ch.index() as u32 + 1);
                d.dram_write(
                    &f.w4,
                    ch.range(BASE + t as u64 * (1 << 20), 1 << 20).unwrap(),
                    &data,
                )
                .unwrap();
            }
        }
        // Every tile's L1 a known pattern before the writes (the reads
        // overwrite it).
        let src = pattern(L1_SPAN as usize, 5);
        let card = f.c.gddr_card();
        let chan = f.c.gddr_channel();
        // `AGG_TILES=4,120` runs only those counts.
        let counts: Vec<usize> = match std::env::var("AGG_TILES") {
            Ok(s) => s
                .split(',')
                .map(|n| n.trim().parse::<usize>().unwrap().min(have))
                .collect(),
            Err(_) => {
                let mut c: Vec<usize> = [1, 2, 4, 8, 16, 32, 64]
                    .into_iter()
                    .filter(|&n| n < have)
                    .collect();
                c.push(have);
                c
            }
        };
        for mix in [Mix::Read, Mix::Write, Mix::Half, Mix::OneChannel] {
            if mix == Mix::Write {
                for m in &f.movers {
                    d.write(&f.w, m.tile(), L1 as u64, &src).unwrap();
                }
            }
            let peak = if mix == Mix::OneChannel {
                chan.as_ref()
            } else {
                card.as_ref()
            };
            for &n in &counts {
                measure(
                    d,
                    &mut f,
                    n,
                    mix,
                    Shape::AGG,
                    peak,
                    &format!("gddr {mix:?} {n:>3} tiles"),
                );
            }
            if mix == Mix::Read {
                // The timed lists reuse L1 slots across channels, so which
                // entry's bytes a slot ends with is a race. The check runs a
                // tile's last four entries alone -- one slot each, through the
                // same mover -- and compares them with the GDDR they read.
                for &t in &check {
                    let list = agg_list(&f, t, have, Mix::Read, Shape::AGG);
                    let last = &list[list.len() - 4..];
                    f.movers[t].run_list(d, &f.w, last).unwrap();
                    let mut back = vec![0u8; L1_SPAN as usize];
                    d.l1_read(&f.w, f.movers[t].tile(), L1 as u64, &mut back)
                        .unwrap();
                    for (k, e) in last.iter().enumerate() {
                        let ch = e[1];
                        let at = (e[3] as u64 - BASE - t as u64 * (1 << 20)) as usize;
                        let want = pattern(1 << 20, t as u32 * 16 + ch + 1);
                        let l1 = (e[4] - L1) as usize;
                        assert!(
                            back[l1..l1 + AGG_LEN as usize] == want[at..at + AGG_LEN as usize],
                            "tile {t}'s read entry {k} (ch {ch}, +{at:#x}) did not land"
                        );
                    }
                }
            }
        }
        // A tile's write landed: tile 0's first entry, its L1's first 64 KiB.
        let list = agg_list(&f, 0, have, Mix::Write, Shape::AGG);
        let ch = f.dram.channel(list[1][1] as u8).unwrap();
        let mut back = vec![0u8; AGG_LEN as usize];
        d.dram_read(
            &f.w4,
            ch.range(list[1][3] as u64, AGG_LEN as u64).unwrap(),
            &mut back,
        )
        .unwrap();
        assert!(
            back[..] == src[..AGG_LEN as usize],
            "tile 0's write did not land"
        );
        let bound = f.skew_bound.iter().copied().max().unwrap_or(0);
        println!(
            "MEASURE gddr_aggregate: 'together' spans carry up to {bound} cycles ({:.2} us) of counter-offset uncertainty",
            f.c.cycles_to_us(bound as f64)
        );
        let Fleet { w, movers, .. } = f;
        for m in movers {
            let t = m.tile();
            d.write32(&w, t, dm::TRACE, 0).unwrap();
            m.stop(d, &w).unwrap();
        }
    });
}

/// More than the NIU's 8-bit counter holds, under the worst contention:
/// every tile reading 300 x 16 KiB -- 300 requests a list -- from one
/// channel. The in-flight cap (`tt_isa::noc::niu::InFlight`) must engage,
/// and the card's total must stay at or under the channel's ceiling: a list
/// reported done before its data landed would show as more than the channel
/// can carry. Each checked tile's last entries are then run alone and their
/// bytes compared.
#[test]
#[ignore = "benchmark"]
fn gddr_in_flight() {
    on_card(|d| {
        let mut f = fleet(d);
        let have = f.movers.len();
        let shape = Shape {
            entries: 300,
            len: 16384,
        };
        let chan = f.c.gddr_channel();
        let per_tile = shape.bytes() as usize;
        let check = [0usize, have - 1];
        let ch0 = f.dram.channels().next().unwrap();
        for &t in &check {
            let data = pattern(per_tile, t as u32 + 101);
            d.dram_write(
                &f.w4,
                ch0.range(BASE + (t * per_tile) as u64, per_tile as u64)
                    .unwrap(),
                &data,
            )
            .unwrap();
        }
        let runs = measure(
            d,
            &mut f,
            have,
            Mix::OneChannel,
            shape,
            chan.as_ref(),
            &format!(
                "gddr in flight {have:>3} tiles x {} x {} B, one channel",
                shape.entries, shape.len
            ),
        );
        let stalls: u64 = runs.iter().map(|r| r.stalls).sum();
        assert!(
            stalls > 0,
            "300 requests a tile on one channel never reached the in-flight cap"
        );
        let ceiling = chan.as_ref().unwrap().bytes_per_sec;
        for r in &runs {
            assert!(
                r.together <= ceiling * 1.02,
                "{:.1} GB/s through one channel of {:.1}: a list ended before its data landed",
                r.together / 1e9,
                ceiling / 1e9
            );
        }
        for &t in &check {
            let list = agg_list(&f, t, have, Mix::OneChannel, shape);
            let last = &list[list.len() - (L1_SPAN / shape.len) as usize..];
            f.movers[t].run_list(d, &f.w, last).unwrap();
            let mut back = vec![0u8; L1_SPAN as usize];
            d.l1_read(&f.w, f.movers[t].tile(), L1 as u64, &mut back)
                .unwrap();
            let want = pattern(per_tile, t as u32 + 101);
            for e in last {
                let at = (e[3] as u64 - BASE) as usize - t * per_tile;
                let l1 = (e[4] - L1) as usize;
                assert!(
                    back[l1..l1 + shape.len as usize] == want[at..at + shape.len as usize],
                    "tile {t}: the entry at +{at:#x} did not land"
                );
            }
        }
        let Fleet { w, movers, .. } = f;
        for m in movers {
            let t = m.tile();
            d.write32(&w, t, dm::TRACE, 0).unwrap();
            m.stop(d, &w).unwrap();
        }
    });
}
