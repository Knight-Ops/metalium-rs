//! H8 (`docs/learnings/dataflow-compiler.md` §13): what a cross-core channel's credit costs, against
//! the op barrier, on the same two units.
//!
//! `cargo xtask bench --device 1 --filter silicon_bench_channel_credit`. Host-timed: both movers'
//! lists open with a wait on a start word (`op::WAIT_CREDIT`), every list queued before the host
//! releases them, so a sample is the card's work from the release to the host seeing both done,
//! over `ITERATIONS` handoffs or barriers -- the release and the final poll, a few microseconds,
//! are spread over them (and counted in them: an upper bound).
//!
//! - **ping-pong:** a depth-1 ring of 64-byte pages between RISCV NC on the producer and RISCV B
//!   on the consumer (`tt_kernels::channel::Ring`): the producer writes a page and credits it, the
//!   consumer waits for it and credits it free, the producer waits for that. Two handoffs a page;
//!   the record is a handoff, half a round trip.
//! - **stream:** the same ring at depth 2 with 4 KiB pages, the consumer copying each page out
//!   before it frees it: a page's cost when credits overlap the data.
//! - **barrier:** both units' B arriving at `ITERATIONS` barriers coordinated by the producer's
//!   tile, polling its counter (the release word 0) and released by its multicast (1, a session's
//!   default).
//!
//! Every page the consumer is handed is checked against the producer's source after the runs.

#![cfg(feature = "silicon")]

use std::time::{Duration, Instant};

use tt_device::tlb::WindowKind;
use tt_device::Window;
use tt_isa::dm::{self, op, Mover};
use tt_isa::noc::niu::Niu;
use tt_isa::noc::Noc0;
use tt_kernels::channel::{self, Ring, Slots};
use tt_kernels::dm::{DataMover, L1Writes};
use tt_tests::backend::{device_index, GATE_TILE};
use tt_tests::bench::{on_card, pattern, report, Conditions, Stats, REPS};
use tt_tests::harness::{tile, Dev};

const PRODUCER: (u8, u8) = GATE_TILE;
const CONSUMER: (u8, u8) = (GATE_TILE.0 + 1, GATE_TILE.1);
/// Handoffs or barriers a sample: few enough that every list fits a mover's ring at once (at most
/// three entries an iteration, in chunks of `CHUNK`), which they must, since each is queued before
/// the movers are released.
const ITERATIONS: u32 = 100;
const CHUNK: usize = 128;
const _: () = assert!(3 * ITERATIONS as usize + 1 + CHUNK <= dm::LIST_MAX as usize);
const SRC: u32 = 0x2_0000;
const OUT: u32 = 0x4_0000;
const WINDOW_LEN: u64 = 64 << 10;

fn window() -> std::ops::Range<u64> {
    tt_isa::l1::DATA.end - WINDOW_LEN..tt_isa::l1::DATA.end
}
/// Each tile's start word, the consumer's `received`, the producer's `freed`.
fn start_word() -> u32 {
    (window().end - 96) as u32
}
fn received() -> u32 {
    (window().end - 64) as u32
}
fn freed() -> u32 {
    (window().end - 32) as u32
}

fn ring(depth: u16, page_bytes: u32) -> Ring {
    Ring {
        producer: PRODUCER,
        consumers: vec![CONSUMER],
        depth,
        page_bytes,
        slots: Slots::L1 {
            base: window().start as u32,
        },
        received: received(),
        freed: freed(),
    }
}

struct Pair {
    p_nc: DataMover<Noc0>,
    p_b: DataMover<Noc0>,
    c_b: DataMover<Noc0>,
    c_nc: DataMover<Noc0>,
}

fn start(d: &mut Dev<'_>, w: &Window, at: (u8, u8), mover: Mover) -> DataMover<Noc0> {
    let dram = d.dram_grid(w).unwrap();
    let t = tile(d, at.0, at.1);
    let image = match mover {
        Mover::B => tt_firmware_images::DM_B.1,
        _ => tt_firmware_images::DM_NC.1,
    };
    DataMover::start_on(d, w, t, &dram, mover, image).unwrap()
}

fn pair(d: &mut Dev<'_>, w: &Window) -> Pair {
    let mut p_nc = start(d, w, PRODUCER, Mover::NC);
    p_nc.set_l1_writes(L1Writes {
        tiles: vec![CONSUMER],
        window: window(),
    })
    .unwrap();
    p_nc.set_write_noc(d, w, Niu::Noc1).unwrap();
    let p_b = start(d, w, PRODUCER, Mover::B);
    let mut c_b = start(d, w, CONSUMER, Mover::B);
    c_b.set_l1_writes(L1Writes {
        tiles: vec![PRODUCER],
        window: window(),
    })
    .unwrap();
    let c_nc = start(d, w, CONSUMER, Mover::NC);
    Pair {
        p_nc,
        p_b,
        c_b,
        c_nc,
    }
}

fn gate() -> [u32; 8] {
    [op::WAIT_CREDIT, start_word(), 1, 0, 0, 0, 0, 0]
}

/// Queue `lists` on their movers (each list opening with the start gate), release both tiles and
/// time until every list is done.
fn timed(
    d: &mut Dev<'_>,
    w: &Window,
    lists: Vec<(&mut DataMover<Noc0>, Vec<[u32; 8]>)>,
) -> Duration {
    for at in [PRODUCER, CONSUMER] {
        let t = tile(d, at.0, at.1);
        channel::prepare(d, w, t, &[start_word(), received(), freed()]).unwrap();
    }
    let mut last = Vec::new();
    let mut movers = Vec::new();
    for (m, list) in lists {
        let mut n = m.enqueue(d, w, &[gate()]).unwrap();
        for chunk in list.chunks(CHUNK) {
            n = m.enqueue(d, w, chunk).unwrap();
        }
        last.push(n);
        movers.push(m);
    }
    let began = Instant::now();
    for at in [PRODUCER, CONSUMER] {
        let t = tile(d, at.0, at.1);
        d.write32(w, t, start_word() as u64, 1).unwrap();
    }
    for (m, n) in movers.into_iter().zip(last) {
        m.wait_for(d, w, n).unwrap();
    }
    began.elapsed()
}

fn handoffs(d: &mut Dev<'_>, w: &Window, p: &mut Pair, r: &Ring, copy: bool) -> Duration {
    let producer: Vec<[u32; 8]> = (0..ITERATIONS)
        .flat_map(|k| r.send(k, SRC + (k % 8) * r.page_bytes).unwrap())
        .collect();
    r.check_send(&producer).unwrap();
    let consumer: Vec<[u32; 8]> = (0..ITERATIONS)
        .flat_map(|k| {
            let mut e = vec![r.receive(k)];
            if copy {
                let words = r.page_bytes / 4;
                e.push([
                    op::COPY_WORDS,
                    r.slot(k),
                    OUT + (k % 8) * r.page_bytes,
                    words,
                    4,
                    4,
                    0,
                    0,
                ]);
            }
            e.push(r.free());
            e
        })
        .collect();
    timed(d, w, vec![(&mut p.p_nc, producer), (&mut p.c_b, consumer)])
}

fn barriers(d: &mut Dev<'_>, w: &Window, p: &mut Pair, release: bool) -> Duration {
    let leader = tile(d, PRODUCER.0, PRODUCER.1);
    d.write32(w, leader, dm::BARRIER_COUNTER, 0).unwrap();
    let rect = tt_isa::noc::multicast::corners_hi(CONSUMER, CONSUMER, Niu::Noc0);
    d.write32(w, leader, dm::BARRIER_RECTS, release as u32)
        .unwrap();
    d.write32(w, leader, dm::BARRIER_RECTS + 4, rect).unwrap();
    d.write32(w, leader, dm::BARRIER_RECTS + 8, 1).unwrap();
    for at in [PRODUCER, CONSUMER] {
        let t = tile(d, at.0, at.1);
        d.write32(w, t, dm::BARRIER_RELEASE, 0).unwrap();
    }
    let list: Vec<[u32; 8]> = (1..=ITERATIONS)
        .map(|i| {
            [
                op::BARRIER,
                2 * i,
                PRODUCER.0 as u32,
                PRODUCER.1 as u32,
                0,
                0,
                release as u32,
                0,
            ]
        })
        .collect();
    timed(d, w, vec![(&mut p.p_b, list.clone()), (&mut p.c_b, list)])
}

fn us_per(t: Duration, n: u32) -> f64 {
    t.as_secs_f64() * 1e6 / n as f64
}

#[test]
#[ignore = "benchmark"]
fn credit_handoff_against_the_barrier() {
    on_card(|d| {
        let t = tile(d, PRODUCER.0, PRODUCER.1);
        Conditions::measure(d, device_index(), t).print();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let mut p = pair(d, &w);
        let t = tile(d, PRODUCER.0, PRODUCER.1);
        let source = pattern(8 * 4096, 0x8);
        d.l1_write(&w, t, SRC as u64, &source).unwrap();
        let mut pingpong = Vec::new();
        let mut stream = Vec::new();
        let mut polled = Vec::new();
        let mut released = Vec::new();
        // One warm-up of each, then the samples.
        for rep in 0..=REPS {
            let a = handoffs(d, &w, &mut p, &ring(1, 64), false);
            let b = handoffs(d, &w, &mut p, &ring(2, 4096), true);
            let c = barriers(d, &w, &mut p, false);
            let e = barriers(d, &w, &mut p, true);
            if rep > 0 {
                pingpong.push(us_per(a, 2 * ITERATIONS));
                stream.push(us_per(b, ITERATIONS));
                polled.push(us_per(c, ITERATIONS));
                released.push(us_per(e, ITERATIONS));
            }
        }
        // The stream's last eight pages, copied out, are the source's.
        let mut back = vec![0u8; 8 * 4096];
        let c = tile(d, CONSUMER.0, CONSUMER.1);
        d.l1_read(&w, c, OUT as u64, &mut back).unwrap();
        assert!(back == source, "the stream's pages differ from the source");
        let timed = "host, release to both done";
        report("credit_handoff_64B", "us", timed, Stats::of(pingpong));
        report(
            "credit_stream_page_4KiB_depth2",
            "us",
            timed,
            Stats::of(stream),
        );
        report("barrier_two_units_polled", "us", timed, Stats::of(polled));
        report(
            "barrier_two_units_released",
            "us",
            timed,
            Stats::of(released),
        );
        for m in [p.p_nc, p.p_b, p.c_b, p.c_nc] {
            m.stop(d, &w).unwrap();
        }
    });
}
