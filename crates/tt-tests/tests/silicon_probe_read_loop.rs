//! Probe: how much of a mover read entry is the interpreter, and how much is
//! the NoC's in-flight limit?
//!
//! `cargo xtask silicon --release --include-ignored --filter silicon_probe_read_loop`
//!
//! Background: `mover_read_shapes` measures 0.48 us per request for 64 B and
//! 4 KiB reads alike, with the in-flight cap at 8 (`dm::TILE_IN_FLIGHT_CAP`). That is
//! what either an issue path that costs ~650 cycles per entry or a request that takes
//! ~3.8 us to come back, eight at a time (Little's law), would look like. The
//! interpreter (`Entry::decode`, `DramMove`, `exec`) only matters if it is the
//! first. These probes separate the two:
//!
//! 1. `interpreter_cap_sweep`: the shipped B image, unchanged, at caps 1 to 128.
//!    If the per-entry time falls as the cap rises, the entry is latency-bound
//!    and decode is not what limits it.
//! 2. `bare_loop_against_interpreter`: a B image with no decode at all
//!    (`read_loop_b`), at the same caps.
//!
//! Each asserts only that the bytes arrived: a loop that is fast because it moved
//! the wrong bytes measures nothing.

#![cfg(feature = "silicon")]

use std::time::{Duration, Instant};

use tt_device::tlb::WindowKind;
use tt_isa::dm::op;
use tt_kernels::dm::DataMover;
use tt_tests::bench::{median, on_card, pattern, report, Stats, REPS};
use tt_tests::harness::Dev;

const L1: u32 = 0x2_0000;
/// Destinations cycle through 256 KiB, as `mover_read_shapes`' do.
const L1_SPAN: u32 = 0x4_0000;
const TOTAL: u32 = 1 << 20;
/// Entries per list, as `mover_read_shapes`'.
const ENTRIES: u32 = 256;
const CAPS: [u32; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// `interpreter_cap_sweep`: `ENTRIES` one-port reads of `len` bytes through the shipped
/// mover, a list per cap, timed as `mover_read_shapes` times them (host clock
/// around `submit_list`/`wait`, less an empty list's time) and validated.
#[test]
#[ignore = "benchmark"]
fn interpreter_cap_sweep() {
    on_card(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tt_tests::harness::tensix_tile();
        let (_, image, _) = tt_firmware_images::DM_B;
        let mut m = DataMover::start(d, &w, t, &dram, image).unwrap();
        let chans: Vec<_> = dram.channels().collect();
        let base = 64u64 << 20;
        let data = pattern(TOTAL as usize, 7);
        d.dram_write(&w4, chans[0].range(base, TOTAL as u64).unwrap(), &data)
            .unwrap();
        let list = |len: u32| -> Vec<[u32; 8]> {
            (0..ENTRIES)
                .map(|i| {
                    [
                        op::READ,
                        chans[0].index() as u32,
                        0,
                        (base as u32) + i * len,
                        L1 + (i * len) % L1_SPAN,
                        len,
                        0,
                        0,
                    ]
                })
                .collect()
        };
        let time = |d: &mut Dev<'_>, m: &mut DataMover<tt_isa::noc::Noc0>, l: &[[u32; 8]]| {
            let t0 = Instant::now();
            m.submit_list(d, &w, l).unwrap();
            m.wait(d, &w).unwrap();
            t0.elapsed()
        };
        let empty_list = [[op::WAIT, 0, 0, 0, 0, 0, 0, 0]];
        for len in [64u32, 4096] {
            let reads = list(len);
            for cap in CAPS {
                m.set_in_flight_cap(d, &w, cap).unwrap();
                let empty = median((0..REPS).map(|_| time(d, &mut m, &empty_list)).collect());
                time(d, &mut m, &reads); // warm-up
                let samples: Vec<f64> = (0..REPS)
                    .map(|_| {
                        let dt = time(d, &mut m, &reads).saturating_sub(empty);
                        dt.as_secs_f64() * 1e6 / reads.len() as f64
                    })
                    .collect();
                report(
                    &format!("interpreter {len} B read, cap {cap}: per entry"),
                    "us",
                    "host",
                    Stats::of(samples),
                );
                // The bytes arrived: what the last 256 KiB of destinations hold.
                let mut back = vec![0u8; L1_SPAN as usize];
                d.l1_read(&w, t, L1 as u64, &mut back).unwrap();
                let span = (reads.len() as u32 * len).min(L1_SPAN);
                let from = (reads.len() as u32 * len - span) as usize;
                let tail = &data[from..from + span as usize];
                assert!(
                    back[..span as usize] == *tail,
                    "{len} B reads at cap {cap} did not land"
                );
            }
        }
        m.set_in_flight_cap(d, &w, tt_isa::dm::TILE_IN_FLIGHT_CAP)
            .unwrap();
        m.stop(d, &w).unwrap();
    });
}

/// The tile counter's rate, and the host-timed interpreter's conversion: 1,350 MHz
/// on card 0 (`bench::Conditions` records the card's own).
const CYCLES_PER_US: f64 = 1350.0;

/// `bare_loop_against_interpreter`: the same 256 one-port reads through `read_loop_b`
/// (no fetch, decode, validation or dispatch per request: the host encodes the
/// first request with `DramMove`, the image steps it or replays a table of them)
/// and through the shipped mover, at caps 8 (the default) and 128.
///
/// The probe is timed on the device (the tile's wall clock around each repetition,
/// which ends when the last read has landed); the interpreter by the host clock as in
/// `interpreter_cap_sweep`. The probe checksums its destination window after every
/// repetition and refills it with poison, and the host compares each sum and the last
/// window byte for byte with what the reads must have written, so a loop that is fast
/// because it moved nothing, or the wrong bytes, in any repetition, fails.
#[test]
#[ignore = "benchmark"]
fn bare_loop_against_interpreter() {
    use tt_isa::mailbox::{self, status as mbox};
    use tt_isa::noc::niu::{DramMove, Niu, TxnId};
    use tt_isa::noc::read_loop::{self as rl, mode};
    on_card(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tt_tests::harness::tensix_tile();
        let chans: Vec<_> = dram.channels().collect();
        let base = 64u64 << 20;
        let data = pattern(TOTAL as usize, 7);
        d.dram_write(&w4, chans[0].range(base, TOTAL as u64).unwrap(), &data)
            .unwrap();
        // The mover's own ID for B (`TXN` in mover.rs), and the coordinate it reads as.
        let txn = TxnId::new(2).unwrap();
        let me = (t.x() as u8, t.y() as u8);
        let request = |len: u32, i: u32| {
            let off = i * len;
            let range = chans[0].range(base + off as u64, len as u64).unwrap();
            let port = range.channel().port_for(Niu::Noc0, 0);
            DramMove::new(range, port, L1 + off % L1_SPAN, false, me, txn, Niu::Noc0)
                .unwrap()
                .words()
                .next()
                .unwrap()
        };
        let words = |r: tt_isa::noc::niu::DramRequest| {
            [r.targ, r.targ_hi, r.ret, r.ret_hi, r.tag, r.ctrl, r.len]
        };
        let names = ["stepped", "fast", "table", "decoded"];
        // One run of the probe image: `count` requests of `len` bytes per repetition in
        // `m`, at `cap`. Returns each repetition's cycles and window checksum, and how
        // many requests waited for room. The core is held in reset afterwards.
        let launch = |d: &mut Dev<'_>, m: u32, len: u32, cap: u32, count: u32| {
            d.l1_write(&w, t, L1 as u64, &vec![0xA5u8; L1_SPAN as usize])
                .unwrap();
            for (at, v) in [
                (rl::SEQ, 0),
                (rl::DONE, 0),
                (rl::MODE, m),
                (rl::COUNT, count),
                (rl::CAP, cap),
                (rl::TXN, 2),
                (rl::RET_END, L1 + L1_SPAN),
                (rl::REPS, REPS as u32),
            ] {
                d.write32(&w, t, at, v).unwrap();
            }
            for (k, v) in words(request(len, 0)).into_iter().enumerate() {
                d.write32(&w, t, rl::REQUEST + 4 * k as u64, v).unwrap();
            }
            d.write32(&w, t, rl::USABLE, dram.usable_mask() as u32)
                .unwrap();
            d.write32(&w, t, rl::ME_X, t.x() as u32).unwrap();
            d.write32(&w, t, rl::ME_Y, t.y() as u32).unwrap();
            if m == mode::TABLE || m == mode::DECODED {
                let mut table = Vec::new();
                for i in 0..count {
                    // The encoded request, or the shipped list entry it is decoded from.
                    let entry: [u32; 8] = if m == mode::TABLE {
                        let r = words(request(len, i));
                        [r[0], r[1], r[2], r[3], r[4], r[5], r[6], 0]
                    } else {
                        [
                            op::READ,
                            chans[0].index() as u32,
                            0,
                            (base as u32) + i * len,
                            L1 + (i * len) % L1_SPAN,
                            len,
                            0,
                            0,
                        ]
                    };
                    for v in entry {
                        table.extend_from_slice(&v.to_le_bytes());
                    }
                }
                d.l1_write(&w, t, rl::TABLE, &table).unwrap();
            }
            d.write32(&w, t, mailbox::STATUS, 0).unwrap();
            let (core, image, at) = tt_firmware_images::READ_LOOP_B;
            d.load_and_start(&w, t, core, image, at).unwrap();
            d.wait_for_status(&w, t, 1_000_000, |s| s == mbox::RUNNING)
                .unwrap()
                .expect("the probe did not start");
            d.write32(&w, t, rl::SEQ, 1).unwrap();
            let t0 = Instant::now();
            while d.read32(&w, t, rl::DONE).unwrap() != 1 {
                assert!(
                    t0.elapsed() < Duration::from_secs(5),
                    "the probe did not finish"
                );
            }
            let cycles: Vec<f64> = (0..REPS as u64)
                .map(|r| d.read32(&w, t, rl::CYCLES + 4 * r).unwrap() as f64)
                .collect();
            let checks: Vec<u32> = (0..REPS as u64)
                .map(|r| d.read32(&w, t, rl::CHECKS + 4 * r).unwrap())
                .collect();
            let stalls = d.read32(&w, t, rl::STALLS).unwrap();
            d.set_core_reset(&w, t, core, true).unwrap();
            (cycles, checks, stalls)
        };
        // What the window holds after a repetition in which the first `count` reads
        // landed: their bytes where they go, the poison everywhere else.
        let window = |len: u32, count: u32| {
            let mut expected = vec![0xA5u8; L1_SPAN as usize];
            for i in 0..count {
                let (from, to) = ((i * len) as usize, ((i * len) % L1_SPAN) as usize);
                expected[to..to + len as usize].copy_from_slice(&data[from..from + len as usize]);
            }
            let sum = expected.chunks_exact(4).fold(0u32, |a, c| {
                a.wrapping_add(u32::from_le_bytes(c.try_into().unwrap()))
            });
            (expected, sum)
        };
        for len in [64u32, 4096] {
            let (expected, sum) = window(len, ENTRIES);
            for cap in [8u32, 128] {
                for m in [mode::STEPPED, mode::FAST, mode::TABLE, mode::DECODED] {
                    let (cycles, checks, stalls) = launch(d, m, len, cap, ENTRIES);
                    for (r, c) in checks.iter().enumerate() {
                        assert_eq!(
                            *c, sum,
                            "{} {len} B reads at cap {cap}: repetition {r} did not land its data",
                            names[m as usize]
                        );
                    }
                    let mut back = vec![0u8; L1_SPAN as usize];
                    d.l1_read(&w, t, L1 as u64, &mut back).unwrap();
                    assert!(
                        back == expected,
                        "{} {len} B reads at cap {cap} did not land",
                        names[m as usize]
                    );
                    println!(
                        "MEASURE probe {} {len} B cap {cap}: {stalls} of {} requests waited for room",
                        names[m as usize],
                        ENTRIES * (REPS as u32 + 1)
                    );
                    report(
                        &format!(
                            "probe {} {len} B read, cap {cap}: per request",
                            names[m as usize]
                        ),
                        "cycles",
                        "device",
                        Stats::of(cycles.iter().map(|c| c / ENTRIES as f64)),
                    );
                }
            }
        }
        // The negative control: the same probe with its last request dropped must not
        // match the full window, in every repetition, in each mode. Without it
        // the checksum would only be shown to accept.
        let (_, full) = window(4096, ENTRIES);
        let (short, short_sum) = window(4096, ENTRIES - 1);
        assert_ne!(short, window(4096, ENTRIES).0);
        for m in [mode::STEPPED, mode::FAST, mode::TABLE, mode::DECODED] {
            let (_, checks, _) = launch(d, m, 4096, 8, ENTRIES - 1);
            assert!(
                checks.iter().all(|c| *c != full && *c == short_sum),
                "{}: a dropped request was not caught (or not the one dropped): {checks:x?}",
                names[m as usize]
            );
        }

        // The shipped mover, for the same reads.
        let (_, image, _) = tt_firmware_images::DM_B;
        let mut mv = DataMover::start(d, &w, t, &dram, image).unwrap();
        let empty_list = [[op::WAIT, 0, 0, 0, 0, 0, 0, 0]];
        for len in [64u32, 4096] {
            let reads: Vec<[u32; 8]> = (0..ENTRIES)
                .map(|i| {
                    [
                        op::READ,
                        chans[0].index() as u32,
                        0,
                        (base as u32) + i * len,
                        L1 + (i * len) % L1_SPAN,
                        len,
                        0,
                        0,
                    ]
                })
                .collect();
            for cap in [8u32, 128] {
                mv.set_in_flight_cap(d, &w, cap).unwrap();
                let mut time = |d: &mut Dev<'_>, l: &[[u32; 8]]| {
                    let t0 = Instant::now();
                    mv.submit_list(d, &w, l).unwrap();
                    mv.wait(d, &w).unwrap();
                    t0.elapsed()
                };
                let empty = median((0..REPS).map(|_| time(d, &empty_list)).collect());
                time(d, &reads);
                let samples: Vec<f64> = (0..REPS)
                    .map(|_| {
                        let dt = time(d, &reads).saturating_sub(empty);
                        dt.as_secs_f64() * 1e6 * CYCLES_PER_US / reads.len() as f64
                    })
                    .collect();
                report(
                    &format!("interpreter {len} B read, cap {cap}: per request"),
                    "cycles",
                    "host",
                    Stats::of(samples),
                );
            }
        }
        mv.set_in_flight_cap(d, &w, tt_isa::dm::TILE_IN_FLIGHT_CAP)
            .unwrap();
        mv.stop(d, &w).unwrap();
    });
}
