//! Probe: what a written tile costs RISCV NC's mover as a plain `WRITE` entry and as a tile of a
//! `WRITE_RUN` record, the writing counterpart of `silicon_probe_read_loop::records_against_entries`.
//!
//! `cargo xtask bench --device 1 --filter silicon_probe_nc_writes`
//!
//! Resident lists rerun from the sequence word (no upload), device cycles by the host clock less an empty
//! list's time, per tile, on NoC #1 (NC's writes, `WRITE_NOC`), for N = 64, 256 and 448 tiles of a tensor
//! spread over the eight channels. Before every repetition the destination range is filled with poison and
//! afterwards each tile's datums are compared with the L1 slot they came from, so a write that is cheap
//! because it moved nothing, or the wrong bytes, fails. It decides the scope of the NC write fast path
//! (`docs/plans`, phase 5): what a plain write entry costs against a record tile.
//!
//! `gddr_aggregate_nocs` (silicon_bench_memory) cannot do this: every one of its lists begins with a
//! `BARRIER`, which only B runs, so NC answers `DIRECTION`; a fleet-wide start for NC would need another
//! mechanism, and none of this needs one.

#![cfg(feature = "silicon")]

use std::time::{Duration, Instant};

use tt_device::tlb::WindowKind;
use tt_isa::dm::record::{self, TensorRef};
use tt_isa::dm::{self, op, Mover};
use tt_kernels::dm::{DataMover, WriteNoc};
use tt_tests::bench::{median, on_card, pattern, report, Stats, REPS};
use tt_tests::harness::Dev;

/// Source slots in L1: the writer's tile slots, 64 of them (0x2_0000 on, as the reads' destinations are).
const L1: u32 = 0x2_0000;
const SLOT: u32 = dm::TILE_SLOT as u32;
const DATA: u32 = dm::TILE_DATA as u32;
const DATUM_BYTES: usize = 4096;
const MAX_TILES: u32 = 448;
const CYCLES_PER_US: f64 = 1350.0;

#[test]
#[ignore = "benchmark"]
fn nc_writes_against_records() {
    on_card(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tt_tests::harness::tensix_tile();
        let chans: Vec<_> = dram.channels().collect();
        assert_eq!(chans.len(), 8, "the tensor spans all eight channels");
        let base = 128u32 << 20;
        let per_channel = (MAX_TILES / 8 + 1) * SLOT;
        // The source: slot s holds a pattern of its own, so a tile landing in the wrong place shows.
        let src: Vec<u8> = (0..64)
            .flat_map(|s| pattern(SLOT as usize, 100 + s))
            .collect();
        d.l1_write(&w, t, L1 as u64, &src).unwrap();
        let channels: [u8; 8] = core::array::from_fn(|c| chans[c].index());
        let x = TensorRef {
            n: 8,
            channels,
            base: [base; 8],
            first: 0,
            ct: 32,
        };
        // A tile's datums (past the 16-byte header, before the slot's padding) are what both forms write.
        let datums = |k: u32| {
            let at = ((k % 64) * SLOT + DATA) as usize;
            &src[at..at + DATUM_BYTES]
        };
        let plain = |n: u32| -> Vec<[u32; 8]> {
            (0..n)
                .map(|k| {
                    [
                        op::WRITE,
                        chans[(k % 8) as usize].index() as u32,
                        0,
                        base + (k / 8) * SLOT + DATA,
                        L1 + (k % 64) * SLOT + DATA,
                        DATUM_BYTES as u32,
                        0,
                        0,
                    ]
                })
                .collect()
        };
        let records = |n: u32| -> Vec<[u32; 8]> {
            (0..n / 32)
                .flat_map(|k| {
                    let [meta, bases] = x.encode();
                    [
                        [
                            record::WRITE_RUN,
                            32 * k,
                            32,
                            L1 + (k % 2) * 32 * SLOT,
                            0,
                            0,
                            0,
                            0,
                        ],
                        meta,
                        bases,
                    ]
                })
                .collect()
        };
        let image = tt_firmware_images::DM_NC.1;
        let empty_list = [[op::WAIT, 0, 0, 0, 0, 0, 0, 0]];
        for n in [64u32, 256, 448] {
            for (name, as_records) in [("plain WRITE", false), ("WRITE_RUN record", true)] {
                let list = if as_records { records(n) } else { plain(n) };
                let mut m = DataMover::start_on(d, &w, t, &dram, Mover::NC, image).unwrap();
                m.set_write_noc(d, &w, WriteNoc::Noc1).unwrap();
                let mover = m.mover();
                m.submit_list(d, &w, &empty_list).unwrap();
                m.wait(d, &w).unwrap();
                let rerun = |d: &mut Dev<'_>| {
                    let seq = d
                        .read32(&w, t, mover.at(dm::SEQ))
                        .unwrap()
                        .wrapping_add(1)
                        .max(1);
                    let t0 = Instant::now();
                    d.write32(&w, t, mover.at(dm::SEQ), seq).unwrap();
                    while d.read32(&w, t, mover.at(dm::DONE)).unwrap() != seq {
                        assert!(
                            t0.elapsed() < Duration::from_secs(5),
                            "NC's mover did not finish"
                        );
                    }
                    t0.elapsed()
                };
                rerun(d);
                let empty = median((0..REPS).map(|_| rerun(d)).collect());
                m.submit_list(d, &w, &list).unwrap();
                m.wait(d, &w).unwrap();
                rerun(d);
                let range_len = per_channel as u64;
                let poison = vec![0xA5u8; per_channel as usize];
                let samples: Vec<f64> = (0..REPS)
                    .map(|_| {
                        for ch in &chans {
                            d.dram_write(&w4, ch.range(base as u64, range_len).unwrap(), &poison)
                                .unwrap();
                        }
                        let dt = rerun(d).saturating_sub(empty);
                        let mut back: Vec<Vec<u8>> = Vec::new();
                        for ch in &chans {
                            let mut b = vec![0u8; per_channel as usize];
                            d.dram_read(&w4, ch.range(base as u64, range_len).unwrap(), &mut b)
                                .unwrap();
                            back.push(b);
                        }
                        for k in 0..n {
                            let at = ((k / 8) * SLOT) as usize;
                            let got = &back[(k % 8) as usize][at + 16..at + 16 + DATUM_BYTES];
                            let want = datums(k);
                            let bad = got.iter().zip(want).filter(|(a, b)| a != b).count();
                            assert!(
                                bad == 0,
                                "{name} N={n}: tile {k} did not land: {bad} of {} bytes differ, first at {:?}",
                                want.len(),
                                got.iter().zip(want).position(|(a, b)| a != b)
                            );
                            assert!(
                                back[(k % 8) as usize][at..at + 16]
                                    .iter()
                                    .all(|&b| b == 0xA5),
                                "{name} N={n}: tile {k} wrote its header"
                            );
                        }
                        dt.as_secs_f64() * 1e6 * CYCLES_PER_US / n as f64
                    })
                    .collect();
                // The list the card ran is the form under test.
                let first = d.read32(&w, t, mover.list).unwrap();
                assert_eq!(
                    first,
                    if as_records {
                        record::WRITE_RUN
                    } else {
                        op::WRITE
                    }
                );
                report(
                    &format!("NC {name}, N={n}: per tile"),
                    "cycles",
                    "host",
                    Stats::of(samples),
                );
                m.stop(d, &w).unwrap();
            }
        }
    });
}
