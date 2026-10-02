//! The data mover keeps its NoC requests in flight under a cap
//! (`tt_isa::noc::niu::InFlight`), so the 8-bit counter its completion wait
//! reads cannot wrap: with 256 in flight it would read 0, and a list would be
//! reported done with data still arriving.
//!
//! A list of more than 256 requests lands whole, and with the cap forced down
//! to one -- every request waits for the last -- it lands identically, with
//! the waits counted where the host reads them (`DataMover::throttle`). The
//! waits are asserted on silicon only: ttsim completes each request before the
//! next is issued (divergence row 72).

use tt_device::tlb::WindowKind;
use tt_isa::dm::op;
use tt_kernels::dm::DataMover;
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile};

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

/// More requests than the counter holds: one NoC request per entry.
const ENTRIES: u32 = 500;
const LEN: u32 = 1024;
/// Each entry's own L1 slot, so the check needs no ordering between them.
const L1_AT: u32 = 0x2_0000;
/// Where the pattern sits in each channel.
const DRAM_AT: u64 = 64 << 20;

#[test]
fn more_requests_than_the_counter_holds_all_land_with_and_without_waiting() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let mut m = DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap();
        let chans: Vec<_> = dram.channels().collect();
        let per = (ENTRIES as usize).div_ceil(chans.len()) * LEN as usize;
        let data: Vec<Vec<u8>> = chans
            .iter()
            .map(|c| {
                let p = pattern(per, c.index() as u32 + 1);
                d.dram_write(&w4, c.range(DRAM_AT, per as u64).unwrap(), &p)
                    .unwrap();
                p
            })
            .collect();
        // Entry i reads channel i % n, its (i / n)-th KiB, into slot i.
        let n = chans.len() as u32;
        let list: Vec<[u32; 8]> = (0..ENTRIES)
            .map(|i| {
                let off = DRAM_AT as u32 + (i / n) * LEN;
                [
                    op::READ,
                    chans[(i % n) as usize].index() as u32,
                    0,
                    off,
                    L1_AT + i * LEN,
                    LEN,
                    0,
                    0,
                ]
            })
            .collect();
        let check = |d: &mut tt_tests::harness::Dev<'_>, what: &str| {
            let mut back = vec![0u8; (ENTRIES * LEN) as usize];
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            for i in 0..ENTRIES as usize {
                let (c, k) = (i % n as usize, i / n as usize * LEN as usize);
                assert!(
                    back[i * LEN as usize..][..LEN as usize] == data[c][k..k + LEN as usize],
                    "{what}: entry {i} did not land"
                );
            }
        };
        let zero = vec![0u8; (ENTRIES * LEN) as usize];
        for (cap, what) in [
            (0u32, "the maximum cap"),
            (tt_isa::dm::TILE_IN_FLIGHT_CAP, "the tile movers' cap"),
            (1, "a cap of one"),
        ] {
            d.write(&w, t, L1_AT as u64, &zero).unwrap();
            m.set_in_flight_cap(d, &w, cap).unwrap();
            let before = m.throttle(d, &w).unwrap();
            m.run_list(d, &w, &list).unwrap();
            check(d, what);
            let th = m.throttle(d, &w).unwrap();
            println!(
                "{what}: {} stalls, {} cycles",
                th.stalls - before.stalls,
                th.cycles.wrapping_sub(before.cycles)
            );
            // On silicon every request after the first finds the one before
            // it still in flight. ttsim completes each request before the
            // next is issued (divergence row 72), so there the cap never
            // binds and only the data is checked.
            if cap == 1 && tt_tests::backend::ON_SILICON {
                assert!(
                    th.stalls > before.stalls,
                    "{what}: the throttle never engaged"
                );
            }
        }
        m.set_in_flight_cap(d, &w, tt_isa::dm::TILE_IN_FLIGHT_CAP)
            .unwrap();
        m.stop(d, &w).unwrap();
    });
}
