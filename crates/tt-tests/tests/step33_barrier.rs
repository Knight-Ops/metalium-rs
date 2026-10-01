//! Phase 10 gate (X4b): a barrier across data movers by NoC atomics.
//!
//! `tt_isa::dm::op::BARRIER`: each mover waits for its own moves, adds one to
//! a counter in the coordinating tile's L1 with a NoC atomic increment, and
//! polls the counter over the NoC until every unit has arrived. The claim:
//! work on one tile after a barrier sees what another tile did before it --
//! here a GDDR write by tile B read by tile A, A's list submitted first so
//! that without the barrier A reads the stale bytes (watched: it does) -- and
//! the counter counts every arrival of every barrier, so the `k`-th of a
//! session with `n` units completes at `k * n`.

use tt_device::tlb::WindowKind;
use tt_isa::dm::{self, op};
use tt_kernels::dm::DataMover;
use tt_tests::harness::{in_device, tensix_grid};

const L1_AT: u32 = 0x2_0000;
const LEN: u32 = 4096;

fn pattern(seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    (0..LEN)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as u8
        })
        .collect()
}

#[test]
fn work_after_a_barrier_sees_another_tile_s_work_before_it() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let grid = tensix_grid(d);
        let mut tiles = grid.tiles::<tt_isa::noc::Noc0>();
        let (ta, tb) = (tiles.next().unwrap(), tiles.next().unwrap());
        let (_, image, _) = tt_firmware_images::DM_B;
        let mut a = DataMover::start(d, &w, ta, &dram, image).unwrap();
        let mut b = DataMover::start(d, &w, tb, &dram, image).unwrap();
        // The coordinator is A; its counter starts at zero.
        d.write32(&w, ta, dm::BARRIER_COUNTER, 0).unwrap();
        let ch = dram.channels().next().unwrap();
        let barrier = |target: u32| {
            [
                op::BARRIER,
                target,
                ta.x() as u32,
                ta.y() as u32,
                0,
                0,
                0,
                0,
            ]
        };
        for round in 1..=3u32 {
            let region = ch
                .range(0x40_0000 + round as u64 * 0x1_0000, LEN as u64)
                .unwrap();
            d.dram_write(&w4, region, &vec![0u8; LEN as usize]).unwrap();
            let data = pattern(round);
            d.l1_write(&w, tb, L1_AT as u64, &data).unwrap();
            let entry = |o: u32| {
                [
                    o,
                    ch.index() as u32,
                    0,
                    region.offset() as u32,
                    L1_AT,
                    LEN,
                    0,
                    0,
                ]
            };
            // A: wait for B, then read what B wrote. Submitted first, so
            // without the barrier the read would see the zeros.
            a.submit_list(d, &w, &[barrier(2 * round), entry(op::READ)])
                .unwrap();
            b.submit_list(d, &w, &[entry(op::WRITE), barrier(2 * round)])
                .unwrap();
            a.wait(d, &w).unwrap();
            b.wait(d, &w).unwrap();
            let mut back = vec![0u8; LEN as usize];
            d.l1_read(&w, ta, L1_AT as u64, &mut back).unwrap();
            assert!(
                back == data,
                "round {round}: A read before B's write landed"
            );
            assert_eq!(
                d.read32(&w, ta, dm::BARRIER_COUNTER).unwrap(),
                2 * round,
                "every arrival counted"
            );
        }
        a.stop(d, &w).unwrap();
        b.stop(d, &w).unwrap();
    });
}
