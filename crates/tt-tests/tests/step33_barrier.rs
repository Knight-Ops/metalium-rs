//! Phase 10 gate (X4b): a barrier across data movers by NoC atomics.
//!
//! `tt_isa::dm::op::BARRIER`: each mover waits for its own moves, adds one to
//! a counter in the coordinating tile's L1 with a NoC atomic increment, and
//! polls the counter over the NoC until every unit has arrived. The claim:
//! work on one tile after a barrier sees what another tile did before it --
//! here a GDDR write by tile B's NC (the writer) read by tile A, A's list
//! submitted first so that without the barrier A reads the stale bytes
//! (watched: it does) -- and the counter counts every arrival of every
//! barrier, so the `k`-th of a session with `n` units completes at `k * n`.
//! B's barrier is its packet's trailer, run once NC's write is acknowledged.

use tt_device::tlb::WindowKind;
use tt_isa::dataflow::{Action, Stream};
use tt_isa::dm::{self, op};
use tt_kernels::dm::{DataMover, WriteNoc};
use tt_tests::harness::{in_device, tensix_grid};

const L1_AT: u32 = 0x2_0000;
const LEN: u32 = 4096;

fn credit(action: Action) -> [u32; 8] {
    [
        op::CB,
        Stream::Transfer.word(),
        action.word(),
        1,
        0,
        0,
        0,
        0,
    ]
}

/// A one-batch transfer packet: B's side reads `reads` into L1, NC's writes
/// `writes` out; `trailer` (a barrier) runs on B once NC is done.
fn packet(reads: &[[u32; 8]], writes: &[[u32; 8]], trailer: Option<[u32; 8]>) -> Vec<[u32; 8]> {
    let mut reader = vec![credit(Action::Reserve)];
    reader.extend_from_slice(reads);
    reader.push(credit(Action::Push));
    let mut writer = vec![credit(Action::Wait)];
    writer.extend_from_slice(writes);
    writer.push(credit(Action::Pop));
    let mut p = vec![[
        op::PAIR,
        reader.len() as u32,
        writer.len() as u32,
        1,
        trailer.is_some() as u32,
        0,
        0,
        0,
    ]];
    p.extend(reader);
    p.extend(writer);
    p.extend(trailer);
    p
}

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
        let nc = DataMover::start_on(d, &w, tb, &dram, dm::Mover::NC, tt_firmware_images::DM_NC.1)
            .unwrap();
        nc.set_write_noc(d, &w, WriteNoc::Noc1).unwrap();
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
            let n = b
                .enqueue(
                    d,
                    &w,
                    &packet(&[], &[entry(op::WRITE)], Some(barrier(2 * round))),
                )
                .unwrap();
            a.wait(d, &w).unwrap();
            b.wait_for(d, &w, n).unwrap();
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
        nc.stop(d, &w).unwrap();
        b.stop(d, &w).unwrap();
    });
}

/// X4a: lists queued without waiting run in order. Forty lists -- more than
/// the sixteen slots, and more entries than the ring holds, so the host
/// waits for room -- each copy the previous list's output region to the next
/// through L1; the last region is the first's bytes only if every list ran,
/// and in order.
#[test]
fn queued_lists_run_in_order_without_waiting() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tensix_grid(d).tiles::<tt_isa::noc::Noc0>().next().unwrap();
        let (_, image, _) = tt_firmware_images::DM_B;
        let mut m = DataMover::start(d, &w, t, &dram, image).unwrap();
        let nc = DataMover::start_on(d, &w, t, &dram, dm::Mover::NC, tt_firmware_images::DM_NC.1)
            .unwrap();
        nc.set_write_noc(d, &w, WriteNoc::Noc1).unwrap();
        let ch = dram.channels().next().unwrap();
        let region = |k: u32| ch.range(0x80_0000 + k as u64 * 0x2000, LEN as u64).unwrap();
        let data = pattern(99);
        d.dram_write(&w4, region(0), &data).unwrap();
        let lists = 40u32;
        for k in 1..=lists {
            d.dram_write(&w4, region(k), &vec![0u8; LEN as usize])
                .unwrap();
        }
        let mut last = 0;
        for k in 0..lists {
            let (from, to) = (region(k), region(k + 1));
            let e = |o: u32, r: tt_isa::dram::DramRange| {
                [o, ch.index() as u32, 0, r.offset() as u32, L1_AT, LEN, 0, 0]
            };
            // B reads, NC writes (a transfer packet, which B reports done once
            // NC's write is acknowledged: what orders the next list after
            // it). Padded with waits, so forty lists overrun the ring's 512
            // entries.
            let mut reads = vec![e(op::READ, from)];
            reads.extend(std::iter::repeat_n([op::WAIT, 0, 0, 0, 0, 0, 0, 0], 20));
            last = m
                .enqueue(d, &w, &packet(&reads, &[e(op::WRITE, to)], None))
                .unwrap();
        }
        m.drain(d, &w).unwrap();
        assert_eq!(last, lists);
        assert!(!m.busy());
        let mut back = vec![0u8; LEN as usize];
        d.dram_read(&w4, region(lists), &mut back).unwrap();
        assert!(back == data, "the chain broke");
        nc.stop(d, &w).unwrap();
        m.stop(d, &w).unwrap();
    });
}
