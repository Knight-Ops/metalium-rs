//! B's and NC's movers on one tile wait for each other through their progress
//! words (`tt_isa::dm::op::SIGNAL`, `WAIT_PEER`): the ordering the
//! reader / writer split needs, when NC writes out what B must read next, or
//! B must not overwrite what NC is still writing out.
//!
//! B's list is queued before NC's and waits on it, so B's read sees NC's write
//! only if the wait holds. And a peer that fails ends the wait with
//! `dm::error::PEER` rather than leaving the waiter spinning.

use tt_device::tlb::WindowKind;
use tt_isa::dm::{self, op, Mover};
use tt_isa::noc::Noc0;
use tt_kernels::dm::{DataMover, DmError, WriteNoc};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile, Dev};

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

const LEN: u32 = 16384;
const OUT_L1: u32 = 0x2_0000;
const IN_L1: u32 = 0x4_0000;
const DRAM_AT: u32 = 400 << 20;
/// Where NC's busy work lands.
const BUSY_L1: u32 = 0x6_0000;

fn movers(d: &mut Dev<'_>) -> (DataMover<Noc0>, DataMover<Noc0>) {
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let dram = d.dram_grid(&w).unwrap();
    let t = tile(d, GATE_TILE.0, GATE_TILE.1);
    let b = DataMover::start_on(d, &w, t, &dram, Mover::B, tt_firmware_images::DM_B.1).unwrap();
    let nc = DataMover::start_on(d, &w, t, &dram, Mover::NC, tt_firmware_images::DM_NC.1).unwrap();
    (b, nc)
}

fn wait_peer(peer: Mover, target: u32) -> [u32; 8] {
    let p = if peer == Mover::NC { 1 } else { 0 };
    [op::WAIT_PEER, p, target, 0, 0, 0, 0, 0]
}
const SIGNAL: [u32; 8] = [op::SIGNAL, 0, 0, 0, 0, 0, 0, 0];

#[test]
fn b_reads_what_nc_wrote_once_nc_signals() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let (mut b, mut nc) = movers(d);
        let t = b.tile();
        nc.set_write_noc(d, &w, WriteNoc::Noc1).unwrap();
        let ch = dram.channels().next().unwrap();
        let r = ch.range(DRAM_AT as u64, LEN as u64).unwrap();
        for round in 1..=3u32 {
            d.dram_write(&w4, r, &vec![0u8; LEN as usize]).unwrap();
            d.l1_write(&w, t, IN_L1 as u64, &vec![0xEEu8; LEN as usize])
                .unwrap();
            let data = pattern(LEN as usize, round);
            d.l1_write(&w, t, OUT_L1 as u64, &data).unwrap();
            let read = [op::READ, ch.index() as u32, 0, DRAM_AT, IN_L1, LEN, 0, 0];
            let write = [op::WRITE, ch.index() as u32, 1, DRAM_AT, OUT_L1, LEN, 0, 0];
            // B first: it must wait for NC's `round`-th signal.
            let bn = b
                .enqueue(d, &w, &[wait_peer(Mover::NC, round), read])
                .unwrap();
            // NC works a while before it writes: 64 reads into a scratch
            // slot. Without B's wait, B's read is long done by then.
            let busy = [
                op::READ,
                ch.index() as u32,
                0,
                DRAM_AT + LEN,
                BUSY_L1,
                LEN,
                0,
                0,
            ];
            let mut nc_list = vec![busy; 64];
            nc_list.extend([write, SIGNAL]);
            let nn = nc.enqueue(d, &w, &nc_list).unwrap();
            nc.wait_for(d, &w, nn).unwrap();
            b.wait_for(d, &w, bn).unwrap();
            let mut back = vec![0u8; LEN as usize];
            d.l1_read(&w, t, IN_L1 as u64, &mut back).unwrap();
            assert!(
                back == data,
                "round {round}: B read before NC's write landed"
            );
            assert_eq!(d.read32(&w, t, Mover::NC.at(dm::PROGRESS)).unwrap(), round);
        }
        nc.stop(d, &w).unwrap();
        b.stop(d, &w).unwrap();
    });
}

/// NC's queue stops on an error the host cannot see coming (a list written
/// to its ring directly, naming an op the mover refuses); B, waiting for a
/// signal that will never come, reports `PEER` instead of spinning.
#[test]
fn a_failed_peer_ends_the_wait() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let (mut b, nc) = movers(d);
        let t = b.tile();
        let bn = b.enqueue(d, &w, &[wait_peer(Mover::NC, 1)]).unwrap();
        // NC's list, by hand: one entry with an unknown op, in slot 0.
        let bad: Vec<u8> = [99u32, 0, 0, 0, 0, 0, 0, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let m = Mover::NC;
        d.l1_write(&w, t, m.list, &bad).unwrap();
        d.write32(&w, t, m.at(dm::QUEUE_SLOTS), dm::queue_slot(0, 1))
            .unwrap();
        d.write32(&w, t, m.at(dm::QUEUE_HEAD), 1).unwrap();
        let e = b.wait_for(d, &w, bn).unwrap_err();
        assert!(
            matches!(
                e,
                DmError::Queued {
                    code: dm::error::PEER,
                    ..
                }
            ),
            "{e}"
        );
        assert_eq!(
            d.read32(&w, t, m.at(dm::QUEUE_ERROR)).unwrap(),
            dm::error::OP
        );
        nc.stop(d, &w).unwrap();
        b.stop(d, &w).unwrap();
    });
}

/// Malformed waits and signals are refused before they reach a mover.
#[test]
fn malformed_peer_entries_are_refused_on_the_host() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let (mut b, nc) = movers(d);
        for bad in [
            [op::WAIT_PEER, 2, 1, 0, 0, 0, 0, 0],
            [op::WAIT_PEER, 1, 1, 5, 0, 0, 0, 0],
            [op::SIGNAL, 1, 0, 0, 0, 0, 0, 0],
        ] {
            let e = b.enqueue(d, &w, &[bad]).unwrap_err();
            assert!(matches!(e, DmError::Invalid(dm::error::OP)), "{bad:?}: {e}");
        }
        nc.stop(d, &w).unwrap();
        b.stop(d, &w).unwrap();
    });
}
