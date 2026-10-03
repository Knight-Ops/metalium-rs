//! RISCV NC's data mover (`tt_isa::dm::Mover::NC`, `dm_nc`): the same mover
//! as B's, on NC's own mailbox, list ring and scratch, so the two run side by
//! side on one tile.
//!
//! NC passes what `step16_dm` and `step49_in_flight` ask of B -- reads through
//! every channel and port, writes on either NoC, a refused descriptor answered,
//! more requests than the NIU counter holds -- and then B and NC move disjoint
//! data at the same time, B reading on NoC #0 while NC writes on NoC #1, the
//! reader / writer split (`docs/tt-metal-concepts-review.md`, G6).

use tt_device::tlb::WindowKind;
use tt_isa::dm::{self, op, Mover};
use tt_isa::dram::PORTS;
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

const L1_AT: u32 = 0x2_0000;
const DRAM_AT: u64 = 300 << 20;
/// Three NoC requests, the last short.
const LEN: usize = 32 * 1024 + 64;

fn start(d: &mut Dev<'_>, mover: Mover) -> DataMover<Noc0> {
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let dram = d.dram_grid(&w).unwrap();
    let t = tile(d, GATE_TILE.0, GATE_TILE.1);
    let image = match mover.core {
        tt_isa::tensix::Core::NC => tt_firmware_images::DM_NC.1,
        _ => tt_firmware_images::DM_B.1,
    };
    DataMover::start_on(d, &w, t, &dram, mover, image).unwrap()
}

#[test]
fn nc_reads_and_writes_every_channel_on_either_noc() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let mut m = start(d, Mover::NC);
        let t = m.tile();
        for (k, ch) in dram.channels().enumerate() {
            for port in 0..PORTS {
                let r = ch
                    .range(DRAM_AT + port as u64 * 0x10_0000, LEN as u64)
                    .unwrap();
                let data = pattern(LEN, (k * 3) as u32 + port as u32);
                d.dram_write(&w4, r, &data).unwrap();
                m.read(d, &w, r, port, L1_AT).unwrap();
                let mut back = vec![0u8; LEN];
                d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
                assert!(back == data, "NC read, channel {} port {port}", ch.index());
            }
            for noc in [WriteNoc::Noc1, WriteNoc::Noc0, WriteNoc::Alternate] {
                m.set_write_noc(d, &w, noc).unwrap();
                let r = ch.range(DRAM_AT + 0x40_0000, LEN as u64).unwrap();
                let data = pattern(LEN, k as u32 + 100);
                d.l1_write(&w, t, L1_AT as u64, &data).unwrap();
                m.write(d, &w, L1_AT, r, 1).unwrap();
                let mut back = vec![0u8; LEN];
                d.dram_read(&w4, r, &mut back).unwrap();
                assert!(back == data, "NC write {noc:?}, channel {}", ch.index());
            }
        }
        m.stop(d, &w).unwrap();
    });
}

#[test]
fn nc_refuses_a_bad_descriptor_and_survives() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let mut m = start(d, Mover::NC);
        let t = m.tile();
        let at = |word| Mover::NC.at(word);
        d.write32(&w, t, at(dm::OP), 7).unwrap();
        d.write32(&w, t, at(dm::LEN), 64).unwrap();
        let seq = d.read32(&w, t, at(dm::DONE)).unwrap() + 1;
        d.write32(&w, t, at(dm::SEQ), seq).unwrap();
        let mut polls = 0;
        while d.read32(&w, t, at(dm::DONE)).unwrap() != seq {
            d.tick(tt_device::core_control::CYCLES_PER_POLL);
            polls += 1;
            assert!(polls < 1_000_000, "NC's mover did not answer");
        }
        assert_eq!(d.read32(&w, t, at(dm::ERROR)).unwrap(), dm::error::OP);
        d.write32(&w, t, at(dm::SEQ), 0).unwrap();
        d.write32(&w, t, at(dm::DONE), 0).unwrap();
        let ch = dram.channel(0).unwrap();
        m.read(d, &w, ch.range(0x40, 64).unwrap(), 0, L1_AT)
            .unwrap();
        let e = m
            .read(d, &w, ch.range(0x10, 64).unwrap(), 0, L1_AT)
            .unwrap_err();
        assert!(matches!(e, DmError::Invalid(dm::error::ALIGNMENT)), "{e}");
        m.stop(d, &w).unwrap();
    });
}

/// More requests than the 8-bit counter holds, queued, on NC.
#[test]
fn nc_lands_more_requests_than_the_counter_holds() {
    const ENTRIES: u32 = 300;
    const SMALL: u32 = 1024;
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let mut m = start(d, Mover::NC);
        let t = m.tile();
        let ch = dram.channels().next().unwrap();
        let data = pattern((ENTRIES * SMALL) as usize, 5);
        d.dram_write(&w4, ch.range(DRAM_AT, data.len() as u64).unwrap(), &data)
            .unwrap();
        let list: Vec<[u32; 8]> = (0..ENTRIES)
            .map(|i| {
                let at = DRAM_AT as u32 + i * SMALL;
                [
                    op::READ,
                    ch.index() as u32,
                    i % 3,
                    at,
                    L1_AT + i * SMALL,
                    SMALL,
                    0,
                    0,
                ]
            })
            .collect();
        for cap in [0u32, dm::TILE_IN_FLIGHT_CAP, 1] {
            d.l1_write(&w, t, L1_AT as u64, &vec![0u8; data.len()])
                .unwrap();
            m.set_in_flight_cap(d, &w, cap).unwrap();
            let n = m.enqueue(d, &w, &list).unwrap();
            m.wait_for(d, &w, n).unwrap();
            let mut back = vec![0u8; data.len()];
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            assert!(back == data, "cap {cap}: the list did not land whole");
        }
        m.stop(d, &w).unwrap();
    });
}

/// B and NC on one tile at once: B reads a buffer in on NoC #0 while NC writes
/// another out on NoC #1, both queued before either is waited on. Each lands
/// whole, and neither mover's mailbox or ring disturbs the other's.
#[test]
fn b_reads_while_nc_writes_on_the_same_tile() {
    const N: usize = 12;
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let mut b = start(d, Mover::B);
        let mut nc = start(d, Mover::NC);
        let t = b.tile();
        nc.set_write_noc(d, &w, WriteNoc::Noc1).unwrap();
        let chans: Vec<_> = dram.channels().collect();
        let read_l1 = L1_AT;
        let write_l1 = L1_AT + (N * LEN) as u32;
        let src = pattern(N * LEN, 21);
        let out = pattern(N * LEN, 22);
        d.l1_write(&w, t, write_l1 as u64, &out).unwrap();
        d.l1_write(&w, t, read_l1 as u64, &vec![0u8; N * LEN])
            .unwrap();
        let range = |i: usize, base: u64| {
            chans[i % chans.len()]
                .range(base + (i * LEN) as u64, LEN as u64)
                .unwrap()
        };
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        for i in 0..N {
            let r = range(i, DRAM_AT);
            d.dram_write(&w4, r, &src[i * LEN..][..LEN]).unwrap();
            reads.push([
                op::READ,
                r.channel().index() as u32,
                (i % 3) as u32,
                r.offset() as u32,
                read_l1 + (i * LEN) as u32,
                LEN as u32,
                0,
                0,
            ]);
            let o = range(i, DRAM_AT + (64 << 20));
            writes.push([
                op::WRITE,
                o.channel().index() as u32,
                (i % 3) as u32,
                o.offset() as u32,
                write_l1 + (i * LEN) as u32,
                LEN as u32,
                0,
                0,
            ]);
        }
        let rn = b.enqueue(d, &w, &reads).unwrap();
        let wn = nc.enqueue(d, &w, &writes).unwrap();
        b.wait_for(d, &w, rn).unwrap();
        nc.wait_for(d, &w, wn).unwrap();
        let mut back = vec![0u8; N * LEN];
        d.l1_read(&w, t, read_l1 as u64, &mut back).unwrap();
        assert!(back == src, "B's reads did not land");
        for i in 0..N {
            let mut back = vec![0u8; LEN];
            d.dram_read(&w4, range(i, DRAM_AT + (64 << 20)), &mut back)
                .unwrap();
            assert!(back == out[i * LEN..][..LEN], "NC's write {i} did not land");
        }
        nc.stop(d, &w).unwrap();
        b.stop(d, &w).unwrap();
    });
}
