//! Phase 9 gate: a Tensix tile moves data between GDDR and its own L1.
//!
//! The `dm_b` data mover (`tt_isa::dm`) runs resident on the gate tile's RISCV B.
//! The host puts a pattern in DRAM, asks the mover to pull it into L1, and reads
//! L1; then the reverse. Every channel, every endpoint port, and transfers that
//! span several NoC requests. What crosses PCIe for each move is a descriptor,
//! and that is asserted, not assumed: it is the point of the phase.

use tt_device::tlb::WindowKind;
use tt_isa::dm;
use tt_isa::dram::{Dram, PORTS};
use tt_kernels::dm::{DataMover, DmError};
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

fn with_mover(f: impl FnOnce(&mut Dev<'_>, &mut DataMover<tt_isa::noc::Noc0>, &Dram)) {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let (_, image, _) = tt_firmware_images::DM_B;
        let mut mover = DataMover::start(d, &w, t, &dram, image).unwrap();
        f(d, &mut mover, &dram);
        mover.stop(d, &w).unwrap();
    });
}

#[test]
fn dram_reaches_l1_through_every_channel_and_port() {
    with_mover(|d, m, dram| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let t = m.tile();
        for ch in dram.channels() {
            for port in 0..PORTS {
                // 40 KiB: three NoC requests, the last one short.
                let len = 40 * 1024 + 96;
                let off = 0x20_0000 * (1 + port as u64) + 0x60;
                let data = pattern(len, ch.index() as u32 * 7 + port as u32);
                let r = ch.range(off, len as u64).unwrap();
                d.dram_write(&w4, r, &data).unwrap();
                let l1 = L1_AT + (off % tt_isa::dram::ALIGN) as u32;
                let before = d.traffic();
                m.read(d, &w, r, port, l1).unwrap();
                let moved = d.traffic() - before;
                let mut back = vec![0u8; len];
                d.l1_read(&w, t, l1 as u64, &mut back).unwrap();
                assert!(back == data, "channel {} port {port}", ch.index());
                // Seven descriptor words and nothing else went down; the data
                // did not cross PCIe.
                assert!(
                    moved.bytes_written <= 7 * 4 + 12 * moved.retargets,
                    "{moved:?}"
                );
            }
        }
    });
}

/// GDDR writes are RISCV NC's: B only reads.
#[test]
fn l1_reaches_dram_through_every_channel() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let mut m =
            DataMover::start_on(d, &w, t, &dram, dm::Mover::NC, tt_firmware_images::DM_NC.1)
                .unwrap();
        for ch in dram.channels() {
            let len = 33 * 1024 + 16;
            let data = pattern(len, 0xBEEF ^ ch.index() as u32);
            d.l1_write(&w, t, L1_AT as u64 + 16, &data).unwrap();
            let r = ch.range(0x300_0010, len as u64).unwrap();
            m.write(d, &w, L1_AT + 16, r, ch.index() % PORTS).unwrap();
            let mut back = vec![0u8; len];
            d.dram_read(&w4, r, &mut back).unwrap();
            assert!(back == data, "channel {}", ch.index());
        }
        m.stop(d, &w).unwrap();
    });
}

/// B refuses a write on the host and, if one is sent anyway, on the tile.
#[test]
fn b_does_not_write_gddr() {
    with_mover(|d, m, dram| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let t = m.tile();
        let ch = dram.channel(0).unwrap();
        let r = ch.range(0x300_0040, 64).unwrap();
        let before = pattern(64, 9);
        d.dram_write(&w4, r, &before).unwrap();
        d.l1_write(&w, t, L1_AT as u64, &pattern(64, 10)).unwrap();
        let e = m.write(d, &w, L1_AT, r, 0).unwrap_err();
        assert!(matches!(e, DmError::Invalid(dm::error::DIRECTION)), "{e}");
        // A raw descriptor the host did not check: the tile answers with the
        // same code and writes nothing.
        for (word, v) in [
            (dm::OP, dm::op::WRITE),
            (dm::CHANNEL, 0),
            (dm::PORT, 0),
            (dm::DRAM_OFFSET, 0x300_0040),
            (dm::L1_ADDR, L1_AT),
            (dm::LEN, 64),
        ] {
            d.write32(&w, t, word, v).unwrap();
        }
        let seq = d.read32(&w, t, dm::DONE).unwrap() + 1;
        d.write32(&w, t, dm::SEQ, seq).unwrap();
        let mut polls = 0;
        while d.read32(&w, t, dm::DONE).unwrap() != seq {
            d.tick(tt_device::core_control::CYCLES_PER_POLL);
            polls += 1;
            assert!(polls < 1_000_000, "the mover did not answer");
        }
        assert_eq!(d.read32(&w, t, dm::ERROR).unwrap(), dm::error::DIRECTION);
        let mut back = vec![0u8; 64];
        d.dram_read(&w4, r, &mut back).unwrap();
        assert!(back == before, "B's refused write landed");
        d.write32(&w, t, dm::SEQ, 0).unwrap();
        d.write32(&w, t, dm::DONE, 0).unwrap();
    });
}

#[test]
fn a_bad_descriptor_is_refused_on_both_sides_and_the_mover_survives() {
    with_mover(|d, m, dram| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let ch = dram.channel(0).unwrap();
        // The host refuses a read not congruent mod 32 without sending anything.
        let before = d.traffic();
        let e = m
            .read(d, &w, ch.range(0x10, 64).unwrap(), 0, L1_AT)
            .unwrap_err();
        assert!(matches!(e, DmError::Invalid(dm::error::ALIGNMENT)), "{e}");
        assert_eq!((d.traffic() - before).bytes_written, 0);

        // The firmware refuses one the host did not check: a raw descriptor with
        // an unknown op. It must answer, not hang.
        let t = m.tile();
        d.write32(&w, t, dm::OP, 7).unwrap();
        d.write32(&w, t, dm::LEN, 64).unwrap();
        let seq = d.read32(&w, t, dm::DONE).unwrap() + 1;
        d.write32(&w, t, dm::SEQ, seq).unwrap();
        let mut polls = 0;
        while d.read32(&w, t, dm::DONE).unwrap() != seq {
            d.tick(tt_device::core_control::CYCLES_PER_POLL);
            polls += 1;
            assert!(polls < 1_000_000, "the mover did not answer");
        }
        assert_eq!(d.read32(&w, t, dm::ERROR).unwrap(), dm::error::OP);
        // Put the sequence back in step with the host-side mover, then prove it
        // still works.
        d.write32(&w, t, dm::SEQ, 0).unwrap();
        d.write32(&w, t, dm::DONE, 0).unwrap();
        let r = ch.range(0x40, 64).unwrap();
        m.read(d, &w, r, 0, L1_AT).unwrap();
    });
}
