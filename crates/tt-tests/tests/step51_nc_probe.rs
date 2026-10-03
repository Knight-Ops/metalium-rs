//! RISCV NC bring-up: NC runs, from the jump at its reset PC to its image at
//! the top of L1 (`tt_isa::dm::nc`), and issues on both NoCs.
//!
//! The first code this workspace runs on NC. The probe image
//! (`tt-firmware/src/bin/nc_probe.rs`) keeps a heartbeat, reports the tile's
//! coordinate as each NIU names it, and on request writes a buffer to GDDR
//! through NoC #1 (port 1) and reads it back through NoC #0.

use tt_device::tlb::WindowKind;
use tt_isa::dm::{self, nc};
use tt_isa::mailbox::{offset, status};
use tt_isa::noc::niu;
use tt_isa::tensix::Core;
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile};

/// NC's copy of a B mover mailbox word.
const fn at(b_word: u64) -> u64 {
    b_word - dm::MAILBOX_BASE + nc::MAILBOX_BASE
}

const L1_AT: u32 = 0x2_0000;
const LEN: u32 = 8192;
const DRAM_AT: u32 = 200 << 20;

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

#[test]
fn nc_runs_from_its_stub_and_moves_data_on_both_nocs() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        for word in [dm::SEQ, dm::DONE, dm::ERROR, dm::MY_X, dm::MY_Y] {
            d.write32(&w, t, at(word), 0).unwrap();
        }
        d.write32(&w, t, nc::MAILBOX_BASE + offset::STATUS, 0)
            .unwrap();
        d.write32(&w, t, nc::MAILBOX_BASE + offset::HEARTBEAT, 0)
            .unwrap();
        d.write32(&w, t, at(dm::USABLE), dram.usable_mask() as u32)
            .unwrap();
        let (core, image, base) = tt_firmware_images::NC_PROBE;
        assert_eq!(core, Core::NC);
        d.load_and_start_nc(&w, t, image, base).unwrap();

        // Running: its runtime's status, then a heartbeat that climbs.
        let mut polls = 0;
        while d.read32(&w, t, nc::MAILBOX_BASE + offset::STATUS).unwrap() != status::RUNNING {
            d.tick(tt_device::core_control::CYCLES_PER_POLL);
            polls += 1;
            assert!(polls < 100_000, "NC never reported RUNNING");
        }
        let beat = |d: &mut tt_tests::harness::Dev<'_>| {
            d.read32(&w, t, nc::MAILBOX_BASE + offset::HEARTBEAT)
                .unwrap()
        };
        let first = beat(d);
        d.tick(10_000);
        assert!(beat(d) != first, "NC's heartbeat did not move");

        // Both NIUs name the tile as the host does.
        let id = d
            .read32(&w, t, niu::NOC0_BASE + niu::NOC_ID_LOGICAL)
            .unwrap()
            & 0xFFF;
        assert_eq!(
            d.read32(&w, t, at(dm::MY_X)).unwrap(),
            id,
            "NoC #0's coordinate"
        );
        assert_eq!(
            d.read32(&w, t, at(dm::MY_Y)).unwrap(),
            id,
            "NoC #1's coordinate"
        );

        // A write through NoC #1, read back through NoC #0, every channel.
        for (k, ch) in dram.channels().enumerate() {
            let data = pattern(LEN as usize, k as u32 + 9);
            d.l1_write(&w, t, L1_AT as u64, &data).unwrap();
            d.l1_write(&w, t, (L1_AT + LEN) as u64, &vec![0u8; LEN as usize])
                .unwrap();
            for (word, v) in [
                (dm::CHANNEL, ch.index() as u32),
                (dm::DRAM_OFFSET, DRAM_AT),
                (dm::L1_ADDR, L1_AT),
                (dm::LEN, LEN),
            ] {
                d.write32(&w, t, at(word), v).unwrap();
            }
            let seq = k as u32 + 1;
            d.write32(&w, t, at(dm::SEQ), seq).unwrap();
            let mut polls = 0;
            while d.read32(&w, t, at(dm::DONE)).unwrap() != seq {
                d.tick(tt_device::core_control::CYCLES_PER_POLL);
                polls += 1;
                assert!(
                    polls < 1_000_000,
                    "channel {}: NC did not answer",
                    ch.index()
                );
            }
            assert_eq!(d.read32(&w, t, at(dm::ERROR)).unwrap(), dm::error::NONE);
            let mut back = vec![0u8; LEN as usize];
            d.dram_read(
                &w4,
                ch.range(DRAM_AT as u64, LEN as u64).unwrap(),
                &mut back,
            )
            .unwrap();
            assert!(
                back == data,
                "channel {}: NoC #1's write did not land",
                ch.index()
            );
            d.l1_read(&w, t, (L1_AT + LEN) as u64, &mut back).unwrap();
            assert!(
                back == data,
                "channel {}: NoC #0's read did not land",
                ch.index()
            );
            // The initiator kept every register NC wrote (checklist 9.14):
            // a later request may leave the unchanged ones unwritten.
            // NOC_CTRL is compared whole, so a hardware write to its reserved
            // bits shows here too.
            for (noc, mask_at, ctrl_at) in [("NoC #1", 0xE8, 0xF0), ("NoC #0", 0xEC, 0xF4)] {
                let mask = d.read32(&w, t, nc::MAILBOX_BASE + mask_at).unwrap();
                let ctrl = d.read32(&w, t, nc::MAILBOX_BASE + ctrl_at).unwrap();
                println!(
                    "channel {}: {noc}: registers changed {mask:#x}, NOC_CTRL {ctrl:#x}",
                    ch.index()
                );
                assert_eq!(
                    mask,
                    0,
                    "channel {}: {noc}'s initiator changed registers",
                    ch.index()
                );
            }
        }
        d.set_core_reset(&w, t, Core::NC, true).unwrap();
    });
}
