//! Phase 2 on silicon: the NoC-visible local data RAM, the zeroing window, and
//! `DISABLE_RESET`.
//!
//! ttsim models none of this (divergence rows 25, 26 and 43; `step6_local_ram.rs`
//! watches it refuse), so these are the first measurements of it anywhere.
//!
//! # The rule these tests exist under
//!
//! A core's local data RAM does not answer the NoC while that core is held in
//! soft reset. The first version of this file wrote all five RAMs with every
//! core held -- the harness's resting state -- and the access never completed:
//! the NoC hung, the ARC watchdog reset the chip, and the host went down with the
//! PCIe link (2026-09-30). Tenstorrent's debugger encodes the same rule:
//! `tt-exalens`' `ensure_private_memory_access` never touches private memory with
//! the core in reset, and parks it in a `jal x0, 0` loop first.
//!
//! `tt-device` now makes the mistake unrepresentable -- `Device::read`/`write`
//! refuse the aperture, and `local_ram_read`/`write` refuse a core in reset and
//! wait out the post-release zeroing -- so every access here goes through
//! [`Device::park_core`] and those accessors.
//!
//! `DISABLE_RESET` outlives the process that sets it, so every test that writes
//! it puts it back to zero before returning, even on a panic.
//!
//! [`Device::park_core`]: tt_device::Device::park_core

#![cfg(feature = "silicon")]

use tt_device::tlb::WindowKind;
use tt_device::Window;
use tt_isa::noc::{Noc0, NocCoord};
use tt_isa::tensix::{self, Core};
use tt_tests::harness::{self, assert_on_silicon, in_device, Dev};

/// Where the parked cores' `j .` lives: clear of the firmware load address and
/// of the mailbox. RISCV B cannot be redirected and parks at 0.
const LOOP_ADDR: u64 = 0x4_0000;

fn park_address(core: Core) -> u64 {
    if core == Core::B {
        0
    } else {
        LOOP_ADDR
    }
}

fn measure(key: &str, value: impl std::fmt::Display) {
    println!("MEASURE {key} = {value}");
}

/// A pattern that differs per word and per core, so aliasing between cores is
/// visible.
fn pattern(core: Core, words: usize) -> Vec<u8> {
    (0..words as u32)
        .flat_map(|i| {
            (0xC0DE_0000 ^ ((core as u32) << 12) ^ i.wrapping_mul(0x9E37_79B9)).to_le_bytes()
        })
        .collect()
}

/// How many words of `core`'s RAM still hold its pattern, and how many are zero.
fn census(dev: &mut Dev<'_>, w: &Window, tile: NocCoord<Noc0>, core: Core) -> (usize, usize) {
    let bytes = core.local_data_ram_size() as usize;
    let want = pattern(core, bytes / 4);
    let mut got = vec![0u8; bytes];
    dev.local_ram_read(w, tile, core, 0, &mut got).unwrap();
    let kept = got
        .chunks_exact(4)
        .zip(want.chunks_exact(4))
        .filter(|(a, b)| a == b)
        .count();
    let zero = got.chunks_exact(4).filter(|c| c == &[0; 4]).count();
    (kept, zero)
}

/// Run `f`, then put `DISABLE_RESET` back to zero whether or not `f` panicked.
fn with_disable_reset_restored(dev: &mut Dev<'_>, f: impl FnOnce(&mut Dev<'_>)) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(dev)));
    let tile = harness::tensix_tile();
    let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
    dev.write32(&w, tile, tensix::DISABLE_RESET, 0).unwrap();
    dev.free_window(w);
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

/// The smallest claim the fix rests on: with T1 parked, one word of its RAM
/// round-trips. Run this first and alone.
#[test]
fn l0_one_word_on_a_parked_core() {
    assert_on_silicon();
    in_device(|dev| {
        let tile = harness::tensix_tile();
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        // The refusal first, on the chip itself: the harness holds every core,
        // so this must come back as an error without reaching the NoC.
        let mut word = [0u8; 4];
        let refused = dev.local_ram_read(&w, tile, Core::T1, 0, &mut word);
        assert!(
            refused.is_err(),
            "a read of a held core's RAM was not refused"
        );

        dev.park_core(&w, tile, Core::T1, LOOP_ADDR).unwrap();
        dev.local_ram_write(&w, tile, Core::T1, 0, &0x1234_5678u32.to_le_bytes())
            .unwrap();
        dev.local_ram_read(&w, tile, Core::T1, 0, &mut word)
            .unwrap();
        measure(
            "local_ram.T1.word0",
            format!("{:#010x}", u32::from_le_bytes(word)),
        );
        assert_eq!(u32::from_le_bytes(word), 0x1234_5678);
        dev.free_window(w);
    });
}

/// Every core's local RAM round-trips through the slow-path aperture while the
/// core is parked, and the five RAMs do not alias one another.
#[test]
fn l1_each_local_ram_round_trips_over_the_noc() {
    assert_on_silicon();
    in_device(|dev| {
        let tile = harness::tensix_tile();
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        for core in Core::ALL {
            dev.park_core(&w, tile, core, park_address(core)).unwrap();
        }
        for core in Core::ALL {
            let words = core.local_data_ram_size() as usize / 4;
            dev.local_ram_write(&w, tile, core, 0, &pattern(core, words))
                .unwrap();
        }
        // Checked only after every RAM is written, so a write that landed in
        // another core's RAM shows up as a mismatch there.
        for core in Core::ALL {
            let words = core.local_data_ram_size() as usize / 4;
            let (kept, _) = census(dev, &w, tile, core);
            measure(
                &format!("local_ram.{}.round_trip", core.name()),
                format!("{kept} of {words}"),
            );
            assert_eq!(
                kept,
                words,
                "{}'s local RAM did not round-trip",
                core.name()
            );
        }
        dev.free_window(w);
    });
}

/// `DISABLE_RESET` reads back what was written. ttsim refuses both directions
/// (row 26); the read-modify-write convention needs both. A tile debug register,
/// not the aperture.
#[test]
fn l2_disable_reset_round_trips() {
    assert_on_silicon();
    in_device(|dev| {
        with_disable_reset_restored(dev, |dev| {
            let tile = harness::tensix_tile();
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            let before = dev.read32(&w, tile, tensix::DISABLE_RESET).unwrap();
            measure("disable_reset.initial", format!("{before:#x}"));
            for value in [0x3FFu32, 0x155, 0x0] {
                dev.write32(&w, tile, tensix::DISABLE_RESET, value).unwrap();
                let back = dev.read32(&w, tile, tensix::DISABLE_RESET).unwrap();
                measure(
                    &format!("disable_reset.wrote_{value:#x}"),
                    format!("{back:#x}"),
                );
                assert_eq!(back & 0x3FF, value, "DISABLE_RESET did not read back");
            }
            dev.free_window(w);
        });
    });
}

/// The zeroing that follows a release, both ways: re-releasing T0 wipes its
/// local RAM unless its `DISABLE_RESET` bit is set.
///
/// Staged with the core parked, then put through reset and released again, and
/// read only afterwards -- nothing touches the RAM while the core is held. The
/// accessors wait out the zeroing before reading. The `pc` snapshot confirms T0
/// is back in the loop: its address or the word after, never anywhere else.
#[test]
fn l3_release_zeroes_local_ram_unless_disable_reset_is_set() {
    assert_on_silicon();
    in_device(|dev| {
        with_disable_reset_restored(dev, |dev| {
            let tile = harness::tensix_tile();
            let core = Core::T0;
            let (ram_bit, _) = core.disable_reset_bits();
            let words = core.local_data_ram_size() as usize / 4;
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();

            for (label, disable) in [("bit_clear", false), ("bit_set", true)] {
                dev.park_core(&w, tile, core, LOOP_ADDR).unwrap();
                dev.local_ram_write(&w, tile, core, 0, &pattern(core, words))
                    .unwrap();
                let (staged, _) = census(dev, &w, tile, core);
                assert_eq!(staged, words, "staging into T0's local RAM failed");

                let bits = if disable { 1 << ram_bit } else { 0 };
                dev.write32(&w, tile, tensix::DISABLE_RESET, bits).unwrap();
                // Through reset and out again; the reset PC still names the loop.
                dev.set_core_reset(&w, tile, core, true).unwrap();
                dev.set_core_reset(&w, tile, core, false).unwrap();

                let (kept, zero) = census(dev, &w, tile, core);
                // Speculative (`BabyRISCV/README.md:163-173`): the first silicon
                // run of a `j .` read loop + 4, the sequential fetch past the jump,
                // so the snapshot names the loop or the word after it, nothing else.
                let pcs: std::collections::BTreeSet<u32> = (0..16)
                    .map(|_| dev.read_pc_snapshot(&w, tile, core).unwrap())
                    .collect();
                measure(&format!("zeroing.{label}.pc_samples"), format!("{pcs:#x?}"));
                measure(
                    &format!("zeroing.{label}"),
                    format!("kept {kept}, zero {zero}, of {words}"),
                );
                let lo = LOOP_ADDR as u32;
                assert!(
                    pcs.iter().all(|&pc| pc == lo || pc == lo + 4),
                    "T0 is not running the loop at {lo:#x}: pc snapshots {pcs:#x?}"
                );
                if disable {
                    assert_eq!(
                        kept, words,
                        "DISABLE_RESET set, yet release changed local RAM"
                    );
                } else {
                    assert_eq!(zero, words, "release did not zero T0's local RAM");
                }
            }
            dev.free_window(w);
        });
    });
}
