//! Phase 2 on silicon: the NoC-visible local data RAM, the zeroing window, and
//! `DISABLE_RESET`.
//!
//! ttsim models none of this (divergence rows 25 and 26; `step6_local_ram.rs`
//! watches it refuse), so these are the first measurements of it anywhere. They
//! answer open question 6 for reset and zeroing, and the checklist's question
//! about the upper half of a T-core's slow-path window.
//!
//! The core under test runs a single instruction, `j .`, placed in L1. The
//! heartbeat firmware would do for "something is running", but it keeps its
//! stack in local data RAM, which is exactly what is being observed here; a loop
//! that touches no memory cannot confound the result. It also makes the `pc`
//! snapshot exact rather than speculative, which is a stronger form of step 3's
//! `pc_snapshot_lands_in_the_loaded_image`.
//!
//! `DISABLE_RESET` outlives the process that sets it, so every test that writes
//! it puts it back to zero before returning, and a panic is caught first so the
//! restore still happens.

#![cfg(feature = "silicon")]

use tt_device::tlb::WindowKind;
use tt_device::Window;
use tt_isa::noc::{Noc0, NocCoord};
use tt_isa::tensix::{self, Core};
use tt_tests::harness::{self, advance, assert_on_silicon, in_device, Dev};

/// Where the one-instruction loop lives: clear of the firmware load address and
/// of the mailbox.
const LOOP_ADDR: u64 = 0x4_0000;
/// `jal x0, 0`: jump to itself.
const J_SELF: u32 = 0x0000_006F;
/// More than the documented 2048-cycle zeroing window, at any plausible clock.
const ZEROING_WAIT_CYCLES: u32 = 1_000_000;

fn measure(key: &str, value: impl std::fmt::Display) {
    println!("MEASURE {key} = {value}");
}

/// A pattern that differs per word and per core, so aliasing between cores or
/// between halves of a window is visible.
fn pattern(core: Core, word: u64) -> u32 {
    0xC0DE_0000 ^ ((core as u32) << 12) ^ (word as u32).wrapping_mul(0x9E37_79B9)
}

fn fill(dev: &mut Dev<'_>, w: &Window, tile: NocCoord<Noc0>, core: Core, base: u64, words: u64) {
    for i in 0..words {
        dev.write32(w, tile, base + 4 * i, pattern(core, i))
            .unwrap();
    }
}

/// How many of `words` still hold the pattern, and how many are zero.
fn census(
    dev: &mut Dev<'_>,
    w: &Window,
    tile: NocCoord<Noc0>,
    core: Core,
    base: u64,
    words: u64,
) -> (u64, u64) {
    let (mut kept, mut zero) = (0, 0);
    for i in 0..words {
        let v = dev.read32(w, tile, base + 4 * i).unwrap();
        kept += u64::from(v == pattern(core, i));
        zero += u64::from(v == 0);
    }
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

/// Every core's local RAM round-trips through the slow-path aperture while the
/// core is held in reset, and the five RAMs do not alias one another.
#[test]
fn l1_each_local_ram_round_trips_over_the_noc() {
    assert_on_silicon();
    in_device(|dev| {
        let tile = harness::tensix_tile();
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        for core in Core::ALL {
            let words = u64::from(core.local_data_ram_size()) / 4;
            fill(
                dev,
                &w,
                tile,
                core,
                core.local_data_ram_noc_address(),
                words,
            );
        }
        // Checked only after every core's RAM is written, so a write that landed
        // in another core's RAM shows up as a mismatch there.
        for core in Core::ALL {
            let words = u64::from(core.local_data_ram_size()) / 4;
            let (kept, _) = census(
                dev,
                &w,
                tile,
                core,
                core.local_data_ram_noc_address(),
                words,
            );
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
/// (row 26); the read-modify-write convention needs both.
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

/// The zeroing window, both ways: releasing T0 wipes its local RAM unless its
/// `DISABLE_RESET` bit is set, in which case staged data survives.
///
/// This is the behaviour `load_and_start_staged` will rest on, measured before
/// the API is written. It also checks the loop is really what is running, by
/// asserting the `pc` snapshot is exactly the loop's address.
#[test]
fn l3_release_zeroes_local_ram_unless_disable_reset_is_set() {
    assert_on_silicon();
    in_device(|dev| {
        with_disable_reset_restored(dev, |dev| {
            let tile = harness::tensix_tile();
            let core = Core::T0;
            let (ram_bit, _) = core.disable_reset_bits();
            let base = core.local_data_ram_noc_address();
            let words = u64::from(core.local_data_ram_size()) / 4;
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();

            dev.write32(&w, tile, LOOP_ADDR, J_SELF).unwrap();
            for (label, disable) in [("bit_clear", false), ("bit_set", true)] {
                dev.set_core_reset(&w, tile, core, true).unwrap();
                dev.write32(
                    &w,
                    tile,
                    tensix::DISABLE_RESET,
                    if disable { 1 << ram_bit } else { 0 },
                )
                .unwrap();
                fill(dev, &w, tile, core, base, words);
                let (staged, _) = census(dev, &w, tile, core, base, words);
                assert_eq!(staged, words, "staging into T0's local RAM failed");

                dev.set_reset_pc(&w, tile, core, LOOP_ADDR as u32).unwrap();
                dev.set_core_reset(&w, tile, core, false).unwrap();
                advance(dev, ZEROING_WAIT_CYCLES);
                let pc = dev.read_pc_snapshot(&w, tile, core).unwrap();
                dev.set_core_reset(&w, tile, core, true).unwrap();

                let (kept, zero) = census(dev, &w, tile, core, base, words);
                measure(&format!("zeroing.{label}.pc"), format!("{pc:#x}"));
                measure(
                    &format!("zeroing.{label}"),
                    format!("kept {kept}, zero {zero}, of {words}"),
                );
                assert_eq!(pc, LOOP_ADDR as u32, "T0 is not running the loop");
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

/// What is in the upper 4 KiB of a T-core's 8 KiB slow-path window?
///
/// The memory map gives each T-core two identically-labelled 4 KiB rows while
/// its RAM is 4 KiB (`tt_isa::tensix::Core::local_data_ram_noc_window`). Three
/// candidates: an alias of the lower half, a reserved region that reads as a
/// constant, or something else. Written with a pattern distinct from the lower
/// half's, then both halves read back, which tells an alias from separate
/// storage.
///
/// **The one probe here that touches an undocumented address.** It is inside
/// the tile's documented debug aperture, but run it last and alone.
#[test]
fn l4_upper_half_of_a_t_core_window() {
    assert_on_silicon();
    in_device(|dev| {
        let tile = harness::tensix_tile();
        let core = Core::T1;
        let lower = core.local_data_ram_noc_address();
        let size = u64::from(core.local_data_ram_size());
        let upper = lower + size;
        assert!(upper + size <= lower + core.local_data_ram_noc_window());
        let words = size / 4;
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();

        let before: Vec<u32> = (0..8)
            .map(|i| dev.read32(&w, tile, upper + 4 * i).unwrap())
            .collect();
        measure("upper_half.first_words_before", format!("{before:08x?}"));

        fill(dev, &w, tile, core, lower, words);
        let (upper_matches_lower, _) = census(dev, &w, tile, core, upper, words);
        measure(
            "upper_half.equals_lower_after_writing_lower",
            format!("{upper_matches_lower} of {words}"),
        );

        // A different pattern into the upper half, attributed to NC so it
        // cannot match T1's.
        fill(dev, &w, tile, Core::NC, upper, words);
        let (upper_kept, upper_zero) = census(dev, &w, tile, Core::NC, upper, words);
        let (lower_kept, _) = census(dev, &w, tile, core, lower, words);
        measure(
            "upper_half.after_writing_upper",
            format!(
                "upper kept {upper_kept} (zero {upper_zero}), lower kept {lower_kept}, of {words}"
            ),
        );
        dev.free_window(w);
    });
}
