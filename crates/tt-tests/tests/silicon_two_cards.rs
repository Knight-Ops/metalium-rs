//! The Phase 1 multi-chip gate on real hardware: both p150a cards, one process.
//!
//! `step2_multichip.rs` proves this against ttsim's dual-chip build, where two
//! chips share one simulator and are told apart only by a stride in the physical
//! address. Here they are two PCIe devices with two BARs, two drivers' worth of
//! state and two independently fused ASICs, so "different data at the same
//! address" is a claim about the host side keeping two `Device`s apart -- windows,
//! shadow tables, cleanup writes -- not about the simulator's address decode.
//!
//! Each card's grid is read from its own ARC by `open_card`, and every tile used
//! is checked against the card it is on, because the two cards need not be fused
//! alike.

#![cfg(feature = "silicon")]

use tt_device::tlb::WindowKind;
use tt_tests::backend::{open_card, scrub};
use tt_tests::harness::{assert_on_silicon, tensix_grid, tile, Dev};
use tt_ttsim::fork_scope;

fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut x = seed | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as u8
        })
        .collect()
}

#[track_caller]
fn in_two_cards(f: impl FnOnce(&mut Dev<'_>, &mut Dev<'_>)) {
    assert_on_silicon();
    if let Err(e) = fork_scope(|| {
        let mut a = open_card(0);
        let mut b = open_card(1);
        f(&mut a, &mut b);
        scrub(&mut a);
        scrub(&mut b);
    }) {
        panic!("{e}");
    }
}

#[test]
fn two_cards_report_their_own_chip_ids() {
    in_two_cards(|a, b| {
        assert_eq!(a.chip().0, 0);
        assert_eq!(b.chip().0, 1);
        // Each card's grid came from its own ARC. Printed rather than compared:
        // the cards need not be fused alike, and a difference is information.
        println!(
            "MEASURE card0.tensix_columns = {}",
            tensix_grid(a).enabled_column_count()
        );
        println!(
            "MEASURE card1.tensix_columns = {}",
            tensix_grid(b).enabled_column_count()
        );
    });
}

#[test]
fn the_same_tile_on_two_cards_holds_different_data() {
    in_two_cards(|a, b| {
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let ta = tile(a, 3, 4);
        let tb = tile(b, 3, 4);
        let from_a = pattern(1024, 0xA1);
        let from_b = pattern(1024, 0xB2);

        a.write(&wa, ta, 0x2_0000, &from_a).unwrap();
        b.write(&wb, tb, 0x2_0000, &from_b).unwrap();

        let mut back_a = vec![0u8; from_a.len()];
        let mut back_b = vec![0u8; from_b.len()];
        a.read(&wa, ta, 0x2_0000, &mut back_a).unwrap();
        b.read(&wb, tb, 0x2_0000, &mut back_b).unwrap();
        assert_eq!(
            back_a, from_a,
            "card 0's L1 was overwritten by a write aimed at card 1"
        );
        assert_eq!(back_b, from_b, "card 1 did not keep its own data");
        a.free_window(wa);
        b.free_window(wb);
    });
}

#[test]
fn the_same_window_index_on_both_cards_does_not_collide() {
    in_two_cards(|a, b| {
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        assert_eq!(
            wa.index(),
            wb.index(),
            "window indices are per card, so both allocators start at the same one"
        );
        // Different tiles, so a configuration write that reached the wrong
        // card's TLB table would retarget the other window and show up as a
        // mismatch below.
        let far_x = tensix_grid(b).columns().last().unwrap();
        let ta = tile(a, 1, 2);
        let tb = tile(b, far_x, 11);
        let from_a = pattern(256, 0xC3);
        let from_b = pattern(256, 0xD4);

        a.write(&wa, ta, 0x1_0000, &from_a).unwrap();
        b.write(&wb, tb, 0x1_0000, &from_b).unwrap();

        let mut back_a = vec![0u8; from_a.len()];
        let mut back_b = vec![0u8; from_b.len()];
        a.read(&wa, ta, 0x1_0000, &mut back_a).unwrap();
        b.read(&wb, tb, 0x1_0000, &mut back_b).unwrap();
        assert_eq!(back_a, from_a);
        assert_eq!(back_b, from_b);
        a.free_window(wa);
        b.free_window(wb);
    });
}

#[test]
fn a_pattern_round_trips_through_two_windows_on_card_1() {
    in_two_cards(|_a, b| {
        let write_window = b.alloc_window(WindowKind::TwoMib).unwrap();
        let read_window = b.alloc_window(WindowKind::TwoMib).unwrap();
        assert_ne!(write_window.index(), read_window.index());
        let t = tile(b, 3, 4);
        let data = pattern(4096, 0x5EC0_2DEF);
        b.write(&write_window, t, 0x2_0000, &data).unwrap();
        let mut back = vec![0u8; data.len()];
        b.read(&read_window, t, 0x2_0000, &mut back).unwrap();
        assert_eq!(
            back, data,
            "card 1's second window must see the first's writes"
        );
        b.free_window(write_window);
        b.free_window(read_window);
    });
}
