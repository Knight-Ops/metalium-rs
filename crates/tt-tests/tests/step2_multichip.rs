//! Phase 1 gate, dual-chip: two chips, two `Device`s, one process.
//!
//! The gate proper is `the_same_tile_on_two_chips_holds_different_data`. Writing
//! to one chip and reading back from the same chip would pass even if the chip
//! were dropped from address formation entirely, since both halves would simply
//! go to whichever chip the transport really addressed. Using the *same tile
//! coordinate and the same address* on both chips is what makes the chip the only
//! variable.
//!
//! Runs against `libttsim_bh_x2.so`, which `cargo xtask fetch-ttsim` fetches
//! alongside the single-chip build. Integration tests are one binary per file and
//! every body forks, so the process-wide `Simulator` singleton is not a conflict:
//! this file's children load the dual-chip build and nothing else does.

// Drives the dual-chip simulator directly. The two-card silicon counterpart is `silicon_two_cards.rs`.
#![cfg(not(feature = "silicon"))]

use tt_device::{tlb::WindowKind, Bar, Device, Transport, Window};
use tt_isa::noc::{grid, ChipId, Noc0, NocCoord};
use tt_ttsim::{chip_bar_base, chip_bdf, fork_scope, Simulator};

type Dev<'a> = Device<tt_ttsim::LibTtsim<'a>>;

/// Two `Device`s, one per chip, in a forked child.
///
/// `sim.transports()` hands out both at once because `&mut self` is reborrowed
/// once for all of them; two separate accessor calls could not both be live,
/// which is the whole requirement here.
#[track_caller]
fn in_two_chips(f: impl FnOnce(&mut Dev<'_>, &mut Dev<'_>)) {
    let result = fork_scope(|| {
        let mut sim = Simulator::open_path(tt_ttsim::x2_lib_path())
            .unwrap_or_else(|e| panic!("could not open the dual-chip simulator: {e}"));
        // Before any memory access: an access aimed at a chip that is not there
        // lands in an unpopulated window and is fatal, not an error.
        assert_eq!(
            sim.chip_count(),
            2,
            "libttsim_bh_x2.so should present two chips"
        );

        let mut transports = sim.transports().into_iter();
        let mut a = Device::open(transports.next().unwrap()).unwrap_or_else(|e| panic!("{e}"));
        let mut b = Device::open(transports.next().unwrap()).unwrap_or_else(|e| panic!("{e}"));
        f(&mut a, &mut b);
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}

/// Geometry-only, which is safe *here* and nowhere else.
///
/// This file opens `libttsim_bh_x2.so` directly rather than going through the
/// backend, so it can only ever run against the simulator, and ttsim models
/// unharvested chips. Every other gate takes a [`grid::Tensix`] and asks the chip,
/// because on silicon a coordinate that passes this predicate can still be a
/// fused-off tile — and addressing one hangs the NoC rather than returning an
/// error. If this file is ever pointed at real cards, this helper has to change
/// with it.
fn tensix(x: u8, y: u8) -> NocCoord<Noc0> {
    assert!(
        grid::is_tensix_geometry(x, y),
        "({x},{y}) is not a Tensix tile"
    );
    NocCoord::new(x, y).unwrap()
}

/// Deterministic but not self-similar, so a misaddressed read is unlikely to
/// coincidentally match.
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
fn two_chips_are_present_and_report_their_own_ids() {
    in_two_chips(|a, b| {
        assert_eq!(a.chip(), ChipId(0));
        assert_eq!(b.chip(), ChipId(1));
        a.transport().verify_is_blackhole().unwrap();
        b.transport().verify_is_blackhole().unwrap();
    });
}

#[test]
fn chip_layout_matches_the_stride_formula() {
    // libttsim places device i's BARs at device 0's bases plus i * 64 GiB
    // (PER_DEVICE_PADDR_STRIDE), and reports a slot present only for bus 0,
    // function 0, device < chip count -- so the chip number is the bdf device
    // field. Both are read back from the library rather than assumed.
    // A free function rather than a loop over `[a, b]`: the two devices carry
    // distinct borrow lifetimes and `&mut` is invariant, so an array of them
    // does not typecheck.
    fn check_bases(dev: &mut Dev<'_>) {
        let chip = dev.chip();
        for bar in [Bar::Bar0, Bar::Bar2, Bar::Bar4] {
            assert_eq!(
                dev.transport().bar_base(bar).unwrap(),
                chip_bar_base(chip, bar),
                "chip {}'s {bar:?} base disagrees with the stride formula",
                chip.0
            );
        }
    }

    in_two_chips(|a, b| {
        check_bases(a);
        check_bases(b);
    });

    assert_eq!(chip_bdf(ChipId(0)), 0x00);
    assert_eq!(chip_bdf(ChipId(1)), 0x08, "bus 0, device 1, function 0");
    assert_eq!(chip_bar_base(ChipId(1), Bar::Bar0), 0x11_0000_0000);
    assert_eq!(chip_bar_base(ChipId(1), Bar::Bar2), 0x11_2000_0000);
    assert_eq!(chip_bar_base(ChipId(1), Bar::Bar4), 0x18_0000_0000);
}

#[test]
fn pattern_round_trips_through_two_windows_on_chip_1() {
    // The step 2 gate, repeated on the second chip alone. Proves chip 1 is
    // genuinely usable -- its TLB configuration array decodes, its windows
    // retarget, its L1 answers -- rather than merely present in config space.
    in_two_chips(|_a, b| {
        let write_window = b.alloc_window(WindowKind::TwoMib).unwrap();
        let read_window = b.alloc_window(WindowKind::TwoMib).unwrap();
        assert_ne!(write_window.index(), read_window.index());

        let tile = tensix(3, 4);
        let data = pattern(4096, 0x5EC0_2DEF);
        b.write(&write_window, tile, 0x2_0000, &data).unwrap();

        let mut back = vec![0u8; data.len()];
        b.read(&read_window, tile, 0x2_0000, &mut back).unwrap();
        assert_eq!(
            back, data,
            "chip 1's second window must see the first's writes"
        );
    });
}

#[test]
fn the_same_tile_on_two_chips_holds_different_data() {
    // The gate. Same coordinate, same address, different chips.
    in_two_chips(|a, b| {
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();

        let tile = tensix(3, 4);
        let from_a = pattern(1024, 0xA1);
        let from_b = pattern(1024, 0xB2);
        assert_ne!(from_a, from_b);

        a.write(&wa, tile, 0x2_0000, &from_a).unwrap();
        b.write(&wb, tile, 0x2_0000, &from_b).unwrap();

        let mut back_a = vec![0u8; from_a.len()];
        let mut back_b = vec![0u8; from_b.len()];
        a.read(&wa, tile, 0x2_0000, &mut back_a).unwrap();
        b.read(&wb, tile, 0x2_0000, &mut back_b).unwrap();

        assert_eq!(
            back_a, from_a,
            "chip 0's L1 was overwritten by a write aimed at chip 1"
        );
        assert_eq!(back_b, from_b, "chip 1 did not keep its own data");
    });
}

#[test]
fn the_same_window_index_on_both_chips_does_not_collide() {
    // The sharp version of the gate above: the window *configuration* writes,
    // not just the data, must land in separate apertures. Both devices keep
    // their own free list, so both hand out the same index first.
    in_two_chips(|a, b| {
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        assert_eq!(
            wa.index(),
            wb.index(),
            "window indices are per-chip, so both allocators start at the same one"
        );

        let tile_a = tensix(1, 2);
        let tile_b = tensix(16, 11);
        let from_a = pattern(256, 0xC3);
        let from_b = pattern(256, 0xD4);

        a.write(&wa, tile_a, 0x1_0000, &from_a).unwrap();
        b.write(&wb, tile_b, 0x1_0000, &from_b).unwrap();

        let mut back_a = vec![0u8; from_a.len()];
        let mut back_b = vec![0u8; from_b.len()];
        a.read(&wa, tile_a, 0x1_0000, &mut back_a).unwrap();
        b.read(&wb, tile_b, 0x1_0000, &mut back_b).unwrap();
        assert_eq!(back_a, from_a);
        assert_eq!(back_b, from_b);
    });
}

#[test]
fn every_tensix_tile_is_addressable_on_chip_1() {
    // The full-grid sweep from step 2, on the second chip. The only thing here
    // that would catch a chip 1 harvested differently from chip 0.
    //
    // `Tensix::FULL` is correct *here specifically*, and stated rather than
    // assumed: this gate drives the dual-chip simulator directly rather than going
    // through the backend, and ttsim models unharvested chips. On silicon the same
    // sweep must take each chip's grid from its own ARC, because two cards in one
    // host are two ASICs with independent fuses -- which is exactly the divergence
    // this gate claims to be looking for.
    in_two_chips(|_a, b| {
        let grid = grid::Tensix::FULL;
        let window: Window = b.alloc_window(WindowKind::TwoMib).unwrap();
        let tiles: Vec<_> = grid.tiles::<Noc0>().collect();
        assert_eq!(tiles.len(), grid.tile_count());

        for (i, tile) in tiles.iter().enumerate() {
            b.write32(&window, *tile, 0x3000, 0x7000_0000 | i as u32)
                .unwrap();
        }
        for (i, tile) in tiles.iter().enumerate() {
            assert_eq!(
                b.read32(&window, *tile, 0x3000).unwrap(),
                0x7000_0000 | i as u32,
                "tile {tile:?} on chip 1 aliases another tile"
            );
        }
    });
}

#[test]
fn a_third_chip_is_not_present() {
    // Guards the bdf walk against inventing chips: it scans every slot a bdf can
    // name rather than stopping at the first gap, so an over-eager probe would
    // show up here rather than as accesses into an unpopulated window.
    let result = fork_scope(|| {
        let sim = Simulator::open_path(tt_ttsim::x2_lib_path()).unwrap();
        assert_eq!(sim.chip_count(), 2);
        assert_eq!(sim.chips().map(|c| c.0).collect::<Vec<_>>(), vec![0, 1]);
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}
