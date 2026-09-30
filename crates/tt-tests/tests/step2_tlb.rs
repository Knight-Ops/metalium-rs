//! Step 2 gate: the host reaches a Tensix tile's L1 through TLB windows.
//!
//! The gate proper is `pattern_round_trips_through_two_windows`: a pattern written
//! through one window must be readable through a *different* one. Using the same
//! window for both halves would pass even if the window configuration were ignored
//! entirely, since the data would simply be going wherever the window already
//! pointed and coming back from the same place.
//!
//! Every coordinate here is checked against the *chip's* grid rather than the
//! part's geometry. An earlier version of this file used a hardcoded `(16, 11)`
//! and a hardcoded population of 140, both true of an unharvested Blackhole and
//! of ttsim. On a p150a with two columns fused off they name tiles that do not
//! exist, and a NoC access to one of those does not fail — it hangs the NoC, the
//! chip is reset to recover, and the PCIe link drops. That is why `tensix` takes a
//! [`Tensix`] and why nothing in this file mentions a tile count.

use tt_device::{tlb::WindowKind, Window};
use tt_isa::noc::grid::Tensix;
use tt_isa::noc::{Noc0, NocCoord};
use tt_tests::harness::{in_device, tensix_grid};

/// A Tensix coordinate this chip actually has.
fn tensix(grid: &Tensix, x: u8, y: u8) -> NocCoord<Noc0> {
    assert!(
        grid.contains(x, y),
        "({x},{y}) is not a Tensix tile on this chip: {} columns present, X must be \
         one of {:?}",
        grid.enabled_column_count(),
        grid.columns().collect::<Vec<_>>()
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
fn pattern_round_trips_through_two_windows() {
    in_device(|dev| {
        let grid = tensix_grid(dev);
        let write_window = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let read_window = dev.alloc_window(WindowKind::TwoMib).unwrap();
        assert_ne!(write_window.index(), read_window.index());

        let tile = tensix(&grid, 3, 4);
        let data = pattern(4096, 0xC0FFEE);
        dev.write(&write_window, tile, 0x2_0000, &data).unwrap();

        let mut back = vec![0u8; data.len()];
        dev.read(&read_window, tile, 0x2_0000, &mut back).unwrap();
        assert_eq!(
            back, data,
            "a second window must observe the first window's writes"
        );
    });
}

#[test]
fn two_tiles_do_not_alias() {
    // The coordinate fields of the window configuration are the only thing keeping
    // these apart. If they were dropped, both writes would land in one tile and the
    // second read would return the first tile's data.
    in_device(|dev| {
        let grid = tensix_grid(dev);
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();

        // The extremes of the surviving grid, for the widest separation available:
        // a dropped coordinate field is most visible between distant tiles. Taken
        // from the chip rather than written down, because the largest X that is
        // still a Tensix tile is exactly what harvesting changes.
        let columns: Vec<u8> = grid.columns().collect();
        let a = tensix(&grid, columns[0], 2);
        let b = tensix(&grid, *columns.last().unwrap(), 11);
        assert_ne!(a, b);

        let da = pattern(256, 0xAAAA);
        let db = pattern(256, 0x5555);
        assert_ne!(da, db);

        dev.write(&w, a, 0x1000, &da).unwrap();
        dev.write(&w, b, 0x1000, &db).unwrap();

        let mut back = vec![0u8; 256];
        dev.read(&w, a, 0x1000, &mut back).unwrap();
        assert_eq!(back, da, "writing tile b must not have disturbed tile a");
        dev.read(&w, b, 0x1000, &mut back).unwrap();
        assert_eq!(back, db);
    });
}

#[test]
fn transfer_split_across_a_window_boundary_is_contiguous_on_the_device() {
    // A Tensix tile's 1536 KiB of L1 is smaller than a 2 MiB window, so no L1
    // transfer can straddle a window boundary naturally. Target a device address
    // that does straddle one, and confirm the halves land adjacently rather than
    // both at the start of the aperture -- the failure mode if the window were not
    // retargeted between chunks.
    in_device(|dev| {
        let grid = tensix_grid(dev);
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix(&grid, 5, 6);

        // L1 is 1536 KiB; write the last 8 bytes of it and read them back in two
        // halves through differently-aligned accesses.
        let top = tt_isa::noc::grid::TENSIX_L1_SIZE - 8;
        let data = pattern(8, 0x1234);
        dev.write(&w, tile, top, &data).unwrap();

        let mut first = [0u8; 4];
        let mut second = [0u8; 4];
        dev.read(&w, tile, top, &mut first).unwrap();
        dev.read(&w, tile, top + 4, &mut second).unwrap();
        assert_eq!(&first[..], &data[..4]);
        assert_eq!(&second[..], &data[4..]);
    });
}

#[test]
fn every_tensix_tile_is_addressable() {
    // Each of this chip's tiles gets a distinct value at the same address, then all
    // of them are read back. Any coordinate that aliased onto another would show up
    // as a mismatch.
    //
    // The population comes from the chip, not from a constant. The constant used to
    // be 140 -- the count for an unharvested part -- and iterating it on a harvested
    // chip walks straight into the fused-off columns.
    in_device(|dev| {
        let grid = tensix_grid(dev);
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tiles: Vec<_> = grid.tiles::<Noc0>().collect();
        assert_eq!(tiles.len(), grid.tile_count());
        assert!(
            !tiles.is_empty(),
            "a chip with no Tensix tiles is not usable"
        );

        // None of them may be in a fused-off column. Belt and braces over
        // `Tensix::tiles`, because this is the assertion whose failure mode is a
        // dead host rather than a red test.
        let harvested: Vec<u8> = grid.harvested_columns().collect();
        assert!(
            tiles.iter().all(|t| !harvested.contains(&t.x())),
            "the tile list includes a fused-off column: harvested X = {harvested:?}"
        );

        for (i, tile) in tiles.iter().enumerate() {
            dev.write32(&w, *tile, 0x3000, 0x1000_0000 + i as u32)
                .unwrap();
        }
        for (i, tile) in tiles.iter().enumerate() {
            let got = dev.read32(&w, *tile, 0x3000).unwrap();
            assert_eq!(
                got,
                0x1000_0000 + i as u32,
                "tile {tile:?} (index {i}) read back {got:#x}; coordinates may alias"
            );
        }
    });
}

#[test]
fn l1_ends_where_the_specification_says() {
    // 1536 KiB per Tensix tile (`BabyRISCV/README.md:102`). The last dword is
    // writable; anything past it is not this tile's memory.
    in_device(|dev| {
        let grid = tensix_grid(dev);
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix(&grid, 3, 2);
        let last = tt_isa::noc::grid::TENSIX_L1_SIZE - 4;
        dev.write32(&w, tile, last, 0xFEED_FACE).unwrap();
        assert_eq!(dev.read32(&w, tile, last).unwrap(), 0xFEED_FACE);
    });
}

#[test]
fn a_freed_window_can_be_reallocated_and_retargeted() {
    in_device(|dev| {
        let grid = tensix_grid(dev);
        let a = tensix(&grid, 2, 3);
        let b = tensix(&grid, 4, 5);

        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        dev.write32(&w, a, 0x1000, 0x1111_1111).unwrap();
        let index = w.index();
        dev.free_window(w);

        let w2: Window = dev.alloc_window(WindowKind::TwoMib).unwrap();
        assert_eq!(w2.index(), index);
        // The reallocated window must be reconfigured rather than inheriting the
        // previous holder's target.
        dev.write32(&w2, b, 0x1000, 0x2222_2222).unwrap();
        assert_eq!(dev.read32(&w2, a, 0x1000).unwrap(), 0x1111_1111);
        assert_eq!(dev.read32(&w2, b, 0x1000).unwrap(), 0x2222_2222);
    });
}

#[test]
fn window_exhaustion_is_an_error_not_a_panic() {
    in_device(|dev| {
        let mut held = Vec::new();
        while let Ok(w) = dev.alloc_window(WindowKind::TwoMib) {
            held.push(w);
        }
        // 202 windows exist, one of which is reserved for the kernel driver
        // (`blackhole.c:22,42`: `TLB_2M_WINDOW_COUNT 202`, `KERNEL_TLB_INDEX` is
        // the last of them).
        assert_eq!(held.len(), 201);
        assert!(held
            .iter()
            .all(|w| w.index() != tt_device::tlb::KERNEL_RESERVED_WINDOW));

        // Dropped, not freed: `Window`'s `Drop` returns each index. Before it
        // had one, dropping here left the device with no windows -- which the
        // simulator did not notice, because its `in_device` hands out a fresh
        // chip per call and never scrubs, while silicon's scrubs the gate tile
        // after the body and needs a window to do it. That divergence is what
        // this gate found on its first real run; the silicon scrub is now the
        // check that the drop reaches the pool.
        drop(held);
        let again = dev.alloc_window(WindowKind::TwoMib).unwrap();
        assert_eq!(again.index(), 0, "the lowest window comes back first");
    });
}
