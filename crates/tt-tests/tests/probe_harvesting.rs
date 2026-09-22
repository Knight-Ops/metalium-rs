//! Exploratory, silicon only: ask a card what it is, touching nothing but its ARC.
//!
//! Run with:
//!
//! ```text
//! cargo test -p tt-tests --features silicon --test probe_harvesting -- --ignored --nocapture
//! ```
//!
//! This is the step that has to come before any silicon gate, and it is a separate
//! probe rather than part of one because of what it deliberately does *not* do.
//! It does not go through `harness::in_device`: that calls `backend::open`, which
//! scrubs the gate tile, and scrubbing is a write to a Tensix tile. The whole
//! point here is to learn which Tensix tiles exist before writing to any of them,
//! so this opens the card directly and stays inside the ARC.
//!
//! It also prints the raw telemetry words before deriving anything from them. If
//! `ENABLED_TENSIX_COL` is not the contiguous prefix that `NoC/Coordinates.md:54`
//! implies, `chip_telemetry` refuses to interpret it — and the refusal is the
//! interesting result, so the evidence has to already be on stdout by then rather
//! than being swallowed with the error.
//!
//! Set `TT_SILICON_DEVICE` to pick a card; both are worth running, because two
//! cards in one host are two ASICs with independent fuses.

#![cfg(feature = "silicon")]

use tt_device::telemetry::TelemetryTable;
use tt_device::{tlb::WindowKind, Device};
use tt_isa::arc;
use tt_isa::noc::grid;
use tt_kmd::Kmd;

#[test]
#[ignore = "requires a card; run explicitly with --ignored --nocapture"]
fn report_what_this_chip_is() {
    let index: u16 = std::env::var("TT_SILICON_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let kmd =
        Kmd::open(index).unwrap_or_else(|e| panic!("could not open /dev/tenstorrent/{index}: {e}"));
    let mut dev = Device::open(kmd).unwrap_or_else(|e| panic!("{e}"));

    let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
    println!("\n=== /dev/tenstorrent/{index} ===");
    println!("ARC tile: raw ({}, {})", arc::ARC_X, arc::ARC_Y);

    // Walk the directory first, so the tag inventory survives any later failure.
    let table = TelemetryTable::read(&mut dev, &w)
        .unwrap_or_else(|e| panic!("could not read the ARC telemetry table: {e}"));

    let tags: Vec<u16> = table.tags().collect();
    println!("telemetry tags published ({}): {:?}", tags.len(), tags);

    // The ARC has to be alive for any of this to mean anything, and the heartbeat
    // is the published way to tell (`docs/sysfs-attributes.md:68`). Reading it
    // twice is the whole check: a frozen counter means a wedged ARC, in which case
    // every other word here is stale rather than wrong.
    let hb1 = table
        .read_tag(&mut dev, &w, arc::tag::TIMER_HEARTBEAT)
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let hb2 = table
        .read_tag(&mut dev, &w, arc::tag::TIMER_HEARTBEAT)
        .unwrap();
    println!(
        "heartbeat: {hb1:?} -> {hb2:?}  ({})",
        match (hb1, hb2) {
            (Some(a), Some(b)) if a != b => "ARC alive",
            (Some(_), Some(_)) => "** FROZEN -- ARC may be wedged **",
            _ => "not published",
        }
    );

    for (name, tag) in [
        ("ENABLED_TENSIX_COL", arc::tag::ENABLED_TENSIX_COL),
        ("NOC_TRANSLATION", arc::tag::NOC_TRANSLATION),
        ("HARVESTING_STATE", arc::tag::HARVESTING_STATE),
    ] {
        match table.read_tag(&mut dev, &w, tag).unwrap() {
            Some(v) => println!("{name} (tag {tag}) = {v:#010x}  ({v:#034b})"),
            None => println!("{name} (tag {tag}) = NOT PUBLISHED"),
        }
    }

    // Only now derive. A refusal here is a real finding, and everything needed to
    // understand it is already printed above.
    let telemetry = dev.chip_telemetry(&w);
    dev.free_window(w);

    let telemetry = match telemetry {
        Ok(t) => t,
        Err(e) => panic!("\nrefused to derive the Tensix grid:\n  {e}\n"),
    };

    println!("\nderived: {telemetry}");
    println!(
        "  Tensix columns present ({}): {:?}",
        telemetry.tensix.enabled_column_count(),
        telemetry.tensix.columns().collect::<Vec<_>>()
    );
    println!(
        "  Tensix columns FUSED OFF -- never address these: {:?}",
        telemetry.tensix.harvested_columns().collect::<Vec<_>>()
    );
    println!("  Tensix tiles: {}", telemetry.tensix.tile_count());
    println!(
        "  full part would be: {} tiles in {} columns",
        grid::FULL_TENSIX_TILE_COUNT,
        grid::TENSIX_COLUMNS.len()
    );

    let (gx, gy) = tt_tests::backend::GATE_TILE;
    println!(
        "  gate tile ({gx},{gy}): {}",
        if telemetry.tensix.contains(gx, gy) {
            "present"
        } else {
            "** FUSED OFF -- the gates cannot run on this card as configured **"
        }
    );

    // Assertions, not just printing: a probe that only prints cannot fail, and the
    // reason to run this is to find out whether the assumptions hold.
    assert!(
        telemetry.noc_translation,
        "this chip reports NoC translation disabled; the grid derivation assumes it"
    );
    assert_ne!(
        telemetry.tensix.tile_count(),
        0,
        "a chip with no Tensix tiles is not usable"
    );
    assert!(
        telemetry.tensix.contains(gx, gy),
        "the gates' tile ({gx},{gy}) is fused off on this card"
    );

    // Not an assertion about what the answer *should* be. 120 is what a p150a with
    // two columns fused off reports and what prompted this probe, but the count is
    // a property of the ASIC, so a different one is information rather than a bug.
    if telemetry.tensix.tile_count() != grid::FULL_TENSIX_TILE_COUNT {
        println!(
            "\nnote: this chip is harvested -- {} of {} Tensix tiles.",
            telemetry.tensix.tile_count(),
            grid::FULL_TENSIX_TILE_COUNT
        );
    }
}
