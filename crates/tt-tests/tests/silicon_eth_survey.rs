//! Phase 8 spike, silicon, **read-only**: what do the cards say about their
//! Ethernet tiles?
//!
//! Split into steps that each touch strictly more than the last, run one test per
//! process by `cargo xtask silicon`, so a hang is attributed to the one access
//! that caused it (Silicon operating notes). Nothing here writes to an Ethernet
//! tile.
//!
//! 1. `telemetry_directory` -- the ARC only, which every gate already touches.
//! 2. `ethernet_niu_identity` -- NIU registers of row-1 tiles, as `ethdump.c:462-474`
//!    reads them, and nothing else.
//! 3. `ethernet_link_state` -- L1 boot results and queue control, only on tiles the
//!    NIU says are Ethernet and not harvested.

#![cfg(feature = "silicon")]

use tt_device::telemetry::TelemetryTable;
use tt_device::tlb::WindowKind;
use tt_isa::noc::{niu, Noc0, NocCoord, TileType};
use tt_tests::backend::{open_card, scrub};
use tt_tests::harness::{assert_on_silicon, Dev};
use tt_ttsim::fork_scope;

fn each_card(f: impl Fn(u16, &mut Dev<'_>)) {
    assert_on_silicon();
    for card in 0..2u16 {
        if let Err(e) = fork_scope(|| {
            let mut d = open_card(card);
            f(card, &mut d);
            scrub(&mut d);
        }) {
            panic!("{e}");
        }
    }
}

#[test]
fn telemetry_directory() {
    each_card(|card, d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let t = TelemetryTable::read(d, &w).unwrap();
        for tag in t.tags().collect::<Vec<_>>() {
            let v = t.read_tag(d, &w, tag).unwrap().unwrap();
            println!("MEASURE card{card}.telemetry[{tag}] = {v:#010x}");
        }
    });
}

/// Row 1, minus the non-memory and DRAM columns (`ethdump.c:462-463`). Raw
/// coordinates: `NoC/Coordinates.md:28-29` does not translate X on rows 0 and 1.
fn candidates() -> impl Iterator<Item = u8> {
    (1..=16u8).filter(|x| *x != 8 && *x != 9)
}

#[test]
fn ethernet_niu_identity() {
    each_card(|card, d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        for x in candidates() {
            let c = NocCoord::<Noc0>::new(x, 1).unwrap();
            let id = d
                .read32(&w, c, niu::NOC0_BASE + niu::NOC_ENDPOINT_ID)
                .unwrap();
            let cfg = d.read32(&w, c, niu::NOC0_BASE + niu::NIU_CFG_0).unwrap();
            let logical = d
                .read32(&w, c, niu::NOC0_BASE + niu::NOC_ID_LOGICAL)
                .unwrap();
            println!(
                "MEASURE card{card}.eth x={x:2} endpoint={id:#010x} ({:?}) niu_cfg_0={cfg:#010x} \
                 harvested={} logical=({},{})",
                TileType::from_endpoint_id(id),
                cfg & niu::NIU_CFG_0_HARVESTED != 0,
                logical & 0x3F,
                (logical >> 6) & 0x3F,
            );
        }
    });
}

const BOOT_RESULTS: u64 = 0x7_CC00;
const BOOT_PARAMS: u64 = 0x7_C000;

#[test]
fn ethernet_link_state() {
    each_card(|card, d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        for x in candidates() {
            let c = NocCoord::<Noc0>::new(x, 1).unwrap();
            let id = d
                .read32(&w, c, niu::NOC0_BASE + niu::NOC_ENDPOINT_ID)
                .unwrap();
            let cfg = d.read32(&w, c, niu::NOC0_BASE + niu::NIU_CFG_0).unwrap();
            if TileType::from_endpoint_id(id) != TileType::Ethernet
                || cfg & niu::NIU_CFG_0_HARVESTED != 0
            {
                println!("MEASURE card{card}.eth x={x:2} skipped (endpoint {id:#x}, cfg {cfg:#x})");
                continue;
            }
            let mut results = vec![0u8; 256 * 4];
            d.read(&w, c, BOOT_RESULTS, &mut results).unwrap();
            let r = |i: usize| u32::from_le_bytes(results[i * 4..i * 4 + 4].try_into().unwrap());
            let mut params = vec![0u8; 64 * 4];
            d.read(&w, c, BOOT_PARAMS, &mut params).unwrap();
            let p = |i: usize| u32::from_le_bytes(params[i * 4..i * 4 + 4].try_into().unwrap());
            let mut q = |base: u64| d.read32(&w, c, base).unwrap();
            let mut ctrl = String::new();
            for i in 0..3u64 {
                let t = q(0xFFB9_0000 + i * 0x1000);
                let rx = q(0xFFB9_4000 + i * 0x1000);
                ctrl += &format!(" txq{i}={t:#x} rxq{i}={rx:#x}");
            }
            let reset = q(0xFFB1_21B0);
            println!(
                "MEASURE card{card}.eth x={x:2} port={} train={} postcode={:#x} lanes={:#x} \
                 peer[240..248]={:08x?} mac_params=({:#x},{:#x}) soft_reset={reset:#x}{ctrl}",
                r(1),
                r(2),
                r(32),
                r(34),
                (240..248).map(r).collect::<Vec<_>>(),
                p(36),
                p(37),
            );
            let nonzero: Vec<usize> = (0..256).filter(|&i| r(i) != 0).collect();
            println!("MEASURE card{card}.eth x={x:2} boot_results nonzero words {nonzero:?}");
        }
    });
}

/// S6: which parts of a live tile's L1 hold anything, sampled twice. Read-only.
/// A region that is zero both times is a *candidate* for our buffers, not proof.
#[test]
fn ethernet_l1_occupancy() {
    each_card(|card, d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        for x in [3u8, 13] {
            let c = NocCoord::<Noc0>::new(x, 1).unwrap();
            let mut snap = || {
                let mut l1 = vec![0u8; 512 * 1024];
                d.read(&w, c, 0, &mut l1).unwrap();
                l1
            };
            let first = snap();
            std::thread::sleep(std::time::Duration::from_secs(3));
            let second = snap();
            // Occupancy per 4 KiB page: 'Z' zero both times, '.' nonzero and
            // stable, '*' changed between samples.
            let map: String = (0..128)
                .map(|p| {
                    let (a, b) = (&first[p * 4096..][..4096], &second[p * 4096..][..4096]);
                    if a != b {
                        '*'
                    } else if a.iter().all(|&v| v == 0) {
                        'Z'
                    } else {
                        '.'
                    }
                })
                .collect();
            println!("MEASURE card{card}.eth x={x:2} l1 pages {map}");
            let w32 =
                |i: usize| u32::from_le_bytes(second[0x7_CC00 + i * 4..][..4].try_into().unwrap());
            println!(
                "MEASURE card{card}.eth x={x:2} boot_results[238..256]={:08x?}",
                (238..256).map(w32).collect::<Vec<_>>()
            );
        }
    });
}
