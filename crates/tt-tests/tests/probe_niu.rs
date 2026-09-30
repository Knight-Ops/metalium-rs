//! Exploratory: measure how much memory each row of the NoC grid exposes.
//!
//! ttsim does not implement `NOC_ENDPOINT_ID`, so a tile cannot ask what it is.
//! Its memory size is the next best discriminator, and the spec gives distinct
//! values: a Tensix tile has 1536 KiB of L1, an Ethernet tile 512 KiB
//! (`EthernetTile/README.md`), and a DRAM tile a whole channel.
//!
//! Run with `cargo test -p tt-tests --test probe_niu -- --ignored --nocapture`.

// Simulator only, permanently: bisecting the highest addressable byte at every coordinate is the sweep that hangs on a fused-off tile.
#![cfg(not(feature = "silicon"))]
use std::fs;
use std::path::PathBuf;

use tt_device::{tlb::WindowKind, Device};
use tt_isa::noc::{Noc0, NocCoord};
use tt_ttsim::{fork_scope, Simulator};

/// Highest address that accepts a write, found by doubling then bisecting.
fn probe_top(x: u8, y: u8, dir: &std::path::Path) -> Option<u64> {
    let out = dir.join(format!("size-{x}-{y}"));
    let _ = fs::remove_file(&out);

    // `lo` is known good, `hi` known bad. Each trial runs in its own process
    // because a bad address is fatal; the child records the verdict before dying.
    let trial = |addr: u64| -> bool {
        let verdict = dir.join("trial");
        let _ = fs::remove_file(&verdict);
        let _ = fork_scope(|| {
            let mut sim = Simulator::open().unwrap();
            let mut dev = Device::open(sim.transport()).unwrap();
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            let coord = NocCoord::<Noc0>::new(x, y).unwrap();
            let ok = dev.write32(&w, coord, addr, 0xA5A5_A5A5).is_ok()
                && dev.read32(&w, coord, addr).is_ok_and(|v| v == 0xA5A5_A5A5);
            fs::write(&verdict, if ok { "1" } else { "0" }).unwrap();
        });
        fs::read_to_string(&verdict)
            .map(|s| s == "1")
            .unwrap_or(false)
    };

    if !trial(0x1000) {
        return None;
    }
    // Double until it fails, capped well above any plausible tile memory.
    let mut lo = 0x1000u64;
    let mut hi = 0x2000u64;
    while trial(hi) {
        lo = hi;
        hi *= 2;
        if hi > 1 << 40 {
            return Some(hi);
        }
    }
    // Bisect to the first failing address, to 4 KiB.
    while hi - lo > 0x1000 {
        let mid = lo + (hi - lo) / 2;
        if trial(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let _ = fs::write(&out, lo.to_string());
    Some(hi)
}

#[test]
#[ignore = "exploratory, slow -- forks per bisection step"]
fn measure_tile_memory_sizes() {
    let dir = PathBuf::from(std::env::var_os("CARGO_TARGET_TMPDIR").unwrap_or("/tmp".into()))
        .join("size-scan");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    // One representative coordinate per distinct row/column class found by the
    // L1 scan, rather than all 204.
    let samples = [
        ("x=0  (left column)", 0u8, 5u8),
        ("x=9  (middle column)", 9, 5),
        ("y=0  row, x=0", 0, 0),
        ("y=1  row", 3, 1),
        ("y=2  row", 3, 2),
        ("y=11 row", 3, 11),
        ("x=16 column", 16, 6),
    ];
    for (label, x, y) in samples {
        match probe_top(x, y, &dir) {
            Some(top) => println!(
                "{label:22} ({x:2},{y:2}) -> first bad address {top:#x} ({} KiB)",
                top / 1024
            ),
            None => println!("{label:22} ({x:2},{y:2}) -> nothing addressable"),
        }
    }
}
