//! Exploratory: find which NoC coordinates have addressable L1.
//!
//! Ignored by default. Run with
//! `cargo test -p tt-tests --test scan -- --ignored --nocapture`.
//!
//! Two constraints force this shape:
//!
//! * `NOC_ENDPOINT_ID` is not implemented by ttsim (`base_noc_regs_rd32:
//!   UnimplementedFunctionality`), so the tile-identification probe `ethdump.c:462`
//!   uses is unavailable on the simulator. The only portable question is "does an
//!   L1 round-trip work here".
//! * Asking that question of the wrong tile is fatal, so each coordinate is probed
//!   in its own forked process.

// Simulator only, permanently: a blind grid walk is the sweep that hangs on a fused-off tile.
#![cfg(not(feature = "silicon"))]
use std::fs;
use std::path::PathBuf;

use tt_device::{tlb::WindowKind, Device};
use tt_isa::noc::{Noc0, NocCoord};
use tt_ttsim::{fork_scope, Simulator};

#[test]
#[ignore = "exploratory, prints the topology"]
fn find_addressable_l1() {
    let dir = PathBuf::from(std::env::var_os("CARGO_TARGET_TMPDIR").unwrap_or("/tmp".into()))
        .join("l1-scan");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    const PATTERN: [u8; 8] = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x23, 0x45, 0x67];

    for y in 0..12u8 {
        for x in 0..17u8 {
            let out = dir.join(format!("{x}-{y}"));
            let _ = fork_scope(|| {
                let mut sim = Simulator::open().unwrap();
                let mut dev = Device::open(sim.transport()).unwrap();
                let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
                let coord = NocCoord::<Noc0>::new(x, y).unwrap();
                // L1 starts at 0 in a tile's own address space; 0x1000 is clear of
                // the reset vectors.
                let text = match dev.write(&w, coord, 0x1000, &PATTERN) {
                    Err(e) => format!("w:{e}"),
                    Ok(()) => {
                        let mut back = [0u8; 8];
                        match dev.read(&w, coord, 0x1000, &mut back) {
                            Err(e) => format!("r:{e}"),
                            Ok(()) if back == PATTERN => "L1".to_string(),
                            Ok(()) => format!("{back:02x?}"),
                        }
                    }
                };
                fs::write(&out, text).unwrap();
            });
        }
    }

    println!(
        "\n      {}",
        (0..17).map(|x| format!("{x:>7}")).collect::<String>()
    );
    for y in 0..12u8 {
        let mut row = String::new();
        for x in 0..17u8 {
            let cell = fs::read_to_string(dir.join(format!("{x}-{y}"))).unwrap_or("FATAL".into());
            row.push_str(&format!("{:>7}", cell.chars().take(6).collect::<String>()));
        }
        println!("y={y:2}  {row}");
    }
}
