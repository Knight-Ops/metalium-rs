//! Exploratory: what ttsim models of the GDDR6 behind the DRAM tiles.
//!
//! Phase 9 keeps tensors on the card, so DRAM becomes load-bearing, and nothing
//! in this workspace has touched it beyond `probe_niu`'s size bisection. Three
//! questions, each answered on the simulator before the silicon survey asks it:
//!
//! 1. Do the three NoC endpoints of a channel alias one memory
//!    (`BlackholeA0/README.md:5`, "each 4 GiB is exposed identically on 3 tiles")?
//! 2. Does ttsim answer at the translated DRAM coordinates, X 17-18 and
//!    Y 12-23 (`NoC/Coordinates.md:35-42`), or only at the raw ones?
//! 3. Is a 4 GiB window a working path into a channel?
//!
//! Run with `cargo test -p tt-tests --test probe_dram -- --ignored --nocapture`.

#![cfg(not(feature = "silicon"))]
use std::fs;
use std::path::{Path, PathBuf};

use tt_device::{tlb::WindowKind, Device};
use tt_isa::noc::{Noc0, NocCoord};
use tt_ttsim::{fork_scope, Simulator};

/// UMD's `DRAM_CORES_NOC0` (`blackhole_implementation.hpp:96-105`): channel `c`'s
/// three endpoints, raw NoC #0, in port order.
const RAW: [[(u8, u8); 3]; 8] = [
    [(0, 0), (0, 1), (0, 11)],
    [(0, 2), (0, 10), (0, 3)],
    [(0, 9), (0, 4), (0, 8)],
    [(0, 5), (0, 7), (0, 6)],
    [(9, 0), (9, 1), (9, 11)],
    [(9, 2), (9, 10), (9, 3)],
    [(9, 9), (9, 4), (9, 8)],
    [(9, 5), (9, 7), (9, 6)],
];

/// UMD's unharvested translation (`blackhole_coordinate_manager.cpp:303-346`):
/// channels 0-3 at X 17, 4-7 at X 18, three consecutive Y each from 12.
fn translated(channel: usize, port: usize) -> (u8, u8) {
    let x = 17 + (channel / 4) as u8;
    let y = 12 + (3 * (channel % 4) + port) as u8;
    (x, y)
}

fn scratch(name: &str) -> PathBuf {
    let dir =
        PathBuf::from(std::env::var_os("CARGO_TARGET_TMPDIR").unwrap_or("/tmp".into())).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Run `f` against a fresh simulator in a child, and report what it wrote, or
/// `None` if the child died (ttsim kills the process on an unmodelled access).
fn trial(
    dir: &Path,
    f: impl FnOnce(&mut Device<tt_ttsim::LibTtsim<'_>>) -> String,
) -> Option<String> {
    let verdict = dir.join("verdict");
    let _ = fs::remove_file(&verdict);
    let _ = fork_scope(|| {
        let mut sim = Simulator::open().unwrap();
        let mut dev = Device::open(sim.transport()).unwrap();
        let out = f(&mut dev);
        fs::write(&verdict, out).unwrap();
    });
    fs::read_to_string(&verdict).ok()
}

fn c(x: u8, y: u8) -> NocCoord<Noc0> {
    NocCoord::new(x, y).unwrap()
}

#[test]
#[ignore = "exploratory"]
fn which_endpoints_alias_one_channel() {
    let dir = scratch("dram-alias");
    let out = trial(&dir, |dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        const AT: u64 = 0x1000;
        // Every endpoint writes its own tag at the same address, in order; then
        // every endpoint reads. Aliased endpoints see the last writer of their group.
        let all: Vec<(u8, u8)> = RAW.iter().flatten().copied().collect();
        for (i, &(x, y)) in all.iter().enumerate() {
            dev.write32(&w, c(x, y), AT, 0xD000_0000 | i as u32)
                .unwrap();
        }
        let mut s = String::new();
        for &(x, y) in &all {
            let v = dev.read32(&w, c(x, y), AT).unwrap();
            let writer = all[(v & 0xFF) as usize];
            s += &format!("({x},{y}) reads {v:#x}, last written through {writer:?}\n");
        }
        s
    });
    print!("{}", out.expect("child died"));
}

#[test]
#[ignore = "exploratory"]
fn does_ttsim_answer_at_translated_dram_coordinates() {
    let dir = scratch("dram-translated");
    for (channel, ports) in RAW.iter().enumerate() {
        let (x, y) = translated(channel, 0);
        let (rx, ry) = ports[0];
        let out = trial(&dir, |dev| {
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            dev.write32(&w, c(rx, ry), 0x2000, 0xC0DE_0000 | channel as u32)
                .unwrap();
            let v = dev.read32(&w, c(x, y), 0x2000).unwrap();
            format!("{v:#x}")
        });
        println!(
            "channel {channel}: translated ({x},{y}) -> {}",
            out.as_deref().unwrap_or("child died")
        );
    }
}

#[test]
#[ignore = "exploratory"]
fn a_four_gib_window_reaches_the_whole_channel() {
    let dir = scratch("dram-4g");
    for at in [
        0x0u64,
        0x20_0000,
        0x1000_0000,
        0xFEFF_FFF0,
        0xFF00_0000,
        0xFFFF_FFF0,
    ] {
        let out = trial(&dir, |dev| {
            let w = dev.alloc_window(WindowKind::FourGib).unwrap();
            let (x, y) = RAW[3][0];
            dev.write32(&w, c(x, y), at, 0x5EED_0000 | (at as u32 >> 16))
                .unwrap();
            let v = dev.read32(&w, c(x, y), at).unwrap();
            format!("{v:#x}")
        });
        println!("{at:#012x}: {}", out.as_deref().unwrap_or("child died"));
    }
}

/// What congruence ttsim demands of a DRAM -> L1 read. Wormhole's table says
/// C32 (`WormholeB0/NoC/Alignment.md:18`); Blackhole's `MemoryMap.md:106` links
/// an `Alignment.md` its tree does not have. Measured with the `dm_b` mover and
/// raw descriptors, so the host-side check does not decide the answer.
#[test]
#[ignore = "exploratory"]
fn which_congruence_ttsim_demands_of_a_dram_read() {
    use tt_isa::dm;
    for (dram_at, l1_at) in [
        (0x1000u32, 0x2_0000u32),
        (0x1020, 0x2_0000),
        (0x1040, 0x2_0000),
        (0x1080, 0x2_0000),
        (0x1040, 0x2_0040),
        (0x1010, 0x2_0010),
    ] {
        let dir = scratch("dram-congruence");
        let out = trial(&dir, |d| {
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let t = c(3, 4);
            let (_, image, _) = tt_firmware_images::DM_B;
            let mut m =
                tt_kernels::dm::DataMover::start(d, &w, t, &tt_isa::dram::Dram::FULL, image)
                    .unwrap();
            let _ = &mut m;
            for (at, v) in [
                (dm::OP, dm::op::READ),
                (dm::CHANNEL, 0),
                (dm::PORT, 0),
                (dm::DRAM_OFFSET, dram_at),
                (dm::L1_ADDR, l1_at),
                (dm::LEN, 256),
                (dm::SEQ, 1),
            ] {
                d.write32(&w, t, at, v).unwrap();
            }
            for _ in 0..100_000 {
                if d.read32(&w, t, dm::DONE).unwrap() == 1 {
                    return format!("done, error {}", d.read32(&w, t, dm::ERROR).unwrap());
                }
                d.tick(64);
            }
            "no answer".into()
        });
        println!(
            "dram {dram_at:#x} -> l1 {l1_at:#x}: {}",
            out.as_deref().unwrap_or("ttsim refused")
        );
    }
}
