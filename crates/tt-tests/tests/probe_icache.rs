//! How big RISCV B's and RISCV NC's instruction caches are on Blackhole: not
//! documented (the Blackhole tree links a Wormhole page; `riscv-guide-review.md`),
//! and on card 0 the mover's per-entry cost moved with code layout. The probe
//! (`tt-firmware/src/icache_probe.rs`) runs the last `size` bytes of two 8 KiB
//! blocks -- straight-line `nop`s, and a chain of jumps 32 bytes apart -- and
//! reports cycles per pass; where the cycles per byte step up is the capacity.
//!
//! Exploratory: the numbers are printed, and only their presence is
//! asserted. The simulator does not model the cache, so its numbers mean
//! nothing; run it on silicon.

use tt_device::tlb::WindowKind;
use tt_isa::mailbox::{offset, status, MAILBOX_BASE};
use tt_isa::tensix::Core;
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile};

const SIZES: usize = 14;
const PASSES: f64 = 16.0;

#[test]
#[ignore = "exploratory: prints the instruction-cache curve"]
fn instruction_cache_size_by_timing() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let role = |t: u64| tt_isa::mailbox::role::BASE + t * tt_isa::mailbox::role::STRIDE;
        for (core, image, mailbox, results) in [
            (
                Core::B,
                tt_firmware_images::ICACHE_B,
                MAILBOX_BASE,
                tt_isa::mailbox::icache_results(0),
            ),
            (
                Core::NC,
                tt_firmware_images::ICACHE_NC,
                tt_isa::dm::nc::MAILBOX_BASE,
                tt_isa::mailbox::icache_results(1),
            ),
            (
                Core::T0,
                tt_firmware_images::ICACHE_T[0],
                role(0),
                tt_isa::mailbox::icache_results(2),
            ),
            (
                Core::T1,
                tt_firmware_images::ICACHE_T[1],
                role(1),
                tt_isa::mailbox::icache_results(3),
            ),
            (
                Core::T2,
                tt_firmware_images::ICACHE_T[2],
                role(2),
                tt_isa::mailbox::icache_results(4),
            ),
        ] {
            d.write32(&w, t, mailbox + offset::STATUS, 0).unwrap();
            let (_, bytes, at) = image;
            match core {
                Core::NC => d.load_and_start_nc(&w, t, bytes, at).unwrap(),
                _ => d.load_and_start(&w, t, core, bytes, at).unwrap(),
            }
            let done = d
                .wait_for_mailbox(
                    &w,
                    t,
                    mailbox + offset::STATUS,
                    mailbox + offset::PANIC_CODE,
                    50_000_000,
                    |s| s == status::DONE,
                )
                .unwrap();
            assert!(done.is_ok(), "{core:?}'s probe did not finish: {done:?}");
            println!("MEASURE icache {core:?}: size, straight-line cycles/pass, cycles/byte, jump-chain cycles/pass, cycles/jump");
            for k in 0..SIZES {
                let mut r = |i: u64| d.read32(&w, t, results + k as u64 * 12 + i * 4).unwrap();
                let (size, line, jump) = (r(0), r(1) as f64 / PASSES, r(2) as f64 / PASSES);
                assert!(size > 0);
                if line == 0.0 {
                    continue; // past this core's blocks
                }
                println!(
                    "MEASURE icache {core:?} {size:>5} B  line {line:9.1} ({:5.2}/B)  jump {jump:9.1} ({:5.2}/jump)",
                    line / size as f64,
                    jump / (size as f64 / 32.0)
                );
            }
            d.set_core_reset(&w, t, core, true).unwrap();
        }
    });
}

/// How the cache is organised, which decides how hot code must be placed
/// (`tt-firmware/src/icache_alias.rs`): calls at `n` addresses `k` bytes apart, then
/// every `step` bytes over a footprint, ascending and descending, each address one
/// `ret`. Printed per call; a step up of ~5 cycles a call is a miss.
///
/// * Ways: the largest `n` that stays cheap at `k` = the set stride (capacity / ways).
///   Direct-mapped: two addresses 4 KiB apart already miss.
/// * Line: over the 8 KiB footprint the cost per call stops rising at `step` = the line.
/// * Prefetch: ascending cheaper than descending at small steps.
#[test]
#[ignore = "exploratory: prints the instruction cache's geometry"]
fn instruction_cache_geometry_by_aliasing() {
    // As `icache_alias.rs`.
    const WAY_ROUNDS: f64 = 64.0;
    const LINE_ROUNDS: f64 = 8.0;
    const ROWS: u64 = 6 * 8 + 2 * 7 * 2;
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let results = tt_isa::mailbox::ICACHE_ALIAS_RESULTS;
        for (core, image, mailbox) in [
            (Core::B, tt_firmware_images::ICACHE_ALIAS_B, MAILBOX_BASE),
            (
                Core::NC,
                tt_firmware_images::ICACHE_ALIAS_NC,
                tt_isa::dm::nc::MAILBOX_BASE,
            ),
        ] {
            for k in 0..ROWS {
                d.write32(&w, t, results + k * 12 + 8, 0).unwrap();
            }
            d.write32(&w, t, mailbox + offset::STATUS, 0).unwrap();
            let (_, bytes, at) = image;
            match core {
                Core::NC => d.load_and_start_nc(&w, t, bytes, at).unwrap(),
                _ => d.load_and_start(&w, t, core, bytes, at).unwrap(),
            }
            let done = d
                .wait_for_mailbox(
                    &w,
                    t,
                    mailbox + offset::STATUS,
                    mailbox + offset::PANIC_CODE,
                    50_000_000,
                    |s| s == status::DONE,
                )
                .unwrap();
            assert!(done.is_ok(), "{core:?}'s probe did not finish: {done:?}");
            let mut row = |k: u64| {
                let mut r = |i: u64| d.read32(&w, t, results + k * 12 + i * 4).unwrap();
                (r(0), r(1), r(2))
            };
            println!("MEASURE icache-ways {core:?}: cycles a call at n addresses k bytes apart");
            for kk in 0..6u64 {
                let mut line = String::new();
                let mut k_bytes = 0;
                for n in 0..8u64 {
                    let (k, nn, cycles) = row(kk * 8 + n);
                    assert_eq!(nn as u64, n + 1);
                    k_bytes = k;
                    if cycles == 0 {
                        line.push_str("      -");
                    } else {
                        line.push_str(&format!(
                            " {:6.2}",
                            cycles as f64 / (WAY_ROUNDS * nn as f64)
                        ));
                    }
                }
                println!("MEASURE icache-ways {core:?} k {k_bytes:>5}: n=1..8 {line}");
            }
            println!("MEASURE icache-line {core:?}: cycles a call every `step` bytes");
            for k in 48..ROWS {
                let (a, step, cycles) = row(k);
                let (footprint, down) = (a & 0x7fff_ffff, a >> 31 != 0);
                if cycles == 0 {
                    continue;
                }
                let calls = (footprint / step) as f64 * LINE_ROUNDS;
                println!(
                    "MEASURE icache-line {core:?} footprint {footprint:>5} step {step:>3} {}: {:6.2}",
                    if down { "down" } else { "up  " },
                    cycles as f64 / calls
                );
            }
            d.set_core_reset(&w, t, core, true).unwrap();
        }
    });
}
