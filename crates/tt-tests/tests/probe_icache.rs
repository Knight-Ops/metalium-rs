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
                MAILBOX_BASE + 0x3000,
            ),
            (
                Core::NC,
                tt_firmware_images::ICACHE_NC,
                tt_isa::dm::nc::MAILBOX_BASE,
                MAILBOX_BASE + 0x3100,
            ),
            (
                Core::T0,
                tt_firmware_images::ICACHE_T[0],
                role(0),
                MAILBOX_BASE + 0x3200,
            ),
            (
                Core::T1,
                tt_firmware_images::ICACHE_T[1],
                role(1),
                MAILBOX_BASE + 0x3300,
            ),
            (
                Core::T2,
                tt_firmware_images::ICACHE_T[2],
                role(2),
                MAILBOX_BASE + 0x3400,
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
