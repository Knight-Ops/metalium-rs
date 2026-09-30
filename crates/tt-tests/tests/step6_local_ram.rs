//! Phase 2: the NoC-visible local data RAM, and the post-reset zeroing window.
//!
//! Blackhole maps each baby RISCV's local data RAM a second time, in
//! `0xFFB1_4000..0xFFB1_DFFF`, reachable over the NoC so the host can initialize
//! it (`BabyRISCV/README.md:144-157`). Staging there is hazardous: the RAM spends
//! up to 2048 cycles zeroing itself after the core leaves reset, and unlike the
//! core's own accesses, NoC accesses are not stalled for it. Either the
//! `RISCV_DEBUG_REG_DISABLE_RESET` bit is set first — which abolishes the window
//! rather than shortening it — or the window is waited out.
//!
//! **ttsim models none of this.** It decodes neither the aperture nor
//! `DISABLE_RESET`, so there is no simulator gate for the behaviour, and the
//! tests that check it are `#[cfg(feature = "silicon")]`. What is gated here is
//! the refusal itself: if either ever starts working, the simulator has gained a
//! model and these tests say so rather than silently passing.

// Every test here asserts what ttsim *refuses*. The behaviour itself is measured
// on silicon by `silicon_local_ram.rs`.
#![cfg(not(feature = "silicon"))]

use tt_device::{tlb::WindowKind, Device};
use tt_isa::noc::{grid, Noc0, NocCoord};
use tt_isa::tensix::{self, Core};
use tt_ttsim::{fork_scope, Simulator};

type Dev<'a> = Device<tt_ttsim::LibTtsim<'a>>;

fn survives(f: impl FnOnce(&mut Dev<'_>)) -> bool {
    fork_scope(|| {
        let mut sim = Simulator::open().unwrap();
        let mut dev = Device::open(sim.transport()).unwrap();
        f(&mut dev);
    })
    .is_ok()
}

fn tile() -> NocCoord<Noc0> {
    assert!(grid::is_tensix_geometry(3, 4));
    NocCoord::<Noc0>::new(3, 4).unwrap()
}

#[test]
fn ttsim_does_not_decode_the_local_ram_noc_aperture() {
    // The control comes first and is not decoration: without it, "every access
    // died" is equally consistent with a misconfigured window, a bad tile, or a
    // broken harness, and the refusals below would prove nothing.
    assert!(
        survives(|dev| {
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            dev.read_soft_reset(&w, tile()).unwrap();
        }),
        "the control must reach the device: this window and tile are fine"
    );

    for core in Core::ALL {
        let base = core.local_data_ram_noc_address();
        assert!(
            !survives(|dev| {
                let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
                dev.write32(&w, tile(), base, 0x5A5A_5A5A).unwrap();
            }),
            "ttsim now decodes {}'s local data RAM at {base:#x}. Its tile MMIO \
             switch had arms for the TDMA, debug, NoC, overlay, Dst, regfile, \
             PCBuf, mailbox and config regions and nothing for local RAM; if that \
             has changed, divergence row 25 needs revisiting and the silicon-only \
             staging tests below can move to the simulator.",
            core.name()
        );
    }
}

#[test]
fn ttsim_does_not_model_the_disable_reset_register() {
    // Both directions, because the distinction decides the API: a register that
    // took a write but refused a read could not be driven by the read-modify-write
    // the rest of the tile-register path uses. ttsim refuses both, naming the
    // register -- it knows it exists and declines to model it.
    for (what, ran) in [
        (
            "read",
            survives(|dev| {
                let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
                dev.read32(&w, tile(), tensix::DISABLE_RESET).unwrap();
            }),
        ),
        (
            "write",
            survives(|dev| {
                let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
                let (bit, _) = Core::B.disable_reset_bits();
                dev.write32(&w, tile(), tensix::DISABLE_RESET, 1 << bit)
                    .unwrap();
            }),
        ),
    ] {
        assert!(
            !ran,
            "a {what} of RISCV_DEBUG_REG_DISABLE_RESET now succeeds on ttsim \
             (`riscv_debug_regs_{what}32: DISABLE_RESET` used to be \
             UnsupportedFunctionality). Divergence row 26 needs revisiting."
        );
    }
}
