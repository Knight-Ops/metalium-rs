//! Step 3 gate: Rust executes on a baby RISC-V core.
//!
//! The host writes an image into a Tensix tile's L1, releases RISCV T0 from reset,
//! and watches a counter climb. Nothing above this is possible without it: `Dst`
//! and `INSTRN_BUF_BASE` are both unmapped to the NoC, so the host cannot push a
//! Tensix instruction or read a compute result. A core has to sit in the middle.

use tt_device::{core_control::WaitError, tlb::WindowKind, Device};
use tt_isa::mailbox::{self, status};
use tt_isa::noc::{grid, Noc0, NocCoord};
use tt_isa::tensix::{self, Core};
use tt_tests::firmware;
use tt_ttsim::{fork_scope, Simulator};

type Dev<'a> = Device<tt_ttsim::LibTtsim<'a>>;

#[track_caller]
fn in_device(f: impl FnOnce(&mut Dev<'_>)) {
    let result = fork_scope(|| {
        let mut sim = Simulator::open().unwrap_or_else(|e| panic!("could not open simulator: {e}"));
        let mut dev = Device::open(sim.transport()).unwrap_or_else(|e| panic!("{e}"));
        f(&mut dev);
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}

fn tensix_tile(x: u8, y: u8) -> NocCoord<Noc0> {
    assert!(grid::is_tensix(x, y));
    NocCoord::new(x, y).unwrap()
}

/// Budget for the firmware to reach its prologue. Generous: the point of the gate
/// is whether it runs at all, not how fast.
const STARTUP_BUDGET: u64 = 200_000;

#[test]
fn heartbeat_climbs() {
    in_device(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix_tile(3, 4);

        dev.load_and_start(&w, tile, Core::T0, firmware::HEARTBEAT, firmware::LOAD_ADDRESS)
            .unwrap();

        // First, the firmware must reach its prologue.
        match dev.wait_for_status(&w, tile, STARTUP_BUDGET, |s| s == status::RUNNING).unwrap() {
            Ok(_) => {}
            Err(WaitError::Panicked { code }) => panic!("firmware panicked, code {code}"),
            Err(e) => panic!("{e}"),
        }

        // Then the counter must strictly increase across independent reads. A
        // single non-zero sample could be a stray write; a monotone sequence
        // requires something to actually be executing.
        let mut samples = Vec::new();
        for _ in 0..8 {
            samples.push(dev.read32(&w, tile, mailbox::HEARTBEAT).unwrap());
            dev.tick(4096);
        }
        assert!(
            samples.windows(2).all(|p| p[1] > p[0]),
            "heartbeat did not increase monotonically: {samples:?}"
        );
        assert!(samples[0] > 0, "heartbeat never left zero");
    });
}

#[test]
fn a_core_held_in_reset_does_nothing() {
    // The negative control for `heartbeat_climbs`. Without it, that test would
    // still pass if the mailbox happened to contain plausible values, or if the
    // image were somehow running before the loader released reset.
    in_device(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix_tile(5, 6);

        dev.set_core_reset(&w, tile, Core::T0, true).unwrap();
        dev.write(&w, tile, firmware::LOAD_ADDRESS, firmware::HEARTBEAT).unwrap();
        dev.set_reset_pc(&w, tile, Core::T0, firmware::LOAD_ADDRESS as u32).unwrap();

        // Clear the mailbox, then let a lot of time pass with the core still held.
        dev.write32(&w, tile, mailbox::STATUS, 0).unwrap();
        dev.write32(&w, tile, mailbox::HEARTBEAT, 0).unwrap();
        dev.tick(100_000);

        assert_eq!(dev.read_status(&w, tile).unwrap(), 0, "a core in reset must not run");
        assert_eq!(dev.read32(&w, tile, mailbox::HEARTBEAT).unwrap(), 0);

        // And releasing it starts the same image.
        dev.set_core_reset(&w, tile, Core::T0, false).unwrap();
        dev.wait_for_status(&w, tile, STARTUP_BUDGET, |s| s == status::RUNNING)
            .unwrap()
            .unwrap();
    });
}

#[test]
fn reset_state_round_trips() {
    in_device(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix_tile(1, 2);

        // Out of the box every core is held in reset.
        for core in Core::ALL {
            assert!(
                dev.is_core_in_reset(&w, tile, core).unwrap(),
                "{} should start held in reset",
                core.name()
            );
        }

        // Releasing one core must not disturb the others -- the register has no
        // atomic bit operations, so this is checking the read-modify-write.
        dev.set_core_reset(&w, tile, Core::T1, false).unwrap();
        assert!(!dev.is_core_in_reset(&w, tile, Core::T1).unwrap());
        for core in [Core::B, Core::T0, Core::T2, Core::NC] {
            assert!(
                dev.is_core_in_reset(&w, tile, core).unwrap(),
                "releasing T1 also released {}",
                core.name()
            );
        }

        dev.set_core_reset(&w, tile, Core::T1, true).unwrap();
        assert!(dev.is_core_in_reset(&w, tile, Core::T1).unwrap());
    });
}

/// `pc` snapshots are not modelled by ttsim, so this cannot be a simulator gate.
///
/// Reading `0xFFB1_3140` raises `UnimplementedFunctionality: t_tile_mmio_rd32`.
/// See `docs/ttsim-divergence.md`. Kept compiled and ready so that it runs
/// unchanged at the first silicon gate, where it is the strongest available
/// statement about *where* a core is executing rather than merely that it is.
#[cfg(feature = "silicon")]
#[test]
fn pc_snapshot_lands_in_the_loaded_image() {
    in_device(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix_tile(7, 3);

        dev.load_and_start(&w, tile, Core::T0, firmware::HEARTBEAT, firmware::LOAD_ADDRESS)
            .unwrap();
        dev.wait_for_status(&w, tile, STARTUP_BUDGET, |s| s == status::RUNNING)
            .unwrap()
            .unwrap();

        // Speculative, so no exact value can be asserted -- but it must point
        // inside the image.
        let lo = firmware::LOAD_ADDRESS as u32;
        let hi = lo + firmware::HEARTBEAT.len() as u32;
        for _ in 0..4 {
            let pc = dev.read_pc_snapshot(&w, tile, Core::T0).unwrap();
            assert!(
                (lo..hi).contains(&pc),
                "T0's pc snapshot {pc:#x} is outside the loaded image {lo:#x}..{hi:#x}"
            );
            dev.tick(1024);
        }
    });
}

#[test]
fn two_tiles_run_independently() {
    in_device(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let a = tensix_tile(2, 2);
        let b = tensix_tile(16, 11);

        dev.load_and_start(&w, a, Core::T0, firmware::HEARTBEAT, firmware::LOAD_ADDRESS).unwrap();
        dev.wait_for_status(&w, a, STARTUP_BUDGET, |s| s == status::RUNNING).unwrap().unwrap();

        // b has had no image loaded and is still in reset.
        assert_ne!(dev.read_status(&w, b).unwrap(), status::RUNNING);

        dev.load_and_start(&w, b, Core::T0, firmware::HEARTBEAT, firmware::LOAD_ADDRESS).unwrap();
        dev.wait_for_status(&w, b, STARTUP_BUDGET, |s| s == status::RUNNING).unwrap().unwrap();

        dev.tick(8192);
        assert!(dev.read32(&w, a, mailbox::HEARTBEAT).unwrap() > 0);
        assert!(dev.read32(&w, b, mailbox::HEARTBEAT).unwrap() > 0);
    });
}

#[test]
fn risc_b_rejects_a_reset_pc_override() {
    // B's entry point is hardwired to L1 offset 0. The API refuses rather than
    // writing to a register that does not exist.
    in_device(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix_tile(3, 2);
        let err = dev.set_reset_pc(&w, tile, Core::B, 0x6000).unwrap_err();
        assert!(err.to_string().contains("RISCV B"), "{err}");

        // And loading B anywhere but offset 0 is refused for the same reason.
        let err = dev
            .load_and_start(&w, tile, Core::B, firmware::HEARTBEAT, 0x6000)
            .unwrap_err();
        assert!(err.to_string().contains("L1 offset 0"), "{err}");
    });
}

#[test]
fn an_image_that_would_not_fit_in_l1_is_refused() {
    in_device(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix_tile(3, 2);
        let at = tensix::L1_SIZE - 8;
        let err = dev
            .load_and_start(&w, tile, Core::T0, firmware::HEARTBEAT, at)
            .unwrap_err();
        assert!(err.to_string().contains("outside"), "{err}");
    });
}
