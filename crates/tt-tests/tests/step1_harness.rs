//! Step 1 gate: the simulator comes up, identifies as a Blackhole, and time moves.
//!
//! Each test body runs inside `fork_scope`, for two reasons. `libttsim_init` may be
//! called once per process and `cargo test` runs tests on separate threads of one
//! process, so without forking only the first test could open the simulator. And a
//! contract violation terminates the process outright, which without forking would
//! take the whole test binary with it.

use tt_device::{Bar, ConfigOffset, Transport};
use tt_isa::noc::ChipId;
use tt_ttsim::{fork_scope, Simulator};

/// Run a simulator test body in its own process, failing the test if it does not
/// come back cleanly.
#[track_caller]
fn in_simulator(f: impl FnOnce(&mut Simulator)) {
    let result = fork_scope(|| {
        let mut sim = Simulator::open().unwrap_or_else(|e| panic!("could not open simulator: {e}"));
        f(&mut sim);
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}

#[test]
fn simulator_reports_blackhole() {
    in_simulator(|sim| {
        let mut t = sim.transport();
        let id = t.config_read32(ConfigOffset::VendorDevice).unwrap();
        assert_eq!(
            id, 0xB140_1E52,
            "expected Blackhole vendor:device in config offset 0, got {id:#010x}"
        );
        t.verify_is_blackhole()
            .expect("verify_is_blackhole should agree");
    });
}

#[test]
fn class_code_is_processing_accelerator() {
    in_simulator(|sim| {
        let mut t = sim.transport();
        let cr = t.config_read32(ConfigOffset::ClassRevision).unwrap();
        // Class 0x12 (processing accelerator), subclass 0x00, prog-if 0x00.
        assert_eq!(cr >> 8, 0x0012_0000, "class code was {:#010x}", cr >> 8);
        // Blackhole A0 is revision 0; Wormhole B0 is 1. A cheap second opinion on
        // which simulator build got loaded.
        assert_eq!(cr & 0xFF, 0, "expected revision 0 for Blackhole A0");
    });
}

#[test]
fn bar_bases_match_the_addresses_libttsim_decodes() {
    in_simulator(|sim| {
        let mut t = sim.transport();
        for bar in [Bar::Bar0, Bar::Bar2, Bar::Bar4] {
            assert_eq!(
                t.bar_base(bar).unwrap(),
                tt_ttsim::expected_bar_base(bar),
                "{bar:?} base in config space disagrees with libttsim's decode map"
            );
        }
    });
}

#[test]
fn the_single_chip_build_exposes_exactly_one_chip() {
    // `Simulator::open` walks all 32 device slots a bdf can name, so this is what
    // licenses that walk for every other test in the workspace: on this build the
    // 31 absent slots read all-ones rather than terminating the process, and the
    // probe does not invent a chip that is not there.
    in_simulator(|sim| {
        assert_eq!(
            sim.chip_count(),
            1,
            "libttsim_bh.so is the single-chip build; the dual-chip gate lives in \
             step2_multichip.rs"
        );
        assert_eq!(sim.transport().chip(), ChipId(0));
    });
}

#[test]
fn there_is_no_capability_list() {
    // Blackhole exposes no capability structures, so there is no MSI/MSI-X to find.
    // Asserted because the natural thing to write next is a capability walk, and a
    // walk that starts from a garbage pointer would read an undecoded offset and
    // terminate the process.
    in_simulator(|sim| {
        let mut t = sim.transport();
        assert_eq!(
            t.config_read32(ConfigOffset::CapabilitiesPointer).unwrap(),
            0
        );
    });
}

#[test]
fn clock_advances_without_incident() {
    in_simulator(|sim| {
        // Nothing observable is asserted here beyond survival: with no core out of
        // reset there is nothing to advance. The point is that clocking an idle
        // device is well-defined, since every later poll loop depends on it.
        for _ in 0..16 {
            sim.clock(1024);
        }
    });
}

#[test]
fn opening_the_simulator_twice_is_refused_not_fatal() {
    // libttsim_init has no re-init path and aborts if called while running. The
    // wrapper must intercept that, so the failure is a Rust error rather than a
    // dead process.
    in_simulator(|_sim| {
        let second = Simulator::open();
        assert!(
            matches!(second, Err(tt_ttsim::OpenError::AlreadyOpen)),
            "second open should be refused by the wrapper"
        );
    });
}

#[test]
fn invalid_accesses_are_rejected_before_reaching_the_library() {
    // The whole point of the validation layer. Every access below would terminate
    // the process if it reached libttsim; all of them must come back as errors,
    // and the process must still be alive at the end to say so.
    in_simulator(|sim| {
        let mut t = sim.transport();
        let mut buf = [0u8; 4];

        let cases: &[(&str, tt_device::Result<()>)] = &[
            (
                "read past the end of BAR0",
                t.bar_read(Bar::Bar0, Bar::Bar0.size(), &mut buf),
            ),
            (
                "read an undecoded BAR0 hole",
                t.bar_read(Bar::Bar0, 0x1A00_0000, &mut buf),
            ),
            (
                "read a write-only TLB config register",
                t.bar_read(Bar::Bar0, 0x1FC0_0000, &mut buf),
            ),
            (
                "write a read-only PCIe NIU register",
                t.bar_write(Bar::Bar0, 0x1FD0_4000, &buf),
            ),
            (
                "unaligned dword in BAR2",
                t.bar_read(Bar::Bar2, 0x1002, &mut buf),
            ),
        ];
        for (what, outcome) in cases {
            assert!(outcome.is_err(), "{what} should have been rejected");
        }

        // Still alive, and the device still answers.
        t.verify_is_blackhole()
            .expect("simulator should be unharmed");
    });
}

#[test]
fn straddling_a_window_boundary_is_rejected() {
    in_simulator(|sim| {
        let mut t = sim.transport();
        let window_size = tt_ttsim::transport::TLB_2MIB_SIZE;
        let mut buf = [0u8; 8];
        let err = t
            .bar_read(Bar::Bar0, window_size - 4, &mut buf)
            .expect_err("an access spanning two windows should be rejected");
        assert!(err.to_string().contains("straddles"), "{err}");
    });
}
