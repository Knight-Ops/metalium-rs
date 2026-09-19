//! Exploratory: does ttsim decode the NoC-visible second mapping of local data
//! RAM, and does it model the post-reset zeroing window?
//!
//! Blackhole maps each baby RISCV's local data RAM twice: at `MEM_LOCAL_BASE`,
//! reachable only by its own core, and again somewhere in
//! `0xFFB1_4000..0xFFB1_DFFF`, reachable from any core *and over the NoC*
//! (`BabyRISCV/README.md:144-157`). The second mapping is new in Blackhole and
//! exists so "the host [can] more easily initialize the local data RAM".
//!
//! Every design decision about staging local RAM branches on whether ttsim
//! models it, and an undecoded tile address is not an error there — it prints a
//! message and `_Exit`s. So each probe runs in its own `fork_scope` and **the
//! child's death is the observation**.
//!
//! Addresses are literals rather than `tt_isa` constants on purpose: this runs
//! before any of them are written, so that the design follows the measurement.

use tt_device::{tlb::WindowKind, Device};
use tt_isa::noc::{Noc0, NocCoord};
use tt_ttsim::{fork_scope, Simulator};

type Dev<'a> = Device<tt_ttsim::LibTtsim<'a>>;

/// Per-core base of the slow access path, from the memory map at
/// `BabyRISCV/README.md:109-116`. A uniform 0x2000 stride in hardware core
/// order — the same order as the `pc` snapshot block, and not the soft-reset
/// bit order.
const SLOW_PATH: &[(&str, u64, u64)] = &[
    // name, base, backing RAM size
    ("B", 0xFFB1_4000, 8 * 1024),
    ("NC", 0xFFB1_6000, 8 * 1024),
    ("T0", 0xFFB1_8000, 4 * 1024),
    ("T1", 0xFFB1_A000, 4 * 1024),
    ("T2", 0xFFB1_C000, 4 * 1024),
];

/// `RISCV_DEBUG_REG_DISABLE_RESET` (`SoftReset.md:42-54`).
const DISABLE_RESET: u64 = 0xFFB1_2224;

fn survives(f: impl FnOnce(&mut Dev<'_>)) -> bool {
    fork_scope(|| {
        let mut sim = Simulator::open().unwrap();
        let mut dev = Device::open(sim.transport()).unwrap();
        f(&mut dev);
    })
    .is_ok()
}

fn tile() -> NocCoord<Noc0> {
    NocCoord::<Noc0>::new(3, 4).unwrap()
}

#[test]
#[ignore = "exploratory"]
fn p1_is_the_slow_path_aperture_decoded() {
    // Control first: without it, "everything died" is equally consistent with a
    // broken harness.
    let control = survives(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let v = dev.read_soft_reset(&w, tile()).unwrap();
        println!("control: SOFT_RESET_0 = {v:#010x}");
    });
    println!("control (read SOFT_RESET_0): {}", verdict(control));
    assert!(control, "the harness itself must reach the device");

    for (name, base, _) in SLOW_PATH {
        let wrote = survives(|dev| {
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            dev.write32(&w, tile(), *base, 0x5A5A_5A5A).unwrap();
        });
        let read = survives(|dev| {
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            let v = dev.read32(&w, tile(), *base).unwrap();
            println!("  {name} @ {base:#x} reads {v:#010x}");
        });
        println!(
            "{name:>2} @ {base:#010x}: write {}, read {}",
            verdict(wrote),
            verdict(read)
        );
    }
}

#[test]
#[ignore = "exploratory"]
fn p2_does_it_behave_as_ram_and_does_the_upper_half_alias() {
    // Only meaningful if P1 survived. The T-cores get 8 KiB of aperture for a
    // 4 KiB RAM -- the memory map lists two rows each, with identical labels --
    // and the specification does not say what the upper half is.
    for (name, base, size) in SLOW_PATH {
        let ok = survives(|dev| {
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            dev.write32(&w, tile(), *base, 0xC0FF_EE11).unwrap();
            dev.write32(&w, tile(), base + 0x1000, 0xDEAD_BEEF).unwrap();
            let low = dev.read32(&w, tile(), *base).unwrap();
            let high = dev.read32(&w, tile(), base + 0x1000).unwrap();
            println!(
                "  {name} ({size} B RAM): +0x0000 = {low:#010x}, +0x1000 = {high:#010x}{}",
                if low == 0xDEAD_BEEF {
                    "  <- upper half aliases the lower"
                } else {
                    ""
                }
            );
        });
        println!("{name:>2}: {}", verdict(ok));
    }
}

#[test]
#[ignore = "exploratory"]
fn p3_is_the_zeroing_window_modelled() {
    // Stage while held in reset, release, read back. `BabyRISCV/README.md:152`:
    // the RAM spends up to 2048 cycles zeroing itself after the core leaves
    // reset, and NoC accesses are *not* stalled for it.
    use tt_isa::tensix::Core;
    let ok = survives(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        dev.set_core_reset(&w, tile(), Core::T0, true).unwrap();
        dev.write32(&w, tile(), 0xFFB1_8000, 0xC0FF_EE11).unwrap();
        let before = dev.read32(&w, tile(), 0xFFB1_8000).unwrap();
        dev.set_core_reset(&w, tile(), Core::T0, false).unwrap();
        let immediately = dev.read32(&w, tile(), 0xFFB1_8000).unwrap();
        dev.tick(4096);
        let after = dev.read32(&w, tile(), 0xFFB1_8000).unwrap();
        println!(
            "  staged {before:#010x}; after release {immediately:#010x}; \
             after 4096 cycles {after:#010x}"
        );
    });
    println!("p3: {}", verdict(ok));
}

#[test]
#[ignore = "exploratory"]
fn p4_does_disable_reset_round_trip() {
    // Expected to work: 0xFFB1_2224 is inside RISCV_DEBUG_REGS (0xFFB1_2000..
    // 0x2FFF), which ttsim decodes. Bit 6 is RISCV B's local-data-RAM bit.
    // Split read from write: a register that accepts a write but refuses a read
    // cannot be driven with a read-modify-write, which is the convention the
    // rest of the tile-register path uses.
    let read = survives(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let v = dev.read32(&w, tile(), DISABLE_RESET).unwrap();
        println!("  read DISABLE_RESET = {v:#010x}");
    });
    let write = survives(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        dev.write32(&w, tile(), DISABLE_RESET, 1 << 6).unwrap();
        println!("  wrote bit 6 (RISCV B local data RAM)");
    });
    println!("p4: read {}, write {}", verdict(read), verdict(write));
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "survived"
    } else {
        "DIED"
    }
}
