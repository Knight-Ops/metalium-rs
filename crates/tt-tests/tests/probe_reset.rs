//! Exploratory: what does SOFT_RESET_0 read at simulator start?

// Reads ttsim's reset state; on silicon the reset state is whatever the last process left.
#![cfg(not(feature = "silicon"))]
use tt_device::{tlb::WindowKind, Device};
use tt_isa::noc::{Noc0, NocCoord};
use tt_isa::tensix::{self, Core};
use tt_ttsim::{fork_scope, Simulator};

#[test]
#[ignore = "exploratory"]
fn dump_soft_reset() {
    fork_scope(|| {
        let mut sim = Simulator::open().unwrap();
        let mut dev = Device::open(sim.transport()).unwrap();
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = NocCoord::<Noc0>::new(3, 4).unwrap();
        let v = dev.read_soft_reset(&w, tile).unwrap();
        println!("SOFT_RESET_0 = {v:#034b} ({v:#010x})");
        for core in Core::ALL {
            println!(
                "  {:>2}: bit {:2} = {}",
                core.name(),
                core.soft_reset_bit(),
                (v >> core.soft_reset_bit()) & 1
            );
        }
        println!(
            "  BACKEND_RESET_MASK = {:#010x}",
            tensix::BACKEND_RESET_MASK
        );
        println!(
            "  backend bits set    = {:#010x}",
            v & tensix::BACKEND_RESET_MASK
        );
    })
    .unwrap();
}
