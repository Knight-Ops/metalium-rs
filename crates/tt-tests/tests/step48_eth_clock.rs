//! The E1 mover's trace on the dual-chip simulator: what can be checked
//! without the cycle counter it stamps events with, which ttsim does not model
//! (divergence row 71).
//!
//! The trace is off unless the host turns it on, and the host refuses to here,
//! so the image that carries it still moves data exactly as before. The
//! counter itself is gated on silicon, alone: `silicon_eth_clock`.

#![cfg(not(feature = "silicon"))]

use tt_device::{tlb::WindowKind, Device};
use tt_isa::eth::Ethernet;
use tt_kernels::link::{Dest, Dir, Link, Mover, Source};
use tt_ttsim::{fork_scope, Simulator};

#[test]
fn tracing_is_refused_and_the_untraced_mover_still_moves_data() {
    if let Err(e) = fork_scope(|| {
        let mut sim = Simulator::open_path(tt_ttsim::x2_lib_path()).unwrap();
        let mut ts = sim.transports().into_iter();
        let mut a = Device::open(ts.next().unwrap()).unwrap();
        let mut b = Device::open(ts.next().unwrap()).unwrap();
        let (x0, x1) = tt_tests::topology::BH_X2[0].2;
        let link = Link {
            a: Ethernet::FULL.tile(x0).unwrap(),
            b: Ethernet::FULL.tile(x1).unwrap(),
        };
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let mut m =
            Mover::start(&mut a, &wa, &mut b, &wb, link, tt_firmware_images::ETH_E1).unwrap();
        assert!(
            m.set_trace(&mut a, &wa, link.a, true).is_err(),
            "tracing on ttsim"
        );
        m.set_trace(&mut a, &wa, link.a, false).unwrap();
        let data: Vec<u8> = (0..4096u32).map(|i| (i * 7) as u8).collect();
        for _ in 0..2 {
            m.stage(&mut a, &wa, Dir::AToB, &data).unwrap();
            m.send(&mut a, &wa, Dir::AToB, Source::Staged, Dest::Landed, 4096)
                .unwrap();
        }
        // Nothing recorded: the trace word was zeroed at start.
        assert!(m.take_trace(&mut a, &wa, link.a).unwrap().is_empty());
        assert!(m.take_trace(&mut b, &wb, link.b).unwrap().is_empty());
        let mut back = vec![0u8; 4096];
        m.landed(&mut b, &wb, Dir::AToB, &mut back).unwrap();
        assert_eq!(back, data);
    }) {
        panic!("{e}");
    }
}
