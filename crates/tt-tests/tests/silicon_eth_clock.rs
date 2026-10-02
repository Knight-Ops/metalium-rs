//! E1's cycle counter on silicon, alone: the one hardware access the Ethernet
//! benchmarks add (`tt_firmware::cycles`, `mcycle`), which ttsim does not
//! model (divergence row 71).
//!
//! `cargo xtask silicon --release --filter silicon_eth_clock`
//!
//! Run by itself, on a healthy link, before `silicon_bench_eth` leans on it.
//! Two small traced sends a known host interval apart: every step recorded,
//! in order, on a counter that runs -- and how fast it runs, since an Ethernet
//! core's clock need not be AICLK.

#![cfg(feature = "silicon")]

use std::time::{Duration, Instant};

use tt_device::tlb::WindowKind;
use tt_isa::eth::mover::event;
use tt_kernels::link::{discover, Dest, Dir, Mover, Source};
use tt_tests::bench::{report, with_cards, Stats};

#[test]
fn e1_counts_cycles_through_a_traced_send() {
    with_cards(|a, b| {
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let (ga, gb) = (a.ethernet_grid(&wa).unwrap(), b.ethernet_grid(&wb).unwrap());
        let l = discover(a, &wa, &ga, b, &wb, &gb).unwrap()[0];
        let mut m = Mover::start(a, &wa, b, &wb, l, tt_firmware_images::ETH_E1).unwrap();
        m.set_trace(a, &wa, l.a, true).unwrap();
        m.set_trace(b, &wb, l.b, true).unwrap();
        let data: Vec<u8> = (0..16u8).collect();
        m.stage(a, &wa, Dir::AToB, &data).unwrap();
        let mut host = Vec::new();
        for _ in 0..2 {
            host.push(Instant::now());
            m.send(a, &wa, Dir::AToB, Source::Staged, Dest::Landed, 16)
                .unwrap();
            std::thread::sleep(Duration::from_millis(100));
        }
        let sent = m.take_trace(a, &wa, l.a).unwrap();
        let got = m.take_trace(b, &wb, l.b).unwrap();
        m.set_trace(a, &wa, l.a, false).unwrap();
        m.set_trace(b, &wb, l.b, false).unwrap();
        a.park_e1(&wa, l.a).unwrap();
        b.park_e1(&wb, l.b).unwrap();
        println!("MEASURE sender {sent:?}");
        println!("MEASURE receiver {got:?}");
        let kinds =
            |v: &[tt_kernels::link::EthEvent]| v.iter().map(|e| e.event).collect::<Vec<_>>();
        let send = [
            event::SEND_PICKUP,
            event::DATA_SENT,
            event::RECORD_SENT,
            event::ACK_SEEN,
        ];
        let recv = [event::RECORD_SEEN, event::LANDED, event::ACK_SENT];
        assert_eq!(kinds(&sent), [send, send].concat(), "the sender's steps");
        assert_eq!(kinds(&got), [recv, recv].concat(), "the receiver's steps");
        for v in [&sent, &got] {
            assert!(
                v.windows(2)
                    .all(|p| p[1].cycles.wrapping_sub(p[0].cycles) as i32 > 0),
                "the counter runs: {v:?}"
            );
        }
        // The counter's rate: the two pickups against the host's two sends.
        let cycles = sent[4].cycles.wrapping_sub(sent[0].cycles) as f64;
        let us = (host[1] - host[0]).as_secs_f64() * 1e6;
        let mhz = cycles / us;
        report("E1 cycle counter rate", "MHz", "host", Stats::of([mhz]));
        assert!(
            (500.0..3000.0).contains(&mhz),
            "an implausible E1 clock: {mhz} MHz"
        );
        let mut back = vec![0u8; 16];
        m.landed(b, &wb, Dir::AToB, &mut back).unwrap();
        assert_eq!(back, data);
    });
}
