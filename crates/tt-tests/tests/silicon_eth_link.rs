//! Phase 8 spike S4 on silicon: a TT-link L1 write driven by the host through a
//! TLB window, with no firmware of ours anywhere, across the QSFP-DD cable.
//!
//! The first test in this workspace that *writes* to an Ethernet tile. What it
//! writes: 4 KiB of pattern into the sending tile's L1, a sentinel into the
//! receiving tile's L1, both inside `0x20000..0x60000`, which
//! `silicon_eth_survey::ethernet_l1_occupancy` found zero on every live tile at
//! two samples; and four TXQ2 registers. `SEL_SW` is read, not written: the
//! header entry firmware selected is the one that carries the peer's MAC.

#![cfg(feature = "silicon")]

use std::time::{Duration, Instant};

use tt_device::tlb::WindowKind;
use tt_isa::noc::{Noc0, NocCoord};
use tt_tests::backend::{open_card, scrub};
use tt_tests::harness::{assert_on_silicon, Dev};
use tt_ttsim::fork_scope;

const BOOT_RESULTS: u64 = 0x7_CC00;
const TXQ2: u64 = 0xFFB9_2000;
const BUF: u64 = 0x4_0000;
const LEN: usize = 4096;

fn eth(x: u8) -> NocCoord<Noc0> {
    NocCoord::new(x, 1).unwrap()
}

fn port_up(d: &mut Dev<'_>, x: u8) -> bool {
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    d.read32(&w, eth(x), BOOT_RESULTS + 4).unwrap() == 1
}

fn pattern(seed: u8) -> Vec<u8> {
    (0..LEN)
        .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
        .collect()
}

fn send(a: &mut Dev<'_>, b: &mut Dev<'_>, x: u8, seed: u8) {
    assert!(
        port_up(a, x) && port_up(b, x),
        "link at x={x} is not Up on both ends"
    );
    let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
    let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
    let data = pattern(seed);
    a.write(&wa, eth(x), BUF, &data).unwrap();
    b.write(&wb, eth(x), BUF, &vec![0xEE; LEN + 64]).unwrap();

    for (n, o) in [
        ("SEL_SW", 0x80),
        ("MAX_PKT", 0x0C),
        ("STATUS", 0x08),
        ("TRANSFER_CNT", 0x30),
    ] {
        let v = a.read32(&wa, eth(x), TXQ2 + o).unwrap();
        println!("MEASURE sender x={x} TXQ2_{n} = {v:#x}");
    }
    a.write32(&wa, eth(x), TXQ2 + 0x14, BUF as u32).unwrap();
    a.write32(&wa, eth(x), TXQ2 + 0x18, LEN as u32).unwrap();
    a.write32(&wa, eth(x), TXQ2 + 0x1C, BUF as u32).unwrap();
    a.write32(&wa, eth(x), TXQ2 + 0x04, 2).unwrap();
    // EthernetTxRx.md: read CMD back before STATUS so the two are not reordered.
    let _ = a.read32(&wa, eth(x), TXQ2 + 0x04).unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    while a.read32(&wa, eth(x), TXQ2 + 0x08).unwrap() & (1 << 16) != 0 {
        assert!(Instant::now() < deadline, "TXQ2 never accepted the command");
    }

    let mut got = vec![0u8; LEN + 64];
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        b.read(&wb, eth(x), BUF, &mut got).unwrap();
        if got[..LEN] == data[..] || Instant::now() > deadline {
            break;
        }
    }
    let same = got[..LEN].iter().zip(&data).filter(|(g, d)| g == d).count();
    println!("MEASURE x={x}: {same}/{LEN} bytes arrived");
    assert_eq!(same, LEN, "the payload did not arrive intact");
    assert!(got[LEN..].iter().all(|&v| v == 0xEE), "wrote past the end");
    let cnt = a.read32(&wa, eth(x), TXQ2 + 0x30).unwrap();
    println!("MEASURE sender x={x} TXQ2_TRANSFER_CNT after = {cnt:#x}");
    assert!(port_up(a, x) && port_up(b, x), "the link dropped");
}

fn two_cards(f: impl FnOnce(&mut Dev<'_>, &mut Dev<'_>)) {
    assert_on_silicon();
    if let Err(e) = fork_scope(|| {
        let mut a = open_card(0);
        let mut b = open_card(1);
        f(&mut a, &mut b);
        scrub(&mut a);
        scrub(&mut b);
    }) {
        panic!("{e}");
    }
}

#[test]
fn host_driven_tt_link_card0_to_card1_x3() {
    two_cards(|a, b| send(a, b, 3, 0x11));
}

#[test]
fn host_driven_tt_link_card1_to_card0_x3() {
    two_cards(|a, b| send(b, a, 3, 0x22));
}

#[test]
fn host_driven_tt_link_card0_to_card1_x13() {
    two_cards(|a, b| send(a, b, 13, 0x33));
}

#[test]
fn host_driven_tt_link_card1_to_card0_x13() {
    two_cards(|a, b| send(b, a, 13, 0x44));
}

/// S5 on silicon: Rust on E1, via the checked `tt_device::ethernet` API, with the
/// grid from the chip's own ARC. E0 is left running throughout; the link must
/// still be Up afterwards.
fn e1_heartbeat(x: u8) {
    use tt_isa::mailbox::{offset, status};
    two_cards(|a, _| {
        let w = a.alloc_window(WindowKind::TwoMib).unwrap();
        let grid = a.ethernet_grid(&w).unwrap();
        let t = grid.tile(x).expect("an enabled Ethernet tile");
        let before = a.eth_link_state(&w, t).unwrap();
        println!(
            "MEASURE x={x} before: {before:?}, reset_pc={:#x}",
            a.e1_reset_pc(&w, t).unwrap()
        );
        a.load_and_start_e1(&w, t, tt_firmware_images::ETH_E1)
            .unwrap();
        let s = a
            .wait_for_e1_status(&w, t, 0, |s| s == status::RUNNING)
            .unwrap();
        assert!(s.is_ok(), "E1 did not start: {s:?}");
        let mut last = a.e1_mailbox_read(&w, t, offset::HEARTBEAT).unwrap();
        for _ in 0..4 {
            std::thread::sleep(Duration::from_millis(1));
            let now = a.e1_mailbox_read(&w, t, offset::HEARTBEAT).unwrap();
            assert_ne!(now, last, "heartbeat did not climb");
            last = now;
        }
        a.park_e1(&w, t).unwrap();
        let held = a.e1_mailbox_read(&w, t, offset::HEARTBEAT).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(
            a.e1_mailbox_read(&w, t, offset::HEARTBEAT).unwrap(),
            held,
            "parked E1 ran on"
        );
        let after = a.eth_link_state(&w, t).unwrap();
        println!(
            "MEASURE x={x} after: {after:?}, soft_reset={:#x}",
            a.read32(&w, t.coord(), tt_isa::eth::SOFT_RESET_0).unwrap()
        );
        assert_eq!(before.port, after.port, "the port changed state");
    });
}

#[test]
fn e1_heartbeat_on_a_portless_tile() {
    e1_heartbeat(1);
}

#[test]
fn e1_heartbeat_on_a_live_link_tile() {
    e1_heartbeat(3);
}

// --- The E1 mover and the sharded matmul, across the cable ------------------

mod mover {
    use super::*;
    use tt_kernels::link::{discover, Dest, Dir, Link, LinkError, Mover, Source};
    use tt_tests::harness::tile;

    /// The first cabled pair, read from both chips.
    fn first_link(a: &mut Dev<'_>, b: &mut Dev<'_>) -> Link {
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let (ga, gb) = (a.ethernet_grid(&wa).unwrap(), b.ethernet_grid(&wb).unwrap());
        let links = discover(a, &wa, &ga, b, &wb, &gb).unwrap();
        println!("MEASURE links {links:?}");
        assert_eq!(links.len(), 2, "one QSFP-DD port: two Ethernet links");
        links[0]
    }

    #[test]
    fn staged_data_lands_on_the_other_card_both_ways() {
        two_cards(|a, b| {
            let l = first_link(a, b);
            let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
            let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
            let mut m = Mover::start(a, &wa, b, &wb, l, tt_firmware_images::ETH_E1).unwrap();
            for (i, dir) in [Dir::AToB, Dir::BToA, Dir::AToB].into_iter().enumerate() {
                let data = pattern(0x50 + i as u8).repeat(32); // 128 KiB, a whole transfer
                let (s, ws, r, wr) = match dir {
                    Dir::AToB => (&mut *a, &wa, &mut *b, &wb),
                    Dir::BToA => (&mut *b, &wb, &mut *a, &wa),
                };
                m.stage(s, ws, dir, &data).unwrap();
                m.send(s, ws, dir, Source::Staged, Dest::Landed, data.len() as u32)
                    .unwrap_or_else(|e| panic!("{dir:?}: {e}"));
                let mut got = vec![0u8; data.len()];
                m.landed(r, wr, dir, &mut got).unwrap();
                assert!(got == data, "{dir:?}: payload differs");
            }
            for (d, w, t) in [(&mut *a, &wa, l.a), (&mut *b, &wb, l.b)] {
                d.park_e1(w, t).unwrap();
                assert!(d.eth_link_state(w, t).unwrap().is_up(), "the link dropped");
            }
        });
    }

    #[test]
    fn tensix_to_tensix_across_cards_without_the_host_touching_the_receiver() {
        two_cards(|a, b| {
            let l = first_link(a, b);
            let (src, dst) = (tile(a, 3, 4), tile(b, 4, 4));
            let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
            let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
            const LEN: usize = 128 * 1024;
            let data = pattern(0x77).repeat(LEN / 4096);
            a.write(&wa, src, 0x8_0000, &data).unwrap();
            b.write(&wb, dst, 0x9_0000, &vec![0xEE; LEN + 256]).unwrap();
            let mut m = Mover::start(a, &wa, b, &wb, l, tt_firmware_images::ETH_E1).unwrap();
            let t0 = Instant::now();
            m.send(
                a,
                &wa,
                Dir::AToB,
                Source::Tensix(src, 0x8_0000),
                Dest::Tensix(dst, 0x9_0000),
                LEN as u32,
            )
            .unwrap_or_else(|e| panic!("{e}"));
            println!(
                "MEASURE 128 KiB Tensix->Tensix across cards, acknowledged in {:?}",
                t0.elapsed()
            );
            let mut got = vec![0u8; LEN + 256];
            b.read(&wb, dst, 0x9_0000, &mut got).unwrap();
            assert!(got[..LEN] == data[..], "payload differs");
            assert!(got[LEN..].iter().all(|&v| v == 0xEE), "wrote past the end");
            a.park_e1(&wa, l.a).unwrap();
            b.park_e1(&wb, l.b).unwrap();
        });
    }

    #[test]
    fn without_a_receiver_there_is_no_acknowledgement() {
        two_cards(|a, b| {
            let l = first_link(a, b);
            let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
            let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
            let mut m = Mover::start(a, &wa, b, &wb, l, tt_firmware_images::ETH_E1).unwrap();
            b.park_e1(&wb, l.b).unwrap();
            m.stage(a, &wa, Dir::AToB, &pattern(5)[..256]).unwrap();
            match m.send(a, &wa, Dir::AToB, Source::Staged, Dest::Landed, 256) {
                Err(LinkError::TimedOut { .. }) => {}
                other => panic!("expected a timeout, got {other:?}"),
            }
            a.park_e1(&wa, l.a).unwrap();
        });
    }

    #[test]
    fn a_matmul_sharded_across_the_cable_is_bit_identical() {
        use tt_kernels::matmul::{Fidelity, SrcRoute};
        use tt_kernels::shard::{Chip, Fabric};
        assert_on_silicon();
        if let Err(e) = fork_scope(|| {
            let mut a = open_card(0);
            let mut b = open_card(1);
            let l = first_link(&mut a, &mut b);
            let (ca, ra) = (tile(&mut a, 3, 4), tile(&mut a, 4, 4));
            let (cb, rb) = (tile(&mut b, 3, 4), tile(&mut b, 4, 4));
            let chips = vec![Chip::new(a, ca, ra).unwrap(), Chip::new(b, cb, rb).unwrap()];
            let mut fab = Fabric::new(
                chips,
                &[(0, 1, l)],
                tt_firmware_images::ROLES,
                tt_firmware_images::ETH_E1,
            )
            .unwrap_or_else(|e| panic!("{e}"));
            let mkn = [64, 784, 128];
            let (m, k, n) = (mkn[0], mkn[1], mkn[2]);
            let f = |len: usize, seed: u32| -> Vec<f32> {
                let mut s = seed | 1;
                (0..len)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 17;
                        s ^= s << 5;
                        (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
                    })
                    .collect()
            };
            let (x, w) = (f(m * k, 11), f(k * n, 23));
            let (route, fid) = (SrcRoute::Tf32FromFp32, Fidelity::HiFi4);
            let t0 = Instant::now();
            let sharded = fab
                .matmul(&x, &w, mkn, route, fid, 400_000)
                .unwrap_or_else(|e| panic!("{e}"));
            println!("MEASURE sharded [64,784]@[784,128] in {:?}", t0.elapsed());
            let t0 = Instant::now();
            let single = tt_kernels::session::matmul_on(
                &mut fab.chips[0].dev,
                ca,
                &tt_firmware_images::ROLES,
                &x,
                &w,
                mkn,
                route,
                fid,
                400_000,
            )
            .unwrap();
            println!("MEASURE single-chip in {:?}", t0.elapsed());
            let diff = sharded
                .iter()
                .zip(&single)
                .filter(|(s, o)| s.to_bits() != o.to_bits())
                .count();
            assert_eq!(
                diff, 0,
                "{diff} elements differ from the single-chip product"
            );
            assert!(single.iter().any(|&v| v != 0.0));
            for c in fab.chips.iter_mut() {
                scrub(&mut c.dev);
            }
        }) {
            panic!("{e}");
        }
    }
}
