//! Phase 8 gates, dual-chip simulator: Ethernet tiles, TT-link writes, and E1.
//!
//! The silicon twins are in `silicon_eth_link.rs`. ttsim models every tile of
//! both chips present ([`Ethernet::FULL`]) and does not model the base
//! firmware's chip-info exchange, so the link map here is the measured one
//! (`probe_eth::host_driven_tt_link_l1_write`), not one read from the chips.

#![cfg(not(feature = "silicon"))]

use tt_device::{tlb::WindowKind, Device};
use tt_isa::eth::{self, EthTile, Ethernet, PortStatus};
use tt_isa::mailbox::offset;
use tt_ttsim::{fork_scope, Simulator};

type Dev<'a> = Device<tt_ttsim::LibTtsim<'a>>;

/// `bh_x2`'s two links, as `(chip 0 x, chip 1 x)`. Measured: a TT-link write
/// from each end lands on exactly one tile of the other chip. The first is
/// `tt_tests::topology::BH_X2`'s; that table keeps one link per pair, and this
/// gate wants both.
const BH_X2_LINKS: [(u8, u8); 2] = [(2, 12), (15, 5)];
const _: () = assert!(
    tt_tests::topology::BH_X2[0].2 .0 == BH_X2_LINKS[0].0
        && tt_tests::topology::BH_X2[0].2 .1 == BH_X2_LINKS[0].1
);

fn tile(x: u8) -> EthTile {
    Ethernet::FULL.tile(x).unwrap()
}

#[track_caller]
fn in_two_chips(f: impl FnOnce(&mut Dev<'_>, &mut Dev<'_>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = Simulator::open_path(tt_ttsim::x2_lib_path()).unwrap();
        let mut ts = sim.transports().into_iter();
        let mut a = Device::open(ts.next().unwrap()).unwrap();
        let mut b = Device::open(ts.next().unwrap()).unwrap();
        f(&mut a, &mut b);
    }) {
        panic!("{e}");
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(13).wrapping_add(seed))
        .collect()
}

/// Send over one link and return what arrived, plus the sentinel beyond it.
fn send(a: &mut Dev<'_>, b: &mut Dev<'_>, from: u8, to: u8, data: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
    let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
    let (src, dst) = (eth::BUFFERS.start, eth::BUFFERS.start + 0x1_0000);
    a.eth_write(&wa, tile(from), src, data).unwrap();
    b.eth_write(&wb, tile(to), dst, &vec![0xEE; data.len() + 64])
        .unwrap();
    a.eth_tt_link_write(&wa, tile(from), src, dst, data.len())
        .unwrap()
        .unwrap();
    // Delivery is not latching: give the link time, then look.
    a.tick(100_000);
    let mut got = vec![0u8; data.len() + 64];
    b.eth_read(&wb, tile(to), dst, &mut got).unwrap();
    let tail = got.split_off(data.len());
    (got, tail)
}

#[test]
fn every_tile_reports_its_port_and_the_links_are_up() {
    in_two_chips(|a, b| {
        let w = a.alloc_window(WindowKind::TwoMib).unwrap();
        let up: Vec<u8> = Ethernet::FULL
            .tiles()
            .filter(|&x| a.eth_link_state(&w, tile(x)).unwrap().port == PortStatus::Up)
            .collect();
        assert_eq!(up, [2, 15], "chip 0's live tiles");
        let w = b.alloc_window(WindowKind::TwoMib).unwrap();
        let up: Vec<u8> = Ethernet::FULL
            .tiles()
            .filter(|&x| b.eth_link_state(&w, tile(x)).unwrap().port == PortStatus::Up)
            .collect();
        assert_eq!(
            up,
            [5, 12],
            "chip 1's live tiles, in endpoint order (E8, E9)"
        );
    });
}

#[test]
fn a_tt_link_write_arrives_intact_on_the_linked_tile_only() {
    in_two_chips(|a, b| {
        for (i, (x0, x1)) in BH_X2_LINKS.into_iter().enumerate() {
            let data = pattern(4096, i as u8);
            let (got, tail) = send(a, b, x0, x1, &data);
            assert_eq!(got, data, "chip 0 x={x0} -> chip 1 x={x1}");
            assert!(tail.iter().all(|&v| v == 0xEE), "wrote past the end");
            let data = pattern(4096, 0x80 | i as u8);
            let (got, _) = send(b, a, x1, x0, &data);
            assert_eq!(got, data, "chip 1 x={x1} -> chip 0 x={x0}");
        }
    });
}

#[test]
fn a_corrupted_byte_moves_exactly_one_byte() {
    in_two_chips(|a, b| {
        let (x0, x1) = BH_X2_LINKS[0];
        let mut data = pattern(4096, 7);
        let clean = data.clone();
        data[1234] ^= 0x5A;
        let (got, _) = send(a, b, x0, x1, &data);
        let diffs: Vec<usize> = (0..got.len()).filter(|&i| got[i] != clean[i]).collect();
        assert_eq!(diffs, [1234]);
    });
}

#[test]
fn the_wrong_tile_of_the_peer_receives_nothing() {
    in_two_chips(|a, b| {
        let (x0, _) = BH_X2_LINKS[0];
        let (_, other) = BH_X2_LINKS[1];
        let (got, _) = send(a, b, x0, other, &pattern(4096, 9));
        assert!(
            got.iter().all(|&v| v == 0xEE),
            "a tile off the link received data"
        );
    });
}

#[test]
fn firmware_l1_and_misalignment_are_refused_before_anything_is_sent() {
    in_two_chips(|a, _| {
        let w = a.alloc_window(WindowKind::TwoMib).unwrap();
        let t = tile(2);
        assert!(a.eth_write(&w, t, eth::BOOT_RESULTS, &[0; 4]).is_err());
        assert!(a
            .eth_tt_link_write(&w, t, eth::BUFFERS.start, eth::BOOT_PARAMS, 16)
            .is_err());
        assert!(a
            .eth_tt_link_write(&w, t, eth::BUFFERS.start + 8, eth::BUFFERS.start, 16)
            .is_err());
        assert!(a
            .eth_tt_link_write(&w, t, eth::BUFFERS.start, eth::BUFFERS.start, 24)
            .is_err());
    });
}

#[test]
fn rust_runs_on_e1() {
    in_two_chips(|a, _| {
        let w = a.alloc_window(WindowKind::TwoMib).unwrap();
        let t = tile(2);
        a.load_and_start_e1(&w, t, tt_firmware_images::ETH_E1)
            .unwrap_or_else(|e| panic!("{e}"));
        a.wait_for_e1_status(&w, t, 200_000, |s| s == tt_isa::mailbox::status::RUNNING)
            .unwrap()
            .unwrap();
        let mut last = 0;
        for _ in 0..4 {
            a.tick(10_000);
            let now = a.e1_mailbox_read(&w, t, offset::HEARTBEAT).unwrap();
            assert!(now > last, "heartbeat did not climb: {last} then {now}");
            last = now;
        }
        a.park_e1(&w, t).unwrap();
        assert!(a.is_e1_in_reset(&w, t).unwrap());
    });
}

#[test]
fn e1_held_in_reset_does_nothing() {
    in_two_chips(|a, _| {
        let w = a.alloc_window(WindowKind::TwoMib).unwrap();
        let t = tile(2);
        a.load_and_start_e1(&w, t, tt_firmware_images::ETH_E1)
            .unwrap_or_else(|e| panic!("{e}"));
        a.park_e1(&w, t).unwrap();
        a.tick(10_000);
        let held = a.e1_mailbox_read(&w, t, offset::HEARTBEAT).unwrap();
        a.tick(100_000);
        assert_eq!(a.e1_mailbox_read(&w, t, offset::HEARTBEAT).unwrap(), held);
    });
}

mod mover {
    use super::*;
    use tt_isa::noc::{Noc0, NocCoord};
    use tt_kernels::link::{Dest, Dir, Link, LinkError, Mover, Source};

    fn link() -> Link {
        let (x0, x1) = BH_X2_LINKS[0];
        Link {
            a: tile(x0),
            b: tile(x1),
        }
    }

    fn tensix(x: u8, y: u8) -> NocCoord<Noc0> {
        assert!(tt_isa::noc::grid::is_tensix_geometry(x, y));
        NocCoord::new(x, y).unwrap()
    }

    #[test]
    fn staged_data_lands_on_the_partner_both_ways() {
        in_two_chips(|a, b| {
            let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
            let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
            let mut m = Mover::start(a, &wa, b, &wb, link(), tt_firmware_images::ETH_E1).unwrap();
            for (dir, seed) in [(Dir::AToB, 1u8), (Dir::BToA, 2), (Dir::AToB, 3)] {
                let data = pattern(8192, seed);
                let got = match dir {
                    Dir::AToB => round_trip(&mut m, a, &wa, b, &wb, dir, &data),
                    Dir::BToA => round_trip(&mut m, b, &wb, a, &wa, dir, &data),
                };
                assert_eq!(got, data, "{dir:?}");
            }
        });
    }

    fn round_trip<S: tt_device::Transport, R: tt_device::Transport>(
        m: &mut Mover,
        s: &mut Device<S>,
        ws: &tt_device::Window,
        r: &mut Device<R>,
        wr: &tt_device::Window,
        dir: Dir,
        data: &[u8],
    ) -> Vec<u8> {
        m.stage(s, ws, dir, data).unwrap();
        m.send(s, ws, dir, Source::Staged, Dest::Landed, data.len() as u32)
            .unwrap_or_else(|e| panic!("{dir:?}: {e}"));
        let mut got = vec![0u8; data.len()];
        m.landed(r, wr, dir, &mut got).unwrap();
        got
    }

    #[test]
    fn tensix_to_tensix_across_chips_without_the_host_touching_the_receiver() {
        in_two_chips(|a, b| {
            let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
            let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
            let (src, dst) = (tensix(3, 4), tensix(6, 7));
            const LEN: usize = 64 * 1024;
            let data = pattern(LEN, 0x42);
            a.write(&wa, src, 0x8_0000, &data).unwrap();
            b.write(&wb, dst, 0x9_0000, &vec![0xEE; LEN + 256]).unwrap();
            let mut m = Mover::start(a, &wa, b, &wb, link(), tt_firmware_images::ETH_E1).unwrap();
            // From here to the read-back, only chip 0 is accessed.
            m.send(
                a,
                &wa,
                Dir::AToB,
                Source::Tensix(src, 0x8_0000),
                Dest::Tensix(dst, 0x9_0000),
                LEN as u32,
            )
            .unwrap_or_else(|e| panic!("{e}"));
            let mut got = vec![0u8; LEN + 256];
            b.read(&wb, dst, 0x9_0000, &mut got).unwrap();
            assert_eq!(&got[..LEN], &data[..], "payload");
            assert!(got[LEN..].iter().all(|&v| v == 0xEE), "wrote past the end");
        });
    }

    #[test]
    fn without_a_receiver_there_is_no_acknowledgement() {
        in_two_chips(|a, b| {
            let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
            let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
            let l = link();
            let mut m = Mover::start(a, &wa, b, &wb, l, tt_firmware_images::ETH_E1).unwrap();
            b.park_e1(&wb, l.b).unwrap();
            m.stage(a, &wa, Dir::AToB, &pattern(256, 5)).unwrap();
            match m.send(a, &wa, Dir::AToB, Source::Staged, Dest::Landed, 256) {
                Err(LinkError::TimedOut { .. }) => {}
                other => panic!("expected a timeout, got {other:?}"),
            }
        });
    }

    #[test]
    fn a_misaligned_or_oversized_request_is_refused_before_it_is_sent() {
        in_two_chips(|a, b| {
            let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
            let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
            let mut m = Mover::start(a, &wa, b, &wb, link(), tt_firmware_images::ETH_E1).unwrap();
            let t = tensix(3, 4);
            for (src, len) in [
                (Source::Staged, 24),
                (Source::Staged, 0),
                (Source::Staged, 0x2_0010),
                (Source::Tensix(t, 8), 16),
            ] {
                assert!(matches!(
                    m.send(a, &wa, Dir::AToB, src, Dest::Landed, len),
                    Err(LinkError::Invalid(_))
                ));
            }
        });
    }
}
