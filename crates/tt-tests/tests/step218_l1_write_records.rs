//! `op::WRITE_L1`: RISCV NC writes its own L1 into another tile's L1
//! (`docs/plans/planned-l1-residency.md` Phase 2b, the first record of the hang class: a write to
//! a tile the chip does not have is a NoC access nothing answers).
//!
//! The claims:
//! - **The bytes land, and nothing else moves.** Writes of 64 B to 16 KiB from the gate tile into
//!   two peers' persistent windows, on NoC #1 (the session's) and NoC #0, leave each peer's window
//!   holding the source bytes where written and the poison elsewhere, the guard below the window
//!   intact, and a tile no entry names untouched. The list's end waits for the writes (they are
//!   response-marked), so a read right after it sees them.
//! - **The host refuses what its session does not own** (`DataMover::set_l1_writes`): a tile not
//!   in its list -- a geometry-valid tile this chip might not have among them -- an address outside
//!   the window, any `WRITE_L1` before the session names its tiles, a list naming a coordinate no
//!   Tensix tile has, and the entry on RISCV B.
//! - **A multicast lands in every tile of its rectangle** (`op::WRITE_L1_MCAST`, NoC #1): four
//!   peers in a 2x2 rectangle hold the bytes, two tiles beside it the poison; in one list with
//!   unicasts before and after it, every write lands (NC counts the multicast's acknowledgements
//!   with its earlier writes landed and nothing else in flight). The host refuses a rectangle with
//!   a tile not the session's; the tile refuses one holding itself, a reversed one, one across the
//!   non-memory column, and any through NoC #0, whose acknowledgement counter B's barrier release
//!   moves.
//! - **A run of multicasts counted once lands as lone ones do** (`dm::MCAST_RUN_ACKS`, 2f (c)):
//!   multicasts marked as one run, between unicasts, two runs split by an unmarked multicast, and a
//!   run ending the list, each leave every tile of the rectangle holding the bytes when the list
//!   ends, and the unicasts after a run land too.
//! - **B reads a peer's L1** (`op::READ_L1`, 2f): 64 B to 16 KiB from two peers' windows and the
//!   gate tile's own, at three skews, land byte for byte in the gate tile's L1 and nothing past them.
//!   The host refuses a read from a tile not the session's, a harvested column's, outside the window
//!   (the remote side), before any tile is named, and on NC; the tile refuses the decode's cases
//!   sent raw, and writes nothing.
//! - **The tile refuses a malformed entry by itself** (sent past the host's check,
//!   `enqueue_raw`): a coordinate outside the Tensix rows and columns, a length of zero or past one
//!   request, an address outside `DATA`, addresses not congruent mod 16, and the entry on B. It
//!   writes nothing for any of them. Only what the card's decode refuses is sent raw: a
//!   geometry-valid tile the chip may not have is never sent past the host.

use tt_device::tlb::WindowKind;
use tt_device::Window;
use tt_isa::dm::{self, op, Mover};
use tt_isa::noc::niu::Niu;
use tt_isa::noc::Noc0;
use tt_kernels::dm::{DataMover, DmError, L1Writes};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile, Dev};

const POISON: u8 = 0xA5;
/// The source, in the gate tile's arena.
const SRC: u32 = 0x4_0000;
/// The window the writes may land in: the top of every peer's arena.
const WINDOW_LEN: u64 = 64 << 10;
/// Poisoned below the window, and checked untouched.
const GUARD: u64 = 16 << 10;
/// The peers written to, and one written to by nothing.
const PEERS: [(u8, u8); 2] = [
    (GATE_TILE.0 + 1, GATE_TILE.1),
    (GATE_TILE.0, GATE_TILE.1 + 1),
];
const BYSTANDER: (u8, u8) = (GATE_TILE.0 + 1, GATE_TILE.1 + 1);
/// Lengths written, each at the next 64-byte boundary of the window.
const LENS: [u32; 5] = [64, 1040, 4160, 16384, 8256];

fn window() -> std::ops::Range<u64> {
    tt_isa::l1::DATA.end - WINDOW_LEN..tt_isa::l1::DATA.end
}

fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as u8
        })
        .collect()
}

fn start(d: &mut Dev<'_>, w: &Window, mover: Mover) -> DataMover<Noc0> {
    let dram = d.dram_grid(w).unwrap();
    let t = tile(d, GATE_TILE.0, GATE_TILE.1);
    let image = match mover {
        Mover::B => tt_firmware_images::DM_B.1,
        _ => tt_firmware_images::DM_NC.1,
    };
    DataMover::start_on(d, w, t, &dram, mover, image).unwrap()
}

/// Every peer's window and the guard below it poisoned.
fn poison(d: &mut Dev<'_>, w: &Window, tiles: &[(u8, u8)]) {
    let from = window().start - GUARD;
    for &(x, y) in tiles {
        let t = tile(d, x, y);
        d.l1_write(w, t, from, &vec![POISON; (WINDOW_LEN + GUARD) as usize])
            .unwrap();
    }
}

fn read_back(d: &mut Dev<'_>, w: &Window, at: (u8, u8)) -> Vec<u8> {
    let t = tile(d, at.0, at.1);
    let mut back = vec![0u8; (WINDOW_LEN + GUARD) as usize];
    d.l1_read(w, t, window().start - GUARD, &mut back).unwrap();
    back
}

/// The entries writing `LENS` from the source into `peer`'s window, and the bytes the window
/// and guard should hold after, starting at `skew` bytes into each 64-byte unit.
fn writes_to(peer: (u8, u8), src: &[u8], skew: u32) -> (Vec<[u32; 8]>, Vec<u8>) {
    let mut want = vec![POISON; (WINDOW_LEN + GUARD) as usize];
    let mut list = Vec::new();
    let (mut from, mut to) = (SRC + skew, window().start as u32 + skew);
    for len in LENS {
        list.push([
            op::WRITE_L1,
            peer.0 as u32,
            peer.1 as u32,
            to,
            from,
            len,
            0,
            0,
        ]);
        let at = (to as u64 - (window().start - GUARD)) as usize;
        let s = (from - SRC) as usize;
        want[at..at + len as usize].copy_from_slice(&src[s..s + len as usize]);
        from += len.next_multiple_of(64);
        to += len.next_multiple_of(64);
    }
    (list, want)
}

fn first_difference(got: &[u8], want: &[u8]) -> Option<usize> {
    got.iter().zip(want).position(|(a, b)| a != b)
}

#[test]
fn writes_land_in_each_peers_window_on_both_nocs() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let mut nc = start(d, &w, Mover::NC);
        nc.set_l1_writes(L1Writes {
            tiles: PEERS.to_vec(),
            window: window(),
        })
        .unwrap();
        let src = pattern(64 << 10, 0x218);
        let gate = nc.tile();
        d.l1_write(&w, gate, SRC as u64, &src).unwrap();
        let mut written = 0;
        for (niu, skew) in [(Niu::Noc1, 0), (Niu::Noc0, 16), (Niu::Noc1, 48)] {
            nc.set_write_noc(d, &w, niu).unwrap();
            let mut all = PEERS.to_vec();
            all.push(BYSTANDER);
            poison(d, &w, &all);
            let mut list = Vec::new();
            let mut wants = Vec::new();
            for (k, &peer) in PEERS.iter().enumerate() {
                // Each peer from its own part of the source.
                let (mut l, want) = writes_to(peer, &src[k * 1024..], skew);
                for e in &mut l {
                    e[4] += k as u32 * 1024;
                }
                written += l.len();
                list.extend(l);
                wants.push(want);
            }
            nc.run_list(d, &w, &list).unwrap();
            for (&peer, want) in PEERS.iter().zip(&wants) {
                let got = read_back(d, &w, peer);
                assert!(
                    first_difference(&got, want).is_none(),
                    "{niu:?}, skew {skew}: peer {peer:?} differs at byte {:?} past the guard's start",
                    first_difference(&got, want)
                );
            }
            let untouched = read_back(d, &w, BYSTANDER);
            assert!(
                untouched.iter().all(|&b| b == POISON),
                "{niu:?}: the bystander {BYSTANDER:?} was written"
            );
        }
        assert_eq!(written, 3 * PEERS.len() * LENS.len());
        nc.stop(d, &w).unwrap();
    });
}

#[test]
fn the_host_refuses_what_its_session_does_not_own() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let mut nc = start(d, &w, Mover::NC);
        let to = window().start as u32;
        let e = |x: u8, y: u8, to: u32| [op::WRITE_L1, x as u32, y as u32, to, SRC, 4160, 0, 0];
        let refused = |r: Result<(), DmError>, what: &str| match r {
            Err(DmError::Invalid(code)) => code,
            other => panic!("the host took {what}: {other:?}"),
        };
        // No tiles named yet: every write refused.
        let r = nc.run_list(d, &w, &[e(PEERS[0].0, PEERS[0].1, to)]);
        assert_eq!(
            refused(r, "a write before any tile was named"),
            dm::error::RANGE
        );
        // A coordinate no Tensix tile has is no tile to name.
        let r = nc.set_l1_writes(L1Writes {
            tiles: vec![(8, GATE_TILE.1)],
            window: window(),
        });
        assert!(r.is_err(), "the non-memory column named: {r:?}");
        let r = nc.set_l1_writes(L1Writes {
            tiles: PEERS.to_vec(),
            window: 0x1_0000..tt_isa::l1::DATA.end,
        });
        assert!(r.is_err(), "a window below DATA: {r:?}");
        // A window ending below `DATA`'s end, so a write past it is still inside `DATA`.
        let end = window().end - 8192;
        nc.set_l1_writes(L1Writes {
            tiles: PEERS.to_vec(),
            window: window().start..end,
        })
        .unwrap();
        let cases = [
            ("a tile not the session's", e(BYSTANDER.0, BYSTANDER.1, to)),
            // Translated X 15 and 16 are fused off on both cards here: never sent past the host.
            ("a harvested column's tile", e(16, GATE_TILE.1, to)),
            ("below the window", e(PEERS[0].0, PEERS[0].1, to - 64)),
            (
                "past the window",
                e(PEERS[0].0, PEERS[0].1, end as u32 - 4096),
            ),
        ];
        for (what, entry) in cases {
            let r = nc.run_list(d, &w, &[entry]);
            assert_eq!(refused(r, what), dm::error::RANGE, "{what}");
        }
        // What it admits, it runs.
        nc.run_list(d, &w, &[e(PEERS[1].0, PEERS[1].1, to)])
            .unwrap();
        nc.stop(d, &w).unwrap();
        // RISCV B writes no L1 of another tile.
        let mut b = start(d, &w, Mover::B);
        let r = b.run_list(d, &w, &[e(PEERS[0].0, PEERS[0].1, to)]);
        assert_eq!(refused(r, "WRITE_L1 on B"), dm::error::DIRECTION);
        b.stop(d, &w).unwrap();
    });
}

#[test]
fn the_tile_refuses_a_malformed_write_by_itself() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let (x, y) = (PEERS[0].0 as u32, PEERS[0].1 as u32);
        let to = window().start as u32;
        let bad: [(&str, [u32; 8], u32, Mover); 7] = [
            (
                "the non-memory column",
                [op::WRITE_L1, 8, y, to, SRC, 64, 0, 0],
                dm::error::RANGE,
                Mover::NC,
            ),
            (
                "the DRAM column",
                [op::WRITE_L1, 0, y, to, SRC, 64, 0, 0],
                dm::error::RANGE,
                Mover::NC,
            ),
            (
                "a length of none",
                [op::WRITE_L1, x, y, to, SRC, 0, 0, 0],
                dm::error::LENGTH,
                Mover::NC,
            ),
            (
                "past one request",
                [op::WRITE_L1, x, y, to - 32768, SRC, 16384 + 16, 0, 0],
                dm::error::LENGTH,
                Mover::NC,
            ),
            (
                "below DATA",
                [op::WRITE_L1, x, y, 0x1_0000, SRC, 64, 0, 0],
                dm::error::ALIGNMENT,
                Mover::NC,
            ),
            (
                "addresses not congruent mod 16",
                [op::WRITE_L1, x, y, to + 8, SRC, 64, 0, 0],
                dm::error::ALIGNMENT,
                Mover::NC,
            ),
            (
                "the entry on B",
                [op::WRITE_L1, x, y, to, SRC, 64, 0, 0],
                dm::error::DIRECTION,
                Mover::B,
            ),
        ];
        for (what, entry, code, mover) in bad {
            // A fresh mover each time: a refused list stops the one it was sent to.
            let mut m = start(d, &w, mover);
            poison(d, &w, &PEERS);
            let n = m.enqueue_raw(d, &w, &[entry]).unwrap();
            let got = m.wait_for(d, &w, n);
            assert!(
                matches!(got, Err(DmError::Queued { code: c, .. }) if c == code),
                "the tile took {what}: {got:?}"
            );
            for peer in PEERS {
                assert!(
                    read_back(d, &w, peer).iter().all(|&b| b == POISON),
                    "{what}: peer {peer:?} was written"
                );
            }
            m.stop(d, &w).unwrap();
        }
    });
}

/// The multicast's rectangle (beside the gate tile, not holding it) and two tiles beside it.
const RECT: ((u8, u8), (u8, u8)) = (
    (GATE_TILE.0 + 1, GATE_TILE.1),
    (GATE_TILE.0 + 2, GATE_TILE.1 + 1),
);
const BESIDE: [(u8, u8); 2] = [
    (GATE_TILE.0, GATE_TILE.1 + 1),
    (GATE_TILE.0 + 3, GATE_TILE.1 + 1),
];

fn rect_tiles() -> Vec<(u8, u8)> {
    let ((x0, y0), (x1, y1)) = RECT;
    (y0..=y1)
        .flat_map(|y| (x0..=x1).map(move |x| (x, y)))
        .collect()
}

#[test]
fn a_multicast_lands_in_every_tile_of_its_rectangle() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let mut nc = start(d, &w, Mover::NC);
        let mut ours = rect_tiles();
        ours.extend(BESIDE);
        nc.set_l1_writes(L1Writes {
            tiles: ours,
            window: window(),
        })
        .unwrap();
        nc.set_write_noc(d, &w, Niu::Noc1).unwrap();
        let src = pattern(64 << 10, 0x2181);
        d.l1_write(&w, nc.tile(), SRC as u64, &src).unwrap();
        let corners = dm::mcast_corners(RECT.0, RECT.1);
        let base = window().start - GUARD;
        for skew in [0u32, 48] {
            let mut all = rect_tiles();
            all.extend(BESIDE);
            poison(d, &w, &all);
            // A unicast to one beside tile, the multicasts, a unicast to the other.
            let (mut list, mut want_beside0) = writes_to(BESIDE[0], &src, skew);
            list.truncate(1);
            want_beside0[(window().start - base) as usize + skew as usize + 64..].fill(POISON);
            let mut want_rect = vec![POISON; (WINDOW_LEN + GUARD) as usize];
            let (mut from, mut to) = (SRC + skew, window().start as u32 + skew);
            for len in LENS {
                list.push([op::WRITE_L1_MCAST, corners, 0, to, from, len, 0, 0]);
                let at = (to as u64 - base) as usize;
                let s = (from - SRC) as usize;
                want_rect[at..at + len as usize].copy_from_slice(&src[s..s + len as usize]);
                from += len.next_multiple_of(64);
                to += len.next_multiple_of(64);
            }
            let (last, want_beside1) = writes_to(BESIDE[1], &src, skew);
            list.push(last[0]);
            let mut want_beside1 = want_beside1;
            want_beside1[(window().start - base) as usize + skew as usize + 64..].fill(POISON);
            nc.run_list(d, &w, &list).unwrap();
            for t in rect_tiles() {
                let got = read_back(d, &w, t);
                assert!(
                    first_difference(&got, &want_rect).is_none(),
                    "skew {skew}: {t:?} differs at byte {:?}",
                    first_difference(&got, &want_rect)
                );
            }
            for (t, want) in BESIDE.iter().zip([&want_beside0, &want_beside1]) {
                let got = read_back(d, &w, *t);
                assert!(
                    first_difference(&got, want).is_none(),
                    "skew {skew}: beside {t:?} differs at byte {:?}",
                    first_difference(&got, want)
                );
            }
        }
        // The host: a rectangle with a tile not the session's, or over a harvested column.
        nc.set_l1_writes(L1Writes {
            tiles: rect_tiles(),
            window: window(),
        })
        .unwrap();
        let to = window().start as u32;
        for (what, c) in [
            (
                "a tile beside the session's",
                dm::mcast_corners(RECT.0, (RECT.1 .0 + 1, RECT.1 .1)),
            ),
            ("a harvested column", dm::mcast_corners((14, 4), (16, 5))),
        ] {
            let r = nc.run_list(d, &w, &[[op::WRITE_L1_MCAST, c, 0, to, SRC, 64, 0, 0]]);
            assert!(
                matches!(r, Err(DmError::Invalid(code)) if code == dm::error::RANGE),
                "the host took {what}: {r:?}"
            );
        }
        nc.stop(d, &w).unwrap();
    });
}

#[test]
fn the_tile_refuses_a_multicast_it_cannot_count_or_should_not_send() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let to = window().start as u32;
        let gate = (GATE_TILE.0, GATE_TILE.1);
        let mc = |s: (u8, u8), e: (u8, u8)| {
            [
                op::WRITE_L1_MCAST,
                dm::mcast_corners(s, e),
                0,
                to,
                SRC,
                64,
                0,
                0,
            ]
        };
        let mut undefined = mc(RECT.0, RECT.1);
        undefined[6] = 2;
        let bad: [(&str, [u32; 8], u32, Niu); 5] = [
            (
                "a bit word 6 does not define",
                undefined,
                dm::error::LENGTH,
                Niu::Noc1,
            ),
            (
                "a rectangle holding itself",
                mc(gate, RECT.1),
                dm::error::RANGE,
                Niu::Noc1,
            ),
            (
                "a reversed rectangle",
                mc(RECT.1, RECT.0),
                dm::error::RANGE,
                Niu::Noc1,
            ),
            (
                "across the non-memory column",
                mc((7, 4), (10, 4)),
                dm::error::RANGE,
                Niu::Noc1,
            ),
            (
                "through NoC #0",
                mc(RECT.0, RECT.1),
                dm::error::DIRECTION,
                Niu::Noc0,
            ),
        ];
        for (what, entry, code, niu) in bad {
            let mut m = start(d, &w, Mover::NC);
            m.set_write_noc(d, &w, niu).unwrap();
            poison(d, &w, &rect_tiles());
            let n = m.enqueue_raw(d, &w, &[entry]).unwrap();
            let got = m.wait_for(d, &w, n);
            assert!(
                matches!(got, Err(DmError::Queued { code: c, .. }) if c == code),
                "the tile took {what}: {got:?}"
            );
            for t in rect_tiles() {
                assert!(
                    read_back(d, &w, t).iter().all(|&b| b == POISON),
                    "{what}: {t:?} was written"
                );
            }
            m.stop(d, &w).unwrap();
        }
    });
}

/// Where B's reads land in the gate tile's own L1, and how much of it is checked.
const DST: u32 = 0x4_0000;
const DST_LEN: usize = 96 << 10;

/// The tiles B reads from: the two peers and the gate tile itself (a loopback).
fn sources() -> [(u8, u8); 3] {
    [PEERS[0], PEERS[1], (GATE_TILE.0, GATE_TILE.1)]
}

fn read_dst(d: &mut Dev<'_>, w: &Window) -> Vec<u8> {
    let t = tile(d, GATE_TILE.0, GATE_TILE.1);
    let mut back = vec![0u8; DST_LEN];
    d.l1_read(w, t, DST as u64, &mut back).unwrap();
    back
}

fn poison_dst(d: &mut Dev<'_>, w: &Window) {
    let t = tile(d, GATE_TILE.0, GATE_TILE.1);
    d.l1_write(w, t, DST as u64, &vec![POISON; DST_LEN])
        .unwrap();
}

#[test]
fn a_read_pulls_a_peers_window() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let mut b = start(d, &w, Mover::B);
        b.set_l1_writes(L1Writes {
            tiles: sources().to_vec(),
            window: window(),
        })
        .unwrap();
        let span: u32 = LENS.iter().map(|l| l.next_multiple_of(64)).sum();
        let mut read = 0;
        for skew in [0u32, 16, 48] {
            // Each source's window holds its own pattern.
            let mut held = Vec::new();
            for (k, &(x, y)) in sources().iter().enumerate() {
                let p = pattern(WINDOW_LEN as usize, 0x2182 + k as u32 * 7 + skew);
                let t = tile(d, x, y);
                d.l1_write(&w, t, window().start, &p).unwrap();
                held.push(p);
            }
            poison_dst(d, &w);
            let mut want = vec![POISON; DST_LEN];
            let mut list = Vec::new();
            for (k, &(x, y)) in sources().iter().enumerate() {
                let (mut from, mut to) =
                    (window().start as u32 + skew, DST + k as u32 * span + skew);
                for len in LENS {
                    list.push([op::READ_L1, x as u32, y as u32, from, to, len, 0, 0]);
                    let s = (from as u64 - window().start) as usize;
                    let at = (to - DST) as usize;
                    want[at..at + len as usize].copy_from_slice(&held[k][s..s + len as usize]);
                    from += len.next_multiple_of(64);
                    to += len.next_multiple_of(64);
                }
            }
            assert!(DST as usize + DST_LEN <= window().start as usize - GUARD as usize);
            read += list.len();
            b.run_list(d, &w, &list).unwrap();
            let got = read_dst(d, &w);
            assert!(
                first_difference(&got, &want).is_none(),
                "skew {skew}: the gate tile's reads differ at byte {:?} past {DST:#x}",
                first_difference(&got, &want)
            );
        }
        assert_eq!(read, 3 * sources().len() * LENS.len());
        b.stop(d, &w).unwrap();
    });
}

#[test]
fn the_host_refuses_a_read_its_session_does_not_own() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let mut b = start(d, &w, Mover::B);
        let from = window().start as u32;
        let e = |x: u8, y: u8, from: u32| [op::READ_L1, x as u32, y as u32, from, DST, 4160, 0, 0];
        let refused = |r: Result<(), DmError>, what: &str| match r {
            Err(DmError::Invalid(code)) => code,
            other => panic!("the host took {what}: {other:?}"),
        };
        let r = b.run_list(d, &w, &[e(PEERS[0].0, PEERS[0].1, from)]);
        assert_eq!(
            refused(r, "a read before any tile was named"),
            dm::error::RANGE
        );
        let end = window().end - 8192;
        b.set_l1_writes(L1Writes {
            tiles: PEERS.to_vec(),
            window: window().start..end,
        })
        .unwrap();
        let cases = [
            (
                "a tile not the session's",
                e(BYSTANDER.0, BYSTANDER.1, from),
            ),
            ("a harvested column's tile", e(16, GATE_TILE.1, from)),
            ("below the window", e(PEERS[0].0, PEERS[0].1, from - 64)),
            (
                "past the window",
                e(PEERS[0].0, PEERS[0].1, end as u32 - 4096),
            ),
        ];
        for (what, entry) in cases {
            let r = b.run_list(d, &w, &[entry]);
            assert_eq!(refused(r, what), dm::error::RANGE, "{what}");
        }
        b.run_list(d, &w, &[e(PEERS[1].0, PEERS[1].1, from)])
            .unwrap();
        b.stop(d, &w).unwrap();
        // RISCV NC reads no L1 of another tile.
        let mut nc = start(d, &w, Mover::NC);
        let r = nc.run_list(d, &w, &[e(PEERS[0].0, PEERS[0].1, from)]);
        assert_eq!(refused(r, "READ_L1 on NC"), dm::error::DIRECTION);
        nc.stop(d, &w).unwrap();
    });
}

#[test]
fn the_tile_refuses_a_malformed_read_by_itself() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let (x, y) = (PEERS[0].0 as u32, PEERS[0].1 as u32);
        let from = window().start as u32;
        let bad: [(&str, [u32; 8], u32, Mover); 6] = [
            (
                "the non-memory column",
                [op::READ_L1, 8, y, from, DST, 64, 0, 0],
                dm::error::RANGE,
                Mover::B,
            ),
            (
                "a length of none",
                [op::READ_L1, x, y, from, DST, 0, 0, 0],
                dm::error::LENGTH,
                Mover::B,
            ),
            (
                "the remote side below DATA",
                [op::READ_L1, x, y, 0x1_0000, DST, 64, 0, 0],
                dm::error::ALIGNMENT,
                Mover::B,
            ),
            (
                "the local side below DATA",
                [op::READ_L1, x, y, from, 0x1_0000, 64, 0, 0],
                dm::error::ALIGNMENT,
                Mover::B,
            ),
            (
                "addresses not congruent mod 16",
                [op::READ_L1, x, y, from + 8, DST, 64, 0, 0],
                dm::error::ALIGNMENT,
                Mover::B,
            ),
            (
                "the entry on NC",
                [op::READ_L1, x, y, from, DST, 64, 0, 0],
                dm::error::DIRECTION,
                Mover::NC,
            ),
        ];
        for (what, entry, code, mover) in bad {
            let mut m = start(d, &w, mover);
            poison_dst(d, &w);
            let n = m.enqueue_raw(d, &w, &[entry]).unwrap();
            let got = m.wait_for(d, &w, n);
            assert!(
                matches!(got, Err(DmError::Queued { code: c, .. }) if c == code),
                "the tile took {what}: {got:?}"
            );
            assert!(
                read_dst(d, &w).iter().all(|&b| b == POISON),
                "{what}: the gate tile's L1 was written"
            );
            m.stop(d, &w).unwrap();
        }
    });
}

#[test]
fn a_run_of_multicasts_lands_before_the_next_entry() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let mut nc = start(d, &w, Mover::NC);
        let mut ours = rect_tiles();
        ours.extend(BESIDE);
        nc.set_l1_writes(L1Writes {
            tiles: ours,
            window: window(),
        })
        .unwrap();
        nc.set_write_noc(d, &w, Niu::Noc1).unwrap();
        let src = pattern(64 << 10, 0x2183);
        d.l1_write(&w, nc.tile(), SRC as u64, &src).unwrap();
        let corners = dm::mcast_corners(RECT.0, RECT.1);
        let base = window().start - GUARD;
        // Which of the five multicasts are marked, and whether unicasts frame them.
        let shapes: [(&str, [bool; 5], bool); 3] = [
            ("one run between unicasts", [true; 5], true),
            (
                "two runs split by a lone multicast",
                [true, true, false, true, true],
                true,
            ),
            ("a run ending the list", [true; 5], false),
        ];
        for (what, marked, framed) in shapes {
            for skew in [0u32, 48] {
                let mut all = rect_tiles();
                all.extend(BESIDE);
                poison(d, &w, &all);
                let mut list = Vec::new();
                let mut wants_beside = [None, None];
                if framed {
                    let (l, mut want) = writes_to(BESIDE[0], &src, skew);
                    list.push(l[0]);
                    want[(window().start - base) as usize + skew as usize + 64..].fill(POISON);
                    wants_beside[0] = Some(want);
                }
                let mut want_rect = vec![POISON; (WINDOW_LEN + GUARD) as usize];
                let (mut from, mut to) = (SRC + skew, window().start as u32 + skew);
                for (len, run) in LENS.into_iter().zip(marked) {
                    let w6 = if run { dm::MCAST_RUN_ACKS } else { 0 };
                    list.push([op::WRITE_L1_MCAST, corners, 0, to, from, len, w6, 0]);
                    let at = (to as u64 - base) as usize;
                    let s = (from - SRC) as usize;
                    want_rect[at..at + len as usize].copy_from_slice(&src[s..s + len as usize]);
                    from += len.next_multiple_of(64);
                    to += len.next_multiple_of(64);
                }
                if framed {
                    let (l, mut want) = writes_to(BESIDE[1], &src, skew);
                    list.push(l[0]);
                    want[(window().start - base) as usize + skew as usize + 64..].fill(POISON);
                    wants_beside[1] = Some(want);
                }
                nc.run_list(d, &w, &list).unwrap();
                for t in rect_tiles() {
                    let got = read_back(d, &w, t);
                    assert!(
                        first_difference(&got, &want_rect).is_none(),
                        "{what}, skew {skew}: {t:?} differs at byte {:?}",
                        first_difference(&got, &want_rect)
                    );
                }
                for (t, want) in BESIDE.iter().zip(&wants_beside) {
                    let got = read_back(d, &w, *t);
                    match want {
                        Some(want) => assert!(
                            first_difference(&got, want).is_none(),
                            "{what}, skew {skew}: beside {t:?} differs at byte {:?}",
                            first_difference(&got, want)
                        ),
                        None => assert!(
                            got.iter().all(|&b| b == POISON),
                            "{what}: beside {t:?} was written"
                        ),
                    }
                }
            }
        }
        nc.stop(d, &w).unwrap();
    });
}
