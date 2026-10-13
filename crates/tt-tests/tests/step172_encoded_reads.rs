//! Host-encoded reads (`op::READ_ENCODED`, `tt_isa::dm::encoded`) and the two ways the
//! B mover is built (firmware feature `fast-reads`; `cargo xtask silicon --feature
//! fast-reads`, or `--features tt-tests/fast-reads` on ttsim).
//!
//! **Safe mode (the default build).** The card decodes and checks every read itself.
//! The image says it has no encoded reads (`dm::CAPS`); the host refuses to encode for it
//! (`DmError::FastReadsNotBuilt`) and refuses an encoded entry as an unknown op; and an
//! encoded entry sent past the host is refused by the card, reaches no NIU and moves
//! nothing.
//!
//! **Fast mode.** The host sends the NIU's request words, validated in full
//! (`encoded::validate`); the card verifies a checksum and issues. This holds that:
//! - an encoded read lands exactly the bytes the plain read lands, on every channel and
//!   port, at every request size the encoding takes, in lists that also hold entries the
//!   encoding leaves alone;
//! - the list the card runs really holds `READ_ENCODED` entries when the host encodes and
//!   plain ones when it does not, so the equality above is not one path against itself;
//! - the host refuses a damaged or wrong encoded entry before it costs any PCIe, and the
//!   card refuses a damaged one that reaches it anyway (`error::ENCODED`), moving nothing
//!   and leaving the mover able to run the next list. With `encoded-guard` it refuses
//!   consistent-but-wrong ones (a checksum cannot) as well.
//!
//! On silicon the entries sent past the host are only ones whose failure to be refused
//! would still be a valid read into this test's own L1 and DRAM: a bad coordinate is the
//! hang `AGENTS.md` warns of, so the refusal of those is held on ttsim and, for the guard's
//! decision, by `tt_isa::dm::encoded`'s unit tests (the function the card runs).
//!
//! In both modes NC refuses an encoded read, on the host and on the card.

use tt_device::tlb::WindowKind;
#[cfg(feature = "fast-reads")]
use tt_device::Transport;
use tt_device::Window;
use tt_isa::dm::{self, encoded, op, Mover};
#[cfg(feature = "fast-reads")]
use tt_isa::dram::PORTS;
use tt_isa::noc::Noc0;
use tt_kernels::dm::{DataMover, DmError};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile, Dev};

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

const L1_AT: u32 = 0x2_0000;
const DRAM_AT: u64 = 300 << 20;
const POISON: u8 = 0xA5;
/// Reads per list, each into its own slot.
const READS: u32 = 8;

fn start(d: &mut Dev<'_>, w: &Window, mover: Mover) -> DataMover<Noc0> {
    let dram = d.dram_grid(w).unwrap();
    let t = tile(d, GATE_TILE.0, GATE_TILE.1);
    let image = match mover.core {
        tt_isa::tensix::Core::NC => tt_firmware_images::DM_NC.1,
        _ => tt_firmware_images::DM_B.1,
    };
    DataMover::start_on(d, w, t, &dram, mover, image).unwrap()
}

fn plain(channel: u32, port: u32, offset: u32, l1: u32, len: u32) -> [u32; 8] {
    [op::READ, channel, port, offset, l1, len, 0, 0]
}

/// Where `good`'s source data lies in channel 0.
const GOOD_AT: u64 = DRAM_AT + 0xC0_0000;

/// A good encoded read of 4 KiB from channel 0 into `L1_AT`, bound to `m`'s context (its key,
/// tile and chip): an entry made for one mover verifies on no other.
fn entry_for(m: &DataMover<Noc0>) -> [u32; 8] {
    encoded::encode(m.encode_context(), plain(0, 0, GOOD_AT as u32, L1_AT, 4096))
        .unwrap()
        .unwrap()
}

/// A good encoded read of 4 KiB from channel 0 into `L1_AT`, and its source data.
fn good(d: &mut Dev<'_>, w: &Window, w4: &Window, m: &DataMover<Noc0>) -> ([u32; 8], Vec<u8>) {
    let dram = d.dram_grid(w).unwrap();
    let ch = dram.channel(0).unwrap();
    let data = pattern(4096, 99);
    d.dram_write(w4, ch.range(GOOD_AT, 4096).unwrap(), &data)
        .unwrap();
    (entry_for(m), data)
}

#[test]
fn nc_refuses_an_encoded_read() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let b = start(d, &w, Mover::B);
        let (e, _) = good(d, &w, &w4, &b);
        b.stop(d, &w).unwrap();
        let mut m = start(d, &w, Mover::NC);
        assert!(!m.fast_reads(), "NC has no encoded reads in any build");
        let r = m.run_list(d, &w, &[e]);
        assert!(
            matches!(r, Err(DmError::Invalid(dm::error::DIRECTION))),
            "{r:?}"
        );
        m.stop(d, &w).unwrap();
    });
}

/// Safe mode: the default build.
#[cfg(not(feature = "fast-reads"))]
mod safe {
    use super::*;

    #[test]
    fn a_safe_image_says_so_and_the_host_will_not_encode_for_it() {
        in_device(|d| {
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let mut m = start(d, &w, Mover::B);
            const { assert!(!tt_firmware_images::FAST_READS) };
            assert!(!m.fast_reads());
            assert!(!m.encodes_reads(), "a safe image is never encoded for");
            let r = m.set_encode_reads(true);
            assert!(matches!(r, Err(DmError::FastReadsNotBuilt)), "{r:?}");
            m.set_encode_reads(false).unwrap();
            m.stop(d, &w).unwrap();
        });
    }

    #[test]
    fn plain_reads_are_sent_and_run_as_plain_reads() {
        in_device(|d| {
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
            let dram = d.dram_grid(&w).unwrap();
            let mut m = start(d, &w, Mover::B);
            let t = m.tile();
            let ch = dram.channel(2).unwrap();
            let base = DRAM_AT + 0x40_0000;
            let total = (READS * 4096) as usize;
            let data = pattern(total, 3);
            d.dram_write(&w4, ch.range(base, total as u64).unwrap(), &data)
                .unwrap();
            let list: Vec<[u32; 8]> = (0..READS)
                .map(|i| plain(2, 0, base as u32 + i * 4096, L1_AT + i * 4096, 4096))
                .collect();
            d.l1_write(&w, t, L1_AT as u64, &vec![POISON; total])
                .unwrap();
            m.run_list(d, &w, &list).unwrap();
            let mut back = vec![0u8; total];
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            assert!(back == data);
            for i in 0..READS as u64 {
                let first = d
                    .read32(&w, t, m.mover().list + i * dm::ENTRY_BYTES)
                    .unwrap();
                assert_eq!(first, op::READ, "entry {i}");
            }
            m.stop(d, &w).unwrap();
        });
    }

    /// The card verifies before running: an encoded entry that reaches it is an unknown
    /// op, refused with `error::OP` before any NIU write, and the mover survives.
    #[test]
    fn the_card_refuses_an_encoded_read_that_reaches_it() {
        in_device(|d| {
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
            let mut m = start(d, &w, Mover::B);
            let t = m.tile();
            let (e, data) = good(d, &w, &w4, &m);
            d.l1_write(&w, t, L1_AT as u64, &[POISON; 4096]).unwrap();
            // The host refuses it first, with the same code the card gives.
            let r = m.run_list(d, &w, &[e]);
            assert!(matches!(r, Err(DmError::Invalid(dm::error::OP))), "{r:?}");
            // `enqueue_checked` refuses it as well, on a safe image an unknown op.
            let r = m.enqueue_checked(d, &w, &[e]);
            assert!(matches!(r, Err(DmError::Invalid(dm::error::OP))), "{r:?}");
            // Sent past the host altogether (`enqueue_raw` trusts its caller), a valid entry
            // is still refused: the entry is good, so a card that ran it would be
            // a valid read into this test's L1, which the poison would show.
            let n = m.enqueue_raw(d, &w, &[e]).unwrap();
            let r = m.wait_for(d, &w, n);
            assert!(
                matches!(r, Err(DmError::Queued { code, .. }) if code == dm::error::OP),
                "the card answered {r:?}"
            );
            let mut back = [0u8; 4096];
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            assert!(
                back.iter().all(|&b| b == POISON),
                "the card ran an encoded read"
            );
            m.stop(d, &w).unwrap();
            // The control, and the mover's survival: a restarted mover runs the plain read.
            let mut m = start(d, &w, Mover::B);
            let plain_read = plain(0, 0, (DRAM_AT + 0xC0_0000) as u32, L1_AT, 4096);
            m.run_list(d, &w, &[plain_read]).unwrap();
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            assert!(back[..] == data[..]);
            m.stop(d, &w).unwrap();
        });
    }
}

/// Fast mode: the firmware built with `fast-reads`.
#[cfg(feature = "fast-reads")]
mod fast {
    use super::*;

    #[test]
    fn a_fast_image_says_so_and_is_encoded_for_by_default() {
        in_device(|d| {
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let mut m = start(d, &w, Mover::B);
            const { assert!(tt_firmware_images::FAST_READS) };
            assert!(m.fast_reads());
            assert!(m.encodes_reads());
            m.set_encode_reads(false).unwrap();
            assert!(!m.encodes_reads());
            m.set_encode_reads(true).unwrap();
            m.stop(d, &w).unwrap();
        });
    }

    #[test]
    fn encoded_reads_land_what_plain_reads_land() {
        in_device(|d| {
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
            let dram = d.dram_grid(&w).unwrap();
            let mut m = start(d, &w, Mover::B);
            let t = m.tile();
            let list_at = m.mover().list;
            for (k, ch) in dram.channels().enumerate() {
                for port in 0..PORTS {
                    for len in [64u32, 2048, 4096, 16384] {
                        let base = DRAM_AT + 0x40_0000 + port as u64 * 0x40_0000;
                        let total = (READS * len) as usize;
                        let data = pattern(total, (k * 7) as u32 + len + port as u32);
                        d.dram_write(&w4, ch.range(base, total as u64).unwrap(), &data)
                            .unwrap();
                        let list: Vec<[u32; 8]> = (0..READS)
                            .map(|i| {
                                plain(
                                    ch.index() as u32,
                                    port as u32,
                                    base as u32 + i * len,
                                    L1_AT + i * len,
                                    len,
                                )
                            })
                            .collect();
                        for encode in [false, true] {
                            m.set_encode_reads(encode).unwrap();
                            d.l1_write(&w, t, L1_AT as u64, &vec![POISON; total])
                                .unwrap();
                            m.run_list(d, &w, &list).unwrap();
                            let mut back = vec![0u8; total];
                            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
                            assert!(
                                back == data,
                                "channel {} port {port}, {len} B reads, encoded {encode}",
                                ch.index()
                            );
                            // The list the card ran: the path under test is the path taken.
                            for i in 0..READS as u64 {
                                let first = d.read32(&w, t, list_at + i * dm::ENTRY_BYTES).unwrap();
                                assert_eq!(
                                    first,
                                    if encode { op::READ_ENCODED } else { op::READ },
                                    "entry {i}, encoded {encode}"
                                );
                            }
                        }
                    }
                }
            }
            m.stop(d, &w).unwrap();
        });
    }

    #[test]
    fn entries_the_encoding_leaves_alone_still_run_in_order() {
        in_device(|d| {
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
            let dram = d.dram_grid(&w).unwrap();
            let mut m = start(d, &w, Mover::B);
            let t = m.tile();
            let list_at = m.mover().list;
            let ch = dram.channel(1).unwrap();
            let base = DRAM_AT + 0x80_0000;
            // A short read, a wait, a read of three requests (it stays plain), a short read.
            let long = 32 * 1024 + 64;
            let data = pattern(4096 + long as usize + 4096, 5);
            d.dram_write(&w4, ch.range(base, data.len() as u64).unwrap(), &data)
                .unwrap();
            let list = [
                plain(1, 0, base as u32, L1_AT, 4096),
                [op::WAIT, 0, 0, 0, 0, 0, 0, 0],
                plain(1, 0, base as u32 + 4096, L1_AT + 4096, long),
                plain(1, 0, base as u32 + 4096 + long, L1_AT + 4096 + long, 4096),
            ];
            d.l1_write(&w, t, L1_AT as u64, &vec![POISON; data.len()])
                .unwrap();
            m.run_list(d, &w, &list).unwrap();
            let mut back = vec![0u8; data.len()];
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            assert!(back == data);
            let ops: Vec<u32> = (0..4)
                .map(|i| d.read32(&w, t, list_at + i * dm::ENTRY_BYTES).unwrap())
                .collect();
            assert_eq!(
                ops,
                [op::READ_ENCODED, op::WAIT, op::READ, op::READ_ENCODED]
            );
            m.stop(d, &w).unwrap();
        });
    }

    /// `e` with word `k` set to `v`: damage, which the checksum no longer matches.
    fn damaged(e: [u32; 8], k: usize, v: u32) -> [u32; 8] {
        let mut e = e;
        e[k] = v;
        e
    }

    /// The same with the checksum made good again: a bug in the host's encoder, or a
    /// forger, which a checksum cannot see.
    fn forged(seed: u32, e: [u32; 8], k: usize, v: u32) -> [u32; 8] {
        let mut e = damaged(e, k, v);
        e[encoded::CHECK_WORD] = encoded::check_with(&e, seed);
        e
    }

    /// Each way an encoded read can be damaged, by what it does. The flag is whether
    /// sending it past the host is safe on silicon: a request the checksum failing to
    /// refuse would still make is a valid read into this test's own L1 and DRAM.
    fn damage(good: [u32; 8]) -> Vec<(&'static str, [u32; 8], bool)> {
        let mut swapped = good;
        swapped.swap(1, 3);
        vec![
            (
                "the target moved by 64",
                damaged(good, 1, good[1] + 64),
                true,
            ),
            (
                "the destination moved by 64",
                damaged(good, 3, good[3] + 64),
                true,
            ),
            (
                "a flipped bit in the checksum",
                damaged(good, 5, good[5] ^ 1),
                true,
            ),
            ("a reserved word set", damaged(good, 6, 1), true),
            ("half the length", damaged(good, 7, good[7] / 2), true),
            ("target and destination swapped", swapped, false),
            ("zero length", damaged(good, 7, 0), false),
            (
                "a tile that is not DRAM",
                damaged(good, 2, encoded::endpoint_hi(0, 0)),
                false,
            ),
            (
                "another tile's return",
                damaged(good, 4, encoded::endpoint_hi(5, 6)),
                false,
            ),
        ]
    }

    /// Consistent and wrong: what only `validate` (and, built with `encoded-guard`, the card)
    /// can refuse.
    fn wrong(seed: u32, good: [u32; 8]) -> Vec<(&'static str, [u32; 8])> {
        vec![
            ("zero length", forged(seed, good, 7, 0)),
            ("two requests", forged(seed, good, 7, 16384 + 64)),
            ("target at MMIO", forged(seed, good, 1, 0xFF00_0000 - 64)),
            (
                "destination at MMIO",
                forged(seed, good, 3, 0xFF00_0000 - 64),
            ),
            (
                "destination not congruent",
                forged(seed, good, 3, good[3] + 16),
            ),
            (
                "another tile's return",
                forged(seed, good, 4, encoded::endpoint_hi(5, 6)),
            ),
            (
                "a tile that is not DRAM",
                forged(seed, good, 2, encoded::endpoint_hi(0, 0)),
            ),
            ("a reserved word set", forged(seed, good, 6, 1)),
        ]
    }

    #[test]
    fn the_host_refuses_a_damaged_or_wrong_encoded_read_before_any_pcie() {
        in_device(|d| {
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
            let mut m = start(d, &w, Mover::B);
            let t = m.tile();
            let (e, data) = good(d, &w, &w4, &m);
            d.l1_write(&w, t, L1_AT as u64, &[POISON; 4096]).unwrap();
            let bad = damage(e)
                .into_iter()
                .map(|(what, e, _)| (what, e))
                .chain(wrong(m.encode_context().seed(op::READ_ENCODED), e));
            for (what, bad) in bad {
                let r = m.run_list(d, &w, &[bad]);
                assert!(
                    matches!(r, Err(DmError::Invalid(dm::error::ENCODED))),
                    "{what}: {r:?}"
                );
                // The queue's trusting entry point validates an encoded read too.
                let r = m.enqueue_checked(d, &w, &[bad]);
                assert!(
                    matches!(r, Err(DmError::Invalid(dm::error::ENCODED))),
                    "{what}: enqueue_checked answered {r:?}"
                );
            }
            let mut back = [0u8; 4096];
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            assert!(
                back.iter().all(|&b| b == POISON),
                "a refused entry moved data"
            );
            // The control: the unbroken entry passes the same check and lands.
            m.run_list(d, &w, &[e]).unwrap();
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            assert!(back[..] == data[..]);
            m.stop(d, &w).unwrap();
        });
    }

    /// Sends `entry` to the card as it is, on a fresh mover (a failed list stops the queue
    /// until the mover restarts), and returns what the card answered and whether any
    /// destination byte changed.
    ///
    /// `make` builds the entry from the fresh mover: its checksum is bound to that mover's key,
    /// so an entry made for another mover would fail the card's check whatever else it was.
    fn send(
        d: &mut Dev<'_>,
        w: &Window,
        make: impl FnOnce(&DataMover<Noc0>) -> [u32; 8],
    ) -> (Result<(), DmError>, bool) {
        let mut m = start(d, w, Mover::B);
        let entry = make(&m);
        let t = m.tile();
        d.l1_write(w, t, L1_AT as u64, &[POISON; 4096]).unwrap();
        // `enqueue_raw` trusts its caller entirely: the entry goes to the card as it is.
        let n = m.enqueue_raw(d, w, &[entry]).unwrap();
        let r = m.wait_for(d, w, n);
        let mut back = [0u8; 4096];
        d.l1_read(w, t, L1_AT as u64, &mut back).unwrap();
        m.stop(d, w).unwrap();
        (r, back.iter().any(|&b| b != POISON))
    }

    #[test]
    fn the_card_refuses_a_damaged_encoded_read_the_host_did_not_check() {
        in_device(|d| {
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
            let simulated = d.transport().is_simulated();
            let m = start(d, &w, Mover::B);
            let (e, data) = good(d, &w, &w4, &m);
            m.stop(d, &w).unwrap();
            let mut ran = 0;
            for (i, (what, _, safe_on_silicon)) in damage(e).into_iter().enumerate() {
                if !simulated && !safe_on_silicon {
                    continue;
                }
                let (r, moved) = send(d, &w, |m| damage(entry_for(m)).swap_remove(i).1);
                assert!(
                    matches!(r, Err(DmError::Queued { code, .. }) if code == dm::error::ENCODED),
                    "{what}: the card answered {r:?}"
                );
                assert!(!moved, "{what}: the card moved data");
                ran += 1;
            }
            // All of them on ttsim; on silicon only the ones that are safe to send.
            assert!(
                ran >= if simulated { 9 } else { 5 },
                "the checksum's decisions were not exercised ({ran})"
            );
            // The control, and the mover's survival: a restarted mover runs the good entry.
            let (r, moved) = send(d, &w, entry_for);
            r.unwrap();
            assert!(moved);
            let m = start(d, &w, Mover::B);
            let t = m.tile();
            let mut back = [0u8; 4096];
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            assert!(back[..] == data[..]);
            m.stop(d, &w).unwrap();
        });
    }

    /// The checksum is bound to the mover's key (`encoded::context_seed`), fresh at every start:
    /// an entry made for an earlier mover instance -- a list an earlier run left in L1 -- is refused
    /// by the next one's card and moves nothing, though it is a valid read in every other respect
    /// (so sending it is safe on silicon). The control: the same read made for the new instance runs.
    #[test]
    fn the_card_refuses_an_entry_made_for_an_earlier_mover() {
        in_device(|d| {
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
            let m = start(d, &w, Mover::B);
            let (stale, data) = good(d, &w, &w4, &m);
            let key = m.encode_context().key;
            m.stop(d, &w).unwrap();
            let (r, moved) = send(d, &w, |m| {
                assert_ne!(m.encode_context().key, key, "two starts shared a key");
                stale
            });
            assert!(
                matches!(r, Err(DmError::Queued { code, .. }) if code == dm::error::ENCODED),
                "the card answered {r:?}"
            );
            assert!(!moved, "the card ran an entry from an earlier mover");
            let (r, moved) = send(d, &w, entry_for);
            r.unwrap();
            assert!(moved);
            let m = start(d, &w, Mover::B);
            let mut back = [0u8; 4096];
            d.l1_read(&w, m.tile(), L1_AT as u64, &mut back).unwrap();
            assert!(back[..] == data[..]);
            m.stop(d, &w).unwrap();
        });
    }

    /// An entry that is consistent and wrong carries a good checksum, so the card's
    /// checksum cannot refuse it: only an `encoded-guard` build does, and that is held on
    /// ttsim alone (on silicon these are the NoC hang).
    #[cfg(feature = "encoded-guard")]
    #[test]
    fn the_guard_refuses_a_consistent_but_wrong_encoded_read() {
        in_device(|d| {
            if !d.transport().is_simulated() {
                return;
            }
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
            let m = start(d, &w, Mover::B);
            let (e, _) = good(d, &w, &w4, &m);
            m.stop(d, &w).unwrap();
            for (i, (what, _)) in wrong(0, e).into_iter().enumerate() {
                let (r, moved) = send(d, &w, |m| {
                    let seed = m.encode_context().seed(op::READ_ENCODED);
                    wrong(seed, entry_for(m)).swap_remove(i).1
                });
                assert!(
                    matches!(r, Err(DmError::Queued { code, .. }) if code == dm::error::ENCODED),
                    "{what}: the card answered {r:?}"
                );
                assert!(!moved, "{what}: the card moved data");
            }
        });
    }
}
