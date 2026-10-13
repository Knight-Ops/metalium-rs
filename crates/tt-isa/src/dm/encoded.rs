//! Host-encoded GDDR reads (`op::READ_ENCODED`).
//!
//! A plain `READ` entry names a channel, port, offset, L1 address and length; the
//! mover decodes it, checks it whole as a `DramMove` and encodes the NIU's request
//! words, which costs ~190 of the ~360 cycles an entry takes
//! (`docs/learnings/performance-log.md`, "Read-loop probe"). The host knows all of
//! that already, so it sends the request itself, in the same 32 bytes:
//!
//! ```text
//! [READ_ENCODED, TARG, TARG_HI, RET, RET_HI, CHECK, 0, LEN]
//! ```
//!
//! `TAG` and `CTRL` of a B read are constants the card supplies ([`TAG`], [`CTRL`]),
//! so they are not carried; the freed word is [`check`], an integrity word over the rest.
//!
//! **Who checks what.** [`validate`] is the whole check, on the host: the entry must
//! equal, word for word, what [`encode`] makes (from `Descriptor::decode` and
//! `DramMove`, the code the mover itself ran) for some valid plain `READ`. The card
//! verifies only [`check`], a few ALU operations: it refuses an entry damaged on its way
//! to L1 or overwritten there, and an entry that did not come from [`encode`] (a hand-made
//! one has the right check only by deliberately computing it). What a checksum cannot
//! catch is an entry that is consistent and wrong, which only a bug in [`encode`] makes;
//! [`guard`] is the independent card-side check for that (constant-time compares and a
//! usable-endpoint bitmap staged at [`ENDPOINTS`]), built only with the `encoded-guard`
//! feature, ~50 cycles an entry.
//!
//! One request per entry: a read longer than `MAX_REQUEST_BYTES` stays a plain entry
//! (its decode is amortised over its requests anyway).

use super::{error, op, Descriptor, MAILBOX_BASE, TXN_B};
use crate::dram::{Dram, CHANNELS, PORTS};
use crate::noc::niu::{
    DramMove, Endpoint, Niu, TxnId, CMD_RD, MAX_REQUEST_BYTES, MMIO_START, RESP_MARKED, STATIC_VC_1,
};

/// The bitmap of valid read targets for [`guard`]: bit `hi` of 4,096 (`hi` is an
/// endpoint's `TARG_ADDR_HI`, `x | y << 6`), set for every usable channel's NoC #0
/// endpoints. In B's mailbox page; the host writes it when it starts the mover.
pub const ENDPOINTS: u64 = MAILBOX_BASE + 0x200;
pub const ENDPOINT_WORDS: usize = 128;
const _: () = assert!(ENDPOINTS + ENDPOINT_WORDS as u64 * 4 <= MAILBOX_BASE + 0x1000);
const _: () = assert!(ENDPOINTS >= MAILBOX_BASE + 0x100);

/// `PACKET_TAG` of B's reads: its transaction ID in bits 10..14.
pub const TAG: u32 = (TXN_B as u32) << 10;
/// `NOC_CTRL` of a read request, as `DramMove` makes it.
pub const CTRL: u32 = CMD_RD | RESP_MARKED | STATIC_VC_1;

/// Word index of [`check`] in an encoded entry.
pub const CHECK_WORD: usize = 5;
/// Added into [`check`] so an entry of zeros is not its own checksum.
const CHECK_SEED: u32 = 0x9E37_79B9;

/// The integrity word over an entry's other words (all but the op and the check itself):
/// each multiplied by its own odd constant and XORed, so a change to any one word changes
/// it (an odd multiplier is a bijection of `u32`), and so, in practice, does swapping two
/// words -- the check word with one of them included, which is why no word is left unmixed
/// (found by `the_checksum_catches_damage`). Multiplies, not rotations: a rotation is three
/// instructions on these cores and a multiply one.
#[inline(always)]
pub const fn check(e: &[u32; 8]) -> u32 {
    e[1].wrapping_mul(0x9E37_79B1)
        ^ e[2].wrapping_mul(0x85EB_CA77)
        ^ e[3].wrapping_mul(0xC2B2_AE3D)
        ^ e[4].wrapping_mul(0x27D4_EB2F)
        ^ e[6].wrapping_mul(0x1656_67B1)
        ^ e[7].wrapping_mul(0xD3A2_646D)
        ^ CHECK_SEED
}

/// `TARG_ADDR_HI` / `RET_ADDR_HI` for the tile at `(x, y)`.
pub const fn endpoint_hi(x: u8, y: u8) -> u32 {
    Endpoint { x, y, addr: 0 }.hi()
}

/// The endpoint bitmap for the chip's `usable` channels.
pub fn endpoint_bitmap(usable: u32) -> [u32; ENDPOINT_WORDS] {
    let mut bits = [0u32; ENDPOINT_WORDS];
    let dram = Dram::from_usable_mask(usable as u8);
    for ch in dram.channels() {
        for port in 0..PORTS {
            if let Some(at) = ch.endpoint(Niu::Noc0, port) {
                let hi = endpoint_hi(at.x(), at.y()) as usize;
                bits[hi / 32] |= 1 << (hi % 32);
            }
        }
    }
    bits
}

/// A plain `READ` entry as an encoded one, for the tile at `me`. `Ok(None)` if
/// `entry` is not a `READ` of at most one request (it stays as it is); `Err` if it
/// is a `READ` the mover would refuse.
pub fn encode(usable: u32, me: (u8, u8), entry: [u32; 8]) -> Result<Option<[u32; 8]>, u32> {
    if entry[0] != op::READ {
        return Ok(None);
    }
    let d = Descriptor::decode(
        usable,
        op::READ,
        entry[1],
        entry[2],
        entry[3],
        entry[4],
        entry[5],
    )?;
    if d.range.len() > MAX_REQUEST_BYTES as u64 {
        return Ok(None);
    }
    let port = d.range.channel().port_for(Niu::Noc0, d.port);
    let txn = TxnId::new(TXN_B).expect("B's transaction ID is below 16");
    let mv = DramMove::new(d.range, port, d.l1, false, me, txn, Niu::Noc0)
        .map_err(|_| error::ALIGNMENT)?;
    let r = mv
        .words()
        .next()
        .expect("a move of at least one byte has a request");
    // What the card supplies instead of carrying.
    if r.tag != TAG || r.ctrl != CTRL {
        return Err(error::OP);
    }
    let mut e = [
        op::READ_ENCODED,
        r.targ,
        r.targ_hi,
        r.ret,
        r.ret_hi,
        0,
        0,
        r.len,
    ];
    e[CHECK_WORD] = check(&e);
    Ok(Some(e))
}

/// The host's full check of an encoded entry: it equals what [`encode`] makes for
/// some valid plain `READ` on this chip, for the tile at `me`.
pub fn validate(usable: u32, me: (u8, u8), e: [u32; 8]) -> Result<(), u32> {
    if e[0] != op::READ_ENCODED {
        return Err(error::OP);
    }
    // Which (channel, port) the target coordinate names; the offset, L1 address and
    // length are the entry's own words.
    for channel in 0..CHANNELS as u32 {
        for port in 0..PORTS as u32 {
            let plain = [op::READ, channel, port, e[1], e[3], e[7], 0, 0];
            if encode(usable, me, plain) == Ok(Some(e)) {
                return Ok(());
            }
        }
    }
    Err(error::ENCODED)
}

/// The card's integrity check of an encoded entry.
#[inline(always)]
pub const fn verify(e: &[u32; 8]) -> bool {
    e[CHECK_WORD] == check(e)
}

/// The optional semantic check, for the `encoded-guard` build: constant time, no loops,
/// branch-free. `me_hi` is [`endpoint_hi`] of this tile; `endpoint_ok` answers whether a
/// `TARG_ADDR_HI` is in the [`ENDPOINTS`] bitmap, and is always called, with the word masked
/// into the bitmap's range (the range test refuses a wild coordinate).
#[inline(always)]
pub fn guard(e: &[u32; 8], me_hi: u32, endpoint_ok: impl FnOnce(u32) -> bool) -> bool {
    let (targ, hi, ret, ret_hi, len) = (e[1], e[2], e[3], e[4], e[7]);
    // `len - 1 < MAX` is `1 <= len <= MAX`; `MMIO_START - len` wraps for a `len` the first
    // test has already refused.
    let room = MMIO_START.wrapping_sub(len);
    (len.wrapping_sub(1) < MAX_REQUEST_BYTES)
        & (targ <= room)
        & (ret <= room)
        & ((targ ^ ret) & (crate::dram::ALIGN as u32 - 1) == 0)
        & (ret_hi == me_hi)
        & (e[6] == 0)
        & (hi < ENDPOINT_WORDS as u32 * 32)
        & endpoint_ok(hi & (ENDPOINT_WORDS as u32 * 32 - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ME: (u8, u8) = (1, 2);

    fn plain(channel: u32, port: u32, offset: u32, l1: u32, len: u32) -> [u32; 8] {
        [op::READ, channel, port, offset, l1, len, 0, 0]
    }

    fn ok_bit(bits: &[u32; ENDPOINT_WORDS]) -> impl Fn(u32) -> bool + '_ {
        move |hi| bits[(hi / 32) as usize] >> (hi % 32) & 1 != 0
    }

    fn good() -> [u32; 8] {
        encode(0xFF, ME, plain(1, 0, 0x1000, 0x2_1000, 4096))
            .unwrap()
            .unwrap()
    }

    /// `e` with word `k` set to `v` and the check recomputed: what a forger who
    /// knows the checksum, or a bug in `encode`, would make.
    fn forged(mut e: [u32; 8], k: usize, v: u32) -> [u32; 8] {
        e[k] = v;
        e[CHECK_WORD] = check(&e);
        e
    }

    #[test]
    fn an_encoded_read_is_the_typed_encoders_request() {
        let e = good();
        assert_eq!(e[0], op::READ_ENCODED);
        assert_eq!((e[6], e[7]), (0, 4096));
        assert_eq!(e[4], endpoint_hi(ME.0, ME.1));
        assert!(verify(&e));
        assert_eq!(validate(0xFF, ME, e), Ok(()));
        // What the card supplies in place of the two words the entry does not carry is
        // exactly what `DramMove` makes.
        let t = TxnId::new(TXN_B).unwrap();
        let d = Descriptor::decode(0xFF, op::READ, 1, 0, 0x1000, 0x2_1000, 4096).unwrap();
        let port = d.range.channel().port_for(Niu::Noc0, 0);
        let r = DramMove::new(d.range, port, d.l1, false, ME, t, Niu::Noc0)
            .unwrap()
            .words()
            .next()
            .unwrap();
        assert_eq!((r.tag, r.ctrl), (TAG, CTRL));
        assert_eq!(
            [r.targ, r.targ_hi, r.ret, r.ret_hi],
            [e[1], e[2], e[3], e[4]]
        );
    }

    #[test]
    fn what_is_not_a_single_request_read_stays_plain() {
        assert_eq!(encode(0xFF, ME, [op::WAIT, 0, 0, 0, 0, 0, 0, 0]), Ok(None));
        assert_eq!(
            encode(0xFF, ME, plain(0, 0, 0, 0x2_0000, MAX_REQUEST_BYTES + 64)),
            Ok(None)
        );
        // ... and what the mover would refuse is refused, with its code.
        assert_eq!(
            encode(0xFF, ME, plain(9, 0, 0, 0x2_0000, 64)),
            Err(error::RANGE)
        );
        assert_eq!(
            encode(0xFF, ME, plain(0, 0, 0x40, 0x2_0020, 64)),
            Err(error::ALIGNMENT)
        );
        assert_eq!(
            encode(0xFF, ME, plain(0, 0, 0, 0x2_0000, 0)),
            Err(error::LENGTH)
        );
        // A channel the chip does not report usable.
        assert_eq!(
            encode(0xFE, ME, plain(0, 0, 0, 0x2_0000, 64)),
            Err(error::RANGE)
        );
    }

    /// The checksum catches what it is for: any one word damaged in any one bit, and
    /// any two words swapped.
    #[test]
    fn the_checksum_catches_damage() {
        let e = good();
        for k in [1, 2, 3, 4, 5, 6, 7] {
            for bit in 0..32 {
                let mut bad = e;
                bad[k] ^= 1 << bit;
                assert!(!verify(&bad), "word {k} bit {bit} went unnoticed");
            }
        }
        for i in 1..8 {
            for j in i + 1..8 {
                let mut bad = e;
                bad.swap(i, j);
                if bad != e {
                    assert!(!verify(&bad), "words {i} and {j} swapped went unnoticed");
                }
            }
        }
        // An entry of zeros, and one that is a plain `READ` with the op word changed.
        assert!(!verify(&[op::READ_ENCODED, 0, 0, 0, 0, 0, 0, 0]));
        assert!(!verify(&[
            op::READ_ENCODED,
            1,
            0x2000_0000,
            0x2_0000,
            5,
            0,
            0,
            4096
        ]));
    }

    /// The same over many entries, not the one: single-bit damage and swaps of every pair of
    /// words (the check word included) in 20,000 pseudo-random well-formed-looking entries.
    #[test]
    fn the_checksum_catches_damage_in_any_entry() {
        let mut s = 0x1234_5678u32;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for _ in 0..20_000 {
            let mut e = [
                op::READ_ENCODED,
                next(),
                next() & 0xFFF,
                next(),
                next() & 0xFFF,
                0,
                0,
                next() & 0x3FFF,
            ];
            e[CHECK_WORD] = check(&e);
            assert!(verify(&e));
            for k in [1, 2, 3, 4, 5, 6, 7] {
                let bit = next() % 32;
                let mut bad = e;
                bad[k] ^= 1 << bit;
                assert!(!verify(&bad), "{e:x?}: word {k} bit {bit}");
            }
            for i in 1..8 {
                for j in i + 1..8 {
                    let mut bad = e;
                    bad.swap(i, j);
                    if bad != e {
                        assert!(!verify(&bad), "{e:x?}: words {i} and {j} swapped");
                    }
                }
            }
        }
    }

    /// A damaged entry is also refused by the host's `validate`, and so is every entry
    /// that is consistent and wrong -- the case a checksum cannot see.
    #[test]
    fn validate_refuses_what_the_checksum_cannot() {
        let e = good();
        let t = [
            ("zero length", forged(e, 7, 0)),
            ("two requests", forged(e, 7, MAX_REQUEST_BYTES + 64)),
            ("target at MMIO", forged(e, 1, MMIO_START - 64)),
            ("destination at MMIO", forged(e, 3, MMIO_START - 64)),
            ("destination not congruent", forged(e, 3, e[3] + 16)),
            ("another tile's return", forged(e, 4, endpoint_hi(5, 6))),
            ("a tile that is not DRAM", forged(e, 2, endpoint_hi(0, 0))),
            ("a reserved word set", forged(e, 6, 1)),
        ];
        for (what, bad) in t {
            assert!(
                verify(&bad),
                "{what}: the forged entry should carry a good check"
            );
            assert!(
                validate(0xFF, ME, bad).is_err(),
                "validate let through {what}: {bad:x?}"
            );
        }
        // A channel the chip does not report usable.
        assert_eq!(validate(0xFD, ME, good()), Err(error::ENCODED));
        // And a plain entry is not an encoded one.
        assert_eq!(
            validate(0xFF, ME, plain(1, 0, 0, 0x2_0000, 64)),
            Err(error::OP)
        );
    }

    #[test]
    fn the_guard_accepts_what_validate_accepts() {
        let bits = endpoint_bitmap(0xFF);
        for (c, p, off, l1, len) in [
            (0, 0, 0, 0x2_0000, 64),
            (7, 2, 0x10_0000, 0x3_0000, 4096),
            (4, 1, 0x40, 0xF_F000 + 0x40, 2048),
            (2, 0, 0, 0x2_0000, MAX_REQUEST_BYTES),
        ] {
            let e = encode(0xFF, ME, plain(c, p, off, l1, len))
                .unwrap()
                .unwrap();
            assert_eq!(validate(0xFF, ME, e), Ok(()), "{e:x?}");
            assert!(verify(&e));
            assert!(guard(&e, endpoint_hi(ME.0, ME.1), ok_bit(&bits)), "{e:x?}");
        }
    }

    /// The guard's own decision on each word, with the checksum made good, as an
    /// encoder bug would leave it.
    #[test]
    fn the_guard_refuses_what_is_consistent_and_wrong() {
        let bits = endpoint_bitmap(0xFF);
        let e = good();
        let me_hi = endpoint_hi(ME.0, ME.1);
        assert!(guard(&e, me_hi, ok_bit(&bits)));
        for (what, bad) in [
            ("zero length", forged(e, 7, 0)),
            ("two requests", forged(e, 7, MAX_REQUEST_BYTES + 1)),
            ("length wraps", forged(e, 7, u32::MAX)),
            ("target at MMIO", forged(e, 1, MMIO_START - 64)),
            ("destination at MMIO", forged(e, 3, MMIO_START - 64)),
            ("destination not congruent", forged(e, 3, e[3] + 16)),
            ("not this tile", forged(e, 4, endpoint_hi(5, 6))),
            ("a harvested tile", forged(e, 2, endpoint_hi(0, 0))),
            ("an impossible coordinate", forged(e, 2, 0xFFFF_FFFF)),
            ("a reserved word set", forged(e, 6, 1)),
        ] {
            assert!(
                !guard(&bad, me_hi, ok_bit(&bits)),
                "the guard let through {what}: {bad:x?}"
            );
        }
    }

    #[test]
    fn the_bitmap_holds_exactly_the_usable_channels_endpoints() {
        for usable in [0xFFu32, 0x0F, 0x81] {
            let bits = endpoint_bitmap(usable);
            let dram = Dram::from_usable_mask(usable as u8);
            let mut want = 0;
            for ch in dram.channels() {
                for p in 0..PORTS {
                    if let Some(at) = ch.endpoint(Niu::Noc0, p) {
                        let hi = endpoint_hi(at.x(), at.y());
                        assert!(ok_bit(&bits)(hi));
                        want += 1;
                    }
                }
            }
            let set: u32 = bits.iter().map(|w| w.count_ones()).sum();
            assert!(set <= want, "{usable:#x}: {set} bits for {want} endpoints");
            assert!(set > 0);
        }
    }
}
