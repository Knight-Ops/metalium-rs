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
//! so they are not carried; the freed word is [`check_with`], an integrity word over the rest.
//!
//! **Who checks what.** [`validate`] is the whole check, on the host: the entry must
//! equal, word for word, what [`encode`] makes (from `Descriptor::decode` and
//! `DramMove`, the code the mover itself ran) for some valid plain `READ`, and must also
//! pass [`guard`], the independent semantic check, so a bug that made `encode` and
//! `validate` agree on a wrong request is still refused. The card verifies only the
//! checksum, a few ALU operations: it refuses an entry damaged on its way to L1 or
//! overwritten there, and an entry that did not come from [`encode`] (a hand-made one has
//! the right check only by deliberately computing it).
//!
//! **The checksum is bound to a context** ([`Context::seed`]): a random key the host wrote
//! when it started this mover, the tile, the usable-channel mask and the op. An entry
//! encoded for another mover instance (an earlier run's list left in L1), another tile or chip,
//! or another op fails the check, and a trace replay uses a different domain, so an encoded
//! entry inside a trace fails it. This is a binding against mix-ups, not a MAC: whoever can
//! write L1 can read the key.
//!
//! What a checksum cannot catch is an entry that is consistent and wrong, which only a bug in
//! [`encode`] makes; [`guard`] is the independent card-side check for that (constant-time
//! compares and a usable-endpoint bitmap staged at [`ENDPOINTS`]), built only with the
//! `encoded-guard` feature, ~40 cycles an entry.
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

// The target bound of `guard` (`MMIO_START`) is the channel's own extent, as `Descriptor::decode` checks it.
const _: () = assert!(crate::dram::CHANNEL_BYTES == MMIO_START as u64);

/// `PACKET_TAG` of B's reads: its transaction ID in bits 10..14.
pub const TAG: u32 = (TXN_B as u32) << 10;
/// `NOC_CTRL` of a read request, as `DramMove` makes it.
pub const CTRL: u32 = CMD_RD | RESP_MARKED | STATIC_VC_1;

/// Word index of the checksum in an encoded entry.
pub const CHECK_WORD: usize = 5;
/// Mixed into every [`context_seed`] so an all-zero context is not its own checksum.
const CHECK_SEED: u32 = 0x9E37_79B9;
/// What a trace replay (`CALL`) flips in the seed: no encoded entry a host sent as
/// a list entry verifies inside a trace, and the host never encodes one there.
pub const TRACE_DOMAIN: u32 = 0x5A5A_C3C3;

/// The seed of a checksum for `op` on the tile at `me` of a chip whose usable channels are `usable`,
/// for a mover the host keyed with `key`. A bijection of `key` for any other three, and of the
/// others for any `key`, so two contexts never share a seed.
pub const fn context_seed(key: u32, me: (u8, u8), usable: u32, op: u32) -> u32 {
    let ctx = me.0 as u32 | (me.1 as u32) << 8 | (usable & 0xFF) << 16 | (op & 0xFF) << 24;
    (key ^ ctx).wrapping_mul(0x9E37_79B1) ^ CHECK_SEED
}

/// The integrity word over an entry's other words (all but the op and the check itself),
/// started from `seed` ([`context_seed`]): each word multiplied by its own odd constant and
/// XORed, so a change to any one word changes it (an odd multiplier is a bijection of
/// `u32`), and so, in practice, does swapping two words -- the check word with one of them
/// included, which is why no word is left unmixed (found by `the_checksum_catches_damage`).
/// Multiplies, not rotations: a rotation is three instructions on these cores and a multiply one.
#[inline(always)]
pub const fn check_with(e: &[u32; 8], seed: u32) -> u32 {
    e[1].wrapping_mul(0x9E37_79B1)
        ^ e[2].wrapping_mul(0x85EB_CA77)
        ^ e[3].wrapping_mul(0xC2B2_AE3D)
        ^ e[4].wrapping_mul(0x27D4_EB2F)
        ^ e[6].wrapping_mul(0x1656_67B1)
        ^ e[7].wrapping_mul(0xD3A2_646D)
        ^ seed
}

/// What the host knows about the mover an entry is for: the chip's usable channels, the tile
/// the mover runs on, and the key written to its mailbox ([`super::ENCODE_KEY`]) when it started.
/// Holds the endpoint bitmap [`guard`] needs, built once.
#[derive(Clone)]
pub struct Context {
    pub usable: u32,
    pub me: (u8, u8),
    pub key: u32,
    bitmap: [u32; ENDPOINT_WORDS],
}

impl Context {
    pub fn new(usable: u32, me: (u8, u8), key: u32) -> Self {
        Context {
            usable,
            me,
            key,
            bitmap: endpoint_bitmap(usable),
        }
    }

    /// The seed for `op`'s checksums in this context.
    pub const fn seed(&self, op: u32) -> u32 {
        context_seed(self.key, self.me, self.usable, op)
    }

    /// The bitmap of usable read targets the card holds at [`ENDPOINTS`].
    pub fn bitmap(&self) -> &[u32; ENDPOINT_WORDS] {
        &self.bitmap
    }

    /// Whether `hi` (a `TARG_ADDR_HI`) is in the bitmap.
    fn endpoint_ok(&self, hi: u32) -> bool {
        self.bitmap[(hi / 32) as usize % ENDPOINT_WORDS] >> (hi % 32) & 1 != 0
    }
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
pub fn encode(ctx: &Context, entry: [u32; 8]) -> Result<Option<[u32; 8]>, u32> {
    let (usable, me) = (ctx.usable, ctx.me);
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
    e[CHECK_WORD] = check_with(&e, ctx.seed(op::READ_ENCODED));
    Ok(Some(e))
}

/// The host's full check of an encoded entry: it equals what [`encode`] makes for
/// some valid plain `READ` in this context, and it passes [`guard`] -- two checks that share
/// no code, so an encoder that agrees with this function about a wrong request is still caught.
pub fn validate(ctx: &Context, e: [u32; 8]) -> Result<(), u32> {
    if e[0] != op::READ_ENCODED {
        return Err(error::OP);
    }
    // Which (channel, port) the target coordinate names; the offset, L1 address and
    // length are the entry's own words.
    for channel in 0..CHANNELS as u32 {
        for port in 0..PORTS as u32 {
            let plain = [op::READ, channel, port, e[1], e[3], e[7], 0, 0];
            if encode(ctx, plain) == Ok(Some(e)) {
                return if guard(&e, endpoint_hi(ctx.me.0, ctx.me.1), |hi| {
                    ctx.endpoint_ok(hi)
                }) {
                    Ok(())
                } else {
                    Err(error::ENCODED)
                };
            }
        }
    }
    Err(error::ENCODED)
}

/// The card's integrity check of an encoded entry, against the seed it computed when it started
/// (`context_seed(.., op::READ_ENCODED)`, flipped by [`TRACE_DOMAIN`] in a trace).
#[inline(always)]
pub const fn verify_with(e: &[u32; 8], seed: u32) -> bool {
    e[CHECK_WORD] == check_with(e, seed)
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
    // The destination is L1, which `Descriptor::decode` bounds by `L1_SIZE`, not by `MMIO_START` as `DramMove` does.
    let l1_room = (crate::tensix::L1_SIZE as u32).wrapping_sub(len);
    (len.wrapping_sub(1) < MAX_REQUEST_BYTES)
        & (targ <= room)
        & (ret <= l1_room)
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
    const KEY: u32 = 0xC0FF_EE11;

    fn ctx() -> Context {
        Context::new(0xFF, ME, KEY)
    }

    /// The checksum in `ctx()`: what a forger who knows the key, or a bug in `encode`, computes.
    fn check(e: &[u32; 8]) -> u32 {
        check_with(e, ctx().seed(op::READ_ENCODED))
    }

    fn verify(e: &[u32; 8]) -> bool {
        verify_with(e, ctx().seed(op::READ_ENCODED))
    }

    fn plain(channel: u32, port: u32, offset: u32, l1: u32, len: u32) -> [u32; 8] {
        [op::READ, channel, port, offset, l1, len, 0, 0]
    }

    fn ok_bit(bits: &[u32; ENDPOINT_WORDS]) -> impl Fn(u32) -> bool + '_ {
        move |hi| bits[(hi / 32) as usize] >> (hi % 32) & 1 != 0
    }

    fn good() -> [u32; 8] {
        encode(&ctx(), plain(1, 0, 0x1000, 0x2_1000, 4096))
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
        assert_eq!(validate(&ctx(), e), Ok(()));
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
        assert_eq!(encode(&ctx(), [op::WAIT, 0, 0, 0, 0, 0, 0, 0]), Ok(None));
        assert_eq!(
            encode(&ctx(), plain(0, 0, 0, 0x2_0000, MAX_REQUEST_BYTES + 64)),
            Ok(None)
        );
        // ... and what the mover would refuse is refused, with its code.
        assert_eq!(
            encode(&ctx(), plain(9, 0, 0, 0x2_0000, 64)),
            Err(error::RANGE)
        );
        assert_eq!(
            encode(&ctx(), plain(0, 0, 0x40, 0x2_0020, 64)),
            Err(error::ALIGNMENT)
        );
        assert_eq!(
            encode(&ctx(), plain(0, 0, 0, 0x2_0000, 0)),
            Err(error::LENGTH)
        );
        // A channel the chip does not report usable.
        assert_eq!(
            encode(&Context::new(0xFE, ME, KEY), plain(0, 0, 0, 0x2_0000, 64)),
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

    /// An entry made for one mover verifies for no other: not after a restart (a new key), not on
    /// another tile or chip, not as another op, and not in a trace's domain.
    #[test]
    fn an_entry_is_bound_to_the_mover_it_was_made_for() {
        let e = good();
        let seed = ctx().seed(op::READ_ENCODED);
        assert!(verify_with(&e, seed));
        let others = [
            (
                "another key",
                context_seed(KEY ^ 1, ME, 0xFF, op::READ_ENCODED),
            ),
            (
                "another x",
                context_seed(KEY, (ME.0 + 1, ME.1), 0xFF, op::READ_ENCODED),
            ),
            (
                "another y",
                context_seed(KEY, (ME.0, ME.1 + 1), 0xFF, op::READ_ENCODED),
            ),
            (
                "another chip",
                context_seed(KEY, ME, 0xFE, op::READ_ENCODED),
            ),
            ("another op", context_seed(KEY, ME, 0xFF, op::READ)),
            ("a trace", seed ^ TRACE_DOMAIN),
        ];
        for (what, other) in others {
            assert!(!verify_with(&e, other), "an entry verified for {what}");
        }
        // The host refuses it the same way: it validates in its own context.
        assert!(validate(&Context::new(0xFF, ME, KEY ^ 1), e).is_err());
        assert!(validate(&Context::new(0xFF, (ME.0 + 1, ME.1), KEY), e).is_err());
    }

    /// No two contexts share a seed: one flipped bit of any component changes it.
    #[test]
    fn a_seed_depends_on_every_bit_of_its_context() {
        let seed =
            |key: u32, x: u8, y: u8, usable: u32, o: u32| context_seed(key, (x, y), usable, o);
        let base = seed(KEY, 1, 2, 0xFF, op::READ_ENCODED);
        for bit in 0..32 {
            assert_ne!(
                seed(KEY ^ 1 << bit, 1, 2, 0xFF, op::READ_ENCODED),
                base,
                "key {bit}"
            );
        }
        for bit in 0..8 {
            assert_ne!(
                seed(KEY, 1 ^ 1 << bit, 2, 0xFF, op::READ_ENCODED),
                base,
                "x {bit}"
            );
            assert_ne!(
                seed(KEY, 1, 2 ^ 1 << bit, 0xFF, op::READ_ENCODED),
                base,
                "y {bit}"
            );
            assert_ne!(
                seed(KEY, 1, 2, 0xFF ^ 1 << bit, op::READ_ENCODED),
                base,
                "usable {bit}"
            );
            assert_ne!(
                seed(KEY, 1, 2, 0xFF, op::READ_ENCODED ^ 1 << bit),
                base,
                "op {bit}"
            );
        }
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
                validate(&ctx(), bad).is_err(),
                "validate let through {what}: {bad:x?}"
            );
        }
        // A channel the chip does not report usable.
        assert_eq!(
            validate(&Context::new(0xFD, ME, KEY), good()),
            Err(error::ENCODED)
        );
        // And a plain entry is not an encoded one.
        assert_eq!(
            validate(&ctx(), plain(1, 0, 0, 0x2_0000, 64)),
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
            let e = encode(&ctx(), plain(c, p, off, l1, len)).unwrap().unwrap();
            assert_eq!(validate(&ctx(), e), Ok(()), "{e:x?}");
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
            (
                "destination past L1",
                forged(e, 3, crate::tensix::L1_SIZE as u32 - 64 + 128),
            ),
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

/// Differential property tests: the encoder against an independent assembly of the same request
/// (`Command::Read`, which builds the NIU words by hand, not through `DramMove`), and the card's guard
/// against the facts it promises. 100,000 cases by default; `TT_FUZZ_ITERS` sets the count and the
/// `#[ignore]` tests run millions.
#[cfg(test)]
mod fuzz {
    extern crate std;
    use super::*;
    use crate::dm::Mover;
    use crate::noc::niu::Command;
    use crate::tensix::L1_SIZE;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
        fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
            xs[self.below(xs.len() as u64) as usize]
        }
    }

    fn iters(default: u64) -> u64 {
        std::env::var("TT_FUZZ_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    }

    const KEY: u32 = 0x5EED_1234;
    const CHANNEL: u32 = crate::dram::CHANNEL_BYTES as u32;
    const MAX_REQ: u32 = MAX_REQUEST_BYTES;

    /// A plain `READ` the mover might be sent: mostly well-formed, biased to the edges of every check.
    fn plain_case(r: &mut Rng) -> (u32, (u8, u8), [u32; 8]) {
        let (r_usable, r_len, r_off, r_page, r_l1, r_any, r_ch) = (
            r.below(256) as u32,
            r.below(40_000) as u32,
            (r.below(1 << 18) as u32) * 64,
            (r.below(20_000) as u32) * 64,
            r.next() as u32 & 0x00FF_FFFF,
            r.next() as u32,
            r.below(12) as u32,
        );
        let usable = r.pick(&[0xFFu32, 0xFF, 0xFF, 0x0F, 0xF0, 0x81, r_usable]);
        let me = (r.below(64) as u8, r.below(64) as u8);
        let len = r.pick(&[
            0,
            1,
            16,
            63,
            64,
            65,
            2048,
            4096,
            4160,
            MAX_REQ - 64,
            MAX_REQ - 1,
            MAX_REQ,
            MAX_REQ + 1,
            MAX_REQ + 64,
            r_len,
        ]);
        let off = r.pick(&[
            0,
            64,
            r_off,
            CHANNEL.wrapping_sub(len),
            CHANNEL.wrapping_sub(len).wrapping_add(64),
            CHANNEL - 64,
            r_any,
        ]);
        let top = (L1_SIZE as u32).wrapping_sub(len) / 64 * 64 + off % 64;
        let l1 = r.pick(&[
            off % 64 + r_page,
            off % 64,
            top,
            top.wrapping_add(64),
            off % 64 + 16,
            off.wrapping_add(16),
            r_l1,
        ]);
        let ch = r.pick(&[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, r_ch]);
        let port = r.pick(&[0u32, 1, 2, 3]);
        (usable, me, [op::READ, ch, port, off, l1, len, 0, 0])
    }

    /// What the safe mover's checks and an independent assembly of the request say about `e`.
    /// `Err(code)` refused; `Ok(None)` more than one request; `Ok(Some(words))` the request.
    fn oracle(usable: u32, me: (u8, u8), e: [u32; 8]) -> Result<Option<[u32; 7]>, u32> {
        assert!(Mover::B.permits(op::READ));
        let d = Descriptor::decode(usable, op::READ, e[1], e[2], e[3], e[4], e[5])?;
        if d.range.len() > MAX_REQ as u64 {
            return Ok(None);
        }
        let port = d.range.channel().port_for(Niu::Noc0, d.port);
        let at = d
            .range
            .channel()
            .endpoint(Niu::Noc0, port)
            .ok_or(error::ALIGNMENT)?;
        let from = Endpoint {
            x: at.x(),
            y: at.y(),
            addr: d.range.offset() as u32,
        };
        let txn = TxnId::new(TXN_B).unwrap();
        let regs = Command::Read {
            from,
            to_local: d.l1,
            len: d.range.len() as u32,
        }
        .registers(me, txn, Niu::Noc0)
        .map_err(|_| error::ALIGNMENT)?;
        let v: [u32; 10] = core::array::from_fn(|i| regs[i].1);
        // (TARG_LO, MID, TARG_HI, RET_LO, MID, RET_HI, TAG, CTRL, LEN, DATA)
        assert_eq!((v[1], v[4], v[9]), (0, 0, 0));
        Ok(Some([v[0], v[2], v[3], v[5], v[6], v[7], v[8]]))
    }

    #[test]
    fn the_encoder_agrees_with_an_independent_assembly_of_the_request() {
        let mut r = Rng(0xE1);
        let (mut accepted, mut refused, mut long) = (0u64, 0u64, 0u64);
        for _ in 0..iters(100_000) {
            let (usable, me, e) = plain_case(&mut r);
            let ctx = Context::new(usable, me, KEY);
            let got = encode(&ctx, e);
            match oracle(usable, me, e) {
                Err(code) => {
                    assert!(
                        got.is_err(),
                        "{usable:#x} {me:?} {e:x?}: the oracle refuses ({code}), encode accepts"
                    );
                    // The decoder's refusals keep their codes; a request-level refusal is `ALIGNMENT`.
                    if Descriptor::decode(usable, op::READ, e[1], e[2], e[3], e[4], e[5]).is_err() {
                        assert_eq!(got, Err(code), "{e:x?}");
                    }
                    refused += 1;
                }
                Ok(None) => {
                    assert_eq!(got, Ok(None), "{usable:#x} {e:x?}");
                    long += 1;
                }
                Ok(Some([targ, targ_hi, ret, ret_hi, tag, ctrl, len])) => {
                    assert_eq!((tag, ctrl), (TAG, CTRL), "{e:x?}");
                    let mut want = [op::READ_ENCODED, targ, targ_hi, ret, ret_hi, 0, 0, len];
                    want[CHECK_WORD] = check_with(&want, ctx.seed(op::READ_ENCODED));
                    assert_eq!(got, Ok(Some(want)), "{usable:#x} {me:?} {e:x?}");
                    assert_eq!(validate(&ctx, want), Ok(()), "{usable:#x} {me:?} {want:x?}");
                    accepted += 1;
                }
            }
        }
        // The generator reaches all three outcomes.
        assert!(
            accepted > 100 && refused > 100 && long > 10,
            "{accepted} {refused} {long}"
        );
    }

    /// Entries that are encoder output with some words replaced and the checksum made good again,
    /// or built from scratch: what a bug in `encode` or a forger would hand the card.
    fn raw_case(r: &mut Rng) -> (u32, (u8, u8), [u32; 8]) {
        let (usable, me, p) = plain_case(r);
        let scratch = [
            op::READ_ENCODED,
            r.next() as u32,
            r.below(1 << 13) as u32,
            r.next() as u32,
            r.below(1 << 13) as u32,
            0,
            0,
            r.below(40_000) as u32,
        ];
        let ctx = Context::new(usable, me, KEY);
        let mut e = match encode(&ctx, p) {
            Ok(Some(e)) => e,
            _ => scratch,
        };
        for _ in 0..r.below(3) {
            let k = r.pick(&[1usize, 2, 3, 4, 6, 7]);
            let (any, small) = (r.next() as u32, r.below(1 << 13) as u32);
            let l1_end = (L1_SIZE as u32).wrapping_sub(e[7]);
            e[k] = r.pick(&[
                0,
                any,
                small,
                e[k].wrapping_add(16),
                e[k].wrapping_add(64),
                l1_end,
                l1_end.wrapping_add(64),
                CHANNEL.wrapping_sub(e[7]),
                endpoint_hi(me.0, me.1),
            ]);
        }
        e[CHECK_WORD] = check_with(&e, ctx.seed(op::READ_ENCODED));
        (usable, me, e)
    }

    #[test]
    fn the_guard_promises_what_the_safe_decode_checks() {
        let mut r = Rng(0x6A);
        let (mut passed, mut valid) = (0u64, 0u64);
        for _ in 0..iters(100_000) {
            let (usable, me, e) = raw_case(&mut r);
            let ctx = Context::new(usable, me, KEY);
            let bits = endpoint_bitmap(usable);
            let me_hi = endpoint_hi(me.0, me.1);
            let ok = |hi: u32| bits[(hi / 32) as usize] >> (hi % 32) & 1 != 0;
            let g = guard(&e, me_hi, ok);
            if validate(&ctx, e).is_ok() {
                valid += 1;
                assert!(
                    g,
                    "validate accepts, the guard refuses: {usable:#x} {me:?} {e:x?}"
                );
            }
            if g {
                passed += 1;
                let (targ, hi, ret, ret_hi, len) = (e[1], e[2], e[3], e[4], e[7]);
                assert!((1..=MAX_REQ).contains(&len), "{e:x?}");
                assert!(
                    targ as u64 + len as u64 <= CHANNEL as u64,
                    "target past the channel: {e:x?}"
                );
                assert!(
                    ret as u64 + len as u64 <= L1_SIZE,
                    "destination past L1: {usable:#x} {me:?} {e:x?}"
                );
                assert_eq!((targ ^ ret) & 63, 0, "{e:x?}");
                assert_eq!(ret_hi, me_hi, "{e:x?}");
                assert!(
                    hi < 4096 && ok(hi),
                    "target is not a usable endpoint: {e:x?}"
                );
                assert_eq!(e[6], 0, "{e:x?}");
            }
        }
        assert!(passed > 100 && valid > 100, "{passed} {valid}");
    }

    #[test]
    #[ignore = "millions of cases; TT_FUZZ_ITERS overrides"]
    fn millions_of_cases() {
        let n = iters(5_000_000);
        std::env::set_var("TT_FUZZ_ITERS", std::format!("{n}"));
        the_encoder_agrees_with_an_independent_assembly_of_the_request();
        the_guard_promises_what_the_safe_decode_checks();
    }
}
