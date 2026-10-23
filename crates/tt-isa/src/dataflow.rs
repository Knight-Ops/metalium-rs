//! Shared B-reader / role-stream / NC-writer ownership protocol.
//!
//! A streamed program is `[VERSION, batches, capacity, role]` followed by
//! `[body_address, body_length, 0, 0]` per batch. Body lengths retain the
//! existing loop flag but cannot themselves name streamed programs. Credits
//! represent contiguous compute-sized batches, with one producer and consumer.
//! Counter pairs are received/consumed pages, wrapping modulo 2^16; capacities
//! stay below half the counter range. Wait slots are B, T0, T1, T2, NC.
//!
//! A third channel, [`Stream::Transfer`], runs B to NC with no compute between:
//! a standalone transfer (an upload, a copy, an index gather) reads into L1 on
//! B and writes out on NC, one batch a credit. A packet of only transfer
//! batches has no `LAUNCH` and no `KERNEL_WAIT`.

use crate::{dm, l1, mailbox};

pub const STREAMED: u32 = 1 << 30;
pub const LENGTH_MASK: u32 = !(STREAMED | mailbox::loops::LOOPED);
pub const VERSION: u32 = 1;
pub const STEP_WORDS: u32 = 4;

/// A role program may open with a two-word *seed directive*
/// (`seed_directive`): the role firmware does not push them, it stores the
/// second word to the PRNG seed register with a full-width RISC-V store, fences,
/// and waits [`SEED_SETTLE_NOPS`] NOP iterations -- the one restart procedure
/// that restarts silicon's per-lane streams (`step91_seeded_prng`; a `WRCFG`
/// does not). Its opcode byte (0xF5) is one no Tensix instruction uses
/// (`seed_directive_opcode_is_unused`), so no program can contain it by accident.
pub const SEED_DIRECTIVE: u32 = 0xF500_5EED;
/// NOP iterations after the seed store (`step91`'s validated, conservative
/// settling interval; not a measured minimum).
pub const SEED_SETTLE_NOPS: u32 = 512;
/// Words a seed directive takes at the head of a program.
pub const SEED_WORDS: u32 = 2;

/// The two program words that restart the PRNG from `seed` before anything
/// after them runs.
pub const fn seed_directive(seed: u32) -> [u32; 2] {
    [SEED_DIRECTIVE, seed]
}
pub const INPUT: u64 = dm::nc::MAILBOX_BASE + 0x100;
pub const OUTPUT: u64 = INPUT + 8;
pub const ABORT: u64 = OUTPUT + 8;
pub const ROLE_PROGRESS: u64 = ABORT + 4;
pub const WAIT_REASON: u64 = ROLE_PROGRESS + 12;
/// The transfer channel's received and consumed counters.
pub const TRANSFER: u64 = WAIT_REASON + 20;
pub const END: u64 = TRANSFER + 8;
/// NC -> host, in an image built with the profile feature only: the tile cycles NC has spent
/// waiting, summed (wrapping) until the host zeroes them, a word each in [`NcWait`] order.
pub const NC_WAITS: u64 = END;
/// What NC waits for, in [`NC_WAITS`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum NcWait {
    /// A chunk of a `CALL`'s stream (a `PAIR_CALL`'s writer section) arriving from GDDR.
    Fetch = 0,
    /// Results to write: a `CB` `Wait` on the output stream's credits.
    Credits = 1,
    /// Its writes landing before a `CB` `Pop` hands their slots back.
    Drain = 2,
    /// Issuing its writes into other tiles' L1 (`op::WRITE_L1`, `WRITE_L1_MCAST`): the NIU taking
    /// each request.
    Issue = 3,
    /// Not a wait: the whole of each `CALL` (a pair's writer section), from taking it to reporting
    /// it done, which the others are parts of.
    Call = 4,
    /// B's, not NC's: issuing queued prefetch reads from its waits (the mover's `pump`), the time
    /// a wait for a barrier, the roles or NC goes on past what it waited for.
    Pump = 5,
    /// Not cycles: how many of [`NcWait::Pump`]'s turns issued a read.
    PumpTurns = 6,
    /// Not a sum: the longest of those turns, in cycles.
    PumpLongest = 7,
}
/// Words at [`NC_WAITS`].
pub const NC_WAIT_WORDS: u64 = 8;
pub const OUTER_CHUNK: u64 = 0x1_3000;
/// Where B reads a layer body's argument table ([`dm::op::CALL_BODY`]), and NC reads it from
/// while a body's pair runs: the free L1 between NC's stub and [`OUTER_CHUNK`].
pub const ARG_TABLE: u64 = dm::nc::STUB_AT + 0x40;
/// Entries [`ARG_TABLE`] holds: a body whose call needs more is not made
/// (`tt_kernels::subtrace`).
pub const ARG_TABLE_ENTRIES: u32 = ((OUTER_CHUNK - ARG_TABLE) / dm::ENTRY_BYTES) as u32;
const _: () = assert!(ARG_TABLE % 64 == 0 && ARG_TABLE >= dm::nc::STUB_AT + 8);
const _: () = assert!(OUTER_CHUNK >= dm::nc::STUB_AT + 8);
const _: () = assert!(OUTER_CHUNK + dm::TRACE_CHUNK_ENTRIES as u64 * dm::ENTRY_BYTES <= dm::LIST);
/// Where B reads a trace's next [`dm::op::PAIR_CALL`]'s first reader chunk ahead, on
/// [`dm::TXN_FETCH`], while it runs what comes before the pair: the free L1 between
/// [`OUTER_CHUNK`] and the list, one chunk.
pub const AHEAD_CHUNK: u64 = OUTER_CHUNK + dm::TRACE_CHUNK_ENTRIES as u64 * dm::ENTRY_BYTES;
const _: () = assert!(AHEAD_CHUNK + dm::TRACE_CHUNK_ENTRIES as u64 * dm::ENTRY_BYTES <= dm::LIST);
const _: () = assert!(NC_WAITS + NC_WAIT_WORDS * 4 <= mailbox::PROBE_SCRATCH);

/// Which credit channel a `CB` entry names (`dm::op::CB` word 1).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Stream {
    /// B to T0: operands in L1.
    Input,
    /// T2 to NC: results in L1.
    Output,
    /// B to NC: a standalone transfer's tiles, read and then written out.
    Transfer,
}

impl Stream {
    pub const fn decode(word: u32) -> Option<Self> {
        match word {
            0 => Some(Self::Input),
            1 => Some(Self::Output),
            2 => Some(Self::Transfer),
            _ => None,
        }
    }
    pub const fn word(self) -> u32 {
        match self {
            Self::Input => 0,
            Self::Output => 1,
            Self::Transfer => 2,
        }
    }
    /// The counters' address: received, then consumed.
    pub const fn address(self) -> u64 {
        match self {
            Self::Input => INPUT,
            Self::Output => OUTPUT,
            Self::Transfer => TRANSFER,
        }
    }
    pub const fn producer(self) -> Endpoint {
        match self {
            Self::Input | Self::Transfer => Endpoint::Reader,
            Self::Output => Endpoint::Pack,
        }
    }
    pub const fn consumer(self) -> Endpoint {
        match self {
            Self::Input => Endpoint::Unpack,
            Self::Output | Self::Transfer => Endpoint::Writer,
        }
    }
}

/// What a `RELEASED` entry (`dm::op::RELEASED` word 2) waits for.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Release {
    /// Output batches written out and released by NC.
    Written,
    /// Output batches packed by T2.
    Packed,
    /// Transfer batches written out and released by NC.
    Transferred,
}

impl Release {
    pub const fn decode(word: u32) -> Option<Self> {
        match word {
            0 => Some(Self::Written),
            1 => Some(Self::Packed),
            2 => Some(Self::Transferred),
            _ => None,
        }
    }
    pub const fn word(self) -> u32 {
        match self {
            Self::Written => 0,
            Self::Packed => 1,
            Self::Transferred => 2,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Endpoint {
    Reader,
    Unpack,
    Pack,
    Writer,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Action {
    Reserve,
    Wait,
    Push,
    Pop,
}

impl Action {
    pub const fn word(self) -> u32 {
        match self {
            Self::Reserve => 0,
            Self::Wait => 1,
            Self::Push => 2,
            Self::Pop => 3,
        }
    }
    pub const fn decode(word: u32) -> Option<Self> {
        match word {
            0 => Some(Self::Reserve),
            1 => Some(Self::Wait),
            2 => Some(Self::Push),
            3 => Some(Self::Pop),
            _ => None,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Channel {
    pub address: u64,
    pub capacity: u16,
    pub producer: Endpoint,
    pub consumer: Endpoint,
}

impl Channel {
    pub fn of(stream: Stream, capacity: u16) -> Option<Self> {
        (capacity > 0 && capacity < 0x8000).then_some(Self {
            address: stream.address(),
            capacity,
            producer: stream.producer(),
            consumer: stream.consumer(),
        })
    }
    pub fn permits(self, endpoint: Endpoint, action: Action) -> bool {
        match action {
            Action::Reserve | Action::Push => endpoint == self.producer,
            Action::Wait | Action::Pop => endpoint == self.consumer,
        }
    }
}

pub const fn available(received: u16, consumed: u16) -> u16 {
    received.wrapping_sub(consumed)
}
pub const fn ready(received: u16, consumed: u16, count: u16) -> bool {
    available(received, consumed) >= count
}
pub const fn room(received: u16, consumed: u16, capacity: u16, count: u16) -> bool {
    let used = available(received, consumed);
    used <= capacity && count <= capacity - used
}
/// Has a counter at `counter` reached `target`? Both count modulo 2^16, so
/// the difference is read as signed: a target up to 2^15 - 1 ahead is not
/// reached yet, and one behind is.
pub const fn reached(counter: u16, target: u16) -> bool {
    counter.wrapping_sub(target) as i16 >= 0
}
pub fn program(address: u32, length: u32) -> bool {
    let words = length & !mailbox::loops::LOOPED;
    length & STREAMED == 0
        && address % 16 == 0
        && words <= mailbox::PROGRAM_MAX
        && l1::PROGRAM_CACHE.contains(address as u64, words as u64 * 4)
}
pub fn packet_length(header: [u32; 8]) -> Result<usize, u32> {
    if header[0] != dm::op::PAIR
        || header[3] == 0
        || header[3] >= 0x8000
        || header[4] > 1
        || header[5..].iter().any(|&word| word != 0)
        || header[1] == 0
        || header[2] == 0
    {
        return Err(dm::error::OP);
    }
    let length = 1u64 + header[1] as u64 + header[2] as u64 + header[4] as u64;
    if length > dm::LIST_MAX as u64 {
        return Err(dm::error::LENGTH);
    }
    Ok(length as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn seed_directive_opcode_is_unused() {
        let opcode = (SEED_DIRECTIVE >> 24) as u8;
        assert!(crate::isa::generated::ALL
            .iter()
            .all(|def| def.opcode() != opcode));
        assert_eq!(seed_directive(7), [SEED_DIRECTIVE, 7]);
    }
    #[test]
    fn counters_wrap_and_support_large_rings() {
        for capacity in [1u16, 2, 16, 192, 32767] {
            for consumed in [0u16, 0xfff0, 0xffff] {
                for used in 0..=capacity {
                    let received = consumed.wrapping_add(used);
                    assert_eq!(available(received, consumed), used);
                    assert_eq!(room(received, consumed, capacity, 1), used < capacity);
                    assert_eq!(ready(received, consumed, 1), used != 0);
                }
            }
        }
    }
    /// Reservations of `count` pages, every counter position across the wrap:
    /// room is exactly `capacity - used`, ready exactly `used`, and a count the
    /// ring cannot hold is never granted.
    #[test]
    fn contiguous_reservations_wrap_at_every_position() {
        for capacity in [1u16, 2, 3, 15, 16, 17, 192] {
            for consumed in (0..=0x30u16).chain(0xffd0..=0xffff) {
                for used in 0..=capacity {
                    let received = consumed.wrapping_add(used);
                    for count in 0..=capacity + 2 {
                        assert_eq!(
                            room(received, consumed, capacity, count),
                            count <= capacity - used,
                            "room: capacity {capacity} consumed {consumed} used {used} count {count}"
                        );
                        assert_eq!(
                            ready(received, consumed, count),
                            count <= used,
                            "ready: capacity {capacity} consumed {consumed} used {used} count {count}"
                        );
                    }
                }
            }
        }
        // More in use than the ring holds is a corrupt counter: nothing fits.
        assert!(!room(5, 0, 4, 0));
        assert!(!room(0, 1, 4, 1));
    }
    /// The release wait compares across the wrap: a target at or behind the
    /// counter is reached, one up to 2^15 - 1 ahead is not.
    #[test]
    fn a_release_target_is_reached_across_the_wrap() {
        for counter in (0..=0x30u16).chain(0x7fe0..=0x8020).chain(0xffd0..=0xffff) {
            for ahead in 0..=0x7fffu16 {
                if ahead > 0x40 && ahead < 0x7fc0 {
                    continue;
                }
                assert_eq!(reached(counter, counter.wrapping_add(ahead)), ahead == 0);
                assert!(reached(counter, counter.wrapping_sub(ahead)));
            }
        }
        assert!(reached(0, 0xffff) && reached(2, 0xfffe) && !reached(0xfffe, 2));
    }
    #[test]
    fn ownership_is_checked() {
        let channel = Channel::of(Stream::Input, 2).unwrap();
        assert!(channel.permits(Endpoint::Reader, Action::Push));
        assert!(!channel.permits(Endpoint::Writer, Action::Push));
        assert!(Channel::of(Stream::Input, 0x8000).is_none());
        // A transfer runs from the reader to the writer, and no other way.
        let transfer = Channel::of(Stream::Transfer, 2).unwrap();
        assert!(transfer.permits(Endpoint::Reader, Action::Reserve));
        assert!(transfer.permits(Endpoint::Writer, Action::Pop));
        assert!(!transfer.permits(Endpoint::Writer, Action::Push));
        assert!(!transfer.permits(Endpoint::Reader, Action::Wait));
        assert!(!transfer.permits(Endpoint::Unpack, Action::Wait));
    }
    #[test]
    fn streams_do_not_share_counters() {
        let at = [Stream::Input, Stream::Output, Stream::Transfer].map(Stream::address);
        for (i, a) in at.iter().enumerate() {
            for b in &at[i + 1..] {
                assert!(a.abs_diff(*b) >= 8);
            }
            assert!(*a >= INPUT && *a + 8 <= END);
        }
    }
    #[test]
    fn packet_bounds_are_checked() {
        assert_eq!(packet_length([dm::op::PAIR, 5, 7, 2, 0, 0, 0, 0]), Ok(13));
        assert!(packet_length([dm::op::PAIR, u32::MAX, 7, 2, 0, 0, 0, 0]).is_err());
    }
}
