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
pub const INPUT: u64 = dm::nc::MAILBOX_BASE + 0x100;
pub const OUTPUT: u64 = INPUT + 8;
pub const ABORT: u64 = OUTPUT + 8;
pub const ROLE_PROGRESS: u64 = ABORT + 4;
pub const WAIT_REASON: u64 = ROLE_PROGRESS + 12;
/// The transfer channel's received and consumed counters.
pub const TRANSFER: u64 = WAIT_REASON + 20;
pub const END: u64 = TRANSFER + 8;
pub const OUTER_CHUNK: u64 = 0x1_3000;
const _: () = assert!(OUTER_CHUNK >= dm::nc::STUB_AT + 8);
const _: () = assert!(OUTER_CHUNK + dm::TRACE_CHUNK_ENTRIES as u64 * dm::ENTRY_BYTES <= dm::LIST);
const _: () = assert!(END <= mailbox::PROGRAM_REGION);

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
