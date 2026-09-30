//! The data mover on RISCV B: GDDR <-> Tensix L1, driven by descriptors.
//!
//! Phase 9 keeps tensors in DRAM, and a Tensix kernel can only read L1 (the
//! unpackers take their operands from L1, `UNPACR_Regular.md:1`), so something
//! on the tile has to pull tiles in and push results out. That is RISCV B's job
//! here, as it is tt-metal's: the three T cores drive the coprocessor, and B is
//! free. B's image is resident -- loaded once and left running -- and the host
//! (later, the compute roles) hands it one descriptor at a time.
//!
//! The protocol, host side:
//! 1. wait until [`DONE`] equals the last sequence number submitted;
//! 2. write the descriptor words;
//! 3. write the next sequence number to [`SEQ`].
//!
//! The mover sees `SEQ` change, performs the copy in [`crate::noc::niu`] requests
//! of at most 16 KiB, waits for them all to complete, and writes `SEQ` to
//! [`DONE`]. On a refused descriptor it writes the reason to [`ERROR`] and
//! `SEQ` to `DONE` without moving anything, so the host never waits forever.

use crate::dram::{Dram, DramRange};

/// The mover's mailbox: the slot after the three role mailboxes
/// ([`crate::mailbox::role`]), which ends exactly where the program region begins.
pub const MAILBOX_BASE: u64 = crate::mailbox::role::BASE + 3 * crate::mailbox::role::STRIDE;
const _: () = assert!(MAILBOX_BASE + 0x100 <= crate::mailbox::PROGRAM_REGION);
const _: () = assert!(MAILBOX_BASE % 16 == 0);

/// Where the image is linked and loaded: RISCV B has no reset-PC override and
/// always starts at L1 offset 0 (`tensix::Core::B`).
pub const IMAGE_BASE: u64 = crate::tensix::Core::B.default_reset_pc() as u64;
/// The image may not reach T0's default reset PC.
pub const IMAGE_MAX: u64 = crate::tensix::Core::T0.default_reset_pc() as u64 - IMAGE_BASE;

// The runtime's own words (status, heartbeat, panic code) sit at the start of
// the mailbox, as in every image (`crate::mailbox::offset`); the descriptor
// follows them.

/// Host -> mover: the sequence number of the descriptor to run. Never 0.
pub const SEQ: u64 = MAILBOX_BASE + 0x20;
/// Mover -> host: the last sequence number finished (or refused).
pub const DONE: u64 = MAILBOX_BASE + 0x24;
/// Mover -> host: why the last descriptor was refused, or [`error::NONE`].
pub const ERROR: u64 = MAILBOX_BASE + 0x28;
/// [`op::READ`] or [`op::WRITE`].
pub const OP: u64 = MAILBOX_BASE + 0x2C;
/// DRAM channel index.
pub const CHANNEL: u64 = MAILBOX_BASE + 0x30;
/// Which of the channel's three endpoints to use.
pub const PORT: u64 = MAILBOX_BASE + 0x34;
/// Byte offset within the channel.
pub const DRAM_OFFSET: u64 = MAILBOX_BASE + 0x38;
/// Byte address in this tile's L1.
pub const L1_ADDR: u64 = MAILBOX_BASE + 0x3C;
/// Bytes to move.
pub const LEN: u64 = MAILBOX_BASE + 0x40;
/// This tile's own NoC #0 coordinate, as the host addresses it (translated on
/// silicon): the return address of every read. Written once, before start.
pub const MY_X: u64 = MAILBOX_BASE + 0x44;
pub const MY_Y: u64 = MAILBOX_BASE + 0x48;
/// The chip's usable-channel mask, as the host read it from the ARC
/// ([`Dram::usable_mask`]). Written once, before start: the mover can only
/// name channels the chip said it has.
pub const USABLE: u64 = MAILBOX_BASE + 0x4C;

pub mod op {
    /// DRAM -> L1.
    pub const READ: u32 = 1;
    /// L1 -> DRAM.
    pub const WRITE: u32 = 2;
}

pub mod error {
    pub const NONE: u32 = 0;
    /// An op that is neither read nor write.
    pub const OP: u32 = 1;
    /// A channel the chip does not report usable, or a range past its extent.
    pub const RANGE: u32 = 2;
    /// A port beyond the channel's three.
    pub const PORT: u32 = 3;
    /// DRAM and L1 addresses not congruent mod 64 (read, `dram::ALIGN`) or 16
    /// (write), or an L1 range outside L1.
    pub const ALIGNMENT: u32 = 4;
    /// Zero bytes.
    pub const LENGTH: u32 = 5;
}

/// A descriptor, as both sides see it.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Descriptor {
    pub op: u32,
    pub range: DramRange,
    pub port: u8,
    pub l1: u32,
}

impl Descriptor {
    /// Decode the raw words the mover reads, against the chip's `usable` mask.
    /// Everything the mover does starts here, so every refusal is here too.
    pub fn decode(
        usable: u32,
        op: u32,
        channel: u32,
        port: u32,
        offset: u32,
        l1: u32,
        len: u32,
    ) -> Result<Self, u32> {
        if op != op::READ && op != op::WRITE {
            return Err(error::OP);
        }
        if len == 0 {
            return Err(error::LENGTH);
        }
        if port >= crate::dram::PORTS as u32 {
            return Err(error::PORT);
        }
        let dram = Dram::from_usable_mask(usable as u8);
        let range = u8::try_from(channel)
            .ok()
            .and_then(|c| dram.channel(c))
            .and_then(|c| c.range(offset as u64, len as u64))
            .ok_or(error::RANGE)?;
        let modulus = if op == op::READ {
            crate::dram::ALIGN
        } else {
            16
        };
        let l1_end = l1 as u64 + len as u64;
        if (offset as u64) % modulus != (l1 as u64) % modulus || l1_end > crate::tensix::L1_SIZE {
            return Err(error::ALIGNMENT);
        }
        Ok(Descriptor {
            op,
            range,
            port: port as u8,
            l1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: u32 = 0xFF;

    #[test]
    fn a_good_read_decodes() {
        let d = Descriptor::decode(ALL, op::READ, 3, 2, 0x40, 0x2_0040, 4096).unwrap();
        assert_eq!(d.range.channel().index(), 3);
        assert_eq!((d.range.offset(), d.range.len(), d.port), (0x40, 4096, 2));
    }

    #[test]
    fn every_refusal_has_its_code() {
        let dec = |u, op, ch, p, off, l1, len| Descriptor::decode(u, op, ch, p, off, l1, len);
        assert_eq!(dec(ALL, 3, 0, 0, 0, 0x2_0000, 16), Err(error::OP));
        assert_eq!(dec(ALL, op::READ, 0, 0, 0, 0x2_0000, 0), Err(error::LENGTH));
        assert_eq!(dec(ALL, op::READ, 0, 3, 0, 0x2_0000, 16), Err(error::PORT));
        assert_eq!(dec(ALL, op::READ, 8, 0, 0, 0x2_0000, 16), Err(error::RANGE));
        // A channel the chip did not report cannot be named.
        assert_eq!(
            dec(0xFE, op::READ, 0, 0, 0, 0x2_0000, 16),
            Err(error::RANGE)
        );
        assert_eq!(
            dec(ALL, op::READ, 0, 0, 0xFEFF_FFF0, 0x2_0000, 32),
            Err(error::RANGE)
        );
        // C64 for reads, C16 for writes.
        assert_eq!(
            dec(ALL, op::READ, 0, 0, 0x20, 0x2_0000, 16),
            Err(error::ALIGNMENT)
        );
        assert_eq!(
            dec(ALL, op::READ, 0, 0, 0x10, 0x2_0000, 16),
            Err(error::ALIGNMENT)
        );
        assert!(dec(ALL, op::WRITE, 0, 0, 0x10, 0x2_0010, 16).is_ok());
        assert_eq!(
            dec(ALL, op::WRITE, 0, 0, 0x8, 0x2_0000, 16),
            Err(error::ALIGNMENT)
        );
        // L1 ends at 1.5 MiB.
        assert_eq!(
            dec(
                ALL,
                op::READ,
                0,
                0,
                0,
                crate::tensix::L1_SIZE as u32 - 64,
                128
            ),
            Err(error::ALIGNMENT)
        );
    }
}
