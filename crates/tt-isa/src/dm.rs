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
    /// Run [`super::LEN`] list entries ([`super::Entry`]) from [`super::LIST`],
    /// in order. The other descriptor words are unused.
    pub const LIST: u32 = 3;
    /// As [`READ`], of exactly one tile slot ([`super::TILE_SLOT`]), transposing
    /// the tile on the way: the slot lands in [`super::SCRATCH`] and the mover
    /// writes its transpose to the destination. Only in a list entry.
    pub const READ_TRANSPOSED: u32 = 4;
    /// Element-wise arithmetic on whole tiles already in L1, by the mover's
    /// own FP32 unit: `[COMPUTE, kind, scalar, 0, dst, a, b, 0]`, each address a
    /// tile slot. See [`super::kind`]. Only in a list entry.
    pub const COMPUTE: u32 = 5;
    /// Run the tile's resident roles once: `[KERNEL, generation, 0, ...]`.
    /// The mover waits for every move before it, then writes `generation` to
    /// the three role mailboxes' `GENERATION` (`crate::mailbox::role`) and waits
    /// until each has acknowledged it -- or reports [`super::error::ROLE`] if
    /// one panics. The host has staged the roles' programs and descriptors and
    /// says which generation is next, so a whole op -- gather, compute, scatter
    /// -- is one list and one host round trip. Only in a list entry.
    pub const KERNEL: u32 = 6;
    /// Wait for every move before it to complete: the boundary between what
    /// were separate lists, whose entries may reuse each other's L1 slots.
    /// Only in a list entry.
    pub const WAIT: u32 = 7;
}

/// What an [`op::COMPUTE`] entry computes, datum by datum over a tile's 1024
/// (`dst`, `a`, `b` are slots; `s` is the entry's scalar, as FP32 bits).
///
/// Done with the baby RISC-V's `fadd.s`/`fsub.s`/`fmul.s`, which round to
/// nearest even and flush denormals (`BabyRISCV/InstructionSet.md:18-22`) --
/// the IEEE result for every normal operand and result. Never `fmadd.s`: its
/// semantics are neither fused nor separate, and the firmware's instruction
/// gate refuses it.
pub mod kind {
    /// `a + b`.
    pub const ADD: u32 = 1;
    /// `a - b`.
    pub const SUB: u32 = 2;
    /// `a * b`.
    pub const MUL: u32 = 3;
    /// `a * s`.
    pub const MUL_SCALAR: u32 = 4;
    /// `max(a, 0)`, as `burn-flex` has it.
    pub const RELU: u32 = 5;
    /// `a > 0 ? b : 0`: `a` the forward output, `b` the gradient.
    pub const RELU_BACKWARD: u32 = 6;
    /// `a[r, c] + b[0, c]`: `b`'s first row broadcast down the tile.
    pub const ADD_ROW: u32 = 7;
    /// `dst[0, c] = acc + a[0, c] + a[1, c] + ... + a[31, c]`, added in row
    /// order, where `acc` is `dst[0, c]`, or `+0.0` when the scalar is
    /// non-zero (the first tile of a column). Only `dst`'s row 0 is written.
    /// Over a column of tiles in order, this is `burn-flex`'s `sum_dim(0)`
    /// order exactly (`ops/reduce.rs:959-989`: from `0.0`, rows in order).
    pub const COL_SUM: u32 = 8;
    pub const LAST: u32 = COL_SUM;
}

/// Where a descriptor list lives, and how many entries it may hold.
///
/// Between NC's default reset PC region (NC is not used) and the matmul
/// staging area (`tt_kernels::matmul::MATMUL_STAGE`, `0x2_0000`).
pub const LIST: u64 = 0x1_4000;
pub const LIST_MAX: u32 = 512;
/// Bytes per list entry: eight words, `[op, channel, port, offset, l1, len, 0, 0]`.
pub const ENTRY_BYTES: u64 = 32;
/// The transpose scratch slot, after the list.
pub const SCRATCH: u64 = LIST + LIST_MAX as u64 * ENTRY_BYTES;
const _: () = assert!(SCRATCH + TILE_SLOT <= 0x2_0000);

/// One FP32 32x32 tile as it is stored on the device: the 16-byte header
/// (zero, as `tt_layout` writes it), 1024 datums in face order, and padding to
/// a multiple of [`crate::dram::ALIGN`] -- so every slot is 64-byte aligned
/// wherever it sits, and any slot may be copied to any other under the C64
/// read rule (divergence row 64).
pub const TILE_SLOT: u64 = 4160;
/// Where a tile's datums start within its slot.
pub const TILE_DATA: u64 = 16;
const _: () = assert!(TILE_SLOT % crate::dram::ALIGN == 0);
const _: () = assert!(TILE_DATA + 4096 <= TILE_SLOT);

/// The datum index of tile position `(row, col)` in the face order `tt_layout`
/// and the packer use: four 16x16 faces, `[0 1; 2 3]`, each row-major.
pub const fn face_index(row: usize, col: usize) -> usize {
    ((row / 16) * 2 + col / 16) * 256 + (row % 16) * 16 + (col % 16)
}

/// A decoded list entry: a move ([`Descriptor`], possibly a transposed tile
/// read), or element-wise compute on tiles in L1.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Entry {
    Move {
        descriptor: Descriptor,
        transpose: bool,
    },
    Compute {
        kind: u32,
        scalar: u32,
        dst: u32,
        a: u32,
        b: u32,
    },
    /// [`op::KERNEL`]: post `generation` to the resident roles, and wait for it.
    Kernel { generation: u32 },
    /// [`op::WAIT`].
    Wait,
}

impl Entry {
    /// Decode entry words against the `usable` mask. A transposed read must be
    /// exactly one slot into a 16-aligned L1 slot inside L1.
    pub fn decode(usable: u32, w: [u32; 8]) -> Result<Self, u32> {
        let transpose = w[0] == op::READ_TRANSPOSED;
        if transpose {
            if w[5] as u64 != TILE_SLOT || w[4] % 16 != 0 {
                return Err(error::LENGTH);
            }
            // The NoC half lands in the scratch slot, 64-aligned; check it so.
            let d = Descriptor::decode(usable, op::READ, w[1], w[2], w[3], SCRATCH as u32, w[5])?;
            if w[4] as u64 + TILE_SLOT > crate::tensix::L1_SIZE {
                return Err(error::ALIGNMENT);
            }
            return Ok(Entry::Move {
                descriptor: Descriptor { l1: w[4], ..d },
                transpose,
            });
        }
        if w[0] == op::LIST {
            return Err(error::OP);
        }
        if w[0] == op::KERNEL {
            // Zero is what a resident runner reads as "not resident".
            if w[1] == 0 {
                return Err(error::GENERATION);
            }
            return Ok(Entry::Kernel { generation: w[1] });
        }
        if w[0] == op::WAIT {
            return Ok(Entry::Wait);
        }
        if w[0] == op::COMPUTE {
            let slot = |at: u32| at % 16 == 0 && at as u64 + TILE_SLOT <= crate::tensix::L1_SIZE;
            if w[1] == 0 || w[1] > kind::LAST {
                return Err(error::OP);
            }
            if !slot(w[4]) || !slot(w[5]) || !slot(w[6]) {
                return Err(error::ALIGNMENT);
            }
            return Ok(Entry::Compute {
                kind: w[1],
                scalar: w[2],
                dst: w[4],
                a: w[5],
                b: w[6],
            });
        }
        Descriptor::decode(usable, w[0], w[1], w[2], w[3], w[4], w[5]).map(|descriptor| {
            Entry::Move {
                descriptor,
                transpose,
            }
        })
    }
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
    /// A role panicked while running an [`super::op::KERNEL`] entry.
    pub const ROLE: u32 = 6;
    /// A [`super::op::KERNEL`] entry with generation zero.
    pub const GENERATION: u32 = 7;
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
        // (`op::LIST` is dispatched before a descriptor is decoded.)
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
    fn face_index_matches_the_four_face_order() {
        assert_eq!(face_index(0, 0), 0);
        assert_eq!(face_index(0, 16), 256);
        assert_eq!(face_index(16, 0), 512);
        assert_eq!(face_index(31, 31), 1023);
        assert_eq!(face_index(1, 2), 18);
    }

    #[test]
    fn list_entries_decode_and_refuse() {
        let e = Entry::decode(
            ALL,
            [
                op::READ_TRANSPOSED,
                2,
                0,
                0x1040,
                0x2_0010,
                TILE_SLOT as u32,
                0,
                0,
            ],
        )
        .unwrap();
        let Entry::Move {
            descriptor,
            transpose,
        } = e
        else {
            panic!("{e:?}")
        };
        assert!(transpose);
        assert_eq!(
            (descriptor.l1, descriptor.range.offset()),
            (0x2_0010, 0x1040)
        );
        // A transposed read is one whole slot, 64-aligned in DRAM (the scratch is).
        assert!(Entry::decode(
            ALL,
            [op::READ_TRANSPOSED, 2, 0, 0x1040, 0x2_0000, 4096, 0, 0]
        )
        .is_err());
        assert!(Entry::decode(
            ALL,
            [
                op::READ_TRANSPOSED,
                2,
                0,
                0x1010,
                0x2_0000,
                TILE_SLOT as u32,
                0,
                0
            ]
        )
        .is_err());
        // No lists inside lists.
        assert_eq!(
            Entry::decode(ALL, [op::LIST, 0, 0, 0, 0, 1, 0, 0]),
            Err(error::OP)
        );
        let w = Entry::decode(ALL, [op::WRITE, 1, 1, 0x40, 0x2_0040, 64, 0, 0]).unwrap();
        assert!(matches!(
            w,
            Entry::Move {
                transpose: false,
                ..
            }
        ));
        // Compute: a known kind, three slots inside L1.
        let c = [
            op::COMPUTE,
            kind::ADD,
            0,
            0,
            0x2_0000,
            0x2_1040,
            0x2_2080,
            0,
        ];
        assert!(matches!(
            Entry::decode(ALL, c),
            Ok(Entry::Compute {
                kind: kind::ADD,
                ..
            })
        ));
        let mut bad = c;
        bad[1] = kind::LAST + 1;
        assert_eq!(Entry::decode(ALL, bad), Err(error::OP));
        let mut bad = c;
        bad[6] = crate::tensix::L1_SIZE as u32 - 64;
        assert_eq!(Entry::decode(ALL, bad), Err(error::ALIGNMENT));
        // A kernel entry names a non-zero generation; a wait takes nothing.
        assert_eq!(
            Entry::decode(ALL, [op::KERNEL, 7, 0, 0, 0, 0, 0, 0]),
            Ok(Entry::Kernel { generation: 7 })
        );
        assert_eq!(
            Entry::decode(ALL, [op::KERNEL, 0, 0, 0, 0, 0, 0, 0]),
            Err(error::GENERATION)
        );
        assert_eq!(
            Entry::decode(ALL, [op::WAIT, 0, 0, 0, 0, 0, 0, 0]),
            Ok(Entry::Wait)
        );
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
