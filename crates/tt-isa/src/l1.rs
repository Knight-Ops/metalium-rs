//! What a Tensix tile's L1 holds, region by region.
//!
//! Everything fixed -- firmware images, the mover's list, the mailboxes, the
//! program slots, the trace buffer -- is one entry in [`REGIONS`], checked at
//! compile time to be in order, non-overlapping and inside L1. What kernels
//! use for their operands, results and circular buffers is [`DATA`], and no
//! kernel names an address in it: each declares what it needs and
//! `tt_kernels::l1` places it there, so two kernels fused into one cannot
//! collide by construction.

use crate::{dm, mailbox, tensix};

/// A byte range of L1, `[base, end)`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Region {
    pub name: &'static str,
    pub base: u64,
    pub end: u64,
}

impl Region {
    pub const fn len(&self) -> u64 {
        self.end - self.base
    }

    pub const fn is_empty(&self) -> bool {
        self.end == self.base
    }

    pub const fn contains(&self, at: u64, len: u64) -> bool {
        at >= self.base && at + len <= self.end
    }
}

/// The five baby RISC-V images, from B's hardwired reset PC 0 to the end of
/// NC's slot (which holds only the jump to NC's mover image, `dm::nc::stub`).
pub const IMAGES: Region = Region {
    name: "firmware images",
    base: 0,
    end: tensix::Core::NC.default_reset_pc() as u64 + 0x2000,
};

/// The data mover's descriptor list, its transpose scratch slot, and the
/// chunk a trace streams through (`dm::op::CALL`).
pub const MOVER: Region = Region {
    name: "mover list, scratch and trace chunk",
    base: dm::LIST,
    end: dm::TRACE_CHUNK + dm::TRACE_CHUNK_ENTRIES as u64 * dm::ENTRY_BYTES,
};

const _: () = assert!(dm::SCRATCH + dm::TILE_SLOT <= dm::TRACE_CHUNK);
const _: () = assert!(dm::TRACE_CHUNK % 16 == 0);

/// The data arena: every kernel's operands, results, staging and circular
/// buffers, placed by `tt_kernels::l1`.
pub const DATA: Region = Region {
    name: "data arena",
    base: 0x2_0000,
    end: mailbox::MAILBOX_BASE,
};

/// The single-core, role and mover mailboxes, with the `Dst` dump.
pub const MAILBOXES: Region = Region {
    name: "mailboxes",
    base: mailbox::MAILBOX_BASE,
    end: mailbox::PROGRAM_REGION,
};

/// The four fixed program slots the role runners push from.
pub const PROGRAMS: Region = Region {
    name: "program slots",
    base: mailbox::PROGRAM_REGION,
    end: mailbox::PROGRAM_REGION_END,
};

/// The timestamper's event buffer.
pub const TRACE: Region = Region {
    name: "trace buffer",
    base: mailbox::TRACE_BUFFER,
    end: mailbox::TRACE_BUFFER + mailbox::TRACE_BUFFER_BYTES,
};

/// The rest of L1 below [`NC_MOVER`], kept for resident kernel programs
/// (Phase 9.7c).
pub const PROGRAM_CACHE: Region = Region {
    name: "program cache",
    base: mailbox::TRACE_BUFFER + mailbox::TRACE_BUFFER_BYTES,
    end: dm::nc::IMAGE_BASE,
};

/// RISCV NC's mover: its image, list ring, scratch and trace chunk
/// (`dm::nc`), at the top of L1.
pub const NC_MOVER: Region = Region {
    name: "NC mover image, list, scratch and trace chunk",
    base: dm::nc::IMAGE_BASE,
    end: tensix::L1_SIZE,
};
const _: () = assert!(dm::nc::END <= tensix::L1_SIZE);

/// Every region, in address order. The gap between [`MOVER`] and [`DATA`] is
/// unused.
pub const REGIONS: [Region; 8] = [
    IMAGES,
    MOVER,
    DATA,
    MAILBOXES,
    PROGRAMS,
    TRACE,
    PROGRAM_CACHE,
    NC_MOVER,
];

const fn ordered(r: &[Region]) -> bool {
    let mut i = 0;
    while i < r.len() {
        if r[i].base > r[i].end || r[i].end > tensix::L1_SIZE {
            return false;
        }
        if i > 0 && r[i - 1].end > r[i].base {
            return false;
        }
        i += 1;
    }
    true
}

const _: () = assert!(ordered(&REGIONS));
// The mover's mailbox, last of the four, ends where the program slots begin.
const _: () = assert!(dm::MAILBOX_BASE + 0x100 <= MAILBOXES.end);
// Every image fits its slot below the next core's reset PC.
const _: () = assert!(tensix::Core::T0.default_reset_pc() as u64 == dm::IMAGE_BASE + dm::IMAGE_MAX);
// The data arena is slot- and DRAM-aligned at its base.
const _: () = assert!(DATA.base % crate::dram::ALIGN == 0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_regions_tile_l1_in_order() {
        assert!(ordered(&REGIONS));
        assert_eq!(REGIONS.last().unwrap().end, tensix::L1_SIZE);
        assert_eq!(DATA.len(), 0xE_0000);
        // Overlapping regions are refused.
        let mut bad = REGIONS;
        bad[2].base = MOVER.base;
        assert!(!ordered(&bad));
    }
}
