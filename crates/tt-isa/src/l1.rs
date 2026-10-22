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

/// Where the three role images (T0 unpack, T1 math, T2 pack) are linked and
/// loaded: packed against the end of the image area, each in a [`roles::SLOT`]
/// reached through its core's reset-PC override (`Device::load_and_start` sets it
/// for every core but B). The slots are what T0..T2's default reset PCs would
/// leave -- 16 KiB each, of which the images used about half -- and the space
/// they free goes to B's mover ([`dm::IMAGE_MAX`]), which cannot be moved: its
/// entry point is L1 offset 0.
pub mod roles {
    use crate::tensix::Core;

    /// One role image's slot.
    pub const SLOT: u64 = 0x2800;
    /// The end of the last slot: NC's default reset PC, where its stub lives.
    pub const END: u64 = Core::NC.default_reset_pc() as u64;
    /// The load address (and link address) of the T0, T1 and T2 role images.
    pub const BASE: [u64; 3] = [END - 3 * SLOT, END - 2 * SLOT, END - SLOT];
    const _: () = assert!(BASE[0] % 32 == 0);
}

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
    end: 0x10_0000,
};

/// What the mailboxes left when they were packed above the program slots
/// (`docs/plans/decode-to-bandwidth.md` Phase 8, "Room for a second job's `B`",
/// M2): L1 between [`DATA`] and [`PROGRAMS`] that only a trace's capture plans
/// into, with the program slots ([`TRACE_ARENA`]). Ops outside a capture never
/// see it, so their layouts, block widths and run lengths are sized from
/// [`DATA`] alone.
pub const CAPTURE_ONLY: Region = Region {
    name: "capture-only arena",
    base: DATA.end,
    end: mailbox::PROGRAM_REGION,
};

/// The four fixed program slots the role runners push from, right after
/// [`CAPTURE_ONLY`].
pub const PROGRAMS: Region = Region {
    name: "program slots",
    base: mailbox::PROGRAM_REGION,
    end: mailbox::PROGRAM_REGION_END,
};

/// The single-core, role and mover mailboxes, with their `Dst` dumps, the
/// probes' scratch, and RISCV NC's mover list ring, scratch and trace chunk
/// (`dm::nc::LIST..END`): each block in [`FIXED`].
pub const MAILBOXES: Region = Region {
    name: "mailboxes",
    base: mailbox::MAILBOX_BASE,
    end: mailbox::MAILBOX_REGION_END,
};

/// The timestamper's event buffer.
pub const TRACE: Region = Region {
    name: "trace buffer",
    base: mailbox::TRACE_BUFFER,
    end: mailbox::TRACE_BUFFER + mailbox::TRACE_BUFFER_BYTES,
};

/// The rest of L1, kept for resident kernel programs (Phase 9.7c).
pub const PROGRAM_CACHE: Region = Region {
    name: "program cache",
    base: mailbox::TRACE_BUFFER + mailbox::TRACE_BUFFER_BYTES,
    end: tensix::L1_SIZE,
};

/// RISCV NC's mover image (`dm::nc`), in what was the free L1 between B's
/// mover area and the data arena.
pub const NC_IMAGE: Region = Region {
    name: "NC mover image",
    base: dm::nc::IMAGE_BASE,
    end: dm::nc::IMAGE_BASE + dm::nc::IMAGE_MAX,
};

/// Every region, in address order. The sliver between [`MOVER`] and
/// [`NC_IMAGE`] is unused.
pub const REGIONS: [Region; 9] = [
    IMAGES,
    MOVER,
    NC_IMAGE,
    DATA,
    CAPTURE_ONLY,
    PROGRAMS,
    MAILBOXES,
    TRACE,
    PROGRAM_CACHE,
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

/// A mailbox's words and its guard block (`mailbox::guard`), from its base.
const fn mailbox_words(name: &'static str, m: mailbox::role::Mailbox) -> Region {
    Region {
        name,
        base: m.base(),
        end: m.base() + mailbox::guard::OFFSET + mailbox::guard::Guard::BYTES,
    }
}

/// A mailbox's `Dst` dump.
const fn mailbox_dump(name: &'static str, m: mailbox::role::Mailbox) -> Region {
    let base = m.dump_offset(0, 0);
    Region {
        name,
        base,
        end: base + (mailbox::DUMP_MAX_ROWS * mailbox::DUMP_ROW_WORDS * 4) as u64,
    }
}

/// Every block of [`MAILBOXES`] the firmware and the host name by a constant, in address order:
/// whatever of the region is not in one of these is unused.
pub const FIXED: [Region; 12] = {
    use mailbox::role::Mailbox;
    [
        mailbox_words("single-core mailbox", Mailbox::single_core()),
        mailbox_dump("single-core dump", Mailbox::single_core()),
        mailbox_words("role 0 mailbox", Mailbox::of(0)),
        mailbox_dump("role 0 dump", Mailbox::of(0)),
        mailbox_words("role 1 mailbox", Mailbox::of(1)),
        mailbox_dump("role 1 dump", Mailbox::of(1)),
        mailbox_words("role 2 mailbox", Mailbox::of(2)),
        mailbox_dump("role 2 dump", Mailbox::of(2)),
        Region {
            name: "B mover mailbox and read endpoints",
            base: dm::MAILBOX_BASE,
            end: dm::encoded::ENDPOINTS + dm::encoded::ENDPOINT_WORDS as u64 * 4,
        },
        Region {
            name: "NC mover mailbox and stream words",
            base: dm::nc::MAILBOX_BASE,
            end: crate::dataflow::NC_WAITS + crate::dataflow::NC_WAIT_WORDS * 4,
        },
        Region {
            name: "probe scratch",
            base: mailbox::PROBE_SCRATCH,
            end: mailbox::PROBE_SCRATCH + mailbox::PROBE_SCRATCH_BYTES,
        },
        Region {
            name: "NC mover list, scratch and trace chunk",
            base: dm::nc::LIST,
            end: dm::nc::END,
        },
    ]
};

const fn inside(r: &[Region], outer: Region) -> bool {
    let mut i = 0;
    while i < r.len() {
        if r[i].base < outer.base || r[i].end > outer.end {
            return false;
        }
        i += 1;
    }
    true
}

const _: () = assert!(ordered(&FIXED) && inside(&FIXED, MAILBOXES));
// The capture-only arena and the program slots follow the data arena, so a trace can plan
// into all three as one.
const _: () = assert!(DATA.end == CAPTURE_ONLY.base && CAPTURE_ONLY.end == PROGRAMS.base);
// Every image fits its slot below the next core's reset PC.
const _: () = assert!(roles::BASE[0] == dm::IMAGE_BASE + dm::IMAGE_MAX);
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
        assert_eq!(CAPTURE_ONLY.len(), 0x1_6000);
        // Overlapping regions are refused.
        let mut bad = REGIONS;
        bad[2].base = MOVER.base;
        assert!(!ordered(&bad));
    }

    #[test]
    fn every_fixed_block_lies_in_mailboxes_and_none_overlap() {
        assert!(ordered(&FIXED) && inside(&FIXED, MAILBOXES));
        // A block moved onto its neighbour, or past the region, is refused.
        let mut bad = FIXED;
        bad[9].base = FIXED[8].base;
        assert!(!ordered(&bad));
        let mut bad = FIXED;
        bad[11].end = MAILBOXES.end + 4;
        assert!(!inside(&bad, MAILBOXES));
    }
}
