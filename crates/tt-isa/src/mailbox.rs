//! The L1 contract between host and firmware.
//!
//! Both sides compile against this module — the host to poll, the firmware to
//! publish — so the layout cannot drift between them. That is the whole reason
//! `tt-isa` is `no_std` with no dependencies: it is the one crate both targets
//! can share.
//!
//! The mailbox sits at 1 MiB into L1, clear of every core's default reset PC
//! (the highest is NC's at `0x1_2000`) with a wide margin for code growth.

/// Byte offset of the mailbox within a tile's L1.
pub const MAILBOX_BASE: u64 = 0x0010_0000;

/// Firmware liveness and status. See [`status`].
pub const STATUS: u64 = MAILBOX_BASE;
/// Monotonically incrementing counter, the heartbeat proper.
pub const HEARTBEAT: u64 = MAILBOX_BASE + 0x04;
/// Set alongside [`status::PANICKED`] to say where.
pub const PANIC_CODE: u64 = MAILBOX_BASE + 0x08;
/// Where a computation leaves its answer.
pub const RESULT: u64 = MAILBOX_BASE + 0x0C;
/// Second result word, for results wider than 32 bits.
pub const RESULT_HI: u64 = MAILBOX_BASE + 0x10;
/// First operand, written by the host before the core is released.
pub const OPERAND_A: u64 = MAILBOX_BASE + 0x14;
/// Second operand.
pub const OPERAND_B: u64 = MAILBOX_BASE + 0x18;
/// Which Tensix thread the running core drives: 0 for T0, 1 for T1, 2 for T2.
///
/// Written by the host because the core cannot work it out: `mhartid` reads zero
/// on every core and `misa` lies, so identity comes from outside or from the
/// build, never from a CSR.
pub const THREAD_INDEX: u64 = MAILBOX_BASE + 0x1C;
/// Requested `RISC_DEST_ACCESS_CTRL_SEC*.fmt`, so a test can vary it.
pub const DST_ACCESS_FMT: u64 = MAILBOX_BASE + 0x20;

/// How many instruction words the host has staged at [`PROGRAM`].
pub const PROGRAM_LEN: u64 = MAILBOX_BASE + 0x24;
/// How many `Dst` rows the firmware should copy to [`DUMP`] afterwards.
pub const DUMP_ROW_COUNT: u64 = MAILBOX_BASE + 0x28;
/// First `Dst` row to copy.
pub const DUMP_ROW_FIRST: u64 = MAILBOX_BASE + 0x2C;

/// Total size the firmware may assume is its own.
pub const MAILBOX_SIZE: u64 = 0x40;

/// A Tensix instruction stream, staged by the host.
///
/// The descriptor lives in the mailbox; the program itself does not, because it is
/// unbounded in a way the mailbox is not. Putting it here is what makes the corpus
/// firmware generic: the host encodes a program with `tt_isa::isa`, writes the
/// words, and the same image runs it. Adding an instruction to the corpus is then a
/// host-side test case rather than a firmware change.
pub const PROGRAM: u64 = MAILBOX_BASE + 0x1000;
/// Most instructions a program may hold.
pub const PROGRAM_MAX: u32 = 256;

/// Where the firmware copies `Dst` rows for the host to read.
///
/// `Dst` is not reachable over the NoC, so this is the only way a host sees a
/// compute result. Sixteen 32-bit datums per row.
pub const DUMP: u64 = MAILBOX_BASE + 0x2000;
/// Most `Dst` rows a run may copy out.
pub const DUMP_MAX_ROWS: u32 = 16;
/// Datums in one `Dst` row.
pub const DUMP_ROW_WORDS: u32 = 16;

/// Byte offset of `Dst` row `row`, datum `column`, within [`DUMP`].
pub const fn dump_offset(row: u32, column: u32) -> u64 {
    DUMP + ((row * DUMP_ROW_WORDS + column) as u64) * 4
}

// Compile-time rather than a test: if the mailbox ever moved past the end of L1,
// every access to it would be out of bounds, and that should stop the build rather
// than fail a test run.
const _: () = assert!(MAILBOX_BASE + MAILBOX_SIZE <= crate::tensix::L1_SIZE);
const _: () = assert!(PROGRAM + (PROGRAM_MAX as u64) * 4 <= DUMP);
const _: () =
    assert!(dump_offset(DUMP_MAX_ROWS - 1, DUMP_ROW_WORDS - 1) + 4 <= crate::tensix::L1_SIZE);
// The descriptor words have to stay inside the mailbox the firmware owns.
const _: () = assert!(DUMP_ROW_FIRST + 4 <= MAILBOX_BASE + MAILBOX_SIZE);

/// One mailbox per Tensix thread, for the three-role datapath.
///
/// tt-metal's LLK splits a kernel across the three Tensix threads -- thread 0
/// unpacks, thread 1 does math, thread 2 packs -- and so does the role harness,
/// because much of the coprocessor's state is per thread and some of it is tied
/// to a role by the hardware (`UNPACR` counts with thread 0's ADCs; see
/// `docs/ttsim-divergence.md` row 45). Each role image runs its own program and
/// reports through its own mailbox, laid out like the single-core one above:
/// the same offsets, from a per-thread base.
pub mod role {
    /// First role mailbox, clear of the single-core mailbox, program and dump.
    pub const BASE: u64 = 0x0011_0000;
    /// Distance between role mailboxes: room for the program and the dump.
    pub const STRIDE: u64 = 0x4000;

    /// The mailbox of the role that runs on Tensix thread `thread` (0, 1 or 2).
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Mailbox {
        base: u64,
    }

    impl Mailbox {
        /// The single-core mailbox at [`super::MAILBOX_BASE`], which has the
        /// same layout.
        pub const fn single_core() -> Mailbox {
            Mailbox {
                base: super::MAILBOX_BASE,
            }
        }
        pub const fn of(thread: u32) -> Mailbox {
            assert!(thread < 3, "there are three Tensix threads");
            Mailbox {
                base: BASE + thread as u64 * STRIDE,
            }
        }
        pub const fn base(self) -> u64 {
            self.base
        }
        pub const fn status(self) -> u64 {
            self.base + (super::STATUS - super::MAILBOX_BASE)
        }
        pub const fn panic_code(self) -> u64 {
            self.base + (super::PANIC_CODE - super::MAILBOX_BASE)
        }
        pub const fn thread_index(self) -> u64 {
            self.base + (super::THREAD_INDEX - super::MAILBOX_BASE)
        }
        pub const fn dst_access_fmt(self) -> u64 {
            self.base + (super::DST_ACCESS_FMT - super::MAILBOX_BASE)
        }
        pub const fn program_len(self) -> u64 {
            self.base + (super::PROGRAM_LEN - super::MAILBOX_BASE)
        }
        pub const fn dump_row_count(self) -> u64 {
            self.base + (super::DUMP_ROW_COUNT - super::MAILBOX_BASE)
        }
        pub const fn dump_row_first(self) -> u64 {
            self.base + (super::DUMP_ROW_FIRST - super::MAILBOX_BASE)
        }
        pub const fn program(self) -> u64 {
            self.base + (super::PROGRAM - super::MAILBOX_BASE)
        }
        pub const fn dump_offset(self, row: u32, column: u32) -> u64 {
            self.base + (super::dump_offset(row, column) - super::MAILBOX_BASE)
        }
    }

    // The last role's dump must stay inside L1.
    const _: () = assert!(
        Mailbox::of(2).dump_offset(super::DUMP_MAX_ROWS - 1, super::DUMP_ROW_WORDS - 1) + 4
            <= crate::tensix::L1_SIZE
    );
    // And every role's program must fit before its dump.
    const _: () = assert!(
        super::PROGRAM - super::MAILBOX_BASE + (super::PROGRAM_MAX as u64) * 4
            <= super::DUMP - super::MAILBOX_BASE
    );
    const _: () =
        assert!(super::dump_offset(super::DUMP_MAX_ROWS, 0) - super::MAILBOX_BASE <= STRIDE);
}

/// Values written to [`STATUS`].
///
/// Distinctive constants rather than small integers, so that a zero, a stale
/// value, or a misaddressed read is never mistaken for a real status. L1 contents
/// are undefined before firmware runs.
pub mod status {
    /// Firmware reached `_start` and finished its prologue.
    pub const RUNNING: u32 = 0x5747_0001;
    /// Firmware completed its work; [`super::RESULT`] is valid.
    pub const DONE: u32 = 0x5747_0002;
    /// Firmware panicked; [`super::PANIC_CODE`] says where.
    ///
    /// `ebreak` halts the core and needs an external agent to resume it, so it
    /// cannot back Rust's panic path. Publishing a status word and spinning is what
    /// replaces it.
    pub const PANICKED: u32 = 0x5747_DEAD;
}

/// Identifies the panic site. Kept small and hand-assigned: formatting a message
/// on a core with no allocator and 4 KiB of stack is not worth the code size.
pub mod panic_code {
    /// `panic!` was reached in firmware code.
    pub const EXPLICIT: u32 = 1;
    /// A checked arithmetic operation or bounds check failed.
    pub const ARITHMETIC: u32 = 2;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mailbox_clears_every_default_reset_pc() {
        for core in crate::tensix::Core::ALL {
            assert!(
                (core.default_reset_pc() as u64) < MAILBOX_BASE,
                "{} starts at {:#x}, which collides with the mailbox",
                core.name(),
                core.default_reset_pc()
            );
        }
    }

    #[test]
    fn status_values_are_distinct_and_not_plausible_garbage() {
        let all = [status::RUNNING, status::DONE, status::PANICKED];
        for (i, a) in all.iter().enumerate() {
            assert_ne!(*a, 0, "zero is what uninitialised L1 may read as");
            assert_ne!(*a, u32::MAX, "all-ones is what an absent device reads as");
            for b in &all[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
