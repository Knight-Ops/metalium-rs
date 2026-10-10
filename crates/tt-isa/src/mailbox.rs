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
pub mod guard;

pub const MAILBOX_BASE: u64 = 0x0010_0000;

/// Offsets of the words the firmware runtime itself writes, from whichever base
/// the image was linked with: [`MAILBOX_BASE`] for Tensix images,
/// [`crate::eth::MAILBOX_BASE`] for the Ethernet one.
pub mod offset {
    pub const STATUS: u64 = 0x00;
    pub const HEARTBEAT: u64 = 0x04;
    pub const PANIC_CODE: u64 = 0x08;
    pub const RESULT: u64 = 0x0C;
}

/// Firmware liveness and status. See [`status`].
pub const STATUS: u64 = MAILBOX_BASE + offset::STATUS;
/// Monotonically incrementing counter, the heartbeat proper.
pub const HEARTBEAT: u64 = MAILBOX_BASE + offset::HEARTBEAT;
/// Set alongside [`status::PANICKED`] to say where.
pub const PANIC_CODE: u64 = MAILBOX_BASE + offset::PANIC_CODE;
/// Where a computation leaves its answer.
pub const RESULT: u64 = MAILBOX_BASE + offset::RESULT;
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

/// Non-zero: the firmware records its progress through the tile's timestamper
/// (`tensix::timestamper`), which the host has configured. Zero on the
/// simulator, whose timestamper support is probed separately.
pub const TRACE: u64 = MAILBOX_BASE + 0x30;

/// Non-zero `n`: after every `n` instruction words pushed, the firmware waits
/// for the coprocessor to retire everything it has pushed so far. Zero: it
/// never does.
///
/// Silicon needs no flow control -- a push into a full FIFO stalls the pushing
/// core until there is room (`PushTensixInstruction.md:11`). ttsim does not
/// model that stall: the push is fatal (`tensix_push_inst_fifo: pipe N inst
/// fifo full`, divergence row 55), and a thread blocked on another -- an
/// unpacker waiting for the Matrix Unit to hand a bank back -- reaches it as
/// soon as it gets more than a FIFO's depth ahead. The register the page offers
/// for watching the FIFO, `RISCV_DEBUG_REG_INSTRN_BUF_STATUS`, is refused by
/// ttsim too, so the window uses Manual TTSync, which it does model. The host
/// sets [`SIM_PUSH_WINDOW`] on the simulator and zero on silicon.
///
/// Waiting for the thread's own pushes to retire cannot deadlock a correct
/// schedule: every instruction already pushed depends only on instructions the
/// other threads push before theirs, in program order.
pub const PUSH_WINDOW: u64 = MAILBOX_BASE + 0x34;

/// Non-zero: the runner is **resident**. It runs the staged program, writes
/// this value to [`ACK`], and then, instead of stopping, waits for the host to
/// write a different non-zero value here, re-reads its descriptor, and runs
/// again. Zero at start: one run, then `DONE` and a spin, as before.
///
/// Phase 9: loading three images and releasing three cores per run cost more
/// than most runs. A resident runner is loaded once per session.
pub const GENERATION: u64 = MAILBOX_BASE + 0x38;
/// The last [`GENERATION`] the resident runner finished. The host waits on this
/// rather than on [`STATUS`], which a resident runner leaves at `DONE` between
/// runs.
pub const ACK: u64 = MAILBOX_BASE + 0x3C;

/// Zero: the runner pushes its program from its fixed slot ([`PROGRAM`], or
/// its role's). Non-zero: from this address -- a resident program in the
/// program cache (`crate::l1::PROGRAM_CACHE`), placed there by the host and
/// named, with [`PROGRAM_LEN`], by whoever starts the run: the host, or the
/// data mover's `KERNEL` entry (`crate::dm::op::KERNEL`). The runner refuses an
/// address outside the cache region.
pub const PROGRAM_ADDR: u64 = MAILBOX_BASE + 0x40;

/// The [`PUSH_WINDOW`] the host uses on the simulator: well under the 28
/// instructions the first frontend FIFO holds (`PushTensixInstruction.md:15`).
pub const SIM_PUSH_WINDOW: u32 = 16;

/// Non-zero: before pushing, the runner releases every Tensix semaphore a
/// thread may be blocked on, from the RISC-V side
/// (`crate::tensix::SEMAPHORE_ACCESS`): posting each that reads zero, round
/// after round. A thread left in `SEMWAIT` by a failed kernel survives the
/// backend reset (divergence row 65), and a reset program pushed behind it
/// would queue forever; a semaphore post by the RISC-V core does not queue
/// behind anything. Set by the host's tile reset on silicon only.
pub const UNWEDGE: u64 = MAILBOX_BASE + 0x44;

/// Non-zero: [`MOP_CFG`] holds this role's MOP Expander configuration, which
/// the runner loads before pushing (`crate::frontend::mop`). Zero: the
/// expander is left as it is.
pub const MOP_CFG_VALID: u64 = MAILBOX_BASE + 0x48;

/// The nine `MopCfg` words ([`crate::frontend::mop::MopConfig::config_words`]),
/// written to the thread's `TENSIX_MOP_CFG_BASE` once its expander is idle.
pub const MOP_CFG: u64 = MAILBOX_BASE + 0x4C;

/// Total size the firmware may assume is its own.
pub const MAILBOX_SIZE: u64 = 0x70;

/// A program's block repeats: with [`loops::LOOPED`] set in its length
/// ([`PROGRAM_LEN`], a `KERNEL` entry's), the program's first word is how
/// many entries follow ([`loops::entry`], at most [`loops::MAX`]), and the
/// code after them; the runner pushes each `[start, start + len)` of the code
/// `count` times where it is stored once -- how an SFPU row loop too long for
/// the replay buffer (`frontend::REPLAY_BUFFER`) runs without being unrolled
/// into the program slot. Metadata stored with the program, not instructions,
/// and not a descriptor word: it travels with the program through the
/// program cache and a `KERNEL` entry, so kernels with different loops queue
/// back to back under one descriptor.
pub mod loops {
    /// Set in a program length: a loop header leads the program.
    pub const LOOPED: u32 = 1 << 31;
    /// Entries a program may have.
    pub const MAX: usize = 4;
    /// Bits of each field: `start` in 0..13, `len` in 13..25, `count - 1` in
    /// 25..32.
    pub const START_BITS: u32 = 13;
    pub const LEN_BITS: u32 = 12;
    pub const COUNT_BITS: u32 = 7;

    /// `[start, start + len)` pushed `count` times: `None` if a field does not
    /// fit (`start < 8192`, `1 <= len < 4096`, `1 <= count <= 128`).
    pub const fn entry(start: u32, len: u32, count: u32) -> Option<u32> {
        if start >= 1 << START_BITS
            || len == 0
            || len >= 1 << LEN_BITS
            || count == 0
            || count > 1 << COUNT_BITS
        {
            return None;
        }
        Some(start | (len << START_BITS) | ((count - 1) << (START_BITS + LEN_BITS)))
    }

    /// `(start, len, count)` of an [`entry`].
    pub const fn decode(e: u32) -> (u32, u32, u32) {
        (
            e & ((1 << START_BITS) - 1),
            (e >> START_BITS) & ((1 << LEN_BITS) - 1),
            (e >> (START_BITS + LEN_BITS)) + 1,
        )
    }
}

/// Where the host points the timestamper's event buffer: after the program
/// slots, 1024 events. Sized for a profiled list (`trace`): the mover records
/// two events per list entry or record, never per expanded move, and each
/// role three per run, so a full list of records with a kernel apiece fits.
pub const TRACE_BUFFER: u64 = PROGRAM_REGION_END;
/// Bytes in [`TRACE_BUFFER`].
pub const TRACE_BUFFER_BYTES: u64 = 1024 * crate::tensix::timestamper::EVENT_BYTES;

/// What the firmware traces. A token is 29 bits: the event in bits 0..8, its
/// source in bits 8..12 (a Tensix thread's role runner, or the data mover),
/// and a detail in bits 12..29 (for the mover, the entry's op or record
/// kind). A role's tokens have no detail, so `token >> 8` is still its thread.
pub mod trace {
    /// The firmware has read its mailbox and is about to push.
    pub const START: u32 = 1;
    /// The last instruction word has been pushed.
    pub const PUSHED: u32 = 2;
    /// The coprocessor has retired the program.
    pub const RETIRED: u32 = 3;
    /// A resident runner saw a new generation, before reading its
    /// descriptor. Only on a resident run after the first.
    pub const WOKE: u32 = 4;
    /// A resident runner acknowledged its generation (`Dst` dumped, `DONE`
    /// about to be stored).
    pub const ACKED: u32 = 5;

    /// The data mover (`crate::dm`) began a list.
    pub const LIST_BEGIN: u32 = 16;
    /// The data mover finished a list, every move in it landed.
    pub const LIST_END: u32 = 17;
    /// The mover began a list entry or op record; the detail is its op
    /// (`crate::dm::op`) or record kind (`crate::dm::record`).
    pub const ENTRY_BEGIN: u32 = 18;
    /// The mover finished the entry or record it last began. Moves it issued
    /// may still be in flight: only a `KERNEL`, `WAIT` or `COMPUTE` entry, and
    /// the list's end, wait for them.
    pub const ENTRY_END: u32 = 19;
    /// The mover posted a `KERNEL`'s generation to the three roles; the
    /// detail is the generation.
    pub const KICK: u32 = 20;
    /// The mover saw all three roles acknowledge the generation it last
    /// posted.
    pub const ROLES_DONE: u32 = 21;
    /// Requests of the list just ended waited for room under the in-flight
    /// cap (`crate::noc::niu::InFlight`); the detail is the cycles they
    /// waited in all, saturated at [`DETAIL_MAX`]. One per list, after its
    /// `LIST_END`, and only when something waited.
    pub const THROTTLE: u32 = 22;

    /// The source of the data mover's events. Role runners use their Tensix
    /// thread, 0..3.
    pub const MOVER: u32 = 3;

    /// Bits a detail may use.
    pub const DETAIL_MAX: u32 = (1 << 17) - 1;

    /// The token for `event` on `thread`.
    pub const fn token(thread: u32, event: u32) -> u32 {
        (thread << 8) | event
    }

    /// The token for `event` from `source` with `detail` (masked to
    /// [`DETAIL_MAX`]).
    pub const fn token_with(source: u32, event: u32, detail: u32) -> u32 {
        ((detail & DETAIL_MAX) << 12) | ((source & 0xf) << 8) | (event & 0xff)
    }

    /// `(source, event)` from a token.
    pub const fn split(token: u32) -> (u32, u32) {
        ((token >> 8) & 0xf, token & 0xff)
    }

    /// The detail of a token.
    pub const fn detail(token: u32) -> u32 {
        token >> 12
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn tokens_round_trip_and_fit_29_bits() {
            let t = token_with(MOVER, ENTRY_BEGIN, 0x12);
            assert_eq!(split(t), (MOVER, ENTRY_BEGIN));
            assert_eq!(detail(t), 0x12);
            assert!(token_with(0xf, 0xff, DETAIL_MAX) < 1 << 29);
            // A role's token is what it always was.
            assert_eq!(token(2, RETIRED), token_with(2, RETIRED, 0));
            assert_eq!(split(token(2, RETIRED)), (2, RETIRED));
        }
    }
}

/// Where Tensix instruction streams are staged by the host: one fixed slot per
/// mailbox, the single-core one first and then one per role.
///
/// The descriptor lives in the mailbox; the program itself does not, because it is
/// unbounded in a way the mailbox is not. Putting it here is what makes the corpus
/// firmware generic: the host encodes a program with `tt_isa::isa`, writes the
/// words, and the same image runs it. Adding an instruction to the corpus is then a
/// host-side test case rather than a firmware change.
///
/// The slots are fixed rather than named by an address the host writes, so the
/// firmware never pushes from wherever a stale or corrupt pointer says. They were
/// once 256 words inside each mailbox, which a 32x32 matmul outgrows several times
/// over: its flushed configuration alone is a hundred-odd.
pub const PROGRAM_REGION: u64 = 0x0012_0000;
/// Bytes in one program slot.
pub const PROGRAM_SLOT: u64 = 0x8000;
/// Slots: the single-core mailbox's, then one per Tensix thread.
pub const PROGRAM_SLOTS: u64 = 4;
/// End of the program region.
pub const PROGRAM_REGION_END: u64 = PROGRAM_REGION + PROGRAM_SLOTS * PROGRAM_SLOT;
/// The single-core mailbox's program.
pub const PROGRAM: u64 = PROGRAM_REGION;
/// Most instructions a program may hold.
pub const PROGRAM_MAX: u32 = (PROGRAM_SLOT / 4) as u32;

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
const _: () = assert!(PROGRAM_REGION_END <= crate::tensix::L1_SIZE);
const _: () = assert!(PROGRAM_REGION % 16 == 0 && PROGRAM_SLOT % 16 == 0);
const _: () =
    assert!(dump_offset(DUMP_MAX_ROWS - 1, DUMP_ROW_WORDS - 1) + 4 <= crate::tensix::L1_SIZE);
// The descriptor words have to stay inside the mailbox the firmware owns.
const _: () = assert!(DUMP_ROW_FIRST + 4 <= MAILBOX_BASE + MAILBOX_SIZE);
const _: () = assert!(TRACE + 4 <= MAILBOX_BASE + MAILBOX_SIZE);
const _: () = assert!(PUSH_WINDOW + 4 <= MAILBOX_BASE + MAILBOX_SIZE);
const _: () = assert!(ACK + 4 <= MAILBOX_BASE + MAILBOX_SIZE);
const _: () = assert!(PROGRAM_ADDR + 4 <= MAILBOX_BASE + MAILBOX_SIZE);
const _: () = assert!(UNWEDGE + 4 <= MOP_CFG_VALID);
const _: () = assert!(MOP_CFG + 9 * 4 <= MAILBOX_BASE + MAILBOX_SIZE);
const _: () = assert!(TRACE_BUFFER + TRACE_BUFFER_BYTES <= crate::tensix::L1_SIZE);
const _: () = assert!(TRACE_BUFFER % 16 == 0);

/// One mailbox per Tensix thread, for the three-role datapath.
///
/// tt-metal's LLK splits a kernel across the three Tensix threads -- thread 0
/// unpacks, thread 1 does math, thread 2 packs -- and so does the role harness,
/// because much of the coprocessor's state is per thread and some of it is tied
/// to a role by the hardware (`UNPACR` counts with thread 0's ADCs; see
/// `docs/learnings/ttsim-divergence.md` row 45). Each role image runs its own program and
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
        program: u64,
    }

    impl Mailbox {
        /// The single-core mailbox at [`super::MAILBOX_BASE`], which has the
        /// same layout.
        pub const fn single_core() -> Mailbox {
            Mailbox {
                base: super::MAILBOX_BASE,
                program: super::PROGRAM,
            }
        }
        pub const fn of(thread: u32) -> Mailbox {
            assert!(thread < 3, "there are three Tensix threads");
            Mailbox {
                base: BASE + thread as u64 * STRIDE,
                program: super::PROGRAM_REGION + (1 + thread as u64) * super::PROGRAM_SLOT,
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
        pub const fn unwedge(self) -> u64 {
            self.base + (super::UNWEDGE - super::MAILBOX_BASE)
        }
        pub const fn mop_cfg_valid(self) -> u64 {
            self.base + (super::MOP_CFG_VALID - super::MAILBOX_BASE)
        }
        /// Word `k` (0..9) of [`super::MOP_CFG`].
        pub const fn mop_cfg(self, k: u32) -> u64 {
            assert!(k < 9, "MopCfg is nine words");
            self.base + (super::MOP_CFG - super::MAILBOX_BASE) + 4 * k as u64
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
        pub const fn trace(self) -> u64 {
            self.base + (super::TRACE - super::MAILBOX_BASE)
        }
        pub const fn push_window(self) -> u64 {
            self.base + (super::PUSH_WINDOW - super::MAILBOX_BASE)
        }
        pub const fn generation(self) -> u64 {
            self.base + (super::GENERATION - super::MAILBOX_BASE)
        }
        pub const fn ack(self) -> u64 {
            self.base + (super::ACK - super::MAILBOX_BASE)
        }
        pub const fn program_addr(self) -> u64 {
            self.base + (super::PROGRAM_ADDR - super::MAILBOX_BASE)
        }
        /// This mailbox's program slot in [`super::PROGRAM_REGION`].
        pub const fn program(self) -> u64 {
            self.program
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
    // The role mailboxes end before the program region begins, and the last
    // role's slot is the region's last.
    const _: () = assert!(BASE + 3 * STRIDE <= super::PROGRAM_REGION);
    const _: () =
        assert!(Mailbox::of(2).program() + super::PROGRAM_SLOT == super::PROGRAM_REGION_END);
    const _: () =
        assert!(super::dump_offset(super::DUMP_MAX_ROWS, 0) - super::MAILBOX_BASE <= STRIDE);
}

/// Everything a runner reads from its mailbox at the start of a run, written
/// whole ([`Descriptor::writes`]) so no field is left as an earlier run left
/// it. L1 survives between processes on silicon, so a field one writer forgets
/// is whatever the last process put there: a stale [`PROGRAM_ADDR`] once
/// pointed a tile reset's runner into the program cache, and it hung. ttsim
/// starts every process from zeroed L1, which is why it hid there.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Descriptor {
    pub thread_index: u32,
    pub dst_access_fmt: u32,
    pub program_len: u32,
    pub dump_row_first: u32,
    pub dump_row_count: u32,
    pub trace: u32,
    pub push_window: u32,
    /// Zero: the fixed slot ([`PROGRAM_ADDR`]).
    pub program_addr: u32,
    /// [`UNWEDGE`].
    pub unwedge: u32,
    /// [`MOP_CFG`], if this run loads a MOP Expander configuration.
    pub mop_cfg: Option<[u32; 9]>,
}

/// Words a [`Descriptor`] writes.
pub const DESCRIPTOR_WORDS: usize = 19;

impl Descriptor {
    /// `(address, value)` for every field, in `mb`. The MOP words are written
    /// whether or not the run loads them, so two descriptors compare word by
    /// word (a queued kernel's configuration must not change under it).
    pub const fn writes(&self, mb: role::Mailbox) -> [(u64, u32); DESCRIPTOR_WORDS] {
        let (valid, cfg) = match self.mop_cfg {
            Some(c) => (1, c),
            None => (0, [0; 9]),
        };
        [
            (mb.thread_index(), self.thread_index),
            (mb.dst_access_fmt(), self.dst_access_fmt),
            (mb.program_len(), self.program_len),
            (mb.dump_row_first(), self.dump_row_first),
            (mb.dump_row_count(), self.dump_row_count),
            (mb.trace(), self.trace),
            (mb.push_window(), self.push_window),
            (mb.program_addr(), self.program_addr),
            (mb.unwedge(), self.unwedge),
            (mb.mop_cfg_valid(), valid),
            (mb.mop_cfg(0), cfg[0]),
            (mb.mop_cfg(1), cfg[1]),
            (mb.mop_cfg(2), cfg[2]),
            (mb.mop_cfg(3), cfg[3]),
            (mb.mop_cfg(4), cfg[4]),
            (mb.mop_cfg(5), cfg[5]),
            (mb.mop_cfg(6), cfg[6]),
            (mb.mop_cfg(7), cfg[7]),
            (mb.mop_cfg(8), cfg[8]),
        ]
    }
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
    fn every_mailbox_has_its_own_program_slot() {
        let all = [
            role::Mailbox::single_core(),
            role::Mailbox::of(0),
            role::Mailbox::of(1),
            role::Mailbox::of(2),
        ];
        for (i, a) in all.iter().enumerate() {
            let a_end = a.program() + PROGRAM_SLOT;
            assert!(a.program() >= PROGRAM_REGION && a_end <= PROGRAM_REGION_END);
            for b in &all[i + 1..] {
                assert!(a_end <= b.program() || b.program() + PROGRAM_SLOT <= a.program());
            }
            // Nothing else the host stages lands in a slot.
            assert!(a.dump_offset(DUMP_MAX_ROWS - 1, DUMP_ROW_WORDS - 1) < PROGRAM_REGION);
        }
    }

    /// A descriptor names every word the runner reads before pushing: the
    /// ones from [`THREAD_INDEX`] to [`PUSH_WINDOW`], [`PROGRAM_ADDR`],
    /// [`UNWEDGE`] and the MOP configuration.
    #[test]
    fn a_descriptor_writes_every_word_the_runner_reads() {
        let mb = role::Mailbox::single_core();
        let mut at = Descriptor::default().writes(mb).map(|w| w.0);
        at.sort_unstable();
        let mut want = [
            THREAD_INDEX,
            DST_ACCESS_FMT,
            PROGRAM_LEN,
            DUMP_ROW_COUNT,
            DUMP_ROW_FIRST,
            TRACE,
            PUSH_WINDOW,
            PROGRAM_ADDR,
            UNWEDGE,
            MOP_CFG_VALID,
            MOP_CFG,
            MOP_CFG + 4,
            MOP_CFG + 8,
            MOP_CFG + 12,
            MOP_CFG + 16,
            MOP_CFG + 20,
            MOP_CFG + 24,
            MOP_CFG + 28,
            MOP_CFG + 32,
        ];
        want.sort_unstable();
        assert_eq!(at, want);
    }

    #[test]
    fn loop_entries_round_trip_and_refuse_what_does_not_fit() {
        for (s, l, c) in [(0, 1, 1), (8191, 4095, 128), (17, 295, 32), (3, 300, 64)] {
            assert_eq!(loops::decode(loops::entry(s, l, c).unwrap()), (s, l, c));
        }
        for (s, l, c) in [
            (8192, 1, 1),
            (0, 0, 1),
            (0, 4096, 1),
            (0, 1, 0),
            (0, 1, 129),
        ] {
            assert_eq!(loops::entry(s, l, c), None, "{s} {l} {c}");
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
