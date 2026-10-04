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

pub mod record;

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
/// Host -> mover: non-zero, record each list's entries and records through
/// the tile's timestamper (`crate::mailbox::trace`), which the host has
/// configured. Read at the start of every list, so the host may turn it on or
/// off between lists. Zero on the simulator, which does not model the event
/// stream (divergence row 54).
pub const TRACE: u64 = MAILBOX_BASE + 0x50;
/// Mover -> host: how many of the mover's NoC requests had to wait for room
/// under the in-flight cap (`crate::noc::niu::InFlight`), since the host last
/// zeroed it. Written only when one did, so the normal path never touches it.
pub const THROTTLE_STALLS: u64 = MAILBOX_BASE + 0x54;
/// Mover -> host: the tile-counter cycles those requests waited, in total
/// (wrapping).
pub const THROTTLE_CYCLES: u64 = MAILBOX_BASE + 0x58;
/// Host -> mover: the in-flight cap to use, 0 for
/// `crate::noc::niu::MAX_IN_FLIGHT`. Read at the start of every list. Lower
/// caps are for the gates that force the throttle.
pub const IN_FLIGHT_CAP: u64 = MAILBOX_BASE + 0x5C;
/// The in-flight cap a tile's mover starts with (`tt_kernels::dm::DataMover::
/// start`), against `MAX_IN_FLIGHT` (128) for 0: with every tile reading, eight
/// requests in flight each spread the NoC's service evenly between tiles, and
/// the card's reads rise from 425 to 467 GB/s at 64 KiB entries and to 496 at
/// 16 KiB, against 3-4% off one tile's small reads
/// (`docs/firmware-performance.md`, "What holds card reads").
pub const TILE_IN_FLIGHT_CAP: u32 = 8;
const _: () = assert!(IN_FLIGHT_CAP + 4 <= BARRIER_COUNTER);
/// The barrier counter [`op::BARRIER`] increments, in the coordinating tile's
/// mover mailbox; zeroed by the host before the session's first barrier.
pub const BARRIER_COUNTER: u64 = MAILBOX_BASE + 0x60;
/// Where a mover's barrier polls land in its own L1: the same offset modulo
/// 16 as the counter, as a NoC read between L1s requires.
pub const BARRIER_POLL: u64 = MAILBOX_BASE + 0x70;
/// Where the atomic increment's old value lands.
pub const BARRIER_RET: u64 = MAILBOX_BASE + 0x80;
const _: () = assert!(BARRIER_COUNTER % 16 == BARRIER_POLL % 16);

/// The mover's list queue (`hardware-coverage.md` X4a): the host writes a
/// list into the ring of entries at [`LIST`] (never across its end), a slot
/// `(first entry, entries)` into [`QUEUE_SLOTS`], and then bumps
/// [`QUEUE_HEAD`]; the mover runs queued lists in order, bumping
/// [`QUEUE_DONE`] after each. So the host enqueues and goes on, and waits only
/// when it needs a result. A list that fails stops the queue:
/// [`QUEUE_ERROR`] says why and [`QUEUE_ERROR_AT`] which list, and nothing
/// more runs until the host restarts the mover.
pub const QUEUE_HEAD: u64 = MAILBOX_BASE + 0x90;
/// Lists run so far (wrapping).
pub const QUEUE_DONE: u64 = MAILBOX_BASE + 0x94;
/// The first failed list's `tt_isa::dm::error` code, or [`error::NONE`].
pub const QUEUE_ERROR: u64 = MAILBOX_BASE + 0x98;
/// The failed list's number: `QUEUE_DONE + 1` when it failed.
pub const QUEUE_ERROR_AT: u64 = MAILBOX_BASE + 0x9C;
/// The slots: list `n` is in slot `n % QUEUE_LEN`, as `first | entries << 16`.
pub const QUEUE_SLOTS: u64 = MAILBOX_BASE + 0xA0;
/// Slots in the queue.
pub const QUEUE_LEN: u32 = 16;
const _: () = assert!(QUEUE_SLOTS + QUEUE_LEN as u64 * 4 <= WRITE_NOC);

/// Host -> mover: which NIU the mover's GDDR writes go out on, [`write_noc`].
/// Read at the start of every list, by NC's image only: B writes on NoC #0. Reads always go out on NoC #0: on NoC #1
/// their data climbs the DRAM columns and shares one row's link, a single
/// link's bandwidth for the whole row (`silicon_bench_memory::
/// gddr_aggregate_nocs`, 4 tiles: 86 GB/s against 320). Barriers stay on
/// NoC #0. Each request's port is the entry's mapped onto the ports its NIU
/// owns (`crate::dram::DramChannel::port_for`), so no endpoint ever sees both
/// NoCs.
pub const WRITE_NOC: u64 = MAILBOX_BASE + 0xE0;
const _: () = assert!(WRITE_NOC + 4 <= MAILBOX_BASE + 0x100);

/// [`WRITE_NOC`]'s values. Anything else is [`write_noc::NOC0`].
pub mod write_noc {
    /// Writes through NoC #0's NIU, with the reads: what the mover always did.
    pub const NOC0: u32 = 0;
    /// Writes through NoC #1's NIU, on GDDR port 1 of every channel. Both
    /// NIUs translate coordinates the same way (`NoC/Coordinates.md`,
    /// "Coordinate Translation"), so the entries are unchanged.
    pub const NOC1: u32 = 1;
    /// Write entries alternate between the two NIUs, NoC #0 first in each
    /// list, each through a port its NIU owns. The two NoCs' write paths into
    /// GDDR share no links -- NoC #0's run down the two DRAM columns, NoC
    /// #1's along port 1's rows into them -- so their bandwidths can add.
    pub const ALTERNATE: u32 = 2;
}

/// A queue slot's word: a list of `entries` from ring entry `first`.
pub const fn queue_slot(first: u32, entries: u32) -> u32 {
    first | (entries << 16)
}

pub mod op {
    /// Shared packet: `[PAIR, reader_entries, writer_entries, capacity,
    /// barrier_trailers, 0, 0, 0]`, then reader, writer and optional barrier.
    pub const PAIR: u32 = 0x40;
    /// Retained region: `[PAIR_CALL, read_ch, read_off, read_count,
    /// write_ch, write_off, write_count, capacity]`. Only B starts the pair.
    pub const PAIR_CALL: u32 = 0x41;
    /// Buffer action: `[CB, stream, action, capacity, 0, 0, 0, 0]`, `stream`
    /// 0 for the input credits, 1 for the output's, 2 for a transfer's
    /// ([`crate::dataflow::Stream`]).
    pub const CB: u32 = 0x42;
    /// NC descriptor: run immutable writer entries from B's list ring.
    pub const SHARED: u32 = 0x43;
    /// B waits on a counter: `[RELEASED, target_u16, which, 0, ...]`
    /// ([`crate::dataflow::Release`]). `which` 0 waits for `target` output
    /// batches written out and released by NC; 1 for `target` output batches
    /// packed by T2 (their pack retired, whatever NC is still writing); 2 for
    /// `target` transfer batches written out and released by NC.
    pub const RELEASED: u32 = 0x44;
    /// DRAM -> L1.
    pub const READ: u32 = 1;
    /// Copy aligned words inside the data arena: [COPY_WORDS, src, dst,
    /// count, src_stride, dst_stride, 0, 0]. Strides are in bytes. B waits
    /// for preceding reads before copying; this performs no arithmetic.
    pub const COPY_WORDS: u32 = 0x45;
    /// L1 -> DRAM.
    pub const WRITE: u32 = 2;
    /// Run [`super::LEN`] list entries ([`super::Entry`]) from [`super::LIST`],
    /// in order. The other descriptor words are unused.
    pub const LIST: u32 = 3;
    /// As [`READ`], of exactly one tile slot ([`super::TILE_SLOT`]), transposing
    /// the tile on the way: the slot lands in [`super::SCRATCH`] and the mover
    /// writes its transpose to the destination. Only in a list entry.
    pub const READ_TRANSPOSED: u32 = 4;
    /// Set a tile's padding in L1: `[FILL, value, param, dst, 0, 0, 0, 0]` --
    /// `value` stored at every datum of the slot `dst` outside its first
    /// `param & 0xff` rows and `param >> 8` columns (each `0` meaning 32;
    /// [`super::fill::param`]). Stores only: the mover moves data and does no
    /// arithmetic -- that is the SFPU's and the Matrix Unit's. Only in a list
    /// entry.
    pub const FILL: u32 = 5;
    /// Run the tile's resident roles once: `[KERNEL, generation, a0, l0, a1,
    /// l1, a2, l2]`. The mover waits for every move before it; then, for each
    /// role `t` whose `at` is non-zero, writes `at` and `lt` to its mailbox's
    /// `PROGRAM_ADDR` and `PROGRAM_LEN` -- a program resident in the program
    /// cache (`crate::l1::PROGRAM_CACHE`) -- and then `generation` to all three
    /// `GENERATION`s (`crate::mailbox::role`), and waits until each has
    /// acknowledged it, or reports [`super::error::ROLE`] if one panics. With
    /// every `at` zero the roles run what the host staged. So a whole op, of
    /// any number of block shapes, is one list and one host round trip. Only in
    /// a list entry.
    pub const KERNEL: u32 = 6;
    /// Wait for every move before it to complete: the boundary between what
    /// were separate lists, whose entries may reuse each other's L1 slots.
    /// Only in a list entry.
    pub const WAIT: u32 = 7;
    /// As [`READ_TRANSPOSED`], but the mover writes the tile with its column
    /// 0 copied into every column: datum `(r, c)` from `(r, 0)`. A `[rows, 1]`
    /// tensor so read is a whole tile to broadcast across a `[rows, cols]`
    /// one, element for element. Only in a list entry.
    pub const READ_BROADCAST_COL: u32 = 8;
    /// Wait until every unit of a session has reached this point:
    /// `[BARRIER, target, x, y, 0, 0, 0, 0]`. The mover waits for its own
    /// moves, adds one to the counter at [`super::BARRIER_COUNTER`] in the L1
    /// of the coordinating tile `(x, y)` by a NoC atomic increment, and polls
    /// that counter over the NoC until it reaches `target` (compared modulo
    /// 2^32, so the counter may wrap). With `n` units, the `k`-th barrier's
    /// target is `k * n`. Only in a list entry.
    pub const BARRIER: u32 = 9;
    /// Run a list held in GDDR -- a trace (`tt_kernels::trace`):
    /// `[CALL, channel, offset, count, generation_base, barrier_base, 0, 0]`.
    /// The mover reads the `count` entries from `offset` of `channel` in
    /// chunks of [`super::TRACE_CHUNK_ENTRIES`] into [`super::TRACE_CHUNK`]
    /// and runs them as a list's, except that each `KERNEL`'s generation is
    /// offset by `generation_base` and each `BARRIER`'s target by
    /// `barrier_base`, the values a replay's run gives them. A record never
    /// spans a chunk (the capture pads with `WAIT`s), and a `CALL` inside a
    /// call is refused. Only in a list entry.
    pub const CALL: u32 = 10;
    /// Write one word of a role's mailbox: `[POKE, address, value, 0, ...]`,
    /// the address a word of `crate::mailbox::role`'s three mailboxes. A
    /// trace's role descriptors, which the host writes for an ordinary list
    /// and cannot write during a replay. Only in a list entry.
    pub const POKE: u32 = 11;
    // 12 and 13 were `SIGNAL` and `WAIT_PEER`, the two movers' direct
    // handshake. Nothing emits them since a shared packet's credits carry
    // that ordering, and the decode refuses them.
    /// The first half of [`KERNEL`]: `[LAUNCH, generation, a0, l0, a1, l1,
    /// a2, l2]`. The mover waits for every move before it, points the roles
    /// at their programs and posts `generation`, as `KERNEL` does, and goes on
    /// to the next entry without waiting for the roles -- so it can move the
    /// next block in and the last one out while they compute (checklist
    /// 9.15). A [`KERNEL_WAIT`] collects it. Only in a list entry.
    pub const LAUNCH: u32 = 14;
    /// The second half of [`KERNEL`]: `[KERNEL_WAIT, generation, 0, ...]`.
    /// Waits until each role has acknowledged `generation`, or reports
    /// [`super::error::ROLE`] if one panics. `generation` must be the last one
    /// the mover launched: anything else is [`super::error::GENERATION`], not
    /// a wait that could never end. Only in a list entry.
    pub const KERNEL_WAIT: u32 = 15;
    /// Host memory -> L1, the card's own DMA: `[HOST_READ, host_lo,
    /// host_hi, 0, l1, len, 0, 0]`. `host` is a NoC address at the
    /// host-connected PCIe tile ([`crate::noc::niu::PCIE_HOST`]): the driver's
    /// for memory it pinned for the card (`PIN_PAGES` with `NOC_DMA`). Only
    /// the PCIe tile's two plain windows to the host are accepted
    /// ([`super::host_window`]); `host` and `l1` congruent mod 64, as a GDDR
    /// read's, and a move never crosses a 4 GiB boundary of `host`. Through
    /// NoC #0. Only in a list entry.
    pub const HOST_READ: u32 = 0x20;
    /// L1 -> host memory: `[HOST_WRITE, host_lo, host_hi, 0, l1, len, 0, 0]`,
    /// as [`HOST_READ`].
    pub const HOST_WRITE: u32 = 0x21;
    /// A row-major block into a tile slot, in L1: `[TILIZE, src, stride,
    /// dst, valid, 0, 0, 0]`, `stride` at least the valid columns' bytes. Datum `(r, c)` of the 32x32 block whose first
    /// datum is at `src`, its rows `stride` bytes apart, goes to its place in
    /// the tile's faces at `dst` (a slot's datums, past its header); datums
    /// outside the block's first `valid & 0xff` rows and `valid >> 8` columns
    /// ([`super::fill::param`], `0` meaning 32) are zero, as a host tilize
    /// pads. Waits for every move before it. Only in a list entry.
    pub const TILIZE: u32 = 0x22;
    /// A tile slot's datums into a row-major block, in L1: `[UNTILIZE, src,
    /// stride, dst, valid, 0, 0, 0]`, [`TILIZE`]'s inverse -- only the valid
    /// rows and columns are written, so a band of blocks side by side keeps
    /// its neighbours'. Waits for every move before it. Only in a list entry.
    pub const UNTILIZE: u32 = 0x23;
}

// Entry ops and record ops (`record`, from 0x10) share one numbering.
const _: () = {
    let ops = [op::HOST_READ, op::HOST_WRITE, op::TILIZE, op::UNTILIZE];
    let mut i = 0;
    while i < ops.len() {
        assert!(!record::is_record(ops[i]));
        assert!(ops[i] < record::GATHER || ops[i] > record::PAD_WRITE);
        i += 1;
    }
};

/// Whether `host` is in one of the PCIe tile's two windows a mover may use
/// (`PCIExpressTile/README.md`, "NoC to Host"): `0x0...` (to the host's
/// IOMMU) and `0x1000_0000_0000_0000` (through the outbound iATU, where the
/// driver's pins land). Its other windows reach the PCIe controller's own
/// configuration and serdeses, which a stray write would corrupt.
pub const fn host_window(host: u64) -> bool {
    let window = host >> 58;
    window == 0 || window == 4
}

/// [`op::FILL`]'s parameter.
pub mod fill {
    /// The parameter for a tile whose first `rows` rows and `cols` columns
    /// are data, each 1..=32.
    pub const fn param(rows: u32, cols: u32) -> u32 {
        (rows % 32) | (cols % 32) << 8
    }

    /// The valid rows (or columns) a parameter field names: `0` is 32.
    pub const fn extent(field: u32) -> u32 {
        if field == 0 {
            32
        } else {
            field
        }
    }
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
/// Where an [`op::CALL`] streams its entries, a chunk at a time: in the free
/// L1 between the scratch slot and the data arena.
pub const TRACE_CHUNK: u64 = 0x1_9100;
/// Entries one chunk holds.
pub const TRACE_CHUNK_ENTRIES: u32 = 64;
const _: () = assert!(SCRATCH + TILE_SLOT <= 0x2_0000);

/// One tile's data mover, by core: where its image, mailbox, list ring,
/// scratch and trace chunk live. B's are this module's constants; NC's are
/// [`nc`]'s. Both speak the same protocol, so NC's mailbox is B's layout
/// moved ([`Mover::at`]).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Mover {
    pub core: crate::tensix::Core,
    pub image_base: u64,
    pub image_max: u64,
    pub mailbox: u64,
    pub list: u64,
    pub scratch: u64,
    pub trace_chunk: u64,
}

impl Mover {
    /// Whether this mover runs the entry or record whose first word is `op`.
    /// The direction table both sides use: the host refuses what the firmware
    /// would, and each image compiles in only what it runs.
    ///
    /// RISCV B is the reader: it never writes GDDR -- [`op::WRITE`] and the
    /// writing records ([`record::Direction::Write`]) are its
    /// [`error::DIRECTION`] -- and it keeps everything else: the control
    /// entries, the roles' launches, barriers, PCIe moves ([`op::HOST_WRITE`]
    /// included), and the reading records. RISCV NC is the writer: only
    /// [`op::WRITE`], the writing records, [`op::WAIT`] and the credits
    /// ([`op::CB`]). The packet entries ([`op::PAIR`], [`op::PAIR_CALL`],
    /// [`op::RELEASED`]) and [`op::CALL`] are B's; NC is started by B's
    /// packet, through its `SHARED` and `CALL` words, which are not entries.
    pub const fn permits(self, op: u32) -> bool {
        match self.core {
            crate::tensix::Core::B => {
                op != op::WRITE && !matches!(record::direction(op), Some(record::Direction::Write))
            }
            _ => {
                op == op::WRITE
                    || op == op::WAIT
                    || op == op::CB
                    || matches!(record::direction(op), Some(record::Direction::Write))
            }
        }
    }

    /// RISCV B's mover: the image at its hardwired reset PC.
    pub const B: Mover = Mover {
        core: crate::tensix::Core::B,
        image_base: IMAGE_BASE,
        image_max: IMAGE_MAX,
        mailbox: MAILBOX_BASE,
        list: LIST,
        scratch: SCRATCH,
        trace_chunk: TRACE_CHUNK,
    };
    /// RISCV NC's mover: the image at the top of L1, reached by the stub at
    /// NC's reset PC ([`nc::stub`]).
    pub const NC: Mover = Mover {
        core: crate::tensix::Core::NC,
        image_base: nc::IMAGE_BASE,
        image_max: nc::IMAGE_MAX,
        mailbox: nc::MAILBOX_BASE,
        list: nc::LIST,
        scratch: nc::SCRATCH,
        trace_chunk: nc::TRACE_CHUNK,
    };

    /// This mover's copy of mailbox word `b_word`, given as B's (one of
    /// [`SEQ`], [`DONE`], [`QUEUE_HEAD`], ...).
    pub const fn at(self, b_word: u64) -> u64 {
        b_word - MAILBOX_BASE + self.mailbox
    }
}

/// RISCV NC's mover: where its image, list ring, scratch and mailbox live.
///
/// NC's own slot, from its default reset PC (`0x1_2000`) to the mover's list
/// (`0x1_4000`), is 8 KiB, and the mover image is twice that. So the image
/// lives in the free L1 between B's mover area and the data arena
/// ([`crate::l1::NC_IMAGE`]), its list ring, scratch and trace chunk in the
/// mailbox region's free stretch below the role mailboxes -- the program
/// cache keeps all of the top of L1, which a long column sum needs
/// (`step21_one_launch`) -- and NC's reset PC holds a two-instruction jump to
/// the image ([`nc::stub`]). Its reset PC is never moved:
/// ttsim refuses NC's reset-PC override (divergence row 43), and the default
/// is the same on silicon. NC fetches only from L1 on Blackhole -- no
/// instruction RAM (`BabyRISCV/README.md:39`) -- so the image runs in place.
pub mod nc {
    /// Where NC starts on leaving reset: the stub.
    pub const STUB_AT: u64 = crate::tensix::Core::NC.default_reset_pc() as u64;
    /// Where the image is linked and loaded: the first 4 KiB boundary past
    /// B's mover area (the stub's jump needs one), up to the data arena.
    pub const IMAGE_BASE: u64 = 0x1_A000;
    pub const IMAGE_MAX: u64 = 0x2_0000 - IMAGE_BASE;
    const _: () = assert!(
        IMAGE_BASE >= super::TRACE_CHUNK + super::TRACE_CHUNK_ENTRIES as u64 * super::ENTRY_BYTES
    );
    /// NC's list ring, as [`super::LIST`] is B's: in the mailbox region, past
    /// the single-core mailbox's `Dst` dump (`crate::mailbox::DUMP`) and
    /// before the role mailboxes (`crate::mailbox::role::BASE`).
    pub const LIST: u64 = crate::mailbox::MAILBOX_BASE + 0x4000;
    const _: () = assert!(
        LIST >= crate::mailbox::DUMP
            + (crate::mailbox::DUMP_MAX_ROWS * crate::mailbox::DUMP_ROW_WORDS * 4) as u64
    );
    /// NC's transpose scratch slot, as [`super::SCRATCH`] is B's.
    pub const SCRATCH: u64 = LIST + super::LIST_MAX as u64 * super::ENTRY_BYTES;
    /// Where NC's `CALL`s stream their entries, as [`super::TRACE_CHUNK`].
    pub const TRACE_CHUNK: u64 = SCRATCH + super::TILE_SLOT.next_multiple_of(256);
    /// The end of what NC's mover uses.
    pub const END: u64 = TRACE_CHUNK + super::TRACE_CHUNK_ENTRIES as u64 * super::ENTRY_BYTES;
    /// NC's mover mailbox: B's layout ([`super::SEQ`] and on, offset by
    /// `MAILBOX_BASE - super::MAILBOX_BASE`), one page after B's.
    pub const MAILBOX_BASE: u64 = super::MAILBOX_BASE + 0x1000;
    const _: () = assert!(MAILBOX_BASE + 0x100 <= crate::mailbox::PROGRAM_REGION);
    const _: () = assert!(END <= crate::mailbox::role::BASE);
    const _: () = assert!(STUB_AT + 8 <= super::LIST);

    /// The two instructions at [`STUB_AT`]: `lui t0, %hi(target)` and
    /// `jalr x0, %lo(target)(t0)`, for a `target` that is a multiple of 4 KiB
    /// (so the low part is 0).
    pub const fn stub(target: u64) -> [u32; 2] {
        assert!(target % 0x1000 == 0 && target < 1 << 31);
        let lui_t0 = (target as u32) | (5 << 7) | 0x37;
        let jalr_x0_t0 = (5 << 15) | 0x67;
        [lui_t0, jalr_x0_t0]
    }
}

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
/// What the mover does to a tile it reads before anything else sees it.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Transform {
    /// Nothing: the bytes land where the descriptor says.
    None,
    /// [`op::READ_TRANSPOSED`].
    Transpose,
    /// [`op::READ_BROADCAST_COL`].
    BroadcastCol0,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Entry {
    CopyWords {
        src: u32,
        dst: u32,
        count: u32,
        src_stride: u32,
        dst_stride: u32,
    },
    Released {
        target: u16,
        which: crate::dataflow::Release,
    },
    Buffer {
        stream: crate::dataflow::Stream,
        action: crate::dataflow::Action,
        capacity: u16,
    },
    PairCall {
        reader: [u32; 3],
        writer: [u32; 3],
        capacity: u16,
    },
    Move {
        descriptor: Descriptor,
        transform: Transform,
    },
    /// [`op::FILL`].
    Fill {
        value: u32,
        /// The valid region ([`fill::param`]).
        param: u32,
        dst: u32,
    },
    /// [`op::KERNEL`]: point each role at its program, if named, post
    /// `generation`, and wait for it.
    Kernel {
        generation: u32,
        /// Per role, `(address, words)` of a resident program, or `(0, 0)`.
        programs: [(u32, u32); 3],
    },
    /// [`op::WAIT`].
    Wait,
    /// [`op::BARRIER`].
    Barrier { target: u32, x: u8, y: u8 },
    /// [`op::CALL`]: run `count` entries from GDDR.
    Call {
        channel: u32,
        offset: u32,
        count: u32,
        generation_base: u32,
        barrier_base: u32,
    },
    /// [`op::POKE`]: one role-mailbox word.
    Poke { address: u32, value: u32 },
    /// [`op::LAUNCH`]: [`Entry::Kernel`] without the wait.
    Launch {
        generation: u32,
        programs: [(u32, u32); 3],
    },
    /// [`op::KERNEL_WAIT`]: the wait.
    KernelWait { generation: u32 },
    /// [`op::HOST_READ`] and [`op::HOST_WRITE`]. The address in two words, not
    /// a `u64`: a `u64` would raise every `Entry`'s alignment to 8.
    Host {
        write: bool,
        host_lo: u32,
        host_hi: u32,
        l1: u32,
        len: u32,
    },
    /// [`op::TILIZE`] (`tilize`) and [`op::UNTILIZE`]: `block` is the
    /// row-major side, `slot` the tile's datums.
    Tilize {
        tilize: bool,
        block: u32,
        stride: u32,
        slot: u32,
        valid: u32,
    },
}

impl Entry {
    /// Decode entry words against the `usable` mask. A transposed read must be
    /// exactly one slot into a 16-aligned L1 slot inside L1.
    pub fn decode(usable: u32, w: [u32; 8]) -> Result<Self, u32> {
        // The hot path first: a plain read or write is most of every list
        // (`silicon_perf::mover_read_shapes` times it per entry).
        if w[0] == op::READ || w[0] == op::WRITE {
            return Descriptor::decode(usable, w[0], w[1], w[2], w[3], w[4], w[5]).map(
                |descriptor| Entry::Move {
                    descriptor,
                    transform: Transform::None,
                },
            );
        }
        if w[0] == op::COPY_WORDS {
            if w[3] == 0 || w[3] > 1024 || w[6] != 0 || w[7] != 0 {
                return Err(error::LENGTH);
            }
            for (at, stride) in [(w[1], w[4]), (w[2], w[5])] {
                let end = at as u64 + (w[3] - 1) as u64 * stride as u64 + 4;
                if at % 4 != 0
                    || stride % 4 != 0
                    || (at as u64) < crate::l1::DATA.base
                    || end > crate::l1::DATA.end
                {
                    return Err(error::ALIGNMENT);
                }
            }
            return Ok(Self::CopyWords {
                src: w[1],
                dst: w[2],
                count: w[3],
                src_stride: w[4],
                dst_stride: w[5],
            });
        }
        if w[0] == op::RELEASED {
            let which = crate::dataflow::Release::decode(w[2]).ok_or(error::OP)?;
            if w[1] > u16::MAX as u32 || w[3..].iter().any(|&word| word != 0) {
                return Err(error::OP);
            }
            return Ok(Entry::Released {
                target: w[1] as u16,
                which,
            });
        }
        if w[0] == op::CB {
            let action = crate::dataflow::Action::decode(w[2]).ok_or(error::OP)?;
            let stream = crate::dataflow::Stream::decode(w[1]).ok_or(error::OP)?;
            if w[3] == 0 || w[3] >= 0x8000 || w[4..].iter().any(|&word| word != 0) {
                return Err(error::OP);
            }
            return Ok(Entry::Buffer {
                stream,
                action,
                capacity: w[3] as u16,
            });
        }
        if w[0] == op::PAIR_CALL {
            if w[7] == 0 || w[7] >= 0x8000 {
                return Err(error::OP);
            }
            for stream in [&w[1..4], &w[4..7]] {
                let stream_end = stream[1] as u64 + stream[2] as u64 * ENTRY_BYTES;
                if stream[0] >= crate::dram::CHANNELS as u32
                    || usable & (1 << stream[0]) == 0
                    || stream[1] % ENTRY_BYTES as u32 != 0
                    || stream[2] == 0
                    || stream_end > crate::dram::CHANNEL_BYTES
                {
                    return Err(error::RANGE);
                }
            }
            return Ok(Entry::PairCall {
                reader: [w[1], w[2], w[3]],
                writer: [w[4], w[5], w[6]],
                capacity: w[7] as u16,
            });
        }
        let transform = match w[0] {
            op::READ_TRANSPOSED => Transform::Transpose,
            op::READ_BROADCAST_COL => Transform::BroadcastCol0,
            _ => Transform::None,
        };
        if transform != Transform::None {
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
                transform,
            });
        }
        if w[0] == op::LIST {
            return Err(error::OP);
        }
        if w[0] == op::KERNEL_WAIT {
            if w[1] == 0 || w[2..].iter().any(|&v| v != 0) {
                return Err(if w[1] == 0 {
                    error::GENERATION
                } else {
                    error::OP
                });
            }
            return Ok(Entry::KernelWait { generation: w[1] });
        }
        if w[0] == op::KERNEL || w[0] == op::LAUNCH {
            // Zero is what a resident runner reads as "not resident".
            if w[1] == 0 {
                return Err(error::GENERATION);
            }
            let cache = crate::l1::PROGRAM_CACHE;
            let mut programs = [(0, 0); 3];
            for (t, p) in programs.iter_mut().enumerate() {
                let (at, len) = (w[2 + 2 * t], w[3 + 2 * t]);
                if at == 0 && len == 0 {
                    continue;
                }
                // The length word may carry `mailbox::loops::LOOPED` (a loop
                // header leads the program), which the runner reads; the
                // bounds are the words'.
                let words = len & crate::dataflow::LENGTH_MASK;
                let bytes = words as u64 * 4;
                if at % 16 != 0
                    || words > crate::mailbox::PROGRAM_MAX
                    || !cache.contains(at as u64, bytes)
                {
                    return Err(error::PROGRAM);
                }
                *p = (at, len);
            }
            return Ok(if w[0] == op::LAUNCH {
                Entry::Launch {
                    generation: w[1],
                    programs,
                }
            } else {
                Entry::Kernel {
                    generation: w[1],
                    programs,
                }
            });
        }
        if w[0] == op::WAIT {
            return Ok(Entry::Wait);
        }
        if w[0] == op::BARRIER {
            if w[2] > 0x3f || w[3] > 0x3f || w[4..].iter().any(|&v| v != 0) {
                return Err(error::OP);
            }
            return Ok(Entry::Barrier {
                target: w[1],
                x: w[2] as u8,
                y: w[3] as u8,
            });
        }
        if w[0] == op::CALL {
            // Entries are 32 bytes, read 32-byte aligned; a channel's offset
            // fits the word (`crate::dram`).
            if w[3] == 0 || w[2] % ENTRY_BYTES as u32 != 0 || w[6] != 0 || w[7] != 0 {
                return Err(error::OP);
            }
            if w[1] >= crate::dram::CHANNELS as u32 || usable & (1 << w[1]) == 0 {
                return Err(error::RANGE);
            }
            return Ok(Entry::Call {
                channel: w[1],
                offset: w[2],
                count: w[3],
                generation_base: w[4],
                barrier_base: w[5],
            });
        }
        if w[0] == op::POKE {
            let base = crate::mailbox::role::BASE;
            let end = base + 3 * crate::mailbox::role::STRIDE;
            let ok = w[1] % 4 == 0
                && (w[1] as u64) >= base
                && (w[1] as u64) < end
                && (w[1] as u64 - base) % crate::mailbox::role::STRIDE
                    < crate::mailbox::MAILBOX_SIZE;
            if !ok || w[3..].iter().any(|&v| v != 0) {
                return Err(error::OP);
            }
            return Ok(Entry::Poke {
                address: w[1],
                value: w[2],
            });
        }
        if w[0] == op::HOST_READ || w[0] == op::HOST_WRITE {
            let host = (w[2] as u64) << 32 | w[1] as u64;
            let (l1, len) = (w[4], w[5]);
            if w[3] != 0 || w[6] != 0 || w[7] != 0 || !host_window(host) {
                return Err(error::OP);
            }
            if len == 0 {
                return Err(error::LENGTH);
            }
            let in_l1 = (l1 as u64 + len as u64) <= crate::tensix::L1_SIZE;
            let one_word = (w[1] as u64) + len as u64 <= 1 << 32;
            if host % crate::dram::ALIGN != l1 as u64 % crate::dram::ALIGN || !in_l1 || !one_word {
                return Err(error::ALIGNMENT);
            }
            return Ok(Entry::Host {
                write: w[0] == op::HOST_WRITE,
                host_lo: w[1],
                host_hi: w[2],
                l1,
                len,
            });
        }
        if w[0] == op::TILIZE || w[0] == op::UNTILIZE {
            let (block, stride, slot, valid) = (w[1], w[2], w[3], w[4]);
            if valid & !0x1f1f != 0 || w[5..].iter().any(|&v| v != 0) {
                return Err(error::OP);
            }
            let l1 = crate::tensix::L1_SIZE;
            // What it touches of the block: its valid rows' valid columns.
            let (rows, cols) = (
                fill::extent(valid & 0xff) as u64,
                fill::extent(valid >> 8) as u64,
            );
            let block_end = block as u64 + (rows - 1) * stride as u64 + 4 * cols;
            if block % 4 != 0
                || stride % 4 != 0
                || (stride as u64) < 4 * cols
                || slot % 16 != 0
                || block_end > l1
                || slot as u64 + 4096 > l1
            {
                return Err(error::ALIGNMENT);
            }
            return Ok(Entry::Tilize {
                tilize: w[0] == op::TILIZE,
                block,
                stride,
                slot,
                valid,
            });
        }
        if w[0] == op::FILL {
            let slot = |at: u32| at % 16 == 0 && at as u64 + TILE_SLOT <= crate::tensix::L1_SIZE;
            if !slot(w[3]) {
                return Err(error::ALIGNMENT);
            }
            // Only the region's bits, and nothing past them: a valid-row
            // count past 32 would walk past the tile.
            if w[2] & !0x1f1f != 0 || w[4..].iter().any(|&v| v != 0) {
                return Err(error::OP);
            }
            return Ok(Entry::Fill {
                value: w[1],
                param: w[2],
                dst: w[3],
            });
        }
        Descriptor::decode(usable, w[0], w[1], w[2], w[3], w[4], w[5]).map(|descriptor| {
            Entry::Move {
                descriptor,
                transform,
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
    /// A [`super::op::KERNEL`] entry naming a program outside the program
    /// cache, misaligned, or longer than a program may be.
    pub const PROGRAM: u32 = 8;
    /// The tile's other mover has stopped on an error or panicked, or is not
    /// idle when a packet needs it.
    pub const PEER: u32 = 9;
    /// An entry or record the mover's direction does not run: a GDDR write on
    /// RISCV B, which only reads (`Mover::permits`).
    pub const DIRECTION: u32 = 10;
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
    #[test]
    fn local_word_copies_are_checked_and_reader_owned() {
        use super::*;
        let at = crate::l1::DATA.base as u32;
        let entry = [op::COPY_WORDS, at, at + 4096, 16, 4, 4, 0, 0];
        assert!(matches!(
            Entry::decode(0xff, entry),
            Ok(Entry::CopyWords { count: 16, .. })
        ));
        assert!(Mover::B.permits(op::COPY_WORDS));
        assert!(!Mover::NC.permits(op::COPY_WORDS));
        for (index, value) in [
            (1, at + 1),
            (2, crate::l1::DATA.end as u32),
            (3, 0),
            (3, 1025),
            (4, 3),
            (5, u32::MAX),
            (6, 1),
        ] {
            let mut bad = entry;
            bad[index] = value;
            assert!(Entry::decode(0xff, bad).is_err());
        }
    }

    #[test]
    fn host_moves_decode_only_inside_the_host_windows() {
        use super::*;
        let iatu = 0x1000_0000_0000_0000u64;
        let e = |op, host: u64, l1: u32, len: u32| {
            Entry::decode(
                0xFF,
                [op, host as u32, (host >> 32) as u32, 0, l1, len, 0, 0],
            )
        };
        assert_eq!(
            e(op::HOST_READ, iatu + 0x40, 0x2_0040, 4096),
            Ok(Entry::Host {
                write: false,
                host_lo: 0x40,
                host_hi: 0x1000_0000,
                l1: 0x2_0040,
                len: 4096
            })
        );
        assert!(matches!(
            e(op::HOST_WRITE, 0x1_4000_0000, 0x2_0000, 64),
            Ok(Entry::Host { write: true, .. })
        ));
        // The PCIe controller's DBI, its serdes configuration, an unused
        // window: refused.
        for host in [
            0xF800_0000_0000_0000u64,
            0xFFFF_FFFF_E000_0000,
            0x2000_0000_0000_0000,
        ] {
            assert_eq!(
                e(op::HOST_WRITE, host, 0x2_0000, 64),
                Err(error::OP),
                "{host:#x}"
            );
        }
        assert_eq!(e(op::HOST_READ, iatu, 0x2_0000, 0), Err(error::LENGTH));
        assert_eq!(
            e(op::HOST_READ, iatu + 16, 0x2_0000, 64),
            Err(error::ALIGNMENT)
        );
        assert_eq!(
            e(op::HOST_READ, iatu, crate::tensix::L1_SIZE as u32 - 64, 128),
            Err(error::ALIGNMENT)
        );
        // Across a 4 GiB boundary of the host address.
        assert_eq!(
            e(op::HOST_READ, iatu + 0xFFFF_FFC0, 0x2_0000, 128),
            Err(error::ALIGNMENT)
        );
        let mut extra = [op::HOST_READ, 0, 0x1000_0000, 0, 0x2_0000, 64, 0, 0];
        extra[3] = 1;
        assert_eq!(Entry::decode(0xFF, extra), Err(error::OP));
    }

    #[test]
    fn launch_and_kernel_wait_decode_as_kernel_halves() {
        use super::*;
        let at = crate::l1::PROGRAM_CACHE.base as u32;
        let launch = [op::LAUNCH, 7, at, 4, 0, 0, at + 64, 8];
        assert_eq!(
            Entry::decode(0xFF, launch),
            Ok(Entry::Launch {
                generation: 7,
                programs: [(at, 4), (0, 0), (at + 64, 8)],
            })
        );
        assert_eq!(
            Entry::decode(0xFF, [op::KERNEL_WAIT, 7, 0, 0, 0, 0, 0, 0]),
            Ok(Entry::KernelWait { generation: 7 })
        );
        // Generation 0 is "not resident"; a wait carries nothing else.
        let mut zero = launch;
        zero[1] = 0;
        assert_eq!(Entry::decode(0xFF, zero), Err(error::GENERATION));
        assert_eq!(
            Entry::decode(0xFF, [op::KERNEL_WAIT, 0, 0, 0, 0, 0, 0, 0]),
            Err(error::GENERATION)
        );
        assert_eq!(
            Entry::decode(0xFF, [op::KERNEL_WAIT, 7, 1, 0, 0, 0, 0, 0]),
            Err(error::OP)
        );
        // A launch's programs are checked as a kernel's.
        let mut bad = launch;
        bad[2] = 8;
        assert_eq!(Entry::decode(0xFF, bad), Err(error::PROGRAM));
    }

    #[test]
    fn the_nc_stub_jumps_to_the_image() {
        // `lui t0, 0x170` and `jr t0`, as llvm-objdump decodes these words.
        assert_eq!(nc::stub(0x17_0000), [0x0017_02B7, 0x0002_8067]);
        assert_eq!(nc::stub(nc::IMAGE_BASE), [0x0001_A2B7, 0x0002_8067]);
    }

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
    fn b_reads_and_nc_writes() {
        use record::{GATHER, PAD_WRITE, READ_RUN, SCATTER, WRITE_RUN};
        let writes = [op::WRITE, SCATTER, WRITE_RUN, PAD_WRITE];
        let reads = [
            op::READ,
            op::READ_TRANSPOSED,
            op::READ_BROADCAST_COL,
            GATHER,
            READ_RUN,
            record::FILL_PAD,
        ];
        let control = [
            op::FILL,
            op::KERNEL,
            op::LAUNCH,
            op::KERNEL_WAIT,
            op::BARRIER,
            op::POKE,
            op::CALL,
            op::HOST_READ,
            op::HOST_WRITE,
            op::TILIZE,
            op::UNTILIZE,
            op::PAIR,
            op::PAIR_CALL,
            op::RELEASED,
        ];
        let shared = [op::WAIT, op::CB];
        for o in writes {
            assert!(!Mover::B.permits(o) && Mover::NC.permits(o), "{o:#x}");
        }
        for o in reads.into_iter().chain(control) {
            assert!(Mover::B.permits(o) && !Mover::NC.permits(o), "{o:#x}");
        }
        for o in shared {
            assert!(Mover::B.permits(o) && Mover::NC.permits(o), "{o:#x}");
        }
    }

    #[test]
    fn retired_peer_ops_are_refused() {
        // `SIGNAL` and `WAIT_PEER` (12 and 13) had their own entries once.
        for retired in [12, 13] {
            assert!(Entry::decode(ALL, [retired, 0, 1, 0, 0, 0, 0, 0]).is_err());
        }
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
            transform,
        } = e
        else {
            panic!("{e:?}")
        };
        assert_eq!(transform, Transform::Transpose);
        let b = Entry::decode(
            ALL,
            [
                op::READ_BROADCAST_COL,
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
        assert!(matches!(
            b,
            Entry::Move {
                transform: Transform::BroadcastCol0,
                ..
            }
        ));
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
                transform: Transform::None,
                ..
            }
        ));
        // A fill: a slot inside L1 and a region, nothing else.
        let fill = [
            op::FILL,
            0xff80_0000,
            fill::param(5, 32),
            0x2_0000,
            0,
            0,
            0,
            0,
        ];
        assert_eq!(
            Entry::decode(ALL, fill),
            Ok(Entry::Fill {
                value: 0xff80_0000,
                param: fill::param(5, 32),
                dst: 0x2_0000
            })
        );
        let mut bad = fill;
        bad[2] = 0x2000;
        assert_eq!(Entry::decode(ALL, bad), Err(error::OP), "a region past 32");
        let mut bad = fill;
        bad[3] = crate::tensix::L1_SIZE as u32 - 64;
        assert_eq!(Entry::decode(ALL, bad), Err(error::ALIGNMENT));
        let mut bad = fill;
        bad[5] = 1;
        assert_eq!(Entry::decode(ALL, bad), Err(error::OP), "a stray word");
        // No arithmetic: what was `COMPUTE`'s add is a fill's shape now,
        // and the old kinds' words are refused as one.
        assert_eq!(
            Entry::decode(ALL, [op::FILL, 1, 0, 0x2_0000, 0x2_1040, 0x2_2080, 0, 0]),
            Err(error::OP)
        );
        assert_eq!((fill::extent(0), fill::extent(7)), (32, 7));
        // A kernel entry names a non-zero generation; a wait takes nothing.
        assert_eq!(
            Entry::decode(ALL, [op::KERNEL, 7, 0, 0, 0, 0, 0, 0]),
            Ok(Entry::Kernel {
                generation: 7,
                programs: [(0, 0); 3]
            })
        );
        // Resident programs: inside the cache, aligned, no longer than a slot.
        let at = crate::l1::PROGRAM_CACHE.base as u32;
        assert_eq!(
            Entry::decode(ALL, [op::KERNEL, 7, at, 10, 0, 0, at + 64, 3]),
            Ok(Entry::Kernel {
                generation: 7,
                programs: [(at, 10), (0, 0), (at + 64, 3)]
            })
        );
        // A looped program's length word passes through, flag and all.
        let looped = 10 | crate::mailbox::loops::LOOPED;
        assert_eq!(
            Entry::decode(ALL, [op::KERNEL, 7, at, looped, 0, 0, 0, 0]),
            Ok(Entry::Kernel {
                generation: 7,
                programs: [(at, looped), (0, 0), (0, 0)]
            })
        );
        for bad in [
            [op::KERNEL, 7, 0x2_0000, 10, 0, 0, 0, 0],
            [op::KERNEL, 7, at + 4, 10, 0, 0, 0, 0],
            [
                op::KERNEL,
                7,
                at,
                crate::mailbox::PROGRAM_MAX + 1,
                0,
                0,
                0,
                0,
            ],
            [
                op::KERNEL,
                7,
                crate::tensix::L1_SIZE as u32 - 16,
                8,
                0,
                0,
                0,
                0,
            ],
        ] {
            assert_eq!(Entry::decode(ALL, bad), Err(error::PROGRAM), "{bad:x?}");
        }
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
