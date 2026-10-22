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

pub mod encoded;
pub mod record;

/// The mover's mailbox: the slot after the three role mailboxes
/// ([`crate::mailbox::role`]), which ends exactly where the program region begins.
pub const MAILBOX_BASE: u64 = crate::mailbox::role::BASE + 3 * crate::mailbox::role::STRIDE;
const _: () = assert!(MAILBOX_BASE + 0x100 <= crate::mailbox::PROGRAM_REGION);
const _: () = assert!(MAILBOX_BASE % 16 == 0);

/// Where the image is linked and loaded: RISCV B has no reset-PC override and
/// always starts at L1 offset 0 (`tensix::Core::B`).
pub const IMAGE_BASE: u64 = crate::tensix::Core::B.default_reset_pc() as u64;
/// The image may not reach the first role image ([`crate::l1::roles`]): the
/// three role images are packed against NC's slot with reset-PC overrides, so
/// B's slot is everything below them rather than only what lies under T0's
/// default reset PC.
pub const IMAGE_MAX: u64 = crate::l1::roles::BASE[0] - IMAGE_BASE;

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

/// [`TRACE`]'s non-zero values.
pub mod trace_mode {
    /// Every list, entry and record, as the host has always asked for: two timestamper stores around each
    /// entry, which also keep a run of reads out of the tight loop (the cost of tracing: ~94 cycles an
    /// encoded read, `docs/learnings/performance-log.md`, "What bounds one tile's read rate").
    pub const FULL: u32 = 1;
    /// The list's own begin and end and every entry but a plain or encoded read (a barrier's release, a
    /// kernel, a wait): what a benchmark that times lists needs, with the reads running as they do untraced.
    pub const LISTS: u32 = 2;
    /// The outermost list's begin and end and only the entries that bound an op's phases: its control
    /// (`LAUNCH`, `KERNEL`, `CB`, `POKE`), its kernels' wait (`KERNEL_WAIT`) and its barrier (`BARRIER`).
    /// Data entries are not stamped (the span less the rest is their time), so an op costs ~8 events, not
    /// ~25, and reads keep their untraced fast path: the profile mode's default
    /// (`tt_kernels::profile`).
    pub const OPS: u32 = 3;

    /// What B stamps, in `OPS` mode, as an entry headed by this op around each of a
    /// trace's fetches from GDDR: a chunk of its stream, or of a `PAIR_CALL`'s reader section (the
    /// read and the wait for it). No entry is headed by `CALL` (a stream's `CALL` is refused), so
    /// the stamp is never a real entry's.
    pub const FETCH: u32 = super::op::CALL;
    /// What B stamps, in `OPS` mode, as an entry headed by this op around its wait for
    /// NC to finish a pair's writer section (its fetch and its writes). A `PAIR_CALL` entry itself
    /// is never stamped, so the stamp is never a real entry's.
    pub const WRITER_WAIT: u32 = super::op::PAIR_CALL;
    /// What B stamps, in `OPS` mode, as an entry headed by this op around a prefetch `WAIT`
    /// ([`super::WAIT_PREFETCH`]): issuing the queued prefetch's last reads and waiting for them to
    /// land. `OPS` stamps no `WAIT` as itself, so there the stamp is the prefetch's; `FULL` stamps
    /// every `WAIT`, and the host counts those as data.
    pub const PREFETCH_WAIT: u32 = super::op::WAIT;

    /// Is an entry headed by `op` stamped in `OPS` mode?
    pub const fn ops_stamped(op: u32) -> bool {
        matches!(
            op,
            super::op::BARRIER
                | super::op::KERNEL_WAIT
                | super::op::LAUNCH
                | super::op::KERNEL
                | super::op::CB
                | super::op::POKE
        )
    }
}
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
/// (`docs/learnings/performance-log.md`, "What holds card reads").
pub const TILE_IN_FLIGHT_CAP: u32 = 8;
/// The NoC transaction ID RISCV B's requests carry (`PACKET_TAG` bits 10..14),
/// and NC's: the mover's `TXN`.
pub const TXN_B: u8 = 2;
pub const TXN_NC: u8 = 4;
/// The NoC transaction ID of RISCV B's prefetch reads ([`record::GATHER_PREFETCH`]): apart from
/// [`TXN_B`], so the entries that drain B's moves -- a kernel's launch, a barrier, a chunk's end
/// -- leave them in flight. Only an [`op::WAIT`] with [`WAIT_PREFETCH`] drains them, which the
/// op that reads what they bring puts before its kernel (`docs/plans/decode-to-bandwidth.md`
/// Phase 8). Its cap is [`TILE_IN_FLIGHT_CAP`], set once when the mover starts.
pub const TXN_PREFETCH: u8 = 6;
/// The NoC transaction ID of RISCV NC's multicasts ([`op::WRITE_L1_MCAST`]): apart from
/// [`TXN_NC`], whose counter a broadcast would leave meaningless. Nothing waits on it: a multicast
/// is counted by its acknowledgements.
pub const TXN_NC_MCAST: u8 = 7;
/// The NoC transaction ID of RISCV B's trace fetches read ahead: the next [`op::PAIR_CALL`]'s first
/// reader chunk, into [`crate::dataflow::AHEAD_CHUNK`] while B runs the entries before the pair.
/// Apart from [`TXN_B`], whose drain at every chunk's end would wait for it. Only B waits on it,
/// before it runs that chunk and at a list's end: an ID's outstanding count is the tile's NIU's,
/// so NC waiting on it would wait for B's reads (`docs/learnings/silicon-operating-notes.md`,
/// "A transaction ID's outstanding count is the tile's"). Its cap is [`TILE_IN_FLIGHT_CAP`], set
/// once when the mover starts.
pub const TXN_FETCH: u8 = 8;
/// Word 6 of an [`op::WRITE_L1_MCAST`]: counted with the run it belongs to, not on its own.
pub const MCAST_RUN_ACKS: u32 = 1;

/// [`op::WRITE_L1_MCAST`]'s word 1 for the rectangle `start..=end`.
pub const fn mcast_corners(start: (u8, u8), end: (u8, u8)) -> u32 {
    u32::from_le_bytes([start.0, start.1, end.0, end.1])
}
/// [`op::WAIT`]'s word 1 for a wait that also drains the prefetch reads ([`TXN_PREFETCH`]).
pub const WAIT_PREFETCH: u32 = 1;
const _: () = assert!(IN_FLIGHT_CAP + 4 <= BARRIER_COUNTER);
/// The barrier counter [`op::BARRIER`] increments, in the coordinating tile's
/// mover mailbox; zeroed by the host before the session's first barrier.
pub const BARRIER_COUNTER: u64 = MAILBOX_BASE + 0x60;
/// Where a mover's barrier polls land in its own L1: the same offset modulo
/// 16 as the counter, as a NoC read between L1s requires.
pub const BARRIER_POLL: u64 = MAILBOX_BASE + 0x70;
/// Where the atomic increment's old value lands.
pub const BARRIER_RET: u64 = MAILBOX_BASE + 0x80;
/// The word of a wait-only `op::BARRIER` that says how many barriers it stands for: the
/// profile mode's window counts it as that many arrivals.
pub const BARRIER_STANDS_FOR: usize = 5;
const _: () = assert!(BARRIER_COUNTER % 16 == BARRIER_POLL % 16);
/// The word of an `op::BARRIER` that asks for the coordinator's release (1) rather than polling
/// its counter (0).
pub const BARRIER_RELEASE_MODE: usize = 6;
/// The last barrier target the coordinator released to this tile (`op::BARRIER` word 6): written
/// by the coordinator's NoC multicast, read by this tile's B in its own L1. Zeroed by the host
/// with the counter.
pub const BARRIER_RELEASE: u64 = MAILBOX_BASE + 0x100;
/// The coordinator's copy of the target it multicasts, the same offset modulo 16 as
/// [`BARRIER_RELEASE`], as a NoC write between L1s requires.
pub const BARRIER_RELEASE_STAGE: u64 = MAILBOX_BASE + 0x110;
const _: () = assert!(BARRIER_RELEASE % 16 == BARRIER_RELEASE_STAGE % 16);
/// Host -> the coordinator: the rectangles its release reaches, `[count, (hi, acks) x count]`,
/// `hi` a rectangle's `NOC_RET_ADDR_HI` on NoC #0 (`noc::multicast::Rect::hi`) and `acks` its
/// tile count. Every tile of every rectangle is a surviving Tensix tile (`Rect::new`), and none
/// is the coordinator's.
pub const BARRIER_RECTS: u64 = MAILBOX_BASE + 0x120;
/// Most rectangles at [`BARRIER_RECTS`]: `noc::multicast::full_grid_except`'s.
pub const BARRIER_RECTS_MAX: u32 = 8;
const _: () = assert!(BARRIER_RECTS + 4 + BARRIER_RECTS_MAX as u64 * 8 <= encoded::ENDPOINTS);

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
/// Completed trace chunks (wrapping), including nested retained streams.
/// Unlike queue completion, this advances during a long `CALL`. Only completed
/// work counts: a stalled read or kernel must not refresh the host watchdog.
pub const TRACE_PROGRESS: u64 = MAILBOX_BASE + 0xE4;
/// Host -> mover: non-zero, RISCV B issues its GDDR reads through the fast path
/// (X6): request initiator 1 of NoC #0 keeps every register whose value does not
/// change from one request to the next (the return coordinate, tag, control word
/// and, until they change, the target coordinate and length), so a request writes
/// only the address words, the words that changed and the command. Zero (the
/// default) is the path that writes every register of initiator 0 per request.
/// Read at the start of every list, so the host may switch between lists; NC
/// ignores it (its writes are not on this path).
pub const READ_FAST: u64 = MAILBOX_BASE + 0xE8;
const _: () = assert!(TRACE_PROGRESS + 4 <= READ_FAST);

/// Mover -> host: what this image was built to do ([`caps`]), written once as the
/// mover starts. Zero until it has: the host waits for [`caps::VALID`]. The host
/// reads it instead of assuming which firmware it loaded.
pub const CAPS: u64 = MAILBOX_BASE + 0xEC;
const _: () = assert!(READ_FAST + 4 <= CAPS);

/// Host -> mover: the key the checksum of a host-encoded read is bound to
/// ([`encoded::context_seed`]), written before the image starts and read once at boot.
/// A fresh random word each start, so an encoded entry left in L1 by an earlier run, or
/// made for another mover, fails the card's check. Ignored by an image without `fast-reads`.
pub const ENCODE_KEY: u64 = MAILBOX_BASE + 0xF0;
const _: () = assert!(CAPS + 4 <= ENCODE_KEY);
/// Host -> mover: which ops of a list are stamped while [`TRACE`] is on, `lo | hi << 16`: only the entries
/// run while `lo <= barriers < hi`, `barriers` counting the `BARRIER` entries the outermost list has run
/// so far (each op of a multi-unit list ends in one, so it is the op's index). `hi == 0` stamps every
/// entry. A replay too long for the stamp buffer is profiled a window at a time over successive replays
/// (`tt_kernels::profile`). Read at the start of each outermost list.
pub const TRACE_WINDOW: u64 = MAILBOX_BASE + 0xF4;
const _: () = assert!(ENCODE_KEY + 4 <= TRACE_WINDOW);
/// Host -> RISCV B: [`TXN_PREFETCH`]'s in-flight cap, applied as each list starts (nothing is then
/// in flight on it: the last list's end drained it); 0 is [`TILE_IN_FLIGHT_CAP`]. Prefetch reads
/// share GDDR with the ops they run under, whose latency-bound reads queue behind them
/// (`docs/plans/decode-to-bandwidth.md` Phase 8).
pub const PREFETCH_CAP: u64 = MAILBOX_BASE + 0xF8;
/// The [`PREFETCH_CAP`] a host sets unless told otherwise (`tt_kernels::dm::DataMover::start`,
/// `TT_PREFETCH_CAP`): two reads in flight, where busy-unit prefetch measured best on a 1.7B decode
/// step (18.40 ms against 18.73 at 4 and 19.08 at 8, `docs/plans/planned-l1-residency.md` 3b).
pub const PREFETCH_CAP_DEFAULT: u32 = 2;
/// Host -> RISCV B: the in-flight cap of a weight stream's reads (`record::GATHER_STREAM`),
/// read as each such record starts; 0 is [`TILE_IN_FLIGHT_CAP`].
pub const STREAM_CAP: u64 = MAILBOX_BASE + 0xFC;
const _: () = assert!(PREFETCH_CAP + 4 <= STREAM_CAP);
const _: () = assert!(STREAM_CAP + 4 <= MAILBOX_BASE + 0x100);
/// The [`STREAM_CAP`] a host sets unless told otherwise (`tt_kernels::dm::DataMover::start`,
/// `TT_STREAM_CAP`): two reads in flight. With every B read at 2, a 1.7B decode step measured
/// 17.27 / 18.91 ms against 18.28 / 19.84 at the tile cap of 8 (4: 17.80 / 19.32, 1: 18.13 /
/// 19.99), its products' units served evenly (`docs/plans/decode-to-bandwidth.md` Phase 8).
pub const STREAM_CAP_DEFAULT: u32 = 2;
const _: () = assert!(TRACE_WINDOW + 4 <= PREFETCH_CAP);
const _: () = assert!(PREFETCH_CAP + 4 <= MAILBOX_BASE + 0x100);

/// [`CAPS`]'s bits.
pub mod caps {
    /// The word has been written (every image sets it, so zero means "not yet").
    pub const VALID: u32 = 1 << 31;
    /// The image runs host-encoded reads ([`super::op::READ_ENCODED`]): built with the
    /// `fast-reads` feature. Without it the card decodes and checks every read
    /// itself and refuses `READ_ENCODED` as an unknown op.
    pub const FAST_READS: u32 = 1 << 0;
    /// With `FAST_READS`, the image also runs [`super::encoded::guard`] on each
    /// encoded read (the `encoded-guard` feature), not only its checksum.
    pub const GUARD: u32 = 1 << 1;
    /// The image stamps its work through the tile's timestamper when [`super::TRACE`] asks (the
    /// `profile` feature). Without it the stamping code is not in the image at all, so a run that
    /// does not profile pays nothing for it, and the host refuses to profile.
    pub const STAMPS: u32 = 1 << 2;
}

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
    /// Copy aligned 16-bit datums in the data arena, with byte strides.
    /// Same descriptor as COPY_WORDS. Used for bit-preserving BF16 views.
    pub const COPY_HALFWORDS: u32 = 0x46;
    /// BF16 tile padding fill, same descriptor/region convention as FILL.
    pub const FILL_HALFWORDS: u32 = 0x47;
    /// Validate a packed tile of canonical device status flags before NC
    /// writes a result: `[CHECK_FLAGS, slot, rows, cols, 0, ...]`.
    /// Only valid logical lanes are checked; zero reports DOMAIN.
    pub const CHECK_FLAGS: u32 = 0x48;
    /// B selects one datum per row using resident I32 indices. All three slots
    /// are in L1: `[INDEX_PICK, indices, source, output, bound, first, rows,
    /// datum_bytes]`. Source holds a 32-column chunk beginning at `first`.
    /// Indices outside `0..bound` report DOMAIN before any indexed access.
    pub const INDEX_PICK: u32 = 0x49;
    /// B issues one GDDR read the host already encoded ([`super::encoded`]):
    /// `[READ_ENCODED, TARG, TARG_HI, RET, RET_HI, CHECK, 0, LEN]`, at most
    /// `MAX_REQUEST_BYTES` of it. The card verifies `CHECK` (an integrity word the host's
    /// encoder computes over the others), not the move: the host validated that
    /// ([`super::encoded::validate`]). Only an image built with `fast-reads` runs it
    /// ([`super::caps::FAST_READS`]); B only.
    pub const READ_ENCODED: u32 = 0x4A;
    /// L1 -> DRAM.
    pub const WRITE: u32 = 2;
    /// L1 -> another Tensix tile's L1: `[WRITE_L1, x, y, to, from, len, 0, 0]`, [`WRITE`]'s
    /// shape with a tile for a channel and port. RISCV NC writes `len` bytes of its own tile's L1
    /// at `from` to `to` in the L1 of the tile at `(x, y)`, through the NIU its writes go out on
    /// ([`super::WRITE_NOC`]). `(x, y)` is the tile's coordinate as the session names it, which
    /// with translation on is the same on both NoCs (`step50_noc1`). Both ranges inside
    /// `crate::l1::DATA`, congruent mod 16; `len` at most `crate::noc::niu::MAX_REQUEST_BYTES`;
    /// `(x, y)` a coordinate a Tensix tile could have (`crate::noc::grid::is_tensix_geometry`),
    /// which is all the card can check: the host narrows it to the tiles of its session that read
    /// what is written, and every one of them exists (`docs/plans/planned-l1-residency.md`).
    /// Response-marked, so the drain at a list's end waits for it: a unit's writes have landed
    /// before it reports done and arrives at its barrier. NC's only. Only in a list entry.
    pub const WRITE_L1: u32 = 0x24;
    /// L1 -> the L1 of every tile of a rectangle: `[WRITE_L1_MCAST, corners, 0, to, from, len, 0,
    /// 0]`, [`WRITE_L1`] to each of the tiles `x0..=x1` by `y0..=y1`, `corners` `x0 | y0 << 8 |
    /// x1 << 16 | y1 << 24` ([`super::mcast_corners`]). One response-marked, path-reserved
    /// multicast (`crate::noc::multicast::registers_to`) through NoC #1, which NC's writes must go
    /// out on ([`super::WRITE_NOC`]; NoC #0 is [`super::error::DIRECTION`]): RISCV NC counts the
    /// rectangle's acknowledgements on that NIU's write-ack counter before it runs the next
    /// entry, having first waited for its earlier writes. A multicast's ID counter proves nothing
    /// (one up, one down a recipient), nothing else NC sends is then in flight to add to the
    /// count, and B, which shares the tile's NIUs, never writes through NoC #1 (NoC #0's counter
    /// is B's barrier release's). So its writes too have landed before NC reports done. Every tile of the
    /// rectangle must be one a Tensix tile could have (`crate::noc::grid::is_tensix_geometry`):
    /// no non-memory or DRAM column inside it, which a broadcast would otherwise rely on masks to
    /// skip; the tile running it is never one (no broadcast to itself); the ranges as
    /// [`WRITE_L1`]'s. The host narrows the rectangle to its session's tiles. NC's only. Only in
    /// a list entry.
    ///
    /// Word 6 may be [`super::MCAST_RUN_ACKS`]: a multicast so marked that follows another so
    /// marked neither waits for earlier writes nor counts its own acknowledgements before the
    /// next entry; the run's acknowledgements are counted together before the first entry that is
    /// not one of it, and at the list's end (`docs/plans/planned-l1-residency.md` 2f (c)). The
    /// run's first still waits for NC's earlier writes, so nothing else is in flight to add to the
    /// count.
    pub const WRITE_L1_MCAST: u32 = 0x25;
    /// Another (or this) Tensix tile's L1 -> this tile's L1: `[READ_L1, x, y, from, to, len, 0,
    /// 0]`, [`WRITE_L1`]'s mirror on the reader. RISCV B reads `len` bytes at `from` in the L1 of
    /// the tile at `(x, y)` -- this tile's own, a loopback -- into its own L1 at `to`, one request
    /// through NoC #0 under its read ID, so what drains its reads (a kernel's launch, a list's
    /// end) drains it. The checks are [`WRITE_L1`]'s; the host narrows `(x, y)` to its session's
    /// tiles and `from` to the window it keeps (`docs/plans/planned-l1-residency.md`). B's only.
    /// Only in a list entry.
    pub const READ_L1: u32 = 0x26;
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
    /// `[WAIT, prefetch, 0, ...]`: `prefetch` [`super::WAIT_PREFETCH`] also waits for the
    /// prefetch reads ([`super::TXN_PREFETCH`]), 0 does not; anything else is refused.
    /// Only in a list entry.
    pub const WAIT: u32 = 7;
    /// As [`READ_TRANSPOSED`], but the mover writes the tile with its column
    /// 0 copied into every column: datum `(r, c)` from `(r, 0)`. A `[rows, 1]`
    /// tensor so read is a whole tile to broadcast across a `[rows, cols]`
    /// one, element for element. Only in a list entry.
    pub const READ_BROADCAST_COL: u32 = 8;
    /// Wait until the units an op ran on have reached this point:
    /// `[BARRIER, target, x, y, wait_only, 0, 0, 0]`. The mover waits for its
    /// own moves, adds one to the counter at [`super::BARRIER_COUNTER`] in the
    /// L1 of the coordinating tile `(x, y)` by a NoC atomic increment, and
    /// polls that counter over the NoC until it reaches `target` (compared
    /// modulo 2^32, so the counter may wrap). The counter counts arrivals, so
    /// a barrier's target is every arrival before it plus its own units'.
    ///
    /// `wait_only` 1: poll without adding one. A unit that ran none of the
    /// ops since it last passed a barrier waits this way for the latest
    /// target before its next list, which orders it after those ops' writes
    /// without making it arrive at barriers it had no part in
    /// (`tt_kernels::session`). Only 0 or 1. Only in a list entry.
    ///
    /// Word 5 ([`super::BARRIER_STANDS_FOR`]): for a wait, the barriers it stands for -- in a
    /// trace, the captured ops' barriers the unit sat out -- which the profile mode's window
    /// (`TRACE_WINDOW`) counts as an arrival counts one, so every unit's count is the op's index.
    /// Zero for an arrival, at most `0xffff`.
    ///
    /// Word 6 ([`super::BARRIER_RELEASE_MODE`]), 0 or 1: 1 waits for the coordinator's release,
    /// not by polling its counter. The coordinator (the mover whose tile is `(x, y)`) waits on
    /// its own counter, then writes `target` into [`super::BARRIER_RELEASE`] of every tile of the
    /// rectangles at [`super::BARRIER_RECTS`] by NoC multicast; every other mover waits on its own
    /// [`super::BARRIER_RELEASE`], in its own L1, with no NoC traffic. The host sets it only where
    /// the coordinator reaches the target too (it arrives at that barrier, or at a later one).
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
    let ops = [
        op::HOST_READ,
        op::HOST_WRITE,
        op::TILIZE,
        op::UNTILIZE,
        op::WRITE_L1,
        op::WRITE_L1_MCAST,
        op::READ_L1,
    ];
    let mut i = 0;
    while i < ops.len() {
        assert!(!record::is_record(ops[i]));
        assert!(ops[i] < record::GATHER || ops[i] > record::SCATTER_ROWS);
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

/// A word a mover may write with [`op::POKE`] or [`record::POKE_RUN`]: aligned, and
/// inside one of the three role mailboxes. Shared so the host's check, the entry decode
/// and the firmware's run of a record agree on what is pokeable.
pub const fn is_role_mailbox_word(address: u32) -> bool {
    let base = crate::mailbox::role::BASE;
    let end = base + 3 * crate::mailbox::role::STRIDE;
    address % 4 == 0
        && (address as u64) >= base
        && (address as u64) < end
        && (address as u64 - base) % crate::mailbox::role::STRIDE < crate::mailbox::MAILBOX_SIZE
}

impl Mover {
    /// Whether this mover runs the entry or record whose first word is `op`.
    /// The direction table both sides use: the host refuses what the firmware
    /// would, and each image compiles in only what it runs.
    ///
    /// RISCV B is the reader: it never writes GDDR -- [`op::WRITE`] and the
    /// writing records ([`record::Direction::Write`]) are its
    /// [`error::DIRECTION`], as is [`op::WRITE_L1`] -- and it keeps everything else: the control
    /// entries, the roles' launches, barriers, PCIe moves ([`op::HOST_WRITE`]
    /// included), and the reading records. RISCV NC is the writer: only
    /// [`op::WRITE`], [`op::WRITE_L1`], the writing records, [`op::WAIT`] and the credits
    /// ([`op::CB`]). The packet entries ([`op::PAIR`], [`op::PAIR_CALL`],
    /// [`op::RELEASED`]) and [`op::CALL`] are B's; NC is started by B's
    /// packet, through its `SHARED` and `CALL` words, which are not entries.
    pub const fn permits(self, op: u32) -> bool {
        match self.core {
            crate::tensix::Core::B => {
                op != op::WRITE
                    && op != op::WRITE_L1
                    && op != op::WRITE_L1_MCAST
                    && op != op::CHECK_FLAGS
                    && !matches!(record::direction(op), Some(record::Direction::Write))
            }
            _ => {
                op == op::WRITE
                    || op == op::WRITE_L1
                    || op == op::WRITE_L1_MCAST
                    || op == op::CHECK_FLAGS
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
/// BF16's 1024 two-byte datums plus header, rounded to 64-byte slot alignment.
pub const BF16_TILE_SLOT: u64 = 2112;
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
    IndexPick {
        indices: u32,
        source: u32,
        output: u32,
        bound: u32,
        first: u32,
        rows: u32,
        bytes: u32,
    },
    CopyWords {
        src: u32,
        dst: u32,
        count: u32,
        src_stride: u32,
        dst_stride: u32,
        halfwords: bool,
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
    /// [`op::WRITE_L1_MCAST`]: [`Entry::L1Write`] to every tile of `start..=end`.
    L1Multicast {
        start: (u8, u8),
        end: (u8, u8),
        to: u32,
        from: u32,
        len: u32,
        /// Counted with its run ([`MCAST_RUN_ACKS`]).
        run: bool,
    },
    /// [`op::WRITE_L1`]: `len` bytes of this tile's L1 at `from` to `to` in the L1 of `(x, y)`.
    L1Write {
        x: u8,
        y: u8,
        to: u32,
        from: u32,
        len: u32,
    },
    /// [`op::READ_L1`]: `len` bytes at `from` in the L1 of `(x, y)` to `to` in this tile's.
    L1Read {
        x: u8,
        y: u8,
        from: u32,
        to: u32,
        len: u32,
    },
    /// [`op::FILL`].
    Fill {
        value: u32,
        /// The valid region ([`fill::param`]).
        param: u32,
        dst: u32,
        halfwords: bool,
    },
    /// [`op::KERNEL`]: point each role at its program, if named, post
    /// `generation`, and wait for it.
    Kernel {
        generation: u32,
        /// Per role, `(address, words)` of a resident program, or `(0, 0)`.
        programs: [(u32, u32); 3],
    },
    /// [`op::WAIT`]; `prefetch`, whether it also drains [`TXN_PREFETCH`].
    Wait { prefetch: bool },
    /// [`op::BARRIER`]: `arrive` false is its wait-only form; `release`, word 6.
    Barrier {
        target: u32,
        x: u8,
        y: u8,
        arrive: bool,
        release: bool,
    },
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
    /// Device-generated domain flags, checked by the writer before scatter.
    CheckFlags { slot: u32, rows: u32, cols: u32 },
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
        if w[0] == op::CHECK_FLAGS {
            let (slot, rows, cols) = (w[1], w[2], w[3]);
            if !(1..=32).contains(&rows)
                || !(1..=32).contains(&cols)
                || w[4..].iter().any(|&v| v != 0)
            {
                return Err(error::OP);
            }
            if slot % 16 != 0
                || (slot as u64) < crate::l1::DATA.base
                || slot as u64 + TILE_SLOT > crate::l1::DATA.end
            {
                return Err(error::ALIGNMENT);
            }
            return Ok(Entry::CheckFlags { slot, rows, cols });
        }
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
        if matches!(w[0], op::COPY_WORDS | op::COPY_HALFWORDS) {
            if w[3] == 0 || w[3] > 1024 || w[6] != 0 || w[7] != 0 {
                return Err(error::LENGTH);
            }
            let width = if w[0] == op::COPY_HALFWORDS { 2 } else { 4 };
            for (at, stride) in [(w[1], w[4]), (w[2], w[5])] {
                let end = at as u64 + (w[3] - 1) as u64 * stride as u64 + width as u64;
                if at % width != 0
                    || stride % width != 0
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
                halfwords: width == 2,
            });
        }
        if w[0] == op::WRITE_L1_MCAST {
            let (to, from, len) = (w[3], w[4], w[5]);
            let [x0, y0, x1, y1] = w[1].to_le_bytes();
            if len == 0
                || len > crate::noc::niu::MAX_REQUEST_BYTES
                || w[2] != 0
                || w[6] & !MCAST_RUN_ACKS != 0
                || w[7] != 0
            {
                return Err(error::LENGTH);
            }
            if x0 > x1 || y0 > y1 {
                return Err(error::RANGE);
            }
            let mut x = x0;
            while x <= x1 {
                // Rows are a range, so the two corner rows bound every tile's.
                if !crate::noc::grid::is_tensix_geometry(x, y0)
                    || !crate::noc::grid::is_tensix_geometry(x, y1)
                {
                    return Err(error::RANGE);
                }
                x += 1;
            }
            if from % 16 != to % 16
                || !crate::l1::DATA.contains(from as u64, len as u64)
                || !crate::l1::DATA.contains(to as u64, len as u64)
            {
                return Err(error::ALIGNMENT);
            }
            return Ok(Entry::L1Multicast {
                start: (x0, y0),
                end: (x1, y1),
                to,
                from,
                len,
                run: w[6] == MCAST_RUN_ACKS,
            });
        }
        if w[0] == op::WRITE_L1 || w[0] == op::READ_L1 {
            // The same five words: the remote address third, the local fourth.
            let (x, y, to, from, len) = (w[1], w[2], w[3], w[4], w[5]);
            if len == 0 || len > crate::noc::niu::MAX_REQUEST_BYTES || w[6] != 0 || w[7] != 0 {
                return Err(error::LENGTH);
            }
            if x > 0x3f || y > 0x3f || !crate::noc::grid::is_tensix_geometry(x as u8, y as u8) {
                return Err(error::RANGE);
            }
            if from % 16 != to % 16
                || !crate::l1::DATA.contains(from as u64, len as u64)
                || !crate::l1::DATA.contains(to as u64, len as u64)
            {
                return Err(error::ALIGNMENT);
            }
            return Ok(if w[0] == op::READ_L1 {
                Entry::L1Read {
                    x: x as u8,
                    y: y as u8,
                    from: to,
                    to: from,
                    len,
                }
            } else {
                Entry::L1Write {
                    x: x as u8,
                    y: y as u8,
                    to,
                    from,
                    len,
                }
            });
        }
        if w[0] == op::INDEX_PICK {
            if w[4] == 0
                || w[4] > i32::MAX as u32
                || w[5] >= w[4]
                || w[5] % 32 != 0
                || w[6] == 0
                || w[6] > 32
                || !matches!(w[7], 2 | 4)
            {
                return Err(error::LENGTH);
            }
            for (at, len) in [
                (w[1], TILE_SLOT),
                (w[2], if w[7] == 2 { BF16_TILE_SLOT } else { TILE_SLOT }),
                (w[3], if w[7] == 2 { BF16_TILE_SLOT } else { TILE_SLOT }),
            ] {
                if at % 64 != 0
                    || (at as u64) < crate::l1::DATA.base
                    || at as u64 + len > crate::l1::DATA.end
                {
                    return Err(error::ALIGNMENT);
                }
            }
            return Ok(Self::IndexPick {
                indices: w[1],
                source: w[2],
                output: w[3],
                bound: w[4],
                first: w[5],
                rows: w[6],
                bytes: w[7],
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
            return match w[1] {
                0 => Ok(Entry::Wait { prefetch: false }),
                WAIT_PREFETCH => Ok(Entry::Wait { prefetch: true }),
                _ => Err(error::RANGE),
            };
        }
        if w[0] == op::BARRIER {
            // Word 5, a wait's barriers ([`BARRIER_STANDS_FOR`]): zero on an arrival.
            if w[2] > 0x3f
                || w[3] > 0x3f
                || w[4] > 1
                || (w[4] == 0 && w[5] != 0)
                || w[5] > 0xffff
                || w[6] > 1
                || w[7] != 0
            {
                return Err(error::OP);
            }
            return Ok(Entry::Barrier {
                target: w[1],
                x: w[2] as u8,
                y: w[3] as u8,
                arrive: w[4] == 0,
                release: w[6] == 1,
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
            if !is_role_mailbox_word(w[1]) || w[3..].iter().any(|&v| v != 0) {
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
        if matches!(w[0], op::FILL | op::FILL_HALFWORDS) {
            if w[0] == op::FILL_HALFWORDS && w[1] > u16::MAX as u32 {
                return Err(error::OP);
            }
            let bytes = if w[0] == op::FILL_HALFWORDS {
                BF16_TILE_SLOT
            } else {
                TILE_SLOT
            };
            let slot = |at: u32| at % 16 == 0 && at as u64 + bytes <= crate::tensix::L1_SIZE;
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
                halfwords: w[0] == op::FILL_HALFWORDS,
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
    /// A native arithmetic operation rejected a logical input (division by zero).
    pub const DOMAIN: u32 = 11;
    /// A host-encoded read ([`super::op::READ_ENCODED`]) failed the card's checksum
    /// (or, with `encoded-guard`, its guard, [`super::encoded::guard`]): not what the
    /// host's encoder makes, or damaged on the way.
    pub const ENCODED: u32 = 12;
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
    /// `WRITE_L1`: a Tensix coordinate, both ranges in `DATA` and congruent mod 16, at most one
    /// request, the reserved words zero; and only NC runs it.
    #[test]
    fn an_l1_write_decodes_only_whole_and_only_on_nc() {
        use super::*;
        let to = crate::l1::DATA.end as u32 - 4096;
        let good = [op::WRITE_L1, 4, 4, to, 0x4_0010, 4096, 0, 0];
        assert_eq!(
            Entry::decode(0xFF, [op::WRITE_L1, 4, 4, to + 16, 0x4_0010, 64, 0, 0]),
            Ok(Entry::L1Write {
                x: 4,
                y: 4,
                to: to + 16,
                from: 0x4_0010,
                len: 64
            })
        );
        let with = |k: usize, v: u32| {
            let mut e = good;
            e[k] = v;
            Entry::decode(0xFF, e)
        };
        assert!(matches!(
            Entry::decode(0xFF, good),
            Ok(Entry::L1Write { .. })
        ));
        assert_eq!(with(1, 8), Err(error::RANGE), "the non-memory column");
        assert_eq!(with(1, 9), Err(error::RANGE), "a DRAM column");
        assert_eq!(with(2, 1), Err(error::RANGE), "above the Tensix rows");
        assert_eq!(with(1, 64), Err(error::RANGE), "past six bits");
        assert_eq!(with(5, 0), Err(error::LENGTH));
        assert_eq!(with(5, 16384 + 16), Err(error::LENGTH));
        assert_eq!(with(6, 1), Err(error::LENGTH), "a reserved word");
        assert_eq!(
            with(3, to + 8),
            Err(error::ALIGNMENT),
            "not congruent mod 16"
        );
        assert_eq!(with(3, 0x1_0000), Err(error::ALIGNMENT), "below DATA");
        assert_eq!(
            with(3, crate::l1::DATA.end as u32 - 64),
            Err(error::ALIGNMENT),
            "past DATA"
        );
        assert!(Mover::NC.permits(op::WRITE_L1));
        assert!(!Mover::B.permits(op::WRITE_L1));
    }

    /// `READ_L1`: the remote address third and the local fourth, otherwise `WRITE_L1`'s checks;
    /// any Tensix tile, not only the reader's own; and only B runs it.
    #[test]
    fn an_l1_read_decodes_only_whole_and_only_on_b() {
        use super::*;
        let from = crate::l1::DATA.end as u32 - 4096;
        let good = [op::READ_L1, 5, 4, from, 0x4_0010, 4096, 0, 0];
        assert_eq!(
            Entry::decode(0xFF, [op::READ_L1, 5, 4, from + 16, 0x4_0010, 64, 0, 0]),
            Ok(Entry::L1Read {
                x: 5,
                y: 4,
                from: from + 16,
                to: 0x4_0010,
                len: 64
            })
        );
        let with = |k: usize, v: u32| {
            let mut e = good;
            e[k] = v;
            Entry::decode(0xFF, e)
        };
        assert!(matches!(
            Entry::decode(0xFF, good),
            Ok(Entry::L1Read { .. })
        ));
        assert_eq!(with(1, 8), Err(error::RANGE), "the non-memory column");
        assert_eq!(with(1, 9), Err(error::RANGE), "a DRAM column");
        assert_eq!(with(2, 1), Err(error::RANGE), "above the Tensix rows");
        assert_eq!(with(1, 64), Err(error::RANGE), "past six bits");
        assert_eq!(with(5, 0), Err(error::LENGTH));
        assert_eq!(with(5, 16384 + 16), Err(error::LENGTH));
        assert_eq!(with(7, 1), Err(error::LENGTH), "a reserved word");
        assert_eq!(
            with(3, from + 8),
            Err(error::ALIGNMENT),
            "not congruent mod 16"
        );
        assert_eq!(
            with(3, 0x1_0000),
            Err(error::ALIGNMENT),
            "the remote side below DATA"
        );
        assert_eq!(
            with(4, 0x1_0010),
            Err(error::ALIGNMENT),
            "the local side below DATA"
        );
        assert_eq!(
            with(4, crate::l1::DATA.end as u32 - 48),
            Err(error::ALIGNMENT),
            "the local side past DATA"
        );
        assert!(Mover::B.permits(op::READ_L1));
        assert!(!Mover::NC.permits(op::READ_L1));
    }

    /// `WRITE_L1_MCAST`: a rectangle of Tensix geometry, `start <= end`, no non-memory or DRAM
    /// column inside it; the ranges as `WRITE_L1`'s; and only NC runs it.
    #[test]
    fn a_multicast_decodes_only_over_tensix_columns() {
        use super::*;
        let to = crate::l1::DATA.end as u32 - 4096;
        let mc = |s: (u8, u8), e: (u8, u8), len: u32| {
            Entry::decode(
                0xFF,
                [
                    op::WRITE_L1_MCAST,
                    mcast_corners(s, e),
                    0,
                    to,
                    0x4_0000,
                    len,
                    0,
                    0,
                ],
            )
        };
        assert_eq!(
            mc((1, 2), (7, 11), 64),
            Ok(Entry::L1Multicast {
                start: (1, 2),
                end: (7, 11),
                to,
                from: 0x4_0000,
                len: 64,
                run: false,
            })
        );
        assert_eq!(mc((10, 2), (14, 11), 4096).map(|_| ()), Ok(()));
        let flagged = |w6: u32| {
            Entry::decode(
                0xFF,
                [
                    op::WRITE_L1_MCAST,
                    mcast_corners((1, 2), (2, 2)),
                    0,
                    to,
                    0x4_0000,
                    64,
                    w6,
                    0,
                ],
            )
        };
        assert!(matches!(
            flagged(MCAST_RUN_ACKS),
            Ok(Entry::L1Multicast { run: true, .. })
        ));
        assert_eq!(
            flagged(2),
            Err(error::LENGTH),
            "a bit word 6 does not define"
        );
        assert_eq!(flagged(MCAST_RUN_ACKS | 1 << 31), Err(error::LENGTH));
        assert_eq!(
            mc((7, 2), (10, 2), 64),
            Err(error::RANGE),
            "across columns 8 and 9"
        );
        assert_eq!(
            mc((0, 2), (3, 2), 64),
            Err(error::RANGE),
            "from the DRAM column"
        );
        assert_eq!(
            mc((3, 1), (3, 4), 64),
            Err(error::RANGE),
            "above the Tensix rows"
        );
        assert_eq!(mc((4, 4), (3, 4), 64), Err(error::RANGE), "reversed");
        assert_eq!(mc((3, 4), (4, 4), 0), Err(error::LENGTH));
        assert!(Mover::NC.permits(op::WRITE_L1_MCAST));
        assert!(!Mover::B.permits(op::WRITE_L1_MCAST));
    }

    /// Word 4 of a `BARRIER` says whether the unit arrives (0, as every barrier did before
    /// the wait-only form) or only waits (1); anything else, or a word past it, is refused.
    #[test]
    fn a_barrier_arrives_or_only_waits_and_nothing_else() {
        use super::*;
        let decode = |w4: u32, w5: u32| Entry::decode(0xFF, [op::BARRIER, 7, 1, 2, w4, w5, 0, 0]);
        assert_eq!(
            decode(0, 0),
            Ok(Entry::Barrier {
                target: 7,
                x: 1,
                y: 2,
                arrive: true,
                release: false,
            })
        );
        assert_eq!(
            decode(1, 0),
            Ok(Entry::Barrier {
                target: 7,
                x: 1,
                y: 2,
                arrive: false,
                release: false,
            })
        );
        assert_eq!(decode(2, 0), Err(error::OP));
        // A wait may say how many barriers it stands for; an arrival may not.
        assert!(decode(1, 3).is_ok());
        assert_eq!(decode(1, 0x1_0000), Err(error::OP));
        assert_eq!(decode(0, 1), Err(error::OP));
        // Word 6: 1 waits for the coordinator's release; nothing else, and word 7 zero.
        let word = |w6: u32, w7: u32| Entry::decode(0xFF, [op::BARRIER, 7, 1, 2, 0, 0, w6, w7]);
        assert_eq!(
            word(1, 0),
            Ok(Entry::Barrier {
                target: 7,
                x: 1,
                y: 2,
                arrive: true,
                release: true,
            })
        );
        assert_eq!(word(2, 0), Err(error::OP));
        assert_eq!(word(0, 1), Err(error::OP));
    }

    #[test]
    fn resident_index_selection_checks_every_slot_and_geometry() {
        use super::*;
        let at = crate::l1::DATA.base as u32;
        for bytes in [2, 4] {
            let valid = [
                op::INDEX_PICK,
                at,
                at + 0x1040,
                at + 0x2080,
                35,
                32,
                32,
                bytes,
            ];
            assert!(matches!(
                Entry::decode(0xff, valid),
                Ok(Entry::IndexPick { .. })
            ));
            for (field, value) in [
                (4, 0),
                (4, i32::MAX as u32 + 1),
                (5, 35),
                (5, 1),
                (6, 0),
                (6, 33),
                (7, 1),
            ] {
                let mut invalid = valid;
                invalid[field] = value;
                assert_eq!(Entry::decode(0xff, invalid), Err(error::LENGTH));
            }
            for field in 1..=3 {
                let mut invalid = valid;
                invalid[field] += 1;
                assert_eq!(Entry::decode(0xff, invalid), Err(error::ALIGNMENT));
                invalid[field] = crate::l1::DATA.end as u32 - 64;
                assert_eq!(Entry::decode(0xff, invalid), Err(error::ALIGNMENT));
            }
        }
        assert!(Mover::B.permits(op::INDEX_PICK));
        assert!(!Mover::NC.permits(op::INDEX_PICK));
    }

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
                dst: 0x2_0000,
                halfwords: false
            })
        );
        let half = [
            op::FILL_HALFWORDS,
            0x7fc1,
            fill::param(3, 7),
            0x2_0000,
            0,
            0,
            0,
            0,
        ];
        assert!(matches!(
            Entry::decode(ALL, half),
            Ok(Entry::Fill {
                halfwords: true,
                ..
            })
        ));
        let mut bad_half = half;
        bad_half[1] = 0x1_0000;
        assert!(Entry::decode(ALL, bad_half).is_err());
        let mut end_half = half;
        end_half[3] = (crate::tensix::L1_SIZE - BF16_TILE_SLOT) as u32;
        assert!(Entry::decode(ALL, end_half).is_ok());
        end_half[3] += 16;
        assert!(Entry::decode(ALL, end_half).is_err());
        let copy_half = [op::COPY_HALFWORDS, 0x2_0012, 0x2_1052, 1024, 0, 2, 0, 0];
        assert!(matches!(
            Entry::decode(ALL, copy_half),
            Ok(Entry::CopyWords {
                halfwords: true,
                ..
            })
        ));
        for opcode in [op::COPY_HALFWORDS, op::FILL_HALFWORDS] {
            assert!(Mover::B.permits(opcode));
            assert!(!Mover::NC.permits(opcode));
        }
        let mut bad_half = copy_half;
        bad_half[1] += 1;
        assert!(Entry::decode(ALL, bad_half).is_err());
        let mut bad_half = copy_half;
        bad_half[2] = crate::l1::DATA.end as u32 - 2;
        assert!(Entry::decode(ALL, bad_half).is_err());
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
            Ok(Entry::Wait { prefetch: false })
        );
        assert_eq!(
            Entry::decode(ALL, [op::WAIT, WAIT_PREFETCH, 0, 0, 0, 0, 0, 0]),
            Ok(Entry::Wait { prefetch: true })
        );
        assert_eq!(
            Entry::decode(ALL, [op::WAIT, 2, 0, 0, 0, 0, 0, 0]),
            Err(error::RANGE)
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
