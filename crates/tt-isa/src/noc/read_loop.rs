//! Layout of the bare read-loop probe (`tt-firmware/src/bin/read_loop_b.rs`),
//! shared by the image and its host test (`silicon_probe_read_loop`).
//!
//! A development measurement, not a protocol the product uses: it asks what a
//! B read costs with every part of the interpreter removed. The host encodes the
//! first request with the typed encoder (`DramMove::words`), so the image owns
//! no encoding; the image then issues `count` requests, each `len` bytes on from
//! the last (or replayed from a table of encoded requests), and reports the
//! cycles each repetition took by the tile's wall clock, and a checksum of the
//! destination window after each, which is then refilled with poison so that every
//! repetition has to land its own data.
//!
//! The control block sits in the sliver between B's mover area and NC's image
//! ([`crate::l1::REGIONS`]: unused), the table in the mover's list ring, which
//! nothing uses while this image runs instead of the mover.

/// The control block: `u32` words at these offsets.
pub const CONTROL: u64 = 0x1_9900;
/// Host -> image: the run's sequence number. Never 0.
pub const SEQ: u64 = CONTROL;
/// Image -> host: the last sequence number finished.
pub const DONE: u64 = CONTROL + 0x04;
/// One of [`mode`].
pub const MODE: u64 = CONTROL + 0x08;
/// Requests per repetition.
pub const COUNT: u64 = CONTROL + 0x0C;
/// In-flight cap (`noc::set_cap`: 0 for the maximum).
pub const CAP: u64 = CONTROL + 0x10;
/// Transaction ID.
pub const TXN: u64 = CONTROL + 0x14;
/// The first request, as `DramRequest` carries it: `TARG`, `TARG_HI`, `RET`,
/// `RET_HI`, `TAG`, `CTRL`, `LEN`.
pub const REQUEST: u64 = CONTROL + 0x18;
/// Words in a request.
pub const REQUEST_WORDS: u64 = 7;
/// Where the return address wraps back to `RET`: the end of the destination
/// window.
pub const RET_END: u64 = CONTROL + 0x34;
/// Repetitions after one untimed warm-up (at most [`MAX_REPS`]).
pub const REPS: u64 = CONTROL + 0x38;
/// Image -> host: how many requests waited for room under the cap, over the run.
pub const STALLS: u64 = CONTROL + 0x40;
/// Host -> image, for [`mode::DECODED`]: the DRAM channels the mover may use
/// (`dm::USABLE`) and this tile's coordinate (`dm::MY_X`, `dm::MY_Y`).
pub const USABLE: u64 = CONTROL + 0x44;
pub const ME_X: u64 = CONTROL + 0x48;
pub const ME_Y: u64 = CONTROL + 0x4C;
/// Image -> host: cycles per repetition, one word each.
pub const CYCLES: u64 = CONTROL + 0x100;
/// Image -> host: the wrapping sum of the destination window's words after each
/// repetition (the warm-up's is not kept), one word each.
pub const CHECKS: u64 = CONTROL + 0x200;
pub const MAX_REPS: u32 = 64;
/// What the window is refilled with between repetitions.
pub const POISON: u32 = 0xA5A5_A5A5;

/// [`mode::TABLE`]'s encoded requests: `count` records of [`TABLE_STRIDE`] bytes,
/// the seven request words and one of padding.
pub const TABLE: u64 = 0x1_4000;
pub const TABLE_STRIDE: u64 = 32;
pub const TABLE_MAX: u32 = 512;

const _: () = assert!(CYCLES + MAX_REPS as u64 * 4 <= CHECKS);
const _: () = assert!(CHECKS + MAX_REPS as u64 * 4 <= 0x1_A000);
const _: () = assert!(TABLE + TABLE_MAX as u64 * TABLE_STRIDE <= 0x1_8000);

/// How the image issues its requests.
pub mod mode {
    /// `noc::issue_dram_on`: every initiator register per request, the
    /// interpreter's default read path, with the address stepped by arithmetic.
    pub const STEPPED: u32 = 0;
    /// `noc::issue_read`: request initiator 1, writing only what changes (X6).
    pub const FAST: u32 = 1;
    /// `noc::issue_dram_on` with each request loaded from [`super::TABLE`]: the
    /// thinnest possible interpreter, a replay of encoded requests.
    pub const TABLE: u32 = 2;
    /// The mover's own per-entry work, without its list loop or `exec`: each
    /// request is `Entry::decode`d from a list entry in [`super::TABLE`] (the
    /// shipped format, `op::READ`), checked once as a `DramMove`, and issued as
    /// `issue_via` issues it. What it costs over [`STEPPED`] is decode and
    /// validation; what the mover costs over it is everything else.
    pub const DECODED: u32 = 3;
}
