//! Layout of the NoC probe (`tt-firmware/src/bin/noc_probe_b.rs`), shared by the
//! image and the host gates (`step115`-`step117`).
//!
//! The image replays requests the host encoded with the typed encoders
//! ([`super::atomic::AtomicRequest`], [`super::multicast::MulticastWrite`]): it
//! owns no encoding of its own, so what the probe sends is exactly what the host
//! gates model. For each request it waits for the initiator, writes the register
//! values, issues, then waits for completion the way the request class needs and
//! records what it saw.
//!
//! Everything is staged in the initiating tile's L1 at addresses clear of the
//! image (0), the mailbox (`mailbox::MAILBOX_BASE`) and the other probes.

use super::niu::initiator::*;

/// Script header: word 0 the number of requests (at most [`MAX_REQUESTS`]),
/// word 1 the spin budget (polls of a completion counter before a request is
/// reported as timed out).
pub const SCRIPT: u64 = 0x2_0000;
pub const MAX_REQUESTS: u32 = 8;
/// First request record.
pub const REQUESTS: u64 = SCRIPT + 0x10;
/// Bytes per request record.
pub const REQUEST_STRIDE: u64 = 0x40;
/// Result records.
pub const RESULTS: u64 = 0x2_1000;
pub const RESULT_STRIDE: u64 = 0x20;
/// One breadcrumb word, `(request << 8) | phase`, published with a fence before
/// each phase: after a hang it names the phase that never finished.
pub const STAGE: u64 = 0x2_1800;
/// Register snapshots, one per request, taken only for a [`flag::PARTIAL`]
/// request: the ten initiator words ([`REGISTERS`] order) read back once the
/// request has completed, so a host gate can say which registers kept the value
/// software wrote (`step118`).
pub const SNAPSHOTS: u64 = 0x2_1900;
/// Bytes per snapshot record.
pub const SNAPSHOT_STRIDE: u64 = 0x40;
/// Where the host stages source data and atomic response words, per probe.
pub const DATA: u64 = 0x2_2000;

/// The registers of a request record, in the order the image writes them. The
/// last is written only for a multicast ([`kind::MULTICAST`]).
pub const REGISTERS: [u64; 11] = [
    TARG_ADDR_LO,
    TARG_ADDR_MID,
    TARG_ADDR_HI,
    RET_ADDR_LO,
    RET_ADDR_MID,
    RET_ADDR_HI,
    PACKET_TAG,
    CTRL,
    AT_LEN_BE,
    AT_DATA,
    BRCST_EXCLUDE,
];

/// Request record: word 0 [`kind`], word 1 [`flag`]s, word 2 the transaction
/// ID, word 3 the acknowledgements a multicast expects, words 4..15 the register
/// values in [`REGISTERS`] order.
pub mod word {
    pub const KIND: u64 = 0;
    pub const FLAGS: u64 = 4;
    pub const TXN: u64 = 8;
    pub const EXPECTED_ACKS: u64 = 12;
    pub const REGISTERS: u64 = 16;
    /// With [`super::flag::PARTIAL`]: bit `i` set writes register `i` of
    /// [`super::REGISTERS`] and clear leaves it as it is. In the last word of
    /// the record, after the eleventh register.
    pub const MASK: u64 = 60;
}

pub mod kind {
    /// A unicast (atomic or write): complete when the ID's outstanding counter
    /// is back to zero.
    pub const UNICAST: u32 = 0;
    /// A response-marked broadcast: complete when `NIU_MST_WR_ACK_RECEIVED` has
    /// advanced by the expected recipient count, after which the ID is cleared.
    pub const MULTICAST: u32 = 1;
}

pub mod flag {
    /// Issue through NoC #1's NIU.
    pub const NOC1: u32 = 1 << 0;
    /// Record `NIU_TRANS_COUNT_RTZ_SOURCE` before issuing and after completion
    /// (the polling form of the completion interrupt, `Interrupts.md`).
    pub const RTZ: u32 = 1 << 1;
    /// With [`RTZ`]: clear the ID's `SOURCE` bit (`NIU_TRANS_COUNT_RTZ_CLR`)
    /// before the first read, so a set bit afterwards is this request's
    /// positive-to-zero transition and not a sticky leftover.
    pub const RTZ_CLEAR: u32 = 1 << 3;
    /// Do not write `NIU_BASE + 0x60` after a multicast. For the simulator,
    /// which refuses that write (divergence row 88): the ID's counter is then
    /// left at `1 - recipients`, which is only acceptable on a simulator that is
    /// about to be discarded.
    pub const NO_CLEAR: u32 = 1 << 2;
    /// Write only the registers `word::MASK` selects, and record the initiator's
    /// ten words in `SNAPSHOTS` after completion: the set-state / with-state
    /// persistence probe (`step118`). The initiator is otherwise left as the
    /// previous request left it.
    pub const PARTIAL: u32 = 1 << 4;
    /// Use request initiator 1 (`NIU_BASE + 0x800`, `initiator::STRIDE`) instead
    /// of initiator 0: where the mover's fast read path keeps its own registers.
    pub const INITIATOR_1: u32 = 1 << 5;
}

/// Result record words (bytes, from `RESULTS + k * RESULT_STRIDE`).
pub mod result {
    pub const STATUS: u64 = 0;
    pub const ACK_BEFORE: u64 = 4;
    pub const ACK_AFTER: u64 = 8;
    pub const OUTSTANDING_AFTER: u64 = 12;
    pub const RTZ_BEFORE: u64 = 16;
    pub const RTZ_AFTER: u64 = 20;
    pub const POLLS: u64 = 24;
    pub const OUTSTANDING_BEFORE: u64 = 28;
}

/// `result::STATUS` values.
pub mod status {
    pub const OK: u32 = 0;
    /// The spin budget ran out before completion; nothing was cleared.
    pub const TIMEOUT: u32 = 1;
    /// A multicast saw more acknowledgements than recipients.
    pub const EXCESS_ACKS: u32 = 2;
    /// The ID's counter was not zero before the request: not issued.
    pub const NOT_QUIESCENT: u32 = 3;
    /// The initiator never became free: not issued.
    pub const INITIATOR_BUSY: u32 = 4;
    /// The record was malformed: not issued.
    pub const BAD_RECORD: u32 = 5;
}

/// Phases of one request.
pub mod phase {
    pub const BEGIN: u32 = 1;
    pub const INITIATOR_FREE: u32 = 2;
    pub const REGISTERS_WRITTEN: u32 = 3;
    /// `CMD_CTRL` written; if this is the last phase the NIU never took the
    /// request (or the read-back never returned).
    pub const ISSUED: u32 = 4;
    pub const COMPLETE: u32 = 5;
    pub const DONE: u32 = 6;

    pub const fn name(phase: u32) -> &'static str {
        match phase {
            BEGIN => "begin",
            INITIATOR_FREE => "initiator free",
            REGISTERS_WRITTEN => "registers written",
            ISSUED => "issued",
            COMPLETE => "complete",
            DONE => "done",
            _ => "unknown",
        }
    }
}

const _: () = assert!(word::REGISTERS + 11 * 4 <= word::MASK);
const _: () = assert!(word::MASK + 4 <= REQUEST_STRIDE);
const _: () = assert!(10 * 4 <= SNAPSHOT_STRIDE);
const _: () = assert!(result::OUTSTANDING_BEFORE + 4 <= RESULT_STRIDE);

pub const fn stage(request: u32, phase: u32) -> u32 {
    (request << 8) | phase
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_regions_are_disjoint_and_clear_of_the_image_and_mailbox() {
        let regions = [
            (SCRIPT, REQUESTS + MAX_REQUESTS as u64 * REQUEST_STRIDE),
            (RESULTS, RESULTS + MAX_REQUESTS as u64 * RESULT_STRIDE),
            (STAGE, STAGE + 4),
            (SNAPSHOTS, SNAPSHOTS + MAX_REQUESTS as u64 * SNAPSHOT_STRIDE),
            (DATA, DATA + 0x1000),
        ];
        for (i, a) in regions.iter().enumerate() {
            assert!(a.0 >= crate::dm::IMAGE_BASE + crate::dm::IMAGE_MAX);
            assert!(a.1 <= crate::mailbox::MAILBOX_BASE);
            for b in &regions[i + 1..] {
                assert!(a.1 <= b.0 || b.1 <= a.0, "{a:x?} overlaps {b:x?}");
            }
        }
    }
}
