//! The role-side deadline, host-visible blocked status and release route that
//! make a blocking instruction safe to run (`tt-firmware/src/corpus.rs`).
//!
//! A Tensix thread parked in a Wait Gate (`ATGETM` behind a mutex that is held,
//! an `ATCAS` whose compare never holds, an `ATINCGETPTR` on a full or empty
//! FIFO) takes no further instruction, and the RISC-V `wait_for_coprocessor`
//! load that normally follows a program stalls inside the memory subsystem with
//! no software timeout. A **guarded** run therefore replaces that wait with a
//! bounded poll of a **completion semaphore** that the program's own last
//! instruction posts (`SyncUnit.md`: a RISC-V core can `lw` any semaphore),
//! reports [`BLOCKED`] when the bound expires, and keeps polling for a release
//! request ([`Guard::release`]) and the completion. A semaphore rather than an
//! L1 word because the program has no store ttsim runs (`STOREIND`, divergence
//! row 77's neighbour) and a post is the Sync Unit's own ordered completion.
//!
//! Each poll also publishes a snapshot of all eight semaphores
//! ([`Guard::snapshot`]), so the host can observe what the threads have posted:
//! the gates use a semaphore per thread as a "this thread got here" marker.
//!
//! The block sits at a fixed offset inside each role mailbox, away from every
//! word [`crate::mailbox::Descriptor`] writes. It is **one-shot**: the runner
//! clears the arming word as soon as it reads it, so a stale arming word left by
//! an earlier process (L1 survives on silicon) can never turn a later ordinary
//! run into a guarded one.

use crate::mailbox::role::Mailbox;

/// Offset of the guard block from a role mailbox's base.
pub const OFFSET: u64 = 0x100;
/// Written to the arming word by the host to guard the next run.
pub const ARMED: u32 = 0x4755_4152;
/// The role's [`STATUS`](crate::mailbox::STATUS) while a guarded program has
/// not completed within its deadline.
pub const BLOCKED: u32 = 0x5747_b10c;
/// The panic code of a runner that gave up after the grace period.
pub const ABANDONED: u32 = 0x4755_0001;
/// The panic code of a refused guarded program.
pub const REFUSED: u32 = 0x4755_0002;
/// The longest program a guarded run takes. A blocked thread's instruction
/// FIFO is the only place later pushes can wait, and a push into a full
/// FIFO stalls the pushing core with no timeout; the first FIFO holds 28
/// (`PushTensixInstruction.md:15`), so the program, completion post
/// included, must stay under that.
pub const MAX_WORDS: u32 = 24;
/// The largest deadline or grace count the host may ask for (polls).
pub const MAX_POLLS: u32 = 1 << 30;
/// Semaphores a release request may post: the low eight bits.
pub const RELEASE_MASK: u32 = 0xff;
/// Set in a release request: before posting, the runner stores
/// [`Guard::poke_value`] to the L1 word at [`Guard::poke_addr`] -- a store by
/// this role's RISC-V core, an agent other than the host and every Tensix
/// thread, which is what frees an `ATCAS` or `ATINCGETPTR` that is polling
/// that word. The address must be a word in the data arena.
pub const POKE: u32 = 1 << 31;

/// The guard block of one role.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Guard {
    base: u64,
}

impl Guard {
    pub const fn of(mailbox: Mailbox) -> Guard {
        Guard {
            base: mailbox.base() + OFFSET,
        }
    }
    /// One-shot arming word.
    pub const fn arm(self) -> u64 {
        self.base
    }
    /// Polls before the runner reports [`BLOCKED`].
    pub const fn deadline(self) -> u64 {
        self.base + 0x04
    }
    /// Further polls, once blocked, before the runner gives up.
    pub const fn grace(self) -> u64 {
        self.base + 0x08
    }
    /// Polls the runner spent waiting (diagnostic; written at the end).
    pub const fn polls(self) -> u64 {
        self.base + 0x0c
    }
    /// Which semaphore (0..8) the program posts when it has completed.
    pub const fn complete_semaphore(self) -> u64 {
        self.base + 0x10
    }
    /// The eight semaphore values, one nibble each (semaphore `i` in bits
    /// `4i..4i+4`), as of the runner's last poll.
    pub const fn snapshot(self) -> u64 {
        self.base + 0x14
    }
    /// Host-written release request; the runner consumes it.
    pub const fn release(self) -> u64 {
        self.base + 0x18
    }
    /// Release requests the runner has carried out.
    pub const fn released(self) -> u64 {
        self.base + 0x1c
    }
    /// Breadcrumb: the last [`stage`] the runner reached (the host zeroes it
    /// when arming). Written, and fenced, at each step of a guarded run.
    pub const fn stage(self) -> u64 {
        self.base + 0x20
    }
    /// The [`PollMode`] the host asked for.
    pub const fn mode(self) -> u64 {
        self.base + 0x24
    }
    /// The L1 address a [`POKE`] request stores to.
    pub const fn poke_addr(self) -> u64 {
        self.base + 0x28
    }
    /// The value a [`POKE`] request stores.
    pub const fn poke_value(self) -> u64 {
        self.base + 0x2c
    }
    /// The completion word of [`PollMode::L1Word`]: 16-byte aligned, so the
    /// program stores to it with a zero offset.
    pub const fn complete_word(self) -> u64 {
        self.base + 0x30
    }
    /// Bytes the block spans.
    pub const BYTES: u64 = 0x40;

    /// Everything the host writes to arm the next run: `(address, value)`.
    pub const fn arm_writes(
        self,
        deadline_polls: u32,
        grace_polls: u32,
        complete_semaphore: u32,
        mode: PollMode,
    ) -> [(u64, u32); 13] {
        [
            (self.snapshot(), 0),
            (self.release(), 0),
            (self.released(), 0),
            (self.deadline(), deadline_polls),
            (self.grace(), grace_polls),
            (self.complete_semaphore(), complete_semaphore),
            (self.polls(), 0),
            (self.stage(), 0),
            (self.mode(), mode as u32),
            (self.complete_word(), 0),
            (self.poke_addr(), 0),
            (self.poke_value(), 0),
            // Last, so the runner never sees a half-written block.
            (self.arm(), ARMED),
        ]
    }
}

/// How the runner learns that a guarded program has completed.
///
/// All three keep the deadline, the BLOCKED status and the release path.
/// They differ only in what the runner loads each poll, so a silicon hang
/// can be bisected between them.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum PollMode {
    /// Each poll loads all eight Tensix semaphores back to back and
    /// publishes them ([`Guard::snapshot`]); the program posts a semaphore.
    ///
    /// **Unsafe beyond the first poll on silicon; kept as a recorded
    /// finding, not for use.** On Blackhole card 0 a role that had to wait
    /// (a blocked program) showed stage `FIRST_POLL_DONE`, status Running
    /// and a live poll count of 0 for 3 s (`guard_poll_rate_calibration_full`,
    /// run id: to be filled in by the coordinator), while short programs
    /// that were already complete at the first poll finished. The first
    /// poll has a fence between its loads and later ones do not, which is
    /// consistent with the `ManualTTSync.md:18` rule against overlapping
    /// loads in `PC_BUF_BASE..+0xFFFF`. Not proven. Use [`PollMode::Light`].
    Full = 0,
    /// Each poll loads only the completion semaphore, with the load's result
    /// consumed (`andi`) before anything else loads, and every load of the
    /// periodic snapshot is consumed the same way, so one load is in flight
    /// at a time (`ManualTTSync.md:18`). The full snapshot is taken every
    /// [`LIGHT_SNAPSHOT_EVERY`] polls and once at completion. **The
    /// default on silicon and ttsim.** Polling a semaphore is the page's
    /// own mechanism (`ManualTTSync.md`, "Tensix semaphores": reads never
    /// contend with other agents).
    Light = 1,
    /// No semaphore loads at all: the program's epilogue stores
    /// [`COMPLETE_TOKEN`] to [`Guard::complete_word`] with `STOREIND`
    /// (`ManualTTSync.md`: a viable polled address space) and the runner
    /// polls that L1 word. Silicon only (ttsim refuses `STOREIND`), and no
    /// snapshot.
    L1Word = 2,
}

/// Polls between full snapshots in [`PollMode::Light`].
pub const LIGHT_SNAPSHOT_EVERY: u32 = 64;
/// What a [`PollMode::L1Word`] program stores at its end.
pub const COMPLETE_TOKEN: u32 = 0x474f_4f44;
/// GPRs the L1-word epilogue uses (address, zero offset, token).
pub const RESERVED_GPRS: [u32; 3] = [61, 62, 63];

/// Breadcrumb values of [`Guard::stage`], in the order a run passes them.
/// `FIRST_LOAD + i` is written just before the first poll's load of
/// semaphore `i`, so a hang inside a load names which one.
pub mod stage {
    /// The runner read and consumed the arming word.
    pub const ARM_SEEN: u32 = 1;
    /// Deadline, grace and completion semaphore passed their checks.
    pub const PARAMS_OK: u32 = 2;
    /// Any post left on the completion semaphore (or word) was cleared.
    pub const CLEARED: u32 = 3;
    /// Every word of the program was pushed.
    pub const PUSHED: u32 = 4;
    /// The poll loop was entered.
    pub const POLL_ENTERED: u32 = 5;
    /// The first poll's snapshot pass finished.
    pub const FIRST_POLL_DONE: u32 = 6;
    /// The deadline expired and BLOCKED was published.
    pub const BLOCKED: u32 = 7;
    /// The completion was seen and the poll loop left.
    pub const POLL_DONE: u32 = 8;
    /// `wait_for_coprocessor` returned.
    pub const DRAINED: u32 = 9;
    /// Written just before the first poll's load of semaphore `i`.
    pub const FIRST_LOAD: u32 = 0x100;

    /// A readable name.
    pub fn name(stage: u32) -> &'static str {
        match stage {
            0 => "not armed or not reached",
            ARM_SEEN => "ARM_SEEN",
            PARAMS_OK => "PARAMS_OK",
            CLEARED => "CLEARED",
            PUSHED => "PUSHED",
            POLL_ENTERED => "POLL_ENTERED",
            FIRST_POLL_DONE => "FIRST_POLL_DONE",
            BLOCKED => "BLOCKED",
            POLL_DONE => "POLL_DONE",
            DRAINED => "DRAINED",
            s if (FIRST_LOAD..FIRST_LOAD + 8).contains(&s) => "in the first poll's load",
            _ => "unknown",
        }
    }
}

/// The value of semaphore `index` in a [`Guard::snapshot`] word.
pub const fn snapshot_value(snapshot: u32, index: u32) -> u32 {
    (snapshot >> (4 * index)) & 0xf
}

const _: () = assert!(OFFSET % 16 == 0 && OFFSET > crate::mailbox::MAILBOX_SIZE);
const _: () = assert!(Guard::BYTES + OFFSET < 0x2000);
const _: () = assert!(Guard::BYTES % 16 == 0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_block_stays_clear_of_the_descriptor() {
        use crate::mailbox::Descriptor;
        for t in 0..3 {
            let mb = Mailbox::of(t);
            let g = Guard::of(mb);
            for (at, _) in Descriptor::default().writes(mb) {
                assert!(at < g.arm() || at >= g.arm() + Guard::BYTES);
            }
            assert!(g.released() + 4 <= g.arm() + Guard::BYTES);
            assert!(g.arm() + Guard::BYTES < mb.dump_offset(0, 0));
            // The arm write is last, so a half-written block is never read,
            // and every word of the block is written (nothing stale survives).
            let writes = g.arm_writes(1, 1, 7, PollMode::Light);
            assert_eq!(writes[12].0, g.arm());
            assert_eq!(
                writes.iter().find(|w| w.0 == g.mode()).unwrap().1,
                1,
                "the mode is written"
            );
            let mut at: [u64; 13] = core::array::from_fn(|k| writes[k].0);
            at.sort_unstable();
            let mut want = [
                g.arm(),
                g.deadline(),
                g.grace(),
                g.polls(),
                g.complete_semaphore(),
                g.snapshot(),
                g.release(),
                g.released(),
                g.stage(),
                g.mode(),
                g.complete_word(),
                g.poke_addr(),
                g.poke_value(),
            ];
            want.sort_unstable();
            assert_eq!(at, want);
            assert_eq!(g.complete_word() % 16, 0);
            assert!(g.complete_word() + 4 <= g.arm() + Guard::BYTES);
            assert_eq!(snapshot_value(0x7654_3210, 5), 5);
            assert_eq!(
                stage::name(stage::FIRST_LOAD + 3),
                "in the first poll's load"
            );
        }
    }
}
