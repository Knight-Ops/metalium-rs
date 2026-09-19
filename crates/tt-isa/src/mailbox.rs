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

/// Total size the firmware may assume is its own.
pub const MAILBOX_SIZE: u64 = 0x40;

// Compile-time rather than a test: if the mailbox ever moved past the end of L1,
// every access to it would be out of bounds, and that should stop the build rather
// than fail a test run.
const _: () = assert!(MAILBOX_BASE + MAILBOX_SIZE <= crate::tensix::L1_SIZE);

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
