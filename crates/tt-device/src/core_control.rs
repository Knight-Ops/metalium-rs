//! Starting and stopping the baby RISC-V cores of a Tensix tile.
//!
//! All of this works through the tile-control/debug/status register block at
//! `0xFFB1_2000`, which is one of the few non-L1 regions reachable over the NoC
//! (`BabyRISCV/README.md:106`). That NoC visibility is the only reason a host can
//! start a core at all.

use tt_isa::mailbox::{self, status};
use tt_isa::noc::{NocCoord, NocId};
use tt_isa::tensix::{self, Core};

use crate::device::Window;
use crate::{Device, Result, Transport, TransportError};

/// Why a core did not reach the state we waited for.
#[derive(Debug)]
pub enum WaitError {
    /// The budget expired.
    TimedOut {
        /// Simulated cycles advanced, on a simulated transport; polls times
        /// [`CYCLES_PER_POLL`] on silicon, where it measures nothing but effort.
        waited_cycles: u64,
        /// How long the wait actually took, on silicon. `None` on the simulator,
        /// where wall-clock time says nothing about the device.
        wall_clock: Option<std::time::Duration>,
        last_status: u32,
    },
    /// The firmware published [`status::PANICKED`].
    Panicked { code: u32 },
}

impl std::fmt::Display for WaitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaitError::TimedOut {
                waited_cycles,
                wall_clock,
                last_status,
            } => {
                let hint = match *last_status {
                    0 => {
                        " (the mailbox is still zero: the core may never have started, or is \
                          running code that does not reach its prologue)"
                    }
                    s if s == status::RUNNING => " (the core started but has not finished)",
                    _ => "",
                };
                match wall_clock {
                    Some(t) => write!(
                        f,
                        "core did not respond within {} ms; \
                           last status {last_status:#010x}{hint}",
                        t.as_millis()
                    ),
                    None => write!(
                        f,
                        "core did not respond within {waited_cycles} cycles; \
                           last status {last_status:#010x}{hint}"
                    ),
                }
            }
            WaitError::Panicked { code } => {
                write!(f, "firmware panicked with code {code}")
            }
        }
    }
}

impl std::error::Error for WaitError {}

/// How many simulated cycles to advance between polls.
///
/// ttsim's own consumers use 256 clocks per BAR read (`ttsim-riscv64`) and 1000
/// per 100 µs (`ttsim-qemu`), so this sits between them. On silicon `tick` is a
/// no-op and this only sets the polling granularity.
pub const CYCLES_PER_POLL: u32 = 512;

/// The shortest wall-clock wait on silicon, whatever the budget says.
///
/// A budget is written in simulated cycles, and on silicon it is converted at a
/// nominal 1 GHz -- so the gates' 200 000- and 400 000-cycle budgets would be a
/// fraction of a millisecond, less than a handful of MMIO round trips through a
/// TLB window. The floor makes a silicon timeout mean "the core is not doing
/// this", not "the host asked too quickly".
pub const SILICON_MIN_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

impl<T: Transport> Device<T> {
    /// Read the tile-wide soft-reset register.
    pub fn read_soft_reset<N: NocId>(&mut self, window: &Window, tile: NocCoord<N>) -> Result<u32> {
        self.read32(window, tile, tensix::SOFT_RESET_0)
    }

    /// Hold `core` in reset, or release it.
    ///
    /// The soft-reset register has no atomic bit operations, so this is a
    /// read-modify-write and *software must provide its own mutual exclusion*
    /// (`SoftReset.md:3-4`). Taking `&mut self` provides it within this process;
    /// if anything else on the host can touch the same tile, that is not enough
    /// and the caller must serialise at a higher level.
    pub fn set_core_reset<N: NocId>(
        &mut self,
        window: &Window,
        tile: NocCoord<N>,
        core: Core,
        held: bool,
    ) -> Result<()> {
        let current = self.read_soft_reset(window, tile)?;
        let updated = if held {
            current | core.soft_reset_mask()
        } else {
            current & !core.soft_reset_mask()
        };
        if updated != current {
            self.write32(window, tile, tensix::SOFT_RESET_0, updated)?;
        }
        Ok(())
    }

    /// Is `core` currently held in reset?
    pub fn is_core_in_reset<N: NocId>(
        &mut self,
        window: &Window,
        tile: NocCoord<N>,
        core: Core,
    ) -> Result<bool> {
        Ok(self.read_soft_reset(window, tile)? & core.soft_reset_mask() != 0)
    }

    /// Point a core at `address` when it next leaves reset.
    ///
    /// Returns an error for RISCV B, whose entry point is hardwired to L1 offset 0
    /// and has no override register.
    pub fn set_reset_pc<N: NocId>(
        &mut self,
        window: &Window,
        tile: NocCoord<N>,
        core: Core,
        address: u32,
    ) -> Result<()> {
        let (Some(pc_reg), Some((override_reg, override_bit))) =
            (core.reset_pc_register(), core.reset_pc_override())
        else {
            return Err(TransportError::Misaligned {
                bar: crate::Bar::Bar0,
                offset: address as u64,
                len: 0,
                reason: "RISCV B has no reset-PC override; its entry point is fixed at L1 offset 0",
            });
        };
        self.write32(window, tile, pc_reg, address)?;
        let current = self.read32(window, tile, override_reg)?;
        self.write32(window, tile, override_reg, current | override_bit)
    }

    /// Load a firmware image into L1 and start `core` on it.
    ///
    /// The order matters. The image is written *while the core is held in reset*,
    /// for two reasons that both come from things Blackhole does not have:
    ///
    /// * There is no `fence.i` — Zifencei is unimplemented and `fence.i` is a
    ///   `nop`. Leaving reset invalidates the instruction cache
    ///   (`SoftReset.md:116`), so loading before release is the only sequence that
    ///   needs no explicit invalidation. Reloading a *running* core would need
    ///   `RISCV_IC_INVALIDATE_InvalidateAll` in Tensix backend config, a register
    ///   that is not NoC-accessible.
    /// * Local data RAM spends up to 2048 cycles zeroing itself after release, and
    ///   NoC accesses in that window are silently discarded
    ///   (`BabyRISCV/README.md:152`). Nothing here is staged into local RAM — the
    ///   image lives entirely in L1 and the core sets up its own stack — so the
    ///   window is never a hazard.
    pub fn load_and_start<N: NocId>(
        &mut self,
        window: &Window,
        tile: NocCoord<N>,
        core: Core,
        image: &[u8],
        load_address: u64,
    ) -> Result<()> {
        if load_address + image.len() as u64 > tensix::L1_SIZE {
            return Err(TransportError::OutOfBounds {
                bar: crate::Bar::Bar0,
                offset: load_address,
                len: image.len() as u64,
            });
        }

        self.set_core_reset(window, tile, core, true)?;
        self.write(window, tile, load_address, image)?;
        // L1 survives a run on silicon, so the previous image's `RUNNING` or
        // `DONE` is still in the mailbox, and a wait for the new image would be
        // satisfied by the old one before it has executed an instruction. The
        // simulator hides this by starting every run from a fresh chip.
        self.write32(window, tile, mailbox::STATUS, 0)?;

        // B cannot be redirected; for every other core, say explicitly where to
        // start rather than relying on the image happening to sit at the default.
        if core.reset_pc_register().is_some() {
            self.set_reset_pc(window, tile, core, load_address as u32)?;
        } else if load_address != core.default_reset_pc() as u64 {
            return Err(TransportError::Misaligned {
                bar: crate::Bar::Bar0,
                offset: load_address,
                len: image.len() as u64,
                reason: "RISCV B always starts at L1 offset 0",
            });
        }

        self.set_core_reset(window, tile, core, false)?;
        Ok(())
    }

    /// Release the Tensix backend — unpackers, packers, Matrix Unit, Vector Unit,
    /// and the rest — from soft reset.
    ///
    /// A precondition for any compute. While bit 10 is held, new Vector Unit
    /// instructions "will not start (they might or might not be silently
    /// discarded)" (`SoftReset.md`); silent discard is the dangerous half, because
    /// the symptom is a stale `Dst` rather than an error.
    ///
    /// Leaves the five baby RISC-V cores alone, so it composes with
    /// [`Device::load_and_start`] in either order.
    pub fn release_tensix_backend<N: NocId>(
        &mut self,
        window: &Window,
        tile: NocCoord<N>,
    ) -> Result<()> {
        let current = self.read_soft_reset(window, tile)?;
        self.write32(
            window,
            tile,
            tensix::SOFT_RESET_0,
            current & !tensix::BACKEND_RESET_MASK,
        )
    }

    /// Hold the Tensix backend in soft reset: the inverse of
    /// [`Device::release_tensix_backend`], leaving the baby RISC-V cores alone.
    ///
    /// For negative controls on silicon, where the backend's state outlives the
    /// process that set it -- so "never released it" is not the same as "held".
    /// **Fatal on the simulator**, which accepts only the baby RISC-V bits of
    /// `SOFT_RESET_0` (divergence row 16).
    pub fn hold_tensix_backend<N: NocId>(
        &mut self,
        window: &Window,
        tile: NocCoord<N>,
    ) -> Result<()> {
        let current = self.read_soft_reset(window, tile)?;
        self.write32(
            window,
            tile,
            tensix::SOFT_RESET_0,
            current | tensix::BACKEND_RESET_MASK,
        )
    }

    /// Read the firmware mailbox status word.
    pub fn read_status<N: NocId>(&mut self, window: &Window, tile: NocCoord<N>) -> Result<u32> {
        self.read32(window, tile, mailbox::STATUS)
    }

    /// Advance the clock until `predicate` accepts the status word.
    ///
    /// On the simulator nothing happens outside `tick`, so a poll loop that does
    /// not tick spins against a frozen device forever. The budget is in simulated
    /// cycles for that reason, rather than wall-clock time. On silicon, where
    /// `tick` does nothing, the same budget is read as nanoseconds and floored at
    /// [`SILICON_MIN_WAIT`] -- see [`Transport::is_simulated`].
    pub fn wait_for_status<N: NocId>(
        &mut self,
        window: &Window,
        tile: NocCoord<N>,
        budget_cycles: u64,
        mut predicate: impl FnMut(u32) -> bool,
    ) -> Result<std::result::Result<u32, WaitError>> {
        let mut waited = 0u64;
        let deadline = (!self.transport().is_simulated()).then(|| {
            let started = std::time::Instant::now();
            let limit = std::time::Duration::from_nanos(budget_cycles).max(SILICON_MIN_WAIT);
            (started, started + limit)
        });
        loop {
            let value = self.read_status(window, tile)?;
            if predicate(value) {
                return Ok(Ok(value));
            }
            // Report a panic as itself rather than letting it time out: the
            // firmware has already said what went wrong.
            if value == status::PANICKED {
                let code = self.read32(window, tile, mailbox::PANIC_CODE)?;
                return Ok(Err(WaitError::Panicked { code }));
            }
            let expired = match deadline {
                Some((_, end)) => std::time::Instant::now() >= end,
                None => waited >= budget_cycles,
            };
            if expired {
                return Ok(Err(WaitError::TimedOut {
                    waited_cycles: waited,
                    wall_clock: deadline.map(|(started, _)| started.elapsed()),
                    last_status: value,
                }));
            }
            self.tick(CYCLES_PER_POLL);
            waited += CYCLES_PER_POLL as u64;
        }
    }

    /// Read this core's speculative `pc` snapshot (`BabyRISCV/README.md:163-173`).
    ///
    /// Taken as instructions leave the frontend, so it may name an instruction that
    /// never executes. Good for liveness, useless as a precise fault PC.
    pub fn read_pc_snapshot<N: NocId>(
        &mut self,
        window: &Window,
        tile: NocCoord<N>,
        core: Core,
    ) -> Result<u32> {
        self.read32(window, tile, core.pc_snapshot())
    }
}
