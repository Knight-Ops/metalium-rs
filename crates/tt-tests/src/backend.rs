//! Where a gate's [`Device`] comes from, and what "isolated" means on each target.
//!
//! The simulator and silicon differ in one way that matters more than the
//! transport: the simulator hands out a *fresh chip* per call, and silicon does
//! not. `Dst` has no power-on reset value and nothing scrubs it (`Dst.md:15`), L1
//! keeps whatever the last run left there, and every baby RISC-V keeps running
//! whatever it was running. On the simulator that isolation is free; here it has
//! to be performed, and the cost of forgetting is a gate that passes on the
//! previous run's data.
//!
//! [`Device`]: tt_device::Device

use tt_device::Device;

/// Which Tensix tile the shared gates use.
///
/// Also the tile [`silicon`] scrubs and registers a crash-cleanup write against,
/// so a gate that runs somewhere else is responsible for its own hygiene.
pub const GATE_TILE: (u8, u8) = (3, 4);

#[cfg(feature = "silicon")]
pub use silicon::*;
#[cfg(not(feature = "silicon"))]
pub use simulator::*;

#[cfg(not(feature = "silicon"))]
mod simulator {
    use super::*;
    use tt_isa::noc::grid::Tensix;
    use tt_ttsim::{fork_scope, Simulator};

    /// Which Tensix tiles the simulator has: all of them.
    ///
    /// ttsim models an unharvested Blackhole, which is exactly why the gates could
    /// not discover harvesting here and why the first silicon run addressed a
    /// fused-off tile. Stated as a value rather than left implicit so that a gate
    /// written against the simulator is already written against a *chip's* grid,
    /// and gains nothing to change when it runs on silicon.
    pub fn tensix_grid(_dev: &mut Dev<'_>) -> Tensix {
        Tensix::FULL
    }

    /// A device backed by the simulator, for the lifetime of one `fork_scope`.
    pub type Dev<'a> = Device<tt_ttsim::LibTtsim<'a>>;

    /// Run `f` against a fresh simulator, inside a fork.
    ///
    /// A fresh simulator per call is not just isolation from ttsim's `_Exit`:
    /// `Dst` has no power-on reset value, so two runs sharing a simulator leave
    /// the second reading the first's leftovers.
    #[track_caller]
    pub fn in_device(f: impl FnOnce(&mut Dev<'_>)) {
        if let Err(e) = fork_scope(|| {
            let mut sim =
                Simulator::open().unwrap_or_else(|e| panic!("could not open simulator: {e}"));
            let mut dev = Device::open(sim.transport()).unwrap_or_else(|e| panic!("{e}"));
            f(&mut dev);
        }) {
            panic!("{e}");
        }
    }

    /// Did `f` run to completion, or did ttsim refuse something in it?
    ///
    /// The discovery primitive: a refusal becomes `false` rather than a dead
    /// runner, so a probe can assert that the simulator declines a configuration
    /// *and* that a control of the same shape survives.
    pub fn survives(f: impl FnOnce(&mut Dev<'_>)) -> bool {
        fork_scope(|| {
            let mut sim = Simulator::open().unwrap();
            let mut dev = Device::open(sim.transport()).unwrap();
            f(&mut dev);
        })
        .is_ok()
    }
}

#[cfg(feature = "silicon")]
mod silicon {
    use super::*;
    use tt_isa::noc::grid::Tensix;
    use tt_isa::noc::{Noc0, NocCoord};
    use tt_isa::tensix::{self, Core};
    use tt_kmd::Kmd;
    use tt_ttsim::fork_scope;

    /// Which Tensix tiles this card has, read from its ARC.
    ///
    /// Never assumed. A p150a commonly has two columns fused off, and a read to a
    /// fused-off tile does not fail — it hangs the NoC, and the recovery for that
    /// is a chip reset that drops the PCIe link. On a card passed through to a VM
    /// that takes the host down with it.
    pub fn tensix_grid(dev: &mut Dev<'_>) -> Tensix {
        let w = dev
            .alloc_window(tt_device::tlb::WindowKind::TwoMib)
            .unwrap_or_else(|e| panic!("could not get a window to read telemetry: {e}"));
        let grid = dev.tensix_grid(&w);
        dev.free_window(w);
        grid.unwrap_or_else(|e| panic!("could not read this chip's Tensix grid: {e}"))
    }

    /// A device backed by a real card.
    ///
    /// The lifetime is vestigial — `Kmd` owns its file descriptor and borrows
    /// nothing — but it keeps the alias interchangeable with the simulator's, so
    /// a gate written as `fn(&mut Dev<'_>)` compiles against either.
    pub type Dev<'a> = Device<Kmd>;

    /// Which card to run against: `TT_SILICON_DEVICE`, defaulting to 0.
    pub const DEVICE_ENV: &str = "TT_SILICON_DEVICE";

    pub fn device_index() -> u16 {
        std::env::var(DEVICE_ENV)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    /// Open the card and leave it in a known state.
    ///
    /// The order of the first three steps is the whole point. Reading the chip's
    /// grid comes before *any* access to a Tensix tile, including the driver's
    /// registered cleanup write, because a fused-off tile does not reject a write
    /// — it hangs the NoC, and a cleanup write registered against one would fire
    /// on every close from then on, including the close that follows the hang.
    pub fn open() -> Dev<'static> {
        let index = device_index();
        let kmd = Kmd::open(index)
            .unwrap_or_else(|e| panic!("could not open /dev/tenstorrent/{index}: {e}"));

        let mut dev = Device::open(kmd).unwrap_or_else(|e| panic!("{e}"));

        // Step 1: ask the ARC what this chip is. Touches only the ARC tile, which
        // is not harvestable and whose coordinate means the same thing whether or
        // not the NoC is translating -- so this is safe to do knowing neither.
        let grid = tensix_grid(&mut dev);

        // Step 2: refuse to proceed if the gates' own tile was fused off on this
        // particular ASIC. Panicking here costs a test run; addressing it costs
        // the host.
        let (x, y) = GATE_TILE;
        assert!(
            grid.contains(x, y),
            "the gates' tile ({x},{y}) is fused off on /dev/tenstorrent/{index}: \
             this chip has {} Tensix columns, so X must be one of {:?}. Pick a \
             GATE_TILE inside the surviving columns.",
            grid.enabled_column_count(),
            grid.columns().collect::<Vec<_>>()
        );

        // Step 3: register the crash-cleanup write, now that its target is known
        // to exist. Registered before anything is *started*, because its whole
        // purpose is to cover the cases where we never get to run cleanup code: a
        // panic that aborts, a segfault, the OOM killer. The driver performs this
        // write when the file descriptor closes, however it closes.
        dev.transport()
            .set_cleanup_write(x, y, 0, tensix::SOFT_RESET_0, ALL_BABIES_HELD)
            .unwrap_or_else(|e| panic!("could not register the crash-cleanup write: {e}"));

        scrub(&mut dev);
        dev
    }

    /// Every baby RISC-V held in reset.
    ///
    /// Assembled from `Core`'s own bits rather than written as a literal, so it
    /// cannot drift from the masks `set_core_reset` uses.
    const ALL_BABIES_HELD: u32 = Core::B.soft_reset_mask()
        | Core::T0.soft_reset_mask()
        | Core::T1.soft_reset_mask()
        | Core::T2.soft_reset_mask()
        | Core::NC.soft_reset_mask();

    /// Put the gate tile back to a known state.
    ///
    /// Only the cores, for now. `Dst` is the other piece of state that survives a
    /// run, and scrubbing it needs a Tensix program rather than a register write;
    /// the gates that read `Dst` pre-fill their dump area with a sentinel, which
    /// catches a stale datum but does not remove it.
    pub fn scrub(dev: &mut Dev<'_>) {
        let (x, y) = GATE_TILE;
        let tile: NocCoord<Noc0> = NocCoord::new(x, y).unwrap();
        // Not `.unwrap()`: the way this fails in practice is a gate that took
        // windows and dropped them instead of freeing them, since `Window` has no
        // `Drop` that reaches the free list. The bare `OutOfBounds` that produces
        // says nothing about why, and the simulator never reaches this code at all.
        let w = dev
            .alloc_window(tt_device::tlb::WindowKind::TwoMib)
            .unwrap_or_else(|e| {
                panic!(
                    "no TLB window left to scrub the gate tile with: {e}. A gate \
                     that allocates windows must `free_window` them; dropping a \
                     `Window` leaks it from the free list."
                )
            });
        dev.write32(&w, tile, tensix::SOFT_RESET_0, ALL_BABIES_HELD)
            .unwrap_or_else(|e| panic!("could not hold the gate tile's cores in reset: {e}"));
        dev.free_window(w);
    }

    /// Run `f` against the card, inside a fork.
    ///
    /// The fork is no longer about surviving ttsim's `_Exit` — silicon does not
    /// terminate the process — but it earns its keep twice over here. A gate that
    /// wedges a core cannot take the runner with it, and the child's file
    /// descriptor closing is what fires the driver's cleanup write.
    #[track_caller]
    pub fn in_device(f: impl FnOnce(&mut Dev<'_>)) {
        if let Err(e) = fork_scope(|| {
            let mut dev = open();
            f(&mut dev);
            scrub(&mut dev);
        }) {
            panic!("{e}");
        }
    }

    /// Did `f` run to completion?
    ///
    /// The same shape as the simulator's, but different evidence. There it means
    /// "ttsim did not refuse"; here it means "the child exited cleanly", which a
    /// hung core does not.
    pub fn survives(f: impl FnOnce(&mut Dev<'_>)) -> bool {
        fork_scope(|| {
            let mut dev = open();
            f(&mut dev);
            scrub(&mut dev);
        })
        .is_ok()
    }
}
