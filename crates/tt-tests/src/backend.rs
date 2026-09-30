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

/// Fail unless this build's [`Dev`] reaches a real card.
///
/// For the silicon-only twins. Several of them once compiled under
/// `--features silicon` while going through a file-local harness that opened the
/// *simulator*, so they would have run against ttsim and passed or failed on its
/// say-so. Asked of the device type itself rather than of the feature flag, so it
/// checks what the gate will actually talk to.
#[track_caller]
pub fn assert_on_silicon() {
    let dev = std::any::type_name::<Dev<'static>>();
    assert!(
        dev.contains("Kmd"),
        "this gate is silicon-only, but it is running against {dev}"
    );
}

#[cfg(feature = "silicon")]
pub use silicon::*;
#[cfg(not(feature = "silicon"))]
pub use simulator::*;

#[cfg(not(feature = "silicon"))]
mod simulator {
    use super::*;
    use tt_isa::noc::grid::Tensix;
    use tt_isa::noc::{Noc0, NocCoord};
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

    /// Is this build's [`Dev`] a real card? See the silicon twin.
    pub const ON_SILICON: bool = false;

    /// Let `cycles` of device time pass: exactly that many simulated clocks.
    pub fn advance(dev: &mut Dev<'_>, cycles: u32) {
        dev.tick(cycles);
    }

    /// A Tensix tile a gate may use. On the simulator every Tensix coordinate is
    /// present, so this checks geometry only; see the silicon twin for why the
    /// call exists at all.
    #[track_caller]
    pub fn tile(_dev: &mut Dev<'_>, x: u8, y: u8) -> NocCoord<Noc0> {
        assert!(
            Tensix::FULL.contains(x, y),
            "({x},{y}) is not a Tensix coordinate"
        );
        NocCoord::new(x, y).unwrap()
    }

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

    /// Is this build's [`Dev`] a real card?
    ///
    /// For the silicon-only twins to assert, so that a twin which ends up
    /// running against the simulator -- the state several of them were in until
    /// every gate went through this module -- fails instead of passing on
    /// ttsim's say-so.
    pub const ON_SILICON: bool = true;

    // Tiles other than [`GATE_TILE`] that this process has started using.
    //
    // Each carries its own open `Kmd`, held only for the cleanup write it
    // registered. tt-kmd keeps exactly **one** cleanup write per open file
    // (`chardev.c:539` overwrites `priv->noc_cleanup`), so covering a second
    // tile takes a second file descriptor. Thread-local because `Kmd` owns raw
    // mappings; every gate runs single-threaded inside its own fork anyway.
    thread_local! {
        static CLAIMED: std::cell::RefCell<Vec<(u16, NocCoord<Noc0>, Kmd)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// A Tensix tile a gate may use, checked against *this chip's* grid.
    ///
    /// The only way a gate should name a tile other than [`GATE_TILE`]. It
    /// refuses a coordinate the chip does not have before anything is sent to it
    /// -- `step3_heartbeat` and `step4_tensix` both named `(16, 11)`, fused off on both
    /// cards here, and addressing it hangs the NoC and takes the host down (row
    /// 35 of the divergence log). It also registers the same crash-cleanup write
    /// [`open`] registers for the gate tile, through a file descriptor of its
    /// own, and [`scrub`] holds the tile's cores in reset at the end.
    #[track_caller]
    pub fn tile(dev: &mut Dev<'_>, x: u8, y: u8) -> NocCoord<Noc0> {
        // The card this device is, not `DEVICE_ENV`: a gate holding both cards
        // claims tiles on each.
        let index = dev.chip().0;
        let grid = tensix_grid(dev);
        assert!(
            grid.contains(x, y),
            "({x},{y}) is not a Tensix tile on /dev/tenstorrent/{index}: this chip has \
             {} Tensix columns, so X must be one of {:?}",
            grid.enabled_column_count(),
            grid.columns().collect::<Vec<_>>()
        );
        let coord = NocCoord::new(x, y).unwrap();
        let claimed = CLAIMED.with(|c| {
            c.borrow()
                .iter()
                .any(|(i, t, _)| *i == index && *t == coord)
        });
        if (x, y) != GATE_TILE && !claimed {
            let guard = Kmd::open(index)
                .unwrap_or_else(|e| panic!("could not open /dev/tenstorrent/{index}: {e}"));
            guard
                .set_cleanup_write(x, y, 0, tensix::SOFT_RESET_0, ALL_BABIES_HELD)
                .unwrap_or_else(|e| {
                    panic!("could not register a cleanup write for ({x},{y}): {e}")
                });
            CLAIMED.with(|c| c.borrow_mut().push((index, coord, guard)));
            // Start from the state the simulator starts from: every core held.
            // Whatever the last process left running on this tile is stopped
            // before the gate looks at it.
            let w = dev
                .alloc_window(tt_device::tlb::WindowKind::TwoMib)
                .unwrap_or_else(|e| panic!("no TLB window left to claim ({x},{y}): {e}"));
            reset_tile(dev, &w, coord);
            dev.free_window(w);
        }
        coord
    }

    /// Let roughly `cycles` of device time pass.
    ///
    /// `tick` does nothing on silicon, so a gate that ticks between two samples
    /// to see a counter move would otherwise sample back to back. Converted at a
    /// nominal 1 GHz: the gates only need *some* time to pass, not a measured
    /// amount, and no gate may draw a timing conclusion from this.
    pub fn advance(_dev: &mut Dev<'_>, cycles: u32) {
        std::thread::sleep(std::time::Duration::from_nanos(cycles as u64));
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
        open_card(device_index())
    }

    /// [`open`], for a card named explicitly rather than by [`DEVICE_ENV`]: for
    /// the gates that hold both cards at once.
    pub fn open_card(index: u16) -> Dev<'static> {
        report_watchdog();
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

    /// Say, once per card opened, whether the ARC watchdog is armed.
    ///
    /// Armed, a gate that hangs the NoC becomes a chip reset that drops the PCIe
    /// link, which on this passed-through VM is the host. That is a reason to fix
    /// the access that hangs -- as `Device`'s local-RAM guard now does -- not to
    /// refuse to run, so this only reports.
    fn report_watchdog() {
        match tt_kmd::auto_reset_timeout() {
            Ok(0) => {}
            Ok(t) => eprintln!(
                "note: ARC watchdog armed (auto_reset_timeout={t}); a NoC hang will \
                 reset the chip and drop the PCIe link"
            ),
            Err(e) => eprintln!(
                "note: cannot read {}: {e}",
                tt_kmd::AUTO_RESET_TIMEOUT_PARAM
            ),
        }
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

    /// Hold every core on `tile`, and put its Tensix backend through a reset
    /// pulse, leaving the backend released and the cores held.
    ///
    /// Holding the cores alone is not enough. On silicon a program that hangs
    /// the coprocessor -- a `STALLWAIT` on an unpacker that never finishes --
    /// leaves the backend stuck after its core is stopped, and every later gate
    /// on the tile then times out: the first run of the Stage 2 probes wedged the
    /// gate tile that way and took the rest of the suite down with it. The pulse
    /// is `SoftReset.md`'s remedy: entering reset aborts in-flight
    /// `UNPACR`/`PACR`, Matrix Unit and Vector Unit work, resets the `Src` bank
    /// ownership and zeroes the THCON configuration; the cores being held
    /// discards anything queued in their Tensix FIFOs. `Dst` survives it.
    ///
    /// Silicon only: ttsim accepts only the baby RISC-V bits of `SOFT_RESET_0`
    /// (divergence row 16).
    fn reset_tile(dev: &mut Dev<'_>, w: &tt_device::Window, tile: NocCoord<Noc0>) {
        dev.write32(
            w,
            tile,
            tensix::SOFT_RESET_0,
            ALL_BABIES_HELD | tensix::BACKEND_RESET_MASK,
        )
        .unwrap_or_else(|e| panic!("could not reset {tile:?}'s backend: {e}"));
        dev.write32(w, tile, tensix::SOFT_RESET_0, ALL_BABIES_HELD)
            .unwrap_or_else(|e| panic!("could not release {tile:?}'s backend: {e}"));
    }

    /// Put the gate tile, and every tile claimed through [`tile`], back to a
    /// known state: cores held, backend reset and released ([`reset_tile`]).
    ///
    /// `Dst` is the piece of state that survives this, and scrubbing it needs a
    /// Tensix program rather than a register write; the gates that read `Dst`
    /// pre-fill their dump area with a sentinel, which catches a stale datum but
    /// does not remove it.
    pub fn scrub(dev: &mut Dev<'_>) {
        let (x, y) = GATE_TILE;
        let mut tiles: Vec<NocCoord<Noc0>> = vec![NocCoord::new(x, y).unwrap()];
        let index = dev.chip().0;
        tiles.extend(CLAIMED.with(|c| {
            c.borrow()
                .iter()
                .filter(|(i, _, _)| *i == index)
                .map(|(_, t, _)| *t)
                .collect::<Vec<_>>()
        }));
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
        for tile in tiles {
            reset_tile(dev, &w, tile);
        }
        dev.free_window(w);
    }

    /// Run `f` against the card, inside a fork.
    ///
    /// The fork is no longer about surviving ttsim's `_Exit` — silicon does not
    /// terminate the process — but it earns its keep twice over here. A gate that
    /// wedges a core cannot take the runner with it, and the child's file
    /// descriptor closing is what fires the driver's cleanup write.
    /// Reset the gate thread's own Tensix state before a gate runs:
    /// [`crate::datapath::thread_state_reset`], run through the harness on the
    /// gate tile. The per-thread half of the scrub that [`reset_tile`] cannot
    /// do from outside.
    fn reset_thread_state(dev: &mut Dev<'_>) {
        let program = crate::datapath::thread_state_reset();
        let _ = crate::harness::run(dev, &crate::harness::Run::new(&program).dump_rows(0));
    }

    #[track_caller]
    pub fn in_device(f: impl FnOnce(&mut Dev<'_>)) {
        if let Err(e) = fork_scope(|| {
            let mut dev = open();
            reset_thread_state(&mut dev);
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
            reset_thread_state(&mut dev);
            f(&mut dev);
            scrub(&mut dev);
        })
        .is_ok()
    }
}
