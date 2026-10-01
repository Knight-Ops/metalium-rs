//! Owning a chip for compute: the bring-up order, and the reset between runs.
//!
//! Established by the silicon campaign in `tt-tests/src/backend.rs` and moved
//! here so a backend gets the same sequence the gates do. Every step exists
//! because its absence cost a run, and in two cases the host:
//!
//! 1. **Read the Tensix grid from the ARC before touching any Tensix tile.** A
//!    fused-off tile does not reject an access; nothing answers, the NoC hangs,
//!    and the chip reset that recovers it drops the PCIe link (divergence row 35).
//! 2. **Refuse a tile this chip does not have**, as an error.
//! 3. **Register the driver's crash-cleanup write**, now that its target is known
//!    to exist -- registered against a fused-off tile it would fire on every close.
//! 4. **Reset the tile**: hold every core, pulse the Tensix backend, and put the
//!    per-thread state (`Config`, `ThreadConfig`, RWCs, ADCs) back to zero, which
//!    nothing outside the tile can do and which otherwise outlives the program
//!    that set it (rows 47 and 49).
//!
//! On the simulator steps 1, 3 and 4 have nothing to do: ttsim models an
//! unharvested chip, has no driver, starts every tile at zero, and refuses both
//! the backend reset bits (row 16) and `STATE_RESET_EN` (row 49).
//!
//! # Many tiles
//!
//! A session computes on one or more tiles ([`TileChoice`]), each a *unit*
//! with its own resident roles and data mover, steps 2-4 done for each. A
//! GDDR op ([`crate::tensor`]) is a set of independent jobs, dealt to the
//! units round-robin and run in waves: one step is started on every unit,
//! then every unit is waited for. The units compute at the same time -- on
//! the simulator because one clock moves every tile while the host waits on
//! any of them (divergence row 23), on silicon because they are separate
//! tiles -- and since no job's arithmetic depends on where it runs, the
//! result is the same bits whatever the number of units.

use tt_device::tlb::WindowKind;
use tt_device::{Device, Transport, TransportError};
use tt_isa::noc::grid::Tensix;
use tt_isa::noc::{Noc0, NocCoord};
use tt_isa::tensix::{self, Core};

use crate::datapath;
use crate::dm::DataMover;
use crate::matmul::{self, Fidelity, SrcRoute};
use crate::runtime::{self, Kernel, Resident, RoleImages, RunError, Schedule};
use crate::tensor::{self, DramAlloc, DramTensor, Step, TensorError};

/// Every baby RISC-V held in reset: what the cleanup write leaves behind, and
/// the resting state between runs.
///
/// Assembled from `Core`'s own bits so it cannot drift from `set_core_reset`.
pub const ALL_BABIES_HELD: u32 = Core::B.soft_reset_mask()
    | Core::T0.soft_reset_mask()
    | Core::T1.soft_reset_mask()
    | Core::T2.soft_reset_mask()
    | Core::NC.soft_reset_mask();

/// Simulated cycles each role may take in the reset program. On silicon the
/// runner's one-second floor applies.
const RESET_BUDGET: u64 = 400_000;

/// Which of this chip's Tensix tiles a session computes on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileChoice {
    /// This one, or an error if the chip does not have it.
    Exactly(u8, u8),
    /// The first surviving tile in translated order (lowest column, then row).
    First,
    /// The first `n` surviving tiles, row by row (`grid::Tensix::tiles`), or
    /// an error if the chip has fewer.
    Count(usize),
    /// Every surviving tile.
    All,
}

/// Why a session could not be opened.
#[derive(Debug)]
pub enum SessionError {
    Transport(TransportError),
    /// The requested tile is fused off (or never existed) on this chip.
    NoSuchTile {
        x: u8,
        y: u8,
        columns: Vec<u8>,
    },
    /// More tiles were asked for than this chip has.
    TooFewTiles {
        asked: usize,
        have: usize,
    },
    /// The reset program did not run.
    Reset(RunError),
}

impl From<TransportError> for SessionError {
    fn from(e: TransportError) -> Self {
        SessionError::Transport(e)
    }
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Transport(e) => write!(f, "{e}"),
            SessionError::NoSuchTile { x, y, columns } => write!(
                f,
                "({x},{y}) is not a Tensix tile on this chip: it has {} Tensix columns, \
                 so X must be one of {columns:?}",
                columns.len()
            ),
            SessionError::TooFewTiles { asked, have } => write!(
                f,
                "{asked} Tensix tiles asked for, and this chip has {have}"
            ),
            SessionError::Reset(e) => write!(f, "resetting the tile: {e}"),
        }
    }
}

impl std::error::Error for SessionError {}

/// This chip's Tensix grid, asked of the ARC.
///
/// Touches only the ARC tile, whose coordinate means the same thing whether or
/// not translation is on, so it is safe before anything else is known. On the
/// simulator, [`Tensix::FULL`]: ttsim models an unharvested chip (row 35) and no
/// telemetry.
pub fn tensix_grid<T: Transport>(dev: &mut Device<T>) -> Result<Tensix, TransportError> {
    if dev.transport().is_simulated() {
        return Ok(Tensix::FULL);
    }
    let w = dev.alloc_window(WindowKind::TwoMib)?;
    dev.tensix_grid(&w)
}

/// Hold every core on `tile` and pulse its Tensix backend, leaving the backend
/// released and the cores held.
///
/// Holding the cores alone is not enough: a program that hangs the coprocessor
/// leaves the backend stuck after its core is stopped. Entering reset aborts
/// in-flight unpacker, packer, Matrix Unit and Vector Unit work, resets `Src`
/// bank ownership and zeroes the THCON configuration (`SoftReset.md`). `Dst`
/// survives it. A no-op on the simulator (row 16).
pub fn reset_tile<T: Transport>(
    dev: &mut Device<T>,
    tile: NocCoord<Noc0>,
) -> Result<(), TransportError> {
    if dev.transport().is_simulated() {
        return Ok(());
    }
    let w = dev.alloc_window(WindowKind::TwoMib)?;
    dev.write32(
        &w,
        tile,
        tensix::SOFT_RESET_0,
        ALL_BABIES_HELD | tensix::BACKEND_RESET_MASK,
    )?;
    dev.write32(&w, tile, tensix::SOFT_RESET_0, ALL_BABIES_HELD)
}

/// Zero `Config` and all three threads' per-thread state on `tile`
/// ([`datapath::thread_state_reset`]). A no-op on the simulator, which starts
/// every run from zero and refuses parts of the program (rows 36 and 49).
pub fn reset_thread_state<T: Transport>(
    dev: &mut Device<T>,
    tile: NocCoord<Noc0>,
    images: &RoleImages<'_>,
) -> Result<(), RunError> {
    if dev.transport().is_simulated() {
        return Ok(());
    }
    let program = datapath::thread_state_reset();
    // Thread 0 first releases anything a failed kernel left blocked on a
    // semaphore, which the backend pulse does not (row 65); otherwise thread 1
    // or 2's reset would queue behind it forever.
    let first = [datapath::release_semaphores(), program.clone()].concat();
    let kernel = Kernel {
        dump_rows: 0,
        ..Kernel::new([&first, &program, &program], Schedule::InOrder)
    };
    runtime::run(dev, tile, images, &kernel, RESET_BUDGET).map(|_| ())
}

/// `A[m,k] @ B[k,n]`, row-major, on `tile`, in as many runs as it takes
/// ([`matmul::matmul_chunked`]), each preceded by the tile reset
/// ([`reset_tile`], [`reset_thread_state`]) so no run inherits another's state.
#[allow(clippy::too_many_arguments)]
pub fn matmul_on<T: Transport>(
    dev: &mut Device<T>,
    tile: NocCoord<Noc0>,
    images: &RoleImages<'_>,
    a: &[f32],
    b: &[f32],
    mkn: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    budget: u64,
) -> Result<Vec<f32>, RunError> {
    matmul::matmul_chunked(a, b, mkn, route, fidelity, |a, b, mkn| {
        reset_tile(dev, tile)?;
        reset_thread_state(dev, tile, images)?;
        matmul::matmul(dev, tile, images, a, b, mkn, route, fidelity, budget)
    })
}

/// A chip opened for compute on one or more Tensix tiles, in the order the
/// module documentation gives.
pub struct Session<T: Transport> {
    dev: Device<T>,
    units: Vec<Unit>,
    grid: Tensix,
    images: RoleImages<'static>,
    /// Every run's phases since the last [`Session::take_profile`].
    profile: runtime::Profile,
    /// GDDR, once [`Session::enable_dram`] has been called.
    dram: Option<DramState>,
    /// Whatever keeps each unit's crash-cleanup write registered beyond the
    /// first (on silicon, one driver file descriptor per tile: the driver keeps
    /// one cleanup write per descriptor).
    _cleanup: Vec<Box<dyn std::any::Any>>,
}

/// One tile a session computes on.
struct Unit {
    tile: NocCoord<Noc0>,
    /// The role images, resident since the last [`Session::prepare`]
    /// (`runtime::Resident`): one reset and one load per session rather than
    /// per run. `None` only between a failed run and the unit's next prepare.
    resident: Option<Resident<Noc0>>,
    /// The data mover on this tile's RISCV B, started when first needed. It
    /// is reached through the resident roles' window.
    mover: Option<DataMover<Noc0>>,
    /// Steps completed on this tile, for a gate that wants to know every
    /// unit did its share.
    steps: u64,
}

/// What a session needs to keep tensors in GDDR: the chip's channels, an
/// allocator, the data mover image, and a window for each size of access.
struct DramState {
    alloc: DramAlloc,
    dram: tt_isa::dram::Dram,
    image: &'static [u8],
    w4: tt_device::Window,
}

/// A step of a wave, started on one unit and waiting to be finished.
enum Started<'s> {
    List,
    Matmul(Kernel<'s>),
}

impl<T: Transport> Session<T> {
    /// Bring `dev` up for compute on the tiles `choice` names.
    ///
    /// `register_cleanup` is step 3: given the transport and a chosen tile, it
    /// arranges for [`ALL_BABIES_HELD`] to be written to the tile's
    /// `SOFT_RESET_0` however the process ends, returning whatever must be
    /// kept alive for that to stay true. It is called once per tile, only once
    /// every tile is known to exist. [`Session::open_card`] supplies the
    /// driver's.
    pub fn open(
        mut dev: Device<T>,
        images: RoleImages<'static>,
        choice: TileChoice,
        mut register_cleanup: impl FnMut(
            &mut T,
            NocCoord<Noc0>,
        )
            -> Result<Option<Box<dyn std::any::Any>>, TransportError>,
    ) -> Result<Self, SessionError> {
        let grid = tensix_grid(&mut dev)?;
        let have = grid.tile_count();
        let tiles: Vec<(u8, u8)> = match choice {
            TileChoice::Exactly(x, y) => vec![(x, y)],
            TileChoice::First => {
                let t = grid
                    .tiles::<Noc0>()
                    .min_by_key(|t| (t.x(), t.y()))
                    .expect("a chip with no Tensix tiles would have failed telemetry");
                vec![(t.x(), t.y())]
            }
            TileChoice::Count(n) if n == 0 || n > have => {
                return Err(SessionError::TooFewTiles { asked: n, have })
            }
            TileChoice::Count(n) => grid
                .tiles::<Noc0>()
                .take(n)
                .map(|t| (t.x(), t.y()))
                .collect(),
            TileChoice::All => grid.tiles::<Noc0>().map(|t| (t.x(), t.y())).collect(),
        };
        for &(x, y) in &tiles {
            if !grid.contains(x, y) {
                return Err(SessionError::NoSuchTile {
                    x,
                    y,
                    columns: grid.columns().collect(),
                });
            }
        }
        let mut cleanup = Vec::new();
        let mut units = Vec::with_capacity(tiles.len());
        for (x, y) in tiles {
            let tile = NocCoord::new(x, y).expect("a grid tile is a NoC coordinate");
            cleanup.extend(register_cleanup(dev.transport(), tile)?);
            units.push(Unit {
                tile,
                resident: None,
                mover: None,
                steps: 0,
            });
        }
        let mut session = Session {
            dev,
            units,
            grid,
            images,
            profile: runtime::Profile::default(),
            dram: None,
            _cleanup: cleanup,
        };
        session.prepare().map_err(SessionError::Reset)?;
        Ok(session)
    }

    /// Put every tile back to a known state: step 4 again, then the role
    /// images loaded and left resident.
    pub fn prepare(&mut self) -> Result<(), RunError> {
        for u in 0..self.units.len() {
            self.prepare_unit(u)?;
        }
        Ok(())
    }

    fn prepare_unit(&mut self, u: usize) -> Result<(), RunError> {
        let Session {
            dev, units, images, ..
        } = self;
        let unit = &mut units[u];
        // The reset holds RISCV B too; GDDR contents survive, the mover does
        // not.
        unit.mover = None;
        if let Some(r) = unit.resident.take() {
            r.stop(dev, images)?;
        }
        reset_tile(dev, unit.tile)?;
        reset_thread_state(dev, unit.tile, images)?;
        unit.resident = Some(Resident::start(dev, unit.tile, images, RESET_BUDGET)?);
        Ok(())
    }

    /// Keep tensors in GDDR from now on: read the chip's channels and set up an
    /// allocator. `dm_image` is `tt_firmware_images::DM_B`'s bytes, started on
    /// each unit's RISCV B when first needed.
    pub fn enable_dram(&mut self, dm_image: &'static [u8]) -> Result<(), TransportError> {
        if self.dram.is_some() {
            return Ok(());
        }
        let w4 = self.dev.alloc_window(WindowKind::FourGib)?;
        let dram = {
            let w = self.dev.alloc_window(WindowKind::TwoMib)?;
            self.dev.dram_grid(&w)?
        };
        self.dram = Some(DramState {
            alloc: DramAlloc::new(&dram),
            dram,
            image: dm_image,
            w4,
        });
        Ok(())
    }

    fn dram_state(&mut self) -> Result<&mut DramState, TensorError> {
        self.dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled on this session".into()))
    }

    /// Upload a row-major `[rows, cols]` matrix to GDDR.
    pub fn upload(
        &mut self,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<DramTensor, TensorError> {
        let Session { dev, dram, .. } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        DramTensor::upload(dev, &d.w4, &mut d.alloc, values, rows, cols)
    }

    /// Download a tensor to row-major values.
    pub fn download(&mut self, t: &DramTensor) -> Result<Vec<f32>, TensorError> {
        let Session { dev, dram, .. } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        t.download(dev, &d.w4)
    }

    /// Give a tensor's slots back.
    pub fn free(&mut self, t: DramTensor) -> Result<(), TensorError> {
        self.dram_state()?.alloc.free(&t.placement);
        Ok(())
    }

    /// Free GDDR bytes on the fullest channel.
    pub fn dram_free_bytes(&self) -> u64 {
        self.dram.as_ref().map_or(0, |d| d.alloc.free_bytes())
    }

    /// Element-wise `a (op) b` in GDDR ([`tensor::eltwise`]), on the data
    /// movers' FP32 units. No Tensix run, so the resident roles are untouched.
    pub fn eltwise(
        &mut self,
        op: tensor::Eltwise,
        a: &DramTensor,
        b: Option<&DramTensor>,
    ) -> Result<DramTensor, TensorError> {
        let units = self.units.len();
        let work = tensor::eltwise(&mut self.dram_state()?.alloc, op, a, b, units)?;
        self.execute(work, RESET_BUDGET)
    }

    /// The sum over rows of `a`, `[1, cols]`, in `burn-flex`'s order
    /// ([`tensor::sum_rows`]).
    pub fn sum_rows(&mut self, a: &DramTensor) -> Result<DramTensor, TensorError> {
        let units = self.units.len();
        let work = tensor::sum_rows(&mut self.dram_state()?.alloc, a, units)?;
        self.execute(work, RESET_BUDGET)
    }

    /// `op(A) @ op(B)` with every operand and the result in GDDR
    /// ([`tensor::matmul_dram`]), on the resident roles.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_dram(
        &mut self,
        a: &DramTensor,
        a_transposed: bool,
        b: &DramTensor,
        b_transposed: bool,
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<DramTensor, TensorError> {
        let units = self.units.len();
        let work = tensor::matmul_dram(
            &mut self.dram_state()?.alloc,
            a,
            a_transposed,
            b,
            b_transposed,
            route,
            fidelity,
            units,
        )?;
        self.execute(work, budget)
    }

    /// Run an op's jobs over the units (see the module documentation) and
    /// return its output, or free the output and return the first error.
    ///
    /// A unit whose step fails is recovered before the error is returned --
    /// its mover restarted for a list, its tile reset and roles restarted for a
    /// kernel -- so the session stays usable. The other units' steps of the
    /// same wave are finished first, so nothing is left running.
    fn execute(&mut self, work: tensor::Work, budget: u64) -> Result<DramTensor, TensorError> {
        let tensor::Work { out, jobs } = work;
        let n = self.units.len();
        let mut queues: Vec<Vec<Step>> = vec![Vec::new(); n];
        for (j, job) in jobs.into_iter().enumerate() {
            let q = &mut queues[j % n];
            for step in job {
                match step {
                    Step::List { what, entries } => {
                        for piece in entries.chunks(tt_isa::dm::LIST_MAX as usize) {
                            q.push(Step::List {
                                what,
                                entries: piece.to_vec(),
                            });
                        }
                    }
                    s => q.push(s),
                }
            }
        }
        let waves = queues.iter().map(Vec::len).max().unwrap_or(0);
        for w in 0..waves {
            let what = match queues.iter().find_map(|q| q.get(w)) {
                Some(Step::List { what, .. }) => *what,
                _ => "matmul compute",
            };
            let failed = tensor::stats::timed(what, || self.wave(&queues, w, budget));
            if let Some((failed, error)) = failed {
                self.recover(&failed);
                if let Ok(d) = self.dram_state() {
                    d.alloc.free(&out.placement);
                }
                return Err(error);
            }
        }
        Ok(out)
    }

    /// Step `w` of every queue that has one: started on every unit, then
    /// finished on every unit. Returns the units that failed, how, and the
    /// first error.
    fn wave(
        &mut self,
        queues: &[Vec<Step>],
        w: usize,
        budget: u64,
    ) -> Option<(Vec<(usize, bool)>, TensorError)> {
        let mut failed = Vec::new();
        let mut first: Option<TensorError> = None;
        let mut started = Vec::new();
        for (u, q) in queues.iter().enumerate() {
            let Some(step) = q.get(w) else { continue };
            match self.start(u, step, budget) {
                Ok(Some(s)) => started.push((u, s)),
                Ok(None) => self.units[u].steps += 1,
                Err(e) => {
                    failed.push((u, matches!(step, Step::Matmul(_))));
                    first.get_or_insert(e);
                }
            }
        }
        for (u, s) in started {
            let kernel = matches!(s, Started::Matmul(_));
            match self.finish(u, s, budget) {
                Ok(()) => self.units[u].steps += 1,
                Err(e) => {
                    failed.push((u, kernel));
                    first.get_or_insert(e);
                }
            }
        }
        first.map(|e| (failed, e))
    }

    /// Start `step` on unit `u`: `Ok(None)` if there was nothing to start.
    fn start<'s>(
        &mut self,
        u: usize,
        step: &'s Step,
        budget: u64,
    ) -> Result<Option<Started<'s>>, TensorError> {
        if self.units[u].resident.is_none() {
            self.prepare_unit(u)?;
        }
        let Session {
            dev,
            units,
            images,
            dram,
            ..
        } = self;
        let unit = &mut units[u];
        let r = unit.resident.as_mut().expect("prepared above");
        match step {
            Step::List { entries, .. } if entries.is_empty() => Ok(None),
            Step::List { entries, .. } => {
                let d = dram
                    .as_ref()
                    .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
                if unit.mover.is_none() {
                    unit.mover = Some(DataMover::start(
                        dev,
                        r.window(),
                        unit.tile,
                        &d.dram,
                        d.image,
                    )?);
                }
                let mover = unit.mover.as_mut().expect("started above");
                mover.submit_list(dev, r.window(), entries)?;
                Ok(Some(Started::List))
            }
            Step::Matmul(roles) => {
                let [unpack, math, pack] = &**roles;
                let kernel = Kernel {
                    // `TILE_SEMAPHORES`: every run leaves them as it found them.
                    restores_semaphores: true,
                    ..Kernel::new(
                        [unpack, math, pack],
                        Schedule::Concurrent(&matmul::TILE_SEMAPHORES),
                    )
                };
                r.submit(dev, images, &kernel, budget)?;
                Ok(Some(Started::Matmul(kernel)))
            }
        }
    }

    /// Wait for a started step on unit `u` to finish.
    fn finish(&mut self, u: usize, s: Started<'_>, budget: u64) -> Result<(), TensorError> {
        let Session {
            dev,
            units,
            images,
            profile,
            ..
        } = self;
        let unit = &mut units[u];
        let r = unit.resident.as_mut().expect("a step was started on it");
        match s {
            Started::List => {
                let mover = unit.mover.as_ref().expect("a list was started on it");
                mover.wait(dev, r.window())?;
            }
            Started::Matmul(kernel) => {
                let o = r.complete(dev, images, &kernel, budget)?;
                profile.phases.extend_from_slice(&o.profile.phases);
            }
        }
        Ok(())
    }

    /// After a failed wave: restart the mover of a unit whose list failed, and
    /// reset and restart the roles of a unit whose kernel did.
    fn recover(&mut self, failed: &[(usize, bool)]) {
        for &(u, kernel) in failed {
            if kernel {
                self.units[u].resident = None;
                if let Err(e) = self.prepare_unit(u) {
                    eprintln!("session: recovery after a failed kernel failed as well: {e}");
                }
            } else {
                self.units[u].mover = None;
            }
        }
    }

    /// Run `kernel` on the first unit's resident roles. A failure resets the
    /// tile and restarts them before it is returned, so the session stays
    /// usable.
    pub fn run(&mut self, kernel: &Kernel<'_>, budget: u64) -> Result<runtime::Outcome, RunError> {
        if self.units[0].resident.is_none() {
            self.prepare_unit(0)?;
        }
        let r = self.units[0].resident.as_mut().expect("prepared above");
        let out = r.run(&mut self.dev, &self.images, kernel, budget);
        if let Ok(o) = &out {
            self.profile.phases.extend_from_slice(&o.profile.phases);
        }
        if out.is_err() {
            self.units[0].resident = None;
            // The kernel's error is the one to report; a recovery that fails
            // too leaves `resident` empty, and the next run tries again.
            if let Err(e) = self.prepare_unit(0) {
                eprintln!("session: recovery after a failed kernel failed as well: {e}");
            }
        }
        out
    }

    /// `A[m,k] @ B[k,n]`, row-major, on the first unit's tile, chunked as
    /// [`matmul_on`] chunks it but on the resident roles: no reset or image load
    /// between chunks. Bit-identical to `matmul_on` (`step17_resident`).
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        mkn: [usize; 3],
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<Vec<f32>, RunError> {
        matmul::matmul_chunked(a, b, mkn, route, fidelity, |a, b, mkn| {
            matmul::matmul_with(a, b, mkn, route, fidelity, |k| self.run(k, budget))
        })
    }

    /// The phases of every run since the last call, and forget them.
    pub fn take_profile(&mut self) -> runtime::Profile {
        std::mem::take(&mut self.profile)
    }

    pub fn device(&mut self) -> &mut Device<T> {
        &mut self.dev
    }

    /// The first unit's tile: the only one, unless the session was opened
    /// with [`TileChoice::Count`] or [`TileChoice::All`].
    pub fn tile(&self) -> NocCoord<Noc0> {
        self.units[0].tile
    }

    /// Every tile the session computes on, in unit order.
    pub fn tiles(&self) -> Vec<NocCoord<Noc0>> {
        self.units.iter().map(|u| u.tile).collect()
    }

    /// Steps of GDDR ops completed on each unit so far, in unit order.
    pub fn steps_per_tile(&self) -> Vec<u64> {
        self.units.iter().map(|u| u.steps).collect()
    }

    pub fn grid(&self) -> &Tensix {
        &self.grid
    }

    pub fn images(&self) -> &RoleImages<'static> {
        &self.images
    }

    /// The device, with every unit's role cores held again.
    pub fn into_device(mut self) -> Device<T> {
        for u in &mut self.units {
            if let Some(r) = u.resident.take() {
                let _ = r.stop(&mut self.dev, &self.images);
            }
        }
        self.dev
    }
}

impl Session<tt_kmd::Kmd> {
    /// Open `/dev/tenstorrent/{index}` for compute, with the driver's cleanup
    /// write as step 3: the first tile's on this descriptor, every further
    /// tile's on a descriptor of its own, since the driver keeps one per
    /// descriptor.
    pub fn open_card(
        index: u16,
        images: RoleImages<'static>,
        choice: TileChoice,
    ) -> Result<Self, SessionError> {
        let kmd = tt_kmd::Kmd::open(index)?;
        let dev = Device::open(kmd)?;
        let mut first = true;
        Self::open(dev, images, choice, |kmd, tile| {
            let (x, y) = (tile.x(), tile.y());
            if std::mem::take(&mut first) {
                kmd.set_cleanup_write(x, y, 0, tensix::SOFT_RESET_0, ALL_BABIES_HELD)?;
                return Ok(None);
            }
            let extra = tt_kmd::CleanupWrite::register(
                index,
                x,
                y,
                0,
                tensix::SOFT_RESET_0,
                ALL_BABIES_HELD,
            )?;
            Ok(Some(Box::new(extra)))
        })
    }
}
