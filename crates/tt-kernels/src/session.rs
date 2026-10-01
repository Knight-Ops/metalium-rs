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

use std::sync::Arc;

use tt_device::tlb::WindowKind;
use tt_device::{Device, Transport, TransportError};
use tt_isa::isa::Instruction;
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
    /// Mover lists submitted to this tile: one host round trip each.
    lists: u64,
}

/// What a session needs to keep tensors in GDDR: the chip's channels, an
/// allocator, the data mover image, and a window for each size of access.
struct DramState {
    alloc: DramAlloc,
    dram: tt_isa::dram::Dram,
    image: &'static [u8],
    w4: tt_device::Window,
}

/// One mover list for one unit: list entries, the `KERNEL` entries among
/// them (whose generation is filled in when the list is started), and the
/// matmul programs those kernels run.
#[derive(Default)]
struct Segment {
    /// What it is, for [`tensor::stats`]: its first step's.
    what: &'static str,
    entries: Vec<[u32; 8]>,
    /// Indices into `entries` of the `KERNEL` entries.
    kernels: Vec<usize>,
    roles: Option<Arc<[Vec<Instruction>; 3]>>,
    /// How many of the op's steps it covers, for [`Session::steps_per_tile`].
    steps: u64,
}

/// One unit's steps as mover lists, in order. Consecutive steps share a list
/// until it is full (`tt_isa::dm::LIST_MAX`) or a kernel's programs differ from
/// the list's: a tile's role program slots hold one kernel at a time. What
/// were separate lists are separated by a `WAIT` entry, since their entries
/// may reuse each other's L1 slots; a `KERNEL` entry waits by itself.
fn segments(steps: Vec<Step>) -> Vec<Segment> {
    use tt_isa::dm::{op, LIST_MAX};
    let mut out = Vec::new();
    let mut cur = Segment::default();
    let close = |cur: &mut Segment, out: &mut Vec<Segment>| {
        if !cur.entries.is_empty() {
            out.push(std::mem::take(cur));
        }
    };
    // One entry, or one whole op record (`tt_isa::dm::record`), which a list
    // never splits.
    let push = |cur: &mut Segment, out: &mut Vec<Segment>, e: &[[u32; 8]]| {
        // A full list ends here; the mover waits for all of it before it
        // reports done, so the next list starts from a clean boundary.
        if cur.entries.len() + e.len() > LIST_MAX as usize {
            close(cur, out);
        }
        cur.entries.extend_from_slice(e);
    };
    let mut after_list = false;
    for step in steps {
        match step {
            Step::List { what, entries } => {
                if entries.is_empty() {
                    continue;
                }
                if cur.what.is_empty() {
                    cur.what = what;
                }
                if after_list && !cur.entries.is_empty() {
                    push(&mut cur, &mut out, &[[op::WAIT, 0, 0, 0, 0, 0, 0, 0]]);
                }
                let mut i = 0;
                while i < entries.len() {
                    let n = tt_isa::dm::record::len(entries[i][0]).min(entries.len() - i);
                    push(&mut cur, &mut out, &entries[i..i + n]);
                    i += n;
                    if cur.what.is_empty() {
                        // A list that spilled into a new segment.
                        cur.what = what;
                    }
                }
                cur.steps += 1;
                after_list = true;
            }
            Step::Matmul(roles) => {
                let same = cur.roles.as_ref().is_none_or(|r| {
                    Arc::ptr_eq(r, &roles)
                        || r.iter().zip(roles.iter()).all(|(a, b)| {
                            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.word() == y.word())
                        })
                });
                if !same || cur.entries.len() == LIST_MAX as usize {
                    close(&mut cur, &mut out);
                }
                if cur.what.is_empty() {
                    cur.what = "matmul";
                }
                cur.roles = Some(roles);
                cur.kernels.push(cur.entries.len());
                cur.entries.push([op::KERNEL, 0, 0, 0, 0, 0, 0, 0]);
                cur.steps += 1;
                after_list = false;
            }
        }
    }
    close(&mut cur, &mut out);
    out
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
                lists: 0,
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
    /// Each unit's steps are run as few mover lists as they fit in
    /// ([`segments`]): the mover runs the matmul kernels itself
    /// (`tt_isa::dm::op::KERNEL`), so a unit's whole share of an op -- gathers,
    /// kernels, scatters -- is usually one list, one submission and one wait.
    ///
    /// A unit whose list fails is recovered before the error is returned -- its
    /// mover restarted, and if the list ran kernels, its tile reset and roles
    /// restarted -- so the session stays usable. The other units' lists of the
    /// same wave are finished first, so nothing is left running.
    fn execute(&mut self, work: tensor::Work, budget: u64) -> Result<DramTensor, TensorError> {
        let tensor::Work { out, jobs } = work;
        let n = self.units.len();
        let mut queues: Vec<Vec<Step>> = vec![Vec::new(); n];
        for (j, job) in jobs.into_iter().enumerate() {
            queues[j % n].extend(job);
        }
        let queues: Vec<Vec<Segment>> = queues.into_iter().map(segments).collect();
        let waves = queues.iter().map(Vec::len).max().unwrap_or(0);
        for w in 0..waves {
            let what = queues.iter().find_map(|q| q.get(w)).map_or("", |s| s.what);
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

    /// Segment `w` of every queue that has one: started on every unit, then
    /// finished on every unit. Returns the units that failed, whether their
    /// segment ran kernels, and the first error.
    fn wave(
        &mut self,
        queues: &[Vec<Segment>],
        w: usize,
        budget: u64,
    ) -> Option<(Vec<(usize, bool)>, TensorError)> {
        let mut failed = Vec::new();
        let mut first: Option<TensorError> = None;
        let mut started = Vec::new();
        for (u, q) in queues.iter().enumerate() {
            let Some(seg) = q.get(w) else { continue };
            match self.start(u, seg, budget) {
                Ok(s) => {
                    self.units[u].lists += 1;
                    started.push((u, seg, s))
                }
                Err(e) => {
                    failed.push((u, seg.roles.is_some()));
                    first.get_or_insert(e);
                }
            }
        }
        for (u, seg, s) in started {
            match self.finish(u, s) {
                Ok(()) => self.units[u].steps += seg.steps,
                Err(e) => {
                    failed.push((u, seg.roles.is_some()));
                    first.get_or_insert(e);
                }
            }
        }
        first.map(|e| (failed, e))
    }

    /// Start `seg` on unit `u`: stage its kernel and reserve a generation per
    /// kernel entry, if it has any, then submit the list.
    fn start<'s>(
        &mut self,
        u: usize,
        seg: &'s Segment,
        budget: u64,
    ) -> Result<Option<Kernel<'s>>, TensorError> {
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
        let mut entries = seg.entries.clone();
        let kernel = match &seg.roles {
            None => None,
            Some(roles) => {
                let [unpack, math, pack] = &**roles;
                let kernel = Kernel {
                    // `TILE_SEMAPHORES`: every run leaves them as it found them,
                    // which is what lets the mover run them back to back.
                    restores_semaphores: true,
                    ..Kernel::new(
                        [unpack, math, pack],
                        Schedule::Concurrent(&matmul::TILE_SEMAPHORES),
                    )
                };
                let generations =
                    r.reserve(dev, images, &kernel, budget, seg.kernels.len() as u32)?;
                for (&at, g) in seg.kernels.iter().zip(generations) {
                    entries[at][1] = g;
                }
                Some(kernel)
            }
        };
        let mover = unit.mover.as_mut().expect("started above");
        if let Err(e) = mover.submit_list(dev, r.window(), &entries) {
            if let Some(k) = &kernel {
                let _ = r.reserved_done(dev, k, false);
            }
            return Err(e.into());
        }
        Ok(kernel)
    }

    /// Wait for unit `u`'s list to finish, and close its kernel reservation.
    fn finish(&mut self, u: usize, kernel: Option<Kernel<'_>>) -> Result<(), TensorError> {
        let Session { dev, units, .. } = self;
        let unit = &mut units[u];
        let r = unit.resident.as_mut().expect("a list was started on it");
        let mover = unit.mover.as_ref().expect("a list was started on it");
        let out = mover.wait(dev, r.window());
        if let Some(k) = &kernel {
            r.reserved_done(dev, k, out.is_ok())?;
        }
        Ok(out?)
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

    /// Mover lists submitted to each unit so far, in unit order: the host's
    /// round trips for GDDR ops.
    pub fn lists_per_tile(&self) -> Vec<u64> {
        self.units.iter().map(|u| u.lists).collect()
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

#[cfg(test)]
mod tests {
    use super::{segments, Step};
    use std::sync::Arc;
    use tt_isa::dm::{op, LIST_MAX};

    fn list(n: usize, tag: u32) -> Step {
        Step::List {
            what: "test",
            entries: vec![[op::READ, tag, 0, 0, 0, 0, 0, 0]; n],
        }
    }

    fn ops(s: &super::Segment) -> Vec<u32> {
        s.entries.iter().map(|e| e[0]).collect()
    }

    #[test]
    fn separate_lists_share_one_with_a_wait_between_them() {
        let segs = segments(vec![list(2, 1), list(1, 2)]);
        assert_eq!(segs.len(), 1);
        assert_eq!(ops(&segs[0]), [op::READ, op::READ, op::WAIT, op::READ]);
        assert!(segs[0].kernels.is_empty() && segs[0].roles.is_none());
    }

    #[test]
    fn a_kernel_needs_no_wait_on_either_side_and_blocks_reuse_its_programs() {
        let roles = Arc::new([Vec::new(), Vec::new(), Vec::new()]);
        let block = || vec![list(2, 1), Step::Matmul(roles.clone()), list(1, 2)];
        let segs = segments([block(), block()].concat());
        assert_eq!(segs.len(), 1, "same programs: one list");
        assert_eq!(
            ops(&segs[0]),
            [
                op::READ,
                op::READ,
                op::KERNEL,
                op::READ,
                op::WAIT,
                op::READ,
                op::READ,
                op::KERNEL,
                op::READ
            ]
        );
        assert_eq!(segs[0].kernels, [2, 7]);
    }

    #[test]
    fn different_programs_start_a_new_list() {
        let a = Arc::new([Vec::new(), Vec::new(), Vec::new()]);
        let b = Arc::new([vec![tt_isa::sfpu::nop()], Vec::new(), Vec::new()]);
        let segs = segments(vec![Step::Matmul(a), list(1, 1), Step::Matmul(b)]);
        assert_eq!(segs.len(), 2);
        assert_eq!(ops(&segs[1]), [op::KERNEL]);
    }

    #[test]
    fn a_long_list_spills_into_the_next_at_the_limit() {
        let n = LIST_MAX as usize + 3;
        let segs = segments(vec![list(n, 1)]);
        assert_eq!(
            segs.iter().map(|s| s.entries.len()).collect::<Vec<_>>(),
            [LIST_MAX as usize, 3]
        );
        assert!(segments(vec![list(0, 1)]).is_empty());
    }
}
