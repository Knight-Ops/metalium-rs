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
use crate::program_cache::ProgramCache;
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
    /// Each unit's timestamper stream so far, between
    /// [`Session::profile_start`] and [`Session::profile_stop`].
    profiling: Option<Vec<crate::profile::UnitProfile>>,
    /// Which unit element-wise ops run on ([`Session::set_eltwise_unit`]).
    eltwise_unit: tensor::EltwiseUnit,
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
    /// The host's mirror of this tile's resident programs
    /// (`tt_isa::l1::PROGRAM_CACHE`).
    programs: ProgramCache,
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
    /// The programs of each `KERNEL` entry, in order.
    kernel_roles: Vec<Arc<[Vec<Instruction>; 3]>>,
    /// Do its kernels run resident programs (`crate::program_cache`), each
    /// entry naming its own? Otherwise they all run the one program set the
    /// host stages in the fixed slots, `roles`.
    resident: bool,
    /// Bytes of the distinct programs its kernels name, resident.
    resident_bytes: u64,
    roles: Option<Arc<[Vec<Instruction>; 3]>>,
    /// The semaphores those programs use, initialised as their kernel's setup
    /// would (`matmul::MatmulSemaphores::init`).
    init: Vec<runtime::SemaphoreInit>,
    /// How many of the op's steps it covers, for [`Session::steps_per_tile`].
    steps: u64,
}

/// Make every program `seg`'s kernels name resident on the tile, uploading
/// what is missing, and return each kernel's `(address, words)` per role
/// (`(0, 0)` for an empty one). Everything placed stays pinned until the list
/// has finished. If fragmentation leaves no room beside what is pinned, the
/// cache starts again from empty: `segments` keeps a list's programs within
/// the region, so they always fit a fresh one.
fn place_programs<T: Transport>(
    dev: &mut Device<T>,
    window: &tt_device::Window,
    tile: NocCoord<Noc0>,
    cache: &mut ProgramCache,
    seg: &Segment,
) -> Result<Vec<[(u32, u32); 3]>, TensorError> {
    use crate::program_cache::{CacheError, Placed};
    for attempt in 0..2 {
        let mut placed: Vec<[(u32, u32); 3]> = Vec::with_capacity(seg.kernel_roles.len());
        let mut full = false;
        'kernels: for (k, roles) in seg.kernel_roles.iter().enumerate() {
            if let Some(j) = seg.kernel_roles[..k]
                .iter()
                .position(|r| Arc::ptr_eq(r, roles))
            {
                placed.push(placed[j]);
                continue;
            }
            let mut p = [(0, 0); 3];
            for (t, program) in roles.iter().enumerate() {
                if program.is_empty() {
                    continue;
                }
                let words: Vec<u32> = program.iter().map(|i| i.word()).collect();
                let at = match cache.place(&words) {
                    Ok(Placed::Hit(at)) => at,
                    Ok(Placed::Upload(at)) => {
                        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                        dev.l1_write(window, tile, at, &bytes)?;
                        at
                    }
                    Ok(Placed::Bypass) => {
                        return Err(TensorError::Shape(
                            "a resident list names a program too large to cache".into(),
                        ))
                    }
                    Err(CacheError::Full { .. }) => {
                        full = true;
                        break 'kernels;
                    }
                };
                p[t] = (at as u32, words.len() as u32);
            }
            placed.push(p);
        }
        if !full {
            return Ok(placed);
        }
        cache.unpin_all();
        if attempt == 0 {
            cache.clear();
        }
    }
    Err(TensorError::Shape(
        "a list's programs do not fit an empty program cache".into(),
    ))
}

/// One unit's steps as mover lists, in order. Consecutive steps share a list
/// until it is full (`tt_isa::dm::LIST_MAX`), or until a kernel cannot join it:
/// one whose semaphores start differently (a list's kernels share one setup),
/// and -- since the fixed slots hold one kernel at a time -- one whose programs
/// differ from the list's, unless every program involved is resident
/// (`crate::program_cache`) and together they fit the cache. What
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
            Step::Kernel { roles, init } => {
                let resident = roles
                    .iter()
                    .all(|p| p.is_empty() || crate::program_cache::admitted(p.len()));
                let bytes: u64 = roles.iter().map(|p| p.len() as u64 * 4).sum();
                let seen = cur.kernel_roles.iter().any(|r| Arc::ptr_eq(r, &roles));
                let same_init = cur.init.is_empty() || cur.init == init;
                let fits = if resident {
                    (cur.kernels.is_empty() || cur.resident)
                        && (seen || cur.resident_bytes + bytes <= tt_isa::l1::PROGRAM_CACHE.len())
                } else {
                    (cur.kernels.is_empty() || !cur.resident)
                        && cur.roles.as_ref().is_none_or(|r| {
                            Arc::ptr_eq(r, &roles)
                                || r.iter().zip(roles.iter()).all(|(a, b)| {
                                    a.len() == b.len()
                                        && a.iter().zip(b).all(|(x, y)| x.word() == y.word())
                                })
                        })
                };
                if !same_init || !fits || cur.entries.len() == LIST_MAX as usize {
                    close(&mut cur, &mut out);
                }
                cur.resident = resident;
                if resident && !cur.kernel_roles.iter().any(|r| Arc::ptr_eq(r, &roles)) {
                    cur.resident_bytes += bytes;
                }
                cur.kernel_roles.push(roles.clone());
                if cur.what.is_empty() {
                    cur.what = "matmul";
                }
                cur.roles = Some(roles);
                cur.init = init;
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
                programs: ProgramCache::new(tt_isa::l1::PROGRAM_CACHE),
            });
        }
        let mut session = Session {
            dev,
            units,
            grid,
            images,
            profile: runtime::Profile::default(),
            dram: None,
            profiling: None,
            eltwise_unit: tensor::EltwiseUnit::default(),
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
        // not, and what L1 holds is no longer the host's to vouch for.
        unit.mover = None;
        unit.programs.clear();
        if let Some(r) = unit.resident.take() {
            r.stop(dev, images)?;
        }
        reset_tile(dev, unit.tile)?;
        reset_thread_state(dev, unit.tile, images)?;
        let mut r = Resident::start(dev, unit.tile, images, RESET_BUDGET)?;
        if self.profiling.is_some() {
            // What the stream held since the last drain belonged to the run
            // that failed; the profile goes on from an empty one.
            r.set_profiling(true);
            dev.configure_trace(
                r.window(),
                unit.tile,
                tt_isa::mailbox::TRACE_BUFFER,
                tt_isa::mailbox::TRACE_BUFFER_BYTES,
            )?;
        }
        unit.resident = Some(r);
        Ok(())
    }

    /// Record what every unit's data mover and role runners do from now on,
    /// through each tile's debug timestamper (`crate::profile`), until
    /// [`Session::profile_stop`]. Each unit's stream is drained after every
    /// wave, so a profile may span any number of ops.
    ///
    /// Refused on the simulator, which does not model the event stream
    /// (divergence row 54).
    pub fn profile_start(&mut self) -> Result<(), RunError> {
        if self.dev.transport().is_simulated() {
            return Err(RunError::Transport(TransportError::Hazard {
                address: tt_isa::tensix::timestamper::TIMESTAMP,
                reason: "ttsim does not model the timestamper's event stream (divergence row 54)",
            }));
        }
        if self.profiling.is_some() {
            return Ok(());
        }
        self.profiling = Some(Vec::new());
        let mut units = Vec::with_capacity(self.units.len());
        for u in 0..self.units.len() {
            if self.units[u].resident.is_none() {
                self.prepare_unit(u)?;
            }
            let Session { dev, units: us, .. } = self;
            let unit = &mut us[u];
            let r = unit.resident.as_mut().expect("prepared above");
            dev.configure_trace(
                r.window(),
                unit.tile,
                tt_isa::mailbox::TRACE_BUFFER,
                tt_isa::mailbox::TRACE_BUFFER_BYTES,
            )?;
            r.set_profiling(true);
            if unit.mover.is_some() {
                dev.write32(r.window(), unit.tile, tt_isa::dm::TRACE, 1)?;
            }
            let counter_at_start = dev.wall_clock(r.window(), unit.tile)?;
            units.push(crate::profile::UnitProfile {
                tile: unit.tile,
                events: Vec::new(),
                counter_at_start,
                host_at_start: std::time::Instant::now(),
            });
        }
        self.profiling = Some(units);
        Ok(())
    }

    /// Stop profiling and return what was recorded since
    /// [`Session::profile_start`], with the tiles' clock measured over it.
    pub fn profile_stop(&mut self) -> Result<crate::profile::DeviceProfile, RunError> {
        let Some(mut units) = self.profiling.take() else {
            return Err(RunError::Transport(TransportError::Hazard {
                address: 0,
                reason: "profile_stop without profile_start",
            }));
        };
        let mut ticks_per_us = 0.0;
        for (u, p) in units.iter_mut().enumerate() {
            let Session { dev, units: us, .. } = self;
            let unit = &mut us[u];
            let Some(r) = unit.resident.as_mut() else {
                continue;
            };
            p.events.extend(dev.read_trace(
                r.window(),
                unit.tile,
                tt_isa::mailbox::TRACE_BUFFER,
            )?);
            r.set_profiling(false);
            if unit.mover.is_some() {
                dev.write32(r.window(), unit.tile, tt_isa::dm::TRACE, 0)?;
            }
            if u == 0 {
                let now = dev.wall_clock(r.window(), unit.tile)?;
                let us_elapsed = p.host_at_start.elapsed().as_secs_f64() * 1e6;
                ticks_per_us = (now - p.counter_at_start) as f64 / us_elapsed;
            }
        }
        Ok(crate::profile::DeviceProfile {
            units,
            ticks_per_us,
        })
    }

    /// Move unit `u`'s events into the profile and start its stream empty
    /// again, so the 1024-event buffer bounds one wave, not a whole profile.
    fn drain(&mut self, u: usize) -> Result<(), TransportError> {
        let Session {
            dev,
            units,
            profiling,
            ..
        } = self;
        let (Some(p), Some(r)) = (profiling.as_mut(), units[u].resident.as_ref()) else {
            return Ok(());
        };
        let tile = units[u].tile;
        let events = dev.read_trace(r.window(), tile, tt_isa::mailbox::TRACE_BUFFER)?;
        p[u].events.extend(events);
        dev.configure_trace(
            r.window(),
            tile,
            tt_isa::mailbox::TRACE_BUFFER,
            tt_isa::mailbox::TRACE_BUFFER_BYTES,
        )
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

    /// Every datum of every tile of `t`, padding included, row-major
    /// `[32 * rt, 32 * ct]` (`DramTensor::download_padded`).
    pub fn download_padded(&mut self, t: &DramTensor) -> Result<Vec<f32>, TensorError> {
        let Session { dev, dram, .. } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        t.download_padded(dev, &d.w4)
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

    /// Element-wise `a (op) b` in GDDR, on the SFPU ([`tensor::sfpu_eltwise`])
    /// or the data movers' FP32 units ([`tensor::eltwise`]), as
    /// [`Session::set_eltwise_unit`] says.
    pub fn eltwise(
        &mut self,
        op: tensor::Eltwise,
        a: &DramTensor,
        b: Option<&DramTensor>,
    ) -> Result<DramTensor, TensorError> {
        use tensor::OpPadding;
        let units = self.units.len();
        let unit = self.eltwise_unit;
        let alloc = &mut self.dram_state()?.alloc;
        let [rt, ct] = a.grid();
        let (kind, bcast) = tensor::broadcast_of(op, a, b)?;
        use crate::sfpu::ops::Broadcast;
        // What the mover can do: its own kinds, the same shape or `ADD_ROW`.
        let mover_op = match (kind, bcast) {
            (k, Broadcast::None) if crate::sfpu::ops::mover_has(k) => Some(op),
            (tt_isa::dm::kind::ADD, Broadcast::Row) => Some(tensor::Eltwise {
                kind: tt_isa::dm::kind::ADD_ROW,
                ..op
            }),
            _ => None,
        };
        // Anything else goes to the SFPU whatever the setting.
        let sfpu = mover_op.is_none()
            || match unit {
                tensor::EltwiseUnit::Sfpu => true,
                tensor::EltwiseUnit::Mover => false,
                tensor::EltwiseUnit::Auto => tensor::sfpu_is_cheaper(kind, rt * ct, units),
            };
        let work = if sfpu {
            tensor::sfpu_eltwise(alloc, op, a, b, units)?
        } else {
            None
        };
        let work = match (work, mover_op) {
            (Some(w), _) => w,
            (None, Some(m)) => tensor::eltwise(alloc, m, a, b, units)?,
            (None, None) => {
                return Err(TensorError::Shape(format!(
                    "element-wise {kind:#x} with {bcast:?}: no unit computes it"
                )))
            }
        };
        let out = self.execute(work, RESET_BUDGET)?;
        out.set_pad(op.produces(&[Some(a), b].into_iter().flatten().collect::<Vec<_>>()));
        Ok(out)
    }

    /// Run element-wise ops on `unit` from now on: by default whichever is
    /// cheaper for the op's size (`tensor::sfpu_is_cheaper`), or always the
    /// SFPU (where the op has a program), or always the data mover's FP32
    /// unit, which stays as the reference and the fallback. Bit-identical
    /// whichever (`step19_eltwise`).
    pub fn set_eltwise_unit(&mut self, unit: tensor::EltwiseUnit) {
        self.eltwise_unit = unit;
    }

    pub fn eltwise_unit(&self) -> tensor::EltwiseUnit {
        self.eltwise_unit
    }

    /// The sum over rows of `a`, `[1, cols]`, in `burn-flex`'s order
    /// ([`tensor::sum_rows`]).
    pub fn sum_rows(&mut self, a: &DramTensor) -> Result<DramTensor, TensorError> {
        use tensor::OpPadding;
        let units = self.units.len();
        let work = tensor::sum_rows(&mut self.dram_state()?.alloc, a, units)?;
        let out = self.execute(work, RESET_BUDGET)?;
        out.set_pad(tensor::SumRows.produces(&[a]));
        Ok(out)
    }

    /// Make `t`'s padding what an op needs (`tensor::PadNeed`): nothing when
    /// it already is, or when `t` has no ragged edge; otherwise its edge tiles
    /// refilled ([`tensor::fill_pad`]). In place when `t` owns its slots; a
    /// view shares its parent's, so it is copied first, and the copy --
    /// returned, for the caller to use instead and free -- is filled.
    ///
    /// In-place fills write zeros only, so no view of `t` that claims zero
    /// padding is ever made wrong by one.
    fn meet(
        &mut self,
        t: &DramTensor,
        need: tensor::PadNeed,
    ) -> Result<Option<DramTensor>, TensorError> {
        use tensor::{Pad, PadNeed};
        if need == PadNeed::Any || t.pad() == Pad::Zero {
            return Ok(None);
        }
        let units = self.units.len();
        if t.placement.owned() {
            let jobs = tensor::fill_pad(t, 0.0, units)?;
            self.run_jobs(jobs, RESET_BUDGET)?;
            t.set_pad(Pad::Zero);
            return Ok(None);
        }
        let copy = self.eltwise(
            tensor::Eltwise {
                kind: tt_isa::dm::kind::COPY,
                scalar: 0.0,
            },
            t,
            None,
        )?;
        let jobs = tensor::fill_pad(&copy, 0.0, units)?;
        if let Err(e) = self.run_jobs(jobs, RESET_BUDGET) {
            let _ = self.free(copy);
            return Err(e);
        }
        copy.set_pad(Pad::Zero);
        Ok(Some(copy))
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
        use tensor::OpPadding;
        let need = tensor::MatmulPadding;
        let ca = self.meet(a, need.requires(0))?;
        let cb = match self.meet(b, need.requires(1)) {
            Ok(c) => c,
            Err(e) => {
                if let Some(c) = ca {
                    let _ = self.free(c);
                }
                return Err(e);
            }
        };
        let units = self.units.len();
        let out = self
            .dram_state()
            .and_then(|d| {
                tensor::matmul_dram(
                    &mut d.alloc,
                    ca.as_ref().unwrap_or(a),
                    a_transposed,
                    cb.as_ref().unwrap_or(b),
                    b_transposed,
                    route,
                    fidelity,
                    units,
                )
            })
            .and_then(|work| self.execute(work, budget));
        for c in [ca, cb].into_iter().flatten() {
            self.free(c)?;
        }
        let out = out?;
        out.set_pad(need.produces(&[a, b]));
        Ok(out)
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
        if let Err(e) = self.run_jobs(jobs, budget) {
            if let Ok(d) = self.dram_state() {
                d.alloc.free(&out.placement);
            }
            return Err(e);
        }
        Ok(out)
    }

    /// The body of [`Session::execute`], for jobs whose output already exists
    /// (an in-place fill): deal them out, run them in waves, recover a unit
    /// whose list failed.
    fn run_jobs(&mut self, jobs: Vec<tensor::Job>, budget: u64) -> Result<(), TensorError> {
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
                return Err(error);
            }
        }
        Ok(())
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
            match self.finish(u, s).and_then(|()| Ok(self.drain(u)?)) {
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
            profiling,
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
            if profiling.is_some() {
                dev.write32(r.window(), unit.tile, tt_isa::dm::TRACE, 1)?;
            }
        }
        let mut entries = seg.entries.clone();
        if seg.resident {
            let window = r.window();
            let placed = place_programs(dev, window, unit.tile, &mut unit.programs, seg)?;
            for (&at, p) in seg.kernels.iter().zip(placed) {
                for (t, (addr, len)) in p.into_iter().enumerate() {
                    entries[at][2 + 2 * t] = addr;
                    entries[at][3 + 2 * t] = len;
                }
            }
        }
        let kernel = match seg.kernel_roles.first() {
            None => None,
            Some(roles) => {
                let [unpack, math, pack] = &**roles;
                let kernel = Kernel {
                    // `MatmulSemaphores::init`: every run leaves them as it
                    // found them, which is what lets the mover run them back
                    // to back.
                    restores_semaphores: true,
                    ..Kernel::new([unpack, math, pack], Schedule::Concurrent(&seg.init))
                };
                let generations = r.reserve(
                    dev,
                    images,
                    &kernel,
                    budget,
                    seg.kernels.len() as u32,
                    seg.resident,
                )?;
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
        // Nothing in flight names a program any more.
        unit.programs.unpin_all();
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
        let mut out = r.run(&mut self.dev, &self.images, kernel, budget);
        if let Ok(o) = &out {
            self.profile.phases.extend_from_slice(&o.profile.phases);
            if let Err(e) = self.drain(0) {
                out = Err(e.into());
            }
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

    /// Each unit's program cache counters, in unit order.
    pub fn program_cache_stats(&self) -> Vec<crate::program_cache::CacheStats> {
        self.units.iter().map(|u| u.programs.stats()).collect()
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
    use super::{runtime, segments, Instruction, Step};
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
        let (_, init) = crate::matmul::MatmulSemaphores::alone();
        let block = || {
            vec![
                list(2, 1),
                Step::Kernel {
                    roles: roles.clone(),
                    init: init.clone(),
                },
                list(1, 2),
            ]
        };
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
    fn kernels_of_different_programs_share_a_list_when_resident() {
        let a = Arc::new([vec![tt_isa::sfpu::nop()], Vec::new(), Vec::new()]);
        let b = Arc::new([vec![tt_isa::sfpu::nop(); 2], Vec::new(), Vec::new()]);
        let (_, init) = crate::matmul::MatmulSemaphores::alone();
        let k =
            |roles: &Arc<[Vec<Instruction>; 3]>, init: &Vec<runtime::SemaphoreInit>| Step::Kernel {
                roles: roles.clone(),
                init: init.clone(),
            };
        let segs = segments(vec![k(&a, &init), list(1, 1), k(&b, &init), k(&a, &init)]);
        assert_eq!(segs.len(), 1, "every program is resident: one list");
        assert!(segs[0].resident);
        assert_eq!(segs[0].kernel_roles.len(), 3);
        // Two distinct program sets, counted once each.
        assert_eq!(segs[0].resident_bytes, 4 + 8);
        // Other semaphore starting values: another setup, so another list.
        let mut other = init.clone();
        other[0].1 = 1;
        assert_eq!(segments(vec![k(&a, &init), k(&a, &other)]).len(), 2);
    }

    #[test]
    fn programs_too_large_to_cache_take_the_fixed_slots_one_set_per_list() {
        let words = (tt_isa::l1::PROGRAM_CACHE.len() / 2 / 4 + 1) as usize;
        let big = |n| Arc::new([vec![tt_isa::sfpu::nop(); n], Vec::new(), Vec::new()]);
        let (a, b) = (big(words), big(words + 1));
        let (_, init) = crate::matmul::MatmulSemaphores::alone();
        let k = |roles: &Arc<[Vec<Instruction>; 3]>| Step::Kernel {
            roles: roles.clone(),
            init: init.clone(),
        };
        let segs = segments(vec![k(&a), k(&a), k(&b)]);
        assert_eq!(segs.len(), 2);
        assert!(!segs[0].resident && segs[0].kernels.len() == 2);
        // A resident kernel does not join a fixed-slot list either.
        let small = Arc::new([vec![tt_isa::sfpu::nop()], Vec::new(), Vec::new()]);
        assert_eq!(segments(vec![k(&a), k(&small)]).len(), 2);
    }

    #[test]
    fn a_list_holds_no_more_resident_programs_than_the_cache() {
        // Each just under the admission bound: two fit the region, three do not.
        let words = (tt_isa::l1::PROGRAM_CACHE.len() / 2 / 4) as usize - 16;
        let p = |n: usize| Arc::new([vec![tt_isa::sfpu::nop(); words - n], Vec::new(), Vec::new()]);
        let (_, init) = crate::matmul::MatmulSemaphores::alone();
        let steps = (0..3)
            .map(|n| Step::Kernel {
                roles: p(n),
                init: init.clone(),
            })
            .collect();
        let segs = segments(steps);
        assert_eq!(
            segs.iter().map(|s| s.kernels.len()).collect::<Vec<_>>(),
            [2, 1]
        );
        assert!(segs
            .iter()
            .all(|s| s.resident_bytes <= tt_isa::l1::PROGRAM_CACHE.len()));
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
