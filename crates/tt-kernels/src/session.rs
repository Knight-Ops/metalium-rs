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

use tt_device::core_control::WaitError;
use tt_device::tlb::WindowKind;
use tt_device::{Device, Transport, TransportError};
use tt_isa::isa::Instruction;
use tt_isa::noc::grid::Tensix;
use tt_isa::noc::{Noc0, NocCoord};
use tt_isa::tensix::{self, Core};

use crate::datapath;
use crate::dm::{DataMover, Throttle};
use crate::matmul::{self, Fidelity, SrcRoute};
use crate::program_cache::ProgramCache;
use crate::runtime::{self, Kernel, Resident, RoleImages, RunError, Schedule};
use crate::tensor::{self, DramAlloc, DramTensor, Step, TensorError};
use crate::trace::{self, TraceError, TraceId};

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
    /// Fewer tiles than asked for came out of reset; the rest are wedged
    /// ([`RunError::Wedged`]) and need a board reset.
    TooFewHealthy {
        asked: usize,
        healthy: usize,
        wedged: Vec<(u8, u8)>,
    },
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
            SessionError::TooFewHealthy {
                asked,
                healthy,
                wedged,
            } => write!(
                f,
                "{asked} Tensix tiles asked for, and only {healthy} came out of reset: \
                 {wedged:?} are wedged, which software cannot clear; reset the board \
                 (`tt-smi -r`, or a power cycle)"
            ),
        }
    }
}

impl std::error::Error for SessionError {}

/// Try `candidates` in order until `want` of them pass `probe`, returning
/// those that passed and those that did not. `probe` says whether a tile came
/// out of reset (`Ok(false)`: wedged, skipped); its error ends the search.
fn healthy_tiles<C: Copy, E>(
    candidates: &[C],
    want: usize,
    mut probe: impl FnMut(C) -> Result<bool, E>,
) -> Result<(Vec<C>, Vec<C>), E> {
    let (mut good, mut bad) = (Vec::new(), Vec::new());
    for &c in candidates {
        if good.len() == want {
            break;
        }
        if probe(c)? {
            good.push(c);
        } else {
            bad.push(c);
        }
    }
    Ok((good, bad))
}

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
    // And the role cores, before pushing anything, release from the RISC-V
    // side a semaphore their own thread is blocked on: thread 0's reset could
    // not, queued behind the block (`mailbox::UNWEDGE`).
    let kernel = Kernel {
        dump_rows: 0,
        unwedge: true,
        ..Kernel::new([&first, &program, &program], Schedule::InOrder)
    };
    // The reset program is a few hundred instructions on an idle tile: a role
    // that does not finish it is blocked on state a failed run left behind,
    // and no host access gets it moving again (the hazard table).
    runtime::run(dev, tile, images, &kernel, RESET_BUDGET)
        .map(|_| ())
        .map_err(|e| wedged_if_stuck(e, tile))
}

/// Roles that all timed out are a wedged tile ([`RunError::Wedged`]).
fn wedged_if_stuck(e: RunError, tile: NocCoord<Noc0>) -> RunError {
    match e {
        RunError::Roles(stuck)
            if stuck
                .iter()
                .all(|(_, _, e)| matches!(e, WaitError::TimedOut { .. })) =>
        {
            RunError::Wedged {
                tile: (tile.x(), tile.y()),
                roles: stuck.into_iter().map(|(_, core, _)| core).collect(),
            }
        }
        e => e,
    }
}

/// The per-thread reset, then the roles started resident: what a tile needs
/// after the backend pulse. A tile that does not come back is
/// [`RunError::Wedged`].
pub fn restart_roles<T: Transport>(
    dev: &mut Device<T>,
    tile: NocCoord<Noc0>,
    images: &RoleImages<'_>,
) -> Result<Resident<Noc0>, RunError> {
    reset_thread_state(dev, tile, images)?;
    Resident::start(dev, tile, images, RESET_BUDGET).map_err(|e| wedged_if_stuck(e, tile))
}

/// Release a thread stuck on a Matrix Unit instruction starved of `Src` (X5b):
/// such an instruction waits for its banks by itself, and the backend pulse
/// hands them all to the unpackers, so it outlives the pulse and its thread
/// takes nothing more. Thread 0 runs [`datapath::src_feeder`] -- four plain
/// `UNPACR`s, one into each bank of each `Src` -- which gives the instruction
/// its banks; then the pulse again, which takes them back. Returns whether the
/// feeding run finished: the stuck thread, if it was one, ran on.
pub fn unwedge_tile<T: Transport>(
    dev: &mut Device<T>,
    tile: NocCoord<Noc0>,
    images: &RoleImages<'_>,
) -> Result<bool, RunError> {
    reset_tile(dev, tile)?;
    let feeder = datapath::src_feeder();
    let nothing = Vec::new();
    // In order: the feeding thread first, then the others with nothing to run
    // -- each finishes only once the instruction ahead of it in its thread has.
    let kernel = Kernel {
        dump_rows: 0,
        unwedge: true,
        ..Kernel::new([&feeder, &nothing, &nothing], Schedule::InOrder)
    };
    let fed = runtime::run(dev, tile, images, &kernel, RESET_BUDGET).is_ok();
    reset_tile(dev, tile)?;
    Ok(fed)
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
    /// Whether GDDR matmuls double-buffer their blocks so a unit's moves
    /// overlap its kernels (`Session::set_pipeline`, checklist 9.15). On by
    /// default.
    pipeline: bool,
    /// NC's mover image, when pipelined groups' scatters go to NC
    /// ([`Session::set_scatter_mover`]); `None` keeps every move on B.
    scatter_on_nc: Option<&'static [u8]>,
    /// [`Session::host_times`].
    host: HostTimes,
    /// Host memory the card moves tensors through ([`Session::host_dma`]),
    /// pinned on first use; `Err` once it could not be, with why.
    staging: Option<Result<Box<dyn tt_device::HostMemory>, String>>,
    /// Whether tensor transfers go through `staging` where they can.
    host_dma: bool,
    /// While set, queued work takes no barrier: a host DMA transfer, which
    /// the session syncs on at once, so nothing can run past it.
    unbarriered: bool,
    /// The next free byte of `staging`, used as a ring by queued uploads;
    /// and whether anything queued may still read what is behind it.
    staging_at: usize,
    staging_busy: bool,
    /// [`Session::pipelined_blocks`].
    pipelined: u64,
    /// [`Session::set_profile_roles`].
    profile_roles: bool,
    /// [`Session::drains`].
    drains: u64,
    profile: runtime::Profile,
    /// GDDR, once [`Session::enable_dram`] has been called.
    dram: Option<DramState>,
    /// Each unit's timestamper stream so far, between
    /// [`Session::profile_start`] and [`Session::profile_stop`].
    profiling: Option<Vec<crate::profile::UnitProfile>>,
    /// Queue ops on the movers and wait only at a sync point
    /// ([`Session::sync`]), rather than one host round trip per op.
    batching: bool,
    /// Barriers queued so far: the next one's target is `(barriers + 1) * n`.
    barriers: u32,
    /// Placements freed while lists that may read them are queued: given back
    /// at the next sync.
    pending_frees: Vec<tensor::Placement>,
    /// The trace being captured ([`Session::begin_trace`]).
    capture: Option<trace::Capture>,
    /// Finished traces, by number.
    traces: std::collections::HashMap<u64, trace::Trace>,
    next_trace: u64,
    /// Moves on whenever a tile's state is lost -- a reset, a mover restart --
    /// which every trace captured before is [`TraceError::Stale`] against.
    epoch: u64,
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
    /// Lists enqueued on this tile's mover and not yet retired, oldest first
    /// ([`Session::sync`]).
    queued: std::collections::VecDeque<QueuedList>,
    /// The data mover on this tile's RISCV NC, writing out pipelined groups'
    /// scatters ([`Session::set_scatter_mover`]); started when first needed,
    /// and again whenever B's is. B's lists wait for it, so a unit is idle
    /// once B's are done.
    nc: Option<DataMover<Noc0>>,
    /// `SIGNAL`s queued on B's and NC's movers since each started: the base
    /// a segment's relative `WAIT_PEER` targets count from.
    b_signals: u32,
    nc_signals: u32,
}

/// A list on a mover's queue: its number, whether it reserved kernels (to
/// close when it is retired), and what it was, for an error.
struct QueuedList {
    number: u32,
    kernels: bool,
    what: &'static str,
}

/// The host memory a session pins for the card's DMA of tensors: one 1 GiB
/// hugepage on silicon (`tt_kmd::host`); a transfer larger moves in parts.
const HOST_DMA_STAGING: usize = 1 << 30;

/// Fewest tiles an upload or write takes by the card's DMA: every one. Queued,
/// a 50-tile upload costs the host ~19 us, a 2-tile one through the BAR 66.
const HOST_DMA_MIN_UPLOAD: usize = 1;

/// `n` zeros, in memory the kernel backs with 2 MiB pages where it can
/// (`MADV_HUGEPAGE`, before anything touches it): a download's output is
/// written once, all of it, and in this VM every 4 KiB page's first touch is
/// a fault that cost a 32 MB download more than the card's DMA of it.
fn huge_zeroed(n: usize) -> Vec<f32> {
    let v = vec![0f32; n];
    let bytes = n * 4;
    const HUGE: usize = 2 << 20;
    if bytes >= 2 * HUGE {
        let start = (v.as_ptr() as usize).next_multiple_of(HUGE);
        let end = (v.as_ptr() as usize + bytes) / HUGE * HUGE;
        if end > start {
            // SAFETY: advice only, on whole pages inside `v`'s allocation,
            // which `vec!` zeroed without touching (a fresh mapping).
            unsafe {
                libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE);
            }
        }
    }
    v
}

/// What an upload or write takes: FP32 values, or any element's datums as
/// their bits -- tilized from either without converting the other.
#[derive(Copy, Clone)]
enum Src<'a> {
    F32(&'a [f32]),
    Bits(&'a [u32]),
}

impl Src<'_> {
    fn len(&self) -> usize {
        match self {
            Src::F32(v) => v.len(),
            Src::Bits(v) => v.len(),
        }
    }

    fn bits(&self) -> Option<&[u32]> {
        match self {
            Src::F32(_) => None,
            Src::Bits(v) => Some(v),
        }
    }

    /// `t`'s tiles `tiles` into `out`, one every `stride` bytes.
    fn tilize(&self, t: &DramTensor, tiles: std::ops::Range<usize>, out: &mut [u8], stride: usize) {
        use crate::matmul::tilize_into;
        match *self {
            Src::F32(v) => tilize_into(|i| v[i].to_bits(), t.rows, t.cols, tiles, out, stride),
            Src::Bits(v) => tilize_into(|i| v[i], t.rows, t.cols, tiles, out, stride),
        }
    }
}

/// Where the host's time goes queueing ops ([`Session::host_times`],
/// checklist 9.17): each stage's wall time, PCIe traffic and how often it ran,
/// summed since the session opened or [`Session::reset_host_times`].
#[derive(Clone, Debug, Default)]
pub struct HostTimes {
    pub stages: [(HostStage, HostStageTotal); HostStage::ALL.len()],
    /// Ops queued (`Session::execute` with batching).
    pub ops: u64,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum HostStage {
    /// The whole of queueing an op: everything below, and what is between.
    #[default]
    Enqueue,
    /// Steps into mover lists (`segments`).
    Segments,
    /// A unit's roles and mover checked, started if they are not.
    Ensure,
    /// A kernel's descriptors checked against what queued lists read.
    IdleCheck,
    /// Programs looked up in the cache, uploaded if missing.
    Place,
    /// Kernel generations reserved (`Resident::reserve`).
    Reserve,
    /// NC's list, with the scatters on NC.
    NcList,
    /// The list written to the mover's ring and its doorbell rung.
    List,
    /// Every unit's barrier entry, on a multi-unit op.
    Barrier,
    /// Waiting for a unit to drain, for programs or descriptors.
    Drain,
}

impl HostStage {
    pub const ALL: [HostStage; 10] = [
        HostStage::Enqueue,
        HostStage::Segments,
        HostStage::Ensure,
        HostStage::IdleCheck,
        HostStage::Place,
        HostStage::Reserve,
        HostStage::NcList,
        HostStage::List,
        HostStage::Barrier,
        HostStage::Drain,
    ];
}

#[derive(Copy, Clone, Debug, Default)]
pub struct HostStageTotal {
    pub time: std::time::Duration,
    pub traffic: tt_device::Traffic,
    pub count: u64,
}

impl HostTimes {
    fn new() -> Self {
        let mut t = HostTimes::default();
        for (i, s) in HostStage::ALL.into_iter().enumerate() {
            t.stages[i].0 = s;
        }
        t
    }

    fn add(
        &mut self,
        stage: HostStage,
        since: (std::time::Instant, tt_device::Traffic),
        now: tt_device::Traffic,
    ) {
        let i = HostStage::ALL.iter().position(|&s| s == stage).unwrap();
        let t = &mut self.stages[i].1;
        t.time += since.0.elapsed();
        t.traffic = t.traffic + diff(now, since.1);
        t.count += 1;
    }
}

fn diff(a: tt_device::Traffic, b: tt_device::Traffic) -> tt_device::Traffic {
    tt_device::Traffic {
        bytes_written: a.bytes_written - b.bytes_written,
        bytes_read: a.bytes_read - b.bytes_read,
        write_calls: a.write_calls - b.write_calls,
        read_calls: a.read_calls - b.read_calls,
        retargets: a.retargets - b.retargets,
    }
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
    /// Indices into `entries` of the `KERNEL` (or `LAUNCH`) entries.
    kernels: Vec<usize>,
    /// `KERNEL_WAIT` entries: each one's index into `entries`, and the index
    /// into `kernels` of the launch it waits for (in the same list).
    waits: Vec<(usize, usize)>,
    /// The programs of each `KERNEL` entry, in order.
    kernel_roles: Vec<Arc<[Vec<Instruction>; 3]>>,
    /// Each kernel's block repeats, beside its roles: stored with its
    /// programs (`crate::code::Code::stored`).
    kernel_loops: Vec<Arc<[Vec<crate::code::Loop>; 3]>>,
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
    /// Its kernels' MOP configurations: one descriptor serves the list, so
    /// every kernel in it has the same (`tensor::Step::Kernel::mop`).
    mop: [Option<tt_isa::frontend::mop::MopConfig>; 3],
    /// The block repeats of the programs a list of fixed-slot kernels shares
    /// (resident lists' kernels each carry their own, `kernel_loops`).
    loops: Arc<[Vec<crate::code::Loop>; 3]>,
    /// How many of the op's steps it covers, for [`Session::steps_per_tile`].
    steps: u64,
    /// With the scatters on NC ([`Session::set_scatter_mover`]): NC's list,
    /// queued beside this one.
    nc_entries: Vec<[u32; 8]>,
    /// `WAIT_PEER` entries in `entries` (on NC) and in `nc_entries` (on B):
    /// each one's index, and the peer's `SIGNAL`s it waits for, counted from
    /// the peer's last before this segment. Made absolute when queued.
    b_waits_on_nc: Vec<(usize, u32)>,
    nc_waits_on_b: Vec<(usize, u32)>,
    /// `SIGNAL`s in `entries` and `nc_entries`.
    b_signals: u32,
    nc_signals: u32,
}

/// Make every program `seg`'s kernels name resident on the tile, uploading
/// what is missing, and return each kernel's `(address, words)` per role
/// (`(0, 0)` for an empty one). Everything placed stays pinned until the list
/// has finished. If fragmentation leaves no room beside what is pinned, the
/// cache starts again from empty: `segments` keeps a list's programs within
/// the region, so they always fit a fresh one.
/// Role `t`'s program of a kernel as the cache stores it (`Code::stored`),
/// with its [`program_cache::hash`]: worked out once per program, not once
/// per kernel enqueued -- on 8 tiles, re-encoding and rehashing every
/// program on every list cost the host more than the device spent on the op.
/// Keyed by the programs' allocations, which the memo keeps alive, so a key
/// is never reused; cleared when it grows past `MEMO_MAX`, since some ops
/// build fresh programs every call.
fn stored_program(
    roles: &Arc<[Vec<Instruction>; 3]>,
    loops: &Arc<[Vec<crate::code::Loop>; 3]>,
    t: usize,
) -> Result<(Arc<[u32]>, u32, u64), TensorError> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Stored = (Arc<[u32]>, u32, u64);
    type Entry = (
        Arc<[Vec<Instruction>; 3]>,
        Arc<[Vec<crate::code::Loop>; 3]>,
        [Option<Stored>; 3],
    );
    const MEMO_MAX: usize = 4096;
    static MEMO: OnceLock<Mutex<HashMap<(usize, usize), Entry>>> = OnceLock::new();
    // No loops is one key, whichever allocation carries it: matmuls and
    // reductions build a fresh empty table for every kernel.
    let loops_key = if loops.iter().all(Vec::is_empty) {
        0
    } else {
        Arc::as_ptr(loops) as usize
    };
    let key = (Arc::as_ptr(roles) as usize, loops_key);
    let mut memo = MEMO
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if let Some(s) = memo.get(&key).and_then(|e| e.2[t].clone()) {
        return Ok(s);
    }
    let code = crate::code::Code {
        ins: roles[t].clone(),
        loops: loops[t].clone(),
    };
    let (words, len_word) = code
        .stored()
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let h = crate::program_cache::hash(&words);
    let stored: Stored = (Arc::from(words), len_word, h);
    if memo.len() >= MEMO_MAX && !memo.contains_key(&key) {
        memo.clear();
    }
    memo.entry(key)
        .or_insert_with(|| (roles.clone(), loops.clone(), [None, None, None]))
        .2[t] = Some(stored.clone());
    Ok(stored)
}

fn place_programs<T: Transport>(
    dev: &mut Device<T>,
    window: &tt_device::Window,
    tile: NocCoord<Noc0>,
    cache: &mut ProgramCache,
    seg: &Segment,
    in_flight: bool,
) -> Result<Vec<[(u32, u32); 3]>, PlaceError> {
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
                let (words, len_word, h) =
                    stored_program(roles, &seg.kernel_loops[k], t).map_err(PlaceError::Failed)?;
                let at = match cache.place_hashed(&words, h) {
                    Ok(Placed::Hit(at)) => at,
                    Ok(Placed::Upload(at)) => {
                        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                        dev.l1_write(window, tile, at, &bytes)
                            .map_err(|e| PlaceError::Failed(e.into()))?;
                        at
                    }
                    Ok(Placed::Bypass) => {
                        return Err(PlaceError::Failed(TensorError::Shape(
                            "a resident list names a program too large to cache".into(),
                        )))
                    }
                    Err(CacheError::Full { .. }) => {
                        full = true;
                        break 'kernels;
                    }
                };
                p[t] = (at as u32, len_word);
            }
            placed.push(p);
        }
        if !full {
            return Ok(placed);
        }
        // Lists still queued run programs pinned here: making room now would
        // overwrite them under a running list (as it did, on silicon, some
        // fourteen thousand lists into a batched MNIST run). The caller
        // drains the queue and asks again.
        if in_flight {
            return Err(PlaceError::Full);
        }
        cache.unpin_all();
        if attempt == 0 {
            // All but what a trace holds, which its replays run.
            cache.clear_unheld();
        }
    }
    Err(PlaceError::Failed(TensorError::Shape(
        "a list's programs do not fit an empty program cache".into(),
    )))
}

/// Why [`place_programs`] placed nothing.
enum PlaceError {
    /// The cache is full of programs queued lists still name.
    Full,
    Failed(TensorError),
}

impl From<PlaceError> for TensorError {
    fn from(p: PlaceError) -> Self {
        match p {
            PlaceError::Full => TensorError::Shape(
                "the program cache is full of programs queued lists still run".into(),
            ),
            PlaceError::Failed(e) => e,
        }
    }
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
    segments_split(steps, false)
}

/// [`segments`], with each pipelined group's scatters on NC if `split`.
fn segments_split(steps: Vec<Step>, split: bool) -> Vec<Segment> {
    use tt_isa::dm::op;
    let mut out = Vec::new();
    let mut cur = Segment::default();
    let mut after_list = false;
    for item in pipeline_groups(steps) {
        match item {
            Item::Step(Step::List { what, entries }) => {
                if entries.is_empty() {
                    continue;
                }
                add_list(&mut cur, &mut out, what, &entries, after_list);
                after_list = true;
            }
            Item::Step(Step::Kernel {
                roles,
                init,
                mop,
                loops,
                half: _,
            }) => {
                add_kernel(&mut cur, &mut out, roles, init, *mop, loops, op::KERNEL);
                after_list = false;
            }
            Item::Group(blocks) => {
                // A group stays in one list from its first launch to its last
                // wait, so nothing the host does between lists (a drain, a
                // role reconfiguration) can fall between a launch and its
                // wait. It starts a list of its own; `pipeline_groups` kept
                // it under `LIST_MAX` entries and its programs in the cache.
                if !cur.entries.is_empty() {
                    if after_list {
                        // Its first gather may reuse the slots the list
                        // before wrote out of.
                        push_entries(&mut cur, &mut out, &[[op::WAIT, 0, 0, 0, 0, 0, 0, 0]]);
                    }
                    close_segment(&mut cur, &mut out);
                }
                let before = out.len();
                // `G0, L0`, then for each next block `Gk, W(k-1), Lk,
                // S(k-1)`, then `W(last), S(last)`: block k+1 moves in, and
                // block k-1 out, while block k computes. Every hazard is
                // ordered: a launch waits for every move before it (its
                // gather, and the scatter that emptied its half's outputs);
                // a gather refills a half only after the kernel that read it
                // was waited for; a scatter reads only waited-for outputs.
                // The scatter and gather that sit next to each other touch
                // different slots (outputs, inputs), so no `WAIT` between.
                if split {
                    add_split_group(&mut cur, &mut out, blocks);
                } else {
                    add_group(&mut cur, &mut out, blocks);
                }
                assert_eq!(out.len(), before, "a pipelined group split across lists");
                after_list = true;
            }
        }
    }
    close_segment(&mut cur, &mut out);
    out
}

/// A pipelined group into the current list, every move on B: `G0, L0`, then
/// for each next block `Gk, W(k-1), Lk, S(k-1)`, then `W(last), S(last)`.
fn add_group(cur: &mut Segment, out: &mut Vec<Segment>, blocks: Vec<Block>) {
    use tt_isa::dm::op;
    let mut launched = Vec::with_capacity(blocks.len());
    let mut scatters = Vec::with_capacity(blocks.len());
    for (k, block) in blocks.into_iter().enumerate() {
        let Block {
            gather,
            roles,
            init,
            mop,
            loops,
            scatter,
        } = block;
        add_list(cur, out, gather.0, &gather.1, false);
        if k > 0 {
            let w = launched[k - 1];
            cur.waits.push((cur.entries.len(), w));
            cur.entries.push([op::KERNEL_WAIT, 0, 0, 0, 0, 0, 0, 0]);
            cur.steps += 1;
        }
        launched.push(cur.kernels.len());
        add_kernel(cur, out, roles, init, mop, loops, op::LAUNCH);
        if k > 0 {
            let (what, entries): (&'static str, Vec<[u32; 8]>) =
                std::mem::take(&mut scatters[k - 1]);
            add_list(cur, out, what, &entries, false);
        }
        scatters.push(scatter);
    }
    let last = launched.len() - 1;
    cur.waits.push((cur.entries.len(), launched[last]));
    cur.entries.push([op::KERNEL_WAIT, 0, 0, 0, 0, 0, 0, 0]);
    cur.steps += 1;
    let (what, entries) = std::mem::take(&mut scatters[last]);
    add_list(cur, out, what, &entries, false);
}

/// A `WAIT_PEER` entry's word naming the mover it waits for
/// (`tt_isa::dm::Peer`).
const PEER_B: u32 = 0;
const PEER_NC: u32 = 1;

/// A pipelined group with its scatters on NC (`Session::set_scatter_mover`).
/// B gathers and launches as [`add_group`] does, and after each kernel's wait
/// signals NC, which writes that block out and signals back:
///
/// ```text
/// B:  G0 L0 | G1 W0 s L1 | G2 [NC>=1] W1 s L2 | ... | W(n-1) s [NC>=n]
/// NC: [B>=1] S0 s | [B>=2] S1 s | ... | [B>=n] S(n-1) s
/// ```
///
/// Block k's launch waits for NC to have written out block k-2, whose
/// outputs share its half; B's list ends waiting for NC's last, so the next
/// list's gathers (another op's slots) cannot land under a scatter, and B's
/// list done means the unit is idle. A `SIGNAL` counts only once the
/// signaller's writes have landed.
fn add_split_group(cur: &mut Segment, out: &mut Vec<Segment>, blocks: Vec<Block>) {
    use tt_isa::dm::op;
    let n = blocks.len();
    let mut launched = Vec::with_capacity(n);
    for (k, block) in blocks.into_iter().enumerate() {
        let Block {
            gather,
            roles,
            init,
            mop,
            loops,
            scatter,
        } = block;
        add_list(cur, out, gather.0, &gather.1, false);
        if k > 0 {
            cur.waits.push((cur.entries.len(), launched[k - 1]));
            cur.entries.push([op::KERNEL_WAIT, 0, 0, 0, 0, 0, 0, 0]);
            cur.entries.push([op::SIGNAL, 0, 0, 0, 0, 0, 0, 0]);
            cur.b_signals += 1;
            cur.steps += 1;
        }
        if k > 1 {
            cur.b_waits_on_nc.push((cur.entries.len(), (k - 1) as u32));
            cur.entries.push([op::WAIT_PEER, PEER_NC, 0, 0, 0, 0, 0, 0]);
        }
        launched.push(cur.kernels.len());
        add_kernel(cur, out, roles, init, mop, loops, op::LAUNCH);
        cur.nc_waits_on_b
            .push((cur.nc_entries.len(), (k + 1) as u32));
        cur.nc_entries
            .push([op::WAIT_PEER, PEER_B, 0, 0, 0, 0, 0, 0]);
        cur.nc_entries.extend_from_slice(&scatter.1);
        cur.nc_entries.push([op::SIGNAL, 0, 0, 0, 0, 0, 0, 0]);
        cur.nc_signals += 1;
        cur.steps += 1;
    }
    cur.waits.push((cur.entries.len(), launched[n - 1]));
    cur.entries.push([op::KERNEL_WAIT, 0, 0, 0, 0, 0, 0, 0]);
    cur.entries.push([op::SIGNAL, 0, 0, 0, 0, 0, 0, 0]);
    cur.b_signals += 1;
    cur.b_waits_on_nc.push((cur.entries.len(), n as u32));
    cur.entries.push([op::WAIT_PEER, PEER_NC, 0, 0, 0, 0, 0, 0]);
    cur.steps += 1;
}

/// Each `KERNEL` (or `LAUNCH`) entry of `seg` its generation, in order, and
/// each `KERNEL_WAIT` the generation of the launch it waits for.
fn fill_generations(
    entries: &mut [[u32; 8]],
    seg: &Segment,
    generations: impl IntoIterator<Item = u32>,
) {
    let gens: Vec<u32> = generations.into_iter().collect();
    for (&at, &g) in seg.kernels.iter().zip(&gens) {
        entries[at][1] = g;
    }
    for &(at, k) in &seg.waits {
        entries[at][1] = gens[k];
    }
}

fn close_segment(cur: &mut Segment, out: &mut Vec<Segment>) {
    if !cur.entries.is_empty() {
        out.push(std::mem::take(cur));
    }
}

/// One entry, or one whole op record (`tt_isa::dm::record`), which a list
/// never splits. A full list ends here; the mover waits for all of it before
/// it reports done, so the next list starts from a clean boundary.
fn push_entries(cur: &mut Segment, out: &mut Vec<Segment>, e: &[[u32; 8]]) {
    if cur.entries.len() + e.len() > tt_isa::dm::LIST_MAX as usize {
        close_segment(cur, out);
    }
    cur.entries.extend_from_slice(e);
}

/// A list step's entries into the current list, after a `WAIT` if
/// `wait_before` (the step before was a list too, whose slots these may reuse).
fn add_list(
    cur: &mut Segment,
    out: &mut Vec<Segment>,
    what: &'static str,
    entries: &[[u32; 8]],
    wait_before: bool,
) {
    use tt_isa::dm::op;
    if cur.what.is_empty() {
        cur.what = what;
    }
    if wait_before && !cur.entries.is_empty() {
        push_entries(cur, out, &[[op::WAIT, 0, 0, 0, 0, 0, 0, 0]]);
    }
    let mut i = 0;
    while i < entries.len() {
        let n = tt_isa::dm::record::len(entries[i][0]).min(entries.len() - i);
        push_entries(cur, out, &entries[i..i + n]);
        i += n;
        if cur.what.is_empty() {
            // A list that spilled into a new segment.
            cur.what = what;
        }
    }
    cur.steps += 1;
}

/// A kernel step as a `KERNEL` (or `LAUNCH`) placeholder in the current list,
/// closing it first if the kernel cannot join: one whose semaphores start
/// differently (a list's kernels share one setup), and -- since the fixed
/// slots hold one kernel at a time -- one whose programs differ from the
/// list's, unless every program involved is resident (`crate::program_cache`)
/// and together they fit the cache.
fn add_kernel(
    cur: &mut Segment,
    out: &mut Vec<Segment>,
    roles: Arc<[Vec<Instruction>; 3]>,
    init: Vec<runtime::SemaphoreInit>,
    mop: [Option<tt_isa::frontend::mop::MopConfig>; 3],
    loops: Arc<[Vec<crate::code::Loop>; 3]>,
    op_code: u32,
) {
    use tt_isa::dm::LIST_MAX;
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
                        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.word() == y.word())
                    })
            })
    };
    let same_mop = cur.kernels.is_empty() || cur.mop == mop;
    // Resident programs carry their loops; fixed-slot kernels share one
    // program, so one table.
    let same_loops = resident || cur.kernels.is_empty() || cur.loops == loops;
    if !same_init || !same_mop || !same_loops || !fits || cur.entries.len() == LIST_MAX as usize {
        close_segment(cur, out);
    }
    cur.mop = mop;
    cur.loops = loops.clone();
    cur.resident = resident;
    if resident && !seen {
        cur.resident_bytes += bytes;
    }
    cur.kernel_roles.push(roles.clone());
    cur.kernel_loops.push(loops);
    if cur.what.is_empty() {
        cur.what = "matmul";
    }
    cur.roles = Some(roles);
    cur.init = init;
    cur.kernels.push(cur.entries.len());
    cur.entries.push([op_code, 0, 0, 0, 0, 0, 0, 0]);
    cur.steps += 1;
}

/// One double-buffered block: its gather, its kernel, its scatter.
struct Block {
    gather: (&'static str, Vec<[u32; 8]>),
    roles: Arc<[Vec<Instruction>; 3]>,
    init: Vec<runtime::SemaphoreInit>,
    mop: [Option<tt_isa::frontend::mop::MopConfig>; 3],
    loops: Arc<[Vec<crate::code::Loop>; 3]>,
    scatter: (&'static str, Vec<[u32; 8]>),
}

/// A unit's steps, as [`segments`] lays them out.
enum Item {
    Step(Step),
    /// Consecutive double-buffered blocks to overlap.
    Group(Vec<Block>),
}

/// Blocks per group at most: a group is one list, and a block is a gather
/// record, a launch, a wait and a scatter record.
const GROUP_MAX: usize = 40;

/// A unit's steps with its runs of double-buffered blocks gathered into
/// groups: `[gather, kernel (staged in a half), scatter]` jobs, one after
/// another, in alternating halves, whose programs are resident and fit the
/// cache together. Anything else is left as it is; so is a lone block.
fn pipeline_groups(steps: Vec<Step>) -> Vec<Item> {
    use tt_isa::dm::LIST_MAX;
    let mut items: Vec<Item> = Vec::new();
    let mut run: Vec<Block> = Vec::new();
    let mut run_entries = 0usize;
    let flush = |run: &mut Vec<Block>, items: &mut Vec<Item>| match run.len() {
        0 => {}
        1 => {
            let b = run.pop().unwrap();
            items.push(Item::Step(Step::List {
                what: b.gather.0,
                entries: b.gather.1,
            }));
            items.push(Item::Step(Step::Kernel {
                roles: b.roles,
                init: b.init,
                mop: Box::new(b.mop),
                loops: b.loops,
                half: None,
            }));
            items.push(Item::Step(Step::List {
                what: b.scatter.0,
                entries: b.scatter.1,
            }));
        }
        _ => items.push(Item::Group(std::mem::take(run))),
    };
    let mut steps = steps.into_iter().peekable();
    let mut last_half: Option<u8> = None;
    let mut resident_bytes = 0u64;
    let mut seen: Vec<Arc<[Vec<Instruction>; 3]>> = Vec::new();
    while let Some(step) = steps.next() {
        // A block: this list, then a kernel staged in a half, then a list.
        let is_block = matches!(step, Step::List { .. })
            && matches!(steps.peek(), Some(Step::Kernel { half: Some(_), roles, .. })
                if roles.iter().all(|p| p.is_empty() || crate::program_cache::admitted(p.len())));
        if !is_block {
            flush(&mut run, &mut items);
            last_half = None;
            items.push(Item::Step(step));
            continue;
        }
        let Step::List { what, entries } = step else {
            unreachable!()
        };
        let Some(Step::Kernel {
            roles,
            init,
            mop,
            loops,
            half,
        }) = steps.next()
        else {
            unreachable!()
        };
        let scatter = match steps.next() {
            Some(Step::List { what, entries }) => (what, entries),
            other => {
                // Not a block after all: put it back as plain steps.
                flush(&mut run, &mut items);
                last_half = None;
                items.push(Item::Step(Step::List { what, entries }));
                items.push(Item::Step(Step::Kernel {
                    roles,
                    init,
                    mop,
                    loops,
                    half: None,
                }));
                if let Some(o) = other {
                    items.push(Item::Step(o));
                }
                continue;
            }
        };
        let block_entries = entries.len() + scatter.1.len() + 2;
        let bytes: u64 = roles.iter().map(|p| p.len() as u64 * 4).sum();
        let new = !seen.iter().any(|r| Arc::ptr_eq(r, &roles));
        let joins = !run.is_empty()
            && half != last_half
            && run.len() < GROUP_MAX
            && run_entries + block_entries + 1 < LIST_MAX as usize
            && (!new || resident_bytes + bytes <= tt_isa::l1::PROGRAM_CACHE.len())
            // Not the loops: a group's programs are all resident, and a
            // resident program carries its own (`add_kernel`) -- an
            // element-wise run's repeat its tile count.
            && run
                .last()
                .is_some_and(|b| b.init == init && b.mop == *mop);
        if !joins {
            flush(&mut run, &mut items);
            run_entries = 0;
            resident_bytes = 0;
            seen.clear();
        }
        if !seen.iter().any(|r| Arc::ptr_eq(r, &roles)) {
            resident_bytes += bytes;
            seen.push(roles.clone());
        }
        run_entries += block_entries;
        last_half = half;
        run.push(Block {
            gather: (what, entries),
            roles,
            init,
            mop: *mop,
            loops,
            scatter,
        });
    }
    flush(&mut run, &mut items);
    items
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
        // The tiles to try, in order, how many to keep, and whether a wedged
        // one is an error (a tile named exactly) or skipped with a warning.
        let (candidates, want, strict): (Vec<(u8, u8)>, usize, bool) = match choice {
            TileChoice::Exactly(x, y) => (vec![(x, y)], 1, true),
            TileChoice::First => {
                let mut all: Vec<(u8, u8)> = grid.tiles::<Noc0>().map(|t| (t.x(), t.y())).collect();
                all.sort_unstable();
                (all, 1, false)
            }
            TileChoice::Count(n) if n == 0 || n > have => {
                return Err(SessionError::TooFewTiles { asked: n, have })
            }
            TileChoice::Count(n) => (
                grid.tiles::<Noc0>().map(|t| (t.x(), t.y())).collect(),
                n,
                false,
            ),
            TileChoice::All => (
                grid.tiles::<Noc0>().map(|t| (t.x(), t.y())).collect(),
                usize::MAX,
                false,
            ),
        };
        for &(x, y) in &candidates {
            if !grid.contains(x, y) {
                return Err(SessionError::NoSuchTile {
                    x,
                    y,
                    columns: grid.columns().collect(),
                });
            }
        }
        let mut session = Session {
            dev,
            units: Vec::new(),
            grid,
            images,
            profile: runtime::Profile::default(),
            pipeline: true,
            scatter_on_nc: None,
            host: HostTimes::new(),
            staging: None,
            host_dma: true,
            unbarriered: false,
            staging_at: 0,
            staging_busy: false,
            pipelined: 0,
            profile_roles: true,
            drains: 0,
            dram: None,
            profiling: None,
            batching: std::env::var("TT_BATCH").map_or(true, |v| v != "0"),
            barriers: 0,
            pending_frees: Vec::new(),
            capture: None,
            traces: Default::default(),
            next_trace: 0,
            epoch: 0,
            _cleanup: Vec::new(),
        };
        let (_, wedged) = healthy_tiles(&candidates, want, |(x, y)| {
            let tile = NocCoord::new(x, y).expect("a grid tile is a NoC coordinate");
            session
                ._cleanup
                .extend(register_cleanup(session.dev.transport(), tile)?);
            session.units.push(Unit {
                tile,
                resident: None,
                mover: None,
                steps: 0,
                lists: 0,
                programs: ProgramCache::new(tt_isa::l1::PROGRAM_CACHE),
                queued: Default::default(),
                nc: None,
                b_signals: 0,
                nc_signals: 0,
            });
            match session.prepare_unit(session.units.len() - 1) {
                Ok(()) => Ok(true),
                Err(e @ RunError::Wedged { .. }) if !strict => {
                    eprintln!("session: {e}; skipped");
                    session.units.pop();
                    Ok(false)
                }
                Err(e) => Err(SessionError::Reset(e)),
            }
        })?;
        let asked = if want == usize::MAX { 1 } else { want };
        if session.units.len() < asked {
            return Err(SessionError::TooFewHealthy {
                asked,
                healthy: session.units.len(),
                wedged,
            });
        }
        Ok(session)
    }

    /// Put every tile back to a known state: step 4 again, then the role
    /// images loaded and left resident.
    pub fn prepare(&mut self) -> Result<(), RunError> {
        self.sync_run()?;
        for u in 0..self.units.len() {
            self.prepare_unit(u)?;
        }
        Ok(())
    }

    fn prepare_unit(&mut self, u: usize) -> Result<(), RunError> {
        let profile_roles = self.profile_roles;
        let Session {
            dev,
            units,
            images,
            epoch,
            ..
        } = self;
        let unit = &mut units[u];
        // The reset holds RISCV B too; GDDR contents survive, the mover does
        // not, and what L1 holds is no longer the host's to vouch for.
        *epoch += 1;
        unit.mover = None;
        unit.nc = None;
        unit.programs.clear();
        if let Some(r) = unit.resident.take() {
            r.stop(dev, images)?;
        }
        reset_tile(dev, unit.tile)?;
        let mut r = match restart_roles(dev, unit.tile, images) {
            // ttsim has no backend pulse (divergence row 16), the half of the
            // recovery that takes the fed banks back: there the tile would
            // compute from them, wrongly. It stays an error.
            Err(e @ RunError::Wedged { .. }) if dev.transport().is_simulated() => return Err(e),
            Err(RunError::Wedged { .. }) => {
                let (x, y) = (unit.tile.x(), unit.tile.y());
                let fed = unwedge_tile(dev, unit.tile, images)?;
                let again = restart_roles(dev, unit.tile, images);
                eprintln!(
                    "session: tile ({x},{y}) was wedged; fed its `Src` banks (the feeding run \
                     {}): {}",
                    if fed { "finished" } else { "did not finish" },
                    if again.is_ok() {
                        "recovered"
                    } else {
                        "still wedged"
                    }
                );
                again?
            }
            other => other?,
        };
        if self.profiling.is_some() {
            // What the stream held since the last drain belonged to the run
            // that failed; the profile goes on from an empty one.
            r.set_profiling(true);
            r.set_profile_roles(profile_roles);
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
        self.sync_run()?;
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
            let profile_roles = self.profile_roles;
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
            r.set_profile_roles(profile_roles);
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
        self.sync_run()?;
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
        self.upload_src(Src::F32(values), rows, cols, crate::tensor::Elem::F32)
    }

    /// Upload a row-major `[rows, cols]` matrix of `elem` datums, as their
    /// bits (`DramTensor::upload_bits`).
    pub fn upload_bits(
        &mut self,
        values: &[u32],
        rows: usize,
        cols: usize,
        elem: crate::tensor::Elem,
    ) -> Result<DramTensor, TensorError> {
        self.upload_src(Src::Bits(values), rows, cols, elem)
    }

    fn upload_src(
        &mut self,
        values: Src<'_>,
        rows: usize,
        cols: usize,
        elem: crate::tensor::Elem,
    ) -> Result<DramTensor, TensorError> {
        if values.len() != rows * cols {
            return Err(TensorError::Shape(format!(
                "{} values for a [{rows}, {cols}] tensor",
                values.len()
            )));
        }
        let t = DramTensor::alloc_elem(&mut self.dram_state()?.alloc, rows, cols, elem)?;
        if let Err(e) = self.write_src(&t, values, true) {
            let _ = self.free(t);
            return Err(e);
        }
        Ok(t)
    }

    /// `values` into `t`'s slots: by the card from pinned host memory where
    /// it can ([`Session::set_host_dma`]), queued behind what is queued, else
    /// through the BAR -- which, for slots that are not `fresh`, waits for
    /// queued work first, since it may still read them.
    fn write_src(
        &mut self,
        t: &DramTensor,
        values: Src<'_>,
        fresh: bool,
    ) -> Result<(), TensorError> {
        t.check_write(values.len(), values.bits())?;
        let [rt, ct] = t.grid();
        let tiles = rt * ct;
        if tiles >= HOST_DMA_MIN_UPLOAD && self.capture.is_none() {
            if let Some(()) = self.dma_upload(t, values, tiles)? {
                t.set_pad(tensor::Pad::Zero);
                return Ok(());
            }
        }
        if !fresh {
            self.sync()?;
        }
        let mut images = vec![0u8; tiles * crate::matmul::TILE_IMAGE_BYTES];
        values.tilize(t, 0..tiles, &mut images, crate::matmul::TILE_IMAGE_BYTES);
        let Session { dev, dram, .. } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        t.write_images(dev, &d.w4, &images)
    }

    /// Download a tensor of any element type to row-major datums' bits.
    pub fn download_bits(&mut self, t: &DramTensor) -> Result<Vec<u32>, TensorError> {
        self.refuse_while_capturing("download")?;
        self.sync()?;
        if let Some(v) = self.dma_download(t)? {
            return Ok(v.iter().map(|v| v.to_bits()).collect());
        }
        let Session { dev, dram, .. } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        t.download_bits(dev, &d.w4)
    }

    /// Download a tensor to row-major values.
    pub fn download(&mut self, t: &DramTensor) -> Result<Vec<f32>, TensorError> {
        t.expect("a download as FP32 values", tensor::Elem::F32)?;
        self.refuse_while_capturing("download")?;
        self.sync()?;
        if let Some(v) = self.dma_download(t)? {
            return Ok(v);
        }
        let Session { dev, dram, .. } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        t.download(dev, &d.w4)
    }

    /// Move tensors between the host and GDDR by the card's own DMA, through
    /// pinned host memory (`Transport::host_memory`, one 1 GiB hugepage on
    /// silicon), rather than the host's stores and loads through a BAR: on
    /// card 0, ~20 GB/s up and ~27 down against 0.15 and 0.04 through this
    /// VM's uncached BAR (`silicon_bench_host_dma`). On by default; where no
    /// host memory can be pinned, transfers take the BAR and the reason is
    /// said once.
    pub fn set_host_dma(&mut self, on: bool) {
        self.host_dma = on;
    }

    /// The pinned staging buffer, pinned on first use; `None` (said once) if
    /// it cannot be, or host DMA is off.
    fn staging(&mut self) -> Option<&mut Box<dyn tt_device::HostMemory>> {
        if !self.host_dma {
            return None;
        }
        if self.staging.is_none() {
            let got = self
                .dev
                .transport()
                .host_memory(HOST_DMA_STAGING)
                .map_err(|e| e.to_string());
            if let Err(e) = &got {
                eprintln!(
                    "session: tensor transfers go through the BAR, not the card's DMA: \
                     no host memory for it ({e})"
                );
            }
            self.staging = Some(got);
        }
        self.staging.as_mut().and_then(|s| s.as_mut().ok())
    }

    /// [`Session::write_src`] by the card: tilized into the next free part of
    /// the staging ring, then each unit's share queued like an op's -- behind
    /// what is queued on it (whose lists end with their kernels done, so the
    /// L1 it stages through is free), with a barrier after, as an op has (so
    /// no later op on another unit reads a tile before it lands, and no write
    /// overtakes an earlier op still reading). Nothing waits: the ring syncs
    /// only when it wraps onto memory a queued transfer may still read.
    /// `None` if there is no host memory.
    fn dma_upload(
        &mut self,
        t: &DramTensor,
        values: Src<'_>,
        tiles: usize,
    ) -> Result<Option<()>, TensorError> {
        if self.staging().is_none() {
            return Ok(None);
        }
        let slot = tt_isa::dm::TILE_SLOT as usize;
        let per = HOST_DMA_STAGING / slot;
        for first in (0..tiles).step_by(per) {
            let n = (tiles - first).min(per);
            let at = self.staging_region(n * slot)?;
            let host = self.staging().expect("checked above");
            // Tilized straight into the pinned memory the card reads.
            host.with_bytes(&mut |buf| values.tilize(t, first..first + n, &mut buf[at..], slot));
            let base = host.noc_address() + at as u64;
            let jobs = tensor::host_dma_jobs(
                t.tensor_ref(),
                first..first + n,
                true,
                base,
                self.units.len(),
            );
            self.staging_busy = true;
            self.submit_jobs(jobs, RESET_BUDGET)?;
        }
        Ok(Some(()))
    }

    /// `bytes` of the staging ring from its next free byte, syncing first if
    /// they would wrap onto memory a queued upload may still read.
    fn staging_region(&mut self, bytes: usize) -> Result<usize, TensorError> {
        if self.staging_at + bytes > HOST_DMA_STAGING {
            if self.staging_busy {
                self.sync()?;
            }
            self.staging_at = 0;
        }
        let at = self.staging_at;
        self.staging_at += bytes;
        Ok(at)
    }

    /// Run a transfer's jobs to their end, without a barrier (nothing is
    /// queued behind them before the sync).
    fn submit_dma(&mut self, jobs: Vec<tensor::Job>) -> Result<(), TensorError> {
        self.unbarriered = true;
        let r = self.submit_jobs(jobs, RESET_BUDGET);
        self.unbarriered = false;
        r?;
        self.sync()
    }

    /// A download by the card: each unit moves its share of `t`'s tiles' datums
    /// to host memory. `None` if there is no host memory (or a capture is on).
    /// The caller has synced.
    fn dma_download(&mut self, t: &DramTensor) -> Result<Option<Vec<f32>>, TensorError> {
        if self.capture.is_some() || self.staging().is_none() {
            return Ok(None);
        }
        let [rt, ct] = t.grid();
        let tiles = rt * ct;
        let per = HOST_DMA_STAGING / tt_isa::dm::TILE_SLOT as usize;
        let mut out = huge_zeroed(t.rows * t.cols);
        for first in (0..tiles).step_by(per) {
            let n = (tiles - first).min(per);
            let base = self.staging().expect("checked above").noc_address();
            let jobs = tensor::host_dma_jobs(
                t.tensor_ref(),
                first..first + n,
                false,
                base,
                self.units.len(),
            );
            self.submit_dma(jobs)?;
            let host = self.staging().expect("checked above");
            // Detilized straight out of the pinned memory the card wrote.
            host.with_bytes(&mut |buf| {
                crate::matmul::detilize_from(
                    buf,
                    tt_isa::dm::TILE_SLOT as usize,
                    tt_isa::dm::TILE_DATA as usize,
                    t.rows,
                    t.cols,
                    first..first + n,
                    &mut out,
                )
            });
        }
        Ok(Some(out))
    }

    /// Every datum of every tile of `t`, padding included, row-major
    /// `[32 * rt, 32 * ct]` (`DramTensor::download_padded`).
    pub fn download_padded(&mut self, t: &DramTensor) -> Result<Vec<f32>, TensorError> {
        self.refuse_while_capturing("download")?;
        self.sync()?;
        let Session { dev, dram, .. } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        t.download_padded(dev, &d.w4)
    }

    /// Give a tensor's slots back.
    pub fn free(&mut self, t: DramTensor) -> Result<(), TensorError> {
        self.free_placement(t.placement)
    }

    fn free_placement(&mut self, p: tensor::Placement) -> Result<(), TensorError> {
        if !p.owned() {
            return Ok(());
        }
        // A trace's: its replays read or write it (`crate::trace`).
        if let Some(c) = self.capture.as_mut() {
            c.freed.push(p);
            return Ok(());
        }
        if let Some(t) = self.traces.values_mut().find(|t| t.holds(&p)) {
            t.freed.push(p);
            return Ok(());
        }
        // A queued list may still read it, and the next allocation must not
        // hand its slots to an upload that would land first.
        if self.units.iter().any(|u| !u.queued.is_empty()) {
            self.pending_frees.push(p);
            return Ok(());
        }
        self.dram_state()?.alloc.free(&p);
        Ok(())
    }

    /// Queue ops on the movers and wait only at a sync point (the default),
    /// or run each op to completion. Syncs first.
    pub fn set_batching(&mut self, on: bool) -> Result<(), TensorError> {
        if self.capture.is_some() && !on {
            return Err(TraceError::Capturing.into());
        }
        self.sync()?;
        self.batching = on;
        Ok(())
    }

    /// Wait for everything queued on every unit, close its kernels, give back
    /// what was freed meanwhile, and report the first failure -- naming the op
    /// whose list failed -- after recovering every unit (a unit waiting at a
    /// barrier for the one that failed is stopped too).
    pub fn sync(&mut self) -> Result<(), TensorError> {
        let mut first: Option<TensorError> = None;
        for u in 0..self.units.len() {
            if let Err(e) = self.drain_unit(u) {
                first.get_or_insert(e);
            }
        }
        if let Some(e) = first {
            for u in 0..self.units.len() {
                let had_kernels = self.units[u].queued.iter().any(|q| q.kernels);
                self.units[u].queued.clear();
                self.recover(&[(u, had_kernels)]);
            }
            self.barriers = 0;
            if let Some(m) = self.units.first().and_then(|u| u.resident.as_ref()) {
                let (w, t) = (m.window(), self.units[0].tile);
                let _ = self.dev.write32(w, t, tt_isa::dm::BARRIER_COUNTER, 0);
            }
            self.apply_pending_frees();
            self.staging_at = 0;
            self.staging_busy = false;
            return Err(e);
        }
        self.apply_pending_frees();
        // Nothing queued reads host memory any more.
        self.staging_at = 0;
        self.staging_busy = false;
        Ok(())
    }

    fn apply_pending_frees(&mut self) {
        let frees = std::mem::take(&mut self.pending_frees);
        if let Ok(d) = self.dram_state() {
            for p in &frees {
                d.alloc.free(p);
            }
        }
    }

    /// Wait for unit `u`'s queue to drain, closing each list's kernels in
    /// order, and unpin its programs.
    fn drain_unit(&mut self, u: usize) -> Result<(), TensorError> {
        if self.units[u].queued.is_empty() {
            return Ok(());
        }
        self.drains += 1;
        let Session { dev, units, .. } = self;
        let unit = &mut units[u];
        let (Some(r), Some(m)) = (unit.resident.as_mut(), unit.mover.as_mut()) else {
            unit.queued.clear();
            return Ok(());
        };
        let result = m.drain(dev, r.window());
        let ok = result.is_ok();
        while let Some(q) = unit.queued.front() {
            if q.kernels {
                let _ = r.reserved_done(dev, ok);
            }
            if ok {
                unit.queued.pop_front();
            } else {
                break;
            }
        }
        if ok {
            unit.programs.unpin_all();
            self.drain(u)?;
            return Ok(());
        }
        let (what, number) = unit.queued.front().map_or(("", 0), |q| (q.what, q.number));
        Err(TensorError::Shape(format!(
            "queued list {number}, `{what}`, on tile ({}, {}) failed: {}",
            unit.tile.x(),
            unit.tile.y(),
            result.err().map_or_else(String::new, |e| e.to_string())
        )))
    }

    /// Start unit `u`'s roles and mover if they are not running.
    fn ensure_unit(&mut self, u: usize) -> Result<(), TensorError> {
        if self.units[u].resident.is_none() {
            self.prepare_unit(u)?;
        }
        let Session {
            dev,
            units,
            dram,
            profiling,
            barriers,
            epoch,
            ..
        } = self;
        let idle = units.iter().all(|u| u.queued.is_empty());
        let unit = &mut units[u];
        let r = unit.resident.as_ref().expect("prepared above");
        let d = dram
            .as_ref()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        if unit.mover.is_none() {
            // The barrier counter starts again below, and nothing the mover
            // was running survives: traces captured before are stale.
            *epoch += 1;
            // B's progress word starts again from zero, so NC -- which may
            // be waiting on it -- starts again too, when next needed.
            if let Some(nc) = unit.nc.take() {
                nc.stop(dev, r.window())?;
            }
            unit.b_signals = 0;
            unit.nc_signals = 0;
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
            // Unit 0's tile holds the barrier counter, and L1 keeps whatever
            // an earlier session left there: a count already past every
            // target lets each barrier through at once, and multi-unit ops
            // overlap (silicon: a batched four-tile MNIST diverged). The count
            // and the session's barrier number start again together, with
            // nothing queued anywhere that still waits on the old count.
            if u == 0 {
                debug_assert!(idle, "the coordinator restarted under queued barriers");
                dev.write32(r.window(), unit.tile, tt_isa::dm::BARRIER_COUNTER, 0)?;
                *barriers = 0;
            }
        }
        Ok(())
    }

    /// Queue an op's jobs ([`Session::execute`] with batching): each unit's
    /// share as few lists as fit, then -- with more than one unit -- a barrier
    /// on every unit, since the next op may read what any unit wrote.
    fn enqueue_work(&mut self, jobs: Vec<tensor::Job>, budget: u64) -> Result<(), TensorError> {
        let starts: Option<Vec<usize>> = self
            .capture
            .as_ref()
            .map(|c| c.units.iter().map(|u| u.stream.len()).collect());
        let mut what = "";
        let start = self.mark();
        let result = self.enqueue_work_inner(jobs, budget, &mut what);
        self.host.ops += 1;
        self.stage(HostStage::Enqueue, start);
        if let (Some(c), Some(starts)) = (self.capture.as_mut(), starts) {
            // Part of an op captured is no op a replay could run.
            c.failed |= result.is_err();
            c.ops.push(trace::OpRecord {
                what,
                entries: c
                    .units
                    .iter()
                    .zip(starts)
                    .map(|(u, s)| s..u.stream.len())
                    .collect(),
                barrier: self.units.len() > 1,
            });
        }
        result
    }

    fn enqueue_work_inner(
        &mut self,
        jobs: Vec<tensor::Job>,
        budget: u64,
        what: &mut &'static str,
    ) -> Result<(), TensorError> {
        let n = self.units.len();
        let mut queues: Vec<Vec<Step>> = vec![Vec::new(); n];
        for (j, job) in jobs.into_iter().enumerate() {
            queues[j % n].extend(job);
        }
        if n > 1 {
            for u in 0..n {
                self.ensure_unit(u)?;
            }
        }
        // In rounds: every unit's first list, then every unit's second, and
        // so on. A unit whose share is more than its ring holds makes the
        // host wait for room; enqueued unit by unit, that wait came before
        // the other units had anything to do, and they ran one after another
        // (a pipelined 1024^3 matmul on 8 tiles: 9 ms in the call, against
        // 0.4).
        let split = self.scatter_on_nc.is_some() && self.capture.is_none();
        let start = self.mark();
        let mut per_unit: Vec<std::collections::VecDeque<Segment>> = queues
            .into_iter()
            .map(|steps| segments_split(steps, split).into())
            .collect();
        self.stage(HostStage::Segments, start);
        // With more than one unit, a barrier on every unit after its share,
        // since the next op may read what any unit wrote. It rides at the
        // end of the unit's last list where there is room -- a list of its
        // own was a second enqueue a unit an op, half the host's time on
        // many tiles (checklist 9.17) -- except while capturing, whose
        // streams take it as an entry of its own.
        let barrier = (n > 1 && !self.unbarriered).then(|| {
            self.barriers = self.barriers.wrapping_add(1);
            let target = self.barriers.wrapping_mul(n as u32);
            let c = self.units[0].tile;
            [
                tt_isa::dm::op::BARRIER,
                target,
                c.x() as u32,
                c.y() as u32,
                0,
                0,
                0,
                0,
            ]
        });
        let fold = barrier.filter(|_| self.capture.is_none());
        let mut folded = vec![false; n];
        while per_unit.iter().any(|q| !q.is_empty()) {
            for (u, q) in per_unit.iter_mut().enumerate() {
                if let Some(mut seg) = q.pop_front() {
                    if what.is_empty() {
                        *what = seg.what;
                    }
                    if let Some(entry) = fold {
                        if q.is_empty() && seg.entries.len() < tt_isa::dm::LIST_MAX as usize {
                            seg.entries.push(entry);
                            folded[u] = true;
                        }
                    }
                    self.enqueue_segment(u, &seg, budget)?;
                }
            }
        }
        if let Some(entry) = barrier {
            let start = self.mark();
            if let Some(c) = self.capture.as_mut() {
                // Relative to the capture's first: a replay adds its own.
                let rel = entry[1].wrapping_sub(c.barriers_base.wrapping_mul(n as u32));
                c.barriers += 1;
                for uc in &mut c.units {
                    uc.stream
                        .push([entry[0], rel, entry[2], entry[3], 0, 0, 0, 0]);
                }
            }
            for u in (0..n).filter(|&u| !folded[u]) {
                let Session { dev, units, .. } = self;
                let unit = &mut units[u];
                let (r, m) = (
                    unit.resident.as_ref().unwrap(),
                    unit.mover.as_mut().unwrap(),
                );
                let number = m.enqueue(dev, r.window(), &[entry])?;
                unit.queued.push_back(QueuedList {
                    number,
                    kernels: false,
                    what: "barrier",
                });
            }
            self.stage(HostStage::Barrier, start);
        }
        Ok(())
    }

    fn mark(&self) -> (std::time::Instant, tt_device::Traffic) {
        (std::time::Instant::now(), self.dev.traffic())
    }

    fn stage(&mut self, stage: HostStage, since: (std::time::Instant, tt_device::Traffic)) {
        let now = self.dev.traffic();
        self.host.add(stage, since, now);
    }

    /// Where the host's time went queueing ops (checklist 9.17).
    pub fn host_times(&self) -> &HostTimes {
        &self.host
    }

    pub fn reset_host_times(&mut self) {
        self.host = HostTimes::new();
    }

    /// Queue one segment on unit `u`: its programs placed (and pinned until
    /// the queue drains), its kernels reserved, its list enqueued. A kernel
    /// that would write what queued runs read (`Resident::needs_idle`) waits
    /// for the unit to drain first, as does a full program cache.
    fn enqueue_segment(&mut self, u: usize, seg: &Segment, budget: u64) -> Result<(), TensorError> {
        if self.capture.is_some() && !seg.kernel_roles.is_empty() && !seg.resident {
            return Err(TraceError::NotResident.into());
        }
        let start = self.mark();
        self.ensure_unit(u)?;
        self.stage(HostStage::Ensure, start);
        // A drain the descriptors need comes before the programs are placed:
        // a drain unpins every program, and those placed for this list must
        // stay pinned until it has run -- the next placement would otherwise
        // be free to evict them under it.
        let kernel = seg.kernel_roles.first().map(|roles| {
            let [unpack, math, pack] = &**roles;
            Kernel {
                restores_semaphores: true,
                mop: seg.mop,
                loops: [&seg.loops[0], &seg.loops[1], &seg.loops[2]],
                ..Kernel::new([unpack, math, pack], Schedule::Concurrent(&seg.init))
            }
        });
        let start = self.mark();
        if let Some(kernel) = &kernel {
            let idle = {
                let Session { dev, units, .. } = self;
                let r = units[u].resident.as_ref().unwrap();
                r.needs_idle(dev, kernel, seg.resident)
            };
            if idle {
                self.stage(HostStage::IdleCheck, start);
                let start = self.mark();
                self.drain_unit(u)?;
                self.stage(HostStage::Drain, start);
            } else {
                self.stage(HostStage::IdleCheck, start);
            }
        }
        let start = self.mark();
        if !seg.waits.is_empty() {
            self.pipelined += seg.waits.len() as u64;
        }
        let mut entries = seg.entries.clone();
        if seg.resident && !seg.kernels.is_empty() {
            let placed = {
                let Session { dev, units, .. } = self;
                let unit = &mut units[u];
                let w = unit.resident.as_ref().unwrap().window();
                let in_flight = !unit.queued.is_empty();
                match place_programs(dev, w, unit.tile, &mut unit.programs, seg, in_flight) {
                    Ok(p) => Ok(p),
                    Err(PlaceError::Full) => Err(()),
                    Err(PlaceError::Failed(e)) => return Err(e),
                }
            };
            let placed = match placed {
                Ok(p) => p,
                Err(()) => {
                    let d0 = self.mark();
                    self.drain_unit(u)?;
                    self.stage(HostStage::Drain, d0);
                    let Session { dev, units, .. } = self;
                    let unit = &mut units[u];
                    let w = unit.resident.as_ref().unwrap().window();
                    place_programs(dev, w, unit.tile, &mut unit.programs, seg, false)?
                }
            };
            for (&at, p) in seg.kernels.iter().zip(placed) {
                for (t, (addr, len)) in p.into_iter().enumerate() {
                    entries[at][2 + 2 * t] = addr;
                    entries[at][3 + 2 * t] = len;
                }
            }
        }
        self.stage(HostStage::Place, start);
        let start = self.mark();
        if let Some(kernel) = &kernel {
            let Session {
                dev, units, images, ..
            } = self;
            let unit = &mut units[u];
            let r = unit.resident.as_mut().unwrap();
            let generations = r.reserve(
                dev,
                images,
                kernel,
                budget,
                seg.kernels.len() as u32,
                seg.resident,
            )?;
            fill_generations(&mut entries, seg, generations);
        }
        if self.capture.is_some() {
            if let Err(e) = self.capture_segment(u, seg, &entries) {
                let r = self.units[u].resident.as_mut().unwrap();
                if !seg.kernel_roles.is_empty() {
                    let _ = r.reserved_done(&mut self.dev, false);
                }
                return Err(e);
            }
        }
        self.stage(HostStage::Reserve, start);
        if !seg.nc_entries.is_empty() {
            let start = self.mark();
            let r = self.enqueue_nc(u, seg, &mut entries);
            self.stage(HostStage::NcList, start);
            if let Err(e) = r {
                let r = self.units[u].resident.as_mut().unwrap();
                if !seg.kernel_roles.is_empty() {
                    let _ = r.reserved_done(&mut self.dev, false);
                }
                return Err(e);
            }
        }
        let start = self.mark();
        let Session { dev, units, .. } = self;
        let unit = &mut units[u];
        let (r, m) = (
            unit.resident.as_mut().unwrap(),
            unit.mover.as_mut().unwrap(),
        );
        let enqueued = m.enqueue(dev, r.window(), &entries);
        self.stage(HostStage::List, start);
        let Session { dev, units, .. } = self;
        let unit = &mut units[u];
        let r = unit.resident.as_mut().unwrap();
        let number = match enqueued {
            Ok(n) => n,
            Err(e) => {
                if !seg.kernel_roles.is_empty() {
                    let _ = r.reserved_done(dev, false);
                }
                return Err(e.into());
            }
        };
        unit.queued.push_back(QueuedList {
            number,
            kernels: !seg.kernel_roles.is_empty(),
            what: seg.what,
        });
        unit.lists += 1;
        unit.steps += seg.steps;
        unit.b_signals = unit.b_signals.wrapping_add(seg.b_signals);
        Ok(())
    }

    /// Queue `seg`'s NC list on unit `u`, starting NC's mover if it is not
    /// running, and make both lists' `WAIT_PEER` targets absolute (B's in
    /// `entries`). NC's lists are not waited on: B's list ends waiting for
    /// NC, so B's done means NC's is, and an NC failure fails B's wait.
    fn enqueue_nc(
        &mut self,
        u: usize,
        seg: &Segment,
        entries: &mut [[u32; 8]],
    ) -> Result<(), TensorError> {
        let image = self
            .scatter_on_nc
            .expect("an NC list only with the scatters on NC");
        let Session {
            dev, units, dram, ..
        } = self;
        let d = dram
            .as_ref()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        let unit = &mut units[u];
        let w = unit.resident.as_ref().unwrap().window();
        if unit.nc.is_none() {
            let nc = DataMover::start_on(dev, w, unit.tile, &d.dram, tt_isa::dm::Mover::NC, image)?;
            // Writes on NoC #1, reads on NoC #0: each GDDR endpoint stays
            // with one NoC (`DramChannel::owns`), and the two movers' traffic
            // goes out on different NoCs.
            nc.set_write_noc(dev, w, crate::dm::WriteNoc::Noc1)?;
            unit.nc = Some(nc);
            unit.nc_signals = 0;
        }
        for &(at, rel) in &seg.b_waits_on_nc {
            entries[at][2] = unit.nc_signals.wrapping_add(rel);
        }
        let mut nc_entries = seg.nc_entries.clone();
        for &(at, rel) in &seg.nc_waits_on_b {
            nc_entries[at][2] = unit.b_signals.wrapping_add(rel);
        }
        unit.nc.as_mut().unwrap().enqueue(dev, w, &nc_entries)?;
        unit.nc_signals = unit.nc_signals.wrapping_add(seg.nc_signals);
        Ok(())
    }

    /// Add a segment about to be enqueued on unit `u` to the capture, as a
    /// replay will run it: before its kernel, the setup run [`Resident::reserve`]
    /// did for it, if any, as a kernel of its own, and `POKE`s for whatever
    /// of the roles' descriptors the stream has not set; then its entries,
    /// each `KERNEL`'s generation relative and its programs held.
    fn capture_segment(
        &mut self,
        u: usize,
        seg: &Segment,
        entries: &[[u32; 8]],
    ) -> Result<(), TensorError> {
        use tt_isa::dm::op;
        let mut kernel_descriptors = None;
        let mut setup = None;
        if let Some(roles) = seg.kernel_roles.first() {
            let [unpack, math, pack] = &**roles;
            let kernel = Kernel {
                restores_semaphores: true,
                mop: seg.mop,
                loops: [&seg.loops[0], &seg.loops[1], &seg.loops[2]],
                ..Kernel::new([unpack, math, pack], Schedule::Concurrent(&seg.init))
            };
            let Session { dev, units, .. } = self;
            let r = units[u].resident.as_mut().unwrap();
            let mut d = Vec::with_capacity(3);
            for t in 0..3 {
                d.push(r.queued_descriptor(dev, &kernel, t)?);
            }
            kernel_descriptors = Some((d, kernel.dst_fmt));
            setup = r.take_setup_run();
        }
        let mut held = Vec::new();
        let mut out = std::mem::take(&mut self.capture.as_mut().unwrap().units[u]);
        let base = out.generation_base;
        let push_window = if self.dev.transport().is_simulated() {
            tt_isa::mailbox::SIM_PUSH_WINDOW
        } else {
            0
        };
        let result = (|| {
            if let (Some((generation, program)), Some((_, dst_fmt))) = (&setup, &kernel_descriptors)
            {
                // As the host staged it (`Resident::begin`): thread 0 alone,
                // the others given nothing to run.
                let at = self.place_held(u, program)?;
                held.push(at);
                out.poke_descriptor(
                    0,
                    &tt_isa::mailbox::Descriptor {
                        thread_index: 0,
                        dst_access_fmt: *dst_fmt,
                        push_window,
                        ..Default::default()
                    },
                );
                let (at, len) = (at as u32, program.len() as u32);
                let rel = generation.wrapping_sub(base);
                out.stream.push([op::KERNEL, rel, at, len, at, 0, at, 0]);
            }
            if let Some((descriptors, _)) = &kernel_descriptors {
                for (t, d) in descriptors.iter().enumerate() {
                    out.poke_descriptor(t, d);
                }
            }
            for e in entries {
                let mut e = *e;
                if e[0] == op::KERNEL {
                    e[1] = e[1].wrapping_sub(base);
                    for t in 0..3 {
                        let at = e[2 + 2 * t] as u64;
                        if at != 0 && self.units[u].programs.hold(at) {
                            held.push(at);
                        }
                    }
                }
                out.stream.push(e);
            }
            Ok(())
        })();
        out.held_programs.extend(held);
        self.capture.as_mut().unwrap().units[u] = out;
        result
    }

    /// Put `program` in unit `u`'s program cache and hold it there for a
    /// trace: its address.
    fn place_held(&mut self, u: usize, program: &[Instruction]) -> Result<u64, TensorError> {
        use crate::program_cache::{CacheError, Placed};
        let words: Vec<u32> = program.iter().map(|i| i.word()).collect();
        for attempt in 0..2 {
            let Session { dev, units, .. } = self;
            let unit = &mut units[u];
            let w = unit.resident.as_ref().unwrap().window();
            let at = match unit.programs.place(&words) {
                Ok(Placed::Hit(at)) => at,
                Ok(Placed::Upload(at)) => {
                    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                    dev.l1_write(w, unit.tile, at, &bytes)?;
                    at
                }
                Ok(Placed::Bypass) => return Err(TraceError::NotResident.into()),
                // What queued lists run is pinned: room once they are done.
                Err(CacheError::Full { .. }) if attempt == 0 => {
                    self.drain_unit(u)?;
                    continue;
                }
                Err(CacheError::Full { bytes }) => {
                    return Err(TensorError::Shape(format!(
                        "no {bytes} bytes in the program cache beside what traces hold"
                    )))
                }
            };
            unit.programs.hold(at);
            return Ok(at);
        }
        unreachable!("the second attempt returns")
    }

    fn refuse_while_capturing(&self, what: &'static str) -> Result<(), TensorError> {
        match self.capture {
            Some(_) => Err(TraceError::HostTransfer(what).into()),
            None => Ok(()),
        }
    }

    /// Start capturing a trace (`crate::trace`): every op from here to
    /// [`Session::end_trace`] runs as usual and is recorded, to be run again
    /// by [`Session::replay`]. Only on a batching session. Waits for what is
    /// queued, and forgets what the host knows of the tiles' descriptors and
    /// semaphores, so the capture records all its kernels need.
    pub fn begin_trace(&mut self) -> Result<(), TensorError> {
        if self.capture.is_some() {
            return Err(TraceError::Capturing.into());
        }
        if !self.batching {
            return Err(TraceError::NotBatching.into());
        }
        self.dram_state()?;
        self.sync()?;
        for u in 0..self.units.len() {
            self.ensure_unit(u)?;
        }
        let units = self
            .units
            .iter_mut()
            .map(|unit| {
                let r = unit.resident.as_mut().expect("started above");
                r.forget_tile_state();
                // A setup run before the capture is not the capture's.
                let _ = r.take_setup_run();
                trace::UnitCapture {
                    generation_base: r.generation(),
                    ..Default::default()
                }
            })
            .collect();
        self.capture = Some(trace::Capture {
            epoch: self.epoch,
            barriers_base: self.barriers,
            barriers: 0,
            units,
            freed: Vec::new(),
            ops: Vec::new(),
            failed: false,
        });
        Ok(())
    }

    /// End the capture: wait for its ops, and store each unit's stream in
    /// GDDR. On any failure nothing is kept, and what the capture held is
    /// given back.
    pub fn end_trace(&mut self) -> Result<TraceId, TensorError> {
        let Some(capture) = self.capture.take() else {
            return Err(TraceError::NotCapturing.into());
        };
        let held: Vec<Vec<u64>> = capture
            .units
            .iter()
            .map(|u| u.held_programs.clone())
            .collect();
        let result = self.store_trace(capture);
        if result.is_err() {
            for (u, held) in held.iter().enumerate() {
                for &at in held {
                    self.units[u].programs.release(at);
                }
            }
        }
        result
    }

    fn store_trace(&mut self, capture: trace::Capture) -> Result<TraceId, TensorError> {
        let synced = self.sync();
        let trace::Capture {
            epoch,
            barriers,
            units,
            freed,
            ops,
            failed,
            ..
        } = capture;
        // What the capture deferred is the session's to give back now.
        let give_back = |s: &mut Self, freed: Vec<tensor::Placement>| {
            for p in freed {
                let _ = s.free_placement(p);
            }
        };
        if let Err(e) = synced {
            give_back(self, freed);
            return Err(e);
        }
        if epoch != self.epoch {
            give_back(self, freed);
            return Err(TraceError::Stale.into());
        }
        if failed {
            give_back(self, freed);
            return Err(TensorError::Shape(
                "an op failed during the trace capture: nothing was kept".into(),
            ));
        }
        if units.iter().all(|u| u.stream.is_empty()) {
            give_back(self, freed);
            return Err(TraceError::Empty.into());
        }
        let mut stored: Vec<Option<trace::UnitTrace>> = Vec::with_capacity(units.len());
        let mut failure = None;
        for (u, uc) in units.into_iter().enumerate() {
            if uc.stream.is_empty() {
                stored.push(None);
                continue;
            }
            match self.store_stream(u, &uc) {
                Ok(t) => stored.push(Some(t)),
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = failure {
            for t in stored.into_iter().flatten() {
                if let Ok(d) = self.dram_state() {
                    d.alloc.free(&t.stream);
                }
            }
            give_back(self, freed);
            return Err(e);
        }
        let free_at_end = self.dram_state()?.alloc.snapshot();
        let id = self.next_trace;
        self.next_trace += 1;
        self.traces.insert(
            id,
            trace::Trace {
                epoch,
                units: stored,
                barriers,
                free_at_end,
                freed,
                ops,
            },
        );
        Ok(TraceId(id))
    }

    /// Unit `u`'s captured stream, laid out for streaming and written to GDDR.
    fn store_stream(
        &mut self,
        u: usize,
        uc: &trace::UnitCapture,
    ) -> Result<trace::UnitTrace, TensorError> {
        let words = trace::chunked(&uc.stream);
        let bytes: Vec<u8> = words
            .iter()
            .flatten()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let generations = {
            let r = self.units[u].resident.as_ref().unwrap();
            r.generation().wrapping_sub(uc.generation_base)
        };
        let Session { dev, dram, .. } = self;
        let d = dram.as_mut().expect("checked at the capture's start");
        // Each unit's on its own channel where there are enough, so the
        // movers' reads of their streams spread out.
        let stream = d
            .alloc
            .alloc_on(u % d.alloc.channel_count(), bytes.len() as u64)?;
        let range = stream.region().expect("a one-channel placement");
        // Whole slots, as every GDDR write here is.
        let mut bytes = bytes;
        bytes.resize(range.len() as usize, 0);
        let written = dev.dram_write(&d.w4, range, &bytes);
        if let Err(e) = written {
            d.alloc.free(&stream);
            return Err(e.into());
        }
        Ok(trace::UnitTrace {
            channel: range.channel().index() as u32,
            offset: range.offset() as u32,
            count: words.len() as u32,
            generations,
            held_programs: uc.held_programs.clone(),
            stream,
        })
    }

    /// Run trace `id` again: one `CALL` entry on each unit's mover
    /// (`tt_isa::dm::op::CALL`), queued like any op. Between replays,
    /// [`Session::write`] puts new values into the tensors it reads.
    pub fn replay(&mut self, id: TraceId) -> Result<(), TensorError> {
        if self.capture.is_some() {
            return Err(TraceError::Capturing.into());
        }
        if !self.batching {
            return Err(TraceError::NotBatching.into());
        }
        if !self.traces.contains_key(&id.0) {
            return Err(TraceError::Unknown(id.0).into());
        }
        // A mover stopped since (a failed list) starts here, which moves the
        // epoch on as well.
        for u in 0..self.units.len() {
            self.ensure_unit(u)?;
        }
        let t = &self.traces[&id.0];
        if t.epoch != self.epoch {
            return Err(TraceError::Stale.into());
        }
        let calls: Vec<Option<[u32; 4]>> = t
            .units
            .iter()
            .map(|ut| {
                ut.as_ref()
                    .map(|ut| [ut.channel, ut.offset, ut.count, ut.generations])
            })
            .collect();
        let barriers = t.barriers;
        let n = self.units.len() as u32;
        let barrier_base = self.barriers.wrapping_mul(n);
        let Session { dev, units, .. } = self;
        for (unit, call) in units.iter_mut().zip(calls) {
            let Some([channel, offset, count, generations]) = call else {
                continue;
            };
            let r = unit.resident.as_mut().unwrap();
            let generation_base = r.take_generations(generations).wrapping_sub(1);
            let entry = [
                tt_isa::dm::op::CALL,
                channel,
                offset,
                count,
                generation_base,
                barrier_base,
                0,
                0,
            ];
            let m = unit.mover.as_mut().unwrap();
            let number = m.enqueue(dev, r.window(), &[entry])?;
            unit.queued.push_back(QueuedList {
                number,
                kernels: false,
                what: "trace",
            });
            unit.lists += 1;
            // The replay sets the roles' descriptors and semaphores without
            // the host.
            r.forget_tile_state();
        }
        self.barriers = self.barriers.wrapping_add(barriers);
        Ok(())
    }

    /// Give trace `id` back: once nothing queued runs it, its programs may be
    /// evicted, its stream's slots are free, and the frees it deferred happen.
    pub fn release_trace(&mut self, id: TraceId) -> Result<(), TensorError> {
        if !self.traces.contains_key(&id.0) {
            return Err(TraceError::Unknown(id.0).into());
        }
        self.sync()?;
        let t = self.traces.remove(&id.0).expect("checked above");
        for (unit, ut) in self.units.iter_mut().zip(t.units) {
            let Some(ut) = ut else { continue };
            for at in ut.held_programs {
                unit.programs.release(at);
            }
            if let Some(d) = self.dram.as_mut() {
                d.alloc.free(&ut.stream);
            }
        }
        for p in t.freed {
            // Another trace may hold it too.
            self.free_placement(p)?;
        }
        Ok(())
    }

    /// The ops trace `id` captured, in order (`crate::trace::OpRecord`).
    pub fn trace_ops(&self, id: TraceId) -> Result<&[trace::OpRecord], TensorError> {
        self.traces
            .get(&id.0)
            .map(|t| &t.ops[..])
            .ok_or(TraceError::Unknown(id.0).into())
    }

    /// Is a capture open?
    pub fn capturing(&self) -> bool {
        self.capture.is_some()
    }

    /// Overwrite `t`'s values in place (`DramTensor::write`): a trace's input
    /// between replays. Queued behind what is queued, which may read it
    /// (`Session::dma_upload`); through the BAR, it waits for that instead.
    pub fn write(&mut self, t: &DramTensor, values: &[f32]) -> Result<(), TensorError> {
        self.refuse_while_capturing("write")?;
        t.expect("a write of FP32 values", tensor::Elem::F32)?;
        self.write_src(t, Src::F32(values), false)
    }

    /// Free GDDR bytes on the fullest channel.
    pub fn dram_free_bytes(&self) -> u64 {
        self.dram.as_ref().map_or(0, |d| d.alloc.free_bytes())
    }

    /// Element-wise `a (op) b` in GDDR, on the SFPU ([`tensor::sfpu_eltwise`]).
    /// An op with no SFPU program is refused: the data mover does no
    /// arithmetic.
    pub fn eltwise(
        &mut self,
        op: tensor::Eltwise,
        a: &DramTensor,
        b: Option<&DramTensor>,
    ) -> Result<DramTensor, TensorError> {
        self.eltwise3(op, a, b, None)
    }

    /// [`Session::eltwise`] with a ternary op's third operand
    /// (`sfpu::ops::kind_sfpu::MASK_WHERE`): the SFPU's alone.
    pub fn eltwise3(
        &mut self,
        op: tensor::Eltwise,
        a: &DramTensor,
        b: Option<&DramTensor>,
        c: Option<&DramTensor>,
    ) -> Result<DramTensor, TensorError> {
        use tensor::OpPadding;
        let units = self.units.len();
        let pipeline = self.pipeline && self.capture.is_none();
        let alloc = &mut self.dram_state()?.alloc;
        let (kind, bcast) = tensor::broadcast_of(op, a, b)?;
        // The SFPU's, or refused: the mover only moves data.
        let Some(work) = tensor::sfpu_eltwise(alloc, op, a, b, c, units, pipeline)? else {
            return Err(TensorError::Shape(format!(
                "element-wise {kind:#x} with {bcast:?}: no SFPU program computes it"
            )));
        };
        let out = self.execute(work, RESET_BUDGET)?;
        out.set_pad(op.produces(&[Some(a), b, c].into_iter().flatten().collect::<Vec<_>>()));
        Ok(out)
    }

    /// `x^y` as `powf` (S4, 10.2d): `e^(y ln|x|)` and the special values, one
    /// SFPU op (`POW`, `POW_S`, `POW_I`); within `sfpu::ops::pow_bound`. `y`
    /// the same shape as `x`, a scalar, or an `I32` tensor (`as f32` in the
    /// program).
    pub fn pow(&mut self, x: &DramTensor, y: PowExponent<'_>) -> Result<DramTensor, TensorError> {
        use crate::sfpu::ops::kind_sfpu::*;
        let op = |kind, scalar| tensor::Eltwise {
            kind,
            scalar,
            scalar2: 0.0,
        };
        let (op, t) = match y {
            PowExponent::Tensor(t) => (op(POW, 0.0), Some(t)),
            PowExponent::Int(t) => (op(POW_I, 0.0), Some(t)),
            PowExponent::Scalar(s) => (op(POW_S, s), None),
        };
        if let Some(t) = t {
            if (t.rows, t.cols) != (x.rows, x.cols) {
                return Err(TensorError::Shape(format!(
                    "pow: an exponent [{}, {}] for [{}, {}]",
                    t.rows, t.cols, x.rows, x.cols
                )));
            }
        }
        self.eltwise(op, x, t)
    }

    /// `t`, bit for bit, as a tensor of its own ([`tensor::copy`]): data
    /// movement only, any element type.
    pub fn copy(&mut self, t: &DramTensor) -> Result<DramTensor, TensorError> {
        let units = self.units.len();
        let work = tensor::copy(&mut self.dram_state()?.alloc, t, units)?;
        let out = self.execute(work, RESET_BUDGET)?;
        out.set_pad(if t.pad() == tensor::Pad::Zero {
            tensor::Pad::Zero
        } else {
            tensor::Pad::Undefined
        });
        Ok(out)
    }

    /// The sum over rows of `a`, `[1, cols]`, in `burn-flex`'s order
    /// ([`tensor::sum_rows`]).
    pub fn sum_rows(&mut self, a: &DramTensor) -> Result<DramTensor, TensorError> {
        use tensor::OpPadding;
        let units = self.units.len();
        let pipeline = self.pipeline && self.capture.is_none();
        let work = tensor::sum_rows(&mut self.dram_state()?.alloc, a, units, pipeline)?;
        let out = self.execute(work, RESET_BUDGET)?;
        out.set_pad(tensor::SumRows.produces(&[a]));
        Ok(out)
    }

    /// `a` reduced over `axis` by `op`: `[rows, 1]` over columns, `[1, cols]`
    /// over rows. On the SFPU ([`tensor::sfpu_reduce`]); a sum over rows in
    /// Flex's order ([`Session::sum_rows`]).
    /// Padding is masked in the kernel, so `a`'s is never read; the output's
    /// padding holds copies of the result (over columns) or is undefined.
    pub fn reduce(
        &mut self,
        a: &DramTensor,
        op: crate::sfpu::reduce::ReduceOp,
        axis: crate::sfpu::reduce::Axis,
    ) -> Result<DramTensor, TensorError> {
        use crate::sfpu::reduce::{Axis, ReduceOp};
        if (op, axis) == (ReduceOp::Sum, Axis::Rows) {
            return self.sum_rows(a);
        }
        let units = self.units.len();
        let pipeline = self.pipeline && self.capture.is_none();
        let work =
            tensor::sfpu_reduce(&mut self.dram_state()?.alloc, a, op, axis, units, pipeline)?;
        let out = self.execute(work, RESET_BUDGET)?;
        out.set_pad(tensor::Pad::Undefined);
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
            self.submit_jobs(jobs, RESET_BUDGET)?;
            t.set_pad(Pad::Zero);
            return Ok(None);
        }
        let copy = self.copy(t)?;
        let jobs = tensor::fill_pad(&copy, 0.0, units)?;
        if let Err(e) = self.submit_jobs(jobs, RESET_BUDGET) {
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
        // Before any padding fill: an integer tensor is refused untouched.
        a.expect("a matmul", tensor::Elem::F32)?;
        b.expect("a matmul", tensor::Elem::F32)?;
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
        // No `MOP` in the matmul: with its loops replayed it is backend-bound,
        // and the math role's `MOP` measured no faster end to end on silicon
        // (row AE) -- while ttsim's FIFO overflows under one (row 68), so the
        // default would be a path only silicon runs. `step36_mop` and
        // `step37_loops` keep the expander gated.
        let allow_mop = false;
        let pipeline = self.pipeline && self.capture.is_none();
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
                    allow_mop,
                    pipeline,
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
        if self.batching {
            let queued = self.enqueue_work(jobs, budget);
            // A profile drains each tile's stream after every op, so its
            // buffer never holds more than one op's events.
            let synced = match &queued {
                Ok(()) if self.profiling.is_some() => self.sync(),
                Ok(()) => Ok(()),
                Err(_) => {
                    // Part of the op may be queued without its barrier:
                    // drain everything and start the barriers again.
                    let _ = self.sync();
                    self.barriers = 0;
                    let c = self.units[0].tile;
                    if let Some(r) = self.units[0].resident.as_ref() {
                        let _ = self
                            .dev
                            .write32(r.window(), c, tt_isa::dm::BARRIER_COUNTER, 0);
                    }
                    Ok(())
                }
            };
            if let Err(e) = queued.and(synced) {
                if let Ok(d) = self.dram_state() {
                    d.alloc.free(&out.placement);
                }
                return Err(e);
            }
            return Ok(out);
        }
        if let Err(e) = self.run_jobs(jobs, budget) {
            if let Ok(d) = self.dram_state() {
                d.alloc.free(&out.placement);
            }
            return Err(e);
        }
        Ok(out)
    }

    /// Jobs whose output already exists (an in-place fill), by whichever path
    /// the session dispatches on: queued behind what is queued, or run now.
    /// Never the direct path while lists are queued: it writes the mover's
    /// list ring over them.
    fn submit_jobs(&mut self, jobs: Vec<tensor::Job>, budget: u64) -> Result<(), TensorError> {
        if self.batching {
            self.enqueue_work(jobs, budget)
        } else {
            self.run_jobs(jobs, budget)
        }
    }

    /// The body of [`Session::execute`], for jobs whose output already exists
    /// (an in-place fill): deal them out, run them in waves, recover a unit
    /// whose list failed.
    fn run_jobs(&mut self, jobs: Vec<tensor::Job>, budget: u64) -> Result<(), TensorError> {
        debug_assert!(
            self.units.iter().all(|u| u.queued.is_empty()),
            "the direct path would write the list ring under queued lists"
        );
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
        self.pipelined += seg.waits.len() as u64;
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
            let placed = place_programs(dev, window, unit.tile, &mut unit.programs, seg, false)?;
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
                    mop: seg.mop,
                    loops: [&seg.loops[0], &seg.loops[1], &seg.loops[2]],
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
                fill_generations(&mut entries, seg, generations);
                Some(kernel)
            }
        };
        let mover = unit.mover.as_mut().expect("started above");
        if let Err(e) = mover.submit_list(dev, r.window(), &entries) {
            if kernel.is_some() {
                let _ = r.reserved_done(dev, false);
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
        if kernel.is_some() {
            r.reserved_done(dev, out.is_ok())?;
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
        self.sync_run()?;
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

    /// Double-buffer GDDR matmuls from the next op on (checklist 9.15; on by
    /// default): each block staged in half the data arena, a unit's
    /// consecutive blocks in alternate halves, so its mover gathers the next
    /// block and scatters the last while the roles compute one
    /// (`tt_isa::dm::op::LAUNCH`, `KERNEL_WAIT`). Only where it pays
    /// (`tensor::pipelining_pays`): two blocks a unit or more, and at most
    /// twice the operand tiles gathered. The bits are the same. Not while a
    /// trace is being captured.
    pub fn set_pipeline(&mut self, on: bool) {
        self.pipeline = on;
    }

    /// Write pipelined groups' outputs out from each tile's RISCV NC rather
    /// than B (checklist 9.15, the reader / writer split), from the next op
    /// on: `Some(image)` with `tt_firmware_images::DM_NC`'s bytes, `None`
    /// (the default) for every move on B. B gathers and runs the kernels;
    /// NC scatters each block once B signals its kernel done, writing on
    /// NoC #1. The bits are the same. Not while a trace is being captured.
    pub fn set_scatter_mover(&mut self, nc_image: Option<&'static [u8]>) {
        self.scatter_on_nc = nc_image;
    }

    /// How many blocks have run overlapped with their neighbours' moves
    /// (`Session::set_pipeline`), since the session opened.
    pub fn pipelined_blocks(&self) -> u64 {
        self.pipelined
    }

    /// How many times the host has waited for a unit's queued lists to finish
    /// before it could go on enqueueing -- for programs to place, or role
    /// configuration to change -- since the session opened. Each one stops
    /// that unit's work overlapping the host's.
    pub fn drains(&self) -> u64 {
        self.drains
    }

    /// Whether a profile records the roles' events as well as the mover's
    /// (the default; `Resident::set_profile_roles`). A profile of pipelined
    /// work wants the mover's alone.
    pub fn set_profile_roles(&mut self, on: bool) {
        self.profile_roles = on;
    }

    /// Steps of GDDR ops completed on each unit so far, in unit order.
    pub fn steps_per_tile(&self) -> Vec<u64> {
        self.units.iter().map(|u| u.steps).collect()
    }

    /// Each unit's program cache counters, in unit order.
    pub fn program_cache_stats(&self) -> Vec<crate::program_cache::CacheStats> {
        self.units.iter().map(|u| u.programs.stats()).collect()
    }

    /// Shrink every unit's program cache to its first `bytes`, emptied: for
    /// gates that need it to fill (`step34_batching`). Waits for queued work
    /// first, since what is resident is forgotten.
    #[doc(hidden)]
    pub fn limit_program_cache(&mut self, bytes: u64) -> Result<(), TensorError> {
        self.sync()?;
        let full = tt_isa::l1::PROGRAM_CACHE;
        for unit in &mut self.units {
            unit.programs = ProgramCache::new(tt_isa::l1::Region {
                end: full.base + bytes.min(full.len()),
                ..full
            });
        }
        Ok(())
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
    /// [`Session::sync`] for the paths that report a `RunError`.
    /// Wait for what is queued before a host-run kernel -- which no trace can
    /// capture, so one during a capture is refused.
    fn sync_run(&mut self) -> Result<(), RunError> {
        self.refuse_while_capturing("host-run kernel")
            .map_err(|e| RunError::Queued(e.to_string()))?;
        self.sync().map_err(|e| RunError::Queued(e.to_string()))
    }

    /// What the in-flight cap has cost every unit's mover since it started
    /// (`DataMover::throttle`), summed: requests that waited for room, and
    /// the cycles they waited. A PCIe read pair per unit. Not reported
    /// otherwise: the tile cap (`dm::TILE_IN_FLIGHT_CAP`) is chosen for
    /// fairness between tiles, so any large move waits under it by design --
    /// a host DMA's 266 KB batches always do -- and a warning for it was noise.
    pub fn throttle(&mut self) -> Result<Throttle, TensorError> {
        let mut sum = Throttle::default();
        for u in 0..self.units.len() {
            let Session { dev, units, .. } = self;
            let unit = &units[u];
            if let (Some(r), Some(m)) = (unit.resident.as_ref(), unit.mover.as_ref()) {
                sum = sum + m.throttle(dev, r.window())?;
            }
        }
        Ok(sum)
    }

    pub fn into_device(mut self) -> Device<T> {
        let _ = self.sync();
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
    use super::{healthy_tiles, runtime, segments, Instruction, RunError, SessionError, Step};
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
                    mop: Box::new([None; 3]),
                    loops: Default::default(),
                    half: None,
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
                mop: Box::new([None; 3]),
                loops: Default::default(),
                half: None,
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
            mop: Box::new([None; 3]),
            loops: Default::default(),
            half: None,
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
                mop: Box::new([None; 3]),
                loops: Default::default(),
                half: None,
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

    /// A wedged tile is passed over for the next healthy one, and the search
    /// stops once enough are found -- a wedged tile after that is never probed.
    #[test]
    fn wedged_tiles_are_skipped_for_the_next_healthy_one() {
        let wedged = [2, 5];
        let mut probed = Vec::new();
        let (good, bad) = healthy_tiles(&[1, 2, 3, 4, 5, 6], 3, |t| {
            probed.push(t);
            Ok::<_, ()>(!wedged.contains(&t))
        })
        .unwrap();
        assert_eq!(good, [1, 3, 4]);
        assert_eq!(bad, [2]);
        assert_eq!(probed, [1, 2, 3, 4], "5 is never reached");
        // Too few healthy: every candidate tried, the caller reports the gap.
        let (good, bad) =
            healthy_tiles(&[2, 5, 7], 2, |t| Ok::<_, ()>(!wedged.contains(&t))).unwrap();
        assert_eq!((good, bad), (vec![7], vec![2, 5]));
        // Any other failure ends the search.
        assert_eq!(
            healthy_tiles(&[1, 2], 2, |t| if t == 1 { Err("io") } else { Ok(true) }),
            Err("io")
        );
    }

    /// Both errors say the tile, that software cannot clear it, and what does.
    #[test]
    fn a_wedged_tile_says_what_clears_it() {
        let e = RunError::Wedged {
            tile: (1, 2),
            roles: vec![tt_isa::tensix::Core::T1],
        };
        let text = e.to_string();
        assert!(text.contains("tile (1,2) is wedged"), "{text}");
        assert!(text.contains("T1") && text.contains("tt-smi -r"), "{text}");
        let e = SessionError::TooFewHealthy {
            asked: 4,
            healthy: 3,
            wedged: vec![(1, 2)],
        };
        let text = e.to_string();
        assert!(
            text.contains("only 3") && text.contains("(1, 2)") && text.contains("tt-smi -r"),
            "{text}"
        );
    }
}

/// What [`Session::pow`] raises to.
#[derive(Copy, Clone, Debug)]
pub enum PowExponent<'a> {
    /// An `F32` tensor of the base's shape.
    Tensor(&'a DramTensor),
    /// One value.
    Scalar(f32),
    /// An `I32` tensor of the base's shape, `as f32` first.
    Int(&'a DramTensor),
}
