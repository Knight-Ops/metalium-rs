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
//! units round-robin and submitted in rounds to their command queues. Compute
//! regions use B reads, resident roles and NC writes; cross-tile barriers join
//! completed outputs before dependent operations. The units compute at the same time -- on
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
pub(crate) const RESET_BUDGET: u64 = 400_000;

/// How many of a session's units an op runs on ([`Session::set_spread`]).
///
/// Each unit an op runs on costs the host about 5 us an op (its list built, checked and
/// enqueued: `silicon_bench_path::host_time_per_op`), so on many tiles a small op is the
/// host's, and the best tile count differs by workload (card 1, 2026-10-14: the `tt-mnist`
/// transformer fastest on 2 tiles, MNIST on 8). Choosing per op lets a session open every
/// tile and still give a small op only a few.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Spread {
    /// Every unit: what every op did before this setting existed.
    #[default]
    All,
    /// At most this many units.
    Fixed(usize),
    /// One unit for each `share` of the op's work, at least `min` and at most every unit.
    /// The work is output tiles for an element-wise op, input tiles for a reduction, tiles
    /// moved for a copy or a gather, and tile products (`mt * nt * kt`) for a matmul. The
    /// floor matters: a small op on one unit leaves that unit doing every op's fixed cost
    /// alone (the `tt-mnist` transformer at a share of 16, nearly every op on one tile, was
    /// as slow as a one-tile session).
    Work { share: usize, min: usize },
}

/// The units a one-tile-row product's weight stream saturates GDDR with: each streams ~7 GB/s,
/// and 64 reach ~450 GB/s dealt a column a unit (Blackhole, card 0,
/// `docs/learnings/silicon-operating-notes.md`, "A decode product's weights"). Past it, SmolLM2-1.7B's
/// `q|k|v`, `gate|up` and head ran slower on 120 units than on 64 (180 / 383 / 1,051 against 160 /
/// 352 / 1,024 us).
pub const THIN_STREAM_UNITS: usize = 64;

/// The `K` (tiles) from which a one-tile-row product is a weight stream ([`THIN_STREAM_UNITS`]):
/// with a short `K` each unit's `A` reads and kernel are a larger share, and more units help (the
/// 135M head, `K` = 18 tiles: 308 us on 120 units, 318 on 64).
pub const THIN_STREAM_K_TILES: usize = 32;

/// Most units a product's weight prefetch takes ([`Session::set_weight_prefetch`]): each streams
/// ~7 GB/s, and the ops the prefetch runs under read GDDR too.
pub const PREFETCH_UNITS: usize = 40;

/// The `K` (tiles) from which a one-row product packs its `A` ([`Session::set_packed_thin_a`]):
/// every unit's thin `A` reads cost ~0.27 us an entry (64 at `K` = 2048, 256 at 8192), the packing
/// op one read and two writes a tile spread over every unit.
pub const PACKED_A_K_TILES: usize = 32;

/// How [`Session::probe_stream`] deals a tensor's tiles over its units.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamOrder {
    /// Unit `u` reads columns `u, u + units, ..`, each down its tile rows: how an `m = 1`
    /// product deals its `B`, a column a job.
    Columns,
    /// Each unit reads one contiguous share of the tiles, row-major: consecutive tiles, so
    /// consecutive channels.
    Rows,
}

impl Spread {
    /// The units an op with `work` (as [`Spread::Work`] counts it) runs on, of `units`.
    pub fn units_for(self, work: usize, units: usize) -> usize {
        let n = units.max(1);
        match self {
            Spread::All => n,
            Spread::Fixed(k) => k.clamp(1, n),
            Spread::Work { share, min } => work.div_ceil(share.max(1)).max(min).clamp(1, n),
        }
    }
}

/// Tiles of `t`, the unit [`Spread::Work`] counts in.
fn tiles_of(t: &DramTensor) -> usize {
    let [r, c] = t.grid();
    r * c
}

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
    /// `TT_MATH` named no math mode (`sfpu::approx::MathMode`).
    Math(String),
    /// Another descriptor -- another process, or another session in this one
    /// -- holds the card ([`tt_kmd::Kmd::lock_card`]).
    CardBusy {
        card: u16,
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
            SessionError::CardBusy { card } => write!(
                f,
                "card {card} is in use by another process, or another session in this one"
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
            SessionError::Math(e) => write!(f, "{e}"),
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

/// Host-side ownership region and batch counts since opening.
#[derive(Clone, Debug, Default)]
pub struct DataflowStats {
    pub regions: u64,
    pub batches: u64,
    /// Slot reuses that waited only for the earlier batch's pack, so the
    /// gather overlapped NC writing it out.
    pub pack_waits: u64,
    /// Slot reuses that waited for NC's release: the gather lands on what an
    /// earlier scatter reads, or its footprint is not modelled.
    pub release_waits: u64,
    /// Packets of standalone transfers (no kernel): B reads, NC writes.
    pub transfer_packets: u64,
}

/// A best-effort snapshot; input/output are received and consumed page counts.
/// Waits are ordered B, T0, T1, T2, NC. Counts wrap modulo 2^16.
#[derive(Clone, Debug, Default)]
pub struct DataflowProgress {
    pub input: [u16; 2],
    pub output: [u16; 2],
    pub abort: u32,
    pub roles: [u32; 3],
    pub waits: [u32; 5],
}

/// How [`Session::matmul_dram_batched_with`] lays out its left operand and
/// output when a product's `m` is not whole tiles.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BatchedLayout {
    /// `a` holds a block per whole tile rows already (a previous product's
    /// output left unstacked): each left block starts on a tile row and the
    /// rows to the next block are its own, whatever they hold -- they are `M`,
    /// so they only make output rows that are not read. Never gathered.
    pub lhs_padded: bool,
    /// The output is the stacked `[batch m, n]`; otherwise `[batch mt 32, n]`,
    /// product `i` from tile row `i mt`, its tail rows the products of the left
    /// block's tail rows.
    pub stacked: bool,
}

impl BatchedLayout {
    /// The products stacked, every operand gathered as it needs.
    pub const STACKED: BatchedLayout = BatchedLayout {
        lhs_padded: false,
        stacked: true,
    };
}

/// A chip opened for compute on one or more Tensix tiles, in the order the
/// module documentation gives.
pub struct Session<T: Transport> {
    dataflow: DataflowStats,
    dev: Device<T>,
    units: Vec<Unit>,
    grid: Tensix,
    images: RoleImages<'static>,
    /// Every run's phases since the last [`Session::take_profile`].
    /// Whether GDDR matmuls double-buffer their blocks so a unit's moves
    /// overlap its kernels (`Session::set_pipeline`, checklist 9.15). On by
    /// default.
    pipeline: bool,
    /// How many units an op runs on ([`Session::set_spread`]).
    spread: Spread,
    matmul_k_block_limit: Option<std::num::NonZeroUsize>,
    /// [`Session::set_thin_matmul_reads`].
    thin_matmul_reads: bool,
    /// [`Session::set_packed_thin_a`].
    packed_thin_a: bool,
    /// [`Session::set_packed_rms`].
    packed_rms: bool,
    /// [`Session::set_streamed_k`].
    streamed_k: bool,
    /// [`Session::set_weight_prefetch`].
    weight_prefetch: bool,
    /// While a prefetching product is queued in a capture: the barrier its `PREFETCH` lists wait
    /// for (`enqueue_work_inner`).
    prefetch_anchor: Option<u32>,
    /// [`Session::set_resident_matmul_a`].
    resident_matmul_a: bool,
    /// [`Session::set_unpack_transposed_b`].
    unpack_transposed_b: bool,
    /// [`Session::set_thin_stream_units`].
    thin_stream_units: bool,
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
    /// The caller's name for the ops it queues now ([`Session::set_op_label`]),
    /// kept in a capture's [`trace::OpRecord`]s.
    op_label: &'static str,
    /// Where tensors take and lose the tile layout ([`Session::set_tilize`]).
    tilize: Tilize,
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
    /// The profile mode ([`Session::profile_mode`]): what the stamps are attributed to, and each
    /// unit's stamps since the last attribution.
    profiler: Option<crate::profile_mode::Profiler>,
    /// Count what each captured op moves through GDDR when its trace is stored
    /// ([`Session::set_count_trace_bytes`]); the profile mode counts whether or not it is set.
    count_trace_bytes: bool,
    profile_drained: Vec<Vec<tt_device::trace::TraceEvent>>,
    /// Where each unit's timestamper stream goes, and its bytes
    /// ([`Session::set_profile_buffer`]): the fixed trace buffer unless moved.
    profile_buffer: (u64, u64),
    /// Queue ops on the movers and wait only at a sync point
    /// ([`Session::sync`]), rather than one host round trip per op.
    batching: bool,
    /// How the transcendentals run ([`Session::set_math_mode`]).
    math: crate::sfpu::approx::MathMode,
    /// Arrivals at the barrier counter queued so far (`dm::op::BARRIER`): the
    /// latest barrier's target. A barrier adds the units its op ran on, not
    /// every unit, so an op on a few of many tiles enqueues on those few
    /// (`Session::catch_up` orders the others before their next list).
    barriers: u32,
    /// Placements freed while lists that may read them are queued: given back
    /// at the next sync.
    pending_frees: Vec<tensor::Placement>,
    /// The trace being captured ([`Session::begin_trace`]).
    pub(crate) capture: Option<trace::Capture>,
    /// Finished traces, by number.
    pub(crate) traces: std::collections::HashMap<u64, trace::Trace>,
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
    /// The fixed NC writer, started for a compute region and restarted with B.
    nc: Option<DataMover<Noc0>>,
    /// The barrier target this tile last waited for, by arriving or by a
    /// catch-up ([`Session::catch_up`]): behind [`Session::barriers`] while it
    /// sits ops out.
    barrier_seen: u32,
}

/// A list on a mover's queue: its number, whether it reserved kernels (to
/// close when it is retired), and what it was, for an error.
struct QueuedList {
    roles: bool,
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

/// Where a tensor takes the tile layout ([`Session::set_tilize`]).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Tilize {
    /// On the host, before the card's DMA: the host's cores convert, and
    /// tiles cross PCIe (`tensor::host_dma_jobs`).
    Host,
    /// On the card: rows cross PCIe and each unit's mover tilizes them in L1
    /// (`tt_isa::dm::op::TILIZE`, `tensor::row_major_dma_jobs`). The tile
    /// layout never leaves the card.
    Card,
}

/// `rows` rows of `row` bytes from `src` (rows `src.len() / rows` apart) to
/// `dst` (rows `dst_stride` apart) -- or the other way, as strides say: the
/// host's whole share of a row-major transfer, split over threads when large.
fn copy_rows(src: &[u8], rows: usize, src_stride: usize, dst: &mut [u8], dst_stride: usize) {
    let row = src_stride.min(dst_stride);
    if rows == 0 {
        return;
    }
    let one = |src: &[u8], dst: &mut [u8], n: usize| {
        if src_stride == dst_stride {
            dst[..n * row].copy_from_slice(&src[..n * row]);
        } else {
            for r in 0..n {
                dst[r * dst_stride..][..row].copy_from_slice(&src[r * src_stride..][..row]);
            }
        }
    };
    let bytes = rows * row;
    let threads = if bytes >= 4 << 20 { 8 } else { 1 };
    if threads == 1 {
        return one(src, dst, rows);
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (k, d) in dst[..rows * dst_stride]
            .chunks_mut(per * dst_stride)
            .enumerate()
        {
            let n = per.min(rows - k * per);
            let sr = &src[k * per * src_stride..];
            scope.spawn(move || one(sr, d, n));
        }
    });
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

    /// Its datums' bytes, little-endian as the card takes them.
    fn bytes(&self) -> &[u8] {
        // SAFETY: plain 4-byte values; x86's and RISC-V's byte order alike.
        unsafe {
            match *self {
                Src::F32(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4),
                Src::Bits(v) => std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4),
            }
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
    /// Ops queued through the ownership scheduler, with or without batching.
    pub ops: u64,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum HostStage {
    /// The whole of queueing an op: everything below, and what is between.
    #[default]
    Enqueue,
    /// Steps into ownership regions and control lists.
    Segments,
    /// A unit's roles and mover checked, started if they are not.
    Ensure,
    /// A kernel's descriptors checked against what queued lists read.
    IdleCheck,
    /// Programs looked up in the cache, uploaded if missing.
    Place,
    /// Kernel generations reserved (`Resident::reserve`).
    Reserve,
    /// The list decoded as the mover will, before any of it is written.
    Check,
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
        HostStage::Check,
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
    nc_image: &'static [u8],
    w4: tt_device::Window,
    /// The zero rows a row gather's padding reads ([`Session::zero_row`]), one per
    /// width and datum kind, kept for as long as GDDR is.
    zero_rows: Vec<(usize, tensor::Elem, DramTensor)>,
}

/// One mover list for one unit: list entries, the `KERNEL` entries among
/// them (whose generation is filled in when the list is started), and the
/// matmul programs those kernels run.
#[derive(Default)]
struct Segment {
    streaming: bool,
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
}

/// Make every program `seg`'s kernels name resident on the tile, uploading
/// what is missing, and return each kernel's `(address, words)` per role
/// (`(0, 0)` for an empty one). Everything placed stays pinned until the list
/// has finished. If fragmentation leaves no room beside what is pinned, the
/// cache starts again from empty: ownership grouping keeps programs within
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
            if let Some(j) =
                seg.kernel_roles[..k]
                    .iter()
                    .enumerate()
                    .position(|(index, roles_before)| {
                        Arc::ptr_eq(roles_before, roles)
                            && seg.kernel_loops[index] == seg.kernel_loops[k]
                    })
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
        if !full && seg.streaming {
            let capacity = seg.entries[0][3];
            let mut scripts = [(0u32, 0u32); 3];
            for (role, script) in scripts.iter_mut().enumerate() {
                let mut words = vec![
                    tt_isa::dataflow::VERSION,
                    placed.len() as u32,
                    capacity,
                    role as u32,
                ];
                for programs in &placed {
                    words.extend([programs[role].0, programs[role].1, 0, 0]);
                }
                let length = words.len() as u32 | tt_isa::dataflow::STREAMED;
                let words: Arc<[u32]> = words.into();
                let hash = crate::program_cache::hash(&words);
                let address = match cache.place_hashed(&words, hash) {
                    Ok(Placed::Hit(address)) => address,
                    Ok(Placed::Upload(address)) => {
                        let bytes: Vec<u8> =
                            words.iter().flat_map(|word| word.to_le_bytes()).collect();
                        dev.l1_write(window, tile, address, &bytes)
                            .map_err(|error| PlaceError::Failed(error.into()))?;
                        address
                    }
                    Err(CacheError::Full { .. }) => {
                        full = true;
                        break;
                    }
                    _ => {
                        return Err(PlaceError::Failed(TensorError::Shape(
                            "streaming program does not fit the resident cache".into(),
                        )))
                    }
                };
                *script = (address as u32, length);
            }
            if !full {
                return Ok(vec![scripts]);
            }
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

/// Standalone transfers and control kernels, ordered on B's command queue.
fn control_segments(steps: Vec<Step>) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut cur = Segment::default();
    let mut after_list = false;
    for step in steps {
        match step {
            Step::List { what, entries } => {
                if !entries.is_empty() {
                    add_list(&mut cur, &mut out, what, &entries, after_list);
                    after_list = true;
                }
            }
            Step::Kernel {
                roles,
                init,
                mop,
                loops,
                ..
            } => {
                add_kernel(
                    &mut cur,
                    &mut out,
                    roles,
                    init,
                    *mop,
                    loops,
                    tt_isa::dm::op::KERNEL,
                );
                after_list = false;
            }
            Step::Transfer { .. } => {
                unreachable!("`streaming_segments` gives transfers packets of their own")
            }
        }
    }
    close_segment(&mut cur, &mut out);
    out
}

/// Consecutive standalone transfers (`Step::Transfer`) as shared packets
/// (`dm::op::PAIR`): B's section reserves a credit, reads a batch in and
/// pushes it; NC's waits for it, writes it out and pops it. No kernel runs, so
/// the packet has no `LAUNCH` and no `KERNEL_WAIT`; B's slot is reported done
/// when NC's last write is acknowledged, which is what orders a later list
/// after these transfers' writes.
///
/// Transfers of one depth share a packet while it has room (a batch is never
/// split). At depth two a drain waits, between transfers, for every batch
/// NC has yet to write out, since the next transfer's slots and reads may meet
/// what an earlier one wrote; at depth one the next reserve waits for that.
fn transfer_segments(steps: Vec<Step>) -> Vec<Segment> {
    use tt_isa::dataflow::{Action, Release, Stream};
    use tt_isa::dm::{op, LIST_MAX};
    struct Packet {
        what: &'static str,
        depth: u16,
        reader: Vec<[u32; 8]>,
        writer: Vec<[u32; 8]>,
        sent: u32,
        steps: u64,
    }
    fn close(packet: Option<Packet>, out: &mut Vec<Segment>) {
        use tt_isa::dataflow::Action;
        let Some(mut p) = packet.filter(|p| p.sent > 0) else {
            return;
        };
        // The last batch's `Pop` frees nothing anyone waits for: the counters
        // restart with the next packet, and NC's list ends by waiting for its
        // writes (`run_entries`).
        if p.writer.last().is_some_and(|e| e[2] == Action::Pop.word()) {
            p.writer.pop();
        }
        let mut segment = Segment {
            streaming: true,
            what: p.what,
            steps: p.steps,
            ..Default::default()
        };
        segment.entries.push([
            op::PAIR,
            p.reader.len() as u32,
            p.writer.len() as u32,
            p.depth as u32,
            0,
            0,
            0,
            0,
        ]);
        segment.entries.extend(p.reader);
        segment.entries.extend(p.writer);
        out.push(segment);
    }
    let credit = |action: Action, depth: u16| {
        [
            op::CB,
            Stream::Transfer.word(),
            action.word(),
            depth as u32,
            0,
            0,
            0,
            0,
        ]
    };
    let mut out = Vec::new();
    let mut packet: Option<Packet> = None;
    for step in steps {
        let Step::Transfer {
            what,
            depth,
            batches,
        } = step
        else {
            unreachable!("only transfers reach here")
        };
        if packet.as_ref().is_some_and(|p| p.depth != depth) {
            close(packet.take(), &mut out);
        }
        let mut counted = false;
        for batch in batches {
            let (reads, writes) = (batch.read.len() + 2, batch.write.len() + 2);
            assert!(
                reads.max(writes) < LIST_MAX as usize / 2,
                "a transfer batch of {what} is too long for a packet"
            );
            // The header, both sides, a drain and a barrier trailer must fit.
            let fits = |p: &Packet| {
                2 + p.reader.len() + 1 + reads + p.writer.len() + writes <= LIST_MAX as usize
            };
            if packet.as_ref().is_some_and(|p| !fits(p)) {
                close(packet.take(), &mut out);
                counted = false;
            }
            let p = packet.get_or_insert_with(|| Packet {
                what,
                depth,
                reader: Vec::new(),
                writer: Vec::new(),
                sent: 0,
                steps: 0,
            });
            if !counted {
                p.steps += 1;
                counted = true;
                if depth > 1 && p.sent > 0 {
                    p.reader.push([
                        op::RELEASED,
                        p.sent,
                        Release::Transferred.word(),
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]);
                }
            }
            // The counters start at zero with the packet, so the first `depth`
            // batches' reserves cannot wait: left out, each is 32 bytes less
            // through uncached PCIe writes on every tile.
            if p.sent >= depth as u32 {
                p.reader.push(credit(Action::Reserve, depth));
            }
            p.reader.extend(batch.read);
            p.reader.push(credit(Action::Push, depth));
            p.writer.push(credit(Action::Wait, depth));
            p.writer.extend(batch.write);
            p.writer.push(credit(Action::Pop, depth));
            p.sent += 1;
        }
    }
    close(packet, &mut out);
    out
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

fn capture_commands(
    entries: &[[u32; 8]],
    base: u32,
    mut retain: impl FnMut(u64),
) -> Result<Vec<[u32; 8]>, TensorError> {
    use tt_isa::dm::{op, record};
    let mut relative = Vec::with_capacity(entries.len());
    let mut index = 0;
    while index < entries.len() {
        let mut entry = entries[index];
        let count = record::len(entry[0]);
        if index + count > entries.len() {
            return Err(TensorError::Shape(
                "truncated captured command record".into(),
            ));
        }
        if matches!(entry[0], op::KERNEL | op::LAUNCH | op::KERNEL_WAIT) {
            entry[1] = entry[1].wrapping_sub(base);
        }
        if matches!(entry[0], op::KERNEL | op::LAUNCH) {
            for role in 0..3 {
                let address = entry[2 + 2 * role] as u64;
                if address != 0 {
                    retain(address);
                }
            }
        }
        relative.push(entry);
        relative.extend_from_slice(&entries[index + 1..index + count]);
        index += count;
    }
    Ok(relative)
}

fn close_segment(cur: &mut Segment, out: &mut Vec<Segment>) {
    if !cur.entries.is_empty() {
        out.push(std::mem::take(cur));
    }
}

/// The L1 bytes a list touches, sorted and merged: what its reads write
/// (`gather`), or what its writes read -- from record headers, without
/// expanding them, and over-approximated where that is simpler (safe: a wider
/// footprint only waits more). `None` when an entry's footprint is not
/// modelled -- anything but GDDR moves, fills and the move records -- which
/// keeps a region's slot reuse waiting for NC's release.
fn l1_footprint(entries: &[[u32; 8]], gather: bool) -> Option<Vec<std::ops::Range<u64>>> {
    use tt_isa::dm::{op, record, Entry, Transform, TILE_SLOT};
    let slots = |at: u32, n: u64| at as u64..at as u64 + n * TILE_SLOT;
    let mut ranges = Vec::new();
    let mut index = 0;
    while index < entries.len() {
        let head = entries[index];
        let count = record::len(head[0]);
        if count > 1 {
            entries.get(index..index + count)?;
            match head[0] {
                record::READ_RUN if gather => ranges.push(slots(head[3], head[2] as u64)),
                record::WRITE_RUN if !gather => ranges.push(slots(head[3], head[2] as u64)),
                record::GATHER if gather => {
                    let kt = record::gather_kt(head[1]) as u64;
                    // A resident `A` is not written: its slots hold what an earlier gather read.
                    if head[1] & record::GATHER_A_RESIDENT == 0 {
                        ranges.push(slots(head[2], head[5] as u64 * kt));
                    }
                    ranges.push(slots(head[3], kt * head[7] as u64));
                }
                record::SCATTER if !gather => {
                    let outputs = head[4] as u64 * head[6] as u64;
                    let stride = (head[2] as u64).max(TILE_SLOT);
                    ranges.push(head[1] as u64..head[1] as u64 + outputs * stride);
                }
                _ => return None,
            }
        } else {
            match Entry::decode(u32::MAX, head).ok()? {
                Entry::Move {
                    descriptor,
                    transform,
                } if (descriptor.op == op::WRITE) != gather => {
                    let mut len = descriptor.range.len();
                    if transform != Transform::None {
                        len = len.max(TILE_SLOT);
                    }
                    ranges.push(descriptor.l1 as u64..descriptor.l1 as u64 + len);
                }
                Entry::Fill { dst, .. } if gather => ranges.push(slots(dst, 1)),
                _ => return None,
            }
        }
        index += count;
    }
    ranges.sort_by_key(|range| range.start);
    let mut merged: Vec<std::ops::Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    Some(merged)
}

/// Whether two sorted, merged footprints share a byte (unknown ones do).
fn footprints_meet(
    a: &Option<Vec<std::ops::Range<u64>>>,
    b: &Option<Vec<std::ops::Range<u64>>>,
) -> bool {
    let (Some(a), Some(b)) = (a, b) else {
        return true;
    };
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].start < b[j].end && b[j].start < a[i].end {
            return true;
        }
        if a[i].end <= b[j].end {
            i += 1;
        } else {
            j += 1;
        }
    }
    false
}

/// [`crate::program_cache::admitted`], for a cache of `cache` bytes.
fn admitted(words: usize, cache: u64) -> bool {
    words > 0 && words as u64 * 4 <= cache / 2
}

/// Put `entry`, a B entry, ahead of everything else `seg` runs: first in a packet's
/// reader section (NC's writes wait on credits only the reader and the roles release, so
/// they follow it too), or first in a plain list. `false`, and `seg` unchanged, if the
/// list is already as long as a list may be.
/// A captured stream's top-level `BARRIER` entries: `[arrivals, wait-only, the barriers both
/// stand for]` (`dm::op::BARRIER` words 4 and 5).
fn barrier_entries(stream: &[[u32; 8]]) -> [u32; 3] {
    let mut out = [0u32; 3];
    let mut i = 0;
    while i < stream.len() {
        let e = stream[i];
        if e[0] == tt_isa::dm::op::BARRIER {
            let wait = e[4] == 1;
            out[wait as usize] += 1;
            out[2] += if wait {
                e[tt_isa::dm::BARRIER_STANDS_FOR]
            } else {
                1
            };
        }
        i += tt_isa::dm::record::len(e[0]).max(1);
    }
    out
}

fn lead_with(seg: &mut Segment, entry: [u32; 8]) -> bool {
    if seg.entries.len() >= tt_isa::dm::LIST_MAX as usize {
        return false;
    }
    let packet = seg
        .entries
        .first()
        .is_some_and(|e| e[0] == tt_isa::dm::op::PAIR);
    let at = usize::from(packet);
    seg.entries.insert(at, entry);
    if packet {
        seg.entries[0][1] += 1;
    }
    for k in &mut seg.kernels {
        if *k >= at {
            *k += 1;
        }
    }
    for (w, _) in &mut seg.waits {
        if *w >= at {
            *w += 1;
        }
    }
    true
}

/// A unit's steps as ownership regions and control lists, for a program cache
/// of `cache` bytes (`Session::limit_program_cache`).
fn streaming_segments(steps: Vec<Step>, cache: u64) -> Result<Vec<Segment>, TensorError> {
    use tt_isa::dataflow::Action;
    use tt_isa::dm::op;
    let mut result = Vec::new();
    // Consecutive standalone transfers share one control list (and one queue
    // commit), as `control_segments` lays them out; consecutive `Transfer`s
    // share a packet (`transfer_segments`).
    let mut control = Vec::new();
    let mut transfers = Vec::new();
    let mut last_what = "";
    for item in execution_groups(steps, cache) {
        let blocks = match item {
            Item::Group(blocks) => blocks,
            Item::Step(Step::Kernel { roles, .. }) => {
                let reason = match roles
                    .iter()
                    .map(Vec::len)
                    .find(|&words| words > 0 && !admitted(words, cache))
                {
                    Some(words) => format!(
                        "a role program of {words} words exceeds half the {cache}-byte \
                         resident program cache"
                    ),
                    None => "it is not between a gather list and a scatter list".into(),
                };
                return Err(TensorError::Shape(format!(
                    "GDDR compute after `{last_what}` cannot form a resident \
                     gather/compute/scatter ownership region: {reason}"
                )));
            }
            Item::Step(step @ Step::Transfer { .. }) => {
                if !control.is_empty() {
                    result.extend(control_segments(std::mem::take(&mut control)));
                }
                if let Step::Transfer { what, .. } = &step {
                    last_what = what;
                }
                transfers.push(step);
                continue;
            }
            Item::Step(step) => {
                if !transfers.is_empty() {
                    result.extend(transfer_segments(std::mem::take(&mut transfers)));
                }
                if let Step::List { what, .. } = &step {
                    last_what = what;
                }
                control.push(step);
                continue;
            }
        };
        if !control.is_empty() {
            result.extend(control_segments(std::mem::take(&mut control)));
        }
        if !transfers.is_empty() {
            result.extend(transfer_segments(std::mem::take(&mut transfers)));
        }
        let capacity = if blocks[0].half.is_some() { 2 } else { 1 };
        let mut segment = Segment {
            streaming: true,
            resident: true,
            ..Default::default()
        };
        let mut reader = vec![[op::LAUNCH, 0, 0, 0, 0, 0, 0, 0]];
        let mut writer = Vec::new();
        let depth = capacity as usize;
        // Only a gather at least `depth` batches in reuses a slot, against a
        // scatter at least `depth` batches before the end: a region of
        // `depth` batches or fewer (one block per unit, often) needs none.
        let batches = blocks.len();
        let footprints: Vec<_> = blocks
            .iter()
            .enumerate()
            .map(|(index, block)| {
                (
                    (index >= depth)
                        .then(|| l1_footprint(&block.gather.1, true))
                        .flatten(),
                    (index + depth < batches)
                        .then(|| l1_footprint(&block.scatter.1, false))
                        .flatten(),
                )
            })
            .collect();
        for (index, block) in blocks.into_iter().enumerate() {
            if segment.what.is_empty() {
                segment.what = block.gather.0;
            }
            last_what = block.scatter.0;
            segment.init = block.init;
            segment.mop = block.mop;
            segment.loops = block.loops.clone();
            segment.kernel_roles.push(block.roles);
            segment.kernel_loops.push(block.loops);
            // Before this batch's gather reuses its slot, the batch that last
            // used the slot must be packed: whatever L1 T2 wrote, it has
            // stopped. The staging layout keeps a gather off the outputs of
            // the batches since (the other half), but a batch up to that one
            // -- NC may lag `2 * depth - 1` behind, by the output credits --
            // must also be released if this gather lands on what its scatter
            // reads, as an unknown footprint is taken to. Disjoint, the gather
            // overlaps NC's writes.
            let reuse = (index + 1).saturating_sub(depth);
            let released = ((index + 1).saturating_sub(2 * depth)..reuse)
                .rev()
                .find(|&k| footprints_meet(&footprints[index].0, &footprints[k].1))
                .map_or(0, |k| k + 1);
            if released > 0 {
                reader.push([op::RELEASED, released as u32, 0, 0, 0, 0, 0, 0]);
            }
            if reuse > released {
                reader.push([op::RELEASED, reuse as u32, 1, 0, 0, 0, 0, 0]);
            }
            reader.push([op::CB, 0, Action::Reserve.word(), capacity, 0, 0, 0, 0]);
            reader.extend(block.gather.1);
            reader.push([op::CB, 0, Action::Push.word(), capacity, 0, 0, 0, 0]);
            writer.push([op::CB, 1, Action::Wait.word(), capacity, 0, 0, 0, 0]);
            writer.extend(block.scatter.1);
            writer.push([op::CB, 1, Action::Pop.word(), capacity, 0, 0, 0, 0]);
            segment.steps += 3;
        }
        segment.roles = segment.kernel_roles.first().cloned();
        segment.kernels.push(1);
        segment.waits.push((reader.len() + 1, 0));
        reader.push([op::KERNEL_WAIT, 0, 0, 0, 0, 0, 0, 0]);
        segment.entries.push([
            op::PAIR,
            reader.len() as u32,
            writer.len() as u32,
            capacity,
            0,
            0,
            0,
            0,
        ]);
        segment.entries.extend(reader);
        segment.entries.extend(writer);
        result.push(segment);
    }
    if !control.is_empty() {
        result.extend(control_segments(control));
    }
    if !transfers.is_empty() {
        result.extend(transfer_segments(transfers));
    }
    Ok(result)
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
    half: Option<u8>,
    gather: (&'static str, Vec<[u32; 8]>),
    roles: Arc<[Vec<Instruction>; 3]>,
    init: Vec<runtime::SemaphoreInit>,
    mop: [Option<tt_isa::frontend::mop::MopConfig>; 3],
    loops: Arc<[Vec<crate::code::Loop>; 3]>,
    scatter: (&'static str, Vec<[u32; 8]>),
}

/// A unit's steps, as [`control_segments`] lays them out.
enum Item {
    Step(Step),
    /// Consecutive double-buffered blocks to overlap.
    Group(Vec<Block>),
}

/// Blocks per group at most: a group is one list, and a block is a gather
/// record, a launch, a wait and a scatter record.
const GROUP_MAX: usize = 40;

/// Compatible gather/compute/scatter batches, including single-buffered ones.
fn execution_groups(steps: Vec<Step>, cache: u64) -> Vec<Item> {
    use tt_isa::dm::LIST_MAX;
    let mut items: Vec<Item> = Vec::new();
    let mut run: Vec<Block> = Vec::new();
    let mut run_entries = 0usize;
    let flush = |run: &mut Vec<Block>, items: &mut Vec<Item>| {
        if !run.is_empty() {
            items.push(Item::Group(std::mem::take(run)));
        }
    };
    let mut steps = steps.into_iter().peekable();
    let mut last_half: Option<u8> = None;
    let mut resident_bytes = 0u64;
    let mut seen: Vec<Arc<[Vec<Instruction>; 3]>> = Vec::new();
    while let Some(step) = steps.next() {
        // A block: this list, then a kernel staged in a half, then a list.
        let is_block = matches!(step, Step::List { .. })
            && matches!(steps.peek(), Some(Step::Kernel { roles, .. })
                if roles.iter().all(|p| p.is_empty() || admitted(p.len(), cache)));
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
        // Its credits (reserve, push, wait, pop) and up to two slot-reuse
        // waits (`streaming_segments`).
        let block_entries = entries.len() + scatter.1.len() + 6;
        let bytes: u64 = roles.iter().map(|p| p.len() as u64 * 4).sum();
        let new = !seen.iter().any(|r| Arc::ptr_eq(r, &roles));
        let joins = !run.is_empty()
            && (half != last_half && half.is_some() && last_half.is_some() || half.is_none() && last_half.is_none())
            && run.len() < GROUP_MAX
            && run_entries + block_entries + 3 < LIST_MAX as usize
            && (!new || resident_bytes + bytes + 3 * (4 + 4 * (run.len() + 1)) as u64 * 4 <= cache)
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
            half,
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

/// The events unit stream `uc`'s entries `range` could stamp in the profile mode, at most, in
/// [`crate::profile_mode::Detail::Ops`] and in `Full`: three a stamped entry (its begin and end and a
/// `KICK` or `ROLES_DONE` inside), and for a `PAIR_CALL`, its reader section's entries (B runs them;
/// NC, which runs the writer's, does not stamp).
fn op_stamps(uc: &trace::UnitCapture, range: core::ops::Range<usize>) -> [u32; 2] {
    use crate::profile_mode::{Detail, ProfileOptions};
    let options = |detail| ProfileOptions {
        detail,
        ..ProfileOptions::default()
    };
    let (ops, full) = (options(Detail::Ops), options(Detail::Full));
    let mut count = [0u32; 2];
    let mut add = |entries: &[[u32; 8]]| {
        let mut i = 0;
        while i < entries.len() {
            let head = entries[i][0];
            count[0] += ops.stamps(head);
            count[1] += full.stamps(head);
            i += (tt_isa::dm::record::len(head) as usize).max(1);
        }
    };
    let mut i = range.start;
    while i < range.end.min(uc.stream.len()) {
        let head = uc.stream[i][0];
        if head == tt_isa::dm::op::PAIR_CALL {
            if let Some(k) = uc.pair_calls.iter().position(|&at| at == i) {
                if let Some(reader) = uc.pending_sections.get(2 * k) {
                    add(reader);
                }
            }
        } else {
            add(&uc.stream[i..i + 1]);
        }
        i += (tt_isa::dm::record::len(head) as usize).max(1);
    }
    count
}

/// Add what unit stream `uc`'s entries `range` move through GDDR to `bytes`: for a `PAIR_CALL`, both
/// its sections (B runs the reader's, NC the writer's).
fn op_bytes(
    uc: &trace::UnitCapture,
    range: core::ops::Range<usize>,
    bytes: &mut crate::profile_mode::OpBytes,
) {
    let mut i = range.start;
    while i < range.end.min(uc.stream.len()) {
        let head = uc.stream[i][0];
        let n = (tt_isa::dm::record::len(head) as usize).max(1);
        if head == tt_isa::dm::op::PAIR_CALL {
            if let Some(k) = uc.pair_calls.iter().position(|&at| at == i) {
                for section in uc.pending_sections.iter().skip(2 * k).take(2) {
                    bytes.count(section);
                }
            }
        } else {
            bytes.count(&uc.stream[i..(i + n).min(uc.stream.len())]);
        }
        i += n;
    }
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
        let math = crate::sfpu::approx::MathMode::from_env().map_err(SessionError::Math)?;
        let mut session = Session {
            dataflow: DataflowStats::default(),
            dev,
            units: Vec::new(),
            grid,
            images,
            profile: runtime::Profile::default(),
            pipeline: true,
            spread: Spread::All,
            matmul_k_block_limit: None,
            thin_matmul_reads: true,
            packed_thin_a: true,
            packed_rms: true,
            streamed_k: true,
            weight_prefetch: true,
            prefetch_anchor: None,
            resident_matmul_a: true,
            unpack_transposed_b: true,
            thin_stream_units: true,
            host: HostTimes::new(),
            staging: None,
            host_dma: true,
            unbarriered: false,
            op_label: "",
            staging_at: 0,
            staging_busy: false,
            tilize: Tilize::Host,
            pipelined: 0,
            profile_roles: true,
            drains: 0,
            dram: None,
            profiling: None,
            profiler: None,
            count_trace_bytes: false,
            profile_drained: Vec::new(),
            profile_buffer: (
                tt_isa::mailbox::TRACE_BUFFER,
                tt_isa::mailbox::TRACE_BUFFER_BYTES,
            ),
            batching: std::env::var("TT_BATCH").map_or(true, |v| v != "0"),
            math,
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
                barrier_seen: 0,
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
            // The buffer the profile reads (`set_profile_buffer` may have moved it).
            let (buffer, bytes) = self.profile_buffer;
            dev.configure_trace(r.window(), unit.tile, buffer, bytes)?;
        }
        unit.resident = Some(r);
        Ok(())
    }

    /// Record what every unit's data mover and role runners do from now on,
    /// through each tile's debug timestamper (`crate::profile`), until
    /// [`Session::profile_stop`]. Each unit's stream is drained after every
    /// operation, so a profile may span any number of ops.
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
        // Stamping is compiled into the firmware only with its `profile` feature; without it there
        // is nothing to record. Checked on unit 0's mover, once it has started.
        if self.units.first().is_some_and(|u| u.resident.is_none()) {
            self.prepare_unit(0)?;
        }
        if self.dram.is_some() && self.units[0].mover.is_none() {
            // Movers start with their first list; this one is needed now, to ask it.
            self.ensure_unit(0).map_err(|e| match e {
                TensorError::Run(r) => r,
                TensorError::Transport(t) => RunError::Transport(t),
                _ => RunError::Transport(TransportError::Hazard {
                    address: tt_isa::dm::CAPS,
                    reason: "unit 0's data mover did not start, so whether it can stamp is unknown",
                }),
            })?;
        }
        if self.units[0].mover.as_ref().is_some_and(|m| !m.stamps()) {
            return Err(RunError::Transport(TransportError::Hazard {
                address: tt_isa::dm::CAPS,
                reason: "this firmware cannot stamp its work: build with the `profile` feature \
                         (`burn-tt/profile`, `tt-tests/profile`; `cargo xtask silicon --profile`)",
            }));
        }
        self.profiling = Some(Vec::new());
        let (buffer, bytes) = self.profile_buffer;
        let trace_word = self.trace_word();
        let mut units = Vec::with_capacity(self.units.len());
        for u in 0..self.units.len() {
            if self.units[u].resident.is_none() {
                self.prepare_unit(u)?;
            }
            let profile_roles = self.profile_roles;
            let Session { dev, units: us, .. } = self;
            let unit = &mut us[u];
            let r = unit.resident.as_mut().expect("prepared above");
            dev.configure_trace(r.window(), unit.tile, buffer, bytes)?;
            r.set_profiling(true);
            r.set_profile_roles(profile_roles);
            if unit.mover.is_some() {
                dev.write32(r.window(), unit.tile, tt_isa::dm::TRACE, trace_word)?;
                dev.write32(r.window(), unit.tile, tt_isa::dm::TRACE_WINDOW, 0)?;
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
        let (buffer, _) = self.profile_buffer;
        for (u, p) in units.iter_mut().enumerate() {
            let Session { dev, units: us, .. } = self;
            let unit = &mut us[u];
            let Some(r) = unit.resident.as_mut() else {
                continue;
            };
            p.events
                .extend(dev.read_trace(r.window(), unit.tile, buffer)?);
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

    /// The mover's `tt_isa::dm::TRACE` word while profiling: the profile mode's detail, or every
    /// entry for a bare [`Session::profile_start`].
    fn trace_word(&self) -> u32 {
        self.profiler
            .as_ref()
            .map_or(tt_isa::dm::trace_mode::FULL, |p| p.options.trace_mode())
    }

    /// Turn the profile mode on (`Some`) or off (`None`, returning what it recorded): where the
    /// device's time goes, op by op, for whatever runs, eager or traced (`crate::profile_mode`).
    /// While it is on, every op is synced and timed alone, and every replay stamps the next window
    /// of its ops. Refused on the simulator, which does not model the stamp stream.
    pub fn profile_mode(
        &mut self,
        options: Option<crate::profile_mode::ProfileOptions>,
    ) -> Result<Option<crate::profile_mode::ProfileReport>, RunError> {
        match options {
            Some(options) => {
                if self.profiler.is_some() || self.profiling.is_some() {
                    return Err(RunError::Transport(TransportError::Hazard {
                        address: 0,
                        reason: "the profile mode is already on, or a bare profile is running",
                    }));
                }
                self.profile_roles = options.roles;
                self.profiler = Some(crate::profile_mode::Profiler::new(options, 0));
                if let Err(e) = self.profile_start() {
                    self.profiler = None;
                    return Err(e);
                }
                let counter = self.unit0_clock()?;
                self.profiler = Some(crate::profile_mode::Profiler::new(options, counter));
                Ok(None)
            }
            None => {
                let report = self.profile_report()?;
                self.profiler = None;
                self.profile_drained.clear();
                if self.profiling.is_some() {
                    self.profile_stop()?;
                }
                Ok(report)
            }
        }
    }

    /// What the profile mode has recorded so far, if it is on.
    pub fn profile_report(
        &mut self,
    ) -> Result<Option<crate::profile_mode::ProfileReport>, RunError> {
        if self.profiler.is_none() {
            return Ok(None);
        }
        self.sync_run()?;
        self.profile_collect();
        let counter = self.unit0_clock()?;
        Ok(self.profiler.as_ref().map(|p| p.report(counter)))
    }

    fn unit0_clock(&mut self) -> Result<u64, RunError> {
        let Session { dev, units, .. } = self;
        match units
            .first()
            .and_then(|u| u.resident.as_ref().map(|r| (r, u.tile)))
        {
            Some((r, tile)) => Ok(dev.wall_clock(r.window(), tile)?),
            None => Ok(0),
        }
    }

    /// Hand the stamps drained since the last call to the profile mode, for what it was waiting
    /// to attribute (an eager op, or a replay).
    fn profile_collect(&mut self) {
        let Some(p) = self.profiler.as_mut() else {
            return;
        };
        let drained = std::mem::take(&mut self.profile_drained);
        let was_replay = p.expecting_replay();
        p.drained(&drained);
        if was_replay {
            // The next list is no replay: it stamps every entry again.
            let Session { dev, units, .. } = self;
            for unit in units.iter() {
                if let (Some(r), Some(_)) = (unit.resident.as_ref(), unit.mover.as_ref()) {
                    let _ = dev.write32(r.window(), unit.tile, tt_isa::dm::TRACE_WINDOW, 0);
                }
            }
        }
    }

    /// Move unit `u`'s events into the profile and start its stream empty
    /// again, so the 1024-event buffer bounds one operation, not a whole profile.
    fn drain(&mut self, u: usize) -> Result<(), TransportError> {
        let Session {
            dev,
            units,
            profiling,
            profiler,
            profile_drained,
            profile_buffer: (buffer, bytes),
            ..
        } = self;
        let (Some(p), Some(r)) = (profiling.as_mut(), units[u].resident.as_ref()) else {
            return Ok(());
        };
        let tile = units[u].tile;
        let events = match (dev.read_trace(r.window(), tile, *buffer), profiler.as_mut()) {
            // The profile mode never fails a run: an op whose stamps outran the buffer is reported as
            // such, and the buffer starts again.
            (
                Err(TransportError::Hazard {
                    reason: tt_device::trace::OVERFLOWED,
                    ..
                }),
                Some(pr),
            ) => {
                pr.overflowed();
                Vec::new()
            }
            (events, _) => events?,
        };
        if profiler.is_some() {
            // The profile mode attributes them ([`Session::profile_collect`]) and keeps no raw stream.
            if profile_drained.len() <= u {
                profile_drained.resize(u + 1, Vec::new());
            }
            profile_drained[u].extend(events);
        } else {
            p[u].events.extend(events);
        }
        dev.configure_trace(r.window(), tile, *buffer, *bytes)
    }

    /// Enable GDDR with fixed resident B-reader / NC-writer compute ownership.
    /// Synchronizes queued work and rejects changes during capture. Both images must support
    /// the streaming protocol; NC is started lazily on participating tiles.
    pub fn enable_dram(
        &mut self,
        b_image: &'static [u8],
        nc_image: &'static [u8],
    ) -> Result<(), TensorError> {
        if self.capture.is_some() {
            return Err(TraceError::Capturing.into());
        }
        self.sync()?;
        let mut requirements = crate::l1::Requirements::new(1);
        let input = requirements.scratch("input page counters", 8, 4, 0..1);
        let output = requirements.scratch("output page counters", 8, 4, 0..1);
        let state = requirements.scratch(
            "stream cancellation, progress and waits",
            tt_isa::dataflow::END - tt_isa::dataflow::ABORT,
            4,
            0..1,
        );
        let arena = tt_isa::l1::Region {
            name: "stream control",
            base: tt_isa::dataflow::INPUT,
            end: tt_isa::dataflow::END,
        };
        let plan = requirements
            .plan(arena)
            .map_err(|error| TensorError::Shape(error.to_string()))?;
        crate::l1::check(&requirements, &plan, arena).map_err(TensorError::Shape)?;
        if plan.addr(input) != tt_isa::dataflow::INPUT
            || plan.addr(output) != tt_isa::dataflow::OUTPUT
            || plan.addr(state) != tt_isa::dataflow::ABORT
        {
            return Err(TensorError::Shape(
                "stream-control plan disagrees with firmware".into(),
            ));
        }
        if let Some(dram) = &self.dram {
            if dram.image != b_image || dram.nc_image != nc_image {
                return Err(TensorError::Shape(
                    "GDDR firmware is already configured".into(),
                ));
            }
            return Ok(());
        }
        let w4 = self.dev.alloc_window(WindowKind::FourGib)?;
        let window = self.dev.alloc_window(WindowKind::TwoMib)?;
        let dram = self.dev.dram_grid(&window)?;
        self.dram = Some(DramState {
            alloc: DramAlloc::new(&dram),
            dram,
            image: b_image,
            nc_image,
            w4,
            zero_rows: Vec::new(),
        });
        Ok(())
    }

    pub fn dataflow_stats(&self) -> &DataflowStats {
        &self.dataflow
    }

    pub fn dataflow_progress(&mut self) -> Result<Vec<Option<DataflowProgress>>, TensorError> {
        let mut result = Vec::with_capacity(self.units.len());
        for unit in &self.units {
            let Some(resident) = unit.resident.as_ref().filter(|_| unit.nc.is_some()) else {
                result.push(None);
                continue;
            };
            let mut words = [0u32; 13];
            for (index, word) in words.iter_mut().enumerate() {
                *word = self.dev.read32(
                    resident.window(),
                    unit.tile,
                    tt_isa::dataflow::INPUT + index as u64 * 4,
                )?;
            }
            result.push(Some(DataflowProgress {
                input: [words[0] as u16, words[1] as u16],
                output: [words[2] as u16, words[3] as u16],
                abort: words[4],
                roles: [words[5], words[6], words[7]],
                waits: [words[8], words[9], words[10], words[11], words[12]],
            }));
        }
        Ok(result)
    }

    fn dram_state(&mut self) -> Result<&mut DramState, TensorError> {
        self.dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled on this session".into()))
    }

    /// Whether the transport is the simulator, which picks the PRNG
    /// model's lane initialisation (`prng::Target`).
    pub fn is_simulated(&mut self) -> bool {
        self.dev.transport().is_simulated()
    }

    /// The GDDR allocator, for ops built outside this file (`sfpu::sort`).
    pub(crate) fn dram_alloc(&mut self) -> Result<&mut DramAlloc, TensorError> {
        Ok(&mut self.dram_state()?.alloc)
    }

    /// Allocate device storage without uploading tensor data.
    pub fn empty(
        &mut self,
        rows: usize,
        cols: usize,
        elem: tensor::Elem,
    ) -> Result<DramTensor, TensorError> {
        DramTensor::alloc_elem(&mut self.dram_state()?.alloc, rows, cols, elem)
    }

    /// Convert resident F32 data to physically packed BF16 using ties-even.
    /// NaNs are quieted; BF16 subnormals become signed zero.
    pub fn bf16_from_f32(
        &mut self,
        input: &DramTensor,
    ) -> Result<crate::bf16::Bf16Tensor, TensorError> {
        let (out, jobs) = crate::bf16::compress(&mut self.dram_state()?.alloc, input)?;
        if let Err(error) = self.submit_jobs(jobs, RESET_BUDGET) {
            self.dram_state()?.alloc.free(&out.placement);
            return Err(error);
        }
        Ok(out)
    }

    /// Widen physically packed BF16 through SrcA and MOVA2D on Tensix.
    pub fn bf16_to_f32(
        &mut self,
        input: &crate::bf16::Bf16Tensor,
    ) -> Result<DramTensor, TensorError> {
        let work = crate::bf16::expand(&mut self.dram_state()?.alloc, input)?;
        self.execute(work, RESET_BUDGET)
    }

    pub fn free_bf16(&mut self, tensor: crate::bf16::Bf16Tensor) -> Result<(), TensorError> {
        self.free_placement(tensor.placement)
    }

    /// Cast logical F32 to deterministic resident BFP storage. Zero padding
    /// cannot influence an exponent group. A dirty view is repaired via a copy.
    pub fn bfp_from_f32(
        &mut self,
        input: &DramTensor,
        format: crate::bfp::BfpFormat,
    ) -> Result<crate::bfp::BfpTensor, TensorError> {
        input.expect("BFP conversion", tensor::Elem::F32)?;
        let repaired = self.meet(input, tensor::PadNeed::Zero)?;
        let result = (|| {
            let (out, jobs) = crate::bfp::compress(
                &mut self.dram_state()?.alloc,
                repaired.as_ref().unwrap_or(input),
                format,
            )?;
            if let Err(error) = self.submit_jobs(jobs, RESET_BUDGET) {
                self.dram_state()?.alloc.free(&out.placement);
                return Err(error);
            }
            Ok(out)
        })();
        if let Some(repaired) = repaired {
            let _ = self.free(repaired);
        }
        result
    }

    /// Decode BFP storage to resident F32 through SrcA/MOVA2D on Tensix.
    /// Decoded subnormals normalize to zero, matching the measured Src path.
    pub fn bfp_to_f32(&mut self, input: &crate::bfp::BfpTensor) -> Result<DramTensor, TensorError> {
        let work = crate::bfp::expand(&mut self.dram_state()?.alloc, input)?;
        self.execute(work, RESET_BUDGET)
    }

    pub fn free_bfp(&mut self, tensor: crate::bfp::BfpTensor) -> Result<(), TensorError> {
        self.free_placement(tensor.placement)
    }

    /// Ordinary readback returns decoded logical F32, never packed numerics.
    pub fn download_bfp(&mut self, input: &crate::bfp::BfpTensor) -> Result<Vec<f32>, TensorError> {
        self.refuse_while_capturing("download")?;
        let decoded = self.bfp_to_f32(input)?;
        let result = self.download(&decoded);
        let free = self.free(decoded);
        result.and_then(|values| free.map(|()| values))
    }

    /// Diagnostic physical TileImage bytes per tile. The unused header bytes
    /// have no semantic value. Slot alignment bytes are excluded.
    pub fn download_bfp_raw(
        &mut self,
        input: &crate::bfp::BfpTensor,
    ) -> Result<Vec<Vec<u8>>, TensorError> {
        self.refuse_while_capturing("download")?;
        self.sync()?;
        let Session { dev, dram, .. } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        (0..input.tile_count())
            .map(|tile| {
                let slot = input.slot(tile);
                let mut bytes = vec![0; input.format().tile_image().total_bytes()];
                let range = slot
                    .channel()
                    .range(slot.offset(), bytes.len() as u64)
                    .ok_or_else(|| TensorError::Shape("BFP diagnostic range overflow".into()))?;
                dev.dram_read(&d.w4, range, &mut bytes)?;
                Ok(bytes)
            })
            .collect()
    }

    /// Native F32 product of decoded BFP operands. Transposes, ragged edges
    /// and K blocking follow the same scheduler as ordinary resident products.
    pub fn bfp_matmul(
        &mut self,
        a: &crate::bfp::BfpTensor,
        a_transposed: bool,
        b: &crate::bfp::BfpTensor,
        b_transposed: bool,
        fidelity: Fidelity,
    ) -> Result<DramTensor, TensorError> {
        if !a_transposed && !b_transposed && a.format() == b.format() {
            let limit = self.matmul_k_block_limit.map(std::num::NonZeroUsize::get);
            let work = crate::bfp::matmul(&mut self.dram_state()?.alloc, a, b, fidelity, limit)?;
            return self.execute(work, RESET_BUDGET);
        }
        let aa = self.bfp_to_f32(a)?;
        let result = match self.bfp_to_f32(b) {
            Ok(bb) => {
                let result = self.matmul_dram(
                    &aa,
                    a_transposed,
                    &bb,
                    b_transposed,
                    SrcRoute::Tf32FromFp32,
                    fidelity,
                    RESET_BUDGET,
                );
                let _ = self.free(bb);
                result
            }
            Err(error) => Err(error),
        };
        let _ = self.free(aa);
        result
    }

    pub fn repack_bf16(
        &mut self,
        input: &crate::bf16::Bf16Tensor,
        sources: &[[usize; 2]],
        dims: [usize; 2],
    ) -> Result<crate::bf16::Bf16Tensor, TensorError> {
        let (out, jobs) = crate::bf16::repack(&mut self.dram_state()?.alloc, input, sources, dims)?;
        if let Err(error) = self.submit_jobs(jobs, RESET_BUDGET) {
            self.dram_state()?.alloc.free(&out.placement);
            return Err(error);
        }
        Ok(out)
    }

    pub(crate) fn repack_bf16_padded(
        &mut self,
        input: &crate::bf16::Bf16Tensor,
        sources: &[Option<[usize; 2]>],
        dims: [usize; 2],
        padding: u16,
    ) -> Result<crate::bf16::Bf16Tensor, TensorError> {
        let (out, jobs) = crate::bf16::repack_padded(
            &mut self.dram_state()?.alloc,
            input,
            sources,
            dims,
            padding,
        )?;
        if let Err(error) = self.submit_jobs(jobs, RESET_BUDGET) {
            self.dram_state()?.alloc.free(&out.placement);
            return Err(error);
        }
        Ok(out)
    }

    pub(crate) fn repack_bf16_padded_into(
        &mut self,
        input: &crate::bf16::Bf16Tensor,
        sources: &[Option<[usize; 2]>],
        output: &crate::bf16::Bf16Tensor,
        padding: u16,
    ) -> Result<(), TensorError> {
        let jobs = crate::bf16::repack_padded_into(input, sources, output, padding)?;
        self.submit_jobs(jobs, RESET_BUDGET)
    }

    /// General BF16 FPU windows, grouped into sixteen columns and continued
    /// over sixteen-datum blocks. Results remain F32 until the caller casts.
    pub fn bf16_pool_windows(
        &mut self,
        input: &crate::bf16::Bf16Tensor,
        windows: &[Vec<[usize; 2]>],
        divisors: &[usize],
        dims: [usize; 2],
    ) -> Result<DramTensor, TensorError> {
        crate::fpu::windows(self, input, windows, divisors, dims)
    }

    /// Upload BF16 storage bits without converting their values.
    pub fn upload_bf16(
        &mut self,
        values: &[u16],
        rows: usize,
        cols: usize,
    ) -> Result<crate::bf16::Bf16Tensor, TensorError> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(values.len()) {
            return Err(TensorError::Shape("invalid BF16 upload shape".into()));
        }
        self.refuse_while_capturing("upload BF16")?;
        let tiles = rows
            .div_ceil(32)
            .checked_mul(cols.div_ceil(32))
            .ok_or_else(|| TensorError::Shape("BF16 tile count overflow".into()))?;
        let output = crate::bf16::Bf16Tensor {
            rows,
            cols,
            placement: self
                .dram_state()?
                .alloc
                .alloc_slots(tiles, crate::bf16::TILE_SLOT)?,
        };
        let result = (|| {
            let d = self.dram.as_ref().expect("enabled above");
            for t in 0..tiles {
                let mut image = vec![0u8; crate::bf16::TILE_SLOT as usize];
                for r in 0..32 {
                    for c in 0..32 {
                        let row = t / cols.div_ceil(32) * 32 + r;
                        let col = t % cols.div_ceil(32) * 32 + c;
                        if row < rows && col < cols {
                            let offset =
                                tt_isa::dm::TILE_DATA as usize + tt_isa::dm::face_index(r, c) * 2;
                            image[offset..offset + 2]
                                .copy_from_slice(&values[row * cols + col].to_le_bytes());
                        }
                    }
                }
                self.dev.dram_write(&d.w4, output.slot(t), &image)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.dram_state()?.alloc.free(&output.placement);
            return Err(error);
        }
        Ok(output)
    }

    /// Explicit raw BF16 readback. Never used for native arithmetic.
    pub fn download_bf16(
        &mut self,
        tensor: &crate::bf16::Bf16Tensor,
    ) -> Result<Vec<u16>, TensorError> {
        self.refuse_while_capturing("download BF16")?;
        self.sync()?;
        let mut values = vec![0; tensor.rows * tensor.cols];
        let ct = tensor.cols.div_ceil(32);
        let d = self.dram.as_ref().expect("BF16 requires enabled GDDR");
        for t in 0..tensor.tile_count() {
            let mut image = vec![0u8; crate::bf16::TILE_SLOT as usize];
            self.dev.dram_read(&d.w4, tensor.slot(t), &mut image)?;
            for r in 0..32 {
                for c in 0..32 {
                    let row = t / ct * 32 + r;
                    let col = t % ct * 32 + c;
                    if row < tensor.rows && col < tensor.cols {
                        let at = tt_isa::dm::TILE_DATA as usize + tt_isa::dm::face_index(r, c) * 2;
                        values[row * tensor.cols + col] =
                            u16::from_le_bytes([image[at], image[at + 1]]);
                    }
                }
            }
        }
        Ok(values)
    }

    /// Opt-in BF16 Src pooling of a resident 16x16 block. General F32
    /// reductions retain their SFPU numerical contracts.
    pub fn fpu_pool_block(
        &mut self,
        tensor: &DramTensor,
        op: crate::fpu::PoolOp,
    ) -> Result<DramTensor, TensorError> {
        let work = crate::fpu::block(&mut self.dram_state()?.alloc, tensor, op)?;
        self.execute(work, RESET_BUDGET)
    }

    /// Transpose a resident 16x16 F32-storage face through an explicit Src
    /// conversion. This operation truncates operands according to `route`;
    /// raw tensor copies and payload-preserving transpose use repack instead.
    pub fn transpose_src_block(
        &mut self,
        tensor: &DramTensor,
        route: SrcRoute,
    ) -> Result<DramTensor, TensorError> {
        let work = crate::fpu::transpose_block(&mut self.dram_state()?.alloc, tensor, route)?;
        self.execute(work, RESET_BUDGET)
    }

    /// Pool a physically packed BF16 16x16 block directly through SrcA.
    /// The result is F32; BF16 inputs do not undergo another rounding step.
    pub fn bf16_pool_block(
        &mut self,
        tensor: &crate::bf16::Bf16Tensor,
        op: crate::fpu::PoolOp,
    ) -> Result<DramTensor, TensorError> {
        let work = crate::fpu::bf16_block(&mut self.dram_state()?.alloc, tensor, op)?;
        self.execute(work, RESET_BUDGET)
    }

    /// Multiply packed BF16 matrices with F32 accumulation/output. Large K
    /// reloads the prior F32 accumulator between packed operand blocks.
    pub fn matmul_bf16(
        &mut self,
        a: &crate::bf16::Bf16Tensor,
        b: &crate::bf16::Bf16Tensor,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<DramTensor, TensorError> {
        let (mt, kt, nt) = (
            a.rows.div_ceil(32),
            a.cols.div_ceil(32),
            b.cols.div_ceil(32),
        );
        let units = self.spread_for(mt * nt * kt);
        let k_limit = self.matmul_k_block_limit.map(std::num::NonZeroUsize::get);
        let work = crate::bf16::matmul(
            &mut self.dram_state()?.alloc,
            a,
            b,
            fidelity,
            units,
            k_limit,
        )?;
        self.execute_on(work, units, budget)
    }

    /// Traceable shape/index constants encoded as mover immediates. Callers
    /// must use upload for tensor data; this API is for operation metadata.
    pub fn metadata(
        &mut self,
        bits: &[u32],
        dims: [usize; 2],
        elem: tensor::Elem,
    ) -> Result<DramTensor, TensorError> {
        let work = tensor::metadata(&mut self.dram_state()?.alloc, bits, dims, elem)?;
        self.execute(work, RESET_BUDGET)
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

    /// Where tensors take the tile layout on their way to the card, and lose
    /// it on the way back: [`Tilize::Host`] (the default) or
    /// [`Tilize::Card`]. Either way the session's callers -- Burn among them
    /// -- see row-major data only, and the bits are the same. The host is
    /// the default because it is faster everywhere card 0 measured
    /// (`silicon_bench_host_dma::session_transfers`): the host copies the
    /// rows into pinned memory either way, at about the cost of tilizing
    /// them, and a mover tilizes at ~2.6 us a tile (1024² up, 32 tiles: card
    /// 0.98 ms, host 0.55). Applies to the card's DMA
    /// ([`Session::set_host_dma`]); the BAR path always tilizes on the host.
    pub fn set_tilize(&mut self, at: Tilize) {
        self.tilize = at;
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
        let stride = tensor::row_major_stride(t.cols);
        if self.tilize == Tilize::Card && t.rows * stride <= HOST_DMA_STAGING {
            // Row-major into the pinned memory; the card makes the tiles.
            let at = self.staging_region(t.rows * stride)?;
            let host = self.staging().expect("checked above");
            host.with_bytes(&mut |buf| {
                copy_rows(values.bytes(), t.rows, t.cols * 4, &mut buf[at..], stride)
            });
            let base = host.noc_address() + at as u64;
            let units = self.spread_for(tiles);
            let jobs =
                tensor::row_major_dma_jobs(t.tensor_ref(), [t.rows, t.cols], true, base, units);
            self.staging_busy = true;
            self.submit_jobs_on(jobs, units, RESET_BUDGET)?;
            return Ok(Some(()));
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
            let units = self.spread_for(n);
            let jobs = tensor::host_dma_jobs(t.tensor_ref(), first..first + n, true, base, units);
            self.staging_busy = true;
            self.submit_jobs_on(jobs, units, RESET_BUDGET)?;
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

    /// A development probe (`docs/plans/decode-step-bandwidth.md` B2, the cross-op weight
    /// prefetch): read `t`'s tiles into L1 on `units`, `per_unit` consecutive tiles a unit (one
    /// `READ_RUN` each, unit `units.start + k` reading tiles `k * per_unit..`) into slots from
    /// `at`, queued without a barrier, so ops queued after it on other units do not wait for it;
    /// [`Session::sync`] does. What a prefetch on units an op leaves idle would issue. Nothing
    /// reads the slots, and an op on those units plans over them.
    #[doc(hidden)]
    pub fn probe_prefetch(
        &mut self,
        t: &DramTensor,
        units: std::ops::Range<usize>,
        at: u64,
        per_unit: usize,
    ) -> Result<(), TensorError> {
        use tt_isa::dm::{record, TILE_SLOT};
        let refuse = |why: String| Err(TensorError::Shape(format!("probe_prefetch: {why}")));
        let [rt, ct] = t.grid();
        let data = tt_isa::l1::DATA;
        if self.capture.is_some() {
            return refuse("not in a capture".into());
        }
        if units.is_empty() || units.end > self.units.len() || per_unit == 0 {
            return refuse(format!("units {units:?} of {}", self.units.len()));
        }
        if units.len() * per_unit > rt * ct {
            return refuse(format!("{} tiles of {}", units.len() * per_unit, rt * ct));
        }
        if at % 64 != 0 || at < data.base || at + per_unit as u64 * TILE_SLOT > data.end {
            return refuse(format!("{per_unit} slots at {at:#x}"));
        }
        let x = t.tensor_ref();
        let jobs = (0..units.end)
            .map(|u| match u.checked_sub(units.start) {
                None => Vec::new(),
                Some(k) => {
                    let mut entries = vec![[
                        record::READ_RUN,
                        (k * per_unit) as u32,
                        per_unit as u32,
                        at as u32,
                        0,
                        ct as u32,
                        0,
                        0,
                    ]];
                    entries.extend(x.encode());
                    vec![Step::List {
                        what: "probe prefetch",
                        entries,
                    }]
                }
            })
            .collect();
        self.unbarriered = true;
        let r = self.submit_jobs_on(jobs, units.end, RESET_BUDGET);
        self.unbarriered = false;
        r
    }

    /// A development probe (`docs/plans/decode-to-bandwidth.md` Phase 0, the read ceiling): every
    /// tile of `t` read once into L1 on the first `units` units, as one op with its barrier, so it
    /// can be captured and replayed and its time is the card's. `order` deals the tiles as
    /// [`StreamOrder`] says; each unit reads them as `READ_RUN`s of at most `window` tiles into the
    /// same `window` slots at the top of `DATA`, which nothing reads. Returns, for each unit, the
    /// tiles `(i, j)` its last run left in slots `0..` (for the caller to check).
    #[doc(hidden)]
    pub fn probe_stream(
        &mut self,
        t: &DramTensor,
        units: usize,
        order: StreamOrder,
        window: usize,
    ) -> Result<Vec<Vec<[usize; 2]>>, TensorError> {
        use tt_isa::dm::{record, TILE_SLOT};
        let refuse = |why: String| Err(TensorError::Shape(format!("probe_stream: {why}")));
        let [rt, ct] = t.grid();
        let data = tt_isa::l1::DATA;
        if units == 0 || units > self.units.len() {
            return refuse(format!("{units} units of {}", self.units.len()));
        }
        if window == 0 || data.base + window as u64 * TILE_SLOT > data.end {
            return refuse(format!("{window} slots"));
        }
        let at = (data.end - window as u64 * TILE_SLOT) as u32;
        let x = t.tensor_ref();
        // Each unit's runs, as `(first, count, column_major)` over `t`'s grid.
        let runs: Vec<Vec<(usize, usize, bool)>> = match order {
            StreamOrder::Columns => (0..units)
                .map(|u| {
                    (u..ct)
                        .step_by(units)
                        .flat_map(|j| {
                            (0..rt)
                                .step_by(window)
                                .map(move |i| (j * rt + i, window.min(rt - i), true))
                        })
                        .collect()
                })
                .collect(),
            StreamOrder::Rows => tensor::runs(rt * ct, units, usize::MAX)
                .into_iter()
                .map(|r| {
                    r.clone()
                        .step_by(window)
                        .map(|n| (n, window.min(r.end - n), false))
                        .collect()
                })
                .collect(),
        };
        let mut last = Vec::with_capacity(units);
        let mut jobs: Vec<tensor::Job> = Vec::with_capacity(units);
        for unit_runs in &runs {
            let mut entries = Vec::new();
            for &(first, count, column_major) in unit_runs {
                entries.push([
                    record::READ_RUN,
                    first as u32,
                    count as u32,
                    at,
                    if column_major { 4 } else { 0 },
                    ct as u32,
                    if column_major { rt as u32 } else { 0 },
                    0,
                ]);
                entries.extend(x.encode());
            }
            last.push(unit_runs.last().map_or(Vec::new(), |&(first, count, cm)| {
                (first..first + count)
                    .map(|n| {
                        if cm {
                            [n % rt, n / rt]
                        } else {
                            [n / ct, n % ct]
                        }
                    })
                    .collect()
            }));
            jobs.push(if entries.is_empty() {
                Vec::new()
            } else {
                vec![Step::List {
                    what: "probe stream",
                    entries,
                }]
            });
        }
        self.submit_jobs_on(jobs, units, RESET_BUDGET)?;
        Ok(last)
    }

    /// Run a transfer's jobs to their end, without a barrier (nothing is
    /// queued behind them before the sync).
    fn submit_dma(&mut self, jobs: Vec<tensor::Job>, units: usize) -> Result<(), TensorError> {
        self.unbarriered = true;
        let r = self.submit_jobs_on(jobs, units, RESET_BUDGET);
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
        let mut out = huge_zeroed(t.rows * t.cols);
        let stride = tensor::row_major_stride(t.cols);
        // Rows that fill at most a quarter of their tile rows come back untilized by
        // the card whatever the setting: the tiles would carry four times the data
        // or more across PCIe (a `[1, 49152]` row of logits: 6.3 MB of tiles for its
        // 192 KiB, about 1 ms on card 1), against ~2.6 us of mover time a tile spread
        // over the units (`docs/plans/replay-fixed-cost.md`).
        let mostly_padding = 4 * t.rows <= 32 * rt;
        if (self.tilize == Tilize::Card || mostly_padding) && t.rows * stride <= HOST_DMA_STAGING {
            // The card untilizes; the rows come back as they are. A unit for every 96
            // tiles: each unit is a list and a drain for the host, so past that more of
            // them cost more than their untilizing saves (the `[1, 49152]` logits on card
            // 1: 0.95 ms on 120 units, 0.46 on 16, 0.55 on 8; `replay-fixed-cost.md`).
            let base = self.staging().expect("checked above").noc_address();
            let units = self.spread_for(tiles).min(tiles.div_ceil(96));
            let jobs =
                tensor::row_major_dma_jobs(t.tensor_ref(), [t.rows, t.cols], false, base, units);
            self.submit_dma(jobs, units)?;
            let host = self.staging().expect("checked above");
            // SAFETY: `out` is `rows * cols` plain f32s, whose bytes any
            // pattern is.
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4)
            };
            host.with_bytes(&mut |buf| copy_rows(buf, t.rows, stride, bytes, t.cols * 4));
            return Ok(Some(out));
        }
        let per = HOST_DMA_STAGING / tt_isa::dm::TILE_SLOT as usize;
        for first in (0..tiles).step_by(per) {
            let n = (tiles - first).min(per);
            let base = self.staging().expect("checked above").noc_address();
            let units = self.spread_for(n);
            let jobs = tensor::host_dma_jobs(t.tensor_ref(), first..first + n, false, base, units);
            self.submit_dma(jobs, units)?;
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
                let had_kernels = self.units[u].queued.iter().any(|q| q.roles);
                self.units[u].queued.clear();
                self.recover(&[(u, had_kernels)]);
            }
            self.restart_barriers();
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
        // Every list has run, so every barrier's units have arrived: no tile
        // has anything to catch up on.
        let target = self.barriers;
        for unit in &mut self.units {
            unit.barrier_seen = target;
        }
        Ok(())
    }

    /// A wait-only `BARRIER` for `target` (`dm::op::BARRIER` word 4 = 1).
    fn barrier_wait(&self, target: u32) -> [u32; 8] {
        let c = self.units[0].tile;
        [
            tt_isa::dm::op::BARRIER,
            target,
            c.x() as u32,
            c.y() as u32,
            1,
            0,
            0,
            0,
        ]
    }

    /// The barrier counter and the session's count start again at zero, and no
    /// tile has a target to wait for: on a recovery, with nothing queued.
    fn restart_barriers(&mut self) {
        self.barriers = 0;
        for unit in &mut self.units {
            unit.barrier_seen = 0;
        }
    }

    /// Order unit `u` after every barrier it sat out, before its next list: one
    /// wait-only `BARRIER` for the latest target (`dm::op::BARRIER`), which the
    /// units that ran those ops reach only once their writes have landed. A unit
    /// that has passed the latest barrier, or a session of one unit, needs none.
    ///
    /// This is what lets an op's barrier involve only the units it ran on: a tile
    /// an op does not use enqueues nothing for it, where it used to enqueue a
    /// barrier list of its own (one PCIe enqueue a tile an op: a 2-tile op on 120
    /// tiles was 118 of them, and the all-tiles transformer 355,000 a step).
    fn catch_up(&mut self, u: usize) -> Result<(), TensorError> {
        let target = self.barriers;
        self.catch_up_to(u, target)
    }

    /// [`Session::catch_up`] to `target`: the latest barrier before the op being
    /// queued, which its own barrier may already have moved past.
    fn catch_up_to(&mut self, u: usize, target: u32) -> Result<(), TensorError> {
        if self.units.len() < 2 || self.units[u].barrier_seen == target {
            return Ok(());
        }
        // Inside a capture this is the capture's own run catching up: the trace's
        // stream holds its own wait (`enqueue_work_inner`).
        self.ensure_unit(u)?;
        let entry = self.barrier_wait(target);
        let Session { dev, units, .. } = self;
        let unit = &mut units[u];
        let (r, m) = (
            unit.resident.as_ref().unwrap(),
            unit.mover.as_mut().unwrap(),
        );
        let number = m.enqueue(dev, r.window(), &[entry])?;
        unit.queued.push_back(QueuedList {
            roles: false,
            number,
            kernels: false,
            what: "barrier catch-up",
        });
        unit.lists += 1;
        unit.barrier_seen = target;
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
        let operation = dev
            .read32(
                r.window(),
                unit.tile,
                tt_isa::dm::MAILBOX_BASE + tt_isa::mailbox::offset::RESULT,
            )
            .unwrap_or(0);
        Err(TensorError::Shape(format!(
            "queued list {number}, `{what}`, on tile ({}, {}) failed at op {operation:#x}: {}",
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
        let trace_word = self.trace_word();
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
            // NC may still be waiting on B's packet: it stops, and starts
            // again with B below.
            if let Some(nc) = unit.nc.take() {
                nc.stop(dev, r.window())?;
            }
            unit.mover = Some(DataMover::start(
                dev,
                r.window(),
                unit.tile,
                &d.dram,
                d.image,
            )?);
            // NC, the fixed writer every compute region needs, starts with B,
            // so the first region after a restart does not load its image.
            let nc = DataMover::start_on(
                dev,
                r.window(),
                unit.tile,
                &d.dram,
                tt_isa::dm::Mover::NC,
                d.nc_image,
            )?;
            nc.set_write_noc(dev, r.window(), crate::dm::WriteNoc::Noc1)?;
            unit.nc = Some(nc);
            if profiling.is_some() {
                dev.write32(r.window(), unit.tile, tt_isa::dm::TRACE, trace_word)?;
                dev.write32(r.window(), unit.tile, tt_isa::dm::TRACE_WINDOW, 0)?;
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
                // Every tile's last target was on the old count; with nothing
                // queued, none of them has anything to wait for.
                for unit in units.iter_mut() {
                    unit.barrier_seen = 0;
                }
            }
        }
        Ok(())
    }

    /// Queue an op's jobs ([`Session::execute`] with batching): each unit's
    /// share as few lists as fit, then -- with more than one unit -- a barrier
    /// on every unit, since the next op may read what any unit wrote.
    fn enqueue_work(
        &mut self,
        jobs: Vec<tensor::Job>,
        units: usize,
        budget: u64,
    ) -> Result<(), TensorError> {
        let starts: Option<Vec<usize>> = self
            .capture
            .as_ref()
            .map(|c| c.units.iter().map(|u| u.stream.len()).collect());
        let mut what = "";
        let start = self.mark();
        let result = self.enqueue_work_inner(jobs, units, budget, &mut what);
        self.host.ops += 1;
        self.stage(HostStage::Enqueue, start);
        if let (Some(c), Some(starts)) = (self.capture.as_mut(), starts) {
            // Part of an op captured is no op a replay could run.
            c.failed |= result.is_err();
            c.ops.push(trace::OpRecord {
                what,
                label: self.op_label,
                entries: c
                    .units
                    .iter()
                    .zip(starts)
                    .map(|(u, s)| s..u.stream.len())
                    .collect(),
                barrier: self.units.len() > 1,
                stamps: [0; 2],
                bytes: Default::default(),
            });
        }
        result
    }

    fn enqueue_work_inner(
        &mut self,
        jobs: Vec<tensor::Job>,
        units: usize,
        budget: u64,
        what: &mut &'static str,
    ) -> Result<(), TensorError> {
        let n = self.units.len();
        // The units the jobs were built for: the first `units` of the session's.
        let spread = units.clamp(1, n);
        // One queue a unit the op can run on, not one a unit of the session: on many
        // tiles the empty ones were most of the host's time laying an op out (a
        // 120-tile session at a spread of 2: 7 us an op against 1.6 on 2 tiles).
        let mut queues: Vec<Vec<Step>> = vec![Vec::new(); spread];
        for (j, job) in jobs.into_iter().enumerate() {
            queues[j % spread].extend(job);
        }
        // The units that arrive at this op's barrier: the ones it runs on, in a capture
        // too, whose trace then records how many arrived (`Capture::barriers`) for its
        // replay to count. Nothing below loops over the session's other units: on many
        // tiles that was a host cost every op paid.
        let capturing = self.capture.is_some();
        let arriving: Vec<usize> = (0..spread).filter(|&u| !queues[u].is_empty()).collect();
        // What each unit the op runs on must wait for first: the latest barrier
        // before this op, if it sat out the ops up to it. Taken before this op's
        // barrier is counted, since that target includes the unit's own arrival,
        // which it would wait for in vain. It rides at the front of the unit's first
        // list for the op ([`lead_with`]), not in a list of its own.
        let mut behind: Vec<Option<u32>> = vec![None; spread];
        if n > 1 {
            for &u in &arriving {
                self.ensure_unit(u)?;
            }
            for (u, b) in behind.iter_mut().enumerate() {
                if !queues[u].is_empty() && self.units[u].barrier_seen != self.barriers {
                    *b = Some(self.barriers);
                }
            }
        }
        // In a capture, what the trace's stream of each unit the op runs on must wait for: the
        // latest captured barrier, if the unit sat out ops since it last accounted for one. This
        // is the trace's own count, not `behind`: the capture's run may have synced since (the
        // profile mode syncs after every op), which catches every unit up on the card but puts
        // no wait in a replay's stream.
        let trace_behind: Vec<Option<u32>> = match &self.capture {
            Some(c) if n > 1 => (0..spread)
                .map(|u| {
                    (!queues[u].is_empty() && c.units[u].barriers_accounted != c.barrier_ops)
                        .then(|| self.barriers.wrapping_sub(c.barriers_base))
                })
                .collect(),
            _ => vec![None; spread],
        };
        let mut trace_behind = trace_behind;
        // In rounds: every unit's first list, then every unit's second, and
        // so on. A unit whose share is more than its ring holds makes the
        // host wait for room; enqueued unit by unit, that wait came before
        // the other units had anything to do, and they ran one after another
        // (a pipelined 1024^3 matmul on 8 tiles: 9 ms in the call, against
        // 0.4).
        let start = self.mark();
        // Every unit's cache has the same bounds (`limit_program_cache`).
        let cache = self.units[0].programs.region().len();
        let mut per_unit: Vec<std::collections::VecDeque<Segment>> = queues
            .into_iter()
            .map(|steps| match steps.is_empty() {
                true => Ok(Default::default()),
                false => streaming_segments(steps, cache).map(Into::into),
            })
            .collect::<Result<_, _>>()?;
        self.stage(HostStage::Segments, start);
        // With more than one unit, a barrier on each unit this op ran on, after
        // its share, since the next op may read what any of them wrote; a unit
        // it did not run on enqueues nothing, and catches up before its next
        // list ([`Session::catch_up`]). It rides at the end of the unit's last
        // list where there is room -- a list of its own was a second enqueue a
        // unit an op, half the host's time on many tiles (checklist 9.17) --
        // except while capturing, whose streams take it as an entry of its own.
        let arrivals = arriving.len() as u32;
        let barrier = (n > 1 && !self.unbarriered && arrivals > 0).then(|| {
            self.barriers = self.barriers.wrapping_add(arrivals);
            let target = self.barriers;
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
        let mut folded = vec![false; spread];
        // A prefetching product's `PREFETCH` lists (`tensor::BlockPlan::push_prefetched_jobs`): in
        // a capture each goes into its unit's stream ahead of the unit's catch-up wait, behind a
        // wait for the anchor -- the last weight-streaming product -- alone, so the unit reads
        // its `B` while the ops since that product run. The unit ran nothing since it (the
        // session took only such units), so nothing between touches the slots it reads into.
        if let (Some(anchor), true) = (self.prefetch_anchor, capturing) {
            for (u, q) in per_unit.iter_mut().enumerate() {
                if q.front().is_some_and(|seg| seg.what == tensor::PREFETCH) {
                    let seg = q.pop_front().expect("a front segment");
                    let wait = self.barrier_wait(anchor);
                    self.capture.as_mut().unwrap().units[u].stream.push(wait);
                    self.enqueue_segment(u, &seg, budget)?;
                }
            }
        }
        while per_unit.iter().any(|q| !q.is_empty()) {
            for (u, q) in per_unit.iter_mut().enumerate() {
                if let Some(mut seg) = q.pop_front() {
                    if what.is_empty() {
                        *what = seg.what;
                    }
                    if let Some(rel) = trace_behind[u].take() {
                        // The trace's stream waits for the barrier, relative to the capture's
                        // first as every captured target is, standing for the barriers the unit
                        // sat out (which the profile mode's window counts as it counts
                        // arrivals).
                        let mut wait = self.barrier_wait(rel);
                        let c = self.capture.as_mut().unwrap();
                        wait[tt_isa::dm::BARRIER_STANDS_FOR] =
                            c.barrier_ops - c.units[u].barriers_accounted;
                        c.units[u].barriers_accounted = c.barrier_ops;
                        c.units[u].stream.push(wait);
                    }
                    if let Some(target) = behind[u].take() {
                        if capturing {
                            // The capture's own run waits now, in a list of its own the trace
                            // does not hold.
                            let start = self.mark();
                            self.catch_up_to(u, target)?;
                            self.stage(HostStage::Barrier, start);
                        } else if lead_with(&mut seg, self.barrier_wait(target)) {
                            self.units[u].barrier_seen = target;
                        } else {
                            // A list already as long as a list may be: the wait goes
                            // ahead of it as a list of its own.
                            let start = self.mark();
                            self.catch_up_to(u, target)?;
                            self.stage(HostStage::Barrier, start);
                        }
                    }
                    if let Some(entry) = fold {
                        if q.is_empty() && seg.entries.len() < tt_isa::dm::LIST_MAX as usize {
                            if seg.streaming {
                                seg.entries[0][4] = 1;
                            }
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
                // Relative to the capture's first: a replay adds its own. Only the
                // units the op ran on arrive; one that sat it out waits before its
                // next op (above) and at the trace's end (`end_trace`).
                let rel = entry[1].wrapping_sub(c.barriers_base);
                c.barriers += arrivals;
                c.barrier_ops += 1;
                for &u in &arriving {
                    c.units[u].barriers_accounted = c.barrier_ops;
                    c.units[u]
                        .stream
                        .push([entry[0], rel, entry[2], entry[3], 0, 0, 0, 0]);
                }
            }
            for &u in arriving
                .iter()
                .filter(|&&u| !folded.get(u).copied().unwrap_or(false))
            {
                let Session { dev, units, .. } = self;
                let unit = &mut units[u];
                let (r, m) = (
                    unit.resident.as_ref().unwrap(),
                    unit.mover.as_mut().unwrap(),
                );
                let number = m.enqueue(dev, r.window(), &[entry])?;
                unit.queued.push_back(QueuedList {
                    roles: false,
                    number,
                    kernels: false,
                    what: "barrier",
                });
            }
            for &u in &arriving {
                self.units[u].barrier_seen = entry[1];
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
        if seg.streaming {
            self.ensure_nc(u)?;
        }
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
        if seg.streaming && seg.entries[0][3] == 2 {
            self.pipelined += seg.kernel_roles.len() as u64;
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
        let start = self.mark();
        let checked = self.units[u].mover.as_ref().unwrap().check_list(&entries);
        self.stage(HostStage::Check, start);
        let start = self.mark();
        let Session { dev, units, .. } = self;
        let unit = &mut units[u];
        let (r, m) = (
            unit.resident.as_mut().unwrap(),
            unit.mover.as_mut().unwrap(),
        );
        // A list that fails its check is never written.
        let enqueued = match checked {
            Ok(()) => m.enqueue_checked(dev, r.window(), &entries),
            Err(e) => Err(e),
        };
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
            roles: !seg.kernel_roles.is_empty(),
            number,
            kernels: !seg.kernel_roles.is_empty(),
            what: seg.what,
        });
        unit.lists += 1;
        unit.steps += seg.steps;
        if seg.streaming && seg.kernel_roles.is_empty() {
            self.dataflow.transfer_packets += 1;
        } else if seg.streaming {
            self.dataflow.regions += 1;
            self.dataflow.batches += seg.kernel_roles.len() as u64;
            for entry in seg
                .entries
                .iter()
                .filter(|e| e[0] == tt_isa::dm::op::RELEASED)
            {
                if entry[2] != 0 {
                    self.dataflow.pack_waits += 1;
                } else {
                    self.dataflow.release_waits += 1;
                }
            }
        }
        Ok(())
    }

    fn ensure_nc(&mut self, u: usize) -> Result<(), TensorError> {
        if self.units[u].nc.is_some() {
            return Ok(());
        }
        let Session {
            dev, units, dram, ..
        } = self;
        let d = dram
            .as_ref()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        let unit = &mut units[u];
        let window = unit.resident.as_ref().unwrap().window();
        let nc = DataMover::start_on(
            dev,
            window,
            unit.tile,
            &d.dram,
            tt_isa::dm::Mover::NC,
            d.nc_image,
        )?;
        nc.set_write_noc(dev, window, crate::dm::WriteNoc::Noc1)?;
        unit.nc = Some(nc);
        Ok(())
    }

    /// Add a segment about to be enqueued on unit `u` to the capture, as a
    /// replay will run it: before its kernel, the setup run [`Resident::reserve`]
    /// did for it, if any, as a kernel of its own, and `POKE`s for whatever
    /// of the roles' descriptors the stream has not set; then its entries,
    /// each `KERNEL`'s, `LAUNCH`'s and `KERNEL_WAIT`'s generation relative to
    /// the capture and the programs they name held, a packet's sections stored
    /// as streams of their own.
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
                held.push((self.units[u].programs.generation(), at));
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
            let relative = capture_commands(entries, base, |address| {
                let programs = &mut self.units[u].programs;
                if programs.hold(address) {
                    held.push((programs.generation(), address));
                }
            })?;
            if seg.streaming {
                for (index, roles) in seg.kernel_roles.iter().enumerate() {
                    for role in 0..3 {
                        if roles[role].is_empty() {
                            continue;
                        }
                        let (words, _, hash) =
                            stored_program(roles, &seg.kernel_loops[index], role)?;
                        let address = match self.units[u].programs.place_hashed(&words, hash) {
                            Ok(crate::program_cache::Placed::Hit(address)) => address,
                            _ => {
                                return Err(TensorError::Shape(
                                    "stream body disappeared during capture".into(),
                                ))
                            }
                        };
                        let programs = &mut self.units[u].programs;
                        if programs.hold(address) {
                            held.push((programs.generation(), address));
                        }
                    }
                }
                crate::dm::check_pair(&relative).map_err(|code| {
                    TensorError::Shape(format!(
                        "captured `{}` breaks the ownership rules: mover error {code}",
                        seg.what
                    ))
                })?;
                // The sections are stored with the stream when the capture ends, so a
                // trace takes one region a unit (`store_stream`); the offsets are
                // filled in then.
                let writer_at = 1 + relative[0][1] as usize;
                let mut counts = [0u32; 2];
                for (i, part) in [&relative[1..writer_at], &relative[writer_at..]]
                    .into_iter()
                    .enumerate()
                {
                    let words = trace::chunked(part);
                    out.sections_composition
                        .add(&trace::Composition::of(part, words.len()));
                    counts[i] = words.len() as u32;
                    out.pending_sections.push(words);
                }
                out.sections.extend(counts);
                out.pair_calls.push(out.stream.len());
                out.stream.push([
                    op::PAIR_CALL,
                    0,
                    0,
                    counts[0],
                    0,
                    0,
                    counts[1],
                    relative[0][3],
                ]);
            } else {
                out.stream.extend(relative);
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
            barrier_ops: 0,
            units,
            freed: Vec::new(),
            ops: Vec::new(),
            failed: false,
            weight_anchor: None,
        });
        Ok(())
    }

    /// End the capture: wait for its ops, and store each unit's stream in
    /// GDDR. On any failure nothing is kept, and what the capture held is
    /// given back.
    pub fn end_trace(&mut self) -> Result<TraceId, TensorError> {
        let Some(mut capture) = self.capture.take() else {
            return Err(TraceError::NotCapturing.into());
        };
        // A unit that sat out the trace's last ops waits for its last barrier at the end of
        // its stream, so a replay leaves every unit it ran on past all of its barriers.
        let last = self.barriers.wrapping_sub(capture.barriers_base);
        let ops = capture.barrier_ops;
        for uc in capture.units.iter_mut() {
            if !uc.stream.is_empty() && uc.barriers_accounted != ops {
                let mut wait = self.barrier_wait(last);
                wait[tt_isa::dm::BARRIER_STANDS_FOR] = ops - uc.barriers_accounted;
                uc.barriers_accounted = ops;
                uc.stream.push(wait);
            }
        }
        let held: Vec<Vec<(u64, u64)>> = capture
            .units
            .iter()
            .map(|u| u.held_programs.clone())
            .collect();
        let result = self.store_trace(capture);
        if result.is_err() {
            for (u, held) in held.iter().enumerate() {
                for &(generation, at) in held {
                    self.units[u].programs.release_held(generation, at);
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
            mut ops,
            failed,
            ..
        } = capture;
        // What each op could stamp in each detail of the profile mode, on its busiest unit: what
        // sizes a profiled replay's window (`profile_mode`).
        for op in &mut ops {
            let mut most = [0u32; 2];
            for (uc, range) in units.iter().zip(&op.entries) {
                let s = op_stamps(uc, range.clone());
                most = [most[0].max(s[0]), most[1].max(s[1])];
            }
            op.stamps = most;
        }
        // In the profile mode, what each op moves through GDDR: its records expanded on the host,
        // as the movers expand them. The streams are not kept once stored.
        if self.profiler.is_some() || self.count_trace_bytes {
            for op in &mut ops {
                for (uc, range) in units.iter().zip(&op.entries) {
                    op_bytes(uc, range.clone(), &mut op.bytes);
                }
            }
        }
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

    /// Unit `u`'s captured stream and its packets' sections, laid out for
    /// streaming and written to GDDR in one region: the stream first, then each
    /// section from a 64-byte boundary (what a mover's chunk read needs,
    /// `tt_isa::dram::ALIGN`), each `PAIR_CALL` given its sections' offsets. One
    /// allocation and one write a unit, rather than a whole slot and a write for
    /// each section.
    fn store_stream(
        &mut self,
        u: usize,
        uc: &trace::UnitCapture,
    ) -> Result<trace::UnitTrace, TensorError> {
        use tt_isa::dm::ENTRY_BYTES;
        debug_assert_eq!(uc.pending_sections.len(), 2 * uc.pair_calls.len());
        let align = |bytes: u64| bytes.next_multiple_of(tt_isa::dram::ALIGN);
        let top = trace::chunked(&uc.stream).len();
        let mut offsets = Vec::with_capacity(uc.pending_sections.len());
        let mut end = align(top as u64 * ENTRY_BYTES);
        for section in &uc.pending_sections {
            offsets.push(end);
            end = align(end + section.len() as u64 * ENTRY_BYTES);
        }
        let generations = {
            let r = self.units[u].resident.as_ref().unwrap();
            r.generation().wrapping_sub(uc.generation_base)
        };
        let Session { dev, dram, .. } = self;
        let d = dram.as_mut().expect("checked at the capture's start");
        // Each unit's on its own channel where there are enough, so the
        // movers' reads of their streams spread out.
        let region = d.alloc.alloc_on(u % d.alloc.channel_count(), end)?;
        let range = region.region().expect("a one-channel placement");
        let (channel, base) = (range.channel().index() as u32, range.offset());
        let mut stream = uc.stream.clone();
        for (i, &at) in uc.pair_calls.iter().enumerate() {
            let entry = &mut stream[at];
            entry[1] = channel;
            entry[2] = (base + offsets[2 * i]) as u32;
            entry[4] = channel;
            entry[5] = (base + offsets[2 * i + 1]) as u32;
        }
        let words = trace::chunked(&stream);
        debug_assert_eq!(words.len(), top, "the offsets do not move a record");
        let own = trace::Composition::of(&stream, words.len());
        // Whole slots, as every GDDR write here is; the gaps are zeros.
        let mut bytes = vec![0u8; range.len() as usize];
        let mut put = |at: u64, words: &[[u32; 8]]| {
            for (i, word) in words.iter().flatten().enumerate() {
                let at = at as usize + 4 * i;
                bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        };
        put(0, &words);
        for (section, &at) in uc.pending_sections.iter().zip(&offsets) {
            put(at, section);
        }
        if let Err(e) = dev.dram_write(&d.w4, range, &bytes) {
            d.alloc.free(&region);
            return Err(e.into());
        }
        let mut composition = own;
        composition.add(&uc.sections_composition);
        Ok(trace::UnitTrace {
            channel,
            offset: base as u32,
            count: words.len() as u32,
            generations,
            held_programs: uc.held_programs.clone(),
            sections: uc.sections.clone(),
            composition,
            top_level_entries: words.len() as u32,
            gddr_bytes: range.len(),
            barriers: barrier_entries(&uc.stream),
            stream: region,
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
        // Every unit that runs the trace starts after the ops before it.
        for (u, call) in calls.iter().enumerate() {
            if call.is_some() {
                self.catch_up(u)?;
            }
        }
        // The profile mode stamps a window of the trace's ops on every unit.
        let window = match self.profiler.is_some() {
            true => Some(self.profile_window(id)),
            false => None,
        };
        let barrier_base = self.barriers;
        let ran: Vec<bool> = calls.iter().map(Option::is_some).collect();
        let Session { dev, units, .. } = self;
        if let Some(w) = window {
            for unit in units.iter() {
                if let (Some(r), Some(_)) = (unit.resident.as_ref(), unit.mover.as_ref()) {
                    dev.write32(r.window(), unit.tile, tt_isa::dm::TRACE_WINDOW, w)?;
                }
            }
        }
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
                roles: generations != 0,
                number,
                kernels: false,
                what: "trace",
            });
            unit.lists += 1;
            // The replay sets the roles' descriptors and semaphores without
            // the host.
            r.forget_tile_state();
        }
        // The arrivals the capture counted: each captured barrier's units.
        self.barriers = self.barriers.wrapping_add(barriers);
        if barriers > 0 {
            let target = self.barriers;
            for (unit, ran) in self.units.iter_mut().zip(ran) {
                if ran {
                    unit.barrier_seen = target;
                }
            }
        }
        if window.is_some() {
            // Profiled, a replay is timed alone, as an eager op is.
            self.sync()?;
            self.profile_collect();
        }
        Ok(())
    }

    /// The window of trace `id`'s ops its next replay stamps (`tt_isa::dm::TRACE_WINDOW`), its ops
    /// made known to the profile mode on its first replay.
    fn profile_window(&mut self, id: TraceId) -> u32 {
        let detail = match self.profiler.as_ref().map(|p| p.options.detail) {
            Some(crate::profile_mode::Detail::Full) => 1,
            _ => 0,
        };
        let ops = self.traces.get(&id.0).map(|t| &t.ops[..]).unwrap_or(&[]);
        let Some(p) = self.profiler.as_mut() else {
            return 0;
        };
        p.know_trace(id.0, || {
            let info = ops
                .iter()
                .map(|o| crate::profile_mode::OpInfo {
                    label: o.label,
                    what: o.what,
                    units: o
                        .entries
                        .iter()
                        .filter(|r| r.len() > o.barrier as usize)
                        .count(),
                    bytes: o.bytes,
                })
                .collect();
            let stamps = ops.iter().map(|o| o.stamps[detail]).collect();
            let segmented = !ops.is_empty() && ops.iter().all(|o| o.barrier);
            (info, stamps, segmented)
        });
        p.next_window(id.0)
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
            // A reset since the capture cleared the cache: what a hold named
            // is gone, and its address may be another trace's program now.
            for (generation, at) in ut.held_programs {
                unit.programs.release_held(generation, at);
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

    /// Name the ops queued from here on, for a capture's records
    /// ([`trace::OpRecord::label`]): a caller's own name for what it is running
    /// (burn-tt sets the Burn op whose job it is), so a replay's device time can be
    /// charged back to it. Nothing else reads it.
    pub fn set_op_label(&mut self, label: &'static str) {
        self.op_label = label;
    }

    /// The arrivals a replay of trace `id` adds to the barrier counter: each captured op's
    /// barrier counts the units the op ran on.
    #[doc(hidden)]
    pub fn trace_arrivals(&self, id: TraceId) -> Result<u32, TensorError> {
        self.traces
            .get(&id.0)
            .map(|t| t.barriers)
            .ok_or_else(|| TraceError::Unknown(id.0).into())
    }

    /// For each unit, the `BARRIER` entries in trace `id`'s stream: `[arrivals, waits, the
    /// barriers both stand for]` (`None` for a unit the trace does not run on).
    #[doc(hidden)]
    pub fn trace_barrier_entries(&self, id: TraceId) -> Result<Vec<Option<[u32; 3]>>, TensorError> {
        let t = self.traces.get(&id.0).ok_or(TraceError::Unknown(id.0))?;
        Ok(t.units
            .iter()
            .map(|u| u.as_ref().map(|u| u.barriers))
            .collect())
    }

    /// The ops trace `id` captured, in order (`crate::trace::OpRecord`).
    pub fn trace_ops(&self, id: TraceId) -> Result<&[trace::OpRecord], TensorError> {
        self.traces
            .get(&id.0)
            .map(|t| &t.ops[..])
            .ok_or(TraceError::Unknown(id.0).into())
    }

    /// Entries in each reader and writer section trace `id` stores in GDDR, per
    /// unit and then per packet (reader, writer), chunk padding included. The
    /// mover fetches a section [`tt_isa::dm::TRACE_CHUNK_ENTRIES`] at a time.
    pub fn trace_sections(&self, id: TraceId) -> Result<Vec<Vec<u32>>, TensorError> {
        let t = self.traces.get(&id.0).ok_or(TraceError::Unknown(id.0))?;
        Ok(t.units
            .iter()
            .map(|u| u.as_ref().map_or_else(Vec::new, |u| u.sections.clone()))
            .collect())
    }

    /// What trace `id` costs each unit (`None` for a unit it does not use): its
    /// stream's entries by op and its chunk padding, the GDDR bytes the stream and its
    /// sections take, and the program-cache bytes the trace pins, against the cache's
    /// size. The two limits that bound a trace are the program cache and GDDR, so a
    /// change that touches traces reports these beside its replay time.
    pub fn trace_footprint(
        &self,
        id: TraceId,
    ) -> Result<Vec<Option<trace::UnitFootprint>>, TensorError> {
        let t = self.traces.get(&id.0).ok_or(TraceError::Unknown(id.0))?;
        Ok(t.units
            .iter()
            .zip(&self.units)
            .map(|(ut, unit)| {
                let ut = ut.as_ref()?;
                // One hold per reference, so the same program appears often; a hold
                // from before a reset names nothing.
                let live: std::collections::BTreeSet<u64> = ut
                    .held_programs
                    .iter()
                    .filter(|&&(generation, _)| generation == unit.programs.generation())
                    .map(|&(_, at)| at)
                    .collect();
                let region = unit.programs.region();
                Some(trace::UnitFootprint {
                    stream: ut.composition.clone(),
                    top_level_entries: ut.top_level_entries,
                    gddr_bytes: ut.gddr_bytes,
                    programs: live.len() as u32,
                    program_bytes: live
                        .iter()
                        .filter_map(|&at| unit.programs.program_bytes(at))
                        .sum(),
                    cache_held_bytes: unit.programs.held_bytes(),
                    cache_bytes: region.end - region.base,
                })
            })
            .collect())
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

    /// Overwrite `t`'s values in place as raw bits: a trace's integer, bool,
    /// or bit-encoded input between replays.
    pub fn write_bits(&mut self, t: &DramTensor, values: &[u32]) -> Result<(), TensorError> {
        self.refuse_while_capturing("write")?;
        self.write_src(t, Src::Bits(values), false)
    }

    /// Free GDDR bytes on the fullest channel.
    pub fn dram_free_bytes(&self) -> u64 {
        self.dram.as_ref().map_or(0, |d| d.alloc.free_bytes())
    }

    /// Explicit reduced-Src matrix arithmetic; outputs accumulate/store F32.
    /// Only RHS row, column and scalar broadcasts are accepted.
    pub fn matrix_eltwise(
        &mut self,
        op: crate::matrix_eltwise::MatrixEltwiseOp,
        a: &DramTensor,
        b: &DramTensor,
        precision: crate::matrix_eltwise::SrcPrecision,
        fidelity: Fidelity,
    ) -> Result<DramTensor, TensorError> {
        use crate::matrix_eltwise as m;
        a.expect("matrix elementwise", tensor::Elem::F32)?;
        b.expect("matrix elementwise", tensor::Elem::F32)?;
        let dims = [a.rows, a.cols];
        let broadcast = m::broadcast(dims, [b.rows, b.cols])?;
        let units = self.spread_for(dims[0].div_ceil(32) * dims[1].div_ceil(32));
        let simulated = self.dev.transport().is_simulated();
        let work = m::build(
            &mut self.dram_state()?.alloc,
            &a.placement,
            &b.placement,
            dims,
            m::Config {
                packed: false,
                simulated,
                precision,
                fidelity,
                op,
                broadcast,
                tail: None,
            },
            units,
        )?;
        self.execute_on(work, units, RESET_BUDGET)
    }

    /// Two matrix stages with an explicit truncated Src precision boundary.
    /// Accepts equal-shape resident F32 operands; allocates only the final F32
    /// output. Padding is undefined, as for matrix elementwise arithmetic.
    pub fn matrix_eltwise_chain(
        &mut self,
        first: crate::matrix_eltwise::MatrixEltwiseOp,
        tail: crate::matrix_eltwise::MatrixChainTail,
        a: &DramTensor,
        b: &DramTensor,
        precision: crate::matrix_eltwise::SrcPrecision,
        fidelity: Fidelity,
    ) -> Result<DramTensor, TensorError> {
        use crate::matrix_eltwise as m;
        a.expect("matrix chain", tensor::Elem::F32)?;
        b.expect("matrix chain", tensor::Elem::F32)?;
        let dims = [a.rows, a.cols];
        if dims != [b.rows, b.cols] {
            return Err(TensorError::Shape(
                "matrix chain requires equal shapes".into(),
            ));
        }
        let units = self.spread_for(dims[0].div_ceil(32) * dims[1].div_ceil(32));
        let simulated = self.dev.transport().is_simulated();
        let work = m::build(
            &mut self.dram_state()?.alloc,
            &a.placement,
            &b.placement,
            dims,
            m::Config {
                packed: false,
                simulated,
                precision,
                fidelity,
                op: first,
                broadcast: m::SrcBroadcast::None,
                tail: Some(tail),
            },
            units,
        )?;
        self.execute_on(work, units, RESET_BUDGET)
    }

    /// Packed BF16 operands stay packed through unpack; the result is F32.
    pub fn matrix_eltwise_bf16(
        &mut self,
        op: crate::matrix_eltwise::MatrixEltwiseOp,
        a: &crate::bf16::Bf16Tensor,
        b: &crate::bf16::Bf16Tensor,
        fidelity: Fidelity,
    ) -> Result<DramTensor, TensorError> {
        use crate::matrix_eltwise as m;
        let dims = [a.rows, a.cols];
        let broadcast = m::broadcast(dims, [b.rows, b.cols])?;
        let units = self.spread_for(dims[0].div_ceil(32) * dims[1].div_ceil(32));
        let simulated = self.dev.transport().is_simulated();
        let work = m::build(
            &mut self.dram_state()?.alloc,
            &a.placement,
            &b.placement,
            dims,
            m::Config {
                packed: true,
                simulated,
                precision: m::SrcPrecision::Bf16,
                fidelity,
                op,
                broadcast,
                tail: None,
            },
            units,
        )?;
        self.execute_on(work, units, RESET_BUDGET)
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
        // The session's math mode picks the program: an Approx twin has a kind
        // of its own, so every memo and cache key below carries the mode.
        let op = op.in_mode(self.math);
        let units = self.spread_for(tiles_of(a).max(b.map_or(0, tiles_of)));
        let overlap = self.overlap();
        let alloc = &mut self.dram_state()?.alloc;
        let (kind, bcast) = tensor::broadcast_of(op, a, b)?;
        // The SFPU's, or refused: the mover only moves data.
        let Some(work) = tensor::sfpu_eltwise(alloc, op, a, b, c, units, overlap)? else {
            return Err(TensorError::Shape(format!(
                "element-wise {kind:#x} with {bcast:?}: no SFPU program computes it"
            )));
        };
        let out = self.execute_on(work, units, RESET_BUDGET)?;
        out.set_pad(match tensor::eltwise_rows(kind, a.rows) {
            // Rows past `a`'s are left as `Dst` held them.
            32 => op.produces(&[Some(a), b, c].into_iter().flatten().collect::<Vec<_>>()),
            _ => tensor::Pad::Undefined,
        });
        Ok(out)
    }

    /// [`Session::eltwise3`] over blocks of resident tensors
    /// ([`tensor::sfpu_eltwise_blocks`]): each operand read where it lies, at
    /// its parent's row stride, the result a new tensor of `a`'s block's dims.
    pub fn eltwise_blocks(
        &mut self,
        op: tensor::Eltwise,
        a: tensor::BlockOperand<'_>,
        b: Option<tensor::BlockOperand<'_>>,
        c: Option<tensor::BlockOperand<'_>>,
    ) -> Result<DramTensor, TensorError> {
        use tensor::OpPadding;
        let op = op.in_mode(self.math);
        let tiles = |x: &tensor::BlockOperand<'_>| x.dims[0].div_ceil(32) * x.dims[1].div_ceil(32);
        let units = self.spread_for(tiles(&a).max(b.as_ref().map_or(0, tiles)));
        let overlap = self.overlap();
        let alloc = &mut self.dram_state()?.alloc;
        let Some(work) = tensor::sfpu_eltwise_blocks(alloc, op, a, b, c, units, overlap)? else {
            return Err(TensorError::Shape(format!(
                "element-wise {:#x} on blocks: no SFPU program computes it",
                op.kind
            )));
        };
        let out = self.execute_on(work, units, RESET_BUDGET)?;
        let shapes: Vec<DramTensor> = [Some(a), b, c]
            .into_iter()
            .flatten()
            .map(|x| x.shape_only())
            .collect();
        let (kind, _) = tensor::broadcast_of(op, &shapes[0], b.is_some().then(|| &shapes[1]))?;
        out.set_pad(match tensor::eltwise_rows(kind, a.dims[0]) {
            // Rows past `a`'s are left as `Dst` held them.
            32 => op.produces(&shapes.iter().collect::<Vec<_>>()),
            _ => tensor::Pad::Undefined,
        });
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
    /// `t` (`[1, K]`, `K` whole tiles) laid out `[K / 32, 32]` ([`tensor::pack_row`]), on every
    /// unit: what a one-row product reads as its packed `A` ([`Session::set_packed_thin_a`]).
    pub fn pack_row(&mut self, t: &DramTensor) -> Result<DramTensor, TensorError> {
        let units = self.units.len();
        let work = tensor::pack_row(&mut self.dram_state()?.alloc, t, units)?;
        self.execute_on(work, units, RESET_BUDGET)
    }

    pub fn copy(&mut self, t: &DramTensor) -> Result<DramTensor, TensorError> {
        let units = self.spread_for(tiles_of(t));
        let work = tensor::copy(&mut self.dram_state()?.alloc, t, units)?;
        let out = self.execute_on(work, units, RESET_BUDGET)?;
        out.set_pad(if t.pad() == tensor::Pad::Zero {
            tensor::Pad::Zero
        } else {
            tensor::Pad::Undefined
        });
        Ok(out)
    }

    /// Copy logical datums bit for bit through the Tensix L1 mover. Output
    /// padding is physically zero. Explicit, single-card operation; row views
    /// are supported. B reads on NoC0 and NC writes on NoC1.
    pub fn copy_xmov(&mut self, source: &DramTensor) -> Result<DramTensor, TensorError> {
        self.movement_xmov(Some(source), [source.rows, source.cols], source.elem)
    }

    /// Create physically zeroed 32-bit storage through the Tensix L1 mover.
    pub fn zeros_xmov(
        &mut self,
        dims: [usize; 2],
        elem: tensor::Elem,
    ) -> Result<DramTensor, TensorError> {
        self.movement_xmov(None, dims, elem)
    }

    fn movement_xmov(
        &mut self,
        source: Option<&DramTensor>,
        dims: [usize; 2],
        elem: tensor::Elem,
    ) -> Result<DramTensor, TensorError> {
        crate::local_movement::geometry(dims, 4)?;
        let out = DramTensor::alloc_elem(&mut self.dram_state()?.alloc, dims[0], dims[1], elem)?;
        let jobs = match crate::local_movement::jobs(
            source.map(|s| &s.placement),
            &out.placement,
            dims,
            4,
        ) {
            Ok(jobs) => jobs,
            Err(error) => {
                self.dram_state()?.alloc.free(&out.placement);
                return Err(error);
            }
        };
        out.set_pad(tensor::Pad::Zero);
        self.execute(tensor::Work { out, jobs }, RESET_BUDGET)
    }

    /// Copy raw BF16 storage bits through Tensix, with physically zero padding.
    pub fn copy_bf16_xmov(
        &mut self,
        source: &crate::bf16::Bf16Tensor,
    ) -> Result<crate::bf16::Bf16Tensor, TensorError> {
        self.movement_bf16_xmov(Some(source), [source.rows, source.cols])
    }

    /// Create raw BF16 positive zero storage through the Tensix L1 mover.
    pub fn zeros_bf16_xmov(
        &mut self,
        dims: [usize; 2],
    ) -> Result<crate::bf16::Bf16Tensor, TensorError> {
        self.movement_bf16_xmov(None, dims)
    }

    fn movement_bf16_xmov(
        &mut self,
        source: Option<&crate::bf16::Bf16Tensor>,
        dims: [usize; 2],
    ) -> Result<crate::bf16::Bf16Tensor, TensorError> {
        let tiles = crate::local_movement::geometry(dims, 2)?;
        let out = crate::bf16::Bf16Tensor {
            rows: dims[0],
            cols: dims[1],
            placement: self
                .dram_state()?
                .alloc
                .alloc_slots(tiles, tt_isa::dm::BF16_TILE_SLOT)?,
        };
        let result =
            crate::local_movement::jobs(source.map(|s| &s.placement), &out.placement, dims, 2)
                .and_then(|jobs| self.submit_jobs(jobs, RESET_BUDGET));
        if let Err(error) = result {
            self.dram_state()?.alloc.free(&out.placement);
            return Err(error);
        }
        Ok(out)
    }

    /// Copy an F32 rectangle through unpacker XY counters and Dst, bit for bit.
    /// Column origin and width must be multiples of sixteen; row boundaries
    /// may be ragged. Only logical datums are read; output padding is zero.
    pub fn copy_rect_adc(
        &mut self,
        source: &DramTensor,
        origin: [usize; 2],
        dims: [usize; 2],
    ) -> Result<DramTensor, TensorError> {
        let work = crate::adc_copy::build(&mut self.dram_state()?.alloc, source, origin, dims)?;
        self.execute(work, RESET_BUDGET)
    }

    /// Copy complete F32 Y/X planes from a W/Z rectangle, preserving raw bits.
    /// Source storage is `[W * Z * Y, X]`; output is `[dims[0] * dims[1] * Y, X]`.
    /// Reads only logical datums and produces zero padding, including ragged planes.
    pub fn copy_planes_adc(
        &mut self,
        source: &DramTensor,
        shape: [usize; 4],
        origin: [usize; 2],
        dims: [usize; 2],
    ) -> Result<DramTensor, TensorError> {
        let work = crate::adc_copy::build_planes(
            &mut self.dram_state()?.alloc,
            source,
            shape,
            origin,
            dims,
        )?;
        self.execute(work, RESET_BUDGET)
    }

    /// Copy `src`, bit for bit, into an existing allocated tensor `dst`.
    pub fn copy_into(&mut self, src: &DramTensor, dst: &DramTensor) -> Result<(), TensorError> {
        let units = self.spread_for(tiles_of(src));
        let work = tensor::copy_into(src, dst, units)?;
        self.submit_jobs_on(work.jobs, units, RESET_BUDGET)?;
        dst.set_pad(if src.pad() == tensor::Pad::Zero {
            tensor::Pad::Zero
        } else {
            tensor::Pad::Undefined
        });
        Ok(())
    }

    /// Preserve BFP exponent groups and packed bits in an existing allocation.
    pub fn copy_into_bfp(
        &mut self,
        src: &crate::bfp::BfpTensor,
        dst: &crate::bfp::BfpTensor,
    ) -> Result<(), TensorError> {
        let jobs = crate::bfp::copy_into(src, dst)?;
        self.submit_jobs(jobs, RESET_BUDGET)
    }

    /// Copy logical BF16 payloads into an existing allocation, without conversion.
    /// Parameter-update traces use this to retain the original buffer identity.
    pub fn copy_into_bf16(
        &mut self,
        src: &crate::bf16::Bf16Tensor,
        dst: &crate::bf16::Bf16Tensor,
    ) -> Result<(), TensorError> {
        if [src.rows, src.cols] != [dst.rows, dst.cols] {
            return Err(TensorError::Shape("BF16 copy_into shapes differ".into()));
        }
        let count = src
            .rows
            .checked_mul(src.cols)
            .ok_or_else(|| TensorError::Shape("BF16 copy_into shape overflow".into()))?;
        let sources: Vec<_> = (0..count)
            .map(|i| Some([i / src.cols, i % src.cols]))
            .collect();
        self.repack_bf16_padded_into(src, &sources, dst, 0)
    }

    /// Repack logical source coordinates into a new matrix on the card.
    /// Preserves bits and leaves ragged output padding undefined.
    pub fn repack(
        &mut self,
        t: &DramTensor,
        sources: &[[usize; 2]],
        dims: [usize; 2],
    ) -> Result<DramTensor, TensorError> {
        let work = tensor::repack(&mut self.dram_state()?.alloc, t, sources, dims)?;
        let out = self.execute(work, RESET_BUDGET)?;
        out.set_pad(tensor::Pad::Undefined);
        Ok(out)
    }

    /// Copy raw datums from multiple resident inputs using logical geometry.
    pub fn repack_many(
        &mut self,
        inputs: &[&DramTensor],
        sources: &[(usize, [usize; 2])],
        dims: [usize; 2],
    ) -> Result<DramTensor, TensorError> {
        let work = tensor::repack_many(&mut self.dram_state()?.alloc, inputs, sources, dims)?;
        let out = self.execute(work, RESET_BUDGET)?;
        out.set_pad(tensor::Pad::Undefined);
        Ok(out)
    }

    /// Initialize physically packed BF16 zeros using replayable native fills.
    pub fn zeros_bf16(&mut self, dims: [usize; 2]) -> Result<crate::bf16::Bf16Tensor, TensorError> {
        let (out, jobs) = crate::bf16::zeros(&mut self.dram_state()?.alloc, dims)?;
        if let Err(error) = self.submit_jobs(jobs, RESET_BUDGET) {
            self.dram_state()?.alloc.free(&out.placement);
            return Err(error);
        }
        Ok(out)
    }

    /// Raw resident selection from a matrix with at most 32 logical rows.
    pub fn gather_indexed(
        &mut self,
        input: &DramTensor,
        indices: &DramTensor,
    ) -> Result<DramTensor, TensorError> {
        let out = DramTensor::alloc_elem(&mut self.dram_state()?.alloc, input.rows, 1, input.elem)?;
        let result = crate::index::jobs(
            &input.placement,
            indices,
            &out.placement,
            [input.rows, input.cols],
            4,
        )
        .and_then(|jobs| self.submit_jobs(jobs, RESET_BUDGET));
        if let Err(error) = result {
            self.dram_state()?.alloc.free(&out.placement);
            return Err(error);
        }
        out.set_pad(tensor::Pad::Zero);
        Ok(out)
    }

    /// Rows of `table` chosen by `indices`, a resident `[1, n]` I32 tensor:
    /// `out[i] = table[indices[i]]`, bit for bit, with no host address
    /// ([`crate::index::select_rows_jobs`]). An index outside `0..table.rows`
    /// is `DOMAIN` from the mover, queued as any list's error. The padding is
    /// zero where the table's is: the rows past `n` are filled with zero.
    pub fn select_rows(
        &mut self,
        table: &DramTensor,
        indices: &DramTensor,
    ) -> Result<DramTensor, TensorError> {
        let units = self.spread_for(indices.cols.div_ceil(32) * table.cols.div_ceil(32));
        let out = DramTensor::alloc_elem(
            &mut self.dram_state()?.alloc,
            indices.cols,
            table.cols,
            table.elem,
        )?;
        let result = crate::index::select_rows_jobs(table, indices, &out, units)
            .and_then(|jobs| self.submit_jobs_on(jobs, units, RESET_BUDGET));
        if let Err(error) = result {
            self.dram_state()?.alloc.free(&out.placement);
            return Err(error);
        }
        out.set_pad(if table.pad() == tensor::Pad::Zero {
            tensor::Pad::Zero
        } else {
            tensor::Pad::Undefined
        });
        Ok(out)
    }

    /// Rows of `table` for several taps side by side: `indices` a resident
    /// `[taps, n]` I32 tensor, the result `[n, select_taps_cols(table.cols,
    /// taps)]` with row `i` holding row `indices[t][i]` of `table` for each
    /// tap `t` from tile column `t x ceil(table.cols / 32)`
    /// ([`crate::index::select_taps_jobs`]) -- an im2col with no host
    /// address when `table` is an NHWC activation.
    ///
    /// With `zero_row`, an index may also be `table.rows`, which reads a row
    /// of zeros: what a convolution's padding taps read. It is the table's
    /// first padding row, made zero; a table with no padding row (whole tile
    /// rows) is first copied, by whole tiles, into a zeroed one a row longer.
    /// Every tap but the last takes the table's padding columns into its own,
    /// so they are made zero too whenever `taps > 1` (a matmul's `K` runs over
    /// them). Bit for bit; an index past the bound is `DOMAIN` from the mover.
    pub fn select_taps(
        &mut self,
        table: &DramTensor,
        indices: &DramTensor,
        zero_row: bool,
    ) -> Result<DramTensor, TensorError> {
        let longer = if zero_row && table.rows % 32 == 0 {
            let out = DramTensor::alloc_elem(
                &mut self.dram_state()?.alloc,
                table.rows + 1,
                table.cols,
                table.elem,
            )?;
            let copied = out
                .rows_view(0, table.rows)
                .and_then(|rows| self.copy_into(table, &rows))
                .and_then(|()| crate::index::zero_tile_row_jobs(&out, table.rows / 32))
                .and_then(|jobs| self.submit_jobs(jobs, RESET_BUDGET));
            if let Err(e) = copied {
                let _ = self.free(out);
                return Err(e);
            }
            // The copy brought the table's padding columns.
            out.set_pad(table.pad());
            Some(out)
        } else {
            None
        };
        let source = longer.as_ref().unwrap_or(table);
        let need = if zero_row || indices.rows > 1 {
            tensor::PadNeed::Zero
        } else {
            tensor::PadNeed::Any
        };
        let result = (|| {
            let repaired = self.meet(source, need)?;
            let used = repaired.as_ref().unwrap_or(source);
            let cols = crate::index::select_taps_cols(used.cols, indices.rows.max(1));
            let units = self.spread_for(indices.cols.div_ceil(32) * cols.div_ceil(32));
            let made = (|| {
                let out = DramTensor::alloc_elem(
                    &mut self.dram_state()?.alloc,
                    indices.cols,
                    cols,
                    used.elem,
                )?;
                let bound = table.rows + usize::from(zero_row);
                let result = crate::index::select_taps_jobs(used, indices, bound, &out, units)
                    .and_then(|jobs| self.submit_jobs_on(jobs, units, RESET_BUDGET));
                if let Err(error) = result {
                    self.dram_state()?.alloc.free(&out.placement);
                    return Err(error);
                }
                out.set_pad(if used.pad() == tensor::Pad::Zero {
                    tensor::Pad::Zero
                } else {
                    tensor::Pad::Undefined
                });
                Ok(out)
            })();
            if let Some(copy) = repaired {
                let _ = self.free(copy);
            }
            made
        })();
        if let Some(copy) = longer {
            let _ = self.free(copy);
        }
        result
    }

    pub fn gather_indexed_bf16(
        &mut self,
        input: &crate::bf16::Bf16Tensor,
        indices: &DramTensor,
    ) -> Result<crate::bf16::Bf16Tensor, TensorError> {
        let out = crate::bf16::Bf16Tensor {
            rows: input.rows,
            cols: 1,
            placement: self
                .dram_state()?
                .alloc
                .alloc_slots(input.rows.div_ceil(32), tt_isa::dm::BF16_TILE_SLOT)?,
        };
        let result = crate::index::jobs(
            &input.placement,
            indices,
            &out.placement,
            [input.rows, input.cols],
            2,
        )
        .and_then(|jobs| self.submit_jobs(jobs, RESET_BUDGET));
        if let Err(error) = result {
            self.dram_state()?.alloc.free(&out.placement);
            return Err(error);
        }
        Ok(out)
    }

    pub fn repack_many_bf16(
        &mut self,
        inputs: &[&crate::bf16::Bf16Tensor],
        sources: &[(usize, [usize; 2])],
        dims: [usize; 2],
    ) -> Result<crate::bf16::Bf16Tensor, TensorError> {
        let (out, jobs) =
            crate::bf16::repack_many(&mut self.dram_state()?.alloc, inputs, sources, dims)?;
        if let Err(error) = self.submit_jobs(jobs, RESET_BUDGET) {
            self.dram_state()?.alloc.free(&out.placement);
            return Err(error);
        }
        Ok(out)
    }

    /// A new `dims` tensor assembled from blocks of `t`
    /// ([`tensor::copy_blocks`]): a tile-moving reshape or permute, on the
    /// card. Its padding is `t`'s: a block ragged at all is ragged at both
    /// tensors' edges.
    pub fn copy_blocks(
        &mut self,
        t: &DramTensor,
        moves: &[tensor::BlockMove],
        dims: [usize; 2],
    ) -> Result<DramTensor, TensorError> {
        let units = self.spread_for(dims[0].div_ceil(32) * dims[1].div_ceil(32));
        let work = tensor::copy_blocks(&mut self.dram_state()?.alloc, t, moves, dims, units)?;
        let out = self.execute_on(work, units, RESET_BUDGET)?;
        out.set_pad(if t.pad() == tensor::Pad::Zero {
            tensor::Pad::Zero
        } else {
            tensor::Pad::Undefined
        });
        Ok(out)
    }

    /// Rows of `sources` gathered into a new tensor ([`tensor::gather_rows`]):
    /// an embedding's lookup on the card. Its padding is zero only where every
    /// source's is and the rows fill whole tiles.
    pub fn gather_rows(
        &mut self,
        sources: &[&DramTensor],
        rows: &[(usize, usize)],
        cols: usize,
    ) -> Result<DramTensor, TensorError> {
        let units = self.spread_for(rows.len().div_ceil(32) * cols.div_ceil(32));
        let work = tensor::gather_rows(&mut self.dram_state()?.alloc, sources, rows, cols, units)?;
        let out = self.execute_on(work, units, RESET_BUDGET)?;
        let zero = rows.len() % 32 == 0
            && sources.iter().all(|t| t.pad() == tensor::Pad::Zero)
            && rows.iter().all(|&(k, _)| k != tensor::GATHER_HOLE);
        out.set_pad(if zero {
            tensor::Pad::Zero
        } else {
            tensor::Pad::Undefined
        });
        Ok(out)
    }

    /// Rows of `src` over rows of `dst`, in place ([`tensor::write_rows`]).
    pub fn write_rows(
        &mut self,
        dst: &DramTensor,
        src: &DramTensor,
        rows: &[(usize, usize)],
    ) -> Result<(), TensorError> {
        let units = self.spread_for(tiles_of(src));
        let jobs = tensor::write_rows(dst, src, rows, units)?;
        self.submit_jobs_on(jobs, units, RESET_BUDGET)?;
        if src.pad() != tensor::Pad::Zero {
            dst.set_pad(tensor::Pad::Undefined);
        }
        Ok(())
    }

    /// A copy of `t` with rows of `value` written over some of its rows: `(d, s)`
    /// puts `value`'s row `s` at row `d` ([`tensor::write_rows`] over a tile copy,
    /// [`Session::copy`]). Burn's `slice_assign` of whole rows, a cache write:
    /// the host builds a few run records for the copy and entries for the
    /// written rows only, and `t` is not written.
    pub fn assign_rows(
        &mut self,
        t: &DramTensor,
        value: &DramTensor,
        rows: &[(usize, usize)],
    ) -> Result<DramTensor, TensorError> {
        let out = self.copy(t)?;
        if let Err(e) = self.write_rows(&out, value, rows) {
            self.free(out)?;
            return Err(e);
        }
        Ok(out)
    }

    /// `values`' row `i` written over `dst`'s row `indices[i]`, in place, the
    /// indices a resident I32 tensor, `[1, n]` or (`column`) `[n, 1]`
    /// ([`crate::index::scatter_rows_jobs`]): no host address, so a trace
    /// replays it with the indices it finds. A repeated index's last row is
    /// written; one outside `0..dst.rows` is `DOMAIN` from the mover, queued as
    /// any list's error.
    pub fn scatter_rows_into(
        &mut self,
        dst: &DramTensor,
        indices: &DramTensor,
        values: &DramTensor,
        column: bool,
    ) -> Result<(), TensorError> {
        let n = if column { indices.rows } else { indices.cols };
        let units = self.spread_for(crate::index::scatter_rows_work(dst, n, tiles_of(values)));
        let jobs = crate::index::scatter_rows_jobs(dst, indices, values, column, units)?;
        self.submit_jobs_on(jobs, units, RESET_BUDGET)?;
        if values.pad() != tensor::Pad::Zero {
            dst.set_pad(tensor::Pad::Undefined);
        }
        Ok(())
    }

    /// A copy of `table` with `values`' row `i` at its row `indices[i]`
    /// ([`Session::scatter_rows_into`] over a tile copy, [`Session::copy`]):
    /// Burn's `scatter_nd` with `Assign` over whole rows, a cache write at a
    /// position the card holds. `table` is not written.
    pub fn scatter_rows(
        &mut self,
        table: &DramTensor,
        indices: &DramTensor,
        values: &DramTensor,
        column: bool,
    ) -> Result<DramTensor, TensorError> {
        let out = self.copy(table)?;
        if let Err(e) = self.scatter_rows_into(&out, indices, values, column) {
            self.free(out)?;
            return Err(e);
        }
        Ok(out)
    }

    /// `t` with `value`'s row `i` added to its row `indices[i]`, for every `i`
    /// in order -- Burn's `select_add` along rows, an embedding's gradient --
    /// on the card, in a new tensor. Only the rows the indices touch are
    /// computed: they are gathered ([`tensor::gather_rows`]); round `r` adds
    /// each one's `r`-th occurrence in `value` (or a `-0` row, which adds
    /// nothing, where it has fewer), so each row's additions are made in
    /// `indices` order, as Flex makes them; and the rows are written over a
    /// tile copy of `t` ([`tensor::write_rows`]). The work grows with the
    /// indices and the most any row repeats, not with `t`'s rows beyond one
    /// copy.
    pub fn rows_add(
        &mut self,
        t: &DramTensor,
        indices: &[usize],
        value: &DramTensor,
    ) -> Result<DramTensor, TensorError> {
        let cols = t.cols;
        if value.rows != indices.len() || value.cols != cols {
            return Err(TensorError::Shape(format!(
                "{} indices and a [{}, {}] value for a [{}, {cols}] tensor",
                indices.len(),
                value.rows,
                value.cols,
                t.rows
            )));
        }
        if let Some(&i) = indices.iter().find(|&&i| i >= t.rows) {
            return Err(TensorError::Shape(format!(
                "index {i} into a tensor of {} rows",
                t.rows
            )));
        }
        // The touched rows, in order of first use, and each one's occurrences.
        let mut slot_of = std::collections::HashMap::new();
        let mut occurrences: Vec<Vec<usize>> = Vec::new();
        let mut touched = Vec::new();
        for (i, &r) in indices.iter().enumerate() {
            let k = *slot_of.entry(r).or_insert_with(|| {
                touched.push(r);
                occurrences.push(Vec::new());
                touched.len() - 1
            });
            occurrences[k].push(i);
        }
        let rounds = occurrences.iter().map(Vec::len).max().unwrap_or(0);
        let zero = self.upload(&vec![-0.0; cols], 1, cols)?;
        let rows: Vec<(usize, usize)> = touched.iter().map(|&r| (0, r)).collect();
        let mut acc = self.gather_rows(&[t], &rows, cols)?;
        let add = tensor::Eltwise {
            kind: crate::kind::ADD,
            scalar: 0.0,
            scalar2: 0.0,
        };
        for round in 0..rounds {
            let rows: Vec<(usize, usize)> = occurrences
                .iter()
                .map(|o| o.get(round).map_or((1, 0), |&i| (0, i)))
                .collect();
            let addend = self.gather_rows(&[value, &zero], &rows, cols)?;
            let next = self.eltwise(add, &acc, Some(&addend))?;
            self.free(addend)?;
            self.free(std::mem::replace(&mut acc, next))?;
        }
        self.free(zero)?;
        let out = self.copy(t)?;
        let rows: Vec<(usize, usize)> = touched.iter().enumerate().map(|(k, &r)| (r, k)).collect();
        let written = self.write_rows(&out, &acc, &rows);
        self.free(acc)?;
        written?;
        Ok(out)
    }

    /// The sum over rows of `a`, `[1, cols]`, in `burn-flex`'s order
    /// ([`tensor::sum_rows`]).
    pub fn sum_rows(&mut self, a: &DramTensor) -> Result<DramTensor, TensorError> {
        use tensor::OpPadding;
        let units = self.spread_for(tiles_of(a));
        let overlap = self.overlap();
        let work = tensor::sum_rows(&mut self.dram_state()?.alloc, a, units, overlap)?;
        let out = self.execute_on(work, units, RESET_BUDGET)?;
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
        self.reduce_then(a, op, axis, None)
    }

    /// [`Session::reduce`] with `then`'s exact unary kind applied to the result before its pack
    /// ([`tensor::sfpu_reduce_then`]): one op where there were two, bit for bit. Refused for a
    /// sum over rows (Flex's order, its own kernel) and a reduction too long for one tile.
    pub fn reduce_then(
        &mut self,
        a: &DramTensor,
        op: crate::sfpu::reduce::ReduceOp,
        axis: crate::sfpu::reduce::Axis,
        then: tensor::ReduceThen,
    ) -> Result<DramTensor, TensorError> {
        use crate::sfpu::reduce::{Axis, ReduceOp};
        if (op, axis) == (ReduceOp::Sum, Axis::Rows) {
            if then.is_some() {
                return Err(TensorError::Shape(
                    "a sum over rows takes no finishing kind".into(),
                ));
            }
            return self.sum_rows(a);
        }
        let units = self.spread_for(tiles_of(a));
        let overlap = self.overlap();
        let alloc = &mut self.dram_state()?.alloc;
        let work = tensor::sfpu_reduce_then(alloc, a, op, axis, units, overlap, then)?;
        let out = self.execute_on(work, units, RESET_BUDGET)?;
        out.set_pad(tensor::Pad::Undefined);
        Ok(out)
    }

    /// Each row's `sqrt(mean(a a) + eps)` over its columns, `[rows, 1]`: RMSNorm's statistic,
    /// one job a tile row (`tensor::sfpu_rms`), bit for bit the ops it replaces.
    pub fn rms(&mut self, a: &DramTensor, eps: f32) -> Result<DramTensor, TensorError> {
        let kt = a.cols / 32;
        if self.packed_rms
            && a.rows == 1
            && a.cols % 32 == 0
            && kt >= PACKED_A_K_TILES
            && a.elem == tensor::Elem::F32
        {
            // One unit reading the row's `K / 32` tiles is bound by its entries (~0.33 us
            // each): the packing spreads them over the units, the statistic reads `K / 1024`.
            let units = self.spread_for(kt);
            let work = tensor::pack_row(&mut self.dram_state()?.alloc, a, units)?;
            let packed = self.execute_on(work, units, RESET_BUDGET)?;
            let stat = self
                .dram_state()
                .and_then(|d| tensor::sfpu_rms_packed(&mut d.alloc, &packed, a.cols, eps))
                .and_then(|work| self.execute_on(work, 1, RESET_BUDGET));
            self.free(packed)?;
            let out = stat?;
            out.set_pad(tensor::Pad::Undefined);
            return Ok(out);
        }
        let units = self.spread_for(a.grid()[0]);
        let work = tensor::sfpu_rms(&mut self.dram_state()?.alloc, a, eps)?;
        let out = self.execute_on(work, units, RESET_BUDGET)?;
        out.set_pad(tensor::Pad::Undefined);
        Ok(out)
    }

    /// Inclusive scan down rows, carrying each prefix between tiles: FP32
    /// sum/product, raw-total-order or Flex-order (NaN-propagating, IEEE)
    /// min/max -- these preserve the selected datum's bits -- or the wrapping
    /// and signed I32 scans, which take and give an I32 tensor
    /// ([`crate::sfpu::scan::ScanOp::elem`]).
    pub fn scan(
        &mut self,
        a: &DramTensor,
        op: crate::sfpu::scan::ScanOp,
    ) -> Result<DramTensor, TensorError> {
        let work = tensor::sfpu_scan(&mut self.dram_state()?.alloc, a, op)?;
        let out = self.execute(work, RESET_BUDGET)?;
        out.set_pad(tensor::Pad::Undefined);
        Ok(out)
    }

    /// Opt-in hardware precision reduction. Stochastic state is inherited
    /// from each executing core, so replay advances it. See the functional
    /// model for documented non-IEEE zero/NaN and rounding behavior.
    pub fn hardware_round(
        &mut self,
        a: &DramTensor,
        precision: tt_isa::numerics::stochastic::Precision,
        rounding: tt_isa::numerics::stochastic::Rounding,
    ) -> Result<DramTensor, TensorError> {
        use tt_isa::numerics::stochastic::Precision;
        let offset = if precision == Precision::Bf16 { 0 } else { 3 };
        self.eltwise(
            tensor::Eltwise {
                kind: crate::sfpu::ops::kind_sfpu::HARDWARE_ROUND + offset + rounding as u32,
                scalar: 0.0,
                scalar2: 0.0,
            },
            a,
            None,
        )
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
        let units = self.spread_for(tiles_of(t));
        if t.placement.owned() {
            let jobs = tensor::fill_pad(t, 0.0, units)?;
            self.submit_jobs_on(jobs, units, RESET_BUDGET)?;
            t.set_pad(Pad::Zero);
            return Ok(None);
        }
        let copy = self.copy(t)?;
        let jobs = tensor::fill_pad(&copy, 0.0, units)?;
        if let Err(e) = self.submit_jobs_on(jobs, units, RESET_BUDGET) {
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
        // A single-face TF32 operand has a validated Src transpose route.
        // Conversion is already required by this product; arbitrary storage
        // transpose and physically packed BF16 remain raw-copy operations.
        if b_transposed && [b.rows, b.cols] == [16, 16] && route == SrcRoute::Tf32FromFp32 {
            let prepared = self.transpose_src_block(b, route)?;
            let result =
                self.matmul_dram(a, a_transposed, &prepared, false, route, fidelity, budget);
            let _ = self.free(prepared);
            return result;
        }
        let need = tensor::MatmulPadding;
        // Only padding along `K` reaches a real datum: an output row reads its row
        // of `op(A)` alone, an output column its column of `op(B)`. An operand
        // whose `K` is whole tiles has no padding there, so it is used as it is
        // (no refill: SmolLM2's projections refilled a `[1, 576]` input's 31
        // padding rows every time), and what it holds past `M` or `N` lands in
        // the output's padding ([`tensor::matmul_padding_off_k`]).
        let off_k = tensor::matmul_padding_off_k(a, a_transposed, b, b_transposed);
        let needs = |i: usize| match off_k[i] {
            true => tensor::PadNeed::Any,
            false => need.requires(i),
        };
        let ca = self.meet(a, needs(0))?;
        let cb = match self.meet(b, needs(1)) {
            Ok(c) => c,
            Err(e) => {
                if let Some(c) = ca {
                    let _ = self.free(c);
                }
                return Err(e);
            }
        };
        let [ar, ac] = a.grid();
        let [br, bc] = b.grid();
        let (mt, kt) = if a_transposed { (ac, ar) } else { (ar, ac) };
        let nt = if b_transposed { br } else { bc };
        let units = self.spread_for(mt * nt * kt);
        // A one-tile-row product over a long `K` streams its weights: GDDR is saturated by
        // `THIN_STREAM_UNITS` units, and more only share it while each pays its own `A` reads and
        // the deal's last round (`silicon_probe_decode_stream`).
        let units = if self.thin_stream_units && mt == 1 && kt >= THIN_STREAM_K_TILES {
            units.min(THIN_STREAM_UNITS)
        } else {
            units
        };
        // No `MOP` in the matmul: with its loops replayed it is backend-bound,
        // and the math role's `MOP` measured no faster end to end on silicon
        // (row AE) -- while ttsim's FIFO overflows under one (row 68), so the
        // default would be a path only silicon runs. `step36_mop` and
        // `step37_loops` keep the expander gated.
        let allow_mop = false;
        let pipeline = self.pipeline;
        let max_k_tiles = self.matmul_k_block_limit;
        let m = if a_transposed { a.cols } else { a.rows };
        let thin = self.thin_matmul_reads && tensor::thin_rows(m, a_transposed) != 0;
        let resident_a = self.resident_matmul_a;
        let unpack_b = self.unpack_transposed_b;
        // One row over a long `K`: every unit would read `A`'s `K / 32` tiles, an entry each,
        // so it is packed `[K / 32, 32]` once, spread over the product's units, and each unit
        // reads `K / 1024`.
        let packed = if self.packed_thin_a
            && m == 1
            && !a_transposed
            && a.cols % 32 == 0
            && kt >= PACKED_A_K_TILES
            && route == SrcRoute::Tf32FromFp32
        {
            // On the product's own units: the rest stay out of the op (a unit that sits out ops
            // is free for other work, `decode-to-bandwidth.md` Phase 5).
            let packing = self
                .dram_state()
                .and_then(|d| tensor::pack_row(&mut d.alloc, ca.as_ref().unwrap_or(a), units))
                .and_then(|work| self.execute_on(work, units, budget));
            match packing {
                Ok(p) => Some(p),
                Err(e) => {
                    for c in [ca, cb].into_iter().flatten() {
                        let _ = self.free(c);
                    }
                    return Err(e);
                }
            }
        } else {
            None
        };
        let packed_a = packed.as_ref().map(|p| p.tensor_ref());
        let stream_k = self.streamed_k;
        // In a capture, the units past the product's own that ran nothing since the last
        // weight-streaming product: each can read a job's `B` while the ops between run.
        let spread = self.units.len();
        let (prefetch, anchor): (Vec<usize>, Option<u32>) = match &self.capture {
            Some(c) if self.weight_prefetch && packed_a.is_some() => match &c.weight_anchor {
                Some((rel, lens)) => (
                    (units..spread)
                        .filter(|&u| c.units[u].stream.len() == lens[u])
                        .take(PREFETCH_UNITS)
                        .collect(),
                    Some(*rel),
                ),
                None => (Vec::new(), None),
            },
            _ => (Vec::new(), None),
        };
        let out = self
            .dram_state()
            .and_then(|d| {
                tensor::matmul_dram_prefetching(
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
                    max_k_tiles,
                    tensor::Reads {
                        thin,
                        resident_a,
                        unpack_b,
                        packed_a,
                        stream_k,
                    },
                    &prefetch,
                    spread,
                )
            })
            .and_then(|(work, dealt)| {
                self.prefetch_anchor = dealt.and(anchor);
                let r = self.execute_on(work, dealt.unwrap_or(units), budget);
                self.prefetch_anchor = None;
                r
            });
        // This product is the next one's anchor: a unit idle after it can prefetch for that one.
        if out.is_ok() && packed_a.is_some() {
            let rel = self.barriers;
            if let Some(c) = self.capture.as_mut() {
                let rel = rel.wrapping_sub(c.barriers_base);
                c.weight_anchor = Some((rel, c.units.iter().map(|u| u.stream.len()).collect()));
            }
        }
        for c in [ca, cb, packed].into_iter().flatten() {
            self.free(c)?;
        }
        let out = out?;
        // An operand used with its padding off `K` as it was leaves that padding's
        // image in the output's; a thin read of `A` leaves whatever its L1 slots
        // held past the rows it read.
        let dirty = (off_k[0] && a.pad() != tensor::Pad::Zero)
            || (off_k[1] && b.pad() != tensor::Pad::Zero)
            || thin
            || packed_a.is_some();
        out.set_pad(match dirty {
            true => tensor::Pad::Undefined,
            false => need.produces(&[a, b]),
        });
        Ok(out)
    }

    /// `op(A_i) @ op(B_i)` for each pair of blocks in `items`, every product
    /// `[m, k] @ [k, n]`, into one `[items.len() m, n]` tensor in GDDR
    /// ([`tensor::matmul_dram_batched`]): a batched matmul whose operands are
    /// resident tensors or blocks of them, each product [`Session::matmul_dram`]'s
    /// bit for bit.
    ///
    /// A block is read where it lies when it starts on a tile and ends on one
    /// or at its tensor's edge ([`tensor::block_fits`]). An operand with a block
    /// that does not, but whose blocks are each whole rows of it (attention's
    /// heads stacked in a `[heads s, d]` buffer, `s` not whole tiles), is first
    /// gathered by rows into a block per whole tile rows with zero tail rows --
    /// one row gather, not a copy a block. When `m` is not whole tiles each
    /// product's output starts on a tile row of its own, and one row gather
    /// stacks the products' rows.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_dram_batched(
        &mut self,
        a: &DramTensor,
        b: &DramTensor,
        items: &[(tensor::Block, tensor::Block)],
        mkn: [usize; 3],
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<DramTensor, TensorError> {
        self.matmul_dram_batched_with(
            a,
            b,
            items,
            mkn,
            route,
            fidelity,
            budget,
            BatchedLayout::STACKED,
        )
    }

    /// [`Session::matmul_dram_batched`] with the left operand, the output, or
    /// both laid out a product per whole tile rows ([`BatchedLayout`]): a
    /// composition that keeps its products' rows apart between two batched
    /// products (attention's scores, from `Q K^T` to `P V`) skips the stacking
    /// gather after the first and the operand gather before the second.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_dram_batched_with(
        &mut self,
        a: &DramTensor,
        b: &DramTensor,
        items: &[(tensor::Block, tensor::Block)],
        mkn: [usize; 3],
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
        layout: BatchedLayout,
    ) -> Result<DramTensor, TensorError> {
        use tensor::OpPadding;
        a.expect("a matmul", tensor::Elem::F32)?;
        b.expect("a matmul", tensor::Elem::F32)?;
        // A block that reaches its tensor's ragged edge reads the tensor's padding: as `K`'s
        // when the edge is along `K`, which must then be zero, the fill `matmul_dram` asks
        // for. Along `M` or `N` it reaches only the output's padding (or rows a stacked output
        // leaves out), so an operand whose stored `K` dimension is whole tiles is used as it is
        // (`tensor::matmul_padding_off_k`), and the output's padding is then undefined.
        let need = tensor::MatmulPadding;
        let k_ragged = |t: &DramTensor, side: usize| {
            items.iter().any(|(ba, bb)| {
                let transposed = if side == 0 {
                    ba.transposed
                } else {
                    bb.transposed
                };
                // `K` is `op(A)`'s columns and `op(B)`'s rows.
                let k_is_cols = (side == 0) != transposed;
                (if k_is_cols { t.cols } else { t.rows }) % 32 != 0
            })
        };
        let needs = |t: &DramTensor, side: usize| match k_ragged(t, side) {
            true => need.requires(side),
            false => tensor::PadNeed::Any,
        };
        let dirty = (!k_ragged(a, 0) && a.pad() != tensor::Pad::Zero)
            || (!k_ragged(b, 1) && b.pad() != tensor::Pad::Zero);
        let mut temporary = Vec::new();
        let result = (|| {
            let ca = self.meet(a, needs(a, 0))?;
            let a = match ca {
                Some(c) => {
                    temporary.push(c);
                    temporary.last().unwrap().clone()
                }
                None => a.clone(),
            };
            let cb = self.meet(b, needs(b, 1))?;
            let b = match cb {
                Some(c) => {
                    temporary.push(c);
                    temporary.last().unwrap().clone()
                }
                None => b.clone(),
            };
            let [m, k, n] = mkn;
            let extent_a = |t: bool| if t { [k, m] } else { [m, k] };
            let extent_b = |t: bool| if t { [n, k] } else { [k, n] };
            let mut items = items.to_vec();
            let mut padded = [false; 2];
            let mut operands = [a, b];
            if layout.lhs_padded {
                // Each block starts on a tile row of `a` and owns the rows to the next
                // block: its tail rows only make output rows nobody reads.
                padded[0] = true;
            }
            for side in usize::from(layout.lhs_padded)..2 {
                let t = &operands[side];
                let extent = |block: &tensor::Block| {
                    if side == 0 {
                        extent_a(block.transposed)
                    } else {
                        extent_b(block.transposed)
                    }
                };
                let pick = |item: &(tensor::Block, tensor::Block)| {
                    if side == 0 {
                        item.0
                    } else {
                        item.1
                    }
                };
                if items
                    .iter()
                    .all(|i| tensor::block_fits(t, pick(i).at, extent(&pick(i))))
                {
                    continue;
                }
                // An untransposed left block's tail rows are `M` rows: they make only output rows
                // past each product's `m`, which a stacked output leaves out. Anything else
                // (a `K` row, an output kept padded) is read: zero.
                let zero_tails =
                    !(side == 0 && layout.stacked && items.iter().all(|i| !i.0.transposed));
                let Some((p, blocks)) = self.rows_padded(
                    t,
                    &items.iter().map(pick).collect::<Vec<_>>(),
                    extent,
                    zero_tails,
                )?
                else {
                    // Not whole rows: the product refuses the block by name.
                    continue;
                };
                for (item, block) in items.iter_mut().zip(blocks) {
                    *(if side == 0 { &mut item.0 } else { &mut item.1 }) = block;
                }
                temporary.push(p.clone());
                operands[side] = p;
                padded[side] = true;
            }
            let [a, b] = operands;
            let units =
                self.spread_for(items.len() * m.div_ceil(32) * n.div_ceil(32) * k.div_ceil(32));
            let pipeline = self.pipeline;
            let max_k_tiles = self.matmul_k_block_limit;
            let unpack_b = self.unpack_transposed_b;
            let out = self
                .dram_state()
                .and_then(|d| {
                    tensor::matmul_dram_batched_limited(
                        &mut d.alloc,
                        &a,
                        &b,
                        &items,
                        mkn,
                        route,
                        fidelity,
                        units,
                        false,
                        pipeline,
                        max_k_tiles,
                        padded,
                        unpack_b,
                    )
                })
                .and_then(|work| self.execute_on(work, units, budget))?;
            out.set_pad(match dirty {
                true => tensor::Pad::Undefined,
                false => need.produces(&[&a, &b]),
            });
            if !layout.stacked || items.len() == 1 || m % 32 == 0 {
                return Ok(out);
            }
            // Each product's rows from its own tile row: stack them.
            let stride = m.div_ceil(32) * 32;
            let rows: Vec<_> = (0..items.len())
                .flat_map(|i| (0..m).map(move |r| (0, i * stride + r)))
                .collect();
            let stacked = self.gather_rows(&[&out], &rows, n);
            temporary.push(out);
            stacked
        })();
        for t in temporary {
            self.free(t)?;
        }
        result
    }

    /// The blocks of `t` gathered by rows into a new tensor, a block per whole
    /// tile rows with zero tail rows, and the blocks' places in it -- if each
    /// block (of `extent(block)` elements, read transposed or not) is whole rows
    /// of `t`. Blocks at the same place share theirs. `None` otherwise.
    fn rows_padded(
        &mut self,
        t: &DramTensor,
        blocks: &[tensor::Block],
        extent: impl Fn(&tensor::Block) -> [usize; 2],
        zero_tails: bool,
    ) -> Result<Option<(DramTensor, Vec<tensor::Block>)>, TensorError> {
        let whole_rows = blocks.iter().all(|b| {
            let [r, c] = extent(b);
            b.at[1] == 0 && c == t.cols && b.at[0] + r <= t.rows
        });
        if !whole_rows || blocks.is_empty() {
            return Ok(None);
        }
        // `(first row, rows, place)` of each distinct block.
        let mut places: Vec<(usize, usize, usize)> = Vec::new();
        let mut at = Vec::with_capacity(blocks.len());
        let mut rows = Vec::new();
        for b in blocks {
            let r = extent(b)[0];
            let first = match places.iter().find(|p| (p.0, p.1) == (b.at[0], r)) {
                Some(p) => p.2,
                None => {
                    let first = rows.len();
                    places.push((b.at[0], r, first));
                    let padded = r.div_ceil(32) * 32;
                    let tail = if zero_tails {
                        (1, 0)
                    } else {
                        (tensor::GATHER_HOLE, 0)
                    };
                    rows.extend((0..padded).map(|i| if i < r { (0, b.at[0] + i) } else { tail }));
                    first
                }
            };
            at.push(tensor::Block {
                at: [first, 0],
                transposed: b.transposed,
            });
        }
        let gathered = if zero_tails {
            let zero = self.zero_row(t.cols, t.elem)?;
            self.gather_rows(&[t, &zero], &rows, t.cols)?
        } else {
            self.gather_rows(&[t], &rows, t.cols)?
        };
        Ok(Some((gathered, at)))
    }

    /// A `[1, cols]` row of zero `elem` datums, with zero padding (`metadata` fills each
    /// tile before writing it), for a row gather's tail rows to read; works on ttsim,
    /// unlike the XMOV zero fill. Made once per width and datum kind and kept as long as
    /// GDDR is: nothing writes it, and each one made per use was a `metadata` write an op
    /// (34 us of every replayed SmolLM2 layer's attention, twice:
    /// `docs/plans/replay-fixed-cost.md`). The caller must not free it.
    fn zero_row(&mut self, cols: usize, elem: tensor::Elem) -> Result<DramTensor, TensorError> {
        let kept = &self.dram_state()?.zero_rows;
        if let Some((_, _, z)) = kept.iter().find(|(c, e, _)| (*c, *e) == (cols, elem)) {
            return Ok(z.clone());
        }
        let z = self.metadata(&vec![0; cols], [1, cols], elem)?;
        self.dram_state()?.zero_rows.push((cols, elem, z.clone()));
        Ok(z)
    }

    /// Run an op's jobs over the units (see the module documentation) and
    /// return its output, or free the output and return the first error.
    ///
    /// Enqueue the operation through the ownership scheduler and optionally wait.
    fn execute(&mut self, work: tensor::Work, budget: u64) -> Result<DramTensor, TensorError> {
        // Work built without a unit count: its jobs go round-robin over the units the
        // spread gives an op its output's size.
        let units = self.spread_for(tiles_of(&work.out));
        self.execute_on(work, units, budget)
    }

    /// [`Session::execute`] for work built for `units` units (`Session::spread`): its
    /// jobs go to units `0..units`, as the builder that made them for that many expects.
    fn execute_on(
        &mut self,
        work: tensor::Work,
        units: usize,
        budget: u64,
    ) -> Result<DramTensor, TensorError> {
        let tensor::Work { out, jobs } = work;
        if let Err(error) = self.submit_jobs_on(jobs, units, budget) {
            if let Ok(dram) = self.dram_state() {
                dram.alloc.free(&out.placement);
            }
            return Err(error);
        }
        Ok(out)
    }

    pub(crate) fn submit_jobs(
        &mut self,
        jobs: Vec<tensor::Job>,
        budget: u64,
    ) -> Result<(), TensorError> {
        // Jobs built without a unit count, and with no output to size them by: as many
        // units as the spread gives that many jobs' work.
        let units = self.spread_for(jobs.len());
        self.submit_jobs_on(jobs, units, budget)
    }

    /// [`Session::submit_jobs`] for jobs built for `units` units.
    fn submit_jobs_on(
        &mut self,
        jobs: Vec<tensor::Job>,
        units: usize,
        budget: u64,
    ) -> Result<(), TensorError> {
        if self.profiler.is_some() {
            // The profile mode times this submission alone, under its label.
            self.profile_collect();
            let what = jobs
                .iter()
                .flatten()
                .find_map(|step| match step {
                    Step::List { what, .. } | Step::Transfer { what, .. } => Some(*what),
                    Step::Kernel { .. } => None,
                })
                .unwrap_or("control kernel");
            let label = self.op_label;
            if let Some(p) = self.profiler.as_mut() {
                p.expect_eager(label, what);
            }
            let result = self.submit_jobs_inner(jobs, units, budget);
            self.profile_collect();
            return result;
        }
        if self.batching {
            return self.submit_jobs_inner(jobs, units, budget);
        }
        let what = jobs
            .iter()
            .flatten()
            .find_map(|step| match step {
                Step::List { what, .. } | Step::Transfer { what, .. } => Some(*what),
                Step::Kernel { .. } => None,
            })
            .unwrap_or("control kernel");
        tensor::stats::timed(what, || self.submit_jobs_inner(jobs, units, budget))
    }

    fn submit_jobs_inner(
        &mut self,
        jobs: Vec<tensor::Job>,
        units: usize,
        budget: u64,
    ) -> Result<(), TensorError> {
        let queued = self.enqueue_work(jobs, units, budget);
        let synced = match &queued {
            Ok(()) if self.profiling.is_some() || !self.batching => self.sync(),
            Ok(()) => Ok(()),
            Err(_) => {
                let _ = self.sync();
                self.restart_barriers();
                if let Some(resident) = self.units[0].resident.as_ref() {
                    let _ = self.dev.write32(
                        resident.window(),
                        self.units[0].tile,
                        tt_isa::dm::BARRIER_COUNTER,
                        0,
                    );
                }
                Ok(())
            }
        };
        queued.and(synced)
    }

    /// After a failed queue: restart the mover of a unit whose list failed, and
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

    /// Limit K tiles per matmul block for reproducible chunking and gates.
    /// `None` lets the resident planner choose the block length.
    pub fn set_matmul_k_block_limit(&mut self, limit: Option<std::num::NonZeroUsize>) {
        self.matmul_k_block_limit = limit;
    }

    /// Whether a GDDR matmul reads only the rows of `op(A)`'s last tile row it
    /// uses, when that is 16 or fewer ([`tensor::thin_rows`]; on by default).
    /// Off, it reads whole tiles, as before: the gates compare the two.
    pub fn set_thin_matmul_reads(&mut self, on: bool) {
        self.thin_matmul_reads = on;
    }

    /// Whether a one-row product over a long `K` packs its `A` first ([`tensor::pack_row`],
    /// [`PACKED_A_K_TILES`]; on by default): each unit then reads `A` in `K / 1024` tiles, not
    /// a thin read a `K` tile. Off, every unit reads `A`'s `K / 32` tiles: the gates compare the
    /// two.
    pub fn set_packed_thin_a(&mut self, on: bool) {
        self.packed_thin_a = on;
    }

    /// Whether RMSNorm's statistic of one row over a long `K` reads the row packed
    /// ([`tensor::pack_row`], [`PACKED_A_K_TILES`]; on by default): the packing spread over the
    /// units, then `K / 1024` tile reads on the statistic's one unit, where the row takes `K / 32`
    /// entries there. Bit for bit either way (`sfpu::norm::math_packed`); off, the gates compare.
    pub fn set_packed_rms(&mut self, on: bool) {
        self.packed_rms = on;
    }

    /// Whether a one-row product with a packed `A` streams each output column's `K` in chunks,
    /// the mover reading one while the kernel computes the one before (on by default; see
    /// `tensor::BlockPlan::push_stream_jobs`). Off, a column's whole `K` is read before its
    /// kernel starts: the gates compare the two.
    pub fn set_streamed_k(&mut self, on: bool) {
        self.streamed_k = on;
    }

    /// Whether a captured one-row product with a packed `A` and its `K` whole puts its first
    /// columns on units idle since the last such product, each reading its columns' `B` as soon
    /// as that product is done -- under the ops between the two -- and the product then reading
    /// only `A` there (on by default; `tensor::BlockPlan::push_prefetched_jobs`). Only in a
    /// capture, whose streams show which units ran nothing in between. Off, every column is read
    /// by the product: the gates compare the two.
    pub fn set_weight_prefetch(&mut self, on: bool) {
        self.weight_prefetch = on;
    }

    /// Whether a GDDR matmul's later blocks on a unit skip reading the `A` rows its previous block
    /// there read to the same slots ([`tt_isa::dm::record::GATHER_A_RESIDENT`]; on by default).
    /// Off, every block reads its `A`: the gates compare the two.
    pub fn set_resident_matmul_a(&mut self, on: bool) {
        self.resident_matmul_a = on;
    }

    /// Whether a matmul's transposed `B` (on the TF32 route) is transposed by the kernel's
    /// unpacker (`record::GATHER_B_UNPACK_TRANSPOSE`): on by default. Off, the mover transposes
    /// each tile as it reads it (`READ_TRANSPOSED`, ~10 us a tile on silicon), the same bits.
    pub fn set_unpack_transposed_b(&mut self, on: bool) {
        self.unpack_transposed_b = on;
    }

    /// Whether a one-tile-row GDDR product over a long `K` (at least [`THIN_STREAM_K_TILES`]
    /// tiles: a decode step's weight stream) runs on at most [`THIN_STREAM_UNITS`] units (on by
    /// default). The bits are the same on any number of units.
    pub fn set_thin_stream_units(&mut self, on: bool) {
        self.thin_stream_units = on;
    }

    /// How the transcendentals run from the next element-wise op on
    /// (`sfpu::approx::MathMode`, S10; `TT_MATH=approx` sets it at open).
    /// `Precise`, the default, is the full-accuracy programs; `Approx` runs
    /// the fast ones, within their own looser bounds, for the ops that have
    /// one. Traces captured before the change keep the mode they recorded.
    pub fn set_math_mode(&mut self, mode: crate::sfpu::approx::MathMode) {
        self.math = mode;
    }

    /// The mode [`Session::set_math_mode`] set.
    pub fn math_mode(&self) -> crate::sfpu::approx::MathMode {
        self.math
    }

    /// Double-buffer GDDR matmuls from the next op on (checklist 9.15; on by
    /// default): each block staged in half the data arena, a unit's
    /// consecutive blocks in alternate halves, so its mover gathers the next
    /// block and scatters the last while the roles compute one
    /// (`tt_isa::dm::op::LAUNCH`, `KERNEL_WAIT`). Only where it pays
    /// (`tensor::pipelining_pays`): two blocks a unit or more, and at most
    /// twice the operand tiles gathered. The bits are the same. Captures retain
    /// the selected schedule, and replays do not replan it. Streaming mode with
    /// this disabled still uses NC, but waits for each output before slot reuse.
    /// How many of the session's units each op runs on ([`Spread`]); [`Spread::All`],
    /// every unit, by default. The results are the same bits whatever it is: a job runs
    /// the same wherever it runs, and an op's barrier involves the units it ran on
    /// ([`Session::catch_up`]).
    pub fn set_spread(&mut self, spread: Spread) {
        self.spread = spread;
    }

    /// See [`Session::set_spread`].
    pub fn spread(&self) -> Spread {
        self.spread
    }

    /// The units an op with `work` runs on, by [`Session::set_spread`].
    fn spread_for(&self, work: usize) -> usize {
        self.spread.units_for(work, self.units.len())
    }

    pub fn set_pipeline(&mut self, on: bool) {
        self.pipeline = on;
    }

    /// How an element-wise op or reduction queued now may overlap its runs
    /// ([`tensor::Overlap`]): not at all with pipelining off, as a capture
    /// plans it while one is open, as a fresh op otherwise.
    fn overlap(&self) -> tensor::Overlap {
        match (self.pipeline, self.capture.is_some()) {
            (false, _) => tensor::Overlap::Off,
            (true, true) => tensor::Overlap::Captured,
            (true, false) => tensor::Overlap::Fresh,
        }
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

    /// Give the timestamper `bytes` of L1 at the top of the program cache instead of the
    /// fixed 1,024-event buffer, the cache shrunk to what is left below them and emptied
    /// ([`Session::limit_program_cache`]); `None` puts both back. For profiling a stretch
    /// of ops the fixed buffer cannot hold, one stream to a [`Session::sync`] -- a trace
    /// replay above all, which runs to its end with no sync inside
    /// (`docs/plans/replay-fixed-cost.md`). Refused while a capture is open, a trace
    /// lives (it holds programs in the cache) or a profile runs.
    #[doc(hidden)]
    /// Count what each op of every trace stored from here on moves through GDDR
    /// ([`trace::OpRecord::bytes`]), as the profile mode does, without the profile mode: host
    /// work at the trace's store, nothing on the card, so it runs on the simulator too.
    pub fn set_count_trace_bytes(&mut self, on: bool) {
        self.count_trace_bytes = on;
    }

    pub fn set_profile_buffer(&mut self, bytes: Option<u64>) -> Result<(), TensorError> {
        let full = tt_isa::l1::PROGRAM_CACHE;
        let refuse = |why: &str| Err(TensorError::Shape(format!("set_profile_buffer: {why}")));
        if self.capture.is_some() || !self.traces.is_empty() || self.profiling.is_some() {
            return refuse("a capture, a trace or a profile is live");
        }
        let (buffer, cache) = match bytes {
            None => (
                (
                    tt_isa::mailbox::TRACE_BUFFER,
                    tt_isa::mailbox::TRACE_BUFFER_BYTES,
                ),
                full.len(),
            ),
            Some(b) if b % tt_isa::tensix::timestamper::EVENT_BYTES == 0 && b < full.len() => {
                ((full.end - b, b), full.len() - b)
            }
            Some(_) => return refuse("whole 16-byte events, less than the program cache"),
        };
        self.limit_program_cache(cache)?;
        self.profile_buffer = buffer;
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
    /// Open `/dev/tenstorrent/{index}` for compute, holding the card's driver
    /// lock until the session drops ([`SessionError::CardBusy`] if another
    /// session has it), with the driver's cleanup write as step 3: the first tile's on this descriptor, every further
    /// tile's on a descriptor of its own, since the driver keeps one per
    /// descriptor.
    pub fn open_card(
        index: u16,
        images: RoleImages<'static>,
        choice: TileChoice,
    ) -> Result<Self, SessionError> {
        let kmd = tt_kmd::Kmd::open(index)?;
        // Before anything is written to the card: a second session resetting
        // tiles under a running one would corrupt both. The lock lives as long
        // as `kmd`'s descriptor, which the `Device` and so the session keep.
        if !kmd.lock_card()? {
            return Err(SessionError::CardBusy { card: index });
        }
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
    use super::{
        control_segments, healthy_tiles, runtime, streaming_segments, transfer_segments,
        Instruction, RunError, SessionError, Spread, Step,
    };

    #[test]
    fn a_spread_gives_an_op_between_one_unit_and_all_of_them() {
        for units in [1, 4, 120] {
            assert_eq!(Spread::All.units_for(0, units), units);
            assert_eq!(Spread::All.units_for(1 << 20, units), units);
            assert_eq!(Spread::Fixed(2).units_for(1 << 20, units), 2.min(units));
            assert_eq!(Spread::Fixed(0).units_for(1, units), 1);
            assert_eq!(Spread::Fixed(1000).units_for(1, units), units);
            // One unit for each share of the work, rounded up, never none.
            let work = |share, min| Spread::Work { share, min };
            assert_eq!(work(8, 0).units_for(0, units), 1);
            assert_eq!(work(8, 1).units_for(8, units), 1);
            assert_eq!(work(8, 1).units_for(9, units), 2.min(units));
            assert_eq!(work(8, 1).units_for(8 * 1000, units), units);
            assert_eq!(work(0, 1).units_for(3, units), 3.min(units));
            // The floor: a small op still gets `min` units, a large one more.
            assert_eq!(work(64, 2).units_for(1, units), 2.min(units));
            assert_eq!(work(64, 2).units_for(64 * 5, units), 5.min(units));
        }
    }
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
    fn capture_rebases_headers_without_treating_tensor_payloads_as_commands() {
        use tt_isa::dm::record;
        let entries = [
            [record::READ_RUN, 0, 0, 0, 0, 0, 0, 0],
            [op::KERNEL, 999, 888, 777, 666, 555, 444, 333],
            [op::LAUNCH, 999, 888, 777, 666, 555, 444, 333],
            [op::LAUNCH, 0, 16, 1, 32, 1, 48, 1],
            [op::KERNEL_WAIT, 0, 0, 0, 0, 0, 0, 0],
        ];
        let mut held = Vec::new();
        let captured =
            super::capture_commands(&entries, u32::MAX, |address| held.push(address)).unwrap();
        assert_eq!(&captured[..3], &entries[..3]);
        assert_eq!(captured[3][1], 1);
        assert_eq!(captured[4][1], 1);
        assert_eq!(held, [16, 32, 48]);
        assert!(super::capture_commands(&entries[..2], 0, |_| {}).is_err());
    }

    #[test]
    fn compute_without_a_resident_ownership_region_is_rejected() {
        let kernel = |words| Step::Kernel {
            roles: Arc::new([vec![tt_isa::sfpu::nop(); words], Vec::new(), Vec::new()]),
            init: Vec::new(),
            mop: Box::new([None; 3]),
            loops: Default::default(),
            half: None,
        };
        let cache = tt_isa::l1::PROGRAM_CACHE.len();
        let message = |steps| match streaming_segments(steps, cache) {
            Err(crate::tensor::TensorError::Shape(message)) => message,
            other => panic!("expected a rejection, got {:?}", other.map(|s| s.len())),
        };
        // A kernel on its own is not between a gather and a scatter, and
        // nothing precedes it.
        let bare = message(vec![kernel(1)]);
        assert!(bare.contains("after `` cannot form"), "{bare}");
        assert!(
            bare.contains("not between a gather list and a scatter list"),
            "{bare}"
        );
        // One over half the cache is named by size, after the list before it.
        let oversized = (cache / 2 / 4 + 1) as usize;
        let big = message(vec![list(1, 1), kernel(oversized), list(1, 2)]);
        assert!(big.contains("after `test` cannot form"), "{big}");
        assert!(
            big.contains(&format!("a role program of {oversized} words exceeds half")),
            "{big}"
        );
    }

    /// A transfer step: `batches` batches of one read and one write each.
    fn transfer(what: &'static str, depth: u16, batches: usize, side: usize) -> Step {
        Step::Transfer {
            what,
            depth,
            batches: (0..batches)
                .map(|_| crate::tensor::TransferBatch {
                    read: vec![[op::READ, 0, 0, 0x40, 0x2_0040, 64, 0, 0]; side],
                    write: vec![[op::WRITE, 0, 0, 0x40, 0x2_0040, 64, 0, 0]; side],
                })
                .collect(),
        }
    }

    #[test]
    fn transfers_are_kernelless_packets_that_pass_the_ownership_rules() {
        use tt_isa::dataflow::{Action, Release, Stream};
        for depth in [1u16, 2] {
            let segments = transfer_segments(vec![
                transfer("upload", depth, 5, 3),
                transfer("copy", depth, 2, 4),
            ]);
            assert_eq!(segments.len(), 1, "both fit one packet");
            let packet = &segments[0];
            assert!(
                packet.streaming && packet.kernels.is_empty() && packet.kernel_roles.is_empty()
            );
            assert_eq!(packet.steps, 2);
            assert_eq!(packet.entries[0][3], depth as u32);
            assert_eq!(
                tt_isa::dataflow::packet_length(packet.entries[0]),
                Ok(packet.entries.len())
            );
            crate::dm::check_pair(&packet.entries).unwrap();
            // Every credit is the transfer channel's; no launch, no join.
            for entry in &packet.entries[1..] {
                assert!(
                    !matches!(entry[0], op::LAUNCH | op::KERNEL_WAIT),
                    "a transfer runs no kernel"
                );
                if entry[0] == op::CB {
                    assert_eq!(entry[1], Stream::Transfer.word());
                }
            }
            // At depth two a drain separates the transfers; at depth one the
            // next reserve already waits for the last pop.
            let drains: Vec<u32> = packet
                .entries
                .iter()
                .filter(|e| e[0] == op::RELEASED)
                .map(|e| {
                    assert_eq!(e[2], Release::Transferred.word());
                    e[1]
                })
                .collect();
            assert_eq!(drains, if depth == 2 { vec![5] } else { vec![] });
            // Credits that cannot block are left out: the first `depth`
            // batches' reserves, and the last batch's pop.
            let count = |range: std::ops::Range<usize>, action: u32| {
                packet.entries[range]
                    .iter()
                    .filter(|e| e[0] == op::CB && e[2] == action)
                    .count()
            };
            let writer_at = 1 + packet.entries[0][1] as usize;
            let all = 1..packet.entries.len();
            assert_eq!(
                count(1..writer_at, Action::Reserve.word()),
                7 - depth as usize
            );
            assert_eq!(count(1..writer_at, Action::Push.word()), 7);
            assert_eq!(count(writer_at..all.end, Action::Wait.word()), 7);
            assert_eq!(count(writer_at..all.end, Action::Pop.word()), 6);
        }
    }

    #[test]
    fn transfers_of_different_depths_or_too_long_for_one_packet_split_whole_batches() {
        let segments = transfer_segments(vec![transfer("a", 1, 1, 2), transfer("b", 2, 1, 2)]);
        assert_eq!(segments.len(), 2, "a depth change closes the packet");
        // 100 batches of 2 entries a side: 8 entries a batch beside its
        // credits is far over one packet, which closes between batches.
        let segments = transfer_segments(vec![transfer("many", 1, 100, 2)]);
        assert!(segments.len() > 1);
        let mut batches = 0;
        for packet in &segments {
            assert!(
                packet.entries.len() < tt_isa::dm::LIST_MAX as usize,
                "room for a trailer"
            );
            crate::dm::check_pair(&packet.entries).unwrap();
            batches += packet
                .entries
                .iter()
                .filter(|e| e[0] == op::CB && e[2] == 2)
                .count();
        }
        assert_eq!(batches, 100, "every batch pushed once, none split");
    }

    #[test]
    fn a_transfer_between_lists_keeps_its_place() {
        let list = |what| Step::List {
            what,
            entries: vec![[op::WAIT, 0, 0, 0, 0, 0, 0, 0]],
        };
        let segments = streaming_segments(
            vec![list("before"), transfer("t", 1, 1, 1), list("after")],
            tt_isa::l1::PROGRAM_CACHE.len(),
        )
        .unwrap();
        let order: Vec<(&str, bool)> = segments.iter().map(|s| (s.what, s.streaming)).collect();
        assert_eq!(order, [("before", false), ("t", true), ("after", false)]);
    }

    #[test]
    fn a_writer_section_with_a_reader_credit_is_refused() {
        let mut packet = transfer_segments(vec![transfer("t", 1, 1, 1)])
            .remove(0)
            .entries;
        let writer = 1 + packet[0][1] as usize;
        // NC may not push a transfer it consumes.
        packet[writer][2] = tt_isa::dataflow::Action::Push.word();
        assert!(crate::dm::check_pair(&packet).is_err());
    }

    /// A depth-two transfer packet, and where its writer section starts.
    fn packet() -> (Vec<[u32; 8]>, usize) {
        let packet = transfer_segments(vec![transfer("t", 2, 3, 1)])
            .remove(0)
            .entries;
        let writer = 1 + packet[0][1] as usize;
        (packet, writer)
    }

    #[test]
    fn the_ownership_checks_pass_the_packet_they_break() {
        let (packet, writer) = packet();
        assert!(crate::dm::check_pair(&packet).is_ok());
        // Both sections hold credits, so the cases below change real ones.
        let credit = |from: usize, to: usize| (from..to).find(|&i| packet[i][0] == op::CB);
        assert!(credit(1, writer).is_some() && credit(writer, packet.len()).is_some());
    }

    #[test]
    fn a_reader_holding_an_output_credit_is_refused() {
        use tt_isa::dataflow::{Action, Stream};
        let (mut packet, writer) = packet();
        let at = (1..writer).find(|&i| packet[i][0] == op::CB).unwrap();
        // The output's producer is T2 and its consumer NC: B is neither.
        for action in [Action::Reserve, Action::Push, Action::Wait, Action::Pop] {
            packet[at][1] = Stream::Output.word();
            packet[at][2] = action.word();
            assert!(crate::dm::check_pair(&packet).is_err(), "{action:?}");
        }
        // Nor does a reader take the transfer's consumer side.
        packet[at][1] = Stream::Transfer.word();
        for action in [Action::Wait, Action::Pop] {
            packet[at][2] = action.word();
            assert!(crate::dm::check_pair(&packet).is_err(), "{action:?}");
        }
    }

    #[test]
    fn a_credit_at_another_capacity_than_the_header_is_refused() {
        let (packet, writer) = packet();
        for section in [1..writer, writer..packet.len()] {
            let at = section.clone().find(|&i| packet[i][0] == op::CB).unwrap();
            for capacity in [1, 3, 0x7fff] {
                let mut changed = packet.clone();
                changed[at][3] = capacity;
                assert!(crate::dm::check_pair(&changed).is_err(), "{capacity}");
            }
        }
    }

    #[test]
    fn a_capacity_of_zero_or_over_the_counters_is_refused() {
        for capacity in [0u32, 0x8000, 0xffff] {
            let (mut packet, _) = packet();
            packet[0][3] = capacity;
            for entry in &mut packet {
                if entry[0] == op::CB {
                    entry[3] = capacity;
                }
            }
            assert!(crate::dm::check_pair(&packet).is_err(), "{capacity}");
        }
    }

    #[test]
    fn a_writer_section_may_not_read() {
        let (mut packet, writer) = packet();
        // NC issues on NoC #1 beside a reading B and never reads GDDR itself.
        let at = (writer..packet.len())
            .find(|&i| packet[i][0] == op::WRITE)
            .unwrap();
        packet[at][0] = op::READ;
        assert!(crate::dm::check_pair(&packet).is_err());
    }

    #[test]
    fn a_section_that_does_not_fill_the_packet_is_refused() {
        let (mut packet, _) = packet();
        packet[0][2] += 1;
        assert!(crate::dm::check_pair(&packet).is_err());
        packet[0][2] -= 2;
        assert!(crate::dm::check_pair(&packet).is_err());
    }

    #[test]
    fn streaming_launches_scale_with_regions_and_serialize_slot_reuse() {
        for capacity in [1usize, 2] {
            let mut steps = Vec::new();
            for index in 0..super::GROUP_MAX + 3 {
                steps.push(list(1, 1));
                steps.push(Step::Kernel {
                    roles: Arc::new([Vec::new(), Vec::new(), Vec::new()]),
                    init: Vec::new(),
                    mop: Box::new([None; 3]),
                    loops: Default::default(),
                    half: (capacity == 2).then_some((index % 2) as u8),
                });
                steps.push(Step::List {
                    what: "scatter",
                    entries: vec![[op::WRITE, 0, 0, 0, 0, 0, 0, 0]],
                });
            }
            let regions = streaming_segments(steps, tt_isa::l1::PROGRAM_CACHE.len()).unwrap();
            assert_eq!(regions.len(), 2);
            for region in &regions {
                assert!(region.streaming);
                assert_eq!(region.entries[0][3], capacity as u32);
                assert_eq!(
                    tt_isa::dataflow::packet_length(region.entries[0]),
                    Ok(region.entries.len())
                );
                assert_eq!(region.kernels, [1]);
                assert_eq!(region.waits.len(), 1);
                // These moves' footprints are not modelled (zero lengths), so
                // each slot reuse waits for NC's release of the batch that
                // last used it; the first `capacity` batches wait for nothing.
                let waits: Vec<[u32; 2]> = region
                    .entries
                    .iter()
                    .filter(|entry| entry[0] == op::RELEASED)
                    .map(|entry| [entry[1], entry[2]])
                    .collect();
                let batches = region.kernel_roles.len();
                assert_eq!(waits.len(), batches.saturating_sub(capacity));
                for (index, wait) in waits.into_iter().enumerate() {
                    assert_eq!(wait, [index as u32 + 1, 0]);
                }
            }
            assert_eq!(regions[0].kernel_roles.len(), super::GROUP_MAX);
            assert_eq!(regions[1].kernel_roles.len(), 3);
        }
    }

    /// A gather that cannot land on what an older batch's scatter reads waits
    /// only for that batch's pack, so it overlaps NC's writes; one that can
    /// waits for NC's release.
    #[test]
    fn slot_reuse_waits_for_release_only_where_a_gather_meets_a_scatter() {
        let block = |input: u32, output: u32| {
            let gather = Step::List {
                what: "gather",
                entries: vec![[op::READ, 0, 0, 0, input, 4096, 0, 0]],
            };
            let kernel = Step::Kernel {
                roles: Arc::new([Vec::new(), Vec::new(), Vec::new()]),
                init: Vec::new(),
                mop: Box::new([None; 3]),
                loops: Default::default(),
                half: None,
            };
            let scatter = Step::List {
                what: "scatter",
                entries: vec![[op::WRITE, 0, 0, 0, output, 4096, 0, 0]],
            };
            [gather, kernel, scatter]
        };
        let waits = |steps: Vec<Step>| -> Vec<[u32; 2]> {
            let regions = streaming_segments(steps, tt_isa::l1::PROGRAM_CACHE.len()).unwrap();
            assert_eq!(regions.len(), 1);
            regions[0]
                .entries
                .iter()
                .filter(|entry| entry[0] == op::RELEASED)
                .map(|entry| [entry[1], entry[2]])
                .collect()
        };
        // Inputs and outputs apart: pack waits only.
        let apart = [
            block(0x2_0000, 0x3_0000),
            block(0x2_0000, 0x3_0000),
            block(0x2_0000, 0x3_0000),
        ];
        assert_eq!(waits(apart.concat()), [[1, 1], [2, 1]]);
        // Outputs where the next gather lands: release waits.
        let shared = [
            block(0x2_0000, 0x2_0800),
            block(0x2_0000, 0x2_0800),
            block(0x2_0000, 0x2_0800),
        ];
        assert_eq!(waits(shared.concat()), [[1, 0], [2, 0]]);
        // A dependent GDDR reload needs NC completion even when its L1
        // destination is separate. A WAIT in the gather conservatively
        // marks this dependency and works for fresh and captured lists.
        let mut dependent = apart;
        for steps in &mut dependent[1..] {
            if let Step::List { entries, .. } = &mut steps[0] {
                entries.insert(0, [op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
            }
        }
        assert_eq!(waits(dependent.concat()), [[1, 0], [2, 0]]);
    }

    #[test]
    fn separate_lists_share_one_with_a_wait_between_them() {
        let segs = control_segments(vec![list(2, 1), list(1, 2)]);
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
        let segs = control_segments([block(), block()].concat());
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
        let segs = control_segments(vec![k(&a, &init), list(1, 1), k(&b, &init), k(&a, &init)]);
        assert_eq!(segs.len(), 1, "every program is resident: one list");
        assert!(segs[0].resident);
        assert_eq!(segs[0].kernel_roles.len(), 3);
        // Two distinct program sets, counted once each.
        assert_eq!(segs[0].resident_bytes, 4 + 8);
        // Other semaphore starting values: another setup, so another list.
        let mut other = init.clone();
        other[0].1 = 1;
        assert_eq!(control_segments(vec![k(&a, &init), k(&a, &other)]).len(), 2);
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
        let segs = control_segments(vec![k(&a), k(&a), k(&b)]);
        assert_eq!(segs.len(), 2);
        assert!(!segs[0].resident && segs[0].kernels.len() == 2);
        // A resident kernel does not join a fixed-slot list either.
        let small = Arc::new([vec![tt_isa::sfpu::nop()], Vec::new(), Vec::new()]);
        assert_eq!(control_segments(vec![k(&a), k(&small)]).len(), 2);
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
        let segs = control_segments(steps);
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
        let segs = control_segments(vec![list(n, 1)]);
        assert_eq!(
            segs.iter().map(|s| s.entries.len()).collect::<Vec<_>>(),
            [LIST_MAX as usize, 3]
        );
        assert!(control_segments(vec![list(0, 1)]).is_empty());
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
