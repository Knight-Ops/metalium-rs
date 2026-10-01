//! Running a host-built Tensix program through the `corpus` firmware.
//!
//! Every gate from step 4 onwards has the same shape: stage some bytes in L1,
//! stage an instruction stream, start a core, wait for it to report `DONE`, then
//! read `Dst` and L1 back. This is that shape, once, so a new gate is a program
//! and a set of assertions rather than another copy of the staging sequence.
//!
//! # Why everything runs inside a fork
//!
//! ttsim `_Exit`s the process on any contract violation, so a refusal would
//! otherwise take the test runner with it. [`in_device`] forks first, which turns
//! a refusal into a failing assertion, and [`survives`] turns it into a *value* —
//! which is what makes the probes in this crate discovery tools rather than just
//! tests.
//!
//! The fork survives the move to silicon, where there is no `_Exit` to catch, for
//! two other reasons: a gate that wedges a baby RISC-V cannot take the runner
//! with it, and it is the child's file descriptor closing that fires the driver's
//! registered cleanup write. See [`crate::backend`].

use tt_device::core_control::WaitError;
use tt_device::tlb::WindowKind;
use tt_device::Window;
use tt_isa::isa::Instruction;
use tt_isa::mailbox::{self, status};
use tt_isa::noc::{grid, Noc0, NocCoord};
use tt_isa::tensix::Core;

/// Re-exported so a gate can say `harness::Dev` without caring which target it
/// is built for. [`crate::backend`] is where the choice is made.
pub use crate::backend::{
    advance, assert_on_silicon, in_device, survives, tensix_grid, tile, Dev, ON_SILICON,
};

/// Which core pushes the program, and so which Tensix thread runs it.
///
/// **T0 on silicon, T1 on the simulator**, and the difference is load-bearing.
/// `UNPACR` in multi-context mode -- the only mode the specification supports --
/// takes its X counters from `ADCs[ContextADC]`, an instruction field that is 0,
/// not from the issuing thread (`UNPACR_Regular.md:45-59`). The packer and the
/// `SETADC*` instructions use the issuing thread. So an unpack issued from thread
/// 1 is programmed through thread 1's ADCs and executed with thread 0's: on
/// silicon it moved exactly one datum (m10 in `silicon_measure.rs`). ttsim uses
/// the issuing thread's ADCs (divergence row 45), which hid it, and ttsim only
/// lets T1 read `Dst` (row 12), which forced T1 there. tt-metal's LLK unpacks
/// from TRISC0 for the same reason.
#[cfg(feature = "silicon")]
pub const CORE: Core = Core::T0;
#[cfg(not(feature = "silicon"))]
pub const CORE: Core = Core::T1;
/// The Tensix thread `CORE` pushes to, and the value the firmware cross-checks
/// against `mailbox::THREAD_INDEX`.
#[cfg(feature = "silicon")]
pub const CORE_THREAD: u32 = 0;
#[cfg(not(feature = "silicon"))]
pub const CORE_THREAD: u32 = 1;
/// `RISC_DEST_ACCESS_CTRL_SEC*.fmt` 0 is `float Dst32b[512][16]`.
pub const DST_FMT_FP32: u32 = 0;
/// Simulator cycles a run may take before it is called hung.
pub const BUDGET: u64 = 400_000;
/// Pre-filled into every dumped `Dst` datum, so a datum the firmware never wrote
/// is distinguishable from a computed zero.
pub const SENTINEL: u32 = tt_kernels::runtime::DUMP_SENTINEL;

/// The Tensix tile the gates use.
pub fn tensix_tile() -> NocCoord<Noc0> {
    let (x, y) = crate::backend::GATE_TILE;
    assert!(
        grid::is_tensix_geometry(x, y),
        "the gates' tile must be a Tensix tile"
    );
    NocCoord::new(x, y).unwrap()
}

/// The relay tile of a multi-chip gate ([`crate::backend::RELAY_TILE`]).
pub fn relay_tile() -> NocCoord<Noc0> {
    let (x, y) = crate::backend::RELAY_TILE;
    assert!(
        grid::is_tensix_geometry(x, y),
        "the relay tile must be a Tensix tile"
    );
    NocCoord::new(x, y).unwrap()
}

/// What to stage before a run and what to read after it.
pub struct Run<'a> {
    /// The instruction stream, pushed verbatim by the firmware.
    pub program: &'a [Instruction],
    /// Bytes to write into L1 first, as `(address, data)`.
    pub stage: &'a [(u64, &'a [u8])],
    /// How many `Dst` rows the firmware should copy back.
    pub dump_rows: u32,
    /// L1 ranges to read after the run, as `(address, length)`.
    pub read_back: &'a [(u64, usize)],
    /// `RISC_DEST_ACCESS_CTRL_SEC*.fmt`.
    pub dst_fmt: u32,
    /// Zero the dumped `Dst` rows before the program ([`dst_clear_prelude`]).
    /// On by default; a probe of the prelude itself turns it off.
    pub clear_dst: bool,
    /// Run somewhere other than the gate tile. Claimed through [`tile`], so it
    /// is checked against the chip's grid and scrubbed afterwards.
    pub tile: Option<(u8, u8)>,
    /// Split the kernel across the three Tensix threads, as LLK does; see
    /// [`Roles`]. When set, `program` is ignored.
    pub roles: Option<Roles<'a>>,
    /// Run the roles at the same time rather than in order, with these
    /// semaphores initialised first; see [`Run::concurrent`].
    pub concurrent: Option<&'a [SemaphoreInit]>,
    /// Record each role's progress through the tile's timestamper; see
    /// [`Run::traced`].
    pub trace: bool,
}

pub use tt_kernels::runtime::SemaphoreInit;

/// A kernel split the way tt-metal's LLK splits it: thread 0 unpacks, thread 1
/// does math (Matrix Unit and SFPU), thread 2 packs, each program pushed by its
/// own core (T0, T1, T2) and reporting through its own mailbox
/// (`tt_isa::mailbox::role`).
///
/// Much of the coprocessor's state is per thread -- ADCs, RWCs, address
/// modifiers, `ThreadConfig` -- and the hardware ties some of it to a role:
/// `UNPACR` counts with thread 0's ADCs whichever thread issues it (divergence
/// row 45). A single-thread program shares one set of all of it between roles
/// LLK keeps apart.
///
/// The roles run **in order**, each to completion -- unpack, then math, then
/// pack -- which is one valid schedule of the same three programs, and needs
/// no cross-thread semaphores. Work crosses threads the way it does in LLK: an
/// unpack into `Src` hands the bank to the Matrix Unit (`FlipSrc`), and `Dst`
/// is shared.
#[derive(Copy, Clone)]
pub struct Roles<'a> {
    pub unpack: &'a [Instruction],
    pub math: &'a [Instruction],
    pub pack: &'a [Instruction],
}

impl<'a> Run<'a> {
    /// A run that stages nothing and reads back only `Dst`.
    pub fn new(program: &'a [Instruction]) -> Self {
        Run {
            program,
            stage: &[],
            dump_rows: 4,
            read_back: &[],
            dst_fmt: DST_FMT_FP32,
            clear_dst: true,
            tile: None,
            roles: None,
            concurrent: None,
            trace: false,
        }
    }

    /// A kernel split across the three threads; see [`Roles`].
    pub fn roles(roles: Roles<'a>) -> Self {
        Run {
            roles: Some(roles),
            ..Run::new(&[])
        }
    }

    pub fn stage(mut self, stage: &'a [(u64, &'a [u8])]) -> Self {
        self.stage = stage;
        self
    }

    pub fn dump_rows(mut self, rows: u32) -> Self {
        self.dump_rows = rows;
        self
    }

    /// Release the three roles together and wait for all of them, rather than
    /// running each to completion in turn.
    ///
    /// The in-order schedule deadlocks as soon as one role waits on another:
    /// an unpacker that fills more `Src` banks than exist waits for a Matrix Unit
    /// whose thread has not started. Concurrently, the roles synchronise the way
    /// LLK's do -- `Src` banks by hardware, `Dst` through the semaphores in
    /// `init` (`tt_isa::sync`). Those are set, and the `Dst` rows about to be
    /// dumped zeroed, by a setup run on thread 0 that completes before any role
    /// starts: a semaphore outlives the program that last used it on silicon,
    /// so a consumer released alongside its own `SEMINIT` could read a stale
    /// value first.
    pub fn concurrent(mut self, init: &'a [SemaphoreInit]) -> Self {
        self.concurrent = Some(init);
        self
    }

    /// Have each role record `START`, `PUSHED` and `RETIRED` through the
    /// tile's timestamper (`tt_isa::mailbox::trace`), returned in
    /// [`Outcome::trace`]. Silicon only: ttsim's support is its own probe.
    pub fn traced(mut self) -> Self {
        self.trace = true;
        self
    }

    pub fn read_back(mut self, ranges: &'a [(u64, usize)]) -> Self {
        self.read_back = ranges;
        self
    }
}

/// What a run produced.
pub struct Outcome {
    /// `Dst`, flattened as `row * mailbox::DUMP_ROW_WORDS + column`.
    pub dst: Vec<u32>,
    /// One buffer per [`Run::read_back`] range, in order.
    pub l1: Vec<Vec<u8>>,
    /// The timestamper events of a [`Run::traced`] run, in order.
    pub trace: Vec<tt_device::trace::TraceEvent>,
}

impl Outcome {
    /// `Dst[row][column]`, by the firmware's dump indexing.
    pub fn dst_at(&self, row: usize, column: usize) -> u32 {
        self.dst[row * mailbox::DUMP_ROW_WORDS as usize + column]
    }

    /// Every `Dst` datum the firmware wrote that is neither zero nor the sentinel,
    /// as `(flat index, value)`.
    pub fn dst_nonzero(&self) -> Vec<(usize, u32)> {
        self.dst
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0 && v != SENTINEL)
            .map(|(i, &v)| (i, v))
            .collect()
    }
}

/// Zero the `Dst` rows a run will dump, before its program runs.
///
/// `Dst` has no reset value and survives everything the harness can do to a tile
/// from outside -- the backend soft-reset pulse included (`SoftReset.md`) -- so on
/// silicon a run reads whatever the last one left: the first silicon run of
/// `probe_unpack` found an earlier probe's `-7.0` marker in rows it asserts are
/// untouched. ttsim hands out a fresh, zeroed chip per run, which is what the
/// gates were written against. Done through the SFPU, whose `Dst` writes read
/// back correctly on silicon (`step5_corpus`), and identically on both targets so
/// the programs stay the same; the wait keeps it ahead of anything the program
/// then writes.
fn dst_clear_prelude(dump_rows: u32) -> Vec<Instruction> {
    use tt_isa::sfpu;
    let mut p = Vec::new();
    p.extend(sfpu::load_f32(0, 0).unwrap());
    for group in 0..dump_rows.div_ceil(4) {
        for cols in [0, sfpu::DST_ODD_COLUMNS] {
            p.push(sfpu::store(0, sfpu::store_format::FP32, 0, (group * 4) | cols).unwrap());
        }
    }
    p.push(tt_isa::backend::wait_for_sfpu(tt_isa::backend::Before::EVERYTHING).unwrap());
    p
}

/// Stage, run, and read back.
///
/// Panics rather than returning an error: every failure here is a broken gate, not
/// a condition a caller could handle.
pub fn run(dev: &mut Dev<'_>, spec: &Run<'_>) -> Outcome {
    if let Some(roles) = spec.roles {
        return run_roles(dev, spec, roles);
    }
    let mut program = if spec.clear_dst {
        dst_clear_prelude(spec.dump_rows)
    } else {
        Vec::new()
    };
    program.extend_from_slice(spec.program);
    assert!(
        program.len() as u32 <= mailbox::PROGRAM_MAX,
        "program is {} instructions with the Dst-clearing prelude; the mailbox holds {}",
        program.len(),
        mailbox::PROGRAM_MAX
    );
    assert!(spec.dump_rows <= mailbox::DUMP_MAX_ROWS);

    let tile = match spec.tile {
        Some((x, y)) => crate::backend::tile(dev, x, y),
        None => tensix_tile(),
    };
    let w: Window = dev.alloc_window(WindowKind::TwoMib).unwrap();

    // The backend has to be out of reset before the coprocessor executes anything,
    // and before the core can start pushing into it.
    dev.release_tensix_backend(&w, tile).unwrap();

    for (addr, data) in spec.stage {
        dev.write(&w, tile, *addr, data).unwrap();
    }

    dev.write32(&w, tile, mailbox::STATUS, 0).unwrap();
    assert!(!spec.trace, "tracing is for role runs");
    let push_window = if ON_SILICON {
        0
    } else {
        mailbox::SIM_PUSH_WINDOW
    };
    // Whole, so nothing a previous process left in L1 survives into this run.
    let d = mailbox::Descriptor {
        thread_index: CORE_THREAD,
        dst_access_fmt: spec.dst_fmt,
        program_len: program.len() as u32,
        dump_row_count: spec.dump_rows,
        push_window,
        ..Default::default()
    };
    for (at, v) in d.writes(mailbox::role::Mailbox::single_core()) {
        dev.write32(&w, tile, at, v).unwrap();
    }
    dev.write(&w, tile, mailbox::PROGRAM, &program_bytes(&program))
        .unwrap();
    for row in 0..spec.dump_rows {
        for col in 0..mailbox::DUMP_ROW_WORDS {
            dev.write32(&w, tile, mailbox::dump_offset(row, col), SENTINEL)
                .unwrap();
        }
    }

    dev.load_and_start(
        &w,
        tile,
        CORE,
        crate::firmware::CORPUS,
        crate::firmware::LOAD_ADDRESS,
    )
    .unwrap();
    match dev
        .wait_for_status(&w, tile, BUDGET, |s| s == status::DONE)
        .unwrap()
    {
        Ok(_) => {}
        Err(WaitError::Panicked { code }) => panic!("firmware panicked, code {code}"),
        Err(e) => panic!("{e}"),
    }

    let mut dst = Vec::new();
    for row in 0..spec.dump_rows {
        for col in 0..mailbox::DUMP_ROW_WORDS {
            dst.push(
                dev.read32(&w, tile, mailbox::dump_offset(row, col))
                    .unwrap(),
            );
        }
    }
    let trace = Vec::new();
    let mut l1 = Vec::new();
    for (addr, len) in spec.read_back {
        let mut buf = vec![0u8; *len];
        dev.read(&w, tile, *addr, &mut buf).unwrap();
        l1.push(buf);
    }
    Outcome { dst, l1, trace }
}

/// A program as the little-endian words the firmware pushes, for one bulk
/// write rather than an MMIO round trip per instruction.
pub fn program_bytes(program: &[Instruction]) -> Vec<u8> {
    program
        .iter()
        .flat_map(|i| i.word().to_le_bytes())
        .collect()
}

/// [`run`] for a [`Roles`] kernel: `tt_kernels::runtime::run`, with a failure
/// turned into a panic naming the stuck roles.
fn run_roles(dev: &mut Dev<'_>, spec: &Run<'_>, roles: Roles<'_>) -> Outcome {
    use tt_kernels::runtime::{self, Kernel, Schedule};
    let tile = match spec.tile {
        Some((x, y)) => crate::backend::tile(dev, x, y),
        None => tensix_tile(),
    };
    let schedule = match spec.concurrent {
        Some(init) => Schedule::Concurrent(init),
        None => Schedule::InOrder,
    };
    let kernel = Kernel {
        stage: spec.stage,
        read_back: spec.read_back,
        dump_rows: spec.dump_rows,
        clear_dst: spec.clear_dst,
        dst_fmt: spec.dst_fmt,
        trace: spec.trace,
        ..Kernel::new([roles.unpack, roles.math, roles.pack], schedule)
    };
    match runtime::run(dev, tile, &crate::firmware::ROLES, &kernel, BUDGET) {
        Ok(out) => Outcome {
            dst: out.dst,
            l1: out.l1,
            trace: out.trace,
        },
        Err(e) => panic!("{e}"),
    }
}
