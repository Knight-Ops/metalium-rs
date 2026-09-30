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
pub const SENTINEL: u32 = 0xDEAD_BEEF;

/// The Tensix tile the gates use.
pub fn tensix_tile() -> NocCoord<Noc0> {
    let (x, y) = crate::backend::GATE_TILE;
    assert!(
        grid::is_tensix_geometry(x, y),
        "the gates' tile must be a Tensix tile"
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

/// A semaphore's starting `Value` and `Max`, set before a concurrent run.
pub type SemaphoreInit = (tt_isa::sync::Semaphore, u8, u8);

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
    dev.write32(&w, tile, mailbox::THREAD_INDEX, CORE_THREAD)
        .unwrap();
    dev.write32(&w, tile, mailbox::DST_ACCESS_FMT, spec.dst_fmt)
        .unwrap();
    dev.write32(&w, tile, mailbox::PROGRAM_LEN, program.len() as u32)
        .unwrap();
    dev.write32(&w, tile, mailbox::DUMP_ROW_FIRST, 0).unwrap();
    dev.write32(&w, tile, mailbox::DUMP_ROW_COUNT, spec.dump_rows)
        .unwrap();
    assert!(!spec.trace, "tracing is for role runs");
    dev.write32(&w, tile, mailbox::TRACE, 0).unwrap();
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

/// [`run`] for a [`Roles`] kernel.
fn run_roles(dev: &mut Dev<'_>, spec: &Run<'_>, roles: Roles<'_>) -> Outcome {
    use tt_isa::mailbox::role::Mailbox;

    // The `Dst` rows about to be dumped are zeroed before *anything* runs. In
    // order, that is simply the head of the unpack program (the math role may
    // read what the unpack role wrote into `Dst`); concurrently it is a run of
    // its own, with the semaphore initialisation, ahead of all three roles.
    let clear = if spec.clear_dst {
        dst_clear_prelude(spec.dump_rows)
    } else {
        Vec::new()
    };
    let (setup, unpack) = match spec.concurrent {
        None => (None, [clear.as_slice(), roles.unpack].concat()),
        Some(init) => {
            let mut setup = clear;
            for &(sem, value, max) in init {
                setup.push(tt_isa::sync::init(sem, value, max).unwrap());
            }
            (Some(setup), roles.unpack.to_vec())
        }
    };
    let programs: [&[Instruction]; 3] = [&unpack, roles.math, roles.pack];
    for (i, p) in programs.iter().enumerate() {
        assert!(
            p.len() as u32 <= mailbox::PROGRAM_MAX,
            "role {i} program is {} instructions; a program slot holds {}",
            p.len(),
            mailbox::PROGRAM_MAX
        );
    }
    assert!(spec.dump_rows <= mailbox::DUMP_MAX_ROWS);

    let tile = match spec.tile {
        Some((x, y)) => crate::backend::tile(dev, x, y),
        None => tensix_tile(),
    };
    let w: Window = dev.alloc_window(WindowKind::TwoMib).unwrap();
    dev.release_tensix_backend(&w, tile).unwrap();
    for (addr, data) in spec.stage {
        dev.write(&w, tile, *addr, data).unwrap();
    }

    let stage_role =
        |dev: &mut Dev<'_>, thread: usize, program: &[Instruction], dump: u32, traced: bool| {
            let mb = Mailbox::of(thread as u32);
            dev.write32(&w, tile, mb.status(), 0).unwrap();
            dev.write32(&w, tile, mb.thread_index(), thread as u32)
                .unwrap();
            dev.write32(&w, tile, mb.dst_access_fmt(), spec.dst_fmt)
                .unwrap();
            dev.write32(&w, tile, mb.program_len(), program.len() as u32)
                .unwrap();
            dev.write32(&w, tile, mb.dump_row_first(), 0).unwrap();
            dev.write32(&w, tile, mb.dump_row_count(), dump).unwrap();
            dev.write32(&w, tile, mb.trace(), u32::from(traced))
                .unwrap();
            dev.write(&w, tile, mb.program(), &program_bytes(program))
                .unwrap();
            for row in 0..dump {
                for col in 0..mailbox::DUMP_ROW_WORDS {
                    dev.write32(&w, tile, mb.dump_offset(row, col), SENTINEL)
                        .unwrap();
                }
            }
        };
    let dump_of = |thread: usize| if thread == 1 { spec.dump_rows } else { 0 };
    let start = |dev: &mut Dev<'_>, thread: usize| {
        let (core, image, at) = crate::firmware::ROLES[thread];
        dev.load_and_start(&w, tile, core, image, at).unwrap();
    };
    let wait = |dev: &mut Dev<'_>, thread: usize| -> Result<(), String> {
        let (core, _, _) = crate::firmware::ROLES[thread];
        let mb = Mailbox::of(thread as u32);
        match dev
            .wait_for_mailbox(&w, tile, mb.status(), mb.panic_code(), BUDGET, |s| {
                s == status::DONE
            })
            .unwrap()
        {
            Ok(_) => Ok(()),
            Err(WaitError::Panicked { code }) => Err(format!(
                "role {thread} ({}) firmware panicked, code {code}",
                core.name()
            )),
            Err(e) => Err(format!("role {thread} ({}): {e}", core.name())),
        }
    };

    if let Some(setup) = &setup {
        stage_role(dev, 0, setup, 0, false);
        start(dev, 0);
        if let Err(e) = wait(dev, 0) {
            panic!("setup run: {e}");
        }
    }
    for (thread, program) in programs.iter().enumerate() {
        stage_role(dev, thread, program, dump_of(thread), spec.trace);
    }
    if spec.trace {
        dev.configure_trace(&w, tile, mailbox::TRACE_BUFFER, mailbox::TRACE_BUFFER_BYTES)
            .unwrap();
    }
    if setup.is_some() {
        // Together, by one write to the soft-reset register: released one by
        // one, each after its image load, the first roles finish a short
        // program before the last has started, which is a sequential run by
        // another name (`Device::load_and_start_together`).
        dev.load_and_start_together(&w, tile, &crate::firmware::ROLES)
            .unwrap();
        let stuck: Vec<String> = (0..3).filter_map(|t| wait(dev, t).err()).collect();
        assert!(
            stuck.is_empty(),
            "concurrent roles did not all finish: {}",
            stuck.join("; ")
        );
    } else {
        // In order, each to completion.
        for thread in 0..3 {
            start(dev, thread);
            if let Err(e) = wait(dev, thread) {
                panic!("{e}");
            }
        }
    }

    // Leave the three cores the way the harness found them: held. They are
    // spinning after `DONE`, and a gate that runs next may depend on a core being
    // in reset (`Device::local_ram_read` refuses otherwise, rightly).
    for (core, _, _) in crate::firmware::ROLES.iter() {
        dev.set_core_reset(&w, tile, *core, true).unwrap();
    }

    let math = Mailbox::of(1);
    let mut dst = Vec::new();
    for row in 0..spec.dump_rows {
        for col in 0..mailbox::DUMP_ROW_WORDS {
            dst.push(dev.read32(&w, tile, math.dump_offset(row, col)).unwrap());
        }
    }
    let trace = if spec.trace {
        dev.read_trace(&w, tile, mailbox::TRACE_BUFFER).unwrap()
    } else {
        Vec::new()
    };
    let mut l1 = Vec::new();
    for (addr, len) in spec.read_back {
        let mut buf = vec![0u8; *len];
        dev.read(&w, tile, *addr, &mut buf).unwrap();
        l1.push(buf);
    }
    Outcome { dst, l1, trace }
}
