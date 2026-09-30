//! Running a three-role kernel on one Tensix tile.
//!
//! Stage the operands, stage each role's program in its slot
//! (`tt_isa::mailbox::PROGRAM_REGION`), start the three role images, wait for
//! all three to report `DONE`, and read back what was asked for.
//!
//! # Two schedules
//!
//! [`Schedule::InOrder`] runs unpack, then math, then pack, each to completion.
//! It needs no synchronisation and is enough for a kernel whose unpacker never
//! gets more than the two `Src` banks ahead.
//!
//! [`Schedule::Concurrent`] is what everything bigger needs. A setup program on
//! thread 0 zeroes the `Dst` rows that will be dumped and initialises the
//! kernel's semaphores; *then* all three roles are released by one write to
//! `SOFT_RESET_0` (`Device::load_and_start_together`). Both halves of that are
//! load-bearing:
//!
//! * A semaphore outlives the program that last used it on silicon, so a
//!   consumer released alongside its own `SEMINIT` could read a stale value
//!   first.
//! * Released one by one, each after loading its image, the first roles of a
//!   short kernel finish before the last has started -- a sequential run by
//!   another name, which hid a missing hand-off on silicon that ttsim showed at
//!   once (divergence row 53).

use std::time::{Duration, Instant};
use tt_device::core_control::WaitError;
use tt_device::tlb::WindowKind;
use tt_device::trace::TraceEvent;

use tt_device::{Device, Traffic, Transport, TransportError};
use tt_isa::backend::{self, Before};
use tt_isa::isa::Instruction;
use tt_isa::mailbox::role::Mailbox;
use tt_isa::mailbox::{self, status};
use tt_isa::noc::{NocCoord, NocId};
use tt_isa::sfpu;
use tt_isa::sync::Semaphore;
use tt_isa::tensix::Core;

/// A semaphore's starting `Value` and `Max`, set before a concurrent run.
pub type SemaphoreInit = (Semaphore, u8, u8);

/// The three role firmware images, as `(core, image, load address)` for
/// threads 0, 1 and 2: unpack on T0, math on T1, pack on T2, each linked at its
/// core's default reset PC.
pub type RoleImages<'a> = [(Core, &'a [u8], u64); 3];

/// How the three roles are run; see the module documentation.
#[derive(Copy, Clone, Debug)]
pub enum Schedule<'a> {
    InOrder,
    Concurrent(&'a [SemaphoreInit]),
}

/// Pre-filled into every dumped `Dst` datum, so a datum the firmware never
/// wrote is distinguishable from a computed zero.
pub const DUMP_SENTINEL: u32 = 0xDEAD_BEEF;

/// `RISC_DEST_ACCESS_CTRL_SEC*.fmt` 0: `float Dst32b[512][16]`.
pub const DST_FMT_FP32: u32 = 0;

/// A kernel, and what to stage and collect around it.
#[derive(Copy, Clone, Debug)]
pub struct Kernel<'a> {
    /// The programs of threads 0, 1 and 2.
    pub roles: [&'a [Instruction]; 3],
    pub schedule: Schedule<'a>,
    /// Bytes written to L1 first, as `(address, data)`.
    pub stage: &'a [(u64, &'a [u8])],
    /// L1 ranges read afterwards, as `(address, length)`.
    pub read_back: &'a [(u64, usize)],
    /// `Dst` rows the math role copies out through its mailbox, and which are
    /// zeroed before anything runs. At most `mailbox::DUMP_MAX_ROWS`.
    pub dump_rows: u32,
    /// Zero the dumped rows first. `Dst` has no reset value and survives
    /// everything the host can do to a tile from outside.
    pub clear_dst: bool,
    /// `RISC_DEST_ACCESS_CTRL_SEC*.fmt` for the firmware's `Dst` view.
    pub dst_fmt: u32,
    /// Record each role's `START`, `PUSHED` and `RETIRED` through the tile's
    /// timestamper. Silicon only: ttsim does not model the event stream
    /// (divergence row 54).
    pub trace: bool,
}

impl<'a> Kernel<'a> {
    /// A kernel with nothing staged, nothing read back and nothing dumped.
    pub fn new(roles: [&'a [Instruction]; 3], schedule: Schedule<'a>) -> Self {
        Kernel {
            roles,
            schedule,
            stage: &[],
            read_back: &[],
            dump_rows: 0,
            clear_dst: true,
            dst_fmt: DST_FMT_FP32,
            trace: false,
        }
    }
}

/// What a run produced.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    /// The dumped `Dst` rows, flattened as `row * 16 + column`.
    pub dst: Vec<u32>,
    /// One buffer per [`Kernel::read_back`] range, in order.
    pub l1: Vec<Vec<u8>>,
    /// The timestamper events of a traced run, in order.
    pub trace: Vec<TraceEvent>,
    /// Where the run's host time and transport traffic went.
    pub profile: Profile,
}

/// The phases of one [`run`], each with its wall-clock time and what it moved
/// across the transport. Always collected: it costs one `Instant` per phase.
///
/// Host time only. On ttsim it measures the simulator, not a chip, and is no
/// performance number (divergence row 11); on silicon `Wait` is the one phase
/// the device, rather than PCIe, can dominate.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub phases: Vec<(Phase, Duration, Traffic)>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Phase {
    /// Backend release and the kernel's L1 staging (operands).
    Stage,
    /// The semaphore-initialising setup program, loaded, run and waited for.
    Setup,
    /// Mailbox words and role programs written to their slots.
    Programs,
    /// Firmware images loaded and the role cores released.
    Launch,
    /// Polling the role mailboxes for `DONE`.
    Wait,
    /// `Dst` dump, trace and L1 read-back.
    ReadBack,
}

impl Profile {
    /// Total time and traffic of every phase named `phase`.
    pub fn of(&self, phase: Phase) -> (Duration, Traffic) {
        self.phases
            .iter()
            .filter(|p| p.0 == phase)
            .fold((Duration::ZERO, Traffic::default()), |(d, t), p| {
                (d + p.1, t + p.2)
            })
    }
}

/// Records consecutive phases against one device.
struct Stopwatch {
    at: Instant,
    traffic: Traffic,
    profile: Profile,
}

impl Stopwatch {
    fn start<T: Transport>(dev: &Device<T>) -> Self {
        Stopwatch {
            at: Instant::now(),
            traffic: dev.traffic(),
            profile: Profile::default(),
        }
    }

    /// Close the phase that has been running since the last lap.
    fn lap<T: Transport>(&mut self, dev: &Device<T>, phase: Phase) {
        let (now, traffic) = (Instant::now(), dev.traffic());
        self.profile
            .phases
            .push((phase, now - self.at, traffic - self.traffic));
        (self.at, self.traffic) = (now, traffic);
    }
}

impl Outcome {
    /// `Dst[row][column]`, by the firmware's dump indexing.
    pub fn dst_at(&self, row: usize, column: usize) -> u32 {
        self.dst[row * mailbox::DUMP_ROW_WORDS as usize + column]
    }
}

/// Why a run failed.
#[derive(Debug)]
pub enum RunError {
    Transport(TransportError),
    /// A role's program does not fit its slot.
    ProgramTooLong {
        thread: usize,
        len: usize,
    },
    /// More `Dst` rows asked for than the mailbox dump holds.
    TooManyDumpRows {
        rows: u32,
    },
    /// The concurrent schedule's setup program did not finish.
    Setup(WaitError),
    /// Roles that did not finish, as `(thread, core, why)`. The cores are held
    /// in reset again before this is returned.
    Roles(Vec<(usize, Core, WaitError)>),
    /// Something staged in L1 does not fit the region reserved for it.
    DoesNotFit {
        what: &'static str,
        bytes: u64,
        limit: u64,
    },
}

impl From<TransportError> for RunError {
    fn from(e: TransportError) -> Self {
        RunError::Transport(e)
    }
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Transport(e) => write!(f, "{e}"),
            RunError::ProgramTooLong { thread, len } => write!(
                f,
                "role {thread} program is {len} instructions; a program slot holds {}",
                mailbox::PROGRAM_MAX
            ),
            RunError::TooManyDumpRows { rows } => write!(
                f,
                "{rows} Dst rows asked for; the dump holds {}",
                mailbox::DUMP_MAX_ROWS
            ),
            RunError::Setup(e) => write!(f, "setup run: {e}"),
            RunError::DoesNotFit { what, bytes, limit } => write!(
                f,
                "{what} need {bytes} bytes of L1; the region holds {limit}"
            ),
            RunError::Roles(stuck) => {
                let parts: Vec<String> = stuck
                    .iter()
                    .map(|(t, core, e)| format!("role {t} ({}): {e}", core.name()))
                    .collect();
                write!(f, "{}", parts.join("; "))
            }
        }
    }
}

impl std::error::Error for RunError {}

/// Zero `rows` of FP32 `Dst` through the SFPU, whose `Dst` writes read back
/// correctly on both targets, then wait for it.
fn dst_clear(rows: u32) -> Vec<Instruction> {
    let mut p = Vec::new();
    p.extend(sfpu::load_f32(0, 0).expect("LReg 0 is writable"));
    for group in 0..rows.div_ceil(4) {
        for cols in [0, sfpu::DST_ODD_COLUMNS] {
            p.push(
                sfpu::store(0, sfpu::store_format::FP32, 0, (group * 4) | cols)
                    .expect("dump rows are inside Dst"),
            );
        }
    }
    p.push(backend::wait_for_sfpu(Before::EVERYTHING).expect("named bits"));
    p
}

fn program_bytes(program: &[Instruction]) -> Vec<u8> {
    program
        .iter()
        .flat_map(|i| i.word().to_le_bytes())
        .collect()
}

/// The setup program (for a concurrent schedule) and thread 0's program, with
/// the `Dst` clear in front of whichever runs first.
#[allow(clippy::type_complexity)]
fn assemble(kernel: &Kernel<'_>) -> Result<(Option<Vec<Instruction>>, Vec<Instruction>), RunError> {
    if kernel.dump_rows > mailbox::DUMP_MAX_ROWS {
        return Err(RunError::TooManyDumpRows {
            rows: kernel.dump_rows,
        });
    }
    let clear = if kernel.clear_dst && kernel.dump_rows > 0 {
        dst_clear(kernel.dump_rows)
    } else {
        Vec::new()
    };
    let (setup, unpack) = match kernel.schedule {
        Schedule::InOrder => (None, [clear.as_slice(), kernel.roles[0]].concat()),
        Schedule::Concurrent(init) => {
            let mut setup = clear;
            for &(sem, value, max) in init {
                setup.push(tt_isa::sync::init(sem, value, max).map_err(|_| {
                    RunError::Transport(TransportError::Hazard {
                        address: 0,
                        reason: "a semaphore value or max does not fit in four bits",
                    })
                })?);
            }
            (Some(setup), kernel.roles[0].to_vec())
        }
    };
    Ok((setup, unpack))
}

/// Run `kernel` on the Tensix tile at `tile`, waiting up to `budget` simulated
/// cycles for each role (`Device::wait_for_mailbox`; on silicon a floor of one
/// second applies).
///
/// The tile's backend is released first; the three role cores are held in
/// reset again afterwards, success or not.
pub fn run<T: Transport, N: NocId>(
    dev: &mut Device<T>,
    tile: NocCoord<N>,
    images: &RoleImages<'_>,
    kernel: &Kernel<'_>,
    budget: u64,
) -> Result<Outcome, RunError> {
    let (setup, unpack) = assemble(kernel)?;
    let programs: [&[Instruction]; 3] = [&unpack, kernel.roles[1], kernel.roles[2]];
    for (thread, p) in programs.iter().enumerate() {
        if p.len() > mailbox::PROGRAM_MAX as usize {
            return Err(RunError::ProgramTooLong {
                thread,
                len: p.len(),
            });
        }
    }

    let mut clock = Stopwatch::start(dev);
    let w = dev.alloc_window(WindowKind::TwoMib)?;
    dev.release_tensix_backend(&w, tile)?;
    for (addr, data) in kernel.stage {
        dev.write(&w, tile, *addr, data)?;
    }
    clock.lap(dev, Phase::Stage);

    let push_window = if dev.transport().is_simulated() {
        mailbox::SIM_PUSH_WINDOW
    } else {
        0
    };
    let stage_role = |dev: &mut Device<T>,
                      thread: usize,
                      program: &[Instruction],
                      dump: u32,
                      traced: bool|
     -> Result<(), RunError> {
        let mb = Mailbox::of(thread as u32);
        dev.write32(&w, tile, mb.status(), 0)?;
        dev.write32(&w, tile, mb.thread_index(), thread as u32)?;
        dev.write32(&w, tile, mb.dst_access_fmt(), kernel.dst_fmt)?;
        dev.write32(&w, tile, mb.program_len(), program.len() as u32)?;
        dev.write32(&w, tile, mb.dump_row_first(), 0)?;
        dev.write32(&w, tile, mb.dump_row_count(), dump)?;
        dev.write32(&w, tile, mb.trace(), u32::from(traced))?;
        dev.write32(&w, tile, mb.push_window(), push_window)?;
        dev.write(&w, tile, mb.program(), &program_bytes(program))?;
        for row in 0..dump {
            for col in 0..mailbox::DUMP_ROW_WORDS {
                dev.write32(&w, tile, mb.dump_offset(row, col), DUMP_SENTINEL)?;
            }
        }
        Ok(())
    };
    let wait = |dev: &mut Device<T>, thread: usize| -> Result<Result<(), WaitError>, RunError> {
        let mb = Mailbox::of(thread as u32);
        Ok(dev
            .wait_for_mailbox(&w, tile, mb.status(), mb.panic_code(), budget, |s| {
                s == status::DONE
            })?
            .map(|_| ()))
    };
    let hold_all = |dev: &mut Device<T>| -> Result<(), RunError> {
        for (core, _, _) in images.iter() {
            dev.set_core_reset(&w, tile, *core, true)?;
        }
        Ok(())
    };

    if let Some(setup) = &setup {
        stage_role(dev, 0, setup, 0, false)?;
        let (core, image, at) = images[0];
        dev.load_and_start(&w, tile, core, image, at)?;
        if let Err(e) = wait(dev, 0)? {
            hold_all(dev)?;
            return Err(RunError::Setup(e));
        }
        clock.lap(dev, Phase::Setup);
    }
    for (thread, program) in programs.iter().enumerate() {
        let dump = if thread == 1 { kernel.dump_rows } else { 0 };
        stage_role(dev, thread, program, dump, kernel.trace)?;
    }
    if kernel.trace {
        dev.configure_trace(&w, tile, mailbox::TRACE_BUFFER, mailbox::TRACE_BUFFER_BYTES)?;
    }
    clock.lap(dev, Phase::Programs);

    let mut stuck = Vec::new();
    if setup.is_some() {
        // Every role released before any is waited for, by one write.
        dev.load_and_start_together(&w, tile, images)?;
        clock.lap(dev, Phase::Launch);
        for (thread, (core, _, _)) in images.iter().enumerate() {
            if let Err(e) = wait(dev, thread)? {
                stuck.push((thread, *core, e));
            }
        }
    } else {
        for (thread, &(core, image, at)) in images.iter().enumerate() {
            dev.load_and_start(&w, tile, core, image, at)?;
            clock.lap(dev, Phase::Launch);
            let waited = wait(dev, thread)?;
            clock.lap(dev, Phase::Wait);
            if let Err(e) = waited {
                stuck.push((thread, core, e));
                break;
            }
        }
    }
    if setup.is_some() {
        clock.lap(dev, Phase::Wait);
    }
    // Leave the cores the way they were found: held. They spin after `DONE`,
    // and whatever runs next may depend on a core being in reset.
    hold_all(dev)?;
    if !stuck.is_empty() {
        return Err(RunError::Roles(stuck));
    }

    let math = Mailbox::of(1);
    let mut dst = Vec::with_capacity((kernel.dump_rows * mailbox::DUMP_ROW_WORDS) as usize);
    for row in 0..kernel.dump_rows {
        for col in 0..mailbox::DUMP_ROW_WORDS {
            dst.push(dev.read32(&w, tile, math.dump_offset(row, col))?);
        }
    }
    let trace = if kernel.trace {
        dev.read_trace(&w, tile, mailbox::TRACE_BUFFER)?
    } else {
        Vec::new()
    };
    let mut l1 = Vec::with_capacity(kernel.read_back.len());
    for (addr, len) in kernel.read_back {
        let mut buf = vec![0u8; *len];
        dev.read(&w, tile, *addr, &mut buf)?;
        l1.push(buf);
    }
    clock.lap(dev, Phase::ReadBack);
    Ok(Outcome {
        dst,
        l1,
        trace,
        profile: clock.profile,
    })
}

/// The three role images, loaded once and left running between kernels.
///
/// [`run`] loads three images and releases three cores for every kernel; on
/// silicon that is most of a short kernel's cost, and a matmul is many short
/// kernels. A `Resident` does it once: [`Resident::start`] loads the images with
/// a non-zero `mailbox::GENERATION`, which makes each runner wait for the next
/// generation instead of stopping, and [`Resident::run`] then costs only the
/// descriptor words, the programs and the operands.
///
/// Leaves state between kernels exactly as consecutive kernels in one program
/// would: every kernel this crate builds sets the configuration it depends on
/// (`matmul`'s preludes, the datapath's `*_config` programs), so nothing it
/// reads is inherited. A kernel that fails leaves the cores in an unknown
/// state; `run` refuses to go on after one, and the caller resets the tile and
/// starts again.
pub struct Resident<N: NocId> {
    tile: NocCoord<N>,
    window: tt_device::Window,
    generation: u32,
    poisoned: bool,
    /// What each role's program slot holds, as last written: a slot already
    /// holding the program is not written again. Consecutive chunks of one
    /// matmul usually share their programs, which were most of what a
    /// DRAM-resident matmul still sent over PCIe.
    slots: std::cell::RefCell<[Vec<Instruction>; 3]>,
}

impl<N: NocId> Resident<N> {
    /// Release the tile's backend, load the three role images and start them,
    /// and wait for each to acknowledge an empty first generation.
    pub fn start<T: Transport>(
        dev: &mut Device<T>,
        tile: NocCoord<N>,
        images: &RoleImages<'_>,
        budget: u64,
    ) -> Result<Self, RunError> {
        let window = dev.alloc_window(WindowKind::TwoMib)?;
        dev.release_tensix_backend(&window, tile)?;
        let r = Resident {
            tile,
            window,
            generation: 1,
            poisoned: false,
            slots: Default::default(),
        };
        for thread in 0..3 {
            r.stage(dev, thread, &[], 0, false, DST_FMT_FP32, false)?;
            dev.write32(&r.window, tile, Mailbox::of(thread as u32).generation(), 1)?;
        }
        dev.load_and_start_together(&r.window, tile, images)?;
        let stuck = r.wait(dev, &[0, 1, 2], images, budget)?;
        if !stuck.is_empty() {
            r.hold(dev, images)?;
            return Err(RunError::Roles(stuck));
        }
        Ok(r)
    }

    pub fn tile(&self) -> NocCoord<N> {
        self.tile
    }

    /// Run `kernel` on the resident roles. The same contract as [`run`], less
    /// the image loads and core releases.
    pub fn run<T: Transport>(
        &mut self,
        dev: &mut Device<T>,
        images: &RoleImages<'_>,
        kernel: &Kernel<'_>,
        budget: u64,
    ) -> Result<Outcome, RunError> {
        if self.poisoned {
            return Err(RunError::Transport(TransportError::Hazard {
                address: 0,
                reason: "a resident kernel failed; reset the tile and start again",
            }));
        }
        let (setup, unpack) = assemble(kernel)?;
        let programs: [&[Instruction]; 3] = [&unpack, kernel.roles[1], kernel.roles[2]];
        let mut clock = Stopwatch::start(dev);
        let tile = self.tile;
        for (addr, data) in kernel.stage {
            dev.l1_write(&self.window, tile, *addr, data)?;
        }
        clock.lap(dev, Phase::Stage);

        if let Some(setup) = &setup {
            self.generation += 1;
            self.stage(dev, 0, setup, 0, false, kernel.dst_fmt, true)?;
            let stuck = self.wait(dev, &[0], images, budget)?;
            if let Some((_, _, e)) = stuck.into_iter().next() {
                self.poisoned = true;
                return Err(RunError::Setup(e));
            }
            clock.lap(dev, Phase::Setup);
        }
        if kernel.trace {
            dev.configure_trace(
                &self.window,
                tile,
                mailbox::TRACE_BUFFER,
                mailbox::TRACE_BUFFER_BYTES,
            )?;
        }
        self.generation += 1;
        // Every role's program and descriptor is in place before any role's
        // generation moves, so the three start as close together as the host's
        // three final writes allow.
        for (thread, program) in programs.iter().enumerate() {
            let dump = if thread == 1 { kernel.dump_rows } else { 0 };
            self.stage(
                dev,
                thread,
                program,
                dump,
                kernel.trace,
                kernel.dst_fmt,
                false,
            )?;
        }
        clock.lap(dev, Phase::Programs);
        let go = |dev: &mut Device<T>, thread: usize| {
            let mb = Mailbox::of(thread as u32);
            dev.write32(&self.window, tile, mb.generation(), self.generation)
        };
        let stuck = if setup.is_some() {
            for thread in 0..3 {
                go(dev, thread)?;
            }
            clock.lap(dev, Phase::Launch);
            self.wait(dev, &[0, 1, 2], images, budget)?
        } else {
            // In order: each role runs to completion before the next moves,
            // exactly as `run` sequences them.
            let mut stuck = Vec::new();
            for thread in 0..3 {
                go(dev, thread)?;
                stuck = self.wait(dev, &[thread], images, budget)?;
                if !stuck.is_empty() {
                    break;
                }
            }
            stuck
        };
        clock.lap(dev, Phase::Wait);
        if !stuck.is_empty() {
            self.poisoned = true;
            return Err(RunError::Roles(stuck));
        }

        let math = Mailbox::of(1);
        let mut dst = Vec::with_capacity((kernel.dump_rows * mailbox::DUMP_ROW_WORDS) as usize);
        for row in 0..kernel.dump_rows {
            for col in 0..mailbox::DUMP_ROW_WORDS {
                dst.push(dev.read32(&self.window, tile, math.dump_offset(row, col))?);
            }
        }
        let trace = if kernel.trace {
            dev.read_trace(&self.window, tile, mailbox::TRACE_BUFFER)?
        } else {
            Vec::new()
        };
        let mut l1 = Vec::with_capacity(kernel.read_back.len());
        for (addr, len) in kernel.read_back {
            let mut buf = vec![0u8; *len];
            dev.l1_read(&self.window, tile, *addr, &mut buf)?;
            l1.push(buf);
        }
        clock.lap(dev, Phase::ReadBack);
        Ok(Outcome {
            dst,
            l1,
            trace,
            profile: clock.profile,
        })
    }

    /// Hold the three role cores in reset again.
    pub fn stop<T: Transport>(
        self,
        dev: &mut Device<T>,
        images: &RoleImages<'_>,
    ) -> Result<(), RunError> {
        self.hold(dev, images)
    }

    fn hold<T: Transport>(
        &self,
        dev: &mut Device<T>,
        images: &RoleImages<'_>,
    ) -> Result<(), RunError> {
        for (core, _, _) in images.iter() {
            dev.set_core_reset(&self.window, self.tile, *core, true)?;
        }
        Ok(())
    }

    /// Write role `thread`'s descriptor and program; with `go`, also move its
    /// generation so it runs.
    #[allow(clippy::too_many_arguments)]
    fn stage<T: Transport>(
        &self,
        dev: &mut Device<T>,
        thread: usize,
        program: &[Instruction],
        dump: u32,
        traced: bool,
        dst_fmt: u32,
        go: bool,
    ) -> Result<(), RunError> {
        let (w, tile) = (&self.window, self.tile);
        let mb = Mailbox::of(thread as u32);
        if program.len() > mailbox::PROGRAM_MAX as usize {
            return Err(RunError::ProgramTooLong {
                thread,
                len: program.len(),
            });
        }
        let push_window = if dev.transport().is_simulated() {
            mailbox::SIM_PUSH_WINDOW
        } else {
            0
        };
        dev.write32(w, tile, mb.thread_index(), thread as u32)?;
        dev.write32(w, tile, mb.dst_access_fmt(), dst_fmt)?;
        dev.write32(w, tile, mb.program_len(), program.len() as u32)?;
        dev.write32(w, tile, mb.dump_row_first(), 0)?;
        dev.write32(w, tile, mb.dump_row_count(), dump)?;
        dev.write32(w, tile, mb.trace(), u32::from(traced))?;
        dev.write32(w, tile, mb.push_window(), push_window)?;
        let mut slots = self.slots.borrow_mut();
        if !program.is_empty() && slots[thread] != program {
            // Forget the slot first: if the write fails, it holds neither.
            slots[thread].clear();
            dev.l1_write(w, tile, mb.program(), &program_bytes(program))?;
            slots[thread] = program.to_vec();
        }
        drop(slots);
        for row in 0..dump {
            for col in 0..mailbox::DUMP_ROW_WORDS {
                dev.write32(w, tile, mb.dump_offset(row, col), DUMP_SENTINEL)?;
            }
        }
        if go {
            dev.write32(w, tile, mb.generation(), self.generation)?;
        }
        Ok(())
    }

    /// Wait for each of `threads` to acknowledge the current generation.
    fn wait<T: Transport>(
        &self,
        dev: &mut Device<T>,
        threads: &[usize],
        images: &RoleImages<'_>,
        budget: u64,
    ) -> Result<Vec<(usize, Core, WaitError)>, RunError> {
        let mut stuck = Vec::new();
        for &thread in threads {
            let mb = Mailbox::of(thread as u32);
            let generation = self.generation;
            let r = dev.wait_for_mailbox(
                &self.window,
                self.tile,
                mb.ack(),
                mb.panic_code(),
                budget,
                |v| v == generation,
            )?;
            if let Err(e) = r {
                stuck.push((thread, images[thread].0, e));
            }
        }
        Ok(stuck)
    }
}
