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

use tt_device::core_control::WaitError;
use tt_device::tlb::WindowKind;
use tt_device::trace::TraceEvent;
use tt_device::{Device, Transport, TransportError};
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
    let programs: [&[Instruction]; 3] = [&unpack, kernel.roles[1], kernel.roles[2]];
    for (thread, p) in programs.iter().enumerate() {
        if p.len() > mailbox::PROGRAM_MAX as usize {
            return Err(RunError::ProgramTooLong {
                thread,
                len: p.len(),
            });
        }
    }

    let w = dev.alloc_window(WindowKind::TwoMib)?;
    dev.release_tensix_backend(&w, tile)?;
    for (addr, data) in kernel.stage {
        dev.write(&w, tile, *addr, data)?;
    }

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
    }
    for (thread, program) in programs.iter().enumerate() {
        let dump = if thread == 1 { kernel.dump_rows } else { 0 };
        stage_role(dev, thread, program, dump, kernel.trace)?;
    }
    if kernel.trace {
        dev.configure_trace(&w, tile, mailbox::TRACE_BUFFER, mailbox::TRACE_BUFFER_BYTES)?;
    }

    let mut stuck = Vec::new();
    if setup.is_some() {
        // Every role released before any is waited for, by one write.
        dev.load_and_start_together(&w, tile, images)?;
        for (thread, (core, _, _)) in images.iter().enumerate() {
            if let Err(e) = wait(dev, thread)? {
                stuck.push((thread, *core, e));
            }
        }
    } else {
        for (thread, &(core, image, at)) in images.iter().enumerate() {
            dev.load_and_start(&w, tile, core, image, at)?;
            if let Err(e) = wait(dev, thread)? {
                stuck.push((thread, core, e));
                break;
            }
        }
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
    Ok(Outcome { dst, l1, trace })
}
