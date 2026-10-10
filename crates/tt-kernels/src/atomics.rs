//! Running Tensix mutexes and L1 atomics, blocking forms included, safely.
//!
//! `ATGETM`, `ATCAS` and `ATINCGETPTR` block their Tensix thread until something
//! *else* happens: another thread releases a mutex, another agent makes an L1
//! word equal a value, a FIFO gains or loses an element. If nothing does, the
//! thread never takes another instruction, and the RISC-V core that pushed the
//! program cannot tell: the load it normally ends a run with
//! (`tt_firmware::tensix::wait_for_coprocessor`) stalls inside the memory
//! subsystem. [`runtime::run`](crate::runtime::run) would wait out its budget,
//! hold the cores and report a failure, leaving the Tensix thread parked.
//!
//! A blocking program is therefore only ever run as a [`GuardedProgram`], which
//! has three parts (`tt_isa::mailbox::guard`):
//!
//! * **a role-side deadline.** The role firmware polls a completion word that
//!   the program's own last instructions store, for a bounded number of polls,
//!   instead of issuing the drain load;
//! * **a host-visible blocked status.** When the deadline expires the role
//!   publishes [`RoleState::Blocked`] in its status word and keeps polling for a
//!   grace period, after which it gives up ([`RoleState::Panicked`] with
//!   `guard::ABANDONED`) without ever issuing the drain;
//! * **a release route.** The host can ask a blocked role's runner, with
//!   [`Launch::release`], to post semaphores (the way a thread parked in a
//!   `SEMWAIT` holding a mutex is let go), and can write the L1 words an `ATCAS`
//!   or `ATINCGETPTR` is polling directly.
//!
//! Programs are limited to [`guard::MAX_WORDS`] instructions, so a blocked
//! thread's instruction FIFO is never filled by the runner's own pushes (a push
//! into a full FIFO stalls the core with no timeout). A mutex a thread holds
//! can only be released by that thread, so a [`GuardedProgram`] must release
//! every mutex it takes ([`tt_isa::sync::mutex::check_scope`]); a program parked
//! *while holding* a mutex is let go by posting the semaphore it waits on.
//!
//! What this cannot do: free a thread parked on something the host cannot
//! satisfy -- an `ATGETM` of an index that does not exist (those cannot be
//! built: [`tt_isa::sync::mutex::Mutex`]) or an `ATCAS` whose compare value the host
//! never writes. The grace period then expires, the role reports
//! `guard::ABANDONED`, and the tile must be treated as wedged.

use crate::l1::{Buf, Plan, Requirements};
use crate::runtime::{word_bytes, RoleImages, SemaphoreInit};
use std::ops::Range;
use std::time::{Duration, Instant};
use tt_device::core_control::{WaitError, CYCLES_PER_POLL};
use tt_device::tlb::WindowKind;
use tt_device::{Device, Transport, TransportError};
use tt_isa::backend::{self, Before};
use tt_isa::isa::Instruction;
use tt_isa::mailbox::guard::{self, Guard, PollMode};
use tt_isa::mailbox::role::Mailbox;
use tt_isa::mailbox::{self, status};
use tt_isa::noc::{NocCoord, NocId};
use tt_isa::scalar::atomic::{self as atomic, AtomicError, Region16};
use tt_isa::scalar::{self, OffsetHalf, OffsetIncrement, TransferWidth};
use tt_isa::sync::{self, Semaphore};

/// Why a guarded run could not be built or did not finish.
#[derive(Debug)]
pub enum AtomicsError {
    Transport(TransportError),
    /// The program or its guard was refused before anything ran.
    Refused(String),
    /// A bad atomic operand.
    Operand(AtomicError),
    /// The setup run (semaphore initialisation) did not finish.
    Setup(WaitError),
    /// A role panicked.
    Panicked {
        thread: usize,
        code: u32,
    },
    /// The host's own budget expired waiting for `what`. The cores are left
    /// running; call [`Launch::abort`].
    HostTimeout {
        what: String,
    },
}

impl From<TransportError> for AtomicsError {
    fn from(e: TransportError) -> Self {
        AtomicsError::Transport(e)
    }
}

impl From<AtomicError> for AtomicsError {
    fn from(e: AtomicError) -> Self {
        AtomicsError::Operand(e)
    }
}

impl std::fmt::Display for AtomicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomicsError::Transport(e) => write!(f, "{e}"),
            AtomicsError::Refused(why) => write!(f, "refused: {why}"),
            AtomicsError::Operand(e) => write!(f, "{e}"),
            AtomicsError::Setup(e) => write!(f, "setup run: {e}"),
            AtomicsError::Panicked { thread, code } => {
                write!(f, "role {thread} panicked with code {code:#x}")
            }
            AtomicsError::HostTimeout { what } => write!(f, "host timed out waiting for {what}"),
        }
    }
}

impl std::error::Error for AtomicsError {}

/// Role-side polls per second of a guarded run on silicon.
///
/// Measured on card 0 by `step105_mutex::guard_poll_rate_calibration_light`:
/// Light 34.7 million polls a second, L1Word 32.8 million. A grace in seconds
/// must keep `seconds * rate` under `guard::MAX_POLLS` (2^30), which this rate
/// allows up to about 35 s. ttsim counts polls in simulated cycles and ignores
/// this.
pub const SILICON_POLLS_PER_SECOND: u32 = 30_000_000;

/// The deadline and grace of a guarded run, in role-side polls
/// (`guard::MAX_POLLS` at most; each poll is a fence and two L1 loads).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GuardSpec {
    deadline_polls: u32,
    grace_polls: u32,
    mode: PollMode,
}

impl GuardSpec {
    /// A deadline and grace given as seconds, at `polls_per_second` role-side
    /// polls a second (on silicon: [`SILICON_POLLS_PER_SECOND`]). Both are at
    /// least one poll.
    pub fn for_seconds(
        deadline_seconds: f64,
        grace_seconds: f64,
        polls_per_second: u32,
    ) -> Result<Self, AtomicsError> {
        let polls = |seconds: f64| ((seconds * f64::from(polls_per_second)) as u64).max(1);
        let (deadline, grace) = (polls(deadline_seconds), polls(grace_seconds));
        let max = u64::from(guard::MAX_POLLS);
        if deadline > max || grace > max {
            return Err(AtomicsError::Refused(format!(
                "{deadline_seconds} s and {grace_seconds} s at {polls_per_second} polls/s exceed {max} polls"
            )));
        }
        Self::new(deadline as u32, grace as u32)
    }

    /// The same bounds, polled another way ([`PollMode`]): how a silicon hang
    /// in the poll is bisected.
    pub const fn with_mode(mut self, mode: PollMode) -> Self {
        self.mode = mode;
        self
    }

    pub const fn mode(self) -> PollMode {
        self.mode
    }

    /// Bounds in polls, polled with [`PollMode::Light`], the default:
    /// [`PollMode::Full`] hangs beyond its first poll on silicon.
    pub fn new(deadline_polls: u32, grace_polls: u32) -> Result<Self, AtomicsError> {
        for (name, v) in [("deadline", deadline_polls), ("grace", grace_polls)] {
            if v == 0 || v > guard::MAX_POLLS {
                return Err(AtomicsError::Refused(format!(
                    "guard {name} of {v} polls is outside 1..={}",
                    guard::MAX_POLLS
                )));
            }
        }
        Ok(GuardSpec {
            deadline_polls,
            grace_polls,
            mode: PollMode::Light,
        })
    }

    pub const fn deadline_polls(self) -> u32 {
        self.deadline_polls
    }

    pub const fn grace_polls(self) -> u32 {
        self.grace_polls
    }
}

/// Whether any register field of `i` names one of `gprs` (half-register fields
/// by the register they fall in).
fn touches_gpr(i: Instruction, gprs: &[u32]) -> bool {
    i.def().fields().iter().any(|f| {
        let v = f.extract(i.word());
        if f.name().contains("HalfReg") {
            gprs.contains(&(v / 2))
        } else {
            f.name().contains("Reg") && gprs.contains(&v)
        }
    })
}

/// Whether `i` is a Sync Unit semaphore instruction that names `sem`.
fn touches_semaphore(i: Instruction, sem: Semaphore) -> bool {
    matches!(
        i.def().mnemonic(),
        "SEMINIT" | "SEMPOST" | "SEMGET" | "SEMWAIT"
    ) && i.operand("SemaphoreMask").unwrap_or(0) & sem.mask() != 0
}

/// A role's program with a completion post appended and a deadline attached:
/// the only way this module runs a blocking instruction.
#[derive(Clone, Debug)]
pub struct GuardedProgram {
    thread: usize,
    words: Vec<Instruction>,
    spec: GuardSpec,
    complete: Semaphore,
}

impl GuardedProgram {
    /// `body` for Tensix thread `thread`, followed by the epilogue that drains
    /// the Scalar Unit (C0, so every asynchronous atomic in the body has landed
    /// first) and then posts `complete`, the semaphore the runner polls.
    ///
    /// `complete` belongs to this program alone (declare it through
    /// `Requirements::semaphore`, one per guarded role); the runner takes the
    /// post down again.
    ///
    /// Refused: a body that holds a mutex at its end or takes one twice
    /// ([`tt_isa::sync::mutex::check_scope`]); one that names `complete`; one too
    /// long for `guard::MAX_WORDS` with the epilogue; or a bad thread.
    pub fn new(
        thread: usize,
        body: &[Instruction],
        spec: GuardSpec,
        complete: Semaphore,
    ) -> Result<Self, AtomicsError> {
        if thread >= 3 {
            return Err(AtomicsError::Refused(format!(
                "thread {thread} is not 0..3"
            )));
        }
        tt_isa::sync::mutex::check_scope(body)
            .map_err(|e| AtomicsError::Refused(format!("mutex scope: {e:?}")))?;
        if let Some(bad) = body.iter().find(|i| touches_semaphore(**i, complete)) {
            return Err(AtomicsError::Refused(format!(
                "{} names semaphore {}, which is the completion semaphore",
                bad.def().mnemonic(),
                complete.index()
            )));
        }
        let mut words = body.to_vec();
        words.push(atomic::consume());
        if spec.mode == PollMode::L1Word {
            // No semaphore: store the token to the role's completion word.
            let [reg_addr, reg_off, reg_data] = guard::RESERVED_GPRS;
            if let Some(bad) = body
                .iter()
                .find(|i| touches_gpr(**i, &guard::RESERVED_GPRS))
            {
                return Err(AtomicsError::Refused(format!(
                    "{} uses a GPR reserved for the L1-word epilogue ({:?})",
                    bad.def().mnemonic(),
                    guard::RESERVED_GPRS
                )));
            }
            let word = Guard::of(Mailbox::of(thread as u32)).complete_word();
            words.extend(backend::set_gpr(reg_addr, (word / 16) as u32).expect("reserved GPR"));
            words.extend(backend::set_gpr(reg_off, 0).expect("reserved GPR"));
            words.extend(backend::set_gpr(reg_data, guard::COMPLETE_TOKEN).expect("reserved GPR"));
            words.push(
                scalar::store_indirect_l1(
                    TransferWidth::Word,
                    OffsetHalf::new(reg_off * 2).expect("reserved GPR"),
                    OffsetIncrement::None,
                    reg_data,
                    reg_addr,
                )
                .expect("distinct reserved GPRs"),
            );
            words.push(atomic::consume());
        } else {
            words.push(sync::post(complete));
        }
        if words.len() > guard::MAX_WORDS as usize {
            return Err(AtomicsError::Refused(format!(
                "{} instructions with the completion post; a guarded run takes {}",
                words.len(),
                guard::MAX_WORDS
            )));
        }
        Ok(GuardedProgram {
            thread,
            words,
            spec,
            complete,
        })
    }

    /// The semaphore the program posts last.
    pub fn complete(&self) -> Semaphore {
        self.complete
    }

    pub fn thread(&self) -> usize {
        self.thread
    }

    /// The whole program, epilogue included.
    pub fn instructions(&self) -> &[Instruction] {
        &self.words
    }

    pub fn spec(&self) -> GuardSpec {
        self.spec
    }

    /// The host writes that arm this role's next run, the arming word last.
    pub fn arm_writes(&self) -> [(u64, u32); 13] {
        Guard::of(Mailbox::of(self.thread as u32)).arm_writes(
            self.spec.deadline_polls,
            self.spec.grace_polls,
            self.complete.index() as u32,
            self.spec.mode,
        )
    }

    /// This role's guard block.
    pub fn guard(&self) -> Guard {
        Guard::of(Mailbox::of(self.thread as u32))
    }
}

/// What one role runs.
#[derive(Copy, Clone, Debug)]
pub enum RoleProgram<'a> {
    /// Nothing: the role's runner pushes nothing and reports `DONE`.
    Idle,
    /// A program that cannot block, run as `runtime::run` would run it.
    Plain(&'a [Instruction]),
    /// A program that may block, under its deadline.
    Guarded(&'a GuardedProgram),
}

/// A concurrent run of up to three roles, some of them guarded.
#[derive(Copy, Clone, Debug)]
pub struct Spec<'a> {
    pub roles: [RoleProgram<'a>; 3],
    /// Initialised by a setup run before any role starts, as
    /// `runtime::Schedule::Concurrent`.
    pub semaphores: &'a [SemaphoreInit],
    /// Bytes written to L1 first, as `(address, data)`.
    pub stage: &'a [(u64, &'a [u8])],
}

/// A role's state as its status word reports it.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RoleState {
    /// Not started (status zero).
    Idle,
    Running,
    /// A guarded program has not completed within its deadline.
    Blocked,
    Done,
    Panicked(u32),
    /// Any other value: not one this runtime writes.
    Unknown(u32),
}

/// How long the host waits for something, on each target.
#[derive(Copy, Clone, Debug)]
pub struct Budget {
    /// Simulated cycles, on a simulator.
    pub sim_cycles: u64,
    /// Wall-clock time, on silicon.
    pub silicon: Duration,
}

impl Budget {
    pub const fn new(sim_cycles: u64, silicon: Duration) -> Self {
        Budget {
            sim_cycles,
            silicon,
        }
    }
}

/// A started run: the window and tile to poll, until [`Launch::finish`] or
/// [`Launch::abort`].
pub struct Launch<N: NocId> {
    tile: NocCoord<N>,
    window: tt_device::Window,
    cores: Vec<tt_isa::tensix::Core>,
    read_back: Vec<(u64, usize)>,
}

/// Start `spec` on `tile`: backend released, semaphores initialised by a setup
/// run, every role programmed, guarded roles armed, and all three released
/// together.
pub fn start<T: Transport, N: NocId>(
    dev: &mut Device<T>,
    tile: NocCoord<N>,
    images: &RoleImages<'_>,
    spec: &Spec<'_>,
    setup_budget: Budget,
) -> Result<Launch<N>, AtomicsError> {
    for (thread, role) in spec.roles.iter().enumerate() {
        let len = match role {
            RoleProgram::Idle => 0,
            RoleProgram::Plain(p) => p.len(),
            RoleProgram::Guarded(g) => {
                if g.thread != thread {
                    return Err(AtomicsError::Refused(format!(
                        "a program built for thread {} given to role {thread}",
                        g.thread
                    )));
                }
                g.words.len()
            }
        };
        if len > mailbox::PROGRAM_MAX as usize {
            return Err(AtomicsError::Refused(format!(
                "role {thread}: {len} instructions"
            )));
        }
    }
    let simulated = dev.transport().is_simulated();
    let push_window = if simulated {
        mailbox::SIM_PUSH_WINDOW
    } else {
        0
    };
    let w = dev.alloc_window(WindowKind::TwoMib)?;
    dev.release_tensix_backend(&w, tile)?;
    for (addr, data) in spec.stage {
        dev.write(&w, tile, *addr, data)?;
    }
    let stage_role =
        |dev: &mut Device<T>, thread: usize, program: &[Instruction]| -> Result<(), AtomicsError> {
            let mb = Mailbox::of(thread as u32);
            dev.write32(&w, tile, mb.status(), 0)?;
            let d = mailbox::Descriptor {
                thread_index: thread as u32,
                dst_access_fmt: crate::runtime::DST_FMT_FP32,
                program_len: program.len() as u32,
                push_window,
                ..Default::default()
            };
            for (at, v) in d.writes(mb) {
                dev.write32(&w, tile, at, v)?;
            }
            let words: Vec<u32> = program.iter().map(|i| i.word()).collect();
            dev.write(&w, tile, mb.program(), &word_bytes(&words))?;
            Ok(())
        };
    // The semaphores first, on thread 0, completed before any role starts: a
    // semaphore outlives the program that last used it on silicon.
    //
    // Every one of the eight is initialised, those the spec names to its value
    // and the rest to zero (maximum 15): nothing a run reads may depend on what
    // an earlier program or process left in the Sync Unit.
    let mut initial = [(0u8, 15u8); Semaphore::COUNT as usize];
    for &(sem, value, max) in spec.semaphores {
        initial[sem.index() as usize] = (value, max);
    }
    let mut setup = Vec::new();
    for (index, &(value, max)) in initial.iter().enumerate() {
        let sem = Semaphore::new(index as u8).expect("below the semaphore count");
        setup.push(
            tt_isa::sync::init(sem, value, max)
                .map_err(|e| AtomicsError::Refused(format!("semaphore init: {e:?}")))?,
        );
    }
    // The guard block of every role -- armed or not -- is cleared whole, with a
    // read-back fence, before anything runs: its semaphore snapshot, arming and
    // breadcrumb words survive between processes on silicon, and an unguarded
    // role's would otherwise show the previous run's semaphores to the host
    // (`Launch::semaphores`).
    for thread in 0..3u32 {
        let g = Guard::of(Mailbox::of(thread));
        for k in 0..Guard::BYTES / 4 {
            dev.write32(&w, tile, g.arm() + 4 * k, 0)?;
        }
        if dev.read32(&w, tile, g.arm() + Guard::BYTES - 4)? != 0 {
            return Err(AtomicsError::Refused(format!(
                "role {thread}'s guard block did not clear"
            )));
        }
    }
    stage_role(dev, 0, &setup)?;
    let (core0, image0, at0) = images[0];
    dev.load_and_start(&w, tile, core0, image0, at0)?;
    let mb0 = Mailbox::of(0);
    let done = dev.wait_for_mailbox(
        &w,
        tile,
        mb0.status(),
        mb0.panic_code(),
        setup_budget.sim_cycles,
        |s| s == status::DONE,
    )?;
    if let Err(e) = done {
        for (core, _, _) in images.iter() {
            dev.set_core_reset(&w, tile, *core, true)?;
        }
        return Err(AtomicsError::Setup(e));
    }
    for (thread, role) in spec.roles.iter().enumerate() {
        match role {
            RoleProgram::Idle => stage_role(dev, thread, &[])?,
            RoleProgram::Plain(p) => stage_role(dev, thread, p)?,
            RoleProgram::Guarded(g) => {
                stage_role(dev, thread, &g.words)?;
                // The arming word last (`Guard::arm_writes`).
                for (at, v) in g.arm_writes() {
                    dev.write32(&w, tile, at, v)?;
                }
            }
        }
    }
    dev.load_and_start_together(&w, tile, images)?;
    Ok(Launch {
        tile,
        window: w,
        cores: images.iter().map(|(c, _, _)| *c).collect(),
        read_back: Vec::new(),
    })
}

impl<N: NocId> Launch<N> {
    /// Ask for these L1 ranges to be returned by [`Launch::finish`].
    pub fn read_back(&mut self, ranges: &[(u64, usize)]) {
        self.read_back = ranges.to_vec();
    }

    /// Role `thread`'s state, from its status word.
    pub fn state<T: Transport>(
        &self,
        dev: &mut Device<T>,
        thread: usize,
    ) -> Result<RoleState, AtomicsError> {
        let mb = Mailbox::of(thread as u32);
        let s = dev.read32(&self.window, self.tile, mb.status())?;
        Ok(match s {
            0 => RoleState::Idle,
            s if s == status::RUNNING => RoleState::Running,
            s if s == guard::BLOCKED => RoleState::Blocked,
            s if s == status::DONE => RoleState::Done,
            s if s == status::PANICKED => {
                RoleState::Panicked(dev.read32(&self.window, self.tile, mb.panic_code())?)
            }
            s => RoleState::Unknown(s),
        })
    }

    /// How far role `thread`'s guarded run got, decoded: the breadcrumb stage,
    /// the runner's poll count, the raw status word and the state. Reads L1
    /// only, so it is safe on a role whose Tensix thread is hung.
    pub fn breadcrumbs<T: Transport>(
        &self,
        dev: &mut Device<T>,
        thread: usize,
    ) -> Result<String, AtomicsError> {
        let mb = Mailbox::of(thread as u32);
        let g = Guard::of(mb);
        let stage = self.read32(dev, g.stage())?;
        let polls = self.read32(dev, g.polls())?;
        let raw = self.read32(dev, mb.status())?;
        let arm = self.read32(dev, g.arm())?;
        let state = self.state(dev, thread)?;
        Ok(format!(
            "role {thread}: stage {stage:#x} ({}) polls {polls} status {raw:#010x} ({state:?}) arm word {arm:#x}",
            guard::stage::name(stage)
        ))
    }

    /// The eight semaphore values as role `thread`'s runner last saw them
    /// (`Guard::snapshot`): how the host observes what the threads have posted.
    /// Only a guarded role's runner publishes it.
    pub fn semaphores<T: Transport>(
        &self,
        dev: &mut Device<T>,
        thread: usize,
    ) -> Result<[u32; 8], AtomicsError> {
        let g = Guard::of(Mailbox::of(thread as u32));
        let word = self.read32(dev, g.snapshot())?;
        Ok(core::array::from_fn(|i| {
            guard::snapshot_value(word, i as u32)
        }))
    }

    pub fn read32<T: Transport>(&self, dev: &mut Device<T>, at: u64) -> Result<u32, AtomicsError> {
        Ok(dev.read32(&self.window, self.tile, at)?)
    }

    pub fn write32<T: Transport>(
        &self,
        dev: &mut Device<T>,
        at: u64,
        value: u32,
    ) -> Result<(), AtomicsError> {
        Ok(dev.write32(&self.window, self.tile, at, value)?)
    }

    pub fn read<T: Transport>(
        &self,
        dev: &mut Device<T>,
        at: u64,
        len: usize,
    ) -> Result<Vec<u8>, AtomicsError> {
        let mut buf = vec![0u8; len];
        dev.read(&self.window, self.tile, at, &mut buf)?;
        Ok(buf)
    }

    /// Release semaphores `mask` (bit `i` posts semaphore `i`) from role
    /// `thread`'s own runner -- the host route to let go a thread parked in a
    /// `SEMWAIT`. Waits until the runner has carried the request out.
    ///
    /// Only a guarded role's runner services this; asking an unguarded role is
    /// refused rather than left unanswered.
    pub fn release<T: Transport>(
        &self,
        dev: &mut Device<T>,
        thread: usize,
        mask: u32,
        budget: Budget,
    ) -> Result<(), AtomicsError> {
        if mask == 0 || mask & !guard::RELEASE_MASK != 0 {
            return Err(AtomicsError::Refused(format!(
                "release mask {mask:#x} must name semaphores 0..8"
            )));
        }
        let g = Guard::of(Mailbox::of(thread as u32));
        let before = self.read32(dev, g.released())?;
        self.write32(dev, g.release(), mask)?;
        self.wait(
            dev,
            budget,
            "the runner to carry out a release",
            |dev, l| Ok(l.read32(dev, g.release())? == 0 && l.read32(dev, g.released())? != before),
        )
    }

    /// Have role `thread`'s RISC-V core store `value` to the L1 word at `addr`
    /// (a word of the data arena, [`guard::POKE`]): an agent other than the host
    /// and every Tensix thread. This is the producer for an `ATCAS` or
    /// `ATINCGETPTR` parked on that word. **A Tensix thread cannot be:** the
    /// Scalar Unit executes one instruction at a time for all three threads
    /// (`ScalarUnit.md`), so a thread parked in either atomic keeps every other
    /// thread's scalar instruction -- `SETDMAREG` included -- from issuing.
    pub fn poke<T: Transport>(
        &self,
        dev: &mut Device<T>,
        thread: usize,
        addr: u64,
        value: u32,
        budget: Budget,
    ) -> Result<(), AtomicsError> {
        if addr % 4 != 0 || !tt_isa::l1::DATA.contains(addr, 4) {
            return Err(AtomicsError::Refused(format!(
                "poke address {addr:#x} is not a word of the data arena"
            )));
        }
        let g = Guard::of(Mailbox::of(thread as u32));
        let before = self.read32(dev, g.released())?;
        self.write32(dev, g.poke_addr(), addr as u32)?;
        self.write32(dev, g.poke_value(), value)?;
        self.write32(dev, g.release(), guard::POKE)?;
        self.wait(dev, budget, "the runner to carry out a poke", |dev, l| {
            Ok(l.read32(dev, g.release())? == 0 && l.read32(dev, g.released())? != before)
        })
    }

    /// Poll `done` until it is true or `budget` is spent, advancing simulated
    /// time between polls.
    pub fn wait<T: Transport>(
        &self,
        dev: &mut Device<T>,
        budget: Budget,
        what: &str,
        mut done: impl FnMut(&mut Device<T>, &Self) -> Result<bool, AtomicsError>,
    ) -> Result<(), AtomicsError> {
        let simulated = dev.transport().is_simulated();
        let end = Instant::now() + budget.silicon;
        let mut waited = 0u64;
        loop {
            if done(dev, self)? {
                return Ok(());
            }
            for thread in 0..3 {
                if let RoleState::Panicked(code) = self.state(dev, thread)? {
                    return Err(AtomicsError::Panicked { thread, code });
                }
            }
            let expired = if simulated {
                waited >= budget.sim_cycles
            } else {
                Instant::now() >= end
            };
            if expired {
                return Err(AtomicsError::HostTimeout {
                    what: what.to_string(),
                });
            }
            dev.tick(CYCLES_PER_POLL);
            waited += CYCLES_PER_POLL as u64;
        }
    }

    /// Wait for every role to report `DONE`, hold the cores in reset, and
    /// return the ranges asked for with [`Launch::read_back`].
    pub fn finish<T: Transport>(
        &self,
        dev: &mut Device<T>,
        budget: Budget,
    ) -> Result<Vec<Vec<u8>>, AtomicsError> {
        let waited = self.wait(dev, budget, "every role to finish", |dev, l| {
            for thread in 0..3 {
                if l.state(dev, thread)? != RoleState::Done {
                    return Ok(false);
                }
            }
            Ok(true)
        });
        if let Err(e) = waited {
            self.abort(dev)?;
            return Err(e);
        }
        let mut out = Vec::new();
        for &(at, len) in &self.read_back {
            out.push(self.read(dev, at, len)?);
        }
        self.abort(dev)?;
        Ok(out)
    }

    /// Hold every role core in reset. Does nothing for a Tensix thread already
    /// parked: only releasing what it waits on frees it.
    pub fn abort<T: Transport>(&self, dev: &mut Device<T>) -> Result<(), AtomicsError> {
        for core in &self.cores {
            dev.set_core_reset(&self.window, self.tile, *core, true)?;
        }
        Ok(())
    }
}

impl Launch<tt_isa::noc::Noc0> {
    /// [`Launch::abort`], then the repository's recovery for a tile whose Tensix
    /// threads a failed run left parked (`session::reset_thread_state`): the role
    /// cores post every semaphore a thread may wait on from the RISC-V side and
    /// reset each thread's state, which the backend reset pulse alone does not
    /// do (divergence row 65).
    ///
    /// **Every failure path of a blocking gate must end here.** A thread left
    /// parked behind a semaphore or mutex wait is not freed by the next
    /// process's ordinary reset, whose program queues behind it, so the next
    /// `harness::in_device` fails with a role timeout (`harness.rs`
    /// `run_roles`) that has nothing to do with that gate. On a simulator the
    /// recovery is a no-op.
    pub fn abort_and_recover<T: Transport>(
        &self,
        dev: &mut Device<T>,
        images: &RoleImages<'_>,
    ) -> Result<(), AtomicsError> {
        self.abort(dev)?;
        crate::session::reset_thread_state(dev, self.tile, images)
            .map_err(|e| AtomicsError::Refused(format!("recovery after abort: {e}")))
    }
}

/// A 16-byte block for atomics, declared like any other L1 scratch so the plan
/// keeps it clear of concurrent buffers and guards it with its own alignment.
pub fn block(req: &mut Requirements, name: &'static str, live: Range<u32>) -> Buf {
    req.scratch(name, 16, 16, live)
}

/// The checked [`Region16`] where `plan` put `buf`.
pub fn region(plan: &Plan, buf: Buf) -> Result<Region16, AtomicError> {
    Region16::new(plan.addr(buf))
}

/// The prologue that points `addr_reg` at `region`: a whole-GPR write, so no
/// earlier value survives.
pub fn point_at(addr_reg: u32, region: Region16) -> [Instruction; 2] {
    backend::set_gpr(addr_reg, region.gpr_value()).expect("GPR checked by the caller's operand")
}

/// A `Before::EVERYTHING` wait on the Scalar Unit: C0.
pub fn drain() -> Instruction {
    backend::wait_for_scalar(Before::EVERYTHING).expect("named bits")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tt_isa::sync::mutex::{self, Mutex};

    fn spec() -> GuardSpec {
        GuardSpec::new(100, 100).unwrap()
    }

    fn done() -> Semaphore {
        Semaphore::new(7).unwrap()
    }

    #[test]
    fn guard_spec_bounds() {
        assert!(GuardSpec::new(0, 1).is_err());
        assert!(GuardSpec::new(1, 0).is_err());
        assert!(GuardSpec::new(guard::MAX_POLLS + 1, 1).is_err());
        assert!(GuardSpec::new(guard::MAX_POLLS, guard::MAX_POLLS).is_ok());
    }

    #[test]
    fn specs_in_seconds_scale_with_the_poll_rate() {
        let s = GuardSpec::for_seconds(3.0, 120.0, 100).unwrap();
        assert_eq!((s.deadline_polls(), s.grace_polls()), (300, 12_000));
        let fast = GuardSpec::for_seconds(3.0, 120.0, 1_000).unwrap();
        assert_eq!(fast.deadline_polls(), 3_000);
        assert_eq!(
            GuardSpec::for_seconds(0.0, 0.0, 100)
                .unwrap()
                .deadline_polls(),
            1
        );
        assert!(GuardSpec::for_seconds(1e9, 1.0, 1_000_000).is_err());
    }

    #[test]
    fn a_guarded_program_ends_with_the_completion_post() {
        let m = Mutex::new(2).unwrap();
        let p = GuardedProgram::new(1, &[mutex::acquire(m), mutex::release(m)], spec(), done())
            .unwrap();
        let w = p.instructions();
        assert_eq!(w[0].def().mnemonic(), "ATGETM");
        assert_eq!(w[1].def().mnemonic(), "ATRELM");
        // A Scalar Unit drain, then the post of the completion semaphore, last.
        assert_eq!(w[w.len() - 2].def().mnemonic(), "STALLWAIT");
        assert_eq!(w[w.len() - 1].def().mnemonic(), "SEMPOST");
        assert_eq!(w[w.len() - 1].operand("SemaphoreMask"), Some(done().mask()));
        assert_eq!(p.arm_writes()[5].1, 7, "the runner is told which semaphore");
        assert_eq!(p.arm_writes()[12].0, p.guard().arm());
    }

    #[test]
    fn the_l1_word_epilogue_stores_the_token_and_reserves_its_registers() {
        let l1 = spec().with_mode(PollMode::L1Word);
        let p = GuardedProgram::new(2, &[backend::nop()], l1, done()).unwrap();
        let w = p.instructions();
        assert!(w.iter().all(|i| i.def().mnemonic() != "SEMPOST"));
        assert_eq!(w[w.len() - 2].def().mnemonic(), "STOREIND");
        assert_eq!(w[w.len() - 1].def().mnemonic(), "STALLWAIT");
        let mode = p
            .arm_writes()
            .iter()
            .find(|(at, _)| *at == p.guard().mode())
            .unwrap()
            .1;
        assert_eq!(mode, PollMode::L1Word as u32);
        let reserved = backend::set_gpr(guard::RESERVED_GPRS[1], 1).unwrap();
        assert!(GuardedProgram::new(2, &reserved, l1, done()).is_err());
        // The semaphore designs have no such reservation.
        assert!(GuardedProgram::new(2, &reserved, spec(), done()).is_ok());
    }

    #[test]
    fn unbalanced_long_and_completion_clashing_bodies_are_refused() {
        let m = Mutex::new(0).unwrap();
        assert!(GuardedProgram::new(0, &[mutex::acquire(m)], spec(), done()).is_err());
        assert!(GuardedProgram::new(3, &[], spec(), done()).is_err());
        let long = vec![backend::nop(); guard::MAX_WORDS as usize];
        assert!(GuardedProgram::new(0, &long, spec(), done()).is_err());
        let ok = vec![backend::nop(); guard::MAX_WORDS as usize - 2];
        assert!(GuardedProgram::new(0, &ok, spec(), done()).is_ok());
        let post = [sync::post(done())];
        assert!(GuardedProgram::new(0, &post, spec(), done()).is_err());
        let other = [sync::post(Semaphore::new(6).unwrap())];
        assert!(GuardedProgram::new(0, &other, spec(), done()).is_ok());
    }
}
