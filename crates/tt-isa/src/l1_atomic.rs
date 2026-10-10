//! Checked Scalar Unit L1 atomics: `ATCAS`, `ATSWAP`, `ATINCGET`, `ATINCGETPTR`.
//!
//! **UNVERIFIED on Blackhole.** Only Wormhole pages exist
//! (`WormholeB0/TensixTile/TensixCoprocessor/{ATCAS,ATSWAP,ATINCGET,ATINCGETPTR}.md`)
//! and the generated encodings are `Provenance::WormholeOnly`; ttsim refuses all
//! four (divergence row 77). Nothing here is evidence that Blackhole executes them
//! or that the layouts match. Each helper states the Wormhole page's semantics and
//! rejects what the page leaves undefined, so a silicon probe starts from a
//! program that cannot be wrong for a reason the page already explains.
//!
//! # What each instruction really does
//!
//! * [`compare_and_set`] (`ATCAS`) is **not** a conventional compare-and-swap. It
//!   blocks the Scalar Unit, retrying every ~15 cycles, until the whole 32-bit L1
//!   word *equals* the 4-bit compare value, then stores the 4-bit set value over
//!   the whole word. It reports nothing and cannot fail. A word that never equals
//!   the compare value wedges the thread, so a blocking use needs the deadline and
//!   release path in `tt_kernels::atomics`.
//! * [`masked_store`] (`ATSWAP`, four-GPR form only) is a store of up to 128
//!   bits at 16-bit granularity. It returns **nothing**: the name is a misnomer
//!   (`ATSWAP.md`: "nothing gets written back"). The unmasked halfwords keep
//!   their old values. The single-register form is excluded: measured on
//!   Blackhole its lane placement matches no rule.
//! * [`increment_and_get`] (`ATINCGET`) adds a GPR to the low `width` bits of an
//!   L1 word (wrapping inside the field, upper bits preserved) and later writes
//!   the word's **whole original value** to the same GPR. The write-back is
//!   asynchronous: wait for C0 ([`consume`]) before reading the GPR.
//! * [`fifo`] (`ATINCGETPTR`) waits until a pointer FIFO is not empty/not full
//!   and then advances a counter. It is a blocking instruction and needs counters
//!   in the low `width` bits with the rest zero (never free-running 32-bit ones).
//!
//! # Addresses
//!
//! Every instruction reads a GPR holding an L1 address in **16-byte units**. The
//! 16-byte block it names must lie in L1 (`>= TENSIX_SRAM_SIZE` is
//! `UndefinedBehavior`), so [`Region16`] validates alignment and extent before a
//! program can be built.

use crate::backend::{self, Before};
use crate::isa::generated::encode;
use crate::isa::Instruction;
use crate::tensix::L1_SIZE;

/// Why an operand was refused.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AtomicError {
    /// A GPR index outside `0..64`.
    BadGpr { index: u32 },
    /// The data/result register aliases the address register. The pages capture
    /// both at issue so the hardware may allow it, but the later write-back would
    /// silently change the address of the next operation.
    Alias { register: u32 },
    /// A four-GPR group not starting at a multiple of four.
    MisalignedGroup { index: u32 },
    /// An L1 address that is not 16-byte aligned.
    Unaligned { address: u64 },
    /// A 16-byte block outside L1.
    OutsideL1 { address: u64 },
    /// A compare/set value above fifteen.
    NotANibble { value: u32 },
    /// A word index above three.
    BadWord { index: u32 },
    /// A field width outside `1..=32`.
    BadFieldWidth { width: u32 },
    /// A FIFO counter width outside `1..=15`.
    BadFifoWidth { width: u32 },
    /// A batch of `2^log2` elements that does not fit the FIFO's capacity.
    BadBatch { log2: u32, capacity: u32 },
    /// A counter with bits above the FIFO's width: a free-running counter, which
    /// `ATINCGETPTR` cannot use.
    FreeRunningCounter { value: u32, mask: u32 },
    /// Counters that describe more elements than the capacity, or a count that
    /// is not a whole number of batches (so the full test could never fire).
    BadOccupancy { occupancy: u32 },
}

impl core::fmt::Display for AtomicError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            AtomicError::BadGpr { index } => write!(f, "GPR {index} does not exist"),
            AtomicError::Alias { register } => {
                write!(f, "GPR {register} is both the address and the data/result")
            }
            AtomicError::MisalignedGroup { index } => {
                write!(f, "four-GPR group starts at {index}, not a multiple of 4")
            }
            AtomicError::Unaligned { address } => {
                write!(f, "L1 address {address:#x} is not 16-byte aligned")
            }
            AtomicError::OutsideL1 { address } => {
                write!(f, "16-byte block at {address:#x} is outside L1")
            }
            AtomicError::NotANibble { value } => write!(f, "{value} does not fit four bits"),
            AtomicError::BadWord { index } => write!(f, "word {index} is outside 0..=3"),
            AtomicError::BadFieldWidth { width } => {
                write!(f, "field width {width} is outside 1..=32")
            }
            AtomicError::BadFifoWidth { width } => {
                write!(f, "FIFO counter width {width} is outside 1..=15")
            }
            AtomicError::BadBatch { log2, capacity } => {
                write!(f, "a batch of 2^{log2} does not fit capacity {capacity}")
            }
            AtomicError::FreeRunningCounter { value, mask } => write!(
                f,
                "counter {value:#x} has bits outside the width mask {mask:#x}"
            ),
            AtomicError::BadOccupancy { occupancy } => {
                write!(
                    f,
                    "occupancy {occupancy} is not a whole batch within capacity"
                )
            }
        }
    }
}

const fn gpr(index: u32) -> Result<u32, AtomicError> {
    if index < backend::GPR_COUNT {
        Ok(index)
    } else {
        Err(AtomicError::BadGpr { index })
    }
}

fn built(r: Result<Instruction, crate::isa::EncodeError>) -> Instruction {
    // Every field below was checked to fit before the encoder runs.
    r.expect("operands were checked to fit their fields")
}

/// A 16-byte-aligned block of L1, named by the GPR value the instructions take.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Region16 {
    address: u64,
}

impl Region16 {
    /// The block at byte `address`.
    pub const fn new(address: u64) -> Result<Region16, AtomicError> {
        if address % 16 != 0 {
            return Err(AtomicError::Unaligned { address });
        }
        if address + 16 > L1_SIZE {
            return Err(AtomicError::OutsideL1 { address });
        }
        Ok(Region16 { address })
    }

    pub const fn address(self) -> u64 {
        self.address
    }

    /// What the address GPR must hold.
    pub const fn gpr_value(self) -> u32 {
        (self.address / 16) as u32
    }

    /// Byte address of word `word` (0..=3) of the block.
    pub const fn word(self, word: Word) -> u64 {
        self.address + 4 * word.index() as u64
    }
}

/// One of the four words of a 16-byte block (the `Ofs` field).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Word(u8);

impl Word {
    pub const W0: Word = Word(0);
    pub const W1: Word = Word(1);
    pub const W2: Word = Word(2);
    pub const W3: Word = Word(3);

    pub const fn new(index: u32) -> Result<Word, AtomicError> {
        if index < 4 {
            Ok(Word(index as u8))
        } else {
            Err(AtomicError::BadWord { index })
        }
    }

    pub const fn index(self) -> u32 {
        self.0 as u32
    }
}

/// A four-bit compare or set value.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Nibble(u8);

impl Nibble {
    pub const fn new(value: u32) -> Result<Nibble, AtomicError> {
        if value < 16 {
            Ok(Nibble(value as u8))
        } else {
            Err(AtomicError::NotANibble { value })
        }
    }

    pub const fn value(self) -> u32 {
        self.0 as u32
    }
}

/// `ATCAS`: block until the 32-bit word equals `compare`, then store `set`.
///
/// Waits, retrying, for **every bit** of the word to equal the 4-bit compare
/// value (so its upper 28 bits must be zero), and then replaces the *whole word*
/// with the 4-bit set value. Never returns a status. The caller must make sure
/// something else will make the word equal `compare` -- see the module notes.
pub fn compare_and_set(
    set: Nibble,
    compare: Nibble,
    word: Word,
    addr_reg: u32,
) -> Result<Instruction, AtomicError> {
    let addr_reg = gpr(addr_reg)?;
    Ok(built(encode::atcas(
        set.value(),
        compare.value(),
        word.index(),
        addr_reg,
    )))
}

/// A mask of the eight 16-bit halfwords of a 16-byte block (bit `i` selects
/// halfword `i`, bytes `2i..2i+2`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HalfwordMask(pub u8);

impl HalfwordMask {
    pub const ALL: HalfwordMask = HalfwordMask(0xff);

    pub const fn bits(self) -> u32 {
        self.0 as u32
    }
}

/// `ATSWAP`, four-GPR form: write the masked halfwords of a 16-byte block from
/// the four consecutive GPRs starting at `group` (a multiple of four); return
/// nothing.
///
/// The write is captured at issue and lands later: other L1 clients see it only
/// after it reaches L1, so drain C0 ([`consume`]) before handing the block to
/// another agent. Unmasked halfwords are not touched.
///
/// The single-register form (`SingleDataReg` set) is **not representable**.
/// On Blackhole silicon its destination lanes follow neither the Wormhole page
/// nor any consistent rule (the evidence is
/// `step106_l1_atomics::silicon_gates::atswap_single_form_sweep_diagnostic`),
/// so it takes the `[-]` exit and `SingleDataReg` is always encoded as zero.
pub fn masked_store(
    mask: HalfwordMask,
    group: u32,
    addr_reg: u32,
) -> Result<Instruction, AtomicError> {
    let addr_reg = gpr(addr_reg)?;
    let group = gpr(group)?;
    if group % 4 != 0 {
        return Err(AtomicError::MisalignedGroup { index: group });
    }
    Ok(built(encode::atswap(0, mask.bits(), group, addr_reg)))
}

/// A field width of 1 to 32 bits (`ATINCGET`'s `IntWidth + 1`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FieldWidth(u8);

impl FieldWidth {
    pub const fn new(width: u32) -> Result<FieldWidth, AtomicError> {
        if width >= 1 && width <= 32 {
            Ok(FieldWidth(width as u8))
        } else {
            Err(AtomicError::BadFieldWidth { width })
        }
    }

    pub const fn bits(self) -> u32 {
        self.0 as u32
    }

    /// The mask of the field's bits.
    pub const fn mask(self) -> u32 {
        if self.0 == 32 {
            u32::MAX
        } else {
            (1u32 << self.0) - 1
        }
    }
}

/// `ATINCGET`: add GPR `inout` to the low `width` bits of the L1 word (wrapping
/// within the field, higher bits preserved) and return the word's whole original
/// value to GPR `inout`.
///
/// The increment is read at issue; the GPR is overwritten later by the old value.
/// Follow with [`consume`] before using it. `inout` must not be the address
/// register.
pub fn increment_and_get(
    width: FieldWidth,
    word: Word,
    inout: u32,
    addr_reg: u32,
) -> Result<Instruction, AtomicError> {
    let (inout, addr_reg) = (gpr(inout)?, gpr(addr_reg)?);
    if inout == addr_reg {
        return Err(AtomicError::Alias { register: inout });
    }
    Ok(built(encode::atincget(
        width.bits() - 1,
        word.index(),
        inout,
        addr_reg,
    )))
}

/// The C0 wait that must separate an asynchronous Scalar Unit atomic from
/// whatever reads its GPR result or hands its memory write to another agent.
pub fn consume() -> Instruction {
    backend::wait_for_scalar(Before::EVERYTHING).expect("named bits")
}

/// A pointer FIFO's geometry, for `ATINCGETPTR`.
///
/// Counters are `width` bits wide (`IntWidth`), so a FIFO of `2^(width-1)`
/// elements; the instruction pushes or pops `2^batch_log2` elements at a time.
/// Every push and pop of one FIFO must use the same batch (or a multiple), since
/// the hardware only tests "not full" and "not empty" and the full test is exact
/// only when occupancy stays a whole number of batches.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FifoGeometry {
    width: u8,
    batch_log2: u8,
}

impl FifoGeometry {
    pub const MAX_WIDTH: u32 = 15;

    pub const fn new(width: u32, batch_log2: u32) -> Result<FifoGeometry, AtomicError> {
        if width < 1 || width > Self::MAX_WIDTH {
            return Err(AtomicError::BadFifoWidth { width });
        }
        let capacity = 1u32 << (width - 1);
        // A batch larger than the capacity could never be pushed whole.
        if batch_log2 >= width {
            return Err(AtomicError::BadBatch {
                log2: batch_log2,
                capacity,
            });
        }
        Ok(FifoGeometry {
            width: width as u8,
            batch_log2: batch_log2 as u8,
        })
    }

    /// Elements the FIFO holds.
    pub const fn capacity(self) -> u32 {
        1 << (self.width - 1)
    }

    pub const fn batch(self) -> u32 {
        1 << self.batch_log2
    }

    pub const fn width(self) -> u32 {
        self.width as u32
    }

    pub const fn batch_log2(self) -> u32 {
        self.batch_log2 as u32
    }

    /// The mask of valid counter bits.
    pub const fn counter_mask(self) -> u32 {
        (1u32 << self.width) - 1
    }
}

/// Which counter an `ATINCGETPTR` advances.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FifoSide {
    /// Wait until not full; advances the write counter (word 1).
    Push,
    /// Wait until not empty; advances the read counter (word 0).
    Pop,
}

/// What an `ATINCGETPTR` does once its wait is over.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FifoAction {
    /// Only wait (`NoIncr`); the counter's current value is returned.
    Wait,
    /// Advance the counter by one batch; the value before is returned.
    Advance,
}

/// `ATINCGETPTR`: wait for room (push) or data (pop) in the FIFO whose counters
/// are at the 16-byte block the address GPR names, optionally advance, and
/// return the counter's previous value to `result_reg`.
///
/// Blocking: if the FIFO stays full (push) or empty (pop) the thread waits for
/// ever. `result_reg` must not be the address register (the address is re-read
/// on every retry).
pub fn fifo(
    geometry: FifoGeometry,
    side: FifoSide,
    action: FifoAction,
    result_reg: u32,
    addr_reg: u32,
) -> Result<Instruction, AtomicError> {
    let (result_reg, addr_reg) = (gpr(result_reg)?, gpr(addr_reg)?);
    if result_reg == addr_reg {
        return Err(AtomicError::Alias {
            register: result_reg,
        });
    }
    let (no_incr, log2) = match action {
        FifoAction::Wait => (1, 0),
        FifoAction::Advance => (0, geometry.batch_log2()),
    };
    Ok(built(
        encode::Atincgetptr::ZERO
            .no_incr(no_incr)
            .incr_log2(log2)
            .int_width(geometry.width())
            .ofs(match side {
                FifoSide::Pop => 0,
                FifoSide::Push => 1,
            })
            .result_reg(result_reg)
            .addr_reg(addr_reg)
            .encode(),
    ))
}

/// The in-L1 state of a pointer FIFO: `Rd`, `Wr` and two padding words.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FifoCounters {
    pub read: u32,
    pub write: u32,
}

impl FifoCounters {
    /// Counters for `geometry`, refusing a free-running counter (bits above the
    /// width), an occupancy above the capacity, and one that is not a whole
    /// number of batches.
    pub const fn new(
        geometry: FifoGeometry,
        read: u32,
        write: u32,
    ) -> Result<FifoCounters, AtomicError> {
        let mask = geometry.counter_mask();
        if read & !mask != 0 {
            return Err(AtomicError::FreeRunningCounter { value: read, mask });
        }
        if write & !mask != 0 {
            return Err(AtomicError::FreeRunningCounter { value: write, mask });
        }
        let occupancy = write.wrapping_sub(read) & mask;
        if occupancy > geometry.capacity() || occupancy % geometry.batch() != 0 {
            return Err(AtomicError::BadOccupancy { occupancy });
        }
        Ok(FifoCounters { read, write })
    }

    /// The four words to write at the block, padding zero.
    pub const fn words(self) -> [u32; 4] {
        [self.read, self.write, 0, 0]
    }
}

/// The role-side deadline, host-visible blocked status and release route that
/// make a blocking instruction safe to run (`tt-firmware/src/corpus.rs`).
///
/// A Tensix thread parked in a Wait Gate (`ATGETM` behind a mutex that is held,
/// an `ATCAS` whose compare never holds, an `ATINCGETPTR` on a full or empty
/// FIFO) takes no further instruction, and the RISC-V `wait_for_coprocessor`
/// load that normally follows a program stalls inside the memory subsystem with
/// no software timeout. A **guarded** run therefore replaces that wait with a
/// bounded poll of a **completion semaphore** that the program's own last
/// instruction posts (`SyncUnit.md`: a RISC-V core can `lw` any semaphore),
/// reports [`BLOCKED`] when the bound expires, and keeps polling for a release
/// request ([`Guard::release`]) and the completion. A semaphore rather than an
/// L1 word because the program has no store ttsim runs (`STOREIND`, divergence
/// row 77's neighbour) and a post is the Sync Unit's own ordered completion.
///
/// Each poll also publishes a snapshot of all eight semaphores
/// ([`Guard::snapshot`]), so the host can observe what the threads have posted:
/// the gates use a semaphore per thread as a "this thread got here" marker.
///
/// The block sits at a fixed offset inside each role mailbox, away from every
/// word [`crate::mailbox::Descriptor`] writes. It is **one-shot**: the runner
/// clears the arming word as soon as it reads it, so a stale arming word left by
/// an earlier process (L1 survives on silicon) can never turn a later ordinary
/// run into a guarded one.
pub mod guard {
    use crate::mailbox::role::Mailbox;

    /// Offset of the guard block from a role mailbox's base.
    pub const OFFSET: u64 = 0x100;
    /// Written to the arming word by the host to guard the next run.
    pub const ARMED: u32 = 0x4755_4152;
    /// The role's [`STATUS`](crate::mailbox::STATUS) while a guarded program has
    /// not completed within its deadline.
    pub const BLOCKED: u32 = 0x5747_b10c;
    /// The panic code of a runner that gave up after the grace period.
    pub const ABANDONED: u32 = 0x4755_0001;
    /// The panic code of a refused guarded program.
    pub const REFUSED: u32 = 0x4755_0002;
    /// The longest program a guarded run takes. A blocked thread's instruction
    /// FIFO is the only place later pushes can wait, and a push into a full
    /// FIFO stalls the pushing core with no timeout; the first FIFO holds 28
    /// (`PushTensixInstruction.md:15`), so the program, completion post
    /// included, must stay under that.
    pub const MAX_WORDS: u32 = 24;
    /// The largest deadline or grace count the host may ask for (polls).
    pub const MAX_POLLS: u32 = 1 << 30;
    /// Semaphores a release request may post: the low eight bits.
    pub const RELEASE_MASK: u32 = 0xff;
    /// Set in a release request: before posting, the runner stores
    /// [`Guard::poke_value`] to the L1 word at [`Guard::poke_addr`] -- a store by
    /// this role's RISC-V core, an agent other than the host and every Tensix
    /// thread, which is what frees an `ATCAS` or `ATINCGETPTR` that is polling
    /// that word. The address must be a word in the data arena.
    pub const POKE: u32 = 1 << 31;

    /// The guard block of one role.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Guard {
        base: u64,
    }

    impl Guard {
        pub const fn of(mailbox: Mailbox) -> Guard {
            Guard {
                base: mailbox.base() + OFFSET,
            }
        }
        /// One-shot arming word.
        pub const fn arm(self) -> u64 {
            self.base
        }
        /// Polls before the runner reports [`BLOCKED`].
        pub const fn deadline(self) -> u64 {
            self.base + 0x04
        }
        /// Further polls, once blocked, before the runner gives up.
        pub const fn grace(self) -> u64 {
            self.base + 0x08
        }
        /// Polls the runner spent waiting (diagnostic; written at the end).
        pub const fn polls(self) -> u64 {
            self.base + 0x0c
        }
        /// Which semaphore (0..8) the program posts when it has completed.
        pub const fn complete_semaphore(self) -> u64 {
            self.base + 0x10
        }
        /// The eight semaphore values, one nibble each (semaphore `i` in bits
        /// `4i..4i+4`), as of the runner's last poll.
        pub const fn snapshot(self) -> u64 {
            self.base + 0x14
        }
        /// Host-written release request; the runner consumes it.
        pub const fn release(self) -> u64 {
            self.base + 0x18
        }
        /// Release requests the runner has carried out.
        pub const fn released(self) -> u64 {
            self.base + 0x1c
        }
        /// Breadcrumb: the last [`stage`] the runner reached (the host zeroes it
        /// when arming). Written, and fenced, at each step of a guarded run.
        pub const fn stage(self) -> u64 {
            self.base + 0x20
        }
        /// The [`PollMode`] the host asked for.
        pub const fn mode(self) -> u64 {
            self.base + 0x24
        }
        /// The L1 address a [`POKE`] request stores to.
        pub const fn poke_addr(self) -> u64 {
            self.base + 0x28
        }
        /// The value a [`POKE`] request stores.
        pub const fn poke_value(self) -> u64 {
            self.base + 0x2c
        }
        /// The completion word of [`PollMode::L1Word`]: 16-byte aligned, so the
        /// program stores to it with a zero offset.
        pub const fn complete_word(self) -> u64 {
            self.base + 0x30
        }
        /// Bytes the block spans.
        pub const BYTES: u64 = 0x40;

        /// Everything the host writes to arm the next run: `(address, value)`.
        pub const fn arm_writes(
            self,
            deadline_polls: u32,
            grace_polls: u32,
            complete_semaphore: u32,
            mode: PollMode,
        ) -> [(u64, u32); 13] {
            [
                (self.snapshot(), 0),
                (self.release(), 0),
                (self.released(), 0),
                (self.deadline(), deadline_polls),
                (self.grace(), grace_polls),
                (self.complete_semaphore(), complete_semaphore),
                (self.polls(), 0),
                (self.stage(), 0),
                (self.mode(), mode as u32),
                (self.complete_word(), 0),
                (self.poke_addr(), 0),
                (self.poke_value(), 0),
                // Last, so the runner never sees a half-written block.
                (self.arm(), ARMED),
            ]
        }
    }

    /// How the runner learns that a guarded program has completed.
    ///
    /// All three keep the deadline, the BLOCKED status and the release path.
    /// They differ only in what the runner loads each poll, so a silicon hang
    /// can be bisected between them.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    #[repr(u32)]
    pub enum PollMode {
        /// Each poll loads all eight Tensix semaphores back to back and
        /// publishes them ([`Guard::snapshot`]); the program posts a semaphore.
        ///
        /// **Unsafe beyond the first poll on silicon; kept as a recorded
        /// finding, not for use.** On Blackhole card 0 a role that had to wait
        /// (a blocked program) showed stage `FIRST_POLL_DONE`, status Running
        /// and a live poll count of 0 for 3 s (`guard_poll_rate_calibration_full`,
        /// run id: to be filled in by the coordinator), while short programs
        /// that were already complete at the first poll finished. The first
        /// poll has a fence between its loads and later ones do not, which is
        /// consistent with the `ManualTTSync.md:18` rule against overlapping
        /// loads in `PC_BUF_BASE..+0xFFFF`. Not proven. Use [`PollMode::Light`].
        Full = 0,
        /// Each poll loads only the completion semaphore, with the load's result
        /// consumed (`andi`) before anything else loads, and every load of the
        /// periodic snapshot is consumed the same way, so one load is in flight
        /// at a time (`ManualTTSync.md:18`). The full snapshot is taken every
        /// [`LIGHT_SNAPSHOT_EVERY`] polls and once at completion. **The
        /// default on silicon and ttsim.** Polling a semaphore is the page's
        /// own mechanism (`ManualTTSync.md`, "Tensix semaphores": reads never
        /// contend with other agents).
        Light = 1,
        /// No semaphore loads at all: the program's epilogue stores
        /// [`COMPLETE_TOKEN`] to [`Guard::complete_word`] with `STOREIND`
        /// (`ManualTTSync.md`: a viable polled address space) and the runner
        /// polls that L1 word. Silicon only (ttsim refuses `STOREIND`), and no
        /// snapshot.
        L1Word = 2,
    }

    /// Polls between full snapshots in [`PollMode::Light`].
    pub const LIGHT_SNAPSHOT_EVERY: u32 = 64;
    /// What a [`PollMode::L1Word`] program stores at its end.
    pub const COMPLETE_TOKEN: u32 = 0x474f_4f44;
    /// GPRs the L1-word epilogue uses (address, zero offset, token).
    pub const RESERVED_GPRS: [u32; 3] = [61, 62, 63];

    /// Breadcrumb values of [`Guard::stage`], in the order a run passes them.
    /// `FIRST_LOAD + i` is written just before the first poll's load of
    /// semaphore `i`, so a hang inside a load names which one.
    pub mod stage {
        /// The runner read and consumed the arming word.
        pub const ARM_SEEN: u32 = 1;
        /// Deadline, grace and completion semaphore passed their checks.
        pub const PARAMS_OK: u32 = 2;
        /// Any post left on the completion semaphore (or word) was cleared.
        pub const CLEARED: u32 = 3;
        /// Every word of the program was pushed.
        pub const PUSHED: u32 = 4;
        /// The poll loop was entered.
        pub const POLL_ENTERED: u32 = 5;
        /// The first poll's snapshot pass finished.
        pub const FIRST_POLL_DONE: u32 = 6;
        /// The deadline expired and BLOCKED was published.
        pub const BLOCKED: u32 = 7;
        /// The completion was seen and the poll loop left.
        pub const POLL_DONE: u32 = 8;
        /// `wait_for_coprocessor` returned.
        pub const DRAINED: u32 = 9;
        /// Written just before the first poll's load of semaphore `i`.
        pub const FIRST_LOAD: u32 = 0x100;

        /// A readable name.
        pub fn name(stage: u32) -> &'static str {
            match stage {
                0 => "not armed or not reached",
                ARM_SEEN => "ARM_SEEN",
                PARAMS_OK => "PARAMS_OK",
                CLEARED => "CLEARED",
                PUSHED => "PUSHED",
                POLL_ENTERED => "POLL_ENTERED",
                FIRST_POLL_DONE => "FIRST_POLL_DONE",
                BLOCKED => "BLOCKED",
                POLL_DONE => "POLL_DONE",
                DRAINED => "DRAINED",
                s if (FIRST_LOAD..FIRST_LOAD + 8).contains(&s) => "in the first poll's load",
                _ => "unknown",
            }
        }
    }

    /// The value of semaphore `index` in a [`Guard::snapshot`] word.
    pub const fn snapshot_value(snapshot: u32, index: u32) -> u32 {
        (snapshot >> (4 * index)) & 0xf
    }

    const _: () = assert!(OFFSET % 16 == 0 && OFFSET > crate::mailbox::MAILBOX_SIZE);
    const _: () = assert!(Guard::BYTES + OFFSET < 0x2000);
    const _: () = assert!(Guard::BYTES % 16 == 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_checks_alignment_and_extent() {
        assert!(Region16::new(0x1_0000).is_ok());
        assert_eq!(
            Region16::new(0x1_0004),
            Err(AtomicError::Unaligned { address: 0x1_0004 })
        );
        assert!(Region16::new(L1_SIZE - 16).is_ok());
        assert_eq!(
            Region16::new(L1_SIZE),
            Err(AtomicError::OutsideL1 { address: L1_SIZE })
        );
        assert_eq!(Region16::new(0x1_0010).unwrap().gpr_value(), 0x1001);
    }

    #[test]
    fn cas_fields_land_where_the_wormhole_page_says() {
        let i = compare_and_set(
            Nibble::new(0xa).unwrap(),
            Nibble::new(0x5).unwrap(),
            Word::new(3).unwrap(),
            9,
        )
        .unwrap();
        assert_eq!(i.word(), 0x64 << 24 | 0xa << 18 | 0x5 << 14 | 3 << 12 | 9);
        assert!(Nibble::new(16).is_err());
        assert!(Word::new(4).is_err());
        assert!(compare_and_set(
            Nibble::new(0).unwrap(),
            Nibble::new(0).unwrap(),
            Word::W0,
            64
        )
        .is_err());
    }

    #[test]
    fn swap_forms_and_alignment() {
        let i = masked_store(HalfwordMask(0xa5), 8, 3).unwrap();
        assert_eq!(i.word(), 0x63 << 24 | 0xa5 << 14 | 8 << 6 | 3);
        // The excluded single-register form is never produced: bit 22 is clear
        // for every group and mask.
        for group in (0..64).step_by(4) {
            for mask in [0u8, 1, 0x55, 0xff] {
                let w = masked_store(HalfwordMask(mask), group, 3).unwrap().word();
                assert_eq!(w >> 22 & 1, 0);
            }
        }
        assert_eq!(
            masked_store(HalfwordMask::ALL, 9, 3),
            Err(AtomicError::MisalignedGroup { index: 9 })
        );
        assert!(masked_store(HalfwordMask::ALL, 64, 3).is_err());
    }

    #[test]
    fn incget_width_is_one_based_and_distinct_registers() {
        for (width, field) in [(1, 0), (2, 1), (16, 15), (32, 31)] {
            let i = increment_and_get(FieldWidth::new(width).unwrap(), Word::W2, 5, 6).unwrap();
            assert_eq!(i.word(), 0x61 << 24 | field << 14 | 2 << 12 | 5 << 6 | 6);
        }
        assert!(FieldWidth::new(0).is_err() && FieldWidth::new(33).is_err());
        assert_eq!(FieldWidth::new(32).unwrap().mask(), u32::MAX);
        assert_eq!(FieldWidth::new(1).unwrap().mask(), 1);
        assert_eq!(
            increment_and_get(FieldWidth::new(4).unwrap(), Word::W0, 5, 5),
            Err(AtomicError::Alias { register: 5 })
        );
    }

    #[test]
    fn fifo_geometry_bounds_and_encoding() {
        assert!(FifoGeometry::new(0, 0).is_err());
        assert!(FifoGeometry::new(16, 0).is_err());
        assert!(FifoGeometry::new(3, 3).is_err(), "batch 8 > capacity 4");
        assert!(FifoGeometry::new(3, 2).is_ok(), "batch equals capacity");
        let g = FifoGeometry::new(3, 1).unwrap();
        assert_eq!((g.capacity(), g.batch(), g.counter_mask()), (4, 2, 7));
        let push = fifo(g, FifoSide::Push, FifoAction::Advance, 5, 6).unwrap();
        assert_eq!(
            push.word(),
            0x62 << 24 | 1 << 18 | 3 << 14 | 1 << 12 | 5 << 6 | 6
        );
        let wait = fifo(g, FifoSide::Pop, FifoAction::Wait, 5, 6).unwrap();
        assert_eq!(wait.word(), 0x62 << 24 | 1 << 22 | 3 << 14 | 5 << 6 | 6);
        assert!(fifo(g, FifoSide::Pop, FifoAction::Wait, 6, 6).is_err());
    }

    #[test]
    fn counters_reject_free_running_and_overfull_states() {
        let g = FifoGeometry::new(3, 1).unwrap();
        assert!(FifoCounters::new(g, 0, 0).is_ok());
        assert!(FifoCounters::new(g, 6, 2).is_ok(), "wrapped, occupancy 4");
        assert!(matches!(
            FifoCounters::new(g, 8, 8),
            Err(AtomicError::FreeRunningCounter { .. })
        ));
        assert!(matches!(
            FifoCounters::new(g, 0, 6),
            Err(AtomicError::BadOccupancy { occupancy: 6 })
        ));
        assert!(
            matches!(
                FifoCounters::new(g, 0, 1),
                Err(AtomicError::BadOccupancy { occupancy: 1 })
            ),
            "a single element is not a whole batch of two"
        );
        assert_eq!(FifoCounters::new(g, 6, 2).unwrap().words(), [6, 2, 0, 0]);
    }

    #[test]
    fn guard_block_stays_clear_of_the_descriptor() {
        use crate::mailbox::{role::Mailbox, Descriptor};
        for t in 0..3 {
            let mb = Mailbox::of(t);
            let g = guard::Guard::of(mb);
            for (at, _) in Descriptor::default().writes(mb) {
                assert!(at < g.arm() || at >= g.arm() + guard::Guard::BYTES);
            }
            assert!(g.released() + 4 <= g.arm() + guard::Guard::BYTES);
            assert!(g.arm() + guard::Guard::BYTES < mb.dump_offset(0, 0));
            // The arm write is last, so a half-written block is never read,
            // and every word of the block is written (nothing stale survives).
            let writes = g.arm_writes(1, 1, 7, guard::PollMode::Light);
            assert_eq!(writes[12].0, g.arm());
            assert_eq!(
                writes.iter().find(|w| w.0 == g.mode()).unwrap().1,
                1,
                "the mode is written"
            );
            let mut at: [u64; 13] = core::array::from_fn(|k| writes[k].0);
            at.sort_unstable();
            let mut want = [
                g.arm(),
                g.deadline(),
                g.grace(),
                g.polls(),
                g.complete_semaphore(),
                g.snapshot(),
                g.release(),
                g.released(),
                g.stage(),
                g.mode(),
                g.complete_word(),
                g.poke_addr(),
                g.poke_value(),
            ];
            want.sort_unstable();
            assert_eq!(at, want);
            assert_eq!(g.complete_word() % 16, 0);
            assert!(g.complete_word() + 4 <= g.arm() + guard::Guard::BYTES);
            assert_eq!(guard::snapshot_value(0x7654_3210, 5), 5);
            assert_eq!(
                guard::stage::name(guard::stage::FIRST_LOAD + 3),
                "in the first poll's load"
            );
        }
    }
}
