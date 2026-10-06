//! Writing Tensix backend configuration, and waiting for backend units.
//!
//! The encodings are generated — see [`crate::isa`]. What is here is the layer
//! above, in the same split [`crate::sfpu`] uses: the generated layer encodes what
//! the bits allow, and this layer refuses what the hardware does not honour and
//! cites the page that says so.
//!
//! # Why configuration is written as instructions, not as stores
//!
//! RISC-V can write `Config` with `sw` (see `tt_firmware::cfg`), but it cannot write
//! `ThreadConfig` at all — `SETC16` is the only path, and `UNPACR` with
//! `UnpackToDst` reads `ThreadConfig[CurrentThread].SRCA_SET_SetOvrdWithAddr` and
//! calls `UndefinedBehavior()` if it is set (`UNPACR_Regular.md:313`), so at least
//! one `SETC16` is mandatory for the unpacker path regardless.
//!
//! Emitting configuration as Tensix instructions also means a host-side test stages
//! it the same way it stages a kernel: as words in `mailbox::PROGRAM`. The firmware
//! stays a runner that pushes words and knows nothing about what they do.
//!
//! # `RMWCIB` is deliberately absent
//!
//! `RMWCIB` writes up to eight bits without a GPR and would be the obvious tool for
//! a sub-word field. **`libttsim_bh.so` has no handler for it** — it carries
//! `tensix_wrcfg` and `tensix_setgpr` but no `rmwcib` symbol of any kind — so a
//! program using it cannot be developed against the simulator. Whole-word `WRCFG`
//! is the path this module takes; see [`ConfigWords`] for how sub-word fields are
//! composed into a word before it is written.

use crate::cfg::{ConfigField, ConfigSpan, ThreadConfigField};
use crate::isa::generated::encode;
use crate::isa::{self};

pub use crate::isa::Instruction;

/// Tensix GPRs per thread (`ScalarUnit.md:30`: `uint32_t GPRs[3][64]`).
pub const GPR_COUNT: u32 = 64;

/// `Config` words in one bank, and therefore the exclusive upper bound on a
/// `WRCFG` `CfgIndex` (`WRCFG.md`: `if (CfgIndex >= (CFG_STATE_SIZE*4))
/// UndefinedBehavior()`).
pub const CONFIG_INDEX_LIMIT: u32 = crate::cfg::CONFIG_WORDS_PER_BANK;

/// The `Config` word whose every write is a configuration reset.
///
/// `BackendConfiguration.md:40`: writing anything here, except via `RMWCIB`, zeroes
/// `Config[i][0 .. GLOBAL_CFGREG_BASE_ADDR32]`.
pub const STATE_RESET_EN_ADDR32: u16 = crate::cfg::generated::alu::STATE_RESET_EN.addr32();

/// `ThreadConfig` entries, and the exclusive upper bound on a `SETC16` `CfgIndex`
/// (`SETC16.md`: `if (CfgIndex >= THD_STATE_SIZE) UndefinedBehavior()`).
pub const THREAD_CONFIG_INDEX_LIMIT: u32 = crate::cfg::THD_STATE_SIZE;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EncodeError {
    /// A GPR index outside `0..GPR_COUNT`.
    BadGpr { index: u32 },
    /// A `Config` index `WRCFG` would treat as out of bounds.
    ConfigIndexOutOfBounds { index: u32 },
    /// A `ThreadConfig` index `SETC16` would treat as out of bounds.
    ThreadConfigIndexOutOfBounds { index: u32 },
    /// A 128-bit `WRCFG` whose GPR or `Config` index is not a multiple of four.
    ///
    /// The instruction masks both with `& ~3` rather than refusing, so an
    /// unaligned request would silently write somewhere else.
    Misaligned128 { gpr: u32, index: u32 },
    /// A [`ConfigSpan`] that is not the four words a 128-bit `WRCFG` moves.
    SpanNotFourWords { words: u32 },
    /// A value too large for the field it was to be placed in.
    ValueTooLarge { value: u32, max: u32 },
    /// More distinct `Config` words than a [`ConfigWords`] can hold.
    TooManyWords { capacity: usize },
    /// A `WRCFG` aimed at `STATE_RESET_EN`, which would zero most of `Config`.
    WouldResetConfig,
    /// A field value too large for its bit width, as the generated layer saw it.
    FieldTooLarge {
        name: &'static str,
        value: u32,
        bits: u32,
    },
}

impl EncodeError {
    const fn from_isa(e: isa::EncodeError) -> Self {
        match e {
            isa::EncodeError::FieldTooLarge {
                field,
                value,
                width,
                ..
            } => EncodeError::FieldTooLarge {
                name: field,
                value,
                bits: width as u32,
            },
        }
    }
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EncodeError::BadGpr { index } => {
                write!(f, "GPR {index} does not exist; there are {GPR_COUNT}")
            }
            EncodeError::ConfigIndexOutOfBounds { index } => write!(
                f,
                "Config index {index} is past {CONFIG_INDEX_LIMIT}; WRCFG calls that \
                 UndefinedBehavior"
            ),
            EncodeError::ThreadConfigIndexOutOfBounds { index } => write!(
                f,
                "ThreadConfig index {index} is past {THREAD_CONFIG_INDEX_LIMIT}; SETC16 \
                 calls that UndefinedBehavior"
            ),
            EncodeError::Misaligned128 { gpr, index } => write!(
                f,
                "a 128-bit WRCFG needs GPR and Config index aligned to four; got \
                 GPR {gpr}, index {index}"
            ),
            EncodeError::SpanNotFourWords { words } => write!(
                f,
                "a 128-bit WRCFG moves exactly four words; this span is {words}"
            ),
            EncodeError::ValueTooLarge { value, max } => {
                write!(f, "{value} does not fit the field, whose maximum is {max}")
            }
            EncodeError::TooManyWords { capacity } => {
                write!(f, "more than {capacity} distinct Config words")
            }
            EncodeError::WouldResetConfig => write!(
                f,
                "a write to STATE_RESET_EN zeroes every Config word below \
                 GLOBAL_CFGREG_BASE_ADDR32 (BackendConfiguration.md:40), not just \
                 the field asked for"
            ),
            EncodeError::FieldTooLarge { name, value, bits } => {
                write!(f, "{name} = {value} does not fit in {bits} bits")
            }
        }
    }
}

const fn check_gpr(index: u32) -> Result<(), EncodeError> {
    if index < GPR_COUNT {
        Ok(())
    } else {
        Err(EncodeError::BadGpr { index })
    }
}

/// Set all 32 bits of Tensix GPR `gpr`, as the two `SETDMAREG` halves that
/// `SETDMAREG_Immediate.md` says it takes.
///
/// Returned in the order they must be pushed. `ResultHalfReg` indexes 16-bit
/// halves, so GPR *n* occupies halves `2n` and `2n + 1`.
pub const fn set_gpr(gpr: u32, value: u32) -> Result<[Instruction; 2], EncodeError> {
    if let Err(e) = check_gpr(gpr) {
        return Err(e);
    }
    let low = match encode::setdmareg_immediate(value & 0xFFFF, gpr * 2) {
        Ok(i) => i,
        Err(e) => return Err(EncodeError::from_isa(e)),
    };
    let high = match encode::setdmareg_immediate(value >> 16, gpr * 2 + 1) {
        Ok(i) => i,
        Err(e) => return Err(EncodeError::from_isa(e)),
    };
    Ok([low, high])
}

/// Write the global backend PRNG seed register.
///
/// Clobbers `gpr`. The caller must select the configuration bank and drain
/// unpack, math and pack work before issuing this sequence: the register also
/// is documented to reset the stochastic conversion generators as well as the
/// Vector Unit. However, `step91_seeded_prng` observes a continuing Vector Unit
/// stream after an identical WRCFG seed write within one program on both
/// Blackhole cards. The separately gated RISC-V full-width configuration-store
/// path, fence and settling interval does restart the stream; this WRCFG helper
/// is a diagnostic register write, not an application RNG contract.
pub fn write_prng_seed(gpr: u32, seed: u32) -> Result<[Instruction; 4], EncodeError> {
    let halves = set_gpr(gpr, seed)?;
    let write = write_word(
        gpr,
        crate::cfg::generated::global::PRNG_SEED_Seed_Val.addr32(),
    )?;
    Ok([halves[0], halves[1], write, nop()])
}

/// `WRCFG`: copy one GPR into one `Config` word of the bank the issuing thread's
/// `CFG_STATE_ID_StateID` selects.
///
/// Refuses [`STATE_RESET_EN_ADDR32`]. `BackendConfiguration.md:40` says writing
/// *anything* to that word — by any instruction but `RMWCIB` — instantaneously
/// zeroes every `Config` word below `GLOBAL_CFGREG_BASE_ADDR32`. It is a
/// whole-configuration reset wearing the shape of an ordinary field write, and
/// this module has no other way to express "I meant that".
///
/// Whole words only. A sub-word field has to be composed into the word first —
/// [`ConfigWords`] is what does that — because `WRCFG` overwrites all 32 bits and
/// this module deliberately does not use `RMWCIB` (see the module documentation).
///
/// **The instruction after this one must not consume what it wrote** (`WRCFG.md`,
/// "Instruction scheduling"). [`ConfigWords::program`] places the [`nop`] for you.
pub const fn write_word(gpr: u32, addr32: u16) -> Result<Instruction, EncodeError> {
    if let Err(e) = check_gpr(gpr) {
        return Err(e);
    }
    if addr32 as u32 >= CONFIG_INDEX_LIMIT {
        return Err(EncodeError::ConfigIndexOutOfBounds {
            index: addr32 as u32,
        });
    }
    if addr32 == STATE_RESET_EN_ADDR32 {
        return Err(EncodeError::WouldResetConfig);
    }
    match encode::wrcfg(gpr, 0, addr32 as u32) {
        Ok(i) => Ok(i),
        Err(e) => Err(EncodeError::from_isa(e)),
    }
}

/// Zero every `Config` word below `GLOBAL_CFGREG_BASE_ADDR32`, on purpose.
///
/// The spec's own whole-configuration reset: any write to `STATE_RESET_EN` other
/// than by `RMWCIB` does it (`BackendConfiguration.md:40`). [`write_word`] refuses
/// that write because an *accidental* one looks like an ordinary field write;
/// this is the deliberate form. Needed on silicon, where `Config` outlives the
/// program that wrote it -- including tt-metal's: a tile it had used still held
/// `ALU_ACC_CTRL_SFPU_Fp32_enabled = 1`, which changes how `SFPSTORE` lays FP32
/// into `Dst`, and the step 4 gates read the previous run's product. ttsim starts
/// each run with `Config` zero.
///
/// `SETDMAREG` x2, `WRCFG`, then a wait on the Configuration Unit (C12) that
/// holds every following instruction back until the reset has landed.
pub const fn reset_config(gpr: u32) -> Result<[Instruction; 4], EncodeError> {
    let pair = match set_gpr(gpr, 0) {
        Ok(p) => p,
        Err(e) => return Err(e),
    };
    let reset = match encode::wrcfg(gpr, 0, STATE_RESET_EN_ADDR32 as u32) {
        Ok(i) => i,
        Err(e) => return Err(EncodeError::from_isa(e)),
    };
    let wait = match stallwait(Before::EVERYTHING.mask(), cond::CONFIG_BUSY) {
        Ok(i) => i,
        Err(e) => return Err(e),
    };
    Ok([pair[0], pair[1], reset, wait])
}

/// `WRCFG` in its 128-bit form: copy GPRs `gpr..gpr+4` into the four `Config` words
/// of `span`.
///
/// This is how a `TileDescriptor` reaches
/// `THCON_SEC*_REG0_TileDescriptor` in one instruction rather than four.
///
/// `WRCFG.md` masks both indices with `& ~3` instead of refusing an unaligned one,
/// so a misaligned request would silently write four words somewhere else. Refused
/// here rather than rounded.
pub const fn write_span(gpr: u32, span: ConfigSpan) -> Result<Instruction, EncodeError> {
    if let Err(e) = check_gpr(gpr) {
        return Err(e);
    }
    if span.words() != 4 {
        return Err(EncodeError::SpanNotFourWords {
            words: span.words() as u32,
        });
    }
    let index = span.addr32() as u32;
    if index >= CONFIG_INDEX_LIMIT {
        return Err(EncodeError::ConfigIndexOutOfBounds { index });
    }
    if gpr % 4 != 0 || index % 4 != 0 {
        return Err(EncodeError::Misaligned128 { gpr, index });
    }
    match encode::wrcfg(gpr, 1, index) {
        Ok(i) => Ok(i),
        Err(e) => Err(EncodeError::from_isa(e)),
    }
}

/// The four `set_gpr` pairs plus the 128-bit [`write_span`] that stage `words` into
/// `span`, in push order. Nine instructions.
///
/// `gpr` is the first of four consecutive GPRs and must be a multiple of four.
pub const fn write_span_words(
    gpr: u32,
    span: ConfigSpan,
    words: [u32; 4],
) -> Result<[Instruction; 9], EncodeError> {
    let a = match set_gpr(gpr, words[0]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let b = match set_gpr(gpr + 1, words[1]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let c = match set_gpr(gpr + 2, words[2]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let d = match set_gpr(gpr + 3, words[3]) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    let w = match write_span(gpr, span) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };
    Ok([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1], w])
}

/// `SETC16`: write one whole 16-bit `ThreadConfig` entry of the issuing thread.
///
/// Entry-level rather than field-level on purpose. `SETC16` has no mask, so it
/// replaces every field sharing the entry — `CLR_DVALID_SrcA_Disable` and
/// `CLR_DVALID_SrcB_Disable` are both entry 7, for instance. A field-level helper
/// would silently clear the neighbour; [`ThreadConfigEntry`] composes instead.
pub const fn set_thread_entry(addr32: u16, value: u16) -> Result<Instruction, EncodeError> {
    if addr32 as u32 >= THREAD_CONFIG_INDEX_LIMIT {
        return Err(EncodeError::ThreadConfigIndexOutOfBounds {
            index: addr32 as u32,
        });
    }
    match encode::setc16(addr32 as u32, value as u32) {
        Ok(i) => Ok(i),
        Err(e) => Err(EncodeError::from_isa(e)),
    }
}

/// One `ThreadConfig` entry under construction.
///
/// Starts from a base entry value and sets fields into it, so that a `SETC16` can
/// write several fields of one entry without clearing the rest. The base is the
/// caller's problem: `ThreadConfig` cannot be read back by RISC-V under ttsim
/// (`docs/learnings/ttsim-divergence.md` row 19), so simulator-bound code starts from zero and
/// has to mean it.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ThreadConfigEntry {
    addr32: u16,
    value: u16,
}

impl ThreadConfigEntry {
    /// An entry whose every bit is zero.
    pub const fn zeroed(addr32: u16) -> Self {
        ThreadConfigEntry { addr32, value: 0 }
    }

    /// An entry starting from a value read from the hardware.
    pub const fn from_value(addr32: u16, value: u16) -> Self {
        ThreadConfigEntry { addr32, value }
    }

    pub const fn addr32(self) -> u16 {
        self.addr32
    }

    pub const fn value(self) -> u16 {
        self.value
    }

    /// Place `value` into `field`, which must belong to this entry.
    pub const fn set(mut self, field: ThreadConfigField, value: u16) -> Result<Self, EncodeError> {
        if field.addr32() != self.addr32 {
            return Err(EncodeError::ThreadConfigIndexOutOfBounds {
                index: field.addr32() as u32,
            });
        }
        if !field.fits(value) {
            return Err(EncodeError::ValueTooLarge {
                value: value as u32,
                max: field.max_value() as u32,
            });
        }
        self.value = field.insert(self.value, value);
        Ok(self)
    }

    pub const fn encode(self) -> Result<Instruction, EncodeError> {
        set_thread_entry(self.addr32, self.value)
    }
}

/// A Tensix `NOP`.
///
/// Required after `WRCFG` before anything consumes what it wrote
/// (`WRCFG.md`, "Instruction scheduling").
pub fn nop() -> Instruction {
    encode::nop().expect("NOP has no operands")
}

/// How many distinct `Config` words one [`ConfigWords`] can stage.
///
/// Fixed rather than growable because `tt-isa` is `no_std` with no allocator.
/// Configuring the unpacker and the packer together touches about twenty-four
/// words, so this is sized for that with room to spare; raising it costs only
/// stack, and [`EncodeError::TooManyWords`] names the limit when it is hit.
pub const MAX_CONFIG_WORDS: usize = 40;

/// `Config` words under construction, keyed by word index.
///
/// `WRCFG` writes whole words, so several fields of one word must be merged before
/// it is written or the last one wins. Accumulating by word also means the emitted
/// program has one `WRCFG` per *word* rather than per field.
///
/// Bits no [`set`](Self::set) call touches are left at the value the word was
/// [`seeded`](Self::seed) with, which is zero unless the caller says otherwise.
/// That is a real assumption, not a safe default: `Config` has no documented
/// power-on value, and ttsim makes a read of a word it has not modelled fatal
/// (`docs/learnings/ttsim-divergence.md` row 21). Seed deliberately.
#[derive(Copy, Clone, Debug)]
pub struct ConfigWords {
    words: [(u16, u32); MAX_CONFIG_WORDS],
    len: usize,
}

impl Default for ConfigWords {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigWords {
    pub const fn new() -> Self {
        ConfigWords {
            words: [(0, 0); MAX_CONFIG_WORDS],
            len: 0,
        }
    }

    fn slot(&mut self, addr32: u16) -> Result<&mut u32, EncodeError> {
        for i in 0..self.len {
            if self.words[i].0 == addr32 {
                return Ok(&mut self.words[i].1);
            }
        }
        if self.len == MAX_CONFIG_WORDS {
            return Err(EncodeError::TooManyWords {
                capacity: MAX_CONFIG_WORDS,
            });
        }
        if addr32 as u32 >= CONFIG_INDEX_LIMIT {
            return Err(EncodeError::ConfigIndexOutOfBounds {
                index: addr32 as u32,
            });
        }
        self.words[self.len] = (addr32, 0);
        self.len += 1;
        Ok(&mut self.words[self.len - 1].1)
    }

    /// Give a word a starting value, for the bits no field will set.
    pub fn seed(&mut self, addr32: u16, value: u32) -> Result<&mut Self, EncodeError> {
        *self.slot(addr32)? = value;
        Ok(self)
    }

    /// Place `value` into `field`, leaving the rest of its word alone.
    pub fn set(&mut self, field: ConfigField, value: u32) -> Result<&mut Self, EncodeError> {
        if !field.fits(value) {
            return Err(EncodeError::ValueTooLarge {
                value,
                max: field.max_value(),
            });
        }
        let word = self.slot(field.addr32())?;
        *word = field.insert(*word, value);
        Ok(self)
    }

    /// The staged words, in the order they were first touched.
    pub fn words(&self) -> &[(u16, u32)] {
        &self.words[..self.len]
    }

    /// Instructions [`Self::program`] will emit for the words staged so far.
    pub fn program_len(&self) -> usize {
        if self.len == 0 {
            0
        } else {
            self.len * 3 + 1
        }
    }

    /// Emit `SETDMAREG` ×2 + `WRCFG` per staged word, then one `NOP`, into `out`.
    ///
    /// Returns how many instructions were written. `out` must hold at least
    /// [`Self::program_len`]; a slice-filling signature rather than a returned
    /// `Vec` because `tt-isa` is `no_std` with no allocator.
    ///
    /// All of it goes through a single scratch GPR, since each word is consumed by
    /// its `WRCFG` before the next `SETDMAREG` pair overwrites it. The trailing
    /// `NOP` is the separation `WRCFG.md` requires before anything reads the
    /// configuration.
    pub fn program(&self, scratch_gpr: u32, out: &mut [Instruction]) -> Result<usize, EncodeError> {
        let needed = self.program_len();
        if out.len() < needed {
            return Err(EncodeError::TooManyWords {
                capacity: out.len(),
            });
        }
        let mut n = 0;
        for &(addr32, value) in self.words() {
            let pair = set_gpr(scratch_gpr, value)?;
            out[n] = pair[0];
            out[n + 1] = pair[1];
            out[n + 2] = write_word(scratch_gpr, addr32)?;
            n += 3;
        }
        if self.len != 0 {
            out[n] = nop();
            n += 1;
        }
        Ok(n)
    }
}

/// `STALLWAIT` block-mask bits B0..B8 (`STALLWAIT.md`, "Block mask").
///
/// Blackhole renumbers these relative to Wormhole, and the specification tells
/// software to abstract the difference rather than hardcode it. This is that
/// abstraction; it is the Blackhole numbering.
pub mod block {
    /// B0 — Miscellaneous Unit, Mover, Scalar Unit, Packer *and* Unpacker.
    pub const ANY_DMA: u32 = 1 << 0;
    /// B1 — Sync Unit.
    pub const SYNC: u32 = 1 << 1;
    /// B2 — Packer.
    pub const PACKER: u32 = 1 << 2;
    /// B3 — Unpackers.
    pub const UNPACKER: u32 = 1 << 3;
    /// B4 — Mover.
    pub const MOVER: u32 = 1 << 4;
    /// B5 — Scalar Unit (ThCon).
    pub const SCALAR: u32 = 1 << 5;
    /// B6 — Matrix Unit (FPU).
    pub const MATRIX: u32 = 1 << 6;
    /// B7 — Configuration Unit.
    pub const CONFIG: u32 = 1 << 7;
    /// B8 — Vector Unit (SFPU).
    pub const SFPU: u32 = 1 << 8;
}

/// `STALLWAIT` condition-mask bits C0..C12 (`STALLWAIT.md`, "Condition mask").
///
/// Each reads "keep on waiting if…", so a set bit is a reason to stall.
pub mod cond {
    /// C0 — the Scalar Unit has memory requests outstanding for this thread.
    pub const SCALAR_OUTSTANDING: u32 = 1 << 0;
    /// C1 — this thread has an instruction anywhere in unpacker 0's pipeline.
    pub const UNPACKER0_BUSY: u32 = 1 << 1;
    /// C2 — this thread has an instruction anywhere in unpacker 1's pipeline.
    pub const UNPACKER1_BUSY: u32 = 1 << 2;
    /// C3 — this thread has an instruction anywhere in the packer pipeline.
    pub const PACKER_BUSY: u32 = 1 << 3;
    /// C4 — this thread has an instruction anywhere in the Matrix Unit pipeline.
    pub const MATRIX_BUSY: u32 = 1 << 4;
    /// C5 — `SrcA[Unpackers[0].SrcBank].AllowedClient != Unpackers`.
    pub const SRCA_NOT_UNPACKER: u32 = 1 << 5;
    /// C6 — `SrcB[Unpackers[1].SrcBank].AllowedClient != Unpackers`.
    pub const SRCB_NOT_UNPACKER: u32 = 1 << 6;
    /// C7 — `SrcA[MatrixUnit.SrcABank].AllowedClient != MatrixUnit`.
    pub const SRCA_NOT_MATRIX: u32 = 1 << 7;
    /// C8 — `SrcB[MatrixUnit.SrcBBank].AllowedClient != MatrixUnit`.
    pub const SRCB_NOT_MATRIX: u32 = 1 << 8;
    /// C9 — the Mover has memory requests outstanding, from any thread.
    pub const MOVER_OUTSTANDING: u32 = 1 << 9;
    /// C10 — this thread's RISC-V core has a request against GPRs, configuration
    /// or TDMA-RISC that has been emitted but not processed.
    pub const RISCV_REQUEST_PENDING: u32 = 1 << 10;
    /// C11 — this thread has an instruction anywhere in the SFPU pipeline.
    pub const SFPU_BUSY: u32 = 1 << 11;
    /// C12 — *any* thread has an instruction in the Configuration Unit pipeline.
    pub const CONFIG_BUSY: u32 = 1 << 12;
}

/// `STALLWAIT` with an arbitrary pair of masks.
///
/// Prefer the paired helpers below. The specification's own notes on the condition
/// mask say which block bit each condition needs, and a condition set without its
/// block bit waits without stopping anything from running past it — which looks
/// like a race rather than a missing bit.
pub const fn stallwait(block_mask: u32, condition_mask: u32) -> Result<Instruction, EncodeError> {
    match encode::stallwait(block_mask, condition_mask) {
        Ok(i) => Ok(i),
        Err(e) => Err(EncodeError::from_isa(e)),
    }
}

/// Which of this thread's *following* instructions a wait must hold back: the
/// units that consume what the waited-on unit produces.
///
/// `STALLWAIT`'s block mask does not name what to wait *for* -- the condition mask
/// does that -- it names which later instructions stall until the condition
/// clears (`STALLWAIT.md`, "Block mask"). Everything else runs past it. So a wait
/// for unpacker 0 that blocks only B3 (the unpackers) holds back the next
/// `UNPACR` and lets the `SFPLOAD`s that read what it wrote go ahead. The first
/// silicon run of the elementwise gate did exactly that and read `Dst` before the
/// unpacker had written it; ttsim evaluates each unit synchronously and cannot
/// show the race. Every wait helper therefore takes the consumer as an argument
/// it cannot omit, and blocks the producer's own unit in addition.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Before(u32);

impl Before {
    pub const UNPACKER: Before = Before(block::UNPACKER);
    pub const PACKER: Before = Before(block::PACKER);
    pub const MATRIX: Before = Before(block::MATRIX);
    pub const SFPU: Before = Before(block::SFPU);
    pub const CONFIG: Before = Before(block::CONFIG);
    /// Every unit: a full barrier on this thread. What the end of a program, or
    /// anything a RISC-V core or the host will read afterwards, wants.
    pub const EVERYTHING: Before = Before(
        block::ANY_DMA
            | block::SYNC
            | block::PACKER
            | block::UNPACKER
            | block::MOVER
            | block::SCALAR
            | block::MATRIX
            | block::CONFIG
            | block::SFPU,
    );

    /// Both sets of consumers.
    pub const fn and(self, other: Before) -> Before {
        Before(self.0 | other.0)
    }

    pub const fn mask(self) -> u32 {
        self.0
    }
}

/// Wait until this thread has drained unpacker 0 (C1), holding back its own
/// unit (B3) and `before`.
pub const fn wait_for_unpacker0(before: Before) -> Result<Instruction, EncodeError> {
    stallwait(block::UNPACKER | before.0, cond::UNPACKER0_BUSY)
}

/// Wait until this thread has drained unpacker 1 (C2), holding back B3 and
/// `before`.
pub const fn wait_for_unpacker1(before: Before) -> Result<Instruction, EncodeError> {
    stallwait(block::UNPACKER | before.0, cond::UNPACKER1_BUSY)
}

/// Wait until this thread has drained the packer (C3), holding back B2 and
/// `before`.
pub const fn wait_for_packer(before: Before) -> Result<Instruction, EncodeError> {
    stallwait(block::PACKER | before.0, cond::PACKER_BUSY)
}

/// Wait until this thread has drained the SFPU (C11), holding back B8 and
/// `before`.
pub const fn wait_for_sfpu(before: Before) -> Result<Instruction, EncodeError> {
    stallwait(block::SFPU | before.0, cond::SFPU_BUSY)
}

/// Wait until this thread has drained the Matrix Unit (C4), holding back B6 and
/// `before`.
///
/// `ZEROACC` runs on the Matrix Unit, not on the SFPU — easy to get wrong, since
/// everything around it in a `Dst` scrub is SFPU work.
pub const fn wait_for_matrix(before: Before) -> Result<Instruction, EncodeError> {
    stallwait(block::MATRIX | before.0, cond::MATRIX_BUSY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::generated::{alu, thcon, thread};
    use crate::isa::generated::defs;

    #[test]
    fn diagnostic_seed_sequence_checks_the_register_and_scheduling_bubble() {
        let [lo, hi, write, bubble] = write_prng_seed(8, 0x12345678).unwrap();
        assert_eq!(lo.operand("NewValue"), Some(0x5678));
        assert_eq!(hi.operand("NewValue"), Some(0x1234));
        assert_eq!(write.operand("CfgIndex"), Some(186));
        assert_eq!(write.operand("InputReg"), Some(8));
        assert_eq!(bubble.def().key(), "NOP");
        assert!(write_prng_seed(GPR_COUNT - 1, 0).is_ok());
        assert!(write_prng_seed(GPR_COUNT, 0).is_err());
        assert!(write_prng_seed(u32::MAX, 0).is_err());
    }

    #[test]
    fn set_gpr_writes_both_halves_in_push_order() {
        let [low, high] = set_gpr(3, 0xDEAD_BEEF).unwrap();
        // `ResultHalfReg` indexes 16-bit halves, so GPR 3 is halves 6 and 7.
        assert_eq!(low.operand("ResultHalfReg"), Some(6));
        assert_eq!(low.operand("NewValue"), Some(0xBEEF));
        assert_eq!(high.operand("ResultHalfReg"), Some(7));
        assert_eq!(high.operand("NewValue"), Some(0xDEAD));
    }

    #[test]
    fn write_word_names_the_config_index() {
        let insn = write_word(3, 72).unwrap();
        assert_eq!(insn.def().key(), "WRCFG");
        assert_eq!(insn.operand("InputReg"), Some(3));
        assert_eq!(insn.operand("Is128Bit"), Some(0));
        assert_eq!(insn.operand("CfgIndex"), Some(72));
    }

    #[test]
    fn write_span_is_the_128_bit_form() {
        let span = thcon::THCON_SEC0_REG0_TileDescriptor;
        let insn = write_span(4, span).unwrap();
        assert_eq!(insn.operand("Is128Bit"), Some(1));
        assert_eq!(insn.operand("InputReg"), Some(4));
        assert_eq!(insn.operand("CfgIndex"), Some(span.addr32() as u32));
    }

    /// `WRCFG.md` masks both indices with `& ~3` rather than refusing, so an
    /// unaligned 128-bit write silently lands four words away. Refused here.
    #[test]
    fn a_misaligned_128_bit_write_is_refused_rather_than_rounded() {
        let span = thcon::THCON_SEC0_REG0_TileDescriptor;
        assert!(span.addr32() % 4 == 0, "the span itself is aligned");
        assert_eq!(
            write_span(5, span),
            Err(EncodeError::Misaligned128 {
                gpr: 5,
                index: span.addr32() as u32
            })
        );
    }

    /// The trap that makes a field write into a configuration reset.
    #[test]
    fn writing_state_reset_en_is_refused() {
        assert_eq!(STATE_RESET_EN_ADDR32, alu::STATE_RESET_EN.addr32());
        assert_eq!(
            write_word(0, STATE_RESET_EN_ADDR32),
            Err(EncodeError::WouldResetConfig)
        );
        // And it is reachable the obvious way, so the refusal is load-bearing.
        let mut words = ConfigWords::new();
        words.set(alu::STATE_RESET_EN, 1).unwrap();
        let mut out = [nop(); 8];
        assert_eq!(
            words.program(0, &mut out).err(),
            Some(EncodeError::WouldResetConfig)
        );
    }

    #[test]
    fn out_of_bounds_indices_are_refused() {
        assert_eq!(
            write_word(0, CONFIG_INDEX_LIMIT as u16),
            Err(EncodeError::ConfigIndexOutOfBounds {
                index: CONFIG_INDEX_LIMIT
            })
        );
        assert_eq!(
            set_thread_entry(THREAD_CONFIG_INDEX_LIMIT as u16, 0),
            Err(EncodeError::ThreadConfigIndexOutOfBounds {
                index: THREAD_CONFIG_INDEX_LIMIT
            })
        );
        assert_eq!(
            set_gpr(GPR_COUNT, 0),
            Err(EncodeError::BadGpr { index: 64 })
        );
    }

    /// The whole point of accumulating by word: two fields of one word must merge,
    /// not race, and they must cost one `WRCFG` rather than two.
    #[test]
    fn two_fields_of_one_word_merge_into_a_single_write() {
        let a = alu::RISC_DEST_ACCESS_CTRL_SEC0_fmt;
        let b = alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt;
        assert_eq!(a.addr32(), b.addr32(), "the premise of this test");

        let mut words = ConfigWords::new();
        words.set(a, 3).unwrap().set(b, 2).unwrap();
        assert_eq!(words.words().len(), 1);

        let (addr32, value) = words.words()[0];
        assert_eq!(addr32, a.addr32());
        assert_eq!(a.extract(value), 3, "the first field survived the second");
        assert_eq!(b.extract(value), 2);

        assert_eq!(words.program_len(), 4);
    }

    #[test]
    fn distinct_words_each_get_their_own_write_and_one_trailing_nop() {
        let mut words = ConfigWords::new();
        words
            .set(alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt, 0)
            .unwrap()
            .set(thcon::THCON_SEC0_REG2_Out_data_format, 5)
            .unwrap();

        let mut out = [nop(); 16];
        let n = words.program(0, &mut out).unwrap();
        assert_eq!(n, 7, "two words: (SETDMAREG x2 + WRCFG) x2, then one NOP");
        assert_eq!(out[2].def().key(), "WRCFG");
        assert_eq!(out[5].def().key(), "WRCFG");
        assert_eq!(
            out[6].def().key(),
            "NOP",
            "WRCFG.md requires separation before anything consumes the write"
        );
    }

    #[test]
    fn an_empty_plan_emits_nothing_at_all() {
        let words = ConfigWords::new();
        let mut out = [nop(); 4];
        assert_eq!(words.program_len(), 0);
        assert_eq!(words.program(0, &mut out).unwrap(), 0, "not even the NOP");
    }

    #[test]
    fn a_value_too_large_for_its_field_is_refused_rather_than_truncated() {
        let f = alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt;
        let mut words = ConfigWords::new();
        assert_eq!(
            words.set(f, f.max_value() + 1).err(),
            Some(EncodeError::ValueTooLarge {
                value: f.max_value() + 1,
                max: f.max_value()
            })
        );
        assert!(words.words().is_empty(), "a refused set must stage nothing");
    }

    /// `SETC16` has no mask, so it replaces every field sharing the entry. The
    /// entry-level API is what stops a field-level one from clearing a neighbour
    /// silently.
    #[test]
    fn a_thread_config_entry_carries_every_field_that_shares_it() {
        let a = thread::CLR_DVALID_SrcA_Disable;
        let b = thread::CLR_DVALID_SrcB_Disable;
        assert_eq!(a.addr32(), b.addr32(), "the premise of this test");

        let entry = ThreadConfigEntry::zeroed(a.addr32())
            .set(a, 1)
            .unwrap()
            .set(b, 1)
            .unwrap();
        assert_eq!(a.extract(entry.value()), 1);
        assert_eq!(b.extract(entry.value()), 1);

        let insn = entry.encode().unwrap();
        assert_eq!(insn.def().key(), "SETC16");
        assert_eq!(insn.operand("CfgIndex"), Some(a.addr32() as u32));
        assert_eq!(insn.operand("NewValue"), Some(entry.value() as u32));
    }

    #[test]
    fn a_field_from_another_entry_cannot_be_set_into_this_one() {
        let entry = ThreadConfigEntry::zeroed(thread::CLR_DVALID_SrcA_Disable.addr32());
        assert!(entry.set(thread::CFG_STATE_ID_StateID, 1).is_err());
    }

    /// Each helper waits on its own condition, always holds back its own unit --
    /// a condition without its block bit waits without stopping anything -- and
    /// also holds back the consumers it is given.
    #[test]
    fn the_stall_helpers_block_their_own_unit_and_the_named_consumers() {
        for (insn, block, cond) in [
            (
                wait_for_unpacker0(Before::SFPU),
                block::UNPACKER | block::SFPU,
                cond::UNPACKER0_BUSY,
            ),
            (
                wait_for_packer(Before::EVERYTHING),
                Before::EVERYTHING.mask(),
                cond::PACKER_BUSY,
            ),
            (
                wait_for_sfpu(Before::PACKER),
                block::SFPU | block::PACKER,
                cond::SFPU_BUSY,
            ),
            (
                wait_for_matrix(Before::MATRIX),
                block::MATRIX,
                cond::MATRIX_BUSY,
            ),
        ] {
            let insn = insn.unwrap();
            assert_eq!(insn.def().key(), "STALLWAIT_BH");
            assert_eq!(insn.operand("BlockMask"), Some(block));
            assert_eq!(insn.operand("ConditionMask"), Some(cond));
        }
    }

    /// The case that broke the elementwise gate on silicon: an unpack into `Dst`
    /// followed by SFPU reads of it. The wait must hold the SFPU, not only the
    /// unpackers.
    #[test]
    fn an_unpack_consumed_by_the_sfpu_holds_the_sfpu() {
        let insn = wait_for_unpacker0(Before::SFPU).unwrap();
        let mask = insn.operand("BlockMask").unwrap();
        assert_ne!(mask & block::SFPU, 0, "the SFPU would run past the wait");
    }

    /// Blackhole renumbers the condition mask relative to Wormhole, so these
    /// constants are the Blackhole ones and must not be reused for the superseded
    /// encoding.
    #[test]
    fn the_stall_masks_are_blackholes() {
        assert_eq!(
            defs::STALLWAIT.page(),
            "BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md"
        );
        assert!(defs::STALLWAIT.provenance().is_documented_for_blackhole());
        // Nine block bits and thirteen condition bits, per the page's own tables.
        assert_eq!(defs::STALLWAIT.field("BlockMask").unwrap().width(), 9);
        assert_eq!(defs::STALLWAIT.field("ConditionMask").unwrap().width(), 13);
        assert_eq!(block::SFPU, 1 << 8, "B8 is the highest block bit");
        assert_eq!(
            cond::CONFIG_BUSY,
            1 << 12,
            "C12 is the highest condition bit"
        );
    }
}
