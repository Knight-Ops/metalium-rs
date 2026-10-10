//! `SFPLOADMACRO`: the Vector Unit's one instruction that starts more than one
//! instruction per cycle, and the configuration it executes.
//!
//! `SFPLOADMACRO.md` describes an `SFPLOAD` that also *schedules* up to four more
//! vector instructions, one per sub-unit (Simple, MAD, Round, Store), each after a
//! delay, from state that `SFPCONFIG` wrote beforehand:
//!
//! ```text
//! LoadMacroConfig.Misc               12 bits: StoreMod0 | UsesLoadMod0ForStore | UnitDelayKind
//! LoadMacroConfig.Sequence[macro]    32 bits: one byte per sub-unit
//! LoadMacroConfig.InstructionTemplate[t]   a whole vector instruction word
//! ```
//!
//! This module is the *checked* half: types whose constructors refuse what the
//! page says is undefined, and which turn a configuration into the exact
//! `SFPLOADI` + `SFPCONFIG` words that write it. What a configured macro then
//! *does* over time -- the delay counters, the forgetting of a colliding pending
//! instruction, the substituted operands -- lives in
//! `tt_kernels::sfpu::macro_sched`, which holds a model written from the page and
//! is where the cycle-level rules (a `SFPMAD` result is not ready for a cycle, two
//! macros' instructions meeting on one sub-unit) are enforced.
//!
//! # Byte of a `Sequence`
//!
//! Per `SFPLOADMACRO.md`'s functional model, byte `i` of `Sequence[macro]` drives
//! sub-unit `i` (Simple, MAD, Round, Store):
//!
//! | bits | meaning |
//! |---|---|
//! | `2:0` | source: 0 idle, 1 **undefined**, 2 `SFPNOP`, 3 `SFPSTORE` with `VD = 0`, 4..=7 `InstructionTemplate[0..=3]` |
//! | `5:3` | delay, 0..=7; **7 also skips the forgetting of a colliding pending instruction** |
//! | `6` | result register is `LReg[16]` instead of the macro's `VD` (for Store: *read* `LReg[16]`) |
//! | `7` | substitute the macro's `VD` for `VB` instead of `VC` (for Store: keep the template's own `VD`) |
//!
//! # Provenance
//!
//! The Blackhole instruction layout is the generated `SFPLOADMACRO_BH`
//! (`Diagrams/Out/Bits32_SFPLOADMACRO_BH.svg`: `VDHi` bit 0, `Imm9` 9:1, `AddrMod`
//! 15:13, `Mod0` 19:16, `VDLo` 21:20, `MacroIndex` 23:22, opcode `0x93`). It agrees
//! with the page's syntax block, field for field; the Wormhole layout differs only
//! in `AddrMod`'s width. `SFPCONFIG` is documented once, in the Wormhole tree, with
//! the Blackhole page stating the behaviour is identical.
//!
//! Every form here is **documented but not yet measured on silicon**: the one
//! measurement in this repository is that an unconfigured `SFPLOADMACRO` begins as
//! an `SFPLOAD` (`step5_corpus::sfploadmacro_begins_as_an_sfpload_from_dst`).

use crate::isa::generated::{defs, encode, ALL};
use crate::isa::{Instruction, InstructionDef};
use crate::sfpu;

/// Macros a configuration holds (`LoadMacroConfig.Sequence[4]`).
pub const MACROS: usize = 4;
/// Instruction templates a configuration holds (`InstructionTemplate[4]`).
pub const TEMPLATES: usize = 4;
/// The largest delay a sequence byte can carry.
pub const MAX_DELAY: u8 = 7;
/// The delay that schedules without forgetting a pending instruction.
pub const NO_FORGET: u8 = 7;
/// `LReg[16]`: written only by scheduled Simple/MAD/Round instructions, read only
/// by a scheduled `SFPSTORE` (`LReg.md`).
pub const LREG_BONUS: u32 = 16;
/// `SFPNOP`s that retire everything a macro can have left pending: a delay is at
/// most seven, and one more issue lets the last countdown reach its cycle.
pub const DRAIN_NOPS: usize = 8;
/// `SFPCONFIG_MOD1_IMM16_IS_VALUE`.
const CONFIG_IMM16_IS_VALUE: u32 = 1;
/// The most instructions [`MacroConfig::writes`] produces: a leading `SFPNOP` and
/// `SFPENCC`, five each for four templates and four sequences (two `SFPLOADI`s, an
/// `SFPNOP`, the `SFPCONFIG` and an `SFPNOP`), and two for `Misc`.
pub const CONFIG_MAX: usize = 2 + 5 * (TEMPLATES + MACROS) + 2;

/// `SFPCONFIG`'s `VD` for each macro register (`SFPCONFIG.md`).
pub mod config_target {
    /// `InstructionTemplate[t]`.
    pub const fn template(t: u32) -> u32 {
        t
    }
    /// `Sequence[m]`.
    pub const fn sequence(m: u32) -> u32 {
        4 + m
    }
    /// `Misc`.
    pub const MISC: u32 = 8;
}

/// Why a macro configuration or `SFPLOADMACRO` was refused.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MacroError {
    /// A macro index outside 0..4.
    MacroIndex { value: u32 },
    /// A template index outside 0..4.
    TemplateIndex { value: u32 },
    /// A delay above 7.
    Delay { value: u32 },
    /// `SFPLOADMACRO`'s destination: three bits only (`VD = VDHi << 2 | VDLo`),
    /// so `LReg[0..8]`; `LReg[8..]` would be dropped by the load.
    LoadVd { value: u32 },
    /// An address wider than ten bits.
    Imm10 { value: u32 },
    /// An `AddrMod` wider than three bits.
    AddrMod { value: u32 },
    /// `Mod0`/`StoreMod0` wider than four bits.
    Mod0 { value: u32 },
    /// `VDHi` is the low bit of the address, so `VD >= 4` needs an odd `Imm10` and
    /// `VD < 4` an even one (`SFPLOADMACRO.md`: `Imm10 = Imm9 << 1 | VDHi`).
    VdHiCoupled { vd: u32, imm10: u32 },
    /// Sequence source code 1, which the page calls `UndefinedBehavior`.
    UndefinedSource,
    /// A sequence slot that names a template the configuration does not hold.
    TemplateMissing { macro_index: u32, template: u32 },
    /// A template word that is not a Vector Unit instruction the page lists.
    NotAVectorInstruction { word: u32 },
    /// A template (or built-in source) the sub-unit cannot execute: the hardware
    /// substitutes `SFPNOP`, which is never what was meant.
    ClassMismatch {
        macro_index: u32,
        unit: SubUnit,
        mnemonic: &'static str,
    },
    /// `SFPNOP` fed to the Store sub-unit, which "should not be fed `SFPNOP`".
    StoreNop { macro_index: u32 },
    /// A Simple and a Round instruction on one cycle both write `LReg[16]` or both
    /// do not.
    SimpleRoundDestination { macro_index: u32 },
    /// `SFPSWAP` in the Simple sub-unit without an `SFPNOP` on the MAD sub-unit for
    /// the same time.
    SwapNeedsMadNop { macro_index: u32 },
    /// `SFPSWAP` in the Simple sub-unit with a non-`SFPNOP` on the Round sub-unit on
    /// the next cycle.
    SwapNextCycleBusy { macro_index: u32 },
    /// `SFPSWAP`'s MAD `SFPNOP` is on a different delay kind to the swap, so
    /// "the same time" is not a fact.
    SwapDelayKind { macro_index: u32 },
    /// An unrepresentable field, as the generated encoder reports it.
    Encode(sfpu::EncodeError),
}

impl core::fmt::Display for MacroError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MacroError::MacroIndex { value } => write!(f, "macro index {value} is not in 0..4"),
            MacroError::TemplateIndex { value } => {
                write!(f, "template index {value} is not in 0..4")
            }
            MacroError::Delay { value } => write!(f, "delay {value} does not fit three bits"),
            MacroError::LoadVd { value } => write!(
                f,
                "SFPLOADMACRO's VD is three bits (LReg[0..8]); {value} is out of range"
            ),
            MacroError::Imm10 { value } => write!(f, "address {value} does not fit ten bits"),
            MacroError::AddrMod { value } => write!(f, "AddrMod {value} does not fit three bits"),
            MacroError::Mod0 { value } => write!(f, "Mod0 {value} does not fit four bits"),
            MacroError::VdHiCoupled { vd, imm10 } => write!(
                f,
                "VD {vd} and address {imm10:#x} disagree: VDHi (VD bit 2) is address bit 0, \
                 so VD >= 4 needs an odd address and VD < 4 an even one"
            ),
            MacroError::UndefinedSource => {
                write!(f, "sequence source 1 is UndefinedBehavior (SFPLOADMACRO.md)")
            }
            MacroError::TemplateMissing {
                macro_index,
                template,
            } => write!(
                f,
                "macro {macro_index} schedules template {template}, which the configuration does not hold"
            ),
            MacroError::NotAVectorInstruction { word } => write!(
                f,
                "template word {word:#010x} is not a Vector Unit instruction"
            ),
            MacroError::ClassMismatch {
                macro_index,
                unit,
                mnemonic,
            } => write!(
                f,
                "macro {macro_index}: the {} sub-unit cannot execute {mnemonic}; the hardware \
                 would substitute SFPNOP",
                unit.name()
            ),
            MacroError::StoreNop { macro_index } => write!(
                f,
                "macro {macro_index}: SFPNOP on the Store sub-unit is UndefinedBehavior"
            ),
            MacroError::SimpleRoundDestination { macro_index } => write!(
                f,
                "macro {macro_index}: Simple and Round execute on the same cycle, so exactly one \
                 of them must write LReg[16]"
            ),
            MacroError::SwapNeedsMadNop { macro_index } => write!(
                f,
                "macro {macro_index}: SFPSWAP on the Simple sub-unit needs SFPNOP on the MAD \
                 sub-unit for the same time"
            ),
            MacroError::SwapNextCycleBusy { macro_index } => write!(
                f,
                "macro {macro_index}: after SFPSWAP the Round sub-unit must be idle or SFPNOP \
                 on the next cycle"
            ),
            MacroError::SwapDelayKind { macro_index } => write!(
                f,
                "macro {macro_index}: SFPSWAP and its MAD SFPNOP use different delay kinds"
            ),
            MacroError::Encode(e) => write!(f, "{e}"),
        }
    }
}

impl From<sfpu::EncodeError> for MacroError {
    fn from(e: sfpu::EncodeError) -> Self {
        MacroError::Encode(e)
    }
}

/// One of the four sub-units a macro schedules onto, in sequence-byte order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SubUnit {
    Simple = 0,
    Mad = 1,
    Round = 2,
    Store = 3,
}

impl SubUnit {
    /// In sequence-byte order.
    pub const ALL: [SubUnit; 4] = [
        SubUnit::Simple,
        SubUnit::Mad,
        SubUnit::Round,
        SubUnit::Store,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn name(self) -> &'static str {
        match self {
            SubUnit::Simple => "Simple",
            SubUnit::Mad => "MAD",
            SubUnit::Round => "Round",
            SubUnit::Store => "Store",
        }
    }

    /// The table at the top of `SFPLOADMACRO.md`: may this sub-unit execute an
    /// instruction with this mnemonic (the generated tables' spelling, so
    /// `SFPSTOCHRND` is `SFP_STOCH_RND`)?
    pub fn can_execute(self, mnemonic: &str) -> bool {
        match self {
            SubUnit::Simple => matches!(
                mnemonic,
                "SFPABS"
                    | "SFPAND"
                    | "SFPARECIP"
                    | "SFPCAST"
                    | "SFPCOMPC"
                    | "SFPCONFIG"
                    | "SFPDIVP2"
                    | "SFPENCC"
                    | "SFPEXEXP"
                    | "SFPEXMAN"
                    | "SFPGT"
                    | "SFPIADD"
                    | "SFPLE"
                    | "SFPLZ"
                    | "SFPMOV"
                    | "SFPNOP"
                    | "SFPNOT"
                    | "SFPOR"
                    | "SFPPOPC"
                    | "SFPPUSHC"
                    | "SFPSETCC"
                    | "SFPSETEXP"
                    | "SFPSETMAN"
                    | "SFPSETSGN"
                    | "SFPSHFT"
                    | "SFPSWAP"
                    | "SFPTRANSP"
                    | "SFPXOR"
            ),
            SubUnit::Mad => matches!(
                mnemonic,
                "SFPADD"
                    | "SFPADDI"
                    | "SFPLUT"
                    | "SFPLUTFP32"
                    | "SFPMAD"
                    | "SFPMUL"
                    | "SFPMULI"
                    | "SFPMUL24"
                    | "SFPNOP"
            ),
            SubUnit::Round => matches!(mnemonic, "SFPNOP" | "SFPSHFT2" | "SFP_STOCH_RND"),
            SubUnit::Store => mnemonic == "SFPSTORE",
        }
    }
}

/// What one sub-unit is told to do (`SequenceBits & 7`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// Code 0: nothing is scheduled (the delay field is still honoured as a
    /// forgetting delay, which is why [`Slot::IDLE`] uses delay 7).
    Idle,
    /// Code 2: an `SFPNOP`.
    Nop,
    /// Code 3: `SFPSTORE` with `VD = 0`. Store sub-unit only.
    Store,
    /// Codes 4..=7: `InstructionTemplate[t]`.
    Template(u8),
}

impl Source {
    pub const fn code(self) -> u8 {
        match self {
            Source::Idle => 0,
            Source::Nop => 2,
            Source::Store => 3,
            Source::Template(t) => 4 + t,
        }
    }

    pub const fn from_code(code: u8) -> Result<Source, MacroError> {
        match code & 7 {
            0 => Ok(Source::Idle),
            1 => Err(MacroError::UndefinedSource),
            2 => Ok(Source::Nop),
            3 => Ok(Source::Store),
            c => Ok(Source::Template(c - 4)),
        }
    }
}

/// One sub-unit's byte of a sequence.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Slot {
    source: Source,
    delay: u8,
    vd16: bool,
    vb: bool,
}

impl Slot {
    /// Schedules nothing and forgets nothing: delay 7, which the page exempts from
    /// the forgetting step. The idle a sequence should use.
    pub const IDLE: Slot = Slot {
        source: Source::Idle,
        delay: NO_FORGET,
        vd16: false,
        vb: false,
    };

    /// A scheduled instruction `delay` after the macro. `Template(t)` needs `t < 4`.
    pub const fn new(source: Source, delay: u32) -> Result<Slot, MacroError> {
        if delay > MAX_DELAY as u32 {
            return Err(MacroError::Delay { value: delay });
        }
        if let Source::Template(t) = source {
            if t as usize >= TEMPLATES {
                return Err(MacroError::TemplateIndex { value: t as u32 });
            }
        }
        Ok(Slot {
            source,
            delay: delay as u8,
            vd16: false,
            vb: false,
        })
    }

    /// Write `LReg[16]` rather than the macro's `VD`; on the Store sub-unit, read it.
    pub const fn vd16(mut self) -> Slot {
        self.vd16 = true;
        self
    }

    /// Substitute the macro's `VD` for the template's `VB` rather than its `VC`; on
    /// the Store sub-unit, keep the template's own `VD` as the source register.
    pub const fn substitute_vb(mut self) -> Slot {
        self.vb = true;
        self
    }

    pub const fn source(self) -> Source {
        self.source
    }
    pub const fn delay(self) -> u8 {
        self.delay
    }
    pub const fn writes_lreg16(self) -> bool {
        self.vd16
    }
    pub const fn substitutes_vb(self) -> bool {
        self.vb
    }
    pub const fn is_idle(self) -> bool {
        matches!(self.source, Source::Idle)
    }

    pub const fn bits(self) -> u8 {
        self.source.code()
            | (self.delay << 3)
            | if self.vd16 { 0x40 } else { 0 }
            | if self.vb { 0x80 } else { 0 }
    }

    pub const fn from_bits(bits: u8) -> Result<Slot, MacroError> {
        match Source::from_code(bits) {
            Ok(source) => Ok(Slot {
                source,
                delay: (bits >> 3) & 7,
                vd16: bits & 0x40 != 0,
                vb: bits & 0x80 != 0,
            }),
            Err(e) => Err(e),
        }
    }
}

/// `Sequence[macro]`: one [`Slot`] per sub-unit.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Sequence([Slot; 4]);

impl Sequence {
    /// Every sub-unit idle with delay 7: schedules nothing, forgets nothing.
    pub const IDLE: Sequence = Sequence([Slot::IDLE; 4]);

    pub const fn new(simple: Slot, mad: Slot, round: Slot, store: Slot) -> Sequence {
        Sequence([simple, mad, round, store])
    }

    pub const fn slot(self, unit: SubUnit) -> Slot {
        self.0[unit.index()]
    }

    pub const fn bits(self) -> u32 {
        (self.0[0].bits() as u32)
            | (self.0[1].bits() as u32) << 8
            | (self.0[2].bits() as u32) << 16
            | (self.0[3].bits() as u32) << 24
    }

    pub const fn from_bits(bits: u32) -> Result<Sequence, MacroError> {
        let mut slots = [Slot::IDLE; 4];
        let mut i = 0;
        while i < 4 {
            slots[i] = match Slot::from_bits((bits >> (8 * i)) as u8) {
                Ok(s) => s,
                Err(e) => return Err(e),
            };
            i += 1;
        }
        Ok(Sequence(slots))
    }
}

/// `LoadMacroConfig.Misc`: twelve bits, least significant first.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Misc {
    store_mod0: u8,
    uses_load_mod0: u8,
    delay_instructions: u8,
}

impl Misc {
    /// `StoreMod0 = 0`, every macro's store using it, every sub-unit counting delays
    /// in cycles.
    pub const ZERO: Misc = Misc {
        store_mod0: 0,
        uses_load_mod0: 0,
        delay_instructions: 0,
    };

    /// The `Mod0` scheduled stores use unless a macro says otherwise.
    pub const fn store_mod0(mut self, mod0: u32) -> Result<Misc, MacroError> {
        if mod0 > 15 {
            return Err(MacroError::Mod0 { value: mod0 });
        }
        self.store_mod0 = mod0 as u8;
        Ok(self)
    }

    /// Macro `m`'s scheduled store takes the `SFPLOADMACRO`'s own `Mod0`.
    pub const fn store_uses_load_mod0(mut self, m: u32) -> Result<Misc, MacroError> {
        if m as usize >= MACROS {
            return Err(MacroError::MacroIndex { value: m });
        }
        self.uses_load_mod0 |= 1 << m;
        Ok(self)
    }

    /// `unit` counts its delay in elapsed vector *instructions* rather than cycles
    /// (`WaitForElapsedInstructions`). Instruction counting does not depend on how
    /// fast the thread pushes, which is why it is the kind for a pushed stream.
    pub const fn count_instructions(mut self, unit: SubUnit) -> Misc {
        self.delay_instructions |= 1 << unit.index();
        self
    }

    /// Every sub-unit counts instructions.
    pub const fn count_all_instructions(mut self) -> Misc {
        self.delay_instructions = 0xf;
        self
    }

    pub const fn store_mod0_value(self) -> u32 {
        self.store_mod0 as u32
    }
    pub const fn store_uses_load_mod0_of(self, m: u32) -> bool {
        self.uses_load_mod0 >> m & 1 != 0
    }
    pub const fn counts_instructions(self, unit: SubUnit) -> bool {
        self.delay_instructions >> unit.index() & 1 != 0
    }

    /// The twelve bits as `SFPCONFIG` writes them.
    pub const fn bits(self) -> u32 {
        (self.store_mod0 as u32)
            | (self.uses_load_mod0 as u32) << 4
            | (self.delay_instructions as u32) << 8
    }

    pub const fn from_bits(bits: u32) -> Misc {
        Misc {
            store_mod0: (bits & 0xf) as u8,
            uses_load_mod0: (bits >> 4 & 0xf) as u8,
            delay_instructions: (bits >> 8 & 0xf) as u8,
        }
    }
}

/// `InstructionTemplate[t]`: the 32-bit word of any Vector Unit instruction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Template(u32);

impl Template {
    pub const fn from_instruction(i: Instruction) -> Template {
        Template(i.word())
    }

    /// A template from a raw word, checked to be a Vector Unit instruction.
    pub fn from_word(word: u32) -> Result<Template, MacroError> {
        decode(word).map(|_| Template(word))
    }

    pub const fn word(self) -> u32 {
        self.0
    }

    /// The instruction this word is, by the generated tables.
    pub fn def(self) -> Result<&'static InstructionDef, MacroError> {
        decode(self.0)
    }

    /// The template as an [`Instruction`], to read its operands.
    pub fn instruction(self) -> Result<Instruction, MacroError> {
        Ok(Instruction::new(self.0, self.def()?))
    }
}

/// Which Vector Unit instruction a word is.
///
/// Only encodings the documentation lists for Blackhole (its own, or shared with
/// Wormhole) are candidates. Two opcodes carry more than one form --
/// `SFPSTOCHRND`'s `FloatFloat` and `IntInt`/`FloatInt` flavours (opcode `0x8e`)
/// and `SFPSHFT2`'s register and immediate forms (`0x94`) -- and the bits that
/// tell them apart are a mode, so: `0x8e` with `Mod1 <= 1` and its unused bits
/// clear is the float form; any other `0x8e` is the integer form; `0x94` is the
/// register form (the immediate form is never needed to *classify* the word, and
/// no model executes either).
pub fn decode(word: u32) -> Result<&'static InstructionDef, MacroError> {
    let opcode = (word >> 24) as u8;
    if opcode == 0x8e {
        let float = &defs::SFPSTOCHRND;
        let int = &defs::SFPSTOCHRNDi;
        return Ok(if word & 0xf <= 1 && word & float.unspecified() == 0 {
            float
        } else {
            int
        });
    }
    if opcode == 0x94 {
        return Ok(&defs::SFPSHFT2);
    }
    let mut found: Option<&'static InstructionDef> = None;
    for def in ALL {
        if def.mnemonic().starts_with("SFP")
            && def.provenance().is_documented_for_blackhole()
            && def.matches(word)
        {
            found = Some(*def);
            break;
        }
    }
    found.ok_or(MacroError::NotAVectorInstruction { word })
}

/// `SFPLOADMACRO` itself, with its fields checked.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MacroLoad {
    macro_index: u8,
    vd: u8,
    mod0: u8,
    addr_mod: u8,
    imm10: u16,
}

impl MacroLoad {
    /// Macro `macro_index`, loading `Dst[imm10]` into `LReg[vd]`.
    ///
    /// `vd` is `VDHi << 2 | VDLo`, three bits, and `VDHi` *is* address bit 0
    /// (`Imm10 = Imm9 << 1 | VDHi`), so `imm10`'s low bit must equal `vd >> 2`.
    /// An `SFPLOAD` address's bit 0 is unused unless a nonzero `Dst` RWC offset
    /// carries into bit 1, so an odd address with `vd >= 4` is harmless exactly
    /// when the thread's `RWC.Dst + DEST_REGW_BASE` is a multiple of four; the
    /// model (`macro_sched`) computes the carry rather than assuming it away.
    pub const fn new(
        macro_index: u32,
        vd: u32,
        mod0: u32,
        addr_mod: u32,
        imm10: u32,
    ) -> Result<MacroLoad, MacroError> {
        if macro_index as usize >= MACROS {
            return Err(MacroError::MacroIndex { value: macro_index });
        }
        if vd > 7 {
            return Err(MacroError::LoadVd { value: vd });
        }
        if mod0 > 15 {
            return Err(MacroError::Mod0 { value: mod0 });
        }
        if addr_mod > 7 {
            return Err(MacroError::AddrMod { value: addr_mod });
        }
        if imm10 > 0x3ff {
            return Err(MacroError::Imm10 { value: imm10 });
        }
        if imm10 & 1 != vd >> 2 {
            return Err(MacroError::VdHiCoupled { vd, imm10 });
        }
        Ok(MacroLoad {
            macro_index: macro_index as u8,
            vd: vd as u8,
            mod0: mod0 as u8,
            addr_mod: addr_mod as u8,
            imm10: imm10 as u16,
        })
    }

    pub const fn macro_index(self) -> u32 {
        self.macro_index as u32
    }
    pub const fn vd(self) -> u32 {
        self.vd as u32
    }
    pub const fn mod0(self) -> u32 {
        self.mod0 as u32
    }
    pub const fn addr_mod(self) -> u32 {
        self.addr_mod as u32
    }
    pub const fn imm10(self) -> u32 {
        self.imm10 as u32
    }

    /// The instruction, through the generated `SFPLOADMACRO_BH` encoder.
    pub const fn encode(self) -> Instruction {
        let r = encode::Sfploadmacro::ZERO
            .macro_index(self.macro_index as u32)
            .vd_lo(self.vd as u32 & 3)
            .mod0(self.mod0 as u32)
            .addr_mod(self.addr_mod as u32)
            .imm9(self.imm10 as u32 >> 1)
            .vd_hi(self.vd as u32 >> 2)
            .encode();
        match r {
            Ok(i) => i,
            // Every field was range-checked in `new`.
            Err(_) => unreachable!(),
        }
    }

    /// Read a `MacroLoad` back out of an instruction (for tests and the model).
    pub fn from_instruction(i: Instruction) -> Option<MacroLoad> {
        if !core::ptr::eq(i.def(), &defs::SFPLOADMACRO) {
            return None;
        }
        let op = |n: &str| i.operand(n).unwrap_or(0);
        let vd = op("VDHi") << 2 | op("VDLo");
        let imm10 = op("Imm9") << 1 | op("VDHi");
        MacroLoad::new(op("MacroIndex"), vd, op("Mod0"), op("AddrMod"), imm10).ok()
    }
}

/// The `SFPLOADI`/`SFPCONFIG` words that write a configuration.
#[derive(Copy, Clone, Debug)]
pub struct ConfigWrites {
    ins: [Instruction; CONFIG_MAX],
    len: usize,
}

impl ConfigWrites {
    const fn new() -> ConfigWrites {
        ConfigWrites {
            ins: [sfpu::nop(); CONFIG_MAX],
            len: 0,
        }
    }

    fn push(&mut self, i: Instruction) {
        self.ins[self.len] = i;
        self.len += 1;
    }

    pub fn as_slice(&self) -> &[Instruction] {
        &self.ins[..self.len]
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Stage `word` in `LReg[0]` and `SFPCONFIG` it into `target`.
    fn write_word(&mut self, target: u32, word: u32) -> Result<(), MacroError> {
        for i in sfpu::load_f32(0, word)? {
            self.push(i);
        }
        // `SFPCONFIG` reads `LReg[0]`; the page documents no stall for that read, so
        // let the staging load retire first. (A precaution, not a documented need.)
        self.push(sfpu::nop());
        let c = encode::sfpconfig(0, target, 0).map_err(|e| {
            MacroError::Encode(match e {
                crate::isa::EncodeError::FieldTooLarge {
                    field,
                    value,
                    width,
                    ..
                } => sfpu::EncodeError::FieldTooLarge {
                    name: field,
                    value,
                    bits: width as u32,
                },
            })
        })?;
        self.push(c);
        // `SFPCONFIG` has up to two cycles of latency (`VectorUnit.md`).
        self.push(sfpu::nop());
        Ok(())
    }

    /// `SFPCONFIG` of a 16-bit immediate (`MOD1_IMM16_IS_VALUE`) into `target`:
    /// needs no staging register, so it cannot depend on `LReg[0]`.
    fn write_imm(&mut self, target: u32, value: u32) -> Result<(), MacroError> {
        let c = encode::sfpconfig(value, target, CONFIG_IMM16_IS_VALUE)
            .map_err(|_| MacroError::Mod0 { value })?;
        self.push(c);
        self.push(sfpu::nop());
        Ok(())
    }

    /// The preamble of every configuration write: an `SFPNOP`, then `SFPENCC`
    /// turning predication off with every lane's flag set. `SFPCONFIG` honours
    /// `UseLaneFlagsForLaneEnable`/`LaneFlags` of lanes 0..8, so a flag state
    /// left by an earlier program on this (persistent) Vector Unit would make the
    /// write reach only some lanes -- and the page requires every lane's Store
    /// sub-unit to be scheduled alike.
    fn preamble(&mut self) {
        self.push(sfpu::nop());
        self.push(encode::sfpencc(2, 0, 2 | 8).unwrap_or(sfpu::nop()));
    }
}

/// A whole macro configuration: what `SFPCONFIG` has to write before any
/// `SFPLOADMACRO` runs.
///
/// The state is **persistent hardware state**: nothing resets it between programs
/// or processes, so a program that uses a macro writes every register it reads
/// ([`MacroConfig::writes`]), puts [`MacroConfig::key`] in its program key so two
/// programs that differ only in configuration are not mistaken for one, and
/// restores [`MacroConfig::teardown`] when it is done so that an unconfigured
/// `SFPLOADMACRO` elsewhere still begins as a plain `SFPLOAD`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MacroConfig {
    misc: Misc,
    write_misc: bool,
    sequence: [Option<Sequence>; MACROS],
    template: [Option<Template>; TEMPLATES],
}

impl MacroConfig {
    pub const fn new(misc: Misc) -> MacroConfig {
        MacroConfig {
            misc,
            write_misc: true,
            sequence: [None; MACROS],
            template: [None; TEMPLATES],
        }
    }

    /// Leave `Misc` unwritten. An idle macro reads only its `Sequence`: `Misc` is
    /// consulted per scheduled slot (`UnitDelayKind`, `StoreMod0`), and an idle
    /// slot schedules nothing.
    pub const fn without_misc(mut self) -> MacroConfig {
        self.write_misc = false;
        self
    }

    pub const fn with_sequence(mut self, m: u32, s: Sequence) -> Result<MacroConfig, MacroError> {
        if m as usize >= MACROS {
            return Err(MacroError::MacroIndex { value: m });
        }
        self.sequence[m as usize] = Some(s);
        Ok(self)
    }

    pub const fn with_template(mut self, t: u32, w: Template) -> Result<MacroConfig, MacroError> {
        if t as usize >= TEMPLATES {
            return Err(MacroError::TemplateIndex { value: t });
        }
        self.template[t as usize] = Some(w);
        Ok(self)
    }

    pub const fn misc(&self) -> Misc {
        self.misc
    }
    pub const fn sequence(&self, m: usize) -> Option<Sequence> {
        self.sequence[m]
    }
    pub const fn template(&self, t: usize) -> Option<Template> {
        self.template[t]
    }

    /// Check every configured macro's static rules; see [`MacroConfig::validate_macro`].
    pub fn validate(&self) -> Result<(), MacroError> {
        for m in 0..MACROS {
            if self.sequence[m].is_some() {
                self.validate_macro(m as u32)?;
            }
        }
        Ok(())
    }

    /// The rules of `SFPLOADMACRO.md` that a single macro can break by itself:
    /// source/template resolution, the sub-unit class table, the Store sub-unit's
    /// `SFPNOP`, the Simple/Round `VD == 16` pairing and the `SFPSWAP` conditions.
    /// Everything that depends on *when* macros meet is the model's.
    pub fn validate_macro(&self, m: u32) -> Result<(), MacroError> {
        let seq = self
            .sequence
            .get(m as usize)
            .copied()
            .flatten()
            .ok_or(MacroError::MacroIndex { value: m })?;
        // What each sub-unit would execute, by mnemonic, and whether it is a NOP.
        let mut mnemonic: [Option<&'static str>; 4] = [None; 4];
        for unit in SubUnit::ALL {
            let slot = seq.slot(unit);
            let name = match slot.source() {
                Source::Idle => continue,
                Source::Nop => "SFPNOP",
                Source::Store => "SFPSTORE",
                Source::Template(t) => {
                    let tpl = self.template[t as usize].ok_or(MacroError::TemplateMissing {
                        macro_index: m,
                        template: t as u32,
                    })?;
                    tpl.def()?.mnemonic()
                }
            };
            if unit == SubUnit::Store && name == "SFPNOP" {
                return Err(MacroError::StoreNop { macro_index: m });
            }
            if !unit.can_execute(name) {
                return Err(MacroError::ClassMismatch {
                    macro_index: m,
                    unit,
                    mnemonic: name,
                });
            }
            mnemonic[unit.index()] = Some(name);
        }
        let simple = seq.slot(SubUnit::Simple);
        let mad = seq.slot(SubUnit::Mad);
        let round = seq.slot(SubUnit::Round);
        // Same delay value on both sub-units: they execute together unless the
        // delay kinds differ, in which case "together" is not a fact, so be strict.
        if !simple.is_idle()
            && !round.is_idle()
            && simple.delay() == round.delay()
            && simple.writes_lreg16() == round.writes_lreg16()
        {
            return Err(MacroError::SimpleRoundDestination { macro_index: m });
        }
        if mnemonic[SubUnit::Simple.index()] == Some("SFPSWAP") {
            let mad_nop =
                mnemonic[SubUnit::Mad.index()] == Some("SFPNOP") && mad.delay() == simple.delay();
            if !mad_nop {
                return Err(MacroError::SwapNeedsMadNop { macro_index: m });
            }
            if self.misc.counts_instructions(SubUnit::Mad)
                != self.misc.counts_instructions(SubUnit::Simple)
            {
                return Err(MacroError::SwapDelayKind { macro_index: m });
            }
            if !round.is_idle()
                && round.delay() == simple.delay().wrapping_add(1)
                && mnemonic[SubUnit::Round.index()] != Some("SFPNOP")
            {
                return Err(MacroError::SwapNextCycleBusy { macro_index: m });
            }
        }
        Ok(())
    }

    /// The instructions that write this configuration: a leading `SFPNOP` (so a
    /// `SFPMAD` still in flight cannot land on `LReg[0]` after the staging load),
    /// then for each template and sequence two `SFPLOADI`s into `LReg[0]` and an
    /// `SFPCONFIG` from it, and `Misc` as an immediate `SFPCONFIG`.
    ///
    /// **Clobbers `LReg[0]`.** `SFPCONFIG` takes lanes 0..8 of `LReg[0]` and
    /// broadcasts them, and honours `UseLaneFlagsForLaneEnable`/`LaneFlags` of lanes
    /// 0..8, so it belongs outside any predicated scope with every lane enabled
    /// (the model refuses a write that reaches only some lanes). Not validated
    /// here -- call [`MacroConfig::validate`] first.
    pub fn writes(&self) -> Result<ConfigWrites, MacroError> {
        let mut w = ConfigWrites::new();
        w.preamble();
        for (t, tpl) in self.template.iter().enumerate() {
            if let Some(tpl) = tpl {
                w.write_word(config_target::template(t as u32), tpl.word())?;
            }
        }
        for (m, seq) in self.sequence.iter().enumerate() {
            if let Some(seq) = seq {
                w.write_word(config_target::sequence(m as u32), seq.bits())?;
            }
        }
        if self.write_misc {
            w.write_imm(config_target::MISC, self.misc.bits())?;
        }
        Ok(w)
    }

    /// The configuration as words, for a program key or descriptor: `Misc`, then
    /// the four sequences, then the four templates. An unconfigured entry is `0`
    /// with its presence in the second return value.
    pub const fn descriptor(&self) -> ([u32; 9], u16) {
        let mut words = [0u32; 9];
        let mut present = 0u16;
        words[0] = self.misc.bits();
        if self.write_misc {
            present |= 1 << 8;
        }
        let mut i = 0;
        while i < MACROS {
            if let Some(s) = self.sequence[i] {
                words[1 + i] = s.bits();
                present |= 1 << i;
            }
            i += 1;
        }
        let mut t = 0;
        while t < TEMPLATES {
            if let Some(w) = self.template[t] {
                words[5 + t] = w.word();
                present |= 1 << (4 + t);
            }
            t += 1;
        }
        (words, present)
    }

    /// A stable 64-bit digest of [`MacroConfig::descriptor`] (FNV-1a), for program
    /// cache keys.
    pub const fn key(&self) -> u64 {
        let (words, present) = self.descriptor();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut i = 0;
        while i < 9 {
            let w = words[i];
            let mut b = 0;
            while b < 4 {
                h ^= ((w >> (8 * b)) & 0xff) as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
                b += 1;
            }
            i += 1;
        }
        h ^= present as u64;
        h.wrapping_mul(0x0000_0100_0000_01b3)
    }

    /// Restore the macro registers to zero: every `Sequence` and `Misc`, by
    /// immediate `SFPCONFIG` (no staging register). Zero is a sequence of idle
    /// sub-units (source 0 everywhere), so nothing can be scheduled by a later
    /// `SFPLOADMACRO`. **The pages state no reset value** for any macro register;
    /// zero is this repository's convention for "nothing configured", chosen
    /// because an idle sequence is the one value the page defines the effect of.
    /// Templates are left as they are: they are read only through a non-idle
    /// slot. Run after a program that used a macro, and (with a drain) before one
    /// that must start from a known state.
    pub fn teardown() -> Result<ConfigWrites, MacroError> {
        let mut w = ConfigWrites::new();
        w.preamble();
        for m in 0..MACROS as u32 {
            w.write_imm(config_target::sequence(m), 0)?;
        }
        w.write_imm(config_target::MISC, 0)?;
        Ok(w)
    }
}

/// The `SFPNOP`s that retire everything pending ([`DRAIN_NOPS`]).
pub fn drain() -> [Instruction; DRAIN_NOPS] {
    [sfpu::nop(); DRAIN_NOPS]
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;
    use std::vec::Vec;

    fn slot(source: Source, delay: u32) -> Slot {
        Slot::new(source, delay).unwrap()
    }

    /// The byte layout, bit by bit, from the page's functional model:
    /// `Delay = (b >> 3) & 7`, template `b & 7`, `0x40` VD16, `0x80` VB.
    #[test]
    fn a_slot_byte_is_the_pages_layout() {
        assert_eq!(slot(Source::Template(0), 0).bits(), 0x04);
        assert_eq!(slot(Source::Template(3), 5).bits(), 0x07 | 5 << 3);
        assert_eq!(slot(Source::Nop, 2).bits(), 0x02 | 2 << 3);
        assert_eq!(slot(Source::Store, 7).bits(), 0x03 | 0x38);
        assert_eq!(
            slot(Source::Template(1), 1).vd16().bits(),
            0x05 | 0x08 | 0x40
        );
        assert_eq!(
            slot(Source::Template(2), 3).substitute_vb().bits(),
            0x06 | 3 << 3 | 0x80
        );
        for bits in 0u8..=255 {
            match Slot::from_bits(bits) {
                Ok(s) => assert_eq!(s.bits(), bits),
                Err(e) => assert_eq!((bits & 7, e), (1, MacroError::UndefinedSource)),
            }
        }
        // Code 1 is the one undefined source.
        assert_eq!(Slot::from_bits(0x01), Err(MacroError::UndefinedSource));
        assert_eq!(
            Slot::new(Source::Nop, 8),
            Err(MacroError::Delay { value: 8 })
        );
        assert_eq!(
            Slot::new(Source::Template(4), 0),
            Err(MacroError::TemplateIndex { value: 4 })
        );
    }

    #[test]
    fn a_sequence_puts_sub_unit_i_in_byte_i() {
        let s = Sequence::new(
            slot(Source::Template(0), 1),
            slot(Source::Template(1), 2),
            slot(Source::Template(2), 3),
            slot(Source::Store, 4),
        );
        assert_eq!(s.bits(), 0x0c | 0x15 << 8 | 0x1e << 16 | 0x23 << 24);
        assert_eq!(Sequence::from_bits(s.bits()), Ok(s));
        assert_eq!(Sequence::IDLE.bits(), 0x3838_3838);
    }

    /// `Misc`: StoreMod0 in bits 3:0, UsesLoadMod0ForStore 7:4, UnitDelayKind 11:8.
    #[test]
    fn misc_is_twelve_tightly_packed_bits() {
        let m = Misc::ZERO
            .store_mod0(3)
            .unwrap()
            .store_uses_load_mod0(1)
            .unwrap()
            .count_instructions(SubUnit::Round)
            .count_instructions(SubUnit::Store);
        assert_eq!(m.bits(), 0x3 | 0x2 << 4 | 0b1100 << 8);
        assert_eq!(Misc::from_bits(m.bits()), m);
        assert!(m.counts_instructions(SubUnit::Round) && !m.counts_instructions(SubUnit::Mad));
        assert_eq!(
            Misc::ZERO.store_mod0(16),
            Err(MacroError::Mod0 { value: 16 })
        );
        assert_eq!(
            Misc::ZERO.store_uses_load_mod0(4),
            Err(MacroError::MacroIndex { value: 4 })
        );
    }

    /// The generated `SFPLOADMACRO_BH` against the page's syntax block:
    /// `(MacroIndex << 2) + VDLo`, `Mod0`, `AddrMod`, `(Imm9 << 1) + VDHi`.
    #[test]
    fn the_load_encodes_as_the_page_says() {
        let l = MacroLoad::new(2, 5, 3, 6, 0x155).unwrap();
        let i = l.encode();
        let w = i.word();
        assert_eq!(w >> 24, 0x93);
        assert_eq!(w >> 22 & 3, 2, "MacroIndex");
        assert_eq!(w >> 20 & 3, 1, "VDLo");
        assert_eq!(w >> 16 & 15, 3, "Mod0");
        assert_eq!(w >> 13 & 7, 6, "AddrMod");
        assert_eq!(w >> 1 & 0x1ff, 0xaa, "Imm9");
        assert_eq!(w & 1, 1, "VDHi");
        assert_eq!(w & 0x1c00, 0, "bits 12:10 are unused");
        assert_eq!(MacroLoad::from_instruction(i), Some(l));
    }

    #[test]
    fn the_load_refuses_what_it_cannot_say() {
        assert_eq!(
            MacroLoad::new(4, 0, 0, 0, 0),
            Err(MacroError::MacroIndex { value: 4 })
        );
        assert_eq!(
            MacroLoad::new(0, 8, 0, 0, 1),
            Err(MacroError::LoadVd { value: 8 })
        );
        assert_eq!(
            MacroLoad::new(0, 0, 16, 0, 0),
            Err(MacroError::Mod0 { value: 16 })
        );
        assert_eq!(
            MacroLoad::new(0, 0, 0, 8, 0),
            Err(MacroError::AddrMod { value: 8 })
        );
        assert_eq!(
            MacroLoad::new(0, 0, 0, 0, 1024),
            Err(MacroError::Imm10 { value: 1024 })
        );
        // VDHi is address bit 0.
        assert_eq!(
            MacroLoad::new(0, 4, 0, 0, 0),
            Err(MacroError::VdHiCoupled { vd: 4, imm10: 0 })
        );
        assert_eq!(
            MacroLoad::new(0, 3, 0, 0, 1),
            Err(MacroError::VdHiCoupled { vd: 3, imm10: 1 })
        );
        for vd in 0..8 {
            assert!(
                MacroLoad::new(0, vd, 3, 0, 8 | (vd >> 2)).is_ok(),
                "vd {vd}"
            );
        }
    }

    fn store_template(vd: u32) -> Template {
        Template::from_instruction(encode::sfpstore(vd, 3, 0, 0).unwrap())
    }

    fn tpl(i: Instruction) -> Template {
        Template::from_instruction(i)
    }

    #[test]
    fn templates_decode_to_their_instruction() {
        let cases = [
            (encode::sfpnot(1, 2).unwrap(), "SFPNOT"),
            (sfpu::mad(0, 1, 2, 3, 0).unwrap(), "SFPMAD"),
            (
                encode::Sfpmul::ZERO
                    .va(0)
                    .vb(1)
                    .vc(9)
                    .vd(3)
                    .encode()
                    .unwrap(),
                "SFPMUL",
            ),
            (encode::sfpmuli(0x4040, 1, 0).unwrap(), "SFPMULI"),
            (encode::sfpconfig(0, 5, 0).unwrap(), "SFPCONFIG"),
            (
                encode::Sfpstochrnd::ZERO
                    .mod1(1)
                    .vc(1)
                    .vb(1)
                    .vd(2)
                    .encode()
                    .unwrap(),
                "SFP_STOCH_RND",
            ),
            (encode::sfpstore(0, 3, 0, 0).unwrap(), "SFPSTORE"),
            (sfpu::nop(), "SFPNOP"),
        ];
        for (i, want) in cases {
            let t = Template::from_instruction(i);
            assert_eq!(t.def().unwrap().mnemonic(), want, "{i:?}");
            assert_eq!(t.instruction().unwrap(), i);
        }
        // A non-vector word is refused.
        assert!(matches!(
            Template::from_word(0x0100_0000),
            Err(MacroError::NotAVectorInstruction { .. })
        ));
    }

    fn base() -> MacroConfig {
        MacroConfig::new(Misc::ZERO)
            .with_template(0, tpl(encode::sfpnot(0, 0).unwrap()))
            .unwrap()
            .with_template(1, tpl(encode::sfpmuli(0x4040, 0, 0).unwrap()))
            .unwrap()
            .with_template(2, tpl(encode::Sfpstochrnd::ZERO.mod1(1).encode().unwrap()))
            .unwrap()
            .with_template(3, store_template(2))
            .unwrap()
    }

    fn with_seq(c: MacroConfig, s: Sequence) -> MacroConfig {
        c.with_sequence(0, s).unwrap()
    }

    #[test]
    fn a_well_formed_macro_validates() {
        let s = Sequence::new(
            slot(Source::Template(0), 2),
            slot(Source::Template(1), 0),
            slot(Source::Template(2), 3),
            slot(Source::Template(3), 4).substitute_vb(),
        );
        with_seq(base(), s).validate().unwrap();
        with_seq(base(), Sequence::IDLE).validate().unwrap();
    }

    #[test]
    fn static_rules_refuse_what_the_page_calls_undefined_or_wrong() {
        let c = base();
        // A MAD instruction on the Simple sub-unit would become an SFPNOP.
        let s = Sequence::new(
            slot(Source::Template(1), 0),
            Slot::IDLE,
            Slot::IDLE,
            Slot::IDLE,
        );
        assert_eq!(
            with_seq(c, s).validate(),
            Err(MacroError::ClassMismatch {
                macro_index: 0,
                unit: SubUnit::Simple,
                mnemonic: "SFPMULI"
            })
        );
        // SFPSTORE as a source on a non-Store unit, and SFPNOP on the Store unit.
        let s = Sequence::new(slot(Source::Store, 0), Slot::IDLE, Slot::IDLE, Slot::IDLE);
        assert!(matches!(
            with_seq(c, s).validate(),
            Err(MacroError::ClassMismatch {
                mnemonic: "SFPSTORE",
                ..
            })
        ));
        let s = Sequence::new(Slot::IDLE, Slot::IDLE, Slot::IDLE, slot(Source::Nop, 0));
        assert_eq!(
            with_seq(c, s).validate(),
            Err(MacroError::StoreNop { macro_index: 0 })
        );
        // A non-store template on the Store sub-unit.
        let s = Sequence::new(
            Slot::IDLE,
            Slot::IDLE,
            Slot::IDLE,
            slot(Source::Template(0), 0),
        );
        assert!(matches!(
            with_seq(c, s).validate(),
            Err(MacroError::ClassMismatch {
                unit: SubUnit::Store,
                ..
            })
        ));
        // A template the configuration does not hold.
        let empty = MacroConfig::new(Misc::ZERO);
        let s = Sequence::new(
            slot(Source::Template(0), 0),
            Slot::IDLE,
            Slot::IDLE,
            Slot::IDLE,
        );
        assert_eq!(
            with_seq(empty, s).validate(),
            Err(MacroError::TemplateMissing {
                macro_index: 0,
                template: 0
            })
        );
    }

    #[test]
    fn simple_and_round_on_one_cycle_need_exactly_one_lreg16() {
        let c = base();
        let mk = |simple16: bool, round16: bool, round_delay: u32| {
            let mut a = slot(Source::Template(0), 2);
            let mut b = slot(Source::Template(2), round_delay);
            if simple16 {
                a = a.vd16();
            }
            if round16 {
                b = b.vd16();
            }
            with_seq(c, Sequence::new(a, Slot::IDLE, b, Slot::IDLE)).validate()
        };
        let bad = Err(MacroError::SimpleRoundDestination { macro_index: 0 });
        assert_eq!(mk(false, false, 2), bad);
        assert_eq!(mk(true, true, 2), bad);
        assert_eq!(mk(true, false, 2), Ok(()));
        assert_eq!(mk(false, true, 2), Ok(()));
        // Different cycles: no constraint.
        assert_eq!(mk(false, false, 3), Ok(()));
    }

    #[test]
    fn sfpswap_in_the_simple_sub_unit_has_its_conditions() {
        let swap = tpl(encode::sfpswap(1, 2, 0).unwrap());
        let c = base().with_template(0, swap).unwrap();
        let nop_mad = slot(Source::Nop, 1);
        // The MAD NOP for the same time, and nothing after: fine.
        let ok = Sequence::new(
            slot(Source::Template(0), 1),
            nop_mad,
            Slot::IDLE,
            Slot::IDLE,
        );
        assert_eq!(with_seq(c, ok).validate(), Ok(()));
        // No MAD NOP.
        let s = Sequence::new(
            slot(Source::Template(0), 1),
            Slot::IDLE,
            Slot::IDLE,
            Slot::IDLE,
        );
        assert_eq!(
            with_seq(c, s).validate(),
            Err(MacroError::SwapNeedsMadNop { macro_index: 0 })
        );
        // MAD NOP at another time.
        let s = Sequence::new(
            slot(Source::Template(0), 1),
            slot(Source::Nop, 2),
            Slot::IDLE,
            Slot::IDLE,
        );
        assert_eq!(
            with_seq(c, s).validate(),
            Err(MacroError::SwapNeedsMadNop { macro_index: 0 })
        );
        // A Round instruction on the next cycle.
        let s = Sequence::new(
            slot(Source::Template(0), 1),
            nop_mad,
            slot(Source::Template(2), 2),
            Slot::IDLE,
        );
        assert_eq!(
            with_seq(c, s).validate(),
            Err(MacroError::SwapNextCycleBusy { macro_index: 0 })
        );
        // ... unless it is an SFPNOP.
        let s = Sequence::new(
            slot(Source::Template(0), 1),
            nop_mad,
            slot(Source::Nop, 2),
            Slot::IDLE,
        );
        assert_eq!(with_seq(c, s).validate(), Ok(()));
        // Different delay kinds make "the same time" untrue.
        let kinds = MacroConfig::new(Misc::ZERO.count_instructions(SubUnit::Mad))
            .with_template(0, swap)
            .unwrap();
        assert_eq!(
            with_seq(kinds, ok).validate(),
            Err(MacroError::SwapDelayKind { macro_index: 0 })
        );
    }

    /// The configuration is written the way `SFPCONFIG.md` reads it: a template or
    /// sequence word comes from `LReg[0]` (`Mod1 = 0`), `Misc` from `Imm16` with
    /// `MOD1_IMM16_IS_VALUE`, `VD` selecting the register.
    #[test]
    fn a_configuration_is_written_through_sfpconfig() {
        let s = Sequence::new(
            slot(Source::Template(0), 1),
            Slot::IDLE,
            Slot::IDLE,
            slot(Source::Store, 2),
        );
        let c = MacroConfig::new(Misc::ZERO.count_all_instructions())
            .with_template(1, tpl(encode::sfpnot(1, 1).unwrap()))
            .unwrap()
            .with_template(0, tpl(encode::sfpnot(0, 0).unwrap()))
            .unwrap()
            .with_sequence(2, s)
            .unwrap();
        let w = c.writes().unwrap();
        let ins = w.as_slice();
        assert_eq!(ins[0].def().mnemonic(), "SFPNOP");
        let configs: Vec<(u32, u32, u32)> = ins
            .iter()
            .filter(|i| i.def().mnemonic() == "SFPCONFIG")
            .map(|i| {
                (
                    i.operand("VD").unwrap(),
                    i.operand("Mod1").unwrap(),
                    i.operand("Imm16").unwrap(),
                )
            })
            .collect();
        assert_eq!(
            configs,
            [(0, 0, 0), (1, 0, 0), (6, 0, 0), (8, 1, 0xf00)],
            "template 0, template 1, sequence 2, then Misc"
        );
        // Each staged word is the one named, `UPPER` half first.
        let loadi: Vec<(u32, u32)> = ins
            .iter()
            .filter(|i| i.def().mnemonic() == "SFPLOADI")
            .map(|i| (i.operand("Mod0").unwrap(), i.operand("Imm16").unwrap()))
            .collect();
        let sw = s.bits();
        assert_eq!(loadi[4], (sfpu::loadi_mode::UPPER, sw >> 16));
        assert_eq!(loadi[5], (sfpu::loadi_mode::LOWER, sw & 0xffff));
        assert!(w.len() <= CONFIG_MAX);
        // The full configuration is the stated maximum.
        let mut full = base();
        for m in 0..4 {
            full = full.with_sequence(m, Sequence::IDLE).unwrap();
        }
        assert_eq!(full.writes().unwrap().len(), CONFIG_MAX);
    }

    #[test]
    fn the_key_tells_configurations_apart() {
        let a = with_seq(base(), Sequence::IDLE);
        let b = a.with_sequence(1, Sequence::IDLE).unwrap();
        let c = MacroConfig::new(Misc::ZERO.count_all_instructions());
        assert_eq!(a.key(), a.key());
        assert_ne!(a.key(), b.key());
        assert_ne!(a.key(), c.key());
        assert_ne!(
            a.key(),
            with_seq(
                base().with_template(1, store_template(1)).unwrap(),
                Sequence::IDLE
            )
            .key()
        );
        // Presence is part of the key: an unwritten register is not a zero one.
        let zero_seq = MacroConfig::new(Misc::ZERO)
            .with_sequence(0, Sequence::from_bits(0).unwrap())
            .unwrap();
        assert_ne!(zero_seq.key(), MacroConfig::new(Misc::ZERO).key());
    }

    #[test]
    fn teardown_leaves_every_sequence_idle() {
        let w = MacroConfig::teardown().unwrap();
        let ins = w.as_slice();
        let targets: Vec<u32> = ins
            .iter()
            .filter(|i| i.def().mnemonic() == "SFPCONFIG")
            .map(|i| i.operand("VD").unwrap())
            .collect();
        assert_eq!(targets, [4, 5, 6, 7, 8]);
        assert_eq!(drain().len(), DRAIN_NOPS);
        // Every write is an immediate zero: no staging register, no `SFPLOADI`.
        for i in ins.iter().filter(|i| i.def().mnemonic() == "SFPCONFIG") {
            assert_eq!((i.operand("Imm16"), i.operand("Mod1")), (Some(0), Some(1)));
        }
        assert!(ins.iter().all(|i| i.def().mnemonic() != "SFPLOADI"));
        // The preamble turns predication off (`EI`, `RI`; `Imm2 = 2`: flags set,
        // not used).
        let encc = ins[1];
        assert_eq!(encc.def().mnemonic(), "SFPENCC");
        assert_eq!(
            (encc.operand("Imm2"), encc.operand("Mod1")),
            (Some(2), Some(10))
        );
    }

    #[test]
    fn an_idle_macro_needs_only_its_sequence() {
        let c = MacroConfig::new(Misc::ZERO)
            .without_misc()
            .with_sequence(0, Sequence::IDLE)
            .unwrap();
        let targets: Vec<u32> = c
            .writes()
            .unwrap()
            .as_slice()
            .iter()
            .filter(|i| i.def().mnemonic() == "SFPCONFIG")
            .map(|i| i.operand("VD").unwrap())
            .collect();
        assert_eq!(targets, [4]);
        // Leaving Misc out is part of the key: it is a different configuration.
        let with = MacroConfig::new(Misc::ZERO)
            .with_sequence(0, Sequence::IDLE)
            .unwrap();
        assert_ne!(c.key(), with.key());
    }
}
