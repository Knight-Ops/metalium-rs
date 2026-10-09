//! A schedule model for `SFPLOADMACRO` (hardware-coverage S9, lane H).
//!
//! ttsim does not execute `SFPLOADMACRO` (divergence row 7), so this is the only
//! oracle for what a configured macro does. It is written from
//! `SFPLOADMACRO.md`'s functional model and `SFPCONFIG.md`, **not** from
//! `tt_isa::sfpu_macro`'s checked helpers: it re-reads the raw `Sequence`,
//! `Misc` and template words the way the page's pseudocode does, so a helper
//! that encoded a byte wrongly disagrees with this model rather than agreeing
//! with itself. What it shares with the helper is only the generated
//! instruction tables.
//!
//! # Time
//!
//! One regular instruction (or a bubble) is one cycle. At each cycle, in this
//! order: scheduled instructions whose delay has run out execute (Simple, MAD,
//! Round, Store, in that order); the regular instruction executes, unless a
//! scheduled one claims its sub-unit (the hardware discards it silently; this
//! model refuses, because that is never what software meant); the outstanding
//! delays count down (every cycle, or -- if any outstanding instruction has the
//! `WaitForElapsedInstructions` kind -- once per issued instruction); and an
//! `SFPLOADMACRO` issued this cycle forgets what it collides with and schedules
//! its own. A delay of `d` therefore executes `d + 1` cycles after the macro.
//!
//! # What it refuses
//!
//! Everything the page calls `UndefinedBehavior`, and every hazard it states:
//! an undefined source code, a template the sub-unit cannot execute (the
//! hardware substitutes `SFPNOP`; the Store sub-unit "should not be fed
//! `SFPNOP`"), a Simple and a Round instruction on one cycle that do not split
//! `LReg[16]`, the `SFPSWAP` restrictions, a `SFPMAD` result consumed on the
//! next cycle (`SFPMAD.md`: automatic stalling does not apply inside a macro),
//! two scheduled instructions meeting on one sub-unit, a regular instruction
//! meeting a scheduled one, the load's address carry when `VDHi` -- which *is*
//! address bit 0 -- is set, a regular instruction with `VD >= 12` (it would
//! overwrite a template), and `SFPCONFIG` while anything is still pending.
//! Read-after-write between two instructions on one cycle, whose order the page
//! does not state, is refused too.
//!
//! # What it models
//!
//! Execution is the interpreter's ([`super::interp::Vector`]); this module
//! decides *which instruction runs when, with which registers*. The templates
//! it can execute are `SFPMAD`/`SFPADD`/`SFPMUL`, `SFPMULI`/`SFPADDI`, `SFPMOV`,
//! `SFPABS`, `SFPNOT`, `SFPSWAP` and float `SFPSTOCHRND`, plus `SFPNOP` and
//! `SFPSTORE`; any other template is refused by name rather than approximated.
//! `LReg[16]` is held here, since the interpreter's registers stop at 15.

use super::interp::{InterpError, Vector};
use tt_isa::isa::generated::{defs, encode, ALL};
use tt_isa::isa::{Instruction, InstructionDef};

/// The four sub-units a macro schedules onto.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Unit {
    Simple,
    Mad,
    Round,
    Store,
}

impl Unit {
    const ORDER: [Unit; 4] = [Unit::Simple, Unit::Mad, Unit::Round, Unit::Store];

    fn index(self) -> usize {
        self as usize
    }

    /// The table at the top of `SFPLOADMACRO.md`.
    fn can_execute(self, mnemonic: &str) -> bool {
        const SIMPLE: &[&str] = &[
            "SFPABS",
            "SFPAND",
            "SFPARECIP",
            "SFPCAST",
            "SFPCOMPC",
            "SFPCONFIG",
            "SFPDIVP2",
            "SFPENCC",
            "SFPEXEXP",
            "SFPEXMAN",
            "SFPGT",
            "SFPIADD",
            "SFPLE",
            "SFPLZ",
            "SFPMOV",
            "SFPNOP",
            "SFPNOT",
            "SFPOR",
            "SFPPOPC",
            "SFPPUSHC",
            "SFPSETCC",
            "SFPSETEXP",
            "SFPSETMAN",
            "SFPSETSGN",
            "SFPSHFT",
            "SFPSWAP",
            "SFPTRANSP",
            "SFPXOR",
        ];
        const MAD: &[&str] = &[
            "SFPADD",
            "SFPADDI",
            "SFPLUT",
            "SFPLUTFP32",
            "SFPMAD",
            "SFPMUL",
            "SFPMULI",
            "SFPMUL24",
            "SFPNOP",
        ];
        const ROUND: &[&str] = &["SFPNOP", "SFPSHFT2", "SFP_STOCH_RND"];
        match self {
            Unit::Simple => SIMPLE.contains(&mnemonic),
            Unit::Mad => MAD.contains(&mnemonic),
            Unit::Round => ROUND.contains(&mnemonic),
            Unit::Store => mnemonic == "SFPSTORE",
        }
    }
}

/// Why a stream was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SchedError {
    /// The interpreter could not execute an instruction.
    Interp(InterpError),
    /// A macro register the stream never wrote. Silicon retains whatever an
    /// earlier program left, so "unconfigured" is not "zero".
    Unconfigured { what: &'static str, index: u32 },
    /// Sequence source code 1 (`UndefinedBehavior`).
    UndefinedSource { macro_index: u32, unit: Unit },
    /// A template the sub-unit cannot execute (the hardware would substitute `SFPNOP`).
    ClassMismatch { unit: Unit, mnemonic: &'static str },
    /// `SFPNOP` on the Store sub-unit.
    StoreNop,
    /// A template word that is not a Vector Unit instruction.
    NotVector { word: u32 },
    /// A template this model has no execution for.
    Unmodelled { what: String },
    /// Simple and Round on one cycle must split `LReg[16]`.
    SimpleRoundDestination { cycle: u64 },
    /// `SFPSWAP` without the MAD `SFPNOP` of the same time.
    SwapNeedsMadNop { cycle: u64 },
    /// A Simple/Round instruction (or any regular one) the cycle after `SFPSWAP`.
    SwapNextCycle { cycle: u64 },
    /// Two scheduled instructions on one sub-unit on one cycle.
    Collision { unit: Unit, cycle: u64 },
    /// A regular instruction on a sub-unit a scheduled instruction claims.
    RegularCollision { unit: Unit, cycle: u64 },
    /// A `SFPMAD`-class result read on the next cycle.
    MadHazard { reg: u32, cycle: u64 },
    /// One instruction reads or writes what another writes on the same cycle.
    SameCycle { reg: u32, cycle: u64 },
    /// `VD >= 4` makes the load address odd; a thread `Dst` offset with bit 0 set
    /// then carries into the column select.
    VdHiCarry { vd: u32, imm10: u32, rwc_dst: u32 },
    /// A regular instruction whose `VD >= 12` writes an `InstructionTemplate`.
    Backdoor { word: u32 },
    /// `SFPCONFIG` changing macro state while instructions are pending.
    ConfigWhilePending { pending: usize },
    /// An `SFPCONFIG` that does not reach every lane identically.
    ConfigNotUniform,
    /// A mode of `SFPCONFIG` this model does not hold.
    ConfigUnsupported { what: String },
    /// `REPLAY` and other frontend-expanded streams are not modelled here.
    Replay,
}

impl From<InterpError> for SchedError {
    fn from(e: InterpError) -> Self {
        SchedError::Interp(e)
    }
}

impl std::fmt::Display for SchedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SchedError {}

/// What a scheduled slot will do when it executes, resolved from the page's
/// override rules at the time the macro issued.
#[derive(Clone, Debug)]
enum Action {
    Nop,
    /// A Simple, MAD or Round instruction: the template word, the macro's `VD`
    /// and the two override bits.
    Compute {
        word: u32,
        macro_vd: u32,
        vd16: bool,
        vb_sub: bool,
    },
    /// The Store sub-unit's `SFPSTORE`.
    Store {
        src: StoreSource,
        mod0: u32,
        addr: u32,
    },
}

#[derive(Copy, Clone, Debug)]
enum StoreSource {
    Reg(u32),
    Lreg16,
}

#[derive(Clone, Debug)]
struct Pending {
    unit: Unit,
    action: Action,
    remaining: u8,
    by_instructions: bool,
}

/// One executed scheduled instruction, for tests that pin timing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub cycle: u64,
    pub unit: Unit,
    pub what: &'static str,
}

/// `LoadMacroConfig` as the stream has written it, and the schedule it drives.
#[derive(Clone)]
pub struct Machine {
    pub vector: Vector,
    misc: Option<u32>,
    sequence: [Option<u32>; 4],
    template: [Option<u32>; 4],
    pending: Vec<Pending>,
    cycle: u64,
    /// Registers a scheduled MAD wrote on the previous cycle (`reg`, scheduled?).
    mad_busy: Vec<(u32, bool)>,
    /// The cycle on which `SFPSWAP` (or a multi-cycle `SFPSHFT2`) owns the unit.
    owned_cycle: Option<u64>,
    /// `LReg[16]`, which the interpreter's register file does not hold.
    lreg16: [u32; 32],
    /// Scheduled instructions executed so far, oldest first.
    pub events: Vec<Event>,
}

/// Is a register named in a field of this instruction?
fn field(i: Instruction, name: &str) -> Option<u32> {
    i.operand(name)
}

fn with_field(i: Instruction, name: &str, value: u32) -> Instruction {
    let f = i.def().field(name).expect("field present");
    Instruction::new((i.word() & !f.mask()) | f.place(value), i.def())
}

/// The definition of a Vector Unit word: its own lookup, so a template the model
/// accepts is one the generated tables name.
fn decode(word: u32) -> Result<&'static InstructionDef, SchedError> {
    ALL.iter()
        .copied()
        .find(|d| {
            d.mnemonic().starts_with("SFP")
                && d.provenance().is_documented_for_blackhole()
                && d.matches(word)
                && d.key() != defs::SFPSTOCHRNDi.key()
                && d.key() != defs::SFPSHFT2b.key()
        })
        .ok_or(SchedError::NotVector { word })
}

/// Which sub-unit a *regular* instruction occupies; load-style instructions
/// (`SFPLOAD`, `SFPLOADI`, `SFPLOADMACRO`, `SFPNOP`) occupy none of the four.
fn regular_unit(mnemonic: &str) -> Option<Unit> {
    if matches!(mnemonic, "SFPLOAD" | "SFPLOADI" | "SFPLOADMACRO" | "SFPNOP") {
        return None;
    }
    if mnemonic == "SFPSTORE" {
        return Some(Unit::Store);
    }
    [Unit::Simple, Unit::Mad, Unit::Round]
        .into_iter()
        .find(|u| u.can_execute(mnemonic))
}

impl Machine {
    pub fn new(vector: Vector) -> Self {
        Machine {
            vector,
            misc: None,
            sequence: [None; 4],
            template: [None; 4],
            pending: Vec::new(),
            cycle: 0,
            mad_busy: Vec::new(),
            owned_cycle: None,
            lreg16: [0; 32],
            events: Vec::new(),
        }
    }

    /// Instructions scheduled and not yet executed.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    pub fn cycle(&self) -> u64 {
        self.cycle
    }

    /// `LReg[16]`, lane by lane.
    pub fn lreg16(&self) -> [u32; 32] {
        self.lreg16
    }

    /// The raw configuration words, `None` where the stream never wrote one.
    pub fn state(&self) -> (Option<u32>, [Option<u32>; 4], [Option<u32>; 4]) {
        (self.misc, self.sequence, self.template)
    }

    /// Run a stream, one cycle per instruction.
    pub fn run(&mut self, program: &[Instruction]) -> Result<(), SchedError> {
        for &i in program {
            self.issue(i)?;
        }
        Ok(())
    }

    /// A cycle in which the thread issues nothing.
    pub fn bubble(&mut self) -> Result<(), SchedError> {
        self.step(None)
    }

    /// One regular instruction, one cycle.
    pub fn issue(&mut self, ins: Instruction) -> Result<(), SchedError> {
        self.step(Some(ins))
    }

    // ---- the cycle ---------------------------------------------------------

    fn step(&mut self, regular: Option<Instruction>) -> Result<(), SchedError> {
        let cycle = self.cycle + 1;
        self.cycle = cycle;

        // 1. What executes now: delays that have run out, one per sub-unit.
        let mut now: Vec<Pending> = Vec::new();
        let mut keep: Vec<Pending> = Vec::new();
        for p in self.pending.drain(..) {
            if p.remaining == 0 {
                if now.iter().any(|q| q.unit == p.unit) {
                    return Err(SchedError::Collision {
                        unit: p.unit,
                        cycle,
                    });
                }
                now.push(p);
            } else {
                keep.push(p);
            }
        }
        self.pending = keep;
        now.sort_by_key(|p| p.unit.index());

        // 2. The regular instruction against the scheduled ones.
        let mnemonic = regular.map(|i| i.def().mnemonic());
        if let Some(i) = regular {
            let m = i.def().mnemonic();
            if let Some(u) = regular_unit(m) {
                if now.iter().any(|p| p.unit == u) {
                    return Err(SchedError::RegularCollision { unit: u, cycle });
                }
                // The Simple/Round split applies to a regular instruction too: its
                // `VD` is never 16, so a scheduled partner has to be.
                if matches!(u, Unit::Simple | Unit::Round) {
                    let other = if u == Unit::Simple {
                        Unit::Round
                    } else {
                        Unit::Simple
                    };
                    if let Some(p) = now.iter().find(|p| p.unit == other) {
                        if !Self::writes16(&p.action) {
                            return Err(SchedError::SimpleRoundDestination { cycle });
                        }
                    }
                }
            }
            if m != "SFPNOP" && self.owned_cycle == Some(cycle) {
                return Err(SchedError::SwapNextCycle { cycle });
            }
            if i.def().field("VD").is_some()
                && !matches!(m, "SFPCONFIG" | "SFPLOADMACRO")
                && field(i, "VD").unwrap_or(0) >= 12
            {
                return Err(SchedError::Backdoor { word: i.word() });
            }
        }

        // 3. Simple/Round destinations; the `SFPSWAP` conditions; ownership.
        let simple = now.iter().find(|p| p.unit == Unit::Simple);
        let round = now.iter().find(|p| p.unit == Unit::Round);
        if let (Some(s), Some(r)) = (simple, round) {
            if Self::writes16(&s.action) == Self::writes16(&r.action) {
                return Err(SchedError::SimpleRoundDestination { cycle });
            }
        }
        if self.owned_cycle == Some(cycle) {
            for p in &now {
                if matches!(p.unit, Unit::Simple | Unit::Round) && !matches!(p.action, Action::Nop)
                {
                    return Err(SchedError::SwapNextCycle { cycle });
                }
            }
        }
        let mut owns_next = false;
        for p in &now {
            if let Action::Compute { word, .. } = p.action {
                let d = decode(word)?;
                let m = d.mnemonic();
                if m == "SFPSWAP" {
                    if !now
                        .iter()
                        .any(|q| q.unit == Unit::Mad && matches!(q.action, Action::Nop))
                    {
                        return Err(SchedError::SwapNeedsMadNop { cycle });
                    }
                    owns_next = true;
                }
                // `SFPSHFT2.md`: the three sub-vector modes need an idle cycle after.
                if m == "SFPSHFT2"
                    && matches!(Instruction::new(word, d).operand("Mod1"), Some(2..=4))
                {
                    owns_next = true;
                }
            }
        }

        // 4. Hazards on registers, then execution.
        let mut reads: Vec<(u32, usize)> = Vec::new();
        let mut writes: Vec<(u32, usize)> = Vec::new();
        for (k, p) in now.iter().enumerate() {
            let (r, w) = self.access(&p.action)?;
            reads.extend(r.into_iter().map(|x| (x, k)));
            writes.extend(w.into_iter().map(|x| (x, k)));
        }
        let regular_k = now.len();
        if let Some(i) = regular {
            let (r, w) = Self::regular_access(i);
            reads.extend(r.into_iter().map(|x| (x, regular_k)));
            writes.extend(w.into_iter().map(|x| (x, regular_k)));
        }
        for &(reg, k) in &reads {
            let consumer_scheduled = k != regular_k;
            if self
                .mad_busy
                .iter()
                .any(|&(b, by_sched)| b == reg && (consumer_scheduled || by_sched))
            {
                return Err(SchedError::MadHazard { reg, cycle });
            }
        }
        for &(reg, k) in &writes {
            if reads.iter().any(|&(r, j)| r == reg && j != k)
                || writes.iter().any(|&(w, j)| w == reg && j != k)
            {
                return Err(SchedError::SameCycle { reg, cycle });
            }
        }

        let mut new_busy: Vec<(u32, bool)> = Vec::new();
        for p in &now {
            let what = self.execute(&p.action)?;
            self.events.push(Event {
                cycle,
                unit: p.unit,
                what,
            });
            if p.unit == Unit::Mad {
                let (_, w) = self.access(&p.action)?;
                new_busy.extend(w.into_iter().map(|r| (r, true)));
            }
        }
        if let Some(i) = regular {
            self.execute_regular(i)?;
            if regular_unit(i.def().mnemonic()) == Some(Unit::Mad) {
                let (_, w) = Self::regular_access(i);
                new_busy.extend(w.into_iter().map(|r| (r, false)));
            }
        }
        self.mad_busy = new_busy;
        self.owned_cycle = if owns_next { Some(cycle + 1) } else { None };

        // 5. Count down: per instruction if any outstanding delay says so.
        let by_instruction = self.pending.iter().any(|p| p.by_instructions);
        if !by_instruction || mnemonic.is_some() {
            for p in &mut self.pending {
                p.remaining -= 1;
            }
        }

        // 6. A macro issued now schedules.
        if let Some(i) = regular {
            if i.def().mnemonic() == "SFPLOADMACRO" {
                self.schedule(i)?;
            }
        }
        Ok(())
    }

    fn writes16(a: &Action) -> bool {
        matches!(a, Action::Compute { vd16: true, .. })
    }

    // ---- scheduling (the page's per-sub-unit loop) -------------------------

    fn schedule(&mut self, mload: Instruction) -> Result<(), SchedError> {
        let op = |n: &str| field(mload, n).unwrap_or(0);
        let (macro_index, mod0) = (op("MacroIndex"), op("Mod0"));
        let vd = op("VDHi") << 2 | op("VDLo");
        let imm10 = op("Imm9") << 1 | op("VDHi");
        let sequence = self.sequence[macro_index as usize].ok_or(SchedError::Unconfigured {
            what: "Sequence",
            index: macro_index,
        })?;
        // `Misc` is consulted per scheduled slot (delay kind, `StoreMod0`); an idle
        // sequence never reads it, so an idle macro needs only its `Sequence`.
        let misc_state = self.misc;
        // The address the load resolved, which the scheduled store reuses.
        let addr = (imm10 + self.vector.rwc_dst) & 0x3ff;

        for unit in Unit::ORDER {
            let i = unit.index();
            let bits = (sequence >> (8 * i)) as u8;
            let delay = (bits >> 3) & 7;
            if delay != 7 {
                self.pending
                    .retain(|p| !(p.unit == unit && p.remaining == delay));
            }
            let source = bits & 7;
            let template_word = |t: usize| {
                self.template[t].ok_or(SchedError::Unconfigured {
                    what: "InstructionTemplate",
                    index: t as u32,
                })
            };
            // The instruction this slot starts from, as (mnemonic, word).
            let (mnemonic, word): (&'static str, Option<u32>) = match source {
                0 => continue,
                1 => return Err(SchedError::UndefinedSource { macro_index, unit }),
                2 => ("SFPNOP", None),
                3 => ("SFPSTORE", None),
                t => {
                    let w = template_word(t as usize - 4)?;
                    (decode(w)?.mnemonic(), Some(w))
                }
            };
            if !unit.can_execute(mnemonic) {
                if unit == Unit::Store {
                    return Err(SchedError::StoreNop);
                }
                return Err(SchedError::ClassMismatch { unit, mnemonic });
            }
            let misc = misc_state.ok_or(SchedError::Unconfigured {
                what: "Misc",
                index: 0,
            })?;
            let (vd16, vb_sub) = (bits & 0x40 != 0, bits & 0x80 != 0);
            let action = if unit == Unit::Store {
                let src = if vd16 {
                    StoreSource::Lreg16
                } else if vb_sub {
                    // The template's own `VD` (code 3 is `SFPSTORE` with `VD = 0`).
                    StoreSource::Reg(word.map_or(0, |w| (w >> 4) & 0xf))
                } else {
                    StoreSource::Reg(vd)
                };
                let store_mod0 = misc & 0xf;
                let uses_load_mod0 = (misc >> 4) >> macro_index & 1 != 0;
                Action::Store {
                    src,
                    mod0: if uses_load_mod0 { mod0 } else { store_mod0 },
                    addr,
                }
            } else if let Some(word) = word {
                Action::Compute {
                    word,
                    macro_vd: vd,
                    vd16,
                    vb_sub,
                }
            } else {
                Action::Nop
            };
            self.pending.push(Pending {
                unit,
                action,
                remaining: delay,
                by_instructions: (misc >> 8) >> i & 1 != 0,
            });
        }
        Ok(())
    }

    // ---- what an action reads and writes -----------------------------------

    /// The registers a resolved action reads and writes (`16` is `LReg[16]`).
    fn access(&self, a: &Action) -> Result<(Vec<u32>, Vec<u32>), SchedError> {
        Ok(match a {
            Action::Nop => (vec![], vec![]),
            Action::Store { src, .. } => (
                vec![match src {
                    StoreSource::Reg(r) => *r,
                    StoreSource::Lreg16 => 16,
                }],
                vec![],
            ),
            Action::Compute { .. } => {
                let r = Self::resolve(a)?;
                (r.reads, r.writes)
            }
        })
    }

    /// Reads and writes of a regular instruction: every register field it names,
    /// except a `VD` it only writes. Conservative where the page does not say.
    fn regular_access(i: Instruction) -> (Vec<u32>, Vec<u32>) {
        let m = i.def().mnemonic();
        let mut reads = Vec::new();
        for n in ["VA", "VB", "VC"] {
            if let Some(v) = field(i, n) {
                reads.push(v);
            }
        }
        const WRITE_ONLY_VD: &[&str] = &[
            "SFPLOAD",
            "SFPLOADI",
            "SFPMOV",
            "SFPMAD",
            "SFPADD",
            "SFPMUL",
            "SFPABS",
            "SFPNOT",
            "SFPLOADMACRO",
            "SFP_STOCH_RND",
            "SFPSETSGN",
        ];
        let vd = field(i, "VD");
        if m == "SFPCONFIG" {
            // Reads `LReg[0]`; its `VD` selects a configuration register.
            return (vec![0], vec![]);
        }
        if m == "SFPLOADMACRO" {
            return (vec![], vec![]);
        }
        let mut writes = Vec::new();
        if let Some(v) = vd {
            if m == "SFPSTORE" || !WRITE_ONLY_VD.contains(&m) {
                reads.push(v);
            }
            if m != "SFPSTORE" {
                writes.push(v);
            }
        }
        if m == "SFPSWAP" {
            if let Some(v) = field(i, "VC") {
                writes.push(v);
            }
        }
        (reads, writes)
    }

    // ---- the page's operand overrides --------------------------------------

    fn resolve(a: &Action) -> Result<Resolved, SchedError> {
        let Action::Compute {
            word,
            macro_vd,
            vd16,
            vb_sub,
        } = *a
        else {
            unreachable!("only compute actions resolve")
        };
        let def = decode(word)?;
        let i = Instruction::new(word, def);
        let m = def.mnemonic();
        let class = match m {
            "SFPMAD" | "SFPADD" | "SFPMUL" => Class::Explicit,
            "SFPMOV" | "SFPABS" | "SFPNOT" | "SFP_STOCH_RND" => Class::Explicit,
            "SFPMULI" | "SFPADDI" => Class::ImmediateVc,
            "SFPSWAP" => Class::Swap,
            other => {
                return Err(SchedError::Unmodelled {
                    what: format!("a {other} template"),
                })
            }
        };
        let has = |n: &str| def.field(n).is_some();
        let tpl = |n: &str| field(i, n).unwrap_or(0);
        // "if (SequenceBits & 0x80) { VB = VD; if VC is None, VC = Insn.VD }
        //  else { VC = VD; if VB is None, VB = Insn.VD }"
        let (vb, vc) = if vb_sub {
            (macro_vd, if has("VC") { tpl("VC") } else { tpl("VD") })
        } else {
            (if has("VB") { tpl("VB") } else { tpl("VD") }, macro_vd)
        };
        let vd = if vd16 { 16 } else { macro_vd };
        let mut reads = Vec::new();
        let mut writes = vec![vd];
        match class {
            Class::Explicit => {
                if has("VA") {
                    reads.push(tpl("VA"));
                }
                if has("VB") {
                    reads.push(vb);
                }
                if has("VC") {
                    reads.push(vc);
                }
            }
            Class::ImmediateVc => reads.push(vc),
            Class::Swap => {
                if vd == 16 {
                    return Err(SchedError::Unmodelled {
                        what: "SFPSWAP writing LReg[16]".into(),
                    });
                }
                reads.extend([vc, vd]);
                writes.push(vc);
            }
        }
        Ok(Resolved {
            ins: i,
            class,
            vb,
            vc,
            vd,
            reads,
            writes,
        })
    }

    // ---- execution ---------------------------------------------------------

    fn execute(&mut self, a: &Action) -> Result<&'static str, SchedError> {
        match a {
            Action::Nop => Ok("SFPNOP"),
            Action::Store { src, mod0, addr } => {
                let scratch = 7;
                let saved = self.vector.lreg[scratch];
                let reg = match *src {
                    StoreSource::Reg(r) => r,
                    StoreSource::Lreg16 => {
                        self.vector.lreg[scratch] = Some(self.lreg16);
                        scratch as u32
                    }
                };
                // `SFPSTORE`'s own address arithmetic is replaced by the macro's:
                // the address was resolved at the `SFPLOADMACRO`, and the address
                // modifier is not applied (`ApplyPartialAddrMod` is skipped).
                let rwc = self.vector.rwc_dst;
                self.vector.rwc_dst = 0;
                let st =
                    encode::sfpstore(reg, *mod0, 0, *addr).map_err(|e| SchedError::Unmodelled {
                        what: e.to_string(),
                    })?;
                let r = self.vector.run(&[st]);
                self.vector.rwc_dst = rwc;
                self.vector.lreg[scratch] = saved;
                r?;
                Ok("SFPSTORE")
            }
            Action::Compute { .. } => {
                let r = Self::resolve(a)?;
                let m = r.ins.def().mnemonic();
                if m == "SFPMOV" && r.ins.operand("Mod1").unwrap_or(0) & !4 == 2 {
                    return Err(SchedError::Unmodelled {
                        what: "SFPMOV that writes every lane".into(),
                    });
                }
                let mut ins = r.ins;
                if ins.def().field("VB").is_some() {
                    ins = with_field(ins, "VB", r.vb);
                }
                if ins.def().field("VC").is_some() {
                    ins = with_field(ins, "VC", r.vc);
                }
                match r.class {
                    Class::Swap => {
                        ins = with_field(ins, "VD", r.vd);
                        self.vector.run(&[ins])?;
                    }
                    Class::Explicit | Class::ImmediateVc => {
                        // Run into a scratch register, then land the result where
                        // the macro says (possibly `LReg[16]`), lane by lane under
                        // the lane enables in force now.
                        let scratch = (0..8u32).find(|s| !r.reads.contains(s)).unwrap_or(7);
                        let saved = self.vector.lreg[scratch as usize];
                        if r.class == Class::ImmediateVc {
                            // `VC` defaults to `VD` in these two instructions.
                            self.vector.lreg[scratch as usize] = self.vector.lreg[r.vc as usize];
                        } else {
                            self.vector.lreg[scratch as usize] = Some([0; 32]);
                        }
                        ins = with_field(ins, "VD", scratch);
                        let run = self.vector.run(&[ins]);
                        let result = self.vector.lreg[scratch as usize].unwrap_or([0; 32]);
                        self.vector.lreg[scratch as usize] = saved;
                        run?;
                        let target = r.vd as usize;
                        let mut dest = if target == 16 {
                            Some(self.lreg16)
                        } else {
                            self.vector.lreg[target]
                        };
                        let d = dest.get_or_insert([0; 32]);
                        for l in 0..32 {
                            if !self.vector.use_flags[l] || self.vector.lane_flags[l] {
                                d[l] = result[l];
                            }
                        }
                        if target == 16 {
                            self.lreg16 = *d;
                        } else {
                            self.vector.lreg[target] = dest;
                        }
                    }
                }
                Ok(m)
            }
        }
    }

    fn execute_regular(&mut self, i: Instruction) -> Result<(), SchedError> {
        let m = i.def().mnemonic();
        match m {
            "REPLAY" => Err(SchedError::Replay),
            "SFPCONFIG" => self.config(i),
            "SFPLOADMACRO" => {
                let op = |n: &str| field(i, n).unwrap_or(0);
                let vd = op("VDHi") << 2 | op("VDLo");
                let imm10 = op("Imm9") << 1 | op("VDHi");
                // `VDHi` is address bit 0. A thread `Dst` offset with bit 0 set
                // carries into the column select when `VD >= 4` made it odd.
                if imm10 & 1 == 1 && self.vector.rwc_dst & 1 == 1 {
                    return Err(SchedError::VdHiCarry {
                        vd,
                        imm10,
                        rwc_dst: self.vector.rwc_dst,
                    });
                }
                // The load half is an ordinary `SFPLOAD`.
                let load = encode::sfpload(vd, op("Mod0"), op("AddrMod"), imm10).map_err(|e| {
                    SchedError::Unmodelled {
                        what: e.to_string(),
                    }
                })?;
                self.vector.run(&[load])?;
                Ok(())
            }
            _ => {
                self.vector.run(&[i])?;
                Ok(())
            }
        }
    }

    /// `SFPCONFIG` for the macro registers (targets 0..=8); the rest is the
    /// interpreter's.
    fn config(&mut self, i: Instruction) -> Result<(), SchedError> {
        let (imm, vd, mod1) = (
            field(i, "Imm16").unwrap_or(0),
            field(i, "VD").unwrap_or(0),
            field(i, "Mod1").unwrap_or(0),
        );
        if vd > 8 {
            self.vector.run(&[i])?;
            return Ok(());
        }
        if !self.pending.is_empty() {
            return Err(SchedError::ConfigWhilePending {
                pending: self.pending.len(),
            });
        }
        if mod1 & 8 != 0 {
            return Err(SchedError::ConfigUnsupported {
                what: "a lane mask".into(),
            });
        }
        if (0..32).any(|l| self.vector.use_flags[l] && !self.vector.lane_flags[l]) {
            return Err(SchedError::ConfigNotUniform);
        }
        // Templates always come from `LReg[0]` ("not enough Imm16 bits to specify a
        // 32-bit instruction"); sequences and `Misc` from `Imm16` when asked to.
        let value = if vd >= 4 && mod1 & 1 != 0 {
            imm
        } else {
            let r = self.vector.lreg[0]
                .ok_or(SchedError::Interp(InterpError::Unknown { at: 0, lreg: 0 }))?;
            // The first eight lanes, broadcast: all of them must agree for one
            // configuration to describe every lane.
            if (0..8).any(|l| r[l] != r[0]) {
                return Err(SchedError::ConfigNotUniform);
            }
            r[0]
        };
        match vd {
            0..=3 => self.template[vd as usize] = Some(value),
            4..=7 => self.sequence[vd as usize - 4] = Some(value),
            _ => {
                let v = value & 0xfff;
                self.misc = Some(match mod1 & 6 {
                    0 => v,
                    op => {
                        let old = self.misc.ok_or(SchedError::Unconfigured {
                            what: "Misc",
                            index: 0,
                        })?;
                        match op {
                            2 => old | v,
                            4 => old & v,
                            _ => old ^ v,
                        }
                    }
                });
            }
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Class {
    /// Every operand is a field of the instruction.
    Explicit,
    /// `SFPMULI`/`SFPADDI`: `VC` defaults to `VD`, so the override supplies it.
    ImmediateVc,
    /// `SFPSWAP`: reads and writes `VC` and `VD`.
    Swap,
}

struct Resolved {
    ins: Instruction,
    class: Class,
    vb: u32,
    vc: u32,
    vd: u32,
    reads: Vec<u32>,
    writes: Vec<u32>,
}
