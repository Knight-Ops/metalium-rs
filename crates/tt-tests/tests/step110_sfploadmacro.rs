//! `SFPLOADMACRO` (hardware-coverage S9, lane H): the macro configuration helper,
//! a schedule model, and the silicon probes that decide whether the page is true.
//!
//! ttsim does not execute `SFPLOADMACRO` (divergence row 7, watched by
//! `step5_corpus::instructions_ttsim_declines_to_execute`), so a device result
//! can only be judged against a model. Three independent things are checked
//! here, none against itself:
//!
//! * `tt_isa::sfpu::load_macro` (checked helper, writes the configuration words) is
//!   read back by `tt_kernels::sfpu::macro_sched` (a model written from the
//!   page's functional model that re-reads the raw words), so a wrongly placed
//!   bit disagrees rather than agrees;
//! * every macro program is compared, bit for bit, with the *ordinary*
//!   sequence of the same computation (`SFPLOAD`, the operations, `SFPSTORE`)
//!   run by the SFPU interpreter, which is the oracle the rest of the
//!   repository trusts;
//! * mutants of the configuration -- swapped template, swapped delays -- change
//!   the data, so a model (or a device) that ignored either would be caught.
//!
//! # Silicon probes, in order of risk
//!
//! A misused macro can wedge the Vector Unit, so the silicon gates are ordered
//! and numbered `s00`..`s13`; a coordinator runs them in order and stops at the
//! first failure. All are `#[cfg(feature = "silicon")]`; none runs by default.
//!
//! | probe | what | risk class |
//! |---|---|---|
//! | `s00` | `SFPCONFIG` writes of the macro registers, then a macro whose sequence is all idle: must still begin as an `SFPLOAD`. Inside `harness::survives`. | documented (WH `SFPCONFIG.md`; BH page says identical) |
//! | `s01` | control: the same stream with the store slot idle; Dst unchanged | documented |
//! | `s02`..`s05` | one sub-unit each: Store, MAD, Simple, Round | documented |
//! | `s06` | `LReg[16]`: Simple writes it, Store reads it | documented |
//! | `s07` | substituted operand: `VB` (bit 7) and `VC` | documented |
//! | `s08` | MAD -> Simple -> Round -> Store chain, plus the flush `StoreMod0` | documented |
//! | `s09` | swapped-template and swapped-delay mutants | documented |
//! | `s10` | four macros in flight | documented |
//! | `s11` | lane predication of scheduled instructions | documented |
//! | `s12` | forgetting a colliding pending instruction | documented |
//! | `s13` | `SFPSWAP` in the Simple sub-unit | UNVERIFIED (two-cycle unit; the only restriction-heavy form) |
//!
//! Every program writes every macro register it reads, uses instruction-counted
//! delays (so the firmware's push rate cannot matter), drains with `SFPNOP`s,
//! and ends with `MacroConfig::teardown`, because the configuration is
//! persistent state the next program and process inherit.

use tt_isa::isa::generated::encode;
use tt_isa::isa::Instruction;
use tt_isa::sfpu::load_macro::{
    drain, MacroConfig, MacroError, MacroLoad, Misc, Sequence, Slot, Source, SubUnit, Template,
};
use tt_isa::sfpu::{self, mod0_fmt};
use tt_kernels::sfpu::interp::Vector;
use tt_kernels::sfpu::macro_sched::{Machine, SchedError, Unit};
#[cfg(feature = "silicon")]
use tt_tests::harness::Dev;
use tt_tests::harness::{self, Run};

/// `Dst` rows a probe dumps (the firmware's maximum): four row groups.
const ROWS: usize = 16;
/// Row group x column half: eight independent 32-lane vectors per probe.
const SLOTS: usize = 8;
/// `SFPLOAD`/`SFPSTORE` without conversion, so subnormals and NaN payloads survive.
const RAW: u32 = mod0_fmt::INT32;

const SET_A: [u32; SLOTS] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // smallest subnormal
    0x8040_0000, // a negative subnormal
    0x3fc0_0000, // 1.5
    0xbfc0_0000, // -1.5
    0x7f80_0000, // +inf
    0x7fc0_0000, // NaN
];
const SET_B: [u32; SLOTS] = [
    0x3f40_0000, // 0.75
    0xff7f_ffff, // -max
    0x7f7f_ffff, // +max (overflows when tripled)
    0x4080_3000, // 4.0117..., a bf16 rounding case
    0xff80_0000, // -inf
    0x0080_0000, // smallest normal
    0xbf40_0000, // -0.75
    0x4000_0000, // 2.0
];

/// The Dst address (`Imm10`) of slot `s`: row group `s / 2`, column half `s % 2`.
fn addr(s: usize) -> u32 {
    ((s as u32 / 2) * 4) | ((s as u32 % 2) * sfpu::DST_ODD_COLUMNS)
}

fn seed(values: &[u32; SLOTS]) -> Vec<Instruction> {
    let mut p = Vec::new();
    for (s, &v) in values.iter().enumerate() {
        p.extend(sfpu::load_f32(0, v).unwrap());
        p.push(sfpu::store(0, RAW, 0, addr(s)).unwrap());
    }
    p
}

fn loadf(vd: u32, v: f32) -> Vec<Instruction> {
    sfpu::load_f32(vd, v.to_bits()).unwrap().to_vec()
}

fn nops(n: usize) -> Vec<Instruction> {
    vec![sfpu::nop(); n]
}

fn store_at(vd: u32, a: u32) -> Instruction {
    sfpu::store(vd, RAW, 0, a).unwrap()
}

fn tpl(i: Instruction) -> Template {
    Template::from_instruction(i)
}

fn slot(src: Source, delay: u32) -> Slot {
    Slot::new(src, delay).unwrap()
}

const THREE: u32 = 0x4040; // bf16 3.0
const NEG_TWO: u32 = 0xc000; // bf16 -2.0

fn t_abs() -> Template {
    tpl(encode::sfpabs(0, 0, 1).unwrap())
}
fn t_neg() -> Template {
    // `SFPMOV` negating: a Simple-class instruction that is not `SFPABS`.
    tpl(encode::sfpmov(0, 0, 1).unwrap())
}
fn t_copy() -> Template {
    tpl(encode::sfpmov(0, 0, 0).unwrap())
}
fn t_muli() -> Template {
    tpl(encode::sfpmuli(THREE, 0, 0).unwrap())
}
fn t_addi() -> Template {
    tpl(encode::sfpaddi(NEG_TWO, 0, 0).unwrap())
}
fn t_round() -> Template {
    // bf16, round to nearest.
    tpl(encode::Sfpstochrnd::ZERO.mod1(1).encode().unwrap())
}
fn mload(m: u32, vd: u32, a: u32) -> Instruction {
    // `VDHi` is address bit 0: an odd address for `VD >= 4`.
    MacroLoad::new(m, vd, RAW, 0, a | (vd >> 2))
        .unwrap()
        .encode()
}

fn all_instruction_counted() -> Misc {
    Misc::ZERO.count_all_instructions()
}

// ---------------------------------------------------------------------------
// Cases: one macro program and the ordinary sequence it stands for.
// ---------------------------------------------------------------------------

struct Case {
    name: &'static str,
    config: MacroConfig,
    /// Instructions to write the configuration instead of `config.writes()`.
    writes_override: Option<Vec<Instruction>>,
    /// Both programs, after the seed (and, for the macro one, the configuration).
    pre: Vec<Instruction>,
    macro_body: Vec<Instruction>,
    plain_body: Vec<Instruction>,
    /// Both programs, after the macro one has drained.
    post: Vec<Instruction>,
}

impl Case {
    fn macro_program(&self, values: &[u32; SLOTS]) -> Vec<Instruction> {
        let mut p = seed(values);
        match &self.writes_override {
            Some(w) => p.extend(w),
            None => p.extend(self.config.writes().unwrap().as_slice()),
        }
        p.extend(&self.pre);
        p.extend(&self.macro_body);
        p.extend(drain());
        p.extend(&self.post);
        p.extend(MacroConfig::teardown().unwrap().as_slice());
        p.extend(drain());
        p
    }

    fn plain_program(&self, values: &[u32; SLOTS]) -> Vec<Instruction> {
        let mut p = seed(values);
        p.extend(&self.pre);
        p.extend(&self.plain_body);
        p.extend(&self.post);
        p
    }
}

fn per_slot(
    f: impl Fn(u32) -> (Vec<Instruction>, Vec<Instruction>),
) -> (Vec<Instruction>, Vec<Instruction>) {
    let (mut m, mut p) = (Vec::new(), Vec::new());
    for s in 0..SLOTS {
        let (a, b) = f(addr(s));
        m.extend(a);
        m.extend(drain());
        p.extend(b);
    }
    (m, p)
}

fn case(
    name: &'static str,
    config: MacroConfig,
    pre: Vec<Instruction>,
    body: (Vec<Instruction>, Vec<Instruction>),
) -> Case {
    Case {
        name,
        config,
        writes_override: None,
        pre,
        macro_body: body.0,
        plain_body: body.1,
        post: Vec::new(),
    }
}

/// Configure macro 0 only, with the other three idle.
fn config_one(misc: Misc, seq: Sequence, templates: &[(u32, Template)]) -> MacroConfig {
    let mut c = MacroConfig::new(misc).with_sequence(0, seq).unwrap();
    for m in 1..4 {
        c = c.with_sequence(m, Sequence::IDLE).unwrap();
    }
    for &(i, t) in templates {
        c = c.with_template(i, t).unwrap();
    }
    c
}

fn idle() -> Slot {
    Slot::IDLE
}

/// `s00`/`s01`: every slot idle, the macro must still be an `SFPLOAD`.
fn c_idle() -> Case {
    let cfg = config_one(
        all_instruction_counted().store_mod0(RAW).unwrap(),
        Sequence::IDLE,
        &[],
    );
    case(
        "idle",
        cfg,
        vec![],
        per_slot(|a| {
            let mut m = loadf(0, 9.0);
            m.push(mload(0, 0, a));
            m.push(store_at(0, a));
            let mut p = loadf(0, 9.0);
            p.push(sfpu::load(0, RAW, 0, a).unwrap());
            p.push(store_at(0, a));
            (m, p)
        }),
    )
}

/// An idle macro that needs only its `Sequence`: no `Misc`, no templates, no
/// other macro. `zero` selects the all-zero sequence word (source 0, delay 0, the
/// value `teardown` writes, so no staging register is involved) over the idle word
/// with delay 7 (`0x38383838`, which the page exempts from forgetting).
fn c_idle_minimal(zero: bool) -> Case {
    let seq = if zero {
        Sequence::from_bits(0).unwrap()
    } else {
        Sequence::IDLE
    };
    let cfg = MacroConfig::new(Misc::ZERO)
        .without_misc()
        .with_sequence(0, seq)
        .unwrap();
    let mut c = case(
        if zero { "idle_zero" } else { "idle_minimal" },
        cfg,
        vec![],
        per_slot(|a| {
            let mut m = loadf(0, 9.0);
            m.push(mload(0, 0, a));
            m.push(store_at(0, a));
            let mut p = loadf(0, 9.0);
            p.push(sfpu::load(0, RAW, 0, a).unwrap());
            p.push(store_at(0, a));
            (m, p)
        }),
    );
    if zero {
        // Nothing to write: zero is what `teardown` restores.
        c.writes_override = Some(MacroConfig::teardown().unwrap().as_slice().to_vec());
    }
    c
}

/// The Store sub-unit alone: an `SFPLOADI` after the macro changes `LReg[0]`
/// before the scheduled store reads it, so the store is observable in place.
fn c_store(store_delay: u32, active: bool) -> Case {
    let seq = Sequence::new(
        idle(),
        idle(),
        idle(),
        if active {
            slot(Source::Store, store_delay)
        } else {
            idle()
        },
    );
    let cfg = config_one(all_instruction_counted().store_mod0(RAW).unwrap(), seq, &[]);
    case(
        if active { "store" } else { "store_control" },
        cfg,
        vec![],
        per_slot(|a| {
            let mut m = vec![mload(0, 0, a)];
            m.extend(loadf(0, 5.0));
            m.push(sfpu::nop());
            let mut p = vec![sfpu::load(0, RAW, 0, a).unwrap()];
            if active {
                p.extend(loadf(0, 5.0));
                p.push(store_at(0, a));
            }
            (m, p)
        }),
    )
}

fn c_unit(name: &'static str, unit: SubUnit, t: Template, plain: Instruction, gap: usize) -> Case {
    let s = slot(Source::Template(0), 0);
    let seq = match unit {
        SubUnit::Simple => Sequence::new(s, idle(), idle(), idle()),
        SubUnit::Mad => Sequence::new(idle(), s, idle(), idle()),
        SubUnit::Round => Sequence::new(idle(), idle(), s, idle()),
        SubUnit::Store => unreachable!(),
    };
    let cfg = config_one(all_instruction_counted(), seq, &[(0, t)]);
    case(
        name,
        cfg,
        vec![],
        per_slot(|a| {
            let mut m = vec![mload(0, 0, a)];
            m.extend(nops(gap));
            m.push(store_at(0, a));
            let p = vec![
                sfpu::load(0, RAW, 0, a).unwrap(),
                plain,
                sfpu::nop(),
                store_at(0, a),
            ];
            (m, p)
        }),
    )
}

/// Simple writes `LReg[16]` (bit 6), the Store sub-unit reads it (bit 6): the
/// register that nothing but a macro can reach.
fn c_lreg16() -> Case {
    let seq = Sequence::new(
        slot(Source::Template(0), 0).vd16(),
        idle(),
        idle(),
        slot(Source::Store, 2).vd16(),
    );
    let cfg = config_one(
        all_instruction_counted().store_mod0(RAW).unwrap(),
        seq,
        &[(0, t_abs())],
    );
    case(
        "lreg16",
        cfg,
        vec![],
        per_slot(|a| {
            let mut m = vec![mload(0, 0, a)];
            m.extend(nops(3));
            let p = vec![
                sfpu::load(0, RAW, 0, a).unwrap(),
                encode::sfpabs(0, 0, 1).unwrap(),
                store_at(0, a),
            ];
            (m, p)
        }),
    )
}

/// Store with bit 7 keeps the template's (here code 3's, `VD = 0`) register as the
/// source rather than the macro's `VD`: the macro works on `LReg[2]`, and the store
/// writes `LReg[0]`, which the program preloaded with 7.0.
fn c_store_keeps_vd() -> Case {
    let seq = Sequence::new(
        slot(Source::Template(0), 0),
        idle(),
        idle(),
        slot(Source::Store, 1).substitute_vb(),
    );
    let cfg = config_one(
        all_instruction_counted().store_mod0(RAW).unwrap(),
        seq,
        &[(0, t_abs())],
    );
    case(
        "store_keeps_vd",
        cfg,
        loadf(0, 7.0),
        per_slot(|a| {
            (
                vec![mload(0, 2, a), sfpu::nop(), sfpu::nop()],
                vec![sfpu::load(2, RAW, 0, a).unwrap(), store_at(0, a)],
            )
        }),
    )
}

/// A substituted operand: with bit 7 the macro's `VD` replaces `VB` and `VC`
/// keeps the template's; without it, `VC` is replaced and `VB` keeps its own.
fn c_substitute(vb: bool) -> Case {
    let mad = tpl(sfpu::mad(3, if vb { 0 } else { 1 }, 9, 0, 0).unwrap());
    let mut s = slot(Source::Template(0), 0);
    if vb {
        s = s.substitute_vb();
    }
    let cfg = config_one(
        all_instruction_counted(),
        Sequence::new(idle(), s, idle(), idle()),
        &[(0, mad)],
    );
    let mut pre = loadf(3, 2.0);
    pre.extend(loadf(1, 4.0));
    case(
        if vb { "substitute_vb" } else { "substitute_vc" },
        cfg,
        pre,
        per_slot(|a| {
            let mut m = vec![mload(0, 0, a)];
            m.extend(nops(3));
            m.push(store_at(0, a));
            // VB-substituted: 2.0 * x + 0.  VC-substituted: 2.0 * 4.0 + x.
            let op = if vb {
                sfpu::mad(3, 0, 9, 0, 0).unwrap()
            } else {
                sfpu::mad(3, 1, 0, 0, 0).unwrap()
            };
            let p = vec![
                sfpu::load(0, RAW, 0, a).unwrap(),
                op,
                sfpu::nop(),
                store_at(0, a),
            ];
            (m, p)
        }),
    )
}

/// The chain `|(x - 2)|` rounded to bf16, MAD -> Simple -> Round -> Store, each
/// stage waiting for the one before. `delays` are the sub-units' (Simple, MAD,
/// Round, Store); `simple` is the Simple template index, which a mutant swaps.
fn chain_config(
    delays: [u32; 4],
    simple: u8,
    store_mod0: u32,
    uses_load_mod0: bool,
) -> MacroConfig {
    let mut misc = all_instruction_counted().store_mod0(store_mod0).unwrap();
    if uses_load_mod0 {
        for m in 0..4 {
            misc = misc.store_uses_load_mod0(m).unwrap();
        }
    }
    let seq = Sequence::new(
        slot(Source::Template(simple), delays[0]),
        slot(Source::Template(1), delays[1]),
        slot(Source::Template(2), delays[2]),
        slot(Source::Store, delays[3]),
    );
    config_one(
        misc,
        seq,
        &[(0, t_abs()), (1, t_addi()), (2, t_round()), (3, t_neg())],
    )
}

fn chain_plain(a: u32, fmt: u32) -> Vec<Instruction> {
    vec![
        sfpu::load(0, RAW, 0, a).unwrap(),
        encode::sfpaddi(NEG_TWO, 0, 0).unwrap(),
        sfpu::nop(),
        encode::sfpabs(0, 0, 1).unwrap(),
        encode::Sfpstochrnd::ZERO.mod1(1).encode().unwrap(),
        sfpu::store(0, fmt, 0, a).unwrap(),
    ]
}

fn c_chain(name: &'static str, cfg: MacroConfig, plain_fmt: u32) -> Case {
    case(
        name,
        cfg,
        vec![],
        per_slot(|a| (vec![mload(0, 0, a)], chain_plain(a, plain_fmt))),
    )
}

/// The real chain: MAD at +1, Simple at +3, Round at +4, Store at +5.
fn c_chain_real() -> Case {
    c_chain("chain", chain_config([2, 0, 3, 4], 0, 0, true), RAW)
}

/// Swapped template: the Simple slot names `SFPMOV`-negate (template 3) where the
/// real chain has `SFPABS` (template 0). Same classes, different data.
fn c_chain_mutant_template() -> Case {
    let mut c = c_chain(
        "chain_mutant_template",
        chain_config([2, 0, 3, 4], 3, 0, true),
        RAW,
    );
    // The mutant's ordinary sequence negates instead of taking the absolute value.
    c.plain_body = per_slot(|a| {
        let mut p = chain_plain(a, RAW);
        p[3] = encode::sfpmov(0, 0, 1).unwrap();
        (vec![], p)
    })
    .1;
    c
}

/// Swapped delays: Simple (abs) first, then the MAD (subtract two): `|x| - 2`.
fn c_chain_mutant_delay() -> Case {
    let mut c = c_chain(
        "chain_mutant_delay",
        chain_config([0, 2, 4, 5], 0, 0, true),
        RAW,
    );
    c.plain_body = per_slot(|a| {
        (
            vec![],
            vec![
                sfpu::load(0, RAW, 0, a).unwrap(),
                encode::sfpabs(0, 0, 1).unwrap(),
                encode::sfpaddi(NEG_TWO, 0, 0).unwrap(),
                sfpu::nop(),
                encode::Sfpstochrnd::ZERO.mod1(1).encode().unwrap(),
                store_at(0, a),
            ],
        )
    })
    .1;
    c
}

/// `Misc.StoreMod0` (not the load's `Mod0`) selects the store format: here FP32,
/// which flushes subnormals, while the load is raw.
fn c_flush() -> Case {
    let seq = Sequence::new(
        slot(Source::Template(0), 0),
        idle(),
        idle(),
        slot(Source::Store, 1),
    );
    let cfg = config_one(
        all_instruction_counted()
            .store_mod0(mod0_fmt::FP32)
            .unwrap(),
        seq,
        &[(0, t_copy())],
    );
    case(
        "flush",
        cfg,
        vec![],
        per_slot(|a| {
            (
                vec![mload(0, 0, a)],
                vec![
                    sfpu::load(0, RAW, 0, a).unwrap(),
                    encode::sfpmov(0, 0, 0).unwrap(),
                    sfpu::store(0, mod0_fmt::FP32, 0, a).unwrap(),
                ],
            )
        }),
    )
}

/// Four macros in flight, each on its own register, two cycles apart.
fn c_pipelined() -> Case {
    let seq = chain_config([2, 0, 3, 4], 0, 0, true);
    let mut cfg = seq;
    for m in 1..4 {
        cfg = cfg.with_sequence(m, seq.sequence(0).unwrap()).unwrap();
    }
    let (mut m, mut p) = (Vec::new(), Vec::new());
    for wave in 0..2 {
        for k in 0..4 {
            let s = wave * 4 + k;
            m.push(mload(k as u32, k as u32, addr(s)));
            m.push(sfpu::nop());
            // The ordinary sequence, on the same register.
            p.extend(chain_plain_on(k as u32, addr(s)));
        }
        m.extend(drain());
    }
    case("pipelined", cfg, vec![], (m, p))
}

fn chain_plain_on(r: u32, a: u32) -> Vec<Instruction> {
    vec![
        sfpu::load(r, RAW, 0, a).unwrap(),
        encode::sfpaddi(NEG_TWO, r, 0).unwrap(),
        sfpu::nop(),
        encode::sfpabs(r, r, 1).unwrap(),
        encode::Sfpstochrnd::ZERO
            .mod1(1)
            .vc(r)
            .vd(r)
            .encode()
            .unwrap(),
        store_at(r, a),
    ]
}

/// `VD >= 4`: the address is odd (`VDHi` is its bit 0), and `LReg[5]` is the
/// macro's register.
fn c_vd5() -> Case {
    let cfg = chain_config([2, 0, 3, 4], 0, 0, true);
    case(
        "vd5",
        cfg,
        vec![],
        per_slot(|a| (vec![mload(0, 5, a)], chain_plain_on(5, a))),
    )
}

/// Lane predication: lane 0 is disabled while the scheduled instructions run.
fn c_predicated() -> Case {
    let mut c = c_chain("predicated", chain_config([2, 0, 3, 4], 0, 0, true), RAW);
    // `SFPENCC` enabling flags, then `SFPSETCC` on `LReg[15]` (twice the lane
    // number): flags are `lane != 0`.
    c.pre = vec![
        encode::sfpencc(3, 0, 2 | 8).unwrap(),
        encode::sfpsetcc(0, 15, 0, 2).unwrap(),
    ];
    c.post = vec![encode::sfpencc(0, 0, 2 | 8).unwrap()];
    c
}

/// Forgetting: macro 1 schedules an idle slot with delay 3 on the Store sub-unit,
/// which cancels macro 0's store (delay 4 issued one cycle earlier). With the
/// idle slot's delay 7 nothing is forgotten and the store happens.
fn c_forget(forget: bool) -> Case {
    let a = Sequence::new(idle(), idle(), idle(), slot(Source::Store, 4));
    let b = Sequence::new(
        idle(),
        idle(),
        idle(),
        slot(Source::Idle, if forget { 3 } else { 7 }),
    );
    let cfg = MacroConfig::new(all_instruction_counted().store_mod0(RAW).unwrap())
        .with_sequence(0, a)
        .unwrap()
        .with_sequence(1, b)
        .unwrap()
        .with_sequence(2, Sequence::IDLE)
        .unwrap()
        .with_sequence(3, Sequence::IDLE)
        .unwrap();
    case(
        if forget { "forget" } else { "forget_control" },
        cfg,
        vec![],
        per_slot(|ad| {
            let mut m = vec![mload(0, 0, ad), mload(1, 0, ad)];
            m.extend(loadf(0, 5.0));
            m.push(sfpu::nop());
            let mut p = vec![sfpu::load(0, RAW, 0, ad).unwrap()];
            if !forget {
                p.extend(loadf(0, 5.0));
                p.push(store_at(0, ad));
            }
            (m, p)
        }),
    )
}

/// `SFPSWAP` in the Simple sub-unit: `LReg[0] = min(LReg[0], 4.0)`. The MAD
/// sub-unit gets an `SFPNOP` for the same time, and the next cycle is an `SFPNOP`.
fn c_swap() -> Case {
    let swap = tpl(encode::sfpswap(1, 0, 1).unwrap());
    let seq = Sequence::new(
        slot(Source::Template(0), 0).substitute_vb(),
        slot(Source::Nop, 0),
        idle(),
        idle(),
    );
    let cfg = config_one(all_instruction_counted(), seq, &[(0, swap)]);
    case(
        "swap",
        cfg,
        vec![],
        per_slot(|a| {
            let mut m = loadf(1, 4.0);
            m.push(mload(0, 0, a));
            m.extend(nops(3));
            m.push(store_at(0, a));
            let mut p = loadf(1, 4.0);
            p.push(sfpu::load(0, RAW, 0, a).unwrap());
            p.push(encode::sfpswap(1, 0, 1).unwrap());
            p.push(sfpu::nop());
            p.push(store_at(0, a));
            (m, p)
        }),
    )
}

/// The probes in risk order.
fn probes() -> Vec<Case> {
    vec![
        c_idle_minimal(true),
        c_idle_minimal(false),
        c_idle(),
        c_store(2, false),
        c_store(2, true),
        c_unit(
            "mad",
            SubUnit::Mad,
            t_muli(),
            encode::sfpmuli(THREE, 0, 0).unwrap(),
            3,
        ),
        c_unit(
            "simple",
            SubUnit::Simple,
            t_abs(),
            encode::sfpabs(0, 0, 1).unwrap(),
            1,
        ),
        c_unit(
            "round",
            SubUnit::Round,
            t_round(),
            encode::Sfpstochrnd::ZERO.mod1(1).encode().unwrap(),
            1,
        ),
        c_lreg16(),
        c_store_keeps_vd(),
        c_substitute(true),
        c_substitute(false),
        c_chain_real(),
        c_flush(),
        c_vd5(),
        c_pipelined(),
        c_predicated(),
        c_forget(false),
        c_forget(true),
        c_swap(),
    ]
}

// ---------------------------------------------------------------------------
// Host: the model against the ordinary sequence.
// ---------------------------------------------------------------------------

fn dump(v: &Vector) -> Vec<u32> {
    (0..ROWS).flat_map(|r| v.dst[r]).collect()
}

fn model(program: &[Instruction]) -> Result<Vec<u32>, SchedError> {
    let mut m = Machine::new(Vector::new());
    m.run(program)?;
    assert_eq!(
        m.pending(),
        0,
        "a macro program must leave nothing scheduled"
    );
    Ok(dump(&m.vector))
}

fn ordinary(program: &[Instruction]) -> Vec<u32> {
    let mut v = Vector::new();
    v.run(program).unwrap();
    dump(&v)
}

#[test]
fn every_macro_program_equals_its_ordinary_sequence_bit_for_bit() {
    let mut cases = probes();
    cases.extend([c_chain_mutant_template(), c_chain_mutant_delay()]);
    for c in &cases {
        for values in [&SET_A, &SET_B] {
            let got = model(&c.macro_program(values))
                .unwrap_or_else(|e| panic!("{}: the model refused it: {e:?}", c.name));
            let want = ordinary(&c.plain_program(values));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g,
                    w,
                    "{}: Dst word {i} (row {}, column {}): macro {g:#010x}, ordinary {w:#010x}",
                    c.name,
                    i / 16,
                    i % 16
                );
            }
        }
    }
}

/// The comparison above is vacuous if the macro does nothing: the cases that
/// act must change `Dst` from the seed, and the controls must not.
#[test]
fn the_macro_cases_are_observable_and_the_controls_are_not() {
    for c in probes() {
        let acts = !matches!(
            c.name,
            "store_control" | "forget" | "idle" | "idle_minimal" | "idle_zero"
        );
        let changed = [&SET_A, &SET_B]
            .iter()
            .any(|v| ordinary(&c.plain_program(v)) != ordinary(&seed(v)));
        assert_eq!(changed, acts, "{}", c.name);
    }
}

#[test]
fn swapped_template_and_swapped_delay_change_the_data() {
    let real = model(&c_chain_real().macro_program(&SET_B)).unwrap();
    let t = model(&c_chain_mutant_template().macro_program(&SET_B)).unwrap();
    let d = model(&c_chain_mutant_delay().macro_program(&SET_B)).unwrap();
    assert_ne!(real, t, "a swapped template must change the result");
    assert_ne!(real, d, "swapped delays must change the result");
    assert_ne!(t, d);
    // The mutants are valid macros: the model ran them to completion. What they
    // compute is what the ordinary sequence of their own order computes.
}

#[test]
fn the_helper_and_the_model_read_the_same_configuration() {
    let c = chain_config([2, 0, 3, 4], 0, 0, true);
    let mut m = Machine::new(Vector::new());
    m.run(c.writes().unwrap().as_slice()).unwrap();
    let (words, present) = c.descriptor();
    let (misc, seq, tp) = m.state();
    assert_eq!(misc, Some(words[0]));
    for i in 0..4 {
        assert_eq!(seq[i].is_some(), present >> i & 1 != 0);
        assert_eq!(seq[i].unwrap_or(0), words[1 + i]);
        assert_eq!(tp[i].is_some(), present >> (4 + i) & 1 != 0);
        assert_eq!(tp[i].unwrap_or(0), words[5 + i]);
    }
}

/// Where each sub-unit executes, by the page's delay rule: a delay of `d` runs
/// `d + 1` cycles after the macro.
#[test]
fn delays_execute_one_cycle_after_their_count() {
    let c = c_chain_real();
    let mut m = Machine::new(Vector::new());
    m.run(&seed(&SET_B)).unwrap();
    m.run(c.config.writes().unwrap().as_slice()).unwrap();
    m.issue(mload(0, 0, addr(0))).unwrap();
    let t0 = m.cycle();
    for _ in 0..8 {
        m.issue(sfpu::nop()).unwrap();
    }
    let got: Vec<(u64, Unit, &str)> = m
        .events
        .iter()
        .map(|e| (e.cycle - t0, e.unit, e.what))
        .collect();
    assert_eq!(
        got,
        [
            (1, Unit::Mad, "SFPADDI"),
            (3, Unit::Simple, "SFPABS"),
            (4, Unit::Round, "SFP_STOCH_RND"),
            (5, Unit::Store, "SFPSTORE"),
        ]
    );
}

/// Instruction-counted delays do not advance on a bubble; cycle-counted ones do.
#[test]
fn delay_kinds_differ_on_a_bubble() {
    let seq = Sequence::new(idle(), idle(), idle(), slot(Source::Store, 1));
    for (misc, store_cycle) in [
        (Misc::ZERO, 2u64),
        (Misc::ZERO.count_instructions(SubUnit::Store), 4),
    ] {
        let cfg = config_one(misc.store_mod0(RAW).unwrap(), seq, &[]);
        let mut m = Machine::new(Vector::new());
        m.run(&seed(&SET_A)).unwrap();
        m.run(cfg.writes().unwrap().as_slice()).unwrap();
        m.issue(mload(0, 0, addr(0))).unwrap();
        let t0 = m.cycle();
        // Two bubbles, then instructions.
        m.bubble().unwrap();
        m.bubble().unwrap();
        for _ in 0..4 {
            m.issue(sfpu::nop()).unwrap();
        }
        let when: Vec<u64> = m.events.iter().map(|e| e.cycle - t0).collect();
        assert_eq!(when, [store_cycle], "{misc:?}");
    }
}

/// Everything the page calls undefined or hazardous is refused by name.
#[test]
fn the_model_refuses_what_the_page_forbids() {
    // Raw words, written the way `SFPCONFIG.md` reads them, independent of the helper.
    fn raw(misc: u32, seqs: [Option<u32>; 4], tpls: [Option<u32>; 4]) -> Vec<Instruction> {
        let mut p = Vec::new();
        for (i, t) in tpls.iter().enumerate() {
            if let Some(w) = t {
                p.extend(sfpu::load_f32(0, *w).unwrap());
                p.push(encode::sfpconfig(0, i as u32, 0).unwrap());
            }
        }
        for (i, s) in seqs.iter().enumerate() {
            if let Some(w) = s {
                p.extend(sfpu::load_f32(0, *w).unwrap());
                p.push(encode::sfpconfig(0, 4 + i as u32, 0).unwrap());
            }
        }
        p.push(encode::sfpconfig(misc, 8, 1).unwrap());
        p
    }
    const IDLE_BYTE: u32 = 0x38;
    // Byte `unit` of macro 0's sequence is `b`, the rest idle with delay 7.
    let seq = |unit: usize, b: u32| {
        let mut w = 0x3838_3838u32;
        w &= !(0xff << (8 * unit));
        Some(w | b << (8 * unit))
    };
    let two = |a: (usize, u32), b: (usize, u32)| {
        let mut w = 0x3838_3838u32;
        for (u, byte) in [a, b] {
            w &= !(0xff << (8 * u));
            w |= byte << (8 * u);
        }
        Some(w)
    };
    let none = [None; 4];
    let tw = |w: Instruction| [Some(w.word()), None, None, None];
    let run = |prog: Vec<Instruction>, body: Vec<Instruction>| {
        let mut m = Machine::new(Vector::new());
        m.run(&seed(&SET_A))?;
        m.run(&prog)?;
        m.run(&body)
    };
    let go = |tpls, simple_seq: Option<u32>, extra: Vec<Instruction>| {
        let mut body = vec![mload(0, 0, addr(0))];
        body.extend(extra);
        body.extend(nops(10));
        run(raw(0xf00, [simple_seq, None, None, None], tpls), body)
    };
    let abs = encode::sfpabs(0, 0, 1).unwrap();
    let muli = encode::sfpmuli(THREE, 0, 0).unwrap();
    let mad = sfpu::mad(3, 1, 0, 0, 0).unwrap();
    let swap = encode::sfpswap(1, 0, 1).unwrap();
    let round = encode::Sfpstochrnd::ZERO.mod1(1).encode().unwrap();
    let _ = IDLE_BYTE;

    // Sequence source 1.
    assert!(matches!(
        go(none, seq(0, 0x01), vec![]),
        Err(SchedError::UndefinedSource {
            unit: Unit::Simple,
            ..
        })
    ));
    // A MAD template on the Simple sub-unit.
    assert!(matches!(
        go(tw(muli), seq(0, 0x04), vec![]),
        Err(SchedError::ClassMismatch {
            unit: Unit::Simple,
            mnemonic: "SFPMULI"
        })
    ));
    // `SFPNOP` on the Store sub-unit.
    assert!(matches!(
        go(none, seq(3, 0x02), vec![]),
        Err(SchedError::StoreNop)
    ));
    // Simple and Round on one cycle, neither or both on LReg[16].
    let both = two((0, 0x04 | 1 << 3), (2, 0x05 | 1 << 3));
    assert!(matches!(
        run(
            raw(
                0xf00,
                [both, None, None, None],
                [Some(abs.word()), Some(round.word()), None, None]
            ),
            vec![mload(0, 0, addr(0))]
                .into_iter()
                .chain(nops(10))
                .collect()
        ),
        Err(SchedError::SimpleRoundDestination { .. })
    ));
    // SFPSWAP: no MAD SFPNOP.
    assert!(matches!(
        go(tw(swap), seq(0, 0x04 | 1 << 3 | 0x80), vec![]),
        Err(SchedError::SwapNeedsMadNop { .. })
    ));
    // SFPSWAP with its MAD NOP, but a Round instruction on the next cycle.
    let swap_round =
        Some(0x04 | 1 << 3 | 0x80 | (0x02 | 1 << 3) << 8 | (0x05 | 2 << 3) << 16 | 0x38 << 24);
    assert!(matches!(
        run(
            raw(
                0xf00,
                [swap_round, None, None, None],
                [Some(swap.word()), Some(round.word()), None, None]
            ),
            vec![mload(0, 0, addr(0))]
                .into_iter()
                .chain(nops(10))
                .collect()
        ),
        Err(SchedError::SwapNextCycle { .. })
    ));
    // A SFPMAD result consumed on the next cycle by the Store sub-unit.
    let mad_store = two((1, 0x04), (3, 0x03 | 1 << 3));
    assert!(matches!(
        run(
            raw(0xf00, [mad_store, None, None, None], tw(mad)),
            vec![mload(0, 0, addr(0))]
                .into_iter()
                .chain(nops(10))
                .collect()
        ),
        Err(SchedError::MadHazard { .. })
    ));
    // The same sequence one cycle later is fine.
    let mad_store_ok = two((1, 0x04), (3, 0x03 | 2 << 3));
    assert!(run(
        raw(0xf04, [mad_store_ok, None, None, None], tw(mad)),
        vec![mload(0, 0, addr(0))]
            .into_iter()
            .chain(nops(10))
            .collect()
    )
    .is_ok());
    // A regular instruction on a sub-unit a scheduled one claims.
    assert!(matches!(
        run(
            raw(
                0xf00,
                [two((1, 0x04), (1, 0x04)), None, None, None],
                tw(mad)
            ),
            vec![mload(0, 0, addr(0)), mad]
        ),
        Err(SchedError::RegularCollision {
            unit: Unit::Mad,
            ..
        })
    ));
    // Simple's result read by Store on the same cycle.
    let same = two((0, 0x04), (3, 0x03));
    assert!(matches!(
        run(
            raw(0xf00, [same, None, None, None], tw(abs)),
            vec![mload(0, 0, addr(0))]
                .into_iter()
                .chain(nops(10))
                .collect()
        ),
        Err(SchedError::SameCycle { reg: 0, .. })
    ));
    // `SFPCONFIG` while something is pending.
    let pending = {
        let mut p = raw(0xf00, [seq(3, 0x03 | 6 << 3), None, None, None], none);
        p.push(mload(0, 0, addr(0)));
        p.push(encode::sfpconfig(0xf00, 8, 1).unwrap());
        p
    };
    assert!(matches!(
        run(vec![], pending),
        Err(SchedError::ConfigWhilePending { pending: 1 })
    ));
    // A regular instruction whose VD >= 12 would overwrite a template.
    assert!(matches!(
        run(vec![], vec![encode::sfpabs(0, 12, 1).unwrap()]),
        Err(SchedError::Backdoor { .. })
    ));
    // A macro the stream never configured: silicon retains whatever was there.
    assert!(matches!(
        run(vec![], vec![mload(0, 0, addr(0))]),
        Err(SchedError::Unconfigured { .. })
    ));
    // A template that is not a Vector Unit instruction.
    assert!(matches!(
        go([Some(0x0100_0000), None, None, None], seq(0, 0x04), vec![]),
        Err(SchedError::NotVector { .. })
    ));
    // A Round template with no execution model here is refused by name.
    let shft2 = encode::sfpshft2(0, 0, 0, 2).unwrap();
    assert!(matches!(
        run(
            raw(0xf00, [seq(2, 0x04), None, None, None], tw(shft2)),
            vec![mload(0, 0, addr(0))]
                .into_iter()
                .chain(nops(10))
                .collect()
        ),
        Err(SchedError::Unmodelled { .. })
    ));
    // The load's address: `VD >= 4` makes it odd, and an odd thread offset carries.
    let mut m = Machine::new(Vector::new());
    m.run(&raw(0xf00, [Some(0x3838_3838), None, None, None], none))
        .unwrap();
    m.vector.rwc_dst = 1;
    assert!(matches!(
        m.issue(mload(0, 4, addr(0))),
        Err(SchedError::VdHiCarry { vd: 4, .. })
    ));
    // `SFPCONFIG` that does not reach every lane alike.
    let mut m = Machine::new(Vector::new());
    m.vector.lreg[0] = Some(std::array::from_fn(|l| l as u32));
    assert_eq!(
        m.issue(encode::sfpconfig(0, 4, 0).unwrap()),
        Err(SchedError::ConfigNotUniform)
    );
}

/// What recovery restores: after any configuration, `teardown` leaves `Misc` and
/// every `Sequence` zero and no template needed, and a macro is then exactly an
/// `SFPLOAD` -- the state `step5_corpus::sfploadmacro_begins_as_an_sfpload_from_dst`
/// relies on. The leading `SFPENCC` also clears a predication state left behind.
#[test]
fn teardown_restores_zero_and_a_macro_is_then_a_plain_load() {
    let full = c_chain_real();
    let mut m = Machine::new(Vector::new());
    m.run(&seed(&SET_A)).unwrap();
    // Leave predication half-set, as an earlier program on this unit might have.
    m.vector.use_flags = [true; 32];
    m.vector.lane_flags = std::array::from_fn(|l| l % 2 == 0);
    // The configuration write must be immune to it (its preamble clears it).
    m.run(full.config.writes().unwrap().as_slice()).unwrap();
    m.run(MacroConfig::teardown().unwrap().as_slice()).unwrap();
    assert_eq!(m.state().0, Some(0));
    assert_eq!(m.state().1, [Some(0); 4]);
    // Templates are untouched: an idle sequence never reads them.
    assert!(m.state().2.iter().all(|t| t.is_some()));
    // A zero sequence: the macro is its load half only.
    let mut p = loadf(0, 9.0);
    p.push(mload(0, 0, addr(0)));
    p.push(store_at(0, addr(0)));
    p.extend(drain());
    m.run(&p).unwrap();
    assert_eq!(m.vector.dst[0][0], SET_A[0]);
    assert_eq!(m.vector.dst[0][2], SET_A[0]);
    assert_eq!(m.pending(), 0);
}

/// The helper refuses statically what the model refuses dynamically.
#[test]
fn the_helper_and_the_model_refuse_the_same_classes() {
    let s = Sequence::new(slot(Source::Template(0), 0), idle(), idle(), idle());
    let c = config_one(Misc::ZERO, s, &[(0, t_muli())]);
    assert!(matches!(
        c.validate(),
        Err(MacroError::ClassMismatch { .. })
    ));
    // The model agrees on the same configuration.
    let mut m = Machine::new(Vector::new());
    m.run(c.writes().unwrap().as_slice()).unwrap();
    assert!(matches!(
        m.issue(mload(0, 0, 0)),
        Err(SchedError::ClassMismatch { .. })
    ));
}

// ---------------------------------------------------------------------------
// ttsim: refuses the instruction; the control survives.
// ---------------------------------------------------------------------------

/// ttsim has no `SFPLOADMACRO` (row 7). A *configured* macro is refused as the
/// bare one is; the same stream with the macro replaced by an `SFPNOP` is the
/// control and must run, so the refusal is the instruction and not the
/// configuration writes or the harness.
#[cfg(not(feature = "silicon"))]
#[test]
fn ttsim_refuses_a_configured_macro_and_runs_its_control() {
    let c = c_chain_real();
    let run = |with_macro: bool| {
        let mut p = seed(&SET_A);
        p.extend(c.config.writes().unwrap().as_slice());
        p.push(if with_macro {
            mload(0, 0, addr(0))
        } else {
            sfpu::nop()
        });
        p.extend(drain());
        p.extend(MacroConfig::teardown().unwrap().as_slice());
        harness::survives(move |dev| {
            let _ = harness::run(dev, &Run::new(&p).dump_rows(ROWS as u32));
        })
    };
    assert!(run(false), "the control (configuration, no macro) must run");
    assert!(
        !run(true),
        "ttsim declined SFPLOADMACRO before; if a configured one runs, the simulator \
         gained a model and divergence 7 needs revisiting"
    );
}

// ---------------------------------------------------------------------------
// Silicon (never run by default): the probes, in order.
// ---------------------------------------------------------------------------

#[cfg(feature = "silicon")]
fn on_card(dev: &mut Dev<'_>, program: &[Instruction]) -> Vec<u32> {
    harness::run(dev, &Run::new(program).dump_rows(ROWS as u32)).dst
}

/// Run `case` on the card for both value sets: the ordinary sequence must equal
/// the interpreter (the control for the oracle), and the macro program must equal
/// the model.
#[cfg(feature = "silicon")]
fn probe(c: &Case) {
    harness::assert_on_silicon();
    for (label, values) in [("A", &SET_A), ("B", &SET_B)] {
        let (mp, pp) = (c.macro_program(values), c.plain_program(values));
        let want_macro = model(&mp).unwrap();
        let want_plain = ordinary(&pp);
        harness::in_device(|dev| {
            let plain = on_card(dev, &pp);
            assert_eq!(
                plain, want_plain,
                "{} set {label}: the ordinary sequence disagrees with the interpreter",
                c.name
            );
            let got = on_card(dev, &mp);
            let bad: Vec<usize> = (0..got.len())
                .filter(|&i| got[i] != want_macro[i])
                .collect();
            println!(
                "MEASURE sfploadmacro.{}.{label}.mismatches = {} of {}",
                c.name,
                bad.len(),
                got.len()
            );
            assert!(
                bad.is_empty(),
                "{} set {label}: the card disagrees with the page's model at words {:?}: card {:08x?}, model {:08x?}",
                c.name,
                &bad[..bad.len().min(8)],
                bad.iter().take(8).map(|&i| got[i]).collect::<Vec<_>>(),
                bad.iter().take(8).map(|&i| want_macro[i]).collect::<Vec<_>>(),
            );
        });
    }
}

#[cfg(feature = "silicon")]
fn named(name: &str) -> Case {
    probes().into_iter().find(|c| c.name == name).unwrap()
}

/// Run a configuration-only program (no macro) in its own child; the program ends
/// with a drain and the restoring `teardown`, which is not reached if it hangs.
#[cfg(feature = "silicon")]
fn config_only(name: &str, mut p: Vec<Instruction>) {
    harness::assert_on_silicon();
    p.extend(drain());
    p.extend(MacroConfig::teardown().unwrap().as_slice());
    p.extend(drain());
    // The model must accept it (and ends with every register zero).
    let mut m = Machine::new(Vector::new());
    m.run(&p).unwrap();
    assert_eq!(m.state().0, Some(0));
    assert!(
        harness::survives(|dev| {
            let _ = on_card(dev, &p);
        }),
        "{name}: the configuration writes did not run to completion; the macro \
         registers may now hold unknown state: run `s_restore_default_macro_state`"
    );
}

/// RECOVERY, run first (and again after any macro probe that hangs): write zero to
/// every `Sequence` and to `Misc`, by immediate `SFPCONFIG` -- no `SFPLOADI`, no
/// staging register -- with predication turned off first, then drain.
///
/// The pages state **no reset value** for any macro register, so there is no
/// documented value to restore to. Zero is the sequence of idle sub-units (source
/// 0 on every slot), the one value whose effect the page defines ("Do not schedule
/// anything for this sub-unit"). Templates are not rewritten: they are read only
/// through a non-idle slot. Whether the card's own power-on state is zero is not
/// documented; `step5_corpus::sfploadmacro_begins_as_an_sfpload_from_dst` passed
/// on a card in whatever state it was then, which this does not claim to equal.
#[cfg(feature = "silicon")]
#[test]
fn s_restore_default_macro_state() {
    harness::assert_on_silicon();
    let mut p = MacroConfig::teardown().unwrap().as_slice().to_vec();
    p.extend(drain());
    assert!(
        harness::survives(|dev| {
            let _ = on_card(dev, &p);
        }),
        "even the immediate zero writes of the macro registers did not complete"
    );
}

/// `s00a1`: one staged `SFPCONFIG` (two `SFPLOADI`s into `LReg[0]`, then the write)
/// of the idle sequence word `0x38383838` into `Sequence[0]`. Then restore.
#[cfg(feature = "silicon")]
#[test]
fn s00a1_staged_sequence_write_only() {
    let c = MacroConfig::new(Misc::ZERO)
        .without_misc()
        .with_sequence(0, Sequence::IDLE)
        .unwrap();
    config_only("s00a1", c.writes().unwrap().as_slice().to_vec());
}

/// `s00a2`: `Misc = 0xf04` (all delays instruction-counted, `StoreMod0` 4) by
/// immediate `SFPCONFIG`, nothing else. Then restore.
#[cfg(feature = "silicon")]
#[test]
fn s00a2_misc_write_only() {
    let c = MacroConfig::new(all_instruction_counted().store_mod0(RAW).unwrap());
    config_only("s00a2", c.writes().unwrap().as_slice().to_vec());
}

/// `s00a`: the whole configuration the idle macro of `s00c` uses (`Misc` and four
/// idle sequences), then drain and `teardown`. No `SFPLOADMACRO`.
#[cfg(feature = "silicon")]
#[test]
fn s00a_config_writes_only_then_teardown() {
    let c = named("idle");
    config_only("s00a", c.config.writes().unwrap().as_slice().to_vec());
}

// `s00b` is deliberately absent: an *unconfigured* macro is not defined by the
// pages. `SFPLOADMACRO.md` says only that the `LoadMacroConfig` state "is written
// via `SFPCONFIG`" and gives no reset value for `Misc`, `Sequence` or
// `InstructionTemplate`, so what a macro does before any write is whatever the
// card last held. step5's measurement is of that, not of the page.

/// First macro, isolated: zero sequence (source 0 on every slot, delay 0), which
/// `s_restore_default_macro_state` just wrote, so nothing is written here at all
/// beyond the same immediate zero writes.
#[cfg(feature = "silicon")]
#[test]
fn s00c0_zero_sequence_idle_macro() {
    harness::assert_on_silicon();
    let c = named("idle_zero");
    let p = c.macro_program(&SET_A);
    let want = model(&p).unwrap();
    assert!(
        harness::survives(|dev| {
            assert_eq!(
                on_card(dev, &p),
                want,
                "an idle macro must behave as its SFPLOAD"
            );
        }),
        "the first SFPLOADMACRO (zero sequence, nothing staged) did not complete or disagreed"
    );
}

/// The minimal configured idle macro: only `Sequence[0] = 0x38383838` is written
/// (idle with the forgetting-exempt delay 7); no `Misc`, templates or other macro.
#[cfg(feature = "silicon")]
#[test]
fn s00c1_minimal_idle_macro() {
    harness::assert_on_silicon();
    let c = named("idle_minimal");
    let p = c.macro_program(&SET_A);
    let want = model(&p).unwrap();
    assert!(
        harness::survives(|dev| {
            assert_eq!(
                on_card(dev, &p),
                want,
                "an idle macro must behave as its SFPLOAD"
            );
        }),
        "the minimal configured SFPLOADMACRO did not complete or disagreed"
    );
}

/// The full configuration (`Misc` instruction-counted, four idle sequences) and an
/// idle macro: the program that first hung (now with the preamble and NOPs).
#[cfg(feature = "silicon")]
#[test]
fn s00c_configured_idle_macro() {
    harness::assert_on_silicon();
    let c = named("idle");
    let p = c.macro_program(&SET_A);
    let want = model(&p).unwrap();
    assert!(
        harness::survives(|dev| {
            assert_eq!(
                on_card(dev, &p),
                want,
                "an idle macro must behave as its SFPLOAD"
            );
        }),
        "the fully configured idle SFPLOADMACRO did not complete or disagreed"
    );
}

#[cfg(feature = "silicon")]
#[test]
fn s01_store_control_leaves_dst_alone() {
    probe(&named("store_control"));
}
#[cfg(feature = "silicon")]
#[test]
fn s02_store_sub_unit() {
    probe(&named("store"));
}
#[cfg(feature = "silicon")]
#[test]
fn s03_mad_sub_unit() {
    probe(&named("mad"));
}
#[cfg(feature = "silicon")]
#[test]
fn s04_simple_sub_unit() {
    probe(&named("simple"));
}
#[cfg(feature = "silicon")]
#[test]
fn s05_round_sub_unit() {
    probe(&named("round"));
}
#[cfg(feature = "silicon")]
#[test]
fn s06_lreg16_written_by_simple_read_by_store() {
    probe(&named("lreg16"));
    probe(&named("store_keeps_vd"));
}
#[cfg(feature = "silicon")]
#[test]
fn s07_substituted_operands() {
    probe(&named("substitute_vb"));
    probe(&named("substitute_vc"));
}
#[cfg(feature = "silicon")]
#[test]
fn s08_chain_and_flush_store_mod0() {
    probe(&named("chain"));
    probe(&named("flush"));
    probe(&named("vd5"));
}

/// The data mutants: the card must follow each mutated configuration (the model's
/// answer for it) and the mutants must differ from the real chain, so a card that
/// ignored the template or delay bytes would fail one or the other.
#[cfg(feature = "silicon")]
#[test]
fn s09_swapped_template_and_delay_mutants() {
    probe(&c_chain_mutant_template());
    probe(&c_chain_mutant_delay());
    let real = model(&c_chain_real().macro_program(&SET_B)).unwrap();
    for m in [c_chain_mutant_template(), c_chain_mutant_delay()] {
        let want = model(&m.macro_program(&SET_B)).unwrap();
        assert_ne!(want, real, "{}", m.name);
        harness::in_device(|dev| {
            let got = on_card(dev, &m.macro_program(&SET_B));
            assert_ne!(
                got, real,
                "{}: the card produced the real chain's result for a mutated configuration",
                m.name
            );
        });
    }
}
#[cfg(feature = "silicon")]
#[test]
fn s10_pipelined_macros() {
    probe(&named("pipelined"));
}
#[cfg(feature = "silicon")]
#[test]
fn s11_predicated_lanes() {
    probe(&named("predicated"));
}
#[cfg(feature = "silicon")]
#[test]
fn s12_forgetting() {
    probe(&named("forget_control"));
    probe(&named("forget"));
}
/// Last: the two-cycle `SFPSWAP` with its three restrictions is the form most
/// likely to differ from the page, and the one a wrong guess could wedge.
#[cfg(feature = "silicon")]
#[test]
fn s13_sfpswap_in_the_simple_sub_unit() {
    harness::assert_on_silicon();
    let c = named("swap");
    let (mp, want) = (
        c.macro_program(&SET_B),
        model(&c.macro_program(&SET_B)).unwrap(),
    );
    assert!(
        harness::survives(|dev| {
            assert_eq!(on_card(dev, &mp), want);
        }),
        "SFPSWAP in a macro wedged or disagreed with the page"
    );
    probe(&c);
}
