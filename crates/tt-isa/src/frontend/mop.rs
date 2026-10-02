//! The MOP Expander (WH `MOPExpander.md`; the Blackhole page says the unit is
//! "similar, but not identical", with ten-bit template-1 counts).
//!
//! One `MOP` instruction expands, at the front of its thread's frontend, into
//! a loop of instructions held in the thread's `MopCfg`: nine words that only
//! the thread's own RISC-V core can write, through a write-only window at
//! `TENSIX_MOP_CFG_BASE` (`BabyRISCV/README.md:120`). The instruction stream
//! cannot set them, so a kernel passes them in its role's mailbox
//! (`crate::mailbox::MOP_CFG`): the runner, before pushing the program, waits
//! until the expander is idle (`MOPExpanderDoneCheck`, `ManualTTSync.md:40-55`
//! -- reconfiguring under an expansion is undefined) and writes the nine
//! words. One configuration per run.
//!
//! `MOP` and `MOP_CFG` are drawn only on Wormhole pages; their layouts are
//! `CONFIRMED` unchanged on Blackhole, ttsim and silicon, by
//! `step36_mop` (`xtask/src/gen_isa/measured.rs`, `CONFIRMED`).
//!
//! [`expand`] is the page's functional model, the oracle for every program
//! that uses one.

use crate::isa::generated::{defs, encode};
use crate::isa::Instruction;

/// The largest template-1 loop count on Blackhole (ten bits).
pub const COUNT_MAX: u32 = 1023;

/// Why a MOP configuration was refused.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MopError {
    /// An outer count of zero, or a count above [`COUNT_MAX`].
    Count { outer: u32, inner: u32 },
    /// No start op, no inner loop, an end op: the page's hardware-bug case
    /// (`UnsupportedFunctionality`, an iteration count off by one or more).
    BuggyShape,
    /// A loop slot holding `MOP` or `MOP_CFG`: the expander does not expand
    /// its own output.
    Nested,
    /// More than 32 template-0 iterations (`Count1` + 1 against the 32-bit mask).
    Iterations(u32),
}

impl core::fmt::Display for MopError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MopError::Count { outer, inner } => write!(
                f,
                "a MOP loop of {outer} x {inner}: the outer count must be 1..={COUNT_MAX}, \
                 the inner 0..={COUNT_MAX}"
            ),
            MopError::BuggyShape => write!(
                f,
                "a MOP with no start op and no inner loop but an end op hits the expander's \
                 iteration-count bug (MOPExpander.md); give it a start op or an inner loop"
            ),
            MopError::Nested => write!(f, "a MOP loop slot holds a MOP or MOP_CFG"),
            MopError::Iterations(n) => write!(f, "{n} template-0 iterations; the mask covers 32"),
        }
    }
}

/// Template 1: `outer` times, `start`, then `inner` loop ops (alternating
/// `loop_op` and `loop_op1` when there is one, which doubles the count), the
/// last of them replaced by `last_of_last` on the final outer iteration and by
/// `last` on the others, then `end`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Template1 {
    pub outer: u32,
    pub inner: u32,
    pub start: Option<Instruction>,
    pub end: Option<(Instruction, Option<Instruction>)>,
    pub loop_op: Instruction,
    pub loop_op1: Option<Instruction>,
    /// `Loop0Last`: the inner loop's last op on the last outer iteration.
    pub last_of_last: Instruction,
    /// `Loop1Last`: the inner loop's last op on every other outer iteration.
    pub last: Instruction,
}

/// Template 0: for each of `Count1 + 1` iterations, mask bit clear: `a0`
/// (then `a123` if any, then `b` if any); bit set: `skip_a0` (then `skip_b`
/// if `b`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Template0 {
    pub a0: Instruction,
    pub a123: Option<[Instruction; 3]>,
    pub b: Option<Instruction>,
    pub skip_a0: Instruction,
    pub skip_b: Instruction,
}

/// One thread's `MopCfg`, as the expander reads it for one template.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MopConfig {
    Template0(Template0),
    Template1(Template1),
}

fn nop() -> Instruction {
    encode::nop().expect("no operands")
}

fn is_nop(i: Instruction) -> bool {
    core::ptr::eq(i.def(), &defs::NOP)
}

fn nested(i: Instruction) -> bool {
    core::ptr::eq(i.def(), &defs::MOP) || core::ptr::eq(i.def(), &defs::MOP_CFG)
}

impl MopConfig {
    /// Refuse what the expander cannot do as the page describes it.
    pub fn check(&self) -> Result<(), MopError> {
        let slots = self.slots();
        if slots.iter().any(|&i| nested(i)) {
            return Err(MopError::Nested);
        }
        if let MopConfig::Template1(t) = self {
            if t.outer == 0 || t.outer > COUNT_MAX || t.inner > COUNT_MAX {
                return Err(MopError::Count {
                    outer: t.outer,
                    inner: t.inner,
                });
            }
            if t.start.is_none() && t.inner == 0 && t.end.is_some() {
                return Err(MopError::BuggyShape);
            }
        }
        Ok(())
    }

    /// `MopCfg[2..9]`, the instruction words, in order.
    fn slots(&self) -> [Instruction; 7] {
        match *self {
            MopConfig::Template0(t) => {
                let [a1, a2, a3] = t.a123.unwrap_or([nop(); 3]);
                [
                    t.b.unwrap_or_else(nop),
                    t.a0,
                    a1,
                    a2,
                    a3,
                    t.skip_a0,
                    t.skip_b,
                ]
            }
            MopConfig::Template1(t) => {
                let (e0, e1) = t
                    .end
                    .map_or((nop(), nop()), |(e0, e1)| (e0, e1.unwrap_or_else(nop)));
                [
                    t.start.unwrap_or_else(nop),
                    e0,
                    e1,
                    t.loop_op,
                    t.loop_op1.unwrap_or_else(nop),
                    t.last_of_last,
                    t.last,
                ]
            }
        }
    }

    /// All nine `MopCfg` words, as the runner writes them: checked first.
    pub fn config_words(&self) -> Result<[u32; 9], MopError> {
        self.check()?;
        Ok(self.words())
    }

    /// All nine `MopCfg` words.
    pub fn words(&self) -> [u32; 9] {
        let (w0, w1) = match *self {
            MopConfig::Template0(t) => (
                0,
                u32::from(t.b.is_some()) | u32::from(t.a123.is_some()) << 1,
            ),
            MopConfig::Template1(t) => (t.outer, t.inner),
        };
        let mut w = [w0, w1, 0, 0, 0, 0, 0, 0, 0];
        for (k, i) in self.slots().into_iter().enumerate() {
            w[2 + k] = i.word();
        }
        w
    }
}

/// A template-1 `MOP`: its counts come from `MopCfg` (the Blackhole override
/// fields, bits 0..20, are `UnsupportedFunctionality` and left zero).
pub fn mop_template1() -> Instruction {
    encode::mop(1, 0, 0).expect("fields in range")
}

/// A template-0 `MOP` over `iterations` (1..=32) with `mask` -- bit `i` set
/// takes the skip path on iteration `i`: the `MOP_CFG` that sets the mask's
/// high half (per-thread state, so always set), then the `MOP`.
pub fn mop_template0(iterations: u32, mask: u32) -> Result<[Instruction; 2], MopError> {
    if iterations == 0 || iterations > 32 {
        return Err(MopError::Iterations(iterations));
    }
    Ok([
        encode::mop_cfg(mask >> 16).expect("sixteen bits"),
        encode::mop(0, iterations - 1, mask & 0xFFFF).expect("fields in range"),
    ])
}

/// The page's functional model: what the expander emits for `mop` under
/// `cfg`, with `mask` the full 32-bit template-0 mask (`MaskHi` from the last
/// `MOP_CFG`, `MaskLo` from the `MOP`).
pub fn expand(cfg: &MopConfig, mop: Instruction, mask_hi: u32, out: &mut impl Extend<Instruction>) {
    let mut push = |i: Instruction| out.extend(core::iter::once(i));
    match cfg {
        MopConfig::Template0(t) => {
            let count1 = mop.operand("Count1").unwrap_or(0);
            let mut mask = (mask_hi << 16) | mop.operand("MaskLo").unwrap_or(0);
            for _ in 0..=count1 {
                if mask & 1 == 0 {
                    push(t.a0);
                    if let Some(a) = t.a123 {
                        a.into_iter().for_each(&mut push);
                    }
                    if let Some(b) = t.b {
                        push(b);
                    }
                } else {
                    push(t.skip_a0);
                    if t.b.is_some() {
                        push(t.skip_b);
                    }
                }
                mask >>= 1;
            }
        }
        MopConfig::Template1(t) => {
            let alternate = t.loop_op1.filter(|&o| !is_nop(o));
            let inner = if alternate.is_some() {
                t.inner * 2
            } else {
                t.inner
            };
            for j in 0..t.outer {
                if let Some(s) = t.start.filter(|&s| !is_nop(s)) {
                    push(s);
                }
                let mut op = t.loop_op;
                for i in 0..inner {
                    push(if i != inner - 1 {
                        op
                    } else if j != t.outer - 1 {
                        t.last
                    } else {
                        t.last_of_last
                    });
                    if let Some(o1) = alternate {
                        op = if op == t.loop_op { o1 } else { t.loop_op };
                    }
                }
                if let Some((e0, e1)) = t.end.filter(|(e0, _)| !is_nop(*e0)) {
                    push(e0);
                    if let Some(e1) = e1.filter(|&e| !is_nop(e)) {
                        push(e1);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    fn expanded(cfg: &MopConfig, mop: Instruction, hi: u32) -> Vec<u32> {
        let mut v = Vec::new();
        expand(cfg, mop, hi, &mut v);
        v.iter().map(|i| i.operand("Imm16").unwrap_or(99)).collect()
    }

    fn op(n: u32) -> Instruction {
        // Distinct, harmless instructions to tell the slots apart.
        crate::sfpu::loadi(0, 0, n).expect("an immediate")
    }

    #[test]
    fn template1_expands_as_the_page_models_it() {
        let t = Template1 {
            outer: 3,
            inner: 2,
            start: Some(op(1)),
            end: Some((op(2), Some(op(3)))),
            loop_op: op(4),
            loop_op1: None,
            last_of_last: op(5),
            last: op(6),
        };
        let got = expanded(&MopConfig::Template1(t), mop_template1(), 0);
        assert_eq!(got, [1, 4, 6, 2, 3, 1, 4, 6, 2, 3, 1, 4, 5, 2, 3]);
        // Alternating ops double the inner count.
        let alt = Template1 {
            loop_op1: Some(op(7)),
            start: None,
            end: None,
            outer: 1,
            ..t
        };
        let got = expanded(&MopConfig::Template1(alt), mop_template1(), 0);
        assert_eq!(got, [4, 7, 4, 5]);
    }

    #[test]
    fn template0_follows_the_mask() {
        let t = Template0 {
            a0: op(1),
            a123: None,
            b: Some(op(2)),
            skip_a0: op(3),
            skip_b: op(4),
        };
        let [cfg, m] = mop_template0(4, 0b0110).unwrap();
        let hi = cfg.operand("MaskHi").unwrap();
        let got = expanded(&MopConfig::Template0(t), m, hi);
        assert_eq!(got, [1, 2, 3, 4, 3, 4, 1, 2]);
    }

    #[test]
    fn a_config_lays_out_its_words_and_refuses_what_the_expander_cannot_do() {
        let t = Template1 {
            outer: 1023,
            inner: 5,
            start: None,
            end: None,
            loop_op: op(4),
            loop_op1: None,
            last_of_last: op(5),
            last: op(6),
        };
        let w = MopConfig::Template1(t).config_words().unwrap();
        assert_eq!((w[0], w[1]), (1023, 5));
        assert_eq!(w[2], nop().word(), "no start op: a NOP");
        assert_eq!(
            [w[5], w[7], w[8]],
            [op(4).word(), op(5).word(), op(6).word()]
        );
        let bad = |t: Template1| MopConfig::Template1(t).config_words();
        assert_eq!(
            bad(Template1 { outer: 0, ..t }),
            Err(MopError::Count { outer: 0, inner: 5 })
        );
        assert_eq!(
            bad(Template1 { outer: 1024, ..t }),
            Err(MopError::Count {
                outer: 1024,
                inner: 5
            })
        );
        assert_eq!(
            bad(Template1 {
                inner: 0,
                end: Some((op(2), None)),
                ..t
            }),
            Err(MopError::BuggyShape)
        );
        assert_eq!(
            bad(Template1 {
                loop_op: mop_template1(),
                ..t
            }),
            Err(MopError::Nested)
        );
        assert_eq!(mop_template0(33, 0), Err(MopError::Iterations(33)));
        let t0 = Template0 {
            a0: op(1),
            a123: None,
            b: Some(op(2)),
            skip_a0: op(3),
            skip_b: op(4),
        };
        assert_eq!(MopConfig::Template0(t0).words()[1], 1, "HasB");
    }
}
