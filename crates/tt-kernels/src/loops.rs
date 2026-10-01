//! Loops in a role program, lowered to the frontend's expanders.
//!
//! A role's runner pushes every word of its program, at ~2.8 cycles a word
//! (`silicon_perf::role_push_rate`), and a matmul's unpack and math roles push
//! no faster than that: the kernel's pace is its word count. The frontend has
//! two expanders that turn a few pushed words into many instructions
//! (`tt_isa::frontend`): the Replay Expander, which records up to 32
//! instructions and replays them for one `REPLAY` word; and before it the MOP
//! Expander, which turns one `MOP` into a loop of the instructions in the
//! thread's `MopCfg` -- which may themselves be `REPLAY`s, so one `MOP`
//! repeats an arbitrary recorded body.
//!
//! A program states its loops as [`Item::Repeat`]; [`lower`] picks each one's
//! form and says which it picked and why ([`Lowered::loops`]):
//!
//! * a body of at most [`tt_isa::frontend::REPLAY_BUFFER`] instructions, none
//!   of them a `REPLAY` or `MOP`, is recorded once (executing as it records)
//!   and replayed: one word for each further iteration;
//! * the loop that saves the most words beyond that takes the thread's one MOP
//!   configuration ([`crate::runtime::Kernel::mop`]: one per thread per run):
//!   template 0 with `A0` the body's `REPLAY`, the iteration count in the
//!   `MOP` itself, so two words stand for up to 32 iterations (the mask's
//!   width; `Count1` reaches 128, but past 32 the expander would read mask
//!   bits shifted in, which no gate has measured yet);
//! * anything else is unrolled.
//!
//! The replay buffer is per-thread state the planner owns while a loop runs:
//! a program that uses `REPLAY` itself (the SFPU builder's row loops) is
//! lowered without it, every loop unrolled, since the planner cannot know
//! which of the buffer's slots that program relies on.
//!
//! The claim, held by [`frontend_stream`] in the tests and the gates: the
//! instructions the backend receives from the lowered program are exactly the
//! unrolled program's, word for word.

use tt_isa::frontend::mop::{self, MopConfig, Template0};
use tt_isa::frontend::{self, REPLAY_BUFFER};
use tt_isa::isa::generated::defs;
use tt_isa::isa::Instruction;

/// One element of a role program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    I(Instruction),
    /// `body`, `times` times over.
    Repeat {
        times: u32,
        body: Vec<Item>,
    },
}

impl Item {
    /// Every instruction, every iteration written out.
    pub fn unrolled(items: &[Item]) -> Vec<Instruction> {
        let mut out = Vec::new();
        for i in items {
            match i {
                Item::I(x) => out.push(*x),
                Item::Repeat { times, body } => {
                    let b = Item::unrolled(body);
                    for _ in 0..*times {
                        out.extend_from_slice(&b);
                    }
                }
            }
        }
        out
    }
}

/// What a loop became.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoopForm {
    /// Recorded and replayed under the thread's MOP configuration: `mops`
    /// `MOP`s (with their `MOP_CFG`s) for the iterations after the first.
    Mop { times: u32, body: usize, mops: u32 },
    /// Recorded and replayed: one `REPLAY` per further iteration.
    Replayed { times: u32, body: usize },
    /// Written out.
    Unrolled {
        times: u32,
        body: usize,
        why: Unrolled,
    },
}

/// Why a loop was written out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Unrolled {
    /// Its body is longer than the replay buffer.
    TooLong,
    /// Its body holds a loop that was itself written out, or a `REPLAY` or
    /// `MOP` of the program's own.
    Nested,
    /// The program uses `REPLAY` itself, so the buffer is not the planner's.
    BufferInUse,
    /// One iteration or none: nothing to save.
    Once,
}

/// A lowered role program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lowered {
    pub words: Vec<Instruction>,
    /// The thread's MOP configuration for the run, if a loop took it.
    pub mop: Option<MopConfig>,
    /// Every top-level loop's form, in program order.
    pub loops: Vec<LoopForm>,
}

/// The most iterations one template-0 `MOP` runs here: one per mask bit
/// (`tt_isa::frontend::mop::mop_template0`).
const MOP_ITERATIONS: u32 = 32;

fn is_expander_insn(i: &Instruction) -> bool {
    core::ptr::eq(i.def(), &defs::REPLAY)
        || core::ptr::eq(i.def(), &defs::MOP)
        || core::ptr::eq(i.def(), &defs::MOP_CFG)
}

/// A loop's body flattened, if it can be recorded: no loop inside it that is
/// not itself written out, and no expander instruction.
fn flat_body(body: &[Item]) -> Option<Vec<Instruction>> {
    let b = Item::unrolled(body);
    (!b.iter().any(is_expander_insn)).then_some(b)
}

/// Words a loop costs in each form.
fn words_replayed(times: u32, len: usize) -> usize {
    1 + len + (times as usize - 1)
}
fn mop_count(times: u32) -> u32 {
    (times - 1).div_ceil(MOP_ITERATIONS)
}
fn words_mop(times: u32, len: usize) -> usize {
    1 + len + 2 * mop_count(times) as usize
}

/// Lower `items`: see the module documentation.
pub fn lower(items: &[Item]) -> Lowered {
    let own_replay = items.iter().any(|i| match i {
        Item::I(x) => is_expander_insn(x),
        Item::Repeat { body, .. } => Item::unrolled(body).iter().any(is_expander_insn),
    });
    // The loop the MOP configuration saves the most on, if any.
    let mut best: Option<(usize, usize)> = None; // (item index, words saved)
    if !own_replay {
        for (k, it) in items.iter().enumerate() {
            if let Item::Repeat { times, body } = it {
                if let Some(b) = flat_body(body) {
                    if *times > 1 && b.len() <= REPLAY_BUFFER as usize {
                        let saved = words_replayed(*times, b.len())
                            .saturating_sub(words_mop(*times, b.len()));
                        if saved > 0 && best.is_none_or(|(_, s)| saved > s) {
                            best = Some((k, saved));
                        }
                    }
                }
            }
        }
    }
    let mut out = Lowered {
        words: Vec::new(),
        mop: None,
        loops: Vec::new(),
    };
    for (k, it) in items.iter().enumerate() {
        let (times, body) = match it {
            Item::I(x) => {
                out.words.push(*x);
                continue;
            }
            Item::Repeat { times, body } => (*times, body),
        };
        let unrolled = Item::unrolled(body);
        let len = unrolled.len();
        let write_out = |out: &mut Lowered, why| {
            for _ in 0..times {
                out.words.extend_from_slice(&unrolled);
            }
            out.loops.push(LoopForm::Unrolled {
                times,
                body: len,
                why,
            });
        };
        if times <= 1 {
            write_out(&mut out, Unrolled::Once);
            continue;
        }
        if own_replay {
            write_out(&mut out, Unrolled::BufferInUse);
            continue;
        }
        let Some(b) = flat_body(body) else {
            write_out(&mut out, Unrolled::Nested);
            continue;
        };
        if b.len() > REPLAY_BUFFER as usize {
            write_out(&mut out, Unrolled::TooLong);
            continue;
        }
        // The first iteration runs as it is recorded, from slot 0.
        frontend::record(0, &b, true, &mut out.words).expect("checked: fits, no REPLAY");
        let replay = frontend::replay(0, b.len()).expect("checked: fits");
        if best.is_some_and(|(at, _)| at == k) {
            out.mop = Some(MopConfig::Template0(Template0 {
                a0: replay,
                a123: None,
                b: None,
                skip_a0: replay,
                skip_b: replay,
            }));
            let mut left = times - 1;
            let mut mops = 0;
            while left > 0 {
                let n = left.min(MOP_ITERATIONS);
                out.words
                    .extend(mop::mop_template0(n, 0).expect("1..=32 iterations"));
                left -= n;
                mops += 1;
            }
            out.loops.push(LoopForm::Mop {
                times,
                body: b.len(),
                mops,
            });
        } else {
            out.words
                .extend(core::iter::repeat_n(replay, times as usize - 1));
            out.loops.push(LoopForm::Replayed {
                times,
                body: b.len(),
            });
        }
    }
    out
}

/// What the backend receives from `words` under `mop`: the MOP Expander's
/// output (`tt_isa::frontend::mop::expand`) through the Replay Expander's
/// (`REPLAY.md`: `Load` records `Count` instructions from `Index`, executing
/// them if `Exec`; otherwise it expands to the recorded ones). The model the
/// lowering is held to; `None` for a stream the model refuses (a replay of
/// slots never recorded, a `MOP` with no configuration).
pub fn frontend_stream(words: &[Instruction], mop: Option<&MopConfig>) -> Option<Vec<Instruction>> {
    // The MOP Expander first.
    let mut stage1 = Vec::new();
    let mut mask_hi = 0;
    for &w in words {
        if core::ptr::eq(w.def(), &defs::MOP_CFG) {
            mask_hi = w.operand("MaskHi")?;
        } else if core::ptr::eq(w.def(), &defs::MOP) {
            mop::expand(mop?, w, mask_hi, &mut stage1);
        } else {
            stage1.push(w);
        }
    }
    // Then the Replay Expander.
    let mut buffer: [Option<Instruction>; REPLAY_BUFFER as usize] = [None; REPLAY_BUFFER as usize];
    let mut out = Vec::new();
    let mut it = stage1.into_iter();
    while let Some(w) = it.next() {
        if !core::ptr::eq(w.def(), &defs::REPLAY) {
            out.push(w);
            continue;
        }
        let (index, count) = (w.operand("Index")? as usize, w.operand("Count")? as usize);
        if w.operand("Load")? != 0 {
            let exec = w.operand("Exec")? != 0;
            for slot in buffer.iter_mut().skip(index).take(count) {
                let i = it.next()?;
                *slot = Some(i);
                if exec {
                    out.push(i);
                }
            }
        } else {
            for slot in buffer.iter().skip(index).take(count) {
                out.push((*slot)?);
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(n: u32) -> Instruction {
        tt_isa::sfpu::loadi(0, 0, n).unwrap()
    }
    fn body(range: std::ops::Range<u32>) -> Vec<Item> {
        range.map(|n| Item::I(op(n))).collect()
    }

    fn holds(items: &[Item]) -> Lowered {
        let l = lower(items);
        let want = Item::unrolled(items);
        let got = frontend_stream(&l.words, l.mop.as_ref()).expect("the model takes it");
        assert_eq!(got, want, "the backend's stream is the unrolled program's");
        l
    }

    #[test]
    fn the_biggest_saving_takes_the_mop_and_the_rest_replay() {
        let items = vec![
            Item::I(op(1000)),
            Item::Repeat {
                times: 5,
                body: body(0..3),
            },
            Item::Repeat {
                times: 100,
                body: body(10..14),
            },
            Item::I(op(1001)),
        ];
        let l = holds(&items);
        assert_eq!(
            l.loops,
            [
                LoopForm::Replayed { times: 5, body: 3 },
                LoopForm::Mop {
                    times: 100,
                    body: 4,
                    mops: 4
                },
            ]
        );
        // 1 + (1 + 3 + 4) + (1 + 4 + 2 * 4) + 1 words, against 1 + 15 + 400 + 1.
        assert_eq!(l.words.len(), 23);
        assert!(l.mop.is_some());
    }

    #[test]
    fn what_cannot_be_recorded_is_written_out_and_says_why() {
        let long = Item::Repeat {
            times: 3,
            body: body(0..33),
        };
        let once = Item::Repeat {
            times: 1,
            body: body(0..2),
        };
        let l = holds(&[long, once]);
        assert_eq!(
            l.loops,
            [
                LoopForm::Unrolled {
                    times: 3,
                    body: 33,
                    why: Unrolled::TooLong
                },
                LoopForm::Unrolled {
                    times: 1,
                    body: 2,
                    why: Unrolled::Once
                },
            ]
        );
        // A program with a REPLAY of its own keeps the buffer to itself.
        let own = frontend::replay(4, 2).unwrap();
        let mut rec = Vec::new();
        frontend::record(4, &[op(7), op(8)], true, &mut rec).unwrap();
        let mut items: Vec<Item> = rec.into_iter().map(Item::I).collect();
        items.push(Item::I(own));
        items.push(Item::Repeat {
            times: 4,
            body: body(0..2),
        });
        let l = lower(&items);
        assert_eq!(
            l.loops,
            [LoopForm::Unrolled {
                times: 4,
                body: 2,
                why: Unrolled::BufferInUse
            }]
        );
        assert!(l.mop.is_none());
    }

    #[test]
    fn nested_loops_flatten_into_one_recorded_body() {
        let inner = Item::Repeat {
            times: 3,
            body: body(0..2),
        };
        let items = vec![Item::Repeat {
            times: 40,
            body: vec![Item::I(op(50)), inner],
        }];
        let l = holds(&items);
        assert_eq!(
            l.loops,
            [LoopForm::Mop {
                times: 40,
                body: 7,
                mops: 2
            }]
        );
    }

    #[test]
    fn the_model_refuses_what_the_hardware_would_not_run() {
        // A replay of slots nothing recorded, a MOP with no configuration.
        assert_eq!(
            frontend_stream(&[frontend::replay(0, 3).unwrap()], None),
            None
        );
        assert_eq!(frontend_stream(&[mop::mop_template1()], None), None);
    }
}
