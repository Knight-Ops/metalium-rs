//! The Tensix frontend's expanders: `REPLAY` (WH `REPLAY.md`).
//!
//! Each Tensix thread has a Replay Expander between the MOP Expander and the
//! Wait Gate, with a 32-instruction buffer of its own. `REPLAY` with `Load`
//! records the next `Count` instructions into the buffer from `Index` --
//! executing them as well if `Exec` -- and without it, expands to `Count`
//! instructions from the buffer. One `REPLAY` word then stands for a whole
//! loop body, which is what keeps a tile's worth of row-group iterations from
//! being thirty-two copies of it in the instruction stream.
//!
//! The buffer is per-thread state, and per-thread state survives between
//! programs (divergence rows 47, 49): a program records what it replays
//! before replaying it, and never assumes what an earlier one left. A body
//! must not itself contain a `REPLAY`: while recording, the expander takes
//! the next `Count` instructions whatever they are.

use crate::isa::generated::{defs, encode};
use crate::isa::Instruction;

pub mod mop;

/// Instructions the replay buffer holds, per thread.
pub const REPLAY_BUFFER: u32 = 32;

/// Why a `REPLAY` was refused.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReplayError {
    /// Nothing to record, or more than the buffer holds from `index` without
    /// wrapping onto itself.
    Length { index: u32, len: usize },
    /// A recorded body containing a `REPLAY`.
    Nested { at: usize },
}

impl core::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReplayError::Length { index, len } => write!(
                f,
                "a replay body of {len} instructions from slot {index} does not fit the \
                 {REPLAY_BUFFER}-entry buffer"
            ),
            ReplayError::Nested { at } => {
                write!(f, "instruction {at} of a replay body is itself a REPLAY")
            }
        }
    }
}

fn check(index: u32, body_len: usize) -> Result<(), ReplayError> {
    if body_len == 0 || index as usize + body_len > REPLAY_BUFFER as usize {
        return Err(ReplayError::Length {
            index,
            len: body_len,
        });
    }
    Ok(())
}

/// Record `body` into the buffer from slot `index`, executing it as it is
/// recorded if `exec`: the `REPLAY` word, followed by the body.
pub fn record(
    index: u32,
    body: &[Instruction],
    exec: bool,
    out: &mut impl Extend<Instruction>,
) -> Result<(), ReplayError> {
    check(index, body.len())?;
    if let Some(at) = body
        .iter()
        .position(|i| core::ptr::eq(i.def(), &defs::REPLAY))
    {
        return Err(ReplayError::Nested { at });
    }
    let r = encode::replay(index, body.len() as u32, u32::from(exec), 1)
        .expect("index and count checked above");
    out.extend(core::iter::once(r));
    out.extend(body.iter().copied());
    Ok(())
}

/// Expand the `len` instructions recorded from slot `index`.
pub fn replay(index: u32, len: usize) -> Result<Instruction, ReplayError> {
    check(index, len)?;
    Ok(encode::replay(index, len as u32, 0, 0).expect("index and count checked above"))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Sink(usize, Option<Instruction>);
    impl Extend<Instruction> for Sink {
        fn extend<I: IntoIterator<Item = Instruction>>(&mut self, it: I) {
            for i in it {
                if self.0 == 0 {
                    self.1 = Some(i);
                }
                self.0 += 1;
            }
        }
    }

    #[test]
    fn record_and_replay_encode_their_fields_and_refuse_what_does_not_fit() {
        let nop = crate::sfpu::nop();
        let body = [nop; 5];
        let mut sink = Sink(0, None);
        record(3, &body, true, &mut sink).unwrap();
        assert_eq!(sink.0, 6, "the REPLAY word and the body");
        let r = sink.1.unwrap();
        assert_eq!(
            (
                r.operand("Index"),
                r.operand("Count"),
                r.operand("Exec"),
                r.operand("Load")
            ),
            (Some(3), Some(5), Some(1), Some(1))
        );
        let p = replay(3, 5).unwrap();
        assert_eq!((p.operand("Count"), p.operand("Load")), (Some(5), Some(0)));
        assert_eq!(replay(0, 0), Err(ReplayError::Length { index: 0, len: 0 }));
        assert_eq!(
            replay(1, 32),
            Err(ReplayError::Length { index: 1, len: 32 })
        );
        assert!(replay(0, 32).is_ok());
        let nested = [nop, p];
        assert_eq!(
            record(0, &nested, false, &mut Sink(0, None)),
            Err(ReplayError::Nested { at: 1 })
        );
    }
}
