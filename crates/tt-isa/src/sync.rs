//! Tensix semaphores: how one Tensix thread hands work to another.
//!
//! The three threads of a split kernel (`harness::Roles` in the test crate, LLK's
//! unpack / math / pack) share most of the backend. Two hand-offs need no
//! software at all: the unpacker gives a `Src` bank to the Matrix Unit by
//! flipping it, and the Matrix Unit waits for that by itself. The third does:
//! nothing in hardware tells the pack thread that the math thread has finished
//! writing `Dst`, or tells the math thread the packer is done reading it. LLK
//! uses the Sync Unit's eight semaphores for both, and so does this.
//!
//! # The model (`SyncUnit.md`, `SEMINIT.md`, `SEMPOST.md`, `SEMGET.md`, `SEMWAIT.md`)
//!
//! Eight semaphores, each a four-bit `Value` and a four-bit `Max`. `SEMPOST`
//! increments (saturating at 15), `SEMGET` decrements (saturating at 0), `SEMINIT`
//! sets both fields, and `SEMWAIT` latches a wait in the issuing thread's Wait
//! Gate that holds back the instructions its block mask names while any selected
//! semaphore has `Value == 0` (C0) or `Value >= Max` (C1). There is no combined
//! wait-then-decrement, so a consumer is `SEMWAIT` then `SEMGET`, and the `SEMGET`
//! has to be one of the instructions the wait holds.
//!
//! # What is encoded rather than commented
//!
//! * A semaphore is a [`Semaphore`], constructed only for indices `0..8`, so the
//!   producer and the consumer name the same one by construction.
//! * Every wait and every [`post_after`] blocks **B1**, the Sync Unit.
//!   `SEMWAIT.md`: once any `SEMWAIT` is in use, every `STALLWAIT` and `SEMWAIT`
//!   should block B1, or a later `SEMWAIT` can latch while an earlier wait is still
//!   in place and the earlier one is forgotten. It is also what makes a
//!   `SEMPOST`/`SEMGET` after a wait wait too.
//! * A `SEMWAIT` with no condition is `UndefinedBehavior`; the helpers always set
//!   one.

pub mod mutex;

use crate::backend::{self, Before, EncodeError};
use crate::isa::generated::encode;
use crate::isa::Instruction;

/// One of the Sync Unit's eight semaphores.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Semaphore(u8);

impl Semaphore {
    /// How many there are (`SyncUnit.md`: `Semaphores[8]`).
    pub const COUNT: u8 = 8;

    /// Semaphore `index`, if there is one.
    pub const fn new(index: u8) -> Option<Semaphore> {
        if index < Self::COUNT {
            Some(Semaphore(index))
        } else {
            None
        }
    }

    pub const fn index(self) -> u8 {
        self.0
    }

    /// The instruction field's one-hot selection of this semaphore.
    pub const fn mask(self) -> u32 {
        1 << self.0
    }
}

/// The largest `Value` or `Max` a semaphore holds: both are four bits.
pub const MAX_VALUE: u8 = 15;

const fn ok(r: Result<Instruction, crate::isa::EncodeError>) -> Instruction {
    match r {
        Ok(i) => i,
        // Every operand below is in range by construction: a one-hot mask of an
        // index below 8, four-bit values checked by the callers, and block and
        // condition masks built from named bits.
        Err(_) => panic!("semaphore operands are in range by construction"),
    }
}

/// `SEMINIT`: set `sem`'s `Value` to `value` and its `Max` to `max`.
///
/// A four-bit field each; anything larger is refused.
pub const fn init(sem: Semaphore, value: u8, max: u8) -> Result<Instruction, EncodeError> {
    if value > MAX_VALUE {
        return Err(EncodeError::ValueTooLarge {
            value: value as u32,
            max: MAX_VALUE as u32,
        });
    }
    if max > MAX_VALUE {
        return Err(EncodeError::ValueTooLarge {
            value: max as u32,
            max: MAX_VALUE as u32,
        });
    }
    Ok(ok(encode::seminit(max as u32, value as u32, sem.mask())))
}

/// `SEMPOST`: increment `sem`, saturating at 15.
///
/// Issued bare, it can run as soon as the Sync Unit sees it -- before the work it
/// announces has finished. [`post_after`] is almost always what is wanted.
pub const fn post(sem: Semaphore) -> Instruction {
    ok(encode::sempost(sem.mask()))
}

/// `SEMGET`: decrement `sem`, saturating at 0.
pub const fn get(sem: Semaphore) -> Instruction {
    ok(encode::semget(sem.mask()))
}

/// Announce, through `sem`, that this thread's `unit` work has *finished*:
/// a `STALLWAIT` on the unit's busy condition blocking the Sync Unit (B1), so
/// the `SEMPOST` behind it waits (`SEMPOST.md`, "Instruction scheduling").
pub fn post_after(unit: Unit, sem: Semaphore) -> [Instruction; 2] {
    let before = Before::EVERYTHING;
    let wait = match unit {
        Unit::Unpacker0 => backend::wait_for_unpacker0(before),
        Unit::Unpacker1 => backend::wait_for_unpacker1(before),
        Unit::Packer => backend::wait_for_packer(before),
        Unit::Matrix => backend::wait_for_matrix(before),
        Unit::Sfpu => backend::wait_for_sfpu(before),
    };
    [wait.expect("the wait helpers take named bits"), post(sem)]
}

/// The backend units a [`post_after`] can wait for.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Unit {
    Unpacker0,
    Unpacker1,
    Packer,
    Matrix,
    Sfpu,
}

/// Hold back `before` -- and the Sync Unit, B1 -- until `sem` is non-zero (C0).
///
/// The consumer's half of a hand-off: follow it with [`get`] to take what was
/// posted, which the B1 block holds until the wait clears.
pub const fn wait_nonzero(sem: Semaphore, before: Before) -> Instruction {
    ok(encode::semwait(
        before.mask() | backend::block::SYNC,
        sem.mask(),
        COND_ZERO,
    ))
}

/// Hold back `before` and B1 until `sem`'s `Value` is below its `Max` (C1).
pub const fn wait_below_max(sem: Semaphore, before: Before) -> Instruction {
    ok(encode::semwait(
        before.mask() | backend::block::SYNC,
        sem.mask(),
        COND_AT_MAX,
    ))
}

/// `SEMWAIT` C0: keep waiting while any selected semaphore has `Value == 0`.
const COND_ZERO: u32 = 1 << 0;
/// `SEMWAIT` C1: keep waiting while any selected semaphore has `Value >= Max`.
const COND_AT_MAX: u32 = 1 << 1;

/// Wait for `sem` to be non-zero, then take one: the consumer's whole half of a
/// hand-off, with `before` held until it has happened.
pub const fn take(sem: Semaphore, before: Before) -> [Instruction; 2] {
    [wait_nonzero(sem, before), get(sem)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_eight_semaphores_exist() {
        assert!(Semaphore::new(7).is_some());
        assert!(Semaphore::new(8).is_none());
        assert_eq!(Semaphore::new(3).unwrap().mask(), 0b1000);
    }

    #[test]
    fn init_places_value_and_max_and_refuses_five_bits() {
        let s = Semaphore::new(2).unwrap();
        let i = init(s, 1, 4).unwrap();
        assert_eq!(i.def().key(), "SEMINIT");
        assert_eq!(i.operand("NewValue"), Some(1));
        assert_eq!(i.operand("NewMax"), Some(4));
        assert_eq!(i.operand("SemaphoreMask"), Some(0b100));
        assert!(init(s, 16, 0).is_err());
        assert!(init(s, 0, 16).is_err());
    }

    #[test]
    fn every_wait_blocks_the_sync_unit_and_names_a_condition() {
        let s = Semaphore::new(5).unwrap();
        for i in [
            wait_nonzero(s, Before::PACKER),
            wait_below_max(s, Before::MATRIX),
        ] {
            assert_eq!(i.def().key(), "SEMWAIT");
            let block = i.operand("BlockMask").unwrap();
            assert_ne!(block & backend::block::SYNC, 0, "B1 must be blocked");
            assert_ne!(i.operand("ConditionMask").unwrap(), 0, "no condition is UB");
            assert_eq!(i.operand("SemaphoreMask"), Some(1 << 5));
        }
        assert_ne!(
            wait_nonzero(s, Before::PACKER)
                .operand("BlockMask")
                .unwrap()
                & backend::block::PACKER,
            0
        );
    }

    #[test]
    fn post_after_waits_for_the_unit_and_holds_the_post() {
        let s = Semaphore::new(0).unwrap();
        let [wait, post] = post_after(Unit::Matrix, s);
        assert_eq!(wait.def().mnemonic(), "STALLWAIT");
        assert_ne!(wait.operand("BlockMask").unwrap() & backend::block::SYNC, 0);
        assert_eq!(post.def().key(), "SEMPOST");
        assert_eq!(post.operand("SemaphoreMask"), Some(1));
    }
}
