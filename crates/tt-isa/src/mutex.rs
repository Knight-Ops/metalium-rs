//! Tensix mutexes: `ATGETM` and `ATRELM` (Sync Unit).
//!
//! `BlackholeA0/TensixTile/TensixCoprocessor/{ATGETM,ATRELM,SyncUnit}.md`.
//! These are mutexes of the *Sync Unit*, shared by the three Tensix threads of
//! one tile. They are not words in L1 and no RISC-V core can read or change
//! them: the only way to take or drop one is a Tensix instruction issued by the
//! thread that is to hold it.
//!
//! # Which indices exist
//!
//! Blackhole has four mutexes, at **indices 0, 2, 3 and 4**. The pages say that
//! `Index == 1` and `Index > 4` "cause an infinite wait" at the Wait Gate, for
//! `ATRELM` as well as `ATGETM`, and nothing a thread can issue clears a Wait
//! Gate that is waiting on an impossible condition. [`Mutex`] therefore cannot
//! be built for any other index, so neither encoder can emit the wedge; there
//! is no unchecked constructor.
//!
//! # What blocks
//!
//! [`acquire`] holds the issuing thread's later instructions at the Wait Gate
//! until no *other* thread holds the mutex. If every holder is itself blocked,
//! nothing releases it: a blocking program needs a progress path outside the
//! blocked thread (`tt_kernels::atomics`, which arms a role-side deadline and a
//! release route). [`release`] never blocks for long and changes nothing if the
//! issuing thread does not hold the mutex.
//!
//! # Fairness
//!
//! When a mutex held by thread `i` is released and both other threads are
//! waiting in `ATGETM`, thread `(i + 1) % 3` acquires it (`ATRELM.md`). With one
//! waiter that waiter acquires it. Re-acquiring a mutex the thread already
//! holds completes without effect on the page's model; ttsim instead refuses
//! it (divergence row 76), so [`check_scope`] rejects it too.

use crate::isa::generated::encode;
use crate::isa::Instruction;

/// One of the four Blackhole Tensix mutexes.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Mutex(u8);

impl Mutex {
    /// How many mutexes a Blackhole Tensix tile has (`SyncUnit.md`).
    pub const COUNT: usize = 4;
    /// Every valid mutex, in index order.
    pub const ALL: [Mutex; Mutex::COUNT] = [Mutex(0), Mutex(2), Mutex(3), Mutex(4)];

    /// The mutex at hardware `index`, if Blackhole has one there. Indices 1 and
    /// 5 and above are refused, because the instruction waits forever on them.
    pub const fn new(index: u32) -> Option<Mutex> {
        match index {
            0 | 2 | 3 | 4 => Some(Mutex(index as u8)),
            _ => None,
        }
    }

    /// The instruction field's value.
    pub const fn index(self) -> u32 {
        self.0 as u32
    }

    /// A dense `0..4` number, for tables.
    pub const fn slot(self) -> usize {
        match self.0 {
            0 => 0,
            2 => 1,
            3 => 2,
            _ => 3,
        }
    }
}

/// `ATGETM`: wait until no other thread holds `mutex`, then hold it.
///
/// Blocks the issuing thread. Pair every call with [`release`] on the same
/// thread; see [`check_scope`].
pub const fn acquire(mutex: Mutex) -> Instruction {
    match encode::atgetm(mutex.index()) {
        Ok(i) => i,
        Err(_) => panic!("a mutex index is sixteen bits"),
    }
}

/// `ATRELM`: release `mutex` if the issuing thread holds it; otherwise complete
/// without any effect.
pub const fn release(mutex: Mutex) -> Instruction {
    match encode::atrelm(mutex.index()) {
        Ok(i) => i,
        Err(_) => panic!("a mutex index is sixteen bits"),
    }
}

/// Why a thread's program mishandles its mutexes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ScopeError {
    /// `ATGETM` of a mutex the thread already holds (ttsim refuses it; the page
    /// says it completes, possibly after a short wait).
    AlreadyHeld { mutex: Mutex, at: usize },
    /// `ATRELM` of a mutex the thread does not hold: the page says it changes
    /// nothing, which means the program's acquire/release pairing is wrong.
    NotHeld { mutex: Mutex, at: usize },
    /// The program ends still holding `mutex`. Nothing but this thread can
    /// release it, so every other thread's `ATGETM` of it would wait forever.
    HeldAtEnd { mutex: Mutex },
    /// An `ATGETM`/`ATRELM` word whose index is not a Blackhole mutex.
    BadIndex { index: u32, at: usize },
}

/// Check one thread's program for balanced, non-nested-on-itself mutex use:
/// every `ATGETM` is of a mutex the thread does not hold, every `ATRELM` of one
/// it does, and none is held at the end. Instructions that are neither are
/// ignored.
pub fn check_scope(program: &[Instruction]) -> Result<(), ScopeError> {
    let mut held = [false; Mutex::COUNT];
    for (at, i) in program.iter().enumerate() {
        let name = i.def().mnemonic();
        let get = name == "ATGETM";
        if !get && name != "ATRELM" {
            continue;
        }
        let index = i.operand("Index").unwrap_or(u32::MAX);
        let Some(mutex) = Mutex::new(index) else {
            return Err(ScopeError::BadIndex { index, at });
        };
        let slot = &mut held[mutex.slot()];
        match (get, *slot) {
            (true, true) => return Err(ScopeError::AlreadyHeld { mutex, at }),
            (false, false) => return Err(ScopeError::NotHeld { mutex, at }),
            _ => *slot = get,
        }
    }
    for mutex in Mutex::ALL {
        if held[mutex.slot()] {
            return Err(ScopeError::HeldAtEnd { mutex });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_four_blackhole_indices_exist() {
        for index in 0..70_000u32 {
            assert_eq!(
                Mutex::new(index).is_some(),
                matches!(index, 0 | 2 | 3 | 4),
                "index {index}"
            );
        }
        let all: [u32; 4] = core::array::from_fn(|k| Mutex::ALL[k].index());
        assert_eq!(all, [0, 2, 3, 4]);
        for (k, m) in Mutex::ALL.into_iter().enumerate() {
            assert_eq!(m.slot(), k);
        }
    }

    #[test]
    fn encoders_carry_the_index_in_the_low_sixteen_bits() {
        for m in Mutex::ALL {
            let (g, r) = (acquire(m), release(m));
            assert_eq!(g.word() & 0xffff, m.index());
            assert_eq!(r.word() & 0xffff, m.index());
            assert_eq!(g.word() >> 24, 0xa0);
            assert_eq!(r.word() >> 24, 0xa1);
        }
    }

    #[test]
    fn scope_check_rejects_unbalanced_programs() {
        let (m0, m2) = (Mutex::new(0).unwrap(), Mutex::new(2).unwrap());
        assert_eq!(check_scope(&[]), Ok(()));
        assert_eq!(check_scope(&[acquire(m0), release(m0)]), Ok(()));
        assert_eq!(
            check_scope(&[acquire(m0), acquire(m2), release(m0), release(m2)]),
            Ok(())
        );
        assert_eq!(
            check_scope(&[acquire(m0), acquire(m0)]),
            Err(ScopeError::AlreadyHeld { mutex: m0, at: 1 })
        );
        assert_eq!(
            check_scope(&[release(m0)]),
            Err(ScopeError::NotHeld { mutex: m0, at: 0 })
        );
        assert_eq!(
            check_scope(&[acquire(m2)]),
            Err(ScopeError::HeldAtEnd { mutex: m2 })
        );
        // A raw encoding of the index that waits forever is refused too.
        let bad = encode::atgetm(1).unwrap();
        assert_eq!(
            check_scope(&[bad]),
            Err(ScopeError::BadIndex { index: 1, at: 0 })
        );
        let bad = encode::atrelm(5).unwrap();
        assert_eq!(
            check_scope(&[bad]),
            Err(ScopeError::BadIndex { index: 5, at: 0 })
        );
    }
}
