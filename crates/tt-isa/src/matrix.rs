//! `SrcA`/`SrcB` bank ownership, in the types.
//!
//! Each of `SrcA` and `SrcB` has two banks, and each bank is owned either by the
//! unpackers or by the Matrix Unit (`WormholeB0/.../SrcASrcB.md`). The unpacker
//! and the Matrix Unit each keep their *own* pointer to the bank they are using,
//! and the two pointers only move when an instruction says so: an `UNPACR` with
//! `FlipSrc` hands the unpacker's bank to the Matrix Unit and moves the unpacker
//! on (`UNPACR_Regular.md:468-474`); an `MVMUL` or `SETRWC` with `FlipSrcA`/`B`
//! hands the Matrix Unit's bank back and moves *it* on (`MVMUL.md`, `SETRWC.md`).
//! Nothing else moves them, and the hardware waits -- rather than faults -- on a
//! bank the wrong client owns.
//!
//! So the failure modes are not faults. They are a deadlock (waiting on a bank
//! nobody will hand over) or, worse, the two pointers drifting apart so that the
//! Matrix Unit reads a bank the unpacker is no longer writing. The Phase 6 probes
//! hit exactly that by accident: an `UNPACR` with `FlipSrc`, then a `MOVA2D`
//! (which does not flip), then a second `UNPACR` -- which wrote bank 1 while the
//! move read bank 0 again, and returned the first run's data with no diagnostic
//! (`crates/tt-tests/tests/probe_src.rs`, `a_corrupted_datum_moves_exactly_one_src_element`).
//!
//! # The discipline this enforces
//!
//! **Lockstep.** Every bank the unpacker hands over is handed back by exactly one
//! flipping consumer before the unpacker writes again. Under that discipline the
//! two pointers always agree, and each operand is in one of three states:
//!
//! * [`Empty`] -- the unpacker owns the current bank and nothing is staged in it.
//! * [`Filling`] -- the unpacker has written some rows without handing the bank
//!   over (`UNPACR` with `FlipSrc` clear), and more are to come.
//! * [`Loaded`] -- the Matrix Unit owns the bank; any number of non-flipping
//!   consumers may read it (fidelity phases, several `SrcB` blocks against one
//!   `SrcA`), and a flipping one releases it back to [`Empty`].
//!
//! This is stricter than the hardware -- double-buffering, where the unpacker fills
//! bank 1 while the Matrix Unit reads bank 0, is legal and is Phase 9's reason to
//! exist -- but it is the discipline in which a bank mix-up cannot be written, and
//! relaxing it later means adding states, not removing checks.
//!
//! # `UnpackToDst` and `SrcA`
//!
//! `UnpackToDst` leaves `SrcA[Bank]` as `UnpredictableValue` (`UNPACR_Regular.md:441`),
//! where `Bank` is the unpacker's current bank. It waits for that bank to be the
//! unpackers' first, so it cannot clobber data the Matrix Unit holds -- but it
//! *does* destroy rows staged by a non-flipping `UNPACR` still waiting for the rest
//! of their operand. So [`Banks::unpack_to_dst`] requires `SrcA` to be [`Empty`],
//! which is what makes the two unpack modes mutually exclusive, as the checklist
//! asks.
//!
//! # What is deliberately absent
//!
//! * `SETDVALID`: ttsim says its "interaction … with implied src format is
//!   ill-specified (use UNPACR_NOP instead)", and on Blackhole the implied format
//!   is how `MVMUL` learns what the bank holds. `FlipSrc` on the `UNPACR` itself is
//!   the handover used here.
//! * `CLEARDVALID` with `Reset` set: "unsafe and drops SrcA/B banks"
//!   ([tt-metal#22383](https://github.com/tenstorrent/tt-metal/issues/22383)), as
//!   both ttsim and the hardware bug register say.
//! * `STALLWAIT` on the bank conditions C5-C8: the hardware already waits at the
//!   Wait Gate on the owner (`MVMUL.md`, `UNPACR_Regular.md:401,408`), so an
//!   explicit wait adds nothing to correctness here.
//!
//! # Where the state starts
//!
//! [`Banks::after_reset`] asserts both operands [`Empty`], which is the documented
//! reset state (`AllowedClient = Unpackers`, every bank pointer 0). It is *not*
//! true after a program that ended with an operand [`Loaded`]: the state lives in
//! the coprocessor, not in the program, and outlives it.

use core::marker::PhantomData;

use crate::isa::generated::{defs, encode};
use crate::isa::{EncodeError, Instruction};

/// `UNPACR` bit 0: claimed by no field in the specification, and required.
///
/// ttsim calls it `last` and refuses the instruction without it (`tensix_unpacr:
/// last=0`); `docs/ttsim-divergence.md` row F.
pub const UNPACR_LAST: u32 = 1;

/// The unpacker owns the operand's current bank, and nothing is staged in it.
#[derive(Debug)]
pub struct Empty;
/// Rows have been unpacked without handing the bank over; more are to come.
#[derive(Debug)]
pub struct Filling;
/// The Matrix Unit owns the operand's current bank.
#[derive(Debug)]
pub struct Loaded;

/// The ownership state of `SrcA` (`A`) and `SrcB` (`B`) for one Tensix thread's
/// program. Zero-sized; moved through every instruction that changes it.
#[must_use = "the bank state is the only record of who owns each bank"]
#[derive(Debug)]
pub struct Banks<A, B> {
    _state: PhantomData<(A, B)>,
}

const fn banks<A, B>() -> Banks<A, B> {
    Banks {
        _state: PhantomData,
    }
}

impl Banks<Empty, Empty> {
    /// Both operands empty, as after a reset. See the module note: this is a
    /// claim about the coprocessor, which the caller makes.
    pub const fn after_reset() -> Self {
        banks()
    }
}

fn unpacr(which: u32, flip: bool, base: encode::UnpacrRegular) -> Result<Instruction, EncodeError> {
    let i = base
        .which_unpacker(which)
        .flip_src(u32::from(flip))
        .encode()?;
    Ok(Instruction::new(
        i.word() | UNPACR_LAST,
        &defs::UNPACR_Regular,
    ))
}

fn mvmul(flip_a: bool, flip_b: bool, base: encode::Mvmul) -> Result<Instruction, EncodeError> {
    base.flip_src_a(u32::from(flip_a))
        .flip_src_b(u32::from(flip_b))
        .encode()
}

fn release(flip_a: bool, flip_b: bool) -> Result<Instruction, EncodeError> {
    encode::Setrwc::ZERO
        .flip_src_a(u32::from(flip_a))
        .flip_src_b(u32::from(flip_b))
        .encode()
}

// --- Unpacker 0: `SrcA` --------------------------------------------------------

impl<B> Banks<Empty, B> {
    /// Unpack a whole `SrcA` operand and hand it to the Matrix Unit.
    ///
    /// `base` carries the ADC increments and context fields; `WhichUnpacker`,
    /// `FlipSrc` and bit 0 are set here.
    pub fn unpack_a(
        self,
        base: encode::UnpacrRegular,
    ) -> Result<(Instruction, Banks<Loaded, B>), EncodeError> {
        Ok((unpacr(0, true, base)?, banks()))
    }

    /// Unpack part of a `SrcA` operand, keeping the bank.
    pub fn unpack_a_partial(
        self,
        base: encode::UnpacrRegular,
    ) -> Result<(Instruction, Banks<Filling, B>), EncodeError> {
        Ok((unpacr(0, false, base)?, banks()))
    }

    /// `UnpackToDst` through unpacker 0. Leaves `SrcA`'s current bank
    /// unpredictable, which is harmless only because nothing is staged in it.
    ///
    /// The instruction is the same `UNPACR` as a `SrcA` unpack; what makes it a
    /// `Dst` unpack is configuration (`Unpack_if_sel_cntx`), which this cannot see.
    /// The type records the caller's intent.
    pub fn unpack_to_dst(
        self,
        base: encode::UnpacrRegular,
    ) -> Result<(Instruction, Banks<Empty, B>), EncodeError> {
        Ok((unpacr(0, false, base)?, banks()))
    }
}

impl<B> Banks<Filling, B> {
    /// More of the `SrcA` operand, still keeping the bank.
    pub fn unpack_a_partial(
        self,
        base: encode::UnpacrRegular,
    ) -> Result<(Instruction, Banks<Filling, B>), EncodeError> {
        Ok((unpacr(0, false, base)?, banks()))
    }

    /// The last of the `SrcA` operand, handing the bank to the Matrix Unit.
    pub fn unpack_a(
        self,
        base: encode::UnpacrRegular,
    ) -> Result<(Instruction, Banks<Loaded, B>), EncodeError> {
        Ok((unpacr(0, true, base)?, banks()))
    }
}

// --- Unpacker 1: `SrcB` --------------------------------------------------------

impl<A> Banks<A, Empty> {
    /// Unpack a whole `SrcB` operand and hand it to the Matrix Unit.
    pub fn unpack_b(
        self,
        base: encode::UnpacrRegular,
    ) -> Result<(Instruction, Banks<A, Loaded>), EncodeError> {
        Ok((unpacr(1, true, base)?, banks()))
    }

    /// Unpack part of a `SrcB` operand, keeping the bank.
    pub fn unpack_b_partial(
        self,
        base: encode::UnpacrRegular,
    ) -> Result<(Instruction, Banks<A, Filling>), EncodeError> {
        Ok((unpacr(1, false, base)?, banks()))
    }
}

impl<A> Banks<A, Filling> {
    pub fn unpack_b_partial(
        self,
        base: encode::UnpacrRegular,
    ) -> Result<(Instruction, Banks<A, Filling>), EncodeError> {
        Ok((unpacr(1, false, base)?, banks()))
    }

    pub fn unpack_b(
        self,
        base: encode::UnpacrRegular,
    ) -> Result<(Instruction, Banks<A, Loaded>), EncodeError> {
        Ok((unpacr(1, true, base)?, banks()))
    }
}

// --- Matrix Unit consumers -----------------------------------------------------

impl Banks<Loaded, Loaded> {
    /// `MVMUL`, keeping both operands. `base` carries `DstRow`, `AddrMod` and
    /// `BroadcastSrcBRow`; the flip bits are set here.
    pub fn mvmul(self, base: encode::Mvmul) -> Result<(Instruction, Self), EncodeError> {
        Ok((mvmul(false, false, base)?, banks()))
    }

    /// `MVMUL`, then hand `SrcA` back.
    pub fn mvmul_release_a(
        self,
        base: encode::Mvmul,
    ) -> Result<(Instruction, Banks<Empty, Loaded>), EncodeError> {
        Ok((mvmul(true, false, base)?, banks()))
    }

    /// `MVMUL`, then hand `SrcB` back -- the usual step through `SrcB` blocks
    /// against one `SrcA`.
    pub fn mvmul_release_b(
        self,
        base: encode::Mvmul,
    ) -> Result<(Instruction, Banks<Loaded, Empty>), EncodeError> {
        Ok((mvmul(false, true, base)?, banks()))
    }

    /// `MVMUL`, then hand both back.
    pub fn mvmul_release_both(
        self,
        base: encode::Mvmul,
    ) -> Result<(Instruction, Banks<Empty, Empty>), EncodeError> {
        Ok((mvmul(true, true, base)?, banks()))
    }
}

impl<B> Banks<Loaded, B> {
    /// `MOVA2D`: copy `SrcA` rows into `Dst`. Does not flip, so the operand stays
    /// [`Loaded`] -- the step the accidental bank mix-up was missing.
    pub fn mova2d(self, base: encode::Mova2D) -> Result<(Instruction, Self), EncodeError> {
        Ok((base.encode()?, banks()))
    }

    /// `SETRWC` with only `FlipSrcA`: hand `SrcA` back without consuming it again.
    pub fn release_a(self) -> Result<(Instruction, Banks<Empty, B>), EncodeError> {
        Ok((release(true, false)?, banks()))
    }
}

impl<A> Banks<A, Loaded> {
    /// `MOVB2D`: copy `SrcB` rows into `Dst`. Does not flip.
    pub fn movb2d(self, base: encode::Movb2D) -> Result<(Instruction, Self), EncodeError> {
        Ok((base.encode()?, banks()))
    }

    /// `SETRWC` with only `FlipSrcB`.
    pub fn release_b(self) -> Result<(Instruction, Banks<A, Empty>), EncodeError> {
        Ok((release(false, true)?, banks()))
    }
}

/// The rules, each watched failing by removing it.
///
/// A `SrcA` operand cannot be unpacked twice without being consumed:
///
/// ```compile_fail
/// use tt_isa::isa::generated::encode::UnpacrRegular;
/// use tt_isa::matrix::Banks;
/// let (_, b) = Banks::after_reset().unpack_a(UnpacrRegular::ZERO).unwrap();
/// let _ = b.unpack_a(UnpacrRegular::ZERO);
/// ```
///
/// `MVMUL` needs both operands:
///
/// ```compile_fail
/// use tt_isa::isa::generated::encode::{Mvmul, UnpacrRegular};
/// use tt_isa::matrix::Banks;
/// let (_, b) = Banks::after_reset().unpack_a(UnpacrRegular::ZERO).unwrap();
/// let _ = b.mvmul(Mvmul::ZERO);
/// ```
///
/// A non-flipping move leaves the bank with the Matrix Unit, so the next unpack
/// must wait for a release -- the accidental bank mix-up, now a type error:
///
/// ```compile_fail
/// use tt_isa::isa::generated::encode::{Mova2D, UnpacrRegular};
/// use tt_isa::matrix::Banks;
/// let (_, b) = Banks::after_reset().unpack_a(UnpacrRegular::ZERO).unwrap();
/// let (_, b) = b.mova2d(Mova2D::ZERO.move8_rows(1)).unwrap();
/// let _ = b.unpack_a(UnpacrRegular::ZERO);
/// ```
///
/// `UnpackToDst` cannot run while a `SrcA` operand is half-staged:
///
/// ```compile_fail
/// use tt_isa::isa::generated::encode::UnpacrRegular;
/// use tt_isa::matrix::Banks;
/// let (_, b) = Banks::after_reset().unpack_a_partial(UnpacrRegular::ZERO).unwrap();
/// let _ = b.unpack_to_dst(UnpacrRegular::ZERO);
/// ```
///
/// And the legal sequence the gates use compiles:
///
/// ```
/// use tt_isa::isa::generated::encode::{Mova2D, Mvmul, UnpacrRegular};
/// use tt_isa::matrix::Banks;
/// let (_, b) = Banks::after_reset().unpack_a(UnpacrRegular::ZERO).unwrap();
/// let (_, b) = b.unpack_b(UnpacrRegular::ZERO).unwrap();
/// let (_, b) = b.mvmul(Mvmul::ZERO).unwrap();
/// let (_, b) = b.mvmul_release_both(Mvmul::ZERO).unwrap();
/// let (_, b) = b.unpack_a(UnpacrRegular::ZERO).unwrap();
/// let (_, b) = b.mova2d(Mova2D::ZERO.move8_rows(1)).unwrap();
/// let (_, _b) = b.release_a().unwrap();
/// ```
pub mod rules {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_sets_the_unpacker_the_flip_and_the_last_bit() {
        let (i, b) = Banks::after_reset()
            .unpack_a(encode::UnpacrRegular::ZERO)
            .unwrap();
        let flip = encode::UnpacrRegular::ZERO.flip_src(1).encode().unwrap();
        assert_eq!(i.word(), flip.word() | UNPACR_LAST);
        let (i, _) = b.unpack_b_partial(encode::UnpacrRegular::ZERO).unwrap();
        let one = encode::UnpacrRegular::ZERO
            .which_unpacker(1)
            .encode()
            .unwrap();
        assert_eq!(i.word(), one.word() | UNPACR_LAST);
    }

    #[test]
    fn mvmul_flips_exactly_what_its_name_says() {
        let loaded = || -> Banks<Loaded, Loaded> { banks() };
        let word = |a, b| {
            encode::Mvmul::ZERO
                .flip_src_a(a)
                .flip_src_b(b)
                .encode()
                .unwrap()
                .word()
        };
        assert_eq!(
            loaded().mvmul(encode::Mvmul::ZERO).unwrap().0.word(),
            word(0, 0)
        );
        assert_eq!(
            loaded()
                .mvmul_release_a(encode::Mvmul::ZERO)
                .unwrap()
                .0
                .word(),
            word(1, 0)
        );
        assert_eq!(
            loaded()
                .mvmul_release_b(encode::Mvmul::ZERO)
                .unwrap()
                .0
                .word(),
            word(0, 1)
        );
        assert_eq!(
            loaded()
                .mvmul_release_both(encode::Mvmul::ZERO)
                .unwrap()
                .0
                .word(),
            word(1, 1)
        );
    }

    #[test]
    fn a_caller_cannot_set_the_flip_bits_through_the_base() {
        let base = encode::Mvmul::ZERO.flip_src_a(1).flip_src_b(1);
        let loaded: Banks<Loaded, Loaded> = banks();
        let (i, _) = loaded.mvmul(base).unwrap();
        assert_eq!(i.word(), encode::Mvmul::ZERO.encode().unwrap().word());
        let base = encode::UnpacrRegular::ZERO.flip_src(1).which_unpacker(1);
        let (i, _) = Banks::after_reset().unpack_a_partial(base).unwrap();
        assert_eq!(
            i.word(),
            encode::UnpacrRegular::ZERO.encode().unwrap().word() | UNPACR_LAST
        );
    }
}
