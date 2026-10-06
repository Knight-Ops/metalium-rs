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

/// Reset all four packer exponent histograms and their maximum-exponent state.
/// This is unrelated to selecting a block's shared BFP exponent. Drain pack
/// work before issuing it, then wait for the Matrix Unit before reading state.
pub fn clear_exponent_history() -> Instruction {
    encode::clrexphist().expect("CLREXPHIST has no operands")
}

/// `UNPACR` bit 0: claimed by no field in the specification, and required.
///
/// ttsim calls it `last` and refuses the instruction without it (`tensix_unpacr:
/// last=0`); `docs/learnings/ttsim-divergence.md` row F.
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

/// SrcB addressing within an aligned 8×16 elementwise block.
/// ELW consumers require both loaded banks and release ownership in their types.
///
/// ```compile_fail
/// use tt_isa::{matrix::{Banks,SrcBroadcast},isa::generated::encode::Elwadd};
/// let _ = Banks::after_reset().elwadd(Elwadd::ZERO,SrcBroadcast::None);
/// ```
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SrcBroadcast {
    None,
    Row,
    Column,
    Scalar,
}

impl SrcBroadcast {
    pub const fn row(self) -> u32 {
        matches!(self, Self::Row | Self::Scalar) as u32
    }
    pub const fn column(self) -> u32 {
        matches!(self, Self::Column | Self::Scalar) as u32
    }
    pub const fn coordinate(self, row: usize, col: usize) -> (usize, usize) {
        (
            if self.row() != 0 { 0 } else { row },
            if self.column() != 0 { 0 } else { col },
        )
    }
}

// The caller supplies addressing/accumulation fields; ownership and broadcast
// fields are always overwritten by the typed consumer.
macro_rules! elw_consumer {
    ($keep:ident, $a:ident, $b:ident, $both:ident, $encoder:ty) => {
        impl Banks<Loaded, Loaded> {
            pub fn $keep(
                self,
                base: $encoder,
                broadcast: SrcBroadcast,
            ) -> Result<(Instruction, Self), EncodeError> {
                Ok((
                    base.flip_src_a(0)
                        .flip_src_b(0)
                        .broadcast_src_b_row(broadcast.row())
                        .broadcast_src_b_col0(broadcast.column())
                        .encode()?,
                    banks(),
                ))
            }
            pub fn $a(
                self,
                base: $encoder,
                broadcast: SrcBroadcast,
            ) -> Result<(Instruction, Banks<Empty, Loaded>), EncodeError> {
                Ok((
                    base.flip_src_a(1)
                        .flip_src_b(0)
                        .broadcast_src_b_row(broadcast.row())
                        .broadcast_src_b_col0(broadcast.column())
                        .encode()?,
                    banks(),
                ))
            }
            pub fn $b(
                self,
                base: $encoder,
                broadcast: SrcBroadcast,
            ) -> Result<(Instruction, Banks<Loaded, Empty>), EncodeError> {
                Ok((
                    base.flip_src_a(0)
                        .flip_src_b(1)
                        .broadcast_src_b_row(broadcast.row())
                        .broadcast_src_b_col0(broadcast.column())
                        .encode()?,
                    banks(),
                ))
            }
            pub fn $both(
                self,
                base: $encoder,
                broadcast: SrcBroadcast,
            ) -> Result<(Instruction, Banks<Empty, Empty>), EncodeError> {
                Ok((
                    base.flip_src_a(1)
                        .flip_src_b(1)
                        .broadcast_src_b_row(broadcast.row())
                        .broadcast_src_b_col0(broadcast.column())
                        .encode()?,
                    banks(),
                ))
            }
        }
    };
}
elw_consumer!(
    elwadd,
    elwadd_release_a,
    elwadd_release_b,
    elwadd_release_both,
    encode::Elwadd
);
elw_consumer!(
    elwsub,
    elwsub_release_a,
    elwsub_release_b,
    elwsub_release_both,
    encode::Elwsub
);
elw_consumer!(
    elwmul,
    elwmul_release_a,
    elwmul_release_b,
    elwmul_release_both,
    encode::Elwmul
);

// --- Matrix Unit consumers -----------------------------------------------------

impl Banks<Loaded, Loaded> {
    /// Copy one or four aligned SrcB rows to SrcA, retaining both banks.
    ///
    /// ```compile_fail
    /// use tt_isa::{matrix::Banks, isa::generated::encode};
    /// let (_, b) = Banks::after_reset().unpack_a(encode::UnpacrRegular::ZERO).unwrap();
    /// let (_, b) = b.unpack_b_partial(encode::UnpacrRegular::ZERO).unwrap();
    /// let _ = b.movb2a(0, 0, true, 0);
    /// ```
    ///
    /// ```compile_fail
    /// use tt_isa::{matrix::Banks, isa::generated::encode};
    /// let (_, b) = Banks::after_reset().unpack_a_partial(encode::UnpacrRegular::ZERO).unwrap();
    /// let (_, b) = b.unpack_b(encode::UnpacrRegular::ZERO).unwrap();
    /// let _ = b.movb2a(0, 0, true, 0);
    /// ```
    ///
    /// ```compile_fail
    /// use tt_isa::matrix::Banks;
    /// let _ = Banks::after_reset().movb2a(0, 0, false, 0);
    /// ```
    pub fn movb2a(
        self,
        src_a_row: u32,
        addr_mod: u32,
        four_rows: bool,
        src_b_row: u32,
    ) -> Result<(Instruction, Self), EncodeError> {
        Ok((
            encode::movb2_a(src_a_row, addr_mod, u32::from(four_rows), src_b_row)?,
            banks(),
        ))
    }

    /// `GMPOOL`, keeping both sources for accumulation. Scaling, NaNs and
    /// partial argmax follow the matrix-unit contract, not IEEE max.
    pub fn gmpool(self, base: encode::Gmpool) -> Result<(Instruction, Self), EncodeError> {
        Ok((base.flip_src_a(0).flip_src_b(0).encode()?, banks()))
    }

    /// `GAPOOL`'s four-row matrix product, keeping both sources. A mean
    /// requires an explicit SrcB matrix of averaging weights.
    pub fn gapool(self, addr_mod: u32, dst_row: u32) -> Result<(Instruction, Self), EncodeError> {
        Ok((encode::gapool(0, 0, addr_mod, dst_row)?, banks()))
    }

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
    /// Copy one or four aligned Dst rows to SrcA, retaining ownership.
    ///
    /// ```compile_fail
    /// use tt_isa::{matrix::Banks, isa::generated::encode};
    /// let (_, b) = Banks::after_reset().unpack_a_partial(encode::UnpacrRegular::ZERO).unwrap();
    /// let _ = b.movd2a(encode::Movd2A::ZERO);
    /// ```
    /// The measured encoder checks row and modifier field widths.
    ///
    /// ```compile_fail
    /// use tt_isa::{matrix::Banks, isa::generated::encode};
    /// let _ = Banks::after_reset().movd2a(encode::Movd2A::ZERO);
    /// ```
    pub fn movd2a(self, base: encode::Movd2A) -> Result<(Instruction, Self), EncodeError> {
        Ok((base.encode()?, banks()))
    }

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
    /// Copy one or four aligned Dst rows to SrcB, retaining ownership.
    ///
    /// ```compile_fail
    /// use tt_isa::{matrix::Banks, isa::generated::encode};
    /// let (_, b) = Banks::after_reset().unpack_b_partial(encode::UnpacrRegular::ZERO).unwrap();
    /// let _ = b.movd2b(encode::Movd2B::ZERO);
    /// ```
    ///
    /// ```compile_fail
    /// use tt_isa::{matrix::Banks, isa::generated::encode};
    /// let _ = Banks::after_reset().movd2b(encode::Movd2B::ZERO);
    /// ```
    pub fn movd2b(self, base: encode::Movd2B) -> Result<(Instruction, Self), EncodeError> {
        Ok((base.encode()?, banks()))
    }

    /// Transpose the aligned 16x16 block in SrcB rows 16..32, keeping
    /// ownership. This moves reduced-precision Src datums, not F32 words.
    /// Blackhole silicon validation is pending; the encoding retains its
    /// Wormhole-only provenance until measured on both cards.
    ///
    /// ```compile_fail
    /// use tt_isa::matrix::Banks;
    /// let _ = Banks::after_reset().transpose_b();
    /// ```
    pub fn transpose_b(self) -> Result<(Instruction, Self), EncodeError> {
        Ok((encode::trnspsrcb()?, banks()))
    }

    /// `MOVB2D`: copy `SrcB` rows into `Dst`. Does not flip.
    pub fn movb2d(self, base: encode::Movb2D) -> Result<(Instruction, Self), EncodeError> {
        Ok((base.encode()?, banks()))
    }

    /// `SETRWC` with only `FlipSrcB`.
    pub fn release_b(self) -> Result<(Instruction, Banks<A, Empty>), EncodeError> {
        Ok((release(false, true)?, banks()))
    }
}

/// The documented TRNSPSRCB permutation on raw Src datums. Rows 0..16
/// remain untouched; format conversion belongs to the unpacker model.
pub fn transpose_b_reference(src: &mut [[u32; 16]; 32]) {
    for i in 0..16 {
        for j in 0..i {
            let a = src[16 + i][j];
            src[16 + i][j] = src[16 + j][i];
            src[16 + j][i] = a;
        }
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
    fn histogram_reset_is_the_measured_operand_free_word() {
        assert_eq!(clear_exponent_history().word(), 0x21000000);
    }

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

/// Independent ports of the pinned MOVD2A/MOVD2B/MOVB2A functional models,
/// with the step97 both-card correction for unaligned Src write addresses.
/// Inputs use physical Dst/Src bit layouts, rather than IEEE float bits.
pub mod moves {
    /// Destination-to-source conversion style (MOVD2B also uses SrcA's format).
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum Style {
        Bf16,
        Fp16,
        Tf32,
    }

    /// Blackhole row arguments plus RWC/base offsets. Only the read side
    /// aligns a four-row block; the Src write side wraps without alignment.
    /// Step97 on both cards corrects the pinned model's write-side alignment
    /// (see the step97 addendum in docs/learnings/ttsim-divergence.md).
    pub const fn rows(dst: u32, src: u32, four: bool) -> (usize, usize, usize) {
        if four {
            ((dst & 0x3fc) as usize, (src & 0x3f) as usize, 4)
        } else {
            ((dst & 0x3ff) as usize, (src & 0x3f) as usize, 1)
        }
    }

    /// One physical Dst datum. Unsupported 16-bit/TF32 and low-half forms
    /// return None, matching the specification's undefined behavior.
    pub const fn dst_to_src(mut x: u32, wide: bool, low: bool, style: Style) -> Option<u32> {
        if !wide && (low || matches!(style, Style::Tf32)) {
            return None;
        }
        if wide {
            if low {
                x = (x << 16) | (x & 0xffff);
            }
            if matches!(style, Style::Tf32) {
                if low {
                    return Some(x & 0x1fff);
                }
                x >>= 13;
                return Some((x & 0x7f800) | ((x & 7) << 8) | ((x & 0x7f8) >> 3));
            }
            x >>= 16;
        }
        Some(match style {
            Style::Bf16 => ((x & 0xff00) << 3) | (x & 0xff),
            Style::Fp16 => ((x & 0xffe0) << 3) | (x & 0x1f),
            Style::Tf32 => unreachable!(),
        })
    }

    /// Blackhole TruncateSrc is the identity for every format. MOVB2A flushes
    /// zero-exponent datums (including signed zero) first when enabled.
    pub const fn b_to_a(x: u32, flush_denormals: bool) -> u32 {
        if flush_denormals && x & 0xff == 0 {
            0
        } else {
            x
        }
    }

    /// MOVB2A on raw Blackhole Src datums, with independent six-bit row
    /// wrapping on both banks, alignment on the read side and column masks.
    pub fn copy_b(
        src_b: &[[u32; 16]; 64],
        src_a: &mut [[u32; 16]; 64],
        a_row: u32,
        b_row: u32,
        four: bool,
        lane_masks: [u8; 8],
        flush: bool,
    ) {
        let mask = if four { 0x3c } else { 0x3f };
        let a = (a_row & 0x3f) as usize;
        let b = (b_row & mask) as usize;
        for r in 0..if four { 4 } else { 1 } {
            for c in 0..16 {
                if lane_masks[c / 2] & (1 << (c & 1)) == 0 {
                    src_a[(a + r) & 63][c] = b_to_a(src_b[b + r][c], flush);
                }
            }
        }
    }

    /// Apply a MOVD2A/B to a 1024-row logical Dst view and a Src bank. Each
    /// lane pair's two mask bits inhibit the corresponding column writes.
    pub fn copy_dst(
        dst: &[[u32; 16]; 1024],
        src: &mut [[u32; 16]; 64],
        row_args: (u32, u32, bool),
        lane_masks: [u8; 8],
        style: Style,
        wide: bool,
        low: bool,
    ) -> Option<()> {
        let (d, s, n) = rows(row_args.0, row_args.1, row_args.2);
        for r in 0..n {
            for c in 0..16 {
                if lane_masks[c / 2] & (1 << (c & 1)) == 0 {
                    src[(s + r) & 63][c] = dst_to_src(dst[d + r][c], wide, low, style)?;
                }
            }
        }
        Some(())
    }
}

#[cfg(test)]
mod register_move_tests {
    use super::*;
    #[test]
    fn checked_moves_keep_banks_and_measured_fields() {
        for four in [false, true] {
            for modifier in 0..8 {
                let (_, b) = Banks::after_reset()
                    .unpack_a(encode::UnpacrRegular::ZERO)
                    .unwrap();
                let (_, b) = b.unpack_b(encode::UnpacrRegular::ZERO).unwrap();
                let base = encode::Movd2A::ZERO
                    .move4_rows(u32::from(four))
                    .src_row(63)
                    .dst_row(1023)
                    .addr_mod(modifier);
                let (a, b) = b.movd2a(base).unwrap();
                assert_eq!(a.word(), base.encode().unwrap().word());
                assert_eq!(a.word() & (7 << 14), modifier << 14);
                let base = encode::Movd2B::ZERO
                    .move4_rows(u32::from(four))
                    .src_row(63)
                    .dst_row(1023)
                    .addr_mod(modifier);
                let (i, b) = b.movd2b(base).unwrap();
                assert_eq!(i.word(), base.encode().unwrap().word());
                let (i, b) = b.movb2a(63, modifier, four, 63).unwrap();
                assert_eq!(
                    i.word(),
                    encode::movb2_a(63, modifier, u32::from(four), 63)
                        .unwrap()
                        .word()
                );
                let (_, b) = b.release_a().unwrap();
                let _ = b.release_b().unwrap();
            }
        }
        let (_, b) = Banks::after_reset()
            .unpack_a(encode::UnpacrRegular::ZERO)
            .unwrap();
        assert!(b.movd2a(encode::Movd2A::ZERO.src_row(64)).is_err());
        let (_, b) = Banks::after_reset()
            .unpack_b(encode::UnpacrRegular::ZERO)
            .unwrap();
        assert!(b.movd2b(encode::Movd2B::ZERO.dst_row(1024)).is_err());
    }
}
