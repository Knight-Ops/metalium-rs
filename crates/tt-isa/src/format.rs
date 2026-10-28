//! One descriptor for every fact that depends on a tile's format (ADR 0004;
//! `docs/learnings/dataflow-compiler.md` §4).
//!
//! The Matrix Unit computes in three numeric families. Every L1 format is converted into
//! one of them by the unpacker and out of one by the packer
//! (`Unpackers/FormatConversion.md`, `Packers/FormatConversion.md`, `SrcASrcB.md` "Data
//! types"). So a format decides a tile's bytes, the unpacker's and packer's configuration
//! and, through the bits it brings into `Src`, how many fidelity phases an exact product
//! takes: never the schedule of compute. [`TileFormat`] carries those facts for the stack's
//! tile (1024 datums, four 16x16 faces, one 16-byte header) so that nothing above it has to
//! name a format to size a slot, configure an unpack or choose a fidelity.
//!
//! [`crate::tile::L1Format`] stays the shape-free datum format; this is that format at the
//! stack's tile shape.

use crate::tile::{L1Format, TileDescriptor, TileImage};

/// The numeric family a format computes in (`SrcASrcB.md`, the `MVMUL` combination
/// table): an `MVMUL` takes both operands from one family.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Family {
    /// 8-bit exponent: TF32 or BF16 in `Src`, FP32 or BF16 in `Dst`.
    E8,
    /// 5-bit exponent: FP16 in `Src`, FP32 or FP16 in `Dst`.
    E5,
    /// Integer "8" in `Src`, integer "32" in `Dst`.
    Int,
}

/// A datum type in `SrcA`/`SrcB` (`SrcASrcB.md`, "Data types").
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SrcType {
    Tf32,
    Bf16,
    Fp16,
    Int8,
}

impl SrcType {
    /// The family this `Src` type multiplies in.
    pub const fn family(self) -> Family {
        match self {
            SrcType::Tf32 | SrcType::Bf16 => Family::E8,
            SrcType::Fp16 => Family::E5,
            SrcType::Int8 => Family::Int,
        }
    }

    /// The mantissa bits below the implicit one that each fidelity slice of an operand
    /// takes, `(high, low)`, in `SrcA` and in `SrcB` (`SrcASrcB.md`, "Fidelity phases
    /// (floating-point)"). Phase `p` multiplies `SrcA`'s slice `p & 1` by `SrcB`'s slice
    /// `p >> 1` (0 high, 1 low). `None` for integers, whose phases follow magnitudes.
    const fn slices(self) -> Option<((u32, u32), (u32, u32))> {
        match self {
            // BF16, and BFP losslessly converted to it: SrcA 4 + 3, SrcB 6 + 1.
            SrcType::Bf16 => Some(((4, 3), (6, 1))),
            // TF32 and FP16: SrcA 4 + 5, one bit taken by no phase; SrcB 6 + 4.
            SrcType::Tf32 | SrcType::Fp16 => Some(((4, 5), (6, 4))),
            SrcType::Int8 => None,
        }
    }
}

/// A tile of the stack's shape in one L1 format.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TileFormat {
    l1: L1Format,
}

impl TileFormat {
    /// Datums in a tile: four 16x16 faces, laid out flat (`tt_kernels::datapath`).
    pub const TILE_DATUMS: u32 = 1024;
    /// A slot's size is a multiple of this: a GDDR read or write needs its L1 and GDDR
    /// addresses congruent mod 64 ([`crate::dram::ALIGN`]), and slots placed end to end
    /// keep that congruence.
    pub const SLOT_ALIGN: usize = 64;

    pub const FP32: TileFormat = TileFormat::new(L1Format::Fp32);
    pub const BF16: TileFormat = TileFormat::new(L1Format::Bf16);
    pub const FP16: TileFormat = TileFormat::new(L1Format::Fp16);
    pub const BFP8: TileFormat = TileFormat::new(L1Format::Bfp8);
    pub const BFP4: TileFormat = TileFormat::new(L1Format::Bfp4);
    pub const BFP2: TileFormat = TileFormat::new(L1Format::Bfp2);

    pub const fn new(l1: L1Format) -> Self {
        TileFormat { l1 }
    }

    /// The datum format in L1.
    pub const fn l1(self) -> L1Format {
        self.l1
    }

    /// The tile's descriptor: 1024 datums in one row (zero `Z` and `W` mean one,
    /// `UNPACR_Regular.md:70-71`), uncompressed, no digest. The shape alone: the data format
    /// fields are the unpack's to set.
    pub const fn descriptor(self) -> TileDescriptor {
        TileDescriptor::zeroed()
            .with_x_dim(Self::TILE_DATUMS)
            .with_y_dim(1)
            .with_z_dim(0)
            .with_w_dim(0)
            .with_is_uncompressed(true)
    }

    /// The tile's image: header, exponent section (BFP), datums.
    pub const fn image(self) -> TileImage {
        match TileImage::new(self.descriptor(), self.l1) {
            Ok(image) => image,
            Err(_) => panic!("a 1024-datum row fits every format"),
        }
    }

    /// Bytes of one tile's image in L1, header included.
    pub const fn tile_bytes(self) -> usize {
        self.image().total_bytes()
    }

    /// Bytes a tile takes as a slot: its image rounded up to [`Self::SLOT_ALIGN`]. FP32's
    /// is [`crate::dm::TILE_SLOT`] and BF16's [`crate::dm::BF16_TILE_SLOT`].
    pub const fn slot_bytes(self) -> u64 {
        self.tile_bytes().next_multiple_of(Self::SLOT_ALIGN) as u64
    }

    /// The family the format computes in, or `None` for a format no `MVMUL` takes
    /// (INT16 is opaque transfer only; INT32 reaches `Dst` only).
    pub const fn family(self) -> Option<Family> {
        match self.l1 {
            L1Format::Fp32
            | L1Format::Tf32
            | L1Format::Bf16
            | L1Format::Bfp8
            | L1Format::Bfp4
            | L1Format::Bfp2 => Some(Family::E8),
            L1Format::Fp16
            | L1Format::Fp8
            | L1Format::Bfp8a
            | L1Format::Bfp4a
            | L1Format::Bfp2a => Some(Family::E5),
            L1Format::Int8 | L1Format::Uint8 => Some(Family::Int),
            L1Format::Int32 | L1Format::Int16 => None,
        }
    }

    /// The significant mantissa bits below the implicit one that a datum of this format
    /// brings into `Src` as `to`, or `None` where the unpacker has no such conversion
    /// (`Unpackers/FormatConversion.md`). A block-float datum is sign and magnitude under a
    /// shared exponent: BFP8's seven magnitude bits normalize to at most six below the
    /// leading one, BFP4's three to two, BFP2's one to none.
    pub const fn src_fraction_bits(self, to: SrcType) -> Option<u32> {
        use L1Format as F;
        use SrcType as S;
        match (self.l1, to) {
            (F::Fp32 | F::Tf32, S::Tf32 | S::Fp16) => Some(10),
            (F::Fp32 | F::Tf32, S::Bf16) => Some(7),
            (F::Bf16, S::Tf32 | S::Bf16) => Some(7),
            (F::Bfp8, S::Tf32 | S::Bf16) | (F::Bfp8a, S::Fp16) => Some(6),
            (F::Bfp4, S::Tf32 | S::Bf16) | (F::Bfp4a, S::Fp16) | (F::Fp8, S::Fp16) => Some(2),
            (F::Bfp2, S::Tf32 | S::Bf16) | (F::Bfp2a, S::Fp16) => Some(0),
            (F::Fp16, S::Fp16) => Some(10),
            (F::Int8 | F::Uint8, S::Int8) => Some(0),
            _ => None,
        }
    }

    /// The unpacker's `(InDataFormat, OutDataFormat)` codes for this format into `Src` as
    /// `to` (`Unpackers/FormatConversion.md`, "Configuring the conversion"), where both
    /// codes are measured ([`L1Format::code`]); `None` for a conversion the hardware does
    /// not do or whose code is not yet measured.
    pub const fn src_codes(self, to: SrcType) -> Option<(u32, u32)> {
        if self.src_fraction_bits(to).is_none() {
            return None;
        }
        let out = match (self.l1, to) {
            // FP32 (or TF32) is cut to the `Src` type: `OutDataFormat` names it.
            (L1Format::Fp32 | L1Format::Tf32, SrcType::Tf32) => L1Format::Tf32,
            (L1Format::Fp32 | L1Format::Tf32, SrcType::Bf16) => L1Format::Bf16,
            (L1Format::Fp32 | L1Format::Tf32, SrcType::Fp16) => L1Format::Fp16,
            // Everything else is moved as it is: both codes name the L1 format.
            (other, _) => other,
        };
        let input = match self.l1 {
            // TF32 is staged in L1 as FP32 (its low mantissa bits zero).
            L1Format::Tf32 => L1Format::Fp32,
            other => other,
        };
        match (input.code(), out.code()) {
            (Some(i), Some(o)) => Some((i, o)),
            _ => None,
        }
    }
}

/// The fidelity phases an `MVMUL` needs to take every mantissa bit both operands bring
/// into `Src`: `src_a` the operand in `SrcA` (a product's `B`, in1, in this stack) and
/// `src_b` the one in `SrcB` (`A`, in0), each as its L1 format and `Src` type.
///
/// Phases run in order from 0, so the count is one past the highest phase whose two
/// slices can both be non-zero. `None` when no count takes every bit (TF32 or FP16 with
/// all ten mantissa bits in `SrcA`, whose lowest bit no phase takes), when the two types
/// are not one family, or when either conversion does not exist. Integer operands take
/// four: their phases follow magnitudes, not formats (`SrcASrcB.md`, "Fidelity phases
/// (integer)").
///
/// For two operands of one format this is the vendor table's "phases for full precision"
/// column. Whether running these phases gives the same bits as running all four is a
/// separate question, hypothesis H3 of `docs/learnings/dataflow-compiler.md`.
pub const fn full_precision_phases(
    src_a: (TileFormat, SrcType),
    src_b: (TileFormat, SrcType),
) -> Option<u32> {
    let (a_fmt, a_ty) = src_a;
    let (b_fmt, b_ty) = src_b;
    if !matches!(
        (a_ty.family(), b_ty.family()),
        (Family::E8, Family::E8) | (Family::E5, Family::E5) | (Family::Int, Family::Int)
    ) {
        return None;
    }
    let (Some(a_bits), Some(b_bits)) =
        (a_fmt.src_fraction_bits(a_ty), b_fmt.src_fraction_bits(b_ty))
    else {
        return None;
    };
    let (Some(((a_hi, a_lo), _)), Some((_, (b_hi, b_lo)))) = (a_ty.slices(), b_ty.slices()) else {
        return Some(4);
    };
    if a_bits > a_hi + a_lo || b_bits > b_hi + b_lo {
        return None;
    }
    let a_low = a_bits > a_hi;
    let b_low = b_bits > b_hi;
    Some(match (a_low, b_low) {
        (true, true) => 4,
        (false, true) => 3,
        (true, false) => 2,
        (false, false) => 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stack's slot constants are this descriptor's, and the BFP slots are the sizes
    /// `tt_kernels::bfp` has used (1104 / 592 / 336-byte images in 1152 / 640 / 384).
    #[test]
    fn slots_are_the_images_rounded_to_64() {
        assert_eq!(TileFormat::FP32.tile_bytes(), 16 + 4096);
        assert_eq!(TileFormat::FP32.slot_bytes(), crate::dm::TILE_SLOT);
        assert_eq!(TileFormat::BF16.slot_bytes(), crate::dm::BF16_TILE_SLOT);
        for (f, image, slot) in [
            (TileFormat::BFP8, 1104, 1152),
            (TileFormat::BFP4, 592, 640),
            (TileFormat::BFP2, 336, 384),
        ] {
            assert_eq!(f.tile_bytes(), image, "{f:?}");
            assert_eq!(f.slot_bytes(), slot, "{f:?}");
        }
    }

    /// Two operands of one format reproduce `SrcASrcB.md`'s "phases for full precision".
    #[test]
    fn one_format_matches_the_vendor_table() {
        let both = |f: TileFormat, t: SrcType| full_precision_phases((f, t), (f, t));
        assert_eq!(both(TileFormat::BFP2, SrcType::Bf16), Some(1));
        assert_eq!(both(TileFormat::BFP4, SrcType::Bf16), Some(1));
        assert_eq!(both(TileFormat::new(L1Format::Fp8), SrcType::Fp16), Some(1));
        assert_eq!(
            both(TileFormat::new(L1Format::Bfp4a), SrcType::Fp16),
            Some(1)
        );
        assert_eq!(both(TileFormat::BFP8, SrcType::Bf16), Some(2));
        assert_eq!(
            both(TileFormat::new(L1Format::Bfp8a), SrcType::Fp16),
            Some(2)
        );
        assert_eq!(both(TileFormat::BF16, SrcType::Bf16), Some(4));
        // "Not possible for SrcA": its lowest TF32/FP16 mantissa bit is in no phase.
        assert_eq!(both(TileFormat::FP32, SrcType::Tf32), None);
        assert_eq!(both(TileFormat::FP16, SrcType::Fp16), None);
    }

    /// `Fidelity::Lo` is exact for at most seven significant bits in `A` (`SrcB`) and five
    /// in `B` (`SrcA`) (`tt_kernels::matmul::Fidelity`): six and four below the leading one.
    #[test]
    fn mixed_pairs_follow_each_operands_bits() {
        let p = full_precision_phases;
        let a = |f| (f, SrcType::Bf16);
        // BFP4 weights in SrcA against BFP8 activations in SrcB: only high slices.
        assert_eq!(p(a(TileFormat::BFP4), a(TileFormat::BFP8)), Some(1));
        // BFP8 weights in SrcA need SrcA's low slice: phases 0 and 1.
        assert_eq!(p(a(TileFormat::BFP8), a(TileFormat::BFP4)), Some(2));
        // BF16 activations in SrcB need SrcB's low slice; BFP4 weights do not need SrcA's:
        // phases 0 and 2, so three in order.
        assert_eq!(p(a(TileFormat::BFP4), a(TileFormat::BF16)), Some(3));
        // The stack's default today: TF32 activations in SrcB, BF16 weights in SrcA.
        assert_eq!(
            p(
                (TileFormat::BF16, SrcType::Tf32),
                (TileFormat::FP32, SrcType::Tf32)
            ),
            Some(4)
        );
        // Families do not mix, and missing conversions are refused.
        assert_eq!(
            p(a(TileFormat::BF16), (TileFormat::FP16, SrcType::Fp16)),
            None
        );
        assert_eq!(
            p((TileFormat::BF16, SrcType::Fp16), a(TileFormat::BF16)),
            None
        );
    }

    /// The unpacker codes are the ones `tt_kernels::matmul::SrcRoute` uses.
    #[test]
    fn src_codes_are_the_measured_routes() {
        assert_eq!(TileFormat::FP32.src_codes(SrcType::Tf32), Some((0, 4)));
        assert_eq!(TileFormat::FP32.src_codes(SrcType::Bf16), Some((0, 5)));
        assert_eq!(TileFormat::BF16.src_codes(SrcType::Bf16), Some((5, 5)));
        assert_eq!(TileFormat::BF16.src_codes(SrcType::Tf32), Some((5, 5)));
        assert_eq!(TileFormat::BFP8.src_codes(SrcType::Bf16), Some((6, 6)));
        assert_eq!(TileFormat::BFP4.src_codes(SrcType::Tf32), Some((7, 7)));
        assert_eq!(TileFormat::BFP2.src_codes(SrcType::Bf16), Some((15, 15)));
        assert_eq!(TileFormat::FP16.src_codes(SrcType::Fp16), Some((1, 1)));
        // BF16 into FP16 does not exist; BFP8a's code is not measured.
        assert_eq!(TileFormat::BF16.src_codes(SrcType::Fp16), None);
        assert_eq!(
            TileFormat::new(L1Format::Bfp8a).src_codes(SrcType::Fp16),
            None
        );
    }

    #[test]
    fn families() {
        assert_eq!(TileFormat::BFP2.family(), Some(Family::E8));
        assert_eq!(TileFormat::FP16.family(), Some(Family::E5));
        assert_eq!(TileFormat::new(L1Format::Uint8).family(), Some(Family::Int));
        assert_eq!(TileFormat::new(L1Format::Int32).family(), None);
    }
}
