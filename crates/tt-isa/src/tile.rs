//! The layout of a tile of datums in L1: what the unpacker reads and the packer
//! writes.
//!
//! # What a "tile" actually is
//!
//! Not a 32x32 square. `UNPACR_Regular.md`'s input address generator treats the
//! datums of a tile as a flat four-dimensional array indexed `W`, `Z`, `Y`, `X`
//! with `X` varying fastest:
//!
//! ```text
//! FirstDatum = ((W * ZDim + Z) * YDim + Y) * XDim + X
//! ```
//!
//! The dimensions come from a [`TileDescriptor`] in backend configuration, so the
//! familiar 32x32-tile-of-four-16x16-faces shape is a *choice of descriptor*, not a
//! property of the hardware. Building a layout library around a hardcoded 32x32
//! would bake in an assumption the specification does not make, so this module
//! models the descriptor and leaves the shape to the caller.
//!
//! # Sources, and how far they can be trusted
//!
//! `BlackholeA0/.../UNPACR_Regular.md` is a three-line redirect stating that the
//! document is shared and conditionalized inline with `TTArchitecture`, so the
//! Wormhole page is authoritative for Blackhole and is cited directly below.
//! `FloatBitPatterns.md` is absent from the Blackhole tree entirely and is *not*
//! marked shared, so everything sourced from it is `UNVERIFIED` — see
//! [`bfp8_to_bf16`] and the FP16 notes on [`L1Format`].
//!
//! # This is not `isa::generated::datum`
//!
//! That table describes the `SrcA`/`SrcB`/`Dst` **register files**: 19-bit datums in
//! `Src`, 16- or 32-bit swizzled ones in `Dst`. This module describes datums **in
//! L1**, which are plain and unswizzled. Converting a host buffer for the unpacker
//! is this module's job; reading a result back out of `Dst` is that table's.

use crate::cfg::generated::thcon;

/// The format of a datum as it sits in L1.
///
/// The thirteen the packer can emit and the unpacker can consume
/// (`Packers/FormatConversion.md:85-104`). Conversion is implemented for
/// [`L1Format::Fp32`], [`L1Format::Bf16`] and [`L1Format::Fp16`]; the rest are
/// enumerated because the address arithmetic depends only on
/// [`L1Format::datum_bits`] and on whether there is an exponent section, and both
/// are known for all of them.
///
/// The numeric `InDataFormat`/`OutDataFormat` encoding is deliberately absent: it
/// appears in neither the specification tree nor `cfg_defines.h`. See the module
/// note in `docs/implementation-checklist.md` open question 9.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum L1Format {
    /// IEEE754 FP32.
    Fp32,
    /// FP32 with the mantissa cut to 10 bits; the low 13 bits are always zero.
    /// Still 32 bits per datum in L1.
    Tf32,
    /// FP32 truncated to a 7-bit mantissa.
    Bf16,
    /// IEEE754 FP16 for normal values, but **no bit pattern means NaN** and
    /// infinity is encoded differently (`FloatBitPatterns.md:67-88`).
    Fp16,
    /// e5m2. Follows the FP16 patterns with the mantissa cut to 2 bits.
    Fp8,
    /// Block float: 8-bit sign-magnitude datum, 8-bit exponent per 16 datums.
    Bfp8,
    /// As [`L1Format::Bfp8`], but the shared exponent is FP16-biased and only its
    /// low five bits are used.
    Bfp8a,
    Bfp4,
    Bfp4a,
    Bfp2,
    Bfp2a,
    /// Sign-magnitude, one sign bit and 31 magnitude bits. Not two's complement.
    Int32,
    /// Sign-magnitude, one sign bit and 15 magnitude bits. Often used to carry
    /// opaque 16-bit data instead.
    Int16,
    /// Sign-magnitude, one sign bit and 7 magnitude bits.
    Int8,
    /// Plain `u8`, 0 through 255.
    Uint8,
}

impl L1Format {
    /// Width of one datum **in bits**.
    ///
    /// Bits rather than bytes because BFP4 packs two datums per byte and BFP2 packs
    /// four (`Packers/FormatConversion.md:103`), so a byte-valued accessor could not
    /// describe them. The unpacker's own model has the same shape, carrying
    /// `DatumSizeBytes` as a fraction (`UNPACR_Regular.md:92-98`).
    /// The 4-bit `InDataFormat` / `OutDataFormat` code, where it has been measured.
    ///
    /// **`MEASURED`, not documented** — a different status from the `UNVERIFIED`
    /// marker the rest of this crate uses for Wormhole-sourced facts. Those are
    /// hypotheses taken from a document; these are measurements taken from the
    /// simulator, because there is no document: the encoding appears in neither the
    /// specification tree nor `cfg_defines.h`.
    ///
    /// # How these three were established
    ///
    /// `crates/tt-tests/tests/probe_unpack.rs::survey_the_data_format_codes` stages
    /// known FP32 datums in L1 and unpacks them to `Dst` once for each of the 256
    /// `(InDataFormat, OutDataFormat)` pairs, each in its own `fork_scope`. ttsim
    /// validates pairs and names its refusals, which separates three cases: a pair
    /// it rejects as `incompatible`/`mismatches`, one it has not modelled
    /// (`unpack_to_dst=1 in_data_format=N`), and one that runs. Exactly three run:
    /// `(0, 0)`, `(0, 4)` and `(8, 8)`, and all three return the staged FP32 bits
    /// unchanged.
    ///
    /// That identifies them, given two independent constraints from the
    /// documentation:
    ///
    /// * `Packers/InputAddressGenerator.md` switches on `In_data_format & 3`, with
    ///   `0` meaning four bytes per datum. Codes 0, 4 and 8 are all four-byte, and
    ///   `UNPACR_Regular.md:95` says the four-byte formats are exactly FP32, TF32
    ///   and INT32.
    /// * `(0, 8)` is refused as `incompatible` while `(0, 4)` runs, so 0 and 4 are
    ///   the same *kind* and 8 is not — float against integer. `UNPACR_Regular.md`
    ///   also notes that "when unpacking to `Dst`, TF32 means FP32", which is why
    ///   `(0, 4)` returns FP32 bits unchanged rather than truncating a mantissa.
    ///
    /// # What is deliberately still `None`
    ///
    /// Every 16-bit and block-float code. ttsim declines `UnpackToDst` for them
    /// outright (`UnimplementedFunctionality: tensix_unpacr: unpack_to_dst=1
    /// in_data_format=1`), so this path cannot measure them and guessing from
    /// tt-metal's `DataFormat` enum would be transcription. They are reachable
    /// through the packer instead, which is where they should be pinned.
    ///
    /// Re-derive all of it at the first silicon gate; a mismatch is a finding.
    pub const fn code(self) -> Option<u32> {
        match self {
            L1Format::Fp32 => Some(0),
            L1Format::Tf32 => Some(4),
            L1Format::Int32 => Some(8),
            _ => None,
        }
    }

    /// The format a code names, for the codes [`Self::code`] has measured.
    pub const fn from_code(code: u32) -> Option<L1Format> {
        match code {
            0 => Some(L1Format::Fp32),
            4 => Some(L1Format::Tf32),
            8 => Some(L1Format::Int32),
            _ => None,
        }
    }

    pub const fn datum_bits(self) -> u32 {
        match self {
            L1Format::Bfp2 | L1Format::Bfp2a => 2,
            L1Format::Bfp4 | L1Format::Bfp4a => 4,
            L1Format::Bfp8 | L1Format::Bfp8a | L1Format::Fp8 | L1Format::Int8 | L1Format::Uint8 => {
                8
            }
            L1Format::Bf16 | L1Format::Fp16 | L1Format::Int16 => 16,
            L1Format::Fp32 | L1Format::Tf32 | L1Format::Int32 => 32,
        }
    }

    /// Is this one of the six block-float formats?
    pub const fn is_bfp(self) -> bool {
        matches!(
            self,
            L1Format::Bfp2
                | L1Format::Bfp2a
                | L1Format::Bfp4
                | L1Format::Bfp4a
                | L1Format::Bfp8
                | L1Format::Bfp8a
        )
    }

    /// Does a tile in this format carry a separate exponent stream in L1?
    ///
    /// "For all BFP formats, there is a separate output stream in L1 for the
    /// exponents, consisting of one byte per 16 datums"
    /// (`Packers/FormatConversion.md:102`).
    pub const fn has_exponent_section(self) -> bool {
        self.is_bfp()
    }

    /// How many datums share one exponent. Meaningless unless
    /// [`L1Format::has_exponent_section`].
    pub const DATUMS_PER_EXPONENT: u32 = 16;
}

/// Bit layout of the 128-bit `TileDescriptor`, from `UNPACR_Regular.md:669-688`.
///
/// # Why this is hand-written
///
/// "The various fields of `TileDescriptor` are not described in `cfg_defines.h`"
/// (`UNPACR_Regular.md:669`), so `cargo xtask gen-cfg` cannot produce it: the header
/// carries only a 128-bit mask with a shift of zero, which the generator renders as
/// an opaque [`crate::cfg::ConfigSpan`]. The field table is in prose on that page
/// and nowhere else, which makes this the same kind of artefact as the hand-written
/// `libttsim` bindings — transcribed, cited, and tested, because there is nothing to
/// generate from.
///
/// Reserved runs are carried as must-be-zero rather than ignored, for the reason
/// [`crate::isa::InstructionDef::fixed`] gives: code that *builds* one of these has
/// to know which bits to leave alone.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct TileDescriptor {
    words: [u32; 4],
}

/// The descriptor is exactly the width of the configuration span that holds it.
///
/// If a regenerated `cfg_defines.h` ever disagreed, this would fail the build rather
/// than letting a four-word struct be written into a differently-sized register run.
const _: () = assert!(
    thcon::THCON_SEC0_REG0_TileDescriptor.words() as usize == TileDescriptor::WORDS,
    "TileDescriptor is four Config words"
);

impl TileDescriptor {
    /// Length of the descriptor in 32-bit configuration words.
    pub const WORDS: usize = 4;

    pub const fn zeroed() -> Self {
        TileDescriptor { words: [0; 4] }
    }

    /// The raw words, in ascending configuration-register order.
    pub const fn words(self) -> [u32; 4] {
        self.words
    }

    pub const fn from_words(words: [u32; 4]) -> Self {
        TileDescriptor { words }
    }

    /// Read `width` bits starting at `first_bit` of the 128-bit descriptor.
    ///
    /// Handles a field that straddles a word boundary, because one does:
    /// `BlobsYStart` is 32 bits at bit 80, so it occupies the top half of word 2 and
    /// the bottom half of word 3. Treating the descriptor as four independent words
    /// would silently truncate it.
    const fn get(self, first_bit: u32, width: u32) -> u32 {
        let word = (first_bit / 32) as usize;
        let shift = first_bit % 32;
        let in_this_word = 32 - shift;
        let mask = if width >= 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        let low = self.words[word] >> shift;
        if width <= in_this_word {
            low & mask
        } else {
            // Only reachable when `shift > 0`, so `in_this_word < 32` and the shift
            // below is well defined.
            (low | (self.words[word + 1] << in_this_word)) & mask
        }
    }

    const fn set(mut self, first_bit: u32, width: u32, value: u32) -> Self {
        let word = (first_bit / 32) as usize;
        let shift = first_bit % 32;
        let in_this_word = 32 - shift;
        let mask = if width >= 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        let value = value & mask;
        if width <= in_this_word {
            self.words[word] = (self.words[word] & !(mask << shift)) | (value << shift);
        } else {
            self.words[word] = (self.words[word] & !(mask << shift)) | (value << shift);
            let high_mask = mask >> in_this_word;
            self.words[word + 1] = (self.words[word + 1] & !high_mask) | (value >> in_this_word);
        }
        self
    }

    /// `InDataFormat`, bits 0..4. Optionally ignored in `MultiContextMode`.
    ///
    /// Returned raw because the symbolic-to-numeric mapping is undocumented; see
    /// [`L1Format`].
    pub const fn in_data_format_raw(self) -> u32 {
        self.get(0, 4)
    }

    pub const fn with_in_data_format_raw(self, value: u32) -> Self {
        self.set(0, 4, value)
    }

    /// `IsUncompressed`, bit 4. `false` when unpacking compressed data.
    pub const fn is_uncompressed(self) -> bool {
        self.get(4, 1) != 0
    }

    pub const fn with_is_uncompressed(self, value: bool) -> Self {
        self.set(4, 1, value as u32)
    }

    /// `NoBFPExpSection`, bit 5. No effect for non-BFP formats.
    pub const fn no_bfp_exp_section(self) -> bool {
        self.get(5, 1) != 0
    }

    pub const fn with_no_bfp_exp_section(self, value: bool) -> Self {
        self.set(5, 1, value as u32)
    }

    /// `BlobsPerXYPlane`, bits 8..11.
    pub const fn blobs_per_xy_plane(self) -> u32 {
        self.get(8, 3)
    }

    pub const fn with_blobs_per_xy_plane(self, value: u32) -> Self {
        self.set(8, 3, value)
    }

    /// `XDim`, bits 16..32. Datums per row.
    pub const fn x_dim(self) -> u32 {
        self.get(16, 16)
    }

    pub const fn with_x_dim(self, value: u32) -> Self {
        self.set(16, 16, value)
    }

    /// `YDim`, bits 32..40. Rows per xy plane.
    ///
    /// Unlike `ZDim` and `WDim`, a zero here is **not** rewritten to one: the
    /// specification gives that rule for `ZDim` and `WDim` only
    /// (`UNPACR_Regular.md:81-82`).
    pub const fn y_dim(self) -> u32 {
        self.get(32, 8)
    }

    pub const fn with_y_dim(self, value: u32) -> Self {
        self.set(32, 8, value)
    }

    /// `ZDim`, bits 48..56, with the documented zero-means-one rule applied
    /// (`UNPACR_Regular.md:81`).
    pub const fn z_dim(self) -> u32 {
        let raw = self.get(48, 8);
        if raw == 0 {
            1
        } else {
            raw
        }
    }

    /// `ZDim` exactly as stored, before the zero-means-one rule.
    pub const fn z_dim_raw(self) -> u32 {
        self.get(48, 8)
    }

    pub const fn with_z_dim(self, value: u32) -> Self {
        self.set(48, 8, value)
    }

    /// `WDim`, bits 64..72, with the documented zero-means-one rule applied
    /// (`UNPACR_Regular.md:82`).
    pub const fn w_dim(self) -> u32 {
        let raw = self.get(64, 8);
        if raw == 0 {
            1
        } else {
            raw
        }
    }

    /// `WDim` exactly as stored.
    pub const fn w_dim_raw(self) -> u32 {
        self.get(64, 8)
    }

    pub const fn with_w_dim(self, value: u32) -> Self {
        self.set(64, 8, value)
    }

    /// `BlobsYStart`, bits 80..112. Used when `BlobsPerXYPlane != 0`.
    pub const fn blobs_y_start(self) -> u32 {
        self.get(80, 32)
    }

    pub const fn with_blobs_y_start(self, value: u32) -> Self {
        self.set(80, 32, value)
    }

    /// `DigestSize`, bits 120..128. The tile header is `(1 + DigestSize) * 16` bytes.
    pub const fn digest_size(self) -> u32 {
        self.get(120, 8)
    }

    pub const fn with_digest_size(self, value: u32) -> Self {
        self.set(120, 8, value)
    }

    /// Bits the table marks Reserved / "Not used for anything": 6..8, 11..16,
    /// 40..48, 56..64, 72..80, 112..120.
    ///
    /// Carried so that a descriptor built field by field can be checked for stray
    /// bits rather than trusted.
    pub const fn reserved_bits_are_zero(self) -> bool {
        self.get(6, 2) == 0
            && self.get(11, 5) == 0
            && self.get(40, 8) == 0
            && self.get(56, 8) == 0
            && self.get(72, 8) == 0
            && self.get(112, 8) == 0
    }
}

/// Where each datum of a tile sits, relative to the start of the tile's L1 image.
///
/// Transcribes the first half of `UNPACR_Regular.md`'s functional model
/// (`:62-212`): the header, then the exponent section for BFP formats, then the
/// datums themselves in `W`/`Z`/`Y`/`X` order.
///
/// Offsets are **relative to the tile's base**. The unpacker computes the absolute
/// address as `(Base + Offset + 1 + DigestSize) * 16`, i.e. the configured base is
/// in 16-byte units and already steps past the header; keeping this type relative
/// means the same arithmetic serves a host buffer that has no configured base yet.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TileImage {
    descriptor: TileDescriptor,
    format: L1Format,
}

/// Why a tile image could not be described.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TileImageError {
    /// A dimension is zero where the specification has no zero-means-one rule.
    ZeroDimension { name: &'static str },
    /// The datum count overflows what the address generator can reach. `X` reaches
    /// the input address generator as a 13-bit value scaled into a 14-bit nibble
    /// offset, i.e. 8192 bytes of datums (`UNPACR_Regular.md:196-199`).
    RowTooWide { x_dim: u32, limit: u32 },
    /// A sub-byte format whose row does not end on a byte boundary. Nothing in the
    /// specification describes a datum split across the boundary between rows, so
    /// this is refused rather than guessed at.
    RaggedSubByteRow { x_dim: u32, datum_bits: u32 },
}

impl core::fmt::Display for TileImageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TileImageError::ZeroDimension { name } => {
                write!(f, "{name} is zero, and has no zero-means-one rule")
            }
            TileImageError::RowTooWide { x_dim, limit } => write!(
                f,
                "XDim = {x_dim} exceeds the {limit} datums the input address \
                 generator can reach"
            ),
            TileImageError::RaggedSubByteRow { x_dim, datum_bits } => write!(
                f,
                "{x_dim} datums of {datum_bits} bits do not fill whole bytes"
            ),
        }
    }
}

impl TileImage {
    /// Bytes the datum region is aligned and padded to, and the unit the tile base
    /// address is expressed in (`UNPACR_Regular.md:113`).
    pub const ALIGNMENT: usize = 16;

    /// Describe the image of a tile with this descriptor and format.
    pub const fn new(descriptor: TileDescriptor, format: L1Format) -> Result<Self, TileImageError> {
        if descriptor.x_dim() == 0 {
            return Err(TileImageError::ZeroDimension { name: "XDim" });
        }
        if descriptor.y_dim() == 0 {
            return Err(TileImageError::ZeroDimension { name: "YDim" });
        }

        // `XLimit = 8192 / max(1, DatumSizeBytes)` (`UNPACR_Regular.md:196`). For
        // sub-byte formats the divisor clamps to 1, so the limit is 8192 datums.
        let bits = format.datum_bits();
        let limit = if bits <= 8 { 8192 } else { 8192 / (bits / 8) };
        if descriptor.x_dim() > limit {
            return Err(TileImageError::RowTooWide {
                x_dim: descriptor.x_dim(),
                limit,
            });
        }

        if bits < 8 && (descriptor.x_dim() * bits) % 8 != 0 {
            return Err(TileImageError::RaggedSubByteRow {
                x_dim: descriptor.x_dim(),
                datum_bits: bits,
            });
        }

        Ok(TileImage { descriptor, format })
    }

    pub const fn descriptor(self) -> TileDescriptor {
        self.descriptor
    }

    pub const fn format(self) -> L1Format {
        self.format
    }

    /// Size of the tile header: `(1 + DigestSize) * 16` bytes
    /// (`UNPACR_Regular.md:688`).
    pub const fn header_bytes(self) -> usize {
        (1 + self.descriptor.digest_size() as usize) * Self::ALIGNMENT
    }

    /// `XDim * YDim * ZDim * WDim` (`UNPACR_Regular.md:134`).
    pub const fn datum_count(self) -> usize {
        (self.descriptor.x_dim() as usize)
            * (self.descriptor.y_dim() as usize)
            * (self.descriptor.z_dim() as usize)
            * (self.descriptor.w_dim() as usize)
    }

    /// Bytes the shared-exponent section occupies, zero for non-BFP formats.
    ///
    /// One exponent per 16 datums, then rounded up to the 16-byte alignment:
    /// `ceil(ceil(NumElements / 16) / 16) * 16` (`UNPACR_Regular.md:132-136`).
    pub const fn exponent_section_bytes(self) -> usize {
        if !self.format.has_exponent_section() || self.descriptor.no_bfp_exp_section() {
            return 0;
        }
        let exponents = self
            .datum_count()
            .div_ceil(L1Format::DATUMS_PER_EXPONENT as usize);
        exponents.div_ceil(Self::ALIGNMENT) * Self::ALIGNMENT
    }

    /// Flat datum index of the logical coordinate, `X` fastest.
    ///
    /// `FirstDatum = ((W * ZDim + Z) * YDim + Y) * XDim + X` (`UNPACR_Regular.md:182`).
    /// This single line is what makes a tile a tile; everything else here is sizing
    /// around it.
    pub const fn datum_index(self, w: u32, z: u32, y: u32, x: u32) -> usize {
        let d = &self.descriptor;
        (((w as usize * d.z_dim() as usize) + z as usize) * d.y_dim() as usize + y as usize)
            * d.x_dim() as usize
            + x as usize
    }

    /// Byte offset of where the datum stream begins: past the header, past the
    /// exponent section.
    pub const fn datums_offset(self) -> usize {
        self.header_bytes() + self.exponent_section_bytes()
    }

    /// Bit offset of datum `index` from the start of the image.
    ///
    /// A bit offset rather than a byte one so that BFP4 and BFP2 are expressible:
    /// `InAddr_Datums += FirstDatum * DatumSizeBytes` with a fractional
    /// `DatumSizeBytes` (`UNPACR_Regular.md:204`).
    pub const fn datum_bit_offset(self, index: usize) -> usize {
        self.datums_offset() * 8 + index * self.format.datum_bits() as usize
    }

    /// Byte offset of the exponent shared by datum `index`, or `None` when the
    /// format has no exponent section.
    ///
    /// `InAddr_Exponents += FirstDatum / 16.` (`UNPACR_Regular.md:202`).
    pub const fn exponent_byte_offset(self, index: usize) -> Option<usize> {
        if self.exponent_section_bytes() == 0 {
            return None;
        }
        Some(self.header_bytes() + index / L1Format::DATUMS_PER_EXPONENT as usize)
    }

    /// Bytes the datum stream occupies, rounded up to [`TileImage::ALIGNMENT`].
    pub const fn datum_section_bytes(self) -> usize {
        let bits = self.datum_count() * self.format.datum_bits() as usize;
        bits.div_ceil(8).div_ceil(Self::ALIGNMENT) * Self::ALIGNMENT
    }

    /// Total size of one tile's image in L1.
    pub const fn total_bytes(self) -> usize {
        self.datums_offset() + self.datum_section_bytes()
    }
}

/// Decode one BFP8 datum to its BF16 bit pattern.
///
/// Transcribed verbatim from `FloatBitPatterns.md:119-131`, which gives the logic as
/// C. This is the hardware's own conversion, so it is the reference an encoder is
/// checked against rather than something to improve on.
///
/// Note the asymmetry the table records: `Sign == 1` with `Mag == 0` is **not**
/// negative zero, it is `-2^128` or `-Infinity` (`FloatBitPatterns.md:98`). A naive
/// round trip of `-0.0` through BFP8 therefore does not return `-0.0`.
///
/// **`UNVERIFIED`**: `FloatBitPatterns.md` exists only in the Wormhole tree, and the
/// Blackhole page that links to it does not exist.
pub const fn bfp8_to_bf16(datum_bits: u8, exp_bits: u8) -> u16 {
    let sign = (datum_bits >> 7) as u16;
    let mag = datum_bits << 1;
    if mag == 0 {
        if sign != 0 {
            0xff80
        } else {
            0
        }
    } else {
        let lz = mag.leading_zeros();
        let mag = mag << lz;
        // `ExpBits -= LZ` in the C, on a `uint8_t`, so it wraps.
        let exp = exp_bits.wrapping_sub(lz as u8);
        (sign << 15) | ((exp as u16) << 7) | ((mag & 0x7e) as u16)
    }
}

/// Decode one BFP4 datum to BF16 (`FloatBitPatterns.md:133-135`).
///
/// `datum_bits` carries the datum in its low four bits.
pub const fn bfp4_to_bf16(datum_bits: u8, exp_bits: u8) -> u16 {
    bfp8_to_bf16(datum_bits << 4, exp_bits)
}

/// Decode one BFP2 datum to BF16 (`FloatBitPatterns.md:137-139`).
///
/// `datum_bits` carries the datum in its low two bits.
pub const fn bfp2_to_bf16(datum_bits: u8, exp_bits: u8) -> u16 {
    bfp8_to_bf16(datum_bits << 6, exp_bits)
}

/// Decode one BFP8a datum to its FP16 bit pattern
/// (`FloatBitPatterns.md:143-156`).
///
/// `None` where the C calls `UndefinedBehavior()` — when the exponent, after the
/// normalising subtraction, has any of its top three bits set. Returning `None`
/// rather than a number is the point: the hardware's behaviour there is not
/// defined, so neither is ours.
pub const fn bfp8a_to_fp16(datum_bits: u8, exp_bits: u8) -> Option<u16> {
    let sign = (datum_bits >> 7) as u16;
    let mag = datum_bits << 1;
    if mag == 0 {
        return Some(if sign != 0 { 0xfc00 } else { 0 });
    }
    let lz = mag.leading_zeros();
    let mag = mag << lz;
    let exp = exp_bits.wrapping_sub(lz as u8);
    if exp & 0xe0 != 0 {
        return None;
    }
    Some((sign << 15) | ((exp as u16) << 10) | (((mag & 0x7e) as u16) << 3))
}

/// Decode one BFP4a datum to FP16 (`FloatBitPatterns.md:158-160`).
pub const fn bfp4a_to_fp16(datum_bits: u8, exp_bits: u8) -> Option<u16> {
    bfp8a_to_fp16(datum_bits << 4, exp_bits)
}

/// Decode one BFP2a datum to FP16 (`FloatBitPatterns.md:162-164`).
pub const fn bfp2a_to_fp16(datum_bits: u8, exp_bits: u8) -> Option<u16> {
    bfp8a_to_fp16(datum_bits << 6, exp_bits)
}

/// Widen a BF16 bit pattern to FP32. Always exact: BF16 *is* FP32 with the low 16
/// mantissa bits dropped (`FloatBitPatterns.md:49`).
pub const fn bf16_to_fp32(bits: u16) -> u32 {
    (bits as u32) << 16
}

/// Narrow FP32 to BF16 by truncation — drop the low 16 bits.
///
/// One of the two modes the packer offers ("Truncating", via `Read_raw`;
/// `Packers/FormatConversion.md:118-124`), so which one a caller wants is a real
/// choice and not a detail to hide.
pub const fn fp32_to_bf16_truncate(bits: u32) -> u16 {
    (bits >> 16) as u16
}

/// Narrow FP32 to BF16 with round-to-nearest, ties to even.
///
/// The packer's other documented mode. Rounding a NaN cannot carry it into the
/// exponent, because a NaN's exponent is already all ones and its mantissa stays
/// non-zero under this rounding unless every retained mantissa bit is zero — which
/// is the +/-Infinity pattern, and is what IEEE rounding of a NaN to a narrower
/// type would do anyway.
pub const fn fp32_to_bf16_round(bits: u32) -> u16 {
    let lsb = (bits >> 16) & 1;
    let bias = 0x7fff + lsb;
    ((bits.wrapping_add(bias)) >> 16) as u16
}

/// What an FP16 bit pattern means, and whether the coprocessor agrees with IEEE.
///
/// FP16 is the one format where `FloatBitPatterns.md:5` tells software outright to
/// pre- and post-process: "This is especially true for FP16 data." The coprocessor
/// has **no bit pattern meaning NaN**, and reads `Exp == 31` as an ordinary value of
/// exponent 16 where IEEE reads NaN or Infinity.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Fp16Reading {
    /// IEEE and the coprocessor agree. The FP32 bit pattern is exact.
    Agreed(u32),
    /// `Exp == 31`. IEEE reads NaN or Infinity; the coprocessor reads the finite
    /// value whose FP32 bit pattern this carries (`FloatBitPatterns.md:69-72`).
    NotNanOrInfinity(u32),
    /// `Exp == 0` with a non-zero mantissa: an IEEE denormal, and the coprocessor's
    /// reading depends on *which unit reads it* — `SFPLOADI` treats it as normal,
    /// `SFPLOAD` produces an FP32 denormal that later arithmetic flushes to zero,
    /// and the Matrix Unit reads it as zero (`FloatBitPatterns.md:73-77`).
    ///
    /// No single FP32 value is the right answer, so none is offered.
    ConsumerDependent,
}

/// Read an FP16 bit pattern the way the coprocessor does.
pub const fn fp16_to_fp32(bits: u16) -> Fp16Reading {
    let sign = (bits as u32 & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return Fp16Reading::Agreed(sign); // +/-0, agreed by everything
        }
        return Fp16Reading::ConsumerDependent;
    }

    // Exponents 1..=31 are all read as normal values by the coprocessor. For 1..=30
    // IEEE agrees; at 31 it does not.
    let fp32 = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    if exp == 31 {
        Fp16Reading::NotNanOrInfinity(fp32)
    } else {
        Fp16Reading::Agreed(fp32)
    }
}

/// Why an FP32 value has no FP16 encoding the coprocessor would read back the same
/// way.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Fp16EncodeError {
    /// The coprocessor has no bit pattern meaning NaN, so there is nowhere to put
    /// one (`FloatBitPatterns.md:67`).
    NoNanEncoding,
    /// `Exp == 31` is a finite value here, not Infinity, so encoding Infinity would
    /// silently become 2^16 or larger.
    NoInfinityEncoding,
    /// Too large for the format even using `Exp == 31`, whose largest value is
    /// `(2 - 2^-10) * 2^16`.
    Overflow,
    /// Would round to a denormal, whose meaning is consumer-dependent — see
    /// [`Fp16Reading::ConsumerDependent`]. Encoding it would produce a value that
    /// reads back differently depending on which unit reads it.
    WouldBeDenormal,
}

/// Narrow FP32 to the FP16 the coprocessor reads, rounding to nearest, ties to even.
///
/// Refuses rather than approximates wherever the coprocessor's reading would differ
/// from the host's. That is the whole point: a silent NaN-to-finite substitution is
/// exactly the bug `FloatBitPatterns.md:5` warns about.
pub const fn fp32_to_fp16(bits: u32) -> Result<u16, Fp16EncodeError> {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        return Err(if mant != 0 {
            Fp16EncodeError::NoNanEncoding
        } else {
            Fp16EncodeError::NoInfinityEncoding
        });
    }
    if exp == 0 {
        // FP32 zero or denormal. Zero is fine; an FP32 denormal is far below FP16's
        // denormal range and would round to zero, which loses the value silently.
        return if mant == 0 {
            Ok(sign)
        } else {
            Err(Fp16EncodeError::WouldBeDenormal)
        };
    }

    // Unbiased exponent. The coprocessor's FP16 range is 2^-14 .. 2^16 inclusive of
    // the exponent-31 row, which IEEE reserves.
    let unbiased = exp - 127;
    if unbiased < -14 {
        return Err(Fp16EncodeError::WouldBeDenormal);
    }
    if unbiased > 16 {
        return Err(Fp16EncodeError::Overflow);
    }

    // Round the 23-bit mantissa to 10 bits, nearest-even.
    let lsb = (mant >> 13) & 1;
    let rounded = mant + 0x0fff + lsb;
    // Rounding may carry into the exponent.
    let (unbiased, mant10) = if rounded & 0x0080_0000 != 0 {
        (unbiased + 1, 0)
    } else {
        (unbiased, (rounded >> 13) & 0x3ff)
    };
    if unbiased > 16 {
        return Err(Fp16EncodeError::Overflow);
    }
    Ok(sign | (((unbiased + 15) as u16) << 10) | mant10 as u16)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;

    /// The descriptor the rest of these tests use: one 16x16 face, FP32.
    fn face() -> TileDescriptor {
        TileDescriptor::zeroed()
            .with_x_dim(16)
            .with_y_dim(16)
            .with_is_uncompressed(true)
    }

    #[test]
    fn descriptor_fields_land_where_the_table_says() {
        // Bit positions from `UNPACR_Regular.md:672-688`. Each field is set alone to
        // all-ones, so a field written to the wrong offset shows up as a stray word.
        // Whole 4-word images are compared because `BlobsYStart` straddles words 2
        // and 3 -- a per-word expectation could not express it.
        let cases: &[(&str, TileDescriptor, [u32; 4])] = &[
            (
                "InDataFormat 0..4",
                TileDescriptor::zeroed().with_in_data_format_raw(u32::MAX),
                [0x0000_000f, 0, 0, 0],
            ),
            (
                "IsUncompressed 4",
                TileDescriptor::zeroed().with_is_uncompressed(true),
                [0x0000_0010, 0, 0, 0],
            ),
            (
                "NoBFPExpSection 5",
                TileDescriptor::zeroed().with_no_bfp_exp_section(true),
                [0x0000_0020, 0, 0, 0],
            ),
            (
                "BlobsPerXYPlane 8..11",
                TileDescriptor::zeroed().with_blobs_per_xy_plane(u32::MAX),
                [0x0000_0700, 0, 0, 0],
            ),
            (
                "XDim 16..32",
                TileDescriptor::zeroed().with_x_dim(u32::MAX),
                [0xffff_0000, 0, 0, 0],
            ),
            (
                "YDim 32..40",
                TileDescriptor::zeroed().with_y_dim(u32::MAX),
                [0, 0x0000_00ff, 0, 0],
            ),
            (
                "ZDim 48..56",
                TileDescriptor::zeroed().with_z_dim(u32::MAX),
                [0, 0x00ff_0000, 0, 0],
            ),
            (
                "WDim 64..72",
                TileDescriptor::zeroed().with_w_dim(u32::MAX),
                [0, 0, 0x0000_00ff, 0],
            ),
            (
                "BlobsYStart 80..112, straddling words 2 and 3",
                TileDescriptor::zeroed().with_blobs_y_start(u32::MAX),
                [0, 0, 0xffff_0000, 0x0000_ffff],
            ),
            (
                "DigestSize 120..128",
                TileDescriptor::zeroed().with_digest_size(u32::MAX),
                [0, 0, 0, 0xff00_0000],
            ),
        ];
        for (name, d, expected) in cases {
            assert_eq!(&d.words(), expected, "{name}");
        }

        // Each setter is the inverse of its getter, including across the straddle.
        let d = TileDescriptor::zeroed().with_blobs_y_start(0xdead_beef);
        assert_eq!(d.blobs_y_start(), 0xdead_beef, "straddling round trip");
        assert_eq!(TileDescriptor::zeroed().with_x_dim(1234).x_dim(), 1234);
        assert_eq!(
            TileDescriptor::zeroed().with_digest_size(7).digest_size(),
            7
        );
    }

    /// Every bit the table calls Reserved really is reserved by the accessors.
    ///
    /// The complement of the field test above: together they say the 128 bits are
    /// partitioned, with nothing claimed twice and nothing unaccounted for.
    #[test]
    fn reserved_runs_are_not_reachable_through_any_setter() {
        let full = TileDescriptor::zeroed()
            .with_in_data_format_raw(u32::MAX)
            .with_is_uncompressed(true)
            .with_no_bfp_exp_section(true)
            .with_blobs_per_xy_plane(u32::MAX)
            .with_x_dim(u32::MAX)
            .with_y_dim(u32::MAX)
            .with_z_dim(u32::MAX)
            .with_w_dim(u32::MAX)
            .with_blobs_y_start(u32::MAX)
            .with_digest_size(u32::MAX);
        assert!(
            full.reserved_bits_are_zero(),
            "a setter reached a reserved bit: {:08x?}",
            full.words()
        );

        // 4 + 1 + 1 + 3 + 16 + 8 + 8 + 8 + 32 + 8 = 89 documented bits.
        let set: u32 = full.words().iter().map(|w| w.count_ones()).sum();
        assert_eq!(set, 89, "the documented fields cover 89 of the 128 bits");
    }

    #[test]
    fn zero_z_and_w_mean_one_but_zero_y_does_not() {
        // `UNPACR_Regular.md:81-82` gives the rule for ZDim and WDim only.
        let d = face();
        assert_eq!(d.z_dim_raw(), 0);
        assert_eq!(d.z_dim(), 1);
        assert_eq!(d.w_dim_raw(), 0);
        assert_eq!(d.w_dim(), 1);

        let no_y = TileDescriptor::zeroed().with_x_dim(16);
        assert_eq!(
            TileImage::new(no_y, L1Format::Fp32),
            Err(TileImageError::ZeroDimension { name: "YDim" }),
            "YDim has no zero-means-one rule, so zero is an error rather than one row"
        );
    }

    #[test]
    fn the_datum_index_is_x_fastest_then_y_then_z_then_w() {
        // `FirstDatum = ((W * ZDim + Z) * YDim + Y) * XDim + X`.
        let d = face().with_z_dim(4).with_w_dim(2);
        let img = TileImage::new(d, L1Format::Fp32).unwrap();

        assert_eq!(img.datum_index(0, 0, 0, 0), 0);
        assert_eq!(img.datum_index(0, 0, 0, 1), 1, "X is fastest");
        assert_eq!(img.datum_index(0, 0, 1, 0), 16, "Y steps by XDim");
        assert_eq!(img.datum_index(0, 1, 0, 0), 256, "Z steps by XDim * YDim");
        assert_eq!(
            img.datum_index(1, 0, 0, 0),
            1024,
            "W steps by XDim*YDim*ZDim"
        );
        assert_eq!(img.datum_count(), 16 * 16 * 4 * 2);

        // The index is a bijection onto 0..datum_count.
        let mut seen = std::vec![false; img.datum_count()];
        for w in 0..2 {
            for z in 0..4 {
                for y in 0..16 {
                    for x in 0..16 {
                        let i = img.datum_index(w, z, y, x);
                        assert!(!seen[i], "index {i} produced twice");
                        seen[i] = true;
                    }
                }
            }
        }
        assert!(seen.iter().all(|s| *s), "some index was never produced");
    }

    #[test]
    fn the_header_is_sixteen_bytes_per_digest_step() {
        // "Tile header size is `(1 + DigestSize) * 16` bytes."
        for digest in [0u32, 1, 3, 255] {
            let img = TileImage::new(face().with_digest_size(digest), L1Format::Fp32).unwrap();
            assert_eq!(img.header_bytes(), (1 + digest as usize) * 16);
        }
    }

    #[test]
    fn non_bfp_formats_have_no_exponent_section() {
        for f in [
            L1Format::Fp32,
            L1Format::Bf16,
            L1Format::Fp16,
            L1Format::Int32,
        ] {
            let img = TileImage::new(face(), f).unwrap();
            assert_eq!(img.exponent_section_bytes(), 0, "{f:?}");
            assert_eq!(img.exponent_byte_offset(0), None, "{f:?}");
            assert_eq!(img.datums_offset(), img.header_bytes(), "{f:?}");
        }
    }

    /// The exponent section is sized and placed for a format nothing implements yet.
    ///
    /// This is the BFP-readiness claim made checkable: the scope of this phase is
    /// FP32/BF16/FP16, but if the *shape* of the image were wrong for BFP the model
    /// would need redesigning rather than extending.
    #[test]
    fn a_bfp8_tile_reserves_one_exponent_byte_per_sixteen_datums() {
        let d = face().with_z_dim(4); // 1024 datums
        let img = TileImage::new(d, L1Format::Bfp8).unwrap();

        assert_eq!(img.datum_count(), 1024);
        // 1024 / 16 = 64 exponents, already a multiple of 16 bytes.
        assert_eq!(img.exponent_section_bytes(), 64);
        assert_eq!(img.datums_offset(), img.header_bytes() + 64);

        // Exponents sit between the header and the datums, one per 16 datums.
        assert_eq!(img.exponent_byte_offset(0), Some(img.header_bytes()));
        assert_eq!(img.exponent_byte_offset(15), Some(img.header_bytes()));
        assert_eq!(img.exponent_byte_offset(16), Some(img.header_bytes() + 1));
        assert_eq!(
            img.exponent_byte_offset(1023),
            Some(img.header_bytes() + 63)
        );

        // `NoBFPExpSection` removes it.
        let none = TileImage::new(d.with_no_bfp_exp_section(true), L1Format::Bfp8).unwrap();
        assert_eq!(none.exponent_section_bytes(), 0);

        // Rounding up: 17 datums need 2 exponents, padded to the 16-byte alignment.
        let small = TileImage::new(
            TileDescriptor::zeroed().with_x_dim(17).with_y_dim(1),
            L1Format::Bfp8,
        )
        .unwrap();
        assert_eq!(small.datum_count(), 17);
        assert_eq!(small.exponent_section_bytes(), 16);
    }

    /// Sub-byte datums are addressed in bits, so BFP4 and BFP2 need no new API.
    #[test]
    fn sub_byte_formats_pack_several_datums_to_the_byte() {
        let d = TileDescriptor::zeroed().with_x_dim(16).with_y_dim(1);

        let bfp4 = TileImage::new(d, L1Format::Bfp4).unwrap();
        assert_eq!(bfp4.format().datum_bits(), 4);
        let base = bfp4.datums_offset() * 8;
        assert_eq!(bfp4.datum_bit_offset(0), base);
        assert_eq!(bfp4.datum_bit_offset(1), base + 4, "two datums per byte");
        assert_eq!(bfp4.datum_bit_offset(2), base + 8);

        let bfp2 = TileImage::new(d, L1Format::Bfp2).unwrap();
        assert_eq!(bfp2.datum_bit_offset(4) - bfp2.datum_bit_offset(0), 8);

        // A row that does not fill whole bytes is refused rather than guessed at.
        let ragged = TileDescriptor::zeroed().with_x_dim(3).with_y_dim(1);
        assert_eq!(
            TileImage::new(ragged, L1Format::Bfp2),
            Err(TileImageError::RaggedSubByteRow {
                x_dim: 3,
                datum_bits: 2
            })
        );
    }

    #[test]
    fn a_row_wider_than_the_address_generator_can_reach_is_refused() {
        // "X reaches the input address generator as a 13-bit value, which then gets
        // scaled into a nibble offset held in a 14-bit field, i.e. 8192 bytes of
        // datums" -- so the datum limit falls as the datum grows.
        let too_wide = TileDescriptor::zeroed().with_x_dim(2049).with_y_dim(1);
        assert_eq!(
            TileImage::new(too_wide, L1Format::Fp32),
            Err(TileImageError::RowTooWide {
                x_dim: 2049,
                limit: 2048
            })
        );
        assert!(TileImage::new(
            TileDescriptor::zeroed().with_x_dim(2048).with_y_dim(1),
            L1Format::Fp32
        )
        .is_ok());
        // Same XDim is fine at 16 bits per datum.
        assert!(TileImage::new(too_wide, L1Format::Bf16).is_ok());
    }

    /// `bfp8_to_bf16` against the C in `FloatBitPatterns.md:119-131`, for every
    /// input there is.
    ///
    /// 65536 cases is nothing, and it means the transcription is not spot-checked
    /// but proven equal to the document over its whole domain.
    #[test]
    fn bfp8_to_bf16_matches_the_documented_routine_exhaustively() {
        /// The specification's C, transliterated as literally as Rust allows and
        /// deliberately *not* sharing code with the implementation.
        fn reference(datum_bits: u8, exp_bits: u8) -> u16 {
            let sign = datum_bits >> 7;
            let mag = datum_bits.wrapping_shl(1);
            if mag == 0 {
                if sign != 0 {
                    0xff80
                } else {
                    0
                }
            } else {
                let lz = mag.leading_zeros();
                let mag = mag.wrapping_shl(lz);
                let exp = exp_bits.wrapping_sub(lz as u8);
                ((sign as u16) << 15) | ((exp as u16) << 7) | ((mag & 0x7e) as u16)
            }
        }

        for datum in 0u16..=255 {
            for exp in 0u16..=255 {
                let (d, e) = (datum as u8, exp as u8);
                assert_eq!(
                    bfp8_to_bf16(d, e),
                    reference(d, e),
                    "datum {d:#04x}, exponent {e:#04x}"
                );
            }
        }
    }

    /// The trap the table records: sign-with-zero-magnitude is not negative zero.
    #[test]
    fn a_negative_zero_magnitude_bfp8_datum_is_not_negative_zero() {
        // `FloatBitPatterns.md:98`: Sign 1, Mag 0 means -2^128 or -Infinity.
        assert_eq!(bfp8_to_bf16(0x00, 0x7f), 0x0000, "+0 is +0");
        let negative = bfp8_to_bf16(0x80, 0x7f);
        assert_eq!(negative, 0xff80, "-2^128 / -Infinity, not -0 (0x8000)");
        assert_ne!(
            negative, 0x8000,
            "a round trip of -0.0 does not survive BFP8"
        );

        // The a-variants use the FP16 bias and say -2^16 instead.
        assert_eq!(bfp8a_to_fp16(0x80, 0x0f), Some(0xfc00));
        assert_eq!(bfp8a_to_fp16(0x00, 0x0f), Some(0x0000));
    }

    /// The `UndefinedBehavior()` arm of `BFP8aToFP16` becomes `None`, not a number.
    #[test]
    fn bfp8a_refuses_the_undefined_exponent_range() {
        // Magnitude 0x40 -> mag = 0x80, no leading zeros, exponent unchanged.
        assert_eq!(bfp8a_to_fp16(0x40, 0x1f), Some(0x7c00));
        // An exponent with a top-three bit set after normalisation is undefined.
        assert_eq!(bfp8a_to_fp16(0x40, 0x20), None);
        assert_eq!(bfp8a_to_fp16(0x40, 0xff), None);
        // The 4- and 2-bit variants inherit it.
        assert_eq!(bfp4a_to_fp16(0x4, 0x20), None);
        assert_eq!(bfp2a_to_fp16(0x1, 0x20), None);
    }

    #[test]
    fn bfp4_and_bfp2_are_bfp8_with_the_datum_shifted_up() {
        for exp in [0u8, 0x3f, 0x7f, 0xff] {
            for d in 0u8..16 {
                assert_eq!(bfp4_to_bf16(d, exp), bfp8_to_bf16(d << 4, exp));
            }
            for d in 0u8..4 {
                assert_eq!(bfp2_to_bf16(d, exp), bfp8_to_bf16(d << 6, exp));
            }
        }
    }

    #[test]
    fn bf16_widens_exactly_and_truncation_is_the_inverse() {
        for bits in [0u16, 0x3f80, 0xbf80, 0x7f80, 0xffc0, 0x0001] {
            assert_eq!(fp32_to_bf16_truncate(bf16_to_fp32(bits)), bits);
        }
        assert_eq!(bf16_to_fp32(0x3f80), 1.0f32.to_bits());
        assert_eq!(bf16_to_fp32(0xc000), (-2.0f32).to_bits());
    }

    #[test]
    fn bf16_rounding_and_truncation_differ_where_the_documentation_says_they_may() {
        // Exactly representable: both modes agree.
        assert_eq!(fp32_to_bf16_round(1.0f32.to_bits()), 0x3f80);
        assert_eq!(fp32_to_bf16_truncate(1.0f32.to_bits()), 0x3f80);

        // Just above 1.0, with the dropped bits above half: rounding goes up,
        // truncation does not. This is why the packer exposes both.
        let just_above = f32::from_bits(0x3f80_c000); // 1.0 + 3/512
        assert_eq!(fp32_to_bf16_truncate(just_above.to_bits()), 0x3f80);
        assert_eq!(fp32_to_bf16_round(just_above.to_bits()), 0x3f81);

        // A tie rounds to even.
        assert_eq!(fp32_to_bf16_round(0x3f80_8000), 0x3f80, "tie down to even");
        assert_eq!(fp32_to_bf16_round(0x3f81_8000), 0x3f82, "tie up to even");
    }

    /// The headline FP16 divergence, as a named test rather than a comment.
    #[test]
    fn fp16_exponent_31_is_a_finite_value_not_nan_or_infinity() {
        // IEEE: +Infinity. Coprocessor: (1 + 0) * 2^16 = 65536.
        match fp16_to_fp32(0x7c00) {
            Fp16Reading::NotNanOrInfinity(bits) => {
                assert_eq!(f32::from_bits(bits), 65536.0);
            }
            other => panic!("expected a finite reading, got {other:?}"),
        }
        // IEEE: +NaN. Coprocessor: a finite number just above 65536.
        match fp16_to_fp32(0x7c01) {
            Fp16Reading::NotNanOrInfinity(bits) => {
                let v = f32::from_bits(bits);
                assert!(v.is_finite(), "{v} should be finite on device");
                assert!(v > 65536.0);
            }
            other => panic!("expected a finite reading, got {other:?}"),
        }

        // Ordinary exponents agree with IEEE exactly.
        for (bits, expected) in [(0x3c00u16, 1.0f32), (0xc000, -2.0), (0x0000, 0.0)] {
            match fp16_to_fp32(bits) {
                Fp16Reading::Agreed(b) => assert_eq!(f32::from_bits(b), expected),
                other => panic!("{bits:#06x}: expected agreement, got {other:?}"),
            }
        }

        // A denormal has no single right answer.
        assert_eq!(fp16_to_fp32(0x0001), Fp16Reading::ConsumerDependent);
        assert_eq!(fp16_to_fp32(0x8200), Fp16Reading::ConsumerDependent);
    }

    #[test]
    fn encoding_to_fp16_refuses_what_the_device_cannot_mean() {
        assert_eq!(
            fp32_to_fp16(f32::NAN.to_bits()),
            Err(Fp16EncodeError::NoNanEncoding)
        );
        assert_eq!(
            fp32_to_fp16(f32::INFINITY.to_bits()),
            Err(Fp16EncodeError::NoInfinityEncoding)
        );
        assert_eq!(
            fp32_to_fp16(f32::NEG_INFINITY.to_bits()),
            Err(Fp16EncodeError::NoInfinityEncoding)
        );
        // 2^17 is past the top of the range even using exponent 31.
        assert_eq!(
            fp32_to_fp16(131072.0f32.to_bits()),
            Err(Fp16EncodeError::Overflow)
        );
        // Below the smallest normal.
        assert_eq!(
            fp32_to_fp16(1e-8f32.to_bits()),
            Err(Fp16EncodeError::WouldBeDenormal)
        );

        assert_eq!(fp32_to_fp16(1.0f32.to_bits()), Ok(0x3c00));
        assert_eq!(fp32_to_fp16((-2.0f32).to_bits()), Ok(0xc000));
        assert_eq!(fp32_to_fp16(0.0f32.to_bits()), Ok(0x0000));
        assert_eq!(fp32_to_fp16((-0.0f32).to_bits()), Ok(0x8000));
        // The extended range the coprocessor reads at exponent 31 is reachable.
        assert_eq!(fp32_to_fp16(65536.0f32.to_bits()), Ok(0x7c00));
    }

    /// Every value that round-trips does so exactly, over the whole FP16 domain.
    #[test]
    fn fp16_round_trips_exactly_wherever_it_round_trips_at_all() {
        for bits in 0u32..=0xffff {
            let bits = bits as u16;
            let widened = match fp16_to_fp32(bits) {
                Fp16Reading::Agreed(b) | Fp16Reading::NotNanOrInfinity(b) => b,
                Fp16Reading::ConsumerDependent => continue,
            };
            assert_eq!(
                fp32_to_fp16(widened),
                Ok(bits),
                "{bits:#06x} did not survive the round trip"
            );
        }
    }
}

#[cfg(test)]
mod format_code_tests {
    use super::*;

    /// Pins the measured mapping, so a re-run of the probe that disagrees shows up
    /// here rather than as wrong data three phases later.
    #[test]
    fn the_measured_format_codes_are_what_the_probe_found() {
        assert_eq!(L1Format::Fp32.code(), Some(0));
        assert_eq!(L1Format::Tf32.code(), Some(4));
        assert_eq!(L1Format::Int32.code(), Some(8));
    }

    /// Every code this crate claims must round-trip, and every format it does not
    /// claim must stay unclaimed -- otherwise a half-finished addition reads as a
    /// measurement.
    #[test]
    fn codes_round_trip_and_nothing_else_is_claimed() {
        let claimed = [L1Format::Fp32, L1Format::Tf32, L1Format::Int32];
        for f in claimed {
            let code = f.code().expect("claimed formats have a code");
            assert_eq!(L1Format::from_code(code), Some(f));
        }
        for f in [
            L1Format::Bf16,
            L1Format::Fp16,
            L1Format::Fp8,
            L1Format::Bfp8,
            L1Format::Bfp8a,
            L1Format::Bfp4,
            L1Format::Bfp4a,
            L1Format::Bfp2,
            L1Format::Bfp2a,
            L1Format::Int16,
            L1Format::Int8,
            L1Format::Uint8,
        ] {
            assert_eq!(
                f.code(),
                None,
                "{f:?} has no measured code; ttsim declines UnpackToDst for the \
                 16-bit and block-float formats, so this path cannot establish one"
            );
        }
    }

    /// The independent cross-check: `Packers/InputAddressGenerator.md` switches on
    /// `In_data_format & 3`, where zero means four bytes per datum. Every code this
    /// crate claims must agree with the datum width the format already reports.
    #[test]
    fn every_measured_code_agrees_with_the_documented_size_rule() {
        for f in [L1Format::Fp32, L1Format::Tf32, L1Format::Int32] {
            let code = f.code().unwrap();
            assert_eq!(
                code & 3,
                0,
                "{f:?} has code {code}, whose low two bits say it is not four bytes"
            );
            assert_eq!(f.datum_bits(), 32, "{f:?} should be a 32-bit datum");
        }
    }
}
