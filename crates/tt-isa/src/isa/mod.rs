//! The Tensix instruction set, generated from the specification.
//!
//! [`generated`] is produced by `cargo xtask gen-isa` from `Diagrams/Src/Bits32.lua`
//! at the revision `PINS.toml` pins, cross-checked against the `TT_*(…)` syntax
//! block on the page that embeds each diagram. It is **generated, never
//! transcribed**: 145 encodings and 814 bit fields, and hand-copying them is a
//! guaranteed bug source. The five SFPU encoders that predate it were hand-written,
//! and the tests that pinned them are what now check the generator.
//!
//! # Two layers, and why the line falls where it does
//!
//! What is here is the **raw** layer: it encodes what the bits allow. A field is
//! refused only when the value does not fit the field.
//!
//! What the hardware *honours* is a separate question, and it lives in the
//! hand-written wrappers next door in [`crate::sfpu`]. `SFPLOADI` can encode any
//! four-bit `VD`, but `LReg[8]` upwards are constants and a write to one is
//! silently discarded — so `sfpu::loadi` refuses it. `SFPMUL` can encode any `VC`,
//! but the specification says not to use it unless `VC == 9` — so `sfpu::mul` does
//! not offer the parameter.
//!
//! Keeping that split explicit is the point. Semantic rules belong where they can
//! cite a reason; the generated layer must stay a faithful description of the
//! encoding, or regenerating it after a specification bump becomes a merge.
//!
//! # Provenance
//!
//! The Blackhole documentation is a delta over Wormhole's, and `crate`'s module doc
//! states the consequence: a fact sourced from a Wormhole page is a hypothesis. For
//! the instruction set that is 79 of 145 encodings — too many to mark by hand, and
//! the specification already knows the answer. Every [`InstructionDef`] carries its
//! [`Provenance`], derived from which tree embeds its diagram. See that type.

pub mod generated;

/// One bit field of an instruction.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Field {
    name: &'static str,
    first_bit: u8,
    width: u8,
    signed: bool,
}

impl Field {
    /// Called by generated code only.
    #[doc(hidden)]
    pub const fn new(name: &'static str, first_bit: u8, width: u8, signed: bool) -> Self {
        Field {
            name,
            first_bit,
            width,
            signed,
        }
    }

    pub const fn name(self) -> &'static str {
        self.name
    }

    pub const fn first_bit(self) -> u8 {
        self.first_bit
    }

    pub const fn width(self) -> u8 {
        self.width
    }

    /// Is this field a two's complement immediate?
    ///
    /// Three are: `SFPIADD`, `SFPSHFT` and `SFPSHFT2b` all carry `Imm12 (signed)`.
    /// The diagrams annotate the label and the syntax blocks type it `/* i12 */`,
    /// and the two sources agree exactly.
    pub const fn signed(self) -> bool {
        self.signed
    }

    /// Bits this field occupies.
    pub const fn mask(self) -> u32 {
        if self.width >= 32 {
            u32::MAX
        } else {
            ((1u32 << self.width) - 1) << self.first_bit
        }
    }

    /// Largest value the field can hold, treated as unsigned.
    pub const fn max_value(self) -> u32 {
        if self.width >= 32 {
            u32::MAX
        } else {
            (1u32 << self.width) - 1
        }
    }

    pub const fn fits(self, value: u32) -> bool {
        value <= self.max_value()
    }

    /// Place a value into its bits, discarding anything that does not fit.
    pub const fn place(self, value: u32) -> u32 {
        (value & self.max_value()) << self.first_bit
    }

    /// Read this field out of an instruction word.
    pub const fn extract(self, word: u32) -> u32 {
        (word >> self.first_bit) & self.max_value()
    }
}

/// Which chip an encoding is evidence for.
///
/// Derived from which documentation tree embeds the diagram, which is a fact rather
/// than a naming convention — classifying by a `_BH` suffix would be a guess.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Provenance {
    /// A Blackhole page embeds this diagram. Authoritative.
    Blackhole,
    /// Only a Wormhole page embeds it, but the Blackhole tree carries the same page
    /// stating the document is shared and the behaviour identical. Documented to
    /// apply here, rather than assumed to.
    SharedWithWormhole,
    /// Only a Wormhole page embeds it, and the Blackhole page of the same name
    /// embeds a *different* diagram. **This layout is Wormhole's.** Such encodings
    /// live under `generated::encode::wormhole`, so reaching one on Blackhole has to
    /// be deliberate.
    SupersededOnBlackhole { by: &'static str },
    /// Only a Wormhole page embeds it, and Blackhole has no such page at all.
    /// **`UNVERIFIED`**: a hypothesis until silicon or the simulator says otherwise.
    WormholeOnly,
}

impl Provenance {
    /// Does this encoding describe Blackhole on the documentation's own authority?
    pub const fn is_documented_for_blackhole(self) -> bool {
        matches!(self, Provenance::Blackhole | Provenance::SharedWithWormhole)
    }
}

/// One Tensix instruction encoding.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct InstructionDef {
    key: &'static str,
    mnemonic: &'static str,
    opcode: u8,
    fields: &'static [Field],
    fixed: &'static [(Field, u32)],
    unspecified: u32,
    provenance: Provenance,
    page: &'static str,
}

impl InstructionDef {
    /// Called by generated code only.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        key: &'static str,
        mnemonic: &'static str,
        opcode: u8,
        fields: &'static [Field],
        fixed: &'static [(Field, u32)],
        unspecified: u32,
        provenance: Provenance,
        page: &'static str,
    ) -> Self {
        InstructionDef {
            key,
            mnemonic,
            opcode,
            fields,
            fixed,
            unspecified,
            provenance,
            page,
        }
    }

    /// The `Bits32.lua` diagram key, e.g. `SFPSTORE_BH`.
    ///
    /// Not the mnemonic: mode variants such as `UNPACR_Regular` and
    /// `UNPACR_FlushCache` are separate encodings of one mnemonic.
    pub const fn key(self) -> &'static str {
        self.key
    }

    /// The `TT_*` macro name the specification documents, without its prefix.
    pub const fn mnemonic(self) -> &'static str {
        self.mnemonic
    }

    pub const fn opcode(self) -> u8 {
        self.opcode
    }

    /// Operand fields, from the high bits of the word down — the order the
    /// specification's syntax block lists them in.
    pub const fn fields(self) -> &'static [Field] {
        self.fields
    }

    /// Fields that must carry a particular value.
    ///
    /// Not only the opcode. The six `UNPACR_NOP_*` encodings share opcode `0x43`
    /// and distinguish themselves by a sub-opcode in bits 0..3, and several
    /// instructions have bits documented as must-be-zero.
    pub const fn fixed(self) -> &'static [(Field, u32)] {
        self.fixed
    }

    /// Bits no field claims and nothing documents.
    ///
    /// The diagrams label only the bits that carry meaning, so most encodings leave
    /// gaps. They are recorded rather than assumed to be zero: the encoders write
    /// zeros there, but an encoding whose behaviour turns out to depend on one of
    /// these bits has somewhere for that finding to land.
    pub const fn unspecified(self) -> u32 {
        self.unspecified
    }

    pub const fn provenance(self) -> Provenance {
        self.provenance
    }

    /// The specification page this came from, relative to the documentation root.
    pub const fn page(self) -> &'static str {
        self.page
    }

    /// The instruction word with every fixed field set and every operand zero.
    pub const fn skeleton(self) -> u32 {
        let mut word = (self.opcode as u32) << 24;
        let mut i = 0;
        while i < self.fixed.len() {
            let (field, value) = self.fixed[i];
            word |= field.place(value);
            i += 1;
        }
        word
    }

    /// Does this word decode as this instruction?
    ///
    /// Opcode alone is not enough: opcodes are shared between mode variants, and
    /// the fixed fields are what tell them apart.
    pub const fn matches(self, word: u32) -> bool {
        if (word >> 24) as u8 != self.opcode {
            return false;
        }
        let mut i = 0;
        while i < self.fixed.len() {
            let (field, value) = self.fixed[i];
            if field.extract(word) != value {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Look a field up by name. Linear, but these lists are at most 13 long.
    pub fn field(self, name: &str) -> Option<Field> {
        self.fields.iter().copied().find(|f| f.name == name)
    }
}

/// The bit layout of one datum type in `Src` or `Dst`.
///
/// These are not instructions and have no opcode. They are what Phase 4 needs to do
/// FP32/BF16/FP16 conversion against the documented layouts rather than against
/// IEEE 754 assumptions, which the coprocessor does not entirely follow.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DatumLayout {
    key: &'static str,
    nbits: u8,
    fields: &'static [Field],
    page: &'static str,
}

impl DatumLayout {
    /// Called by generated code only.
    #[doc(hidden)]
    pub const fn new(
        key: &'static str,
        nbits: u8,
        fields: &'static [Field],
        page: &'static str,
    ) -> Self {
        DatumLayout {
            key,
            nbits,
            fields,
            page,
        }
    }

    pub const fn key(self) -> &'static str {
        self.key
    }

    /// Width of the datum: 19 bits in `SrcA`/`SrcB`, 16 or 32 in `Dst`.
    pub const fn nbits(self) -> u8 {
        self.nbits
    }

    pub const fn fields(self) -> &'static [Field] {
        self.fields
    }

    pub const fn page(self) -> &'static str {
        self.page
    }

    pub fn field(self, name: &str) -> Option<Field> {
        self.fields.iter().copied().find(|f| f.name == name)
    }
}

/// Why an instruction could not be encoded.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EncodeError {
    /// A value too large for the field it was given to.
    FieldTooLarge {
        instruction: &'static str,
        field: &'static str,
        value: u32,
        width: u8,
    },
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EncodeError::FieldTooLarge {
                instruction,
                field,
                value,
                width,
            } => write!(
                f,
                "{instruction}: {field} = {value} does not fit in {width} bits"
            ),
        }
    }
}

/// An encoded Tensix instruction, and what it is.
///
/// Carrying the definition rather than a small `enum` of kinds means every
/// instruction has an identity, not just the handful something happened to need.
/// That is what lets a hazard list name instructions instead of enumerating cases.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Instruction {
    word: u32,
    def: &'static InstructionDef,
}

impl Instruction {
    /// Called by generated code only.
    #[doc(hidden)]
    pub const fn new(word: u32, def: &'static InstructionDef) -> Self {
        Instruction { word, def }
    }

    /// The raw instruction word, as pushed by a `sw` to `INSTRN_BUF_BASE`.
    pub const fn word(self) -> u32 {
        self.word
    }

    pub const fn def(self) -> &'static InstructionDef {
        self.def
    }

    /// Read one of this instruction's operands back out.
    pub fn operand(self, name: &str) -> Option<u32> {
        self.def.field(name).map(|f| f.extract(self.word))
    }

    /// The same instruction encoded as the operand of a `.ttinsn` pseudo-instruction.
    ///
    /// `.ttinsn IMM32` is `IMM32` rotated left by two bits
    /// (`PushTensixInstruction.md:21`); the core rotates it back and stores the
    /// result to `INSTRN_BUF_BASE`, which is what a `sw` there does. The point of
    /// the encoding is that adjacent `.ttinsn`s sit in the instruction stream, where
    /// the T0/T1/T2 instruction caches can fuse up to four of them into one push.
    ///
    /// `None` when the instruction cannot be expressed this way: the form is only
    /// valid for `IMM32 < 0xC000_0000`, because after rotation the low two bits must
    /// not be `0b11` — that is the space uncompressed RISC-V instructions occupy,
    /// and `.ttinsn` lives in the compressed encoding space these cores do not
    /// implement.
    pub const fn ttinsn_word(self) -> Option<u32> {
        if self.word < 0xC000_0000 {
            Some(self.word.rotate_left(2))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::generated::{datum, defs, ALL, ALL_LAYOUTS};
    use super::*;
    extern crate std;
    use std::collections::BTreeMap;
    use std::vec::Vec;

    #[test]
    fn the_table_has_the_expected_shape() {
        assert_eq!(
            ALL.len(),
            148,
            "instruction encodings: 145 diagrams, of which RMWCIB is four"
        );
        assert_eq!(ALL_LAYOUTS.len(), 19, "Src/Dst/NoC datum layouts");
    }

    /// The provenance split is a measurement of the specification, so pinning it
    /// means the next re-sync has to be looked at rather than absorbed.
    #[test]
    fn provenance_is_where_the_documentation_puts_it() {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for def in ALL {
            let name = match def.provenance() {
                Provenance::Blackhole => "Blackhole",
                Provenance::SharedWithWormhole => "SharedWithWormhole",
                Provenance::SupersededOnBlackhole { .. } => "SupersededOnBlackhole",
                Provenance::WormholeOnly => "WormholeOnly",
            };
            *counts.entry(name).or_default() += 1;
        }
        assert_eq!(counts["Blackhole"], 39);
        assert_eq!(counts["SharedWithWormhole"], 24);
        assert_eq!(counts["SupersededOnBlackhole"], 11);
        assert_eq!(
            counts["WormholeOnly"], 74,
            "half the instruction set is a hypothesis; that is the point of recording it"
        );
    }

    #[test]
    fn every_superseded_encoding_names_a_replacement_that_exists() {
        for def in ALL {
            if let Provenance::SupersededOnBlackhole { by } = def.provenance() {
                assert!(
                    ALL.iter().any(|d| d.key() == by),
                    "{} is superseded by {by}, which is not in the table",
                    def.key()
                );
            }
        }
    }

    #[test]
    fn fields_stay_inside_their_word_and_do_not_overlap() {
        for def in ALL {
            let opcode_byte = 0xFF00_0000u32;
            let mut claimed = 0u32;
            let all_fields = def
                .fields()
                .iter()
                .copied()
                .chain(def.fixed().iter().map(|(f, _)| *f));
            for f in all_fields {
                assert!(
                    f.first_bit() as u32 + f.width() as u32 <= 32,
                    "{}: {} runs past the word",
                    def.key(),
                    f.name()
                );
                assert_eq!(
                    claimed & f.mask(),
                    0,
                    "{}: {} overlaps another field",
                    def.key(),
                    f.name()
                );
                claimed |= f.mask();
            }
            assert_eq!(
                claimed & opcode_byte,
                0,
                "{}: a field encroaches on the opcode byte",
                def.key()
            );
            assert_eq!(
                claimed & def.unspecified(),
                0,
                "{}: a field claims bits also recorded as unspecified",
                def.key()
            );
        }
    }

    #[test]
    fn fixed_values_fit_their_fields_and_the_skeleton_carries_them() {
        for def in ALL {
            let skeleton = def.skeleton();
            assert_eq!(
                (skeleton >> 24) as u8,
                def.opcode(),
                "{}: skeleton loses the opcode",
                def.key()
            );
            for (f, value) in def.fixed() {
                assert!(f.fits(*value), "{}: fixed value does not fit", def.key());
                assert_eq!(
                    f.extract(skeleton),
                    *value,
                    "{}: skeleton loses a fixed field",
                    def.key()
                );
            }
            assert!(
                def.matches(skeleton),
                "{}: its own skeleton does not decode as it",
                def.key()
            );
        }
    }

    /// Opcodes are shared between mode variants — `0x43` by six `UNPACR_NOP_*`
    /// forms alone — so what tells them apart is their fixed fields. Phases 5 and 6
    /// depend on that being true rather than assumed.
    #[test]
    fn instructions_sharing_an_opcode_are_distinguishable() {
        let mut ambiguous: Vec<(&str, &str)> = Vec::new();
        let mut by_opcode: BTreeMap<u8, Vec<&InstructionDef>> = BTreeMap::new();
        for def in ALL {
            // A Wormhole form and the Blackhole form that replaces it are the same
            // instruction on different chips, not two instructions to tell apart.
            if matches!(def.provenance(), Provenance::SupersededOnBlackhole { .. }) {
                continue;
            }
            by_opcode.entry(def.opcode()).or_default().push(def);
        }
        for (opcode, group) in by_opcode {
            for (i, a) in group.iter().enumerate() {
                for b in &group[i + 1..] {
                    if a.matches(b.skeleton()) && b.matches(a.skeleton()) {
                        ambiguous.push((a.key(), b.key()));
                    }
                    let _ = opcode;
                }
            }
        }
        ambiguous.sort_unstable();
        // Two pairs are genuinely two views of one instruction rather than two
        // instructions: the `i` form adds an immediate, selected by a bit the base
        // diagram leaves undrawn and the base form's macro passes as a literal
        // zero. Every other shared opcode -- the six `UNPACR_NOP_*` forms, the six
        // `*DMAREG`/`*DMAREGi` pairs, `STOREIND`'s three, `SETDMAREG`'s two,
        // `REG2FLOP`'s two, `UNPACR`'s three -- is told apart by its fixed fields,
        // which is what Phases 5 and 6 rely on.
        assert_eq!(
            ambiguous,
            [
                ("SFPSHFT2", "SFPSHFT2b"),
                ("SFPSTOCHRND_BH", "SFPSTOCHRNDi_BH"),
            ]
        );
    }

    /// Encode, then read the operands back out. Catches a generator that placed a
    /// field at the wrong offset or emitted its arguments in the wrong order.
    #[test]
    fn operands_survive_a_round_trip_through_the_word() {
        for def in ALL {
            let mut word = def.skeleton();
            // A distinctive value per field, so a transposition cannot pass.
            for (i, f) in def.fields().iter().enumerate() {
                let value = ((i as u32 * 7 + 3) ^ 0x2B) & f.max_value();
                word |= f.place(value);
            }
            for (i, f) in def.fields().iter().enumerate() {
                let value = ((i as u32 * 7 + 3) ^ 0x2B) & f.max_value();
                assert_eq!(
                    f.extract(word),
                    value,
                    "{}: {} does not survive a round trip",
                    def.key(),
                    f.name()
                );
            }
            assert!(def.matches(word), "{}: operands broke the match", def.key());
        }
    }

    /// The single most consequential Blackhole/Wormhole encoding difference. It used
    /// to be protected by one careful hand-written test; now the two encodings are
    /// separate values and the wrong one is not reachable by the plain name.
    #[test]
    fn blackhole_moved_sfpstores_addrmod_and_both_forms_are_recorded() {
        let bh = defs::SFPSTORE.field("AddrMod").unwrap();
        assert_eq!((bh.first_bit(), bh.width()), (13, 3), "Blackhole");

        let wh = defs::wormhole::SFPSTORE.field("AddrMod").unwrap();
        assert_eq!((wh.first_bit(), wh.width()), (14, 2), "Wormhole");

        assert_eq!(
            defs::wormhole::SFPSTORE.provenance(),
            Provenance::SupersededOnBlackhole { by: "SFPSTORE_BH" }
        );
        // Encoding an odd AddrMod is where the two genuinely differ.
        assert_ne!(bh.place(0b101), wh.place(0b101));
    }

    /// `RMWCIB`'s opcode is written `0xB3 + Index1`, and the `+` is arithmetic: the
    /// four opcodes are 0xB3 through 0xB6. Reading the digit as *bits* of the
    /// opcode byte would give 0xB0..0xB3, and 0xB0, 0xB1 and 0xB2 belong to three
    /// other instructions — so that reading would have silently aliased them.
    #[test]
    fn rmwcibs_four_encodings_are_consecutive_and_collide_with_nothing() {
        let rmwcib: Vec<&InstructionDef> = ALL
            .iter()
            .copied()
            .filter(|d| d.key().starts_with("RMWCIB"))
            .collect();
        let opcodes: Vec<u8> = rmwcib.iter().map(|d| d.opcode()).collect();
        assert_eq!(opcodes, [0xB3, 0xB4, 0xB5, 0xB6]);

        for def in ALL {
            if def.key().starts_with("RMWCIB") {
                continue;
            }
            assert!(
                !opcodes.contains(&def.opcode()),
                "{} at {:#04x} collides with an RMWCIB encoding",
                def.key(),
                def.opcode()
            );
        }
    }

    #[test]
    fn signed_immediates_are_exactly_the_three_the_documentation_marks() {
        let mut signed = Vec::new();
        for def in ALL {
            for f in def.fields() {
                if f.signed() {
                    signed.push((def.key(), f.name()));
                }
            }
        }
        signed.sort_unstable();
        assert_eq!(
            signed,
            [
                ("SFPIADD", "Imm12"),
                ("SFPSHFT", "Imm12"),
                ("SFPSHFT2b", "Imm12"),
            ]
        );
    }

    #[test]
    fn every_encoding_fits_the_ttinsn_form() {
        // `.ttinsn` needs the word below 0xC000_0000 so the rotated low two bits are
        // not 0b11. No opcode reaches that high, but assert it rather than assume.
        for def in ALL {
            let mut word = def.skeleton();
            for f in def.fields() {
                word |= f.place(f.max_value());
            }
            let instruction = Instruction::new(word, def);
            let rotated = instruction
                .ttinsn_word()
                .unwrap_or_else(|| panic!("{} cannot be pushed as .ttinsn", def.key()));
            assert_ne!(rotated & 3, 3);
            assert_eq!(rotated.rotate_right(2), word);
        }
    }

    #[test]
    fn the_datum_layouts_describe_the_widths_they_claim() {
        for layout in ALL_LAYOUTS {
            for f in layout.fields() {
                assert!(
                    f.first_bit() as u32 + f.width() as u32 <= layout.nbits() as u32,
                    "{}: {} runs past a {}-bit datum",
                    layout.key(),
                    f.name(),
                    layout.nbits()
                );
            }
        }
        // `Src` holds 19-bit datums; `Dst` holds 16- or 32-bit ones.
        assert_eq!(datum::SRC_TF32.nbits(), 19);
        assert_eq!(datum::DST16_FP16.nbits(), 16);
        assert_eq!(datum::DST32_FP32.nbits(), 32);
        // BF16 in Dst is sign + 8-bit exponent + 7-bit mantissa.
        assert_eq!(datum::DST16_BF16.field("Exponent").unwrap().width(), 8);
        assert_eq!(datum::DST16_BF16.field("Mantissa").unwrap().width(), 7);
        assert_eq!(datum::DST16_BF16.field("Sign").unwrap().first_bit(), 15);
    }
}
