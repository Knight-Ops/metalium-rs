//! What a parsed bit-layout diagram is.
//!
//! `Bits32.lua` is a drawing script: each entry places labelled rectangles over a
//! 16-, 19- or 32-bit word. The labels are what carry the meaning, and they are not
//! all the same kind of thing — see [`Label`].

/// One entry of `Bits32.lua`'s `diagrams` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagram {
    /// The table key, e.g. `SFPSTORE_BH`. Not the mnemonic: mode variants such as
    /// `UNPACR_Regular` and `UNPACR_FlushCache` are separate keys of one mnemonic.
    pub key: String,
    /// Width of the word being drawn. 32 for instructions, 16 or 19 for the `Src`
    /// and `Dst` datum layouts.
    pub nbits: u8,
    /// In the order the diagram lists them, which is descending `first_bit`.
    pub fields: Vec<DrawnField>,
    /// Line in `Bits32.lua` the entry starts on, for error messages.
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrawnField {
    pub first_bit: u8,
    pub width: u8,
    pub label: Label,
    pub line: usize,
}

/// What a diagram label means.
///
/// Distinguishing these is the whole job of the label parser. A label that is a
/// bare number is not a field at all — it is a bit pattern the instruction must
/// carry — and treating the two alike would generate encoders that let software
/// write a sub-opcode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Label {
    /// An operand: `VD`, `Imm10`, `AddrMod`. `note` holds a parenthesised
    /// annotation, as in `Imm12 (signed)` or `Magnitude (low)`.
    Named { name: String, note: Option<String> },
    /// A constant the instruction carries. Usually the opcode at bits 24..31, but
    /// also sub-opcode selectors — `UNPACR_NOP_*` distinguish themselves with
    /// values 0..7 in bits 0..3 — and must-be-zero bits.
    Fixed { value: u32 },
    /// An opcode that carries an operand: `0xB3 + Index1` in `RMWCIB`, where the
    /// digit at the end of the instruction name is part of the encoding.
    Computed { base: u32, addend: String },
}

impl DrawnField {
    /// Bits this field occupies.
    pub fn mask(&self) -> u32 {
        if self.width >= 32 {
            u32::MAX
        } else {
            ((1u32 << self.width) - 1) << self.first_bit
        }
    }

    pub fn last_bit(&self) -> u32 {
        self.first_bit as u32 + self.width as u32 - 1
    }
}

impl Diagram {
    /// Bits no field claims.
    ///
    /// These are the norm rather than the exception — only a small minority of
    /// diagrams cover their whole word — so they are recorded rather than assumed
    /// to be zero.
    pub fn undrawn(&self) -> u32 {
        let covered = self.fields.iter().fold(0u32, |acc, f| acc | f.mask());
        let all = if self.nbits >= 32 {
            u32::MAX
        } else {
            (1u32 << self.nbits) - 1
        };
        all & !covered
    }

    /// The opcode field, if this diagram has one at bits 24..31.
    ///
    /// Its presence is what separates an instruction from a datum layout.
    pub fn opcode_field(&self) -> Option<&DrawnField> {
        self.fields
            .iter()
            .find(|f| f.first_bit == 24 && f.width == 8)
    }
}
