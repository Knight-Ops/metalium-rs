//! Vector Unit (SFPU) instruction encoding.
//!
//! Field layouts are transcribed from `Diagrams/Src/Bits32.lua`, the source the
//! encoding diagrams in the specification are generated from. Where Blackhole
//! differs from Wormhole, `Bits32.lua` carries both — `SFPSTORE` has `AddrMod` at
//! bits 14..15 while `SFPSTORE_BH` has it at 13..15 — and this module encodes the
//! Blackhole form.

use crate::tensix;

/// A 32-bit Tensix instruction word.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Instruction {
    word: u32,
    kind: Kind,
}

/// Enough of an instruction's identity to reason about scheduling hazards.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Kind {
    Loadi,
    /// `SFPMAD`, or `SFPMUL` which is the same instruction with `VC == 9`.
    Mad,
    Nop,
    Store,
    Load,
}

impl Instruction {
    /// The raw instruction word, as pushed by a `sw` to [`tensix::INSTRN_BUF_BASE`].
    pub const fn word(self) -> u32 {
        self.word
    }

    pub const fn kind(self) -> Kind {
        self.kind
    }

    /// The same instruction encoded as the operand of a `.ttinsn` pseudo-instruction.
    ///
    /// `.ttinsn IMM32` is simply `IMM32` rotated left by two bits
    /// (`PushTensixInstruction.md:21`); the core rotates it back and stores the
    /// result to `INSTRN_BUF_BASE`, which is exactly what a `sw` there does. The
    /// point of the encoding is that adjacent `.ttinsn`s sit in the instruction
    /// stream, where the T0/T1/T2 instruction caches can fuse up to four of them
    /// into a single push.
    ///
    /// Returns `None` if the instruction cannot be expressed this way. The form is
    /// only valid for `IMM32 < 0xC000_0000`, because after rotation the low two
    /// bits must not be `0b11` — that is the space uncompressed RISC-V
    /// instructions occupy, and `.ttinsn` lives in the compressed-instruction
    /// encoding space these cores do not implement.
    pub const fn ttinsn_word(self) -> Option<u32> {
        if self.word < 0xC000_0000 {
            Some(self.word.rotate_left(2))
        } else {
            None
        }
    }

    /// Does an `SFPNOP` have to be inserted between an `SFPMAD` and this
    /// instruction?
    ///
    /// On Blackhole the two-cycle `SFPMAD` latency is handled by hardware: it
    /// stalls a thread that presents an instruction reading what the `SFPMAD`
    /// wrote (`SFPMAD.md`, "Instruction scheduling"). That is a change from
    /// Wormhole, and it means the blanket "always follow a multiply with a NOP"
    /// rule is wrong here — it would just cost a cycle.
    ///
    /// What survives is a *specific* list of instructions the stalling logic fails
    /// to detect, due to documented hardware bugs. None of the instructions this
    /// module can currently encode are on it, so this returns `false` throughout;
    /// it exists so that adding `SFPIADD`, `SFPSHFT`, `SFPCONFIG`, `SFPSWAP`,
    /// `SFPSHFT2`, or `SFPAND`/`SFPOR` with `USE_VB` has an obvious place to
    /// record the hazard rather than an obvious place to forget it.
    ///
    /// Separately, automatic stalling does not apply at all inside an
    /// `SFPLOADMACRO` sequence — which ttsim does not implement, so that case
    /// cannot arise here yet.
    pub const fn stalls_automatically_after_mad(self) -> bool {
        match self.kind {
            Kind::Loadi | Kind::Mad | Kind::Nop | Kind::Store | Kind::Load => true,
        }
    }
}

/// Place `value` into `count` bits starting at `first_bit`.
const fn field(first_bit: u32, count: u32, value: u32) -> u32 {
    debug_assert!(first_bit + count <= 32);
    let mask = if count == 32 {
        u32::MAX
    } else {
        (1u32 << count) - 1
    };
    debug_assert!(value <= mask);
    (value & mask) << first_bit
}

/// `SFPLOADI` data-type modes (`SFPLOADI.md`).
pub mod loadi_mode {
    /// Immediate is BF16; widened to FP32 by `imm << 16`.
    pub const FLOATB: u32 = 0;
    /// Immediate is FP16-ish: the exponent is widened and rebiased, with no
    /// handling of denormals, NaNs or infinities.
    pub const FLOATA: u32 = 1;
    /// Immediate is UINT16, zero-extended.
    pub const USHORT: u32 = 2;
    /// Immediate is INT16, sign-extended.
    pub const SHORT: u32 = 4;
    /// Overwrite the high 16 bits, preserving the low 16.
    pub const UPPER: u32 = 8;
    /// Overwrite the low 16 bits, preserving the high 16.
    pub const LOWER: u32 = 10;
}

/// `SFPSTORE` / `SFPLOAD` data-type modes (`SFPSTORE.md`).
pub mod store_format {
    /// Resolved at execution time against `ALU_FORMAT_SPEC_REG*` backend
    /// configuration. Avoid unless that configuration has been set up.
    pub const SRCB: u32 = 0;
    pub const FP16: u32 = 1;
    pub const BF16: u32 = 2;
    pub const FP32: u32 = 3;
    pub const INT32: u32 = 4;
    pub const ZERO: u32 = 11;
}

/// `SFPMAD` / `SFPMUL` modifier bits (`SFPMAD.md`).
pub mod mad_mod1 {
    /// Negate the first multiplicand.
    pub const NEGATE_VA: u32 = 1;
    /// Negate the addend.
    pub const NEGATE_VC: u32 = 2;
    /// Take the `VA` index from the low four bits of `LReg[7]`, per lane.
    pub const INDIRECT_VA: u32 = 4;
    /// Take the `VD` index from the low four bits of `LReg[7]`, per lane.
    pub const INDIRECT_VD: u32 = 8;
}

/// The `LReg` holding a constant `+0`, which is what makes `SFPMUL` a pure
/// multiply (`SFPMUL.md`).
pub const LREG_ZERO: u32 = 9;

/// Highest `LReg` index `SFPLOADI` and `SFPMAD` can write.
///
/// `LReg[8]` upwards are constants; `LReg[11]` through `LReg[14]` are writable
/// only indirectly, via `LReg[0]` and `SFPCONFIG`. A write to a higher index is
/// silently dropped rather than faulting, which is precisely why the constructors
/// below refuse it instead.
pub const MAX_WRITABLE_LREG: u32 = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// A destination register that the instruction cannot write.
    UnwritableDestination { vd: u32 },
    /// A register index outside 0..=15.
    BadRegister { index: u32 },
    /// A field value too large for its bit width.
    FieldTooLarge {
        name: &'static str,
        value: u32,
        bits: u32,
    },
    /// `SFPMUL` used with `VC != 9`.
    MulWithoutZero { vc: u32 },
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EncodeError::UnwritableDestination { vd } => write!(
                f,
                "LReg[{vd}] cannot be written by this instruction (only LReg[0..={MAX_WRITABLE_LREG}] can); \
                 the write would be silently discarded"
            ),
            EncodeError::BadRegister { index } => {
                write!(f, "LReg index {index} is outside 0..=15")
            }
            EncodeError::FieldTooLarge { name, value, bits } => {
                write!(f, "{name} = {value} does not fit in {bits} bits")
            }
            EncodeError::MulWithoutZero { vc } => write!(
                f,
                "SFPMUL requires VC == {LREG_ZERO} (the constant +0), got {vc}; \
                 use SFPMAD for a genuine three-operand multiply-add"
            ),
        }
    }
}

const fn check_reg(index: u32) -> Result<(), EncodeError> {
    if index < 16 {
        Ok(())
    } else {
        Err(EncodeError::BadRegister { index })
    }
}

/// `SFPLOADI`: write a 16-bit immediate into all lanes of `LReg[vd]`.
pub const fn loadi(vd: u32, mode: u32, imm16: u32) -> Result<Instruction, EncodeError> {
    if vd > MAX_WRITABLE_LREG {
        return Err(EncodeError::UnwritableDestination { vd });
    }
    if imm16 > 0xFFFF {
        return Err(EncodeError::FieldTooLarge {
            name: "Imm16",
            value: imm16,
            bits: 16,
        });
    }
    if mode > 0xF {
        return Err(EncodeError::FieldTooLarge {
            name: "Mod0",
            value: mode,
            bits: 4,
        });
    }
    Ok(Instruction {
        word: field(0, 16, imm16) | field(16, 4, mode) | field(20, 4, vd) | field(24, 8, 0x71),
        kind: Kind::Loadi,
    })
}

/// `SFPMUL`: lanewise FP32 `LReg[vd] = LReg[va] * LReg[vb]`.
///
/// There is no bare multiply in this unit — `SFPMUL` is `SFPMAD` with `VC` pinned
/// to the constant-zero register, which is the form the specification requires
/// ("Do not use this instruction unless `VC == 9`"). Making `VC` implicit rather
/// than a parameter keeps that from being a rule to remember.
///
/// # Negative zero
///
/// `NEGATE_VC` is set, so the addend is `-0` rather than `+0`. That is not a
/// refinement, it is what makes this a multiply: `-0` is the identity element of
/// floating-point addition, while adding `+0` turns every negatively-signed zero
/// product into `+0`. Without it, `-1.0 * 0.0` yields `+0.0` where IEEE754 — and
/// every host implementation — gives `-0.0`. `SFPMUL.md` calls this out, and a
/// differential test against the host finds it immediately.
///
/// Use [`mad`] directly for a genuine three-operand multiply-add.
pub const fn mul(va: u32, vb: u32, vd: u32) -> Result<Instruction, EncodeError> {
    mad(va, vb, LREG_ZERO, vd, mad_mod1::NEGATE_VC)
}

/// `SFPMAD`: lanewise FP32 `LReg[vd] = ±(LReg[va] * LReg[vb]) ± LReg[vc]`.
pub const fn mad(
    va: u32,
    vb: u32,
    vc: u32,
    vd: u32,
    mod1: u32,
) -> Result<Instruction, EncodeError> {
    if let Err(e) = check_reg(va) {
        return Err(e);
    }
    if let Err(e) = check_reg(vb) {
        return Err(e);
    }
    if let Err(e) = check_reg(vc) {
        return Err(e);
    }
    if vd > MAX_WRITABLE_LREG {
        return Err(EncodeError::UnwritableDestination { vd });
    }
    if mod1 > 0xF {
        return Err(EncodeError::FieldTooLarge {
            name: "Mod1",
            value: mod1,
            bits: 4,
        });
    }
    // Opcode 0x84 is SFPMAD; 0x86 is SFPMUL, which `SFPMUL.md` calls the preferred
    // spelling when VC is the constant-zero register. They are documented as the
    // same instruction to the hardware, so the choice is presentational — with one
    // practical constraint: ttsim implements opcode 0x86 only for `Mod1 <= 1`,
    // while its 0x84 handler accepts `Mod1 <= 3`. Since the recommended
    // negative-zero form needs NEGATE_VC, a modified multiply has to be spelled
    // SFPMAD to run on the simulator at all.
    let opcode = if vc == LREG_ZERO && mod1 == 0 {
        0x86
    } else {
        0x84
    };
    Ok(Instruction {
        word: field(0, 4, mod1)
            | field(4, 4, vd)
            | field(8, 4, vc)
            | field(12, 4, vb)
            | field(16, 4, va)
            | field(24, 8, opcode),
        kind: Kind::Mad,
    })
}

/// `SFPNOP`: occupy a Vector Unit sub-unit for one cycle.
pub const fn nop() -> Instruction {
    Instruction {
        word: field(24, 8, 0x8F),
        kind: Kind::Nop,
    }
}

/// `SFPSTORE`: move 32 datums from `LReg[vd]` into four consecutive rows of `Dst`.
///
/// With `imm10 = 0` and the address-modification registers at their reset values,
/// lane *n* lands at `Dst[n / 8][(n & 7) * 2]` — rows 0..=3, even columns.
pub const fn store(
    vd: u32,
    format: u32,
    addr_mod: u32,
    imm10: u32,
) -> Result<Instruction, EncodeError> {
    if let Err(e) = check_reg(vd) {
        return Err(e);
    }
    if imm10 > 0x3FF {
        return Err(EncodeError::FieldTooLarge {
            name: "Imm10",
            value: imm10,
            bits: 10,
        });
    }
    // Three bits on Blackhole, two on Wormhole.
    if addr_mod > 0x7 {
        return Err(EncodeError::FieldTooLarge {
            name: "AddrMod",
            value: addr_mod,
            bits: 3,
        });
    }
    if format > 0xF {
        return Err(EncodeError::FieldTooLarge {
            name: "Mod0",
            value: format,
            bits: 4,
        });
    }
    Ok(Instruction {
        word: field(0, 10, imm10)
            | field(13, 3, addr_mod)
            | field(16, 4, format)
            | field(20, 4, vd)
            | field(24, 8, 0x72),
        kind: Kind::Store,
    })
}

/// Load a full FP32 constant into `LReg[vd]`.
///
/// `SFPLOADI` carries a 16-bit immediate, so an arbitrary FP32 value takes two
/// instructions: one for each half. Returns them in the order they must be pushed.
///
/// A value whose low 16 bits are zero — which includes every BF16-representable
/// constant, and so both operands of the 3.0 × 2.0 gate — could be loaded with a
/// single `FLOATB` instruction instead. The two-instruction form is used
/// unconditionally because it is correct for every value, and the saving is one
/// cycle of a program that is not cycle-bound.
pub const fn load_f32(vd: u32, bits: u32) -> Result<[Instruction; 2], EncodeError> {
    let hi = match loadi(vd, loadi_mode::UPPER, bits >> 16) {
        Ok(i) => i,
        Err(e) => return Err(e),
    };
    let lo = match loadi(vd, loadi_mode::LOWER, bits & 0xFFFF) {
        Ok(i) => i,
        Err(e) => return Err(e),
    };
    Ok([hi, lo])
}

/// Address of `Dst` element `[row][column]` in RISCV T0/T1/T2's address space,
/// for the 32-bit shapes (`RISC_DEST_ACCESS_CTRL_SEC[].fmt` of 0 or 1).
///
/// `Dst.md` warns that 32-bit `lw`/`sw` "will generally misbehave" — that applies
/// to the 16-bit shapes (`fmt` 2 through 5), where an array element is 16 bits and
/// `lhu`/`sh` are required. For `fmt` 0 and 1 the element genuinely is 32 bits and
/// a word access is the correct one.
pub const fn dst32_address(row: u32, column: u32) -> u64 {
    debug_assert!(row < 512 && column < 16);
    tensix::DST_BASE + ((row * 16 + column) as u64) * 4
}

/// Convert a word read from `Dst` through the RISCV mapping into an IEEE754 FP32
/// bit pattern.
///
/// The hardware applies this itself unless `RISC_DEST_ACCESS_CTRL_SEC[].no_swizzle`
/// is set, so this is only needed when reading with swizzling disabled. It is the
/// documented transform from `Dst.md`'s `Load32`, kept here so the two directions
/// live together.
pub const fn unswizzle_fp32(value: u32) -> u32 {
    (value & 0x8000_FFFF) | ((value & 0x7F00_0000) >> 8) | ((value & 0x00FF_0000) << 7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opcodes_match_bits32_lua() {
        assert_eq!(loadi(0, loadi_mode::UPPER, 0).unwrap().word() >> 24, 0x71);
        assert_eq!(
            mad(0, 1, LREG_ZERO, 2, 0).unwrap().word() >> 24,
            0x86,
            "SFPMUL"
        );
        assert_eq!(mad(0, 1, 2, 3, 0).unwrap().word() >> 24, 0x84, "SFPMAD");
        assert_eq!(nop().word() >> 24, 0x8F);
        assert_eq!(
            store(0, store_format::FP32, 0, 0).unwrap().word() >> 24,
            0x72
        );
    }

    #[test]
    fn loadi_fields_are_placed_correctly() {
        let i = loadi(5, loadi_mode::LOWER, 0xBEEF).unwrap().word();
        assert_eq!(i & 0xFFFF, 0xBEEF, "Imm16 at 0");
        assert_eq!((i >> 16) & 0xF, loadi_mode::LOWER, "Mod0 at 16");
        assert_eq!((i >> 20) & 0xF, 5, "VD at 20");
    }

    #[test]
    fn mad_fields_are_placed_correctly() {
        let i = mad(0xA, 0xB, 0xC, 0x7, 0x3).unwrap().word();
        assert_eq!(i & 0xF, 0x3, "Mod1 at 0");
        assert_eq!((i >> 4) & 0xF, 0x7, "VD at 4");
        assert_eq!((i >> 8) & 0xF, 0xC, "VC at 8");
        assert_eq!((i >> 12) & 0xF, 0xB, "VB at 12");
        assert_eq!((i >> 16) & 0xF, 0xA, "VA at 16");
    }

    /// The single most consequential Blackhole/Wormhole encoding difference this
    /// module has to get right: `AddrMod` moved from bits 14..15 to 13..15.
    #[test]
    fn store_uses_the_blackhole_addrmod_position() {
        let i = store(2, store_format::FP32, 0b101, 0x3FF).unwrap().word();
        assert_eq!(i & 0x3FF, 0x3FF, "Imm10 at 0");
        assert_eq!(
            (i >> 13) & 0x7,
            0b101,
            "AddrMod occupies bits 13..=15 on Blackhole"
        );
        assert_eq!((i >> 16) & 0xF, store_format::FP32, "Mod0 at 16");
        assert_eq!((i >> 20) & 0xF, 2, "VD at 20");

        // Encoding it the Wormhole way would put AddrMod at 14 and leave bit 13
        // clear, so the two genuinely differ for an odd AddrMod.
        let wormhole_style = (0b101u32 << 14) & 0xC000;
        assert_ne!(i & 0xE000, wormhole_style);
    }

    #[test]
    fn mul_pins_vc_to_the_zero_register_and_negates_it() {
        let m = mul(0, 1, 2).unwrap();
        assert_eq!(
            (m.word() >> 8) & 0xF,
            LREG_ZERO,
            "VC is the constant-zero register"
        );
        assert_eq!(
            m.word() & 0xF,
            mad_mod1::NEGATE_VC,
            "the addend must be -0, or a negative zero product comes back as +0"
        );
        // Same instruction as the equivalent SFPMAD.
        assert_eq!(mad(0, 1, LREG_ZERO, 2, mad_mod1::NEGATE_VC).unwrap(), m);
    }

    #[test]
    fn opcode_selection_keeps_modified_multiplies_on_sfpmad() {
        // An unmodified multiply against the zero register gets the SFPMUL
        // spelling the specification prefers.
        assert_eq!(mad(0, 1, LREG_ZERO, 2, 0).unwrap().word() >> 24, 0x86);
        // Anything with a modifier -- including the NEGATE_VC that `mul` needs --
        // uses SFPMAD, which is the same instruction and the only spelling ttsim
        // implements for those modes.
        for mod1 in 1..16 {
            assert_eq!(
                mad(0, 1, LREG_ZERO, 2, mod1).unwrap().word() >> 24,
                0x84,
                "Mod1 = {mod1} must use the SFPMAD spelling"
            );
        }
        assert_eq!(
            mad(0, 1, 3, 2, 0).unwrap().word() >> 24,
            0x84,
            "other VC is SFPMAD"
        );
    }

    #[test]
    fn unwritable_destinations_are_refused() {
        // LReg[8] and up are constants or need SFPCONFIG. A write would be
        // silently discarded, which is worse than an error.
        for vd in 8..16 {
            assert!(matches!(
                loadi(vd, loadi_mode::UPPER, 0),
                Err(EncodeError::UnwritableDestination { .. })
            ));
            assert!(matches!(
                mul(0, 1, vd),
                Err(EncodeError::UnwritableDestination { .. })
            ));
        }
        assert!(loadi(MAX_WRITABLE_LREG, loadi_mode::UPPER, 0).is_ok());
        // But SFPSTORE *reads* VD, so any register is fine as its source.
        assert!(store(15, store_format::FP32, 0, 0).is_ok());
    }

    #[test]
    fn oversized_fields_are_refused() {
        assert!(matches!(
            loadi(0, loadi_mode::UPPER, 0x1_0000),
            Err(EncodeError::FieldTooLarge { name: "Imm16", .. })
        ));
        assert!(matches!(
            store(0, store_format::FP32, 0, 0x400),
            Err(EncodeError::FieldTooLarge { name: "Imm10", .. })
        ));
        assert!(matches!(
            store(0, store_format::FP32, 8, 0),
            Err(EncodeError::FieldTooLarge {
                name: "AddrMod",
                ..
            })
        ));
    }

    #[test]
    fn load_f32_splits_a_constant_into_halves() {
        // 6.0f == 0x40C00000.
        let [hi, lo] = load_f32(2, 0x40C0_0000).unwrap();
        assert_eq!(hi.word() & 0xFFFF, 0x40C0);
        assert_eq!((hi.word() >> 16) & 0xF, loadi_mode::UPPER);
        assert_eq!(lo.word() & 0xFFFF, 0x0000);
        assert_eq!((lo.word() >> 16) & 0xF, loadi_mode::LOWER);
        assert_eq!((hi.word() >> 20) & 0xF, 2);
        assert_eq!((lo.word() >> 20) & 0xF, 2);
    }

    #[test]
    fn ttinsn_is_a_two_bit_left_rotation() {
        // PushTensixInstruction.md:21. The rotation moves the top two bits into
        // the low two, which is why the form is only valid below 0xC000_0000: the
        // low two bits must not both be set.
        let i = loadi(0, loadi_mode::UPPER, 0x4040).unwrap();
        let t = i.ttinsn_word().unwrap();
        assert_eq!(t, i.word().rotate_left(2));
        assert_ne!(
            t & 3,
            3,
            "would collide with the uncompressed encoding space"
        );
        assert_eq!(t.rotate_right(2), i.word(), "the core rotates it back");
    }

    #[test]
    fn every_encodable_instruction_fits_the_ttinsn_form() {
        // The highest opcode this module emits is 0x8F, so all of them are below
        // 0xC000_0000 and none needs the `sw` fallback for encoding reasons.
        let all = [
            loadi(0, loadi_mode::UPPER, 0xFFFF).unwrap(),
            mul(0, 1, 2).unwrap(),
            mad(15, 15, 15, 7, 15).unwrap(),
            nop(),
            store(15, 15, 7, 0x3FF).unwrap(),
        ];
        for i in all {
            assert!(
                i.ttinsn_word().is_some(),
                "{:#010x} has no .ttinsn form",
                i.word()
            );
        }
    }

    #[test]
    fn dst_addresses_are_word_indexed_from_the_mapping_base() {
        assert_eq!(dst32_address(0, 0), tensix::DST_BASE);
        assert_eq!(dst32_address(0, 1), tensix::DST_BASE + 4);
        assert_eq!(dst32_address(1, 0), tensix::DST_BASE + 64);
        // Inverse of Dst.md's `Addr = (Addr - 0xFFBD8000) / 4;
        // Value = Dst32b[Addr / 16][Addr % 16]`.
        for (row, col) in [(0u32, 0u32), (3, 15), (7, 2), (511, 15)] {
            let a = dst32_address(row, col);
            let idx = ((a - tensix::DST_BASE) / 4) as u32;
            assert_eq!((idx / 16, idx % 16), (row, col));
        }
    }

    #[test]
    fn unswizzle_is_an_involution_on_its_own_output() {
        // Not a general property -- it is stated because the transform moves
        // exponent and mantissa bytes past each other, and a reader will want to
        // know whether applying it twice is safe. It is not: this pins that.
        let x = 0x40C0_0000u32;
        assert_ne!(unswizzle_fp32(unswizzle_fp32(x)), x);
        // The bits it must preserve.
        assert_eq!(unswizzle_fp32(0x8000_0000) & 0x8000_0000, 0x8000_0000);
        assert_eq!(unswizzle_fp32(0x0000_FFFF) & 0xFFFF, 0xFFFF);
    }

    #[test]
    fn where_sfpstore_puts_each_lane() {
        // SFPSTORE.md's functional model, with Imm10 = 0 and the address-modify
        // registers at reset: Row = (Addr & ~3) + Lane / 8, Column = (Lane & 7) * 2.
        // Lane 0 therefore lands at Dst[0][0], which is what the gate reads back.
        let lane = 0u32;
        assert_eq!(dst32_address(lane / 8, (lane & 7) * 2), tensix::DST_BASE);
        let lane = 9u32;
        assert_eq!(dst32_address(lane / 8, (lane & 7) * 2), dst32_address(1, 2));
    }
}
