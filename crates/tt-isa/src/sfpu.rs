//! Vector Unit (SFPU) instruction encoding, with the hazards the hardware imposes.
//!
//! The encodings themselves are generated — see [`crate::isa`], which turns
//! `Diagrams/Src/Bits32.lua` into a table and a `const fn` per instruction. What is
//! here is the layer above: the rules the specification states about *using* these
//! instructions, which the bits alone do not express.
//!
//! The split is deliberate. `SFPLOADI` can encode any four-bit `VD`, so the
//! generated encoder accepts one; but `LReg[8]` upwards are constants and a write
//! to one is silently discarded, so [`loadi`] refuses it. `SFPMAD` can encode any
//! `VC`, but the specification says not to use `SFPMUL` unless `VC == 9`, so [`mul`]
//! does not offer the parameter. Every refusal here can cite a page; nothing here
//! is a guess about the encoding, and nothing in the generated layer is a guess
//! about the hardware.

pub mod load_macro;

use crate::isa::generated::encode;
use crate::isa::{self};
use crate::tensix;

pub use crate::isa::Instruction;

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

/// `SFPLOAD` / `SFPSTORE` `Mod0` data-type modes.
///
/// One table, because both instructions use it and Blackhole documents the two
/// `#define` blocks identically (`SFPLOAD.md:144-159`, `SFPSTORE.md:138-153`).
/// Which conversion each mode performs differs by direction — `SFPLOAD`'s
/// `MOD0_FMT_BF16` widens `Dst`'s BF16 to FP32, `SFPSTORE`'s narrows FP32 to BF16 —
/// but the *numbering* is shared, so duplicating it would be two places to get
/// wrong.
pub mod mod0_fmt {
    /// Resolved at execution time against `ALU_FORMAT_SPEC_REG*` backend
    /// configuration. Avoid unless that configuration has been set up.
    pub const SRCB: u32 = 0;
    pub const FP16: u32 = 1;
    pub const BF16: u32 = 2;
    pub const FP32: u32 = 3;
    pub const INT32: u32 = 4;
    pub const INT8: u32 = 5;
    pub const UINT16: u32 = 6;
    pub const HI16: u32 = 7;
    pub const INT16: u32 = 8;
    pub const LO16: u32 = 9;
    /// Blackhole overhauled this mode's addressing; it is not a plain `INT32`.
    pub const INT32_ALL: u32 = 10;
    pub const ZERO: u32 = 11;
    /// Deprecated on Blackhole: no longer performs a data type conversion.
    pub const INT32_SM: u32 = 12;
    /// Deprecated on Blackhole: no longer performs a data type conversion.
    pub const INT8_COMP: u32 = 13;
    pub const LO16_ONLY: u32 = 14;
    pub const HI16_ONLY: u32 = 15;
}

/// The spelling [`store`] was written against, kept so existing call sites read
/// the same. [`mod0_fmt`] is the full table and applies to [`load`] too.
pub use mod0_fmt as store_format;

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

/// Modifier bits named by `SFPMAD.md`'s list of cases automatic stalling misses.
///
/// Kept here rather than generated: they live in prose tables on each
/// instruction's page, not in the encoding diagrams, and the generated layer
/// deliberately describes bits rather than meanings.
mod stall_modes {
    /// `SFPAND_MOD1_USE_VB` / `SFPOR_MOD1_USE_VB`, new in Blackhole.
    pub const USE_VB: u32 = 1;
    /// `SFPSWAP_MOD1_SWAP` — the one `SFPSWAP` mode stalling handles correctly.
    pub const SWAP_UNCONDITIONAL: u32 = 0;
    /// `SFPSHFT2` modes the stalling logic does not see reads in
    /// (`SUBVEC_SHFLROR1_AND_COPY4`, `SUBVEC_SHFLROR1`, `SUBVEC_SHFLSHR1`), and the
    /// two where it watches the wrong register (`SHFT_LREG`, `SHFT_IMM`).
    pub const SHFT2_MISSED: [u32; 5] = [2, 3, 4, 5, 6];
}

/// The `LReg` holding a constant `+0`, which is what makes `SFPMUL` a pure
/// multiply (`SFPMUL.md`).
pub const LREG_ZERO: u32 = 9;

/// The `LReg` holding a constant `1.0` in every lane (`LReg.md:13`), which is what
/// makes `SFPADD` a pure add: `VD = ±(1.0 * VB) ± VC`.
pub const LREG_ONE: u32 = 10;

/// One of the Vector Unit's seventeen `LReg`s (WH `LReg.md`), as an index the
/// type has already checked: what an instruction may read, and -- through
/// [`LReg::writable`] -- what it may write.
///
/// `LReg[0..8]` are general purpose. `LReg[8]`, `[9]`, `[10]` and `[15]` are
/// read-only constants (`0.8373`, zero, `1.0`, and `2 * lane`); `LReg[11..15]`
/// are written only through `SFPCONFIG` ([`ConfigLReg`]); `LReg[16]` belongs
/// to `SFPLOADMACRO` and is not offered at all. A write to anything but
/// `LReg[0..8]` is silently dropped by the hardware, which is why the type
/// keeps the two apart rather than leaving it to a comment.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LReg(u8);

impl LReg {
    pub const L0: LReg = LReg(0);
    pub const L1: LReg = LReg(1);
    pub const L2: LReg = LReg(2);
    pub const L3: LReg = LReg(3);
    pub const L4: LReg = LReg(4);
    pub const L5: LReg = LReg(5);
    pub const L6: LReg = LReg(6);
    pub const L7: LReg = LReg(7);
    /// Read-only: `0.8373` in every lane.
    pub const C0_8373: LReg = LReg(8);
    /// Read-only: zero in every lane, the same bits for every type.
    pub const ZERO: LReg = LReg(LREG_ZERO as u8);
    /// Read-only: `1.0` in every lane.
    pub const ONE: LReg = LReg(LREG_ONE as u8);
    /// Read-only: lane `i` holds the integer `2 * i`.
    pub const LANE_X2: LReg = LReg(15);

    /// General-purpose register `i`, if `i < 8`.
    pub const fn general(i: u32) -> Option<LReg> {
        if i <= MAX_WRITABLE_LREG {
            Some(LReg(i as u8))
        } else {
            None
        }
    }

    pub const fn index(self) -> u32 {
        self.0 as u32
    }

    /// Can an instruction write it? Only `LReg[0..8]`.
    pub const fn writable(self) -> bool {
        self.0 as u32 <= MAX_WRITABLE_LREG
    }
}

/// `LReg[11..15]`: readable like any other, written only by `SFPCONFIG` from
/// `LReg[0]` (`SFPCONFIG.md`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConfigLReg(u8);

impl ConfigLReg {
    pub const L11: ConfigLReg = ConfigLReg(11);
    pub const L12: ConfigLReg = ConfigLReg(12);
    pub const L13: ConfigLReg = ConfigLReg(13);
    pub const L14: ConfigLReg = ConfigLReg(14);

    /// The register, to read.
    pub const fn lreg(self) -> LReg {
        LReg(self.0)
    }
}

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

impl EncodeError {
    /// Translate the generated layer's refusal, which knows the field but not what
    /// the field is for.
    const fn from_isa(e: isa::EncodeError) -> Self {
        match e {
            isa::EncodeError::FieldTooLarge {
                field,
                value,
                width,
                ..
            } => EncodeError::FieldTooLarge {
                name: field,
                value,
                bits: width as u32,
            },
        }
    }
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

impl Instruction {
    /// Does an `SFPNOP` have to be inserted between an `SFPMAD` and this
    /// instruction?
    ///
    /// On Blackhole the two-cycle `SFPMAD` latency is handled by hardware: it
    /// stalls a thread that presents an instruction reading what the `SFPMAD`
    /// wrote (`SFPMAD.md`, "Instruction scheduling"). That is a change from
    /// Wormhole, and it means the blanket "always follow a multiply with a NOP"
    /// rule is wrong here — it would just cost a cycle.
    ///
    /// What survives is a specific list of cases the stalling logic fails to
    /// detect, which `SFPMAD.md` enumerates as hardware bugs. Three are
    /// unconditional; four depend on the consuming instruction's `Mod1`, and since
    /// an [`Instruction`] carries its definition, that can be read rather than
    /// assumed.
    ///
    /// Separately, automatic stalling does not apply at all inside an
    /// `SFPLOADMACRO` sequence — which ttsim does not implement, so that case
    /// cannot arise against the simulator.
    pub fn stalls_automatically_after_mad(self) -> bool {
        let mod1 = self.operand("Mod1").unwrap_or(0);
        let missed = match self.def().mnemonic() {
            // The stalling logic ignores `USE_VB`, so it believes these always read
            // `VD` and never `VB`.
            "SFPAND" | "SFPOR" => mod1 & stall_modes::USE_VB != 0,
            // It does not realise these read from `VD` at all, nor that `SFPCONFIG`
            // can read `LReg[0]`.
            "SFPIADD" | "SFPSHFT" | "SFPCONFIG" => true,
            // Every mode but the unconditional swap compares `VC` and `VD` on its
            // first cycle, and the stalling logic sees no reads there.
            "SFPSWAP" => mod1 != stall_modes::SWAP_UNCONDITIONAL,
            // Three modes it sees no reads in at all, and two where it watches `VD`
            // while the instruction reads `VB`.
            "SFPSHFT2" => {
                let mut i = 0;
                let mut found = false;
                while i < stall_modes::SHFT2_MISSED.len() {
                    found |= stall_modes::SHFT2_MISSED[i] == mod1;
                    i += 1;
                }
                found
            }
            _ => false,
        };
        !missed
    }
}

/// `SFPLOADI`: write a 16-bit immediate into all lanes of `LReg[vd]`.
pub const fn loadi(vd: u32, mode: u32, imm16: u32) -> Result<Instruction, EncodeError> {
    if vd > MAX_WRITABLE_LREG {
        return Err(EncodeError::UnwritableDestination { vd });
    }
    match encode::sfploadi(vd, mode, imm16) {
        Ok(i) => Ok(i),
        Err(e) => Err(EncodeError::from_isa(e)),
    }
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
    // `SFPMUL.md` calls opcode 0x86 the preferred spelling when VC is the
    // constant-zero register, and the two are documented as the same instruction
    // to the hardware -- so the choice is presentational, with one practical
    // constraint: ttsim implements 0x86 only for `Mod1 <= 1`, while its 0x84
    // handler accepts `Mod1 <= 3`. Since the recommended negative-zero form needs
    // NEGATE_VC, a modified multiply has to be spelled SFPMAD to run on the
    // simulator at all. See docs/learnings/ttsim-divergence.md entry 17.
    let encoded = if vc == LREG_ZERO && mod1 == 0 {
        encode::Sfpmul::ZERO
            .va(va)
            .vb(vb)
            .vc(vc)
            .vd(vd)
            .mod1(mod1)
            .encode()
    } else {
        encode::Sfpmad::ZERO
            .va(va)
            .vb(vb)
            .vc(vc)
            .vd(vd)
            .mod1(mod1)
            .encode()
    };
    match encoded {
        Ok(i) => Ok(i),
        Err(e) => Err(EncodeError::from_isa(e)),
    }
}

/// `SFPNOP`: occupy a Vector Unit sub-unit for one cycle.
pub const fn nop() -> Instruction {
    match encode::sfpnop() {
        Ok(i) => i,
        // No operands, so nothing can fail to fit.
        Err(_) => unreachable!(),
    }
}

/// `SFPSTORE`: move 32 datums from `LReg[vd]` into four consecutive rows of `Dst`.
///
/// With `imm10 = 0` and the address-modification registers at their reset values,
/// lane *n* lands at `Dst[n / 8][(n & 7) * 2]` — rows 0..=3, even columns.
///
/// This is the Blackhole encoding, with `AddrMod` three bits wide at bit 13 rather
/// than Wormhole's two at bit 14. That is no longer a fact to remember: the two are
/// separate entries in the generated table, and the Wormhole one is only reachable
/// as `isa::generated::defs::wormhole::SFPSTORE`.
pub const fn store(
    vd: u32,
    format: u32,
    addr_mod: u32,
    imm10: u32,
) -> Result<Instruction, EncodeError> {
    if let Err(e) = check_reg(vd) {
        return Err(e);
    }
    match encode::sfpstore(vd, format, addr_mod, imm10) {
        Ok(i) => Ok(i),
        Err(e) => Err(EncodeError::from_isa(e)),
    }
}

/// `SFPLOAD`: move 32 datums from `Dst` into `LReg[vd]`.
///
/// The mirror of [`store`], and the instruction that lets the SFPU read what the
/// unpacker wrote. `format` is a [`mod0_fmt`] mode.
///
/// # Addressing
///
/// `SFPLOAD.md`: "the top 8 bits of `Addr` end up selecting an aligned group of
/// four rows of `Dst`, the next bit selects between even and odd columns, and the
/// low bit goes unused". So one `SFPLOAD` reaches **half** of a four-row block —
/// lane *n* reads `Dst[(Addr & ~3) + n/8][(n & 7) * 2 + ((Addr & 2) != 0)]` — and
/// covering all sixteen columns takes two, one with bit 1 clear and one with it
/// set. [`DST_ODD_COLUMNS`] is that bit.
///
/// # Refusals
///
/// `vd` must be below 8. The functional model guards the whole lane loop with
/// `if (VD < 8)`, so a load into a constant register is silently dropped rather
/// than faulting — the same shape as the `SFPLOADI` case [`loadi`] refuses.
///
/// Reading a `Dst` row whose `DstRowValid` is false is `UndefinedBehavior`
/// (`ZEROACC.md`), and nothing here can check that: `Dst` has no power-on reset
/// value, so a program that loads before anything has written must scrub first
/// (`Dst.md:15` wants an `SFPSTORE` sweep, not merely a `ZEROACC`).
pub const fn load(
    vd: u32,
    format: u32,
    addr_mod: u32,
    imm10: u32,
) -> Result<Instruction, EncodeError> {
    if let Err(e) = check_reg(vd) {
        return Err(e);
    }
    if vd > MAX_WRITABLE_LREG {
        return Err(EncodeError::UnwritableDestination { vd });
    }
    match encode::sfpload(vd, format, addr_mod, imm10) {
        Ok(i) => Ok(i),
        Err(e) => Err(EncodeError::from_isa(e)),
    }
}

/// The bit of an `SFPLOAD` / `SFPSTORE` address that selects odd `Dst` columns.
///
/// `SFPLOAD.md` / `SFPSTORE.md`: bit 1 of the resolved address adds one to the
/// column, so the even and odd halves of a four-row block are two instructions with
/// the same row group.
pub const DST_ODD_COLUMNS: u32 = 2;

/// `SFPADD`: lanewise FP32 `LReg[vd] = LReg[vb] + LReg[vc]`.
///
/// `SFPADD.md` calls itself "identical to `SFPMAD`, but the preferred opcode when
/// `VA == 10`", because [`LREG_ONE`] makes the product a no-op. `VA` is therefore
/// not a parameter here, for the same reason [`mul`] does not offer `VC`: the
/// specification says not to use the instruction any other way.
///
/// Emitted as the `SFPMAD` spelling rather than the `SFPADD` one. `SFPADD.md`
/// defines its functional model, conformance and scheduling entirely by reference
/// to `SFPMAD` and says nothing the opcode changes, so the two are the same
/// instruction; and there is precedent for preferring the general spelling —
/// `docs/learnings/ttsim-divergence.md` row 17 records the simulator accepting `Mod1` values
/// on `SFPMAD` that it rejects on the narrower `SFPMUL` opcode.
pub const fn add(vb: u32, vc: u32, vd: u32) -> Result<Instruction, EncodeError> {
    mad(LREG_ONE, vb, vc, vd, 0)
}

/// `SFPADD` with the addend negated: `LReg[vd] = LReg[vb] - LReg[vc]`.
pub const fn sub(vb: u32, vc: u32, vd: u32) -> Result<Instruction, EncodeError> {
    mad(LREG_ONE, vb, vc, vd, mad_mod1::NEGATE_VC)
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

    /// This predicate used to return `true` for every instruction the module could
    /// encode, which made it a comment with a function signature. It can answer the
    /// real question now, because an [`Instruction`] carries its definition and its
    /// operands — so the four mode-dependent cases in `SFPMAD.md`'s list can be
    /// decided rather than assumed.
    #[test]
    fn the_cases_automatic_stalling_misses_are_the_documented_ones() {
        use crate::isa::generated::encode;

        // Nothing this module's own wrappers emit is on the list, which is why the
        // step 4 firmware pushes SFPMUL straight into SFPSTORE with no SFPNOP.
        assert!(store(2, store_format::FP32, 0, 0)
            .unwrap()
            .stalls_automatically_after_mad());
        assert!(nop().stalls_automatically_after_mad());
        assert!(loadi(0, loadi_mode::UPPER, 0)
            .unwrap()
            .stalls_automatically_after_mad());

        // Unconditional: the logic does not see that these read VD.
        assert!(!encode::sfpiadd(0, 0, 1, 0)
            .unwrap()
            .stalls_automatically_after_mad());
        assert!(!encode::sfpshft(0, 0, 1, 0)
            .unwrap()
            .stalls_automatically_after_mad());

        // Conditional on Mod1: SFPAND is only a hazard with USE_VB.
        let and = |mod1| {
            encode::sfpand(0, 1, 2, mod1)
                .unwrap()
                .stalls_automatically_after_mad()
        };
        assert!(and(0), "without USE_VB the logic tracks it correctly");
        assert!(!and(stall_modes::USE_VB));

        // SFPSWAP is safe in exactly one mode, the unconditional swap.
        let swap = |mod1| {
            encode::sfpswap(0, 1, mod1)
                .unwrap()
                .stalls_automatically_after_mad()
        };
        assert!(swap(stall_modes::SWAP_UNCONDITIONAL));
        for mod1 in 1..=8 {
            assert!(!swap(mod1), "SFPSWAP mode {mod1} is on the list");
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
