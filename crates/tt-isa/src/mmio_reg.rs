//! Checked Scalar Unit MMIO access: `LOADREG`, `STOREREG` and `STOREIND` (MMIO).
//!
//! There is **no arbitrary-address API** here. The Scalar Unit's MMIO window is
//! `0xFFB0_0000 + 20 bits`, and the pinned Wormhole pages make every address below
//! `0xFFB1_1000` undefined behaviour while the rest of the window holds the TDMA,
//! debug, PIC, NoC and overlay registers, most of which are destructive. Blackhole
//! has no `LOADREG`/`STOREREG`/`STOREIND` page at all (all three layouts are
//! `UNVERIFIED`, see `docs/plans/hardware-coverage-closeout.md` lane F), so the
//! only addresses these helpers will name are an allowlist of [`Scratch`] words.
//!
//! # The one allowlisted target class
//!
//! `SW_INT_PC[i]` of the tile's PIC (`BlackholeA0/TensixTile/PIC.md`, "Memory Map":
//! `0xFFB1_30A8 + 4 * i`, "Read / write", "containing the `pc` of the interrupt
//! handler"). Writing it has no effect other than holding the value, **provided
//! software IRQ `i` is not enabled** for RISCV B or NC (`BRISC_SW_INT_EN`,
//! `NCRISC_SW_INT_EN`); nothing in this repository enables interrupts. The page
//! lists the register block as visible to the NoC, so the host can read the word
//! back independently. Only `i` in 28..=31 is allowed: the neighbours are not.
//! `SW_INT[i]` (`0xFFB1_3018`) *raises* IRQs and a read of it clears them, and
//! `HW_INT_PC` is deliberately out too, so a misaddressed store is a
//! [`MmioError::NotAllowlisted`] at encode time rather than a register write.
//!
//! This module is `UNVERIFIED` on silicon: nothing here executed on a Blackhole
//! card when it was written. Simulator evidence is refusal only (divergence rows
//! 78-79).

use crate::{
    backend::{self, Before, EncodeError},
    isa::{generated::encode, Instruction},
    scalar::{OffsetHalf, OffsetIncrement},
};

/// Base of the Scalar Unit's MMIO window; the 20 bits above it are the offset.
pub const WINDOW_BASE: u32 = 0xFFB0_0000;
/// Exclusive end of the window (`WINDOW_BASE + 2^20`).
pub const WINDOW_END: u32 = 0xFFC0_0000;
/// Addresses below this are `UndefinedBehavior()` in the pinned models.
pub const FORBIDDEN_BELOW: u32 = 0xFFB1_1000;
/// `AddrLo` of `LOADREG`/`STOREREG` is 18 bits of word index.
pub const ADDR_LO_LIMIT: u32 = 1 << 18;
/// The bits of `STOREIND`'s MMIO sum that reach the address (`& 0x000F_FFFC`).
pub const INDIRECT_ADDRESS_MASK: u32 = 0x000F_FFFC;

/// `SW_INT_PC[0]` (`PIC.md`).
pub const SW_INT_PC_BASE: u32 = 0xFFB1_30A8;
/// `BRISC_SW_INT_EN`: must have the target's IRQ bit clear.
pub const BRISC_SW_INT_EN: u32 = 0xFFB1_3000;
/// `NCRISC_SW_INT_EN`: must have the target's IRQ bit clear.
pub const NCRISC_SW_INT_EN: u32 = 0xFFB1_300C;
/// `SW_INT_PC[27]`, the guard word below the allowlist.
pub const GUARD_BELOW: u32 = 0xFFB1_30A8 + 27 * 4;
/// `HW_INT_PC[0]`, the guard word above the allowlist (`PIC.md`: `0xFFB1_3128`).
pub const GUARD_ABOVE: u32 = 0xFFB1_3128;

/// Why a helper refused to build or resolve an access.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MmioError {
    /// An operand the backend encoder rejects (bad GPR, aliasing, ...).
    Encode(EncodeError),
    /// Outside the window, or below [`FORBIDDEN_BELOW`].
    Forbidden { address: u32 },
    /// Inside the window and above the forbidden region, but not allowlisted.
    NotAllowlisted { address: u32 },
    /// `LOADREG`/`STOREREG` `AddrLo` does not fit 18 bits.
    AddrLoTooLarge { value: u32 },
    /// The `STOREIND` sum has bits the hardware would silently drop.
    Truncated { sum: u32 },
    /// A software-IRQ enable register has the target's bit set.
    InterruptEnabled { enable: u32, mask: u32 },
}
impl From<EncodeError> for MmioError {
    fn from(e: EncodeError) -> Self {
        Self::Encode(e)
    }
}
impl core::fmt::Display for MmioError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Encode(e) => write!(f, "{e}"),
            Self::Forbidden { address } => write!(
                f,
                "MMIO address {address:#010x} is outside {FORBIDDEN_BELOW:#x}..{WINDOW_END:#x}"
            ),
            Self::NotAllowlisted { address } => write!(
                f,
                "MMIO address {address:#010x} is not one of the allowlisted scratch words"
            ),
            Self::AddrLoTooLarge { value } => {
                write!(f, "AddrLo {value:#x} does not fit 18 bits")
            }
            Self::Truncated { sum } => write!(
                f,
                "STOREIND address sum {sum:#x} has bits outside {INDIRECT_ADDRESS_MASK:#x}"
            ),
            Self::InterruptEnabled { enable, mask } => write!(
                f,
                "software IRQ enable {enable:#x} overlaps target IRQ mask {mask:#x}"
            ),
        }
    }
}

/// A word the Scalar Unit may read or write: `SW_INT_PC[28..=31]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scratch {
    Irq28,
    Irq29,
    Irq30,
    Irq31,
}
impl Scratch {
    pub const ALL: [Scratch; 4] = [Self::Irq28, Self::Irq29, Self::Irq30, Self::Irq31];
    /// The software IRQ index whose handler `pc` this word is.
    pub const fn irq(self) -> u32 {
        match self {
            Self::Irq28 => 28,
            Self::Irq29 => 29,
            Self::Irq30 => 30,
            Self::Irq31 => 31,
        }
    }
    /// The bit of `BRISC_SW_INT_EN`/`NCRISC_SW_INT_EN` that would make this word live.
    pub const fn irq_mask(self) -> u32 {
        1 << self.irq()
    }
    /// The full MMIO address, which the host reads back over the NoC.
    pub const fn address(self) -> u32 {
        SW_INT_PC_BASE + 4 * self.irq()
    }
    /// `AddrLo` for `LOADREG`/`STOREREG`.
    pub const fn addr_lo(self) -> u32 {
        (self.address() - WINDOW_BASE) >> 2
    }
    /// Offset from [`WINDOW_BASE`], the unit `STOREIND`'s address sum is in.
    pub const fn window_offset(self) -> u32 {
        self.address() - WINDOW_BASE
    }
    /// Resolve an absolute address. Every non-target is an error, never a clamp.
    pub const fn from_address(address: u32) -> Result<Self, MmioError> {
        if address < FORBIDDEN_BELOW || address >= WINDOW_END {
            return Err(MmioError::Forbidden { address });
        }
        let mut i = 0;
        while i < Self::ALL.len() {
            if Self::ALL[i].address() == address {
                return Ok(Self::ALL[i]);
            }
            i += 1;
        }
        Err(MmioError::NotAllowlisted { address })
    }
    /// Resolve a `LOADREG`/`STOREREG` `AddrLo`. No truncation: 18 bits or error.
    pub const fn from_addr_lo(addr_lo: u32) -> Result<Self, MmioError> {
        if addr_lo >= ADDR_LO_LIMIT {
            return Err(MmioError::AddrLoTooLarge { value: addr_lo });
        }
        Self::from_address(WINDOW_BASE + (addr_lo << 2))
    }
    /// Host precondition before the first access: the IRQ must not be enabled for
    /// either core, or the stored value would be a live interrupt vector.
    pub const fn require_irq_disabled(
        self,
        brisc_en: u32,
        ncrisc_en: u32,
    ) -> Result<(), MmioError> {
        let mask = self.irq_mask();
        if brisc_en & mask != 0 {
            return Err(MmioError::InterruptEnabled {
                enable: brisc_en,
                mask,
            });
        }
        if ncrisc_en & mask != 0 {
            return Err(MmioError::InterruptEnabled {
                enable: ncrisc_en,
                mask,
            });
        }
        Ok(())
    }
}

/// `LOADREG`: asynchronously read `target` into `result`. Drain C0 before use.
pub fn load(result: u32, target: Scratch) -> Result<Instruction, MmioError> {
    backend::check_gpr(result)?;
    encode::loadreg(result, target.addr_lo())
        .map_err(|e| MmioError::Encode(EncodeError::from_isa(e)))
}
/// `STOREREG`: write `data` to `target`. Other clients do not observe the write
/// until it reaches the PIC; use [`store_then_read_back`] before any external use.
pub fn store(data: u32, target: Scratch) -> Result<Instruction, MmioError> {
    backend::check_gpr(data)?;
    encode::storereg(data, target.addr_lo())
        .map_err(|e| MmioError::Encode(EncodeError::from_isa(e)))
}
/// `STOREREG`, drain, `LOADREG` of the same word into `scratch`, drain.
///
/// The Scalar Unit sees its own write; other clients do not until it lands
/// (`STOREREG.md`). Reading the word back through the same path is the fence a
/// later consumer needs, and leaves the stored value in `scratch` for checking.
pub fn store_then_read_back(
    data: u32,
    scratch: u32,
    target: Scratch,
) -> Result<[Instruction; 4], MmioError> {
    if data == scratch {
        return Err(MmioError::Encode(EncodeError::MemoryRegisterAlias));
    }
    Ok([
        store(data, target)?,
        backend::wait_for_scalar(Before::EVERYTHING)?,
        load(scratch, target)?,
        backend::wait_for_scalar(Before::EVERYTHING)?,
    ])
}

/// Operands that make `STOREIND` (MMIO) address one [`Scratch`] word.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndirectOperands {
    /// Value for the address GPR.
    pub address_gpr: u32,
    /// Value of the 16-bit offset half. Its **high 12 bits** reach the address.
    pub offset_half: u16,
}

/// What a `STOREIND` (MMIO) with these GPR values writes: the pinned model
/// `0xFFB00000 + ((G + (offset >> 4)) & 0x000FFFFC)`.
///
/// The offset is shifted right by **four**, unlike the L1 variant, where the
/// offset adds to `16 * base` unshifted. A sum with bits outside the mask would be
/// silently truncated by hardware, so it is an error here.
pub const fn resolve_indirect(address_gpr: u32, offset_half: u16) -> Result<Scratch, MmioError> {
    let sum = address_gpr.wrapping_add((offset_half >> 4) as u32);
    if sum & !INDIRECT_ADDRESS_MASK != 0 {
        return Err(MmioError::Truncated { sum });
    }
    Scratch::from_address(WINDOW_BASE + sum)
}
/// Operands that resolve to `target` for a starting `offset_half`.
pub const fn plan_indirect(
    target: Scratch,
    offset_half: u16,
) -> Result<IndirectOperands, MmioError> {
    let high = (offset_half >> 4) as u32;
    // Every target's window offset exceeds the largest shifted offset (0xFFF).
    let address_gpr = target.window_offset() - high;
    match resolve_indirect(address_gpr, offset_half) {
        Ok(t) if t as u32 == target as u32 => Ok(IndirectOperands {
            address_gpr,
            offset_half,
        }),
        Ok(t) => Err(MmioError::NotAllowlisted {
            address: t.address(),
        }),
        Err(e) => Err(e),
    }
}
/// The offset half after one `STOREIND`: raw increments 0, 2, 4 or 16 (not bytes
/// of the MMIO address: a 16 raw increment advances the address sum by one), wrapping at 16 bits.
pub const fn offset_after(offset_half: u16, increment: OffsetIncrement) -> u16 {
    offset_half.wrapping_add(match increment {
        OffsetIncrement::None => 0,
        OffsetIncrement::Bytes2 => 2,
        OffsetIncrement::Bytes4 => 4,
        OffsetIncrement::Bytes16 => 16,
    })
}
/// `STOREIND` (MMIO): write `data` to the word the GPR pair resolves to.
///
/// Data, address and offset GPRs must be disjoint, as for the L1 forms. The caller
/// establishes the GPR values with [`plan_indirect`] and re-resolves them with
/// [`resolve_indirect`] when they are computed at run time.
pub fn store_indirect(
    offset: OffsetHalf,
    increment: OffsetIncrement,
    data: u32,
    address: u32,
) -> Result<Instruction, MmioError> {
    backend::check_gpr(data)?;
    backend::check_gpr(address)?;
    let offset_gpr = offset.index() / 2;
    if data == address || data == offset_gpr || address == offset_gpr {
        return Err(MmioError::Encode(EncodeError::MemoryRegisterAlias));
    }
    encode::storeind_mmio(offset.index(), increment as u32, data, address)
        .map_err(|e| MmioError::Encode(EncodeError::from_isa(e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_are_the_four_pinned_words() {
        // Literal addresses from `PIC.md`: 0xFFB1_30A8 + 4 * i for i in 28..=31.
        let want = [0xFFB1_3118, 0xFFB1_311C, 0xFFB1_3120, 0xFFB1_3124];
        for (t, w) in Scratch::ALL.iter().zip(want) {
            assert_eq!(t.address(), w);
            assert_eq!(Scratch::from_address(w), Ok(*t));
            assert_eq!(Scratch::from_addr_lo(t.addr_lo()), Ok(*t));
        }
        assert_eq!(GUARD_ABOVE, Scratch::Irq31.address() + 4);
        assert_eq!(GUARD_BELOW + 4, Scratch::Irq28.address());
    }
}
