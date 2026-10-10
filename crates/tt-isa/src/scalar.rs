//! Checked Scalar Unit arithmetic and L1 memory operands. GPRs are thread-local.
//!
//! Semantics follow the pinned SUBDMAREG, MULDMAREG, CMPDMAREG,
//! SHIFTDMAREG and BITWOPDMAREG functional models. Operand/result aliasing
//! is supported for arithmetic. Memory data/address/offset registers must be
//! disjoint. Register and immediate layouts retain their generated provenance.
//!
//! For tight replay streams, establish an explicit Configuration Unit barrier
//! before publishing a scalar result with WRCFG: use `backend::stallwait` with
//! `Before::EVERYTHING` and `cond::CONFIG_BUSY`. Step100's descriptor gate
//! observes stale publication without this boundary on Blackhole; this does
//! not change scalar arithmetic or the register aliasing contract.
pub mod atomic;
pub mod mmio;

use crate::{
    backend::EncodeError,
    isa::{generated::encode, Instruction},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operand {
    Register(u32),
    Immediate(u32),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Comparison {
    Greater,
    Less,
    Equal,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Direction {
    Left,
    Right,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Bitwise {
    And,
    Or,
    Xor,
}

fn checked(result: u32, left: u32, right: Operand, max: u32) -> Result<u32, EncodeError> {
    crate::backend::check_gpr(result)?;
    crate::backend::check_gpr(left)?;
    match right {
        Operand::Register(r) => {
            crate::backend::check_gpr(r)?;
            Ok(r)
        }
        Operand::Immediate(v) if v <= max => Ok(v),
        Operand::Immediate(value) => Err(EncodeError::ValueTooLarge { value, max }),
    }
}
/// Wrapping 32-bit subtraction, `left - right`.
pub fn sub(result: u32, left: u32, right: Operand) -> Result<Instruction, EncodeError> {
    let r = checked(result, left, right, 63)?;
    match right {
        Operand::Register(_) => encode::subdmareg(result, r, left),
        Operand::Immediate(_) => encode::subdmare_gi(result, r, left),
    }
    .map_err(EncodeError::from_isa)
}
/// Full unsigned product of the low 16 bits of each operand.
pub fn mul_u16(result: u32, left: u32, right: Operand) -> Result<Instruction, EncodeError> {
    let r = checked(result, left, right, 63)?;
    match right {
        Operand::Register(_) => encode::muldmareg(result, r, left),
        Operand::Immediate(_) => encode::muldmare_gi(result, r, left),
    }
    .map_err(EncodeError::from_isa)
}
/// Unsigned comparison; produces exactly zero or one.
pub fn compare(
    result: u32,
    left: u32,
    right: Operand,
    mode: Comparison,
) -> Result<Instruction, EncodeError> {
    let r = checked(result, left, right, 63)?;
    match right {
        Operand::Register(_) => encode::cmpdmareg(mode as u32, result, r, left),
        Operand::Immediate(_) => encode::cmpdmare_gi(mode as u32, result, r, left),
    }
    .map_err(EncodeError::from_isa)
}
/// Logical shift. Register counts use only their low five bits.
pub fn shift(
    result: u32,
    left: u32,
    right: Operand,
    direction: Direction,
) -> Result<Instruction, EncodeError> {
    let r = checked(result, left, right, 31)?;
    match right {
        Operand::Register(_) => encode::shiftdmareg(direction as u32, result, r, left),
        Operand::Immediate(_) => encode::shiftdmare_gi(direction as u32, result, r, left),
    }
    .map_err(EncodeError::from_isa)
}
pub fn bitwise(
    result: u32,
    left: u32,
    right: Operand,
    mode: Bitwise,
) -> Result<Instruction, EncodeError> {
    let r = checked(result, left, right, 63)?;
    match right {
        Operand::Register(_) => encode::bitwopdmareg(mode as u32, result, r, left),
        Operand::Immediate(_) => encode::bitwopdmare_gi(mode as u32, result, r, left),
    }
    .map_err(EncodeError::from_isa)
}

/// L1 transfer size. Partial loads preserve the remaining high result bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum TransferWidth {
    Quadword = 0,
    Word = 1,
    Halfword = 2,
    Byte = 3,
}
impl TransferWidth {
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Quadword => 16,
            Self::Word => 4,
            Self::Halfword => 2,
            Self::Byte => 1,
        }
    }
}
/// Offset update in bytes, independent of the transfer width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum OffsetIncrement {
    None = 0,
    Bytes2 = 1,
    Bytes4 = 2,
    Bytes16 = 3,
}
impl OffsetIncrement {
    pub const fn bytes(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Bytes2 => 2,
            Self::Bytes4 => 4,
            Self::Bytes16 => 16,
        }
    }
}
/// One of the 128 little-endian half-registers in the issuing thread.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OffsetHalf(u32);
impl OffsetHalf {
    pub fn new(index: u32) -> Result<Self, EncodeError> {
        if index > 127 {
            return Err(EncodeError::ValueTooLarge {
                value: index,
                max: 127,
            });
        }
        Ok(Self(index))
    }
    pub const fn index(self) -> u32 {
        self.0
    }
}
fn memory_operands(
    width: TransferWidth,
    data: u32,
    base: u32,
    offset: OffsetHalf,
) -> Result<(), EncodeError> {
    crate::backend::check_gpr(data)?;
    crate::backend::check_gpr(base)?;
    let count = if width == TransferWidth::Quadword {
        4
    } else {
        1
    };
    if count == 4 && data % 4 != 0 {
        return Err(EncodeError::MisalignedMemoryGroup { index: data });
    }
    let group = data..data + count;
    if group.contains(&base) || group.contains(&(offset.0 / 2)) || base == offset.0 / 2 {
        return Err(EncodeError::MemoryRegisterAlias);
    }
    Ok(())
}
/// Asynchronously load from `16 * base + offset`. Runtime alignment, extent and
/// final offset update must be checked by the kernel; drain C0 before consuming.
pub fn load_indirect(
    width: TransferWidth,
    offset: OffsetHalf,
    increment: OffsetIncrement,
    result: u32,
    base: u32,
) -> Result<Instruction, EncodeError> {
    memory_operands(width, result, base, offset)?;
    encode::Loadind::ZERO
        .size(width as u32)
        .offset_half_reg(offset.0)
        .offset_increment(increment as u32)
        .result_reg(result)
        .addr_reg(base)
        .encode()
        .map_err(EncodeError::from_isa)
}
/// Asynchronously store low data bits to declared L1. Drain C0 before publishing.
/// Blackhole STOREIND size values are 0=16, 1=2, 2=4, 3=1 bytes, unlike
/// LOADIND. Measured by step101 `measure_indirect_widths` (card 0, 1791385416).
/// This is a semantic value mapping; generated field positions are unchanged.
pub fn store_indirect_l1(
    width: TransferWidth,
    offset: OffsetHalf,
    increment: OffsetIncrement,
    data: u32,
    base: u32,
) -> Result<Instruction, EncodeError> {
    memory_operands(width, data, base, offset)?;
    encode::StoreindL1::ZERO
        .size(match width {
            TransferWidth::Quadword => 0,
            TransferWidth::Word => 2,
            TransferWidth::Halfword => 1,
            TransferWidth::Byte => 3,
        })
        .offset_half_reg(offset.0)
        .offset_increment(increment as u32)
        .data_reg(data)
        .addr_reg(base)
        .encode()
        .map_err(EncodeError::from_isa)
}
/// Diagnostic no-op; never a substitute for a memory completion barrier.
pub fn dma_nop() -> Instruction {
    encode::dmanop().expect("DMANOP has no operands")
}
