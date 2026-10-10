//! NoC atomic requests beyond the increment (`BlackholeA0/NoC/Atomics.md`).
//!
//! Every form is one `NOC_CMD_AT` request against the L1 of a Tensix or Ethernet
//! tile (never MMIO, never DRAM), always response-marked, so that the old word
//! reaches the initiator's L1 and the request is visible on the transaction ID's
//! `NIU_MST_REQS_OUTSTANDING_ID` counter. Broadcast atomics exist in hardware and
//! are deliberately not constructible here: a broadcast leaves the outstanding
//! counter meaningless (`Interrupts.md:19`).
//!
//! The `NOC_AT_LEN_BE` layouts are the page's bit diagrams
//! (`Diagrams/Src/Bits32.lua`, `NOC_AT_LEN_BE_*`), field by field:
//!
//! | form | layout (bit ranges are `[low, low + width)`) |
//! |---|---|
//! | Increment | `Ofs` 0..2, `IntWidth` 2..7, opcode 1 at 12..16 |
//! | CAS | `Ofs` 0..2, `CmpVal` 2..6, `SetVal` 6..10, opcode 4 |
//! | Swap, mask | `Mask` 2..10, opcode 3 |
//! | Swap, index (6) | `Ofs` 0..2, bit 2 set, opcode 6 |
//! | Swap, index (7) | `Ofs` 2..4, opcode 7 |
//! | Swap, index (10) | `Ofs` 0..2, bits 8..12 = 3, opcode 10 |
//! | Zaamo | `Ofs` 0..2, `Op` 8..11, bit 11 set, opcode 10 |
//! | Parallel add | `Fmt` 0..4, opcode 9 |
//!
//! The compare and set values of CAS are four bits wide, as in the Tensix `ATCAS`
//! instruction the page compares it to.

use super::niu::{initiator::*, Endpoint, TxnId, CMD_AT, MMIO_START, RESP_MARKED, STATIC_VC_1};

/// Opcode (`NOC_AT_LEN_BE[12..16]`) of each form.
mod opcode {
    pub const INCREMENT: u32 = 1;
    pub const SWAP_MASK: u32 = 3;
    pub const CAS: u32 = 4;
    pub const SWAP_INDEX_6: u32 = 6;
    pub const SWAP_INDEX_7: u32 = 7;
    pub const ACC: u32 = 9;
    /// Both the third swap-by-index layout and Zaamo, told apart by bit 11.
    pub const SWAP_INDEX_10_OR_ZAAMO: u32 = 10;
}

/// The three layouts the page lists for "swap, index variant". They have the
/// same pseudocode; only the place of `Ofs` (and the fixed bits) differs.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SwapForm {
    /// Opcode 6, `Ofs` in bits 0..2.
    Six,
    /// Opcode 7, `Ofs` in bits 2..4.
    Seven,
    /// Opcode 10 with bits 8..12 = 3, `Ofs` in bits 0..2.
    Ten,
}

/// The eight operations of "Zaamo operations", numbered as the page's `switch`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Zaamo {
    /// `amoadd.w`
    Add = 0,
    /// `amoxor.w`
    Xor = 1,
    /// `amoor.w`
    Or = 2,
    /// `amoand.w`
    And = 3,
    /// `amomin.w` (signed)
    Min = 4,
    /// `amomax.w` (signed)
    Max = 5,
    /// `amominu.w`
    MinU = 6,
    /// `amomaxu.w`
    MaxU = 7,
}

/// The `Fmt` values of "Parallel addition", numbered as the page's table. The
/// page gives codes 8..=13 and 15 the same interpretation as the code with the
/// high bit clear (or, for 12, 13 and 15, as a 32-bit or 8-bit integer): it does
/// not say what the high bit changes, so they are separate variants and only the
/// ones the model can reproduce are gated.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AccFormat {
    /// 0: 4x fp32, denormals flushed.
    Fp32 = 0,
    /// 1: 8x fp16, denormals flushed.
    Fp16 = 1,
    /// 2: 8x bf16, denormals flushed.
    Bf16 = 2,
    /// 4: 4x u32, wrapping.
    U32 = 4,
    /// 7: 16x u8, **saturating**.
    U8Saturating = 7,
    /// 8: as 0.
    Fp32Alt = 8,
    /// 9: as 1.
    Fp16Alt = 9,
    /// 10: as 2.
    Bf16Alt = 10,
    /// 12: as 4.
    U32Alt12 = 12,
    /// 13: as 4.
    U32Alt13 = 13,
    /// 15: 16x u8, **wrapping** (table note 2; code 7 saturates).
    U8Wrapping = 15,
}

impl AccFormat {
    pub const fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::Fp32,
            1 => Self::Fp16,
            2 => Self::Bf16,
            4 => Self::U32,
            7 => Self::U8Saturating,
            8 => Self::Fp32Alt,
            9 => Self::Fp16Alt,
            10 => Self::Bf16Alt,
            12 => Self::U32Alt12,
            13 => Self::U32Alt13,
            15 => Self::U8Wrapping,
            _ => return None,
        })
    }

    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// One atomic operation and its operands.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AtomicOp {
    /// Add `value` to the word at the target, within the low `int_width + 1` bits
    /// (31: all 32). The old word comes back.
    Increment { value: u32, int_width: u8 },
    /// Store `set` (4 bits) if the word equals `cmp` (4 bits, zero-extended).
    CompareSwap { cmp: u8, set: u8 },
    /// Write `data`'s low half to every even and its high half to every odd
    /// 16-bit lane of the target's 16-byte unit selected by `mask` (bit `i` for
    /// halfword `i`). The old word at the target comes back.
    SwapMask { mask: u8, data: u32 },
    /// Store `data` at the target word.
    SwapIndex { form: SwapForm, data: u32 },
    /// `op` of the target word and `data`.
    Zaamo { op: Zaamo, data: u32 },
    /// Add `data` (broadcast across lanes) to the target's 16-byte unit. The
    /// response word is undefined.
    Accumulate { fmt: AccFormat, data: u32 },
}

/// Why an [`AtomicRequest`] was refused.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AtomicError {
    /// The target or return address is not a word in L1 (or, for an
    /// accumulate, not 16-byte aligned).
    Alignment,
    /// `int_width` above 31, or a compare or set value above 15.
    Operand,
}

/// An atomic against the L1 of another tile, with its response written to this
/// tile's L1 at `ret_local`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AtomicRequest {
    pub to: Endpoint,
    pub ret_local: u32,
    pub op: AtomicOp,
}

impl AtomicRequest {
    /// The `NOC_AT_LEN_BE` and `NOC_AT_DATA` words for this operation, checked.
    pub fn len_be_and_data(&self) -> Result<(u32, u32), AtomicError> {
        let a = self.to.addr;
        if a % 4 != 0 || self.ret_local % 4 != 0 || a >= MMIO_START || self.ret_local >= MMIO_START
        {
            return Err(AtomicError::Alignment);
        }
        let ofs = (a >> 2) & 3;
        let ins = |op: u32| op << 12;
        Ok(match self.op {
            AtomicOp::Increment { value, int_width } => {
                if int_width > 31 {
                    return Err(AtomicError::Operand);
                }
                (
                    ins(opcode::INCREMENT) | (u32::from(int_width) << 2) | ofs,
                    value,
                )
            }
            AtomicOp::CompareSwap { cmp, set } => {
                if cmp > 15 || set > 15 {
                    return Err(AtomicError::Operand);
                }
                (
                    ins(opcode::CAS) | (u32::from(set) << 6) | (u32::from(cmp) << 2) | ofs,
                    0,
                )
            }
            AtomicOp::SwapMask { mask, data } => {
                (ins(opcode::SWAP_MASK) | (u32::from(mask) << 2), data)
            }
            AtomicOp::SwapIndex { form, data } => (
                match form {
                    SwapForm::Six => ins(opcode::SWAP_INDEX_6) | (1 << 2) | ofs,
                    SwapForm::Seven => ins(opcode::SWAP_INDEX_7) | (ofs << 2),
                    SwapForm::Ten => ins(opcode::SWAP_INDEX_10_OR_ZAAMO) | (3 << 8) | ofs,
                },
                data,
            ),
            AtomicOp::Zaamo { op, data } => (
                ins(opcode::SWAP_INDEX_10_OR_ZAAMO) | (1 << 11) | ((op as u32) << 8) | ofs,
                data,
            ),
            AtomicOp::Accumulate { fmt, data } => {
                if a % 16 != 0 {
                    return Err(AtomicError::Alignment);
                }
                (ins(opcode::ACC) | u32::from(fmt.code()), data)
            }
        })
    }

    /// The initiator registers to write before `CMD_CTRL`, in `Command::registers`'
    /// order, for an initiator at `me` under `txn`. Unicast, response-marked,
    /// static virtual channel 1 (as every unicast request in this workspace).
    pub fn registers(&self, me: (u8, u8), txn: TxnId) -> Result<[(u64, u32); 10], AtomicError> {
        let (len_be, data) = self.len_be_and_data()?;
        let ret = Endpoint {
            x: me.0,
            y: me.1,
            addr: self.ret_local,
        };
        Ok([
            (TARG_ADDR_LO, self.to.addr),
            (TARG_ADDR_MID, 0),
            (TARG_ADDR_HI, self.to.hi()),
            (RET_ADDR_LO, ret.addr),
            (RET_ADDR_MID, 0),
            (RET_ADDR_HI, ret.hi()),
            (PACKET_TAG, (txn.index() as u32) << 10),
            (CTRL, CMD_AT | RESP_MARKED | STATIC_VC_1),
            (AT_LEN_BE, len_be),
            (AT_DATA, data),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noc::niu::{Command, Niu};

    const T: TxnId = match TxnId::new(3) {
        Some(t) => t,
        None => panic!(),
    };
    fn to(addr: u32) -> Endpoint {
        Endpoint { x: 4, y: 5, addr }
    }

    #[test]
    fn the_full_width_increment_is_the_existing_command() {
        let a = AtomicRequest {
            to: to(0x1_0004),
            ret_local: 0x2_0000,
            op: AtomicOp::Increment {
                value: 7,
                int_width: 31,
            },
        };
        let b = Command::AtomicIncrement {
            to: to(0x1_0004),
            value: 7,
            ret_local: 0x2_0000,
        };
        assert_eq!(
            a.registers((3, 4), T).unwrap(),
            b.registers((3, 4), T, Niu::Noc0).unwrap()
        );
    }

    #[test]
    fn layouts_follow_the_page_diagrams() {
        let r = |op| {
            AtomicRequest {
                to: to(0x1_0008),
                ret_local: 0x2_0000,
                op,
            }
            .len_be_and_data()
            .unwrap()
        };
        // Ofs = (0x1_0008 >> 2) & 3 = 2.
        assert_eq!(
            r(AtomicOp::CompareSwap { cmp: 0xA, set: 0x5 }),
            ((4 << 12) | (5 << 6) | (0xA << 2) | 2, 0)
        );
        assert_eq!(
            r(AtomicOp::SwapMask {
                mask: 0xA5,
                data: 9
            }),
            ((3 << 12) | (0xA5 << 2), 9)
        );
        let sw = |form| r(AtomicOp::SwapIndex { form, data: 1 }).0;
        assert_eq!(sw(SwapForm::Six), (6 << 12) | 4 | 2);
        assert_eq!(sw(SwapForm::Seven), (7 << 12) | (2 << 2));
        assert_eq!(sw(SwapForm::Ten), (0xA << 12) | (3 << 8) | 2);
        assert_eq!(
            r(AtomicOp::Zaamo {
                op: Zaamo::MaxU,
                data: 1
            })
            .0,
            (0xA << 12) | (1 << 11) | (7 << 8) | 2
        );
        assert_eq!(
            AtomicRequest {
                to: to(0x1_0010),
                ret_local: 0x2_0000,
                op: AtomicOp::Accumulate {
                    fmt: AccFormat::U8Saturating,
                    data: 0x0102_0304
                }
            }
            .len_be_and_data()
            .unwrap(),
            ((9 << 12) | 7, 0x0102_0304)
        );
    }

    #[test]
    fn operands_and_alignment_are_refused() {
        let mk = |addr, ret_local, op| AtomicRequest {
            to: to(addr),
            ret_local,
            op,
        };
        let cas = |cmp, set| AtomicOp::CompareSwap { cmp, set };
        assert_eq!(
            mk(0x1_0000, 0x2_0000, cas(16, 0)).len_be_and_data(),
            Err(AtomicError::Operand)
        );
        assert_eq!(
            mk(0x1_0000, 0x2_0000, cas(0, 16)).len_be_and_data(),
            Err(AtomicError::Operand)
        );
        assert_eq!(
            mk(
                0x1_0000,
                0x2_0000,
                AtomicOp::Increment {
                    value: 1,
                    int_width: 32
                }
            )
            .len_be_and_data(),
            Err(AtomicError::Operand)
        );
        // Not a word; MMIO target; MMIO return; accumulate off a 16-byte unit.
        for (a, r) in [
            (0x1_0002, 0x2_0000),
            (0xFFB2_0000, 0x2_0000),
            (0x1_0000, 0xFFB2_0000),
        ] {
            assert_eq!(
                mk(a, r, cas(1, 2)).len_be_and_data(),
                Err(AtomicError::Alignment)
            );
        }
        let acc = AtomicOp::Accumulate {
            fmt: AccFormat::U32,
            data: 1,
        };
        assert_eq!(
            mk(0x1_0004, 0x2_0000, acc).len_be_and_data(),
            Err(AtomicError::Alignment)
        );
        assert!(mk(0x1_0010, 0x2_0000, acc).len_be_and_data().is_ok());
    }

    #[test]
    fn acc_format_codes_round_trip_and_the_gaps_are_refused() {
        for c in 0..=255u8 {
            match AccFormat::from_code(c) {
                Some(f) => assert_eq!(f.code(), c),
                None => assert!(![0, 1, 2, 4, 7, 8, 9, 10, 12, 13, 15].contains(&c)),
            }
        }
    }
}
