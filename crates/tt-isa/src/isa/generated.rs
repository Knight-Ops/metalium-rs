//! The Tensix instruction set, generated from `Diagrams/Src/Bits32.lua`.
//!
//! **Do not edit.** Regenerate with `cargo xtask gen-isa`; CI checks that this
//! file matches the specification revision pinned in `PINS.toml`.
//!
//! Source: tt-isa-documentation `f848eb668c2aeae742a88a49a86157e24a0a20c6`,
//! 165 instruction encodings and 19 datum layouts, each
//! cross-checked against the `TT_*(…)` syntax block on the page that embeds
//! its diagram — an independently written description of the same bits.
//!
//! Provenance: 50 documented for Blackhole, 24 shared with Wormhole and stated
//! to be identical, 28 superseded on Blackhole, 58 Wormhole-only and therefore
//! **`UNVERIFIED`**, 17 **`MEASURED`** on ttsim or silicon where the specification
//! draws only Wormhole's layout (`xtask/src/gen_isa/Bits32_BH.lua`), and 4
//! Wormhole-only layouts **`CONFIRMED`** unchanged on Blackhole by a gate
//! (`xtask/src/gen_isa/measured.rs`, `CONFIRMED`).
//!
//! Names are the `Bits32.lua` diagram keys, so a name in the specification can
//! be found here without translation — except that a Blackhole-specific form
//! drops its `_BH` suffix and the Wormhole form it replaces moves into the
//! `wormhole` submodule, so the plain name is always the one to use here.
//! Names keep the specification's spelling, so a name in the documentation can
//! be searched for here without translation. That is why this module allows
//! globals that are not upper case.
#![allow(non_upper_case_globals)]
#![allow(clippy::unreadable_literal)]

use super::{DatumLayout, EncodeError, Field, Instruction, InstructionDef, Provenance};

/// Every instruction encoding, by name. Blackhole's form takes the plain
/// name; the Wormhole form it supersedes is under [`defs::wormhole`].
pub mod defs {
    use super::*;

    /// `ATCAS`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/ATCAS.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static ATCAS: InstructionDef = InstructionDef::new(
        "ATCAS",
        "ATCAS",
        0x64,
        &[
            Field::new("SetVal", 18, 4, false, None),
            Field::new("CmpVal", 14, 4, false, None),
            Field::new("Ofs", 12, 2, false, None),
            Field::new("AddrReg", 0, 6, false, None),
        ],
        &[],
        0x00c00fc0,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/ATCAS.md",
    );

    /// `ATSWAP`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/ATSWAP.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static ATSWAP: InstructionDef = InstructionDef::new(
        "ATSWAP",
        "ATSWAP",
        0x63,
        &[
            Field::new("SingleDataReg", 22, 1, false, None),
            Field::new("Mask", 14, 8, false, None),
            Field::new("DataReg", 6, 6, false, None),
            Field::new("AddrReg", 0, 6, false, None),
        ],
        &[],
        0x00803000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/ATSWAP.md",
    );

    /// `RMWCIB0`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/RMWCIB.md`.
    pub static RMWCIB0: InstructionDef = InstructionDef::new(
        "RMWCIB0",
        "RMWCIB0",
        0xb3,
        &[
            Field::new("Mask", 16, 8, false, None),
            Field::new("NewValue", 8, 8, false, None),
            Field::new("Index4", 0, 8, false, None),
        ],
        &[],
        0x00000000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/RMWCIB.md",
    );

    /// `RMWCIB1`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/RMWCIB.md`.
    pub static RMWCIB1: InstructionDef = InstructionDef::new(
        "RMWCIB1",
        "RMWCIB1",
        0xb4,
        &[
            Field::new("Mask", 16, 8, false, None),
            Field::new("NewValue", 8, 8, false, None),
            Field::new("Index4", 0, 8, false, None),
        ],
        &[],
        0x00000000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/RMWCIB.md",
    );

    /// `RMWCIB2`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/RMWCIB.md`.
    pub static RMWCIB2: InstructionDef = InstructionDef::new(
        "RMWCIB2",
        "RMWCIB2",
        0xb5,
        &[
            Field::new("Mask", 16, 8, false, None),
            Field::new("NewValue", 8, 8, false, None),
            Field::new("Index4", 0, 8, false, None),
        ],
        &[],
        0x00000000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/RMWCIB.md",
    );

    /// `RMWCIB3`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/RMWCIB.md`.
    pub static RMWCIB3: InstructionDef = InstructionDef::new(
        "RMWCIB3",
        "RMWCIB3",
        0xb6,
        &[
            Field::new("Mask", 16, 8, false, None),
            Field::new("NewValue", 8, 8, false, None),
            Field::new("Index4", 0, 8, false, None),
        ],
        &[],
        0x00000000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/RMWCIB.md",
    );

    /// `SETDMAREG_Immediate`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SETDMAREG_Immediate.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SETDMAREG_Immediate: InstructionDef = InstructionDef::new(
        "SETDMAREG_Immediate",
        "SETDMAREG",
        0x45,
        &[
            Field::new("NewValue", 8, 16, false, None),
            Field::new("ResultHalfReg", 0, 7, false, None),
        ],
        &[(Field::new("", 7, 1, false, None), 0)],
        0x00000000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SETDMAREG_Immediate.md",
    );

    /// `SETDMAREG_Special`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SETDMAREG_Special.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SETDMAREG_Special: InstructionDef = InstructionDef::new(
        "SETDMAREG_Special",
        "SETDMAREG",
        0x45,
        &[
            Field::new("ResultSize", 22, 2, false, None),
            Field::new("WhichPackers", 15, 4, false, None),
            Field::new("InputSource", 11, 4, false, None),
            Field::new("InputHalfReg", 8, 3, false, None),
            Field::new("ResultHalfReg", 0, 7, false, None),
        ],
        &[(Field::new("", 7, 1, false, None), 1)],
        0x00380000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SETDMAREG_Special.md",
    );

    /// `STALLWAIT_BH`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md`.
    pub static STALLWAIT: InstructionDef = InstructionDef::new(
        "STALLWAIT_BH",
        "STALLWAIT",
        0xa2,
        &[
            Field::new("BlockMask", 15, 9, false, None),
            Field::new("ConditionMask", 0, 13, false, None),
        ],
        &[],
        0x00006000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md",
    );

    /// `SEMWAIT`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SEMWAIT.md`.
    pub static SEMWAIT: InstructionDef = InstructionDef::new(
        "SEMWAIT",
        "SEMWAIT",
        0xa6,
        &[
            Field::new("BlockMask", 15, 9, false, None),
            Field::new("SemaphoreMask", 2, 8, false, None),
            Field::new("ConditionMask", 0, 2, false, None),
        ],
        &[],
        0x00007c00,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SEMWAIT.md",
    );

    /// `STREAMWAIT`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/STREAMWAIT.md`.
    pub static STREAMWAIT: InstructionDef = InstructionDef::new(
        "STREAMWAIT",
        "STREAMWAIT",
        0xa7,
        &[
            Field::new("BlockMask", 15, 9, false, None),
            Field::new("TargetValueLo", 4, 10, false, None),
            Field::new("ConditionIndex", 3, 1, false, None),
            Field::new("StreamSelect", 0, 2, false, None),
        ],
        &[],
        0x00004004,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/STREAMWAIT.md",
    );

    /// `SEMINIT`. `WormholeB0/TensixTile/TensixCoprocessor/SEMINIT.md`, which the Blackhole tree states is shared and identical.
    pub static SEMINIT: InstructionDef = InstructionDef::new(
        "SEMINIT",
        "SEMINIT",
        0xa3,
        &[
            Field::new("NewMax", 20, 4, false, None),
            Field::new("NewValue", 16, 4, false, None),
            Field::new("SemaphoreMask", 2, 8, false, None),
        ],
        &[],
        0x0000fc03,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SEMINIT.md",
    );

    /// `REPLAY`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/REPLAY.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static REPLAY: InstructionDef = InstructionDef::new(
        "REPLAY",
        "REPLAY",
        0x04,
        &[
            Field::new("Index", 14, 5, false, None),
            Field::new("Count", 4, 6, false, None),
            Field::new("Exec", 1, 1, false, None),
            Field::new("Load", 0, 1, false, None),
        ],
        &[],
        0x00f83c0c,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/REPLAY.md",
    );

    /// `PACR_BH`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/PACR.md`.
    pub static PACR: InstructionDef = InstructionDef::new(
        "PACR_BH",
        "PACR",
        0x41,
        &[
            Field::new("CfgContext", 21, 2, false, None),
            Field::new("RowPadZero", 18, 3, false, None),
            Field::new("DstAccessMode", 17, 1, false, None),
            Field::new("AddrMod", 15, 2, false, None),
            Field::new("AddrCntContext", 13, 2, false, None),
            Field::new("ZeroWrite", 12, 1, false, None),
            Field::new("ReadIntfSel", 8, 4, false, None),
            Field::new("OvrdThreadId", 7, 1, false, None),
            Field::new("Concat", 4, 1, false, None),
            Field::new("CtxtCtrl", 2, 2, false, None),
            Field::new("Flush", 1, 1, false, None),
            Field::new("Last", 0, 1, false, None),
        ],
        &[
            (Field::new("", 5, 2, false, None), 0),
            (Field::new("", 23, 1, false, None), 0),
        ],
        0x00000000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/PACR.md",
    );

    /// `PACR_SETREG`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/PACR_SETREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static PACR_SETREG: InstructionDef = InstructionDef::new(
        "PACR_SETREG",
        "PACR_SETREG",
        0x4a,
        &[
            Field::new("AddrSel", 22, 1, false, None),
            Field::new("Value10", 12, 10, false, None),
            Field::new("AddrMid", 2, 6, false, None),
        ],
        &[
            (Field::new("", 1, 1, false, None), 1),
            (Field::new("", 8, 4, false, None), 15),
            (Field::new("", 23, 1, false, None), 1),
        ],
        0x00000001,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/PACR_SETREG.md",
    );

    /// `SFPSHFT2`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPSHFT2.md`.
    pub static SFPSHFT2: InstructionDef = InstructionDef::new(
        "SFPSHFT2",
        "SFPSHFT2",
        0x94,
        &[
            Field::new("VB", 12, 4, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00ff0000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPSHFT2.md",
    );

    /// `SFPSHFT2b`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPSHFT2.md`.
    pub static SFPSHFT2b: InstructionDef = InstructionDef::new(
        "SFPSHFT2b",
        "SFPSHFT2",
        0x94,
        &[
            Field::new("Imm12", 12, 12, true, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00000f00,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPSHFT2.md",
    );

    /// `SFPLUTFP32_BH`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPLUTFP32.md`.
    pub static SFPLUTFP32: InstructionDef = InstructionDef::new(
        "SFPLUTFP32_BH",
        "SFPLUTFP32",
        0x95,
        &[
            Field::new("Mod1Mirror", 16, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00f0ff00,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPLUTFP32.md",
    );

    /// `REG2FLOP_Configuration`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/REG2FLOP_Configuration.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static REG2FLOP_Configuration: InstructionDef = InstructionDef::new(
        "REG2FLOP_Configuration",
        "REG2FLOP",
        0x48,
        &[
            Field::new("SizeSel", 22, 2, false, None),
            Field::new("ThConCfgIndex", 6, 7, false, None),
            Field::new("InputReg", 0, 6, false, None),
        ],
        &[(Field::new("", 20, 2, false, None), 0)],
        0x000fe000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/REG2FLOP_Configuration.md",
    );

    /// `REG2FLOP_ADC`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/REG2FLOP_ADC.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static REG2FLOP_ADC: InstructionDef = InstructionDef::new(
        "REG2FLOP_ADC",
        "REG2FLOP",
        0x48,
        &[
            Field::new("SizeSel", 22, 2, false, None),
            Field::new("OverrideThread", 20, 1, false, None),
            Field::new("Shift8", 18, 2, false, None),
            Field::new("ThreadSel", 16, 2, false, None),
            Field::new("Channel", 11, 1, false, None),
            Field::new("ADCSel", 9, 2, false, None),
            Field::new("Cr", 8, 1, false, None),
            Field::new("XYZW", 6, 2, false, None),
            Field::new("InputReg", 0, 6, false, None),
        ],
        &[(Field::new("", 21, 1, false, None), 1)],
        0x0000f000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/REG2FLOP_ADC.md",
    );

    /// `XMOV`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/XMOV.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static XMOV: InstructionDef = InstructionDef::new(
        "XMOV",
        "XMOV",
        0x40,
        &[],
        &[(Field::new("", 23, 1, false, None), 0)],
        0x007fffff,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/XMOV.md",
    );

    /// `WRCFG`. `WormholeB0/TensixTile/TensixCoprocessor/WRCFG.md`, which the Blackhole tree states is shared and identical.
    pub static WRCFG: InstructionDef = InstructionDef::new(
        "WRCFG",
        "WRCFG",
        0xb0,
        &[
            Field::new("InputReg", 16, 6, false, None),
            Field::new("Is128Bit", 15, 1, false, None),
            Field::new("CfgIndex", 0, 11, false, None),
        ],
        &[],
        0x00c07800,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/WRCFG.md",
    );

    /// `RDCFG`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/RDCFG.md`.
    pub static RDCFG: InstructionDef = InstructionDef::new(
        "RDCFG",
        "RDCFG",
        0xb1,
        &[
            Field::new("ResultReg", 16, 6, false, None),
            Field::new("CfgIndex", 0, 11, false, None),
        ],
        &[],
        0x00c0f800,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/RDCFG.md",
    );

    /// `SETC16`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SETC16.md`.
    pub static SETC16: InstructionDef = InstructionDef::new(
        "SETC16",
        "SETC16",
        0xb2,
        &[
            Field::new("CfgIndex", 16, 8, false, None),
            Field::new("NewValue", 0, 16, false, None),
        ],
        &[],
        0x00000000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SETC16.md",
    );

    /// `STREAMWRCFG`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/STREAMWRCFG.md`.
    pub static STREAMWRCFG: InstructionDef = InstructionDef::new(
        "STREAMWRCFG",
        "STREAMWRCFG",
        0xb7,
        &[
            Field::new("StreamSelect", 21, 2, false, None),
            Field::new("RegIndex", 11, 10, false, None),
            Field::new("CfgIndex", 0, 11, false, None),
        ],
        &[],
        0x00800000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/STREAMWRCFG.md",
    );

    /// `CFGSHIFTMASK`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/CFGSHIFTMASK.md`.
    pub static CFGSHIFTMASK: InstructionDef = InstructionDef::new(
        "CFGSHIFTMASK",
        "CFGSHIFTMASK",
        0xb8,
        &[
            Field::new("MaskMode", 23, 1, false, None),
            Field::new("AluMode", 20, 3, false, None),
            Field::new("MaskWidth", 15, 5, false, None),
            Field::new("RotateAmt", 10, 5, false, None),
            Field::new("ScratchIndex", 8, 2, false, None),
            Field::new("CfgIndex", 0, 8, false, None),
        ],
        &[],
        0x00000000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/CFGSHIFTMASK.md",
    );

    /// `SETADC`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SETADC.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SETADC: InstructionDef = InstructionDef::new(
        "SETADC",
        "SETADC",
        0x50,
        &[
            Field::new("PK", 23, 1, false, None),
            Field::new("U1", 22, 1, false, None),
            Field::new("U0", 21, 1, false, None),
            Field::new("Channel", 20, 1, false, None),
            Field::new("XYZW", 18, 2, false, None),
            Field::new("NewValue", 0, 18, false, None),
        ],
        &[],
        0x00000000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SETADC.md",
    );

    /// `SETADCXY`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SETADCXY.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SETADCXY: InstructionDef = InstructionDef::new(
        "SETADCXY",
        "SETADCXY",
        0x51,
        &[
            Field::new("PK", 23, 1, false, None),
            Field::new("U1", 22, 1, false, None),
            Field::new("U0", 21, 1, false, None),
            Field::new("ThreadOverride", 18, 2, false, None),
            Field::new("Y1Val", 15, 3, false, None),
            Field::new("X1Val", 12, 3, false, None),
            Field::new("Y0Val", 9, 3, false, None),
            Field::new("X0Val", 6, 3, false, None),
            Field::new("Y1", 3, 1, false, None),
            Field::new("X1", 2, 1, false, None),
            Field::new("Y0", 1, 1, false, None),
            Field::new("X0", 0, 1, false, None),
        ],
        &[],
        0x00100030,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SETADCXY.md",
    );

    /// `INCADCXY`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/INCADCXY.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static INCADCXY: InstructionDef = InstructionDef::new(
        "INCADCXY",
        "INCADCXY",
        0x52,
        &[
            Field::new("PK", 23, 1, false, None),
            Field::new("U1", 22, 1, false, None),
            Field::new("U0", 21, 1, false, None),
            Field::new("ThreadOverride", 18, 2, false, None),
            Field::new("Y1Inc", 15, 3, false, None),
            Field::new("X1Inc", 12, 3, false, None),
            Field::new("Y0Inc", 9, 3, false, None),
            Field::new("X0Inc", 6, 3, false, None),
        ],
        &[],
        0x0010003f,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/INCADCXY.md",
    );

    /// `ADDRCRXY`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/ADDRCRXY.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static ADDRCRXY: InstructionDef = InstructionDef::new(
        "ADDRCRXY",
        "ADDRCRXY",
        0x53,
        &[
            Field::new("PK", 23, 1, false, None),
            Field::new("U1", 22, 1, false, None),
            Field::new("U0", 21, 1, false, None),
            Field::new("ThreadOverride", 18, 2, false, None),
            Field::new("Y1Inc", 15, 3, false, None),
            Field::new("X1Inc", 12, 3, false, None),
            Field::new("Y0Inc", 9, 3, false, None),
            Field::new("X0Inc", 6, 3, false, None),
            Field::new("Y1", 3, 1, false, None),
            Field::new("X1", 2, 1, false, None),
            Field::new("Y0", 1, 1, false, None),
            Field::new("X0", 0, 1, false, None),
        ],
        &[],
        0x00100030,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/ADDRCRXY.md",
    );

    /// `SETADCZW`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SETADCZW.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SETADCZW: InstructionDef = InstructionDef::new(
        "SETADCZW",
        "SETADCZW",
        0x54,
        &[
            Field::new("PK", 23, 1, false, None),
            Field::new("U1", 22, 1, false, None),
            Field::new("U0", 21, 1, false, None),
            Field::new("ThreadOverride", 18, 2, false, None),
            Field::new("W1Val", 15, 3, false, None),
            Field::new("Z1Val", 12, 3, false, None),
            Field::new("W0Val", 9, 3, false, None),
            Field::new("Z0Val", 6, 3, false, None),
            Field::new("W1", 3, 1, false, None),
            Field::new("Z1", 2, 1, false, None),
            Field::new("W0", 1, 1, false, None),
            Field::new("Z0", 0, 1, false, None),
        ],
        &[],
        0x00100030,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SETADCZW.md",
    );

    /// `INCADCZW`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/INCADCZW.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static INCADCZW: InstructionDef = InstructionDef::new(
        "INCADCZW",
        "INCADCZW",
        0x55,
        &[
            Field::new("PK", 23, 1, false, None),
            Field::new("U1", 22, 1, false, None),
            Field::new("U0", 21, 1, false, None),
            Field::new("ThreadOverride", 18, 2, false, None),
            Field::new("W1Inc", 15, 3, false, None),
            Field::new("Z1Inc", 12, 3, false, None),
            Field::new("W0Inc", 9, 3, false, None),
            Field::new("Z0Inc", 6, 3, false, None),
        ],
        &[],
        0x0010003f,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/INCADCZW.md",
    );

    /// `ADDRCRZW`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/ADDRCRZW.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static ADDRCRZW: InstructionDef = InstructionDef::new(
        "ADDRCRZW",
        "ADDRCRZW",
        0x56,
        &[
            Field::new("PK", 23, 1, false, None),
            Field::new("U1", 22, 1, false, None),
            Field::new("U0", 21, 1, false, None),
            Field::new("ThreadOverride", 18, 2, false, None),
            Field::new("W1Inc", 15, 3, false, None),
            Field::new("Z1Inc", 12, 3, false, None),
            Field::new("W0Inc", 9, 3, false, None),
            Field::new("Z0Inc", 6, 3, false, None),
            Field::new("W1", 3, 1, false, None),
            Field::new("Z1", 2, 1, false, None),
            Field::new("W0", 1, 1, false, None),
            Field::new("Z0", 0, 1, false, None),
        ],
        &[],
        0x00100030,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/ADDRCRZW.md",
    );

    /// `SETADCXX`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SETADCXX.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SETADCXX: InstructionDef = InstructionDef::new(
        "SETADCXX",
        "SETADCXX",
        0x5e,
        &[
            Field::new("PK", 23, 1, false, None),
            Field::new("U1", 22, 1, false, None),
            Field::new("U0", 21, 1, false, None),
            Field::new("X1Val", 10, 10, false, None),
            Field::new("X0Val", 0, 10, false, None),
        ],
        &[],
        0x00100000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SETADCXX.md",
    );

    /// `NOP`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/NOP.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static NOP: InstructionDef = InstructionDef::new(
        "NOP",
        "NOP",
        0x02,
        &[],
        &[],
        0x00ffffff,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/NOP.md",
    );

    /// `SEMPOST`. `WormholeB0/TensixTile/TensixCoprocessor/SEMPOST.md`, which the Blackhole tree states is shared and identical.
    pub static SEMPOST: InstructionDef = InstructionDef::new(
        "SEMPOST",
        "SEMPOST",
        0xa4,
        &[Field::new("SemaphoreMask", 2, 8, false, None)],
        &[],
        0x00fffc03,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SEMPOST.md",
    );

    /// `SEMGET`. `WormholeB0/TensixTile/TensixCoprocessor/SEMGET.md`, which the Blackhole tree states is shared and identical.
    pub static SEMGET: InstructionDef = InstructionDef::new(
        "SEMGET",
        "SEMGET",
        0xa5,
        &[Field::new("SemaphoreMask", 2, 8, false, None)],
        &[],
        0x00fffc03,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SEMGET.md",
    );

    /// `SETDVALID`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SETDVALID.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SETDVALID: InstructionDef = InstructionDef::new(
        "SETDVALID",
        "SETDVALID",
        0x57,
        &[
            Field::new("FlipSrcB", 1, 1, false, None),
            Field::new("FlipSrcA", 0, 1, false, None),
        ],
        &[],
        0x00fffffc,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SETDVALID.md",
    );

    /// `GATESRCRST`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/GATESRCRST.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static GATESRCRST: InstructionDef = InstructionDef::new(
        "GATESRCRST",
        "GATESRCRST",
        0x35,
        &[Field::new("InvalidateSrcBCache", 1, 1, false, None)],
        &[],
        0x00fffffd,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/GATESRCRST.md",
    );

    /// `ZEROSRC`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/ZEROSRC.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static ZEROSRC: InstructionDef = InstructionDef::new(
        "ZEROSRC",
        "ZEROSRC",
        0x11,
        &[
            Field::new("NegativeInfSrcA", 4, 1, false, None),
            Field::new("SingleBankMatrixUnit", 3, 1, false, None),
            Field::new("BothBanks", 2, 1, false, None),
            Field::new("ClearSrcB", 1, 1, false, None),
            Field::new("ClearSrcA", 0, 1, false, None),
        ],
        &[],
        0x00ffffe0,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/ZEROSRC.md",
    );

    /// `TRNSPSRCB`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/TRNSPSRCB.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static TRNSPSRCB: InstructionDef = InstructionDef::new(
        "TRNSPSRCB",
        "TRNSPSRCB",
        0x16,
        &[],
        &[],
        0x00ffffff,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/TRNSPSRCB.md",
    );

    /// `SHIFTXA`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SHIFTXA.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SHIFTXA: InstructionDef = InstructionDef::new(
        "SHIFTXA",
        "SHIFTXA",
        0x17,
        &[Field::new("Direction", 0, 2, false, None)],
        &[],
        0x00fffffc,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SHIFTXA.md",
    );

    /// `CLREXPHIST`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/CLREXPHIST.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static CLREXPHIST: InstructionDef = InstructionDef::new(
        "CLREXPHIST",
        "CLREXPHIST",
        0x21,
        &[],
        &[],
        0x00ffffff,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/CLREXPHIST.md",
    );

    /// `CLEARDVALID`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/CLEARDVALID.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static CLEARDVALID: InstructionDef = InstructionDef::new(
        "CLEARDVALID",
        "CLEARDVALID",
        0x36,
        &[
            Field::new("FlipSrcB", 23, 1, false, None),
            Field::new("FlipSrcA", 22, 1, false, None),
            Field::new("KeepReadingSameSrc", 1, 1, false, None),
            Field::new("Reset", 0, 1, false, None),
        ],
        &[],
        0x003ffffc,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/CLEARDVALID.md",
    );

    /// `SETRWC`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SETRWC.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SETRWC: InstructionDef = InstructionDef::new(
        "SETRWC",
        "SETRWC",
        0x37,
        &[
            Field::new("FlipSrcB", 23, 1, false, None),
            Field::new("FlipSrcA", 22, 1, false, None),
            Field::new("DstCtoCr", 21, 1, false, None),
            Field::new("DstCr", 20, 1, false, None),
            Field::new("SrcBCr", 19, 1, false, None),
            Field::new("SrcACr", 18, 1, false, None),
            Field::new("DstVal", 14, 4, false, None),
            Field::new("SrcBVal", 10, 4, false, None),
            Field::new("SrcAVal", 6, 4, false, None),
            Field::new("Fidelity", 3, 1, false, None),
            Field::new("Dst", 2, 1, false, None),
            Field::new("SrcB", 1, 1, false, None),
            Field::new("SrcA", 0, 1, false, None),
        ],
        &[],
        0x00000030,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SETRWC.md",
    );

    /// `INCRWC`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/INCRWC.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static INCRWC: InstructionDef = InstructionDef::new(
        "INCRWC",
        "INCRWC",
        0x38,
        &[
            Field::new("DstCr", 20, 1, false, None),
            Field::new("SrcBCr", 19, 1, false, None),
            Field::new("SrcACr", 18, 1, false, None),
            Field::new("DstInc", 14, 4, false, None),
            Field::new("SrcBInc", 10, 4, false, None),
            Field::new("SrcAInc", 6, 4, false, None),
        ],
        &[],
        0x00e0003f,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/INCRWC.md",
    );

    /// `LOADIND`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/LOADIND.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static LOADIND: InstructionDef = InstructionDef::new(
        "LOADIND",
        "LOADIND",
        0x49,
        &[
            Field::new("Size", 22, 2, false, None),
            Field::new("OffsetHalfReg", 14, 7, false, None),
            Field::new("OffsetIncrement", 12, 2, false, None),
            Field::new("ResultReg", 6, 6, false, None),
            Field::new("AddrReg", 0, 6, false, None),
        ],
        &[],
        0x00200000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/LOADIND.md",
    );

    /// `STOREIND_L1`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/STOREIND_L1.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static STOREIND_L1: InstructionDef = InstructionDef::new(
        "STOREIND_L1",
        "STOREIND",
        0x66,
        &[
            Field::new("Size", 21, 2, false, None),
            Field::new("OffsetHalfReg", 14, 7, false, None),
            Field::new("OffsetIncrement", 12, 2, false, None),
            Field::new("DataReg", 6, 6, false, None),
            Field::new("AddrReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 1)],
        0x00000000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/STOREIND_L1.md",
    );

    /// `STOREIND_MMIO`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/STOREIND_MMIO.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static STOREIND_MMIO: InstructionDef = InstructionDef::new(
        "STOREIND_MMIO",
        "STOREIND",
        0x66,
        &[
            Field::new("OffsetHalfReg", 14, 7, false, None),
            Field::new("OffsetIncrement", 12, 2, false, None),
            Field::new("DataReg", 6, 6, false, None),
            Field::new("AddrReg", 0, 6, false, None),
        ],
        &[
            (Field::new("", 22, 1, false, None), 1),
            (Field::new("", 23, 1, false, None), 0),
        ],
        0x00200000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/STOREIND_MMIO.md",
    );

    /// `STOREIND_Src`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/STOREIND_Src.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static STOREIND_Src: InstructionDef = InstructionDef::new(
        "STOREIND_Src",
        "STOREIND",
        0x66,
        &[
            Field::new("StoreToSrcB", 21, 1, false, None),
            Field::new("OffsetHalfReg", 14, 7, false, None),
            Field::new("OffsetIncrement", 12, 2, false, None),
            Field::new("DataReg", 6, 6, false, None),
            Field::new("AddrReg", 0, 6, false, None),
        ],
        &[
            (Field::new("", 22, 1, false, None), 0),
            (Field::new("", 23, 1, false, None), 0),
        ],
        0x00000000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/STOREIND_Src.md",
    );

    /// `LOADREG`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/LOADREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static LOADREG: InstructionDef = InstructionDef::new(
        "LOADREG",
        "LOADREG",
        0x68,
        &[
            Field::new("ResultReg", 18, 6, false, None),
            Field::new("AddrLo", 0, 18, false, None),
        ],
        &[],
        0x00000000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/LOADREG.md",
    );

    /// `STOREREG`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/STOREREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static STOREREG: InstructionDef = InstructionDef::new(
        "STOREREG",
        "STOREREG",
        0x67,
        &[
            Field::new("DataReg", 18, 6, false, None),
            Field::new("AddrLo", 0, 18, false, None),
        ],
        &[],
        0x00000000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/STOREREG.md",
    );

    /// `FLUSHDMA`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/FLUSHDMA.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static FLUSHDMA: InstructionDef = InstructionDef::new(
        "FLUSHDMA",
        "FLUSHDMA",
        0x46,
        &[Field::new("ConditionMask", 0, 4, false, None)],
        &[],
        0x00fffff0,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/FLUSHDMA.md",
    );

    /// `DMANOP`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/DMANOP.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static DMANOP: InstructionDef = InstructionDef::new(
        "DMANOP",
        "DMANOP",
        0x60,
        &[],
        &[],
        0x00ffffff,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/DMANOP.md",
    );

    /// `ADDDMAREG`. **`CONFIRMED`** on Blackhole, ttsim and silicon, by `crates/tt-tests/tests/step38_gpr_add.rs::adddmareg_adds_two_registers`: the layout of `WormholeB0/TensixTile/TensixCoprocessor/ADDDMAREG.md` (a Wormhole page; Blackhole has none), every field exercised.
    pub static ADDDMAREG: InstructionDef = InstructionDef::new(
        "ADDDMAREG",
        "ADDDMAREG",
        0x58,
        &[
            Field::new("ResultReg", 12, 6, false, None),
            Field::new("RightReg", 6, 6, false, None),
            Field::new("LeftReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 0)],
        0x007c0000,
        Provenance::Confirmed {
            evidence: "crates/tt-tests/tests/step38_gpr_add.rs::adddmareg_adds_two_registers",
        },
        "WormholeB0/TensixTile/TensixCoprocessor/ADDDMAREG.md",
    );

    /// `ADDDMAREGi`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/ADDDMAREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static ADDDMAREGi: InstructionDef = InstructionDef::new(
        "ADDDMAREGi",
        "ADDDMAREG",
        0x58,
        &[
            Field::new("ResultReg", 12, 6, false, None),
            Field::new("RightImm6", 6, 6, false, None),
            Field::new("LeftReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 1)],
        0x007c0000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/ADDDMAREG.md",
    );

    /// `SUBDMAREG`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SUBDMAREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SUBDMAREG: InstructionDef = InstructionDef::new(
        "SUBDMAREG",
        "SUBDMAREG",
        0x59,
        &[
            Field::new("ResultReg", 12, 6, false, None),
            Field::new("RightReg", 6, 6, false, None),
            Field::new("LeftReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 0)],
        0x007c0000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SUBDMAREG.md",
    );

    /// `SUBDMAREGi`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SUBDMAREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SUBDMAREGi: InstructionDef = InstructionDef::new(
        "SUBDMAREGi",
        "SUBDMAREG",
        0x59,
        &[
            Field::new("ResultReg", 12, 6, false, None),
            Field::new("RightImm6", 6, 6, false, None),
            Field::new("LeftReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 1)],
        0x007c0000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SUBDMAREG.md",
    );

    /// `MULDMAREG`. **`CONFIRMED`** on Blackhole, ttsim and silicon, by `crates/tt-tests/tests/step100_scalar_config.rs::scalar_register_and_immediate_semantics`: the layout of `WormholeB0/TensixTile/TensixCoprocessor/MULDMAREG.md` (a Wormhole page; Blackhole has none), every field exercised.
    pub static MULDMAREG: InstructionDef = InstructionDef::new(
        "MULDMAREG",
        "MULDMAREG",
        0x5a,
        &[Field::new("ResultReg", 12, 6, false, None), Field::new("RightReg", 6, 6, false, None), Field::new("LeftReg", 0, 6, false, None)],
        &[(Field::new("", 23, 1, false, None), 0)],
        0x007c0000,
        Provenance::Confirmed { evidence: "crates/tt-tests/tests/step100_scalar_config.rs::scalar_register_and_immediate_semantics" },
        "WormholeB0/TensixTile/TensixCoprocessor/MULDMAREG.md",
    );

    /// `MULDMAREGi`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/MULDMAREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static MULDMAREGi: InstructionDef = InstructionDef::new(
        "MULDMAREGi",
        "MULDMAREG",
        0x5a,
        &[
            Field::new("ResultReg", 12, 6, false, None),
            Field::new("RightImm6", 6, 6, false, None),
            Field::new("LeftReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 1)],
        0x007c0000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/MULDMAREG.md",
    );

    /// `BITWOPDMAREG`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/BITWOPDMAREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static BITWOPDMAREG: InstructionDef = InstructionDef::new(
        "BITWOPDMAREG",
        "BITWOPDMAREG",
        0x5b,
        &[
            Field::new("Mode", 18, 3, false, None),
            Field::new("ResultReg", 12, 6, false, None),
            Field::new("RightReg", 6, 6, false, None),
            Field::new("LeftReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 0)],
        0x00600000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/BITWOPDMAREG.md",
    );

    /// `BITWOPDMAREGi`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/BITWOPDMAREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static BITWOPDMAREGi: InstructionDef = InstructionDef::new(
        "BITWOPDMAREGi",
        "BITWOPDMAREG",
        0x5b,
        &[
            Field::new("Mode", 18, 3, false, None),
            Field::new("ResultReg", 12, 6, false, None),
            Field::new("RightImm6", 6, 6, false, None),
            Field::new("LeftReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 1)],
        0x00600000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/BITWOPDMAREG.md",
    );

    /// `SHIFTDMAREG`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SHIFTDMAREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SHIFTDMAREG: InstructionDef = InstructionDef::new(
        "SHIFTDMAREG",
        "SHIFTDMAREG",
        0x5c,
        &[
            Field::new("Mode", 18, 3, false, None),
            Field::new("ResultReg", 12, 6, false, None),
            Field::new("RightReg", 6, 6, false, None),
            Field::new("LeftReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 0)],
        0x00600000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SHIFTDMAREG.md",
    );

    /// `SHIFTDMAREGi`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/SHIFTDMAREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static SHIFTDMAREGi: InstructionDef = InstructionDef::new(
        "SHIFTDMAREGi",
        "SHIFTDMAREG",
        0x5c,
        &[
            Field::new("Mode", 18, 3, false, None),
            Field::new("ResultReg", 12, 6, false, None),
            Field::new("RightImm5", 6, 5, false, None),
            Field::new("LeftReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 1)],
        0x00600800,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/SHIFTDMAREG.md",
    );

    /// `CMPDMAREG`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/CMPDMAREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static CMPDMAREG: InstructionDef = InstructionDef::new(
        "CMPDMAREG",
        "CMPDMAREG",
        0x5d,
        &[
            Field::new("Mode", 18, 3, false, None),
            Field::new("ResultReg", 12, 6, false, None),
            Field::new("RightReg", 6, 6, false, None),
            Field::new("LeftReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 0)],
        0x00600000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/CMPDMAREG.md",
    );

    /// `CMPDMAREGi`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/CMPDMAREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static CMPDMAREGi: InstructionDef = InstructionDef::new(
        "CMPDMAREGi",
        "CMPDMAREG",
        0x5d,
        &[
            Field::new("Mode", 18, 3, false, None),
            Field::new("ResultReg", 12, 6, false, None),
            Field::new("RightImm6", 6, 6, false, None),
            Field::new("LeftReg", 0, 6, false, None),
        ],
        &[(Field::new("", 23, 1, false, None), 1)],
        0x00600000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/CMPDMAREG.md",
    );

    /// `ATGETM`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/ATGETM.md`.
    pub static ATGETM: InstructionDef = InstructionDef::new(
        "ATGETM",
        "ATGETM",
        0xa0,
        &[Field::new("Index", 0, 16, false, None)],
        &[],
        0x00ff0000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/ATGETM.md",
    );

    /// `ATRELM`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/ATRELM.md`.
    pub static ATRELM: InstructionDef = InstructionDef::new(
        "ATRELM",
        "ATRELM",
        0xa1,
        &[Field::new("Index", 0, 16, false, None)],
        &[],
        0x00ff0000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/ATRELM.md",
    );

    /// `ATINCGET`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/ATINCGET.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static ATINCGET: InstructionDef = InstructionDef::new(
        "ATINCGET",
        "ATINCGET",
        0x61,
        &[
            Field::new("IntWidth", 14, 5, false, None),
            Field::new("Ofs", 12, 2, false, None),
            Field::new("InOutReg", 6, 6, false, None),
            Field::new("AddrReg", 0, 6, false, None),
        ],
        &[],
        0x00f80000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/ATINCGET.md",
    );

    /// `ATINCGETPTR`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/ATINCGETPTR.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static ATINCGETPTR: InstructionDef = InstructionDef::new(
        "ATINCGETPTR",
        "ATINCGETPTR",
        0x62,
        &[
            Field::new("NoIncr", 22, 1, false, None),
            Field::new("IncrLog2", 18, 4, false, None),
            Field::new("IntWidth", 14, 4, false, None),
            Field::new("Ofs", 12, 2, false, None),
            Field::new("ResultReg", 6, 6, false, None),
            Field::new("AddrReg", 0, 6, false, None),
        ],
        &[],
        0x00800000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/ATINCGETPTR.md",
    );

    /// `SFPLOAD_BH`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md`.
    pub static SFPLOAD: InstructionDef = InstructionDef::new(
        "SFPLOAD_BH",
        "SFPLOAD",
        0x70,
        &[
            Field::new("VD", 20, 4, false, None),
            Field::new("Mod0", 16, 4, false, None),
            Field::new("AddrMod", 13, 3, false, None),
            Field::new("Imm10", 0, 10, false, None),
        ],
        &[],
        0x00001c00,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md",
    );

    /// `SFPLOADMACRO_BH`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPLOADMACRO.md`.
    pub static SFPLOADMACRO: InstructionDef = InstructionDef::new(
        "SFPLOADMACRO_BH",
        "SFPLOADMACRO",
        0x93,
        &[
            Field::new("MacroIndex", 22, 2, false, None),
            Field::new("VDLo", 20, 2, false, None),
            Field::new("Mod0", 16, 4, false, None),
            Field::new("AddrMod", 13, 3, false, None),
            Field::new("Imm9", 1, 9, false, None),
            Field::new("VDHi", 0, 1, false, None),
        ],
        &[],
        0x00001c00,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPLOADMACRO.md",
    );

    /// `SFPSTORE_BH`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPSTORE.md`.
    pub static SFPSTORE: InstructionDef = InstructionDef::new(
        "SFPSTORE_BH",
        "SFPSTORE",
        0x72,
        &[
            Field::new("VD", 20, 4, false, None),
            Field::new("Mod0", 16, 4, false, None),
            Field::new("AddrMod", 13, 3, false, None),
            Field::new("Imm10", 0, 10, false, None),
        ],
        &[],
        0x00001c00,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPSTORE.md",
    );

    /// `MOP`. **`CONFIRMED`** on Blackhole, ttsim and silicon, by `crates/tt-tests/tests/step36_mop.rs::a_mop_expands_as_the_page_models_it`: the layout of `WormholeB0/TensixTile/TensixCoprocessor/MOP.md` (a Wormhole page; Blackhole has none), every field exercised.
    pub static MOP: InstructionDef = InstructionDef::new(
        "MOP",
        "MOP",
        0x01,
        &[
            Field::new("Template", 23, 1, false, None),
            Field::new("Count1", 16, 7, false, None),
            Field::new("MaskLo", 0, 16, false, None),
        ],
        &[],
        0x00000000,
        Provenance::Confirmed {
            evidence: "crates/tt-tests/tests/step36_mop.rs::a_mop_expands_as_the_page_models_it",
        },
        "WormholeB0/TensixTile/TensixCoprocessor/MOP.md",
    );

    /// `MOP_CFG`. **`CONFIRMED`** on Blackhole, ttsim and silicon, by `crates/tt-tests/tests/step36_mop.rs::a_mop_expands_as_the_page_models_it`: the layout of `WormholeB0/TensixTile/TensixCoprocessor/MOP_CFG.md` (a Wormhole page; Blackhole has none), every field exercised.
    pub static MOP_CFG: InstructionDef = InstructionDef::new(
        "MOP_CFG",
        "MOP_CFG",
        0x03,
        &[Field::new("MaskHi", 0, 16, false, None)],
        &[],
        0x00ff0000,
        Provenance::Confirmed {
            evidence: "crates/tt-tests/tests/step36_mop.rs::a_mop_expands_as_the_page_models_it",
        },
        "WormholeB0/TensixTile/TensixCoprocessor/MOP_CFG.md",
    );

    /// `SFPLOADI`. `WormholeB0/TensixTile/TensixCoprocessor/SFPLOADI.md`, which the Blackhole tree states is shared and identical.
    pub static SFPLOADI: InstructionDef = InstructionDef::new(
        "SFPLOADI",
        "SFPLOADI",
        0x71,
        &[
            Field::new("VD", 20, 4, false, None),
            Field::new("Mod0", 16, 4, false, None),
            Field::new("Imm16", 0, 16, false, None),
        ],
        &[],
        0x00000000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPLOADI.md",
    );

    /// `SFPIADD`. `WormholeB0/TensixTile/TensixCoprocessor/SFPIADD.md`, which the Blackhole tree states is shared and identical.
    pub static SFPIADD: InstructionDef = InstructionDef::new(
        "SFPIADD",
        "SFPIADD",
        0x79,
        &[
            Field::new("Imm12", 12, 12, true, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00000000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPIADD.md",
    );

    /// `SFPSWAP`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPSWAP.md`.
    pub static SFPSWAP: InstructionDef = InstructionDef::new(
        "SFPSWAP",
        "SFPSWAP",
        0x92,
        &[
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00fff000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPSWAP.md",
    );

    /// `SFPCONFIG`. `WormholeB0/TensixTile/TensixCoprocessor/SFPCONFIG.md`, which the Blackhole tree states is shared and identical.
    pub static SFPCONFIG: InstructionDef = InstructionDef::new(
        "SFPCONFIG",
        "SFPCONFIG",
        0x91,
        &[
            Field::new("Imm16", 8, 16, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00000000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPCONFIG.md",
    );

    /// `SFPMAD`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPMAD.md`.
    pub static SFPMAD: InstructionDef = InstructionDef::new(
        "SFPMAD",
        "SFPMAD",
        0x84,
        &[
            Field::new("VA", 16, 4, false, None),
            Field::new("VB", 12, 4, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00f00000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPMAD.md",
    );

    /// `SFPADD`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPADD.md`.
    pub static SFPADD: InstructionDef = InstructionDef::new(
        "SFPADD",
        "SFPADD",
        0x85,
        &[
            Field::new("VA", 16, 4, false, None),
            Field::new("VB", 12, 4, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00f00000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPADD.md",
    );

    /// `SFPMUL`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPMUL.md`.
    pub static SFPMUL: InstructionDef = InstructionDef::new(
        "SFPMUL",
        "SFPMUL",
        0x86,
        &[
            Field::new("VA", 16, 4, false, None),
            Field::new("VB", 12, 4, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00f00000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPMUL.md",
    );

    /// `SFPLUT`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPLUT.md`.
    pub static SFPLUT: InstructionDef = InstructionDef::new(
        "SFPLUT",
        "SFPLUT",
        0x73,
        &[
            Field::new("VD", 20, 4, false, None),
            Field::new("Mod0", 16, 4, false, None),
        ],
        &[],
        0x0000ffff,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPLUT.md",
    );

    /// `SFPMULI`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPMULI.md`.
    pub static SFPMULI: InstructionDef = InstructionDef::new(
        "SFPMULI",
        "SFPMULI",
        0x74,
        &[
            Field::new("Imm16", 8, 16, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00000000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPMULI.md",
    );

    /// `SFPADDI`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPADDI.md`.
    pub static SFPADDI: InstructionDef = InstructionDef::new(
        "SFPADDI",
        "SFPADDI",
        0x75,
        &[
            Field::new("Imm16", 8, 16, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00000000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPADDI.md",
    );

    /// `SFPDIVP2`. `WormholeB0/TensixTile/TensixCoprocessor/SFPDIVP2.md`, which the Blackhole tree states is shared and identical.
    pub static SFPDIVP2: InstructionDef = InstructionDef::new(
        "SFPDIVP2",
        "SFPDIVP2",
        0x76,
        &[
            Field::new("Imm8", 12, 8, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00f00000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPDIVP2.md",
    );

    /// `SFPEXEXP`. `WormholeB0/TensixTile/TensixCoprocessor/SFPEXEXP.md`, which the Blackhole tree states is shared and identical.
    pub static SFPEXEXP: InstructionDef = InstructionDef::new(
        "SFPEXEXP",
        "SFPEXEXP",
        0x77,
        &[
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00fff000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPEXEXP.md",
    );

    /// `SFPEXMAN`. `WormholeB0/TensixTile/TensixCoprocessor/SFPEXMAN.md`, which the Blackhole tree states is shared and identical.
    pub static SFPEXMAN: InstructionDef = InstructionDef::new(
        "SFPEXMAN",
        "SFPEXMAN",
        0x78,
        &[
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00fff000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPEXMAN.md",
    );

    /// `SFPSHFT`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPSHFT.md`.
    pub static SFPSHFT: InstructionDef = InstructionDef::new(
        "SFPSHFT",
        "SFPSHFT",
        0x7a,
        &[
            Field::new("Imm12", 12, 12, true, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00000000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPSHFT.md",
    );

    /// `SFPSETCC`. `WormholeB0/TensixTile/TensixCoprocessor/SFPSETCC.md`, which the Blackhole tree states is shared and identical.
    pub static SFPSETCC: InstructionDef = InstructionDef::new(
        "SFPSETCC",
        "SFPSETCC",
        0x7b,
        &[
            Field::new("Imm1", 12, 1, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00ffe000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPSETCC.md",
    );

    /// `SFPMOV`. `WormholeB0/TensixTile/TensixCoprocessor/SFPMOV.md`, which the Blackhole tree states is shared and identical.
    pub static SFPMOV: InstructionDef = InstructionDef::new(
        "SFPMOV",
        "SFPMOV",
        0x7c,
        &[
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00fff000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPMOV.md",
    );

    /// `SFPABS`. `WormholeB0/TensixTile/TensixCoprocessor/SFPABS.md`, which the Blackhole tree states is shared and identical.
    pub static SFPABS: InstructionDef = InstructionDef::new(
        "SFPABS",
        "SFPABS",
        0x7d,
        &[
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00fff000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPABS.md",
    );

    /// `SFPAND_BH`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPAND.md`.
    pub static SFPAND: InstructionDef = InstructionDef::new(
        "SFPAND_BH",
        "SFPAND",
        0x7e,
        &[
            Field::new("VB", 12, 4, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00ff0000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPAND.md",
    );

    /// `SFPOR_BH`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPOR.md`.
    pub static SFPOR: InstructionDef = InstructionDef::new(
        "SFPOR_BH",
        "SFPOR",
        0x7f,
        &[
            Field::new("VB", 12, 4, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00ff0000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPOR.md",
    );

    /// `SFPNOT`. `WormholeB0/TensixTile/TensixCoprocessor/SFPNOT.md`, which the Blackhole tree states is shared and identical.
    pub static SFPNOT: InstructionDef = InstructionDef::new(
        "SFPNOT",
        "SFPNOT",
        0x80,
        &[
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
        ],
        &[],
        0x00fff00f,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPNOT.md",
    );

    /// `SFPLZ`. `WormholeB0/TensixTile/TensixCoprocessor/SFPLZ.md`, which the Blackhole tree states is shared and identical.
    pub static SFPLZ: InstructionDef = InstructionDef::new(
        "SFPLZ",
        "SFPLZ",
        0x81,
        &[
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00fff000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPLZ.md",
    );

    /// `SFPSETEXP`. `WormholeB0/TensixTile/TensixCoprocessor/SFPSETEXP.md`, which the Blackhole tree states is shared and identical.
    pub static SFPSETEXP: InstructionDef = InstructionDef::new(
        "SFPSETEXP",
        "SFPSETEXP",
        0x82,
        &[
            Field::new("Imm8", 12, 8, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00f00000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPSETEXP.md",
    );

    /// `SFPSETMAN`. `WormholeB0/TensixTile/TensixCoprocessor/SFPSETMAN.md`, which the Blackhole tree states is shared and identical.
    pub static SFPSETMAN: InstructionDef = InstructionDef::new(
        "SFPSETMAN",
        "SFPSETMAN",
        0x83,
        &[
            Field::new("Imm12", 12, 12, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00000000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPSETMAN.md",
    );

    /// `SFPPUSHC_BH`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPPUSHC.md`.
    pub static SFPPUSHC: InstructionDef = InstructionDef::new(
        "SFPPUSHC_BH",
        "SFPPUSHC",
        0x87,
        &[
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00ffff00,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPPUSHC.md",
    );

    /// `SFPPOPC`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPPOPC.md`.
    pub static SFPPOPC: InstructionDef = InstructionDef::new(
        "SFPPOPC",
        "SFPPOPC",
        0x88,
        &[
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00ffff00,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPPOPC.md",
    );

    /// `SFPSETSGN`. `WormholeB0/TensixTile/TensixCoprocessor/SFPSETSGN.md`, which the Blackhole tree states is shared and identical.
    pub static SFPSETSGN: InstructionDef = InstructionDef::new(
        "SFPSETSGN",
        "SFPSETSGN",
        0x89,
        &[
            Field::new("Imm1", 12, 1, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00ffe000,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPSETSGN.md",
    );

    /// `SFPENCC`. `WormholeB0/TensixTile/TensixCoprocessor/SFPENCC.md`, which the Blackhole tree states is shared and identical.
    pub static SFPENCC: InstructionDef = InstructionDef::new(
        "SFPENCC",
        "SFPENCC",
        0x8a,
        &[
            Field::new("Imm2", 12, 2, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00ffcf00,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPENCC.md",
    );

    /// `SFPCOMPC`. `WormholeB0/TensixTile/TensixCoprocessor/SFPCOMPC.md`, which the Blackhole tree states is shared and identical.
    pub static SFPCOMPC: InstructionDef = InstructionDef::new(
        "SFPCOMPC",
        "SFPCOMPC",
        0x8b,
        &[Field::new("VD", 4, 4, false, None)],
        &[],
        0x00ffff0f,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPCOMPC.md",
    );

    /// `SFPTRANSP`. `WormholeB0/TensixTile/TensixCoprocessor/SFPTRANSP.md`, which the Blackhole tree states is shared and identical.
    pub static SFPTRANSP: InstructionDef = InstructionDef::new(
        "SFPTRANSP",
        "SFPTRANSP",
        0x8c,
        &[Field::new("VD", 4, 4, false, None)],
        &[],
        0x00ffff0f,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPTRANSP.md",
    );

    /// `SFPXOR`. `WormholeB0/TensixTile/TensixCoprocessor/SFPXOR.md`, which the Blackhole tree states is shared and identical.
    pub static SFPXOR: InstructionDef = InstructionDef::new(
        "SFPXOR",
        "SFPXOR",
        0x8d,
        &[
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
        ],
        &[],
        0x00fff00f,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPXOR.md",
    );

    /// `SFPSTOCHRND_BH`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md`.
    pub static SFPSTOCHRND: InstructionDef = InstructionDef::new(
        "SFPSTOCHRND_BH",
        "SFP_STOCH_RND",
        0x8e,
        &[
            Field::new("RoundingMode", 21, 2, false, None),
            Field::new("VB", 12, 4, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 3, false, None),
        ],
        &[],
        0x009f0008,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md",
    );

    /// `SFPSTOCHRNDi_BH`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_IntInt.md`.
    pub static SFPSTOCHRNDi: InstructionDef = InstructionDef::new(
        "SFPSTOCHRNDi_BH",
        "SFP_STOCH_RND",
        0x8e,
        &[
            Field::new("RoundingMode", 21, 2, false, None),
            Field::new("Imm5", 16, 5, false, None),
            Field::new("VB", 12, 4, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("UseImm5", 3, 1, false, None),
            Field::new("Mod1", 0, 3, false, None),
        ],
        &[],
        0x00800000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_IntInt.md",
    );

    /// `SFPNOP`. `WormholeB0/TensixTile/TensixCoprocessor/SFPNOP.md`, which the Blackhole tree states is shared and identical.
    pub static SFPNOP: InstructionDef = InstructionDef::new(
        "SFPNOP",
        "SFPNOP",
        0x8f,
        &[],
        &[(Field::new("", 7, 1, false, None), 0)],
        0x00ffff7f,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/SFPNOP.md",
    );

    /// `SFPCAST`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPCAST_IntAbs.md`.
    pub static SFPCAST: InstructionDef = InstructionDef::new(
        "SFPCAST",
        "SFPCAST",
        0x90,
        &[
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00fff000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPCAST_IntAbs.md",
    );

    /// `SFPLE`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPLE.md`.
    pub static SFPLE: InstructionDef = InstructionDef::new(
        "SFPLE",
        "SFPLE",
        0x96,
        &[
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00fff000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPLE.md",
    );

    /// `SFPGT`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPGT.md`.
    pub static SFPGT: InstructionDef = InstructionDef::new(
        "SFPGT",
        "SFPGT",
        0x97,
        &[
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00fff000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPGT.md",
    );

    /// `SFPMUL24`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPMUL24.md`.
    pub static SFPMUL24: InstructionDef = InstructionDef::new(
        "SFPMUL24",
        "SFPMUL24",
        0x98,
        &[
            Field::new("VA", 16, 4, false, None),
            Field::new("VB", 12, 4, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00f00000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPMUL24.md",
    );

    /// `SFPARECIP`. Documented for Blackhole in `BlackholeA0/TensixTile/TensixCoprocessor/SFPARECIP.md`.
    pub static SFPARECIP: InstructionDef = InstructionDef::new(
        "SFPARECIP",
        "SFPARECIP",
        0x99,
        &[
            Field::new("VB", 12, 4, false, None),
            Field::new("VC", 8, 4, false, None),
            Field::new("VD", 4, 4, false, None),
            Field::new("Mod1", 0, 4, false, None),
        ],
        &[],
        0x00ff0000,
        Provenance::Blackhole,
        "BlackholeA0/TensixTile/TensixCoprocessor/SFPARECIP.md",
    );

    /// `UNPACR_Regular`. `WormholeB0/TensixTile/TensixCoprocessor/UNPACR_Regular.md`, which the Blackhole tree states is shared and identical.
    pub static UNPACR_Regular: InstructionDef = InstructionDef::new(
        "UNPACR_Regular",
        "UNPACR",
        0x42,
        &[
            Field::new("WhichUnpacker", 23, 1, false, None),
            Field::new("Ch1YInc", 21, 2, false, None),
            Field::new("Ch1ZInc", 19, 2, false, None),
            Field::new("Ch0YInc", 17, 2, false, None),
            Field::new("Ch0ZInc", 15, 2, false, None),
            Field::new("ContextNumber", 10, 3, false, None),
            Field::new("ContextADC", 8, 2, false, None),
            Field::new("MultiContextMode", 7, 1, false, None),
            Field::new("FlipSrc", 6, 1, false, None),
            Field::new("AllDatumsAreZero", 4, 1, false, None),
            Field::new("UseContextCounter", 3, 1, false, None),
            Field::new("RowSearch", 2, 1, false, None),
        ],
        &[
            (Field::new("", 1, 1, false, None), 0),
            (Field::new("", 13, 1, false, None), 0),
        ],
        0x00004021,
        Provenance::SharedWithWormhole,
        "WormholeB0/TensixTile/TensixCoprocessor/UNPACR_Regular.md",
    );

    /// `UNPACR_IncrementContextCounter`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/UNPACR_IncrementContextCounter.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static UNPACR_IncrementContextCounter: InstructionDef = InstructionDef::new(
        "UNPACR_IncrementContextCounter",
        "UNPACR",
        0x42,
        &[Field::new("WhichUnpacker", 23, 1, false, None)],
        &[
            (Field::new("", 1, 1, false, None), 0),
            (Field::new("", 13, 1, false, None), 1),
        ],
        0x007fdffd,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/UNPACR_IncrementContextCounter.md",
    );

    /// `UNPACR_FlushCache`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/UNPACR_FlushCache.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static UNPACR_FlushCache: InstructionDef = InstructionDef::new(
        "UNPACR_FlushCache",
        "UNPACR",
        0x42,
        &[
            Field::new("WhichUnpacker", 23, 1, false, None),
            Field::new("MultiContextMode", 7, 1, false, None),
        ],
        &[(Field::new("", 1, 1, false, None), 1)],
        0x007fff7d,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/UNPACR_FlushCache.md",
    );

    /// `UNPACR_NOP_OverlayClear0`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_OverlayClear.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static UNPACR_NOP_OverlayClear0: InstructionDef = InstructionDef::new(
        "UNPACR_NOP_OverlayClear0",
        "UNPACR_NOP",
        0x43,
        &[Field::new("WhichUnpacker", 23, 1, false, None)],
        &[(Field::new("", 0, 3, false, None), 0)],
        0x007ffff8,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_OverlayClear.md",
    );

    /// `UNPACR_NOP_OverlayClear3`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_OverlayClear.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static UNPACR_NOP_OverlayClear3: InstructionDef = InstructionDef::new(
        "UNPACR_NOP_OverlayClear3",
        "UNPACR_NOP",
        0x43,
        &[
            Field::new("WhichUnpacker", 23, 1, false, None),
            Field::new("WhichStream", 16, 6, false, None),
            Field::new("ClearCount", 4, 11, false, None),
        ],
        &[(Field::new("", 0, 3, false, None), 3)],
        0x00408008,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_OverlayClear.md",
    );

    /// `UNPACR_NOP_Nop`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_Nop.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static UNPACR_NOP_Nop: InstructionDef = InstructionDef::new(
        "UNPACR_NOP_Nop",
        "UNPACR_NOP",
        0x43,
        &[Field::new("WhichUnpacker", 23, 1, false, None)],
        &[(Field::new("", 0, 3, false, None), 2)],
        0x007ffff8,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_Nop.md",
    );

    /// `UNPACR_NOP_SETREG`. **`UNVERIFIED`.** `WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_SETREG.md` is a Wormhole page and Blackhole has none, so this layout is a hypothesis until silicon or the simulator confirms it.
    pub static UNPACR_NOP_SETREG: InstructionDef = InstructionDef::new(
        "UNPACR_NOP_SETREG",
        "UNPACR_NOP",
        0x43,
        &[
            Field::new("WhichUnpacker", 23, 1, false, None),
            Field::new("AddrSel", 22, 1, false, None),
            Field::new("AddrMid", 16, 6, false, None),
            Field::new("Value11", 4, 11, false, None),
            Field::new("Accumulate", 3, 1, false, None),
        ],
        &[(Field::new("", 0, 3, false, None), 4)],
        0x00008000,
        Provenance::WormholeOnly,
        "WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_SETREG.md",
    );

    /// `UNPACR_NOP_SETDVALID_BH`. **`MEASURED`** on Blackhole silicon by `crates/tt-tests/tests/step104_unpacker_handover.rs::llk_nonclearing_dvalid_a, crates/tt-tests/tests/step104_unpacker_handover.rs::llk_nonclearing_dvalid_b`. Replaces Wormhole fixed mode 7 with the bounded non-clearing publication profile 0x1e9; only WhichUnpacker varies. C1/C2 retirement and C5/C6 ownership waits remain required. ttsim refuses this format-selector flavor.
    pub static UNPACR_NOP_SETDVALID: InstructionDef = InstructionDef::new(
        "UNPACR_NOP_SETDVALID_BH",
        "UNPACR_NOP",
        0x43,
        &[Field::new("WhichUnpacker", 23, 1, false, None)],
        &[(Field::new("", 0, 9, false, None), 489)],
        0x007ffe00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step104_unpacker_handover.rs::llk_nonclearing_dvalid_a, crates/tt-tests/tests/step104_unpacker_handover.rs::llk_nonclearing_dvalid_b", moved: &[], dropped: &[], widened: &[] },
        "WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_SETDVALID.md",
    );

    /// `UNPACR_NOP_ZEROSRC_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step103_source_banks.rs::unpacr_zero_blackhole_bank_and_clear_value_fields, crates/tt-tests/tests/step103_source_banks.rs::unpacr_zero_waits_on_current_unpacker_bank_with_matrix_bank_held`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_ZEROSRC.md`), and on Blackhole `BothBanks`, `WaitLikeUnpacr` sit elsewhere. `NegativeInfSrcA` has a different width on Blackhole. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static UNPACR_NOP_ZEROSRC: InstructionDef = InstructionDef::new(
        "UNPACR_NOP_ZEROSRC_BH",
        "UNPACR_NOP",
        0x43,
        &[Field::new("WhichUnpacker", 23, 1, false, None), Field::new("WaitLikeUnpacr", 5, 1, false, None), Field::new("BothBanks", 4, 1, false, None), Field::new("NegativeInfSrcA", 2, 2, false, None)],
        &[(Field::new("", 0, 2, false, None), 1), (Field::new("", 6, 1, false, None), 0)],
        0x007fff80,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step103_source_banks.rs::unpacr_zero_blackhole_bank_and_clear_value_fields, crates/tt-tests/tests/step103_source_banks.rs::unpacr_zero_waits_on_current_unpacker_bank_with_matrix_bank_held", moved: &["BothBanks", "WaitLikeUnpacr"], dropped: &[], widened: &["NegativeInfSrcA"] },
        "WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_ZEROSRC.md",
    );

    /// `GMPOOL_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step75_fpu_pooling.rs::matrix_pooling_uses_explicit_weights_and_releases_banks`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/GMPOOL.md`), and on Blackhole  sit elsewhere. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static GMPOOL: InstructionDef = InstructionDef::new(
        "GMPOOL_BH",
        "GMPOOL",
        0x33,
        &[Field::new("FlipSrcB", 23, 1, false, None), Field::new("FlipSrcA", 22, 1, false, None), Field::new("AddrMod", 15, 2, false, None), Field::new("ArgMax", 14, 1, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[(Field::new("", 19, 1, false, None), 1)],
        0x00363c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step75_fpu_pooling.rs::matrix_pooling_uses_explicit_weights_and_releases_banks", moved: &[], dropped: &[], widened: &[] },
        "WormholeB0/TensixTile/TensixCoprocessor/GMPOOL.md",
    );

    /// `GAPOOL_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step75_fpu_pooling.rs::matrix_pooling_uses_explicit_weights_and_releases_banks`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/GAPOOL.md`), and on Blackhole  sit elsewhere. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static GAPOOL: InstructionDef = InstructionDef::new(
        "GAPOOL_BH",
        "GAPOOL",
        0x34,
        &[Field::new("FlipSrcB", 23, 1, false, None), Field::new("FlipSrcA", 22, 1, false, None), Field::new("AddrMod", 15, 2, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[(Field::new("", 19, 1, false, None), 1)],
        0x00367c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step75_fpu_pooling.rs::matrix_pooling_uses_explicit_weights_and_releases_banks", moved: &[], dropped: &[], widened: &[] },
        "WormholeB0/TensixTile/TensixCoprocessor/GAPOOL.md",
    );

    /// `MVMUL_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step9_matmul.rs::mvmul_addr_mod_sits_one_bit_lower_on_blackhole`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md`), and on Blackhole `AddrMod` sits elsewhere. `AddrMod` has a different width on Blackhole. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static MVMUL: InstructionDef = InstructionDef::new(
        "MVMUL_BH",
        "MVMUL",
        0x26,
        &[Field::new("FlipSrcB", 23, 1, false, None), Field::new("FlipSrcA", 22, 1, false, None), Field::new("BroadcastSrcBRow", 19, 1, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[],
        0x00363c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step9_matmul.rs::mvmul_addr_mod_sits_one_bit_lower_on_blackhole", moved: &["AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md",
    );

    /// `MOVA2D_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/probe_src.rs::mov_to_dst_addr_mod_sits_one_bit_lower_on_blackhole`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/MOVA2D.md`), and on Blackhole `AddrMod` sits elsewhere. `AddrMod` has a different width on Blackhole. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static MOVA2D: InstructionDef = InstructionDef::new(
        "MOVA2D_BH",
        "MOVA2D",
        0x12,
        &[Field::new("UseDst32bLo", 23, 1, false, None), Field::new("SrcRow", 17, 6, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("Move8Rows", 13, 1, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[],
        0x00001c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/probe_src.rs::mov_to_dst_addr_mod_sits_one_bit_lower_on_blackhole", moved: &["AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/MOVA2D.md",
    );

    /// `MOVB2D_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/probe_src.rs::movb2d_move4_rows_is_bit_13_on_blackhole, crates/tt-tests/tests/probe_src.rs::mov_to_dst_addr_mod_sits_one_bit_lower_on_blackhole`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/MOVB2D.md`), and on Blackhole `BroadcastCol0`, `Broadcast1RowTo8`, `Move4Rows`, `AddrMod` sit elsewhere. `AddrMod` has a different width on Blackhole. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static MOVB2D: InstructionDef = InstructionDef::new(
        "MOVB2D_BH",
        "MOVB2D",
        0x13,
        &[Field::new("UseDst32bLo", 23, 1, false, None), Field::new("SrcRow", 17, 6, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("Move4Rows", 13, 1, false, None), Field::new("Broadcast1RowTo8", 12, 1, false, None), Field::new("BroadcastCol0", 11, 1, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[],
        0x00000400,
        Provenance::Measured { evidence: "crates/tt-tests/tests/probe_src.rs::movb2d_move4_rows_is_bit_13_on_blackhole, crates/tt-tests/tests/probe_src.rs::mov_to_dst_addr_mod_sits_one_bit_lower_on_blackhole", moved: &["BroadcastCol0", "Broadcast1RowTo8", "Move4Rows", "AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/MOVB2D.md",
    );

    /// `MOVD2A_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step9_matmul.rs::mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/MOVD2A.md`), and on Blackhole `AddrMod` sits elsewhere. `AddrMod` has a different width on Blackhole. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static MOVD2A: InstructionDef = InstructionDef::new(
        "MOVD2A_BH",
        "MOVD2A",
        0x08,
        &[Field::new("UseDst32bLo", 23, 1, false, None), Field::new("SrcRow", 17, 6, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("Move4Rows", 13, 1, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[],
        0x00001c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step9_matmul.rs::mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole", moved: &["AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/MOVD2A.md",
    );

    /// `MOVD2B_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step9_matmul.rs::mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/MOVD2B.md`), and on Blackhole `AddrMod` sits elsewhere. `AddrMod` has a different width on Blackhole. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static MOVD2B: InstructionDef = InstructionDef::new(
        "MOVD2B_BH",
        "MOVD2B",
        0x0a,
        &[Field::new("UseDst32bLo", 23, 1, false, None), Field::new("SrcRow", 17, 6, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("Move4Rows", 13, 1, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[],
        0x00001c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step9_matmul.rs::mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole", moved: &["AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/MOVD2B.md",
    );

    /// `MOVB2A_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step9_matmul.rs::mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/MOVB2A.md`), and on Blackhole `AddrMod` sits elsewhere. `AddrMod` has a different width on Blackhole. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static MOVB2A: InstructionDef = InstructionDef::new(
        "MOVB2A_BH",
        "MOVB2A",
        0x0b,
        &[Field::new("SrcARow", 17, 6, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("Move4Rows", 13, 1, false, None), Field::new("SrcBRow", 0, 6, false, None)],
        &[],
        0x00801fc0,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step9_matmul.rs::mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole", moved: &["AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/MOVB2A.md",
    );

    /// `ELWADD_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step9_matmul.rs::matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole, crates/tt-tests/tests/step9_matmul.rs::elw_broadcast_assignment_and_destination_fields, crates/tt-tests/tests/step90_matrix_eltwise.rs::resident_matrix_arithmetic_and_rhs_broadcasts`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/ELWADD.md`), and on Blackhole `AddrMod` sits elsewhere. `AddrMod` has a different width on Blackhole. Other fields are carried from that diagram. Step9 and step90 validate broadcast, assignment, Dst addressing and repeated bank release on ttsim and both Blackhole cards; see silicon run 1791254100 and silicon-operating-notes.md. Floating arithmetic is not IEEE754.
    pub static ELWADD: InstructionDef = InstructionDef::new(
        "ELWADD_BH",
        "ELWADD",
        0x28,
        &[Field::new("FlipSrcB", 23, 1, false, None), Field::new("FlipSrcA", 22, 1, false, None), Field::new("AddDst", 21, 1, false, None), Field::new("BroadcastSrcBRow", 20, 1, false, None), Field::new("BroadcastSrcBCol0", 19, 1, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[],
        0x00063c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step9_matmul.rs::matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole, crates/tt-tests/tests/step9_matmul.rs::elw_broadcast_assignment_and_destination_fields, crates/tt-tests/tests/step90_matrix_eltwise.rs::resident_matrix_arithmetic_and_rhs_broadcasts", moved: &["AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/ELWADD.md",
    );

    /// `ELWSUB_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step9_matmul.rs::matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole, crates/tt-tests/tests/step9_matmul.rs::elw_broadcast_assignment_and_destination_fields, crates/tt-tests/tests/step90_matrix_eltwise.rs::resident_matrix_arithmetic_and_rhs_broadcasts`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/ELWSUB.md`), and on Blackhole `AddrMod` sits elsewhere. `AddrMod` has a different width on Blackhole. Other fields are carried from that diagram. Step9 and step90 validate broadcast, assignment, Dst addressing and repeated bank release on ttsim and both Blackhole cards; see silicon run 1791254100 and silicon-operating-notes.md. Floating arithmetic is not IEEE754.
    pub static ELWSUB: InstructionDef = InstructionDef::new(
        "ELWSUB_BH",
        "ELWSUB",
        0x30,
        &[Field::new("FlipSrcB", 23, 1, false, None), Field::new("FlipSrcA", 22, 1, false, None), Field::new("AddDst", 21, 1, false, None), Field::new("BroadcastSrcBRow", 20, 1, false, None), Field::new("BroadcastSrcBCol0", 19, 1, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[],
        0x00063c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step9_matmul.rs::matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole, crates/tt-tests/tests/step9_matmul.rs::elw_broadcast_assignment_and_destination_fields, crates/tt-tests/tests/step90_matrix_eltwise.rs::resident_matrix_arithmetic_and_rhs_broadcasts", moved: &["AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/ELWSUB.md",
    );

    /// `ELWMUL_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step9_matmul.rs::matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole, crates/tt-tests/tests/step9_matmul.rs::elw_broadcast_assignment_and_destination_fields, crates/tt-tests/tests/step90_matrix_eltwise.rs::resident_matrix_arithmetic_and_rhs_broadcasts`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/ELWMUL.md`), and on Blackhole `AddrMod` sits elsewhere. `AddrMod` has a different width on Blackhole. Other fields are carried from that diagram. Step9 and step90 validate broadcast, assignment, Dst addressing and repeated bank release on ttsim and both Blackhole cards; see silicon run 1791254100 and silicon-operating-notes.md. Floating arithmetic is not IEEE754.
    pub static ELWMUL: InstructionDef = InstructionDef::new(
        "ELWMUL_BH",
        "ELWMUL",
        0x27,
        &[Field::new("FlipSrcB", 23, 1, false, None), Field::new("FlipSrcA", 22, 1, false, None), Field::new("BroadcastSrcBRow", 20, 1, false, None), Field::new("BroadcastSrcBCol0", 19, 1, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[],
        0x00263c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step9_matmul.rs::matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole, crates/tt-tests/tests/step9_matmul.rs::elw_broadcast_assignment_and_destination_fields, crates/tt-tests/tests/step90_matrix_eltwise.rs::resident_matrix_arithmetic_and_rhs_broadcasts", moved: &["AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/ELWMUL.md",
    );

    /// `DOTPV_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step9_matmul.rs::matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/DOTPV.md`), and on Blackhole `AddrMod` sits elsewhere. `AddrMod` has a different width on Blackhole. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static DOTPV: InstructionDef = InstructionDef::new(
        "DOTPV_BH",
        "DOTPV",
        0x29,
        &[Field::new("FlipSrcB", 23, 1, false, None), Field::new("FlipSrcA", 22, 1, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[],
        0x003e3c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step9_matmul.rs::matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole", moved: &["AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/DOTPV.md",
    );

    /// `MOVDBGA2D_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step9_matmul.rs::matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/MOVDBGA2D.md`), and on Blackhole `AddrMod` sits elsewhere. `AddrMod` has a different width on Blackhole. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static MOVDBGA2D: InstructionDef = InstructionDef::new(
        "MOVDBGA2D_BH",
        "MOVDBGA2D",
        0x09,
        &[Field::new("UseDst32bLo", 23, 1, false, None), Field::new("SrcRow", 17, 6, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("Move8Rows", 13, 1, false, None), Field::new("DstRow", 0, 10, false, None)],
        &[],
        0x00001c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step9_matmul.rs::matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole", moved: &["AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/MOVDBGA2D.md",
    );

    /// `SHIFTXB_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step9_matmul.rs::shiftxb_addr_mod_sits_one_bit_lower_on_blackhole`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/SHIFTXB.md`), and on Blackhole `AddrMod` sits elsewhere. `AddrMod` has a different width on Blackhole. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static SHIFTXB: InstructionDef = InstructionDef::new(
        "SHIFTXB_BH",
        "SHIFTXB",
        0x18,
        &[Field::new("AddrMod", 14, 3, false, None), Field::new("ShiftInZero", 10, 1, false, None), Field::new("SrcRow", 0, 6, false, None)],
        &[],
        0x00fe3bc0,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step9_matmul.rs::shiftxb_addr_mod_sits_one_bit_lower_on_blackhole", moved: &["AddrMod"], dropped: &[], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/SHIFTXB.md",
    );

    /// `ZEROACC_BH`. **`MEASURED`** on ttsim or silicon by `crates/tt-tests/tests/step9_matmul.rs::zeroacc_addr_mod_and_use_dst32b_on_blackhole`, not documented: the only diagram is Wormhole's (`WormholeB0/TensixTile/TensixCoprocessor/ZEROACC.md`), and on Blackhole `AddrMod`, `UseDst32b` sit elsewhere. `AddrMod` has a different width on Blackhole. `Revert` is not carried: its bits hold something else on Blackhole and its own position is unknown. Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon.
    pub static ZEROACC: InstructionDef = InstructionDef::new(
        "ZEROACC_BH",
        "ZEROACC",
        0x10,
        &[Field::new("Mode", 19, 2, false, None), Field::new("UseDst32b", 18, 1, false, None), Field::new("AddrMod", 14, 3, false, None), Field::new("Imm10", 0, 10, false, None)],
        &[],
        0x00e23c00,
        Provenance::Measured { evidence: "crates/tt-tests/tests/step9_matmul.rs::zeroacc_addr_mod_and_use_dst32b_on_blackhole", moved: &["AddrMod", "UseDst32b"], dropped: &["Revert"], widened: &["AddrMod"] },
        "WormholeB0/TensixTile/TensixCoprocessor/ZEROACC.md",
    );

    /// Encodings Blackhole replaces. Present so that the difference is
    /// visible and testable, not so that they can be used here.
    pub mod wormhole {
        use super::super::*;

        /// `STALLWAIT`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/STALLWAIT.md`. Blackhole replaces it with `STALLWAIT_BH`; use that instead.
        pub static STALLWAIT: InstructionDef = InstructionDef::new(
            "STALLWAIT",
            "STALLWAIT",
            0xa2,
            &[
                Field::new("BlockMask", 15, 9, false, None),
                Field::new("ConditionMask", 0, 15, false, None),
            ],
            &[],
            0x00000000,
            Provenance::SupersededOnBlackhole { by: "STALLWAIT_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/STALLWAIT.md",
        );

        /// `PACR`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/PACR.md`. Blackhole replaces it with `PACR_BH`; use that instead.
        pub static PACR: InstructionDef = InstructionDef::new(
            "PACR",
            "PACR",
            0x41,
            &[
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("ZeroWrite", 12, 1, false, None),
                Field::new("PackerMask", 8, 4, false, None),
                Field::new("OvrdThreadId", 7, 1, false, None),
                Field::new("Concat", 4, 1, false, None),
                Field::new("Flush", 1, 1, false, None),
                Field::new("Last", 0, 1, false, None),
            ],
            &[(Field::new("", 23, 1, false, None), 0)],
            0x007e606c,
            Provenance::SupersededOnBlackhole { by: "PACR_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/PACR.md",
        );

        /// `SFPLUTFP32`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/SFPLUTFP32.md`. Blackhole replaces it with `SFPLUTFP32_BH`; use that instead.
        pub static SFPLUTFP32: InstructionDef = InstructionDef::new(
            "SFPLUTFP32",
            "SFPLUTFP32",
            0x95,
            &[
                Field::new("VD", 4, 4, false, None),
                Field::new("Mod1", 0, 4, false, None),
            ],
            &[],
            0x00ffff00,
            Provenance::SupersededOnBlackhole {
                by: "SFPLUTFP32_BH",
            },
            "WormholeB0/TensixTile/TensixCoprocessor/SFPLUTFP32.md",
        );

        /// `MOVD2A`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/MOVD2A.md`. Blackhole replaces it with `MOVD2A_BH`; use that instead.
        pub static MOVD2A: InstructionDef = InstructionDef::new(
            "MOVD2A",
            "MOVD2A",
            0x08,
            &[
                Field::new("UseDst32bLo", 23, 1, false, None),
                Field::new("SrcRow", 17, 6, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("Move4Rows", 13, 1, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x00005c00,
            Provenance::SupersededOnBlackhole { by: "MOVD2A_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/MOVD2A.md",
        );

        /// `MOVD2B`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/MOVD2B.md`. Blackhole replaces it with `MOVD2B_BH`; use that instead.
        pub static MOVD2B: InstructionDef = InstructionDef::new(
            "MOVD2B",
            "MOVD2B",
            0x0a,
            &[
                Field::new("UseDst32bLo", 23, 1, false, None),
                Field::new("SrcRow", 17, 6, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("Move4Rows", 13, 1, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x00005c00,
            Provenance::SupersededOnBlackhole { by: "MOVD2B_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/MOVD2B.md",
        );

        /// `MOVB2A`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/MOVB2A.md`. Blackhole replaces it with `MOVB2A_BH`; use that instead.
        pub static MOVB2A: InstructionDef = InstructionDef::new(
            "MOVB2A",
            "MOVB2A",
            0x0b,
            &[
                Field::new("SrcARow", 17, 6, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("Move4Rows", 13, 1, false, None),
                Field::new("SrcBRow", 0, 6, false, None),
            ],
            &[],
            0x00805fc0,
            Provenance::SupersededOnBlackhole { by: "MOVB2A_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/MOVB2A.md",
        );

        /// `MOVDBGA2D`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/MOVDBGA2D.md`. Blackhole replaces it with `MOVDBGA2D_BH`; use that instead.
        pub static MOVDBGA2D: InstructionDef = InstructionDef::new(
            "MOVDBGA2D",
            "MOVDBGA2D",
            0x09,
            &[
                Field::new("UseDst32bLo", 23, 1, false, None),
                Field::new("SrcRow", 17, 6, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("Move8Rows", 13, 1, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x00005c00,
            Provenance::SupersededOnBlackhole { by: "MOVDBGA2D_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/MOVDBGA2D.md",
        );

        /// `MOVA2D`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/MOVA2D.md`. Blackhole replaces it with `MOVA2D_BH`; use that instead.
        pub static MOVA2D: InstructionDef = InstructionDef::new(
            "MOVA2D",
            "MOVA2D",
            0x12,
            &[
                Field::new("UseDst32bLo", 23, 1, false, None),
                Field::new("SrcRow", 17, 6, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("Move8Rows", 13, 1, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x00005c00,
            Provenance::SupersededOnBlackhole { by: "MOVA2D_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/MOVA2D.md",
        );

        /// `MOVB2D`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/MOVB2D.md`. Blackhole replaces it with `MOVB2D_BH`; use that instead.
        pub static MOVB2D: InstructionDef = InstructionDef::new(
            "MOVB2D",
            "MOVB2D",
            0x13,
            &[
                Field::new("UseDst32bLo", 23, 1, false, None),
                Field::new("SrcRow", 17, 6, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("Move4Rows", 14, 1, false, None),
                Field::new("Broadcast1RowTo8", 13, 1, false, None),
                Field::new("BroadcastCol0", 12, 1, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x00000c00,
            Provenance::SupersededOnBlackhole { by: "MOVB2D_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/MOVB2D.md",
        );

        /// `ELWMUL`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/ELWMUL.md`. Blackhole replaces it with `ELWMUL_BH`; use that instead.
        pub static ELWMUL: InstructionDef = InstructionDef::new(
            "ELWMUL",
            "ELWMUL",
            0x27,
            &[
                Field::new("FlipSrcB", 23, 1, false, None),
                Field::new("FlipSrcA", 22, 1, false, None),
                Field::new("BroadcastSrcBRow", 20, 1, false, None),
                Field::new("BroadcastSrcBCol0", 19, 1, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x00267c00,
            Provenance::SupersededOnBlackhole { by: "ELWMUL_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/ELWMUL.md",
        );

        /// `ELWADD`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/ELWADD.md`. Blackhole replaces it with `ELWADD_BH`; use that instead.
        pub static ELWADD: InstructionDef = InstructionDef::new(
            "ELWADD",
            "ELWADD",
            0x28,
            &[
                Field::new("FlipSrcB", 23, 1, false, None),
                Field::new("FlipSrcA", 22, 1, false, None),
                Field::new("AddDst", 21, 1, false, None),
                Field::new("BroadcastSrcBRow", 20, 1, false, None),
                Field::new("BroadcastSrcBCol0", 19, 1, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x00067c00,
            Provenance::SupersededOnBlackhole { by: "ELWADD_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/ELWADD.md",
        );

        /// `ELWSUB`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/ELWSUB.md`. Blackhole replaces it with `ELWSUB_BH`; use that instead.
        pub static ELWSUB: InstructionDef = InstructionDef::new(
            "ELWSUB",
            "ELWSUB",
            0x30,
            &[
                Field::new("FlipSrcB", 23, 1, false, None),
                Field::new("FlipSrcA", 22, 1, false, None),
                Field::new("AddDst", 21, 1, false, None),
                Field::new("BroadcastSrcBRow", 20, 1, false, None),
                Field::new("BroadcastSrcBCol0", 19, 1, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x00067c00,
            Provenance::SupersededOnBlackhole { by: "ELWSUB_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/ELWSUB.md",
        );

        /// `GMPOOL`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/GMPOOL.md`. Blackhole replaces it with `GMPOOL_BH`; use that instead.
        pub static GMPOOL: InstructionDef = InstructionDef::new(
            "GMPOOL",
            "GMPOOL",
            0x33,
            &[
                Field::new("FlipSrcB", 23, 1, false, None),
                Field::new("FlipSrcA", 22, 1, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("ArgMax", 14, 1, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x003e3c00,
            Provenance::SupersededOnBlackhole { by: "GMPOOL_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/GMPOOL.md",
        );

        /// `ZEROACC`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/ZEROACC.md`. Blackhole replaces it with `ZEROACC_BH`; use that instead.
        pub static ZEROACC: InstructionDef = InstructionDef::new(
            "ZEROACC",
            "ZEROACC",
            0x10,
            &[
                Field::new("UseDst32b", 21, 1, false, None),
                Field::new("Mode", 19, 2, false, None),
                Field::new("Revert", 18, 1, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("Imm10", 0, 10, false, None),
            ],
            &[],
            0x00c27c00,
            Provenance::SupersededOnBlackhole { by: "ZEROACC_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/ZEROACC.md",
        );

        /// `SHIFTXB`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/SHIFTXB.md`. Blackhole replaces it with `SHIFTXB_BH`; use that instead.
        pub static SHIFTXB: InstructionDef = InstructionDef::new(
            "SHIFTXB",
            "SHIFTXB",
            0x18,
            &[
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("ShiftInZero", 10, 1, false, None),
                Field::new("SrcRow", 0, 6, false, None),
            ],
            &[],
            0x00fe7bc0,
            Provenance::SupersededOnBlackhole { by: "SHIFTXB_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/SHIFTXB.md",
        );

        /// `MVMUL`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md`. Blackhole replaces it with `MVMUL_BH`; use that instead.
        pub static MVMUL: InstructionDef = InstructionDef::new(
            "MVMUL",
            "MVMUL",
            0x26,
            &[
                Field::new("FlipSrcB", 23, 1, false, None),
                Field::new("FlipSrcA", 22, 1, false, None),
                Field::new("BroadcastSrcBRow", 19, 1, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x00367c00,
            Provenance::SupersededOnBlackhole { by: "MVMUL_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md",
        );

        /// `DOTPV`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/DOTPV.md`. Blackhole replaces it with `DOTPV_BH`; use that instead.
        pub static DOTPV: InstructionDef = InstructionDef::new(
            "DOTPV",
            "DOTPV",
            0x29,
            &[
                Field::new("FlipSrcB", 23, 1, false, None),
                Field::new("FlipSrcA", 22, 1, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x003e7c00,
            Provenance::SupersededOnBlackhole { by: "DOTPV_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/DOTPV.md",
        );

        /// `GAPOOL`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/GAPOOL.md`. Blackhole replaces it with `GAPOOL_BH`; use that instead.
        pub static GAPOOL: InstructionDef = InstructionDef::new(
            "GAPOOL",
            "GAPOOL",
            0x34,
            &[
                Field::new("FlipSrcB", 23, 1, false, None),
                Field::new("FlipSrcA", 22, 1, false, None),
                Field::new("AddrMod", 15, 2, false, None),
                Field::new("DstRow", 0, 10, false, None),
            ],
            &[],
            0x003e7c00,
            Provenance::SupersededOnBlackhole { by: "GAPOOL_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/GAPOOL.md",
        );

        /// `SFPLOAD`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/SFPLOAD.md`. Blackhole replaces it with `SFPLOAD_BH`; use that instead.
        pub static SFPLOAD: InstructionDef = InstructionDef::new(
            "SFPLOAD",
            "SFPLOAD",
            0x70,
            &[
                Field::new("VD", 20, 4, false, None),
                Field::new("Mod0", 16, 4, false, None),
                Field::new("AddrMod", 14, 2, false, None),
                Field::new("Imm10", 0, 10, false, None),
            ],
            &[],
            0x00003c00,
            Provenance::SupersededOnBlackhole { by: "SFPLOAD_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/SFPLOAD.md",
        );

        /// `SFPLOADMACRO`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/SFPLOADMACRO.md`. Blackhole replaces it with `SFPLOADMACRO_BH`; use that instead.
        pub static SFPLOADMACRO: InstructionDef = InstructionDef::new(
            "SFPLOADMACRO",
            "SFPLOADMACRO",
            0x93,
            &[
                Field::new("MacroIndex", 22, 2, false, None),
                Field::new("VDLo", 20, 2, false, None),
                Field::new("Mod0", 16, 4, false, None),
                Field::new("AddrMod", 14, 2, false, None),
                Field::new("Imm9", 1, 9, false, None),
                Field::new("VDHi", 0, 1, false, None),
            ],
            &[],
            0x00003c00,
            Provenance::SupersededOnBlackhole {
                by: "SFPLOADMACRO_BH",
            },
            "WormholeB0/TensixTile/TensixCoprocessor/SFPLOADMACRO.md",
        );

        /// `SFPSTORE`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/SFPSTORE.md`. Blackhole replaces it with `SFPSTORE_BH`; use that instead.
        pub static SFPSTORE: InstructionDef = InstructionDef::new(
            "SFPSTORE",
            "SFPSTORE",
            0x72,
            &[
                Field::new("VD", 20, 4, false, None),
                Field::new("Mod0", 16, 4, false, None),
                Field::new("AddrMod", 14, 2, false, None),
                Field::new("Imm10", 0, 10, false, None),
            ],
            &[],
            0x00003c00,
            Provenance::SupersededOnBlackhole { by: "SFPSTORE_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/SFPSTORE.md",
        );

        /// `SFPAND`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/SFPAND.md`. Blackhole replaces it with `SFPAND_BH`; use that instead.
        pub static SFPAND: InstructionDef = InstructionDef::new(
            "SFPAND",
            "SFPAND",
            0x7e,
            &[
                Field::new("VC", 8, 4, false, None),
                Field::new("VD", 4, 4, false, None),
            ],
            &[],
            0x00fff00f,
            Provenance::SupersededOnBlackhole { by: "SFPAND_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/SFPAND.md",
        );

        /// `SFPOR`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/SFPOR.md`. Blackhole replaces it with `SFPOR_BH`; use that instead.
        pub static SFPOR: InstructionDef = InstructionDef::new(
            "SFPOR",
            "SFPOR",
            0x7f,
            &[
                Field::new("VC", 8, 4, false, None),
                Field::new("VD", 4, 4, false, None),
            ],
            &[],
            0x00fff00f,
            Provenance::SupersededOnBlackhole { by: "SFPOR_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/SFPOR.md",
        );

        /// `SFPPUSHC`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/SFPPUSHC.md`. Blackhole replaces it with `SFPPUSHC_BH`; use that instead.
        pub static SFPPUSHC: InstructionDef = InstructionDef::new(
            "SFPPUSHC",
            "SFPPUSHC",
            0x87,
            &[Field::new("VD", 4, 4, false, None)],
            &[(Field::new("", 0, 4, false, None), 0)],
            0x00ffff00,
            Provenance::SupersededOnBlackhole { by: "SFPPUSHC_BH" },
            "WormholeB0/TensixTile/TensixCoprocessor/SFPPUSHC.md",
        );

        /// `SFPSTOCHRND`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md`. Blackhole replaces it with `SFPSTOCHRND_BH`; use that instead.
        pub static SFPSTOCHRND: InstructionDef = InstructionDef::new(
            "SFPSTOCHRND",
            "SFP_STOCH_RND",
            0x8e,
            &[
                Field::new("StochasticRounding", 21, 1, false, None),
                Field::new("VC", 8, 4, false, None),
                Field::new("VD", 4, 4, false, None),
                Field::new("Mod1", 0, 3, false, None),
            ],
            &[],
            0x00dff008,
            Provenance::SupersededOnBlackhole {
                by: "SFPSTOCHRND_BH",
            },
            "WormholeB0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md",
        );

        /// `SFPSTOCHRNDi`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/SFPSTOCHRND_IntInt.md`. Blackhole replaces it with `SFPSTOCHRNDi_BH`; use that instead.
        pub static SFPSTOCHRNDi: InstructionDef = InstructionDef::new(
            "SFPSTOCHRNDi",
            "SFP_STOCH_RND",
            0x8e,
            &[
                Field::new("StochasticRounding", 21, 1, false, None),
                Field::new("Imm5", 16, 5, false, None),
                Field::new("VB", 12, 4, false, None),
                Field::new("VC", 8, 4, false, None),
                Field::new("VD", 4, 4, false, None),
                Field::new("UseImm5", 3, 1, false, None),
                Field::new("Mod1", 0, 3, false, None),
            ],
            &[],
            0x00c00000,
            Provenance::SupersededOnBlackhole {
                by: "SFPSTOCHRNDi_BH",
            },
            "WormholeB0/TensixTile/TensixCoprocessor/SFPSTOCHRND_IntInt.md",
        );

        /// `UNPACR_NOP_ZEROSRC`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_ZEROSRC.md`. Blackhole replaces it with `UNPACR_NOP_ZEROSRC_BH`; use that instead.
        pub static UNPACR_NOP_ZEROSRC: InstructionDef = InstructionDef::new(
            "UNPACR_NOP_ZEROSRC",
            "UNPACR_NOP",
            0x43,
            &[
                Field::new("WhichUnpacker", 23, 1, false, None),
                Field::new("WaitLikeUnpacr", 4, 1, false, None),
                Field::new("BothBanks", 3, 1, false, None),
                Field::new("NegativeInfSrcA", 2, 1, false, None),
            ],
            &[
                (Field::new("", 0, 2, false, None), 1),
                (Field::new("", 6, 1, false, None), 0),
            ],
            0x007fffa0,
            Provenance::SupersededOnBlackhole {
                by: "UNPACR_NOP_ZEROSRC_BH",
            },
            "WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_ZEROSRC.md",
        );

        /// `UNPACR_NOP_SETDVALID`. **Wormhole's encoding**, from `WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_SETDVALID.md`. Blackhole replaces it with `UNPACR_NOP_SETDVALID_BH`; use that instead.
        pub static UNPACR_NOP_SETDVALID: InstructionDef = InstructionDef::new(
            "UNPACR_NOP_SETDVALID",
            "UNPACR_NOP",
            0x43,
            &[Field::new("WhichUnpacker", 23, 1, false, None)],
            &[(Field::new("", 0, 3, false, None), 7)],
            0x007ffff8,
            Provenance::SupersededOnBlackhole {
                by: "UNPACR_NOP_SETDVALID_BH",
            },
            "WormholeB0/TensixTile/TensixCoprocessor/UNPACR_NOP_SETDVALID.md",
        );
    }
}

/// Every instruction encoding, for table-wide checks.
pub static ALL: &[&InstructionDef] = &[
    &defs::ATCAS,
    &defs::ATSWAP,
    &defs::RMWCIB0,
    &defs::RMWCIB1,
    &defs::RMWCIB2,
    &defs::RMWCIB3,
    &defs::SETDMAREG_Immediate,
    &defs::SETDMAREG_Special,
    &defs::wormhole::STALLWAIT,
    &defs::STALLWAIT,
    &defs::SEMWAIT,
    &defs::STREAMWAIT,
    &defs::SEMINIT,
    &defs::REPLAY,
    &defs::wormhole::PACR,
    &defs::PACR,
    &defs::PACR_SETREG,
    &defs::SFPSHFT2,
    &defs::SFPSHFT2b,
    &defs::wormhole::SFPLUTFP32,
    &defs::SFPLUTFP32,
    &defs::REG2FLOP_Configuration,
    &defs::REG2FLOP_ADC,
    &defs::XMOV,
    &defs::WRCFG,
    &defs::RDCFG,
    &defs::SETC16,
    &defs::STREAMWRCFG,
    &defs::CFGSHIFTMASK,
    &defs::SETADC,
    &defs::SETADCXY,
    &defs::INCADCXY,
    &defs::ADDRCRXY,
    &defs::SETADCZW,
    &defs::INCADCZW,
    &defs::ADDRCRZW,
    &defs::SETADCXX,
    &defs::NOP,
    &defs::SEMPOST,
    &defs::SEMGET,
    &defs::SETDVALID,
    &defs::GATESRCRST,
    &defs::wormhole::MOVD2A,
    &defs::wormhole::MOVD2B,
    &defs::wormhole::MOVB2A,
    &defs::wormhole::MOVDBGA2D,
    &defs::ZEROSRC,
    &defs::wormhole::MOVA2D,
    &defs::wormhole::MOVB2D,
    &defs::wormhole::ELWMUL,
    &defs::wormhole::ELWADD,
    &defs::wormhole::ELWSUB,
    &defs::wormhole::GMPOOL,
    &defs::wormhole::ZEROACC,
    &defs::TRNSPSRCB,
    &defs::SHIFTXA,
    &defs::wormhole::SHIFTXB,
    &defs::CLREXPHIST,
    &defs::wormhole::MVMUL,
    &defs::wormhole::DOTPV,
    &defs::wormhole::GAPOOL,
    &defs::CLEARDVALID,
    &defs::SETRWC,
    &defs::INCRWC,
    &defs::LOADIND,
    &defs::STOREIND_L1,
    &defs::STOREIND_MMIO,
    &defs::STOREIND_Src,
    &defs::LOADREG,
    &defs::STOREREG,
    &defs::FLUSHDMA,
    &defs::DMANOP,
    &defs::ADDDMAREG,
    &defs::ADDDMAREGi,
    &defs::SUBDMAREG,
    &defs::SUBDMAREGi,
    &defs::MULDMAREG,
    &defs::MULDMAREGi,
    &defs::BITWOPDMAREG,
    &defs::BITWOPDMAREGi,
    &defs::SHIFTDMAREG,
    &defs::SHIFTDMAREGi,
    &defs::CMPDMAREG,
    &defs::CMPDMAREGi,
    &defs::ATGETM,
    &defs::ATRELM,
    &defs::ATINCGET,
    &defs::ATINCGETPTR,
    &defs::wormhole::SFPLOAD,
    &defs::SFPLOAD,
    &defs::wormhole::SFPLOADMACRO,
    &defs::SFPLOADMACRO,
    &defs::wormhole::SFPSTORE,
    &defs::SFPSTORE,
    &defs::MOP,
    &defs::MOP_CFG,
    &defs::SFPLOADI,
    &defs::SFPIADD,
    &defs::SFPSWAP,
    &defs::SFPCONFIG,
    &defs::SFPMAD,
    &defs::SFPADD,
    &defs::SFPMUL,
    &defs::SFPLUT,
    &defs::SFPMULI,
    &defs::SFPADDI,
    &defs::SFPDIVP2,
    &defs::SFPEXEXP,
    &defs::SFPEXMAN,
    &defs::SFPSHFT,
    &defs::SFPSETCC,
    &defs::SFPMOV,
    &defs::SFPABS,
    &defs::wormhole::SFPAND,
    &defs::SFPAND,
    &defs::wormhole::SFPOR,
    &defs::SFPOR,
    &defs::SFPNOT,
    &defs::SFPLZ,
    &defs::SFPSETEXP,
    &defs::SFPSETMAN,
    &defs::wormhole::SFPPUSHC,
    &defs::SFPPUSHC,
    &defs::SFPPOPC,
    &defs::SFPSETSGN,
    &defs::SFPENCC,
    &defs::SFPCOMPC,
    &defs::SFPTRANSP,
    &defs::SFPXOR,
    &defs::wormhole::SFPSTOCHRND,
    &defs::wormhole::SFPSTOCHRNDi,
    &defs::SFPSTOCHRND,
    &defs::SFPSTOCHRNDi,
    &defs::SFPNOP,
    &defs::SFPCAST,
    &defs::SFPLE,
    &defs::SFPGT,
    &defs::SFPMUL24,
    &defs::SFPARECIP,
    &defs::UNPACR_Regular,
    &defs::UNPACR_IncrementContextCounter,
    &defs::UNPACR_FlushCache,
    &defs::UNPACR_NOP_OverlayClear0,
    &defs::UNPACR_NOP_OverlayClear3,
    &defs::wormhole::UNPACR_NOP_ZEROSRC,
    &defs::UNPACR_NOP_Nop,
    &defs::UNPACR_NOP_SETREG,
    &defs::wormhole::UNPACR_NOP_SETDVALID,
    &defs::UNPACR_NOP_SETDVALID,
    &defs::UNPACR_NOP_ZEROSRC,
    &defs::GMPOOL,
    &defs::GAPOOL,
    &defs::MVMUL,
    &defs::MOVA2D,
    &defs::MOVB2D,
    &defs::MOVD2A,
    &defs::MOVD2B,
    &defs::MOVB2A,
    &defs::ELWADD,
    &defs::ELWSUB,
    &defs::ELWMUL,
    &defs::DOTPV,
    &defs::MOVDBGA2D,
    &defs::SHIFTXB,
    &defs::ZEROACC,
];

/// The documented bit layout of each datum type in `Src` and `Dst`.
///
/// The coprocessor does not entirely follow IEEE 754, so conversions are
/// built against these rather than against assumptions.
pub mod datum {
    use super::*;

    /// `Src_TF32`, from `WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md`.
    pub static SRC_TF32: DatumLayout = DatumLayout::new(
        "Src_TF32",
        19,
        &[
            Field::new("Sign", 18, 1, false, None),
            Field::new("Mantissa", 8, 10, false, None),
            Field::new("Exponent", 0, 8, false, None),
        ],
        &[],
        "WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md",
    );

    /// `Src_BF16`, from `WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md`.
    pub static SRC_BF16: DatumLayout = DatumLayout::new(
        "Src_BF16",
        19,
        &[
            Field::new("Sign", 18, 1, false, None),
            Field::new("Mantissa", 11, 7, false, None),
            Field::new("Exponent", 0, 8, false, None),
        ],
        &[(Field::new("", 8, 3, false, None), 0)],
        "WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md",
    );

    /// `Src_FP16`, from `WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md`.
    pub static SRC_FP16: DatumLayout = DatumLayout::new(
        "Src_FP16",
        19,
        &[
            Field::new("Sign", 18, 1, false, None),
            Field::new("Mantissa", 8, 10, false, None),
            Field::new("Exponent", 0, 5, false, None),
        ],
        &[(Field::new("", 5, 3, false, None), 0)],
        "WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md",
    );

    /// `Src_INT8`, from `WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md`.
    pub static SRC_INT8: DatumLayout = DatumLayout::new(
        "Src_INT8",
        19,
        &[
            Field::new("Sign", 18, 1, false, None),
            Field::new("Magnitude", 8, 10, false, None),
            Field::new("Exponent", 0, 5, false, Some("16 or 0")),
        ],
        &[(Field::new("", 5, 3, false, None), 0)],
        "WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md",
    );

    /// `Src_INT16`, from `WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md`.
    pub static SRC_INT16: DatumLayout = DatumLayout::new(
        "Src_INT16",
        19,
        &[
            Field::new("Sign", 18, 1, false, None),
            Field::new("Magnitude", 11, 7, false, Some("high")),
            Field::new("Magnitude", 0, 8, false, Some("low")),
        ],
        &[(Field::new("", 8, 3, false, None), 0)],
        "WormholeB0/TensixTile/TensixCoprocessor/SrcASrcB.md",
    );

    /// `Dst16_BF16`, from `BlackholeA0/TensixTile/TensixCoprocessor/Dst.md`.
    pub static DST16_BF16: DatumLayout = DatumLayout::new(
        "Dst16_BF16",
        16,
        &[
            Field::new("Sign", 15, 1, false, None),
            Field::new("Mantissa", 8, 7, false, None),
            Field::new("Exponent", 0, 8, false, None),
        ],
        &[],
        "BlackholeA0/TensixTile/TensixCoprocessor/Dst.md",
    );

    /// `Dst16_FP16`, from `BlackholeA0/TensixTile/TensixCoprocessor/Dst.md`.
    pub static DST16_FP16: DatumLayout = DatumLayout::new(
        "Dst16_FP16",
        16,
        &[
            Field::new("Sign", 15, 1, false, None),
            Field::new("Mantissa", 5, 10, false, None),
            Field::new("Exponent", 0, 5, false, None),
        ],
        &[],
        "BlackholeA0/TensixTile/TensixCoprocessor/Dst.md",
    );

    /// `Dst16_INT8`, from `BlackholeA0/TensixTile/TensixCoprocessor/Dst.md`.
    pub static DST16_INT8: DatumLayout = DatumLayout::new(
        "Dst16_INT8",
        16,
        &[
            Field::new("Sign", 15, 1, false, None),
            Field::new("Magnitude", 5, 10, false, None),
            Field::new("Exponent", 0, 5, false, Some("16 or 0")),
        ],
        &[],
        "BlackholeA0/TensixTile/TensixCoprocessor/Dst.md",
    );

    /// `Dst16_INT16`, from `BlackholeA0/TensixTile/TensixCoprocessor/Dst.md`.
    pub static DST16_INT16: DatumLayout = DatumLayout::new(
        "Dst16_INT16",
        16,
        &[
            Field::new("Sign", 15, 1, false, None),
            Field::new("Magnitude", 0, 15, false, None),
        ],
        &[],
        "BlackholeA0/TensixTile/TensixCoprocessor/Dst.md",
    );

    /// `Dst32_FP32`, from `BlackholeA0/TensixTile/TensixCoprocessor/Dst.md`.
    pub static DST32_FP32: DatumLayout = DatumLayout::new(
        "Dst32_FP32",
        32,
        &[
            Field::new("Sign", 31, 1, false, None),
            Field::new("Mantissa", 24, 7, false, Some("high")),
            Field::new("Exponent", 16, 8, false, None),
            Field::new("Mantissa", 0, 16, false, Some("low")),
        ],
        &[],
        "BlackholeA0/TensixTile/TensixCoprocessor/Dst.md",
    );

    /// `Dst32_INT32`, from `BlackholeA0/TensixTile/TensixCoprocessor/Dst.md`.
    pub static DST32_INT32: DatumLayout = DatumLayout::new(
        "Dst32_INT32",
        32,
        &[
            Field::new("Sign", 31, 1, false, None),
            Field::new("Magnitude", 24, 7, false, Some("middle")),
            Field::new("Magnitude", 16, 8, false, Some("high")),
            Field::new("Magnitude", 0, 16, false, Some("low")),
        ],
        &[],
        "BlackholeA0/TensixTile/TensixCoprocessor/Dst.md",
    );

    /// `NOC_AT_LEN_BE_Increment`, from `BlackholeA0/NoC/Atomics.md`.
    pub static NOC_AT_LEN_BE_INCREMENT: DatumLayout = DatumLayout::new(
        "NOC_AT_LEN_BE_Increment",
        32,
        &[
            Field::new("IntWidth", 2, 5, false, None),
            Field::new("Ofs", 0, 2, false, None),
        ],
        &[(Field::new("", 12, 4, false, None), 1)],
        "BlackholeA0/NoC/Atomics.md",
    );

    /// `NOC_AT_LEN_BE_CAS`, from `BlackholeA0/NoC/Atomics.md`.
    pub static NOC_AT_LEN_BE_CAS: DatumLayout = DatumLayout::new(
        "NOC_AT_LEN_BE_CAS",
        32,
        &[
            Field::new("SetVal", 6, 4, false, None),
            Field::new("CmpVal", 2, 4, false, None),
            Field::new("Ofs", 0, 2, false, None),
        ],
        &[(Field::new("", 12, 4, false, None), 4)],
        "BlackholeA0/NoC/Atomics.md",
    );

    /// `NOC_AT_LEN_BE_SwapMask`, from `BlackholeA0/NoC/Atomics.md`.
    pub static NOC_AT_LEN_BE_SWAPMASK: DatumLayout = DatumLayout::new(
        "NOC_AT_LEN_BE_SwapMask",
        32,
        &[Field::new("Mask", 2, 8, false, None)],
        &[(Field::new("", 12, 4, false, None), 3)],
        "BlackholeA0/NoC/Atomics.md",
    );

    /// `NOC_AT_LEN_BE_SwapIndex6`, from `BlackholeA0/NoC/Atomics.md`.
    pub static NOC_AT_LEN_BE_SWAPINDEX6: DatumLayout = DatumLayout::new(
        "NOC_AT_LEN_BE_SwapIndex6",
        32,
        &[Field::new("Ofs", 0, 2, false, None)],
        &[
            (Field::new("", 2, 1, false, None), 1),
            (Field::new("", 12, 4, false, None), 6),
        ],
        "BlackholeA0/NoC/Atomics.md",
    );

    /// `NOC_AT_LEN_BE_SwapIndex7`, from `BlackholeA0/NoC/Atomics.md`.
    pub static NOC_AT_LEN_BE_SWAPINDEX7: DatumLayout = DatumLayout::new(
        "NOC_AT_LEN_BE_SwapIndex7",
        32,
        &[Field::new("Ofs", 2, 2, false, None)],
        &[(Field::new("", 12, 4, false, None), 7)],
        "BlackholeA0/NoC/Atomics.md",
    );

    /// `NOC_AT_LEN_BE_SwapIndex10`, from `BlackholeA0/NoC/Atomics.md`.
    pub static NOC_AT_LEN_BE_SWAPINDEX10: DatumLayout = DatumLayout::new(
        "NOC_AT_LEN_BE_SwapIndex10",
        32,
        &[Field::new("Ofs", 0, 2, false, None)],
        &[
            (Field::new("", 8, 4, false, None), 3),
            (Field::new("", 12, 4, false, None), 10),
        ],
        "BlackholeA0/NoC/Atomics.md",
    );

    /// `NOC_AT_LEN_BE_Zaamo`, from `BlackholeA0/NoC/Atomics.md`.
    pub static NOC_AT_LEN_BE_ZAAMO: DatumLayout = DatumLayout::new(
        "NOC_AT_LEN_BE_Zaamo",
        32,
        &[
            Field::new("Op", 8, 3, false, None),
            Field::new("Ofs", 0, 2, false, None),
        ],
        &[
            (Field::new("", 11, 1, false, None), 1),
            (Field::new("", 12, 4, false, None), 10),
        ],
        "BlackholeA0/NoC/Atomics.md",
    );

    /// `NOC_AT_LEN_BE_Acc`, from `BlackholeA0/NoC/Atomics.md`.
    pub static NOC_AT_LEN_BE_ACC: DatumLayout = DatumLayout::new(
        "NOC_AT_LEN_BE_Acc",
        32,
        &[Field::new("Fmt", 0, 4, false, None)],
        &[(Field::new("", 12, 4, false, None), 9)],
        "BlackholeA0/NoC/Atomics.md",
    );
}

/// Every datum layout, for table-wide checks.
pub static ALL_LAYOUTS: &[&DatumLayout] = &[
    &datum::SRC_TF32,
    &datum::SRC_BF16,
    &datum::SRC_FP16,
    &datum::SRC_INT8,
    &datum::SRC_INT16,
    &datum::DST16_BF16,
    &datum::DST16_FP16,
    &datum::DST16_INT8,
    &datum::DST16_INT16,
    &datum::DST32_FP32,
    &datum::DST32_INT32,
    &datum::NOC_AT_LEN_BE_INCREMENT,
    &datum::NOC_AT_LEN_BE_CAS,
    &datum::NOC_AT_LEN_BE_SWAPMASK,
    &datum::NOC_AT_LEN_BE_SWAPINDEX6,
    &datum::NOC_AT_LEN_BE_SWAPINDEX7,
    &datum::NOC_AT_LEN_BE_SWAPINDEX10,
    &datum::NOC_AT_LEN_BE_ZAAMO,
    &datum::NOC_AT_LEN_BE_ACC,
];

/// One encoder per instruction, taking its operands from the high bits of
/// the word down — the order the specification lists them in.
pub mod encode {
    use super::*;

    /// `ATCAS`.
    pub const fn atcas(
        set_val: u32,
        cmp_val: u32,
        ofs: u32,
        addr_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::ATCAS;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(set_val) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: set_val,
                width: f.width(),
            });
        }
        word |= f.place(set_val);
        let f = def.fields()[1];
        if !f.fits(cmp_val) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: cmp_val,
                width: f.width(),
            });
        }
        word |= f.place(cmp_val);
        let f = def.fields()[2];
        if !f.fits(ofs) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: ofs,
                width: f.width(),
            });
        }
        word |= f.place(ofs);
        let f = def.fields()[3];
        if !f.fits(addr_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_reg,
                width: f.width(),
            });
        }
        word |= f.place(addr_reg);
        Ok(Instruction::new(word, def))
    }

    /// `ATSWAP`.
    pub const fn atswap(
        single_data_reg: u32,
        mask: u32,
        data_reg: u32,
        addr_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::ATSWAP;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(single_data_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: single_data_reg,
                width: f.width(),
            });
        }
        word |= f.place(single_data_reg);
        let f = def.fields()[1];
        if !f.fits(mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mask,
                width: f.width(),
            });
        }
        word |= f.place(mask);
        let f = def.fields()[2];
        if !f.fits(data_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: data_reg,
                width: f.width(),
            });
        }
        word |= f.place(data_reg);
        let f = def.fields()[3];
        if !f.fits(addr_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_reg,
                width: f.width(),
            });
        }
        word |= f.place(addr_reg);
        Ok(Instruction::new(word, def))
    }

    /// `RMWCIB0`.
    pub const fn rmwcib0(
        mask: u32,
        new_value: u32,
        index4: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::RMWCIB0;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mask,
                width: f.width(),
            });
        }
        word |= f.place(mask);
        let f = def.fields()[1];
        if !f.fits(new_value) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: new_value,
                width: f.width(),
            });
        }
        word |= f.place(new_value);
        let f = def.fields()[2];
        if !f.fits(index4) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: index4,
                width: f.width(),
            });
        }
        word |= f.place(index4);
        Ok(Instruction::new(word, def))
    }

    /// `RMWCIB1`.
    pub const fn rmwcib1(
        mask: u32,
        new_value: u32,
        index4: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::RMWCIB1;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mask,
                width: f.width(),
            });
        }
        word |= f.place(mask);
        let f = def.fields()[1];
        if !f.fits(new_value) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: new_value,
                width: f.width(),
            });
        }
        word |= f.place(new_value);
        let f = def.fields()[2];
        if !f.fits(index4) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: index4,
                width: f.width(),
            });
        }
        word |= f.place(index4);
        Ok(Instruction::new(word, def))
    }

    /// `RMWCIB2`.
    pub const fn rmwcib2(
        mask: u32,
        new_value: u32,
        index4: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::RMWCIB2;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mask,
                width: f.width(),
            });
        }
        word |= f.place(mask);
        let f = def.fields()[1];
        if !f.fits(new_value) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: new_value,
                width: f.width(),
            });
        }
        word |= f.place(new_value);
        let f = def.fields()[2];
        if !f.fits(index4) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: index4,
                width: f.width(),
            });
        }
        word |= f.place(index4);
        Ok(Instruction::new(word, def))
    }

    /// `RMWCIB3`.
    pub const fn rmwcib3(
        mask: u32,
        new_value: u32,
        index4: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::RMWCIB3;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mask,
                width: f.width(),
            });
        }
        word |= f.place(mask);
        let f = def.fields()[1];
        if !f.fits(new_value) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: new_value,
                width: f.width(),
            });
        }
        word |= f.place(new_value);
        let f = def.fields()[2];
        if !f.fits(index4) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: index4,
                width: f.width(),
            });
        }
        word |= f.place(index4);
        Ok(Instruction::new(word, def))
    }

    /// `SETDMAREG_Immediate`.
    pub const fn setdmareg_immediate(
        new_value: u32,
        result_half_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SETDMAREG_Immediate;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(new_value) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: new_value,
                width: f.width(),
            });
        }
        word |= f.place(new_value);
        let f = def.fields()[1];
        if !f.fits(result_half_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_half_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_half_reg);
        Ok(Instruction::new(word, def))
    }

    /// `SETDMAREG_Special`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `SetdmaregSpecial::ZERO.result_size(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct SetdmaregSpecial {
        result_size: u32,
        which_packers: u32,
        input_source: u32,
        input_half_reg: u32,
        result_half_reg: u32,
    }

    impl SetdmaregSpecial {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = SetdmaregSpecial {
            result_size: 0,
            which_packers: 0,
            input_source: 0,
            input_half_reg: 0,
            result_half_reg: 0,
        };

        pub const fn result_size(mut self, value: u32) -> Self {
            self.result_size = value;
            self
        }

        pub const fn which_packers(mut self, value: u32) -> Self {
            self.which_packers = value;
            self
        }

        pub const fn input_source(mut self, value: u32) -> Self {
            self.input_source = value;
            self
        }

        pub const fn input_half_reg(mut self, value: u32) -> Self {
            self.input_half_reg = value;
            self
        }

        pub const fn result_half_reg(mut self, value: u32) -> Self {
            self.result_half_reg = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SETDMAREG_Special;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.result_size) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.result_size,
                    width: f.width(),
                });
            }
            word |= f.place(self.result_size);
            let f = def.fields()[1];
            if !f.fits(self.which_packers) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.which_packers,
                    width: f.width(),
                });
            }
            word |= f.place(self.which_packers);
            let f = def.fields()[2];
            if !f.fits(self.input_source) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.input_source,
                    width: f.width(),
                });
            }
            word |= f.place(self.input_source);
            let f = def.fields()[3];
            if !f.fits(self.input_half_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.input_half_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.input_half_reg);
            let f = def.fields()[4];
            if !f.fits(self.result_half_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.result_half_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.result_half_reg);
            Ok(Instruction::new(word, def))
        }
    }

    /// `STALLWAIT_BH`.
    pub const fn stallwait(
        block_mask: u32,
        condition_mask: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::STALLWAIT;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(block_mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: block_mask,
                width: f.width(),
            });
        }
        word |= f.place(block_mask);
        let f = def.fields()[1];
        if !f.fits(condition_mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: condition_mask,
                width: f.width(),
            });
        }
        word |= f.place(condition_mask);
        Ok(Instruction::new(word, def))
    }

    /// `SEMWAIT`.
    pub const fn semwait(
        block_mask: u32,
        semaphore_mask: u32,
        condition_mask: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SEMWAIT;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(block_mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: block_mask,
                width: f.width(),
            });
        }
        word |= f.place(block_mask);
        let f = def.fields()[1];
        if !f.fits(semaphore_mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: semaphore_mask,
                width: f.width(),
            });
        }
        word |= f.place(semaphore_mask);
        let f = def.fields()[2];
        if !f.fits(condition_mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: condition_mask,
                width: f.width(),
            });
        }
        word |= f.place(condition_mask);
        Ok(Instruction::new(word, def))
    }

    /// `STREAMWAIT`.
    pub const fn streamwait(
        block_mask: u32,
        target_value_lo: u32,
        condition_index: u32,
        stream_select: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::STREAMWAIT;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(block_mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: block_mask,
                width: f.width(),
            });
        }
        word |= f.place(block_mask);
        let f = def.fields()[1];
        if !f.fits(target_value_lo) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: target_value_lo,
                width: f.width(),
            });
        }
        word |= f.place(target_value_lo);
        let f = def.fields()[2];
        if !f.fits(condition_index) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: condition_index,
                width: f.width(),
            });
        }
        word |= f.place(condition_index);
        let f = def.fields()[3];
        if !f.fits(stream_select) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: stream_select,
                width: f.width(),
            });
        }
        word |= f.place(stream_select);
        Ok(Instruction::new(word, def))
    }

    /// `SEMINIT`.
    pub const fn seminit(
        new_max: u32,
        new_value: u32,
        semaphore_mask: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SEMINIT;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(new_max) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: new_max,
                width: f.width(),
            });
        }
        word |= f.place(new_max);
        let f = def.fields()[1];
        if !f.fits(new_value) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: new_value,
                width: f.width(),
            });
        }
        word |= f.place(new_value);
        let f = def.fields()[2];
        if !f.fits(semaphore_mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: semaphore_mask,
                width: f.width(),
            });
        }
        word |= f.place(semaphore_mask);
        Ok(Instruction::new(word, def))
    }

    /// `REPLAY`.
    pub const fn replay(
        index: u32,
        count: u32,
        exec: u32,
        load: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::REPLAY;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(index) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: index,
                width: f.width(),
            });
        }
        word |= f.place(index);
        let f = def.fields()[1];
        if !f.fits(count) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: count,
                width: f.width(),
            });
        }
        word |= f.place(count);
        let f = def.fields()[2];
        if !f.fits(exec) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: exec,
                width: f.width(),
            });
        }
        word |= f.place(exec);
        let f = def.fields()[3];
        if !f.fits(load) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: load,
                width: f.width(),
            });
        }
        word |= f.place(load);
        Ok(Instruction::new(word, def))
    }

    /// `PACR_BH`, built field by field.
    ///
    /// 12 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Pacr::ZERO.cfg_context(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Pacr {
        cfg_context: u32,
        row_pad_zero: u32,
        dst_access_mode: u32,
        addr_mod: u32,
        addr_cnt_context: u32,
        zero_write: u32,
        read_intf_sel: u32,
        ovrd_thread_id: u32,
        concat: u32,
        ctxt_ctrl: u32,
        flush: u32,
        last: u32,
    }

    impl Pacr {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Pacr {
            cfg_context: 0,
            row_pad_zero: 0,
            dst_access_mode: 0,
            addr_mod: 0,
            addr_cnt_context: 0,
            zero_write: 0,
            read_intf_sel: 0,
            ovrd_thread_id: 0,
            concat: 0,
            ctxt_ctrl: 0,
            flush: 0,
            last: 0,
        };

        pub const fn cfg_context(mut self, value: u32) -> Self {
            self.cfg_context = value;
            self
        }

        pub const fn row_pad_zero(mut self, value: u32) -> Self {
            self.row_pad_zero = value;
            self
        }

        pub const fn dst_access_mode(mut self, value: u32) -> Self {
            self.dst_access_mode = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn addr_cnt_context(mut self, value: u32) -> Self {
            self.addr_cnt_context = value;
            self
        }

        pub const fn zero_write(mut self, value: u32) -> Self {
            self.zero_write = value;
            self
        }

        pub const fn read_intf_sel(mut self, value: u32) -> Self {
            self.read_intf_sel = value;
            self
        }

        pub const fn ovrd_thread_id(mut self, value: u32) -> Self {
            self.ovrd_thread_id = value;
            self
        }

        pub const fn concat(mut self, value: u32) -> Self {
            self.concat = value;
            self
        }

        pub const fn ctxt_ctrl(mut self, value: u32) -> Self {
            self.ctxt_ctrl = value;
            self
        }

        pub const fn flush(mut self, value: u32) -> Self {
            self.flush = value;
            self
        }

        pub const fn last(mut self, value: u32) -> Self {
            self.last = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::PACR;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.cfg_context) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.cfg_context,
                    width: f.width(),
                });
            }
            word |= f.place(self.cfg_context);
            let f = def.fields()[1];
            if !f.fits(self.row_pad_zero) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.row_pad_zero,
                    width: f.width(),
                });
            }
            word |= f.place(self.row_pad_zero);
            let f = def.fields()[2];
            if !f.fits(self.dst_access_mode) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_access_mode,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_access_mode);
            let f = def.fields()[3];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[4];
            if !f.fits(self.addr_cnt_context) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_cnt_context,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_cnt_context);
            let f = def.fields()[5];
            if !f.fits(self.zero_write) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.zero_write,
                    width: f.width(),
                });
            }
            word |= f.place(self.zero_write);
            let f = def.fields()[6];
            if !f.fits(self.read_intf_sel) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.read_intf_sel,
                    width: f.width(),
                });
            }
            word |= f.place(self.read_intf_sel);
            let f = def.fields()[7];
            if !f.fits(self.ovrd_thread_id) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.ovrd_thread_id,
                    width: f.width(),
                });
            }
            word |= f.place(self.ovrd_thread_id);
            let f = def.fields()[8];
            if !f.fits(self.concat) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.concat,
                    width: f.width(),
                });
            }
            word |= f.place(self.concat);
            let f = def.fields()[9];
            if !f.fits(self.ctxt_ctrl) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.ctxt_ctrl,
                    width: f.width(),
                });
            }
            word |= f.place(self.ctxt_ctrl);
            let f = def.fields()[10];
            if !f.fits(self.flush) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flush,
                    width: f.width(),
                });
            }
            word |= f.place(self.flush);
            let f = def.fields()[11];
            if !f.fits(self.last) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.last,
                    width: f.width(),
                });
            }
            word |= f.place(self.last);
            Ok(Instruction::new(word, def))
        }
    }

    /// `PACR_SETREG`.
    pub const fn pacr_setreg(
        addr_sel: u32,
        value10: u32,
        addr_mid: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::PACR_SETREG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(addr_sel) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_sel,
                width: f.width(),
            });
        }
        word |= f.place(addr_sel);
        let f = def.fields()[1];
        if !f.fits(value10) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: value10,
                width: f.width(),
            });
        }
        word |= f.place(value10);
        let f = def.fields()[2];
        if !f.fits(addr_mid) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_mid,
                width: f.width(),
            });
        }
        word |= f.place(addr_mid);
        Ok(Instruction::new(word, def))
    }

    /// `SFPSHFT2`.
    pub const fn sfpshft2(
        vb: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPSHFT2;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vb) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vb,
                width: f.width(),
            });
        }
        word |= f.place(vb);
        let f = def.fields()[1];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[2];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[3];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPSHFT2b`.
    pub const fn sfpshft2b(imm12: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPSHFT2b;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm12) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm12,
                width: f.width(),
            });
        }
        word |= f.place(imm12);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPLUTFP32_BH`.
    pub const fn sfplutfp32(
        mod1_mirror: u32,
        vd: u32,
        mod1: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPLUTFP32;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mod1_mirror) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1_mirror,
                width: f.width(),
            });
        }
        word |= f.place(mod1_mirror);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `REG2FLOP_Configuration`.
    pub const fn reg2_flop_configuration(
        size_sel: u32,
        th_con_cfg_index: u32,
        input_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::REG2FLOP_Configuration;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(size_sel) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: size_sel,
                width: f.width(),
            });
        }
        word |= f.place(size_sel);
        let f = def.fields()[1];
        if !f.fits(th_con_cfg_index) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: th_con_cfg_index,
                width: f.width(),
            });
        }
        word |= f.place(th_con_cfg_index);
        let f = def.fields()[2];
        if !f.fits(input_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: input_reg,
                width: f.width(),
            });
        }
        word |= f.place(input_reg);
        Ok(Instruction::new(word, def))
    }

    /// `REG2FLOP_ADC`, built field by field.
    ///
    /// 9 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Reg2FlopAdc::ZERO.size_sel(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Reg2FlopAdc {
        size_sel: u32,
        override_thread: u32,
        shift8: u32,
        thread_sel: u32,
        channel: u32,
        adc_sel: u32,
        cr: u32,
        xyzw: u32,
        input_reg: u32,
    }

    impl Reg2FlopAdc {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Reg2FlopAdc {
            size_sel: 0,
            override_thread: 0,
            shift8: 0,
            thread_sel: 0,
            channel: 0,
            adc_sel: 0,
            cr: 0,
            xyzw: 0,
            input_reg: 0,
        };

        pub const fn size_sel(mut self, value: u32) -> Self {
            self.size_sel = value;
            self
        }

        pub const fn override_thread(mut self, value: u32) -> Self {
            self.override_thread = value;
            self
        }

        pub const fn shift8(mut self, value: u32) -> Self {
            self.shift8 = value;
            self
        }

        pub const fn thread_sel(mut self, value: u32) -> Self {
            self.thread_sel = value;
            self
        }

        pub const fn channel(mut self, value: u32) -> Self {
            self.channel = value;
            self
        }

        pub const fn adc_sel(mut self, value: u32) -> Self {
            self.adc_sel = value;
            self
        }

        pub const fn cr(mut self, value: u32) -> Self {
            self.cr = value;
            self
        }

        pub const fn xyzw(mut self, value: u32) -> Self {
            self.xyzw = value;
            self
        }

        pub const fn input_reg(mut self, value: u32) -> Self {
            self.input_reg = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::REG2FLOP_ADC;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.size_sel) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.size_sel,
                    width: f.width(),
                });
            }
            word |= f.place(self.size_sel);
            let f = def.fields()[1];
            if !f.fits(self.override_thread) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.override_thread,
                    width: f.width(),
                });
            }
            word |= f.place(self.override_thread);
            let f = def.fields()[2];
            if !f.fits(self.shift8) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.shift8,
                    width: f.width(),
                });
            }
            word |= f.place(self.shift8);
            let f = def.fields()[3];
            if !f.fits(self.thread_sel) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.thread_sel,
                    width: f.width(),
                });
            }
            word |= f.place(self.thread_sel);
            let f = def.fields()[4];
            if !f.fits(self.channel) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.channel,
                    width: f.width(),
                });
            }
            word |= f.place(self.channel);
            let f = def.fields()[5];
            if !f.fits(self.adc_sel) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.adc_sel,
                    width: f.width(),
                });
            }
            word |= f.place(self.adc_sel);
            let f = def.fields()[6];
            if !f.fits(self.cr) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.cr,
                    width: f.width(),
                });
            }
            word |= f.place(self.cr);
            let f = def.fields()[7];
            if !f.fits(self.xyzw) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.xyzw,
                    width: f.width(),
                });
            }
            word |= f.place(self.xyzw);
            let f = def.fields()[8];
            if !f.fits(self.input_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.input_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.input_reg);
            Ok(Instruction::new(word, def))
        }
    }

    /// `XMOV`.
    pub const fn xmov() -> Result<Instruction, EncodeError> {
        let def = &defs::XMOV;
        let word = def.skeleton();
        Ok(Instruction::new(word, def))
    }

    /// `WRCFG`.
    pub const fn wrcfg(
        input_reg: u32,
        is128_bit: u32,
        cfg_index: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::WRCFG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(input_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: input_reg,
                width: f.width(),
            });
        }
        word |= f.place(input_reg);
        let f = def.fields()[1];
        if !f.fits(is128_bit) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: is128_bit,
                width: f.width(),
            });
        }
        word |= f.place(is128_bit);
        let f = def.fields()[2];
        if !f.fits(cfg_index) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: cfg_index,
                width: f.width(),
            });
        }
        word |= f.place(cfg_index);
        Ok(Instruction::new(word, def))
    }

    /// `RDCFG`.
    pub const fn rdcfg(result_reg: u32, cfg_index: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::RDCFG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[1];
        if !f.fits(cfg_index) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: cfg_index,
                width: f.width(),
            });
        }
        word |= f.place(cfg_index);
        Ok(Instruction::new(word, def))
    }

    /// `SETC16`.
    pub const fn setc16(cfg_index: u32, new_value: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SETC16;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(cfg_index) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: cfg_index,
                width: f.width(),
            });
        }
        word |= f.place(cfg_index);
        let f = def.fields()[1];
        if !f.fits(new_value) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: new_value,
                width: f.width(),
            });
        }
        word |= f.place(new_value);
        Ok(Instruction::new(word, def))
    }

    /// `STREAMWRCFG`.
    pub const fn streamwrcfg(
        stream_select: u32,
        reg_index: u32,
        cfg_index: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::STREAMWRCFG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(stream_select) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: stream_select,
                width: f.width(),
            });
        }
        word |= f.place(stream_select);
        let f = def.fields()[1];
        if !f.fits(reg_index) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: reg_index,
                width: f.width(),
            });
        }
        word |= f.place(reg_index);
        let f = def.fields()[2];
        if !f.fits(cfg_index) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: cfg_index,
                width: f.width(),
            });
        }
        word |= f.place(cfg_index);
        Ok(Instruction::new(word, def))
    }

    /// `CFGSHIFTMASK`, built field by field.
    ///
    /// 6 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Cfgshiftmask::ZERO.mask_mode(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Cfgshiftmask {
        mask_mode: u32,
        alu_mode: u32,
        mask_width: u32,
        rotate_amt: u32,
        scratch_index: u32,
        cfg_index: u32,
    }

    impl Cfgshiftmask {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Cfgshiftmask {
            mask_mode: 0,
            alu_mode: 0,
            mask_width: 0,
            rotate_amt: 0,
            scratch_index: 0,
            cfg_index: 0,
        };

        pub const fn mask_mode(mut self, value: u32) -> Self {
            self.mask_mode = value;
            self
        }

        pub const fn alu_mode(mut self, value: u32) -> Self {
            self.alu_mode = value;
            self
        }

        pub const fn mask_width(mut self, value: u32) -> Self {
            self.mask_width = value;
            self
        }

        pub const fn rotate_amt(mut self, value: u32) -> Self {
            self.rotate_amt = value;
            self
        }

        pub const fn scratch_index(mut self, value: u32) -> Self {
            self.scratch_index = value;
            self
        }

        pub const fn cfg_index(mut self, value: u32) -> Self {
            self.cfg_index = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::CFGSHIFTMASK;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.mask_mode) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.mask_mode,
                    width: f.width(),
                });
            }
            word |= f.place(self.mask_mode);
            let f = def.fields()[1];
            if !f.fits(self.alu_mode) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.alu_mode,
                    width: f.width(),
                });
            }
            word |= f.place(self.alu_mode);
            let f = def.fields()[2];
            if !f.fits(self.mask_width) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.mask_width,
                    width: f.width(),
                });
            }
            word |= f.place(self.mask_width);
            let f = def.fields()[3];
            if !f.fits(self.rotate_amt) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.rotate_amt,
                    width: f.width(),
                });
            }
            word |= f.place(self.rotate_amt);
            let f = def.fields()[4];
            if !f.fits(self.scratch_index) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.scratch_index,
                    width: f.width(),
                });
            }
            word |= f.place(self.scratch_index);
            let f = def.fields()[5];
            if !f.fits(self.cfg_index) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.cfg_index,
                    width: f.width(),
                });
            }
            word |= f.place(self.cfg_index);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SETADC`, built field by field.
    ///
    /// 6 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Setadc::ZERO.pk(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Setadc {
        pk: u32,
        u1: u32,
        u0: u32,
        channel: u32,
        xyzw: u32,
        new_value: u32,
    }

    impl Setadc {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Setadc {
            pk: 0,
            u1: 0,
            u0: 0,
            channel: 0,
            xyzw: 0,
            new_value: 0,
        };

        pub const fn pk(mut self, value: u32) -> Self {
            self.pk = value;
            self
        }

        pub const fn u1(mut self, value: u32) -> Self {
            self.u1 = value;
            self
        }

        pub const fn u0(mut self, value: u32) -> Self {
            self.u0 = value;
            self
        }

        pub const fn channel(mut self, value: u32) -> Self {
            self.channel = value;
            self
        }

        pub const fn xyzw(mut self, value: u32) -> Self {
            self.xyzw = value;
            self
        }

        pub const fn new_value(mut self, value: u32) -> Self {
            self.new_value = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SETADC;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.pk) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.pk,
                    width: f.width(),
                });
            }
            word |= f.place(self.pk);
            let f = def.fields()[1];
            if !f.fits(self.u1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u1,
                    width: f.width(),
                });
            }
            word |= f.place(self.u1);
            let f = def.fields()[2];
            if !f.fits(self.u0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u0,
                    width: f.width(),
                });
            }
            word |= f.place(self.u0);
            let f = def.fields()[3];
            if !f.fits(self.channel) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.channel,
                    width: f.width(),
                });
            }
            word |= f.place(self.channel);
            let f = def.fields()[4];
            if !f.fits(self.xyzw) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.xyzw,
                    width: f.width(),
                });
            }
            word |= f.place(self.xyzw);
            let f = def.fields()[5];
            if !f.fits(self.new_value) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.new_value,
                    width: f.width(),
                });
            }
            word |= f.place(self.new_value);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SETADCXY`, built field by field.
    ///
    /// 12 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Setadcxy::ZERO.pk(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Setadcxy {
        pk: u32,
        u1: u32,
        u0: u32,
        thread_override: u32,
        y1_val: u32,
        x1_val: u32,
        y0_val: u32,
        x0_val: u32,
        y1: u32,
        x1: u32,
        y0: u32,
        x0: u32,
    }

    impl Setadcxy {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Setadcxy {
            pk: 0,
            u1: 0,
            u0: 0,
            thread_override: 0,
            y1_val: 0,
            x1_val: 0,
            y0_val: 0,
            x0_val: 0,
            y1: 0,
            x1: 0,
            y0: 0,
            x0: 0,
        };

        pub const fn pk(mut self, value: u32) -> Self {
            self.pk = value;
            self
        }

        pub const fn u1(mut self, value: u32) -> Self {
            self.u1 = value;
            self
        }

        pub const fn u0(mut self, value: u32) -> Self {
            self.u0 = value;
            self
        }

        pub const fn thread_override(mut self, value: u32) -> Self {
            self.thread_override = value;
            self
        }

        pub const fn y1_val(mut self, value: u32) -> Self {
            self.y1_val = value;
            self
        }

        pub const fn x1_val(mut self, value: u32) -> Self {
            self.x1_val = value;
            self
        }

        pub const fn y0_val(mut self, value: u32) -> Self {
            self.y0_val = value;
            self
        }

        pub const fn x0_val(mut self, value: u32) -> Self {
            self.x0_val = value;
            self
        }

        pub const fn y1(mut self, value: u32) -> Self {
            self.y1 = value;
            self
        }

        pub const fn x1(mut self, value: u32) -> Self {
            self.x1 = value;
            self
        }

        pub const fn y0(mut self, value: u32) -> Self {
            self.y0 = value;
            self
        }

        pub const fn x0(mut self, value: u32) -> Self {
            self.x0 = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SETADCXY;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.pk) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.pk,
                    width: f.width(),
                });
            }
            word |= f.place(self.pk);
            let f = def.fields()[1];
            if !f.fits(self.u1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u1,
                    width: f.width(),
                });
            }
            word |= f.place(self.u1);
            let f = def.fields()[2];
            if !f.fits(self.u0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u0,
                    width: f.width(),
                });
            }
            word |= f.place(self.u0);
            let f = def.fields()[3];
            if !f.fits(self.thread_override) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.thread_override,
                    width: f.width(),
                });
            }
            word |= f.place(self.thread_override);
            let f = def.fields()[4];
            if !f.fits(self.y1_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.y1_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.y1_val);
            let f = def.fields()[5];
            if !f.fits(self.x1_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x1_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.x1_val);
            let f = def.fields()[6];
            if !f.fits(self.y0_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.y0_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.y0_val);
            let f = def.fields()[7];
            if !f.fits(self.x0_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x0_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.x0_val);
            let f = def.fields()[8];
            if !f.fits(self.y1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.y1,
                    width: f.width(),
                });
            }
            word |= f.place(self.y1);
            let f = def.fields()[9];
            if !f.fits(self.x1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x1,
                    width: f.width(),
                });
            }
            word |= f.place(self.x1);
            let f = def.fields()[10];
            if !f.fits(self.y0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.y0,
                    width: f.width(),
                });
            }
            word |= f.place(self.y0);
            let f = def.fields()[11];
            if !f.fits(self.x0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x0,
                    width: f.width(),
                });
            }
            word |= f.place(self.x0);
            Ok(Instruction::new(word, def))
        }
    }

    /// `INCADCXY`, built field by field.
    ///
    /// 8 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Incadcxy::ZERO.pk(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Incadcxy {
        pk: u32,
        u1: u32,
        u0: u32,
        thread_override: u32,
        y1_inc: u32,
        x1_inc: u32,
        y0_inc: u32,
        x0_inc: u32,
    }

    impl Incadcxy {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Incadcxy {
            pk: 0,
            u1: 0,
            u0: 0,
            thread_override: 0,
            y1_inc: 0,
            x1_inc: 0,
            y0_inc: 0,
            x0_inc: 0,
        };

        pub const fn pk(mut self, value: u32) -> Self {
            self.pk = value;
            self
        }

        pub const fn u1(mut self, value: u32) -> Self {
            self.u1 = value;
            self
        }

        pub const fn u0(mut self, value: u32) -> Self {
            self.u0 = value;
            self
        }

        pub const fn thread_override(mut self, value: u32) -> Self {
            self.thread_override = value;
            self
        }

        pub const fn y1_inc(mut self, value: u32) -> Self {
            self.y1_inc = value;
            self
        }

        pub const fn x1_inc(mut self, value: u32) -> Self {
            self.x1_inc = value;
            self
        }

        pub const fn y0_inc(mut self, value: u32) -> Self {
            self.y0_inc = value;
            self
        }

        pub const fn x0_inc(mut self, value: u32) -> Self {
            self.x0_inc = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::INCADCXY;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.pk) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.pk,
                    width: f.width(),
                });
            }
            word |= f.place(self.pk);
            let f = def.fields()[1];
            if !f.fits(self.u1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u1,
                    width: f.width(),
                });
            }
            word |= f.place(self.u1);
            let f = def.fields()[2];
            if !f.fits(self.u0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u0,
                    width: f.width(),
                });
            }
            word |= f.place(self.u0);
            let f = def.fields()[3];
            if !f.fits(self.thread_override) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.thread_override,
                    width: f.width(),
                });
            }
            word |= f.place(self.thread_override);
            let f = def.fields()[4];
            if !f.fits(self.y1_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.y1_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.y1_inc);
            let f = def.fields()[5];
            if !f.fits(self.x1_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x1_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.x1_inc);
            let f = def.fields()[6];
            if !f.fits(self.y0_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.y0_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.y0_inc);
            let f = def.fields()[7];
            if !f.fits(self.x0_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x0_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.x0_inc);
            Ok(Instruction::new(word, def))
        }
    }

    /// `ADDRCRXY`, built field by field.
    ///
    /// 12 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Addrcrxy::ZERO.pk(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Addrcrxy {
        pk: u32,
        u1: u32,
        u0: u32,
        thread_override: u32,
        y1_inc: u32,
        x1_inc: u32,
        y0_inc: u32,
        x0_inc: u32,
        y1: u32,
        x1: u32,
        y0: u32,
        x0: u32,
    }

    impl Addrcrxy {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Addrcrxy {
            pk: 0,
            u1: 0,
            u0: 0,
            thread_override: 0,
            y1_inc: 0,
            x1_inc: 0,
            y0_inc: 0,
            x0_inc: 0,
            y1: 0,
            x1: 0,
            y0: 0,
            x0: 0,
        };

        pub const fn pk(mut self, value: u32) -> Self {
            self.pk = value;
            self
        }

        pub const fn u1(mut self, value: u32) -> Self {
            self.u1 = value;
            self
        }

        pub const fn u0(mut self, value: u32) -> Self {
            self.u0 = value;
            self
        }

        pub const fn thread_override(mut self, value: u32) -> Self {
            self.thread_override = value;
            self
        }

        pub const fn y1_inc(mut self, value: u32) -> Self {
            self.y1_inc = value;
            self
        }

        pub const fn x1_inc(mut self, value: u32) -> Self {
            self.x1_inc = value;
            self
        }

        pub const fn y0_inc(mut self, value: u32) -> Self {
            self.y0_inc = value;
            self
        }

        pub const fn x0_inc(mut self, value: u32) -> Self {
            self.x0_inc = value;
            self
        }

        pub const fn y1(mut self, value: u32) -> Self {
            self.y1 = value;
            self
        }

        pub const fn x1(mut self, value: u32) -> Self {
            self.x1 = value;
            self
        }

        pub const fn y0(mut self, value: u32) -> Self {
            self.y0 = value;
            self
        }

        pub const fn x0(mut self, value: u32) -> Self {
            self.x0 = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::ADDRCRXY;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.pk) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.pk,
                    width: f.width(),
                });
            }
            word |= f.place(self.pk);
            let f = def.fields()[1];
            if !f.fits(self.u1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u1,
                    width: f.width(),
                });
            }
            word |= f.place(self.u1);
            let f = def.fields()[2];
            if !f.fits(self.u0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u0,
                    width: f.width(),
                });
            }
            word |= f.place(self.u0);
            let f = def.fields()[3];
            if !f.fits(self.thread_override) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.thread_override,
                    width: f.width(),
                });
            }
            word |= f.place(self.thread_override);
            let f = def.fields()[4];
            if !f.fits(self.y1_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.y1_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.y1_inc);
            let f = def.fields()[5];
            if !f.fits(self.x1_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x1_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.x1_inc);
            let f = def.fields()[6];
            if !f.fits(self.y0_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.y0_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.y0_inc);
            let f = def.fields()[7];
            if !f.fits(self.x0_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x0_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.x0_inc);
            let f = def.fields()[8];
            if !f.fits(self.y1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.y1,
                    width: f.width(),
                });
            }
            word |= f.place(self.y1);
            let f = def.fields()[9];
            if !f.fits(self.x1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x1,
                    width: f.width(),
                });
            }
            word |= f.place(self.x1);
            let f = def.fields()[10];
            if !f.fits(self.y0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.y0,
                    width: f.width(),
                });
            }
            word |= f.place(self.y0);
            let f = def.fields()[11];
            if !f.fits(self.x0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x0,
                    width: f.width(),
                });
            }
            word |= f.place(self.x0);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SETADCZW`, built field by field.
    ///
    /// 12 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Setadczw::ZERO.pk(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Setadczw {
        pk: u32,
        u1: u32,
        u0: u32,
        thread_override: u32,
        w1_val: u32,
        z1_val: u32,
        w0_val: u32,
        z0_val: u32,
        w1: u32,
        z1: u32,
        w0: u32,
        z0: u32,
    }

    impl Setadczw {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Setadczw {
            pk: 0,
            u1: 0,
            u0: 0,
            thread_override: 0,
            w1_val: 0,
            z1_val: 0,
            w0_val: 0,
            z0_val: 0,
            w1: 0,
            z1: 0,
            w0: 0,
            z0: 0,
        };

        pub const fn pk(mut self, value: u32) -> Self {
            self.pk = value;
            self
        }

        pub const fn u1(mut self, value: u32) -> Self {
            self.u1 = value;
            self
        }

        pub const fn u0(mut self, value: u32) -> Self {
            self.u0 = value;
            self
        }

        pub const fn thread_override(mut self, value: u32) -> Self {
            self.thread_override = value;
            self
        }

        pub const fn w1_val(mut self, value: u32) -> Self {
            self.w1_val = value;
            self
        }

        pub const fn z1_val(mut self, value: u32) -> Self {
            self.z1_val = value;
            self
        }

        pub const fn w0_val(mut self, value: u32) -> Self {
            self.w0_val = value;
            self
        }

        pub const fn z0_val(mut self, value: u32) -> Self {
            self.z0_val = value;
            self
        }

        pub const fn w1(mut self, value: u32) -> Self {
            self.w1 = value;
            self
        }

        pub const fn z1(mut self, value: u32) -> Self {
            self.z1 = value;
            self
        }

        pub const fn w0(mut self, value: u32) -> Self {
            self.w0 = value;
            self
        }

        pub const fn z0(mut self, value: u32) -> Self {
            self.z0 = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SETADCZW;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.pk) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.pk,
                    width: f.width(),
                });
            }
            word |= f.place(self.pk);
            let f = def.fields()[1];
            if !f.fits(self.u1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u1,
                    width: f.width(),
                });
            }
            word |= f.place(self.u1);
            let f = def.fields()[2];
            if !f.fits(self.u0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u0,
                    width: f.width(),
                });
            }
            word |= f.place(self.u0);
            let f = def.fields()[3];
            if !f.fits(self.thread_override) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.thread_override,
                    width: f.width(),
                });
            }
            word |= f.place(self.thread_override);
            let f = def.fields()[4];
            if !f.fits(self.w1_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.w1_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.w1_val);
            let f = def.fields()[5];
            if !f.fits(self.z1_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.z1_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.z1_val);
            let f = def.fields()[6];
            if !f.fits(self.w0_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.w0_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.w0_val);
            let f = def.fields()[7];
            if !f.fits(self.z0_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.z0_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.z0_val);
            let f = def.fields()[8];
            if !f.fits(self.w1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.w1,
                    width: f.width(),
                });
            }
            word |= f.place(self.w1);
            let f = def.fields()[9];
            if !f.fits(self.z1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.z1,
                    width: f.width(),
                });
            }
            word |= f.place(self.z1);
            let f = def.fields()[10];
            if !f.fits(self.w0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.w0,
                    width: f.width(),
                });
            }
            word |= f.place(self.w0);
            let f = def.fields()[11];
            if !f.fits(self.z0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.z0,
                    width: f.width(),
                });
            }
            word |= f.place(self.z0);
            Ok(Instruction::new(word, def))
        }
    }

    /// `INCADCZW`, built field by field.
    ///
    /// 8 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Incadczw::ZERO.pk(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Incadczw {
        pk: u32,
        u1: u32,
        u0: u32,
        thread_override: u32,
        w1_inc: u32,
        z1_inc: u32,
        w0_inc: u32,
        z0_inc: u32,
    }

    impl Incadczw {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Incadczw {
            pk: 0,
            u1: 0,
            u0: 0,
            thread_override: 0,
            w1_inc: 0,
            z1_inc: 0,
            w0_inc: 0,
            z0_inc: 0,
        };

        pub const fn pk(mut self, value: u32) -> Self {
            self.pk = value;
            self
        }

        pub const fn u1(mut self, value: u32) -> Self {
            self.u1 = value;
            self
        }

        pub const fn u0(mut self, value: u32) -> Self {
            self.u0 = value;
            self
        }

        pub const fn thread_override(mut self, value: u32) -> Self {
            self.thread_override = value;
            self
        }

        pub const fn w1_inc(mut self, value: u32) -> Self {
            self.w1_inc = value;
            self
        }

        pub const fn z1_inc(mut self, value: u32) -> Self {
            self.z1_inc = value;
            self
        }

        pub const fn w0_inc(mut self, value: u32) -> Self {
            self.w0_inc = value;
            self
        }

        pub const fn z0_inc(mut self, value: u32) -> Self {
            self.z0_inc = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::INCADCZW;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.pk) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.pk,
                    width: f.width(),
                });
            }
            word |= f.place(self.pk);
            let f = def.fields()[1];
            if !f.fits(self.u1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u1,
                    width: f.width(),
                });
            }
            word |= f.place(self.u1);
            let f = def.fields()[2];
            if !f.fits(self.u0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u0,
                    width: f.width(),
                });
            }
            word |= f.place(self.u0);
            let f = def.fields()[3];
            if !f.fits(self.thread_override) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.thread_override,
                    width: f.width(),
                });
            }
            word |= f.place(self.thread_override);
            let f = def.fields()[4];
            if !f.fits(self.w1_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.w1_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.w1_inc);
            let f = def.fields()[5];
            if !f.fits(self.z1_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.z1_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.z1_inc);
            let f = def.fields()[6];
            if !f.fits(self.w0_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.w0_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.w0_inc);
            let f = def.fields()[7];
            if !f.fits(self.z0_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.z0_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.z0_inc);
            Ok(Instruction::new(word, def))
        }
    }

    /// `ADDRCRZW`, built field by field.
    ///
    /// 12 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Addrcrzw::ZERO.pk(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Addrcrzw {
        pk: u32,
        u1: u32,
        u0: u32,
        thread_override: u32,
        w1_inc: u32,
        z1_inc: u32,
        w0_inc: u32,
        z0_inc: u32,
        w1: u32,
        z1: u32,
        w0: u32,
        z0: u32,
    }

    impl Addrcrzw {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Addrcrzw {
            pk: 0,
            u1: 0,
            u0: 0,
            thread_override: 0,
            w1_inc: 0,
            z1_inc: 0,
            w0_inc: 0,
            z0_inc: 0,
            w1: 0,
            z1: 0,
            w0: 0,
            z0: 0,
        };

        pub const fn pk(mut self, value: u32) -> Self {
            self.pk = value;
            self
        }

        pub const fn u1(mut self, value: u32) -> Self {
            self.u1 = value;
            self
        }

        pub const fn u0(mut self, value: u32) -> Self {
            self.u0 = value;
            self
        }

        pub const fn thread_override(mut self, value: u32) -> Self {
            self.thread_override = value;
            self
        }

        pub const fn w1_inc(mut self, value: u32) -> Self {
            self.w1_inc = value;
            self
        }

        pub const fn z1_inc(mut self, value: u32) -> Self {
            self.z1_inc = value;
            self
        }

        pub const fn w0_inc(mut self, value: u32) -> Self {
            self.w0_inc = value;
            self
        }

        pub const fn z0_inc(mut self, value: u32) -> Self {
            self.z0_inc = value;
            self
        }

        pub const fn w1(mut self, value: u32) -> Self {
            self.w1 = value;
            self
        }

        pub const fn z1(mut self, value: u32) -> Self {
            self.z1 = value;
            self
        }

        pub const fn w0(mut self, value: u32) -> Self {
            self.w0 = value;
            self
        }

        pub const fn z0(mut self, value: u32) -> Self {
            self.z0 = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::ADDRCRZW;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.pk) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.pk,
                    width: f.width(),
                });
            }
            word |= f.place(self.pk);
            let f = def.fields()[1];
            if !f.fits(self.u1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u1,
                    width: f.width(),
                });
            }
            word |= f.place(self.u1);
            let f = def.fields()[2];
            if !f.fits(self.u0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u0,
                    width: f.width(),
                });
            }
            word |= f.place(self.u0);
            let f = def.fields()[3];
            if !f.fits(self.thread_override) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.thread_override,
                    width: f.width(),
                });
            }
            word |= f.place(self.thread_override);
            let f = def.fields()[4];
            if !f.fits(self.w1_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.w1_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.w1_inc);
            let f = def.fields()[5];
            if !f.fits(self.z1_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.z1_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.z1_inc);
            let f = def.fields()[6];
            if !f.fits(self.w0_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.w0_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.w0_inc);
            let f = def.fields()[7];
            if !f.fits(self.z0_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.z0_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.z0_inc);
            let f = def.fields()[8];
            if !f.fits(self.w1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.w1,
                    width: f.width(),
                });
            }
            word |= f.place(self.w1);
            let f = def.fields()[9];
            if !f.fits(self.z1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.z1,
                    width: f.width(),
                });
            }
            word |= f.place(self.z1);
            let f = def.fields()[10];
            if !f.fits(self.w0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.w0,
                    width: f.width(),
                });
            }
            word |= f.place(self.w0);
            let f = def.fields()[11];
            if !f.fits(self.z0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.z0,
                    width: f.width(),
                });
            }
            word |= f.place(self.z0);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SETADCXX`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Setadcxx::ZERO.pk(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Setadcxx {
        pk: u32,
        u1: u32,
        u0: u32,
        x1_val: u32,
        x0_val: u32,
    }

    impl Setadcxx {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Setadcxx {
            pk: 0,
            u1: 0,
            u0: 0,
            x1_val: 0,
            x0_val: 0,
        };

        pub const fn pk(mut self, value: u32) -> Self {
            self.pk = value;
            self
        }

        pub const fn u1(mut self, value: u32) -> Self {
            self.u1 = value;
            self
        }

        pub const fn u0(mut self, value: u32) -> Self {
            self.u0 = value;
            self
        }

        pub const fn x1_val(mut self, value: u32) -> Self {
            self.x1_val = value;
            self
        }

        pub const fn x0_val(mut self, value: u32) -> Self {
            self.x0_val = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SETADCXX;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.pk) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.pk,
                    width: f.width(),
                });
            }
            word |= f.place(self.pk);
            let f = def.fields()[1];
            if !f.fits(self.u1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u1,
                    width: f.width(),
                });
            }
            word |= f.place(self.u1);
            let f = def.fields()[2];
            if !f.fits(self.u0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.u0,
                    width: f.width(),
                });
            }
            word |= f.place(self.u0);
            let f = def.fields()[3];
            if !f.fits(self.x1_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x1_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.x1_val);
            let f = def.fields()[4];
            if !f.fits(self.x0_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.x0_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.x0_val);
            Ok(Instruction::new(word, def))
        }
    }

    /// `NOP`.
    pub const fn nop() -> Result<Instruction, EncodeError> {
        let def = &defs::NOP;
        let word = def.skeleton();
        Ok(Instruction::new(word, def))
    }

    /// `SEMPOST`.
    pub const fn sempost(semaphore_mask: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SEMPOST;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(semaphore_mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: semaphore_mask,
                width: f.width(),
            });
        }
        word |= f.place(semaphore_mask);
        Ok(Instruction::new(word, def))
    }

    /// `SEMGET`.
    pub const fn semget(semaphore_mask: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SEMGET;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(semaphore_mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: semaphore_mask,
                width: f.width(),
            });
        }
        word |= f.place(semaphore_mask);
        Ok(Instruction::new(word, def))
    }

    /// `SETDVALID`.
    pub const fn setdvalid(flip_src_b: u32, flip_src_a: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SETDVALID;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(flip_src_b) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: flip_src_b,
                width: f.width(),
            });
        }
        word |= f.place(flip_src_b);
        let f = def.fields()[1];
        if !f.fits(flip_src_a) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: flip_src_a,
                width: f.width(),
            });
        }
        word |= f.place(flip_src_a);
        Ok(Instruction::new(word, def))
    }

    /// `GATESRCRST`.
    pub const fn gatesrcrst(invalidate_src_b_cache: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::GATESRCRST;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(invalidate_src_b_cache) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: invalidate_src_b_cache,
                width: f.width(),
            });
        }
        word |= f.place(invalidate_src_b_cache);
        Ok(Instruction::new(word, def))
    }

    /// `ZEROSRC`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Zerosrc::ZERO.negative_inf_src_a(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Zerosrc {
        negative_inf_src_a: u32,
        single_bank_matrix_unit: u32,
        both_banks: u32,
        clear_src_b: u32,
        clear_src_a: u32,
    }

    impl Zerosrc {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Zerosrc {
            negative_inf_src_a: 0,
            single_bank_matrix_unit: 0,
            both_banks: 0,
            clear_src_b: 0,
            clear_src_a: 0,
        };

        pub const fn negative_inf_src_a(mut self, value: u32) -> Self {
            self.negative_inf_src_a = value;
            self
        }

        pub const fn single_bank_matrix_unit(mut self, value: u32) -> Self {
            self.single_bank_matrix_unit = value;
            self
        }

        pub const fn both_banks(mut self, value: u32) -> Self {
            self.both_banks = value;
            self
        }

        pub const fn clear_src_b(mut self, value: u32) -> Self {
            self.clear_src_b = value;
            self
        }

        pub const fn clear_src_a(mut self, value: u32) -> Self {
            self.clear_src_a = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::ZEROSRC;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.negative_inf_src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.negative_inf_src_a,
                    width: f.width(),
                });
            }
            word |= f.place(self.negative_inf_src_a);
            let f = def.fields()[1];
            if !f.fits(self.single_bank_matrix_unit) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.single_bank_matrix_unit,
                    width: f.width(),
                });
            }
            word |= f.place(self.single_bank_matrix_unit);
            let f = def.fields()[2];
            if !f.fits(self.both_banks) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.both_banks,
                    width: f.width(),
                });
            }
            word |= f.place(self.both_banks);
            let f = def.fields()[3];
            if !f.fits(self.clear_src_b) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.clear_src_b,
                    width: f.width(),
                });
            }
            word |= f.place(self.clear_src_b);
            let f = def.fields()[4];
            if !f.fits(self.clear_src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.clear_src_a,
                    width: f.width(),
                });
            }
            word |= f.place(self.clear_src_a);
            Ok(Instruction::new(word, def))
        }
    }

    /// `TRNSPSRCB`.
    pub const fn trnspsrcb() -> Result<Instruction, EncodeError> {
        let def = &defs::TRNSPSRCB;
        let word = def.skeleton();
        Ok(Instruction::new(word, def))
    }

    /// `SHIFTXA`.
    pub const fn shiftxa(direction: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SHIFTXA;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(direction) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: direction,
                width: f.width(),
            });
        }
        word |= f.place(direction);
        Ok(Instruction::new(word, def))
    }

    /// `CLREXPHIST`.
    pub const fn clrexphist() -> Result<Instruction, EncodeError> {
        let def = &defs::CLREXPHIST;
        let word = def.skeleton();
        Ok(Instruction::new(word, def))
    }

    /// `CLEARDVALID`.
    pub const fn cleardvalid(
        flip_src_b: u32,
        flip_src_a: u32,
        keep_reading_same_src: u32,
        reset: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::CLEARDVALID;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(flip_src_b) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: flip_src_b,
                width: f.width(),
            });
        }
        word |= f.place(flip_src_b);
        let f = def.fields()[1];
        if !f.fits(flip_src_a) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: flip_src_a,
                width: f.width(),
            });
        }
        word |= f.place(flip_src_a);
        let f = def.fields()[2];
        if !f.fits(keep_reading_same_src) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: keep_reading_same_src,
                width: f.width(),
            });
        }
        word |= f.place(keep_reading_same_src);
        let f = def.fields()[3];
        if !f.fits(reset) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: reset,
                width: f.width(),
            });
        }
        word |= f.place(reset);
        Ok(Instruction::new(word, def))
    }

    /// `SETRWC`, built field by field.
    ///
    /// 13 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Setrwc::ZERO.flip_src_b(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Setrwc {
        flip_src_b: u32,
        flip_src_a: u32,
        dst_cto_cr: u32,
        dst_cr: u32,
        src_b_cr: u32,
        src_a_cr: u32,
        dst_val: u32,
        src_b_val: u32,
        src_a_val: u32,
        fidelity: u32,
        dst: u32,
        src_b: u32,
        src_a: u32,
    }

    impl Setrwc {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Setrwc {
            flip_src_b: 0,
            flip_src_a: 0,
            dst_cto_cr: 0,
            dst_cr: 0,
            src_b_cr: 0,
            src_a_cr: 0,
            dst_val: 0,
            src_b_val: 0,
            src_a_val: 0,
            fidelity: 0,
            dst: 0,
            src_b: 0,
            src_a: 0,
        };

        pub const fn flip_src_b(mut self, value: u32) -> Self {
            self.flip_src_b = value;
            self
        }

        pub const fn flip_src_a(mut self, value: u32) -> Self {
            self.flip_src_a = value;
            self
        }

        pub const fn dst_cto_cr(mut self, value: u32) -> Self {
            self.dst_cto_cr = value;
            self
        }

        pub const fn dst_cr(mut self, value: u32) -> Self {
            self.dst_cr = value;
            self
        }

        pub const fn src_b_cr(mut self, value: u32) -> Self {
            self.src_b_cr = value;
            self
        }

        pub const fn src_a_cr(mut self, value: u32) -> Self {
            self.src_a_cr = value;
            self
        }

        pub const fn dst_val(mut self, value: u32) -> Self {
            self.dst_val = value;
            self
        }

        pub const fn src_b_val(mut self, value: u32) -> Self {
            self.src_b_val = value;
            self
        }

        pub const fn src_a_val(mut self, value: u32) -> Self {
            self.src_a_val = value;
            self
        }

        pub const fn fidelity(mut self, value: u32) -> Self {
            self.fidelity = value;
            self
        }

        pub const fn dst(mut self, value: u32) -> Self {
            self.dst = value;
            self
        }

        pub const fn src_b(mut self, value: u32) -> Self {
            self.src_b = value;
            self
        }

        pub const fn src_a(mut self, value: u32) -> Self {
            self.src_a = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SETRWC;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.flip_src_b) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_b,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_b);
            let f = def.fields()[1];
            if !f.fits(self.flip_src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_a,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_a);
            let f = def.fields()[2];
            if !f.fits(self.dst_cto_cr) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_cto_cr,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_cto_cr);
            let f = def.fields()[3];
            if !f.fits(self.dst_cr) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_cr,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_cr);
            let f = def.fields()[4];
            if !f.fits(self.src_b_cr) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_b_cr,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_b_cr);
            let f = def.fields()[5];
            if !f.fits(self.src_a_cr) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_a_cr,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_a_cr);
            let f = def.fields()[6];
            if !f.fits(self.dst_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_val);
            let f = def.fields()[7];
            if !f.fits(self.src_b_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_b_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_b_val);
            let f = def.fields()[8];
            if !f.fits(self.src_a_val) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_a_val,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_a_val);
            let f = def.fields()[9];
            if !f.fits(self.fidelity) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.fidelity,
                    width: f.width(),
                });
            }
            word |= f.place(self.fidelity);
            let f = def.fields()[10];
            if !f.fits(self.dst) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst);
            let f = def.fields()[11];
            if !f.fits(self.src_b) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_b,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_b);
            let f = def.fields()[12];
            if !f.fits(self.src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_a,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_a);
            Ok(Instruction::new(word, def))
        }
    }

    /// `INCRWC`, built field by field.
    ///
    /// 6 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Incrwc::ZERO.dst_cr(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Incrwc {
        dst_cr: u32,
        src_b_cr: u32,
        src_a_cr: u32,
        dst_inc: u32,
        src_b_inc: u32,
        src_a_inc: u32,
    }

    impl Incrwc {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Incrwc {
            dst_cr: 0,
            src_b_cr: 0,
            src_a_cr: 0,
            dst_inc: 0,
            src_b_inc: 0,
            src_a_inc: 0,
        };

        pub const fn dst_cr(mut self, value: u32) -> Self {
            self.dst_cr = value;
            self
        }

        pub const fn src_b_cr(mut self, value: u32) -> Self {
            self.src_b_cr = value;
            self
        }

        pub const fn src_a_cr(mut self, value: u32) -> Self {
            self.src_a_cr = value;
            self
        }

        pub const fn dst_inc(mut self, value: u32) -> Self {
            self.dst_inc = value;
            self
        }

        pub const fn src_b_inc(mut self, value: u32) -> Self {
            self.src_b_inc = value;
            self
        }

        pub const fn src_a_inc(mut self, value: u32) -> Self {
            self.src_a_inc = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::INCRWC;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.dst_cr) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_cr,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_cr);
            let f = def.fields()[1];
            if !f.fits(self.src_b_cr) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_b_cr,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_b_cr);
            let f = def.fields()[2];
            if !f.fits(self.src_a_cr) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_a_cr,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_a_cr);
            let f = def.fields()[3];
            if !f.fits(self.dst_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_inc);
            let f = def.fields()[4];
            if !f.fits(self.src_b_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_b_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_b_inc);
            let f = def.fields()[5];
            if !f.fits(self.src_a_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_a_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_a_inc);
            Ok(Instruction::new(word, def))
        }
    }

    /// `LOADIND`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Loadind::ZERO.size(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Loadind {
        size: u32,
        offset_half_reg: u32,
        offset_increment: u32,
        result_reg: u32,
        addr_reg: u32,
    }

    impl Loadind {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Loadind {
            size: 0,
            offset_half_reg: 0,
            offset_increment: 0,
            result_reg: 0,
            addr_reg: 0,
        };

        pub const fn size(mut self, value: u32) -> Self {
            self.size = value;
            self
        }

        pub const fn offset_half_reg(mut self, value: u32) -> Self {
            self.offset_half_reg = value;
            self
        }

        pub const fn offset_increment(mut self, value: u32) -> Self {
            self.offset_increment = value;
            self
        }

        pub const fn result_reg(mut self, value: u32) -> Self {
            self.result_reg = value;
            self
        }

        pub const fn addr_reg(mut self, value: u32) -> Self {
            self.addr_reg = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::LOADIND;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.size) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.size,
                    width: f.width(),
                });
            }
            word |= f.place(self.size);
            let f = def.fields()[1];
            if !f.fits(self.offset_half_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.offset_half_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.offset_half_reg);
            let f = def.fields()[2];
            if !f.fits(self.offset_increment) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.offset_increment,
                    width: f.width(),
                });
            }
            word |= f.place(self.offset_increment);
            let f = def.fields()[3];
            if !f.fits(self.result_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.result_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.result_reg);
            let f = def.fields()[4];
            if !f.fits(self.addr_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_reg);
            Ok(Instruction::new(word, def))
        }
    }

    /// `STOREIND_L1`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `StoreindL1::ZERO.size(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct StoreindL1 {
        size: u32,
        offset_half_reg: u32,
        offset_increment: u32,
        data_reg: u32,
        addr_reg: u32,
    }

    impl StoreindL1 {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = StoreindL1 {
            size: 0,
            offset_half_reg: 0,
            offset_increment: 0,
            data_reg: 0,
            addr_reg: 0,
        };

        pub const fn size(mut self, value: u32) -> Self {
            self.size = value;
            self
        }

        pub const fn offset_half_reg(mut self, value: u32) -> Self {
            self.offset_half_reg = value;
            self
        }

        pub const fn offset_increment(mut self, value: u32) -> Self {
            self.offset_increment = value;
            self
        }

        pub const fn data_reg(mut self, value: u32) -> Self {
            self.data_reg = value;
            self
        }

        pub const fn addr_reg(mut self, value: u32) -> Self {
            self.addr_reg = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::STOREIND_L1;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.size) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.size,
                    width: f.width(),
                });
            }
            word |= f.place(self.size);
            let f = def.fields()[1];
            if !f.fits(self.offset_half_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.offset_half_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.offset_half_reg);
            let f = def.fields()[2];
            if !f.fits(self.offset_increment) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.offset_increment,
                    width: f.width(),
                });
            }
            word |= f.place(self.offset_increment);
            let f = def.fields()[3];
            if !f.fits(self.data_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.data_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.data_reg);
            let f = def.fields()[4];
            if !f.fits(self.addr_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_reg);
            Ok(Instruction::new(word, def))
        }
    }

    /// `STOREIND_MMIO`.
    pub const fn storeind_mmio(
        offset_half_reg: u32,
        offset_increment: u32,
        data_reg: u32,
        addr_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::STOREIND_MMIO;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(offset_half_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: offset_half_reg,
                width: f.width(),
            });
        }
        word |= f.place(offset_half_reg);
        let f = def.fields()[1];
        if !f.fits(offset_increment) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: offset_increment,
                width: f.width(),
            });
        }
        word |= f.place(offset_increment);
        let f = def.fields()[2];
        if !f.fits(data_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: data_reg,
                width: f.width(),
            });
        }
        word |= f.place(data_reg);
        let f = def.fields()[3];
        if !f.fits(addr_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_reg,
                width: f.width(),
            });
        }
        word |= f.place(addr_reg);
        Ok(Instruction::new(word, def))
    }

    /// `STOREIND_Src`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `StoreindSrc::ZERO.store_to_src_b(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct StoreindSrc {
        store_to_src_b: u32,
        offset_half_reg: u32,
        offset_increment: u32,
        data_reg: u32,
        addr_reg: u32,
    }

    impl StoreindSrc {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = StoreindSrc {
            store_to_src_b: 0,
            offset_half_reg: 0,
            offset_increment: 0,
            data_reg: 0,
            addr_reg: 0,
        };

        pub const fn store_to_src_b(mut self, value: u32) -> Self {
            self.store_to_src_b = value;
            self
        }

        pub const fn offset_half_reg(mut self, value: u32) -> Self {
            self.offset_half_reg = value;
            self
        }

        pub const fn offset_increment(mut self, value: u32) -> Self {
            self.offset_increment = value;
            self
        }

        pub const fn data_reg(mut self, value: u32) -> Self {
            self.data_reg = value;
            self
        }

        pub const fn addr_reg(mut self, value: u32) -> Self {
            self.addr_reg = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::STOREIND_Src;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.store_to_src_b) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.store_to_src_b,
                    width: f.width(),
                });
            }
            word |= f.place(self.store_to_src_b);
            let f = def.fields()[1];
            if !f.fits(self.offset_half_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.offset_half_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.offset_half_reg);
            let f = def.fields()[2];
            if !f.fits(self.offset_increment) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.offset_increment,
                    width: f.width(),
                });
            }
            word |= f.place(self.offset_increment);
            let f = def.fields()[3];
            if !f.fits(self.data_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.data_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.data_reg);
            let f = def.fields()[4];
            if !f.fits(self.addr_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_reg);
            Ok(Instruction::new(word, def))
        }
    }

    /// `LOADREG`.
    pub const fn loadreg(result_reg: u32, addr_lo: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::LOADREG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[1];
        if !f.fits(addr_lo) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_lo,
                width: f.width(),
            });
        }
        word |= f.place(addr_lo);
        Ok(Instruction::new(word, def))
    }

    /// `STOREREG`.
    pub const fn storereg(data_reg: u32, addr_lo: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::STOREREG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(data_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: data_reg,
                width: f.width(),
            });
        }
        word |= f.place(data_reg);
        let f = def.fields()[1];
        if !f.fits(addr_lo) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_lo,
                width: f.width(),
            });
        }
        word |= f.place(addr_lo);
        Ok(Instruction::new(word, def))
    }

    /// `FLUSHDMA`.
    pub const fn flushdma(condition_mask: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::FLUSHDMA;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(condition_mask) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: condition_mask,
                width: f.width(),
            });
        }
        word |= f.place(condition_mask);
        Ok(Instruction::new(word, def))
    }

    /// `DMANOP`.
    pub const fn dmanop() -> Result<Instruction, EncodeError> {
        let def = &defs::DMANOP;
        let word = def.skeleton();
        Ok(Instruction::new(word, def))
    }

    /// `ADDDMAREG`.
    pub const fn adddmareg(
        result_reg: u32,
        right_reg: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::ADDDMAREG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[1];
        if !f.fits(right_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_reg,
                width: f.width(),
            });
        }
        word |= f.place(right_reg);
        let f = def.fields()[2];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `ADDDMAREGi`.
    pub const fn adddmare_gi(
        result_reg: u32,
        right_imm6: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::ADDDMAREGi;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[1];
        if !f.fits(right_imm6) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_imm6,
                width: f.width(),
            });
        }
        word |= f.place(right_imm6);
        let f = def.fields()[2];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `SUBDMAREG`.
    pub const fn subdmareg(
        result_reg: u32,
        right_reg: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SUBDMAREG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[1];
        if !f.fits(right_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_reg,
                width: f.width(),
            });
        }
        word |= f.place(right_reg);
        let f = def.fields()[2];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `SUBDMAREGi`.
    pub const fn subdmare_gi(
        result_reg: u32,
        right_imm6: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SUBDMAREGi;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[1];
        if !f.fits(right_imm6) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_imm6,
                width: f.width(),
            });
        }
        word |= f.place(right_imm6);
        let f = def.fields()[2];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `MULDMAREG`.
    pub const fn muldmareg(
        result_reg: u32,
        right_reg: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::MULDMAREG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[1];
        if !f.fits(right_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_reg,
                width: f.width(),
            });
        }
        word |= f.place(right_reg);
        let f = def.fields()[2];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `MULDMAREGi`.
    pub const fn muldmare_gi(
        result_reg: u32,
        right_imm6: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::MULDMAREGi;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[1];
        if !f.fits(right_imm6) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_imm6,
                width: f.width(),
            });
        }
        word |= f.place(right_imm6);
        let f = def.fields()[2];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `BITWOPDMAREG`.
    pub const fn bitwopdmareg(
        mode: u32,
        result_reg: u32,
        right_reg: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::BITWOPDMAREG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mode) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mode,
                width: f.width(),
            });
        }
        word |= f.place(mode);
        let f = def.fields()[1];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[2];
        if !f.fits(right_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_reg,
                width: f.width(),
            });
        }
        word |= f.place(right_reg);
        let f = def.fields()[3];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `BITWOPDMAREGi`.
    pub const fn bitwopdmare_gi(
        mode: u32,
        result_reg: u32,
        right_imm6: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::BITWOPDMAREGi;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mode) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mode,
                width: f.width(),
            });
        }
        word |= f.place(mode);
        let f = def.fields()[1];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[2];
        if !f.fits(right_imm6) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_imm6,
                width: f.width(),
            });
        }
        word |= f.place(right_imm6);
        let f = def.fields()[3];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `SHIFTDMAREG`.
    pub const fn shiftdmareg(
        mode: u32,
        result_reg: u32,
        right_reg: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SHIFTDMAREG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mode) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mode,
                width: f.width(),
            });
        }
        word |= f.place(mode);
        let f = def.fields()[1];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[2];
        if !f.fits(right_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_reg,
                width: f.width(),
            });
        }
        word |= f.place(right_reg);
        let f = def.fields()[3];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `SHIFTDMAREGi`.
    pub const fn shiftdmare_gi(
        mode: u32,
        result_reg: u32,
        right_imm5: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SHIFTDMAREGi;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mode) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mode,
                width: f.width(),
            });
        }
        word |= f.place(mode);
        let f = def.fields()[1];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[2];
        if !f.fits(right_imm5) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_imm5,
                width: f.width(),
            });
        }
        word |= f.place(right_imm5);
        let f = def.fields()[3];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `CMPDMAREG`.
    pub const fn cmpdmareg(
        mode: u32,
        result_reg: u32,
        right_reg: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::CMPDMAREG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mode) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mode,
                width: f.width(),
            });
        }
        word |= f.place(mode);
        let f = def.fields()[1];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[2];
        if !f.fits(right_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_reg,
                width: f.width(),
            });
        }
        word |= f.place(right_reg);
        let f = def.fields()[3];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `CMPDMAREGi`.
    pub const fn cmpdmare_gi(
        mode: u32,
        result_reg: u32,
        right_imm6: u32,
        left_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::CMPDMAREGi;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mode) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mode,
                width: f.width(),
            });
        }
        word |= f.place(mode);
        let f = def.fields()[1];
        if !f.fits(result_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: result_reg,
                width: f.width(),
            });
        }
        word |= f.place(result_reg);
        let f = def.fields()[2];
        if !f.fits(right_imm6) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: right_imm6,
                width: f.width(),
            });
        }
        word |= f.place(right_imm6);
        let f = def.fields()[3];
        if !f.fits(left_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: left_reg,
                width: f.width(),
            });
        }
        word |= f.place(left_reg);
        Ok(Instruction::new(word, def))
    }

    /// `ATGETM`.
    pub const fn atgetm(index: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::ATGETM;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(index) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: index,
                width: f.width(),
            });
        }
        word |= f.place(index);
        Ok(Instruction::new(word, def))
    }

    /// `ATRELM`.
    pub const fn atrelm(index: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::ATRELM;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(index) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: index,
                width: f.width(),
            });
        }
        word |= f.place(index);
        Ok(Instruction::new(word, def))
    }

    /// `ATINCGET`.
    pub const fn atincget(
        int_width: u32,
        ofs: u32,
        in_out_reg: u32,
        addr_reg: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::ATINCGET;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(int_width) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: int_width,
                width: f.width(),
            });
        }
        word |= f.place(int_width);
        let f = def.fields()[1];
        if !f.fits(ofs) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: ofs,
                width: f.width(),
            });
        }
        word |= f.place(ofs);
        let f = def.fields()[2];
        if !f.fits(in_out_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: in_out_reg,
                width: f.width(),
            });
        }
        word |= f.place(in_out_reg);
        let f = def.fields()[3];
        if !f.fits(addr_reg) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_reg,
                width: f.width(),
            });
        }
        word |= f.place(addr_reg);
        Ok(Instruction::new(word, def))
    }

    /// `ATINCGETPTR`, built field by field.
    ///
    /// 6 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Atincgetptr::ZERO.no_incr(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Atincgetptr {
        no_incr: u32,
        incr_log2: u32,
        int_width: u32,
        ofs: u32,
        result_reg: u32,
        addr_reg: u32,
    }

    impl Atincgetptr {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Atincgetptr {
            no_incr: 0,
            incr_log2: 0,
            int_width: 0,
            ofs: 0,
            result_reg: 0,
            addr_reg: 0,
        };

        pub const fn no_incr(mut self, value: u32) -> Self {
            self.no_incr = value;
            self
        }

        pub const fn incr_log2(mut self, value: u32) -> Self {
            self.incr_log2 = value;
            self
        }

        pub const fn int_width(mut self, value: u32) -> Self {
            self.int_width = value;
            self
        }

        pub const fn ofs(mut self, value: u32) -> Self {
            self.ofs = value;
            self
        }

        pub const fn result_reg(mut self, value: u32) -> Self {
            self.result_reg = value;
            self
        }

        pub const fn addr_reg(mut self, value: u32) -> Self {
            self.addr_reg = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::ATINCGETPTR;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.no_incr) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.no_incr,
                    width: f.width(),
                });
            }
            word |= f.place(self.no_incr);
            let f = def.fields()[1];
            if !f.fits(self.incr_log2) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.incr_log2,
                    width: f.width(),
                });
            }
            word |= f.place(self.incr_log2);
            let f = def.fields()[2];
            if !f.fits(self.int_width) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.int_width,
                    width: f.width(),
                });
            }
            word |= f.place(self.int_width);
            let f = def.fields()[3];
            if !f.fits(self.ofs) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.ofs,
                    width: f.width(),
                });
            }
            word |= f.place(self.ofs);
            let f = def.fields()[4];
            if !f.fits(self.result_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.result_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.result_reg);
            let f = def.fields()[5];
            if !f.fits(self.addr_reg) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_reg,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_reg);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SFPLOAD_BH`.
    pub const fn sfpload(
        vd: u32,
        mod0: u32,
        addr_mod: u32,
        imm10: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPLOAD;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[1];
        if !f.fits(mod0) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod0,
                width: f.width(),
            });
        }
        word |= f.place(mod0);
        let f = def.fields()[2];
        if !f.fits(addr_mod) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_mod,
                width: f.width(),
            });
        }
        word |= f.place(addr_mod);
        let f = def.fields()[3];
        if !f.fits(imm10) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm10,
                width: f.width(),
            });
        }
        word |= f.place(imm10);
        Ok(Instruction::new(word, def))
    }

    /// `SFPLOADMACRO_BH`, built field by field.
    ///
    /// 6 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Sfploadmacro::ZERO.macro_index(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Sfploadmacro {
        macro_index: u32,
        vd_lo: u32,
        mod0: u32,
        addr_mod: u32,
        imm9: u32,
        vd_hi: u32,
    }

    impl Sfploadmacro {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Sfploadmacro {
            macro_index: 0,
            vd_lo: 0,
            mod0: 0,
            addr_mod: 0,
            imm9: 0,
            vd_hi: 0,
        };

        pub const fn macro_index(mut self, value: u32) -> Self {
            self.macro_index = value;
            self
        }

        pub const fn vd_lo(mut self, value: u32) -> Self {
            self.vd_lo = value;
            self
        }

        pub const fn mod0(mut self, value: u32) -> Self {
            self.mod0 = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn imm9(mut self, value: u32) -> Self {
            self.imm9 = value;
            self
        }

        pub const fn vd_hi(mut self, value: u32) -> Self {
            self.vd_hi = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SFPLOADMACRO;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.macro_index) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.macro_index,
                    width: f.width(),
                });
            }
            word |= f.place(self.macro_index);
            let f = def.fields()[1];
            if !f.fits(self.vd_lo) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vd_lo,
                    width: f.width(),
                });
            }
            word |= f.place(self.vd_lo);
            let f = def.fields()[2];
            if !f.fits(self.mod0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.mod0,
                    width: f.width(),
                });
            }
            word |= f.place(self.mod0);
            let f = def.fields()[3];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[4];
            if !f.fits(self.imm9) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.imm9,
                    width: f.width(),
                });
            }
            word |= f.place(self.imm9);
            let f = def.fields()[5];
            if !f.fits(self.vd_hi) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vd_hi,
                    width: f.width(),
                });
            }
            word |= f.place(self.vd_hi);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SFPSTORE_BH`.
    pub const fn sfpstore(
        vd: u32,
        mod0: u32,
        addr_mod: u32,
        imm10: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPSTORE;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[1];
        if !f.fits(mod0) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod0,
                width: f.width(),
            });
        }
        word |= f.place(mod0);
        let f = def.fields()[2];
        if !f.fits(addr_mod) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_mod,
                width: f.width(),
            });
        }
        word |= f.place(addr_mod);
        let f = def.fields()[3];
        if !f.fits(imm10) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm10,
                width: f.width(),
            });
        }
        word |= f.place(imm10);
        Ok(Instruction::new(word, def))
    }

    /// `MOP`.
    pub const fn mop(template: u32, count1: u32, mask_lo: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::MOP;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(template) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: template,
                width: f.width(),
            });
        }
        word |= f.place(template);
        let f = def.fields()[1];
        if !f.fits(count1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: count1,
                width: f.width(),
            });
        }
        word |= f.place(count1);
        let f = def.fields()[2];
        if !f.fits(mask_lo) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mask_lo,
                width: f.width(),
            });
        }
        word |= f.place(mask_lo);
        Ok(Instruction::new(word, def))
    }

    /// `MOP_CFG`.
    pub const fn mop_cfg(mask_hi: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::MOP_CFG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mask_hi) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mask_hi,
                width: f.width(),
            });
        }
        word |= f.place(mask_hi);
        Ok(Instruction::new(word, def))
    }

    /// `SFPLOADI`.
    pub const fn sfploadi(vd: u32, mod0: u32, imm16: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPLOADI;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[1];
        if !f.fits(mod0) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod0,
                width: f.width(),
            });
        }
        word |= f.place(mod0);
        let f = def.fields()[2];
        if !f.fits(imm16) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm16,
                width: f.width(),
            });
        }
        word |= f.place(imm16);
        Ok(Instruction::new(word, def))
    }

    /// `SFPIADD`.
    pub const fn sfpiadd(
        imm12: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPIADD;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm12) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm12,
                width: f.width(),
            });
        }
        word |= f.place(imm12);
        let f = def.fields()[1];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[2];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[3];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPSWAP`.
    pub const fn sfpswap(vc: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPSWAP;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPCONFIG`.
    pub const fn sfpconfig(imm16: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPCONFIG;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm16) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm16,
                width: f.width(),
            });
        }
        word |= f.place(imm16);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPMAD`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Sfpmad::ZERO.va(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Sfpmad {
        va: u32,
        vb: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    }

    impl Sfpmad {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Sfpmad {
            va: 0,
            vb: 0,
            vc: 0,
            vd: 0,
            mod1: 0,
        };

        pub const fn va(mut self, value: u32) -> Self {
            self.va = value;
            self
        }

        pub const fn vb(mut self, value: u32) -> Self {
            self.vb = value;
            self
        }

        pub const fn vc(mut self, value: u32) -> Self {
            self.vc = value;
            self
        }

        pub const fn vd(mut self, value: u32) -> Self {
            self.vd = value;
            self
        }

        pub const fn mod1(mut self, value: u32) -> Self {
            self.mod1 = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SFPMAD;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.va) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.va,
                    width: f.width(),
                });
            }
            word |= f.place(self.va);
            let f = def.fields()[1];
            if !f.fits(self.vb) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vb,
                    width: f.width(),
                });
            }
            word |= f.place(self.vb);
            let f = def.fields()[2];
            if !f.fits(self.vc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vc,
                    width: f.width(),
                });
            }
            word |= f.place(self.vc);
            let f = def.fields()[3];
            if !f.fits(self.vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vd,
                    width: f.width(),
                });
            }
            word |= f.place(self.vd);
            let f = def.fields()[4];
            if !f.fits(self.mod1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.mod1,
                    width: f.width(),
                });
            }
            word |= f.place(self.mod1);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SFPADD`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Sfpadd::ZERO.va(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Sfpadd {
        va: u32,
        vb: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    }

    impl Sfpadd {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Sfpadd {
            va: 0,
            vb: 0,
            vc: 0,
            vd: 0,
            mod1: 0,
        };

        pub const fn va(mut self, value: u32) -> Self {
            self.va = value;
            self
        }

        pub const fn vb(mut self, value: u32) -> Self {
            self.vb = value;
            self
        }

        pub const fn vc(mut self, value: u32) -> Self {
            self.vc = value;
            self
        }

        pub const fn vd(mut self, value: u32) -> Self {
            self.vd = value;
            self
        }

        pub const fn mod1(mut self, value: u32) -> Self {
            self.mod1 = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SFPADD;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.va) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.va,
                    width: f.width(),
                });
            }
            word |= f.place(self.va);
            let f = def.fields()[1];
            if !f.fits(self.vb) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vb,
                    width: f.width(),
                });
            }
            word |= f.place(self.vb);
            let f = def.fields()[2];
            if !f.fits(self.vc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vc,
                    width: f.width(),
                });
            }
            word |= f.place(self.vc);
            let f = def.fields()[3];
            if !f.fits(self.vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vd,
                    width: f.width(),
                });
            }
            word |= f.place(self.vd);
            let f = def.fields()[4];
            if !f.fits(self.mod1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.mod1,
                    width: f.width(),
                });
            }
            word |= f.place(self.mod1);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SFPMUL`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Sfpmul::ZERO.va(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Sfpmul {
        va: u32,
        vb: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    }

    impl Sfpmul {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Sfpmul {
            va: 0,
            vb: 0,
            vc: 0,
            vd: 0,
            mod1: 0,
        };

        pub const fn va(mut self, value: u32) -> Self {
            self.va = value;
            self
        }

        pub const fn vb(mut self, value: u32) -> Self {
            self.vb = value;
            self
        }

        pub const fn vc(mut self, value: u32) -> Self {
            self.vc = value;
            self
        }

        pub const fn vd(mut self, value: u32) -> Self {
            self.vd = value;
            self
        }

        pub const fn mod1(mut self, value: u32) -> Self {
            self.mod1 = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SFPMUL;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.va) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.va,
                    width: f.width(),
                });
            }
            word |= f.place(self.va);
            let f = def.fields()[1];
            if !f.fits(self.vb) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vb,
                    width: f.width(),
                });
            }
            word |= f.place(self.vb);
            let f = def.fields()[2];
            if !f.fits(self.vc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vc,
                    width: f.width(),
                });
            }
            word |= f.place(self.vc);
            let f = def.fields()[3];
            if !f.fits(self.vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vd,
                    width: f.width(),
                });
            }
            word |= f.place(self.vd);
            let f = def.fields()[4];
            if !f.fits(self.mod1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.mod1,
                    width: f.width(),
                });
            }
            word |= f.place(self.mod1);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SFPLUT`.
    pub const fn sfplut(vd: u32, mod0: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPLUT;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[1];
        if !f.fits(mod0) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod0,
                width: f.width(),
            });
        }
        word |= f.place(mod0);
        Ok(Instruction::new(word, def))
    }

    /// `SFPMULI`.
    pub const fn sfpmuli(imm16: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPMULI;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm16) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm16,
                width: f.width(),
            });
        }
        word |= f.place(imm16);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPADDI`.
    pub const fn sfpaddi(imm16: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPADDI;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm16) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm16,
                width: f.width(),
            });
        }
        word |= f.place(imm16);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPDIVP2`.
    pub const fn sfpdivp2(
        imm8: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPDIVP2;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm8) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm8,
                width: f.width(),
            });
        }
        word |= f.place(imm8);
        let f = def.fields()[1];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[2];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[3];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPEXEXP`.
    pub const fn sfpexexp(vc: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPEXEXP;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPEXMAN`.
    pub const fn sfpexman(vc: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPEXMAN;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPSHFT`.
    pub const fn sfpshft(
        imm12: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPSHFT;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm12) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm12,
                width: f.width(),
            });
        }
        word |= f.place(imm12);
        let f = def.fields()[1];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[2];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[3];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPSETCC`.
    pub const fn sfpsetcc(
        imm1: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPSETCC;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm1,
                width: f.width(),
            });
        }
        word |= f.place(imm1);
        let f = def.fields()[1];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[2];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[3];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPMOV`.
    pub const fn sfpmov(vc: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPMOV;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPABS`.
    pub const fn sfpabs(vc: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPABS;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPAND_BH`.
    pub const fn sfpand(vb: u32, vc: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPAND;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vb) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vb,
                width: f.width(),
            });
        }
        word |= f.place(vb);
        let f = def.fields()[1];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[2];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[3];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPOR_BH`.
    pub const fn sfpor(vb: u32, vc: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPOR;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vb) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vb,
                width: f.width(),
            });
        }
        word |= f.place(vb);
        let f = def.fields()[1];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[2];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[3];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPNOT`.
    pub const fn sfpnot(vc: u32, vd: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPNOT;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        Ok(Instruction::new(word, def))
    }

    /// `SFPLZ`.
    pub const fn sfplz(vc: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPLZ;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPSETEXP`.
    pub const fn sfpsetexp(
        imm8: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPSETEXP;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm8) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm8,
                width: f.width(),
            });
        }
        word |= f.place(imm8);
        let f = def.fields()[1];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[2];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[3];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPSETMAN`.
    pub const fn sfpsetman(
        imm12: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPSETMAN;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm12) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm12,
                width: f.width(),
            });
        }
        word |= f.place(imm12);
        let f = def.fields()[1];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[2];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[3];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPPUSHC_BH`.
    pub const fn sfppushc(vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPPUSHC;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[1];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPPOPC`.
    pub const fn sfppopc(vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPPOPC;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[1];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPSETSGN`.
    pub const fn sfpsetsgn(
        imm1: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPSETSGN;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm1,
                width: f.width(),
            });
        }
        word |= f.place(imm1);
        let f = def.fields()[1];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[2];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[3];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPENCC`.
    pub const fn sfpencc(imm2: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPENCC;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(imm2) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm2,
                width: f.width(),
            });
        }
        word |= f.place(imm2);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPCOMPC`.
    pub const fn sfpcompc(vd: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPCOMPC;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        Ok(Instruction::new(word, def))
    }

    /// `SFPTRANSP`.
    pub const fn sfptransp(vd: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPTRANSP;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        Ok(Instruction::new(word, def))
    }

    /// `SFPXOR`.
    pub const fn sfpxor(vc: u32, vd: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPXOR;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        Ok(Instruction::new(word, def))
    }

    /// `SFPSTOCHRND_BH`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Sfpstochrnd::ZERO.rounding_mode(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Sfpstochrnd {
        rounding_mode: u32,
        vb: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    }

    impl Sfpstochrnd {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Sfpstochrnd {
            rounding_mode: 0,
            vb: 0,
            vc: 0,
            vd: 0,
            mod1: 0,
        };

        pub const fn rounding_mode(mut self, value: u32) -> Self {
            self.rounding_mode = value;
            self
        }

        pub const fn vb(mut self, value: u32) -> Self {
            self.vb = value;
            self
        }

        pub const fn vc(mut self, value: u32) -> Self {
            self.vc = value;
            self
        }

        pub const fn vd(mut self, value: u32) -> Self {
            self.vd = value;
            self
        }

        pub const fn mod1(mut self, value: u32) -> Self {
            self.mod1 = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SFPSTOCHRND;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.rounding_mode) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.rounding_mode,
                    width: f.width(),
                });
            }
            word |= f.place(self.rounding_mode);
            let f = def.fields()[1];
            if !f.fits(self.vb) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vb,
                    width: f.width(),
                });
            }
            word |= f.place(self.vb);
            let f = def.fields()[2];
            if !f.fits(self.vc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vc,
                    width: f.width(),
                });
            }
            word |= f.place(self.vc);
            let f = def.fields()[3];
            if !f.fits(self.vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vd,
                    width: f.width(),
                });
            }
            word |= f.place(self.vd);
            let f = def.fields()[4];
            if !f.fits(self.mod1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.mod1,
                    width: f.width(),
                });
            }
            word |= f.place(self.mod1);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SFPSTOCHRNDi_BH`, built field by field.
    ///
    /// 7 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `SfpstochrnDi::ZERO.rounding_mode(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct SfpstochrnDi {
        rounding_mode: u32,
        imm5: u32,
        vb: u32,
        vc: u32,
        vd: u32,
        use_imm5: u32,
        mod1: u32,
    }

    impl SfpstochrnDi {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = SfpstochrnDi {
            rounding_mode: 0,
            imm5: 0,
            vb: 0,
            vc: 0,
            vd: 0,
            use_imm5: 0,
            mod1: 0,
        };

        pub const fn rounding_mode(mut self, value: u32) -> Self {
            self.rounding_mode = value;
            self
        }

        pub const fn imm5(mut self, value: u32) -> Self {
            self.imm5 = value;
            self
        }

        pub const fn vb(mut self, value: u32) -> Self {
            self.vb = value;
            self
        }

        pub const fn vc(mut self, value: u32) -> Self {
            self.vc = value;
            self
        }

        pub const fn vd(mut self, value: u32) -> Self {
            self.vd = value;
            self
        }

        pub const fn use_imm5(mut self, value: u32) -> Self {
            self.use_imm5 = value;
            self
        }

        pub const fn mod1(mut self, value: u32) -> Self {
            self.mod1 = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SFPSTOCHRNDi;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.rounding_mode) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.rounding_mode,
                    width: f.width(),
                });
            }
            word |= f.place(self.rounding_mode);
            let f = def.fields()[1];
            if !f.fits(self.imm5) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.imm5,
                    width: f.width(),
                });
            }
            word |= f.place(self.imm5);
            let f = def.fields()[2];
            if !f.fits(self.vb) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vb,
                    width: f.width(),
                });
            }
            word |= f.place(self.vb);
            let f = def.fields()[3];
            if !f.fits(self.vc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vc,
                    width: f.width(),
                });
            }
            word |= f.place(self.vc);
            let f = def.fields()[4];
            if !f.fits(self.vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vd,
                    width: f.width(),
                });
            }
            word |= f.place(self.vd);
            let f = def.fields()[5];
            if !f.fits(self.use_imm5) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.use_imm5,
                    width: f.width(),
                });
            }
            word |= f.place(self.use_imm5);
            let f = def.fields()[6];
            if !f.fits(self.mod1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.mod1,
                    width: f.width(),
                });
            }
            word |= f.place(self.mod1);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SFPNOP`.
    pub const fn sfpnop() -> Result<Instruction, EncodeError> {
        let def = &defs::SFPNOP;
        let word = def.skeleton();
        Ok(Instruction::new(word, def))
    }

    /// `SFPCAST`.
    pub const fn sfpcast(vc: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPCAST;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPLE`.
    pub const fn sfple(vc: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPLE;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPGT`.
    pub const fn sfpgt(vc: u32, vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPGT;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[1];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[2];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `SFPMUL24`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Sfpmul24::ZERO.va(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Sfpmul24 {
        va: u32,
        vb: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    }

    impl Sfpmul24 {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Sfpmul24 {
            va: 0,
            vb: 0,
            vc: 0,
            vd: 0,
            mod1: 0,
        };

        pub const fn va(mut self, value: u32) -> Self {
            self.va = value;
            self
        }

        pub const fn vb(mut self, value: u32) -> Self {
            self.vb = value;
            self
        }

        pub const fn vc(mut self, value: u32) -> Self {
            self.vc = value;
            self
        }

        pub const fn vd(mut self, value: u32) -> Self {
            self.vd = value;
            self
        }

        pub const fn mod1(mut self, value: u32) -> Self {
            self.mod1 = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::SFPMUL24;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.va) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.va,
                    width: f.width(),
                });
            }
            word |= f.place(self.va);
            let f = def.fields()[1];
            if !f.fits(self.vb) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vb,
                    width: f.width(),
                });
            }
            word |= f.place(self.vb);
            let f = def.fields()[2];
            if !f.fits(self.vc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vc,
                    width: f.width(),
                });
            }
            word |= f.place(self.vc);
            let f = def.fields()[3];
            if !f.fits(self.vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.vd,
                    width: f.width(),
                });
            }
            word |= f.place(self.vd);
            let f = def.fields()[4];
            if !f.fits(self.mod1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.mod1,
                    width: f.width(),
                });
            }
            word |= f.place(self.mod1);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SFPARECIP`.
    pub const fn sfparecip(
        vb: u32,
        vc: u32,
        vd: u32,
        mod1: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SFPARECIP;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(vb) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vb,
                width: f.width(),
            });
        }
        word |= f.place(vb);
        let f = def.fields()[1];
        if !f.fits(vc) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vc,
                width: f.width(),
            });
        }
        word |= f.place(vc);
        let f = def.fields()[2];
        if !f.fits(vd) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: vd,
                width: f.width(),
            });
        }
        word |= f.place(vd);
        let f = def.fields()[3];
        if !f.fits(mod1) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mod1,
                width: f.width(),
            });
        }
        word |= f.place(mod1);
        Ok(Instruction::new(word, def))
    }

    /// `UNPACR_Regular`, built field by field.
    ///
    /// 12 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `UnpacrRegular::ZERO.which_unpacker(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct UnpacrRegular {
        which_unpacker: u32,
        ch1_y_inc: u32,
        ch1_z_inc: u32,
        ch0_y_inc: u32,
        ch0_z_inc: u32,
        context_number: u32,
        context_adc: u32,
        multi_context_mode: u32,
        flip_src: u32,
        all_datums_are_zero: u32,
        use_context_counter: u32,
        row_search: u32,
    }

    impl UnpacrRegular {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = UnpacrRegular {
            which_unpacker: 0,
            ch1_y_inc: 0,
            ch1_z_inc: 0,
            ch0_y_inc: 0,
            ch0_z_inc: 0,
            context_number: 0,
            context_adc: 0,
            multi_context_mode: 0,
            flip_src: 0,
            all_datums_are_zero: 0,
            use_context_counter: 0,
            row_search: 0,
        };

        pub const fn which_unpacker(mut self, value: u32) -> Self {
            self.which_unpacker = value;
            self
        }

        pub const fn ch1_y_inc(mut self, value: u32) -> Self {
            self.ch1_y_inc = value;
            self
        }

        pub const fn ch1_z_inc(mut self, value: u32) -> Self {
            self.ch1_z_inc = value;
            self
        }

        pub const fn ch0_y_inc(mut self, value: u32) -> Self {
            self.ch0_y_inc = value;
            self
        }

        pub const fn ch0_z_inc(mut self, value: u32) -> Self {
            self.ch0_z_inc = value;
            self
        }

        pub const fn context_number(mut self, value: u32) -> Self {
            self.context_number = value;
            self
        }

        pub const fn context_adc(mut self, value: u32) -> Self {
            self.context_adc = value;
            self
        }

        pub const fn multi_context_mode(mut self, value: u32) -> Self {
            self.multi_context_mode = value;
            self
        }

        pub const fn flip_src(mut self, value: u32) -> Self {
            self.flip_src = value;
            self
        }

        pub const fn all_datums_are_zero(mut self, value: u32) -> Self {
            self.all_datums_are_zero = value;
            self
        }

        pub const fn use_context_counter(mut self, value: u32) -> Self {
            self.use_context_counter = value;
            self
        }

        pub const fn row_search(mut self, value: u32) -> Self {
            self.row_search = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::UNPACR_Regular;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.which_unpacker) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.which_unpacker,
                    width: f.width(),
                });
            }
            word |= f.place(self.which_unpacker);
            let f = def.fields()[1];
            if !f.fits(self.ch1_y_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.ch1_y_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.ch1_y_inc);
            let f = def.fields()[2];
            if !f.fits(self.ch1_z_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.ch1_z_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.ch1_z_inc);
            let f = def.fields()[3];
            if !f.fits(self.ch0_y_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.ch0_y_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.ch0_y_inc);
            let f = def.fields()[4];
            if !f.fits(self.ch0_z_inc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.ch0_z_inc,
                    width: f.width(),
                });
            }
            word |= f.place(self.ch0_z_inc);
            let f = def.fields()[5];
            if !f.fits(self.context_number) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.context_number,
                    width: f.width(),
                });
            }
            word |= f.place(self.context_number);
            let f = def.fields()[6];
            if !f.fits(self.context_adc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.context_adc,
                    width: f.width(),
                });
            }
            word |= f.place(self.context_adc);
            let f = def.fields()[7];
            if !f.fits(self.multi_context_mode) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.multi_context_mode,
                    width: f.width(),
                });
            }
            word |= f.place(self.multi_context_mode);
            let f = def.fields()[8];
            if !f.fits(self.flip_src) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src);
            let f = def.fields()[9];
            if !f.fits(self.all_datums_are_zero) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.all_datums_are_zero,
                    width: f.width(),
                });
            }
            word |= f.place(self.all_datums_are_zero);
            let f = def.fields()[10];
            if !f.fits(self.use_context_counter) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.use_context_counter,
                    width: f.width(),
                });
            }
            word |= f.place(self.use_context_counter);
            let f = def.fields()[11];
            if !f.fits(self.row_search) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.row_search,
                    width: f.width(),
                });
            }
            word |= f.place(self.row_search);
            Ok(Instruction::new(word, def))
        }
    }

    /// `UNPACR_IncrementContextCounter`.
    pub const fn unpacr_increment_context_counter(
        which_unpacker: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::UNPACR_IncrementContextCounter;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(which_unpacker) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: which_unpacker,
                width: f.width(),
            });
        }
        word |= f.place(which_unpacker);
        Ok(Instruction::new(word, def))
    }

    /// `UNPACR_FlushCache`.
    pub const fn unpacr_flush_cache(
        which_unpacker: u32,
        multi_context_mode: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::UNPACR_FlushCache;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(which_unpacker) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: which_unpacker,
                width: f.width(),
            });
        }
        word |= f.place(which_unpacker);
        let f = def.fields()[1];
        if !f.fits(multi_context_mode) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: multi_context_mode,
                width: f.width(),
            });
        }
        word |= f.place(multi_context_mode);
        Ok(Instruction::new(word, def))
    }

    /// `UNPACR_NOP_OverlayClear0`.
    pub const fn unpacr_nop_overlay_clear0(
        which_unpacker: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::UNPACR_NOP_OverlayClear0;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(which_unpacker) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: which_unpacker,
                width: f.width(),
            });
        }
        word |= f.place(which_unpacker);
        Ok(Instruction::new(word, def))
    }

    /// `UNPACR_NOP_OverlayClear3`.
    pub const fn unpacr_nop_overlay_clear3(
        which_unpacker: u32,
        which_stream: u32,
        clear_count: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::UNPACR_NOP_OverlayClear3;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(which_unpacker) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: which_unpacker,
                width: f.width(),
            });
        }
        word |= f.place(which_unpacker);
        let f = def.fields()[1];
        if !f.fits(which_stream) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: which_stream,
                width: f.width(),
            });
        }
        word |= f.place(which_stream);
        let f = def.fields()[2];
        if !f.fits(clear_count) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: clear_count,
                width: f.width(),
            });
        }
        word |= f.place(clear_count);
        Ok(Instruction::new(word, def))
    }

    /// `UNPACR_NOP_Nop`.
    pub const fn unpacr_nop_nop(which_unpacker: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::UNPACR_NOP_Nop;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(which_unpacker) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: which_unpacker,
                width: f.width(),
            });
        }
        word |= f.place(which_unpacker);
        Ok(Instruction::new(word, def))
    }

    /// `UNPACR_NOP_SETREG`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `UnpacrNopSetreg::ZERO.which_unpacker(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct UnpacrNopSetreg {
        which_unpacker: u32,
        addr_sel: u32,
        addr_mid: u32,
        value11: u32,
        accumulate: u32,
    }

    impl UnpacrNopSetreg {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = UnpacrNopSetreg {
            which_unpacker: 0,
            addr_sel: 0,
            addr_mid: 0,
            value11: 0,
            accumulate: 0,
        };

        pub const fn which_unpacker(mut self, value: u32) -> Self {
            self.which_unpacker = value;
            self
        }

        pub const fn addr_sel(mut self, value: u32) -> Self {
            self.addr_sel = value;
            self
        }

        pub const fn addr_mid(mut self, value: u32) -> Self {
            self.addr_mid = value;
            self
        }

        pub const fn value11(mut self, value: u32) -> Self {
            self.value11 = value;
            self
        }

        pub const fn accumulate(mut self, value: u32) -> Self {
            self.accumulate = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::UNPACR_NOP_SETREG;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.which_unpacker) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.which_unpacker,
                    width: f.width(),
                });
            }
            word |= f.place(self.which_unpacker);
            let f = def.fields()[1];
            if !f.fits(self.addr_sel) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_sel,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_sel);
            let f = def.fields()[2];
            if !f.fits(self.addr_mid) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mid,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mid);
            let f = def.fields()[3];
            if !f.fits(self.value11) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.value11,
                    width: f.width(),
                });
            }
            word |= f.place(self.value11);
            let f = def.fields()[4];
            if !f.fits(self.accumulate) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.accumulate,
                    width: f.width(),
                });
            }
            word |= f.place(self.accumulate);
            Ok(Instruction::new(word, def))
        }
    }

    /// `UNPACR_NOP_SETDVALID_BH`.
    pub const fn unpacr_nop_setdvalid(which_unpacker: u32) -> Result<Instruction, EncodeError> {
        let def = &defs::UNPACR_NOP_SETDVALID;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(which_unpacker) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: which_unpacker,
                width: f.width(),
            });
        }
        word |= f.place(which_unpacker);
        Ok(Instruction::new(word, def))
    }

    /// `UNPACR_NOP_ZEROSRC_BH`.
    pub const fn unpacr_nop_zerosrc(
        which_unpacker: u32,
        wait_like_unpacr: u32,
        both_banks: u32,
        negative_inf_src_a: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::UNPACR_NOP_ZEROSRC;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(which_unpacker) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: which_unpacker,
                width: f.width(),
            });
        }
        word |= f.place(which_unpacker);
        let f = def.fields()[1];
        if !f.fits(wait_like_unpacr) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: wait_like_unpacr,
                width: f.width(),
            });
        }
        word |= f.place(wait_like_unpacr);
        let f = def.fields()[2];
        if !f.fits(both_banks) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: both_banks,
                width: f.width(),
            });
        }
        word |= f.place(both_banks);
        let f = def.fields()[3];
        if !f.fits(negative_inf_src_a) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: negative_inf_src_a,
                width: f.width(),
            });
        }
        word |= f.place(negative_inf_src_a);
        Ok(Instruction::new(word, def))
    }

    /// `GMPOOL_BH`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Gmpool::ZERO.flip_src_b(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Gmpool {
        flip_src_b: u32,
        flip_src_a: u32,
        addr_mod: u32,
        arg_max: u32,
        dst_row: u32,
    }

    impl Gmpool {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Gmpool {
            flip_src_b: 0,
            flip_src_a: 0,
            addr_mod: 0,
            arg_max: 0,
            dst_row: 0,
        };

        pub const fn flip_src_b(mut self, value: u32) -> Self {
            self.flip_src_b = value;
            self
        }

        pub const fn flip_src_a(mut self, value: u32) -> Self {
            self.flip_src_a = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn arg_max(mut self, value: u32) -> Self {
            self.arg_max = value;
            self
        }

        pub const fn dst_row(mut self, value: u32) -> Self {
            self.dst_row = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::GMPOOL;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.flip_src_b) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_b,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_b);
            let f = def.fields()[1];
            if !f.fits(self.flip_src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_a,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_a);
            let f = def.fields()[2];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[3];
            if !f.fits(self.arg_max) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.arg_max,
                    width: f.width(),
                });
            }
            word |= f.place(self.arg_max);
            let f = def.fields()[4];
            if !f.fits(self.dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_row);
            Ok(Instruction::new(word, def))
        }
    }

    /// `GAPOOL_BH`.
    pub const fn gapool(
        flip_src_b: u32,
        flip_src_a: u32,
        addr_mod: u32,
        dst_row: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::GAPOOL;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(flip_src_b) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: flip_src_b,
                width: f.width(),
            });
        }
        word |= f.place(flip_src_b);
        let f = def.fields()[1];
        if !f.fits(flip_src_a) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: flip_src_a,
                width: f.width(),
            });
        }
        word |= f.place(flip_src_a);
        let f = def.fields()[2];
        if !f.fits(addr_mod) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_mod,
                width: f.width(),
            });
        }
        word |= f.place(addr_mod);
        let f = def.fields()[3];
        if !f.fits(dst_row) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: dst_row,
                width: f.width(),
            });
        }
        word |= f.place(dst_row);
        Ok(Instruction::new(word, def))
    }

    /// `MVMUL_BH`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Mvmul::ZERO.flip_src_b(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Mvmul {
        flip_src_b: u32,
        flip_src_a: u32,
        broadcast_src_b_row: u32,
        addr_mod: u32,
        dst_row: u32,
    }

    impl Mvmul {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Mvmul {
            flip_src_b: 0,
            flip_src_a: 0,
            broadcast_src_b_row: 0,
            addr_mod: 0,
            dst_row: 0,
        };

        pub const fn flip_src_b(mut self, value: u32) -> Self {
            self.flip_src_b = value;
            self
        }

        pub const fn flip_src_a(mut self, value: u32) -> Self {
            self.flip_src_a = value;
            self
        }

        pub const fn broadcast_src_b_row(mut self, value: u32) -> Self {
            self.broadcast_src_b_row = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn dst_row(mut self, value: u32) -> Self {
            self.dst_row = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::MVMUL;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.flip_src_b) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_b,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_b);
            let f = def.fields()[1];
            if !f.fits(self.flip_src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_a,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_a);
            let f = def.fields()[2];
            if !f.fits(self.broadcast_src_b_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.broadcast_src_b_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.broadcast_src_b_row);
            let f = def.fields()[3];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[4];
            if !f.fits(self.dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_row);
            Ok(Instruction::new(word, def))
        }
    }

    /// `MOVA2D_BH`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Mova2D::ZERO.use_dst32b_lo(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Mova2D {
        use_dst32b_lo: u32,
        src_row: u32,
        addr_mod: u32,
        move8_rows: u32,
        dst_row: u32,
    }

    impl Mova2D {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Mova2D {
            use_dst32b_lo: 0,
            src_row: 0,
            addr_mod: 0,
            move8_rows: 0,
            dst_row: 0,
        };

        pub const fn use_dst32b_lo(mut self, value: u32) -> Self {
            self.use_dst32b_lo = value;
            self
        }

        pub const fn src_row(mut self, value: u32) -> Self {
            self.src_row = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn move8_rows(mut self, value: u32) -> Self {
            self.move8_rows = value;
            self
        }

        pub const fn dst_row(mut self, value: u32) -> Self {
            self.dst_row = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::MOVA2D;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.use_dst32b_lo) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.use_dst32b_lo,
                    width: f.width(),
                });
            }
            word |= f.place(self.use_dst32b_lo);
            let f = def.fields()[1];
            if !f.fits(self.src_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_row);
            let f = def.fields()[2];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[3];
            if !f.fits(self.move8_rows) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.move8_rows,
                    width: f.width(),
                });
            }
            word |= f.place(self.move8_rows);
            let f = def.fields()[4];
            if !f.fits(self.dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_row);
            Ok(Instruction::new(word, def))
        }
    }

    /// `MOVB2D_BH`, built field by field.
    ///
    /// 7 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Movb2D::ZERO.use_dst32b_lo(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Movb2D {
        use_dst32b_lo: u32,
        src_row: u32,
        addr_mod: u32,
        move4_rows: u32,
        broadcast1_row_to8: u32,
        broadcast_col0: u32,
        dst_row: u32,
    }

    impl Movb2D {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Movb2D {
            use_dst32b_lo: 0,
            src_row: 0,
            addr_mod: 0,
            move4_rows: 0,
            broadcast1_row_to8: 0,
            broadcast_col0: 0,
            dst_row: 0,
        };

        pub const fn use_dst32b_lo(mut self, value: u32) -> Self {
            self.use_dst32b_lo = value;
            self
        }

        pub const fn src_row(mut self, value: u32) -> Self {
            self.src_row = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn move4_rows(mut self, value: u32) -> Self {
            self.move4_rows = value;
            self
        }

        pub const fn broadcast1_row_to8(mut self, value: u32) -> Self {
            self.broadcast1_row_to8 = value;
            self
        }

        pub const fn broadcast_col0(mut self, value: u32) -> Self {
            self.broadcast_col0 = value;
            self
        }

        pub const fn dst_row(mut self, value: u32) -> Self {
            self.dst_row = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::MOVB2D;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.use_dst32b_lo) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.use_dst32b_lo,
                    width: f.width(),
                });
            }
            word |= f.place(self.use_dst32b_lo);
            let f = def.fields()[1];
            if !f.fits(self.src_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_row);
            let f = def.fields()[2];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[3];
            if !f.fits(self.move4_rows) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.move4_rows,
                    width: f.width(),
                });
            }
            word |= f.place(self.move4_rows);
            let f = def.fields()[4];
            if !f.fits(self.broadcast1_row_to8) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.broadcast1_row_to8,
                    width: f.width(),
                });
            }
            word |= f.place(self.broadcast1_row_to8);
            let f = def.fields()[5];
            if !f.fits(self.broadcast_col0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.broadcast_col0,
                    width: f.width(),
                });
            }
            word |= f.place(self.broadcast_col0);
            let f = def.fields()[6];
            if !f.fits(self.dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_row);
            Ok(Instruction::new(word, def))
        }
    }

    /// `MOVD2A_BH`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Movd2A::ZERO.use_dst32b_lo(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Movd2A {
        use_dst32b_lo: u32,
        src_row: u32,
        addr_mod: u32,
        move4_rows: u32,
        dst_row: u32,
    }

    impl Movd2A {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Movd2A {
            use_dst32b_lo: 0,
            src_row: 0,
            addr_mod: 0,
            move4_rows: 0,
            dst_row: 0,
        };

        pub const fn use_dst32b_lo(mut self, value: u32) -> Self {
            self.use_dst32b_lo = value;
            self
        }

        pub const fn src_row(mut self, value: u32) -> Self {
            self.src_row = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn move4_rows(mut self, value: u32) -> Self {
            self.move4_rows = value;
            self
        }

        pub const fn dst_row(mut self, value: u32) -> Self {
            self.dst_row = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::MOVD2A;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.use_dst32b_lo) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.use_dst32b_lo,
                    width: f.width(),
                });
            }
            word |= f.place(self.use_dst32b_lo);
            let f = def.fields()[1];
            if !f.fits(self.src_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_row);
            let f = def.fields()[2];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[3];
            if !f.fits(self.move4_rows) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.move4_rows,
                    width: f.width(),
                });
            }
            word |= f.place(self.move4_rows);
            let f = def.fields()[4];
            if !f.fits(self.dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_row);
            Ok(Instruction::new(word, def))
        }
    }

    /// `MOVD2B_BH`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Movd2B::ZERO.use_dst32b_lo(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Movd2B {
        use_dst32b_lo: u32,
        src_row: u32,
        addr_mod: u32,
        move4_rows: u32,
        dst_row: u32,
    }

    impl Movd2B {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Movd2B {
            use_dst32b_lo: 0,
            src_row: 0,
            addr_mod: 0,
            move4_rows: 0,
            dst_row: 0,
        };

        pub const fn use_dst32b_lo(mut self, value: u32) -> Self {
            self.use_dst32b_lo = value;
            self
        }

        pub const fn src_row(mut self, value: u32) -> Self {
            self.src_row = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn move4_rows(mut self, value: u32) -> Self {
            self.move4_rows = value;
            self
        }

        pub const fn dst_row(mut self, value: u32) -> Self {
            self.dst_row = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::MOVD2B;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.use_dst32b_lo) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.use_dst32b_lo,
                    width: f.width(),
                });
            }
            word |= f.place(self.use_dst32b_lo);
            let f = def.fields()[1];
            if !f.fits(self.src_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_row);
            let f = def.fields()[2];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[3];
            if !f.fits(self.move4_rows) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.move4_rows,
                    width: f.width(),
                });
            }
            word |= f.place(self.move4_rows);
            let f = def.fields()[4];
            if !f.fits(self.dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_row);
            Ok(Instruction::new(word, def))
        }
    }

    /// `MOVB2A_BH`.
    pub const fn movb2_a(
        src_a_row: u32,
        addr_mod: u32,
        move4_rows: u32,
        src_b_row: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::MOVB2A;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(src_a_row) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: src_a_row,
                width: f.width(),
            });
        }
        word |= f.place(src_a_row);
        let f = def.fields()[1];
        if !f.fits(addr_mod) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_mod,
                width: f.width(),
            });
        }
        word |= f.place(addr_mod);
        let f = def.fields()[2];
        if !f.fits(move4_rows) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: move4_rows,
                width: f.width(),
            });
        }
        word |= f.place(move4_rows);
        let f = def.fields()[3];
        if !f.fits(src_b_row) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: src_b_row,
                width: f.width(),
            });
        }
        word |= f.place(src_b_row);
        Ok(Instruction::new(word, def))
    }

    /// `ELWADD_BH`, built field by field.
    ///
    /// 7 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Elwadd::ZERO.flip_src_b(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Elwadd {
        flip_src_b: u32,
        flip_src_a: u32,
        add_dst: u32,
        broadcast_src_b_row: u32,
        broadcast_src_b_col0: u32,
        addr_mod: u32,
        dst_row: u32,
    }

    impl Elwadd {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Elwadd {
            flip_src_b: 0,
            flip_src_a: 0,
            add_dst: 0,
            broadcast_src_b_row: 0,
            broadcast_src_b_col0: 0,
            addr_mod: 0,
            dst_row: 0,
        };

        pub const fn flip_src_b(mut self, value: u32) -> Self {
            self.flip_src_b = value;
            self
        }

        pub const fn flip_src_a(mut self, value: u32) -> Self {
            self.flip_src_a = value;
            self
        }

        pub const fn add_dst(mut self, value: u32) -> Self {
            self.add_dst = value;
            self
        }

        pub const fn broadcast_src_b_row(mut self, value: u32) -> Self {
            self.broadcast_src_b_row = value;
            self
        }

        pub const fn broadcast_src_b_col0(mut self, value: u32) -> Self {
            self.broadcast_src_b_col0 = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn dst_row(mut self, value: u32) -> Self {
            self.dst_row = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::ELWADD;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.flip_src_b) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_b,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_b);
            let f = def.fields()[1];
            if !f.fits(self.flip_src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_a,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_a);
            let f = def.fields()[2];
            if !f.fits(self.add_dst) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.add_dst,
                    width: f.width(),
                });
            }
            word |= f.place(self.add_dst);
            let f = def.fields()[3];
            if !f.fits(self.broadcast_src_b_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.broadcast_src_b_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.broadcast_src_b_row);
            let f = def.fields()[4];
            if !f.fits(self.broadcast_src_b_col0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.broadcast_src_b_col0,
                    width: f.width(),
                });
            }
            word |= f.place(self.broadcast_src_b_col0);
            let f = def.fields()[5];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[6];
            if !f.fits(self.dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_row);
            Ok(Instruction::new(word, def))
        }
    }

    /// `ELWSUB_BH`, built field by field.
    ///
    /// 7 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Elwsub::ZERO.flip_src_b(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Elwsub {
        flip_src_b: u32,
        flip_src_a: u32,
        add_dst: u32,
        broadcast_src_b_row: u32,
        broadcast_src_b_col0: u32,
        addr_mod: u32,
        dst_row: u32,
    }

    impl Elwsub {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Elwsub {
            flip_src_b: 0,
            flip_src_a: 0,
            add_dst: 0,
            broadcast_src_b_row: 0,
            broadcast_src_b_col0: 0,
            addr_mod: 0,
            dst_row: 0,
        };

        pub const fn flip_src_b(mut self, value: u32) -> Self {
            self.flip_src_b = value;
            self
        }

        pub const fn flip_src_a(mut self, value: u32) -> Self {
            self.flip_src_a = value;
            self
        }

        pub const fn add_dst(mut self, value: u32) -> Self {
            self.add_dst = value;
            self
        }

        pub const fn broadcast_src_b_row(mut self, value: u32) -> Self {
            self.broadcast_src_b_row = value;
            self
        }

        pub const fn broadcast_src_b_col0(mut self, value: u32) -> Self {
            self.broadcast_src_b_col0 = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn dst_row(mut self, value: u32) -> Self {
            self.dst_row = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::ELWSUB;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.flip_src_b) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_b,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_b);
            let f = def.fields()[1];
            if !f.fits(self.flip_src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_a,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_a);
            let f = def.fields()[2];
            if !f.fits(self.add_dst) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.add_dst,
                    width: f.width(),
                });
            }
            word |= f.place(self.add_dst);
            let f = def.fields()[3];
            if !f.fits(self.broadcast_src_b_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.broadcast_src_b_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.broadcast_src_b_row);
            let f = def.fields()[4];
            if !f.fits(self.broadcast_src_b_col0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.broadcast_src_b_col0,
                    width: f.width(),
                });
            }
            word |= f.place(self.broadcast_src_b_col0);
            let f = def.fields()[5];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[6];
            if !f.fits(self.dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_row);
            Ok(Instruction::new(word, def))
        }
    }

    /// `ELWMUL_BH`, built field by field.
    ///
    /// 6 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Elwmul::ZERO.flip_src_b(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Elwmul {
        flip_src_b: u32,
        flip_src_a: u32,
        broadcast_src_b_row: u32,
        broadcast_src_b_col0: u32,
        addr_mod: u32,
        dst_row: u32,
    }

    impl Elwmul {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Elwmul {
            flip_src_b: 0,
            flip_src_a: 0,
            broadcast_src_b_row: 0,
            broadcast_src_b_col0: 0,
            addr_mod: 0,
            dst_row: 0,
        };

        pub const fn flip_src_b(mut self, value: u32) -> Self {
            self.flip_src_b = value;
            self
        }

        pub const fn flip_src_a(mut self, value: u32) -> Self {
            self.flip_src_a = value;
            self
        }

        pub const fn broadcast_src_b_row(mut self, value: u32) -> Self {
            self.broadcast_src_b_row = value;
            self
        }

        pub const fn broadcast_src_b_col0(mut self, value: u32) -> Self {
            self.broadcast_src_b_col0 = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn dst_row(mut self, value: u32) -> Self {
            self.dst_row = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::ELWMUL;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.flip_src_b) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_b,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_b);
            let f = def.fields()[1];
            if !f.fits(self.flip_src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.flip_src_a,
                    width: f.width(),
                });
            }
            word |= f.place(self.flip_src_a);
            let f = def.fields()[2];
            if !f.fits(self.broadcast_src_b_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.broadcast_src_b_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.broadcast_src_b_row);
            let f = def.fields()[3];
            if !f.fits(self.broadcast_src_b_col0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.broadcast_src_b_col0,
                    width: f.width(),
                });
            }
            word |= f.place(self.broadcast_src_b_col0);
            let f = def.fields()[4];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[5];
            if !f.fits(self.dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_row);
            Ok(Instruction::new(word, def))
        }
    }

    /// `DOTPV_BH`.
    pub const fn dotpv(
        flip_src_b: u32,
        flip_src_a: u32,
        addr_mod: u32,
        dst_row: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::DOTPV;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(flip_src_b) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: flip_src_b,
                width: f.width(),
            });
        }
        word |= f.place(flip_src_b);
        let f = def.fields()[1];
        if !f.fits(flip_src_a) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: flip_src_a,
                width: f.width(),
            });
        }
        word |= f.place(flip_src_a);
        let f = def.fields()[2];
        if !f.fits(addr_mod) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_mod,
                width: f.width(),
            });
        }
        word |= f.place(addr_mod);
        let f = def.fields()[3];
        if !f.fits(dst_row) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: dst_row,
                width: f.width(),
            });
        }
        word |= f.place(dst_row);
        Ok(Instruction::new(word, def))
    }

    /// `MOVDBGA2D_BH`, built field by field.
    ///
    /// 5 operands is too many to pass positionally without inviting a
    /// transposition, so each is named: `Movdbga2D::ZERO.use_dst32b_lo(1).encode()`.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Movdbga2D {
        use_dst32b_lo: u32,
        src_row: u32,
        addr_mod: u32,
        move8_rows: u32,
        dst_row: u32,
    }

    impl Movdbga2D {
        /// Every operand zero. Fixed bits are added by [`Self::encode`].
        pub const ZERO: Self = Movdbga2D {
            use_dst32b_lo: 0,
            src_row: 0,
            addr_mod: 0,
            move8_rows: 0,
            dst_row: 0,
        };

        pub const fn use_dst32b_lo(mut self, value: u32) -> Self {
            self.use_dst32b_lo = value;
            self
        }

        pub const fn src_row(mut self, value: u32) -> Self {
            self.src_row = value;
            self
        }

        pub const fn addr_mod(mut self, value: u32) -> Self {
            self.addr_mod = value;
            self
        }

        pub const fn move8_rows(mut self, value: u32) -> Self {
            self.move8_rows = value;
            self
        }

        pub const fn dst_row(mut self, value: u32) -> Self {
            self.dst_row = value;
            self
        }

        pub const fn encode(self) -> Result<Instruction, EncodeError> {
            let def = &defs::MOVDBGA2D;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(self.use_dst32b_lo) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.use_dst32b_lo,
                    width: f.width(),
                });
            }
            word |= f.place(self.use_dst32b_lo);
            let f = def.fields()[1];
            if !f.fits(self.src_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.src_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.src_row);
            let f = def.fields()[2];
            if !f.fits(self.addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(self.addr_mod);
            let f = def.fields()[3];
            if !f.fits(self.move8_rows) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.move8_rows,
                    width: f.width(),
                });
            }
            word |= f.place(self.move8_rows);
            let f = def.fields()[4];
            if !f.fits(self.dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: self.dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(self.dst_row);
            Ok(Instruction::new(word, def))
        }
    }

    /// `SHIFTXB_BH`.
    pub const fn shiftxb(
        addr_mod: u32,
        shift_in_zero: u32,
        src_row: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::SHIFTXB;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(addr_mod) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_mod,
                width: f.width(),
            });
        }
        word |= f.place(addr_mod);
        let f = def.fields()[1];
        if !f.fits(shift_in_zero) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: shift_in_zero,
                width: f.width(),
            });
        }
        word |= f.place(shift_in_zero);
        let f = def.fields()[2];
        if !f.fits(src_row) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: src_row,
                width: f.width(),
            });
        }
        word |= f.place(src_row);
        Ok(Instruction::new(word, def))
    }

    /// `ZEROACC_BH`.
    pub const fn zeroacc(
        mode: u32,
        use_dst32b: u32,
        addr_mod: u32,
        imm10: u32,
    ) -> Result<Instruction, EncodeError> {
        let def = &defs::ZEROACC;
        let mut word = def.skeleton();
        let f = def.fields()[0];
        if !f.fits(mode) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: mode,
                width: f.width(),
            });
        }
        word |= f.place(mode);
        let f = def.fields()[1];
        if !f.fits(use_dst32b) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: use_dst32b,
                width: f.width(),
            });
        }
        word |= f.place(use_dst32b);
        let f = def.fields()[2];
        if !f.fits(addr_mod) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: addr_mod,
                width: f.width(),
            });
        }
        word |= f.place(addr_mod);
        let f = def.fields()[3];
        if !f.fits(imm10) {
            return Err(EncodeError::FieldTooLarge {
                instruction: def.key(),
                field: f.name(),
                value: imm10,
                width: f.width(),
            });
        }
        word |= f.place(imm10);
        Ok(Instruction::new(word, def))
    }

    /// Encoders for the forms Blackhole replaces. Reaching one has to be
    /// deliberate.
    pub mod wormhole {
        use super::super::*;

        /// `STALLWAIT`.
        pub const fn stallwait(
            block_mask: u32,
            condition_mask: u32,
        ) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::STALLWAIT;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(block_mask) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: block_mask,
                    width: f.width(),
                });
            }
            word |= f.place(block_mask);
            let f = def.fields()[1];
            if !f.fits(condition_mask) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: condition_mask,
                    width: f.width(),
                });
            }
            word |= f.place(condition_mask);
            Ok(Instruction::new(word, def))
        }

        /// `PACR`, built field by field.
        ///
        /// 7 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Pacr::ZERO.addr_mod(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Pacr {
            addr_mod: u32,
            zero_write: u32,
            packer_mask: u32,
            ovrd_thread_id: u32,
            concat: u32,
            flush: u32,
            last: u32,
        }

        impl Pacr {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Pacr {
                addr_mod: 0,
                zero_write: 0,
                packer_mask: 0,
                ovrd_thread_id: 0,
                concat: 0,
                flush: 0,
                last: 0,
            };

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn zero_write(mut self, value: u32) -> Self {
                self.zero_write = value;
                self
            }

            pub const fn packer_mask(mut self, value: u32) -> Self {
                self.packer_mask = value;
                self
            }

            pub const fn ovrd_thread_id(mut self, value: u32) -> Self {
                self.ovrd_thread_id = value;
                self
            }

            pub const fn concat(mut self, value: u32) -> Self {
                self.concat = value;
                self
            }

            pub const fn flush(mut self, value: u32) -> Self {
                self.flush = value;
                self
            }

            pub const fn last(mut self, value: u32) -> Self {
                self.last = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::PACR;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[1];
                if !f.fits(self.zero_write) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.zero_write,
                        width: f.width(),
                    });
                }
                word |= f.place(self.zero_write);
                let f = def.fields()[2];
                if !f.fits(self.packer_mask) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.packer_mask,
                        width: f.width(),
                    });
                }
                word |= f.place(self.packer_mask);
                let f = def.fields()[3];
                if !f.fits(self.ovrd_thread_id) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.ovrd_thread_id,
                        width: f.width(),
                    });
                }
                word |= f.place(self.ovrd_thread_id);
                let f = def.fields()[4];
                if !f.fits(self.concat) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.concat,
                        width: f.width(),
                    });
                }
                word |= f.place(self.concat);
                let f = def.fields()[5];
                if !f.fits(self.flush) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.flush,
                        width: f.width(),
                    });
                }
                word |= f.place(self.flush);
                let f = def.fields()[6];
                if !f.fits(self.last) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.last,
                        width: f.width(),
                    });
                }
                word |= f.place(self.last);
                Ok(Instruction::new(word, def))
            }
        }

        /// `SFPLUTFP32`.
        pub const fn sfplutfp32(vd: u32, mod1: u32) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::SFPLUTFP32;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: vd,
                    width: f.width(),
                });
            }
            word |= f.place(vd);
            let f = def.fields()[1];
            if !f.fits(mod1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: mod1,
                    width: f.width(),
                });
            }
            word |= f.place(mod1);
            Ok(Instruction::new(word, def))
        }

        /// `MOVD2A`, built field by field.
        ///
        /// 5 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Movd2A::ZERO.use_dst32b_lo(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Movd2A {
            use_dst32b_lo: u32,
            src_row: u32,
            addr_mod: u32,
            move4_rows: u32,
            dst_row: u32,
        }

        impl Movd2A {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Movd2A {
                use_dst32b_lo: 0,
                src_row: 0,
                addr_mod: 0,
                move4_rows: 0,
                dst_row: 0,
            };

            pub const fn use_dst32b_lo(mut self, value: u32) -> Self {
                self.use_dst32b_lo = value;
                self
            }

            pub const fn src_row(mut self, value: u32) -> Self {
                self.src_row = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn move4_rows(mut self, value: u32) -> Self {
                self.move4_rows = value;
                self
            }

            pub const fn dst_row(mut self, value: u32) -> Self {
                self.dst_row = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::MOVD2A;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.use_dst32b_lo) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.use_dst32b_lo,
                        width: f.width(),
                    });
                }
                word |= f.place(self.use_dst32b_lo);
                let f = def.fields()[1];
                if !f.fits(self.src_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.src_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.src_row);
                let f = def.fields()[2];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[3];
                if !f.fits(self.move4_rows) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.move4_rows,
                        width: f.width(),
                    });
                }
                word |= f.place(self.move4_rows);
                let f = def.fields()[4];
                if !f.fits(self.dst_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.dst_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.dst_row);
                Ok(Instruction::new(word, def))
            }
        }

        /// `MOVD2B`, built field by field.
        ///
        /// 5 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Movd2B::ZERO.use_dst32b_lo(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Movd2B {
            use_dst32b_lo: u32,
            src_row: u32,
            addr_mod: u32,
            move4_rows: u32,
            dst_row: u32,
        }

        impl Movd2B {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Movd2B {
                use_dst32b_lo: 0,
                src_row: 0,
                addr_mod: 0,
                move4_rows: 0,
                dst_row: 0,
            };

            pub const fn use_dst32b_lo(mut self, value: u32) -> Self {
                self.use_dst32b_lo = value;
                self
            }

            pub const fn src_row(mut self, value: u32) -> Self {
                self.src_row = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn move4_rows(mut self, value: u32) -> Self {
                self.move4_rows = value;
                self
            }

            pub const fn dst_row(mut self, value: u32) -> Self {
                self.dst_row = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::MOVD2B;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.use_dst32b_lo) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.use_dst32b_lo,
                        width: f.width(),
                    });
                }
                word |= f.place(self.use_dst32b_lo);
                let f = def.fields()[1];
                if !f.fits(self.src_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.src_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.src_row);
                let f = def.fields()[2];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[3];
                if !f.fits(self.move4_rows) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.move4_rows,
                        width: f.width(),
                    });
                }
                word |= f.place(self.move4_rows);
                let f = def.fields()[4];
                if !f.fits(self.dst_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.dst_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.dst_row);
                Ok(Instruction::new(word, def))
            }
        }

        /// `MOVB2A`.
        pub const fn movb2_a(
            src_a_row: u32,
            addr_mod: u32,
            move4_rows: u32,
            src_b_row: u32,
        ) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::MOVB2A;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(src_a_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: src_a_row,
                    width: f.width(),
                });
            }
            word |= f.place(src_a_row);
            let f = def.fields()[1];
            if !f.fits(addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(addr_mod);
            let f = def.fields()[2];
            if !f.fits(move4_rows) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: move4_rows,
                    width: f.width(),
                });
            }
            word |= f.place(move4_rows);
            let f = def.fields()[3];
            if !f.fits(src_b_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: src_b_row,
                    width: f.width(),
                });
            }
            word |= f.place(src_b_row);
            Ok(Instruction::new(word, def))
        }

        /// `MOVDBGA2D`, built field by field.
        ///
        /// 5 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Movdbga2D::ZERO.use_dst32b_lo(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Movdbga2D {
            use_dst32b_lo: u32,
            src_row: u32,
            addr_mod: u32,
            move8_rows: u32,
            dst_row: u32,
        }

        impl Movdbga2D {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Movdbga2D {
                use_dst32b_lo: 0,
                src_row: 0,
                addr_mod: 0,
                move8_rows: 0,
                dst_row: 0,
            };

            pub const fn use_dst32b_lo(mut self, value: u32) -> Self {
                self.use_dst32b_lo = value;
                self
            }

            pub const fn src_row(mut self, value: u32) -> Self {
                self.src_row = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn move8_rows(mut self, value: u32) -> Self {
                self.move8_rows = value;
                self
            }

            pub const fn dst_row(mut self, value: u32) -> Self {
                self.dst_row = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::MOVDBGA2D;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.use_dst32b_lo) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.use_dst32b_lo,
                        width: f.width(),
                    });
                }
                word |= f.place(self.use_dst32b_lo);
                let f = def.fields()[1];
                if !f.fits(self.src_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.src_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.src_row);
                let f = def.fields()[2];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[3];
                if !f.fits(self.move8_rows) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.move8_rows,
                        width: f.width(),
                    });
                }
                word |= f.place(self.move8_rows);
                let f = def.fields()[4];
                if !f.fits(self.dst_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.dst_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.dst_row);
                Ok(Instruction::new(word, def))
            }
        }

        /// `MOVA2D`, built field by field.
        ///
        /// 5 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Mova2D::ZERO.use_dst32b_lo(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Mova2D {
            use_dst32b_lo: u32,
            src_row: u32,
            addr_mod: u32,
            move8_rows: u32,
            dst_row: u32,
        }

        impl Mova2D {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Mova2D {
                use_dst32b_lo: 0,
                src_row: 0,
                addr_mod: 0,
                move8_rows: 0,
                dst_row: 0,
            };

            pub const fn use_dst32b_lo(mut self, value: u32) -> Self {
                self.use_dst32b_lo = value;
                self
            }

            pub const fn src_row(mut self, value: u32) -> Self {
                self.src_row = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn move8_rows(mut self, value: u32) -> Self {
                self.move8_rows = value;
                self
            }

            pub const fn dst_row(mut self, value: u32) -> Self {
                self.dst_row = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::MOVA2D;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.use_dst32b_lo) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.use_dst32b_lo,
                        width: f.width(),
                    });
                }
                word |= f.place(self.use_dst32b_lo);
                let f = def.fields()[1];
                if !f.fits(self.src_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.src_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.src_row);
                let f = def.fields()[2];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[3];
                if !f.fits(self.move8_rows) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.move8_rows,
                        width: f.width(),
                    });
                }
                word |= f.place(self.move8_rows);
                let f = def.fields()[4];
                if !f.fits(self.dst_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.dst_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.dst_row);
                Ok(Instruction::new(word, def))
            }
        }

        /// `MOVB2D`, built field by field.
        ///
        /// 7 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Movb2D::ZERO.use_dst32b_lo(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Movb2D {
            use_dst32b_lo: u32,
            src_row: u32,
            addr_mod: u32,
            move4_rows: u32,
            broadcast1_row_to8: u32,
            broadcast_col0: u32,
            dst_row: u32,
        }

        impl Movb2D {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Movb2D {
                use_dst32b_lo: 0,
                src_row: 0,
                addr_mod: 0,
                move4_rows: 0,
                broadcast1_row_to8: 0,
                broadcast_col0: 0,
                dst_row: 0,
            };

            pub const fn use_dst32b_lo(mut self, value: u32) -> Self {
                self.use_dst32b_lo = value;
                self
            }

            pub const fn src_row(mut self, value: u32) -> Self {
                self.src_row = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn move4_rows(mut self, value: u32) -> Self {
                self.move4_rows = value;
                self
            }

            pub const fn broadcast1_row_to8(mut self, value: u32) -> Self {
                self.broadcast1_row_to8 = value;
                self
            }

            pub const fn broadcast_col0(mut self, value: u32) -> Self {
                self.broadcast_col0 = value;
                self
            }

            pub const fn dst_row(mut self, value: u32) -> Self {
                self.dst_row = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::MOVB2D;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.use_dst32b_lo) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.use_dst32b_lo,
                        width: f.width(),
                    });
                }
                word |= f.place(self.use_dst32b_lo);
                let f = def.fields()[1];
                if !f.fits(self.src_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.src_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.src_row);
                let f = def.fields()[2];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[3];
                if !f.fits(self.move4_rows) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.move4_rows,
                        width: f.width(),
                    });
                }
                word |= f.place(self.move4_rows);
                let f = def.fields()[4];
                if !f.fits(self.broadcast1_row_to8) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.broadcast1_row_to8,
                        width: f.width(),
                    });
                }
                word |= f.place(self.broadcast1_row_to8);
                let f = def.fields()[5];
                if !f.fits(self.broadcast_col0) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.broadcast_col0,
                        width: f.width(),
                    });
                }
                word |= f.place(self.broadcast_col0);
                let f = def.fields()[6];
                if !f.fits(self.dst_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.dst_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.dst_row);
                Ok(Instruction::new(word, def))
            }
        }

        /// `ELWMUL`, built field by field.
        ///
        /// 6 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Elwmul::ZERO.flip_src_b(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Elwmul {
            flip_src_b: u32,
            flip_src_a: u32,
            broadcast_src_b_row: u32,
            broadcast_src_b_col0: u32,
            addr_mod: u32,
            dst_row: u32,
        }

        impl Elwmul {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Elwmul {
                flip_src_b: 0,
                flip_src_a: 0,
                broadcast_src_b_row: 0,
                broadcast_src_b_col0: 0,
                addr_mod: 0,
                dst_row: 0,
            };

            pub const fn flip_src_b(mut self, value: u32) -> Self {
                self.flip_src_b = value;
                self
            }

            pub const fn flip_src_a(mut self, value: u32) -> Self {
                self.flip_src_a = value;
                self
            }

            pub const fn broadcast_src_b_row(mut self, value: u32) -> Self {
                self.broadcast_src_b_row = value;
                self
            }

            pub const fn broadcast_src_b_col0(mut self, value: u32) -> Self {
                self.broadcast_src_b_col0 = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn dst_row(mut self, value: u32) -> Self {
                self.dst_row = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::ELWMUL;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.flip_src_b) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.flip_src_b,
                        width: f.width(),
                    });
                }
                word |= f.place(self.flip_src_b);
                let f = def.fields()[1];
                if !f.fits(self.flip_src_a) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.flip_src_a,
                        width: f.width(),
                    });
                }
                word |= f.place(self.flip_src_a);
                let f = def.fields()[2];
                if !f.fits(self.broadcast_src_b_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.broadcast_src_b_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.broadcast_src_b_row);
                let f = def.fields()[3];
                if !f.fits(self.broadcast_src_b_col0) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.broadcast_src_b_col0,
                        width: f.width(),
                    });
                }
                word |= f.place(self.broadcast_src_b_col0);
                let f = def.fields()[4];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[5];
                if !f.fits(self.dst_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.dst_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.dst_row);
                Ok(Instruction::new(word, def))
            }
        }

        /// `ELWADD`, built field by field.
        ///
        /// 7 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Elwadd::ZERO.flip_src_b(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Elwadd {
            flip_src_b: u32,
            flip_src_a: u32,
            add_dst: u32,
            broadcast_src_b_row: u32,
            broadcast_src_b_col0: u32,
            addr_mod: u32,
            dst_row: u32,
        }

        impl Elwadd {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Elwadd {
                flip_src_b: 0,
                flip_src_a: 0,
                add_dst: 0,
                broadcast_src_b_row: 0,
                broadcast_src_b_col0: 0,
                addr_mod: 0,
                dst_row: 0,
            };

            pub const fn flip_src_b(mut self, value: u32) -> Self {
                self.flip_src_b = value;
                self
            }

            pub const fn flip_src_a(mut self, value: u32) -> Self {
                self.flip_src_a = value;
                self
            }

            pub const fn add_dst(mut self, value: u32) -> Self {
                self.add_dst = value;
                self
            }

            pub const fn broadcast_src_b_row(mut self, value: u32) -> Self {
                self.broadcast_src_b_row = value;
                self
            }

            pub const fn broadcast_src_b_col0(mut self, value: u32) -> Self {
                self.broadcast_src_b_col0 = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn dst_row(mut self, value: u32) -> Self {
                self.dst_row = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::ELWADD;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.flip_src_b) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.flip_src_b,
                        width: f.width(),
                    });
                }
                word |= f.place(self.flip_src_b);
                let f = def.fields()[1];
                if !f.fits(self.flip_src_a) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.flip_src_a,
                        width: f.width(),
                    });
                }
                word |= f.place(self.flip_src_a);
                let f = def.fields()[2];
                if !f.fits(self.add_dst) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.add_dst,
                        width: f.width(),
                    });
                }
                word |= f.place(self.add_dst);
                let f = def.fields()[3];
                if !f.fits(self.broadcast_src_b_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.broadcast_src_b_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.broadcast_src_b_row);
                let f = def.fields()[4];
                if !f.fits(self.broadcast_src_b_col0) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.broadcast_src_b_col0,
                        width: f.width(),
                    });
                }
                word |= f.place(self.broadcast_src_b_col0);
                let f = def.fields()[5];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[6];
                if !f.fits(self.dst_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.dst_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.dst_row);
                Ok(Instruction::new(word, def))
            }
        }

        /// `ELWSUB`, built field by field.
        ///
        /// 7 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Elwsub::ZERO.flip_src_b(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Elwsub {
            flip_src_b: u32,
            flip_src_a: u32,
            add_dst: u32,
            broadcast_src_b_row: u32,
            broadcast_src_b_col0: u32,
            addr_mod: u32,
            dst_row: u32,
        }

        impl Elwsub {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Elwsub {
                flip_src_b: 0,
                flip_src_a: 0,
                add_dst: 0,
                broadcast_src_b_row: 0,
                broadcast_src_b_col0: 0,
                addr_mod: 0,
                dst_row: 0,
            };

            pub const fn flip_src_b(mut self, value: u32) -> Self {
                self.flip_src_b = value;
                self
            }

            pub const fn flip_src_a(mut self, value: u32) -> Self {
                self.flip_src_a = value;
                self
            }

            pub const fn add_dst(mut self, value: u32) -> Self {
                self.add_dst = value;
                self
            }

            pub const fn broadcast_src_b_row(mut self, value: u32) -> Self {
                self.broadcast_src_b_row = value;
                self
            }

            pub const fn broadcast_src_b_col0(mut self, value: u32) -> Self {
                self.broadcast_src_b_col0 = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn dst_row(mut self, value: u32) -> Self {
                self.dst_row = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::ELWSUB;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.flip_src_b) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.flip_src_b,
                        width: f.width(),
                    });
                }
                word |= f.place(self.flip_src_b);
                let f = def.fields()[1];
                if !f.fits(self.flip_src_a) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.flip_src_a,
                        width: f.width(),
                    });
                }
                word |= f.place(self.flip_src_a);
                let f = def.fields()[2];
                if !f.fits(self.add_dst) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.add_dst,
                        width: f.width(),
                    });
                }
                word |= f.place(self.add_dst);
                let f = def.fields()[3];
                if !f.fits(self.broadcast_src_b_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.broadcast_src_b_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.broadcast_src_b_row);
                let f = def.fields()[4];
                if !f.fits(self.broadcast_src_b_col0) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.broadcast_src_b_col0,
                        width: f.width(),
                    });
                }
                word |= f.place(self.broadcast_src_b_col0);
                let f = def.fields()[5];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[6];
                if !f.fits(self.dst_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.dst_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.dst_row);
                Ok(Instruction::new(word, def))
            }
        }

        /// `GMPOOL`, built field by field.
        ///
        /// 5 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Gmpool::ZERO.flip_src_b(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Gmpool {
            flip_src_b: u32,
            flip_src_a: u32,
            addr_mod: u32,
            arg_max: u32,
            dst_row: u32,
        }

        impl Gmpool {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Gmpool {
                flip_src_b: 0,
                flip_src_a: 0,
                addr_mod: 0,
                arg_max: 0,
                dst_row: 0,
            };

            pub const fn flip_src_b(mut self, value: u32) -> Self {
                self.flip_src_b = value;
                self
            }

            pub const fn flip_src_a(mut self, value: u32) -> Self {
                self.flip_src_a = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn arg_max(mut self, value: u32) -> Self {
                self.arg_max = value;
                self
            }

            pub const fn dst_row(mut self, value: u32) -> Self {
                self.dst_row = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::GMPOOL;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.flip_src_b) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.flip_src_b,
                        width: f.width(),
                    });
                }
                word |= f.place(self.flip_src_b);
                let f = def.fields()[1];
                if !f.fits(self.flip_src_a) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.flip_src_a,
                        width: f.width(),
                    });
                }
                word |= f.place(self.flip_src_a);
                let f = def.fields()[2];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[3];
                if !f.fits(self.arg_max) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.arg_max,
                        width: f.width(),
                    });
                }
                word |= f.place(self.arg_max);
                let f = def.fields()[4];
                if !f.fits(self.dst_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.dst_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.dst_row);
                Ok(Instruction::new(word, def))
            }
        }

        /// `ZEROACC`, built field by field.
        ///
        /// 5 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Zeroacc::ZERO.use_dst32b(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Zeroacc {
            use_dst32b: u32,
            mode: u32,
            revert: u32,
            addr_mod: u32,
            imm10: u32,
        }

        impl Zeroacc {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Zeroacc {
                use_dst32b: 0,
                mode: 0,
                revert: 0,
                addr_mod: 0,
                imm10: 0,
            };

            pub const fn use_dst32b(mut self, value: u32) -> Self {
                self.use_dst32b = value;
                self
            }

            pub const fn mode(mut self, value: u32) -> Self {
                self.mode = value;
                self
            }

            pub const fn revert(mut self, value: u32) -> Self {
                self.revert = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn imm10(mut self, value: u32) -> Self {
                self.imm10 = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::ZEROACC;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.use_dst32b) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.use_dst32b,
                        width: f.width(),
                    });
                }
                word |= f.place(self.use_dst32b);
                let f = def.fields()[1];
                if !f.fits(self.mode) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.mode,
                        width: f.width(),
                    });
                }
                word |= f.place(self.mode);
                let f = def.fields()[2];
                if !f.fits(self.revert) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.revert,
                        width: f.width(),
                    });
                }
                word |= f.place(self.revert);
                let f = def.fields()[3];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[4];
                if !f.fits(self.imm10) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.imm10,
                        width: f.width(),
                    });
                }
                word |= f.place(self.imm10);
                Ok(Instruction::new(word, def))
            }
        }

        /// `SHIFTXB`.
        pub const fn shiftxb(
            addr_mod: u32,
            shift_in_zero: u32,
            src_row: u32,
        ) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::SHIFTXB;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(addr_mod);
            let f = def.fields()[1];
            if !f.fits(shift_in_zero) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: shift_in_zero,
                    width: f.width(),
                });
            }
            word |= f.place(shift_in_zero);
            let f = def.fields()[2];
            if !f.fits(src_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: src_row,
                    width: f.width(),
                });
            }
            word |= f.place(src_row);
            Ok(Instruction::new(word, def))
        }

        /// `MVMUL`, built field by field.
        ///
        /// 5 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Mvmul::ZERO.flip_src_b(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Mvmul {
            flip_src_b: u32,
            flip_src_a: u32,
            broadcast_src_b_row: u32,
            addr_mod: u32,
            dst_row: u32,
        }

        impl Mvmul {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Mvmul {
                flip_src_b: 0,
                flip_src_a: 0,
                broadcast_src_b_row: 0,
                addr_mod: 0,
                dst_row: 0,
            };

            pub const fn flip_src_b(mut self, value: u32) -> Self {
                self.flip_src_b = value;
                self
            }

            pub const fn flip_src_a(mut self, value: u32) -> Self {
                self.flip_src_a = value;
                self
            }

            pub const fn broadcast_src_b_row(mut self, value: u32) -> Self {
                self.broadcast_src_b_row = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn dst_row(mut self, value: u32) -> Self {
                self.dst_row = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::MVMUL;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.flip_src_b) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.flip_src_b,
                        width: f.width(),
                    });
                }
                word |= f.place(self.flip_src_b);
                let f = def.fields()[1];
                if !f.fits(self.flip_src_a) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.flip_src_a,
                        width: f.width(),
                    });
                }
                word |= f.place(self.flip_src_a);
                let f = def.fields()[2];
                if !f.fits(self.broadcast_src_b_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.broadcast_src_b_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.broadcast_src_b_row);
                let f = def.fields()[3];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[4];
                if !f.fits(self.dst_row) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.dst_row,
                        width: f.width(),
                    });
                }
                word |= f.place(self.dst_row);
                Ok(Instruction::new(word, def))
            }
        }

        /// `DOTPV`.
        pub const fn dotpv(
            flip_src_b: u32,
            flip_src_a: u32,
            addr_mod: u32,
            dst_row: u32,
        ) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::DOTPV;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(flip_src_b) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: flip_src_b,
                    width: f.width(),
                });
            }
            word |= f.place(flip_src_b);
            let f = def.fields()[1];
            if !f.fits(flip_src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: flip_src_a,
                    width: f.width(),
                });
            }
            word |= f.place(flip_src_a);
            let f = def.fields()[2];
            if !f.fits(addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(addr_mod);
            let f = def.fields()[3];
            if !f.fits(dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(dst_row);
            Ok(Instruction::new(word, def))
        }

        /// `GAPOOL`.
        pub const fn gapool(
            flip_src_b: u32,
            flip_src_a: u32,
            addr_mod: u32,
            dst_row: u32,
        ) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::GAPOOL;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(flip_src_b) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: flip_src_b,
                    width: f.width(),
                });
            }
            word |= f.place(flip_src_b);
            let f = def.fields()[1];
            if !f.fits(flip_src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: flip_src_a,
                    width: f.width(),
                });
            }
            word |= f.place(flip_src_a);
            let f = def.fields()[2];
            if !f.fits(addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(addr_mod);
            let f = def.fields()[3];
            if !f.fits(dst_row) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: dst_row,
                    width: f.width(),
                });
            }
            word |= f.place(dst_row);
            Ok(Instruction::new(word, def))
        }

        /// `SFPLOAD`.
        pub const fn sfpload(
            vd: u32,
            mod0: u32,
            addr_mod: u32,
            imm10: u32,
        ) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::SFPLOAD;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: vd,
                    width: f.width(),
                });
            }
            word |= f.place(vd);
            let f = def.fields()[1];
            if !f.fits(mod0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: mod0,
                    width: f.width(),
                });
            }
            word |= f.place(mod0);
            let f = def.fields()[2];
            if !f.fits(addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(addr_mod);
            let f = def.fields()[3];
            if !f.fits(imm10) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: imm10,
                    width: f.width(),
                });
            }
            word |= f.place(imm10);
            Ok(Instruction::new(word, def))
        }

        /// `SFPLOADMACRO`, built field by field.
        ///
        /// 6 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `Sfploadmacro::ZERO.macro_index(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct Sfploadmacro {
            macro_index: u32,
            vd_lo: u32,
            mod0: u32,
            addr_mod: u32,
            imm9: u32,
            vd_hi: u32,
        }

        impl Sfploadmacro {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = Sfploadmacro {
                macro_index: 0,
                vd_lo: 0,
                mod0: 0,
                addr_mod: 0,
                imm9: 0,
                vd_hi: 0,
            };

            pub const fn macro_index(mut self, value: u32) -> Self {
                self.macro_index = value;
                self
            }

            pub const fn vd_lo(mut self, value: u32) -> Self {
                self.vd_lo = value;
                self
            }

            pub const fn mod0(mut self, value: u32) -> Self {
                self.mod0 = value;
                self
            }

            pub const fn addr_mod(mut self, value: u32) -> Self {
                self.addr_mod = value;
                self
            }

            pub const fn imm9(mut self, value: u32) -> Self {
                self.imm9 = value;
                self
            }

            pub const fn vd_hi(mut self, value: u32) -> Self {
                self.vd_hi = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::SFPLOADMACRO;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.macro_index) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.macro_index,
                        width: f.width(),
                    });
                }
                word |= f.place(self.macro_index);
                let f = def.fields()[1];
                if !f.fits(self.vd_lo) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.vd_lo,
                        width: f.width(),
                    });
                }
                word |= f.place(self.vd_lo);
                let f = def.fields()[2];
                if !f.fits(self.mod0) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.mod0,
                        width: f.width(),
                    });
                }
                word |= f.place(self.mod0);
                let f = def.fields()[3];
                if !f.fits(self.addr_mod) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.addr_mod,
                        width: f.width(),
                    });
                }
                word |= f.place(self.addr_mod);
                let f = def.fields()[4];
                if !f.fits(self.imm9) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.imm9,
                        width: f.width(),
                    });
                }
                word |= f.place(self.imm9);
                let f = def.fields()[5];
                if !f.fits(self.vd_hi) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.vd_hi,
                        width: f.width(),
                    });
                }
                word |= f.place(self.vd_hi);
                Ok(Instruction::new(word, def))
            }
        }

        /// `SFPSTORE`.
        pub const fn sfpstore(
            vd: u32,
            mod0: u32,
            addr_mod: u32,
            imm10: u32,
        ) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::SFPSTORE;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: vd,
                    width: f.width(),
                });
            }
            word |= f.place(vd);
            let f = def.fields()[1];
            if !f.fits(mod0) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: mod0,
                    width: f.width(),
                });
            }
            word |= f.place(mod0);
            let f = def.fields()[2];
            if !f.fits(addr_mod) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: addr_mod,
                    width: f.width(),
                });
            }
            word |= f.place(addr_mod);
            let f = def.fields()[3];
            if !f.fits(imm10) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: imm10,
                    width: f.width(),
                });
            }
            word |= f.place(imm10);
            Ok(Instruction::new(word, def))
        }

        /// `SFPAND`.
        pub const fn sfpand(vc: u32, vd: u32) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::SFPAND;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(vc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: vc,
                    width: f.width(),
                });
            }
            word |= f.place(vc);
            let f = def.fields()[1];
            if !f.fits(vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: vd,
                    width: f.width(),
                });
            }
            word |= f.place(vd);
            Ok(Instruction::new(word, def))
        }

        /// `SFPOR`.
        pub const fn sfpor(vc: u32, vd: u32) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::SFPOR;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(vc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: vc,
                    width: f.width(),
                });
            }
            word |= f.place(vc);
            let f = def.fields()[1];
            if !f.fits(vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: vd,
                    width: f.width(),
                });
            }
            word |= f.place(vd);
            Ok(Instruction::new(word, def))
        }

        /// `SFPPUSHC`.
        pub const fn sfppushc(vd: u32) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::SFPPUSHC;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: vd,
                    width: f.width(),
                });
            }
            word |= f.place(vd);
            Ok(Instruction::new(word, def))
        }

        /// `SFPSTOCHRND`.
        pub const fn sfpstochrnd(
            stochastic_rounding: u32,
            vc: u32,
            vd: u32,
            mod1: u32,
        ) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::SFPSTOCHRND;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(stochastic_rounding) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: stochastic_rounding,
                    width: f.width(),
                });
            }
            word |= f.place(stochastic_rounding);
            let f = def.fields()[1];
            if !f.fits(vc) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: vc,
                    width: f.width(),
                });
            }
            word |= f.place(vc);
            let f = def.fields()[2];
            if !f.fits(vd) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: vd,
                    width: f.width(),
                });
            }
            word |= f.place(vd);
            let f = def.fields()[3];
            if !f.fits(mod1) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: mod1,
                    width: f.width(),
                });
            }
            word |= f.place(mod1);
            Ok(Instruction::new(word, def))
        }

        /// `SFPSTOCHRNDi`, built field by field.
        ///
        /// 7 operands is too many to pass positionally without inviting a
        /// transposition, so each is named: `SfpstochrnDi::ZERO.stochastic_rounding(1).encode()`.
        #[derive(Copy, Clone, Debug, Eq, PartialEq)]
        pub struct SfpstochrnDi {
            stochastic_rounding: u32,
            imm5: u32,
            vb: u32,
            vc: u32,
            vd: u32,
            use_imm5: u32,
            mod1: u32,
        }

        impl SfpstochrnDi {
            /// Every operand zero. Fixed bits are added by [`Self::encode`].
            pub const ZERO: Self = SfpstochrnDi {
                stochastic_rounding: 0,
                imm5: 0,
                vb: 0,
                vc: 0,
                vd: 0,
                use_imm5: 0,
                mod1: 0,
            };

            pub const fn stochastic_rounding(mut self, value: u32) -> Self {
                self.stochastic_rounding = value;
                self
            }

            pub const fn imm5(mut self, value: u32) -> Self {
                self.imm5 = value;
                self
            }

            pub const fn vb(mut self, value: u32) -> Self {
                self.vb = value;
                self
            }

            pub const fn vc(mut self, value: u32) -> Self {
                self.vc = value;
                self
            }

            pub const fn vd(mut self, value: u32) -> Self {
                self.vd = value;
                self
            }

            pub const fn use_imm5(mut self, value: u32) -> Self {
                self.use_imm5 = value;
                self
            }

            pub const fn mod1(mut self, value: u32) -> Self {
                self.mod1 = value;
                self
            }

            pub const fn encode(self) -> Result<Instruction, EncodeError> {
                let def = &defs::wormhole::SFPSTOCHRNDi;
                let mut word = def.skeleton();
                let f = def.fields()[0];
                if !f.fits(self.stochastic_rounding) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.stochastic_rounding,
                        width: f.width(),
                    });
                }
                word |= f.place(self.stochastic_rounding);
                let f = def.fields()[1];
                if !f.fits(self.imm5) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.imm5,
                        width: f.width(),
                    });
                }
                word |= f.place(self.imm5);
                let f = def.fields()[2];
                if !f.fits(self.vb) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.vb,
                        width: f.width(),
                    });
                }
                word |= f.place(self.vb);
                let f = def.fields()[3];
                if !f.fits(self.vc) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.vc,
                        width: f.width(),
                    });
                }
                word |= f.place(self.vc);
                let f = def.fields()[4];
                if !f.fits(self.vd) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.vd,
                        width: f.width(),
                    });
                }
                word |= f.place(self.vd);
                let f = def.fields()[5];
                if !f.fits(self.use_imm5) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.use_imm5,
                        width: f.width(),
                    });
                }
                word |= f.place(self.use_imm5);
                let f = def.fields()[6];
                if !f.fits(self.mod1) {
                    return Err(EncodeError::FieldTooLarge {
                        instruction: def.key(),
                        field: f.name(),
                        value: self.mod1,
                        width: f.width(),
                    });
                }
                word |= f.place(self.mod1);
                Ok(Instruction::new(word, def))
            }
        }

        /// `UNPACR_NOP_ZEROSRC`.
        pub const fn unpacr_nop_zerosrc(
            which_unpacker: u32,
            wait_like_unpacr: u32,
            both_banks: u32,
            negative_inf_src_a: u32,
        ) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::UNPACR_NOP_ZEROSRC;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(which_unpacker) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: which_unpacker,
                    width: f.width(),
                });
            }
            word |= f.place(which_unpacker);
            let f = def.fields()[1];
            if !f.fits(wait_like_unpacr) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: wait_like_unpacr,
                    width: f.width(),
                });
            }
            word |= f.place(wait_like_unpacr);
            let f = def.fields()[2];
            if !f.fits(both_banks) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: both_banks,
                    width: f.width(),
                });
            }
            word |= f.place(both_banks);
            let f = def.fields()[3];
            if !f.fits(negative_inf_src_a) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: negative_inf_src_a,
                    width: f.width(),
                });
            }
            word |= f.place(negative_inf_src_a);
            Ok(Instruction::new(word, def))
        }

        /// `UNPACR_NOP_SETDVALID`.
        pub const fn unpacr_nop_setdvalid(which_unpacker: u32) -> Result<Instruction, EncodeError> {
            let def = &defs::wormhole::UNPACR_NOP_SETDVALID;
            let mut word = def.skeleton();
            let f = def.fields()[0];
            if !f.fits(which_unpacker) {
                return Err(EncodeError::FieldTooLarge {
                    instruction: def.key(),
                    field: f.name(),
                    value: which_unpacker,
                    width: f.width(),
                });
            }
            word |= f.place(which_unpacker);
            Ok(Instruction::new(word, def))
        }
    }
}
