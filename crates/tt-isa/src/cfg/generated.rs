//! Tensix backend-configuration fields, generated from `cfg_defines.h`.
//!
//! **Do not edit.** Regenerate with `cargo xtask gen-cfg`; CI checks that this
//! file matches the header pinned in `PINS.toml`.
//!
//! Source: tt-metal `81989dcdb8f9b340c932ae7a71a346f4f08703eb`, `tt_metal/hw/inc/blackhole/cfg_defines.h`
//! (820 fields across 7 sections).
//!
//! Field names are kept exactly as `cfg_defines.h` spells them, so that a
//! name in the specification can be searched for here without translation.
//! That is why this module allows non-upper-case globals.
#![allow(non_upper_case_globals)]
#![allow(clippy::unreadable_literal)]

use super::{ConfigField, ConfigSpan, ThreadConfigField};

/// `uint32_t Config[2][CFG_STATE_SIZE * 4]` — the multiplier is in the
/// declaration, so one bank holds `CFG_STATE_SIZE * 4` words.
pub const CFG_STATE_SIZE: u32 = 56;

/// `struct {uint16_t Value, Padding;} ThreadConfig[3][THD_STATE_SIZE]`.
pub const THD_STATE_SIZE: u32 = 68;

/// `// Registers for ALU` — 49 fields, indexing `Config` (write with `WRCFG`).
pub mod alu {
    use super::*;

    /// First word index of this section, from `ALU_CFGREG_BASE_ADDR32`.
    pub const CFGREG_BASE: u16 = 0;

    pub const ALU_ACC_CTRL_Fp32_enabled: ConfigField = ConfigField::new(1, 29, 0x20000000);
    pub const ALU_ACC_CTRL_INT8_math_enabled: ConfigField = ConfigField::new(1, 31, 0x80000000);
    pub const ALU_ACC_CTRL_SFPU_Fp32_enabled: ConfigField = ConfigField::new(1, 30, 0x40000000);
    pub const ALU_ACC_CTRL_Zero_Flag_disabled_dst: ConfigField = ConfigField::new(2, 1, 0x2);
    pub const ALU_ACC_CTRL_Zero_Flag_disabled_src: ConfigField = ConfigField::new(2, 0, 0x1);
    pub const ALU_FORMAT_SPEC_REG0_SrcA: ConfigField = ConfigField::new(1, 17, 0x1e0000);
    pub const ALU_FORMAT_SPEC_REG0_SrcAUnsigned: ConfigField = ConfigField::new(1, 15, 0x8000);
    pub const ALU_FORMAT_SPEC_REG0_SrcBUnsigned: ConfigField = ConfigField::new(1, 16, 0x10000);
    pub const ALU_FORMAT_SPEC_REG1_SrcB: ConfigField = ConfigField::new(1, 21, 0x1e00000);
    pub const ALU_FORMAT_SPEC_REG2_Dstacc: ConfigField = ConfigField::new(1, 25, 0x1e000000);
    pub const ALU_FORMAT_SPEC_REG_Dstacc_override: ConfigField = ConfigField::new(0, 14, 0x4000);
    pub const ALU_FORMAT_SPEC_REG_Dstacc_val: ConfigField = ConfigField::new(0, 10, 0x3c00);
    pub const ALU_FORMAT_SPEC_REG_SrcA_override: ConfigField = ConfigField::new(0, 4, 0x10);
    pub const ALU_FORMAT_SPEC_REG_SrcA_val: ConfigField = ConfigField::new(0, 0, 0xf);
    pub const ALU_FORMAT_SPEC_REG_SrcB_override: ConfigField = ConfigField::new(0, 9, 0x200);
    pub const ALU_FORMAT_SPEC_REG_SrcB_val: ConfigField = ConfigField::new(0, 5, 0x1e0);
    pub const ALU_ROUNDING_MODE_Bfp8_HF: ConfigField = ConfigField::new(1, 14, 0x4000);
    pub const ALU_ROUNDING_MODE_Fpu_srnd_en: ConfigField = ConfigField::new(1, 0, 0x1);
    pub const ALU_ROUNDING_MODE_GS_LF: ConfigField = ConfigField::new(1, 13, 0x2000);
    pub const ALU_ROUNDING_MODE_Gasket_srnd_en: ConfigField = ConfigField::new(1, 1, 0x2);
    pub const ALU_ROUNDING_MODE_Packer_srnd_en: ConfigField = ConfigField::new(1, 2, 0x4);
    pub const ALU_ROUNDING_MODE_Padding: ConfigField = ConfigField::new(1, 3, 0x1ff8);
    pub const DEST_OFFSET_Enable: ConfigField = ConfigField::new(5, 0, 0x1);
    pub const DEST_REGW_BASE_Base: ConfigField = ConfigField::new(6, 0, 0xffff);
    pub const DEST_SP_BASE_Base: ConfigField = ConfigField::new(7, 0, 0xffff);
    pub const DISABLE_RISC_BP_Disable_bmp_clear_main: ConfigField =
        ConfigField::new(2, 27, 0x8000000);
    pub const DISABLE_RISC_BP_Disable_bmp_clear_ncrisc: ConfigField =
        ConfigField::new(2, 31, 0x80000000);
    pub const DISABLE_RISC_BP_Disable_bmp_clear_trisc: ConfigField =
        ConfigField::new(2, 28, 0x70000000);
    pub const DISABLE_RISC_BP_Disable_main: ConfigField = ConfigField::new(2, 22, 0x400000);
    pub const DISABLE_RISC_BP_Disable_ncrisc: ConfigField = ConfigField::new(2, 26, 0x4000000);
    pub const DISABLE_RISC_BP_Disable_trisc: ConfigField = ConfigField::new(2, 23, 0x3800000);
    pub const ECC_SCRUBBER_Delay: ConfigField = ConfigField::new(3, 3, 0x3ff8);
    pub const ECC_SCRUBBER_Enable: ConfigField = ConfigField::new(3, 0, 0x1);
    pub const ECC_SCRUBBER_Scrub_On_Error: ConfigField = ConfigField::new(3, 1, 0x2);
    pub const ECC_SCRUBBER_Scrub_On_Error_Immediately: ConfigField = ConfigField::new(3, 2, 0x4);
    pub const INT_DESCALE_Enable: ConfigField = ConfigField::new(8, 0, 0x1);
    pub const INT_DESCALE_Mode: ConfigField = ConfigField::new(8, 1, 0x2);
    pub const RISC_DEST_ACCESS_CTRL_SEC0_fmt: ConfigField = ConfigField::new(3, 16, 0x70000);
    pub const RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle: ConfigField = ConfigField::new(3, 14, 0x4000);
    pub const RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int: ConfigField =
        ConfigField::new(3, 15, 0x8000);
    pub const RISC_DEST_ACCESS_CTRL_SEC1_fmt: ConfigField = ConfigField::new(3, 21, 0xe00000);
    pub const RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle: ConfigField = ConfigField::new(3, 19, 0x80000);
    pub const RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int: ConfigField =
        ConfigField::new(3, 20, 0x100000);
    pub const RISC_DEST_ACCESS_CTRL_SEC2_fmt: ConfigField = ConfigField::new(3, 26, 0x1c000000);
    pub const RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle: ConfigField =
        ConfigField::new(3, 24, 0x1000000);
    pub const RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int: ConfigField =
        ConfigField::new(3, 25, 0x2000000);
    pub const STACC_RELU_ApplyRelu: ConfigField = ConfigField::new(2, 2, 0x3c);
    pub const STACC_RELU_ReluThreshold: ConfigField = ConfigField::new(2, 6, 0x3fffc0);
    pub const STATE_RESET_EN: ConfigField = ConfigField::new(4, 0, 0x1);
}

/// `// Registers for GLOBAL` — 63 fields, indexing `Config` (write with `WRCFG`).
pub mod global {
    use super::*;

    /// First word index of this section, from `GLOBAL_CFGREG_BASE_ADDR32`.
    pub const CFGREG_BASE: u16 = 180;

    pub const BRISC_END_PC_PC: ConfigField = ConfigField::new(206, 0, 0xffffffff);
    pub const CG_SRC_PIPELINE_GateSrcAPipeEn: ConfigField = ConfigField::new(184, 0, 0x1);
    pub const CG_SRC_PIPELINE_GateSrcBPipeEn: ConfigField = ConfigField::new(184, 1, 0x2);
    pub const CHICKEN_BITS_sfpu_scbd_disable: ConfigField = ConfigField::new(222, 0, 0x1);
    pub const DEST_ACCESS_CFG_disable_full_write_dest_q_bypass: ConfigField =
        ConfigField::new(220, 2, 0x4);
    pub const DEST_ACCESS_CFG_remap_addrs: ConfigField = ConfigField::new(220, 1, 0x2);
    pub const DEST_ACCESS_CFG_swizzle_32b: ConfigField = ConfigField::new(220, 0, 0x1);
    pub const DEST_ACCESS_CFG_zeroacc_absolute_tile_mode: ConfigField =
        ConfigField::new(220, 3, 0x8);
    pub const DEST_TARGET_REG_CFG_PACK_SEC0_Offset: ConfigField = ConfigField::new(180, 0, 0xfff);
    pub const DEST_TARGET_REG_CFG_PACK_SEC0_ZOffset: ConfigField =
        ConfigField::new(180, 12, 0x3f000);
    pub const DEST_TARGET_REG_CFG_PACK_SEC1_Offset: ConfigField = ConfigField::new(181, 0, 0xfff);
    pub const DEST_TARGET_REG_CFG_PACK_SEC1_ZOffset: ConfigField =
        ConfigField::new(181, 12, 0x3f000);
    pub const DEST_TARGET_REG_CFG_PACK_SEC2_Offset: ConfigField = ConfigField::new(182, 0, 0xfff);
    pub const DEST_TARGET_REG_CFG_PACK_SEC2_ZOffset: ConfigField =
        ConfigField::new(182, 12, 0x3f000);
    pub const DEST_TARGET_REG_CFG_PACK_SEC3_Offset: ConfigField = ConfigField::new(183, 0, 0xfff);
    pub const DEST_TARGET_REG_CFG_PACK_SEC3_ZOffset: ConfigField =
        ConfigField::new(183, 12, 0x3f000);
    pub const INT_DESCALE_VALUES_SEC0_Value: ConfigField = ConfigField::new(187, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC10_Value: ConfigField = ConfigField::new(197, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC11_Value: ConfigField = ConfigField::new(198, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC12_Value: ConfigField = ConfigField::new(199, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC13_Value: ConfigField = ConfigField::new(200, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC14_Value: ConfigField = ConfigField::new(201, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC15_Value: ConfigField = ConfigField::new(202, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC1_Value: ConfigField = ConfigField::new(188, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC2_Value: ConfigField = ConfigField::new(189, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC3_Value: ConfigField = ConfigField::new(190, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC4_Value: ConfigField = ConfigField::new(191, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC5_Value: ConfigField = ConfigField::new(192, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC6_Value: ConfigField = ConfigField::new(193, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC7_Value: ConfigField = ConfigField::new(194, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC8_Value: ConfigField = ConfigField::new(195, 0, 0xffffffff);
    pub const INT_DESCALE_VALUES_SEC9_Value: ConfigField = ConfigField::new(196, 0, 0xffffffff);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_bit_section_start_addr: ConfigField =
        ConfigField::new(218, 0, 0x1ffff);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_chk: ConfigField =
        ConfigField::new(218, 17, 0x20000);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_offset: ConfigField =
        ConfigField::new(219, 0, 0xffffff);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_End_Addr: ConfigField = ConfigField::new(213, 0, 0x1ffff);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Search_Enable: ConfigField = ConfigField::new(212, 0, 0x1);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Start_Addr: ConfigField = ConfigField::new(212, 1, 0x3fffe);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_high: ConfigField =
        ConfigField::new(215, 0, 0xffffffff);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_low: ConfigField =
        ConfigField::new(214, 0, 0xffffffff);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Tag_Width: ConfigField = ConfigField::new(216, 0, 0x3);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Tag_alloc: ConfigField =
        ConfigField::new(219, 26, 0x4000000);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv: ConfigField = ConfigField::new(219, 24, 0x1000000);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv_all: ConfigField =
        ConfigField::new(219, 25, 0x2000000);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_end_addr: ConfigField =
        ConfigField::new(217, 0, 0x1ffff);
    pub const L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_start_addr: ConfigField =
        ConfigField::new(216, 2, 0x7fffc);
    pub const NOC_RISC_END_PC_PC: ConfigField = ConfigField::new(207, 0, 0xffffffff);
    pub const PRNG_SEED_Seed_Val: ConfigField = ConfigField::new(186, 0, 0xffffffff);
    pub const RISCV_IC_INVALIDATE_InvalidateAll: ConfigField = ConfigField::new(185, 0, 0x1f);
    pub const RISC_PREFETCH_CTRL_Enable_Brisc: ConfigField = ConfigField::new(208, 3, 0x8);
    pub const RISC_PREFETCH_CTRL_Enable_NocRisc: ConfigField = ConfigField::new(208, 4, 0x10);
    pub const RISC_PREFETCH_CTRL_Enable_Trisc: ConfigField = ConfigField::new(208, 0, 0x7);
    pub const RISC_PREFETCH_CTRL_Max_Req_Count: ConfigField = ConfigField::new(208, 5, 0x1fe0);
    pub const SCRATCH_SEC0_val: ConfigField = ConfigField::new(209, 0, 0xffffffff);
    pub const SCRATCH_SEC1_val: ConfigField = ConfigField::new(210, 0, 0xffffffff);
    pub const SCRATCH_SEC2_val: ConfigField = ConfigField::new(211, 0, 0xffffffff);
    pub const SRC_ACCESS_CFG_disable_contig_srca_dvalid_phase: ConfigField =
        ConfigField::new(221, 2, 0x4);
    pub const SRC_ACCESS_CFG_disable_contig_srcb_dvalid_phase: ConfigField =
        ConfigField::new(221, 3, 0x8);
    pub const SRC_ACCESS_CFG_math_view_srca_as_one_bank: ConfigField =
        ConfigField::new(221, 0, 0x1);
    pub const SRC_ACCESS_CFG_math_view_srcb_as_one_bank: ConfigField =
        ConfigField::new(221, 1, 0x2);
    pub const TRISC_END_PC_SEC0_PC: ConfigField = ConfigField::new(203, 0, 0xffffffff);
    pub const TRISC_END_PC_SEC1_PC: ConfigField = ConfigField::new(204, 0, 0xffffffff);
    pub const TRISC_END_PC_SEC2_PC: ConfigField = ConfigField::new(205, 0, 0xffffffff);
}

/// `// Registers for PACK0` — 175 fields, indexing `Config` (write with `WRCFG`).
pub mod pack0 {
    use super::*;

    /// First word index of this section, from `PACK0_CFGREG_BASE_ADDR32`.
    pub const CFGREG_BASE: u16 = 12;

    pub const PACK_CONCAT_MASK_SEC0_pack_concat_mask: ConfigField = ConfigField::new(32, 0, 0xffff);
    pub const PACK_CONCAT_MASK_SEC1_pack_concat_mask: ConfigField = ConfigField::new(33, 0, 0xffff);
    pub const PACK_CONCAT_MASK_SEC2_pack_concat_mask: ConfigField = ConfigField::new(34, 0, 0xffff);
    pub const PACK_CONCAT_MASK_SEC3_pack_concat_mask: ConfigField = ConfigField::new(35, 0, 0xffff);
    pub const PACK_COUNTERS_SEC0_auto_ctxt_inc_xys_cnt: ConfigField =
        ConfigField::new(28, 24, 0xff000000);
    pub const PACK_COUNTERS_SEC0_pack_per_xy_plane: ConfigField = ConfigField::new(28, 0, 0xff);
    pub const PACK_COUNTERS_SEC0_pack_reads_per_xy_plane: ConfigField =
        ConfigField::new(28, 8, 0xff00);
    pub const PACK_COUNTERS_SEC0_pack_xys_per_tile: ConfigField =
        ConfigField::new(28, 16, 0x7f0000);
    pub const PACK_COUNTERS_SEC0_pack_yz_transposed: ConfigField =
        ConfigField::new(28, 23, 0x800000);
    pub const PACK_COUNTERS_SEC1_auto_ctxt_inc_xys_cnt: ConfigField =
        ConfigField::new(29, 24, 0xff000000);
    pub const PACK_COUNTERS_SEC1_pack_per_xy_plane: ConfigField = ConfigField::new(29, 0, 0xff);
    pub const PACK_COUNTERS_SEC1_pack_reads_per_xy_plane: ConfigField =
        ConfigField::new(29, 8, 0xff00);
    pub const PACK_COUNTERS_SEC1_pack_xys_per_tile: ConfigField =
        ConfigField::new(29, 16, 0x7f0000);
    pub const PACK_COUNTERS_SEC1_pack_yz_transposed: ConfigField =
        ConfigField::new(29, 23, 0x800000);
    pub const PACK_COUNTERS_SEC2_auto_ctxt_inc_xys_cnt: ConfigField =
        ConfigField::new(30, 24, 0xff000000);
    pub const PACK_COUNTERS_SEC2_pack_per_xy_plane: ConfigField = ConfigField::new(30, 0, 0xff);
    pub const PACK_COUNTERS_SEC2_pack_reads_per_xy_plane: ConfigField =
        ConfigField::new(30, 8, 0xff00);
    pub const PACK_COUNTERS_SEC2_pack_xys_per_tile: ConfigField =
        ConfigField::new(30, 16, 0x7f0000);
    pub const PACK_COUNTERS_SEC2_pack_yz_transposed: ConfigField =
        ConfigField::new(30, 23, 0x800000);
    pub const PACK_COUNTERS_SEC3_auto_ctxt_inc_xys_cnt: ConfigField =
        ConfigField::new(31, 24, 0xff000000);
    pub const PACK_COUNTERS_SEC3_pack_per_xy_plane: ConfigField = ConfigField::new(31, 0, 0xff);
    pub const PACK_COUNTERS_SEC3_pack_reads_per_xy_plane: ConfigField =
        ConfigField::new(31, 8, 0xff00);
    pub const PACK_COUNTERS_SEC3_pack_xys_per_tile: ConfigField =
        ConfigField::new(31, 16, 0x7f0000);
    pub const PACK_COUNTERS_SEC3_pack_yz_transposed: ConfigField =
        ConfigField::new(31, 23, 0x800000);
    pub const PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain: ConfigField =
        ConfigField::new(40, 0, 0x1);
    pub const PCK0_ADDR_BASE_REG_0_Base: ConfigField = ConfigField::new(16, 0, 0x3ffff);
    pub const PCK0_ADDR_BASE_REG_1_Base: ConfigField = ConfigField::new(17, 0, 0x3ffff);
    pub const PCK0_ADDR_CTRL_XY_REG_0_Xstride: ConfigField = ConfigField::new(12, 0, 0xffff);
    pub const PCK0_ADDR_CTRL_XY_REG_0_Ystride: ConfigField = ConfigField::new(12, 16, 0xffff0000);
    pub const PCK0_ADDR_CTRL_XY_REG_1_Xstride: ConfigField = ConfigField::new(14, 0, 0xffff);
    pub const PCK0_ADDR_CTRL_XY_REG_1_Ystride: ConfigField = ConfigField::new(14, 16, 0xffff0000);
    pub const PCK0_ADDR_CTRL_ZW_REG_0_Wstride: ConfigField = ConfigField::new(13, 16, 0xffff0000);
    pub const PCK0_ADDR_CTRL_ZW_REG_0_Zstride: ConfigField = ConfigField::new(13, 0, 0xffff);
    pub const PCK0_ADDR_CTRL_ZW_REG_1_Wstride: ConfigField = ConfigField::new(15, 16, 0xffff0000);
    pub const PCK0_ADDR_CTRL_ZW_REG_1_Zstride: ConfigField = ConfigField::new(15, 0, 0xffff);
    pub const PCK_DEST_RD_CTRL_Read_32b_data: ConfigField = ConfigField::new(18, 0, 0x1);
    pub const PCK_DEST_RD_CTRL_Read_int8: ConfigField = ConfigField::new(18, 2, 0x4);
    pub const PCK_DEST_RD_CTRL_Read_unsigned: ConfigField = ConfigField::new(18, 1, 0x2);
    pub const PCK_DEST_RD_CTRL_Round_10b_mant: ConfigField = ConfigField::new(18, 3, 0x8);
    pub const PCK_EDGE_MODE_mode: ConfigField = ConfigField::new(24, 16, 0x10000);
    pub const PCK_EDGE_OFFSET_SEC0_mask: ConfigField = ConfigField::new(24, 0, 0xffff);
    pub const PCK_EDGE_OFFSET_SEC1_mask: ConfigField = ConfigField::new(25, 0, 0xffff);
    pub const PCK_EDGE_OFFSET_SEC2_mask: ConfigField = ConfigField::new(26, 0, 0xffff);
    pub const PCK_EDGE_OFFSET_SEC3_mask: ConfigField = ConfigField::new(27, 0, 0xffff);
    pub const PCK_EDGE_TILE_FACE_SET_SELECT_enable: ConfigField = ConfigField::new(19, 8, 0x100);
    pub const PCK_EDGE_TILE_FACE_SET_SELECT_select: ConfigField = ConfigField::new(19, 0, 0xff);
    pub const PCK_EDGE_TILE_ROW_SET_SELECT_select: ConfigField =
        ConfigField::new(24, 17, 0x1fe0000);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_0: ConfigField =
        ConfigField::new(36, 0, 0x3);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_1: ConfigField =
        ConfigField::new(36, 2, 0xc);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_10: ConfigField =
        ConfigField::new(36, 20, 0x300000);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_11: ConfigField =
        ConfigField::new(36, 22, 0xc00000);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_12: ConfigField =
        ConfigField::new(36, 24, 0x3000000);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_13: ConfigField =
        ConfigField::new(36, 26, 0xc000000);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_14: ConfigField =
        ConfigField::new(36, 28, 0x30000000);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_15: ConfigField =
        ConfigField::new(36, 30, 0xc0000000);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_2: ConfigField =
        ConfigField::new(36, 4, 0x30);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_3: ConfigField =
        ConfigField::new(36, 6, 0xc0);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_4: ConfigField =
        ConfigField::new(36, 8, 0x300);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_5: ConfigField =
        ConfigField::new(36, 10, 0xc00);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_6: ConfigField =
        ConfigField::new(36, 12, 0x3000);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_7: ConfigField =
        ConfigField::new(36, 14, 0xc000);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_8: ConfigField =
        ConfigField::new(36, 16, 0x30000);
    pub const TILE_FACE_SET_MAPPING_0_face_set_mapping_9: ConfigField =
        ConfigField::new(36, 18, 0xc0000);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_0: ConfigField =
        ConfigField::new(37, 0, 0x3);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_1: ConfigField =
        ConfigField::new(37, 2, 0xc);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_10: ConfigField =
        ConfigField::new(37, 20, 0x300000);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_11: ConfigField =
        ConfigField::new(37, 22, 0xc00000);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_12: ConfigField =
        ConfigField::new(37, 24, 0x3000000);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_13: ConfigField =
        ConfigField::new(37, 26, 0xc000000);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_14: ConfigField =
        ConfigField::new(37, 28, 0x30000000);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_15: ConfigField =
        ConfigField::new(37, 30, 0xc0000000);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_2: ConfigField =
        ConfigField::new(37, 4, 0x30);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_3: ConfigField =
        ConfigField::new(37, 6, 0xc0);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_4: ConfigField =
        ConfigField::new(37, 8, 0x300);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_5: ConfigField =
        ConfigField::new(37, 10, 0xc00);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_6: ConfigField =
        ConfigField::new(37, 12, 0x3000);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_7: ConfigField =
        ConfigField::new(37, 14, 0xc000);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_8: ConfigField =
        ConfigField::new(37, 16, 0x30000);
    pub const TILE_FACE_SET_MAPPING_1_face_set_mapping_9: ConfigField =
        ConfigField::new(37, 18, 0xc0000);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_0: ConfigField =
        ConfigField::new(38, 0, 0x3);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_1: ConfigField =
        ConfigField::new(38, 2, 0xc);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_10: ConfigField =
        ConfigField::new(38, 20, 0x300000);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_11: ConfigField =
        ConfigField::new(38, 22, 0xc00000);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_12: ConfigField =
        ConfigField::new(38, 24, 0x3000000);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_13: ConfigField =
        ConfigField::new(38, 26, 0xc000000);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_14: ConfigField =
        ConfigField::new(38, 28, 0x30000000);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_15: ConfigField =
        ConfigField::new(38, 30, 0xc0000000);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_2: ConfigField =
        ConfigField::new(38, 4, 0x30);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_3: ConfigField =
        ConfigField::new(38, 6, 0xc0);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_4: ConfigField =
        ConfigField::new(38, 8, 0x300);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_5: ConfigField =
        ConfigField::new(38, 10, 0xc00);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_6: ConfigField =
        ConfigField::new(38, 12, 0x3000);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_7: ConfigField =
        ConfigField::new(38, 14, 0xc000);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_8: ConfigField =
        ConfigField::new(38, 16, 0x30000);
    pub const TILE_FACE_SET_MAPPING_2_face_set_mapping_9: ConfigField =
        ConfigField::new(38, 18, 0xc0000);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_0: ConfigField =
        ConfigField::new(39, 0, 0x3);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_1: ConfigField =
        ConfigField::new(39, 2, 0xc);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_10: ConfigField =
        ConfigField::new(39, 20, 0x300000);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_11: ConfigField =
        ConfigField::new(39, 22, 0xc00000);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_12: ConfigField =
        ConfigField::new(39, 24, 0x3000000);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_13: ConfigField =
        ConfigField::new(39, 26, 0xc000000);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_14: ConfigField =
        ConfigField::new(39, 28, 0x30000000);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_15: ConfigField =
        ConfigField::new(39, 30, 0xc0000000);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_2: ConfigField =
        ConfigField::new(39, 4, 0x30);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_3: ConfigField =
        ConfigField::new(39, 6, 0xc0);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_4: ConfigField =
        ConfigField::new(39, 8, 0x300);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_5: ConfigField =
        ConfigField::new(39, 10, 0xc00);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_6: ConfigField =
        ConfigField::new(39, 12, 0x3000);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_7: ConfigField =
        ConfigField::new(39, 14, 0xc000);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_8: ConfigField =
        ConfigField::new(39, 16, 0x30000);
    pub const TILE_FACE_SET_MAPPING_3_face_set_mapping_9: ConfigField =
        ConfigField::new(39, 18, 0xc0000);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_0: ConfigField = ConfigField::new(20, 0, 0x3);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_1: ConfigField = ConfigField::new(20, 2, 0xc);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_10: ConfigField =
        ConfigField::new(20, 20, 0x300000);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_11: ConfigField =
        ConfigField::new(20, 22, 0xc00000);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_12: ConfigField =
        ConfigField::new(20, 24, 0x3000000);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_13: ConfigField =
        ConfigField::new(20, 26, 0xc000000);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_14: ConfigField =
        ConfigField::new(20, 28, 0x30000000);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_15: ConfigField =
        ConfigField::new(20, 30, 0xc0000000);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_2: ConfigField = ConfigField::new(20, 4, 0x30);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_3: ConfigField = ConfigField::new(20, 6, 0xc0);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_4: ConfigField =
        ConfigField::new(20, 8, 0x300);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_5: ConfigField =
        ConfigField::new(20, 10, 0xc00);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_6: ConfigField =
        ConfigField::new(20, 12, 0x3000);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_7: ConfigField =
        ConfigField::new(20, 14, 0xc000);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_8: ConfigField =
        ConfigField::new(20, 16, 0x30000);
    pub const TILE_ROW_SET_MAPPING_0_row_set_mapping_9: ConfigField =
        ConfigField::new(20, 18, 0xc0000);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_0: ConfigField = ConfigField::new(21, 0, 0x3);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_1: ConfigField = ConfigField::new(21, 2, 0xc);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_10: ConfigField =
        ConfigField::new(21, 20, 0x300000);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_11: ConfigField =
        ConfigField::new(21, 22, 0xc00000);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_12: ConfigField =
        ConfigField::new(21, 24, 0x3000000);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_13: ConfigField =
        ConfigField::new(21, 26, 0xc000000);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_14: ConfigField =
        ConfigField::new(21, 28, 0x30000000);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_15: ConfigField =
        ConfigField::new(21, 30, 0xc0000000);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_2: ConfigField = ConfigField::new(21, 4, 0x30);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_3: ConfigField = ConfigField::new(21, 6, 0xc0);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_4: ConfigField =
        ConfigField::new(21, 8, 0x300);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_5: ConfigField =
        ConfigField::new(21, 10, 0xc00);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_6: ConfigField =
        ConfigField::new(21, 12, 0x3000);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_7: ConfigField =
        ConfigField::new(21, 14, 0xc000);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_8: ConfigField =
        ConfigField::new(21, 16, 0x30000);
    pub const TILE_ROW_SET_MAPPING_1_row_set_mapping_9: ConfigField =
        ConfigField::new(21, 18, 0xc0000);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_0: ConfigField = ConfigField::new(22, 0, 0x3);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_1: ConfigField = ConfigField::new(22, 2, 0xc);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_10: ConfigField =
        ConfigField::new(22, 20, 0x300000);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_11: ConfigField =
        ConfigField::new(22, 22, 0xc00000);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_12: ConfigField =
        ConfigField::new(22, 24, 0x3000000);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_13: ConfigField =
        ConfigField::new(22, 26, 0xc000000);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_14: ConfigField =
        ConfigField::new(22, 28, 0x30000000);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_15: ConfigField =
        ConfigField::new(22, 30, 0xc0000000);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_2: ConfigField = ConfigField::new(22, 4, 0x30);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_3: ConfigField = ConfigField::new(22, 6, 0xc0);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_4: ConfigField =
        ConfigField::new(22, 8, 0x300);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_5: ConfigField =
        ConfigField::new(22, 10, 0xc00);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_6: ConfigField =
        ConfigField::new(22, 12, 0x3000);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_7: ConfigField =
        ConfigField::new(22, 14, 0xc000);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_8: ConfigField =
        ConfigField::new(22, 16, 0x30000);
    pub const TILE_ROW_SET_MAPPING_2_row_set_mapping_9: ConfigField =
        ConfigField::new(22, 18, 0xc0000);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_0: ConfigField = ConfigField::new(23, 0, 0x3);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_1: ConfigField = ConfigField::new(23, 2, 0xc);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_10: ConfigField =
        ConfigField::new(23, 20, 0x300000);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_11: ConfigField =
        ConfigField::new(23, 22, 0xc00000);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_12: ConfigField =
        ConfigField::new(23, 24, 0x3000000);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_13: ConfigField =
        ConfigField::new(23, 26, 0xc000000);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_14: ConfigField =
        ConfigField::new(23, 28, 0x30000000);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_15: ConfigField =
        ConfigField::new(23, 30, 0xc0000000);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_2: ConfigField = ConfigField::new(23, 4, 0x30);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_3: ConfigField = ConfigField::new(23, 6, 0xc0);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_4: ConfigField =
        ConfigField::new(23, 8, 0x300);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_5: ConfigField =
        ConfigField::new(23, 10, 0xc00);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_6: ConfigField =
        ConfigField::new(23, 12, 0x3000);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_7: ConfigField =
        ConfigField::new(23, 14, 0xc000);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_8: ConfigField =
        ConfigField::new(23, 16, 0x30000);
    pub const TILE_ROW_SET_MAPPING_3_row_set_mapping_9: ConfigField =
        ConfigField::new(23, 18, 0xc0000);
}

/// `// Registers for THCON` — 282 fields, indexing `Config` (write with `WRCFG`).
pub mod thcon {
    use super::*;

    /// First word index of this section, from `THCON_CFGREG_BASE_ADDR32`.
    pub const CFGREG_BASE: u16 = 64;

    /// `Config[64..68]` — a 128-bit aggregate, not a bitfield.
    pub const THCON_SEC0_REG0_TileDescriptor: ConfigSpan = ConfigSpan::new(64, 4);
    pub const THCON_SEC0_REG10_Packer_Reg_Wr_Addr: ConfigField = ConfigField::new(107, 0, 0xffffff);
    pub const THCON_SEC0_REG10_Unpack_fifo_size: ConfigField = ConfigField::new(105, 0, 0x1ffff);
    pub const THCON_SEC0_REG10_Unpack_limit_address: ConfigField =
        ConfigField::new(104, 0, 0x1ffff);
    pub const THCON_SEC0_REG10_Unpack_limit_address_en: ConfigField =
        ConfigField::new(105, 17, 0x20000);
    pub const THCON_SEC0_REG10_Unpacker_Reg_Wr_Addr: ConfigField =
        ConfigField::new(106, 0, 0xffffff);
    pub const THCON_SEC0_REG11_Metadata_cntxt_switch_unpacr_count: ConfigField =
        ConfigField::new(111, 8, 0xff00);
    pub const THCON_SEC0_REG11_Metadata_fifo_size: ConfigField =
        ConfigField::new(110, 0, 0xffffffff);
    pub const THCON_SEC0_REG11_Metadata_l1_addr: ConfigField = ConfigField::new(108, 0, 0xffffffff);
    pub const THCON_SEC0_REG11_Metadata_limit_addr: ConfigField =
        ConfigField::new(109, 0, 0xffffffff);
    pub const THCON_SEC0_REG11_Metadata_z_cntr_rst_unpacr_count: ConfigField =
        ConfigField::new(111, 0, 0xff);
    pub const THCON_SEC0_REG1_Add_l1_dest_addr_offset: ConfigField = ConfigField::new(70, 1, 0x2);
    pub const THCON_SEC0_REG1_Add_tile_header_size: ConfigField =
        ConfigField::new(70, 22, 0x400000);
    pub const THCON_SEC0_REG1_All_pack_disable_zero_compress_ovrd: ConfigField =
        ConfigField::new(70, 21, 0x200000);
    pub const THCON_SEC0_REG1_Auto_set_last_pacr_intf_sel: ConfigField =
        ConfigField::new(70, 13, 0x2000);
    pub const THCON_SEC0_REG1_Dis_shared_exp_assembler: ConfigField =
        ConfigField::new(70, 12, 0x1000);
    pub const THCON_SEC0_REG1_Disable_pack_zero_flags: ConfigField = ConfigField::new(70, 2, 0x4);
    pub const THCON_SEC0_REG1_Disable_zero_compress: ConfigField = ConfigField::new(70, 0, 0x1);
    pub const THCON_SEC0_REG1_Downsample_mask: ConfigField = ConfigField::new(71, 0, 0xffff);
    pub const THCON_SEC0_REG1_Downsample_rate: ConfigField = ConfigField::new(71, 16, 0x70000);
    pub const THCON_SEC0_REG1_Enable_out_fifo: ConfigField = ConfigField::new(70, 14, 0x4000);
    pub const THCON_SEC0_REG1_Exp_section_size: ConfigField = ConfigField::new(68, 16, 0xffff0000);
    pub const THCON_SEC0_REG1_Exp_threshold: ConfigField = ConfigField::new(71, 24, 0xff000000);
    pub const THCON_SEC0_REG1_Exp_threshold_en: ConfigField = ConfigField::new(71, 20, 0x100000);
    pub const THCON_SEC0_REG1_In_data_format: ConfigField = ConfigField::new(70, 8, 0xf00);
    pub const THCON_SEC0_REG1_L1_Dest_addr: ConfigField = ConfigField::new(69, 0, 0xffffffff);
    pub const THCON_SEC0_REG1_L1_source_addr: ConfigField = ConfigField::new(70, 24, 0xff000000);
    pub const THCON_SEC0_REG1_Out_data_format: ConfigField = ConfigField::new(70, 4, 0xf0);
    pub const THCON_SEC0_REG1_Pac_LF8_4b_exp: ConfigField = ConfigField::new(71, 23, 0x800000);
    pub const THCON_SEC0_REG1_Pack_L1_Acc: ConfigField = ConfigField::new(71, 19, 0x80000);
    pub const THCON_SEC0_REG1_Row_start_section_size: ConfigField = ConfigField::new(68, 0, 0xffff);
    pub const THCON_SEC0_REG1_Source_interface_selection: ConfigField =
        ConfigField::new(70, 16, 0x10000);
    pub const THCON_SEC0_REG1_Sub_l1_tile_header_size: ConfigField =
        ConfigField::new(70, 15, 0x8000);
    pub const THCON_SEC0_REG1_Unp_LF8_4b_exp: ConfigField = ConfigField::new(71, 22, 0x400000);
    pub const THCON_SEC0_REG1_ovrd_default_throttle_mode: ConfigField =
        ConfigField::new(70, 3, 0x8);
    pub const THCON_SEC0_REG1_pack_dis_y_pos_start_offset: ConfigField =
        ConfigField::new(70, 23, 0x800000);
    pub const THCON_SEC0_REG1_pack_start_intf_pos: ConfigField = ConfigField::new(70, 17, 0x1e0000);
    pub const THCON_SEC0_REG2_Context_count: ConfigField = ConfigField::new(72, 6, 0xc0);
    pub const THCON_SEC0_REG2_Context_count_non_log2: ConfigField = ConfigField::new(73, 9, 0xe00);
    pub const THCON_SEC0_REG2_Context_count_non_log2_en: ConfigField =
        ConfigField::new(73, 12, 0x1000);
    pub const THCON_SEC0_REG2_Disable_zero_compress_cntx0: ConfigField =
        ConfigField::new(73, 0, 0x1);
    pub const THCON_SEC0_REG2_Disable_zero_compress_cntx1: ConfigField =
        ConfigField::new(73, 1, 0x2);
    pub const THCON_SEC0_REG2_Disable_zero_compress_cntx2: ConfigField =
        ConfigField::new(73, 2, 0x4);
    pub const THCON_SEC0_REG2_Disable_zero_compress_cntx3: ConfigField =
        ConfigField::new(73, 3, 0x8);
    pub const THCON_SEC0_REG2_Disable_zero_compress_cntx4: ConfigField =
        ConfigField::new(73, 16, 0x10000);
    pub const THCON_SEC0_REG2_Disable_zero_compress_cntx5: ConfigField =
        ConfigField::new(73, 17, 0x20000);
    pub const THCON_SEC0_REG2_Disable_zero_compress_cntx6: ConfigField =
        ConfigField::new(73, 18, 0x40000);
    pub const THCON_SEC0_REG2_Disable_zero_compress_cntx7: ConfigField =
        ConfigField::new(73, 19, 0x80000);
    pub const THCON_SEC0_REG2_Force_shared_exp: ConfigField = ConfigField::new(73, 8, 0x100);
    pub const THCON_SEC0_REG2_Haloize_mode: ConfigField = ConfigField::new(72, 8, 0x100);
    pub const THCON_SEC0_REG2_Metadata_x_end: ConfigField = ConfigField::new(73, 24, 0xff000000);
    pub const THCON_SEC0_REG2_Out_data_format: ConfigField = ConfigField::new(72, 0, 0xf);
    pub const THCON_SEC0_REG2_Ovrd_data_format: ConfigField = ConfigField::new(72, 14, 0x4000);
    pub const THCON_SEC0_REG2_Shift_amount_cntx0: ConfigField = ConfigField::new(72, 16, 0xf0000);
    pub const THCON_SEC0_REG2_Shift_amount_cntx1: ConfigField = ConfigField::new(72, 20, 0xf00000);
    pub const THCON_SEC0_REG2_Shift_amount_cntx2: ConfigField = ConfigField::new(72, 24, 0xf000000);
    pub const THCON_SEC0_REG2_Shift_amount_cntx3: ConfigField =
        ConfigField::new(72, 28, 0xf0000000);
    pub const THCON_SEC0_REG2_Throttle_mode: ConfigField = ConfigField::new(72, 4, 0x30);
    pub const THCON_SEC0_REG2_Tileize_mode: ConfigField = ConfigField::new(72, 9, 0x200);
    pub const THCON_SEC0_REG2_Unpack_If_Sel: ConfigField = ConfigField::new(72, 11, 0x800);
    pub const THCON_SEC0_REG2_Unpack_Src_Reg_Set_Upd: ConfigField = ConfigField::new(72, 10, 0x400);
    pub const THCON_SEC0_REG2_Unpack_fifo_size: ConfigField = ConfigField::new(75, 0, 0x1ffff);
    pub const THCON_SEC0_REG2_Unpack_if_sel_cntx0: ConfigField = ConfigField::new(73, 4, 0x10);
    pub const THCON_SEC0_REG2_Unpack_if_sel_cntx1: ConfigField = ConfigField::new(73, 5, 0x20);
    pub const THCON_SEC0_REG2_Unpack_if_sel_cntx2: ConfigField = ConfigField::new(73, 6, 0x40);
    pub const THCON_SEC0_REG2_Unpack_if_sel_cntx3: ConfigField = ConfigField::new(73, 7, 0x80);
    pub const THCON_SEC0_REG2_Unpack_if_sel_cntx4: ConfigField = ConfigField::new(73, 20, 0x100000);
    pub const THCON_SEC0_REG2_Unpack_if_sel_cntx5: ConfigField = ConfigField::new(73, 21, 0x200000);
    pub const THCON_SEC0_REG2_Unpack_if_sel_cntx6: ConfigField = ConfigField::new(73, 22, 0x400000);
    pub const THCON_SEC0_REG2_Unpack_if_sel_cntx7: ConfigField = ConfigField::new(73, 23, 0x800000);
    pub const THCON_SEC0_REG2_Unpack_limit_address: ConfigField = ConfigField::new(74, 0, 0x1ffff);
    pub const THCON_SEC0_REG2_Upsample_and_interleave: ConfigField =
        ConfigField::new(72, 15, 0x8000);
    pub const THCON_SEC0_REG2_Upsample_rate: ConfigField = ConfigField::new(72, 12, 0x3000);
    pub const THCON_SEC0_REG3_Base_address: ConfigField = ConfigField::new(76, 0, 0xffffffff);
    pub const THCON_SEC0_REG3_Base_cntx1_address: ConfigField = ConfigField::new(77, 0, 0xffffffff);
    pub const THCON_SEC0_REG3_Base_cntx2_address: ConfigField = ConfigField::new(78, 0, 0xffffffff);
    pub const THCON_SEC0_REG3_Base_cntx3_address: ConfigField = ConfigField::new(79, 0, 0xffffffff);
    pub const THCON_SEC0_REG4_Base_cntx4_address: ConfigField = ConfigField::new(80, 0, 0xffffffff);
    pub const THCON_SEC0_REG4_Base_cntx5_address: ConfigField = ConfigField::new(81, 0, 0xffffffff);
    pub const THCON_SEC0_REG4_Base_cntx6_address: ConfigField = ConfigField::new(82, 0, 0xffffffff);
    pub const THCON_SEC0_REG4_Base_cntx7_address: ConfigField = ConfigField::new(83, 0, 0xffffffff);
    pub const THCON_SEC0_REG5_Dest_cntx0_address: ConfigField = ConfigField::new(84, 0, 0xffff);
    pub const THCON_SEC0_REG5_Dest_cntx1_address: ConfigField =
        ConfigField::new(84, 16, 0xffff0000);
    pub const THCON_SEC0_REG5_Dest_cntx2_address: ConfigField = ConfigField::new(85, 0, 0xffff);
    pub const THCON_SEC0_REG5_Dest_cntx3_address: ConfigField =
        ConfigField::new(85, 16, 0xffff0000);
    pub const THCON_SEC0_REG5_Tile_x_dim_cntx0: ConfigField = ConfigField::new(86, 0, 0xffff);
    pub const THCON_SEC0_REG5_Tile_x_dim_cntx1: ConfigField = ConfigField::new(86, 16, 0xffff0000);
    pub const THCON_SEC0_REG5_Tile_x_dim_cntx2: ConfigField = ConfigField::new(87, 0, 0xffff);
    pub const THCON_SEC0_REG5_Tile_x_dim_cntx3: ConfigField = ConfigField::new(87, 16, 0xffff0000);
    pub const THCON_SEC0_REG6_Buffer_size: ConfigField = ConfigField::new(90, 0, 0x3fffffff);
    pub const THCON_SEC0_REG6_Destination_address: ConfigField =
        ConfigField::new(89, 0, 0xffffffff);
    pub const THCON_SEC0_REG6_Metadata_misc: ConfigField = ConfigField::new(91, 0, 0xffffffff);
    pub const THCON_SEC0_REG6_Source_address: ConfigField = ConfigField::new(88, 0, 0xffffffff);
    pub const THCON_SEC0_REG6_Transfer_direction: ConfigField =
        ConfigField::new(90, 30, 0xc0000000);
    pub const THCON_SEC0_REG7_Offset_address: ConfigField = ConfigField::new(92, 0, 0xffff);
    pub const THCON_SEC0_REG7_Offset_cntx1_address: ConfigField = ConfigField::new(93, 0, 0xffff);
    pub const THCON_SEC0_REG7_Offset_cntx2_address: ConfigField = ConfigField::new(94, 0, 0xffff);
    pub const THCON_SEC0_REG7_Offset_cntx3_address: ConfigField = ConfigField::new(95, 0, 0xffff);
    pub const THCON_SEC0_REG7_Unpack_data_format_cntx0: ConfigField =
        ConfigField::new(92, 16, 0xf0000);
    pub const THCON_SEC0_REG7_Unpack_data_format_cntx1: ConfigField =
        ConfigField::new(93, 16, 0xf0000);
    pub const THCON_SEC0_REG7_Unpack_data_format_cntx2: ConfigField =
        ConfigField::new(94, 16, 0xf0000);
    pub const THCON_SEC0_REG7_Unpack_data_format_cntx3: ConfigField =
        ConfigField::new(95, 16, 0xf0000);
    pub const THCON_SEC0_REG7_Unpack_data_format_cntx4: ConfigField =
        ConfigField::new(92, 24, 0xf000000);
    pub const THCON_SEC0_REG7_Unpack_data_format_cntx5: ConfigField =
        ConfigField::new(93, 24, 0xf000000);
    pub const THCON_SEC0_REG7_Unpack_data_format_cntx6: ConfigField =
        ConfigField::new(94, 24, 0xf000000);
    pub const THCON_SEC0_REG7_Unpack_data_format_cntx7: ConfigField =
        ConfigField::new(95, 24, 0xf000000);
    pub const THCON_SEC0_REG7_Unpack_out_data_format_cntx0: ConfigField =
        ConfigField::new(92, 20, 0xf00000);
    pub const THCON_SEC0_REG7_Unpack_out_data_format_cntx1: ConfigField =
        ConfigField::new(93, 20, 0xf00000);
    pub const THCON_SEC0_REG7_Unpack_out_data_format_cntx2: ConfigField =
        ConfigField::new(94, 20, 0xf00000);
    pub const THCON_SEC0_REG7_Unpack_out_data_format_cntx3: ConfigField =
        ConfigField::new(95, 20, 0xf00000);
    pub const THCON_SEC0_REG7_Unpack_out_data_format_cntx4: ConfigField =
        ConfigField::new(92, 28, 0xf0000000);
    pub const THCON_SEC0_REG7_Unpack_out_data_format_cntx5: ConfigField =
        ConfigField::new(93, 28, 0xf0000000);
    pub const THCON_SEC0_REG7_Unpack_out_data_format_cntx6: ConfigField =
        ConfigField::new(94, 28, 0xf0000000);
    pub const THCON_SEC0_REG7_Unpack_out_data_format_cntx7: ConfigField =
        ConfigField::new(95, 28, 0xf0000000);
    pub const THCON_SEC0_REG8_Add_l1_dest_addr_offset: ConfigField = ConfigField::new(98, 1, 0x2);
    pub const THCON_SEC0_REG8_Add_tile_header_size: ConfigField = ConfigField::new(98, 17, 0x20000);
    pub const THCON_SEC0_REG8_Auto_set_last_pacr_intf_sel: ConfigField =
        ConfigField::new(98, 13, 0x2000);
    pub const THCON_SEC0_REG8_Dis_shared_exp_assembler: ConfigField =
        ConfigField::new(98, 12, 0x1000);
    pub const THCON_SEC0_REG8_Disable_pack_zero_flags: ConfigField = ConfigField::new(98, 2, 0x4);
    pub const THCON_SEC0_REG8_Disable_zero_compress: ConfigField = ConfigField::new(98, 0, 0x1);
    pub const THCON_SEC0_REG8_Downsample_mask: ConfigField = ConfigField::new(99, 0, 0xffff);
    pub const THCON_SEC0_REG8_Downsample_rate: ConfigField = ConfigField::new(99, 16, 0x70000);
    pub const THCON_SEC0_REG8_Enable_out_fifo: ConfigField = ConfigField::new(98, 14, 0x4000);
    pub const THCON_SEC0_REG8_Exp_section_size: ConfigField = ConfigField::new(96, 16, 0xffff0000);
    pub const THCON_SEC0_REG8_Exp_threshold: ConfigField = ConfigField::new(99, 24, 0xff000000);
    pub const THCON_SEC0_REG8_Exp_threshold_en: ConfigField = ConfigField::new(99, 20, 0x100000);
    pub const THCON_SEC0_REG8_In_data_format: ConfigField = ConfigField::new(98, 8, 0xf00);
    pub const THCON_SEC0_REG8_L1_Dest_addr: ConfigField = ConfigField::new(97, 0, 0xffffffff);
    pub const THCON_SEC0_REG8_L1_source_addr: ConfigField = ConfigField::new(98, 24, 0xff000000);
    pub const THCON_SEC0_REG8_Out_data_format: ConfigField = ConfigField::new(98, 4, 0xf0);
    pub const THCON_SEC0_REG8_Pack_L1_Acc: ConfigField = ConfigField::new(99, 19, 0x80000);
    pub const THCON_SEC0_REG8_Row_start_section_size: ConfigField = ConfigField::new(96, 0, 0xffff);
    pub const THCON_SEC0_REG8_Source_interface_selection: ConfigField =
        ConfigField::new(98, 16, 0x10000);
    pub const THCON_SEC0_REG8_Sub_l1_tile_header_size: ConfigField =
        ConfigField::new(98, 15, 0x8000);
    pub const THCON_SEC0_REG8_Unused1: ConfigField = ConfigField::new(98, 3, 0x8);
    pub const THCON_SEC0_REG8_pack_dis_y_pos_start_offset: ConfigField =
        ConfigField::new(98, 18, 0x40000);
    pub const THCON_SEC0_REG8_unpack_tile_offset: ConfigField = ConfigField::new(98, 19, 0xf80000);
    pub const THCON_SEC0_REG9_Pack_0_2_fifo_size: ConfigField = ConfigField::new(101, 0, 0x1ffff);
    pub const THCON_SEC0_REG9_Pack_0_2_limit_address: ConfigField =
        ConfigField::new(100, 0, 0x1ffff);
    pub const THCON_SEC0_REG9_Pack_1_3_fifo_size: ConfigField = ConfigField::new(103, 0, 0x1ffff);
    pub const THCON_SEC0_REG9_Pack_1_3_limit_address: ConfigField =
        ConfigField::new(102, 0, 0x1ffff);
    /// `Config[112..116]` — a 128-bit aggregate, not a bitfield.
    pub const THCON_SEC1_REG0_TileDescriptor: ConfigSpan = ConfigSpan::new(112, 4);
    pub const THCON_SEC1_REG10_Packer_Reg_Wr_Addr: ConfigField = ConfigField::new(155, 0, 0xffffff);
    pub const THCON_SEC1_REG10_Unpack_fifo_size: ConfigField = ConfigField::new(153, 0, 0x1ffff);
    pub const THCON_SEC1_REG10_Unpack_limit_address: ConfigField =
        ConfigField::new(152, 0, 0x1ffff);
    pub const THCON_SEC1_REG10_Unpack_limit_address_en: ConfigField =
        ConfigField::new(153, 17, 0x20000);
    pub const THCON_SEC1_REG10_Unpacker_Reg_Wr_Addr: ConfigField =
        ConfigField::new(154, 0, 0xffffff);
    pub const THCON_SEC1_REG11_Metadata_cntxt_switch_unpacr_count: ConfigField =
        ConfigField::new(159, 8, 0xff00);
    pub const THCON_SEC1_REG11_Metadata_fifo_size: ConfigField =
        ConfigField::new(158, 0, 0xffffffff);
    pub const THCON_SEC1_REG11_Metadata_l1_addr: ConfigField = ConfigField::new(156, 0, 0xffffffff);
    pub const THCON_SEC1_REG11_Metadata_limit_addr: ConfigField =
        ConfigField::new(157, 0, 0xffffffff);
    pub const THCON_SEC1_REG11_Metadata_z_cntr_rst_unpacr_count: ConfigField =
        ConfigField::new(159, 0, 0xff);
    pub const THCON_SEC1_REG1_Add_l1_dest_addr_offset: ConfigField = ConfigField::new(118, 1, 0x2);
    pub const THCON_SEC1_REG1_Add_tile_header_size: ConfigField =
        ConfigField::new(118, 22, 0x400000);
    pub const THCON_SEC1_REG1_All_pack_disable_zero_compress_ovrd: ConfigField =
        ConfigField::new(118, 21, 0x200000);
    pub const THCON_SEC1_REG1_Auto_set_last_pacr_intf_sel: ConfigField =
        ConfigField::new(118, 13, 0x2000);
    pub const THCON_SEC1_REG1_Dis_shared_exp_assembler: ConfigField =
        ConfigField::new(118, 12, 0x1000);
    pub const THCON_SEC1_REG1_Disable_pack_zero_flags: ConfigField = ConfigField::new(118, 2, 0x4);
    pub const THCON_SEC1_REG1_Disable_zero_compress: ConfigField = ConfigField::new(118, 0, 0x1);
    pub const THCON_SEC1_REG1_Downsample_mask: ConfigField = ConfigField::new(119, 0, 0xffff);
    pub const THCON_SEC1_REG1_Downsample_rate: ConfigField = ConfigField::new(119, 16, 0x70000);
    pub const THCON_SEC1_REG1_Enable_out_fifo: ConfigField = ConfigField::new(118, 14, 0x4000);
    pub const THCON_SEC1_REG1_Exp_section_size: ConfigField = ConfigField::new(116, 16, 0xffff0000);
    pub const THCON_SEC1_REG1_Exp_threshold: ConfigField = ConfigField::new(119, 24, 0xff000000);
    pub const THCON_SEC1_REG1_Exp_threshold_en: ConfigField = ConfigField::new(119, 20, 0x100000);
    pub const THCON_SEC1_REG1_In_data_format: ConfigField = ConfigField::new(118, 8, 0xf00);
    pub const THCON_SEC1_REG1_L1_Dest_addr: ConfigField = ConfigField::new(117, 0, 0xffffffff);
    pub const THCON_SEC1_REG1_L1_source_addr: ConfigField = ConfigField::new(118, 24, 0xff000000);
    pub const THCON_SEC1_REG1_Out_data_format: ConfigField = ConfigField::new(118, 4, 0xf0);
    pub const THCON_SEC1_REG1_Pac_LF8_4b_exp: ConfigField = ConfigField::new(119, 23, 0x800000);
    pub const THCON_SEC1_REG1_Pack_L1_Acc: ConfigField = ConfigField::new(119, 19, 0x80000);
    pub const THCON_SEC1_REG1_Row_start_section_size: ConfigField =
        ConfigField::new(116, 0, 0xffff);
    pub const THCON_SEC1_REG1_Source_interface_selection: ConfigField =
        ConfigField::new(118, 16, 0x10000);
    pub const THCON_SEC1_REG1_Sub_l1_tile_header_size: ConfigField =
        ConfigField::new(118, 15, 0x8000);
    pub const THCON_SEC1_REG1_Unp_LF8_4b_exp: ConfigField = ConfigField::new(119, 22, 0x400000);
    pub const THCON_SEC1_REG1_ovrd_default_throttle_mode: ConfigField =
        ConfigField::new(118, 3, 0x8);
    pub const THCON_SEC1_REG1_pack_dis_y_pos_start_offset: ConfigField =
        ConfigField::new(118, 23, 0x800000);
    pub const THCON_SEC1_REG1_pack_start_intf_pos: ConfigField =
        ConfigField::new(118, 17, 0x1e0000);
    pub const THCON_SEC1_REG2_Context_count: ConfigField = ConfigField::new(120, 6, 0xc0);
    pub const THCON_SEC1_REG2_Context_count_non_log2: ConfigField = ConfigField::new(121, 9, 0xe00);
    pub const THCON_SEC1_REG2_Context_count_non_log2_en: ConfigField =
        ConfigField::new(121, 12, 0x1000);
    pub const THCON_SEC1_REG2_Disable_zero_compress_cntx0: ConfigField =
        ConfigField::new(121, 0, 0x1);
    pub const THCON_SEC1_REG2_Disable_zero_compress_cntx1: ConfigField =
        ConfigField::new(121, 1, 0x2);
    pub const THCON_SEC1_REG2_Disable_zero_compress_cntx2: ConfigField =
        ConfigField::new(121, 2, 0x4);
    pub const THCON_SEC1_REG2_Disable_zero_compress_cntx3: ConfigField =
        ConfigField::new(121, 3, 0x8);
    pub const THCON_SEC1_REG2_Disable_zero_compress_cntx4: ConfigField =
        ConfigField::new(121, 16, 0x10000);
    pub const THCON_SEC1_REG2_Disable_zero_compress_cntx5: ConfigField =
        ConfigField::new(121, 17, 0x20000);
    pub const THCON_SEC1_REG2_Disable_zero_compress_cntx6: ConfigField =
        ConfigField::new(121, 18, 0x40000);
    pub const THCON_SEC1_REG2_Disable_zero_compress_cntx7: ConfigField =
        ConfigField::new(121, 19, 0x80000);
    pub const THCON_SEC1_REG2_Force_shared_exp: ConfigField = ConfigField::new(121, 8, 0x100);
    pub const THCON_SEC1_REG2_Haloize_mode: ConfigField = ConfigField::new(120, 8, 0x100);
    pub const THCON_SEC1_REG2_Metadata_x_end: ConfigField = ConfigField::new(121, 24, 0xff000000);
    pub const THCON_SEC1_REG2_Out_data_format: ConfigField = ConfigField::new(120, 0, 0xf);
    pub const THCON_SEC1_REG2_Ovrd_data_format: ConfigField = ConfigField::new(120, 14, 0x4000);
    pub const THCON_SEC1_REG2_Shift_amount_cntx0: ConfigField = ConfigField::new(120, 16, 0xf0000);
    pub const THCON_SEC1_REG2_Shift_amount_cntx1: ConfigField = ConfigField::new(120, 20, 0xf00000);
    pub const THCON_SEC1_REG2_Shift_amount_cntx2: ConfigField =
        ConfigField::new(120, 24, 0xf000000);
    pub const THCON_SEC1_REG2_Shift_amount_cntx3: ConfigField =
        ConfigField::new(120, 28, 0xf0000000);
    pub const THCON_SEC1_REG2_Throttle_mode: ConfigField = ConfigField::new(120, 4, 0x30);
    pub const THCON_SEC1_REG2_Tileize_mode: ConfigField = ConfigField::new(120, 9, 0x200);
    pub const THCON_SEC1_REG2_Unpack_If_Sel: ConfigField = ConfigField::new(120, 11, 0x800);
    pub const THCON_SEC1_REG2_Unpack_Src_Reg_Set_Upd: ConfigField =
        ConfigField::new(120, 10, 0x400);
    pub const THCON_SEC1_REG2_Unpack_fifo_size: ConfigField = ConfigField::new(123, 0, 0x1ffff);
    pub const THCON_SEC1_REG2_Unpack_if_sel_cntx0: ConfigField = ConfigField::new(121, 4, 0x10);
    pub const THCON_SEC1_REG2_Unpack_if_sel_cntx1: ConfigField = ConfigField::new(121, 5, 0x20);
    pub const THCON_SEC1_REG2_Unpack_if_sel_cntx2: ConfigField = ConfigField::new(121, 6, 0x40);
    pub const THCON_SEC1_REG2_Unpack_if_sel_cntx3: ConfigField = ConfigField::new(121, 7, 0x80);
    pub const THCON_SEC1_REG2_Unpack_if_sel_cntx4: ConfigField =
        ConfigField::new(121, 20, 0x100000);
    pub const THCON_SEC1_REG2_Unpack_if_sel_cntx5: ConfigField =
        ConfigField::new(121, 21, 0x200000);
    pub const THCON_SEC1_REG2_Unpack_if_sel_cntx6: ConfigField =
        ConfigField::new(121, 22, 0x400000);
    pub const THCON_SEC1_REG2_Unpack_if_sel_cntx7: ConfigField =
        ConfigField::new(121, 23, 0x800000);
    pub const THCON_SEC1_REG2_Unpack_limit_address: ConfigField = ConfigField::new(122, 0, 0x1ffff);
    pub const THCON_SEC1_REG2_Upsample_and_interleave: ConfigField =
        ConfigField::new(120, 15, 0x8000);
    pub const THCON_SEC1_REG2_Upsample_rate: ConfigField = ConfigField::new(120, 12, 0x3000);
    pub const THCON_SEC1_REG3_Base_address: ConfigField = ConfigField::new(124, 0, 0xffffffff);
    pub const THCON_SEC1_REG3_Base_cntx1_address: ConfigField =
        ConfigField::new(125, 0, 0xffffffff);
    pub const THCON_SEC1_REG3_Base_cntx2_address: ConfigField =
        ConfigField::new(126, 0, 0xffffffff);
    pub const THCON_SEC1_REG3_Base_cntx3_address: ConfigField =
        ConfigField::new(127, 0, 0xffffffff);
    pub const THCON_SEC1_REG4_Base_cntx4_address: ConfigField =
        ConfigField::new(128, 0, 0xffffffff);
    pub const THCON_SEC1_REG4_Base_cntx5_address: ConfigField =
        ConfigField::new(129, 0, 0xffffffff);
    pub const THCON_SEC1_REG4_Base_cntx6_address: ConfigField =
        ConfigField::new(130, 0, 0xffffffff);
    pub const THCON_SEC1_REG4_Base_cntx7_address: ConfigField =
        ConfigField::new(131, 0, 0xffffffff);
    pub const THCON_SEC1_REG5_Dest_cntx0_address: ConfigField = ConfigField::new(132, 0, 0xffff);
    pub const THCON_SEC1_REG5_Dest_cntx1_address: ConfigField =
        ConfigField::new(132, 16, 0xffff0000);
    pub const THCON_SEC1_REG5_Dest_cntx2_address: ConfigField = ConfigField::new(133, 0, 0xffff);
    pub const THCON_SEC1_REG5_Dest_cntx3_address: ConfigField =
        ConfigField::new(133, 16, 0xffff0000);
    pub const THCON_SEC1_REG5_Tile_x_dim_cntx0: ConfigField = ConfigField::new(134, 0, 0xffff);
    pub const THCON_SEC1_REG5_Tile_x_dim_cntx1: ConfigField = ConfigField::new(134, 16, 0xffff0000);
    pub const THCON_SEC1_REG5_Tile_x_dim_cntx2: ConfigField = ConfigField::new(135, 0, 0xffff);
    pub const THCON_SEC1_REG5_Tile_x_dim_cntx3: ConfigField = ConfigField::new(135, 16, 0xffff0000);
    pub const THCON_SEC1_REG6_Buffer_size: ConfigField = ConfigField::new(138, 0, 0x3fffffff);
    pub const THCON_SEC1_REG6_Destination_address: ConfigField =
        ConfigField::new(137, 0, 0xffffffff);
    pub const THCON_SEC1_REG6_Metadata_misc: ConfigField = ConfigField::new(139, 0, 0xffffffff);
    pub const THCON_SEC1_REG6_Source_address: ConfigField = ConfigField::new(136, 0, 0xffffffff);
    pub const THCON_SEC1_REG6_Transfer_direction: ConfigField =
        ConfigField::new(138, 30, 0xc0000000);
    pub const THCON_SEC1_REG7_Offset_address: ConfigField = ConfigField::new(140, 0, 0xffff);
    pub const THCON_SEC1_REG7_Offset_cntx1_address: ConfigField = ConfigField::new(141, 0, 0xffff);
    pub const THCON_SEC1_REG7_Offset_cntx2_address: ConfigField = ConfigField::new(142, 0, 0xffff);
    pub const THCON_SEC1_REG7_Offset_cntx3_address: ConfigField = ConfigField::new(143, 0, 0xffff);
    pub const THCON_SEC1_REG7_Unpack_data_format_cntx0: ConfigField =
        ConfigField::new(140, 16, 0xf0000);
    pub const THCON_SEC1_REG7_Unpack_data_format_cntx1: ConfigField =
        ConfigField::new(141, 16, 0xf0000);
    pub const THCON_SEC1_REG7_Unpack_data_format_cntx2: ConfigField =
        ConfigField::new(142, 16, 0xf0000);
    pub const THCON_SEC1_REG7_Unpack_data_format_cntx3: ConfigField =
        ConfigField::new(143, 16, 0xf0000);
    pub const THCON_SEC1_REG7_Unpack_data_format_cntx4: ConfigField =
        ConfigField::new(140, 24, 0xf000000);
    pub const THCON_SEC1_REG7_Unpack_data_format_cntx5: ConfigField =
        ConfigField::new(141, 24, 0xf000000);
    pub const THCON_SEC1_REG7_Unpack_data_format_cntx6: ConfigField =
        ConfigField::new(142, 24, 0xf000000);
    pub const THCON_SEC1_REG7_Unpack_data_format_cntx7: ConfigField =
        ConfigField::new(143, 24, 0xf000000);
    pub const THCON_SEC1_REG7_Unpack_out_data_format_cntx0: ConfigField =
        ConfigField::new(140, 20, 0xf00000);
    pub const THCON_SEC1_REG7_Unpack_out_data_format_cntx1: ConfigField =
        ConfigField::new(141, 20, 0xf00000);
    pub const THCON_SEC1_REG7_Unpack_out_data_format_cntx2: ConfigField =
        ConfigField::new(142, 20, 0xf00000);
    pub const THCON_SEC1_REG7_Unpack_out_data_format_cntx3: ConfigField =
        ConfigField::new(143, 20, 0xf00000);
    pub const THCON_SEC1_REG7_Unpack_out_data_format_cntx4: ConfigField =
        ConfigField::new(140, 28, 0xf0000000);
    pub const THCON_SEC1_REG7_Unpack_out_data_format_cntx5: ConfigField =
        ConfigField::new(141, 28, 0xf0000000);
    pub const THCON_SEC1_REG7_Unpack_out_data_format_cntx6: ConfigField =
        ConfigField::new(142, 28, 0xf0000000);
    pub const THCON_SEC1_REG7_Unpack_out_data_format_cntx7: ConfigField =
        ConfigField::new(143, 28, 0xf0000000);
    pub const THCON_SEC1_REG8_Add_l1_dest_addr_offset: ConfigField = ConfigField::new(146, 1, 0x2);
    pub const THCON_SEC1_REG8_Add_tile_header_size: ConfigField =
        ConfigField::new(146, 17, 0x20000);
    pub const THCON_SEC1_REG8_Auto_set_last_pacr_intf_sel: ConfigField =
        ConfigField::new(146, 13, 0x2000);
    pub const THCON_SEC1_REG8_Dis_shared_exp_assembler: ConfigField =
        ConfigField::new(146, 12, 0x1000);
    pub const THCON_SEC1_REG8_Disable_pack_zero_flags: ConfigField = ConfigField::new(146, 2, 0x4);
    pub const THCON_SEC1_REG8_Disable_zero_compress: ConfigField = ConfigField::new(146, 0, 0x1);
    pub const THCON_SEC1_REG8_Downsample_mask: ConfigField = ConfigField::new(147, 0, 0xffff);
    pub const THCON_SEC1_REG8_Downsample_rate: ConfigField = ConfigField::new(147, 16, 0x70000);
    pub const THCON_SEC1_REG8_Enable_out_fifo: ConfigField = ConfigField::new(146, 14, 0x4000);
    pub const THCON_SEC1_REG8_Exp_section_size: ConfigField = ConfigField::new(144, 16, 0xffff0000);
    pub const THCON_SEC1_REG8_Exp_threshold: ConfigField = ConfigField::new(147, 24, 0xff000000);
    pub const THCON_SEC1_REG8_Exp_threshold_en: ConfigField = ConfigField::new(147, 20, 0x100000);
    pub const THCON_SEC1_REG8_In_data_format: ConfigField = ConfigField::new(146, 8, 0xf00);
    pub const THCON_SEC1_REG8_L1_Dest_addr: ConfigField = ConfigField::new(145, 0, 0xffffffff);
    pub const THCON_SEC1_REG8_L1_source_addr: ConfigField = ConfigField::new(146, 24, 0xff000000);
    pub const THCON_SEC1_REG8_Out_data_format: ConfigField = ConfigField::new(146, 4, 0xf0);
    pub const THCON_SEC1_REG8_Pack_L1_Acc: ConfigField = ConfigField::new(147, 19, 0x80000);
    pub const THCON_SEC1_REG8_Row_start_section_size: ConfigField =
        ConfigField::new(144, 0, 0xffff);
    pub const THCON_SEC1_REG8_Source_interface_selection: ConfigField =
        ConfigField::new(146, 16, 0x10000);
    pub const THCON_SEC1_REG8_Sub_l1_tile_header_size: ConfigField =
        ConfigField::new(146, 15, 0x8000);
    pub const THCON_SEC1_REG8_Unused1: ConfigField = ConfigField::new(146, 3, 0x8);
    pub const THCON_SEC1_REG8_pack_dis_y_pos_start_offset: ConfigField =
        ConfigField::new(146, 18, 0x40000);
    pub const THCON_SEC1_REG8_unpack_tile_offset: ConfigField = ConfigField::new(146, 19, 0xf80000);
    pub const THCON_SEC1_REG9_Pack_0_2_fifo_size: ConfigField = ConfigField::new(149, 0, 0x1ffff);
    pub const THCON_SEC1_REG9_Pack_0_2_limit_address: ConfigField =
        ConfigField::new(148, 0, 0x1ffff);
    pub const THCON_SEC1_REG9_Pack_1_3_fifo_size: ConfigField = ConfigField::new(151, 0, 0x1ffff);
    pub const THCON_SEC1_REG9_Pack_1_3_limit_address: ConfigField =
        ConfigField::new(150, 0, 0x1ffff);
}

/// `// Registers for THREAD` — 223 fields, indexing `ThreadConfig` (write with `SETC16`).
pub mod thread {
    use super::*;

    /// First word index of this section, from `THREAD_CFGREG_BASE_ADDR32`.
    pub const CFGREG_BASE: u16 = 0;

    pub const ADDR_MOD_AB2_SEC0_SrcAIncr: ThreadConfigField = ThreadConfigField::new(20, 0, 0x1);
    pub const ADDR_MOD_AB2_SEC0_SrcBIncr: ThreadConfigField = ThreadConfigField::new(20, 1, 0x2);
    pub const ADDR_MOD_AB2_SEC1_SrcAIncr: ThreadConfigField = ThreadConfigField::new(21, 0, 0x1);
    pub const ADDR_MOD_AB2_SEC1_SrcBIncr: ThreadConfigField = ThreadConfigField::new(21, 1, 0x2);
    pub const ADDR_MOD_AB2_SEC2_SrcAIncr: ThreadConfigField = ThreadConfigField::new(22, 0, 0x1);
    pub const ADDR_MOD_AB2_SEC2_SrcBIncr: ThreadConfigField = ThreadConfigField::new(22, 1, 0x2);
    pub const ADDR_MOD_AB2_SEC3_SrcAIncr: ThreadConfigField = ThreadConfigField::new(23, 0, 0x1);
    pub const ADDR_MOD_AB2_SEC3_SrcBIncr: ThreadConfigField = ThreadConfigField::new(23, 1, 0x2);
    pub const ADDR_MOD_AB2_SEC4_SrcAIncr: ThreadConfigField = ThreadConfigField::new(24, 0, 0x1);
    pub const ADDR_MOD_AB2_SEC4_SrcBIncr: ThreadConfigField = ThreadConfigField::new(24, 1, 0x2);
    pub const ADDR_MOD_AB2_SEC5_SrcAIncr: ThreadConfigField = ThreadConfigField::new(25, 0, 0x1);
    pub const ADDR_MOD_AB2_SEC5_SrcBIncr: ThreadConfigField = ThreadConfigField::new(25, 1, 0x2);
    pub const ADDR_MOD_AB2_SEC6_SrcAIncr: ThreadConfigField = ThreadConfigField::new(26, 0, 0x1);
    pub const ADDR_MOD_AB2_SEC6_SrcBIncr: ThreadConfigField = ThreadConfigField::new(26, 1, 0x2);
    pub const ADDR_MOD_AB2_SEC7_SrcAIncr: ThreadConfigField = ThreadConfigField::new(27, 0, 0x1);
    pub const ADDR_MOD_AB2_SEC7_SrcBIncr: ThreadConfigField = ThreadConfigField::new(27, 1, 0x2);
    pub const ADDR_MOD_AB_SEC0_SrcACR: ThreadConfigField = ThreadConfigField::new(12, 6, 0x40);
    pub const ADDR_MOD_AB_SEC0_SrcAClear: ThreadConfigField = ThreadConfigField::new(12, 7, 0x80);
    pub const ADDR_MOD_AB_SEC0_SrcAIncr: ThreadConfigField = ThreadConfigField::new(12, 0, 0x3f);
    pub const ADDR_MOD_AB_SEC0_SrcBCR: ThreadConfigField = ThreadConfigField::new(12, 14, 0x4000);
    pub const ADDR_MOD_AB_SEC0_SrcBClear: ThreadConfigField =
        ThreadConfigField::new(12, 15, 0x8000);
    pub const ADDR_MOD_AB_SEC0_SrcBIncr: ThreadConfigField = ThreadConfigField::new(12, 8, 0x3f00);
    pub const ADDR_MOD_AB_SEC1_SrcACR: ThreadConfigField = ThreadConfigField::new(13, 6, 0x40);
    pub const ADDR_MOD_AB_SEC1_SrcAClear: ThreadConfigField = ThreadConfigField::new(13, 7, 0x80);
    pub const ADDR_MOD_AB_SEC1_SrcAIncr: ThreadConfigField = ThreadConfigField::new(13, 0, 0x3f);
    pub const ADDR_MOD_AB_SEC1_SrcBCR: ThreadConfigField = ThreadConfigField::new(13, 14, 0x4000);
    pub const ADDR_MOD_AB_SEC1_SrcBClear: ThreadConfigField =
        ThreadConfigField::new(13, 15, 0x8000);
    pub const ADDR_MOD_AB_SEC1_SrcBIncr: ThreadConfigField = ThreadConfigField::new(13, 8, 0x3f00);
    pub const ADDR_MOD_AB_SEC2_SrcACR: ThreadConfigField = ThreadConfigField::new(14, 6, 0x40);
    pub const ADDR_MOD_AB_SEC2_SrcAClear: ThreadConfigField = ThreadConfigField::new(14, 7, 0x80);
    pub const ADDR_MOD_AB_SEC2_SrcAIncr: ThreadConfigField = ThreadConfigField::new(14, 0, 0x3f);
    pub const ADDR_MOD_AB_SEC2_SrcBCR: ThreadConfigField = ThreadConfigField::new(14, 14, 0x4000);
    pub const ADDR_MOD_AB_SEC2_SrcBClear: ThreadConfigField =
        ThreadConfigField::new(14, 15, 0x8000);
    pub const ADDR_MOD_AB_SEC2_SrcBIncr: ThreadConfigField = ThreadConfigField::new(14, 8, 0x3f00);
    pub const ADDR_MOD_AB_SEC3_SrcACR: ThreadConfigField = ThreadConfigField::new(15, 6, 0x40);
    pub const ADDR_MOD_AB_SEC3_SrcAClear: ThreadConfigField = ThreadConfigField::new(15, 7, 0x80);
    pub const ADDR_MOD_AB_SEC3_SrcAIncr: ThreadConfigField = ThreadConfigField::new(15, 0, 0x3f);
    pub const ADDR_MOD_AB_SEC3_SrcBCR: ThreadConfigField = ThreadConfigField::new(15, 14, 0x4000);
    pub const ADDR_MOD_AB_SEC3_SrcBClear: ThreadConfigField =
        ThreadConfigField::new(15, 15, 0x8000);
    pub const ADDR_MOD_AB_SEC3_SrcBIncr: ThreadConfigField = ThreadConfigField::new(15, 8, 0x3f00);
    pub const ADDR_MOD_AB_SEC4_SrcACR: ThreadConfigField = ThreadConfigField::new(16, 6, 0x40);
    pub const ADDR_MOD_AB_SEC4_SrcAClear: ThreadConfigField = ThreadConfigField::new(16, 7, 0x80);
    pub const ADDR_MOD_AB_SEC4_SrcAIncr: ThreadConfigField = ThreadConfigField::new(16, 0, 0x3f);
    pub const ADDR_MOD_AB_SEC4_SrcBCR: ThreadConfigField = ThreadConfigField::new(16, 14, 0x4000);
    pub const ADDR_MOD_AB_SEC4_SrcBClear: ThreadConfigField =
        ThreadConfigField::new(16, 15, 0x8000);
    pub const ADDR_MOD_AB_SEC4_SrcBIncr: ThreadConfigField = ThreadConfigField::new(16, 8, 0x3f00);
    pub const ADDR_MOD_AB_SEC5_SrcACR: ThreadConfigField = ThreadConfigField::new(17, 6, 0x40);
    pub const ADDR_MOD_AB_SEC5_SrcAClear: ThreadConfigField = ThreadConfigField::new(17, 7, 0x80);
    pub const ADDR_MOD_AB_SEC5_SrcAIncr: ThreadConfigField = ThreadConfigField::new(17, 0, 0x3f);
    pub const ADDR_MOD_AB_SEC5_SrcBCR: ThreadConfigField = ThreadConfigField::new(17, 14, 0x4000);
    pub const ADDR_MOD_AB_SEC5_SrcBClear: ThreadConfigField =
        ThreadConfigField::new(17, 15, 0x8000);
    pub const ADDR_MOD_AB_SEC5_SrcBIncr: ThreadConfigField = ThreadConfigField::new(17, 8, 0x3f00);
    pub const ADDR_MOD_AB_SEC6_SrcACR: ThreadConfigField = ThreadConfigField::new(18, 6, 0x40);
    pub const ADDR_MOD_AB_SEC6_SrcAClear: ThreadConfigField = ThreadConfigField::new(18, 7, 0x80);
    pub const ADDR_MOD_AB_SEC6_SrcAIncr: ThreadConfigField = ThreadConfigField::new(18, 0, 0x3f);
    pub const ADDR_MOD_AB_SEC6_SrcBCR: ThreadConfigField = ThreadConfigField::new(18, 14, 0x4000);
    pub const ADDR_MOD_AB_SEC6_SrcBClear: ThreadConfigField =
        ThreadConfigField::new(18, 15, 0x8000);
    pub const ADDR_MOD_AB_SEC6_SrcBIncr: ThreadConfigField = ThreadConfigField::new(18, 8, 0x3f00);
    pub const ADDR_MOD_AB_SEC7_SrcACR: ThreadConfigField = ThreadConfigField::new(19, 6, 0x40);
    pub const ADDR_MOD_AB_SEC7_SrcAClear: ThreadConfigField = ThreadConfigField::new(19, 7, 0x80);
    pub const ADDR_MOD_AB_SEC7_SrcAIncr: ThreadConfigField = ThreadConfigField::new(19, 0, 0x3f);
    pub const ADDR_MOD_AB_SEC7_SrcBCR: ThreadConfigField = ThreadConfigField::new(19, 14, 0x4000);
    pub const ADDR_MOD_AB_SEC7_SrcBClear: ThreadConfigField =
        ThreadConfigField::new(19, 15, 0x8000);
    pub const ADDR_MOD_AB_SEC7_SrcBIncr: ThreadConfigField = ThreadConfigField::new(19, 8, 0x3f00);
    pub const ADDR_MOD_BIAS_SEC0_BiasClear: ThreadConfigField = ThreadConfigField::new(47, 4, 0x10);
    pub const ADDR_MOD_BIAS_SEC0_BiasIncr: ThreadConfigField = ThreadConfigField::new(47, 0, 0xf);
    pub const ADDR_MOD_BIAS_SEC1_BiasClear: ThreadConfigField = ThreadConfigField::new(48, 4, 0x10);
    pub const ADDR_MOD_BIAS_SEC1_BiasIncr: ThreadConfigField = ThreadConfigField::new(48, 0, 0xf);
    pub const ADDR_MOD_BIAS_SEC2_BiasClear: ThreadConfigField = ThreadConfigField::new(49, 4, 0x10);
    pub const ADDR_MOD_BIAS_SEC2_BiasIncr: ThreadConfigField = ThreadConfigField::new(49, 0, 0xf);
    pub const ADDR_MOD_BIAS_SEC3_BiasClear: ThreadConfigField = ThreadConfigField::new(50, 4, 0x10);
    pub const ADDR_MOD_BIAS_SEC3_BiasIncr: ThreadConfigField = ThreadConfigField::new(50, 0, 0xf);
    pub const ADDR_MOD_BIAS_SEC4_BiasClear: ThreadConfigField = ThreadConfigField::new(51, 4, 0x10);
    pub const ADDR_MOD_BIAS_SEC4_BiasIncr: ThreadConfigField = ThreadConfigField::new(51, 0, 0xf);
    pub const ADDR_MOD_BIAS_SEC5_BiasClear: ThreadConfigField = ThreadConfigField::new(52, 4, 0x10);
    pub const ADDR_MOD_BIAS_SEC5_BiasIncr: ThreadConfigField = ThreadConfigField::new(52, 0, 0xf);
    pub const ADDR_MOD_BIAS_SEC6_BiasClear: ThreadConfigField = ThreadConfigField::new(53, 4, 0x10);
    pub const ADDR_MOD_BIAS_SEC6_BiasIncr: ThreadConfigField = ThreadConfigField::new(53, 0, 0xf);
    pub const ADDR_MOD_BIAS_SEC7_BiasClear: ThreadConfigField = ThreadConfigField::new(54, 4, 0x10);
    pub const ADDR_MOD_BIAS_SEC7_BiasIncr: ThreadConfigField = ThreadConfigField::new(54, 0, 0xf);
    pub const ADDR_MOD_DST_SEC0_DestCR: ThreadConfigField = ThreadConfigField::new(28, 10, 0x400);
    pub const ADDR_MOD_DST_SEC0_DestCToCR: ThreadConfigField =
        ThreadConfigField::new(28, 12, 0x1000);
    pub const ADDR_MOD_DST_SEC0_DestClear: ThreadConfigField =
        ThreadConfigField::new(28, 11, 0x800);
    pub const ADDR_MOD_DST_SEC0_DestIncr: ThreadConfigField = ThreadConfigField::new(28, 0, 0x3ff);
    pub const ADDR_MOD_DST_SEC0_FidelityClear: ThreadConfigField =
        ThreadConfigField::new(28, 15, 0x8000);
    pub const ADDR_MOD_DST_SEC0_FidelityIncr: ThreadConfigField =
        ThreadConfigField::new(28, 13, 0x6000);
    pub const ADDR_MOD_DST_SEC1_DestCR: ThreadConfigField = ThreadConfigField::new(29, 10, 0x400);
    pub const ADDR_MOD_DST_SEC1_DestCToCR: ThreadConfigField =
        ThreadConfigField::new(29, 12, 0x1000);
    pub const ADDR_MOD_DST_SEC1_DestClear: ThreadConfigField =
        ThreadConfigField::new(29, 11, 0x800);
    pub const ADDR_MOD_DST_SEC1_DestIncr: ThreadConfigField = ThreadConfigField::new(29, 0, 0x3ff);
    pub const ADDR_MOD_DST_SEC1_FidelityClear: ThreadConfigField =
        ThreadConfigField::new(29, 15, 0x8000);
    pub const ADDR_MOD_DST_SEC1_FidelityIncr: ThreadConfigField =
        ThreadConfigField::new(29, 13, 0x6000);
    pub const ADDR_MOD_DST_SEC2_DestCR: ThreadConfigField = ThreadConfigField::new(30, 10, 0x400);
    pub const ADDR_MOD_DST_SEC2_DestCToCR: ThreadConfigField =
        ThreadConfigField::new(30, 12, 0x1000);
    pub const ADDR_MOD_DST_SEC2_DestClear: ThreadConfigField =
        ThreadConfigField::new(30, 11, 0x800);
    pub const ADDR_MOD_DST_SEC2_DestIncr: ThreadConfigField = ThreadConfigField::new(30, 0, 0x3ff);
    pub const ADDR_MOD_DST_SEC2_FidelityClear: ThreadConfigField =
        ThreadConfigField::new(30, 15, 0x8000);
    pub const ADDR_MOD_DST_SEC2_FidelityIncr: ThreadConfigField =
        ThreadConfigField::new(30, 13, 0x6000);
    pub const ADDR_MOD_DST_SEC3_DestCR: ThreadConfigField = ThreadConfigField::new(31, 10, 0x400);
    pub const ADDR_MOD_DST_SEC3_DestCToCR: ThreadConfigField =
        ThreadConfigField::new(31, 12, 0x1000);
    pub const ADDR_MOD_DST_SEC3_DestClear: ThreadConfigField =
        ThreadConfigField::new(31, 11, 0x800);
    pub const ADDR_MOD_DST_SEC3_DestIncr: ThreadConfigField = ThreadConfigField::new(31, 0, 0x3ff);
    pub const ADDR_MOD_DST_SEC3_FidelityClear: ThreadConfigField =
        ThreadConfigField::new(31, 15, 0x8000);
    pub const ADDR_MOD_DST_SEC3_FidelityIncr: ThreadConfigField =
        ThreadConfigField::new(31, 13, 0x6000);
    pub const ADDR_MOD_DST_SEC4_DestCR: ThreadConfigField = ThreadConfigField::new(32, 10, 0x400);
    pub const ADDR_MOD_DST_SEC4_DestCToCR: ThreadConfigField =
        ThreadConfigField::new(32, 12, 0x1000);
    pub const ADDR_MOD_DST_SEC4_DestClear: ThreadConfigField =
        ThreadConfigField::new(32, 11, 0x800);
    pub const ADDR_MOD_DST_SEC4_DestIncr: ThreadConfigField = ThreadConfigField::new(32, 0, 0x3ff);
    pub const ADDR_MOD_DST_SEC4_FidelityClear: ThreadConfigField =
        ThreadConfigField::new(32, 15, 0x8000);
    pub const ADDR_MOD_DST_SEC4_FidelityIncr: ThreadConfigField =
        ThreadConfigField::new(32, 13, 0x6000);
    pub const ADDR_MOD_DST_SEC5_DestCR: ThreadConfigField = ThreadConfigField::new(33, 10, 0x400);
    pub const ADDR_MOD_DST_SEC5_DestCToCR: ThreadConfigField =
        ThreadConfigField::new(33, 12, 0x1000);
    pub const ADDR_MOD_DST_SEC5_DestClear: ThreadConfigField =
        ThreadConfigField::new(33, 11, 0x800);
    pub const ADDR_MOD_DST_SEC5_DestIncr: ThreadConfigField = ThreadConfigField::new(33, 0, 0x3ff);
    pub const ADDR_MOD_DST_SEC5_FidelityClear: ThreadConfigField =
        ThreadConfigField::new(33, 15, 0x8000);
    pub const ADDR_MOD_DST_SEC5_FidelityIncr: ThreadConfigField =
        ThreadConfigField::new(33, 13, 0x6000);
    pub const ADDR_MOD_DST_SEC6_DestCR: ThreadConfigField = ThreadConfigField::new(34, 10, 0x400);
    pub const ADDR_MOD_DST_SEC6_DestCToCR: ThreadConfigField =
        ThreadConfigField::new(34, 12, 0x1000);
    pub const ADDR_MOD_DST_SEC6_DestClear: ThreadConfigField =
        ThreadConfigField::new(34, 11, 0x800);
    pub const ADDR_MOD_DST_SEC6_DestIncr: ThreadConfigField = ThreadConfigField::new(34, 0, 0x3ff);
    pub const ADDR_MOD_DST_SEC6_FidelityClear: ThreadConfigField =
        ThreadConfigField::new(34, 15, 0x8000);
    pub const ADDR_MOD_DST_SEC6_FidelityIncr: ThreadConfigField =
        ThreadConfigField::new(34, 13, 0x6000);
    pub const ADDR_MOD_DST_SEC7_DestCR: ThreadConfigField = ThreadConfigField::new(35, 10, 0x400);
    pub const ADDR_MOD_DST_SEC7_DestCToCR: ThreadConfigField =
        ThreadConfigField::new(35, 12, 0x1000);
    pub const ADDR_MOD_DST_SEC7_DestClear: ThreadConfigField =
        ThreadConfigField::new(35, 11, 0x800);
    pub const ADDR_MOD_DST_SEC7_DestIncr: ThreadConfigField = ThreadConfigField::new(35, 0, 0x3ff);
    pub const ADDR_MOD_DST_SEC7_FidelityClear: ThreadConfigField =
        ThreadConfigField::new(35, 15, 0x8000);
    pub const ADDR_MOD_DST_SEC7_FidelityIncr: ThreadConfigField =
        ThreadConfigField::new(35, 13, 0x6000);
    pub const ADDR_MOD_PACK_SEC0_YdstCR: ThreadConfigField = ThreadConfigField::new(37, 10, 0x400);
    pub const ADDR_MOD_PACK_SEC0_YdstClear: ThreadConfigField =
        ThreadConfigField::new(37, 11, 0x800);
    pub const ADDR_MOD_PACK_SEC0_YdstIncr: ThreadConfigField = ThreadConfigField::new(37, 6, 0x3c0);
    pub const ADDR_MOD_PACK_SEC0_YsrcCR: ThreadConfigField = ThreadConfigField::new(37, 4, 0x10);
    pub const ADDR_MOD_PACK_SEC0_YsrcClear: ThreadConfigField = ThreadConfigField::new(37, 5, 0x20);
    pub const ADDR_MOD_PACK_SEC0_YsrcIncr: ThreadConfigField = ThreadConfigField::new(37, 0, 0xf);
    pub const ADDR_MOD_PACK_SEC0_ZdstClear: ThreadConfigField =
        ThreadConfigField::new(37, 15, 0x8000);
    pub const ADDR_MOD_PACK_SEC0_ZdstIncr: ThreadConfigField =
        ThreadConfigField::new(37, 14, 0x4000);
    pub const ADDR_MOD_PACK_SEC0_ZsrcClear: ThreadConfigField =
        ThreadConfigField::new(37, 13, 0x2000);
    pub const ADDR_MOD_PACK_SEC0_ZsrcIncr: ThreadConfigField =
        ThreadConfigField::new(37, 12, 0x1000);
    pub const ADDR_MOD_PACK_SEC1_YdstCR: ThreadConfigField = ThreadConfigField::new(38, 10, 0x400);
    pub const ADDR_MOD_PACK_SEC1_YdstClear: ThreadConfigField =
        ThreadConfigField::new(38, 11, 0x800);
    pub const ADDR_MOD_PACK_SEC1_YdstIncr: ThreadConfigField = ThreadConfigField::new(38, 6, 0x3c0);
    pub const ADDR_MOD_PACK_SEC1_YsrcCR: ThreadConfigField = ThreadConfigField::new(38, 4, 0x10);
    pub const ADDR_MOD_PACK_SEC1_YsrcClear: ThreadConfigField = ThreadConfigField::new(38, 5, 0x20);
    pub const ADDR_MOD_PACK_SEC1_YsrcIncr: ThreadConfigField = ThreadConfigField::new(38, 0, 0xf);
    pub const ADDR_MOD_PACK_SEC1_ZdstClear: ThreadConfigField =
        ThreadConfigField::new(38, 15, 0x8000);
    pub const ADDR_MOD_PACK_SEC1_ZdstIncr: ThreadConfigField =
        ThreadConfigField::new(38, 14, 0x4000);
    pub const ADDR_MOD_PACK_SEC1_ZsrcClear: ThreadConfigField =
        ThreadConfigField::new(38, 13, 0x2000);
    pub const ADDR_MOD_PACK_SEC1_ZsrcIncr: ThreadConfigField =
        ThreadConfigField::new(38, 12, 0x1000);
    pub const ADDR_MOD_PACK_SEC2_YdstCR: ThreadConfigField = ThreadConfigField::new(39, 10, 0x400);
    pub const ADDR_MOD_PACK_SEC2_YdstClear: ThreadConfigField =
        ThreadConfigField::new(39, 11, 0x800);
    pub const ADDR_MOD_PACK_SEC2_YdstIncr: ThreadConfigField = ThreadConfigField::new(39, 6, 0x3c0);
    pub const ADDR_MOD_PACK_SEC2_YsrcCR: ThreadConfigField = ThreadConfigField::new(39, 4, 0x10);
    pub const ADDR_MOD_PACK_SEC2_YsrcClear: ThreadConfigField = ThreadConfigField::new(39, 5, 0x20);
    pub const ADDR_MOD_PACK_SEC2_YsrcIncr: ThreadConfigField = ThreadConfigField::new(39, 0, 0xf);
    pub const ADDR_MOD_PACK_SEC2_ZdstClear: ThreadConfigField =
        ThreadConfigField::new(39, 15, 0x8000);
    pub const ADDR_MOD_PACK_SEC2_ZdstIncr: ThreadConfigField =
        ThreadConfigField::new(39, 14, 0x4000);
    pub const ADDR_MOD_PACK_SEC2_ZsrcClear: ThreadConfigField =
        ThreadConfigField::new(39, 13, 0x2000);
    pub const ADDR_MOD_PACK_SEC2_ZsrcIncr: ThreadConfigField =
        ThreadConfigField::new(39, 12, 0x1000);
    pub const ADDR_MOD_PACK_SEC3_YdstCR: ThreadConfigField = ThreadConfigField::new(40, 10, 0x400);
    pub const ADDR_MOD_PACK_SEC3_YdstClear: ThreadConfigField =
        ThreadConfigField::new(40, 11, 0x800);
    pub const ADDR_MOD_PACK_SEC3_YdstIncr: ThreadConfigField = ThreadConfigField::new(40, 6, 0x3c0);
    pub const ADDR_MOD_PACK_SEC3_YsrcCR: ThreadConfigField = ThreadConfigField::new(40, 4, 0x10);
    pub const ADDR_MOD_PACK_SEC3_YsrcClear: ThreadConfigField = ThreadConfigField::new(40, 5, 0x20);
    pub const ADDR_MOD_PACK_SEC3_YsrcIncr: ThreadConfigField = ThreadConfigField::new(40, 0, 0xf);
    pub const ADDR_MOD_PACK_SEC3_ZdstClear: ThreadConfigField =
        ThreadConfigField::new(40, 15, 0x8000);
    pub const ADDR_MOD_PACK_SEC3_ZdstIncr: ThreadConfigField =
        ThreadConfigField::new(40, 14, 0x4000);
    pub const ADDR_MOD_PACK_SEC3_ZsrcClear: ThreadConfigField =
        ThreadConfigField::new(40, 13, 0x2000);
    pub const ADDR_MOD_PACK_SEC3_ZsrcIncr: ThreadConfigField =
        ThreadConfigField::new(40, 12, 0x1000);
    pub const CFG_STATE_ID_StateID: ThreadConfigField = ThreadConfigField::new(0, 0, 0x1);
    pub const CLR_DVALID_SrcA_Disable: ThreadConfigField = ThreadConfigField::new(7, 0, 0x1);
    pub const CLR_DVALID_SrcB_Disable: ThreadConfigField = ThreadConfigField::new(7, 1, 0x2);
    pub const DEST_TARGET_REG_CFG_MATH_Offset: ThreadConfigField =
        ThreadConfigField::new(1, 0, 0xfff);
    pub const DISABLE_IMPLIED_SRCA_FMT_Base: ThreadConfigField = ThreadConfigField::new(2, 0, 0x1);
    pub const DISABLE_IMPLIED_SRCB_FMT_Base: ThreadConfigField = ThreadConfigField::new(3, 0, 0x1);
    pub const ENABLE_ACC_STATS_Enable: ThreadConfigField = ThreadConfigField::new(45, 0, 0x1);
    pub const FIDELITY_BASE_Phase: ThreadConfigField = ThreadConfigField::new(11, 0, 0x3);
    pub const FP16A_FORCE_Enable: ThreadConfigField = ThreadConfigField::new(55, 0, 0x1);
    pub const FPU_BIAS_SEL_Pointer: ThreadConfigField = ThreadConfigField::new(46, 0, 0x1);
    pub const NOC_OVERLAY_MSG_CLEAR_MsgNum_0: ThreadConfigField =
        ThreadConfigField::new(42, 8, 0x700);
    pub const NOC_OVERLAY_MSG_CLEAR_MsgNum_1: ThreadConfigField =
        ThreadConfigField::new(43, 8, 0x700);
    pub const NOC_OVERLAY_MSG_CLEAR_StreamId_0: ThreadConfigField =
        ThreadConfigField::new(42, 0, 0x3f);
    pub const NOC_OVERLAY_MSG_CLEAR_StreamId_1: ThreadConfigField =
        ThreadConfigField::new(43, 0, 0x3f);
    pub const PACK_SCBD_BANK_MASK_32b_Enable: ThreadConfigField = ThreadConfigField::new(9, 0, 0x1);
    pub const PERF_CNT_CMD_Cmd0Start: ThreadConfigField = ThreadConfigField::new(44, 0, 0x1);
    pub const PERF_CNT_CMD_Cmd0Stop: ThreadConfigField = ThreadConfigField::new(44, 1, 0x2);
    pub const PERF_CNT_CMD_Cmd1Start: ThreadConfigField = ThreadConfigField::new(44, 2, 0x4);
    pub const PERF_CNT_CMD_Cmd1Stop: ThreadConfigField = ThreadConfigField::new(44, 3, 0x8);
    pub const PERF_CNT_CMD_Cmd2Start: ThreadConfigField = ThreadConfigField::new(44, 4, 0x10);
    pub const PERF_CNT_CMD_Cmd2Stop: ThreadConfigField = ThreadConfigField::new(44, 5, 0x20);
    pub const PERF_CNT_CMD_Cmd3Start: ThreadConfigField = ThreadConfigField::new(44, 6, 0x40);
    pub const PERF_CNT_CMD_Cmd3Stop: ThreadConfigField = ThreadConfigField::new(44, 7, 0x80);
    pub const SCBD_BANK_MASK_32b_Enable: ThreadConfigField = ThreadConfigField::new(8, 0, 0x1);
    pub const SFPU_DEST_FMT_Base: ThreadConfigField = ThreadConfigField::new(4, 1, 0x1e);
    pub const SFPU_DEST_FMT_Enable: ThreadConfigField = ThreadConfigField::new(4, 0, 0x1);
    pub const SFPU_STACK_Incr: ThreadConfigField = ThreadConfigField::new(36, 0, 0x3ff);
    pub const SRCA_SET_Base: ThreadConfigField = ThreadConfigField::new(5, 0, 0x3);
    pub const SRCA_SET_SetOvrdWithAddr: ThreadConfigField = ThreadConfigField::new(5, 2, 0x4);
    pub const SRCB_SET_Base: ThreadConfigField = ThreadConfigField::new(6, 0, 0x3);
    pub const STREAMWAIT_NUM_MSGS_HI_Val: ThreadConfigField = ThreadConfigField::new(58, 0, 0x7f);
    pub const STREAMWAIT_PHASE_HI_Val: ThreadConfigField = ThreadConfigField::new(57, 0, 0x3ff);
    pub const STREAM_ID_SYNC_SEC0_BankSel: ThreadConfigField = ThreadConfigField::new(59, 0, 0x3f);
    pub const STREAM_ID_SYNC_SEC1_BankSel: ThreadConfigField = ThreadConfigField::new(60, 0, 0x3f);
    pub const STREAM_ID_SYNC_SEC2_BankSel: ThreadConfigField = ThreadConfigField::new(61, 0, 0x3f);
    pub const STREAM_ID_SYNC_SEC3_BankSel: ThreadConfigField = ThreadConfigField::new(62, 0, 0x3f);
    pub const STREAM_ID_TRISC_SEC0_BankSel: ThreadConfigField = ThreadConfigField::new(63, 0, 0x3f);
    pub const STREAM_ID_TRISC_SEC1_BankSel: ThreadConfigField = ThreadConfigField::new(64, 0, 0x3f);
    pub const STREAM_ID_TRISC_SEC2_BankSel: ThreadConfigField = ThreadConfigField::new(65, 0, 0x3f);
    pub const STREAM_ID_TRISC_SEC3_BankSel: ThreadConfigField = ThreadConfigField::new(66, 0, 0x3f);
    pub const TENSIX_CSR_CONFIG_RawBusyStatus: ThreadConfigField =
        ThreadConfigField::new(67, 0, 0x1);
    pub const TENSIX_TRISC_SYNC_EnSubdividedCfgForUnpacr: ThreadConfigField =
        ThreadConfigField::new(56, 1, 0x2);
    pub const TENSIX_TRISC_SYNC_TrackGPR: ThreadConfigField = ThreadConfigField::new(56, 2, 0x4);
    pub const TENSIX_TRISC_SYNC_TrackGlobalCfg: ThreadConfigField =
        ThreadConfigField::new(56, 0, 0x1);
    pub const TENSIX_TRISC_SYNC_TrackTDMARegs: ThreadConfigField =
        ThreadConfigField::new(56, 3, 0x8);
    pub const TENSIX_TRISC_SYNC_TrackTensixInstructions: ThreadConfigField =
        ThreadConfigField::new(56, 4, 0x10);
    pub const UNPACK_MISC_CFG_CfgContextCntInc_0: ThreadConfigField =
        ThreadConfigField::new(41, 5, 0x20);
    pub const UNPACK_MISC_CFG_CfgContextCntInc_1: ThreadConfigField =
        ThreadConfigField::new(41, 13, 0x2000);
    pub const UNPACK_MISC_CFG_CfgContextCntReset_0: ThreadConfigField =
        ThreadConfigField::new(41, 4, 0x10);
    pub const UNPACK_MISC_CFG_CfgContextCntReset_1: ThreadConfigField =
        ThreadConfigField::new(41, 12, 0x1000);
    pub const UNPACK_MISC_CFG_CfgContextCntReset_metadata: ThreadConfigField =
        ThreadConfigField::new(41, 14, 0x4000);
    pub const UNPACK_MISC_CFG_CfgContextCntReset_metadata_zstart: ThreadConfigField =
        ThreadConfigField::new(41, 15, 0x8000);
    pub const UNPACK_MISC_CFG_CfgContextOffset_0: ThreadConfigField =
        ThreadConfigField::new(41, 0, 0xf);
    pub const UNPACK_MISC_CFG_CfgContextOffset_1: ThreadConfigField =
        ThreadConfigField::new(41, 8, 0xf00);
    pub const UNPACK_SCBD_BANK_MASK_32b_Enable: ThreadConfigField =
        ThreadConfigField::new(10, 0, 0x1);
}

/// `// Registers for UNPACK0` — 15 fields, indexing `Config` (write with `WRCFG`).
pub mod unpack0 {
    use super::*;

    /// First word index of this section, from `UNPACK0_CFGREG_BASE_ADDR32`.
    pub const CFGREG_BASE: u16 = 44;

    pub const UNP0_ADDR_BASE_REG_0_Base: ConfigField = ConfigField::new(48, 0, 0x3ffff);
    pub const UNP0_ADDR_BASE_REG_1_Base: ConfigField = ConfigField::new(49, 0, 0x3ffff);
    pub const UNP0_ADDR_CTRL_XY_REG_0_Xstride: ConfigField = ConfigField::new(44, 0, 0xffff);
    pub const UNP0_ADDR_CTRL_XY_REG_0_Ystride: ConfigField = ConfigField::new(44, 16, 0xffff0000);
    pub const UNP0_ADDR_CTRL_ZW_REG_0_Wstride: ConfigField = ConfigField::new(45, 16, 0xffff0000);
    pub const UNP0_ADDR_CTRL_ZW_REG_0_Zstride: ConfigField = ConfigField::new(45, 0, 0xffff);
    pub const UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr: ConfigField =
        ConfigField::new(50, 8, 0x100);
    pub const UNP0_BLOBS_Y_START_CNTX_01_blobs_y_start: ConfigField =
        ConfigField::new(51, 0, 0xffffffff);
    pub const UNP0_BLOBS_Y_START_CNTX_23_blobs_y_start: ConfigField =
        ConfigField::new(52, 0, 0xffffffff);
    pub const UNP0_FORCED_SHARED_EXP_shared_exp: ConfigField = ConfigField::new(50, 0, 0xff);
    pub const UNP0_NOP_REG_CLR_VAL_nop_reg_clr_val: ConfigField =
        ConfigField::new(53, 0, 0xffffffff);
    pub const UNP1_ADDR_CTRL_XY_REG_0_Xstride: ConfigField = ConfigField::new(46, 0, 0xffff);
    pub const UNP1_ADDR_CTRL_XY_REG_0_Ystride: ConfigField = ConfigField::new(46, 16, 0xffff0000);
    pub const UNP1_ADDR_CTRL_ZW_REG_0_Wstride: ConfigField = ConfigField::new(47, 16, 0xffff0000);
    pub const UNP1_ADDR_CTRL_ZW_REG_0_Zstride: ConfigField = ConfigField::new(47, 0, 0xffff);
}

/// `// Registers for UNPACK1` — 13 fields, indexing `Config` (write with `WRCFG`).
pub mod unpack1 {
    use super::*;

    /// First word index of this section, from `UNPACK1_CFGREG_BASE_ADDR32`.
    pub const CFGREG_BASE: u16 = 56;

    pub const UNP0_ADDR_CTRL_XY_REG_1_Xstride: ConfigField = ConfigField::new(56, 0, 0xffff);
    pub const UNP0_ADDR_CTRL_XY_REG_1_Ystride: ConfigField = ConfigField::new(56, 16, 0xffff0000);
    pub const UNP0_ADDR_CTRL_ZW_REG_1_Wstride: ConfigField = ConfigField::new(57, 16, 0xffff0000);
    pub const UNP0_ADDR_CTRL_ZW_REG_1_Zstride: ConfigField = ConfigField::new(57, 0, 0xffff);
    pub const UNP1_ADDR_BASE_REG_0_Base: ConfigField = ConfigField::new(60, 0, 0x3ffff);
    pub const UNP1_ADDR_BASE_REG_1_Base: ConfigField = ConfigField::new(61, 0, 0x3ffff);
    pub const UNP1_ADDR_CTRL_XY_REG_1_Xstride: ConfigField = ConfigField::new(58, 0, 0xffff);
    pub const UNP1_ADDR_CTRL_XY_REG_1_Ystride: ConfigField = ConfigField::new(58, 16, 0xffff0000);
    pub const UNP1_ADDR_CTRL_ZW_REG_1_Wstride: ConfigField = ConfigField::new(59, 16, 0xffff0000);
    pub const UNP1_ADDR_CTRL_ZW_REG_1_Zstride: ConfigField = ConfigField::new(59, 0, 0xffff);
    pub const UNP1_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr: ConfigField =
        ConfigField::new(62, 8, 0x100);
    pub const UNP1_FORCED_SHARED_EXP_shared_exp: ConfigField = ConfigField::new(62, 0, 0xff);
    pub const UNP1_NOP_REG_CLR_VAL_nop_reg_clr_val: ConfigField =
        ConfigField::new(63, 0, 0xffffffff);
}

/// Every `Config` bitfield, for table-wide invariant checks.
pub static ALL_CONFIG_FIELDS: &[(&str, ConfigField)] = &[
    ("ALU_ACC_CTRL_Fp32_enabled", alu::ALU_ACC_CTRL_Fp32_enabled),
    (
        "ALU_ACC_CTRL_INT8_math_enabled",
        alu::ALU_ACC_CTRL_INT8_math_enabled,
    ),
    (
        "ALU_ACC_CTRL_SFPU_Fp32_enabled",
        alu::ALU_ACC_CTRL_SFPU_Fp32_enabled,
    ),
    (
        "ALU_ACC_CTRL_Zero_Flag_disabled_dst",
        alu::ALU_ACC_CTRL_Zero_Flag_disabled_dst,
    ),
    (
        "ALU_ACC_CTRL_Zero_Flag_disabled_src",
        alu::ALU_ACC_CTRL_Zero_Flag_disabled_src,
    ),
    ("ALU_FORMAT_SPEC_REG0_SrcA", alu::ALU_FORMAT_SPEC_REG0_SrcA),
    (
        "ALU_FORMAT_SPEC_REG0_SrcAUnsigned",
        alu::ALU_FORMAT_SPEC_REG0_SrcAUnsigned,
    ),
    (
        "ALU_FORMAT_SPEC_REG0_SrcBUnsigned",
        alu::ALU_FORMAT_SPEC_REG0_SrcBUnsigned,
    ),
    ("ALU_FORMAT_SPEC_REG1_SrcB", alu::ALU_FORMAT_SPEC_REG1_SrcB),
    (
        "ALU_FORMAT_SPEC_REG2_Dstacc",
        alu::ALU_FORMAT_SPEC_REG2_Dstacc,
    ),
    (
        "ALU_FORMAT_SPEC_REG_Dstacc_override",
        alu::ALU_FORMAT_SPEC_REG_Dstacc_override,
    ),
    (
        "ALU_FORMAT_SPEC_REG_Dstacc_val",
        alu::ALU_FORMAT_SPEC_REG_Dstacc_val,
    ),
    (
        "ALU_FORMAT_SPEC_REG_SrcA_override",
        alu::ALU_FORMAT_SPEC_REG_SrcA_override,
    ),
    (
        "ALU_FORMAT_SPEC_REG_SrcA_val",
        alu::ALU_FORMAT_SPEC_REG_SrcA_val,
    ),
    (
        "ALU_FORMAT_SPEC_REG_SrcB_override",
        alu::ALU_FORMAT_SPEC_REG_SrcB_override,
    ),
    (
        "ALU_FORMAT_SPEC_REG_SrcB_val",
        alu::ALU_FORMAT_SPEC_REG_SrcB_val,
    ),
    ("ALU_ROUNDING_MODE_Bfp8_HF", alu::ALU_ROUNDING_MODE_Bfp8_HF),
    (
        "ALU_ROUNDING_MODE_Fpu_srnd_en",
        alu::ALU_ROUNDING_MODE_Fpu_srnd_en,
    ),
    ("ALU_ROUNDING_MODE_GS_LF", alu::ALU_ROUNDING_MODE_GS_LF),
    (
        "ALU_ROUNDING_MODE_Gasket_srnd_en",
        alu::ALU_ROUNDING_MODE_Gasket_srnd_en,
    ),
    (
        "ALU_ROUNDING_MODE_Packer_srnd_en",
        alu::ALU_ROUNDING_MODE_Packer_srnd_en,
    ),
    ("ALU_ROUNDING_MODE_Padding", alu::ALU_ROUNDING_MODE_Padding),
    ("DEST_OFFSET_Enable", alu::DEST_OFFSET_Enable),
    ("DEST_REGW_BASE_Base", alu::DEST_REGW_BASE_Base),
    ("DEST_SP_BASE_Base", alu::DEST_SP_BASE_Base),
    (
        "DISABLE_RISC_BP_Disable_bmp_clear_main",
        alu::DISABLE_RISC_BP_Disable_bmp_clear_main,
    ),
    (
        "DISABLE_RISC_BP_Disable_bmp_clear_ncrisc",
        alu::DISABLE_RISC_BP_Disable_bmp_clear_ncrisc,
    ),
    (
        "DISABLE_RISC_BP_Disable_bmp_clear_trisc",
        alu::DISABLE_RISC_BP_Disable_bmp_clear_trisc,
    ),
    (
        "DISABLE_RISC_BP_Disable_main",
        alu::DISABLE_RISC_BP_Disable_main,
    ),
    (
        "DISABLE_RISC_BP_Disable_ncrisc",
        alu::DISABLE_RISC_BP_Disable_ncrisc,
    ),
    (
        "DISABLE_RISC_BP_Disable_trisc",
        alu::DISABLE_RISC_BP_Disable_trisc,
    ),
    ("ECC_SCRUBBER_Delay", alu::ECC_SCRUBBER_Delay),
    ("ECC_SCRUBBER_Enable", alu::ECC_SCRUBBER_Enable),
    (
        "ECC_SCRUBBER_Scrub_On_Error",
        alu::ECC_SCRUBBER_Scrub_On_Error,
    ),
    (
        "ECC_SCRUBBER_Scrub_On_Error_Immediately",
        alu::ECC_SCRUBBER_Scrub_On_Error_Immediately,
    ),
    ("INT_DESCALE_Enable", alu::INT_DESCALE_Enable),
    ("INT_DESCALE_Mode", alu::INT_DESCALE_Mode),
    (
        "RISC_DEST_ACCESS_CTRL_SEC0_fmt",
        alu::RISC_DEST_ACCESS_CTRL_SEC0_fmt,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle",
        alu::RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int",
        alu::RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC1_fmt",
        alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle",
        alu::RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int",
        alu::RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC2_fmt",
        alu::RISC_DEST_ACCESS_CTRL_SEC2_fmt,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle",
        alu::RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int",
        alu::RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int,
    ),
    ("STACC_RELU_ApplyRelu", alu::STACC_RELU_ApplyRelu),
    ("STACC_RELU_ReluThreshold", alu::STACC_RELU_ReluThreshold),
    ("STATE_RESET_EN", alu::STATE_RESET_EN),
    ("BRISC_END_PC_PC", global::BRISC_END_PC_PC),
    (
        "CG_SRC_PIPELINE_GateSrcAPipeEn",
        global::CG_SRC_PIPELINE_GateSrcAPipeEn,
    ),
    (
        "CG_SRC_PIPELINE_GateSrcBPipeEn",
        global::CG_SRC_PIPELINE_GateSrcBPipeEn,
    ),
    (
        "CHICKEN_BITS_sfpu_scbd_disable",
        global::CHICKEN_BITS_sfpu_scbd_disable,
    ),
    (
        "DEST_ACCESS_CFG_disable_full_write_dest_q_bypass",
        global::DEST_ACCESS_CFG_disable_full_write_dest_q_bypass,
    ),
    (
        "DEST_ACCESS_CFG_remap_addrs",
        global::DEST_ACCESS_CFG_remap_addrs,
    ),
    (
        "DEST_ACCESS_CFG_swizzle_32b",
        global::DEST_ACCESS_CFG_swizzle_32b,
    ),
    (
        "DEST_ACCESS_CFG_zeroacc_absolute_tile_mode",
        global::DEST_ACCESS_CFG_zeroacc_absolute_tile_mode,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC0_Offset",
        global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC0_ZOffset",
        global::DEST_TARGET_REG_CFG_PACK_SEC0_ZOffset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC1_Offset",
        global::DEST_TARGET_REG_CFG_PACK_SEC1_Offset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC1_ZOffset",
        global::DEST_TARGET_REG_CFG_PACK_SEC1_ZOffset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC2_Offset",
        global::DEST_TARGET_REG_CFG_PACK_SEC2_Offset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC2_ZOffset",
        global::DEST_TARGET_REG_CFG_PACK_SEC2_ZOffset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC3_Offset",
        global::DEST_TARGET_REG_CFG_PACK_SEC3_Offset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC3_ZOffset",
        global::DEST_TARGET_REG_CFG_PACK_SEC3_ZOffset,
    ),
    (
        "INT_DESCALE_VALUES_SEC0_Value",
        global::INT_DESCALE_VALUES_SEC0_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC10_Value",
        global::INT_DESCALE_VALUES_SEC10_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC11_Value",
        global::INT_DESCALE_VALUES_SEC11_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC12_Value",
        global::INT_DESCALE_VALUES_SEC12_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC13_Value",
        global::INT_DESCALE_VALUES_SEC13_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC14_Value",
        global::INT_DESCALE_VALUES_SEC14_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC15_Value",
        global::INT_DESCALE_VALUES_SEC15_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC1_Value",
        global::INT_DESCALE_VALUES_SEC1_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC2_Value",
        global::INT_DESCALE_VALUES_SEC2_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC3_Value",
        global::INT_DESCALE_VALUES_SEC3_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC4_Value",
        global::INT_DESCALE_VALUES_SEC4_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC5_Value",
        global::INT_DESCALE_VALUES_SEC5_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC6_Value",
        global::INT_DESCALE_VALUES_SEC6_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC7_Value",
        global::INT_DESCALE_VALUES_SEC7_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC8_Value",
        global::INT_DESCALE_VALUES_SEC8_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC9_Value",
        global::INT_DESCALE_VALUES_SEC9_Value,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_bit_section_start_addr",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_bit_section_start_addr,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_chk",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_chk,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_offset",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_offset,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_End_Addr",
        global::L1_CACHE_TAG_SEARCH_ACCEL_End_Addr,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Search_Enable",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Search_Enable,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Start_Addr",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Start_Addr,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_high",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_high,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_low",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_low,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_Width",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_Width,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_alloc",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_alloc,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv_all",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv_all,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_end_addr",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_end_addr,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_start_addr",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_start_addr,
    ),
    ("NOC_RISC_END_PC_PC", global::NOC_RISC_END_PC_PC),
    ("PRNG_SEED_Seed_Val", global::PRNG_SEED_Seed_Val),
    (
        "RISCV_IC_INVALIDATE_InvalidateAll",
        global::RISCV_IC_INVALIDATE_InvalidateAll,
    ),
    (
        "RISC_PREFETCH_CTRL_Enable_Brisc",
        global::RISC_PREFETCH_CTRL_Enable_Brisc,
    ),
    (
        "RISC_PREFETCH_CTRL_Enable_NocRisc",
        global::RISC_PREFETCH_CTRL_Enable_NocRisc,
    ),
    (
        "RISC_PREFETCH_CTRL_Enable_Trisc",
        global::RISC_PREFETCH_CTRL_Enable_Trisc,
    ),
    (
        "RISC_PREFETCH_CTRL_Max_Req_Count",
        global::RISC_PREFETCH_CTRL_Max_Req_Count,
    ),
    ("SCRATCH_SEC0_val", global::SCRATCH_SEC0_val),
    ("SCRATCH_SEC1_val", global::SCRATCH_SEC1_val),
    ("SCRATCH_SEC2_val", global::SCRATCH_SEC2_val),
    (
        "SRC_ACCESS_CFG_disable_contig_srca_dvalid_phase",
        global::SRC_ACCESS_CFG_disable_contig_srca_dvalid_phase,
    ),
    (
        "SRC_ACCESS_CFG_disable_contig_srcb_dvalid_phase",
        global::SRC_ACCESS_CFG_disable_contig_srcb_dvalid_phase,
    ),
    (
        "SRC_ACCESS_CFG_math_view_srca_as_one_bank",
        global::SRC_ACCESS_CFG_math_view_srca_as_one_bank,
    ),
    (
        "SRC_ACCESS_CFG_math_view_srcb_as_one_bank",
        global::SRC_ACCESS_CFG_math_view_srcb_as_one_bank,
    ),
    ("TRISC_END_PC_SEC0_PC", global::TRISC_END_PC_SEC0_PC),
    ("TRISC_END_PC_SEC1_PC", global::TRISC_END_PC_SEC1_PC),
    ("TRISC_END_PC_SEC2_PC", global::TRISC_END_PC_SEC2_PC),
    (
        "PACK_CONCAT_MASK_SEC0_pack_concat_mask",
        pack0::PACK_CONCAT_MASK_SEC0_pack_concat_mask,
    ),
    (
        "PACK_CONCAT_MASK_SEC1_pack_concat_mask",
        pack0::PACK_CONCAT_MASK_SEC1_pack_concat_mask,
    ),
    (
        "PACK_CONCAT_MASK_SEC2_pack_concat_mask",
        pack0::PACK_CONCAT_MASK_SEC2_pack_concat_mask,
    ),
    (
        "PACK_CONCAT_MASK_SEC3_pack_concat_mask",
        pack0::PACK_CONCAT_MASK_SEC3_pack_concat_mask,
    ),
    (
        "PACK_COUNTERS_SEC0_auto_ctxt_inc_xys_cnt",
        pack0::PACK_COUNTERS_SEC0_auto_ctxt_inc_xys_cnt,
    ),
    (
        "PACK_COUNTERS_SEC0_pack_per_xy_plane",
        pack0::PACK_COUNTERS_SEC0_pack_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC0_pack_reads_per_xy_plane",
        pack0::PACK_COUNTERS_SEC0_pack_reads_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC0_pack_xys_per_tile",
        pack0::PACK_COUNTERS_SEC0_pack_xys_per_tile,
    ),
    (
        "PACK_COUNTERS_SEC0_pack_yz_transposed",
        pack0::PACK_COUNTERS_SEC0_pack_yz_transposed,
    ),
    (
        "PACK_COUNTERS_SEC1_auto_ctxt_inc_xys_cnt",
        pack0::PACK_COUNTERS_SEC1_auto_ctxt_inc_xys_cnt,
    ),
    (
        "PACK_COUNTERS_SEC1_pack_per_xy_plane",
        pack0::PACK_COUNTERS_SEC1_pack_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC1_pack_reads_per_xy_plane",
        pack0::PACK_COUNTERS_SEC1_pack_reads_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC1_pack_xys_per_tile",
        pack0::PACK_COUNTERS_SEC1_pack_xys_per_tile,
    ),
    (
        "PACK_COUNTERS_SEC1_pack_yz_transposed",
        pack0::PACK_COUNTERS_SEC1_pack_yz_transposed,
    ),
    (
        "PACK_COUNTERS_SEC2_auto_ctxt_inc_xys_cnt",
        pack0::PACK_COUNTERS_SEC2_auto_ctxt_inc_xys_cnt,
    ),
    (
        "PACK_COUNTERS_SEC2_pack_per_xy_plane",
        pack0::PACK_COUNTERS_SEC2_pack_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC2_pack_reads_per_xy_plane",
        pack0::PACK_COUNTERS_SEC2_pack_reads_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC2_pack_xys_per_tile",
        pack0::PACK_COUNTERS_SEC2_pack_xys_per_tile,
    ),
    (
        "PACK_COUNTERS_SEC2_pack_yz_transposed",
        pack0::PACK_COUNTERS_SEC2_pack_yz_transposed,
    ),
    (
        "PACK_COUNTERS_SEC3_auto_ctxt_inc_xys_cnt",
        pack0::PACK_COUNTERS_SEC3_auto_ctxt_inc_xys_cnt,
    ),
    (
        "PACK_COUNTERS_SEC3_pack_per_xy_plane",
        pack0::PACK_COUNTERS_SEC3_pack_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC3_pack_reads_per_xy_plane",
        pack0::PACK_COUNTERS_SEC3_pack_reads_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC3_pack_xys_per_tile",
        pack0::PACK_COUNTERS_SEC3_pack_xys_per_tile,
    ),
    (
        "PACK_COUNTERS_SEC3_pack_yz_transposed",
        pack0::PACK_COUNTERS_SEC3_pack_yz_transposed,
    ),
    (
        "PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain",
        pack0::PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain,
    ),
    (
        "PCK0_ADDR_BASE_REG_0_Base",
        pack0::PCK0_ADDR_BASE_REG_0_Base,
    ),
    (
        "PCK0_ADDR_BASE_REG_1_Base",
        pack0::PCK0_ADDR_BASE_REG_1_Base,
    ),
    (
        "PCK0_ADDR_CTRL_XY_REG_0_Xstride",
        pack0::PCK0_ADDR_CTRL_XY_REG_0_Xstride,
    ),
    (
        "PCK0_ADDR_CTRL_XY_REG_0_Ystride",
        pack0::PCK0_ADDR_CTRL_XY_REG_0_Ystride,
    ),
    (
        "PCK0_ADDR_CTRL_XY_REG_1_Xstride",
        pack0::PCK0_ADDR_CTRL_XY_REG_1_Xstride,
    ),
    (
        "PCK0_ADDR_CTRL_XY_REG_1_Ystride",
        pack0::PCK0_ADDR_CTRL_XY_REG_1_Ystride,
    ),
    (
        "PCK0_ADDR_CTRL_ZW_REG_0_Wstride",
        pack0::PCK0_ADDR_CTRL_ZW_REG_0_Wstride,
    ),
    (
        "PCK0_ADDR_CTRL_ZW_REG_0_Zstride",
        pack0::PCK0_ADDR_CTRL_ZW_REG_0_Zstride,
    ),
    (
        "PCK0_ADDR_CTRL_ZW_REG_1_Wstride",
        pack0::PCK0_ADDR_CTRL_ZW_REG_1_Wstride,
    ),
    (
        "PCK0_ADDR_CTRL_ZW_REG_1_Zstride",
        pack0::PCK0_ADDR_CTRL_ZW_REG_1_Zstride,
    ),
    (
        "PCK_DEST_RD_CTRL_Read_32b_data",
        pack0::PCK_DEST_RD_CTRL_Read_32b_data,
    ),
    (
        "PCK_DEST_RD_CTRL_Read_int8",
        pack0::PCK_DEST_RD_CTRL_Read_int8,
    ),
    (
        "PCK_DEST_RD_CTRL_Read_unsigned",
        pack0::PCK_DEST_RD_CTRL_Read_unsigned,
    ),
    (
        "PCK_DEST_RD_CTRL_Round_10b_mant",
        pack0::PCK_DEST_RD_CTRL_Round_10b_mant,
    ),
    ("PCK_EDGE_MODE_mode", pack0::PCK_EDGE_MODE_mode),
    (
        "PCK_EDGE_OFFSET_SEC0_mask",
        pack0::PCK_EDGE_OFFSET_SEC0_mask,
    ),
    (
        "PCK_EDGE_OFFSET_SEC1_mask",
        pack0::PCK_EDGE_OFFSET_SEC1_mask,
    ),
    (
        "PCK_EDGE_OFFSET_SEC2_mask",
        pack0::PCK_EDGE_OFFSET_SEC2_mask,
    ),
    (
        "PCK_EDGE_OFFSET_SEC3_mask",
        pack0::PCK_EDGE_OFFSET_SEC3_mask,
    ),
    (
        "PCK_EDGE_TILE_FACE_SET_SELECT_enable",
        pack0::PCK_EDGE_TILE_FACE_SET_SELECT_enable,
    ),
    (
        "PCK_EDGE_TILE_FACE_SET_SELECT_select",
        pack0::PCK_EDGE_TILE_FACE_SET_SELECT_select,
    ),
    (
        "PCK_EDGE_TILE_ROW_SET_SELECT_select",
        pack0::PCK_EDGE_TILE_ROW_SET_SELECT_select,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_0",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_0,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_1",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_1,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_10",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_10,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_11",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_11,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_12",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_12,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_13",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_13,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_14",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_14,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_15",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_15,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_2",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_2,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_3",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_3,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_4",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_4,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_5",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_5,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_6",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_6,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_7",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_7,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_8",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_8,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_9",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_9,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_0",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_0,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_1",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_1,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_10",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_10,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_11",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_11,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_12",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_12,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_13",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_13,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_14",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_14,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_15",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_15,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_2",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_2,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_3",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_3,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_4",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_4,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_5",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_5,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_6",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_6,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_7",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_7,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_8",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_8,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_9",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_9,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_0",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_0,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_1",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_1,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_10",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_10,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_11",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_11,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_12",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_12,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_13",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_13,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_14",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_14,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_15",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_15,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_2",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_2,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_3",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_3,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_4",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_4,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_5",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_5,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_6",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_6,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_7",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_7,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_8",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_8,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_9",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_9,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_0",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_0,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_1",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_1,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_10",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_10,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_11",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_11,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_12",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_12,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_13",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_13,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_14",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_14,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_15",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_15,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_2",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_2,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_3",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_3,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_4",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_4,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_5",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_5,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_6",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_6,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_7",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_7,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_8",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_8,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_9",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_9,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_0",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_0,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_1",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_1,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_10",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_10,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_11",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_11,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_12",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_12,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_13",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_13,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_14",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_14,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_15",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_15,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_2",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_2,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_3",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_3,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_4",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_4,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_5",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_5,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_6",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_6,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_7",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_7,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_8",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_8,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_9",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_9,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_0",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_0,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_1",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_1,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_10",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_10,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_11",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_11,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_12",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_12,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_13",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_13,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_14",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_14,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_15",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_15,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_2",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_2,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_3",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_3,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_4",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_4,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_5",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_5,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_6",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_6,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_7",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_7,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_8",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_8,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_9",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_9,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_0",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_0,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_1",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_1,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_10",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_10,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_11",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_11,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_12",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_12,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_13",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_13,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_14",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_14,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_15",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_15,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_2",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_2,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_3",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_3,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_4",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_4,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_5",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_5,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_6",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_6,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_7",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_7,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_8",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_8,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_9",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_9,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_0",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_0,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_1",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_1,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_10",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_10,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_11",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_11,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_12",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_12,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_13",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_13,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_14",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_14,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_15",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_15,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_2",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_2,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_3",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_3,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_4",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_4,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_5",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_5,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_6",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_6,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_7",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_7,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_8",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_8,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_9",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_9,
    ),
    (
        "THCON_SEC0_REG10_Packer_Reg_Wr_Addr",
        thcon::THCON_SEC0_REG10_Packer_Reg_Wr_Addr,
    ),
    (
        "THCON_SEC0_REG10_Unpack_fifo_size",
        thcon::THCON_SEC0_REG10_Unpack_fifo_size,
    ),
    (
        "THCON_SEC0_REG10_Unpack_limit_address",
        thcon::THCON_SEC0_REG10_Unpack_limit_address,
    ),
    (
        "THCON_SEC0_REG10_Unpack_limit_address_en",
        thcon::THCON_SEC0_REG10_Unpack_limit_address_en,
    ),
    (
        "THCON_SEC0_REG10_Unpacker_Reg_Wr_Addr",
        thcon::THCON_SEC0_REG10_Unpacker_Reg_Wr_Addr,
    ),
    (
        "THCON_SEC0_REG11_Metadata_cntxt_switch_unpacr_count",
        thcon::THCON_SEC0_REG11_Metadata_cntxt_switch_unpacr_count,
    ),
    (
        "THCON_SEC0_REG11_Metadata_fifo_size",
        thcon::THCON_SEC0_REG11_Metadata_fifo_size,
    ),
    (
        "THCON_SEC0_REG11_Metadata_l1_addr",
        thcon::THCON_SEC0_REG11_Metadata_l1_addr,
    ),
    (
        "THCON_SEC0_REG11_Metadata_limit_addr",
        thcon::THCON_SEC0_REG11_Metadata_limit_addr,
    ),
    (
        "THCON_SEC0_REG11_Metadata_z_cntr_rst_unpacr_count",
        thcon::THCON_SEC0_REG11_Metadata_z_cntr_rst_unpacr_count,
    ),
    (
        "THCON_SEC0_REG1_Add_l1_dest_addr_offset",
        thcon::THCON_SEC0_REG1_Add_l1_dest_addr_offset,
    ),
    (
        "THCON_SEC0_REG1_Add_tile_header_size",
        thcon::THCON_SEC0_REG1_Add_tile_header_size,
    ),
    (
        "THCON_SEC0_REG1_All_pack_disable_zero_compress_ovrd",
        thcon::THCON_SEC0_REG1_All_pack_disable_zero_compress_ovrd,
    ),
    (
        "THCON_SEC0_REG1_Auto_set_last_pacr_intf_sel",
        thcon::THCON_SEC0_REG1_Auto_set_last_pacr_intf_sel,
    ),
    (
        "THCON_SEC0_REG1_Dis_shared_exp_assembler",
        thcon::THCON_SEC0_REG1_Dis_shared_exp_assembler,
    ),
    (
        "THCON_SEC0_REG1_Disable_pack_zero_flags",
        thcon::THCON_SEC0_REG1_Disable_pack_zero_flags,
    ),
    (
        "THCON_SEC0_REG1_Disable_zero_compress",
        thcon::THCON_SEC0_REG1_Disable_zero_compress,
    ),
    (
        "THCON_SEC0_REG1_Downsample_mask",
        thcon::THCON_SEC0_REG1_Downsample_mask,
    ),
    (
        "THCON_SEC0_REG1_Downsample_rate",
        thcon::THCON_SEC0_REG1_Downsample_rate,
    ),
    (
        "THCON_SEC0_REG1_Enable_out_fifo",
        thcon::THCON_SEC0_REG1_Enable_out_fifo,
    ),
    (
        "THCON_SEC0_REG1_Exp_section_size",
        thcon::THCON_SEC0_REG1_Exp_section_size,
    ),
    (
        "THCON_SEC0_REG1_Exp_threshold",
        thcon::THCON_SEC0_REG1_Exp_threshold,
    ),
    (
        "THCON_SEC0_REG1_Exp_threshold_en",
        thcon::THCON_SEC0_REG1_Exp_threshold_en,
    ),
    (
        "THCON_SEC0_REG1_In_data_format",
        thcon::THCON_SEC0_REG1_In_data_format,
    ),
    (
        "THCON_SEC0_REG1_L1_Dest_addr",
        thcon::THCON_SEC0_REG1_L1_Dest_addr,
    ),
    (
        "THCON_SEC0_REG1_L1_source_addr",
        thcon::THCON_SEC0_REG1_L1_source_addr,
    ),
    (
        "THCON_SEC0_REG1_Out_data_format",
        thcon::THCON_SEC0_REG1_Out_data_format,
    ),
    (
        "THCON_SEC0_REG1_Pac_LF8_4b_exp",
        thcon::THCON_SEC0_REG1_Pac_LF8_4b_exp,
    ),
    (
        "THCON_SEC0_REG1_Pack_L1_Acc",
        thcon::THCON_SEC0_REG1_Pack_L1_Acc,
    ),
    (
        "THCON_SEC0_REG1_Row_start_section_size",
        thcon::THCON_SEC0_REG1_Row_start_section_size,
    ),
    (
        "THCON_SEC0_REG1_Source_interface_selection",
        thcon::THCON_SEC0_REG1_Source_interface_selection,
    ),
    (
        "THCON_SEC0_REG1_Sub_l1_tile_header_size",
        thcon::THCON_SEC0_REG1_Sub_l1_tile_header_size,
    ),
    (
        "THCON_SEC0_REG1_Unp_LF8_4b_exp",
        thcon::THCON_SEC0_REG1_Unp_LF8_4b_exp,
    ),
    (
        "THCON_SEC0_REG1_ovrd_default_throttle_mode",
        thcon::THCON_SEC0_REG1_ovrd_default_throttle_mode,
    ),
    (
        "THCON_SEC0_REG1_pack_dis_y_pos_start_offset",
        thcon::THCON_SEC0_REG1_pack_dis_y_pos_start_offset,
    ),
    (
        "THCON_SEC0_REG1_pack_start_intf_pos",
        thcon::THCON_SEC0_REG1_pack_start_intf_pos,
    ),
    (
        "THCON_SEC0_REG2_Context_count",
        thcon::THCON_SEC0_REG2_Context_count,
    ),
    (
        "THCON_SEC0_REG2_Context_count_non_log2",
        thcon::THCON_SEC0_REG2_Context_count_non_log2,
    ),
    (
        "THCON_SEC0_REG2_Context_count_non_log2_en",
        thcon::THCON_SEC0_REG2_Context_count_non_log2_en,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx0",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx0,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx1",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx1,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx2",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx2,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx3",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx3,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx4",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx4,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx5",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx5,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx6",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx6,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx7",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx7,
    ),
    (
        "THCON_SEC0_REG2_Force_shared_exp",
        thcon::THCON_SEC0_REG2_Force_shared_exp,
    ),
    (
        "THCON_SEC0_REG2_Haloize_mode",
        thcon::THCON_SEC0_REG2_Haloize_mode,
    ),
    (
        "THCON_SEC0_REG2_Metadata_x_end",
        thcon::THCON_SEC0_REG2_Metadata_x_end,
    ),
    (
        "THCON_SEC0_REG2_Out_data_format",
        thcon::THCON_SEC0_REG2_Out_data_format,
    ),
    (
        "THCON_SEC0_REG2_Ovrd_data_format",
        thcon::THCON_SEC0_REG2_Ovrd_data_format,
    ),
    (
        "THCON_SEC0_REG2_Shift_amount_cntx0",
        thcon::THCON_SEC0_REG2_Shift_amount_cntx0,
    ),
    (
        "THCON_SEC0_REG2_Shift_amount_cntx1",
        thcon::THCON_SEC0_REG2_Shift_amount_cntx1,
    ),
    (
        "THCON_SEC0_REG2_Shift_amount_cntx2",
        thcon::THCON_SEC0_REG2_Shift_amount_cntx2,
    ),
    (
        "THCON_SEC0_REG2_Shift_amount_cntx3",
        thcon::THCON_SEC0_REG2_Shift_amount_cntx3,
    ),
    (
        "THCON_SEC0_REG2_Throttle_mode",
        thcon::THCON_SEC0_REG2_Throttle_mode,
    ),
    (
        "THCON_SEC0_REG2_Tileize_mode",
        thcon::THCON_SEC0_REG2_Tileize_mode,
    ),
    (
        "THCON_SEC0_REG2_Unpack_If_Sel",
        thcon::THCON_SEC0_REG2_Unpack_If_Sel,
    ),
    (
        "THCON_SEC0_REG2_Unpack_Src_Reg_Set_Upd",
        thcon::THCON_SEC0_REG2_Unpack_Src_Reg_Set_Upd,
    ),
    (
        "THCON_SEC0_REG2_Unpack_fifo_size",
        thcon::THCON_SEC0_REG2_Unpack_fifo_size,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx0",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx1",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx1,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx2",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx2,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx3",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx3,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx4",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx4,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx5",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx5,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx6",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx6,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx7",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx7,
    ),
    (
        "THCON_SEC0_REG2_Unpack_limit_address",
        thcon::THCON_SEC0_REG2_Unpack_limit_address,
    ),
    (
        "THCON_SEC0_REG2_Upsample_and_interleave",
        thcon::THCON_SEC0_REG2_Upsample_and_interleave,
    ),
    (
        "THCON_SEC0_REG2_Upsample_rate",
        thcon::THCON_SEC0_REG2_Upsample_rate,
    ),
    (
        "THCON_SEC0_REG3_Base_address",
        thcon::THCON_SEC0_REG3_Base_address,
    ),
    (
        "THCON_SEC0_REG3_Base_cntx1_address",
        thcon::THCON_SEC0_REG3_Base_cntx1_address,
    ),
    (
        "THCON_SEC0_REG3_Base_cntx2_address",
        thcon::THCON_SEC0_REG3_Base_cntx2_address,
    ),
    (
        "THCON_SEC0_REG3_Base_cntx3_address",
        thcon::THCON_SEC0_REG3_Base_cntx3_address,
    ),
    (
        "THCON_SEC0_REG4_Base_cntx4_address",
        thcon::THCON_SEC0_REG4_Base_cntx4_address,
    ),
    (
        "THCON_SEC0_REG4_Base_cntx5_address",
        thcon::THCON_SEC0_REG4_Base_cntx5_address,
    ),
    (
        "THCON_SEC0_REG4_Base_cntx6_address",
        thcon::THCON_SEC0_REG4_Base_cntx6_address,
    ),
    (
        "THCON_SEC0_REG4_Base_cntx7_address",
        thcon::THCON_SEC0_REG4_Base_cntx7_address,
    ),
    (
        "THCON_SEC0_REG5_Dest_cntx0_address",
        thcon::THCON_SEC0_REG5_Dest_cntx0_address,
    ),
    (
        "THCON_SEC0_REG5_Dest_cntx1_address",
        thcon::THCON_SEC0_REG5_Dest_cntx1_address,
    ),
    (
        "THCON_SEC0_REG5_Dest_cntx2_address",
        thcon::THCON_SEC0_REG5_Dest_cntx2_address,
    ),
    (
        "THCON_SEC0_REG5_Dest_cntx3_address",
        thcon::THCON_SEC0_REG5_Dest_cntx3_address,
    ),
    (
        "THCON_SEC0_REG5_Tile_x_dim_cntx0",
        thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0,
    ),
    (
        "THCON_SEC0_REG5_Tile_x_dim_cntx1",
        thcon::THCON_SEC0_REG5_Tile_x_dim_cntx1,
    ),
    (
        "THCON_SEC0_REG5_Tile_x_dim_cntx2",
        thcon::THCON_SEC0_REG5_Tile_x_dim_cntx2,
    ),
    (
        "THCON_SEC0_REG5_Tile_x_dim_cntx3",
        thcon::THCON_SEC0_REG5_Tile_x_dim_cntx3,
    ),
    (
        "THCON_SEC0_REG6_Buffer_size",
        thcon::THCON_SEC0_REG6_Buffer_size,
    ),
    (
        "THCON_SEC0_REG6_Destination_address",
        thcon::THCON_SEC0_REG6_Destination_address,
    ),
    (
        "THCON_SEC0_REG6_Metadata_misc",
        thcon::THCON_SEC0_REG6_Metadata_misc,
    ),
    (
        "THCON_SEC0_REG6_Source_address",
        thcon::THCON_SEC0_REG6_Source_address,
    ),
    (
        "THCON_SEC0_REG6_Transfer_direction",
        thcon::THCON_SEC0_REG6_Transfer_direction,
    ),
    (
        "THCON_SEC0_REG7_Offset_address",
        thcon::THCON_SEC0_REG7_Offset_address,
    ),
    (
        "THCON_SEC0_REG7_Offset_cntx1_address",
        thcon::THCON_SEC0_REG7_Offset_cntx1_address,
    ),
    (
        "THCON_SEC0_REG7_Offset_cntx2_address",
        thcon::THCON_SEC0_REG7_Offset_cntx2_address,
    ),
    (
        "THCON_SEC0_REG7_Offset_cntx3_address",
        thcon::THCON_SEC0_REG7_Offset_cntx3_address,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx0",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx0,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx1",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx1,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx2",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx2,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx3",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx3,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx4",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx4,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx5",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx5,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx6",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx6,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx7",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx7,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx0",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx0,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx1",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx1,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx2",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx2,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx3",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx3,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx4",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx4,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx5",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx5,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx6",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx6,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx7",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx7,
    ),
    (
        "THCON_SEC0_REG8_Add_l1_dest_addr_offset",
        thcon::THCON_SEC0_REG8_Add_l1_dest_addr_offset,
    ),
    (
        "THCON_SEC0_REG8_Add_tile_header_size",
        thcon::THCON_SEC0_REG8_Add_tile_header_size,
    ),
    (
        "THCON_SEC0_REG8_Auto_set_last_pacr_intf_sel",
        thcon::THCON_SEC0_REG8_Auto_set_last_pacr_intf_sel,
    ),
    (
        "THCON_SEC0_REG8_Dis_shared_exp_assembler",
        thcon::THCON_SEC0_REG8_Dis_shared_exp_assembler,
    ),
    (
        "THCON_SEC0_REG8_Disable_pack_zero_flags",
        thcon::THCON_SEC0_REG8_Disable_pack_zero_flags,
    ),
    (
        "THCON_SEC0_REG8_Disable_zero_compress",
        thcon::THCON_SEC0_REG8_Disable_zero_compress,
    ),
    (
        "THCON_SEC0_REG8_Downsample_mask",
        thcon::THCON_SEC0_REG8_Downsample_mask,
    ),
    (
        "THCON_SEC0_REG8_Downsample_rate",
        thcon::THCON_SEC0_REG8_Downsample_rate,
    ),
    (
        "THCON_SEC0_REG8_Enable_out_fifo",
        thcon::THCON_SEC0_REG8_Enable_out_fifo,
    ),
    (
        "THCON_SEC0_REG8_Exp_section_size",
        thcon::THCON_SEC0_REG8_Exp_section_size,
    ),
    (
        "THCON_SEC0_REG8_Exp_threshold",
        thcon::THCON_SEC0_REG8_Exp_threshold,
    ),
    (
        "THCON_SEC0_REG8_Exp_threshold_en",
        thcon::THCON_SEC0_REG8_Exp_threshold_en,
    ),
    (
        "THCON_SEC0_REG8_In_data_format",
        thcon::THCON_SEC0_REG8_In_data_format,
    ),
    (
        "THCON_SEC0_REG8_L1_Dest_addr",
        thcon::THCON_SEC0_REG8_L1_Dest_addr,
    ),
    (
        "THCON_SEC0_REG8_L1_source_addr",
        thcon::THCON_SEC0_REG8_L1_source_addr,
    ),
    (
        "THCON_SEC0_REG8_Out_data_format",
        thcon::THCON_SEC0_REG8_Out_data_format,
    ),
    (
        "THCON_SEC0_REG8_Pack_L1_Acc",
        thcon::THCON_SEC0_REG8_Pack_L1_Acc,
    ),
    (
        "THCON_SEC0_REG8_Row_start_section_size",
        thcon::THCON_SEC0_REG8_Row_start_section_size,
    ),
    (
        "THCON_SEC0_REG8_Source_interface_selection",
        thcon::THCON_SEC0_REG8_Source_interface_selection,
    ),
    (
        "THCON_SEC0_REG8_Sub_l1_tile_header_size",
        thcon::THCON_SEC0_REG8_Sub_l1_tile_header_size,
    ),
    ("THCON_SEC0_REG8_Unused1", thcon::THCON_SEC0_REG8_Unused1),
    (
        "THCON_SEC0_REG8_pack_dis_y_pos_start_offset",
        thcon::THCON_SEC0_REG8_pack_dis_y_pos_start_offset,
    ),
    (
        "THCON_SEC0_REG8_unpack_tile_offset",
        thcon::THCON_SEC0_REG8_unpack_tile_offset,
    ),
    (
        "THCON_SEC0_REG9_Pack_0_2_fifo_size",
        thcon::THCON_SEC0_REG9_Pack_0_2_fifo_size,
    ),
    (
        "THCON_SEC0_REG9_Pack_0_2_limit_address",
        thcon::THCON_SEC0_REG9_Pack_0_2_limit_address,
    ),
    (
        "THCON_SEC0_REG9_Pack_1_3_fifo_size",
        thcon::THCON_SEC0_REG9_Pack_1_3_fifo_size,
    ),
    (
        "THCON_SEC0_REG9_Pack_1_3_limit_address",
        thcon::THCON_SEC0_REG9_Pack_1_3_limit_address,
    ),
    (
        "THCON_SEC1_REG10_Packer_Reg_Wr_Addr",
        thcon::THCON_SEC1_REG10_Packer_Reg_Wr_Addr,
    ),
    (
        "THCON_SEC1_REG10_Unpack_fifo_size",
        thcon::THCON_SEC1_REG10_Unpack_fifo_size,
    ),
    (
        "THCON_SEC1_REG10_Unpack_limit_address",
        thcon::THCON_SEC1_REG10_Unpack_limit_address,
    ),
    (
        "THCON_SEC1_REG10_Unpack_limit_address_en",
        thcon::THCON_SEC1_REG10_Unpack_limit_address_en,
    ),
    (
        "THCON_SEC1_REG10_Unpacker_Reg_Wr_Addr",
        thcon::THCON_SEC1_REG10_Unpacker_Reg_Wr_Addr,
    ),
    (
        "THCON_SEC1_REG11_Metadata_cntxt_switch_unpacr_count",
        thcon::THCON_SEC1_REG11_Metadata_cntxt_switch_unpacr_count,
    ),
    (
        "THCON_SEC1_REG11_Metadata_fifo_size",
        thcon::THCON_SEC1_REG11_Metadata_fifo_size,
    ),
    (
        "THCON_SEC1_REG11_Metadata_l1_addr",
        thcon::THCON_SEC1_REG11_Metadata_l1_addr,
    ),
    (
        "THCON_SEC1_REG11_Metadata_limit_addr",
        thcon::THCON_SEC1_REG11_Metadata_limit_addr,
    ),
    (
        "THCON_SEC1_REG11_Metadata_z_cntr_rst_unpacr_count",
        thcon::THCON_SEC1_REG11_Metadata_z_cntr_rst_unpacr_count,
    ),
    (
        "THCON_SEC1_REG1_Add_l1_dest_addr_offset",
        thcon::THCON_SEC1_REG1_Add_l1_dest_addr_offset,
    ),
    (
        "THCON_SEC1_REG1_Add_tile_header_size",
        thcon::THCON_SEC1_REG1_Add_tile_header_size,
    ),
    (
        "THCON_SEC1_REG1_All_pack_disable_zero_compress_ovrd",
        thcon::THCON_SEC1_REG1_All_pack_disable_zero_compress_ovrd,
    ),
    (
        "THCON_SEC1_REG1_Auto_set_last_pacr_intf_sel",
        thcon::THCON_SEC1_REG1_Auto_set_last_pacr_intf_sel,
    ),
    (
        "THCON_SEC1_REG1_Dis_shared_exp_assembler",
        thcon::THCON_SEC1_REG1_Dis_shared_exp_assembler,
    ),
    (
        "THCON_SEC1_REG1_Disable_pack_zero_flags",
        thcon::THCON_SEC1_REG1_Disable_pack_zero_flags,
    ),
    (
        "THCON_SEC1_REG1_Disable_zero_compress",
        thcon::THCON_SEC1_REG1_Disable_zero_compress,
    ),
    (
        "THCON_SEC1_REG1_Downsample_mask",
        thcon::THCON_SEC1_REG1_Downsample_mask,
    ),
    (
        "THCON_SEC1_REG1_Downsample_rate",
        thcon::THCON_SEC1_REG1_Downsample_rate,
    ),
    (
        "THCON_SEC1_REG1_Enable_out_fifo",
        thcon::THCON_SEC1_REG1_Enable_out_fifo,
    ),
    (
        "THCON_SEC1_REG1_Exp_section_size",
        thcon::THCON_SEC1_REG1_Exp_section_size,
    ),
    (
        "THCON_SEC1_REG1_Exp_threshold",
        thcon::THCON_SEC1_REG1_Exp_threshold,
    ),
    (
        "THCON_SEC1_REG1_Exp_threshold_en",
        thcon::THCON_SEC1_REG1_Exp_threshold_en,
    ),
    (
        "THCON_SEC1_REG1_In_data_format",
        thcon::THCON_SEC1_REG1_In_data_format,
    ),
    (
        "THCON_SEC1_REG1_L1_Dest_addr",
        thcon::THCON_SEC1_REG1_L1_Dest_addr,
    ),
    (
        "THCON_SEC1_REG1_L1_source_addr",
        thcon::THCON_SEC1_REG1_L1_source_addr,
    ),
    (
        "THCON_SEC1_REG1_Out_data_format",
        thcon::THCON_SEC1_REG1_Out_data_format,
    ),
    (
        "THCON_SEC1_REG1_Pac_LF8_4b_exp",
        thcon::THCON_SEC1_REG1_Pac_LF8_4b_exp,
    ),
    (
        "THCON_SEC1_REG1_Pack_L1_Acc",
        thcon::THCON_SEC1_REG1_Pack_L1_Acc,
    ),
    (
        "THCON_SEC1_REG1_Row_start_section_size",
        thcon::THCON_SEC1_REG1_Row_start_section_size,
    ),
    (
        "THCON_SEC1_REG1_Source_interface_selection",
        thcon::THCON_SEC1_REG1_Source_interface_selection,
    ),
    (
        "THCON_SEC1_REG1_Sub_l1_tile_header_size",
        thcon::THCON_SEC1_REG1_Sub_l1_tile_header_size,
    ),
    (
        "THCON_SEC1_REG1_Unp_LF8_4b_exp",
        thcon::THCON_SEC1_REG1_Unp_LF8_4b_exp,
    ),
    (
        "THCON_SEC1_REG1_ovrd_default_throttle_mode",
        thcon::THCON_SEC1_REG1_ovrd_default_throttle_mode,
    ),
    (
        "THCON_SEC1_REG1_pack_dis_y_pos_start_offset",
        thcon::THCON_SEC1_REG1_pack_dis_y_pos_start_offset,
    ),
    (
        "THCON_SEC1_REG1_pack_start_intf_pos",
        thcon::THCON_SEC1_REG1_pack_start_intf_pos,
    ),
    (
        "THCON_SEC1_REG2_Context_count",
        thcon::THCON_SEC1_REG2_Context_count,
    ),
    (
        "THCON_SEC1_REG2_Context_count_non_log2",
        thcon::THCON_SEC1_REG2_Context_count_non_log2,
    ),
    (
        "THCON_SEC1_REG2_Context_count_non_log2_en",
        thcon::THCON_SEC1_REG2_Context_count_non_log2_en,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx0",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx0,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx1",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx1,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx2",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx2,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx3",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx3,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx4",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx4,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx5",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx5,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx6",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx6,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx7",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx7,
    ),
    (
        "THCON_SEC1_REG2_Force_shared_exp",
        thcon::THCON_SEC1_REG2_Force_shared_exp,
    ),
    (
        "THCON_SEC1_REG2_Haloize_mode",
        thcon::THCON_SEC1_REG2_Haloize_mode,
    ),
    (
        "THCON_SEC1_REG2_Metadata_x_end",
        thcon::THCON_SEC1_REG2_Metadata_x_end,
    ),
    (
        "THCON_SEC1_REG2_Out_data_format",
        thcon::THCON_SEC1_REG2_Out_data_format,
    ),
    (
        "THCON_SEC1_REG2_Ovrd_data_format",
        thcon::THCON_SEC1_REG2_Ovrd_data_format,
    ),
    (
        "THCON_SEC1_REG2_Shift_amount_cntx0",
        thcon::THCON_SEC1_REG2_Shift_amount_cntx0,
    ),
    (
        "THCON_SEC1_REG2_Shift_amount_cntx1",
        thcon::THCON_SEC1_REG2_Shift_amount_cntx1,
    ),
    (
        "THCON_SEC1_REG2_Shift_amount_cntx2",
        thcon::THCON_SEC1_REG2_Shift_amount_cntx2,
    ),
    (
        "THCON_SEC1_REG2_Shift_amount_cntx3",
        thcon::THCON_SEC1_REG2_Shift_amount_cntx3,
    ),
    (
        "THCON_SEC1_REG2_Throttle_mode",
        thcon::THCON_SEC1_REG2_Throttle_mode,
    ),
    (
        "THCON_SEC1_REG2_Tileize_mode",
        thcon::THCON_SEC1_REG2_Tileize_mode,
    ),
    (
        "THCON_SEC1_REG2_Unpack_If_Sel",
        thcon::THCON_SEC1_REG2_Unpack_If_Sel,
    ),
    (
        "THCON_SEC1_REG2_Unpack_Src_Reg_Set_Upd",
        thcon::THCON_SEC1_REG2_Unpack_Src_Reg_Set_Upd,
    ),
    (
        "THCON_SEC1_REG2_Unpack_fifo_size",
        thcon::THCON_SEC1_REG2_Unpack_fifo_size,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx0",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx0,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx1",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx1,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx2",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx2,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx3",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx3,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx4",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx4,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx5",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx5,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx6",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx6,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx7",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx7,
    ),
    (
        "THCON_SEC1_REG2_Unpack_limit_address",
        thcon::THCON_SEC1_REG2_Unpack_limit_address,
    ),
    (
        "THCON_SEC1_REG2_Upsample_and_interleave",
        thcon::THCON_SEC1_REG2_Upsample_and_interleave,
    ),
    (
        "THCON_SEC1_REG2_Upsample_rate",
        thcon::THCON_SEC1_REG2_Upsample_rate,
    ),
    (
        "THCON_SEC1_REG3_Base_address",
        thcon::THCON_SEC1_REG3_Base_address,
    ),
    (
        "THCON_SEC1_REG3_Base_cntx1_address",
        thcon::THCON_SEC1_REG3_Base_cntx1_address,
    ),
    (
        "THCON_SEC1_REG3_Base_cntx2_address",
        thcon::THCON_SEC1_REG3_Base_cntx2_address,
    ),
    (
        "THCON_SEC1_REG3_Base_cntx3_address",
        thcon::THCON_SEC1_REG3_Base_cntx3_address,
    ),
    (
        "THCON_SEC1_REG4_Base_cntx4_address",
        thcon::THCON_SEC1_REG4_Base_cntx4_address,
    ),
    (
        "THCON_SEC1_REG4_Base_cntx5_address",
        thcon::THCON_SEC1_REG4_Base_cntx5_address,
    ),
    (
        "THCON_SEC1_REG4_Base_cntx6_address",
        thcon::THCON_SEC1_REG4_Base_cntx6_address,
    ),
    (
        "THCON_SEC1_REG4_Base_cntx7_address",
        thcon::THCON_SEC1_REG4_Base_cntx7_address,
    ),
    (
        "THCON_SEC1_REG5_Dest_cntx0_address",
        thcon::THCON_SEC1_REG5_Dest_cntx0_address,
    ),
    (
        "THCON_SEC1_REG5_Dest_cntx1_address",
        thcon::THCON_SEC1_REG5_Dest_cntx1_address,
    ),
    (
        "THCON_SEC1_REG5_Dest_cntx2_address",
        thcon::THCON_SEC1_REG5_Dest_cntx2_address,
    ),
    (
        "THCON_SEC1_REG5_Dest_cntx3_address",
        thcon::THCON_SEC1_REG5_Dest_cntx3_address,
    ),
    (
        "THCON_SEC1_REG5_Tile_x_dim_cntx0",
        thcon::THCON_SEC1_REG5_Tile_x_dim_cntx0,
    ),
    (
        "THCON_SEC1_REG5_Tile_x_dim_cntx1",
        thcon::THCON_SEC1_REG5_Tile_x_dim_cntx1,
    ),
    (
        "THCON_SEC1_REG5_Tile_x_dim_cntx2",
        thcon::THCON_SEC1_REG5_Tile_x_dim_cntx2,
    ),
    (
        "THCON_SEC1_REG5_Tile_x_dim_cntx3",
        thcon::THCON_SEC1_REG5_Tile_x_dim_cntx3,
    ),
    (
        "THCON_SEC1_REG6_Buffer_size",
        thcon::THCON_SEC1_REG6_Buffer_size,
    ),
    (
        "THCON_SEC1_REG6_Destination_address",
        thcon::THCON_SEC1_REG6_Destination_address,
    ),
    (
        "THCON_SEC1_REG6_Metadata_misc",
        thcon::THCON_SEC1_REG6_Metadata_misc,
    ),
    (
        "THCON_SEC1_REG6_Source_address",
        thcon::THCON_SEC1_REG6_Source_address,
    ),
    (
        "THCON_SEC1_REG6_Transfer_direction",
        thcon::THCON_SEC1_REG6_Transfer_direction,
    ),
    (
        "THCON_SEC1_REG7_Offset_address",
        thcon::THCON_SEC1_REG7_Offset_address,
    ),
    (
        "THCON_SEC1_REG7_Offset_cntx1_address",
        thcon::THCON_SEC1_REG7_Offset_cntx1_address,
    ),
    (
        "THCON_SEC1_REG7_Offset_cntx2_address",
        thcon::THCON_SEC1_REG7_Offset_cntx2_address,
    ),
    (
        "THCON_SEC1_REG7_Offset_cntx3_address",
        thcon::THCON_SEC1_REG7_Offset_cntx3_address,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx0",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx0,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx1",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx1,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx2",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx2,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx3",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx3,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx4",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx4,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx5",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx5,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx6",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx6,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx7",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx7,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx0",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx0,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx1",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx1,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx2",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx2,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx3",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx3,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx4",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx4,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx5",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx5,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx6",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx6,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx7",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx7,
    ),
    (
        "THCON_SEC1_REG8_Add_l1_dest_addr_offset",
        thcon::THCON_SEC1_REG8_Add_l1_dest_addr_offset,
    ),
    (
        "THCON_SEC1_REG8_Add_tile_header_size",
        thcon::THCON_SEC1_REG8_Add_tile_header_size,
    ),
    (
        "THCON_SEC1_REG8_Auto_set_last_pacr_intf_sel",
        thcon::THCON_SEC1_REG8_Auto_set_last_pacr_intf_sel,
    ),
    (
        "THCON_SEC1_REG8_Dis_shared_exp_assembler",
        thcon::THCON_SEC1_REG8_Dis_shared_exp_assembler,
    ),
    (
        "THCON_SEC1_REG8_Disable_pack_zero_flags",
        thcon::THCON_SEC1_REG8_Disable_pack_zero_flags,
    ),
    (
        "THCON_SEC1_REG8_Disable_zero_compress",
        thcon::THCON_SEC1_REG8_Disable_zero_compress,
    ),
    (
        "THCON_SEC1_REG8_Downsample_mask",
        thcon::THCON_SEC1_REG8_Downsample_mask,
    ),
    (
        "THCON_SEC1_REG8_Downsample_rate",
        thcon::THCON_SEC1_REG8_Downsample_rate,
    ),
    (
        "THCON_SEC1_REG8_Enable_out_fifo",
        thcon::THCON_SEC1_REG8_Enable_out_fifo,
    ),
    (
        "THCON_SEC1_REG8_Exp_section_size",
        thcon::THCON_SEC1_REG8_Exp_section_size,
    ),
    (
        "THCON_SEC1_REG8_Exp_threshold",
        thcon::THCON_SEC1_REG8_Exp_threshold,
    ),
    (
        "THCON_SEC1_REG8_Exp_threshold_en",
        thcon::THCON_SEC1_REG8_Exp_threshold_en,
    ),
    (
        "THCON_SEC1_REG8_In_data_format",
        thcon::THCON_SEC1_REG8_In_data_format,
    ),
    (
        "THCON_SEC1_REG8_L1_Dest_addr",
        thcon::THCON_SEC1_REG8_L1_Dest_addr,
    ),
    (
        "THCON_SEC1_REG8_L1_source_addr",
        thcon::THCON_SEC1_REG8_L1_source_addr,
    ),
    (
        "THCON_SEC1_REG8_Out_data_format",
        thcon::THCON_SEC1_REG8_Out_data_format,
    ),
    (
        "THCON_SEC1_REG8_Pack_L1_Acc",
        thcon::THCON_SEC1_REG8_Pack_L1_Acc,
    ),
    (
        "THCON_SEC1_REG8_Row_start_section_size",
        thcon::THCON_SEC1_REG8_Row_start_section_size,
    ),
    (
        "THCON_SEC1_REG8_Source_interface_selection",
        thcon::THCON_SEC1_REG8_Source_interface_selection,
    ),
    (
        "THCON_SEC1_REG8_Sub_l1_tile_header_size",
        thcon::THCON_SEC1_REG8_Sub_l1_tile_header_size,
    ),
    ("THCON_SEC1_REG8_Unused1", thcon::THCON_SEC1_REG8_Unused1),
    (
        "THCON_SEC1_REG8_pack_dis_y_pos_start_offset",
        thcon::THCON_SEC1_REG8_pack_dis_y_pos_start_offset,
    ),
    (
        "THCON_SEC1_REG8_unpack_tile_offset",
        thcon::THCON_SEC1_REG8_unpack_tile_offset,
    ),
    (
        "THCON_SEC1_REG9_Pack_0_2_fifo_size",
        thcon::THCON_SEC1_REG9_Pack_0_2_fifo_size,
    ),
    (
        "THCON_SEC1_REG9_Pack_0_2_limit_address",
        thcon::THCON_SEC1_REG9_Pack_0_2_limit_address,
    ),
    (
        "THCON_SEC1_REG9_Pack_1_3_fifo_size",
        thcon::THCON_SEC1_REG9_Pack_1_3_fifo_size,
    ),
    (
        "THCON_SEC1_REG9_Pack_1_3_limit_address",
        thcon::THCON_SEC1_REG9_Pack_1_3_limit_address,
    ),
    (
        "UNP0_ADDR_BASE_REG_0_Base",
        unpack0::UNP0_ADDR_BASE_REG_0_Base,
    ),
    (
        "UNP0_ADDR_BASE_REG_1_Base",
        unpack0::UNP0_ADDR_BASE_REG_1_Base,
    ),
    (
        "UNP0_ADDR_CTRL_XY_REG_0_Xstride",
        unpack0::UNP0_ADDR_CTRL_XY_REG_0_Xstride,
    ),
    (
        "UNP0_ADDR_CTRL_XY_REG_0_Ystride",
        unpack0::UNP0_ADDR_CTRL_XY_REG_0_Ystride,
    ),
    (
        "UNP0_ADDR_CTRL_ZW_REG_0_Wstride",
        unpack0::UNP0_ADDR_CTRL_ZW_REG_0_Wstride,
    ),
    (
        "UNP0_ADDR_CTRL_ZW_REG_0_Zstride",
        unpack0::UNP0_ADDR_CTRL_ZW_REG_0_Zstride,
    ),
    (
        "UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr",
        unpack0::UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr,
    ),
    (
        "UNP0_BLOBS_Y_START_CNTX_01_blobs_y_start",
        unpack0::UNP0_BLOBS_Y_START_CNTX_01_blobs_y_start,
    ),
    (
        "UNP0_BLOBS_Y_START_CNTX_23_blobs_y_start",
        unpack0::UNP0_BLOBS_Y_START_CNTX_23_blobs_y_start,
    ),
    (
        "UNP0_FORCED_SHARED_EXP_shared_exp",
        unpack0::UNP0_FORCED_SHARED_EXP_shared_exp,
    ),
    (
        "UNP0_NOP_REG_CLR_VAL_nop_reg_clr_val",
        unpack0::UNP0_NOP_REG_CLR_VAL_nop_reg_clr_val,
    ),
    (
        "UNP1_ADDR_CTRL_XY_REG_0_Xstride",
        unpack0::UNP1_ADDR_CTRL_XY_REG_0_Xstride,
    ),
    (
        "UNP1_ADDR_CTRL_XY_REG_0_Ystride",
        unpack0::UNP1_ADDR_CTRL_XY_REG_0_Ystride,
    ),
    (
        "UNP1_ADDR_CTRL_ZW_REG_0_Wstride",
        unpack0::UNP1_ADDR_CTRL_ZW_REG_0_Wstride,
    ),
    (
        "UNP1_ADDR_CTRL_ZW_REG_0_Zstride",
        unpack0::UNP1_ADDR_CTRL_ZW_REG_0_Zstride,
    ),
    (
        "UNP0_ADDR_CTRL_XY_REG_1_Xstride",
        unpack1::UNP0_ADDR_CTRL_XY_REG_1_Xstride,
    ),
    (
        "UNP0_ADDR_CTRL_XY_REG_1_Ystride",
        unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride,
    ),
    (
        "UNP0_ADDR_CTRL_ZW_REG_1_Wstride",
        unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Wstride,
    ),
    (
        "UNP0_ADDR_CTRL_ZW_REG_1_Zstride",
        unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride,
    ),
    (
        "UNP1_ADDR_BASE_REG_0_Base",
        unpack1::UNP1_ADDR_BASE_REG_0_Base,
    ),
    (
        "UNP1_ADDR_BASE_REG_1_Base",
        unpack1::UNP1_ADDR_BASE_REG_1_Base,
    ),
    (
        "UNP1_ADDR_CTRL_XY_REG_1_Xstride",
        unpack1::UNP1_ADDR_CTRL_XY_REG_1_Xstride,
    ),
    (
        "UNP1_ADDR_CTRL_XY_REG_1_Ystride",
        unpack1::UNP1_ADDR_CTRL_XY_REG_1_Ystride,
    ),
    (
        "UNP1_ADDR_CTRL_ZW_REG_1_Wstride",
        unpack1::UNP1_ADDR_CTRL_ZW_REG_1_Wstride,
    ),
    (
        "UNP1_ADDR_CTRL_ZW_REG_1_Zstride",
        unpack1::UNP1_ADDR_CTRL_ZW_REG_1_Zstride,
    ),
    (
        "UNP1_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr",
        unpack1::UNP1_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr,
    ),
    (
        "UNP1_FORCED_SHARED_EXP_shared_exp",
        unpack1::UNP1_FORCED_SHARED_EXP_shared_exp,
    ),
    (
        "UNP1_NOP_REG_CLR_VAL_nop_reg_clr_val",
        unpack1::UNP1_NOP_REG_CLR_VAL_nop_reg_clr_val,
    ),
];

/// Every `ThreadConfig` bitfield, for table-wide invariant checks.
pub static ALL_THREAD_CONFIG_FIELDS: &[(&str, ThreadConfigField)] = &[
    (
        "ADDR_MOD_AB2_SEC0_SrcAIncr",
        thread::ADDR_MOD_AB2_SEC0_SrcAIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC0_SrcBIncr",
        thread::ADDR_MOD_AB2_SEC0_SrcBIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC1_SrcAIncr",
        thread::ADDR_MOD_AB2_SEC1_SrcAIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC1_SrcBIncr",
        thread::ADDR_MOD_AB2_SEC1_SrcBIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC2_SrcAIncr",
        thread::ADDR_MOD_AB2_SEC2_SrcAIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC2_SrcBIncr",
        thread::ADDR_MOD_AB2_SEC2_SrcBIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC3_SrcAIncr",
        thread::ADDR_MOD_AB2_SEC3_SrcAIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC3_SrcBIncr",
        thread::ADDR_MOD_AB2_SEC3_SrcBIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC4_SrcAIncr",
        thread::ADDR_MOD_AB2_SEC4_SrcAIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC4_SrcBIncr",
        thread::ADDR_MOD_AB2_SEC4_SrcBIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC5_SrcAIncr",
        thread::ADDR_MOD_AB2_SEC5_SrcAIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC5_SrcBIncr",
        thread::ADDR_MOD_AB2_SEC5_SrcBIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC6_SrcAIncr",
        thread::ADDR_MOD_AB2_SEC6_SrcAIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC6_SrcBIncr",
        thread::ADDR_MOD_AB2_SEC6_SrcBIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC7_SrcAIncr",
        thread::ADDR_MOD_AB2_SEC7_SrcAIncr,
    ),
    (
        "ADDR_MOD_AB2_SEC7_SrcBIncr",
        thread::ADDR_MOD_AB2_SEC7_SrcBIncr,
    ),
    ("ADDR_MOD_AB_SEC0_SrcACR", thread::ADDR_MOD_AB_SEC0_SrcACR),
    (
        "ADDR_MOD_AB_SEC0_SrcAClear",
        thread::ADDR_MOD_AB_SEC0_SrcAClear,
    ),
    (
        "ADDR_MOD_AB_SEC0_SrcAIncr",
        thread::ADDR_MOD_AB_SEC0_SrcAIncr,
    ),
    ("ADDR_MOD_AB_SEC0_SrcBCR", thread::ADDR_MOD_AB_SEC0_SrcBCR),
    (
        "ADDR_MOD_AB_SEC0_SrcBClear",
        thread::ADDR_MOD_AB_SEC0_SrcBClear,
    ),
    (
        "ADDR_MOD_AB_SEC0_SrcBIncr",
        thread::ADDR_MOD_AB_SEC0_SrcBIncr,
    ),
    ("ADDR_MOD_AB_SEC1_SrcACR", thread::ADDR_MOD_AB_SEC1_SrcACR),
    (
        "ADDR_MOD_AB_SEC1_SrcAClear",
        thread::ADDR_MOD_AB_SEC1_SrcAClear,
    ),
    (
        "ADDR_MOD_AB_SEC1_SrcAIncr",
        thread::ADDR_MOD_AB_SEC1_SrcAIncr,
    ),
    ("ADDR_MOD_AB_SEC1_SrcBCR", thread::ADDR_MOD_AB_SEC1_SrcBCR),
    (
        "ADDR_MOD_AB_SEC1_SrcBClear",
        thread::ADDR_MOD_AB_SEC1_SrcBClear,
    ),
    (
        "ADDR_MOD_AB_SEC1_SrcBIncr",
        thread::ADDR_MOD_AB_SEC1_SrcBIncr,
    ),
    ("ADDR_MOD_AB_SEC2_SrcACR", thread::ADDR_MOD_AB_SEC2_SrcACR),
    (
        "ADDR_MOD_AB_SEC2_SrcAClear",
        thread::ADDR_MOD_AB_SEC2_SrcAClear,
    ),
    (
        "ADDR_MOD_AB_SEC2_SrcAIncr",
        thread::ADDR_MOD_AB_SEC2_SrcAIncr,
    ),
    ("ADDR_MOD_AB_SEC2_SrcBCR", thread::ADDR_MOD_AB_SEC2_SrcBCR),
    (
        "ADDR_MOD_AB_SEC2_SrcBClear",
        thread::ADDR_MOD_AB_SEC2_SrcBClear,
    ),
    (
        "ADDR_MOD_AB_SEC2_SrcBIncr",
        thread::ADDR_MOD_AB_SEC2_SrcBIncr,
    ),
    ("ADDR_MOD_AB_SEC3_SrcACR", thread::ADDR_MOD_AB_SEC3_SrcACR),
    (
        "ADDR_MOD_AB_SEC3_SrcAClear",
        thread::ADDR_MOD_AB_SEC3_SrcAClear,
    ),
    (
        "ADDR_MOD_AB_SEC3_SrcAIncr",
        thread::ADDR_MOD_AB_SEC3_SrcAIncr,
    ),
    ("ADDR_MOD_AB_SEC3_SrcBCR", thread::ADDR_MOD_AB_SEC3_SrcBCR),
    (
        "ADDR_MOD_AB_SEC3_SrcBClear",
        thread::ADDR_MOD_AB_SEC3_SrcBClear,
    ),
    (
        "ADDR_MOD_AB_SEC3_SrcBIncr",
        thread::ADDR_MOD_AB_SEC3_SrcBIncr,
    ),
    ("ADDR_MOD_AB_SEC4_SrcACR", thread::ADDR_MOD_AB_SEC4_SrcACR),
    (
        "ADDR_MOD_AB_SEC4_SrcAClear",
        thread::ADDR_MOD_AB_SEC4_SrcAClear,
    ),
    (
        "ADDR_MOD_AB_SEC4_SrcAIncr",
        thread::ADDR_MOD_AB_SEC4_SrcAIncr,
    ),
    ("ADDR_MOD_AB_SEC4_SrcBCR", thread::ADDR_MOD_AB_SEC4_SrcBCR),
    (
        "ADDR_MOD_AB_SEC4_SrcBClear",
        thread::ADDR_MOD_AB_SEC4_SrcBClear,
    ),
    (
        "ADDR_MOD_AB_SEC4_SrcBIncr",
        thread::ADDR_MOD_AB_SEC4_SrcBIncr,
    ),
    ("ADDR_MOD_AB_SEC5_SrcACR", thread::ADDR_MOD_AB_SEC5_SrcACR),
    (
        "ADDR_MOD_AB_SEC5_SrcAClear",
        thread::ADDR_MOD_AB_SEC5_SrcAClear,
    ),
    (
        "ADDR_MOD_AB_SEC5_SrcAIncr",
        thread::ADDR_MOD_AB_SEC5_SrcAIncr,
    ),
    ("ADDR_MOD_AB_SEC5_SrcBCR", thread::ADDR_MOD_AB_SEC5_SrcBCR),
    (
        "ADDR_MOD_AB_SEC5_SrcBClear",
        thread::ADDR_MOD_AB_SEC5_SrcBClear,
    ),
    (
        "ADDR_MOD_AB_SEC5_SrcBIncr",
        thread::ADDR_MOD_AB_SEC5_SrcBIncr,
    ),
    ("ADDR_MOD_AB_SEC6_SrcACR", thread::ADDR_MOD_AB_SEC6_SrcACR),
    (
        "ADDR_MOD_AB_SEC6_SrcAClear",
        thread::ADDR_MOD_AB_SEC6_SrcAClear,
    ),
    (
        "ADDR_MOD_AB_SEC6_SrcAIncr",
        thread::ADDR_MOD_AB_SEC6_SrcAIncr,
    ),
    ("ADDR_MOD_AB_SEC6_SrcBCR", thread::ADDR_MOD_AB_SEC6_SrcBCR),
    (
        "ADDR_MOD_AB_SEC6_SrcBClear",
        thread::ADDR_MOD_AB_SEC6_SrcBClear,
    ),
    (
        "ADDR_MOD_AB_SEC6_SrcBIncr",
        thread::ADDR_MOD_AB_SEC6_SrcBIncr,
    ),
    ("ADDR_MOD_AB_SEC7_SrcACR", thread::ADDR_MOD_AB_SEC7_SrcACR),
    (
        "ADDR_MOD_AB_SEC7_SrcAClear",
        thread::ADDR_MOD_AB_SEC7_SrcAClear,
    ),
    (
        "ADDR_MOD_AB_SEC7_SrcAIncr",
        thread::ADDR_MOD_AB_SEC7_SrcAIncr,
    ),
    ("ADDR_MOD_AB_SEC7_SrcBCR", thread::ADDR_MOD_AB_SEC7_SrcBCR),
    (
        "ADDR_MOD_AB_SEC7_SrcBClear",
        thread::ADDR_MOD_AB_SEC7_SrcBClear,
    ),
    (
        "ADDR_MOD_AB_SEC7_SrcBIncr",
        thread::ADDR_MOD_AB_SEC7_SrcBIncr,
    ),
    (
        "ADDR_MOD_BIAS_SEC0_BiasClear",
        thread::ADDR_MOD_BIAS_SEC0_BiasClear,
    ),
    (
        "ADDR_MOD_BIAS_SEC0_BiasIncr",
        thread::ADDR_MOD_BIAS_SEC0_BiasIncr,
    ),
    (
        "ADDR_MOD_BIAS_SEC1_BiasClear",
        thread::ADDR_MOD_BIAS_SEC1_BiasClear,
    ),
    (
        "ADDR_MOD_BIAS_SEC1_BiasIncr",
        thread::ADDR_MOD_BIAS_SEC1_BiasIncr,
    ),
    (
        "ADDR_MOD_BIAS_SEC2_BiasClear",
        thread::ADDR_MOD_BIAS_SEC2_BiasClear,
    ),
    (
        "ADDR_MOD_BIAS_SEC2_BiasIncr",
        thread::ADDR_MOD_BIAS_SEC2_BiasIncr,
    ),
    (
        "ADDR_MOD_BIAS_SEC3_BiasClear",
        thread::ADDR_MOD_BIAS_SEC3_BiasClear,
    ),
    (
        "ADDR_MOD_BIAS_SEC3_BiasIncr",
        thread::ADDR_MOD_BIAS_SEC3_BiasIncr,
    ),
    (
        "ADDR_MOD_BIAS_SEC4_BiasClear",
        thread::ADDR_MOD_BIAS_SEC4_BiasClear,
    ),
    (
        "ADDR_MOD_BIAS_SEC4_BiasIncr",
        thread::ADDR_MOD_BIAS_SEC4_BiasIncr,
    ),
    (
        "ADDR_MOD_BIAS_SEC5_BiasClear",
        thread::ADDR_MOD_BIAS_SEC5_BiasClear,
    ),
    (
        "ADDR_MOD_BIAS_SEC5_BiasIncr",
        thread::ADDR_MOD_BIAS_SEC5_BiasIncr,
    ),
    (
        "ADDR_MOD_BIAS_SEC6_BiasClear",
        thread::ADDR_MOD_BIAS_SEC6_BiasClear,
    ),
    (
        "ADDR_MOD_BIAS_SEC6_BiasIncr",
        thread::ADDR_MOD_BIAS_SEC6_BiasIncr,
    ),
    (
        "ADDR_MOD_BIAS_SEC7_BiasClear",
        thread::ADDR_MOD_BIAS_SEC7_BiasClear,
    ),
    (
        "ADDR_MOD_BIAS_SEC7_BiasIncr",
        thread::ADDR_MOD_BIAS_SEC7_BiasIncr,
    ),
    ("ADDR_MOD_DST_SEC0_DestCR", thread::ADDR_MOD_DST_SEC0_DestCR),
    (
        "ADDR_MOD_DST_SEC0_DestCToCR",
        thread::ADDR_MOD_DST_SEC0_DestCToCR,
    ),
    (
        "ADDR_MOD_DST_SEC0_DestClear",
        thread::ADDR_MOD_DST_SEC0_DestClear,
    ),
    (
        "ADDR_MOD_DST_SEC0_DestIncr",
        thread::ADDR_MOD_DST_SEC0_DestIncr,
    ),
    (
        "ADDR_MOD_DST_SEC0_FidelityClear",
        thread::ADDR_MOD_DST_SEC0_FidelityClear,
    ),
    (
        "ADDR_MOD_DST_SEC0_FidelityIncr",
        thread::ADDR_MOD_DST_SEC0_FidelityIncr,
    ),
    ("ADDR_MOD_DST_SEC1_DestCR", thread::ADDR_MOD_DST_SEC1_DestCR),
    (
        "ADDR_MOD_DST_SEC1_DestCToCR",
        thread::ADDR_MOD_DST_SEC1_DestCToCR,
    ),
    (
        "ADDR_MOD_DST_SEC1_DestClear",
        thread::ADDR_MOD_DST_SEC1_DestClear,
    ),
    (
        "ADDR_MOD_DST_SEC1_DestIncr",
        thread::ADDR_MOD_DST_SEC1_DestIncr,
    ),
    (
        "ADDR_MOD_DST_SEC1_FidelityClear",
        thread::ADDR_MOD_DST_SEC1_FidelityClear,
    ),
    (
        "ADDR_MOD_DST_SEC1_FidelityIncr",
        thread::ADDR_MOD_DST_SEC1_FidelityIncr,
    ),
    ("ADDR_MOD_DST_SEC2_DestCR", thread::ADDR_MOD_DST_SEC2_DestCR),
    (
        "ADDR_MOD_DST_SEC2_DestCToCR",
        thread::ADDR_MOD_DST_SEC2_DestCToCR,
    ),
    (
        "ADDR_MOD_DST_SEC2_DestClear",
        thread::ADDR_MOD_DST_SEC2_DestClear,
    ),
    (
        "ADDR_MOD_DST_SEC2_DestIncr",
        thread::ADDR_MOD_DST_SEC2_DestIncr,
    ),
    (
        "ADDR_MOD_DST_SEC2_FidelityClear",
        thread::ADDR_MOD_DST_SEC2_FidelityClear,
    ),
    (
        "ADDR_MOD_DST_SEC2_FidelityIncr",
        thread::ADDR_MOD_DST_SEC2_FidelityIncr,
    ),
    ("ADDR_MOD_DST_SEC3_DestCR", thread::ADDR_MOD_DST_SEC3_DestCR),
    (
        "ADDR_MOD_DST_SEC3_DestCToCR",
        thread::ADDR_MOD_DST_SEC3_DestCToCR,
    ),
    (
        "ADDR_MOD_DST_SEC3_DestClear",
        thread::ADDR_MOD_DST_SEC3_DestClear,
    ),
    (
        "ADDR_MOD_DST_SEC3_DestIncr",
        thread::ADDR_MOD_DST_SEC3_DestIncr,
    ),
    (
        "ADDR_MOD_DST_SEC3_FidelityClear",
        thread::ADDR_MOD_DST_SEC3_FidelityClear,
    ),
    (
        "ADDR_MOD_DST_SEC3_FidelityIncr",
        thread::ADDR_MOD_DST_SEC3_FidelityIncr,
    ),
    ("ADDR_MOD_DST_SEC4_DestCR", thread::ADDR_MOD_DST_SEC4_DestCR),
    (
        "ADDR_MOD_DST_SEC4_DestCToCR",
        thread::ADDR_MOD_DST_SEC4_DestCToCR,
    ),
    (
        "ADDR_MOD_DST_SEC4_DestClear",
        thread::ADDR_MOD_DST_SEC4_DestClear,
    ),
    (
        "ADDR_MOD_DST_SEC4_DestIncr",
        thread::ADDR_MOD_DST_SEC4_DestIncr,
    ),
    (
        "ADDR_MOD_DST_SEC4_FidelityClear",
        thread::ADDR_MOD_DST_SEC4_FidelityClear,
    ),
    (
        "ADDR_MOD_DST_SEC4_FidelityIncr",
        thread::ADDR_MOD_DST_SEC4_FidelityIncr,
    ),
    ("ADDR_MOD_DST_SEC5_DestCR", thread::ADDR_MOD_DST_SEC5_DestCR),
    (
        "ADDR_MOD_DST_SEC5_DestCToCR",
        thread::ADDR_MOD_DST_SEC5_DestCToCR,
    ),
    (
        "ADDR_MOD_DST_SEC5_DestClear",
        thread::ADDR_MOD_DST_SEC5_DestClear,
    ),
    (
        "ADDR_MOD_DST_SEC5_DestIncr",
        thread::ADDR_MOD_DST_SEC5_DestIncr,
    ),
    (
        "ADDR_MOD_DST_SEC5_FidelityClear",
        thread::ADDR_MOD_DST_SEC5_FidelityClear,
    ),
    (
        "ADDR_MOD_DST_SEC5_FidelityIncr",
        thread::ADDR_MOD_DST_SEC5_FidelityIncr,
    ),
    ("ADDR_MOD_DST_SEC6_DestCR", thread::ADDR_MOD_DST_SEC6_DestCR),
    (
        "ADDR_MOD_DST_SEC6_DestCToCR",
        thread::ADDR_MOD_DST_SEC6_DestCToCR,
    ),
    (
        "ADDR_MOD_DST_SEC6_DestClear",
        thread::ADDR_MOD_DST_SEC6_DestClear,
    ),
    (
        "ADDR_MOD_DST_SEC6_DestIncr",
        thread::ADDR_MOD_DST_SEC6_DestIncr,
    ),
    (
        "ADDR_MOD_DST_SEC6_FidelityClear",
        thread::ADDR_MOD_DST_SEC6_FidelityClear,
    ),
    (
        "ADDR_MOD_DST_SEC6_FidelityIncr",
        thread::ADDR_MOD_DST_SEC6_FidelityIncr,
    ),
    ("ADDR_MOD_DST_SEC7_DestCR", thread::ADDR_MOD_DST_SEC7_DestCR),
    (
        "ADDR_MOD_DST_SEC7_DestCToCR",
        thread::ADDR_MOD_DST_SEC7_DestCToCR,
    ),
    (
        "ADDR_MOD_DST_SEC7_DestClear",
        thread::ADDR_MOD_DST_SEC7_DestClear,
    ),
    (
        "ADDR_MOD_DST_SEC7_DestIncr",
        thread::ADDR_MOD_DST_SEC7_DestIncr,
    ),
    (
        "ADDR_MOD_DST_SEC7_FidelityClear",
        thread::ADDR_MOD_DST_SEC7_FidelityClear,
    ),
    (
        "ADDR_MOD_DST_SEC7_FidelityIncr",
        thread::ADDR_MOD_DST_SEC7_FidelityIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC0_YdstCR",
        thread::ADDR_MOD_PACK_SEC0_YdstCR,
    ),
    (
        "ADDR_MOD_PACK_SEC0_YdstClear",
        thread::ADDR_MOD_PACK_SEC0_YdstClear,
    ),
    (
        "ADDR_MOD_PACK_SEC0_YdstIncr",
        thread::ADDR_MOD_PACK_SEC0_YdstIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC0_YsrcCR",
        thread::ADDR_MOD_PACK_SEC0_YsrcCR,
    ),
    (
        "ADDR_MOD_PACK_SEC0_YsrcClear",
        thread::ADDR_MOD_PACK_SEC0_YsrcClear,
    ),
    (
        "ADDR_MOD_PACK_SEC0_YsrcIncr",
        thread::ADDR_MOD_PACK_SEC0_YsrcIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC0_ZdstClear",
        thread::ADDR_MOD_PACK_SEC0_ZdstClear,
    ),
    (
        "ADDR_MOD_PACK_SEC0_ZdstIncr",
        thread::ADDR_MOD_PACK_SEC0_ZdstIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC0_ZsrcClear",
        thread::ADDR_MOD_PACK_SEC0_ZsrcClear,
    ),
    (
        "ADDR_MOD_PACK_SEC0_ZsrcIncr",
        thread::ADDR_MOD_PACK_SEC0_ZsrcIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC1_YdstCR",
        thread::ADDR_MOD_PACK_SEC1_YdstCR,
    ),
    (
        "ADDR_MOD_PACK_SEC1_YdstClear",
        thread::ADDR_MOD_PACK_SEC1_YdstClear,
    ),
    (
        "ADDR_MOD_PACK_SEC1_YdstIncr",
        thread::ADDR_MOD_PACK_SEC1_YdstIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC1_YsrcCR",
        thread::ADDR_MOD_PACK_SEC1_YsrcCR,
    ),
    (
        "ADDR_MOD_PACK_SEC1_YsrcClear",
        thread::ADDR_MOD_PACK_SEC1_YsrcClear,
    ),
    (
        "ADDR_MOD_PACK_SEC1_YsrcIncr",
        thread::ADDR_MOD_PACK_SEC1_YsrcIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC1_ZdstClear",
        thread::ADDR_MOD_PACK_SEC1_ZdstClear,
    ),
    (
        "ADDR_MOD_PACK_SEC1_ZdstIncr",
        thread::ADDR_MOD_PACK_SEC1_ZdstIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC1_ZsrcClear",
        thread::ADDR_MOD_PACK_SEC1_ZsrcClear,
    ),
    (
        "ADDR_MOD_PACK_SEC1_ZsrcIncr",
        thread::ADDR_MOD_PACK_SEC1_ZsrcIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC2_YdstCR",
        thread::ADDR_MOD_PACK_SEC2_YdstCR,
    ),
    (
        "ADDR_MOD_PACK_SEC2_YdstClear",
        thread::ADDR_MOD_PACK_SEC2_YdstClear,
    ),
    (
        "ADDR_MOD_PACK_SEC2_YdstIncr",
        thread::ADDR_MOD_PACK_SEC2_YdstIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC2_YsrcCR",
        thread::ADDR_MOD_PACK_SEC2_YsrcCR,
    ),
    (
        "ADDR_MOD_PACK_SEC2_YsrcClear",
        thread::ADDR_MOD_PACK_SEC2_YsrcClear,
    ),
    (
        "ADDR_MOD_PACK_SEC2_YsrcIncr",
        thread::ADDR_MOD_PACK_SEC2_YsrcIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC2_ZdstClear",
        thread::ADDR_MOD_PACK_SEC2_ZdstClear,
    ),
    (
        "ADDR_MOD_PACK_SEC2_ZdstIncr",
        thread::ADDR_MOD_PACK_SEC2_ZdstIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC2_ZsrcClear",
        thread::ADDR_MOD_PACK_SEC2_ZsrcClear,
    ),
    (
        "ADDR_MOD_PACK_SEC2_ZsrcIncr",
        thread::ADDR_MOD_PACK_SEC2_ZsrcIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC3_YdstCR",
        thread::ADDR_MOD_PACK_SEC3_YdstCR,
    ),
    (
        "ADDR_MOD_PACK_SEC3_YdstClear",
        thread::ADDR_MOD_PACK_SEC3_YdstClear,
    ),
    (
        "ADDR_MOD_PACK_SEC3_YdstIncr",
        thread::ADDR_MOD_PACK_SEC3_YdstIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC3_YsrcCR",
        thread::ADDR_MOD_PACK_SEC3_YsrcCR,
    ),
    (
        "ADDR_MOD_PACK_SEC3_YsrcClear",
        thread::ADDR_MOD_PACK_SEC3_YsrcClear,
    ),
    (
        "ADDR_MOD_PACK_SEC3_YsrcIncr",
        thread::ADDR_MOD_PACK_SEC3_YsrcIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC3_ZdstClear",
        thread::ADDR_MOD_PACK_SEC3_ZdstClear,
    ),
    (
        "ADDR_MOD_PACK_SEC3_ZdstIncr",
        thread::ADDR_MOD_PACK_SEC3_ZdstIncr,
    ),
    (
        "ADDR_MOD_PACK_SEC3_ZsrcClear",
        thread::ADDR_MOD_PACK_SEC3_ZsrcClear,
    ),
    (
        "ADDR_MOD_PACK_SEC3_ZsrcIncr",
        thread::ADDR_MOD_PACK_SEC3_ZsrcIncr,
    ),
    ("CFG_STATE_ID_StateID", thread::CFG_STATE_ID_StateID),
    ("CLR_DVALID_SrcA_Disable", thread::CLR_DVALID_SrcA_Disable),
    ("CLR_DVALID_SrcB_Disable", thread::CLR_DVALID_SrcB_Disable),
    (
        "DEST_TARGET_REG_CFG_MATH_Offset",
        thread::DEST_TARGET_REG_CFG_MATH_Offset,
    ),
    (
        "DISABLE_IMPLIED_SRCA_FMT_Base",
        thread::DISABLE_IMPLIED_SRCA_FMT_Base,
    ),
    (
        "DISABLE_IMPLIED_SRCB_FMT_Base",
        thread::DISABLE_IMPLIED_SRCB_FMT_Base,
    ),
    ("ENABLE_ACC_STATS_Enable", thread::ENABLE_ACC_STATS_Enable),
    ("FIDELITY_BASE_Phase", thread::FIDELITY_BASE_Phase),
    ("FP16A_FORCE_Enable", thread::FP16A_FORCE_Enable),
    ("FPU_BIAS_SEL_Pointer", thread::FPU_BIAS_SEL_Pointer),
    (
        "NOC_OVERLAY_MSG_CLEAR_MsgNum_0",
        thread::NOC_OVERLAY_MSG_CLEAR_MsgNum_0,
    ),
    (
        "NOC_OVERLAY_MSG_CLEAR_MsgNum_1",
        thread::NOC_OVERLAY_MSG_CLEAR_MsgNum_1,
    ),
    (
        "NOC_OVERLAY_MSG_CLEAR_StreamId_0",
        thread::NOC_OVERLAY_MSG_CLEAR_StreamId_0,
    ),
    (
        "NOC_OVERLAY_MSG_CLEAR_StreamId_1",
        thread::NOC_OVERLAY_MSG_CLEAR_StreamId_1,
    ),
    (
        "PACK_SCBD_BANK_MASK_32b_Enable",
        thread::PACK_SCBD_BANK_MASK_32b_Enable,
    ),
    ("PERF_CNT_CMD_Cmd0Start", thread::PERF_CNT_CMD_Cmd0Start),
    ("PERF_CNT_CMD_Cmd0Stop", thread::PERF_CNT_CMD_Cmd0Stop),
    ("PERF_CNT_CMD_Cmd1Start", thread::PERF_CNT_CMD_Cmd1Start),
    ("PERF_CNT_CMD_Cmd1Stop", thread::PERF_CNT_CMD_Cmd1Stop),
    ("PERF_CNT_CMD_Cmd2Start", thread::PERF_CNT_CMD_Cmd2Start),
    ("PERF_CNT_CMD_Cmd2Stop", thread::PERF_CNT_CMD_Cmd2Stop),
    ("PERF_CNT_CMD_Cmd3Start", thread::PERF_CNT_CMD_Cmd3Start),
    ("PERF_CNT_CMD_Cmd3Stop", thread::PERF_CNT_CMD_Cmd3Stop),
    (
        "SCBD_BANK_MASK_32b_Enable",
        thread::SCBD_BANK_MASK_32b_Enable,
    ),
    ("SFPU_DEST_FMT_Base", thread::SFPU_DEST_FMT_Base),
    ("SFPU_DEST_FMT_Enable", thread::SFPU_DEST_FMT_Enable),
    ("SFPU_STACK_Incr", thread::SFPU_STACK_Incr),
    ("SRCA_SET_Base", thread::SRCA_SET_Base),
    ("SRCA_SET_SetOvrdWithAddr", thread::SRCA_SET_SetOvrdWithAddr),
    ("SRCB_SET_Base", thread::SRCB_SET_Base),
    (
        "STREAMWAIT_NUM_MSGS_HI_Val",
        thread::STREAMWAIT_NUM_MSGS_HI_Val,
    ),
    ("STREAMWAIT_PHASE_HI_Val", thread::STREAMWAIT_PHASE_HI_Val),
    (
        "STREAM_ID_SYNC_SEC0_BankSel",
        thread::STREAM_ID_SYNC_SEC0_BankSel,
    ),
    (
        "STREAM_ID_SYNC_SEC1_BankSel",
        thread::STREAM_ID_SYNC_SEC1_BankSel,
    ),
    (
        "STREAM_ID_SYNC_SEC2_BankSel",
        thread::STREAM_ID_SYNC_SEC2_BankSel,
    ),
    (
        "STREAM_ID_SYNC_SEC3_BankSel",
        thread::STREAM_ID_SYNC_SEC3_BankSel,
    ),
    (
        "STREAM_ID_TRISC_SEC0_BankSel",
        thread::STREAM_ID_TRISC_SEC0_BankSel,
    ),
    (
        "STREAM_ID_TRISC_SEC1_BankSel",
        thread::STREAM_ID_TRISC_SEC1_BankSel,
    ),
    (
        "STREAM_ID_TRISC_SEC2_BankSel",
        thread::STREAM_ID_TRISC_SEC2_BankSel,
    ),
    (
        "STREAM_ID_TRISC_SEC3_BankSel",
        thread::STREAM_ID_TRISC_SEC3_BankSel,
    ),
    (
        "TENSIX_CSR_CONFIG_RawBusyStatus",
        thread::TENSIX_CSR_CONFIG_RawBusyStatus,
    ),
    (
        "TENSIX_TRISC_SYNC_EnSubdividedCfgForUnpacr",
        thread::TENSIX_TRISC_SYNC_EnSubdividedCfgForUnpacr,
    ),
    (
        "TENSIX_TRISC_SYNC_TrackGPR",
        thread::TENSIX_TRISC_SYNC_TrackGPR,
    ),
    (
        "TENSIX_TRISC_SYNC_TrackGlobalCfg",
        thread::TENSIX_TRISC_SYNC_TrackGlobalCfg,
    ),
    (
        "TENSIX_TRISC_SYNC_TrackTDMARegs",
        thread::TENSIX_TRISC_SYNC_TrackTDMARegs,
    ),
    (
        "TENSIX_TRISC_SYNC_TrackTensixInstructions",
        thread::TENSIX_TRISC_SYNC_TrackTensixInstructions,
    ),
    (
        "UNPACK_MISC_CFG_CfgContextCntInc_0",
        thread::UNPACK_MISC_CFG_CfgContextCntInc_0,
    ),
    (
        "UNPACK_MISC_CFG_CfgContextCntInc_1",
        thread::UNPACK_MISC_CFG_CfgContextCntInc_1,
    ),
    (
        "UNPACK_MISC_CFG_CfgContextCntReset_0",
        thread::UNPACK_MISC_CFG_CfgContextCntReset_0,
    ),
    (
        "UNPACK_MISC_CFG_CfgContextCntReset_1",
        thread::UNPACK_MISC_CFG_CfgContextCntReset_1,
    ),
    (
        "UNPACK_MISC_CFG_CfgContextCntReset_metadata",
        thread::UNPACK_MISC_CFG_CfgContextCntReset_metadata,
    ),
    (
        "UNPACK_MISC_CFG_CfgContextCntReset_metadata_zstart",
        thread::UNPACK_MISC_CFG_CfgContextCntReset_metadata_zstart,
    ),
    (
        "UNPACK_MISC_CFG_CfgContextOffset_0",
        thread::UNPACK_MISC_CFG_CfgContextOffset_0,
    ),
    (
        "UNPACK_MISC_CFG_CfgContextOffset_1",
        thread::UNPACK_MISC_CFG_CfgContextOffset_1,
    ),
    (
        "UNPACK_SCBD_BANK_MASK_32b_Enable",
        thread::UNPACK_SCBD_BANK_MASK_32b_Enable,
    ),
];

/// Every multi-word `Config` aggregate.
pub static ALL_CONFIG_SPANS: &[(&str, ConfigSpan)] = &[
    (
        "THCON_SEC0_REG0_TileDescriptor",
        thcon::THCON_SEC0_REG0_TileDescriptor,
    ),
    (
        "THCON_SEC1_REG0_TileDescriptor",
        thcon::THCON_SEC1_REG0_TileDescriptor,
    ),
];

/// Every `Config` bitfield in the `alu` section.
pub static ALU_FIELDS: &[(&str, ConfigField)] = &[
    ("ALU_ACC_CTRL_Fp32_enabled", alu::ALU_ACC_CTRL_Fp32_enabled),
    (
        "ALU_ACC_CTRL_INT8_math_enabled",
        alu::ALU_ACC_CTRL_INT8_math_enabled,
    ),
    (
        "ALU_ACC_CTRL_SFPU_Fp32_enabled",
        alu::ALU_ACC_CTRL_SFPU_Fp32_enabled,
    ),
    (
        "ALU_ACC_CTRL_Zero_Flag_disabled_dst",
        alu::ALU_ACC_CTRL_Zero_Flag_disabled_dst,
    ),
    (
        "ALU_ACC_CTRL_Zero_Flag_disabled_src",
        alu::ALU_ACC_CTRL_Zero_Flag_disabled_src,
    ),
    ("ALU_FORMAT_SPEC_REG0_SrcA", alu::ALU_FORMAT_SPEC_REG0_SrcA),
    (
        "ALU_FORMAT_SPEC_REG0_SrcAUnsigned",
        alu::ALU_FORMAT_SPEC_REG0_SrcAUnsigned,
    ),
    (
        "ALU_FORMAT_SPEC_REG0_SrcBUnsigned",
        alu::ALU_FORMAT_SPEC_REG0_SrcBUnsigned,
    ),
    ("ALU_FORMAT_SPEC_REG1_SrcB", alu::ALU_FORMAT_SPEC_REG1_SrcB),
    (
        "ALU_FORMAT_SPEC_REG2_Dstacc",
        alu::ALU_FORMAT_SPEC_REG2_Dstacc,
    ),
    (
        "ALU_FORMAT_SPEC_REG_Dstacc_override",
        alu::ALU_FORMAT_SPEC_REG_Dstacc_override,
    ),
    (
        "ALU_FORMAT_SPEC_REG_Dstacc_val",
        alu::ALU_FORMAT_SPEC_REG_Dstacc_val,
    ),
    (
        "ALU_FORMAT_SPEC_REG_SrcA_override",
        alu::ALU_FORMAT_SPEC_REG_SrcA_override,
    ),
    (
        "ALU_FORMAT_SPEC_REG_SrcA_val",
        alu::ALU_FORMAT_SPEC_REG_SrcA_val,
    ),
    (
        "ALU_FORMAT_SPEC_REG_SrcB_override",
        alu::ALU_FORMAT_SPEC_REG_SrcB_override,
    ),
    (
        "ALU_FORMAT_SPEC_REG_SrcB_val",
        alu::ALU_FORMAT_SPEC_REG_SrcB_val,
    ),
    ("ALU_ROUNDING_MODE_Bfp8_HF", alu::ALU_ROUNDING_MODE_Bfp8_HF),
    (
        "ALU_ROUNDING_MODE_Fpu_srnd_en",
        alu::ALU_ROUNDING_MODE_Fpu_srnd_en,
    ),
    ("ALU_ROUNDING_MODE_GS_LF", alu::ALU_ROUNDING_MODE_GS_LF),
    (
        "ALU_ROUNDING_MODE_Gasket_srnd_en",
        alu::ALU_ROUNDING_MODE_Gasket_srnd_en,
    ),
    (
        "ALU_ROUNDING_MODE_Packer_srnd_en",
        alu::ALU_ROUNDING_MODE_Packer_srnd_en,
    ),
    ("ALU_ROUNDING_MODE_Padding", alu::ALU_ROUNDING_MODE_Padding),
    ("DEST_OFFSET_Enable", alu::DEST_OFFSET_Enable),
    ("DEST_REGW_BASE_Base", alu::DEST_REGW_BASE_Base),
    ("DEST_SP_BASE_Base", alu::DEST_SP_BASE_Base),
    (
        "DISABLE_RISC_BP_Disable_bmp_clear_main",
        alu::DISABLE_RISC_BP_Disable_bmp_clear_main,
    ),
    (
        "DISABLE_RISC_BP_Disable_bmp_clear_ncrisc",
        alu::DISABLE_RISC_BP_Disable_bmp_clear_ncrisc,
    ),
    (
        "DISABLE_RISC_BP_Disable_bmp_clear_trisc",
        alu::DISABLE_RISC_BP_Disable_bmp_clear_trisc,
    ),
    (
        "DISABLE_RISC_BP_Disable_main",
        alu::DISABLE_RISC_BP_Disable_main,
    ),
    (
        "DISABLE_RISC_BP_Disable_ncrisc",
        alu::DISABLE_RISC_BP_Disable_ncrisc,
    ),
    (
        "DISABLE_RISC_BP_Disable_trisc",
        alu::DISABLE_RISC_BP_Disable_trisc,
    ),
    ("ECC_SCRUBBER_Delay", alu::ECC_SCRUBBER_Delay),
    ("ECC_SCRUBBER_Enable", alu::ECC_SCRUBBER_Enable),
    (
        "ECC_SCRUBBER_Scrub_On_Error",
        alu::ECC_SCRUBBER_Scrub_On_Error,
    ),
    (
        "ECC_SCRUBBER_Scrub_On_Error_Immediately",
        alu::ECC_SCRUBBER_Scrub_On_Error_Immediately,
    ),
    ("INT_DESCALE_Enable", alu::INT_DESCALE_Enable),
    ("INT_DESCALE_Mode", alu::INT_DESCALE_Mode),
    (
        "RISC_DEST_ACCESS_CTRL_SEC0_fmt",
        alu::RISC_DEST_ACCESS_CTRL_SEC0_fmt,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle",
        alu::RISC_DEST_ACCESS_CTRL_SEC0_no_swizzle,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int",
        alu::RISC_DEST_ACCESS_CTRL_SEC0_unsigned_int,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC1_fmt",
        alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle",
        alu::RISC_DEST_ACCESS_CTRL_SEC1_no_swizzle,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int",
        alu::RISC_DEST_ACCESS_CTRL_SEC1_unsigned_int,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC2_fmt",
        alu::RISC_DEST_ACCESS_CTRL_SEC2_fmt,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle",
        alu::RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle,
    ),
    (
        "RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int",
        alu::RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int,
    ),
    ("STACC_RELU_ApplyRelu", alu::STACC_RELU_ApplyRelu),
    ("STACC_RELU_ReluThreshold", alu::STACC_RELU_ReluThreshold),
    ("STATE_RESET_EN", alu::STATE_RESET_EN),
];

/// Every `Config` bitfield in the `global` section.
pub static GLOBAL_FIELDS: &[(&str, ConfigField)] = &[
    ("BRISC_END_PC_PC", global::BRISC_END_PC_PC),
    (
        "CG_SRC_PIPELINE_GateSrcAPipeEn",
        global::CG_SRC_PIPELINE_GateSrcAPipeEn,
    ),
    (
        "CG_SRC_PIPELINE_GateSrcBPipeEn",
        global::CG_SRC_PIPELINE_GateSrcBPipeEn,
    ),
    (
        "CHICKEN_BITS_sfpu_scbd_disable",
        global::CHICKEN_BITS_sfpu_scbd_disable,
    ),
    (
        "DEST_ACCESS_CFG_disable_full_write_dest_q_bypass",
        global::DEST_ACCESS_CFG_disable_full_write_dest_q_bypass,
    ),
    (
        "DEST_ACCESS_CFG_remap_addrs",
        global::DEST_ACCESS_CFG_remap_addrs,
    ),
    (
        "DEST_ACCESS_CFG_swizzle_32b",
        global::DEST_ACCESS_CFG_swizzle_32b,
    ),
    (
        "DEST_ACCESS_CFG_zeroacc_absolute_tile_mode",
        global::DEST_ACCESS_CFG_zeroacc_absolute_tile_mode,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC0_Offset",
        global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC0_ZOffset",
        global::DEST_TARGET_REG_CFG_PACK_SEC0_ZOffset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC1_Offset",
        global::DEST_TARGET_REG_CFG_PACK_SEC1_Offset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC1_ZOffset",
        global::DEST_TARGET_REG_CFG_PACK_SEC1_ZOffset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC2_Offset",
        global::DEST_TARGET_REG_CFG_PACK_SEC2_Offset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC2_ZOffset",
        global::DEST_TARGET_REG_CFG_PACK_SEC2_ZOffset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC3_Offset",
        global::DEST_TARGET_REG_CFG_PACK_SEC3_Offset,
    ),
    (
        "DEST_TARGET_REG_CFG_PACK_SEC3_ZOffset",
        global::DEST_TARGET_REG_CFG_PACK_SEC3_ZOffset,
    ),
    (
        "INT_DESCALE_VALUES_SEC0_Value",
        global::INT_DESCALE_VALUES_SEC0_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC10_Value",
        global::INT_DESCALE_VALUES_SEC10_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC11_Value",
        global::INT_DESCALE_VALUES_SEC11_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC12_Value",
        global::INT_DESCALE_VALUES_SEC12_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC13_Value",
        global::INT_DESCALE_VALUES_SEC13_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC14_Value",
        global::INT_DESCALE_VALUES_SEC14_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC15_Value",
        global::INT_DESCALE_VALUES_SEC15_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC1_Value",
        global::INT_DESCALE_VALUES_SEC1_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC2_Value",
        global::INT_DESCALE_VALUES_SEC2_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC3_Value",
        global::INT_DESCALE_VALUES_SEC3_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC4_Value",
        global::INT_DESCALE_VALUES_SEC4_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC5_Value",
        global::INT_DESCALE_VALUES_SEC5_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC6_Value",
        global::INT_DESCALE_VALUES_SEC6_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC7_Value",
        global::INT_DESCALE_VALUES_SEC7_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC8_Value",
        global::INT_DESCALE_VALUES_SEC8_Value,
    ),
    (
        "INT_DESCALE_VALUES_SEC9_Value",
        global::INT_DESCALE_VALUES_SEC9_Value,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_bit_section_start_addr",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_bit_section_start_addr,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_chk",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_chk,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_offset",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_offset,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_End_Addr",
        global::L1_CACHE_TAG_SEARCH_ACCEL_End_Addr,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Search_Enable",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Search_Enable,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Start_Addr",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Start_Addr,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_high",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_high,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_low",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_low,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_Width",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_Width,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_alloc",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_alloc,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv_all",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv_all,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_end_addr",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_end_addr,
    ),
    (
        "L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_start_addr",
        global::L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_start_addr,
    ),
    ("NOC_RISC_END_PC_PC", global::NOC_RISC_END_PC_PC),
    ("PRNG_SEED_Seed_Val", global::PRNG_SEED_Seed_Val),
    (
        "RISCV_IC_INVALIDATE_InvalidateAll",
        global::RISCV_IC_INVALIDATE_InvalidateAll,
    ),
    (
        "RISC_PREFETCH_CTRL_Enable_Brisc",
        global::RISC_PREFETCH_CTRL_Enable_Brisc,
    ),
    (
        "RISC_PREFETCH_CTRL_Enable_NocRisc",
        global::RISC_PREFETCH_CTRL_Enable_NocRisc,
    ),
    (
        "RISC_PREFETCH_CTRL_Enable_Trisc",
        global::RISC_PREFETCH_CTRL_Enable_Trisc,
    ),
    (
        "RISC_PREFETCH_CTRL_Max_Req_Count",
        global::RISC_PREFETCH_CTRL_Max_Req_Count,
    ),
    ("SCRATCH_SEC0_val", global::SCRATCH_SEC0_val),
    ("SCRATCH_SEC1_val", global::SCRATCH_SEC1_val),
    ("SCRATCH_SEC2_val", global::SCRATCH_SEC2_val),
    (
        "SRC_ACCESS_CFG_disable_contig_srca_dvalid_phase",
        global::SRC_ACCESS_CFG_disable_contig_srca_dvalid_phase,
    ),
    (
        "SRC_ACCESS_CFG_disable_contig_srcb_dvalid_phase",
        global::SRC_ACCESS_CFG_disable_contig_srcb_dvalid_phase,
    ),
    (
        "SRC_ACCESS_CFG_math_view_srca_as_one_bank",
        global::SRC_ACCESS_CFG_math_view_srca_as_one_bank,
    ),
    (
        "SRC_ACCESS_CFG_math_view_srcb_as_one_bank",
        global::SRC_ACCESS_CFG_math_view_srcb_as_one_bank,
    ),
    ("TRISC_END_PC_SEC0_PC", global::TRISC_END_PC_SEC0_PC),
    ("TRISC_END_PC_SEC1_PC", global::TRISC_END_PC_SEC1_PC),
    ("TRISC_END_PC_SEC2_PC", global::TRISC_END_PC_SEC2_PC),
];

/// Every `Config` bitfield in the `pack0` section.
pub static PACK0_FIELDS: &[(&str, ConfigField)] = &[
    (
        "PACK_CONCAT_MASK_SEC0_pack_concat_mask",
        pack0::PACK_CONCAT_MASK_SEC0_pack_concat_mask,
    ),
    (
        "PACK_CONCAT_MASK_SEC1_pack_concat_mask",
        pack0::PACK_CONCAT_MASK_SEC1_pack_concat_mask,
    ),
    (
        "PACK_CONCAT_MASK_SEC2_pack_concat_mask",
        pack0::PACK_CONCAT_MASK_SEC2_pack_concat_mask,
    ),
    (
        "PACK_CONCAT_MASK_SEC3_pack_concat_mask",
        pack0::PACK_CONCAT_MASK_SEC3_pack_concat_mask,
    ),
    (
        "PACK_COUNTERS_SEC0_auto_ctxt_inc_xys_cnt",
        pack0::PACK_COUNTERS_SEC0_auto_ctxt_inc_xys_cnt,
    ),
    (
        "PACK_COUNTERS_SEC0_pack_per_xy_plane",
        pack0::PACK_COUNTERS_SEC0_pack_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC0_pack_reads_per_xy_plane",
        pack0::PACK_COUNTERS_SEC0_pack_reads_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC0_pack_xys_per_tile",
        pack0::PACK_COUNTERS_SEC0_pack_xys_per_tile,
    ),
    (
        "PACK_COUNTERS_SEC0_pack_yz_transposed",
        pack0::PACK_COUNTERS_SEC0_pack_yz_transposed,
    ),
    (
        "PACK_COUNTERS_SEC1_auto_ctxt_inc_xys_cnt",
        pack0::PACK_COUNTERS_SEC1_auto_ctxt_inc_xys_cnt,
    ),
    (
        "PACK_COUNTERS_SEC1_pack_per_xy_plane",
        pack0::PACK_COUNTERS_SEC1_pack_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC1_pack_reads_per_xy_plane",
        pack0::PACK_COUNTERS_SEC1_pack_reads_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC1_pack_xys_per_tile",
        pack0::PACK_COUNTERS_SEC1_pack_xys_per_tile,
    ),
    (
        "PACK_COUNTERS_SEC1_pack_yz_transposed",
        pack0::PACK_COUNTERS_SEC1_pack_yz_transposed,
    ),
    (
        "PACK_COUNTERS_SEC2_auto_ctxt_inc_xys_cnt",
        pack0::PACK_COUNTERS_SEC2_auto_ctxt_inc_xys_cnt,
    ),
    (
        "PACK_COUNTERS_SEC2_pack_per_xy_plane",
        pack0::PACK_COUNTERS_SEC2_pack_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC2_pack_reads_per_xy_plane",
        pack0::PACK_COUNTERS_SEC2_pack_reads_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC2_pack_xys_per_tile",
        pack0::PACK_COUNTERS_SEC2_pack_xys_per_tile,
    ),
    (
        "PACK_COUNTERS_SEC2_pack_yz_transposed",
        pack0::PACK_COUNTERS_SEC2_pack_yz_transposed,
    ),
    (
        "PACK_COUNTERS_SEC3_auto_ctxt_inc_xys_cnt",
        pack0::PACK_COUNTERS_SEC3_auto_ctxt_inc_xys_cnt,
    ),
    (
        "PACK_COUNTERS_SEC3_pack_per_xy_plane",
        pack0::PACK_COUNTERS_SEC3_pack_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC3_pack_reads_per_xy_plane",
        pack0::PACK_COUNTERS_SEC3_pack_reads_per_xy_plane,
    ),
    (
        "PACK_COUNTERS_SEC3_pack_xys_per_tile",
        pack0::PACK_COUNTERS_SEC3_pack_xys_per_tile,
    ),
    (
        "PACK_COUNTERS_SEC3_pack_yz_transposed",
        pack0::PACK_COUNTERS_SEC3_pack_yz_transposed,
    ),
    (
        "PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain",
        pack0::PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain,
    ),
    (
        "PCK0_ADDR_BASE_REG_0_Base",
        pack0::PCK0_ADDR_BASE_REG_0_Base,
    ),
    (
        "PCK0_ADDR_BASE_REG_1_Base",
        pack0::PCK0_ADDR_BASE_REG_1_Base,
    ),
    (
        "PCK0_ADDR_CTRL_XY_REG_0_Xstride",
        pack0::PCK0_ADDR_CTRL_XY_REG_0_Xstride,
    ),
    (
        "PCK0_ADDR_CTRL_XY_REG_0_Ystride",
        pack0::PCK0_ADDR_CTRL_XY_REG_0_Ystride,
    ),
    (
        "PCK0_ADDR_CTRL_XY_REG_1_Xstride",
        pack0::PCK0_ADDR_CTRL_XY_REG_1_Xstride,
    ),
    (
        "PCK0_ADDR_CTRL_XY_REG_1_Ystride",
        pack0::PCK0_ADDR_CTRL_XY_REG_1_Ystride,
    ),
    (
        "PCK0_ADDR_CTRL_ZW_REG_0_Wstride",
        pack0::PCK0_ADDR_CTRL_ZW_REG_0_Wstride,
    ),
    (
        "PCK0_ADDR_CTRL_ZW_REG_0_Zstride",
        pack0::PCK0_ADDR_CTRL_ZW_REG_0_Zstride,
    ),
    (
        "PCK0_ADDR_CTRL_ZW_REG_1_Wstride",
        pack0::PCK0_ADDR_CTRL_ZW_REG_1_Wstride,
    ),
    (
        "PCK0_ADDR_CTRL_ZW_REG_1_Zstride",
        pack0::PCK0_ADDR_CTRL_ZW_REG_1_Zstride,
    ),
    (
        "PCK_DEST_RD_CTRL_Read_32b_data",
        pack0::PCK_DEST_RD_CTRL_Read_32b_data,
    ),
    (
        "PCK_DEST_RD_CTRL_Read_int8",
        pack0::PCK_DEST_RD_CTRL_Read_int8,
    ),
    (
        "PCK_DEST_RD_CTRL_Read_unsigned",
        pack0::PCK_DEST_RD_CTRL_Read_unsigned,
    ),
    (
        "PCK_DEST_RD_CTRL_Round_10b_mant",
        pack0::PCK_DEST_RD_CTRL_Round_10b_mant,
    ),
    ("PCK_EDGE_MODE_mode", pack0::PCK_EDGE_MODE_mode),
    (
        "PCK_EDGE_OFFSET_SEC0_mask",
        pack0::PCK_EDGE_OFFSET_SEC0_mask,
    ),
    (
        "PCK_EDGE_OFFSET_SEC1_mask",
        pack0::PCK_EDGE_OFFSET_SEC1_mask,
    ),
    (
        "PCK_EDGE_OFFSET_SEC2_mask",
        pack0::PCK_EDGE_OFFSET_SEC2_mask,
    ),
    (
        "PCK_EDGE_OFFSET_SEC3_mask",
        pack0::PCK_EDGE_OFFSET_SEC3_mask,
    ),
    (
        "PCK_EDGE_TILE_FACE_SET_SELECT_enable",
        pack0::PCK_EDGE_TILE_FACE_SET_SELECT_enable,
    ),
    (
        "PCK_EDGE_TILE_FACE_SET_SELECT_select",
        pack0::PCK_EDGE_TILE_FACE_SET_SELECT_select,
    ),
    (
        "PCK_EDGE_TILE_ROW_SET_SELECT_select",
        pack0::PCK_EDGE_TILE_ROW_SET_SELECT_select,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_0",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_0,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_1",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_1,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_10",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_10,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_11",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_11,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_12",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_12,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_13",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_13,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_14",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_14,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_15",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_15,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_2",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_2,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_3",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_3,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_4",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_4,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_5",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_5,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_6",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_6,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_7",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_7,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_8",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_8,
    ),
    (
        "TILE_FACE_SET_MAPPING_0_face_set_mapping_9",
        pack0::TILE_FACE_SET_MAPPING_0_face_set_mapping_9,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_0",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_0,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_1",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_1,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_10",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_10,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_11",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_11,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_12",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_12,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_13",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_13,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_14",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_14,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_15",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_15,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_2",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_2,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_3",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_3,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_4",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_4,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_5",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_5,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_6",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_6,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_7",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_7,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_8",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_8,
    ),
    (
        "TILE_FACE_SET_MAPPING_1_face_set_mapping_9",
        pack0::TILE_FACE_SET_MAPPING_1_face_set_mapping_9,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_0",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_0,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_1",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_1,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_10",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_10,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_11",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_11,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_12",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_12,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_13",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_13,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_14",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_14,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_15",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_15,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_2",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_2,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_3",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_3,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_4",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_4,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_5",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_5,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_6",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_6,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_7",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_7,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_8",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_8,
    ),
    (
        "TILE_FACE_SET_MAPPING_2_face_set_mapping_9",
        pack0::TILE_FACE_SET_MAPPING_2_face_set_mapping_9,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_0",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_0,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_1",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_1,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_10",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_10,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_11",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_11,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_12",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_12,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_13",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_13,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_14",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_14,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_15",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_15,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_2",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_2,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_3",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_3,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_4",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_4,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_5",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_5,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_6",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_6,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_7",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_7,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_8",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_8,
    ),
    (
        "TILE_FACE_SET_MAPPING_3_face_set_mapping_9",
        pack0::TILE_FACE_SET_MAPPING_3_face_set_mapping_9,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_0",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_0,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_1",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_1,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_10",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_10,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_11",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_11,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_12",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_12,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_13",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_13,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_14",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_14,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_15",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_15,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_2",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_2,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_3",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_3,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_4",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_4,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_5",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_5,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_6",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_6,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_7",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_7,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_8",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_8,
    ),
    (
        "TILE_ROW_SET_MAPPING_0_row_set_mapping_9",
        pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_9,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_0",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_0,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_1",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_1,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_10",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_10,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_11",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_11,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_12",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_12,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_13",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_13,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_14",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_14,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_15",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_15,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_2",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_2,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_3",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_3,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_4",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_4,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_5",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_5,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_6",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_6,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_7",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_7,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_8",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_8,
    ),
    (
        "TILE_ROW_SET_MAPPING_1_row_set_mapping_9",
        pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_9,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_0",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_0,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_1",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_1,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_10",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_10,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_11",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_11,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_12",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_12,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_13",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_13,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_14",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_14,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_15",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_15,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_2",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_2,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_3",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_3,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_4",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_4,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_5",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_5,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_6",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_6,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_7",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_7,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_8",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_8,
    ),
    (
        "TILE_ROW_SET_MAPPING_2_row_set_mapping_9",
        pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_9,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_0",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_0,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_1",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_1,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_10",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_10,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_11",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_11,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_12",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_12,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_13",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_13,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_14",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_14,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_15",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_15,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_2",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_2,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_3",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_3,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_4",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_4,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_5",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_5,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_6",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_6,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_7",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_7,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_8",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_8,
    ),
    (
        "TILE_ROW_SET_MAPPING_3_row_set_mapping_9",
        pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_9,
    ),
];

/// Every `Config` bitfield in the `thcon` section.
pub static THCON_FIELDS: &[(&str, ConfigField)] = &[
    (
        "THCON_SEC0_REG10_Packer_Reg_Wr_Addr",
        thcon::THCON_SEC0_REG10_Packer_Reg_Wr_Addr,
    ),
    (
        "THCON_SEC0_REG10_Unpack_fifo_size",
        thcon::THCON_SEC0_REG10_Unpack_fifo_size,
    ),
    (
        "THCON_SEC0_REG10_Unpack_limit_address",
        thcon::THCON_SEC0_REG10_Unpack_limit_address,
    ),
    (
        "THCON_SEC0_REG10_Unpack_limit_address_en",
        thcon::THCON_SEC0_REG10_Unpack_limit_address_en,
    ),
    (
        "THCON_SEC0_REG10_Unpacker_Reg_Wr_Addr",
        thcon::THCON_SEC0_REG10_Unpacker_Reg_Wr_Addr,
    ),
    (
        "THCON_SEC0_REG11_Metadata_cntxt_switch_unpacr_count",
        thcon::THCON_SEC0_REG11_Metadata_cntxt_switch_unpacr_count,
    ),
    (
        "THCON_SEC0_REG11_Metadata_fifo_size",
        thcon::THCON_SEC0_REG11_Metadata_fifo_size,
    ),
    (
        "THCON_SEC0_REG11_Metadata_l1_addr",
        thcon::THCON_SEC0_REG11_Metadata_l1_addr,
    ),
    (
        "THCON_SEC0_REG11_Metadata_limit_addr",
        thcon::THCON_SEC0_REG11_Metadata_limit_addr,
    ),
    (
        "THCON_SEC0_REG11_Metadata_z_cntr_rst_unpacr_count",
        thcon::THCON_SEC0_REG11_Metadata_z_cntr_rst_unpacr_count,
    ),
    (
        "THCON_SEC0_REG1_Add_l1_dest_addr_offset",
        thcon::THCON_SEC0_REG1_Add_l1_dest_addr_offset,
    ),
    (
        "THCON_SEC0_REG1_Add_tile_header_size",
        thcon::THCON_SEC0_REG1_Add_tile_header_size,
    ),
    (
        "THCON_SEC0_REG1_All_pack_disable_zero_compress_ovrd",
        thcon::THCON_SEC0_REG1_All_pack_disable_zero_compress_ovrd,
    ),
    (
        "THCON_SEC0_REG1_Auto_set_last_pacr_intf_sel",
        thcon::THCON_SEC0_REG1_Auto_set_last_pacr_intf_sel,
    ),
    (
        "THCON_SEC0_REG1_Dis_shared_exp_assembler",
        thcon::THCON_SEC0_REG1_Dis_shared_exp_assembler,
    ),
    (
        "THCON_SEC0_REG1_Disable_pack_zero_flags",
        thcon::THCON_SEC0_REG1_Disable_pack_zero_flags,
    ),
    (
        "THCON_SEC0_REG1_Disable_zero_compress",
        thcon::THCON_SEC0_REG1_Disable_zero_compress,
    ),
    (
        "THCON_SEC0_REG1_Downsample_mask",
        thcon::THCON_SEC0_REG1_Downsample_mask,
    ),
    (
        "THCON_SEC0_REG1_Downsample_rate",
        thcon::THCON_SEC0_REG1_Downsample_rate,
    ),
    (
        "THCON_SEC0_REG1_Enable_out_fifo",
        thcon::THCON_SEC0_REG1_Enable_out_fifo,
    ),
    (
        "THCON_SEC0_REG1_Exp_section_size",
        thcon::THCON_SEC0_REG1_Exp_section_size,
    ),
    (
        "THCON_SEC0_REG1_Exp_threshold",
        thcon::THCON_SEC0_REG1_Exp_threshold,
    ),
    (
        "THCON_SEC0_REG1_Exp_threshold_en",
        thcon::THCON_SEC0_REG1_Exp_threshold_en,
    ),
    (
        "THCON_SEC0_REG1_In_data_format",
        thcon::THCON_SEC0_REG1_In_data_format,
    ),
    (
        "THCON_SEC0_REG1_L1_Dest_addr",
        thcon::THCON_SEC0_REG1_L1_Dest_addr,
    ),
    (
        "THCON_SEC0_REG1_L1_source_addr",
        thcon::THCON_SEC0_REG1_L1_source_addr,
    ),
    (
        "THCON_SEC0_REG1_Out_data_format",
        thcon::THCON_SEC0_REG1_Out_data_format,
    ),
    (
        "THCON_SEC0_REG1_Pac_LF8_4b_exp",
        thcon::THCON_SEC0_REG1_Pac_LF8_4b_exp,
    ),
    (
        "THCON_SEC0_REG1_Pack_L1_Acc",
        thcon::THCON_SEC0_REG1_Pack_L1_Acc,
    ),
    (
        "THCON_SEC0_REG1_Row_start_section_size",
        thcon::THCON_SEC0_REG1_Row_start_section_size,
    ),
    (
        "THCON_SEC0_REG1_Source_interface_selection",
        thcon::THCON_SEC0_REG1_Source_interface_selection,
    ),
    (
        "THCON_SEC0_REG1_Sub_l1_tile_header_size",
        thcon::THCON_SEC0_REG1_Sub_l1_tile_header_size,
    ),
    (
        "THCON_SEC0_REG1_Unp_LF8_4b_exp",
        thcon::THCON_SEC0_REG1_Unp_LF8_4b_exp,
    ),
    (
        "THCON_SEC0_REG1_ovrd_default_throttle_mode",
        thcon::THCON_SEC0_REG1_ovrd_default_throttle_mode,
    ),
    (
        "THCON_SEC0_REG1_pack_dis_y_pos_start_offset",
        thcon::THCON_SEC0_REG1_pack_dis_y_pos_start_offset,
    ),
    (
        "THCON_SEC0_REG1_pack_start_intf_pos",
        thcon::THCON_SEC0_REG1_pack_start_intf_pos,
    ),
    (
        "THCON_SEC0_REG2_Context_count",
        thcon::THCON_SEC0_REG2_Context_count,
    ),
    (
        "THCON_SEC0_REG2_Context_count_non_log2",
        thcon::THCON_SEC0_REG2_Context_count_non_log2,
    ),
    (
        "THCON_SEC0_REG2_Context_count_non_log2_en",
        thcon::THCON_SEC0_REG2_Context_count_non_log2_en,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx0",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx0,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx1",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx1,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx2",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx2,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx3",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx3,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx4",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx4,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx5",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx5,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx6",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx6,
    ),
    (
        "THCON_SEC0_REG2_Disable_zero_compress_cntx7",
        thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx7,
    ),
    (
        "THCON_SEC0_REG2_Force_shared_exp",
        thcon::THCON_SEC0_REG2_Force_shared_exp,
    ),
    (
        "THCON_SEC0_REG2_Haloize_mode",
        thcon::THCON_SEC0_REG2_Haloize_mode,
    ),
    (
        "THCON_SEC0_REG2_Metadata_x_end",
        thcon::THCON_SEC0_REG2_Metadata_x_end,
    ),
    (
        "THCON_SEC0_REG2_Out_data_format",
        thcon::THCON_SEC0_REG2_Out_data_format,
    ),
    (
        "THCON_SEC0_REG2_Ovrd_data_format",
        thcon::THCON_SEC0_REG2_Ovrd_data_format,
    ),
    (
        "THCON_SEC0_REG2_Shift_amount_cntx0",
        thcon::THCON_SEC0_REG2_Shift_amount_cntx0,
    ),
    (
        "THCON_SEC0_REG2_Shift_amount_cntx1",
        thcon::THCON_SEC0_REG2_Shift_amount_cntx1,
    ),
    (
        "THCON_SEC0_REG2_Shift_amount_cntx2",
        thcon::THCON_SEC0_REG2_Shift_amount_cntx2,
    ),
    (
        "THCON_SEC0_REG2_Shift_amount_cntx3",
        thcon::THCON_SEC0_REG2_Shift_amount_cntx3,
    ),
    (
        "THCON_SEC0_REG2_Throttle_mode",
        thcon::THCON_SEC0_REG2_Throttle_mode,
    ),
    (
        "THCON_SEC0_REG2_Tileize_mode",
        thcon::THCON_SEC0_REG2_Tileize_mode,
    ),
    (
        "THCON_SEC0_REG2_Unpack_If_Sel",
        thcon::THCON_SEC0_REG2_Unpack_If_Sel,
    ),
    (
        "THCON_SEC0_REG2_Unpack_Src_Reg_Set_Upd",
        thcon::THCON_SEC0_REG2_Unpack_Src_Reg_Set_Upd,
    ),
    (
        "THCON_SEC0_REG2_Unpack_fifo_size",
        thcon::THCON_SEC0_REG2_Unpack_fifo_size,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx0",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx1",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx1,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx2",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx2,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx3",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx3,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx4",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx4,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx5",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx5,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx6",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx6,
    ),
    (
        "THCON_SEC0_REG2_Unpack_if_sel_cntx7",
        thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx7,
    ),
    (
        "THCON_SEC0_REG2_Unpack_limit_address",
        thcon::THCON_SEC0_REG2_Unpack_limit_address,
    ),
    (
        "THCON_SEC0_REG2_Upsample_and_interleave",
        thcon::THCON_SEC0_REG2_Upsample_and_interleave,
    ),
    (
        "THCON_SEC0_REG2_Upsample_rate",
        thcon::THCON_SEC0_REG2_Upsample_rate,
    ),
    (
        "THCON_SEC0_REG3_Base_address",
        thcon::THCON_SEC0_REG3_Base_address,
    ),
    (
        "THCON_SEC0_REG3_Base_cntx1_address",
        thcon::THCON_SEC0_REG3_Base_cntx1_address,
    ),
    (
        "THCON_SEC0_REG3_Base_cntx2_address",
        thcon::THCON_SEC0_REG3_Base_cntx2_address,
    ),
    (
        "THCON_SEC0_REG3_Base_cntx3_address",
        thcon::THCON_SEC0_REG3_Base_cntx3_address,
    ),
    (
        "THCON_SEC0_REG4_Base_cntx4_address",
        thcon::THCON_SEC0_REG4_Base_cntx4_address,
    ),
    (
        "THCON_SEC0_REG4_Base_cntx5_address",
        thcon::THCON_SEC0_REG4_Base_cntx5_address,
    ),
    (
        "THCON_SEC0_REG4_Base_cntx6_address",
        thcon::THCON_SEC0_REG4_Base_cntx6_address,
    ),
    (
        "THCON_SEC0_REG4_Base_cntx7_address",
        thcon::THCON_SEC0_REG4_Base_cntx7_address,
    ),
    (
        "THCON_SEC0_REG5_Dest_cntx0_address",
        thcon::THCON_SEC0_REG5_Dest_cntx0_address,
    ),
    (
        "THCON_SEC0_REG5_Dest_cntx1_address",
        thcon::THCON_SEC0_REG5_Dest_cntx1_address,
    ),
    (
        "THCON_SEC0_REG5_Dest_cntx2_address",
        thcon::THCON_SEC0_REG5_Dest_cntx2_address,
    ),
    (
        "THCON_SEC0_REG5_Dest_cntx3_address",
        thcon::THCON_SEC0_REG5_Dest_cntx3_address,
    ),
    (
        "THCON_SEC0_REG5_Tile_x_dim_cntx0",
        thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0,
    ),
    (
        "THCON_SEC0_REG5_Tile_x_dim_cntx1",
        thcon::THCON_SEC0_REG5_Tile_x_dim_cntx1,
    ),
    (
        "THCON_SEC0_REG5_Tile_x_dim_cntx2",
        thcon::THCON_SEC0_REG5_Tile_x_dim_cntx2,
    ),
    (
        "THCON_SEC0_REG5_Tile_x_dim_cntx3",
        thcon::THCON_SEC0_REG5_Tile_x_dim_cntx3,
    ),
    (
        "THCON_SEC0_REG6_Buffer_size",
        thcon::THCON_SEC0_REG6_Buffer_size,
    ),
    (
        "THCON_SEC0_REG6_Destination_address",
        thcon::THCON_SEC0_REG6_Destination_address,
    ),
    (
        "THCON_SEC0_REG6_Metadata_misc",
        thcon::THCON_SEC0_REG6_Metadata_misc,
    ),
    (
        "THCON_SEC0_REG6_Source_address",
        thcon::THCON_SEC0_REG6_Source_address,
    ),
    (
        "THCON_SEC0_REG6_Transfer_direction",
        thcon::THCON_SEC0_REG6_Transfer_direction,
    ),
    (
        "THCON_SEC0_REG7_Offset_address",
        thcon::THCON_SEC0_REG7_Offset_address,
    ),
    (
        "THCON_SEC0_REG7_Offset_cntx1_address",
        thcon::THCON_SEC0_REG7_Offset_cntx1_address,
    ),
    (
        "THCON_SEC0_REG7_Offset_cntx2_address",
        thcon::THCON_SEC0_REG7_Offset_cntx2_address,
    ),
    (
        "THCON_SEC0_REG7_Offset_cntx3_address",
        thcon::THCON_SEC0_REG7_Offset_cntx3_address,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx0",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx0,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx1",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx1,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx2",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx2,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx3",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx3,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx4",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx4,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx5",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx5,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx6",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx6,
    ),
    (
        "THCON_SEC0_REG7_Unpack_data_format_cntx7",
        thcon::THCON_SEC0_REG7_Unpack_data_format_cntx7,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx0",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx0,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx1",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx1,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx2",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx2,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx3",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx3,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx4",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx4,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx5",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx5,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx6",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx6,
    ),
    (
        "THCON_SEC0_REG7_Unpack_out_data_format_cntx7",
        thcon::THCON_SEC0_REG7_Unpack_out_data_format_cntx7,
    ),
    (
        "THCON_SEC0_REG8_Add_l1_dest_addr_offset",
        thcon::THCON_SEC0_REG8_Add_l1_dest_addr_offset,
    ),
    (
        "THCON_SEC0_REG8_Add_tile_header_size",
        thcon::THCON_SEC0_REG8_Add_tile_header_size,
    ),
    (
        "THCON_SEC0_REG8_Auto_set_last_pacr_intf_sel",
        thcon::THCON_SEC0_REG8_Auto_set_last_pacr_intf_sel,
    ),
    (
        "THCON_SEC0_REG8_Dis_shared_exp_assembler",
        thcon::THCON_SEC0_REG8_Dis_shared_exp_assembler,
    ),
    (
        "THCON_SEC0_REG8_Disable_pack_zero_flags",
        thcon::THCON_SEC0_REG8_Disable_pack_zero_flags,
    ),
    (
        "THCON_SEC0_REG8_Disable_zero_compress",
        thcon::THCON_SEC0_REG8_Disable_zero_compress,
    ),
    (
        "THCON_SEC0_REG8_Downsample_mask",
        thcon::THCON_SEC0_REG8_Downsample_mask,
    ),
    (
        "THCON_SEC0_REG8_Downsample_rate",
        thcon::THCON_SEC0_REG8_Downsample_rate,
    ),
    (
        "THCON_SEC0_REG8_Enable_out_fifo",
        thcon::THCON_SEC0_REG8_Enable_out_fifo,
    ),
    (
        "THCON_SEC0_REG8_Exp_section_size",
        thcon::THCON_SEC0_REG8_Exp_section_size,
    ),
    (
        "THCON_SEC0_REG8_Exp_threshold",
        thcon::THCON_SEC0_REG8_Exp_threshold,
    ),
    (
        "THCON_SEC0_REG8_Exp_threshold_en",
        thcon::THCON_SEC0_REG8_Exp_threshold_en,
    ),
    (
        "THCON_SEC0_REG8_In_data_format",
        thcon::THCON_SEC0_REG8_In_data_format,
    ),
    (
        "THCON_SEC0_REG8_L1_Dest_addr",
        thcon::THCON_SEC0_REG8_L1_Dest_addr,
    ),
    (
        "THCON_SEC0_REG8_L1_source_addr",
        thcon::THCON_SEC0_REG8_L1_source_addr,
    ),
    (
        "THCON_SEC0_REG8_Out_data_format",
        thcon::THCON_SEC0_REG8_Out_data_format,
    ),
    (
        "THCON_SEC0_REG8_Pack_L1_Acc",
        thcon::THCON_SEC0_REG8_Pack_L1_Acc,
    ),
    (
        "THCON_SEC0_REG8_Row_start_section_size",
        thcon::THCON_SEC0_REG8_Row_start_section_size,
    ),
    (
        "THCON_SEC0_REG8_Source_interface_selection",
        thcon::THCON_SEC0_REG8_Source_interface_selection,
    ),
    (
        "THCON_SEC0_REG8_Sub_l1_tile_header_size",
        thcon::THCON_SEC0_REG8_Sub_l1_tile_header_size,
    ),
    ("THCON_SEC0_REG8_Unused1", thcon::THCON_SEC0_REG8_Unused1),
    (
        "THCON_SEC0_REG8_pack_dis_y_pos_start_offset",
        thcon::THCON_SEC0_REG8_pack_dis_y_pos_start_offset,
    ),
    (
        "THCON_SEC0_REG8_unpack_tile_offset",
        thcon::THCON_SEC0_REG8_unpack_tile_offset,
    ),
    (
        "THCON_SEC0_REG9_Pack_0_2_fifo_size",
        thcon::THCON_SEC0_REG9_Pack_0_2_fifo_size,
    ),
    (
        "THCON_SEC0_REG9_Pack_0_2_limit_address",
        thcon::THCON_SEC0_REG9_Pack_0_2_limit_address,
    ),
    (
        "THCON_SEC0_REG9_Pack_1_3_fifo_size",
        thcon::THCON_SEC0_REG9_Pack_1_3_fifo_size,
    ),
    (
        "THCON_SEC0_REG9_Pack_1_3_limit_address",
        thcon::THCON_SEC0_REG9_Pack_1_3_limit_address,
    ),
    (
        "THCON_SEC1_REG10_Packer_Reg_Wr_Addr",
        thcon::THCON_SEC1_REG10_Packer_Reg_Wr_Addr,
    ),
    (
        "THCON_SEC1_REG10_Unpack_fifo_size",
        thcon::THCON_SEC1_REG10_Unpack_fifo_size,
    ),
    (
        "THCON_SEC1_REG10_Unpack_limit_address",
        thcon::THCON_SEC1_REG10_Unpack_limit_address,
    ),
    (
        "THCON_SEC1_REG10_Unpack_limit_address_en",
        thcon::THCON_SEC1_REG10_Unpack_limit_address_en,
    ),
    (
        "THCON_SEC1_REG10_Unpacker_Reg_Wr_Addr",
        thcon::THCON_SEC1_REG10_Unpacker_Reg_Wr_Addr,
    ),
    (
        "THCON_SEC1_REG11_Metadata_cntxt_switch_unpacr_count",
        thcon::THCON_SEC1_REG11_Metadata_cntxt_switch_unpacr_count,
    ),
    (
        "THCON_SEC1_REG11_Metadata_fifo_size",
        thcon::THCON_SEC1_REG11_Metadata_fifo_size,
    ),
    (
        "THCON_SEC1_REG11_Metadata_l1_addr",
        thcon::THCON_SEC1_REG11_Metadata_l1_addr,
    ),
    (
        "THCON_SEC1_REG11_Metadata_limit_addr",
        thcon::THCON_SEC1_REG11_Metadata_limit_addr,
    ),
    (
        "THCON_SEC1_REG11_Metadata_z_cntr_rst_unpacr_count",
        thcon::THCON_SEC1_REG11_Metadata_z_cntr_rst_unpacr_count,
    ),
    (
        "THCON_SEC1_REG1_Add_l1_dest_addr_offset",
        thcon::THCON_SEC1_REG1_Add_l1_dest_addr_offset,
    ),
    (
        "THCON_SEC1_REG1_Add_tile_header_size",
        thcon::THCON_SEC1_REG1_Add_tile_header_size,
    ),
    (
        "THCON_SEC1_REG1_All_pack_disable_zero_compress_ovrd",
        thcon::THCON_SEC1_REG1_All_pack_disable_zero_compress_ovrd,
    ),
    (
        "THCON_SEC1_REG1_Auto_set_last_pacr_intf_sel",
        thcon::THCON_SEC1_REG1_Auto_set_last_pacr_intf_sel,
    ),
    (
        "THCON_SEC1_REG1_Dis_shared_exp_assembler",
        thcon::THCON_SEC1_REG1_Dis_shared_exp_assembler,
    ),
    (
        "THCON_SEC1_REG1_Disable_pack_zero_flags",
        thcon::THCON_SEC1_REG1_Disable_pack_zero_flags,
    ),
    (
        "THCON_SEC1_REG1_Disable_zero_compress",
        thcon::THCON_SEC1_REG1_Disable_zero_compress,
    ),
    (
        "THCON_SEC1_REG1_Downsample_mask",
        thcon::THCON_SEC1_REG1_Downsample_mask,
    ),
    (
        "THCON_SEC1_REG1_Downsample_rate",
        thcon::THCON_SEC1_REG1_Downsample_rate,
    ),
    (
        "THCON_SEC1_REG1_Enable_out_fifo",
        thcon::THCON_SEC1_REG1_Enable_out_fifo,
    ),
    (
        "THCON_SEC1_REG1_Exp_section_size",
        thcon::THCON_SEC1_REG1_Exp_section_size,
    ),
    (
        "THCON_SEC1_REG1_Exp_threshold",
        thcon::THCON_SEC1_REG1_Exp_threshold,
    ),
    (
        "THCON_SEC1_REG1_Exp_threshold_en",
        thcon::THCON_SEC1_REG1_Exp_threshold_en,
    ),
    (
        "THCON_SEC1_REG1_In_data_format",
        thcon::THCON_SEC1_REG1_In_data_format,
    ),
    (
        "THCON_SEC1_REG1_L1_Dest_addr",
        thcon::THCON_SEC1_REG1_L1_Dest_addr,
    ),
    (
        "THCON_SEC1_REG1_L1_source_addr",
        thcon::THCON_SEC1_REG1_L1_source_addr,
    ),
    (
        "THCON_SEC1_REG1_Out_data_format",
        thcon::THCON_SEC1_REG1_Out_data_format,
    ),
    (
        "THCON_SEC1_REG1_Pac_LF8_4b_exp",
        thcon::THCON_SEC1_REG1_Pac_LF8_4b_exp,
    ),
    (
        "THCON_SEC1_REG1_Pack_L1_Acc",
        thcon::THCON_SEC1_REG1_Pack_L1_Acc,
    ),
    (
        "THCON_SEC1_REG1_Row_start_section_size",
        thcon::THCON_SEC1_REG1_Row_start_section_size,
    ),
    (
        "THCON_SEC1_REG1_Source_interface_selection",
        thcon::THCON_SEC1_REG1_Source_interface_selection,
    ),
    (
        "THCON_SEC1_REG1_Sub_l1_tile_header_size",
        thcon::THCON_SEC1_REG1_Sub_l1_tile_header_size,
    ),
    (
        "THCON_SEC1_REG1_Unp_LF8_4b_exp",
        thcon::THCON_SEC1_REG1_Unp_LF8_4b_exp,
    ),
    (
        "THCON_SEC1_REG1_ovrd_default_throttle_mode",
        thcon::THCON_SEC1_REG1_ovrd_default_throttle_mode,
    ),
    (
        "THCON_SEC1_REG1_pack_dis_y_pos_start_offset",
        thcon::THCON_SEC1_REG1_pack_dis_y_pos_start_offset,
    ),
    (
        "THCON_SEC1_REG1_pack_start_intf_pos",
        thcon::THCON_SEC1_REG1_pack_start_intf_pos,
    ),
    (
        "THCON_SEC1_REG2_Context_count",
        thcon::THCON_SEC1_REG2_Context_count,
    ),
    (
        "THCON_SEC1_REG2_Context_count_non_log2",
        thcon::THCON_SEC1_REG2_Context_count_non_log2,
    ),
    (
        "THCON_SEC1_REG2_Context_count_non_log2_en",
        thcon::THCON_SEC1_REG2_Context_count_non_log2_en,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx0",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx0,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx1",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx1,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx2",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx2,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx3",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx3,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx4",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx4,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx5",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx5,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx6",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx6,
    ),
    (
        "THCON_SEC1_REG2_Disable_zero_compress_cntx7",
        thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx7,
    ),
    (
        "THCON_SEC1_REG2_Force_shared_exp",
        thcon::THCON_SEC1_REG2_Force_shared_exp,
    ),
    (
        "THCON_SEC1_REG2_Haloize_mode",
        thcon::THCON_SEC1_REG2_Haloize_mode,
    ),
    (
        "THCON_SEC1_REG2_Metadata_x_end",
        thcon::THCON_SEC1_REG2_Metadata_x_end,
    ),
    (
        "THCON_SEC1_REG2_Out_data_format",
        thcon::THCON_SEC1_REG2_Out_data_format,
    ),
    (
        "THCON_SEC1_REG2_Ovrd_data_format",
        thcon::THCON_SEC1_REG2_Ovrd_data_format,
    ),
    (
        "THCON_SEC1_REG2_Shift_amount_cntx0",
        thcon::THCON_SEC1_REG2_Shift_amount_cntx0,
    ),
    (
        "THCON_SEC1_REG2_Shift_amount_cntx1",
        thcon::THCON_SEC1_REG2_Shift_amount_cntx1,
    ),
    (
        "THCON_SEC1_REG2_Shift_amount_cntx2",
        thcon::THCON_SEC1_REG2_Shift_amount_cntx2,
    ),
    (
        "THCON_SEC1_REG2_Shift_amount_cntx3",
        thcon::THCON_SEC1_REG2_Shift_amount_cntx3,
    ),
    (
        "THCON_SEC1_REG2_Throttle_mode",
        thcon::THCON_SEC1_REG2_Throttle_mode,
    ),
    (
        "THCON_SEC1_REG2_Tileize_mode",
        thcon::THCON_SEC1_REG2_Tileize_mode,
    ),
    (
        "THCON_SEC1_REG2_Unpack_If_Sel",
        thcon::THCON_SEC1_REG2_Unpack_If_Sel,
    ),
    (
        "THCON_SEC1_REG2_Unpack_Src_Reg_Set_Upd",
        thcon::THCON_SEC1_REG2_Unpack_Src_Reg_Set_Upd,
    ),
    (
        "THCON_SEC1_REG2_Unpack_fifo_size",
        thcon::THCON_SEC1_REG2_Unpack_fifo_size,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx0",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx0,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx1",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx1,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx2",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx2,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx3",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx3,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx4",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx4,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx5",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx5,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx6",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx6,
    ),
    (
        "THCON_SEC1_REG2_Unpack_if_sel_cntx7",
        thcon::THCON_SEC1_REG2_Unpack_if_sel_cntx7,
    ),
    (
        "THCON_SEC1_REG2_Unpack_limit_address",
        thcon::THCON_SEC1_REG2_Unpack_limit_address,
    ),
    (
        "THCON_SEC1_REG2_Upsample_and_interleave",
        thcon::THCON_SEC1_REG2_Upsample_and_interleave,
    ),
    (
        "THCON_SEC1_REG2_Upsample_rate",
        thcon::THCON_SEC1_REG2_Upsample_rate,
    ),
    (
        "THCON_SEC1_REG3_Base_address",
        thcon::THCON_SEC1_REG3_Base_address,
    ),
    (
        "THCON_SEC1_REG3_Base_cntx1_address",
        thcon::THCON_SEC1_REG3_Base_cntx1_address,
    ),
    (
        "THCON_SEC1_REG3_Base_cntx2_address",
        thcon::THCON_SEC1_REG3_Base_cntx2_address,
    ),
    (
        "THCON_SEC1_REG3_Base_cntx3_address",
        thcon::THCON_SEC1_REG3_Base_cntx3_address,
    ),
    (
        "THCON_SEC1_REG4_Base_cntx4_address",
        thcon::THCON_SEC1_REG4_Base_cntx4_address,
    ),
    (
        "THCON_SEC1_REG4_Base_cntx5_address",
        thcon::THCON_SEC1_REG4_Base_cntx5_address,
    ),
    (
        "THCON_SEC1_REG4_Base_cntx6_address",
        thcon::THCON_SEC1_REG4_Base_cntx6_address,
    ),
    (
        "THCON_SEC1_REG4_Base_cntx7_address",
        thcon::THCON_SEC1_REG4_Base_cntx7_address,
    ),
    (
        "THCON_SEC1_REG5_Dest_cntx0_address",
        thcon::THCON_SEC1_REG5_Dest_cntx0_address,
    ),
    (
        "THCON_SEC1_REG5_Dest_cntx1_address",
        thcon::THCON_SEC1_REG5_Dest_cntx1_address,
    ),
    (
        "THCON_SEC1_REG5_Dest_cntx2_address",
        thcon::THCON_SEC1_REG5_Dest_cntx2_address,
    ),
    (
        "THCON_SEC1_REG5_Dest_cntx3_address",
        thcon::THCON_SEC1_REG5_Dest_cntx3_address,
    ),
    (
        "THCON_SEC1_REG5_Tile_x_dim_cntx0",
        thcon::THCON_SEC1_REG5_Tile_x_dim_cntx0,
    ),
    (
        "THCON_SEC1_REG5_Tile_x_dim_cntx1",
        thcon::THCON_SEC1_REG5_Tile_x_dim_cntx1,
    ),
    (
        "THCON_SEC1_REG5_Tile_x_dim_cntx2",
        thcon::THCON_SEC1_REG5_Tile_x_dim_cntx2,
    ),
    (
        "THCON_SEC1_REG5_Tile_x_dim_cntx3",
        thcon::THCON_SEC1_REG5_Tile_x_dim_cntx3,
    ),
    (
        "THCON_SEC1_REG6_Buffer_size",
        thcon::THCON_SEC1_REG6_Buffer_size,
    ),
    (
        "THCON_SEC1_REG6_Destination_address",
        thcon::THCON_SEC1_REG6_Destination_address,
    ),
    (
        "THCON_SEC1_REG6_Metadata_misc",
        thcon::THCON_SEC1_REG6_Metadata_misc,
    ),
    (
        "THCON_SEC1_REG6_Source_address",
        thcon::THCON_SEC1_REG6_Source_address,
    ),
    (
        "THCON_SEC1_REG6_Transfer_direction",
        thcon::THCON_SEC1_REG6_Transfer_direction,
    ),
    (
        "THCON_SEC1_REG7_Offset_address",
        thcon::THCON_SEC1_REG7_Offset_address,
    ),
    (
        "THCON_SEC1_REG7_Offset_cntx1_address",
        thcon::THCON_SEC1_REG7_Offset_cntx1_address,
    ),
    (
        "THCON_SEC1_REG7_Offset_cntx2_address",
        thcon::THCON_SEC1_REG7_Offset_cntx2_address,
    ),
    (
        "THCON_SEC1_REG7_Offset_cntx3_address",
        thcon::THCON_SEC1_REG7_Offset_cntx3_address,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx0",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx0,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx1",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx1,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx2",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx2,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx3",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx3,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx4",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx4,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx5",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx5,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx6",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx6,
    ),
    (
        "THCON_SEC1_REG7_Unpack_data_format_cntx7",
        thcon::THCON_SEC1_REG7_Unpack_data_format_cntx7,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx0",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx0,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx1",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx1,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx2",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx2,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx3",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx3,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx4",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx4,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx5",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx5,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx6",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx6,
    ),
    (
        "THCON_SEC1_REG7_Unpack_out_data_format_cntx7",
        thcon::THCON_SEC1_REG7_Unpack_out_data_format_cntx7,
    ),
    (
        "THCON_SEC1_REG8_Add_l1_dest_addr_offset",
        thcon::THCON_SEC1_REG8_Add_l1_dest_addr_offset,
    ),
    (
        "THCON_SEC1_REG8_Add_tile_header_size",
        thcon::THCON_SEC1_REG8_Add_tile_header_size,
    ),
    (
        "THCON_SEC1_REG8_Auto_set_last_pacr_intf_sel",
        thcon::THCON_SEC1_REG8_Auto_set_last_pacr_intf_sel,
    ),
    (
        "THCON_SEC1_REG8_Dis_shared_exp_assembler",
        thcon::THCON_SEC1_REG8_Dis_shared_exp_assembler,
    ),
    (
        "THCON_SEC1_REG8_Disable_pack_zero_flags",
        thcon::THCON_SEC1_REG8_Disable_pack_zero_flags,
    ),
    (
        "THCON_SEC1_REG8_Disable_zero_compress",
        thcon::THCON_SEC1_REG8_Disable_zero_compress,
    ),
    (
        "THCON_SEC1_REG8_Downsample_mask",
        thcon::THCON_SEC1_REG8_Downsample_mask,
    ),
    (
        "THCON_SEC1_REG8_Downsample_rate",
        thcon::THCON_SEC1_REG8_Downsample_rate,
    ),
    (
        "THCON_SEC1_REG8_Enable_out_fifo",
        thcon::THCON_SEC1_REG8_Enable_out_fifo,
    ),
    (
        "THCON_SEC1_REG8_Exp_section_size",
        thcon::THCON_SEC1_REG8_Exp_section_size,
    ),
    (
        "THCON_SEC1_REG8_Exp_threshold",
        thcon::THCON_SEC1_REG8_Exp_threshold,
    ),
    (
        "THCON_SEC1_REG8_Exp_threshold_en",
        thcon::THCON_SEC1_REG8_Exp_threshold_en,
    ),
    (
        "THCON_SEC1_REG8_In_data_format",
        thcon::THCON_SEC1_REG8_In_data_format,
    ),
    (
        "THCON_SEC1_REG8_L1_Dest_addr",
        thcon::THCON_SEC1_REG8_L1_Dest_addr,
    ),
    (
        "THCON_SEC1_REG8_L1_source_addr",
        thcon::THCON_SEC1_REG8_L1_source_addr,
    ),
    (
        "THCON_SEC1_REG8_Out_data_format",
        thcon::THCON_SEC1_REG8_Out_data_format,
    ),
    (
        "THCON_SEC1_REG8_Pack_L1_Acc",
        thcon::THCON_SEC1_REG8_Pack_L1_Acc,
    ),
    (
        "THCON_SEC1_REG8_Row_start_section_size",
        thcon::THCON_SEC1_REG8_Row_start_section_size,
    ),
    (
        "THCON_SEC1_REG8_Source_interface_selection",
        thcon::THCON_SEC1_REG8_Source_interface_selection,
    ),
    (
        "THCON_SEC1_REG8_Sub_l1_tile_header_size",
        thcon::THCON_SEC1_REG8_Sub_l1_tile_header_size,
    ),
    ("THCON_SEC1_REG8_Unused1", thcon::THCON_SEC1_REG8_Unused1),
    (
        "THCON_SEC1_REG8_pack_dis_y_pos_start_offset",
        thcon::THCON_SEC1_REG8_pack_dis_y_pos_start_offset,
    ),
    (
        "THCON_SEC1_REG8_unpack_tile_offset",
        thcon::THCON_SEC1_REG8_unpack_tile_offset,
    ),
    (
        "THCON_SEC1_REG9_Pack_0_2_fifo_size",
        thcon::THCON_SEC1_REG9_Pack_0_2_fifo_size,
    ),
    (
        "THCON_SEC1_REG9_Pack_0_2_limit_address",
        thcon::THCON_SEC1_REG9_Pack_0_2_limit_address,
    ),
    (
        "THCON_SEC1_REG9_Pack_1_3_fifo_size",
        thcon::THCON_SEC1_REG9_Pack_1_3_fifo_size,
    ),
    (
        "THCON_SEC1_REG9_Pack_1_3_limit_address",
        thcon::THCON_SEC1_REG9_Pack_1_3_limit_address,
    ),
];

/// Every `Config` bitfield in the `unpack0` section.
pub static UNPACK0_FIELDS: &[(&str, ConfigField)] = &[
    (
        "UNP0_ADDR_BASE_REG_0_Base",
        unpack0::UNP0_ADDR_BASE_REG_0_Base,
    ),
    (
        "UNP0_ADDR_BASE_REG_1_Base",
        unpack0::UNP0_ADDR_BASE_REG_1_Base,
    ),
    (
        "UNP0_ADDR_CTRL_XY_REG_0_Xstride",
        unpack0::UNP0_ADDR_CTRL_XY_REG_0_Xstride,
    ),
    (
        "UNP0_ADDR_CTRL_XY_REG_0_Ystride",
        unpack0::UNP0_ADDR_CTRL_XY_REG_0_Ystride,
    ),
    (
        "UNP0_ADDR_CTRL_ZW_REG_0_Wstride",
        unpack0::UNP0_ADDR_CTRL_ZW_REG_0_Wstride,
    ),
    (
        "UNP0_ADDR_CTRL_ZW_REG_0_Zstride",
        unpack0::UNP0_ADDR_CTRL_ZW_REG_0_Zstride,
    ),
    (
        "UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr",
        unpack0::UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr,
    ),
    (
        "UNP0_BLOBS_Y_START_CNTX_01_blobs_y_start",
        unpack0::UNP0_BLOBS_Y_START_CNTX_01_blobs_y_start,
    ),
    (
        "UNP0_BLOBS_Y_START_CNTX_23_blobs_y_start",
        unpack0::UNP0_BLOBS_Y_START_CNTX_23_blobs_y_start,
    ),
    (
        "UNP0_FORCED_SHARED_EXP_shared_exp",
        unpack0::UNP0_FORCED_SHARED_EXP_shared_exp,
    ),
    (
        "UNP0_NOP_REG_CLR_VAL_nop_reg_clr_val",
        unpack0::UNP0_NOP_REG_CLR_VAL_nop_reg_clr_val,
    ),
    (
        "UNP1_ADDR_CTRL_XY_REG_0_Xstride",
        unpack0::UNP1_ADDR_CTRL_XY_REG_0_Xstride,
    ),
    (
        "UNP1_ADDR_CTRL_XY_REG_0_Ystride",
        unpack0::UNP1_ADDR_CTRL_XY_REG_0_Ystride,
    ),
    (
        "UNP1_ADDR_CTRL_ZW_REG_0_Wstride",
        unpack0::UNP1_ADDR_CTRL_ZW_REG_0_Wstride,
    ),
    (
        "UNP1_ADDR_CTRL_ZW_REG_0_Zstride",
        unpack0::UNP1_ADDR_CTRL_ZW_REG_0_Zstride,
    ),
];

/// Every `Config` bitfield in the `unpack1` section.
pub static UNPACK1_FIELDS: &[(&str, ConfigField)] = &[
    (
        "UNP0_ADDR_CTRL_XY_REG_1_Xstride",
        unpack1::UNP0_ADDR_CTRL_XY_REG_1_Xstride,
    ),
    (
        "UNP0_ADDR_CTRL_XY_REG_1_Ystride",
        unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride,
    ),
    (
        "UNP0_ADDR_CTRL_ZW_REG_1_Wstride",
        unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Wstride,
    ),
    (
        "UNP0_ADDR_CTRL_ZW_REG_1_Zstride",
        unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride,
    ),
    (
        "UNP1_ADDR_BASE_REG_0_Base",
        unpack1::UNP1_ADDR_BASE_REG_0_Base,
    ),
    (
        "UNP1_ADDR_BASE_REG_1_Base",
        unpack1::UNP1_ADDR_BASE_REG_1_Base,
    ),
    (
        "UNP1_ADDR_CTRL_XY_REG_1_Xstride",
        unpack1::UNP1_ADDR_CTRL_XY_REG_1_Xstride,
    ),
    (
        "UNP1_ADDR_CTRL_XY_REG_1_Ystride",
        unpack1::UNP1_ADDR_CTRL_XY_REG_1_Ystride,
    ),
    (
        "UNP1_ADDR_CTRL_ZW_REG_1_Wstride",
        unpack1::UNP1_ADDR_CTRL_ZW_REG_1_Wstride,
    ),
    (
        "UNP1_ADDR_CTRL_ZW_REG_1_Zstride",
        unpack1::UNP1_ADDR_CTRL_ZW_REG_1_Zstride,
    ),
    (
        "UNP1_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr",
        unpack1::UNP1_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr,
    ),
    (
        "UNP1_FORCED_SHARED_EXP_shared_exp",
        unpack1::UNP1_FORCED_SHARED_EXP_shared_exp,
    ),
    (
        "UNP1_NOP_REG_CLR_VAL_nop_reg_clr_val",
        unpack1::UNP1_NOP_REG_CLR_VAL_nop_reg_clr_val,
    ),
];
