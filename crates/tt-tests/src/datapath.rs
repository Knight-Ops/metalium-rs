//! Configuring the unpacker and the packer for a single flat run of FP32 datums.
//!
//! Shared by every datapath gate, because the configuration is long, fiddly and
//! entirely determined by the functional models -- so a second copy would be a
//! second thing to keep in step with `UNPACR_Regular.md` and the `Packers/` pages.
//!
//! # What this deliberately does not do
//!
//! It configures **one flat run**, not a tiled 32x32 image: `XDim` datums in one
//! row of one plane. That is enough to exercise the datapath and the numerics, and
//! it keeps the address arithmetic small enough to check against the specification
//! by hand. Driving a real `tt_layout::Layout` through it is Phase 6 work, and it
//! is where the `Z`-plane-to-face convention finally gets settled.
//!
//! Everything from `WormholeB0/.../Packers/` is `UNVERIFIED`: that directory has no
//! Blackhole counterpart. Blackhole's own `PACR.md` is self-labelled "basic".

use tt_isa::backend::ConfigWords;
use tt_isa::backend::ThreadConfigEntry;
use tt_isa::cfg::generated::{alu, global, pack0, thcon, thread, unpack1};
use tt_isa::isa::generated::{defs, encode};
use tt_isa::isa::Instruction;
use tt_isa::tile::{L1Format, TileDescriptor, TileImage};

/// Where the input tile image is staged, and where the packer writes.
///
/// Both clear of the firmware at `0x6000` and the mailbox at `0x10_0000`, and
/// 16-byte aligned: the packer writes L1 in 16-byte units (`PACR.md`), and a tile
/// base must be 16-byte aligned anyway (`UNPACR_Regular.md:113`).
pub const STAGE: u64 = 0x2_0000;
pub const OUT: u64 = 0x3_0000;

pub const SCRATCH_GPR: u32 = 8;
/// See `probe_unpack.rs`: the smallest `REG5_Dest_cntx0_address` the unpacker
/// accepts, which is `Dst` row 0.
pub const DST_BASE: u32 = 64;

pub const UNPACR_LAST: u32 = 1;

pub fn flat_descriptor(datums: u32) -> TileDescriptor {
    TileDescriptor::zeroed()
        .with_x_dim(datums)
        .with_y_dim(1)
        // Zero means one (`UNPACR_Regular.md:70-71`), and it keeps words 2 and 3 of
        // the descriptor zero -- which ttsim will not let us write at all.
        .with_z_dim(0)
        .with_w_dim(0)
        .with_is_uncompressed(true)
        .with_in_data_format_raw(L1Format::Fp32.code().expect("FP32's code is measured"))
}

pub fn staged_image(descriptor: TileDescriptor, datums: u32) -> Vec<u8> {
    let image = TileImage::new(descriptor, L1Format::Fp32).unwrap();
    let mut staged = vec![0u8; image.total_bytes()];
    for i in 0..datums as usize {
        let v = 1.0f32 + i as f32;
        let off = image.datum_bit_offset(i) / 8;
        staged[off..off + 4].copy_from_slice(&v.to_bits().to_le_bytes());
    }
    staged
}

pub fn thread_config() -> Vec<Instruction> {
    vec![
        // `SETC16.md`: after coming out of reset this must be executed for some
        // value before anything else relies on the configuration bank.
        ThreadConfigEntry::zeroed(thread::CFG_STATE_ID_StateID.addr32())
            .set(thread::CFG_STATE_ID_StateID, 0)
            .unwrap()
            .encode()
            .unwrap(),
        // `UnpackToDst` with this set is `UndefinedBehavior` (`UNPACR_Regular.md:313`).
        ThreadConfigEntry::zeroed(thread::SRCA_SET_SetOvrdWithAddr.addr32())
            .set(thread::SRCA_SET_SetOvrdWithAddr, 0)
            .unwrap()
            .encode()
            .unwrap(),
        ThreadConfigEntry::zeroed(thread::UNPACK_MISC_CFG_CfgContextOffset_0.addr32())
            .set(thread::UNPACK_MISC_CFG_CfgContextOffset_0, 0)
            .unwrap()
            .encode()
            .unwrap(),
    ]
}

/// `REG3_Base_address` for a tile image staged at `l1_base`.
///
/// `InAddr = (Base_address + 1 + DigestSize) * 16` (`UNPACR_Regular.md:112-115`):
/// the `+ 1` is the unpacker stepping over the tile header, so `Base_address`
/// names the *start of the image, header included* -- which is where
/// [`TileImage`] puts the header. This was once `l1_base / 16 - 1`, subtracting
/// the header a second time: the unpacker then read the header's 16 zero bytes
/// as the first datums and dropped as many from the end. That is the whole of
/// what divergence rows 30 and 35 recorded as a "hidden output base" scaling
/// with the input width -- ttsim, silicon and the specification all agreed, and
/// the error was here.
pub fn tile_base_units(l1_base: u64) -> u32 {
    (l1_base / TileImage::ALIGNMENT as u64) as u32
}

/// Unpacker configuration, as established by `probe_unpack.rs`.
pub fn unpack_config(words: &mut ConfigWords, descriptor: TileDescriptor, l1_base: u64) {
    let base_units = tile_base_units(l1_base);
    words
        .set(thcon::THCON_SEC0_REG3_Base_address, base_units)
        .unwrap()
        .set(thcon::THCON_SEC0_REG7_Offset_address, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0, descriptor.x_dim())
        .unwrap()
        .set(
            thcon::THCON_SEC0_REG2_Out_data_format,
            L1Format::Fp32.code().unwrap(),
        )
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Ovrd_data_format, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0, 1)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx0, 1)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Haloize_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Tileize_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Throttle_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Upsample_rate, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Upsample_and_interleave, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Shift_amount_cntx0, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Force_shared_exp, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG5_Dest_cntx0_address, DST_BASE)
        .unwrap()
        .set(unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, 0)
        .unwrap()
        .set(unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride, 0)
        .unwrap();

    // Only the descriptor words that carry something: ttsim models words 0 and 1 of
    // the span and refuses 2 and 3 (divergence row 29). Both of those are zero here.
    for (i, word) in descriptor.words().iter().enumerate() {
        if *word != 0 {
            words
                .seed(
                    thcon::THCON_SEC0_REG0_TileDescriptor.addr32() + i as u16,
                    *word,
                )
                .unwrap();
        }
    }
}

/// Which unpacker: 0 feeds `SrcA` (or `Dst`), 1 feeds `SrcB` only.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Unpacker {
    SrcA = 0,
    SrcB = 1,
}

/// The `ThreadConfig` the `Src` path needs, in place of [`thread_config`].
///
/// `SRCA_SET_SetOvrdWithAddr` is the reverse of the `Dst` path's setting: with it
/// clear, a `Src` unpack is `UnsupportedFunctionality` (`UNPACR_Regular.md:316`),
/// and ttsim says so by name (`!unpack_to_dst: SRCA_SET_SetOvrdWithAddr=0`). It
/// shares entry 5 with `SRCA_SET_Base`, which stays zero.
pub fn src_thread_config() -> Vec<Instruction> {
    vec![
        ThreadConfigEntry::zeroed(thread::CFG_STATE_ID_StateID.addr32())
            .set(thread::CFG_STATE_ID_StateID, 0)
            .unwrap()
            .encode()
            .unwrap(),
        ThreadConfigEntry::zeroed(thread::SRCA_SET_SetOvrdWithAddr.addr32())
            .set(thread::SRCA_SET_SetOvrdWithAddr, 1)
            .unwrap()
            .encode()
            .unwrap(),
        ThreadConfigEntry::zeroed(thread::UNPACK_MISC_CFG_CfgContextOffset_0.addr32())
            .set(thread::UNPACK_MISC_CFG_CfgContextOffset_0, 0)
            .unwrap()
            .encode()
            .unwrap(),
    ]
}

/// Configure `unpacker` to move FP32 datums from `l1_base` into `SrcA`/`SrcB`,
/// converting to `out` (`UNPACR_Regular.md:495-520`: FP32 in, TF32 or a 16-bit
/// type out; FP32 *into* `Src` is `UndefinedBehavior`).
///
/// Unpacker 1 is configured through far fewer fields than unpacker 0, because ttsim
/// refuses most of `THCON_SEC1_REG5`/`REG7` and `UNP1_ADDR_*` outright
/// (`probe_src::map_the_src_path_configuration_surface`). None of the refused ones
/// is read on this path: unpacker 1 takes `XDim` from the descriptor rather than
/// from `Tile_x_dim_cntx` (`UNPACR_Regular.md:73`), has no `Dest_cntx` term, and a
/// single flat row needs no `Ystride`. `REG7_Offset_address` *is* read and cannot
/// be written, so this relies on its reset value being zero.
pub fn unpack_src_config(
    words: &mut ConfigWords,
    unpacker: Unpacker,
    descriptor: TileDescriptor,
    l1_base: u64,
    out: u32,
) {
    let base_units = tile_base_units(l1_base);
    let descriptor_span = match unpacker {
        Unpacker::SrcA => {
            words
                .set(thcon::THCON_SEC0_REG3_Base_address, base_units)
                .unwrap()
                .set(thcon::THCON_SEC0_REG7_Offset_address, 0)
                .unwrap()
                .set(thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0, descriptor.x_dim())
                .unwrap()
                .set(thcon::THCON_SEC0_REG2_Out_data_format, out)
                .unwrap()
                // `Src`, not `Dst`.
                .set(thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0, 0)
                .unwrap()
                .set(thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx0, 1)
                .unwrap()
                // With `UnpackToDst` clear this *replaces* the output address rather
                // than adding to it (`UNPACR_Regular.md:265-270`); `SrcA` row 0 is
                // `OutAddr / 16 - 4`, so 64 is row 0.
                .set(thcon::THCON_SEC0_REG5_Dest_cntx0_address, DST_BASE)
                .unwrap()
                .set(unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, 0)
                .unwrap()
                .set(unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride, 0)
                .unwrap();
            thcon::THCON_SEC0_REG0_TileDescriptor
        }
        Unpacker::SrcB => {
            words
                .set(thcon::THCON_SEC1_REG3_Base_address, base_units)
                .unwrap()
                .set(thcon::THCON_SEC1_REG2_Out_data_format, out)
                .unwrap()
                .set(thcon::THCON_SEC1_REG2_Disable_zero_compress_cntx0, 1)
                .unwrap()
                .set(unpack1::UNP1_ADDR_CTRL_ZW_REG_1_Zstride, 0)
                .unwrap();
            thcon::THCON_SEC1_REG0_TileDescriptor
        }
    };
    // Every other field of the words touched above is zero by construction, which
    // is what `Ovrd_data_format`, `Haloize_mode`, `Tileize_mode`, `Throttle_mode`,
    // upsampling, `Shift_amount` and `Unpack_Src_Reg_Set_Upd` must be here.

    // Only the non-zero descriptor words: row 29.
    for (i, word) in descriptor.words().iter().enumerate() {
        if *word != 0 {
            words
                .seed(descriptor_span.addr32() + i as u16, *word)
                .unwrap();
        }
    }
}

/// `SETADCXX` for either unpacker's channels.
pub fn set_adc_x(unpacker: Unpacker, first: u32, last: u32) -> Instruction {
    let adc = encode::Setadcxx::ZERO.x0_val(first).x1_val(last);
    match unpacker {
        Unpacker::SrcA => adc.u0(1),
        Unpacker::SrcB => adc.u1(1),
    }
    .encode()
    .unwrap()
}

/// `UNPACR` into `SrcA`/`SrcB`, handing the bank to the Matrix Unit if `flip`.
pub fn unpack_src_instruction(unpacker: Unpacker, flip: bool) -> Instruction {
    let base = encode::UnpacrRegular::ZERO
        .which_unpacker(unpacker as u32)
        .multi_context_mode(1)
        .flip_src(u32::from(flip))
        .encode()
        .unwrap();
    Instruction::new(base.word() | UNPACR_LAST, &defs::UNPACR_Regular)
}

/// Packer configuration: read `Dst` as 32-bit data, write FP32 to `l1_dest`.
pub fn pack_config(words: &mut ConfigWords, l1_dest: u64) {
    // `Packers/OutputAddressGenerator.md`: `Addr = L1_Dest_addr +
    // !Sub_l1_tile_header_size`, and on Blackhole the `YZW_Addr` term is in bytes
    // and gets `>>= 4`, so the whole address is in 16-byte units.
    // The address is `L1_Dest_addr + !Sub_l1_tile_header_size`, so with that field
    // at zero the packer always skips one 16-byte unit. ttsim refuses to let it be
    // written -- `tensix_cfg_wr32: THCON_SEC0_REG1_Sub_l1_tile_header_size`, a
    // named field refusal rather than the `reg=N` the unpacker path gives -- so the
    // `+1` is unavoidable here and gets compensated in the base instead.
    let dest_units = l1_dest / 16 - 1;
    words
        .set(thcon::THCON_SEC0_REG1_L1_Dest_addr, dest_units as u32)
        .unwrap()
        .set(
            thcon::THCON_SEC0_REG1_In_data_format,
            L1Format::Fp32.code().unwrap(),
        )
        .unwrap()
        .set(
            thcon::THCON_SEC0_REG1_Out_data_format,
            L1Format::Fp32.code().unwrap(),
        )
        .unwrap()
        // Input comes from `Dst`, not from L1 (`Packers/InputAddressGenerator.md`:
        // `Source_interface_selection == 1` is the L1 path).
        .set(thcon::THCON_SEC0_REG1_Source_interface_selection, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Disable_zero_compress, 1)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Add_l1_dest_addr_offset, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Enable_out_fifo, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Exp_section_size, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Row_start_section_size, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Downsample_rate, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Downsample_mask, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Exp_threshold_en, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_Pack_L1_Acc, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG1_pack_start_intf_pos, 0)
        .unwrap();

    // The output address generator's Y/Z/W term, all zero for one contiguous run.
    words
        .set(pack0::PCK0_ADDR_BASE_REG_1_Base, 0)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_XY_REG_1_Ystride, 0)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_ZW_REG_1_Zstride, 0)
        .unwrap()
        // The input (`Dst`) side's strides.
        .set(pack0::PCK0_ADDR_CTRL_XY_REG_0_Xstride, 0)
        .unwrap()
        .set(pack0::PCK0_ADDR_CTRL_ZW_REG_0_Zstride, 0)
        .unwrap()
        // 32-bit reads out of `Dst`, matching the FP32 format everywhere else.
        .set(pack0::PCK_DEST_RD_CTRL_Read_32b_data, 1)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_unsigned, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Read_int8, 0)
        .unwrap()
        .set(pack0::PCK_DEST_RD_CTRL_Round_10b_mant, 0)
        .unwrap()
        // `PACR.md`: `TilePosition` returns to zero only when advancing it lands
        // exactly on `pack_reads_per_xy_plane`, so it must match the number of read
        // interfaces each `PACR` enables.
        //
        // Only this one of the three counters is set. ttsim refuses a non-zero
        // `pack_per_xy_plane` or `pack_xys_per_tile` at *configuration* time
        // (`tensix_cfg_wr32: PACK_COUNTERS_SEC0_pack_per_xy_plane` -- a
        // value-dependent refusal on a register whose zero write it accepts), and
        // refuses a *zero* `pack_reads_per_xy_plane` at *execution* time
        // (`tensix_pacr: pack_reads_per_xy_plane=0`). The intersection is exactly
        // this.
        .set(pack0::PACK_COUNTERS_SEC0_pack_reads_per_xy_plane, 4)
        .unwrap()
        // The `Dst` read offset, added by the input address generator.
        .set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, 0)
        .unwrap()
        // The packer edge-masking path, off.
        .set(pack0::PCK_EDGE_MODE_mode, 0)
        .unwrap()
        .set(pack0::PCK_EDGE_OFFSET_SEC0_mask, 0xFFFF)
        .unwrap();

    // `Dst` is read as FP32 by the packer too; `ALU_ACC_CTRL_Fp32_enabled` is what
    // tells the rest of the backend that.
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
}

pub fn set_adc_x_unpack(first: u32, last: u32) -> Instruction {
    encode::Setadcxx::ZERO
        .u0(1)
        .x0_val(first)
        .x1_val(last)
        .encode()
        .unwrap()
}

/// `SETADCXX` for the packer channels (`PK`), which set the `Dst` datum range.
pub fn set_adc_x_pack(first: u32, last: u32) -> Instruction {
    encode::Setadcxx::ZERO
        .pk(1)
        .x0_val(first)
        .x1_val(last)
        .encode()
        .unwrap()
}

pub fn unpack_instruction() -> Instruction {
    // Bit 0 is undocumented and ttsim refuses the instruction without it; see
    // `probe_unpack::unpacr_without_the_undocumented_last_bit_is_refused`.
    let base = encode::UnpacrRegular::ZERO
        .which_unpacker(0)
        .multi_context_mode(1)
        .encode()
        .unwrap();
    Instruction::new(base.word() | UNPACR_LAST, &defs::UNPACR_Regular)
}

/// One `PACR` enabling `read_intf_sel` read interfaces.
pub fn pack_instruction(read_intf_sel: u32, last: bool) -> Instruction {
    encode::Pacr::ZERO
        .read_intf_sel(read_intf_sel)
        .last(u32::from(last))
        .encode()
        .unwrap()
}
