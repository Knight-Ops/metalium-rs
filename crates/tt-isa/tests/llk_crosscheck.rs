//! Our instruction encodings against tt-metal's LLK, the vendor's own Blackhole
//! encodings (`tt_llk_blackhole/common/inc/ckernel_ops.h`, `TT_OP_*` macros).
//!
//! The generated table comes from the specification's diagrams, and for the
//! Matrix Unit and the ADC/RWC instructions those diagrams are Wormhole's
//! (`Provenance::WormholeOnly`). Twice now a field has sat one bit lower on
//! Blackhole and applied the wrong operand silently: `MVMUL`'s `AddrMod`
//! (divergence row 42), and `MOVB2D`'s `Move4Rows`, found here -- the whole of
//! what row 38 recorded as ttsim "moving one row" -- along with the `AddrMod` of
//! every `MOV*` between `Src` and `Dst`. All are now measured layouts
//! (`Bits32_BH.lua`). The
//! LLK macros are a second, independent source that runs on these chips, so every
//! encoding the datapath emits is checked against one.
//!
//! Each case builds the same instruction twice: through our encoder, and from
//! the LLK macro's shifts, transcribed with the macro named.

use tt_isa::backend;
use tt_isa::isa::generated::encode;
use tt_isa::sfpu;

/// `TT_OP(opcode, params)`: the opcode in the top byte.
fn op(opcode: u32, params: u32) -> u32 {
    (opcode << 24) | params
}

#[test]
fn the_datapath_encodings_agree_with_llk() {
    let cases: &[(&str, u32, u32)] = &[
        // TT_OP_MOVA2D(dest_32b_lo<<23, src<<17, addr_mode<<14, instr_mod<<12, dst); MOV_8_ROWS = 2
        (
            "MOVA2D Move8Rows",
            encode::Mova2D::ZERO.move8_rows(1).encode().unwrap().word(),
            op(0x12, 2 << 12),
        ),
        (
            "MOVA2D rows",
            encode::Mova2D::ZERO
                .src_row(5)
                .dst_row(8)
                .encode()
                .unwrap()
                .word(),
            op(0x12, (5 << 17) | 8),
        ),
        (
            "MOVA2D AddrMod",
            encode::Mova2D::ZERO.addr_mod(1).encode().unwrap().word(),
            op(0x12, 1 << 14),
        ),
        // TT_OP_MOVB2D(dest_32b_lo<<23, src<<17, addr_mode<<14, instr_mod<<11, dst)
        // MOV_1_ROW_D0_BRCST = 1, MOV_8_ROW_BRCST = 2, MOV_4_ROWS = 4
        (
            "MOVB2D Move4Rows",
            encode::Movb2D::ZERO.move4_rows(1).encode().unwrap().word(),
            op(0x13, 4 << 11),
        ),
        (
            "MOVB2D BroadcastCol0",
            encode::Movb2D::ZERO
                .broadcast_col0(1)
                .encode()
                .unwrap()
                .word(),
            op(0x13, 1 << 11),
        ),
        (
            "MOVB2D Broadcast1RowTo8",
            encode::Movb2D::ZERO
                .broadcast1_row_to8(1)
                .encode()
                .unwrap()
                .word(),
            op(0x13, 2 << 11),
        ),
        (
            "MOVB2D AddrMod",
            encode::Movb2D::ZERO.addr_mod(1).encode().unwrap().word(),
            op(0x13, 1 << 14),
        ),
        (
            "MOVB2D rows",
            encode::Movb2D::ZERO
                .src_row(3)
                .dst_row(2)
                .encode()
                .unwrap()
                .word(),
            op(0x13, (3 << 17) | 2),
        ),
        // TT_OP_MOVD2A: as MOVA2D, opcode 0x08
        (
            "MOVD2A rows",
            encode::Movd2A::ZERO
                .src_row(1)
                .dst_row(1)
                .encode()
                .unwrap()
                .word(),
            op(0x08, (1 << 17) | 1),
        ),
        // p_movd2a::MOV_4_ROWS = 2 at instr_mod<<12
        (
            "MOVD2A Move4Rows",
            encode::Movd2A::ZERO.move4_rows(1).encode().unwrap().word(),
            op(0x08, 2 << 12),
        ),
        (
            "MOVD2A AddrMod",
            encode::Movd2A::ZERO.addr_mod(1).encode().unwrap().word(),
            op(0x08, 1 << 14),
        ),
        // TT_OP_MOVD2B: as MOVD2A, opcode 0x0a; p_movd2b::MOV_4_ROWS = 2
        (
            "MOVD2B Move4Rows",
            encode::Movd2B::ZERO.move4_rows(1).encode().unwrap().word(),
            op(0x0a, 2 << 12),
        ),
        (
            "MOVD2B AddrMod",
            encode::Movd2B::ZERO.addr_mod(1).encode().unwrap().word(),
            op(0x0a, 1 << 14),
        ),
        // TT_OP_MOVB2A(srca<<17, addr_mode<<14, instr_mod<<12, srcb); p_movb2a::MOV_4_ROWS = 2
        (
            "MOVB2A rows",
            encode::movb2_a(5, 0, 0, 3).unwrap().word(),
            op(0x0b, (5 << 17) | 3),
        ),
        (
            "MOVB2A Move4Rows",
            encode::movb2_a(0, 0, 1, 0).unwrap().word(),
            op(0x0b, 2 << 12),
        ),
        (
            "MOVB2A AddrMod",
            encode::movb2_a(0, 1, 0, 0).unwrap().word(),
            op(0x0b, 1 << 14),
        ),
        // TT_OP_ZEROACC(clear_mode<<19, use_32_bit_mode<<18, clear_zero_flags<<17, addr_mode<<14, where); CLR_ALL = 3
        (
            "ZEROACC CLR_ALL",
            encode::Zeroacc::ZERO.mode(3).encode().unwrap().word(),
            op(0x10, 3 << 19),
        ),
        // TT_OP_MVMUL(clear_dvalid<<22, instr_mod19<<19, addr_mode<<14, dst)
        (
            "MVMUL AddrMod",
            encode::Mvmul::ZERO.addr_mod(1).encode().unwrap().word(),
            op(0x26, 1 << 14),
        ),
        // TT_OP_SETADCXX(CntSetMask<<21, x_end2<<10, x_start)
        (
            "SETADCXX",
            encode::Setadcxx::ZERO
                .u0(1)
                .x1_val(19)
                .encode()
                .unwrap()
                .word(),
            op(0x5e, (1 << 21) | (19 << 10)),
        ),
        // TT_OP_SETRWC(clear_ab_vld<<22, rwc_cr<<18, rwc_d<<14, rwc_b<<10, rwc_a<<6, BitMask)
        (
            "SETRWC",
            encode::Setrwc::ZERO
                .src_a(1)
                .src_a_val(8)
                .encode()
                .unwrap()
                .word(),
            op(0x37, (8 << 6) | 1),
        ),
        // TT_OP_UNPACR(block<<23, ..., OvrdThreadId<<7, SetDatValid<<6, ..., Last)
        (
            "UNPACR SrcA",
            encode::UnpacrRegular::ZERO
                .which_unpacker(0)
                .multi_context_mode(1)
                .flip_src(1)
                .encode()
                .unwrap()
                .word()
                | 1,
            op(0x42, (1 << 7) | (1 << 6) | 1),
        ),
        (
            "UNPACR SrcB",
            encode::UnpacrRegular::ZERO
                .which_unpacker(1)
                .multi_context_mode(1)
                .flip_src(1)
                .encode()
                .unwrap()
                .word()
                | 1,
            op(0x42, (1 << 23) | (1 << 7) | (1 << 6) | 1),
        ),
        // TT_OP_PACR(..., ReadIntfSel<<8, ..., Last)
        (
            "PACR",
            encode::Pacr::ZERO
                .read_intf_sel(0b1111)
                .last(1)
                .encode()
                .unwrap()
                .word(),
            op(0x41, (0b1111 << 8) | 1),
        ),
        // TT_OP_STALLWAIT(stall_res<<15, wait_res)
        (
            "STALLWAIT",
            backend::stallwait(1 << 3, 1 << 1).unwrap().word(),
            op(0xa2, (8 << 15) | 2),
        ),
        // TT_OP_SFPLOAD / TT_OP_SFPSTORE(lreg<<20, instr_mod0<<16, sfpu_addr_mode<<13, dest_reg_addr)
        (
            "SFPLOAD",
            sfpu::load(2, 0, 0, 4).unwrap().word(),
            op(0x70, (2 << 20) | 4),
        ),
        (
            "SFPSTORE",
            sfpu::store(2, 3, 1, 4).unwrap().word(),
            op(0x72, (2 << 20) | (3 << 16) | (1 << 13) | 4),
        ),
    ];
    let wrong: Vec<String> = cases
        .iter()
        .filter(|(_, ours, llk)| ours != llk)
        .map(|(n, ours, llk)| format!("{n}: ours {ours:#010x}, LLK {llk:#010x}"))
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
