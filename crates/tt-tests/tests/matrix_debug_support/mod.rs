//! Helpers shared by `step111_matrix_debug` and `probe_addr_mod_sweep`: the
//! `SrcA` source rows, the address-modifier entry the sweeps use, and the two
//! one- and eight-row move forms.
#![allow(dead_code)]

use tt_isa::matrix::debug::{AddrModEntry, AddrModTable, Rows};

type MatA = [[f32; 16]; 16];

/// `SrcA` as FP32 bits, rows 0..16: specials first, then a deterministic spread.
/// Every value is finite (the unpacker's NaN/infinity behavior on silicon is not what
/// this gate measures; the host model test covers them).
pub fn a_bits() -> Vec<u32> {
    let mut bits = vec![
        0x0000_0000, // +0
        0x8000_0000, // -0: flushed to +0 by the move
        0x0000_0001, // smallest subnormal
        0x807f_ffff, // largest negative subnormal
        0x0080_0000, // smallest normal
        0x8080_0000,
        0x7f7f_ffff, // largest finite: truncation keeps it finite
        0xff7f_ffff,
        0x3f80_1fff, // low 13 mantissa bits set: truncated, not rounded
        0xbf80_1fff,
        0x3fff_ffff,
        0xbfff_ffff,
        0x3f80_0000,
        0xbf80_0000,
        0x4048_f5c3,
        0xc0c9_0fdb,
    ];
    let mut x = 0x1234_5678u32;
    while bits.len() < 256 {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let exp = 100 + (x >> 24) % 40;
        bits.push((x & 0x8000_0000) ^ (exp << 23) ^ (x.rotate_left(7) & 0x007f_ffff));
    }
    bits
}

pub fn rows16(bits: &[u32]) -> MatA {
    let mut m = [[0f32; 16]; 16];
    for (i, &b) in bits.iter().enumerate() {
        m[i / 16][i % 16] = f32::from_bits(b);
    }
    m
}

/// Entry 1 advances `SrcA` by `src` rows and `Dst` by `dst`; every other entry
/// (and every other field of every entry) is zero.
pub fn entry1(src: u32, dst: u32) -> AddrModTable {
    AddrModTable::ZERO
        .with(
            1,
            AddrModEntry {
                src_a_incr: src,
                dst_incr: dst,
            },
        )
        .unwrap()
}

pub fn one(src_row: u32, dst_row: u32) -> Rows {
    Rows::One { src_row, dst_row }
}

pub fn eight(src_row: u32, dst_row: u32) -> Rows {
    Rows::Eight { src_row, dst_row }
}
