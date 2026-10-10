//! Lane T1: the integer and boolean selects and `int_abs`, held to host oracles through the
//! interpreter. The bit patterns are the ones an `f32` path would damage: NaN payloads, the sign
//! bit alone, the denormal one, and `2^24 + 1`.

use super::super::interp::Vector;
use super::kind_sfpu::*;
use super::*;
use crate::sfpu::kernel::C_ROW;

const PATTERNS: [u32; 12] = [
    0x7fc0_0001,
    0x8000_0000,
    0x0000_0001,
    16_777_217,
    0x7f80_0000,
    0xffc0_0000,
    0x007f_ffff,
    0xffff_ffff,
    0,
    0x7fff_ffff,
    0x8000_0001,
    0x0100_0001,
];

fn lanes(seed: u32) -> Vec<u32> {
    let mut s = seed | 1;
    (0..1024)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if i % 3 == 0 {
                PATTERNS[(i / 3 + seed as usize) % PATTERNS.len()]
            } else {
                s
            }
        })
        .collect()
}

fn run(kind: u32, scalar: u32, rows: &[(u32, &[u32])]) -> Vec<u32> {
    let (_, math) = program(kind, f32::from_bits(scalar)).unwrap();
    let mut v = Vector::new();
    for (row, data) in rows {
        v.put_tile(*row as usize, data);
    }
    v.run(&math)
        .unwrap_or_else(|e| panic!("kind {kind:#x}: {e}"));
    v.tile(OUT_ROW as usize)
}

#[test]
fn signatures_and_operands() {
    use crate::tensor::Elem::{Bool, I32};
    let sig = |k| (elems(k).inputs.to_vec(), elems(k).out);
    assert_eq!(sig(INT_MASK_FILL), (vec![I32, Bool], I32));
    assert_eq!(sig(BOOL_MASK_FILL), (vec![Bool, Bool], Bool));
    assert_eq!(sig(INT_MASK_WHERE), (vec![I32, Bool, I32], I32));
    assert_eq!(sig(BOOL_MASK_WHERE), (vec![Bool, Bool, Bool], Bool));
    assert_eq!(sig(INT_ABS), (vec![I32], I32));
    assert_eq!(operands(INT_ABS), Some(Operands::Unary));
    assert_eq!(operands(INT_MASK_FILL), Some(Operands::Binary));
    assert_eq!(operands(INT_MASK_WHERE), Some(Operands::Ternary));
    assert!(broadcasts(INT_MASK_FILL) && broadcasts(BOOL_MASK_FILL));
}

/// `|x|` wraps: `i32::MIN` stays itself, every other lane is `wrapping_abs`, whatever its bits
/// look like as a float.
#[test]
fn int_abs_is_wrapping_abs_on_every_lane() {
    for seed in [3, 5, 7] {
        let a = lanes(seed);
        let got = run(INT_ABS, 0, &[(A_ROW, &a)]);
        for (i, (&x, &g)) in a.iter().zip(&got).enumerate() {
            assert_eq!(g, (x as i32).wrapping_abs() as u32, "lane {i}: {x:#010x}");
        }
    }
    let got = run(INT_ABS, 0, &[(A_ROW, &[i32::MIN as u32; 1024])]);
    assert!(got.iter().all(|&g| g == i32::MIN as u32));
}

/// The selects are predicated moves of raw words: each lane's output is exactly the chosen
/// operand's word. The program is the float `MASK_WHERE`'s, instruction for instruction.
#[test]
fn selects_move_raw_words() {
    for kind in [INT_MASK_WHERE, BOOL_MASK_WHERE] {
        assert_eq!(
            program(kind, 0.0).unwrap().1,
            program(MASK_WHERE, 0.0).unwrap().1,
            "kind {kind:#x}"
        );
        let (a, c) = (lanes(11), lanes(13));
        let mask: Vec<u32> = lanes(17).iter().map(|w| w & 1).collect();
        let got = run(kind, 0, &[(A_ROW, &a), (B_ROW, &mask), (C_ROW, &c)]);
        for i in 0..1024 {
            assert_eq!(got[i], if mask[i] != 0 { c[i] } else { a[i] }, "lane {i}");
        }
    }
}

/// The fill's scalar is the exact 32-bit word, for every pattern, and only masked lanes take it.
#[test]
fn fills_carry_raw_scalar_bits() {
    let a = lanes(19);
    let mask: Vec<u32> = lanes(23).iter().map(|w| (w >> 5) & 1).collect();
    for kind in [INT_MASK_FILL, BOOL_MASK_FILL] {
        for scalar in PATTERNS {
            let got = run(kind, scalar, &[(A_ROW, &a), (B_ROW, &mask)]);
            for i in 0..1024 {
                assert_eq!(
                    got[i],
                    if mask[i] != 0 { scalar } else { a[i] },
                    "kind {kind:#x}, scalar {scalar:#010x}, lane {i}"
                );
            }
        }
    }
}
