//! RMSNorm's statistic in one kernel (`docs/plans/decode-small-ops.md` S3): each row's
//! `sqrt(mean(x x) + eps)` over its columns, bit for bit the ops Burn composes it from on
//! `TtBackend` -- `MUL(x, x)`, the sum over columns, `MUL_SCALAR(1/n)`, `ADD_SCALAR(eps)` and
//! `SQRT` -- as the `[rows, 1]` tensor those leave, with none of their intermediate tensors. The
//! division by it stays `DIV`'s own op, which spreads a row's tiles over the units.
//!
//! A reduction's kernel (`sfpu::reduce`, one output tile a tile row) with two changes:
//!
//! 1. **The sum of squares**: each tile squared in `Dst` as it is accumulated
//!    ([`super::reduce::accumulate_squares`]), in the order a sum over columns adds `MUL(x, x)`'s
//!    tiles, then folded ([`super::reduce::finish`]).
//! 2. **The statistic** ([`statistic`]): `MUL_SCALAR`'s, `ADD_SCALAR`'s and `SQRT`'s bodies on
//!    the folded sums, lanewise, before the packer writes them. The fold leaves each lane its own
//!    rounding of the row's sum; the separate ops compute every lane too, and `DIV`'s column
//!    broadcast reads column 0's, so lane for lane the same instructions give the same tensor.
//!
//! A tile row with fewer than 32 rows (a decode step's `[1, n]`) runs all three over only the
//! row groups its rows occupy ([`super::reduce::row_offsets`]: one of eight for one row). No
//! value moves between rows, so the rows computed are the same bits; the output's other rows
//! are whatever `Dst` held, padding, as the output claims (`Pad::Undefined`).
//!
//! **Why not the division too.** A first version divided in the same kernel, the row's tiles
//! unpacked a second time: 640 us a 2048-wide norm on silicon, against 106 us for the five ops.
//! `DIV`'s body is too long for the replay buffer, so each of its instructions is pushed from
//! the role program, and one unit pushed all 64 tiles' (`performance-log.md`, 2026-10-16).

use tt_isa::isa::Instruction;

use super::kernel::OUT_ROW;
use super::ops::sqrt_program;
use super::reduce::{accumulate_squares, accumulate_squares_in_order, finish_rows, Axis, ReduceOp};
use super::{Format, LReg, LoopPolicy, Program};

/// Most tiles a row may have: its slots fit the data arena and its role programs a program
/// slot (gate below).
pub const MAX_TILES: usize = 128;

/// Whether a row of `cols` columns is one job.
pub fn fits(cols: usize) -> bool {
    (1..=MAX_TILES).contains(&cols.div_ceil(32))
}

/// The statistic of `n` columns, in place on the folded sums at `OUT_ROW`:
/// `sqrt(sum (1/n) + eps)`, every lane of the row groups of the tile's first `rows` rows.
pub fn statistic(n: usize, eps: f32, rows: u32) -> Vec<Instruction> {
    let (l0, l2, l3, l7) = (LReg::L0, LReg::L2, LReg::L3, LReg::L7);
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    // The scalars as `float_mean_dim` and `float_add_scalar` give them.
    let inv = 1.0 / n as f32;
    // Tensor rows 0..16 are in faces 0 and 1, 16..32 in faces 2 and 3; a group of four rows in
    // each, its even columns at `left`, its odd at `left | 2` (`reduce::finish` stores the same
    // lanes at all four).
    for lower in [0, 32] {
        for g in 0..4 {
            if lower / 2 + 4 * g >= rows {
                continue;
            }
            let (left, right) = (lower + 4 * g, lower + 16 + 4 * g);
            p.load(l0, Format::Fp32, OUT_ROW + left);
            // `MUL_SCALAR`, `ADD_SCALAR`, `SQRT`: each op's body, its result where the next
            // loads it from.
            p.loadi(l3, inv);
            p.mul(l0, l3, l2);
            p.mov(l2, l0);
            p.loadi(l3, eps);
            p.add(l0, l3, l2);
            p.mov(l2, l0);
            sqrt_program(&mut p, l0, l7, false);
            for a in [left, left | 2, right, right | 2] {
                p.store(l7, Format::Fp32, OUT_ROW + a);
            }
        }
    }
    p.finish()
}

/// The math programs of a tile row of `rows` rows and `per` tiles, whose last has `valid`
/// columns: each tile's sum-of-squares program in order, then the fold and the statistic as one
/// finishing program (`sfpu::reduce::roles` runs them).
pub fn math(
    per: usize,
    valid: u32,
    eps: f32,
    rows: u32,
) -> (Vec<Vec<Instruction>>, Vec<Instruction>) {
    let cols = (per - 1) * 32 + valid as usize;
    let masked = valid < 32;
    let squares = (0..per)
        .map(|n| accumulate_squares(n == 0, (n + 1 == per && masked).then_some(valid), rows))
        .collect();
    let mut fin = finish_rows(ReduceOp::Sum, Axis::Cols, rows);
    fin.extend(statistic(cols, eps, rows));
    (squares, fin)
}

/// [`math`] for one row of `cols` (whole tiles) laid out `[cols / 32, 32]`
/// (`tensor::pack_row`): its `cols / 1024` tiles' squares summed down their rows in order
/// ([`accumulate_squares_in_order`]), then row 0 folded and the statistic. The same sums in the
/// same order as [`math`]'s over the row's `cols / 32` tiles, so the same bits, from an
/// entry a packed tile where the row takes one a tile.
pub fn math_packed(cols: usize, eps: f32) -> (Vec<Vec<Instruction>>, Vec<Instruction>) {
    assert!(
        cols % 32 == 0 && cols > 0,
        "a packed row is whole tiles, not {cols} columns"
    );
    let rows = cols / 32;
    let tiles = rows.div_ceil(32);
    let squares = (0..tiles)
        .map(|n| accumulate_squares_in_order(n == 0, (rows - 32 * n).min(32) as u32))
        .collect();
    let mut fin = finish_rows(ReduceOp::Sum, Axis::Cols, 1);
    fin.extend(statistic(cols, eps, 1));
    (squares, fin)
}

/// What the device computes for `x` (`[rows, cols]`, row-major): `[rows]`, each row's
/// statistic as column 0 holds it, the same programs run by the interpreter over each tile row.
/// Padding is a NaN, which no row's statistic may see.
pub fn reference(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    reference_with(x, rows, cols, eps, |r| r)
}

/// [`reference`], each tile row's programs made for `programmed(r)` rows where it has `r`.
fn reference_with(
    x: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
    programmed: impl Fn(u32) -> u32,
) -> Vec<f32> {
    use super::interp::Vector;
    use tt_isa::dm::face_index;
    let (rt, per) = (rows.div_ceil(32), cols.div_ceil(32));
    let valid = match cols % 32 {
        0 => 32,
        v => v as u32,
    };
    let tile = |i: usize, j: usize| -> Vec<u32> {
        let mut t = vec![0x7fc0_dead_u32; 1024];
        for r in 0..32 {
            for c in 0..32 {
                let (gr, gc) = (32 * i + r, 32 * j + c);
                if gr < rows && gc < cols {
                    t[face_index(r, c)] = x[gr * cols + gc].to_bits();
                }
            }
        }
        t
    };
    let mut out = Vec::with_capacity(rows);
    for i in 0..rt {
        let (squares, fin) = math(per, valid, eps, programmed((rows - 32 * i).min(32) as u32));
        let mut v = Vector::new();
        for (j, program) in squares.iter().enumerate() {
            v.put_tile(super::kernel::A_ROW as usize, &tile(i, j));
            v.run(program)
                .unwrap_or_else(|e| panic!("sum of squares: {e}"));
        }
        v.run(&fin).unwrap_or_else(|e| panic!("statistic: {e}"));
        let t = v.tile(OUT_ROW as usize);
        for r in 0..32.min(rows - 32 * i) {
            out.push(f32::from_bits(t[face_index(r, 0)]));
        }
    }
    out
}

/// What the device computes for one row `x` of whole tiles from its packed layout
/// ([`math_packed`]): the statistic as column 0 holds it, the packed tiles built as
/// `tensor::pack_row` lays them (row `kk` the row's columns `32 kk..32 kk + 32`), their unused
/// rows NaN.
pub fn reference_packed(x: &[f32], eps: f32) -> f32 {
    use super::interp::Vector;
    use tt_isa::dm::face_index;
    let cols = x.len();
    let (squares, fin) = math_packed(cols, eps);
    let mut v = Vector::new();
    for (n, program) in squares.iter().enumerate() {
        let mut t = vec![0x7fc0_dead_u32; 1024];
        for r in 0..32 {
            let kk = 32 * n + r;
            for c in 0..32 {
                if kk * 32 + c < cols {
                    t[face_index(r, c)] = x[kk * 32 + c].to_bits();
                }
            }
        }
        v.put_tile(super::kernel::A_ROW as usize, &t);
        v.run(program)
            .unwrap_or_else(|e| panic!("packed sum of squares: {e}"));
    }
    v.run(&fin).unwrap_or_else(|e| panic!("statistic: {e}"));
    f32::from_bits(v.tile(OUT_ROW as usize)[face_index(0, 0)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kind;
    use crate::sfpu::ops::{kind_sfpu, reference as op};

    /// Every special value among ordinary ones of several magnitudes, and a row of zeros.
    fn values(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
        const SPECIAL: [u32; 10] = [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x807f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0x7f7f_ffff,
            0x1f80_0000,
            0x6080_0000,
        ];
        let mut s = seed | 1;
        (0..rows * cols)
            .map(|i| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let (r, c) = (i / cols, i % cols);
                if r == 1 {
                    // A row of signed zeros.
                    return if c % 2 == 0 { 0.0 } else { -0.0 };
                }
                // Rows 2 and 3 carry one special value each; the rest are ordinary.
                if (r == 2 || r == 3) && c == (s as usize >> 8) % cols {
                    return f32::from_bits(SPECIAL[(s >> 4) as usize % SPECIAL.len()]);
                }
                if r == 4 && c % 7 == 3 {
                    return f32::from_bits(SPECIAL[c % SPECIAL.len()]);
                }
                let scale = [1.0e-3f32, 1.0, 37.0, 1.0e4][r % 4];
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
            })
            .collect()
    }

    /// The composition the kernel replaces, each op's own program on the interpreter.
    fn composed(x: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
        let squares = op(kind::MUL, 0.0, x, Some(x), rows, cols);
        let sums = super::super::reduce::reference(ReduceOp::Sum, Axis::Cols, &squares, rows, cols);
        let mean = op(kind::MUL_SCALAR, 1.0 / cols as f32, &sums, None, rows, 1);
        let shifted = op(kind::ADD_SCALAR, eps, &mean, None, rows, 1);
        op(kind_sfpu::SQRT, 0.0, &shifted, None, rows, 1)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn the_statistic_is_the_composed_ops_bit_for_bit() {
        for (rows, cols) in [
            (1, 64),
            (5, 37),
            (33, 96),
            (2, 576),
            (1, 2048),
            (3, 4096),
            (4, 64),
            (16, 40),
            (17, 64),
            (31, 33),
            (32, 64),
            (49, 70),
        ] {
            for eps in [1e-5f32, 1e-6] {
                let x = values(rows, cols, (rows * 1000 + cols) as u32);
                let want = composed(&x, rows, cols, eps);
                let got = reference(&x, rows, cols, eps);
                assert_eq!(bits(&got), bits(&want), "[{rows}, {cols}], eps {eps:e}");
            }
        }
    }

    /// Control: the gate sees the sum's order. Summing the same squares in another order (the
    /// row's tiles reversed) gives a different statistic somewhere.
    #[test]
    fn a_sum_in_another_order_differs() {
        let (rows, cols) = (8, 2048);
        let x = values(rows, cols, 9);
        let reversed: Vec<f32> = x
            .chunks(cols)
            .flat_map(|row| row.chunks(32).rev().flatten().copied().collect::<Vec<_>>())
            .collect();
        assert_ne!(
            bits(&reference(&x, rows, cols, 1e-5)),
            bits(&reference(&reversed, rows, cols, 1e-5)),
            "the order of the sum changed no bit"
        );
    }

    /// Control: the gate sees the row groups. The last tile row's programs made for one group
    /// of rows fewer than it has leave its last rows' statistic undone, at a group boundary, at a
    /// face boundary, and past a whole tile row.
    #[test]
    fn a_row_group_too_few_differs() {
        for (rows, programmed) in [(5, 4), (17, 16), (37, 4)] {
            let (cols, last) = (96, (rows - 1) % 32 + 1);
            let x = values(rows, cols, 7);
            let want = bits(&reference(&x, rows, cols, 1e-5));
            let got = bits(&reference_with(&x, rows, cols, 1e-5, |r| {
                if r == last as u32 {
                    programmed
                } else {
                    r
                }
            }));
            let kept = rows - last + programmed as usize;
            assert_ne!(
                got, want,
                "{rows} rows, the last {last} programmed as {programmed}"
            );
            assert_eq!(
                got[..kept],
                want[..kept],
                "{rows} rows: the rows programmed still run"
            );
        }
    }

    /// The longest row's role programs fit a program slot, and its slots the arena.
    #[test]
    fn the_longest_row_fits() {
        use crate::sfpu::reduce::{plan_layout, roles};
        let (squares, fin) = math(MAX_TILES, 32, 1e-5, 32);
        let layout = plan_layout(1, MAX_TILES).unwrap();
        for (role, p) in roles(&layout, &squares, &fin).iter().enumerate() {
            assert!(
                p.len() <= tt_isa::mailbox::PROGRAM_MAX as usize,
                "role {role}: {} instructions",
                p.len()
            );
        }
        assert!(fits(32 * MAX_TILES) && !fits(32 * MAX_TILES + 1));
    }

    /// A row's statistic from its packed layout is the row's, bit for bit, at widths of 18 to
    /// 256 tiles (one packed tile, several, a ragged last one) over ordinary values, signed
    /// zeros, every special value, and two NaNs of different payloads in one row. Control: the
    /// packed rows summed bottom to top differ, on a row whose sum the order changes.
    #[test]
    fn a_packed_row_s_statistic_is_the_row_s() {
        let mut differ = 0;
        for cols in [576usize, 1024, 1056, 2048, 4096, 8192] {
            let rows = values(5, cols, cols as u32);
            let mut cases: Vec<Vec<f32>> = rows.chunks(cols).map(<[f32]>::to_vec).collect();
            let mut two_nans = cases[0].clone();
            two_nans[3] = f32::from_bits(0x7fc1_2345);
            two_nans[cols - 5] = f32::from_bits(0xffa0_0001);
            cases.push(two_nans);
            for (i, x) in cases.iter().enumerate() {
                for eps in [1e-5f32, 1e-6] {
                    let want = reference(x, 1, cols, eps)[0];
                    let got = reference_packed(x, eps);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{cols} columns, row {i}, eps {eps}"
                    );
                    if i == 0 {
                        // The first tile's squares `2^24`, the rest `1`: each `1` is lost
                        // added to `2^24`, and not when the ones come first.
                        let x: Vec<f32> = (0..cols)
                            .map(|k| if k < 32 { 4096.0 } else { 1.0 })
                            .collect();
                        let want = reference(&x, 1, cols, eps)[0];
                        assert_eq!(reference_packed(&x, eps).to_bits(), want.to_bits());
                        let reversed: Vec<f32> =
                            x.chunks(32).rev().flat_map(|c| c.iter().copied()).collect();
                        differ += usize::from(
                            reference_packed(&reversed, eps).to_bits() != want.to_bits(),
                        );
                    }
                }
            }
        }
        assert_eq!(
            differ,
            12,
            "control: reversed rows matched in {} of 12",
            12 - differ
        );
    }
}
