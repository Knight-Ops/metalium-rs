//! The math of attention's softmax over rows too long for L1, as `crate::softmax` streams them
//! (`docs/plans/roadmap.md` Track 1, "Then the long-context softmax"): bit for bit the ops
//! burn-tt's attention composes it from -- `SCALE_MASK_FILL`, the row maximum and its clamp,
//! `SUB_EXP`, the row sum and its clamp, `DIV` -- in three passes over a tile row:
//!
//! 1. [`scale_mask_max`]: each tile's `SCALE_MASK_FILL` (`s`, which the packer writes out), its
//!    maximum accumulated; then [`finish`] (the fold and the clamp).
//! 2. [`sub_exp_sum`]: each tile of `s` through `SUB_EXP` against the maximum, its sum
//!    accumulated; then [`finish`].
//! 3. [`sub_exp_div`]: each tile of `s` through `SUB_EXP` again and `DIV` by the sum: the output.
//!
//! **Why the same bits.** Each link is the kind's own body on the row group
//! (`ops::chain_link`), which loads its operands from `Dst` and stores its result there in the
//! format its op stores it: where the composition packs a result to L1 and the next op unpacks
//! it, both raw, the next link loads what the last stored. The accumulation is a reduction's
//! (`reduce::accumulate`): the first tile copied, each later one combined in tile order, and
//! one fold at the end, so a row streamed in chunks -- its unfolded accumulator carried between
//! them raw, as `reduce::continuation_roles` carries it -- sums as one run does. A statistic is
//! read as `SUB_EXP` and `DIV` read their column operand: the reduction's output tile with its
//! column 0 broadcast ([`Program::broadcast_col0`]), here once a kernel into a row no unpacker
//! writes per tile ([`MAX_ROW`], [`SUM_ROW`]).

use tt_isa::isa::Instruction;

use super::kernel::{A_ROW, B_ROW, C_ROW, OUT_ROW};
use super::ops::{chain_link, exact_in_place, kind_sfpu};
use super::reduce::{combine, finish_rows, Axis, ReduceOp};
use super::{LReg, LoopPolicy, Program};

/// Where the row maximum sits in `Dst` while a kernel reads it, its column 0 broadcast: past
/// `SPILL_ROW`'s 64 rows, inside 32-bit `Dst`'s 512.
pub const MAX_ROW: u32 = 320;
/// Where the row sum sits, as [`MAX_ROW`].
pub const SUM_ROW: u32 = 384;

/// The clamp each statistic takes after its fold, as burn-tt's attention applies it.
pub fn clamp(op: ReduceOp) -> f32 {
    match op {
        ReduceOp::Max => -f32::MAX,
        ReduceOp::Sum => f32::MIN_POSITIVE,
        _ => panic!("softmax has no {op:?} statistic"),
    }
}

/// The reduction's own accumulate for the row group at `o`, its input at `from`: the first tile
/// copied, a later one combined into `OUT_ROW` (`reduce::accumulate`, no padding columns).
fn accumulate(p: &mut Program, op: ReduceOp, first: bool, from: u32, o: u32) {
    let fmt = op.format();
    p.load(LReg::L0, fmt, from + o);
    if first {
        p.store(LReg::L0, fmt, OUT_ROW + o);
    } else {
        p.load(LReg::L1, fmt, OUT_ROW + o);
        combine(p, op, LReg::L0, LReg::L1);
        p.store(LReg::L1, fmt, OUT_ROW + o);
    }
}

/// Pass 1, one tile: `x` at `A_ROW`, its mask at `B_ROW`; `s = SCALE_MASK_FILL(x, mask)` stored
/// at `C_ROW` (the packer writes it out), its maximum accumulated at `OUT_ROW`.
pub fn scale_mask_max(scale: f32, first: bool) -> crate::code::Code {
    let mut p = Program::new();
    p.for_each_row_group(64, |p, o| {
        chain_link(
            p,
            kind_sfpu::SCALE_MASK_FILL,
            [scale, f32::NEG_INFINITY],
            A_ROW + o,
            B_ROW + o,
            C_ROW + o,
        );
        accumulate(p, ReduceOp::Max, first, C_ROW, o);
    });
    p.finish_code()
}

/// Pass 2, one tile: `s` at `A_ROW`; `SUB_EXP(s, max)` stored in place, its sum accumulated at
/// `OUT_ROW`. The maximum is at [`MAX_ROW`], broadcast ([`statistics`]).
pub fn sub_exp_sum(first: bool) -> crate::code::Code {
    let mut p = Program::new();
    p.for_each_row_group(64, |p, o| {
        chain_link(
            p,
            kind_sfpu::SUB_EXP,
            [0.0; 2],
            A_ROW + o,
            MAX_ROW + o,
            A_ROW + o,
        );
        accumulate(p, ReduceOp::Sum, first, A_ROW, o);
    });
    p.finish_code()
}

/// Pass 3, one tile: `s` at `A_ROW`; `DIV(SUB_EXP(s, max), sum)` at `OUT_ROW`, `SUB_EXP`'s result
/// stored at `C_ROW` between them. The statistics are at [`MAX_ROW`] and [`SUM_ROW`].
pub fn sub_exp_div() -> crate::code::Code {
    let mut p = Program::new();
    p.for_each_row_group(64, |p, o| {
        chain_link(
            p,
            kind_sfpu::SUB_EXP,
            [0.0; 2],
            A_ROW + o,
            MAX_ROW + o,
            C_ROW + o,
        );
        chain_link(
            p,
            kind_sfpu::DIV,
            [0.0; 2],
            C_ROW + o,
            SUM_ROW + o,
            OUT_ROW + o,
        );
    });
    p.finish_code()
}

/// The `Dst` rows a streamed pass's staged link hands its registers on through: clear of the
/// operands, the result, the accumulator at `OUT_ROW` and the statistics at [`MAX_ROW`] and
/// [`SUM_ROW`] (the default stage rows' third is [`MAX_ROW`]'s).
const STAGES: [u32; 3] = [C_ROW, super::kernel::SPILL_ROW, 448];

/// `kind`'s staged program (`ops::staged_at`: row loops short enough to replay, bit for bit its
/// body) over `a_at` and `b_at` into `out_at`, or its one-loop link where it has no staged form.
fn link(kind: u32, [a_at, b_at, out_at]: [u32; 3]) -> crate::code::Code {
    super::ops::staged_at(kind, [a_at, b_at, out_at], STAGES).unwrap_or_else(|| {
        let mut p = Program::new();
        p.for_each_row_group(64, |p, o| {
            chain_link(p, kind, [0.0; 2], a_at + o, b_at + o, out_at + o)
        });
        p.finish_code()
    })
}

/// Pass 2 when `e` is written ([`crate::softmax::Mode::Written`]), one tile: `s` at `A_ROW`;
/// `e = SUB_EXP(s, max)` stored at `B_ROW` (the packer writes it out), then its sum accumulated
/// at `OUT_ROW`, a row loop of its own after `SUB_EXP`'s staged ones. The maximum is at
/// [`MAX_ROW`], broadcast ([`statistics`]).
pub fn sub_exp_sum_out(first: bool) -> crate::code::Code {
    let mut p = Program::new();
    p.for_each_row_group(64, |p, o| accumulate(p, ReduceOp::Sum, first, B_ROW, o));
    link(kind_sfpu::SUB_EXP, [A_ROW, MAX_ROW, B_ROW]).then(p.finish_code())
}

/// Pass 3 when `e` is written, one tile: `e` at `A_ROW`; `DIV(e, sum)` at `OUT_ROW`, staged. The
/// sum is at [`SUM_ROW`], broadcast.
pub fn div_by_sum() -> crate::code::Code {
    link(kind_sfpu::DIV, [A_ROW, SUM_ROW, OUT_ROW])
}

/// A kernel's first math: each statistic tile, as unpacked at its row, with its column 0
/// broadcast across every column (what `SUB_EXP` and `DIV` do to their column operand).
pub fn statistics(rows: &[u32]) -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    for &r in rows {
        p.broadcast_col0(r);
    }
    p.finish()
}

/// The finishing program of a statistic: the reduction's fold over columns, then its clamp in
/// place (`reduce_then`'s finish), at `OUT_ROW`.
pub fn finish(op: ReduceOp) -> Vec<Instruction> {
    let mut fin = finish_rows(op, Axis::Cols, 32);
    fin.extend(
        exact_in_place(kind_sfpu::CLAMP_MIN, [clamp(op), 0.0], OUT_ROW)
            .expect("CLAMP_MIN is an exact unary kind"),
    );
    fin
}

/// What the three passes compute for a `[rows, cols]` tile-aligned `x` and its `mask` (`0` or
/// `1` as raw bits, the same shape), each tile row streamed in chunks of `chunk` tiles with the
/// unfolded accumulator carried between them: `(s, max, sum, p)`, the statistics `[rows]` as
/// column 0 holds them. The same programs, run by the interpreter.
pub fn reference(
    x: &[f32],
    mask: &[u32],
    rows: usize,
    cols: usize,
    scale: f32,
    chunk: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    reference_in(x, mask, rows, cols, scale, chunk, false)
}

/// [`reference`] for the passes that write `e` ([`crate::softmax::Mode::Written`]): pass 2 is
/// [`sub_exp_sum_out`], pass 3 [`div_by_sum`] over the `e` pass 2 wrote. The same programs, run by
/// the interpreter.
pub fn reference_written(
    x: &[f32],
    mask: &[u32],
    rows: usize,
    cols: usize,
    scale: f32,
    chunk: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    reference_in(x, mask, rows, cols, scale, chunk, true)
}

fn reference_in(
    x: &[f32],
    mask: &[u32],
    rows: usize,
    cols: usize,
    scale: f32,
    chunk: usize,
    written: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    use super::interp::Vector;
    use tt_isa::dm::face_index;
    assert!(rows % 32 == 0 && cols % 32 == 0 && chunk > 0);
    let (rt, ct) = (rows / 32, cols / 32);
    let tile = |v: &dyn Fn(usize) -> u32, i: usize, j: usize| -> Vec<u32> {
        let mut t = vec![0u32; 1024];
        for r in 0..32 {
            for c in 0..32 {
                t[face_index(r, c)] = v((32 * i + r) * cols + 32 * j + c);
            }
        }
        t
    };
    let put = |out: &mut [f32], t: &[u32], i: usize, j: usize| {
        for r in 0..32 {
            for c in 0..32 {
                out[(32 * i + r) * cols + 32 * j + c] = f32::from_bits(t[face_index(r, c)]);
            }
        }
    };
    let code = |c: crate::code::Code| c.expand();
    let (mut s, mut p) = (vec![0.0; rows * cols], vec![0.0; rows * cols]);
    let mut e_out = vec![0.0; rows * cols];
    let (mut max, mut sum) = (vec![0.0; rows], vec![0.0; rows]);
    let column0 = |t: &[u32], i: usize, into: &mut [f32]| {
        for r in 0..32 {
            into[32 * i + r] = f32::from_bits(t[face_index(r, 0)]);
        }
    };
    for i in 0..rt {
        // A pass over the row, a kernel a chunk: what it unpacks first, then its tiles.
        let pass = |stats: &[(u32, Vec<u32>)],
                    per_tile: &dyn Fn(&mut Vector, usize),
                    program: &dyn Fn(bool) -> Vec<Instruction>,
                    after: &mut dyn FnMut(&Vector, usize),
                    fin: Option<Vec<Instruction>>|
         -> Vec<u32> {
            let mut acc: Option<Vec<u32>> = None;
            for c0 in (0..ct).step_by(chunk) {
                let mut v = Vector::new();
                for (row, t) in stats {
                    v.put_tile(*row as usize, t);
                }
                let rows: Vec<u32> = stats.iter().map(|(r, _)| *r).collect();
                v.run(&statistics(&rows)).unwrap();
                if let Some(a) = &acc {
                    v.put_tile(OUT_ROW as usize, a);
                }
                for j in c0..(c0 + chunk).min(ct) {
                    per_tile(&mut v, j);
                    v.run(&program(j == 0)).unwrap();
                    after(&v, j);
                }
                if c0 + chunk >= ct {
                    if let Some(f) = &fin {
                        v.run(f).unwrap();
                    }
                }
                acc = Some(v.tile(OUT_ROW as usize));
            }
            acc.unwrap()
        };
        let m = pass(
            &[],
            &|v, j| {
                v.put_tile(A_ROW as usize, &tile(&|k| x[k].to_bits(), i, j));
                v.put_tile(B_ROW as usize, &tile(&|k| mask[k], i, j));
            },
            &|first| code(scale_mask_max(scale, first)),
            &mut |v, j| put(&mut s, &v.tile(C_ROW as usize), i, j),
            Some(finish(ReduceOp::Max)),
        );
        column0(&m, i, &mut max);
        let s_tile = |j: usize| tile(&|k| s[k].to_bits(), i, j);
        if written {
            let e = pass(
                &[(MAX_ROW, m)],
                &|v, j| v.put_tile(A_ROW as usize, &s_tile(j)),
                &|first| code(sub_exp_sum_out(first)),
                &mut |v, j| put(&mut e_out, &v.tile(B_ROW as usize), i, j),
                Some(finish(ReduceOp::Sum)),
            );
            column0(&e, i, &mut sum);
            let e_tile = |j: usize| tile(&|k| e_out[k].to_bits(), i, j);
            pass(
                &[(SUM_ROW, e)],
                &|v, j| v.put_tile(A_ROW as usize, &e_tile(j)),
                &|_| code(div_by_sum()),
                &mut |v, j| put(&mut p, &v.tile(OUT_ROW as usize), i, j),
                None,
            );
            continue;
        }
        let e = pass(
            &[(MAX_ROW, m.clone())],
            &|v, j| v.put_tile(A_ROW as usize, &s_tile(j)),
            &|first| code(sub_exp_sum(first)),
            &mut |_, _| {},
            Some(finish(ReduceOp::Sum)),
        );
        column0(&e, i, &mut sum);
        pass(
            &[(MAX_ROW, m), (SUM_ROW, e)],
            &|v, j| v.put_tile(A_ROW as usize, &s_tile(j)),
            &|_| code(sub_exp_div()),
            &mut |v, j| put(&mut p, &v.tile(OUT_ROW as usize), i, j),
            None,
        );
    }
    (s, max, sum, p)
}

#[cfg(test)]
mod tests {
    use super::super::ops::{reference_op, Broadcast};
    use super::super::reduce;
    use super::*;

    /// The composition burn-tt's attention runs, each op's own reference.
    fn composition(
        x: &[f32],
        mask: &[u32],
        rows: usize,
        cols: usize,
        scale: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let m: Vec<f32> = mask.iter().map(|&b| f32::from_bits(b)).collect();
        let none = Broadcast::None;
        let s = reference_op(
            kind_sfpu::SCALE_MASK_FILL,
            [scale, f32::NEG_INFINITY],
            none,
            &[x, &m],
            rows,
            cols,
        );
        let stat = |op: ReduceOp, a: &[f32]| {
            let r = reduce::reference(op, Axis::Cols, a, rows, cols);
            reference_op(kind_sfpu::CLAMP_MIN, [clamp(op), 0.0], none, &[&r], rows, 1)
        };
        let max = stat(ReduceOp::Max, &s);
        let col = Broadcast::Col;
        let e = reference_op(kind_sfpu::SUB_EXP, [0.0; 2], col, &[&s, &max], rows, cols);
        let sum = stat(ReduceOp::Sum, &e);
        let p = reference_op(kind_sfpu::DIV, [0.0; 2], col, &[&e, &sum], rows, cols);
        (s, max, sum, p)
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Scores with every special a softmax meets, and a causal-style mask with rows wholly
    /// masked.
    fn operands(rows: usize, cols: usize, seed: u32) -> (Vec<f32>, Vec<u32>) {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let specials = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE / 4.0,
            -f32::MIN_POSITIVE / 8.0,
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0xffc0_1234),
            300.0,
            -300.0,
        ];
        let x = (0..rows * cols)
            .map(|k| {
                let r = next();
                if k % 97 == 5 {
                    specials[(r % specials.len() as u32) as usize]
                } else {
                    ((r % 20_000) as f32 - 10_000.0) / 997.0
                }
            })
            .collect();
        let mask = (0..rows * cols)
            .map(|k| {
                let (r, c) = (k / cols, k % cols);
                // Row 3 wholly masked; others causal past a row-dependent point.
                u32::from(r % 32 == 3 || c > (r * 37) % cols)
            })
            .collect();
        (x, mask)
    }

    #[test]
    fn the_passes_are_the_composition_bit_for_bit() {
        for (rows, cols, chunk) in [(32, 64, 1), (64, 160, 2), (32, 288, 3), (32, 256, 16)] {
            let (x, mask) = operands(rows, cols, 0x2468 + cols as u32);
            for scale in [0.125, 1.0 / 8f32.sqrt(), -0.5] {
                let fused = reference(&x, &mask, rows, cols, scale, chunk);
                let composed = composition(&x, &mask, rows, cols, scale);
                assert_eq!(bits(&fused.0), bits(&composed.0), "s, {rows}x{cols}");
                assert_eq!(bits(&fused.1), bits(&composed.1), "max, {rows}x{cols}");
                assert_eq!(bits(&fused.2), bits(&composed.2), "sum, {rows}x{cols}");
                assert_eq!(bits(&fused.3), bits(&composed.3), "p, {rows}x{cols}");
            }
        }
    }

    /// The passes that write `e` (staged `SUB_EXP` with the sum, then a staged `DIV` by it) are
    /// the composition bit for bit, `e` included.
    #[test]
    fn the_written_passes_are_the_composition_bit_for_bit() {
        for (rows, cols, chunk) in [(32, 64, 1), (64, 160, 2), (32, 288, 3), (32, 256, 16)] {
            let (x, mask) = operands(rows, cols, 0x1357 + cols as u32);
            for scale in [0.125, 1.0 / 8f32.sqrt(), -0.5] {
                let fused = reference_written(&x, &mask, rows, cols, scale, chunk);
                let composed = composition(&x, &mask, rows, cols, scale);
                assert_eq!(bits(&fused.0), bits(&composed.0), "s, {rows}x{cols}");
                assert_eq!(bits(&fused.1), bits(&composed.1), "max, {rows}x{cols}");
                assert_eq!(bits(&fused.2), bits(&composed.2), "sum, {rows}x{cols}");
                assert_eq!(bits(&fused.3), bits(&composed.3), "p, {rows}x{cols}");
            }
        }
    }

    /// The written passes' links are the staged programs (each stage a replayed row loop), not
    /// the one-loop bodies the recomputing passes run.
    #[test]
    fn the_written_passes_are_staged() {
        let staged = |kind, rows| {
            super::super::ops::staged_at(kind, rows, STAGES)
                .unwrap()
                .ins
        };
        let sub_exp = staged(kind_sfpu::SUB_EXP, [A_ROW, MAX_ROW, B_ROW]);
        assert_eq!(sub_exp_sum_out(true).ins[..sub_exp.len()], sub_exp[..]);
        assert_eq!(
            div_by_sum().ins,
            staged(kind_sfpu::DIV, [A_ROW, SUM_ROW, OUT_ROW])
        );
        // Clear of what the pass keeps: the statistics and the accumulator.
        for r in STAGES {
            for kept in [OUT_ROW, MAX_ROW, SUM_ROW, A_ROW, B_ROW] {
                assert!(r + 64 <= kept || kept + 64 <= r, "stage row {r} and {kept}");
            }
        }
    }

    #[test]
    fn chunks_do_not_change_the_bits() {
        let (rows, cols) = (32, 320);
        let (x, mask) = operands(rows, cols, 99);
        let one = reference(&x, &mask, rows, cols, 0.125, 10);
        for chunk in [1, 3, 4, 7] {
            let r = reference(&x, &mask, rows, cols, 0.125, chunk);
            assert_eq!(bits(&r.3), bits(&one.3), "chunks of {chunk}");
        }
    }

    /// The negative control: the same exponentials summed in another order -- each chunk of two
    /// tiles folded on its own, the folds then added -- give other bits on these data, so the
    /// equality above holds the passes to the composition's order.
    #[test]
    fn a_sum_folded_a_chunk_differs() {
        let (rows, cols, part) = (32, 512, 64);
        let x: Vec<f32> = (0..rows * cols)
            .map(|k| 1.0 + (k % 251) as f32 * 1.1e-3)
            .collect();
        let mask = vec![0u32; rows * cols];
        let (s, max, sum, _) = reference(&x, &mask, rows, cols, 1.0, 2);
        let e = reference_op(
            kind_sfpu::SUB_EXP,
            [0.0; 2],
            Broadcast::Col,
            &[&s, &max],
            rows,
            cols,
        );
        let mut folds = vec![0.0f32; rows];
        for c0 in (0..cols).step_by(part) {
            let block: Vec<f32> = (0..rows)
                .flat_map(|r| e[r * cols + c0..r * cols + c0 + part].to_vec())
                .collect();
            let f = reduce::reference(ReduceOp::Sum, Axis::Cols, &block, rows, part);
            for r in 0..rows {
                folds[r] += f[r];
            }
        }
        assert_ne!(bits(&sum), bits(&folds));
    }
}
