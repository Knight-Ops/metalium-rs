//! The index of a row's largest element, as `torch.argmax` and burn-tt's argmax choose it: the
//! first maximum, any NaN above every number (the first NaN), `-0` and `+0` equal. A decode step's
//! next token from its `[1, vocab]` logits, chosen on the card.
//!
//! Two phases, each a reduction kernel's roles with one body a tile ([`roles`]: stored once and
//! repeated, so a unit's program does not grow with its tiles):
//! 1. **Partials:** a job's run of the row's tiles. Each value becomes a key whose sign-magnitude
//!    order (`Cond::Less`) is the choice's: every NaN the one positive quiet NaN, which sorts
//!    above `+inf`, and `-0` `+0`; a column past the row's end the smallest key. Each lane keeps
//!    the best key it has seen and that key's index (an exact float: the run's running first
//!    column, held in `Dst`, plus the lane's column), replacing them only on a strictly larger
//!    key, so a later equal one never wins. [`fold`] then takes the best of the run's 32 columns,
//!    a larger key or an equal key at a smaller index: a total order, so the fold's order does not
//!    matter. Row 0 of the output tile holds the key, row 4 the row's index.
//! 2. **The choice:** one job reads the partials in order and keeps the best the same way;
//!    [`index_out`] writes the index as an I32 at `(0, 0)`.
//!
//! Only tile row 0 is a row; the other rows of each tile are computed and never read.

use tt_isa::isa::Instruction;

use super::kernel::{A_ROW, C_ROW, OUT_ROW};
use super::{Cond, Format, LReg, LoopPolicy, Program};

/// `Dst` offsets of tile row 0's four groups of eight columns: even and odd columns of faces 0
/// and 1 (`reduce::row_offsets(1)`).
const ROW0: [u32; 4] = [0, 2, 16, 18];

/// Where phase 1 keeps the next tile's first column, as a float: a group the accumulators leave.
const BASE: u32 = C_ROW + 4;

/// The smallest key in sign-magnitude order: a column past the row's end, and the start.
const NONE: u32 = 0xffff_ffff;

/// The column of each lane of row 0 at `ROW0[g]`, past the even/odd and face split.
fn column_base(g: usize) -> i32 {
    let o = ROW0[g];
    i32::from(o & 2 != 0) + if o >= 16 { 16 } else { 0 }
}

/// `x` in `L0` as its key: NaN the positive quiet NaN, `-0` `+0`. Uses `L4`, `L5`.
fn canonical(p: &mut Program) {
    p.loadi_bits(LReg::L4, 0x7fff_ffff);
    p.and(LReg::L0, LReg::L4, LReg::L5);
    p.loadi_bits(LReg::L4, 0x7f80_0000);
    p.if_(Cond::Less(LReg::L4, LReg::L5), |p| {
        p.loadi_bits(LReg::L0, 0x7fc0_0000)
    });
    p.if_(Cond::Eq0(LReg::L5), |p| p.loadi_bits(LReg::L0, 0));
}

/// `(ka, ia) = (kb, ib)` where `kb` is the larger key, or the equal key at the smaller index.
fn take_better(p: &mut Program, [ka, ia]: [LReg; 2], [kb, ib]: [LReg; 2]) {
    let take = |p: &mut Program| {
        p.mov(kb, ka);
        p.mov(ib, ia);
    };
    p.if_else(Cond::Less(ka, kb), take, |p| {
        // Here `kb <= ka`: equal where also `ka <= kb`.
        p.if_(Cond::LessEq(ka, kb), |p| p.if_(Cond::Less(ib, ia), take));
    });
}

/// Phase 1, before its first tile: no best yet, and the run's first column `start`.
pub fn prologue(start: usize) -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    p.loadi_bits(LReg::L0, NONE);
    p.loadi_bits(LReg::L1, 0);
    for &o in &ROW0 {
        p.store(LReg::L0, Format::Int32, OUT_ROW + o);
        p.store(LReg::L1, Format::Int32, C_ROW + o);
    }
    p.loadi(LReg::L1, start as f32);
    p.store(LReg::L1, Format::Int32, BASE);
    p.finish()
}

/// Phase 1, each tile: the running key in `OUT_ROW + o` and index in `C_ROW + o`, over a row
/// whose last column is `last`. The same for every tile of every run.
pub fn accumulate(last: usize) -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    for (g, &o) in ROW0.iter().enumerate() {
        p.load(LReg::L0, Format::Int32, A_ROW + o);
        canonical(&mut p);
        // The column: `2 (lane & 7)` plus the group's base, as a float, plus the tile's first.
        p.loadi_bits(LReg::L6, 14);
        p.and(LReg::LANE_X2, LReg::L6, LReg::L6);
        let base = column_base(g);
        if base != 0 {
            p.iadd_imm(LReg::L6, base, LReg::L6);
        }
        p.sm32_to_float(LReg::L6, LReg::L6);
        p.load(LReg::L7, Format::Int32, BASE);
        p.add(LReg::L6, LReg::L7, LReg::L6);
        // Past the row's end: the smallest key, which never wins.
        p.loadi(LReg::L7, last as f32);
        p.if_(Cond::Less(LReg::L7, LReg::L6), |p| {
            p.loadi_bits(LReg::L0, NONE)
        });
        p.load(LReg::L2, Format::Int32, OUT_ROW + o);
        p.load(LReg::L3, Format::Int32, C_ROW + o);
        take_better(&mut p, [LReg::L2, LReg::L3], [LReg::L0, LReg::L6]);
        p.store(LReg::L2, Format::Int32, OUT_ROW + o);
        p.store(LReg::L3, Format::Int32, C_ROW + o);
    }
    p.load(LReg::L7, Format::Int32, BASE);
    p.loadi(LReg::L6, 32.0);
    p.add(LReg::L7, LReg::L6, LReg::L7);
    p.store(LReg::L7, Format::Int32, BASE);
    p.finish()
}

/// Phase 1's finish: the best of the run's 32 columns, its key in row 0 and its index in row 4
/// of the output tile (every lane of lane row 0 holds it).
pub fn fold() -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    p.load(LReg::L0, Format::Int32, OUT_ROW + ROW0[0]);
    p.load(LReg::L1, Format::Int32, C_ROW + ROW0[0]);
    for &o in &ROW0[1..] {
        p.load(LReg::L2, Format::Int32, OUT_ROW + o);
        p.load(LReg::L3, Format::Int32, C_ROW + o);
        take_better(&mut p, [LReg::L0, LReg::L1], [LReg::L2, LReg::L3]);
    }
    // Across the eight lanes of the lane row: each lane meets the other seven.
    p.mov(LReg::L0, LReg::L2);
    p.mov(LReg::L1, LReg::L3);
    for _ in 0..7 {
        p.rotate_row(LReg::L2, LReg::L2);
        p.rotate_row(LReg::L3, LReg::L3);
        take_better(&mut p, [LReg::L0, LReg::L1], [LReg::L2, LReg::L3]);
    }
    p.store(LReg::L0, Format::Int32, OUT_ROW);
    p.store(LReg::L1, Format::Int32, OUT_ROW + 4);
    p.finish()
}

/// Phase 2, before its first partial: no best yet.
pub fn choice_prologue() -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    p.loadi_bits(LReg::L0, NONE);
    p.loadi_bits(LReg::L1, 0);
    p.store(LReg::L0, Format::Int32, OUT_ROW);
    p.store(LReg::L1, Format::Int32, C_ROW);
    p.finish()
}

/// Phase 2, each partial in order: its best against the running one in `OUT_ROW` and `C_ROW`.
pub fn combine() -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    p.load(LReg::L0, Format::Int32, A_ROW);
    p.load(LReg::L1, Format::Int32, A_ROW + 4);
    p.load(LReg::L2, Format::Int32, OUT_ROW);
    p.load(LReg::L3, Format::Int32, C_ROW);
    take_better(&mut p, [LReg::L2, LReg::L3], [LReg::L0, LReg::L1]);
    p.store(LReg::L2, Format::Int32, OUT_ROW);
    p.store(LReg::L3, Format::Int32, C_ROW);
    p.finish()
}

/// Phase 2's finish: the index as an I32 at `(0, 0)` of the output tile (as `INDEX_TO_I32`
/// converts: `x + 2^23`, whose low mantissa bits are `x`).
pub fn index_out() -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    p.load(LReg::L0, Format::Fp32, C_ROW);
    p.loadi(LReg::L3, 8_388_608.0);
    p.add(LReg::L0, LReg::L3, LReg::L2);
    p.loadi_bits(LReg::L4, 0x4b00_0000);
    p.isub_from(LReg::L2, LReg::L4);
    p.store(LReg::L4, Format::Int32, OUT_ROW);
    p.finish()
}

/// A reduction kernel's three roles over `tiles` input tiles at `layout` (one output), the math
/// role `prologue`, then `body` once a tile stored once and repeated by the runner, then
/// `finish`: what `reduce::roles` makes with every input's program the same, at one body's size.
pub fn roles(
    layout: &super::reduce::Layout,
    tiles: usize,
    prologue: &[Instruction],
    body: &[Instruction],
    finish: &[Instruction],
) -> ([Vec<Instruction>; 3], [Vec<crate::code::Loop>; 3]) {
    use crate::datapath::{
        clear_unpacker0_adcs, config_program, pack_tile_from_dst, state_id, thread_config,
        tile_unpack_config, unpack_tile_to_dst,
    };
    use tt_isa::backend::{self, Before, ConfigWords};
    use tt_isa::sync::{self, Unit};
    let s = layout.sems;
    let mut unpack = thread_config();
    unpack.extend(clear_unpacker0_adcs());
    let mut words = ConfigWords::new();
    tile_unpack_config(&mut words, layout.in_at);
    unpack.extend(config_program(&words));
    for n in 0..tiles {
        unpack.extend(sync::take(s.consumed, Before::UNPACKER));
        unpack.extend(unpack_tile_to_dst(
            layout.in_at + n as u64 * tt_isa::dm::TILE_SLOT,
            A_ROW,
        ));
        unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
    }
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    let mut math = vec![state_id()];
    math.extend(sync::take(s.packed, Before::SFPU));
    math.extend_from_slice(prologue);
    let start = math.len() as u32;
    math.extend(sync::take(s.unpacked, Before::SFPU));
    math.extend_from_slice(body);
    math.extend(sync::post_after(Unit::Sfpu, s.consumed));
    let len = math.len() as u32 - start;
    math.extend_from_slice(finish);
    math.extend(sync::post_after(Unit::Sfpu, s.computed));
    let mut pack = vec![state_id()];
    pack.extend(sync::take(s.computed, Before::PACKER));
    pack.extend(pack_tile_from_dst(
        layout.out_at + tt_isa::dm::TILE_DATA,
        OUT_ROW,
    ));
    pack.extend(sync::post_after(Unit::Packer, s.packed));
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    let loops = if tiles > 1 {
        vec![crate::code::Loop {
            start,
            len,
            count: tiles as u32,
        }]
    } else {
        Vec::new()
    };
    ([unpack, math, pack], [Vec::new(), loops, Vec::new()])
}

/// The host's choice, as the programs make it: the first maximum, NaN above every number (the
/// first NaN), `-0` and `+0` equal.
pub fn host(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in row.iter().enumerate().skip(1) {
        let b = row[best];
        let better = if x.is_nan() {
            !b.is_nan()
        } else {
            !b.is_nan() && x > b
        };
        if better {
            best = i;
        }
    }
    best
}

/// What the device computes for `row` dealt in runs of `run` tiles, on the interpreter: each
/// run's partial, then the choice.
pub fn reference(row: &[f32], run: usize) -> usize {
    use super::interp::Vector;
    use tt_isa::dm::face_index;
    let ct = row.len().div_ceil(32);
    let tile = |j: usize| -> Vec<u32> {
        // Padding rows and columns hold NaN: never read, or never chosen.
        let mut t = vec![0x7fc0_dead; 1024];
        for c in 0..32 {
            if let Some(&x) = row.get(32 * j + c) {
                t[face_index(0, c)] = x.to_bits();
            }
        }
        t
    };
    let body = accumulate(row.len() - 1);
    let mut partials = Vec::new();
    for first in (0..ct).step_by(run.max(1)) {
        let len = run.min(ct - first);
        let mut v = Vector::new();
        v.run(&prologue(32 * first)).unwrap();
        for n in 0..len {
            v.put_tile(A_ROW as usize, &tile(first + n));
            v.run(&body).unwrap_or_else(|e| panic!("accumulate: {e}"));
        }
        v.run(&fold()).unwrap_or_else(|e| panic!("fold: {e}"));
        partials.push(v.tile(OUT_ROW as usize));
    }
    let mut v = Vector::new();
    v.run(&choice_prologue()).unwrap();
    for t in &partials {
        v.put_tile(A_ROW as usize, t);
        v.run(&combine()).unwrap_or_else(|e| panic!("combine: {e}"));
    }
    v.run(&index_out()).unwrap_or_else(|e| panic!("out: {e}"));
    v.tile(OUT_ROW as usize)[face_index(0, 0)] as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 40.0 - 20.0
            })
            .collect()
    }

    #[test]
    fn the_programs_choose_the_host_s_index() {
        for (n, run) in [
            (1024usize, 4usize),
            (2048, 13),
            (3000, 7),
            (49152, 13),
            (100, 1),
        ] {
            let mut cases = vec![row(n as u64, n)];
            // Ties: the maximum at three places, the first chosen.
            let mut tie = row(7, n);
            for &i in &[n / 3, n / 2, n - 1] {
                tie[i] = 25.0;
            }
            cases.push(tie);
            // A NaN beats infinity; the first of two NaNs; -0 and +0 equal.
            let mut nan = row(9, n);
            nan[n / 4] = f32::INFINITY;
            nan[n / 2] = -f32::NAN;
            nan[n - 1] = f32::NAN;
            cases.push(nan);
            let mut zeros = vec![-1.0f32; n];
            zeros[n / 3] = -0.0;
            zeros[n / 2] = 0.0;
            cases.push(zeros);
            // Every value equal: index 0. Subnormals and the extremes ordered as numbers.
            cases.push(vec![-3.5; n]);
            let mut small = vec![-f32::MAX; n];
            small[n / 5] = f32::from_bits(1);
            small[n / 7] = f32::from_bits(0x8000_0001);
            cases.push(small);
            for (k, case) in cases.iter().enumerate() {
                assert_eq!(
                    reference(case, run),
                    host(case),
                    "n {n}, runs of {run}, case {k}"
                );
            }
        }
    }

    #[test]
    fn the_host_rule_is_torch_s() {
        assert_eq!(host(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(host(&[1.0, f32::NAN, f32::INFINITY, f32::NAN]), 1);
        assert_eq!(host(&[-0.0, 0.0]), 0);
        assert_eq!(host(&[0.0, -0.0]), 0);
        assert_eq!(host(&[f32::NEG_INFINITY]), 0);
    }
}
