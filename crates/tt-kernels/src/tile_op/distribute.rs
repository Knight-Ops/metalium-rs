//! Grid distribution (X3): an op's output items dealt over units in contiguous runs.
//!
//! Items are the parallel iterators' tiles, row-major; each folds [`TileOp::weight`] tiles of
//! input. A run holds at most the body's group ([`Microkernel::group`]). Where the op is long
//! enough on every unit, runs are pipelined: staged in alternating halves of the arena, so a
//! unit's mover moves one run while its roles compute another (`tensor::pipelined_runs`).
//!
//! A body that computes blocks ([`Microkernel::block`]) is dealt blocks instead: the two parallel
//! iterators' tiles cut into `[rows, cols]` blocks, rows outer, a block a run.
//!
//! An operand whose map does not use the innermost parallel iterator reads one tile for a whole
//! row of items. Where runs that keep inside one such row are cheaper in transfers than the
//! plain runs (`tensor::row_runs`), the op takes them and each run reads that operand's tile
//! once and holds it.

use std::ops::Range;

use super::{Dim, Layout, TileOp};
use crate::tensor::{self, Overlap, Result, TensorError};

/// How an op's items are dealt: run `j` to unit `j % units`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Deal {
    pub runs: Vec<Range<usize>>,
    /// Where the body computes blocks, each run a block of the two parallel iterators' tiles,
    /// `[rows, cols]`, rows outer; `runs` is then empty.
    pub blocks: Vec<[Range<usize>; 2]>,
    /// Staged in alternating halves of the arena.
    pub pipelined: bool,
    /// Each operand's one tile read once a run and held (the runs keep inside its rows).
    pub held: Vec<bool>,
}

pub fn deal(op: &TileOp, units: usize, overlap: Overlap) -> Result<Deal> {
    let (items, weight) = (op.items(), op.weight());
    // Partials through GDDR chain an item's chunks on one unit: a job an item
    // (`lower::chunked_jobs`).
    if super::lower::partials_chunk(op).is_some() {
        return Ok(Deal {
            runs: (0..items).map(|i| i..i + 1).collect(),
            blocks: Vec::new(),
            pipelined: false,
            held: vec![false; op.operands.len()],
        });
    }
    if let Some(block) = op.body.block() {
        return blocks(op, block);
    }
    let group = |half: bool| op.body.group(half);
    let piped = tensor::pipelined_runs(items, weight, units, group(true).unwrap_or(0), overlap);
    let pipelined = piped.is_some();
    let plain = match piped {
        Some(r) => r,
        None => {
            let g = group(false).ok_or_else(|| {
                TensorError::Shape(format!(
                    "a run of one item of {weight} tiles does not fit one tile's L1 and program \
                     slots"
                ))
            })?;
            tensor::runs(items, units, g)
        }
    };
    let holdable = holdable(op);
    let by_rows = match (holdable.iter().any(|&h| h), op.parallel().last()) {
        (true, Some(inner)) if tensor::col_once() => {
            let ct = op.domain[inner].tiles();
            let rt = items / ct.max(1);
            let moved = |held: bool| {
                op.operands
                    .iter()
                    .zip(&holdable)
                    .filter(|(_, &h)| !(held && h))
                    .count()
                    + op.results
                        .iter()
                        .filter(|r| matches!(r.layout, Layout::Tiles(_)))
                        .count()
            };
            let held = holdable.iter().filter(|&&h| h).count();
            let max = group(pipelined).unwrap_or(0);
            tensor::row_runs(
                rt,
                ct,
                units,
                max,
                pipelined,
                &plain,
                [moved(false), moved(true), held],
            )
        }
        _ => None,
    };
    Ok(match by_rows {
        Some(runs) => Deal {
            runs,
            blocks: Vec::new(),
            pipelined,
            held: holdable,
        },
        None => Deal {
            runs: plain,
            blocks: Vec::new(),
            pipelined,
            held: vec![false; op.operands.len()],
        },
    })
}

/// The two parallel iterators' tiles in `[mb, nb]` blocks, rows outer.
fn blocks(op: &TileOp, [mb, nb]: [usize; 2]) -> Result<Deal> {
    let parallel: Vec<usize> = op.parallel().collect();
    let &[outer, inner] = &parallel[..] else {
        return Err(TensorError::Shape(format!(
            "blocks of a domain of {} parallel iterators",
            parallel.len()
        )));
    };
    let [rt, ct] = [op.domain[outer].tiles(), op.domain[inner].tiles()];
    if mb == 0 || nb == 0 {
        return Err(TensorError::Shape("an empty block".into()));
    }
    let blocks = (0..rt)
        .step_by(mb)
        .flat_map(|i0| {
            (0..ct)
                .step_by(nb)
                .map(move |j0| [i0..(i0 + mb).min(rt), j0..(j0 + nb).min(ct)])
        })
        .collect();
    Ok(Deal {
        runs: Vec::new(),
        blocks,
        pipelined: false,
        held: vec![false; op.operands.len()],
    })
}

/// Each operand that reads one tile for a whole row of items: its map does not use the
/// innermost parallel iterator, and does use an outer one (one tile a row, not one for all).
fn holdable(op: &TileOp) -> Vec<bool> {
    let parallel: Vec<usize> = op.parallel().collect();
    let Some((&inner, outer)) = parallel.split_last() else {
        return vec![false; op.operands.len()];
    };
    op.operands
        .iter()
        .map(|a| {
            op.reductions().next().is_none()
                && !a.map.uses(inner)
                && outer
                    .iter()
                    .any(|&o| matches!(a.map.rows, Dim::Iter { iter, wrap: None } if iter == o))
        })
        .collect()
}
