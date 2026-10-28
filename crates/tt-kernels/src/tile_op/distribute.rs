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

use super::{Blocks, Cols, Dim, Layout, TileOp};
use crate::tensor::{self, Result, TensorError};

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
    if let Some(cut) = op.body.block() {
        return blocks(op, cut);
    }
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
    let group = |half: bool| op.body.group(half);
    let piped = pipelined_runs(items, weight, units, group(true).unwrap_or(0), overlap);
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
        (true, Some(inner)) if col_once() => {
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
            row_runs(
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

/// The two parallel iterators' tiles in blocks as `cut` says, rows outer.
fn blocks(op: &TileOp, cut: Blocks<'_>) -> Result<Deal> {
    let parallel: Vec<usize> = op.parallel().collect();
    let &[outer, inner] = &parallel[..] else {
        return Err(TensorError::Shape(format!(
            "blocks of a domain of {} parallel iterators",
            parallel.len()
        )));
    };
    let [rt, ct] = [op.domain[outer].tiles(), op.domain[inner].tiles()];
    let mb = cut.rows;
    let cols: Vec<Range<usize>> = match cut.cols {
        Cols::Every(nb) if nb > 0 => (0..ct)
            .step_by(nb)
            .map(|j0| j0..(j0 + nb).min(ct))
            .collect(),
        Cols::Runs(runs) if runs.iter().all(|r| !r.is_empty() && r.end <= ct) => runs.to_vec(),
        _ => {
            return Err(TensorError::Shape(
                "a block's columns are empty or past the domain".into(),
            ))
        }
    };
    if mb == 0 {
        return Err(TensorError::Shape("an empty block".into()));
    }
    let blocks = (0..rt)
        .step_by(mb)
        .flat_map(|i0| cols.iter().map(move |c| [i0..(i0 + mb).min(rt), c.clone()]))
        .collect();
    Ok(Deal {
        runs: Vec::new(),
        blocks,
        pipelined: cut.pipelined,
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

/// Fewest tiles a pipelined element-wise or reduce run may hold. Each run
/// is a launch and a gather and scatter record of its own: card 0 lost
/// 5-50% splitting ops into runs of 1-4 tiles.
pub const MIN_PIPELINED_RUN: usize = 12;

/// How the runs of an element-wise or reduce op may overlap: the movers moving
/// one run while the roles compute another. More runs cost the host a list
/// each, which a captured op pays once and a replay never, so a trace may
/// overlap where a fresh op may not.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Overlap {
    /// One run a unit, serialized (`Session::set_pipeline(false)`).
    Off,
    /// An op queued now: the host pays for every run.
    Fresh,
    /// An op being captured into a trace.
    Captured,
}

/// Fewest tiles a unit's share of a fresh pipelined op may hold on one unit.
/// Card 0 (`streaming_performance_sweep`, `sfpu_pipeline_sweep`; host end to
/// end, run 1791079325 and after): on one tile every element-wise op or
/// reduction from 49 tiles gained (0.63-0.89), and small ones forced to
/// overlap still did (a max over rows of 64 tiles 0.88).
pub const PIPELINE_SHARE: usize = 48;

/// What each further unit adds to [`PIPELINE_SHARE`], in tiles a unit. The host
/// queues the units' lists one after another (~6 us a unit an op), so with many
/// units a fresh op is the host's, and the extra runs overlap makes cost more
/// than it saves until each unit has a great deal to move. Adds on 8 tiles
/// lost 1.07-1.20 up to 1152 tiles a unit and gained from 1568 (0.97 there,
/// 0.87 at 2048 a unit); on 32 tiles they lost 1.37-1.44 at every size to 1152
/// a unit; reductions on 8 and 32 tiles lost up to 1.28 or broke even.
pub const PIPELINE_SHARE_PER_UNIT: usize = 180;

/// Fewest tiles a unit's share of a captured pipelined op may hold, per unit.
/// A replay pays no host time for its runs. Adds on 8 tiles gained 0.90 at 32
/// tiles a unit and 0.77 at 128; on 32 tiles 0.95 at 32 and 0.93 at 128, and
/// lost 1.016 at 8 a unit, where [`MIN_PIPELINED_RUN`] leaves one run anyway.
pub const CAPTURED_PIPELINE_SHARE: usize = 32;

/// Tiles a unit's share must hold for `overlap` on `units` units, scaled by
/// [`set_pipeline_share_percent`].
pub fn pipeline_share(overlap: Overlap, units: usize) -> usize {
    let share = match overlap {
        Overlap::Off => usize::MAX,
        Overlap::Fresh => PIPELINE_SHARE + PIPELINE_SHARE_PER_UNIT * (units.max(1) - 1),
        Overlap::Captured => CAPTURED_PIPELINE_SHARE,
    };
    if share == usize::MAX {
        return share;
    }
    share * PIPELINE_SHARE_PERCENT.load(std::sync::atomic::Ordering::Relaxed) / 100
}

/// Percent of [`pipeline_share`] that [`pipelined_runs`] asks for; 100 is the
/// constants as they stand. A benchmark lowers it to see what overlap does
/// where the constants say no (`sfpu_pipeline_sweep`,
/// `streaming_performance_sweep`); nothing else sets it.
static PIPELINE_SHARE_PERCENT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(100);

/// Set [`PIPELINE_SHARE_PERCENT`]; returns what it was.
pub fn set_pipeline_share_percent(percent: usize) -> usize {
    PIPELINE_SHARE_PERCENT.swap(percent, std::sync::atomic::Ordering::Relaxed)
}

/// The runs of a pipelined element-wise or reduce op over `len` items of
/// `weight` tiles each, at most `half_max` items a run (half the arena's
/// worth): at least two a unit, so every unit has a run moving while one
/// computes, and the same number on every unit -- the op takes as long as its
/// busiest unit, and one extra run on a few units cost card 0 up to 25%
/// (`exp` over 1024 tiles on 8: 18 runs, two units doing three). `None` where
/// `overlap` is off, that would leave a run under [`MIN_PIPELINED_RUN`] tiles,
/// a unit's share under [`pipeline_share`] tiles, or `half_max` is zero.
pub fn pipelined_runs(
    len: usize,
    weight: usize,
    units: usize,
    half_max: usize,
    overlap: Overlap,
) -> Option<Vec<std::ops::Range<usize>>> {
    let units = units.max(1);
    let share = pipeline_share(overlap, units);
    if half_max == 0 || overlap == Overlap::Off || len * weight < share.saturating_mul(units) {
        return None;
    }
    let parts = len.div_ceil(half_max).div_ceil(units).max(2) * units;
    // Contiguous, as even as integer division allows.
    let r: Vec<_> = (0..parts)
        .map(|p| p * len / parts..(p + 1) * len / parts)
        .collect();
    let shortest = r.iter().map(|r| r.len()).min()?;
    (shortest * weight >= MIN_PIPELINED_RUN).then_some(r)
}

/// Which half of the arena run `j` of a pipelined op is staged in: a unit
/// takes runs `j`, `j + units`, ... in turn, and alternates halves.
pub(crate) fn half_of(j: usize, units: usize) -> u8 {
    ((j / units.max(1)) % 2) as u8
}

/// Whether a column broadcast runs by tile rows ([`row_runs`]); on unless `TT_COL_ONCE=0`.
static COL_ONCE: std::sync::LazyLock<std::sync::atomic::AtomicBool> =
    std::sync::LazyLock::new(|| {
        std::sync::atomic::AtomicBool::new(std::env::var("TT_COL_ONCE").map_or(true, |v| v != "0"))
    });

/// Set whether a column broadcast's runs keep to tile rows and read their `B` tile once
/// ([`row_runs`]); returns what it was. A gate's negative control and a benchmark's comparison.
pub fn set_col_once(on: bool) -> bool {
    COL_ONCE.swap(on, std::sync::atomic::Ordering::Relaxed)
}

/// Whether a column broadcast's runs keep to tile rows ([`set_col_once`]).
pub(crate) fn col_once() -> bool {
    COL_ONCE.load(std::sync::atomic::Ordering::Relaxed)
}

/// A run's fixed cost (its lists, the kernel's start and the handshakes) in tile transfers, which
/// [`row_runs`] weighs against the tiles a busiest unit moves.
const RUN_FIXED_TRANSFERS: usize = 4;

/// The runs of an op with an operand that reads one tile a row (a column broadcast, `B` a
/// `[rows, 1]` column) over `rt x ct` tiles, each inside one tile row, so a run reads its row's
/// one tile of that operand once (`kernel::Layout::b_once`), not once an output tile: on a long
/// softmax row that read was half the op's bytes, 120 units reading 64 small tiles
/// (`docs/plans/long-context.md`, "Long-prompt softmax passes"). Each row is cut in `s` runs of
/// at most `max` tiles, as even as integer division allows. The runs are dealt `j % units`, so an
/// op takes as long as its busiest unit: `[old, new, held]` are the tiles a run moves for each of
/// its items in `old`'s runs (the ones it would replace: every operand in and every result out;
/// three for a column's broadcast), the same with the held operands read once (two), and the
/// held operands' tiles a run (one), each run with [`RUN_FIXED_TRANSFERS`]. `s` is the
/// cheapest from `ct.div_ceil(max)` up to twice that where pipelined (each run at least
/// [`MIN_PIPELINED_RUN`] tiles and two a unit), or up to one tile a run otherwise. `None` unless
/// that is cheaper than `old`: a decode row or a short chunk's rows keep their spread over the
/// units (cut by rows, one unit's share of a short row would be a whole one).
pub fn row_runs(
    rt: usize,
    ct: usize,
    units: usize,
    max: usize,
    pipelined: bool,
    old: &[std::ops::Range<usize>],
    [per_old, per_new, held]: [usize; 3],
) -> Option<Vec<std::ops::Range<usize>>> {
    let units = units.max(1);
    if max == 0 || rt == 0 || ct == 0 || (pipelined && ct < MIN_PIPELINED_RUN) {
        return None;
    }
    let least = ct.div_ceil(max);
    let most = if pipelined {
        (2 * least).min(ct / MIN_PIPELINED_RUN)
    } else {
        ct
    };
    let busiest = |runs: usize, longest: usize, per_tile: usize, extra: usize| {
        runs.div_ceil(units) * (longest * per_tile + extra + RUN_FIXED_TRANSFERS)
    };
    let now = busiest(old.len(), old.iter().map(|r| r.len()).max()?, per_old, 0);
    let (cost, s) = (least..=most)
        .filter(|&s| !pipelined || rt * s >= 2 * units)
        // On a tie, more units reading (up to one run each), then fewer runs.
        .map(|s| {
            (
                busiest(rt * s, ct.div_ceil(s), per_new, held),
                std::cmp::Reverse((rt * s).min(units)),
                s,
            )
        })
        .min()
        .map(|(cost, _, s)| (cost, s))?;
    if cost >= now {
        return None;
    }
    Some(
        (0..rt)
            .flat_map(|i| (0..s).map(move |p| i * ct + p * ct / s..i * ct + (p + 1) * ct / s))
            .collect(),
    )
}
