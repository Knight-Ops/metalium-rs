//! Lowering (L1 to L3): each run of a [`Deal`] becomes a job of three steps, the mover's gather
//! of every operand's tiles into the run's slots, the body's kernel, and the mover's scatter of
//! the results.
//!
//! What the mover reads for an operand follows from its [`TileMap`] against the order a run
//! walks the domain: items (the parallel iterators, row-major), then each item's
//! [`TileOp::weight`] tiles (the reduction iterators). The mover's addressing modes are the
//! architecture's (`tt_isa::dm::record::READ_RUN`); which one an operand takes is read off its
//! map, never off the op:
//!
//! | rows | cols | read |
//! |---|---|---|
//! | outer parallel | inner parallel | the run's tiles, row-major |
//! | none | inner parallel | the row's tile for every item (`flags` 1) |
//! | outer parallel | none | the column's tile as it lies (`READ_RUN_COL_AS_IT_LIES`), once a run where held |
//! | outer parallel, wrapped | inner parallel | the wrapped rows, a `READ_RUN` a wrap (`tensor::block_reads`) |
//! | parallel | reduction | each item's tiles in turn, row-major |
//! | reduction | parallel | each item's tiles in turn, column-major (`flags` 4) |
//!
//! Any other map is refused by name: a mover access the description has no lowering for is a
//! missing concept (H6), not a special case.
//!
//! A reduction whose contract carries its partials through GDDR ([`Combine::GddrPartials`]) is
//! lowered by [`chunked_jobs`]: a job an item, a chunk of its terms a step, each chunk after the
//! first reloading the partial the last wrote to the result.

use tt_isa::dm::record;

use super::distribute::Deal;
use super::{Access, Combine, Dim, IterKind, Layout, TileMap, TileOp};
use crate::tensor::{self, Job, Result, Step, TensorError};

pub fn jobs(op: &TileOp, deal: &Deal, units: usize) -> Result<Vec<Job>> {
    let parallel: Vec<usize> = op.parallel().collect();
    let reductions: Vec<usize> = op.reductions().collect();
    if let Some(chunk) = partials_chunk(op) {
        return chunked_jobs(op, chunk);
    }
    let [gather_what, scatter_what] = op.body.lists(false);
    let weight = op.weight();
    let mut jobs = Vec::with_capacity(deal.runs.len());
    for (j, run) in deal.runs.iter().enumerate() {
        let len = run.len();
        let half = deal.pipelined.then(|| tensor::half_of(j, units));
        let code = op.body.run(len, half, &deal.held)?;
        let mut gather = Vec::new();
        for (o, a) in op.operands.iter().enumerate() {
            if let Some(at) = code.operands.get(o).copied().flatten() {
                let span = run.start * weight..run.end * weight;
                gather.extend(read(
                    op,
                    a,
                    &parallel,
                    &reductions,
                    run,
                    span,
                    at,
                    deal.held[o],
                )?);
            }
        }
        let mut scatter = Vec::new();
        for r in &op.results {
            scatter.extend(write(r, run, code.out)?);
        }
        jobs.push(vec![
            Step::List {
                what: gather_what,
                entries: gather,
            },
            Step::Kernel {
                roles: code.roles,
                init: code.init,
                mop: Box::new([None; 3]),
                loops: code.loops,
                half,
            },
            Step::List {
                what: scatter_what,
                entries: scatter,
            },
        ]);
    }
    Ok(jobs)
}

/// The chunk a reduction iterator's contract splits its terms in, carrying partials through GDDR.
pub fn partials_chunk(op: &TileOp) -> Option<usize> {
    op.domain.iter().find_map(|i| match i.kind {
        IterKind::Reduction(c) => match c.combine {
            Combine::GddrPartials { chunk } => Some(chunk),
            Combine::InDst => None,
        },
        IterKind::Parallel => None,
    })
}

/// A reduction in chunks of `chunk` terms: each item's chunks in order, one job, so its partial
/// passes from chunk to chunk through the result's tile on one unit. A chunk after the first waits
/// for the last chunk's write, then reloads the partial ([`super::RunCode::prior`]).
pub fn chunked_jobs(op: &TileOp, chunk: usize) -> Result<Vec<Job>> {
    let parallel: Vec<usize> = op.parallel().collect();
    let reductions: Vec<usize> = op.reductions().collect();
    let [gather_what, scatter_what] = op.body.lists(true);
    let weight = op.weight();
    let result = op
        .results
        .first()
        .ok_or_else(|| TensorError::Shape("a chunked reduction has no result".into()))?;
    let partial = tiles_of(result)?.encode();
    let mut jobs = Vec::with_capacity(op.items());
    for item in 0..op.items() {
        let mut steps = Vec::new();
        for first in (0..weight).step_by(chunk.max(1)) {
            let count = chunk.min(weight - first);
            let code = op.body.chunk(first > 0, count, first + count == weight)?;
            let mut gather = Vec::new();
            if first > 0 {
                let prior = code.prior.ok_or_else(|| {
                    TensorError::Shape("a chunk after the first has no slot for its partial".into())
                })?;
                // The reload depends on the last chunk's write to GDDR.
                gather.push([tt_isa::dm::op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
                gather.extend([
                    [record::READ_RUN, item as u32, 1, prior as u32, 0, 0, 0, 0],
                    partial[0],
                    partial[1],
                ]);
            }
            let items = item..item + 1;
            let span = item * weight + first..item * weight + first + count;
            for (o, a) in op.operands.iter().enumerate() {
                if let Some(at) = code.operands.get(o).copied().flatten() {
                    gather.extend(read(
                        op,
                        a,
                        &parallel,
                        &reductions,
                        &items,
                        span.clone(),
                        at,
                        false,
                    )?);
                }
            }
            steps.push(Step::List {
                what: gather_what,
                entries: gather,
            });
            steps.push(Step::Kernel {
                roles: code.roles,
                init: code.init,
                mop: Box::new([None; 3]),
                loops: code.loops,
                half: None,
            });
            steps.push(Step::List {
                what: scatter_what,
                entries: write(result, &items, code.out)?,
            });
        }
        jobs.push(steps);
    }
    Ok(jobs)
}

/// The `READ_RUN` header of a run of `count` tiles from `first`, into slots from `at`.
fn run_record(first: usize, count: usize, at: u64, flags: u32, ct: usize, rt: u32) -> [u32; 8] {
    [
        record::READ_RUN,
        first as u32,
        count as u32,
        at as u32,
        flags,
        ct as u32,
        rt,
        0,
    ]
}

fn tiles_of(a: &Access) -> Result<tt_isa::dm::record::TensorRef> {
    match &a.layout {
        Layout::Tiles(r) => Ok(*r),
        Layout::PackedRow(_) => Err(TensorError::Shape(
            "an operand read packed one row is not a tile edge".into(),
        )),
    }
}

/// An operand's tiles for the items `run`, as its map says (the module's table): `span`, the
/// tiles of the walk they fold (items times [`TileOp::weight`], or a chunk of one item's).
#[allow(clippy::too_many_arguments)]
fn read(
    op: &TileOp,
    a: &Access,
    parallel: &[usize],
    reductions: &[usize],
    run: &std::ops::Range<usize>,
    span: std::ops::Range<usize>,
    at: u64,
    held: bool,
) -> Result<Vec<[u32; 8]>> {
    let r = tiles_of(a)?.encode();
    let tiles = |i: usize| op.domain[i].tiles();
    let with_ref = |h: [u32; 8]| [h, r[0], r[1]];
    let len = run.len();
    let unsupported = || {
        TensorError::Shape(format!(
            "no mover access lowers the map {:?} over parallel {parallel:?} and reductions \
             {reductions:?}",
            a.map
        ))
    };
    let headers: Vec<[u32; 8]> = match (parallel, reductions, a.map) {
        // Element-wise: the domain's two parallel iterators.
        ([outer, inner], [], TileMap { rows, cols }) => {
            let ct = tiles(*inner);
            match (rows, cols) {
                (
                    Dim::Iter { iter, wrap: None },
                    Dim::Iter {
                        iter: c,
                        wrap: None,
                    },
                ) if iter == *outer && c == *inner => {
                    vec![run_record(run.start, len, at, 0, ct, 0)]
                }
                (
                    Dim::Fixed,
                    Dim::Iter {
                        iter: c,
                        wrap: None,
                    },
                ) if c == *inner => {
                    vec![run_record(run.start, len, at, 1, ct, 0)]
                }
                (Dim::Iter { iter, wrap: None }, Dim::Fixed) if iter == *outer => {
                    let count = if held { 1 } else { len };
                    vec![run_record(
                        run.start,
                        count,
                        at,
                        record::READ_RUN_COL_AS_IT_LIES,
                        ct,
                        0,
                    )]
                }
                (
                    Dim::Iter {
                        iter,
                        wrap: Some(w),
                    },
                    Dim::Iter {
                        iter: c,
                        wrap: None,
                    },
                ) if iter == *outer && c == *inner => {
                    tensor::block_reads(run.start, len, w * ct, at, ct)
                }
                _ => return Err(unsupported()),
            }
        }
        // A reduction: each item's terms in turn.
        ([p], [k], TileMap { rows, cols }) => match (rows, cols) {
            (
                Dim::Iter { iter, wrap: None },
                Dim::Iter {
                    iter: c,
                    wrap: None,
                },
            ) if iter == *p && c == *k => {
                vec![run_record(span.start, span.len(), at, 0, tiles(*k), 0)]
            }
            (
                Dim::Iter { iter, wrap: None },
                Dim::Iter {
                    iter: c,
                    wrap: None,
                },
            ) if iter == *k && c == *p => {
                vec![run_record(
                    span.start,
                    span.len(),
                    at,
                    4,
                    tiles(*p),
                    tiles(*k) as u32,
                )]
            }
            _ => return Err(unsupported()),
        },
        _ => return Err(unsupported()),
    };
    Ok(headers.into_iter().flat_map(with_ref).collect())
}

/// A result's tiles for one run, from the slots at `at`.
fn write(r: &Access, run: &std::ops::Range<usize>, at: u64) -> Result<Vec<[u32; 8]>> {
    Ok(match &r.layout {
        Layout::Tiles(t) => {
            let t = t.encode();
            vec![
                [
                    record::WRITE_RUN,
                    run.start as u32,
                    run.len() as u32,
                    at as u32,
                    0,
                    0,
                    0,
                    0,
                ],
                t[0],
                t[1],
            ]
        }
        Layout::PackedRow(twin) => tensor::packed_twin_writes(twin, run.start, run.len(), at),
    })
}
