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
//! A domain of two parallel iterators and one reduction iterator, its runs blocks
//! ([`super::distribute::Deal::blocks`]), reads its **contraction pair** as one `GATHER`: the
//! operand whose map uses the outer parallel iterator and the reduction (`A`), and the one that
//! uses the reduction and the inner parallel iterator (`B`), each stored either way round (a map
//! with the reduction as `A`'s rows, or as `B`'s columns, is read transposed: `GATHER` bits 0
//! and 1). The record names the block's rows and columns and a chunk's terms; its formats are
//! the two edges' (bits 22-24 and 29-31), and an `A` whose last tile row is ragged is read to its
//! live rows where its access asks (bits 3-7).
//!
//! Any other map is refused by name: a mover access the description has no lowering for is a
//! missing concept (H6), not a special case.
//!
//! A reduction whose contract carries its partials through GDDR ([`Combine::GddrPartials`]) is
//! lowered by [`chunked_jobs`]: a job an item, a chunk of its terms a step, each chunk after the
//! first reloading the partial the last wrote to the result.

use tt_isa::dm::record;

use tt_isa::format::TileFormat;

use super::distribute::Deal;
use super::{Access, Combine, Dim, IterKind, Layout, RunCode, TileMap, TileOp};
use crate::tensor::{self, Job, Result, Step, TensorError};

pub fn jobs(op: &TileOp, deal: &Deal, units: usize) -> Result<Vec<Job>> {
    let parallel: Vec<usize> = op.parallel().collect();
    let reductions: Vec<usize> = op.reductions().collect();
    if let Some(chunk) = partials_chunk(op) {
        return chunked_jobs(op, chunk);
    }
    if !deal.blocks.is_empty() {
        return block_jobs(op, deal);
    }
    let [gather_what, scatter_what] = op.body.lists(false);
    let weight = op.weight();
    let mut jobs = Vec::with_capacity(deal.runs.len());
    for (j, run) in deal.runs.iter().enumerate() {
        let len = run.len();
        let half = deal.pipelined.then(|| tensor::half_of(j, units));
        let code = op.body.run(len, half, &deal.held)?;
        let mut gather = Vec::with_capacity(3 * op.operands.len());
        for (o, a) in op.operands.iter().enumerate() {
            if let Some(at) = code.operands.get(o).copied().flatten() {
                let span = run.start * weight..run.end * weight;
                read(
                    op,
                    a,
                    &parallel,
                    &reductions,
                    run,
                    span,
                    at,
                    deal.held[o],
                    &mut gather,
                )?;
            }
        }
        let mut scatter = Vec::with_capacity(3 * op.results.len());
        for r in &op.results {
            write(r, run, code.out, &mut scatter);
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
            Combine::InDst | Combine::L1Halves { .. } => None,
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
                    read(
                        op,
                        a,
                        &parallel,
                        &reductions,
                        &items,
                        span.clone(),
                        at,
                        false,
                        &mut gather,
                    )?;
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
            let mut scatter = Vec::with_capacity(3);
            write(result, &items, code.out, &mut scatter);
            steps.push(Step::List {
                what: scatter_what,
                entries: scatter,
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
    out: &mut Vec<[u32; 8]>,
) -> Result<()> {
    let r = tiles_of(a)?.encode();
    let tiles = |i: usize| op.domain[i].tiles();
    let len = run.len();
    let unsupported = || {
        TensorError::Shape(format!(
            "no mover access lowers the map {:?} over parallel {parallel:?} and reductions \
             {reductions:?}",
            a.map
        ))
    };
    let mut push = |h: [u32; 8]| out.extend([h, r[0], r[1]]);
    match (parallel, reductions, a.map) {
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
                    push(run_record(run.start, len, at, 0, ct, 0))
                }
                (
                    Dim::Fixed,
                    Dim::Iter {
                        iter: c,
                        wrap: None,
                    },
                ) if c == *inner => push(run_record(run.start, len, at, 1, ct, 0)),
                (Dim::Iter { iter, wrap: None }, Dim::Fixed) if iter == *outer => {
                    let count = if held { 1 } else { len };
                    push(run_record(
                        run.start,
                        count,
                        at,
                        record::READ_RUN_COL_AS_IT_LIES,
                        ct,
                        0,
                    ))
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
                    for h in tensor::block_reads(run.start, len, w * ct, at, ct) {
                        push(h);
                    }
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
                push(run_record(span.start, span.len(), at, 0, tiles(*k), 0))
            }
            (
                Dim::Iter { iter, wrap: None },
                Dim::Iter {
                    iter: c,
                    wrap: None,
                },
            ) if iter == *k && c == *p => push(run_record(
                span.start,
                span.len(),
                at,
                4,
                tiles(*p),
                tiles(*k) as u32,
            )),
            _ => return Err(unsupported()),
        },
        _ => return Err(unsupported()),
    }
    Ok(())
}

/// A result's tiles for one run, from the slots at `at`.
fn write(r: &Access, run: &std::ops::Range<usize>, at: u64, out: &mut Vec<[u32; 8]>) {
    match &r.layout {
        Layout::Tiles(t) => {
            let t = t.encode();
            out.extend([
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
            ]);
        }
        Layout::PackedRow(twin) => {
            out.extend(tensor::packed_twin_writes(twin, run.start, run.len(), at))
        }
    }
}

/// The contraction a domain of two parallel iterators and one reduction holds: `(a, b)`, the
/// operands of the pair, and whether each is stored transposed.
struct Contraction {
    a: usize,
    b: usize,
    transposed: [bool; 2],
}

fn contraction(op: &TileOp) -> Result<(Contraction, [usize; 3])> {
    let parallel: Vec<usize> = op.parallel().collect();
    let reductions: Vec<usize> = op.reductions().collect();
    let (&[p, q], &[k]) = (&parallel[..], &reductions[..]) else {
        return Err(TensorError::Shape(format!(
            "a contraction over parallel {parallel:?} and reductions {reductions:?}"
        )));
    };
    let plain = |i| Dim::Iter {
        iter: i,
        wrap: None,
    };
    // Each operand's map read as `[x, k]` (`A`) or `[k, y]` (`B`), either way round.
    let side = |m: TileMap, x: usize, a: bool| match (m.rows, m.cols) {
        (r, c) if r == plain(x) && c == plain(k) => Some(!a),
        (r, c) if r == plain(k) && c == plain(x) => Some(a),
        _ => None,
    };
    let find = |x: usize, a: bool| {
        op.operands
            .iter()
            .enumerate()
            .find_map(|(o, acc)| side(acc.map, x, a).map(|t| (o, t)))
    };
    match (find(p, true), find(q, false)) {
        (Some((a, ta)), Some((b, tb))) if a != b => Ok((
            Contraction {
                a,
                b,
                transposed: [ta, tb],
            },
            [p, k, q],
        )),
        _ => Err(TensorError::Shape(format!(
            "no operand pair of maps {:?} contracts iterator {k}",
            op.operands.iter().map(|a| a.map).collect::<Vec<_>>()
        ))),
    }
}

/// `record::gather_a_format` or `gather_b_format` of an edge read in its own format, 0 for FP32.
fn edge_format(a: &Access, of: fn(tt_isa::tile::L1Format) -> Option<u32>) -> Result<u32> {
    let f = a.format();
    if f == TileFormat::FP32 {
        return Ok(0);
    }
    of(f.l1()).ok_or_else(|| TensorError::Shape(format!("no gather reads a {f:?} edge")))
}

/// The jobs of a contraction dealt in blocks, its terms in chunks continued in L1
/// ([`Combine::L1Halves`]): a job a block, three steps a chunk, the results written by the last.
fn block_jobs(op: &TileOp, deal: &Deal) -> Result<Vec<Job>> {
    let (pair, [p, k, _]) = contraction(op)?;
    let chunk = match op.domain[k].kind {
        IterKind::Reduction(c) => match c.combine {
            Combine::L1Halves { chunk } => chunk,
            other => {
                return Err(TensorError::Shape(format!(
                    "blocks whose terms combine {other:?}"
                )))
            }
        },
        IterKind::Parallel => unreachable!("a contraction's reduction"),
    };
    let [gather_what, scatter_what] = op.body.lists(false);
    let weight = op.weight();
    let chunks = weight.div_ceil(chunk.max(1));
    let (ra, rb) = (
        tiles_of(&op.operands[pair.a])?,
        tiles_of(&op.operands[pair.b])?,
    );
    let result = op
        .results
        .first()
        .ok_or_else(|| TensorError::Shape("a contraction has no result".into()))?;
    let rc = tiles_of(result)?.encode();
    let [ta, tb] = pair.transposed;
    let formats = edge_format(&op.operands[pair.a], record::gather_a_format)?
        | edge_format(&op.operands[pair.b], record::gather_b_format)?;
    let unpack_b = op.body.transposes(pair.b);
    // The outer iterator's last tile row: the rows that hold data.
    let (rt, ragged) = (op.domain[p].tiles(), op.domain[p].elems % 32);
    let reversed = tensor::probing_reverse_chunks();
    let mut jobs = Vec::with_capacity(deal.blocks.len());
    for [rows, cols] in &deal.blocks {
        let last_row = rows.end == rt && ragged != 0;
        let live = if last_row { ragged as u32 } else { 32 };
        // `A` read to its live rows: one row of each tile for a one-row product.
        let thin = match last_row && op.operands[pair.a].live_rows && !ta {
            true if ragged <= 16 => ragged as u32,
            _ => 0,
        };
        let mut steps = Vec::with_capacity(3 * chunks);
        for c in 0..chunks {
            let range = if reversed { chunks - 1 - c } else { c };
            let k0 = range * chunk;
            let count = chunk.min(weight - k0);
            let code: RunCode = op
                .body
                .block_chunk(c, [rows.len(), count, cols.len()], live)?;
            let (mut ar, mut br) = (ra, rb);
            ar.first += if ta { k0 * ar.ct as usize } else { k0 } as u32;
            br.first += if tb { k0 } else { k0 * br.ct as usize } as u32;
            let (a_at, b_at) = match (code.operands[pair.a], code.operands[pair.b]) {
                (Some(a), Some(b)) => (a, b),
                _ => {
                    return Err(TensorError::Shape(
                        "a contraction's body names no slots for its pair".into(),
                    ))
                }
            };
            let flags = u32::from(ta)
                | u32::from(tb) << 1
                | record::gather_rows(thin)
                | (count as u32) << 8
                | if tensor::probing_skip_a() {
                    record::GATHER_A_RESIDENT
                } else {
                    0
                }
                | if unpack_b {
                    record::GATHER_B_UNPACK_TRANSPOSE
                } else {
                    0
                }
                | formats;
            let gather = if tensor::probing_skip_block_gathers() {
                vec![[tt_isa::dm::op::WAIT, 0, 0, 0, 0, 0, 0, 0]]
            } else {
                let (ar, br) = (ar.encode(), br.encode());
                vec![
                    [
                        record::GATHER,
                        flags,
                        a_at as u32,
                        b_at as u32,
                        rows.start as u32,
                        rows.len() as u32,
                        cols.start as u32,
                        cols.len() as u32,
                    ],
                    ar[0],
                    ar[1],
                    br[0],
                    br[1],
                ]
            };
            let scatter = if c + 1 == chunks {
                vec![
                    [
                        record::SCATTER,
                        code.out as u32,
                        code.stride as u32,
                        rows.start as u32,
                        rows.len() as u32,
                        cols.start as u32,
                        cols.len() as u32,
                        0,
                    ],
                    rc[0],
                    rc[1],
                ]
            } else {
                Vec::new()
            };
            steps.extend([
                Step::List {
                    what: gather_what,
                    entries: gather,
                },
                Step::Kernel {
                    roles: code.roles,
                    init: code.init,
                    mop: Box::new([None; 3]),
                    loops: code.loops,
                    half: Some((c % 2) as u8),
                },
                Step::List {
                    what: scatter_what,
                    entries: scatter,
                },
            ]);
        }
        jobs.push(steps);
    }
    Ok(jobs)
}
