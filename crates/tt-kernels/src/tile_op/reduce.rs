//! A reduction over one axis as a [`TileOp`] (`sfpu::reduce`): `[rows, cols]` to `[rows, 1]` over
//! the columns, or to `[1, cols]` over the rows, with a finishing kind applied before the pack
//! ([`ReduceThen`]).

use super::{
    allocate, Access, At, Combine, Dim, Iter, Layout, Microkernel, NumericContract,
    ReductionContract, RunCode, TileMap, TileOp,
};
use crate::l1::{Placed, Requirements};
use crate::sfpu::reduce::{continuation_roles, Axis, ChunkLayout, ReduceOp};
use crate::tensor::{self, DramTensor, ReduceThen, Result, TensorError};
use std::sync::{Arc, OnceLock};

/// `a` reduced over `axis` by `op`, then `then`'s exact unary kind on the result before its pack:
/// bit for bit the kind run on the stored result (a softmax's `clamp_min(max(x))`). Each output
/// tile folds its row (or column) of tiles in one run where its programs fit one tile; a longer
/// one is chunked, its unfolded partial carried through the result's tile in GDDR
/// ([`Combine::GddrPartials`], chunks of `sfpu::reduce::ROW_CHUNK`), which takes no `then`.
pub fn reduce(a: &DramTensor, op: ReduceOp, axis: Axis, then: ReduceThen) -> Result<TileOp> {
    a.expect("a reduction", op.elem())?;
    let (per, valid) = match axis {
        Axis::Cols => (a.grid()[1], a.cols % 32),
        Axis::Rows => (a.grid()[0], a.rows % 32),
    };
    let valid = if valid == 0 { 32 } else { valid as u32 };
    let fits = tensor::reduce_group(op, axis, per, valid, false, then).is_some();
    let combine = if fits {
        Combine::InDst
    } else {
        if then.is_some() {
            return Err(TensorError::Shape(
                "a reduction too long for one tile's programs takes no finishing kind".into(),
            ));
        }
        Combine::GddrPartials {
            chunk: crate::sfpu::reduce::ROW_CHUNK,
        }
    };
    let contract = ReductionContract { combine };
    // The output's iterator first, then the folded one: rows then columns over the columns, and
    // over the rows the rows (folded) then the columns.
    let (domain, out_map, dims) = match axis {
        Axis::Cols => (
            vec![Iter::parallel(a.rows), Iter::reduction(a.cols, contract)],
            TileMap {
                rows: Dim::of(0),
                cols: Dim::Fixed,
            },
            [a.rows, 1],
        ),
        Axis::Rows => (
            vec![Iter::reduction(a.rows, contract), Iter::parallel(a.cols)],
            TileMap {
                rows: Dim::Fixed,
                cols: Dim::of(1),
            },
            [1, a.cols],
        ),
    };
    Ok(TileOp {
        domain,
        operands: vec![Access {
            layout: Layout::Tiles(a.tensor_ref()),
            map: TileMap {
                rows: Dim::of(0),
                cols: Dim::of(1),
            },
            elem: a.elem,
            live_rows: false,
        }],
        results: vec![Access {
            // Placed when the op is built (`TileOp::build`).
            layout: Layout::Tiles(Default::default()),
            map: out_map,
            elem: op.elem(),
            live_rows: false,
        }],
        body: Box::new(Reduce {
            op,
            axis,
            per,
            valid,
            then,
            chunked: !fits,
        }),
        contract: NumericContract::Program,
        ahead: Vec::new(),
        share: None,
        out: (dims, op.elem()),
    })
}

/// A reduction's programs over a run of output tiles (`sfpu::reduce`): the mover stages each
/// output's `per` input tiles, the roles fold them and pack the outputs.
struct Reduce {
    op: ReduceOp,
    axis: Axis,
    /// Input tiles an output folds.
    per: usize,
    /// Valid rows (or columns) of the last of them.
    valid: u32,
    then: ReduceThen,
    /// Its terms in chunks ([`Combine::GddrPartials`]).
    chunked: bool,
}

impl Microkernel for Reduce {
    fn lists(&self, chunked: bool) -> [&'static str; 2] {
        if chunked {
            ["reduce continuation gather", "reduce continuation scatter"]
        } else {
            ["reduce gather", "reduce scatter"]
        }
    }

    fn group(&self, half: bool) -> Option<usize> {
        tensor::reduce_group(self.op, self.axis, self.per, self.valid, half, self.then)
    }

    fn needs(&self, at: &At<'_>) -> Result<Arc<Requirements>> {
        type Memo = allocate::Declared<(usize, usize)>;
        static RUNS: Memo = OnceLock::new();
        static CHUNKS: Memo = OnceLock::new();
        match *at {
            At::Run { len, .. } => {
                let per = self.per;
                Ok(allocate::declared(&RUNS, (len, per), || {
                    crate::sfpu::reduce::requirements(len, per)
                }))
            }
            At::Item { .. } if self.chunked => Ok(allocate::declared(&CHUNKS, (1, 0), || {
                crate::sfpu::reduce::chunk_requirements(1)
            })),
            _ => Err(at.refused()),
        }
    }

    fn code(&self, at: &At<'_>, l1: &Placed) -> Result<RunCode> {
        match *at {
            At::Run { len, half, .. } => self.run(len, half, l1),
            At::Item { prior, count, last } if self.chunked => self.chunk(prior, count, last, l1),
            _ => Err(at.refused()),
        }
    }
}

impl Reduce {
    fn run(&self, len: usize, half: Option<u8>, l1: &Placed) -> Result<RunCode> {
        let (layout, roles, loops) = tensor::reduce_programs(
            self.op, self.axis, len, self.per, self.valid, half, self.then, l1,
        )?;
        Ok(RunCode {
            operands: [Some(layout.in_at), None, None],
            prior: None,
            out: layout.out_at,
            stride: 0,
            resident: None,
            mop: [None; 3],
            init: layout.init.clone(),
            roles,
            loops,
        })
    }

    fn chunk(&self, prior: bool, count: usize, last: bool, l1: &Placed) -> Result<RunCode> {
        let c = &ChunkLayout::placed(1, l1);
        let valid = if last { self.valid } else { 32 };
        let (roles, loops) = continuation_roles(c, self.op, self.axis, prior, count, valid, last);
        Ok(RunCode {
            operands: [Some(c.layout.in_at), None, None],
            prior: Some(c.prior_at),
            out: c.layout.out_at,
            stride: 0,
            resident: None,
            mop: [None; 3],
            init: c.layout.init.clone(),
            roles: Arc::new(roles),
            loops: Arc::new(loops),
        })
    }
}
