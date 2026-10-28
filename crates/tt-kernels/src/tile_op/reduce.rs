//! A reduction over one axis as a [`TileOp`] (`sfpu::reduce`): `[rows, cols]` to `[rows, 1]` over
//! the columns, or to `[1, cols]` over the rows, with a finishing kind applied before the pack
//! ([`ReduceThen`]).

use super::{
    Access, Combine, Dim, Iter, Layout, Microkernel, NumericContract, ReductionContract, RunCode,
    TileMap, TileOp,
};
use crate::sfpu::reduce::{continuation_roles, plan_chunk_layout, Axis, ChunkLayout, ReduceOp};
use crate::tensor::{self, DramTensor, ReduceThen, Result, TensorError};
use std::sync::Arc;

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
    let (combine, chunk) = if fits {
        (Combine::InDst, None)
    } else {
        if then.is_some() {
            return Err(TensorError::Shape(
                "a reduction too long for one tile's programs takes no finishing kind".into(),
            ));
        }
        let c = plan_chunk_layout(1).map_err(|e| TensorError::Shape(e.to_string()))?;
        (
            Combine::GddrPartials {
                chunk: crate::sfpu::reduce::ROW_CHUNK,
            },
            Some(c),
        )
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
        }],
        results: vec![Access {
            // Placed when the op is built (`TileOp::build`).
            layout: Layout::Tiles(Default::default()),
            map: out_map,
            elem: op.elem(),
        }],
        body: Box::new(Reduce {
            op,
            axis,
            per,
            valid,
            then,
            chunk,
        }),
        contract: NumericContract::Program,
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
    /// A chunked reduction's layout, planned once.
    chunk: Option<ChunkLayout>,
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

    fn run(&self, len: usize, half: Option<u8>, _held: &[bool]) -> Result<RunCode> {
        let (layout, roles, loops) = tensor::reduce_programs(
            self.op, self.axis, len, self.per, self.valid, half, self.then,
        )?;
        Ok(RunCode {
            operands: [Some(layout.in_at), None, None],
            prior: None,
            out: layout.out_at,
            init: layout.init.clone(),
            roles,
            loops,
        })
    }

    fn chunk(&self, prior: bool, count: usize, last: bool) -> Result<RunCode> {
        let c = self.chunk.as_ref().ok_or_else(|| {
            TensorError::Shape("a reduction that fits one run has no chunks".into())
        })?;
        let valid = if last { self.valid } else { 32 };
        let (roles, loops) = continuation_roles(c, self.op, self.axis, prior, count, valid, last);
        Ok(RunCode {
            operands: [Some(c.layout.in_at), None, None],
            prior: Some(c.prior_at),
            out: c.layout.out_at,
            init: c.layout.init.clone(),
            roles: Arc::new(roles),
            loops: Arc::new(loops),
        })
    }
}
