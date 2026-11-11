//! The pointwise SFPU family as a [`TileOp`]: every kind of `sfpu::ops` (`Eltwise`), unary,
//! binary with any broadcast, and ternary, over blocks of resident tensors.

use std::sync::{Arc, OnceLock};

use super::{
    allocate, Access, At, Dim, Iter, Layout, Microkernel, NumericContract, RunCode, TileMap, TileOp,
};
use crate::l1::{Placed, Requirements};
use crate::sfpu::kernel::Operands;
use crate::sfpu::ops::Broadcast;
use crate::tensor::{self, BlockOperand, DramTensor, Eltwise, Result, TensorError};

/// `a (op) b` -- or `a (op) scalar`, or a ternary op's `c` -- over blocks of resident tensors
/// ([`BlockOperand`]: each operand's tiles read where they lie, at its parent's row stride), the
/// result a new tensor of `a`'s block's dims; with `twin`, a one-row result's data rows also
/// written packed (`tensor::packed_twin_writes`). `None` when the op has no SFPU program.
/// Integer division is refused here: it reads whole tensors (`tensor::sfpu_eltwise`).
pub fn eltwise(
    op: Eltwise,
    a: BlockOperand<'_>,
    b: Option<BlockOperand<'_>>,
    c: Option<BlockOperand<'_>>,
    twin: Option<&DramTensor>,
) -> Result<Option<TileOp>> {
    let what = "an element-wise op's block";
    let (ra, rb, rc) = (
        a.tensor_ref(what)?,
        b.map(|b| b.tensor_ref(what)).transpose()?,
        c.map(|c| c.tensor_ref(what)).transpose()?,
    );
    let (a, b, c) = (
        a.shape_only(),
        b.map(|b| b.shape_only()),
        c.map(|c| c.shape_only()),
    );
    let (a, b, c) = (&a, b.as_ref(), c.as_ref());
    let (kind, bcast) = tensor::broadcast_of(op, a, b)?;
    let formats = tensor::eltwise_formats(
        kind,
        &[Some(a), b, c].into_iter().flatten().collect::<Vec<_>>(),
    )?;
    let op = Eltwise { kind, ..op };
    if crate::sfpu::integer::operation(kind).is_some_and(|(op, _)| op == 15 || op == 16) {
        return Err(TensorError::Shape(
            "integer division reads whole tensors, not blocks".into(),
        ));
    }
    let Some((operands, _)) = crate::sfpu::ops::program_for(kind, [op.scalar, op.scalar2], bcast)
    else {
        return Ok(None);
    };
    if formats.bf16_in && crate::sfpu::kernel::widened_rows(operands).is_none() {
        return Err(TensorError::Shape(format!(
            "element-wise {kind:#x} on BF16 operands with {bcast:?}: no BF16 form (widen them)"
        )));
    }
    // A ternary op's third operand: `A`'s shape, of the kind's type.
    let rc = match (operands, c) {
        (Operands::Ternary, Some(c)) => {
            if (c.rows, c.cols) != (a.rows, a.cols) {
                return Err(TensorError::Shape(format!(
                    "a ternary op's third operand [{}, {}] for [{}, {}]",
                    c.rows, c.cols, a.rows, a.cols
                )));
            }
            // Its type is checked with the others' (`eltwise_formats`).
            rc
        }
        (Operands::Ternary, None) => {
            return Err(TensorError::Shape(
                "a ternary op needs three operands".into(),
            ))
        }
        (_, Some(_)) => {
            return Err(TensorError::Shape(format!(
                "{kind:#x} takes no third operand"
            )))
        }
        _ => None,
    };
    let ct = a.grid()[1];
    let out_elem = tensor::result_elem(op.kind, formats);
    if let Some(twin) = twin.filter(|t| t.elem != out_elem) {
        return Err(TensorError::Shape(format!(
            "a packed {:?} twin for a {out_elem:?} result",
            twin.elem
        )));
    }
    if let Some(twin) = twin {
        if (a.rows, twin.rows, twin.cols) != (1, ct, 32) {
            return Err(TensorError::Shape(format!(
                "a packed twin [{}, {}] for a [{}, {}] result",
                twin.rows, twin.cols, a.rows, a.cols
            )));
        }
    }
    let same = TileMap {
        rows: Dim::of(0),
        cols: Dim::of(1),
    };
    let mut accesses = vec![Access {
        layout: Layout::Tiles(ra),
        map: same,
        elem: a.elem,
        live_rows: false,
    }];
    if let (Some(rb), Some(b)) = (rb, b) {
        let map = match bcast {
            Broadcast::None => same,
            Broadcast::Row => TileMap {
                rows: Dim::Fixed,
                cols: Dim::of(1),
            },
            Broadcast::Col => TileMap {
                rows: Dim::of(0),
                cols: Dim::Fixed,
            },
            Broadcast::Block => TileMap {
                rows: Dim::Iter {
                    iter: 0,
                    wrap: Some(b.grid()[0]),
                },
                cols: Dim::of(1),
            },
        };
        accesses.push(Access {
            layout: Layout::Tiles(rb),
            map,
            elem: b.elem,
            live_rows: false,
        });
    }
    if let (Some(rc), Some(c)) = (rc, c) {
        accesses.push(Access {
            layout: Layout::Tiles(rc),
            map: same,
            elem: c.elem,
            live_rows: false,
        });
    }
    let mut results = vec![Access {
        // Placed when the op is built (`TileOp::build`).
        layout: Layout::Tiles(Default::default()),
        map: same,
        elem: out_elem,
        live_rows: false,
    }];
    if let Some(twin) = twin {
        results.push(Access {
            layout: Layout::PackedRow(twin.clone()),
            map: same,
            elem: out_elem,
            live_rows: false,
        });
    }
    Ok(Some(TileOp {
        domain: vec![Iter::parallel(a.rows), Iter::parallel(a.cols)],
        operands: accesses,
        results,
        body: Box::new(Sfpu {
            op,
            bcast,
            operands,
            rows: tensor::eltwise_rows(op.kind, a.rows),
            formats,
        }),
        contract: NumericContract::Program,
        ahead: Vec::new(),
        share: None,
        out: ([a.rows, a.cols], out_elem),
    }))
}

/// One SFPU program over a run of tiles (`sfpu::kernel`): the mover stages each operand's tiles,
/// the roles unpack them to `Dst`, run the kind's program and pack the results.
struct Sfpu {
    op: Eltwise,
    bcast: Broadcast,
    operands: Operands,
    /// The rows of each tile the program computes (`tensor::eltwise_rows`).
    rows: u32,
    /// FP32, or BF16 operands and result (`sfpu::kernel::Formats`; roadmap W1).
    formats: crate::sfpu::kernel::Formats,
}

impl Microkernel for Sfpu {
    fn lists(&self, _chunked: bool) -> [&'static str; 2] {
        ["sfpu gather", "sfpu scatter"]
    }

    fn group(&self, half: bool) -> Option<usize> {
        Some(tensor::sfpu_group(
            self.op,
            self.bcast,
            self.operands,
            half,
            self.formats,
        ))
    }

    fn needs(&self, at: &At<'_>) -> Result<Arc<Requirements>> {
        static MEMO: allocate::Declared<(usize, Operands)> = OnceLock::new();
        let &At::Run { len, .. } = at else {
            return Err(at.refused());
        };
        let operands = self.operands;
        Ok(allocate::declared(&MEMO, (len, operands), || {
            crate::sfpu::kernel::requirements(len, operands)
        }))
    }

    fn code(&self, at: &At<'_>, l1: &Placed) -> Result<RunCode> {
        let &At::Run { len, half, held } = at else {
            return Err(at.refused());
        };
        // Only `B` is ever one tile a row (a column's broadcast).
        let b_once = held.get(1).copied().unwrap_or(false);
        let (layout, roles, loops) = tensor::sfpu_programs(
            self.op,
            self.bcast,
            self.operands,
            len,
            half,
            self.rows,
            b_once,
            self.formats,
            l1,
        )?;
        Ok(RunCode {
            operands: [Some(layout.a_at), layout.b_at, layout.c_at],
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
}
