//! The pointwise SFPU family as a [`TileOp`]: every kind of `sfpu::ops` (`Eltwise`), unary,
//! binary with any broadcast, and ternary, over blocks of resident tensors.

use super::{Access, Dim, Iter, Layout, Microkernel, NumericContract, RunCode, TileMap, TileOp};
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
    // A ternary op's third operand: `A`'s shape, of the kind's type.
    let rc = match (operands, c) {
        (Operands::Ternary, Some(c)) => {
            if (c.rows, c.cols) != (a.rows, a.cols) {
                return Err(TensorError::Shape(format!(
                    "a ternary op's third operand [{}, {}] for [{}, {}]",
                    c.rows, c.cols, a.rows, a.cols
                )));
            }
            if let Some(&want) = crate::sfpu::ops::elems(kind).inputs.get(2) {
                c.expect(&format!("element-wise op {kind:#x}"), want)?;
            }
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
        });
    }
    if let (Some(rc), Some(c)) = (rc, c) {
        accesses.push(Access {
            layout: Layout::Tiles(rc),
            map: same,
            elem: c.elem,
        });
    }
    let out_elem = tensor::output_elem(op.kind);
    let mut results = vec![Access {
        // Placed when the op is built (`TileOp::build`).
        layout: Layout::Tiles(Default::default()),
        map: same,
        elem: out_elem,
    }];
    if let Some(twin) = twin {
        results.push(Access {
            layout: Layout::PackedRow(twin.clone()),
            map: same,
            elem: out_elem,
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
        }),
        contract: NumericContract::Program,
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
}

impl Microkernel for Sfpu {
    fn lists(&self, _chunked: bool) -> [&'static str; 2] {
        ["sfpu gather", "sfpu scatter"]
    }

    fn group(&self, half: bool) -> Option<usize> {
        Some(tensor::sfpu_group(self.op, self.bcast, self.operands, half))
    }

    fn run(&self, len: usize, half: Option<u8>, held: &[bool]) -> Result<RunCode> {
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
        )?;
        Ok(RunCode {
            operands: [Some(layout.a_at), layout.b_at, layout.c_at],
            prior: None,
            out: layout.out_at,
            init: layout.init.clone(),
            roles,
            loops,
        })
    }
}
