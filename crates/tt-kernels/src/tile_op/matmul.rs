//! A product `op(A) @ op(B)` as a [`TileOp`]: the domain `[m, n, k]`, `m` and `n` parallel and `k`
//! a reduction whose terms are added in ascending order, `A` mapped `[m, k]` and `B` `[k, n]`
//! (either stored transposed), the result `[m, n]`. The body is the Matrix Unit's (`Mma`): every
//! output tile's `K` chain through FP32 partials in `Dst`, at the route's `Src` formats and the
//! fidelity's phases.
//!
//! How the product is cut is its [`Schedule`], chosen before the description is built (the cost
//! models in `tensor`), and always one of the bit-equivalent options under the default contract.

use tt_isa::dm::record::TensorRef;
use tt_isa::dm::{TILE_DATA, TILE_SLOT};

use super::{
    Access, Combine, Dim, Iter, Layout, Microkernel, NumericContract, ReductionContract, RunCode,
    TileMap, TileOp,
};
use crate::matmul::{self, BlockStreamLayout, Fidelity, LiveRows, SrcRoute};
use crate::tensor::{Elem, Result, TensorError};

/// One operand of a product: its tiles where they lie (a block of a parent where the reference
/// says), whether it is stored transposed, and its edge's element type.
#[derive(Copy, Clone, Debug)]
pub struct Side {
    pub at: TensorRef,
    pub transposed: bool,
    pub elem: Elem,
    /// A ragged last tile row read to its live rows ([`Access::live_rows`]).
    pub live_rows: bool,
}

/// How a product's output tiles and terms are cut into runs.
pub enum Schedule {
    /// `[mb, nb]` blocks of output tiles, a job each, `K` in chunks of `kq` continued in L1
    /// halves ([`Combine::L1Halves`]; `docs/plans/prefill-operand-reuse.md`).
    BlockStream { layout: BlockStreamLayout },
}

/// `op(A) @ op(B)` of `[m, k] @ [k, n]` (after the transposes) as a [`TileOp`] under `schedule`;
/// `unpack_b`: a transposed `B` is transposed by the unpacker, read as it lies.
pub fn matmul(
    a: Side,
    b: Side,
    [m, k, n]: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    unpack_b: bool,
    schedule: Schedule,
) -> Result<TileOp> {
    let combine = match &schedule {
        Schedule::BlockStream { layout } => Combine::L1Halves {
            chunk: layout.tiles[1],
        },
    };
    let contract = ReductionContract { combine };
    let map = |transposed: bool, rows: usize, cols: usize| match transposed {
        false => TileMap {
            rows: Dim::of(rows),
            cols: Dim::of(cols),
        },
        true => TileMap {
            rows: Dim::of(cols),
            cols: Dim::of(rows),
        },
    };
    let access = |s: Side, map: TileMap| Access {
        layout: Layout::Tiles(s.at),
        map,
        elem: s.elem,
        live_rows: s.live_rows,
    };
    Ok(TileOp {
        domain: vec![
            Iter::parallel(m),
            Iter::parallel(n),
            Iter::reduction(k, contract),
        ],
        operands: vec![
            access(a, map(a.transposed, 0, 2)),
            access(b, map(b.transposed, 2, 1)),
        ],
        results: vec![Access {
            // Placed when the op is built (`TileOp::build`), or by the caller of `TileOp::jobs`.
            layout: Layout::Tiles(Default::default()),
            map: map(false, 0, 1),
            elem: Elem::F32,
            live_rows: false,
        }],
        body: Box::new(Mma {
            route,
            fidelity,
            unpack_b: unpack_b && b.transposed,
            schedule,
        }),
        contract: NumericContract::Program,
        out: ([m, n], Elem::F32),
    })
}

/// The Matrix Unit's programs over a run of output tiles.
struct Mma {
    route: SrcRoute,
    fidelity: Fidelity,
    unpack_b: bool,
    schedule: Schedule,
}

impl Microkernel for Mma {
    fn lists(&self, _chunked: bool) -> [&'static str; 2] {
        match self.schedule {
            Schedule::BlockStream { .. } => ["blocked matmul gather", "blocked matmul scatter"],
        }
    }

    fn group(&self, _half: bool) -> Option<usize> {
        None
    }

    fn run(&self, _len: usize, _half: Option<u8>, _held: &[bool]) -> Result<RunCode> {
        Err(TensorError::Shape(
            "a product runs in blocks of output tiles".into(),
        ))
    }

    fn block(&self) -> Option<[usize; 2]> {
        match &self.schedule {
            Schedule::BlockStream { layout } => Some([layout.tiles[0], layout.tiles[2]]),
        }
    }

    fn block_chunk(&self, c: usize, tiles: [usize; 3], live: u32) -> Result<RunCode> {
        let Schedule::BlockStream { layout } = &self.schedule;
        let [rows, _, cols] = tiles;
        let (chunk, _) = layout.chunk(c, tiles);
        let (roles, loops) = matmul::block_stream_programs(
            layout,
            c,
            tiles,
            self.route,
            self.fidelity,
            self.unpack_b,
            LiveRows::for_last(rows, live),
        );
        Ok(RunCode {
            operands: [Some(chunk.a_at), Some(chunk.b_at), None],
            prior: None,
            out: layout.out_slot(c % 2, cols, 0, 0) + TILE_DATA,
            stride: TILE_SLOT,
            init: layout.init.clone(),
            roles,
            loops,
        })
    }

    fn transposes(&self, o: usize) -> bool {
        o == 1 && self.unpack_b
    }
}
