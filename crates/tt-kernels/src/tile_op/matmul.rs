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

use std::ops::Range;

use super::{
    Access, Blocks, Chunk, Cols, Combine, Dim, Iter, Layout, Microkernel, NumericContract,
    ReductionContract, RunCode, TileMap, TileOp,
};
use crate::matmul::{self, BlockStreamLayout, Fidelity, LiveRows, SrcRoute, Staging, StreamLayout};
use crate::tensor::{self, Elem, Result, TensorError};

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
    /// `K` whole in `Dst` ([`Combine::InDst`]): `[mc, nc]` blocks, or one tile row's columns in
    /// the runs `cols` deals (`tensor::thin_runs`), staged in alternating halves under
    /// [`Staging::SlotsHalf`]; `A` held across a unit's blocks where `resident_a` (or packed).
    Blocks {
        block: [usize; 2],
        cols: Option<Vec<Range<usize>>>,
        staging: Staging,
        resident_a: bool,
        allow_mop: bool,
    },
    /// One output tile a job, `K` in blocks of `k_block`, each block's partial through the
    /// result in GDDR ([`Combine::GddrPartials`]).
    KBlocks { k_block: usize },
    /// One output column a job (a tile row, `A` packed), in the order of `cols` or every
    /// column, `K` in chunks of `kq` continued in L1 halves; the packed `A` read once a unit.
    Stream {
        kq: usize,
        layout: StreamLayout,
        cols: Option<Vec<Range<usize>>>,
    },
}

/// `op(A) @ op(B)` of `[m, k] @ [k, n]` (after the transposes) as a [`TileOp`] under `schedule`;
/// `unpack_b`: a transposed `B` is transposed by the unpacker, read as it lies; `packed`, a
/// one-row `A` also packed `[K / 32, 32]` (`tensor::pack_row`), read from there.
#[allow(clippy::too_many_arguments)]
pub fn matmul(
    a: Side,
    b: Side,
    [m, k, n]: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    unpack_b: bool,
    packed: Option<TensorRef>,
    schedule: Schedule,
) -> Result<TileOp> {
    let combine = match &schedule {
        Schedule::BlockStream { layout } => Combine::L1Halves {
            chunk: layout.tiles[1],
        },
        Schedule::Blocks { .. } => Combine::InDst,
        Schedule::KBlocks { k_block } => Combine::GddrPartials { chunk: *k_block },
        Schedule::Stream { kq, .. } => Combine::L1Halves { chunk: *kq },
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
    let access = |s: Side, map: TileMap, layout: Layout| Access {
        layout,
        map,
        elem: s.elem,
        live_rows: s.live_rows,
    };
    let a_layout = match packed {
        Some(row) => Layout::Packed { tiles: a.at, row },
        None => Layout::Tiles(a.at),
    };
    Ok(TileOp {
        domain: vec![
            Iter::parallel(m),
            Iter::parallel(n),
            Iter::reduction(k, contract),
        ],
        operands: vec![
            access(a, map(a.transposed, 0, 2), a_layout),
            access(b, map(b.transposed, 2, 1), Layout::Tiles(b.at)),
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
            packed: packed.is_some(),
            kt: k.div_ceil(32),
            schedule,
        }),
        contract: NumericContract::Program,
        out: ([m, n], Elem::F32),
        ahead: Vec::new(),
    })
}

/// The Matrix Unit's programs over a run of output tiles.
struct Mma {
    route: SrcRoute,
    fidelity: Fidelity,
    unpack_b: bool,
    /// `A` read from its packed row (`matmul::packed_a_pairs`).
    packed: bool,
    kt: usize,
    schedule: Schedule,
}

/// The columns `cols` deals, or every `every`.
fn cols_of(cols: &Option<Vec<Range<usize>>>, every: usize) -> Cols<'_> {
    match cols {
        Some(runs) => Cols::Runs(runs),
        None => Cols::Every(every),
    }
}

impl Microkernel for Mma {
    fn lists(&self, _chunked: bool) -> [&'static str; 2] {
        match self.schedule {
            Schedule::BlockStream { .. } => ["blocked matmul gather", "blocked matmul scatter"],
            Schedule::Blocks { .. } => ["matmul gather", "matmul scatter"],
            Schedule::KBlocks { .. } => ["K-block matmul gather", "K-block matmul scatter"],
            Schedule::Stream { .. } => ["streamed matmul gather", "streamed matmul scatter"],
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

    fn block(&self) -> Option<Blocks<'_>> {
        Some(match &self.schedule {
            Schedule::BlockStream { layout } => Blocks {
                rows: layout.tiles[0],
                cols: Cols::Every(layout.tiles[2]),
                pipelined: false,
            },
            Schedule::Blocks {
                block,
                cols,
                staging,
                ..
            } => Blocks {
                rows: block[0],
                cols: cols_of(cols, block[1]),
                pipelined: *staging == Staging::SlotsHalf,
            },
            Schedule::KBlocks { .. } => Blocks {
                rows: 1,
                cols: Cols::Every(1),
                pipelined: false,
            },
            Schedule::Stream { cols, .. } => Blocks {
                rows: 1,
                cols: cols_of(cols, 1),
                pipelined: false,
            },
        })
    }

    fn block_chunk(&self, chunk: &Chunk) -> Result<RunCode> {
        let &Chunk {
            c,
            k0,
            tiles,
            live,
            half,
        } = chunk;
        let [rows, count, cols] = tiles;
        let live = LiveRows::for_last(rows, live);
        let (route, fidelity, unpack_b) = (self.route, self.fidelity, self.unpack_b);
        let code = |a_at, b_at, out, init, roles, loops, prior, resident, mop| RunCode {
            operands: [Some(a_at), Some(b_at), None],
            prior,
            out,
            stride: TILE_SLOT,
            resident,
            init,
            roles,
            loops,
            mop,
        };
        match &self.schedule {
            Schedule::BlockStream { layout } => {
                let (chunk, _) = layout.chunk(c, tiles);
                let (roles, loops) = matmul::block_stream_programs(
                    layout, c, tiles, route, fidelity, unpack_b, live,
                );
                Ok(code(
                    chunk.a_at,
                    chunk.b_at,
                    layout.out_slot(c % 2, cols, 0, 0) + TILE_DATA,
                    layout.init.clone(),
                    roles,
                    loops,
                    None,
                    None,
                    [None; 3],
                ))
            }
            Schedule::Blocks {
                staging, allow_mop, ..
            } => {
                let in_fmt = route.formats().0;
                // The second half's layout is the first's moved up, and noted there.
                let up = if half == Some(1) { matmul::half() } else { 0 };
                let planned = crate::l1::planned_shifted(up, || match self.packed {
                    true => matmul::plan_layout_packed_a(
                        tiles,
                        in_fmt,
                        *staging,
                        tensor::packed_a_slots(self.kt),
                        route.b_slot(),
                    ),
                    false => matmul::plan_layout_route(tiles, in_fmt, *staging, route.slots()),
                })?;
                let matmul::Layout {
                    a_at,
                    b_at,
                    outputs,
                    sems,
                    init,
                } = match half {
                    Some(1) => planned.shifted(matmul::half()),
                    _ => planned,
                };
                let outputs = match self.packed {
                    true => matmul::packed_a_pairs(&outputs, a_at, 0),
                    false => outputs,
                };
                // The cache tells the stagings and halves apart: their programs name different
                // addresses.
                let variant =
                    half.map_or(0, |h| 1 + u32::from(h)) | if self.packed { 0x40 } else { 0 };
                let (roles, mop) = matmul::kernel_programs(
                    tiles,
                    variant,
                    route,
                    fidelity,
                    sems,
                    *allow_mop,
                    unpack_b,
                    live,
                    || {
                        matmul::matmul_kernel_with(
                            &outputs,
                            sems,
                            route.src_formats(),
                            fidelity,
                            *allow_mop,
                            unpack_b,
                            live,
                        )
                    },
                );
                // Only the datums go back: the packer writes nothing else, and the unpacker skips
                // the header whatever it holds (`step18_dram_matmul`). The outputs sit one slot
                // apart from the first (`plan_layout_in`).
                let out = outputs[0].out;
                debug_assert!(outputs
                    .iter()
                    .enumerate()
                    .all(|(k, o)| o.out == out + k as u64 * TILE_SLOT));
                Ok(code(
                    a_at,
                    b_at,
                    out,
                    init,
                    roles,
                    Default::default(),
                    None,
                    None,
                    mop,
                ))
            }
            Schedule::KBlocks { .. } => {
                let a_slots = match self.packed {
                    true => tensor::packed_a_slots(count),
                    false => count,
                };
                let (layout, reload) = matmul::k_block_layout_with(count, a_slots, route.slots())?;
                let layout = match self.packed {
                    true => matmul::Layout {
                        outputs: matmul::packed_a_pairs(&layout.outputs, layout.a_at, k0),
                        ..layout
                    },
                    false => layout,
                };
                let variant = if c > 0 { 0x81 } else { 0x80 } | if self.packed { 0x40 } else { 0 };
                let (roles, _) = matmul::kernel_programs(
                    [1, count, 1],
                    variant,
                    route,
                    fidelity,
                    layout.sems,
                    false,
                    unpack_b,
                    live,
                    || {
                        (
                            matmul::k_block_kernel(
                                &layout,
                                (c > 0).then_some(reload),
                                route,
                                fidelity,
                                unpack_b,
                                live,
                            ),
                            [None; 3],
                        )
                    },
                );
                Ok(code(
                    layout.a_at,
                    layout.b_at,
                    layout.outputs[0].out,
                    layout.init,
                    roles,
                    Default::default(),
                    Some(reload.prior),
                    None,
                    [None; 3],
                ))
            }
            Schedule::Stream { layout, .. } => {
                let (chunk, reload) = layout.chunk(c, k0, count);
                // The programs name this chunk's `K` range in the packed `A`, its half and the
                // layout `A`'s slots fix.
                let variant = 0x1_0000 | (tensor::packed_a_slots(self.kt) as u32) << 8 | c as u32;
                let (roles, _) = matmul::kernel_programs(
                    [1, count, 1],
                    variant,
                    route,
                    fidelity,
                    chunk.sems,
                    false,
                    unpack_b,
                    live,
                    || {
                        (
                            matmul::k_block_kernel(&chunk, reload, route, fidelity, unpack_b, live),
                            [None; 3],
                        )
                    },
                );
                Ok(code(
                    chunk.a_at,
                    chunk.b_at,
                    chunk.outputs[0].out,
                    chunk.init.clone(),
                    roles,
                    Default::default(),
                    None,
                    // The packed `A` holds every term, read once a unit.
                    Some((0..self.kt, layout.a_at)),
                    [None; 3],
                ))
            }
        }
    }

    fn transposes(&self, o: usize) -> bool {
        o == 1 && self.unpack_b
    }

    fn holds(&self, o: usize) -> bool {
        o == 0
            && match &self.schedule {
                Schedule::Blocks { resident_a, .. } => *resident_a,
                Schedule::Stream { .. } => true,
                Schedule::BlockStream { .. } | Schedule::KBlocks { .. } => false,
            }
    }
}
