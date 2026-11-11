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
use tt_isa::format::TileFormat;

use std::ops::Range;
use std::sync::{Arc, OnceLock};

use super::{
    allocate, Access, At, Blocks, Chunk, Cols, Combine, Dim, Iter, Layout, Microkernel,
    NumericContract, ReductionContract, RunCode, TileMap, TileOp,
};
use crate::l1::{Placed, Requirements};
use crate::matmul::{
    self, BlockStreamLayout, Fidelity, LiveRows, Slots, SrcRoute, Staging, StreamLayout,
};
use crate::tensor::{
    self, stream_chunks, thin_runs, Elem, Job, Overlap, Result, Step, TensorError, PREFETCH,
};

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
    /// `[mb, nb]` blocks of output tiles (`tiles`, `[mb, kq, nb]`), a job each, `K` in chunks of
    /// `kq` continued in L1 halves ([`Combine::L1Halves`]; `docs/plans/prefill-operand-reuse.md`).
    BlockStream { tiles: [usize; 3] },
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
        Schedule::BlockStream { tiles } => Combine::L1Halves { chunk: tiles[1] },
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
    if matches!(
        schedule,
        Schedule::Blocks {
            staging: Staging::Host,
            ..
        }
    ) {
        return Err(TensorError::Shape(
            "a product on TileOp reads its tiles into slots, not as the host stages them".into(),
        ));
    }
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
            b_by_column: packed.is_some() && !b.transposed && b.at.col_tiles != 0,
            kt: k.div_ceil(32),
            schedule,
        }),
        contract: NumericContract::Program,
        out: ([m, n], Elem::F32),
        ahead: Vec::new(),
        share: None,
    })
}

/// The Matrix Unit's programs over a run of output tiles.
struct Mma {
    route: SrcRoute,
    fidelity: Fidelity,
    unpack_b: bool,
    /// `A` read from its packed row (`matmul::packed_a_pairs`).
    packed: bool,
    /// A one-row product's `B` laid by column (`tensor::Layout::ByColumn`), taken into L1 by
    /// column: tile `(kk, j)` at `b_at + (j * kt + kk) * slot`, each column one run read in large
    /// pieces (`tt_isa::dm::record::REF_L1_BY_COLUMN`, which the gathers carry where this does).
    /// The pairs name the same tiles in the same order, so the sum is the same.
    b_by_column: bool,
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

/// What a run of [`Mma`] needs, by shape: the requirements are built once a shape.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Shape {
    BlockStream([usize; 3], Slots),
    Slots([usize; 3], usize, bool, Slots),
    KBlock(usize, usize, Slots),
    Stream(usize, usize, u64),
}

impl Mma {
    /// The slot staging's `A` slots, slot sizes and whether `B` lies against the arena's end, for
    /// a block of `tiles`: a packed `A`'s ring holds the packed rows its `K` spans
    /// (`matmul::plan_layout_packed_a`).
    fn slots(&self, [mt, kt, _]: [usize; 3], staging: Staging) -> (usize, Slots, bool) {
        match self.packed {
            true => (
                tensor::packed_a_slots(self.kt),
                self.route.packed_row_slots(),
                staging == Staging::Slots,
            ),
            false => (mt * kt, self.route.slots(), false),
        }
    }

    /// A K block's `A` slots: a packed `A`'s, or one a term.
    fn k_block_slots(&self, count: usize) -> usize {
        match self.packed {
            true => tensor::packed_a_slots(count),
            false => count,
        }
    }

    /// A K block's slot sizes: a packed `A`'s ring in [`TILE_SLOT`]s whatever its format
    /// ([`SrcRoute::packed_row_slots`]), else each operand's own.
    fn k_block_slot_sizes(&self) -> Slots {
        match self.packed {
            true => self.route.packed_row_slots(),
            false => self.route.slots(),
        }
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

    fn needs(&self, at: &At<'_>) -> Result<Arc<Requirements>> {
        static MEMO: allocate::Declared<Shape> = OnceLock::new();
        let At::Block(chunk) = at else {
            return Err(at.refused());
        };
        let shape = match &self.schedule {
            Schedule::BlockStream { tiles } => Shape::BlockStream(*tiles, self.route.slots()),
            Schedule::Blocks { staging, .. } => {
                let (a_slots, slots, top_b) = self.slots(chunk.tiles, *staging);
                Shape::Slots(chunk.tiles, a_slots, top_b, slots)
            }
            Schedule::KBlocks { .. } => {
                let count = chunk.tiles[1];
                Shape::KBlock(count, self.k_block_slots(count), self.k_block_slot_sizes())
            }
            Schedule::Stream { kq, .. } => {
                Shape::Stream(*kq, tensor::packed_a_slots(self.kt), self.route.b_slot())
            }
        };
        Ok(allocate::declared(&MEMO, shape, || match shape {
            Shape::BlockStream(tiles, slots) => matmul::block_stream_requirements(tiles, slots),
            Shape::Slots(tiles, a_slots, top_b, slots) => {
                matmul::matmul_requirements_placed(tiles, a_slots, top_b, slots).req
            }
            Shape::KBlock(count, a_slots, slots) => {
                matmul::k_block_requirements(count, a_slots, slots)
            }
            Shape::Stream(kq, a_slots, b_slot) => {
                matmul::k_stream_requirements(kq, a_slots, b_slot)
            }
        }))
    }

    fn code(&self, at: &At<'_>, l1: &Placed) -> Result<RunCode> {
        let At::Block(chunk) = at else {
            return Err(at.refused());
        };
        let &Chunk {
            c,
            k0,
            tiles,
            live,
            half,
        } = *chunk;
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
            Schedule::BlockStream { tiles: shape } => {
                let layout = BlockStreamLayout::placed(*shape, route.slots(), l1);
                let (chunk, _) = layout.chunk(c, tiles);
                // The `Dst` pass: the block row's subblocks, half-sync (`tile_op::dst`).
                let dst = matmul::dst_row_plan(cols);
                let (roles, loops, held) = matmul::block_stream_programs(
                    &layout,
                    c,
                    tiles,
                    route,
                    fidelity,
                    unpack_b,
                    live,
                    dst.as_ref(),
                );
                let init = match held {
                    Some(plan) => matmul::subblock_init(&layout.init, layout.sems, plan),
                    None => layout.init.clone(),
                };
                Ok(code(
                    chunk.a_at,
                    chunk.b_at,
                    layout.out_slot(c % 2, cols, 0, 0) + TILE_DATA,
                    init,
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
                assert!(
                    TileFormat::new(in_fmt).slot_bytes() <= TILE_SLOT,
                    "slot staging must fit the operand image"
                );
                let (_, slots, _) = self.slots(tiles, *staging);
                let matmul::Layout {
                    a_at,
                    b_at,
                    mut outputs,
                    sems,
                    init,
                } = matmul::slots_layout(tiles, slots, l1);
                if self.b_by_column {
                    let [_, kt, nt] = tiles;
                    for (o, out) in outputs.iter_mut().enumerate() {
                        let j = o % nt;
                        for (kk, pair) in out.pairs.iter_mut().enumerate() {
                            pair.1 = b_at + (j * kt + kk) as u64 * slots.b;
                        }
                    }
                }
                let outputs = match self.packed {
                    true => matmul::packed_a_pairs(&outputs, a_at, 0, route.packed_row_bytes()),
                    false => outputs,
                };
                // The cache tells the stagings and halves apart: their programs name different
                // addresses.
                let variant = half.map_or(0, |h| 1 + u32::from(h))
                    | if self.packed { 0x40 } else { 0 }
                    | if self.b_by_column { 0x100 } else { 0 };
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
                // apart from the first (`matmul::slots_layout`).
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
                let (layout, reload) = matmul::k_block_placed(count, self.k_block_slot_sizes(), l1);
                let layout = match self.packed {
                    true => matmul::Layout {
                        outputs: matmul::packed_a_pairs(
                            &layout.outputs,
                            layout.a_at,
                            k0,
                            route.packed_row_bytes(),
                        ),
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
            Schedule::Stream { .. } => {
                let layout = StreamLayout::placed(route.b_slot(), l1);
                let (chunk, reload) = layout.chunk(c, k0, count, route.packed_row_bytes());
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

    fn block(&self) -> Option<Blocks<'_>> {
        Some(match &self.schedule {
            Schedule::BlockStream { tiles } => Blocks {
                rows: tiles[0],
                cols: Cols::Every(tiles[2]),
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

/// How one `[m, k, n]` product is cut into jobs: what [`matmul_dram`] and
/// [`matmul_dram_batched`] share.
#[derive(Clone)]
pub(crate) struct Plan {
    pub(crate) tiles: [usize; 3],
    /// The product's rows: a job of its last tile row alone computes only those it holds
    /// ([`matmul::LiveRows`]).
    pub(crate) m: usize,
    pub(crate) block: [usize; 2],
    pub(crate) k_block: usize,
    pub(crate) staging: Staging,
    pub(crate) units: usize,
    pub(crate) route: SrcRoute,
    pub(crate) fidelity: Fidelity,
    pub(crate) allow_mop: bool,
    /// [`thin_rows`] of `op(A)`, set on the gathers of its last tile row.
    pub(crate) thin: u32,
    /// Mark a block [`record::GATHER_A_RESIDENT`] when its unit's previous block of the same
    /// `push_jobs` read the same `A` rows to the same slots. Only under [`Staging::Slots`]:
    /// [`Staging::SlotsHalf`] moves `A` between halves.
    pub(crate) resident_a: bool,
    /// The output's columns, one run a job in dealing order, when [`thin_runs`]
    /// deals them (instead of `block`'s width).
    pub(crate) col_runs: Option<Vec<std::ops::Range<usize>>>,
    /// A transposed `B`'s tiles are read as they lie and transposed by the kernel's unpacker
    /// ([`record::GATHER_B_UNPACK_TRANSPOSE`]), not one datum at a time by the mover
    /// (`READ_TRANSPOSED`, ~10 us a tile). Only on the TF32 route, where `B`'s unpack to `SrcA`
    /// converts it anyway.
    pub(crate) unpack_b: bool,
    /// A one-row `A` packed `[K / 32, 32]` ([`pack_row`]): a block reads the packed tiles
    /// holding its `K` range ([`record::READ_RUN`]) and its gather reads only `B`
    /// ([`record::GATHER_A_RESIDENT`]). The kernel unpacks `K` tile `kk`'s faces `(0, k)` from
    /// row `kk % 16` of the packed faces `((kk % 32) / 16, k)` ([`matmul::packed_a_pairs`]):
    /// row 0 of `SrcB` holds the same values, and the other rows only reach output rows past
    /// the product's one.
    pub(crate) packed_a: Option<TensorRef>,
    /// One job a column, its `K` in chunks (`tile_op::matmul::Schedule::Stream`): with `packed_a`,
    /// one tile row, [`Staging::Slots`], and a `K` too long for one block (its chunks no longer
    /// than `k_block`) only.
    pub(crate) stream_k: bool,
    /// Per unit of a capture, the highest `DATA` address the ops it ran since the last
    /// weight-streaming product planned, where all are known: the unit's first job reads its `B`
    /// ahead ([`BUSY_PREFETCH`]) if `B`'s ring lies above it. Empty: none does.
    pub(crate) busy: Vec<Option<u64>>,
    /// With prefetching units and `K` whole, the in-span columns dealt over the own units and the
    /// prefetching ones alike, each unit's share by its weight (indexed by unit; empty: even)
    /// ([`crate::tensor::dealt_runs`], X10's probe). `None`: [`thin_runs`] over the own units.
    pub(crate) deal: Option<Vec<u32>>,
}

impl Plan {
    /// The jobs of `op(A) @ op(B)` as this plan cuts it ([`crate::tile_op::matmul`]), the
    /// operands and the output named by `refs` (`[A, B, C]`, each a tensor or a tile-aligned
    /// block of one: the records read `K` and the output from the ref's first tile on, at the
    /// ref's row stride), appended to `jobs`: job `j` runs on unit `j % units`, so a unit's
    /// blocks and halves continue from the jobs before.
    pub(crate) fn push_jobs(
        &self,
        jobs: &mut Vec<Job>,
        [ra, rb, rc]: [TensorRef; 3],
        [a_transposed, b_transposed]: [bool; 2],
    ) -> Result<()> {
        use crate::tile_op::matmul::{Schedule, Side};
        let [mt, kt, nt] = self.tiles;
        let schedule = if self.stream_k {
            debug_assert_eq!(mt, 1);
            let (kq, _) = stream_chunks(kt, self.k_block, self.route, self.fidelity)?;
            // A column a job: the plan's runs' columns, in order, or every column.
            let cols = self.col_runs.as_ref().map(|runs| {
                runs.iter()
                    .flat_map(|r| r.clone())
                    .map(|j| j..j + 1)
                    .collect()
            });
            Schedule::Stream { kq, cols }
        } else if self.k_block < kt {
            Schedule::KBlocks {
                k_block: self.k_block,
            }
        } else {
            Schedule::Blocks {
                block: self.block,
                cols: self.col_runs.clone(),
                staging: self.staging,
                resident_a: self.resident_a,
                allow_mop: self.allow_mop,
            }
        };
        let elem = |f: Option<TileFormat>| f.map_or(Elem::F32, Elem::Packed);
        let mut op = crate::tile_op::matmul(
            Side {
                at: ra,
                transposed: a_transposed,
                elem: elem(self.route.packed_a()),
                live_rows: self.thin != 0,
            },
            Side {
                at: rb,
                transposed: b_transposed,
                elem: elem(self.route.packed_b()),
                live_rows: false,
            },
            [self.m, kt * 32, nt * 32],
            self.route,
            self.fidelity,
            self.unpack_b,
            self.packed_a,
            schedule,
        )?;
        op.results[0].layout = crate::tile_op::Layout::Tiles(rc);
        op.ahead = self.busy.clone();
        jobs.extend(op.jobs_from(self.units, Overlap::Off, jobs.len())?);
        Ok(())
    }
}

impl Plan {
    /// The jobs of a one-row product with a packed `A`, its `K` whole or streamed
    /// (`tile_op::matmul::Schedule::Stream`), its first columns on the `prefetch` units -- each a job
    /// of the plan's width (a column, streamed), its gather's `B` reads moved into a list of their
    /// own ahead of the job ([`PREFETCH`]; streamed, its first two chunks', which fill the ring's
    /// two halves), its gather left reading the packed `A` -- and the rest dealt over the plan's
    /// units as [`thin_runs`] deals them. Laid out for a deal over `spread` units: a regular unit
    /// `u`'s `k`th job at `k spread + u`, a prefetching unit's at its own index. A prefetching
    /// unit's `B` slots are the job's own, written by its `PREFETCH` list and read by its kernel,
    /// with nothing on the unit between the two (the session takes as prefetching only units
    /// idle since their anchor product; a streamed job's third chunk refills the first half only
    /// after the first chunk's kernel, as unprefetched), so each output column is its
    /// unprefetched self, bit for bit.
    pub(crate) fn push_prefetched_jobs(
        &self,
        jobs: &mut Vec<Job>,
        refs: [TensorRef; 3],
        transposed: [bool; 2],
        prefetch: &[usize],
        spread: usize,
    ) -> Result<()> {
        let [_, kt, nt] = self.tiles;
        let nc = if self.stream_k { 1 } else { self.block[1] };
        // The gathers each prefetched job reads ahead: a streamed column's first two chunks.
        let ahead = if self.stream_k { 2 } else { 1 };
        let firsts = (prefetch.len() * nc).min(nt);
        // X10's probe: the rest over the own units and the prefetching ones alike.
        let deal: Vec<usize> = (0..self.units).chain(prefetch.iter().copied()).collect();
        let dealt = match self.stream_k {
            false => self.deal.as_ref().and_then(|weights| {
                crate::tensor::dealt_runs(firsts..nt, nc, self.units, &deal, weights)
            }),
            true => None,
        };
        // Each unit's runs as a product of its own, so its jobs continue from each other (its first
        // reading ahead where it is busy, `A` resident after it, as when dealt among many).
        let mut per_unit: Vec<Vec<Job>> = Vec::new();
        for (d, runs) in dealt.iter().flatten().enumerate() {
            let mut one = Vec::new();
            if !runs.is_empty() {
                let u = deal[d];
                Plan {
                    units: 1,
                    col_runs: Some(runs.clone()),
                    busy: match u < self.units {
                        true => vec![self.busy.get(u).copied().flatten()],
                        false => Vec::new(),
                    },
                    ..self.clone()
                }
                .push_jobs(&mut one, refs, transposed)?;
            }
            per_unit.push(one);
        }
        let mut regular = Vec::new();
        if dealt.is_none() && firsts < nt {
            let rest = nt - firsts;
            let runs = thin_runs([1, kt, rest], [kt, nc], Staging::Slots, self.units)
                .unwrap_or_else(|| {
                    (0..rest)
                        .step_by(nc)
                        .map(|j| j..(j + nc).min(rest))
                        .collect()
                });
            Plan {
                col_runs: Some(
                    runs.into_iter()
                        .map(|r| r.start + firsts..r.end + firsts)
                        .collect(),
                ),
                ..self.clone()
            }
            .push_jobs(&mut regular, refs, transposed)?;
        }
        let mut early = Vec::new();
        for j0 in (0..firsts).step_by(nc) {
            let plan = Plan {
                units: 1,
                col_runs: Some(std::iter::once(j0..(j0 + nc).min(firsts)).collect()),
                resident_a: false,
                busy: Vec::new(),
                ..self.clone()
            };
            let mut one = Vec::new();
            plan.push_jobs(&mut one, refs, transposed)?;
            let mut job = one.pop().expect("a job a block");
            // Pumped (`GATHER_PREFETCH`, the job's first gather waiting for them): B issues them in
            // its wait loops, so the list may run before the unit's arrival at its anchor's
            // barrier without holding it (`Session::set_idle_ahead_early`). In line, it held the
            // barrier: 0.6 ms a 1.7B step slower (F4a.2).
            let b_reads = super::ahead::take_reads_pumped(&mut job, ahead);
            assert!(!b_reads.is_empty(), "a block's gather record");
            job.insert(
                0,
                Step::List {
                    what: PREFETCH,
                    entries: b_reads,
                },
            );
            early.push(job);
        }
        if dealt.is_some() {
            // Unit `deal[d]`'s `k`th job at `k spread + deal[d]`; a prefetching unit's job read
            // ahead first (`early` is in `prefetch`'s order, as `deal` is past the own units).
            let mut early = early.into_iter();
            let mut laid: Vec<Job> = Vec::new();
            for (d, mut js) in per_unit.into_iter().enumerate() {
                if d >= self.units {
                    if let Some(job) = early.next() {
                        js.insert(0, job);
                    }
                }
                for (k, job) in js.into_iter().enumerate() {
                    let at = k * spread + deal[d];
                    if laid.len() <= at {
                        laid.resize(at + 1, Vec::new());
                    }
                    laid[at] = job;
                }
            }
            laid.resize(laid.len().div_ceil(spread).max(1) * spread, Vec::new());
            jobs.extend(laid);
            return Ok(());
        }
        let units = self.units.max(1);
        let rounds = regular.len().div_ceil(units).max(1);
        let mut laid: Vec<Job> = vec![Vec::new(); rounds * spread];
        for (k, job) in regular.into_iter().enumerate() {
            laid[k / units * spread + k % units] = job;
        }
        for (job, &u) in early.into_iter().zip(prefetch) {
            laid[u] = job;
        }
        jobs.extend(laid);
        Ok(())
    }
}
