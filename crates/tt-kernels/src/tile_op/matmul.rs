//! A product `op(A) @ op(B)` as a [`TileOp`]: the domain `[m, n, k]`, `m` and `n` parallel and `k`
//! a reduction whose terms are added in ascending order, `A` mapped `[m, k]` and `B` `[k, n]`
//! (either stored transposed), the result `[m, n]`. The body is the Matrix Unit's (`Mma`): every
//! output tile's `K` chain through FP32 partials in `Dst`, at the route's `Src` formats and the
//! fidelity's phases.
//!
//! How the product is cut is its [`Schedule`], chosen before the description is built (the cost
//! models in `tensor`), and always one of the bit-equivalent options under the default contract.

use tt_isa::dm::record::{self, TensorRef};
use tt_isa::dm::{TILE_DATA, TILE_SLOT};
use tt_isa::format::TileFormat;

use std::ops::Range;

use super::{
    Access, Blocks, Chunk, Cols, Combine, Dim, Iter, Layout, Microkernel, NumericContract,
    ReductionContract, RunCode, TileMap, TileOp,
};
use crate::matmul::{self, BlockStreamLayout, Fidelity, LiveRows, SrcRoute, Staging, StreamLayout};
use crate::tensor::{
    self, balanced_shares, stream_chunks, thin_runs, Elem, Job, Overlap, Result, Step, TensorError,
    HANDED_AHEAD, PREFETCH,
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
    /// [`Reads::second_job`].
    pub(crate) second_job: bool,
    /// [`Reads::balanced_deal`].
    pub(crate) balanced_deal: bool,
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
            let (kq, layout) = stream_chunks(kt, self.k_block, self.route, self.fidelity)?;
            // A column a job: the plan's runs' columns, in order, or every column.
            let cols = self.col_runs.as_ref().map(|runs| {
                runs.iter()
                    .flat_map(|r| r.clone())
                    .map(|j| j..j + 1)
                    .collect()
            });
            Schedule::Stream { kq, layout, cols }
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
    ///
    /// With `handoff` (`(h, partials)`, streamed only), each column dealt to the plan's units is
    /// handed while a unit of the deal is left with no job: the column's unit computes its first
    /// `h` chunks (`0`: half) and writes the partial to `partials`, the spare unit reads the
    /// packed `A` and the next two chunks' `B` ahead ([`HANDED_AHEAD`]) and, in the second op
    /// returned (laid out as the first), reloads the partial and computes the rest ([`Handoff`]).
    /// Empty: no column is handed.
    pub(crate) fn push_prefetched_jobs(
        &self,
        jobs: &mut Vec<Job>,
        refs: [TensorRef; 3],
        transposed: [bool; 2],
        prefetch: &[usize],
        spread: usize,
        handoff: Option<(usize, TensorRef)>,
    ) -> Result<Vec<Job>> {
        let [_, kt, nt] = self.tiles;
        let nc = if self.stream_k { 1 } else { self.block[1] };
        // The gathers each prefetched job reads ahead: a streamed column's first two chunks.
        let ahead = if self.stream_k { 2 } else { 1 };
        let firsts = (prefetch.len() * nc).min(nt);
        // A `K`-whole product's prefetching units each take a second column after their first
        // job's, its `B` in a ring directly below the first job's (`crate::l1::ring_end` lowered by
        // that ring), read ahead in the same list: what the window carries grows where L1 holds it
        // (in a capture planning into the trace arena), and nothing changes where it does not. Only
        // with `B` against the ring's end ([`Staging::Slots`]): a pipelined product's jobs stage
        // `B` in a half of the arena, the second's where the first's is.
        // And only where the plan's own units would still run more than a round of columns: where
        // the first jobs leave less, the window already carries what the product reads, and a
        // second job is only a column more of compute on a unit the others then wait for (a 1.7B
        // step's `qkv` 51 -> 80 us and `o` 22 -> 33 us, profile `1792721989`).
        let busy_rounds = (nt - firsts).div_ceil((self.units * nc).max(1));
        let seconds = match self.second_job
            && !self.stream_k
            && self.staging == Staging::Slots
            && busy_rounds > 1
        {
            true => self.second_jobs(refs, transposed, firsts, nt, prefetch.len())?,
            false => Vec::new(),
        };
        let moved = firsts + seconds.len();
        // The columns the span reads dealt over the prefetching units too: where the plan's own
        // units would run more than a round, the prefetching units otherwise finish their job read
        // ahead in ~13 us and wait at the barrier for units reading 5 columns more (a 1.7B step's
        // `gate|up`, profile `1792762439`).
        let balanced = self.balanced_deal
            && seconds.is_empty()
            && !self.stream_k
            && self.staging == Staging::Slots
            && busy_rounds > 1;
        let mut regular = Vec::new();
        // The regular jobs' columns, in their order (a streamed job's one).
        let mut regular_cols = Vec::new();
        if moved < nt && !balanced {
            let rest = nt - moved;
            let runs = thin_runs([1, kt, rest], [kt, nc], Staging::Slots, self.units)
                .unwrap_or_else(|| {
                    (0..rest)
                        .step_by(nc)
                        .map(|j| j..(j + nc).min(rest))
                        .collect()
                });
            let plan = Plan {
                col_runs: Some(
                    runs.into_iter()
                        .map(|r| r.start + moved..r.end + moved)
                        .collect(),
                ),
                ..self.clone()
            };
            plan.push_jobs(&mut regular, refs, transposed)?;
            regular_cols = plan
                .col_runs
                .iter()
                .flatten()
                .flat_map(|r| r.clone())
                .collect();
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
            let b_reads = super::ahead::take_reads(&mut job, ahead, false);
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
        let units = self.units.max(1);
        if balanced {
            return self.push_balanced_jobs(jobs, refs, transposed, early, prefetch, spread);
        }
        let mut handed = Vec::new();
        let mut handed_ahead = Vec::new();
        if let Some((at, rp)) = handoff.filter(|_| self.stream_k) {
            let (kq, layout) = stream_chunks(kt, self.k_block, self.route, self.fidelity)?;
            let chunks = kt.div_ceil(kq);
            let h = match at {
                0 => chunks / 2,
                at => at.min(chunks - 1),
            };
            // The units of the deal with no job: neither the plan's nor a prefetching one.
            let taken: Vec<usize> = (0..regular.len().min(units))
                .chain(prefetch.iter().copied().take(early.len()))
                .collect();
            let mut spare = (0..spread).filter(|u| !taken.contains(u));
            for (job, &j) in regular.iter_mut().zip(&regular_cols) {
                if h == 0 {
                    break;
                }
                let Some(y) = spare.next() else {
                    break;
                };
                let plan = Plan {
                    units: 1,
                    col_runs: Some(std::iter::once(j..j + 1).collect()),
                    resident_a: false,
                    busy: Vec::new(),
                    ..self.clone()
                };
                let mut whole = Vec::new();
                plan.push_jobs(&mut whole, refs, transposed)?;
                let mut rest = whole.pop().expect("a job a column");
                let first = rest.drain(..3 * h).collect::<Vec<_>>();
                // The column's unit: its first `h` chunks, the last one's partial to `partials`.
                job.truncate(3 * h);
                let Some(Step::List { entries, .. }) = job.last_mut() else {
                    unreachable!("a chunk ends with its scatter list");
                };
                *entries = vec![
                    [
                        record::SCATTER,
                        (layout.out[(h - 1) % 2] + tt_isa::dm::TILE_DATA) as u32,
                        TILE_SLOT as u32,
                        0,
                        1,
                        j as u32,
                        1,
                        0,
                    ],
                    rp.encode()[0],
                    rp.encode()[1],
                ];
                // The spare unit, first op: the packed `A` (chunk 0's reads before its gather)
                // and the next two chunks' gathers as a prefetch.
                let mut ahead: Vec<[u32; 8]> = match first.first() {
                    Some(Step::List { entries, .. }) => entries
                        .iter()
                        .take_while(|e| e[0] != record::GATHER)
                        .copied()
                        .collect(),
                    _ => unreachable!("a chunk starts with its gather list"),
                };
                ahead.extend(super::ahead::take_reads(&mut rest, 2, true));
                // The second op: the partial into the slot chunk `h` reloads from, after the
                // first op's barrier, and the prefetch drained before chunk `h`'s kernel.
                let Some(Step::List { entries, .. }) = rest.first_mut() else {
                    unreachable!("a chunk starts with its gather list");
                };
                let mut lead = vec![
                    [tt_isa::dm::op::WAIT, 0, 0, 0, 0, 0, 0, 0],
                    [
                        record::READ_RUN,
                        j as u32,
                        1,
                        layout.out[(h - 1) % 2] as u32,
                        0,
                        0,
                        0,
                        0,
                    ],
                    rp.encode()[0],
                    rp.encode()[1],
                    [
                        tt_isa::dm::op::WAIT,
                        tt_isa::dm::WAIT_PREFETCH,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                ];
                lead.append(entries);
                *entries = lead;
                if handed.is_empty() {
                    handed = vec![Vec::new(); spread];
                }
                handed[y] = rest;
                handed_ahead.push((
                    y,
                    vec![Step::List {
                        what: HANDED_AHEAD,
                        entries: ahead,
                    }],
                ));
            }
        }
        let rounds = regular
            .len()
            .div_ceil(units)
            .max(1 + usize::from(!seconds.is_empty()));
        let mut laid: Vec<Job> = vec![Vec::new(); rounds * spread];
        for (k, job) in regular.into_iter().enumerate() {
            laid[k / units * spread + k % units] = job;
        }
        let mut seconds = seconds.into_iter();
        for (mut job, &u) in early.into_iter().zip(prefetch) {
            // The second job's `B` reads after the first's, in its one list.
            if let Some((b_reads, second)) = seconds.next() {
                let Some(Step::List { entries, .. }) = job.first_mut() else {
                    unreachable!("a prefetched job starts with its list");
                };
                entries.extend(b_reads);
                laid[spread + u] = second;
            }
            laid[u] = job;
        }
        for (u, job) in handed_ahead {
            laid[u] = job;
        }
        jobs.extend(laid);
        Ok(handed)
    }

    /// The jobs of a product dealt by [`Reads::balanced_deal`], laid a round of `spread` units at a
    /// time: each prefetching unit's job read ahead (`early`, columns `0..firsts`) and the plan's
    /// own units' first round (read ahead where [`BUSY_PREFETCH`] allows) as without it; then the
    /// columns left, in [`balanced_shares`] over the plan's units and then the prefetching ones, in
    /// runs of at most the plan's width, each unit's after its first job in the same ring. Every
    /// output tile's accumulation is unchanged: only which unit holds it moves.
    fn push_balanced_jobs(
        &self,
        jobs: &mut Vec<Job>,
        refs: [TensorRef; 3],
        transposed: [bool; 2],
        early: Vec<Job>,
        prefetch: &[usize],
        spread: usize,
    ) -> Result<Vec<Job>> {
        let [_, _, nt] = self.tiles;
        let nc = self.block[1];
        let units = self.units.max(1);
        let firsts = (prefetch.len() * nc).min(nt);
        let lead = (units * nc).min(nt - firsts);
        let mut first_round = Vec::new();
        Plan {
            col_runs: Some(
                (0..lead)
                    .step_by(nc)
                    .map(|j| firsts + j..firsts + (j + nc).min(lead))
                    .collect(),
            ),
            ..self.clone()
        }
        .push_jobs(&mut first_round, refs, transposed)?;
        let deal: Vec<usize> = (0..units)
            .chain(prefetch.iter().copied().take(early.len()))
            .collect();
        let mut at = firsts + lead;
        let mut later: Vec<Vec<Job>> = Vec::with_capacity(deal.len());
        for share in balanced_shares(nt - at, deal.len()) {
            let mut own = Vec::new();
            if share > 0 {
                Plan {
                    units: 1,
                    col_runs: Some(
                        (0..share)
                            .step_by(nc)
                            .map(|j| at + j..at + (j + nc).min(share))
                            .collect(),
                    ),
                    busy: Vec::new(),
                    ..self.clone()
                }
                .push_jobs(&mut own, refs, transposed)?;
            }
            at += share;
            later.push(own);
        }
        debug_assert_eq!(at, nt, "every column dealt");
        let rounds = 1 + later.iter().map(Vec::len).max().unwrap_or(0);
        let mut laid: Vec<Job> = vec![Vec::new(); rounds * spread];
        for (u, job) in first_round.into_iter().enumerate() {
            laid[u] = job;
        }
        for (job, &u) in early.into_iter().zip(prefetch) {
            laid[u] = job;
        }
        for (&u, own) in deal.iter().zip(later) {
            for (r, job) in own.into_iter().enumerate() {
                laid[(1 + r) * spread + u] = job;
            }
        }
        jobs.extend(laid);
        Ok(Vec::new())
    }

    /// The second jobs of `count` prefetching units ([`Reads::second_job`]): columns `first..`, one
    /// each, each its gather's `B` reads (split off, for the unit's prefetch list) and the job left
    /// reading the packed `A`. Its `B` ring is directly below the first job's, which sits against
    /// [`crate::l1::ring_end`] and is `kt` columns of the plan's width (a narrower last first job
    /// leaves a gap). None where that does not plan: no second ring fits below the first.
    fn second_jobs(
        &self,
        refs: [TensorRef; 3],
        transposed: [bool; 2],
        first: usize,
        nt: usize,
        count: usize,
    ) -> Result<Vec<(Vec<[u32; 8]>, Job)>> {
        let kt = self.tiles[1];
        let ring = (kt * self.block[1]) as u64 * self.route.b_slot();
        let top = crate::l1::ring_end();
        if top < tt_isa::l1::DATA.base + ring {
            return Ok(Vec::new());
        }
        let was = crate::l1::set_ring_end(Some(top - ring));
        let built = (first..(first + count).min(nt))
            .map(|j| {
                let plan = Plan {
                    units: 1,
                    col_runs: Some(std::iter::once(j..j + 1).collect()),
                    resident_a: false,
                    busy: Vec::new(),
                    second_job: false,
                    balanced_deal: false,
                    ..self.clone()
                };
                let mut one = Vec::new();
                plan.push_jobs(&mut one, refs, transposed)?;
                let mut job = one.pop().expect("a job a block");
                let b_reads = super::ahead::take_reads(&mut job, 1, false);
                assert!(!b_reads.is_empty(), "a block's gather record");
                Ok((b_reads, job))
            })
            .collect::<Result<Vec<_>>>();
        crate::l1::set_ring_end(was);
        // A second ring that does not plan below the first (outside the trace arena) is none.
        Ok(built.unwrap_or_default())
    }
}
