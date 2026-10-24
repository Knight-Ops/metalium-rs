//! Attention for a few query rows a key/value head (a decode step's), as one op: each head's
//! scores, softmax and product with its values computed on one unit, the intermediates kept in
//! L1 (decode-to-bandwidth Phase 1).
//!
//! Bit for bit the composition burn-tt's attention runs for the same operands, which is
//! `Q K^T` (a batched product), the scale and mask (`SCALE_MASK_FILL`), the row maximum with
//! its clamp, `SUB_EXP`, the row sum with its clamp, `DIV`, and `P V`:
//! - a product's output row depends only on its `A` row, so a head's query rows read into rows
//!   `0..m` of `A`'s slots (64-byte face rows, wherever they sit in `q`) give its score rows
//!   there, the same products in the same `K` order;
//! - a reduction over columns never moves a value between rows (`sfpu::reduce::row_offsets`),
//!   and the element-wise kinds are lanewise, so a head's score rows reduce and divide as the
//!   same rows of the stacked `[heads, L]` scores do;
//! - the row-limited programs (`sfpu::ops::code_for_rows`, `reduce::math_programs_rows`; gates
//!   `step200` and the unit gates) compute those `m` rows as the whole-tile programs do. A head's
//!   rows at row 0 matter: built for rows `0..r0 + m`, a head at row 31 of a stacked `q` ran the
//!   whole-tile `SUB_EXP` and `DIV`, ~8 us a score tile (`silicon_probe_attention_rows`).
//!
//! A job is seven blocks on its unit, each a gather list, a kernel and a scatter list
//! (`tensor::Step`): the first reads the head's query rows, keys, mask and values; the next
//! five read nothing (their operands are the previous kernel's outputs, which the roles'
//! retirement between batches orders); the last writes the head's rows of the output.

use std::sync::Arc;

use tt_isa::dm::{op, record, TILE_DATA, TILE_SLOT};
use tt_isa::tile::L1Format;

use crate::matmul::{self, Fidelity, LiveRows, OutputTile, SrcRoute, Staging};
use crate::sfpu::kernel as sk;
use crate::sfpu::ops::{kind_sfpu, Broadcast};
use crate::sfpu::reduce::{self, ReduceOp};
use crate::tensor::{block_ref, DramAlloc, DramTensor, Elem, Result, Step};
use crate::tensor::{TensorError, Work};

/// One key/value head's operands, each the top-left element of its block: `q` of its `[m, d]`
/// query rows, `k` of its `[l, d]` keys (read transposed), `v` of its `[l, dv]` values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Item {
    pub q: [usize; 2],
    pub k: [usize; 2],
    pub v: [usize; 2],
}

/// Which of a job's seven kernels run ([`probe_stages`]): bit `s` for kernel `s`.
static PROBE_STAGES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

/// Run only the kernels whose bit is set in `mask` (`qk`, scale and mask, maximum, `SUB_EXP`,
/// sum, `DIV`, `pv`: bits 0-6), the reads and writes kept, so a probe can time each: the result
/// is then wrong. Kernel 0 always runs. For `silicon_probe_attention_rows` only.
#[doc(hidden)]
pub fn probe_stages(mask: u32) {
    PROBE_STAGES.store(mask | 1, std::sync::atomic::Ordering::SeqCst);
}

/// Whether each head's positions run as two halves on two units ([`probe_split`]).
static PROBE_SPLIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Run each head whose positions are an even number of tiles as two jobs, each over half of
/// them on its own unit, the first half's jobs then the second's, both writing the head's rows:
/// flash-decode's reads and kernels with no exchange between the halves, so a probe can time
/// them (the result is then wrong). For `silicon_probe_attention_rows` and
/// `traced_generation_stream_ceiling` only.
#[doc(hidden)]
pub fn probe_split(on: bool) {
    PROBE_SPLIT.store(on, std::sync::atomic::Ordering::SeqCst);
}

/// How many jobs a head runs as under [`probe_split`]: 2 for an even number of position tiles.
#[doc(hidden)]
pub fn probe_runs(l: usize) -> usize {
    let nt = l / 32;
    if PROBE_SPLIT.load(std::sync::atomic::Ordering::Relaxed) && nt >= 2 && nt % 2 == 0 {
        2
    } else {
        1
    }
}

/// Most row writes a job's last block may hold: two 64-byte face rows a row and output tile, for
/// the result and for its packed twin.
const MAX_ROW_WRITES: usize = 128;

/// The slots of one job, in the data arena.
#[derive(Copy, Clone, Debug)]
struct Slots {
    q: u64,
    k: u64,
    mask: u64,
    s: u64,
    t: u64,
    max: u64,
    sum: u64,
    v: u64,
    o: u64,
}

/// L1 slots a job of `[kt, nt, dvt]` tiles takes: `q`'s tile row, the keys, the mask, two
/// score buffers, the maximum and the sum, the values and the output.
fn slots_needed([kt, nt, dvt]: [usize; 3]) -> usize {
    kt + kt * nt + nt + 2 * nt + 2 + nt * dvt + dvt
}

fn slot_at(base: u64, n: usize) -> u64 {
    base + n as u64 * TILE_SLOT
}

fn plan_slots(tiles: [usize; 3]) -> Option<Slots> {
    let [kt, nt, dvt] = tiles;
    let arena = crate::l1::arena();
    if slots_needed(tiles) as u64 * TILE_SLOT > arena.len() {
        return None;
    }
    let mut at = arena.base;
    let mut take = |n: usize| {
        let here = at;
        at += n as u64 * TILE_SLOT;
        here
    };
    Some(Slots {
        q: take(kt),
        k: take(kt * nt),
        mask: take(nt),
        s: take(nt),
        t: take(nt),
        max: take(1),
        sum: take(1),
        v: take(nt * dvt),
        o: take(dvt),
    })
}

/// Whether [`attention_rows`] takes a problem of these dims (`[m, d, l, dv]`) whose query rows
/// start at row `r0` of their tile: whole tiles along `d`, `l` and `dv`, a head's rows inside
/// one tile row, and a job inside the data arena and one list.
pub fn fits([m, d, l, dv]: [usize; 4], r0: usize) -> bool {
    let tiles = [d / 32, l / 32, dv / 32];
    m >= 1
        && r0 + m <= 32
        && d % 32 == 0
        && l % 32 == 0
        && dv % 32 == 0
        && d > 0
        && l > 0
        && dv > 0
        && 4 * m * tiles[2] <= MAX_ROW_WRITES
        && plan_slots(tiles).is_some()
        && reduce::plan_layout(1, tiles[1]).is_ok()
}

/// The kernels of a job, memoised: they name only L1 addresses, so every head of every step
/// with the same tiles, rows and scale shares them.
#[allow(clippy::too_many_arguments)]
fn kernels(
    tiles: [usize; 3],
    rows: usize,
    scale: f32,
    route: SrcRoute,
    fidelity: Fidelity,
    unpack_b: bool,
) -> Result<Arc<Vec<Step>>> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = ([usize; 3], usize, u32, SrcRoute, Fidelity, bool, u64);
    // Each job's steps with what their plans reached and needed, noted again on every hit
    // (`crate::l1::note_planned`); by the arena's end too, where they were planned.
    type Built = (Arc<Vec<Step>>, crate::l1::Planned);
    static MEMO: OnceLock<Mutex<HashMap<Key, Built>>> = OnceLock::new();
    let key = (
        tiles,
        rows,
        scale.to_bits(),
        route,
        fidelity,
        unpack_b,
        crate::l1::arena().end,
    );
    let memo = MEMO.get_or_init(Default::default);
    if let Some((k, p)) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        crate::l1::note_planned(*p);
        return Ok(k.clone());
    }
    let before = crate::l1::take_planned();
    let built = build_kernels(tiles, rows, scale, route, fidelity, unpack_b);
    let p = crate::l1::take_planned();
    crate::l1::note_planned(before);
    crate::l1::note_planned(p);
    let built = Arc::new(built?);
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key, (built.clone(), p));
    Ok(built)
}

fn shape_error(what: impl std::fmt::Display) -> TensorError {
    TensorError::Shape(format!("attention rows: {what}"))
}

fn build_kernels(
    [kt, nt, dvt]: [usize; 3],
    rows: usize,
    scale: f32,
    route: SrcRoute,
    fidelity: Fidelity,
    unpack_b: bool,
) -> Result<Vec<Step>> {
    let at = plan_slots([kt, nt, dvt]).ok_or_else(|| shape_error("the job does not fit L1"))?;
    let (in_fmt, out_fmt) = route.formats();
    let live = LiveRows::for_rows(rows);
    let slot = |base: u64, n: usize| base + n as u64 * TILE_SLOT;
    let matmul_step = |outputs: Vec<OutputTile>, shape: [usize; 3], b_transposed: bool| {
        let layout =
            matmul::plan_layout_in(shape, L1Format::Fp32, Staging::Slots).map_err(shape_error)?;
        let (roles, mop) = matmul::matmul_kernel(
            &outputs,
            layout.sems,
            in_fmt,
            out_fmt,
            fidelity,
            false,
            b_transposed,
            live,
        );
        Ok::<_, TensorError>(Step::Kernel {
            roles: Arc::new(roles),
            init: layout.init,
            mop: Box::new(mop),
            loops: Default::default(),
            half: None,
        })
    };
    // `S = q K^T`: output tile `j` accumulates `K` tile `kk` of `q`'s row with key tile
    // `(kk, j)` of `K^T`, as `record::GATHER` lays them.
    let qk = (0..nt)
        .map(|j| OutputTile {
            pairs: (0..kt)
                .map(|kk| (slot(at.q, kk), slot(at.k, kk * nt + j)))
                .collect(),
            out: slot(at.s, j) + TILE_DATA,
        })
        .collect();
    let qk = matmul_step(qk, [1, kt, nt], unpack_b)?;
    let sfpu_step = |kind: u32, scalars: [f32; 2], bcast: Broadcast, a: u64, b: u64, out: u64| {
        let (operands, code) =
            crate::sfpu::ops::code_for_rows(kind, scalars, bcast, rows as u32)
                .ok_or_else(|| shape_error(format!("no program for kind {kind:#x}")))?;
        let planned = sk::plan_layout(nt, operands).map_err(shape_error)?;
        let layout = sk::Layout {
            tiles: nt,
            a_at: a,
            b_at: Some(b),
            b_once: bcast == Broadcast::Col,
            c_at: None,
            out_at: out,
            sems: planned.sems,
            init: planned.init,
        };
        let (roles, loops) = sk::roles_code(&layout, operands, &code);
        Ok::<_, TensorError>(Step::Kernel {
            roles: Arc::new(roles),
            init: layout.init,
            mop: Box::new([None; 3]),
            loops: Arc::new(loops),
            half: None,
        })
    };
    let reduce_step = |op: ReduceOp, clamp: f32, input: u64, out: u64| {
        let planned = reduce::plan_layout(1, nt).map_err(shape_error)?;
        let layout = reduce::Layout {
            in_at: input,
            out_at: out,
            ..planned
        };
        // Only the row groups of the head's rows: the reduction and its clamp leave the others.
        let (inputs, mut fin) = reduce::math_programs_rows(op, nt, 32, rows as u32);
        let clamp = crate::sfpu::with_rows(rows as u32, || {
            crate::sfpu::ops::exact_in_place(kind_sfpu::CLAMP_MIN, [clamp, 0.0], sk::OUT_ROW)
        })
        .ok_or_else(|| shape_error("a reduction's clamp is no exact kind"))?;
        fin.extend(clamp);
        Ok::<_, TensorError>(Step::Kernel {
            roles: Arc::new(reduce::roles(&layout, &inputs, &fin)),
            init: layout.init,
            mop: Box::new([None; 3]),
            loops: Default::default(),
            half: None,
        })
    };
    let scale_mask = sfpu_step(
        kind_sfpu::SCALE_MASK_FILL,
        [scale, f32::NEG_INFINITY],
        Broadcast::Row,
        at.s,
        at.mask,
        at.t,
    )?;
    let max = reduce_step(ReduceOp::Max, -f32::MAX, at.t, at.max)?;
    let sub_exp = sfpu_step(
        kind_sfpu::SUB_EXP,
        [0.0, 0.0],
        Broadcast::Col,
        at.t,
        at.max,
        at.s,
    )?;
    let sum = reduce_step(ReduceOp::Sum, f32::MIN_POSITIVE, at.s, at.sum)?;
    let div = sfpu_step(
        kind_sfpu::DIV,
        [0.0, 0.0],
        Broadcast::Col,
        at.s,
        at.sum,
        at.t,
    )?;
    // `O = P V`: `P` is the division's output, `K` tile `kk` of output tile `j` value tile
    // `(kk, j)`.
    let pv = (0..dvt)
        .map(|j| OutputTile {
            pairs: (0..nt)
                .map(|kk| (slot(at.t, kk), slot(at.v, kk * dvt + j)))
                .collect(),
            out: slot(at.o, j) + TILE_DATA,
        })
        .collect();
    let pv = matmul_step(pv, [1, nt, dvt], false)?;
    Ok(vec![qk, scale_mask, max, sub_exp, sum, div, pv])
}

/// `softmax(scale q_i K_i^T, masked) V_i` for each key/value head `i` of `items`, as one op:
/// `q_i` is `[m, d]`, `K_i` `[l, d]`, `V_i` `[l, dv]`, `mask` the `[1, l]` positions not
/// attended (`true`), the same for every row. The result is the `[1, items.len() m dv]` row: the
/// `[items.len() m, dv]` stack of the heads' rows (as burn-tt's batched `P V` stacks them) in
/// row-major order, which is the row a projection after attention reads; its padding is
/// undefined. `twin`, if given (`[items.len() m dv / 32, 32]`), receives the row packed
/// (`tensor::pack_row`'s layout), as an element-wise op's packed twin.
///
/// Refused unless [`fits`] holds for every head, and for F32 operands on the TF32 route.
#[allow(clippy::too_many_arguments)]
pub fn attention_rows(
    alloc: &mut DramAlloc,
    q: &DramTensor,
    k: &DramTensor,
    v: &DramTensor,
    mask: &DramTensor,
    items: &[Item],
    [m, d, l, dv]: [usize; 4],
    scale: f32,
    route: SrcRoute,
    fidelity: Fidelity,
    unpack_b: bool,
    twin: Option<&DramTensor>,
) -> Result<Work> {
    for (t, what) in [(q, "queries"), (k, "keys"), (v, "values")] {
        t.expect(&format!("attention rows' {what}"), Elem::F32)?;
    }
    mask.expect("attention rows' mask", Elem::Bool)?;
    if route != SrcRoute::Tf32FromFp32 {
        return Err(shape_error(format!("route {route:?}")));
    }
    if items.is_empty() {
        return Err(shape_error("no heads"));
    }
    if (mask.rows, mask.cols) != (1, l) {
        return Err(shape_error(format!(
            "a [{}, {}] mask for {l} positions",
            mask.rows, mask.cols
        )));
    }
    let [kt, nt, dvt] = [d / 32, l / 32, dv / 32];
    let out = DramTensor::alloc(alloc, 1, items.len() * m * dv)?;
    if let Some(t) = twin {
        if (t.rows, t.cols) != (items.len() * m * dvt, 32) {
            alloc.free(&out.placement);
            return Err(shape_error(format!(
                "a twin [{}, {}] for a row of {} tiles",
                t.rows,
                t.cols,
                items.len() * m * dvt
            )));
        }
    }
    let runs = probe_runs(l);
    let (l, nt) = (l / runs, nt / runs);
    let result = (|| {
        let at = plan_slots([kt, nt, dvt]).ok_or_else(|| shape_error("the job does not fit L1"))?;
        let rmask = mask.tensor_ref();
        let mut jobs = Vec::with_capacity(runs * items.len());
        for (run, (i, item)) in
            (0..runs).flat_map(|run| items.iter().enumerate().map(move |e| (run, e)))
        {
            let item = &Item {
                k: [item.k[0] + run * l, item.k[1]],
                v: [item.v[0] + run * l, item.v[1]],
                ..*item
            };
            let [qr, qc] = item.q;
            let r0 = qr % 32;
            if !fits([m, d, l, dv], r0) || qr + m > q.rows || qc % 32 != 0 || qc + d > q.cols {
                return Err(shape_error(format!(
                    "head {i}'s [{m}, {d}] query rows at [{qr}, {qc}] of [{}, {}]",
                    q.rows, q.cols
                )));
            }
            let tile_row = qr - r0;
            let rk = block_ref(k, item.k, [l, d], "a head's keys")?;
            let rv = block_ref(v, item.v, [l, dv], "a head's values")?;
            let steps = kernels([kt, nt, dvt], m, scale, route, fidelity, unpack_b)?;
            let gather_flags = |a_t: bool, b_t: bool, kk: usize| {
                u32::from(a_t) | u32::from(b_t) << 1 | (kk as u32) << 8
            };
            // The head's query rows into rows `0..m` of `A`'s slots, two 64-byte face rows a
            // row and `K` tile, wherever they sit in `q`'s tile row: every head then computes
            // only its own rows (a product's output row is its `A` row's alone).
            let mut gather = Vec::with_capacity(2 * m * kt + 5);
            for r in 0..m {
                let rr = (r0 + r) as u64;
                for kk in 0..kt {
                    let src = q.tile(tile_row / 32, qc / 32 + kk);
                    for half in 0..2u64 {
                        gather.push([
                            op::READ,
                            src.channel().index() as u32,
                            ((r + kk) % tt_isa::dram::PORTS as usize) as u32,
                            (src.offset()
                                + TILE_DATA
                                + (2 * (rr / 16) + half) * 1024
                                + (rr % 16) * 64) as u32,
                            (slot_at(at.q, kk)
                                + TILE_DATA
                                + (2 * (r as u64 / 16) + half) * 1024
                                + (r as u64 % 16) * 64) as u32,
                            64,
                            0,
                            0,
                        ]);
                    }
                }
            }
            // The keys: `B` of the first product, its `A` already in L1.
            gather.extend([
                [
                    record::GATHER,
                    gather_flags(false, true, kt)
                        | record::GATHER_A_RESIDENT
                        | if unpack_b {
                            record::GATHER_B_UNPACK_TRANSPOSE
                        } else {
                            0
                        },
                    at.q as u32,
                    at.k as u32,
                    0,
                    1,
                    0,
                    nt as u32,
                ],
                rk.encode()[0],
                rk.encode()[1],
                rk.encode()[0],
                rk.encode()[1],
            ]);
            gather.extend([
                [
                    record::READ_RUN,
                    0,
                    nt as u32,
                    at.mask as u32,
                    0,
                    nt as u32,
                    0,
                    0,
                ],
                rmask.encode()[0],
                rmask.encode()[1],
            ]);
            // The values: `B` of the last product, its `A` (`P`) already in L1.
            gather.extend([
                [
                    record::GATHER,
                    gather_flags(false, false, nt) | record::GATHER_A_RESIDENT,
                    at.t as u32,
                    at.v as u32,
                    0,
                    1,
                    0,
                    dvt as u32,
                ],
                rv.encode()[0],
                rv.encode()[1],
                rv.encode()[0],
                rv.encode()[1],
            ]);
            // Each of the head's rows into its place in the result's row: tile `dvt (i m + r) + j`,
            // its row 0, two 64-byte face rows a tile; and into the twin, if there is one, as
            // `tensor::pack_row` lays the row out (tile `kk`'s data row its row `kk`).
            let face_row =
                |r: u64, half: u64| TILE_DATA + (2 * (r / 16) + half) * 1024 + (r % 16) * 64;
            let mut writes = Vec::with_capacity(4 * m * dvt);
            for r in 0..m {
                for j in 0..dvt {
                    let kk = (i * m + r) * dvt + j;
                    let src = at.o + j as u64 * TILE_SLOT;
                    let mut targets = vec![(out.tile(0, kk), 0u64)];
                    if let Some(twin) = twin {
                        targets.push((twin.tile(kk / 32, 0), (kk % 32) as u64));
                    }
                    for (dst, dr) in targets {
                        for half in 0..2u64 {
                            writes.push([
                                op::WRITE,
                                dst.channel().index() as u32,
                                0,
                                (dst.offset() + face_row(dr, half)) as u32,
                                (src + face_row(r as u64, half)) as u32,
                                64,
                                0,
                                0,
                            ]);
                        }
                    }
                }
            }
            let stages = PROBE_STAGES.load(std::sync::atomic::Ordering::Relaxed);
            let steps: Vec<&Step> = steps
                .iter()
                .enumerate()
                .filter(|(s, _)| stages >> s & 1 != 0)
                .map(|(_, k)| k)
                .collect();
            let last = steps.len() - 1;
            let mut job = Vec::with_capacity(3 * steps.len());
            for (s, &kernel) in steps.iter().enumerate() {
                job.push(Step::List {
                    what: "attention rows gather",
                    entries: if s == 0 {
                        std::mem::take(&mut gather)
                    } else {
                        Vec::new()
                    },
                });
                job.push(kernel.clone());
                job.push(Step::List {
                    what: "attention rows scatter",
                    entries: if s == last {
                        std::mem::take(&mut writes)
                    } else {
                        Vec::new()
                    },
                });
            }
            crate::tensor::share_semaphore_init(&mut job);
            jobs.push(job);
        }
        Ok(jobs)
    })();
    match result {
        Ok(jobs) => Ok(Work { out, jobs }),
        Err(e) => {
            alloc.free(&out.placement);
            Err(e)
        }
    }
}
