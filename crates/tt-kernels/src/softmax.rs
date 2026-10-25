//! Attention's softmax over rows of any length, as one op (`docs/plans/roadmap.md` Track 1, "Then
//! the long-context softmax"): `p = softmax(scale x, masked)` over each row of `x`, bit for bit the
//! composition burn-tt's attention runs -- `SCALE_MASK_FILL`, the row maximum and its clamp,
//! `SUB_EXP`, the row sum and its clamp, `DIV` -- with the scores read three times and written
//! twice instead of eleven passes, and no row too long for it.
//!
//! A job is one tile row, on one unit, in three passes (`sfpu::softmax` has the math and why it
//! is the composition's bits):
//!
//! 1. [`Pass::Max`]: `x` and the mask in, `s = SCALE_MASK_FILL(x, mask)` out (into the output
//!    tensor, which holds `s` until pass 3 overwrites it), the maximum accumulated;
//! 2. [`Pass::Sum`]: `s` in, `SUB_EXP(s, max)`'s sum accumulated;
//! 3. [`Pass::Out`]: `s` in, `DIV(SUB_EXP(s, max), sum)` out, over `s` where it lies.
//!
//! Each pass streams the row in chunks of [`Layout::kq`] tiles, a kernel a chunk, the chunks in
//! alternate halves of L1 so the next chunk's reads and the last one's writes run under the
//! kernel (`Step::Kernel::half`). A statistic's unfolded accumulator stays in L1 between chunks
//! (packed raw, unpacked raw, as `reduce::continuation_roles` carries it through GDDR), and the
//! statistics stay in L1 for the passes after. A pass starts in the half the last one ended in,
//! so its blocks begin a new packet (`session::execution_groups` joins only alternating halves):
//! B starts it only once NC has written the last pass's output, which pass 2 and 3 read back.

use std::sync::Arc;

use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::dm::{record, TILE_DATA, TILE_SLOT};
use tt_isa::isa::Instruction;
use tt_isa::sync::{self, Semaphore, Unit};

use crate::code::{Code, Loop, LoopedRoles};
use crate::datapath::{
    clear_unpacker0_adcs, config_program, pack_dest_units, pack_tile_from_dst,
    pack_tile_from_dst_gpr, state_id, thread_config, tile_base_units, tile_unpack_config,
    unpack_datums_to_dst_gpr, unpack_tile_to_dst, TILE_DATUMS,
};
use crate::l1::{PlanError, Requirements};
use crate::runtime::SemaphoreInit;
use crate::sfpu::kernel::{A_ROW, B_ROW, C_ROW, OUT_ROW};
use crate::sfpu::reduce::ReduceOp;
use crate::sfpu::softmax::{self as math, MAX_ROW, SUM_ROW};
use crate::tensor::{DramAlloc, DramTensor, Elem, Result, Step, TensorError, Work};

/// Most tiles a chunk takes: six slots a tile (two halves of `x`, the mask and the output),
/// inside the data arena with room to spare.
pub const MAX_KQ: usize = 32;

/// One of a job's three passes over its row.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pass {
    /// `s` out, the row maximum.
    Max,
    /// The row sum of `SUB_EXP(s, max)`.
    Sum,
    /// `DIV(SUB_EXP(s, max), sum)` out.
    Out,
}

impl Pass {
    /// The statistic the pass accumulates, if any.
    fn statistic(self) -> Option<ReduceOp> {
        match self {
            Pass::Max => Some(ReduceOp::Max),
            Pass::Sum => Some(ReduceOp::Sum),
            Pass::Out => None,
        }
    }

    /// Whether the packer writes a tile out per tile in: `s`, or the output.
    fn writes_tiles(self) -> bool {
        self != Pass::Sum
    }
}

/// The kernel's four semaphores: `unpacked` (T0 -> T1), `consumed` (T1 -> T0, the inputs read),
/// `computed` (T1 -> T2) and `free` (T2 -> T1, the packer has read what T1 wrote).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Semaphores {
    pub unpacked: Semaphore,
    pub consumed: Semaphore,
    pub computed: Semaphore,
    pub free: Semaphore,
}

/// Where a job's chunks live: each half's `x` (or `s`), mask and output slots, `kq` of each, and
/// one slot each for the accumulator carried between chunks and the two statistics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub kq: usize,
    pub x: [u64; 2],
    pub mask: [u64; 2],
    pub out: [u64; 2],
    pub acc: u64,
    pub max: u64,
    pub sum: u64,
    pub sems: Semaphores,
    pub init: Vec<SemaphoreInit>,
}

/// Plan a job's slots for chunks of `kq` tiles in the current arena (`crate::l1::arena`).
pub fn plan(kq: usize) -> std::result::Result<Layout, PlanError> {
    let mut req = Requirements::new(1);
    let align = tt_isa::dram::ALIGN;
    let run = kq as u64 * TILE_SLOT;
    let x = [
        req.scratch("softmax x, half 0", run, align, 0..1),
        req.scratch("softmax x, half 1", run, align, 0..1),
    ];
    let mask = [
        req.scratch("softmax mask, half 0", run, align, 0..1),
        req.scratch("softmax mask, half 1", run, align, 0..1),
    ];
    let out = [
        req.scratch("softmax out, half 0", run, align, 0..1),
        req.scratch("softmax out, half 1", run, align, 0..1),
    ];
    let acc = req.scratch("softmax accumulator", TILE_SLOT, align, 0..1);
    let max = req.scratch("softmax maximum", TILE_SLOT, align, 0..1);
    let sum = req.scratch("softmax sum", TILE_SLOT, align, 0..1);
    let unpacked = req.semaphore("softmax unpacked", 0, 0..1);
    let consumed = req.semaphore("softmax consumed", 1, 0..1);
    let computed = req.semaphore("softmax computed", 0, 0..1);
    let free = req.semaphore("softmax free", 1, 0..1);
    let plan = req.plan(crate::l1::arena())?;
    Ok(Layout {
        kq,
        x: x.map(|b| plan.addr(b)),
        mask: mask.map(|b| plan.addr(b)),
        out: out.map(|b| plan.addr(b)),
        acc: plan.addr(acc),
        max: plan.addr(max),
        sum: plan.addr(sum),
        sems: Semaphores {
            unpacked: plan.semaphore(unpacked),
            consumed: plan.semaphore(consumed),
            computed: plan.semaphore(computed),
            free: plan.semaphore(free),
        },
        init: plan.semaphore_init(),
    })
}

/// The largest chunk (at most [`MAX_KQ`], at most the row's `ct` tiles) whose layout fits the
/// current arena.
fn layout_for(ct: usize) -> Result<Layout> {
    (1..=MAX_KQ.min(ct))
        .rev()
        .find_map(|kq| plan(kq).ok())
        .ok_or_else(|| shape_error("not even a one-tile chunk fits the data arena"))
}

fn shape_error(what: impl std::fmt::Display) -> TensorError {
    TensorError::Shape(format!("softmax rows: {what}"))
}

/// GPRs the roles keep their addresses in, as `sfpu::kernel`'s looped roles do (clear of
/// `datapath::SCRATCH_GPR`, the matmul's 24..28, `local_movement`'s 12..16 and the guard's
/// 61..64).
const X_GPR: u32 = 32;
const MASK_GPR: u32 = 33;
const UNPACK_STRIDE_GPR: u32 = 36;
const OUT_GPR: u32 = 37;
const PACK_STRIDE_GPR: u32 = 39;

/// The role programs of one chunk: `pass` over `tiles` tiles staged in half `h`; `first` the
/// pass's first chunk (no accumulator to carry in), `last` its last (the statistic folded and
/// kept). Each role's per-tile block is stored once and repeated (`crate::code::Loop`), so a
/// program does not grow with the chunk.
pub fn roles(
    l: &Layout,
    pass: Pass,
    h: usize,
    tiles: usize,
    [first, last]: [bool; 2],
    scale: f32,
) -> LoopedRoles {
    assert!((1..=l.kq).contains(&tiles) && h < 2);
    let s = l.sems;
    let stride = tile_base_units(TILE_SLOT);
    let back = |units: u32| units - stride;
    let set =
        |p: &mut Vec<Instruction>, gpr: u32, v: u32| p.extend(backend::set_gpr(gpr, v).unwrap());
    let repeat = |loops: &mut Vec<Loop>, start: usize, end: usize, count: usize| {
        if count > 1 {
            loops.push(Loop {
                start: start as u32,
                len: (end - start) as u32,
                count: count as u32,
            });
        }
    };

    // What each kernel unpacks once, first: the statistics the pass reads, and the accumulator
    // a later chunk continues from.
    let mut once: Vec<(u64, u32)> = match pass {
        Pass::Max => vec![],
        Pass::Sum => vec![(l.max, MAX_ROW)],
        Pass::Out => vec![(l.max, MAX_ROW), (l.sum, SUM_ROW)],
    };
    let stats: Vec<u32> = once.iter().map(|&(_, r)| r).collect();
    if pass.statistic().is_some() && !first {
        once.push((l.acc, OUT_ROW));
    }

    let mut unpack = thread_config();
    unpack.extend(clear_unpacker0_adcs());
    let mut words = ConfigWords::new();
    tile_unpack_config(&mut words, l.x[h]);
    unpack.extend(config_program(&words));
    for &(at, row) in &once {
        unpack.extend(unpack_tile_to_dst(at, row));
    }
    if !once.is_empty() {
        unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
    }
    set(&mut unpack, UNPACK_STRIDE_GPR, stride);
    set(&mut unpack, X_GPR, back(tile_base_units(l.x[h])));
    set(&mut unpack, MASK_GPR, back(tile_base_units(l.mask[h])));
    let mut unpack_loops = Vec::new();
    let block = unpack.len();
    unpack.extend(sync::take(s.consumed, Before::UNPACKER));
    let step = Some(UNPACK_STRIDE_GPR);
    unpack.extend(unpack_datums_to_dst_gpr(X_GPR, step, TILE_DATUMS, A_ROW));
    if pass == Pass::Max {
        unpack.extend(unpack_datums_to_dst_gpr(MASK_GPR, step, TILE_DATUMS, B_ROW));
    }
    unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
    repeat(&mut unpack_loops, block, unpack.len(), tiles);
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());

    let mut m = vec![state_id()];
    let mut math_loops = Vec::new();
    if !once.is_empty() {
        m.extend(sync::take(s.unpacked, Before::SFPU));
        m.extend(math::statistics(&stats));
    }
    let code = |first_tile: bool| -> Code {
        match pass {
            Pass::Max => math::scale_mask_max(scale, first_tile),
            Pass::Sum => math::sub_exp_sum(first_tile),
            Pass::Out => math::sub_exp_div(),
        }
    };
    let tile = |m: &mut Vec<Instruction>, loops: &mut Vec<Loop>, code: &Code| {
        m.extend(sync::take(s.unpacked, Before::SFPU));
        if pass.writes_tiles() {
            m.extend(sync::take(s.free, Before::SFPU));
        }
        let at = m.len() as u32;
        m.extend_from_slice(&code.ins);
        loops.extend(code.loops.iter().map(|l| Loop {
            start: l.start + at,
            ..*l
        }));
        m.extend(sync::post_after(Unit::Sfpu, s.consumed));
        if pass.writes_tiles() {
            m.extend(sync::post_after(Unit::Sfpu, s.computed));
        }
    };
    // A statistic's first tile is copied, the rest combined in.
    let copies = first && pass.statistic().is_some();
    if copies {
        tile(&mut m, &mut math_loops, &code(true));
    }
    let rest = tiles - usize::from(copies);
    if rest > 0 {
        let block = m.len();
        let mut inner = Vec::new();
        tile(&mut m, &mut inner, &code(false));
        repeat(&mut math_loops, block, m.len(), rest);
        math_loops.extend(inner);
    }
    if let Some(op) = pass.statistic() {
        if last {
            m.extend(math::finish(op));
        }
        m.extend(sync::take(s.free, Before::SFPU));
        m.extend(sync::post_after(Unit::Sfpu, s.computed));
    }

    let mut pack = vec![state_id()];
    let mut pack_loops = Vec::new();
    if pass.writes_tiles() {
        set(&mut pack, PACK_STRIDE_GPR, stride);
        set(
            &mut pack,
            OUT_GPR,
            back(pack_dest_units(l.out[h] + TILE_DATA)),
        );
        let block = pack.len();
        pack.extend(sync::take(s.computed, Before::PACKER));
        let row = if pass == Pass::Max { C_ROW } else { OUT_ROW };
        pack.extend(pack_tile_from_dst_gpr(OUT_GPR, Some(PACK_STRIDE_GPR), row));
        pack.extend(sync::post_after(Unit::Packer, s.free));
        repeat(&mut pack_loops, block, pack.len(), tiles);
    }
    if let Some(op) = pass.statistic() {
        let to = match (last, op) {
            (false, _) => l.acc,
            (true, ReduceOp::Max) => l.max,
            (true, _) => l.sum,
        };
        pack.extend(sync::take(s.computed, Before::PACKER));
        pack.extend(pack_tile_from_dst(to + TILE_DATA, OUT_ROW));
        pack.extend(sync::post_after(Unit::Packer, s.free));
    }
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());

    let code = [unpack, m, pack];
    let loops = [unpack_loops, math_loops, pack_loops];
    for t in 0..3 {
        if let Err(e) = crate::code::check(&loops[t], code[t].len() as u32) {
            panic!("softmax roles' loops, role {t}: {e}");
        }
    }
    (code, loops)
}

/// A chunk's role programs and their block repeats, as `Step::Kernel` holds them.
type Kernel = (Arc<[Vec<Instruction>; 3]>, Arc<[Vec<Loop>; 3]>);

/// [`roles`], memoised: they name only L1 addresses, so every job of every op with the same chunk
/// shares them.
#[allow(clippy::too_many_arguments)]
fn kernel(l: &Layout, pass: Pass, h: usize, tiles: usize, ends: [bool; 2], scale: f32) -> Kernel {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = (usize, Pass, usize, usize, [bool; 2], u32, u64);
    static MEMO: OnceLock<Mutex<HashMap<Key, Kernel>>> = OnceLock::new();
    let key = (
        l.kq,
        pass,
        h,
        tiles,
        ends,
        scale.to_bits(),
        crate::l1::arena().end,
    );
    let memo = MEMO.get_or_init(Default::default);
    if let Some(k) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return k.clone();
    }
    let (code, loops) = roles(l, pass, h, tiles, ends, scale);
    let built = (Arc::new(code), Arc::new(loops));
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key, built.clone());
    built
}

/// Whether [`softmax_rows`] takes `x` (`[rows, cols]`) with `mask` (`[mask_rows, cols]`, repeated
/// down `x` every `mask_rows` rows): F32 scores and a `Bool` mask, both whole tiles, the mask's
/// rows dividing `x`'s.
pub fn fits(x: &DramTensor, mask: &DramTensor) -> bool {
    x.elem == Elem::F32
        && mask.elem == Elem::Bool
        && x.rows % 32 == 0
        && x.cols % 32 == 0
        && x.rows > 0
        && x.cols > 0
        && mask.cols == x.cols
        && mask.rows % 32 == 0
        && mask.rows > 0
        && x.rows % mask.rows == 0
}

/// `softmax(scale x, masked)` over each row of `x`: `mask` (`true` where a position is not
/// attended) is `[mask_rows, cols]`, read for rows `r % mask_rows` of `x` (a GQA group's or a
/// chunk's causal block). The output is `[rows, cols]` F32, bit for bit the composition's. One job
/// a tile row; refused unless [`fits`].
pub fn softmax_rows(
    alloc: &mut DramAlloc,
    x: &DramTensor,
    mask: &DramTensor,
    scale: f32,
) -> Result<Work> {
    if !fits(x, mask) {
        return Err(shape_error(format!(
            "x {:?} [{}, {}] with mask {:?} [{}, {}]: F32 scores and a Bool mask, whole tiles, \
             the mask's rows dividing the scores'",
            x.elem, x.rows, x.cols, mask.elem, mask.rows, mask.cols
        )));
    }
    let [rt, ct] = x.grid();
    let mrt = mask.rows / 32;
    let l = layout_for(ct)?;
    let out = DramTensor::alloc_elem(alloc, x.rows, x.cols, Elem::F32)?;
    let (xr, mr, or) = (x.tensor_ref(), mask.tensor_ref(), out.tensor_ref());
    let read = |first: usize, count: usize, at: u64, t: &tt_isa::dm::record::TensorRef| {
        [
            [
                record::READ_RUN,
                first as u32,
                count as u32,
                at as u32,
                0,
                ct as u32,
                0,
                0,
            ],
            t.encode()[0],
            t.encode()[1],
        ]
    };
    let chunks = ct.div_ceil(l.kq);
    let mut jobs = Vec::with_capacity(rt);
    for i in 0..rt {
        let mut steps = Vec::with_capacity(9 * chunks);
        // Each pass starts in the half the last ended in: a new packet (module documentation).
        let mut h = 0;
        for pass in [Pass::Max, Pass::Sum, Pass::Out] {
            for c in 0..chunks {
                if c > 0 {
                    h ^= 1;
                }
                let j0 = c * l.kq;
                let n = l.kq.min(ct - j0);
                let mut gather = Vec::new();
                match pass {
                    Pass::Max => {
                        gather.extend(read(i * ct + j0, n, l.x[h], &xr));
                        gather.extend(read((i % mrt) * ct + j0, n, l.mask[h], &mr));
                    }
                    Pass::Sum | Pass::Out => gather.extend(read(i * ct + j0, n, l.x[h], &or)),
                }
                let scatter = if pass.writes_tiles() {
                    vec![
                        [
                            record::WRITE_RUN,
                            (i * ct + j0) as u32,
                            n as u32,
                            l.out[h] as u32,
                            0,
                            0,
                            0,
                            0,
                        ],
                        or.encode()[0],
                        or.encode()[1],
                    ]
                } else {
                    Vec::new()
                };
                let (roles, loops) = kernel(&l, pass, h, n, [c == 0, c + 1 == chunks], scale);
                steps.extend([
                    Step::List {
                        what: "softmax rows gather",
                        entries: gather,
                    },
                    Step::Kernel {
                        roles,
                        init: l.init.clone(),
                        mop: Box::new([None; 3]),
                        loops,
                        half: Some(h as u8),
                    },
                    Step::List {
                        what: "softmax rows scatter",
                        entries: scatter,
                    },
                ]);
            }
        }
        jobs.push(steps);
    }
    Ok(Work { out, jobs })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunk_of_the_largest_size_fits_the_arena() {
        let l = plan(MAX_KQ).expect("the largest chunk fits");
        assert_eq!(l.kq, MAX_KQ);
    }

    /// Each role's program is the same length whatever the chunk's tiles (the per-tile block is
    /// repeated), and every combination of pass and ends builds with valid loops.
    #[test]
    fn programs_do_not_grow_with_the_chunk() {
        let l = plan(MAX_KQ).unwrap();
        for pass in [Pass::Max, Pass::Sum, Pass::Out] {
            for ends in [[true, false], [false, false], [false, true], [true, true]] {
                let lens = |n: usize| roles(&l, pass, 1, n, ends, 0.125).0.map(|r| r.len());
                assert_eq!(lens(3), lens(MAX_KQ), "{pass:?} {ends:?}");
            }
        }
    }
}
