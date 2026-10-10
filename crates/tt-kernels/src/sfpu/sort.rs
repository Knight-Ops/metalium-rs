//! Device sort along an axis, stable and deterministic (hardware-coverage
//! close-out lane T5).
//!
//! # Contract
//!
//! The order is a total order on `(key, original index)` pairs. For `F32` the
//! key order is `f32::total_cmp` (`-NaN < -inf < .. < -0 < +0 < .. < +inf <
//! +NaN`, payloads ordered by bits); for `I32` it is two's-complement order. A
//! descending sort reverses the key order and keeps **ties in ascending
//! original index**: it is a stable sort on the reversed key, which is also
//! what Burn's `total_cmp`-reversed comparator asks for, minus the instability
//! Burn's `sort_unstable_by` allows itself.
//!
//! # Why the hardware order needs no key transform
//!
//! `SFPGT`/`SFPLE`/`SFPSWAP` compare 32-bit words as sign-magnitude integers
//! (`SignMagIsSmaller`: for a negative word the low 31 bits are inverted and
//! the result compared as a signed integer). That is *exactly* the orderable
//! key `k = x ^ ((x >> 31) as u32 >> 1)` compared as a signed integer, so an
//! `F32` word is compared by `total_cmp` as it is. An `I32` word `x` is
//! converted once to `s = x ^ ((x >> 31) as u32 >> 1)` (the same map: an
//! involution) so that the hardware's order on `s` is the integer order on `x`,
//! and converted back by the last kernel that touches it.
//!
//! # Layout: planes
//!
//! The axis is laid along *vector slots*, never along lanes, so no
//! compare-exchange ever crosses a lane. An `SFPLOAD` at `Dst` address
//! `tile + 2 * s` moves slot `s` (`face = s / 8`, row group `(s / 2) % 4`, odd
//! columns `s % 2`): 32 lanes, lane `8 j + c` being row `4 g + j` of the face
//! and column `2 c + (s % 2)`. [`plane_coord`] places element `(q, e)` -- problem
//! `q` (one lane of one tile column block), axis position `e` -- at the
//! matrix coordinate whose tile row is `e / 32` and tile column block
//! `q / 32`. A tile therefore holds 32 problems by 32 positions, the same
//! density as a `[rows, 32]` matrix, and every problem of a tile is sorted at
//! once by one program.
//!
//! # Network and padding
//!
//! A bitonic network of `N = 32 * T` wires, `T` a power of two of tiles, every
//! wire a pair of vectors (key and index). Wires `e >= n` carry sentinels that
//! sort last (`+NaN` with all payload bits for ascending, `-NaN` for
//! descending; I32 the same patterns after the conversion) *and* an index
//! `e >= n`, so even a real element equal to the sentinel sorts before it:
//! sentinels are never returned. The exchange compares `(key, index)`
//! lexicographically, so no two wires tie and the network's result is the
//! unique stable order.
//!
//! # Passes
//!
//! A kernel holds at most four tiles in `Dst`, and a program slot is 8192
//! words, so the network runs as [`Pass`]es over GDDR tiles, in [`schedule`]'s
//! order, each pass one gather/kernel/scatter triple of one job (one tile
//! column block's problems; blocks run in parallel on separate units):
//!
//! - `Local`: sort a tile's 32 positions (the network's merges up to size 32),
//!   generating its indices;
//! - then for every merge size `32 kt` (`kt = 2, 4, ..`): `Cross` -- the
//!   stages whose partner is another tile (`p ^ j`, `j >= 32`: the same slot of
//!   tile `t ^ j/32`) -- and `Merge` -- the in-tile stages `j = 16..1`. The
//!   direction of a stage is bit `log2(k)` of the wire index, which for `k >=
//!   64` is a bit of the *tile* index: uniform over a tile, so a pass is one
//!   program with one direction.
//!
//! Axis lengths up to [`MAX_AXIS`] are sorted; beyond it is refused.

use std::sync::{Arc, Mutex, OnceLock};

use tt_device::Transport;
use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::dm::{record, TILE_SLOT};
use tt_isa::isa::generated::encode;
use tt_isa::isa::Instruction;
use tt_isa::sync::{self, Unit};

use super::{Cond, Format, LReg, Program};
use crate::datapath::{
    clear_unpacker0_adcs, config_program, pack_tile_from_dst, state_id, thread_config,
    tile_unpack_config, unpack_tile_to_dst,
};
use crate::l1::Requirements;
use crate::session::Session;
use crate::tensor::{DramTensor, Elem, Pad, Step, TensorError};

/// Slots (vectors) in one tile: the axis positions a tile holds.
pub const SLOTS: usize = 32;

/// `Dst` rows of one tile.
const TILE_ROWS: u32 = 64;

/// The longest axis the device sorts: 32 tiles of positions, an explicit
/// bound on the schedule's length (each merge level adds passes) rather than
/// on memory.
pub const MAX_AXIS: usize = 1024;

/// The longest axis the device sorts (the same with or without indices).
pub const fn max_axis(_indices: bool) -> usize {
    MAX_AXIS
}

/// One sort's shape. Everything here is in the programs, so it is the memo key.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Spec {
    /// `F32` (`total_cmp`) or `I32`.
    pub elem: Elem,
    /// Axis length, `1..=MAX_AXIS`.
    pub n: usize,
    /// Largest first; ties still ascending in original index.
    pub descending: bool,
    /// Also produce the original indices (and break ties by them).
    pub indices: bool,
}

/// Why a sort cannot run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SortError {
    Elem(Elem),
    Empty,
    TooLong {
        n: usize,
        max: usize,
    },
    /// A kernel's program does not fit its slot: a bug in a bound.
    Program {
        kernel: Kernel,
        words: usize,
    },
}

impl std::fmt::Display for SortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SortError::Elem(e) => write!(f, "sort of {e:?} elements (F32 and I32 only)"),
            SortError::Empty => write!(f, "sort of an empty axis"),
            SortError::TooLong { n, max } => {
                write!(
                    f,
                    "axis length {n} exceeds the device sort's limit of {max}"
                )
            }
            SortError::Program { kernel, words } => {
                write!(
                    f,
                    "the {kernel:?} sort program is {words} words, past its slot"
                )
            }
        }
    }
}

impl std::error::Error for SortError {}

impl From<SortError> for TensorError {
    fn from(e: SortError) -> Self {
        TensorError::Shape(e.to_string())
    }
}

impl Spec {
    /// Tiles of axis positions: `n` padded to a power of two of tiles.
    pub fn tiles(&self) -> usize {
        self.n.div_ceil(SLOTS).next_power_of_two()
    }

    /// Wires of the network.
    pub fn wires(&self) -> usize {
        self.tiles() * SLOTS
    }

    pub fn check(&self) -> Result<(), SortError> {
        if !matches!(self.elem, Elem::F32 | Elem::I32) {
            return Err(SortError::Elem(self.elem));
        }
        if self.n == 0 {
            return Err(SortError::Empty);
        }
        if self.n > MAX_AXIS {
            return Err(SortError::TooLong {
                n: self.n,
                max: MAX_AXIS,
            });
        }
        Ok(())
    }

    /// The pattern that sorts after everything (in the hardware order, after
    /// the key conversion of an I32).
    pub fn sentinel(&self) -> u32 {
        if self.descending {
            0xffff_ffff
        } else {
            0x7fff_ffff
        }
    }
}

/// The matrix coordinate `[row, col]` of problem `q`'s axis position `e` in
/// the planes layout (module documentation).
pub fn plane_coord(q: usize, e: usize) -> [usize; 2] {
    let (tr, s) = (e / SLOTS, e % SLOTS);
    let (cb, lane) = (q / 32, q % 32);
    let (face, group, odd) = (s / 8, (s / 2) % 4, s % 2);
    let (j, c) = (lane / 8, lane % 8);
    [
        32 * tr + 16 * (face / 2) + 4 * group + j,
        32 * cb + 16 * (face % 2) + 2 * c + odd,
    ]
}

/// The `(q, e)` that [`plane_coord`] puts at `[row, col]`.
pub fn plane_element([row, col]: [usize; 2]) -> (usize, usize) {
    let (tr, r) = (row / 32, row % 32);
    let (cb, c) = (col / 32, col % 32);
    let face = 2 * (r / 16) + c / 16;
    let (group, j) = ((r % 16) / 4, r % 4);
    let (cc, odd) = ((c % 16) / 2, c % 2);
    (
        32 * cb + 8 * j + cc,
        SLOTS * tr + 8 * face + 2 * group + odd,
    )
}

/// Planes matrix `[rows, cols]` holding `problems` problems of `n` positions.
pub fn plane_dims(problems: usize, n: usize) -> [usize; 2] {
    [
        32 * n.div_ceil(SLOTS).next_power_of_two(),
        32 * problems.div_ceil(32),
    ]
}

const KA: LReg = LReg::L0;
const KB: LReg = LReg::L1;
const IA: LReg = LReg::L2;
const IB: LReg = LReg::L3;
const MASK: LReg = LReg::L4;

/// One step of the schedule, on tiles of one tile column block.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Pass {
    /// Sort tile `tile`'s positions (ascending if its index is even, so the
    /// next merge finds a bitonic pair), creating its indices.
    Local { tile: usize, last: bool },
    /// Exchange slot for slot between tile `lo` and tile `hi`, `lo` taking
    /// the first in sorted order if `ascending`, the last otherwise.
    Cross {
        lo: usize,
        hi: usize,
        ascending: bool,
    },
    /// The in-tile merge stages of tile `tile`.
    Merge {
        tile: usize,
        ascending: bool,
        last: bool,
    },
}

/// The kernel a pass runs. Direction and indices are in the program;
/// positions are not, except a `Local` tile's (its indices and padding).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    Local { tile: usize, last: bool },
    Cross { ascending: bool },
    Merge { ascending: bool, last: bool },
}

impl Pass {
    pub fn kernel(self) -> Kernel {
        match self {
            Pass::Local { tile, last } => Kernel::Local { tile, last },
            Pass::Cross { ascending, .. } => Kernel::Cross { ascending },
            Pass::Merge {
                ascending, last, ..
            } => Kernel::Merge { ascending, last },
        }
    }
}

/// The passes, in an order every dependence respects.
pub fn schedule(spec: &Spec) -> Vec<Pass> {
    let tiles = spec.tiles();
    let mut passes: Vec<Pass> = (0..tiles)
        .map(|tile| Pass::Local {
            tile,
            last: tiles == 1,
        })
        .collect();
    let mut kt = 2;
    while kt <= tiles {
        let mut jt = kt / 2;
        while jt >= 1 {
            for lo in 0..tiles {
                let hi = lo ^ jt;
                if hi > lo {
                    passes.push(Pass::Cross {
                        lo,
                        hi,
                        ascending: lo & kt == 0,
                    });
                }
            }
            jt /= 2;
        }
        for tile in 0..tiles {
            passes.push(Pass::Merge {
                tile,
                ascending: tile & kt == 0,
                last: kt == tiles,
            });
        }
        kt *= 2;
    }
    passes
}

/// `(tiles in, tiles out)` of a kernel.
pub fn tiles_io(spec: &Spec, kernel: Kernel) -> (usize, usize) {
    let (keys, out_keys) = match kernel {
        Kernel::Local { .. } | Kernel::Merge { .. } => (1, 1),
        Kernel::Cross { .. } => (2, 2),
    };
    let indices = if spec.indices { keys } else { 0 };
    // A local sort reads keys only: its indices are made, not read.
    let read = match kernel {
        Kernel::Local { .. } => keys,
        _ => keys + indices,
    };
    (read, out_keys + indices)
}

/// `Dst` row of slot `s` of the tile at `tile` rows (in tiles) from row 0.
fn row(tile: usize, s: usize) -> u32 {
    TILE_ROWS * tile as u32 + 2 * s as u32
}

/// Unconditionally exchange two registers' lanes (`SFPSWAP_MOD1_SWAP`).
fn swap(p: &mut Program, a: LReg, b: LReg) {
    p.raw(encode::sfpswap(b.index(), a.index(), 0).unwrap());
}

/// Where a compare-exchange's four vectors are in `Dst`.
struct Wires {
    ka: u32,
    kb: u32,
    ia: u32,
    ib: u32,
}

/// Compare-exchange wires `a` and `b`: afterwards `a` holds the one that
/// comes first if `ascending`, last otherwise, in the order of `(key, index)`.
fn exchange(p: &mut Program, spec: &Spec, w: &Wires, ascending: bool, tie_break: bool) {
    p.load(KA, Format::Int32, w.ka);
    p.load(KB, Format::Int32, w.kb);
    if spec.indices {
        p.load(IA, Format::Int32, w.ia);
        p.load(IB, Format::Int32, w.ib);
    }
    // Swap exactly when `first` comes before `second` in the sorted order.
    let ((fk, fi), (sk, si)) = if ascending {
        ((KB, IB), (KA, IA))
    } else {
        ((KA, IA), (KB, IB))
    };
    let before = |x: LReg, y: LReg| {
        if spec.descending {
            Cond::Less(y, x)
        } else {
            Cond::Less(x, y)
        }
    };
    if spec.indices && tie_break {
        // Equal keys: order by index, indices only.
        p.if_(Cond::LessEq(KA, KB), |p| {
            p.if_(Cond::LessEq(KB, KA), |p| {
                p.if_(Cond::Less(fi, si), |p| swap(p, IA, IB));
            });
        });
    }
    p.if_(before(fk, sk), |p| {
        swap(p, KA, KB);
        if spec.indices {
            swap(p, IA, IB);
        }
    });
    p.store(KA, Format::Int32, w.ka);
    p.store(KB, Format::Int32, w.kb);
    if spec.indices {
        p.store(IA, Format::Int32, w.ia);
        p.store(IB, Format::Int32, w.ib);
    }
}

/// The wire pairs of the bitonic network over `wires` wires in program order:
/// `(a, b, ascending)` -- the whole network.
pub fn network(wires: usize) -> Vec<(usize, usize, bool)> {
    assert!(wires.is_power_of_two());
    let mut out = Vec::new();
    let mut k = 2;
    while k <= wires {
        let mut j = k / 2;
        while j >= 1 {
            for a in 0..wires {
                let b = a ^ j;
                if b > a {
                    out.push((a, b, a & k == 0));
                }
            }
            j /= 2;
        }
        k *= 2;
    }
    out
}

/// A deliberate defect for a gate's negative control: the sort must be
/// caught being wrong, so the oracle's sensitivity is itself tested.
#[cfg(test)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Mutation {
    /// The kernel's network without its `n`th wire pair.
    DropExchange(usize),
    /// Equal keys are not ordered by original index.
    NoTieBreak,
}

/// A kernel's program. The tile layout in `Dst`: key tiles first, then index
/// tiles; `Local` and `Merge` have one of each, `Cross` two (lower tile's
/// first).
pub fn program(spec: Spec, kernel: Kernel) -> Result<Vec<Instruction>, SortError> {
    build(
        spec,
        kernel,
        #[cfg(test)]
        None,
    )
}

fn build(
    spec: Spec,
    kernel: Kernel,
    #[cfg(test)] mutation: Option<Mutation>,
) -> Result<Vec<Instruction>, SortError> {
    spec.check()?;
    let mut p = Program::new();
    let convert = spec.elem == Elem::I32;
    let (keys, indexed) = match kernel {
        Kernel::Cross { .. } => (2, spec.indices),
        _ => (1, spec.indices),
    };
    // Index tiles follow the key tiles.
    let key = |tile: usize, s: usize| row(tile, s);
    let index = |tile: usize, s: usize| row(keys + tile, s);
    let to_hardware_order = |p: &mut Program, rows: &mut dyn Iterator<Item = u32>| {
        for r in rows {
            p.load(KA, Format::Int32, r);
            p.if_(Cond::Lt0(KA), |p| p.xor(MASK, KA));
            p.store(KA, Format::Int32, r);
        }
    };
    #[allow(unused_mut)]
    let mut tie_break = true;
    #[cfg(test)]
    if mutation == Some(Mutation::NoTieBreak) {
        tie_break = false;
    }
    let mut pairs: Vec<(usize, usize, bool)> = Vec::new();
    match kernel {
        Kernel::Local { tile, last } => {
            let base = tile * SLOTS;
            let valid = spec.n.saturating_sub(base).min(SLOTS);
            if convert {
                p.loadi_bits(MASK, 0x7fff_ffff);
                to_hardware_order(&mut p, &mut (0..valid).map(|s| key(0, s)));
            }
            if valid < SLOTS {
                p.loadi_bits(KA, spec.sentinel());
                for s in valid..SLOTS {
                    p.store(KA, Format::Int32, key(0, s));
                }
            }
            if indexed {
                for s in 0..SLOTS {
                    p.loadi_bits(IA, (base + s) as u32);
                    p.store(IA, Format::Int32, index(0, s));
                }
            }
            // The merges up to size 32; in a longer network the last of them
            // runs backwards on odd tiles, so the next level finds a bitonic
            // sequence.
            let flip = spec.tiles() > 1 && tile % 2 == 1;
            let mut k = 2;
            while k <= SLOTS {
                let mut j = k / 2;
                while j >= 1 {
                    for a in 0..SLOTS {
                        let b = a ^ j;
                        if b > a {
                            pairs.push((a, b, (a & k == 0) != (flip && k == SLOTS)));
                        }
                    }
                    j /= 2;
                }
                k *= 2;
            }
            #[cfg(test)]
            if let Some(Mutation::DropExchange(i)) = mutation {
                if i < pairs.len() {
                    pairs.remove(i);
                }
            }
            for &(a, b, ascending) in &pairs {
                let w = Wires {
                    ka: key(0, a),
                    kb: key(0, b),
                    ia: index(0, a),
                    ib: index(0, b),
                };
                exchange(&mut p, &spec, &w, ascending, tie_break);
            }
            if convert && last {
                p.loadi_bits(MASK, 0x7fff_ffff);
                to_hardware_order(&mut p, &mut (0..SLOTS).map(|s| key(0, s)));
            }
        }
        Kernel::Cross { ascending } => {
            #[allow(unused_mut)]
            let mut all: Vec<usize> = (0..SLOTS).collect();
            #[cfg(test)]
            if let Some(Mutation::DropExchange(i)) = mutation {
                if i < all.len() {
                    all.remove(i);
                }
            }
            for s in all {
                let w = Wires {
                    ka: key(0, s),
                    kb: key(1, s),
                    ia: index(0, s),
                    ib: index(1, s),
                };
                exchange(&mut p, &spec, &w, ascending, tie_break);
            }
        }
        Kernel::Merge { ascending, last } => {
            let mut j = SLOTS / 2;
            while j >= 1 {
                for a in 0..SLOTS {
                    let b = a ^ j;
                    if b > a {
                        pairs.push((a, b, ascending));
                    }
                }
                j /= 2;
            }
            #[cfg(test)]
            if let Some(Mutation::DropExchange(i)) = mutation {
                if i < pairs.len() {
                    pairs.remove(i);
                }
            }
            for &(a, b, asc) in &pairs {
                let w = Wires {
                    ka: key(0, a),
                    kb: key(0, b),
                    ia: index(0, a),
                    ib: index(0, b),
                };
                exchange(&mut p, &spec, &w, asc, tie_break);
            }
            if convert && last {
                p.loadi_bits(MASK, 0x7fff_ffff);
                to_hardware_order(&mut p, &mut (0..SLOTS).map(|s| key(0, s)));
            }
        }
    }
    let ins = p.finish();
    // The role program wraps this in a handful of synchronisation words.
    let max = tt_isa::mailbox::PROGRAM_MAX as usize - 64;
    if ins.len() > max {
        return Err(SortError::Program {
            kernel,
            words: ins.len(),
        });
    }
    Ok(ins)
}

/// Where a sort kernel's slots are in L1, and its semaphores.
#[derive(Clone, Debug)]
pub struct Layout {
    pub in_at: u64,
    pub out_at: u64,
    pub sems: super::kernel::SfpuSemaphores,
    pub init: Vec<crate::runtime::SemaphoreInit>,
}

/// Slots for `inputs` tiles in and `outputs` tiles out, and the three
/// semaphores in the order every SFPU kernel declares them.
pub fn plan_layout(inputs: usize, outputs: usize) -> Result<Layout, crate::l1::PlanError> {
    let mut req = Requirements::new(1);
    let align = tt_isa::dram::ALIGN;
    let a = req.scratch("sort input slots", inputs as u64 * TILE_SLOT, align, 0..1);
    let out = req.scratch("sort output slots", outputs as u64 * TILE_SLOT, align, 0..1);
    let unpacked = req.semaphore("sfpu unpacked", 0, 0..1);
    let free = req.semaphore("sfpu Dst free", 1, 0..1);
    let computed = req.semaphore("sfpu computed", 0, 0..1);
    let plan = req.plan(tt_isa::l1::DATA)?;
    Ok(Layout {
        in_at: plan.addr(a),
        out_at: plan.addr(out),
        sems: super::kernel::SfpuSemaphores {
            unpacked: plan.semaphore(unpacked),
            computed: plan.semaphore(computed),
            free: plan.semaphore(free),
        },
        init: plan.semaphore_init(),
    })
}

/// The three role programs: T0 unpacks `inputs` tiles to `Dst` tiles `0..`,
/// T1 runs `math`, T2 packs `Dst` tiles `0..outputs` to the output slots.
pub fn roles(
    layout: &Layout,
    inputs: usize,
    outputs: usize,
    math: &[Instruction],
) -> [Vec<Instruction>; 3] {
    let s = layout.sems;
    let slot = |base: u64, n: usize| base + n as u64 * TILE_SLOT;
    let mut unpack = thread_config();
    unpack.extend(clear_unpacker0_adcs());
    let mut words = ConfigWords::new();
    tile_unpack_config(&mut words, layout.in_at);
    unpack.extend(config_program(&words));
    let mut m = vec![state_id()];
    let mut pack = vec![state_id()];
    unpack.extend(sync::take(s.free, Before::UNPACKER));
    for t in 0..inputs {
        unpack.extend(unpack_tile_to_dst(
            slot(layout.in_at, t),
            TILE_ROWS * t as u32,
        ));
    }
    unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
    m.extend(sync::take(s.unpacked, Before::SFPU));
    m.extend_from_slice(math);
    m.extend(sync::post_after(Unit::Sfpu, s.computed));
    pack.extend(sync::take(s.computed, Before::PACKER));
    for t in 0..outputs {
        pack.extend(pack_tile_from_dst(
            slot(layout.out_at, t) + tt_isa::dm::TILE_DATA,
            TILE_ROWS * t as u32,
        ));
    }
    pack.extend(sync::post_after(Unit::Packer, s.free));
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    [unpack, m, pack]
}

type Programs = (Layout, Arc<[Vec<Instruction>; 3]>);

/// A kernel's layout and role programs, memoised.
fn programs(spec: Spec, kernel: Kernel) -> Result<Programs, TensorError> {
    type Memo = Mutex<std::collections::HashMap<(Spec, Kernel), Programs>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let memo = MEMO.get_or_init(Default::default);
    if let Some(p) = memo
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(&(spec, kernel))
    {
        return Ok(p.clone());
    }
    let math = program(spec, kernel)?;
    let (inputs, outputs) = tiles_io(&spec, kernel);
    let layout = plan_layout(inputs, outputs).map_err(|e| TensorError::Shape(e.to_string()))?;
    let roles = Arc::new(roles(&layout, inputs, outputs, &math));
    let p = (layout, roles);
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert((spec, kernel), p.clone());
    Ok(p)
}

/// What a sort produced.
pub struct Sorted {
    /// The sorted keys, the planes' shape and element type.
    pub keys: DramTensor,
    /// The original indices as `I32`, if asked.
    pub indices: Option<DramTensor>,
}

impl<T: Transport> Session<T> {
    /// Sort every problem of `planes` (`plane_dims`, built by the caller with
    /// [`plane_coord`]) along its axis positions on the SFPU.
    ///
    /// Only `spec.n` positions of each problem are meaningful; the padding
    /// positions and lanes are read and ignored. The outputs hold garbage there.
    pub fn sort_planes(&mut self, planes: &DramTensor, spec: Spec) -> Result<Sorted, TensorError> {
        spec.check()?;
        planes.expect("a sort", spec.elem)?;
        let tiles = spec.tiles();
        if planes.rows != 32 * tiles || planes.cols % 32 != 0 {
            return Err(TensorError::Shape(format!(
                "sort planes [{}, {}] for {} axis positions (want [{}, a multiple of 32])",
                planes.rows,
                planes.cols,
                spec.n,
                32 * tiles
            )));
        }
        let schedule = schedule(&spec);
        let mut kernels = std::collections::HashMap::new();
        for pass in &schedule {
            let k = pass.kernel();
            if let std::collections::hash_map::Entry::Vacant(v) = kernels.entry(k) {
                v.insert(programs(spec, k)?);
            }
        }
        let [_, ct] = planes.grid();
        let alloc = self.dram_alloc()?;
        let keys = DramTensor::alloc_elem(alloc, planes.rows, planes.cols, spec.elem)?;
        let indices = if spec.indices {
            match DramTensor::alloc_elem(alloc, planes.rows, planes.cols, Elem::I32) {
                Ok(t) => Some(t),
                Err(e) => {
                    alloc.free(&keys.placement);
                    return Err(e);
                }
            }
        } else {
            None
        };
        let run = |tensor: &DramTensor, tile: usize, at: u64, write: bool| {
            let r = tensor.tensor_ref().encode();
            [
                [
                    if write {
                        record::WRITE_RUN
                    } else {
                        record::READ_RUN
                    },
                    tile as u32,
                    1,
                    at as u32,
                    0,
                    0,
                    0,
                    0,
                ],
                r[0],
                r[1],
            ]
        };
        let mut jobs = Vec::new();
        for cb in 0..ct {
            let tile_of = |t: usize| t * ct + cb;
            let mut steps = Vec::new();
            for (n, pass) in schedule.iter().enumerate() {
                let (layout, roles) = &kernels[&pass.kernel()];
                let slot_in = |i: usize| layout.in_at + i as u64 * TILE_SLOT;
                let slot_out = |i: usize| layout.out_at + i as u64 * TILE_SLOT;
                let mut gather = Vec::new();
                let mut scatter = Vec::new();
                // The tiles a pass reads and writes: keys, then indices; the
                // first pass reads the planes, every later one the results.
                match *pass {
                    Pass::Local { tile, .. } => {
                        gather.extend(run(planes, tile_of(tile), slot_in(0), false));
                        scatter.extend(run(&keys, tile_of(tile), slot_out(0), true));
                        if let Some(i) = &indices {
                            scatter.extend(run(i, tile_of(tile), slot_out(1), true));
                        }
                    }
                    Pass::Merge { tile, .. } => {
                        gather.extend(run(&keys, tile_of(tile), slot_in(0), false));
                        scatter.extend(run(&keys, tile_of(tile), slot_out(0), true));
                        if let Some(i) = &indices {
                            gather.extend(run(i, tile_of(tile), slot_in(1), false));
                            scatter.extend(run(i, tile_of(tile), slot_out(1), true));
                        }
                    }
                    Pass::Cross { lo, hi, .. } => {
                        for (i, t) in [lo, hi].into_iter().enumerate() {
                            gather.extend(run(&keys, tile_of(t), slot_in(i), false));
                            scatter.extend(run(&keys, tile_of(t), slot_out(i), true));
                        }
                        if let Some(x) = &indices {
                            for (i, t) in [lo, hi].into_iter().enumerate() {
                                gather.extend(run(x, tile_of(t), slot_in(2 + i), false));
                                scatter.extend(run(x, tile_of(t), slot_out(2 + i), true));
                            }
                        }
                    }
                }
                if n > 0 {
                    // Every pass after the first reads what an earlier one
                    // wrote to GDDR: wait for those writes.
                    gather.insert(0, [tt_isa::dm::op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
                }
                steps.push(Step::List {
                    what: "sort gather",
                    entries: gather,
                });
                steps.push(Step::Kernel {
                    roles: roles.clone(),
                    init: layout.init.clone(),
                    mop: Box::new([None; 3]),
                    loops: Default::default(),
                    half: None,
                });
                steps.push(Step::List {
                    what: "sort scatter",
                    entries: scatter,
                });
            }
            jobs.push(steps);
        }
        keys.set_pad(Pad::Undefined);
        if let Some(i) = &indices {
            i.set_pad(Pad::Undefined);
        }
        if let Err(e) = self.submit_jobs(jobs, crate::session::RESET_BUDGET) {
            let alloc = self.dram_alloc()?;
            alloc.free(&keys.placement);
            if let Some(i) = &indices {
                alloc.free(&i.placement);
            }
            return Err(e);
        }
        Ok(Sorted { keys, indices })
    }
}

/// The independent host oracle for one problem: the stable order of
/// `(key, original index)`. `bits` are the elements as stored; returns the
/// sorted element bits and the original indices.
pub fn reference(elem: Elem, bits: &[u32], descending: bool) -> (Vec<u32>, Vec<u32>) {
    let key = |x: u32| -> i64 {
        match elem {
            Elem::I32 => x as i32 as i64,
            // The orderable integer of `total_cmp`.
            _ => (x ^ (((x as i32) >> 31) as u32 >> 1)) as i32 as i64,
        }
    };
    let mut order: Vec<usize> = (0..bits.len()).collect();
    order.sort_by(|&a, &b| {
        let by_key = key(bits[a]).cmp(&key(bits[b]));
        let by_key = if descending { by_key.reverse() } else { by_key };
        by_key.then(a.cmp(&b))
    });
    (
        order.iter().map(|&i| bits[i]).collect(),
        order.iter().map(|&i| i as u32).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfpu::interp::Vector;

    /// Awkward words of both element types: signed zeros, subnormals, NaNs
    /// of both signs and payloads, infinities, extremes, duplicates.
    fn specials() -> Vec<u32> {
        vec![
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x007f_ffff,
            0x807f_ffff,
            0x0080_0000,
            0x8080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0000,
            0x7f80_0001,
            0xff80_0001,
            0x7fff_ffff,
            0xffff_ffff,
            0x7fc1_2345,
            0xffc5_4321,
            0x0000_0002,
            0xffff_fffe,
            0x8000_0000,
            0x3f80_0000,
            0x0000_0000,
            0x7fff_ffff,
            0xffff_ffff,
            0x4000_0000,
            0xc000_0000,
            0x0000_0001,
        ]
    }

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 32) as u32
    }

    /// One problem per lane, `n` positions each.
    fn columns(spec: &Spec, seed: u64) -> Vec<Vec<u32>> {
        let mut s = seed;
        let specials = specials();
        (0..32)
            .map(|lane| {
                (0..spec.n)
                    .map(|e| match lane % 4 {
                        // Dense specials, with repeats.
                        0 => specials[(e * 7 + lane) % specials.len()],
                        // Few distinct values: ties everywhere.
                        1 => specials[(lcg(&mut s) % 4) as usize],
                        // Random words.
                        2 => lcg(&mut s),
                        _ => {
                            if spec.elem == Elem::I32 {
                                [i32::MIN as u32, 0, i32::MAX as u32, 1, -1i32 as u32]
                                    [lcg(&mut s) as usize % 5]
                            } else {
                                [f32::NAN.to_bits(), 0.0f32.to_bits(), (-0.0f32).to_bits()]
                                    [lcg(&mut s) as usize % 3]
                            }
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// GDDR as the schedule sees it: tiles of keys and of indices.
    struct Gddr {
        keys: Vec<Vec<u32>>,
        indices: Vec<Vec<u32>>,
    }

    /// The schedule on the interpreter, pass by pass, as a job runs it.
    fn simulate(
        spec: Spec,
        input: &[Vec<u32>],
        passes: &[Pass],
        mutate: impl Fn(usize, Kernel) -> Option<Mutation>,
    ) -> Gddr {
        let tiles = spec.tiles();
        let mut g = Gddr {
            keys: vec![vec![0xdead_beef; 1024]; tiles],
            indices: vec![vec![0xdead_beef; 1024]; tiles],
        };
        for (n, pass) in passes.iter().enumerate() {
            let kernel = pass.kernel();
            let (reads, _) = match *pass {
                Pass::Local { tile, .. } => (vec![(tile, input[tile].clone())], ()),
                _ => (vec![], ()),
            };
            let mut v = Vector::new();
            let put_keys: Vec<(usize, Vec<u32>)> = match *pass {
                Pass::Local { .. } => reads.iter().map(|(t, d)| (*t, d.clone())).collect(),
                Pass::Merge { tile, .. } => vec![(tile, g.keys[tile].clone())],
                Pass::Cross { lo, hi, .. } => {
                    vec![(lo, g.keys[lo].clone()), (hi, g.keys[hi].clone())]
                }
            };
            let nkeys = put_keys.len();
            for (i, (_, d)) in put_keys.iter().enumerate() {
                v.put_tile(64 * i, d);
            }
            if spec.indices && !matches!(pass, Pass::Local { .. }) {
                match *pass {
                    Pass::Merge { tile, .. } => v.put_tile(64 * nkeys, &g.indices[tile]),
                    Pass::Cross { lo, hi, .. } => {
                        v.put_tile(64 * nkeys, &g.indices[lo]);
                        v.put_tile(64 * (nkeys + 1), &g.indices[hi]);
                    }
                    Pass::Local { .. } => unreachable!(),
                }
            }
            let ins = match mutate(n, kernel) {
                None => program(spec, kernel).unwrap(),
                Some(m) => build(spec, kernel, Some(m)).unwrap(),
            };
            v.run(&ins).unwrap();
            let outs: Vec<usize> = match *pass {
                Pass::Local { tile, .. } | Pass::Merge { tile, .. } => vec![tile],
                Pass::Cross { lo, hi, .. } => vec![lo, hi],
            };
            for (i, &t) in outs.iter().enumerate() {
                g.keys[t] = v.tile(64 * i);
                if spec.indices {
                    g.indices[t] = v.tile(64 * (outs.len() + i));
                }
            }
        }
        g
    }

    /// The planes' tiles of `columns`, padding with junk.
    fn plane_tiles(spec: &Spec, cols: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut tiles = vec![vec![0xdead_beefu32; 1024]; spec.tiles()];
        for (lane, col) in cols.iter().enumerate() {
            for (e, &x) in col.iter().enumerate() {
                let [r, c] = plane_coord(lane, e);
                tiles[e / SLOTS][tt_isa::dm::face_index(r % 32, c % 32)] = x;
            }
        }
        tiles
    }

    fn check(spec: Spec, seed: u64, g: &Gddr, cols: &[Vec<u32>]) -> Result<(), String> {
        for (lane, col) in cols.iter().enumerate() {
            let (want_keys, want_idx) = reference(spec.elem, col, spec.descending);
            let at = |e: usize| {
                let [r, c] = plane_coord(lane, e);
                (e / SLOTS, tt_isa::dm::face_index(r % 32, c % 32))
            };
            let got_keys: Vec<u32> = (0..spec.n)
                .map(|e| {
                    let (t, i) = at(e);
                    g.keys[t][i]
                })
                .collect();
            if got_keys != want_keys {
                return Err(format!("{spec:?} seed {seed} lane {lane} keys"));
            }
            if spec.indices {
                let got_idx: Vec<u32> = (0..spec.n)
                    .map(|e| {
                        let (t, i) = at(e);
                        g.indices[t][i]
                    })
                    .collect();
                if got_idx != want_idx {
                    return Err(format!("{spec:?} seed {seed} lane {lane} indices"));
                }
            }
        }
        Ok(())
    }

    fn run_on_interpreter(spec: Spec, seed: u64) {
        run_schedule(spec, seed, &schedule(&spec), |_, _| None).unwrap();
    }

    fn run_schedule(
        spec: Spec,
        seed: u64,
        passes: &[Pass],
        mutate: impl Fn(usize, Kernel) -> Option<Mutation>,
    ) -> Result<(), String> {
        let cols = columns(&spec, seed);
        let input = plane_tiles(&spec, &cols);
        let g = simulate(spec, &input, passes, mutate);
        check(spec, seed, &g, &cols)
    }

    #[test]
    fn plane_coordinates_are_a_bijection() {
        let mut seen = std::collections::HashSet::new();
        for q in 0..64 {
            for e in 0..64 {
                let at = plane_coord(q, e);
                assert!(seen.insert(at), "({q}, {e}) collides");
                assert_eq!(plane_element(at), (q, e));
            }
        }
        assert_eq!(seen.len(), 64 * 64);
        assert_eq!(plane_dims(64, 64), [64, 64]);
    }

    /// The schedule is the bitonic network: on 0/1 inputs of every length up
    /// to four tiles it sorts, executing the passes' wire pairs by their
    /// definitions (the independent check of `schedule`'s directions).
    #[test]
    fn the_schedule_is_a_sorting_network() {
        let spec = |n| Spec {
            elem: Elem::F32,
            n,
            descending: false,
            indices: true,
        };
        for n in [33, 64, 100, 128] {
            let s = spec(n);
            let tiles = s.tiles();
            let mut seed = 7u64;
            for _ in 0..300 {
                // A random 0/1 word per wire.
                let mut w: Vec<u32> = (0..tiles * SLOTS).map(|_| lcg(&mut seed) & 1).collect();
                for pass in schedule(&s) {
                    match pass {
                        Pass::Local { tile, .. } => {
                            let flip = tiles > 1 && tile % 2 == 1;
                            let mut k = 2;
                            while k <= SLOTS {
                                let mut j = k / 2;
                                while j >= 1 {
                                    for a in 0..SLOTS {
                                        let b = a ^ j;
                                        if b > a {
                                            let asc = (a & k == 0) != (flip && k == SLOTS);
                                            let (x, y) = (tile * SLOTS + a, tile * SLOTS + b);
                                            if (w[x] > w[y]) == asc {
                                                w.swap(x, y);
                                            }
                                        }
                                    }
                                    j /= 2;
                                }
                                k *= 2;
                            }
                        }
                        Pass::Cross { lo, hi, ascending } => {
                            for s in 0..SLOTS {
                                let (x, y) = (lo * SLOTS + s, hi * SLOTS + s);
                                if (w[x] > w[y]) == ascending {
                                    w.swap(x, y);
                                }
                            }
                        }
                        Pass::Merge {
                            tile, ascending, ..
                        } => {
                            let mut j = SLOTS / 2;
                            while j >= 1 {
                                for a in 0..SLOTS {
                                    let b = a ^ j;
                                    if b > a {
                                        let (x, y) = (tile * SLOTS + a, tile * SLOTS + b);
                                        if (w[x] > w[y]) == ascending {
                                            w.swap(x, y);
                                        }
                                    }
                                }
                                j /= 2;
                            }
                        }
                    }
                }
                assert!(w.windows(2).all(|p| p[0] <= p[1]), "n={n}: {w:?}");
            }
        }
    }

    #[test]
    fn f32_with_indices_matches_the_oracle() {
        for descending in [false, true] {
            for n in [1, 2, 5, 17, 31, 32, 33, 64, 70] {
                run_on_interpreter(
                    Spec {
                        elem: Elem::F32,
                        n,
                        descending,
                        indices: true,
                    },
                    n as u64 + 100,
                );
            }
        }
    }

    #[test]
    fn i32_with_indices_matches_the_oracle() {
        for descending in [false, true] {
            for n in [1, 3, 24, 32, 50, 128] {
                run_on_interpreter(
                    Spec {
                        elem: Elem::I32,
                        n,
                        descending,
                        indices: true,
                    },
                    n as u64 + 200,
                );
            }
        }
    }

    #[test]
    fn values_only_matches_the_oracle() {
        for elem in [Elem::F32, Elem::I32] {
            for descending in [false, true] {
                for n in [1, 9, 32, 33, 50, 64, 129] {
                    run_on_interpreter(
                        Spec {
                            elem,
                            n,
                            descending,
                            indices: false,
                        },
                        n as u64 + 300,
                    );
                }
            }
        }
    }

    #[test]
    fn the_longest_axis_matches_the_oracle() {
        run_on_interpreter(
            Spec {
                elem: Elem::F32,
                n: MAX_AXIS,
                descending: true,
                indices: true,
            },
            1024,
        );
    }

    /// Negative controls, watched to fail: a pass missing from the schedule,
    /// a network missing an exchange, and one that does not break ties by
    /// index, each disagree with the oracle somewhere.
    #[test]
    fn a_missing_pass_exchange_or_tie_break_is_caught() {
        for descending in [false, true] {
            let spec = Spec {
                elem: Elem::F32,
                n: 100,
                descending,
                indices: true,
            };
            let all = schedule(&spec);
            // Drop each kind of pass, first and last of its kind.
            for kind in ["local", "cross", "merge"] {
                let of_kind: Vec<usize> = all
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        matches!(
                            (kind, p),
                            ("local", Pass::Local { .. })
                                | ("cross", Pass::Cross { .. })
                                | ("merge", Pass::Merge { .. })
                        )
                    })
                    .map(|(i, _)| i)
                    .collect();
                for &drop in [of_kind[0], of_kind[of_kind.len() - 1]].iter() {
                    let passes: Vec<Pass> = all
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != drop)
                        .map(|(_, p)| *p)
                        .collect();
                    let caught = (0..4)
                        .any(|seed| run_schedule(spec, 500 + seed, &passes, |_, _| None).is_err());
                    assert!(caught, "dropping the {kind} pass {drop} went unnoticed");
                }
            }
            // One exchange fewer in a kernel, first and last.
            let single = Spec { n: 32, ..spec };
            // The exchanges of a tile's network: 1 + 2 + 3 + 4 + 5 stages of 16.
            let count = 15 * 16;
            for drop in [0, 1, count / 2, count - 16, count - 1] {
                let caught = (0..16).any(|seed| {
                    run_schedule(single, 77 + seed, &schedule(&single), |_, _| {
                        Some(Mutation::DropExchange(drop))
                    })
                    .is_err()
                });
                assert!(
                    caught,
                    "dropping exchange {drop} went unnoticed ({descending})"
                );
            }
            let caught = (0..4).any(|seed| {
                run_schedule(single, 78 + seed, &schedule(&single), |_, _| {
                    Some(Mutation::NoTieBreak)
                })
                .is_err()
            });
            assert!(caught, "no tie break went unnoticed ({descending})");
        }
        // Control of the control: the unmutated schedule passes the same input.
        let spec = Spec {
            elem: Elem::F32,
            n: 32,
            descending: false,
            indices: true,
        };
        run_schedule(spec, 78, &schedule(&spec), |_, _| None).unwrap();
    }

    #[test]
    fn lengths_past_the_bound_are_refused_by_name() {
        let e = Spec {
            elem: Elem::F32,
            n: MAX_AXIS + 1,
            descending: false,
            indices: true,
        }
        .check()
        .unwrap_err();
        assert!(
            e.to_string().contains("1025") && e.to_string().contains("1024"),
            "{e}"
        );
        let e = Spec {
            elem: Elem::Bool,
            n: 4,
            descending: false,
            indices: false,
        }
        .check()
        .unwrap_err();
        assert!(e.to_string().contains("Bool"), "{e}");
    }

    #[test]
    fn every_program_fits_a_slot() {
        for indices in [true, false] {
            for elem in [Elem::F32, Elem::I32] {
                let spec = Spec {
                    elem,
                    n: MAX_AXIS,
                    descending: true,
                    indices,
                };
                for kernel in [
                    Kernel::Local {
                        tile: 31,
                        last: true,
                    },
                    Kernel::Cross { ascending: false },
                    Kernel::Merge {
                        ascending: false,
                        last: true,
                    },
                ] {
                    let p = program(spec, kernel).unwrap();
                    eprintln!("{elem:?} indices={indices} {kernel:?}: {} words", p.len());
                }
            }
        }
    }
}
