//! Tensors that live in GDDR, and the matmul that reads and writes them there.
//!
//! Phase 9's point: a tensor is uploaded once, stays on the card, and ops read
//! and write it there, so what crosses PCIe per op is descriptors, not data.
//!
//! A [`DramTensor`] is a row-major `[rows, cols]` FP32 matrix stored as 32x32
//! tiles, one `tt_isa::dm::TILE_SLOT` each (zero-padded at the ragged edges,
//! which is the identity for every accumulation), interleaved across the chip's
//! usable channels: tile `t` of the row-major tile grid lives on channel
//! `channels[t % n]`, slot `t / n` of the tensor's region there. Spreading tiles
//! over channels is what lets later work read them in parallel; for now it also
//! means the allocator's regions stay small.

use std::collections::BTreeMap;

use std::sync::Arc;

use tt_device::{Device, Transport, TransportError, Window};
use tt_isa::dm::record::{self, TensorRef};
use tt_isa::dm::{TILE_DATA, TILE_SLOT};
use tt_isa::dram::{Dram, DramChannel, DramRange, CHANNEL_BYTES};
use tt_isa::isa::Instruction;
use tt_isa::tile::L1Format;

use crate::dm::DmError;
use crate::matmul::{self, Fidelity, SrcRoute, Staging};
use crate::runtime::RunError;

/// Why a DRAM tensor operation failed.
#[derive(Debug)]
pub enum TensorError {
    Transport(TransportError),
    Mover(DmError),
    Run(RunError),
    /// No channel has room for the tensor's share of slots.
    OutOfMemory {
        slots: u64,
    },
    /// Shapes that do not compose, or an op this path does not support.
    Shape(String),
}

impl From<TransportError> for TensorError {
    fn from(e: TransportError) -> Self {
        TensorError::Transport(e)
    }
}
impl From<DmError> for TensorError {
    fn from(e: DmError) -> Self {
        TensorError::Mover(e)
    }
}
impl From<RunError> for TensorError {
    fn from(e: RunError) -> Self {
        TensorError::Run(e)
    }
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::Transport(e) => write!(f, "{e}"),
            TensorError::Mover(e) => write!(f, "{e}"),
            TensorError::Run(e) => write!(f, "{e}"),
            TensorError::OutOfMemory { slots } => {
                write!(f, "no room in GDDR for {slots} tile slots per channel")
            }
            TensorError::Shape(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for TensorError {}

pub type Result<T> = std::result::Result<T, TensorError>;

/// Where one tensor's tiles are: a region of `slots` slots at `base[i]` on
/// each of `channels[i]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    channels: Vec<DramChannel>,
    base: Vec<u64>,
    slots: u64,
    tiles: usize,
    /// A view's first tile within the allocation it borrows from.
    first: usize,
    /// Did this placement allocate its slots? A view ([`DramTensor::rows_view`])
    /// did not, and freeing it gives nothing back.
    owned: bool,
}

impl Placement {
    /// Tile `t`'s slot.
    pub fn slot(&self, t: usize) -> DramRange {
        assert!(t < self.tiles, "tile {t} of {}", self.tiles);
        let t = t + self.first;
        let n = self.channels.len();
        let (c, i) = (t % n, (t / n) as u64);
        self.channels[c]
            .range(self.base[c] + i * TILE_SLOT, TILE_SLOT)
            .expect("the allocator placed every slot inside its channel")
    }

    pub fn tiles(&self) -> usize {
        self.tiles
    }
}

/// A first-fit allocator of slot regions, one free list per channel.
///
/// Every tensor takes the same number of slots on every usable channel, so the
/// free lists stay in step; each is still kept separately, since nothing
/// guarantees it.
pub struct DramAlloc {
    channels: Vec<DramChannel>,
    /// Per channel: free regions as `offset -> bytes`, coalesced on free.
    free: Vec<BTreeMap<u64, u64>>,
}

impl DramAlloc {
    pub fn new(dram: &Dram) -> Self {
        let channels: Vec<DramChannel> = dram.channels().collect();
        // Whole slots only, so every region stays slot-aligned.
        let usable = CHANNEL_BYTES / TILE_SLOT * TILE_SLOT;
        let free = channels
            .iter()
            .map(|_| BTreeMap::from([(0, usable)]))
            .collect();
        DramAlloc { channels, free }
    }

    /// Room for `tiles` tile slots, interleaved.
    pub fn alloc(&mut self, tiles: usize) -> Result<Placement> {
        let n = self.channels.len();
        let slots = (tiles.max(1)).div_ceil(n) as u64;
        let bytes = slots * TILE_SLOT;
        let mut base = Vec::with_capacity(n);
        for i in 0..n {
            let Some((&at, &len)) = self.free[i].iter().find(|(_, &len)| len >= bytes) else {
                // Give back what this call already took.
                for (j, &b) in base.iter().enumerate() {
                    release(&mut self.free[j], b, bytes);
                }
                return Err(TensorError::OutOfMemory { slots });
            };
            self.free[i].remove(&at);
            if len > bytes {
                self.free[i].insert(at + bytes, len - bytes);
            }
            base.push(at);
        }
        Ok(Placement {
            channels: self.channels.clone(),
            base,
            slots,
            tiles,
            first: 0,
            owned: true,
        })
    }

    /// Give `p`'s slots back; nothing, for a view.
    pub fn free(&mut self, p: &Placement) {
        if !p.owned {
            return;
        }
        for (i, &b) in p.base.iter().enumerate() {
            release(&mut self.free[i], b, p.slots * TILE_SLOT);
        }
    }

    /// Free bytes on the fullest channel.
    pub fn free_bytes(&self) -> u64 {
        self.free
            .iter()
            .map(|f| f.values().sum::<u64>())
            .min()
            .unwrap_or(0)
    }
}

fn release(free: &mut BTreeMap<u64, u64>, at: u64, len: u64) {
    let (mut at, mut len) = (at, len);
    if let Some((&p, &plen)) = free.range(..at).next_back() {
        if p + plen == at {
            free.remove(&p);
            at = p;
            len += plen;
        }
    }
    if let Some(&nlen) = free.get(&(at + len)) {
        free.remove(&(at + len));
        len += nlen;
    }
    free.insert(at, len);
}

/// An FP32 `[rows, cols]` matrix in GDDR, tiled. See the module documentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DramTensor {
    pub rows: usize,
    pub cols: usize,
    pub placement: Placement,
}

impl DramTensor {
    /// Tile rows and columns.
    pub fn grid(&self) -> [usize; 2] {
        [self.rows.div_ceil(32), self.cols.div_ceil(32)]
    }

    /// This tensor as an op record names it (`tt_isa::dm::record::TensorRef`).
    pub fn tensor_ref(&self) -> TensorRef {
        let p = &self.placement;
        let mut r = TensorRef {
            n: p.channels.len() as u8,
            first: p.first as u32,
            ct: self.grid()[1] as u32,
            ..TensorRef::default()
        };
        for (i, (c, b)) in p.channels.iter().zip(&p.base).enumerate() {
            r.channels[i] = c.index();
            // Inside a channel, which `CHANNEL_BYTES` keeps under 4 GiB.
            r.base[i] = *b as u32;
        }
        r
    }

    /// The slot of tile `(i, j)`.
    pub fn tile(&self, i: usize, j: usize) -> DramRange {
        let [_, ct] = self.grid();
        self.placement.slot(i * ct + j)
    }

    /// Allocate a `[rows, cols]` tensor, contents undefined.
    pub fn alloc(alloc: &mut DramAlloc, rows: usize, cols: usize) -> Result<Self> {
        let tiles = rows.div_ceil(32).max(1) * cols.div_ceil(32).max(1);
        Ok(DramTensor {
            rows,
            cols,
            placement: alloc.alloc(tiles)?,
        })
    }

    /// Upload row-major `values`: tilized on the host once, then one bulk
    /// write per channel.
    pub fn upload<T: Transport>(
        dev: &mut Device<T>,
        w: &Window,
        alloc: &mut DramAlloc,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if values.len() != rows * cols {
            return Err(TensorError::Shape(format!(
                "{} values for a [{rows}, {cols}] tensor",
                values.len()
            )));
        }
        let t = Self::alloc(alloc, rows, cols)?;
        let (images, _) = matmul::tilize_f32(values, rows.max(1), cols.max(1), L1Format::Fp32);
        let img = matmul::TILE_IMAGE_BYTES;
        let per = t.placement.channels.len();
        let mut regions = vec![vec![0u8; (t.placement.slots * TILE_SLOT) as usize]; per];
        for (tile, image) in images.chunks_exact(img).enumerate() {
            let (c, i) = (tile % per, tile / per);
            let at = i * TILE_SLOT as usize;
            regions[c][at..at + img].copy_from_slice(image);
        }
        // Only the slots tiles occupy: a one-tile tensor has one slot's worth
        // on one channel, not a slot on every channel.
        let tiles = images.len() / img;
        for (c, bytes) in regions.iter().enumerate() {
            let used = tiles / per + usize::from(c < tiles % per);
            if used == 0 {
                continue;
            }
            let bytes = &bytes[..used * TILE_SLOT as usize];
            let r = t.placement.channels[c]
                .range(t.placement.base[c], bytes.len() as u64)
                .expect("allocated");
            dev.dram_write(w, r, bytes)?;
        }
        Ok(t)
    }

    /// Rows `[first_row, first_row + rows)` of this tensor, all columns, as a
    /// view of the same slots: nothing is copied. `first_row` must start a tile
    /// row, and the view must end at one or at this tensor's last row. Freeing
    /// the view frees nothing; the caller keeps this tensor alive meanwhile.
    pub fn rows_view(&self, first_row: usize, rows: usize) -> Result<Self> {
        let [_, ct] = self.grid();
        let end = first_row + rows;
        if first_row % 32 != 0
            || (end % 32 != 0 && end != self.rows)
            || end > self.rows
            || rows == 0
        {
            return Err(TensorError::Shape(format!(
                "rows {first_row}..{end} of {} are not whole tile rows",
                self.rows
            )));
        }
        Ok(DramTensor {
            rows,
            cols: self.cols,
            placement: Placement {
                tiles: rows.div_ceil(32) * ct,
                first: self.placement.first + first_row / 32 * ct,
                owned: false,
                ..self.placement.clone()
            },
        })
    }

    /// Download to row-major values: one bulk read per channel, or, for a
    /// view or a small tensor, only what its data occupies.
    pub fn download<T: Transport>(&self, dev: &mut Device<T>, w: &Window) -> Result<Vec<f32>> {
        // A small tensor, or a view: tile by tile, and of each tile only the
        // faces and face rows its data reaches -- a `[1, n]` row is 64 bytes
        // from each of two faces, not 4 KiB per tile.
        if self.placement.first != 0 || self.placement.tiles < 2 * self.placement.channels.len() {
            let [rt, ct] = self.grid();
            let mut packed = vec![0u8; self.placement.tiles * 4096];
            for i in 0..rt {
                let rows = (self.rows - 32 * i).min(32);
                for j in 0..ct {
                    let cols = (self.cols - 32 * j).min(32);
                    let s = self.tile(i, j);
                    let tile = &mut packed[(i * ct + j) * 4096..][..4096];
                    for (fr, fc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        if 16 * fr >= rows || 16 * fc >= cols {
                            continue;
                        }
                        let face_rows = (rows - 16 * fr).min(16);
                        let at = ((fr * 2 + fc) * 256 * 4) as u64;
                        let len = face_rows * 16 * 4;
                        let r = s
                            .channel()
                            .range(s.offset() + TILE_DATA + at, len as u64)
                            .expect("in the slot");
                        dev.dram_read(w, r, &mut tile[at as usize..at as usize + len])?;
                    }
                }
            }
            return Ok(matmul::detilize_packed(&packed, self.rows, self.cols));
        }
        let per = self.placement.channels.len();
        let mut regions = Vec::with_capacity(per);
        for c in 0..per {
            let r = self.placement.channels[c]
                .range(self.placement.base[c], self.placement.slots * TILE_SLOT)
                .expect("allocated");
            let mut bytes = vec![0u8; r.len() as usize];
            dev.dram_read(w, r, &mut bytes)?;
            regions.push(bytes);
        }
        let mut packed = Vec::with_capacity(self.placement.tiles * 4096);
        for tile in 0..self.placement.tiles {
            let (c, i) = (tile % per, tile / per);
            let at = i * TILE_SLOT as usize + TILE_DATA as usize;
            packed.extend_from_slice(&regions[c][at..at + 4096]);
        }
        Ok(matmul::detilize_packed(&packed, self.rows, self.cols))
    }
}

/// One step of a [`Job`], run on one tile.
#[derive(Clone, Debug)]
pub enum Step {
    /// A data-mover list on the tile's RISCV B: entries (`tt_isa::dm::Entry`)
    /// and op records (`tt_isa::dm::record`), which the mover expands. Any
    /// length: the session splits it into lists the mover takes, never inside
    /// a record.
    List {
        /// What it is, for [`stats`].
        what: &'static str,
        entries: Vec<[u32; 8]>,
    },
    /// The three role programs of a matmul chunk, run concurrently on the
    /// tile's resident roles with [`matmul::TILE_SEMAPHORES`], which they
    /// restore.
    Matmul(Arc<[Vec<Instruction>; 3]>),
}

/// Steps that must run in order on one tile, from one L1 staging area. The
/// jobs of one op are independent of each other: each reads only GDDR and
/// writes only its own output tiles, so they may run on different tiles, in
/// any order, and give the same bits.
pub type Job = Vec<Step>;

/// What an op leaves to run: its output, already allocated, and its jobs.
pub struct Work {
    pub out: DramTensor,
    pub jobs: Vec<Job>,
}

/// `0..len` in contiguous runs: one per unit while there is an item for each,
/// more if a run would exceed `max` items, and as even as integer division
/// allows (no two runs differ by more than one item).
pub fn runs(len: usize, units: usize, max: usize) -> Vec<std::ops::Range<usize>> {
    let parts = units.max(1).min(len).max(len.div_ceil(max.max(1)));
    (0..parts)
        .map(|p| p * len / parts..(p + 1) * len / parts)
        .collect()
}

/// `[mc, nc]` output tiles per block, shrunk from the one-tile plan's until
/// there are at least `units` blocks or no output tile is left to split.
/// Shrinking only ever makes a block smaller, so it still fits wherever the
/// plan's did, and `K` is never touched: each output tile is the same
/// accumulation, in the same order, whichever block it falls in.
pub fn blocks([mt, nt]: [usize; 2], [mc, nc]: [usize; 2], units: usize) -> [usize; 2] {
    let (mut pm, mut pn) = (mt.div_ceil(mc), nt.div_ceil(nc));
    let (mut mc, mut nc) = (mc, nc);
    while mt.div_ceil(mc) * nt.div_ceil(nc) < units && (mc > 1 || nc > 1) {
        // Split the wider side of the block, so blocks stay square-ish and
        // each tile's gather reuses as much of its operands as it can.
        if nc >= mc && nc > 1 {
            pn += 1;
            nc = nt.div_ceil(pn);
        } else {
            pm += 1;
            mc = mt.div_ceil(pm);
        }
    }
    [mc, nc]
}

/// `op(A) @ op(B)`, where `op` is a transpose when asked, all in GDDR: the
/// data mover gathers each block's tiles into L1 (transposing where needed),
/// the resident roles compute it, and the mover writes the output tiles back.
/// Nothing but descriptors, programs and semaphores crosses PCIe.
///
/// Chunked by [`matmul::plan_in`] with [`Staging::Slots`], and refused if the
/// plan splits `K`: a split `K` means a partial-sum add, which the host path
/// does on the host, and doing it anywhere else would change the rounding.
/// The plan's blocks are then made small enough for `units` tiles to share
/// ([`blocks`]); each block is one [`Job`].
#[allow(clippy::too_many_arguments)]
pub fn matmul_dram(
    alloc: &mut DramAlloc,
    a: &DramTensor,
    a_transposed: bool,
    b: &DramTensor,
    b_transposed: bool,
    route: SrcRoute,
    fidelity: Fidelity,
    units: usize,
) -> Result<Work> {
    let (m, ka) = if a_transposed {
        (a.cols, a.rows)
    } else {
        (a.rows, a.cols)
    };
    let (kb, n) = if b_transposed {
        (b.cols, b.rows)
    } else {
        (b.rows, b.cols)
    };
    if ka != kb {
        return Err(TensorError::Shape(format!(
            "[{m}, {ka}] @ [{kb}, {n}]: inner dimensions differ"
        )));
    }
    let k = ka;
    let (in_fmt, out_fmt) = route.formats();
    let shape = matmul::plan_in([m, k, n], route, fidelity, Staging::Slots)
        .ok_or_else(|| TensorError::Shape(format!("[{m}, {k}] @ [{k}, {n}] fits no chunk")))?;
    let [mc, kc, nc] = shape.tiles;
    let [mt, kt, nt] = [m.div_ceil(32), k.div_ceil(32), n.div_ceil(32)];
    if kc < kt {
        return Err(TensorError::Shape(format!(
            "[{m}, {k}] @ [{k}, {n}] would split K; not on this path"
        )));
    }
    let [mc, nc] = blocks([mt, nt], [mc, nc], units);
    let (ra, rb) = (a.tensor_ref(), b.tensor_ref());
    let c = DramTensor::alloc(alloc, m, n)?;
    let rc = c.tensor_ref();
    let mut jobs = Vec::new();
    for i0 in (0..mt).step_by(mc) {
        let rows = mc.min(mt - i0);
        for j0 in (0..nt).step_by(nc) {
            let cols = nc.min(nt - j0);
            let tiles = [rows, kt, cols];
            let (b_at, outputs) = match matmul::plan_layout_in(tiles, in_fmt, Staging::Slots) {
                Ok(l) => l,
                Err(e) => {
                    alloc.free(&c.placement);
                    return Err(e.into());
                }
            };
            let flags = u32::from(a_transposed) | u32::from(b_transposed) << 1 | (kt as u32) << 8;
            let gather = [
                [
                    record::GATHER,
                    flags,
                    matmul::MATMUL_STAGE as u32,
                    b_at as u32,
                    i0 as u32,
                    rows as u32,
                    j0 as u32,
                    cols as u32,
                ],
                ra.encode()[0],
                ra.encode()[1],
                rb.encode()[0],
                rb.encode()[1],
            ];
            let roles = matmul::programs((tiles, Staging::Slots), route, fidelity, || {
                matmul::matmul_roles(&outputs, in_fmt, out_fmt, fidelity)
            });
            // Only the datums go back: the packer writes nothing else, and the
            // unpacker skips the header whatever it holds (`step18_dram_matmul`).
            // The outputs sit one slot apart from the first (`plan_layout_in`).
            let out_at = outputs[0].out;
            debug_assert!(outputs
                .iter()
                .enumerate()
                .all(|(k, o)| o.out == out_at + k as u64 * TILE_SLOT));
            let scatter = [
                [
                    record::SCATTER,
                    out_at as u32,
                    TILE_SLOT as u32,
                    i0 as u32,
                    rows as u32,
                    j0 as u32,
                    cols as u32,
                    0,
                ],
                rc.encode()[0],
                rc.encode()[1],
            ];
            jobs.push(vec![
                Step::List {
                    what: "matmul gather",
                    entries: gather.to_vec(),
                },
                Step::Matmul(roles),
                Step::List {
                    what: "matmul scatter",
                    entries: scatter.to_vec(),
                },
            ]);
        }
    }
    Ok(Work { out: c, jobs })
}

/// `slots` tile slots of scratch for the mover's own use, planned in the data
/// arena (`crate::l1`): where an element-wise run or a column sum stages its
/// tiles.
fn staging(name: &'static str, slots: usize) -> Result<u64> {
    let mut req = crate::l1::Requirements::new(1);
    let b = req.scratch(name, slots as u64 * TILE_SLOT, tt_isa::dram::ALIGN, 0..1);
    let plan = req
        .plan(tt_isa::l1::DATA)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    Ok(plan.addr(b))
}

/// What [`eltwise`] computes: a `tt_isa::dm::kind`, its scalar where it takes
/// one, and whether `b` is a single row broadcast down `a`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Eltwise {
    pub kind: u32,
    pub scalar: f32,
}

/// Element-wise `a (op) b` -- or `a (op) scalar` -- with everything in GDDR,
/// computed tile by tile by the data mover's FP32 unit in L1
/// (`tt_isa::dm::kind`). For `ADD_ROW`, `b` is `[1, cols]` and its row is
/// added to every row of `a`; otherwise `b`, where the kind takes one, has
/// `a`'s shape.
///
/// Tiles are independent, so they are dealt out in contiguous runs, one
/// [`Job`] each, as many runs as there are `units` (but never more tiles per
/// list than the staging area holds).
pub fn eltwise(
    alloc: &mut DramAlloc,
    op: Eltwise,
    a: &DramTensor,
    b: Option<&DramTensor>,
    units: usize,
) -> Result<Work> {
    use tt_isa::dm::kind;
    let binary = !matches!(op.kind, kind::MUL_SCALAR | kind::RELU);
    let row = op.kind == kind::ADD_ROW;
    match (binary, b) {
        (false, _) => {}
        (true, None) => return Err(TensorError::Shape("a binary op needs two operands".into())),
        (true, Some(b)) if row && (b.rows != 1 || b.cols != a.cols) => {
            return Err(TensorError::Shape(format!(
                "[{}, {}] + row [{}, {}]",
                a.rows, a.cols, b.rows, b.cols
            )))
        }
        (true, Some(b)) if !row && (b.rows, b.cols) != (a.rows, a.cols) => {
            return Err(TensorError::Shape(format!(
                "[{}, {}] and [{}, {}] differ",
                a.rows, a.cols, b.rows, b.cols
            )))
        }
        _ => {}
    }
    // Two slots per tile of a run in flight, as one buffer the mover owns.
    const GROUP: usize = 96;
    let stage = staging("eltwise slots", 2 * GROUP)?;
    let out = DramTensor::alloc(alloc, a.rows, a.cols)?;
    let [rt, ct] = a.grid();
    let (ra, ro) = (a.tensor_ref(), out.tensor_ref());
    let rb = b
        .filter(|_| binary)
        .map_or(TensorRef::default(), DramTensor::tensor_ref);
    let flags = u32::from(binary) | u32::from(row) << 1;
    let mut jobs = Vec::new();
    for run in runs(rt * ct, units, GROUP) {
        let record = [
            [
                record::ELTWISE,
                op.kind,
                op.scalar.to_bits(),
                run.start as u32,
                run.len() as u32,
                flags,
                stage as u32,
                0,
            ],
            ra.encode()[0],
            ra.encode()[1],
            rb.encode()[0],
            rb.encode()[1],
            ro.encode()[0],
            ro.encode()[1],
        ];
        jobs.push(vec![Step::List {
            what: "eltwise list",
            entries: record.to_vec(),
        }]);
    }
    Ok(Work { out, jobs })
}

/// The sum over rows of `a`, as a `[1, cols]` tensor, in `burn-flex`'s order:
/// from `+0.0`, adding rows in order (`tt_isa::dm::kind::COL_SUM`).
///
/// Columns are independent; each keeps its rows in order on one tile. They
/// are dealt out in contiguous runs, one [`Job`] per run, as many as `units`.
pub fn sum_rows(alloc: &mut DramAlloc, a: &DramTensor, units: usize) -> Result<Work> {
    // The schedule -- an accumulator per column, slots for its rows, where a
    // list's worth of slots runs out -- is `tt_isa::dm::record::SUM`'s.
    let stage = staging("column-sum slots", record::SUM_SLOTS)?;
    let out = DramTensor::alloc(alloc, 1, a.cols)?;
    let [rt, ct] = a.grid();
    let (ra, ro) = (a.tensor_ref(), out.tensor_ref());
    let mut jobs = Vec::new();
    for columns in runs(ct, units, ct.max(1)) {
        let record = [
            [
                record::SUM,
                columns.start as u32,
                columns.len() as u32,
                rt as u32,
                stage as u32,
                0,
                0,
                0,
            ],
            ra.encode()[0],
            ra.encode()[1],
            ro.encode()[0],
            ro.encode()[1],
        ];
        jobs.push(vec![Step::List {
            what: "sum list",
            entries: record.to_vec(),
        }]);
    }
    Ok(Work { out, jobs })
}

/// Host wall-clock time in each stage of the DRAM ops, process-wide: where a
/// device-resident op's time goes. Read with [`stats::take`].
pub mod stats {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    static TIMES: Mutex<BTreeMap<&'static str, (u64, Duration)>> = Mutex::new(BTreeMap::new());

    pub(crate) fn timed<R>(what: &'static str, f: impl FnOnce() -> R) -> R {
        let t0 = Instant::now();
        let r = f();
        let d = t0.elapsed();
        let mut t = TIMES.lock().unwrap_or_else(|p| p.into_inner());
        let e = t.entry(what).or_default();
        e.0 += 1;
        e.1 += d;
        r
    }

    /// Every stage's calls and time since the last call, and reset.
    pub fn take() -> Vec<(&'static str, u64, Duration)> {
        let mut t = TIMES.lock().unwrap_or_else(|p| p.into_inner());
        std::mem::take(&mut *t)
            .into_iter()
            .map(|(k, (n, d))| (k, n, d))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{blocks, reference, runs, DramAlloc, DramTensor, Eltwise, Job, Step};
    use tt_isa::dm::{kind, op, record};
    use tt_isa::dram::Dram;

    /// One job as the mover runs it: every entry in order, records expanded,
    /// a `WAIT` between what were separate lists, and each kernel as its
    /// programs' first words (`None` for a list entry).
    fn stream(job: &Job) -> Vec<Result<[u32; 8], Vec<u32>>> {
        let mut out = Vec::new();
        let mut after_list = false;
        for step in job {
            match step {
                Step::List { entries, .. } => {
                    if after_list {
                        out.push(Ok([op::WAIT, 0, 0, 0, 0, 0, 0, 0]));
                    }
                    let mut i = 0;
                    while i < entries.len() {
                        let n = record::len(entries[i][0]);
                        if n == 1 {
                            out.push(Ok(entries[i]));
                        } else {
                            record::expand(&entries[i..i + n], |e| {
                                out.push(Ok(e));
                                Ok(())
                            })
                            .unwrap_or_else(|c| panic!("record refused: code {c}"));
                        }
                        i += n;
                    }
                    after_list = true;
                }
                Step::Matmul(roles) => {
                    out.push(Err(roles.iter().map(|p| p.len() as u32).collect()));
                    after_list = false;
                }
            }
        }
        out
    }

    /// Records expand to the entries the old builders made, job for job.
    fn same_jobs(got: &[Job], want: &[Job], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: jobs");
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            let (g, w) = (stream(g), stream(w));
            assert_eq!(g.len(), w.len(), "{what}: job {k} length");
            for (e, (a, b)) in g.iter().zip(&w).enumerate() {
                assert_eq!(a, b, "{what}: job {k}, entry {e}");
            }
        }
    }

    /// Two allocators in the same state, so both builders place their output
    /// where the other does, and the same operands in each.
    fn pair(dram: &Dram, shapes: &[[usize; 2]]) -> [(DramAlloc, Vec<DramTensor>); 2] {
        [0, 1].map(|_| {
            let mut a = DramAlloc::new(dram);
            let t = shapes
                .iter()
                .map(|&[r, c]| DramTensor::alloc(&mut a, r, c).unwrap())
                .collect();
            (a, t)
        })
    }

    const DRAMS: [u8; 3] = [0xFF, 0b0111_0110, 0b1];

    #[test]
    fn matmul_records_expand_to_the_old_lists() {
        use crate::matmul::{Fidelity, SrcRoute};
        let cases = [
            ([64, 784], false, [784, 128], false),
            ([64, 128], false, [128, 10], false),
            ([64, 10], false, [128, 10], true),
            ([64, 128], true, [64, 10], false),
            ([37, 45], true, [37, 70], false),
            ([96, 320], false, [320, 288], false),
        ];
        for mask in DRAMS {
            let dram = Dram::from_usable_mask(mask);
            for (sa, ta, sb, tb) in cases {
                for units in [1, 3, 8] {
                    let [(mut a1, t1), (mut a2, t2)] = pair(&dram, &[sa, sb]);
                    let route = SrcRoute::Tf32FromFp32;
                    let f = Fidelity::HiFi4;
                    let got = super::matmul_dram(&mut a1, &t1[0], ta, &t1[1], tb, route, f, units)
                        .unwrap();
                    let want =
                        reference::matmul_dram(&mut a2, &t2[0], ta, &t2[1], tb, route, f, units)
                            .unwrap();
                    assert_eq!(got.out, want.out);
                    same_jobs(
                        &got.jobs,
                        &want.jobs,
                        &format!("{sa:?}@{sb:?} {mask:#x} {units}"),
                    );
                }
            }
        }
    }

    #[test]
    fn eltwise_records_expand_to_the_old_lists() {
        for mask in DRAMS {
            let dram = Dram::from_usable_mask(mask);
            for [r, c] in [[37, 70], [320, 320], [784, 128], [1, 1]] {
                for (k, scalar, b) in [
                    (kind::ADD, 0.0, Some(false)),
                    (kind::ADD_ROW, 0.0, Some(true)),
                    (kind::RELU, 0.0, None),
                    (kind::MUL_SCALAR, 0.5, None),
                ] {
                    for units in [1, 3, 8] {
                        let shapes = [[r, c], [if b == Some(true) { 1 } else { r }, c]];
                        let [(mut a1, t1), (mut a2, t2)] = pair(&dram, &shapes);
                        let op = Eltwise { kind: k, scalar };
                        let got =
                            super::eltwise(&mut a1, op, &t1[0], b.map(|_| &t1[1]), units).unwrap();
                        let want =
                            reference::eltwise(&mut a2, op, &t2[0], b.map(|_| &t2[1]), units)
                                .unwrap();
                        assert_eq!(got.out, want.out);
                        same_jobs(
                            &got.jobs,
                            &want.jobs,
                            &format!("kind {k} [{r}, {c}] {mask:#x} {units}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sum_records_expand_to_the_old_lists() {
        for mask in DRAMS {
            let dram = Dram::from_usable_mask(mask);
            for [r, c] in [[37, 70], [7000, 40], [96, 600], [6400, 33]] {
                for units in [1, 3, 8] {
                    let [(mut a1, t1), (mut a2, t2)] = pair(&dram, &[[r, c]]);
                    let got = super::sum_rows(&mut a1, &t1[0], units).unwrap();
                    let want = reference::sum_rows(&mut a2, &t2[0], units).unwrap();
                    assert_eq!(got.out, want.out);
                    same_jobs(
                        &got.jobs,
                        &want.jobs,
                        &format!("sum [{r}, {c}] {mask:#x} {units}"),
                    );
                }
            }
        }
    }

    /// A row view's records name its first tile, so they expand as the
    /// view's own lists did.
    #[test]
    fn records_of_a_row_view_expand_to_the_old_lists() {
        let dram = Dram::FULL;
        let [(mut a1, t1), (mut a2, t2)] = pair(&dram, &[[512, 96], [512, 96]]);
        let (v1, w1) = (
            t1[0].rows_view(64, 128).unwrap(),
            t1[1].rows_view(64, 128).unwrap(),
        );
        let (v2, w2) = (
            t2[0].rows_view(64, 128).unwrap(),
            t2[1].rows_view(64, 128).unwrap(),
        );
        let op = Eltwise {
            kind: kind::MUL,
            scalar: 0.0,
        };
        let got = super::eltwise(&mut a1, op, &v1, Some(&w1), 3).unwrap();
        let want = reference::eltwise(&mut a2, op, &v2, Some(&w2), 3).unwrap();
        same_jobs(&got.jobs, &want.jobs, "view");
        let got = super::sum_rows(&mut a1, &v1, 2).unwrap();
        let want = reference::sum_rows(&mut a2, &v2, 2).unwrap();
        same_jobs(&got.jobs, &want.jobs, "view sum");
    }

    #[test]
    fn runs_cover_every_item_once_evenly_and_within_the_limit() {
        for len in 0..60 {
            for units in 1..12 {
                for max in [1, 5, 96] {
                    let r = runs(len, units, max);
                    let items: Vec<usize> = r.iter().cloned().flatten().collect();
                    assert_eq!(items, (0..len).collect::<Vec<_>>(), "{len} {units} {max}");
                    let sizes: Vec<usize> = r.iter().map(|x| x.len()).collect();
                    assert!(sizes.iter().all(|&n| n <= max), "{sizes:?}");
                    let (lo, hi) = (sizes.iter().min(), sizes.iter().max());
                    if let (Some(lo), Some(hi)) = (lo, hi) {
                        assert!(hi - lo <= 1, "{len} {units} {max}: {sizes:?}");
                    }
                    // One run per unit whenever there are items enough.
                    if len >= units && len <= units * max {
                        assert_eq!(r.len(), units, "{len} {units} {max}");
                    }
                }
            }
        }
    }

    #[test]
    fn blocks_shrink_until_every_unit_has_one_and_never_grow() {
        for (mt, nt) in [(1, 1), (2, 4), (2, 1), (3, 9), (8, 16), (25, 4)] {
            for (mc, nc) in [(mt, nt), (1, nt), (mt.min(2), 1)] {
                for units in [1, 2, 3, 8, 120] {
                    let [bm, bn] = blocks([mt, nt], [mc, nc], units);
                    assert!(bm >= 1 && bm <= mc && bn >= 1 && bn <= nc);
                    let count = mt.div_ceil(bm) * nt.div_ceil(bn);
                    assert!(
                        count >= units.min(mt * nt) || (bm, bn) == (1, 1),
                        "[{mt}, {nt}] from [{mc}, {nc}] for {units}: [{bm}, {bn}] -> {count}"
                    );
                    if units == 1 {
                        assert_eq!([bm, bn], [mc, nc], "one unit keeps the plan's block");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod reference {
    //! The list builders as they were before op records (9.7a), kept as the
    //! specification `tt_isa::dm::record::expand` is held to: every record
    //! must expand to exactly the entries these built, with an `op::WAIT`
    //! wherever one of their lists ended inside a job.
    use super::*;
    use tt_isa::dm;
    use tt_isa::dram::DramRange;

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_dram(
        alloc: &mut DramAlloc,
        a: &DramTensor,
        a_transposed: bool,
        b: &DramTensor,
        b_transposed: bool,
        route: SrcRoute,
        fidelity: Fidelity,
        units: usize,
    ) -> Result<Work> {
        let (m, ka) = if a_transposed {
            (a.cols, a.rows)
        } else {
            (a.rows, a.cols)
        };
        let (kb, n) = if b_transposed {
            (b.cols, b.rows)
        } else {
            (b.rows, b.cols)
        };
        if ka != kb {
            return Err(TensorError::Shape(format!(
                "[{m}, {ka}] @ [{kb}, {n}]: inner dimensions differ"
            )));
        }
        let k = ka;
        let (in_fmt, out_fmt) = route.formats();
        let shape = matmul::plan_in([m, k, n], route, fidelity, Staging::Slots)
            .ok_or_else(|| TensorError::Shape(format!("[{m}, {k}] @ [{k}, {n}] fits no chunk")))?;
        let [mc, kc, nc] = shape.tiles;
        let [mt, kt, nt] = [m.div_ceil(32), k.div_ceil(32), n.div_ceil(32)];
        if kc < kt {
            return Err(TensorError::Shape(format!(
                "[{m}, {k}] @ [{k}, {n}] would split K; not on this path"
            )));
        }
        let [mc, nc] = blocks([mt, nt], [mc, nc], units);
        // Tile (i, j) of op(X), as a list entry into L1 slot `to`.
        let fetch = |x: &DramTensor, transposed: bool, i: usize, j: usize, to: u64| -> [u32; 8] {
            let (slot, op) = if transposed {
                (x.tile(j, i), dm::op::READ_TRANSPOSED)
            } else {
                (x.tile(i, j), dm::op::READ)
            };
            let ch = slot.channel();
            [
                op,
                ch.index() as u32,
                (i + j) as u32 % tt_isa::dram::PORTS as u32,
                slot.offset() as u32,
                to as u32,
                TILE_SLOT as u32,
                0,
                0,
            ]
        };
        let c = DramTensor::alloc(alloc, m, n)?;
        let mut jobs = Vec::new();
        for i0 in (0..mt).step_by(mc) {
            let rows = mc.min(mt - i0);
            for j0 in (0..nt).step_by(nc) {
                let cols = nc.min(nt - j0);
                let tiles = [rows, kt, cols];
                let (b_at, outputs) = match matmul::plan_layout_in(tiles, in_fmt, Staging::Slots) {
                    Ok(l) => l,
                    Err(e) => {
                        alloc.free(&c.placement);
                        return Err(e.into());
                    }
                };
                let mut list = Vec::new();
                for i in 0..rows {
                    for kk in 0..kt {
                        let to = matmul::MATMUL_STAGE + (i * kt + kk) as u64 * TILE_SLOT;
                        list.push(fetch(a, a_transposed, i0 + i, kk, to));
                    }
                }
                for kk in 0..kt {
                    for j in 0..cols {
                        let to = b_at + (kk * cols + j) as u64 * TILE_SLOT;
                        list.push(fetch(b, b_transposed, kk, j0 + j, to));
                    }
                }
                let roles = matmul::programs((tiles, Staging::Slots), route, fidelity, || {
                    matmul::matmul_roles(&outputs, in_fmt, out_fmt, fidelity)
                });

                // Only the datums go back: the packer writes nothing else, and the
                // unpacker skips the header whatever it holds (`step18_dram_matmul`).
                let mut back = Vec::with_capacity(rows * cols);
                for i in 0..rows {
                    for j in 0..cols {
                        let out = outputs[i * cols + j].out;
                        let slot = c.tile(i0 + i, j0 + j);
                        let data = slot
                            .channel()
                            .range(slot.offset() + TILE_DATA, 4096)
                            .expect("inside the slot");
                        back.push([
                            dm::op::WRITE,
                            data.channel().index() as u32,
                            (i + j) as u32 % tt_isa::dram::PORTS as u32,
                            data.offset() as u32,
                            out as u32,
                            4096,
                            0,
                            0,
                        ]);
                    }
                }
                jobs.push(vec![
                    Step::List {
                        what: "matmul gather",
                        entries: list,
                    },
                    Step::Matmul(roles),
                    Step::List {
                        what: "matmul scatter",
                        entries: back,
                    },
                ]);
            }
        }
        Ok(Work { out: c, jobs })
    }

    pub fn eltwise(
        alloc: &mut DramAlloc,
        op: Eltwise,
        a: &DramTensor,
        b: Option<&DramTensor>,
        units: usize,
    ) -> Result<Work> {
        use tt_isa::dm::kind;
        let binary = !matches!(op.kind, kind::MUL_SCALAR | kind::RELU);
        let row = op.kind == kind::ADD_ROW;
        match (binary, b) {
            (false, _) => {}
            (true, None) => {
                return Err(TensorError::Shape("a binary op needs two operands".into()))
            }
            (true, Some(b)) if row && (b.rows != 1 || b.cols != a.cols) => {
                return Err(TensorError::Shape(format!(
                    "[{}, {}] + row [{}, {}]",
                    a.rows, a.cols, b.rows, b.cols
                )))
            }
            (true, Some(b)) if !row && (b.rows, b.cols) != (a.rows, a.cols) => {
                return Err(TensorError::Shape(format!(
                    "[{}, {}] and [{}, {}] differ",
                    a.rows, a.cols, b.rows, b.cols
                )))
            }
            _ => {}
        }
        let out = DramTensor::alloc(alloc, a.rows, a.cols)?;
        let [rt, ct] = a.grid();
        // Two slots per tile in flight, inside the matmul staging area.
        const GROUP: usize = 96;
        let slot = |i: usize| matmul::MATMUL_STAGE + i as u64 * TILE_SLOT;
        const _: () = assert!(
            matmul::MATMUL_STAGE + 2 * GROUP as u64 * TILE_SLOT <= tt_isa::mailbox::MAILBOX_BASE
        );
        let read = |x: DramRange, to: u64, port: usize| {
            [
                dm::op::READ,
                x.channel().index() as u32,
                (port % tt_isa::dram::PORTS as usize) as u32,
                x.offset() as u32,
                to as u32,
                TILE_SLOT as u32,
                0,
                0,
            ]
        };
        let tiles: Vec<(usize, usize)> =
            (0..rt).flat_map(|i| (0..ct).map(move |j| (i, j))).collect();
        let mut jobs = Vec::new();
        for run in runs(tiles.len(), units, GROUP) {
            let group = &tiles[run];
            let mut list = Vec::new();
            for (n, &(i, j)) in group.iter().enumerate() {
                list.push(read(a.tile(i, j), slot(2 * n), n));
                if let Some(b) = b.filter(|_| binary) {
                    let from = if row { b.tile(0, j) } else { b.tile(i, j) };
                    list.push(read(from, slot(2 * n + 1), n + 1));
                }
                list.push([
                    dm::op::COMPUTE,
                    op.kind,
                    op.scalar.to_bits(),
                    0,
                    slot(2 * n) as u32,
                    slot(2 * n) as u32,
                    slot(2 * n + 1) as u32,
                    0,
                ]);
                let o = out.tile(i, j);
                let data = o
                    .channel()
                    .range(o.offset() + TILE_DATA, 4096)
                    .expect("inside the slot");
                list.push([
                    dm::op::WRITE,
                    data.channel().index() as u32,
                    (n % tt_isa::dram::PORTS as usize) as u32,
                    data.offset() as u32,
                    (slot(2 * n) + TILE_DATA) as u32,
                    4096,
                    0,
                    0,
                ]);
            }
            jobs.push(vec![Step::List {
                what: "eltwise list",
                entries: list,
            }]);
        }
        Ok(Work { out, jobs })
    }

    pub fn sum_rows(alloc: &mut DramAlloc, a: &DramTensor, units: usize) -> Result<Work> {
        let out = DramTensor::alloc(alloc, 1, a.cols)?;
        let [rt, ct] = a.grid();
        let slot = |i: usize| matmul::MATMUL_STAGE + i as u64 * TILE_SLOT;
        // Slots in the staging area; each column takes one accumulator and one
        // per tile. Whole columns are packed into one mover list while they fit;
        // a column taller than that spans several lists, accumulating in place.
        const SLOTS: usize = 200;
        let read = |from: DramRange, to: u64, i: usize| {
            [
                dm::op::READ,
                from.channel().index() as u32,
                (i % tt_isa::dram::PORTS as usize) as u32,
                from.offset() as u32,
                to as u32,
                TILE_SLOT as u32,
                0,
                0,
            ]
        };
        let sum = |acc: u64, tile: u64, first: bool| {
            [
                dm::op::COMPUTE,
                dm::kind::COL_SUM,
                u32::from(first),
                0,
                acc as u32,
                tile as u32,
                acc as u32,
                0,
            ]
        };
        let write = |j: usize, acc: u64| {
            let o = out.tile(0, j);
            let data = o
                .channel()
                .range(o.offset() + TILE_DATA, 4096)
                .expect("in the slot");
            [
                dm::op::WRITE,
                data.channel().index() as u32,
                0,
                data.offset() as u32,
                (acc + TILE_DATA) as u32,
                4096,
                0,
                0,
            ]
        };
        let mut jobs = Vec::new();
        for columns in runs(ct, units, ct.max(1)) {
            let mut job = Vec::new();
            let mut list = Vec::new();
            let mut flush = |list: &mut Vec<[u32; 8]>| {
                job.push(Step::List {
                    what: "sum list",
                    entries: std::mem::take(list),
                })
            };
            let mut used = 0;
            for j in columns {
                if used + 1 + rt.min(SLOTS - 1) > SLOTS {
                    flush(&mut list);
                    used = 0;
                }
                let acc = slot(used);
                used += 1;
                for i0 in (0..rt).step_by(SLOTS - 1) {
                    let rows = (rt - i0).min(SLOTS - 1);
                    if used + rows > SLOTS {
                        // Only a column taller than a list reaches here: run what
                        // is queued, keeping the accumulator slot where it is.
                        flush(&mut list);
                        used = 1 + (acc - matmul::MATMUL_STAGE) as usize / TILE_SLOT as usize;
                    }
                    for i in i0..i0 + rows {
                        list.push(read(a.tile(i, j), slot(used + i - i0), i));
                    }
                    for i in i0..i0 + rows {
                        list.push(sum(acc, slot(used + i - i0), i == 0));
                    }
                    used += rows;
                }
                list.push(write(j, acc));
            }
            flush(&mut list);
            jobs.push(job);
        }
        Ok(Work { out, jobs })
    }
}
