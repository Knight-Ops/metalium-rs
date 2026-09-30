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

use tt_device::{Device, Transport, TransportError, Window};
use tt_isa::dm::{self, TILE_DATA, TILE_SLOT};
use tt_isa::dram::{Dram, DramChannel, DramRange, CHANNEL_BYTES};
use tt_isa::noc::NocId;
use tt_isa::tile::L1Format;

use crate::dm::{DataMover, DmError};
use crate::matmul::{self, Fidelity, SrcRoute, Staging};
use crate::runtime::{Kernel, Outcome, RunError, Schedule};

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
}

impl Placement {
    /// Tile `t`'s slot.
    pub fn slot(&self, t: usize) -> DramRange {
        assert!(t < self.tiles, "tile {t} of {}", self.tiles);
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
        })
    }

    pub fn free(&mut self, p: &Placement) {
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
        for (c, bytes) in regions.iter().enumerate() {
            let r = t.placement.channels[c]
                .range(t.placement.base[c], bytes.len() as u64)
                .expect("allocated");
            dev.dram_write(w, r, bytes)?;
        }
        Ok(t)
    }

    /// Download to row-major values: one bulk read per channel.
    pub fn download<T: Transport>(&self, dev: &mut Device<T>, w: &Window) -> Result<Vec<f32>> {
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

/// `op(A) @ op(B)`, where `op` is a transpose when asked, all in GDDR: the
/// data mover gathers each chunk's tiles into L1 (transposing where needed),
/// the resident roles compute it, and the mover writes the output tiles back.
/// Nothing but descriptors, programs and semaphores crosses PCIe.
///
/// Chunked by [`matmul::plan_in`] with [`Staging::Slots`], and refused if the
/// plan splits `K`: a split `K` means a partial-sum add, which the host path
/// does on the host, and doing it anywhere else would change the rounding.
#[allow(clippy::too_many_arguments)]
pub fn matmul_dram<T: Transport, N: NocId>(
    dev: &mut Device<T>,
    w: &Window,
    mover: &mut DataMover<N>,
    alloc: &mut DramAlloc,
    a: &DramTensor,
    a_transposed: bool,
    b: &DramTensor,
    b_transposed: bool,
    route: SrcRoute,
    fidelity: Fidelity,
    mut run: impl FnMut(&mut Device<T>, &Kernel<'_>) -> std::result::Result<Outcome, RunError>,
) -> Result<DramTensor> {
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
    for i0 in (0..mt).step_by(mc) {
        let rows = mc.min(mt - i0);
        for j0 in (0..nt).step_by(nc) {
            let cols = nc.min(nt - j0);
            let tiles = [rows, kt, cols];
            let (b_at, outputs) = matmul::plan_layout_in(tiles, in_fmt, Staging::Slots)?;
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
            stats::timed("matmul gather", || mover.run_list(dev, w, &list))?;

            let roles = matmul::programs((tiles, Staging::Slots), route, fidelity, || {
                matmul::matmul_roles(&outputs, in_fmt, out_fmt, fidelity)
            });
            let [unpack, math, pack] = &*roles;
            let kernel = Kernel {
                // `TILE_SEMAPHORES`: every run leaves them as it found them.
                restores_semaphores: true,
                ..Kernel::new(
                    [unpack, math, pack],
                    Schedule::Concurrent(&matmul::TILE_SEMAPHORES),
                )
            };
            stats::timed("matmul compute", || run(dev, &kernel))?;

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
            stats::timed("matmul scatter", || mover.run_list(dev, w, &back))?;
        }
    }
    Ok(c)
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
pub fn eltwise<T: Transport, N: NocId>(
    dev: &mut Device<T>,
    w: &Window,
    mover: &mut DataMover<N>,
    alloc: &mut DramAlloc,
    op: Eltwise,
    a: &DramTensor,
    b: Option<&DramTensor>,
) -> Result<DramTensor> {
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
    let tiles: Vec<(usize, usize)> = (0..rt).flat_map(|i| (0..ct).map(move |j| (i, j))).collect();
    for group in tiles.chunks(GROUP) {
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
        stats::timed("eltwise list", || mover.run_list(dev, w, &list))?;
    }
    Ok(out)
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
