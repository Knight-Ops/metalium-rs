//! Tensors that live in GDDR, and the matmul that reads and writes them there.
//!
//! Phase 9's point: a tensor is uploaded once, stays on the card, and ops read
//! and write it there, so what crosses PCIe per op is descriptors, not data.
//!
//! A [`DramTensor`] is a row-major `[rows, cols]` FP32 matrix stored as 32x32
//! tiles, one `tt_isa::dm::TILE_SLOT` each, interleaved across the chip's
//! usable channels: tile `t` of the row-major tile grid lives on channel
//! `channels[t % n]`, slot `t / n` of the tensor's region there. Spreading tiles
//! over channels is what lets later work read them in parallel; for now it also
//! means the allocator's regions stay small.
//!
//! **Padding.** A ragged tensor's edge tiles hold datums past its last row or
//! column, and the hardware computes whole tiles, so what they hold reaches
//! anything that accumulates over them. It is a property of the tensor, not an
//! assumption: [`DramTensor::pad`] says what the padding holds ([`Pad`]), every
//! op says what it needs of its inputs' and what it leaves in its output's
//! ([`OpPadding`]), and the session refills edge tiles -- only those, so the
//! cost is the perimeter, never the area -- when a need and a pad differ
//! (`hardware-coverage.md` F0, `tt-metal-concepts-review.md` G1). Upload
//! writes zeros.

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
    /// A trace refused (`crate::trace`).
    Trace(crate::trace::TraceError),
    /// An op given a tensor of an element type it does not compute on: a
    /// matmul of booleans, a sum of integers -- refused, never computed on
    /// the bits as if they were FP32.
    Elem {
        op: String,
        got: Elem,
        wants: Elem,
    },
}

impl From<crate::trace::TraceError> for TensorError {
    fn from(e: crate::trace::TraceError) -> Self {
        TensorError::Trace(e)
    }
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
            TensorError::Trace(e) => write!(f, "{e}"),
            TensorError::Elem { op, got, wants } => write!(
                f,
                "{op} takes {wants:?} tensors, not {got:?}: the device does not compute it on \
                 these, and treating their bits as {wants:?} would be wrong"
            ),
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
    /// Did this placement allocate its slots, or is it a view of another's?
    pub fn owned(&self) -> bool {
        self.owned
    }

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

    /// The whole of a one-channel placement's slots ([`DramAlloc::alloc_on`]).
    pub(crate) fn region(&self) -> Option<DramRange> {
        match self.channels[..] {
            [c] => c.range(self.base[0], self.slots * TILE_SLOT),
            _ => None,
        }
    }
}

/// Where a [`DramAlloc`]'s free space was at one moment: what a trace was
/// captured against (`crate::trace`).
#[derive(Clone, Debug)]
pub(crate) struct FreeSnapshot {
    channels: Vec<DramChannel>,
    free: Vec<BTreeMap<u64, u64>>,
}

impl FreeSnapshot {
    /// Were all of `p`'s slots free then -- so allocated since, and nothing
    /// captured then can name them?
    pub(crate) fn was_free(&self, p: &Placement) -> bool {
        p.channels.iter().zip(&p.base).all(|(c, &at)| {
            let len = p.slots * TILE_SLOT;
            let Some(i) = self.channels.iter().position(|k| k == c) else {
                return false;
            };
            self.free[i]
                .range(..=at)
                .next_back()
                .is_some_and(|(&start, &l)| at + len <= start + l)
        })
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

    /// `bytes` contiguous on channel `channel` (an index into the chip's
    /// channels), in whole slots: a trace's stream (`crate::trace`).
    pub fn alloc_on(&mut self, channel: usize, bytes: u64) -> Result<Placement> {
        let n = self.channels.len();
        if channel >= n {
            return Err(TensorError::Shape(format!("channel {channel} of {n}")));
        }
        let slots = bytes.div_ceil(TILE_SLOT).max(1);
        let len = slots * TILE_SLOT;
        let Some((&at, &free)) = self.free[channel].iter().find(|(_, &l)| l >= len) else {
            return Err(TensorError::Shape(format!(
                "no {len} contiguous bytes left on channel {channel}"
            )));
        };
        self.free[channel].remove(&at);
        if free > len {
            self.free[channel].insert(at + len, free - len);
        }
        Ok(Placement {
            channels: vec![self.channels[channel]],
            base: vec![at],
            slots,
            tiles: 1,
            first: 0,
            owned: true,
        })
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
        // By channel, not position: an `alloc_on` placement has one.
        for (c, &b) in p.channels.iter().zip(&p.base) {
            let i = self
                .channels
                .iter()
                .position(|k| k == c)
                .expect("a placement's channel is the allocator's");
            release(&mut self.free[i], b, p.slots * TILE_SLOT);
        }
    }

    /// Where the free space is now ([`FreeSnapshot`]).
    pub(crate) fn snapshot(&self) -> FreeSnapshot {
        FreeSnapshot {
            channels: self.channels.clone(),
            free: self.free.clone(),
        }
    }

    /// The chip's usable channels, in the allocator's order.
    pub(crate) fn channel_count(&self) -> usize {
        self.channels.len()
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

/// What a tensor's padding holds: the datums of its edge tiles past its last
/// row or column. A tensor with no ragged edge has none, and is always
/// [`Pad::Zero`] (`DramTensor::set_pad`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Pad {
    /// Zeros, of either sign: the identity of a sum, so a matmul's `K` and a
    /// column sum may run over it.
    Zero,
    /// Whatever an op left there.
    Undefined,
}

/// What an op needs of an input's padding.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PadNeed {
    /// Nothing: the op never lets padding reach a real datum.
    Any,
    /// [`Pad::Zero`].
    Zero,
}

/// An op's padding rules, as data: what it needs of each input's padding and
/// what it leaves in its output's, from the op's algebra (`f(0) == 0` is a
/// fact about `f`). Part of every device op's definition
/// (`hardware-coverage.md`, Definition of done, line 7).
pub trait OpPadding {
    fn requires(&self, input: usize) -> PadNeed;
    /// The output's padding, given the inputs' tensors (their pads and shapes).
    fn produces(&self, inputs: &[&DramTensor]) -> Pad;
}

/// What a tensor's 32-bit datums are (`hardware-coverage.md` D3). The device
/// moves all three alike -- the FP32-coded unpack to `Dst` and pack back carry
/// every bit pattern unchanged (`step26_sfpu_isa`'s `INT32` pass-through
/// case) -- so the tag says only what an op may compute on.
///
/// - `I32` is two's complement, as the host has it: the unpacker's and
///   packer's `INT32` is sign-magnitude, but nothing converts between formats
///   here, and the SFPU's integer arithmetic (`SFPIADD`) is two's complement,
///   so the raw bits are the useful form -- and `i32::MIN` has one.
/// - `Bool` is `0` or `1` as an `I32` (never `1.0`: an FP32 op sees `1` as a
///   denormal and would flush it).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    F32,
    I32,
    Bool,
}

/// A `[rows, cols]` matrix of 32-bit datums in GDDR, tiled, FP32 unless
/// [`DramTensor::elem`] says otherwise. See the module documentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DramTensor {
    pub rows: usize,
    pub cols: usize,
    pub elem: Elem,
    pub placement: Placement,
    /// What the padding holds. Behind a cell because filling it changes no
    /// element of the tensor: the session refreshes it through a shared
    /// reference ([`DramTensor::set_pad`]).
    pad: std::cell::Cell<Pad>,
}

impl DramTensor {
    /// What the padding holds.
    pub fn pad(&self) -> Pad {
        self.pad.get()
    }

    /// Record what the padding holds. A tensor with no ragged edge has no
    /// padding, which is trivially zero.
    pub fn set_pad(&self, pad: Pad) {
        self.pad
            .set(if self.has_padding() { pad } else { Pad::Zero });
    }

    /// Does any tile hold datums past the last row or column?
    pub fn has_padding(&self) -> bool {
        self.rows % 32 != 0 || self.cols % 32 != 0
    }

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

    /// Allocate a `[rows, cols]` FP32 tensor, contents undefined.
    pub fn alloc(alloc: &mut DramAlloc, rows: usize, cols: usize) -> Result<Self> {
        Self::alloc_elem(alloc, rows, cols, Elem::F32)
    }

    /// Allocate a `[rows, cols]` tensor of `elem`, contents undefined.
    pub fn alloc_elem(alloc: &mut DramAlloc, rows: usize, cols: usize, elem: Elem) -> Result<Self> {
        let tiles = rows.div_ceil(32).max(1) * cols.div_ceil(32).max(1);
        let t = DramTensor {
            rows,
            cols,
            elem,
            placement: alloc.alloc(tiles)?,
            pad: std::cell::Cell::new(Pad::Undefined),
        };
        t.set_pad(Pad::Undefined);
        Ok(t)
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
        let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
        Self::upload_bits(dev, w, alloc, &bits, rows, cols, Elem::F32)
    }

    /// Upload row-major datums of `elem`, as their bits: an `I32`'s two's
    /// complement, a `Bool`'s `0` or `1` (anything else refused).
    #[allow(clippy::too_many_arguments)]
    pub fn upload_bits<T: Transport>(
        dev: &mut Device<T>,
        w: &Window,
        alloc: &mut DramAlloc,
        values: &[u32],
        rows: usize,
        cols: usize,
        elem: Elem,
    ) -> Result<Self> {
        if values.len() != rows * cols {
            return Err(TensorError::Shape(format!(
                "{} values for a [{rows}, {cols}] tensor",
                values.len()
            )));
        }
        let t = Self::alloc_elem(alloc, rows, cols, elem)?;
        t.write_bits(dev, w, values)?;
        Ok(t)
    }

    /// Refuse anything but `want`, naming `op`.
    pub fn expect(&self, op: &str, want: Elem) -> Result<()> {
        if self.elem == want {
            Ok(())
        } else {
            Err(TensorError::Elem {
                op: op.into(),
                got: self.elem,
                wants: want,
            })
        }
    }

    /// Overwrite every datum of this tensor, in its own slots: a trace's input
    /// between replays (`crate::trace`). The padding becomes zero, as an
    /// upload's. A view (rows of another tensor) is refused: its slots are
    /// someone else's.
    pub fn write<T: Transport>(
        &self,
        dev: &mut Device<T>,
        w: &Window,
        values: &[f32],
    ) -> Result<()> {
        self.expect("a write of FP32 values", Elem::F32)?;
        let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
        self.write_bits(dev, w, &bits)
    }

    /// [`DramTensor::write`] of datums as their bits, whatever the element
    /// type; a `Bool` tensor takes only `0` and `1`.
    pub fn write_bits<T: Transport>(
        &self,
        dev: &mut Device<T>,
        w: &Window,
        values: &[u32],
    ) -> Result<()> {
        if self.elem == Elem::Bool {
            if let Some(i) = values.iter().position(|&v| v > 1) {
                return Err(TensorError::Shape(format!(
                    "a Bool tensor's datum {i} is {:#x}, not 0 or 1",
                    values[i]
                )));
            }
        }
        // The tilizer is FP32's, which moves bits: `from_bits` keeps every
        // pattern, NaN payloads included.
        let values: Vec<f32> = values.iter().map(|&b| f32::from_bits(b)).collect();
        let values = &values[..];
        let (rows, cols) = (self.rows, self.cols);
        if values.len() != rows * cols {
            return Err(TensorError::Shape(format!(
                "{} values for a [{rows}, {cols}] tensor",
                values.len()
            )));
        }
        if !self.placement.owned {
            return Err(TensorError::Shape(
                "a view's slots are another tensor's: write that one".into(),
            ));
        }
        let t = self;
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
        // `tilize_f32` pads with zeros.
        t.set_pad(Pad::Zero);
        Ok(())
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
        let v = DramTensor {
            rows,
            cols: self.cols,
            elem: self.elem,
            placement: Placement {
                tiles: rows.div_ceil(32) * ct,
                first: self.placement.first + first_row / 32 * ct,
                owned: false,
                ..self.placement.clone()
            },
            pad: self.pad.clone(),
        };
        // A view that stops at a tile row has none of the parent's padding
        // rows, only its padding columns: no better known than the parent's.
        v.set_pad(self.pad());
        Ok(v)
    }

    /// Download to row-major values: one bulk read per channel, or, for a
    /// view or a small tensor, only what its data occupies.
    pub fn download<T: Transport>(&self, dev: &mut Device<T>, w: &Window) -> Result<Vec<f32>> {
        self.expect("a download as FP32 values", Elem::F32)?;
        self.download_any(dev, w)
    }

    /// Download to row-major datums as their bits, whatever the element type.
    pub fn download_bits<T: Transport>(&self, dev: &mut Device<T>, w: &Window) -> Result<Vec<u32>> {
        Ok(self
            .download_any(dev, w)?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    }

    fn download_any<T: Transport>(&self, dev: &mut Device<T>, w: &Window) -> Result<Vec<f32>> {
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

    /// Download every datum of every tile, padding included, as a row-major
    /// `[32 * rt, 32 * ct]` matrix: what the hardware sees, for checking what
    /// [`DramTensor::pad`] claims.
    pub fn download_padded<T: Transport>(
        &self,
        dev: &mut Device<T>,
        w: &Window,
    ) -> Result<Vec<f32>> {
        let [rt, ct] = self.grid();
        let mut packed = vec![0u8; rt * ct * 4096];
        for i in 0..rt {
            for j in 0..ct {
                let s = self.tile(i, j);
                let r = s
                    .channel()
                    .range(s.offset() + TILE_DATA, 4096)
                    .expect("in the slot");
                dev.dram_read(w, r, &mut packed[(i * ct + j) * 4096..][..4096])?;
            }
        }
        Ok(matmul::detilize_packed(&packed, 32 * rt, 32 * ct))
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
    /// A kernel on the tile's resident roles: its three role programs, run
    /// concurrently, and the semaphores its plan gave it as a run initialises
    /// them (`crate::l1::Plan::semaphore_init`). It must leave every one where
    /// it started (`Kernel::restores_semaphores`), since the mover runs it
    /// back to back with no setup between runs.
    Kernel {
        roles: Arc<[Vec<Instruction>; 3]>,
        init: Vec<crate::runtime::SemaphoreInit>,
        /// Each role's MOP Expander configuration (`runtime::Kernel::mop`):
        /// the kernels of one list share it, so a list ends where it changes.
        mop: Box<[Option<tt_isa::frontend::mop::MopConfig>; 3]>,
        /// Each role's block repeats (`runtime::Kernel::loops`): shared by a
        /// list's kernels as `mop` is, since the table is a descriptor word.
        loops: Arc<[Vec<crate::code::Loop>; 3]>,
    },
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
    allow_mop: bool,
) -> Result<Work> {
    a.expect("a matmul", Elem::F32)?;
    b.expect("a matmul", Elem::F32)?;
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
            let matmul::Layout {
                b_at,
                outputs,
                sems,
                init,
            } = match matmul::plan_layout_in(tiles, in_fmt, Staging::Slots) {
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
            let (roles, mop) =
                matmul::kernel_programs(tiles, route, fidelity, sems, allow_mop, || {
                    matmul::matmul_kernel(&outputs, sems, in_fmt, out_fmt, fidelity, allow_mop)
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
                Step::Kernel {
                    roles,
                    init,
                    mop: Box::new(mop),
                    loops: Default::default(),
                },
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
    /// A second immediate, for the kinds that take two (`CLAMP`'s bounds,
    /// `HARD_SIGMOID`'s slope and offset); `0.0` for the rest. SFPU only: the
    /// mover's record carries one.
    pub scalar2: f32,
}

/// The shapes an element-wise op accepts: `B` where the kind takes one, `A`'s
/// shape or, for `ADD_ROW`, one row as wide.
fn check_eltwise(op: Eltwise, a: &DramTensor, b: Option<&DramTensor>) -> Result<()> {
    broadcast_of(op, a, b).map(|_| ())
}

/// What an element-wise op is once its operands' shapes are read: its kind
/// (`ADD_ROW` is `ADD` with a row broadcast) and how `b` meets `a` -- the same
/// shape, one row (`[1, cols]`) or one column (`[rows, 1]`), the last two only
/// for the kinds that take one (`sfpu::ops::broadcasts`). Refuses anything
/// else.
pub fn broadcast_of(
    op: Eltwise,
    a: &DramTensor,
    b: Option<&DramTensor>,
) -> Result<(u32, crate::sfpu::ops::Broadcast)> {
    use crate::sfpu::kernel::Operands;
    use crate::sfpu::ops::{broadcasts, Broadcast};
    use tt_isa::dm::kind;
    let name = || format!("element-wise op {:#x}", op.kind);
    // `None`: a kind that moves datums, whatever they are.
    if let Some(sig) = crate::sfpu::ops::elems(op.kind) {
        a.expect(&name(), sig.inputs[0])?;
        if let (Some(b), Some(&want)) = (b, sig.inputs.get(1)) {
            b.expect(&name(), want)?;
        }
    }
    let binary = match crate::sfpu::ops::operands(op.kind) {
        Some(o) => o != Operands::Unary,
        None if crate::sfpu::ops::mover_has(op.kind) => !matches!(
            op.kind,
            kind::MUL_SCALAR | kind::ADD_SCALAR | kind::RELU | kind::COPY
        ),
        None => {
            return Err(TensorError::Shape(format!(
                "no element-wise op {:#x}",
                op.kind
            )))
        }
    };
    let shape = |t: &DramTensor| (t.rows, t.cols);
    match (binary, b) {
        (false, _) => Ok((op.kind, Broadcast::None)),
        (true, None) => Err(TensorError::Shape("a binary op needs two operands".into())),
        (true, Some(b)) if op.kind == kind::ADD_ROW => {
            if shape(b) == (1, a.cols) {
                Ok((kind::ADD, Broadcast::Row))
            } else {
                Err(TensorError::Shape(format!(
                    "[{}, {}] + row [{}, {}]",
                    a.rows, a.cols, b.rows, b.cols
                )))
            }
        }
        (true, Some(b)) if shape(b) == shape(a) => Ok((op.kind, Broadcast::None)),
        (true, Some(b)) if broadcasts(op.kind) && shape(b) == (1, a.cols) => {
            Ok((op.kind, Broadcast::Row))
        }
        (true, Some(b)) if broadcasts(op.kind) && shape(b) == (a.rows, 1) => {
            Ok((op.kind, Broadcast::Col))
        }
        (true, Some(b)) => Err(TensorError::Shape(format!(
            "[{}, {}] and [{}, {}]: neither the same shape nor a row or column to broadcast",
            a.rows, a.cols, b.rows, b.cols
        ))),
    }
}

/// The element type `kind`'s output has, given its first operand.
fn output_elem(kind: u32, a: &DramTensor) -> Elem {
    crate::sfpu::ops::elems(kind).map_or(a.elem, |sig| sig.out)
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
    check_eltwise(op, a, b)?;
    let binary = !matches!(
        op.kind,
        kind::MUL_SCALAR | kind::ADD_SCALAR | kind::RELU | kind::COPY
    );
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
    let out = DramTensor::alloc_elem(alloc, a.rows, a.cols, output_elem(op.kind, a))?;
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

/// Which unit computes an element-wise op. Both give the same bits
/// (`step19_eltwise`); they differ in cost.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum EltwiseUnit {
    /// The data mover's FP32 unit, datum by datum (`tt_isa::dm::kind`).
    Mover,
    /// The SFPU, through `crate::sfpu::kernel` -- where the op has a program
    /// (`crate::sfpu::ops`); the mover otherwise.
    Sfpu,
    /// Whichever [`sfpu_is_cheaper`] predicts is faster for the op's size.
    #[default]
    Auto,
}

/// Does the SFPU finish `kind` over `tiles` tiles on `units` units sooner than
/// the mover? A linear model of each, per op: a fixed cost growing with the
/// units (the host's per-unit submission; the SFPU's kernel reservation costs
/// more than a list), plus a cost per tile of the largest share. The
/// constants are `silicon_perf::eltwise_unit_sweep`'s, on card 0 (divergence
/// measurement Q): the mover takes ~8 us a tile for the `fadd.s`-shaped kinds
/// and ~22 us for the per-datum ones (`RELU`, `RELU_BACKWARD`, `ADD_ROW`), the
/// SFPU ~2-4 us; a list costs ~9 us plus ~2 us a unit, a kernel ~26 us plus
/// ~4.4 us a unit. So a small op spread thin stays on the mover and anything
/// with a few tiles a unit goes to the SFPU.
pub fn sfpu_is_cheaper(kind: u32, tiles: usize, units: usize) -> bool {
    use tt_isa::dm::kind as k;
    let per_unit = tiles.div_ceil(units.max(1)).max(1) as f64;
    let u = units.max(1) as f64 - 1.0;
    let (mover_tile, sfpu_tile) = match kind {
        k::RELU | k::RELU_BACKWARD | k::ADD_ROW => (22.4, 2.4),
        k::MUL_SCALAR | k::ADD_SCALAR => (7.9, 2.4),
        _ => (7.9, 3.0),
    };
    let mover = 9.5 + 2.0 * u + per_unit * mover_tile;
    let sfpu = 26.0 + 4.4 * u + per_unit * sfpu_tile;
    sfpu < mover
}

/// [`eltwise`] on the SFPU: each run of tiles a job of three steps -- the
/// mover gathers the run's operands into L1 (`record::READ_RUN`), the
/// resident roles run the SFPU kernel over them (`crate::sfpu::kernel`), the
/// mover scatters the outputs (`record::WRITE_RUN`). `None` when the op has no
/// SFPU program.
///
/// The roles' programs are memoised by op, scalar and run length, so a
/// model's repeated ops reuse them, and so does the program cache.
pub fn sfpu_eltwise(
    alloc: &mut DramAlloc,
    op: Eltwise,
    a: &DramTensor,
    b: Option<&DramTensor>,
    c: Option<&DramTensor>,
    units: usize,
) -> Result<Option<Work>> {
    use crate::sfpu::kernel::Operands;
    use crate::sfpu::ops::Broadcast;
    let (kind, bcast) = broadcast_of(op, a, b)?;
    let op = Eltwise { kind, ..op };
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
            if let Some(&want) = crate::sfpu::ops::elems(kind).and_then(|s| s.inputs.get(2)) {
                c.expect(&format!("element-wise op {kind:#x}"), want)?;
            }
            Some(c.tensor_ref())
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
    let group = sfpu_group(op, bcast, operands);
    let out = DramTensor::alloc_elem(alloc, a.rows, a.cols, output_elem(op.kind, a))?;
    let [rt, ct] = a.grid();
    let (ra, ro) = (a.tensor_ref(), out.tensor_ref());
    let rb = b.map(DramTensor::tensor_ref);
    let mut jobs = Vec::new();
    for run in runs(rt * ct, units, group) {
        let len = run.len();
        let (layout, roles, loops) = match sfpu_programs(op, bcast, operands, len) {
            Ok(p) => p,
            Err(e) => {
                alloc.free(&out.placement);
                return Err(e);
            }
        };
        let read = |x: &TensorRef, at: u64, flags: u32| {
            [
                [
                    record::READ_RUN,
                    run.start as u32,
                    len as u32,
                    at as u32,
                    flags,
                    ct as u32,
                    0,
                    0,
                ],
                x.encode()[0],
                x.encode()[1],
            ]
        };
        let mut gather = read(&ra, layout.a_at, 0).to_vec();
        if let (Some(rb), Some(b_at)) = (&rb, layout.b_at) {
            let flags = match bcast {
                Broadcast::None => 0,
                Broadcast::Row => 1,
                Broadcast::Col => 2,
            };
            gather.extend(read(rb, b_at, flags));
        }
        if let (Some(rc), Some(c_at)) = (&rc, layout.c_at) {
            gather.extend(read(rc, c_at, 0));
        }
        let scatter = [
            [
                record::WRITE_RUN,
                run.start as u32,
                len as u32,
                layout.out_at as u32,
                0,
                0,
                0,
                0,
            ],
            ro.encode()[0],
            ro.encode()[1],
        ];
        jobs.push(vec![
            Step::List {
                what: "sfpu gather",
                entries: gather,
            },
            Step::Kernel {
                roles,
                init: layout.init.clone(),
                mop: Box::new([None; 3]),
                loops,
            },
            Step::List {
                what: "sfpu scatter",
                entries: scatter.to_vec(),
            },
        ]);
    }
    Ok(Some(Work { out, jobs }))
}

type SfpuPrograms = (
    crate::sfpu::kernel::Layout,
    Arc<[Vec<Instruction>; 3]>,
    Arc<[Vec<crate::code::Loop>; 3]>,
);

/// Most tiles one SFPU run of `op` may take: 64, or fewer if the data arena
/// cannot hold their slots or a role's program -- which grows by a fixed
/// amount per tile -- would outgrow a program slot (`mailbox::PROGRAM_MAX`).
/// Measured from the programs themselves, at one tile and at two, so an op
/// with a long program (`ADD_ROW`'s unrolled loop) gets shorter runs rather
/// than a refusal.
#[cfg(test)]
pub(crate) fn sfpu_group_for_tests(
    kind: u32,
    scalar: f32,
    operands: crate::sfpu::kernel::Operands,
) -> usize {
    let bcast = if kind == tt_isa::dm::kind::ADD_ROW {
        crate::sfpu::ops::Broadcast::Row
    } else {
        crate::sfpu::ops::Broadcast::None
    };
    let kind = if kind == tt_isa::dm::kind::ADD_ROW {
        tt_isa::dm::kind::ADD
    } else {
        kind
    };
    sfpu_group(
        Eltwise {
            scalar2: 0.0,
            kind,
            scalar,
        },
        bcast,
        operands,
    )
}

fn sfpu_group(
    op: Eltwise,
    bcast: crate::sfpu::ops::Broadcast,
    operands: crate::sfpu::kernel::Operands,
) -> usize {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    // Measuring builds the op's role programs twice over: once per op kind,
    // scalar and broadcast, not once per op.
    type Memo = Mutex<HashMap<(u32, u32, u32, crate::sfpu::ops::Broadcast), usize>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let key = (op.kind, op.scalar.to_bits(), op.scalar2.to_bits(), bcast);
    let memo = MEMO.get_or_init(Default::default);
    if let Some(&g) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return g;
    }
    let g = measure_sfpu_group(op, bcast, operands);
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key, g);
    g
}

fn measure_sfpu_group(
    op: Eltwise,
    bcast: crate::sfpu::ops::Broadcast,
    operands: crate::sfpu::kernel::Operands,
) -> usize {
    const GROUP: usize = 64;
    let lens = |n: usize| -> [usize; 3] {
        let layout = crate::sfpu::kernel::plan_layout(n, operands).expect("two tiles always fit");
        let (_, math) = crate::sfpu::ops::code_for(op.kind, [op.scalar, op.scalar2], bcast)
            .expect("checked by the caller");
        crate::sfpu::kernel::roles_code(&layout, operands, &math)
            .0
            .map(|p| p.len())
    };
    let (one, two) = (lens(1), lens(2));
    let max = tt_isa::mailbox::PROGRAM_MAX as usize;
    let by_program = (0..3)
        .map(|r| {
            let per = two[r] - one[r];
            let fixed = one[r] - per;
            (max - fixed) / per.max(1)
        })
        .min()
        .unwrap();
    GROUP
        .min(crate::sfpu::kernel::max_tiles(operands))
        .min(by_program)
        .max(1)
}

/// One run's layout and role programs, memoised by op, scalar and length.
fn sfpu_programs(
    op: Eltwise,
    bcast: crate::sfpu::ops::Broadcast,
    operands: crate::sfpu::kernel::Operands,
    len: usize,
) -> Result<SfpuPrograms> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = (u32, u32, u32, crate::sfpu::ops::Broadcast, usize);
    type Memo = Mutex<HashMap<Key, SfpuPrograms>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let key = (
        op.kind,
        op.scalar.to_bits(),
        op.scalar2.to_bits(),
        bcast,
        len,
    );
    let memo = MEMO.get_or_init(Default::default);
    if let Some(p) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(p.clone());
    }
    let layout = crate::sfpu::kernel::plan_layout(len, operands)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let (_, math) = crate::sfpu::ops::code_for(op.kind, [op.scalar, op.scalar2], bcast)
        .expect("checked by the caller");
    let (roles, loops) = crate::sfpu::kernel::roles_code(&layout, operands, &math);
    let p = (layout, Arc::new(roles), Arc::new(loops));
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key, p.clone());
    Ok(p)
}

/// `a` reduced over `axis` by `op` on the SFPU (`crate::sfpu::reduce`):
/// `[rows, 1]` over columns, `[1, cols]` over rows. Each run of output tiles is
/// a job: the mover gathers their lines of input tiles (column-major for a
/// reduction over rows), the kernel accumulates and folds them, the mover
/// writes the outputs.
pub fn sfpu_reduce(
    alloc: &mut DramAlloc,
    a: &DramTensor,
    op: crate::sfpu::reduce::ReduceOp,
    axis: crate::sfpu::reduce::Axis,
    units: usize,
) -> Result<Work> {
    a.expect("a reduction", Elem::F32)?;
    use crate::sfpu::reduce::Axis;
    let [rt, ct] = a.grid();
    let (outs, per, valid, out) = match axis {
        Axis::Cols => (rt, ct, a.cols % 32, DramTensor::alloc(alloc, a.rows, 1)?),
        Axis::Rows => (ct, rt, a.rows % 32, DramTensor::alloc(alloc, 1, a.cols)?),
    };
    let valid = if valid == 0 { 32 } else { valid as u32 };
    let group = match reduce_group(op, axis, per, valid) {
        Some(g) => g,
        None => {
            alloc.free(&out.placement);
            return Err(TensorError::Shape(format!(
                "a reduction over {per} tiles does not fit one tile's L1 and program slots"
            )));
        }
    };
    let (ra, ro) = (a.tensor_ref(), out.tensor_ref());
    let mut jobs = Vec::new();
    for run in runs(outs, units, group) {
        let len = run.len();
        let (layout, roles) = match reduce_programs(op, axis, len, per, valid) {
            Ok(p) => p,
            Err(e) => {
                alloc.free(&out.placement);
                return Err(e);
            }
        };
        let (flags, grid_rt) = match axis {
            Axis::Cols => (0, 0),
            Axis::Rows => (4, rt as u32),
        };
        let gather = [
            [
                record::READ_RUN,
                (run.start * per) as u32,
                (len * per) as u32,
                layout.in_at as u32,
                flags,
                ct as u32,
                grid_rt,
                0,
            ],
            ra.encode()[0],
            ra.encode()[1],
        ];
        let scatter = [
            [
                record::WRITE_RUN,
                run.start as u32,
                len as u32,
                layout.out_at as u32,
                0,
                0,
                0,
                0,
            ],
            ro.encode()[0],
            ro.encode()[1],
        ];
        jobs.push(vec![
            Step::List {
                what: "reduce gather",
                entries: gather.to_vec(),
            },
            Step::Kernel {
                roles,
                init: layout.init.clone(),
                mop: Box::new([None; 3]),
                loops: Default::default(),
            },
            Step::List {
                what: "reduce scatter",
                entries: scatter.to_vec(),
            },
        ]);
    }
    Ok(Work { out, jobs })
}

type ReducePrograms = (crate::sfpu::reduce::Layout, Arc<[Vec<Instruction>; 3]>);

/// Most output tiles one reduce run may take, from the slots the data arena
/// holds and the role programs' length per output tile; `None` if not even
/// one fits.
fn reduce_group(
    op: crate::sfpu::reduce::ReduceOp,
    axis: crate::sfpu::reduce::Axis,
    per: usize,
    valid: u32,
) -> Option<usize> {
    use crate::sfpu::reduce::{Axis, ReduceOp};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Memo = Mutex<HashMap<(ReduceOp, Axis, usize, u32), Option<usize>>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let key = (op, axis, per, valid);
    let memo = MEMO.get_or_init(Default::default);
    if let Some(&g) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return g;
    }
    let g = measure_reduce_group(op, axis, per, valid);
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key, g);
    g
}

fn measure_reduce_group(
    op: crate::sfpu::reduce::ReduceOp,
    axis: crate::sfpu::reduce::Axis,
    per: usize,
    valid: u32,
) -> Option<usize> {
    use crate::sfpu::reduce::{math_programs, plan_layout, roles};
    const GROUP: usize = 64;
    let slots = (tt_isa::l1::DATA.len() / TILE_SLOT) as usize;
    let by_slots = slots / (per + 1);
    if by_slots == 0 {
        return None;
    }
    let (inputs, fin) = math_programs(op, axis, per, valid);
    let lens = |n: usize| -> Option<[usize; 3]> {
        let layout = plan_layout(n, per).ok()?;
        Some(roles(&layout, &inputs, &fin).map(|p| p.len()))
    };
    let max = tt_isa::mailbox::PROGRAM_MAX as usize;
    let one = lens(1)?;
    if one.iter().any(|&l| l > max) {
        return None;
    }
    if by_slots == 1 {
        return Some(1);
    }
    let two = lens(2)?;
    let by_program = (0..3)
        .map(|r| {
            let per_out = (two[r] - one[r]).max(1);
            (max - (one[r] - per_out)) / per_out
        })
        .min()
        .unwrap();
    Some(GROUP.min(by_slots).min(by_program).max(1))
}

fn reduce_programs(
    op: crate::sfpu::reduce::ReduceOp,
    axis: crate::sfpu::reduce::Axis,
    len: usize,
    per: usize,
    valid: u32,
) -> Result<ReducePrograms> {
    use crate::sfpu::reduce::{math_programs, plan_layout, roles, Axis, ReduceOp};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = (ReduceOp, Axis, usize, usize, u32);
    type Memo = Mutex<HashMap<Key, ReducePrograms>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let key = (op, axis, len, per, valid);
    let memo = MEMO.get_or_init(Default::default);
    if let Some(p) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(p.clone());
    }
    let layout = plan_layout(len, per).map_err(|e| TensorError::Shape(e.to_string()))?;
    let (inputs, fin) = math_programs(op, axis, per, valid);
    let p = (layout.clone(), Arc::new(roles(&layout, &inputs, &fin)));
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key, p.clone());
    Ok(p)
}

/// The sum over rows of `a`, as a `[1, cols]` tensor, in `burn-flex`'s order:
/// from `+0.0`, adding rows in order (`tt_isa::dm::kind::COL_SUM`).
///
/// Columns are independent; each keeps its rows in order on one tile. They
/// are dealt out in contiguous runs, one [`Job`] per run, as many as `units`.
pub fn sum_rows(alloc: &mut DramAlloc, a: &DramTensor, units: usize) -> Result<Work> {
    a.expect("a column sum", Elem::F32)?;
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
                (a.rows % 32) as u32,
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

/// [`sum_rows`]'s padding rules. It reads only a ragged tensor's valid rows
/// (`record::SUM`), so it needs nothing of the input's padding; its output's
/// padding rows are zeroed, and its padding columns are sums of the input's.
pub struct SumRows;

impl OpPadding for SumRows {
    fn requires(&self, _: usize) -> PadNeed {
        PadNeed::Any
    }
    fn produces(&self, inputs: &[&DramTensor]) -> Pad {
        let a = inputs[0];
        if a.pad() == Pad::Zero || a.cols % 32 == 0 {
            Pad::Zero
        } else {
            Pad::Undefined
        }
    }
}

/// [`matmul_dram`]'s padding rules. The Matrix Unit runs over whole tiles, so
/// both operands' padding along `K` enters every product; zero there, and
/// along `M` and `N`, gives zero in the output's padding too.
pub struct MatmulPadding;

impl OpPadding for MatmulPadding {
    fn requires(&self, _: usize) -> PadNeed {
        PadNeed::Zero
    }
    fn produces(&self, _: &[&DramTensor]) -> Pad {
        Pad::Zero
    }
}

impl Eltwise {
    /// For the S2 kinds (10.2c): does a zero `a` -- with `b`, `c` zero where
    /// `b_zero`, `c_zero` say -- give a zero (of either sign)? From each op's
    /// algebra, its scalars known.
    fn zero_at_zero(&self, kind: u32, b_zero: bool, c_zero: bool) -> bool {
        use crate::sfpu::ops::{ieee_compare, kind_sfpu::*};
        let (s, s2) = (self.scalar, self.scalar2);
        let is_zero = |x: f32| x.to_bits() & 0x7fff_ffff == 0;
        match kind {
            NEG | ABS | SIGN | LEAKY_RELU | PRELU | IS_NAN | IS_INF => true,
            // `f32::clamp`: `0` stays where `min <= 0 <= max`.
            CLAMP => s <= 0.0 && 0.0 <= s2,
            // The scalar where it is not below (above) zero, `x` for a NaN one.
            CLAMP_MIN => s.is_nan() || s <= 0.0,
            CLAMP_MAX => s.is_nan() || s >= 0.0,
            // `alpha * 0 + beta`, clamped: `beta` (or zero) where `beta <= 0`.
            HARD_SIGMOID => s.is_finite() && s2 <= 0.0,
            EQ..=LE => b_zero && !ieee_compare(kind, 0.0, 0.0),
            EQ_S..=LE_S => !ieee_compare(kind, 0.0, s),
            MASK_FILL => b_zero || is_zero(s),
            MASK_WHERE => b_zero || c_zero,
            // 10.2d-f: `f(±0) = ±0`.
            SQRT | EXPM1 | TANH | ERF | GELU | SINH | ASINH | ATANH | SIN => true,
            // `0^s = 0` for `s > 0`.
            POW_S => s > 0.0,
            // `g (1/2)` and `g 0 1`: zero with the gradient's padding.
            GELU_BACKWARD | SIGMOID_BACKWARD | LOG_SIGMOID_BACKWARD => b_zero,
            _ => false,
        }
    }
}

impl OpPadding for Eltwise {
    /// The data mover computes a tile's datums independently, so padding
    /// never reaches a real datum.
    fn requires(&self, _: usize) -> PadNeed {
        PadNeed::Any
    }
    fn produces(&self, inputs: &[&DramTensor]) -> Pad {
        use tt_isa::dm::kind;
        let zero = |i: usize| inputs.get(i).is_some_and(|t| t.pad() == Pad::Zero);
        // A broadcast operand goes into the padding of the dimension it is
        // broadcast along: `0 + b` there is `b`. With no padding along it, the
        // other dimension's padding is `0 (op) 0`, zero for `ADD` and `SUB`.
        if let (Some(a), Some(b)) = (inputs.first(), inputs.get(1)) {
            if (b.rows, b.cols) != (a.rows, a.cols) || self.kind == kind::ADD_ROW {
                let along_clear = if b.rows == 1 {
                    a.rows % 32 == 0
                } else {
                    a.cols % 32 == 0
                };
                use crate::sfpu::ops::kind_sfpu;
                // `0 && b` is false wherever `a`'s padding is, and so is
                // `mask ? 0 : 0`.
                let z = (self.kind == kind_sfpu::BOOL_AND && zero(0))
                    || (self.kind == kind_sfpu::MASK_FILL
                        && zero(0)
                        && self.scalar.to_bits() & 0x7fff_ffff == 0)
                    || (matches!(
                        self.kind,
                        kind::ADD
                            | kind::SUB
                            | kind::ADD_ROW
                            | kind_sfpu::BOOL_OR
                            | kind_sfpu::BOOL_XOR
                    ) && zero(0)
                        && zero(1)
                        && along_clear);
                return if z { Pad::Zero } else { Pad::Undefined };
            }
        }
        let z = match self.kind {
            // `0 (op) 0` is a zero.
            kind::ADD | kind::SUB | kind::MUL => zero(0) && zero(1),
            // `0 * s` is a zero unless `s` is infinite or NaN.
            kind::MUL_SCALAR => zero(0) && self.scalar.is_finite(),
            // `0 + s` is a zero only if `s` is.
            kind::ADD_SCALAR => zero(0) && self.scalar == 0.0,
            kind::RELU => zero(0),
            kind::COPY => zero(0),
            // `a > 0 ? g : 0`: zero where either is.
            kind::RELU_BACKWARD => zero(0) || zero(1),
            // The row goes into every row, padding rows included; a tensor
            // with none keeps only padding columns, `0 + 0`.
            kind::ADD_ROW => zero(0) && zero(1) && inputs[0].rows % 32 == 0,
            // `false && b`, and `false || false`, `false != false`; `!false`
            // is true.
            crate::sfpu::ops::kind_sfpu::BOOL_AND => zero(0) || zero(1),
            crate::sfpu::ops::kind_sfpu::BOOL_OR | crate::sfpu::ops::kind_sfpu::BOOL_XOR => {
                zero(0) && zero(1)
            }
            k => zero(0) && self.zero_at_zero(k, zero(1), zero(2)),
        };
        if z {
            Pad::Zero
        } else {
            Pad::Undefined
        }
    }
}

/// Set `t`'s padding to `value` in place: a [`record::FILL_PAD`] over its edge
/// tiles only, dealt out in runs to `units` tiles. Nothing at all when `t`
/// has no ragged edge.
pub fn fill_pad(t: &DramTensor, value: f32, units: usize) -> Result<Vec<Job>> {
    const GROUP: usize = 192;
    let [rt, ct] = t.grid();
    let (rows, cols) = (t.rows % 32, t.cols % 32);
    let edges = if rows != 0 { ct } else { 0 }
        + if cols != 0 {
            rt - usize::from(rows != 0)
        } else {
            0
        };
    if edges == 0 {
        return Ok(Vec::new());
    }
    let stage = staging("fill-pad slots", GROUP)?;
    let r = t.tensor_ref();
    Ok(runs(edges, units, GROUP)
        .into_iter()
        .map(|run| {
            vec![Step::List {
                what: "fill-pad list",
                entries: vec![
                    [
                        record::FILL_PAD,
                        value.to_bits(),
                        run.start as u32,
                        run.len() as u32,
                        rows as u32,
                        cols as u32,
                        stage as u32,
                        rt as u32,
                    ],
                    r.encode()[0],
                    r.encode()[1],
                ],
            }]
        })
        .collect())
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
    use super::{
        blocks, reference, runs, DramAlloc, DramTensor, Eltwise, Job, MatmulPadding, OpPadding,
        Pad, PadNeed, Step, SumRows,
    };
    use tt_isa::dm::TILE_DATA;
    use tt_isa::dm::{kind, op, record};
    use tt_isa::dram::Dram;

    /// Every padding claim of an SFPU kind (`Eltwise::zero_at_zero`) holds
    /// for its program: zero `A` (either sign), and the other operands zero
    /// where the claim takes them zero and anything (a value, `±inf`, NaN)
    /// where not, by the interpreter.
    #[test]
    fn every_zero_at_zero_claim_holds_for_the_program() {
        use crate::sfpu::kernel::Operands;
        use crate::sfpu::ops::{elems, operands, reference_op, Broadcast};
        use crate::tensor::Elem;
        let scalars = [0.0, 1.0, -1.0, 2.0, 0.5, -0.0, f32::NAN];
        let floats = [1.0, -2.0, f32::INFINITY, f32::NAN];
        let mut checked = 0;
        for kind in 0x100..0x140 {
            let Some(ops) = operands(kind) else { continue };
            let n = match ops {
                Operands::Unary => 1,
                Operands::Ternary => 3,
                _ => 2,
            };
            let sig = elems(kind).unwrap();
            let others = |i: usize| -> Vec<f32> {
                match sig.inputs.get(i) {
                    Some(Elem::F32) | None => floats.to_vec(),
                    Some(Elem::Bool) => vec![f32::from_bits(1)],
                    Some(Elem::I32) => [1, u32::MAX, 0x8000_0000].map(f32::from_bits).to_vec(),
                }
            };
            for s in scalars {
                for s2 in scalars {
                    let e = Eltwise {
                        kind,
                        scalar: s,
                        scalar2: s2,
                    };
                    for (bz, cz) in [(true, true), (false, true), (true, false), (false, false)] {
                        if (n < 2 && !bz) || (n < 3 && !cz) || !e.zero_at_zero(kind, bz, cz) {
                            continue;
                        }
                        let bs = if bz { vec![0.0] } else { others(1) };
                        let cs = if cz { vec![0.0] } else { others(2) };
                        for a in [0.0f32, -0.0] {
                            for &b in &bs {
                                for &c in &cs {
                                    let t = [vec![a; 1024], vec![b; 1024], vec![c; 1024]];
                                    let ins: Vec<&[f32]> = t[..n].iter().map(|v| &v[..]).collect();
                                    let out =
                                        reference_op(kind, [s, s2], Broadcast::None, &ins, 32, 32);
                                    assert!(
                                        out[0].to_bits() & 0x7fff_ffff == 0,
                                        "{kind:#x} ({s}, {s2}) of {a}, {b}, {c}: {}",
                                        out[0]
                                    );
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 100, "{checked}");
    }

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
                Step::Kernel { roles, init, .. } => {
                    let mut k: Vec<u32> = roles.iter().map(|p| p.len() as u32).collect();
                    k.extend(
                        init.iter()
                            .flat_map(|(s, v, m)| [s.index() as u32, *v as u32, *m as u32]),
                    );
                    out.push(Err(k));
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
                    let got =
                        super::matmul_dram(&mut a1, &t1[0], ta, &t1[1], tb, route, f, units, false)
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
                        let op = Eltwise {
                            scalar2: 0.0,
                            kind: k,
                            scalar,
                        };
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

    /// A fill touches exactly the edge tiles, each once, with the valid region
    /// of the tile it is: the corner both edges', the rest one edge's.
    #[test]
    fn fill_pad_touches_each_edge_tile_once() {
        use std::collections::BTreeMap;
        for mask in DRAMS {
            let dram = Dram::from_usable_mask(mask);
            for [r, c] in [[37, 70], [64, 70], [37, 64], [64, 64], [1, 1], [500, 33]] {
                for units in [1, 3, 8] {
                    let mut a = DramAlloc::new(&dram);
                    let t = DramTensor::alloc(&mut a, r, c).unwrap();
                    let [rt, ct] = t.grid();
                    let mut want = BTreeMap::new();
                    for i in 0..rt {
                        for j in 0..ct {
                            let vr = if i == rt - 1 { r % 32 } else { 0 };
                            let vc = if j == ct - 1 { c % 32 } else { 0 };
                            if vr != 0 || vc != 0 {
                                let s = t.tile(i, j);
                                let key = (s.channel().index() as u32, s.offset() as u32);
                                want.insert(
                                    key,
                                    kind::fill_param(
                                        if vr == 0 { 32 } else { vr as u32 },
                                        if vc == 0 { 32 } else { vc as u32 },
                                    ),
                                );
                            }
                        }
                    }
                    let jobs = super::fill_pad(&t, f32::NEG_INFINITY, units).unwrap();
                    let mut got = BTreeMap::new();
                    for job in &jobs {
                        let e: Vec<_> = stream(job).into_iter().map(Result::unwrap).collect();
                        for w in e.chunks(3) {
                            let (rd, cp, wr) = (w[0], w[1], w[2]);
                            assert_eq!((rd[0], cp[0], wr[0]), (op::READ, op::COMPUTE, op::WRITE));
                            assert_eq!(cp[1], kind::FILL_PAD);
                            assert_eq!(cp[2], f32::NEG_INFINITY.to_bits());
                            assert_eq!((cp[4], cp[5]), (rd[4], rd[4]), "in place, in its slot");
                            assert_eq!((wr[1], wr[3]), (rd[1], rd[3] + TILE_DATA as u32));
                            assert!(got.insert((rd[1], rd[3]), cp[3]).is_none(), "twice");
                        }
                    }
                    assert_eq!(got, want, "[{r}, {c}] {mask:#x} {units}");
                    assert!(jobs.len() <= units.max(1) || want.len() > 192 * units);
                }
            }
        }
    }

    /// The pad an op leaves, from its algebra.
    #[test]
    fn ops_declare_the_padding_they_leave() {
        let dram = Dram::from_usable_mask(0xFF);
        let mut a = DramAlloc::new(&dram);
        let zero = DramTensor::alloc(&mut a, 37, 70).unwrap();
        zero.set_pad(Pad::Zero);
        let undef = DramTensor::alloc(&mut a, 37, 70).unwrap();
        let row = DramTensor::alloc(&mut a, 1, 70).unwrap();
        row.set_pad(Pad::Zero);
        let whole = DramTensor::alloc(&mut a, 64, 64).unwrap();
        assert_eq!(
            whole.pad(),
            Pad::Zero,
            "no ragged edge, nothing to be wrong"
        );
        assert_eq!(undef.pad(), Pad::Undefined);
        let e = |kind, scalar| Eltwise {
            scalar2: 0.0,
            kind,
            scalar,
        };
        let p = |op: Eltwise, ins: &[&DramTensor]| op.produces(ins);
        assert_eq!(p(e(kind::ADD, 0.0), &[&zero, &zero]), Pad::Zero);
        assert_eq!(p(e(kind::ADD, 0.0), &[&zero, &undef]), Pad::Undefined);
        assert_eq!(p(e(kind::MUL_SCALAR, 3.0), &[&zero]), Pad::Zero);
        assert_eq!(
            p(e(kind::MUL_SCALAR, f32::INFINITY), &[&zero]),
            Pad::Undefined
        );
        assert_eq!(p(e(kind::MUL_SCALAR, f32::NAN), &[&zero]), Pad::Undefined);
        assert_eq!(p(e(kind::RELU_BACKWARD, 0.0), &[&undef, &zero]), Pad::Zero);
        assert_eq!(p(e(kind::ADD_ROW, 0.0), &[&zero, &row]), Pad::Undefined);
        assert_eq!(p(e(kind::COPY, 0.0), &[&zero]), Pad::Zero);
        assert_eq!(SumRows.produces(&[&undef]), Pad::Undefined);
        assert_eq!(SumRows.produces(&[&zero]), Pad::Zero);
        assert_eq!(SumRows.requires(0), PadNeed::Any);
        assert_eq!(MatmulPadding.requires(1), PadNeed::Zero);
        assert_eq!(super::fill_pad(&whole, 0.0, 4).unwrap().len(), 0);
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
            scalar2: 0.0,
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
                let matmul::Layout {
                    b_at,
                    outputs,
                    sems,
                    init,
                } = match matmul::plan_layout_in(tiles, in_fmt, Staging::Slots) {
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
                let roles =
                    matmul::programs((tiles, Staging::Slots), route, fidelity, sems, || {
                        matmul::matmul_roles(&outputs, sems, in_fmt, out_fmt, fidelity)
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
                    Step::Kernel {
                        roles,
                        init,
                        mop: Box::new([None; 3]),
                        loops: Default::default(),
                    },
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
        let binary = !matches!(
            op.kind,
            kind::MUL_SCALAR | kind::ADD_SCALAR | kind::RELU | kind::COPY
        );
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
        // Of the last tile row, only the tensor's valid rows (`0`: all 32).
        let last_rows = (a.rows % 32) as u32;
        let sum = |acc: u64, tile: u64, first: bool, last: bool| {
            [
                dm::op::COMPUTE,
                dm::kind::COL_SUM,
                u32::from(first),
                if last { last_rows } else { 0 },
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
                        list.push(sum(acc, slot(used + i - i0), i == 0, i == rt - 1));
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
