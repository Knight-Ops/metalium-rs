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
    slot_bytes: u64,
    tiles: usize,
    /// A view's first tile within the allocation it borrows from.
    first: usize,
    /// Did this placement allocate its slots? A view ([`DramTensor::rows_view`])
    /// did not, and freeing it gives nothing back.
    owned: bool,
}

impl Placement {
    pub(crate) fn tensor_ref(&self, ct: usize) -> TensorRef {
        let mut r = TensorRef {
            n: self.channels.len() as u8,
            first: self.first as u32,
            ct: ct as u32,
            ..TensorRef::default()
        };
        for (i, (c, b)) in self.channels.iter().zip(&self.base).enumerate() {
            r.channels[i] = c.index();
            r.base[i] = *b as u32;
        }
        r
    }
    pub(crate) fn borrowed_tiles(&self, first: usize, tiles: usize) -> Self {
        Self {
            first: self.first + first,
            tiles,
            owned: false,
            ..self.clone()
        }
    }
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
            .range(self.base[c] + i * self.slot_bytes, self.slot_bytes)
            .expect("the allocator placed every slot inside its channel")
    }

    pub fn tiles(&self) -> usize {
        self.tiles
    }

    /// The whole of a one-channel placement's slots ([`DramAlloc::alloc_on`]).
    pub(crate) fn region(&self) -> Option<DramRange> {
        match self.channels[..] {
            [c] => c.range(self.base[0], self.slots * self.slot_bytes),
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
            let len = p.slots * p.slot_bytes;
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
            slot_bytes: TILE_SLOT,
            tiles: 1,
            first: 0,
            owned: true,
        })
    }

    /// Room for `tiles` tile slots, interleaved.
    pub fn alloc(&mut self, tiles: usize) -> Result<Placement> {
        self.alloc_slots(tiles, TILE_SLOT)
    }

    /// Format-aware tile allocation. The slot includes its header/alignment;
    /// it must preserve the NoC's 64-byte read alignment on every channel.
    pub(crate) fn alloc_slots(&mut self, tiles: usize, slot_bytes: u64) -> Result<Placement> {
        if slot_bytes == 0 || slot_bytes % tt_isa::dram::ALIGN != 0 {
            return Err(TensorError::Shape("unaligned physical tile slot".into()));
        }
        let n = self.channels.len();
        let slots = (tiles.max(1)).div_ceil(n) as u64;
        let bytes = slots * slot_bytes;
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
            slot_bytes,
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
            release(&mut self.free[i], b, p.slots * p.slot_bytes);
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
        self.placement.tensor_ref(self.grid()[1])
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
        let images = self.tile_images(values)?;
        self.write_images(dev, w, &images)
    }

    /// What [`DramTensor::write_bits`] writes: each tile's image
    /// ([`matmul::TILE_IMAGE_BYTES`], header and datums), in tile order,
    /// after every check a write makes.
    pub fn tile_images(&self, values: &[u32]) -> Result<Vec<u8>> {
        self.check_write(values.len(), Some(values))?;
        let tiles = self.placement.tiles;
        let mut out = vec![0u8; tiles * matmul::TILE_IMAGE_BYTES];
        // The tilizer moves bits: every pattern is kept, NaN payloads included.
        matmul::tilize_into(
            |i| values[i],
            self.rows,
            self.cols,
            0..tiles,
            &mut out,
            matmul::TILE_IMAGE_BYTES,
        );
        Ok(out)
    }

    /// The checks every write makes: `len` datums for this tensor, its own
    /// slots (not a view's), and a `Bool` tensor's `bits` only `0` and `1`.
    pub fn check_write(&self, len: usize, bits: Option<&[u32]>) -> Result<()> {
        if let (Elem::Bool, Some(values)) = (self.elem, bits) {
            if let Some(i) = values.iter().position(|&v| v > 1) {
                return Err(TensorError::Shape(format!(
                    "a Bool tensor's datum {i} is {:#x}, not 0 or 1",
                    values[i]
                )));
            }
        }
        let (rows, cols) = (self.rows, self.cols);
        if len != rows * cols {
            return Err(TensorError::Shape(format!(
                "{len} values for a [{rows}, {cols}] tensor"
            )));
        }
        if !self.placement.owned {
            return Err(TensorError::Shape(
                "a view's slots are another tensor's: write that one".into(),
            ));
        }
        Ok(())
    }

    /// Write [`DramTensor::tile_images`]'s images to their slots from the
    /// host, through the BAR.
    pub fn write_images<T: Transport>(
        &self,
        dev: &mut Device<T>,
        w: &Window,
        images: &[u8],
    ) -> Result<()> {
        let t = self;
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

    /// Tile `k`'s slot, in tile order (row-major over the grid).
    pub fn slot(&self, k: usize) -> DramRange {
        self.placement.slot(k)
    }

    /// Row-major values from tiles' datums packed in tile order, 4 KiB each
    /// (what [`host_dma_jobs`]'s downloads leave in host memory).
    pub fn from_packed(&self, packed: &[u8]) -> Vec<f32> {
        matmul::detilize_packed(packed, self.rows, self.cols)
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

/// Tile slots a unit stages a [`host_dma_jobs`] batch in: two halves of
/// this many, a batch moving in or out of one while the other's moves go on.
pub const HOST_DMA_BATCH: usize = 64;
const _: () = assert!(2 * HOST_DMA_BATCH as u64 * TILE_SLOT <= tt_isa::l1::DATA.len());

/// Moves between host memory and tiles `tiles` of the tensor `x`, by the
/// card (`tt_isa::dm::op::HOST_READ` / `HOST_WRITE`), one job a unit, each
/// unit a contiguous run of the tiles. In host memory tile `k` is at
/// `host + (k - tiles.start) * TILE_SLOT`, its datums `TILE_DATA` in -- the
/// same stride and offset as tile slots in L1, so a batch is one host move of
/// its consecutive slots and one record (`record::READ_RUN` /
/// `record::WRITE_RUN`) for its tiles in GDDR. An upload writes the datums
/// (a slot's header is the unpacker's to skip, as a kernel's outputs'); a
/// download brings whole slots.
///
/// Each unit's tiles go in batches of [`HOST_DMA_BATCH`], alternating
/// halves of its staging: batch `b + 1` comes in while batch `b` goes out. An
/// upload is a [`Step::Transfer`] of depth two (B brings a batch in, NC
/// writes it, and the half batch `b + 2` refills is free once NC's writes are
/// acknowledged); a download is a list on B, a `WAIT` between its batches.
/// The data arena must be free of other work: the lists run behind
/// everything queued before them on the unit, whose lists end with their
/// kernels done.
pub fn host_dma_jobs(
    x: TensorRef,
    tiles: std::ops::Range<usize>,
    upload: bool,
    host: u64,
    units: usize,
) -> Vec<Job> {
    use tt_isa::dm::op;
    let len = tiles.len();
    let units = units.max(1).min(len.max(1));
    let base = tt_isa::l1::DATA.base;
    let at = |b: usize| base + ((b % 2) * HOST_DMA_BATCH) as u64 * TILE_SLOT;
    let host_move = |first: usize, n: usize, l1: u64| {
        let h = host + (first - tiles.start) as u64 * TILE_SLOT;
        let kind = if upload {
            op::HOST_READ
        } else {
            op::HOST_WRITE
        };
        [
            kind,
            h as u32,
            (h >> 32) as u32,
            0,
            l1 as u32,
            (n as u64 * TILE_SLOT) as u32,
            0,
            0,
        ]
    };
    let dram_run = |first: usize, n: usize, l1: u64| -> [[u32; 8]; 3] {
        let head = if upload {
            [
                record::WRITE_RUN,
                first as u32,
                n as u32,
                l1 as u32,
                0,
                0,
                0,
                0,
            ]
        } else {
            [
                record::READ_RUN,
                first as u32,
                n as u32,
                l1 as u32,
                0,
                0,
                0,
                0,
            ]
        };
        [head, x.encode()[0], x.encode()[1]]
    };
    let wait = [op::WAIT, 0, 0, 0, 0, 0, 0, 0];
    (0..units)
        .map(|u| {
            let run = tiles.start + u * len / units..tiles.start + (u + 1) * len / units;
            let batches: Vec<(usize, usize)> = run
                .clone()
                .step_by(HOST_DMA_BATCH)
                .map(|f| (f, HOST_DMA_BATCH.min(run.end - f)))
                .collect();
            // An upload is a transfer: B brings a batch in from the host, NC
            // writes it to GDDR, the halves taking turns.
            if upload {
                return vec![Step::Transfer {
                    what: "host dma upload",
                    depth: 2,
                    batches: batches
                        .iter()
                        .enumerate()
                        .map(|(b, &(f, n))| TransferBatch {
                            read: vec![host_move(f, n, at(b))],
                            write: dram_run(f, n, at(b)).to_vec(),
                        })
                        .collect(),
                }];
            }
            // An upload brings a batch in from the host and writes it to
            // GDDR; a download reads it from GDDR and sends it to the host.
            let bring = |b: usize, entries: &mut Vec<[u32; 8]>| {
                let (f, n) = batches[b];
                if upload {
                    entries.push(host_move(f, n, at(b)));
                } else {
                    entries.extend(dram_run(f, n, at(b)));
                }
            };
            let send = |b: usize, entries: &mut Vec<[u32; 8]>| {
                let (f, n) = batches[b];
                if upload {
                    entries.extend(dram_run(f, n, at(b)));
                } else {
                    entries.push(host_move(f, n, at(b)));
                }
            };
            let mut entries = Vec::new();
            for b in 0..batches.len() {
                if b == 0 {
                    bring(b, &mut entries);
                }
                entries.push(wait);
                send(b, &mut entries);
                if b + 1 < batches.len() {
                    bring(b + 1, &mut entries);
                }
            }
            entries.push(wait);
            vec![Step::List {
                what: if upload {
                    "host dma upload"
                } else {
                    "host dma download"
                },
                entries,
            }]
        })
        .collect()
}

/// Bytes between rows of a row-major tensor in host memory for
/// [`row_major_dma_jobs`]: its row's bytes, rounded up to 64 so every row
/// starts congruent with L1 as the card's DMA requires.
pub fn row_major_stride(cols: usize) -> usize {
    (cols * 4).next_multiple_of(64)
}

/// Moves between a row-major `[rows, cols]` matrix in host memory (rows
/// [`row_major_stride`] bytes apart from `host`) and the tensor `x`'s tiles
/// in GDDR, the tile layout made on the card: one job a unit. The matrix goes
/// in chunks -- a tile row's 32 rows, across up to [`TILIZE_CHUNK`] tile
/// columns -- dealt out to the units in turn, so a short wide tensor still
/// uses them all. An upload brings a chunk's rows into L1 (one host move for
/// a whole band, else one a row), `TILIZE`s each tile into a slot, and
/// `WRITE_RUN`s the slots' datums to GDDR; a download `READ_RUN`s the slots,
/// `UNTILIZE`s them into rows, and sends the rows to the host. Rows and
/// columns past the matrix are zeros going up and never written coming back.
/// The data arena must be free of other work, as for [`host_dma_jobs`].
pub fn row_major_dma_jobs(
    x: TensorRef,
    [rows, cols]: [usize; 2],
    upload: bool,
    host: u64,
    units: usize,
) -> Vec<Job> {
    use tt_isa::dm::{fill, op};
    let (rt, ct) = (rows.div_ceil(32).max(1), cols.div_ceil(32).max(1));
    let stride = row_major_stride(cols) as u64;
    let row_bytes = (cols * 4) as u64;
    // Chunks small enough to spread over the units, and to fit the arena.
    let per_unit = (rt * ct).div_ceil(units.max(1));
    let width = TILIZE_CHUNK.min(ct).min(per_unit.max(1));
    let chunks: Vec<(usize, usize, usize)> = (0..rt)
        .flat_map(|i| {
            (0..ct)
                .step_by(width)
                .map(move |j0| (i, j0, width.min(ct - j0)))
        })
        .collect();
    let units = units.max(1).min(chunks.len());
    let slots = tt_isa::l1::DATA.base;
    let band = slots + (TILIZE_CHUNK as u64) * TILE_SLOT;
    let wait = [op::WAIT, 0, 0, 0, 0, 0, 0, 0];
    (0..units)
        .map(|u| {
            let mut entries = Vec::new();
            let mut batches = Vec::new();
            for &(i, j0, n) in chunks.iter().skip(u).step_by(units) {
                let valid_rows = (rows - 32 * i).min(32);
                // A whole band is one host move, its L1 rows `stride` apart;
                // part of one is a move a row, its L1 rows the chunk's width.
                let whole = j0 == 0 && n == ct;
                let l1_stride = if whole { stride } else { n as u64 * 128 };
                let seg = if whole {
                    stride
                } else {
                    // The last chunk ends at the row's end (past it, the next
                    // row's columns, which a download must not overwrite).
                    (n as u64 * 128).min((row_bytes - j0 as u64 * 128).next_multiple_of(64))
                };
                let host_at = |r: usize| host + (32 * i + r) as u64 * stride + j0 as u64 * 128;
                let host_moves = |entries: &mut Vec<[u32; 8]>| {
                    let kind = if upload {
                        op::HOST_READ
                    } else {
                        op::HOST_WRITE
                    };
                    let mv = |h: u64, l1: u64, len: u64| {
                        [
                            kind,
                            h as u32,
                            (h >> 32) as u32,
                            0,
                            l1 as u32,
                            len as u32,
                            0,
                            0,
                        ]
                    };
                    if whole {
                        entries.push(mv(host_at(0), band, valid_rows as u64 * stride));
                    } else {
                        for r in 0..valid_rows {
                            entries.push(mv(host_at(r), band + r as u64 * l1_stride, seg));
                        }
                    }
                };
                let first = (i * ct + j0) as u32;
                let run = |kind: u32| {
                    [
                        [kind, first, n as u32, slots as u32, 0, 0, 0, 0],
                        x.encode()[0],
                        x.encode()[1],
                    ]
                };
                let layout = |kind: u32, entries: &mut Vec<[u32; 8]>| {
                    for k in 0..n {
                        let j = j0 + k;
                        let valid_cols = (cols - 32 * j).min(32);
                        entries.push([
                            kind,
                            (band + k as u64 * 128) as u32,
                            l1_stride as u32,
                            (slots + k as u64 * TILE_SLOT + TILE_DATA) as u32,
                            fill::param(valid_rows as u32, valid_cols as u32),
                            0,
                            0,
                            0,
                        ]);
                    }
                };
                // Each `TILIZE` / `UNTILIZE` waits for every move before it,
                // so the next chunk's moves into the band and the slots
                // never overtake this one's out of them. An upload's chunk is
                // one transfer batch at depth one: B brings the rows in and
                // tilizes them into the slots, NC writes the slots out, and
                // only then does B bring the next chunk's rows.
                if upload {
                    let mut read = Vec::new();
                    host_moves(&mut read);
                    layout(op::TILIZE, &mut read);
                    batches.push(TransferBatch {
                        read,
                        write: run(record::WRITE_RUN).to_vec(),
                    });
                } else {
                    entries.extend(run(record::READ_RUN));
                    layout(op::UNTILIZE, &mut entries);
                    host_moves(&mut entries);
                }
            }
            if upload {
                return vec![Step::Transfer {
                    what: "host dma upload, tilized on the card",
                    depth: 1,
                    batches,
                }];
            }
            entries.push(wait);
            vec![Step::List {
                what: "host dma download, untilized on the card",
                entries,
            }]
        })
        .collect()
}

/// Most tile columns a [`row_major_dma_jobs`] chunk spans: its slots and its
/// band of rows (32 of them, 128 bytes a tile column) inside the data arena,
/// for any matrix whose padded row fits the band (a wider one goes a chunk
/// at a time, a move a row).
pub const TILIZE_CHUNK: usize = 96;
const _: () = assert!(TILIZE_CHUNK as u64 * (TILE_SLOT + 32 * 128) <= tt_isa::l1::DATA.len());

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
        /// Which half of a double-buffered pair this kernel's block is staged
        /// in (`matmul::Staging::SlotsHalf`), if it is: the session may then
        /// overlap the moves of a unit's next and last blocks with it
        /// (checklist 9.15). `None` runs it as one `KERNEL`.
        half: Option<u8>,
    },
    /// A standalone transfer through L1: B reads each batch in and NC writes
    /// it out, one credit a batch (`tt_isa::dataflow::Stream::Transfer`).
    /// Consecutive transfers of the same depth on a tile share one shared
    /// packet (`dm::op::PAIR`) with no kernel in it, so B only reads and NC
    /// only writes GDDR.
    Transfer {
        /// What it is, for [`stats`].
        what: &'static str,
        /// Batches in flight: 1, the next batch reads only after this one's
        /// writes are acknowledged, or 2, alternating halves of a staging
        /// area. The builder places each batch's slots to match.
        depth: u16,
        batches: Vec<TransferBatch>,
    },
}

/// One credit of a [`Step::Transfer`]: the entries B runs to fill its slots
/// (reads, and what shapes them: host moves, tilizes, padding fills) and the
/// entries NC runs to write them out (`WRITE` entries and writing records).
/// Each side must stay within [`TRANSFER_SIDE_MAX`] entries, so a batch fits a
/// packet.
#[derive(Clone, Debug)]
pub struct TransferBatch {
    pub read: Vec<[u32; 8]>,
    pub write: Vec<[u32; 8]>,
}

/// Most entries either side of a [`TransferBatch`] may hold: with its two
/// credits, a batch's side stays under half a packet (`tt_isa::dm::LIST_MAX`).
pub const TRANSFER_SIDE_MAX: usize = 240;

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

/// Whether a GDDR matmul gains by pipelining (`Staging::SlotsHalf`): `K`
/// still whole in half the arena, at least two blocks a unit to overlap, and
/// no more than twice the operand tiles gathered. Half-arena blocks are
/// smaller, so each output block re-gathers its operands more often; past
/// twice, card 0 found the gathers outweigh the overlap (1024^3 on 8 tiles:
/// 1x2-tile blocks, x2.18 the tiles, 1.3x slower), and below two blocks a
/// unit there is nothing to overlap.
pub fn pipelining_pays(
    [m, k, n]: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    units: usize,
) -> bool {
    let [mt, kt, nt] = [m.div_ceil(32), k.div_ceil(32), n.div_ceil(32)];
    // Tiles gathered over the whole op, and blocks, for a staging.
    let gathered = |st: Staging| -> Option<(usize, usize)> {
        let s = matmul::plan_in([m, k, n], route, fidelity, st)?;
        if s.tiles[1] < kt {
            return None;
        }
        let [mc, nc] = blocks([mt, nt], [s.tiles[0], s.tiles[2]], units);
        let (mut tiles, mut count) = (0, 0);
        for i0 in (0..mt).step_by(mc) {
            for j0 in (0..nt).step_by(nc) {
                tiles += (mc.min(mt - i0) + nc.min(nt - j0)) * kt;
                count += 1;
            }
        }
        Some((tiles, count))
    };
    let (Some((full, _)), Some((half, count))) =
        (gathered(Staging::Slots), gathered(Staging::SlotsHalf))
    else {
        return false;
    };
    count >= 2 * units.max(1) && half <= 2 * full
}

/// `op(A) @ op(B)`, where `op` is a transpose when asked, all in GDDR: the
/// data mover gathers each block's tiles into L1 (transposing where needed),
/// the resident roles compute it, and the mover writes the output tiles back.
/// Nothing but descriptors, programs and semaphores crosses PCIe.
///
/// Chunked by [`matmul::plan_in`]. When `K` fits, the output blocks are made
/// small enough for `units` tiles to share ([`blocks`]). Otherwise each output
/// tile is one [`Job`], continuing its FP32 accumulator across resident K
/// blocks in the original Matrix Unit traversal order.
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
    pipeline: bool,
) -> Result<Work> {
    matmul_dram_limited(
        alloc,
        a,
        a_transposed,
        b,
        b_transposed,
        route,
        fidelity,
        units,
        allow_mop,
        pipeline,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_dram_limited(
    alloc: &mut DramAlloc,
    a: &DramTensor,
    a_transposed: bool,
    b: &DramTensor,
    b_transposed: bool,
    route: SrcRoute,
    fidelity: Fidelity,
    units: usize,
    allow_mop: bool,
    pipeline: bool,
    max_k_tiles: Option<std::num::NonZeroUsize>,
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
    let [mt, kt, nt] = [m.div_ceil(32), k.div_ceil(32), n.div_ceil(32)];
    // Pipelined (checklist 9.15): each block in half the arena, consecutive
    // blocks of a unit alternating halves, so the session can move the next
    // block in and the last one out while one computes -- if `K` still fits
    // whole in half the arena.
    let staging = if pipeline && pipelining_pays([m, k, n], route, fidelity, units) {
        Staging::SlotsHalf
    } else {
        Staging::Slots
    };
    let shape = matmul::plan_in([m, k, n], route, fidelity, staging)
        .ok_or_else(|| TensorError::Shape(format!("[{m}, {k}] @ [{k}, {n}] fits no chunk")))?;
    let [mc, kc, nc] = shape.tiles;
    let kc = max_k_tiles.map_or(kc, |limit| kc.min(limit.get()));
    let kc = if kc < kt {
        k_block_size(kc, route, fidelity)?
    } else {
        kc
    };
    let [mc, nc] = blocks([mt, nt], [mc, nc], units);
    let (ra, rb) = (a.tensor_ref(), b.tensor_ref());
    let c = DramTensor::alloc(alloc, m, n)?;
    let rc = c.tensor_ref();
    let mut jobs = Vec::new();
    let plan = BlockPlan {
        tiles: [mt, kt, nt],
        block: [mc, nc],
        k_block: kc,
        staging,
        units,
        route,
        fidelity,
        allow_mop,
    };
    if let Err(e) = plan.push_jobs(&mut jobs, [ra, rb, rc], [a_transposed, b_transposed]) {
        alloc.free(&c.placement);
        return Err(e);
    }
    Ok(Work { out: c, jobs })
}

/// How one `[m, k, n]` product is cut into jobs: what [`matmul_dram`] and
/// [`matmul_dram_batched`] share.
struct BlockPlan {
    tiles: [usize; 3],
    block: [usize; 2],
    k_block: usize,
    staging: Staging,
    units: usize,
    route: SrcRoute,
    fidelity: Fidelity,
    allow_mop: bool,
}

impl BlockPlan {
    /// One job per `[mc, nc]` output block of `op(A) @ op(B)`, the operands
    /// and the output named by `refs` (`[A, B, C]`, each a tensor or a
    /// tile-aligned block of one: the records read `K` and the output from
    /// the ref's first tile on, at the ref's row stride).
    fn push_jobs(
        &self,
        jobs: &mut Vec<Job>,
        [ra, rb, rc]: [TensorRef; 3],
        [a_transposed, b_transposed]: [bool; 2],
    ) -> Result<()> {
        let [mt, kt, nt] = self.tiles;
        if self.k_block < kt {
            return self.push_k_jobs(jobs, [ra, rb, rc], [a_transposed, b_transposed]);
        }
        let [mc, nc] = self.block;
        let (units, staging) = (self.units, self.staging);
        let (in_fmt, out_fmt) = self.route.formats();
        for i0 in (0..mt).step_by(mc) {
            let rows = mc.min(mt - i0);
            for j0 in (0..nt).step_by(nc) {
                let cols = nc.min(nt - j0);
                let tiles = [rows, kt, cols];
                // A session deals job `j` to unit `j % units`, so a unit's blocks
                // are every `units`-th: they alternate halves by `j / units`.
                let half = (staging == Staging::SlotsHalf)
                    .then(|| ((jobs.len() / units.max(1)) % 2) as u8);
                let layout = match matmul::plan_layout_in(tiles, in_fmt, staging) {
                    Ok(l) if half == Some(1) => l.shifted(matmul::HALF),
                    Ok(l) => l,
                    Err(e) => return Err(e.into()),
                };
                let matmul::Layout {
                    a_at,
                    b_at,
                    outputs,
                    sems,
                    init,
                } = layout;
                let flags =
                    u32::from(a_transposed) | u32::from(b_transposed) << 1 | (kt as u32) << 8;
                let gather = [
                    [
                        record::GATHER,
                        flags,
                        a_at as u32,
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
                // The cache tells the stagings and halves apart: their programs
                // name different addresses.
                let variant = half.map_or(0, |h| 1 + h);
                let (fidelity, allow_mop) = (self.fidelity, self.allow_mop);
                let (roles, mop) = matmul::kernel_programs(
                    tiles,
                    variant,
                    self.route,
                    fidelity,
                    sems,
                    allow_mop,
                    || matmul::matmul_kernel(&outputs, sems, in_fmt, out_fmt, fidelity, allow_mop),
                );
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
                        half,
                    },
                    Step::List {
                        what: "matmul scatter",
                        entries: scatter.to_vec(),
                    },
                ]);
            }
        }
        Ok(())
    }
}

/// Size both first and continuation programs against the resident-cache limit.
pub(crate) fn k_block_size(mut kc: usize, route: SrcRoute, fidelity: Fidelity) -> Result<usize> {
    loop {
        if let Ok((layout, reload)) = matmul::k_block_layout(kc) {
            let fits = [None, Some(reload)].into_iter().all(|r| {
                matmul::k_block_kernel(&layout, r, route, fidelity)
                    .iter()
                    .all(|p| {
                        p.len() <= tt_isa::mailbox::PROGRAM_MAX as usize
                            && p.len() * 4 <= tt_isa::l1::PROGRAM_CACHE.len() as usize / 2
                    })
            });
            if fits {
                return Ok(kc);
            }
        }
        if kc == 1 {
            return Err(TensorError::Shape("no resident K-block matmul fits".into()));
        }
        kc = kc.div_ceil(2);
    }
}

impl BlockPlan {
    fn push_k_jobs(
        &self,
        jobs: &mut Vec<Job>,
        [ra, rb, rc]: [TensorRef; 3],
        transposed: [bool; 2],
    ) -> Result<()> {
        let [mt, kt, nt] = self.tiles;
        // One output tile per job keeps its prior accumulator and every K
        // continuation on the same unit; outputs remain independent.
        for i in 0..mt {
            for j in 0..nt {
                let mut steps = Vec::new();
                for k0 in (0..kt).step_by(self.k_block) {
                    let count = self.k_block.min(kt - k0);
                    let (layout, reload) = matmul::k_block_layout(count)?;
                    let mut ar = ra;
                    let mut br = rb;
                    ar.first += if transposed[0] {
                        (k0 * ar.ct as usize) as u32
                    } else {
                        k0 as u32
                    };
                    br.first += if transposed[1] {
                        k0 as u32
                    } else {
                        (k0 * br.ct as usize) as u32
                    };
                    let flags = u32::from(transposed[0])
                        | u32::from(transposed[1]) << 1
                        | (count as u32) << 8;
                    let mut gather = vec![
                        [
                            record::GATHER,
                            flags,
                            layout.a_at as u32,
                            layout.b_at as u32,
                            i as u32,
                            1,
                            j as u32,
                            1,
                        ],
                        ar.encode()[0],
                        ar.encode()[1],
                        br.encode()[0],
                        br.encode()[1],
                    ];
                    if k0 > 0 {
                        // A WAIT marks this dependent gather's footprint as
                        // conservative: the scheduler waits for NC release,
                        // not merely pack retirement, before reading GDDR.
                        gather.insert(0, [tt_isa::dm::op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
                        gather.extend([
                            [
                                record::READ_RUN,
                                (i * nt + j) as u32,
                                1,
                                reload.prior as u32,
                                0,
                                0,
                                0,
                                0,
                            ],
                            rc.encode()[0],
                            rc.encode()[1],
                        ]);
                    }
                    let variant = if k0 > 0 { 0x81 } else { 0x80 };
                    let (roles, _) = matmul::kernel_programs(
                        [1, count, 1],
                        variant,
                        self.route,
                        self.fidelity,
                        layout.sems,
                        false,
                        || {
                            (
                                matmul::k_block_kernel(
                                    &layout,
                                    (k0 > 0).then_some(reload),
                                    self.route,
                                    self.fidelity,
                                ),
                                [None; 3],
                            )
                        },
                    );
                    steps.push(Step::List {
                        what: "K-block matmul gather",
                        entries: gather,
                    });
                    steps.push(Step::Kernel {
                        roles,
                        init: layout.init,
                        mop: Box::new([None; 3]),
                        loops: Default::default(),
                        half: None,
                    });
                    steps.push(Step::List {
                        what: "K-block matmul scatter",
                        entries: vec![
                            [
                                record::SCATTER,
                                layout.outputs[0].out as u32,
                                TILE_SLOT as u32,
                                i as u32,
                                1,
                                j as u32,
                                1,
                                0,
                            ],
                            rc.encode()[0],
                            rc.encode()[1],
                        ],
                    });
                }
                jobs.push(steps);
            }
        }
        Ok(())
    }
}

/// One operand of a [`matmul_dram_batched`] product: the block of a tensor
/// whose top-left element is `at` (a multiple of 32 on both axes), read
/// transposed or not. Its extent is the product's `[m, k]` or `[k, n]`
/// (transposed: `[k, m]`, `[n, k]`) of the tensor's elements.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub at: [usize; 2],
    pub transposed: bool,
}

/// A block of `extent` elements at `at` in `t`, as a record names it: the
/// tensor's ref with its first tile moved to the block's, the row stride
/// kept. Refused unless the block starts on a tile and, where it does not end
/// on one, ends at the tensor's own edge -- a ragged edge inside the tensor
/// would read its neighbour's elements as padding.
fn block_ref(t: &DramTensor, at: [usize; 2], extent: [usize; 2], what: &str) -> Result<TensorRef> {
    let ([r0, c0], [r, c]) = (at, extent);
    let fits = r0 + r <= t.rows && c0 + c <= t.cols;
    let aligned = r0 % 32 == 0
        && c0 % 32 == 0
        && (r % 32 == 0 || r0 + r == t.rows)
        && (c % 32 == 0 || c0 + c == t.cols);
    if !fits || !aligned {
        return Err(TensorError::Shape(format!(
            "{what}: a [{r}, {c}] block at [{r0}, {c0}] of a [{}, {}] tensor is not whole tiles \
             inside it",
            t.rows, t.cols
        )));
    }
    let mut x = t.tensor_ref();
    x.first += ((r0 / 32) * t.grid()[1] + c0 / 32) as u32;
    Ok(x)
}

/// `op(A_i) @ op(B_i)` for every `i` of `items` -- blocks of `a` and `b`,
/// each `[m, k] @ [k, n]` after its transposes -- as one op: the products'
/// jobs together, each product's output written to rows `i m .. (i + 1) m`
/// of one `[items.len() m, n]` tensor. A batched matmul of resident
/// operands, and of views into them (a head's columns of a projection),
/// with no copy: each block is gathered where it lies. Every product is
/// [`matmul_dram`]'s, bit for bit, including accumulator reloads across K
/// blocks. With more than one item, refused unless `m` is whole tiles (each product's
/// output must start on a tile row).
#[allow(clippy::too_many_arguments)]
pub fn matmul_dram_batched(
    alloc: &mut DramAlloc,
    a: &DramTensor,
    b: &DramTensor,
    items: &[(Block, Block)],
    [m, k, n]: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    units: usize,
    allow_mop: bool,
    pipeline: bool,
) -> Result<Work> {
    matmul_dram_batched_limited(
        alloc,
        a,
        b,
        items,
        [m, k, n],
        route,
        fidelity,
        units,
        allow_mop,
        pipeline,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_dram_batched_limited(
    alloc: &mut DramAlloc,
    a: &DramTensor,
    b: &DramTensor,
    items: &[(Block, Block)],
    [m, k, n]: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    units: usize,
    allow_mop: bool,
    pipeline: bool,
    max_k_tiles: Option<std::num::NonZeroUsize>,
) -> Result<Work> {
    a.expect("a matmul", Elem::F32)?;
    b.expect("a matmul", Elem::F32)?;
    let batch = items.len();
    if batch == 0 || m == 0 || k == 0 || n == 0 {
        return Err(TensorError::Shape(format!(
            "{batch} x [{m}, {k}] @ [{k}, {n}]: nothing to compute"
        )));
    }
    if batch > 1 && m % 32 != 0 {
        return Err(TensorError::Shape(format!(
            "{batch} x [{m}, {k}] @ [{k}, {n}]: a batch needs whole tile rows per product"
        )));
    }
    let [mt, kt, nt] = [m.div_ceil(32), k.div_ceil(32), n.div_ceil(32)];
    let per_item = units.div_ceil(batch).max(1);
    let staging = if pipeline && pipelining_pays([m, k, n], route, fidelity, per_item) {
        Staging::SlotsHalf
    } else {
        Staging::Slots
    };
    let shape = matmul::plan_in([m, k, n], route, fidelity, staging)
        .ok_or_else(|| TensorError::Shape(format!("[{m}, {k}] @ [{k}, {n}] fits no chunk")))?;
    let [mc, kc, nc] = shape.tiles;
    let kc = max_k_tiles.map_or(kc, |limit| kc.min(limit.get()));
    let kc = if kc < kt {
        k_block_size(kc, route, fidelity)?
    } else {
        kc
    };
    let mut refs = Vec::with_capacity(batch);
    for (ba, bb) in items {
        let ea = if ba.transposed { [k, m] } else { [m, k] };
        let eb = if bb.transposed { [n, k] } else { [k, n] };
        refs.push((
            block_ref(a, ba.at, ea, "a batched matmul's left operand")?,
            block_ref(b, bb.at, eb, "a batched matmul's right operand")?,
            [ba.transposed, bb.transposed],
        ));
    }
    let plan = BlockPlan {
        tiles: [mt, kt, nt],
        block: blocks([mt, nt], [mc, nc], per_item),
        k_block: kc,
        staging,
        units,
        route,
        fidelity,
        allow_mop,
    };
    let c = DramTensor::alloc(alloc, batch * m, n)?;
    let mut jobs = Vec::new();
    for (i, (ra, rb, t)) in refs.into_iter().enumerate() {
        let mut rc = c.tensor_ref();
        rc.first += (i * mt * nt) as u32;
        if let Err(e) = plan.push_jobs(&mut jobs, [ra, rb, rc], t) {
            alloc.free(&c.placement);
            return Err(e);
        }
    }
    Ok(Work { out: c, jobs })
}

/// Fewest tiles a pipelined element-wise or reduce run may hold. Each run
/// is a launch and a gather and scatter record of its own: card 0 lost
/// 5-50% splitting ops into runs of 1-4 tiles.
pub const MIN_PIPELINED_RUN: usize = 12;

/// How the runs of an element-wise or reduce op may overlap: the movers moving
/// one run while the roles compute another. More runs cost the host a list
/// each, which a captured op pays once and a replay never, so a trace may
/// overlap where a fresh op may not.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Overlap {
    /// One run a unit, serialized (`Session::set_pipeline(false)`).
    Off,
    /// An op queued now: the host pays for every run.
    Fresh,
    /// An op being captured into a trace.
    Captured,
}

/// Fewest tiles a unit's share of a fresh pipelined op may hold on one unit.
/// Card 0 (`streaming_performance_sweep`, `sfpu_pipeline_sweep`; host end to
/// end, run 1791079325 and after): on one tile every element-wise op or
/// reduction from 49 tiles gained (0.63-0.89), and small ones forced to
/// overlap still did (a max over rows of 64 tiles 0.88).
pub const PIPELINE_SHARE: usize = 48;

/// What each further unit adds to [`PIPELINE_SHARE`], in tiles a unit. The host
/// queues the units' lists one after another (~6 us a unit an op), so with many
/// units a fresh op is the host's, and the extra runs overlap makes cost more
/// than it saves until each unit has a great deal to move. Adds on 8 tiles
/// lost 1.07-1.20 up to 1152 tiles a unit and gained from 1568 (0.97 there,
/// 0.87 at 2048 a unit); on 32 tiles they lost 1.37-1.44 at every size to 1152
/// a unit; reductions on 8 and 32 tiles lost up to 1.28 or broke even.
pub const PIPELINE_SHARE_PER_UNIT: usize = 180;

/// Fewest tiles a unit's share of a captured pipelined op may hold, per unit.
/// A replay pays no host time for its runs. Adds on 8 tiles gained 0.90 at 32
/// tiles a unit and 0.77 at 128; on 32 tiles 0.95 at 32 and 0.93 at 128, and
/// lost 1.016 at 8 a unit, where [`MIN_PIPELINED_RUN`] leaves one run anyway.
pub const CAPTURED_PIPELINE_SHARE: usize = 32;

/// Tiles a unit's share must hold for `overlap` on `units` units, scaled by
/// [`set_pipeline_share_percent`].
pub fn pipeline_share(overlap: Overlap, units: usize) -> usize {
    let share = match overlap {
        Overlap::Off => usize::MAX,
        Overlap::Fresh => PIPELINE_SHARE + PIPELINE_SHARE_PER_UNIT * (units.max(1) - 1),
        Overlap::Captured => CAPTURED_PIPELINE_SHARE,
    };
    if share == usize::MAX {
        return share;
    }
    share * PIPELINE_SHARE_PERCENT.load(std::sync::atomic::Ordering::Relaxed) / 100
}

/// Percent of [`pipeline_share`] that [`pipelined_runs`] asks for; 100 is the
/// constants as they stand. A benchmark lowers it to see what overlap does
/// where the constants say no (`sfpu_pipeline_sweep`,
/// `streaming_performance_sweep`); nothing else sets it.
static PIPELINE_SHARE_PERCENT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(100);

/// Set [`PIPELINE_SHARE_PERCENT`]; returns what it was.
pub fn set_pipeline_share_percent(percent: usize) -> usize {
    PIPELINE_SHARE_PERCENT.swap(percent, std::sync::atomic::Ordering::Relaxed)
}

/// The runs of a pipelined element-wise or reduce op over `len` items of
/// `weight` tiles each, at most `half_max` items a run (half the arena's
/// worth): at least two a unit, so every unit has a run moving while one
/// computes, and the same number on every unit -- the op takes as long as its
/// busiest unit, and one extra run on a few units cost card 0 up to 25%
/// (`exp` over 1024 tiles on 8: 18 runs, two units doing three). `None` where
/// `overlap` is off, that would leave a run under [`MIN_PIPELINED_RUN`] tiles,
/// a unit's share under [`pipeline_share`] tiles, or `half_max` is zero.
pub fn pipelined_runs(
    len: usize,
    weight: usize,
    units: usize,
    half_max: usize,
    overlap: Overlap,
) -> Option<Vec<std::ops::Range<usize>>> {
    let units = units.max(1);
    let share = pipeline_share(overlap, units);
    if half_max == 0 || overlap == Overlap::Off || len * weight < share.saturating_mul(units) {
        return None;
    }
    let parts = len.div_ceil(half_max).div_ceil(units).max(2) * units;
    // Contiguous, as even as integer division allows.
    let r: Vec<_> = (0..parts)
        .map(|p| p * len / parts..(p + 1) * len / parts)
        .collect();
    let shortest = r.iter().map(|r| r.len()).min()?;
    (shortest * weight >= MIN_PIPELINED_RUN).then_some(r)
}

/// Which half of the arena run `j` of a pipelined op is staged in: a unit
/// takes runs `j`, `j + units`, ... in turn, and alternates halves.
fn half_of(j: usize, units: usize) -> u8 {
    ((j / units.max(1)) % 2) as u8
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

/// What an element-wise op ([`sfpu_eltwise`]) computes: a `crate::kind` or
/// `sfpu::ops::kind_sfpu` op, and its scalar where it takes one.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Eltwise {
    pub kind: u32,
    pub scalar: f32,
    /// A second immediate, for the kinds that take two (`CLAMP`'s bounds,
    /// `HARD_SIGMOID`'s slope and offset); `0.0` for the rest.
    pub scalar2: f32,
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
    use crate::kind;
    use crate::sfpu::kernel::Operands;
    use crate::sfpu::ops::{broadcasts, Broadcast};
    let name = || format!("element-wise op {:#x}", op.kind);
    let sig = crate::sfpu::ops::elems(op.kind);
    a.expect(&name(), sig.inputs[0])?;
    if let (Some(b), Some(&want)) = (b, sig.inputs.get(1)) {
        b.expect(&name(), want)?;
    }
    let binary = match crate::sfpu::ops::operands(op.kind) {
        Some(o) => o != Operands::Unary,
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

/// The element type `kind`'s output has.
fn output_elem(kind: u32) -> Elem {
    crate::sfpu::ops::elems(kind).out
}

/// Element-wise `a (op) b` -- or `a (op) scalar`, or a ternary op's `c` --
/// with everything in GDDR, on the SFPU: each run of tiles a job of three steps -- the
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
    overlap: Overlap,
) -> Result<Option<Work>> {
    use crate::sfpu::kernel::Operands;
    use crate::sfpu::ops::Broadcast;
    let (kind, bcast) = broadcast_of(op, a, b)?;
    let op = Eltwise { kind, ..op };
    if crate::sfpu::integer::operation(kind).is_some_and(|(op, _)| op == 15 || op == 16) {
        return sfpu_divide(alloc, op, a, b, bcast).map(Some);
    }
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
            if let Some(&want) = crate::sfpu::ops::elems(kind).inputs.get(2) {
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
    let out = DramTensor::alloc_elem(alloc, a.rows, a.cols, output_elem(op.kind))?;
    let [rt, ct] = a.grid();
    let (ra, ro) = (a.tensor_ref(), out.tensor_ref());
    let rb = b.map(DramTensor::tensor_ref);
    // Pipelined (checklist 9.15): runs in alternating halves of the arena,
    // so a unit's mover moves one run while the roles compute another.
    let piped = pipelined_runs(
        rt * ct,
        1,
        units,
        sfpu_group(op, bcast, operands, true),
        overlap,
    );
    let pipelined = piped.is_some();
    let all = piped.unwrap_or_else(|| runs(rt * ct, units, sfpu_group(op, bcast, operands, false)));
    let mut jobs = Vec::new();
    for (j, run) in all.into_iter().enumerate() {
        let len = run.len();
        let half = pipelined.then(|| half_of(j, units));
        let (layout, roles, loops) = match sfpu_programs(op, bcast, operands, len, half) {
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
                half,
            },
            Step::List {
                what: "sfpu scatter",
                entries: scatter.to_vec(),
            },
        ]);
    }
    Ok(Some(Work { out, jobs }))
}

/// Checked full-width integer division. The packer publishes canonical
/// domain flags; NC validates them before scattering, without host reads.
fn sfpu_divide(
    alloc: &mut DramAlloc,
    op: Eltwise,
    a: &DramTensor,
    b: Option<&DramTensor>,
    bcast: crate::sfpu::ops::Broadcast,
) -> Result<Work> {
    use crate::sfpu::{
        kernel,
        ops::{self, Broadcast},
    };
    use tt_isa::dm::op as mover;
    // Reserve C as an independent status buffer through Requirements. The
    // actual unary/binary role signature never gathers or unpacks a C input.
    let layout = kernel::plan_layout(1, kernel::Operands::Ternary)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let (operands, code) = ops::code2(op.kind, [op.scalar, op.scalar2]).expect("integer division");
    let (roles, loops) = kernel::roles_code_validated(&layout, operands, &code, true);
    let roles = Arc::new(roles);
    let loops = Arc::new(loops);
    let out = DramTensor::alloc_elem(alloc, a.rows, a.cols, Elem::I32)?;
    let [rt, ct] = a.grid();
    let ra = a.tensor_ref().encode();
    let rb = b.map(|t| t.tensor_ref().encode());
    let ro = out.tensor_ref().encode();
    let b_at = layout.b_at.expect("divisor operand");
    let flags_at = layout.c_at.expect("declared domain status slots");
    let mut jobs = Vec::new();
    for tile in 0..rt * ct {
        let read = |at: u64, flags: u32, reference: [[u32; 8]; 2]| {
            vec![
                [
                    record::READ_RUN,
                    tile as u32,
                    1,
                    at as u32,
                    flags,
                    ct as u32,
                    0,
                    0,
                ],
                reference[0],
                reference[1],
            ]
        };
        let mut gather = read(layout.a_at, 0, ra);
        if let Some(rb) = rb {
            let flags = match bcast {
                Broadcast::None => 0,
                Broadcast::Row => 1,
                Broadcast::Col => 2,
            };
            gather.extend(read(b_at, flags, rb));
            if bcast == Broadcast::Row {
                // Materialize this one row within its local tile so the
                // long divide body can use the runner's compact row loop.
                for row in 1..32 {
                    for col in [0, 16] {
                        gather.push([
                            mover::COPY_WORDS,
                            (b_at + TILE_DATA + tt_isa::dm::face_index(0, col) as u64 * 4) as u32,
                            (b_at + TILE_DATA + tt_isa::dm::face_index(row, col) as u64 * 4) as u32,
                            16,
                            4,
                            4,
                            0,
                            0,
                        ]);
                    }
                }
            }
        }
        jobs.push(vec![
            Step::List {
                what: "integer division gather",
                entries: gather,
            },
            Step::Kernel {
                roles: roles.clone(),
                init: layout.init.clone(),
                mop: Box::new([None; 3]),
                loops: loops.clone(),
                half: None,
            },
            Step::List {
                what: "checked integer division scatter",
                entries: vec![
                    [
                        mover::CHECK_FLAGS,
                        flags_at as u32,
                        (a.rows - tile / ct * 32).min(32) as u32,
                        (a.cols - tile % ct * 32).min(32) as u32,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        record::WRITE_RUN,
                        tile as u32,
                        1,
                        layout.out_at as u32,
                        0,
                        0,
                        0,
                        0,
                    ],
                    ro[0],
                    ro[1],
                ],
            },
        ]);
    }
    out.set_pad(Pad::Undefined);
    Ok(Work { out, jobs })
}

type SfpuPrograms = (
    crate::sfpu::kernel::Layout,
    Arc<[Vec<Instruction>; 3]>,
    Arc<[Vec<crate::code::Loop>; 3]>,
);

/// Materialize shape/index constants from descriptor immediates. This is
/// deliberately separate from tensor upload: traces replay the same metadata
/// without a host data write, and movers only fill/copy raw words.
pub(crate) fn metadata(
    alloc: &mut DramAlloc,
    bits: &[u32],
    dims: [usize; 2],
    elem: Elem,
) -> Result<Work> {
    use tt_isa::dm::{fill, op};
    let [rows, cols] = dims;
    if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(bits.len()) {
        return Err(TensorError::Shape("invalid metadata dimensions".into()));
    }
    let mut req = crate::l1::Requirements::new(1);
    let dst = req.scratch("metadata destination", TILE_SLOT, 64, 0..1);
    let constant = req.scratch("metadata immediate", TILE_SLOT, 64, 0..1);
    let plan = req
        .plan(tt_isa::l1::DATA)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let (dst, constant) = (plan.addr(dst), plan.addr(constant));
    let out = DramTensor::alloc_elem(alloc, rows, cols, elem)?;
    let mut jobs = Vec::new();
    let ct = cols.div_ceil(32);
    for tile in 0..out.placement.tiles() {
        let mut positions = Vec::new();
        for r in 0..32.min(rows - tile / ct * 32) {
            for c in 0..32.min(cols - tile % ct * 32) {
                positions.push((
                    bits[(tile / ct * 32 + r) * cols + tile % ct * 32 + c],
                    tt_isa::dm::face_index(r, c),
                ));
            }
        }
        positions.sort_unstable();
        let mut batches = Vec::new();
        let range = out.tile(tile / ct, tile % ct);
        for (batch, chunk) in positions.chunks(200).enumerate() {
            let mut read = Vec::new();
            if batch == 0 {
                read.extend([
                    [op::FILL, 0, fill::param(1, 1), constant as u32, 0, 0, 0, 0],
                    [
                        op::COPY_WORDS,
                        (constant + TILE_DATA + 4) as u32,
                        (dst + TILE_DATA) as u32,
                        1024,
                        0,
                        4,
                        0,
                        0,
                    ],
                ]);
            } else {
                read.push([
                    op::READ,
                    range.channel().index() as u32,
                    0,
                    range.offset() as u32,
                    dst as u32,
                    TILE_SLOT as u32,
                    0,
                    0,
                ]);
            }
            let mut prior = None;
            for &(value, index) in chunk {
                if prior != Some(value) {
                    read.push([
                        op::FILL,
                        value,
                        fill::param(1, 1),
                        constant as u32,
                        0,
                        0,
                        0,
                        0,
                    ]);
                    prior = Some(value);
                }
                read.push([
                    op::COPY_WORDS,
                    (constant + TILE_DATA + 4) as u32,
                    (dst + TILE_DATA + index as u64 * 4) as u32,
                    1,
                    0,
                    4,
                    0,
                    0,
                ]);
            }
            batches.push(TransferBatch {
                read,
                write: vec![[
                    op::WRITE,
                    range.channel().index() as u32,
                    0,
                    (range.offset() + TILE_DATA) as u32,
                    (dst + TILE_DATA) as u32,
                    4096,
                    0,
                    0,
                ]],
            });
        }
        jobs.push(vec![Step::Transfer {
            what: "resident metadata",
            depth: 1,
            batches,
        }]);
    }
    out.set_pad(Pad::Zero);
    Ok(Work { out, jobs })
}

#[cfg(test)]
pub(crate) fn sfpu_group_for_tests(
    kind: u32,
    scalar: f32,
    operands: crate::sfpu::kernel::Operands,
) -> usize {
    let bcast = if kind == crate::kind::ADD_ROW {
        crate::sfpu::ops::Broadcast::Row
    } else {
        crate::sfpu::ops::Broadcast::None
    };
    let kind = if kind == crate::kind::ADD_ROW {
        crate::kind::ADD
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
        false,
    )
}

/// `half`: in half the arena, for a pipelined run.
fn sfpu_group(
    op: Eltwise,
    bcast: crate::sfpu::ops::Broadcast,
    operands: crate::sfpu::kernel::Operands,
    half: bool,
) -> usize {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    // Measuring builds the op's role programs twice over: once per op kind,
    // scalar and broadcast, not once per op.
    type Memo = Mutex<HashMap<(u32, u32, u32, crate::sfpu::ops::Broadcast, bool), usize>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let key = (
        op.kind,
        op.scalar.to_bits(),
        op.scalar2.to_bits(),
        bcast,
        half,
    );
    let memo = MEMO.get_or_init(Default::default);
    if let Some(&g) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return g;
    }
    let g = measure_sfpu_group(op, bcast, operands, half);
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key, g);
    g
}

fn measure_sfpu_group(
    op: Eltwise,
    bcast: crate::sfpu::ops::Broadcast,
    operands: crate::sfpu::kernel::Operands,
    half: bool,
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
    let arena = if half {
        crate::matmul::HALF
    } else {
        tt_isa::l1::DATA.len()
    };
    GROUP
        .min(crate::sfpu::kernel::max_tiles_in(operands, arena))
        .min(by_program)
        .max(1)
}

/// One run's layout and role programs, memoised by op, scalar, length and
/// `half` (the half of the arena a pipelined run is staged in; `None` for the
/// whole arena).
fn sfpu_programs(
    op: Eltwise,
    bcast: crate::sfpu::ops::Broadcast,
    operands: crate::sfpu::kernel::Operands,
    len: usize,
    half: Option<u8>,
) -> Result<SfpuPrograms> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = (
        u32,
        u32,
        u32,
        crate::sfpu::ops::Broadcast,
        usize,
        Option<u8>,
    );
    type Memo = Mutex<HashMap<Key, SfpuPrograms>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let key = (
        op.kind,
        op.scalar.to_bits(),
        op.scalar2.to_bits(),
        bcast,
        len,
        half,
    );
    let memo = MEMO.get_or_init(Default::default);
    if let Some(p) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(p.clone());
    }
    let layout = match half {
        None => crate::sfpu::kernel::plan_layout(len, operands),
        Some(h) => crate::sfpu::kernel::plan_layout_in(len, operands, crate::matmul::half_arena())
            .map(|l| l.shifted(h as u64 * crate::matmul::HALF)),
    }
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
    overlap: Overlap,
) -> Result<Work> {
    a.expect("a reduction", op.elem())?;
    use crate::sfpu::reduce::Axis;
    let per = match axis {
        Axis::Cols => a.grid()[1],
        Axis::Rows => a.grid()[0],
    };
    let valid = match axis {
        Axis::Cols => a.cols,
        Axis::Rows => a.rows,
    } % 32;
    if reduce_group(
        op,
        axis,
        per,
        if valid == 0 { 32 } else { valid as u32 },
        false,
    )
    .is_none()
    {
        return reduce_chunked(alloc, a, op, axis, units);
    }
    let [rt, ct] = a.grid();
    let (outs, per, valid, out) = match axis {
        Axis::Cols => (
            rt,
            ct,
            a.cols % 32,
            DramTensor::alloc_elem(alloc, a.rows, 1, op.elem())?,
        ),
        Axis::Rows => (
            ct,
            rt,
            a.rows % 32,
            DramTensor::alloc_elem(alloc, 1, a.cols, op.elem())?,
        ),
    };
    let valid = if valid == 0 { 32 } else { valid as u32 };
    let group = match reduce_group(op, axis, per, valid, false) {
        Some(g) => g,
        None => {
            alloc.free(&out.placement);
            return Err(TensorError::Shape(format!(
                "a reduction over {per} tiles does not fit one tile's L1 and program slots"
            )));
        }
    };
    let (ra, ro) = (a.tensor_ref(), out.tensor_ref());
    // Pipelined as element-wise runs are ([`sfpu_eltwise`]).
    let piped = {
        let half = reduce_group(op, axis, per, valid, true).unwrap_or(0);
        pipelined_runs(outs, per, units, half, overlap)
    };
    let pipelined = piped.is_some();
    let all = piped.unwrap_or_else(|| runs(outs, units, group));
    let mut jobs = Vec::new();
    for (j, run) in all.into_iter().enumerate() {
        let len = run.len();
        let half = pipelined.then(|| half_of(j, units));
        let (layout, roles) = match reduce_programs(op, axis, len, per, valid, half) {
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
                half,
            },
            Step::List {
                what: "reduce scatter",
                entries: scatter.to_vec(),
            },
        ]);
    }
    Ok(Work { out, jobs })
}

/// Inclusive scan down matrix rows. Each column tile is a job, so its
/// continuations stay on one unit. A WAIT protects every prior-output reload.
pub fn sfpu_scan(
    alloc: &mut DramAlloc,
    a: &DramTensor,
    op: crate::sfpu::scan::ScanOp,
) -> Result<Work> {
    use crate::sfpu::{kernel, scan};
    use kernel::Operands;
    a.expect("a scan", Elem::F32)?;
    let [rt, ct] = a.grid();
    let first_layout =
        kernel::plan_layout(1, Operands::Unary).map_err(|e| TensorError::Shape(e.to_string()))?;
    let next_layout =
        kernel::plan_layout(1, Operands::Binary).map_err(|e| TensorError::Shape(e.to_string()))?;
    let first_roles = Arc::new(kernel::roles(
        &first_layout,
        Operands::Unary,
        &scan::program(op, true),
    ));
    let next_roles = Arc::new(kernel::roles(
        &next_layout,
        Operands::Binary,
        &scan::program(op, false),
    ));
    let out = DramTensor::alloc(alloc, a.rows, a.cols)?;
    let read = |tensor: &DramTensor, tile: usize, at: u64| {
        vec![
            [record::READ_RUN, tile as u32, 1, at as u32, 0, 0, 0, 0],
            tensor.tensor_ref().encode()[0],
            tensor.tensor_ref().encode()[1],
        ]
    };
    let mut jobs = Vec::new();
    for c in 0..ct {
        let mut steps = Vec::new();
        for r in 0..rt {
            let tile = r * ct + c;
            let layout = if r == 0 { &first_layout } else { &next_layout };
            let mut gather = read(a, tile, layout.a_at);
            if r > 0 {
                gather.insert(0, [tt_isa::dm::op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
                gather.extend(read(&out, tile - ct, layout.b_at.expect("continuation B")));
            }
            steps.push(Step::List {
                what: "scan gather",
                entries: gather,
            });
            steps.push(Step::Kernel {
                roles: if r == 0 {
                    first_roles.clone()
                } else {
                    next_roles.clone()
                },
                init: layout.init.clone(),
                mop: Box::new([None; 3]),
                loops: Default::default(),
                half: None,
            });
            steps.push(Step::List {
                what: "scan scatter",
                entries: vec![
                    [
                        record::WRITE_RUN,
                        tile as u32,
                        1,
                        layout.out_at as u32,
                        0,
                        0,
                        0,
                        0,
                    ],
                    out.tensor_ref().encode()[0],
                    out.tensor_ref().encode()[1],
                ],
            });
        }
        jobs.push(steps);
    }
    Ok(Work { out, jobs })
}

/// Long reductions carry the full, unfolded accumulator tile in GDDR.
/// Each output group stays on one unit and reloads only after NC releases
/// the preceding batch. Folding occurs once, in the final chunk.
fn reduce_chunked(
    alloc: &mut DramAlloc,
    a: &DramTensor,
    op: crate::sfpu::reduce::ReduceOp,
    axis: crate::sfpu::reduce::Axis,
    units: usize,
) -> Result<Work> {
    use crate::sfpu::reduce::{continuation_roles, plan_chunk_layout, Axis, ROW_CHUNK};
    let [rt, ct] = a.grid();
    let (outs, per, valid, dims) = match axis {
        Axis::Cols => (rt, ct, a.cols % 32, [a.rows, 1]),
        Axis::Rows => (ct, rt, a.rows % 32, [1, a.cols]),
    };
    let valid = if valid == 0 { 32 } else { valid as u32 };
    let c = plan_chunk_layout(1).map_err(|e| TensorError::Shape(e.to_string()))?;
    let out = DramTensor::alloc_elem(alloc, dims[0], dims[1], op.elem())?;
    let mut jobs = Vec::new();
    // One output tile per job bounds the largest continuation program.
    // All chunks of that output remain on the same unit.
    let _ = units;
    for output in 0..outs {
        let mut steps = Vec::new();
        for first in (0..per).step_by(ROW_CHUNK) {
            let count = ROW_CHUNK.min(per - first);
            let last = first + count == per;
            let mut gather = Vec::new();
            if first > 0 {
                // Reload depends on the preceding GDDR write, even though
                // its L1 destination is separate from the packer's output.
                gather.push([tt_isa::dm::op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
                gather.extend([
                    [
                        record::READ_RUN,
                        output as u32,
                        1,
                        c.prior_at as u32,
                        0,
                        0,
                        0,
                        0,
                    ],
                    out.tensor_ref().encode()[0],
                    out.tensor_ref().encode()[1],
                ]);
            }
            gather.extend([
                [
                    record::READ_RUN,
                    (output * per + first) as u32,
                    count as u32,
                    c.layout.in_at as u32,
                    if axis == Axis::Rows { 4 } else { 0 },
                    ct as u32,
                    if axis == Axis::Rows { rt as u32 } else { 0 },
                    0,
                ],
                a.tensor_ref().encode()[0],
                a.tensor_ref().encode()[1],
            ]);
            steps.push(Step::List {
                what: "reduce continuation gather",
                entries: gather,
            });
            steps.push(Step::Kernel {
                roles: Arc::new(continuation_roles(
                    &c,
                    op,
                    axis,
                    first > 0,
                    count,
                    if last { valid } else { 32 },
                    last,
                )),
                init: c.layout.init.clone(),
                mop: Box::new([None; 3]),
                loops: Default::default(),
                half: None,
            });
            steps.push(Step::List {
                what: "reduce continuation scatter",
                entries: vec![
                    [
                        record::WRITE_RUN,
                        output as u32,
                        1,
                        c.layout.out_at as u32,
                        0,
                        0,
                        0,
                        0,
                    ],
                    out.tensor_ref().encode()[0],
                    out.tensor_ref().encode()[1],
                ],
            });
        }
        jobs.push(steps);
    }
    Ok(Work { out, jobs })
}

type ReducePrograms = (crate::sfpu::reduce::Layout, Arc<[Vec<Instruction>; 3]>);

/// Most output tiles one reduce run may take, from the slots the data arena
/// (half of it, if `half`) holds and the role programs' length per output
/// tile; `None` if not even one fits.
fn reduce_group(
    op: crate::sfpu::reduce::ReduceOp,
    axis: crate::sfpu::reduce::Axis,
    per: usize,
    valid: u32,
    half: bool,
) -> Option<usize> {
    use crate::sfpu::reduce::{Axis, ReduceOp};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Memo = Mutex<HashMap<(ReduceOp, Axis, usize, u32, bool), Option<usize>>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let key = (op, axis, per, valid, half);
    let memo = MEMO.get_or_init(Default::default);
    if let Some(&g) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return g;
    }
    let g = measure_reduce_group(op, axis, per, valid, half);
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
    half: bool,
) -> Option<usize> {
    use crate::sfpu::reduce::{math_programs, plan_layout, roles};
    const GROUP: usize = 64;
    let arena = if half {
        crate::matmul::HALF
    } else {
        tt_isa::l1::DATA.len()
    };
    let slots = (arena / TILE_SLOT) as usize;
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

/// One run's layout and role programs; `half` as for [`sfpu_programs`].
fn reduce_programs(
    op: crate::sfpu::reduce::ReduceOp,
    axis: crate::sfpu::reduce::Axis,
    len: usize,
    per: usize,
    valid: u32,
    half: Option<u8>,
) -> Result<ReducePrograms> {
    use crate::sfpu::reduce::{math_programs, plan_layout, plan_layout_in, roles, Axis, ReduceOp};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = (ReduceOp, Axis, usize, usize, u32, Option<u8>);
    type Memo = Mutex<HashMap<Key, ReducePrograms>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let key = (op, axis, len, per, valid, half);
    let memo = MEMO.get_or_init(Default::default);
    if let Some(p) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(p.clone());
    }
    let layout = match half {
        None => plan_layout(len, per),
        Some(h) => plan_layout_in(len, per, crate::matmul::half_arena())
            .map(|l| l.shifted(h as u64 * crate::matmul::HALF)),
    }
    .map_err(|e| TensorError::Shape(e.to_string()))?;
    let (inputs, fin) = math_programs(op, axis, per, valid);
    let p = (layout.clone(), Arc::new(roles(&layout, &inputs, &fin)));
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key, p.clone());
    Ok(p)
}

/// The sum over rows of `a`, as a `[1, cols]` tensor, in `burn-flex`'s order:
/// from `+0.0`, adding rows in order, on the SFPU
/// (`sfpu::reduce::accumulate_in_order`).
///
/// One run per group of output tiles when a column of `a`'s tiles fits one
/// ([`sfpu_reduce`]); otherwise in chunks of `ROW_CHUNK` row tiles, each
/// starting from the last's sums, all of a group's chunks one job on one unit.
pub fn sum_rows(
    alloc: &mut DramAlloc,
    a: &DramTensor,
    units: usize,
    overlap: Overlap,
) -> Result<Work> {
    use crate::sfpu::reduce::{chunk_roles, plan_chunk_layout, Axis, ReduceOp, ROW_CHUNK};
    a.expect("a sum over rows", Elem::F32)?;
    let [rt, ct] = a.grid();
    let valid = match (a.rows % 32) as u32 {
        0 => 32,
        v => v,
    };
    if reduce_group(ReduceOp::Sum, Axis::Rows, rt, valid, false).is_some() {
        return sfpu_reduce(alloc, a, ReduceOp::Sum, Axis::Rows, units, overlap);
    }
    // Longer columns stay plain: each chunk gathers the sums the chunk before
    // scattered, which a pipelined list would gather before they land.
    let group = chunk_group().ok_or_else(|| {
        TensorError::Shape("a chunk of a sum over rows does not fit one tile".into())
    })?;
    let out = DramTensor::alloc(alloc, 1, a.cols)?;
    let (ra, ro) = (a.tensor_ref(), out.tensor_ref());
    let mut jobs = Vec::new();
    for run in runs(ct, units, group) {
        let len = run.len();
        let c = match plan_chunk_layout(len) {
            Ok(c) => c,
            Err(e) => {
                alloc.free(&out.placement);
                return Err(TensorError::Shape(e.to_string()));
            }
        };
        let mut steps = Vec::new();
        for (i, r0) in (0..rt).step_by(ROW_CHUNK).enumerate() {
            let tiles = ROW_CHUNK.min(rt - r0);
            let last_valid = if r0 + tiles == rt { valid } else { 32 };
            let mut gather = Vec::new();
            if i > 0 {
                // Require NC release before reloading the prior sums.
                gather.push([tt_isa::dm::op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
                // The last chunk's sums, read back from the output.
                gather.extend([
                    [
                        record::READ_RUN,
                        run.start as u32,
                        len as u32,
                        c.prior_at as u32,
                        0,
                        ct as u32,
                        0,
                        0,
                    ],
                    ro.encode()[0],
                    ro.encode()[1],
                ]);
            }
            for k in 0..len {
                let at = c.layout.in_at + (k * ROW_CHUNK) as u64 * TILE_SLOT;
                gather.extend([
                    [
                        record::READ_RUN,
                        ((run.start + k) * rt + r0) as u32,
                        tiles as u32,
                        at as u32,
                        4,
                        ct as u32,
                        rt as u32,
                        0,
                    ],
                    ra.encode()[0],
                    ra.encode()[1],
                ]);
            }
            steps.push(Step::List {
                what: "sum gather",
                entries: gather,
            });
            steps.push(Step::Kernel {
                roles: Arc::new(chunk_roles(&c, i > 0, tiles, last_valid)),
                init: c.layout.init.clone(),
                mop: Box::new([None; 3]),
                loops: Default::default(),
                half: None,
            });
            steps.push(Step::List {
                what: "sum scatter",
                entries: vec![
                    [
                        record::WRITE_RUN,
                        run.start as u32,
                        len as u32,
                        c.layout.out_at as u32,
                        0,
                        0,
                        0,
                        0,
                    ],
                    ro.encode()[0],
                    ro.encode()[1],
                ],
            });
        }
        jobs.push(steps);
    }
    Ok(Work { out, jobs })
}

/// Most output tiles one chunk of a long sum over rows may take: by the
/// arena's slots (a prior, `ROW_CHUNK` inputs and an output each) and by the
/// role programs' length; `None` if not even one fits.
fn chunk_group() -> Option<usize> {
    use crate::sfpu::reduce::{chunk_roles, plan_chunk_layout, ROW_CHUNK};
    static MEMO: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *MEMO.get_or_init(|| {
        let slots = (tt_isa::l1::DATA.len() / TILE_SLOT) as usize;
        let by_slots = slots / (ROW_CHUNK + 2);
        let lens = |n: usize| -> Option<[usize; 3]> {
            let c = plan_chunk_layout(n).ok()?;
            Some(chunk_roles(&c, true, ROW_CHUNK, 32).map(|p| p.len()))
        };
        let max = tt_isa::mailbox::PROGRAM_MAX as usize;
        let one = lens(1)?;
        if by_slots == 0 || one.iter().any(|&l| l > max) {
            return None;
        }
        let two = lens(2)?;
        let by_program = (0..3)
            .map(|r| {
                let per_out = (two[r] - one[r]).max(1);
                (max - (one[r] - per_out)) / per_out
            })
            .min()
            .unwrap();
        Some(64.min(by_slots).min(by_program).max(1))
    })
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
            SQRT | EXPM1 | TANH | ERF | GELU | SINH | ASINH | ATANH | SIN | TAN | ATAN | ASIN => {
                true
            }
            // `0^s = 0` for `s > 0`.
            POW_S => s > 0.0,
            // `g (1/2)` and `g 0 1`: zero with the gradient's padding.
            GELU_BACKWARD | SIGMOID_BACKWARD | LOG_SIGMOID_BACKWARD => b_zero,
            // `atan2(+0, +0) = +0`.
            ATAN2 => b_zero,
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
        use crate::kind;
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

/// `t`, bit for bit, into a tensor of its own: every tile read whole into L1
/// and its datums written out (`record::READ_RUN`, `record::WRITE_RUN`). Data
/// movement only, so any element type, denormals and NaN payloads included.
/// What a view needs before it can be refilled without touching its parent.
/// Its padding is the source tiles' -- the parent's data, for a view -- so
/// undefined unless `t`'s is zero.
pub fn copy(alloc: &mut DramAlloc, t: &DramTensor, units: usize) -> Result<Work> {
    const GROUP: usize = 128;
    let stage = staging("copy slots", GROUP)?;
    let out = DramTensor::alloc_elem(alloc, t.rows, t.cols, t.elem)?;
    let [rt, ct] = t.grid();
    let (rs, ro) = (t.tensor_ref(), out.tensor_ref());
    let jobs = runs(rt * ct, units, GROUP)
        .into_iter()
        .map(|run| {
            let (first, count) = (run.start as u32, run.len() as u32);
            // B reads the run into the slots; NC writes them out once the reads
            // have landed (the batch's credit).
            vec![Step::Transfer {
                what: "copy list",
                depth: 1,
                batches: vec![TransferBatch {
                    read: vec![
                        [
                            record::READ_RUN,
                            first,
                            count,
                            stage as u32,
                            0,
                            ct as u32,
                            0,
                            0,
                        ],
                        rs.encode()[0],
                        rs.encode()[1],
                    ],
                    write: vec![
                        [record::WRITE_RUN, first, count, stage as u32, 0, 0, 0, 0],
                        ro.encode()[0],
                        ro.encode()[1],
                    ],
                }],
            }]
        })
        .collect();
    Ok(Work { out, jobs })
}

/// Copy `src`, bit for bit, into an existing allocated tensor `dst`.
/// Both tensors must have matching dimensions and element type.
pub fn copy_into(src: &DramTensor, dst: &DramTensor, units: usize) -> Result<Work> {
    if (src.rows, src.cols, src.elem) != (dst.rows, dst.cols, dst.elem) {
        return Err(TensorError::Shape(format!(
            "copy_into mismatched shapes/elem: src [{}, {}] {:?}, dst [{}, {}] {:?}",
            src.rows, src.cols, src.elem, dst.rows, dst.cols, dst.elem
        )));
    }
    const GROUP: usize = 128;
    let stage = staging("copy slots", GROUP)?;
    let [rt, ct] = src.grid();
    let (rs, ro) = (src.tensor_ref(), dst.tensor_ref());
    let jobs = runs(rt * ct, units, GROUP)
        .into_iter()
        .map(|run| {
            let (first, count) = (run.start as u32, run.len() as u32);
            vec![Step::Transfer {
                what: "copy_into list",
                depth: 1,
                batches: vec![TransferBatch {
                    read: vec![
                        [
                            record::READ_RUN,
                            first,
                            count,
                            stage as u32,
                            0,
                            ct as u32,
                            0,
                            0,
                        ],
                        rs.encode()[0],
                        rs.encode()[1],
                    ],
                    write: vec![
                        [record::WRITE_RUN, first, count, stage as u32, 0, 0, 0, 0],
                        ro.encode()[0],
                        ro.encode()[1],
                    ],
                }],
            }]
        })
        .collect();
    Ok(Work {
        out: dst.clone(),
        jobs,
    })
}

/// One block a [`copy_blocks`] moves: `extent` elements of the output at
/// `to`, from the source's block at `from` -- of the same extent, or with
/// `transposed`, of the transposed extent, read as its transpose. Both
/// tile-aligned.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BlockMove {
    pub from: [usize; 2],
    pub to: [usize; 2],
    pub extent: [usize; 2],
    pub transposed: bool,
}

/// A new `dims` tensor of `t`'s type, assembled from blocks of `t`, bit for
/// bit: a reshape or permute that moves whole tiles -- a head's columns into
/// its own rows, and back, or a transpose of whole tiles -- as one copy. Each
/// block is read where it lies ([`record::READ_RUN`] over the block's grid,
/// transposing each tile if asked) and written where it goes
/// ([`record::WRITE_RUN`] over the same grid, on the output's ref moved to
/// the block). Refused unless every block is whole tiles inside both tensors
/// (a ragged edge only at a tensor's own edge) and the blocks cover the
/// output exactly once.
pub fn copy_blocks(
    alloc: &mut DramAlloc,
    t: &DramTensor,
    moves: &[BlockMove],
    [rows, cols]: [usize; 2],
    units: usize,
) -> Result<Work> {
    const GROUP: usize = 128;
    let out = DramTensor::alloc_elem(alloc, rows, cols, t.elem)?;
    let checked = (|| {
        let [ort, oct] = out.grid();
        let mut covered = vec![false; ort * oct];
        let mut refs = Vec::with_capacity(moves.len());
        for m in moves {
            let src_extent = if m.transposed {
                [m.extent[1], m.extent[0]]
            } else {
                m.extent
            };
            let rs = block_ref(t, m.from, src_extent, "a block copy's source")?;
            let ro = block_ref(&out, m.to, m.extent, "a block copy's destination")?;
            let [r0, c0] = [m.to[0] / 32, m.to[1] / 32];
            for i in r0..r0 + m.extent[0].div_ceil(32) {
                for j in c0..c0 + m.extent[1].div_ceil(32) {
                    if std::mem::replace(&mut covered[i * oct + j], true) {
                        return Err(TensorError::Shape(format!(
                            "a block copy writes output tile ({i}, {j}) twice"
                        )));
                    }
                }
            }
            refs.push((rs, ro, m.extent, m.transposed));
        }
        if let Some(n) = covered.iter().position(|c| !c) {
            return Err(TensorError::Shape(format!(
                "a block copy leaves output tile ({}, {}) unwritten",
                n / oct,
                n % oct
            )));
        }
        Ok(refs)
    })();
    let refs = match checked {
        Ok(r) => r,
        Err(e) => {
            alloc.free(&out.placement);
            return Err(e);
        }
    };
    let stage = staging("copy slots", GROUP)?;
    let per = units.div_ceil(moves.len().max(1)).max(1);
    let mut jobs = Vec::new();
    for (rs, ro, [r, c], transposed) in refs {
        let ct = c.div_ceil(32);
        let flags = if transposed { 8 } else { 0 };
        for run in runs(r.div_ceil(32) * ct, per, GROUP) {
            let (first, count) = (run.start as u32, run.len() as u32);
            jobs.push(vec![Step::Transfer {
                what: "block copy list",
                depth: 1,
                batches: vec![TransferBatch {
                    read: vec![
                        [
                            record::READ_RUN,
                            first,
                            count,
                            stage as u32,
                            flags,
                            ct as u32,
                            0,
                            0,
                        ],
                        rs.encode()[0],
                        rs.encode()[1],
                    ],
                    write: vec![
                        [
                            record::WRITE_RUN,
                            first,
                            count,
                            stage as u32,
                            0,
                            ct as u32,
                            0,
                            0,
                        ],
                        ro.encode()[0],
                        ro.encode()[1],
                    ],
                }],
            }]);
        }
    }
    Ok(Work { out, jobs })
}

/// Repack logical elements on the card, preserving every bit. `sources`
/// names a source coordinate for each output element in row-major order.
/// Only address metadata is built on the host; B copies words after aligned
/// tile reads and NC writes complete output tiles. Large maps are split into
/// ordered transfer batches, reloading the partially assembled output tile.
pub fn repack(
    alloc: &mut DramAlloc,
    t: &DramTensor,
    sources: &[[usize; 2]],
    [rows, cols]: [usize; 2],
) -> Result<Work> {
    let mapping: Vec<_> = sources.iter().copied().map(|at| (0, at)).collect();
    repack_many(alloc, &[t], &mapping, [rows, cols])
}

/// Assemble raw datums from multiple resident tensors using static geometry.
pub fn repack_many(
    alloc: &mut DramAlloc,
    inputs: &[&DramTensor],
    sources: &[(usize, [usize; 2])],
    [rows, cols]: [usize; 2],
) -> Result<Work> {
    use tt_isa::dm::{face_index, op, TILE_DATA};
    let t = inputs
        .first()
        .ok_or_else(|| TensorError::Shape("repack has no inputs".into()))?;
    if inputs.iter().any(|x| x.elem != t.elem)
        || rows == 0
        || cols == 0
        || rows.checked_mul(cols) != Some(sources.len())
        || sources
            .iter()
            .any(|&(i, [r, c])| inputs.get(i).is_none_or(|x| r >= x.rows || c >= x.cols))
    {
        return Err(TensorError::Shape("invalid native repack mapping".into()));
    }
    let mut req = crate::l1::Requirements::new(1);
    let input = req.scratch("repack source tile", TILE_SLOT, 64, 0..1);
    let output = req.scratch("repack destination tile", TILE_SLOT, 64, 0..1);
    let layout = req
        .plan(tt_isa::l1::DATA)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let (input, output) = (layout.addr(input), layout.addr(output));
    let out = DramTensor::alloc_elem(alloc, rows, cols, t.elem)?;
    let [rt, ct] = out.grid();
    let mut jobs = Vec::new();
    for i in 0..rt {
        for j in 0..ct {
            let mut tiles =
                std::collections::BTreeMap::<(usize, usize), Vec<(usize, usize)>>::new();
            for r in 0..32.min(rows - i * 32) {
                for c in 0..32.min(cols - j * 32) {
                    let (source, [sr, sc]) = sources[(i * 32 + r) * cols + j * 32 + c];
                    tiles
                        .entry((source, sr / 32 * inputs[source].grid()[1] + sc / 32))
                        .or_default()
                        .push((face_index(sr % 32, sc % 32), face_index(r, c)));
                }
            }
            let range = out.tile(i, j);
            let read_prior = [
                op::READ,
                range.channel().index() as u32,
                0,
                range.offset() as u32,
                output as u32,
                TILE_SLOT as u32,
                0,
                0,
            ];
            let mut batches = Vec::new();
            let mut read = Vec::new();
            let flush = |read: &mut Vec<[u32; 8]>, batches: &mut Vec<TransferBatch>| {
                if read.is_empty() {
                    return;
                }
                batches.push(TransferBatch {
                    read: std::mem::take(read),
                    write: vec![
                        [
                            record::WRITE_RUN,
                            (i * ct + j) as u32,
                            1,
                            output as u32,
                            0,
                            0,
                            0,
                            0,
                        ],
                        out.tensor_ref().encode()[0],
                        out.tensor_ref().encode()[1],
                    ],
                });
            };
            for ((source, tile), mut words) in tiles {
                let t = inputs[source];
                words.sort_unstable();
                let src = t.tile(tile / t.grid()[1], tile % t.grid()[1]);
                let read_src = [
                    op::READ,
                    src.channel().index() as u32,
                    0,
                    src.offset() as u32,
                    input as u32,
                    TILE_SLOT as u32,
                    0,
                    0,
                ];
                let mut n = 0;
                while n < words.len() {
                    if read.len() >= 240 {
                        flush(&mut read, &mut batches);
                        read.push(read_prior);
                    }
                    read.push(read_src);
                    // Consecutive words commonly cover a face-row. Merge
                    // them without assuming a tile-coherent logical view.
                    while n < words.len() && read.len() < 240 {
                        let (sr, dst) = words[n];
                        let mut count = 1;
                        while n + count < words.len()
                            && words[n + count] == (sr + count, dst + count)
                        {
                            count += 1;
                        }
                        read.push([
                            op::COPY_WORDS,
                            (input + TILE_DATA + sr as u64 * 4) as u32,
                            (output + TILE_DATA + dst as u64 * 4) as u32,
                            count as u32,
                            4,
                            4,
                            0,
                            0,
                        ]);
                        n += count;
                    }
                }
            }
            flush(&mut read, &mut batches);
            jobs.push(vec![Step::Transfer {
                what: "native repack",
                depth: 1,
                batches,
            }]);
        }
    }
    Ok(Work { out, jobs })
}

/// A new `[rows.len(), cols]` tensor whose row `i` is row `rows[i].1` of
/// `sources[rows[i].0]`, bit for bit: an embedding's lookup, and the rows a
/// gradient scatters, gathered on the card.
///
/// A tile's row is two 64-byte face-rows (`TILE_DATA + ((r / 16) 2 + h)
/// 1024 + (r % 16) 64`, `h` the half), each congruent to `TILE_DATA` mod 64
/// in every slot, so any source row's face-row moves straight into any
/// output row's with one 64-byte [`dm::op::READ`] -- the mod-64 congruence a
/// GDDR read needs (`tt_isa::dram::ALIGN`), whatever the rows. Output tiles
/// are built in staging slots (64-aligned) by B, then written out by NC, a few
/// tiles to a batch ([`Step::Transfer`]). Two read entries a row a tile
/// column, ~316 cycles each on the mover. Rows past the last in the last tile row are left as the
/// slot held them: the output's padding is undefined there.
pub fn gather_rows(
    alloc: &mut DramAlloc,
    sources: &[&DramTensor],
    rows: &[(usize, usize)],
    cols: usize,
    units: usize,
) -> Result<Work> {
    use tt_isa::dm::{op, TILE_DATA};
    const GROUP: usize = 64;
    let Some(first) = sources.first() else {
        return Err(TensorError::Shape("a row gather with no source".into()));
    };
    let elem = first.elem;
    for (k, t) in sources.iter().enumerate() {
        if t.cols != cols || t.elem != elem {
            return Err(TensorError::Shape(format!(
                "a row gather's source {k} is [{}, {}] {:?}, not [_, {cols}] {elem:?}",
                t.rows, t.cols, t.elem
            )));
        }
    }
    if rows.is_empty() || cols == 0 {
        return Err(TensorError::Shape("a row gather of nothing".into()));
    }
    if let Some((i, &(k, r))) = rows
        .iter()
        .enumerate()
        .find(|(_, &(k, r))| k >= sources.len() || r >= sources[k].rows)
    {
        return Err(TensorError::Shape(format!(
            "a row gather's row {i} names row {r} of source {k}, which has none"
        )));
    }
    let stage = staging("row gather slots", GROUP)?;
    if stage % tt_isa::dram::ALIGN != 0 {
        return Err(TensorError::Shape(
            "row gather staging not 64-aligned".into(),
        ));
    }
    let out = DramTensor::alloc_elem(alloc, rows.len(), cols, elem)?;
    let [ort, oct] = out.grid();
    let face_row =
        |r: usize, h: usize| TILE_DATA + (((r / 16) * 2 + h) * 1024 + (r % 16) * 64) as u64;
    // Output tiles go in batches of `PER_BATCH`: its 64 face-row reads a tile
    // must fit a packet's side, and two batches alternate halves of the
    // staging so NC writes one out while B reads the next.
    const PER_BATCH: usize = (TRANSFER_SIDE_MAX - 2) / 64;
    const _: () = assert!(2 * PER_BATCH <= GROUP);
    let mut jobs = Vec::new();
    for run in runs(ort * oct, units, GROUP) {
        let tiles: Vec<usize> = run.collect();
        let batches = tiles
            .chunks(PER_BATCH)
            .enumerate()
            .map(|(b, batch)| {
                let first_slot = stage + (b % 2 * PER_BATCH) as u64 * TILE_SLOT;
                let mut read = Vec::new();
                let mut write = Vec::new();
                let mut n = 0u32;
                for (k, &t) in batch.iter().enumerate() {
                    let (oi, oj) = (t / oct, t % oct);
                    let slot = first_slot + k as u64 * TILE_SLOT;
                    for r in 0..32.min(rows.len() - oi * 32) {
                        let (src, sr) = rows[oi * 32 + r];
                        let from = sources[src].tile(sr / 32, oj);
                        for h in 0..2 {
                            read.push([
                                op::READ,
                                from.channel().index() as u32,
                                n % tt_isa::dram::PORTS as u32,
                                (from.offset() + face_row(sr % 32, h)) as u32,
                                (slot + face_row(r, h)) as u32,
                                64,
                                0,
                                0,
                            ]);
                            n += 1;
                        }
                    }
                    let to = out.tile(oi, oj);
                    write.push([
                        op::WRITE,
                        to.channel().index() as u32,
                        k as u32 % tt_isa::dram::PORTS as u32,
                        (to.offset() + TILE_DATA) as u32,
                        (slot + TILE_DATA) as u32,
                        4096,
                        0,
                        0,
                    ]);
                }
                TransferBatch { read, write }
            })
            .collect();
        jobs.push(vec![Step::Transfer {
            what: "row gather list",
            depth: 2,
            batches,
        }]);
    }
    Ok(Work { out, jobs })
}

/// Rows of `src` written over rows of `dst` in place: `dst` row `d` becomes
/// `src` row `s` for each `(d, s)` of `rows` (each `d` at most once), every
/// other row untouched -- how a scatter of rows (an embedding's gradient)
/// lands in a copy of its table without moving the rest. Each face-row is
/// read into a staging slot at its own offset in a tile (64-byte reads, as
/// [`gather_rows`]'s) by B and written straight to its place in `dst` by NC
/// (64-byte writes: the mod-16 congruence a write needs holds), a batch of
/// them to a credit ([`Step::Transfer`]). The jobs only: the session runs them on `dst` itself.
pub fn write_rows(
    dst: &DramTensor,
    src: &DramTensor,
    rows: &[(usize, usize)],
    units: usize,
) -> Result<Vec<Job>> {
    use tt_isa::dm::{op, TILE_DATA};
    const GROUP: usize = 64;
    if !dst.placement.owned {
        return Err(TensorError::Shape(
            "a view's slots are another tensor's: write that one".into(),
        ));
    }
    if dst.cols != src.cols || dst.elem != src.elem {
        return Err(TensorError::Shape(format!(
            "rows of a [{}, {}] {:?} over a [{}, {}] {:?}",
            src.rows, src.cols, src.elem, dst.rows, dst.cols, dst.elem
        )));
    }
    let mut seen = std::collections::HashSet::new();
    for &(d, s) in rows {
        if d >= dst.rows || s >= src.rows || !seen.insert(d) {
            return Err(TensorError::Shape(format!(
                "a row write of row {s} over row {d}: out of range or twice"
            )));
        }
    }
    let stage = staging("row write slots", GROUP)?;
    let ct = dst.grid()[1];
    let face_row =
        |r: usize, h: usize| TILE_DATA + (((r / 16) * 2 + h) * 1024 + (r % 16) * 64) as u64;
    // One staging slot per (row, tile column) pair, its face-rows at the
    // offsets they have in the destination tile: `GROUP` slots hold `32
    // GROUP` such rows, one per slot row.
    let per_slot = 32;
    let pieces: Vec<(usize, usize, usize)> = rows
        .iter()
        .flat_map(|&(d, s)| (0..ct).map(move |j| (d, s, j)))
        .collect();
    // Batches of `PER_BATCH` pieces (two reads and two writes each, inside a
    // packet's side), alternating halves of the staging.
    const PER_BATCH: usize = TRANSFER_SIDE_MAX / 2 - 1;
    let slots_per_batch = PER_BATCH.div_ceil(per_slot);
    const _: () = assert!(2 * PER_BATCH.div_ceil(32) <= GROUP);
    let mut jobs = Vec::new();
    for run in runs(pieces.len(), units, GROUP * per_slot) {
        let batches = pieces[run]
            .chunks(PER_BATCH)
            .enumerate()
            .map(|(b, batch)| {
                let half = stage + (b % 2 * slots_per_batch) as u64 * TILE_SLOT;
                let mut read = Vec::new();
                let mut write = Vec::new();
                for (k, &(d, s, j)) in batch.iter().enumerate() {
                    // Piece `k` is staged at row `k % 32` of slot `k / 32`: its
                    // own place, and every face-row offset is `TILE_DATA` mod 64
                    // wherever it is, so the read's and the write's congruences
                    // hold.
                    let slot = half + (k / per_slot) as u64 * TILE_SLOT;
                    let r = k % per_slot;
                    let from = src.tile(s / 32, j);
                    let to = dst.tile(d / 32, j);
                    for h in 0..2 {
                        read.push([
                            op::READ,
                            from.channel().index() as u32,
                            (2 * k + h) as u32 % tt_isa::dram::PORTS as u32,
                            (from.offset() + face_row(s % 32, h)) as u32,
                            (slot + face_row(r, h)) as u32,
                            64,
                            0,
                            0,
                        ]);
                        write.push([
                            op::WRITE,
                            to.channel().index() as u32,
                            (2 * k + h) as u32 % tt_isa::dram::PORTS as u32,
                            (to.offset() + face_row(d % 32, h)) as u32,
                            (slot + face_row(r, h)) as u32,
                            64,
                            0,
                            0,
                        ]);
                    }
                }
                TransferBatch { read, write }
            })
            .collect();
        jobs.push(vec![Step::Transfer {
            what: "row write list",
            depth: 2,
            batches,
        }]);
    }
    Ok(jobs)
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
    // B reads each edge tile and fills its padding; NC writes it back. Batches
    // of half the staging alternate, so NC writes one while B fills the next.
    const PER_BATCH: usize = GROUP / 2;
    let head = |op: u32, first: usize, count: usize, stage: u64| {
        [
            op,
            value.to_bits(),
            first as u32,
            count as u32,
            rows as u32,
            cols as u32,
            stage as u32,
            rt as u32,
        ]
    };
    Ok(runs(edges, units, GROUP)
        .into_iter()
        .map(|run| {
            let batches = (run.start..run.end)
                .step_by(PER_BATCH)
                .enumerate()
                .map(|(b, first)| {
                    let count = PER_BATCH.min(run.end - first);
                    let at = stage + (b % 2 * PER_BATCH) as u64 * TILE_SLOT;
                    let side =
                        |op: u32| vec![head(op, first, count, at), r.encode()[0], r.encode()[1]];
                    TransferBatch {
                        read: side(record::FILL_PAD),
                        write: side(record::PAD_WRITE),
                    }
                })
                .collect();
            vec![Step::Transfer {
                what: "fill-pad list",
                depth: 2,
                batches,
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
    use crate::kind;
    use tt_isa::dm::TILE_DATA;
    use tt_isa::dm::{fill, op, record};
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
            let sig = elems(kind);
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

    /// `entries` with each record expanded, appended to `out`.
    fn expanded(entries: &[[u32; 8]], out: &mut Vec<Result<[u32; 8], Vec<u32>>>) {
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
    }

    /// One job as the mover runs it: every entry in order, records expanded,
    /// a `WAIT` between what were separate lists (and, in a transfer, between
    /// a batch's reads and its writes, which the credit orders), and each
    /// kernel as its programs' first words (`None` for a list entry).
    fn stream(job: &Job) -> Vec<Result<[u32; 8], Vec<u32>>> {
        let mut out = Vec::new();
        let mut after_list = false;
        for step in job {
            match step {
                Step::List { entries, .. } => {
                    if after_list {
                        out.push(Ok([op::WAIT, 0, 0, 0, 0, 0, 0, 0]));
                    }
                    expanded(entries, &mut out);
                    after_list = true;
                }
                Step::Transfer { batches, .. } => {
                    if after_list {
                        out.push(Ok([op::WAIT, 0, 0, 0, 0, 0, 0, 0]));
                    }
                    for batch in batches {
                        expanded(&batch.read, &mut out);
                        out.push(Ok([op::WAIT, 0, 0, 0, 0, 0, 0, 0]));
                        expanded(&batch.write, &mut out);
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
                    let got = super::matmul_dram(
                        &mut a1, &t1[0], ta, &t1[1], tb, route, f, units, false, false,
                    )
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
                                    fill::param(
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
                        // B reads and fills a batch's tiles; NC writes each back
                        // from its slot, after the batch's credit.
                        let e: Vec<_> = stream(job).into_iter().map(Result::unwrap).collect();
                        let mut filled = BTreeMap::new();
                        for w in e.split(|x| x[0] == op::WAIT) {
                            let reads: Vec<_> = w
                                .iter()
                                .filter(|x| x[0] == op::READ || x[0] == op::FILL)
                                .collect();
                            for pair in reads.chunks(2) {
                                let (rd, cp) = (pair[0], pair[1]);
                                assert_eq!((rd[0], cp[0]), (op::READ, op::FILL));
                                assert_eq!(cp[1], f32::NEG_INFINITY.to_bits());
                                assert_eq!(cp[3], rd[4], "in its slot");
                                assert!(got.insert((rd[1], rd[3]), cp[2]).is_none(), "twice");
                                filled.insert(rd[4] + TILE_DATA as u32, (rd[1], rd[3]));
                            }
                            for wr in w.iter().filter(|x| x[0] == op::WRITE) {
                                let (ch, off) = filled[&wr[4]];
                                assert_eq!((wr[1], wr[3]), (ch, off + TILE_DATA as u32));
                            }
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
        assert_eq!(SumRows.produces(&[&undef]), Pad::Undefined);
        assert_eq!(SumRows.produces(&[&zero]), Pad::Zero);
        assert_eq!(SumRows.requires(0), PadNeed::Any);
        assert_eq!(MatmulPadding.requires(1), PadNeed::Zero);
        assert_eq!(super::fill_pad(&whole, 0.0, 4).unwrap().len(), 0);
    }

    /// A row view's records name its first tile, so they expand as the
    /// view's own lists did.

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
                    a_at: _,
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
                        half: None,
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
}

#[cfg(test)]
mod overlap_tests {
    use super::{pipeline_share, pipelined_runs, Overlap};

    /// `len` one-tile items on `units`, in runs of at most 64 tiles.
    fn runs(len: usize, units: usize, overlap: Overlap) -> bool {
        pipelined_runs(len, 1, units, 64, overlap).is_some()
    }

    #[test]
    fn a_serialized_op_never_overlaps() {
        for units in [1, 8, 32] {
            assert!(!runs(1 << 20, units, Overlap::Off), "{units} units");
        }
    }

    /// Where card 0 measured fresh ops: on one tile overlap gained from 49
    /// tiles; on eight it lost to 1152 tiles a unit and gained from 1568
    /// (`PIPELINE_SHARE_PER_UNIT`); on 32 it lost at every size measured.
    #[test]
    fn a_fresh_op_asks_more_of_each_unit_the_more_units_there_are() {
        assert_eq!(pipeline_share(Overlap::Fresh, 1), 48);
        assert!(runs(48, 1, Overlap::Fresh) && !runs(47, 1, Overlap::Fresh));
        let eight = pipeline_share(Overlap::Fresh, 8);
        assert!((1152..=1568).contains(&eight), "{eight}");
        // 2048 x 2048 is 4096 tiles: 512 a unit, which lost.
        assert!(!runs(4096, 8, Overlap::Fresh));
        assert!(runs(8 * eight, 8, Overlap::Fresh) && !runs(8 * eight - 1, 8, Overlap::Fresh));
        // 4096 x 4096 is 16384 tiles: 2048 a unit, which gained.
        assert!(runs(16384, 8, Overlap::Fresh));
        // 6144 x 6144 is 36864 tiles: 1152 a unit on 32, which lost 1.38.
        assert!(!runs(36864, 32, Overlap::Fresh));
    }

    /// A replay pays no host time for its runs: a capture overlaps from 32
    /// tiles a unit, whatever the units (8 tiles a unit lost 1.6% on 32).
    #[test]
    fn a_captured_op_asks_the_same_of_every_unit() {
        for units in [1, 2, 8, 32] {
            let share = pipeline_share(Overlap::Captured, units);
            assert_eq!(share, 32);
            assert!(runs(share * units, units, Overlap::Captured), "{units}");
            assert!(
                !runs(share * units - 1, units, Overlap::Captured),
                "{units}"
            );
        }
        // 512 x 512 on 8 units, 32 on 32 units.
        assert!(runs(256, 8, Overlap::Captured) && !runs(256, 32, Overlap::Captured));
        assert!(runs(1024, 32, Overlap::Captured));
        // A capture never asks less than a fresh op's one-unit share.
        assert!(pipeline_share(Overlap::Fresh, 1) > pipeline_share(Overlap::Captured, 1));
    }
}
