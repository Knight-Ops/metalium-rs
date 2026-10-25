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
    /// No channel has room for the tensor's share of slots: the channel that refused it had
    /// `free` bytes, the largest run of them `largest` (less than the share's, or it would fit).
    OutOfMemory {
        slots: u64,
        free: u64,
        largest: u64,
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
            TensorError::OutOfMemory {
                slots,
                free,
                largest,
            } => write!(
                f,
                "no room in GDDR for {slots} tile slots per channel: the channel that refused \
                 them has {free} bytes free, the largest run {largest}"
            ),
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
    /// When its slots were allocated, in the allocator's order (a view's are its parent's):
    /// what a [`FreeSnapshot`] tells apart from its own moment.
    serial: u64,
}

impl Placement {
    /// The bytes it takes on each of its channels.
    pub(crate) fn channel_bytes(&self) -> u64 {
        self.slots * self.slot_bytes
    }

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

    /// Each channel's share of the slots, in the order tiles are dealt to them: `(the chip's
    /// channel index, offset, bytes)`. A view's are its allocation's.
    pub(crate) fn extents(&self) -> impl Iterator<Item = (u8, u64, u64)> + '_ {
        let bytes = self.slots * self.slot_bytes;
        self.channels
            .iter()
            .zip(&self.base)
            .map(move |(c, &b)| (c.index(), b, bytes))
    }

    /// Bytes a tile's slot takes.
    pub(crate) fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// Do `self` and `other` name slots of one allocation (a view and what it borrows, or the
    /// same tensor twice)?
    pub(crate) fn same_allocation(&self, other: &Placement) -> bool {
        self.channels == other.channels && self.base == other.base
    }

    /// Are `self` and `other` one allocation by when it was made, not where: where a capture
    /// reuses its own tensors' GDDR (`DramAlloc::start_pool`), a later tensor can take an earlier
    /// one's slots, base for base.
    pub(crate) fn same_serial(&self, other: &Placement) -> bool {
        self.serial == other.serial
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
    /// The serial the allocator's next placement takes.
    serial: u64,
}

impl FreeSnapshot {
    /// Were all of `p`'s slots free then -- so allocated since, and nothing
    /// captured then can name them? A placement allocated after the snapshot was: what then
    /// occupied its slots was freed since, and a trace that named that had its free deferred
    /// until the trace's release, so it was not, or its slots would not have come back. Without
    /// that, a slot in use at a trace's end and freed and reused since would be the trace's for
    /// good, and every eager op beside live traces would leak its buffers into them
    /// (`docs/plans/long-context.md`: an eager 32K prefill beside 13 traces filled 31 GB).
    pub(crate) fn was_free(&self, p: &Placement) -> bool {
        if p.serial >= self.serial {
            return true;
        }
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
    /// The tensors allocated since [`DramAlloc::start_log`], in order, while one is kept: a
    /// capture's own tensors (`crate::residency`).
    log: Option<Vec<Placement>>,
    /// The next placement's [`Placement::serial`].
    serial: u64,
    /// A capture's own freed slots, while it reuses them ([`DramAlloc::start_pool`]).
    pool: Option<Pool>,
}

/// The slots a capture's own tensors gave back during it (`docs/plans/long-context.md` 1.G):
/// later allocations of the capture take them first, and the trace keeps what is left of them
/// when it ends ([`DramAlloc::take_pool`]). Kept apart from the free lists, so nothing allocated
/// outside the capture (or after it, while the trace lives) lands in slots a replay writes.
#[derive(Debug)]
struct Pool {
    /// The capture's first placement's serial: only a tensor from it on is the capture's own.
    from: u64,
    /// Per channel, as `DramAlloc::free`.
    free: Vec<BTreeMap<u64, u64>>,
    /// Placements written by the host during the capture (an upload, a constant): a replay does
    /// not write them again, so their slots are never another tensor's.
    uploads: std::collections::HashSet<u64>,
    /// Set while an upload allocates: it never takes the pool's slots, since the host writes it
    /// at once, ahead of the queued ops that may still read them.
    bypass: bool,
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
        DramAlloc {
            channels,
            free,
            log: None,
            serial: 0,
            pool: None,
        }
    }

    /// Keep the slots of the capture's own tensors freed from now on for its later allocations
    /// ([`DramAlloc::give_to_pool`]), until [`DramAlloc::take_pool`].
    pub(crate) fn start_pool(&mut self) {
        self.pool = Some(Pool {
            from: self.serial,
            free: vec![BTreeMap::new(); self.channels.len()],
            uploads: Default::default(),
            bypass: false,
        });
    }

    /// `f`'s allocations are uploads, written by the host: never from the pool, and never given
    /// back to it.
    pub(crate) fn uploading<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let Some(pool) = self.pool.as_mut() else {
            return f(self);
        };
        pool.bypass = true;
        let first = self.serial;
        let out = f(self);
        let pool = self.pool.as_mut().expect("kept above");
        pool.bypass = false;
        pool.uploads.extend(first..self.serial);
        out
    }

    /// The host wrote `p` during the capture: as an upload, its slots are never given to the pool.
    pub(crate) fn note_host_written(&mut self, p: &Placement) {
        if let Some(pool) = self.pool.as_mut() {
            pool.uploads.insert(p.serial);
        }
    }

    /// Give `p`'s slots to the capture's pool if they are its own to reuse: allocated during the
    /// capture, not by an upload. `false` (nothing done) otherwise, or with no pool.
    pub(crate) fn give_to_pool(&mut self, p: &Placement) -> bool {
        let Some(pool) = self.pool.as_mut() else {
            return false;
        };
        if !p.owned || p.serial < pool.from || pool.uploads.contains(&p.serial) {
            return false;
        }
        for (c, &b) in p.channels.iter().zip(&p.base) {
            let i = self
                .channels
                .iter()
                .position(|k| k == c)
                .expect("a placement's channel is the allocator's");
            release(&mut pool.free[i], b, p.slots * p.slot_bytes);
        }
        true
    }

    /// Stop pooling: the slots the pool still holds, `(channel index, offset, bytes)`, which the
    /// trace keeps until it is released ([`DramAlloc::free_ranges`]).
    pub(crate) fn take_pool(&mut self) -> Vec<(usize, u64, u64)> {
        let Some(pool) = self.pool.take() else {
            return Vec::new();
        };
        pool.free
            .into_iter()
            .enumerate()
            .flat_map(|(i, f)| f.into_iter().map(move |(at, len)| (i, at, len)))
            .collect()
    }

    /// Give back ranges [`DramAlloc::take_pool`] returned.
    pub(crate) fn free_ranges(&mut self, ranges: &[(usize, u64, u64)]) {
        for &(i, at, len) in ranges {
            release(&mut self.free[i], at, len);
        }
    }

    /// Keep every tensor allocated from now on ([`DramAlloc::alloc`]) until
    /// [`DramAlloc::take_log`].
    pub(crate) fn start_log(&mut self) {
        self.log = Some(Vec::new());
    }

    /// What was allocated since [`DramAlloc::start_log`], so far.
    pub(crate) fn logged(&self) -> &[Placement] {
        self.log.as_deref().unwrap_or(&[])
    }

    /// What was allocated since [`DramAlloc::start_log`], and stop keeping it.
    pub(crate) fn take_log(&mut self) -> Vec<Placement> {
        self.log.take().unwrap_or_default()
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
        self.serial += 1;
        Ok(Placement {
            channels: vec![self.channels[channel]],
            base: vec![at],
            slots,
            slot_bytes: TILE_SLOT,
            tiles: 1,
            first: 0,
            owned: true,
            serial: self.serial - 1,
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
        // A capture's freed slots first, where every channel has the room (no placement takes
        // some channels from the pool and some from the free lists).
        if let Some(pool) = self.pool.as_mut().filter(|p| !p.bypass) {
            let found: Vec<Option<u64>> = pool
                .free
                .iter()
                .map(|f| f.iter().find(|(_, &len)| len >= bytes).map(|(&at, _)| at))
                .collect();
            if found.iter().all(Option::is_some) {
                let base: Vec<u64> = found.into_iter().flatten().collect();
                for (f, &at) in pool.free.iter_mut().zip(&base) {
                    let len = f.remove(&at).expect("found above");
                    if len > bytes {
                        f.insert(at + bytes, len - bytes);
                    }
                }
                return Ok(self.placed(base, slots, slot_bytes, tiles));
            }
        }
        let mut base = Vec::with_capacity(n);
        for i in 0..n {
            let Some((&at, &len)) = self.free[i].iter().find(|(_, &len)| len >= bytes) else {
                let free = self.free[i].values().sum();
                let largest = self.free[i].values().copied().max().unwrap_or(0);
                // Give back what this call already took.
                for (j, &b) in base.iter().enumerate() {
                    release(&mut self.free[j], b, bytes);
                }
                return Err(TensorError::OutOfMemory {
                    slots,
                    free,
                    largest,
                });
            };
            self.free[i].remove(&at);
            if len > bytes {
                self.free[i].insert(at + bytes, len - bytes);
            }
            base.push(at);
        }
        Ok(self.placed(base, slots, slot_bytes, tiles))
    }

    /// The placement [`DramAlloc::alloc_slots`] made at `base`, given its serial and logged.
    fn placed(&mut self, base: Vec<u64>, slots: u64, slot_bytes: u64, tiles: usize) -> Placement {
        self.serial += 1;
        let p = Placement {
            channels: self.channels.clone(),
            base,
            slots,
            slot_bytes,
            tiles,
            first: 0,
            owned: true,
            serial: self.serial - 1,
        };
        if let Some(log) = &mut self.log {
            log.push(p.clone());
        }
        p
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
            serial: self.serial,
        }
    }

    /// The chip's usable channels, in the allocator's order.
    pub(crate) fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Free bytes on every channel together.
    pub fn free_bytes_total(&self) -> u64 {
        self.free.iter().flat_map(|f| f.values()).sum()
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
/// (`new-op-definition-of-done.md`, item 7).
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
    // What each staging's plan needs of the arena: noted below only where both answers decide.
    let floors = std::cell::Cell::new(0);
    // Tiles gathered over the whole op, and blocks, for a staging.
    let gathered = |st: Staging| -> Option<(usize, usize)> {
        let (s, floor) = matmul::plan_floor([m, k, n], route, fidelity, st);
        let s = s?;
        if s.tiles[1] < kt {
            return None;
        }
        floors.set(floors.get().max(floor));
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
        // A staging that cannot hold `K` whole cannot in a smaller arena either: no smaller arena
        // changes the answer.
        return false;
    };
    // Both plans decide; a smaller arena that narrowed either could change the answer
    // (`docs/plans/planned-l1-residency.md` 2h).
    crate::l1::note_planned(crate::l1::Planned {
        top: 0,
        need: floors.get(),
    });
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
        Reads {
            thin: true,
            resident_a: false,
            unpack_b: false,
            packed_a: None,
            stream_k: false,
            stream_prefetch: false,
            prefetch_units: [usize::MAX; 2],
            handoff: None,
            second_job: false,
            balanced_deal: false,
            block_stream: BlockStream::Off,
        },
    )
}

/// A development probe (`docs/plans/decode-to-bandwidth.md` Phase 4, the bound on reading `A` once a
/// product): every GDDR matmul block is gathered [`record::GATHER_A_RESIDENT`], reading only `B`.
/// **The products are wrong** (`A`'s slots hold whatever they held); only their time means anything.
static PROBE_SKIP_A: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turn [`PROBE_SKIP_A`] on or off, process-wide. Dev-facing: a timing bound, never a result.
#[doc(hidden)]
pub fn probe_skip_a_reads(on: bool) {
    PROBE_SKIP_A.store(on, std::sync::atomic::Ordering::SeqCst);
}

fn probing_skip_a() -> bool {
    PROBE_SKIP_A.load(std::sync::atomic::Ordering::Relaxed)
}

/// A development probe (`docs/plans/decode-to-bandwidth.md` Phase 8, the stream fully ahead of the
/// products): a product with a packed `A` gathers nothing, so it reads no `B` (its `GATHER` reads
/// only `B`, [`record::GATHER_A_RESIDENT`]), and the session prefetches nothing. What is left of a
/// step is its products' compute, control and barriers and the small ops: the step with its weight
/// stream made free. **The products are wrong**; only their time means anything.
static PROBE_SKIP_WEIGHTS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Turn [`PROBE_SKIP_WEIGHTS`] on or off, process-wide. Dev-facing: a timing bound, never a result.
#[doc(hidden)]
pub fn probe_skip_weight_reads(on: bool) {
    PROBE_SKIP_WEIGHTS.store(on, std::sync::atomic::Ordering::SeqCst);
}

pub(crate) fn probing_skip_weights() -> bool {
    PROBE_SKIP_WEIGHTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// A gather list with its `B`-only `GATHER` left out under [`PROBE_SKIP_WEIGHTS`] (only where `A`
/// is packed, so the record reads nothing else); a `WAIT` stands in where nothing is left.
fn probe_gather(mut gather: Vec<[u32; 8]>, record: [[u32; 8]; 5], packed_a: bool) -> Vec<[u32; 8]> {
    if packed_a && probing_skip_weights() {
        if gather.is_empty() {
            gather.push([tt_isa::dm::op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
        }
    } else {
        gather.extend(record);
    }
    gather
}

/// How a GDDR matmul reads `A` ([`matmul_dram_limited`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reads {
    /// `A`'s last tile row is read as only its [`thin_rows`] (off, whole tiles: the output's
    /// padding rows are then what `A`'s held).
    pub thin: bool,
    /// A unit's later blocks of the same `A` rows do not read it again
    /// ([`record::GATHER_A_RESIDENT`]): with `K` whole under [`Staging::Slots`], `A` is at the
    /// same slots for every block.
    pub resident_a: bool,
    /// A transposed `B` is transposed by the kernel's unpacker (`BlockPlan::unpack_b`).
    pub unpack_b: bool,
    /// `A` (one row) as [`pack_row`] laid it out, `[K / 32, 32]`: each block reads its
    /// `K / 1024` tiles instead of a thin read a `K` tile ([`BlockPlan::packed_a`]).
    pub packed_a: Option<TensorRef>,
    /// With `packed_a`, each output column's `K` in chunks the mover reads while the kernel
    /// computes the chunk before ([`BlockPlan::push_stream_jobs`]).
    pub stream_k: bool,
    /// A streamed product's columns may be prefetched too ([`matmul_dram_prefetching`]).
    pub stream_prefetch: bool,
    /// The most prefetching units a product takes, its `K` whole and streamed: a streamed
    /// column reads only its first chunks ahead, so the two kinds want different counts.
    pub prefetch_units: [usize; 2],
    /// With a streamed product prefetched, each column its own units would read whole is handed
    /// between two units at this chunk (`Some(0)`: half the chunks; [`Handoff`]).
    pub handoff: Option<usize>,
    /// A `K`-whole product's prefetching units each take a second job of one column, its `B` in a
    /// ring below the first's and read ahead with it ([`BlockPlan::push_prefetched_jobs`]).
    pub second_job: bool,
    /// A `K`-whole product whose own units would run more than a round deals the columns its span
    /// reads over its prefetching units too, not its own units alone
    /// ([`BlockPlan::push_prefetched_jobs`], [`balanced_shares`]).
    pub balanced_deal: bool,
    /// Whether a product of several tile rows runs as 2-D blocks with `K` streamed
    /// ([`BlockStream`]).
    pub block_stream: BlockStream,
}

/// Whether a GDDR product of at least two tile rows runs as `[mb, nb]` blocks of output tiles,
/// each job streaming `K` in chunks of at most `kq` tiles and reading its `A` and `B` strips once
/// (`BlockPlan::push_block_stream_jobs`, `docs/plans/prefill-operand-reuse.md`). Bit for bit the
/// other paths: each output tile's `K` chain is the same, in the same order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlockStream {
    /// Never.
    Off,
    /// Where [`block_stream_plan`] finds a shape whose modelled time beats the other paths'.
    #[default]
    Auto,
    /// Always where the product allows one, with this `[mb, kq, nb]` (each clamped to the
    /// product): a gate's or a sweep's.
    Fixed([usize; 3]),
}

impl BlockStream {
    /// `TT_MATMUL_BLOCK`: `off`, `auto` or `mb,kq,nb`; [`BlockStream::Auto`] when unset.
    pub fn from_env() -> Self {
        match std::env::var("TT_MATMUL_BLOCK") {
            Err(_) => BlockStream::Auto,
            Ok(v) => v
                .parse()
                .unwrap_or_else(|e| panic!("TT_MATMUL_BLOCK={v}: {e}")),
        }
    }
}

impl std::str::FromStr for BlockStream {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s {
            "off" | "0" => Ok(BlockStream::Off),
            "auto" | "1" => Ok(BlockStream::Auto),
            _ => {
                let parts: Vec<usize> = s
                    .split(',')
                    .map(|p| p.trim().parse::<usize>())
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|e| format!("expected `off`, `auto` or `mb,kq,nb`: {e}"))?;
                match parts[..] {
                    [mb, kq, nb] if mb > 0 && kq > 0 && nb > 0 => {
                        Ok(BlockStream::Fixed([mb, kq, nb]))
                    }
                    _ => Err("expected `off`, `auto` or three positive `mb,kq,nb`".into()),
                }
            }
        }
    }
}

/// The rows of `op(A)`'s last tile row a matmul reads when it reads only what
/// it uses ([`record::gather_rows`]): `m % 32` when that is 1..=16 and `A` is
/// not transposed, otherwise 0 (whole tiles). An `m = 1` decode product reads
/// one row of each `A` tile instead of 32.
pub fn thin_rows(m: usize, a_transposed: bool) -> u32 {
    match m % 32 {
        r @ 1..=16 if !a_transposed => r as u32,
        _ => 0,
    }
}

/// The packed tiles of a [`pack_row`]'d `A` holding `K` tiles `k0..k0 + count`, read into the
/// slots from `at`: a `READ_RUN` of whole tiles down its one tile column.
fn packed_a_reads(packed: TensorRef, k0: usize, count: usize, at: u64) -> Vec<[u32; 8]> {
    let (first, last) = (k0 / 32, (k0 + count - 1) / 32);
    vec![
        [
            record::READ_RUN,
            first as u32,
            (last - first + 1) as u32,
            at as u32,
            0,
            1,
            0,
            0,
        ],
        packed.encode()[0],
        packed.encode()[1],
    ]
}

/// One row of `K` values (`t`, `[1, K]`, `K` a multiple of 32) laid out `[K / 32, 32]`: row
/// `kk` is `t`'s columns `32 kk..32 kk + 32`, the row of `t`'s tile `kk` that holds data. A
/// product's packed `A` ([`Reads::packed_a`]): it reads `K / 1024` tiles, where a thin read
/// reads `K / 32`, an entry each. Spread over `units`: each reads its tiles' rows (one entry a
/// tile) and writes each row's two face rows into place (two `WRITE`s), B reading and NC
/// writing ([`Step::Transfer`]), no kernel. Data movement only, so bit for bit; the last tile's
/// rows past `K / 32` are not written (padding, `Undefined`).
pub fn pack_row(alloc: &mut DramAlloc, t: &DramTensor, units: usize) -> Result<Work> {
    use tt_isa::dm::op;
    t.expect("a packed row", Elem::F32)?;
    if t.rows != 1 || t.cols % 32 != 0 {
        return Err(TensorError::Shape(format!(
            "a packed row is [1, K] with K whole tiles, not [{}, {}]",
            t.rows, t.cols
        )));
    }
    // Two writes a tile: a batch's side stays within `TRANSFER_SIDE_MAX`.
    const GROUP: usize = 64;
    const FACE: u64 = 1024;
    let kt = t.cols / 32;
    let stage = staging("packed row slots", GROUP.min(kt))?;
    let out = DramTensor::alloc(alloc, kt, 32)?;
    let jobs = runs(kt, units, GROUP)
        .into_iter()
        .map(|run| {
            let (mut read, mut write) = (Vec::new(), Vec::new());
            for (q, kk) in run.enumerate() {
                let (src, slot) = (t.tile(0, kk), stage + q as u64 * TILE_SLOT);
                // Row 0 of faces 0 and 1: face 0 whole, then face 1's first row.
                read.push([
                    op::READ,
                    src.channel().index() as u32,
                    (kk % tt_isa::dram::PORTS as usize) as u32,
                    (src.offset() + TILE_DATA) as u32,
                    (slot + TILE_DATA) as u32,
                    (FACE + 64) as u32,
                    0,
                    0,
                ]);
                let (dst, r) = (out.tile(kk / 32, 0), (kk % 32) as u64);
                for half in 0..2u64 {
                    let face = 2 * (r / 16) + half;
                    write.push([
                        op::WRITE,
                        dst.channel().index() as u32,
                        0,
                        (dst.offset() + TILE_DATA + face * FACE + (r % 16) * 64) as u32,
                        (slot + TILE_DATA + half * FACE) as u32,
                        64,
                        0,
                        0,
                    ]);
                }
            }
            vec![Step::Transfer {
                what: "packed row",
                depth: 1,
                batches: vec![TransferBatch { read, write }],
            }]
        })
        .collect();
    out.set_pad(Pad::Undefined);
    Ok(Work { out, jobs })
}

/// [`matmul_dram`] with `K` blocked to at most `max_k_tiles`, reading `A` as `reads` says.
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
    reads: Reads,
) -> Result<Work> {
    matmul_dram_prefetching(
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
        max_k_tiles,
        reads,
        &[],
        units,
        &[],
    )
    .map(|(work, ..)| work)
}

/// What a list that reads a product's `B` ahead of it is called: the session puts it in a unit's
/// trace stream ahead of the wait that orders the unit after the ops before the product
/// ([`BlockPlan::push_prefetched_jobs`]).
pub const PREFETCH: &str = "weight prefetch";

/// What a list that reads a product's `B` ahead on a unit that runs the ops before the product
/// is called (`docs/plans/decode-to-bandwidth.md` Phase 8): its `GATHER` carries
/// [`record::GATHER_PREFETCH`], so the ops' own drains leave its reads in flight, and the job's
/// gather list ends with the `WAIT` that drains them. The capture moves it back in the unit's
/// stream to just after the last weight-streaming product, or towards the last one the unit took
/// part in (`Session::set_prefetch_anchor`); the capture's own run, and any run that is not
/// captured, read it where the job is.
pub const BUSY_PREFETCH: &str = "busy weight prefetch";

/// What the first op's list on a unit that takes the rest of a handed column is called
/// ([`Handoff`]): the packed `A`, and its first two chunks' `B` as a prefetch
/// ([`record::GATHER_PREFETCH`]), which the op's barrier leaves in flight.
pub const HANDED_AHEAD: &str = "handed column read ahead";

/// The second op of a streamed product whose columns are handed between two units
/// (`docs/plans/decode-to-bandwidth.md` Phase 8, "The stream's rate"): in the first op, a unit
/// that would read a whole column computes its first `h` chunks and writes the FP32 partial to
/// `partials`, while a unit with no job in the product reads the rest of its `B` ahead
/// ([`HANDED_AHEAD`]); after the first op's barrier, that unit reloads the partial into the
/// slot chunk `h` reloads from and computes the rest, writing the column. Each output tile's
/// `K` chain is the same in the same order, through the same FP32 partial, so bit for bit; no
/// unit reads a whole column inside the product.
pub(crate) struct Handoff {
    /// The second op's jobs, laid out as the first op's: a unit's at its own index.
    pub jobs: Vec<Job>,
    /// The handed columns' partials, `C`'s shape: freed once the second op is queued.
    pub partials: DramTensor,
    /// The L1 the jobs plan (the first op's stream layout), noted again for the second op.
    pub planned: crate::l1::Planned,
}

/// [`matmul_dram_limited`], with `prefetch` (units at or past `units`, below `spread`) each
/// taking one job of the product whose `B` its own list reads ahead ([`PREFETCH`],
/// [`BlockPlan::push_prefetched_jobs`]) -- where the product is one row with a packed `A` and
/// its `K` whole or streamed (a prefetching unit's one job is in the first half when pipelined). Returns the work, `Some(spread)` when its jobs are dealt over
/// `spread` units (job `j` to unit `j % spread`), else `None` (over `units`, `prefetch` unused),
/// and, where `reads.handoff` hands columns, the second op (dealt as the first).
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_dram_prefetching(
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
    reads: Reads,
    prefetch: &[usize],
    spread: usize,
    busy: &[Option<u64>],
) -> Result<(Work, Option<usize>, Option<Handoff>)> {
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
    let (shape, floor) = matmul::plan_floor([m, k, n], route, fidelity, staging);
    let shape = shape
        .ok_or_else(|| TensorError::Shape(format!("[{m}, {k}] @ [{k}, {n}] fits no chunk")))?;
    let [mc, kc, nc] = shape.tiles;
    let kc = max_k_tiles.map_or(kc, |limit| kc.min(limit.get()));
    let kc = if kc < kt {
        k_block_size(kc, route, fidelity)?
    } else {
        kc
    };
    // A packed `A` takes a few slots, not `K`'s: its K blocks are as few as fit.
    let kc = match reads.packed_a {
        Some(_) if kc < kt => packed_k_block(kt, max_k_tiles, route, fidelity)?,
        _ => kc,
    };
    // Several tile rows: 2-D blocks with `K` streamed, where the mode asks for them and the
    // modelled time beats this plan's.
    if reads.packed_a.is_none() && max_k_tiles.is_none() {
        let [bmc, bnc] = blocks([mt, nt], [mc, nc], units);
        let other = blocked_paths_model([mt, kt, nt], [bmc, kc, bnc], units);
        if let Some((shape, layout)) = block_stream_plan(
            reads.block_stream,
            [mt, kt, nt],
            units,
            other,
            route,
            fidelity,
        ) {
            let c = DramTensor::alloc(alloc, m, n)?;
            let mut jobs = Vec::new();
            push_block_stream_jobs(
                &mut jobs,
                &shape,
                &layout,
                [mt, kt, nt],
                m,
                [a.tensor_ref(), b.tensor_ref(), c.tensor_ref()],
                [a_transposed, b_transposed],
                if reads.thin {
                    thin_rows(m, a_transposed)
                } else {
                    0
                },
                reads.unpack_b && b_transposed && route == SrcRoute::Tf32FromFp32,
                route,
                fidelity,
            );
            return Ok((Work { out: c, jobs }, None, None));
        }
    }
    // The plan's block is what runs, so the arena must keep it (`matmul::plan_floor`) -- except
    // where a packed `A`'s `K` is blocked: its blocks are `packed_k_block`'s, which notes its own
    // layout, and its jobs a column each, whatever the plan's width; and a smaller arena keeps `K`
    // blocked (`docs/plans/planned-l1-residency.md` 2h-i: `down` at 8192, 8960 and 14,336 laid
    // out and timed alike past the plan's floor).
    if !(reads.packed_a.is_some() && kc < kt) {
        crate::l1::note_planned(crate::l1::Planned {
            top: 0,
            need: floor,
        });
    }
    let col_runs = thin_runs([mt, kt, nt], [kc, nc], staging, units);
    let [mc, nc] = blocks([mt, nt], [mc, nc], units);
    let (ra, rb) = (a.tensor_ref(), b.tensor_ref());
    let c = DramTensor::alloc(alloc, m, n)?;
    let rc = c.tensor_ref();
    let mut jobs = Vec::new();
    if reads.packed_a.is_some() && (m != 1 || a_transposed) {
        return Err(TensorError::Shape(format!(
            "a packed `A` is one row read as it is, not [{m}, {k}] transposed {a_transposed}"
        )));
    }
    let plan = BlockPlan {
        tiles: [mt, kt, nt],
        m,
        block: [mc, nc],
        k_block: kc,
        staging,
        units,
        route,
        fidelity,
        allow_mop,
        thin: if reads.thin {
            thin_rows(m, a_transposed)
        } else {
            0
        },
        resident_a: reads.resident_a && staging == Staging::Slots,
        col_runs,
        unpack_b: reads.unpack_b && route == SrcRoute::Tf32FromFp32,
        packed_a: reads.packed_a,
        // Only where `K` would be blocked: a whole `K` is one batch a column, and each chunk
        // batch's own reload, pack and retire (~3 us a chunk, profile `1792358552`) cost a unit
        // of several columns more than the compute they hide.
        stream_k: reads.stream_k
            && reads.packed_a.is_some()
            && mt == 1
            && staging == Staging::Slots
            && kc < kt,
        busy: busy.to_vec(),
        second_job: reads.second_job,
        balanced_deal: reads.balanced_deal,
    };
    let prefetch = &prefetch[..prefetch
        .len()
        .min(reads.prefetch_units[usize::from(plan.stream_k)])];
    let prefetching = !prefetch.is_empty()
        && plan.packed_a.is_some()
        && mt == 1
        && if plan.stream_k {
            reads.stream_prefetch
        } else {
            kc >= kt
        }
        && prefetch.iter().all(|&u| (units..spread).contains(&u));
    // The handed columns' partials, if any column may be handed.
    let partials = match reads.handoff {
        Some(_) if prefetching && plan.stream_k => match DramTensor::alloc(alloc, m, n) {
            Ok(p) => Some(p),
            Err(e) => {
                alloc.free(&c.placement);
                return Err(e);
            }
        },
        _ => None,
    };
    let handoff = reads.handoff.zip(partials.as_ref().map(|p| p.tensor_ref()));
    let pushed = if prefetching {
        plan.push_prefetched_jobs(
            &mut jobs,
            [ra, rb, rc],
            [a_transposed, b_transposed],
            prefetch,
            spread,
            handoff,
        )
    } else {
        plan.push_jobs(&mut jobs, [ra, rb, rc], [a_transposed, b_transposed])
            .map(|()| Vec::new())
    };
    let handed = match pushed {
        Ok(handed) => handed,
        Err(e) => {
            alloc.free(&c.placement);
            if let Some(p) = &partials {
                alloc.free(&p.placement);
            }
            return Err(e);
        }
    };
    let handoff = match partials {
        Some(partials) if !handed.is_empty() => {
            // What the jobs planned, left noted for the first op.
            let planned = crate::l1::take_planned();
            crate::l1::note_planned(planned);
            Some(Handoff {
                jobs: handed,
                partials,
                planned,
            })
        }
        Some(partials) => {
            alloc.free(&partials.placement);
            None
        }
        None => None,
    };
    Ok((
        Work { out: c, jobs },
        prefetching.then_some(spread),
        handoff,
    ))
}

/// How one `[m, k, n]` product is cut into jobs: what [`matmul_dram`] and
/// [`matmul_dram_batched`] share.
#[derive(Clone)]
struct BlockPlan {
    tiles: [usize; 3],
    /// The product's rows: a job of its last tile row alone computes only those it holds
    /// ([`matmul::LiveRows`]).
    m: usize,
    block: [usize; 2],
    k_block: usize,
    staging: Staging,
    units: usize,
    route: SrcRoute,
    fidelity: Fidelity,
    allow_mop: bool,
    /// [`thin_rows`] of `op(A)`, set on the gathers of its last tile row.
    thin: u32,
    /// Mark a block [`record::GATHER_A_RESIDENT`] when its unit's previous block of the same
    /// `push_jobs` read the same `A` rows to the same slots. Only under [`Staging::Slots`]:
    /// [`Staging::SlotsHalf`] moves `A` between halves.
    resident_a: bool,
    /// The output's columns, one run a job in dealing order, when [`thin_runs`]
    /// deals them (instead of `block`'s width).
    col_runs: Option<Vec<std::ops::Range<usize>>>,
    /// A transposed `B`'s tiles are read as they lie and transposed by the kernel's unpacker
    /// ([`record::GATHER_B_UNPACK_TRANSPOSE`]), not one datum at a time by the mover
    /// (`READ_TRANSPOSED`, ~10 us a tile). Only on the TF32 route, where `B`'s unpack to `SrcA`
    /// converts it anyway.
    unpack_b: bool,
    /// A one-row `A` packed `[K / 32, 32]` ([`pack_row`]): a block reads the packed tiles
    /// holding its `K` range ([`record::READ_RUN`]) and its gather reads only `B`
    /// ([`record::GATHER_A_RESIDENT`]). The kernel unpacks `K` tile `kk`'s faces `(0, k)` from
    /// row `kk % 16` of the packed faces `((kk % 32) / 16, k)` ([`matmul::packed_a_pairs`]):
    /// row 0 of `SrcB` holds the same values, and the other rows only reach output rows past
    /// the product's one.
    packed_a: Option<TensorRef>,
    /// One job a column, its `K` in chunks ([`BlockPlan::push_stream_jobs`]): with `packed_a`,
    /// one tile row, [`Staging::Slots`], and a `K` too long for one block (its chunks no longer
    /// than `k_block`) only.
    stream_k: bool,
    /// Per unit of a capture, the highest `DATA` address the ops it ran since the last
    /// weight-streaming product planned, where all are known: the unit's first job reads its `B`
    /// ahead ([`BUSY_PREFETCH`]) if `B`'s ring lies above it. Empty: none does.
    busy: Vec<Option<u64>>,
    /// [`Reads::second_job`].
    second_job: bool,
    /// [`Reads::balanced_deal`].
    balanced_deal: bool,
}

/// A one-tile-row product's output columns, one run a job, in the order a
/// session deals them (job `j` to unit `j % units`): whole rounds of the plan's
/// width `nc`, then what is left as even runs over as many units as it covers.
/// [`blocks`] cuts every job `nc` wide, so the last round's few jobs leave most
/// units idle while some do a whole block more (`[1, 2048] @ [2048, 8192]`, `nc`
/// 2 on 120 units: 8 units run two jobs of 2, 4 columns; spread, 16 run 2 + 1,
/// and no unit more than the 3 that 256 columns over 120 units need). For
/// `K` whole and unpipelined only (`None` otherwise, and with no columns past
/// one per unit to deal). Each output tile's accumulation is unchanged: only
/// which job holds it moves.
pub fn thin_runs(
    [mt, kt, nt]: [usize; 3],
    [kc, nc]: [usize; 2],
    staging: Staging,
    units: usize,
) -> Option<Vec<std::ops::Range<usize>>> {
    if mt != 1 || kc < kt || staging != Staging::Slots || nt <= units || nc == 0 {
        return None;
    }
    let full = nt / (units * nc) * units * nc;
    let mut out: Vec<_> = (0..full).step_by(nc).map(|j| j..j + nc).collect();
    out.extend(
        runs(nt - full, units, nc)
            .into_iter()
            .map(|r| full + r.start..full + r.end),
    );
    Some(out)
}

/// How many of `left` columns each of `deal` units takes ([`Reads::balanced_deal`]): as even as
/// integer division allows, the remainder to the last units of the deal (a product's prefetching
/// units, which sit after its own and are served fastest; profile `1792629809`).
pub fn balanced_shares(left: usize, deal: usize) -> Vec<usize> {
    let Some(each) = left.checked_div(deal) else {
        return Vec::new();
    };
    let more = deal - left % deal;
    (0..deal).map(|d| each + usize::from(d >= more)).collect()
}

impl BlockPlan {
    /// The rows of each output tile a job of `rows` tile rows from `i0` computes.
    fn live_rows(&self, i0: usize, rows: usize) -> matmul::LiveRows {
        matmul::LiveRows::for_job(self.m, i0, rows)
    }

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
        if self.stream_k {
            return self.push_stream_jobs(jobs, [ra, rb, rc], [a_transposed, b_transposed]);
        }
        if self.k_block < kt {
            return self.push_k_jobs(jobs, [ra, rb, rc], [a_transposed, b_transposed]);
        }
        let unpack_b = self.unpack_b && b_transposed;
        let [mc, nc] = self.block;
        let (units, staging) = (self.units, self.staging);
        let (in_fmt, out_fmt) = self.route.formats();
        // Per unit, the `A` rows and slots its last block here read: a session deals job `j` to
        // unit `j % units`, and a unit runs its jobs in order with nothing between them.
        let mut last_a: Vec<Option<(usize, usize, u64)>> = vec![None; units.max(1)];
        let col_blocks: Vec<(usize, usize)> = match &self.col_runs {
            Some(runs) => runs.iter().map(|r| (r.start, r.len())).collect(),
            None => (0..nt)
                .step_by(nc)
                .map(|j0| (j0, nc.min(nt - j0)))
                .collect(),
        };
        for i0 in (0..mt).step_by(mc) {
            let rows = mc.min(mt - i0);
            for &(j0, cols) in &col_blocks {
                let tiles = [rows, kt, cols];
                // A session deals job `j` to unit `j % units`, so a unit's blocks
                // are every `units`-th: they alternate halves by `j / units`.
                let half = (staging == Staging::SlotsHalf)
                    .then(|| ((jobs.len() / units.max(1)) % 2) as u8);
                // The second half's layout is the first's moved up, and noted there.
                let up = if half == Some(1) { matmul::half() } else { 0 };
                let planned = crate::l1::planned_shifted(up, || match self.packed_a {
                    Some(_) => {
                        matmul::plan_layout_packed_a(tiles, in_fmt, staging, packed_a_slots(kt))
                    }
                    None => matmul::plan_layout_in(tiles, in_fmt, staging),
                });
                let layout = match planned {
                    Ok(l) if half == Some(1) => l.shifted(matmul::half()),
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
                let thin = if i0 + rows == mt && self.packed_a.is_none() {
                    self.thin
                } else {
                    0
                };
                let unit = jobs.len() % units.max(1);
                let resident = probing_skip_a()
                    || (self.resident_a
                        && half.is_none()
                        && last_a[unit] == Some((i0, rows, a_at)));
                last_a[unit] = Some((i0, rows, a_at));
                let flags = u32::from(a_transposed)
                    | u32::from(b_transposed) << 1
                    | record::gather_rows(thin)
                    | (kt as u32) << 8
                    | if resident || self.packed_a.is_some() {
                        record::GATHER_A_RESIDENT
                    } else {
                        0
                    }
                    | if self.packed_a.is_some() {
                        record::GATHER_STREAM
                    } else {
                        0
                    }
                    | if unpack_b {
                        record::GATHER_B_UNPACK_TRANSPOSE
                    } else {
                        0
                    };
                let mut gather = Vec::new();
                if let (Some(pa), false) = (self.packed_a, resident) {
                    gather.extend(packed_a_reads(pa, 0, kt, a_at));
                }
                let gather = probe_gather(
                    gather,
                    [
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
                    ],
                    self.packed_a.is_some(),
                );
                let outputs = match self.packed_a {
                    Some(_) => matmul::packed_a_pairs(&outputs, a_at, 0),
                    None => outputs,
                };
                // The cache tells the stagings and halves apart: their programs
                // name different addresses.
                let variant = half.map_or(0, |h| 1 + u32::from(h))
                    | if self.packed_a.is_some() { 0x40 } else { 0 };
                let (fidelity, allow_mop) = (self.fidelity, self.allow_mop);
                let live = self.live_rows(i0, rows);
                let (roles, mop) = matmul::kernel_programs(
                    tiles,
                    variant,
                    self.route,
                    fidelity,
                    sems,
                    allow_mop,
                    unpack_b,
                    live,
                    || {
                        matmul::matmul_kernel(
                            &outputs, sems, in_fmt, out_fmt, fidelity, allow_mop, unpack_b, live,
                        )
                    },
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
                // A unit's first job reads its `B` ahead if the ops it ran since the last
                // weight-streaming product all planned below `B`'s ring, and the record reads
                // plainly (a prefetch may not transpose).
                let ahead = self.packed_a.is_some()
                    && staging == Staging::Slots
                    && jobs.len() < units
                    && (!b_transposed || unpack_b)
                    && self
                        .busy
                        .get(unit)
                        .copied()
                        .flatten()
                        .is_some_and(|top| b_at >= top);
                let (mut gather, mut job) = (gather, Vec::new());
                if ahead && gather.len() >= 5 && gather[gather.len() - 5][0] == record::GATHER {
                    let mut early = gather.split_off(gather.len() - 5);
                    early[0][1] = early[0][1] & !record::GATHER_STREAM | record::GATHER_PREFETCH;
                    gather.push([
                        tt_isa::dm::op::WAIT,
                        tt_isa::dm::WAIT_PREFETCH,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]);
                    job.push(Step::List {
                        what: BUSY_PREFETCH,
                        entries: early,
                    });
                }
                job.extend([
                    Step::List {
                        what: "matmul gather",
                        entries: gather,
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
                jobs.push(job);
            }
        }
        Ok(())
    }
}

/// `A` slots a K block of `count` tiles takes with `A` packed: the packed tiles its `K` range
/// may span.
fn packed_a_slots(count: usize) -> usize {
    count.div_ceil(32) + 1
}

/// The K block of a product with a packed `A` ([`Reads::packed_a`]): `kt` cut into the fewest
/// blocks of equal size (at most `limit`) whose layout ([`matmul::k_block_layout_with`]) and
/// programs fit. `down` (`K` = 8192) takes two of 128 where a thin `A`'s 64 slots a block left
/// four of 64: half the blocks' compute between reads, and half the partials through GDDR.
fn packed_k_block(
    kt: usize,
    limit: Option<std::num::NonZeroUsize>,
    route: SrcRoute,
    fidelity: Fidelity,
) -> Result<usize> {
    let limit = limit.map_or(kt, |l| l.get());
    for blocks in 2..=kt {
        let kc = kt.div_ceil(blocks);
        if kc > limit {
            continue;
        }
        let Ok((layout, reload)) = matmul::k_block_layout_with(kc, packed_a_slots(kc)) else {
            continue;
        };
        let layout = matmul::Layout {
            outputs: matmul::packed_a_pairs(&layout.outputs, layout.a_at, 0),
            ..layout
        };
        let fits = [None, Some(reload)].into_iter().all(|r| {
            matmul::k_block_kernel(&layout, r, route, fidelity, false, matmul::LiveRows::All)
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
    Err(TensorError::Shape(
        "no K-block matmul of a packed A fits".into(),
    ))
}

impl BlockPlan {
    /// The jobs of a one-row product with a packed `A`, its `K` whole or streamed
    /// ([`BlockPlan::push_stream_jobs`]), its first columns on the `prefetch` units -- each a job
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
    fn push_prefetched_jobs(
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
            let plan = BlockPlan {
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
            let plan = BlockPlan {
                units: 1,
                col_runs: Some(std::iter::once(j0..(j0 + nc).min(firsts)).collect()),
                resident_a: false,
                busy: Vec::new(),
                ..self.clone()
            };
            let mut one = Vec::new();
            plan.push_jobs(&mut one, refs, transposed)?;
            let mut job = one.pop().expect("a job a block");
            let mut b_reads = Vec::new();
            let gathers = job.iter_mut().filter_map(|step| match step {
                Step::List { entries, .. } => entries
                    .iter()
                    .position(|e| e[0] == record::GATHER)
                    .map(|at| (entries, at)),
                _ => None,
            });
            for (entries, at) in gathers.take(ahead) {
                b_reads.extend(entries.split_off(at));
            }
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
                let plan = BlockPlan {
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
                let gathers = rest.iter_mut().filter_map(|step| match step {
                    Step::List { entries, .. } => entries
                        .iter()
                        .position(|e| e[0] == record::GATHER)
                        .map(|at| (entries, at)),
                    _ => None,
                });
                for (entries, at) in gathers.take(2) {
                    let mut gather = entries.split_off(at);
                    gather[0][1] = gather[0][1] & !record::GATHER_STREAM | record::GATHER_PREFETCH;
                    ahead.extend(gather);
                }
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
        BlockPlan {
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
                BlockPlan {
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
        let ring = (kt * self.block[1]) as u64 * TILE_SLOT;
        let top = crate::l1::ring_end();
        if top < tt_isa::l1::DATA.base + ring {
            return Ok(Vec::new());
        }
        let was = crate::l1::set_ring_end(Some(top - ring));
        let built = (first..(first + count).min(nt))
            .map(|j| {
                let plan = BlockPlan {
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
                let Some((entries, at)) = job.iter_mut().find_map(|step| match step {
                    Step::List { entries, .. } => entries
                        .iter()
                        .position(|e| e[0] == record::GATHER)
                        .map(|at| (entries, at)),
                    _ => None,
                }) else {
                    unreachable!("a block's gather record");
                };
                let b_reads = entries.split_off(at);
                Ok((b_reads, job))
            })
            .collect::<Result<Vec<_>>>();
        crate::l1::set_ring_end(was);
        // A second ring that does not plan below the first (outside the trace arena) is none.
        Ok(built.unwrap_or_default())
    }
}

/// Fewest chunks a streamed column's `K` is cut into ([`BlockPlan::push_stream_jobs`]): the
/// mover reads one while the kernel computes the one before, so the last's compute is what is
/// left after the reads.
const MIN_STREAM_CHUNKS: usize = 4;

/// The chunk size of a `kt`-tile `K` streamed with a packed `A` ([`BlockPlan::push_stream_jobs`]),
/// and its layout: an even number of chunks (a column's first in half 0, so a unit's columns
/// alternate halves throughout), at least [`MIN_STREAM_CHUNKS`], each at most `limit` tiles (the
/// K block the product would otherwise take), the fewest whose layout and programs fit.
fn stream_chunks(
    kt: usize,
    limit: usize,
    route: SrcRoute,
    fidelity: Fidelity,
) -> Result<(usize, matmul::StreamLayout)> {
    let a_slots = packed_a_slots(kt);
    for chunks in (MIN_STREAM_CHUNKS..=kt.max(MIN_STREAM_CHUNKS)).step_by(2) {
        let kq = kt.div_ceil(chunks);
        if kt.div_ceil(kq) % 2 != 0 || kq > limit {
            continue;
        }
        let Ok(layout) = matmul::k_stream_layout(kq, a_slots) else {
            continue;
        };
        let fits = (0..2).all(|c| {
            let (l, r) = layout.chunk(c, c * kq, kq);
            matmul::k_block_kernel(&l, r, route, fidelity, false, matmul::LiveRows::All)
                .iter()
                .all(|p| {
                    p.len() <= tt_isa::mailbox::PROGRAM_MAX as usize
                        && p.len() * 4 <= tt_isa::l1::PROGRAM_CACHE.len() as usize / 2
                })
        });
        if fits {
            return Ok((kq, layout));
        }
    }
    Err(TensorError::Shape(format!(
        "no K-streamed matmul of {kt} K tiles fits"
    )))
}

impl BlockPlan {
    /// One job an output column of a one-row product with a packed `A`, its `K` in chunks
    /// ([`stream_chunks`], [`matmul::StreamLayout`]): chunk `c` gathers its `B` tiles into half `c
    /// % 2`, its kernel reloads the partial the chunk before packed in the other half (none for
    /// the first) and packs its own, and only the last chunk's is written out. Consecutive
    /// chunks alternate halves, so the session double-buffers them (`Step::Kernel::half`): the
    /// mover reads chunk `c + 1` while the kernel computes chunk `c`. Each output tile's `K`
    /// chain is the same in the same order, through FP32 partials as a K block's are, so bit
    /// for bit. The packed `A` is read once a unit, by its first chunk.
    fn push_stream_jobs(
        &self,
        jobs: &mut Vec<Job>,
        [ra, rb, rc]: [TensorRef; 3],
        [a_transposed, b_transposed]: [bool; 2],
    ) -> Result<()> {
        let [mt, kt, nt] = self.tiles;
        let packed = self.packed_a.expect("a streamed product's packed A");
        debug_assert_eq!(mt, 1);
        let unpack_b = self.unpack_b && b_transposed;
        let (kq, layout) = stream_chunks(kt, self.k_block, self.route, self.fidelity)?;
        let chunks = kt.div_ceil(kq);
        let units = self.units.max(1);
        let mut a_read = vec![false; units];
        let live = self.live_rows(0, 1);
        // A column a job: the plan's runs' columns, in order, or every column.
        let cols: Vec<usize> = match &self.col_runs {
            Some(runs) => runs.iter().flat_map(|r| r.clone()).collect(),
            None => (0..nt).collect(),
        };
        for j in cols {
            let unit = jobs.len() % units;
            let mut steps = Vec::new();
            for c in 0..chunks {
                let (k0, h) = (c * kq, c % 2);
                let count = kq.min(kt - k0);
                let (chunk, reload) = layout.chunk(c, k0, count);
                let mut gather = Vec::new();
                if !a_read[unit] {
                    gather.extend(packed_a_reads(packed, 0, kt, layout.a_at));
                    a_read[unit] = true;
                }
                let mut br = rb;
                br.first += if b_transposed {
                    k0 as u32
                } else {
                    (k0 * br.ct as usize) as u32
                };
                let flags = u32::from(a_transposed)
                    | u32::from(b_transposed) << 1
                    | (count as u32) << 8
                    | record::GATHER_A_RESIDENT
                    | record::GATHER_STREAM
                    | if unpack_b {
                        record::GATHER_B_UNPACK_TRANSPOSE
                    } else {
                        0
                    };
                let gather = probe_gather(
                    gather,
                    [
                        [
                            record::GATHER,
                            flags,
                            chunk.a_at as u32,
                            chunk.b_at as u32,
                            0,
                            1,
                            j as u32,
                            1,
                        ],
                        ra.encode()[0],
                        ra.encode()[1],
                        br.encode()[0],
                        br.encode()[1],
                    ],
                    true,
                );
                // The programs name this chunk's `K` range in the packed `A`, its half and the
                // layout `A`'s slots fix.
                let variant = 0x1_0000 | (packed_a_slots(kt) as u32) << 8 | c as u32;
                let (route, fidelity) = (self.route, self.fidelity);
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
                let scatter = if c + 1 == chunks {
                    vec![
                        [
                            record::SCATTER,
                            chunk.outputs[0].out as u32,
                            TILE_SLOT as u32,
                            0,
                            1,
                            j as u32,
                            1,
                            0,
                        ],
                        rc.encode()[0],
                        rc.encode()[1],
                    ]
                } else {
                    Vec::new()
                };
                steps.extend([
                    Step::List {
                        what: "streamed matmul gather",
                        entries: gather,
                    },
                    Step::Kernel {
                        roles,
                        init: chunk.init.clone(),
                        mop: Box::new([None; 3]),
                        loops: Default::default(),
                        half: Some(h as u8),
                    },
                    Step::List {
                        what: "streamed matmul scatter",
                        entries: scatter,
                    },
                ]);
            }
            jobs.push(steps);
        }
        Ok(())
    }
}

/// Size both first and continuation programs against the resident-cache limit.
pub(crate) fn k_block_size(mut kc: usize, route: SrcRoute, fidelity: Fidelity) -> Result<usize> {
    loop {
        if let Ok((layout, reload)) = matmul::k_block_layout(kc) {
            let fits = [None, Some(reload)].into_iter().all(|r| {
                matmul::k_block_kernel(&layout, r, route, fidelity, false, matmul::LiveRows::All)
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
        let unpack_b = self.unpack_b && transposed[1];
        // One output tile per job keeps its prior accumulator and every K
        // continuation on the same unit; outputs remain independent.
        for i in 0..mt {
            for j in 0..nt {
                let mut steps = Vec::new();
                for k0 in (0..kt).step_by(self.k_block) {
                    let count = self.k_block.min(kt - k0);
                    let (layout, reload) = match self.packed_a {
                        Some(_) => matmul::k_block_layout_with(count, packed_a_slots(count))?,
                        None => matmul::k_block_layout(count)?,
                    };
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
                    let thin = if i + 1 == mt && self.packed_a.is_none() {
                        self.thin
                    } else {
                        0
                    };
                    let flags = u32::from(transposed[0])
                        | u32::from(transposed[1]) << 1
                        | record::gather_rows(thin)
                        | (count as u32) << 8
                        | if probing_skip_a() || self.packed_a.is_some() {
                            record::GATHER_A_RESIDENT
                        } else {
                            0
                        }
                        | if self.packed_a.is_some() {
                            record::GATHER_STREAM
                        } else {
                            0
                        }
                        | if unpack_b {
                            record::GATHER_B_UNPACK_TRANSPOSE
                        } else {
                            0
                        };
                    let gather = match self.packed_a {
                        Some(pa) => packed_a_reads(pa, k0, count, layout.a_at),
                        None => Vec::new(),
                    };
                    let mut gather = probe_gather(
                        gather,
                        [
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
                        ],
                        self.packed_a.is_some(),
                    );
                    let layout = match self.packed_a {
                        Some(_) => matmul::Layout {
                            outputs: matmul::packed_a_pairs(&layout.outputs, layout.a_at, k0),
                            ..layout
                        },
                        None => layout,
                    };
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
                    let variant = if k0 > 0 { 0x81 } else { 0x80 }
                        | if self.packed_a.is_some() { 0x40 } else { 0 };
                    let live = self.live_rows(i, 1);
                    let (roles, _) = matmul::kernel_programs(
                        [1, count, 1],
                        variant,
                        self.route,
                        self.fidelity,
                        layout.sems,
                        false,
                        unpack_b,
                        live,
                        || {
                            (
                                matmul::k_block_kernel(
                                    &layout,
                                    (k0 > 0).then_some(reload),
                                    self.route,
                                    self.fidelity,
                                    unpack_b,
                                    live,
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

/// What [`block_stream_plan`]'s model takes a unit to spend on a K-tile pair, on an output tile's
/// reload and pack a chunk, and on a chunk's batch (its launch, credits and retire), in us; and
/// GDDR's read rates, a card's and one unit's, in bytes a us. From card 0's sweeps
/// (`docs/plans/prefill-operand-reuse.md`, "Measured": a `kq` sweep of a `4 x 6` block of `down`,
/// `1792857249`-`1792857286`, ~3-5 us a chunk of 24 outputs; blocks of 24-64 outputs, 0.4-0.6 us a
/// pair at HiFi4). The card's rate is what a product reading under its compute reached: shapes
/// that asked ~400 GB/s ran slower than their compute, none of the fast ones read more than ~315.
const BLOCK_PAIR_US: f64 = 0.4;
const BLOCK_OUTPUT_US: f64 = 0.15;
const BLOCK_BATCH_US: f64 = 2.0;
const CARD_READ_BYTES_PER_US: f64 = 320_000.0;
const UNIT_READ_BYTES_PER_US: f64 = 7_000.0;
const TILE_BYTES: f64 = 4096.0;

/// The longest side of a block [`block_stream_plan`] considers: the kernel's programs are
/// unrolled an output tile (P1b loops them).
pub const BLOCK_SIDE_MAX: usize = 16;

/// The programs [`block_stream_plan`] builds to check a candidate's fit, at most, before it gives
/// up on a product.
const BLOCK_FIT_TRIES: usize = 8;

/// A K-streamed blocked product's shape ([`BlockStream`]): `[mb, kq, nb]`, its chunks a job, and
/// what [`block_stream_plan`]'s model makes of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockStreamShape {
    pub tiles: [usize; 3],
    pub chunks: usize,
    /// Modelled time, us.
    pub us: f64,
    /// GDDR bytes read.
    pub bytes: f64,
}

/// [`BlockStreamShape`] of `[mt, kt, nt]` tiles in `[mb, kq, nb]` blocks over `units` units: `None`
/// where the chunks a job are odd (a unit's chunks must alternate halves from job to job) or the
/// block is wider than the product.
pub fn block_stream_model(
    [mt, kt, nt]: [usize; 3],
    [mb, kq, nb]: [usize; 3],
    units: usize,
) -> Option<BlockStreamShape> {
    if mb == 0 || kq == 0 || nb == 0 || mb > mt || nb > nt || kq > kt {
        return None;
    }
    let chunks = kt.div_ceil(kq);
    if chunks % 2 != 0 {
        return None;
    }
    let (row_blocks, col_blocks) = (mt.div_ceil(mb), nt.div_ceil(nb));
    let jobs = row_blocks * col_blocks;
    let rounds = jobs.div_ceil(units.max(1)) as f64;
    // Every block reads its rows' `A` and its columns' `B` once, over the whole of `K`.
    let bytes = (col_blocks * mt + row_blocks * nt) as f64 * kt as f64 * TILE_BYTES;
    let compute = chunks as f64 * BLOCK_BATCH_US
        + (mb * nb) as f64 * (kt as f64 * BLOCK_PAIR_US + chunks as f64 * BLOCK_OUTPUT_US);
    let read = (mb + nb) as f64 * kt as f64 * TILE_BYTES / UNIT_READ_BYTES_PER_US;
    // A chunk's reads run under the chunk before's compute.
    let us = (bytes / CARD_READ_BYTES_PER_US).max(rounds * compute.max(read));
    Some(BlockStreamShape {
        tiles: [mb, kq, nb],
        chunks,
        us,
        bytes,
    })
}

/// The same model of the product's other paths: `[mc, kc, nc]` as `matmul_dram_prefetching`
/// runs it (after [`blocks`]), `K` whole (its reads and compute one after another under
/// [`Staging::Slots`]) or in K blocks of one output tile, the partial through GDDR.
pub fn blocked_paths_model(
    [mt, kt, nt]: [usize; 3],
    [mc, kc, nc]: [usize; 3],
    units: usize,
) -> f64 {
    let units = units.max(1);
    if kc >= kt {
        let (row_blocks, col_blocks) = (mt.div_ceil(mc), nt.div_ceil(nc));
        let rounds = (row_blocks * col_blocks).div_ceil(units) as f64;
        let bytes = (col_blocks * mt + row_blocks * nt) as f64 * kt as f64 * TILE_BYTES;
        let job = BLOCK_BATCH_US
            + (mc * nc) as f64 * (kt as f64 * BLOCK_PAIR_US + BLOCK_OUTPUT_US)
            + (mc + nc) as f64 * kt as f64 * TILE_BYTES / UNIT_READ_BYTES_PER_US;
        (bytes / CARD_READ_BYTES_PER_US).max(rounds * job)
    } else {
        let blocks = kt.div_ceil(kc) as f64;
        let rounds = (mt * nt).div_ceil(units) as f64;
        let tiles = 2.0 * kt as f64 + 2.0 * (blocks - 1.0);
        let bytes = (mt * nt) as f64 * tiles * TILE_BYTES;
        let job = blocks * (BLOCK_BATCH_US + BLOCK_OUTPUT_US)
            + kt as f64 * BLOCK_PAIR_US
            + tiles * TILE_BYTES / UNIT_READ_BYTES_PER_US;
        (bytes / CARD_READ_BYTES_PER_US).max(rounds * job)
    }
}

/// The largest `kq <= limit` whose chunks a job are even, or the smallest above it.
fn even_chunks(kt: usize, limit: usize) -> Option<usize> {
    let even = |kq: usize| kt.div_ceil(kq) % 2 == 0;
    (1..=limit.min(kt))
        .rev()
        .find(|&kq| even(kq))
        .or_else(|| (limit + 1..=kt).find(|&kq| even(kq)))
}

/// Whether a block's first and continuing chunks' programs, as they run
/// ([`matmul::block_stream_programs`]), fit a role's program slot and half the program cache, as
/// [`k_block_size`] asks of a K block's.
fn block_stream_programs_fit(
    layout: &matmul::BlockStreamLayout,
    route: SrcRoute,
    fidelity: Fidelity,
) -> bool {
    (0..2).all(|c| {
        let (roles, loops) = matmul::block_stream_programs(
            layout,
            c,
            layout.tiles,
            route,
            fidelity,
            false,
            matmul::LiveRows::All,
        );
        roles.iter().zip(loops.iter()).all(|(p, l)| {
            let code = crate::code::Code {
                ins: p.clone(),
                loops: l.clone(),
            };
            code.stored().is_ok_and(|(words, _)| {
                words.len() <= tt_isa::mailbox::PROGRAM_MAX as usize
                    && crate::program_cache::admitted(words.len())
            })
        })
    })
}

/// The shape and layout a `[mt, kt, nt]`-tile product runs in under `mode` ([`BlockStream`]) on
/// `units` units, in the current arena; `None` where it runs another way: [`BlockStream::Off`], a
/// product of one tile row or one `K` tile, no shape that fits, or under [`BlockStream::Auto`] none
/// whose modelled time beats `other_us` (the other paths', [`blocked_paths_model`]) by a tenth.
/// Planned once a process for each question.
pub(crate) fn block_stream_plan(
    mode: BlockStream,
    [mt, kt, nt]: [usize; 3],
    units: usize,
    other_us: f64,
    route: SrcRoute,
    fidelity: Fidelity,
) -> Option<(BlockStreamShape, matmul::BlockStreamLayout)> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    if mode == BlockStream::Off || mt < 2 || kt < 2 {
        return None;
    }
    type Key = (BlockStream, [usize; 3], usize, u64, SrcRoute, Fidelity, u64);
    type Value = Option<(
        BlockStreamShape,
        matmul::BlockStreamLayout,
        crate::l1::Planned,
    )>;
    static CACHE: OnceLock<Mutex<HashMap<Key, Value>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let arena = crate::l1::arena();
    let key = (
        mode,
        [mt, kt, nt],
        units,
        other_us.to_bits(),
        route,
        fidelity,
        arena.end,
    );
    if let Some(v) = cache.lock().unwrap().get(&key) {
        // What the layout needs of the arena, noted again as a fresh plan would.
        return v.clone().map(|(s, l, p)| {
            crate::l1::note_planned(p);
            (s, l)
        });
    }
    let fits_arena = |[mb, kq, nb]: [usize; 3]| {
        // Two semaphores pairs' worth of slack past the slots.
        (2 * (kq * (mb + nb) + mb * nb)) as u64 * TILE_SLOT + 256 <= arena.len()
    };
    let mut candidates: Vec<BlockStreamShape> = match mode {
        BlockStream::Off => Vec::new(),
        BlockStream::Fixed([mb, kq, nb]) => {
            // The program cache's variant holds six bits a side ([`push_block_stream_jobs`]).
            let (mb, nb) = (mb.min(mt).min(63), nb.min(nt).min(63));
            even_chunks(kt, kq)
                .and_then(|kq| block_stream_model([mt, kt, nt], [mb, kq, nb], units))
                .into_iter()
                .collect()
        }
        BlockStream::Auto => {
            let mut kqs: Vec<usize> = (1..=kt).filter(|&kq| kt.div_ceil(kq) % 2 == 0).collect();
            // Of the `kq`s with the same chunks a job, the shortest: the same work and reloads
            // in fewer slots.
            kqs.dedup_by_key(|kq| kt.div_ceil(*kq));
            let mut all = Vec::new();
            for mb in 1..=mt.min(BLOCK_SIDE_MAX) {
                for nb in 1..=nt.min(BLOCK_SIDE_MAX) {
                    for &kq in &kqs {
                        if fits_arena([mb, kq, nb]) {
                            all.extend(block_stream_model([mt, kt, nt], [mb, kq, nb], units));
                        }
                    }
                }
            }
            all
        }
    };
    candidates.sort_by(|a, b| a.us.total_cmp(&b.us).then(a.bytes.total_cmp(&b.bytes)));
    // The candidates' layouts are measurements: only the chosen one's plan is noted.
    let chosen = crate::l1::unnoted(|| {
        candidates
            .into_iter()
            .filter(|s| mode != BlockStream::Auto || s.us < 0.9 * other_us)
            .filter(|s| fits_arena(s.tiles))
            .take(BLOCK_FIT_TRIES)
            .find(|s| {
                matmul::block_stream_layout(s.tiles)
                    .is_ok_and(|layout| block_stream_programs_fit(&layout, route, fidelity))
            })
    });
    let value = chosen.and_then(|s| {
        let before = crate::l1::take_planned();
        let layout = matmul::block_stream_layout(s.tiles);
        let planned = crate::l1::take_planned();
        crate::l1::note_planned(before);
        crate::l1::note_planned(planned);
        Some((s, layout.ok()?, planned))
    });
    if let (BlockStream::Fixed(asked), None) = (mode, &value) {
        // Once a question (the cache keeps the answer): a sweep must not time the other paths
        // as if they were the blocked one.
        eprintln!(
            "TT_MATMUL_BLOCK={asked:?}: no such blocked product of {mt} x {kt} x {nt} tiles \
             fits (L1 or a role's program); it runs the other paths"
        );
    }
    cache.lock().unwrap().insert(key, value.clone());
    value.map(|(s, l, _)| (s, l))
}

/// The shape [`block_stream_plan`] chooses, without its layout: what a gate or a benchmark reports.
pub fn block_stream_choice(
    mode: BlockStream,
    tiles: [usize; 3],
    units: usize,
    other_us: f64,
    route: SrcRoute,
    fidelity: Fidelity,
) -> Option<BlockStreamShape> {
    crate::l1::unnoted(|| block_stream_plan(mode, tiles, units, other_us, route, fidelity))
        .map(|(s, _)| s)
}

/// A gate's negative control (`step229_block_stream_product`): a blocked product's chunks take
/// their `K` ranges last first, so each output tile's chain is accumulated out of order. **The
/// products are wrong** wherever the order matters.
static PROBE_REVERSE_CHUNKS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Turn [`PROBE_REVERSE_CHUNKS`] on or off, process-wide. A gate's control, never a result.
#[doc(hidden)]
pub fn probe_reverse_chunks(on: bool) {
    PROBE_REVERSE_CHUNKS.store(on, std::sync::atomic::Ordering::SeqCst);
}

/// One job a block of a K-streamed blocked product ([`BlockStream`], [`matmul::BlockStreamLayout`]):
/// chunk `c` gathers its `A` and `B` tiles into input half `c % 2`, its kernel computes every output
/// tile of the block over the chunk's pairs, continuing each from its partial in output set
/// `1 - c % 2` (none for the first chunk) and packing it into set `c % 2`, and only the last chunk's
/// are written out. A job's chunks are even, so a unit's batches alternate halves throughout and the
/// session double-buffers them (`Step::Kernel::half`): the mover reads chunk `c + 1` while the kernel
/// computes chunk `c`. Each output tile's `K` chain is the same in the same order, through FP32
/// partials as a K block's are, so bit for bit.
#[allow(clippy::too_many_arguments)]
fn push_block_stream_jobs(
    jobs: &mut Vec<Job>,
    shape: &BlockStreamShape,
    layout: &matmul::BlockStreamLayout,
    [mt, kt, nt]: [usize; 3],
    m: usize,
    [ra, rb, rc]: [TensorRef; 3],
    [a_transposed, b_transposed]: [bool; 2],
    thin: u32,
    unpack_b: bool,
    route: SrcRoute,
    fidelity: Fidelity,
) {
    let [mb, kq, nb] = shape.tiles;
    let chunks = shape.chunks;
    debug_assert_eq!(chunks % 2, 0);
    for i0 in (0..mt).step_by(mb) {
        let rows = mb.min(mt - i0);
        let live = matmul::LiveRows::for_job(m, i0, rows);
        let thin = if i0 + rows == mt { thin } else { 0 };
        for j0 in (0..nt).step_by(nb) {
            let cols = nb.min(nt - j0);
            let mut steps = Vec::with_capacity(3 * chunks);
            let reversed = PROBE_REVERSE_CHUNKS.load(std::sync::atomic::Ordering::SeqCst);
            for c in 0..chunks {
                let range = if reversed { chunks - 1 - c } else { c };
                let (k0, h) = (range * kq, c % 2);
                let count = kq.min(kt - k0);
                let (chunk, _) = layout.chunk(c, [rows, count, cols]);
                let (mut ar, mut br) = (ra, rb);
                ar.first += if a_transposed {
                    (k0 * ar.ct as usize) as u32
                } else {
                    k0 as u32
                };
                br.first += if b_transposed {
                    k0 as u32
                } else {
                    (k0 * br.ct as usize) as u32
                };
                let flags = u32::from(a_transposed)
                    | u32::from(b_transposed) << 1
                    | record::gather_rows(thin)
                    | (count as u32) << 8
                    | if probing_skip_a() {
                        record::GATHER_A_RESIDENT
                    } else {
                        0
                    }
                    | if unpack_b {
                        record::GATHER_B_UNPACK_TRANSPOSE
                    } else {
                        0
                    };
                let gather = probe_gather(
                    Vec::new(),
                    [
                        [
                            record::GATHER,
                            flags,
                            chunk.a_at as u32,
                            chunk.b_at as u32,
                            i0 as u32,
                            rows as u32,
                            j0 as u32,
                            cols as u32,
                        ],
                        ar.encode()[0],
                        ar.encode()[1],
                        br.encode()[0],
                        br.encode()[1],
                    ],
                    false,
                );
                let (roles, loops) = matmul::block_stream_programs(
                    layout,
                    c,
                    [rows, count, cols],
                    route,
                    fidelity,
                    unpack_b,
                    live,
                );
                let scatter = if c + 1 == chunks {
                    vec![
                        [
                            record::SCATTER,
                            (layout.out_slot(h, cols, 0, 0) + TILE_DATA) as u32,
                            TILE_SLOT as u32,
                            i0 as u32,
                            rows as u32,
                            j0 as u32,
                            cols as u32,
                            0,
                        ],
                        rc.encode()[0],
                        rc.encode()[1],
                    ]
                } else {
                    Vec::new()
                };
                steps.extend([
                    Step::List {
                        what: "blocked matmul gather",
                        entries: gather,
                    },
                    Step::Kernel {
                        roles,
                        init: layout.init.clone(),
                        mop: Box::new([None; 3]),
                        loops,
                        half: Some(h as u8),
                    },
                    Step::List {
                        what: "blocked matmul scatter",
                        entries: scatter,
                    },
                ]);
            }
            jobs.push(steps);
        }
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

/// Whether a block of `extent` elements at `at` in `t` is one
/// [`matmul_dram_batched`] reads where it lies: inside `t`, starting on a
/// tile and, where it does not end on one, ending at `t`'s own edge.
pub fn block_fits(t: &DramTensor, [r0, c0]: [usize; 2], [r, c]: [usize; 2]) -> bool {
    r0 + r <= t.rows
        && c0 + c <= t.cols
        && r0 % 32 == 0
        && c0 % 32 == 0
        && (r % 32 == 0 || r0 + r == t.rows)
        && (c % 32 == 0 || c0 + c == t.cols)
}

/// [`block_ref`] for an operand laid out a block per whole tile rows, each
/// block's tail rows its own zero padding: the block must start on a tile and
/// its tile rows lie inside `t`, but may end inside it.
fn padded_block_ref(
    t: &DramTensor,
    at: [usize; 2],
    extent: [usize; 2],
    what: &str,
) -> Result<TensorRef> {
    let ([r0, c0], [r, c]) = (at, extent);
    let fits = r0 % 32 == 0
        && r0 + r.div_ceil(32) * 32 <= t.grid()[0] * 32
        && block_fits(t, [0, c0], [t.rows, c]);
    if !fits {
        return Err(TensorError::Shape(format!(
            "{what}: a padded [{r}, {c}] block at [{r0}, {c0}] of a [{}, {}] tensor is not \
             whole tile rows inside it",
            t.rows, t.cols
        )));
    }
    let mut x = t.tensor_ref();
    x.first += ((r0 / 32) * t.grid()[1] + c0 / 32) as u32;
    Ok(x)
}

/// A block of `extent` elements at `at` in `t`, as a record names it: the
/// tensor's ref with its first tile moved to the block's, the row stride
/// kept. Refused unless the block starts on a tile and, where it does not end
/// on one, ends at the tensor's own edge -- a ragged edge inside the tensor
/// would read its neighbour's elements as padding.
pub(crate) fn block_ref(
    t: &DramTensor,
    at: [usize; 2],
    extent: [usize; 2],
    what: &str,
) -> Result<TensorRef> {
    let ([r0, c0], [r, c]) = (at, extent);
    if !block_fits(t, at, extent) {
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
/// jobs together, each product's output written from tile row `i mt` (`mt`
/// the tile rows of `m`) of one `[items.len() mt 32, n]` tensor -- the stacked
/// `[items.len() m, n]` when `m` is whole tiles; otherwise each product's last
/// tile row ends in padding rows of its own ([`Session::matmul_dram_batched`]
/// compacts them). A batched matmul of resident operands, and of views into
/// them (a head's columns of a projection), with no copy: each block is
/// gathered where it lies. Every product is [`matmul_dram`]'s, bit for bit,
/// including accumulator reloads across K blocks.
///
/// [`Session::matmul_dram_batched`]: crate::session::Session::matmul_dram_batched
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
        [false; 2],
        false,
    )
}

/// [`matmul_dram_batched`] with a K block limit, and `padded` saying which
/// operands are laid out a block per whole tile rows, each block's tail rows
/// zero ([`Session::matmul_dram_batched`] makes them): such a block may end
/// inside the tensor, since what follows it to the tile boundary is its own
/// zero padding, not a neighbour's elements.
///
/// [`Session::matmul_dram_batched`]: crate::session::Session::matmul_dram_batched
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
    padded: [bool; 2],
    unpack_b: bool,
) -> Result<Work> {
    a.expect("a matmul", Elem::F32)?;
    b.expect("a matmul", Elem::F32)?;
    let batch = items.len();
    if batch == 0 || m == 0 || k == 0 || n == 0 {
        return Err(TensorError::Shape(format!(
            "{batch} x [{m}, {k}] @ [{k}, {n}]: nothing to compute"
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
        let operand = |t, padded, at, extent, what| {
            if padded {
                padded_block_ref(t, at, extent, what)
            } else {
                block_ref(t, at, extent, what)
            }
        };
        refs.push((
            operand(a, padded[0], ba.at, ea, "a batched matmul's left operand")?,
            operand(b, padded[1], bb.at, eb, "a batched matmul's right operand")?,
            [ba.transposed, bb.transposed],
        ));
    }
    let plan = BlockPlan {
        tiles: [mt, kt, nt],
        m,
        block: blocks([mt, nt], [mc, nc], per_item),
        k_block: kc,
        staging,
        units,
        route,
        fidelity,
        allow_mop,
        // Whole tiles: attention's products are not the decode weights' reads
        // (decode-matmul-bandwidth P1), and their padding claims stay as they were.
        thin: 0,
        resident_a: false,
        col_runs: None,
        unpack_b: unpack_b && route == SrcRoute::Tf32FromFp32,
        packed_a: None,
        stream_k: false,
        busy: Vec::new(),
        second_job: false,
        balanced_deal: false,
    };
    let rows = if batch == 1 { m } else { batch * mt * 32 };
    let c = DramTensor::alloc(alloc, rows, n)?;
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
/// `want` staging slots, or as many as the arena holds ([`crate::l1::arena`]) where fewer: an even
/// number, at least two, so batches can alternate halves.
fn arena_group(want: usize) -> usize {
    let slots = (crate::l1::arena().len() / TILE_SLOT) as usize;
    want.min(slots / 2 * 2).max(2)
}

fn staging(name: &'static str, slots: usize) -> Result<u64> {
    let mut req = crate::l1::Requirements::new(1);
    let b = req.scratch(name, slots as u64 * TILE_SLOT, tt_isa::dram::ALIGN, 0..1);
    let plan = req
        .plan(crate::l1::arena())
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
/// shape, one row (`[1, cols]`), one column (`[rows, 1]`) or a block of whole
/// tile rows repeated down `a`'s (`[R, cols]`, `R` a multiple of 32 dividing
/// `a`'s rows), the last three only for the kinds that take one
/// (`sfpu::ops::broadcasts`). Refuses anything else.
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
        (true, Some(b)) if broadcasts(op.kind) && block_repeats(a, b).is_some() => {
            Ok((op.kind, Broadcast::Block))
        }
        (true, Some(b)) => Err(TensorError::Shape(format!(
            "[{}, {}] and [{}, {}]: neither the same shape nor a row, a column or a block of whole \
             tile rows to broadcast",
            a.rows, a.cols, b.rows, b.cols
        ))),
    }
}

/// How many times `b`'s rows repeat down `a`'s, if `b` is a [`Broadcast::Block`] of `a`: as
/// wide, at least two repeats, and its rows whole tiles (so a repeat never splits one).
///
/// [`Broadcast::Block`]: crate::sfpu::ops::Broadcast::Block
pub fn block_repeats(a: &DramTensor, b: &DramTensor) -> Option<usize> {
    (b.cols == a.cols && b.rows % 32 == 0 && b.rows > 0 && b.rows < a.rows && a.rows % b.rows == 0)
        .then(|| a.rows / b.rows)
}

/// `B`'s reads for a [`Broadcast::Block`] run over `A`'s tiles `first..first + count` (row-major,
/// `ct` a tile row), the slots from `at` on: output tile `t` reads `B`'s tile `t % period`
/// (`period` its tiles, `B`'s tile rows times `ct`), as plain `READ_RUN`s over `B`'s own grid, one
/// for each stretch between wraps. What a row gather expanding `B` to `A`'s shape would put in
/// each slot, with no list of rows.
///
/// [`Broadcast::Block`]: crate::sfpu::ops::Broadcast::Block
fn block_reads(first: usize, count: usize, period: usize, at: u64, ct: usize) -> Vec<[u32; 8]> {
    let mut out = Vec::new();
    let mut n = 0;
    while n < count {
        let t = (first + n) % period;
        let len = (period - t).min(count - n);
        out.push([
            record::READ_RUN,
            t as u32,
            len as u32,
            (at + n as u64 * TILE_SLOT) as u32,
            0,
            ct as u32,
            0,
            0,
        ]);
        n += len;
    }
    out
}

/// The element type `kind`'s output has.
pub(crate) fn output_elem(kind: u32) -> Elem {
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
#[allow(clippy::too_many_arguments)]
pub fn sfpu_eltwise(
    alloc: &mut DramAlloc,
    op: Eltwise,
    a: &DramTensor,
    b: Option<&DramTensor>,
    c: Option<&DramTensor>,
    units: usize,
    overlap: Overlap,
    twin: Option<&DramTensor>,
) -> Result<Option<Work>> {
    let (kind, bcast) = broadcast_of(op, a, b)?;
    if crate::sfpu::integer::operation(kind).is_some_and(|(op, _)| op == 15 || op == 16) {
        return sfpu_divide(alloc, Eltwise { kind, ..op }, a, b, bcast).map(Some);
    }
    sfpu_eltwise_blocks(
        alloc,
        op,
        BlockOperand::whole(a),
        b.map(BlockOperand::whole),
        c.map(BlockOperand::whole),
        units,
        overlap,
        twin,
    )
}

/// The writes that put one-row tiles' data rows, packed, into `twin` (`[K / 32, 32]`, as
/// [`pack_row`] lays them out): output tiles `first..first + count`, each in the slot after the
/// last from `at` with its data in row 0, two 64-byte face-row writes a tile on the writer.
pub fn packed_twin_writes(twin: &DramTensor, first: usize, count: usize, at: u64) -> Vec<[u32; 8]> {
    use tt_isa::dm::op;
    const FACE: u64 = 1024;
    (0..count)
        .flat_map(|n| {
            let (kk, slot) = (first + n, at + n as u64 * TILE_SLOT);
            let (dst, r) = (twin.tile(kk / 32, 0), (kk % 32) as u64);
            (0..2u64).map(move |half| {
                [
                    op::WRITE,
                    dst.channel().index() as u32,
                    0,
                    (dst.offset() + TILE_DATA + (2 * (r / 16) + half) * FACE + (r % 16) * 64)
                        as u32,
                    (slot + TILE_DATA + half * FACE) as u32,
                    64,
                    0,
                    0,
                ]
            })
        })
        .collect()
}

/// One operand of [`sfpu_eltwise_blocks`]: the `dims` block of `tensor` whose
/// top-left element is `at`, read where it lies ([`block_fits`]: on tile
/// boundaries, ragged only at the tensor's own edge) -- a gate's columns of a
/// fused projection, a head's columns of `q`.
#[derive(Copy, Clone, Debug)]
pub struct BlockOperand<'a> {
    pub tensor: &'a DramTensor,
    pub at: [usize; 2],
    pub dims: [usize; 2],
}

impl<'a> BlockOperand<'a> {
    /// All of `t`.
    pub fn whole(t: &'a DramTensor) -> Self {
        BlockOperand {
            tensor: t,
            at: [0, 0],
            dims: [t.rows, t.cols],
        }
    }

    /// The block as a tensor of its own for what reads only its shape, type
    /// and padding (`broadcast_of`, `OpPadding::produces`): never for its
    /// slots, whose row stride is the parent's. Its padding is the parent's
    /// where the block is ragged (only ever at the parent's edge), and none
    /// where it is whole tiles.
    pub fn shape_only(&self) -> DramTensor {
        let t = DramTensor {
            rows: self.dims[0],
            cols: self.dims[1],
            elem: self.tensor.elem,
            placement: Placement {
                owned: false,
                ..self.tensor.placement.clone()
            },
            pad: std::cell::Cell::new(self.tensor.pad()),
        };
        t.set_pad(self.tensor.pad());
        t
    }

    fn tensor_ref(&self, what: &str) -> Result<TensorRef> {
        block_ref(self.tensor, self.at, self.dims, what)
    }
}

/// [`sfpu_eltwise`] over blocks of resident tensors ([`BlockOperand`]), the
/// result a new tensor of `a`'s block's dims: each operand's tiles read where
/// they lie at its parent's row stride (`record::READ_RUN` counts over the
/// output's grid), so a view that is a block of its buffer needs no copy
/// first. Integer division is refused here (it reads whole tensors).
#[allow(clippy::too_many_arguments)]
pub fn sfpu_eltwise_blocks(
    alloc: &mut DramAlloc,
    op: Eltwise,
    a: BlockOperand<'_>,
    b: Option<BlockOperand<'_>>,
    c: Option<BlockOperand<'_>>,
    units: usize,
    overlap: Overlap,
    twin: Option<&DramTensor>,
) -> Result<Option<Work>> {
    use crate::sfpu::kernel::Operands;
    use crate::sfpu::ops::Broadcast;
    let what = "an element-wise op's block";
    let (ra, rb, rc) = (
        a.tensor_ref(what)?,
        b.map(|b| b.tensor_ref(what)).transpose()?,
        c.map(|c| c.tensor_ref(what)).transpose()?,
    );
    let (a, b, c) = (
        a.shape_only(),
        b.map(|b| b.shape_only()),
        c.map(|c| c.shape_only()),
    );
    let (a, b, c) = (&a, b.as_ref(), c.as_ref());
    let (kind, bcast) = broadcast_of(op, a, b)?;
    let op = Eltwise { kind, ..op };
    if crate::sfpu::integer::operation(kind).is_some_and(|(op, _)| op == 15 || op == 16) {
        return Err(TensorError::Shape(
            "integer division reads whole tensors, not blocks".into(),
        ));
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
            rc
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
    let ro = out.tensor_ref();
    let rows = eltwise_rows(op.kind, a.rows);
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
        let (layout, roles, loops) = match sfpu_programs(op, bcast, operands, len, half, rows) {
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
            // A column's broadcast is the program's (`Program::broadcast_col0`): the
            // mover's datum-at-a-time copy cost ~10 us a tile.
            let flags = match bcast {
                Broadcast::None | Broadcast::Block => 0,
                Broadcast::Row => 1,
                Broadcast::Col => record::READ_RUN_COL_AS_IT_LIES,
            };
            if bcast == Broadcast::Block {
                let b = b.expect("a block broadcast's operand");
                for h in block_reads(run.start, len, b.grid()[0] * ct, b_at, ct) {
                    gather.extend([h, rb.encode()[0], rb.encode()[1]]);
                }
            } else {
                gather.extend(read(rb, b_at, flags));
            }
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
        let mut scatter = scatter.to_vec();
        if let Some(twin) = twin {
            // A one-row result's data rows, packed as a product's `A` and RMSNorm's statistic
            // read them, from the slots the scatter writes out: no packing op.
            if (a.rows, twin.rows, twin.cols) != (1, ct, 32) {
                alloc.free(&out.placement);
                return Err(TensorError::Shape(format!(
                    "a packed twin [{}, {}] for a [{}, {}] result",
                    twin.rows, twin.cols, a.rows, a.cols
                )));
            }
            scatter.extend(packed_twin_writes(twin, run.start, len, layout.out_at));
        }
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
                entries: scatter,
            },
        ]);
    }
    Ok(Some(Work { out, jobs }))
}

/// The rows of each tile an element-wise op on a tensor of `rows` rows computes
/// ([`crate::sfpu::ops::code_for_rows`]): a single tile row's own rows (a decode step's `[1, W]`
/// activation is one of 32) for a [`crate::sfpu::ops::row_limited`] kind, else 32. Below 32 the
/// output's padding rows are whatever `Dst` held: `Pad::Undefined`.
pub fn eltwise_rows(kind: u32, rows: usize) -> u32 {
    if rows < 32 && crate::sfpu::ops::row_limited(kind) {
        rows as u32
    } else {
        32
    }
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
    if bcast == Broadcast::Block {
        return Err(TensorError::Shape(format!(
            "integer division {:#x} by a block of rows repeated: not implemented",
            op.kind
        )));
    }
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
                Broadcast::None | Broadcast::Block => 0,
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
        .plan(crate::l1::arena())
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
        let mut next = 0;
        while next < positions.len() {
            let mut read = Vec::new();
            if batches.is_empty() {
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
            let mut taken = 0;
            // Up to 200 words a batch, fewer when each needs its own fill: a word costs two
            // entries unless it repeats the one before, and a side must fit a packet
            // ([`TRANSFER_SIDE_MAX`]); the word count alone cannot bound it.
            while taken < 200 && next + taken < positions.len() {
                let (value, index) = positions[next + taken];
                let fresh = prior != Some(value);
                if read.len() + 1 + usize::from(fresh) > TRANSFER_SIDE_MAX {
                    break;
                }
                if fresh {
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
                taken += 1;
            }
            next += taken;
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
    // By the arena's end too (`crate::l1::arena`): a group is sized from it.
    type Memo = Mutex<HashMap<(u32, u32, u32, crate::sfpu::ops::Broadcast, bool, u64), usize>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let key = (
        op.kind,
        op.scalar.to_bits(),
        op.scalar2.to_bits(),
        bcast,
        half,
        crate::l1::arena().end,
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
    // The programs' lengths, which do not depend on where the layout sits: measured in the whole of
    // `DATA`, whatever the arena, and not noted as the op's.
    let lens = |n: usize| -> [usize; 3] {
        let layout = crate::l1::unnoted(|| {
            crate::sfpu::kernel::plan_layout_in(n, operands, tt_isa::l1::DATA)
        })
        .expect("two tiles always fit DATA");
        let (_, math) = crate::sfpu::ops::code_for(op.kind, [op.scalar, op.scalar2], bcast)
            .expect("checked by the caller");
        crate::sfpu::kernel::roles_code(&layout, operands, &math)
            .0
            .map(|p| p.len())
    };
    // Two tiles and three: a run of two or more repeats one tile's block
    // (`sfpu::kernel::roles_looped`), so nothing grows a tile, and the bound is
    // only the arena's; a role that did grow is bounded by its slot.
    let (two, three) = (lens(2), lens(3));
    let max = tt_isa::mailbox::PROGRAM_MAX as usize;
    let by_program = (0..3)
        .map(|r| {
            let per = three[r].saturating_sub(two[r]);
            let fixed = two[r].saturating_sub(2 * per);
            assert!(fixed < max, "a run's fixed program exceeds its slot");
            (max - fixed).checked_div(per).unwrap_or(usize::MAX)
        })
        .min()
        .unwrap();
    let arena = if half {
        crate::matmul::half()
    } else {
        crate::l1::arena().len()
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
    rows: u32,
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
        u32,
        u64,
    );
    type Memo = Mutex<HashMap<Key, SfpuPrograms>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    // The programs name the layout's addresses, planned in the arena (`crate::l1::arena`).
    let key = (
        op.kind,
        op.scalar.to_bits(),
        op.scalar2.to_bits(),
        bcast,
        len,
        half,
        rows,
        crate::l1::arena().end,
    );
    let memo = MEMO.get_or_init(Default::default);
    // The highest `DATA` address the layout reaches, noted on every hit as a plan notes it on a
    // miss (`crate::l1::note_planned_top`): its last region's slots, `B`'s one where held once.
    let note_top = |l: &crate::sfpu::kernel::Layout| {
        let slots = |at: u64| at + l.tiles as u64 * TILE_SLOT;
        let b = l
            .b_at
            .map(|b| if l.b_once { b + TILE_SLOT } else { slots(b) });
        let top = [
            Some(slots(l.a_at)),
            b,
            l.c_at.map(slots),
            Some(slots(l.out_at)),
        ]
        .into_iter()
        .flatten()
        .max();
        crate::l1::note_planned_top(top.unwrap_or(0));
    };
    if let Some(p) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        note_top(&p.0);
        return Ok(p.clone());
    }
    let layout = match half {
        None => crate::sfpu::kernel::plan_layout(len, operands),
        Some(h) => crate::sfpu::kernel::plan_layout_in(len, operands, crate::matmul::half_arena())
            .map(|l| l.shifted(h as u64 * crate::matmul::half())),
    }
    .map_err(|e| TensorError::Shape(e.to_string()))?;
    let (_, math) = crate::sfpu::ops::code_for_rows(op.kind, [op.scalar, op.scalar2], bcast, rows)
        .expect("checked by the caller");
    let (roles, loops) = crate::sfpu::kernel::roles_code(&layout, operands, &math);
    note_top(&layout);
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
    sfpu_reduce_then(alloc, a, op, axis, units, overlap, None)
}

/// [`sfpu_reduce`] with `then`'s exact unary kind applied to the result before the pack
/// ([`ReduceThen`]): bit for bit the kind run on the stored result. A softmax's
/// `clamp_min(max(x))` and `clamp_min(sum(e))` are each one op so. Refused for a reduction too
/// long for one tile's programs (the chunked form has no finish to add to).
pub fn sfpu_reduce_then(
    alloc: &mut DramAlloc,
    a: &DramTensor,
    op: crate::sfpu::reduce::ReduceOp,
    axis: crate::sfpu::reduce::Axis,
    units: usize,
    overlap: Overlap,
    then: ReduceThen,
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
        then,
    )
    .is_none()
    {
        if then.is_some() {
            return Err(TensorError::Shape(
                "a reduction too long for one tile's programs takes no finishing kind".into(),
            ));
        }
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
    let group = match reduce_group(op, axis, per, valid, false, then) {
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
        let half = reduce_group(op, axis, per, valid, true, then).unwrap_or(0);
        pipelined_runs(outs, per, units, half, overlap)
    };
    let pipelined = piped.is_some();
    let all = piped.unwrap_or_else(|| runs(outs, units, group));
    let mut jobs = Vec::new();
    for (j, run) in all.into_iter().enumerate() {
        let len = run.len();
        let half = pipelined.then(|| half_of(j, units));
        let (layout, roles, loops) = match reduce_programs(op, axis, len, per, valid, half, then) {
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
                loops,
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

/// Each row's `sqrt(mean(a a) + eps)` over its columns, `[rows, 1]`, on the SFPU
/// (`crate::sfpu::norm`): bit for bit `MUL(a, a)`, the sum over columns, `MUL_SCALAR(1/cols)`,
/// `ADD_SCALAR(eps)` and `SQRT`, with no tensor between them. A reduction's kernel, one job a
/// tile row: the mover reads the row, the kernel sums its squares and takes the statistic, the
/// mover writes the column. A row longer than one job takes (`sfpu::norm::fits`) is refused.
pub fn sfpu_rms(alloc: &mut DramAlloc, a: &DramTensor, eps: f32) -> Result<Work> {
    use crate::sfpu::{norm, reduce};
    a.expect("an RMS statistic", Elem::F32)?;
    if !norm::fits(a.cols) {
        return Err(TensorError::Shape(format!(
            "a row of {} columns does not fit one RMS statistic job",
            a.cols
        )));
    }
    let [rt, per] = a.grid();
    let valid = match a.cols % 32 {
        0 => 32,
        v => v as u32,
    };
    let layout = reduce::plan_layout(1, per).map_err(|e| TensorError::Shape(e.to_string()))?;
    // A tile row's programs run over the row groups its rows occupy: a ragged last tile row (a
    // decode step's one row) has its own.
    let roles_for = |rows: usize| {
        let (squares, fin) = norm::math(per, valid, eps, rows as u32);
        Arc::new(reduce::roles(&layout, &squares, &fin))
    };
    let whole = (a.rows >= 32).then(|| roles_for(32));
    let last = match a.rows % 32 {
        0 => whole.clone().expect("a whole tile row"),
        r => roles_for(r),
    };
    let out = DramTensor::alloc_elem(alloc, a.rows, 1, Elem::F32)?;
    let (ra, ro) = (a.tensor_ref(), out.tensor_ref());
    let jobs = (0..rt)
        .map(|k| {
            // As a reduction over columns gathers and scatters a run of one output. Whole
            // tiles even for a decode row: reading only its rows (`1024 + 64 r` bytes a tile)
            // measured slower, 21.0 us a 2048-wide row against 19.2 (decode-small-ops.md S3).
            let gather = [
                [
                    record::READ_RUN,
                    (k * per) as u32,
                    per as u32,
                    layout.in_at as u32,
                    0,
                    per as u32,
                    0,
                    0,
                ],
                ra.encode()[0],
                ra.encode()[1],
            ];
            let scatter = [
                [
                    record::WRITE_RUN,
                    k as u32,
                    1,
                    layout.out_at as u32,
                    0,
                    0,
                    0,
                    0,
                ],
                ro.encode()[0],
                ro.encode()[1],
            ];
            vec![
                Step::List {
                    what: "rms gather",
                    entries: gather.to_vec(),
                },
                Step::Kernel {
                    roles: match &whole {
                        Some(w) if k + 1 < rt => w.clone(),
                        _ => last.clone(),
                    },
                    init: layout.init.clone(),
                    mop: Box::new([None; 3]),
                    loops: Default::default(),
                    half: None,
                },
                Step::List {
                    what: "rms scatter",
                    entries: scatter.to_vec(),
                },
            ]
        })
        .collect();
    Ok(Work { out, jobs })
}

/// [`sfpu_rms`] of one row of `cols` columns (whole tiles) from its packed layout (`packed`,
/// `[cols / 32, 32]`, [`pack_row`]): one job, `cols / 1024` tiles read where the row is
/// `cols / 32` entries. The same sums in the same order (`sfpu::norm::math_packed`), so bit for
/// bit [`sfpu_rms`]'s `[1, 1]`.
pub fn sfpu_rms_packed(
    alloc: &mut DramAlloc,
    packed: &DramTensor,
    cols: usize,
    eps: f32,
) -> Result<Work> {
    use crate::sfpu::{norm, reduce};
    packed.expect("a packed RMS statistic", Elem::F32)?;
    if cols % 32 != 0 || (packed.rows, packed.cols) != (cols / 32, 32) {
        return Err(TensorError::Shape(format!(
            "[{}, {}] is not a packed row of {cols} columns",
            packed.rows, packed.cols
        )));
    }
    let [pt, _] = packed.grid();
    let layout = reduce::plan_layout(1, pt).map_err(|e| TensorError::Shape(e.to_string()))?;
    let (squares, fin) = norm::math_packed(cols, eps);
    let roles = Arc::new(reduce::roles(&layout, &squares, &fin));
    let out = DramTensor::alloc_elem(alloc, 1, 1, Elem::F32)?;
    let (rp, ro) = (packed.tensor_ref(), out.tensor_ref());
    let gather = [
        [
            record::READ_RUN,
            0,
            pt as u32,
            layout.in_at as u32,
            0,
            1,
            0,
            0,
        ],
        rp.encode()[0],
        rp.encode()[1],
    ];
    let scatter = [
        [record::WRITE_RUN, 0, 1, layout.out_at as u32, 0, 0, 0, 0],
        ro.encode()[0],
        ro.encode()[1],
    ];
    let jobs = vec![vec![
        Step::List {
            what: "rms gather",
            entries: gather.to_vec(),
        },
        Step::Kernel {
            roles,
            init: layout.init.clone(),
            mop: Box::new([None; 3]),
            loops: Default::default(),
            half: None,
        },
        Step::List {
            what: "rms scatter",
            entries: scatter.to_vec(),
        },
    ]];
    Ok(Work { out, jobs })
}

/// One semaphore initialisation for every kernel of `job`, where their plans agree on each
/// semaphore's starting value (the matmul's, SFPU's and reduction's do: theirs number from zero in
/// the same order). A kernel leaves every semaphore as it found it, those it never takes
/// included, so the union is each one's own. Consecutive blocks join one ownership region only
/// with equal initialisations: one region a job, not one a kernel (~3.5 us each). Where two
/// disagree, `job` is left as it is.
pub(crate) fn share_semaphore_init(job: &mut Job) {
    let mut all: Vec<crate::runtime::SemaphoreInit> = Vec::new();
    for step in job.iter() {
        if let Step::Kernel { init, .. } = step {
            for &(sem, value, max) in init {
                match all.iter().find(|(s, _, _)| *s == sem) {
                    Some(&(_, v, m)) if (v, m) != (value, max) => return,
                    Some(_) => {}
                    None => all.push((sem, value, max)),
                }
            }
        }
    }
    all.sort_by_key(|(s, _, _)| s.index());
    for step in job.iter_mut() {
        if let Step::Kernel { init, .. } = step {
            *init = all.clone();
        }
    }
}

/// `a / sqrt(mean(a a) + eps)` for a decode row `a` (`[1, K]`) whose packed row (`[K / 32, 32]`,
/// [`pack_row`]) is `packed`: RMSNorm up to its scale, as one op. Each job takes a run of `a`'s
/// tiles and computes the statistic itself from `packed` ([`sfpu_rms_packed`]'s kernel), then
/// divides its tiles by it (`kind_sfpu::DIV`, the column broadcast, one row): bit for bit the
/// statistic op and the `DIV` after it, with no barrier between them and no unit waiting on one
/// that computed the statistic alone. `twin`, if given, receives the result's packed row.
pub fn rms_divide(
    alloc: &mut DramAlloc,
    a: &DramTensor,
    packed: &DramTensor,
    eps: f32,
    units: usize,
    twin: Option<&DramTensor>,
) -> Result<Work> {
    use crate::sfpu::kernel::{self as sk, Operands};
    use crate::sfpu::ops::{code_for_rows, kind_sfpu, Broadcast};
    use crate::sfpu::{norm, reduce};
    a.expect("an RMS division", Elem::F32)?;
    packed.expect("a packed RMS statistic", Elem::F32)?;
    let cols = a.cols;
    if a.rows != 1 || cols % 32 != 0 || (packed.rows, packed.cols) != (cols / 32, 32) {
        return Err(TensorError::Shape(format!(
            "an RMS division of [{}, {cols}] by the packed row [{}, {}]",
            a.rows, packed.rows, packed.cols
        )));
    }
    let [pt, _] = packed.grid();
    let ct = a.grid()[1];
    let div = Eltwise {
        kind: kind_sfpu::DIV,
        scalar: 0.0,
        scalar2: 0.0,
    };
    // A run's slots: the packed row, the statistic, its tiles of `a` and of the result.
    let by_slots = ((crate::l1::arena().len() / TILE_SLOT) as usize).saturating_sub(pt + 1) / 2;
    let group = sfpu_group(div, Broadcast::Col, Operands::ColBroadcast, false).min(by_slots);
    if group == 0 {
        return Err(TensorError::Shape(format!(
            "an RMS division's packed row of {pt} tiles leaves no slots"
        )));
    }
    let stat_layout = reduce::plan_layout(1, pt).map_err(|e| TensorError::Shape(e.to_string()))?;
    let stat_layout = reduce::Layout {
        in_at: tt_isa::l1::DATA.base,
        out_at: tt_isa::l1::DATA.base + pt as u64 * TILE_SLOT,
        ..stat_layout
    };
    let a_at = stat_layout.out_at + TILE_SLOT;
    let (squares, fin) = norm::math_packed(cols, eps);
    let stat_roles = Arc::new(reduce::roles(&stat_layout, &squares, &fin));
    let (_, code) = code_for_rows(
        div.kind,
        [0.0, 0.0],
        Broadcast::Col,
        eltwise_rows(div.kind, 1),
    )
    .ok_or_else(|| TensorError::Shape("no column-broadcast DIV program".into()))?;
    let out = DramTensor::alloc(alloc, 1, cols)?;
    let (rp, ra, ro) = (packed.tensor_ref(), a.tensor_ref(), out.tensor_ref());
    let mut jobs = Vec::new();
    for run in runs(ct, units, group) {
        let len = run.len();
        let planned = sk::plan_layout(len, Operands::ColBroadcast)
            .map_err(|e| TensorError::Shape(e.to_string()))?;
        let layout = sk::Layout {
            tiles: len,
            a_at,
            b_at: Some(stat_layout.out_at),
            b_once: true,
            c_at: None,
            out_at: a_at + len as u64 * TILE_SLOT,
            sems: planned.sems,
            init: planned.init,
        };
        crate::l1::note_planned_top(layout.out_at + len as u64 * TILE_SLOT);
        let (roles, loops) = sk::roles_code(&layout, Operands::ColBroadcast, &code);
        let read = |x: &TensorRef, start: usize, n: usize, at: u64, grid_ct: usize| {
            [
                [
                    record::READ_RUN,
                    start as u32,
                    n as u32,
                    at as u32,
                    0,
                    grid_ct as u32,
                    0,
                    0,
                ],
                x.encode()[0],
                x.encode()[1],
            ]
        };
        let mut gather = read(&rp, 0, pt, stat_layout.in_at, 1).to_vec();
        gather.extend(read(&ra, run.start, len, a_at, ct));
        let mut scatter = vec![
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
        if let Some(twin) = twin {
            scatter.extend(packed_twin_writes(twin, run.start, len, layout.out_at));
        }
        jobs.push(vec![
            Step::List {
                what: "rms divide gather",
                entries: gather,
            },
            Step::Kernel {
                roles: stat_roles.clone(),
                init: stat_layout.init.clone(),
                mop: Box::new([None; 3]),
                loops: Default::default(),
                half: None,
            },
            Step::List {
                what: "rms divide statistic",
                entries: Vec::new(),
            },
            Step::List {
                what: "rms divide",
                entries: Vec::new(),
            },
            Step::Kernel {
                roles: Arc::new(roles),
                init: layout.init.clone(),
                mop: Box::new([None; 3]),
                loops: Arc::new(loops),
                half: None,
            },
            Step::List {
                what: "rms divide scatter",
                entries: scatter,
            },
        ]);
        share_semaphore_init(jobs.last_mut().expect("just pushed"));
    }
    Ok(Work { out, jobs })
}

/// Rotary position embedding in the rotate-half layout, as one op: `x cos + rotate(x) sin` over
/// the `[rows, cols]` block `x`, whose columns are heads of `head` columns each, `cos` and `sin`
/// laid out as `x` is (each head's table row), `sin` carrying `rotate_half`'s sign. `rotate(x)`
/// is each head's halves swapped, read as `x`'s own tiles at the swapped place: no copy. Each job
/// is two kernels on its run of tiles, `MUL` then `MUL_ADD` (`c + a b`, two roundings): bit for
/// bit the product, the product and the sum. `head / 2` must be whole tiles.
pub fn rope_blocks(
    alloc: &mut DramAlloc,
    x: BlockOperand<'_>,
    cos: BlockOperand<'_>,
    sin: BlockOperand<'_>,
    head: usize,
    units: usize,
) -> Result<Work> {
    use crate::sfpu::kernel::{self as sk, Operands};
    use crate::sfpu::ops::{code_for_rows, kind_sfpu, Broadcast};
    let [rows, cols] = x.dims;
    if head == 0 || head % 64 != 0 || cols % head != 0 || [cos.dims, sin.dims] != [x.dims; 2] {
        return Err(TensorError::Shape(format!(
            "RoPE over [{rows}, {cols}] in heads of {head}: halves of whole tiles, tables of x's \
             shape"
        )));
    }
    for t in [x.tensor, cos.tensor, sin.tensor] {
        t.expect("RoPE", Elem::F32)?;
    }
    let what = "a RoPE operand";
    let (rx, rc, rs) = (
        x.tensor_ref(what)?,
        cos.tensor_ref(what)?,
        sin.tensor_ref(what)?,
    );
    let shape = x.shape_only();
    let [rt, ct] = shape.grid();
    let (dt, half) = (head / 32, head / 64);
    let live = eltwise_rows(kind_sfpu::MUL_ADD, rows);
    let (_, mul) = code_for_rows(crate::kind::MUL, [0.0, 0.0], Broadcast::None, live)
        .ok_or_else(|| TensorError::Shape("no MUL program".into()))?;
    let (_, mul_add) = code_for_rows(kind_sfpu::MUL_ADD, [0.0, 0.0], Broadcast::None, live)
        .ok_or_else(|| TensorError::Shape("no MUL_ADD program".into()))?;
    // A run's slots: `x`, `cos`, `rotate(x)`, `sin`, the product and the result.
    let group = (((crate::l1::arena().len() / TILE_SLOT) as usize) / 6).min(64);
    let out = DramTensor::alloc(alloc, rows, cols)?;
    let ro = out.tensor_ref();
    let mut jobs = Vec::new();
    for run in runs(rt * ct, units, group) {
        let len = run.len();
        let slots = len as u64 * TILE_SLOT;
        let base = tt_isa::l1::DATA.base;
        let [x_at, c_at, r_at, s_at, p_at, o_at] = std::array::from_fn(|k| base + k as u64 * slots);
        crate::l1::note_planned_top(o_at + slots);
        let planned = |operands| {
            sk::plan_layout(len, operands).map_err(|e| TensorError::Shape(e.to_string()))
        };
        let (pb, pt) = (planned(Operands::Binary)?, planned(Operands::Ternary)?);
        let product = sk::Layout {
            tiles: len,
            a_at: x_at,
            b_at: Some(c_at),
            b_once: false,
            c_at: None,
            out_at: p_at,
            sems: pb.sems,
            init: pb.init,
        };
        let sum = sk::Layout {
            tiles: len,
            a_at: r_at,
            b_at: Some(s_at),
            b_once: false,
            c_at: Some(p_at),
            out_at: o_at,
            sems: pt.sems,
            init: pt.init,
        };
        let (product_roles, product_loops) = sk::roles_code(&product, Operands::Binary, &mul);
        let (sum_roles, sum_loops) = sk::roles_code(&sum, Operands::Ternary, &mul_add);
        let read_run = |r: &TensorRef, at: u64| {
            [
                [
                    record::READ_RUN,
                    run.start as u32,
                    len as u32,
                    at as u32,
                    0,
                    ct as u32,
                    0,
                    0,
                ],
                r.encode()[0],
                r.encode()[1],
            ]
        };
        let mut gather = read_run(&rx, x_at).to_vec();
        gather.extend(read_run(&rc, c_at));
        gather.extend(read_run(&rs, s_at));
        // `rotate(x)`'s tile `n` is `x`'s tile across the head's halves.
        for n in 0..len {
            let t = run.start + n;
            let (i, j) = (t / ct, t % ct);
            let (h, k) = (j / dt, j % dt);
            let src = x
                .tensor
                .tile(x.at[0] / 32 + i, x.at[1] / 32 + h * dt + (k + half) % dt);
            gather.push([
                tt_isa::dm::op::READ,
                src.channel().index() as u32,
                (n % tt_isa::dram::PORTS as usize) as u32,
                src.offset() as u32,
                (r_at + n as u64 * TILE_SLOT) as u32,
                TILE_SLOT as u32,
                0,
                0,
            ]);
        }
        let scatter = [
            [
                record::WRITE_RUN,
                run.start as u32,
                len as u32,
                o_at as u32,
                0,
                0,
                0,
                0,
            ],
            ro.encode()[0],
            ro.encode()[1],
        ];
        let kernel = |roles, loops, init| Step::Kernel {
            roles: Arc::new(roles),
            init,
            mop: Box::new([None; 3]),
            loops: Arc::new(loops),
            half: None,
        };
        let mut job = vec![
            Step::List {
                what: "rope gather",
                entries: gather,
            },
            kernel(product_roles, product_loops, product.init),
            Step::List {
                what: "rope product",
                entries: Vec::new(),
            },
            Step::List {
                what: "rope sum",
                entries: Vec::new(),
            },
            kernel(sum_roles, sum_loops, sum.init),
            Step::List {
                what: "rope scatter",
                entries: scatter.to_vec(),
            },
        ];
        share_semaphore_init(&mut job);
        jobs.push(job);
    }
    Ok(Work { out, jobs })
}

/// Most tiles of a row one partial of [`argmax_partials`] reads: its programs are a tile's each.
pub const ARGMAX_RUN_MAX: usize = 32;

/// Phase 1 of a row's argmax ([`crate::sfpu::select`]): the `[1, cols]` F32 row `a` dealt in runs
/// of tiles over `units`, each run's best key and index in a tile of the `[1, 32 runs]` result.
/// Also each run's first column, which [`argmax_choose`] adds to its index.
pub fn argmax_partials(
    alloc: &mut DramAlloc,
    a: &DramTensor,
    units: usize,
) -> Result<(Work, Vec<usize>)> {
    use crate::sfpu::{reduce, select};
    a.expect("an argmax", Elem::F32)?;
    if a.rows != 1 || a.cols == 0 {
        return Err(TensorError::Shape(format!(
            "an argmax of [{}, {}]: one row",
            a.rows, a.cols
        )));
    }
    let ct = a.grid()[1];
    let all = runs(ct, units, ARGMAX_RUN_MAX);
    let out = DramTensor::alloc(alloc, 1, 32 * all.len())?;
    let (ra, ro) = (a.tensor_ref(), out.tensor_ref());
    let mut jobs = Vec::with_capacity(all.len());
    let mut starts = Vec::with_capacity(all.len());
    let body = select::accumulate(a.cols - 1);
    let fold = select::fold();
    for (j, run) in all.into_iter().enumerate() {
        let len = run.len();
        let layout = match reduce::plan_layout(1, len) {
            Ok(l) => l,
            Err(e) => {
                alloc.free(&out.placement);
                return Err(TensorError::Shape(e.to_string()));
            }
        };
        let (roles, loops) = select::roles(
            &layout,
            len,
            &select::prologue(32 * run.start),
            &body,
            &fold,
        );
        let (roles, loops) = (Arc::new(roles), Arc::new(loops));
        starts.push(32 * run.start);
        let gather = [
            [
                record::READ_RUN,
                run.start as u32,
                len as u32,
                layout.in_at as u32,
                0,
                ct as u32,
                0,
                0,
            ],
            ra.encode()[0],
            ra.encode()[1],
        ];
        let scatter = [
            [
                record::WRITE_RUN,
                j as u32,
                1,
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
                what: "argmax gather",
                entries: gather.to_vec(),
            },
            Step::Kernel {
                roles,
                init: layout.init.clone(),
                mop: Box::new([None; 3]),
                loops,
                half: None,
            },
            Step::List {
                what: "argmax scatter",
                entries: scatter.to_vec(),
            },
        ]);
    }
    Ok((Work { out, jobs }, starts))
}

/// Phase 2 of a row's argmax: one job over [`argmax_partials`]' tiles, in order, each index moved
/// by its run's first column (`starts`); the chosen index as an I32 `[1, 1]`.
pub fn argmax_choose(
    alloc: &mut DramAlloc,
    partials: &DramTensor,
    starts: &[usize],
) -> Result<Work> {
    use crate::sfpu::{reduce, select};
    let pt = partials.grid()[1];
    if pt != starts.len() || partials.rows != 1 {
        return Err(TensorError::Shape(format!(
            "an argmax's {pt} partials for {} runs",
            starts.len()
        )));
    }
    let layout = reduce::plan_layout(1, pt).map_err(|e| TensorError::Shape(e.to_string()))?;
    let (roles, loops) = select::roles(
        &layout,
        pt,
        &select::choice_prologue(),
        &select::combine(),
        &select::index_out(),
    );
    let (roles, loops) = (Arc::new(roles), Arc::new(loops));
    let out = DramTensor::alloc_elem(alloc, 1, 1, Elem::I32)?;
    let (rp, ro) = (partials.tensor_ref(), out.tensor_ref());
    let gather = [
        [
            record::READ_RUN,
            0,
            pt as u32,
            layout.in_at as u32,
            0,
            pt as u32,
            0,
            0,
        ],
        rp.encode()[0],
        rp.encode()[1],
    ];
    let scatter = [
        [record::WRITE_RUN, 0, 1, layout.out_at as u32, 0, 0, 0, 0],
        ro.encode()[0],
        ro.encode()[1],
    ];
    Ok(Work {
        out,
        jobs: vec![vec![
            Step::List {
                what: "argmax choice gather",
                entries: gather.to_vec(),
            },
            Step::Kernel {
                roles,
                init: layout.init.clone(),
                mop: Box::new([None; 3]),
                loops,
                half: None,
            },
            Step::List {
                what: "argmax choice scatter",
                entries: scatter.to_vec(),
            },
        ]],
    })
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
    a.expect("a scan", op.elem())?;
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
    let out = DramTensor::alloc_elem(alloc, a.rows, a.cols, op.elem())?;
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
            let (roles, loops) = continuation_roles(
                &c,
                op,
                axis,
                first > 0,
                count,
                if last { valid } else { 32 },
                last,
            );
            steps.push(Step::Kernel {
                roles: Arc::new(roles),
                init: c.layout.init.clone(),
                mop: Box::new([None; 3]),
                loops: Arc::new(loops),
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

type ReducePrograms = (
    crate::sfpu::reduce::Layout,
    Arc<[Vec<Instruction>; 3]>,
    Arc<[Vec<crate::code::Loop>; 3]>,
);

/// Most output tiles one reduce run may take, from the slots the data arena
/// (half of it, if `half`) holds and the role programs' length per output
/// tile; `None` if not even one fits.
/// An exact unary kind and its scalar's bits, applied to a reduction's result before its pack
/// ([`sfpu_reduce_then`]).
pub type ReduceThen = Option<(u32, u32)>;

/// A reduction's finishing program: the fold, then `then`'s kind in place on the result
/// (`sfpu::ops::exact_in_place`). `None` if `then` names no exact unary kind.
fn finishing(
    op: crate::sfpu::reduce::ReduceOp,
    axis: crate::sfpu::reduce::Axis,
    then: ReduceThen,
) -> Option<Vec<tt_isa::isa::Instruction>> {
    let mut fin = crate::sfpu::reduce::finish(op, axis);
    if let Some((kind, s)) = then {
        fin.extend(crate::sfpu::ops::exact_in_place(
            kind,
            [f32::from_bits(s), 0.0],
            crate::sfpu::kernel::OUT_ROW,
        )?);
    }
    Some(fin)
}

/// Whether a reduction of a `[rows, cols]` tensor over `axis` with `then` runs as one tile's
/// programs a run ([`sfpu_reduce_then`] takes it), not chunked.
pub fn reduce_then_fits(
    op: crate::sfpu::reduce::ReduceOp,
    axis: crate::sfpu::reduce::Axis,
    [rows, cols]: [usize; 2],
    then: ReduceThen,
) -> bool {
    use crate::sfpu::reduce::Axis;
    let (per, valid) = match axis {
        Axis::Cols => (cols.div_ceil(32), cols % 32),
        Axis::Rows => (rows.div_ceil(32), rows % 32),
    };
    let valid = if valid == 0 { 32 } else { valid as u32 };
    reduce_group(op, axis, per, valid, false, then).is_some()
}

fn reduce_group(
    op: crate::sfpu::reduce::ReduceOp,
    axis: crate::sfpu::reduce::Axis,
    per: usize,
    valid: u32,
    half: bool,
    then: ReduceThen,
) -> Option<usize> {
    use crate::sfpu::reduce::{Axis, ReduceOp};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    // By the arena's end too (`crate::l1::arena`): a group is sized from it.
    type Key = (ReduceOp, Axis, usize, u32, bool, ReduceThen, u64);
    type Memo = Mutex<HashMap<Key, Option<usize>>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let key = (op, axis, per, valid, half, then, crate::l1::arena().end);
    let memo = MEMO.get_or_init(Default::default);
    if let Some(&g) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return g;
    }
    let g = measure_reduce_group(op, axis, per, valid, half, then);
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
    then: ReduceThen,
) -> Option<usize> {
    use crate::sfpu::reduce::{math_programs, plan_layout, roles_code};
    const GROUP: usize = 64;
    let arena = if half {
        crate::matmul::half()
    } else {
        crate::l1::arena().len()
    };
    let slots = (arena / TILE_SLOT) as usize;
    let by_slots = slots / (per + 1);
    if by_slots == 0 {
        return None;
    }
    let (inputs, _) = math_programs(op, axis, per, valid);
    let fin = finishing(op, axis, then)?;
    let lens = |n: usize| -> Option<[usize; 3]> {
        let layout = plan_layout(n, per).ok()?;
        // As stored: a repeated block once, with its loop table
        // (`sfpu::reduce::roles_code`).
        let (code, loops) = roles_code(&layout, &inputs, &fin);
        Some(std::array::from_fn(|t| code[t].len() + loops[t].len() + 1))
    };
    let max = tt_isa::mailbox::PROGRAM_MAX as usize;
    let one = lens(1)?;
    if one.iter().any(|&l| l > max) {
        return None;
    }
    if by_slots == 1 {
        return Some(1);
    }
    // Two outputs and three: looped, a further output costs nothing.
    let two = lens(2)?;
    if two.iter().any(|&l| l > max) {
        return Some(1);
    }
    if by_slots == 2 {
        return Some(2);
    }
    let three = lens(3)?;
    let by_program = (0..3)
        .map(|r| {
            let per_out = three[r].saturating_sub(two[r]);
            let fixed = two[r].saturating_sub(2 * per_out);
            (max - fixed).checked_div(per_out).unwrap_or(usize::MAX)
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
    then: ReduceThen,
) -> Result<ReducePrograms> {
    use crate::sfpu::reduce::{
        math_programs, plan_layout, plan_layout_in, roles_code, Axis, ReduceOp,
    };
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = (
        ReduceOp,
        Axis,
        usize,
        usize,
        u32,
        Option<u8>,
        ReduceThen,
        u64,
    );
    // Each layout with what its plan reached, where it was moved to, noted again on every hit
    // (`crate::l1::note_planned`); by the arena's end too, where it was planned.
    type Memo = Mutex<HashMap<Key, (ReducePrograms, crate::l1::Planned)>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let key = (
        op,
        axis,
        len,
        per,
        valid,
        half,
        then,
        crate::l1::arena().end,
    );
    let memo = MEMO.get_or_init(Default::default);
    if let Some((p, planned)) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        crate::l1::note_planned(*planned);
        return Ok(p.clone());
    }
    let before = crate::l1::take_planned();
    let layout = match half {
        None => plan_layout(len, per),
        Some(h) => crate::l1::planned_shifted(h as u64 * crate::matmul::half(), || {
            plan_layout_in(len, per, crate::matmul::half_arena())
        })
        .map(|l| l.shifted(h as u64 * crate::matmul::half())),
    };
    let planned = crate::l1::take_planned();
    crate::l1::note_planned(before);
    crate::l1::note_planned(planned);
    let layout = layout.map_err(|e| TensorError::Shape(e.to_string()))?;
    let (inputs, _) = math_programs(op, axis, per, valid);
    let fin = finishing(op, axis, then)
        .ok_or_else(|| TensorError::Shape("a reduction's finishing kind is no exact one".into()))?;
    let (roles, loops) = roles_code(&layout, &inputs, &fin);
    let p = (layout.clone(), Arc::new(roles), Arc::new(loops));
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key, (p.clone(), planned));
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
    if reduce_group(ReduceOp::Sum, Axis::Rows, rt, valid, false, None).is_some() {
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
    // By the arena's end (`crate::l1::arena`).
    static MEMO: std::sync::Mutex<std::collections::BTreeMap<u64, Option<usize>>> =
        std::sync::Mutex::new(std::collections::BTreeMap::new());
    let arena = crate::l1::arena();
    if let Some(&g) = MEMO
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get(&arena.end)
    {
        return g;
    }
    let g = (|| {
        let slots = (arena.len() / TILE_SLOT) as usize;
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
    })();
    MEMO.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(arena.end, g);
    g
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

/// For `op(A) @ op(B)`, per operand: is its `K` whole tiles while it has padding
/// elsewhere? Then its padding is all off `K` -- `op(A)`'s past `M`, `op(B)`'s past
/// `N` -- where it reaches only the output's padding rows (columns), never a real
/// datum, so the operand needs no refill whatever it holds there; the output's
/// padding is then undefined unless the operand's was zero. `false` for an
/// operand with no padding at all, or with padding along `K`, which the matmul
/// needs zero ([`MatmulPadding`]).
pub fn matmul_padding_off_k(
    a: &DramTensor,
    a_transposed: bool,
    b: &DramTensor,
    b_transposed: bool,
) -> [bool; 2] {
    // `(K, the other)` of each operand as stored.
    let (ak, am) = if a_transposed {
        (a.rows, a.cols)
    } else {
        (a.cols, a.rows)
    };
    let (bk, bn) = if b_transposed {
        (b.cols, b.rows)
    } else {
        (b.rows, b.cols)
    };
    [ak % 32 == 0 && am % 32 != 0, bk % 32 == 0 && bn % 32 != 0]
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
            // Raw words, so zero is the all-zero word alone (the
            // float kinds' `-0.0` is `i32::MIN` here) and `|0| = 0`.
            INT_MASK_FILL | BOOL_MASK_FILL => b_zero || s.to_bits() == 0,
            INT_MASK_WHERE | BOOL_MASK_WHERE => b_zero || c_zero,
            INT_ABS => true,
            // 10.2d-f: `f(±0) = ±0`.
            SQRT | EXPM1 | TANH | ERF | GELU | SINH | ASINH | ATANH | SIN | TAN | ATAN | ASIN => {
                true
            }
            // The Approx twins of `tanh` and `gelu`.
            crate::sfpu::approx::kind::TANH | crate::sfpu::approx::kind::GELU => true,
            // `0^s = 0` for `s > 0`.
            POW_S => s > 0.0,
            // `g (1/2)` and `g 0 1`: zero with the gradient's padding.
            GELU_BACKWARD | SIGMOID_BACKWARD | LOG_SIGMOID_BACKWARD => b_zero,
            // `0 * 0.5`, then times `y`.
            SILU => true,
            SILU_MUL => b_zero,
            // `atan2(+0, +0) = +0`.
            ATAN2 => b_zero,
            // `((0 % s) + s) % s` is a zero (the sign of `s`) for a
            // finite nonzero `s`; a zero, infinite or NaN `s` gives NaN.
            REM_S => s.is_finite() && !is_zero(s),
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
                        kind_sfpu::INT_MASK_FILL | kind_sfpu::BOOL_MASK_FILL
                    ) && zero(0)
                        && self.scalar.to_bits() == 0)
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
    let group = arena_group(128);
    let stage = staging("copy slots", group)?;
    let out = DramTensor::alloc_elem(alloc, t.rows, t.cols, t.elem)?;
    let [rt, ct] = t.grid();
    let (rs, ro) = (t.tensor_ref(), out.tensor_ref());
    let jobs = runs(rt * ct, units, group)
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
    let group = arena_group(128);
    let stage = staging("copy slots", group)?;
    let [rt, ct] = src.grid();
    let (rs, ro) = (src.tensor_ref(), dst.tensor_ref());
    let jobs = runs(rt * ct, units, group)
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
    let group = arena_group(128);
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
    let stage = staging("copy slots", group)?;
    let per = units.div_ceil(moves.len().max(1)).max(1);
    let mut jobs = Vec::new();
    for (rs, ro, [r, c], transposed) in refs {
        let ct = c.div_ceil(32);
        let flags = if transposed { 8 } else { 0 };
        for run in runs(r.div_ceil(32) * ct, per, group) {
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

/// The source tiles a [`repack_many`] batch reads before it copies: their reads are in flight
/// together.
const REPACK_SLOTS: usize = 32;

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
    let input = req.scratch(
        "repack source tiles",
        REPACK_SLOTS as u64 * TILE_SLOT,
        64,
        0..1,
    );
    let output = req.scratch("repack destination tile", TILE_SLOT, 64, 0..1);
    let layout = req
        .plan(crate::l1::arena())
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
            let write = vec![
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
            ];
            // A batch reads its source tiles into slots of their own, all of them before any copy
            // (B waits for the preceding reads before a `COPY_WORDS`, so the reads are in flight
            // together), then copies their words in the order one slot would have.
            let mut batches = Vec::new();
            let (mut reads, mut copies) = (Vec::new(), Vec::<[u32; 8]>::new());
            let flush = |reads: &mut Vec<[u32; 8]>,
                         copies: &mut Vec<[u32; 8]>,
                         batches: &mut Vec<TransferBatch>| {
                if reads.is_empty() {
                    return;
                }
                let mut read = std::mem::take(reads);
                read.append(copies);
                batches.push(TransferBatch {
                    read,
                    write: write.clone(),
                });
            };
            for ((source, tile), mut words) in tiles {
                let t = inputs[source];
                words.sort_unstable();
                let src = t.tile(tile / t.grid()[1], tile % t.grid()[1]);
                // Consecutive words commonly cover a face-row. Merge
                // them without assuming a tile-coherent logical view.
                let mut merged = Vec::new();
                let mut n = 0;
                while n < words.len() {
                    let (sr, dst) = words[n];
                    let mut count = 1;
                    while n + count < words.len() && words[n + count] == (sr + count, dst + count) {
                        count += 1;
                    }
                    merged.push((sr, dst, count));
                    n += count;
                }
                // A tile whose copies do not fit one batch is read again for each part.
                for part in merged.chunks(TRANSFER_SIDE_MAX - 2) {
                    if reads.len() == REPACK_SLOTS
                        || 1 + reads.len() + 1 + copies.len() + part.len() > TRANSFER_SIDE_MAX
                    {
                        flush(&mut reads, &mut copies, &mut batches);
                    }
                    if reads.is_empty() && !batches.is_empty() {
                        reads.push(read_prior);
                    }
                    let slot = input + (reads.len() as u64) * TILE_SLOT;
                    // Only the 64-byte span of the slot that holds the part's words (a decode row's
                    // two face rows are 1,152 bytes of 4,160): a unit's reads are bound by its own
                    // stream (~7 GB/s), not by their count. Slot offsets on both sides, so the read
                    // keeps the C64 congruence.
                    let (lo, hi) = part
                        .iter()
                        .fold((u64::MAX, 0), |(lo, hi), &(sr, _, count)| {
                            (
                                lo.min(TILE_DATA + 4 * sr as u64),
                                hi.max(TILE_DATA + 4 * (sr + count) as u64),
                            )
                        });
                    let (lo, hi) = (lo / 64 * 64, hi.next_multiple_of(64).min(TILE_SLOT));
                    reads.push([
                        op::READ,
                        src.channel().index() as u32,
                        0,
                        (src.offset() + lo) as u32,
                        (slot + lo) as u32,
                        (hi - lo) as u32,
                        0,
                        0,
                    ]);
                    copies.extend(part.iter().map(|&(sr, dst, count)| {
                        [
                            op::COPY_WORDS,
                            (slot + TILE_DATA + sr as u64 * 4) as u32,
                            (output + TILE_DATA + dst as u64 * 4) as u32,
                            count as u32,
                            4,
                            4,
                            0,
                            0,
                        ]
                    }));
                }
            }
            flush(&mut reads, &mut copies, &mut batches);
            jobs.push(vec![Step::Transfer {
                what: "native repack",
                depth: 1,
                batches,
            }]);
        }
    }
    Ok(Work { out, jobs })
}

/// [`gather_rows`]'s source for a row it leaves as its slot held it (nothing read).
pub const GATHER_HOLE: usize = usize::MAX;

/// The fewest rows a unit takes when [`gather_rows`] deals rows, not tiles: each unit in an op
/// costs it a barrier arrival.
pub const GATHER_DEAL_ROWS: usize = 4;

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
/// slot held them: the output's padding is undefined there. So is a row whose source is
/// [`GATHER_HOLE`]: nothing is read for it. With fewer output tiles than `units`, rows are dealt
/// instead (at least [`GATHER_DEAL_ROWS`] a unit), and only the rows gathered are written: the
/// others keep what GDDR held, as undefined.
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
        .find(|(_, &(k, r))| k != GATHER_HOLE && (k >= sources.len() || r >= sources[k].rows))
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
    if ort * oct < units {
        // Fewer output tiles than units (a stacked product's `[heads, n]`): a tile's 32 rows on
        // one unit would cost it 64 entries. Dealt by rows instead, each a read of its two face
        // rows (1,024 bytes apart in any tile, so one 1,088-byte read lands both where the
        // output row's go) into a slot of its own, and two 64-byte writes into place.
        let items: Vec<(usize, usize, usize, usize)> = (0..oct)
            .flat_map(|oj| {
                rows.iter()
                    .enumerate()
                    .filter(|(_, &(k, _))| k != GATHER_HOLE)
                    .map(move |(i, &(k, sr))| (i, oj, k, sr))
            })
            .collect();
        let per = items.len().div_ceil(units).clamp(GATHER_DEAL_ROWS, GROUP);
        let jobs = items
            .chunks(per)
            .map(|chunk| {
                let (mut read, mut write) = (Vec::new(), Vec::new());
                for (q, &(i, oj, k, sr)) in chunk.iter().enumerate() {
                    let (from, slot) = (sources[k].tile(sr / 32, oj), stage + q as u64 * TILE_SLOT);
                    let r = i % 32;
                    read.push([
                        op::READ,
                        from.channel().index() as u32,
                        (q % tt_isa::dram::PORTS as usize) as u32,
                        (from.offset() + face_row(sr % 32, 0)) as u32,
                        (slot + face_row(r, 0)) as u32,
                        1024 + 64,
                        0,
                        0,
                    ]);
                    let to = out.tile(i / 32, oj);
                    for h in 0..2 {
                        write.push([
                            op::WRITE,
                            to.channel().index() as u32,
                            0,
                            (to.offset() + face_row(r, h)) as u32,
                            (slot + face_row(r, h)) as u32,
                            64,
                            0,
                            0,
                        ]);
                    }
                }
                vec![Step::Transfer {
                    what: "row gather list",
                    depth: 1,
                    batches: vec![TransferBatch { read, write }],
                }]
            })
            .collect();
        return Ok(Work { out, jobs });
    }
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
                        if src == GATHER_HOLE {
                            continue;
                        }
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
    let group = arena_group(192);
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
    let stage = staging("fill-pad slots", group)?;
    let r = t.tensor_ref();
    // B reads each edge tile and fills its padding; NC writes it back. Batches
    // of half the staging alternate, so NC writes one while B fills the next.
    let per_batch = group / 2;
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
    Ok(runs(edges, units, group)
        .into_iter()
        .map(|run| {
            let batches = (run.start..run.end)
                .step_by(per_batch)
                .enumerate()
                .map(|(b, first)| {
                    let count = per_batch.min(run.end - first);
                    let at = stage + (b % 2 * per_batch) as u64 * TILE_SLOT;
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
        block_repeats, blocks, broadcast_of, metadata, reference, runs, sfpu_eltwise, DramAlloc,
        DramTensor, Elem, Eltwise, Job, MatmulPadding, OpPadding, Overlap, Pad, PadNeed, Step,
        SumRows, TransferBatch, TILE_SLOT, TRANSFER_SIDE_MAX,
    };
    use crate::kind;

    /// A placement allocated after a snapshot was free then, though its slots were in use at the
    /// snapshot: what held them was freed and reused since. Before that rule, a slot in use at a
    /// trace's end was the trace's for good, and an eager op's buffer landing on it could not be
    /// freed while the trace lived.
    #[test]
    fn a_placement_allocated_after_a_snapshot_was_free_then() {
        let mut alloc = DramAlloc::new(&tt_isa::dram::Dram::FULL);
        let a = alloc.alloc(64).unwrap();
        let snapshot = alloc.snapshot();
        assert!(!snapshot.was_free(&a), "in use then");
        alloc.free(&a);
        let b = alloc.alloc(64).unwrap();
        assert_eq!(b.base, a.base, "on the same slots");
        assert!(snapshot.was_free(&b));
        assert!(!snapshot.was_free(&a), "what was in use then still was");
    }
    use tt_isa::dm::TILE_DATA;
    use tt_isa::dm::{fill, op, record};
    use tt_isa::dram::Dram;

    /// The transfer batches of `metadata` for one `rows x cols` of `bits`, and how many words
    /// (single-word copies) they move.
    fn metadata_batches(bits: &[u32], dims: [usize; 2]) -> (Vec<TransferBatch>, usize) {
        let mut alloc = DramAlloc::new(&Dram::FULL);
        let work = metadata(&mut alloc, bits, dims, Elem::F32).unwrap();
        let batches: Vec<_> = work
            .jobs
            .into_iter()
            .flatten()
            .flat_map(|step| match step {
                Step::Transfer { batches, .. } => batches,
                _ => unreachable!("metadata is transfers"),
            })
            .collect();
        let words = batches
            .iter()
            .flat_map(|b| &b.read)
            .filter(|e| e[0] == op::COPY_WORDS && e[3] == 1)
            .count();
        (batches, words)
    }

    /// A tile of constants that all differ costs a fill and a copy each: batching by word count
    /// alone overran a packet ("a transfer batch of resident metadata is too long"), which
    /// panicked the server thread. A tile of one value still batches 200 words at a time.
    #[test]
    fn metadata_batches_fit_a_packet_however_many_words_differ() {
        let different: Vec<u32> = (0..1024).map(|i| 0x3f80_0000 + i * 7919).collect();
        let (batches, words) = metadata_batches(&different, [32, 32]);
        assert_eq!(words, 1024);
        assert!(batches.len() > 6, "all-different words need more batches");
        for b in &batches {
            assert!(b.read.len() <= TRANSFER_SIDE_MAX && b.write.len() <= TRANSFER_SIDE_MAX);
        }
        let (batches, words) = metadata_batches(&[0x3f80_0000; 1024], [32, 32]);
        assert_eq!((words, batches.len()), (1024, 6));
    }

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

    /// A [`Broadcast::Block`] op reads, beside each of `A`'s tiles `(i, j)`, `B`'s tile
    /// `(i % B's tile rows, j)`: what a gather expanding `B` to `A`'s shape would have put in the
    /// slot, over runs that start anywhere in a repeat and span several (one unit, several units,
    /// both overlaps), with nothing else of `B` read.
    ///
    /// [`Broadcast::Block`]: crate::sfpu::ops::Broadcast::Block
    #[test]
    fn a_block_broadcast_reads_b_s_tile_rows_again_down_a() {
        use crate::sfpu::ops::Broadcast;
        let op = Eltwise {
            kind: crate::kind::ADD,
            scalar: 0.0,
            scalar2: 0.0,
        };
        let mut checked = 0;
        for (b_rows, repeats, cols) in [(32, 2, 32), (64, 4, 96), (64, 3, 40), (96, 2, 300)] {
            let mut alloc = DramAlloc::new(&Dram::FULL);
            let a = DramTensor::alloc_elem(&mut alloc, b_rows * repeats, cols, Elem::F32).unwrap();
            let b = DramTensor::alloc_elem(&mut alloc, b_rows, cols, Elem::F32).unwrap();
            assert_eq!(broadcast_of(op, &a, Some(&b)).unwrap().1, Broadcast::Block);
            assert_eq!(block_repeats(&a, &b), Some(repeats));
            let ([_, ct], [brt, _]) = (a.grid(), b.grid());
            // What `B`'s tile `t` is read from, by a one-tile plain run.
            let b_ref = b.tensor_ref().encode();
            let source = |t: usize| {
                let mut out = Vec::new();
                expanded(
                    &[
                        [record::READ_RUN, t as u32, 1, 0, 0, ct as u32, 0, 0],
                        b_ref[0],
                        b_ref[1],
                    ],
                    &mut out,
                );
                let e = out[0].clone().unwrap();
                (e[1], e[3])
            };
            for (units, overlap) in [
                (1, Overlap::Off),
                (3, Overlap::Fresh),
                (5, Overlap::Captured),
            ] {
                let work = sfpu_eltwise(&mut alloc, op, &a, Some(&b), None, units, overlap, None)
                    .unwrap()
                    .unwrap();
                for job in &work.jobs {
                    let Step::List { entries, .. } = &job[0] else {
                        panic!("a job begins with its gather")
                    };
                    // `A`'s run first, then `B`'s records.
                    let (first, count) = (entries[0][1] as usize, entries[0][2] as usize);
                    let b_at = entries[3][3];
                    let mut reads = Vec::new();
                    expanded(&entries[3..], &mut reads);
                    let reads: Vec<[u32; 8]> = reads.into_iter().map(Result::unwrap).collect();
                    assert_eq!(reads.len(), count, "a tile of B beside each tile of A");
                    for (n, e) in reads.iter().enumerate() {
                        let t = first + n;
                        let (i, j) = (t / ct, t % ct);
                        assert_eq!(e[4], b_at + (n as u64 * TILE_SLOT) as u32, "slot {n}");
                        assert_eq!(
                            (e[1], e[3]),
                            source((i % brt) * ct + j),
                            "A's tile ({i}, {j}) reads B's ({}, {j})",
                            i % brt
                        );
                        checked += 1;
                    }
                }
                alloc.free(&work.out.placement);
            }
        }
        assert!(checked > 100, "{checked}");
        // Not a block: a `B` whose rows are not whole tiles, or do not divide `A`'s.
        let mut alloc = DramAlloc::new(&Dram::FULL);
        let a = DramTensor::alloc_elem(&mut alloc, 128, 64, Elem::F32).unwrap();
        for rows in [16, 48, 96] {
            let b = DramTensor::alloc_elem(&mut alloc, rows, 64, Elem::F32).unwrap();
            assert!(broadcast_of(op, &a, Some(&b)).is_err(), "[{rows}, 64]");
        }
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

    /// A thin product's columns are dealt so that the busiest unit holds the
    /// fewest a balanced deal can give it, `ceil(nt / units)`, in jobs no wider
    /// than the plan's block, covering every column once and in order -- and
    /// never more than [`blocks`]' deal gave it. Not for `K` blocked, a
    /// pipelined staging, more than one tile row, or no more columns than units.
    #[test]
    fn thin_runs_give_no_unit_more_than_its_share() {
        use super::thin_runs;
        use crate::matmul::Staging;
        let load = |cols: &mut dyn Iterator<Item = usize>, units: usize| {
            let mut load = vec![0; units];
            for (j, c) in cols.enumerate() {
                load[j % units] += c;
            }
            load.into_iter().max().unwrap()
        };
        for units in [1, 7, 64, 120] {
            for nc in [1, 2, 3] {
                for nt in [
                    units + 1,
                    units * nc,
                    units * nc + 1,
                    2 * units * nc + units / 2 + 1,
                    1536,
                ] {
                    let runs = thin_runs([1, 4, nt], [4, nc], Staging::Slots, units);
                    if nt <= units {
                        assert_eq!(runs, None);
                        continue;
                    }
                    let runs = runs.unwrap();
                    let mut next = 0;
                    for r in &runs {
                        assert_eq!(r.start, next);
                        assert!(!r.is_empty() && r.len() <= nc, "{r:?} wider than {nc}");
                        next = r.end;
                    }
                    assert_eq!(next, nt);
                    let thin = load(&mut runs.iter().map(|r| r.len()), units);
                    assert_eq!(thin, nt.div_ceil(units), "nt {nt} units {units} nc {nc}");
                    let [_, bn] = blocks([1, nt], [1, nc], units);
                    let whole = load(&mut (0..nt).step_by(bn).map(|j| bn.min(nt - j)), units);
                    assert!(whole >= thin);
                }
            }
        }
        assert_eq!(thin_runs([1, 64, 256], [2, 2], Staging::Slots, 120), None);
        assert_eq!(
            thin_runs([1, 64, 256], [64, 2], Staging::SlotsHalf, 120),
            None
        );
        assert_eq!(thin_runs([2, 64, 256], [64, 2], Staging::Slots, 120), None);
        assert_eq!(thin_runs([1, 64, 120], [64, 2], Staging::Slots, 120), None);
    }

    /// What a decode step's `m = 1` products ask of GDDR, from the jobs
    /// `matmul_dram` builds (decode-matmul-bandwidth P0): jobs, the rounds the
    /// busiest unit runs, and the bytes its gathers read of `A` and of `B`.
    /// `cargo test -p tt-kernels decode_products -- --nocapture` prints the table.
    #[test]
    fn resident_a_is_read_once_a_unit() {
        use crate::matmul::{Fidelity, SrcRoute};
        let units = 120;
        let count = |k: usize, n: usize, resident_a: bool| {
            let mut alloc = DramAlloc::new(&Dram::FULL);
            let a = DramTensor::alloc(&mut alloc, 1, k).unwrap();
            let b = DramTensor::alloc(&mut alloc, k, n).unwrap();
            let reads = super::Reads {
                thin: true,
                resident_a,
                unpack_b: false,
                packed_a: None,
                stream_k: false,
                stream_prefetch: false,
                prefetch_units: [usize::MAX; 2],
                handoff: None,
                second_job: false,
                balanced_deal: false,
                block_stream: super::BlockStream::Off,
            };
            let work = super::matmul_dram_limited(
                &mut alloc,
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                units,
                false,
                false,
                None,
                reads,
            )
            .unwrap();
            let mut per_unit = vec![crate::profile_mode::OpBytes::default(); units];
            for (j, job) in work.jobs.iter().enumerate() {
                for step in job {
                    if let Step::List { entries, .. } = step {
                        per_unit[j % units].count(entries);
                    }
                }
            }
            (work.jobs, per_unit)
        };
        // `gate|up` (three rounds on some units), `q` (one), `down` (K-blocked: never resident).
        for [k, n] in [[2048, 16384], [2048, 2048], [8192, 2048]] {
            let (jobs, off) = count(k, n, false);
            let (resident_jobs, on) = count(k, n, true);
            assert_eq!(jobs.len(), resident_jobs.len());
            // The thin `A` of `K` whole: one `1024 + 64`-byte read a tile.
            let a_once = (k as u64 / 32) * (1024 + 64);
            for u in 0..units {
                let (off, on) = (off[u], on[u]);
                assert_eq!(on.b, off.b, "[{k}, {n}] unit {u}: B unchanged");
                if k == 8192 || off.a == 0 {
                    assert_eq!(on, off, "[{k}, {n}] unit {u}");
                } else {
                    assert_eq!(on.a, a_once, "[{k}, {n}] unit {u}: A once");
                    assert_eq!(off.reads - on.reads, (off.a - a_once) / 1088, "unit {u}");
                }
            }
        }
    }

    /// A packed `A` ([`super::pack_row`]) is read as its `K / 1024` packed tiles on a unit's first
    /// block, and not at all on its later ones; `B` is read as before. Without it, each block
    /// reads `K / 32` thin `A` tiles (once a unit with `resident_a`).
    #[test]
    fn a_packed_a_is_read_in_k_over_1024_tiles() {
        use crate::matmul::{Fidelity, SrcRoute};
        let units = 64;
        for (k, n, limit) in [
            (2048, 6144, None),
            (2048, 16384, None),
            (8192, 2048, Some(64)),
            (8192, 2048, None),
        ] {
            let reads_per_unit = |packed: bool| {
                let mut alloc = DramAlloc::new(&Dram::FULL);
                let a = DramTensor::alloc(&mut alloc, 1, k).unwrap();
                let b = DramTensor::alloc(&mut alloc, k, n).unwrap();
                let p = DramTensor::alloc(&mut alloc, k / 32, 32).unwrap();
                let work = super::matmul_dram_limited(
                    &mut alloc,
                    &a,
                    false,
                    &b,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    units,
                    false,
                    false,
                    limit.and_then(std::num::NonZeroUsize::new),
                    super::Reads {
                        thin: true,
                        resident_a: true,
                        unpack_b: false,
                        packed_a: packed.then(|| p.tensor_ref()),
                        stream_k: false,
                        stream_prefetch: false,
                        prefetch_units: [usize::MAX; 2],
                        handoff: None,
                        second_job: false,
                        balanced_deal: false,
                        block_stream: super::BlockStream::Off,
                    },
                )
                .unwrap();
                let mut per_unit = vec![crate::profile_mode::OpBytes::default(); units];
                for (j, job) in work.jobs.iter().enumerate() {
                    for step in job {
                        if let Step::List { entries, .. } = step {
                            per_unit[j % units].count(entries);
                        }
                    }
                }
                per_unit
            };
            let (off, on) = (reads_per_unit(false), reads_per_unit(true));
            if k == 8192 && limit.is_none() {
                // Packed, `down` fits two K blocks of 128, not four of 64: its partial written
                // once and read back once, not three times.
                let kc = super::packed_k_block(256, None, SrcRoute::Tf32FromFp32, Fidelity::HiFi4);
                assert_eq!(kc.unwrap(), 128);
            }
            let kt = k as u64 / 32;
            for u in 0..units {
                let (off, on) = (off[u], on[u]);
                assert_eq!(on.b, off.b, "[{k}, {n}] unit {u}: B unchanged");
                // Thin `A`, `K` tiles (once a unit with `K` whole), against the packed tiles; and
                // each K block after the first reads its partial back (`down` unlimited: four
                // thin blocks of 64, two packed of 128).
                let (thin_blocks, packed_blocks) = match (k, limit) {
                    (8192, Some(_)) => (4, 4),
                    (8192, None) => (4, 2),
                    _ => (1, 1),
                };
                let packed_tiles = if k == 8192 { kt / 32 } else { kt.div_ceil(32) };
                assert_eq!(
                    off.reads - on.reads,
                    kt - packed_tiles + (thin_blocks - packed_blocks),
                    "[{k}, {n}] unit {u}"
                );
            }
        }
    }

    /// A streamed one-row product: a job a column, each its `K` in an even number of chunks
    /// (at least [`super::MIN_STREAM_CHUNKS`]) in alternating halves, every chunk's gather reading
    /// only `B` (its `K` range: each `B` tile once) but a unit's first, which reads the packed
    /// `A`; and only each column's last chunk writes.
    #[test]
    fn a_units_first_job_reads_its_b_ahead_where_the_ops_before_plan_below_it() {
        use crate::matmul::{Fidelity, SrcRoute};
        use tt_isa::dm::{op, TILE_SLOT, WAIT_PREFETCH};
        let units = 8;
        let (k, n) = (2048, 1024);
        let mut alloc = DramAlloc::new(&Dram::FULL);
        let a = DramTensor::alloc(&mut alloc, 1, k).unwrap();
        let b = DramTensor::alloc(&mut alloc, k, n).unwrap();
        let bt = DramTensor::alloc(&mut alloc, n, k).unwrap();
        let p = DramTensor::alloc(&mut alloc, k / 32, 32).unwrap();
        let mut jobs = |busy: &[Option<u64>], b_transposed: bool| {
            super::matmul_dram_prefetching(
                &mut alloc,
                &a,
                false,
                if b_transposed { &bt } else { &b },
                b_transposed,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                units,
                false,
                false,
                None,
                super::Reads {
                    thin: true,
                    resident_a: true,
                    unpack_b: false,
                    packed_a: Some(p.tensor_ref()),
                    stream_k: false,
                    stream_prefetch: false,
                    prefetch_units: [usize::MAX; 2],
                    handoff: None,
                    second_job: false,
                    balanced_deal: false,
                    block_stream: super::BlockStream::Off,
                },
                &[],
                units,
                busy,
            )
            .unwrap()
            .0
            .jobs
        };
        let ahead =
            |job: &Job| matches!(&job[0], Step::List { what, .. } if *what == super::BUSY_PREFETCH);
        let data = tt_isa::l1::DATA;
        // Every unit free below the arena's top: each unit's first job, and only it, reads ahead.
        let all = jobs(&vec![Some(data.base); units], false);
        assert!(all.len() > units, "more than a job a unit");
        for (j, job) in all.iter().enumerate() {
            assert_eq!(ahead(job), j < units, "job {j}");
            if j >= units {
                continue;
            }
            let (
                Step::List { entries: early, .. },
                Step::List {
                    entries: gather, ..
                },
            ) = (&job[0], &job[1])
            else {
                unreachable!("two lists first")
            };
            assert_eq!(early.len(), 5, "the GATHER and its operands");
            assert_eq!(early[0][0], record::GATHER);
            assert_ne!(early[0][1] & record::GATHER_PREFETCH, 0, "flagged");
            assert_eq!(gather.last().unwrap()[..2], [op::WAIT, WAIT_PREFETCH]);
            assert!(gather.iter().all(|e| e[0] != record::GATHER));
            // `B`'s ring ends at the arena's end.
            let [kt, cols] = [k as u64 / 32, early[0][7] as u64];
            assert_eq!(early[0][3] as u64 + kt * cols * TILE_SLOT, data.end);
        }
        // Where an op before planned up into `B`'s ring, or is not known, or `B` is read
        // transposed by the mover: no unit reads ahead.
        let mut some = vec![Some(data.end); units];
        some[3] = Some(data.base);
        some[5] = None;
        let mixed = jobs(&some, false);
        for (j, job) in mixed.iter().enumerate() {
            assert_eq!(ahead(job), j == 3, "job {j}");
        }
        assert!(jobs(&vec![Some(data.base); units], true)
            .iter()
            .all(|j| !ahead(j)));
        assert!(jobs(&[], false).iter().all(|j| !ahead(j)));
    }

    #[test]
    fn a_streamed_column_reads_its_k_in_chunks() {
        use crate::matmul::{Fidelity, SrcRoute};
        let units = 64;
        // `down`'s `K` (blocked anyway), and shorter ones under a block limit.
        for (k, n, limit) in [
            (8192, 2048, None),
            (2048, 6144, Some(16)),
            (1056, 200, Some(9)),
        ] {
            let mut alloc = DramAlloc::new(&Dram::FULL);
            let a = DramTensor::alloc(&mut alloc, 1, k).unwrap();
            let b = DramTensor::alloc(&mut alloc, k, n).unwrap();
            let p = DramTensor::alloc(&mut alloc, k / 32, 32).unwrap();
            let work = super::matmul_dram_limited(
                &mut alloc,
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                units,
                false,
                false,
                limit.and_then(std::num::NonZeroUsize::new),
                super::Reads {
                    thin: true,
                    resident_a: true,
                    unpack_b: false,
                    packed_a: Some(p.tensor_ref()),
                    stream_k: true,
                    stream_prefetch: false,
                    prefetch_units: [usize::MAX; 2],
                    handoff: None,
                    second_job: false,
                    balanced_deal: false,
                    block_stream: super::BlockStream::Off,
                },
            )
            .unwrap();
            let (kt, nt) = (k / 32, n.div_ceil(32));
            assert_eq!(work.jobs.len(), nt, "[{k}, {n}]: a job a column");
            let mut first_on_unit = vec![true; units];
            let mut b_tiles = 0;
            for (j, job) in work.jobs.iter().enumerate() {
                let chunks = job.len() / 3;
                assert!(
                    chunks >= super::MIN_STREAM_CHUNKS && chunks % 2 == 0,
                    "{chunks} chunks"
                );
                for (c, step) in job.chunks(3).enumerate() {
                    let [Step::List {
                        entries: gather, ..
                    }, Step::Kernel { half, .. }, Step::List {
                        entries: scatter, ..
                    }] = step
                    else {
                        panic!("a chunk is a gather, a kernel and a scatter");
                    };
                    assert_eq!(
                        *half,
                        Some((c % 2) as u8),
                        "[{k}, {n}] column {j} chunk {c}"
                    );
                    assert_eq!(scatter.is_empty(), c + 1 < chunks, "only the last writes");
                    let reads_a = gather[0][0] == record::READ_RUN;
                    assert_eq!(reads_a, std::mem::take(&mut first_on_unit[j % units]));
                    let mut counted = crate::profile_mode::OpBytes::default();
                    counted.count(gather);
                    assert_eq!(counted.a, 0);
                    b_tiles += counted.b / tt_isa::dm::TILE_SLOT;
                }
            }
            assert_eq!(b_tiles, (kt * nt) as u64, "[{k}, {n}]: every B tile once");
        }
    }

    #[test]
    fn a_streamed_product_s_idle_units_read_their_columns_first_two_chunks_ahead() {
        use crate::matmul::{Fidelity, SrcRoute};
        let (units, spread) = (64, 120);
        let prefetch: Vec<usize> = (units..spread).collect();
        let tiles = |entries: &[[u32; 8]]| {
            let mut counted = crate::profile_mode::OpBytes::default();
            counted.count(entries);
            counted.b / tt_isa::dm::TILE_SLOT
        };
        // `down` (1.7B), and a short `K` under a block limit with fewer columns than idle units;
        // a streamed product takes at most its own count (40) of the 56.
        for (k, n, limit) in [(8192, 2048, None), (1056, 200, Some(9))] {
            for on in [true, false] {
                let mut alloc = DramAlloc::new(&Dram::FULL);
                let a = DramTensor::alloc(&mut alloc, 1, k).unwrap();
                let b = DramTensor::alloc(&mut alloc, k, n).unwrap();
                let p = DramTensor::alloc(&mut alloc, k / 32, 32).unwrap();
                let (work, dealt, _) = super::matmul_dram_prefetching(
                    &mut alloc,
                    &a,
                    false,
                    &b,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    units,
                    false,
                    false,
                    limit.and_then(std::num::NonZeroUsize::new),
                    super::Reads {
                        thin: true,
                        resident_a: true,
                        unpack_b: false,
                        packed_a: Some(p.tensor_ref()),
                        stream_k: true,
                        stream_prefetch: on,
                        prefetch_units: [usize::MAX, 40],
                        handoff: None,
                        second_job: false,
                        balanced_deal: false,
                        block_stream: super::BlockStream::Off,
                    },
                    &prefetch,
                    spread,
                    &[],
                )
                .unwrap();
                let (kt, nt) = ((k / 32) as u64, n.div_ceil(32));
                let mut columns = 0;
                let mut ahead = Vec::new();
                for (slot, job) in work.jobs.iter().enumerate().filter(|(_, j)| !j.is_empty()) {
                    let (pre, rest) = match job.first() {
                        Some(Step::List {
                            what: super::PREFETCH,
                            entries,
                        }) => (Some(tiles(entries)), &job[1..]),
                        _ => (None, &job[..]),
                    };
                    let mut read = pre.unwrap_or(0);
                    let mut first_two = 0;
                    for (c, step) in rest.chunks(3).enumerate() {
                        let [Step::List {
                            entries: gather, ..
                        }, Step::Kernel { half, .. }, Step::List { .. }] = step
                        else {
                            panic!("a chunk is a gather, a kernel and a scatter");
                        };
                        assert_eq!(*half, Some((c % 2) as u8), "slot {slot} chunk {c}");
                        if c < 2 {
                            first_two += tiles(gather);
                        }
                        read += tiles(gather);
                    }
                    assert_eq!(read, kt, "[{k}, {n}] slot {slot}: its column's B once");
                    if let Some(pre) = pre {
                        assert_eq!(first_two, 0, "[{k}, {n}] slot {slot}: chunks 0-1 ahead");
                        assert!(
                            pre < kt,
                            "[{k}, {n}] slot {slot}: only the ring's two halves"
                        );
                        ahead.push(slot);
                    }
                    columns += 1;
                }
                assert_eq!(columns, nt, "[{k}, {n}]: a job a column");
                if on {
                    assert_eq!(dealt, Some(spread));
                    let want: Vec<usize> = prefetch.iter().copied().take(nt.min(40)).collect();
                    assert_eq!(ahead, want, "[{k}, {n}]: the idle units' columns");
                } else {
                    assert_eq!(dealt, None);
                    assert!(ahead.is_empty());
                }
            }
        }
    }

    #[test]
    fn balanced_shares_are_even_with_the_remainder_last() {
        assert!(super::balanced_shares(5, 0).is_empty());
        assert_eq!(super::balanced_shares(0, 3), [0, 0, 0]);
        assert_eq!(super::balanced_shares(7, 3), [2, 2, 3]);
        assert_eq!(super::balanced_shares(2, 4), [0, 0, 1, 1]);
        for (left, deal) in [(304, 104), (128, 104), (1, 1), (999, 7)] {
            let s = super::balanced_shares(left, deal);
            assert_eq!(s.iter().sum::<usize>(), left);
            assert!(
                s.windows(2).all(|w| w[0] <= w[1]),
                "{left} over {deal}: {s:?}"
            );
            assert!(s[deal - 1] - s[0] <= 1, "{left} over {deal}: {s:?}");
        }
    }

    /// A `K`-whole prefetched product dealt balanced ([`super::Reads::balanced_deal`]): the
    /// prefetching units' jobs read ahead and the plan's units' first round as without it, then
    /// the columns left over both, no unit more than its share and the remainder on the
    /// prefetching units; every column exactly once. Nothing changes where the plan's own units
    /// would run one round (`qkv`), or with it off.
    #[test]
    fn a_balanced_deal_gives_each_unit_its_in_span_share() {
        use crate::matmul::{Fidelity, SrcRoute};
        let (units, spread) = (64, 120);
        let prefetch: Vec<usize> = (units..spread).collect();
        let deal = |n: usize, on: bool| {
            let mut alloc = DramAlloc::new(&Dram::FULL);
            let a = DramTensor::alloc(&mut alloc, 1, 2048).unwrap();
            let b = DramTensor::alloc(&mut alloc, 2048, n).unwrap();
            let p = DramTensor::alloc(&mut alloc, 64, 32).unwrap();
            let (work, ..) = super::matmul_dram_prefetching(
                &mut alloc,
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                units,
                false,
                false,
                None,
                super::Reads {
                    thin: true,
                    resident_a: true,
                    unpack_b: false,
                    packed_a: Some(p.tensor_ref()),
                    stream_k: false,
                    stream_prefetch: true,
                    prefetch_units: [40, 40],
                    handoff: None,
                    second_job: false,
                    balanced_deal: on,
                    block_stream: super::BlockStream::Off,
                },
                &prefetch,
                spread,
                &[],
            )
            .unwrap();
            work.jobs
        };
        // Each job's columns, from its gathers' headers, wherever its lists sit.
        let columns = |job: &Job| -> Vec<usize> {
            job.iter()
                .filter_map(|step| match step {
                    Step::List { entries, .. } => Some(entries),
                    _ => None,
                })
                .flatten()
                .filter(|e| e[0] == record::GATHER)
                .flat_map(|e| e[6] as usize..(e[6] + e[7]) as usize)
                .collect()
        };
        for (n, balances) in [(16384, true), (6144, false)] {
            let nt = n / 32;
            let (on, off) = (deal(n, true), deal(n, false));
            for jobs in [&on, &off] {
                let mut all: Vec<usize> = jobs.iter().flat_map(&columns).collect();
                all.sort_unstable();
                assert_eq!(
                    all,
                    (0..nt).collect::<Vec<_>>(),
                    "[2048, {n}]: each column once"
                );
            }
            let same = |x: &[Job], y: &[Job]| format!("{x:?}") == format!("{y:?}");
            assert!(
                same(&on[..spread], &off[..spread]),
                "[2048, {n}]: the first round unchanged"
            );
            if !balances {
                assert!(
                    same(&on, &off),
                    "[2048, {n}]: one round of the plan's units: unchanged"
                );
                continue;
            }
            // Each unit's columns after its first job.
            let later = |jobs: &[Job]| -> Vec<usize> {
                (0..spread)
                    .map(|u| {
                        jobs.iter()
                            .skip(spread + u)
                            .step_by(spread)
                            .map(|j| columns(j).len())
                            .sum()
                    })
                    .collect()
            };
            let (l_on, l_off) = (later(&on), later(&off));
            // `gate|up`: 80 columns read ahead, 128 in the first round, 304 left.
            let left = nt - 80 - 128;
            assert_eq!(l_on.iter().sum::<usize>(), left);
            assert_eq!(l_off.iter().sum::<usize>(), left);
            let dealt: Vec<usize> = (0..104).collect();
            let share = left.div_ceil(dealt.len());
            assert!(dealt
                .iter()
                .all(|&u| l_on[u] + 1 >= share && l_on[u] <= share));
            assert!(l_on[104..].iter().all(|&c| c == 0), "units past the deal");
            assert!(
                l_on[64..104].iter().all(|&c| c == share),
                "the remainder on the prefetching units: {:?}",
                &l_on[64..104]
            );
            assert_eq!(
                l_off.iter().max(),
                Some(&5),
                "off: the plan's units read 5 more"
            );
            assert!(
                l_off[64..].iter().all(|&c| c == 0),
                "off: the prefetching units, none"
            );
        }
    }

    /// A streamed, prefetched product's columns its own units would read whole are handed: the
    /// column's unit computes chunks `0..h` and writes the partial to `partials`, a spare unit
    /// reads the packed `A` and chunks `h, h + 1` ahead in the first op, and in the second reloads
    /// that partial into the slot its first chunk reloads from, drains the prefetch, and computes
    /// the rest, writing the column. Every `B` tile once; nothing handed when off or with no spare
    /// unit.
    #[test]
    fn a_streamed_product_s_whole_columns_are_handed_between_two_units() {
        use crate::matmul::{Fidelity, SrcRoute};
        let tiles = |entries: &[[u32; 8]]| {
            let mut counted = crate::profile_mode::OpBytes::default();
            counted.count(entries);
            counted.b / tt_isa::dm::TILE_SLOT
        };
        let gathers = |entries: &[[u32; 8]]| -> Vec<u32> {
            entries
                .iter()
                .filter(|e| e[0] == record::GATHER)
                .map(|e| e[1])
                .collect()
        };
        // `down` (1.7B: 4 chunks, 24 columns on the plan's units, 40 spare), handed at half and
        // at chunk 1; then 24 units with every other unit prefetching: no spare unit.
        for (units, spread, at) in [
            (64, 120, Some(0)),
            (64, 120, Some(1)),
            (64, 120, None),
            (24, 64, Some(0)),
        ] {
            let (k, n) = (8192, 2048);
            let prefetch: Vec<usize> = (units..spread).collect();
            let mut alloc = DramAlloc::new(&Dram::FULL);
            let a = DramTensor::alloc(&mut alloc, 1, k).unwrap();
            let b = DramTensor::alloc(&mut alloc, k, n).unwrap();
            let p = DramTensor::alloc(&mut alloc, k / 32, 32).unwrap();
            let (work, dealt, handoff) = super::matmul_dram_prefetching(
                &mut alloc,
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                units,
                false,
                false,
                None,
                super::Reads {
                    thin: true,
                    resident_a: true,
                    unpack_b: false,
                    packed_a: Some(p.tensor_ref()),
                    stream_k: true,
                    stream_prefetch: true,
                    prefetch_units: [usize::MAX, 40],
                    handoff: at,
                    second_job: false,
                    balanced_deal: false,
                    block_stream: super::BlockStream::Off,
                },
                &prefetch,
                spread,
                &[],
            )
            .unwrap();
            assert_eq!(dealt, Some(spread));
            let kt = (k / 32) as u64;
            let (c, nt) = (work.out.tensor_ref().encode(), n / 32);
            let regular = nt - prefetch.len().min(40);
            let Some(handoff) = handoff else {
                assert!(
                    at.is_none() || units == regular,
                    "units {units}, spread {spread}: handed with no spare unit"
                );
                assert!(work.jobs.iter().all(|job| !job.iter().any(
                    |s| matches!(s, Step::List { what, .. } if *what == super::HANDED_AHEAD)
                )));
                continue;
            };
            let at = at.unwrap();
            let pr = handoff.partials.tensor_ref().encode();
            let (chunks, h) = (4, if at == 0 { 2 } else { at });
            assert_eq!(handoff.jobs.len(), spread);
            let spare: Vec<usize> = (regular..regular * 2).collect();
            let mut columns = vec![0u64; nt];
            for (x, &y) in spare.iter().enumerate() {
                // The column's unit: chunks `0..h`, the last one's partial out to `partials`.
                let job = &work.jobs[x];
                assert_eq!(job.len(), 3 * h, "unit {x}: its first {h} chunks");
                let Some(Step::List { entries, .. }) = job.last() else {
                    panic!("a chunk ends with its scatter list");
                };
                assert_eq!(entries[0][0], record::SCATTER);
                assert_eq!(&entries[1..], &pr[..], "unit {x}: the partial, not C");
                let (j, from) = (entries[0][5] as usize, entries[0][1] as u64);
                let mut read = 0;
                for step in job {
                    if let Step::List { entries, .. } = step {
                        read += tiles(entries);
                        assert!(gathers(entries)
                            .iter()
                            .all(|f| f & record::GATHER_PREFETCH == 0));
                    }
                }
                // The spare unit, first op: the packed `A`, then two chunks' prefetched gathers.
                let [Step::List { what, entries }] = &work.jobs[y][..] else {
                    panic!("unit {y}: one list in the first op");
                };
                assert_eq!(*what, super::HANDED_AHEAD);
                let flags = gathers(entries);
                assert_eq!(flags.len(), 2.min(chunks - h), "unit {y}");
                assert!(flags
                    .iter()
                    .all(|f| f & record::GATHER_PREFETCH != 0 && f & record::GATHER_STREAM == 0));
                assert_eq!(
                    entries[0][0],
                    record::READ_RUN,
                    "unit {y}: the packed A first"
                );
                read += tiles(entries);
                // The second op: the partial reloaded where X packed it, the prefetch drained,
                // then chunks `h..`, the last writing C.
                let rest = &handoff.jobs[y];
                assert_eq!(rest.len(), 3 * (chunks - h), "unit {y}: chunks {h}..");
                let Some(Step::List { entries: lead, .. }) = rest.first() else {
                    panic!("a chunk starts with its gather list");
                };
                assert_eq!(lead[0][0], op::WAIT);
                assert_eq!(lead[1][0], record::READ_RUN);
                assert_eq!(lead[1][1] as usize, j, "unit {y}: its column's partial");
                assert_eq!(
                    lead[1][3] as u64 + TILE_DATA,
                    from,
                    "unit {y}: into the slot X packed into"
                );
                assert_eq!(&lead[2..4], &pr[..]);
                assert_eq!(lead[4][..2], [op::WAIT, tt_isa::dm::WAIT_PREFETCH]);
                for (q, step) in rest.chunks(3).enumerate() {
                    let [Step::List { entries: g, .. }, Step::Kernel { half, .. }, Step::List { entries: sc, .. }] =
                        step
                    else {
                        panic!("a chunk is a gather, a kernel and a scatter");
                    };
                    assert_eq!(*half, Some(((h + q) % 2) as u8), "unit {y} chunk {}", h + q);
                    read += tiles(g);
                    if q + 1 == chunks - h {
                        assert_eq!(sc[0][5] as usize, j);
                        assert_eq!(&sc[1..], &c[..], "unit {y}: the column to C");
                    } else {
                        assert!(sc.is_empty());
                    }
                }
                assert_eq!(read, kt, "column {j}: its B once");
                columns[j] += 1;
            }
            assert!(columns.iter().filter(|&&c| c > 0).all(|&c| c == 1));
            assert_eq!(
                columns.iter().sum::<u64>(),
                regular as u64,
                "every regular column handed"
            );
            // Nothing else in the second op.
            assert_eq!(
                handoff.jobs.iter().filter(|j| !j.is_empty()).count(),
                regular
            );
        }
    }

    #[test]
    fn decode_products_jobs_rounds_and_bytes() {
        use crate::matmul::{Fidelity, SrcRoute};
        use tt_isa::dm::TILE_SLOT;
        let units = 120;
        // name, [k, n], B transposed (the tied head reads the table transposed)
        let shapes = [
            ("q/k/v/o", [2048, 2048], false),
            ("gate/up", [2048, 8192], false),
            ("down", [8192, 2048], false),
            ("head", [2048, 49152], true),
            // `Llama::fused`'s products (1.7B): `q|k|v` and `gate|up` as one each.
            ("qkv", [2048, 6144], false),
            ("gate|up", [2048, 16384], false),
        ];
        let mut rows = Vec::new();
        for (name, [k, n], bt) in shapes {
            let mut alloc = DramAlloc::new(&Dram::FULL);
            let a = DramTensor::alloc(&mut alloc, 1, k).unwrap();
            let b = match bt {
                true => DramTensor::alloc(&mut alloc, n, k),
                false => DramTensor::alloc(&mut alloc, k, n),
            }
            .unwrap();
            let work = super::matmul_dram(
                &mut alloc,
                &a,
                false,
                &b,
                bt,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                units,
                false,
                true,
            )
            .unwrap();
            let (mut a_bytes, mut b_bytes, mut gathers) = (0u64, 0u64, 0usize);
            let (mut a_whole, mut gather_reads) = (0u64, 0u64);
            // The profile mode's count of the same lists, records expanded.
            let mut counted = crate::profile_mode::OpBytes::default();
            let mut per_unit = vec![0usize; units];
            for (j, job) in work.jobs.iter().enumerate() {
                per_unit[j % units] += 1;
                for step in job {
                    let Step::List { entries, .. } = step else {
                        continue;
                    };
                    counted.count(entries);
                    for e in entries.iter().filter(|e| e[0] == record::GATHER) {
                        let kt = u64::from(record::gather_kt(e[1]));
                        let [rows, cols] = [u64::from(e[5]), u64::from(e[7])];
                        // A thin last tile row reads through row `r` of face 1.
                        let r = u64::from(e[1] >> record::GATHER_ROWS_SHIFT & 0x1f);
                        let last = if r == 0 { TILE_SLOT } else { 1024 + 64 * r };
                        a_bytes += ((rows - 1) * TILE_SLOT + last) * kt;
                        a_whole += rows * TILE_SLOT * kt;
                        b_bytes += kt * cols * TILE_SLOT;
                        gather_reads += (rows + cols) * kt;
                        gathers += 1;
                    }
                }
            }
            // The counter agrees with the headers, operand by operand; the control is `A` read as
            // whole tiles, which a thin product's count must not match.
            assert_eq!((counted.a, counted.b), (a_bytes, b_bytes), "{name}");
            assert_ne!(
                counted.a, a_whole,
                "{name}: thin reads counted as whole tiles"
            );
            assert!(counted.reads >= gather_reads, "{name}");
            assert!(counted.read >= a_bytes + b_bytes, "{name}");
            let plan = crate::matmul::plan_in(
                [1, k, n],
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                crate::matmul::Staging::Slots,
            )
            .unwrap()
            .tiles;
            let busy = per_unit.iter().filter(|&&n| n > 0).count();
            let rounds = *per_unit.iter().max().unwrap();
            let two = per_unit.iter().filter(|&&n| n > 1).count();
            rows.push(format!(
                "{name:8} plan {plan:?} jobs {:4} gathers {gathers:4} busy {busy:3} rounds {rounds} \
                 (units with >1 job: {two:3}) A {:6.1} MB  B {:6.1} MB  A share {:4.1}%  \
                 reads {:6} written {:5.2} MB",
                work.jobs.len(),
                a_bytes as f64 / 1e6,
                b_bytes as f64 / 1e6,
                100.0 * a_bytes as f64 / (a_bytes + b_bytes) as f64,
                counted.reads,
                counted.write as f64 / 1e6,
            ));
            // `B`'s whole tiles are read once: blocks never share a column.
            assert_eq!(b_bytes, (k as u64 / 32) * (n as u64 / 32) * TILE_SLOT);
        }
        println!("{}", rows.join("\n"));
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
                // A job of the last tile row alone computes only its rows.
                let live = matmul::LiveRows::for_job(m, i0, rows);
                let roles =
                    matmul::programs((tiles, Staging::Slots), route, fidelity, sems, live, || {
                        let (words, _) = matmul::matmul_kernel(
                            &outputs, sems, in_fmt, out_fmt, fidelity, false, false, live,
                        );
                        words
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

#[cfg(test)]
mod block_stream_tests {
    use super::*;

    fn reads(mode: BlockStream) -> Reads {
        Reads {
            thin: true,
            resident_a: true,
            unpack_b: true,
            packed_a: None,
            stream_k: false,
            stream_prefetch: false,
            prefetch_units: [usize::MAX; 2],
            handoff: None,
            second_job: false,
            balanced_deal: false,
            block_stream: mode,
        }
    }

    fn work(mkn: [usize; 3], units: usize, mode: BlockStream) -> Work {
        let [m, k, n] = mkn;
        let mut alloc = DramAlloc::new(&Dram::FULL);
        let a = DramTensor::alloc(&mut alloc, m, k).unwrap();
        let b = DramTensor::alloc(&mut alloc, k, n).unwrap();
        matmul_dram_limited(
            &mut alloc,
            &a,
            false,
            &b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            units,
            false,
            true,
            None,
            reads(mode),
        )
        .unwrap()
    }

    #[test]
    fn a_blocked_job_reads_its_strips_once_and_writes_its_block_once() {
        // 8 x 16 x 13 tiles, ragged along `N` against 3 x 5 blocks.
        let [mt, kt, nt] = [8, 16, 13];
        let [mb, kq, nb] = [3, 4, 5];
        let work = work(
            [mt * 32, kt * 32, nt * 32],
            7,
            BlockStream::Fixed([mb, kq, nb]),
        );
        let (row_blocks, col_blocks) = (mt.div_ceil(mb), nt.div_ceil(nb));
        assert_eq!(work.jobs.len(), row_blocks * col_blocks);
        let mut read_tiles = 0;
        let mut written = 0;
        for job in &work.jobs {
            let halves: Vec<u8> = job
                .iter()
                .filter_map(|s| match s {
                    Step::Kernel { half, .. } => *half,
                    _ => None,
                })
                .collect();
            assert_eq!(halves.len(), kt / kq, "a kernel a chunk");
            assert_eq!(
                halves.len() % 2,
                0,
                "chunks even, so units alternate halves"
            );
            assert!(halves.iter().enumerate().all(|(c, &h)| h as usize == c % 2));
            let (mut k_read, mut scatters, mut block) = (0, 0, None);
            for (index, step) in job.iter().enumerate() {
                let Step::List { entries, .. } = step else {
                    continue;
                };
                for e in entries {
                    match e[0] {
                        record::GATHER => {
                            let count = record::gather_kt(e[1]) as usize;
                            let [rows, cols] = [e[5] as usize, e[7] as usize];
                            assert!(block.is_none_or(|b| b == [rows, cols]));
                            block = Some([rows, cols]);
                            read_tiles += (rows + cols) * count;
                            k_read += count;
                        }
                        record::SCATTER => {
                            assert_eq!(index, job.len() - 1, "only the last chunk writes");
                            written += (e[4] * e[6]) as usize;
                            scatters += 1;
                        }
                        _ => {}
                    }
                }
            }
            assert_eq!(k_read, kt, "the job reads its strips over `K` once");
            assert_eq!(scatters, 1);
        }
        assert_eq!(written, mt * nt);
        assert_eq!(read_tiles, (col_blocks * mt + row_blocks * nt) * kt);
    }

    #[test]
    fn off_and_a_single_row_take_the_other_paths() {
        for (mkn, mode) in [
            ([256, 512, 512], BlockStream::Off),
            ([1, 512, 512], BlockStream::Fixed([4, 4, 4])),
            ([256, 32, 512], BlockStream::Fixed([4, 4, 4])),
        ] {
            let work = work(mkn, 8, mode);
            assert!(work.jobs.iter().flatten().all(|s| !matches!(
                s,
                Step::List { what, .. } if what.starts_with("blocked")
            )));
        }
    }

    /// A looped block's programs are one output's block a role, whatever the block
    /// (`docs/plans/prefill-operand-reuse.md` P1b); unrolled, the unpack role grows ~130 words an
    /// output.
    #[test]
    fn looped_block_programs_do_not_grow_with_the_block() {
        let stored = |tiles: [usize; 3], c: usize| -> [usize; 3] {
            let layout = matmul::block_stream_layout(tiles).unwrap();
            let (code, loops) = matmul::block_stream_kernel_looped(
                &layout,
                c,
                tiles,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                false,
                matmul::LiveRows::All,
            )
            .expect("the loops fit");
            std::array::from_fn(|t| {
                crate::code::Code {
                    ins: code[t].clone(),
                    loops: loops[t].clone(),
                }
                .stored()
                .unwrap()
                .0
                .len()
            })
        };
        for c in 0..2 {
            let small = stored([2, 2, 2], c);
            let large = stored([8, 2, 8], c);
            assert_eq!(small, large, "chunk {c}: the same words, more repeats");
            assert!(small.iter().all(|&w| w < 400), "chunk {c}: {small:?} words");
        }
    }

    /// The model's choices for SmolLM2-1.7B's prefill products on 120 units, against its model of
    /// the other paths. Printed for the plan's record; the bound is the model's, not a measurement.
    #[test]
    fn the_17b_prefill_products_plan_blocks_that_read_less() {
        let units = 120;
        let cases = [
            ("gate|up", [2048, 16384]),
            ("down", [8192, 2048]),
            ("qkv", [2048, 3072]),
            ("o", [2048, 2048]),
        ];
        for m in [64, 512, 2048] {
            for (name, [k, n]) in cases {
                let tiles = [m / 32, k / 32, n / 32];
                let (shape, _) = matmul::plan_floor(
                    [m, k, n],
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    Staging::Slots,
                );
                let [mc, kc, nc] = shape.unwrap().tiles;
                let kc = if kc < tiles[1] {
                    k_block_size(kc, SrcRoute::Tf32FromFp32, Fidelity::HiFi4).unwrap()
                } else {
                    kc
                };
                let [mc, nc] = blocks([tiles[0], tiles[2]], [mc, nc], units);
                let other = blocked_paths_model(tiles, [mc, kc, nc], units);
                let choice = block_stream_choice(
                    BlockStream::Auto,
                    tiles,
                    units,
                    other,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                );
                println!(
                    "m {m} {name}: other [{mc}, {kc}, {nc}] {other:.0} us; blocked {choice:?}"
                );
                if m >= 512 {
                    let s = choice.expect("a long prefill's products block");
                    assert!(s.us < other);
                }
            }
        }
    }
}
