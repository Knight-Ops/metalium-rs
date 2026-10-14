//! Op records: a whole op's mover work in a few list entries, expanded on the
//! tile.
//!
//! A plain list entry ([`super::Entry`]) moves or computes one tile, so a list
//! is as long as the op is big, and its bytes -- not the round trips -- were
//! what a GDDR op spent its host time on (the host's MMIO is uncached here,
//! `docs/learnings/ttsim-divergence.md` measurement M). A record names the tensors
//! instead ([`TensorRef`]: a handful of words for any size) and what to do with
//! them, and [`expand`] produces the entries the host used to send, one at a
//! time, for the mover to run as it ran them. The host expands every record
//! first too, so a record the mover would refuse is refused before it crosses
//! PCIe -- and `tt_kernels::tensor`'s tests hold the expansion to the list
//! builders it replaced, entry for entry.
//!
//! A record is a header entry, whose first word is one of the record ops
//! below, followed by two entries per tensor; [`len`] says how many entries in
//! all. Records sit in a list among plain entries, and a list never splits one.

use super::{op, TILE_DATA, TILE_SLOT};

/// One matmul block's operands, GDDR -> L1: `[GATHER, flags, a_at, b_at, i0,
/// rows, j0, cols]` + `A` + `B`. Row `i` of the block's `A` tiles, `K` tile
/// `kk`, goes to `a_at + (i * kt + kk) * TILE_SLOT`; `K` tile `kk` of its
/// column `j` of `B` to `b_at + (kk * cols + j) * TILE_SLOT`. `flags`: bit 0
/// `A` is read transposed, bit 1 `B`, bits 8.. `kt`. Bit 2 selects packed
/// BF16 operands (no transpose): their tensor metadata words 4/5 contain the
/// logical rows/columns and words 6/7 their tile row/column origins.
/// GDDR slots are 2112 bytes; L1 slots remain 4160.
/// Only local ragged lanes are zeroed, without changing the source tensor.
pub const GATHER: u32 = 0x10;
pub const GATHER_BF16: u32 = 1 << 2;
/// One matmul block's outputs, L1 -> GDDR: `[SCATTER, out_at, out_stride, i0,
/// rows, j0, cols, 0]` + `C`. Output `(i, j)`'s datums are at `out_at + (i *
/// cols + j) * out_stride`.
pub const SCATTER: u32 = 0x11;
/// Set the padding of a run of a tensor's edge tiles: `[FILL_PAD, value,
/// first, count, rows, cols, stage, rt]` + `A`, where `rows` and `cols` are
/// the valid rows of the last tile row and columns of the last tile column
/// (`0`: 32, that edge is not ragged) and `rt` the tile rows. The edge tiles,
/// numbered: the last tile row left to right if its rows are ragged, then the
/// last tile column top to bottom (without a corner already counted) if its
/// columns are; tiles `first..first + count` of that numbering are each read
/// into slot `n` of the staging area at `stage`, filled with `value` outside
/// the valid region ([`op::FILL`]) and written back in place.
pub const FILL_PAD: u32 = 0x14;
/// A run of whole tiles, GDDR -> L1: `[READ_RUN, first, count, at, flags, ct,
/// 0, 0]` + `X`. Tiles `first..first + count`, row-major over a grid `ct`
/// tiles wide (the output's; `X`'s own when it is not a broadcast), each into
/// the next tile slot from `at`. `flags` bit 0: `X` is one tile row broadcast
/// down -- tile `(0, j)` read for every `(i, j)` -- which is how a `[1, n]`
/// bias meets a `[m, n]` tensor; bit 1: `X` is one tile column broadcast
/// across -- tile `(i, 0)` read, its column 0 copied into every column
/// ([`op::READ_BROADCAST_COL`]) -- how a `[m, 1]` tensor meets one; bit 2:
/// the run counts column-major, down a grid `rt` tiles tall (word 6) -- tile
/// `n` is `((first + n) % rt, (first + n) / rt)` -- which is the order a
/// reduction over rows reads a column of tiles in; bit 3: each tile `(i, j)`
/// of the run is `X`'s tile `(j, i)` transposed ([`op::READ_TRANSPOSED`]) --
/// a block of `X` read as its transpose, for a block copy (not with bits
/// 0-2).
pub const READ_RUN: u32 = 0x15;
/// A run of whole tiles' datums, L1 -> GDDR: `[WRITE_RUN, first, count, at,
/// 0, 0, 0, 0]` + `X`. Slot `n` from `at` (its datums, past the header) to
/// tile `first + n` of `X`, row-major. What a kernel's packer wrote goes back.
pub const WRITE_RUN: u32 = 0x16;
/// The write half of [`FILL_PAD`], on the writer: `[PAD_WRITE, value, first,
/// count, rows, cols, stage, rt]` + `A`, the same header and tensor. Each edge
/// tile `first..first + count`, as [`FILL_PAD`] numbers them, goes from its
/// slot at `stage` (the reader filled it) back to its place in `A`.
pub const PAD_WRITE: u32 = 0x17;
/// Whole rows of a table, chosen by indices already in L1, GDDR -> L1:
/// `[SELECT_ROWS, idx_at, count, bound, j0, cols, stage_at, flags]` + `X`.
/// The slot at `idx_at` holds one tile of I32 indices, in tile row 0 (a
/// `[1, n]` index tensor's tile): index `r` is the datum at `(0, r)`. Row `q`
/// of a slot is read as row 0 of one `idx_at` moved by that row's place in
/// the face layout, `(q / 16 * 512 + q % 16 * 16) * 4` bytes, which is how a
/// `[taps, n]` index tensor gives each tap its own row
/// (`tt_kernels::index::select_taps_jobs`, a convolution's im2col). For each
/// of the first `count` (1..=32) indices `v`, row `v` of `X` (a table of
/// `bound` rows) goes to row `r` of the output tile, for tile columns
/// `j0..j0 + cols` of `X`, into the slot `j - j0` from `stage_at` -- two
/// 64-byte face-row reads a tile column, as `tensor::gather_rows` makes them,
/// except that the source tile is found here, from the index the card holds.
/// The mover refuses an index at or past `bound` ([`super::error::DOMAIN`])
/// before it issues a read, the contract of [`op::INDEX_PICK`] (which reads
/// the index as `u32`, so a negative one is refused too). Rows `count..32` of
/// each output tile are left as the slot held them.
///
/// `flags` is `0`: 32-bit datums (F32, I32, Bool). **Packed BF16 is not
/// implemented.** Its face-rows are 32 bytes, congruent to `TILE_DATA` mod 64
/// only on an even row, so a read must take the pair a row is half of and the
/// wanted half be moved; that code, in this record, measured 400 bytes more
/// in RISCV B's image, which then had 172 of its 24 KiB left (a mover image
/// may not pass `IMAGE_MAX`; the slot is 42 KiB since the role images were
/// repacked). `docs/plans/roadmap.md`, "Gather without staging", has the
/// options.
///
/// The slots are not checked against the data arena here (the mover's image has
/// no room for it; `tt_kernels::index::select_rows_jobs` places them from its
/// plan and refuses any outside it); each read's L1 range is checked as any
/// read's is.
///
/// The one record whose expansion depends on L1 data: the reader the caller
/// gives [`expand_reads`] says what an index word holds. The host's
/// [`expand`] has no L1, reads every index as `0` and so checks the record's
/// geometry (alignment, range) and nothing about the indices.
pub const SELECT_ROWS: u32 = 0x18;
/// A run of consecutive words of one role's mailbox: `[POKE_RUN, address,
/// count, v0, v1, v2, v3, v4]` + `[v5, ..., v12]`, `count` in
/// `1..=`[`POKE_RUN_MAX`], the values past `count` zero. The same words
/// [`op::POKE`] writes one at a time and in the same order, so a descriptor
/// goes in two records (the live handshake words, `generation` and `ack`,
/// divide it) instead of seventeen entries, each of which fences. Neither a
/// GDDR read nor a write: RISCV B runs it, NC refuses it as a direction it
/// does not serve.
pub const POKE_RUN: u32 = 0x19;
/// Words one [`POKE_RUN`] carries: five in its header, eight in the entry after it.
pub const POKE_RUN_MAX: usize = 13;

/// A [`POKE_RUN`] of `values` into the words from `address`, which the caller
/// has checked is a role mailbox's (`Entry::decode` refuses what is not, of
/// every word the record expands to).
///
/// # Panics
/// If `values` is empty or longer than [`POKE_RUN_MAX`].
pub fn poke_run(address: u32, values: &[u32]) -> [[u32; 8]; 2] {
    assert!(
        !values.is_empty() && values.len() <= POKE_RUN_MAX,
        "a POKE_RUN carries 1..={POKE_RUN_MAX} words, not {}",
        values.len()
    );
    let mut rec = [[0u32; 8]; 2];
    rec[0][0] = POKE_RUN;
    rec[0][1] = address;
    rec[0][2] = values.len() as u32;
    for (k, &v) in values.iter().enumerate() {
        let (entry, slot) = if k < 5 { (0, 3 + k) } else { (1, k - 5) };
        rec[entry][slot] = v;
    }
    rec
}

/// The `k`th word a [`POKE_RUN`] carries, by the layout [`poke_run`] writes.
#[inline(always)]
pub const fn poke_run_word(rec: &[[u32; 8]], k: usize) -> u32 {
    if k < 5 {
        rec[0][3 + k]
    } else {
        rec[1][k - 5]
    }
}

/// Does a record's expansion read GDDR (so B runs it) or write it (so NC
/// does)? `None` for a plain entry.
pub const fn direction(op: u32) -> Option<Direction> {
    match op {
        GATHER | READ_RUN | FILL_PAD | SELECT_ROWS => Some(Direction::Read),
        SCATTER | WRITE_RUN | PAD_WRITE => Some(Direction::Write),
        _ => None,
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Direction {
    /// GDDR to L1: RISCV B's.
    Read,
    /// L1 to GDDR: RISCV NC's.
    Write,
}

/// Most tiles a record may name along any one of its loops: far more than any
/// list holds, and few enough that a corrupt record cannot keep the mover busy
/// for long.
pub const MAX_EXTENT: u32 = 1 << 16;

/// Entries in the record whose header word 0 is `op`: 1 for a plain entry.
pub const fn len(op: u32) -> usize {
    match op {
        GATHER => 5,
        SCATTER | FILL_PAD | PAD_WRITE | READ_RUN | WRITE_RUN | SELECT_ROWS => 3,
        POKE_RUN => 2,
        _ => 1,
    }
}

/// Is `op` a record header?
pub const fn is_record(op: u32) -> bool {
    len(op) > 1
}

/// A tensor as the mover addresses it: FP32 tiles in [`TILE_SLOT`] slots,
/// interleaved over `n` channels, row-major over a grid `ct` tiles wide --
/// tile `t` (counted from `first`, for a view) on `channels[t % n]` at slot
/// `t / n` of the region starting at `base` there. The same placement
/// `tt_kernels::tensor::Placement` keeps.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct TensorRef {
    pub n: u8,
    pub channels: [u8; 8],
    pub base: [u32; 8],
    pub first: u32,
    pub ct: u32,
}

impl TensorRef {
    /// `[n, channels (four bits each), first, ct, 0, 0, 0, 0]` and the bases.
    pub fn encode(&self) -> [[u32; 8]; 2] {
        let mut packed = 0u32;
        for (i, c) in self.channels.iter().enumerate() {
            packed |= ((*c as u32) & 0xF) << (4 * i);
        }
        [
            [self.n as u32, packed, self.first, self.ct, 0, 0, 0, 0],
            self.base,
        ]
    }

    /// The inverse of [`TensorRef::encode`]. `n` may be 0 only for a tensor
    /// that is never read (a record's absent operand); whether each channel is one
    /// the chip has is checked on every entry the tensor produces.
    pub fn decode(w: [[u32; 8]; 2]) -> Result<Self, u32> {
        let [m, base] = w;
        if m[0] > 8 || m[3] > MAX_EXTENT {
            return Err(super::error::RANGE);
        }
        let mut channels = [0u8; 8];
        for (i, c) in channels.iter_mut().enumerate() {
            *c = ((m[1] >> (4 * i)) & 0xF) as u8;
        }
        Ok(TensorRef {
            n: m[0] as u8,
            channels,
            base,
            first: m[2],
            ct: m[3],
        })
    }

    /// `(channel, offset)` of tile `(i, j)`'s slot.
    fn tile(&self, i: u32, j: u32) -> Result<(u32, u32), u32> {
        if self.n == 0 {
            return Err(super::error::RANGE);
        }
        let t = self.first + i * self.ct + j;
        let (s, c) = (t / self.n as u32, t % self.n as u32);
        let offset = (self.base[c as usize] as u64) + s as u64 * TILE_SLOT;
        let offset = u32::try_from(offset).map_err(|_| super::error::RANGE)?;
        Ok((self.channels[c as usize] as u32, offset))
    }
}

/// [`TensorRef::tile`] for consecutive tiles of a row, from `(i, j)` on:
/// one division to find the first, then each next is the next channel, its
/// slot advancing when the channels wrap: a few cycles a tile, against a
/// divide's 6-33 (`BabyRISCV/README.md:51`). The mover's gathers and
/// scatters walk rows. (Before the firmware could divide, the software
/// division this replaced was most of a gathered tile's cost, checklist 9.14.)
struct Cursor<'a> {
    x: &'a TensorRef,
    slot: u32,
    c: u32,
}

impl TensorRef {
    fn cursor(&self, i: u32, j: u32) -> Result<Cursor<'_>, u32> {
        if self.n == 0 {
            return Err(super::error::RANGE);
        }
        let t = self.first + i * self.ct + j;
        let (slot, c) = (t / self.n as u32, t % self.n as u32);
        Ok(Cursor { x: self, slot, c })
    }
}

impl Cursor<'_> {
    /// The tile [`TensorRef::tile`] would give for this position, and on to
    /// the next.
    #[inline(always)]
    fn next(&mut self) -> Result<(u32, u32), u32> {
        self.next_with_slot(TILE_SLOT)
    }

    fn next_with_slot(&mut self, stride: u64) -> Result<(u32, u32), u32> {
        let c = self.c as usize;
        let offset = (self.x.base[c] as u64) + self.slot as u64 * stride;
        let offset = u32::try_from(offset).map_err(|_| super::error::RANGE)?;
        let at = (self.x.channels[c] as u32, offset);
        self.c += 1;
        if self.c == self.x.n as u32 {
            self.c = 0;
            self.slot += 1;
        }
        Ok(at)
    }
}

const PORTS: u32 = crate::dram::PORTS as u32;

fn tensor(rec: &[[u32; 8]], at: usize) -> Result<TensorRef, u32> {
    TensorRef::decode([rec[at], rec[at + 1]])
}

fn extent(v: u32) -> Result<u32, u32> {
    if v == 0 || v > MAX_EXTENT {
        Err(super::error::LENGTH)
    } else {
        Ok(v)
    }
}

/// Expand the record `rec` (exactly [`len`] entries) into list entries, handing
/// each to `emit` in order, and stop at the first error either returns.
///
/// The entries are what `tt_kernels::tensor`'s list builders produced before
/// records existed, with an [`op::WAIT`] wherever one of their lists ended
/// inside a job: exactly the list the session used to send.
///
/// The host's expansion, for validation: it has no L1, so a [`SELECT_ROWS`]
/// index reads as `0` (see there).
pub fn expand(rec: &[[u32; 8]], emit: impl FnMut([u32; 8]) -> Result<(), u32>) -> Result<(), u32> {
    expand_for::<true, true>(rec, &mut |_| 0, emit)
}

/// [`expand`] of the reading records only ([`Direction::Read`]): what RISCV
/// B's firmware calls, so the writing records' code is not in its image. A
/// writing record is [`super::error::DIRECTION`]. `index` reads the L1 word
/// at an address, for [`SELECT_ROWS`]; the caller has made the indices
/// visible (waited for the reads that fetched them) before it calls.
pub fn expand_reads(
    rec: &[[u32; 8]],
    mut index: impl FnMut(u32) -> u32,
    emit: impl FnMut([u32; 8]) -> Result<(), u32>,
) -> Result<(), u32> {
    expand_for::<true, false>(rec, &mut index, emit)
}

/// [`expand`] of the writing records only: what RISCV NC's firmware calls.
pub fn expand_writes(
    rec: &[[u32; 8]],
    emit: impl FnMut([u32; 8]) -> Result<(), u32>,
) -> Result<(), u32> {
    expand_for::<false, true>(rec, &mut |_| 0, emit)
}

#[inline(always)]
fn expand_for<const READS: bool, const WRITES: bool>(
    rec: &[[u32; 8]],
    index: &mut impl FnMut(u32) -> u32,
    mut emit: impl FnMut([u32; 8]) -> Result<(), u32>,
) -> Result<(), u32> {
    let h = rec[0];
    if rec.len() != len(h[0]) || !is_record(h[0]) {
        return Err(super::error::OP);
    }
    match direction(h[0]) {
        Some(Direction::Read) if !READS => return Err(super::error::DIRECTION),
        Some(Direction::Write) if !WRITES => return Err(super::error::DIRECTION),
        _ => {}
    }
    match h[0] {
        GATHER => {
            let [_, flags, a_at, b_at, i0, rows, j0, cols] = h;
            let (a_t, b_t, kt) = (flags & 1 != 0, flags & 2 != 0, extent(flags >> 8)?);
            let (rows, cols) = (extent(rows)?, extent(cols)?);
            let (a, b) = (tensor(rec, 1)?, tensor(rec, 3)?);
            if flags & GATHER_BF16 != 0 {
                if flags & 0xff != GATHER_BF16 {
                    return Err(super::error::RANGE);
                }
                for (x, meta, r0, nr, c0, nc, at) in [
                    (&a, rec[1], i0, rows, 0, kt, a_at),
                    (&b, rec[3], 0, kt, j0, cols, b_at),
                ] {
                    let (height, width) = (meta[4], meta[5]);
                    let r0 = r0.checked_add(meta[6]).ok_or(super::error::RANGE)?;
                    let c0 = c0.checked_add(meta[7]).ok_or(super::error::RANGE)?;
                    if height == 0
                        || width == 0
                        || height > MAX_EXTENT * 32
                        || width > MAX_EXTENT * 32
                        || x.ct != width.div_ceil(32)
                        || r0
                            .checked_add(nr)
                            .is_none_or(|end| end > height.div_ceil(32))
                        || c0.checked_add(nc).is_none_or(|end| end > x.ct)
                    {
                        return Err(super::error::RANGE);
                    }
                    for i in 0..nr {
                        let mut row = x.cursor(r0 + i, c0)?;
                        for j in 0..nc {
                            let (ch, off) = row.next_with_slot(super::BF16_TILE_SLOT)?;
                            let to = (at as u64) + (i as u64 * nc as u64 + j as u64) * TILE_SLOT;
                            let to = u32::try_from(to).map_err(|_| super::error::RANGE)?;
                            emit([
                                op::READ,
                                ch,
                                (i + j) % PORTS,
                                off,
                                to,
                                super::BF16_TILE_SLOT as u32,
                                0,
                                0,
                            ])?;
                            let valid_rows = (height - (r0 + i) * 32).min(32);
                            let valid_cols = (width - (c0 + j) * 32).min(32);
                            if valid_rows != 32 || valid_cols != 32 {
                                emit([
                                    op::FILL_HALFWORDS,
                                    0,
                                    super::fill::param(valid_rows, valid_cols),
                                    to,
                                    0,
                                    0,
                                    0,
                                    0,
                                ])?;
                            }
                        }
                    }
                }
                return Ok(());
            }
            // Tile (i, j) of op(X), into L1 at `to`.
            let fetch =
                |x: &TensorRef, t: bool, i: u32, j: u32, to: u32| -> Result<[u32; 8], u32> {
                    let ((ch, off), op) = if t {
                        (x.tile(j, i)?, op::READ_TRANSPOSED)
                    } else {
                        (x.tile(i, j)?, op::READ)
                    };
                    Ok([op, ch, (i + j) % PORTS, off, to, TILE_SLOT as u32, 0, 0])
                };
            // An untransposed operand's row is consecutive tiles: a cursor
            // steps along it. A transposed one is read down a column.
            let read = |(ch, off): (u32, u32), i: u32, j: u32, to: u32| {
                [
                    op::READ,
                    ch,
                    (i + j) % PORTS,
                    off,
                    to,
                    TILE_SLOT as u32,
                    0,
                    0,
                ]
            };
            for i in 0..rows {
                let mut row = if a_t {
                    None
                } else {
                    Some(a.cursor(i0 + i, 0)?)
                };
                for kk in 0..kt {
                    let to = a_at + (i * kt + kk) * TILE_SLOT as u32;
                    emit(match row.as_mut() {
                        Some(r) => read(r.next()?, i0 + i, kk, to),
                        None => fetch(&a, a_t, i0 + i, kk, to)?,
                    })?;
                }
            }
            for kk in 0..kt {
                let mut row = if b_t { None } else { Some(b.cursor(kk, j0)?) };
                for j in 0..cols {
                    let to = b_at + (kk * cols + j) * TILE_SLOT as u32;
                    emit(match row.as_mut() {
                        Some(r) => read(r.next()?, kk, j0 + j, to),
                        None => fetch(&b, b_t, kk, j0 + j, to)?,
                    })?;
                }
            }
        }
        SCATTER => {
            let [_, out_at, stride, i0, rows, j0, cols, _] = h;
            let (rows, cols) = (extent(rows)?, extent(cols)?);
            let c = tensor(rec, 1)?;
            for i in 0..rows {
                let mut row = c.cursor(i0 + i, j0)?;
                for j in 0..cols {
                    let (ch, off) = row.next()?;
                    let out = out_at + (i * cols + j) * stride;
                    emit([
                        op::WRITE,
                        ch,
                        (i + j) % PORTS,
                        off + TILE_DATA as u32,
                        out,
                        4096,
                        0,
                        0,
                    ])?;
                }
            }
        }
        READ_RUN | WRITE_RUN => {
            let [_, first, count, at, flags, grid_ct, grid_rt, _] = h;
            let count = extent(count)?;
            let x = tensor(rec, 1)?;
            let read = READS && (!WRITES || h[0] == READ_RUN);
            let (row, col) = (read && flags & 1 != 0, read && flags & 2 != 0);
            // The grid the run counts over: the output's for a broadcast, the
            // tensor's own otherwise (and for every older record, whose word
            // 5 is zero).
            let ct = if grid_ct != 0 { grid_ct } else { x.ct };
            if ct == 0 || x.ct == 0 || (row && col) {
                return Err(super::error::LENGTH);
            }
            let column_major = read && flags & 4 != 0;
            if column_major && grid_rt == 0 {
                return Err(super::error::LENGTH);
            }
            let transposed = read && flags & 8 != 0;
            if transposed && (row || col || column_major) {
                return Err(super::error::LENGTH);
            }
            let (mut i, mut j) = if column_major {
                let (q, r) = (first / grid_rt, first % grid_rt);
                (r, q)
            } else {
                (first / ct, first % ct)
            };
            // A plain run over the tensor's own grid is consecutive tiles.
            let mut run = if !row && !col && !column_major && !transposed && ct == x.ct {
                Some(x.cursor(i, j)?)
            } else {
                None
            };
            for n in 0..count {
                if n > 0 && column_major {
                    i += 1;
                    if i == grid_rt {
                        (i, j) = (0, j + 1);
                    }
                } else if n > 0 {
                    j += 1;
                    if j == ct {
                        (i, j) = (i + 1, 0);
                    }
                }
                let slot = at + n * TILE_SLOT as u32;
                let (ch, off) = match run.as_mut() {
                    Some(r) => r.next()?,
                    None if transposed => x.tile(j, i)?,
                    None => x.tile(if row { 0 } else { i }, if col { 0 } else { j })?,
                };
                let op = if col {
                    op::READ_BROADCAST_COL
                } else if transposed {
                    op::READ_TRANSPOSED
                } else {
                    op::READ
                };
                emit(if read {
                    [op, ch, n % PORTS, off, slot, TILE_SLOT as u32, 0, 0]
                } else {
                    [
                        op::WRITE,
                        ch,
                        n % PORTS,
                        off + TILE_DATA as u32,
                        slot + TILE_DATA as u32,
                        4096,
                        0,
                        0,
                    ]
                })?;
            }
        }
        SELECT_ROWS => {
            let [_, idx_at, count, bound, j0, cols, stage_at, flags] = h;
            if flags != 0 {
                // Packed BF16 is not implemented.
                return Err(super::error::RANGE);
            }
            if count == 0 || count > 32 || bound == 0 || bound > i32::MAX as u32 {
                return Err(super::error::LENGTH);
            }
            let cols = extent(cols)?;
            let table = tensor(rec, 1)?;
            // Every tile number a checked index can name fits a `u32` (the
            // cursor adds without checking: an index comes from L1).
            let tiles = bound
                .div_ceil(32)
                .checked_mul(table.ct)
                .and_then(|t| t.checked_add(table.first));
            if table.ct == 0
                || tiles.is_none()
                || j0.checked_add(cols).is_none_or(|end| end > table.ct)
            {
                return Err(super::error::RANGE);
            }
            // Every index is read, and checked, before any read is issued; each
            // is read again where it is used, which keeps the mover's image
            // free of a copy of them.
            // 32-bit arithmetic throughout: on this core a `u64` costs
            // instructions at every step.
            const DATA: u32 = TILE_DATA as u32;
            let word = |r: u32| idx_at + DATA + (r / 16 * 256 + r % 16) * 4;
            for r in 0..count {
                if index(word(r)) >= bound {
                    return Err(super::error::DOMAIN);
                }
            }
            // A tile row's face-row `h` of row `r`: two 64-byte reads a row a
            // tile column, each congruent to `TILE_DATA` mod 64 in every slot.
            let face_row = |r: u32, h: u32| DATA + ((r / 16 * 2 + h) * 16 + r % 16) * 64;
            let mut port = 0;
            for r in 0..count {
                let v = index(word(r));
                let mut row = table.cursor(v / 32, j0)?;
                for j in 0..cols {
                    let (ch, off) = row.next()?;
                    for half in 0..core::hint::black_box(2) {
                        let from = off + face_row(v % 32, half);
                        let to = stage_at + j * TILE_SLOT as u32 + face_row(r, half);
                        emit([op::READ, ch, port, from, to, 64, 0, 0])?;
                        port = if port + 1 == PORTS { 0 } else { port + 1 };
                    }
                }
            }
        }
        POKE_RUN => {
            // RISCV B's, as `Mover::permits` has it: it is not a write, so NC does not
            // reach the mailbox words through a record either.
            if !READS {
                return Err(super::error::DIRECTION);
            }
            let count = h[2] as usize;
            if count == 0 || count > POKE_RUN_MAX {
                return Err(super::error::RANGE);
            }
            // The words past the run are zero, so a record cannot carry a value
            // the expansion never writes and the card's own loop would.
            for k in count..POKE_RUN_MAX {
                if poke_run_word(rec, k) != 0 {
                    return Err(super::error::OP);
                }
            }
            for k in 0..count {
                let at = h[1].checked_add(4 * k as u32).ok_or(super::error::RANGE)?;
                // Every word through `op::POKE`, which bounds it to one role's mailbox.
                emit([op::POKE, at, poke_run_word(rec, k), 0, 0, 0, 0, 0])?;
            }
        }
        FILL_PAD => pad::<false>(rec, &mut emit)?,
        PAD_WRITE => pad::<true>(rec, &mut emit)?,
        _ => return Err(super::error::OP),
    }
    Ok(())
}

/// [`FILL_PAD`]'s edge tiles (`WRITE` false: each read into its slot and
/// filled) or [`PAD_WRITE`]'s (`WRITE` true: each written back from it).
fn pad<const WRITE: bool>(
    rec: &[[u32; 8]],
    emit: &mut impl FnMut([u32; 8]) -> Result<(), u32>,
) -> Result<(), u32> {
    let [_, value, first, count, rows, cols, stage, rt] = rec[0];
    let (count, rt) = (extent(count)?, extent(rt)?);
    let a = tensor(rec, 1)?;
    if a.ct == 0 || rows > 31 || cols > 31 {
        return Err(super::error::LENGTH);
    }
    let (ragged_r, ragged_c) = (rows != 0, cols != 0);
    let along_row = if ragged_r { a.ct } else { 0 };
    let down_col = if ragged_c {
        rt - u32::from(ragged_r)
    } else {
        0
    };
    if first + count > along_row + down_col {
        return Err(super::error::LENGTH);
    }
    for n in 0..count {
        let e = first + n;
        let (i, j) = if e < along_row {
            (rt - 1, e)
        } else {
            (e - along_row, a.ct - 1)
        };
        let (ch, off) = a.tile(i, j)?;
        let at = stage + n * TILE_SLOT as u32;
        if WRITE {
            emit([
                op::WRITE,
                ch,
                n % PORTS,
                off + TILE_DATA as u32,
                at + TILE_DATA as u32,
                4096,
                0,
                0,
            ])?;
        } else {
            emit([op::READ, ch, n % PORTS, off, at, TILE_SLOT as u32, 0, 0])?;
            let vr = if i == rt - 1 { rows } else { 0 };
            let vc = if j == a.ct - 1 { cols } else { 0 };
            emit([op::FILL, value, vr | vc << 8, at, 0, 0, 0, 0])?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    /// The POKEs a record expands to, or the error.
    fn poked(rec: &[[u32; 8]; 2]) -> Result<Vec<(u32, u32)>, u32> {
        let mut got = Vec::new();
        expand(rec, |e| {
            assert_eq!(e[0], op::POKE);
            assert_eq!(&e[3..], &[0; 5], "a POKE carries nothing past its value");
            got.push((e[1], e[2]));
            Ok(())
        })?;
        Ok(got)
    }

    /// The mailbox of role 1, whose words are the ones a trace pokes.
    fn mailbox_word(k: u32) -> u32 {
        (crate::mailbox::role::Mailbox::of(1).base() + 0x1C + 4 * k as u64) as u32
    }

    #[test]
    fn a_poke_run_expands_to_the_pokes_it_was_built_from() {
        for count in 1..=POKE_RUN_MAX {
            let values: Vec<u32> = (0..count as u32).map(|k| 0xA000_0000 + k).collect();
            let at = mailbox_word(0);
            let rec = poke_run(at, &values);
            assert_eq!(len(rec[0][0]), rec.len());
            let want: Vec<(u32, u32)> = values
                .iter()
                .enumerate()
                .map(|(k, &v)| (at + 4 * k as u32, v))
                .collect();
            assert_eq!(poked(&rec).unwrap(), want, "{count} words");
        }
    }

    #[test]
    fn a_poke_run_is_refused_for_a_count_or_a_stray_word_the_pokes_would_not_have() {
        let at = mailbox_word(0);
        let mut rec = poke_run(at, &[1, 2, 3]);
        assert!(poked(&rec).is_ok());
        // A value past the count: the card's own loop would not write it, so the
        // record is not one the host built.
        let mut stray = rec;
        stray[0][6] = 9;
        assert_eq!(poked(&stray), Err(super::super::error::OP));
        let mut stray = rec;
        stray[1][7] = 9;
        assert_eq!(poked(&stray), Err(super::super::error::OP));
        for count in [0, POKE_RUN_MAX as u32 + 1, u32::MAX] {
            rec[0][2] = count;
            assert_eq!(
                poked(&rec),
                Err(super::super::error::RANGE),
                "count {count}"
            );
        }
    }

    #[test]
    fn a_poke_run_outside_a_role_mailbox_is_refused_word_by_word() {
        let usable = 0xFF;
        let decode = |rec: &[[u32; 8]; 2]| -> Result<(), u32> {
            expand(rec, |e| super::super::Entry::decode(usable, e).map(|_| ()))
        };
        // The last word of role 2's mailbox: in, and one word past it: out.
        let last =
            (crate::mailbox::role::Mailbox::of(2).base() + crate::mailbox::MAILBOX_SIZE - 4) as u32;
        assert!(decode(&poke_run(last, &[7])).is_ok());
        assert!(decode(&poke_run(last + 4, &[7])).is_err());
        // A run that starts inside and runs past the end is refused, though its
        // first word is pokeable.
        assert!(decode(&poke_run(last - 4, &[7, 7])).is_ok());
        assert!(decode(&poke_run(last, &[7, 7])).is_err());
        // The mover's own mailbox and L1 are not pokeable at all.
        assert!(decode(&poke_run(0x2_0000, &[7, 7])).is_err());
    }

    #[test]
    fn only_the_reader_runs_a_poke_run() {
        let rec = poke_run(mailbox_word(0), &[1, 2]);
        assert_eq!(direction(POKE_RUN), None);
        assert!(super::super::Mover::B.permits(POKE_RUN));
        assert!(!super::super::Mover::NC.permits(POKE_RUN));
        assert_eq!(
            expand_writes(&rec, |_| Ok(())),
            Err(super::super::error::DIRECTION)
        );
        assert!(expand_reads(&rec, &mut |_| 0, |_| Ok(())).is_ok());
    }

    #[test]
    fn packed_gather_uses_halfword_slots_and_only_fills_edges() {
        let mut a = t(3, 3);
        a.first = 2;
        let b = t(3, 2);
        let [mut a0, a1] = a.encode();
        let [mut b0, b1] = b.encode();
        a0[4] = 37;
        a0[5] = 65;
        b0[4] = 65;
        b0[5] = 35;
        let head = [GATHER, GATHER_BF16 | 3 << 8, 0x20000, 0x60000, 0, 2, 0, 2];
        let mut got = Vec::new();
        expand_reads(
            &[head, a0, a1, b0, b1],
            |_| 0,
            |e| {
                got.push(e);
                Ok(())
            },
        )
        .unwrap();
        let mut next = 0;
        for (x, height, width, nr, nc, at) in
            [(a, 37u32, 65u32, 2, 3, 0x20000), (b, 65, 35, 3, 2, 0x60000)]
        {
            for i in 0..nr {
                for j in 0..nc {
                    let tile = x.first + i * x.ct + j;
                    let channel = (tile % 3) as usize;
                    let off = x.base[channel] + tile / 3 * 2112;
                    let to = at + (i * nc + j) * 4160;
                    assert_eq!(
                        got[next],
                        [
                            op::READ,
                            x.channels[channel] as u32,
                            (i + j) % PORTS,
                            off,
                            to,
                            2112,
                            0,
                            0
                        ]
                    );
                    next += 1;
                    let rows = (height - i * 32).min(32);
                    let cols = (width - j * 32).min(32);
                    if rows != 32 || cols != 32 {
                        assert_eq!(
                            got[next],
                            [
                                op::FILL_HALFWORDS,
                                0,
                                super::super::fill::param(rows, cols),
                                to,
                                0,
                                0,
                                0,
                                0
                            ]
                        );
                        next += 1;
                    }
                }
            }
        }
        assert_eq!(next, 20);
        assert_eq!(next, got.len());
        let mut bad = head;
        bad[1] |= 1; // A BF16 transpose cannot enter the F32 mover transform.
        assert!(expand_reads(&[bad, a0, a1, b0, b1], |_| 0, |_| Ok(())).is_err());
        a0[5] = 32; // Descriptor dimensions must agree with its grid.
        assert!(expand_reads(&[head, a0, a1, b0, b1], |_| 0, |_| Ok(())).is_err());
        assert!(expand_writes(&[head, a0, a1, b0, b1], |_| Ok(())).is_err());
    }

    /// Every tile a record names is the one `TensorRef::tile` gives: the
    /// cursor that steps along rows agrees with dividing per tile, across a
    /// channel wrap mid-row and a tensor that starts mid-grid.
    #[test]
    fn records_walk_rows_to_the_same_tiles_as_dividing_per_tile() {
        let mut a = t(3, 5);
        a.first = 2;
        let mut b = t(3, 7);
        b.first = 4;
        let [a0, a1] = a.encode();
        let [b0, b1] = b.encode();
        let (i0, rows, j0, cols, kt) = (1u32, 2u32, 2u32, 3u32, 4u32);
        for flags in [0u32, 1, 2, 3] {
            let head = [
                GATHER,
                flags | (kt << 8),
                0x2_0000,
                0x6_0000,
                i0,
                rows,
                j0,
                cols,
            ];
            let mut got = Vec::new();
            expand(&[head, a0, a1, b0, b1], |e| {
                got.push((e[1], e[3]));
                Ok(())
            })
            .unwrap();
            let mut want = Vec::new();
            for i in 0..rows {
                for kk in 0..kt {
                    let (r, c) = if flags & 1 != 0 {
                        (kk, i0 + i)
                    } else {
                        (i0 + i, kk)
                    };
                    want.push(a.tile(r, c).unwrap());
                }
            }
            for kk in 0..kt {
                for j in 0..cols {
                    let (r, c) = if flags & 2 != 0 {
                        (j0 + j, kk)
                    } else {
                        (kk, j0 + j)
                    };
                    want.push(b.tile(r, c).unwrap());
                }
            }
            assert_eq!(got, want, "flags {flags}");
        }
        // A scatter, and a plain run from mid-row.
        let head = [SCATTER, 0x2_0000, TILE_SLOT as u32, i0, rows, j0, cols, 0];
        let mut got = Vec::new();
        expand(&[head, b0, b1], |e| {
            got.push((e[1], e[3] - TILE_DATA as u32));
            Ok(())
        })
        .unwrap();
        let want: Vec<_> = (0..rows)
            .flat_map(|i| (0..cols).map(move |j| (i0 + i, j0 + j)))
            .map(|(i, j)| b.tile(i, j).unwrap())
            .collect();
        assert_eq!(got, want, "scatter");
        let head = [READ_RUN, 6, 11, 0x2_0000, 0, 0, 0, 0];
        let mut got = Vec::new();
        expand(&[head, a0, a1], |e| {
            got.push((e[1], e[3]));
            Ok(())
        })
        .unwrap();
        let want: Vec<_> = (6..17).map(|n| a.tile(n / 5, n % 5).unwrap()).collect();
        assert_eq!(got, want, "run");
        // Transposed, over a 3-wide grid: tile `(i, j)` is `(j, i)`, each
        // read through `READ_TRANSPOSED`.
        let head = [READ_RUN, 1, 5, 0x2_0000, 8, 3, 0, 0];
        let mut got = Vec::new();
        expand(&[head, a0, a1], |e| {
            assert_eq!(e[0], op::READ_TRANSPOSED);
            got.push((e[1], e[3]));
            Ok(())
        })
        .unwrap();
        let want: Vec<_> = (1..6).map(|n| a.tile(n % 3, n / 3).unwrap()).collect();
        assert_eq!(got, want, "transposed run");
        // Not with a broadcast or column-major order.
        for flags in [9, 10, 12] {
            let head = [READ_RUN, 0, 1, 0x2_0000, flags, 3, 2, 0];
            assert!(
                expand(&[head, a0, a1], |_| Ok(())).is_err(),
                "flags {flags}"
            );
        }
    }

    const IDX: u32 = 0x2_0000;
    const STAGE: u32 = 0x4_0000;

    /// An L1 holding `values` as the index tile at `IDX`: the reader a record
    /// is expanded with, which refuses any address but an index word's.
    fn l1_of(values: &[u32]) -> impl FnMut(u32) -> u32 {
        let mut words = std::collections::HashMap::new();
        for (r, &v) in values.iter().enumerate() {
            words.insert(IDX + 16 + crate::dm::face_index(0, r) as u32 * 4, v);
        }
        move |at| *words.get(&at).unwrap_or_else(|| panic!("read of {at:#x}"))
    }

    /// A `SELECT_ROWS` of `count` indices from `bound` rows, tile columns
    /// `j0..j0 + cols` of `table`.
    fn select(table: TensorRef, bound: u32, count: u32, j0: u32, cols: u32) -> [[u32; 8]; 3] {
        let [m0, m1] = table.encode();
        [[SELECT_ROWS, IDX, count, bound, j0, cols, STAGE, 0], m0, m1]
    }

    /// Each index's two face-rows of every tile column are read to the output
    /// row's, from the tile `TensorRef::tile` names, in an order and by an
    /// offset the face layout (`face_index`) gives independently of `expand`'s
    /// own arithmetic -- across a channel wrap mid-row, a view that starts
    /// mid-grid, the first and last rows, and a row either side of a tile.
    #[test]
    fn select_rows_reads_the_face_rows_of_each_index() {
        let mut table = t(3, 5);
        table.first = 2;
        let bound = 100; // four tile rows, the last ragged
        let values = [
            0u32, 99, 31, 32, 33, 5, 5, 64, 17, 16, 63, 95, 1, 2, 3, 4, 5, 6,
        ];
        let (j0, cols) = (1u32, 3u32);
        let rec = select(table, bound, values.len() as u32, j0, cols);
        let mut got = Vec::new();
        expand_reads(&rec, l1_of(&values), |e| {
            got.push(e);
            Ok(())
        })
        .unwrap();
        let mut want = Vec::new();
        for (r, &v) in values.iter().enumerate() {
            for j in 0..cols {
                let (ch, off) = table.tile(v / 32, j0 + j).unwrap();
                for h in 0..2usize {
                    let from = off as u64
                        + TILE_DATA
                        + 4 * crate::dm::face_index((v % 32) as usize, 16 * h) as u64;
                    let to = STAGE as u64
                        + j as u64 * TILE_SLOT
                        + TILE_DATA
                        + 4 * crate::dm::face_index(r, 16 * h) as u64;
                    let n = want.len() as u32;
                    want.push([op::READ, ch, n % PORTS, from as u32, to as u32, 64, 0, 0]);
                }
            }
        }
        assert_eq!(got, want);
        // What a GDDR read needs (`Descriptor::decode`): source and
        // destination congruent mod 64, whichever row either is.
        for e in &got {
            assert_eq!(e[3] % 64, e[4] % 64);
        }
        assert_eq!(got.len(), values.len() * cols as usize * 2);
    }

    /// An index at or past `bound`, or a negative one, is `DOMAIN` before a
    /// single read is issued -- whichever position it is in -- and the last
    /// row is the last that passes. The host's `expand` reads every index as 0
    /// and still checks the geometry.
    #[test]
    fn select_rows_refuses_an_index_out_of_range_before_any_read() {
        let table = t(4, 3);
        for (values, ok) in [
            ([0u32, 99, 5], true),
            ([0, 100, 5], false),
            ([0, 1, 100], false),
            ([u32::MAX, 1, 2], false),
            ([1, 2, 0x8000_0000], false),
        ] {
            let rec = select(table, 100, values.len() as u32, 0, 3);
            let mut issued = 0;
            let got = expand_reads(&rec, l1_of(&values), |_| {
                issued += 1;
                Ok(())
            });
            if ok {
                assert_eq!((got, issued), (Ok(()), 3 * 3 * 2), "{values:?}");
            } else {
                assert_eq!(
                    (got, issued),
                    (Err(crate::dm::error::DOMAIN), 0),
                    "{values:?}"
                );
            }
        }
        let rec = select(table, 100, 4, 0, 3);
        let mut issued = 0;
        expand(&rec, |_| {
            issued += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(issued, 4 * 3 * 2);
    }

    #[test]
    fn select_rows_refuses_what_it_should() {
        use crate::dm::error::{DIRECTION, LENGTH, RANGE};
        let table = t(3, 4);
        let ok = select(table, 64, 3, 1, 3);
        let run = |rec: &[[u32; 8]]| expand_reads(rec, |_| 0, |_| Ok(()));
        assert_eq!(run(&ok), Ok(()));
        // The writer's code does not hold it.
        assert_eq!(expand_writes(&ok, |_| Ok(())), Err(DIRECTION));
        assert_eq!(direction(SELECT_ROWS), Some(Direction::Read));
        assert_eq!(len(SELECT_ROWS), 3);
        for (word, value, want) in [
            (2, 0, LENGTH),       // no indices
            (2, 33, LENGTH),      // more than a tile of them
            (3, 0, LENGTH),       // no rows
            (3, 1 << 31, LENGTH), // more than an I32 names
            (5, 0, LENGTH),       // no tile columns
            (4, 2, RANGE),        // columns 2..5 of 4
            (5, 4, RANGE),        // columns 1..5 of 4
            (7, 1, RANGE),        // packed BF16 is not implemented
            (7, 2, RANGE),        // an unknown flag
        ] {
            let mut bad = ok;
            bad[0][word] = value;
            assert_eq!(run(&bad), Err(want), "word {word} = {value:#x}");
        }
        // A table so tall and wide that a tile number could pass a `u32`.
        let wide = TensorRef {
            ct: 1 << 16,
            ..t(3, 1)
        };
        let mut huge = select(wide, i32::MAX as u32, 3, 0, 3);
        assert_eq!(run(&huge), Err(RANGE));
        huge[0][3] = 1 << 10; // 32 tile rows of 65,536 tiles do fit
        assert_eq!(run(&huge), Ok(()));
        // No table: its channel count is zero.
        let mut none = ok;
        none[1][0] = 0;
        assert_eq!(run(&none), Err(RANGE));
    }

    fn t(n: u8, ct: u32) -> TensorRef {
        let mut r = TensorRef {
            n,
            ct,
            ..Default::default()
        };
        for i in 0..n as usize {
            r.channels[i] = i as u8;
            r.base[i] = 0x1000 * (i as u32 + 1);
        }
        r
    }

    #[test]
    fn a_tensor_ref_round_trips() {
        let mut r = t(8, 25);
        r.first = 7;
        r.channels = [7, 6, 5, 4, 3, 2, 1, 0];
        assert_eq!(TensorRef::decode(r.encode()), Ok(r));
        let mut bad = r.encode();
        bad[0][0] = 9;
        assert!(TensorRef::decode(bad).is_err());
    }

    #[test]
    fn tiles_interleave_over_the_channels() {
        let r = t(3, 4);
        // Tile 5 (row 1, column 1): channel 5 % 3 = 2, slot 5 / 3 = 1.
        assert_eq!(r.tile(1, 1), Ok((2, 0x3000 + TILE_SLOT as u32)));
        assert!(t(0, 4).tile(0, 0).is_err());
    }

    #[test]
    fn records_have_their_lengths_and_refuse_what_they_should() {
        let mut rec = [[0u32; 8]; 3];
        rec[0] = [SCATTER, 0x8_0010, TILE_SLOT as u32, 0, 1, 0, 2, 0];
        [rec[1], rec[2]] = t(1, 2).encode();
        let mut got = [[0u32; 8]; 4];
        let mut n = 0;
        expand(&rec, |e| {
            got[n] = e;
            n += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(n, 2);
        assert_eq!(got[1][4], 0x8_0010 + TILE_SLOT as u32);
        // A zero extent, a wrong length, and a plain entry are all refused.
        let mut zero = rec;
        zero[0][4] = 0;
        assert_eq!(expand(&zero, |_| Ok(())), Err(crate::dm::error::LENGTH));
        assert_eq!(expand(&rec[..2], |_| Ok(())), Err(crate::dm::error::OP));
        assert_eq!(
            expand(&[[op::READ, 0, 0, 0, 0, 0, 0, 0]], |_| Ok(())),
            Err(crate::dm::error::OP)
        );
        // `emit`'s error stops the expansion.
        assert_eq!(expand(&rec, |_| Err(42)), Err(42));
    }
}
