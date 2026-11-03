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
/// `A` is read transposed, bit 1 `B`, bits 8..=21 `kt` ([`gather_kt`]), bits 22-24 `A`'s format
/// ([`gather_a_format`]), bit 25 [`GATHER_A_RESIDENT`], bit 26 [`GATHER_B_UNPACK_TRANSPOSE`], bit
/// 27 [`GATHER_PREFETCH`], bit 28 [`GATHER_STREAM`], bits 29-31 `B`'s format ([`gather_b_format`]).
/// Bit 2 [`GATHER_B_RESIDENT`] (once packed BF16, retired). A packed operand's tiles
/// are slots of its format in GDDR and L1 alike: `B`'s ([`gather_b_format`]) at
/// `b_at + (kk * cols + j) * slot`, `A`'s ([`gather_a_format`]) at `a_at + (i * kt + kk) * slot`.
/// Bits 3-7 ([`gather_rows`], F32, `A` untransposed): `r`, 1..=16, the rows of
/// `A`'s last tile row of the block that the product uses, which is then read
/// as its datums up to row `r` of face 1 (one `1024 + 64 r`-byte read a tile:
/// face 0 whole, face 1's first `r` rows) instead of whole tiles. Its other
/// rows in L1 keep whatever they held, and face 0's rows past `r` hold `A`'s
/// padding; either reaches only the same rows of the output -- its padding,
/// then undefined. 0: whole tiles.
pub const GATHER: u32 = 0x10;
/// [`GATHER`]'s flag for a block whose `B` is already in its slots, [`GATHER_A_RESIDENT`]'s
/// mirror: only `A` is read, and `B`'s slots keep what they hold (a strip copied there from
/// another unit's L1 first, `tt_kernels::tile_op::reuse`). With [`GATHER_A_RESIDENT`] the record
/// reads nothing. Not with [`GATHER_PREFETCH`] or [`GATHER_STREAM`], whose reads are `B`'s.
/// Bit 2 was packed BF16 with its ragged lanes filled on the card, retired when any pair of edge
/// formats became one product (`docs/completed-plans/dataflow-compiler.md` F1d).
pub const GATHER_B_RESIDENT: u32 = 1 << 2;
/// [`GATHER`]'s flag for a block whose `A` is already in its slots: the same
/// unit's previous gather of the op read the same `A` block to the same `a_at`,
/// and nothing since wrote there (a product's later rounds on a unit, under one
/// staging). Only `B` is read; `A`'s slots keep what they hold.
pub const GATHER_A_RESIDENT: u32 = 1 << 25;
/// [`GATHER`]'s flag for a transposed `B` (bit 1) whose tiles the kernel's unpacker transposes:
/// tile `(kk, j)` of `op(B)` is `B`'s tile `(j, kk)` read as it lies (`op::READ`, not
/// `op::READ_TRANSPOSED`), so the mover moves whole tiles and transposes nothing. Refused
/// without bit 1, and with packed BF16.
pub const GATHER_B_UNPACK_TRANSPOSE: u32 = 1 << 26;
/// [`GATHER`]'s flag for a prefetch: RISCV B issues the record's reads on
/// [`super::TXN_PREFETCH`], not its own ID, so nothing but an [`super::op::WAIT`] with
/// [`super::WAIT_PREFETCH`] waits for them, and they stay in flight under the ops queued after
/// them on the unit. The reads are the record's as without it.
pub const GATHER_PREFETCH: u32 = 1 << 27;
/// [`GATHER`]'s flag for a weight stream: one of the many units of a product that streams its
/// weights reads its share. RISCV B issues the record's reads at [`super::STREAM_CAP`] in flight,
/// not its own cap: a low cap evens out the NoC's service between the units reading together
/// (`docs/plans/decode-to-bandwidth.md` Phase 8, "The stream's rate"). The reads are the
/// record's as without it. Not with [`GATHER_PREFETCH`], whose reads have a cap of their own.
pub const GATHER_STREAM: u32 = 1 << 28;
/// [`GATHER`]'s bits 29-31: the format of a packed `B` beside an FP32 `A`, by code
/// ([`GATHER_FORMATS`]). Code 0 is an FP32 `B` in [`TILE_SLOT`]s. Any other code names a
/// weight stored in fewer bytes: `B`'s tiles are images of that format in GDDR slots of
/// [`crate::format::TileFormat::slot_bytes`], each read whole into an L1 slot of the same size
/// (`b_at + (kk * cols + j) * slot`: more tiles in a ring than an FP32 `B`'s,
/// `docs/plans/bf16-weight-storage.md` A5b), where the kernel unpacks it in its own format
/// (`tt_kernels::matmul::SrcRoute::Tf32WithB`). A packed `B` is untransposed only (not bit 1,
/// not [`GATHER_B_UNPACK_TRANSPOSE`]); with every other flag, and in a [`PrefetchCursor`], as an FP32 `B` is. Its tensor's padding is
/// the writer's (zero for a stored weight): nothing is filled.
pub const GATHER_B_FORMAT_SHIFT: u32 = 29;
/// [`GATHER`]'s bits that hold `B`'s format code.
pub const GATHER_B_FORMAT_MASK: u32 = 0b111 << GATHER_B_FORMAT_SHIFT;
/// [`GATHER`]'s bits 22-24: the format of a packed `A`, by code ([`GATHER_FORMATS`]), as bits
/// 29-31 name `B`'s. Code 0 is an FP32 `A` in [`TILE_SLOT`]s. Any other code names an operand
/// stored in fewer bytes: `A`'s tiles are images of that format in GDDR slots of
/// [`crate::format::TileFormat::slot_bytes`], each read whole into an L1 slot of the same size
/// (`a_at + (i * kt + kk) * slot`), where the kernel unpacks it in its own format
/// (`docs/learnings/dataflow-compiler.md` §4: one product for every edge format). A packed `A` is
/// untransposed only (not bit 0), read as whole tiles (not [`gather_rows`]), and not with packed
/// BF16 (bit 2). Its tensor's padding is the writer's: nothing is filled.
pub const GATHER_A_FORMAT_SHIFT: u32 = 22;
/// [`GATHER`]'s bits that hold `A`'s format code.
pub const GATHER_A_FORMAT_MASK: u32 = 0b111 << GATHER_A_FORMAT_SHIFT;
/// The operand formats a [`GATHER`] names, by code, for `A` and `B` alike
/// (`docs/learnings/dataflow-compiler.md` §4: a format is an edge's, and its slot is its
/// descriptor's).
pub const GATHER_FORMATS: [crate::tile::L1Format; 5] = {
    use crate::tile::L1Format as F;
    [F::Fp32, F::Bf16, F::Bfp8, F::Bfp4, F::Bfp2]
};
/// [`GATHER_FORMATS`]' code of `format`, or `None` for a format a gather cannot read.
const fn gather_format_code(format: crate::tile::L1Format) -> Option<u32> {
    let mut code = 0;
    while code < GATHER_FORMATS.len() {
        if GATHER_FORMATS[code] as u32 == format as u32 {
            return Some(code as u32);
        }
        code += 1;
    }
    None
}
/// The format [`GATHER_FORMATS`] gives the code in `flags` under `mask` at `shift`, or a refusal
/// for a code no format has.
const fn gather_tile_format(
    flags: u32,
    mask: u32,
    shift: u32,
) -> Result<crate::format::TileFormat, u32> {
    let code = ((flags & mask) >> shift) as usize;
    if code >= GATHER_FORMATS.len() {
        return Err(super::error::RANGE);
    }
    Ok(crate::format::TileFormat::new(GATHER_FORMATS[code]))
}
/// [`GATHER`]'s flags for a `B` of `format`, or `None` for a format a gather cannot read.
pub const fn gather_b_format(format: crate::tile::L1Format) -> Option<u32> {
    match gather_format_code(format) {
        Some(code) => Some(code << GATHER_B_FORMAT_SHIFT),
        None => None,
    }
}
/// [`GATHER`]'s flags for an `A` of `format`, or `None` for a format a gather cannot read.
pub const fn gather_a_format(format: crate::tile::L1Format) -> Option<u32> {
    match gather_format_code(format) {
        Some(code) => Some(code << GATHER_A_FORMAT_SHIFT),
        None => None,
    }
}
/// The format of [`GATHER`]'s `B` under `flags`, or a refusal for a code no format has.
pub const fn gather_b_tile_format(flags: u32) -> Result<crate::format::TileFormat, u32> {
    gather_tile_format(flags, GATHER_B_FORMAT_MASK, GATHER_B_FORMAT_SHIFT)
}
/// The format of [`GATHER`]'s `A` under `flags`, or a refusal for a code no format has.
pub const fn gather_a_tile_format(flags: u32) -> Result<crate::format::TileFormat, u32> {
    gather_tile_format(flags, GATHER_A_FORMAT_MASK, GATHER_A_FORMAT_SHIFT)
}
/// Whether [`GATHER`]'s `B` is packed (any format but FP32).
pub const fn gather_b_packed(flags: u32) -> bool {
    flags & GATHER_B_FORMAT_MASK != 0
}
/// Whether [`GATHER`]'s `A` is packed (any format but FP32).
pub const fn gather_a_packed(flags: u32) -> bool {
    flags & GATHER_A_FORMAT_MASK != 0
}
/// [`GATHER`]'s flags for a `B` stored BF16 (a weight in half the bytes) beside an FP32 `A`:
/// [`gather_b_format`] of BF16, code 1.
pub const GATHER_B_BF16: u32 = 1 << GATHER_B_FORMAT_SHIFT;
const _: () = assert!(matches!(
    gather_b_format(crate::tile::L1Format::Bf16),
    Some(GATHER_B_BF16)
));
/// [`GATHER`]'s flag bits that are not `kt`.
const GATHER_FLAG_BITS: u32 = GATHER_A_RESIDENT
    | GATHER_B_UNPACK_TRANSPOSE
    | GATHER_PREFETCH
    | GATHER_STREAM
    | GATHER_B_FORMAT_MASK
    | GATHER_A_FORMAT_MASK
    | 0xff;

/// `B`'s slot under `flags`, in GDDR and L1 alike, and each read's length: its format's
/// ([`gather_b_tile_format`]), [`TILE_SLOT`] for an FP32 `B`.
const fn gather_b_slot(flags: u32) -> Result<u64, u32> {
    match gather_b_tile_format(flags) {
        Ok(format) => Ok(format.slot_bytes()),
        Err(e) => Err(e),
    }
}

/// `A`'s slot under `flags`, as [`gather_b_slot`] is `B`'s; and a refusal for a packed `A` with
/// what only an FP32 `A` takes: a transposed read (bit 0), thin rows ([`gather_rows`]).
const fn gather_a_slot(flags: u32) -> Result<u64, u32> {
    if gather_a_packed(flags) && (flags & 1 != 0 || (flags >> GATHER_ROWS_SHIFT) & 0x1f != 0) {
        return Err(super::error::RANGE);
    }
    match gather_a_tile_format(flags) {
        Ok(format) => Ok(format.slot_bytes()),
        Err(e) => Err(e),
    }
}

/// A prefetch [`GATHER`]'s reads one at a time: a record with [`GATHER_PREFETCH`] and
/// [`GATHER_A_RESIDENT`] reads only `B`, tile by tile, and a mover can issue them while it would
/// otherwise wait (`docs/plans/decode-to-bandwidth.md` Phase 8, the prefetch pump). Its entries,
/// in order, are exactly what [`expand_reads`] gives the record, and it refuses what that
/// refuses, as soon as [`PrefetchCursor::new`] or the first [`PrefetchCursor::next`] would.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefetchCursor {
    b: TensorRef,
    b_at: u32,
    kt: u32,
    cols: u32,
    j0: u32,
    /// `B` read as it lies, transposed by the kernel ([`GATHER_B_UNPACK_TRANSPOSE`]).
    unpacked: bool,
    /// `B`'s slot in GDDR and L1, and each read's length ([`gather_b_format`]).
    slot: u32,
    /// A one-channel `B`, read in large pieces ([`BSpan`]), as [`expand_reads`] reads it.
    span: Option<BSpan>,
    next: u32,
}

impl PrefetchCursor {
    /// The cursor of `rec`, or `None` for a record that is not a resident-`A` prefetch
    /// [`GATHER`] (it expands as any record does). The checks are [`expand_reads`]'s, in its
    /// order, up to its first read of `B`.
    pub fn new(rec: &[[u32; 8]]) -> Result<Option<Self>, u32> {
        let h = rec[0];
        if h[0] != GATHER
            || h[1] & GATHER_PREFETCH == 0
            || h[1] & GATHER_A_RESIDENT == 0
            || rec.len() != len(GATHER)
        {
            return Ok(None);
        }
        let [_, flags, _, b_at, _, rows, j0, cols] = h;
        let (a_t, b_t, kt) = (flags & 1 != 0, flags & 2 != 0, extent(gather_kt(flags))?);
        if flags & GATHER_B_RESIDENT != 0 {
            return Err(super::error::RANGE);
        }
        let unpacked = flags & GATHER_B_UNPACK_TRANSPOSE != 0;
        if unpacked && !b_t {
            return Err(super::error::RANGE);
        }
        let slot = gather_b_slot(flags)? as u32;
        if gather_b_packed(flags) && b_t {
            return Err(super::error::RANGE);
        }
        gather_a_slot(flags)?;
        if (b_t && !unpacked) || flags & GATHER_STREAM != 0 {
            return Err(super::error::RANGE);
        }
        let (_, cols) = (extent(rows)?, extent(cols)?);
        let (_, b) = (tensor(rec, 1)?, tensor(rec, 3)?);
        let thin = (flags >> GATHER_ROWS_SHIFT) & 0x1f;
        if thin > 16 || (thin != 0 && a_t) {
            return Err(super::error::RANGE);
        }
        let span = match b_t {
            false if laid_for_reader(rec, 3) => {
                BSpan::of(&b, b_at, kt, j0, cols, slot, BSpan::piece(flags, slot))?
            }
            false => None,
            true => None,
        };
        Ok(Some(PrefetchCursor {
            b,
            b_at,
            kt,
            cols,
            j0,
            unpacked,
            slot,
            span,
            next: 0,
        }))
    }

    /// Reads in all.
    fn reads(&self) -> u32 {
        match &self.span {
            Some(s) => s.count(),
            None => self.kt * self.cols,
        }
    }

    /// Reads not yet given.
    pub fn left(&self) -> u32 {
        self.reads() - self.next
    }
}

/// Each read in turn (a refusal where [`expand_reads`] would refuse), then `None`.
impl Iterator for PrefetchCursor {
    type Item = Result<[u32; 8], u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.reads() {
            return None;
        }
        if let Some(span) = &self.span {
            let read = span.read(self.next);
            self.next += 1;
            return Some(Ok(read));
        }
        let (kk, j) = (self.next / self.cols, self.next % self.cols);
        self.next += 1;
        let at = if self.unpacked {
            self.b.tile_in(self.j0 + j, kk, self.slot as u64)
        } else {
            self.b.tile_in(kk, self.j0 + j, self.slot as u64)
        };
        Some(at.map(|(ch, off)| {
            [
                super::op::READ,
                ch,
                (kk + self.j0 + j) % PORTS,
                off,
                self.b_at + (kk * self.cols + j) * self.slot,
                self.slot,
                0,
                0,
            ]
        }))
    }
}

/// [`GATHER`]'s `kt`, from its flags: bits 8..=21, at most 16,383 tiles.
pub const fn gather_kt(flags: u32) -> u32 {
    (flags & !GATHER_FLAG_BITS) >> 8
}
/// Where [`GATHER`]'s `r` sits in its flags.
pub const GATHER_ROWS_SHIFT: u32 = 3;

/// [`GATHER`]'s flag bits for `A`'s last tile row read as its first `r` rows
/// (`0`: whole tiles).
pub const fn gather_rows(r: u32) -> u32 {
    r << GATHER_ROWS_SHIFT
}
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
/// 0-2); bit 4, with bit 1: tile `(i, 0)` read as it lies ([`op::READ`]), its
/// column 0 left for the kernel to broadcast ([`READ_RUN_COL_AS_IT_LIES`]).
pub const READ_RUN: u32 = 0x15;
/// [`READ_RUN`]'s flags for a `[m, 1]` tensor met by a `[m, n]` one whose kernel
/// broadcasts column 0 itself: tile `(i, 0)` for every `(i, j)`, read plainly.
pub const READ_RUN_COL_AS_IT_LIES: u32 = 2 | 16;
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
/// Whole rows written over a tensor's rows chosen by indices already in L1,
/// L1 -> GDDR: [`SELECT_ROWS`] the other way, on the writer. `[SCATTER_ROWS,
/// idx_at, count, bound, j0, cols, stage_at, flags]` + `X`. For each of the
/// first `count` (1..=32) indices `v` (index `r` at `(0, r)` of the slot at
/// `idx_at`, or at `(r, 0)` with [`SCATTER_ROWS_COLUMN`]: an `[n, 1]` index
/// tensor's tile), row `r` of the slots `j - j0` from `stage_at` goes to row
/// `v` of `X` (a tensor of `bound` rows), for tile columns `j0..j0 + cols` --
/// two 64-byte face-row writes a tile column, as `tensor::write_rows` makes
/// them, except that the destination is found here, from the index the card
/// holds. Every other datum of `X` is left as it was.
///
/// An index at or past `bound`, or a negative one (read as `u32`), is
/// [`super::error::DOMAIN`] before any write is issued. When two of the
/// record's indices name one row, **the later one's row is written** and the
/// earlier one's is not: writes to one address through different ports are not
/// ordered, so the record, not the NoC, decides. Between records the writer's
/// credit pop waits for its writes (`Action::Pop`), so a later batch's row
/// lands after an earlier batch's.
///
/// `flags`: [`SCATTER_ROWS_COLUMN`], and a range of the indices to write
/// ([`scatter_rows_range`]): only the rows `r` in it are written, while a
/// repeated row is still written only from the last of all `count` indices,
/// so records that split one batch's indices between them write each address
/// once, from the same index one record would (a row's writer is the record
/// whose range holds its last index). Bit 0 (16-bit datums) is reserved and,
/// as any other bit, [`super::error::RANGE`]. As with [`SELECT_ROWS`], the
/// slots are not checked against the data arena here
/// (`tt_kernels::index::scatter_rows_jobs` places them from its plan); each
/// write's L1 range is checked as any write's is. The host's [`expand`] reads
/// every index as `0`, so it checks the geometry only.
pub const SCATTER_ROWS: u32 = 0x1A;
/// [`SCATTER_ROWS`]'s indices are a column, `(r, 0)` of the slot: an `[n, 1]`
/// index tensor, read where it is with no repack.
pub const SCATTER_ROWS_COLUMN: u32 = 1 << 1;
/// The bits of a [`SCATTER_ROWS`] index range ([`scatter_rows_range`]).
const SCATTER_ROWS_RANGE: u32 = 0x3f << 8 | 0x3f << 16;

/// [`SCATTER_ROWS`]'s flags for writing only indices `rows` (within `0..32`,
/// not empty) of its `count`: the first at bits 8..14, the end at bits 16..22.
/// No range (both zero) is every index.
pub const fn scatter_rows_range(rows: core::ops::Range<u32>) -> u32 {
    rows.start << 8 | rows.end << 16
}

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
        SCATTER | WRITE_RUN | PAD_WRITE | SCATTER_ROWS => Some(Direction::Write),
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
        SCATTER | FILL_PAD | PAD_WRITE | READ_RUN | WRITE_RUN | SELECT_ROWS | SCATTER_ROWS => 3,
        POKE_RUN => 2,
        _ => 1,
    }
}

/// Where record `op`'s tensor references start, counted from its header: two entries each
/// ([`TensorRef::encode`]). None for a plain entry or a [`POKE_RUN`], whose second entry is data.
pub const fn references(op: u32) -> &'static [usize] {
    match op {
        GATHER => &[1, 3],
        SCATTER | FILL_PAD | PAD_WRITE | READ_RUN | WRITE_RUN | SELECT_ROWS | SCATTER_ROWS => &[1],
        _ => &[],
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
        self.tile_in(i, j, TILE_SLOT)
    }

    /// [`TensorRef::tile`] in slots of `slot` bytes.
    fn tile_in(&self, i: u32, j: u32, slot: u64) -> Result<(u32, u32), u32> {
        if self.n == 0 {
            return Err(super::error::RANGE);
        }
        let t = self.first + i * self.ct + j;
        let (s, c) = (t / self.n as u32, t % self.n as u32);
        let offset = (self.base[c as usize] as u64) + s as u64 * slot;
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

/// Word 4 of a tensor reference's first entry ([`TensorRef::encode`] leaves it 0, and
/// [`TensorRef::decode`] does not read it): a one-channel `B` laid out for the record that reads it,
/// its tiles in the order the record reads them (X9, `tt_kernels`' weight layout). Only such a `B`
/// is read in large pieces ([`BSpan`]); any other, a one-channel chip's tensors included, is read a
/// tile at a time.
pub const REF_LAID_FOR_READER: u32 = 1;

/// Whether the reference at `rec[at]` carries [`REF_LAID_FOR_READER`].
fn laid_for_reader(rec: &[[u32; 8]], at: usize) -> bool {
    rec[at][4] & REF_LAID_FOR_READER != 0
}

/// A [`GATHER`]'s `B` in one channel (`n == 1`) and marked [`REF_LAID_FOR_READER`]
/// (`docs/plans/decode-bytes-in-flight.md`, X9's layout: a weight stored for its reader), read as
/// it lies: row `kk`'s `cols` tiles from `j0`
/// are consecutive slots in GDDR, as their slots `b_at + (kk * cols + j) * slot` are in L1, so
/// each row is one run, and with `ct == cols` the rows meet and the block is one run. Each run is
/// read in pieces of [`crate::noc::niu::MAX_REQUEST_BYTES`] ([`BSpan::piece`]; one request each,
/// so the in-flight caps count what they did), the port rotating as a tile's would: the same bytes
/// into the same slots as a read a tile, in a few large reads (X10). A read ahead
/// ([`GATHER_PREFETCH`]) too, under its own cap, which then holds more bytes in flight: the ops it
/// runs under read GDDR behind it (`docs/plans/decode-bytes-in-flight.md` Phase B). Any other `B`
/// is read a tile at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BSpan {
    ch: u32,
    /// GDDR offset and L1 address of the first run's first byte.
    off: u64,
    l1: u32,
    /// Runs, each `bytes` long; GDDR advances `stride` from one run to the next, L1 `bytes`.
    runs: u32,
    bytes: u32,
    stride: u64,
    /// Pieces a run, and each piece's bytes (the last of a run may be shorter).
    pieces: u32,
    piece: u32,
}

impl BSpan {
    /// The piece of a record with `flags`: one request, for a stream and a read ahead alike.
    /// (A read ahead a slot a piece was measured too: its small ops were faster, its products
    /// slower by more, `docs/plans/decode-bytes-in-flight.md` Phase B.)
    fn piece(_flags: u32, _slot: u32) -> u32 {
        crate::noc::niu::MAX_REQUEST_BYTES
    }

    /// The span of a one-channel, untransposed `b`; `None` for any other.
    fn of(
        b: &TensorRef,
        b_at: u32,
        kt: u32,
        j0: u32,
        cols: u32,
        slot: u32,
        piece: u32,
    ) -> Result<Option<Self>, u32> {
        if b.n != 1 || piece == 0 {
            return Ok(None);
        }
        let (ch, off) = b.tile_in(0, j0, slot as u64)?;
        let row = cols.checked_mul(slot).ok_or(super::error::RANGE)?;
        let whole = b.ct == cols;
        let (runs, bytes) = if whole {
            (1, kt.checked_mul(row).ok_or(super::error::RANGE)?)
        } else {
            (kt, row)
        };
        // The last byte read must be a slot the tile-by-tile reads would name.
        let end = off as u64 + ((runs - 1) as u64) * (b.ct as u64 * slot as u64) + bytes as u64;
        if end > u32::MAX as u64 + 1 {
            return Err(super::error::RANGE);
        }
        Ok(Some(BSpan {
            ch,
            off: off as u64,
            l1: b_at,
            runs,
            bytes,
            stride: b.ct as u64 * slot as u64,
            pieces: bytes.div_ceil(piece),
            piece,
        }))
    }

    /// Reads in all.
    fn count(&self) -> u32 {
        self.runs * self.pieces
    }

    /// Read `n` of [`BSpan::count`].
    fn read(&self, n: u32) -> [u32; 8] {
        let (run, piece) = (n / self.pieces, n % self.pieces);
        let at = piece * self.piece;
        [
            op::READ,
            self.ch,
            n % PORTS,
            (self.off + run as u64 * self.stride + at as u64) as u32,
            self.l1 + run * self.bytes + at,
            (self.bytes - at).min(self.piece),
            0,
            0,
        ]
    }
}

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
/// or [`SCATTER_ROWS`] index reads as `0` (see there).
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
/// `index` reads the L1 word at an address, for [`SCATTER_ROWS`]; the caller
/// has made the indices visible before it calls.
pub fn expand_writes(
    rec: &[[u32; 8]],
    mut index: impl FnMut(u32) -> u32,
    emit: impl FnMut([u32; 8]) -> Result<(), u32>,
) -> Result<(), u32> {
    expand_for::<false, true>(rec, &mut index, emit)
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
            let (a_t, b_t, kt) = (flags & 1 != 0, flags & 2 != 0, extent(gather_kt(flags))?);
            let resident = flags & GATHER_A_RESIDENT != 0;
            let b_resident = flags & GATHER_B_RESIDENT != 0;
            // A prefetch or a stream reads `B` with a cap of its own: it has `B` to read.
            if b_resident && flags & (GATHER_PREFETCH | GATHER_STREAM) != 0 {
                return Err(super::error::RANGE);
            }
            let b_unpacked = flags & GATHER_B_UNPACK_TRANSPOSE != 0;
            if b_unpacked && !b_t {
                return Err(super::error::RANGE);
            }
            let b_slot = gather_b_slot(flags)?;
            if gather_b_packed(flags) && b_t {
                return Err(super::error::RANGE);
            }
            let a_slot = gather_a_slot(flags)?;
            // A prefetch's reads stay in flight under the entries after it: plain reads only,
            // never a transposing move (through the scratch, after draining the mover's own ID).
            if flags & GATHER_PREFETCH != 0
                && ((a_t && !resident) || (b_t && !b_unpacked) || flags & GATHER_STREAM != 0)
            {
                return Err(super::error::RANGE);
            }
            let (rows, cols) = (extent(rows)?, extent(cols)?);
            let (a, b) = (tensor(rec, 1)?, tensor(rec, 3)?);
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
            let thin = (flags >> GATHER_ROWS_SHIFT) & 0x1f;
            if thin > 16 || (thin != 0 && a_t) {
                return Err(super::error::RANGE);
            }
            for i in (0..rows).filter(|_| !resident) {
                let mut row = if a_t {
                    None
                } else {
                    Some(a.cursor(i0 + i, 0)?)
                };
                let r = if i + 1 == rows { thin } else { 0 };
                for kk in 0..kt {
                    let to = a_at + (i * kt + kk) * a_slot as u32;
                    match row.as_mut() {
                        // A packed `A` (untransposed, whole tiles): its own slot, read whole.
                        Some(c) if a_slot != TILE_SLOT => {
                            let mut e = read(c.next_with_slot(a_slot)?, i0 + i, kk, to);
                            e[5] = a_slot as u32;
                            emit(e)?;
                        }
                        // One read of the datums through row `r - 1` of face 1:
                        // face 0 whole (its rows past `r` reach only the
                        // output's padding), then face 1's first `r` rows. One
                        // entry a tile, as a whole tile is: the mover's cost is
                        // per entry as much as per byte (two face-row reads a
                        // tile measured slower on small products).
                        Some(c) if r != 0 => {
                            let (ch, off) = c.next()?;
                            const DATA: u32 = TILE_DATA as u32;
                            let port = (i0 + i + kk) % PORTS;
                            emit([
                                op::READ,
                                ch,
                                port,
                                off + DATA,
                                to + DATA,
                                1024 + 64 * r,
                                0,
                                0,
                            ])?;
                        }
                        Some(c) => emit(read(c.next()?, i0 + i, kk, to))?,
                        None => emit(fetch(&a, a_t, i0 + i, kk, to)?)?,
                    }
                }
            }
            let span = match (b_t || b_resident || !laid_for_reader(rec, 3), b_slot) {
                (false, slot) => {
                    let slot = slot as u32;
                    BSpan::of(&b, b_at, kt, j0, cols, slot, BSpan::piece(flags, slot))?
                }
                _ => None,
            };
            if let Some(span) = span {
                for n in 0..span.count() {
                    emit(span.read(n))?;
                }
                return Ok(());
            }
            for kk in (0..kt).filter(|_| !b_resident) {
                let mut row = if b_t { None } else { Some(b.cursor(kk, j0)?) };
                for j in 0..cols {
                    let to = b_at + (kk * cols + j) * b_slot as u32;
                    emit(match row.as_mut() {
                        // A BF16 `B` (untransposed): its own slot, read whole.
                        Some(r) if b_slot != TILE_SLOT => {
                            let mut e = read(r.next_with_slot(b_slot)?, kk, j0 + j, to);
                            e[5] = b_slot as u32;
                            e
                        }
                        Some(r) => read(r.next()?, kk, j0 + j, to),
                        // The kernel transposes it: the tile as it lies.
                        None if b_unpacked => read(b.tile(j0 + j, kk)?, kk, j0 + j, to),
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
                let op = if col && flags & 16 == 0 {
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
        SCATTER_ROWS if WRITES => {
            let [_, idx_at, count, bound, j0, cols, stage_at, flags] = h;
            if flags & !(SCATTER_ROWS_COLUMN | SCATTER_ROWS_RANGE) != 0 {
                // 16-bit datums (bit 0) are not implemented.
                return Err(super::error::RANGE);
            }
            // The indices written: a range of them, or (none given) all.
            let (first, end) = (flags >> 8 & 0x3f, flags >> 16 & 0x3f);
            let (first, end) = if (first, end) == (0, 0) {
                (0, 32)
            } else if first < end && end <= 32 {
                (first, end)
            } else {
                return Err(super::error::RANGE);
            };
            let (x, cols) = indexed_rows(rec, count, bound, j0, cols)?;
            // As `SELECT_ROWS`: every index checked before any write, each
            // read again where it is used, 32-bit arithmetic throughout.
            const DATA: u32 = TILE_DATA as u32;
            let word = |r: u32| {
                idx_at
                    + DATA
                    + if flags & SCATTER_ROWS_COLUMN != 0 {
                        (r / 16 * 512 + r % 16 * 16) * 4
                    } else {
                        (r / 16 * 256 + r % 16) * 4
                    }
            };
            for r in 0..count {
                if index(word(r)) >= bound {
                    return Err(super::error::DOMAIN);
                }
            }
            let face_row = |r: u32, h: u32| DATA + ((r / 16 * 2 + h) * 16 + r % 16) * 64;
            let mut port = 0;
            for r in first..end.min(count) {
                let v = index(word(r));
                // A later index naming the same row is the one written.
                let mut later = r + 1;
                while later < count && index(word(later)) != v {
                    later += 1;
                }
                if later < count {
                    continue;
                }
                let mut row = x.cursor(v / 32, j0)?;
                for j in 0..cols {
                    let (ch, off) = row.next()?;
                    for half in 0..core::hint::black_box(2) {
                        let to = off + face_row(v % 32, half);
                        let from = stage_at + j * TILE_SLOT as u32 + face_row(r, half);
                        emit([op::WRITE, ch, port, to, from, 64, 0, 0])?;
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

/// [`SELECT_ROWS`]'s and [`SCATTER_ROWS`]'s shared header checks: `count`
/// indices of a tensor of `bound` rows, tile columns `j0..j0 + cols` of the
/// record's tensor, which this returns with the checked `cols`.
#[inline(always)]
fn indexed_rows(
    rec: &[[u32; 8]],
    count: u32,
    bound: u32,
    j0: u32,
    cols: u32,
) -> Result<(TensorRef, u32), u32> {
    if count == 0 || count > 32 || bound == 0 || bound > i32::MAX as u32 {
        return Err(super::error::LENGTH);
    }
    let cols = extent(cols)?;
    let x = tensor(rec, 1)?;
    // Every tile number a checked index can name fits a `u32` (the cursor
    // adds without checking: an index comes from L1).
    let tiles = bound
        .div_ceil(32)
        .checked_mul(x.ct)
        .and_then(|t| t.checked_add(x.first));
    if x.ct == 0 || tiles.is_none() || j0.checked_add(cols).is_none_or(|end| end > x.ct) {
        return Err(super::error::RANGE);
    }
    Ok((x, cols))
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
            expand_writes(&rec, |_| 0, |_| Ok(())),
            Err(super::super::error::DIRECTION)
        );
        assert!(expand_reads(&rec, &mut |_| 0, |_| Ok(())).is_ok());
    }

    /// A gather with both operands resident reads nothing, and is still a whole record: its
    /// tensor references are checked as any gather's are, and a prefetch or a weight stream (whose
    /// reads are `B`'s) is refused with it.
    #[test]
    fn a_gather_with_both_operands_resident_reads_nothing() {
        let (a, b) = (t(3, 3), t(3, 2));
        let [a0, a1] = a.encode();
        let [b0, b1] = b.encode();
        let both = GATHER_A_RESIDENT | GATHER_B_RESIDENT;
        let head = |flags: u32| [GATHER, flags | 3 << 8, 0x20000, 0x60000, 0, 2, 0, 2];
        let mut reads = 0;
        assert_eq!(
            expand_reads(
                &[head(both), a0, a1, b0, b1],
                |_| 0,
                |_| {
                    reads += 1;
                    Ok(())
                }
            ),
            Ok(())
        );
        assert_eq!(reads, 0);
        for flags in [both | GATHER_PREFETCH, both | GATHER_STREAM] {
            assert_eq!(
                expand_reads(&[head(flags), a0, a1, b0, b1], |_| 0, |_| Ok(())),
                Err(super::super::error::RANGE),
                "{flags:#x}"
            );
        }
        let [mut x0, x1] = a.encode();
        x0[0] = 9;
        assert_eq!(
            expand_reads(&[head(both), x0, x1, b0, b1], |_| 0, |_| Ok(())),
            Err(super::super::error::RANGE),
            "a reference past eight channels is refused, read or not"
        );
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
        // A column broadcast over a 3-wide grid: tile `(i, 0)` for every
        // `(i, j)`, through `READ_BROADCAST_COL`, or read as it lies.
        for (flags, want_op) in [
            (2, op::READ_BROADCAST_COL),
            (READ_RUN_COL_AS_IT_LIES, op::READ),
        ] {
            let head = [READ_RUN, 1, 5, 0x2_0000, flags, 3, 0, 0];
            let mut got = Vec::new();
            expand(&[head, a0, a1], |e| {
                assert_eq!(e[0], want_op, "flags {flags}");
                got.push((e[1], e[3]));
                Ok(())
            })
            .unwrap();
            let want: Vec<_> = (1..6).map(|n| a.tile(n / 3, 0).unwrap()).collect();
            assert_eq!(got, want, "column run, flags {flags}");
        }
        // Not with a broadcast or column-major order.
        for flags in [9, 10, 12] {
            let head = [READ_RUN, 0, 1, 0x2_0000, flags, 3, 2, 0];
            assert!(
                expand(&[head, a0, a1], |_| Ok(())).is_err(),
                "flags {flags}"
            );
        }
    }

    /// A gather told `A`'s last tile row has `r` rows reads that row's tiles
    /// from their first datum through row `r - 1` of face 1 -- where the face
    /// layout puts it, at the same offset in GDDR and L1, one read a tile as a
    /// whole tile is -- and everything else as whole tiles,
    /// as before; `r` past 16, or with `A` transposed, is refused.
    #[test]
    fn a_thin_gather_reads_the_first_rows_of_a_last_tile_row() {
        let mut a = t(3, 5);
        a.first = 2;
        let mut b = t(3, 7);
        b.first = 4;
        let [a0, a1] = a.encode();
        let [b0, b1] = b.encode();
        let (i0, rows, j0, cols, kt) = (1u32, 2u32, 2u32, 3u32, 4u32);
        let (a_at, b_at) = (0x2_0000u32, 0x6_0000u32);
        let gather = |flags: u32| {
            let head = [GATHER, flags | kt << 8, a_at, b_at, i0, rows, j0, cols];
            let mut got = Vec::new();
            expand(&[head, a0, a1, b0, b1], |e| {
                got.push(e);
                Ok(())
            })
            .map(|_| got)
        };
        let whole = gather(0).unwrap();
        for r in [1u32, 2, 15, 16] {
            for b_t in [0, 2] {
                let got = gather(gather_rows(r) | b_t).unwrap();
                let whole = gather(b_t).unwrap();
                // Row 0's `A` tiles as before, then row 1's thin, then `B` as before.
                let k = kt as usize;
                assert_eq!(got[..k], whole[..k], "r {r}");
                for kk in 0..k {
                    let w = whole[k + kk];
                    let e = got[k + kk];
                    // From the first datum through row `r - 1` of face 1, whose
                    // 16 datums start where the face layout says.
                    let data = 16;
                    let row = crate::dm::face_index(r as usize - 1, 16) as u32;
                    let end = 16 + 4 * (row + 16);
                    assert_eq!(
                        [e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]],
                        [
                            op::READ,
                            w[1],
                            w[2],
                            w[3] + data,
                            w[4] + data,
                            end - data,
                            0,
                            0
                        ],
                        "r {r}, kk {kk}"
                    );
                    // Congruent mod 64 at both ends, as a read must be.
                    assert_eq!(e[3] % 64, e[4] % 64);
                }
                assert_eq!(got[2 * k..], whole[2 * k..], "r {r}: B unchanged");
            }
        }
        assert_eq!(gather(0).unwrap(), whole, "0 is whole tiles");
        for bad in [gather_rows(17), gather_rows(31), gather_rows(1) | 1] {
            assert_eq!(gather(bad), Err(super::super::error::RANGE), "{bad:#x}");
        }
    }

    /// A prefetch cursor gives a resident-`A` prefetch `GATHER`'s reads in [`expand_reads`]'s
    /// order, and refuses what it refuses; any other record is not its.
    #[test]
    fn a_prefetch_cursor_reads_what_the_expansion_does() {
        let mut cases = 0;
        for (n, ct, first) in [(1u8, 5u32, 0u32), (3, 7, 4), (8, 64, 9), (5, 1, 2)] {
            let mut b = t(n, ct);
            b.first = first;
            let a = t(2, 3);
            for (kt, cols, j0) in [(1u32, 1u32, 0u32), (4, 3, 2), (32, 2, 0), (7, 1, 0)] {
                if j0 + cols > ct {
                    continue;
                }
                for flags in [
                    GATHER_A_RESIDENT,
                    GATHER_A_RESIDENT | 1,
                    GATHER_A_RESIDENT | gather_rows(5),
                    GATHER_A_RESIDENT | 2 | GATHER_B_UNPACK_TRANSPOSE,
                    GATHER_A_RESIDENT | GATHER_B_BF16,
                    GATHER_A_RESIDENT | gather_rows(5) | GATHER_B_BF16,
                ] {
                    let head = |f: u32| [GATHER, f | kt << 8, 0x2_0000, 0x6_0000, 0, 1, j0, cols];
                    let rec = |f: u32| {
                        let [a0, a1] = a.encode();
                        let [b0, b1] = b.encode();
                        [head(f), a0, a1, b0, b1]
                    };
                    let mut want = Vec::new();
                    let expanded = expand_reads(
                        &rec(flags | GATHER_PREFETCH),
                        |_| 0,
                        |e| {
                            want.push(e);
                            Ok(())
                        },
                    );
                    let mut c = PrefetchCursor::new(&rec(flags | GATHER_PREFETCH))
                        .map(|c| c.expect("a resident-A prefetch GATHER is the cursor's"));
                    let mut got = Vec::new();
                    let walked = match c.as_mut() {
                        Ok(c) => loop {
                            match c.next() {
                                None => break Ok(()),
                                Some(Ok(e)) => got.push(e),
                                Some(Err(code)) => break Err(code),
                            }
                        },
                        Err(&mut code) => Err(code),
                    };
                    assert_eq!(walked, expanded, "{flags:#x} kt {kt} cols {cols}");
                    if expanded.is_ok() {
                        assert_eq!(got, want, "{flags:#x} kt {kt} cols {cols}");
                    }
                    // Without the flag, or with `A` read, it is not the cursor's.
                    assert_eq!(PrefetchCursor::new(&rec(flags)), Ok(None));
                    let read_a = flags & !GATHER_A_RESIDENT & !1;
                    assert_eq!(
                        PrefetchCursor::new(&rec(read_a | GATHER_PREFETCH)),
                        Ok(None)
                    );
                    cases += 1;
                }
            }
        }
        assert!(cases > 40);
        // What the expansion refuses: an empty `B`, a transposing read, a resident `B`, a weight
        // stream's cap with a prefetch's, `kt` 0.
        let empty = TensorRef::default();
        for (flags, b) in [
            (GATHER_A_RESIDENT, empty),
            (GATHER_A_RESIDENT | 2, t(2, 4)),
            (GATHER_A_RESIDENT | GATHER_B_RESIDENT, t(2, 4)),
            (GATHER_A_RESIDENT | GATHER_STREAM, t(2, 4)),
            (GATHER_A_RESIDENT | GATHER_B_BF16 | 2, t(2, 4)),
            (
                GATHER_A_RESIDENT | GATHER_B_BF16 | 2 | GATHER_B_UNPACK_TRANSPOSE,
                t(2, 4),
            ),
        ] {
            let [a0, a1] = t(2, 4).encode();
            let [b0, b1] = b.encode();
            let rec = [
                [
                    GATHER,
                    flags | GATHER_PREFETCH | 2 << 8,
                    0x2_0000,
                    0x6_0000,
                    0,
                    1,
                    0,
                    2,
                ],
                a0,
                a1,
                b0,
                b1,
            ];
            let expanded = expand_reads(&rec, |_| 0, |_| Ok(()));
            assert!(expanded.is_err(), "{flags:#x}");
            let walked = PrefetchCursor::new(&rec).and_then(|c| {
                let mut c = c.expect("its");
                c.next().transpose().map(|_| ())
            });
            assert_eq!(walked, expanded, "{flags:#x}");
        }
    }

    /// A gather with `A` resident is the same gather without `A`'s entries -- whole, thin,
    /// transposed and packed BF16 -- with `B`'s unchanged, and one with `B` resident the same
    /// gather without `B`'s; `kt` is read past the flags; and a flag bit above it is refused.
    #[test]
    fn a_resident_gather_reads_only_b() {
        let mut a = t(3, 5);
        a.first = 2;
        let mut b = t(3, 7);
        b.first = 4;
        let [a0, a1] = a.encode();
        let [b0, b1] = b.encode();
        let (i0, rows, j0, cols, kt) = (1u32, 2u32, 2u32, 3u32, 4u32);
        let gather = |flags: u32, a0: [u32; 8], b0: [u32; 8]| {
            let head = [
                GATHER,
                flags | kt << 8,
                0x2_0000,
                0x6_0000,
                i0,
                rows,
                j0,
                cols,
            ];
            let mut got = Vec::new();
            expand(&[head, a0, a1, b0, b1], |e| {
                got.push(e);
                Ok(())
            })
            .map(|_| got)
        };
        let a_entries = (rows * kt) as usize;
        for flags in [0, 1, 2, 3, gather_rows(1), gather_rows(16) | 2] {
            let whole = gather(flags, a0, b0).unwrap();
            let resident = gather(flags | GATHER_A_RESIDENT, a0, b0).unwrap();
            assert_eq!(resident, whole[a_entries..], "flags {flags:#x}");
            assert!(resident.iter().all(|e| e[4] >= 0x6_0000), "only B's slots");
        }
        // `B` resident: `A`'s entries, unchanged, and nothing of `B`'s.
        for flags in [0, 1, 2, 3, gather_rows(1), gather_rows(16) | 2] {
            let whole = gather(flags, a0, b0).unwrap();
            let resident = gather(flags | GATHER_B_RESIDENT, a0, b0).unwrap();
            assert_eq!(resident, whole[..a_entries], "flags {flags:#x}");
            assert!(resident.iter().all(|e| e[4] < 0x6_0000), "only A's slots");
            assert_eq!(
                gather(flags | GATHER_A_RESIDENT | GATHER_B_RESIDENT, a0, b0),
                Ok(Vec::new()),
                "flags {flags:#x}"
            );
        }
        assert_eq!(gather_kt(GATHER_B_RESIDENT | 7 << 8), 7);
        assert_eq!(gather_kt(GATHER_A_RESIDENT | 7 << 8 | 0xff), 7);
        // A prefetch reads what the same record reads without the flag (only the ID differs),
        // and only as plain reads: not packed BF16, not a transposing move.
        assert_eq!(gather_kt(GATHER_PREFETCH | 7 << 8), 7);
        for flags in [
            0,
            GATHER_A_RESIDENT,
            GATHER_A_RESIDENT | 1,
            2 | GATHER_B_UNPACK_TRANSPOSE,
        ] {
            assert_eq!(
                gather(flags | GATHER_PREFETCH, a0, b0),
                gather(flags, a0, b0),
                "{flags:#x}"
            );
        }
        for flags in [
            GATHER_B_RESIDENT | GATHER_A_RESIDENT,
            GATHER_B_RESIDENT,
            1,
            2,
        ] {
            assert_eq!(
                gather(flags | GATHER_PREFETCH, a0, b0),
                Err(super::super::error::RANGE),
                "{flags:#x}"
            );
        }
        // A weight stream reads what the record reads; it is no prefetch.
        for flags in [0, GATHER_A_RESIDENT, 2 | GATHER_B_UNPACK_TRANSPOSE] {
            assert_eq!(
                gather(flags | GATHER_STREAM, a0, b0),
                gather(flags, a0, b0),
                "{flags:#x}"
            );
            assert_eq!(
                gather(flags | GATHER_STREAM | GATHER_PREFETCH, a0, b0),
                Err(super::super::error::RANGE),
                "{flags:#x}"
            );
        }
        // Bits 29-31 name `B`'s format: codes past the last format are refused, and each format
        // reads `B` whole in slots of its own size.
        for code in GATHER_FORMATS.len() as u32..8 {
            assert_eq!(
                gather(code << GATHER_B_FORMAT_SHIFT, a0, b0),
                Err(super::super::error::RANGE),
                "code {code}"
            );
        }
        for format in GATHER_FORMATS {
            let flags = gather_b_format(format).unwrap();
            let slot = crate::format::TileFormat::new(format).slot_bytes() as u32;
            let reads = gather(flags | GATHER_A_RESIDENT, a0, b0).unwrap();
            assert!(
                !reads.is_empty() && reads.iter().all(|e| e[5] == slot),
                "{format:?}"
            );
        }
        // Bits 22-24 name `A`'s, alike: each format reads `A` whole in slots of its own size, one
        // after another from `a_at`, and `B` as before; codes past the last are refused, and so is
        // a packed `A` with what only an FP32 one takes (a transposed read, thin rows, packed BF16).
        let fp32 = gather(0, a0, b0).unwrap();
        let a_entries = (rows * kt) as usize;
        for code in GATHER_FORMATS.len() as u32..8 {
            assert_eq!(
                gather(code << GATHER_A_FORMAT_SHIFT, a0, b0),
                Err(super::super::error::RANGE),
                "code {code}"
            );
        }
        for format in GATHER_FORMATS {
            let flags = gather_a_format(format).unwrap();
            let slot = crate::format::TileFormat::new(format).slot_bytes() as u32;
            let reads = gather(flags, a0, b0).unwrap();
            assert_eq!(reads[a_entries..], fp32[a_entries..], "{format:?}: B");
            for (n, e) in reads[..a_entries].iter().enumerate() {
                assert_eq!(e[5], slot, "{format:?}");
                assert_eq!(e[4], 0x2_0000 + n as u32 * slot, "{format:?}");
            }
            assert_eq!(
                gather(flags | GATHER_A_RESIDENT, a0, b0).unwrap(),
                fp32[a_entries..],
                "{format:?}: resident"
            );
            if format != crate::tile::L1Format::Fp32 {
                for bad in [1, gather_rows(3)] {
                    assert_eq!(
                        gather(flags | bad, a0, b0),
                        Err(super::super::error::RANGE),
                        "{format:?}, {bad:#x}"
                    );
                }
            }
        }
        assert_eq!(gather_kt(GATHER_A_FORMAT_MASK | 0x3fff << 8), 0x3fff);
    }

    /// A BF16 `B` ([`GATHER_B_BF16`]) is read from its own slots: each tile's whole
    /// [`super::super::BF16_TILE_SLOT`] from where a BF16 tensor keeps it (slot `t / n` of 2,112
    /// bytes on channel `t % n`), into a 2,112-byte L1 slot of its own (an FP32 `B`'s ring holds
    /// half the tiles in the same bytes), and `A`'s entries are untouched -- whole, thin, transposed, resident, streamed. A transposed `B` and
    /// packed BF16 are refused.
    #[test]
    fn a_bf16_b_reads_its_own_slots_whole() {
        use super::super::BF16_TILE_SLOT;
        let mut a = t(3, 5);
        a.first = 2;
        let mut b = t(3, 7);
        b.first = 4;
        let [a0, a1] = a.encode();
        let [b0, b1] = b.encode();
        let (i0, rows, j0, cols, kt) = (1u32, 2u32, 2u32, 3u32, 4u32);
        let gather = |flags: u32| {
            let head = [
                GATHER,
                flags | kt << 8,
                0x2_0000,
                0x6_0000,
                i0,
                rows,
                j0,
                cols,
            ];
            let mut got = Vec::new();
            expand(&[head, a0, a1, b0, b1], |e| {
                got.push(e);
                Ok(())
            })
            .map(|_| got)
        };
        let mut cases = 0;
        for flags in [
            0,
            1,
            gather_rows(5),
            GATHER_A_RESIDENT,
            GATHER_STREAM,
            GATHER_A_RESIDENT | GATHER_PREFETCH,
        ] {
            let f32_b = gather(flags).unwrap();
            let bf16_b = gather(flags | GATHER_B_BF16).unwrap();
            assert_eq!(f32_b.len(), bf16_b.len(), "{flags:#x}");
            let first_b = f32_b.iter().position(|e| e[4] >= 0x6_0000).unwrap();
            assert_eq!(
                bf16_b[..first_b],
                f32_b[..first_b],
                "A's entries, {flags:#x}"
            );
            for (n, (got, was)) in bf16_b[first_b..].iter().zip(&f32_b[first_b..]).enumerate() {
                let (kk, j) = (n as u32 / cols, n as u32 % cols);
                let tile = b.first + kk * b.ct + j0 + j;
                let (slot, c) = (tile / b.n as u32, (tile % b.n as u32) as usize);
                let want = [
                    op::READ,
                    b.channels[c] as u32,
                    was[2],
                    b.base[c] + slot * BF16_TILE_SLOT as u32,
                    0x6_0000 + (kk * cols + j) * BF16_TILE_SLOT as u32,
                    BF16_TILE_SLOT as u32,
                    0,
                    0,
                ];
                assert_eq!(*got, want, "{flags:#x}, B tile {n}");
                assert_eq!(
                    got[4] - 0x6_0000,
                    (was[4] - 0x6_0000) / TILE_SLOT as u32 * BF16_TILE_SLOT as u32,
                    "the FP32 B's L1 slot, compacted"
                );
                cases += 1;
            }
        }
        assert!(cases > 50);
        for bad in [
            GATHER_B_BF16 | 2,
            GATHER_B_BF16 | 2 | GATHER_B_UNPACK_TRANSPOSE,
        ] {
            assert_eq!(gather(bad), Err(super::super::error::RANGE), "{bad:#x}");
        }
    }

    /// A `B` the unpacker transposes is read as it lies: the same tiles, to the same slots, as
    /// the transposed gather, each a plain `READ` where that one is `READ_TRANSPOSED`. Refused
    /// without the transpose flag, and with packed BF16.
    #[test]
    fn an_unpacker_transposed_b_is_read_as_it_lies() {
        let a = t(3, 5);
        let b = t(7, 4);
        let [a0, a1] = a.encode();
        let [b0, b1] = b.encode();
        let (i0, rows, j0, cols, kt) = (1u32, 2u32, 2u32, 3u32, 4u32);
        let gather = |flags: u32| {
            let head = [
                GATHER,
                flags | kt << 8,
                0x2_0000,
                0x6_0000,
                i0,
                rows,
                j0,
                cols,
            ];
            let mut got = Vec::new();
            expand(&[head, a0, a1, b0, b1], |e| {
                got.push(e);
                Ok(())
            })
            .map(|_| got)
        };
        for a_flag in [0, 1] {
            let software = gather(a_flag | 2).unwrap();
            let unpacked = gather(a_flag | 2 | GATHER_B_UNPACK_TRANSPOSE).unwrap();
            assert_eq!(software.len(), unpacked.len());
            let mut b_reads = 0;
            for (s, u) in software.iter().zip(&unpacked) {
                if s[4] >= 0x6_0000 {
                    assert_eq!(s[0], op::READ_TRANSPOSED);
                    assert_eq!(u[0], op::READ);
                    assert_eq!(s[1..], u[1..], "the same tile to the same slot");
                    b_reads += 1;
                } else {
                    assert_eq!(s, u);
                }
            }
            assert_eq!(b_reads, kt * cols);
        }
        assert_eq!(
            gather(GATHER_B_UNPACK_TRANSPOSE),
            Err(super::super::error::RANGE)
        );
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
        assert_eq!(expand_writes(&ok, |_| 0, |_| Ok(())), Err(DIRECTION));
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

    /// An L1 holding `values` as an index tile at `IDX` in column 0 (an
    /// `[n, 1]` index tensor's tile), refusing any other address.
    fn l1_column_of(values: &[u32]) -> impl FnMut(u32) -> u32 {
        let mut words = std::collections::HashMap::new();
        for (r, &v) in values.iter().enumerate() {
            words.insert(IDX + 16 + crate::dm::face_index(r, 0) as u32 * 4, v);
        }
        move |at| *words.get(&at).unwrap_or_else(|| panic!("read of {at:#x}"))
    }

    /// A `SCATTER_ROWS` of `count` indices into `bound` rows, tile columns
    /// `j0..j0 + cols` of `x`.
    fn scatter(
        x: TensorRef,
        bound: u32,
        count: u32,
        j0: u32,
        cols: u32,
        flags: u32,
    ) -> [[u32; 8]; 3] {
        let [m0, m1] = x.encode();
        [
            [SCATTER_ROWS, IDX, count, bound, j0, cols, STAGE, flags],
            m0,
            m1,
        ]
    }

    /// The writes a `SCATTER_ROWS` of distinct `values` must make, from the
    /// face layout (`face_index`) and `TensorRef::tile`, independently of
    /// `expand`'s arithmetic: staged row `r`'s two face-rows of each tile
    /// column to row `values[r]`'s, the ports rotating.
    fn scattered(x: &TensorRef, values: &[u32], j0: u32, cols: u32) -> Vec<[u32; 8]> {
        let mut want = Vec::new();
        for (r, &v) in values.iter().enumerate() {
            for j in 0..cols {
                let (ch, off) = x.tile(v / 32, j0 + j).unwrap();
                for h in 0..2usize {
                    let to = off as u64
                        + TILE_DATA
                        + 4 * crate::dm::face_index((v % 32) as usize, 16 * h) as u64;
                    let from = STAGE as u64
                        + j as u64 * TILE_SLOT
                        + TILE_DATA
                        + 4 * crate::dm::face_index(r, 16 * h) as u64;
                    let n = want.len() as u32;
                    want.push([op::WRITE, ch, n % PORTS, to as u32, from as u32, 64, 0, 0]);
                }
            }
        }
        want
    }

    /// Each index's staged row goes to its two face-rows of every tile column
    /// of the destination -- across a channel wrap mid-row, a view that starts
    /// mid-grid, the first and last rows, and a row either side of a tile --
    /// with the indices in a row of their tile or in a column of it.
    #[test]
    fn scatter_rows_writes_the_face_rows_of_each_index() {
        let mut x = t(3, 5);
        x.first = 2;
        let bound = 100;
        let values = [
            0u32, 99, 31, 32, 33, 5, 64, 17, 16, 63, 95, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 18, 19, 20, 21, 22, 23, 24,
        ];
        let (j0, cols) = (1u32, 3u32);
        let want = scattered(&x, &values, j0, cols);
        for (flags, l1) in [
            (
                0,
                std::boxed::Box::new(l1_of(&values)) as std::boxed::Box<dyn FnMut(u32) -> u32>,
            ),
            (
                SCATTER_ROWS_COLUMN,
                std::boxed::Box::new(l1_column_of(&values)),
            ),
        ] {
            let rec = scatter(x, bound, values.len() as u32, j0, cols, flags);
            let mut got = Vec::new();
            expand_writes(&rec, l1, |e| {
                got.push(e);
                Ok(())
            })
            .unwrap();
            assert_eq!(got, want, "flags {flags}");
            // What a GDDR write needs: congruent mod 16 (and these mod 64).
            for e in &got {
                assert_eq!(e[3] % 64, e[4] % 64);
            }
        }
    }

    /// When indices repeat, only the last of them is written -- its staged
    /// row, not an earlier one's -- whichever positions they are in.
    #[test]
    fn scatter_rows_writes_a_repeated_row_once_from_its_last_index() {
        let x = t(2, 2);
        let values = [5u32, 7, 5, 9, 7, 7, 40];
        let rec = scatter(x, 64, values.len() as u32, 0, 2, 0);
        let mut got = Vec::new();
        expand_writes(&rec, l1_of(&values), |e| {
            got.push(e);
            Ok(())
        })
        .unwrap();
        // The staged rows written: 2 (row 5), 3 (row 9), 5 (row 7), 6 (row 40).
        let mut want = Vec::new();
        for (r, v) in [(2usize, 5u32), (3, 9), (5, 7), (6, 40)] {
            for j in 0..2 {
                let (ch, off) = x.tile(v / 32, j).unwrap();
                for h in 0..2usize {
                    let to = off as u64
                        + TILE_DATA
                        + 4 * crate::dm::face_index((v % 32) as usize, 16 * h) as u64;
                    let from = STAGE as u64
                        + j as u64 * TILE_SLOT
                        + TILE_DATA
                        + 4 * crate::dm::face_index(r, 16 * h) as u64;
                    want.push((ch, to as u32, from as u32));
                }
            }
        }
        let got: Vec<_> = got.iter().map(|e| (e[1], e[3], e[4])).collect();
        assert_eq!(got, want);
    }

    /// Records that split one batch's indices into ranges make, between them,
    /// exactly the writes of one record over them all, each once: a repeated
    /// row only from its last index, whichever range that is in. A range that
    /// is empty, reversed or past 32 is refused.
    #[test]
    fn scatter_rows_ranges_split_one_records_writes() {
        let x = t(2, 3);
        let values = [5u32, 7, 5, 9, 7, 7, 40, 3, 9, 12, 5];
        let n = values.len() as u32;
        let writes = |flags: u32| {
            let rec = scatter(x, 64, n, 1, 2, flags);
            let mut got = Vec::new();
            expand_writes(&rec, l1_of(&values), |e| {
                got.push(e);
                Ok(())
            })
            .map(|()| got)
        };
        let whole = writes(0).unwrap();
        for cuts in [&[0u32, 11][..], &[0, 3, 11], &[0, 1, 6, 9, 11], &[0, 4, 32]] {
            let mut split = Vec::new();
            for w in cuts.windows(2) {
                split.extend(writes(scatter_rows_range(w[0]..w[1])).unwrap());
            }
            let addresses = |v: &[[u32; 8]]| {
                let mut a: Vec<_> = v.iter().map(|e| (e[1], e[3], e[4])).collect();
                a.sort_unstable();
                a
            };
            assert_eq!(addresses(&split), addresses(&whole), "{cuts:?}");
            let mut to: Vec<_> = split.iter().map(|e| (e[1], e[3])).collect();
            to.sort_unstable();
            to.dedup();
            assert_eq!(to.len(), split.len(), "{cuts:?}: an address written twice");
        }
        for bad in [(3, 3), (5, 2), (0, 33), (40, 41)] {
            assert_eq!(
                writes(bad.0 << 8 | bad.1 << 16),
                Err(crate::dm::error::RANGE),
                "{bad:?}"
            );
        }
    }

    /// An index at or past `bound`, or a negative one, is `DOMAIN` before a
    /// single write is issued, wherever it is; the host's `expand` reads
    /// every index as 0 (one row, written once) and checks the geometry.
    #[test]
    fn scatter_rows_refuses_an_index_out_of_range_before_any_write() {
        let x = t(4, 3);
        for (values, ok) in [
            ([0u32, 99, 5], true),
            ([0, 100, 5], false),
            ([0, 1, 100], false),
            ([u32::MAX, 1, 2], false),
            ([1, 2, 0x8000_0000], false),
        ] {
            let rec = scatter(x, 100, values.len() as u32, 0, 3, 0);
            let mut issued = 0;
            let got = expand_writes(&rec, l1_of(&values), |_| {
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
        let rec = scatter(x, 100, 4, 0, 3, SCATTER_ROWS_COLUMN);
        let mut issued = 0;
        expand(&rec, |_| {
            issued += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(issued, 3 * 2);
    }

    #[test]
    fn scatter_rows_refuses_what_it_should() {
        use crate::dm::error::{DIRECTION, LENGTH, RANGE};
        let x = t(3, 4);
        let ok = scatter(x, 64, 3, 1, 3, 0);
        let run = |rec: &[[u32; 8]]| expand_writes(rec, |_| 0, |_| Ok(()));
        assert_eq!(run(&ok), Ok(()));
        // The reader's code does not hold it, nor does its mover run it.
        assert_eq!(expand_reads(&ok, |_| 0, |_| Ok(())), Err(DIRECTION));
        assert_eq!(direction(SCATTER_ROWS), Some(Direction::Write));
        assert_eq!(len(SCATTER_ROWS), 3);
        assert!(crate::dm::Mover::NC.permits(SCATTER_ROWS));
        assert!(!crate::dm::Mover::B.permits(SCATTER_ROWS));
        for (word, value, want) in [
            (2, 0, LENGTH),       // no indices
            (2, 33, LENGTH),      // more than a tile of them
            (3, 0, LENGTH),       // no rows
            (3, 1 << 31, LENGTH), // more than an I32 names
            (5, 0, LENGTH),       // no tile columns
            (4, 2, RANGE),        // columns 2..5 of 4
            (5, 4, RANGE),        // columns 1..5 of 4
            (7, 1, RANGE),        // 16-bit datums are not implemented
            (7, 4, RANGE),        // an unknown flag
        ] {
            let mut bad = ok;
            bad[0][word] = value;
            assert_eq!(run(&bad), Err(want), "word {word} = {value:#x}");
        }
        let wide = TensorRef {
            ct: 1 << 16,
            ..t(3, 1)
        };
        assert_eq!(run(&scatter(wide, i32::MAX as u32, 3, 0, 3, 0)), Err(RANGE));
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
    /// A gather of `B` only (`A` resident), `kt` terms of `cols` columns from `j0`, `B` in `b`,
    /// marked [`REF_LAID_FOR_READER`] where `laid`.
    fn b_gather(
        flags: u32,
        kt: u32,
        j0: u32,
        cols: u32,
        b: &TensorRef,
        laid: bool,
    ) -> Vec<[u32; 8]> {
        let a = TensorRef {
            n: 1,
            ct: 1,
            ..TensorRef::default()
        };
        let (a, mut b) = (a.encode(), b.encode());
        if laid {
            b[0][4] = REF_LAID_FOR_READER;
        }
        let b_at = 0x2_0000;
        Vec::from([
            [
                GATHER,
                flags | GATHER_A_RESIDENT | kt << 8,
                0x8_0000,
                b_at,
                0,
                1,
                j0,
                cols,
            ],
            a[0],
            a[1],
            b[0],
            b[1],
        ])
    }

    /// Each 64-byte unit a list of reads moves: `(channel, GDDR offset, L1 address)`, sorted.
    fn units_moved(reads: &[[u32; 8]]) -> Vec<(u32, u32, u32)> {
        let mut out = Vec::new();
        for r in reads {
            assert_eq!(r[0], op::READ, "{r:?}");
            assert_eq!(r[5] % 64, 0, "whole 64-byte units: {r:?}");
            for k in (0..r[5]).step_by(64) {
                out.push((r[1], r[3] + k, r[4] + k));
            }
        }
        out.sort_unstable();
        out
    }

    /// X9's layout (`docs/plans/decode-bytes-in-flight.md`): a one-channel `B` laid out for its
    /// reader ([`REF_LAID_FOR_READER`]), read as it lies, is read in pieces of at most one request,
    /// and moves exactly the bytes, to exactly the slots, that a read a tile would (the oracle
    /// below, from the record's own definition: tile `(kk, j0 + j)` of a tensor `ct` wide at
    /// `base + (first + kk * ct + j0 + j) * slot`, into `b_at + (kk * cols + j) * slot`). Whole when
    /// the rows meet (`ct == cols`), a run a row when they do not; a read ahead reads the same
    /// pieces and the pump's cursor gives them; unmarked, or interleaved, a `B` keeps a read a tile.
    #[test]
    fn a_one_channel_b_laid_for_its_reader_is_read_in_large_pieces() {
        let bf16 = gather_b_format(crate::tile::L1Format::Bf16).unwrap();
        for (flags, slot) in [
            (GATHER_STREAM | bf16, 2112u32),
            (GATHER_STREAM, TILE_SLOT as u32),
        ] {
            for (kt, j0, cols, ct, first) in [
                (64u32, 0u32, 3u32, 3u32, 0u32),
                (13, 0, 1, 1, 0),
                (7, 2, 3, 9, 5),
                (64, 0, 5, 5, 128),
            ] {
                let what = std::format!(
                    "slot {slot}, kt {kt}, j0 {j0}, cols {cols}, ct {ct}, first {first}"
                );
                let mut b = TensorRef {
                    n: 1,
                    first,
                    ct,
                    ..TensorRef::default()
                };
                (b.channels[0], b.base[0]) = (6, 0x40_0000);
                let rec = b_gather(flags, kt, j0, cols, &b, true);
                let mut reads = Vec::new();
                expand_reads(
                    &rec,
                    |_| 0,
                    |e| {
                        reads.push(e);
                        Ok(())
                    },
                )
                .unwrap();
                let mut oracle = Vec::new();
                for kk in 0..kt {
                    for j in 0..cols {
                        let t = first + kk * ct + j0 + j;
                        oracle.push([
                            op::READ,
                            6,
                            0,
                            0x40_0000 + t * slot,
                            0x2_0000 + (kk * cols + j) * slot,
                            slot,
                            0,
                            0,
                        ]);
                    }
                }
                assert_eq!(
                    units_moved(&reads),
                    units_moved(&oracle),
                    "{what}: the bytes moved"
                );
                let runs = if ct == cols { 1 } else { kt };
                let per_run = (kt * cols * slot / runs).div_ceil(16384);
                assert_eq!(reads.len() as u32, runs * per_run, "{what}: reads");
                assert!(
                    reads.iter().all(|r| r[5] <= 16384 && r[2] < PORTS),
                    "{what}"
                );
                assert!(
                    reads.iter().all(|r| r[3] % 64 == r[4] % 64),
                    "{what}: congruent"
                );
                // A read ahead: the same bytes to the same slots, in the same pieces, and the
                // pump's cursor gives the record's own expansion.
                let mut ahead = rec.clone();
                ahead[0][1] = ahead[0][1] & !GATHER_STREAM | GATHER_PREFETCH;
                let mut expanded = Vec::new();
                expand_reads(
                    &ahead,
                    |_| 0,
                    |e| {
                        expanded.push(e);
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(
                    units_moved(&expanded),
                    units_moved(&oracle),
                    "{what}: ahead's bytes"
                );
                assert_eq!(expanded, reads, "{what}: ahead's reads are the stream's");
                let cursor = PrefetchCursor::new(&ahead).unwrap().unwrap();
                assert_eq!(
                    cursor.left(),
                    expanded.len() as u32,
                    "{what}: the cursor's count"
                );
                let pumped: Vec<_> = cursor.map(Result::unwrap).collect();
                assert_eq!(pumped, expanded, "{what}: the cursor's reads");
                // Unmarked (a one-channel chip's tensor): a read a tile, as before.
                let mut plain = Vec::new();
                expand_reads(
                    &b_gather(flags, kt, j0, cols, &b, false),
                    |_| 0,
                    |e| {
                        plain.push(e);
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(
                    plain.len() as u32,
                    kt * cols,
                    "{what}: unmarked, a read a tile"
                );
                assert_eq!(
                    units_moved(&plain),
                    units_moved(&oracle),
                    "{what}: unmarked bytes"
                );
            }
        }
        // Interleaved over eight channels: a read a tile, as before.
        let mut b = TensorRef {
            n: 8,
            ct: 4,
            ..TensorRef::default()
        };
        for c in 0..8 {
            (b.channels[c], b.base[c]) = (c as u8, 0x40_0000);
        }
        let rec = b_gather(GATHER_STREAM | bf16, 16, 0, 4, &b, true);
        let mut reads = Vec::new();
        expand_reads(
            &rec,
            |_| 0,
            |e| {
                reads.push(e);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(reads.len(), 64, "a read a tile");
        assert!(reads.iter().all(|r| r[5] == 2112));
    }
}
