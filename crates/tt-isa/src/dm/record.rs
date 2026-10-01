//! Op records: a whole op's mover work in a few list entries, expanded on the
//! tile.
//!
//! A plain list entry ([`super::Entry`]) moves or computes one tile, so a list
//! is as long as the op is big, and its bytes -- not the round trips -- were
//! what a GDDR op spent its host time on (the host's MMIO is uncached here,
//! `docs/ttsim-divergence.md` measurement M). A record names the tensors
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
/// `A` is read transposed, bit 1 `B`, bits 8.. `kt`.
pub const GATHER: u32 = 0x10;
/// One matmul block's outputs, L1 -> GDDR: `[SCATTER, out_at, out_stride, i0,
/// rows, j0, cols, 0]` + `C`. Output `(i, j)`'s datums are at `out_at + (i *
/// cols + j) * out_stride`.
pub const SCATTER: u32 = 0x11;
/// A run of element-wise tiles: `[ELTWISE, kind, scalar, first, count, flags,
/// stage, 0]` + `A` + `B` + `OUT`. Tiles `first..first + count` of `A` in
/// row-major order, each read into slot `2n` of the staging area at `stage`
/// (its `B` into `2n + 1`), computed in place and written out. `flags`: bit 0
/// the kind takes `B`, bit 1 `B` is one row broadcast down `A`. `B` is all zero
/// when the kind takes none.
pub const ELTWISE: u32 = 0x12;
/// A run of column sums: `[SUM, first, count, rt, stage, last_rows, 0, 0]` +
/// `A` + `OUT`. Columns `first..first + count` of `A`'s tile grid, `rt` tiles
/// tall, each summed in row order into row 0 of an accumulator slot; of the
/// last tile row only its first `last_rows` rows (`0`: all 32), so a ragged
/// tensor's padding never reaches a sum.
pub const SUM: u32 = 0x13;
/// Set the padding of a run of a tensor's edge tiles: `[FILL_PAD, value,
/// first, count, rows, cols, stage, rt]` + `A`, where `rows` and `cols` are
/// the valid rows of the last tile row and columns of the last tile column
/// (`0`: 32, that edge is not ragged) and `rt` the tile rows. The edge tiles,
/// numbered: the last tile row left to right if its rows are ragged, then the
/// last tile column top to bottom (without a corner already counted) if its
/// columns are; tiles `first..first + count` of that numbering are each read
/// into slot `n` of the staging area at `stage`, filled with `value` outside
/// the valid region ([`super::kind::FILL_PAD`]) and written back in place.
pub const FILL_PAD: u32 = 0x14;

/// Slots of the staging area a column sum uses at once, accumulator included.
pub const SUM_SLOTS: usize = 200;

/// Most tiles a record may name along any one of its loops: far more than any
/// list holds, and few enough that a corrupt record cannot keep the mover busy
/// for long.
pub const MAX_EXTENT: u32 = 1 << 16;

/// Entries in the record whose header word 0 is `op`: 1 for a plain entry.
pub const fn len(op: u32) -> usize {
    match op {
        GATHER | SUM => 5,
        SCATTER | FILL_PAD => 3,
        ELTWISE => 7,
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
    /// that is never read (`ELTWISE`'s absent `B`); whether each channel is one
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
        let (s, c) = div_rem(t, self.n as u32);
        let offset = (self.base[c as usize] as u64) + s as u64 * TILE_SLOT;
        let offset = u32::try_from(offset).map_err(|_| super::error::RANGE)?;
        Ok((self.channels[c as usize] as u32, offset))
    }
}

const PORTS: u32 = crate::dram::PORTS as u32;

/// `(a / b, a % b)` by shift and subtract, for a `b` known only at run time.
///
/// The mover's image may not contain `divu`/`remu`: the firmware is built to
/// run on any core, and RISCV T2 has no integer divide (`InstructionSet.md`;
/// the instruction gate in `tt-firmware-images/build.rs` refuses it). Division
/// by a constant compiles to a multiply and needs none of this.
pub const fn div_rem(a: u32, b: u32) -> (u32, u32) {
    assert!(b != 0);
    let (mut q, mut r) = (0u32, 0u32);
    let mut bit = 32;
    while bit > 0 {
        bit -= 1;
        r = (r << 1) | ((a >> bit) & 1);
        if r >= b {
            r -= b;
            q |= 1 << bit;
        }
    }
    (q, r)
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
pub fn expand(
    rec: &[[u32; 8]],
    mut emit: impl FnMut([u32; 8]) -> Result<(), u32>,
) -> Result<(), u32> {
    let h = rec[0];
    if rec.len() != len(h[0]) || !is_record(h[0]) {
        return Err(super::error::OP);
    }
    match h[0] {
        GATHER => {
            let [_, flags, a_at, b_at, i0, rows, j0, cols] = h;
            let (a_t, b_t, kt) = (flags & 1 != 0, flags & 2 != 0, extent(flags >> 8)?);
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
            for i in 0..rows {
                for kk in 0..kt {
                    let to = a_at + (i * kt + kk) * TILE_SLOT as u32;
                    emit(fetch(&a, a_t, i0 + i, kk, to)?)?;
                }
            }
            for kk in 0..kt {
                for j in 0..cols {
                    let to = b_at + (kk * cols + j) * TILE_SLOT as u32;
                    emit(fetch(&b, b_t, kk, j0 + j, to)?)?;
                }
            }
        }
        SCATTER => {
            let [_, out_at, stride, i0, rows, j0, cols, _] = h;
            let (rows, cols) = (extent(rows)?, extent(cols)?);
            let c = tensor(rec, 1)?;
            for i in 0..rows {
                for j in 0..cols {
                    let (ch, off) = c.tile(i0 + i, j0 + j)?;
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
        ELTWISE => {
            let [_, kind, scalar, first, count, flags, stage, _] = h;
            let count = extent(count)?;
            let (binary, row) = (flags & 1 != 0, flags & 2 != 0);
            let (a, b, out) = (tensor(rec, 1)?, tensor(rec, 3)?, tensor(rec, 5)?);
            if a.ct == 0 {
                return Err(super::error::LENGTH);
            }
            let slot = |i: u32| stage + i * TILE_SLOT as u32;
            // Row-major from `first`, stepped rather than divided per tile.
            let (mut i, mut j) = div_rem(first, a.ct);
            for n in 0..count {
                if n > 0 {
                    j += 1;
                    if j == a.ct {
                        (i, j) = (i + 1, 0);
                    }
                }
                let (ch, off) = a.tile(i, j)?;
                emit([
                    op::READ,
                    ch,
                    n % PORTS,
                    off,
                    slot(2 * n),
                    TILE_SLOT as u32,
                    0,
                    0,
                ])?;
                if binary {
                    let (ch, off) = if row { b.tile(0, j)? } else { b.tile(i, j)? };
                    emit([
                        op::READ,
                        ch,
                        (n + 1) % PORTS,
                        off,
                        slot(2 * n + 1),
                        TILE_SLOT as u32,
                        0,
                        0,
                    ])?;
                }
                emit([
                    op::COMPUTE,
                    kind,
                    scalar,
                    0,
                    slot(2 * n),
                    slot(2 * n),
                    slot(2 * n + 1),
                    0,
                ])?;
                let (ch, off) = out.tile(i, j)?;
                emit([
                    op::WRITE,
                    ch,
                    n % PORTS,
                    off + TILE_DATA as u32,
                    slot(2 * n) + TILE_DATA as u32,
                    4096,
                    0,
                    0,
                ])?;
            }
        }
        SUM => {
            let [_, first, count, rt, stage, last_rows, ..] = h;
            let (count, rt) = (extent(count)?, extent(rt)?);
            if last_rows > 31 {
                return Err(super::error::LENGTH);
            }
            let (a, out) = (tensor(rec, 1)?, tensor(rec, 3)?);
            sum(
                &a,
                &out,
                first,
                count,
                rt as usize,
                stage,
                last_rows,
                &mut emit,
            )?;
        }
        FILL_PAD => {
            let [_, value, first, count, rows, cols, stage, rt] = h;
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
                emit([op::READ, ch, n % PORTS, off, at, TILE_SLOT as u32, 0, 0])?;
                let vr = if i == rt - 1 { rows } else { 0 };
                let vc = if j == a.ct - 1 { cols } else { 0 };
                emit([
                    op::COMPUTE,
                    super::kind::FILL_PAD,
                    value,
                    vr | vc << 8,
                    at,
                    at,
                    at,
                    0,
                ])?;
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
            }
        }
        _ => return Err(super::error::OP),
    }
    Ok(())
}

/// The column sums of [`SUM`]: the schedule `tt_kernels::tensor::sum_rows`
/// had when it built lists itself. Whole columns are packed into one list's
/// worth of slots while they fit; a column taller than that spans several,
/// accumulating in place. Where it started a new list, a [`op::WAIT`] stands
/// in, since the next entries reuse the slots.
#[allow(clippy::too_many_arguments)]
fn sum(
    a: &TensorRef,
    out: &TensorRef,
    first: u32,
    count: u32,
    rt: usize,
    stage: u32,
    last_rows: u32,
    emit: &mut impl FnMut([u32; 8]) -> Result<(), u32>,
) -> Result<(), u32> {
    const SLOTS: usize = SUM_SLOTS;
    let slot = |i: usize| stage + (i as u64 * TILE_SLOT) as u32;
    // A list boundary of the old schedule: a wait before whatever comes next,
    // if anything has been emitted since the last one.
    let mut since = false;
    let mut pending = false;
    let mut put = |e: [u32; 8], pending: &mut bool, since: &mut bool| -> Result<(), u32> {
        if core::mem::take(pending) {
            emit([op::WAIT, 0, 0, 0, 0, 0, 0, 0])?;
        }
        *since = true;
        emit(e)
    };
    let flush = |pending: &mut bool, since: &mut bool| {
        if core::mem::take(since) {
            *pending = true;
        }
    };
    let mut used = 0usize;
    for j in first..first + count {
        if used + 1 + rt.min(SLOTS - 1) > SLOTS {
            flush(&mut pending, &mut since);
            used = 0;
        }
        let acc = slot(used);
        used += 1;
        let mut i0 = 0;
        while i0 < rt {
            let rows = (rt - i0).min(SLOTS - 1);
            if used + rows > SLOTS {
                // Only a column taller than a list reaches here: keep the
                // accumulator slot where it is.
                flush(&mut pending, &mut since);
                used = 1 + ((acc - stage) as u64 / TILE_SLOT) as usize;
            }
            for i in i0..i0 + rows {
                let (ch, off) = a.tile(i as u32, j)?;
                put(
                    [
                        op::READ,
                        ch,
                        (i % PORTS as usize) as u32,
                        off,
                        slot(used + i - i0),
                        TILE_SLOT as u32,
                        0,
                        0,
                    ],
                    &mut pending,
                    &mut since,
                )?;
            }
            for i in i0..i0 + rows {
                put(
                    [
                        op::COMPUTE,
                        super::kind::COL_SUM,
                        u32::from(i == 0),
                        if i == rt - 1 { last_rows } else { 0 },
                        acc,
                        slot(used + i - i0),
                        acc,
                        0,
                    ],
                    &mut pending,
                    &mut since,
                )?;
            }
            used += rows;
            i0 += SLOTS - 1;
        }
        let (ch, off) = out.tile(0, j)?;
        put(
            [
                op::WRITE,
                ch,
                0,
                off + TILE_DATA as u32,
                acc + TILE_DATA as u32,
                4096,
                0,
                0,
            ],
            &mut pending,
            &mut since,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn div_rem_is_division() {
        for (a, b) in [(0, 1), (7, 3), (u32::MAX, 7), (65_535, 8), (12, 12), (5, 9)] {
            assert_eq!(div_rem(a, b), (a / b, a % b), "{a} / {b}");
        }
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
