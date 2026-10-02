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
/// reduction over rows reads a column of tiles in.
pub const READ_RUN: u32 = 0x15;
/// A run of whole tiles' datums, L1 -> GDDR: `[WRITE_RUN, first, count, at,
/// 0, 0, 0, 0]` + `X`. Slot `n` from `at` (its datums, past the header) to
/// tile `first + n` of `X`, row-major. What a kernel's packer wrote goes back.
pub const WRITE_RUN: u32 = 0x16;

/// Most tiles a record may name along any one of its loops: far more than any
/// list holds, and few enough that a corrupt record cannot keep the mover busy
/// for long.
pub const MAX_EXTENT: u32 = 1 << 16;

/// Entries in the record whose header word 0 is `op`: 1 for a plain entry.
pub const fn len(op: u32) -> usize {
    match op {
        GATHER => 5,
        SCATTER | FILL_PAD | READ_RUN | WRITE_RUN => 3,
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
        READ_RUN | WRITE_RUN => {
            let [_, first, count, at, flags, grid_ct, grid_rt, _] = h;
            let count = extent(count)?;
            let x = tensor(rec, 1)?;
            let read = h[0] == READ_RUN;
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
            let (mut i, mut j) = if column_major {
                let (q, r) = div_rem(first, grid_rt);
                (r, q)
            } else {
                div_rem(first, ct)
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
                let (ch, off) = x.tile(if row { 0 } else { i }, if col { 0 } else { j })?;
                let op = if col {
                    op::READ_BROADCAST_COL
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
                emit([op::FILL, value, vr | vc << 8, at, 0, 0, 0, 0])?;
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
