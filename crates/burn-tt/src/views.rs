//! Strided views of device buffers: reshapes and dimension swaps that move
//! nothing.
//!
//! A device buffer is a matrix (`tensor::stored_dims`). A view says where
//! each element of a tensor of any shape lies in it: element `idx` is at
//! `base + sum_k idx_k strides[k]`, a row and a column. A plain tensor is the
//! view with the row-major strides of its own stored matrix. Attention's head
//! split -- `[b, s, d]` reshaped to `[b, s, h, dk]` and swapped to `[b, h, s,
//! dk]` -- is a view of the projection's `[b s, d]` buffer, each head the
//! `[s, dk]` block at `[i s, j dk]`; `K^T` is the same blocks read
//! transposed. A batched matmul reads such blocks where they lie
//! (`Session::matmul_dram_batched`). Anything else that needs the tensor as
//! a plain matrix materialises it: on the card by whole-tile moves
//! (`Session::copy_blocks`) when the view is tile-coherent, else through
//! the host.

use crate::tensor::{stored_dims, DramRef};
use crate::{Block, BlockMove};

/// A view of `src`'s buffer, which is never itself a transposed view (a
/// transpose is in the strides).
#[derive(Clone, Debug)]
pub(crate) struct Strided {
    pub(crate) src: DramRef,
    pub(crate) base: [usize; 2],
    pub(crate) strides: Vec<[usize; 2]>,
}

/// The row-major strides of a tensor of `shape` stored as
/// `stored_dims(shape)`: the last dimension along a row, every other one
/// down the rows.
pub(crate) fn plain_strides(shape: &[usize]) -> Vec<[usize; 2]> {
    let n = shape.len();
    let mut s = vec![[0, 0]; n];
    if n == 0 {
        return s;
    }
    s[n - 1] = [0, 1];
    let mut rows = 1;
    for k in (0..n - 1).rev() {
        s[k] = [rows, 0];
        rows *= shape[k];
    }
    s
}

impl Strided {
    /// A device copy of a tensor of `shape` as a view.
    pub(crate) fn of(d: &DramRef, shape: &[usize]) -> Strided {
        let mut strides = plain_strides(shape);
        if d.transposed {
            // Only a rank-2 view is ever transposed: element `(i, j)` is the
            // buffer's `(j, i)`.
            debug_assert_eq!(shape.len(), 2);
            strides = vec![[0, 1], [1, 0]];
        }
        Strided {
            src: DramRef {
                buffer: d.buffer.clone(),
                transposed: false,
            },
            base: [0, 0],
            strides,
        }
    }

    fn src_dims(&self) -> [usize; 2] {
        [self.src.buffer.rows, self.src.buffer.cols]
    }

    /// Does this view cover every logical source element exactly once?
    /// Reshapes and swaps can change the stored matrix (a loss's `[n, 1]`
    /// column viewed as `[n]`) without changing a full reduction's inputs.
    pub(crate) fn covers_source(&self, shape: &[usize]) -> bool {
        if self.base != [0, 0] || shape.len() != self.strides.len() || shape.contains(&0) {
            return false;
        }
        let mut dimensions = [Vec::new(), Vec::new()];
        for (&n, &stride) in shape.iter().zip(&self.strides) {
            if n == 1 {
                continue;
            }
            match stride {
                [r, 0] if r > 0 => dimensions[0].push((r, n)),
                [0, c] if c > 0 => dimensions[1].push((c, n)),
                _ => return false,
            }
        }
        let mut covered = [1, 1];
        for (axis, dims) in dimensions.iter_mut().enumerate() {
            dims.sort_unstable();
            for &(stride, n) in dims.iter() {
                if stride != covered[axis] {
                    return false;
                }
                let Some(extent) = covered[axis].checked_mul(n) else {
                    return false;
                };
                covered[axis] = extent;
            }
        }
        covered == self.src_dims()
    }

    /// The element at row-major position `flat` of a tensor of `shape`.
    pub(crate) fn at(&self, shape: &[usize], mut flat: usize) -> [usize; 2] {
        let [mut r, mut c] = self.base;
        for k in (0..shape.len()).rev() {
            let i = flat % shape[k];
            flat /= shape[k];
            r += i * self.strides[k][0];
            c += i * self.strides[k][1];
        }
        [r, c]
    }

    /// The view as a plain device copy, if it is one: `src` itself, all of
    /// it, in row-major order -- or, at rank 2, all of it transposed.
    pub(crate) fn as_plain(&self, shape: &[usize]) -> Option<DramRef> {
        if self.base != [0, 0] {
            return None;
        }
        let dims = self.src_dims();
        let same = |s: &[[usize; 2]]| -> bool {
            s.iter()
                .zip(&self.strides)
                .zip(shape)
                .all(|((a, b), &n)| n == 1 || a == b)
        };
        if stored_dims(shape) == Some(dims) && same(&plain_strides(shape)) {
            return Some(DramRef {
                buffer: self.src.buffer.clone(),
                transposed: false,
            });
        }
        if shape.len() == 2 && [shape[1], shape[0]] == dims && same(&[[0, 1], [1, 0]]) {
            return Some(DramRef {
                buffer: self.src.buffer.clone(),
                transposed: true,
            });
        }
        None
    }

    /// Dimensions `a` and `b` swapped.
    pub(crate) fn swapped(&self, a: usize, b: usize) -> Strided {
        let mut v = self.clone();
        v.strides.swap(a, b);
        v
    }

    /// The view of the same elements as a tensor of `to`, if the reshape
    /// keeps them strided (PyTorch's `computeStride`, over the two
    /// coordinates of a matrix): each run of dimensions that `from`'s
    /// strides make contiguous may be split and merged freely.
    pub(crate) fn reshaped(&self, from: &[usize], to: &[usize]) -> Option<Strided> {
        if from.contains(&0) || to.contains(&0) {
            return None;
        }
        if from.is_empty() || to.is_empty() {
            return None;
        }
        let s = &self.strides;
        let mut out = vec![[0usize, 0]; to.len()];
        let scale = |x: [usize; 2], n: usize| [x[0] * n, x[1] * n];
        let mut view_d = to.len() as isize - 1;
        let mut chunk = s[from.len() - 1];
        let (mut tensor_numel, mut view_numel) = (1usize, 1usize);
        for d in (0..from.len()).rev() {
            tensor_numel *= from[d];
            if d == 0 || (from[d - 1] != 1 && s[d - 1] != scale(chunk, tensor_numel)) {
                while view_d >= 0 && (view_numel < tensor_numel || to[view_d as usize] == 1) {
                    out[view_d as usize] = scale(chunk, view_numel);
                    view_numel *= to[view_d as usize];
                    view_d -= 1;
                }
                if view_numel != tensor_numel {
                    return None;
                }
                if d > 0 {
                    chunk = s[d - 1];
                    tensor_numel = 1;
                    view_numel = 1;
                }
            }
        }
        if view_d != -1 {
            return None;
        }
        Some(Strided {
            src: self.src.clone(),
            base: self.base,
            strides: out,
        })
    }

    /// Is a `[r, c]` block at `at` of the source one the card can read where
    /// it lies: on tile boundaries, ragged only at the buffer's own edge?
    /// (`tt_kernels::tensor::block_ref`'s rule, checked here so an unfit view
    /// takes another path rather than an error.)
    fn block_fits(&self, [r0, c0]: [usize; 2], [r, c]: [usize; 2]) -> bool {
        let [rows, cols] = self.src_dims();
        r0 + r <= rows
            && c0 + c <= cols
            && r0 % 32 == 0
            && c0 % 32 == 0
            && (r % 32 == 0 || r0 + r == rows)
            && (c % 32 == 0 || c0 + c == cols)
    }

    /// The block a batched matmul reads for each index of `batch` (a tensor
    /// of `shape` is `[lead.., r, c]`, `lead` broadcast to `batch` from the
    /// right): its first element, and whether it is read transposed. `None`
    /// unless the last two dimensions are a block of the source (or its
    /// transpose) and every block fits ([`Strided::block_fits`]).
    pub(crate) fn matrix_blocks(&self, shape: &[usize], batch: &[usize]) -> Option<Vec<Block>> {
        let n = shape.len();
        if n < 2 || batch.len() < n - 2 {
            return None;
        }
        let [r, c] = [shape[n - 2], shape[n - 1]];
        let ok = |k: usize, want: [usize; 2]| shape[k] == 1 || self.strides[k] == want;
        let transposed = if ok(n - 2, [1, 0]) && ok(n - 1, [0, 1]) {
            false
        } else if ok(n - 2, [0, 1]) && ok(n - 1, [1, 0]) {
            true
        } else {
            return None;
        };
        let extent = if transposed { [c, r] } else { [r, c] };
        let lead = n - 2;
        let pad = batch.len() - lead;
        let count: usize = batch.iter().product();
        let mut blocks = Vec::with_capacity(count);
        for flat in 0..count {
            let [mut r0, mut c0] = self.base;
            let mut rest = flat;
            for k in (0..batch.len()).rev() {
                let i = rest % batch[k];
                rest /= batch[k];
                if k >= pad && shape[k - pad] != 1 {
                    r0 += i * self.strides[k - pad][0];
                    c0 += i * self.strides[k - pad][1];
                }
            }
            if !self.block_fits([r0, c0], extent) {
                return None;
            }
            blocks.push(Block {
                at: [r0, c0],
                transposed,
            });
        }
        Some(blocks)
    }

    /// Whole-tile moves that build the plain matrix of a tensor of `to`
    /// holding this view's elements (of a tensor of `from`) in row-major
    /// order -- `to` may be `from` itself, or a reshape no strides express,
    /// such as attention's heads merged back into `[b, s, d]`. Each output
    /// tile must be one source tile, or one source tile transposed: the
    /// output whole tiles, the view's last dimension a run of whole tiles
    /// along the source's rows (or, transposed, down its columns), and each
    /// output tile's rows the source's next rows (columns). Adjacent tiles
    /// that are adjacent in the source too are one move.
    pub(crate) fn tile_moves(&self, from: &[usize], to: &[usize]) -> Option<Vec<BlockMove>> {
        let [p, c] = stored_dims(to)?;
        let last = *from.last()?;
        let transposed = match self.strides.last()? {
            [0, 1] => false,
            [1, 0] => true,
            _ => return None,
        };
        if p % 32 != 0 || c % 32 != 0 || last % 32 != 0 {
            return None;
        }
        // Along an output row, and down an output column, in the source.
        let (along, down) = if transposed { (0, 1) } else { (1, 0) };
        let step = |s: [usize; 2], axis: usize, n: usize| {
            let mut s = s;
            s[axis] += n;
            s
        };
        let mut moves: Vec<BlockMove> = Vec::new();
        for ti in 0..p / 32 {
            for tj in 0..c / 32 {
                let f0 = ti * 32 * c + tj * 32;
                let s0 = self.at(from, f0);
                if !self.block_fits(s0, [32, 32]) {
                    return None;
                }
                if (1..32).any(|a| self.at(from, f0 + a * c) != step(s0, down, a)) {
                    return None;
                }
                let to_at = [ti * 32, tj * 32];
                match moves.last_mut() {
                    Some(m)
                        if m.to[0] == to_at[0]
                            && m.to[1] + m.extent[1] == to_at[1]
                            && step(m.from, along, m.extent[1]) == s0 =>
                    {
                        m.extent[1] += 32;
                    }
                    _ => moves.push(BlockMove {
                        from: s0,
                        to: to_at,
                        extent: [32, 32],
                        transposed,
                    }),
                }
            }
        }
        Some(moves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Buffer, DramRef};
    use std::sync::Arc;

    #[test]
    fn full_source_coverage_requires_each_element_once() {
        let plain = view(64, 96, &[64, 96]);
        assert!(plain.covers_source(&[64, 96]));
        let split = plain.reshaped(&[64, 96], &[2, 32, 3, 32]).unwrap();
        for a in 0..4 {
            for b in 0..4 {
                let mut shape = [2, 32, 3, 32];
                shape.swap(a, b);
                assert!(split.swapped(a, b).covers_source(&shape));
            }
        }
        assert!(view(128, 1, &[128, 1])
            .reshaped(&[128, 1], &[128])
            .unwrap()
            .covers_source(&[128]));
        assert!(!plain.covers_source(&[32, 96]));
        let mut invalid = plain.clone();
        invalid.base = [1, 0];
        assert!(!invalid.covers_source(&[64, 96]));
        invalid.base = [0, 0];
        invalid.strides[0] = [0, 0];
        assert!(!invalid.covers_source(&[64, 96]));
        invalid.strides[0] = [2, 0];
        assert!(!invalid.covers_source(&[64, 96]));
    }

    /// A view of a `[rows, cols]` buffer nobody frees (a test-only device).
    pub(super) fn view(rows: usize, cols: usize, shape: &[usize]) -> Strided {
        let buffer = Arc::new(Buffer {
            id: 0,
            device: crate::TtDevice::new(u16::MAX),
            rows,
            cols,
            parent: None,
        });
        let v = Strided::of(
            &DramRef {
                buffer: buffer.clone(),
                transposed: false,
            },
            shape,
        );
        // Never dropped: dropping would free buffer 0 on a device no one
        // attached.
        std::mem::forget(buffer);
        v
    }

    /// Every element of a view, by the definition: unravel, then strides.
    fn elements(v: &Strided, shape: &[usize]) -> Vec<[usize; 2]> {
        (0..shape.iter().product())
            .map(|f| v.at(shape, f))
            .collect()
    }

    #[test]
    fn heads_split_by_a_reshape_and_a_swap_are_blocks_of_the_projection() {
        let (b, s, h, dk) = (2, 64, 2, 32);
        let x = view(b * s, h * dk, &[b, s, h * dk]);
        let split = x.reshaped(&[b, s, h * dk], &[b, s, h, dk]).unwrap();
        assert_eq!(split.strides, vec![[s, 0], [1, 0], [0, dk], [0, 1]]);
        let heads = split.swapped(1, 2);
        let blocks = heads.matrix_blocks(&[b, h, s, dk], &[b, h]).unwrap();
        let at: Vec<_> = blocks.iter().map(|x| (x.at, x.transposed)).collect();
        assert_eq!(
            at,
            vec![
                ([0, 0], false),
                ([0, 32], false),
                ([64, 0], false),
                ([64, 32], false)
            ]
        );
        // `K^T`: the same blocks, read transposed.
        let kt = heads.swapped(2, 3);
        let blocks = kt.matrix_blocks(&[b, h, dk, s], &[b, h]).unwrap();
        assert!(blocks.iter().all(|x| x.transposed));
        assert_eq!(blocks[3].at, [64, 32]);
    }

    #[test]
    fn a_reshape_the_strides_cannot_express_is_refused_and_tile_moves_build_it() {
        let (b, s, h, dk) = (2, 64, 2, 32);
        // A batched product's output, `[b h s, dk]`, as `[b, h, s, dk]`,
        // swapped to `[b, s, h, dk]` and merged to `[b, s, d]`.
        let ctx = view(b * h * s, dk, &[b, h, s, dk]).swapped(1, 2);
        let from = [b, s, h, dk];
        assert!(ctx.reshaped(&from, &[b, s, h * dk]).is_none());
        let moves = ctx.tile_moves(&from, &[b, s, h * dk]).unwrap();
        // Every output element comes from the view's element in that
        // row-major position.
        let want = elements(&ctx, &from);
        assert_eq!(sources(&moves, h * dk, want.len()), want);
        // One move per head and batch element's 2 tile rows.
        assert_eq!(moves.len(), b * h * (s / 32));
    }

    /// Where each output element of `moves` comes from, by the moves.
    fn sources(moves: &[BlockMove], c: usize, n: usize) -> Vec<[usize; 2]> {
        let mut got = vec![[usize::MAX; 2]; n];
        for m in moves {
            for i in 0..m.extent[0] {
                for j in 0..m.extent[1] {
                    got[(m.to[0] + i) * c + m.to[1] + j] = if m.transposed {
                        [m.from[0] + j, m.from[1] + i]
                    } else {
                        [m.from[0] + i, m.from[1] + j]
                    };
                }
            }
        }
        got
    }

    #[test]
    fn a_view_of_transposed_blocks_is_built_by_transposing_tile_moves() {
        // `K`'s gradient through `K^T`: `[b h dk, s]` stacked products as
        // `[b, h, dk, s]`, transposed back to `[b, h, s, dk]`, swapped to
        // `[b, s, h, dk]` and merged to `[b, s, d]`.
        let (b, s, h, dk) = (2, 64, 2, 32);
        let g = view(b * h * dk, s, &[b, h, dk, s])
            .swapped(2, 3)
            .swapped(1, 2);
        let from = [b, s, h, dk];
        let to = [b, s, h * dk];
        let merged = g.reshaped(&from, &to).unwrap_or_else(|| g.clone());
        let from = if merged.strides.len() == 3 {
            &to[..]
        } else {
            &from[..]
        };
        let moves = merged.tile_moves(from, &to).unwrap();
        assert!(moves.iter().all(|m| m.transposed));
        assert_eq!(
            sources(&moves, h * dk, b * s * h * dk),
            elements(&merged, from)
        );
    }

    #[test]
    fn a_restrided_view_names_the_same_elements() {
        for (from, to) in [
            (vec![4, 6, 8], vec![24, 8]),
            (vec![4, 6, 8], vec![4, 48]),
            (vec![4, 6, 8], vec![4, 6, 2, 4]),
            (vec![2, 1, 3, 4], vec![6, 4]),
        ] {
            let v = view(
                from[..from.len() - 1].iter().product(),
                from[from.len() - 1],
                &from,
            );
            match v.reshaped(&from, &to) {
                Some(r) => assert_eq!(elements(&r, &to), elements(&v, &from), "{from:?} -> {to:?}"),
                None => assert_eq!(to, vec![4, 48], "{from:?} -> {to:?} refused"),
            }
        }
    }

    #[test]
    fn a_plain_view_is_recognised_and_a_transpose_too() {
        let v = view(64, 32, &[2, 32, 32]);
        assert!(v.as_plain(&[2, 32, 32]).is_some());
        let t = view(64, 32, &[64, 32]).swapped(0, 1);
        assert!(t.as_plain(&[32, 64]).unwrap().transposed);
        assert!(v.swapped(1, 2).as_plain(&[2, 32, 32]).is_none());
    }
}
