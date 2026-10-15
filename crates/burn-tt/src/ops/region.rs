//! Copy-shaped ops as one device job: `cat`, `repeat_dim` and `slice_assign`
//! (`docs/completed-plans/decode-host-cost.md`).
//!
//! Burn's default `cat` is a zero-filled tensor and one `slice_assign` a piece, and each
//! `slice_assign` maps every element of the whole destination on the host and copies all of it
//! (`slice_assign_native`): a `k`-way `cat` was `k + 1` jobs and `k` whole-destination maps. Here a
//! `cat` is one job:
//!
//! 1. **Rows**: a `cat` along any dimension but the last, of pieces that are whole rows of their
//!    buffers (a tensor is stored as `[leading dimensions, last]`), is one row gather
//!    (`tt_kernels::tensor::gather_rows`). The host names one source row per output row.
//! 2. **Elements**: anything else (a `cat` along the last dimension, a view with a column offset or
//!    a stride, BF16 and F16, whose packed buffers the row gather does not read) is one
//!    `repack_many` over every piece: the element map of today's path, built once, not once per
//!    piece.
//!
//! Both copy bits, so a result equals Burn's composition bit for bit. A piece in block floating
//! point storage, an empty result or a device without resident storage keeps Burn's composition
//! ([`cat_composed`]).
//!
//! `slice_assign` of a region of whole rows (the last dimension written in full, in order; a
//! key/value cache's positions) is a tile copy of the destination with the value's rows written over
//! it (`Session::assign_rows`): the host builds a few run records for the copy and entries for the
//! written rows only ([`assign_written`]). A destination that is a view is one row gather instead,
//! naming the destination's row outside the region and the value's inside ([`assign_rows`]); any
//! other region is the element map ([`assign_elements`]). The result is a new buffer every way: the
//! destination is never written in place, so a tensor that still holds it
//! (`cache.k.clone().slice_assign(..)`) keeps its contents.

use burn_backend::{Shape, TensorMetadata};

use super::{device_result_shaped, slice_assign_native};
use crate::tensor::{is_half, stored_dims, TtTensor};
use crate::views::Strided;

/// Each piece's view and shape, as a `cat` reads it.
type Piece<'a> = (&'a Strided, &'a [usize]);

/// The output shape of a `cat` of `shapes` along `dim`.
fn cat_shape(shapes: &[&[usize]], dim: usize) -> Vec<usize> {
    let mut out = shapes[0].to_vec();
    out[dim] = shapes.iter().map(|s| s[dim]).sum();
    out
}

/// Does `v` lay each row of a tensor of `shape` along a whole row of its buffer, from column 0?
fn whole_rows(v: &Strided, shape: &[usize]) -> bool {
    let Some(&last) = shape.last() else {
        return false;
    };
    v.src.buffer.cols == last
        && v.strides.last() == Some(&[0, 1])
        && v.base[1] == 0
        && (0..shape.len() - 1).all(|k| shape[k] == 1 || v.strides[k][1] == 0)
}

/// The source of each output row of a `cat` along `dim`: `(piece, row of that piece's buffer)`
/// for every stored row of the output, in order. `None` if a row of the output is not a whole row
/// of one piece's buffer: a `cat` along the last dimension, or a piece that is not
/// [`whole_rows`].
pub(crate) fn cat_rows(pieces: &[Piece<'_>], dim: usize) -> Option<Vec<(usize, usize)>> {
    let rank = pieces.first()?.1.len();
    if rank < 2 || dim + 1 >= rank || !pieces.iter().all(|(v, s)| whole_rows(v, s)) {
        return None;
    }
    let shapes: Vec<&[usize]> = pieces.iter().map(|(_, s)| *s).collect();
    let out = cat_shape(&shapes, dim);
    let outer: usize = out[..dim].iter().product();
    let last = out[rank - 1];
    let mut rows = Vec::with_capacity(out[..rank - 1].iter().product());
    for o in 0..outer {
        for (p, (v, s)) in pieces.iter().enumerate() {
            // The piece's rows for this outer index: its `dim` extent times the rows below it.
            let per: usize = s[dim..rank - 1].iter().product();
            for r in o * per..(o + 1) * per {
                rows.push((p, v.at(s, r * last)[0]));
            }
        }
    }
    Some(rows)
}

/// The source of each output element of a `cat` along `dim`: `(piece, [row, column] of that
/// piece's buffer)`, in row-major order.
pub(crate) fn cat_elements(pieces: &[Piece<'_>], dim: usize) -> Vec<(usize, [usize; 2])> {
    let shapes: Vec<&[usize]> = pieces.iter().map(|(_, s)| *s).collect();
    let out = cat_shape(&shapes, dim);
    let outer: usize = out[..dim].iter().product();
    let mut sources = Vec::with_capacity(out.iter().product());
    for o in 0..outer {
        for (p, (v, s)) in pieces.iter().enumerate() {
            let per: usize = s[dim..].iter().product();
            for f in o * per..(o + 1) * per {
                sources.push((p, v.at(s, f)));
            }
        }
    }
    sources
}

/// The output position, row-major in `shape`, of each element of a `slice_assign`'s value of shape
/// `vs`, whose dimension `d` lands at `ranges[d]`.
fn assigned_flat(shape: &[usize], ranges: &[Vec<usize>], vs: &[usize], mut flat: usize) -> usize {
    let mut coordinates = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        coordinates[d] = ranges[d][flat % vs[d]];
        flat /= vs[d];
    }
    coordinates
        .iter()
        .zip(shape)
        .fold(0, |n, (&i, &d)| n * d + i)
}

/// The source of each output element of `slice_assign(tensor, ranges, value)`: `(0, [row, column]
/// of the destination's buffer)` outside the region, `(1, .. of the value's)` inside, row-major.
pub(crate) fn assign_elements(
    dest: Piece<'_>,
    ranges: &[Vec<usize>],
    value: Piece<'_>,
) -> Vec<(usize, [usize; 2])> {
    let ((tv, shape), (vv, vs)) = (dest, value);
    let count: usize = shape.iter().product();
    let mut sources: Vec<_> = (0..count).map(|i| (0, tv.at(shape, i))).collect();
    for flat in 0..vs.iter().product() {
        sources[assigned_flat(shape, ranges, vs, flat)] = (1, vv.at(vs, flat));
    }
    sources
}

/// The rows a `slice_assign` writes: `(stored row of the output, row of the value's buffer)` for
/// each row of the value. `None` unless the region writes the last dimension whole and in order and
/// the value is [`whole_rows`].
pub(crate) fn assign_written(
    shape: &[usize],
    ranges: &[Vec<usize>],
    value: Piece<'_>,
) -> Option<Vec<(usize, usize)>> {
    let (vv, vs) = value;
    let last = *shape.last()?;
    if !ranges.last()?.iter().copied().eq(0..last) || !whole_rows(vv, vs) {
        return None;
    }
    Some(
        (0..vs[..vs.len() - 1].iter().product())
            .map(|r| {
                (
                    assigned_flat(shape, ranges, vs, r * last) / last,
                    vv.at(vs, r * last)[0],
                )
            })
            .collect(),
    )
}

/// [`assign_elements`] by rows: `(0, row of the destination's buffer)` outside the region, `(1, row
/// of the value's)` inside, for every stored row of the output. `None` unless the destination is
/// [`whole_rows`] and [`assign_written`] is not `None`.
pub(crate) fn assign_rows(
    dest: Piece<'_>,
    ranges: &[Vec<usize>],
    value: Piece<'_>,
) -> Option<Vec<(usize, usize)>> {
    let (tv, shape) = dest;
    if !whole_rows(tv, shape) {
        return None;
    }
    let written = assign_written(shape, ranges, value)?;
    let last = *shape.last()?;
    let count: usize = shape[..shape.len() - 1].iter().product();
    let mut rows: Vec<_> = (0..count).map(|r| (0, tv.at(shape, r * last)[0])).collect();
    for (d, s) in written {
        rows[d] = (1, s);
    }
    Some(rows)
}

/// `tensors` concatenated along `dim` as one device job, or `None` where Burn's composition is
/// kept: a piece in block floating point storage, a dtype or device that differs, an empty piece
/// or result, a scalar, or a device without resident storage.
pub(crate) fn cat_native(tensors: &[TtTensor], dim: usize) -> Option<TtTensor> {
    let first = tensors.first()?;
    let (device, dtype) = (first.device, first.dtype());
    if !tensors.iter().all(|t| {
        t.device == device
            && t.dtype() == dtype
            && t.is_storable()
            && t.storage_format() == crate::storage::StorageFormat::F32
            && dim < t.shape().num_dims()
    }) || !crate::server::supports_dram(device)
    {
        return None;
    }
    if let [one] = tensors {
        return Some(one.clone());
    }
    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape().to_vec()).collect();
    let views: Vec<Strided> = tensors
        .iter()
        .zip(&shapes)
        .map(|(t, s)| {
            t.as_strided()
                .unwrap_or_else(|| Strided::of(t.to_dram(), s))
        })
        .collect();
    let pieces: Vec<Piece<'_>> = views
        .iter()
        .zip(&shapes)
        .map(|(v, s)| (v, &s[..]))
        .collect();
    let out = cat_shape(&shapes.iter().map(|s| &s[..]).collect::<Vec<_>>(), dim);
    let dims = stored_dims(&out)?;
    // One source a buffer: the pieces of a `repeat_kv` are views of a few buffers.
    let mut buffers: Vec<crate::server::BufferId> = Vec::new();
    let index: Vec<usize> = views
        .iter()
        .map(|v| {
            let id = v.src.buffer.id;
            buffers.iter().position(|&b| b == id).unwrap_or_else(|| {
                buffers.push(id);
                buffers.len() - 1
            })
        })
        .collect();
    let rows = (!is_half(dtype)).then(|| cat_rows(&pieces, dim)).flatten();
    let (id, dims) = match rows {
        Some(rows) => {
            let rows = rows.into_iter().map(|(p, r)| (index[p], r)).collect();
            crate::server::gather_rows(device, buffers, rows, dims[1])
        }
        None => {
            let sources = cat_elements(&pieces, dim)
                .into_iter()
                .map(|(p, at)| (index[p], at))
                .collect();
            crate::server::repack_many(device, buffers, sources, dims)
        }
    };
    Some(device_result_shaped(
        device,
        id,
        dims,
        Shape::from(out),
        dtype,
    ))
}

/// Burn's own `cat`: `empty(shape)` and one `slice_assign` a piece (`cat_with_slice_assign` in
/// `burn-backend`), for what [`cat_native`] leaves.
pub(crate) fn cat_composed(
    tensors: Vec<TtTensor>,
    dim: usize,
    empty: impl FnOnce(Shape) -> TtTensor,
) -> TtTensor {
    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape().to_vec()).collect();
    let out = cat_shape(&shapes.iter().map(|s| &s[..]).collect::<Vec<_>>(), dim);
    let mut output = empty(Shape::from(out.clone()));
    let mut at = 0;
    for (t, s) in tensors.into_iter().zip(&shapes) {
        let slices: Vec<burn_backend::Slice> = out
            .iter()
            .enumerate()
            .map(|(k, &n)| {
                let (start, end) = if k == dim { (at, at + s[dim]) } else { (0, n) };
                burn_backend::Slice::new(start as isize, Some(end as isize), 1)
            })
            .collect();
        at += s[dim];
        if s.iter().product::<usize>() > 0 {
            output = slice_assign_native(output, &slices, t);
        }
    }
    output
}

/// `cat` for a `float_cat`, `int_cat` or `bool_cat`: one job where it can be, else Burn's
/// composition.
pub(crate) fn cat(
    tensors: Vec<TtTensor>,
    dim: usize,
    empty: impl FnOnce(Shape) -> TtTensor,
) -> TtTensor {
    match cat_native(&tensors, dim) {
        Some(t) => t,
        None => cat_composed(tensors, dim, empty),
    }
}

/// `repeat_dim`: `times` copies of `tensor` concatenated along `dim`, as Burn's default
/// (`repeat_with_slice_assign`) defines it.
pub(crate) fn repeat_dim(
    tensor: TtTensor,
    dim: usize,
    times: usize,
    empty: impl FnOnce(Shape) -> TtTensor,
) -> TtTensor {
    cat(vec![tensor; times], dim, empty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Buffer, DramRef};
    use std::sync::Arc;

    /// A plain view of a `[rows, cols]` buffer `id` nobody frees (a test-only device).
    fn view(id: u64, shape: &[usize]) -> Strided {
        let [rows, cols] = stored_dims(shape).unwrap();
        let buffer = Arc::new(Buffer {
            id,
            device: crate::TtDevice::new(u16::MAX),
            rows,
            cols,
            parent: None,
            storage: crate::storage::StorageFormat::F32,
            plain: Default::default(),
        });
        let v = Strided::of(
            &DramRef {
                buffer: buffer.clone(),
                transposed: false,
            },
            shape,
        );
        // Never dropped: dropping would free the buffer on a device no one attached.
        std::mem::forget(buffer);
        v
    }

    /// The element map of a `cat`, built the slow way: for each output coordinate, the piece
    /// whose range along `dim` holds it and that piece's element, as `slice_assign` would.
    fn reference(pieces: &[Piece<'_>], dim: usize) -> Vec<(usize, [usize; 2])> {
        let shapes: Vec<&[usize]> = pieces.iter().map(|(_, s)| *s).collect();
        let out = cat_shape(&shapes, dim);
        let n: usize = out.iter().product();
        (0..n)
            .map(|flat| {
                let mut coord = vec![0; out.len()];
                let mut f = flat;
                for k in (0..out.len()).rev() {
                    coord[k] = f % out[k];
                    f /= out[k];
                }
                let mut i = coord[dim];
                let p = shapes
                    .iter()
                    .position(|s| {
                        if i < s[dim] {
                            true
                        } else {
                            i -= s[dim];
                            false
                        }
                    })
                    .unwrap();
                coord[dim] = i;
                let s = shapes[p];
                let pf = coord.iter().zip(s).fold(0, |acc, (&c, &d)| acc * d + c);
                (p, pieces[p].0.at(s, pf))
            })
            .collect()
    }

    /// Expand a row map to the element map it moves.
    fn by_element(rows: &[(usize, usize)], cols: usize) -> Vec<(usize, [usize; 2])> {
        rows.iter()
            .flat_map(|&(p, r)| (0..cols).map(move |c| (p, [r, c])))
            .collect()
    }

    #[test]
    fn a_cat_off_the_last_dimension_is_whole_rows_and_moves_what_slice_assign_would() {
        for (shapes, dim) in [
            (
                vec![vec![2, 3, 5, 7], vec![2, 1, 5, 7], vec![2, 4, 5, 7]],
                1,
            ),
            (vec![vec![3, 4, 7], vec![1, 4, 7]], 0),
            (vec![vec![2, 3, 7], vec![2, 5, 7]], 1),
            (vec![vec![1, 1, 1, 64]; 9], 1),
            (vec![vec![4, 33], vec![40, 33]], 0),
        ] {
            let views: Vec<Strided> = shapes
                .iter()
                .enumerate()
                .map(|(i, s)| view(i as u64, s))
                .collect();
            let pieces: Vec<Piece<'_>> = views
                .iter()
                .zip(&shapes)
                .map(|(v, s)| (v, &s[..]))
                .collect();
            let rows = cat_rows(&pieces, dim).expect("whole rows");
            let last = *shapes[0].last().unwrap();
            assert_eq!(
                by_element(&rows, last),
                reference(&pieces, dim),
                "{shapes:?} {dim}"
            );
            assert_eq!(cat_elements(&pieces, dim), reference(&pieces, dim));
        }
    }

    /// A destination's shape, the value's shape, and where each value dimension lands.
    type Region = (Vec<usize>, Vec<usize>, Vec<Vec<usize>>);

    /// Regions along every dimension, written whole on the last.
    fn regions() -> Vec<Region> {
        vec![
            // A key/value cache `[1, 3, 8, 64]`, positions 5..7.
            (
                vec![1, 3, 8, 64],
                vec![1, 3, 2, 64],
                vec![vec![0], vec![0, 1, 2], vec![5, 6], (0..64).collect()],
            ),
            // Rows of a matrix, stepped and reversed.
            (
                vec![40, 33],
                vec![3, 33],
                vec![vec![38, 20, 2], (0..33).collect()],
            ),
            // A head of a rank-3 tensor.
            (
                vec![4, 5, 7],
                vec![1, 5, 7],
                vec![vec![2], (0..5).collect(), (0..7).collect()],
            ),
            // All of it: the value replaces the destination.
            (vec![2, 6], vec![2, 6], vec![vec![0, 1], (0..6).collect()]),
        ]
    }

    #[test]
    fn a_region_of_whole_rows_moves_what_the_element_map_does() {
        for (shape, vs, ranges) in regions() {
            let (dest, value) = (view(0, &shape), view(1, &vs));
            let rows = assign_rows((&dest, &shape), &ranges, (&value, &vs)).expect("whole rows");
            assert_eq!(
                by_element(&rows, *shape.last().unwrap()),
                assign_elements((&dest, &shape), &ranges, (&value, &vs)),
                "{shape:?} <- {vs:?}"
            );
        }
        // The value's elements land where the ranges say, the rest stays: checked once by hand.
        let (shape, vs) = (vec![3, 2], vec![1, 2]);
        let (dest, value) = (view(0, &shape), view(1, &vs));
        let ranges = vec![vec![1], vec![0, 1]];
        assert_eq!(
            assign_elements((&dest, &shape), &ranges, (&value, &vs)),
            vec![
                (0, [0, 0]),
                (0, [0, 1]),
                (1, [0, 0]),
                (1, [0, 1]),
                (0, [2, 0]),
                (0, [2, 1])
            ]
        );
    }

    #[test]
    fn a_region_that_does_not_write_whole_rows_is_elements() {
        let (shape, vs) = (vec![4, 64], vec![4, 32]);
        let (dest, value) = (view(0, &shape), view(1, &vs));
        let half = vec![(0..4).collect(), (32..64).collect()];
        assert_eq!(assign_rows((&dest, &shape), &half, (&value, &vs)), None);
        let reversed = vec![(0..4).collect(), (0..64).rev().collect()];
        let full = view(1, &shape);
        assert_eq!(
            assign_rows((&dest, &shape), &reversed, (&full, &shape)),
            None
        );
    }

    #[test]
    fn a_cat_along_the_last_dimension_is_elements() {
        let shapes = [vec![2, 3, 32], vec![2, 3, 32], vec![2, 3, 5]];
        let views: Vec<Strided> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| view(i as u64, s))
            .collect();
        let pieces: Vec<Piece<'_>> = views
            .iter()
            .zip(&shapes)
            .map(|(v, s)| (v, &s[..]))
            .collect();
        assert_eq!(cat_rows(&pieces, 2), None);
        assert_eq!(cat_elements(&pieces, 2), reference(&pieces, 2));
    }

    #[test]
    fn views_are_read_where_they_lie_and_only_whole_rows_go_by_row() {
        // A head of a `[1, 3, 8, 64]` cache, `[1, 1, 5, 64]`: whole rows at an offset.
        let cache = view(0, &[1, 3, 8, 64]);
        let mut head = cache.clone();
        head.base = [8, 0];
        // A slice of the last dimension, `[2, 4, 32]` of a `[2, 4, 64]`: rows from column 32.
        let wide = view(1, &[2, 4, 64]);
        let mut half = wide.clone();
        half.base = [0, 32];
        // A transposed `[5, 7]` of a `[7, 5]` buffer.
        let t = view(2, &[7, 5]).swapped(0, 1);

        let head_shape = [1, 1, 5, 64];
        let pieces: Vec<Piece<'_>> = vec![(&head, &head_shape), (&head, &head_shape)];
        let rows = cat_rows(&pieces, 1).expect("a head is whole rows");
        assert_eq!(rows[0], (0, 8));
        assert_eq!(by_element(&rows, 64), reference(&pieces, 1));

        let half_shape = [2, 4, 32];
        let pieces: Vec<Piece<'_>> = vec![(&half, &half_shape), (&half, &half_shape)];
        assert_eq!(
            cat_rows(&pieces, 1),
            None,
            "a column offset is not whole rows"
        );
        assert_eq!(cat_elements(&pieces, 1), reference(&pieces, 1));

        let t_shape = [5, 7];
        let plain = view(3, &[2, 7]);
        let pieces: Vec<Piece<'_>> = vec![(&t, &t_shape), (&plain, &[2, 7])];
        assert_eq!(cat_rows(&pieces, 0), None, "a transpose is not whole rows");
        assert_eq!(cat_elements(&pieces, 0), reference(&pieces, 0));
    }
}
