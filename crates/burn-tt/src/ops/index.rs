//! Indexing compositions: `gather_nd`, `scatter_nd`, `cross` and integer matmul.
//!
//! Every method here is a composition of resident primitives; the host builds
//! address metadata only and never sees a tensor value.
//!
//! # `gather_nd` / `scatter_nd`
//!
//! `indices` has shape `[..., K]`, `data` has shape `[d_0, ..., d_{D-1}]` and the
//! last axis of `indices` names the leading `K` axes of `data`; the slice
//! `data[i_0, ..., i_{K-1}]` has the remaining `D - K` axes. Both ops view `data`
//! as `[P, R]` (`P = d_0 * ... * d_{K-1}`, `R = d_K * ... * d_{D-1}`) and the batch
//! of `N = prod(indices.shape[..M-1])` index tuples as `N` rows of it.
//!
//! The flat row `sum_j i_j * stride_j` is computed on the device with the native
//! integer ops. **Every coordinate is bounds-checked against its own axis**, not
//! only the flat row: coordinate `j` is read back through the checked resident
//! gather (`Session::gather_indexed`, DOMAIN code 11 outside `[0, d_j)`) from a
//! device-side `0..d_j` table, and the flat row is built from those checked
//! copies. A tuple like `[0, 5]` into a `[3, 4]` tensor therefore fails although
//! its flat row `5` is inside `[12]`. Flex (the oracle for in-range tuples) does
//! no per-coordinate check: its flat offset wraps negatives into a panic and lets
//! a coordinate overflow its axis into the next one.
//!
//! * `gather_nd` selects the `N` rows of `[P, R]` with the resident select
//!   (raw datums, any F32/BF16/I32 payload bit-exact).
//! * `scatter_nd(Add)` is the resident `select_add` over `[P, R]`: duplicates
//!   fold in index order, `((data + v_0) + v_1) + ...`, as Flex's sequential
//!   `+=` does.
//! * `scatter_nd(Assign)`: **the last writer in index order wins** (Flex's
//!   sequential overwrite). Each index tuple, in order, replaces its row through
//!   a mask select, so a later tuple replaces an earlier one by construction.
//! * `scatter_nd(Mul | Min | Max)` fail naming the operation and the variant: no
//!   gated resident arithmetic exists for them with Flex's duplicate order, and
//!   Burn documents duplicates as undefined for them. `[-]`.
//!
//! # `float_cross`
//!
//! Flex's own composition: three slices of each operand along `dim`, six
//! products, three differences, a concatenation. The rounding is the device's
//! (`SFPMAD` mul and add, not fused, denormals flushed).
//!
//! # `int_matmul`
//!
//! `[.., M, K] x [.., K, N]` is an `I32` product `[.., M, K, N]` summed over `K`.
//! Wrapping `i32` multiplication and addition are associative and commutative,
//! so the result is exact modulo 2^32 in whatever order the reduction folds.
//! Nothing goes through `f32`. The intermediate is bounded by
//! [`INT_MATMUL_BUDGET`] elements; above it the op fails naming the shapes.

use super::*;
use burn_backend::tensor::IndexingUpdateOp;

/// The most elements `int_matmul` materializes: the broadcast product
/// `batch * M * K * N` (each operand copy and the product are that large, in
/// GDDR; the host holds their address lists).
pub const INT_MATMUL_BUDGET: usize = 1 << 22;

/// Multi-dimensional gather: see the module documentation.
pub fn float_gather_nd(data: TtTensor, indices: TtTensor) -> TtTensor {
    gather_nd(data, indices, "float_gather_nd")
}

/// Multi-dimensional scatter: `Add` and `Assign` only.
pub fn float_scatter_nd(
    data: TtTensor,
    indices: TtTensor,
    values: TtTensor,
    reduction: IndexingUpdateOp,
) -> TtTensor {
    scatter_nd(data, indices, values, reduction, "float_scatter_nd")
}

/// The cross product of two tensors along `dim`, which has size 3 in both.
pub fn float_cross(lhs: TtTensor, rhs: TtTensor, dim: usize) -> TtTensor {
    cross(lhs, rhs, dim)
}

pub fn int_gather_nd(data: TtTensor, indices: TtTensor) -> TtTensor {
    gather_nd(data, indices, "int_gather_nd")
}

pub fn int_scatter_nd(
    data: TtTensor,
    indices: TtTensor,
    values: TtTensor,
    reduction: IndexingUpdateOp,
) -> TtTensor {
    scatter_nd(data, indices, values, reduction, "int_scatter_nd")
}

/// Integer matrix product, exact modulo 2^32.
pub fn int_matmul(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
    int_matmul_composed(lhs, rhs)
}

/// `tensor` as `shape`: a free view when the stored matrix is the same,
/// otherwise a device repack of its logical elements in order. (A reshape that
/// changes the stored matrix of an integer tensor has no view form.)
fn restack(tensor: &TtTensor, shape: Vec<usize>) -> TtTensor {
    let from = tensor.shape().to_vec();
    if from == shape {
        return tensor.clone();
    }
    let free = crate::tensor::stored_dims(&from) == crate::tensor::stored_dims(&shape)
        && !tensor.is_view()
        && !tensor.is_transposed();
    if free && tensor.is_storable() {
        return reshaped(tensor.clone(), shape.into());
    }
    let count: usize = shape.iter().product();
    mapped_native(&[tensor], (0..count).map(|i| (0, i)).collect(), shape)
}

/// Geometry shared by the `_nd` ops, validated.
struct Nd {
    /// Index tuples: `prod(indices.shape[..M-1])`.
    n: usize,
    /// Leading data axes the tuples name.
    k: usize,
    /// `prod(data.shape[..k])`.
    rows: usize,
    /// `prod(data.shape[k..])`.
    rest: usize,
    /// Row stride of each named axis.
    strides: Vec<usize>,
    /// `indices.shape[..M-1] ++ data.shape[k..]`.
    out_shape: Vec<usize>,
}

fn nd_geometry(name: &str, data: &TtTensor, indices: &TtTensor, extra: &str) -> Nd {
    let shape = data.shape().to_vec();
    let idx = indices.shape().to_vec();
    let refuse = |why: &str| -> ! {
        fail(
            name,
            format_args!(
                "{why}: data=({}), indices=({}){extra}",
                context(data),
                context(indices)
            ),
        )
    };
    if indices.dtype() != DType::I32 {
        refuse("indices must be I32");
    }
    if data.device != indices.device
        || !data.is_storable()
        || !indices.is_storable()
        || !crate::server::supports_dram(data.device)
    {
        refuse("tensors must be resident-storable on one device with GDDR");
    }
    if idx.is_empty() || shape.is_empty() || idx.contains(&0) || shape.contains(&0) {
        refuse("empty tensors and rank-0 indices are not supported");
    }
    let k = idx[idx.len() - 1];
    if k == 0 || k > shape.len() {
        refuse("the last index axis K must satisfy 1 <= K <= data rank");
    }
    let mut out_shape = idx[..idx.len() - 1].to_vec();
    out_shape.extend_from_slice(&shape[k..]);
    if out_shape.is_empty() {
        refuse("a rank-0 result is not supported");
    }
    let rows: usize = shape[..k].iter().product();
    if rows > i32::MAX as usize {
        refuse("the named axes exceed the I32 index range");
    }
    let mut strides = vec![1; k];
    for j in (0..k.saturating_sub(1)).rev() {
        strides[j] = strides[j + 1] * shape[j + 1];
    }
    Nd {
        n: idx[..idx.len() - 1].iter().product(),
        k,
        rows,
        rest: shape[k..].iter().product(),
        strides,
        out_shape,
    }
}

/// The flat row of every index tuple, `[N]` `I32`, each coordinate checked
/// against its own axis on the device (module documentation).
fn nd_flat_rows(data: &TtTensor, indices: &TtTensor, nd: &Nd) -> TtTensor {
    let device = data.device;
    let shape = data.shape().to_vec();
    let mut flat: Option<TtTensor> = None;
    for (j, &extent) in shape.iter().enumerate().take(nd.k) {
        let coordinate = mapped_native(
            &[indices],
            (0..nd.n).map(|i| (0, i * nd.k + j)).collect(),
            vec![nd.n],
        );
        // The checked resident gather is the domain test: the table holds
        // 0..extent, so the copy equals the coordinate and anything outside
        // [0, extent) is a DOMAIN fault before any datum moves.
        let id = crate::server::metadata(
            device,
            (0..extent).map(|i| i as u32).collect(),
            [1, extent],
            Elem::I32,
        );
        let table = device_result_shaped(device, id, [1, extent], vec![extent].into(), DType::I32);
        let checked = gather_native(0, table, coordinate);
        let term = if nd.strides[j] == 1 {
            checked
        } else {
            crate::ops::int::int_mul_scalar(checked, (nd.strides[j] as i32).into())
        };
        flat = Some(match flat {
            None => term,
            Some(sum) => crate::ops::int::int_add(sum, term),
        });
    }
    flat.expect("K >= 1")
}

fn gather_nd(data: TtTensor, indices: TtTensor, name: &str) -> TtTensor {
    if !matches!(data.dtype(), DType::F32 | DType::BF16 | DType::I32) {
        fail(name, format_args!("data=({})", context(&data)));
    }
    let nd = nd_geometry(name, &data, &indices, "");
    let flat = nd_flat_rows(&data, &indices, &nd);
    let table = restack(&data, vec![nd.rows, nd.rest]);
    let picked = select_native(table, 0, flat);
    restack(&picked, nd.out_shape)
}

fn scatter_nd(
    data: TtTensor,
    indices: TtTensor,
    values: TtTensor,
    reduction: IndexingUpdateOp,
    name: &str,
) -> TtTensor {
    let variant = format!(", reduction={reduction:?}");
    let dtype = data.dtype();
    if !matches!(dtype, DType::F32 | DType::I32) || values.dtype() != dtype {
        fail(
            name,
            format_args!(
                "data=({}), values=({}){variant}",
                context(&data),
                context(&values)
            ),
        );
    }
    if !matches!(reduction, IndexingUpdateOp::Add | IndexingUpdateOp::Assign) {
        fail(
            name,
            format_args!(
                "reduction {reduction:?} is not supported (only Add and Assign; for unique \
                 indices compose gather_nd, the arithmetic and an Assign scatter_nd): \
                 data=({}), indices=({}), values=({})",
                context(&data),
                context(&indices),
                context(&values)
            ),
        );
    }
    let nd = nd_geometry(name, &data, &indices, &variant);
    if values.shape().to_vec() != nd.out_shape
        || values.device != data.device
        || !values.is_storable()
    {
        fail(
            name,
            format_args!(
                "values must have shape {:?} (indices batch ++ data tail) on the data's \
                 device: data=({}), indices=({}), values=({}){variant}",
                nd.out_shape,
                context(&data),
                context(&indices),
                context(&values)
            ),
        );
    }
    let flat = nd_flat_rows(&data, &indices, &nd);
    let table = restack(&data, vec![nd.rows, nd.rest]);
    let rows = restack(&values, vec![nd.n, nd.rest]);
    let updated = match reduction {
        IndexingUpdateOp::Add => select_add_native(table, 0, flat, rows),
        _ => scatter_assign_rows(table, flat, rows),
    };
    restack(&updated, data.shape().to_vec())
}

/// `table` (`[P, R]`) with row `flat[j]` replaced by `rows[j]` for `j = 0..N`
/// in order, so the last writer of a duplicated row wins. `flat` is already
/// domain-checked. Raw datums move (a mask select, never arithmetic).
fn scatter_assign_rows(table: TtTensor, flat: TtTensor, rows: TtTensor) -> TtTensor {
    let device = table.device;
    let dtype = table.dtype();
    let [p, rest] = [table.shape()[0], table.shape()[1]];
    let n = rows.shape()[0];
    let id = crate::server::metadata(
        device,
        (0..p).map(|i| i as u32).collect(),
        [p, 1],
        Elem::I32,
    );
    let positions = device_result_shaped(device, id, [p, 1], vec![p, 1].into(), DType::I32);
    let positions = expanded(&positions, vec![p, rest].into(), "scatter_nd rows");
    let mut current = table;
    for j in 0..n {
        let target = mapped_native(&[&flat], vec![(0, j)], vec![1, 1]);
        let target = expanded(&target, vec![p, rest].into(), "scatter_nd target");
        let mask = device_eltwise_ungated(kind_sfpu::INT_EQ, 0.0, &positions, Some(&target))
            .expect("integer scatter equality");
        let value = mapped_native(
            &[&rows],
            (0..rest).map(|c| (0, j * rest + c)).collect(),
            vec![1, rest],
        );
        let value = expanded(&value, vec![p, rest].into(), "scatter_nd value");
        current = match dtype {
            DType::I32 => {
                let mask = crate::ops::int::int_mul_scalar(
                    crate::ops::bool::bool_into_int(mask, IntDType::I32),
                    (-1i32).into(),
                );
                let old = crate::ops::int::bitwise_and(
                    current,
                    crate::ops::int::bitwise_not(mask.clone()),
                );
                crate::ops::int::bitwise_or(old, crate::ops::int::bitwise_and(value, mask))
            }
            _ => crate::ops::float::float_mask_where(current, mask, value),
        };
    }
    current
}

fn cross(lhs: TtTensor, rhs: TtTensor, dim: usize) -> TtTensor {
    let (a, b) = (lhs.shape().to_vec(), rhs.shape().to_vec());
    let refuse = |why: &str| -> ! {
        fail(
            "float_cross",
            format_args!(
                "{why}: lhs=({}), rhs=({}), dim={dim}",
                context(&lhs),
                context(&rhs)
            ),
        )
    };
    if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 {
        refuse("only F32 operands are supported");
    }
    if a.len() != b.len() || dim >= a.len() || a[dim] != 3 || b[dim] != 3 {
        refuse("both operands need one rank and size 3 along dim");
    }
    if (0..a.len()).any(|d| d != dim && a[d] != b[d] && a[d] != 1 && b[d] != 1) {
        refuse("the other axes must broadcast");
    }
    if lhs.device != rhs.device {
        refuse("operands are on different devices");
    }
    type B = TtBackend;
    let part = |t: &TtTensor, i: usize| {
        let slices: Vec<_> = (0..a.len())
            .map(|d| {
                if d == dim {
                    burn_backend::Slice::new(i as isize, Some(i as isize + 1), 1)
                } else {
                    burn_backend::Slice::full()
                }
            })
            .collect();
        <B as FloatTensorOps<B>>::float_slice(t.clone(), &slices)
    };
    let mul =
        |x: &TtTensor, y: &TtTensor| <B as FloatTensorOps<B>>::float_mul(x.clone(), y.clone());
    let sub = |x: TtTensor, y: TtTensor| <B as FloatTensorOps<B>>::float_sub(x, y);
    let (a0, a1, a2) = (part(&lhs, 0), part(&lhs, 1), part(&lhs, 2));
    let (b0, b1, b2) = (part(&rhs, 0), part(&rhs, 1), part(&rhs, 2));
    // c = a x b: c0 = a1*b2 - a2*b1, c1 = a2*b0 - a0*b2, c2 = a0*b1 - a1*b0.
    let c0 = sub(mul(&a1, &b2), mul(&a2, &b1));
    let c1 = sub(mul(&a2, &b0), mul(&a0, &b2));
    let c2 = sub(mul(&a0, &b1), mul(&a1, &b0));
    <B as FloatTensorOps<B>>::float_cat(vec![c0, c1, c2], dim)
}

fn int_matmul_composed(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
    let (l, r) = (lhs.shape().to_vec(), rhs.shape().to_vec());
    let refuse = |why: &str| -> ! {
        fail(
            "int_matmul",
            format_args!("{why}: lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
        )
    };
    if lhs.dtype() != DType::I32 || rhs.dtype() != DType::I32 {
        refuse("only I32 operands are supported");
    }
    if l.len() < 2 || l.len() != r.len() || lhs.device != rhs.device {
        refuse("operands need one rank >= 2 on one device");
    }
    if !lhs.is_storable() || !rhs.is_storable() || !crate::server::supports_dram(lhs.device) {
        refuse("operands must be resident-storable with GDDR");
    }
    let rank = l.len();
    let (m, k, n) = (l[rank - 2], l[rank - 1], r[rank - 1]);
    if r[rank - 2] != k {
        refuse("inner dimensions differ");
    }
    if l.contains(&0) || r.contains(&0) {
        refuse("empty operands are not supported");
    }
    let mut batch = Vec::new();
    for d in 0..rank - 2 {
        match (l[d], r[d]) {
            (a, b) if a == b || b == 1 => batch.push(a),
            (1, b) => batch.push(b),
            _ => refuse("batch axes do not broadcast"),
        }
    }
    let batches: usize = batch.iter().product();
    let total = batches
        .checked_mul(m)
        .and_then(|x| x.checked_mul(k))
        .and_then(|x| x.checked_mul(n));
    if total.is_none_or(|t| t > INT_MATMUL_BUDGET) {
        fail(
            "int_matmul",
            format_args!(
                "the broadcast product batch*M*K*N = {batches}*{m}*{k}*{n} exceeds the \
                 {INT_MATMUL_BUDGET}-element budget; split the operands: lhs=({}), rhs=({})",
                context(&lhs),
                context(&rhs)
            ),
        );
    }
    // The flat batch of the operand that broadcasts into each output batch.
    let operand_batch = |dims: &[usize]| -> Vec<usize> {
        (0..batches)
            .map(|mut flat| {
                let mut at = 0;
                let mut stride = 1;
                for d in (0..batch.len()).rev() {
                    let coordinate = flat % batch[d];
                    flat /= batch[d];
                    if dims[d] != 1 {
                        at += coordinate * stride;
                    }
                    stride *= dims[d];
                }
                at
            })
            .collect()
    };
    let (lb, rb) = (operand_batch(&l[..rank - 2]), operand_batch(&r[..rank - 2]));
    let mut a_sources = Vec::with_capacity(batches * m * k * n);
    let mut b_sources = Vec::with_capacity(batches * m * k * n);
    for b in 0..batches {
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    a_sources.push((0, lb[b] * m * k + i * k + p));
                    b_sources.push((0, rb[b] * k * n + p * n + j));
                }
            }
        }
    }
    let mut wide = batch.clone();
    wide.extend([m, k, n]);
    let a = mapped_native(&[&lhs], a_sources, wide.clone());
    let b = mapped_native(&[&rhs], b_sources, wide);
    let product = crate::ops::int::int_mul(a, b);
    let summed = crate::ops::int::int_sum_dim(product, batch.len() + 1);
    let mut shape = batch;
    shape.extend([m, n]);
    restack(&summed, shape)
}
