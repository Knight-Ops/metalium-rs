//! Native operation implementations. Required methods without an implementation
//! fail explicitly through `generated/ops.rs`. Each is listed in
//! `xtask/src/gen_burn.rs`'s `OVERRIDDEN`, which the generator checks against
//! the pinned op traits.
//!
//! Three kinds: the ones that run on the device (`float_matmul`), the ones
//! that say or change which device a tensor is on, and the ones returning
//! futures, which the generated dispatch awaits inside the operation report.

// Keep explicit Send/lifetime bounds aligned with the pinned Burn signatures.
#![allow(clippy::manual_async_fn)]

use core::future::Future;

use crate::host::HostBuffer;
use burn_backend::ops::{
    FloatTensorOps, IntTensorOps, TransactionPrimitive, TransactionPrimitiveData,
};
use burn_backend::tensor::{BoolTensor, Device, FloatTensor, IntTensor, QuantizedTensor};
use burn_backend::{DType, ExecutionError, IntDType, TensorData, TensorMetadata};

use crate::server::Elem;
use crate::unsupported::{context, fail};
use crate::{TtBackend, TtDevice, TtQTensor, TtTensor};
use tt_kernels::sfpu::ops::kind_sfpu;

/// Resident conversion used by explicit casts and BF16 arithmetic dispatch.
/// BF16 arithmetic widens to F32 on Tensix, computes using the existing native
/// contract, then rounds its floating-point result once at the operation boundary.
pub(crate) fn cast_native(tensor: TtTensor, dtype: DType) -> TtTensor {
    if tensor.dtype() == dtype {
        return tensor;
    }
    if !matches!(
        (tensor.dtype(), dtype),
        (DType::F32, DType::BF16) | (DType::BF16, DType::F32)
    ) || !tensor.is_storable()
        || !crate::server::supports_dram(tensor.device)
    {
        fail(
            "float_cast",
            format_args!("{}, output={dtype:?}", context(&tensor)),
        );
    }
    let source = tensor.to_dram();
    let (id, dims) = crate::server::cast_float(
        tensor.device,
        source.buffer.id,
        [source.buffer.rows, source.buffer.cols],
        dtype == DType::BF16,
    );
    let mut result = device_view(tensor.device, id, dims, None, dtype)
        .to_dram()
        .clone();
    result.transposed = source.transposed;
    TtTensor::on_device(result, tensor.shape(), dtype, tensor.device)
}

pub(crate) fn float_compute_input(tensor: TtTensor) -> TtTensor {
    if tensor.dtype() == DType::BF16 {
        cast_native(tensor, DType::F32)
    } else {
        tensor
    }
}

pub(crate) fn bf16_matmul(lhs: &TtTensor, rhs: &TtTensor) -> Option<TtTensor> {
    if lhs.dtype() != DType::BF16
        || rhs.dtype() != DType::BF16
        || !lhs.is_storable()
        || !rhs.is_storable()
    {
        return None;
    }
    assert_eq!(
        lhs.device, rhs.device,
        "BF16 matmul operands must share a device"
    );
    if lhs.shape().num_dims() > 2 || rhs.shape().num_dims() > 2 {
        return materialized_batched_matmul(lhs, rhs);
    }
    if !lhs.is_storable_matrix() || !rhs.is_storable_matrix() {
        return None;
    }
    let a = plain_dram(lhs);
    let b = plain_dram(rhs);
    assert_eq!(
        a.buffer.cols, b.buffer.rows,
        "BF16 matmul inner dimensions differ"
    );
    let (id, dims) = crate::server::matmul_bf16(
        lhs.device,
        a.buffer.id,
        b.buffer.id,
        [a.buffer.rows, b.buffer.cols],
    );
    Some(cast_native(
        device_result(lhs.device, id, dims),
        DType::BF16,
    ))
}

/// Materialized matrix products for each logical batch, including ragged,
/// broadcast and strided operands. Repack preserves storage bits; products
/// remain F32 until the joined tensor's single output boundary conversion.
fn materialized_batched_matmul(lhs: &TtTensor, rhs: &TtTensor) -> Option<TtTensor> {
    if !lhs.is_storable() || !rhs.is_storable() || !crate::server::supports_dram(lhs.device) {
        return None;
    }
    let dtype = lhs.dtype();
    let (ls, rs) = (lhs.shape().to_vec(), rhs.shape().to_vec());
    let rank = ls.len().max(rs.len());
    if ls.len() < 2 || rs.len() < 2 {
        return None;
    }
    let (m, k, n) = (ls[ls.len() - 2], ls[ls.len() - 1], rs[rs.len() - 1]);
    if rs[rs.len() - 2] != k {
        return None;
    }
    let lead = |shape: &[usize]| {
        let mut out = vec![1; rank - shape.len()];
        out.extend_from_slice(&shape[..shape.len() - 2]);
        out
    };
    let (la, lb) = (lead(&ls), lead(&rs));
    let batch = la
        .iter()
        .zip(&lb)
        .map(|(&a, &b)| (a == b || a == 1 || b == 1).then_some(a.max(b)))
        .collect::<Option<Vec<_>>>()?;
    let count = batch.iter().try_fold(1usize, |a, &b| a.checked_mul(b))?;
    if count == 0 || [m, k, n].contains(&0) {
        return None;
    }
    let mut shape = batch.clone();
    shape.extend([m, n]);
    // A broadcast weight can consume all input rows in one packed product.
    if lb.iter().all(|&d| d == 1) && ls[..ls.len() - 2].iter().product::<usize>() == count {
        let a = repack_shape(lhs, vec![count.checked_mul(m)?, k]);
        let b = repack_shape(rhs, vec![k, n]);
        let out = if dtype == DType::BF16 {
            bf16_matmul(&a, &b)?
        } else {
            float::float_matmul(a, b)
        };
        return Some(repack_shape(&out, shape));
    }
    let view = |t: &TtTensor| {
        t.as_strided()
            .unwrap_or_else(|| crate::views::Strided::of(t.to_dram(), &t.shape().to_vec()))
    };
    let (va, vb) = (view(lhs), view(rhs));
    let device = lhs.device;
    let mut outputs = Vec::with_capacity(count);
    for batch_index in 0..count {
        let pack = |t: &TtTensor,
                    v: &crate::views::Strided,
                    original: &[usize],
                    lead: &[usize],
                    dims: [usize; 2]| {
            let mut index = batch_index;
            let mut coords = vec![0; batch.len()];
            for d in (0..batch.len()).rev() {
                coords[d] = if lead[d] == 1 { 0 } else { index % batch[d] };
                index /= batch[d];
            }
            let flat_batch = coords.iter().zip(lead).fold(0, |a, (&i, &d)| a * d + i);
            let size = dims[0] * dims[1];
            let sources = (0..size)
                .map(|i| v.at(original, flat_batch * size + i))
                .collect();
            let (id, dims) = crate::server::repack(device, v.src.buffer.id, sources, dims);
            device_result_shaped(t.device, id, dims, dims.into(), dtype)
        };
        let a = pack(lhs, &va, &ls, &la, [m, k]);
        let b = pack(rhs, &vb, &rs, &lb, [k, n]);
        let (id, dims) = if dtype == DType::BF16 {
            crate::server::matmul_bf16(device, a.to_dram().buffer.id, b.to_dram().buffer.id, [m, n])
        } else {
            crate::server::matmul_dram(
                device,
                a.to_dram().buffer.id,
                false,
                b.to_dram().buffer.id,
                false,
            )
        };
        outputs.push(device_result(device, id, dims));
    }
    let ids = outputs.iter().map(|t| t.to_dram().buffer.id).collect();
    let rows = (0..count)
        .flat_map(|b| (0..m).map(move |r| (b, r)))
        .collect();
    let (id, dims) = crate::server::gather_rows(device, ids, rows, n);
    Some(cast_native(
        device_result_shaped(device, id, dims, shape.into(), DType::F32),
        dtype,
    ))
}

pub(crate) fn float_compute_dtype(inputs: &[&TtTensor]) -> DType {
    let dtype = inputs[0].dtype();
    if inputs.iter().any(|input| input.dtype() != dtype) {
        fail(
            "BF16 arithmetic",
            "floating-point operands must have the same dtype",
        );
    }
    dtype
}

/// A tensor moved to `device`. The same device: unchanged. Another: its host
/// copy, retagged -- a device copy belongs to the chip that holds it.
fn retag(tensor: TtTensor, device: &TtDevice) -> TtTensor {
    if tensor.device == *device {
        return tensor;
    }
    TtTensor::new(tensor.into_host(), *device)
}

/// A device result as a tensor.
fn device_result(device: TtDevice, id: crate::server::BufferId, dims: [usize; 2]) -> TtTensor {
    device_view(device, id, dims, None, DType::F32)
}

/// [`device_result`] for a tensor of `shape`, stored as `dims`
/// (`crate::tensor::stored_dims`).
fn device_result_shaped(
    device: TtDevice,
    id: crate::server::BufferId,
    dims: [usize; 2],
    shape: burn_backend::Shape,
    dtype: DType,
) -> TtTensor {
    let t = device_view(device, id, dims, None, dtype);
    let dram = t.dram().expect("made on the device").clone();
    TtTensor::on_device(dram, shape, dtype, device)
}

/// Materialize a matrix transpose with the native block copier, preserving bits.
fn plain_dram(tensor: &TtTensor) -> crate::tensor::DramRef {
    let d = tensor.to_dram();
    if !d.transposed {
        return d.clone();
    }
    let dims = [d.buffer.cols, d.buffer.rows];
    let (id, dims) = crate::server::copy_blocks(
        tensor.device,
        d.buffer.id,
        vec![crate::BlockMove {
            from: [0, 0],
            to: [0, 0],
            extent: dims,
            transposed: true,
        }],
        dims,
    );
    device_view(tensor.device, id, dims, None, tensor.dtype())
        .to_dram()
        .clone()
}

fn mapped_native(
    inputs: &[&TtTensor],
    sources: Vec<(usize, usize)>,
    shape: Vec<usize>,
) -> TtTensor {
    let device = inputs[0].device;
    let dtype = inputs[0].dtype();
    let views: Vec<_> = inputs
        .iter()
        .map(|x| {
            assert_eq!(x.device, device);
            assert_eq!(x.dtype(), dtype);
            let shape = x.shape().to_vec();
            x.as_strided()
                .unwrap_or_else(|| crate::views::Strided::of(x.to_dram(), &shape))
        })
        .collect();
    let sources = sources
        .into_iter()
        .map(|(i, flat)| {
            assert!(
                flat < inputs[i].shape().num_elements(),
                "patch mapping out of range"
            );
            (i, views[i].at(&inputs[i].shape().to_vec(), flat))
        })
        .collect();
    let dims = crate::tensor::stored_dims(&shape).expect("nonempty patch shape");
    let (id, dims) = crate::server::repack_many(
        device,
        views.iter().map(|v| v.src.buffer.id).collect(),
        sources,
        dims,
    );
    device_result_shaped(device, id, dims, shape.into(), dtype)
}

/// Put the selected logical axis in matrix columns; the host constructs addresses only.
fn pack_axis(tensor: &TtTensor, dim: usize) -> Option<TtTensor> {
    let shape = tensor.shape().to_vec();
    if dim >= shape.len() || shape.contains(&0) || !tensor.is_storable() {
        return None;
    }
    let view = tensor
        .as_strided()
        .unwrap_or_else(|| crate::views::Strided::of(tensor.to_dram(), &shape));
    let mut out = shape.clone();
    out[dim] = 1;
    let outputs = out.iter().try_fold(1usize, |n, &d| n.checked_mul(d))?;
    let mut sources = Vec::with_capacity(outputs.checked_mul(shape[dim])?);
    for output in 0..outputs {
        let mut remaining = output;
        let mut indices = vec![0; shape.len()];
        for k in (0..shape.len()).rev() {
            indices[k] = remaining % out[k];
            remaining /= out[k];
        }
        for index in 0..shape[dim] {
            indices[dim] = index;
            let flat = indices.iter().zip(&shape).fold(0, |n, (&i, &d)| n * d + i);
            sources.push(view.at(&shape, flat));
        }
    }
    let (id, dims) = crate::server::repack(
        tensor.device,
        view.src.buffer.id,
        sources,
        [outputs, shape[dim]],
    );
    Some(device_view(tensor.device, id, dims, None, tensor.dtype()))
}

/// Retile a matrix's logical sequence into another shape without reading its data.
fn repack_shape(tensor: &TtTensor, shape: Vec<usize>) -> TtTensor {
    let dims = crate::tensor::stored_dims(&shape).expect("nonempty output shape");
    let source = plain_dram(tensor);
    let cols = source.buffer.cols;
    let sources = (0..shape.iter().product())
        .map(|i| [i / cols, i % cols])
        .collect();
    let (id, dims) = crate::server::repack(tensor.device, source.buffer.id, sources, dims);
    device_result_shaped(
        tensor.device,
        id,
        dims,
        burn_backend::Shape::from(shape),
        tensor.dtype(),
    )
}

/// Copy a stepped logical slice on the card; negative steps traverse the
/// selected interval backward, as Burn's pinned Slice contract specifies.
fn sliced_native(tensor: &TtTensor, slices: &[burn_backend::Slice]) -> Option<TtTensor> {
    let shape = tensor.shape().to_vec();
    if !tensor.is_storable()
        || slices.len() > shape.len()
        || !crate::server::supports_dram(tensor.device)
    {
        return None;
    }
    let mut ranges = Vec::new();
    for (dim, &n) in shape.iter().enumerate() {
        let slice = slices
            .get(dim)
            .cloned()
            .unwrap_or_else(burn_backend::Slice::full);
        let range = slice.to_range(n);
        let step = slice.step.unsigned_abs();
        if step == 0 {
            return None;
        }
        let indices: Vec<_> = if slice.step > 0 {
            range.step_by(step).collect()
        } else {
            range.rev().step_by(step).collect()
        };
        if indices.is_empty() {
            return None;
        }
        ranges.push(indices);
    }
    let output: Vec<_> = ranges.iter().map(Vec::len).collect();
    let count = output.iter().try_fold(1usize, |n, &d| n.checked_mul(d))?;
    let view = tensor
        .as_strided()
        .unwrap_or_else(|| crate::views::Strided::of(tensor.to_dram(), &shape));
    let sources = (0..count)
        .map(|mut i| {
            let mut coords = vec![0; shape.len()];
            for d in (0..shape.len()).rev() {
                coords[d] = ranges[d][i % output[d]];
                i /= output[d];
            }
            let flat = coords.iter().zip(&shape).fold(0, |n, (&i, &d)| n * d + i);
            view.at(&shape, flat)
        })
        .collect();
    let dims = crate::tensor::stored_dims(&output)?;
    let (id, dims) = crate::server::repack(tensor.device, view.src.buffer.id, sources, dims);
    Some(device_result_shaped(
        tensor.device,
        id,
        dims,
        burn_backend::Shape::from(output),
        tensor.dtype(),
    ))
}

/// General raw resident gather. Static staging covers every legal source;
/// B reads bounded I32 indices and rejects the domain before indexed copying.
fn gather_native(dim: usize, tensor: TtTensor, indices: TtTensor) -> TtTensor {
    let shape = tensor.shape().to_vec();
    let output = indices.shape().to_vec();
    assert!(
        dim < shape.len()
            && output.len() == shape.len()
            && shape
                .iter()
                .zip(&output)
                .enumerate()
                .all(|(d, (&a, &b))| d == dim || a == b)
            && tensor.is_storable()
            && indices.dtype() == DType::I32
            && tensor.device == indices.device
            && shape[dim] <= i32::MAX as usize,
        "invalid native gather geometry"
    );
    let mut results = Vec::new();
    for first in (0..indices.shape().num_elements()).step_by(32) {
        let count = 32.min(indices.shape().num_elements() - first);
        let mut sources = Vec::new();
        for flat in first..first + count {
            let mut index = flat;
            let mut coordinates = vec![0; shape.len()];
            for d in (0..shape.len()).rev() {
                coordinates[d] = index % output[d];
                index /= output[d];
            }
            for k in 0..shape[dim] {
                coordinates[dim] = k;
                let at = coordinates
                    .iter()
                    .zip(&shape)
                    .fold(0, |n, (&i, &d)| n * d + i);
                sources.push((0, at));
            }
        }
        let staged = mapped_native(&[&tensor], sources, vec![count, shape[dim]]);
        let index = mapped_native(
            &[&indices],
            (first..first + count).map(|i| (0, i)).collect(),
            vec![count, 1],
        );
        let (id, dims) = crate::server::gather_indexed(
            tensor.device,
            staged.to_dram().buffer.id,
            index.to_dram().buffer.id,
            [count, 1],
        );
        results.push(device_result_shaped(
            tensor.device,
            id,
            dims,
            dims.into(),
            tensor.dtype(),
        ));
    }
    let sources = (0..indices.shape().num_elements())
        .map(|i| (i / 32, i % 32))
        .collect();
    let refs: Vec<_> = results.iter().collect();
    mapped_native(&refs, sources, output)
}

fn select_native(tensor: TtTensor, dim: usize, indices: TtTensor) -> TtTensor {
    let mut shape = tensor.shape().to_vec();
    assert!(dim < shape.len() && indices.shape().num_dims() == 1);
    shape[dim] = indices.shape().num_elements();
    let mut index_shape = vec![1; shape.len()];
    index_shape[dim] = shape[dim];
    let indices = reshaped(indices, index_shape.into());
    let indices = expanded(&indices, shape.into(), "resident selection indices");
    gather_native(dim, tensor, indices)
}

fn scatter_add_native(
    dim: usize,
    tensor: TtTensor,
    indices: TtTensor,
    value: TtTensor,
) -> TtTensor {
    assert_eq!(tensor.dtype(), value.dtype());
    let dtype = tensor.dtype();
    let stored = matches!(dtype, DType::I32 | DType::Bool(_));
    if !stored {
        float_compute_dtype(&[&tensor, &value]);
    }
    let shape = tensor.shape().to_vec();
    let output = indices.shape().to_vec();
    assert!(
        dim < shape.len()
            && output.len() == shape.len()
            && output == value.shape().to_vec()
            && shape
                .iter()
                .zip(&output)
                .enumerate()
                .all(|(d, (&a, &b))| d == dim || a == b)
    );
    let tensor = if stored {
        tensor
    } else {
        float_compute_input(tensor)
    };
    let value = if stored {
        value
    } else {
        float_compute_input(value)
    };
    let mut current = pack_axis(&tensor, dim).expect("native scatter axis");
    let values = pack_axis(&value, dim).expect("native scatter values");
    let indices = pack_axis(&indices, dim).expect("native scatter indices");
    let rows = current.shape().to_vec()[0];
    let count = output[dim];
    let width = shape[dim];
    // Integer column coordinates are replayable metadata, never tensor data.
    let id = crate::server::metadata(
        tensor.device,
        (0..width).map(|i| i as u32).collect(),
        [1, width],
        Elem::I32,
    );
    let columns =
        device_result_shaped(tensor.device, id, [1, width], [1, width].into(), DType::I32);
    let columns = expanded(&columns, [rows, width].into(), "scatter columns");
    for j in 0..count {
        let index = mapped_native(
            &[&indices],
            (0..rows).map(|r| (0, r * count + j)).collect(),
            vec![rows, 1],
        );
        // The checked raw gather is also the domain dependency of the update.
        let selected = gather_native(1, current.clone(), index.clone());
        let value = mapped_native(
            &[&values],
            (0..rows).map(|r| (0, r * count + j)).collect(),
            vec![rows, 1],
        );
        let sum = match dtype {
            DType::I32 => int::int_add(selected, value),
            DType::Bool(_) => bool::bool_or(selected, value),
            _ => float::float_add(selected, value),
        };
        let index = expanded(&index, [rows, width].into(), "scatter indices");
        let mask = device_eltwise_ungated(kind_sfpu::INT_EQ, 0.0, &columns, Some(&index))
            .expect("integer scatter equality");
        let sum = expanded(&sum, [rows, width].into(), "scatter values");
        current = match dtype {
            DType::I32 => {
                let mask =
                    int::int_mul_scalar(bool::bool_into_int(mask, IntDType::I32), (-1i32).into());
                let old = int::bitwise_and(current, int::bitwise_not(mask.clone()));
                int::bitwise_or(old, int::bitwise_and(sum, mask))
            }
            DType::Bool(_) => bool::bool_or(current, bool::bool_and(mask, sum)),
            _ => float::float_mask_where(current, mask, sum),
        };
    }
    let sources = (0..tensor.shape().num_elements())
        .map(|mut flat| {
            let mut coords = vec![0; shape.len()];
            for d in (0..shape.len()).rev() {
                coords[d] = flat % shape[d];
                flat /= shape[d];
            }
            let mut row = 0;
            for d in 0..shape.len() {
                if d != dim {
                    row = row * shape[d] + coords[d];
                }
            }
            (0, row * width + coords[dim])
        })
        .collect();
    cast_native(mapped_native(&[&current], sources, shape), dtype)
}

fn select_add_native(tensor: TtTensor, dim: usize, indices: TtTensor, value: TtTensor) -> TtTensor {
    assert!(indices.shape().num_dims() == 1 && dim < value.shape().num_dims());
    let shape = value.shape().to_vec();
    assert_eq!(shape[dim], indices.shape().num_elements());
    let mut index_shape = vec![1; shape.len()];
    index_shape[dim] = shape[dim];
    let indices = reshaped(indices, index_shape.into());
    let indices = expanded(&indices, shape.into(), "resident select-add indices");
    scatter_add_native(dim, tensor, indices, value)
}

/// Replace a logical slice using raw resident copies, including stepped views.
fn slice_assign_native(
    tensor: TtTensor,
    slices: &[burn_backend::Slice],
    value: TtTensor,
) -> TtTensor {
    let shape = tensor.shape().to_vec();
    assert!(
        tensor.is_storable()
            && value.is_storable()
            && tensor.dtype() == value.dtype()
            && tensor.device == value.device
            && slices.len() <= shape.len()
            && crate::server::supports_dram(tensor.device),
        "invalid native slice assignment"
    );
    let ranges: Vec<Vec<usize>> = shape
        .iter()
        .enumerate()
        .map(|(d, &n)| {
            let slice = slices
                .get(d)
                .cloned()
                .unwrap_or_else(burn_backend::Slice::full);
            let step = slice.step.unsigned_abs();
            assert!(step > 0, "slice assignment step must be nonzero");
            let range = slice.to_range(n);
            if slice.step > 0 {
                range.step_by(step).collect()
            } else {
                range.rev().step_by(step).collect()
            }
        })
        .collect();
    assert_eq!(
        value.shape().to_vec(),
        ranges.iter().map(Vec::len).collect::<Vec<_>>(),
        "slice assignment value shape differs"
    );
    let tv = tensor
        .as_strided()
        .unwrap_or_else(|| crate::views::Strided::of(tensor.to_dram(), &shape));
    let vs = value.shape().to_vec();
    let vv = value
        .as_strided()
        .unwrap_or_else(|| crate::views::Strided::of(value.to_dram(), &vs));
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .expect("slice assignment overflow");
    let mut sources: Vec<_> = (0..count).map(|i| (0, tv.at(&shape, i))).collect();
    for flat in 0..value.shape().num_elements() {
        let mut index = flat;
        let mut coordinates = vec![0; shape.len()];
        for d in (0..shape.len()).rev() {
            coordinates[d] = ranges[d][index % vs[d]];
            index /= vs[d];
        }
        let dst = coordinates
            .iter()
            .zip(&shape)
            .fold(0, |n, (&i, &d)| n * d + i);
        sources[dst] = (1, vv.at(&vs, flat));
    }
    let dims = crate::tensor::stored_dims(&shape).expect("nonempty slice assignment");
    let (id, dims) = crate::server::repack_many(
        tensor.device,
        vec![tv.src.buffer.id, vv.src.buffer.id],
        sources,
        dims,
    );
    device_result_shaped(tensor.device, id, dims, shape.into(), tensor.dtype())
}

fn native_zeros(shape: burn_backend::Shape, device: &TtDevice, dtype: DType) -> Option<TtTensor> {
    if !crate::server::is_attached(*device) || !crate::server::supports_dram(*device) {
        return None;
    }
    let elem = match dtype {
        DType::F32 | DType::BF16 => Elem::F32,
        DType::I32 => Elem::I32,
        DType::Bool(_) => Elem::Bool,
        _ => return None,
    };
    let dims = crate::tensor::stored_dims(&shape.to_vec())?;
    let (id, dims) = crate::server::zeros(*device, dims, elem, dtype == DType::BF16);
    Some(device_result_shaped(*device, id, dims, shape, dtype))
}

/// Broadcast bytes on the device using row gathers and whole-matrix transposes.
/// No arithmetic or host readback is involved, including for scalar tensors.
fn expanded(tensor: &TtTensor, shape: burn_backend::Shape, op: &str) -> TtTensor {
    let from = tensor.shape().to_vec();
    let to = shape.to_vec();
    if from == to {
        return tensor.clone();
    }
    if !tensor.is_storable() || to.len() < from.len() || to.contains(&0) {
        fail(
            op,
            format_args!("{}, target_shape={shape:?}", context(tensor)),
        );
    }
    let mut padded = vec![1; to.len() - from.len()];
    padded.extend(from);
    if !padded.iter().zip(&to).all(|(&a, &b)| a == b || a == 1) {
        fail(
            op,
            format_args!("{}, target_shape={shape:?}", context(tensor)),
        );
    }
    if !crate::server::supports_dram(tensor.device) {
        fail(
            op,
            format_args!("{}, engine has no resident buffers", context(tensor)),
        );
    }
    let rank = to.len();
    let rows: usize = to[..rank - 1].iter().product();
    let cols = to[rank - 1];
    let input_cols = padded[rank - 1];
    let source = plain_dram(tensor);
    let indices: Vec<_> = (0..rows)
        .map(|mut flat| {
            let mut coords = vec![0; rank - 1];
            for d in (0..rank - 1).rev() {
                coords[d] = flat % to[d];
                flat /= to[d];
            }
            let row = coords
                .iter()
                .zip(&padded)
                .fold(0, |acc, (&i, &n)| acc * n + if n == 1 { 0 } else { i });
            (0, row)
        })
        .collect();
    let (id, dims) =
        crate::server::gather_rows(tensor.device, vec![source.buffer.id], indices, input_cols);
    let gathered = device_view(tensor.device, id, dims, None, tensor.dtype());
    if input_cols == cols {
        return TtTensor::on_device(
            gathered.to_dram().clone(),
            shape,
            tensor.dtype(),
            tensor.device,
        );
    }
    // [rows, 1] -> [1, rows] -> [cols, rows] -> [rows, cols].
    let transposed = swapped_view(&gathered, 0, 1).expect("a gathered matrix");
    let transposed = plain_dram(&transposed);
    let (id, dims) = crate::server::gather_rows(
        tensor.device,
        vec![transposed.buffer.id],
        vec![(0, 0); cols],
        rows,
    );
    let repeated = device_view(tensor.device, id, dims, None, tensor.dtype());
    let transposed = swapped_view(&repeated, 0, 1).expect("a gathered matrix");
    TtTensor::on_device(
        plain_dram(&transposed),
        shape,
        tensor.dtype(),
        tensor.device,
    )
}

/// A device result that reads `parent`'s slots, keeping it alive.
fn device_view(
    device: TtDevice,
    id: crate::server::BufferId,
    [m, n]: [usize; 2],
    parent: Option<std::sync::Arc<crate::tensor::Buffer>>,
    dtype: DType,
) -> TtTensor {
    let buffer = std::sync::Arc::new(crate::tensor::Buffer {
        id,
        device,
        rows: m,
        cols: n,
        parent,
    });
    TtTensor::on_device(
        crate::tensor::DramRef {
            buffer,
            transposed: false,
        },
        burn_backend::Shape::from(vec![m, n]),
        dtype,
        device,
    )
}

// Native approximations are held to the bounds in the SFPU operation gates.
// Supported inputs upload as needed; there is no reference-backend mode.

/// The shape `a` and `b` broadcast to (NumPy's rule, which Burn's binary ops
/// follow), or `None` if they do not.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let at = |s: &[usize], i: usize| (i + s.len()).checked_sub(n).map_or(1, |j| s[j]);
    (0..n)
        .map(|i| match (at(a, i), at(b, i)) {
            (x, y) if x == y || y == 1 => Some(x),
            (1, y) => Some(y),
            _ => None,
        })
        .collect()
}

/// Run supported stored dtypes and broadcast shapes on the device, uploading
/// inputs and copying transposed layouts as needed. Return `None` when the
/// engine or kernel cannot handle the inputs; callers report their metadata.
fn device_eltwise(kind: u32, scalar: f32, a: &TtTensor, b: Option<&TtTensor>) -> Option<TtTensor> {
    device_eltwise_ungated(kind, scalar, a, b)
}

/// [`device_eltwise`] for native compositions: for a composition that has
/// already decided, on its whole input, to run on the device, or one that is
/// exact as composed (a gather's masked sum).
fn device_eltwise_ungated(
    kind: u32,
    scalar: f32,
    a: &TtTensor,
    b: Option<&TtTensor>,
) -> Option<TtTensor> {
    let op = tt_kernels::tensor::Eltwise {
        kind,
        scalar,
        scalar2: 0.0,
    };
    device_op_ungated(op, a, b, None)
}

/// [`device_eltwise`] with the whole op (both scalars) and a ternary op's
/// third operand, `a`'s shape.
fn device_op(
    op: tt_kernels::tensor::Eltwise,
    a: &TtTensor,
    b: Option<&TtTensor>,
    c: Option<&TtTensor>,
) -> Option<TtTensor> {
    device_op_ungated(op, a, b, c)
}

fn device_op_ungated(
    op: tt_kernels::tensor::Eltwise,
    a: &TtTensor,
    b: Option<&TtTensor>,
    c: Option<&TtTensor>,
) -> Option<TtTensor> {
    use tt_kernels::kind as k;
    let kind = op.kind;
    let device = a.device;
    // Any rank: each operand as the matrix it is stored as
    // (`crate::tensor::stored_dims`), of the element type the kind computes on.
    let sig = tt_kernels::sfpu::ops::elems(kind);
    let fits = |t: &TtTensor, i: usize| {
        t.is_storable() && t.elem() == sig.inputs.get(i).copied() && t.device == device
    };
    if !fits(a, 0) || b.is_some_and(|b| !fits(b, 1)) || c.is_some_and(|c| !fits(c, 2)) {
        return None;
    }
    if c.is_some_and(|c| c.shape() != a.shape()) {
        return None;
    }
    let (input, output) = (sig.inputs[0], sig.out);
    if !crate::server::supports_dram(device) {
        return None;
    }
    let (sa, sb) = (a.shape().to_vec(), b.map(|b| b.shape().to_vec()));
    let broadcasts = tt_kernels::sfpu::ops::broadcasts(kind);
    // `b` the same shape, or one that broadcasts to `a` as one row or one
    // column of `a`'s matrix: a row is `b` with every leading dimension 1, a
    // column `b` with `a`'s leading dimensions and a last of 1. Read on the
    // matrices, then held to the broadcast's own shape -- `[6, 1, 4] + [6, 1]`
    // is a column of `[6, 4]` by the matrices, but `[6, 6, 4]` by the rule.
    // For an op that commutes bit for bit (addition, multiplication) the
    // broadcast operand may be on the left. The session reads which from the
    // matrices.
    let broadcast_of = |x: &[usize], y: &[usize]| {
        let ([xr, xc], [yr, yc]) = (
            crate::tensor::stored_dims(x).expect("checked"),
            crate::tensor::stored_dims(y).expect("checked"),
        );
        ((yr == 1 && yc == xc) || (yc == 1 && yr == xr))
            && broadcast_shape(x, y).is_some_and(|s| s == x)
    };
    let (a, b) = match &sb {
        None => (a, b),
        Some(s) if *s == sa => (a, b),
        Some(s) if broadcasts && broadcast_of(&sa, s) => (a, b),
        Some(s) if matches!(kind, k::ADD | k::MUL) && broadcast_of(s, &sa) => (b?, Some(a)),
        Some(s) if broadcasts => {
            let shape = broadcast_shape(&sa, s)?;
            if shape != sa {
                let lhs = expanded(a, burn_backend::Shape::from(shape), "broadcast");
                return device_op_ungated(op, &lhs, b, c);
            }
            let rhs = expanded(b?, a.shape(), "broadcast");
            return device_op_ungated(op, a, Some(&rhs), c);
        }
        _ => return None,
    };
    let (da, db, dc) = (plain_dram(a), b.map(plain_dram), c.map(plain_dram));
    let (id, dims) = crate::server::eltwise_op(
        device,
        op,
        da.buffer.id,
        db.as_ref().map(|d| d.buffer.id),
        dc.as_ref().map(|d| d.buffer.id),
    );
    // The input's own dtype where the element type is kept (a bool's store
    // among them), else the output's.
    let dtype = if output == input {
        a.dtype()
    } else {
        match output {
            Elem::F32 => DType::F32,
            Elem::I32 => DType::I32,
            Elem::Bool => DType::Bool(burn_backend::BoolStore::Native),
        }
    };
    Some(device_result_shaped(device, id, dims, a.shape(), dtype))
}

/// What [`device_pow`] raises to.
#[derive(Copy, Clone)]
enum PowY<'a> {
    Tensor(&'a TtTensor),
    Scalar(f32),
    Int(&'a TtTensor),
}

/// `x^y` on the device (`tt_kernels::session::Session::pow`, one op), if
/// that is where the data is -- `x` an `F32` tensor, `y` one of its shape (or
/// an `I32` one) or a scalar, one of them already on the device, none a
/// transposed view; an approximation, so gated as [`device_eltwise`] gates
/// one. `None` otherwise.
fn device_pow(x: &TtTensor, y: PowY<'_>) -> Option<TtTensor> {
    let device = x.device;
    if !x.is_storable() || x.elem() != Some(Elem::F32) {
        return None;
    }
    let yt = match y {
        PowY::Tensor(t) | PowY::Int(t) => {
            let want = if matches!(y, PowY::Int(_)) {
                Elem::I32
            } else {
                Elem::F32
            };
            if t.elem() != Some(want) || t.shape() != x.shape() || t.device != device {
                return None;
            }
            Some(t)
        }
        PowY::Scalar(_) => None,
    };
    if !crate::server::supports_dram(device) {
        return None;
    }
    let dx = x.to_dram();
    let dy = yt.map(|t| t.to_dram());
    if dx.transposed || dy.is_some_and(|d| d.transposed) {
        return None;
    }
    let arg = match y {
        PowY::Tensor(_) => crate::server::PowArg::Tensor(dy?.buffer.id),
        PowY::Int(_) => crate::server::PowArg::Int(dy?.buffer.id),
        PowY::Scalar(v) => crate::server::PowArg::Scalar(v),
    };
    let (id, dims) = crate::server::pow(device, dx.buffer.id, arg);
    Some(device_result_shaped(
        device,
        id,
        dims,
        x.shape(),
        DType::F32,
    ))
}

/// `t`, a device result, as `dtype` -- a comparison's requested bool store,
/// which every store shares on the device (`Elem::Bool`).
fn retyped(t: TtTensor, dtype: DType) -> TtTensor {
    let d = t.dram().expect("a device result").clone();
    TtTensor::on_device(d, t.shape(), dtype, t.device)
}

/// The two dimensions of a matrix already on the device swapped, as a view of
/// the same buffer, whatever its element type -- the device matmul reads it
/// transposed, and a host op downloads it transposed. `None` for anything else.
fn swapped_view(tensor: &TtTensor, dim1: usize, dim2: usize) -> Option<TtTensor> {
    if !(tensor.is_storable_matrix() && dim1 != dim2 && dim1 < 2 && dim2 < 2) {
        return None;
    }
    let d = tensor.dram()?;
    let d = crate::tensor::DramRef {
        buffer: d.buffer.clone(),
        transposed: !d.transposed,
    };
    let s = tensor.shape().to_vec();
    Some(TtTensor::on_device(
        d,
        burn_backend::Shape::from(vec![s[1], s[0]]),
        tensor.dtype(),
        tensor.device,
    ))
}

/// `tensor` reshaped, a view of the same slots when the new shape is stored as
/// the same matrix (`crate::tensor::stored_dims`: `[1, n]` and `[n]`, `[2, 3,
/// 4]` and `[6, 4]`), so a reshape of a device tensor costs nothing and keeps
/// it there; else `host`'s, on the host copy -- sharing the source's
/// device-copy slot where there is none yet, so a bias reshaped every forward
/// pass is uploaded once.
fn reshaped(tensor: TtTensor, shape: burn_backend::Shape) -> TtTensor {
    let same =
        tensor.stored().is_some() && tensor.stored() == crate::tensor::stored_dims(&shape.to_vec());
    if let Some(d) = tensor.dram().filter(|d| !d.transposed) {
        if same && tensor.is_storable() {
            assert_eq!(
                tensor.shape().num_elements(),
                shape.num_elements(),
                "invalid reshape"
            );
            return TtTensor::on_device(d.clone(), shape, tensor.dtype(), tensor.device);
        }
    }
    if tensor.computed_on_device() && crate::server::supports_dram(tensor.device) {
        if let (Some([r, c]), Some(target)) =
            (tensor.stored(), crate::tensor::stored_dims(&shape.to_vec()))
        {
            if target == [c, r] && (r == 1 || c == 1) {
                let plain = plain_dram(&tensor);
                let view = TtTensor::on_device(
                    crate::tensor::DramRef {
                        buffer: plain.buffer,
                        transposed: true,
                    },
                    burn_backend::Shape::new([c, r]),
                    tensor.dtype(),
                    tensor.device,
                );
                return TtTensor::on_device(
                    plain_dram(&view),
                    shape,
                    tensor.dtype(),
                    tensor.device,
                );
            }
        }
    }
    if tensor.computed_on_device() {
        fail(
            "reshape",
            format_args!("{}, target_shape={shape:?}", context(&tensor)),
        );
    }
    let host = tensor.host().clone().reshape(shape.clone());
    if same && tensor.is_storable() && tensor.dram().is_none() {
        return tensor.reshaped_host(host, shape);
    }
    TtTensor::new(host, tensor.device)
}

/// On `device`, and for a tensor the device stores on an engine that keeps
/// tensors, resident there: `Tensor::to_device` is how a caller says "this
/// lives on the card" (a dataset, its labels, a mask).
fn to_device_resident(tensor: TtTensor, device: &TtDevice) -> TtTensor {
    let t = retag(tensor, device);
    if t.is_storable() && crate::server::supports_dram(*device) {
        let _ = t.to_dram();
    }
    t
}

pub mod float {
    use super::*;
    use burn_backend::Scalar;
    use tt_kernels::kind;

    /// First extremum index, with the first NaN taking precedence. Index
    /// metadata is constructed on the host; comparisons and selection stay
    /// on the device. Only I32 output and rank-one/two inputs are supported.
    pub fn float_argmax(tensor: TtTensor, dim: usize, out_dtype: IntDType) -> TtTensor {
        argextreme(tensor, dim, out_dtype, false)
    }

    /// First minimum index, with the same tie and NaN priority as argmax.
    pub fn float_argmin(tensor: TtTensor, dim: usize, out_dtype: IntDType) -> TtTensor {
        argextreme(tensor, dim, out_dtype, true)
    }

    fn argextreme(tensor: TtTensor, dim: usize, out_dtype: IntDType, minimum: bool) -> TtTensor {
        let op = if minimum {
            "float_argmin"
        } else {
            "float_argmax"
        };
        let shape = tensor.shape().to_vec();
        if out_dtype != IntDType::I32
            || shape.is_empty()
            || dim >= shape.len()
            || shape[dim] == 0
            || shape[dim] > 8_388_608
            || !tensor.is_stored_f32()
            || !crate::server::supports_dram(tensor.device)
        {
            fail(
                op,
                format_args!("{}, dim={dim}, out_dtype={out_dtype:?}", context(&tensor)),
            );
        }
        if shape.len() > 2 {
            let packed = pack_axis(&tensor, dim).unwrap_or_else(|| fail(op, context(&tensor)));
            let selected = argextreme(packed, 1, out_dtype, minimum);
            let mut output = shape;
            output[dim] = 1;
            return repack_shape(&selected, output);
        }
        let tensor = if minimum { float_neg(tensor) } else { tensor };
        let axis = if shape.len() == 1 { 1 } else { dim };
        let dims = tensor.stored().expect("matrix");
        let tensor = reshaped(tensor, burn_backend::Shape::new(dims));
        let indices = (0..dims[0] * dims[1])
            .map(|i| {
                let index = if axis == 1 { i % dims[1] } else { i / dims[1] };
                -(index as f32)
            })
            .collect::<Vec<_>>();
        let bits = indices.into_iter().map(f32::to_bits).collect();
        let id = crate::server::metadata(tensor.device, bits, dims, Elem::F32);
        let indices = device_result(tensor.device, id, dims);
        let nan = device_eltwise(kind_sfpu::IS_NAN, 0.0, &tensor, None).expect("F32 predicate");
        let clean = device_eltwise(kind_sfpu::MASK_FILL, f32::NEG_INFINITY, &tensor, Some(&nan))
            .expect("mask");
        let max = float_max_dim(clean, axis);
        let unequal =
            device_eltwise(kind_sfpu::NE, 0.0, &tensor, Some(&max)).expect("comparison broadcast");
        let candidate = device_eltwise(
            kind_sfpu::MASK_FILL,
            f32::NEG_INFINITY,
            &indices,
            Some(&unequal),
        )
        .expect("mask");
        let best = float_max_dim(candidate, axis);
        let not_nan =
            device_eltwise(kind_sfpu::BOOL_NOT, 0.0, &nan, None).expect("predicate negation");
        let nan_indices = device_eltwise(
            kind_sfpu::MASK_FILL,
            f32::NEG_INFINITY,
            &indices,
            Some(&not_nan),
        )
        .expect("NaN index mask");
        let first_nan = float_max_dim(nan_indices, axis);
        let has_nan =
            device_eltwise(kind_sfpu::NE_S, f32::NEG_INFINITY, &first_nan, None).expect("NaN flag");
        let selected = device_op(
            tt_kernels::tensor::Eltwise {
                kind: kind_sfpu::MASK_WHERE,
                scalar: 0.0,
                scalar2: 0.0,
            },
            &best,
            Some(&has_nan),
            Some(&first_nan),
        )
        .expect("index selection");
        let positive = float_neg(selected);
        let index = device_eltwise(kind_sfpu::INDEX_TO_I32, 0.0, &positive, None)
            .expect("native index cast");
        let mut output = shape;
        output[dim] = 1;
        reshaped(index, burn_backend::Shape::from(output))
    }

    pub fn float_from_data(data: TensorData, device: &TtDevice) -> TtTensor {
        if !matches!(data.dtype, DType::F32 | DType::BF16) {
            fail("float_from_data", context(&data));
        }
        TtTensor::new(HostBuffer::from_data(data), *device)
    }

    pub fn float_empty(
        shape: burn_backend::Shape,
        device: &TtDevice,
        dtype: burn_backend::FloatDType,
    ) -> TtTensor {
        if let Some(out) = native_zeros(shape.clone(), device, dtype.into()) {
            return out;
        }
        if DType::from(dtype) == DType::BF16 {
            let mut data = TensorData::zeros::<u16, _>(shape);
            data.dtype = DType::BF16;
            return float_from_data(data, device);
        }
        if DType::from(dtype) != DType::F32 {
            fail(
                "float_empty",
                format_args!("shape={shape:?}, dtype={dtype:?}"),
            );
        }
        float_from_data(TensorData::zeros::<f32, _>(shape), device)
    }

    pub fn float_random(
        shape: burn_backend::Shape,
        distribution: burn_backend::Distribution,
        device: &TtDevice,
        dtype: burn_backend::FloatDType,
    ) -> TtTensor {
        if DType::from(dtype) == DType::BF16 {
            return cast_native(
                float_from_data(crate::random::float(*device, shape, distribution), device),
                DType::BF16,
            );
        }
        if DType::from(dtype) != DType::F32 {
            fail(
                "float_random",
                format_args!("shape={shape:?}, dtype={dtype:?}"),
            );
        }
        float_from_data(crate::random::float(*device, shape, distribution), device)
    }

    pub fn float_expand(tensor: TtTensor, shape: burn_backend::Shape) -> TtTensor {
        expanded(&tensor, shape, "float_expand")
    }

    macro_rules! binary {
        ($name:ident, $kind:expr) => {
            /// On the device where the data is ([`super::device_eltwise`]),
            /// unsupported inputs fail.
            pub fn $name(
                lhs: FloatTensor<TtBackend>,
                rhs: FloatTensor<TtBackend>,
            ) -> FloatTensor<TtBackend> {
                if let Some(t) = device_eltwise($kind, 0.0, &lhs, Some(&rhs)) {
                    return t;
                }
                fail(
                    stringify!($name),
                    format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
                )
            }
        };
    }
    binary!(float_add, kind::ADD);
    binary!(float_sub, kind::SUB);
    binary!(float_mul, kind::MUL);
    // Within one ulp of Flex's correctly rounded quotient, not bit for bit:
    // an SFPU approximation (`tt_kernels::sfpu::ops::kind_sfpu::DIV`).
    binary!(float_div, tt_kernels::sfpu::ops::kind_sfpu::DIV);

    /// `atan2(lhs, rhs)` on the device where the data is, within
    /// `ops::ATAN2_BOUND`, a denormal operand read as a zero of its sign; else
    /// Flex's. Operands of one shape only: a broadcast is refused, not sent to
    /// the host -- the device has no program for it (its row form would not
    /// fit a slot, `hardware-coverage.md` 10.2f), and burn-flex is not to be
    /// relied on.
    pub fn float_atan2(
        lhs: FloatTensor<TtBackend>,
        rhs: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        if lhs.shape() != rhs.shape() {
            fail("float_atan2", format_args!("lhs=({}), rhs=({}); expand operands to one shape first (hardware-coverage.md 10.2f)", context(&lhs), context(&rhs)));
        }
        let kind = tt_kernels::sfpu::ops::kind_sfpu::ATAN2;
        if let Some(t) = device_eltwise(kind, 0.0, &lhs, Some(&rhs)) {
            return t;
        }
        fail(
            "float_atan2",
            format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
        )
    }

    /// `1/x` on the device where the data is, within one ulp of Flex's
    /// (`kind_sfpu::RECIP`), unsupported inputs fail.
    pub fn float_recip(tensor: FloatTensor<TtBackend>) -> FloatTensor<TtBackend> {
        if let Some(t) = device_eltwise(tt_kernels::sfpu::ops::kind_sfpu::RECIP, 0.0, &tensor, None)
        {
            return t;
        }
        fail("float_recip", format_args!("tensor=({})", context(&tensor)))
    }

    macro_rules! unary_sfpu {
        ($name:ident, $kind:expr, $doc:literal) => {
            #[doc = $doc]
            pub fn $name(tensor: FloatTensor<TtBackend>) -> FloatTensor<TtBackend> {
                if let Some(t) = device_eltwise($kind, 0.0, &tensor, None) {
                    return t;
                }
                fail(
                    stringify!($name),
                    format_args!("tensor=({})", context(&tensor)),
                )
            }
        };
    }
    unary_sfpu!(
        float_exp,
        tt_kernels::sfpu::ops::kind_sfpu::EXP,
        "`e^x` on the device where the data is, within `ops::EXP_BOUND` of the exact value, unsupported inputs fail."
    );
    unary_sfpu!(
        float_log,
        tt_kernels::sfpu::ops::kind_sfpu::LOG,
        "`ln x` on the device where the data is, within `ops::LOG_BOUND` of the exact value, unsupported inputs fail."
    );

    /// `x / s` on the device where the data is, within one ulp of Flex's
    /// (`kind_sfpu::DIV_SCALAR`), unsupported inputs fail.
    pub fn float_div_scalar(lhs: FloatTensor<TtBackend>, rhs: Scalar) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        let s = rhs.to_f64().expect("a float scalar") as f32;
        if let Some(t) = device_eltwise(tt_kernels::sfpu::ops::kind_sfpu::DIV_SCALAR, s, &lhs, None)
        {
            return t;
        }
        fail("float_div_scalar", format_args!("lhs=({})", context(&lhs)))
    }

    /// On the device where the data is, unsupported inputs fail. The scalar is converted
    /// exactly as Flex converts it (`to_f64() as f32`, `burn-flex`
    /// `ops/binary.rs:454`).
    pub fn float_mul_scalar(lhs: FloatTensor<TtBackend>, rhs: Scalar) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        let s = rhs.to_f64().expect("a float scalar") as f32;
        if let Some(t) = device_eltwise(kind::MUL_SCALAR, s, &lhs, None) {
            return t;
        }
        fail("float_mul_scalar", format_args!("lhs=({})", context(&lhs)))
    }

    /// On the device where the data is, unsupported inputs fail; the scalar converted as
    /// Flex converts it.
    pub fn float_add_scalar(lhs: FloatTensor<TtBackend>, rhs: Scalar) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        let s = rhs.to_f64().expect("a float scalar") as f32;
        if let Some(t) = device_eltwise(kind::ADD_SCALAR, s, &lhs, None) {
            return t;
        }
        fail("float_add_scalar", format_args!("lhs=({})", context(&lhs)))
    }

    /// `x - s` as `x + (-s)`: the same bits in IEEE arithmetic, signed zeros
    /// included (`ADD_SCALAR`). On the device where the data is, unsupported inputs fail.
    pub fn float_sub_scalar(lhs: FloatTensor<TtBackend>, rhs: Scalar) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        let s = rhs.to_f64().expect("a float scalar") as f32;
        if let Some(t) = device_eltwise(kind::ADD_SCALAR, -s, &lhs, None) {
            return t;
        }
        fail("float_sub_scalar", format_args!("lhs=({})", context(&lhs)))
    }

    /// `lhs @ rhs` over the last two dimensions, with the leading ones
    /// broadcast as Burn broadcasts them, on the device both are tagged with.
    ///
    /// F32 only: other dtypes fail explicitly because the device path
    /// stages FP32 operands and nothing narrower has been gated yet. Each
    /// `[m, k] @ [k, n]` in the batch is one [`crate::server::matmul`] call.
    pub fn float_matmul(
        lhs: FloatTensor<TtBackend>,
        rhs: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        let device = lhs.device;
        assert_eq!(
            device, rhs.device,
            "float_matmul: operands on different devices ({} and {})",
            lhs.device, rhs.device
        );
        if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 {
            fail(
                "float_matmul",
                format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
            );
        }
        // Phase 9: two matrices on an engine that keeps tensors on the device
        // stay there -- operands uploaded once, the result never downloaded
        // unless a host op asks for it (`tensor.rs`).
        if lhs.is_matrix_f32() && rhs.is_matrix_f32() && crate::server::supports_dram(device) {
            let (a, b) = (lhs.to_dram().clone(), rhs.to_dram().clone());
            let (id, [m, n]) = crate::server::matmul_dram(
                device,
                a.buffer.id,
                a.transposed,
                b.buffer.id,
                b.transposed,
            );
            return device_result(device, id, [m, n]);
        }
        // A view on the left is read where it lies by the batched product
        // (its blocks may be transposed: a gradient through `K^T`); folding
        // it would first make it a plain matrix.
        if lhs.is_view() {
            if let Some(t) = batched_matmul(&lhs, &rhs) {
                return t;
            }
        }
        if let Some(t) = folded_matmul(&lhs, &rhs) {
            return t;
        }
        if let Some(t) = batched_matmul(&lhs, &rhs) {
            return t;
        }
        if let Some(t) = materialized_batched_matmul(&lhs, &rhs) {
            return t;
        }
        let ls = lhs.shape().to_vec();
        let rs = rhs.shape().to_vec();
        let rank = ls.len().max(rs.len());
        assert!(
            rank >= 2 && ls.len() >= 2 && rs.len() >= 2,
            "float_matmul needs rank >= 2"
        );
        let (m, k) = (ls[ls.len() - 2], ls[ls.len() - 1]);
        let (k2, n) = (rs[rs.len() - 2], rs[rs.len() - 1]);
        assert_eq!(k, k2, "float_matmul: [.., {m}, {k}] @ [.., {k2}, {n}]");

        // Broadcast the batch dimensions, right-aligned, as Burn does.
        let pad = |s: &[usize]| -> Vec<usize> {
            let mut v = vec![1; rank - s.len()];
            v.extend_from_slice(&s[..s.len() - 2]);
            v
        };
        let (lb, rb) = (pad(&ls), pad(&rs));
        let batch: Vec<usize> = lb
            .iter()
            .zip(&rb)
            .map(|(&a, &b)| {
                assert!(
                    a == b || a == 1 || b == 1,
                    "float_matmul: batch {lb:?} vs {rb:?}"
                );
                a.max(b)
            })
            .collect();
        let expand = |t: HostBuffer, mut tail: Vec<usize>| -> Vec<f32> {
            let mut shape = batch.clone();
            shape.append(&mut tail);
            t.expand(burn_backend::Shape::from(shape))
                .into_data()
                .to_vec::<f32>()
                .expect("F32 input")
        };
        let a = expand(lhs.into_host(), vec![m, k]);
        let b = expand(rhs.into_host(), vec![k, n]);
        let count: usize = batch.iter().product();
        let mut out = Vec::with_capacity(count * m * n);
        for i in 0..count {
            let c = crate::server::matmul(
                device,
                &a[i * m * k..(i + 1) * m * k],
                &b[i * k * n..(i + 1) * k * n],
                [m, k, n],
            );
            out.extend_from_slice(&c);
        }
        let mut shape = batch;
        shape.extend([m, n]);
        TtTensor::new(HostBuffer::from_data(TensorData::new(out, shape)), device)
    }

    /// `[.., m, k] @ [1, .., 1, k, n]` -- a Linear over a rank-N input, whose
    /// weight Burn's `linear` unsqueezes to the input's rank -- as one matrix
    /// product: the batch folds into the rows, `[prod(..) m, k] @ [k, n]`,
    /// exactly the product per batch element, and the reshapes on either side
    /// are views of resident data (`stored_dims`). On the device whenever the
    /// rank-2 product is; `None` when the rhs has a real batch (B6).
    fn folded_matmul(lhs: &TtTensor, rhs: &TtTensor) -> Option<TtTensor> {
        use burn_backend::Shape;
        let (ls, rs) = (lhs.shape().to_vec(), rhs.shape().to_vec());
        if ls.len() < 2 || rs.len() < 2 || ls.len().max(rs.len()) == 2 {
            return None;
        }
        if rs[..rs.len() - 2].iter().any(|&b| b != 1) {
            return None;
        }
        let (k, n) = (rs[rs.len() - 2], rs[rs.len() - 1]);
        if ls[ls.len() - 1] != k {
            return None;
        }
        let rows: usize = ls[..ls.len() - 1].iter().product();
        let mut out = vec![1; rs.len().saturating_sub(ls.len())];
        out.extend_from_slice(&ls[..ls.len() - 1]);
        out.push(n);
        let a = float_reshape(lhs.clone(), Shape::new([rows, k]));
        let b = float_reshape(rhs.clone(), Shape::new([k, n]));
        Some(float_reshape(float_matmul(a, b), Shape::from(out)))
    }

    /// A batched matmul of resident operands, or views of them -- attention's
    /// heads, `K^T` -- on the card: each batch element's operands a block of
    /// its operand's buffer, read where it lies, the products stacked into
    /// one `[batch m, n]` buffer (`Session::matmul_dram_batched`). Operands
    /// still on the host are uploaded, as the 2-D path does. `None` -- the
    /// host-staged path -- when a batch dimension broadcasts other than
    /// from 1, the last two dimensions of an operand are not a block of its
    /// buffer, a block is not whole tiles, or the products are not whole
    /// tile rows each.
    fn batched_matmul(lhs: &TtTensor, rhs: &TtTensor) -> Option<TtTensor> {
        let device = lhs.device;
        let (ls, rs) = (lhs.shape().to_vec(), rhs.shape().to_vec());
        let rank = ls.len().max(rs.len());
        if rank <= 2 || ls.len() < 2 || rs.len() < 2 || !crate::server::supports_dram(device) {
            return None;
        }
        if !(lhs.is_stored_f32() && rhs.is_stored_f32()) {
            return None;
        }
        let (m, k, n) = (ls[ls.len() - 2], ls[ls.len() - 1], rs[rs.len() - 1]);
        if rs[rs.len() - 2] != k {
            return None;
        }
        let lead = |s: &[usize]| -> Vec<usize> {
            let mut v = vec![1; rank - s.len()];
            v.extend_from_slice(&s[..s.len() - 2]);
            v
        };
        let (la, lb) = (lead(&ls), lead(&rs));
        let batch: Vec<usize> = la
            .iter()
            .zip(&lb)
            .map(|(&a, &b)| (a == b || a == 1 || b == 1).then_some(a.max(b)))
            .collect::<Option<_>>()?;
        let count: usize = batch.iter().product();
        if count > 1 && m % 32 != 0 {
            return None;
        }
        let view = |t: &TtTensor| {
            t.as_strided().or_else(|| {
                t.to_dram();
                t.as_strided()
            })
        };
        let (va, vb) = (view(lhs)?, view(rhs)?);
        let ba = va.matrix_blocks(&ls, &batch)?;
        let bb = vb.matrix_blocks(&rs, &batch)?;
        let items: Vec<_> = ba.into_iter().zip(bb).collect();
        let (id, dims) = crate::server::matmul_dram_batched(
            device,
            va.src.buffer.id,
            vb.src.buffer.id,
            items,
            [m, k, n],
        );
        let mut shape = batch;
        shape.extend([m, n]);
        Some(device_result_shaped(
            device,
            id,
            dims,
            burn_backend::Shape::from(shape),
            DType::F32,
        ))
    }

    /// `tensor` with dimensions `dim1` and `dim2` swapped, as a view of its
    /// device copy, if it is stored and has one (or is a view itself): nothing
    /// moves, at any rank. A swap that leaves it a plain matrix or its 2-D
    /// transpose is that, as before.
    pub(crate) fn swapped_strided(tensor: &TtTensor, dim1: usize, dim2: usize) -> Option<TtTensor> {
        if !tensor.is_storable() || dim1 == dim2 {
            return None;
        }
        let v = tensor.as_strided()?;
        let mut shape = tensor.shape().to_vec();
        shape.swap(dim1, dim2);
        Some(TtTensor::view(
            v.swapped(dim1, dim2),
            burn_backend::Shape::from(shape),
            tensor.dtype(),
            tensor.device,
        ))
    }

    /// A reshape of a device-resident F32 tensor (or a view) that changes the
    /// matrix it is stored as: a view when strides still express it -- the
    /// heads split out of a projection -- else a native copy/repack into the
    /// new matrix. Invalid shapes return `None`. A reshape that keeps a plain
    /// tensor's matrix is
    /// [`reshaped`]'s.
    fn reshaped_strided(tensor: &TtTensor, shape: &burn_backend::Shape) -> Option<TtTensor> {
        if !matches!(tensor.dtype(), DType::F32 | DType::BF16) {
            return None;
        }
        let (from, to) = (tensor.shape().to_vec(), shape.to_vec());
        let keeps = crate::tensor::stored_dims(&from) == crate::tensor::stored_dims(&to);
        if keeps && !tensor.is_view() && !tensor.is_transposed() {
            return None;
        }
        let v = tensor.as_strided()?;
        if let Some(r) = v.reshaped(&from, &to) {
            return Some(TtTensor::view(
                r,
                shape.clone(),
                tensor.dtype(),
                tensor.device,
            ));
        }
        let dims = crate::tensor::stored_dims(&to)?;
        let (id, dims) = if let Some(moves) = v.tile_moves(&from, &to) {
            crate::server::copy_blocks(tensor.device, v.src.buffer.id, moves, dims)
        } else {
            let sources = (0..from.iter().product()).map(|f| v.at(&from, f)).collect();
            crate::server::repack(tensor.device, v.src.buffer.id, sources, dims)
        };
        Some(device_result_shaped(
            tensor.device,
            id,
            dims,
            shape.clone(),
            tensor.dtype(),
        ))
    }

    /// Swapping the two dimensions of a matrix already on the device is a view
    /// of the same buffer: the device matmul reads it transposed, and a host op
    /// downloads it transposed. Unsupported inputs fail with metadata.
    pub fn float_swap_dims(
        tensor: FloatTensor<TtBackend>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<TtBackend> {
        if tensor.is_storable()
            && crate::server::is_attached(tensor.device)
            && crate::server::supports_dram(tensor.device)
        {
            tensor.to_dram();
        }
        if let Some(v) = swapped_strided(&tensor, dim1, dim2) {
            return v;
        }
        if let Some(v) = swapped_view(&tensor, dim1, dim2) {
            return v;
        }
        let device = tensor.device;
        {
            if tensor.computed_on_device() {
                fail("float_swap_dims", context(&tensor));
            }
            TtTensor::new(tensor.into_host().swap_dims(dim1, dim2), device)
        }
    }

    /// Reorder dimensions as existing strided views or host staging layouts.
    pub fn float_permute(tensor: TtTensor, axes: &[usize]) -> TtTensor {
        let rank = tensor.shape().num_dims();
        let mut seen = vec![false; rank];
        if axes.len() != rank
            || axes.iter().any(|&a| {
                if a >= rank || seen[a] {
                    true
                } else {
                    seen[a] = true;
                    false
                }
            })
        {
            fail(
                "float_permute",
                format_args!("{}, axes={axes:?}", context(&tensor)),
            );
        }
        let mut order: Vec<_> = (0..rank).collect();
        let mut tensor = tensor;
        for (i, &axis) in axes.iter().enumerate() {
            let j = order
                .iter()
                .position(|&a| a == axis)
                .expect("validated axes");
            if i != j {
                tensor = float_swap_dims(tensor, i, j);
                order.swap(i, j);
            }
        }
        tensor
    }

    /// The last two dimensions swapped; see [`float_swap_dims`].
    pub fn float_transpose(tensor: FloatTensor<TtBackend>) -> FloatTensor<TtBackend> {
        let n = tensor.shape().num_dims();
        if n < 2 {
            return tensor;
        }
        float_swap_dims(tensor, n - 2, n - 1)
    }

    /// A view of the same slots when the new shape is stored as the same
    /// matrix (`crate::tensor::stored_dims`: `[1, n]` and `[n]`, `[2, 3, 4]` and
    /// `[6, 4]`), so a reshape of a device tensor costs nothing and keeps it
    /// there; unsupported inputs fail, on the host copy.
    pub fn float_reshape(
        tensor: FloatTensor<TtBackend>,
        shape: burn_backend::Shape,
    ) -> FloatTensor<TtBackend> {
        if let Some(t) = reshaped_strided(&tensor, &shape) {
            return t;
        }
        reshaped(tensor, shape)
    }

    /// All elements summed on the device: first the stored matrix's columns,
    /// then its rows. Host F32 inputs are uploaded; unsupported inputs panic.
    /// This native reduction uses the SFPU order.
    pub fn float_sum(tensor: FloatTensor<TtBackend>) -> FloatTensor<TtBackend> {
        if let Some(result) = staged_full_reduce(&tensor, false) {
            return result;
        }
        native_full_sum(&tensor, "float_sum")
    }

    /// The full sum divided by the logical element count, on the device
    /// where the data is. Padding is excluded by both reduction kernels.
    pub fn float_mean(tensor: FloatTensor<TtBackend>) -> FloatTensor<TtBackend> {
        if let Some(result) = staged_full_reduce(&tensor, true) {
            return result;
        }
        let sum = native_full_sum(&tensor, "float_mean");
        let (id, dims) = crate::server::eltwise_op(
            tensor.device,
            tt_kernels::tensor::Eltwise {
                kind: kind_sfpu::DIV_SCALAR,
                scalar: tensor.shape().num_elements() as f32,
                scalar2: 0.0,
            },
            sum.to_dram().buffer.id,
            None,
            None,
        );
        device_result_shaped(
            tensor.device,
            id,
            dims,
            burn_backend::Shape::from(vec![1]),
            DType::F32,
        )
    }

    fn check_full_reduce(tensor: &TtTensor, op: &str) {
        assert!(
            tensor.is_stored_f32(),
            "{op}: native full reductions require a nonempty F32 tensor of rank one or more; got {:?} {:?}",
            tensor.dtype(),
            tensor.shape(),
        );
    }

    /// Mesh engines have no resident tensor storage. They still compute
    /// reductions on the SFPU, staging their inputs and scalar result.
    fn staged_full_reduce(tensor: &TtTensor, mean: bool) -> Option<TtTensor> {
        let op = if mean { "float_mean" } else { "float_sum" };
        check_full_reduce(tensor, op);
        if crate::server::supports_dram(tensor.device) {
            return None;
        }
        let values = tensor
            .host()
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("checked F32 input");
        let value = crate::server::full_reduce(
            tensor.device,
            values,
            tensor.stored().expect("checked rank"),
            mean,
        );
        // HostBuffer holds the existing host storage representation.
        // Native kernels computed the value.
        Some(TtTensor::new(
            HostBuffer::from_data(TensorData::new(vec![value], [1])),
            tensor.device,
        ))
    }

    fn native_full_sum(tensor: &TtTensor, op: &str) -> TtTensor {
        use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
        check_full_reduce(tensor, op);
        let device = tensor.device;
        let shape = tensor.shape().to_vec();
        // A full view has the same elements as its source, even when its
        // logical shape is stored differently: reduce the source directly.
        // Other views must be materialisable on the card.
        let d = match tensor.as_strided() {
            Some(view) if view.covers_source(&shape) => view.src.clone(),
            _ => tensor.to_dram().clone(),
        };
        // Preserve the full-sum's established scalar chunk order, even
        // though dimensional column reductions now support continuations.
        // Use the existing bounded chunk size for wider matrices;
        // block copies preserve ragged edges. The row sum already streams
        // long columns. Combine partial scalars in ascending column order.
        let max_cols = 32 * tt_kernels::sfpu::reduce::ROW_CHUNK;
        let [rows, cols] = [d.buffer.rows, d.buffer.cols];
        let mut acc: Option<TtTensor> = None;
        for first in (0..cols).step_by(max_cols) {
            let count = max_cols.min(cols - first);
            let chunk = if cols > max_cols {
                let (id, dims) = crate::server::copy_blocks(
                    device,
                    d.buffer.id,
                    vec![crate::BlockMove {
                        from: [0, first],
                        to: [0, 0],
                        extent: [rows, count],
                        transposed: false,
                    }],
                    [rows, count],
                );
                Some(device_result(device, id, dims))
            } else {
                None
            };
            let source = chunk.as_ref().map_or(d.buffer.id, |t| {
                t.dram().expect("made on the device").buffer.id
            });
            let (id, dims) = crate::server::reduce(device, source, ReduceOp::Sum, Axis::Cols);
            let column = device_result(device, id, dims);
            let (id, dims) = crate::server::reduce(
                device,
                column.to_dram().buffer.id,
                ReduceOp::Sum,
                Axis::Rows,
            );
            let sum = device_result_shaped(
                device,
                id,
                dims,
                burn_backend::Shape::from(vec![1]),
                DType::F32,
            );
            acc = Some(match acc {
                None => sum,
                Some(prior) => {
                    let (id, dims) = crate::server::eltwise_op(
                        device,
                        tt_kernels::tensor::Eltwise {
                            kind: kind::ADD,
                            scalar: 0.0,
                            scalar2: 0.0,
                        },
                        prior.to_dram().buffer.id,
                        Some(sum.to_dram().buffer.id),
                        None,
                    );
                    device_result_shaped(
                        device,
                        id,
                        dims,
                        burn_backend::Shape::from(vec![1]),
                        DType::F32,
                    )
                }
            });
        }
        acc.expect("a nonempty stored matrix has at least one column chunk")
    }

    fn check_reduce_dim(tensor: &TtTensor, dim: usize, op: &str) {
        if dim >= tensor.shape().num_dims() || !tensor.is_stored_f32() {
            fail(op, format_args!("{}, axis={dim}", context(tensor)));
        }
    }

    /// The sum over rows (`dim` 0) of a matrix on the device stays there, in
    /// Flex's order (`tt_kernels::sfpu::reduce::accumulate_in_order`).
    /// Unsupported inputs fail with metadata.
    pub fn float_sum_dim(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        check_reduce_dim(&tensor, dim, "float_sum_dim");
        use tt_kernels::sfpu::reduce::ReduceOp;
        if let Some(t) = device_reduce(&tensor, ReduceOp::Sum, dim) {
            return t;
        }
        if let Some(t) = device_sum_leading(&tensor, dim) {
            return t;
        }
        fail(
            "float_sum_dim",
            format_args!("tensor=({})", context(&tensor)),
        )
    }

    pub fn float_cumsum(tensor: TtTensor, dim: usize) -> TtTensor {
        scan_dim(
            tensor,
            dim,
            tt_kernels::sfpu::scan::ScanOp::Sum,
            "float_cumsum",
        )
    }

    pub fn float_cumprod(tensor: TtTensor, dim: usize) -> TtTensor {
        scan_dim(
            tensor,
            dim,
            tt_kernels::sfpu::scan::ScanOp::Prod,
            "float_cumprod",
        )
    }

    fn scan_dim(
        tensor: TtTensor,
        dim: usize,
        op: tt_kernels::sfpu::scan::ScanOp,
        name: &str,
    ) -> TtTensor {
        check_reduce_dim(&tensor, dim, name);
        let packed = pack_axis(&tensor, dim).unwrap_or_else(|| fail(name, context(&tensor)));
        let transposed = swapped_view(&packed, 0, 1).expect("packed matrix");
        let source = plain_dram(&transposed);
        let (id, dims) = crate::server::scan(tensor.device, source.buffer.id, op);
        let scanned = device_result(tensor.device, id, dims);
        let scanned = swapped_view(&scanned, 0, 1).expect("scanned matrix");
        let plain = plain_dram(&scanned);
        let shape = tensor.shape().to_vec();
        let mut out_shape = shape.clone();
        out_shape[dim] = 1;
        let sources = (0..shape.iter().product())
            .map(|mut flat| {
                let mut coords = vec![0; shape.len()];
                for k in (0..shape.len()).rev() {
                    coords[k] = flat % shape[k];
                    flat /= shape[k];
                }
                let column = coords[dim];
                coords[dim] = 0;
                let row = coords
                    .iter()
                    .zip(&out_shape)
                    .fold(0, |n, (&i, &d)| n * d + i);
                [row, column]
            })
            .collect();
        let dims = crate::tensor::stored_dims(&shape).expect("stored scan shape");
        let (id, dims) = crate::server::repack(tensor.device, plain.buffer.id, sources, dims);
        device_result_shaped(
            tensor.device,
            id,
            dims,
            burn_backend::Shape::from(shape),
            DType::F32,
        )
    }

    /// Direct product, including negative values and zero, on the SFPU.
    pub fn float_prod_dim(tensor: TtTensor, dim: usize) -> TtTensor {
        check_reduce_dim(&tensor, dim, "float_prod_dim");
        device_reduce(&tensor, tt_kernels::sfpu::reduce::ReduceOp::Prod, dim)
            .unwrap_or_else(|| fail("float_prod_dim", context(&tensor)))
    }

    pub fn float_prod(tensor: TtTensor) -> TtTensor {
        reduce_all(
            tensor,
            tt_kernels::sfpu::reduce::ReduceOp::Prod,
            "float_prod",
        )
    }

    /// Reduce logical axes in descending order without flattening a strided view.
    pub(crate) fn reduce_all(
        mut tensor: TtTensor,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        name: &str,
    ) -> TtTensor {
        let rank = tensor.shape().num_dims();
        if rank == 0 {
            fail(name, context(&tensor));
        }
        for dim in (0..rank).rev() {
            tensor =
                device_reduce(&tensor, op, dim).unwrap_or_else(|| fail(name, context(&tensor)));
        }
        reshaped(tensor, burn_backend::Shape::from(vec![1]))
    }

    /// The mean along `dim`: the sum, then a multiplication by the count's
    /// reciprocal -- each on the device where its operand is
    /// ([`float_sum_dim`], `float_mul_scalar`; within one rounding of Flex's
    /// division, inside the sum's own bound), when the tensor is on the
    /// device; supported host inputs upload as needed.
    pub fn float_mean_dim(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        check_reduce_dim(&tensor, dim, "float_mean_dim");
        let n = tensor.shape().to_vec()[dim];
        <TtBackend as FloatTensorOps<TtBackend>>::float_mul_scalar(
            float_sum_dim(tensor, dim),
            (1.0 / n as f32).into(),
        )
    }

    /// The maximum along `dim` of a device-resident matrix, on the SFPU
    /// (exactly Flex's value; `step31_reduce`), unsupported inputs fail.
    pub fn float_max_dim(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        check_reduce_dim(&tensor, dim, "float_max_dim");
        use tt_kernels::sfpu::reduce::ReduceOp;
        if let Some(t) = device_reduce(&tensor, ReduceOp::Max, dim) {
            return t;
        }
        fail(
            "float_max_dim",
            format_args!("tensor=({})", context(&tensor)),
        )
    }

    /// `tensor` reduced along `dim` on the SFPU (`Session::reduce`):
    /// every logical axis of a nonempty F32 tensor. Matrix axes and existing
    /// leading sums retain their order; other axes repack into matrix rows
    /// and use the SFPU column fold. Only address metadata is host-side.
    pub(crate) fn device_reduce(
        tensor: &TtTensor,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        dim: usize,
    ) -> Option<TtTensor> {
        device_reduce_ungated(tensor, op, dim)
    }

    /// The sum over a leading dimension `dim` with nothing before it (`[n,
    /// q.., c]`, every dimension before `dim` of size 1): in the matrix the
    /// tensor is stored as, `n` consecutive blocks of `q` rows each, summed
    /// in order. With `q = 1`, the sum over the matrix's rows; else, for `q`
    /// whole tile rows, `+0` and the blocks' row views added one after
    /// another -- both in Flex's order, exactly. Broadcast gradients take this path
    /// (layer norm's scale and shift over `[b, s, d]`).
    fn device_sum_leading(tensor: &TtTensor, dim: usize) -> Option<TtTensor> {
        use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
        let shape = tensor.shape().to_vec();
        let rank = shape.len();
        if rank < 2 || dim >= rank || dim + 1 >= rank || !tensor.is_stored_f32() {
            return None;
        }
        if shape[..dim].iter().any(|&d| d != 1) || !crate::server::supports_dram(tensor.device) {
            return None;
        }
        let n = shape[dim];
        let q: usize = shape[dim + 1..rank - 1].iter().product();
        let c = shape[rank - 1];
        let device = tensor.device;
        let d = Some(tensor.to_dram()).filter(|d| !d.transposed)?.clone();
        let mut out_shape = shape.clone();
        out_shape[dim] = 1;
        let out_shape = burn_backend::Shape::from(out_shape);
        if q == 1 {
            let (id, dims) = crate::server::reduce(device, d.buffer.id, ReduceOp::Sum, Axis::Rows);
            return Some(device_result_shaped(
                device,
                id,
                dims,
                out_shape,
                DType::F32,
            ));
        }
        if q % 32 != 0 {
            return None;
        }
        let matrix =
            TtTensor::on_device(d, burn_backend::Shape::new([n * q, c]), DType::F32, device);
        let block = |i: usize| {
            row_view(
                &matrix,
                &[burn_backend::Slice::new(
                    (i * q) as isize,
                    Some(((i + 1) * q) as isize),
                    1,
                )],
            )
        };
        // Flex's sum starts from `+0`: `0 + -0` is `+0`, which `x0 + x1`
        // would not give for two `-0`s.
        let mut acc = device_eltwise_ungated(kind::ADD_SCALAR, 0.0, &block(0)?, None)?;
        for i in 1..n {
            acc = device_eltwise_ungated(kind::ADD, 0.0, &acc, Some(&block(i)?))?;
        }
        let acc = acc.dram()?.clone();
        Some(TtTensor::on_device(acc, out_shape, DType::F32, device))
    }

    /// [`device_reduce`] without the size gate, for a composition.
    pub(crate) fn device_reduce_ungated(
        tensor: &TtTensor,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        dim: usize,
    ) -> Option<TtTensor> {
        use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
        let device = tensor.device;
        let rank = tensor.shape().num_dims();
        if !tensor.is_storable()
            || tensor.elem() != Some(op.elem())
            || !crate::server::supports_dram(device)
        {
            return None;
        }
        let shape = tensor.shape().to_vec();
        if dim >= rank || shape.contains(&0) {
            return None;
        }
        // Retain the existing in-order leading-sum path when it applies.
        if op == ReduceOp::Sum && rank > 2 && dim + 1 < rank {
            if let Some(t) = device_sum_leading(tensor, dim) {
                return Some(t);
            }
        }
        let mut out_shape = shape.clone();
        out_shape[dim] = 1;
        let physical = crate::tensor::stored_dims(&out_shape)?;
        if (rank == 2 && dim == 0) || dim + 1 == rank {
            let d = plain_dram(tensor);
            let axis = if rank == 2 && dim == 0 {
                Axis::Rows
            } else {
                Axis::Cols
            };
            let (id, dims) = crate::server::reduce(device, d.buffer.id, op, axis);
            return Some(device_result_shaped(
                device,
                id,
                dims,
                burn_backend::Shape::from(out_shape),
                tensor.dtype(),
            ));
        }
        // Rows enumerate unreduced indices in logical order, columns the
        // reduced axis. Construct only addresses on the caller's thread.
        let view = tensor
            .as_strided()
            .unwrap_or_else(|| crate::views::Strided::of(tensor.to_dram(), &shape));
        let outputs = out_shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))?;
        let count = outputs.checked_mul(shape[dim])?;
        let mut sources = Vec::with_capacity(count);
        for output in 0..outputs {
            let mut remaining = output;
            let mut indices = vec![0; rank];
            for k in (0..rank).rev() {
                indices[k] = remaining % out_shape[k];
                remaining /= out_shape[k];
            }
            for index in 0..shape[dim] {
                indices[dim] = index;
                let flat = indices.iter().zip(&shape).fold(0, |n, (&i, &d)| n * d + i);
                sources.push(view.at(&shape, flat));
            }
        }
        let (id, dims) =
            crate::server::repack(device, view.src.buffer.id, sources, [outputs, shape[dim]]);
        let packed = device_view(device, id, dims, None, tensor.dtype());
        let (id, dims) = crate::server::reduce(device, packed.to_dram().buffer.id, op, Axis::Cols);
        let reduced = device_view(device, id, dims, None, tensor.dtype());
        if physical == dims {
            return Some(TtTensor::on_device(
                reduced.to_dram().clone(),
                burn_backend::Shape::from(out_shape),
                tensor.dtype(),
                device,
            ));
        }
        let (id, dims) = crate::server::repack(
            device,
            reduced.to_dram().buffer.id,
            (0..outputs).map(|i| [i, 0]).collect(),
            physical,
        );
        Some(device_result_shaped(
            device,
            id,
            dims,
            burn_backend::Shape::from(out_shape),
            tensor.dtype(),
        ))
    }

    /// Whole tile rows, all columns, of a matrix on the device: a view of the
    /// same slots, nothing copied -- how a batch is taken from a dataset
    /// uploaded once. Unsupported inputs fail with metadata.
    /// Reverse selected logical axes through native bit-preserving copies.
    pub fn float_flip(tensor: TtTensor, axes: &[usize]) -> TtTensor {
        let mut slices = vec![burn_backend::Slice::full(); tensor.shape().num_dims()];
        for &axis in axes {
            if axis >= slices.len() {
                fail("float_flip", context(&tensor));
            }
            slices[axis] = burn_backend::Slice::with_step(0, None, -1);
        }
        sliced_native(&tensor, &slices).unwrap_or_else(|| fail("float_flip", context(&tensor)))
    }

    pub fn float_slice_assign(
        tensor: TtTensor,
        slices: &[burn_backend::Slice],
        value: TtTensor,
    ) -> TtTensor {
        slice_assign_native(tensor, slices, value)
    }

    /// Sliding logical windows: preserve payloads and append the window axis.
    pub fn float_unfold(tensor: TtTensor, dim: usize, size: usize, step: usize) -> TtTensor {
        let shape = tensor.shape().to_vec();
        assert!(dim < shape.len() && size > 0 && step > 0 && size <= shape[dim]);
        let mut output = shape.clone();
        output[dim] = (shape[dim] - size) / step + 1;
        output.push(size);
        let count = output
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .expect("unfold overflow");
        let view = tensor
            .as_strided()
            .unwrap_or_else(|| crate::views::Strided::of(tensor.to_dram(), &shape));
        let sources = (0..count)
            .map(|mut flat| {
                let offset = flat % size;
                flat /= size;
                let mut indices = vec![0; shape.len()];
                for d in (0..shape.len()).rev() {
                    indices[d] = flat % output[d];
                    flat /= output[d];
                }
                indices[dim] = indices[dim] * step + offset;
                let flat = indices.iter().zip(&shape).fold(0, |n, (&i, &d)| n * d + i);
                view.at(&shape, flat)
            })
            .collect();
        let dims = crate::tensor::stored_dims(&output).unwrap();
        let (id, dims) = crate::server::repack(tensor.device, view.src.buffer.id, sources, dims);
        device_result_shaped(tensor.device, id, dims, output.into(), tensor.dtype())
    }

    pub fn float_slice(
        tensor: FloatTensor<TtBackend>,
        slices: &[burn_backend::Slice],
    ) -> FloatTensor<TtBackend> {
        let device = tensor.device;
        if let Some(view) = row_view(&tensor, slices) {
            return view;
        }
        if let Some(t) = sliced_native(&tensor, slices) {
            return t;
        }
        {
            if tensor.computed_on_device() {
                fail("float_slice", context(&tensor));
            }
            TtTensor::new(tensor.into_host().slice(slices), device)
        }
    }

    pub(crate) fn row_view(tensor: &TtTensor, slices: &[burn_backend::Slice]) -> Option<TtTensor> {
        if !tensor.is_storable_matrix() || slices.is_empty() || slices.len() > 2 {
            return None;
        }
        if !crate::server::supports_dram(tensor.device) {
            return None;
        }
        let d = Some(tensor.to_dram()).filter(|d| !d.transposed)?;
        let [rows, cols] = [d.buffer.rows, d.buffer.cols];
        let r = &slices[0];
        let end = r.end.unwrap_or(rows as isize);
        if r.step != 1 || r.start < 0 || end < r.start || end as usize > rows {
            return None;
        }
        if let Some(c) = slices.get(1) {
            let cend = c.end.unwrap_or(cols as isize);
            if c.step != 1 || c.start != 0 || cend != cols as isize {
                return None;
            }
        }
        let (first, n) = (r.start as usize, (end - r.start) as usize);
        if first % 32 != 0 || (end as usize % 32 != 0 && end as usize != rows) || n == 0 {
            return None;
        }
        if !crate::server::supports_dram(tensor.device) {
            return None;
        }
        let (id, dims) = crate::server::slice_rows(tensor.device, d.buffer.id, first, n);
        Some(device_view(
            tensor.device,
            id,
            dims,
            Some(d.buffer.clone()),
            tensor.dtype(),
        ))
    }

    // S2 (10.2c): exact native bit operations, at any supported size.
    unary_sfpu!(
        float_neg,
        kind_sfpu::NEG,
        "`-x`, the sign bit flipped (NaNs too), on the device where the data is, unsupported inputs fail."
    );
    unary_sfpu!(
        float_abs,
        kind_sfpu::ABS,
        "`|x|`, the sign bit cleared (NaNs too), on the device where the data is, unsupported inputs fail."
    );
    unary_sfpu!(
        float_sign,
        kind_sfpu::SIGN,
        "Flex's `sign` (a NaN itself, `+0` for a zero) on the device where the data is."
    );

    fn scalar_f32(s: Scalar) -> f32 {
        use num_traits::ToPrimitive;
        s.to_f32().expect("a float scalar")
    }

    fn op2(kind: u32, scalar: f32, scalar2: f32) -> tt_kernels::tensor::Eltwise {
        tt_kernels::tensor::Eltwise {
            kind,
            scalar,
            scalar2,
        }
    }

    /// On the device where the data is, as `f32::clamp`; bounds `f32::clamp`
    /// rejects (a NaN or crossed bounds) fail explicitly.
    pub fn float_clamp(
        tensor: FloatTensor<TtBackend>,
        min: Scalar,
        max: Scalar,
    ) -> FloatTensor<TtBackend> {
        let (lo, hi) = (scalar_f32(min), scalar_f32(max));
        if !(lo.is_nan() || hi.is_nan() || lo > hi) {
            if let Some(t) = device_op(op2(kind_sfpu::CLAMP, lo, hi), &tensor, None, None) {
                return t;
            }
        }
        fail("float_clamp", format_args!("tensor=({})", context(&tensor)))
    }

    pub fn float_clamp_min(tensor: FloatTensor<TtBackend>, min: Scalar) -> FloatTensor<TtBackend> {
        let s = scalar_f32(min);
        if let Some(t) = device_eltwise(kind_sfpu::CLAMP_MIN, s, &tensor, None) {
            return t;
        }
        fail(
            "float_clamp_min",
            format_args!("tensor=({})", context(&tensor)),
        )
    }

    pub fn float_clamp_max(tensor: FloatTensor<TtBackend>, max: Scalar) -> FloatTensor<TtBackend> {
        let s = scalar_f32(max);
        if let Some(t) = device_eltwise(kind_sfpu::CLAMP_MAX, s, &tensor, None) {
            return t;
        }
        fail(
            "float_clamp_max",
            format_args!("tensor=({})", context(&tensor)),
        )
    }

    macro_rules! compare {
        ($name:ident, $kind:expr) => {
            /// An IEEE comparison on the device where the data is (a row or
            /// column broadcast on the right), its `Bool` resident; unsupported inputs fail.
            pub fn $name(
                lhs: FloatTensor<TtBackend>,
                rhs: FloatTensor<TtBackend>,
                out_dtype: burn_backend::BoolDType,
            ) -> BoolTensor<TtBackend> {
                if let Some(t) = device_eltwise($kind, 0.0, &lhs, Some(&rhs)) {
                    return retyped(t, out_dtype.into());
                }
                fail(
                    stringify!($name),
                    format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
                )
            }
        };
    }
    compare!(float_equal, kind_sfpu::EQ);
    compare!(float_not_equal, kind_sfpu::NE);
    compare!(float_greater, kind_sfpu::GT);
    compare!(float_greater_equal, kind_sfpu::GE);
    compare!(float_lower, kind_sfpu::LT);
    compare!(float_lower_equal, kind_sfpu::LE);

    macro_rules! compare_elem {
        ($name:ident, $kind:expr) => {
            /// Against a scalar converted as Flex converts it (`f64`, then
            /// `f32`), on the device where the data is; unsupported inputs fail.
            pub fn $name(
                lhs: FloatTensor<TtBackend>,
                rhs: Scalar,
                out_dtype: burn_backend::BoolDType,
            ) -> BoolTensor<TtBackend> {
                use num_traits::ToPrimitive;
                let s = rhs.to_f64().expect("a float scalar") as f32;
                if let Some(t) = device_eltwise($kind, s, &lhs, None) {
                    return retyped(t, out_dtype.into());
                }
                fail(stringify!($name), format_args!("lhs=({})", context(&lhs)))
            }
        };
    }
    compare_elem!(float_equal_elem, kind_sfpu::EQ_S);
    compare_elem!(float_not_equal_elem, kind_sfpu::NE_S);
    compare_elem!(float_greater_elem, kind_sfpu::GT_S);
    compare_elem!(float_greater_equal_elem, kind_sfpu::GE_S);
    compare_elem!(float_lower_elem, kind_sfpu::LT_S);
    compare_elem!(float_lower_equal_elem, kind_sfpu::LE_S);

    macro_rules! predicate {
        ($name:ident, $kind:expr) => {
            /// On the device where the data is, its `Bool` resident; unsupported inputs fail.
            pub fn $name(
                tensor: FloatTensor<TtBackend>,
                out_dtype: burn_backend::BoolDType,
            ) -> BoolTensor<TtBackend> {
                if let Some(t) = device_eltwise($kind, 0.0, &tensor, None) {
                    return retyped(t, out_dtype.into());
                }
                fail(
                    stringify!($name),
                    format_args!("tensor=({})", context(&tensor)),
                )
            }
        };
    }
    predicate!(float_is_nan, kind_sfpu::IS_NAN);
    predicate!(float_is_inf, kind_sfpu::IS_INF);

    /// Select raw datums on B with checked resident indices along any logical axis.
    pub fn float_gather(
        dim: usize,
        tensor: FloatTensor<TtBackend>,
        indices: IntTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        gather_native(dim, tensor, indices)
    }

    /// Pooling uses the same bit-preserving resident selection.
    pub(crate) fn float_gather_bits(dim: usize, tensor: TtTensor, indices: TtTensor) -> TtTensor {
        gather_native(dim, tensor, indices)
    }

    /// Fold duplicate indices in logical axis order; additions stay on Tensix.
    pub fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<TtBackend>,
        indices: IntTensor<TtBackend>,
        value: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        scatter_add_native(dim, tensor, indices, value)
    }

    /// Select slices with resident indices along any logical axis.
    pub fn float_select(
        tensor: FloatTensor<TtBackend>,
        dim: usize,
        indices: IntTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        select_native(tensor, dim, indices)
    }

    /// Add selected slices in index order without downloading indices.
    pub fn float_select_add(
        tensor: FloatTensor<TtBackend>,
        dim: usize,
        indices: IntTensor<TtBackend>,
        value: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        select_add_native(tensor, dim, indices, value)
    }

    /// `mask ? value : x` on the device where the data is (the mask may be a
    /// row or column of `x`'s matrix), every bit of `x` kept; unsupported inputs fail.
    pub fn float_mask_fill(
        tensor: FloatTensor<TtBackend>,
        mask: BoolTensor<TtBackend>,
        value: Scalar,
    ) -> FloatTensor<TtBackend> {
        if let Some(t) = device_eltwise(
            kind_sfpu::MASK_FILL,
            scalar_f32(value),
            &tensor,
            Some(&mask),
        ) {
            return t;
        }
        fail(
            "float_mask_fill",
            format_args!("tensor=({}), mask=({})", context(&tensor), context(&mask)),
        )
    }

    /// `mask ? value : x`, all three one shape, on the device where the data is;
    /// unsupported inputs fail.
    pub fn float_mask_where(
        tensor: FloatTensor<TtBackend>,
        mask: BoolTensor<TtBackend>,
        value: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        let op = op2(kind_sfpu::MASK_WHERE, 0.0, 0.0);
        if mask.shape() == tensor.shape() {
            if let Some(t) = device_op(op, &tensor, Some(&mask), Some(&value)) {
                return t;
            }
        }
        fail(
            "float_mask_where",
            format_args!(
                "tensor=({}), mask=({}), value=({})",
                context(&tensor),
                context(&mask),
                context(&value)
            ),
        )
    }

    // S4 (10.2d): approximations within their derived bounds.
    unary_sfpu!(
        float_sqrt,
        kind_sfpu::SQRT,
        "`sqrt x` on the device where the data is, within one ulp of the correctly rounded root, unsupported inputs fail."
    );
    unary_sfpu!(
        float_log1p,
        kind_sfpu::LOG1P,
        "`ln(1 + x)` on the device where the data is, within `ops::LOG1P_BOUND`, unsupported inputs fail."
    );
    // S4 (10.2e).
    unary_sfpu!(
        float_tanh,
        kind_sfpu::TANH,
        "`tanh x` on the device where the data is, within `ops::TANH_BOUND`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_erf,
        kind_sfpu::ERF,
        "`erf x` on the device where the data is, within `ops::ERF_BOUND`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_sinh,
        kind_sfpu::SINH,
        "`sinh x` on the device where the data is, within `ops::SINH_BOUND`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_cosh,
        kind_sfpu::COSH,
        "`cosh x` on the device where the data is, within `ops::COSH_BOUND`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_asinh,
        kind_sfpu::ASINH,
        "`asinh x` on the device where the data is, within `ops::ASINH_BOUND`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_acosh,
        kind_sfpu::ACOSH,
        "`acosh x` on the device where the data is, within `ops::ACOSH_BOUND`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_atanh,
        kind_sfpu::ATANH,
        "`atanh x` on the device where the data is, within `ops::ATANH_BOUND`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_sin,
        kind_sfpu::SIN,
        "`sin x` on the device where the data is, within `ops::SIN_BOUND` for every finite `x`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_cos,
        kind_sfpu::COS,
        "`cos x` on the device where the data is, within `ops::COS_BOUND` for every finite `x`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_tan,
        kind_sfpu::TAN,
        "`tan x` on the device where the data is, within `ops::TAN_BOUND` for every finite `x`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_atan,
        kind_sfpu::ATAN,
        "`atan x` on the device where the data is, within `ops::ATAN_BOUND`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_asin,
        kind_sfpu::ASIN,
        "`asin x` on the device where the data is, within `ops::ASIN_BOUND`, unsupported inputs fail."
    );
    unary_sfpu!(
        float_acos,
        kind_sfpu::ACOS,
        "`acos x` on the device where the data is, within `ops::ACOS_BOUND`, unsupported inputs fail."
    );

    /// `x^y`, `y` a tensor of `x`'s shape, on the device where the data is
    /// (within `ops::pow_bound`); unsupported inputs fail.
    pub fn float_powf(
        lhs: FloatTensor<TtBackend>,
        rhs: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        if let Some(t) = device_pow(&lhs, PowY::Tensor(&rhs)) {
            return t;
        }
        fail(
            "float_powf",
            format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
        )
    }

    /// `x^y`, `y` an integer tensor, as Flex's `powf(x, y as f32)`.
    pub fn float_powi(
        lhs: FloatTensor<TtBackend>,
        rhs: IntTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        if let Some(t) = device_pow(&lhs, PowY::Int(&rhs)) {
            return t;
        }
        fail(
            "float_powi",
            format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
        )
    }

    /// `x^v` for a non-integer `v`, as Flex's `float_powf_scalar_impl`.
    pub fn float_powf_scalar_impl(
        tensor: FloatTensor<TtBackend>,
        value: Scalar,
    ) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        let v = value.to_f64().expect("a float scalar") as f32;
        if let Some(t) = device_pow(&tensor, PowY::Scalar(v)) {
            return t;
        }
        fail(
            "float_powf_scalar_impl",
            format_args!("tensor=({})", context(&tensor)),
        )
    }

    /// Flex's dispatch, on the device's ops: `0` ones, `1` the tensor, `2` a
    /// product, `-1` and `-2` reciprocals, anything else `powf`.
    pub fn float_powi_scalar(lhs: FloatTensor<TtBackend>, rhs: Scalar) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        match rhs.to_i64().expect("an integer exponent") {
            0 => {
                if let Some(t) = device_eltwise(kind_sfpu::FILL, 1.0, &lhs, None) {
                    return t;
                }
                fail("float_powi_scalar", format_args!("lhs=({})", context(&lhs)))
            }
            1 => lhs,
            2 => float_mul(lhs.clone(), lhs),
            -1 => float_recip(lhs),
            -2 => float_recip(float_mul(lhs.clone(), lhs)),
            _ => float_powf_scalar_impl(lhs, rhs),
        }
    }

    /// Flex's: an integer exponent is `powi_scalar`'s.
    pub fn float_powf_scalar(
        tensor: FloatTensor<TtBackend>,
        value: Scalar,
    ) -> FloatTensor<TtBackend> {
        match value.try_as_integer() {
            Some(exp) => float_powi_scalar(tensor, exp),
            None => float_powf_scalar_impl(tensor, value),
        }
    }

    /// A cast to the tensor's own dtype is the tensor -- Burn's compositions
    /// cast to `F32` what already is (`hard_sigmoid`, forward and backward),
    /// and a host round trip for nothing would undo residency. Other casts are
    /// unsupported (S6).
    pub fn float_round(tensor: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::ROUND, 0.0, &tensor, None)
            .unwrap_or_else(|| fail("float_round", context(&tensor)))
    }
    pub fn float_floor(tensor: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::FLOOR, 0.0, &tensor, None)
            .unwrap_or_else(|| fail("float_floor", context(&tensor)))
    }
    pub fn float_ceil(tensor: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::CEIL, 0.0, &tensor, None)
            .unwrap_or_else(|| fail("float_ceil", context(&tensor)))
    }
    pub fn float_trunc(tensor: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::TRUNC, 0.0, &tensor, None)
            .unwrap_or_else(|| fail("float_trunc", context(&tensor)))
    }
    pub fn float_into_int(tensor: TtTensor, out_dtype: IntDType) -> TtTensor {
        if out_dtype == IntDType::I32 {
            if let Some(t) = device_eltwise(kind_sfpu::F32_TO_I32, 0.0, &tensor, None) {
                return t;
            }
        }
        fail(
            "float_into_int",
            format_args!("{}, out_dtype={out_dtype:?}", context(&tensor)),
        )
    }

    pub fn float_cast(
        tensor: FloatTensor<TtBackend>,
        dtype: burn_backend::FloatDType,
    ) -> FloatTensor<TtBackend> {
        cast_native(tensor, DType::from(dtype))
    }

    pub fn float_device(tensor: &FloatTensor<TtBackend>) -> Device<TtBackend> {
        tensor.device
    }

    /// On `device`, and for an F32 matrix on an engine that keeps tensors,
    /// resident there: `Tensor::to_device` is how a caller says "this lives on
    /// the card" (a dataset, say, uploaded once).
    pub fn float_to_device(
        tensor: FloatTensor<TtBackend>,
        device: &Device<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        to_device_resident(tensor, device)
    }

    pub fn float_into_data(
        tensor: FloatTensor<TtBackend>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        async move { Ok(tensor.into_host().into_data()) }
    }
}

#[path = "pool.rs"]
mod pooling;

#[path = "conv.rs"]
mod convolution;

pub mod module {
    pub use super::convolution::{
        conv2d, conv2d_bias_backward, conv2d_weight_backward, conv2d_x_backward, conv_transpose2d,
        conv_transpose2d_bias_backward, conv_transpose2d_weight_backward,
        conv_transpose2d_x_backward, unfold4d,
    };
    pub use super::pooling::{
        adaptive_avg_pool2d, adaptive_avg_pool2d_backward, avg_pool2d, avg_pool2d_backward,
        max_pool2d, max_pool2d_with_indices, max_pool2d_with_indices_backward,
    };
    use super::*;
    use burn_backend::Shape;

    /// Pinned Burn's attention ordering, with replayable causal geometry.
    /// BF16 inputs widen once; every intermediate remains F32 until the output.
    pub fn attention(
        query: TtTensor,
        key: TtTensor,
        value: TtTensor,
        mask: Option<TtTensor>,
        attn_bias: Option<TtTensor>,
        options: burn_backend::ops::AttentionModuleOptions,
    ) -> TtTensor {
        let dtype = float_compute_dtype(&[&query, &key, &value]);
        let qs = query.shape().to_vec();
        let ks = key.shape().to_vec();
        let vs = value.shape().to_vec();
        assert!(
            qs.len() == 4 && ks.len() == 4 && vs.len() == 4,
            "attention requires rank four"
        );
        assert_eq!(qs[3], ks[3], "attention head dimensions differ");
        assert_eq!(ks[2], vs[2], "attention key/value sequence lengths differ");
        assert!(
            !qs.contains(&0) && !ks.contains(&0) && !vs.contains(&0),
            "attention requires nonempty inputs"
        );
        assert_eq!(query.device, key.device, "attention devices differ");
        assert_eq!(query.device, value.device, "attention devices differ");
        if let Some(cap) = options.softcap {
            assert!(cap > 0.0, "softcap must be positive");
        }
        let device = query.device;
        let scale = options.scale.unwrap_or_else(|| 1.0 / (qs[3] as f64).sqrt());
        let query = float_compute_input(query);
        let key = float_compute_input(key);
        let value = float_compute_input(value);
        let mut scores = float::float_mul_scalar(
            float::float_matmul(query, float::float_transpose(key)),
            scale.into(),
        );
        if let Some(cap) = options.softcap {
            scores = float::float_mul_scalar(
                float::float_tanh(float::float_div_scalar(scores, cap.into())),
                cap.into(),
            );
        }
        if let Some(mask) = mask {
            assert_eq!(device, mask.device, "attention mask device differs");
            scores = float::float_mask_fill(scores, mask, f32::NEG_INFINITY.into());
        }
        if options.is_causal {
            let [sq, sk] = [qs[2], ks[2]];
            let offset = sk as i128 - sq as i128;
            let bits = (0..sq)
                .flat_map(|r| (0..sk).map(move |c| u32::from(c as i128 > r as i128 + offset)))
                .collect();
            let id = crate::server::metadata(device, bits, [sq, sk], Elem::Bool);
            let bool_dtype = burn_backend::get_device_settings::<TtBackend>(&device).bool_dtype;
            let mask = device_result_shaped(
                device,
                id,
                [sq, sk],
                [1, 1, sq, sk].into(),
                bool_dtype.into(),
            );
            scores = float::float_mask_fill(scores, mask, f32::NEG_INFINITY.into());
        }
        if let Some(bias) = attn_bias {
            assert_eq!(device, bias.device, "attention bias device differs");
            assert_eq!(dtype, bias.dtype(), "attention bias dtype differs");
            scores = float::float_add(scores, float_compute_input(bias));
        }
        let max =
            float::float_clamp_min(float::float_max_dim(scores.clone(), 3), (-f32::MAX).into());
        let exp = float::float_exp(float::float_sub(scores, max));
        let sum = float::float_clamp_min(
            float::float_sum_dim(exp.clone(), 3),
            f32::MIN_POSITIVE.into(),
        );
        cast_native(
            float::float_matmul(float::float_div(exp, sum), value),
            dtype,
        )
    }

    /// Burn's own composition (`ModuleOps::embedding`'s default: a
    /// `select` of the table's rows, reshaped), through this backend's ops,
    /// so the lookup runs on the device where the table is
    /// (`float::float_select`); unsupported lookups fail with metadata.
    pub fn embedding(
        weights: FloatTensor<TtBackend>,
        indices: IntTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        let [batch, seq] = indices.shape().dims();
        let d_model = weights.shape().to_vec()[1];
        let indices =
            <TtBackend as IntTensorOps<TtBackend>>::int_reshape(indices, Shape::new([batch * seq]));
        let out = <TtBackend as FloatTensorOps<TtBackend>>::float_select(weights, 0, indices);
        <TtBackend as FloatTensorOps<TtBackend>>::float_reshape(
            out,
            Shape::new([batch, seq, d_model]),
        )
    }

    /// Burn's own composition (`ModuleOps::embedding_backward`'s default: a
    /// `select_add` of the gradient's rows into zeros), through this
    /// backend's ops (`float::float_select_add`).
    pub fn embedding_backward(
        weights: FloatTensor<TtBackend>,
        output_grad: FloatTensor<TtBackend>,
        indices: IntTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        let [batch, seq] = indices.shape().dims();
        let [n, d] = weights.shape().dims();
        let device = weights.device;
        let dtype = output_grad.dtype();
        let indices =
            <TtBackend as IntTensorOps<TtBackend>>::int_reshape(indices, Shape::new([batch * seq]));
        let grad = <TtBackend as FloatTensorOps<TtBackend>>::float_reshape(
            output_grad,
            Shape::new([batch * seq, d]),
        );
        let zeros = <TtBackend as FloatTensorOps<TtBackend>>::float_zeros(
            Shape::new([n, d]),
            &device,
            dtype.into(),
        );
        <TtBackend as FloatTensorOps<TtBackend>>::float_select_add(zeros, 0, indices, grad)
    }

    /// `dW = x^T @ dY` over every row of every batch element at once:
    /// `[d, prod(..)] @ [prod(..), e]`, one matrix product that the device
    /// runs on resident operands (the transpose is a view). Burn's default
    /// takes a batched product per element and sums over the batch, which
    /// for a rank-N input never reaches the device's rank-2 path. The same
    /// sum, in one accumulation rather than per element then across; for a
    /// rank-2 input, the default's exact computation.
    pub fn linear_weight_backward(
        x: FloatTensor<TtBackend>,
        output_grad: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        let (xs, gs) = (x.shape().to_vec(), output_grad.shape().to_vec());
        let (d, e) = (xs[xs.len() - 1], gs[gs.len() - 1]);
        let rows: usize = xs[..xs.len() - 1].iter().product();
        let x = float::float_reshape(x, Shape::new([rows, d]));
        let g = float::float_reshape(output_grad, Shape::new([rows, e]));
        float::float_matmul(float::float_swap_dims(x, 0, 1), g)
    }

    /// `db`, the sum of `dY` over every row of every batch element: one sum
    /// over the rows of `[prod(..), e]` -- on the device, in Flex's order
    /// over those rows (`float_sum_dim`) -- where Burn's default sums one
    /// leading dimension at a time, and a rank-N sum is not a device one.
    pub fn linear_bias_backward(output_grad: FloatTensor<TtBackend>) -> FloatTensor<TtBackend> {
        let gs = output_grad.shape().to_vec();
        let e = gs[gs.len() - 1];
        let rows: usize = gs[..gs.len() - 1].iter().product();
        let g = float::float_reshape(output_grad, Shape::new([rows, e]));
        float::float_reshape(float::float_sum_dim(g, 0), Shape::new([e]))
    }
}

pub mod activation {
    use super::*;
    use burn_backend::Scalar;
    use tt_kernels::kind;

    /// On the device where the data is, unsupported inputs fail.
    pub fn relu(tensor: FloatTensor<TtBackend>) -> FloatTensor<TtBackend> {
        if let Some(t) = device_eltwise(kind::RELU, 0.0, &tensor, None) {
            return t;
        }
        fail("relu", format_args!("tensor=({})", context(&tensor)))
    }

    /// Burn's own composition (`ActivationOps::softmax`'s default) -- a max, a
    /// broadcast subtract, `exp`, a sum and a broadcast divide -- through this
    /// backend's ops, so on a device-resident matrix every step runs on the
    /// device; unsupported inputs fail with metadata.
    pub fn softmax(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        match device_softmax(&tensor, dim, false, false) {
            Some(t) => t,
            None => fail("softmax", format_args!("tensor=({})", context(&tensor))),
        }
    }

    /// Burn's own composition (`ActivationOps::log_softmax`'s default: the
    /// max-shifted log-sum-exp), on the device; unsupported inputs fail.
    pub fn log_softmax(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        match device_softmax(&tensor, dim, true, false) {
            Some(t) => t,
            None => fail("log_softmax", format_args!("tensor=({})", context(&tensor))),
        }
    }

    /// Burn's own composition (`ActivationOps::softmin`'s default: the
    /// softmax of `-x`), on the device -- the negation exact; unsupported inputs fail.
    pub fn softmin(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        match device_softmax(&tensor, dim, false, true) {
            Some(t) => t,
            None => fail("softmin", format_args!("tensor=({})", context(&tensor))),
        }
    }

    /// Softmax (or log-softmax) of a resident matrix along `dim` -- of its
    /// negation where `negate` (softmin) -- every step on the device: decided
    /// once, on the whole input, so the small per-row statistics in the middle
    /// stay there too.
    fn device_softmax(t: &TtTensor, dim: usize, log: bool, negate: bool) -> Option<TtTensor> {
        use super::float::device_reduce_ungated as reduce;
        use tt_kernels::sfpu::ops::kind_sfpu;
        use tt_kernels::sfpu::reduce::ReduceOp;
        if !resident_matrix(t) || dim > 1 {
            return None;
        }
        let neg;
        let t = if negate {
            neg = device_eltwise_ungated(kind_sfpu::NEG, 0.0, t, None)?;
            &neg
        } else {
            t
        };
        let max = reduce(t, ReduceOp::Max, dim)?;
        let shifted = device_eltwise_ungated(kind::SUB, 0.0, t, Some(&max))?;
        let exp = device_eltwise_ungated(kind_sfpu::EXP, 0.0, &shifted, None)?;
        let sum = reduce(&exp, ReduceOp::Sum, dim)?;
        if log {
            let log_sum = device_eltwise_ungated(kind_sfpu::LOG, 0.0, &sum, None)?;
            device_eltwise_ungated(kind::SUB, 0.0, &shifted, Some(&log_sum))
        } else {
            device_eltwise_ungated(kind_sfpu::DIV, 0.0, &exp, Some(&sum))
        }
    }

    /// Is `t` an F32 matrix on a device that keeps tensors in GDDR, with a
    /// supported device storage (softmax is an approximation)?
    fn resident_matrix(t: &TtTensor) -> bool {
        t.is_matrix_f32() && crate::server::supports_dram(t.device) && !t.to_dram().transposed
    }

    /// On the device where the data is, unsupported inputs fail.
    pub fn relu_backward(
        output: FloatTensor<TtBackend>,
        grad: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        if let Some(t) = device_eltwise(kind::RELU_BACKWARD, 0.0, &output, Some(&grad)) {
            return t;
        }
        fail(
            "relu_backward",
            format_args!("output=({}), grad=({})", context(&output), context(&grad)),
        )
    }

    /// Flex's `x >= 0 ? x : slope * x`, one op on the device where the data
    /// is; unsupported inputs fail.
    pub fn leaky_relu(
        tensor: FloatTensor<TtBackend>,
        negative_slope: Scalar,
    ) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        let ns = negative_slope.to_f32().expect("a float scalar");
        if let Some(t) = device_eltwise(
            tt_kernels::sfpu::ops::kind_sfpu::LEAKY_RELU,
            ns,
            &tensor,
            None,
        ) {
            return t;
        }
        fail("leaky_relu", format_args!("tensor=({})", context(&tensor)))
    }

    /// Flex's `(alpha * x + beta).clamp(0, 1)`, one op on the device; unsupported inputs fail.
    pub fn hard_sigmoid(
        tensor: FloatTensor<TtBackend>,
        alpha: Scalar,
        beta: Scalar,
    ) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        let (a, b) = (
            alpha.to_f32().expect("a float"),
            beta.to_f32().expect("a float"),
        );
        let op = tt_kernels::tensor::Eltwise {
            kind: tt_kernels::sfpu::ops::kind_sfpu::HARD_SIGMOID,
            scalar: a,
            scalar2: b,
        };
        if let Some(t) = device_op(op, &tensor, None, None) {
            return t;
        }
        fail(
            "hard_sigmoid",
            format_args!("tensor=({})", context(&tensor)),
        )
    }

    /// Flex's `x >= 0 ? x : alpha * x`, one op on the device -- `alpha` the
    /// same shape, or a row of per-channel slopes; unsupported inputs fail.
    pub fn prelu(
        tensor: FloatTensor<TtBackend>,
        alpha: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        if let Some(t) = device_eltwise(
            tt_kernels::sfpu::ops::kind_sfpu::PRELU,
            0.0,
            &tensor,
            Some(&alpha),
        ) {
            return t;
        }
        fail(
            "prelu",
            format_args!("tensor=({}), alpha=({})", context(&tensor), context(&alpha)),
        )
    }

    // S4 (10.2e): one SFPU op each, as Flex's fused closures are.
    macro_rules! unary {
        ($name:ident, $kind:ident, $doc:literal) => {
            #[doc = $doc]
            pub fn $name(tensor: FloatTensor<TtBackend>) -> FloatTensor<TtBackend> {
                use tt_kernels::sfpu::ops::kind_sfpu;
                if let Some(t) = device_eltwise(kind_sfpu::$kind, 0.0, &tensor, None) {
                    return t;
                }
                fail(
                    stringify!($name),
                    format_args!("tensor=({})", context(&tensor)),
                )
            }
        };
    }
    macro_rules! binary {
        ($name:ident, $kind:ident, $doc:literal) => {
            #[doc = $doc]
            pub fn $name(
                a: FloatTensor<TtBackend>,
                grad: FloatTensor<TtBackend>,
            ) -> FloatTensor<TtBackend> {
                use tt_kernels::sfpu::ops::kind_sfpu;
                if a.shape() == grad.shape() {
                    if let Some(t) = device_eltwise(kind_sfpu::$kind, 0.0, &a, Some(&grad)) {
                        return t;
                    }
                }
                fail(
                    stringify!($name),
                    format_args!("a=({}), grad=({})", context(&a), context(&grad)),
                )
            }
        };
    }
    unary!(
        sigmoid,
        SIGMOID,
        "Flex's two-branch `sigmoid` on the device where the data is, within `ops::SIGMOID_BOUND`; unsupported inputs fail."
    );
    binary!(
        sigmoid_backward,
        SIGMOID_BACKWARD,
        "Flex's `g * s * (1 - s)`, exact, on the device where the data is; unsupported inputs fail."
    );
    unary!(
        gelu,
        GELU,
        "Flex's `0.5 x (1 + erf(x / sqrt 2))` on the device where the data is, within `ops::GELU_BOUND`; unsupported inputs fail."
    );
    binary!(
        gelu_backward,
        GELU_BACKWARD,
        "`g (Phi(x) + x phi(x))` on the device where the data is, within `ops::gelu_backward_bound`; unsupported inputs fail."
    );
    unary!(
        log_sigmoid,
        LOG_SIGMOID,
        "Flex's two-branch `log_sigmoid` on the device where the data is, within `ops::LOG_SIGMOID_BOUND`; unsupported inputs fail."
    );
    binary!(
        log_sigmoid_backward,
        LOG_SIGMOID_BACKWARD,
        "Flex's `g * sigmoid(-x)` on the device where the data is, within `ops::SIGMOID_BOUND` and the product's rounding; unsupported inputs fail."
    );
}

pub mod int {
    use super::*;

    pub fn int_gather(dim: usize, tensor: TtTensor, indices: TtTensor) -> TtTensor {
        gather_native(dim, tensor, indices)
    }
    pub fn int_select(tensor: TtTensor, dim: usize, indices: TtTensor) -> TtTensor {
        select_native(tensor, dim, indices)
    }
    pub fn int_scatter_add(
        dim: usize,
        tensor: TtTensor,
        indices: TtTensor,
        value: TtTensor,
    ) -> TtTensor {
        scatter_add_native(dim, tensor, indices, value)
    }
    pub fn int_select_add(
        tensor: TtTensor,
        dim: usize,
        indices: TtTensor,
        value: TtTensor,
    ) -> TtTensor {
        select_add_native(tensor, dim, indices, value)
    }
    use num_traits::ToPrimitive;

    pub fn int_add(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_ADD, 0.0, &lhs, Some(&rhs)).unwrap_or_else(|| {
            fail(
                "int_add",
                format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
            )
        })
    }
    pub fn int_add_scalar(lhs: TtTensor, rhs: burn_backend::Scalar) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        device_eltwise(kind_sfpu::INT_ADD_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("int_add_scalar", context(&lhs)))
    }
    pub fn int_sub(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_SUB, 0.0, &lhs, Some(&rhs)).unwrap_or_else(|| {
            fail(
                "int_sub",
                format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
            )
        })
    }
    pub fn int_sub_scalar(lhs: TtTensor, rhs: burn_backend::Scalar) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        device_eltwise(kind_sfpu::INT_SUB_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("int_sub_scalar", context(&lhs)))
    }
    pub fn int_mul(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_MUL, 0.0, &lhs, Some(&rhs)).unwrap_or_else(|| {
            fail(
                "int_mul",
                format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
            )
        })
    }
    pub fn int_mul_scalar(lhs: TtTensor, rhs: burn_backend::Scalar) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        device_eltwise(kind_sfpu::INT_MUL_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("int_mul_scalar", context(&lhs)))
    }

    pub fn int_div(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_DIV, 0.0, &lhs, Some(&rhs))
            .unwrap_or_else(|| fail("div(int)", context(&lhs)))
    }
    pub fn int_remainder(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_REM, 0.0, &lhs, Some(&rhs))
            .unwrap_or_else(|| fail("remainder(int)", context(&lhs)))
    }
    pub fn int_div_scalar(lhs: TtTensor, rhs: burn_backend::Scalar) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        device_eltwise(kind_sfpu::INT_DIV_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("div_scalar(int)", context(&lhs)))
    }
    pub fn int_remainder_scalar(lhs: TtTensor, rhs: burn_backend::Scalar) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        device_eltwise(kind_sfpu::INT_REM_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("remainder_scalar(int)", context(&lhs)))
    }
    pub fn bitwise_and(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_AND, 0.0, &lhs, Some(&rhs)).unwrap_or_else(|| {
            fail(
                "bitwise_and",
                format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
            )
        })
    }
    pub fn bitwise_and_scalar(lhs: TtTensor, rhs: burn_backend::Scalar) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        device_eltwise(kind_sfpu::INT_AND_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("bitwise_and_scalar", context(&lhs)))
    }
    pub fn bitwise_or(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_OR, 0.0, &lhs, Some(&rhs)).unwrap_or_else(|| {
            fail(
                "bitwise_or",
                format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
            )
        })
    }
    pub fn bitwise_or_scalar(lhs: TtTensor, rhs: burn_backend::Scalar) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        device_eltwise(kind_sfpu::INT_OR_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("bitwise_or_scalar", context(&lhs)))
    }
    pub fn bitwise_xor(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_XOR, 0.0, &lhs, Some(&rhs)).unwrap_or_else(|| {
            fail(
                "bitwise_xor",
                format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
            )
        })
    }
    pub fn bitwise_xor_scalar(lhs: TtTensor, rhs: burn_backend::Scalar) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        device_eltwise(kind_sfpu::INT_XOR_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("bitwise_xor_scalar", context(&lhs)))
    }
    pub fn bitwise_left_shift(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_SHL, 0.0, &lhs, Some(&rhs)).unwrap_or_else(|| {
            fail(
                "bitwise_left_shift",
                format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
            )
        })
    }
    pub fn bitwise_left_shift_scalar(lhs: TtTensor, rhs: burn_backend::Scalar) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        device_eltwise(kind_sfpu::INT_SHL_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("bitwise_left_shift_scalar", context(&lhs)))
    }
    pub fn bitwise_right_shift(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_SHR, 0.0, &lhs, Some(&rhs)).unwrap_or_else(|| {
            fail(
                "bitwise_right_shift",
                format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
            )
        })
    }
    pub fn bitwise_right_shift_scalar(lhs: TtTensor, rhs: burn_backend::Scalar) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        device_eltwise(kind_sfpu::INT_SHR_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("bitwise_right_shift_scalar", context(&lhs)))
    }
    pub fn bitwise_not(tensor: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_NOT, 0.0, &tensor, None)
            .unwrap_or_else(|| fail("bitwise_not", context(&tensor)))
    }
    pub fn int_equal(lhs: TtTensor, rhs: TtTensor, out_dtype: burn_backend::BoolDType) -> TtTensor {
        let out = device_eltwise(kind_sfpu::INT_EQ, 0.0, &lhs, Some(&rhs))
            .unwrap_or_else(|| fail("int_equal", context(&lhs)));
        retyped(out, out_dtype.into())
    }
    pub fn int_equal_elem(
        lhs: TtTensor,
        rhs: burn_backend::Scalar,
        out_dtype: burn_backend::BoolDType,
    ) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        let out = device_eltwise(kind_sfpu::INT_EQ_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("int_equal_elem", context(&lhs)));
        retyped(out, out_dtype.into())
    }
    pub fn int_greater(
        lhs: TtTensor,
        rhs: TtTensor,
        out_dtype: burn_backend::BoolDType,
    ) -> TtTensor {
        let out = device_eltwise(kind_sfpu::INT_GT, 0.0, &lhs, Some(&rhs))
            .unwrap_or_else(|| fail("int_greater", context(&lhs)));
        retyped(out, out_dtype.into())
    }
    pub fn int_greater_elem(
        lhs: TtTensor,
        rhs: burn_backend::Scalar,
        out_dtype: burn_backend::BoolDType,
    ) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        let out = device_eltwise(kind_sfpu::INT_GT_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("int_greater_elem", context(&lhs)));
        retyped(out, out_dtype.into())
    }
    pub fn int_greater_equal(
        lhs: TtTensor,
        rhs: TtTensor,
        out_dtype: burn_backend::BoolDType,
    ) -> TtTensor {
        let out = device_eltwise(kind_sfpu::INT_GE, 0.0, &lhs, Some(&rhs))
            .unwrap_or_else(|| fail("int_greater_equal", context(&lhs)));
        retyped(out, out_dtype.into())
    }
    pub fn int_greater_equal_elem(
        lhs: TtTensor,
        rhs: burn_backend::Scalar,
        out_dtype: burn_backend::BoolDType,
    ) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        let out = device_eltwise(kind_sfpu::INT_GE_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("int_greater_equal_elem", context(&lhs)));
        retyped(out, out_dtype.into())
    }
    pub fn int_lower(lhs: TtTensor, rhs: TtTensor, out_dtype: burn_backend::BoolDType) -> TtTensor {
        let out = device_eltwise(kind_sfpu::INT_LT, 0.0, &lhs, Some(&rhs))
            .unwrap_or_else(|| fail("int_lower", context(&lhs)));
        retyped(out, out_dtype.into())
    }
    pub fn int_lower_elem(
        lhs: TtTensor,
        rhs: burn_backend::Scalar,
        out_dtype: burn_backend::BoolDType,
    ) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        let out = device_eltwise(kind_sfpu::INT_LT_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("int_lower_elem", context(&lhs)));
        retyped(out, out_dtype.into())
    }
    pub fn int_lower_equal(
        lhs: TtTensor,
        rhs: TtTensor,
        out_dtype: burn_backend::BoolDType,
    ) -> TtTensor {
        let out = device_eltwise(kind_sfpu::INT_LE, 0.0, &lhs, Some(&rhs))
            .unwrap_or_else(|| fail("int_lower_equal", context(&lhs)));
        retyped(out, out_dtype.into())
    }
    pub fn int_lower_equal_elem(
        lhs: TtTensor,
        rhs: burn_backend::Scalar,
        out_dtype: burn_backend::BoolDType,
    ) -> TtTensor {
        let bits = rhs.to_i64().expect("integer scalar") as i32 as u32;
        let out = device_eltwise(kind_sfpu::INT_LE_S, f32::from_bits(bits), &lhs, None)
            .unwrap_or_else(|| fail("int_lower_equal_elem", context(&lhs)));
        retyped(out, out_dtype.into())
    }
    fn reduce_integer(
        tensor: TtTensor,
        dim: Option<usize>,
        op: tt_kernels::sfpu::reduce::ReduceOp,
    ) -> TtTensor {
        if let Some(dim) = dim {
            return float::device_reduce_ungated(&tensor, op, dim)
                .unwrap_or_else(|| fail("integer reduction", context(&tensor)));
        }
        let count = tensor.shape().num_elements();
        let packed = repack_shape(&tensor, vec![1, count]);
        let out = float::device_reduce_ungated(&packed, op, 1)
            .unwrap_or_else(|| fail("integer reduction", context(&tensor)));
        repack_shape(&out, vec![1])
    }
    pub fn int_sum(tensor: TtTensor) -> TtTensor {
        reduce_integer(tensor, None, tt_kernels::sfpu::reduce::ReduceOp::SumI32)
    }
    pub fn int_sum_dim(tensor: TtTensor, dim: usize) -> TtTensor {
        reduce_integer(
            tensor,
            Some(dim),
            tt_kernels::sfpu::reduce::ReduceOp::SumI32,
        )
    }
    pub fn int_mean_dim(tensor: TtTensor, dim: usize) -> TtTensor {
        let count = i32::try_from(tensor.shape()[dim]).expect("integer mean count fits I32");
        assert!(count > 0, "integer mean needs a nonempty axis");
        int_div_scalar(int_sum_dim(tensor, dim), count.into())
    }
    pub fn int_prod(tensor: TtTensor) -> TtTensor {
        reduce_integer(tensor, None, tt_kernels::sfpu::reduce::ReduceOp::ProdI32)
    }
    pub fn int_prod_dim(tensor: TtTensor, dim: usize) -> TtTensor {
        reduce_integer(
            tensor,
            Some(dim),
            tt_kernels::sfpu::reduce::ReduceOp::ProdI32,
        )
    }
    pub fn int_min_dim(tensor: TtTensor, dim: usize) -> TtTensor {
        reduce_integer(
            tensor,
            Some(dim),
            tt_kernels::sfpu::reduce::ReduceOp::MinI32,
        )
    }
    pub fn int_max_dim(tensor: TtTensor, dim: usize) -> TtTensor {
        reduce_integer(
            tensor,
            Some(dim),
            tt_kernels::sfpu::reduce::ReduceOp::MaxI32,
        )
    }
    pub fn int_from_data(data: TensorData, device: &TtDevice) -> TtTensor {
        if data.dtype != DType::I32 {
            fail("int_from_data", context(&data));
        }
        TtTensor::new(HostBuffer::from_data(data), *device)
    }

    pub fn int_empty(shape: burn_backend::Shape, device: &TtDevice, dtype: IntDType) -> TtTensor {
        if DType::from(dtype) != DType::I32 {
            fail(
                "int_empty",
                format_args!("shape={shape:?}, dtype={dtype:?}"),
            );
        }
        native_zeros(shape.clone(), device, dtype.into())
            .unwrap_or_else(|| int_from_data(TensorData::zeros::<i32, _>(shape), device))
    }

    pub fn int_random(
        shape: burn_backend::Shape,
        distribution: burn_backend::Distribution,
        device: &TtDevice,
        dtype: IntDType,
    ) -> TtTensor {
        if DType::from(dtype) != DType::I32 {
            fail(
                "int_random",
                format_args!("shape={shape:?}, dtype={dtype:?}"),
            );
        }
        int_from_data(crate::random::int(*device, shape, distribution), device)
    }

    pub fn int_device(tensor: &IntTensor<TtBackend>) -> Device<TtBackend> {
        tensor.device
    }

    /// On `device`, and an `I32` tensor resident there (D3).
    pub fn int_to_device(
        tensor: IntTensor<TtBackend>,
        device: &Device<TtBackend>,
    ) -> IntTensor<TtBackend> {
        to_device_resident(tensor, device)
    }

    /// `as f32` on the device where the data is (`kind_sfpu::I32_TO_F32`,
    /// exact); other float dtypes are unsupported.
    pub fn int_into_float(
        tensor: IntTensor<TtBackend>,
        out_dtype: burn_backend::FloatDType,
    ) -> FloatTensor<TtBackend> {
        if matches!(DType::from(out_dtype), DType::F32 | DType::BF16) {
            if let Some(t) = device_eltwise(kind_sfpu::I32_TO_F32, 0.0, &tensor, None) {
                return cast_native(t, DType::from(out_dtype));
            }
        }
        fail(
            "int_into_float",
            format_args!("tensor=({})", context(&tensor)),
        )
    }

    /// Broadcast I32 bytes with the native copier.
    pub fn int_expand(tensor: TtTensor, shape: burn_backend::Shape) -> TtTensor {
        expanded(&tensor, shape, "int_expand")
    }

    /// A view where the stored matrix is kept, as `float_reshape`.
    pub fn int_reshape(
        tensor: IntTensor<TtBackend>,
        shape: burn_backend::Shape,
    ) -> IntTensor<TtBackend> {
        reshaped(tensor, shape)
    }

    /// Whole tile rows of a device matrix, a view; as `float_slice`.
    pub fn int_slice_assign(
        tensor: TtTensor,
        slices: &[burn_backend::Slice],
        value: TtTensor,
    ) -> TtTensor {
        slice_assign_native(tensor, slices, value)
    }

    pub fn int_slice(
        tensor: IntTensor<TtBackend>,
        slices: &[burn_backend::Slice],
    ) -> IntTensor<TtBackend> {
        if let Some(v) = super::float::row_view(&tensor, slices) {
            return v;
        }
        if let Some(t) = sliced_native(&tensor, slices) {
            return t;
        }
        let device = tensor.device;
        {
            if tensor.computed_on_device() {
                fail("int_slice", context(&tensor));
            }
            TtTensor::new(tensor.into_host().slice(slices), device)
        }
    }

    /// A view of a device matrix transposed; as `float_swap_dims`.
    pub fn int_swap_dims(
        tensor: IntTensor<TtBackend>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<TtBackend> {
        if let Some(v) = super::float::swapped_strided(&tensor, dim1, dim2) {
            return v;
        }
        if let Some(v) = swapped_view(&tensor, dim1, dim2) {
            return v;
        }
        let device = tensor.device;
        {
            if tensor.computed_on_device() {
                fail("int_swap_dims", context(&tensor));
            }
            TtTensor::new(tensor.into_host().swap_dims(dim1, dim2), device)
        }
    }

    pub fn int_transpose(tensor: IntTensor<TtBackend>) -> IntTensor<TtBackend> {
        let n = tensor.shape().num_dims();
        if n < 2 {
            return tensor;
        }
        int_swap_dims(tensor, n - 2, n - 1)
    }

    pub fn int_into_data(
        tensor: IntTensor<TtBackend>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        async move { Ok(tensor.into_host().into_data()) }
    }
}

pub mod bool {
    use super::*;

    pub fn bool_scatter_or(
        dim: usize,
        tensor: TtTensor,
        indices: TtTensor,
        value: TtTensor,
    ) -> TtTensor {
        scatter_add_native(dim, tensor, indices, value)
    }
    pub fn bool_select_or(
        tensor: TtTensor,
        dim: usize,
        indices: TtTensor,
        value: TtTensor,
    ) -> TtTensor {
        select_add_native(tensor, dim, indices, value)
    }
    pub fn bool_gather(dim: usize, tensor: TtTensor, indices: TtTensor) -> TtTensor {
        gather_native(dim, tensor, indices)
    }
    pub fn bool_select(tensor: TtTensor, dim: usize, indices: TtTensor) -> TtTensor {
        select_native(tensor, dim, indices)
    }

    pub fn bool_from_data(data: TensorData, device: &TtDevice) -> TtTensor {
        if !matches!(data.dtype, DType::Bool(_)) {
            fail("bool_from_data", context(&data));
        }
        TtTensor::new(HostBuffer::from_data(data), *device)
    }

    pub fn bool_empty(
        shape: burn_backend::Shape,
        device: &TtDevice,
        dtype: burn_backend::BoolDType,
    ) -> TtTensor {
        bool_zeros(shape, device, dtype)
    }

    pub fn bool_zeros(
        shape: burn_backend::Shape,
        device: &TtDevice,
        dtype: burn_backend::BoolDType,
    ) -> TtTensor {
        if let Some(out) = native_zeros(shape.clone(), device, dtype.into()) {
            return out;
        }
        bool_from_data(
            TensorData::zeros::<bool, _>(shape).convert_dtype(dtype.into()),
            device,
        )
    }

    pub fn bool_ones(
        shape: burn_backend::Shape,
        device: &TtDevice,
        dtype: burn_backend::BoolDType,
    ) -> TtTensor {
        bool_from_data(
            TensorData::ones::<bool, _>(shape).convert_dtype(dtype.into()),
            device,
        )
    }

    pub fn bool_device(tensor: &BoolTensor<TtBackend>) -> Device<TtBackend> {
        tensor.device
    }

    /// On `device`, and resident there as `0`/`1` (D3).
    pub fn bool_to_device(
        tensor: BoolTensor<TtBackend>,
        device: &Device<TtBackend>,
    ) -> BoolTensor<TtBackend> {
        to_device_resident(tensor, device)
    }

    /// Broadcast Boolean bytes with the native copier.
    pub fn bool_expand(tensor: TtTensor, shape: burn_backend::Shape) -> TtTensor {
        expanded(&tensor, shape, "bool_expand")
    }

    /// Convert device Boolean 0/1 to F32 without reading back.
    pub fn bool_into_float(tensor: TtTensor, out_dtype: burn_backend::FloatDType) -> TtTensor {
        if matches!(
            out_dtype,
            burn_backend::FloatDType::F32 | burn_backend::FloatDType::BF16
        ) {
            if let Some(t) = device_eltwise(kind_sfpu::BOOL_TO_F32, 0.0, &tensor, None) {
                return cast_native(t, DType::from(out_dtype));
            }
        }
        fail(
            "bool_into_float",
            format_args!("{}, out_dtype={out_dtype:?}", context(&tensor)),
        )
    }

    /// Copy Boolean 0/1 to a typed I32 device buffer.
    pub fn bool_into_int(tensor: TtTensor, out_dtype: IntDType) -> TtTensor {
        if out_dtype == IntDType::I32 {
            if let Some(t) = device_eltwise(kind_sfpu::BOOL_TO_I32, 0.0, &tensor, None) {
                return t;
            }
        }
        fail(
            "bool_into_int",
            format_args!("{}, out_dtype={out_dtype:?}", context(&tensor)),
        )
    }

    /// Boolean reductions preserve raw canonical 0/1 storage.
    pub fn bool_any(tensor: TtTensor) -> TtTensor {
        float::reduce_all(tensor, tt_kernels::sfpu::reduce::ReduceOp::Any, "bool_any")
    }

    pub fn bool_all(tensor: TtTensor) -> TtTensor {
        float::reduce_all(tensor, tt_kernels::sfpu::reduce::ReduceOp::All, "bool_all")
    }

    pub fn bool_any_dim(tensor: TtTensor, dim: usize) -> TtTensor {
        check_bool_reduce_dim(&tensor, dim, "bool_any_dim");
        float::device_reduce(&tensor, tt_kernels::sfpu::reduce::ReduceOp::Any, dim)
            .unwrap_or_else(|| fail("bool_any_dim", context(&tensor)))
    }

    pub fn bool_all_dim(tensor: TtTensor, dim: usize) -> TtTensor {
        check_bool_reduce_dim(&tensor, dim, "bool_all_dim");
        float::device_reduce(&tensor, tt_kernels::sfpu::reduce::ReduceOp::All, dim)
            .unwrap_or_else(|| fail("bool_all_dim", context(&tensor)))
    }

    fn check_bool_reduce_dim(tensor: &TtTensor, dim: usize, op: &str) {
        if dim >= tensor.shape().num_dims() || tensor.elem() != Some(Elem::Bool) {
            fail(op, format_args!("{}, axis={dim}", context(tensor)));
        }
    }

    /// Boolean equality, including the existing native broadcasts.
    pub fn bool_equal(lhs: TtTensor, rhs: TtTensor) -> TtTensor {
        if let Some(t) = device_eltwise(kind_sfpu::BOOL_XOR, 0.0, &lhs, Some(&rhs)) {
            return bool_not(t);
        }
        fail(
            "bool_equal",
            format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
        )
    }

    /// Equality with true is an identity; equality with false is native NOT.
    pub fn bool_equal_elem(lhs: TtTensor, rhs: burn_backend::Scalar) -> TtTensor {
        if lhs.elem() != Some(Elem::Bool)
            || !lhs.is_storable()
            || !crate::server::supports_dram(lhs.device)
        {
            fail(
                "bool_equal_elem",
                format_args!("{}, rhs={rhs:?}", context(&lhs)),
            );
        }
        if rhs.elem::<bool>() {
            let _ = lhs.to_dram();
            lhs
        } else {
            bool_not(lhs)
        }
    }

    pub fn bool_reshape(
        tensor: BoolTensor<TtBackend>,
        shape: burn_backend::Shape,
    ) -> BoolTensor<TtBackend> {
        reshaped(tensor, shape)
    }

    pub fn bool_slice_assign(
        tensor: TtTensor,
        slices: &[burn_backend::Slice],
        value: TtTensor,
    ) -> TtTensor {
        slice_assign_native(tensor, slices, value)
    }

    pub fn bool_slice(
        tensor: BoolTensor<TtBackend>,
        slices: &[burn_backend::Slice],
    ) -> BoolTensor<TtBackend> {
        if let Some(v) = super::float::row_view(&tensor, slices) {
            return v;
        }
        if let Some(t) = sliced_native(&tensor, slices) {
            return t;
        }
        let device = tensor.device;
        {
            if tensor.computed_on_device() {
                fail("bool_slice", context(&tensor));
            }
            TtTensor::new(tensor.into_host().slice(slices), device)
        }
    }

    pub fn bool_swap_dims(
        tensor: BoolTensor<TtBackend>,
        dim1: usize,
        dim2: usize,
    ) -> BoolTensor<TtBackend> {
        if let Some(v) = super::float::swapped_strided(&tensor, dim1, dim2) {
            return v;
        }
        if let Some(v) = swapped_view(&tensor, dim1, dim2) {
            return v;
        }
        let device = tensor.device;
        {
            if tensor.computed_on_device() {
                fail("bool_swap_dims", context(&tensor));
            }
            TtTensor::new(tensor.into_host().swap_dims(dim1, dim2), device)
        }
    }

    pub fn bool_transpose(tensor: BoolTensor<TtBackend>) -> BoolTensor<TtBackend> {
        let n = tensor.shape().num_dims();
        if n < 2 {
            return tensor;
        }
        bool_swap_dims(tensor, n - 2, n - 1)
    }

    /// `!x` on the device where the tensor is (`kind_sfpu::BOOL_NOT`).
    pub fn bool_not(tensor: BoolTensor<TtBackend>) -> BoolTensor<TtBackend> {
        if let Some(t) = device_eltwise(kind_sfpu::BOOL_NOT, 0.0, &tensor, None) {
            return t;
        }
        fail("bool_not", format_args!("tensor=({})", context(&tensor)))
    }

    macro_rules! logic {
        ($name:ident, $kind:expr) => {
            /// On the device where the operands are, with a row or column
            /// broadcast; unsupported inputs fail.
            pub fn $name(
                lhs: BoolTensor<TtBackend>,
                rhs: BoolTensor<TtBackend>,
            ) -> BoolTensor<TtBackend> {
                if let Some(t) = device_eltwise($kind, 0.0, &lhs, Some(&rhs)) {
                    return t;
                }
                fail(
                    stringify!($name),
                    format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
                )
            }
        };
    }
    logic!(bool_and, kind_sfpu::BOOL_AND);
    logic!(bool_or, kind_sfpu::BOOL_OR);
    logic!(bool_xor, kind_sfpu::BOOL_XOR);

    pub fn bool_into_data(
        tensor: BoolTensor<TtBackend>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        async move { Ok(tensor.into_host().into_data()) }
    }

    pub fn bool_argwhere(
        tensor: BoolTensor<TtBackend>,
        out_dtype: IntDType,
    ) -> impl Future<Output = IntTensor<TtBackend>> + 'static + Send {
        async move {
            fail(
                "bool_argwhere",
                format_args!("{}, out_dtype={out_dtype:?}", context(&tensor)),
            )
        }
    }
}

pub mod quantized {
    use super::*;

    pub fn q_device(tensor: &QuantizedTensor<TtBackend>) -> Device<TtBackend> {
        tensor.device
    }

    pub fn q_to_device(
        tensor: QuantizedTensor<TtBackend>,
        device: &Device<TtBackend>,
    ) -> QuantizedTensor<TtBackend> {
        TtQTensor {
            shape: tensor.shape,
            scheme: tensor.scheme,
            device: *device,
        }
    }

    pub fn q_into_data(
        tensor: QuantizedTensor<TtBackend>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        async move {
            Err(ExecutionError::WithContext {
                reason: format!("unsupported operation q_into_data: {}", context(&tensor)),
            })
        }
    }
}

pub mod transaction {
    use super::*;

    pub fn tr_execute(
        transaction: TransactionPrimitive<TtBackend>,
    ) -> impl Future<Output = Result<TransactionPrimitiveData, ExecutionError>> + Send {
        async move {
            let mut data = TransactionPrimitiveData::default();
            for t in transaction.read_floats {
                data.read_floats.push(float::float_into_data(t).await?);
            }
            for t in transaction.read_ints {
                data.read_ints.push(int::int_into_data(t).await?);
            }
            for t in transaction.read_bools {
                data.read_bools.push(bool::bool_into_data(t).await?);
            }
            for t in transaction.read_qfloats {
                data.read_qfloats.push(quantized::q_into_data(t).await?);
            }
            Ok(data)
        }
    }
}
