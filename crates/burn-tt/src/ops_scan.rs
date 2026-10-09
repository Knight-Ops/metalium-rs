//! Lane T3 scans and integer arg-extremes: native implementations. Routed from `ops.rs`'s `float`, `int`
//! and `bool` modules by glob re-export, and listed in
//! `xtask/src/gen_burn/overrides/`.
//!
//! Every scan runs `Session::scan` (`tt_kernels::sfpu::scan`) down the rows of
//! the matrix the scanned axis is packed into, as `float_cumsum` does; only
//! address metadata is built on the host. The Flex rules each method follows
//! (`burn-flex` `ops/cumulative.rs`):
//!
//! - `float_cummin`/`float_cummax`: `if val.is_nan() || val < acc { val } else
//!   { acc }` (`>` for the maximum) from `+inf`/`-inf`: a NaN replaces the
//!   accumulator, a NaN accumulator is replaced only by another NaN, and the
//!   earlier element is kept on equal values, `+0`/`-0` included
//!   ([`ScanOp::MinNan`], [`ScanOp::MaxNan`]);
//! - `int_cumsum`/`int_cumprod`: wrapping two's-complement arithmetic (the
//!   only defined behaviour; Flex's plain `+`/`*` wrap in release builds);
//! - `int_cummin`/`int_cummax`: signed order;
//! - `int_argmax`/`int_argmin`: the first index of the extreme.

#![allow(unused_imports)]

use tt_kernels::sfpu::scan::ScanOp;

pub mod float {
    use super::ScanOp;
    use crate::ops::*;

    pub fn float_cummin(tensor: TtTensor, dim: usize) -> TtTensor {
        super::scan_axis(tensor, dim, ScanOp::MinNan, "float_cummin")
    }

    pub fn float_cummax(tensor: TtTensor, dim: usize) -> TtTensor {
        super::scan_axis(tensor, dim, ScanOp::MaxNan, "float_cummax")
    }
}

pub mod int {
    use super::ScanOp;
    use crate::ops::*;

    pub fn int_cumsum(tensor: TtTensor, dim: usize) -> TtTensor {
        super::scan_axis(tensor, dim, ScanOp::ISum, "int_cumsum")
    }

    pub fn int_cumprod(tensor: TtTensor, dim: usize) -> TtTensor {
        super::scan_axis(tensor, dim, ScanOp::IProd, "int_cumprod")
    }

    pub fn int_cummin(tensor: TtTensor, dim: usize) -> TtTensor {
        super::scan_axis(tensor, dim, ScanOp::IMin, "int_cummin")
    }

    pub fn int_cummax(tensor: TtTensor, dim: usize) -> TtTensor {
        super::scan_axis(tensor, dim, ScanOp::IMax, "int_cummax")
    }

    /// First index of the maximum along `dim`.
    pub fn int_argmax(tensor: TtTensor, dim: usize) -> TtTensor {
        super::int_argextreme(tensor, dim, false)
    }

    /// First index of the minimum along `dim`.
    pub fn int_argmin(tensor: TtTensor, dim: usize) -> TtTensor {
        super::int_argextreme(tensor, dim, true)
    }
}

pub mod bool {
    use crate::ops::*;
}

use crate::ops::*;

/// `tensor` scanned along `dim`: its axis packed into matrix rows, scanned on
/// the device, and put back, every step device-resident. A tensor the
/// scan does not take (another dtype, a zero extent, an axis out of range, a
/// device without GDDR) fails naming the operation and the input.
pub(crate) fn scan_axis(tensor: TtTensor, dim: usize, op: ScanOp, name: &str) -> TtTensor {
    let dtype = match op.elem() {
        Elem::I32 => DType::I32,
        _ => DType::F32,
    };
    let rank = tensor.shape().num_dims();
    if tensor.dtype() != dtype
        || tensor.elem() != Some(op.elem())
        || !tensor.is_storable()
        || dim >= rank
        || !crate::server::supports_dram(tensor.device)
        || (dtype == DType::F32 && !tensor.is_stored_f32())
    {
        fail(name, format_args!("{}, dim={dim}", context(&tensor)));
    }
    let packed = pack_axis(&tensor, dim).unwrap_or_else(|| fail(name, context(&tensor)));
    let transposed = swapped_view(&packed, 0, 1).expect("packed matrix");
    let source = plain_dram(&transposed);
    let (id, dims) = crate::server::scan(tensor.device, source.buffer.id, op);
    let scanned = device_view(tensor.device, id, dims, None, dtype);
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
        dtype,
    )
}

/// First index of the signed extreme of an I32 tensor along `dim`. The
/// extreme comes from the exact integer reduction, an integer comparison
/// marks the other elements, and the first unmarked index is the minimum of
/// the negated positions (exact in F32 for an axis up to 2^23, the bound
/// `float_argmax` keeps). The comparison is on the integers themselves, never
/// on a float image of them (an I32 above 2^24 has no exact one). Index
/// metadata is the only host-built input.
pub(crate) fn int_argextreme(tensor: TtTensor, dim: usize, minimum: bool) -> TtTensor {
    let op = if minimum { "int_argmin" } else { "int_argmax" };
    let shape = tensor.shape().to_vec();
    if tensor.dtype() != DType::I32
        || shape.is_empty()
        || dim >= shape.len()
        || shape[dim] == 0
        || shape[dim] > 8_388_608
        || !tensor.is_storable()
        || !crate::server::supports_dram(tensor.device)
    {
        fail(op, format_args!("{}, dim={dim}", context(&tensor)));
    }
    if shape.len() > 2 {
        let packed = pack_axis(&tensor, dim).unwrap_or_else(|| fail(op, context(&tensor)));
        let selected = int_argextreme(packed, 1, minimum);
        let mut output = shape;
        output[dim] = 1;
        return repack_shape(&selected, output);
    }
    let axis = if shape.len() == 1 { 1 } else { dim };
    let dims = tensor.stored().expect("matrix");
    let tensor = reshaped(tensor, burn_backend::Shape::new(dims));
    let extreme = if minimum {
        crate::ops::int::int_min_dim(tensor.clone(), axis)
    } else {
        crate::ops::int::int_max_dim(tensor.clone(), axis)
    };
    let indices = (0..dims[0] * dims[1])
        .map(|i| {
            let index = if axis == 1 { i % dims[1] } else { i / dims[1] };
            (-(index as f32)).to_bits()
        })
        .collect();
    let id = crate::server::metadata(tensor.device, indices, dims, Elem::F32);
    let indices = device_result(tensor.device, id, dims);
    let unequal = device_eltwise(kind_sfpu::INT_NE, 0.0, &tensor, Some(&extreme))
        .unwrap_or_else(|| fail(op, context(&tensor)));
    let candidate = device_eltwise(
        kind_sfpu::MASK_FILL,
        f32::NEG_INFINITY,
        &indices,
        Some(&unequal),
    )
    .unwrap_or_else(|| fail(op, context(&tensor)));
    let best = crate::ops::float::float_max_dim(candidate, axis);
    let positive = crate::ops::float::float_neg(best);
    let index = device_eltwise(kind_sfpu::INDEX_TO_I32, 0.0, &positive, None)
        .unwrap_or_else(|| fail(op, context(&tensor)));
    let mut output = shape;
    output[dim] = 1;
    reshaped(index, burn_backend::Shape::from(output))
}
