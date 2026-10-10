//! Cumulative scans.
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
//! - `int_cummin`/`int_cummax`: signed order.

use super::*;
use tt_kernels::sfpu::scan::ScanOp;

pub fn float_cumsum(tensor: TtTensor, dim: usize) -> TtTensor {
    float::check_reduce_dim(&tensor, dim, "float_cumsum");
    scan_packed(tensor, dim, ScanOp::Sum, "float_cumsum", DType::F32)
}

pub fn float_cumprod(tensor: TtTensor, dim: usize) -> TtTensor {
    float::check_reduce_dim(&tensor, dim, "float_cumprod");
    scan_packed(tensor, dim, ScanOp::Prod, "float_cumprod", DType::F32)
}

pub fn float_cummin(tensor: TtTensor, dim: usize) -> TtTensor {
    scan_axis(tensor, dim, ScanOp::MinNan, "float_cummin")
}

pub fn float_cummax(tensor: TtTensor, dim: usize) -> TtTensor {
    scan_axis(tensor, dim, ScanOp::MaxNan, "float_cummax")
}

pub fn int_cumsum(tensor: TtTensor, dim: usize) -> TtTensor {
    scan_axis(tensor, dim, ScanOp::ISum, "int_cumsum")
}

pub fn int_cumprod(tensor: TtTensor, dim: usize) -> TtTensor {
    scan_axis(tensor, dim, ScanOp::IProd, "int_cumprod")
}

pub fn int_cummin(tensor: TtTensor, dim: usize) -> TtTensor {
    scan_axis(tensor, dim, ScanOp::IMin, "int_cummin")
}

pub fn int_cummax(tensor: TtTensor, dim: usize) -> TtTensor {
    scan_axis(tensor, dim, ScanOp::IMax, "int_cummax")
}

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
    scan_packed(tensor, dim, op, name, dtype)
}

/// The scan itself, after validation: `tensor`'s axis packed into matrix rows,
/// scanned on the device and put back, every step device-resident. `dtype` is
/// the scan's element type; `name` is the operation a failure names.
fn scan_packed(tensor: TtTensor, dim: usize, op: ScanOp, name: &str, dtype: DType) -> TtTensor {
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
