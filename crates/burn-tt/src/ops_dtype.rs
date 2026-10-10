//! Lane T7 FP16 and dtype casts: native implementations. Routed from `ops.rs`'s `float`, `int`
//! and `bool` modules by glob re-export, and listed in
//! `xtask/src/gen_burn/overrides/`.
//!
//! FP16 is IEEE binary16 in BF16's two-byte physical slots (`tt_kernels::fp16`).
//! In Burn it is a *storage and cast* dtype: raw views and movement work like
//! BF16's (the buffers are raw two-byte slots, `crate::tensor::is_half`), and
//! `float_cast` converts F32, BF16 and F16 on Tensix (BF16 to or from F16 goes
//! through F32). Arithmetic on F16 operands is not claimed: it fails explicitly
//! naming the operation and dtype, and `dtype_usage` reports `Storage` only.
//! F64 stays unsupported and its cast names the operation and dtype.

#![allow(unused_imports)]

use crate::ops::*;

/// Resident F32 <-> F16 conversion on Tensix, and BF16 <-> F16 through F32.
/// `None` for any pair involving neither F16 nor a supported partner dtype, so
/// the caller's explicit failure names the operation and the dtypes.
pub(super) fn cast_f16(tensor: &TtTensor, dtype: DType) -> Option<TtTensor> {
    let from = tensor.dtype();
    if (from != DType::F16 && dtype != DType::F16)
        || !matches!(from, DType::F32 | DType::BF16 | DType::F16)
        || !matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
        || !tensor.is_storable()
        || !crate::server::supports_dram(tensor.device)
    {
        return None;
    }
    match (from, dtype) {
        (DType::BF16, DType::F16) => {
            let wide = cast_native(tensor.clone(), DType::F32);
            cast_f16(&wide, DType::F16)
        }
        (DType::F16, DType::BF16) => {
            let wide = cast_f16(tensor, DType::F32)?;
            Some(cast_native(wide, DType::BF16))
        }
        _ => {
            let source = tensor.to_dram();
            let (id, dims) = crate::server::cast_f16(
                tensor.device,
                source.buffer.id,
                [source.buffer.rows, source.buffer.cols],
                dtype == DType::F16,
            );
            let mut result = device_view(tensor.device, id, dims, None, dtype)
                .to_dram()
                .clone();
            result.transposed = source.transposed;
            Some(TtTensor::on_device(
                result,
                tensor.shape(),
                dtype,
                tensor.device,
            ))
        }
    }
}

pub mod float {
    use crate::ops::*;
}

pub mod int {
    use crate::ops::*;
}

pub mod bool {
    use crate::ops::*;
}
