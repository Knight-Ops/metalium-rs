//! Lane T1 int and bool wiring: native implementations. Routed from `ops.rs`'s `float`, `int`
//! and `bool` modules by glob re-export, and listed in
//! `xtask/src/gen_burn/overrides/`.
//!
//! Views and copies (`*_permute`, `*_flip`, `*_unfold`) are the float paths' dtype-generic
//! bodies (`ops::permute_with`, `flip_native`, `unfold_native`): they move raw words through the
//! strided-view and repack machinery, so an `I32` or `Bool` tensor keeps every bit and its
//! element type. The selects and `int_abs` are raw-word SFPU programs
//! (`kind_sfpu::INT_MASK_WHERE` ... `INT_ABS`): no integer ever passes through an `f32`.

#![allow(unused_imports)]

use crate::ops::*;

/// Upload a host-resident tensor first, as `float_swap_dims` does, so a dimension swap is a view
/// of a device buffer rather than a host reshuffle.
fn resident_for_view(tensor: &TtTensor) {
    if tensor.is_storable()
        && crate::server::is_attached(tensor.device)
        && crate::server::supports_dram(tensor.device)
    {
        tensor.to_dram();
    }
}

/// `mask ? value : tensor` for the raw-word kinds: the three operands broadcast to one shape
/// (device copies, as `expand` does), then one ternary SFPU op.
fn mask_where_native(
    op: &str,
    kind: u32,
    tensor: TtTensor,
    mask: TtTensor,
    value: TtTensor,
) -> TtTensor {
    let fail_with = |why: &str| -> ! {
        fail(
            op,
            format_args!(
                "tensor=({}), mask=({}), value=({}): {why}",
                context(&tensor),
                context(&mask),
                context(&value)
            ),
        )
    };
    let shape = broadcast_shape(&tensor.shape().to_vec(), &mask.shape().to_vec())
        .and_then(|s| broadcast_shape(&s, &value.shape().to_vec()))
        .unwrap_or_else(|| fail_with("shapes do not broadcast"));
    let shape = burn_backend::Shape::from(shape);
    let (tensor, mask, value) = (
        expanded(&tensor, shape.clone(), op),
        expanded(&mask, shape.clone(), op),
        expanded(&value, shape, op),
    );
    let eltwise = tt_kernels::tensor::Eltwise {
        kind,
        scalar: 0.0,
        scalar2: 0.0,
    };
    device_op(eltwise, &tensor, Some(&mask), Some(&value))
        .unwrap_or_else(|| fail_with("operands are not resident-compatible"))
}

/// `mask ? scalar : tensor`, `scalar` the raw 32-bit pattern (`f32::from_bits`, never a float
/// conversion).
fn mask_fill_native(op: &str, kind: u32, bits: u32, tensor: TtTensor, mask: TtTensor) -> TtTensor {
    device_eltwise(kind, f32::from_bits(bits), &tensor, Some(&mask)).unwrap_or_else(|| {
        fail(
            op,
            format_args!("tensor=({}), mask=({})", context(&tensor), context(&mask)),
        )
    })
}

pub mod float {
    use crate::ops::*;
}

pub mod int {
    use super::{mask_fill_native, mask_where_native, resident_for_view};
    use crate::ops::*;
    use burn_backend::{DType, IntDType, Scalar};
    use num_traits::ToPrimitive;

    /// Reorder dimensions as strided views of the device buffer.
    pub fn int_permute(tensor: TtTensor, axes: &[usize]) -> TtTensor {
        permute_with(tensor, axes, "int_permute", |t, i, j| {
            resident_for_view(&t);
            crate::ops::int::int_swap_dims(t, i, j)
        })
    }

    /// Reverse selected axes through bit-preserving device copies.
    pub fn int_flip(tensor: TtTensor, axes: &[usize]) -> TtTensor {
        flip_native(tensor, axes, "int_flip")
    }

    /// Sliding windows of an `I32` tensor, every word kept.
    pub fn int_unfold(tensor: TtTensor, dim: usize, size: usize, step: usize) -> TtTensor {
        unfold_native(tensor, dim, size, step)
    }

    /// `mask ? value : tensor` as one raw-word predicated move: denormal and NaN bit patterns and
    /// `i32::MIN` survive.
    pub fn int_mask_where(tensor: TtTensor, mask: TtTensor, value: TtTensor) -> TtTensor {
        mask_where_native(
            "int_mask_where",
            kind_sfpu::INT_MASK_WHERE,
            tensor,
            mask,
            value,
        )
    }

    /// `mask ? value : tensor` with `value` wrapped to I32 as Flex's `as i32` does and carried
    /// as its raw bits: every I32 value, `16777217` and `i32::MIN` included, is filled exactly.
    pub fn int_mask_fill(tensor: TtTensor, mask: TtTensor, value: Scalar) -> TtTensor {
        let Some(v) = value.to_i64() else {
            fail(
                "int_mask_fill",
                format_args!(
                    "tensor=({}), mask=({}), value={value:?} is not an integer",
                    context(&tensor),
                    context(&mask)
                ),
            )
        };
        mask_fill_native(
            "int_mask_fill",
            kind_sfpu::INT_MASK_FILL,
            v as i32 as u32,
            tensor,
            mask,
        )
    }

    /// Wrapping absolute value in the integer ALU: `i32::MIN` stays `i32::MIN`, as Flex's
    /// `wrapping_abs`.
    pub fn int_abs(tensor: TtTensor) -> TtTensor {
        device_eltwise(kind_sfpu::INT_ABS, 0.0, &tensor, None)
            .unwrap_or_else(|| fail("int_abs", context(&tensor)))
    }

    /// I32 to I32 is the tensor itself (as Flex); every other width is not stored on the
    /// device and fails naming both dtypes.
    pub fn int_cast(tensor: TtTensor, dtype: IntDType) -> TtTensor {
        if tensor.dtype() == DType::I32 && DType::from(dtype) == DType::I32 {
            return tensor;
        }
        fail(
            "int_cast",
            format_args!(
                "{}, target dtype={:?}: the device stores I32 only",
                context(&tensor),
                DType::from(dtype)
            ),
        )
    }
}

pub mod bool {
    use super::{mask_fill_native, mask_where_native, resident_for_view};
    use crate::ops::*;
    use burn_backend::Scalar;

    /// Reorder dimensions as strided views of the device buffer.
    pub fn bool_permute(tensor: TtTensor, axes: &[usize]) -> TtTensor {
        permute_with(tensor, axes, "bool_permute", |t, i, j| {
            resident_for_view(&t);
            crate::ops::bool::bool_swap_dims(t, i, j)
        })
    }

    /// Reverse selected axes through bit-preserving device copies.
    pub fn bool_flip(tensor: TtTensor, axes: &[usize]) -> TtTensor {
        flip_native(tensor, axes, "bool_flip")
    }

    /// Sliding windows of a `Bool` tensor, canonical `0`/`1` kept.
    pub fn bool_unfold(tensor: TtTensor, dim: usize, size: usize, step: usize) -> TtTensor {
        unfold_native(tensor, dim, size, step)
    }

    /// `mask ? value : tensor` on canonical `0`/`1` words.
    pub fn bool_mask_where(tensor: TtTensor, mask: TtTensor, value: TtTensor) -> TtTensor {
        mask_where_native(
            "bool_mask_where",
            kind_sfpu::BOOL_MASK_WHERE,
            tensor,
            mask,
            value,
        )
    }

    /// `mask ? value : tensor`, `value` the canonical word `0` or `1`.
    pub fn bool_mask_fill(tensor: TtTensor, mask: TtTensor, value: Scalar) -> TtTensor {
        mask_fill_native(
            "bool_mask_fill",
            kind_sfpu::BOOL_MASK_FILL,
            u32::from(value.elem::<bool>()),
            tensor,
            mask,
        )
    }
}
