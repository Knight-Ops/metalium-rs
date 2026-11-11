//! Ops whose result's dtype is named, each one op on the card (roadmap W1, native BF16
//! arithmetic; the performance contract's BF16 activations, ADR 0006): a model keeps its residual
//! stream and its arithmetic in F32 and hands its products BF16 `A` operands, rounded once where
//! they are made, the way Tenstorrent's own stack computes in FP32 in the registers and narrows
//! only at the pack.
//!
//! - [`rms_norm_as`]: `RmsNorm` with its scale, its F32 result rounded once to the named dtype --
//!   bit for bit `burn_tt::rms_norm(x, gamma).cast(dtype)`.
//! - [`silu_mul_as`]: `silu(gate) * up`, rounded once -- bit for bit the F32 composition then the
//!   cast.
//! - [`matmul_as`]: `x @ w` for a BF16 `x` and a logical F32 weight stored packed, an F32 result
//!   -- bit for bit `x.cast(F32) @ w` (Burn's own `matmul` gives the left operand's dtype, so a
//!   BF16 `x` would give a rounded BF16 product).
//!
//! Each is exact against its composition, so a schedule's choice (ADR 0005), not a numeric one:
//! the rounding is the one the composition's cast makes, done in `Dst` before the pack. `None` on
//! a backend other than `TtBackend` and `Tt`: the caller runs the composition.

use burn_backend::{DType, TensorMetadata};

use super::*;
use crate::TtTensor;

/// [`silu_mul_as`] on `TtBackend`: one SFPU pass (`SILU_MUL`) whose F32 result is rounded once to
/// `out` before its pack, where the F32 op runs as one SFPU pass on plain F32 operands of one
/// shape; else the F32 op (or its composition) then `cast_native`.
pub(crate) fn silu_mul_tt_as(x: TtTensor, y: TtTensor, out: DType) -> TtTensor {
    use tt_kernels::sfpu::ops::kind_sfpu;
    let _op = crate::report::enter("tt_silu_mul", true);
    let plain = |t: &TtTensor| {
        t.dtype() == DType::F32 && t.storage_format() == crate::storage::StorageFormat::F32
    };
    let native = (out != DType::F32
        && plain(&x)
        && plain(&y)
        && x.shape() == y.shape()
        && !matches!(
            crate::server::elementwise_mode(x.device),
            crate::ElementwiseMode::Matrix { .. }
        ))
    .then(|| {
        let op = tt_kernels::tensor::Eltwise {
            kind: kind_sfpu::SILU_MUL,
            scalar: 0.0,
            scalar2: 0.0,
        };
        device_op_as(op, &x, Some(&y), None, Some(out))
    })
    .flatten();
    let made = native.unwrap_or_else(|| {
        let wide = float::float_silu_mul(&x, Some(&y)).unwrap_or_else(|| {
            use burn_backend::ops::{ActivationOps, FloatTensorOps};
            let s = TtBackend::sigmoid(x.clone());
            let t = TtBackend::float_mul(x, s);
            TtBackend::float_mul(t, y)
        });
        if wide.dtype() == out {
            wide
        } else {
            cast_native(wide, out)
        }
    });
    crate::report::made(true);
    made
}

/// [`matmul_as`] on `TtBackend`: `x @ w`, `x` `[.., m, k]` against `w` `[k, n]`, its result
/// `out`. A BF16 `x` against a logical F32 weight stored packed, with an F32 result, is one
/// product reading `x` as it lies ([`packed_weight_matmul_as`]); anything else is `x` cast to
/// `w`'s dtype, Burn's product, and that cast to `out`.
pub(crate) fn matmul_tt_as(x: TtTensor, w: TtTensor, out: DType) -> TtTensor {
    use burn_backend::ops::FloatTensorOps;
    let _op = crate::report::enter("float_matmul", true);
    let made = packed_weight_matmul_as(&x, &w, out).unwrap_or_else(|| {
        let rank = x.shape().num_dims();
        let x = if x.dtype() == w.dtype() {
            x
        } else {
            cast_native(x, w.dtype())
        };
        let mut at = vec![1; rank.saturating_sub(2)];
        at.extend_from_slice(&w.shape().to_vec());
        let w = TtBackend::float_reshape(w, at.into());
        let product = TtBackend::float_matmul(x, w);
        if product.dtype() == out {
            product
        } else {
            cast_native(product, out)
        }
    });
    crate::report::made(true);
    made
}

/// What a [`fused::RoundedOp`] computes.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// `rms_norm(x, gamma)` with this `eps`.
    Norm(f32),
    SiluMul,
    Matmul,
}

fn run(kind: Kind, a: TtTensor, b: TtTensor, out: DType) -> TtTensor {
    match kind {
        Kind::Norm(eps) => norm::rms_norm_tt_as(a, b, eps, out),
        Kind::SiluMul => silu_mul_tt_as(a, b, out),
        Kind::Matmul => matmul_tt_as(a, b, out),
    }
}

/// The custom operation's id on `Tt`: the kind, `eps`'s bits and the result's dtype ride in it
/// (a custom operation's IR holds only tensors).
#[cfg(feature = "fusion")]
const ID: &str = "tt_rounded";

#[cfg(feature = "fusion")]
pub(crate) mod fused {
    use burn_backend::DType;
    use burn_fusion::stream::{Operation, OperationStreams};
    use burn_fusion::FusionTensor;
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use super::Kind;
    use crate::fusion::TtFusionRuntime;
    use crate::{TtBackend, TtHandle};

    #[derive(Debug)]
    struct RoundedOp {
        desc: CustomOpIr,
        kind: Kind,
        out: DType,
    }

    impl Operation<TtFusionRuntime> for RoundedOp {
        fn execute(&self, handles: &mut HandleContainer<TtHandle>) {
            let ([a, b], [out]) = self.desc.as_fixed::<2, 1>();
            let a = handles.get_float_tensor::<TtBackend>(a);
            let b = handles.get_float_tensor::<TtBackend>(b);
            handles
                .register_float_tensor::<TtBackend>(&out.id, super::run(self.kind, a, b, self.out));
        }
    }

    /// The op in `Tt`'s fusion stream as one custom operation of result `shape` and `out`, run in
    /// order with the rest.
    pub(super) fn register(
        kind: Kind,
        a: FusionTensor<TtFusionRuntime>,
        b: FusionTensor<TtFusionRuntime>,
        shape: burn_backend::Shape,
        out: DType,
    ) -> FusionTensor<TtFusionRuntime> {
        let client = a.client.clone();
        let streams = OperationStreams::with_inputs([&a, &b]);
        let result = TensorIr::uninit(client.create_empty_handle(), shape, out);
        let mut desc = CustomOpIr::new(super::ID, &[a.into_ir(), b.into_ir()], &[result]);
        let eps = match kind {
            Kind::Norm(eps) => eps.to_bits(),
            _ => 0,
        };
        desc.id = format!("{}:{kind:?}:{eps:08x}:{out:?}", super::ID);
        client
            .register(
                streams,
                OperationIr::Custom(desc.clone()),
                RoundedOp { desc, kind, out },
            )
            .into_iter()
            .next()
            .expect("one output")
    }
}

/// `kind` on `a` and `b` as one op on `TtBackend` or `Tt`, its result `shape` and `out`; `None`
/// on any other backend.
fn dispatch<B: burn_backend::Backend, const D: usize, const E: usize>(
    kind: Kind,
    a: &burn_tensor::Tensor<B, D>,
    b: &burn_tensor::Tensor<B, E>,
    shape: burn_backend::Shape,
    out: DType,
) -> Option<burn_tensor::Tensor<B, D>> {
    use burn_tensor::{Tensor, TensorPrimitive};
    use std::any::{Any, TypeId};
    fn cast<F: 'static, T: 'static>(x: F) -> Option<T> {
        (Box::new(x) as Box<dyn Any>)
            .downcast::<T>()
            .ok()
            .map(|b| *b)
    }
    fn float<X: burn_backend::Backend>(t: TensorPrimitive<X>) -> Option<X::FloatTensorPrimitive> {
        match t {
            TensorPrimitive::Float(t) => Some(t),
            TensorPrimitive::QFloat(_) => None,
        }
    }
    if TypeId::of::<B>() == TypeId::of::<crate::TtBackend>() {
        let a: Tensor<crate::TtBackend, D> = cast(a.clone())?;
        let b: Tensor<crate::TtBackend, E> = cast(b.clone())?;
        let y = run(
            kind,
            float(a.into_primitive())?,
            float(b.into_primitive())?,
            out,
        );
        return cast(Tensor::<crate::TtBackend, D>::from_primitive(
            TensorPrimitive::Float(y),
        ));
    }
    #[cfg(feature = "fusion")]
    if TypeId::of::<B>() == TypeId::of::<crate::Tt>() {
        let a: Tensor<crate::Tt, D> = cast(a.clone())?;
        let b: Tensor<crate::Tt, E> = cast(b.clone())?;
        let y = fused::register(
            kind,
            float(a.into_primitive())?,
            float(b.into_primitive())?,
            shape,
            out,
        );
        return cast(Tensor::<crate::Tt, D>::from_primitive(
            TensorPrimitive::Float(y),
        ));
    }
    let _ = shape;
    None
}

/// Burn's `RmsNorm` with its scale over the last dimension (`burn_tt::rms_norm`), its F32 result
/// rounded once to `dtype`: bit for bit `rms_norm(x, gamma, eps).cast(dtype)`, as one op on
/// `TtBackend` and `Tt` (a decode row's one op; more rows the statistic, the division and the
/// scale, the last rounding). For F32 `x` and `gamma`; `None` for others, and on any other backend.
pub fn rms_norm_as<B: burn_backend::Backend, const D: usize>(
    x: &burn_tensor::Tensor<B, D>,
    gamma: &burn_tensor::Tensor<B, 1>,
    eps: f64,
    dtype: DType,
) -> Option<burn_tensor::Tensor<B, D>> {
    if (x.dtype(), gamma.dtype()) != (DType::F32, DType::F32) {
        return None;
    }
    let shape = x.shape();
    dispatch(Kind::Norm(eps as f32), x, gamma, shape, dtype)
}

/// `silu(gate) * up` on F32 operands of one shape, its F32 result rounded once to `dtype`: bit
/// for bit `(silu(gate) * up).cast(dtype)`, as one op on `TtBackend` and `Tt`. `None` for other
/// operands, and on any other backend.
pub fn silu_mul_as<B: burn_backend::Backend, const D: usize>(
    gate: &burn_tensor::Tensor<B, D>,
    up: &burn_tensor::Tensor<B, D>,
    dtype: DType,
) -> Option<burn_tensor::Tensor<B, D>> {
    if (gate.dtype(), up.dtype()) != (DType::F32, DType::F32) || gate.shape() != up.shape() {
        return None;
    }
    let shape = gate.shape();
    dispatch(Kind::SiluMul, gate, up, shape, dtype)
}

/// `x @ w` (`x` `[.., m, k]`, `w` `[k, n]`, a linear layer's weight) as the product of `x`'s values
/// in `w`'s dtype, its result `dtype`: bit for bit `x.cast(w.dtype()).matmul(w).cast(dtype)`. A
/// BF16 `x` against a logical F32 weight stored packed, with an F32 result, is one product that
/// reads `x` as it lies. `None` on any other backend, and for a `w` of another rank or `k`.
pub fn matmul_as<B: burn_backend::Backend, const D: usize>(
    x: &burn_tensor::Tensor<B, D>,
    w: &burn_tensor::Tensor<B, 2>,
    dtype: DType,
) -> Option<burn_tensor::Tensor<B, D>> {
    let mut shape = x.shape().to_vec();
    let [k, n] = w.dims();
    if D < 2 || shape[D - 1] != k {
        return None;
    }
    shape[D - 1] = n;
    dispatch(Kind::Matmul, x, w, shape.into(), dtype)
}
