//! RMSNorm with its scale as one op (`Session::rms_norm` with `gamma`), for a model to call where
//! burn-fusion's blocks keep a fuser from seeing the scale: Burn's `RmsNorm::forward` multiplies
//! by `gamma.unsqueeze()`, a `Reshape` of a tensor the norm's chain never read, which burn-fusion
//! puts in a block of its own, so on `Tt` the product by `gamma` was an op after the norm. One op
//! on a decode row here, bit for bit the composition `(x / sqrt(mean(x x) + eps)) * gamma`.

use burn_backend::DType;

use super::*;
use crate::TtTensor;

/// Residual sums run with their norm as one native call since the process started
/// ([`add_rms_norm_count`]).
static RESIDUAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many residual sums have run with the norm after them as one native call in this process
/// (the fuser's match on `Tt`). Dev-facing.
#[doc(hidden)]
pub fn add_rms_norm_count() -> u64 {
    RESIDUAL.load(std::sync::atomic::Ordering::SeqCst)
}

/// Norms scaled inside the norm's op since the process started ([`rms_norm_count`]).
static NATIVE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many RMSNorms have run with their scale as one native call in this process: what a gate
/// reads to see the op taken on `Tt`, whose fusion stream runs off the caller's thread.
/// Dev-facing.
#[doc(hidden)]
pub fn rms_norm_count() -> u64 {
    NATIVE.load(std::sync::atomic::Ordering::SeqCst)
}

/// `x / sqrt(mean(x x) + eps) * gamma` for a single row `x` (F32, `[.., 1, cols]`, `gamma`
/// `[cols]`) as one call (`Session::rms_norm` with `gamma`: one op on a decode row), where the
/// unscaled norm runs natively ([`float::float_rms_norm`]'s conditions) and `gamma` is a plain F32
/// row. `None` otherwise (more rows gain little: the caller composes the norm's op and the
/// product).
pub(crate) fn rms_norm_native(x: &TtTensor, gamma: &TtTensor, eps: f32) -> Option<TtTensor> {
    let shape = x.shape().to_vec();
    let cols = *shape.last()?;
    if gamma.dtype() != DType::F32
        || gamma.storage_format() != crate::storage::StorageFormat::F32
        || gamma.shape().num_elements() != cols
        || x.device != gamma.device
        || !gamma.is_storable()
        || !norm_takes(x, cols)
    {
        return None;
    }
    let (d, g) = (plain_dram(x), plain_dram(gamma));
    if [g.buffer.rows, g.buffer.cols] != [1, cols] {
        return None;
    }
    let (id, dims) = crate::server::rms_norm(x.device, d.buffer.id, Some(g.buffer.id), eps);
    NATIVE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    Some(device_result_shaped(
        x.device,
        id,
        dims,
        burn_backend::Shape::from(shape),
        DType::F32,
    ))
}

/// `s = a + b` and `s / sqrt(mean(s s) + eps)` (times `gamma`, `[cols]`, if given) for single rows
/// `a` and `b` of one shape (F32), `(s, norm)`, as one call (`Session::add_rms_norm`: one op where
/// both rows keep their packed rows, else the `ADD` and the norm), bit for bit `float_add` then
/// the norm. `None` where the norm alone would not run natively ([`rms_norm_native`]'s
/// conditions): the caller runs the operations.
pub(crate) fn add_rms_norm_native(
    a: &TtTensor,
    b: &TtTensor,
    gamma: Option<&TtTensor>,
    eps: f32,
) -> Option<(TtTensor, TtTensor)> {
    let shape = a.shape().to_vec();
    let cols = *shape.last()?;
    if !residual_on()
        || b.shape().to_vec() != shape
        || b.dtype() != DType::F32
        || b.storage_format() != crate::storage::StorageFormat::F32
        || b.device != a.device
        || !b.is_storable()
        || !norm_takes(a, cols)
    {
        return None;
    }
    let g = match gamma {
        Some(g) => {
            if g.dtype() != DType::F32
                || g.storage_format() != crate::storage::StorageFormat::F32
                || g.shape().num_elements() != cols
                || g.device != a.device
                || !g.is_storable()
            {
                return None;
            }
            let g = plain_dram(g);
            if [g.buffer.rows, g.buffer.cols] != [1, cols] {
                return None;
            }
            Some(g)
        }
        None => None,
    };
    let (da, db) = (plain_dram(a), plain_dram(b));
    let [(s, sd), (n, nd)] = crate::server::add_rms_norm(
        a.device,
        da.buffer.id,
        db.buffer.id,
        g.as_ref().map(|g| g.buffer.id),
        eps,
    );
    RESIDUAL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let shaped = |id, dims| {
        device_result_shaped(
            a.device,
            id,
            dims,
            burn_backend::Shape::from(shape.clone()),
            DType::F32,
        )
    };
    Some((shaped(s, sd), shaped(n, nd)))
}

/// Whether `x` is a single F32 row of `cols` the norm runs natively on.
fn norm_takes(x: &TtTensor, cols: usize) -> bool {
    let shape = x.shape().to_vec();
    x.dtype() == DType::F32
        && x.storage_format() == crate::storage::StorageFormat::F32
        && !shape.contains(&0)
        && shape.iter().product::<usize>() == cols
        && x.is_storable()
        && crate::server::supports_dram(x.device)
        && !matches!(
            crate::server::elementwise_mode(x.device),
            crate::ElementwiseMode::Matrix { .. }
        )
        && tt_kernels::sfpu::norm::fits(cols)
}

/// Burn's composition on this backend: the norm (one op or Burn's five), then `float_mul` by
/// `gamma` reshaped to `[1, .., 1, cols]`, as `RmsNorm::forward` makes it.
fn rms_norm_composed(x: TtTensor, gamma: TtTensor, eps: f32) -> TtTensor {
    use burn_backend::ops::FloatTensorOps;
    let rank = x.shape().num_dims();
    let normed = float::float_rms_norm(&x, eps).unwrap_or_else(|| {
        let squares = TtBackend::float_mul(x.clone(), x.clone());
        let mean = TtBackend::float_mean_dim(squares, rank - 1);
        let shifted = TtBackend::float_add_scalar(mean, (eps as f64).into());
        let rms = TtBackend::float_sqrt(shifted);
        TtBackend::float_div(x, rms)
    });
    let mut at = vec![1; rank];
    at[rank - 1] = gamma.shape().num_elements();
    let gamma = TtBackend::float_reshape(gamma, at.into());
    TtBackend::float_mul(normed, gamma)
}

/// [`rms_norm_native`], else [`rms_norm_composed`]: charged to an op of its own, `rms_norm`.
pub(crate) fn rms_norm_tt(x: TtTensor, gamma: TtTensor, eps: f32) -> TtTensor {
    let _op = crate::report::enter("rms_norm", true);
    let out = rms_norm_native(&x, &gamma, eps).unwrap_or_else(|| rms_norm_composed(x, gamma, eps));
    crate::report::made(true);
    out
}

/// Whether [`rms_norm`] takes the norm (`TT_RMS_NORM_SCALED=0` declines it, so a model runs its
/// module: the measurement's baseline). Read once.
fn scaled_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TT_RMS_NORM_SCALED").as_deref() != Ok("0"))
}

/// Whether [`add_rms_norm_native`] takes a residual sum and its norm: off by default
/// (`TT_ADD_RMS_NORM=1` or [`set_add_rms_norm`] turns it on). Bit for bit either way, but slower
/// on card 0 (2026-10-27): the 1.7B's step 11.18 -> 11.26 ms, the 135M's 6.86 -> 6.94, the merged
/// job's five kernels on the norm's units costing more than the barrier it saves.
static RESIDUAL_ON: std::sync::LazyLock<std::sync::atomic::AtomicBool> =
    std::sync::LazyLock::new(|| {
        std::sync::atomic::AtomicBool::new(std::env::var("TT_ADD_RMS_NORM").as_deref() == Ok("1"))
    });

fn residual_on() -> bool {
    RESIDUAL_ON.load(std::sync::atomic::Ordering::SeqCst)
}

/// Turn a residual sum and the norm after it as one op on or off, process-wide (off by default;
/// [`RESIDUAL_ON`]). For gates' and benches' comparisons. Dev-facing.
#[doc(hidden)]
pub fn set_add_rms_norm(on: bool) {
    RESIDUAL_ON.store(on, std::sync::atomic::Ordering::SeqCst);
}

/// The custom operation's id on `Tt`: `eps`'s bits ride in it, since a custom operation's IR
/// holds only tensors (and a fuser reading the stream finds them there).
#[cfg(feature = "fusion")]
pub(crate) const FUSED_ID: &str = "tt_rms_norm_scaled";

/// `eps` from a [`FUSED_ID`] custom operation's id (`tt_rms_norm_scaled:<bits>`), `None` for any
/// other id: what a fuser matching the operation binds.
#[cfg(feature = "fusion")]
pub(crate) fn fused_eps(id: &str) -> Option<f32> {
    let bits = id.strip_prefix(FUSED_ID)?.strip_prefix(':')?;
    u32::from_str_radix(bits, 16).ok().map(f32::from_bits)
}

#[cfg(feature = "fusion")]
pub(crate) mod fused {
    use burn_fusion::stream::{Operation, OperationStreams};
    use burn_fusion::FusionTensor;
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use crate::fusion::TtFusionRuntime;
    use crate::{TtBackend, TtHandle};

    #[derive(Debug)]
    struct RmsNormOp {
        desc: CustomOpIr,
        eps: f32,
    }

    impl Operation<TtFusionRuntime> for RmsNormOp {
        fn execute(&self, handles: &mut HandleContainer<TtHandle>) {
            let ([x, gamma], [out]) = self.desc.as_fixed::<2, 1>();
            let x = handles.get_float_tensor::<TtBackend>(x);
            let gamma = handles.get_float_tensor::<TtBackend>(gamma);
            handles.register_float_tensor::<TtBackend>(
                &out.id,
                super::rms_norm_tt(x, gamma, self.eps),
            );
        }
    }

    /// The scaled norm in `Tt`'s fusion stream as one custom operation, run in order with the
    /// rest.
    pub(crate) fn rms_norm(
        x: FusionTensor<TtFusionRuntime>,
        gamma: FusionTensor<TtFusionRuntime>,
        eps: f32,
    ) -> FusionTensor<TtFusionRuntime> {
        let client = x.client.clone();
        let streams = OperationStreams::with_inputs([&x, &gamma]);
        let out = TensorIr::uninit(client.create_empty_handle(), x.shape.clone(), x.dtype);
        let mut desc = CustomOpIr::new(super::FUSED_ID, &[x.into_ir(), gamma.into_ir()], &[out]);
        desc.id = format!("{}:{:08x}", super::FUSED_ID, eps.to_bits());
        client
            .register(
                streams,
                OperationIr::Custom(desc.clone()),
                RmsNormOp { desc, eps },
            )
            .into_iter()
            .next()
            .expect("one output")
    }
}

/// Burn's `RmsNorm` with its scale, `x / sqrt(mean(x x) + eps) * gamma` over the last dimension,
/// as one op on `TtBackend` and `Tt` (a decode row; elsewhere the norm and the product), bit for
/// bit `RmsNorm::forward` there. `None` on any other backend: the caller runs the module. The
/// manual fusion of an op every pre-norm model runs, where burn-fusion's blocks keep a fuser from
/// seeing the scale.
pub fn rms_norm<B: burn_backend::Backend, const D: usize>(
    x: &burn_tensor::Tensor<B, D>,
    gamma: &burn_tensor::Tensor<B, 1>,
    eps: f64,
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
    // `RmsNorm::forward` takes a half type's statistic in F32 and casts it back: that composition
    // is the module's own.
    if x.dtype() != DType::F32 || gamma.dtype() != DType::F32 || !scaled_on() {
        return None;
    }
    let eps = eps as f32;
    if TypeId::of::<B>() == TypeId::of::<crate::TtBackend>() {
        let x: Tensor<crate::TtBackend, D> = cast(x.clone())?;
        let g: Tensor<crate::TtBackend, 1> = cast(gamma.clone())?;
        let y = rms_norm_tt(float(x.into_primitive())?, float(g.into_primitive())?, eps);
        return cast(Tensor::<crate::TtBackend, D>::from_primitive(
            TensorPrimitive::Float(y),
        ));
    }
    #[cfg(feature = "fusion")]
    if TypeId::of::<B>() == TypeId::of::<crate::Tt>() {
        let x: Tensor<crate::Tt, D> = cast(x.clone())?;
        let g: Tensor<crate::Tt, 1> = cast(gamma.clone())?;
        let y = fused::rms_norm(float(x.into_primitive())?, float(g.into_primitive())?, eps);
        return cast(Tensor::<crate::Tt, D>::from_primitive(
            TensorPrimitive::Float(y),
        ));
    }
    None
}
