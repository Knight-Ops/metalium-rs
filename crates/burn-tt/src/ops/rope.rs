//! Rotary position embedding in the rotate-half layout as one op (`Session::rope`), for a model
//! to call where burn-fusion's blocks keep a fuser from seeing it: `rotate_half`'s `cat` ends the
//! block before the products, so on `Tt` RoPE was a copy and two ops. One op here, bit for bit the
//! composition `x cos + cat(x2, x1) sin` (`sin` carrying `rotate_half`'s sign): the product, then
//! `MUL_ADD`'s product and sum, `rotate(x)` read as `x`'s own tiles across each head's halves.

use burn_backend::DType;

use super::*;
use crate::server::BlockArg;
use crate::TtTensor;

/// RoPEs run as the native op since the process started ([`rope_count`]).
static NATIVE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many RoPEs have run as one native op in this process: what a gate reads to see the op
/// taken on `Tt`, whose fusion stream runs off the caller's thread. Dev-facing.
#[doc(hidden)]
pub fn rope_count() -> u64 {
    NATIVE.load(std::sync::atomic::Ordering::SeqCst)
}

/// `x cos + rotate(x) sin` for `x` `[n, heads, t, head]` (F32, a view or plain) and tables
/// `[1, 1, t, head]` (or `x`'s shape), on the device as one op, where `head / 2` is whole tiles
/// and `x`'s heads are column blocks of its buffer. `None` otherwise: the caller composes it.
pub(crate) fn rope_native(x: &TtTensor, cos: &TtTensor, sin: &TtTensor) -> Option<TtTensor> {
    let sa = x.shape().to_vec();
    if sa.len() != 4 || sa.contains(&0) {
        return None;
    }
    let head = sa[3];
    let device = x.device;
    if head % 64 != 0
        || [x.dtype(), cos.dtype(), sin.dtype()] != [DType::F32; 3]
        || cos.shape() != sin.shape()
        || cos.device != device
        || sin.device != device
        || !crate::server::supports_dram(device)
        || matches!(
            crate::server::elementwise_mode(device),
            crate::ElementwiseMode::Matrix { .. }
        )
    {
        return None;
    }
    let (v, [rows, cols]) = match block_view(x) {
        Some(l) => l,
        None => {
            let d = plain_dram(x);
            if d.transposed || d.buffer.storage != crate::storage::StorageFormat::F32 {
                return None;
            }
            let dims = [d.buffer.rows, d.buffer.cols];
            (crate::views::Strided::of(&d, &sa), dims)
        }
    };
    // Heads as column blocks: the last dimension along columns, consecutive, and every other
    // dimension moving whole heads (or whole rows).
    if v.strides[3] != [0, 1]
        || cols % head != 0
        || v.base[1] % 32 != 0
        || (0..3).any(|d| sa[d] > 1 && v.strides[d][1] % head != 0)
    {
        return None;
    }
    // The tables laid out as `x` is (each head's row), kept beside them as the composition's
    // product keeps them.
    let laid = |t: &TtTensor| -> Option<crate::tensor::DramRef> {
        let t = if t.shape().to_vec() == sa {
            t.clone()
        } else {
            relaid(t, &sa, &v.strides)?
        };
        let d = match block_view(&t) {
            Some((w, dims))
                if w.base == [0, 0] && dims == [w.src.buffer.rows, w.src.buffer.cols] =>
            {
                w.src
            }
            Some(_) => return None,
            None => plain_dram(&t),
        };
        ([d.buffer.rows, d.buffer.cols] == [rows, cols] && !d.transposed).then_some(d)
    };
    let (dc, ds) = (laid(cos)?, laid(sin)?);
    let whole = |d: &crate::tensor::DramRef| BlockArg {
        id: d.buffer.id,
        at: [0, 0],
        dims: [rows, cols],
    };
    let xa = BlockArg {
        id: v.src.buffer.id,
        at: v.base,
        dims: [rows, cols],
    };
    let _held = (&v.src, &dc, &ds);
    let _part = crate::report::part("rope/native");
    NATIVE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let (id, dims) = crate::server::rope_blocks(device, xa, whole(&dc), whole(&ds), head);
    let out = device_view(device, id, dims, None, DType::F32)
        .to_dram()
        .clone();
    Some(TtTensor::view(
        crate::views::Strided {
            src: out,
            base: [0, 0],
            strides: v.strides,
        },
        x.shape(),
        DType::F32,
        device,
    ))
}

/// The composition on this backend: `x cos + cat(x2, x1) sin`.
pub(crate) fn rope_composed(x: TtTensor, cos: TtTensor, sin: TtTensor) -> TtTensor {
    let shape = x.shape().to_vec();
    let (n, h, t, d) = (shape[0], shape[1], shape[2], shape[3]);
    let half = d / 2;
    let x1 = float::float_slice(
        x.clone(),
        &[
            (0..n).into(),
            (0..h).into(),
            (0..t).into(),
            (0..half).into(),
        ],
    );
    let x2 = float::float_slice(
        x.clone(),
        &[
            (0..n).into(),
            (0..h).into(),
            (0..t).into(),
            (half..d).into(),
        ],
    );
    let swapped = float::float_cat(vec![x2, x1], 3);
    float::float_add(float::float_mul(x, cos), float::float_mul(swapped, sin))
}

/// [`rope_native`], else [`rope_composed`]: charged to an op of its own, `rope`, as Burn's ops are.
pub(crate) fn rope_tt(x: TtTensor, cos: TtTensor, sin: TtTensor) -> TtTensor {
    let _op = crate::report::enter("rope", true);
    let out = rope_native(&x, &cos, &sin).unwrap_or_else(|| rope_composed(x, cos, sin));
    // On the device either way (a view of the op's buffer, or the composition's tensor): never
    // ask the view for its buffer here, which would copy it out.
    crate::report::made(true);
    out
}

#[cfg(feature = "fusion")]
pub(crate) mod fused {
    use burn_fusion::stream::{Operation, OperationStreams};
    use burn_fusion::FusionTensor;
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use crate::fusion::TtFusionRuntime;
    use crate::{TtBackend, TtHandle};

    #[derive(Debug)]
    struct RopeOp {
        desc: CustomOpIr,
    }

    impl Operation<TtFusionRuntime> for RopeOp {
        fn execute(&self, handles: &mut HandleContainer<TtHandle>) {
            let ([x, cos, sin], [out]) = self.desc.as_fixed::<3, 1>();
            let x = handles.get_float_tensor::<TtBackend>(x);
            let cos = handles.get_float_tensor::<TtBackend>(cos);
            let sin = handles.get_float_tensor::<TtBackend>(sin);
            handles.register_float_tensor::<TtBackend>(&out.id, super::rope_tt(x, cos, sin));
        }
    }

    /// RoPE in `Tt`'s fusion stream as one custom operation, run in order with the rest.
    pub(crate) fn rope(
        x: FusionTensor<TtFusionRuntime>,
        cos: FusionTensor<TtFusionRuntime>,
        sin: FusionTensor<TtFusionRuntime>,
    ) -> FusionTensor<TtFusionRuntime> {
        let client = x.client.clone();
        let streams = OperationStreams::with_inputs([&x, &cos, &sin]);
        let out = TensorIr::uninit(client.create_empty_handle(), x.shape.clone(), x.dtype);
        let desc = CustomOpIr::new(
            "tt_rope",
            &[x.into_ir(), cos.into_ir(), sin.into_ir()],
            &[out],
        );
        client
            .register(streams, OperationIr::Custom(desc.clone()), RopeOp { desc })
            .into_iter()
            .next()
            .expect("one output")
    }
}

/// `x cos + rotate_half(x) sin` (`x` `[n, heads, t, head]`, the tables `[1, 1, t, head]` with
/// `sin`'s first half negated, the rotate-half layout) as one op on `TtBackend` and `Tt`, bit for
/// bit `x * cos + cat(x2, x1) * sin` there. `None` on any other backend: the caller composes it.
/// RoPE is the manual fusion of an op every rotary model runs, where burn-fusion's blocks keep a
/// fuser from seeing it.
pub fn rope<B: burn_backend::Backend>(
    x: &burn_tensor::Tensor<B, 4>,
    cos: &burn_tensor::Tensor<B, 4>,
    sin: &burn_tensor::Tensor<B, 4>,
) -> Option<burn_tensor::Tensor<B, 4>> {
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
        let [x, c, s]: [Tensor<crate::TtBackend, 4>; 3] =
            [cast(x.clone())?, cast(cos.clone())?, cast(sin.clone())?];
        let y = rope_tt(
            float(x.into_primitive())?,
            float(c.into_primitive())?,
            float(s.into_primitive())?,
        );
        return cast(Tensor::<crate::TtBackend, 4>::from_primitive(
            TensorPrimitive::Float(y),
        ));
    }
    #[cfg(feature = "fusion")]
    if TypeId::of::<B>() == TypeId::of::<crate::Tt>() {
        let [x, c, s]: [Tensor<crate::Tt, 4>; 3] =
            [cast(x.clone())?, cast(cos.clone())?, cast(sin.clone())?];
        let y = fused::rope(
            float(x.into_primitive())?,
            float(c.into_primitive())?,
            float(s.into_primitive())?,
        );
        return cast(Tensor::<crate::Tt, 4>::from_primitive(
            TensorPrimitive::Float(y),
        ));
    }
    None
}
