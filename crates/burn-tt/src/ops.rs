//! The op methods `burn-tt` implements by hand; every other one is forwarded to
//! `burn-flex` by `generated/delegate.rs`. Each is listed in
//! `xtask/src/gen_burn.rs`'s `OVERRIDDEN`, which the generator checks against
//! the pinned op traits.
//!
//! Three kinds: the ones that run on the device (`float_matmul`), the ones
//! that say or change which device a tensor is on, and the ones returning
//! futures, which the generator does not forward.

use core::future::Future;

use burn_backend::ops::{
    BoolTensorOps, FloatTensorOps, IntTensorOps, QTensorOps, TransactionOps, TransactionPrimitive,
    TransactionPrimitiveData,
};
use burn_backend::tensor::{BoolTensor, Device, FloatTensor, IntTensor, QuantizedTensor};
use burn_backend::{DType, ExecutionError, IntDType, TensorData, TensorMetadata};
use burn_flex::{Flex, FlexTensor};

use crate::convert::IntoFlex;
use crate::server::Elem;
use crate::{TtBackend, TtDevice, TtQTensor, TtTensor};
use tt_kernels::sfpu::ops::kind_sfpu;

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

/// `a (kind) b` -- or `a (kind) scalar` with `b` `None` -- on the device, if
/// that is where the data is: every operand an F32 matrix, at least one already
/// on the device, none a transposed view, and the shapes ones the kernels take
/// (`b` the same shape, or for `ADD_ROW` one row). `None` otherwise, and the
/// caller runs Flex's op on the host copies.
/// Tiles below which an SFPU-only op (an approximation: division, `exp`,
/// `log`, the SFPU's reductions) runs on the host after a download instead.
/// Each SFPU kernel op costs 100-200 us whatever its size -- most of it the
/// host's submission (`silicon_perf::softmax_parts`, card 0) -- against ~190
/// us to download a tile; and a small tensor's chain of such ops (autodiff's
/// own `log_softmax` is five forward and more backward) usually ends on the
/// host anyway. MNIST's two-tile logits trained at 7.9 ms/step on one tile
/// with them on the device, 3.6 without. A heuristic until submission is
/// asynchronous or a lookahead exists (`burn-backend-parity.md` B8, B13, B16).
/// Re-measured with batched submission (2026-10-03, card 0, one tile):
/// without the threshold MNIST trains at 1.6 ms/step against 1.0 (its
/// two-tile log-softmax still ends in a host loss), and the transformer of
/// `tt-mnist --model transformer` at 9.0 against 10.4 (its four-tile layer
/// norm statistics feed device ops). Kept until a lookahead can tell the two.
/// Measured again with the loss's gather on the card (D4): MNIST 1.1 against
/// 1.7-1.8 ms/step without the threshold, the transformer 12.0 against
/// 10.0-11.0 -- MNIST's two-tile log-softmax is cheaper on the host still.
const APPROX_MIN_TILES: usize = 8;

/// Tiles in the tile grid of the matrix a tensor is stored as.
fn tiles(t: &TtTensor) -> usize {
    t.stored()
        .map_or(0, |[r, c]| r.div_ceil(32) * c.div_ceil(32))
}

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

fn device_eltwise(kind: u32, scalar: f32, a: &TtTensor, b: Option<&TtTensor>) -> Option<TtTensor> {
    // An approximation: not in exact mode, and not on a tensor too small to
    // pay its fixed cost (`APPROX_MIN_TILES`). An exact op follows the data.
    use tt_kernels::sfpu::ops::{accuracy, Accuracy};
    if accuracy(kind) == Accuracy::Approximate && (crate::exact() || tiles(a) < APPROX_MIN_TILES) {
        return None;
    }
    device_eltwise_ungated(kind, scalar, a, b)
}

/// [`device_eltwise`] without the size gate: for a composition that has
/// already decided, on its whole input, to run on the device (a softmax's
/// `log` of its small per-row sums, say).
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
    use tt_kernels::sfpu::ops::{accuracy, Accuracy};
    if accuracy(op.kind) == Accuracy::Approximate && (crate::exact() || tiles(a) < APPROX_MIN_TILES)
    {
        return None;
    }
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
    if a.dram().is_none() && b.is_none_or(|b| b.dram().is_none()) {
        return None;
    }
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
        _ => return None,
    };
    // A broadcast row or column made on the host usually belongs to a host
    // chain -- a row max for a loss on the host, say: uploading it to meet a
    // device tensor only to download the result for the chain's next host op
    // is two crossings where there were none (MNIST's loss, in exact mode,
    // does exactly that). It goes to the device only if it is there already
    // -- except a row added to every row, a bias, which a host optimiser
    // step leaves on the host and the next forward pass wants on the device.
    let [ar, ac] = a.stored().expect("checked");
    let bias = kind == k::ADD && b.is_some_and(|b| b.stored() == Some([1, ac]) && ar != 1);
    if !bias && b.is_some_and(|b| b.shape() != a.shape() && b.dram().is_none()) {
        return None;
    }
    let (da, db, dc) = (a.to_dram(), b.map(|b| b.to_dram()), c.map(|c| c.to_dram()));
    if da.transposed || db.is_some_and(|d| d.transposed) || dc.is_some_and(|d| d.transposed) {
        return None;
    }
    let (id, dims) = crate::server::eltwise_op(
        device,
        op,
        da.buffer.id,
        db.map(|d| d.buffer.id),
        dc.map(|d| d.buffer.id),
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
    if crate::exact() || tiles(x) < APPROX_MIN_TILES {
        return None;
    }
    let device = x.device;
    if !x.is_storable() || x.elem() != Some(Elem::F32) || !crate::server::supports_dram(device) {
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
    if x.dram().is_none() && yt.is_none_or(|t| t.dram().is_none()) {
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
fn reshaped(
    tensor: TtTensor,
    shape: burn_backend::Shape,
    host: impl Fn(burn_flex::FlexTensor, burn_backend::Shape) -> burn_flex::FlexTensor,
) -> TtTensor {
    let same =
        tensor.stored().is_some() && tensor.stored() == crate::tensor::stored_dims(&shape.to_vec());
    if let Some(d) = tensor.dram().filter(|d| !d.transposed) {
        if same && tensor.is_storable() {
            return TtTensor::on_device(d.clone(), shape, tensor.dtype(), tensor.device);
        }
    }
    if same && tensor.is_storable() && tensor.dram().is_none() {
        let h = host(tensor.host().clone(), shape.clone());
        return tensor.reshaped_host(h, shape);
    }
    let device = tensor.device;
    TtTensor::new(host(tensor.into_host(), shape), device)
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

    macro_rules! binary {
        ($name:ident, $kind:expr) => {
            /// On the device where the data is ([`super::device_eltwise`]),
            /// else Flex's.
            pub fn $name(
                lhs: FloatTensor<TtBackend>,
                rhs: FloatTensor<TtBackend>,
            ) -> FloatTensor<TtBackend> {
                if let Some(t) = device_eltwise($kind, 0.0, &lhs, Some(&rhs)) {
                    return t;
                }
                let device = lhs.device;
                TtTensor::new(
                    <Flex as FloatTensorOps<Flex>>::$name(lhs.into_host(), rhs.into_host()),
                    device,
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
        assert!(
            lhs.shape() == rhs.shape(),
            "float_atan2 of shapes {:?} and {:?}: burn-tt computes atan2 of tensors \
             of one shape only (no broadcast; hardware-coverage.md 10.2f) -- expand \
             the smaller operand first",
            lhs.shape(),
            rhs.shape()
        );
        let kind = tt_kernels::sfpu::ops::kind_sfpu::ATAN2;
        if let Some(t) = device_eltwise(kind, 0.0, &lhs, Some(&rhs)) {
            return t;
        }
        let device = lhs.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_atan2(lhs.into_host(), rhs.into_host()),
            device,
        )
    }

    /// `1/x` on the device where the data is, within one ulp of Flex's
    /// (`kind_sfpu::RECIP`), else Flex's.
    pub fn float_recip(tensor: FloatTensor<TtBackend>) -> FloatTensor<TtBackend> {
        if let Some(t) = device_eltwise(tt_kernels::sfpu::ops::kind_sfpu::RECIP, 0.0, &tensor, None)
        {
            return t;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_recip(tensor.into_host()),
            device,
        )
    }

    macro_rules! unary_sfpu {
        ($name:ident, $kind:expr, $doc:literal) => {
            #[doc = $doc]
            pub fn $name(tensor: FloatTensor<TtBackend>) -> FloatTensor<TtBackend> {
                if let Some(t) = device_eltwise($kind, 0.0, &tensor, None) {
                    return t;
                }
                let device = tensor.device;
                TtTensor::new(
                    <Flex as FloatTensorOps<Flex>>::$name(tensor.into_host()),
                    device,
                )
            }
        };
    }
    unary_sfpu!(
        float_exp,
        tt_kernels::sfpu::ops::kind_sfpu::EXP,
        "`e^x` on the device where the data is, within `ops::EXP_BOUND` of the exact value, else Flex's."
    );
    unary_sfpu!(
        float_log,
        tt_kernels::sfpu::ops::kind_sfpu::LOG,
        "`ln x` on the device where the data is, within `ops::LOG_BOUND` of the exact value, else Flex's."
    );

    /// `x / s` on the device where the data is, within one ulp of Flex's
    /// (`kind_sfpu::DIV_SCALAR`), else Flex's.
    pub fn float_div_scalar(lhs: FloatTensor<TtBackend>, rhs: Scalar) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        let s = rhs.to_f64().expect("a float scalar") as f32;
        if let Some(t) = device_eltwise(tt_kernels::sfpu::ops::kind_sfpu::DIV_SCALAR, s, &lhs, None)
        {
            return t;
        }
        let device = lhs.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_div_scalar(lhs.into_host(), rhs),
            device,
        )
    }

    /// On the device where the data is, else Flex's. The scalar is converted
    /// exactly as Flex converts it (`to_f64() as f32`, `burn-flex`
    /// `ops/binary.rs:454`).
    pub fn float_mul_scalar(lhs: FloatTensor<TtBackend>, rhs: Scalar) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        let s = rhs.to_f64().expect("a float scalar") as f32;
        if let Some(t) = device_eltwise(kind::MUL_SCALAR, s, &lhs, None) {
            return t;
        }
        let device = lhs.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_mul_scalar(lhs.into_host(), rhs),
            device,
        )
    }

    /// On the device where the data is, else Flex's; the scalar converted as
    /// Flex converts it.
    pub fn float_add_scalar(lhs: FloatTensor<TtBackend>, rhs: Scalar) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        let s = rhs.to_f64().expect("a float scalar") as f32;
        if let Some(t) = device_eltwise(kind::ADD_SCALAR, s, &lhs, None) {
            return t;
        }
        let device = lhs.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_add_scalar(lhs.into_host(), rhs),
            device,
        )
    }

    /// `x - s` as `x + (-s)`: the same bits in IEEE arithmetic, signed zeros
    /// included (`ADD_SCALAR`). On the device where the data is, else Flex's.
    pub fn float_sub_scalar(lhs: FloatTensor<TtBackend>, rhs: Scalar) -> FloatTensor<TtBackend> {
        use num_traits::ToPrimitive;
        let s = rhs.to_f64().expect("a float scalar") as f32;
        if let Some(t) = device_eltwise(kind::ADD_SCALAR, -s, &lhs, None) {
            return t;
        }
        let device = lhs.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_sub_scalar(lhs.into_host(), rhs),
            device,
        )
    }

    /// `lhs @ rhs` over the last two dimensions, with the leading ones
    /// broadcast as Burn broadcasts them, on the device both are tagged with.
    ///
    /// F32 only: anything else is computed by Flex, because the device path
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
            let out =
                <Flex as FloatTensorOps<Flex>>::float_matmul(lhs.into_host(), rhs.into_host());
            return TtTensor::new(out, device);
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
        let expand = |t: FlexTensor, mut tail: Vec<usize>| -> Vec<f32> {
            let mut shape = batch.clone();
            shape.append(&mut tail);
            let t = <Flex as FloatTensorOps<Flex>>::float_reshape(
                t.clone(),
                burn_backend::Shape::from({
                    let mut s = vec![1; rank - t.shape().num_dims()];
                    s.extend(t.shape().to_vec());
                    s
                }),
            );
            let t =
                <Flex as FloatTensorOps<Flex>>::float_expand(t, burn_backend::Shape::from(shape));
            t.into_data()
                .to_vec::<f32>()
                .expect("an F32 tensor reads back as f32")
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
        TtTensor::new(FlexTensor::from_data(TensorData::new(out, shape)), device)
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
    /// device copy, if it is F32 and has one (or is a view itself): nothing
    /// moves, at any rank. A swap that leaves it a plain matrix or its 2-D
    /// transpose is that, as before.
    fn swapped_strided(tensor: &TtTensor, dim1: usize, dim2: usize) -> Option<TtTensor> {
        if tensor.dtype() != DType::F32 || dim1 == dim2 {
            return None;
        }
        let v = tensor.as_strided()?;
        let mut shape = tensor.shape().to_vec();
        shape.swap(dim1, dim2);
        Some(TtTensor::view(
            v.swapped(dim1, dim2),
            burn_backend::Shape::from(shape),
            tensor.device,
        ))
    }

    /// A reshape of a device-resident F32 tensor (or a view) that changes the
    /// matrix it is stored as: a view when strides still express it -- the
    /// heads split out of a projection -- else, when it moves whole tiles, a
    /// copy on the card into the new matrix (the heads merged back), else
    /// `None`. A reshape that keeps a plain tensor's matrix is
    /// [`reshaped`]'s.
    fn reshaped_strided(tensor: &TtTensor, shape: &burn_backend::Shape) -> Option<TtTensor> {
        if tensor.dtype() != DType::F32 {
            return None;
        }
        let (from, to) = (tensor.shape().to_vec(), shape.to_vec());
        let keeps = crate::tensor::stored_dims(&from) == crate::tensor::stored_dims(&to);
        if keeps && !tensor.is_view() && !tensor.is_transposed() {
            return None;
        }
        let v = tensor.as_strided()?;
        if let Some(r) = v.reshaped(&from, &to) {
            return Some(TtTensor::view(r, shape.clone(), tensor.device));
        }
        let moves = v.tile_moves(&from, &to)?;
        let dims = crate::tensor::stored_dims(&to)?;
        let (id, dims) = crate::server::copy_blocks(tensor.device, v.src.buffer.id, moves, dims);
        Some(device_result_shaped(
            tensor.device,
            id,
            dims,
            shape.clone(),
            DType::F32,
        ))
    }

    /// Swapping the two dimensions of a matrix already on the device is a view
    /// of the same buffer: the device matmul reads it transposed, and a host op
    /// downloads it transposed. Anything else is Flex's.
    pub fn float_swap_dims(
        tensor: FloatTensor<TtBackend>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<TtBackend> {
        if let Some(v) = swapped_strided(&tensor, dim1, dim2) {
            return v;
        }
        if let Some(v) = swapped_view(&tensor, dim1, dim2) {
            return v;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_swap_dims(tensor.into_host(), dim1, dim2),
            device,
        )
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
    /// there; else Flex's, on the host copy.
    pub fn float_reshape(
        tensor: FloatTensor<TtBackend>,
        shape: burn_backend::Shape,
    ) -> FloatTensor<TtBackend> {
        if let Some(t) = reshaped_strided(&tensor, &shape) {
            return t;
        }
        reshaped(tensor, shape, <Flex as FloatTensorOps<Flex>>::float_reshape)
    }

    /// The sum over rows (`dim` 0) of a matrix on the device stays there, in
    /// Flex's order (`tt_kernels::sfpu::reduce::accumulate_in_order`).
    /// Anything else is Flex's.
    pub fn float_sum_dim(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        use tt_kernels::sfpu::reduce::ReduceOp;
        if let Some(t) = device_reduce(&tensor, ReduceOp::Sum, dim) {
            return t;
        }
        if let Some(t) = device_sum_leading(&tensor, dim) {
            return t;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_sum_dim(tensor.into_host(), dim),
            device,
        )
    }

    /// The mean along `dim`: the sum, then a multiplication by the count's
    /// reciprocal -- each on the device where its operand is
    /// ([`float_sum_dim`], `float_mul_scalar`; within one rounding of Flex's
    /// division, inside the sum's own bound), when the tensor is on the
    /// device. On the host, or in exact mode, Flex's.
    pub fn float_mean_dim(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        let n = tensor.shape().to_vec()[dim];
        if crate::exact() || tensor.dtype() != DType::F32 || tensor.as_strided().is_none() {
            let device = tensor.device;
            return TtTensor::new(
                <Flex as FloatTensorOps<Flex>>::float_mean_dim(tensor.into_host(), dim),
                device,
            );
        }
        <TtBackend as FloatTensorOps<TtBackend>>::float_mul_scalar(
            float_sum_dim(tensor, dim),
            (1.0 / n as f32).into(),
        )
    }

    /// The maximum along `dim` of a device-resident matrix, on the SFPU
    /// (exactly Flex's value; `step31_reduce`), else Flex's.
    pub fn float_max_dim(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        use tt_kernels::sfpu::reduce::ReduceOp;
        if let Some(t) = device_reduce(&tensor, ReduceOp::Max, dim) {
            return t;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_max_dim(tensor.into_host(), dim),
            device,
        )
    }

    /// `tensor` reduced along `dim` on the device, if it is device-resident
    /// F32 (not a transposed view), on the SFPU (`Session::reduce`): a
    /// matrix along either dim, a tensor of any rank along its last -- the
    /// columns of the matrix it is stored as (softmax's and layer norm's
    /// statistics over `[b, s, d]` or `[b, h, s, s]`). A sum over a matrix's
    /// rows is in Flex's order; the rest as `sfpu::reduce` computes them.
    pub(crate) fn device_reduce(
        tensor: &TtTensor,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        dim: usize,
    ) -> Option<TtTensor> {
        use tt_kernels::sfpu::reduce::ReduceOp;
        // Only the sum over rows gives Flex's bits (it adds in Flex's order);
        // the sum over columns is in a tree order, and the maximum prefers `+0`
        // to `-0`. Those, not
        // in exact mode, and not on a tensor too small to pay for themselves.
        let rows_sum = op == ReduceOp::Sum && dim == 0 && tensor.shape().num_dims() == 2;
        if !rows_sum && (crate::exact() || tiles(tensor) < APPROX_MIN_TILES) {
            return None;
        }
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
        if rank < 2 || dim + 1 >= rank || !tensor.is_stored_f32() {
            return None;
        }
        if shape[..dim].iter().any(|&d| d != 1) || !crate::server::supports_dram(tensor.device) {
            return None;
        }
        let n = shape[dim];
        let q: usize = shape[dim + 1..rank - 1].iter().product();
        let c = shape[rank - 1];
        let device = tensor.device;
        let d = tensor.dram().filter(|d| !d.transposed)?.clone();
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
        use tt_kernels::sfpu::reduce::Axis;
        let device = tensor.device;
        let rank = tensor.shape().num_dims();
        if !tensor.is_stored_f32() || !crate::server::supports_dram(device) {
            return None;
        }
        let axis = if rank == 2 && dim == 0 {
            Axis::Rows
        } else if dim + 1 == rank {
            Axis::Cols
        } else {
            return None;
        };
        let d = tensor.dram().filter(|d| !d.transposed)?;
        let (id, dims) = crate::server::reduce(device, d.buffer.id, op, axis);
        let mut shape = tensor.shape().to_vec();
        shape[dim] = 1;
        Some(device_result_shaped(
            device,
            id,
            dims,
            burn_backend::Shape::from(shape),
            DType::F32,
        ))
    }

    /// Whole tile rows, all columns, of a matrix on the device: a view of the
    /// same slots, nothing copied -- how a batch is taken from a dataset
    /// uploaded once. Anything else is Flex's.
    pub fn float_slice(
        tensor: FloatTensor<TtBackend>,
        slices: &[burn_backend::Slice],
    ) -> FloatTensor<TtBackend> {
        let device = tensor.device;
        if let Some(view) = row_view(&tensor, slices) {
            return view;
        }
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_slice(tensor.into_host(), slices),
            device,
        )
    }

    pub(crate) fn row_view(tensor: &TtTensor, slices: &[burn_backend::Slice]) -> Option<TtTensor> {
        if !tensor.is_storable_matrix() || slices.is_empty() || slices.len() > 2 {
            return None;
        }
        let d = tensor.dram().filter(|d| !d.transposed)?;
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

    // S2 (10.2c): exact on the device, so wherever the data is, any size,
    // exact mode included.
    unary_sfpu!(
        float_neg,
        kind_sfpu::NEG,
        "`-x`, the sign bit flipped (NaNs too), on the device where the data is, else Flex's."
    );
    unary_sfpu!(
        float_abs,
        kind_sfpu::ABS,
        "`|x|`, the sign bit cleared (NaNs too), on the device where the data is, else Flex's."
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
    /// panics on (a NaN, or crossed) go to Flex's, which does.
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
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_clamp(tensor.into_host(), min, max),
            device,
        )
    }

    pub fn float_clamp_min(tensor: FloatTensor<TtBackend>, min: Scalar) -> FloatTensor<TtBackend> {
        let s = scalar_f32(min);
        if let Some(t) = device_eltwise(kind_sfpu::CLAMP_MIN, s, &tensor, None) {
            return t;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_clamp_min(tensor.into_host(), min),
            device,
        )
    }

    pub fn float_clamp_max(tensor: FloatTensor<TtBackend>, max: Scalar) -> FloatTensor<TtBackend> {
        let s = scalar_f32(max);
        if let Some(t) = device_eltwise(kind_sfpu::CLAMP_MAX, s, &tensor, None) {
            return t;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_clamp_max(tensor.into_host(), max),
            device,
        )
    }

    macro_rules! compare {
        ($name:ident, $kind:expr) => {
            /// An IEEE comparison on the device where the data is (a row or
            /// column broadcast on the right), its `Bool` resident; else Flex's.
            pub fn $name(
                lhs: FloatTensor<TtBackend>,
                rhs: FloatTensor<TtBackend>,
                out_dtype: burn_backend::BoolDType,
            ) -> BoolTensor<TtBackend> {
                if let Some(t) = device_eltwise($kind, 0.0, &lhs, Some(&rhs)) {
                    return retyped(t, out_dtype.into());
                }
                let device = lhs.device;
                TtTensor::new(
                    <Flex as FloatTensorOps<Flex>>::$name(
                        lhs.into_host(),
                        rhs.into_host(),
                        out_dtype,
                    ),
                    device,
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
            /// `f32`), on the device where the data is; else Flex's.
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
                let device = lhs.device;
                TtTensor::new(
                    <Flex as FloatTensorOps<Flex>>::$name(lhs.into_host(), rhs, out_dtype),
                    device,
                )
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
            /// On the device where the data is, its `Bool` resident; else Flex's.
            pub fn $name(
                tensor: FloatTensor<TtBackend>,
                out_dtype: burn_backend::BoolDType,
            ) -> BoolTensor<TtBackend> {
                if let Some(t) = device_eltwise($kind, 0.0, &tensor, None) {
                    return retyped(t, out_dtype.into());
                }
                let device = tensor.device;
                TtTensor::new(
                    <Flex as FloatTensorOps<Flex>>::$name(tensor.into_host(), out_dtype),
                    device,
                )
            }
        };
    }
    predicate!(float_is_nan, kind_sfpu::IS_NAN);
    predicate!(float_is_inf, kind_sfpu::IS_INF);

    /// `tensor`'s column index (`0..c` along the last dimension) at every
    /// element, as F32 on the device: what a one-index-per-row gather
    /// compares its indices with. Made on the host and uploaded per call (an
    /// attachment-scoped cache waits on `burn-backend-parity.md` B3).
    fn column_indices(tensor: &TtTensor) -> Option<TtTensor> {
        let shape = tensor.shape().to_vec();
        let c = *shape.last()?;
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| (i % c) as f32).collect();
        let t = TtTensor::new(
            FlexTensor::from_data(TensorData::new(v, shape)),
            tensor.device,
        );
        t.to_dram();
        Some(t)
    }

    /// For a gather or scatter along the last dimension with one index per
    /// row (`indices` `tensor`'s shape with a last of 1, as Burn's
    /// `CrossEntropyLoss` gathers its targets): `kind` (`EQ` or `NE`) of each
    /// element's column index and its row's index, a resident `Bool` of
    /// `tensor`'s shape, on the device. `None` unless `tensor` is resident
    /// F32 and the indices are `I32` (uploaded if they are on the host).
    fn index_mask(
        tensor: &TtTensor,
        dim: usize,
        indices: &TtTensor,
        kind: u32,
    ) -> Option<TtTensor> {
        let (ts, is) = (tensor.shape().to_vec(), indices.shape().to_vec());
        let rank = ts.len();
        if rank == 0 || dim + 1 != rank || is.len() != rank || is[rank - 1] != 1 {
            return None;
        }
        if ts[..rank - 1] != is[..rank - 1] || indices.dtype() != DType::I32 {
            return None;
        }
        if !tensor.is_stored_f32() || tensor.as_strided().is_none() {
            return None;
        }
        if !crate::server::supports_dram(tensor.device) || ts[rank - 1] > 1 << 24 {
            return None;
        }
        indices.to_dram();
        let idx = device_eltwise_ungated(kind_sfpu::I32_TO_F32, 0.0, indices, None)?;
        let cols = column_indices(tensor)?;
        device_eltwise_ungated(kind, 0.0, &cols, Some(&idx))
    }

    /// `out[.., 0] = x[.., idx[.., 0]]` along the last dimension, on the
    /// device where `x` is: every other element masked to `-0` and the row
    /// summed -- the gathered element in any order, since `x + -0 = x` for
    /// every `x` (a NaN and the infinities included), but for a gathered
    /// `-0`, which the SFPU's sum returns as `+0` (`ttsim-divergence.md` row
    /// C; keeping the sign would take a second, max-based pass). One index
    /// per row (a loss's targets); anything else, Flex's.
    pub fn float_gather(
        dim: usize,
        tensor: FloatTensor<TtBackend>,
        indices: IntTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        use tt_kernels::sfpu::reduce::ReduceOp;
        let device = tensor.device;
        if let Some(ne) = index_mask(&tensor, dim, &indices, kind_sfpu::NE) {
            if let Some(kept) =
                device_eltwise_ungated(kind_sfpu::MASK_FILL, -0.0, &tensor, Some(&ne))
            {
                if let Some(t) = device_reduce_ungated(&kept, ReduceOp::Sum, dim) {
                    return t;
                }
            }
        }
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_gather(
                dim,
                tensor.into_host(),
                indices.into_host(),
            ),
            device,
        )
    }

    /// `out = x`, then `out[.., idx[.., 0]] += v[.., 0]` along the last
    /// dimension -- a gather's backward -- on the device where `x` is: `x +
    /// v` (a column broadcast) where the column is the row's index, `x`
    /// elsewhere; the one addition Flex makes. One index per row; anything
    /// else, Flex's.
    pub fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<TtBackend>,
        indices: IntTensor<TtBackend>,
        value: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        let device = tensor.device;
        if value.shape() == indices.shape() && value.is_stored_f32() {
            if let Some(eq) = index_mask(&tensor, dim, &indices, kind_sfpu::EQ) {
                value.to_dram();
                if let Some(sum) = device_eltwise_ungated(kind::ADD, 0.0, &tensor, Some(&value)) {
                    let op = op2(kind_sfpu::MASK_WHERE, 0.0, 0.0);
                    if let Some(t) = device_op_ungated(op, &tensor, Some(&eq), Some(&sum)) {
                        return t;
                    }
                }
            }
        }
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_scatter_add(
                dim,
                tensor.into_host(),
                indices.into_host(),
                value.into_host(),
            ),
            device,
        )
    }

    /// An index tensor's values, from its host copy (downloaded first if it
    /// has none: indices are few, and an embedding's come from the host),
    /// if every one is in `0..bound`.
    fn index_values(indices: &TtTensor, bound: usize) -> Option<Vec<usize>> {
        let v = indices
            .host()
            .clone()
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .ok()?;
        v.into_iter()
            .map(|i| usize::try_from(i).ok().filter(|&i| i < bound))
            .collect()
    }

    /// The rows of the matrix `tensor` is stored as that slice `i` along
    /// dimension 0 covers: `i q .. (i + 1) q`, `q` the product of the
    /// dimensions between the first and the last.
    fn rows_of(shape: &[usize], idx: &[usize]) -> Vec<usize> {
        let q: usize = shape[1..shape.len() - 1].iter().product();
        idx.iter().flat_map(|&i| (i * q)..((i + 1) * q)).collect()
    }

    /// The slices `indices` name along dimension 0, on the device: an
    /// embedding's lookup -- each stored row moved where it lies, two 64-byte
    /// reads a tile column (`Session::gather_rows`), bit for bit. The table is
    /// uploaded if it is on the host, as a matmul's operands are: a
    /// parameter only an embedding reads would otherwise never reach the
    /// card. Indices are read on the host. Another dimension, Flex's.
    pub fn float_select(
        tensor: FloatTensor<TtBackend>,
        dim: usize,
        indices: IntTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        let device = tensor.device;
        let shape = tensor.shape().to_vec();
        if dim == 0
            && shape.len() >= 2
            && tensor.is_stored_f32()
            && indices.shape().num_dims() == 1
            && indices.shape().num_elements() > 0
            && crate::server::supports_dram(device)
        {
            if let Some(idx) = index_values(&indices, shape[0]) {
                if let Some(d) = Some(tensor.to_dram()).filter(|d| !d.transposed) {
                    let rows: Vec<(usize, usize)> =
                        rows_of(&shape, &idx).into_iter().map(|r| (0, r)).collect();
                    let cols = shape[shape.len() - 1];
                    let (id, dims) =
                        crate::server::gather_rows(device, vec![d.buffer.id], rows, cols);
                    let mut out = shape.clone();
                    out[0] = idx.len();
                    return device_result_shaped(
                        device,
                        id,
                        dims,
                        burn_backend::Shape::from(out),
                        DType::F32,
                    );
                }
            }
        }
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_select(
                tensor.into_host(),
                dim,
                indices.into_host(),
            ),
            device,
        )
    }

    /// `tensor` with `value`'s slice `i` added to its slice `indices[i]`
    /// along dimension 0, in order -- an embedding's gradient -- on the device
    /// where `value` is (`Session::rows_add`: only the rows the indices touch
    /// are computed, in Flex's order of additions). Indices are read on the
    /// host. Another dimension, or `value` on the host, Flex's.
    pub fn float_select_add(
        tensor: FloatTensor<TtBackend>,
        dim: usize,
        indices: IntTensor<TtBackend>,
        value: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        let device = tensor.device;
        let shape = tensor.shape().to_vec();
        let vshape = value.shape().to_vec();
        if dim == 0
            && shape.len() >= 2
            && vshape.len() == shape.len()
            && vshape[1..] == shape[1..]
            && tensor.is_stored_f32()
            && value.is_stored_f32()
            && value.as_strided().is_some()
            && indices.shape().num_elements() == vshape[0]
            && vshape[0] > 0
            && crate::server::supports_dram(device)
        {
            if let Some(idx) = index_values(&indices, shape[0]) {
                let t = tensor.to_dram().clone();
                if let Some(v) = value.dram().filter(|d| !d.transposed).cloned() {
                    if !t.transposed {
                        let rows = rows_of(&shape, &idx);
                        let (id, dims) =
                            crate::server::rows_add(device, t.buffer.id, rows, v.buffer.id);
                        return device_result_shaped(device, id, dims, tensor.shape(), DType::F32);
                    }
                }
            }
        }
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_select_add(
                tensor.into_host(),
                dim,
                indices.into_host(),
                value.into_host(),
            ),
            device,
        )
    }

    /// `mask ? value : x` on the device where the data is (the mask may be a
    /// row or column of `x`'s matrix), every bit of `x` kept; else Flex's.
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
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_mask_fill(
                tensor.into_host(),
                mask.into_host(),
                value,
            ),
            device,
        )
    }

    /// `mask ? value : x`, all three one shape, on the device where the data is;
    /// else Flex's.
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
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_mask_where(
                tensor.into_host(),
                mask.into_host(),
                value.into_host(),
            ),
            device,
        )
    }

    // S4 (10.2d): approximations within their derived bounds.
    unary_sfpu!(
        float_sqrt,
        kind_sfpu::SQRT,
        "`sqrt x` on the device where the data is, within one ulp of the correctly rounded root, else Flex's."
    );
    unary_sfpu!(
        float_log1p,
        kind_sfpu::LOG1P,
        "`ln(1 + x)` on the device where the data is, within `ops::LOG1P_BOUND`, else Flex's."
    );
    // S4 (10.2e).
    unary_sfpu!(
        float_tanh,
        kind_sfpu::TANH,
        "`tanh x` on the device where the data is, within `ops::TANH_BOUND`, else Flex's."
    );
    unary_sfpu!(
        float_erf,
        kind_sfpu::ERF,
        "`erf x` on the device where the data is, within `ops::ERF_BOUND`, else Flex's."
    );
    unary_sfpu!(
        float_sinh,
        kind_sfpu::SINH,
        "`sinh x` on the device where the data is, within `ops::SINH_BOUND`, else Flex's."
    );
    unary_sfpu!(
        float_cosh,
        kind_sfpu::COSH,
        "`cosh x` on the device where the data is, within `ops::COSH_BOUND`, else Flex's."
    );
    unary_sfpu!(
        float_asinh,
        kind_sfpu::ASINH,
        "`asinh x` on the device where the data is, within `ops::ASINH_BOUND`, else Flex's."
    );
    unary_sfpu!(
        float_acosh,
        kind_sfpu::ACOSH,
        "`acosh x` on the device where the data is, within `ops::ACOSH_BOUND`, else Flex's."
    );
    unary_sfpu!(
        float_atanh,
        kind_sfpu::ATANH,
        "`atanh x` on the device where the data is, within `ops::ATANH_BOUND`, else Flex's."
    );
    unary_sfpu!(
        float_sin,
        kind_sfpu::SIN,
        "`sin x` on the device where the data is, within `ops::SIN_BOUND` for every finite `x`, else Flex's."
    );
    unary_sfpu!(
        float_cos,
        kind_sfpu::COS,
        "`cos x` on the device where the data is, within `ops::COS_BOUND` for every finite `x`, else Flex's."
    );
    unary_sfpu!(
        float_tan,
        kind_sfpu::TAN,
        "`tan x` on the device where the data is, within `ops::TAN_BOUND` for every finite `x`, else Flex's."
    );
    unary_sfpu!(
        float_atan,
        kind_sfpu::ATAN,
        "`atan x` on the device where the data is, within `ops::ATAN_BOUND`, else Flex's."
    );
    unary_sfpu!(
        float_asin,
        kind_sfpu::ASIN,
        "`asin x` on the device where the data is, within `ops::ASIN_BOUND`, else Flex's."
    );
    unary_sfpu!(
        float_acos,
        kind_sfpu::ACOS,
        "`acos x` on the device where the data is, within `ops::ACOS_BOUND`, else Flex's."
    );

    /// `x^y`, `y` a tensor of `x`'s shape, on the device where the data is
    /// (within `ops::pow_bound`); else Flex's.
    pub fn float_powf(
        lhs: FloatTensor<TtBackend>,
        rhs: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        if let Some(t) = device_pow(&lhs, PowY::Tensor(&rhs)) {
            return t;
        }
        let device = lhs.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_powf(lhs.into_host(), rhs.into_host()),
            device,
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
        let device = lhs.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_powi(lhs.into_host(), rhs.into_host()),
            device,
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
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_powf_scalar_impl(tensor.into_host(), value),
            device,
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
                let device = lhs.device;
                TtTensor::new(
                    <Flex as FloatTensorOps<Flex>>::float_powi_scalar(lhs.into_host(), rhs),
                    device,
                )
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
    /// Flex's (S6).
    pub fn float_cast(
        tensor: FloatTensor<TtBackend>,
        dtype: burn_backend::FloatDType,
    ) -> FloatTensor<TtBackend> {
        if DType::from(dtype) == tensor.dtype() {
            return tensor;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_cast(tensor.into_host(), dtype),
            device,
        )
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
        <Flex as FloatTensorOps<Flex>>::float_into_data(tensor.into_host())
    }
}

pub mod module {
    use super::*;
    use burn_backend::Shape;

    /// Burn's own composition (`ModuleOps::embedding`'s default: a
    /// `select` of the table's rows, reshaped), through this backend's ops,
    /// so the lookup runs on the device where the table is
    /// (`float::float_select`); Flex's own embedding otherwise.
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
    use burn_backend::ops::ActivationOps;
    use burn_backend::Scalar;
    use tt_kernels::kind;

    /// On the device where the data is, else Flex's.
    pub fn relu(tensor: FloatTensor<TtBackend>) -> FloatTensor<TtBackend> {
        if let Some(t) = device_eltwise(kind::RELU, 0.0, &tensor, None) {
            return t;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as ActivationOps<Flex>>::relu(tensor.into_host()),
            device,
        )
    }

    /// Burn's own composition (`ActivationOps::softmax`'s default) -- a max, a
    /// broadcast subtract, `exp`, a sum and a broadcast divide -- through this
    /// backend's ops, so on a device-resident matrix every step runs on the
    /// device; anything else is Flex's fused softmax.
    pub fn softmax(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        match device_softmax(&tensor, dim, false, false) {
            Some(t) => t,
            None => {
                let device = tensor.device;
                TtTensor::new(
                    <Flex as ActivationOps<Flex>>::softmax(tensor.into_host(), dim),
                    device,
                )
            }
        }
    }

    /// Burn's own composition (`ActivationOps::log_softmax`'s default: the
    /// max-shifted log-sum-exp), on the device; else Flex's.
    pub fn log_softmax(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        match device_softmax(&tensor, dim, true, false) {
            Some(t) => t,
            None => {
                let device = tensor.device;
                TtTensor::new(
                    <Flex as ActivationOps<Flex>>::log_softmax(tensor.into_host(), dim),
                    device,
                )
            }
        }
    }

    /// Burn's own composition (`ActivationOps::softmin`'s default: the
    /// softmax of `-x`), on the device -- the negation exact; else Flex's.
    pub fn softmin(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        match device_softmax(&tensor, dim, false, true) {
            Some(t) => t,
            None => {
                let device = tensor.device;
                TtTensor::new(
                    <Flex as ActivationOps<Flex>>::softmin(tensor.into_host(), dim),
                    device,
                )
            }
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

    /// Tiles below which a softmax is Flex's after a download rather than
    /// five device ops: each device op costs 100-200 us whatever its size, a
    /// tile's download ~190 us and its share of the composition ~55 us
    /// (`silicon_perf::softmax_parts`, card 0), so the composition pays from
    /// about six tiles. A heuristic until a lookahead can see where the result
    /// goes (Burn fusion, `burn-backend-parity.md` B13/B16): MNIST's
    /// `[64, 10]` logits, two tiles bound for a loss on the host, stay there.
    const SOFTMAX_DEVICE_MIN_TILES: usize = 8;

    /// Is `t` an F32 matrix on a device that keeps tensors in GDDR, with a
    /// device copy already, and big enough to compose a softmax over there?
    fn resident_matrix(t: &TtTensor) -> bool {
        let dims = t.shape().to_vec();
        let tiles =
            dims.first().map_or(0, |r| r.div_ceil(32)) * dims.get(1).map_or(0, |c| c.div_ceil(32));
        !crate::exact()
            && tiles >= SOFTMAX_DEVICE_MIN_TILES
            && t.is_matrix_f32()
            && t.dram().is_some_and(|d| !d.transposed)
            && crate::server::supports_dram(t.device)
    }

    /// On the device where the data is, else Flex's.
    pub fn relu_backward(
        output: FloatTensor<TtBackend>,
        grad: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        if let Some(t) = device_eltwise(kind::RELU_BACKWARD, 0.0, &output, Some(&grad)) {
            return t;
        }
        let device = output.device;
        TtTensor::new(
            <Flex as ActivationOps<Flex>>::relu_backward(output.into_host(), grad.into_host()),
            device,
        )
    }

    /// Flex's `x >= 0 ? x : slope * x`, one op on the device where the data
    /// is; else Flex's.
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
        let device = tensor.device;
        TtTensor::new(
            <Flex as ActivationOps<Flex>>::leaky_relu(tensor.into_host(), negative_slope),
            device,
        )
    }

    /// Flex's `(alpha * x + beta).clamp(0, 1)`, one op on the device; else
    /// Flex's.
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
        let device = tensor.device;
        TtTensor::new(
            <Flex as ActivationOps<Flex>>::hard_sigmoid(tensor.into_host(), alpha, beta),
            device,
        )
    }

    /// Flex's `x >= 0 ? x : alpha * x`, one op on the device -- `alpha` the
    /// same shape, or a row of per-channel slopes; else Flex's.
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
        let device = tensor.device;
        TtTensor::new(
            <Flex as ActivationOps<Flex>>::prelu(tensor.into_host(), alpha.into_host()),
            device,
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
                let device = tensor.device;
                TtTensor::new(
                    <Flex as ActivationOps<Flex>>::$name(tensor.into_host()),
                    device,
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
                let device = a.device;
                TtTensor::new(
                    <Flex as ActivationOps<Flex>>::$name(a.into_host(), grad.into_host()),
                    device,
                )
            }
        };
    }
    unary!(
        sigmoid,
        SIGMOID,
        "Flex's two-branch `sigmoid` on the device where the data is, within `ops::SIGMOID_BOUND`; else Flex's."
    );
    binary!(
        sigmoid_backward,
        SIGMOID_BACKWARD,
        "Flex's `g * s * (1 - s)`, exact, on the device where the data is; else Flex's."
    );
    unary!(
        gelu,
        GELU,
        "Flex's `0.5 x (1 + erf(x / sqrt 2))` on the device where the data is, within `ops::GELU_BOUND`; else Flex's."
    );
    binary!(
        gelu_backward,
        GELU_BACKWARD,
        "`g (Phi(x) + x phi(x))` on the device where the data is, within `ops::gelu_backward_bound`; else Flex's."
    );
    unary!(
        log_sigmoid,
        LOG_SIGMOID,
        "Flex's two-branch `log_sigmoid` on the device where the data is, within `ops::LOG_SIGMOID_BOUND`; else Flex's."
    );
    binary!(
        log_sigmoid_backward,
        LOG_SIGMOID_BACKWARD,
        "Flex's `g * sigmoid(-x)` on the device where the data is, within `ops::SIGMOID_BOUND` and the product's rounding; else Flex's."
    );
}

pub mod int {
    use super::*;

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
    /// exact); other float dtypes, Flex's.
    pub fn int_into_float(
        tensor: IntTensor<TtBackend>,
        out_dtype: burn_backend::FloatDType,
    ) -> FloatTensor<TtBackend> {
        if DType::from(out_dtype) == DType::F32 {
            if let Some(t) = device_eltwise(kind_sfpu::I32_TO_F32, 0.0, &tensor, None) {
                return t;
            }
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as IntTensorOps<Flex>>::int_into_float(tensor.into_host(), out_dtype),
            device,
        )
    }

    /// A view where the stored matrix is kept, as `float_reshape`.
    pub fn int_reshape(
        tensor: IntTensor<TtBackend>,
        shape: burn_backend::Shape,
    ) -> IntTensor<TtBackend> {
        reshaped(tensor, shape, <Flex as IntTensorOps<Flex>>::int_reshape)
    }

    /// Whole tile rows of a device matrix, a view; as `float_slice`.
    pub fn int_slice(
        tensor: IntTensor<TtBackend>,
        slices: &[burn_backend::Slice],
    ) -> IntTensor<TtBackend> {
        if let Some(v) = super::float::row_view(&tensor, slices) {
            return v;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as IntTensorOps<Flex>>::int_slice(tensor.into_host(), slices),
            device,
        )
    }

    /// A view of a device matrix transposed; as `float_swap_dims`.
    pub fn int_swap_dims(
        tensor: IntTensor<TtBackend>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<TtBackend> {
        if let Some(v) = swapped_view(&tensor, dim1, dim2) {
            return v;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as IntTensorOps<Flex>>::int_swap_dims(tensor.into_host(), dim1, dim2),
            device,
        )
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
        <Flex as IntTensorOps<Flex>>::int_into_data(tensor.into_host())
    }
}

pub mod bool {
    use super::*;

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

    pub fn bool_reshape(
        tensor: BoolTensor<TtBackend>,
        shape: burn_backend::Shape,
    ) -> BoolTensor<TtBackend> {
        reshaped(tensor, shape, <Flex as BoolTensorOps<Flex>>::bool_reshape)
    }

    pub fn bool_slice(
        tensor: BoolTensor<TtBackend>,
        slices: &[burn_backend::Slice],
    ) -> BoolTensor<TtBackend> {
        if let Some(v) = super::float::row_view(&tensor, slices) {
            return v;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as BoolTensorOps<Flex>>::bool_slice(tensor.into_host(), slices),
            device,
        )
    }

    pub fn bool_swap_dims(
        tensor: BoolTensor<TtBackend>,
        dim1: usize,
        dim2: usize,
    ) -> BoolTensor<TtBackend> {
        if let Some(v) = swapped_view(&tensor, dim1, dim2) {
            return v;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as BoolTensorOps<Flex>>::bool_swap_dims(tensor.into_host(), dim1, dim2),
            device,
        )
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
        let device = tensor.device;
        TtTensor::new(
            <Flex as BoolTensorOps<Flex>>::bool_not(tensor.into_host()),
            device,
        )
    }

    macro_rules! logic {
        ($name:ident, $kind:expr) => {
            /// On the device where the operands are, with a row or column
            /// broadcast; else Flex's.
            pub fn $name(
                lhs: BoolTensor<TtBackend>,
                rhs: BoolTensor<TtBackend>,
            ) -> BoolTensor<TtBackend> {
                if let Some(t) = device_eltwise($kind, 0.0, &lhs, Some(&rhs)) {
                    return t;
                }
                let device = lhs.device;
                TtTensor::new(
                    <Flex as BoolTensorOps<Flex>>::$name(lhs.into_host(), rhs.into_host()),
                    device,
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
        <Flex as BoolTensorOps<Flex>>::bool_into_data(tensor.into_host())
    }

    pub fn bool_argwhere(
        tensor: BoolTensor<TtBackend>,
        out_dtype: IntDType,
    ) -> impl Future<Output = IntTensor<TtBackend>> + 'static + Send {
        let device = tensor.device;
        let fut = <Flex as BoolTensorOps<Flex>>::bool_argwhere(tensor.into_host(), out_dtype);
        async move { TtTensor::new(fut.await, device) }
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
            inner: tensor.inner,
            device: *device,
        }
    }

    pub fn q_into_data(
        tensor: QuantizedTensor<TtBackend>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        <Flex as QTensorOps<Flex>>::q_into_data(tensor.inner)
    }
}

pub mod transaction {
    use super::*;

    pub fn tr_execute(
        transaction: TransactionPrimitive<TtBackend>,
    ) -> impl Future<Output = Result<TransactionPrimitiveData, ExecutionError>> + Send {
        <Flex as TransactionOps<Flex>>::tr_execute(transaction.into_flex())
    }
}
