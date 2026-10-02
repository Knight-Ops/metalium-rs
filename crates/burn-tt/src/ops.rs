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
    use tt_isa::dm::kind as k;
    let kind = op.kind;
    let device = a.device;
    // Any rank: each operand as the matrix it is stored as
    // (`crate::tensor::stored_dims`), of the element type the kind computes on.
    let sig = tt_kernels::sfpu::ops::elems(kind)?;
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
    use tt_isa::dm::kind;

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
    // `atan2(lhs, rhs)` within `ops::ATAN2_BOUND`, a denormal operand read as
    // a zero; tensors of one shape (a broadcast's row program would not fit
    // a slot), others Flex's.
    binary!(float_atan2, tt_kernels::sfpu::ops::kind_sfpu::ATAN2);

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

    /// Swapping the two dimensions of a matrix already on the device is a view
    /// of the same buffer: the device matmul reads it transposed, and a host op
    /// downloads it transposed. Anything else is Flex's.
    pub fn float_swap_dims(
        tensor: FloatTensor<TtBackend>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<TtBackend> {
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
        reshaped(tensor, shape, <Flex as FloatTensorOps<Flex>>::float_reshape)
    }

    /// The sum over rows (`dim` 0) of a matrix on the device stays there, in
    /// Flex's order (`tt_isa::dm::kind::COL_SUM`). Anything else is Flex's.
    pub fn float_sum_dim(tensor: FloatTensor<TtBackend>, dim: usize) -> FloatTensor<TtBackend> {
        use tt_kernels::sfpu::reduce::ReduceOp;
        if let Some(t) = device_reduce(&tensor, ReduceOp::Sum, dim) {
            return t;
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_sum_dim(tensor.into_host(), dim),
            device,
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

    /// `tensor` reduced along `dim` on the device, if it is a device-resident
    /// F32 matrix (not a transposed view): a sum over rows by the mover in
    /// Flex's order, everything else on the SFPU (`Session::reduce`).
    pub(crate) fn device_reduce(
        tensor: &TtTensor,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        dim: usize,
    ) -> Option<TtTensor> {
        use tt_kernels::sfpu::reduce::ReduceOp;
        // Only the mover's sum over rows gives Flex's bits; the SFPU's sum is
        // in another order, and its maximum prefers `+0` to `-0`. Those, not
        // in exact mode, and not on a tensor too small to pay for themselves.
        if (op, dim) != (ReduceOp::Sum, 0) && (crate::exact() || tiles(tensor) < APPROX_MIN_TILES) {
            return None;
        }
        device_reduce_ungated(tensor, op, dim)
    }

    /// [`device_reduce`] without the size gate, for a composition.
    pub(crate) fn device_reduce_ungated(
        tensor: &TtTensor,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        dim: usize,
    ) -> Option<TtTensor> {
        use tt_kernels::sfpu::reduce::Axis;
        let device = tensor.device;
        if dim > 1 || !tensor.is_matrix_f32() || !crate::server::supports_dram(device) {
            return None;
        }
        let d = tensor.dram().filter(|d| !d.transposed)?;
        let axis = if dim == 0 { Axis::Rows } else { Axis::Cols };
        let (id, dims) = crate::server::reduce(device, d.buffer.id, op, axis);
        Some(device_result(device, id, dims))
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

pub mod activation {
    use super::*;
    use burn_backend::ops::ActivationOps;
    use burn_backend::Scalar;
    use tt_isa::dm::kind;

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
