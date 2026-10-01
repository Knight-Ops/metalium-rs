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
use crate::{TtBackend, TtDevice, TtQTensor, TtTensor};

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
    device_view(device, id, dims, None)
}

/// [`device_result`] for a tensor of `shape`, stored as `dims`
/// (`crate::tensor::stored_dims`).
fn device_result_shaped(
    device: TtDevice,
    id: crate::server::BufferId,
    dims: [usize; 2],
    shape: burn_backend::Shape,
) -> TtTensor {
    let t = device_result(device, id, dims);
    let dram = t.dram().expect("made on the device").clone();
    TtTensor::on_device(dram, shape, device)
}

/// A device result that reads `parent`'s slots, keeping it alive.
fn device_view(
    device: TtDevice,
    id: crate::server::BufferId,
    [m, n]: [usize; 2],
    parent: Option<std::sync::Arc<crate::tensor::Buffer>>,
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
    // The SFPU-only ops are approximations: not in exact mode, and not on a
    // tensor too small to pay their fixed cost (`APPROX_MIN_TILES`).
    if !tt_kernels::sfpu::ops::mover_has(kind) && (crate::exact() || tiles(a) < APPROX_MIN_TILES) {
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
    use tt_isa::dm::kind as k;
    let device = a.device;
    let all = |f: &dyn Fn(&TtTensor) -> bool| f(a) && b.is_none_or(f);
    // Any rank: each operand as the matrix it is stored as
    // (`crate::tensor::stored_dims`).
    if !all(&|t: &TtTensor| t.is_stored_f32() && t.device == device) {
        return None;
    }
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
    let (da, db) = (a.to_dram(), b.map(|b| b.to_dram()));
    if da.transposed || db.is_some_and(|d| d.transposed) {
        return None;
    }
    let (id, dims) =
        crate::server::eltwise(device, kind, scalar, da.buffer.id, db.map(|d| d.buffer.id));
    Some(device_result_shaped(device, id, dims, a.shape()))
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
        if tensor.is_matrix_f32() && dim1 != dim2 && dim1 < 2 && dim2 < 2 {
            if let Some(d) = tensor.dram() {
                let d = crate::tensor::DramRef {
                    buffer: d.buffer.clone(),
                    transposed: !d.transposed,
                };
                let s = tensor.shape().to_vec();
                return TtTensor::on_device(
                    d,
                    burn_backend::Shape::from(vec![s[1], s[0]]),
                    tensor.device,
                );
            }
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
        let same = tensor.stored().is_some()
            && tensor.stored() == crate::tensor::stored_dims(&shape.to_vec());
        if let Some(d) = tensor.dram().filter(|d| !d.transposed) {
            if same && tensor.is_stored_f32() {
                return TtTensor::on_device(d.clone(), shape, tensor.device);
            }
        }
        // On the host, but sharing the source's device-copy slot: a bias
        // reshaped every forward pass is uploaded once, not every pass.
        if same && tensor.is_stored_f32() && tensor.dram().is_none() {
            let host =
                <Flex as FloatTensorOps<Flex>>::float_reshape(tensor.host().clone(), shape.clone());
            return tensor.reshaped_host(host, shape);
        }
        let device = tensor.device;
        TtTensor::new(
            <Flex as FloatTensorOps<Flex>>::float_reshape(tensor.into_host(), shape),
            device,
        )
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

    fn row_view(tensor: &TtTensor, slices: &[burn_backend::Slice]) -> Option<TtTensor> {
        if !tensor.is_matrix_f32() || slices.is_empty() || slices.len() > 2 {
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
        Some(device_view(tensor.device, id, dims, Some(d.buffer.clone())))
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
        let t = retag(tensor, device);
        if t.is_stored_f32() && crate::server::supports_dram(*device) {
            let _ = t.to_dram();
        }
        t
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
        match device_softmax(&tensor, dim, false) {
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
        match device_softmax(&tensor, dim, true) {
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

    /// Softmax (or log-softmax) of a resident matrix along `dim`, every step on
    /// the device: decided once, on the whole input, so the small per-row
    /// statistics in the middle stay there too.
    fn device_softmax(t: &TtTensor, dim: usize, log: bool) -> Option<TtTensor> {
        use super::float::device_reduce_ungated as reduce;
        use tt_kernels::sfpu::ops::kind_sfpu;
        use tt_kernels::sfpu::reduce::ReduceOp;
        if !resident_matrix(t) || dim > 1 {
            return None;
        }
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
}

pub mod int {
    use super::*;

    pub fn int_device(tensor: &IntTensor<TtBackend>) -> Device<TtBackend> {
        tensor.device
    }

    pub fn int_to_device(
        tensor: IntTensor<TtBackend>,
        device: &Device<TtBackend>,
    ) -> IntTensor<TtBackend> {
        retag(tensor, device)
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

    pub fn bool_to_device(
        tensor: BoolTensor<TtBackend>,
        device: &Device<TtBackend>,
    ) -> BoolTensor<TtBackend> {
        retag(tensor, device)
    }

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
