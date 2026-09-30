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
fn device_result(device: TtDevice, id: crate::server::BufferId, [m, n]: [usize; 2]) -> TtTensor {
    let buffer = std::sync::Arc::new(crate::tensor::Buffer {
        id,
        device,
        rows: m,
        cols: n,
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
fn device_eltwise(kind: u32, scalar: f32, a: &TtTensor, b: Option<&TtTensor>) -> Option<TtTensor> {
    use tt_isa::dm::kind as k;
    let device = a.device;
    let all = |f: &dyn Fn(&TtTensor) -> bool| f(a) && b.is_none_or(f);
    if !all(&|t: &TtTensor| t.is_matrix_f32() && t.device == device) {
        return None;
    }
    if a.dram().is_none() && b.is_none_or(|b| b.dram().is_none()) {
        return None;
    }
    if !crate::server::supports_dram(device) {
        return None;
    }
    let (sa, sb) = (a.shape().to_vec(), b.map(|b| b.shape().to_vec()));
    let (kind, a, b) = match (kind, &sb) {
        (_, None) => (kind, a, b),
        (_, Some(s)) if *s == sa => (kind, a, b),
        (k::ADD, Some(s)) if s[0] == 1 && s[1] == sa[1] => (k::ADD_ROW, a, b),
        // Addition commutes bit for bit, so a row on the left is a row too.
        (k::ADD, Some(s)) if sa[0] == 1 && sa[1] == s[1] => (k::ADD_ROW, b?, Some(a)),
        _ => return None,
    };
    let (da, db) = (a.to_dram(), b.map(|b| b.to_dram()));
    if da.transposed || db.is_some_and(|d| d.transposed) {
        return None;
    }
    let (id, dims) =
        crate::server::eltwise(device, kind, scalar, da.buffer.id, db.map(|d| d.buffer.id));
    Some(device_result(device, id, dims))
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

    pub fn float_device(tensor: &FloatTensor<TtBackend>) -> Device<TtBackend> {
        tensor.device
    }

    pub fn float_to_device(
        tensor: FloatTensor<TtBackend>,
        device: &Device<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        retag(tensor, device)
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
