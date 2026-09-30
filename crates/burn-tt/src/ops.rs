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

/// A tensor moved to `device`. The data lives on the host either way (see
/// `tensor.rs`), so this changes the tag and nothing else.
fn retag(tensor: TtTensor, device: &TtDevice) -> TtTensor {
    TtTensor::new(tensor.inner, *device)
}

pub mod float {
    use super::*;

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
            let out = <Flex as FloatTensorOps<Flex>>::float_matmul(lhs.inner, rhs.inner);
            return TtTensor::new(out, device);
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
        let a = expand(lhs.inner, vec![m, k]);
        let b = expand(rhs.inner, vec![k, n]);
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
        <Flex as FloatTensorOps<Flex>>::float_into_data(tensor.inner)
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
        <Flex as IntTensorOps<Flex>>::int_into_data(tensor.inner)
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
        <Flex as BoolTensorOps<Flex>>::bool_into_data(tensor.inner)
    }

    pub fn bool_argwhere(
        tensor: BoolTensor<TtBackend>,
        out_dtype: IntDType,
    ) -> impl Future<Output = IntTensor<TtBackend>> + 'static + Send {
        let device = tensor.device;
        let fut = <Flex as BoolTensorOps<Flex>>::bool_argwhere(tensor.inner, out_dtype);
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
