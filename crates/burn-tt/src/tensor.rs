//! The tensor primitives: a `burn-flex` tensor, and the device it is on.
//!
//! Today every tensor's data lives on the host and the device only borrows it
//! for the ops that run there. The device is still carried per tensor, because
//! Burn asks each tensor which device it is on (`float_device`), and because a
//! second chip (Phase 8) makes the answer matter.

use burn_backend::quantization::QuantScheme;
use burn_backend::{DType, QTensorPrimitive, Shape, TensorMetadata};
use burn_flex::{FlexQTensor, FlexTensor};

use crate::TtDevice;

/// Float, int and bool tensors alike, as Flex uses one primitive for all three.
#[derive(Clone, Debug)]
pub struct TtTensor {
    pub(crate) inner: FlexTensor,
    pub(crate) device: TtDevice,
}

impl TtTensor {
    pub(crate) fn new(inner: FlexTensor, device: TtDevice) -> Self {
        TtTensor { inner, device }
    }
}

impl TensorMetadata for TtTensor {
    fn dtype(&self) -> DType {
        self.inner.dtype()
    }
    fn shape(&self) -> Shape {
        self.inner.shape()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
}

/// A quantized tensor: Flex's, and its device.
#[derive(Clone, Debug)]
pub struct TtQTensor {
    pub(crate) inner: FlexQTensor,
    pub(crate) device: TtDevice,
}

impl TensorMetadata for TtQTensor {
    fn dtype(&self) -> DType {
        self.inner.dtype()
    }
    fn shape(&self) -> Shape {
        self.inner.shape()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
}

impl QTensorPrimitive for TtQTensor {
    fn scheme(&self) -> &QuantScheme {
        self.inner.scheme()
    }
    fn default_scheme() -> QuantScheme {
        FlexQTensor::default_scheme()
    }
}
