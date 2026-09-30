//! What each type in an op signature becomes on the way to `burn-flex` and
//! back.
//!
//! The generated forwarding (`generated/delegate.rs`) is purely syntactic: it
//! calls [`IntoFlex::into_flex`] on every argument whose type names the backend,
//! [`FromFlex::from_flex`] on every such result, and [`HasDevice::tt_device`] on
//! the first argument that carries a device. Everything the conversion means is
//! here, where the compiler checks it: a type the generator meets that has no
//! impl is a build error, not a guess.

use burn_backend::ops::{
    DeformConv2dBackward, MaxPool1dBackward, MaxPool1dWithIndices, MaxPool2dBackward,
    MaxPool2dWithIndices, TransactionPrimitive,
};
use burn_backend::quantization::QuantizationParametersPrimitive;
use burn_backend::TensorPrimitive;
use burn_flex::{Flex, FlexDevice, FlexQTensor, FlexTensor};

use crate::{TtBackend, TtDevice, TtQTensor, TtTensor};

/// `burn-tt` -> `burn-flex`.
pub trait IntoFlex {
    type Flex;
    fn into_flex(self) -> Self::Flex;
}

/// `burn-flex` -> `burn-tt`, on `device`.
pub trait FromFlex<F> {
    fn from_flex(flex: F, device: TtDevice) -> Self;
}

/// Which device a value is on.
pub trait HasDevice {
    fn tt_device(&self) -> TtDevice;
}

// --- Tensors ------------------------------------------------------------------

impl IntoFlex for TtTensor {
    type Flex = FlexTensor;
    fn into_flex(self) -> FlexTensor {
        self.inner
    }
}

impl<'a> IntoFlex for &'a TtTensor {
    type Flex = &'a FlexTensor;
    fn into_flex(self) -> &'a FlexTensor {
        &self.inner
    }
}

impl FromFlex<FlexTensor> for TtTensor {
    fn from_flex(inner: FlexTensor, device: TtDevice) -> Self {
        TtTensor::new(inner, device)
    }
}

impl HasDevice for TtTensor {
    fn tt_device(&self) -> TtDevice {
        self.device
    }
}

impl IntoFlex for TtQTensor {
    type Flex = FlexQTensor;
    fn into_flex(self) -> FlexQTensor {
        self.inner
    }
}

impl<'a> IntoFlex for &'a TtQTensor {
    type Flex = &'a FlexQTensor;
    fn into_flex(self) -> &'a FlexQTensor {
        &self.inner
    }
}

impl FromFlex<FlexQTensor> for TtQTensor {
    fn from_flex(inner: FlexQTensor, device: TtDevice) -> Self {
        TtQTensor { inner, device }
    }
}

impl HasDevice for TtQTensor {
    fn tt_device(&self) -> TtDevice {
        self.device
    }
}

impl<T: HasDevice> HasDevice for &T {
    fn tt_device(&self) -> TtDevice {
        (**self).tt_device()
    }
}

// --- Devices ------------------------------------------------------------------

/// Flex has one device; every `TtDevice` computes there on the host.
impl IntoFlex for &TtDevice {
    type Flex = &'static FlexDevice;
    fn into_flex(self) -> &'static FlexDevice {
        &FlexDevice
    }
}

impl HasDevice for TtDevice {
    fn tt_device(&self) -> TtDevice {
        *self
    }
}

// --- Containers ---------------------------------------------------------------

impl<T: IntoFlex> IntoFlex for Vec<T> {
    type Flex = Vec<T::Flex>;
    fn into_flex(self) -> Self::Flex {
        self.into_iter().map(IntoFlex::into_flex).collect()
    }
}

impl<F, T: FromFlex<F>> FromFlex<Vec<F>> for Vec<T> {
    fn from_flex(flex: Vec<F>, device: TtDevice) -> Self {
        flex.into_iter().map(|f| T::from_flex(f, device)).collect()
    }
}

/// A list's device is its first element's. Burn never hands an op an empty
/// list of tensors; if it did, the device would be the default one.
impl<T: HasDevice> HasDevice for Vec<T> {
    fn tt_device(&self) -> TtDevice {
        self.first().map(HasDevice::tt_device).unwrap_or_default()
    }
}

impl<T: IntoFlex> IntoFlex for Option<T> {
    type Flex = Option<T::Flex>;
    fn into_flex(self) -> Self::Flex {
        self.map(IntoFlex::into_flex)
    }
}

impl<F, T: FromFlex<F>> FromFlex<Option<F>> for Option<T> {
    fn from_flex(flex: Option<F>, device: TtDevice) -> Self {
        flex.map(|f| T::from_flex(f, device))
    }
}

impl<F1, F2, T1: FromFlex<F1>, T2: FromFlex<F2>> FromFlex<(F1, F2)> for (T1, T2) {
    fn from_flex((a, b): (F1, F2), device: TtDevice) -> Self {
        (T1::from_flex(a, device), T2::from_flex(b, device))
    }
}

impl<F1, F2, F3, T1: FromFlex<F1>, T2: FromFlex<F2>, T3: FromFlex<F3>> FromFlex<(F1, F2, F3)>
    for (T1, T2, T3)
{
    fn from_flex((a, b, c): (F1, F2, F3), device: TtDevice) -> Self {
        (
            T1::from_flex(a, device),
            T2::from_flex(b, device),
            T3::from_flex(c, device),
        )
    }
}

// --- Burn's composite types ---------------------------------------------------

impl IntoFlex for TensorPrimitive<TtBackend> {
    type Flex = TensorPrimitive<Flex>;
    fn into_flex(self) -> TensorPrimitive<Flex> {
        match self {
            TensorPrimitive::Float(t) => TensorPrimitive::Float(t.inner),
            TensorPrimitive::QFloat(t) => TensorPrimitive::QFloat(t.inner),
        }
    }
}

impl FromFlex<TensorPrimitive<Flex>> for TensorPrimitive<TtBackend> {
    fn from_flex(flex: TensorPrimitive<Flex>, device: TtDevice) -> Self {
        match flex {
            TensorPrimitive::Float(t) => TensorPrimitive::Float(TtTensor::new(t, device)),
            TensorPrimitive::QFloat(t) => TensorPrimitive::QFloat(TtQTensor { inner: t, device }),
        }
    }
}

impl HasDevice for TensorPrimitive<TtBackend> {
    fn tt_device(&self) -> TtDevice {
        match self {
            TensorPrimitive::Float(t) => t.device,
            TensorPrimitive::QFloat(t) => t.device,
        }
    }
}

impl IntoFlex for QuantizationParametersPrimitive<TtBackend> {
    type Flex = QuantizationParametersPrimitive<Flex>;
    fn into_flex(self) -> Self::Flex {
        QuantizationParametersPrimitive {
            scales: self.scales.inner,
        }
    }
}

/// Through `TransactionPrimitive::new`, whose private read order is empty:
/// `execute_async` takes the order out before it calls `tr_execute`, and puts
/// the results back in it afterwards, so `tr_execute` never sees one.
impl IntoFlex for TransactionPrimitive<TtBackend> {
    type Flex = TransactionPrimitive<Flex>;
    fn into_flex(self) -> Self::Flex {
        TransactionPrimitive::new(
            self.read_floats.into_flex(),
            self.read_qfloats.into_flex(),
            self.read_ints.into_flex(),
            self.read_bools.into_flex(),
        )
    }
}

impl FromFlex<MaxPool1dWithIndices<Flex>> for MaxPool1dWithIndices<TtBackend> {
    fn from_flex(f: MaxPool1dWithIndices<Flex>, device: TtDevice) -> Self {
        MaxPool1dWithIndices {
            output: TtTensor::new(f.output, device),
            indices: TtTensor::new(f.indices, device),
        }
    }
}

impl FromFlex<MaxPool2dWithIndices<Flex>> for MaxPool2dWithIndices<TtBackend> {
    fn from_flex(f: MaxPool2dWithIndices<Flex>, device: TtDevice) -> Self {
        MaxPool2dWithIndices {
            output: TtTensor::new(f.output, device),
            indices: TtTensor::new(f.indices, device),
        }
    }
}

impl FromFlex<MaxPool1dBackward<Flex>> for MaxPool1dBackward<TtBackend> {
    fn from_flex(f: MaxPool1dBackward<Flex>, device: TtDevice) -> Self {
        MaxPool1dBackward {
            x_grad: TtTensor::new(f.x_grad, device),
        }
    }
}

impl FromFlex<MaxPool2dBackward<Flex>> for MaxPool2dBackward<TtBackend> {
    fn from_flex(f: MaxPool2dBackward<Flex>, device: TtDevice) -> Self {
        MaxPool2dBackward {
            x_grad: TtTensor::new(f.x_grad, device),
        }
    }
}

impl FromFlex<DeformConv2dBackward<Flex>> for DeformConv2dBackward<TtBackend> {
    fn from_flex(f: DeformConv2dBackward<Flex>, device: TtDevice) -> Self {
        DeformConv2dBackward {
            x_grad: TtTensor::new(f.x_grad, device),
            offset_grad: TtTensor::new(f.offset_grad, device),
            weight_grad: TtTensor::new(f.weight_grad, device),
            mask_grad: f.mask_grad.map(|t| TtTensor::new(t, device)),
            bias_grad: f.bias_grad.map(|t| TtTensor::new(t, device)),
        }
    }
}
