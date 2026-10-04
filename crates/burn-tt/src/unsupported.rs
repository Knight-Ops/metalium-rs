//! Diagnostics only inspect metadata; refusing an op never downloads a tensor.
use crate::{TtBackend, TtDevice, TtQTensor, TtTensor};
use burn_backend::{TensorData, TensorMetadata, TensorPrimitive};

#[track_caller]
pub(crate) fn fail(op: &str, details: impl std::fmt::Display) -> ! {
    panic!("burn-tt: unsupported operation {op}: {details}")
}

pub(crate) trait Context {
    fn context(&self) -> String;
}
pub(crate) fn context(value: &impl Context) -> String {
    value.context()
}

impl Context for TtTensor {
    fn context(&self) -> String {
        format!(
            "shape={:?}, dtype={:?}, device={}",
            self.shape(),
            self.dtype(),
            self.device
        )
    }
}
impl Context for TtQTensor {
    fn context(&self) -> String {
        format!(
            "shape={:?}, dtype={:?}, device={}",
            self.shape(),
            self.dtype(),
            self.device
        )
    }
}
impl Context for TtDevice {
    fn context(&self) -> String {
        self.to_string()
    }
}
impl Context for TensorData {
    fn context(&self) -> String {
        format!("shape={:?}, dtype={:?}", self.shape, self.dtype)
    }
}
impl<T: Context> Context for &T {
    fn context(&self) -> String {
        (*self).context()
    }
}
impl<T: Context> Context for Option<T> {
    fn context(&self) -> String {
        self.as_ref()
            .map_or_else(|| "None".into(), Context::context)
    }
}
impl<T: Context> Context for Vec<T> {
    fn context(&self) -> String {
        format!(
            "[{}]",
            self.iter()
                .map(Context::context)
                .collect::<Vec<_>>()
                .join("; ")
        )
    }
}
impl Context for TensorPrimitive<TtBackend> {
    fn context(&self) -> String {
        match self {
            Self::Float(t) => t.context(),
            Self::QFloat(t) => t.context(),
        }
    }
}
impl Context for burn_backend::quantization::QuantizationParametersPrimitive<TtBackend> {
    fn context(&self) -> String {
        format!("scales=({})", self.scales.context())
    }
}
