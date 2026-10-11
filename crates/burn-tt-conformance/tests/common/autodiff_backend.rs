// The autodiff suites' backend: `Base` under `Autodiff`.
// Replaces upstream's `common/backend.rs` and `common/autodiff.rs`, where autodiff is
// a property of a `DispatchDevice` chosen at run time. Here it is a type, so
// the entry that includes this file names the checkpointing strategy, as
// `CheckpointStrategy`.

use ctor::ctor;

use super::{CheckpointStrategy, FloatElem, IntElem};

pub type TestBackend = burn_autodiff::Autodiff<super::setup::Base, CheckpointStrategy>;

#[ctor]
fn init_device_settings() {
    super::setup::register_engine();
    super::setup::set_default_dtypes::<TestBackend>(
        <FloatElem as burn_tensor::Element>::dtype(),
        <IntElem as burn_tensor::Element>::dtype(),
    );
}

/// Autodiff-enabled device used for tests. The wrapper is the backend's type,
/// so this is the device of the backend underneath.
pub struct AutodiffDevice;

impl AutodiffDevice {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> burn_tt::TtDevice {
        burn_tt::TtDevice::default()
    }
}

/// Collection of types used across tests
#[allow(unused)]
pub mod prelude {
    pub use burn_tensor::Tensor;

    use super::*;
    pub type TestTensor<const D: usize> = Tensor<TestBackend, D>;
    pub type TestTensorInt<const D: usize> = Tensor<TestBackend, D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> = Tensor<TestBackend, D, burn_tensor::Bool>;
}

#[allow(unused)]
pub use prelude::*;
