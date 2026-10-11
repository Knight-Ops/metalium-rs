// The tensor suite's backend: `Base`, with no autodiff wrapper.
// Replaces upstream's `common/backend.rs` (see `setup.rs`).

use ctor::ctor;

use super::{FloatElem, IntElem};

pub type TestBackend = super::setup::Base;

#[ctor]
fn init_device_settings() {
    super::setup::register_engine();
    super::setup::set_default_dtypes::<TestBackend>(
        <FloatElem as burn_tensor::Element>::dtype(),
        <IntElem as burn_tensor::Element>::dtype(),
    );
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
