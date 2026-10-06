//! A small MLP demonstrating the same named policy API as the CNN.
//!
//! Ordinary backends explicitly select the ordinary F32 policy:
//! ```
//! use burn::tensor::Tensor;
//! use burn_flex::{Flex, FlexDevice};
//! use burn_tt::storage::F32Policy;
//! let model = tt_mnist::precision::Mlp::<Flex>::new(&FlexDevice);
//! let logits = model.forward(Tensor::ones([2, 784], &FlexDevice), &F32Policy);
//! assert_eq!(logits.dims(), [2, 10]);
//! ```
//!
//! A single-card native model keeps F32 masters and casts its named boundaries:
//! ```no_run
//! use burn::{backend::Autodiff, tensor::Tensor};
//! use burn_tt::{TtBackend, TtDevice, Topology, SrcRoute, Fidelity};
//! use burn_tt::storage::{NativeStoragePolicy, StorageFormat as S, TensorStorageExt};
//! let device = TtDevice::new(0);
//! let _server = burn_tt::attach_topology(device, Topology::single(0),
//!     SrcRoute::Tf32FromFp32, Fidelity::HiFi4).unwrap();
//! let model = tt_mnist::precision::Mlp::<Autodiff<TtBackend>>::new(&device);
//! let policy = NativeStoragePolicy(|name| match name {
//!     "mlp.hidden.weight" => S::Bfp2,
//!     "mlp.relu" => S::Bfp4,
//!     "mlp.head.weight" => S::Bfp8,
//!     _ => S::F32,
//! });
//! let logits = model.forward(Tensor::ones([4, 784], &device), &policy);
//! // Biases promote to F32. Sensitive losses still declare their input boundary.
//! let loss = logits.with_storage(S::F32).sum();
//! let _gradients = loss.backward();
//! ```
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::relu, backend::Backend, Tensor},
};
use burn_tt::storage::PrecisionPolicy;
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub hidden: Linear<B>,
    pub head: Linear<B>,
}
impl<B: Backend> Mlp<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            hidden: LinearConfig::new(784, 16).init(device),
            head: LinearConfig::new(16, 10).init(device),
        }
    }
    pub fn forward<P: PrecisionPolicy<B>>(&self, images: Tensor<B, 2>, policy: &P) -> Tensor<B, 2> {
        let weight = policy.apply("mlp.hidden.weight", self.hidden.weight.val());
        let mut hidden = policy.apply("mlp.input", images).matmul(weight);
        if let Some(bias) = &self.hidden.bias {
            hidden = hidden + policy.apply("mlp.hidden.bias", bias.val()).unsqueeze();
        }
        let hidden = policy.apply("mlp.relu", relu(hidden));
        let weight = policy.apply("mlp.head.weight", self.head.weight.val());
        let mut out = hidden.matmul(weight);
        if let Some(bias) = &self.head.bias {
            out = out + policy.apply("mlp.head.bias", bias.val()).unsqueeze();
        }
        policy.apply("mlp.logits", out)
    }
}
