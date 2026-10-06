//! Per-tensor physical storage, separate from the logical F32 dtype.
//!
//! Casts return a fresh resident allocation and use identity backward (a
//! straight-through approximation). Arithmetic widens on device. Unary/scalar
//! results keep storage; binary results use the higher input precision;
//! reductions and rearrangements with new exponent groups return F32.
//! Gradients, master parameters, optimizer state and sensitive losses use F32.
//! Serialization returns decoded F32; reapply a storage policy after loading.
use crate::{
    tensor::{Buffer, DramRef},
    TtBackend, TtTensor,
};
use burn_backend::{DType, TensorMetadata};
use burn_tensor::{Tensor, TensorPrimitive};
use std::sync::Arc;
pub use tt_kernels::bfp::BfpFormat;

/// Physical precision, independent of logical dtype. Higher values have higher
/// precision; mixed arithmetic promotes using this ordering.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum StorageFormat {
    Bfp2,
    Bfp4,
    Bfp8,
    #[default]
    F32,
}
impl StorageFormat {
    fn packed(self) -> Option<BfpFormat> {
        match self {
            Self::F32 => None,
            Self::Bfp8 => Some(BfpFormat::Bfp8),
            Self::Bfp4 => Some(BfpFormat::Bfp4),
            Self::Bfp2 => Some(BfpFormat::Bfp2),
        }
    }
}
impl TtTensor {
    pub fn storage_format(&self) -> StorageFormat {
        self.physical_storage()
    }
    /// Convert logical F32 to explicit single-card physical storage. BF16 must
    /// first be converted to logical F32. No source host cache is retained.
    pub fn with_storage(self, format: StorageFormat) -> Self {
        let _op = crate::report::enter("with_storage", true);
        assert_eq!(self.dtype(), DType::F32, "BFP storage requires logical F32");

        assert!(
            crate::server::mesh_execution(self.device).is_none(),
            "BFP mesh storage is not supported"
        );
        let input = if self.storage_format() != StorageFormat::F32 && format != StorageFormat::F32 {
            self.with_storage(StorageFormat::F32)
        } else {
            self
        };
        let src = input.to_dram();
        let dims = [src.buffer.rows, src.buffer.cols];
        let (id, [rows, cols]) =
            if input.storage_format() == StorageFormat::F32 && format == StorageFormat::F32 {
                let sources = (0..dims[0] * dims[1])
                    .map(|i| [i / dims[1], i % dims[1]])
                    .collect();
                crate::server::repack(input.device, src.buffer.id, sources, dims)
            } else {
                crate::server::cast_bfp(input.device, src.buffer.id, dims, format.packed())
            };
        let dram = DramRef {
            buffer: Arc::new(Buffer {
                id,
                device: input.device,
                rows,
                cols,
                parent: None,
                storage: format,
            }),
            transposed: src.transposed,
        };
        Self::on_device(dram, input.shape(), DType::F32, input.device)
    }
}

/// Backend extension; BFP is not portable Burn quantization.
pub trait StorageBackend: burn_backend::Backend {
    fn with_storage(
        tensor: Self::FloatTensorPrimitive,
        format: StorageFormat,
    ) -> Self::FloatTensorPrimitive;
    fn storage_format(tensor: &Self::FloatTensorPrimitive) -> StorageFormat;
}
impl StorageBackend for TtBackend {
    fn with_storage(t: TtTensor, f: StorageFormat) -> TtTensor {
        t.with_storage(f)
    }
    fn storage_format(t: &TtTensor) -> StorageFormat {
        t.storage_format()
    }
}
impl<B: StorageBackend, C: burn_autodiff::checkpoint::strategy::CheckpointStrategy> StorageBackend
    for burn_autodiff::Autodiff<B, C>
{
    fn with_storage(t: Self::FloatTensorPrimitive, f: StorageFormat) -> Self::FloatTensorPrimitive {
        use burn_autodiff::ops::{unary, Backward, Ops};
        #[derive(Debug)]
        struct Cast;
        impl<B: StorageBackend> Backward<B, 1> for Cast {
            type State = ();
            fn backward(
                self,
                ops: Ops<(), 1>,
                grads: &mut burn_autodiff::grads::Gradients,
                _checkpointer: &mut burn_autodiff::checkpoint::base::Checkpointer,
            ) {
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::with_storage(grad, StorageFormat::F32)
                });
            }
        }
        Cast.prepare::<C>([t.node.clone()])
            .compute_bound()
            .stateless(B::with_storage(t.primitive, f))
    }
    fn storage_format(t: &Self::FloatTensorPrimitive) -> StorageFormat {
        B::storage_format(&t.primitive)
    }
}
#[cfg(feature = "fusion")]
impl StorageBackend for burn_fusion::Fusion<TtBackend> {
    fn with_storage(t: Self::FloatTensorPrimitive, f: StorageFormat) -> Self::FloatTensorPrimitive {
        // Resolving executes pending work before the cast. Register a new
        // handle afterward, so fusers cannot remove the compression boundary.
        let client = t.client.clone();
        let stream = t.stream;
        let native = crate::fusion::resolve_float_tensor(&t).with_storage(f);
        let shape = native.shape();
        let dtype = native.dtype();
        let id = client.register_tensor_handle(crate::TtHandle::Tensor(native));
        burn_fusion::Client::change_client_float::<TtBackend>(
            burn_ir::TensorIr {
                id,
                shape,
                dtype,
                status: burn_ir::TensorStatus::ReadWrite,
            },
            client.clone(),
            client,
            stream,
        )
    }
    fn storage_format(t: &Self::FloatTensorPrimitive) -> StorageFormat {
        crate::fusion::resolve_float_tensor(t).storage_format()
    }
}

/// Import this trait for `Tensor::with_storage` and `storage_format`.
pub trait TensorStorageExt: Sized {
    fn with_storage(self, format: StorageFormat) -> Self;
    fn storage_format(&self) -> StorageFormat;
}
impl<B: StorageBackend, const D: usize> TensorStorageExt for Tensor<B, D> {
    fn with_storage(self, f: StorageFormat) -> Self {
        let TensorPrimitive::Float(t) = self.into_primitive() else {
            panic!("BFP requires a logical float tensor")
        };
        Tensor::from_primitive(TensorPrimitive::Float(B::with_storage(t, f)))
    }
    fn storage_format(&self) -> StorageFormat {
        let TensorPrimitive::Float(t) = self.clone().into_primitive() else {
            panic!("BFP requires a logical float tensor")
        };
        B::storage_format(&t)
    }
}

/// A shared model can name storage boundaries without requiring every backend
/// to implement BFP. Select `F32Policy` explicitly on ordinary backends.
pub trait PrecisionPolicy<B: burn_backend::Backend> {
    fn apply<const D: usize>(&self, name: &str, tensor: Tensor<B, D>) -> Tensor<B, D>;
}
#[derive(Clone, Copy, Debug, Default)]
pub struct F32Policy;
impl<B: burn_backend::Backend> PrecisionPolicy<B> for F32Policy {
    fn apply<const D: usize>(&self, _name: &str, tensor: Tensor<B, D>) -> Tensor<B, D> {
        assert_eq!(
            tensor.dtype(),
            DType::F32,
            "ordinary F32 policy requires logical F32"
        );
        tensor
    }
}
/// Choose a physical format for each named weight or activation boundary.
#[derive(Clone, Copy)]
pub struct NativeStoragePolicy(pub fn(&str) -> StorageFormat);
impl<B: StorageBackend> PrecisionPolicy<B> for NativeStoragePolicy {
    fn apply<const D: usize>(&self, name: &str, tensor: Tensor<B, D>) -> Tensor<B, D> {
        tensor.with_storage((self.0)(name))
    }
}
