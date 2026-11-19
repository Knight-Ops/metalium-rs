//! Per-tensor physical storage, separate from the logical F32 dtype.
//!
//! [`StorageFormat::Bf16`] is a weight in half the bytes: a product reads it as its `B` directly
//! (bit for bit the product of its values in F32, `tt_kernels::matmul::SrcRoute::TF32_WITH_BF16_B`),
//! and every other op widens it first. Storing rounds ties-even, so it is exact for BF16-valued
//! data (a BF16 checkpoint widened to F32).
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
    Bf16,
    #[default]
    F32,
}
impl StorageFormat {
    fn packed(self) -> Option<BfpFormat> {
        match self {
            Self::F32 | Self::Bf16 => None,
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
    /// Convert logical F32 to explicit single-card physical storage, or a logical BF16 tensor to
    /// BFP and back ([`TtTensor::bf16_with_storage`]). No source host cache is retained.
    pub fn with_storage(self, format: StorageFormat) -> Self {
        self.stored_as(format, false)
    }

    /// [`Self::with_storage`] for a tensor written by rows (a block-float KV cache,
    /// `docs/plans/block-float-kv.md`): interleaved, as a row scatter takes it, rather than laid
    /// by column as a stored weight is (`Session::store_bfp`).
    pub fn with_row_storage(self, format: StorageFormat) -> Self {
        self.stored_as(format, true)
    }

    fn stored_as(self, format: StorageFormat, rows: bool) -> Self {
        let _op = crate::report::enter("with_storage", true);
        if self.dtype() == DType::BF16 {
            return self.bf16_with_storage(format);
        }
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
        let from = input.storage_format();
        // A transposed view stored in a packed format is its transpose made plain first: a product
        // reads a stored weight as it lies (the tied output head, `x @ table^T`), and a transposed
        // packed weight would be widened at every call.
        let plain = (format != StorageFormat::F32 && src.transposed)
            .then(|| crate::ops::plain_dram(&input));
        let (src, dims) = match &plain {
            Some(p) => (p, [p.buffer.rows, p.buffer.cols]),
            None => (src, dims),
        };
        let (id, [rows, cols]) = if from == StorageFormat::F32 && format == StorageFormat::F32 {
            let sources = (0..dims[0] * dims[1])
                .map(|i| [i / dims[1], i % dims[1]])
                .collect();
            crate::server::repack(input.device, src.buffer.id, sources, dims)
        } else if format == StorageFormat::Bf16 || from == StorageFormat::Bf16 {
            crate::server::cast_bf16_storage(
                input.device,
                src.buffer.id,
                dims,
                format == StorageFormat::Bf16,
            )
        } else {
            crate::server::cast_bfp(input.device, src.buffer.id, dims, format.packed(), rows)
        };
        let dram = DramRef {
            buffer: Arc::new(Buffer {
                id,
                device: input.device,
                rows,
                cols,
                parent: None,
                storage: format,
                plain: Default::default(),
            }),
            transposed: src.transposed && plain.is_none(),
            generation: 0,
        };
        Self::on_device(dram, input.shape(), DType::F32, input.device)
    }
}

impl TtTensor {
    /// A logical BF16 tensor stored as `format` (roadmap W1, native BF16 arithmetic): BFP8/4/2 --
    /// a BF16 stream's weights, which a product reads as its `B` against a BF16 activation, with a
    /// BF16 result -- or [`StorageFormat::F32`], which for a BF16 tensor is its own BF16 slots. The
    /// BFP block is `Session::store_bfp`'s of the widened values; every value a BFP block decodes
    /// to is a BF16 value (at most 7 significant bits under an 8-bit exponent), so decoding it back
    /// to the tensor's own slots is exact. [`StorageFormat::Bf16`] is refused: a BF16 tensor
    /// already is.
    fn bf16_with_storage(self, format: StorageFormat) -> Self {
        let (from, shape, device) = (self.storage_format(), self.shape(), self.device);
        assert!(
            format != StorageFormat::Bf16 && from != StorageFormat::Bf16,
            "a BF16 tensor's storage is BFP or its own"
        );
        if from == format {
            return self;
        }
        // The same buffer, read as the logical F32 tensor it stores.
        let as_f32 =
            |t: &TtTensor| Self::on_device(t.to_dram().clone(), t.shape(), DType::F32, t.device);
        if from == StorageFormat::F32 {
            let stored = crate::ops::float_compute_input(self).with_storage(format);
            return Self::on_device(stored.to_dram().clone(), shape, DType::BF16, device);
        }
        let decoded = as_f32(&self).with_storage(StorageFormat::F32);
        let own = crate::ops::cast_native(decoded, DType::BF16);
        if format == StorageFormat::F32 {
            own
        } else {
            own.bf16_with_storage(format)
        }
    }
}

/// Backend extension; BFP is not portable Burn quantization.
pub trait StorageBackend: burn_backend::Backend {
    fn with_storage(
        tensor: Self::FloatTensorPrimitive,
        format: StorageFormat,
    ) -> Self::FloatTensorPrimitive;
    fn storage_format(tensor: &Self::FloatTensorPrimitive) -> StorageFormat;
    /// [`StorageBackend::with_storage`] for a tensor written by rows (a block-float KV cache):
    /// laid out as a row scatter takes it. By default the same as `with_storage`.
    fn with_row_storage(
        tensor: Self::FloatTensorPrimitive,
        format: StorageFormat,
    ) -> Self::FloatTensorPrimitive {
        Self::with_storage(tensor, format)
    }
}
impl StorageBackend for TtBackend {
    fn with_storage(t: TtTensor, f: StorageFormat) -> TtTensor {
        t.with_storage(f)
    }
    fn with_row_storage(t: TtTensor, f: StorageFormat) -> TtTensor {
        t.with_row_storage(f)
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
        fusion_stored(t, |native| native.with_storage(f))
    }
    fn with_row_storage(t: Self::FloatTensorPrimitive, f: StorageFormat) -> Self::FloatTensorPrimitive {
        fusion_stored(t, |native| native.with_row_storage(f))
    }
    fn storage_format(t: &Self::FloatTensorPrimitive) -> StorageFormat {
        crate::fusion::resolve_float_tensor(t).storage_format()
    }
}

/// A fused tensor cast by `cast`. Resolving executes pending work before the cast. Register a new
/// handle afterward, so fusers cannot remove the compression boundary.
#[cfg(feature = "fusion")]
fn fusion_stored(
    t: <burn_fusion::Fusion<TtBackend> as burn_backend::BackendTypes>::FloatTensorPrimitive,
    cast: impl FnOnce(TtTensor) -> TtTensor,
) -> <burn_fusion::Fusion<TtBackend> as burn_backend::BackendTypes>::FloatTensorPrimitive {
    let client = t.client.clone();
    let stream = t.stream;
    let native = cast(crate::fusion::resolve_float_tensor(&t));
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

/// Import this trait for `Tensor::with_storage` and `storage_format`.
pub trait TensorStorageExt: Sized {
    fn with_storage(self, format: StorageFormat) -> Self;
    /// [`TensorStorageExt::with_storage`] for a tensor written by rows (a block-float KV cache).
    fn with_row_storage(self, format: StorageFormat) -> Self;
    fn storage_format(&self) -> StorageFormat;
}
impl<B: StorageBackend, const D: usize> TensorStorageExt for Tensor<B, D> {
    fn with_storage(self, f: StorageFormat) -> Self {
        let TensorPrimitive::Float(t) = self.into_primitive() else {
            panic!("BFP requires a logical float tensor")
        };
        Tensor::from_primitive(TensorPrimitive::Float(B::with_storage(t, f)))
    }
    fn with_row_storage(self, f: StorageFormat) -> Self {
        let TensorPrimitive::Float(t) = self.into_primitive() else {
            panic!("BFP requires a logical float tensor")
        };
        Tensor::from_primitive(TensorPrimitive::Float(B::with_row_storage(t, f)))
    }
    fn storage_format(&self) -> StorageFormat {
        let TensorPrimitive::Float(t) = self.clone().into_primitive() else {
            panic!("BFP requires a logical float tensor")
        };
        B::storage_format(&t)
    }
}

/// A stored BFP weight's GDDR image, in the shape a cache file records.
#[derive(Clone, Debug)]
pub struct WeightImageParts {
    pub rows: u32,
    pub cols: u32,
    /// 8, 4 or 2.
    pub format: u8,
    /// 0 interleaved, 1 by column.
    pub layout: u8,
    pub slot_bytes: u32,
    pub channels: Vec<Vec<u8>>,
}

/// [`bfp_image`] as [`WeightImageParts`], `Ok(None)` when `tensor` is not stored BFP.
pub fn bfp_image_parts<const D: usize>(
    tensor: &Tensor<crate::Tt, D>,
) -> Result<Option<WeightImageParts>, crate::TtError> {
    let Some(image) = bfp_image(tensor)? else {
        return Ok(None);
    };
    Ok(Some(WeightImageParts {
        rows: image.rows as u32,
        cols: image.cols as u32,
        format: match image.format {
            tt_kernels::bfp::BfpFormat::Bfp8 => 8,
            tt_kernels::bfp::BfpFormat::Bfp4 => 4,
            tt_kernels::bfp::BfpFormat::Bfp2 => 2,
        },
        layout: match image.layout {
            tt_kernels::tensor::Layout::ByColumn => 1,
            tt_kernels::tensor::Layout::Interleaved => 0,
        },
        slot_bytes: image.format.slot_bytes() as u32,
        channels: image.channels,
    }))
}

/// The GDDR image of a stored BFP weight, `Ok(None)` when `tensor` is not one.
/// A fused tensor is resolved first, so the image is what the card holds.
pub fn bfp_image<const D: usize>(
    tensor: &Tensor<crate::Tt, D>,
) -> Result<Option<tt_kernels::bfp::BfpImage>, crate::TtError> {
    let TensorPrimitive::Float(tensor) = tensor.clone().into_primitive() else {
        return Ok(None);
    };
    native_of(&tensor).bfp_image()
}

/// [`tensor_from_bfp_image`] from the fields a weight-cache record stores.
pub fn tensor_from_bfp_parts(
    device: &crate::TtDevice,
    rows: usize,
    cols: usize,
    format: u8,
    layout: u8,
    slot_bytes: u32,
    channels: Vec<Vec<u8>>,
) -> Result<Tensor<crate::Tt, 2>, crate::TtError> {
    let format = match format {
        8 => tt_kernels::bfp::BfpFormat::Bfp8,
        4 => tt_kernels::bfp::BfpFormat::Bfp4,
        2 => tt_kernels::bfp::BfpFormat::Bfp2,
        _ => {
            return Err(crate::TtError::Device(format!(
                "weight format {format} is not BFP2/4/8"
            )))
        }
    };
    if format.slot_bytes() != slot_bytes as u64 {
        return Err(crate::TtError::Device(format!(
            "slot {slot_bytes} is not {}",
            format.slot_bytes()
        )));
    }
    let layout = match layout {
        0 => tt_kernels::tensor::Layout::Interleaved,
        1 => tt_kernels::tensor::Layout::ByColumn,
        _ => return Err(crate::TtError::Device(format!("weight layout {layout}"))),
    };
    tensor_from_bfp_image(
        device,
        tt_kernels::bfp::BfpImage {
            rows,
            cols,
            format,
            layout,
            channels,
        },
    )
}

/// `image` as a logical F32 matrix on `device`, in `image`'s layout.
pub fn tensor_from_bfp_image(
    device: &crate::TtDevice,
    image: tt_kernels::bfp::BfpImage,
) -> Result<Tensor<crate::Tt, 2>, crate::TtError> {
    let native = crate::TtTensor::from_bfp_image(*device, image)?;
    Ok(Tensor::from_primitive(TensorPrimitive::Float(fuse(native))))
}

#[cfg(not(feature = "fusion"))]
fn native_of(tensor: &crate::TtTensor) -> crate::TtTensor {
    tensor.clone()
}

#[cfg(feature = "fusion")]
fn native_of(
    tensor: &burn_fusion::FusionTensor<crate::fusion::TtFusionRuntime>,
) -> crate::TtTensor {
    crate::fusion::resolve_float_tensor(tensor)
}

#[cfg(not(feature = "fusion"))]
fn fuse(tensor: crate::TtTensor) -> crate::TtTensor {
    tensor
}

#[cfg(feature = "fusion")]
fn fuse(tensor: crate::TtTensor) -> burn_fusion::FusionTensor<crate::fusion::TtFusionRuntime> {
    use burn_tensor::TensorMetadata;
    let device = tensor.device;
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    let client = burn_fusion::get_client::<crate::TtBackend>(&device);
    let id = client.register_tensor_handle(crate::TtHandle::Tensor(tensor));
    burn_fusion::Client::change_client_float::<crate::TtBackend>(
        burn_ir::TensorIr {
            id,
            shape,
            dtype,
            status: burn_ir::TensorStatus::ReadWrite,
        },
        client.clone(),
        client,
        burn_fusion::stream::StreamId::current(),
    )
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
