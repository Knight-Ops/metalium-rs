//! The tensor primitives: data on the host, on the device, or both, and the
//! device it belongs to.
//!
//! A tensor is a shared, immutable cell with two lazily filled copies. The host
//! copy is a `burn-flex` tensor, which every op `burn-tt` does not run itself
//! reads. The device copy is a buffer in the chip's GDDR (Phase 9), which the
//! device ops read and write. An op on the device leaves its result there only;
//! the first host op to need it downloads it once, and the first device op to
//! need a host tensor uploads it once. Clones share both copies -- autodiff
//! clones what it keeps for the backward pass, so what the forward pass
//! uploaded, the backward pass finds on the device.
//!
//! A 2-D transpose of a device tensor is a view: the same buffer, read
//! transposed by the device matmul, and by a host op only once downloaded.

use std::sync::{Arc, OnceLock};

use burn_backend::quantization::QuantScheme;
use burn_backend::{DType, QTensorPrimitive, Shape, TensorData, TensorMetadata};
use burn_flex::{FlexQTensor, FlexTensor};

use crate::server::{self, BufferId};
use crate::TtDevice;

/// Float, int and bool tensors alike, as Flex uses one primitive for all three.
#[derive(Clone, Debug)]
pub struct TtTensor {
    pub(crate) cell: Arc<Cell>,
    pub(crate) device: TtDevice,
}

#[derive(Debug)]
pub(crate) struct Cell {
    host: OnceLock<FlexTensor>,
    dram: OnceLock<DramRef>,
    shape: Shape,
    dtype: DType,
}

/// A device buffer, read transposed or not.
#[derive(Clone, Debug)]
pub(crate) struct DramRef {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) transposed: bool,
}

/// A row-major `[rows, cols]` F32 matrix on the device, freed when the last
/// tensor using it goes.
#[derive(Debug)]
pub(crate) struct Buffer {
    pub(crate) id: BufferId,
    pub(crate) device: TtDevice,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    /// For a view: the buffer whose slots it reads. Held, never read: the
    /// parent's slots are freed only once every view of them has gone.
    #[allow(dead_code)]
    pub(crate) parent: Option<Arc<Buffer>>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        server::free(self.device, self.id);
    }
}

impl TtTensor {
    /// A host tensor.
    pub(crate) fn new(inner: FlexTensor, device: TtDevice) -> Self {
        let (shape, dtype) = (inner.shape(), inner.dtype());
        let host = OnceLock::new();
        let _ = host.set(inner);
        TtTensor {
            cell: Arc::new(Cell {
                host,
                dram: OnceLock::new(),
                shape,
                dtype,
            }),
            device,
        }
    }

    /// A device tensor, of `shape` (the buffer's, or its transpose's).
    pub(crate) fn on_device(dram: DramRef, shape: Shape, device: TtDevice) -> Self {
        let cell = OnceLock::new();
        let _ = cell.set(dram);
        TtTensor {
            cell: Arc::new(Cell {
                host: OnceLock::new(),
                dram: cell,
                shape,
                dtype: DType::F32,
            }),
            device,
        }
    }

    /// Was this tensor computed on the device: a device copy and, so far, no
    /// host one? What a residency gate checks of an op's result -- a host
    /// fallback on operands that still had host copies moves no bytes, so
    /// the traffic counters alone cannot tell.
    pub fn computed_on_device(&self) -> bool {
        self.cell.dram.get().is_some() && self.cell.host.get().is_none()
    }

    /// The host copy, downloaded the first time it is needed.
    pub(crate) fn host(&self) -> &FlexTensor {
        self.cell.host.get_or_init(|| {
            let d = self
                .cell
                .dram
                .get()
                .expect("a tensor with no host copy has a device copy");
            if std::env::var_os("TT_TRACE_FALLBACK").is_some() {
                eprintln!(
                    "burn-tt: downloading {:?}{} for a host op\n{}",
                    self.cell.shape.to_vec(),
                    if d.transposed { " (transposed)" } else { "" },
                    std::backtrace::Backtrace::force_capture()
                );
            }
            let (r, c) = (d.buffer.rows, d.buffer.cols);
            let v = server::download(self.device, d.buffer.id, r, c);
            let v = if d.transposed {
                let mut t = vec![0f32; v.len()];
                for i in 0..r {
                    for j in 0..c {
                        t[j * r + i] = v[i * c + j];
                    }
                }
                t
            } else {
                v
            };
            FlexTensor::from_data(TensorData::new(v, self.cell.shape.clone()))
        })
    }

    /// The host copy, owned: taken if this is the only reference to it.
    pub(crate) fn into_host(self) -> FlexTensor {
        match Arc::try_unwrap(self.cell) {
            Ok(cell) if cell.host.get().is_some() => cell.host.into_inner().expect("checked"),
            Ok(cell) => TtTensor {
                cell: Arc::new(cell),
                device: self.device,
            }
            .host()
            .clone(),
            Err(cell) => TtTensor {
                cell,
                device: self.device,
            }
            .host()
            .clone(),
        }
    }

    /// The device copy, if there is one already.
    pub(crate) fn dram(&self) -> Option<&DramRef> {
        self.cell.dram.get()
    }

    /// The device copy, uploaded the first time it is needed. Only for an F32
    /// matrix: callers check [`TtTensor::is_matrix_f32`].
    pub(crate) fn to_dram(&self) -> &DramRef {
        self.cell.dram.get_or_init(|| {
            let dims = self.cell.shape.to_vec();
            let (rows, cols) = (dims[0], dims[1]);
            let values = self
                .host()
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("an F32 tensor reads back as f32");
            let id = server::upload(self.device, values, rows, cols);
            DramRef {
                buffer: Arc::new(Buffer {
                    id,
                    device: self.device,
                    rows,
                    cols,
                    parent: None,
                }),
                transposed: false,
            }
        })
    }

    /// A rank-2 F32 tensor: what the device path takes.
    pub(crate) fn is_matrix_f32(&self) -> bool {
        self.cell.dtype == DType::F32 && self.cell.shape.num_dims() == 2
    }
}

impl TensorMetadata for TtTensor {
    fn dtype(&self) -> DType {
        self.cell.dtype
    }
    fn shape(&self) -> Shape {
        self.cell.shape.clone()
    }
    fn rank(&self) -> usize {
        self.cell.shape.num_dims()
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
