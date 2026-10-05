//! The tensor primitives: data on the host, on the device, or both, and the
//! device it belongs to.
//!
//! A tensor is a shared, immutable cell with two lazily filled copies. The host
//! copy holds owned tensor bytes for input staging and explicit readback. The device copy is a buffer in the chip's GDDR (Phase 9), which the
//! device ops read and write. An op on the device leaves its result there only;
//! the first host op to need it downloads it once, and the first device op to
//! need a host tensor uploads it once. Clones share both copies -- autodiff
//! clones what it keeps for the backward pass, so what the forward pass
//! uploaded, the backward pass finds on the device.
//!
//! A 2-D transpose of a device tensor is a view: the same buffer, read
//! transposed by the device matmul, and by a host op only once downloaded.

use std::sync::{Arc, OnceLock};

use crate::host::HostBuffer;
use burn_backend::quantization::QuantScheme;
use burn_backend::{DType, QTensorPrimitive, Shape, TensorData, TensorMetadata};

use crate::server::{self, BufferId, Elem};
use crate::views::Strided;

/// The device element type a Burn dtype is stored as (`hardware-coverage.md`
/// D3): `F32`; `I32` -- Burn's `IntElem` here -- as its bits; a bool of any
/// store as `0`/`1`. BF16 has a separate two-byte physical storage path.
pub(crate) fn device_elem(dtype: DType) -> Option<Elem> {
    match dtype {
        DType::F32 => Some(Elem::F32),
        DType::I32 => Some(Elem::I32),
        DType::Bool(_) => Some(Elem::Bool),
        _ => None,
    }
}
use crate::TtDevice;

/// Float, int and bool tensors share owned bytes and device buffer references.
#[derive(Clone, Debug)]
pub struct TtTensor {
    pub(crate) cell: Arc<Cell>,
    pub(crate) device: TtDevice,
}

#[derive(Debug)]
pub(crate) struct Cell {
    host: OnceLock<HostBuffer>,
    /// The device copy, uploaded at most once -- shared by the cells of a
    /// reshape that keeps the stored matrix ([`TtTensor::reshaped_host`]),
    /// so a parameter reshaped on every forward pass (a bias) is uploaded
    /// once, by whichever of them meets the device first.
    dram: Arc<OnceLock<DramRef>>,
    /// For a view of another tensor's device buffer that is not a plain
    /// copy of it (`views`): where each element lies. The device copy is
    /// then made from it, on the card where it can be, when an op needs the
    /// tensor as a matrix ([`TtTensor::dram`]).
    strided: Option<Strided>,
    shape: Shape,
    dtype: DType,
}

/// A device buffer, read transposed or not.
#[derive(Clone, Debug)]
pub(crate) struct DramRef {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) transposed: bool,
}

/// A row-major `[rows, cols]` matrix of stored device elements, freed when the last
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
    pub(crate) fn new(inner: HostBuffer, device: TtDevice) -> Self {
        crate::report::made(false);
        let (shape, dtype) = (inner.shape(), inner.dtype());
        let host = OnceLock::new();
        let _ = host.set(inner);
        TtTensor {
            cell: Arc::new(Cell {
                host,
                dram: Arc::new(OnceLock::new()),
                strided: None,
                shape,
                dtype,
            }),
            device,
        }
    }

    /// A device tensor, of `shape` (the buffer's, or its transpose's) and
    /// `dtype` -- one [`device_elem`] stores.
    pub(crate) fn on_device(dram: DramRef, shape: Shape, dtype: DType, device: TtDevice) -> Self {
        debug_assert!(
            dtype == DType::BF16 || device_elem(dtype).is_some(),
            "{dtype:?} is not stored on the device"
        );
        crate::report::made(true);
        let cell = OnceLock::new();
        let _ = cell.set(dram);
        TtTensor {
            cell: Arc::new(Cell {
                host: OnceLock::new(),
                dram: Arc::new(cell),
                strided: None,
                shape,
                dtype,
            }),
            device,
        }
    }

    /// An F32 tensor of `shape` that is the view `v` of a device buffer: a
    /// plain device tensor if the view is one, else a strided view
    /// materialised when an op needs it as a matrix.
    pub(crate) fn view(v: Strided, shape: Shape, dtype: DType, device: TtDevice) -> Self {
        if let Some(d) = v.as_plain(&shape.to_vec()) {
            return Self::on_device(d, shape, dtype, device);
        }
        crate::report::made(true);
        TtTensor {
            cell: Arc::new(Cell {
                host: OnceLock::new(),
                dram: Arc::new(OnceLock::new()),
                strided: Some(v),
                shape,
                dtype,
            }),
            device,
        }
    }

    /// Is this a 2-D transposed view of a device buffer?
    pub(crate) fn is_transposed(&self) -> bool {
        self.cell.dram.get().is_some_and(|d| d.transposed)
    }

    /// Is this a strided view not yet made a matrix?
    pub(crate) fn is_view(&self) -> bool {
        self.cell.strided.is_some() && self.cell.dram.get().is_none()
    }

    /// This tensor's device copy as a view, without materialising anything:
    /// its own view, or a plain view of its device copy. `None` if it has
    /// neither.
    pub(crate) fn as_strided(&self) -> Option<Strided> {
        if let Some(d) = self.cell.dram.get() {
            return Some(Strided::of(d, &self.cell.shape.to_vec()));
        }
        self.cell.strided.clone()
    }

    /// Does this tensor have device storage and, so far, no host copy?
    /// Residency gates check this alongside traffic to distinguish device
    /// results from host input construction and layout packing.
    pub fn computed_on_device(&self) -> bool {
        (self.cell.dram.get().is_some() || self.cell.strided.is_some())
            && self.cell.host.get().is_none()
    }

    /// The host copy, downloaded the first time it is needed.
    pub(crate) fn host(&self) -> &HostBuffer {
        self.cell.host.get_or_init(|| {
            if let (None, Some(v)) = (self.cell.dram.get(), &self.cell.strided) {
                if !self.materialises_on_device(v) {
                    return self.gathered(v);
                }
            }
            let d = self
                .dram()
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
            fn transposed<T: Copy + Default>(v: Vec<T>, t: bool, r: usize, c: usize) -> Vec<T> {
                if !t {
                    return v;
                }
                let mut out = vec![T::default(); v.len()];
                for i in 0..r {
                    for j in 0..c {
                        out[j * r + i] = v[i * c + j];
                    }
                }
                out
            }
            let shape = self.cell.shape.clone();
            let data = if self.cell.dtype == DType::BF16 {
                let bits = server::download_bf16(self.device, d.buffer.id, r, c);
                let bits = transposed(bits, d.transposed, r, c);
                // Construct typed bytes; explicit readback performs no arithmetic.
                let mut data = TensorData::new(bits, shape);
                data.dtype = DType::BF16;
                data
            } else {
                match device_elem(self.cell.dtype) {
                    Some(Elem::F32) => {
                        let v = server::download(self.device, d.buffer.id, r, c);
                        TensorData::new(transposed(v, d.transposed, r, c), shape)
                    }
                    Some(elem) => {
                        let v = server::download_bits(self.device, d.buffer.id, r, c);
                        let v = transposed(v, d.transposed, r, c);
                        if elem == Elem::I32 {
                            TensorData::new(
                                v.into_iter().map(|b| b as i32).collect::<Vec<_>>(),
                                shape,
                            )
                        } else {
                            TensorData::new(
                                v.into_iter().map(|b| b != 0).collect::<Vec<_>>(),
                                shape,
                            )
                            .convert_dtype(self.cell.dtype)
                        }
                    }
                    None => unreachable!("only a stored dtype has a device copy"),
                }
            };
            HostBuffer::from_data(data)
        })
    }

    /// `self` reshaped on the host to `shape`, which is stored as the same
    /// matrix ([`stored_dims`]): the new tensor holds `host` (the host copy
    /// reshaped) and shares `self`'s device-copy slot, so one upload serves
    /// both. Only for a tensor with no device copy yet, or an untransposed
    /// one -- a transposed view's buffer is not the reshape's.
    pub(crate) fn reshaped_host(&self, host: HostBuffer, shape: Shape) -> TtTensor {
        debug_assert_eq!(stored_dims(&shape.to_vec()), self.stored());
        debug_assert!(self.dram().is_none_or(|d| !d.transposed));
        crate::report::made(false);
        let h = OnceLock::new();
        let _ = h.set(host);
        TtTensor {
            cell: Arc::new(Cell {
                host: h,
                dram: self.cell.dram.clone(),
                strided: None,
                shape,
                dtype: self.cell.dtype,
            }),
            device: self.device,
        }
    }

    /// The host copy, owned: taken if this is the only reference to it.
    pub(crate) fn into_host(self) -> HostBuffer {
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

    /// The device copy, if there is one already -- or, for a strided view,
    /// made now ([`TtTensor::to_dram`]).
    pub(crate) fn dram(&self) -> Option<&DramRef> {
        match (self.cell.dram.get(), &self.cell.strided) {
            (Some(d), _) => Some(d),
            (None, Some(_)) => Some(self.to_dram()),
            (None, None) => None,
        }
    }

    /// Can this view be materialized by native copies or word repacking?
    fn materialises_on_device(&self, _v: &Strided) -> bool {
        self.is_storable() && server::supports_dram(self.device)
    }

    /// A strided view's elements, on the host: its source downloaded and
    /// read through the strides.
    fn gathered(&self, v: &Strided) -> HostBuffer {
        let b = &v.src.buffer;
        if std::env::var_os("TT_TRACE_FALLBACK").is_some() {
            eprintln!(
                "burn-tt: downloading [{}, {}] to read a {:?} view of it on the host\n{}",
                b.rows,
                b.cols,
                self.cell.shape.to_vec(),
                std::backtrace::Backtrace::force_capture()
            );
        }
        let src = server::download(self.device, b.id, b.rows, b.cols);
        let shape = self.cell.shape.to_vec();
        let n: usize = shape.iter().product();
        let values: Vec<f32> = (0..n)
            .map(|f| {
                let [r, c] = v.at(&shape, f);
                src[r * b.cols + c]
            })
            .collect();
        HostBuffer::from_data(TensorData::new(values, shape))
    }

    /// The device copy, uploaded the first time it is needed, as the matrix
    /// [`stored_dims`] gives. Only for an F32 tensor of rank one or more:
    /// callers check [`TtTensor::is_stored_f32`] (or the stricter
    /// [`TtTensor::is_matrix_f32`]).
    pub(crate) fn to_dram(&self) -> &DramRef {
        self.cell.dram.get_or_init(|| {
            let [rows, cols] =
                stored_dims(&self.cell.shape.to_vec()).expect("callers check the rank");
            if let Some(v) = &self.cell.strided {
                let shape = self.cell.shape.to_vec();
                if let Some(moves) = v.tile_moves(&shape, &shape) {
                    let (id, dims) =
                        server::copy_blocks(self.device, v.src.buffer.id, moves, [rows, cols]);
                    return DramRef {
                        buffer: Arc::new(Buffer {
                            id,
                            device: self.device,
                            rows: dims[0],
                            cols: dims[1],
                            parent: None,
                        }),
                        transposed: false,
                    };
                }
                let sources = (0..self.cell.shape.num_elements())
                    .map(|f| v.at(&shape, f))
                    .collect();
                let (id, dims) =
                    server::repack(self.device, v.src.buffer.id, sources, [rows, cols]);
                return DramRef {
                    buffer: Arc::new(Buffer {
                        id,
                        device: self.device,
                        rows: dims[0],
                        cols: dims[1],
                        parent: None,
                    }),
                    transposed: false,
                };
            }
            let data = self.host().clone().into_data();
            let id = if self.cell.dtype == DType::BF16 {
                let mut data = data;
                data.dtype = DType::U16;
                let bits = data.to_vec::<u16>().expect("BF16 storage bits");
                server::upload_bf16(self.device, bits, rows, cols)
            } else {
                match device_elem(self.cell.dtype) {
                    Some(Elem::F32) => {
                        let values = data
                            .to_vec::<f32>()
                            .expect("an F32 tensor reads back as f32");
                        server::upload(self.device, values, rows, cols)
                    }
                    Some(Elem::I32) => {
                        let v = data
                            .to_vec::<i32>()
                            .expect("an I32 tensor reads back as i32");
                        let bits = v.into_iter().map(|x| x as u32).collect();
                        server::upload_bits(self.device, bits, rows, cols, Elem::I32)
                    }
                    Some(Elem::Bool) => {
                        let v = data
                            .convert_dtype(DType::Bool(burn_backend::BoolStore::Native))
                            .to_vec::<bool>()
                            .expect("a bool tensor reads back as bool");
                        let bits = v.into_iter().map(u32::from).collect();
                        server::upload_bits(self.device, bits, rows, cols, Elem::Bool)
                    }
                    None => panic!(
                        "callers check the dtype: {:?} is not stored",
                        self.cell.dtype
                    ),
                }
            };
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

    /// An F32 tensor the device can hold: any rank but zero, not empty, as the
    /// matrix [`stored_dims`] gives. What the element-wise path takes.
    pub(crate) fn is_stored_f32(&self) -> bool {
        self.cell.dtype == DType::F32
            && stored_dims(&self.cell.shape.to_vec()).is_some_and(|[r, c]| r > 0 && c > 0)
    }

    /// A tensor of any dtype the device stores ([`device_elem`]), of any rank
    /// but zero and not empty: what views and residency take.
    pub(crate) fn is_storable(&self) -> bool {
        (self.cell.dtype == DType::BF16 || device_elem(self.cell.dtype).is_some())
            && stored_dims(&self.cell.shape.to_vec()).is_some_and(|[r, c]| r > 0 && c > 0)
    }

    /// A rank-2 tensor of a stored dtype.
    pub(crate) fn is_storable_matrix(&self) -> bool {
        (self.cell.dtype == DType::BF16 || device_elem(self.cell.dtype).is_some())
            && self.cell.shape.num_dims() == 2
    }

    /// The device element type this tensor is (or would be) stored as.
    pub(crate) fn elem(&self) -> Option<Elem> {
        device_elem(self.cell.dtype)
    }

    /// The matrix this tensor is (or would be) stored as on the device.
    pub(crate) fn stored(&self) -> Option<[usize; 2]> {
        stored_dims(&self.cell.shape.to_vec())
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

/// Metadata required by Burn's quantized primitive interface.
/// Quantized construction and computation are explicitly unsupported.
#[derive(Clone, Debug)]
pub struct TtQTensor {
    pub(crate) shape: Shape,
    pub(crate) scheme: QuantScheme,
    pub(crate) device: TtDevice,
}

impl TensorMetadata for TtQTensor {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }
    fn shape(&self) -> Shape {
        self.shape.clone()
    }
    fn rank(&self) -> usize {
        self.shape.num_dims()
    }
}

impl QTensorPrimitive for TtQTensor {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }
    fn default_scheme() -> QuantScheme {
        QuantScheme::default()
    }
}

/// How a tensor of `shape` is laid out on the device: the matrix
/// `[product of the leading dimensions, last dimension]`, rank 1 as one row.
/// Row-major order is the same either way, so a reshape that keeps this
/// matrix is a view, and an element-wise op between two tensors of one
/// shape is the op on their matrices. `None` for a scalar (rank 0).
pub(crate) fn stored_dims(shape: &[usize]) -> Option<[usize; 2]> {
    let (&last, leading) = shape.split_last()?;
    Some([leading.iter().product(), last])
}

#[cfg(test)]
mod tests {
    use super::stored_dims;

    #[test]
    fn a_tensor_is_stored_as_its_leading_dimensions_by_its_last() {
        assert_eq!(stored_dims(&[]), None);
        assert_eq!(stored_dims(&[128]), Some([1, 128]));
        assert_eq!(stored_dims(&[64, 10]), Some([64, 10]));
        assert_eq!(stored_dims(&[2, 3, 4]), Some([6, 4]));
        assert_eq!(stored_dims(&[2, 0, 4]), Some([0, 4]));
    }
}
