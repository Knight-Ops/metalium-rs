//! One server thread per attached device, owning its hardware.
//!
//! Burn requires tensors and devices to be `Send + Sync + 'static` and calls
//! ops from whatever thread it likes. The hardware is neither: a
//! `tt_device::Device` takes `&mut self` for everything, `Device<Kmd>` is not
//! `Sync`, and the simulator is a `!Send` process-wide singleton. So each
//! device's hardware lives on a thread of its own, created by [`attach`], and
//! ops send it jobs and block on the answer. The thread is also where the
//! hardware is *created*: [`attach`] runs the caller's factory there, which is
//! what lets a `!Send` simulator serve at all.
//!
//! Dropping the [`AttachGuard`] closes the job queue, waits for the thread, and
//! so drops the hardware -- which for silicon returns the chip to idle
//! (`tt_device::Device`'s `Drop`). One `Device` per chip applies here as
//! everywhere: attaching a chip twice is refused.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use tt_kernels::tensor::{Block, BlockMove};

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::matrix_eltwise::{ElementwiseMode, MatrixEltwiseOp, SrcPrecision};
use tt_kernels::session::{Session, SessionError, TileChoice};

use crate::TtDevice;
pub use tt_kernels::tensor::Elem;

/// What [`Engine::pow`] raises to: a buffer of the base's shape, a scalar, or
/// an `I32` buffer.
#[derive(Copy, Clone, Debug)]
pub enum PowArg {
    Tensor(BufferId),
    Scalar(f32),
    Int(BufferId),
}

/// What a device can do for the backend, on its server thread.
pub trait Engine {
    fn elementwise_mode(&self) -> ElementwiseMode {
        ElementwiseMode::Sfpu
    }
    /// `A[m, k] @ B[k, n]`, row-major.
    fn matmul(&mut self, a: &[f32], b: &[f32], mkn: [usize; 3]) -> Result<Vec<f32>, EngineError>;

    /// Can this engine keep tensors on the device (Phase 9)? If not, every
    /// tensor stays on the host; native `matmul` and `full_reduce` may stage
    /// through L1 if the engine implements them.
    fn supports_dram(&self) -> bool {
        false
    }
    /// A full F32 reduction computed by native kernels, staging through L1
    /// when this engine has no resident tensor storage. Never host arithmetic.
    fn full_reduce(
        &mut self,
        _values: &[f32],
        _dims: [usize; 2],
        _mean: bool,
    ) -> Result<f32, EngineError> {
        Err(EngineError(
            "native full reduction is unsupported by this engine".into(),
        ))
    }
    /// Put a row-major `[rows, cols]` matrix on the device.
    fn upload(
        &mut self,
        _values: &[f32],
        _rows: usize,
        _cols: usize,
    ) -> Result<BufferId, EngineError> {
        Err(unsupported())
    }
    /// Materialize operation geometry from replayable mover immediates.
    fn metadata(
        &mut self,
        _bits: &[u32],
        _dims: [usize; 2],
        _elem: Elem,
    ) -> Result<BufferId, EngineError> {
        Err(unsupported())
    }
    /// Read one back, row-major.
    fn download(&mut self, _id: BufferId) -> Result<Vec<f32>, EngineError> {
        Err(unsupported())
    }
    fn upload_bf16(
        &mut self,
        _bits: &[u16],
        _rows: usize,
        _cols: usize,
    ) -> Result<BufferId, EngineError> {
        Err(unsupported())
    }
    fn download_bf16(&mut self, _id: BufferId) -> Result<Vec<u16>, EngineError> {
        Err(unsupported())
    }
    fn pool_bf16(
        &mut self,
        _a: BufferId,
        _windows: &[Vec<[usize; 2]>],
        _divisors: &[usize],
        _dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    fn matmul_bf16(
        &mut self,
        _a: BufferId,
        _b: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// Convert between resident F32 and physical BF16 storage on Tensix.
    fn cast_float(
        &mut self,
        _id: BufferId,
        _bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// Explicit physical BFP storage conversion. None widens to F32.
    fn cast_bfp(
        &mut self,
        _id: BufferId,
        _format: Option<tt_kernels::bfp::BfpFormat>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    fn matmul_bfp(
        &mut self,
        _a: BufferId,
        _b: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// Put a row-major `[rows, cols]` matrix of `elem` datums on the device, as
    /// their bits (`hardware-coverage.md` D3): an `I32`'s two's complement, a
    /// `Bool`'s `0` or `1`.
    fn upload_bits(
        &mut self,
        _bits: &[u32],
        _rows: usize,
        _cols: usize,
        _elem: Elem,
    ) -> Result<BufferId, EngineError> {
        Err(unsupported())
    }
    /// Read any buffer back as its datums' bits, row-major.
    fn download_bits(&mut self, _id: BufferId) -> Result<Vec<u32>, EngineError> {
        Err(unsupported())
    }
    /// Forget one.
    fn free(&mut self, _id: BufferId) {}
    /// `op(A) @ op(B)`, each operand transposed if asked, result left on the
    /// device, as `(id, [rows, cols])`.
    fn matmul_dram(
        &mut self,
        _a: BufferId,
        _a_transposed: bool,
        _b: BufferId,
        _b_transposed: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// `op(A_i) @ op(B_i)` for each pair of blocks of `a` and `b` in `items`,
    /// every product `mkn`, stacked into one `[items.len() m, n]` result left
    /// on the device (`Session::matmul_dram_batched`).
    fn matmul_dram_batched(
        &mut self,
        _a: BufferId,
        _b: BufferId,
        _items: &[(Block, Block)],
        _mkn: [usize; 3],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// A new `dims` buffer assembled from blocks of `a`
    /// (`Session::copy_blocks`).
    /// Bit-preserving native repack from source coordinates in output order.
    fn repack(
        &mut self,
        _a: BufferId,
        _sources: &[[usize; 2]],
        _dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    fn copy_blocks(
        &mut self,
        _a: BufferId,
        _moves: &[BlockMove],
        _dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    fn zeros_dram(
        &mut self,
        _dims: [usize; 2],
        _elem: Elem,
        _bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    fn gather_indexed(
        &mut self,
        _input: BufferId,
        _indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    fn repack_many(
        &mut self,
        _inputs: &[BufferId],
        _sources: &[(usize, [usize; 2])],
        _dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// Rows of `sources` gathered into a new buffer
    /// (`Session::gather_rows`): an embedding's lookup.
    fn gather_rows(
        &mut self,
        _sources: &[BufferId],
        _rows: &[(usize, usize)],
        _cols: usize,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// `t` with `value`'s rows added to the rows `indices` name, in order
    /// (`Session::rows_add`): an embedding's gradient.
    fn rows_add(
        &mut self,
        _t: BufferId,
        _indices: &[usize],
        _value: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// The sum over rows of `a`, `[1, cols]`, left on the device.
    fn sum_rows(&mut self, _a: BufferId) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// `a` reduced over `axis` by `op`, left on the device
    /// (`tt_kernels::session::Session::reduce`).
    fn reduce(
        &mut self,
        _a: BufferId,
        _op: tt_kernels::sfpu::reduce::ReduceOp,
        _axis: tt_kernels::sfpu::reduce::Axis,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// Inclusive scan down matrix rows, keeping all prefixes resident.
    fn scan(
        &mut self,
        _a: BufferId,
        _op: tt_kernels::sfpu::scan::ScanOp,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// Rows `[first, first + rows)` of `a` as a view: no copy. The caller
    /// keeps `a` alive while the view is.
    fn slice_rows(
        &mut self,
        _a: BufferId,
        _first: usize,
        _rows: usize,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// Element-wise `a (kind) b` or `a (kind) scalar` (`tt_kernels::kind`),
    /// result left on the device.
    fn eltwise(
        &mut self,
        _kind: u32,
        _scalar: f32,
        _a: BufferId,
        _b: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// [`Engine::eltwise`] with the whole op -- both its scalars -- and a
    /// ternary op's third operand. An engine without its own forwards what
    /// the plain form carries.
    fn eltwise_op(
        &mut self,
        op: tt_kernels::tensor::Eltwise,
        a: BufferId,
        b: Option<BufferId>,
        c: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        if op.scalar2 != 0.0 || c.is_some() {
            return Err(unsupported());
        }
        self.eltwise(op.kind, op.scalar, a, b)
    }
    /// `x^y` as `powf` (`tt_kernels::session::Session::pow`), result on the
    /// device.
    fn pow(&mut self, _x: BufferId, _y: PowArg) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// Everything the engine's device has moved across its transport so far
    /// (`tt_device::Device::traffic`): tensors, descriptors, programs and
    /// polls alike. `None` if the engine has no single device to ask.
    fn device_traffic(&mut self) -> Option<tt_device::Traffic> {
        None
    }
    fn mesh_execution(&mut self) -> Option<tt_kernels::shard::FabricExecution> {
        None
    }
    /// Start capturing a trace (`tt_kernels::trace`; [`crate::Trace`]).
    fn begin_trace(&mut self) -> Result<(), EngineError> {
        Err(no_traces())
    }
    /// End it: the trace's number.
    fn end_trace(&mut self) -> Result<u64, EngineError> {
        Err(no_traces())
    }
    /// Overwrite `input`'s values, replay `trace`, and read `output` back:
    /// one round trip.
    fn run_trace(
        &mut self,
        _trace: u64,
        _input: BufferId,
        _values: &[f32],
        _output: BufferId,
    ) -> Result<TraceRun, EngineError> {
        Err(no_traces())
    }
    /// Give a trace back.
    fn release_trace(&mut self, _trace: u64) {}
    /// Copy `src` into `dst` on the device without allocating a new buffer.
    fn copy_into(&mut self, _src: BufferId, _dst: BufferId) -> Result<(), EngineError> {
        Err(unsupported())
    }
    /// Overwrite multiple inputs, replay `trace`, and read multiple outputs back.
    fn run_generic_trace(
        &mut self,
        _trace: u64,
        _inputs: &[(BufferId, InputPayload)],
        _outputs: &[(BufferId, OutputKind)],
    ) -> Result<GenericTraceRun, EngineError> {
        Err(no_traces())
    }

    // lane:t1_intbool (Engine trait): add this lane's methods below this line only.

    // lane:t2_index (Engine trait): add this lane's methods below this line only.

    // lane:t3_scan (Engine trait): add this lane's methods below this line only.

    // lane:t4_rem (Engine trait): add this lane's methods below this line only.

    // lane:t5_sort (Engine trait): add this lane's methods below this line only.
    /// Sort every problem of `a`, a plane matrix (`tt_kernels::sfpu::sort`),
    /// along its axis positions: the sorted keys and, if `spec.indices`, the
    /// original indices. Both stay on the device.
    fn sort_planes(
        &mut self,
        _a: BufferId,
        _spec: tt_kernels::sfpu::sort::Spec,
    ) -> Result<SortedBuffers, EngineError> {
        Err(unsupported())
    }

    // lane:t6_random (Engine trait): add this lane's methods below this line only.

    // lane:t7_dtype (Engine trait): add this lane's methods below this line only.

    // lane:t8_mathmode (Engine trait): add this lane's methods below this line only.

    // lane:t9_mesh (Engine trait): add this lane's methods below this line only.
}

/// Input payload to write into a trace's input buffer before replay.
#[derive(Clone, Debug)]
pub enum InputPayload {
    F32(Vec<f32>),
    Bits(Vec<u32>),
}

/// Output data kind for trace readback.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OutputKind {
    F32,
    Bits,
}

/// Output payload downloaded from a trace's output buffer after replay.
#[derive(Clone, Debug)]
pub enum OutputPayload {
    F32(Vec<f32>),
    Bits(Vec<u32>),
}

impl OutputPayload {
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            OutputPayload::F32(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_bits(&self) -> Option<&[u32]> {
        match self {
            OutputPayload::Bits(v) => Some(v),
            _ => None,
        }
    }
    pub fn into_f32(self) -> Option<Vec<f32>> {
        match self {
            OutputPayload::F32(v) => Some(v),
            _ => None,
        }
    }
}

/// Results and timings of a multi-tensor generic trace run.
#[derive(Clone, Debug, Default)]
pub struct GenericTraceRun {
    pub outputs: Vec<OutputPayload>,
    pub write: std::time::Duration,
    pub replay: std::time::Duration,
    pub read: std::time::Duration,
}

/// One [`crate::Trace::run`]: the output, and where its time went on the
/// device's side -- the input written from the host, the replay to its end,
/// and the output read back -- each a different cost (the first and last are
/// PCIe's, the middle the card's).
#[derive(Clone, Debug, Default)]
pub struct TraceRun {
    pub output: Vec<f32>,
    pub write: std::time::Duration,
    pub replay: std::time::Duration,
    pub read: std::time::Duration,
}

fn no_traces() -> EngineError {
    EngineError("this engine does not capture traces".into())
}

fn unsupported() -> EngineError {
    EngineError("this engine keeps no tensors on the device".into())
}

/// A tensor kept on the device: by the engine's own numbering inside an
/// [`Engine`]; by a process-wide number, never reused, on the caller's side
/// of the server (`server::Ids` translates).
pub type BufferId = u64;

/// The device-resident tensors of one engine: a [`Session`]'s `DramTensor`s by
/// id. The engines built on a `Session` -- [`KmdEngine`], and the simulator's in
/// `tt-tests` -- forward their DRAM methods here.
#[derive(Default)]
pub struct DramBuffers {
    elementwise: ElementwiseMode,
    next: BufferId,
    live: HashMap<BufferId, tt_kernels::tensor::DramTensor>,
    bf16: HashMap<BufferId, tt_kernels::bf16::Bf16Tensor>,
    bfp: HashMap<BufferId, tt_kernels::bfp::BfpTensor>,
    next_trace: u64,
    traces: HashMap<u64, tt_kernels::trace::TraceId>,
}

impl DramBuffers {
    /// Configure before attachment. There is no setter on an attached engine.
    pub fn with_elementwise_mode(mut self, mode: ElementwiseMode) -> Self {
        self.elementwise = mode;
        self
    }
    pub fn elementwise_mode(&self) -> ElementwiseMode {
        self.elementwise
    }

    fn matrix_eltwise<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        op: tt_kernels::tensor::Eltwise,
        a: BufferId,
        b: Option<BufferId>,
        precision: SrcPrecision,
        fidelity: Fidelity,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        use tt_kernels::kind;
        let operation = match op.kind {
            kind::ADD | kind::ADD_SCALAR => MatrixEltwiseOp::Add,
            kind::SUB => MatrixEltwiseOp::Sub,
            kind::MUL | kind::MUL_SCALAR => MatrixEltwiseOp::Mul,
            _ => return Err(unsupported()),
        };
        let error = |e: tt_kernels::tensor::TensorError| EngineError(e.to_string());
        let packed = self.bf16.contains_key(&a);
        let scalar = if b.is_none() {
            Some(
                s.metadata(&[op.scalar.to_bits()], [1, 1], Elem::F32)
                    .map_err(error)?,
            )
        } else {
            None
        };
        let packed_scalar = if packed {
            scalar
                .as_ref()
                .map(|t| s.bf16_from_f32(t).map_err(error))
                .transpose()
        } else {
            Ok(None)
        };
        let result = (|| {
            let packed_scalar = packed_scalar
                .as_ref()
                .map_err(|e| EngineError(e.0.clone()))?;
            if packed {
                let ta = self.bf16.get(&a).unwrap();
                let tb = if let Some(b) = b {
                    self.bf16.get(&b).ok_or_else(unsupported)?
                } else {
                    packed_scalar.as_ref().unwrap()
                };
                s.matrix_eltwise_bf16(operation, ta, tb, fidelity)
                    .map_err(error)
            } else {
                let ta = self.get(a)?;
                let tb = if let Some(b) = b {
                    self.get(b)?
                } else {
                    scalar.as_ref().unwrap()
                };
                s.matrix_eltwise(operation, ta, tb, precision, fidelity)
                    .map_err(error)
            }
        })();
        if let Ok(Some(t)) = packed_scalar {
            let _ = s.free_bf16(t);
        }
        if let Some(t) = scalar {
            let _ = s.free(t);
        }
        Ok(self.insert(result?))
    }

    pub fn pool_bf16<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        windows: &[Vec<[usize; 2]>],
        divisors: &[usize],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let input = self
            .bf16
            .get(&a)
            .ok_or_else(|| EngineError(format!("no BF16 buffer {a}")))?;
        let out = s
            .bf16_pool_windows(input, windows, divisors, dims)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(out))
    }
    pub fn matmul_bf16<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        b: BufferId,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let ta = self
            .bf16
            .get(&a)
            .ok_or_else(|| EngineError(format!("no BF16 buffer {a}")))?;
        let tb = self
            .bf16
            .get(&b)
            .ok_or_else(|| EngineError(format!("no BF16 buffer {b}")))?;
        let out = s
            .matmul_bf16(ta, tb, fidelity, budget)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(out))
    }
    pub fn upload_bf16<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        bits: &[u16],
        rows: usize,
        cols: usize,
    ) -> Result<BufferId, EngineError> {
        let t = s
            .upload_bf16(bits, rows, cols)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert_bf16(t).0)
    }

    pub fn download_bf16<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
    ) -> Result<Vec<u16>, EngineError> {
        let t = self
            .bf16
            .get(&id)
            .ok_or_else(|| EngineError(format!("no BF16 buffer {id}")))?;
        s.download_bf16(t).map_err(|e| EngineError(e.to_string()))
    }

    pub fn cast_float<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
        bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        if bf16 {
            let t = s
                .bf16_from_f32(self.get(id)?)
                .map_err(|e| EngineError(e.to_string()))?;
            Ok(self.insert_bf16(t))
        } else {
            let t = self
                .bf16
                .get(&id)
                .ok_or_else(|| EngineError(format!("no BF16 buffer {id}")))?;
            let t = s.bf16_to_f32(t).map_err(|e| EngineError(e.to_string()))?;
            Ok(self.insert(t))
        }
    }

    pub fn cast_bfp<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
        format: Option<tt_kernels::bfp::BfpFormat>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let error = |e: tt_kernels::tensor::TensorError| EngineError(e.to_string());
        if let Some(format) = format {
            let t = s.bfp_from_f32(self.get(id)?, format).map_err(error)?;
            let dims = [t.rows(), t.cols()];
            self.next += 1;
            self.bfp.insert(self.next, t);
            Ok((self.next, dims))
        } else {
            let t = self
                .bfp
                .get(&id)
                .ok_or_else(|| EngineError(format!("no BFP buffer {id}")))?;
            let t = s.bfp_to_f32(t).map_err(error)?;
            Ok(self.insert(t))
        }
    }

    pub fn matmul_bfp<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        b: BufferId,
        fidelity: Fidelity,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let a = self.bfp.get(&a).ok_or_else(unsupported)?;
        let b = self.bfp.get(&b).ok_or_else(unsupported)?;
        let out = s
            .bfp_matmul(a, false, b, false, fidelity)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(out))
    }
    fn insert_bf16(&mut self, t: tt_kernels::bf16::Bf16Tensor) -> (BufferId, [usize; 2]) {
        let dims = [t.rows, t.cols];
        self.next += 1;
        self.bf16.insert(self.next, t);
        (self.next, dims)
    }
    pub fn begin_trace<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
    ) -> Result<(), EngineError> {
        s.begin_trace().map_err(|e| EngineError(e.to_string()))
    }

    pub fn end_trace<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
    ) -> Result<u64, EngineError> {
        let id = s.end_trace().map_err(|e| EngineError(e.to_string()))?;
        self.next_trace += 1;
        self.traces.insert(self.next_trace, id);
        Ok(self.next_trace)
    }

    pub fn run_trace<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        trace: u64,
        input: BufferId,
        values: &[f32],
        output: BufferId,
    ) -> Result<TraceRun, EngineError> {
        use std::time::Instant;
        let id = *self
            .traces
            .get(&trace)
            .ok_or_else(|| EngineError(format!("no trace {trace}")))?;
        let e = |e: tt_kernels::tensor::TensorError| EngineError(e.to_string());
        // The write waits for what was queued first: that wait is not the
        // write's.
        s.sync().map_err(e)?;
        let t0 = Instant::now();
        s.write(self.get(input)?, values).map_err(e)?;
        let t1 = Instant::now();
        s.replay(id).map_err(e)?;
        s.sync().map_err(e)?;
        let t2 = Instant::now();
        let output = self.download(s, output)?;
        Ok(TraceRun {
            output,
            write: t1 - t0,
            replay: t2 - t1,
            read: t2.elapsed(),
        })
    }

    pub fn release_trace<T: tt_device::Transport>(&mut self, s: &mut Session<T>, trace: u64) {
        if let Some(id) = self.traces.remove(&trace) {
            let _ = s.release_trace(id);
        }
    }

    pub fn copy_into<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        src: BufferId,
        dst: BufferId,
    ) -> Result<(), EngineError> {
        if let Some(src_t) = self.bfp.get(&src) {
            let dst_t = self
                .bfp
                .get(&dst)
                .ok_or_else(|| EngineError("copy_into storage formats differ".into()))?;
            return s
                .copy_into_bfp(src_t, dst_t)
                .map_err(|e| EngineError(e.to_string()));
        }
        if self.bfp.contains_key(&dst) {
            return Err(EngineError("copy_into storage formats differ".into()));
        }
        if let Some(src_t) = self.bf16.get(&src) {
            let dst_t = self
                .bf16
                .get(&dst)
                .ok_or_else(|| EngineError("copy_into storage formats differ".into()))?;
            return s
                .copy_into_bf16(src_t, dst_t)
                .map_err(|e| EngineError(e.to_string()));
        }
        if self.bf16.contains_key(&dst) {
            return Err(EngineError("copy_into storage formats differ".into()));
        }
        let src_t = self.get(src)?;
        let dst_t = self.get(dst)?;
        s.copy_into(src_t, dst_t)
            .map_err(|e| EngineError(e.to_string()))
    }

    pub fn run_generic_trace<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        trace: u64,
        inputs: &[(BufferId, InputPayload)],
        outputs: &[(BufferId, OutputKind)],
    ) -> Result<GenericTraceRun, EngineError> {
        use std::time::Instant;
        let id = *self
            .traces
            .get(&trace)
            .ok_or_else(|| EngineError(format!("no trace {trace}")))?;
        let e = |e: tt_kernels::tensor::TensorError| EngineError(e.to_string());
        s.sync().map_err(e)?;
        let t0 = Instant::now();
        for (input_id, payload) in inputs {
            let t = self.get(*input_id)?;
            match payload {
                InputPayload::F32(v) => s.write(t, v).map_err(e)?,
                InputPayload::Bits(v) => s.write_bits(t, v).map_err(e)?,
            }
        }
        let t1 = Instant::now();
        s.replay(id).map_err(e)?;
        s.sync().map_err(e)?;
        let t2 = Instant::now();
        let mut out_data = Vec::with_capacity(outputs.len());
        for &(output_id, kind) in outputs {
            match kind {
                OutputKind::F32 => {
                    let data = self.download(s, output_id)?;
                    out_data.push(OutputPayload::F32(data));
                }
                OutputKind::Bits => {
                    let data = s.download_bits(self.get(output_id)?).map_err(e)?;
                    out_data.push(OutputPayload::Bits(data));
                }
            }
        }
        Ok(GenericTraceRun {
            outputs: out_data,
            write: t1 - t0,
            replay: t2 - t1,
            read: t2.elapsed(),
        })
    }

    pub fn upload<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<BufferId, EngineError> {
        let t = s
            .upload(values, rows, cols)
            .map_err(|e| EngineError(e.to_string()))?;
        self.next += 1;
        self.live.insert(self.next, t);
        Ok(self.next)
    }

    pub fn metadata<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        bits: &[u32],
        dims: [usize; 2],
        elem: Elem,
    ) -> Result<BufferId, EngineError> {
        let t = s
            .metadata(bits, dims, elem)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(t).0)
    }

    pub fn download<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
    ) -> Result<Vec<f32>, EngineError> {
        if let Some(t) = self.bfp.get(&id) {
            return s.download_bfp(t).map_err(|e| EngineError(e.to_string()));
        }
        let t = self.get(id)?;
        s.download(t).map_err(|e| EngineError(e.to_string()))
    }

    pub fn pow<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        x: BufferId,
        y: PowArg,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        use tt_kernels::session::PowExponent;
        let tx = self.get(x)?.clone();
        let ty = match y {
            PowArg::Tensor(b) | PowArg::Int(b) => Some(self.get(b)?.clone()),
            PowArg::Scalar(_) => None,
        };
        let exp = match (y, &ty) {
            (PowArg::Tensor(_), Some(t)) => PowExponent::Tensor(t),
            (PowArg::Int(_), Some(t)) => PowExponent::Int(t),
            (PowArg::Scalar(v), _) => PowExponent::Scalar(v),
            _ => unreachable!(),
        };
        let c = s.pow(&tx, exp).map_err(|e| EngineError(e.to_string()))?;
        let dims = [c.rows, c.cols];
        self.next += 1;
        self.live.insert(self.next, c);
        Ok((self.next, dims))
    }

    pub fn upload_bits<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        bits: &[u32],
        rows: usize,
        cols: usize,
        elem: Elem,
    ) -> Result<BufferId, EngineError> {
        let t = s
            .upload_bits(bits, rows, cols, elem)
            .map_err(|e| EngineError(e.to_string()))?;
        self.next += 1;
        self.live.insert(self.next, t);
        Ok(self.next)
    }

    pub fn download_bits<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
    ) -> Result<Vec<u32>, EngineError> {
        let t = self.get(id)?;
        s.download_bits(t).map_err(|e| EngineError(e.to_string()))
    }

    pub fn free<T: tt_device::Transport>(&mut self, s: &mut Session<T>, id: BufferId) {
        if let Some(t) = self.live.remove(&id) {
            let _ = s.free(t);
        }
        if let Some(t) = self.bfp.remove(&id) {
            let _ = s.free_bfp(t);
        }
        if let Some(t) = self.bf16.remove(&id) {
            let _ = s.free_bf16(t);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        a_transposed: bool,
        b: BufferId,
        b_transposed: bool,
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let (ta, tb) = (self.get(a)?.clone(), self.get(b)?.clone());
        let c = s
            .matmul_dram(
                &ta,
                a_transposed,
                &tb,
                b_transposed,
                route,
                fidelity,
                budget,
            )
            .map_err(|e| EngineError(e.to_string()))?;
        let dims = [c.rows, c.cols];
        self.next += 1;
        self.live.insert(self.next, c);
        Ok((self.next, dims))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_batched<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        b: BufferId,
        items: &[(Block, Block)],
        mkn: [usize; 3],
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let (ta, tb) = (self.get(a)?.clone(), self.get(b)?.clone());
        let c = s
            .matmul_dram_batched(&ta, &tb, items, mkn, route, fidelity, budget)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(c))
    }

    pub fn gather_rows<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        sources: &[BufferId],
        rows: &[(usize, usize)],
        cols: usize,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let ts = sources
            .iter()
            .map(|&id| self.get(id).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let refs: Vec<_> = ts.iter().collect();
        let c = s
            .gather_rows(&refs, rows, cols)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(c))
    }

    pub fn rows_add<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        t: BufferId,
        indices: &[usize],
        value: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let (tt, tv) = (self.get(t)?.clone(), self.get(value)?.clone());
        let c = s
            .rows_add(&tt, indices, &tv)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(c))
    }

    pub fn repack<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        sources: &[[usize; 2]],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        if let Some(t) = self.bf16.get(&a) {
            let c = s
                .repack_bf16(t, sources, dims)
                .map_err(|e| EngineError(e.to_string()))?;
            return Ok(self.insert_bf16(c));
        }
        let ta = self.get(a)?.clone();
        let c = s
            .repack(&ta, sources, dims)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(c))
    }

    pub fn zeros<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        dims: [usize; 2],
        elem: Elem,
        bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        if bf16 {
            let out = s.zeros_bf16(dims).map_err(|e| EngineError(e.to_string()))?;
            Ok(self.insert_bf16(out))
        } else {
            let count = dims[0]
                .checked_mul(dims[1])
                .ok_or_else(|| EngineError("zero shape overflow".into()))?;
            let out = s
                .metadata(&vec![0; count], dims, elem)
                .map_err(|e| EngineError(e.to_string()))?;
            Ok(self.insert(out))
        }
    }

    pub fn gather_indexed<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        input: BufferId,
        indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let indices = self.get(indices)?.clone();
        if let Some(t) = self.bf16.get(&input) {
            let out = s
                .gather_indexed_bf16(t, &indices)
                .map_err(|e| EngineError(e.to_string()))?;
            Ok(self.insert_bf16(out))
        } else {
            let out = s
                .gather_indexed(self.get(input)?, &indices)
                .map_err(|e| EngineError(e.to_string()))?;
            Ok(self.insert(out))
        }
    }

    pub fn repack_many<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        inputs: &[BufferId],
        sources: &[(usize, [usize; 2])],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        if inputs.first().is_some_and(|a| self.bf16.contains_key(a)) {
            let ts = inputs
                .iter()
                .map(|a| {
                    self.bf16
                        .get(a)
                        .ok_or_else(|| EngineError("mixed repack storage".into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let out = s
                .repack_many_bf16(&ts, sources, dims)
                .map_err(|e| EngineError(e.to_string()))?;
            return Ok(self.insert_bf16(out));
        }
        let ts = inputs
            .iter()
            .map(|&a| self.get(a))
            .collect::<Result<Vec<_>, _>>()?;
        let out = s
            .repack_many(&ts, sources, dims)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(out))
    }

    pub fn copy_blocks<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        moves: &[BlockMove],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        if let Some(t) = self.bf16.get(&a) {
            let [rows, cols] = dims;
            let count = rows
                .checked_mul(cols)
                .ok_or_else(|| EngineError("BF16 copy shape overflow".into()))?;
            let mut sources = vec![[usize::MAX; 2]; count];
            for m in moves {
                for r in 0..m.extent[0] {
                    for c in 0..m.extent[1] {
                        let [tr, tc] = [m.to[0].checked_add(r), m.to[1].checked_add(c)]
                            .map(|v| v.unwrap_or(usize::MAX));
                        if tr >= rows || tc >= cols || sources[tr * cols + tc] != [usize::MAX; 2] {
                            return Err(EngineError(
                                "invalid or overlapping BF16 block copy".into(),
                            ));
                        }
                        let [sr, sc] = if m.transposed { [c, r] } else { [r, c] };
                        sources[tr * cols + tc] =
                            [m.from[0].checked_add(sr), m.from[1].checked_add(sc)]
                                .map(|v| v.unwrap_or(usize::MAX));
                    }
                }
            }
            let c = s
                .repack_bf16(t, &sources, dims)
                .map_err(|e| EngineError(e.to_string()))?;
            return Ok(self.insert_bf16(c));
        }
        let ta = self.get(a)?.clone();
        let c = s
            .copy_blocks(&ta, moves, dims)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(c))
    }

    pub fn eltwise<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        kind: u32,
        scalar: f32,
        a: BufferId,
        b: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let op = tt_kernels::tensor::Eltwise {
            scalar2: 0.0,
            kind,
            scalar,
        };
        self.eltwise_op(s, op, a, b, None)
    }

    pub fn eltwise_op<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        op: tt_kernels::tensor::Eltwise,
        a: BufferId,
        b: Option<BufferId>,
        c: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        if c.is_none()
            && matches!(
                op.kind,
                tt_kernels::kind::ADD
                    | tt_kernels::kind::SUB
                    | tt_kernels::kind::MUL
                    | tt_kernels::kind::ADD_SCALAR
                    | tt_kernels::kind::MUL_SCALAR
            )
        {
            if let ElementwiseMode::Matrix {
                precision,
                fidelity,
            } = self.elementwise
            {
                return self.matrix_eltwise(s, op, a, b, precision, fidelity);
            }
        }
        let ta = self.get(a)?.clone();
        let tb = b.map(|b| self.get(b).cloned()).transpose()?;
        let tc = c.map(|c| self.get(c).cloned()).transpose()?;
        let c = s
            .eltwise3(op, &ta, tb.as_ref(), tc.as_ref())
            .map_err(|e| EngineError(e.to_string()))?;
        let dims = [c.rows, c.cols];
        self.next += 1;
        self.live.insert(self.next, c);
        Ok((self.next, dims))
    }

    pub fn sum_rows<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let ta = self.get(a)?.clone();
        let c = s.sum_rows(&ta).map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(c))
    }

    pub fn reduce<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        axis: tt_kernels::sfpu::reduce::Axis,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let ta = self.get(a)?.clone();
        let c = s
            .reduce(&ta, op, axis)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(c))
    }

    pub fn scan<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        op: tt_kernels::sfpu::scan::ScanOp,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let ta = self.get(a)?.clone();
        let c = s.scan(&ta, op).map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(c))
    }

    /// Lane T5: [`Engine::sort_planes`], for engines that serve these buffers.
    pub fn sort_planes<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        spec: tt_kernels::sfpu::sort::Spec,
    ) -> Result<SortedBuffers, EngineError> {
        let planes = self.get(a)?.clone();
        let sorted = s
            .sort_planes(&planes, spec)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(SortedBuffers {
            keys: self.insert(sorted.keys),
            indices: sorted.indices.map(|t| self.insert(t)),
        })
    }

    /// A view: freeing it frees nothing (`DramTensor::rows_view`).
    pub fn slice_rows(
        &mut self,
        a: BufferId,
        first: usize,
        rows: usize,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        if let Some(t) = self.bfp.get(&a) {
            let v = t
                .rows_view(first, rows)
                .map_err(|e| EngineError(e.to_string()))?;
            let dims = [v.rows(), v.cols()];
            self.next += 1;
            self.bfp.insert(self.next, v);
            return Ok((self.next, dims));
        }
        if let Some(t) = self.bf16.get(&a) {
            let v = t
                .rows_view(first, rows)
                .map_err(|e| EngineError(e.to_string()))?;
            return Ok(self.insert_bf16(v));
        }
        let v = self
            .get(a)?
            .rows_view(first, rows)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.insert(v))
    }

    fn insert(&mut self, t: tt_kernels::tensor::DramTensor) -> (BufferId, [usize; 2]) {
        let dims = [t.rows, t.cols];
        self.next += 1;
        self.live.insert(self.next, t);
        (self.next, dims)
    }

    /// How many are live.
    pub fn len(&self) -> usize {
        self.live.len() + self.bf16.len() + self.bfp.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty() && self.bf16.is_empty() && self.bfp.is_empty()
    }

    fn get(&self, id: BufferId) -> Result<&tt_kernels::tensor::DramTensor, EngineError> {
        self.live
            .get(&id)
            .ok_or_else(|| EngineError(format!("no device buffer {id}")))
    }
}

/// What [`Engine::sort_planes`] made: each buffer with its `[rows, cols]`.
pub struct SortedBuffers {
    pub keys: (BufferId, [usize; 2]),
    pub indices: Option<(BufferId, [usize; 2])>,
}

/// A sort of `a`'s planes on the device, without waiting (lane T5): the keys'
/// and, if `spec.indices`, the indices' buffers are named now, both `dims`.
pub(crate) fn sort_planes(
    device: TtDevice,
    a: BufferId,
    spec: tt_kernels::sfpu::sort::Spec,
    dims: [usize; 2],
) -> SortedBuffers {
    let keys = name(dims);
    let indices = spec.indices.then(|| name(dims));
    crate::traffic::timed("sort", || {
        send(
            device,
            Box::new(move |engine, ids| {
                let made = ids
                    .get(a)
                    .and_then(|a| engine.sort_planes(a, spec))
                    .and_then(|s| {
                        if s.keys.1 == dims && s.indices.as_ref().is_none_or(|i| i.1 == dims) {
                            Ok(s)
                        } else {
                            engine.free(s.keys.0);
                            if let Some(i) = s.indices {
                                engine.free(i.0);
                            }
                            Err(EngineError(format!("sorted planes are not {dims:?}")))
                        }
                    });
                match made {
                    Ok(s) => {
                        ids.map.insert(keys, Ok(s.keys.0));
                        if let (Some(id), Some(i)) = (indices, s.indices) {
                            ids.map.insert(id, Ok(i.0));
                        }
                    }
                    Err(e) => {
                        let why = Arc::<str>::from(format!("sort {spec:?} on {device}: {e}"));
                        ids.failed.get_or_insert_with(|| why.clone());
                        ids.map.insert(keys, Err(why.clone()));
                        if let Some(id) = indices {
                            ids.map.insert(id, Err(why));
                        }
                    }
                }
            }),
        )
    });
    SortedBuffers {
        keys: (keys, dims),
        indices: indices.map(|i| (i, dims)),
    }
}

/// Why an engine could not start or finish a job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError(pub String);

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EngineError {}

impl From<tt_kernels::runtime::RunError> for EngineError {
    fn from(e: tt_kernels::runtime::RunError) -> Self {
        EngineError(e.to_string())
    }
}

impl From<SessionError> for EngineError {
    fn from(e: SessionError) -> Self {
        EngineError(e.to_string())
    }
}

type Job = Box<dyn FnOnce(&mut dyn Engine, &mut Ids) + Send>;

/// Each buffer the callers have named, as the engine numbers it -- or why it
/// was never made. Kept on the server thread, one per attachment: a caller
/// names a result before it exists (B8, asynchronous dispatch), and the
/// server, running jobs in order, makes it before any job reads it.
#[derive(Default)]
pub(crate) struct Ids {
    map: HashMap<BufferId, Result<BufferId, Arc<str>>>,
    /// The first asynchronous op that failed on this attachment: reported by
    /// every wait from then on (a download, a trace's replay), whatever it
    /// reads, so a failure whose result nobody reads is not lost.
    failed: Option<Arc<str>>,
}

impl Ids {
    /// `Err` with the attachment's first failure, if an op has failed.
    pub(crate) fn healthy(&self) -> Result<(), EngineError> {
        match &self.failed {
            Some(why) => Err(EngineError(format!("an earlier device op failed: {why}"))),
            None => Ok(()),
        }
    }

    /// The engine's number for the caller's `id`; an error if the op that was
    /// to make it failed (its message), or if this attachment never made it
    /// -- a tensor that outlived an earlier attachment of its device.
    pub(crate) fn get(&self, id: BufferId) -> Result<BufferId, EngineError> {
        match self.map.get(&id) {
            Some(Ok(e)) => Ok(*e),
            Some(Err(why)) => Err(EngineError(why.to_string())),
            None => Err(EngineError(format!(
                "buffer {id} is not one of this attachment's: a tensor that outlived \
                 the attachment it was made in"
            ))),
        }
    }
}

/// The next caller-side buffer number: process-wide and never reused, so a
/// stale tensor can never name another attachment's buffer.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Each caller-side buffer's `[rows, cols]`, known when it is named, so
/// asynchronous ops can say their results' shapes without waiting.
static DIMS: Mutex<Option<HashMap<BufferId, [usize; 2]>>> = Mutex::new(None);

fn dims_of(id: BufferId) -> [usize; 2] {
    *DIMS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_or_insert_with(HashMap::new)
        .get(&id)
        .unwrap_or_else(|| panic!("buffer {id} has no recorded shape"))
}

fn name(dims: [usize; 2]) -> BufferId {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    DIMS.lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(id, dims);
    id
}

fn forget(id: BufferId) {
    if let Some(m) = DIMS.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
        m.remove(&id);
    }
}

/// Handed to an [`attach`] factory: call [`Serve::serve`] with the engine once
/// it exists, and it runs the device's jobs until the device is detached.
pub struct Serve {
    ready: Sender<Result<ElementwiseMode, EngineError>>,
    jobs: Receiver<Job>,
}

impl Serve {
    /// Serve jobs on `engine` until the [`AttachGuard`] is dropped.
    pub fn serve(self, engine: &mut dyn Engine) {
        let _ = self.ready.send(Ok(engine.elementwise_mode()));
        let mut ids = Ids::default();
        while let Ok(job) = self.jobs.recv() {
            job(engine, &mut ids);
        }
    }
}

struct Attached {
    jobs: Sender<Job>,
    elementwise: ElementwiseMode,
}

static ATTACHED: Mutex<Option<HashMap<TtDevice, Attached>>> = Mutex::new(None);

fn with_attached<R>(f: impl FnOnce(&mut HashMap<TtDevice, Attached>) -> R) -> R {
    let mut guard = ATTACHED.lock().unwrap_or_else(|p| p.into_inner());
    f(guard.get_or_insert_with(HashMap::new))
}

/// How many devices are attached.
pub(crate) fn attached_count() -> usize {
    with_attached(|a| a.len())
}

pub(crate) fn is_attached(device: TtDevice) -> bool {
    with_attached(|a| a.contains_key(&device))
}

/// Keeps `device` attached; dropping it detaches the device and drops its
/// hardware.
#[must_use = "dropping the guard detaches the device at once"]
pub struct AttachGuard {
    device: TtDevice,
    thread: Option<JoinHandle<()>>,
}

impl Drop for AttachGuard {
    fn drop(&mut self) {
        // Removing the sender closes the queue; the server's `recv` fails and
        // `serve` returns, and so does the factory, dropping the hardware.
        with_attached(|a| a.remove(&self.device));
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Attach `device`: start its server thread and run `factory` on it.
///
/// `factory` builds the engine -- opening the hardware -- and calls
/// [`Serve::serve`] with it. It returns an error, rather than serving, if the
/// hardware cannot be opened; `attach` returns that error. Attaching a device
/// that is already attached is refused.
pub fn attach<F>(device: TtDevice, factory: F) -> Result<AttachGuard, EngineError>
where
    F: FnOnce(Serve) -> Result<(), EngineError> + Send + 'static,
{
    let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel();
    let already = with_attached(|a| match a.entry(device) {
        std::collections::hash_map::Entry::Occupied(_) => true,
        std::collections::hash_map::Entry::Vacant(v) => {
            v.insert(Attached {
                jobs: jobs_tx,
                elementwise: ElementwiseMode::Sfpu,
            });
            false
        }
    });
    if already {
        return Err(EngineError(format!("{device} is already attached")));
    }
    let failed = ready_tx.clone();
    let thread = std::thread::Builder::new()
        .name(format!("burn-tt {device}"))
        .spawn(move || {
            let serve = Serve {
                ready: ready_tx,
                jobs: jobs_rx,
            };
            match factory(serve) {
                Ok(()) => {
                    let _ = failed.send(Err(EngineError(
                        "the engine factory returned without serving".into(),
                    )));
                }
                Err(e) => {
                    let _ = failed.send(Err(e));
                }
            }
        })
        .map_err(|e| EngineError(format!("could not start the server thread: {e}")))?;
    let guard = AttachGuard {
        device,
        thread: Some(thread),
    };
    match ready_rx.recv() {
        Ok(Ok(mode)) => {
            with_attached(|a| a.get_mut(&device).expect("reserved attachment").elementwise = mode);
            Ok(guard)
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err(EngineError("the server thread died before serving".into())),
    }
}

/// Run `job` on `device`'s engine and wait for it.
///
/// # Panics
///
/// If `device` is not attached, or its server thread has gone: Burn's ops
/// cannot return an error, and quietly computing on the host instead would
/// make the device path unobservable.
pub(crate) fn run<R: Send + 'static>(
    device: TtDevice,
    job: impl FnOnce(&mut dyn Engine, &mut Ids) -> R + Send + 'static,
) -> R {
    let (tx, rx) = mpsc::channel();
    send(
        device,
        Box::new(move |engine, ids| {
            let _ = tx.send(job(engine, ids));
        }),
    );
    rx.recv()
        .unwrap_or_else(|_| panic!("{device}'s server thread stopped during a job"))
}

/// Queue `job` on `device`'s server, without waiting.
fn send(device: TtDevice, job: Job) {
    let sender = with_attached(|a| a.get(&device).map(|d| d.jobs.clone()));
    let Some(sender) = sender else {
        panic!("{device} is not attached: call burn_tt::attach before running device ops on it");
    };
    sender
        .send(job)
        .unwrap_or_else(|_| panic!("{device}'s server thread has stopped"));
}

/// [`run`], timed by `kind` (`crate::traffic::device_time`).
fn timed_run<R: Send + 'static>(
    kind: &'static str,
    device: TtDevice,
    job: impl FnOnce(&mut dyn Engine, &mut Ids) -> R + Send + 'static,
) -> R {
    crate::traffic::timed(kind, || run(device, job))
}

/// An op that makes a buffer, without waiting for it (B8): the result is
/// named now, `dims` its shape as the caller computes it; the server runs
/// `op` in order, translating the buffers it reads, and records what it made
/// -- or, if `op` fails, why, which the first wait on the result reports
/// (`what` says which op). An op on a buffer that was never made fails the
/// same way, naming the first failure.
fn submit(
    kind: &'static str,
    device: TtDevice,
    dims: [usize; 2],
    what: impl FnOnce() -> String + Send + 'static,
    op: impl FnOnce(&mut dyn Engine, &Ids) -> Result<(BufferId, [usize; 2]), EngineError>
        + Send
        + 'static,
) -> (BufferId, [usize; 2]) {
    let id = name(dims);
    crate::traffic::timed(kind, || {
        send(
            device,
            Box::new(move |engine, ids| {
                let made = op(engine, ids).and_then(|(e, got)| {
                    if got == dims {
                        Ok(e)
                    } else {
                        engine.free(e);
                        Err(EngineError(format!(
                            "made [{}, {}] where [{}, {}] was expected",
                            got[0], got[1], dims[0], dims[1]
                        )))
                    }
                });
                let made =
                    made.map_err(|e| Arc::<str>::from(format!("{} on {device}: {e}", what())));
                if let Err(why) = &made {
                    ids.failed.get_or_insert_with(|| why.clone());
                }
                ids.map.insert(id, made);
            }),
        )
    });
    (id, dims)
}

/// `A[m, k] @ B[k, n]` on `device`, panicking on a device error.
pub(crate) fn matmul(device: TtDevice, a: &[f32], b: &[f32], mkn: [usize; 3]) -> Vec<f32> {
    crate::report::staged(mkn);
    let (a, b) = (a.to_vec(), b.to_vec());
    timed_run("matmul_host", device, move |engine, _| {
        engine.matmul(&a, &b, mkn)
    })
    .unwrap_or_else(|e| panic!("matmul {mkn:?} on {device}: {e}"))
}

/// A native full reduction for an engine without resident tensor storage.
pub(crate) fn full_reduce(device: TtDevice, values: Vec<f32>, dims: [usize; 2], mean: bool) -> f32 {
    crate::report::staged_reduce(dims);
    timed_run("full_reduce", device, move |engine, _| {
        engine.full_reduce(&values, dims, mean)
    })
    .unwrap_or_else(|e| panic!("native full reduction {dims:?} on {device}: {e}"))
}

pub(crate) fn elementwise_mode(device: TtDevice) -> ElementwiseMode {
    with_attached(|a| {
        a.get(&device)
            .unwrap_or_else(|| panic!("{device} is not attached"))
            .elementwise
    })
}

/// Does `device`'s engine keep tensors on the device? Asked once per device.
pub(crate) fn supports_dram(device: TtDevice) -> bool {
    static KNOWN: Mutex<Option<HashMap<TtDevice, bool>>> = Mutex::new(None);
    if let Some(&k) = KNOWN
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_or_insert_with(HashMap::new)
        .get(&device)
    {
        return k;
    }
    let k = run(device, |engine, _| engine.supports_dram());
    KNOWN
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(device, k);
    k
}

/// Everything `device`'s engine has moved across its transport so far, if it
/// can say ([`Engine::device_traffic`]). Queued behind every job already sent,
/// frees included, so it counts all of them.
pub fn device_traffic(device: TtDevice) -> Option<tt_device::Traffic> {
    run(device, |engine, _| engine.device_traffic())
}

/// Wait for queued work, then report completed per-chip products and mover ACKs.
pub fn mesh_execution(device: TtDevice) -> Option<tt_kernels::shard::FabricExecution> {
    run(device, |engine, _| engine.mesh_execution())
}

/// Upload, panicking on a device error.
pub(crate) fn upload(device: TtDevice, values: Vec<f32>, rows: usize, cols: usize) -> BufferId {
    crate::traffic::uploaded(rows, cols);
    crate::report::uploaded(rows, cols);
    submit(
        "upload",
        device,
        [rows, cols],
        move || format!("upload [{rows}, {cols}]"),
        move |engine, _| {
            engine
                .upload(&values, rows, cols)
                .map(|e| (e, [rows, cols]))
        },
    )
    .0
}

pub(crate) fn metadata(device: TtDevice, bits: Vec<u32>, dims: [usize; 2], elem: Elem) -> BufferId {
    submit(
        "metadata",
        device,
        dims,
        move || format!("pool geometry {dims:?}"),
        move |engine, _| engine.metadata(&bits, dims, elem).map(|id| (id, dims)),
    )
    .0
}

pub(crate) fn upload_bf16(device: TtDevice, bits: Vec<u16>, rows: usize, cols: usize) -> BufferId {
    crate::traffic::uploaded_width(rows, cols, 2);
    crate::report::uploaded_width(rows, cols, 2);
    submit(
        "upload_bf16",
        device,
        [rows, cols],
        move || format!("BF16 [{rows}, {cols}]"),
        move |engine, _| {
            engine
                .upload_bf16(&bits, rows, cols)
                .map(|id| (id, [rows, cols]))
        },
    )
    .0
}

pub(crate) fn download_bf16(device: TtDevice, id: BufferId, rows: usize, cols: usize) -> Vec<u16> {
    crate::traffic::downloaded_width(rows, cols, 2);
    crate::report::downloaded_width(rows, cols, 2);
    timed_run("download_bf16", device, move |engine, ids| {
        engine.download_bf16(ids.get(id)?)
    })
    .unwrap_or_else(|e| panic!("BF16 download [{rows}, {cols}]: {e}"))
}

pub(crate) fn cast_float(
    device: TtDevice,
    id: BufferId,
    dims: [usize; 2],
    bf16: bool,
) -> (BufferId, [usize; 2]) {
    submit(
        "cast_float",
        device,
        dims,
        || "native floating-point storage conversion".into(),
        move |engine, ids| engine.cast_float(ids.get(id)?, bf16),
    )
}

pub(crate) fn cast_bfp(
    device: TtDevice,
    id: BufferId,
    dims: [usize; 2],
    format: Option<tt_kernels::bfp::BfpFormat>,
) -> (BufferId, [usize; 2]) {
    submit(
        "cast_bfp",
        device,
        dims,
        move || format!("BFP storage {format:?}"),
        move |engine, ids| engine.cast_bfp(ids.get(id)?, format),
    )
}

pub(crate) fn matmul_bfp(
    device: TtDevice,
    a: BufferId,
    b: BufferId,
    dims: [usize; 2],
) -> (BufferId, [usize; 2]) {
    submit(
        "matmul_bfp",
        device,
        dims,
        || "packed BFP matmul".into(),
        move |engine, ids| engine.matmul_bfp(ids.get(a)?, ids.get(b)?),
    )
}

/// Upload datums as bits, panicking on a device error.
pub(crate) fn matmul_bf16(
    device: TtDevice,
    a: BufferId,
    b: BufferId,
    dims: [usize; 2],
) -> (BufferId, [usize; 2]) {
    submit(
        "matmul_bf16",
        device,
        dims,
        || "direct BF16 matmul".into(),
        move |engine, ids| engine.matmul_bf16(ids.get(a)?, ids.get(b)?),
    )
}

/// Upload datums as bits, panicking on a device error.
pub(crate) fn upload_bits(
    device: TtDevice,
    bits: Vec<u32>,
    rows: usize,
    cols: usize,
    elem: Elem,
) -> BufferId {
    crate::traffic::uploaded(rows, cols);
    crate::report::uploaded(rows, cols);
    submit(
        "upload",
        device,
        [rows, cols],
        move || format!("upload {elem:?} [{rows}, {cols}]"),
        move |engine, _| {
            engine
                .upload_bits(&bits, rows, cols, elem)
                .map(|e| (e, [rows, cols]))
        },
    )
    .0
}

pub(crate) fn pool_bf16(
    device: TtDevice,
    a: BufferId,
    windows: Vec<Vec<[usize; 2]>>,
    divisors: Vec<usize>,
    dims: [usize; 2],
) -> (BufferId, [usize; 2]) {
    submit(
        "pool_bf16",
        device,
        dims,
        || "BF16 GAPOOL windows".into(),
        move |engine, ids| engine.pool_bf16(ids.get(a)?, &windows, &divisors, dims),
    )
}

/// Download any buffer's datums as bits, panicking on a device error.
pub(crate) fn download_bits(device: TtDevice, id: BufferId, rows: usize, cols: usize) -> Vec<u32> {
    let v = timed_run("download", device, move |engine, ids| {
        ids.healthy()?;
        ids.get(id).and_then(|e| engine.download_bits(e))
    })
    .unwrap_or_else(|e| panic!("download {id} from {device}: {e}"));
    debug_assert_eq!(v.len(), rows * cols);
    crate::traffic::downloaded(rows, cols);
    crate::report::downloaded(rows, cols);
    v
}

/// Download, panicking on a device error.
/// `rows` and `cols` are the buffer's, for the traffic count.
pub(crate) fn download(device: TtDevice, id: BufferId, rows: usize, cols: usize) -> Vec<f32> {
    let v = timed_run("download", device, move |engine, ids| {
        ids.healthy()?;
        ids.get(id).and_then(|e| engine.download(e))
    })
    .unwrap_or_else(|e| panic!("download {id} from {device}: {e}"));
    debug_assert_eq!(v.len(), rows * cols);
    crate::traffic::downloaded(rows, cols);
    crate::report::downloaded(rows, cols);
    v
}

/// Free, without waiting; nothing to do if the device has gone.
pub(crate) fn free(device: TtDevice, id: BufferId) {
    forget(id);
    if let Some(sender) = with_attached(|a| a.get(&device).map(|d| d.jobs.clone())) {
        let _ = sender.send(Box::new(move |engine, ids| {
            // A buffer this attachment never made (a stale tensor's) is not
            // anyone's here: nothing to free.
            if let Some(Ok(e)) = ids.map.remove(&id) {
                engine.free(e);
            }
        }));
    }
}

/// Element-wise on the device -- the whole op, both scalars, and a ternary
/// Element-wise on the device -- the whole op, both scalars, and a ternary
/// op's third operand -- without waiting: the result is `a`'s shape.
pub(crate) fn eltwise_op(
    device: TtDevice,
    op: tt_kernels::tensor::Eltwise,
    a: BufferId,
    b: Option<BufferId>,
    c: Option<BufferId>,
) -> (BufferId, [usize; 2]) {
    submit(
        "eltwise",
        device,
        dims_of(a),
        move || format!("element-wise {:#x}", op.kind),
        move |engine, ids| {
            let b = b.map(|b| ids.get(b)).transpose()?;
            let c = c.map(|c| ids.get(c)).transpose()?;
            engine.eltwise_op(op, ids.get(a)?, b, c)
        },
    )
}

/// `x^y` on the device, without waiting: `x`'s shape.
pub(crate) fn pow(device: TtDevice, x: BufferId, y: PowArg) -> (BufferId, [usize; 2]) {
    submit(
        "pow",
        device,
        dims_of(x),
        || "pow".into(),
        move |engine, ids| {
            let y = match y {
                PowArg::Tensor(t) => PowArg::Tensor(ids.get(t)?),
                PowArg::Int(t) => PowArg::Int(ids.get(t)?),
                s @ PowArg::Scalar(_) => s,
            };
            engine.pow(ids.get(x)?, y)
        },
    )
}

/// A reduction on the device, without waiting: one row (over rows) or one
/// column (over columns) of `a`'s.
pub(crate) fn reduce(
    device: TtDevice,
    a: BufferId,
    op: tt_kernels::sfpu::reduce::ReduceOp,
    axis: tt_kernels::sfpu::reduce::Axis,
) -> (BufferId, [usize; 2]) {
    use tt_kernels::sfpu::reduce::Axis;
    let [r, c] = dims_of(a);
    let dims = match axis {
        Axis::Rows => [1, c],
        Axis::Cols => [r, 1],
    };
    submit(
        "reduce",
        device,
        dims,
        move || format!("{op:?} over {axis:?}"),
        move |engine, ids| engine.reduce(ids.get(a)?, op, axis),
    )
}

pub(crate) fn scan(
    device: TtDevice,
    a: BufferId,
    op: tt_kernels::sfpu::scan::ScanOp,
) -> (BufferId, [usize; 2]) {
    submit(
        "scan",
        device,
        dims_of(a),
        move || format!("{op:?} scan"),
        move |engine, ids| engine.scan(ids.get(a)?, op),
    )
}

/// A row view on the device, without waiting.
pub(crate) fn slice_rows(
    device: TtDevice,
    a: BufferId,
    first: usize,
    rows: usize,
) -> (BufferId, [usize; 2]) {
    let [_, c] = dims_of(a);
    submit(
        "slice_rows",
        device,
        [rows, c],
        move || format!("row view {first}..{}", first + rows),
        move |engine, ids| engine.slice_rows(ids.get(a)?, first, rows),
    )
}

/// `op(A) @ op(B)` on the device, without waiting.
pub(crate) fn matmul_dram(
    device: TtDevice,
    a: BufferId,
    a_transposed: bool,
    b: BufferId,
    b_transposed: bool,
) -> (BufferId, [usize; 2]) {
    let ([ar, ac], [br, bc]) = (dims_of(a), dims_of(b));
    let m = if a_transposed { ac } else { ar };
    let n = if b_transposed { br } else { bc };
    submit(
        "matmul_dram",
        device,
        [m, n],
        move || "matmul".into(),
        move |engine, ids| engine.matmul_dram(ids.get(a)?, a_transposed, ids.get(b)?, b_transposed),
    )
}

/// A batched matmul over blocks on the device, without waiting.
pub(crate) fn matmul_dram_batched(
    device: TtDevice,
    a: BufferId,
    b: BufferId,
    items: Vec<(Block, Block)>,
    mkn: [usize; 3],
) -> (BufferId, [usize; 2]) {
    let [m, _, n] = mkn;
    submit(
        "matmul_dram",
        device,
        [items.len() * m, n],
        || "batched matmul".into(),
        move |engine, ids| engine.matmul_dram_batched(ids.get(a)?, ids.get(b)?, &items, mkn),
    )
}

/// A row gather on the device, without waiting.
pub(crate) fn gather_rows(
    device: TtDevice,
    sources: Vec<BufferId>,
    rows: Vec<(usize, usize)>,
    cols: usize,
) -> (BufferId, [usize; 2]) {
    submit(
        "gather_rows",
        device,
        [rows.len(), cols],
        || "row gather".into(),
        move |engine, ids| {
            let sources = sources
                .iter()
                .map(|&s| ids.get(s))
                .collect::<Result<Vec<_>, _>>()?;
            engine.gather_rows(&sources, &rows, cols)
        },
    )
}

/// Rows added by index on the device, without waiting: `t`'s shape.
pub(crate) fn rows_add(
    device: TtDevice,
    t: BufferId,
    indices: Vec<usize>,
    value: BufferId,
) -> (BufferId, [usize; 2]) {
    submit(
        "rows_add",
        device,
        dims_of(t),
        || "rows added by index".into(),
        move |engine, ids| engine.rows_add(ids.get(t)?, &indices, ids.get(value)?),
    )
}

/// A block copy on the device, without waiting.
pub(crate) fn copy_blocks(
    device: TtDevice,
    a: BufferId,
    moves: Vec<BlockMove>,
    dims: [usize; 2],
) -> (BufferId, [usize; 2]) {
    submit(
        "copy_blocks",
        device,
        dims,
        || "block copy".into(),
        move |engine, ids| engine.copy_blocks(ids.get(a)?, &moves, dims),
    )
}

pub(crate) fn repack(
    device: TtDevice,
    a: BufferId,
    sources: Vec<[usize; 2]>,
    dims: [usize; 2],
) -> (BufferId, [usize; 2]) {
    submit(
        "repack",
        device,
        dims,
        || "native repack".into(),
        move |engine, ids| engine.repack(ids.get(a)?, &sources, dims),
    )
}

pub(crate) fn zeros(
    device: TtDevice,
    dims: [usize; 2],
    elem: Elem,
    bf16: bool,
) -> (BufferId, [usize; 2]) {
    submit(
        "zeros",
        device,
        dims,
        || "native initialization".into(),
        move |engine, _| engine.zeros_dram(dims, elem, bf16),
    )
}

pub(crate) fn gather_indexed(
    device: TtDevice,
    input: BufferId,
    indices: BufferId,
    dims: [usize; 2],
) -> (BufferId, [usize; 2]) {
    submit(
        "gather_indexed",
        device,
        dims,
        || "checked resident indices".into(),
        move |engine, ids| engine.gather_indexed(ids.get(input)?, ids.get(indices)?),
    )
}

pub(crate) fn repack_many(
    device: TtDevice,
    inputs: Vec<BufferId>,
    sources: Vec<(usize, [usize; 2])>,
    dims: [usize; 2],
) -> (BufferId, [usize; 2]) {
    submit(
        "repack_many",
        device,
        dims,
        || "multiple resident sources".into(),
        move |engine, ids| {
            let inputs = inputs
                .iter()
                .map(|&a| ids.get(a))
                .collect::<Result<Vec<_>, _>>()?;
            engine.repack_many(&inputs, &sources, dims)
        },
    )
}

// --- Silicon ------------------------------------------------------------------

/// The silicon engine: a [`Session`] on `/dev/tenstorrent/N`.
pub struct KmdEngine {
    pub session: Session<tt_kmd::Kmd>,
    pub route: SrcRoute,
    pub fidelity: Fidelity,
    /// Per-role budget for each run; on silicon, a floor of one second applies
    /// (`tt_kernels::runtime::run`).
    pub budget: u64,
    /// Tensors kept in GDDR, if the session has it enabled.
    pub buffers: Option<DramBuffers>,
}

impl KmdEngine {
    /// Consuming builder for a factory, before Serve attaches this engine.
    pub fn with_elementwise_mode(mut self, mode: ElementwiseMode) -> Result<Self, EngineError> {
        if let Some(buffers) = self.buffers.take() {
            self.buffers = Some(buffers.with_elementwise_mode(mode));
        } else if mode != ElementwiseMode::Sfpu {
            return Err(EngineError(
                "matrix elementwise requires resident storage".into(),
            ));
        }
        Ok(self)
    }
}

impl Engine for KmdEngine {
    fn elementwise_mode(&self) -> ElementwiseMode {
        self.buffers
            .as_ref()
            .map_or(ElementwiseMode::Sfpu, DramBuffers::elementwise_mode)
    }
    fn pool_bf16(
        &mut self,
        a: BufferId,
        windows: &[Vec<[usize; 2]>],
        divisors: &[usize],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers.as_mut().ok_or_else(unsupported)?.pool_bf16(
            &mut self.session,
            a,
            windows,
            divisors,
            dims,
        )
    }
    fn matmul_bf16(
        &mut self,
        a: BufferId,
        b: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers.as_mut().ok_or_else(unsupported)?.matmul_bf16(
            &mut self.session,
            a,
            b,
            self.fidelity,
            self.budget,
        )
    }
    fn upload_bf16(
        &mut self,
        bits: &[u16],
        rows: usize,
        cols: usize,
    ) -> Result<BufferId, EngineError> {
        self.buffers.as_mut().ok_or_else(unsupported)?.upload_bf16(
            &mut self.session,
            bits,
            rows,
            cols,
        )
    }
    fn download_bf16(&mut self, id: BufferId) -> Result<Vec<u16>, EngineError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .download_bf16(&mut self.session, id)
    }
    fn cast_float(
        &mut self,
        id: BufferId,
        bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .cast_float(&mut self.session, id, bf16)
    }
    fn cast_bfp(
        &mut self,
        id: BufferId,
        format: Option<tt_kernels::bfp::BfpFormat>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .cast_bfp(&mut self.session, id, format)
    }
    fn matmul_bfp(
        &mut self,
        a: BufferId,
        b: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers.as_mut().ok_or_else(unsupported)?.matmul_bfp(
            &mut self.session,
            a,
            b,
            self.fidelity,
        )
    }
    fn matmul(&mut self, a: &[f32], b: &[f32], mkn: [usize; 3]) -> Result<Vec<f32>, EngineError> {
        Ok(self
            .session
            .matmul(a, b, mkn, self.route, self.fidelity, self.budget)?)
    }
    fn supports_dram(&self) -> bool {
        self.buffers.is_some()
    }
    fn upload(&mut self, v: &[f32], rows: usize, cols: usize) -> Result<BufferId, EngineError> {
        let b = self.buffers.as_mut().ok_or_else(unsupported)?;
        b.upload(&mut self.session, v, rows, cols)
    }
    fn metadata(
        &mut self,
        bits: &[u32],
        dims: [usize; 2],
        elem: Elem,
    ) -> Result<BufferId, EngineError> {
        let b = self.buffers.as_mut().ok_or_else(unsupported)?;
        b.metadata(&mut self.session, bits, dims, elem)
    }
    fn download(&mut self, id: BufferId) -> Result<Vec<f32>, EngineError> {
        let b = self.buffers.as_mut().ok_or_else(unsupported)?;
        b.download(&mut self.session, id)
    }
    fn upload_bits(
        &mut self,
        v: &[u32],
        rows: usize,
        cols: usize,
        elem: Elem,
    ) -> Result<BufferId, EngineError> {
        let b = self.buffers.as_mut().ok_or_else(unsupported)?;
        b.upload_bits(&mut self.session, v, rows, cols, elem)
    }
    fn download_bits(&mut self, id: BufferId) -> Result<Vec<u32>, EngineError> {
        let b = self.buffers.as_mut().ok_or_else(unsupported)?;
        b.download_bits(&mut self.session, id)
    }
    fn free(&mut self, id: BufferId) {
        if let Some(b) = self.buffers.as_mut() {
            b.free(&mut self.session, id);
        }
    }
    fn matmul_dram(
        &mut self,
        a: BufferId,
        ta: bool,
        b: BufferId,
        tb: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.matmul(
            &mut self.session,
            a,
            ta,
            b,
            tb,
            self.route,
            self.fidelity,
            self.budget,
        )
    }
    fn eltwise(
        &mut self,
        kind: u32,
        scalar: f32,
        a: BufferId,
        b: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.eltwise(&mut self.session, kind, scalar, a, b)
    }
    fn eltwise_op(
        &mut self,
        op: tt_kernels::tensor::Eltwise,
        a: BufferId,
        b: Option<BufferId>,
        c: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.eltwise_op(&mut self.session, op, a, b, c)
    }
    fn pow(&mut self, x: BufferId, y: PowArg) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.pow(&mut self.session, x, y)
    }
    fn sum_rows(&mut self, a: BufferId) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.sum_rows(&mut self.session, a)
    }
    fn reduce(
        &mut self,
        a: BufferId,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        axis: tt_kernels::sfpu::reduce::Axis,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.reduce(&mut self.session, a, op, axis)
    }
    fn scan(
        &mut self,
        a: BufferId,
        op: tt_kernels::sfpu::scan::ScanOp,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .scan(&mut self.session, a, op)
    }
    fn slice_rows(
        &mut self,
        a: BufferId,
        first: usize,
        rows: usize,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.slice_rows(a, first, rows)
    }
    fn matmul_dram_batched(
        &mut self,
        a: BufferId,
        b: BufferId,
        items: &[(Block, Block)],
        mkn: [usize; 3],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.matmul_batched(
            &mut self.session,
            a,
            b,
            items,
            mkn,
            self.route,
            self.fidelity,
            self.budget,
        )
    }
    fn copy_blocks(
        &mut self,
        a: BufferId,
        moves: &[BlockMove],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.copy_blocks(&mut self.session, a, moves, dims)
    }
    fn repack(
        &mut self,
        a: BufferId,
        sources: &[[usize; 2]],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.repack(&mut self.session, a, sources, dims)
    }
    fn zeros_dram(
        &mut self,
        dims: [usize; 2],
        elem: Elem,
        bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .zeros(&mut self.session, dims, elem, bf16)
    }
    fn gather_indexed(
        &mut self,
        input: BufferId,
        indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .gather_indexed(&mut self.session, input, indices)
    }
    fn repack_many(
        &mut self,
        inputs: &[BufferId],
        sources: &[(usize, [usize; 2])],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.repack_many(&mut self.session, inputs, sources, dims)
    }
    fn gather_rows(
        &mut self,
        sources: &[BufferId],
        rows: &[(usize, usize)],
        cols: usize,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.gather_rows(&mut self.session, sources, rows, cols)
    }
    fn rows_add(
        &mut self,
        t: BufferId,
        indices: &[usize],
        value: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.rows_add(&mut self.session, t, indices, value)
    }
    fn device_traffic(&mut self) -> Option<tt_device::Traffic> {
        Some(self.session.device().traffic())
    }
    fn begin_trace(&mut self) -> Result<(), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.begin_trace(&mut self.session)
    }
    fn end_trace(&mut self) -> Result<u64, EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.end_trace(&mut self.session)
    }
    fn run_trace(
        &mut self,
        trace: u64,
        input: BufferId,
        values: &[f32],
        output: BufferId,
    ) -> Result<TraceRun, EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.run_trace(&mut self.session, trace, input, values, output)
    }
    fn release_trace(&mut self, trace: u64) {
        if let Some(b) = self.buffers.as_mut() {
            b.release_trace(&mut self.session, trace);
        }
    }
    fn copy_into(&mut self, src: BufferId, dst: BufferId) -> Result<(), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.copy_into(&mut self.session, src, dst)
    }
    fn run_generic_trace(
        &mut self,
        trace: u64,
        inputs: &[(BufferId, InputPayload)],
        outputs: &[(BufferId, OutputKind)],
    ) -> Result<GenericTraceRun, EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.run_generic_trace(&mut self.session, trace, inputs, outputs)
    }

    // lane:t1_intbool (KmdEngine): add this lane's methods below this line only.

    // lane:t2_index (KmdEngine): add this lane's methods below this line only.

    // lane:t3_scan (KmdEngine): add this lane's methods below this line only.

    // lane:t4_rem (KmdEngine): add this lane's methods below this line only.

    // lane:t5_sort (KmdEngine): add this lane's methods below this line only.
    fn sort_planes(
        &mut self,
        a: BufferId,
        spec: tt_kernels::sfpu::sort::Spec,
    ) -> Result<SortedBuffers, EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.sort_planes(&mut self.session, a, spec)
    }

    // lane:t6_random (KmdEngine): add this lane's methods below this line only.

    // lane:t7_dtype (KmdEngine): add this lane's methods below this line only.

    // lane:t8_mathmode (KmdEngine): add this lane's methods below this line only.

    // lane:t9_mesh (KmdEngine): add this lane's methods below this line only.
}

pub fn copy_into(device: TtDevice, src: BufferId, dst: BufferId) -> Result<(), EngineError> {
    run(device, move |engine, ids| {
        ids.healthy()?;
        engine.copy_into(ids.get(src)?, ids.get(dst)?)
    })
}

pub fn run_generic_trace(
    device: TtDevice,
    trace: u64,
    inputs: Vec<(BufferId, InputPayload)>,
    outputs: Vec<(BufferId, OutputKind)>,
) -> Result<GenericTraceRun, EngineError> {
    run(device, move |engine, ids| {
        ids.healthy()?;
        let mapped_inputs: Result<Vec<(BufferId, InputPayload)>, EngineError> = inputs
            .into_iter()
            .map(|(buf, p)| ids.get(buf).map(|b| (b, p)))
            .collect();
        let mapped_outputs: Result<Vec<(BufferId, OutputKind)>, EngineError> = outputs
            .into_iter()
            .map(|(buf, k)| ids.get(buf).map(|b| (b, k)))
            .collect();
        engine.run_generic_trace(trace, &mapped_inputs?, &mapped_outputs?)
    })
}

/// Settings of the executors that were removed: refused with what replaces
/// them rather than ignored, so a script that still sets one learns why it no
/// longer does anything. `get` reads a variable, as `std::env::var` does.
fn refuse_retired_settings(get: impl Fn(&str) -> Option<String>) -> Result<(), EngineError> {
    if let Some(value) = get("TT_EXACT") {
        return Err(EngineError(format!("TT_EXACT={value}: retired with the Flex cutover; native numerical bounds apply; unset it")));
    }
    for variable in ["TT_EXECUTION", "TT_SCATTER"] {
        if let Some(value) = get(variable) {
            return Err(EngineError(format!(
                "{variable}={value}: retired; GDDR compute always uses B-reader/NC-writer ownership; unset it (TT_PIPELINE=0 disables overlap without changing ownership)"
            )));
        }
    }
    Ok(())
}

/// A factory for [`attach`] that opens `/dev/tenstorrent/{device.chip}` as a
/// [`Session`] computing on `tile` through `route` at `fidelity`.
pub fn kmd_engine(
    device: TtDevice,
    tile: TileChoice,
    route: SrcRoute,
    fidelity: Fidelity,
) -> impl FnOnce(Serve) -> Result<(), EngineError> + Send + 'static {
    kmd_engine_with_elementwise(device, tile, route, fidelity, ElementwiseMode::Sfpu)
}

/// Silicon factory with an explicit immutable arithmetic selection.
pub fn kmd_engine_with_elementwise(
    device: TtDevice,
    tile: TileChoice,
    route: SrcRoute,
    fidelity: Fidelity,
    elementwise: ElementwiseMode,
) -> impl FnOnce(Serve) -> Result<(), EngineError> + Send + 'static {
    move |serve| {
        let mut session = Session::open_card(device.chip, tt_firmware_images::ROLES, tile)?;
        // Tensors live in GDDR (Phase 9). Bit-identical to the host-staged path
        // (`step18_dram_matmul`), so on by default.
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .map_err(|e| EngineError(e.to_string()))?;
        // `TT_ELTWISE` once chose between the SFPU and the data mover's FP32
        // unit; the mover does no arithmetic now. Refused rather than ignored,
        // so a script that still sets it learns why it no longer does anything.
        if let Ok(v) = std::env::var("TT_ELTWISE") {
            return Err(EngineError(format!(
                "TT_ELTWISE={v}: element-wise ops always run on the SFPU now; unset it"
            )));
        }
        // `TT_PIPELINE=0`: matmuls, element-wise ops and reductions run their
        // blocks and runs one after another rather than overlapping one's
        // moves with the next one's compute (`Session::set_pipeline`); the
        // bits are the same either way.
        match std::env::var("TT_PIPELINE").as_deref() {
            Err(_) | Ok("1") => {}
            Ok("0") => session.set_pipeline(false),
            Ok(v) => return Err(EngineError(format!("TT_PIPELINE={v}: expected 0 or 1"))),
        }
        // `TT_HOST_DMA=0`: tensors cross PCIe by the host's own stores and
        // loads through a BAR, not the card's DMA through pinned host memory
        // (`Session::set_host_dma`).
        match std::env::var("TT_HOST_DMA").as_deref() {
            Err(_) | Ok("1") => {}
            Ok("0") => session.set_host_dma(false),
            Ok(v) => return Err(EngineError(format!("TT_HOST_DMA={v}: expected 0 or 1"))),
        }
        // `TT_TILIZE=card`: tensors take the tile layout on the card's movers
        // rather than the host's cores (`Session::set_tilize`); `host`, the
        // default, is faster on every size card 0 measured.
        match std::env::var("TT_TILIZE").as_deref() {
            Err(_) | Ok("host") => {}
            Ok("card") => session.set_tilize(tt_kernels::session::Tilize::Card),
            Ok(v) => return Err(EngineError(format!("TT_TILIZE={v}: expected host or card"))),
        }
        refuse_retired_settings(|variable| std::env::var(variable).ok())?;
        // `TT_PROFILE=<path>`: a device-side profile of everything this
        // attachment runs, written as Chrome trace JSON when it detaches
        // (`tt_kernels::profile`). `{chip}` in the path becomes the card.
        let profile_to = std::env::var("TT_PROFILE")
            .ok()
            .map(|p| p.replace("{chip}", &device.chip.to_string()));
        if profile_to.is_some() {
            session
                .profile_start()
                .map_err(|e| EngineError(format!("TT_PROFILE: {e}")))?;
        }
        let mut engine = KmdEngine {
            session,
            route,
            fidelity,
            budget: 400_000,
            buffers: Some(DramBuffers::default()),
        }
        .with_elementwise_mode(elementwise)?;
        serve.serve(&mut engine);
        if let Some(path) = profile_to {
            let written = engine
                .session
                .profile_stop()
                .map_err(|e| e.to_string())
                .and_then(|p| p.to_chrome_trace().map_err(|e| e.to_string()))
                .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()));
            match written {
                Ok(()) => eprintln!("burn-tt: device profile written to {path}"),
                Err(e) => eprintln!("burn-tt: TT_PROFILE: {e}"),
            }
        }
        Ok(())
    }
}

// --- Several chips --------------------------------------------------------------

/// An engine over a [`Fabric`](tt_kernels::shard::Fabric): every matmul split
/// along `N` across the fabric's chips, bit-identical to the single-chip
/// product. Attached as *one* Burn device -- the chips are how it computes, not
/// something Burn sees.
pub struct MeshEngine<T: tt_device::Transport> {
    pub fabric: tt_kernels::shard::Fabric<T>,
    pub route: SrcRoute,
    pub fidelity: Fidelity,
    pub budget: u64,
    buffers: DramBuffers,
}

impl<T: tt_device::Transport> MeshEngine<T> {
    pub fn with_elementwise_mode(self, mode: ElementwiseMode) -> Result<Self, EngineError> {
        if mode != ElementwiseMode::Sfpu {
            return Err(EngineError(
                "matrix elementwise mode is unsupported on mesh engines".into(),
            ));
        }
        Ok(self)
    }

    pub fn new(
        mut fabric: tt_kernels::shard::Fabric<T>,
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<Self, EngineError> {
        fabric
            .enable_resident(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(Self {
            fabric,
            route,
            fidelity,
            budget,
            buffers: DramBuffers::default(),
        })
    }
}

impl<T: tt_device::Transport> Engine for MeshEngine<T> {
    fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        [m, k, n]: [usize; 3],
    ) -> Result<Vec<f32>, EngineError> {
        let mut temporary = Vec::new();
        let result = (|| {
            let a = self.upload(a, m, k)?;
            temporary.push(a);
            let b = self.upload(b, k, n)?;
            temporary.push(b);
            let (out, _) = self.matmul_dram(a, false, b, false)?;
            temporary.push(out);
            self.download(out)
        })();
        for t in temporary {
            self.free(t);
        }
        result
    }
    fn full_reduce(
        &mut self,
        values: &[f32],
        [r, c]: [usize; 2],
        mean: bool,
    ) -> Result<f32, EngineError> {
        use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
        let mut temporary = Vec::new();
        let result = (|| {
            let input = self.upload(values, r, c)?;
            temporary.push(input);
            let (column, _) = self.reduce(input, ReduceOp::Sum, Axis::Cols)?;
            temporary.push(column);
            let (mut scalar, _) = self.reduce(column, ReduceOp::Sum, Axis::Rows)?;
            temporary.push(scalar);
            if mean {
                (scalar, _) = self.eltwise(
                    tt_kernels::sfpu::ops::kind_sfpu::DIV_SCALAR,
                    (r * c) as f32,
                    scalar,
                    None,
                )?;
                temporary.push(scalar);
            }
            Ok(self.download(scalar)?[0])
        })();
        for t in temporary {
            self.free(t);
        }
        result
    }
    fn supports_dram(&self) -> bool {
        true
    }
    fn upload_bf16(
        &mut self,
        bits: &[u16],
        rows: usize,
        cols: usize,
    ) -> Result<BufferId, EngineError> {
        self.buffers
            .upload_bf16(self.fabric.chips[0].session(), bits, rows, cols)
    }
    fn download_bf16(&mut self, id: BufferId) -> Result<Vec<u16>, EngineError> {
        self.buffers
            .download_bf16(self.fabric.chips[0].session(), id)
    }
    fn cast_float(
        &mut self,
        id: BufferId,
        bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers
            .cast_float(self.fabric.chips[0].session(), id, bf16)
    }
    fn pool_bf16(
        &mut self,
        a: BufferId,
        windows: &[Vec<[usize; 2]>],
        divisors: &[usize],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers
            .pool_bf16(self.fabric.chips[0].session(), a, windows, divisors, dims)
    }
    fn matmul_bf16(
        &mut self,
        a: BufferId,
        b: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        // The existing Ethernet fabric transports F32 physical tiles. Widen
        // on the root chip, compute BF16 Src products across the mesh, and
        // let the Burn boundary narrow the joined F32 output once.
        let mut temporary = Vec::new();
        let result = (|| {
            let (a, _) = self.cast_float(a, false)?;
            temporary.push(a);
            let (b, _) = self.cast_float(b, false)?;
            temporary.push(b);
            let a = self.buffers.get(a)?.clone();
            let b = self.buffers.get(b)?.clone();
            let out = self
                .fabric
                .matmul_resident(
                    &a,
                    false,
                    &b,
                    false,
                    SrcRoute::Bf16FromFp32,
                    self.fidelity,
                    self.budget,
                )
                .map_err(|e| EngineError(e.to_string()))?;
            Ok(self.buffers.insert(out))
        })();
        for id in temporary {
            self.buffers.free(self.fabric.chips[0].session(), id);
        }
        result
    }
    fn upload(&mut self, v: &[f32], rows: usize, cols: usize) -> Result<BufferId, EngineError> {
        let b = &mut self.buffers;
        b.upload(self.fabric.chips[0].session(), v, rows, cols)
    }
    fn metadata(
        &mut self,
        bits: &[u32],
        dims: [usize; 2],
        elem: Elem,
    ) -> Result<BufferId, EngineError> {
        self.buffers
            .metadata(self.fabric.chips[0].session(), bits, dims, elem)
    }
    fn download(&mut self, id: BufferId) -> Result<Vec<f32>, EngineError> {
        let b = &mut self.buffers;
        b.download(self.fabric.chips[0].session(), id)
    }
    fn upload_bits(
        &mut self,
        v: &[u32],
        rows: usize,
        cols: usize,
        elem: Elem,
    ) -> Result<BufferId, EngineError> {
        let b = &mut self.buffers;
        b.upload_bits(self.fabric.chips[0].session(), v, rows, cols, elem)
    }
    fn download_bits(&mut self, id: BufferId) -> Result<Vec<u32>, EngineError> {
        let b = &mut self.buffers;
        b.download_bits(self.fabric.chips[0].session(), id)
    }
    fn free(&mut self, id: BufferId) {
        self.buffers.free(self.fabric.chips[0].session(), id);
    }
    fn matmul_dram(
        &mut self,
        a: BufferId,
        ta: bool,
        b: BufferId,
        tb: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let a = self.buffers.get(a)?.clone();
        let b = self.buffers.get(b)?.clone();
        let out = self
            .fabric
            .matmul_resident(&a, ta, &b, tb, self.route, self.fidelity, self.budget)
            .map_err(|e| EngineError(e.to_string()))?;
        Ok(self.buffers.insert(out))
    }
    fn eltwise(
        &mut self,
        kind: u32,
        scalar: f32,
        a: BufferId,
        b: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = &mut self.buffers;
        bufs.eltwise(self.fabric.chips[0].session(), kind, scalar, a, b)
    }
    fn eltwise_op(
        &mut self,
        op: tt_kernels::tensor::Eltwise,
        a: BufferId,
        b: Option<BufferId>,
        c: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = &mut self.buffers;
        bufs.eltwise_op(self.fabric.chips[0].session(), op, a, b, c)
    }
    fn pow(&mut self, x: BufferId, y: PowArg) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = &mut self.buffers;
        bufs.pow(self.fabric.chips[0].session(), x, y)
    }
    fn sum_rows(&mut self, a: BufferId) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = &mut self.buffers;
        bufs.sum_rows(self.fabric.chips[0].session(), a)
    }
    fn reduce(
        &mut self,
        a: BufferId,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        axis: tt_kernels::sfpu::reduce::Axis,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = &mut self.buffers;
        bufs.reduce(self.fabric.chips[0].session(), a, op, axis)
    }
    fn scan(
        &mut self,
        a: BufferId,
        op: tt_kernels::sfpu::scan::ScanOp,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers.scan(self.fabric.chips[0].session(), a, op)
    }
    fn slice_rows(
        &mut self,
        a: BufferId,
        first: usize,
        rows: usize,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = &mut self.buffers;
        bufs.slice_rows(a, first, rows)
    }
    fn matmul_dram_batched(
        &mut self,
        a: BufferId,
        b: BufferId,
        items: &[(Block, Block)],
        mkn: [usize; 3],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let [m, k, n] = mkn;
        if items.is_empty() || mkn.contains(&0) {
            return Err(EngineError(
                "mesh batched matmul requires nonempty products".into(),
            ));
        }
        let fits = |id, block: Block, rows, cols| -> Result<(), EngineError> {
            let source = self.buffers.get(id)?;
            let [rows, cols] = if block.transposed {
                [cols, rows]
            } else {
                [rows, cols]
            };
            if block.at[0]
                .checked_add(rows)
                .is_none_or(|end| end > source.rows)
                || block.at[1]
                    .checked_add(cols)
                    .is_none_or(|end| end > source.cols)
            {
                return Err(EngineError(
                    "mesh batched matrix block is out of bounds".into(),
                ));
            }
            Ok(())
        };
        for &(ba, bb) in items {
            fits(a, ba, m, k)?;
            fits(b, bb, k, n)?;
        }
        let mut temporary = Vec::new();
        let result = (|| {
            let mut outputs = Vec::with_capacity(items.len());
            for &(ba, bb) in items {
                let mapping = |block: Block, rows: usize, cols: usize| {
                    (0..rows)
                        .flat_map(move |r| {
                            (0..cols).map(move |c| {
                                if block.transposed {
                                    [block.at[0] + c, block.at[1] + r]
                                } else {
                                    [block.at[0] + r, block.at[1] + c]
                                }
                            })
                        })
                        .collect::<Vec<_>>()
                };
                let (pa, _) = self.repack(a, &mapping(ba, m, k), [m, k])?;
                temporary.push(pa);
                let (pb, _) = self.repack(b, &mapping(bb, k, n), [k, n])?;
                temporary.push(pb);
                // Every batch product uses the column partitions and resident
                // Ethernet transfers, including the backward products.
                let (out, _) = self.matmul_dram(pa, false, pb, false)?;
                // Retain only outputs across batches. Session defers operand
                // frees until their queued work retires.
                temporary.pop();
                self.free(pb);
                temporary.pop();
                self.free(pa);
                temporary.push(out);
                outputs.push(out);
            }
            let rows = (0..items.len())
                .flat_map(|batch| (0..m).map(move |r| (batch, r)))
                .collect::<Vec<_>>();
            self.gather_rows(&outputs, &rows, n)
        })();
        for id in temporary {
            self.free(id);
        }
        result
    }
    fn copy_blocks(
        &mut self,
        a: BufferId,
        moves: &[BlockMove],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = &mut self.buffers;
        bufs.copy_blocks(self.fabric.chips[0].session(), a, moves, dims)
    }
    fn repack(
        &mut self,
        a: BufferId,
        sources: &[[usize; 2]],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers
            .repack(self.fabric.chips[0].session(), a, sources, dims)
    }
    fn zeros_dram(
        &mut self,
        dims: [usize; 2],
        elem: Elem,
        bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers
            .zeros(self.fabric.chips[0].session(), dims, elem, bf16)
    }
    fn gather_indexed(
        &mut self,
        input: BufferId,
        indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers
            .gather_indexed(self.fabric.chips[0].session(), input, indices)
    }
    fn repack_many(
        &mut self,
        inputs: &[BufferId],
        sources: &[(usize, [usize; 2])],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        self.buffers
            .repack_many(self.fabric.chips[0].session(), inputs, sources, dims)
    }

    fn gather_rows(
        &mut self,
        sources: &[BufferId],
        rows: &[(usize, usize)],
        cols: usize,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = &mut self.buffers;
        bufs.gather_rows(self.fabric.chips[0].session(), sources, rows, cols)
    }
    fn rows_add(
        &mut self,
        t: BufferId,
        indices: &[usize],
        value: BufferId,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = &mut self.buffers;
        bufs.rows_add(self.fabric.chips[0].session(), t, indices, value)
    }
    fn mesh_execution(&mut self) -> Option<tt_kernels::shard::FabricExecution> {
        Some(self.fabric.execution().clone())
    }
    fn device_traffic(&mut self) -> Option<tt_device::Traffic> {
        Some(self.fabric.chips[0].device().traffic())
    }

    // lane:t1_intbool (MeshEngine): add this lane's methods below this line only.

    // lane:t2_index (MeshEngine): add this lane's methods below this line only.

    // lane:t3_scan (MeshEngine): add this lane's methods below this line only.

    // lane:t4_rem (MeshEngine): add this lane's methods below this line only.

    // lane:t5_sort (MeshEngine): add this lane's methods below this line only.
    fn sort_planes(
        &mut self,
        a: BufferId,
        spec: tt_kernels::sfpu::sort::Spec,
    ) -> Result<SortedBuffers, EngineError> {
        self.buffers
            .sort_planes(self.fabric.chips[0].session(), a, spec)
    }

    // lane:t6_random (MeshEngine): add this lane's methods below this line only.

    // lane:t7_dtype (MeshEngine): add this lane's methods below this line only.

    // lane:t8_mathmode (MeshEngine): add this lane's methods below this line only.

    // lane:t9_mesh (MeshEngine): add this lane's methods below this line only.
    /// Capture a mesh trace (`tt_kernels::mesh_trace`): one session trace per
    /// chip per stretch between Ethernet transfers, and the transfers
    /// themselves, replayed in the capture's order.
    fn begin_trace(&mut self) -> Result<(), EngineError> {
        self.fabric.begin_mesh_trace().map_err(mesh_trace_error)
    }
    fn end_trace(&mut self) -> Result<u64, EngineError> {
        self.fabric.end_mesh_trace().map_err(mesh_trace_error)
    }
    fn run_trace(
        &mut self,
        trace: u64,
        input: BufferId,
        values: &[f32],
        output: BufferId,
    ) -> Result<TraceRun, EngineError> {
        use std::time::Instant;
        let e = |e: tt_kernels::tensor::TensorError| EngineError(e.to_string());
        // The write waits for what was queued first: that wait is not the
        // write's.
        self.fabric.chips[0].session().sync().map_err(e)?;
        let t0 = Instant::now();
        self.fabric.chips[0]
            .session()
            .write(self.buffers.get(input)?, values)
            .map_err(e)?;
        let t1 = Instant::now();
        self.fabric
            .replay_mesh_trace(trace)
            .map_err(mesh_trace_error)?;
        let t2 = Instant::now();
        let output = self
            .buffers
            .download(self.fabric.chips[0].session(), output)?;
        Ok(TraceRun {
            output,
            write: t1 - t0,
            replay: t2 - t1,
            read: t2.elapsed(),
        })
    }
    fn release_trace(&mut self, trace: u64) {
        let _ = self.fabric.release_mesh_trace(trace);
    }
    fn run_generic_trace(
        &mut self,
        trace: u64,
        inputs: &[(BufferId, InputPayload)],
        outputs: &[(BufferId, OutputKind)],
    ) -> Result<GenericTraceRun, EngineError> {
        use std::time::Instant;
        let e = |e: tt_kernels::tensor::TensorError| EngineError(e.to_string());
        self.fabric.chips[0].session().sync().map_err(e)?;
        let t0 = Instant::now();
        for (input_id, payload) in inputs {
            let t = self.buffers.get(*input_id)?;
            let session = self.fabric.chips[0].session();
            match payload {
                InputPayload::F32(v) => session.write(t, v).map_err(e)?,
                InputPayload::Bits(v) => session.write_bits(t, v).map_err(e)?,
            }
        }
        let t1 = Instant::now();
        self.fabric
            .replay_mesh_trace(trace)
            .map_err(mesh_trace_error)?;
        let t2 = Instant::now();
        let mut out_data = Vec::with_capacity(outputs.len());
        for &(output_id, kind) in outputs {
            match kind {
                OutputKind::F32 => {
                    let data = self
                        .buffers
                        .download(self.fabric.chips[0].session(), output_id)?;
                    out_data.push(OutputPayload::F32(data));
                }
                OutputKind::Bits => {
                    let t = self.buffers.get(output_id)?;
                    let data = self.fabric.chips[0].session().download_bits(t).map_err(e)?;
                    out_data.push(OutputPayload::Bits(data));
                }
            }
        }
        Ok(GenericTraceRun {
            outputs: out_data,
            write: t1 - t0,
            replay: t2 - t1,
            read: t2.elapsed(),
        })
    }
}

fn mesh_trace_error(e: tt_kernels::mesh_trace::MeshTraceError) -> EngineError {
    EngineError(e.to_string())
}

/// A factory for [`attach`] that opens every card in `cards` (the first is chip
/// 0, the host's way in and out), computes on `compute` and relays through
/// `relay` on each, finds the cabled links between them from the chips
/// (`tt_kernels::link::discover`), and serves a [`MeshEngine`].
pub fn kmd_mesh_engine(
    cards: Vec<u16>,
    compute: (u8, u8),
    relay: (u8, u8),
    route: SrcRoute,
    fidelity: Fidelity,
) -> impl FnOnce(Serve) -> Result<(), EngineError> + Send + 'static {
    use tt_device::tlb::WindowKind;
    use tt_kernels::shard::{Chip, Fabric};
    move |serve| {
        refuse_retired_settings(|variable| std::env::var(variable).ok())?;
        let e = |e: tt_device::TransportError| EngineError(e.to_string());
        let mut chips = Vec::new();
        for &card in &cards {
            let s = Session::open_card(
                card,
                tt_firmware_images::ROLES,
                TileChoice::Exactly(compute.0, compute.1),
            )?;
            if !s.grid().contains(relay.0, relay.1) {
                return Err(EngineError(format!(
                    "relay tile {relay:?} is fused off on card {card}"
                )));
            }
            let tile = s.tile();
            let relay = tt_isa::noc::NocCoord::new(relay.0, relay.1).expect("a Tensix coordinate");
            chips.push(Chip::new(s.into_device(), tile, relay).map_err(e)?);
        }
        let mut links = Vec::new();
        for p in 0..chips.len() {
            for q in p + 1..chips.len() {
                let (l, r) = chips.split_at_mut(q);
                let (a, b) = (l[p].device(), r[0].device());
                let wa = a.alloc_window(WindowKind::TwoMib).map_err(e)?;
                let wb = b.alloc_window(WindowKind::TwoMib).map_err(e)?;
                let (ga, gb) = (
                    a.ethernet_grid(&wa).map_err(e)?,
                    b.ethernet_grid(&wb).map_err(e)?,
                );
                let found = tt_kernels::link::discover(a, &wa, &ga, b, &wb, &gb)
                    .map_err(|x| EngineError(x.to_string()))?;
                if let Some(link) = found.first() {
                    links.push((p, q, *link));
                }
            }
        }
        let fabric = Fabric::new(
            chips,
            &links,
            tt_firmware_images::ROLES,
            tt_firmware_images::ETH_E1,
        )
        .map_err(|x| EngineError(x.to_string()))?;
        let mut engine = MeshEngine::new(fabric, route, fidelity, 400_000)?;
        serve.serve(&mut engine);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::refuse_retired_settings;

    #[test]
    fn arithmetic_mode_is_an_attachment_snapshot_without_per_op_server_queries() {
        use super::*;
        struct Mock {
            mode: ElementwiseMode,
            queries: Arc<AtomicU64>,
        }
        impl Engine for Mock {
            fn elementwise_mode(&self) -> ElementwiseMode {
                self.queries.fetch_add(1, Ordering::Relaxed);
                self.mode
            }
            fn matmul(
                &mut self,
                _a: &[f32],
                _b: &[f32],
                _dims: [usize; 3],
            ) -> Result<Vec<f32>, EngineError> {
                Err(unsupported())
            }
        }
        let device = TtDevice::new(u16::MAX);
        let mode = ElementwiseMode::Matrix {
            precision: SrcPrecision::Tf32,
            fidelity: Fidelity::HiFi4,
        };
        for mode in [mode, ElementwiseMode::Sfpu] {
            let queries = Arc::new(AtomicU64::new(0));
            let q = queries.clone();
            let guard = attach(device, move |serve| {
                serve.serve(&mut Mock { mode, queries: q });
                Ok(())
            })
            .unwrap();
            for _ in 0..3 {
                assert_eq!(elementwise_mode(device), mode);
            }
            assert_eq!(
                queries.load(Ordering::Relaxed),
                1,
                "configuration queries must not serialize queued arithmetic"
            );
            drop(guard);
        }
    }

    #[test]
    fn retired_executor_settings_are_refused_with_what_replaces_them() {
        for (variable, value) in [
            ("TT_EXECUTION", "legacy"),
            ("TT_EXECUTION", "streaming"),
            ("TT_SCATTER", "b"),
            ("TT_SCATTER", "nc"),
        ] {
            let error = refuse_retired_settings(|name| (name == variable).then(|| value.into()))
                .unwrap_err()
                .0;
            assert!(
                error.starts_with(&format!("{variable}={value}: retired")),
                "{error}"
            );
            assert!(error.contains("TT_PIPELINE=0"), "{error}");
        }
        assert!(refuse_retired_settings(|_| None).is_ok());
        let error = refuse_retired_settings(|name| (name == "TT_EXACT").then(|| "1".into()))
            .unwrap_err()
            .0;
        assert!(error.contains("TT_EXACT=1: retired"), "{error}");
        assert!(
            error.contains("native numerical bounds") && error.contains("unset"),
            "{error}"
        );
    }
}
