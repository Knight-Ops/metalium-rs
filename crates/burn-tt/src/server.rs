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
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use tt_kernels::tensor::{Block, BlockMove};

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::matrix_eltwise::{ElementwiseMode, MatrixEltwiseOp, SrcPrecision};
use tt_kernels::session::{BatchedLayout, Session, TileChoice};

use crate::error::TtError;
use crate::{MatmulMode, TtDevice};
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
    /// How matmuls accumulate on this engine, fixed for its attachment: the
    /// fast FPU path unless the engine was built for exact F32.
    fn matmul_mode(&self) -> MatmulMode {
        MatmulMode::Fast
    }
    /// Can [`MatmulMode::Exact`] run here? The exact path composes element-wise ops and
    /// reductions, which a mesh runs on chip 0 only, so a mesh says no.
    fn supports_exact_matmul(&self) -> bool {
        true
    }
    /// How this engine's transcendentals run (`MathMode`, S10):
    /// `Precise` unless [`Engine::set_math_mode`] (or `TT_MATH=approx` when
    /// its session opened) chose `Approx`.
    fn math_mode(&self) -> tt_kernels::sfpu::approx::MathMode {
        tt_kernels::sfpu::approx::MathMode::Precise
    }
    /// Run the transcendentals in `mode` from the next op on. An engine with
    /// no device-resident element-wise ops refuses, rather than ignoring it.
    fn set_math_mode(&mut self, mode: tt_kernels::sfpu::approx::MathMode) -> Result<(), TtError> {
        Err(TtError::unsupported(format!(
            "this engine has no math modes (asked for {mode:?})"
        )))
    }
    /// `A[m, k] @ B[k, n]`, row-major.
    fn matmul(&mut self, a: &[f32], b: &[f32], mkn: [usize; 3]) -> Result<Vec<f32>, TtError>;

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
    ) -> Result<f32, TtError> {
        Err(TtError::unsupported(
            "native full reduction is unsupported by this engine",
        ))
    }
    /// Put a row-major `[rows, cols]` matrix on the device.
    fn upload(&mut self, _values: &[f32], _rows: usize, _cols: usize) -> Result<BufferId, TtError> {
        Err(unsupported())
    }
    /// Materialize operation geometry from replayable mover immediates.
    fn metadata(
        &mut self,
        _bits: &[u32],
        _dims: [usize; 2],
        _elem: Elem,
    ) -> Result<BufferId, TtError> {
        Err(unsupported())
    }
    /// Read one back, row-major.
    fn download(&mut self, _id: BufferId) -> Result<Vec<f32>, TtError> {
        Err(unsupported())
    }
    fn upload_bf16(
        &mut self,
        _bits: &[u16],
        _rows: usize,
        _cols: usize,
    ) -> Result<BufferId, TtError> {
        Err(unsupported())
    }
    fn download_bf16(&mut self, _id: BufferId) -> Result<Vec<u16>, TtError> {
        Err(unsupported())
    }
    fn pool_bf16(
        &mut self,
        _a: BufferId,
        _windows: &[Vec<[usize; 2]>],
        _divisors: &[usize],
        _dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    fn matmul_bf16(
        &mut self,
        _a: BufferId,
        _b: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// Convert between resident F32 and physical BF16 storage on Tensix.
    fn cast_float(
        &mut self,
        _id: BufferId,
        _bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// Explicit physical BFP storage conversion. None widens to F32.
    fn cast_bfp(
        &mut self,
        _id: BufferId,
        _format: Option<tt_kernels::bfp::BfpFormat>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    fn matmul_bfp(
        &mut self,
        _a: BufferId,
        _b: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
    ) -> Result<BufferId, TtError> {
        Err(unsupported())
    }
    /// Read any buffer back as its datums' bits, row-major.
    fn download_bits(&mut self, _id: BufferId) -> Result<Vec<u32>, TtError> {
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// [`Engine::matmul_dram`] at `fidelity` in place of the engine's own (`MatmulMode::Accurate`
    /// multiplies integer slices at `Fidelity::Lo`). An engine that cannot refuses.
    fn matmul_dram_at(
        &mut self,
        _a: BufferId,
        _a_transposed: bool,
        _b: BufferId,
        _b_transposed: bool,
        _fidelity: Fidelity,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// [`Engine::matmul_dram_batched`] at `fidelity`; see [`Engine::matmul_dram_at`].
    fn matmul_dram_batched_at(
        &mut self,
        _a: BufferId,
        _b: BufferId,
        _items: &[(Block, Block)],
        _mkn: [usize; 3],
        _fidelity: Fidelity,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// Does [`Engine::matmul_dram_batched_layout`] work here? Asked once an
    /// attachment ([`supports_batched_layout`]).
    fn supports_batched_layout(&self) -> bool {
        false
    }
    /// [`Engine::matmul_dram_batched`] with the left operand, the output or
    /// both a product per whole tile rows (`Session::matmul_dram_batched_with`).
    fn matmul_dram_batched_layout(
        &mut self,
        _a: BufferId,
        _b: BufferId,
        _items: &[(Block, Block)],
        _mkn: [usize; 3],
        _layout: BatchedLayout,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// Attention for a few query rows a key/value head, as one op
    /// (`Session::attention_rows`): `[q, k, v, mask]`, each head's blocks in `items`.
    fn attention_rows(
        &mut self,
        _buffers: [BufferId; 4],
        _items: &[tt_kernels::attention::Item],
        _dims: [usize; 4],
        _scale: f32,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    fn copy_blocks(
        &mut self,
        _a: BufferId,
        _moves: &[BlockMove],
        _dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    fn zeros_dram(
        &mut self,
        _dims: [usize; 2],
        _elem: Elem,
        _bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    fn gather_indexed(
        &mut self,
        _input: BufferId,
        _indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    fn select_rows(
        &mut self,
        _table: BufferId,
        _indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// Rows of `table` for each of the `[taps, n]` resident indices' taps, side
    /// by side (`Session::select_taps`): an im2col with no host address.
    fn select_taps(
        &mut self,
        _table: BufferId,
        _indices: BufferId,
        _zero_row: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    fn repack_many(
        &mut self,
        _inputs: &[BufferId],
        _sources: &[(usize, [usize; 2])],
        _dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// Rows of `sources` gathered into a new buffer
    /// (`Session::gather_rows`): rows a list the host built names (a matmul's
    /// output, a broadcast, a pooling window). A lookup by indices the card
    /// holds, an embedding's, is [`Engine::select_rows`].
    fn gather_rows(
        &mut self,
        _sources: &[BufferId],
        _rows: &[(usize, usize)],
        _cols: usize,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// `t` with `value`'s rows added to the rows `indices` name, in order
    /// (`Session::rows_add`): an embedding's gradient.
    fn rows_add(
        &mut self,
        _t: BufferId,
        _indices: &[usize],
        _value: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// A copy of `t` with `value`'s row `s` at row `d` for each `(d, s)`
    /// (`Session::assign_rows`): a `slice_assign` of whole rows.
    fn assign_rows(
        &mut self,
        _t: BufferId,
        _value: BufferId,
        _rows: &[(usize, usize)],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// A copy of `table` with `values`' row `i` at its row `indices[i]`, the
    /// indices a resident I32 buffer, `[1, n]` or (`column`) `[n, 1]`
    /// (`Session::scatter_rows`): a `scatter_nd` with `Assign` over whole rows.
    fn scatter_rows(
        &mut self,
        _table: BufferId,
        _indices: BufferId,
        _values: BufferId,
        _column: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// [`Engine::scatter_rows`] into `table` itself (`Session::scatter_rows_into`).
    fn scatter_rows_into(
        &mut self,
        _table: BufferId,
        _indices: BufferId,
        _values: BlockArg,
        _column: bool,
    ) -> Result<(), TtError> {
        Err(unsupported())
    }
    /// The sum over rows of `a`, `[1, cols]`, left on the device.
    fn sum_rows(&mut self, _a: BufferId) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// `a` reduced over `axis` by `op`, left on the device
    /// (`tt_kernels::session::Session::reduce`).
    fn reduce(
        &mut self,
        _a: BufferId,
        _op: tt_kernels::sfpu::reduce::ReduceOp,
        _axis: tt_kernels::sfpu::reduce::Axis,
        _then: tt_kernels::tensor::ReduceThen,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// Inclusive scan down matrix rows, keeping all prefixes resident.
    fn scan(
        &mut self,
        _a: BufferId,
        _op: tt_kernels::sfpu::scan::ScanOp,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// Each row's `sqrt(mean(a a) + eps)`, `[rows, 1]`, left on the device
    /// (`tt_kernels::session::Session::rms`).
    fn rms(&mut self, _a: BufferId, _eps: f32) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// `softmax(scale x, masked)` over each row of `x`, `mask` read for rows `r % mask_rows`,
    /// left on the device (`tt_kernels::session::Session::softmax_rows`).
    fn softmax_rows(
        &mut self,
        _x: BufferId,
        _mask: BufferId,
        _scale: f32,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// `a / sqrt(mean(a a) + eps)` over each row, left on the device
    /// (`tt_kernels::session::Session::rms_norm`).
    fn rms_norm(&mut self, _a: BufferId, _eps: f32) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// The index of a `[1, cols]` row's largest element, `[1, 1]` I32, left on the device
    /// (`tt_kernels::session::Session::argmax_row`).
    fn argmax_row(&mut self, _a: BufferId) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// Rows `[first, first + rows)` of `a` as a view: no copy. The caller
    /// keeps `a` alive while the view is.
    fn slice_rows(
        &mut self,
        _a: BufferId,
        _first: usize,
        _rows: usize,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        if op.scalar2 != 0.0 || c.is_some() {
            return Err(unsupported());
        }
        self.eltwise(op.kind, op.scalar, a, b)
    }
    /// [`Engine::eltwise_op`] over blocks of resident F32 buffers
    /// (`Session::eltwise_blocks`): each operand read where it lies, the
    /// result a new buffer of `a`'s block's dims.
    fn eltwise_blocks(
        &mut self,
        _op: tt_kernels::tensor::Eltwise,
        _a: BlockArg,
        _b: Option<BlockArg>,
        _c: Option<BlockArg>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// Rotary position embedding as one op (`tt_kernels::session::Session::rope`): `x cos +
    /// rotate(x) sin` over blocks whose columns are heads of `head` columns.
    fn rope_blocks(
        &mut self,
        _x: BlockArg,
        _cos: BlockArg,
        _sin: BlockArg,
        _head: usize,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
    /// `x^y` as `powf` (`tt_kernels::session::Session::pow`), result on the
    /// device.
    fn pow(&mut self, _x: BufferId, _y: PowArg) -> Result<(BufferId, [usize; 2]), TtError> {
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
    fn begin_trace(&mut self) -> Result<(), TtError> {
        Err(no_traces())
    }
    /// End it: the trace's number.
    fn end_trace(&mut self) -> Result<u64, TtError> {
        Err(no_traces())
    }
    /// Start capturing a trace that applies the residency plan the last plan pass made, kept
    /// under `key` if it holds (`Session::begin_trace_planned_as`): `false`, starting nothing,
    /// without one.
    fn begin_trace_planned(&mut self, _key: u64) -> Result<bool, TtError> {
        Ok(false)
    }
    /// Start capturing a trace that applies the residency plan kept under `key`
    /// (`Session::begin_trace_cached`): `false`, starting nothing, without one.
    fn begin_trace_cached(&mut self, _key: u64) -> Result<bool, TtError> {
        Ok(false)
    }
    /// Whether the last plan pass made a plan for [`Engine::begin_trace_planned`] to apply
    /// (`Session::has_pending_plan`).
    fn has_pending_plan(&mut self) -> Result<bool, TtError> {
        Ok(false)
    }
    /// Start capturing a trace that, with residency on, plans the next
    /// (`Session::begin_trace_plan_pass`); otherwise [`Engine::begin_trace`].
    fn begin_trace_plan_pass(&mut self) -> Result<(), TtError> {
        self.begin_trace()
    }
    /// Plan every capture from here on for L1 residency, in a region of `bytes` a unit (0: off;
    /// `Session::set_residency`).
    #[doc(hidden)]
    fn set_residency(&mut self, _bytes: u64) -> Result<(), TtError> {
        Err(no_traces())
    }
    /// How many captures the session has begun (`Session::probe_captures`).
    #[doc(hidden)]
    fn probe_captures(&mut self) -> Result<u64, TtError> {
        Err(no_traces())
    }
    /// Every unit's NC waits since last taken, zeroed (`Session::probe_nc_waits`).
    #[doc(hidden)]
    fn probe_nc_waits(&mut self) -> Result<Vec<[u32; 8]>, TtError> {
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
    ) -> Result<TraceRun, TtError> {
        Err(no_traces())
    }
    /// Give a trace back.
    fn release_trace(&mut self, _trace: u64) {}
    /// Name the device work queued from here on: the Burn op (or part) whose job
    /// runs next, kept in a capture's records (`Session::set_op_label`).
    fn set_op_label(&mut self, _label: &'static str) {}
    /// The ops `trace` captured, in order, each with its label and its share of
    /// every unit's stream (`tt_kernels::trace::OpRecord`).
    fn trace_ops(&mut self, _trace: u64) -> Result<Vec<tt_kernels::trace::OpRecord>, TtError> {
        Err(no_traces())
    }
    /// What `trace` costs each unit: its stream's GDDR bytes and the program-cache bytes it
    /// holds, beside what every trace on the unit holds (`Session::trace_footprint`).
    /// Name `trace` for the profile mode's report.
    fn set_trace_name(&mut self, _trace: u64, _name: String) {}
    /// Turn the profile mode on (`Some`) or off (`None`, returning what it recorded):
    /// `Session::profile_mode`.
    fn profile_mode(
        &mut self,
        _options: Option<tt_kernels::profile_mode::ProfileOptions>,
    ) -> Result<Option<crate::profile::Profile>, TtError> {
        Err(TtError::Device(
            "this engine has no profile mode (`TT_PROFILE` is for a single card's KMD engine)"
                .into(),
        ))
    }
    /// What the profile mode has recorded so far.
    fn profile_report(&mut self) -> Result<Option<crate::profile::Profile>, TtError> {
        Ok(None)
    }
    fn trace_footprint(
        &mut self,
        _trace: u64,
    ) -> Result<Vec<Option<tt_kernels::trace::UnitFootprint>>, TtError> {
        Err(no_traces())
    }
    /// [`Engine::run_generic_trace`] with the timestamper on around the replay alone
    /// (`Session::profile_start`): every unit's data mover stamps its entries. The roles
    /// stamp only if the capture was made while profiling, which a capture through
    /// burn-tt never is. Each unit's buffer holds 1,024 events, so this is for small
    /// traces (a model of one layer); a larger one fails its sync. Silicon only.
    #[doc(hidden)]
    fn set_profile_buffer(&mut self, _bytes: Option<u64>) -> Result<(), TtError> {
        Err(no_traces())
    }
    /// GDDR bytes free on the fullest channel and in all (`Session::dram_free_bytes_by_channel`).
    #[doc(hidden)]
    fn gddr_free_bytes(&mut self) -> Result<(u64, u64), TtError> {
        Err(no_traces())
    }
    /// Count what each op of every trace captured from here on moves through GDDR
    /// (`Session::set_count_trace_bytes`), read back with [`Engine::trace_ops`].
    #[doc(hidden)]
    fn set_count_trace_bytes(&mut self, _on: bool) -> Result<(), TtError> {
        Err(no_traces())
    }
    /// Let the traces captured from here on reuse their own tensors' GDDR
    /// (`Session::set_capture_reuse`); the previous setting.
    #[doc(hidden)]
    fn set_capture_reuse(&mut self, _on: bool) -> Result<bool, TtError> {
        Err(no_traces())
    }
    /// Keep `bytes` at the top of each unit's L1 arena for a persistent region in every trace
    /// captured from here on (`Session::set_l1_region`).
    #[doc(hidden)]
    fn set_l1_region(&mut self, _bytes: u64) -> Result<(), TtError> {
        Err(no_traces())
    }
    #[doc(hidden)]
    fn profile_generic_trace(
        &mut self,
        _trace: u64,
        _inputs: &[(BufferId, InputPayload)],
        _outputs: &[(BufferId, OutputKind)],
    ) -> Result<(GenericTraceRun, tt_kernels::profile::DeviceProfile), TtError> {
        Err(no_traces())
    }
    /// Copy `src` into `dst` on the device without allocating a new buffer.
    fn copy_into(&mut self, _src: BufferId, _dst: BufferId) -> Result<(), TtError> {
        Err(unsupported())
    }
    /// Overwrite multiple inputs, replay `trace`, and read multiple outputs back.
    fn run_generic_trace(
        &mut self,
        _trace: u64,
        _inputs: &[(BufferId, InputPayload)],
        _outputs: &[(BufferId, OutputKind)],
    ) -> Result<GenericTraceRun, TtError> {
        Err(no_traces())
    }

    /// Sort every problem of `a`, a plane matrix (`tt_kernels::sfpu::sort`),
    /// along its axis positions: the sorted keys and, if `spec.indices`, the
    /// original indices. Both stay on the device.
    fn sort_planes(
        &mut self,
        _a: BufferId,
        _spec: tt_kernels::sfpu::sort::Spec,
    ) -> Result<SortedBuffers, TtError> {
        Err(unsupported())
    }

    /// A `dims` tensor drawn on the device by the seeded tile kernel
    /// (`tt_kernels::prng`), resident, with no upload.
    fn random_dram(
        &mut self,
        _dims: [usize; 2],
        _draw: crate::random::Draw,
        _base: u64,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }

    /// Convert between resident F32 and physically packed IEEE binary16
    /// (`to_f16`), on Tensix. FP16 buffers share BF16's raw two-byte storage.
    fn cast_f16(
        &mut self,
        _id: BufferId,
        _to_f16: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        Err(unsupported())
    }
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

fn no_traces() -> TtError {
    TtError::unsupported("this engine does not capture traces")
}

fn unsupported() -> TtError {
    TtError::unsupported("this engine keeps no tensors on the device")
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
    /// The profile mode's names for traces, by the session's trace number
    /// ([`DramBuffers::set_trace_name`]), kept after a trace is released.
    trace_names: std::collections::BTreeMap<u64, String>,
    /// While the profile mode is on: each replay's input write, replay and output read, by the
    /// session's trace number.
    replay_times: Option<std::collections::BTreeMap<u64, Vec<[std::time::Duration; 3]>>>,
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        use tt_kernels::kind;
        let operation = match op.kind {
            kind::ADD | kind::ADD_SCALAR => MatrixEltwiseOp::Add,
            kind::SUB => MatrixEltwiseOp::Sub,
            kind::MUL | kind::MUL_SCALAR => MatrixEltwiseOp::Mul,
            _ => return Err(unsupported()),
        };
        let error = |e: tt_kernels::tensor::TensorError| TtError::from(e);
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
            let packed_scalar = packed_scalar.as_ref().map_err(Clone::clone)?;
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let input = self
            .bf16
            .get(&a)
            .ok_or_else(|| TtError::Device(format!("no BF16 buffer {a}")))?;
        let out = s
            .bf16_pool_windows(input, windows, divisors, dims)
            .map_err(TtError::from)?;
        Ok(self.insert(out))
    }
    pub fn matmul_bf16<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        b: BufferId,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let ta = self
            .bf16
            .get(&a)
            .ok_or_else(|| TtError::Device(format!("no BF16 buffer {a}")))?;
        let tb = self
            .bf16
            .get(&b)
            .ok_or_else(|| TtError::Device(format!("no BF16 buffer {b}")))?;
        let out = s
            .matmul_bf16(ta, tb, fidelity, budget)
            .map_err(TtError::from)?;
        Ok(self.insert(out))
    }
    pub fn upload_bf16<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        bits: &[u16],
        rows: usize,
        cols: usize,
    ) -> Result<BufferId, TtError> {
        let t = s.upload_bf16(bits, rows, cols).map_err(TtError::from)?;
        Ok(self.insert_bf16(t).0)
    }

    pub fn download_bf16<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
    ) -> Result<Vec<u16>, TtError> {
        let t = self
            .bf16
            .get(&id)
            .ok_or_else(|| TtError::Device(format!("no BF16 buffer {id}")))?;
        s.download_bf16(t).map_err(TtError::from)
    }

    pub fn cast_float<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
        bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        if bf16 {
            let t = s.bf16_from_f32(self.get(id)?).map_err(TtError::from)?;
            Ok(self.insert_bf16(t))
        } else {
            let t = self
                .bf16
                .get(&id)
                .ok_or_else(|| TtError::Device(format!("no BF16 buffer {id}")))?;
            let t = s.bf16_to_f32(t).map_err(TtError::from)?;
            Ok(self.insert(t))
        }
    }

    pub fn cast_bfp<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
        format: Option<tt_kernels::bfp::BfpFormat>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let error = |e: tt_kernels::tensor::TensorError| TtError::from(e);
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
                .ok_or_else(|| TtError::Device(format!("no BFP buffer {id}")))?;
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let a = self.bfp.get(&a).ok_or_else(unsupported)?;
        let b = self.bfp.get(&b).ok_or_else(unsupported)?;
        let out = s
            .bfp_matmul(a, false, b, false, fidelity)
            .map_err(TtError::from)?;
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
    ) -> Result<(), TtError> {
        s.begin_trace().map_err(TtError::from)
    }

    pub fn begin_trace_planned<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        key: u64,
    ) -> Result<bool, TtError> {
        s.begin_trace_planned_as(key).map_err(TtError::from)
    }

    pub fn begin_trace_cached<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        key: u64,
    ) -> Result<bool, TtError> {
        s.begin_trace_cached(key).map_err(TtError::from)
    }

    pub fn begin_trace_plan_pass<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
    ) -> Result<(), TtError> {
        s.begin_trace_plan_pass().map_err(TtError::from)
    }

    pub fn end_trace<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
    ) -> Result<u64, TtError> {
        let id = s.end_trace().map_err(TtError::from)?;
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
    ) -> Result<TraceRun, TtError> {
        use std::time::Instant;
        let id = *self
            .traces
            .get(&trace)
            .ok_or_else(|| TtError::Device(format!("no trace {trace}")))?;
        // The write waits for what was queued first: that wait is not the
        // write's.
        s.sync().map_err(TtError::from)?;
        let t0 = Instant::now();
        s.write(self.get(input)?, values).map_err(TtError::from)?;
        let t1 = Instant::now();
        s.replay(id).map_err(TtError::from)?;
        s.sync().map_err(TtError::from)?;
        let t2 = Instant::now();
        let output = self.download(s, output)?;
        Ok(TraceRun {
            output,
            write: t1 - t0,
            replay: t2 - t1,
            read: t2.elapsed(),
        })
    }

    pub fn trace_ops<T: tt_device::Transport>(
        &self,
        s: &Session<T>,
        trace: u64,
    ) -> Result<Vec<tt_kernels::trace::OpRecord>, TtError> {
        let id = *self
            .traces
            .get(&trace)
            .ok_or_else(|| TtError::Device(format!("no trace {trace}")))?;
        Ok(s.trace_ops(id).map_err(TtError::from)?.to_vec())
    }

    pub fn trace_footprint<T: tt_device::Transport>(
        &self,
        s: &Session<T>,
        trace: u64,
    ) -> Result<Vec<Option<tt_kernels::trace::UnitFootprint>>, TtError> {
        let id = *self
            .traces
            .get(&trace)
            .ok_or_else(|| TtError::Device(format!("no trace {trace}")))?;
        s.trace_footprint(id).map_err(TtError::from)
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
    ) -> Result<(), TtError> {
        if let Some(src_t) = self.bfp.get(&src) {
            let dst_t = self
                .bfp
                .get(&dst)
                .ok_or_else(|| TtError::Device("copy_into storage formats differ".into()))?;
            return s.copy_into_bfp(src_t, dst_t).map_err(TtError::from);
        }
        if self.bfp.contains_key(&dst) {
            return Err(TtError::Device("copy_into storage formats differ".into()));
        }
        if let Some(src_t) = self.bf16.get(&src) {
            let dst_t = self
                .bf16
                .get(&dst)
                .ok_or_else(|| TtError::Device("copy_into storage formats differ".into()))?;
            return s.copy_into_bf16(src_t, dst_t).map_err(TtError::from);
        }
        if self.bf16.contains_key(&dst) {
            return Err(TtError::Device("copy_into storage formats differ".into()));
        }
        let src_t = self.get(src)?;
        let dst_t = self.get(dst)?;
        s.copy_into(src_t, dst_t).map_err(TtError::from)
    }

    pub fn run_generic_trace<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        trace: u64,
        inputs: &[(BufferId, InputPayload)],
        outputs: &[(BufferId, OutputKind)],
    ) -> Result<GenericTraceRun, TtError> {
        use std::time::Instant;
        let id = *self
            .traces
            .get(&trace)
            .ok_or_else(|| TtError::Device(format!("no trace {trace}")))?;
        s.sync().map_err(TtError::from)?;
        let t0 = Instant::now();
        self.write_trace_inputs(s, inputs)?;
        let t1 = Instant::now();
        s.replay(id).map_err(TtError::from)?;
        s.sync().map_err(TtError::from)?;
        let t2 = Instant::now();
        let out_data = self.read_trace_outputs(s, outputs)?;
        let run = GenericTraceRun {
            outputs: out_data,
            write: t1 - t0,
            replay: t2 - t1,
            read: t2.elapsed(),
        };
        if let Some(times) = self.replay_times.as_mut() {
            times
                .entry(id.raw())
                .or_default()
                .push([run.write, run.replay, run.read]);
        }
        Ok(run)
    }

    /// Name trace `trace` for the profile mode's report.
    pub fn set_trace_name(&mut self, trace: u64, name: String) {
        if let Some(id) = self.traces.get(&trace) {
            self.trace_names.insert(id.raw(), name);
        }
    }

    /// Turn the profile mode on or off (`Session::profile_mode`), keeping each replay's timing
    /// while it is on; off, what it recorded, with the traces' names.
    pub fn profile_mode<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        options: Option<tt_kernels::profile_mode::ProfileOptions>,
    ) -> Result<Option<crate::profile::Profile>, TtError> {
        let on = options.is_some();
        let device = s.profile_mode(options).map_err(TtError::from)?;
        let profile = device.map(|device| self.profile(s, device));
        self.replay_times = on.then(Default::default);
        Ok(profile)
    }

    /// What the profile mode has recorded so far.
    pub fn profile_report<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
    ) -> Result<Option<crate::profile::Profile>, TtError> {
        let device = s.profile_report().map_err(TtError::from)?;
        Ok(device.map(|device| self.profile(s, device)))
    }

    fn profile<T: tt_device::Transport>(
        &self,
        s: &Session<T>,
        device: tt_kernels::profile_mode::ProfileReport,
    ) -> crate::profile::Profile {
        crate::profile::Profile {
            device,
            names: self.trace_names.clone(),
            replays: self.replay_times.clone().unwrap_or_default(),
            host: crate::profile::session_text(s),
        }
    }

    pub fn profile_generic_trace<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        trace: u64,
        inputs: &[(BufferId, InputPayload)],
        outputs: &[(BufferId, OutputKind)],
    ) -> Result<(GenericTraceRun, tt_kernels::profile::DeviceProfile), TtError> {
        use std::time::Instant;
        let id = *self
            .traces
            .get(&trace)
            .ok_or_else(|| TtError::Device(format!("no trace {trace}")))?;
        s.sync().map_err(TtError::from)?;
        let t0 = Instant::now();
        self.write_trace_inputs(s, inputs)?;
        // The inputs' writes land first, so the stamps hold the replay alone.
        s.sync().map_err(TtError::from)?;
        s.profile_start().map_err(TtError::from)?;
        let t1 = Instant::now();
        s.replay(id).map_err(TtError::from)?;
        s.sync().map_err(TtError::from)?;
        let t2 = Instant::now();
        let profile = s.profile_stop().map_err(TtError::from)?;
        let out_data = self.read_trace_outputs(s, outputs)?;
        Ok((
            GenericTraceRun {
                outputs: out_data,
                write: t1 - t0,
                replay: t2 - t1,
                read: t2.elapsed(),
            },
            profile,
        ))
    }

    /// Write each of a generic trace's inputs into its buffer.
    pub fn write_trace_inputs<T: tt_device::Transport>(
        &self,
        s: &mut Session<T>,
        inputs: &[(BufferId, InputPayload)],
    ) -> Result<(), TtError> {
        for (input_id, payload) in inputs {
            let t = self.get(*input_id)?;
            match payload {
                InputPayload::F32(v) => s.write(t, v).map_err(TtError::from)?,
                InputPayload::Bits(v) => s.write_bits(t, v).map_err(TtError::from)?,
            }
        }
        Ok(())
    }

    /// Download each of a generic trace's outputs as it asks.
    pub fn read_trace_outputs<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        outputs: &[(BufferId, OutputKind)],
    ) -> Result<Vec<OutputPayload>, TtError> {
        let mut out_data = Vec::with_capacity(outputs.len());
        for &(output_id, kind) in outputs {
            match kind {
                OutputKind::F32 => out_data.push(OutputPayload::F32(self.download(s, output_id)?)),
                OutputKind::Bits => {
                    let data = s
                        .download_bits(self.get(output_id)?)
                        .map_err(TtError::from)?;
                    out_data.push(OutputPayload::Bits(data));
                }
            }
        }
        Ok(out_data)
    }

    pub fn upload<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<BufferId, TtError> {
        let t = s.upload(values, rows, cols).map_err(TtError::from)?;
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
    ) -> Result<BufferId, TtError> {
        let t = s.metadata(bits, dims, elem).map_err(TtError::from)?;
        Ok(self.insert(t).0)
    }

    pub fn download<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
    ) -> Result<Vec<f32>, TtError> {
        if let Some(t) = self.bfp.get(&id) {
            return s.download_bfp(t).map_err(TtError::from);
        }
        let t = self.get(id)?;
        s.download(t).map_err(TtError::from)
    }

    pub fn pow<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        x: BufferId,
        y: PowArg,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
        let c = s.pow(&tx, exp).map_err(TtError::from)?;
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
    ) -> Result<BufferId, TtError> {
        let t = s
            .upload_bits(bits, rows, cols, elem)
            .map_err(TtError::from)?;
        self.next += 1;
        self.live.insert(self.next, t);
        Ok(self.next)
    }

    pub fn download_bits<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
    ) -> Result<Vec<u32>, TtError> {
        let t = self.get(id)?;
        s.download_bits(t).map_err(TtError::from)
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
            .map_err(TtError::from)?;
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
        layout: BatchedLayout,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let (ta, tb) = (self.get(a)?.clone(), self.get(b)?.clone());
        let c = s
            .matmul_dram_batched_with(&ta, &tb, items, mkn, route, fidelity, budget, layout)
            .map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_rows<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        [q, k, v, mask]: [BufferId; 4],
        items: &[tt_kernels::attention::Item],
        dims: [usize; 4],
        scale: f32,
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let [tq, tk, tv, tm] = [q, k, v, mask].map(|id| self.get(id).cloned());
        let (tq, tk, tv, tm) = (tq?, tk?, tv?, tm?);
        let c = s
            .attention_rows(
                &tq, &tk, &tv, &tm, items, dims, scale, route, fidelity, budget,
            )
            .map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn gather_rows<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        sources: &[BufferId],
        rows: &[(usize, usize)],
        cols: usize,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let ts = sources
            .iter()
            .map(|&id| self.get(id).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        let refs: Vec<_> = ts.iter().collect();
        let c = s.gather_rows(&refs, rows, cols).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn rows_add<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        t: BufferId,
        indices: &[usize],
        value: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let (tt, tv) = (self.get(t)?.clone(), self.get(value)?.clone());
        let c = s.rows_add(&tt, indices, &tv).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn assign_rows<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        t: BufferId,
        value: BufferId,
        rows: &[(usize, usize)],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let (tt, tv) = (self.get(t)?.clone(), self.get(value)?.clone());
        let c = s.assign_rows(&tt, &tv, rows).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    /// [`Session::scatter_rows`]: a 32-bit table only, a packed one refused.
    pub fn scatter_rows<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        table: BufferId,
        indices: BufferId,
        values: BufferId,
        column: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        for b in [table, values] {
            if self.bf16.contains_key(&b) || self.bfp.contains_key(&b) {
                return Err(TtError::Device(
                    "a row scatter of a packed (BF16, F16 or BFP) table is not implemented".into(),
                ));
            }
        }
        let (tt, ti, tv) = (
            self.get(table)?.clone(),
            self.get(indices)?.clone(),
            self.get(values)?.clone(),
        );
        let c = s
            .scatter_rows(&tt, &ti, &tv, column)
            .map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn scatter_rows_into<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        table: BufferId,
        indices: BufferId,
        values: BlockArg,
        column: bool,
    ) -> Result<(), TtError> {
        for b in [table, values.id] {
            if self.bf16.contains_key(&b) || self.bfp.contains_key(&b) {
                return Err(TtError::Device(
                    "a row scatter of a packed (BF16, F16 or BFP) table is not implemented".into(),
                ));
            }
        }
        let (tt, ti, tv) = (self.get(table)?, self.get(indices)?, self.get(values.id)?);
        let block = tt_kernels::tensor::BlockOperand {
            tensor: tv,
            at: values.at,
            dims: values.dims,
        };
        s.scatter_rows_into_block(tt, ti, block, column)
            .map_err(TtError::from)
    }

    pub fn repack<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        sources: &[[usize; 2]],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        if let Some(t) = self.bf16.get(&a) {
            let c = s.repack_bf16(t, sources, dims).map_err(TtError::from)?;
            return Ok(self.insert_bf16(c));
        }
        let ta = self.get(a)?.clone();
        let c = s.repack(&ta, sources, dims).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn zeros<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        dims: [usize; 2],
        elem: Elem,
        bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        if bf16 {
            let out = s.zeros_bf16(dims).map_err(TtError::from)?;
            Ok(self.insert_bf16(out))
        } else {
            let count = dims[0]
                .checked_mul(dims[1])
                .ok_or_else(|| TtError::Device("zero shape overflow".into()))?;
            let out = s
                .metadata(&vec![0; count], dims, elem)
                .map_err(TtError::from)?;
            Ok(self.insert(out))
        }
    }

    pub fn gather_indexed<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        input: BufferId,
        indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let indices = self.get(indices)?.clone();
        if let Some(t) = self.bf16.get(&input) {
            let out = s.gather_indexed_bf16(t, &indices).map_err(TtError::from)?;
            Ok(self.insert_bf16(out))
        } else {
            let out = s
                .gather_indexed(self.get(input)?, &indices)
                .map_err(TtError::from)?;
            Ok(self.insert(out))
        }
    }

    /// `table`'s rows chosen by the resident `[1, n]` index tensor `indices`
    /// ([`Session::select_rows`]): a 32-bit table only, a packed one refused.
    pub fn select_rows<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        table: BufferId,
        indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        if self.bf16.contains_key(&table) || self.bfp.contains_key(&table) {
            return Err(TtError::Device(
                "a row select of a packed (BF16, F16 or BFP) table is not implemented".into(),
            ));
        }
        let indices = self.get(indices)?.clone();
        let out = s
            .select_rows(self.get(table)?, &indices)
            .map_err(TtError::from)?;
        Ok(self.insert(out))
    }

    /// `table`'s rows for each tap of the resident `[taps, n]` index tensor
    /// ([`Session::select_taps`]): a 32-bit table only, a packed one refused.
    pub fn select_taps<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        table: BufferId,
        indices: BufferId,
        zero_row: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        if self.bf16.contains_key(&table) || self.bfp.contains_key(&table) {
            return Err(TtError::Device(
                "a tap select of a packed (BF16, F16 or BFP) table is not implemented".into(),
            ));
        }
        let indices = self.get(indices)?.clone();
        let out = s
            .select_taps(self.get(table)?, &indices, zero_row)
            .map_err(TtError::from)?;
        Ok(self.insert(out))
    }

    pub fn repack_many<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        inputs: &[BufferId],
        sources: &[(usize, [usize; 2])],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        if inputs.first().is_some_and(|a| self.bf16.contains_key(a)) {
            let ts = inputs
                .iter()
                .map(|a| {
                    self.bf16
                        .get(a)
                        .ok_or_else(|| TtError::Device("mixed repack storage".into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let out = s
                .repack_many_bf16(&ts, sources, dims)
                .map_err(TtError::from)?;
            return Ok(self.insert_bf16(out));
        }
        let ts = inputs
            .iter()
            .map(|&a| self.get(a))
            .collect::<Result<Vec<_>, _>>()?;
        let out = s.repack_many(&ts, sources, dims).map_err(TtError::from)?;
        Ok(self.insert(out))
    }

    pub fn copy_blocks<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        moves: &[BlockMove],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        if let Some(t) = self.bf16.get(&a) {
            let [rows, cols] = dims;
            let count = rows
                .checked_mul(cols)
                .ok_or_else(|| TtError::Device("BF16 copy shape overflow".into()))?;
            let mut sources = vec![[usize::MAX; 2]; count];
            for m in moves {
                for r in 0..m.extent[0] {
                    for c in 0..m.extent[1] {
                        let [tr, tc] = [m.to[0].checked_add(r), m.to[1].checked_add(c)]
                            .map(|v| v.unwrap_or(usize::MAX));
                        if tr >= rows || tc >= cols || sources[tr * cols + tc] != [usize::MAX; 2] {
                            return Err(TtError::Device(
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
            let c = s.repack_bf16(t, &sources, dims).map_err(TtError::from)?;
            return Ok(self.insert_bf16(c));
        }
        let ta = self.get(a)?.clone();
        let c = s.copy_blocks(&ta, moves, dims).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn eltwise<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        kind: u32,
        scalar: f32,
        a: BufferId,
        b: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
            .map_err(TtError::from)?;
        let dims = [c.rows, c.cols];
        self.next += 1;
        self.live.insert(self.next, c);
        Ok((self.next, dims))
    }

    pub fn eltwise_blocks<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        op: tt_kernels::tensor::Eltwise,
        a: BlockArg,
        b: Option<BlockArg>,
        c: Option<BlockArg>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let block = |x: BlockArg| -> Result<tt_kernels::tensor::BlockOperand<'_>, TtError> {
            Ok(tt_kernels::tensor::BlockOperand {
                tensor: self.get(x.id)?,
                at: x.at,
                dims: x.dims,
            })
        };
        let (ta, tb, tc) = (
            block(a)?,
            b.map(block).transpose()?,
            c.map(block).transpose()?,
        );
        let out = s.eltwise_blocks(op, ta, tb, tc).map_err(TtError::from)?;
        Ok(self.insert(out))
    }

    pub fn rope_blocks<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        x: BlockArg,
        cos: BlockArg,
        sin: BlockArg,
        head: usize,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let block = |b: BlockArg| -> Result<tt_kernels::tensor::BlockOperand<'_>, TtError> {
            Ok(tt_kernels::tensor::BlockOperand {
                tensor: self.get(b.id)?,
                at: b.at,
                dims: b.dims,
            })
        };
        let out = s
            .rope(block(x)?, block(cos)?, block(sin)?, head)
            .map_err(TtError::from)?;
        Ok(self.insert(out))
    }

    pub fn sum_rows<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let ta = self.get(a)?.clone();
        let c = s.sum_rows(&ta).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn reduce<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        axis: tt_kernels::sfpu::reduce::Axis,
        then: tt_kernels::tensor::ReduceThen,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let ta = self.get(a)?.clone();
        let c = s.reduce_then(&ta, op, axis, then).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn rms<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        eps: f32,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let ta = self.get(a)?.clone();
        let c = s.rms(&ta, eps).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn softmax_rows<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        x: BufferId,
        mask: BufferId,
        scale: f32,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let (tx, tm) = (self.get(x)?.clone(), self.get(mask)?.clone());
        let c = s.softmax_rows(&tx, &tm, scale).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn rms_norm<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        eps: f32,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let ta = self.get(a)?.clone();
        let c = s.rms_norm(&ta, eps).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn argmax_row<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let ta = self.get(a)?.clone();
        let c = s.argmax_row(&ta).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    pub fn scan<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        op: tt_kernels::sfpu::scan::ScanOp,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let ta = self.get(a)?.clone();
        let c = s.scan(&ta, op).map_err(TtError::from)?;
        Ok(self.insert(c))
    }

    /// [`Engine::sort_planes`], for engines that serve these buffers.
    pub fn sort_planes<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        spec: tt_kernels::sfpu::sort::Spec,
    ) -> Result<SortedBuffers, TtError> {
        let planes = self.get(a)?.clone();
        let sorted = s.sort_planes(&planes, spec).map_err(TtError::from)?;
        Ok(SortedBuffers {
            keys: self.insert(sorted.keys),
            indices: sorted.indices.map(|t| self.insert(t)),
        })
    }

    /// [`Engine::random_dram`], for engines that serve these buffers.
    pub fn random<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        dims: [usize; 2],
        draw: crate::random::Draw,
        base: u64,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        use crate::random::Draw;
        use tt_kernels::prng::{role, Output};
        let drawn = match draw {
            Draw::Unit => s.random_tiles(dims, Output::Unit, base, role::UNIT),
            Draw::Uniform(lo, hi) => s.random_uniform(dims, base, lo, hi),
            Draw::Bernoulli { p, int } => s.random_bernoulli(dims, base, p, int),
            Draw::Normal {
                mean,
                std,
                int: false,
            } => s.random_normal(dims, base, mean, std),
            Draw::Normal {
                mean,
                std,
                int: true,
            } => s.random_normal_int(dims, base, mean, std),
            Draw::IntRange(lo, hi) => s.random_int_range(dims, base, lo, hi),
            Draw::IntWords => s.random_int_words(dims, base),
        }
        .map_err(TtError::from)?;
        Ok(self.insert(drawn))
    }

    /// FP16 storage shares the raw two-byte `bf16` map, since movement and views
    /// never look at the element format; only these conversions do.
    pub fn cast_f16<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
        to_f16: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let error = |e: tt_kernels::tensor::TensorError| TtError::from(e);
        if to_f16 {
            let t = s.fp16_from_f32(self.get(id)?).map_err(error)?;
            Ok(self.insert_bf16(t.as_raw()))
        } else {
            let raw = self
                .bf16
                .get(&id)
                .ok_or_else(|| TtError::Device(format!("no FP16 buffer {id}")))?;
            let t = tt_kernels::fp16::Fp16Tensor::from_raw(raw.clone());
            let wide = s.fp16_to_f32(&t).map_err(error)?;
            Ok(self.insert(wide))
        }
    }

    /// A view: freeing it frees nothing (`DramTensor::rows_view`).
    pub fn slice_rows(
        &mut self,
        a: BufferId,
        first: usize,
        rows: usize,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        if let Some(t) = self.bfp.get(&a) {
            let v = t.rows_view(first, rows).map_err(TtError::from)?;
            let dims = [v.rows(), v.cols()];
            self.next += 1;
            self.bfp.insert(self.next, v);
            return Ok((self.next, dims));
        }
        if let Some(t) = self.bf16.get(&a) {
            let v = t.rows_view(first, rows).map_err(TtError::from)?;
            return Ok(self.insert_bf16(v));
        }
        let v = self.get(a)?.rows_view(first, rows).map_err(TtError::from)?;
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

    fn get(&self, id: BufferId) -> Result<&tt_kernels::tensor::DramTensor, TtError> {
        self.live
            .get(&id)
            .ok_or_else(|| TtError::Device(format!("no device buffer {id}")))
    }
}

/// What [`Engine::sort_planes`] made: each buffer with its `[rows, cols]`.
pub struct SortedBuffers {
    pub keys: (BufferId, [usize; 2]),
    pub indices: Option<(BufferId, [usize; 2])>,
}

/// A sort of `a`'s planes on the device, without waiting: the keys'
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
                            Err(TtError::Device(format!("sorted planes are not {dims:?}")))
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
                        let why = Arc::new(TtError::Op {
                            what: format!("sort {spec:?}"),
                            device,
                            cause: Arc::new(e),
                        });
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

pub(crate) fn random(
    device: TtDevice,
    dims: [usize; 2],
    draw: crate::random::Draw,
    base: u64,
) -> (BufferId, [usize; 2]) {
    submit(
        "random",
        device,
        dims,
        move || format!("native random {draw:?}"),
        move |engine, _| engine.random_dram(dims, draw, base),
    )
}

pub(crate) fn cast_f16(
    device: TtDevice,
    id: BufferId,
    dims: [usize; 2],
    to_f16: bool,
) -> (BufferId, [usize; 2]) {
    submit(
        "cast_f16",
        device,
        dims,
        || "native FP16 storage conversion".into(),
        move |engine, ids| engine.cast_f16(ids.get(id)?, to_f16),
    )
}

type Job = Box<dyn FnOnce(&mut dyn Engine, &mut Ids) + Send>;

/// `job`, charged to `op` on the server thread for the time it takes
/// (`report::server_report`).
fn timed_job(op: &'static str, job: Job) -> Job {
    Box::new(move |engine, ids| {
        engine.set_op_label(op);
        let t0 = std::time::Instant::now();
        job(engine, ids);
        crate::report::served(op, t0.elapsed());
    })
}

/// `job`, counted against the innermost op on this thread (`OpStat::jobs`)
/// and timed on the server thread under that op.
fn tagged(job: Job) -> Job {
    timed_job(crate::report::queued(), job)
}

/// Each buffer the callers have named, as the engine numbers it -- or why it
/// was never made. Kept on the server thread, one per attachment: a caller
/// names a result before it exists (B8, asynchronous dispatch), and the
/// server, running jobs in order, makes it before any job reads it.
#[derive(Default)]
pub(crate) struct Ids {
    map: HashMap<BufferId, Result<BufferId, Arc<TtError>>>,
    /// The first asynchronous op that failed on this attachment: reported by
    /// every wait from then on (a download, a trace's replay), whatever it
    /// reads, so a failure whose result nobody reads is not lost.
    failed: Option<Arc<TtError>>,
}

impl Ids {
    /// `Err` with the attachment's first failure, if an op has failed.
    pub(crate) fn healthy(&self) -> Result<(), TtError> {
        match &self.failed {
            Some(why) => Err(TtError::EarlierFailure(why.clone())),
            None => Ok(()),
        }
    }

    /// The engine's number for the caller's `id`; an error if the op that was
    /// to make it failed (its message), or if this attachment never made it
    /// -- a tensor that outlived an earlier attachment of its device.
    pub(crate) fn get(&self, id: BufferId) -> Result<BufferId, TtError> {
        match self.map.get(&id) {
            Some(Ok(e)) => Ok(*e),
            Some(Err(why)) => Err(TtError::clone(why)),
            None => Err(TtError::StaleTensor { id }),
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
    ready: Sender<Result<(ElementwiseMode, MatmulMode, bool), TtError>>,
    jobs: Receiver<Job>,
    device: TtDevice,
    poison: Poison,
}

/// Why a server thread stopped running jobs: the panic of the job that
/// killed it. Shared with the callers, who report it instead of "the thread
/// has stopped".
type Poison = Arc<OnceLock<TtError>>;

/// Run `f`, turning a panic into [`TtError::ServerPanicked`] and recording it
/// in `poison`: the engine's state after a panic is not to be trusted, so no
/// later job runs on it.
fn guarded<R>(poison: &Poison, device: TtDevice, f: impl FnOnce() -> R) -> Result<R, TtError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "a panic with no message".into());
        let e = TtError::ServerPanicked { device, message };
        let _ = poison.set(e.clone());
        poison.get().cloned().unwrap_or(e)
    })
}

impl Serve {
    /// Serve jobs on `engine` until the [`AttachGuard`] is dropped.
    ///
    /// A job that panics ends the engine's service, not the thread's: the
    /// queue is drained without running anything, so callers waiting on a
    /// reply get the panic's message ([`TtError::ServerPanicked`]) rather
    /// than a closed channel.
    pub fn serve(self, engine: &mut dyn Engine) {
        let _ = self.ready.send(Ok((
            engine.elementwise_mode(),
            engine.matmul_mode(),
            engine.supports_exact_matmul(),
        )));
        let mut ids = Ids::default();
        while let Ok(job) = self.jobs.recv() {
            if self.poison.get().is_some() {
                continue;
            }
            let _ = guarded(&self.poison, self.device, || job(engine, &mut ids));
        }
    }
}

struct Attached {
    jobs: Sender<Job>,
    elementwise: ElementwiseMode,
    matmul: MatmulMode,
    exact_matmul_ok: bool,
    poison: Poison,
    /// Set once the factory has served the first job's worth of state: until
    /// then `elementwise` is a placeholder.
    ready: bool,
    /// Attached by the registry on first use (or [`init`](crate::init)) rather
    /// than by [`attach`]; the configuration it was attached with.
    config: Option<crate::TtConfig>,
    /// The process's fork count when this was attached: an attachment from
    /// before a `fork` has no server thread in the child.
    epoch: u32,
    /// Whether the engine keeps tensors on the device, asked once.
    supports_dram: Arc<OnceLock<bool>>,
    /// Whether the engine takes `Engine::matmul_dram_batched_layout`, asked once.
    supports_layout: Arc<OnceLock<bool>>,
    /// This attachment's number, unique in the process: what a cache of
    /// device buffers made for it ([`attachment`]) is keyed by, so a later
    /// attachment of the same device never reads them.
    serial: u64,
}

impl Attached {
    fn auto(&self) -> bool {
        self.config.is_some()
    }

    /// Is the server thread of this attachment running in *this* process?
    fn live(&self) -> bool {
        self.epoch == FORKS.load(Ordering::Relaxed)
    }
}

/// How many times this process has been forked from (counted in the child).
///
/// A forked child has only the forking thread: the server threads of the
/// parent's attachments are gone, though their queues are still there, so a
/// job sent to one would wait forever. Attachments from an earlier epoch are
/// treated as absent. (`fork_scope` runs every simulator gate in such a child.)
static FORKS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

extern "C" fn count_fork() {
    FORKS.fetch_add(1, Ordering::SeqCst);
}

fn note_forks() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: `count_fork` is async-signal-safe (one atomic add) and
        // `pthread_atfork` only registers it.
        unsafe {
            libc::pthread_atfork(None, None, Some(count_fork));
        }
    });
}

static ATTACHED: Mutex<Option<HashMap<TtDevice, Attached>>> = Mutex::new(None);

fn locked<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn with_attached<R>(f: impl FnOnce(&mut HashMap<TtDevice, Attached>) -> R) -> R {
    let mut guard = locked(&ATTACHED);
    f(guard.get_or_insert_with(HashMap::new))
}

/// Numbers each attachment ([`Attached::serial`]).
static SERIAL: AtomicU64 = AtomicU64::new(1);

/// The number of `device`'s running attachment in this process, if any: a
/// cache of buffers made on the device keeps it beside them and drops them
/// once it changes (the buffers were another attachment's).
pub(crate) fn attachment(device: TtDevice) -> Option<u64> {
    with_attached(|a| a.get(&device).filter(|d| d.live()).map(|d| d.serial))
}

/// The devices with a running attachment, in this process.
pub(crate) fn attached_devices() -> Vec<TtDevice> {
    with_attached(|a| {
        a.iter()
            .filter(|(_, d)| d.live())
            .map(|(k, _)| *k)
            .collect()
    })
}

pub(crate) fn is_attached(device: TtDevice) -> bool {
    with_attached(|a| a.get(&device).is_some_and(Attached::live))
}

/// Serializes first-use attachment, so one factory call opens one card.
static OPENING: Mutex<()> = Mutex::new(());

/// Devices the registry attached, for the life of the process: the guards are
/// never dropped, since statics are not, and the driver's cleanup write covers
/// process exit (`tt-kmd/src/lib.rs`). [`detach`](crate::detach) removes one.
static AUTO: Mutex<Option<HashMap<TtDevice, AttachGuard>>> = Mutex::new(None);

/// Devices the registry could not attach, and why: the next op says the same
/// thing at once instead of opening the card again.
static FAILED: Mutex<Option<HashMap<TtDevice, TtError>>> = Mutex::new(None);

fn failures<R>(f: impl FnOnce(&mut HashMap<TtDevice, TtError>) -> R) -> R {
    f(locked(&FAILED).get_or_insert_with(HashMap::new))
}

/// Forget every remembered attach failure: the registry changed.
pub(crate) fn forget_failures() {
    failures(|f| f.clear());
}

/// Make sure `device` has an engine, attaching it through the registry
/// ([`crate::set_default_engine`]; silicon unless set) the first time it is
/// needed. An attach that failed fails again with the same error, without
/// calling the factory, until [`crate::init`] or [`crate::detach`].
pub(crate) fn ensure_attached(device: TtDevice) -> Result<(), TtError> {
    match with_attached(|a| a.get(&device).filter(|d| d.live()).map(|d| d.ready)) {
        Some(true) => return Ok(()),
        // Another thread is attaching it: wait for that to finish.
        Some(false) => {
            drop(locked(&OPENING));
            return Ok(());
        }
        None => {}
    }
    let _opening = locked(&OPENING);
    if is_attached(device) {
        return Ok(());
    }
    if let Some(e) = failures(|f| f.get(&device).cloned()) {
        return Err(e);
    }
    let result = crate::TtConfig::from_env().and_then(|cfg| attach_auto(device, cfg));
    if let Err(e) = &result {
        failures(|f| f.insert(device, e.clone()));
    }
    result
}

/// Attach `device` through the registry with `cfg` and keep it. The caller
/// holds [`OPENING`].
pub(crate) fn attach_auto(device: TtDevice, cfg: crate::TtConfig) -> Result<(), TtError> {
    let factory = crate::registry::default_factory(device, &cfg)?;
    let guard = attach_inner(device, factory, Some(cfg))?;
    let old = locked(&AUTO)
        .get_or_insert_with(HashMap::new)
        .insert(device, guard);
    drop(old);
    Ok(())
}

/// [`crate::init`]: attach `device` with `cfg` now, or confirm it already is.
pub(crate) fn init(device: TtDevice, cfg: crate::TtConfig) -> Result<(), TtError> {
    let _opening = locked(&OPENING);
    let existing = with_attached(|a| {
        a.get(&device)
            .filter(|d| d.live())
            .map(|d| (d.auto(), d.config.clone()))
    });
    match existing {
        Some((true, Some(have))) if have == cfg => Ok(()),
        Some((auto, _)) => Err(TtError::AlreadyAttached { device, auto }),
        None => {
            failures(|f| f.remove(&device));
            let result = attach_auto(device, cfg);
            if let Err(e) = &result {
                failures(|f| f.insert(device, e.clone()));
            }
            result
        }
    }
}

/// [`crate::detach`]: detach a device the registry attached, dropping its
/// hardware, and forget a failed attach so the next op tries again.
pub(crate) fn detach(device: TtDevice) -> Result<(), TtError> {
    let _opening = locked(&OPENING);
    failures(|f| f.remove(&device));
    let guard = locked(&AUTO).as_mut().and_then(|m| m.remove(&device));
    match guard {
        Some(guard) => {
            drop(guard);
            Ok(())
        }
        None if is_attached(device) => Err(TtError::Device(format!(
            "{device} was attached with burn_tt::attach: drop its AttachGuard to detach it"
        ))),
        None => Ok(()),
    }
}

/// Keeps `device` attached; dropping it detaches the device and drops its
/// hardware.
#[must_use = "dropping the guard detaches the device at once"]
pub struct AttachGuard {
    device: TtDevice,
    thread: Option<JoinHandle<()>>,
    epoch: u32,
}

impl Drop for AttachGuard {
    fn drop(&mut self) {
        if self.epoch != FORKS.load(Ordering::Relaxed) {
            // Made before a fork: the thread is the parent's, and there is
            // nothing here to join or detach.
            std::mem::forget(self.thread.take());
            return;
        }
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
/// that is already attached is refused ([`TtError::AlreadyAttached`]).
///
/// This is the scoped, explicit way: nothing needs to call it, since a device
/// is attached through the registry the first time an op needs it
/// ([`crate::init`], [`crate::set_default_engine`]).
pub fn attach<F>(device: TtDevice, factory: F) -> Result<AttachGuard, TtError>
where
    F: FnOnce(Serve) -> Result<(), TtError> + Send + 'static,
{
    attach_inner(device, factory, None)
}

/// [`attach`]; `config` is `Some` for the registry's attachments.
fn attach_inner<F>(
    device: TtDevice,
    factory: F,
    config: Option<crate::TtConfig>,
) -> Result<AttachGuard, TtError>
where
    F: FnOnce(Serve) -> Result<(), TtError> + Send + 'static,
{
    note_forks();
    let epoch = FORKS.load(Ordering::SeqCst);
    let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel();
    let poison = Poison::default();
    let served_poison = poison.clone();
    let existing = with_attached(|a| match a.get(&device) {
        Some(d) if d.live() => Some(d.auto()),
        _ => {
            // Absent, or from before a fork: replaced.
            a.insert(
                device,
                Attached {
                    jobs: jobs_tx,
                    elementwise: ElementwiseMode::Sfpu,
                    matmul: MatmulMode::Fast,
                    exact_matmul_ok: true,
                    poison,
                    ready: false,
                    config,
                    epoch,
                    supports_dram: Arc::default(),
                    supports_layout: Arc::default(),
                    serial: SERIAL.fetch_add(1, Ordering::Relaxed),
                },
            );
            None
        }
    });
    if let Some(auto) = existing {
        return Err(TtError::AlreadyAttached { device, auto });
    }
    let failed = ready_tx.clone();
    let thread = std::thread::Builder::new()
        .name(format!("burn-tt {device}"))
        .spawn(move || {
            let serve = Serve {
                ready: ready_tx,
                jobs: jobs_rx,
                device,
                poison: served_poison,
            };
            match factory(serve) {
                Ok(()) => {
                    let _ = failed.send(Err(TtError::Device(
                        "the engine factory returned without serving".into(),
                    )));
                }
                Err(e) => {
                    let _ = failed.send(Err(e));
                }
            }
        })
        .map_err(|e| TtError::Device(format!("could not start the server thread: {e}")))?;
    let guard = AttachGuard {
        device,
        thread: Some(thread),
        epoch,
    };
    match ready_rx.recv() {
        Ok(Ok((mode, matmul, exact_ok))) => {
            with_attached(|a| {
                let d = a.get_mut(&device).expect("reserved attachment");
                d.elementwise = mode;
                d.matmul = matmul;
                d.exact_matmul_ok = exact_ok;
                d.ready = true;
            });
            Ok(guard)
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err(TtError::Device(
            "the server thread died before serving".into(),
        )),
    }
}

/// Run `job` on `device`'s engine and wait for it.
///
/// # Panics
///
/// If `device` has no engine and the registry cannot attach one, or its
/// server thread has gone: Burn's ops cannot return an error, and quietly
/// computing on the host instead would make the device path unobservable.
/// [`try_run`] returns the error.
pub(crate) fn run<R: Send + 'static>(
    device: TtDevice,
    job: impl FnOnce(&mut dyn Engine, &mut Ids) -> R + Send + 'static,
) -> R {
    try_run(device, job).unwrap_or_else(|e| panic!("{e}"))
}

/// The queue to `device`'s server and the cell its panic is recorded in,
/// attaching the device through the registry if nothing serves it yet.
fn handle(device: TtDevice) -> Result<(Sender<Job>, Poison), TtError> {
    let get = || {
        with_attached(|a| {
            a.get(&device)
                .filter(|d| d.live())
                .map(|d| (d.jobs.clone(), d.poison.clone()))
        })
    };
    if let Some(h) = get() {
        return Ok(h);
    }
    ensure_attached(device)?;
    get().ok_or(TtError::NotAttached { device })
}

/// Why `device`'s server cannot answer: its job panicked, or it is gone.
fn stopped(device: TtDevice, poison: &Poison) -> TtError {
    poison.get().cloned().unwrap_or_else(|| {
        TtError::Device(format!("{device}'s server thread stopped during a job"))
    })
}

/// [`run`], returning the error: the device is not attached, or its server
/// panicked (this job's or an earlier one's).
pub(crate) fn try_run<R: Send + 'static>(
    device: TtDevice,
    job: impl FnOnce(&mut dyn Engine, &mut Ids) -> R + Send + 'static,
) -> Result<R, TtError> {
    let (sender, poison) = handle(device)?;
    let (tx, rx) = mpsc::channel();
    let caught = poison.clone();
    sender
        .send(tagged(Box::new(move |engine, ids| {
            // Caught here, not only by the server loop, so the caller is
            // answered with the panic after it is recorded.
            let _ = tx.send(guarded(&caught, device, || job(engine, ids)));
        })))
        .map_err(|_| stopped(device, &poison))?;
    match rx.recv() {
        Ok(reply) => reply,
        Err(_) => Err(stopped(device, &poison)),
    }
}

/// Queue `job` on `device`'s server, without waiting.
fn send(device: TtDevice, job: Job) {
    let (sender, poison) = handle(device).unwrap_or_else(|e| panic!("{e}"));
    sender
        .send(tagged(job))
        .unwrap_or_else(|_| panic!("{}", stopped(device, &poison)));
}

/// [`run`] only if `device` already has an engine: `None` otherwise, and the
/// registry is not asked. For cleanup that must never open a card.
pub(crate) fn run_if_attached<R: Send + 'static>(
    device: TtDevice,
    job: impl FnOnce(&mut dyn Engine, &mut Ids) -> R + Send + 'static,
) -> Option<R> {
    if !is_attached(device) {
        return None;
    }
    try_run(device, job).ok()
}

/// Wait for everything queued on `device`, and report the first failure, if
/// there was one. Nothing to do, and no error, for a device that is not
/// attached.
pub(crate) fn try_sync(device: TtDevice) -> Result<(), TtError> {
    if !is_attached(device) {
        return Ok(());
    }
    try_run(device, |_, ids| ids.healthy())?
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
    op: impl FnOnce(&mut dyn Engine, &Ids) -> Result<(BufferId, [usize; 2]), TtError> + Send + 'static,
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
                        Err(TtError::Device(format!(
                            "made [{}, {}] where [{}, {}] was expected",
                            got[0], got[1], dims[0], dims[1]
                        )))
                    }
                });
                let made = made.map_err(|e| {
                    Arc::new(TtError::Op {
                        what: what(),
                        device,
                        cause: Arc::new(e),
                    })
                });
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
    ensure_attached(device).unwrap_or_else(|e| panic!("{e}"));
    with_attached(|a| a.get(&device).filter(|d| d.live()).map(|d| d.elementwise))
        .unwrap_or_else(|| panic!("{}", TtError::NotAttached { device }))
}

/// How matmuls on `device` accumulate: the attachment's snapshot, with no
/// server round trip.
pub(crate) fn matmul_mode(device: TtDevice) -> MatmulMode {
    ensure_attached(device).unwrap_or_else(|e| panic!("{e}"));
    with_attached(|a| a.get(&device).filter(|d| d.live()).map(|d| d.matmul))
        .unwrap_or_else(|| panic!("{}", TtError::NotAttached { device }))
}

/// Can `device`'s engine run [`MatmulMode::Exact`]? The attachment's snapshot.
pub(crate) fn supports_exact_matmul(device: TtDevice) -> bool {
    ensure_attached(device).unwrap_or_else(|e| panic!("{e}"));
    with_attached(|a| {
        a.get(&device)
            .filter(|d| d.live())
            .map(|d| d.exact_matmul_ok)
    })
    .unwrap_or_else(|| panic!("{}", TtError::NotAttached { device }))
}

/// Does `device`'s engine keep tensors on the device? Asked once per
/// attachment, so a device attached again with another engine is asked again.
///
/// # Panics
///
/// If the device has no engine and the registry cannot attach one
/// ([`ensure_attached`]): the op that asks needs a device.
pub(crate) fn supports_dram(device: TtDevice) -> bool {
    ensure_attached(device).unwrap_or_else(|e| panic!("{e}"));
    let known = with_attached(|a| {
        a.get(&device)
            .filter(|d| d.live())
            .map(|d| d.supports_dram.clone())
    })
    .unwrap_or_else(|| panic!("{}", TtError::NotAttached { device }));
    *known.get_or_init(|| run(device, |engine, _| engine.supports_dram()))
}

/// Does `device`'s engine take a batched matmul's layout
/// ([`matmul_dram_batched_layout`])? Asked once an attachment; `false` for a
/// mesh, whose batched products are its own.
pub(crate) fn supports_batched_layout(device: TtDevice) -> bool {
    if !supports_dram(device) {
        return false;
    }
    let known = with_attached(|a| {
        a.get(&device)
            .filter(|d| d.live())
            .map(|d| d.supports_layout.clone())
    })
    .unwrap_or_else(|| panic!("{}", TtError::NotAttached { device }));
    *known.get_or_init(|| run(device, |engine, _| engine.supports_batched_layout()))
}

/// Can `device` be used for device ops: it has an engine, or the registry
/// attaches one now? `false` for a device nothing can serve (no such card),
/// whose tensors then stay on the host. The reason is remembered and the next
/// op that needs the device ([`supports_dram`]) reports it.
pub(crate) fn attachable(device: TtDevice) -> bool {
    ensure_attached(device).is_ok()
}

/// Everything `device`'s engine has moved across its transport so far, if it
/// can say ([`Engine::device_traffic`]). Queued behind every job already sent,
/// frees included, so it counts all of them.
///
/// `None` for a device with no engine, which this does not attach.
pub fn device_traffic(device: TtDevice) -> Option<tt_device::Traffic> {
    run_if_attached(device, |engine, _| engine.device_traffic()).flatten()
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

/// What a failed download of buffer `id` says: the op, the device and why.
fn download_failed(device: TtDevice, what: String, cause: TtError) -> TtError {
    TtError::Op {
        what,
        device,
        cause: Arc::new(cause),
    }
}

/// Download a BF16 or FP16 buffer's datums as bits.
pub(crate) fn try_download_bf16(
    device: TtDevice,
    id: BufferId,
    rows: usize,
    cols: usize,
) -> Result<Vec<u16>, TtError> {
    crate::traffic::downloaded_width(rows, cols, 2);
    crate::report::downloaded_width(rows, cols, 2);
    crate::traffic::timed("download_bf16", || {
        try_run(device, move |engine, ids| {
            ids.healthy()?;
            ids.get(id).and_then(|e| engine.download_bf16(e))
        })
    })
    .and_then(|r| r)
    .map_err(|e| download_failed(device, format!("BF16 download {id} [{rows}, {cols}]"), e))
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

pub(crate) fn download_bf16(device: TtDevice, id: BufferId, rows: usize, cols: usize) -> Vec<u16> {
    try_download_bf16(device, id, rows, cols).unwrap_or_else(|e| panic!("{e}"))
}

/// Download any buffer's datums as bits.
pub(crate) fn try_download_bits(
    device: TtDevice,
    id: BufferId,
    rows: usize,
    cols: usize,
) -> Result<Vec<u32>, TtError> {
    let v = crate::traffic::timed("download", || {
        try_run(device, move |engine, ids| {
            ids.healthy()?;
            ids.get(id).and_then(|e| engine.download_bits(e))
        })
    })
    .and_then(|r| r)
    .map_err(|e| download_failed(device, format!("download {id}"), e))?;
    debug_assert_eq!(v.len(), rows * cols);
    crate::traffic::downloaded(rows, cols);
    crate::report::downloaded(rows, cols);
    Ok(v)
}

/// [`try_download_bits`], panicking on a device error.
pub(crate) fn download_bits(device: TtDevice, id: BufferId, rows: usize, cols: usize) -> Vec<u32> {
    try_download_bits(device, id, rows, cols).unwrap_or_else(|e| panic!("{e}"))
}

/// Download. `rows` and `cols` are the buffer's, for the traffic count.
pub(crate) fn try_download(
    device: TtDevice,
    id: BufferId,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, TtError> {
    let v = crate::traffic::timed("download", || {
        try_run(device, move |engine, ids| {
            ids.healthy()?;
            ids.get(id).and_then(|e| engine.download(e))
        })
    })
    .and_then(|r| r)
    .map_err(|e| download_failed(device, format!("download {id}"), e))?;
    debug_assert_eq!(v.len(), rows * cols);
    crate::traffic::downloaded(rows, cols);
    crate::report::downloaded(rows, cols);
    Ok(v)
}

/// [`try_download`], panicking on a device error.
pub(crate) fn download(device: TtDevice, id: BufferId, rows: usize, cols: usize) -> Vec<f32> {
    try_download(device, id, rows, cols).unwrap_or_else(|e| panic!("{e}"))
}

/// Free, without waiting; nothing to do if the device has gone.
pub(crate) fn free(device: TtDevice, id: BufferId) {
    forget(id);
    if let Some(sender) =
        with_attached(|a| a.get(&device).filter(|d| d.live()).map(|d| d.jobs.clone()))
    {
        let _ = sender.send(timed_job(
            crate::report::FREE,
            Box::new(move |engine, ids| {
                // A buffer this attachment never made (a stale tensor's) is not
                // anyone's here: nothing to free.
                if let Some(Ok(e)) = ids.map.remove(&id) {
                    engine.free(e);
                }
            }),
        ));
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

/// A block of a resident buffer, as [`Engine::eltwise_blocks`] reads it: the
/// `dims` elements whose top-left is `at` (a tile boundary; ragged only at
/// the buffer's own edge).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BlockArg {
    pub id: BufferId,
    pub at: [usize; 2],
    pub dims: [usize; 2],
}

/// [`eltwise_op`] over blocks of resident buffers, without waiting: the
/// result is `a`'s block's dims.
pub(crate) fn eltwise_blocks(
    device: TtDevice,
    op: tt_kernels::tensor::Eltwise,
    a: BlockArg,
    b: Option<BlockArg>,
    c: Option<BlockArg>,
) -> (BufferId, [usize; 2]) {
    submit(
        "eltwise",
        device,
        a.dims,
        move || format!("element-wise {:#x} on blocks", op.kind),
        move |engine, ids| {
            let get = |x: BlockArg| -> Result<BlockArg, TtError> {
                Ok(BlockArg {
                    id: ids.get(x.id)?,
                    ..x
                })
            };
            let (b, c) = (b.map(get).transpose()?, c.map(get).transpose()?);
            engine.eltwise_blocks(op, get(a)?, b, c)
        },
    )
}

/// Rotary position embedding as one op on the device, without waiting: `x`'s block's dims.
pub(crate) fn rope_blocks(
    device: TtDevice,
    x: BlockArg,
    cos: BlockArg,
    sin: BlockArg,
    head: usize,
) -> (BufferId, [usize; 2]) {
    submit(
        "rope",
        device,
        x.dims,
        move || format!("rope, heads of {head}"),
        move |engine, ids| {
            let get = |b: BlockArg| -> Result<BlockArg, TtError> {
                Ok(BlockArg {
                    id: ids.get(b.id)?,
                    ..b
                })
            };
            engine.rope_blocks(get(x)?, get(cos)?, get(sin)?, head)
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
    reduce_then(device, a, op, axis, None)
}

/// [`reduce`] with `then`'s exact unary kind on the result before its pack
/// (`tt_kernels::tensor::sfpu_reduce_then`): a softmax's clamps.
pub(crate) fn reduce_then(
    device: TtDevice,
    a: BufferId,
    op: tt_kernels::sfpu::reduce::ReduceOp,
    axis: tt_kernels::sfpu::reduce::Axis,
    then: tt_kernels::tensor::ReduceThen,
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
        move |engine, ids| engine.reduce(ids.get(a)?, op, axis, then),
    )
}

/// RMSNorm's statistic on the device, without waiting: each row's `sqrt(mean(a a) + eps)`,
/// `[rows, 1]`. The caller has checked the row fits one job (`tt_kernels::sfpu::norm::fits`).
/// The index of a row's largest element on the device, without waiting.
pub(crate) fn argmax_row(device: TtDevice, a: BufferId) -> (BufferId, [usize; 2]) {
    submit(
        "argmax_row",
        device,
        [1, 1],
        || "argmax row".into(),
        move |engine, ids| engine.argmax_row(ids.get(a)?),
    )
}

/// `softmax(scale x, masked)` over each row of `x` on the device, without waiting
/// (`Session::softmax_rows`).
pub(crate) fn softmax_rows(
    device: TtDevice,
    x: BufferId,
    mask: BufferId,
    scale: f32,
) -> (BufferId, [usize; 2]) {
    submit(
        "softmax_rows",
        device,
        dims_of(x),
        move || format!("scale {scale:e}"),
        move |engine, ids| engine.softmax_rows(ids.get(x)?, ids.get(mask)?, scale),
    )
}

/// `a / sqrt(mean(a a) + eps)` over each row on the device, without waiting.
pub(crate) fn rms_norm(device: TtDevice, a: BufferId, eps: f32) -> (BufferId, [usize; 2]) {
    submit(
        "rms_norm",
        device,
        dims_of(a),
        move || format!("eps {eps:e}"),
        move |engine, ids| engine.rms_norm(ids.get(a)?, eps),
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
    submit("matmul_dram", device, [m, n], move || "matmul".into(), {
        // Frozen at submit time, as the matmul mode is: the closure runs on the server thread.
        let fidelity = crate::matmul_mode::fidelity_override();
        move |engine, ids| match fidelity {
            None => engine.matmul_dram(ids.get(a)?, a_transposed, ids.get(b)?, b_transposed),
            Some(f) => {
                engine.matmul_dram_at(ids.get(a)?, a_transposed, ids.get(b)?, b_transposed, f)
            }
        }
    })
}

/// A batched matmul over blocks on the device with `layout`
/// (`Session::matmul_dram_batched_with`), without waiting. Only where
/// [`supports_batched_layout`], at the attachment's own fidelity.
pub(crate) fn matmul_dram_batched_layout(
    device: TtDevice,
    a: BufferId,
    b: BufferId,
    items: Vec<(Block, Block)>,
    mkn: [usize; 3],
    layout: BatchedLayout,
) -> (BufferId, [usize; 2]) {
    let [m, _, n] = mkn;
    let rows = if layout.stacked || items.len() == 1 {
        items.len() * m
    } else {
        items.len() * m.div_ceil(32) * 32
    };
    submit(
        "matmul_dram",
        device,
        [rows, n],
        || "batched matmul".into(),
        move |engine, ids| {
            engine.matmul_dram_batched_layout(ids.get(a)?, ids.get(b)?, &items, mkn, layout)
        },
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
        {
            let fidelity = crate::matmul_mode::fidelity_override();
            move |engine, ids| match fidelity {
                None => engine.matmul_dram_batched(ids.get(a)?, ids.get(b)?, &items, mkn),
                Some(f) => engine.matmul_dram_batched_at(ids.get(a)?, ids.get(b)?, &items, mkn, f),
            }
        },
    )
}

/// Attention for a few query rows a key/value head as one op on the device
/// (`Session::attention_rows`), without waiting: `[q, k, v, mask]`.
pub(crate) fn attention_rows(
    device: TtDevice,
    buffers: [BufferId; 4],
    items: Vec<tt_kernels::attention::Item>,
    dims: [usize; 4],
    scale: f32,
) -> (BufferId, [usize; 2]) {
    let [m, _, _, dv] = dims;
    submit(
        "attention_rows",
        device,
        [1, items.len() * m * dv],
        || "attention rows".into(),
        move |engine, ids| {
            let buffers = [
                ids.get(buffers[0])?,
                ids.get(buffers[1])?,
                ids.get(buffers[2])?,
                ids.get(buffers[3])?,
            ];
            engine.attention_rows(buffers, &items, dims, scale)
        },
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

/// A copy of `t` with rows of `value` written over it (`Engine::assign_rows`),
/// without waiting: `dims` is `t`'s.
pub(crate) fn assign_rows(
    device: TtDevice,
    t: BufferId,
    value: BufferId,
    rows: Vec<(usize, usize)>,
    dims: [usize; 2],
) -> (BufferId, [usize; 2]) {
    submit(
        "assign_rows",
        device,
        dims,
        || "rows written over a copy".into(),
        move |engine, ids| engine.assign_rows(ids.get(t)?, ids.get(value)?, &rows),
    )
}

/// A copy of `table` with `values`' row `i` at its row `indices[i]`
/// (`Engine::scatter_rows`), the indices a resident I32 buffer, `[1, n]` or
/// (`column`) `[n, 1]`, without waiting: `dims` is `table`'s.
pub(crate) fn scatter_rows(
    device: TtDevice,
    table: BufferId,
    indices: BufferId,
    values: BufferId,
    column: bool,
    dims: [usize; 2],
) -> (BufferId, [usize; 2]) {
    submit(
        "scatter_rows",
        device,
        dims,
        || "checked resident row scatter".into(),
        move |engine, ids| {
            engine.scatter_rows(ids.get(table)?, ids.get(indices)?, ids.get(values)?, column)
        },
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

/// `table`'s rows chosen by `indices` (a resident `[1, n]` I32 buffer): the
/// result is `dims`, `[n, table columns]`.
pub(crate) fn select_rows(
    device: TtDevice,
    table: BufferId,
    indices: BufferId,
    dims: [usize; 2],
) -> (BufferId, [usize; 2]) {
    submit(
        "select_rows",
        device,
        dims,
        || "checked resident row select".into(),
        move |engine, ids| engine.select_rows(ids.get(table)?, ids.get(indices)?),
    )
}

/// `table`'s rows for each tap of `indices` (a resident `[taps, n]` I32
/// buffer), side by side: `dims` is `[n, tt_kernels::index::select_taps_cols(..)]`.
pub(crate) fn select_taps(
    device: TtDevice,
    table: BufferId,
    indices: BufferId,
    zero_row: bool,
    dims: [usize; 2],
) -> (BufferId, [usize; 2]) {
    submit(
        "select_taps",
        device,
        dims,
        || "checked resident tap select".into(),
        move |engine, ids| engine.select_taps(ids.get(table)?, ids.get(indices)?, zero_row),
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
    /// How matmuls accumulate, fixed for the attachment.
    pub matmul: MatmulMode,
}

impl KmdEngine {
    /// Consuming builder for a factory, before Serve attaches this engine.
    pub fn with_elementwise_mode(mut self, mode: ElementwiseMode) -> Result<Self, TtError> {
        if let Some(buffers) = self.buffers.take() {
            self.buffers = Some(buffers.with_elementwise_mode(mode));
        } else if mode != ElementwiseMode::Sfpu {
            return Err(TtError::unsupported(
                "matrix elementwise requires resident storage",
            ));
        }
        Ok(self)
    }
}

impl Engine for KmdEngine {
    fn matmul_mode(&self) -> MatmulMode {
        self.matmul
    }
    fn elementwise_mode(&self) -> ElementwiseMode {
        self.buffers
            .as_ref()
            .map_or(ElementwiseMode::Sfpu, DramBuffers::elementwise_mode)
    }
    fn math_mode(&self) -> tt_kernels::sfpu::approx::MathMode {
        self.session.math_mode()
    }
    fn set_math_mode(&mut self, mode: tt_kernels::sfpu::approx::MathMode) -> Result<(), TtError> {
        self.session.set_math_mode(mode);
        Ok(())
    }
    fn pool_bf16(
        &mut self,
        a: BufferId,
        windows: &[Vec<[usize; 2]>],
        divisors: &[usize],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers.as_mut().ok_or_else(unsupported)?.pool_bf16(
            &mut self.session,
            a,
            windows,
            divisors,
            dims,
        )
    }
    fn matmul_bf16(&mut self, a: BufferId, b: BufferId) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers.as_mut().ok_or_else(unsupported)?.matmul_bf16(
            &mut self.session,
            a,
            b,
            self.fidelity,
            self.budget,
        )
    }
    fn upload_bf16(&mut self, bits: &[u16], rows: usize, cols: usize) -> Result<BufferId, TtError> {
        self.buffers.as_mut().ok_or_else(unsupported)?.upload_bf16(
            &mut self.session,
            bits,
            rows,
            cols,
        )
    }
    fn download_bf16(&mut self, id: BufferId) -> Result<Vec<u16>, TtError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .download_bf16(&mut self.session, id)
    }
    fn cast_float(&mut self, id: BufferId, bf16: bool) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .cast_float(&mut self.session, id, bf16)
    }
    fn cast_bfp(
        &mut self,
        id: BufferId,
        format: Option<tt_kernels::bfp::BfpFormat>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .cast_bfp(&mut self.session, id, format)
    }
    fn matmul_bfp(&mut self, a: BufferId, b: BufferId) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers.as_mut().ok_or_else(unsupported)?.matmul_bfp(
            &mut self.session,
            a,
            b,
            self.fidelity,
        )
    }
    fn matmul(&mut self, a: &[f32], b: &[f32], mkn: [usize; 3]) -> Result<Vec<f32>, TtError> {
        Ok(self
            .session
            .matmul(a, b, mkn, self.route, self.fidelity, self.budget)?)
    }
    fn supports_dram(&self) -> bool {
        self.buffers.is_some()
    }
    fn upload(&mut self, v: &[f32], rows: usize, cols: usize) -> Result<BufferId, TtError> {
        let b = self.buffers.as_mut().ok_or_else(unsupported)?;
        b.upload(&mut self.session, v, rows, cols)
    }
    fn metadata(
        &mut self,
        bits: &[u32],
        dims: [usize; 2],
        elem: Elem,
    ) -> Result<BufferId, TtError> {
        let b = self.buffers.as_mut().ok_or_else(unsupported)?;
        b.metadata(&mut self.session, bits, dims, elem)
    }
    fn download(&mut self, id: BufferId) -> Result<Vec<f32>, TtError> {
        let b = self.buffers.as_mut().ok_or_else(unsupported)?;
        b.download(&mut self.session, id)
    }
    fn upload_bits(
        &mut self,
        v: &[u32],
        rows: usize,
        cols: usize,
        elem: Elem,
    ) -> Result<BufferId, TtError> {
        let b = self.buffers.as_mut().ok_or_else(unsupported)?;
        b.upload_bits(&mut self.session, v, rows, cols, elem)
    }
    fn download_bits(&mut self, id: BufferId) -> Result<Vec<u32>, TtError> {
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
    fn matmul_dram_at(
        &mut self,
        a: BufferId,
        ta: bool,
        b: BufferId,
        tb: bool,
        fidelity: Fidelity,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.matmul(
            &mut self.session,
            a,
            ta,
            b,
            tb,
            self.route,
            fidelity,
            self.budget,
        )
    }
    fn matmul_dram_batched_at(
        &mut self,
        a: BufferId,
        b: BufferId,
        items: &[(Block, Block)],
        mkn: [usize; 3],
        fidelity: Fidelity,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.matmul_batched(
            &mut self.session,
            a,
            b,
            items,
            mkn,
            self.route,
            fidelity,
            self.budget,
            BatchedLayout::STACKED,
        )
    }
    fn eltwise(
        &mut self,
        kind: u32,
        scalar: f32,
        a: BufferId,
        b: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.eltwise(&mut self.session, kind, scalar, a, b)
    }
    fn eltwise_op(
        &mut self,
        op: tt_kernels::tensor::Eltwise,
        a: BufferId,
        b: Option<BufferId>,
        c: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.eltwise_op(&mut self.session, op, a, b, c)
    }
    fn eltwise_blocks(
        &mut self,
        op: tt_kernels::tensor::Eltwise,
        a: BlockArg,
        b: Option<BlockArg>,
        c: Option<BlockArg>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.eltwise_blocks(&mut self.session, op, a, b, c)
    }
    fn rope_blocks(
        &mut self,
        x: BlockArg,
        cos: BlockArg,
        sin: BlockArg,
        head: usize,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.rope_blocks(&mut self.session, x, cos, sin, head)
    }
    fn pow(&mut self, x: BufferId, y: PowArg) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.pow(&mut self.session, x, y)
    }
    fn sum_rows(&mut self, a: BufferId) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.sum_rows(&mut self.session, a)
    }
    fn reduce(
        &mut self,
        a: BufferId,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        axis: tt_kernels::sfpu::reduce::Axis,
        then: tt_kernels::tensor::ReduceThen,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.reduce(&mut self.session, a, op, axis, then)
    }
    fn rms(&mut self, a: BufferId, eps: f32) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.rms(&mut self.session, a, eps)
    }
    fn softmax_rows(
        &mut self,
        x: BufferId,
        mask: BufferId,
        scale: f32,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.softmax_rows(&mut self.session, x, mask, scale)
    }
    fn rms_norm(&mut self, a: BufferId, eps: f32) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.rms_norm(&mut self.session, a, eps)
    }
    fn argmax_row(&mut self, a: BufferId) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.argmax_row(&mut self.session, a)
    }
    fn scan(
        &mut self,
        a: BufferId,
        op: tt_kernels::sfpu::scan::ScanOp,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.slice_rows(a, first, rows)
    }
    fn matmul_dram_batched(
        &mut self,
        a: BufferId,
        b: BufferId,
        items: &[(Block, Block)],
        mkn: [usize; 3],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
            BatchedLayout::STACKED,
        )
    }
    fn supports_batched_layout(&self) -> bool {
        self.buffers.is_some()
    }
    fn matmul_dram_batched_layout(
        &mut self,
        a: BufferId,
        b: BufferId,
        items: &[(Block, Block)],
        mkn: [usize; 3],
        layout: BatchedLayout,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
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
            layout,
        )
    }
    fn attention_rows(
        &mut self,
        buffers: [BufferId; 4],
        items: &[tt_kernels::attention::Item],
        dims: [usize; 4],
        scale: f32,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.attention_rows(
            &mut self.session,
            buffers,
            items,
            dims,
            scale,
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.copy_blocks(&mut self.session, a, moves, dims)
    }
    fn repack(
        &mut self,
        a: BufferId,
        sources: &[[usize; 2]],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.repack(&mut self.session, a, sources, dims)
    }
    fn zeros_dram(
        &mut self,
        dims: [usize; 2],
        elem: Elem,
        bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .zeros(&mut self.session, dims, elem, bf16)
    }
    fn gather_indexed(
        &mut self,
        input: BufferId,
        indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .gather_indexed(&mut self.session, input, indices)
    }
    fn select_rows(
        &mut self,
        table: BufferId,
        indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers.as_mut().ok_or_else(unsupported)?.select_rows(
            &mut self.session,
            table,
            indices,
        )
    }
    fn select_taps(
        &mut self,
        table: BufferId,
        indices: BufferId,
        zero_row: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers.as_mut().ok_or_else(unsupported)?.select_taps(
            &mut self.session,
            table,
            indices,
            zero_row,
        )
    }
    fn repack_many(
        &mut self,
        inputs: &[BufferId],
        sources: &[(usize, [usize; 2])],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.repack_many(&mut self.session, inputs, sources, dims)
    }
    fn gather_rows(
        &mut self,
        sources: &[BufferId],
        rows: &[(usize, usize)],
        cols: usize,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.gather_rows(&mut self.session, sources, rows, cols)
    }
    fn rows_add(
        &mut self,
        t: BufferId,
        indices: &[usize],
        value: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.rows_add(&mut self.session, t, indices, value)
    }
    fn assign_rows(
        &mut self,
        t: BufferId,
        value: BufferId,
        rows: &[(usize, usize)],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.assign_rows(&mut self.session, t, value, rows)
    }
    fn scatter_rows(
        &mut self,
        table: BufferId,
        indices: BufferId,
        values: BufferId,
        column: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.scatter_rows(&mut self.session, table, indices, values, column)
    }
    fn scatter_rows_into(
        &mut self,
        table: BufferId,
        indices: BufferId,
        values: BlockArg,
        column: bool,
    ) -> Result<(), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.scatter_rows_into(&mut self.session, table, indices, values, column)
    }
    fn device_traffic(&mut self) -> Option<tt_device::Traffic> {
        Some(self.session.device().traffic())
    }
    fn begin_trace(&mut self) -> Result<(), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.begin_trace(&mut self.session)
    }
    fn end_trace(&mut self) -> Result<u64, TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.end_trace(&mut self.session)
    }
    fn begin_trace_planned(&mut self, key: u64) -> Result<bool, TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.begin_trace_planned(&mut self.session, key)
    }
    fn begin_trace_cached(&mut self, key: u64) -> Result<bool, TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.begin_trace_cached(&mut self.session, key)
    }
    fn begin_trace_plan_pass(&mut self) -> Result<(), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.begin_trace_plan_pass(&mut self.session)
    }
    fn set_residency(&mut self, bytes: u64) -> Result<(), TtError> {
        self.session.set_residency(bytes);
        Ok(())
    }
    fn probe_captures(&mut self) -> Result<u64, TtError> {
        Ok(self.session.probe_captures())
    }
    fn probe_nc_waits(&mut self) -> Result<Vec<[u32; 8]>, TtError> {
        Ok(self.session.probe_nc_waits()?)
    }
    fn has_pending_plan(&mut self) -> Result<bool, TtError> {
        Ok(self.session.has_pending_plan())
    }
    fn run_trace(
        &mut self,
        trace: u64,
        input: BufferId,
        values: &[f32],
        output: BufferId,
    ) -> Result<TraceRun, TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.run_trace(&mut self.session, trace, input, values, output)
    }
    fn release_trace(&mut self, trace: u64) {
        if let Some(b) = self.buffers.as_mut() {
            b.release_trace(&mut self.session, trace);
        }
    }
    fn set_op_label(&mut self, label: &'static str) {
        self.session.set_op_label(label);
    }
    fn trace_ops(&mut self, trace: u64) -> Result<Vec<tt_kernels::trace::OpRecord>, TtError> {
        let bufs = self.buffers.as_ref().ok_or_else(unsupported)?;
        bufs.trace_ops(&self.session, trace)
    }
    fn trace_footprint(
        &mut self,
        trace: u64,
    ) -> Result<Vec<Option<tt_kernels::trace::UnitFootprint>>, TtError> {
        let bufs = self.buffers.as_ref().ok_or_else(unsupported)?;
        bufs.trace_footprint(&self.session, trace)
    }
    fn set_trace_name(&mut self, trace: u64, name: String) {
        if let Some(b) = self.buffers.as_mut() {
            b.set_trace_name(trace, name);
        }
    }
    fn profile_mode(
        &mut self,
        options: Option<tt_kernels::profile_mode::ProfileOptions>,
    ) -> Result<Option<crate::profile::Profile>, TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.profile_mode(&mut self.session, options)
    }
    fn profile_report(&mut self) -> Result<Option<crate::profile::Profile>, TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.profile_report(&mut self.session)
    }
    fn set_profile_buffer(&mut self, bytes: Option<u64>) -> Result<(), TtError> {
        self.session
            .set_profile_buffer(bytes)
            .map_err(TtError::from)
    }
    fn set_count_trace_bytes(&mut self, on: bool) -> Result<(), TtError> {
        self.session.set_count_trace_bytes(on);
        Ok(())
    }
    fn set_capture_reuse(&mut self, on: bool) -> Result<bool, TtError> {
        let was = self.session.capture_reuse();
        self.session.set_capture_reuse(on);
        Ok(was)
    }
    fn gddr_free_bytes(&mut self) -> Result<(u64, u64), TtError> {
        self.session.sync().map_err(TtError::from)?;
        self.session
            .dram_free_bytes_by_channel()
            .ok_or_else(|| TtError::Device("no GDDR attached".into()))
    }
    fn set_l1_region(&mut self, bytes: u64) -> Result<(), TtError> {
        self.session.set_l1_region(bytes);
        Ok(())
    }
    fn profile_generic_trace(
        &mut self,
        trace: u64,
        inputs: &[(BufferId, InputPayload)],
        outputs: &[(BufferId, OutputKind)],
    ) -> Result<(GenericTraceRun, tt_kernels::profile::DeviceProfile), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.profile_generic_trace(&mut self.session, trace, inputs, outputs)
    }
    fn copy_into(&mut self, src: BufferId, dst: BufferId) -> Result<(), TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.copy_into(&mut self.session, src, dst)
    }
    fn run_generic_trace(
        &mut self,
        trace: u64,
        inputs: &[(BufferId, InputPayload)],
        outputs: &[(BufferId, OutputKind)],
    ) -> Result<GenericTraceRun, TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.run_generic_trace(&mut self.session, trace, inputs, outputs)
    }

    fn sort_planes(
        &mut self,
        a: BufferId,
        spec: tt_kernels::sfpu::sort::Spec,
    ) -> Result<SortedBuffers, TtError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.sort_planes(&mut self.session, a, spec)
    }

    fn random_dram(
        &mut self,
        dims: [usize; 2],
        draw: crate::random::Draw,
        base: u64,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .random(&mut self.session, dims, draw, base)
    }

    fn cast_f16(&mut self, id: BufferId, to_f16: bool) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .as_mut()
            .ok_or_else(unsupported)?
            .cast_f16(&mut self.session, id, to_f16)
    }
}

/// Buffers whose contents something other than the op that made them has
/// written ([`copy_into`], a trace's replays of its inputs): a value derived
/// from one and cached ([`derived`](crate::tensor::TtTensor::derived)) could
/// be stale, so none is kept for them.
static WRITTEN: Mutex<Option<std::collections::HashSet<BufferId>>> = Mutex::new(None);

/// Record that `id`'s contents change after it is made.
pub(crate) fn written_in_place(id: BufferId) {
    locked(&WRITTEN)
        .get_or_insert_with(Default::default)
        .insert(id);
}

/// Has `id` been written after it was made ([`written_in_place`])?
pub(crate) fn is_written_in_place(id: BufferId) -> bool {
    locked(&WRITTEN).as_ref().is_some_and(|w| w.contains(&id))
}

/// Buffers an open capture owns as its state and may write in place
/// (`trace::Donation`), each with its [`write_generation`] when donated. Empty outside a
/// capture with state, so [`donated_since`] is then one atomic load.
static DONATED: Mutex<Option<HashMap<BufferId, u64>>> = Mutex::new(None);
static DONATED_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// How many times a capture has written each buffer in place ([`scatter_rows_into`]).
static GENERATIONS: Mutex<Option<HashMap<BufferId, u64>>> = Mutex::new(None);

/// Give `id` to the open capture as state it may write in place. Anything derived from it
/// and kept is dropped ([`written_in_place`]): the capture changes its contents.
pub(crate) fn donate(id: BufferId) {
    written_in_place(id);
    let generation = write_generation(id);
    let mut d = locked(&DONATED);
    if d.get_or_insert_with(Default::default)
        .insert(id, generation)
        .is_none()
    {
        DONATED_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// End `id`'s donation ([`donate`]).
pub(crate) fn undonate(id: BufferId) {
    let mut d = locked(&DONATED);
    if d.as_mut().is_some_and(|d| d.remove(&id).is_some()) {
        DONATED_COUNT.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// `id`'s generation when the open capture was given it, if it was ([`donate`]).
pub(crate) fn donated_since(id: BufferId) -> Option<u64> {
    if DONATED_COUNT.load(std::sync::atomic::Ordering::SeqCst) == 0 {
        return None;
    }
    locked(&DONATED).as_ref().and_then(|d| d.get(&id).copied())
}

/// How many times a capture has written `id` in place.
pub(crate) fn write_generation(id: BufferId) -> u64 {
    locked(&GENERATIONS)
        .as_ref()
        .and_then(|g| g.get(&id).copied())
        .unwrap_or(0)
}

/// `table` with `values`' row `i` written at its row `indices[i]`, in place
/// (`Session::scatter_rows_into`), for a buffer the open capture owns as its state
/// ([`donate`]). Returns the generation of `table`'s contents after the write: handles made
/// before it are refused from then on ([`crate::tensor::DramRef::current`]).
pub(crate) fn scatter_rows_into(
    device: TtDevice,
    table: BufferId,
    indices: BufferId,
    values: BlockArg,
    column: bool,
) -> Result<u64, TtError> {
    if donated_since(table).is_none() {
        return Err(TtError::Device(format!(
            "buffer {table} is not state of the open capture: it is not written in place"
        )));
    }
    run(device, move |engine, ids| {
        ids.healthy()?;
        let values = BlockArg {
            id: ids.get(values.id)?,
            ..values
        };
        engine.scatter_rows_into(ids.get(table)?, ids.get(indices)?, values, column)
    })?;
    let mut g = locked(&GENERATIONS);
    let n = g
        .get_or_insert_with(Default::default)
        .entry(table)
        .or_insert(0);
    *n += 1;
    Ok(*n)
}

pub fn copy_into(device: TtDevice, src: BufferId, dst: BufferId) -> Result<(), TtError> {
    written_in_place(dst);
    run(device, move |engine, ids| {
        ids.healthy()?;
        engine.copy_into(ids.get(src)?, ids.get(dst)?)
    })
}

/// [`run_generic_trace`] with the replay stamped ([`Engine::profile_generic_trace`]).
pub(crate) fn profile_generic_trace(
    device: TtDevice,
    trace: u64,
    inputs: Vec<(BufferId, InputPayload)>,
    outputs: Vec<(BufferId, OutputKind)>,
) -> Result<(GenericTraceRun, tt_kernels::profile::DeviceProfile), TtError> {
    run(device, move |engine, ids| {
        ids.healthy()?;
        let inputs = inputs
            .into_iter()
            .map(|(buf, p)| ids.get(buf).map(|b| (b, p)))
            .collect::<Result<Vec<_>, TtError>>()?;
        let outputs = outputs
            .into_iter()
            .map(|(buf, k)| ids.get(buf).map(|b| (b, k)))
            .collect::<Result<Vec<_>, TtError>>()?;
        engine.profile_generic_trace(trace, &inputs, &outputs)
    })
}

pub fn run_generic_trace(
    device: TtDevice,
    trace: u64,
    inputs: Vec<(BufferId, InputPayload)>,
    outputs: Vec<(BufferId, OutputKind)>,
) -> Result<GenericTraceRun, TtError> {
    run(device, move |engine, ids| {
        ids.healthy()?;
        let mapped_inputs: Result<Vec<(BufferId, InputPayload)>, TtError> = inputs
            .into_iter()
            .map(|(buf, p)| ids.get(buf).map(|b| (b, p)))
            .collect();
        let mapped_outputs: Result<Vec<(BufferId, OutputKind)>, TtError> = outputs
            .into_iter()
            .map(|(buf, k)| ids.get(buf).map(|b| (b, k)))
            .collect();
        engine.run_generic_trace(trace, &mapped_inputs?, &mapped_outputs?)
    })
}

/// `TT_SPREAD`'s value (`Session::set_spread`): `all`, a positive tile count, or
/// `work:<share>[:<min>]` with positive numbers. `None` for anything else.
fn parse_spread(v: &str) -> Option<tt_kernels::session::Spread> {
    use tt_kernels::session::Spread;
    let positive = |s: &str| s.parse().ok().filter(|&n: &usize| n > 0);
    match v.trim() {
        "all" => Some(Spread::All),
        s => match s.strip_prefix("work:") {
            Some(rest) => match rest.split_once(':') {
                Some((share, min)) => positive(share)
                    .zip(positive(min))
                    .map(|(share, min)| Spread::Work { share, min }),
                None => positive(rest).map(|share| Spread::Work { share, min: 1 }),
            },
            None => positive(s).map(Spread::Fixed),
        },
    }
}

/// Settings of the executors that were removed: refused with what replaces
/// them rather than ignored, so a script that still sets one learns why it no
/// longer does anything. `get` reads a variable, as `std::env::var` does.
fn refuse_retired_settings(get: impl Fn(&str) -> Option<String>) -> Result<(), TtError> {
    if let Some(value) = get("TT_EXACT") {
        return Err(TtError::Device(format!("TT_EXACT={value}: retired with the Flex cutover; native numerical bounds apply; unset it")));
    }
    for variable in ["TT_EXECUTION", "TT_SCATTER"] {
        if let Some(value) = get(variable) {
            return Err(TtError::Device(format!(
                "{variable}={value}: retired; GDDR compute always uses B-reader/NC-writer ownership; unset it (TT_PIPELINE=0 disables overlap without changing ownership)"
            )));
        }
    }
    Ok(())
}

/// A factory that opens `/dev/tenstorrent/{device.chip}` as a [`Session`] computing on
/// `tile` through `route` at `fidelity`, with an immutable arithmetic selection.
pub(crate) fn kmd_engine_with_elementwise(
    device: TtDevice,
    tile: TileChoice,
    route: SrcRoute,
    fidelity: Fidelity,
    elementwise: ElementwiseMode,
    matmul: MatmulMode,
) -> impl FnOnce(Serve) -> Result<(), TtError> + Send + 'static {
    move |serve| {
        let mut session = Session::open_card(device.chip, tt_firmware_images::ROLES, tile)
            .map_err(|e| crate::error::open_error(device.chip, e))?;
        // Tensors live in GDDR (Phase 9). Bit-identical to the host-staged path
        // (`step18_dram_matmul`), so on by default.
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .map_err(TtError::from)?;
        // `TT_ELTWISE` once chose between the SFPU and the data mover's FP32
        // unit; the mover does no arithmetic now. Refused rather than ignored,
        // so a script that still sets it learns why it no longer does anything.
        if let Ok(v) = std::env::var("TT_ELTWISE") {
            return Err(TtError::Device(format!(
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
            Ok(v) => return Err(TtError::Device(format!("TT_PIPELINE={v}: expected 0 or 1"))),
        }
        // `TT_SPREAD`: how many of the session's tiles each op runs on
        // (`Session::set_spread`): `all` (the default), a count, or
        // `work:<share>[:<min>]`, one tile for each `share` of the op's work and at
        // least `min` (1 if not given). The bits are the same.
        if let Ok(v) = std::env::var("TT_SPREAD") {
            match parse_spread(&v) {
                Some(spread) => session.set_spread(spread),
                None => {
                    return Err(TtError::Device(format!(
                        "TT_SPREAD={v}: expected `all`, a tile count, or `work:<share>[:<min>]`"
                    )))
                }
            }
        }
        // `TT_HOST_DMA=0`: tensors cross PCIe by the host's own stores and
        // loads through a BAR, not the card's DMA through pinned host memory
        // (`Session::set_host_dma`).
        match std::env::var("TT_HOST_DMA").as_deref() {
            Err(_) | Ok("1") => {}
            Ok("0") => session.set_host_dma(false),
            Ok(v) => return Err(TtError::Device(format!("TT_HOST_DMA={v}: expected 0 or 1"))),
        }
        // `TT_TILIZE=card`: tensors take the tile layout on the card's movers
        // rather than the host's cores (`Session::set_tilize`); `host`, the
        // default, is faster on every size card 0 measured.
        match std::env::var("TT_TILIZE").as_deref() {
            Err(_) | Ok("host") => {}
            Ok("card") => session.set_tilize(tt_kernels::session::Tilize::Card),
            Ok(v) => {
                return Err(TtError::Device(format!(
                    "TT_TILIZE={v}: expected host or card"
                )))
            }
        }
        refuse_retired_settings(|variable| std::env::var(variable).ok())?;
        // `TT_PROFILE=<path>`: a device-side profile of everything this
        // attachment runs, written as Chrome trace JSON when it detaches
        // (`tt_kernels::profile`). `{chip}` in the path becomes the card.
        //
        // `TT_PROFILE=<dir>`: the profile mode (`crate::profile`): every op's device time,
        // eager and traced, by Burn op, written to `<dir>/chip<N>/` when it detaches.
        // `TT_PROFILE_DETAIL=full` times data entries one by one.
        let profile_to = std::env::var("TT_PROFILE")
            .ok()
            .map(|p| p.replace("{chip}", &device.chip.to_string()));
        let (raw_profile, profile_dir) = match profile_to {
            Some(p) if p.ends_with(".json") => (Some(p), None),
            Some(p) => (None, Some(std::path::PathBuf::from(p))),
            None => (None, None),
        };
        if raw_profile.is_some() {
            session
                .profile_start()
                .map_err(|e| TtError::Device(format!("TT_PROFILE: {e}")))?;
        }
        let mut engine = KmdEngine {
            session,
            route,
            fidelity,
            budget: 400_000,
            buffers: Some(DramBuffers::default()),
            matmul,
        }
        .with_elementwise_mode(elementwise)?;
        if profile_dir.is_some() {
            engine
                .profile_mode(Some(crate::profile::options_from_env()))
                .map_err(|e| TtError::Device(format!("TT_PROFILE: {e}")))?;
        }
        let served = std::time::Instant::now();
        serve.serve(&mut engine);
        // `TT_HOST_TIMES=1`: where the server thread's time went queueing ops
        // (`Session::host_times`), against the whole attachment, and how many lists
        // each tile took, printed when it detaches.
        if std::env::var("TT_HOST_TIMES").as_deref() == Ok("1") {
            let total = served.elapsed();
            let h = engine.session.host_times();
            eprintln!(
                "burn-tt: host times, {} ops queued in {:.1} ms attached",
                h.ops,
                total.as_secs_f64() * 1e3
            );
            for (stage, t) in h.stages.iter().filter(|(_, t)| t.count > 0) {
                eprintln!(
                    "  {:<10} {:>9.1} ms  x{:<8} reads {:<8} writes {}",
                    format!("{stage:?}"),
                    t.time.as_secs_f64() * 1e3,
                    t.count,
                    t.traffic.read_calls,
                    t.traffic.write_calls
                );
            }
            let lists = engine.session.lists_per_tile();
            let used = lists.iter().filter(|&&l| l > 0).count();
            eprintln!(
                "  lists     {} on {used} of {} tiles; most on one tile {}",
                lists.iter().sum::<u64>(),
                lists.len(),
                lists.iter().max().copied().unwrap_or(0)
            );
        }
        if let Some(dir) = profile_dir {
            let dir = dir.join(format!("chip{}", device.chip));
            match engine.profile_mode(None) {
                Ok(Some(profile)) => match profile.write(&dir) {
                    Ok(()) => eprintln!("burn-tt: profile written to {}", dir.display()),
                    Err(e) => eprintln!("burn-tt: TT_PROFILE: {e}"),
                },
                Ok(None) => {}
                Err(e) => eprintln!("burn-tt: TT_PROFILE: {e}"),
            }
        }
        if let Some(path) = raw_profile {
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
    /// The math mode every chip's session runs in ([`Engine::set_math_mode`]).
    mode: tt_kernels::sfpu::approx::MathMode,
}

impl<T: tt_device::Transport> MeshEngine<T> {
    /// Exact matmul would run unpartitioned on chip 0 here: refused by name.
    pub fn with_matmul_mode(self, mode: MatmulMode) -> Result<Self, TtError> {
        if mode != MatmulMode::Fast {
            return Err(TtError::unsupported(
                "exact matmul mode is unsupported on mesh engines",
            ));
        }
        Ok(self)
    }

    pub fn with_elementwise_mode(self, mode: ElementwiseMode) -> Result<Self, TtError> {
        if mode != ElementwiseMode::Sfpu {
            return Err(TtError::unsupported(
                "matrix elementwise mode is unsupported on mesh engines",
            ));
        }
        Ok(self)
    }

    pub fn new(
        mut fabric: tt_kernels::shard::Fabric<T>,
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<Self, TtError> {
        fabric
            .enable_resident(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .map_err(TtError::from)?;
        let mode = fabric.chips[0].session().math_mode();
        Ok(Self {
            fabric,
            route,
            fidelity,
            budget,
            buffers: DramBuffers::default(),
            mode,
        })
    }
}

impl<T: tt_device::Transport> Engine for MeshEngine<T> {
    fn supports_exact_matmul(&self) -> bool {
        false
    }
    fn math_mode(&self) -> tt_kernels::sfpu::approx::MathMode {
        self.mode
    }
    fn set_math_mode(&mut self, mode: tt_kernels::sfpu::approx::MathMode) -> Result<(), TtError> {
        for chip in &mut self.fabric.chips {
            chip.session().set_math_mode(mode);
        }
        self.mode = mode;
        Ok(())
    }
    fn matmul(&mut self, a: &[f32], b: &[f32], [m, k, n]: [usize; 3]) -> Result<Vec<f32>, TtError> {
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
    ) -> Result<f32, TtError> {
        use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
        let mut temporary = Vec::new();
        let result = (|| {
            let input = self.upload(values, r, c)?;
            temporary.push(input);
            let (column, _) = self.reduce(input, ReduceOp::Sum, Axis::Cols, None)?;
            temporary.push(column);
            let (mut scalar, _) = self.reduce(column, ReduceOp::Sum, Axis::Rows, None)?;
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
    fn upload_bf16(&mut self, bits: &[u16], rows: usize, cols: usize) -> Result<BufferId, TtError> {
        self.buffers
            .upload_bf16(self.fabric.chips[0].session(), bits, rows, cols)
    }
    fn download_bf16(&mut self, id: BufferId) -> Result<Vec<u16>, TtError> {
        self.buffers
            .download_bf16(self.fabric.chips[0].session(), id)
    }
    fn cast_float(&mut self, id: BufferId, bf16: bool) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .cast_float(self.fabric.chips[0].session(), id, bf16)
    }
    fn pool_bf16(
        &mut self,
        a: BufferId,
        windows: &[Vec<[usize; 2]>],
        divisors: &[usize],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .pool_bf16(self.fabric.chips[0].session(), a, windows, divisors, dims)
    }
    fn matmul_bf16(&mut self, a: BufferId, b: BufferId) -> Result<(BufferId, [usize; 2]), TtError> {
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
                .map_err(TtError::from)?;
            Ok(self.buffers.insert(out))
        })();
        for id in temporary {
            self.buffers.free(self.fabric.chips[0].session(), id);
        }
        result
    }
    fn upload(&mut self, v: &[f32], rows: usize, cols: usize) -> Result<BufferId, TtError> {
        let b = &mut self.buffers;
        b.upload(self.fabric.chips[0].session(), v, rows, cols)
    }
    fn metadata(
        &mut self,
        bits: &[u32],
        dims: [usize; 2],
        elem: Elem,
    ) -> Result<BufferId, TtError> {
        self.buffers
            .metadata(self.fabric.chips[0].session(), bits, dims, elem)
    }
    fn download(&mut self, id: BufferId) -> Result<Vec<f32>, TtError> {
        let b = &mut self.buffers;
        b.download(self.fabric.chips[0].session(), id)
    }
    fn upload_bits(
        &mut self,
        v: &[u32],
        rows: usize,
        cols: usize,
        elem: Elem,
    ) -> Result<BufferId, TtError> {
        let b = &mut self.buffers;
        b.upload_bits(self.fabric.chips[0].session(), v, rows, cols, elem)
    }
    fn download_bits(&mut self, id: BufferId) -> Result<Vec<u32>, TtError> {
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let a = self.buffers.get(a)?.clone();
        let b = self.buffers.get(b)?.clone();
        let out = self
            .fabric
            .matmul_resident(&a, ta, &b, tb, self.route, self.fidelity, self.budget)
            .map_err(TtError::from)?;
        Ok(self.buffers.insert(out))
    }
    fn eltwise(
        &mut self,
        kind: u32,
        scalar: f32,
        a: BufferId,
        b: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.eltwise(self.fabric.chips[0].session(), kind, scalar, a, b)
    }
    fn eltwise_op(
        &mut self,
        op: tt_kernels::tensor::Eltwise,
        a: BufferId,
        b: Option<BufferId>,
        c: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.eltwise_op(self.fabric.chips[0].session(), op, a, b, c)
    }
    fn eltwise_blocks(
        &mut self,
        op: tt_kernels::tensor::Eltwise,
        a: BlockArg,
        b: Option<BlockArg>,
        c: Option<BlockArg>,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.eltwise_blocks(self.fabric.chips[0].session(), op, a, b, c)
    }
    fn rope_blocks(
        &mut self,
        x: BlockArg,
        cos: BlockArg,
        sin: BlockArg,
        head: usize,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.rope_blocks(self.fabric.chips[0].session(), x, cos, sin, head)
    }
    fn pow(&mut self, x: BufferId, y: PowArg) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.pow(self.fabric.chips[0].session(), x, y)
    }
    fn sum_rows(&mut self, a: BufferId) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.sum_rows(self.fabric.chips[0].session(), a)
    }
    fn reduce(
        &mut self,
        a: BufferId,
        op: tt_kernels::sfpu::reduce::ReduceOp,
        axis: tt_kernels::sfpu::reduce::Axis,
        then: tt_kernels::tensor::ReduceThen,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.reduce(self.fabric.chips[0].session(), a, op, axis, then)
    }
    fn rms(&mut self, a: BufferId, eps: f32) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.rms(self.fabric.chips[0].session(), a, eps)
    }
    fn softmax_rows(
        &mut self,
        x: BufferId,
        mask: BufferId,
        scale: f32,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.softmax_rows(self.fabric.chips[0].session(), x, mask, scale)
    }
    fn rms_norm(&mut self, a: BufferId, eps: f32) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.rms_norm(self.fabric.chips[0].session(), a, eps)
    }
    fn argmax_row(&mut self, a: BufferId) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.argmax_row(self.fabric.chips[0].session(), a)
    }
    fn scan(
        &mut self,
        a: BufferId,
        op: tt_kernels::sfpu::scan::ScanOp,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers.scan(self.fabric.chips[0].session(), a, op)
    }
    fn slice_rows(
        &mut self,
        a: BufferId,
        first: usize,
        rows: usize,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.slice_rows(a, first, rows)
    }
    fn matmul_dram_batched(
        &mut self,
        a: BufferId,
        b: BufferId,
        items: &[(Block, Block)],
        mkn: [usize; 3],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let [m, k, n] = mkn;
        if items.is_empty() || mkn.contains(&0) {
            return Err(TtError::Device(
                "mesh batched matmul requires nonempty products".into(),
            ));
        }
        let fits = |id, block: Block, rows, cols| -> Result<(), TtError> {
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
                return Err(TtError::Device(
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
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.copy_blocks(self.fabric.chips[0].session(), a, moves, dims)
    }
    fn repack(
        &mut self,
        a: BufferId,
        sources: &[[usize; 2]],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .repack(self.fabric.chips[0].session(), a, sources, dims)
    }
    fn zeros_dram(
        &mut self,
        dims: [usize; 2],
        elem: Elem,
        bf16: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .zeros(self.fabric.chips[0].session(), dims, elem, bf16)
    }
    fn gather_indexed(
        &mut self,
        input: BufferId,
        indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .gather_indexed(self.fabric.chips[0].session(), input, indices)
    }
    fn select_rows(
        &mut self,
        table: BufferId,
        indices: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .select_rows(self.fabric.chips[0].session(), table, indices)
    }
    fn select_taps(
        &mut self,
        table: BufferId,
        indices: BufferId,
        zero_row: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .select_taps(self.fabric.chips[0].session(), table, indices, zero_row)
    }
    fn repack_many(
        &mut self,
        inputs: &[BufferId],
        sources: &[(usize, [usize; 2])],
        dims: [usize; 2],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .repack_many(self.fabric.chips[0].session(), inputs, sources, dims)
    }

    fn gather_rows(
        &mut self,
        sources: &[BufferId],
        rows: &[(usize, usize)],
        cols: usize,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.gather_rows(self.fabric.chips[0].session(), sources, rows, cols)
    }
    fn rows_add(
        &mut self,
        t: BufferId,
        indices: &[usize],
        value: BufferId,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.rows_add(self.fabric.chips[0].session(), t, indices, value)
    }
    fn assign_rows(
        &mut self,
        t: BufferId,
        value: BufferId,
        rows: &[(usize, usize)],
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.assign_rows(self.fabric.chips[0].session(), t, value, rows)
    }
    fn scatter_rows(
        &mut self,
        table: BufferId,
        indices: BufferId,
        values: BufferId,
        column: bool,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        let bufs = &mut self.buffers;
        bufs.scatter_rows(
            self.fabric.chips[0].session(),
            table,
            indices,
            values,
            column,
        )
    }
    fn scatter_rows_into(
        &mut self,
        table: BufferId,
        indices: BufferId,
        values: BlockArg,
        column: bool,
    ) -> Result<(), TtError> {
        let bufs = &mut self.buffers;
        bufs.scatter_rows_into(
            self.fabric.chips[0].session(),
            table,
            indices,
            values,
            column,
        )
    }
    fn mesh_execution(&mut self) -> Option<tt_kernels::shard::FabricExecution> {
        Some(self.fabric.execution().clone())
    }
    fn device_traffic(&mut self) -> Option<tt_device::Traffic> {
        Some(self.fabric.chips[0].device().traffic())
    }

    fn sort_planes(
        &mut self,
        a: BufferId,
        spec: tt_kernels::sfpu::sort::Spec,
    ) -> Result<SortedBuffers, TtError> {
        self.buffers
            .sort_planes(self.fabric.chips[0].session(), a, spec)
    }

    fn random_dram(
        &mut self,
        dims: [usize; 2],
        draw: crate::random::Draw,
        base: u64,
    ) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .random(self.fabric.chips[0].session(), dims, draw, base)
    }

    fn cast_f16(&mut self, id: BufferId, to_f16: bool) -> Result<(BufferId, [usize; 2]), TtError> {
        self.buffers
            .cast_f16(self.fabric.chips[0].session(), id, to_f16)
    }

    /// Capture a mesh trace (`tt_kernels::mesh_trace`): one session trace per
    /// chip per stretch between Ethernet transfers, and the transfers
    /// themselves, replayed in the capture's order.
    fn begin_trace(&mut self) -> Result<(), TtError> {
        self.fabric.begin_mesh_trace().map_err(TtError::from)
    }
    fn end_trace(&mut self) -> Result<u64, TtError> {
        self.fabric.end_mesh_trace().map_err(TtError::from)
    }
    fn run_trace(
        &mut self,
        trace: u64,
        input: BufferId,
        values: &[f32],
        output: BufferId,
    ) -> Result<TraceRun, TtError> {
        use std::time::Instant;
        // The write waits for what was queued first: that wait is not the
        // write's.
        self.fabric.chips[0]
            .session()
            .sync()
            .map_err(TtError::from)?;
        let t0 = Instant::now();
        self.fabric.chips[0]
            .session()
            .write(self.buffers.get(input)?, values)
            .map_err(TtError::from)?;
        let t1 = Instant::now();
        self.fabric
            .replay_mesh_trace(trace)
            .map_err(TtError::from)?;
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
    ) -> Result<GenericTraceRun, TtError> {
        use std::time::Instant;
        self.fabric.chips[0]
            .session()
            .sync()
            .map_err(TtError::from)?;
        let t0 = Instant::now();
        self.buffers
            .write_trace_inputs(self.fabric.chips[0].session(), inputs)?;
        let t1 = Instant::now();
        self.fabric
            .replay_mesh_trace(trace)
            .map_err(TtError::from)?;
        let t2 = Instant::now();
        let out_data = self
            .buffers
            .read_trace_outputs(self.fabric.chips[0].session(), outputs)?;
        Ok(GenericTraceRun {
            outputs: out_data,
            write: t1 - t0,
            replay: t2 - t1,
            read: t2.elapsed(),
        })
    }
}

/// A factory for [`attach`] that opens every card in `cards` (the first is chip
/// 0, the host's way in and out), computes on `compute` and relays through
/// `relay` on each, finds the cabled links between them from the chips
/// (`tt_kernels::link::discover`), and serves a [`MeshEngine`].
pub(crate) fn kmd_mesh_engine(
    cards: Vec<u16>,
    compute: (u8, u8),
    relay: (u8, u8),
    route: SrcRoute,
    fidelity: Fidelity,
) -> impl FnOnce(Serve) -> Result<(), TtError> + Send + 'static {
    use tt_device::tlb::WindowKind;
    use tt_kernels::shard::{Chip, Fabric};
    move |serve| {
        refuse_retired_settings(|variable| std::env::var(variable).ok())?;
        let mut chips = Vec::new();
        for &card in &cards {
            let s = Session::open_card(
                card,
                tt_firmware_images::ROLES,
                TileChoice::Exactly(compute.0, compute.1),
            )
            .map_err(|e| crate::error::open_error(card, e))?;
            if !s.grid().contains(relay.0, relay.1) {
                return Err(TtError::Device(format!(
                    "relay tile {relay:?} is fused off on card {card}"
                )));
            }
            let tile = s.tile();
            let relay = tt_isa::noc::NocCoord::new(relay.0, relay.1).expect("a Tensix coordinate");
            chips.push(Chip::new(s.into_device(), tile, relay).map_err(TtError::from)?);
        }
        let mut links = Vec::new();
        for p in 0..chips.len() {
            for q in p + 1..chips.len() {
                let (l, r) = chips.split_at_mut(q);
                let (a, b) = (l[p].device(), r[0].device());
                let wa = a.alloc_window(WindowKind::TwoMib).map_err(TtError::from)?;
                let wb = b.alloc_window(WindowKind::TwoMib).map_err(TtError::from)?;
                let (ga, gb) = (
                    a.ethernet_grid(&wa).map_err(TtError::from)?,
                    b.ethernet_grid(&wb).map_err(TtError::from)?,
                );
                let found =
                    tt_kernels::link::discover(a, &wa, &ga, b, &wb, &gb).map_err(TtError::from)?;
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
        .map_err(TtError::from)?;
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
            ) -> Result<Vec<f32>, TtError> {
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

    /// A BF16 download, like the others, reports the attachment's first failure
    /// even though it reads a buffer that was never touched by it.
    #[test]
    fn a_bf16_download_reports_the_sticky_failure() {
        use super::*;
        struct Idle;
        impl Engine for Idle {
            fn matmul(&mut self, _: &[f32], _: &[f32], _: [usize; 3]) -> Result<Vec<f32>, TtError> {
                Err(unsupported())
            }
        }
        let device = TtDevice::new(u16::MAX - 1);
        let guard = attach(device, |serve| {
            serve.serve(&mut Idle);
            Ok(())
        })
        .unwrap();
        // An op on a buffer this attachment never made fails, asynchronously.
        let (broken, _) = cast_float(device, 999_999_999, [1, 1], true);
        let read = try_download_bf16(device, broken, 1, 1).unwrap_err();
        assert!(read.to_string().contains("outlived"), "{read}");
        // Another buffer's read is refused by the failure, not by itself.
        let (other, _) = cast_float(device, 999_999_998, [1, 1], true);
        let err = try_download_bf16(device, other, 1, 1).unwrap_err();
        assert!(matches!(err.root(), TtError::StaleTensor { .. }), "{err:?}");
        drop(guard);
    }

    #[test]
    fn tt_spread_takes_all_a_count_or_a_share_with_a_floor() {
        use tt_kernels::session::Spread;
        assert_eq!(super::parse_spread("all"), Some(Spread::All));
        assert_eq!(super::parse_spread(" 4 "), Some(Spread::Fixed(4)));
        assert_eq!(
            super::parse_spread("work:64"),
            Some(Spread::Work { share: 64, min: 1 })
        );
        assert_eq!(
            super::parse_spread("work:64:2"),
            Some(Spread::Work { share: 64, min: 2 })
        );
        for bad in [
            "",
            "0",
            "-1",
            "two",
            "work:",
            "work:0",
            "work:64:0",
            "work:64:x",
            "work:a:2",
        ] {
            assert_eq!(super::parse_spread(bad), None, "{bad:?}");
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
                .to_string();
            assert!(
                error.starts_with(&format!("{variable}={value}: retired")),
                "{error}"
            );
            assert!(error.contains("TT_PIPELINE=0"), "{error}");
        }
        assert!(refuse_retired_settings(|_| None).is_ok());
        let error = refuse_retired_settings(|name| (name == "TT_EXACT").then(|| "1".into()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("TT_EXACT=1: retired"), "{error}");
        assert!(
            error.contains("native numerical bounds") && error.contains("unset"),
            "{error}"
        );
    }
}
