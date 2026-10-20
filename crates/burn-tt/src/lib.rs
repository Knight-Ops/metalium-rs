//! A Burn backend for Tenstorrent Blackhole.
//!
//! **What runs where.** Compute uses native Tenstorrent kernels. Supported
//! inputs are uploaded as needed; unsupported operations, shapes and dtypes
//! panic with context. Host buffers serve construction and explicit readback.
//! Burn's default operations compose this backend's primitives. The pinned
//! traits generate native dispatch and unsupported implementations
//! (`cargo xtask gen-burn-ops`). [`report`] records execution and transfers.
//!
//! **Devices.** A [`TtDevice`] names a chip, and needs no set-up: the first op
//! that needs it opens `/dev/tenstorrent/{chip}` (see [`registry`](crate::TtConfig):
//! [`init`] does it eagerly and returns the error, [`detach`] gives the card
//! back). The hardware behind a device is owned by a server thread that runs an
//! [`Engine`] factory *on* it. That is what lets the simulator -- a `!Send`
//! process-wide singleton -- serve a backend whose tensors must be `Send +
//! Sync`: it is created where it lives and never moves. [`set_default_engine`]
//! chooses what a device nobody attached opens, and [`attach`] attaches a
//! factory explicitly for the life of a guard; the simulator engine lives in
//! `tt-tests`, so nothing here depends on the simulator. A card has one owner:
//! a second session on it, in this process or another, is
//! [`TtError::CardBusy`].
//!
//! **Errors.** Failures are a [`TtError`]. Burn's ops return tensors, not
//! results, so a device op that fails does not stop the caller: device ops are
//! dispatched without waiting (`server::submit`), and the failure, naming the op
//! and the engine's error, is returned where a result is next waited for --
//! `try_into_data` and `Backend::sync` give `Err(ExecutionError)`, `into_data`
//! panics (Burn's own `expect`). The first failure on an attachment is reported
//! by every later wait; an engine that panics reports that panic the same way.
//! Strict mode ([`set_strict`], [`strictly`]) panics at the failing wait instead.
//! Ops that have no `Result` to defer to (a host-staged matmul or reduction) and
//! an op on a device nobody attached panic at once. The device path is never
//! silently replaced by the host one.

mod error;
#[cfg(feature = "fusion")]
pub mod fusion;
mod generated;
mod host;
mod matmul_mode;
mod ops;
pub mod profile;
mod random;
mod registry;
mod report;
mod server;
pub mod storage;
mod tensor;
mod topology;
mod trace;
mod traffic;
mod unsupported;
mod views;

pub use error::TtError;
pub use matmul_mode::{accurate_matmul, exact_matmul, with_matmul_mode, MatmulMode};
#[doc(hidden)]
pub use ops::rope::{rope, rope_count};
pub use ops::{
    set_accurate_matmul_slicing, set_argmax_row, set_attention_rows,
    set_exact_matmul_chunk_products,
};
pub use random::Draw;
pub use report::{
    host_ok, report, report_reset, server_report, server_report_reset, set_strict, strict,
    strictly, with_report, OpStat, Report, ServerStat,
};
pub use server::{
    attach, device_traffic, mesh_execution, AttachGuard, BlockArg, BufferId, DramBuffers, Elem,
    Engine, GenericTraceRun, InputPayload, KmdEngine, MeshEngine, OutputKind, OutputPayload,
    PowArg, Serve, SortedBuffers, TraceRun,
};
#[cfg(feature = "fusion")]
pub use tensor::TtHandle;
pub use tensor::{TtQTensor, TtTensor};
#[doc(hidden)]
pub use trace::{set_count_trace_bytes, set_profile_buffer, set_state_in_place};
pub use trace::{
    SetFootprint, SharedStateTraces, StepTiming, Trace, TracedInference, TracedTrainingStep,
};

#[cfg(feature = "fusion")]
pub type Tt = burn_fusion::Fusion<TtBackend>;
#[cfg(feature = "fusion")]
#[doc(hidden)]
pub use fusion::resolve_int_tensor;
#[cfg(feature = "fusion")]
pub use fusion::{
    fused_add_relu_count, fused_count, fused_relu_count, reset_fusion_counters,
    resolve_float_tensor,
};
#[cfg(not(feature = "fusion"))]
pub type Tt = TtBackend;

pub use registry::{
    attach_with, detach, init, refuse_default_engine, reset_default_engine, set_default_engine,
    EngineFactory, TtConfig,
};
pub use topology::{parse_tiles, tiles_from_env, Topology};
pub use traffic::{
    device_time, record_transfers, take_transfers, tensor_traffic, Direction, TensorTraffic,
    Transfer,
};
pub use tt_kernels::matmul::{Fidelity, SrcRoute};
pub use tt_kernels::session::TileChoice;
pub use tt_kernels::tensor::{Block, BlockMove};
pub use tt_kernels::trace::OpRecord;

use burn_backend::{
    Backend, BackendTypes, DType, DTypeUsage, DTypeUsageSet, DeviceId, DeviceOps, ExecutionError,
};

/// The Burn backend. All state lives behind [`TtDevice`]s; the type itself is a
/// marker, as Burn requires.
#[derive(Clone, Copy, Debug, Default)]
pub struct TtBackend;

/// One Blackhole chip, by index: `/dev/tenstorrent/{chip}` on silicon, chip
/// `chip` of the simulator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TtDevice {
    pub chip: u16,
}

impl TtDevice {
    pub const fn new(chip: u16) -> Self {
        TtDevice { chip }
    }

    /// Run this device's transcendentals (`exp`, `log`, `recip`, `sigmoid`,
    /// `tanh`, `gelu`) in `mode` from the next op on: [`MathMode::Precise`],
    /// the default, or the fast [`MathMode::Approx`] programs with their
    /// looser derived bounds (`tt_kernels::sfpu::approx`). `TT_MATH=approx`
    /// chooses it when the device attaches. Not the retired exact mode, which
    /// asked for the host's bits; this asks how many ulps. Fused programs
    /// (softmax, log-softmax, norms) are Precise either way.
    ///
    /// # Panics
    ///
    /// If the device is not attached.
    pub fn set_math_mode(&self, mode: MathMode) -> Result<(), TtError> {
        server::run(*self, move |engine, _| engine.set_math_mode(mode))
    }

    /// The mode [`TtDevice::set_math_mode`] set.
    ///
    /// # Panics
    ///
    /// If the device is not attached.
    pub fn math_mode(&self) -> MathMode {
        server::run(*self, |engine, _| engine.math_mode())
    }
}

impl core::fmt::Display for TtDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "tt:{}", self.chip)
    }
}

/// Burn's device identity: type 0, index = chip.
impl burn_backend::Device for TtDevice {
    fn to_id(&self) -> DeviceId {
        DeviceId::new(0, self.chip)
    }
    fn from_id(id: DeviceId) -> Self {
        TtDevice { chip: id.index_id }
    }
}

impl DeviceOps for TtDevice {}

impl BackendTypes for TtBackend {
    type Device = TtDevice;
    type FloatTensorPrimitive = TtTensor;
    type FloatElem = f32;
    type IntTensorPrimitive = TtTensor;
    type IntElem = i32;
    type BoolTensorPrimitive = TtTensor;
    type BoolElem = bool;
    type QuantizedTensorPrimitive = TtQTensor;
}

impl Backend for TtBackend {
    fn name(device: &Self::Device) -> String {
        format!("tt<{device}>")
    }

    fn seed(device: &Self::Device, seed: u64) {
        crate::random::seed(*device, seed)
    }

    /// The `/dev/tenstorrent/{n}` nodes, and the devices attached some other
    /// way ([`TtDevice::enumerate`]); nothing is opened to count them.
    fn device_count(type_id: u16) -> usize {
        // Every device is type 0 (`Device::to_id`).
        if type_id == 0 {
            TtDevice::enumerate().len()
        } else {
            0
        }
    }

    /// A barrier: every op sent to the device so far has run, and the
    /// first one that failed (or the server's panic) is the error -- the
    /// "any error since the last sync" `try_into_data` documents. Strict mode
    /// panics instead. A device nothing is attached to has nothing to wait
    /// for.
    fn sync(device: &Self::Device) -> Result<(), ExecutionError> {
        match server::try_sync(*device) {
            Ok(()) => Ok(()),
            Err(e) if report::strict_here() => panic!("{e}"),
            Err(e) => Err(ExecutionError::WithContext {
                reason: e.to_string(),
            }),
        }
    }

    fn dtype_usage(_device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        match dtype {
            DType::F32 | DType::BF16 => DTypeUsage::general() | DTypeUsage::Accelerated,
            // Int arithmetic is native (wrapping ALU, comparisons, reductions:
            // burn-op-notes S5/D3), so I32 can be a default dtype
            // (`set_default_dtypes`, which Burn's own backend tests call). Not
            // `Accelerated`: there is no int matmul.
            DType::I32 | DType::Bool(_) => DTypeUsage::general(),
            // FP16 (IEEE binary16) is stored in BF16's two-byte slots and its
            // arithmetic widens to F32 on Tensix, rounding once at the
            // operation boundary, as BF16's does. Not `Accelerated`: there is
            // no packed F16 matmul. F64 stays unsupported.
            DType::F16 => DTypeUsage::general(),
            _ => DTypeUsageSet::empty(),
        }
    }
}

pub use tt_kernels::matrix_eltwise::{ElementwiseMode, SrcPrecision};
pub use tt_kernels::sfpu::approx::MathMode;
