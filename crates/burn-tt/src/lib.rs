//! A Burn backend for Tenstorrent Blackhole.
//!
//! **What runs where.** Compute uses native Tenstorrent kernels. Supported
//! inputs are uploaded as needed; unsupported operations, shapes and dtypes
//! panic with context. Host buffers serve construction and explicit readback.
//! Burn's default operations compose this backend's primitives. The pinned
//! traits generate native dispatch and unsupported implementations
//! (`cargo xtask gen-burn-ops`). [`report`] records execution and transfers.
//!
//! **Devices.** A [`TtDevice`] names a chip; the hardware behind it is owned by
//! a server thread started with [`attach`], which runs a caller-supplied
//! [`Engine`] factory *on* that thread. That is what lets the simulator -- a
//! `!Send` process-wide singleton -- serve a backend whose tensors must be
//! `Send + Sync`: it is created where it lives and never moves. [`kmd_engine`]
//! is the silicon engine; the simulator engine lives in `tt-tests`, so nothing
//! here depends on the simulator.
//!
//! **Errors.** Burn's ops return tensors, not results, so a device op on a
//! device nobody attached **panics** at once, and a device run that fails
//! panics at the next wait for a result (device ops are dispatched without
//! waiting, `server::submit`), naming the op and the engine's error. The device
//! path is never silently replaced by the host one.

#[cfg(feature = "fusion")]
pub mod fusion;
mod generated;
mod host;
mod ops;
mod random;
mod report;
mod server;
pub mod storage;
mod tensor;
mod topology;
mod trace;
mod traffic;
mod unsupported;
mod views;

pub use report::{
    host_ok, report, report_reset, set_strict, strict, strictly, with_report, OpStat, Report,
};
pub use server::{
    attach, device_traffic, kmd_engine, kmd_mesh_engine, mesh_execution, AttachGuard, BufferId,
    DramBuffers, Elem, Engine, EngineError, GenericTraceRun, InputPayload, KmdEngine, MeshEngine,
    OutputKind, OutputPayload, PowArg, Serve, SortedBuffers, TraceRun,
};
#[cfg(feature = "fusion")]
pub use tensor::TtHandle;
pub use tensor::{TtQTensor, TtTensor};
pub use trace::{StepTiming, Trace, TracedInference, TracedTrainingStep};

#[cfg(feature = "fusion")]
pub type Tt = burn_fusion::Fusion<TtBackend>;
#[cfg(feature = "fusion")]
pub use fusion::{
    fused_add_relu_count, fused_relu_count, reset_fusion_counters, resolve_float_tensor,
};
#[cfg(not(feature = "fusion"))]
pub type Tt = TtBackend;

pub use topology::{attach_topology, parse_tiles, tiles_from_env, Topology};
pub use traffic::{
    device_time, record_transfers, take_transfers, tensor_traffic, Direction, TensorTraffic,
    Transfer,
};
pub use tt_kernels::matmul::{Fidelity, SrcRoute};
pub use tt_kernels::session::TileChoice;
pub use tt_kernels::tensor::{Block, BlockMove};

use burn_backend::{Backend, BackendTypes, DType, DTypeUsage, DTypeUsageSet, DeviceId, DeviceOps};

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
    pub fn set_math_mode(&self, mode: MathMode) -> Result<(), EngineError> {
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

    fn device_count(_type_id: u16) -> usize {
        server::attached_count()
    }

    fn dtype_usage(_device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        match dtype {
            DType::F32 | DType::BF16 => DTypeUsage::general() | DTypeUsage::Accelerated,
            DType::Bool(_) => DTypeUsage::general(),
            // FP16 (IEEE binary16): stored and cast on Tensix (lane T7); arithmetic is
            // not claimed, F64 stays unsupported.
            DType::I32 | DType::F16 => DTypeUsage::Storage.into(),
            _ => DTypeUsageSet::empty(),
        }
    }
}

pub use server::kmd_engine_with_elementwise;
pub use tt_kernels::matrix_eltwise::{ElementwiseMode, SrcPrecision};
pub use tt_kernels::sfpu::approx::MathMode;

pub use topology::attach_topology_with_elementwise;
