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

mod generated;
mod host;
mod ops;
mod random;
mod report;
mod server;
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
    attach, device_traffic, kmd_engine, kmd_mesh_engine, AttachGuard, BufferId, DramBuffers, Elem,
    Engine, EngineError, GenericTraceRun, InputPayload, KmdEngine, MeshEngine, OutputKind,
    OutputPayload, PowArg, Serve, TraceRun,
};
pub use tensor::{TtQTensor, TtTensor};
pub use trace::{StepTiming, Trace, TracedInference, TracedTrainingStep};

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
            DType::I32 => DTypeUsage::Storage.into(),
            _ => DTypeUsageSet::empty(),
        }
    }
}
