//! A Burn backend for Tenstorrent Blackhole.
//!
//! **What runs where.** Every op runs on the host through `burn-flex` except
//! the ones listed in `xtask/src/gen_burn.rs`'s `OVERRIDDEN`: matmul, add, sub,
//! mul, mul by a scalar, the sum over rows, row slices and transposes, and the
//! ReLU pair, each on the device when its F32 matrix operands are already there
//! (tensors live in GDDR once uploaded, `tensor::TtTensor`), and on the host
//! otherwise -- counted by [`tensor_traffic`] and named with
//! `TT_TRACE_FALLBACK=1`. The forwarding is generated from the
//! pinned `burn-backend`'s op traits (`cargo xtask gen-burn-delegate`), so
//! `burn-tt` behaves exactly as Flex does wherever it has not been told
//! otherwise, and each op moved to the device is a change behind an unchanged
//! interface, gated against the delegate it replaces.
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
//! device nobody attached, or a device run that fails, **panics** with the
//! error. The device path is never silently replaced by the host one.

mod convert;
mod generated;
mod ops;
mod server;
mod tensor;
mod topology;
mod trace;
mod traffic;

pub use server::{
    attach, device_traffic, kmd_engine, kmd_mesh_engine, AttachGuard, BufferId, DramBuffers, Elem,
    Engine, EngineError, KmdEngine, MeshEngine, PowArg, Serve,
};
pub use tensor::{TtQTensor, TtTensor};
pub use trace::Trace;

/// Keep on the device only what gives `burn-flex`'s bits exactly.
///
/// By default the device runs every op it has, and some are approximations
/// held to derived bounds rather than to Flex's bits: division and the
/// reciprocal (one ulp), `exp` and `log`, sums over columns (a different
/// order), and softmax built from them. A run that must reproduce a host
/// golden bit for bit -- the MNIST golden is one -- sets this, and those ops
/// run on the host instead; so does `TT_EXACT=1`. Per process.
pub fn set_exact(on: bool) {
    EXACT.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

/// Is exact mode on ([`set_exact`], or `TT_EXACT=1`)?
pub fn exact() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match EXACT.load(Relaxed) {
        0 => {
            let on = std::env::var("TT_EXACT").is_ok_and(|v| v == "1");
            EXACT.store(if on { 2 } else { 1 }, Relaxed);
            on
        }
        v => v == 2,
    }
}

/// 0: not yet read from the environment; 1: off; 2: on.
static EXACT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
pub use topology::{attach_topology, parse_tiles, tiles_from_env, Topology};
pub use traffic::{
    device_time, record_transfers, take_transfers, tensor_traffic, Direction, TensorTraffic,
    Transfer,
};
pub use tt_kernels::matmul::{Fidelity, SrcRoute};
pub use tt_kernels::session::TileChoice;

use burn_backend::{Backend, BackendTypes, DType, DTypeUsageSet, DeviceId, DeviceOps};
use burn_flex::{Flex, FlexDevice};

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

    fn seed(_device: &Self::Device, seed: u64) {
        // Every op that draws random numbers runs on Flex, from Flex's seed.
        Flex::seed(&FlexDevice, seed)
    }

    fn device_count(_type_id: u16) -> usize {
        server::attached_count()
    }

    fn dtype_usage(_device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        // What Flex supports: the host runs everything the device does not.
        Flex::dtype_usage(&FlexDevice, dtype)
    }
}
