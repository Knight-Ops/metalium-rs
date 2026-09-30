//! Test-support crate.
//!
//! Carries no production code. It exists so that `tests/` has a home that may
//! depend on both `tt-device` and `tt-ttsim` — `tt-device` must never depend on
//! the simulator binding, so cross-cutting gates cannot live there — and so that
//! the gates can share one harness.

pub mod backend;
pub mod burn_device;
pub mod harness;
pub mod mnist;
#[cfg(not(feature = "silicon"))]
pub mod topology;

/// The kernels moved to `tt-kernels`, which ships; re-exported so the gates
/// that established them still read `tt_tests::datapath`.
pub use tt_kernels::{datapath, matmul};

/// Firmware images, built from `crates/tt-firmware` by `tt-firmware-images`
/// and checked there for instructions Blackhole cannot execute.
pub mod firmware {
    pub use tt_firmware_images::{HEARTBEAT, LOAD_ADDRESS, ROLES, SFPU_MUL};

    /// The generic single-thread Tensix program runner, built for the core
    /// [`crate::harness::CORE`] names: T1 on the simulator, T0 on silicon. See
    /// `tt_firmware::corpus`.
    #[cfg(not(feature = "silicon"))]
    pub const CORPUS: &[u8] = tt_firmware_images::CORPUS;
    #[cfg(feature = "silicon")]
    pub const CORPUS: &[u8] = tt_firmware_images::CORPUS_T0;
}
