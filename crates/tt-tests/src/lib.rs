//! Test-support crate.
//!
//! Carries no production code. It exists so that `tests/` has a home that may
//! depend on both `tt-device` and `tt-ttsim` — `tt-device` must never depend on
//! the simulator binding, so cross-cutting gates cannot live there — and so that
//! `build.rs` can build the device firmware and hand the images to the tests.

pub mod backend;
pub mod datapath;
pub mod harness;

/// Firmware images, built from `crates/tt-firmware` by this crate's `build.rs`
/// and checked for instructions Blackhole cannot execute.
pub mod firmware {
    /// The step 3 heartbeat: increments a counter in L1 forever.
    pub const HEARTBEAT: &[u8] = include_bytes!(env!("FIRMWARE_HEARTBEAT"));

    /// The step 4 gate: computes 3.0 x 2.0 on the Vector Unit and publishes the
    /// FP32 bit pattern.
    pub const SFPU_MUL: &[u8] = include_bytes!(env!("FIRMWARE_SFPU_MUL"));

    /// A generic Tensix program runner: pushes the instruction stream the host
    /// staged in L1 and copies the requested `Dst` rows back out.
    ///
    /// Adding a case to the instruction corpus is a host-side change, not a
    /// firmware one.
    ///
    /// Built for the core [`crate::harness::CORE`] names: T1 on the simulator, T0
    /// on silicon. See `tt_firmware::corpus`.
    #[cfg(not(feature = "silicon"))]
    pub const CORPUS: &[u8] = include_bytes!(env!("FIRMWARE_CORPUS"));
    #[cfg(feature = "silicon")]
    pub const CORPUS: &[u8] = include_bytes!(env!("FIRMWARE_CORPUS_T0"));

    /// Where a firmware image must be loaded in L1.
    ///
    /// Fixed by `crates/tt-firmware/link.x`, which places `.text` at RISCV T0's
    /// default reset PC so the image runs whether or not the loader programs the
    /// reset-PC override.
    pub const LOAD_ADDRESS: u64 = 0x6000;
}
