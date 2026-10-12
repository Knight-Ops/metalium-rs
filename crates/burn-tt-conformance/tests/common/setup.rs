//! What every entry binary shares: the backend under test and its engine.
//!
//! Upstream's `common/backend.rs` names `burn_dispatch::Dispatch`, a closed enum
//! that cannot hold `TtBackend`, so each entry replaces it with its own
//! `backend.rs` and takes the parts that do not differ from here.

// `reference-flex` runs burn-flex on the host. It must never be mistaken for a run of burn-tt, so
// Cargo's feature unification (`--all-features`, `--features silicon,reference-flex`) cannot combine them.
#[cfg(all(
    feature = "reference-flex",
    any(feature = "silicon", feature = "fusion")
))]
compile_error!(
    "`reference-flex` runs burn-flex on the host and cannot be combined with `silicon` or `fusion`"
);

/// The backend under test: `TtBackend`, or `Fusion<TtBackend>` with `--features fusion`.
#[cfg(all(not(feature = "fusion"), not(feature = "reference-flex")))]
pub type Base = burn_tt::TtBackend;
#[cfg(all(feature = "fusion", not(feature = "reference-flex")))]
pub type Base = burn_tt::Tt;
/// With `--features reference-flex` the suite runs on burn-flex instead, on the host: the second
/// implementation that tells a test whose tolerance a dtype cannot meet from a burn-tt defect.
#[cfg(feature = "reference-flex")]
pub type Base = burn_flex::Flex;

/// The device a test asks for (only the autodiff entries use it).
#[allow(dead_code)]
pub type Device = <Base as burn_tensor::backend::BackendTypes>::Device;

/// Make the engine a device nobody attached opens: the simulator on ttsim, and on
/// silicon whatever `burn-tt` opens by default (card 0, `TT_TOPOLOGY`/`TT_TILES`).
///
/// The tests call `Default::default()` for their device, as a user's code does,
/// so nothing attaches explicitly. Runs once per process, before any test: the
/// runner starts one process per test, so the simulator (a once-per-process
/// singleton) is never shared.
pub fn register_engine() {
    #[cfg(all(not(feature = "silicon"), not(feature = "reference-flex")))]
    tt_tests::burn_device::register_simulator(tt_tests::burn_device::Config::default());
}

/// The default element types, which must be set before the first tensor exists
/// (`burn_tensor::set_default_dtypes`).
pub fn set_default_dtypes<B: burn_tensor::backend::Backend>(
    float: burn_tensor::DType,
    int: burn_tensor::DType,
) {
    let device = B::Device::default();
    burn_tensor::set_default_dtypes::<B>(&device, float, int).unwrap();
}
