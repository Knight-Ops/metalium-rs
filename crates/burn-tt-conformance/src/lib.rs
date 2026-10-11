//! Burn's own backend test suite, run against `burn-tt`.
//!
//! The tests live in `vendor/burn-backend-tests/` (`cargo xtask fetch-burn-tests`)
//! and are compiled by the targets under `tests/`, which replace upstream's
//! `common/backend.rs` and `common/autodiff.rs`: those name `burn_dispatch`'s
//! closed enum, which cannot hold `TtBackend`. See `README.md`.

/// The attribute the vendored tests import as `burn_backend_tests::might_panic`.
#[cfg(feature = "run")]
pub use burn_tensor_testgen::might_panic;
