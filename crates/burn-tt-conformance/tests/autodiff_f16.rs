//! Burn autodiff tests with `f16` as the float element, against `Autodiff<burn-tt>` with the
//! default (no checkpointing) strategy.
//!
//! Upstream's `autodiff_f16.rs` runs the suite this way on the GPU backends. The tests are
//! upstream's (`vendor/burn-backend-tests`, `cargo xtask fetch-burn-tests`); only the backend is ours.

#![recursion_limit = "256"]
#![allow(clippy::single_range_in_vec_init, clippy::duplicate_mod)]

extern crate alloc;

pub type FloatElem = burn_tensor::f16;
#[allow(unused)]
pub type IntElem = i32;
pub type CheckpointStrategy = burn_autodiff::checkpoint::strategy::NoCheckpointing;

#[path = "common/setup.rs"]
mod setup;

#[path = "common/autodiff_backend.rs"]
mod backend;
pub use backend::*;

// Upstream's code, verbatim: neither formatted nor linted here (`cargo fmt`
// follows `#[path]` modules, and the pin is a digest of the files as fetched).
#[rustfmt::skip]
#[allow(clippy::all, unused, warnings)]
#[path = "../../../vendor/burn-backend-tests/tests/autodiff/mod.rs"]
mod base;
