//! Burn autodiff tests with `bf16` as the float element, against `Autodiff<burn-tt>` with the
//! default (no checkpointing) strategy.
//!
//! Upstream ships no BF16 entry (its `autodiff_f16.rs` is the model); the tests are upstream's
//! (`vendor/burn-backend-tests`, `cargo xtask fetch-burn-tests`); only the backend is ours.
//!
//! Silicon only: ttsim refuses PACR mode `0x105` (`docs/learnings/ttsim-divergence.md`, "BF16 late
//! narrowing"), so `cargo xtask conformance` runs this binary only with `--silicon`.

#![recursion_limit = "256"]
#![allow(clippy::single_range_in_vec_init, clippy::duplicate_mod)]

extern crate alloc;

pub type FloatElem = burn_tensor::bf16;
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
