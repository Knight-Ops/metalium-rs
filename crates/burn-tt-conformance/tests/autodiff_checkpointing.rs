//! Burn autodiff tests, against `Autodiff<burn-tt, BalancedCheckpointing>`: the
//! same tests as `autodiff.rs`, which upstream runs a second time through
//! `DispatchDevice::autodiff_checkpointed`.

#![recursion_limit = "256"]
#![allow(clippy::single_range_in_vec_init, clippy::duplicate_mod)]

extern crate alloc;

pub type FloatElem = f32;
#[allow(unused)]
pub type IntElem = i32;
pub type CheckpointStrategy = burn_autodiff::checkpoint::strategy::BalancedCheckpointing;

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
