//! Burn backend tensor tests, against burn-tt.
//!
//! The tests are upstream's (`vendor/burn-backend-tests`, `cargo xtask
//! fetch-burn-tests`); only the backend is ours.

#![recursion_limit = "256"]
#![allow(clippy::single_range_in_vec_init)]

extern crate alloc;

pub type FloatElem = f32;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/setup.rs"]
mod setup;

#[path = "common/backend.rs"]
mod backend;
pub use backend::*;

// Upstream's code, verbatim: neither formatted nor linted here (`cargo fmt`
// follows `#[path]` modules, and the pin is a digest of the files as fetched).
#[rustfmt::skip]
#[allow(clippy::all, unused, warnings)]
#[path = "../../../vendor/burn-backend-tests/tests/common/tensor.rs"]
mod tensor;
