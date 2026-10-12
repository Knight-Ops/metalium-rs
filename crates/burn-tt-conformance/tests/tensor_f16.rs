//! Burn backend tensor tests with `f16` as the float element, against burn-tt.
//!
//! Upstream's `tensor_f16.rs` runs the `tensor/float` modules this way on the GPU backends. The tests
//! are upstream's (`vendor/burn-backend-tests`, `cargo xtask fetch-burn-tests`); only the backend is
//! ours. Int and bool are not repeated: they do not depend on the float element.

#![recursion_limit = "256"]
#![allow(clippy::single_range_in_vec_init)]

extern crate alloc;

pub type FloatElem = burn_tensor::f16;
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
#[path = "../../../vendor/burn-backend-tests/tests/tensor/float/mod.rs"]
mod f16;
