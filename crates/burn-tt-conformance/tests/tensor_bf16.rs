//! Burn backend tensor tests with `bf16` as the float element, against burn-tt.
//!
//! Upstream ships no BF16 entry (its `tensor_f16.rs` has `TODO: bf16`); this is that file with `bf16`.
//! The tests are upstream's (`vendor/burn-backend-tests`, `cargo xtask fetch-burn-tests`); only the
//! backend is ours. Int and bool are not repeated: they do not depend on the float element.
//!
//! Silicon only: ttsim refuses PACR mode `0x105`, the narrowing every BF16 result goes through
//! (`docs/learnings/ttsim-divergence.md`, "BF16 late narrowing"), so `cargo xtask conformance` runs this
//! binary only with `--silicon`.

#![recursion_limit = "256"]
#![allow(clippy::single_range_in_vec_init)]

extern crate alloc;

pub type FloatElem = burn_tensor::bf16;
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
mod bf16;
