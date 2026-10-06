//! Host-side Tensix kernels, and the runner that executes them.
//!
//! A kernel is three instruction streams, one per Tensix thread, split the way
//! tt-metal's LLK splits them: thread 0 unpacks, thread 1 does the math, thread
//! 2 packs ([`runtime`]). The streams are built here from `tt_isa`'s encoders
//! ([`datapath`], [`matmul`]); the role firmware on each baby RISC-V pushes
//! them word for word and knows nothing about what they do.
//!
//! Everything in this crate was established one gate at a time in `tt-tests`
//! (`step8_eltwise`, `step9_matmul`, `step10_matmul_tile`) against ttsim and
//! both p150a cards, and moved here once it was the thing a caller -- a Burn
//! backend -- would need. The gates still run through it.
//!
//! The instruction builders `unwrap` their encoders: every operand is a
//! constant or checked before it gets there, so an encode failure is a bug in
//! this crate, not a condition a caller could handle. The runner, which talks
//! to a device, returns [`runtime::RunError`] instead.

pub mod bf16;
pub mod code;
pub mod datapath;
pub mod dm;
pub mod fpu;
mod index;
pub mod kind;
pub mod l1;
pub mod link;
pub mod loops;
pub mod matmul;
pub mod matrix_eltwise;
pub mod profile;
pub mod program_cache;
pub mod runtime;
pub mod session;
pub mod sfpu;
pub mod shard;
pub mod tensor;
pub mod trace;
