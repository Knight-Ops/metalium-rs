//! Blackhole ISA definitions: coordinate spaces, register maps, instruction encoders.
//!
//! `no_std` with zero dependencies, because this crate is compiled twice: once for
//! the host, to *generate* instruction streams, and once for `riscv32im`, to *push*
//! them. Anything that cannot build for both belongs somewhere else.
//!
//! Every constant here cites the specification page it came from. The Blackhole
//! documentation tree is a delta over Wormhole's, so a fact sourced from a Wormhole
//! page is a hypothesis until verified — such facts are marked `UNVERIFIED`.

#![no_std]
#![forbid(unsafe_code)]

pub mod adc;
pub mod arc;
pub mod backend;
pub mod cfg;
pub mod dataflow;
pub mod dm;
pub mod dram;
pub mod eth;
pub mod frontend;
pub mod isa;
pub mod l1;
pub mod l1_atomic;
pub mod mailbox;
pub mod matrix;
pub mod matrix_debug;
pub mod mmio_reg;
pub mod mutex;
pub mod noc;
pub mod numerics;
pub mod pack_modes;
/// Checked thread-local scalar register arithmetic.
pub mod scalar;
pub mod sfpu;
pub mod sfpu_macro;
pub mod sync;
pub mod tag_search;
pub mod tensix;
pub mod tile;
pub mod unpack_modes;
