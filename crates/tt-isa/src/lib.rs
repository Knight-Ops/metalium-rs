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

pub mod mailbox;
pub mod noc;
pub mod sfpu;
pub mod tensix;
