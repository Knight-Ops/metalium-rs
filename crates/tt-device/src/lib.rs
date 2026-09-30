//! Host-side access to a Blackhole device.
//!
//! This crate is written entirely against the [`Transport`] trait so that the
//! simulator and real silicon are the same code path. Nothing here may use
//! `#[cfg]` to distinguish them.

#![forbid(unsafe_code)]

pub mod arc_msg;
pub mod core_control;
pub mod device;
pub mod dram;
pub mod ethernet;
pub mod telemetry;
pub mod tlb;
pub mod trace;
pub mod transport;

pub use device::{Device, PowerPolicy, Tile, Traffic, Window};
pub use transport::{
    Bar, ConfigOffset, Result, Transport, TransportError, DEVICE_ID_BLACKHOLE,
    VENDOR_ID_TENSTORRENT,
};
