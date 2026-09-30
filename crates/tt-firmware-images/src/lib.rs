//! Firmware images for the baby RISC-V cores.
//!
//! Built from `crates/tt-firmware` by this crate's `build.rs` and checked there
//! for instructions Blackhole cannot execute. The role images are what
//! `tt_kernels::runtime::run` loads; the rest serve the bring-up gates in
//! `tt-tests`, and live here so that the firmware is built in one place.

#![no_std]

use tt_isa::tensix::Core;

/// One image: the core it runs on, its bytes, and where it is loaded in L1.
///
/// The same shape as `tt_kernels::runtime::RoleImages`' entries.
pub type Image = (Core, &'static [u8], u64);

/// Where `link.x` places an image: RISCV T0's default reset PC, which every
/// image but the T1 and T2 role images uses (`crates/tt-firmware/build.rs`).
pub const LOAD_ADDRESS: u64 = Core::T0.default_reset_pc() as u64;

/// The three role images of the LLK-shaped datapath -- unpack on T0, math on
/// T1, pack on T2 -- each linked at its core's default reset PC, so all three fit
/// in one tile's L1 at once and none needs a reset-PC override.
pub const ROLES: [Image; 3] = [
    (
        Core::T0,
        include_bytes!(env!("FIRMWARE_ROLE_T0")),
        Core::T0.default_reset_pc() as u64,
    ),
    (
        Core::T1,
        include_bytes!(env!("FIRMWARE_ROLE_T1")),
        Core::T1.default_reset_pc() as u64,
    ),
    (
        Core::T2,
        include_bytes!(env!("FIRMWARE_ROLE_T2")),
        Core::T2.default_reset_pc() as u64,
    ),
];

/// The step 3 heartbeat: increments a counter in L1 forever.
pub const HEARTBEAT: &[u8] = include_bytes!(env!("FIRMWARE_HEARTBEAT"));

/// The step 4 gate: computes 3.0 x 2.0 on the Vector Unit and publishes the FP32
/// bit pattern.
pub const SFPU_MUL: &[u8] = include_bytes!(env!("FIRMWARE_SFPU_MUL"));

/// The generic single-thread Tensix program runner, built for T1 (the one core
/// ttsim lets read `Dst`, divergence row 12).
pub const CORPUS: &[u8] = include_bytes!(env!("FIRMWARE_CORPUS"));

/// [`CORPUS`] built for T0, which is where silicon runs single-thread programs
/// (divergence row 45).
pub const CORPUS_T0: &[u8] = include_bytes!(env!("FIRMWARE_CORPUS_T0"));

/// The Ethernet-tile image, for RISCV E1, loaded at `tt_isa::eth::E1_IMAGE`.
pub const ETH_E1: &[u8] = include_bytes!(env!("FIRMWARE_ETH_E1"));

/// The ELF entry points `build.rs` read, by image name.
const ENTRIES: [(&str, &str); 8] = [
    ("heartbeat", env!("FIRMWARE_HEARTBEAT_ENTRY")),
    ("sfpu_mul", env!("FIRMWARE_SFPU_MUL_ENTRY")),
    ("corpus", env!("FIRMWARE_CORPUS_ENTRY")),
    ("corpus_t0", env!("FIRMWARE_CORPUS_T0_ENTRY")),
    ("role_t0", env!("FIRMWARE_ROLE_T0_ENTRY")),
    ("role_t1", env!("FIRMWARE_ROLE_T1_ENTRY")),
    ("role_t2", env!("FIRMWARE_ROLE_T2_ENTRY")),
    ("eth_e1", env!("FIRMWARE_ETH_E1_ENTRY")),
];

/// The entry point of the image called `name`, as linked.
pub fn entry(name: &str) -> Option<u32> {
    ENTRIES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, e)| e.parse().expect("build.rs writes a decimal entry point"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A role image loaded anywhere but where it was linked runs someone else's
    /// code, or nothing. The load addresses above are derived from the cores'
    /// reset PCs and the link scripts say the same thing independently, so this
    /// checks the two against each other.
    #[test]
    fn each_role_image_is_linked_where_it_is_loaded() {
        for ((core, image, at), name) in ROLES.iter().zip(["role_t0", "role_t1", "role_t2"]) {
            assert!(!image.is_empty(), "{name} is empty");
            assert_eq!(
                entry(name).map(u64::from),
                Some(*at),
                "{name} is linked away from {core:?}'s default reset PC"
            );
        }
    }

    #[test]
    fn every_other_image_is_linked_at_the_load_address() {
        for name in ["heartbeat", "sfpu_mul", "corpus", "corpus_t0"] {
            assert_eq!(entry(name).map(u64::from), Some(LOAD_ADDRESS), "{name}");
        }
    }

    /// `link_e1.x` spells `E1_IMAGE` as a number; this is what holds it to it, and
    /// to the firmware-owned L1 the image must stay out of.
    #[test]
    fn the_ethernet_image_is_linked_where_it_is_loaded() {
        use tt_isa::eth;
        assert_eq!(entry("eth_e1").map(u64::from), Some(eth::E1_IMAGE));
        assert!(ETH_E1.len() as u64 <= eth::E1_IMAGE_MAX);
        assert!(eth::is_customer_l1(eth::E1_IMAGE, eth::E1_IMAGE_MAX));
    }
}
