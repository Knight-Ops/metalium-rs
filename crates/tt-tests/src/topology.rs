//! ttsim's multi-chip topologies, measured.
//!
//! ttsim does not model the base firmware's chip-info exchange (divergence row
//! 58), so the link map the silicon path reads from the chips
//! (`tt_kernels::link::discover`) is not available there. These tables are what
//! sending found instead: a TT-link write from each Up tile lands on exactly one
//! tile of one other chip (`probe_eth::{host_driven_tt_link_l1_write,
//! bh_x4_link_map}`, `step13_ethernet`). One link per adjacent pair, as
//! `(chip at a, chip at b, (a's X, b's X))`.

use tt_isa::eth::Ethernet;
use tt_kernels::link::Link;

/// `bh_x2` (a P300): the pair's first link. The other is chip 0 X 15 <-> chip 1 X 5.
pub const BH_X2: [(usize, usize, (u8, u8)); 1] = [(0, 1, (2, 12))];

/// `bh_x4` (two P300s): a ring 0-1-2-3-0, two links per pair, first of each.
pub const BH_X4: [(usize, usize, (u8, u8)); 4] = [
    (0, 1, (12, 2)),
    (1, 2, (3, 3)),
    (2, 3, (12, 2)),
    (3, 0, (3, 3)),
];

/// A table as the `(p, q, Link)` list `tt_kernels::shard::Fabric::new` takes.
/// ttsim's chips have every Ethernet tile ([`Ethernet::FULL`]).
pub fn links(table: &[(usize, usize, (u8, u8))]) -> Vec<(usize, usize, Link)> {
    let t = |x| {
        Ethernet::FULL
            .tile(x)
            .expect("ttsim has every Ethernet tile")
    };
    table
        .iter()
        .map(|&(p, q, (a, b))| (p, q, Link { a: t(a), b: t(b) }))
        .collect()
}
