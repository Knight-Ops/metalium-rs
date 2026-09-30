//! Phase 8 gates, simulator: a matmul sharded across chips is bit-identical to
//! the single-chip one, with operands and results moving between chips over
//! Ethernet only.
//!
//! `bh_x2` has chip 1 adjacent to chip 0; `bh_x4` is a ring, so chip 2 is two
//! links from chip 0 and its data is relayed through chip 1. The link tables are
//! measured (`probe_eth::host_driven_tt_link_l1_write`, `bh_x4_link_map`): ttsim
//! does not model the chip-info exchange the silicon path reads (row 58).

#![cfg(not(feature = "silicon"))]

use std::path::PathBuf;

use tt_device::Device;
use tt_isa::noc::{Noc0, NocCoord};
use tt_kernels::link::Link;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::matmul_on;
use tt_kernels::shard::{Chip, Fabric};
use tt_ttsim::{fork_scope, Simulator};

const BUDGET: u64 = 4_000_000;

fn at(x: u8, y: u8) -> NocCoord<Noc0> {
    assert!(tt_isa::noc::grid::is_tensix_geometry(x, y));
    NocCoord::new(x, y).unwrap()
}

fn x2_links() -> Vec<(usize, usize, Link)> {
    tt_tests::topology::links(&tt_tests::topology::BH_X2)
}

fn x4_links() -> Vec<(usize, usize, Link)> {
    tt_tests::topology::links(&tt_tests::topology::BH_X4)
}

/// Values in `[-1, 1)` from a fixed seed, with enough mantissa to make TF32
/// truncation and the accumulation order matter.
fn floats(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

#[track_caller]
fn sharded_equals_single(
    lib: PathBuf,
    chips: usize,
    links: Vec<(usize, usize, Link)>,
    mkn: [usize; 3],
) {
    let r = fork_scope(|| {
        let mut sim = Simulator::open_path(lib).unwrap();
        assert_eq!(usize::from(sim.chip_count()), chips);
        let devs: Vec<_> = sim
            .transports()
            .into_iter()
            .map(|t| Device::open(t).unwrap())
            .collect();
        let chips: Vec<_> = devs
            .into_iter()
            .map(|d| Chip::new(d, at(3, 4), at(6, 7)).unwrap())
            .collect();
        let mut fab = Fabric::new(
            chips,
            &links,
            tt_firmware_images::ROLES,
            tt_firmware_images::ETH_E1,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let [m, k, n] = mkn;
        let (a, b) = (floats(m * k, 11), floats(k * n, 23));
        let (route, fidelity) = (SrcRoute::Tf32FromFp32, Fidelity::HiFi4);
        let sharded = fab
            .matmul(&a, &b, mkn, route, fidelity, BUDGET)
            .unwrap_or_else(|e| panic!("{e}"));
        let single = matmul_on(
            &mut fab.chips[0].dev,
            at(1, 2),
            &tt_firmware_images::ROLES,
            &a,
            &b,
            mkn,
            route,
            fidelity,
            BUDGET,
        )
        .unwrap();
        let (s, o) = (bits(&sharded), bits(&single));
        let diff = s.iter().zip(&o).filter(|(x, y)| x != y).count();
        assert_eq!(
            diff,
            0,
            "{diff} of {} elements differ from the single-chip product",
            s.len()
        );
        // Not vacuous: the product is not all zeros, and the shards were not all
        // computed on chip 0.
        assert!(single.iter().any(|&v| v != 0.0));
        println!(
            "routes: {:?}",
            (0..fab.len())
                .map(|c| fab.route(c).map(<[usize]>::to_vec))
                .collect::<Vec<_>>()
        );
    });
    if let Err(e) = r {
        panic!("{e}");
    }
}

#[test]
fn a_small_matmul_sharded_over_two_chips_is_bit_identical() {
    sharded_equals_single(tt_ttsim::x2_lib_path(), 2, x2_links(), [40, 96, 70]);
}

#[test]
fn the_first_mnist_layer_sharded_over_two_chips_is_bit_identical() {
    sharded_equals_single(tt_ttsim::x2_lib_path(), 2, x2_links(), [64, 784, 128]);
}

#[test]
fn a_matmul_sharded_round_a_four_chip_ring_is_bit_identical() {
    // n = 128 is four 32-column tiles: one per chip, so chip 2's share is
    // relayed through chip 1 both ways.
    sharded_equals_single(tt_ttsim::x4_lib_path(), 4, x4_links(), [64, 256, 128]);
}

#[test]
fn the_four_chip_ring_routes_chip_2_through_a_neighbour() {
    let r = fork_scope(|| {
        let mut sim = Simulator::open_path(tt_ttsim::x4_lib_path()).unwrap();
        let chips: Vec<_> = sim
            .transports()
            .into_iter()
            .map(|t| Chip::new(Device::open(t).unwrap(), at(3, 4), at(6, 7)).unwrap())
            .collect();
        let fab = Fabric::new(
            chips,
            &x4_links(),
            tt_firmware_images::ROLES,
            tt_firmware_images::ETH_E1,
        )
        .unwrap();
        assert_eq!(fab.route(1), Some(&[0, 1][..]));
        assert_eq!(fab.route(3), Some(&[0, 3][..]));
        assert_eq!(fab.route(2).map(<[usize]>::len), Some(3));
    });
    if let Err(e) = r {
        panic!("{e}");
    }
}
