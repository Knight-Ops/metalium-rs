//! Phase 9.7 gate: one host launch per op per tile.
//!
//! The data mover runs the resident roles itself (`tt_isa::dm::op::KERNEL`),
//! so a tile's whole share of a GDDR op -- every block's gather, kernel and
//! scatter, or every run of element-wise tiles -- is one mover list: one
//! submission and one wait, however many blocks it has. Lists break only
//! where they must: at `tt_isa::dm::LIST_MAX` entries, or where a tile's
//! kernel programs change. The claims:
//!
//! * the results are the bits the host-sequenced path gave (the matmuls
//!   against the host-staged path, as `step18_dram_matmul`; element-wise and
//!   column sums against `burn-flex`, as `step19_eltwise`);
//! * an op that fits one list per tile is exactly one list per tile, on one
//!   tile and on three, where the host-sequenced path took one per block step;
//! * and since 9.7b, which sends op records (`tt_isa::dm::record`) that the
//!   mover expands on the tile, "fits" covers any op of these kinds: a
//!   record is a few entries however many tiles it names.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::Eltwise;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
}

#[cfg(not(feature = "silicon"))]
fn with_tiles(n: usize, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(n),
            |_, _| Ok(None),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{n} tiles: {e}");
    }
}

#[cfg(feature = "silicon")]
fn with_tiles(n: usize, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(n),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{n} tiles: {e}");
    }
}

/// Lists submitted to each tile by `op`.
fn lists<S>(s: &mut S, lists_of: impl Fn(&S) -> Vec<u64>, op: impl FnOnce(&mut S)) -> Vec<u64> {
    let before = lists_of(s);
    op(s);
    lists_of(s)
        .iter()
        .zip(&before)
        .map(|(a, b)| a - b)
        .collect()
}

/// MNIST's first layer is several blocks on one tile -- its operands do not
/// fit the staging area in one -- and each block was a gather, a kernel and a
/// scatter, each its own host round trip. Now it is one list per tile.
#[test]
fn a_multi_block_matmul_is_one_list_per_tile() {
    for n in [1, 3] {
        with_tiles(n, |s| {
            let ([m, k], cols) = ([64, 784], 128);
            let (av, bv) = (floats(1, m * k), floats(2, k * cols));
            let want = s
                .matmul(&av, &bv, [m, k, cols], ROUTE, Fidelity::HiFi4, BUDGET)
                .unwrap_or_else(|e| panic!("{e}"));
            let a = s.upload(&av, m, k).unwrap();
            let b = s.upload(&bv, k, cols).unwrap();
            let mut c = None;
            let per_tile = lists(s, Session::lists_per_tile, |s| {
                c = Some(
                    s.matmul_dram(&a, false, &b, false, ROUTE, Fidelity::HiFi4, BUDGET)
                        .unwrap_or_else(|e| panic!("{e}")),
                );
            });
            assert_eq!(per_tile, vec![1; n], "{n} tiles: lists per tile");
            let c = c.unwrap();
            assert_bits(&s.download(&c).unwrap(), &want, &format!("x@W1, {n} tiles"));
        });
    }
}

/// An element-wise op over more tiles than one run of the staging area holds
/// (96) was a list per run; the runs now share a list, a `WAIT` apart.
#[test]
fn eltwise_runs_share_a_list() {
    with_tiles(1, |s| {
        let (r, c) = (320, 320); // 100 tiles: two runs of the staging area.
        let (av, bv) = (floats(3, r * c), floats(4, r * c));
        let want: Vec<f32> =
            (Tensor::<Flex, 2>::from_data(TensorData::new(av.clone(), [r, c]), &FlexDevice)
                + Tensor::<Flex, 2>::from_data(TensorData::new(bv.clone(), [r, c]), &FlexDevice))
            .into_data()
            .to_vec()
            .unwrap();
        let a = s.upload(&av, r, c).unwrap();
        let b = s.upload(&bv, r, c).unwrap();
        let mut out = None;
        let per_tile = lists(s, Session::lists_per_tile, |s| {
            out = Some(
                s.eltwise(
                    Eltwise {
                        scalar2: 0.0,
                        kind: kind::ADD,
                        scalar: 0.0,
                    },
                    &a,
                    Some(&b),
                )
                .unwrap_or_else(|e| panic!("{e}")),
            );
        });
        assert_eq!(per_tile, vec![1], "lists");
        assert_bits(&s.download(&out.unwrap()).unwrap(), &want, "add");
    });
}

/// A column sum over 7000 rows -- 219 row tiles, too many for one SFPU run --
/// is done in chunks, each starting from the last's sums
/// (`sfpu::reduce::ROW_CHUNK`), all one job on the tile: two host round trips,
/// against the 878-entry pair of lists it once was, and its rows still add in
/// Flex's order to the bit.
#[test]
fn a_long_column_sum_is_one_job_in_flex_order() {
    with_tiles(1, |s| {
        let (r, c) = (7000, 40);
        let av = floats(5, r * c);
        let want: Vec<f32> =
            Tensor::<Flex, 2>::from_data(TensorData::new(av.clone(), [r, c]), &FlexDevice)
                .sum_dim(0)
                .into_data()
                .to_vec()
                .unwrap();
        let a = s.upload(&av, r, c).unwrap();
        let mut out = None;
        let per_tile = lists(s, Session::lists_per_tile, |s| {
            out = Some(s.sum_rows(&a).unwrap_or_else(|e| panic!("{e}")));
        });
        assert_bits(&s.download(&out.unwrap()).unwrap(), &want, "sum");
        assert!(per_tile.iter().all(|&l| l <= 2), "lists: {per_tile:?}");
    });
}
