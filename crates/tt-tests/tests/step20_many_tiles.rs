//! Phase 9.6 gate: one session computing on many Tensix tiles.
//!
//! A `Session` opened with `TileChoice::Count(n)` deals each GDDR op's jobs
//! (`tt_kernels::tensor::Job`) over `n` tiles -- matmul output blocks with `K`
//! whole, element-wise tile runs, column-sum column runs -- and runs them in
//! waves, every tile at once. The claim is that the number of tiles changes no
//! bit of any result:
//!
//! * a GDDR matmul equals the host-staged matmul on the session's first tile,
//!   which `step18_dram_matmul` ties to the one-tile GDDR path;
//! * element-wise ops and column sums equal `burn-flex`, as `step19_eltwise`
//!   holds the one-tile path to;
//! * and every tile did a share, so "the same bits" cannot mean the first tile
//!   did all the work.
//!
//! The shapes go past MNIST's on purpose: MNIST is the example workload, not
//! the target, and its tensors are too small to give eight tiles a block each.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_device::Transport;
use tt_isa::dm::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::Eltwise;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;

/// Tile counts every gate runs at: one, two, an odd count that divides
/// nothing evenly, and more tiles than some shapes have output blocks.
const COUNTS: [usize; 4] = [1, 2, 3, 8];

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

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = v[r * cols + c];
        }
    }
    t
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
}

fn flex(v: &[f32], r: usize, c: usize) -> Tensor<Flex, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), &FlexDevice)
}

fn flex_values(t: Tensor<Flex, 2>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{n} tiles: {e}");
    }
}

/// Every unit completed at least one step since `before`.
fn every_tile_worked(before: &[u64], after: &[u64], what: &str) {
    for (u, (b, a)) in before.iter().zip(after).enumerate() {
        assert!(
            a > b,
            "{what}: unit {u} did nothing ({before:?} -> {after:?})"
        );
    }
}

#[test]
fn a_session_opens_on_the_tiles_it_is_asked_for() {
    with_tiles(3, |s| {
        let tiles = s.tiles();
        assert_eq!(tiles.len(), 3);
        assert_eq!(tiles[0], s.tile());
        let distinct: std::collections::BTreeSet<_> =
            tiles.iter().map(|t| (t.x(), t.y())).collect();
        assert_eq!(distinct.len(), 3, "{tiles:?}");
        assert!(tiles.iter().all(|t| s.grid().contains(t.x(), t.y())));
    });
}

#[test]
fn more_tiles_than_the_chip_has_is_an_error() {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let have = tt_isa::noc::grid::Tensix::FULL.tile_count();
        let r = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(have + 1),
            |_, _| Ok(None),
        );
        match r {
            Err(tt_kernels::session::SessionError::TooFewTiles { asked, have: h }) => {
                assert_eq!((asked, h), (have + 1, have))
            }
            Err(e) => panic!("the wrong error: {e}"),
            Ok(_) => panic!("{} tiles opened on a chip with {have}", have + 1),
        }
    }) {
        panic!("{e}");
    }
}

/// MNIST's forward and backward products, a ragged transposed one, and two
/// larger than MNIST's, all with operands in GDDR, against the host-staged
/// matmul on the session's first tile.
#[test]
fn gddr_matmuls_over_many_tiles_are_bit_identical() {
    // (label, A stored [r, c], op(A) transposed?, B stored, op(B) transposed?)
    let cases = [
        ("x@W1", [64, 784], false, [784, 128], false),
        ("h@W2", [64, 128], false, [128, 10], false),
        ("g2@W2t", [64, 10], false, [128, 10], true),
        ("xt@g1", [64, 784], true, [64, 128], false),
        ("ragged", [37, 45], true, [37, 70], false),
        ("wide", [96, 320], false, [320, 288], false),
        ("tall", [300, 96], false, [160, 96], true),
    ];
    for n in COUNTS {
        with_tiles(n, |s| {
            for (i, (label, [ar, ac], ta, [br, bc], tb)) in cases.into_iter().enumerate() {
                let (av, bv) = (
                    floats(10 + i as u64, ar * ac),
                    floats(20 + i as u64, br * bc),
                );
                let (a_op, [m, k]) = if ta {
                    (transpose(&av, ar, ac), [ac, ar])
                } else {
                    (av.clone(), [ar, ac])
                };
                let (b_op, cols) = if tb {
                    (transpose(&bv, br, bc), br)
                } else {
                    (bv.clone(), bc)
                };
                let want = s
                    .matmul(&a_op, &b_op, [m, k, cols], ROUTE, Fidelity::HiFi4, BUDGET)
                    .unwrap_or_else(|e| panic!("{e}"));
                let a = s.upload(&av, ar, ac).unwrap();
                let b = s.upload(&bv, br, bc).unwrap();
                let before = s.steps_per_tile();
                let c = s
                    .matmul_dram(&a, ta, &b, tb, ROUTE, Fidelity::HiFi4, BUDGET)
                    .unwrap_or_else(|e| panic!("{label} on {n} tiles: {e}"));
                let after = s.steps_per_tile();
                assert_bits(
                    &s.download(&c).unwrap(),
                    &want,
                    &format!("{label}, {n} tiles"),
                );
                // Every shape here has at least `n` output tiles but `h@W2`
                // and `g2@W2t` (two each), so all but those give every unit
                // a block.
                if m.div_ceil(32) * cols.div_ceil(32) >= n {
                    every_tile_worked(&before, &after, &format!("{label}, {n} tiles"));
                }
                for t in [a, b, c] {
                    s.free(t).unwrap();
                }
            }
        });
    }
}

#[test]
fn eltwise_over_many_tiles_matches_flex_bit_for_bit() {
    for n in COUNTS {
        with_tiles(n, |s| {
            // Ragged, MNIST's activations, and more tiles than one list holds.
            for [r, c] in [[37, 70], [64, 128], [784, 128]] {
                let (av, bv) = (floats(r as u64, r * c), floats(c as u64 + 99, r * c));
                let row = floats(5, c);
                let a = s.upload(&av, r, c).unwrap();
                let b = s.upload(&bv, r, c).unwrap();
                let bias = s.upload(&row, 1, c).unwrap();
                let (fa, fb) = (flex(&av, r, c), flex(&bv, r, c));
                let fbias = flex(&row, 1, c);
                let cases: Vec<(&str, u32, f32, bool, Vec<f32>)> = vec![
                    (
                        "add",
                        kind::ADD,
                        0.0,
                        false,
                        flex_values(fa.clone() + fb.clone()),
                    ),
                    (
                        "mul",
                        kind::MUL,
                        0.0,
                        false,
                        flex_values(fa.clone() * fb.clone()),
                    ),
                    (
                        "relu",
                        kind::RELU,
                        0.0,
                        false,
                        flex_values(burn::tensor::activation::relu(fa.clone())),
                    ),
                    (
                        "add row",
                        kind::ADD_ROW,
                        0.0,
                        true,
                        flex_values(fa.clone() + fbias.clone()),
                    ),
                ];
                for (label, k, scalar, row_op, want) in cases {
                    let other = match k {
                        kind::RELU => None,
                        _ if row_op => Some(&bias),
                        _ => Some(&b),
                    };
                    let before = s.steps_per_tile();
                    let out = s
                        .eltwise(Eltwise { kind: k, scalar }, &a, other)
                        .unwrap_or_else(|e| panic!("{label} [{r}, {c}] on {n} tiles: {e}"));
                    let after = s.steps_per_tile();
                    let what = format!("{label} [{r}, {c}], {n} tiles");
                    assert_bits(&s.download(&out).unwrap(), &want, &what);
                    if r.div_ceil(32) * c.div_ceil(32) >= n {
                        every_tile_worked(&before, &after, &what);
                    }
                    s.free(out).unwrap();
                }
                for t in [a, b, bias] {
                    s.free(t).unwrap();
                }
            }
        });
    }
}

#[test]
fn column_sums_over_many_tiles_match_flex_bit_for_bit() {
    for n in COUNTS {
        with_tiles(n, |s| {
            // A 7000-row column spans several mover lists on whichever tile
            // has it; its rows must still add in order.
            for [r, c] in [[37, 70], [64, 128], [7000, 40], [96, 600]] {
                let av = floats(r as u64 * 3 + c as u64, r * c);
                let a = s.upload(&av, r, c).unwrap();
                let want = flex_values(flex(&av, r, c).sum_dim(0));
                let before = s.steps_per_tile();
                let out = s.sum_rows(&a).unwrap_or_else(|e| panic!("[{r}, {c}]: {e}"));
                let after = s.steps_per_tile();
                let what = format!("sum [{r}, {c}], {n} tiles");
                assert_bits(&s.download(&out).unwrap(), &want, &what);
                if c.div_ceil(32) >= n {
                    every_tile_worked(&before, &after, &what);
                }
                for t in [a, out] {
                    s.free(t).unwrap();
                }
            }
        });
    }
}

/// A kernel that never finishes on the first tile -- math waits on a
/// semaphore nothing posts -- is an error, and (on silicon) the session
/// recovers that tile and a matmul over all three tiles is right after it.
#[test]
fn a_failed_kernel_on_one_tile_leaves_every_tile_usable() {
    use tt_kernels::runtime::{Kernel, Schedule};
    with_tiles(3, |s| {
        let (ar, ac, bc) = (96, 64, 96);
        let (av, bv) = (floats(9, ar * ac), floats(10, ac * bc));
        let want = s
            .matmul(&av, &bv, [ar, ac, bc], ROUTE, Fidelity::HiFi4, BUDGET)
            .unwrap_or_else(|e| panic!("{e}"));
        // Any semaphore will do: math waits on one that starts at zero and
        // that nothing posts.
        let never = tt_isa::sync::Semaphore::new(0).unwrap();
        let stuck = tt_isa::sync::take(never, tt_isa::backend::Before::EVERYTHING).to_vec();
        let nothing: Vec<tt_isa::isa::Instruction> = Vec::new();
        let sems = [(never, 0, 1)];
        let k = Kernel {
            dump_rows: 0,
            ..Kernel::new([&nothing, &stuck, &nothing], Schedule::Concurrent(&sems))
        };
        assert!(s.run(&k, 50_000).is_err(), "the stuck kernel must fail");
        // Recovery is the backend pulse (`session::reset_tile`), which ttsim
        // refuses (divergence row 16), as `step17_resident` says. Silicon only.
        if s.device().transport().is_simulated() {
            return;
        }
        let a = s.upload(&av, ar, ac).unwrap();
        let b = s.upload(&bv, ac, bc).unwrap();
        let before = s.steps_per_tile();
        let c = s
            .matmul_dram(&a, false, &b, false, ROUTE, Fidelity::HiFi4, BUDGET)
            .unwrap_or_else(|e| panic!("after recovery: {e}"));
        every_tile_worked(&before, &s.steps_per_tile(), "after recovery");
        assert_bits(&s.download(&c).unwrap(), &want, "after recovery");
    });
}
