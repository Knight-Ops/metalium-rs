//! Element-wise ops over only the rows a tile row has (`docs/plans/decode-to-bandwidth.md`
//! Phase 3): a tensor of one tile row of fewer than 32 rows -- a decode step's `[1, W]` -- runs a
//! lanewise kind's programs over only the row groups its rows occupy
//! (`tt_kernels::sfpu::ops::code_for_rows`: one group of eight for one row), on the card.
//!
//! The claims:
//! - **Same bits.** Each kind the limit applies to (`sfpu::ops::row_limited`: add, subtract,
//!   multiply, divide, the scalar add and multiply, sigmoid and `silu(gate) * up`) equals the
//!   specification model's run of its whole-tile program (`sfpu::ops::reference`) at 1, 3, 5 and
//!   17 rows, widths 2048 and 70 (ragged), plain and with a row and a column broadcast.
//! - **Every row used is computed.** Before each op, the same op over a `[32, W]` tensor of NaN
//!   runs on the same unit: its NaN fills the same L1 slots and every row of `Dst`, so a row
//!   whose group the programs skip shows as NaN.
//! - **An honest claim:** the output's padding rows are whatever `Dst` held, claimed
//!   `Pad::Undefined`, and a ragged result then feeds a sum over its rows and a matmul along
//!   them (its rows as `K`) exactly, on small integers.
//!
//! Watched failing: with each program's last row group dropped (a used row reads the primed
//! NaN), here and in the unit gate (`row_limited_programs_keep_the_computed_rows`). A false
//! `Pad::Zero` claim did not show in the sum or the product on ttsim (the sum masks padding,
//! and the product's `K` padding met zeros), so the claim itself is asserted.

use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::{self, kind_sfpu};
use tt_kernels::tensor::{Eltwise, Pad};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let scale = [1.0e-3f32, 1.0, 37.0][i % 3];
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        use tt_tests::backend::GATE_TILE;
        let tiles = TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1);
        let mut s = Session::open(dev, tt_firmware_images::ROLES, tiles, |_, _| Ok(None)).unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(1),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn op(kind: u32, scalar: f32) -> Eltwise {
    Eltwise {
        kind,
        scalar,
        scalar2: 0.0,
    }
}

#[derive(Clone, Copy, Debug)]
enum Second {
    None,
    Same,
    Row,
    Col,
}

#[test]
fn row_limited_ops_are_the_models_bit_for_bit_over_primed_l1() {
    let cases = [
        (
            kind::ADD,
            0.0,
            [Second::Same, Second::Row, Second::Col].as_slice(),
        ),
        (kind::SUB, 0.0, &[Second::Same, Second::Row]),
        (kind::MUL, 0.0, &[Second::Same, Second::Row, Second::Col]),
        (kind_sfpu::DIV, 0.0, &[Second::Same, Second::Col]),
        (kind_sfpu::SILU_MUL, 0.0, &[Second::Same]),
        (kind::MUL_SCALAR, 0.125, &[Second::None]),
        (kind::ADD_SCALAR, 1e-5, &[Second::None]),
        (kind_sfpu::SIGMOID, 0.0, &[Second::None]),
    ];
    with_session(|s| {
        for cols in [2048usize, 70] {
            for &(kind, scalar, seconds) in &cases {
                assert!(ops::row_limited(kind));
                for &second in seconds {
                    for rows in [1usize, 3, 5, 17] {
                        let label = format!("{kind:#x} {second:?} [{rows}, {cols}]");
                        let x = floats((rows * 10_000 + cols) as u64 + kind as u64, rows * cols);
                        let (b_rows, b_cols) = match second {
                            Second::None => (0, 0),
                            Second::Same => (rows, cols),
                            Second::Row => (1, cols),
                            Second::Col => (rows, 1),
                        };
                        let y = floats(7 + kind as u64, b_rows * b_cols);
                        let want = ops::reference(
                            kind,
                            scalar,
                            &x,
                            (b_rows > 0).then_some(&y[..]),
                            rows,
                            cols,
                        );
                        let a = s.upload(&x, rows, cols).unwrap();
                        let b = (b_rows > 0).then(|| s.upload(&y, b_rows, b_cols).unwrap());
                        // The same op over whole tiles of NaN on the same unit: its slots and
                        // `Dst` rows hold NaN when the op under test starts.
                        let nan = s.upload(&vec![f32::NAN; 32 * cols], 32, cols).unwrap();
                        let nan_b = (b_rows > 0).then(|| {
                            let r = if b_rows == rows { 32 } else { b_rows };
                            let (r, c) = match second {
                                Second::Col => (32, 1),
                                _ => (r, b_cols),
                            };
                            s.upload(&vec![f32::NAN; r * c], r, c).unwrap()
                        });
                        let primed = s.eltwise(op(kind, scalar), &nan, nan_b.as_ref()).unwrap();
                        let got = s.eltwise(op(kind, scalar), &a, b.as_ref()).unwrap();
                        assert_eq!(got.pad(), Pad::Undefined, "{label}: claim");
                        let got_v = s.download(&got).unwrap();
                        for (i, (g, w)) in got_v.iter().zip(&want).enumerate() {
                            assert_eq!(
                                g.to_bits(),
                                w.to_bits(),
                                "{label}: datum ({}, {}): {g} for {w}",
                                i / cols,
                                i % cols
                            );
                        }
                        for t in [Some(primed), Some(got), Some(a), b, Some(nan), nan_b]
                            .into_iter()
                            .flatten()
                        {
                            s.free(t).unwrap();
                        }
                    }
                }
            }
        }
    });
}

/// A ragged row-limited result whose padding rows hold NaN feeds a sum over its rows and a
/// matmul that takes its rows as `K`: both exact, on small integers. The claim is what lets
/// them refill or mask the padding.
#[test]
fn a_row_limited_result_feeds_a_sum_and_a_product_exactly() {
    with_session(|s| {
        let (rows, cols) = (17usize, 70usize);
        let x: Vec<f32> = (0..rows * cols).map(|i| (i % 7) as f32 - 3.0).collect();
        let y: Vec<f32> = (0..rows * cols).map(|i| (i % 5) as f32 - 2.0).collect();
        let nan = s.upload(&vec![f32::NAN; 32 * cols], 32, cols).unwrap();
        let primed = s.eltwise(op(kind::ADD, 0.0), &nan, Some(&nan)).unwrap();
        let (a, b) = (
            s.upload(&x, rows, cols).unwrap(),
            s.upload(&y, rows, cols).unwrap(),
        );
        let sum = s.eltwise(op(kind::ADD, 0.0), &a, Some(&b)).unwrap();
        assert_eq!(sum.pad(), Pad::Undefined);
        let host: Vec<f32> = x.iter().zip(&y).map(|(p, q)| p + q).collect();
        // Over its rows: each column's sum.
        let col_sums = s.sum_rows(&sum).unwrap();
        let want: Vec<f32> = (0..cols)
            .map(|c| (0..rows).map(|r| host[r * cols + c]).sum())
            .collect();
        assert_eq!(s.download(&col_sums).unwrap(), want, "the sum over rows");
        // Its rows as `K`: `[5, 17] @ [17, 70]`.
        let w: Vec<f32> = (0..5 * rows).map(|i| (i % 3) as f32 - 1.0).collect();
        let wt = s.upload(&w, 5, rows).unwrap();
        let prod = s
            .matmul_dram(
                &wt,
                false,
                &sum,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                BUDGET,
            )
            .unwrap();
        let want: Vec<f32> = (0..5 * cols)
            .map(|i| {
                let (m, c) = (i / cols, i % cols);
                (0..rows)
                    .map(|k| w[m * rows + k] * host[k * cols + c])
                    .sum()
            })
            .collect();
        assert_eq!(
            s.download(&prod).unwrap(),
            want,
            "the product along its rows"
        );
        for t in [nan, primed, a, b, sum, col_sums, wt, prod] {
            s.free(t).unwrap();
        }
    });
}
