//! A `[rows, 1]` operand broadcast across a `[rows, cols]` one by the SFPU, not the mover
//! (`docs/plans/decode-to-bandwidth.md` Phase 3, the small ops): the mover reads the column's
//! tile as it lies (`record::READ_RUN_COL_AS_IT_LIES`), and the program first copies column 0
//! into every column in `Dst` (`sfpu::Program::broadcast_col0`). The mover's datum-at-a-time
//! copy (`READ_BROADCAST_COL`) cost ~10 us a tile: RMSNorm's division and the softmax's
//! `SUB_EXP` and division a decode step.
//!
//! The claims, on the card:
//! - **Same bits.** `ADD`, `SUB`, `MUL`, `DIV` and `SUB_EXP` with a column operand equal the
//!   specification model's run of the plain program over the column expanded on the host to the
//!   full shape (`sfpu::ops::reference`, no broadcast in it), at 1, 3, 17, 32 and 45 rows (one
//!   tile row, row-limited, and two), widths 2048, 70 and 64.
//! - **Only column 0 is read.** The column operand is made on the card as `c / 1` over a
//!   zero-padded `[rows, 1]`, so its padding columns are `0 / 0`, NaN, in GDDR -- asserted --
//!   and its value in column 0 is downloaded for the oracle.
//! - Before each op, the same op over NaN runs on the same unit, so L1 slots and `Dst` hold NaN.
//!
//! Watched failing (ttsim): with the program's broadcast dropped (`code_for`'s column arm
//! returning the plain program), every case reads the NaN columns.

use tt_kernels::kind;
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::{self, kind_sfpu};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_ttsim::fork_scope;

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if i % 11 == 4 {
                return [0.0, -0.0, f32::INFINITY, f32::MAX, -1.0e-40][(s >> 60) as usize % 5];
            }
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

fn op(kind: u32) -> Eltwise {
    Eltwise {
        kind,
        scalar: 0.0,
        scalar2: 0.0,
    }
}

/// `[rows, 1]` whose column 0 holds `values / 1` and whose padding columns hold NaN in GDDR.
fn nan_padded_column<T: tt_device::Transport>(
    s: &mut Session<T>,
    values: &[f32],
) -> (DramTensor, Vec<f32>) {
    let rows = values.len();
    let c = s.upload(values, rows, 1).unwrap();
    let ones = s.upload(&vec![1.0; rows], rows, 1).unwrap();
    let col = s.eltwise(op(kind_sfpu::DIV), &c, Some(&ones)).unwrap();
    let padded = s.download_padded(&col).unwrap();
    let rt = rows.div_ceil(32);
    assert_eq!(padded.len(), rt * 32 * 32, "a [{rows}, 1] tensor's tiles");
    for r in 0..rows {
        for c in 1..32 {
            assert!(
                padded[r * 32 + c].is_nan(),
                "[{rows}, 1]: padding column {c} of row {r} is {}",
                padded[r * 32 + c]
            );
        }
    }
    let value = s.download(&col).unwrap();
    s.free(c).unwrap();
    s.free(ones).unwrap();
    (col, value)
}

#[test]
fn a_column_broadcast_in_the_sfpu_is_the_expanded_operand_s_result() {
    let kinds = [
        kind::ADD,
        kind::SUB,
        kind::MUL,
        kind_sfpu::DIV,
        kind_sfpu::SUB_EXP,
    ];
    with_session(|s| {
        for (rows, cols) in [(1usize, 2048usize), (3, 70), (17, 64), (32, 70), (45, 64)] {
            for kind in kinds {
                let label = format!("{kind:#x} [{rows}, {cols}] with [{rows}, 1]");
                let x = floats((rows * 1_000 + cols) as u64 + kind as u64, rows * cols);
                let (b, col) = nan_padded_column(s, &floats(kind as u64 + 17, rows));
                let expanded: Vec<f32> = (0..rows * cols).map(|i| col[i / cols]).collect();
                let want = ops::reference(kind, 0.0, &x, Some(&expanded), rows, cols);
                let a = s.upload(&x, rows, cols).unwrap();
                let nan = s.upload(&vec![f32::NAN; 32 * cols], 32, cols).unwrap();
                let nan_b = s.upload(&[f32::NAN; 32], 32, 1).unwrap();
                let primed = s.eltwise(op(kind), &nan, Some(&nan_b)).unwrap();
                let got = s.eltwise(op(kind), &a, Some(&b)).unwrap();
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
                for t in [a, b, nan, nan_b, primed, got] {
                    s.free(t).unwrap();
                }
            }
        }
    });
}

/// Rows long enough to hold a run (`tensor::row_runs`): each run keeps to one tile row and reads
/// that row's one tile of the column once, held in one slot for the run (`b_once`), instead of
/// once a tile (the long-context softmax's `SUB_EXP` and division, `docs/plans/long-context.md`,
/// "Long-prompt softmax passes"). The same bits as the specification model over the expanded
/// column, and as the runs that read it a tile (`tensor::set_col_once(false)`), with L1 and `Dst`
/// primed with NaN first. Watched failing (ttsim): the run's tile of `B` read from the next row.
#[test]
fn a_long_row_reads_its_column_tile_once_with_the_same_bits() {
    let kinds = [kind::SUB, kind_sfpu::DIV, kind_sfpu::SUB_EXP];
    with_session(|s| {
        for (rows, cols) in [(45usize, 32 * 90usize), (96, 32 * 120 + 7)] {
            for kind in kinds {
                let label = format!("{kind:#x} [{rows}, {cols}] with [{rows}, 1]");
                let x = floats((rows * 1_000 + cols) as u64 + kind as u64, rows * cols);
                let (b, col) = nan_padded_column(s, &floats(kind as u64 + 29, rows));
                let expanded: Vec<f32> = (0..rows * cols).map(|i| col[i / cols]).collect();
                let want = ops::reference(kind, 0.0, &x, Some(&expanded), rows, cols);
                let a = s.upload(&x, rows, cols).unwrap();
                let nan = s.upload(&vec![f32::NAN; rows * cols], rows, cols).unwrap();
                let nan_b = s.upload(&vec![f32::NAN; rows], rows, 1).unwrap();
                for once in [true, false] {
                    let was = tt_kernels::tensor::set_col_once(once);
                    let primed = s.eltwise(op(kind), &nan, Some(&nan_b)).unwrap();
                    let got = s.eltwise(op(kind), &a, Some(&b)).unwrap();
                    tt_kernels::tensor::set_col_once(was);
                    let got_v = s.download(&got).unwrap();
                    for (i, (g, w)) in got_v.iter().zip(&want).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "{label}, once {once}: datum ({}, {}): {g} for {w}",
                            i / cols,
                            i % cols
                        );
                    }
                    s.free(primed).unwrap();
                    s.free(got).unwrap();
                }
                for t in [a, b, nan, nan_b] {
                    s.free(t).unwrap();
                }
            }
        }
    });
}
