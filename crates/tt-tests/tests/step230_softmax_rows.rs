//! Attention's scale, mask and softmax over rows of any length as one op
//! (`Session::softmax_rows`, `tt_kernels::softmax`; roadmap Track 1, "Then the long-context
//! softmax"): each tile row streamed through L1 in three passes, its statistics kept in L1.
//!
//! The claims:
//! - **The model's bits.** The device equals `sfpu::softmax::reference` (the same programs run by
//!   the specification model, chunk by chunk), which the unit gate
//!   `the_passes_are_the_composition_bit_for_bit` holds to the composition's ops' own models:
//!   rows of one tile, of one chunk, of several chunks with a ragged last one (one tile), and a
//!   mask repeated down the rows (a GQA group's or a chunk's causal block).
//! - **The composition's bits on the card:** the same scores through the six ops burn-tt's
//!   attention runs (`SCALE_MASK_FILL`, the maximum and its clamp, `SUB_EXP`, the sum and its
//!   clamp, `DIV`) download the same bits.
//! - **Specials:** NaNs of both signs, infinities, `f32::MAX`, subnormals, scores of +-300 and rows
//!   wholly masked (their output is `0 / MIN_POSITIVE`-clamped exactly as the composition's).
//! - **Refusals** name the shape.
//! - **Both modes** (`softmax::Mode`): `Written` (the default: `SUB_EXP` computed once, written over
//!   `s`, then a staged `DIV` by the sum; `sfpu::softmax::reference_written`) and `Recompute` (as
//!   first built), each against its model and both against the composition, including rows longer
//!   than one reduction run (`per` > 219 tiles, where the composition's maximum and sum stream
//!   through GDDR in continuations and take their clamp as an op of its own): the rows burn-tt
//!   routes here by default (`docs/plans/long-context.md`, "Long-prompt softmax passes").
//!
//! Watched failing: the pass-1 accumulate reading the wrong column half (unit gate), and on the
//! card with the accumulator carried between chunks dropped (the second chunk starting fresh);
//! in `Written` mode, the sum accumulated from `s` instead of the `e` it writes (ttsim: the
//! composition's bits differ; the model, running the same programs, cannot see it).

use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::kind_sfpu;
use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
use tt_kernels::softmax::Mode;
use tt_kernels::tensor::{Elem, Eltwise};
use tt_ttsim::fork_scope;

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
            TileChoice::All,
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Scores with every special a softmax meets among ordinary ones.
fn scores(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let specials = [
        0.0,
        -0.0,
        f32::MIN_POSITIVE / 4.0,
        -f32::MIN_POSITIVE / 8.0,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xffc0_1234),
        300.0,
        -300.0,
    ];
    (0..rows * cols)
        .map(|k| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if k % 211 == 17 {
                specials[(s >> 33) as usize % specials.len()]
            } else {
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * 12.0
            }
        })
        .collect()
}

/// A `[rows, cols]` block of `true` where not attended: row 3 wholly, the others past a point.
fn mask(rows: usize, cols: usize) -> Vec<u32> {
    (0..rows * cols)
        .map(|k| {
            let (r, c) = (k / cols, k % cols);
            u32::from(r == 3 || c > (r * 37 + 5) % cols)
        })
        .collect()
}

fn mismatches(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// Rows, columns and the mask's rows: one tile; one chunk; three chunks of which the last is
/// one tile; a mask repeated twice down the rows.
const SHAPES: [(usize, usize, usize); 4] =
    [(32, 32, 32), (32, 64, 32), (32, 2080, 32), (64, 1152, 32)];

#[test]
fn softmax_rows_are_the_model_bit_for_bit() {
    with_session(|s| {
        for (rows, cols, mrows) in SHAPES {
            let x = scores(rows, cols, (rows * 7 + cols) as u64);
            let m = mask(mrows, cols);
            let full: Vec<u32> = (0..rows * cols).map(|k| m[k % (mrows * cols)]).collect();
            let xt = s.upload(&x, rows, cols).unwrap();
            let mt = s.upload_bits(&m, mrows, cols, Elem::Bool).unwrap();
            for (scale, mode) in [
                (0.125f32, Mode::Written),
                (-0.5, Mode::Written),
                (0.125, Mode::Recompute),
            ] {
                s.set_softmax_mode(mode);
                let got = s.softmax_rows(&xt, &mt, scale).unwrap();
                let got_v = s.download(&got).unwrap();
                let model = match mode {
                    Mode::Written => tt_kernels::sfpu::softmax::reference_written,
                    Mode::Recompute => tt_kernels::sfpu::softmax::reference,
                };
                let (_, _, _, want) = model(&x, &full, rows, cols, scale, 32);
                assert_eq!(
                    mismatches(&got_v, &want),
                    0,
                    "[{rows}, {cols}], mask rows {mrows}, scale {scale}, {mode:?}"
                );
                // A wholly masked row is the composition's `0 / MIN_POSITIVE`: zeros.
                assert!(got_v[3 * cols..4 * cols].iter().all(|v| v.to_bits() == 0));
                s.free(got).unwrap();
            }
            s.free(xt).unwrap();
            s.free(mt).unwrap();
        }
    });
}

fn op(kind: u32, scalar: f32, scalar2: f32) -> Eltwise {
    Eltwise {
        kind,
        scalar,
        scalar2,
    }
}

/// The six ops burn-tt's attention composes, as it runs them: a clamped reduction in one pass
/// where the row fits one run, else the reduction (in continuations) and the clamp as an op.
fn composition<T: tt_device::Transport>(
    s: &mut Session<T>,
    xt: &tt_kernels::tensor::DramTensor,
    mt: &tt_kernels::tensor::DramTensor,
    scale: f32,
) -> Vec<f32> {
    let [rows, cols] = [xt.rows, xt.cols];
    let sm = s
        .eltwise(
            op(kind_sfpu::SCALE_MASK_FILL, scale, f32::NEG_INFINITY),
            xt,
            Some(mt),
        )
        .unwrap();
    let stat = |s: &mut Session<T>, a: &tt_kernels::tensor::DramTensor, r: ReduceOp, min: f32| {
        let then = Some((kind_sfpu::CLAMP_MIN, min.to_bits()));
        if tt_kernels::tensor::reduce_then_fits(r, Axis::Cols, [rows, cols], then) {
            s.reduce_then(a, r, Axis::Cols, then).unwrap()
        } else {
            let raw = s.reduce(a, r, Axis::Cols).unwrap();
            let c = s
                .eltwise(op(kind_sfpu::CLAMP_MIN, min, 0.0), &raw, None)
                .unwrap();
            s.free(raw).unwrap();
            c
        }
    };
    let max = stat(s, &sm, ReduceOp::Max, -f32::MAX);
    let e = s
        .eltwise(op(kind_sfpu::SUB_EXP, 0.0, 0.0), &sm, Some(&max))
        .unwrap();
    let sum = stat(s, &e, ReduceOp::Sum, f32::MIN_POSITIVE);
    let p = s
        .eltwise(op(kind_sfpu::DIV, 0.0, 0.0), &e, Some(&sum))
        .unwrap();
    let out = s.download(&p).unwrap();
    for t in [sm, max, e, sum, p] {
        s.free(t).unwrap();
    }
    out
}

#[test]
fn softmax_rows_are_the_composition_on_the_card() {
    with_session(|s| {
        // A row of one run, and one of 230 tiles past it with a mask repeated down the rows.
        for (rows, cols, mrows) in [(64, 1152, 64), (64, 32 * 230, 32)] {
            assert_eq!(
                tt_kernels::tensor::reduce_then_fits(
                    ReduceOp::Max,
                    Axis::Cols,
                    [rows, cols],
                    Some((kind_sfpu::CLAMP_MIN, (-f32::MAX).to_bits()))
                ),
                cols == 1152,
                "[{rows}, {cols}]: the composition's reductions in one pass or in continuations"
            );
            let x = scores(rows, cols, 5 + cols as u64);
            let m = mask(mrows, cols);
            let xt = s.upload(&x, rows, cols).unwrap();
            let mt = s.upload_bits(&m, mrows, cols, Elem::Bool).unwrap();
            let scale = 0.125;
            let c = composition(s, &xt, &mt, scale);
            for mode in [Mode::Written, Mode::Recompute] {
                s.set_softmax_mode(mode);
                let fused = s.softmax_rows(&xt, &mt, scale).unwrap();
                let f = s.download(&fused).unwrap();
                assert_eq!(
                    mismatches(&f, &c),
                    0,
                    "[{rows}, {cols}], mask rows {mrows}, {mode:?}"
                );
                s.free(fused).unwrap();
            }
            for t in [xt, mt] {
                s.free(t).unwrap();
            }
        }
    });
}

#[test]
fn shapes_it_does_not_take_are_refused_by_name() {
    with_session(|s| {
        let x = s.upload(&vec![0.0; 32 * 40], 32, 40).unwrap();
        let m = s
            .upload_bits(&vec![0; 32 * 40], 32, 40, Elem::Bool)
            .unwrap();
        let e = s.softmax_rows(&x, &m, 1.0).unwrap_err().to_string();
        assert!(e.contains("softmax rows"), "{e}");
        let x2 = s.upload(&vec![0.0; 32 * 64], 32, 64).unwrap();
        let f = s.upload(&vec![0.0; 32 * 64], 32, 64).unwrap();
        let e = s.softmax_rows(&x2, &f, 1.0).unwrap_err().to_string();
        assert!(e.contains("Bool mask"), "{e}");
        for t in [x, m, x2, f] {
            s.free(t).unwrap();
        }
    });
}
