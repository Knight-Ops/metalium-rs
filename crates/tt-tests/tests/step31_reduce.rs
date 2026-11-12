//! Phase 10 gate (R1): reductions over one dimension on the SFPU.
//!
//! The device is held **bit for bit** to the programs run by the interpreter
//! (`sfpu::reduce::reference`), on ragged shapes and lines of up to 32 tiles,
//! and those to `burn-flex`: a maximum is exactly Flex's value (equal as
//! floats: Flex returns the first of equal values, the SFPU's total order
//! `+0` over `-0`); a sum over columns, computed in a tree order rather than
//! Flex's left to right, within the bound any two orders of `n` additions
//! share, `2 (n - 1) u sum |x|` with `u = 2^-24` (each order is within `(n -
//! 1) u sum |x|` of the exact sum).

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::reduce::{reference, Axis, ReduceOp};
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

fn operands(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            match i % 23 {
                0 => 0.0,
                1 => -0.0,
                2 => -1.0e30,
                _ => {
                    ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
                        * 2.0f32.powi((s % 16) as i32 - 4)
                }
            }
        })
        .collect()
}

fn flex(v: &[f32], r: usize, c: usize) -> Tensor<Flex, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), &FlexDevice)
}

fn values(t: Tensor<Flex, 2>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
            |_, _| Ok(None),
        )
        .unwrap();
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
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[test]
fn reductions_are_the_programs_and_match_flex() {
    with_session(|s| {
        for [r, c] in [[37, 70], [64, 96], [100, 33], [5, 300], [1000, 40]] {
            let av = operands((r * c) as u64, r * c);
            let a = s.upload(&av, r, c).unwrap();
            let fa = flex(&av, r, c);
            let cases = [
                (
                    "max over columns",
                    ReduceOp::Max,
                    Axis::Cols,
                    values(fa.clone().max_dim(1)),
                ),
                (
                    "sum over columns",
                    ReduceOp::Sum,
                    Axis::Cols,
                    values(fa.clone().sum_dim(1)),
                ),
                (
                    "max over rows",
                    ReduceOp::Max,
                    Axis::Rows,
                    values(fa.clone().max_dim(0)),
                ),
            ];
            for (name, op, axis, want) in cases {
                let out = s
                    .reduce(&a, op, axis)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                let got = s.download(&out).unwrap();
                let model = reference(op, axis, &av, r, c);
                assert_eq!(got.len(), model.len());
                for (i, (g, m)) in got.iter().zip(&model).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        m.to_bits(),
                        "[{r}, {c}] {name}: {i}: device {g:e}, program {m:e}"
                    );
                }
                for (i, (m, w)) in model.iter().zip(&want).enumerate() {
                    if op == ReduceOp::Max {
                        assert!(m == w, "[{r}, {c}] {name}: {i}: {m:e} vs {w:e}");
                    } else {
                        let line: f64 = (0..c).map(|j| av[i * c + j].abs() as f64).sum();
                        let bound = 2.0 * (c as f64 - 1.0) * f64::powi(2.0, -24) * line;
                        assert!(
                            (*m as f64 - *w as f64).abs() <= bound,
                            "[{r}, {c}] {name}: {i}: {m:e} vs {w:e}, bound {bound:e}"
                        );
                    }
                }
                s.free(out).unwrap();
            }
            s.free(a).unwrap();
        }
    });
}

/// A line longer than one run holds (`per` > 219 input tiles with the whole `DATA` arena)
/// is reduced in continuations (`tensor::reduce_chunked`): 16-tile chunks, the unfolded
/// accumulator carried through GDDR, folded once at the end. It must be the one run's
/// programs bit for bit, so a long softmax row's maximum and sum are the short row's
/// arithmetic. Specials ride along (NaNs, infinities, signed zeros, a ragged last tile and
/// a whole one). Negative control: the sum folded a chunk at a time and the folds added
/// differs from the device, so the gate sees the carried accumulator.
#[test]
fn chunked_reductions_are_the_programs() {
    let specials = |v: &mut [f32]| {
        for (i, x) in v.iter_mut().enumerate() {
            match i % 4099 {
                7 => *x = f32::NAN,
                11 => *x = f32::INFINITY,
                13 => *x = f32::NEG_INFINITY,
                17 => *x = f32::from_bits(1),
                _ => {}
            }
        }
    };
    with_session(|s| {
        for (r, c, axis) in [
            (40, 32 * 230 + 5, Axis::Cols),
            (33, 32 * 240, Axis::Cols),
            (32 * 230 + 5, 40, Axis::Rows),
        ] {
            let mut av = operands((r * c) as u64 ^ 0x5eed, r * c);
            specials(&mut av);
            let a = s.upload(&av, r, c).unwrap();
            for op in [ReduceOp::Max, ReduceOp::Sum] {
                let out = s
                    .reduce(&a, op, axis)
                    .unwrap_or_else(|e| panic!("[{r}, {c}] {op:?} {axis:?}: {e}"));
                let got = s.download(&out).unwrap();
                let model = reference(op, axis, &av, r, c);
                assert_eq!(got.len(), model.len());
                for (i, (g, m)) in got.iter().zip(&model).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        m.to_bits(),
                        "[{r}, {c}] {op:?} {axis:?}: {i}: device {g:e}, program {m:e}"
                    );
                }
                s.free(out).unwrap();
            }
            s.free(a).unwrap();
        }

        // The control: plain data (no specials to swamp the low bits), the sum over
        // columns of 231 tiles.
        let (r, c) = (32, 32 * 230 + 5);
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let av: Vec<f32> = (0..r * c)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        let a = s.upload(&av, r, c).unwrap();
        let out = s.reduce(&a, ReduceOp::Sum, Axis::Cols).unwrap();
        let got = s.download(&out).unwrap();
        let model = reference(ReduceOp::Sum, Axis::Cols, &av, r, c);
        let chunk = 16 * 32;
        let mut folded = vec![0.0f32; r];
        for first in (0..c).step_by(chunk) {
            let w = chunk.min(c - first);
            let part: Vec<f32> = (0..r)
                .flat_map(|i| av[i * c + first..i * c + first + w].iter().copied())
                .collect();
            for (f, p) in folded
                .iter_mut()
                .zip(reference(ReduceOp::Sum, Axis::Cols, &part, r, w))
            {
                *f += p;
            }
        }
        for (i, (g, m)) in got.iter().zip(&model).enumerate() {
            assert_eq!(g.to_bits(), m.to_bits(), "control {i}: {g:e} vs {m:e}");
        }
        let differ = got
            .iter()
            .zip(&folded)
            .filter(|(g, f)| g.to_bits() != f.to_bits())
            .count();
        assert!(
            differ > 0,
            "the sum folded a chunk at a time matched the device on every row: the gate cannot see the carried accumulator"
        );
        s.free(out).unwrap();
        s.free(a).unwrap();
    });
}
