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
