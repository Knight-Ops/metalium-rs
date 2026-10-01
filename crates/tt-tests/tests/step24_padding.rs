//! Phase 10 gate (F0): a ragged tensor's padding is a property of the tensor.
//!
//! The hardware computes whole tiles, so what a ragged tensor's edge tiles hold
//! past its last row or column reaches anything that accumulates over them.
//! Each op says what it needs of its inputs' padding and what it leaves
//! (`tt_kernels::tensor::OpPadding`), and the session refills edge tiles when
//! the two differ. The claims, against `burn-flex` bit for bit:
//!
//! - every producer that dirties padding (`ADD_ROW` writes the bias into the
//!   padding rows, `MUL_SCALAR` by infinity turns zero padding into NaN) chained
//!   into every consumer that accumulates over it (the column sum, a matmul
//!   whose `K` is ragged, either operand, either orientation) gives Flex's
//!   result;
//! - what each output's `pad()` claims is what its raw tiles hold;
//! - a view, which shares its parent's slots, is filled through a copy: the
//!   parent's padding is left as it was.
//!
//! The values are small integers, so every sum and product is exact in FP32
//! (and in TF32 for the matmul's operands), and the summation order cannot
//! matter: Flex is the answer, not an approximation of it.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_isa::dm::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise, Pad};
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

fn ints(seed: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + seed) % 9) as f32 - 4.0).collect()
}

fn flex(v: &[f32], r: usize, c: usize) -> Tensor<Flex, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), &FlexDevice)
}

fn values(t: Tensor<Flex, 2>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

/// NaNs compare by class: the hardware's NaN payload is its own.
fn assert_same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "{what}: element {i}: {g} vs NaN");
        } else {
            assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
        }
    }
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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// What `t`'s raw tiles hold past its last row or column, as `(row, col,
/// value)`.
fn padding<T: tt_device::Transport>(
    s: &mut Session<T>,
    t: &DramTensor,
) -> Vec<(usize, usize, f32)> {
    let [rt, ct] = t.grid();
    let raw = s.download_padded(t).unwrap();
    let mut out = Vec::new();
    for r in 0..32 * rt {
        for c in 0..32 * ct {
            if r >= t.rows || c >= t.cols {
                out.push((r, c, raw[r * 32 * ct + c]));
            }
        }
    }
    out
}

/// `t`'s `pad()` is what its tiles hold: a claim of zero is checked datum by
/// datum (either sign).
fn pad_is_honest<T: tt_device::Transport>(s: &mut Session<T>, t: &DramTensor, what: &str) {
    if t.pad() == Pad::Zero {
        for (r, c, v) in padding(s, t) {
            assert!(
                v == 0.0,
                "{what}: claims zero padding, holds {v} at ({r}, {c})"
            );
        }
    }
}

fn mm<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    ta: bool,
    b: &DramTensor,
    tb: bool,
) -> DramTensor {
    s.matmul_dram(
        a,
        ta,
        b,
        tb,
        SrcRoute::Tf32FromFp32,
        Fidelity::HiFi4,
        tt_tests::harness::BUDGET,
    )
    .unwrap()
}

/// Every dirtying producer into every accumulating consumer.
#[test]
fn dirty_padding_never_reaches_an_accumulation() {
    with_session(|s| {
        for (r, c) in [(37, 70), (50, 40)] {
            let xv = ints(1, r * c);
            let bv = ints(2, c);
            let yv = ints(3, r * 24);
            let wv = ints(4, c * 24);
            let x = s.upload(&xv, r, c).unwrap();
            let b = s.upload(&bv, 1, c).unwrap();
            let y = s.upload(&yv, r, 24).unwrap();
            let w = s.upload(&wv, c, 24).unwrap();
            let dv = ints(8, 24);
            let d = s.upload(&dv, 1, 24).unwrap();
            let (fx, fb) = (flex(&xv, r, c), flex(&bv, 1, c));
            let (fy, fw) = (flex(&yv, r, 24), flex(&wv, c, 24));
            for x in [&x, &b, &y, &w, &d] {
                assert_eq!(x.pad(), Pad::Zero, "upload pads with zeros");
            }

            let add_row = Eltwise {
                scalar2: 0.0,
                kind: kind::ADD_ROW,
                scalar: 0.0,
            };
            let inf = Eltwise {
                scalar2: 0.0,
                kind: kind::MUL_SCALAR,
                scalar: f32::INFINITY,
            };
            let relu = Eltwise {
                scalar2: 0.0,
                kind: kind::RELU,
                scalar: 0.0,
            };
            let xb = s.eltwise(add_row, &x, Some(&b)).unwrap();
            let fxb = fx.clone() + fb.clone();
            let xi = s.eltwise(inf, &x, None).unwrap();
            let fxi = fx.clone() * f32::INFINITY;
            let rxb = s.eltwise(relu, &xb, None).unwrap();
            let frxb = burn::tensor::activation::relu(fxb.clone());
            // Dirty on the other side of `K` too: a zero there would hide
            // dirt on this side (`bias * 0`), and a gate must not.
            let yd = s.eltwise(add_row, &y, Some(&d)).unwrap();
            let fyd = fy.clone() + flex(&dv, 1, 24);
            for (t, what) in [(&xb, "x + b"), (&xi, "x * inf"), (&rxb, "relu(x + b)")] {
                assert_eq!(t.pad(), Pad::Undefined, "{what} [{r}, {c}]: dirty padding");
                pad_is_honest(s, t, what);
            }

            // The column sum, over each producer: it reads only valid rows.
            for (t, ft, what) in [
                (&xb, &fxb, "x + b"),
                (&xi, &fxi, "x * inf"),
                (&rxb, &frxb, "relu(x + b)"),
            ] {
                let sum = s.sum_rows(t).unwrap();
                assert_same(
                    &s.download(&sum).unwrap(),
                    &values(ft.clone().sum_dim(0)),
                    &format!("[{r}, {c}]: ({what}).sum_dim(0)"),
                );
                pad_is_honest(s, &sum, "the sum");
                s.free(sum).unwrap();
            }

            // A ragged K, in each operand and orientation: the matmul refills
            // its operands' padding first.
            let cases = [
                // `K` = r, the rows: a weight gradient's shape.
                (
                    &xb,
                    true,
                    &yd,
                    false,
                    fxb.clone().transpose().matmul(fyd.clone()),
                    "(x + b)^T @ (y + d)",
                ),
                (
                    &yd,
                    true,
                    &rxb,
                    false,
                    fyd.clone().transpose().matmul(frxb.clone()),
                    "(y + d)^T @ relu(x + b)",
                ),
                (
                    &xb,
                    true,
                    &y,
                    false,
                    fxb.clone().transpose().matmul(fy.clone()),
                    "(x + b)^T @ y",
                ),
                // `K` = c, the columns: a forward layer's shape.
                (
                    &xb,
                    false,
                    &w,
                    false,
                    fxb.clone().matmul(fw.clone()),
                    "(x + b) @ w",
                ),
                (
                    &rxb,
                    false,
                    &w,
                    false,
                    frxb.clone().matmul(fw.clone()),
                    "relu(x + b) @ w",
                ),
            ];
            for (a, ta, bb, tb, want, what) in cases {
                let p = mm(s, a, ta, bb, tb);
                assert_same(
                    &s.download(&p).unwrap(),
                    &values(want),
                    &format!("[{r}, {c}]: {what}"),
                );
                assert_eq!(p.pad(), Pad::Zero);
                pad_is_honest(s, &p, what);
                s.free(p).unwrap();
            }
            // The fills were in place, on tensors that own their slots.
            assert_eq!(xb.pad(), Pad::Zero, "refilled by the first matmul over it");
            pad_is_honest(s, &xb, "x + b after a matmul");
            pad_is_honest(s, &rxb, "relu(x + b) after a matmul");
            for t in [x, b, y, w, d, xb, xi, rxb, yd] {
                s.free(t).unwrap();
            }
        }
    });
}

/// A view shares its parent's slots, so it is refilled through a copy, and
/// the parent's padding -- which other views may rely on -- is left alone.
#[test]
fn a_view_is_filled_through_a_copy() {
    with_session(|s| {
        let (r, c) = (50, 40);
        let (xv, bv, yv, dv) = (ints(5, r * c), ints(6, c), ints(7, 18 * 24), ints(8, 24));
        let x = s.upload(&xv, r, c).unwrap();
        let b = s.upload(&bv, 1, c).unwrap();
        let y0 = s.upload(&yv, 18, 24).unwrap();
        let d = s.upload(&dv, 1, 24).unwrap();
        let add_row = Eltwise {
            scalar2: 0.0,
            kind: kind::ADD_ROW,
            scalar: 0.0,
        };
        let xb = s.eltwise(add_row, &x, Some(&b)).unwrap();
        // Dirty padding rows on the other side of `K` as well.
        let y = s.eltwise(add_row, &y0, Some(&d)).unwrap();
        let before = padding(s, &xb);
        assert!(
            before.iter().any(|&(_, _, v)| v != 0.0),
            "the bias is in the padding rows"
        );
        // Rows 32..50: the last tile row, ragged, ending where the parent does.
        let v = xb.rows_view(32, 18).unwrap();
        assert_eq!(v.pad(), Pad::Undefined);
        let p = mm(s, &v, true, &y, false);
        let fxb = flex(&xv, r, c) + flex(&bv, 1, c);
        let fy = flex(&yv, 18, 24) + flex(&dv, 1, 24);
        let want = fxb.slice([32..50, 0..40]).transpose().matmul(fy);
        assert_same(&s.download(&p).unwrap(), &values(want), "view^T @ (y + d)");
        assert_eq!(xb.pad(), Pad::Undefined, "the parent's claim is unchanged");
        assert_eq!(padding(s, &xb), before, "and so are its tiles");
        for t in [x, b, y0, d, y, p, xb] {
            s.free(t).unwrap();
        }
    });
}
