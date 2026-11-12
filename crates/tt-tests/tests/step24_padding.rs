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
use tt_kernels::kind;
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
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        // The refill rules of whole-tile reads: a thin read of a ragged `A` leaves its output's
        // padding undefined by design, and `step190` gates what follows one.
        s.set_thin_matmul_reads(false);
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
        // The refill rules of whole-tile reads: a thin read of a ragged `A` leaves its output's
        // padding undefined by design, and `step190` gates what follows one.
        s.set_thin_matmul_reads(false);
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

/// Padding off `K` reaches only the output's padding: an output row reads its row of `op(A)`
/// alone, an output column its column of `op(B)`. So an operand whose `K` is whole tiles is used
/// as it is, however dirty its other padding (`tensor::matmul_padding_off_k`): no refill is
/// queued, the real result is Flex's, and the output's padding is declared undefined -- which a
/// consumer that needs it zero (a matmul over it as `K`) then refills. Dirt along `K` is still
/// refilled (`dirty_padding_never_reaches_an_accumulation`).
#[test]
fn padding_off_k_is_used_as_it_is() {
    with_session(|s| {
        // `M` = 37 and `N` = 24 ragged, `K` = 64 whole tiles. Values in -1..=1, so a product's
        // entries (at most 256 in magnitude) are exact TF32 operands for the matmul after it.
        let (m, k, n) = (37, 64, 24);
        let small = |seed: usize, n: usize| -> Vec<f32> {
            (0..n).map(|i| ((i * 7 + seed) % 3) as f32 - 1.0).collect()
        };
        let (xv, bv, wv, vv, ev) = (
            small(1, m * k),
            small(2, k),
            small(3, k * n),
            small(4, n * k),
            small(5, k),
        );
        let x = s.upload(&xv, m, k).unwrap();
        let b = s.upload(&bv, 1, k).unwrap();
        let w = s.upload(&wv, k, n).unwrap();
        let v = s.upload(&vv, n, k).unwrap();
        let e = s.upload(&ev, 1, k).unwrap();
        let yv = small(6, m * n);
        let y = s.upload(&yv, m, n).unwrap();
        let fy = flex(&yv, m, n);
        let add_row = Eltwise {
            scalar2: 0.0,
            kind: kind::ADD_ROW,
            scalar: 0.0,
        };
        // The bias lands in the padding rows: `x + b` is dirty past `M`, and `v + e`, used
        // transposed as `B`, past `N`.
        let xb = s.eltwise(add_row, &x, Some(&b)).unwrap();
        let ve = s.eltwise(add_row, &v, Some(&e)).unwrap();
        for (t, what) in [(&xb, "x + b"), (&ve, "v + e")] {
            assert_eq!(t.pad(), Pad::Undefined, "{what}: dirty padding");
            assert!(
                padding(s, t).iter().any(|&(_, _, p)| p != 0.0),
                "{what}: the bias is in the padding rows"
            );
        }
        let (fx, fb, fw) = (flex(&xv, m, k), flex(&bv, 1, k), flex(&wv, k, n));
        let fxb = fx.clone() + fb;
        let fve = flex(&vv, n, k) + flex(&ev, 1, k);
        // Ops a product queues: a clean pair's is the matmul's own.
        let queued = |s: &mut Session<_>, a: &DramTensor, ta, bb: &DramTensor, tb| {
            let before = s.host_times().ops;
            let p = mm(s, a, ta, bb, tb);
            (p, s.host_times().ops - before)
        };
        let (clean, alone) = queued(s, &x, false, &w, false);
        assert_same(
            &s.download(&clean).unwrap(),
            &values(fx.clone().matmul(fw.clone())),
            "x @ w",
        );
        assert_eq!(clean.pad(), Pad::Zero, "clean operands, clean output");
        s.free(clean).unwrap();
        let cases = [
            (
                &xb,
                &w,
                false,
                fxb.clone().matmul(fw.clone()),
                "(x + b) @ w",
            ),
            (
                &x,
                &ve,
                true,
                fx.clone().matmul(fve.clone().transpose()),
                "x @ (v + e)^T",
            ),
            (
                &xb,
                &ve,
                true,
                fxb.clone().matmul(fve.clone().transpose()),
                "(x + b) @ (v + e)^T",
            ),
        ];
        for (a, bb, tb, want, what) in cases {
            let (p, ops) = queued(s, a, false, bb, tb);
            assert_eq!(ops, alone, "{what}: no refill queued");
            assert_same(&s.download(&p).unwrap(), &values(want.clone()), what);
            assert_eq!(
                p.pad(),
                Pad::Undefined,
                "{what}: the operand's padding is the output's"
            );
            // A consumer that accumulates over the output's padding rows (`K` = `M`, ragged)
            // refills them.
            let pt = mm(s, &p, true, &y, false);
            assert_same(
                &s.download(&pt).unwrap(),
                &values(want.transpose().matmul(fy.clone())),
                &format!("({what})^T @ y"),
            );
            pad_is_honest(s, &pt, what);
            for t in [p, pt] {
                s.free(t).unwrap();
            }
        }
        // Used as they were: neither operand was refilled.
        assert_eq!(xb.pad(), Pad::Undefined);
        assert_eq!(ve.pad(), Pad::Undefined);
        for t in [x, b, w, v, e, y, xb, ve] {
            s.free(t).unwrap();
        }
    });
}
