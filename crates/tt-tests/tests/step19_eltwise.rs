//! Phase 9 gate: element-wise ops on tensors in GDDR, against `burn-flex`.
//!
//! The data mover computes them in L1 with the baby RISC-V's FP32 unit
//! (`tt_isa::dm::kind`). The claim is Flex's result bit for bit, which is what
//! keeps a training run's loss curve on the golden: `fadd.s`/`fsub.s`/`fmul.s`
//! round to nearest even, so they differ from IEEE only where an operand or
//! result is denormal, which the operands here avoid -- and one test records
//! that difference rather than pretending it away.

use burn::tensor::{activation, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_isa::dm::kind;
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::Eltwise;
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

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

/// Values that exercise every branch the kernels have, and no denormals.
fn edgy(seed: u64, n: usize) -> Vec<f32> {
    let specials = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        // Tiny but far enough from the denormals that no product or half of
        // one lands there.
        1e-10,
        -1e-10,
        f32::MAX,
        -f32::MAX,
        1.0,
        -1.0,
    ];
    let mut v = floats(seed, n);
    for (i, x) in v.iter_mut().enumerate() {
        if i % 7 == 0 {
            *x = specials[(i / 7) % specials.len()];
        }
    }
    v
}

fn flex(v: &[f32], r: usize, c: usize) -> Tensor<Flex, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), &FlexDevice)
}

fn bits(t: Tensor<Flex, 2>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

/// NaNs compare by class: the hardware's NaN payload is its own.
fn assert_same(got: &[f32], want: &[u32], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let w = f32::from_bits(*w);
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
            |_, _| Ok(()),
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

#[test]
fn every_kind_matches_flex_bit_for_bit() {
    with_session(|s| {
        // Ragged, more tiles than one mover list holds, and MNIST's own.
        for [r, c] in [[37, 70], [64, 128], [784, 128]] {
            let (av, bv) = (edgy(r as u64, r * c), edgy(c as u64 + 99, r * c));
            let row = edgy(5, c);
            let a = s.upload(&av, r, c).unwrap();
            let b = s.upload(&bv, r, c).unwrap();
            let bias = s.upload(&row, 1, c).unwrap();
            let (fa, fb) = (flex(&av, r, c), flex(&bv, r, c));
            let cases: Vec<(&str, u32, f32, bool, Vec<u32>)> = vec![
                ("add", kind::ADD, 0.0, false, bits(fa.clone() + fb.clone())),
                ("sub", kind::SUB, 0.0, false, bits(fa.clone() - fb.clone())),
                ("mul", kind::MUL, 0.0, false, bits(fa.clone() * fb.clone())),
                (
                    "mul 0.5",
                    kind::MUL_SCALAR,
                    0.5,
                    false,
                    bits(fa.clone() * 0.5),
                ),
                (
                    "mul 0.1",
                    kind::MUL_SCALAR,
                    0.1,
                    false,
                    bits(fa.clone() * 0.1),
                ),
                (
                    "relu",
                    kind::RELU,
                    0.0,
                    false,
                    bits(activation::relu(fa.clone())),
                ),
                (
                    "relu backward",
                    kind::RELU_BACKWARD,
                    0.0,
                    false,
                    // Flex's own closure (`burn-flex` `ops/activation.rs:32`).
                    av.iter()
                        .zip(&bv)
                        .map(|(&o, &g)| if o > 0.0 { g } else { 0.0 }.to_bits())
                        .collect(),
                ),
                (
                    "add row",
                    kind::ADD_ROW,
                    0.0,
                    true,
                    bits(fa.clone() + flex(&row, 1, c)),
                ),
            ];
            for (label, k, scalar, use_row, want) in cases {
                let op = Eltwise { kind: k, scalar };
                let other = if use_row { Some(&bias) } else { Some(&b) };
                let out = s
                    .eltwise(op, &a, other)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                let got = s.download(&out).unwrap();
                assert_same(&got, &want, &format!("[{r}, {c}] {label}"));
                s.free(out).unwrap();
            }
            for t in [a, b, bias] {
                s.free(t).unwrap();
            }
        }
    });
}

/// The one documented difference: a denormal result is flushed to zero, where
/// Flex keeps it (`BabyRISCV/InstructionSet.md:21`).
#[test]
fn a_denormal_result_is_flushed() {
    with_session(|s| {
        let tiny = f32::MIN_POSITIVE; // 2^-126, normal
        let a = s.upload(&[tiny], 1, 1).unwrap();
        let op = Eltwise {
            kind: kind::MUL_SCALAR,
            scalar: 0.5,
        };
        let out = s.eltwise(op, &a, None).unwrap();
        let got = s.download(&out).unwrap()[0];
        assert!(tiny * 0.5 != 0.0, "the host keeps the denormal");
        assert_eq!(got.to_bits(), 0, "the device flushes it to +0");
    });
}

/// The sum over rows, against Flex's `sum_dim(0)` bit for bit: the order of
/// the additions is the point (`tt_isa::dm::kind::COL_SUM`). A column taller
/// than one mover list (MNIST's 60 000 rows are 1875 tiles) is included.
#[test]
fn the_sum_over_rows_matches_flex_bit_for_bit() {
    with_session(|s| {
        for [r, c] in [[37, 70], [64, 128], [7000, 40]] {
            let v = edgy(r as u64 * 3, r * c);
            let want = bits(flex(&v, r, c).sum_dim(0));
            let a = s.upload(&v, r, c).unwrap();
            let out = s.sum_rows(&a).unwrap();
            assert_eq!((out.rows, out.cols), (1, c));
            assert_same(
                &s.download(&out).unwrap(),
                &want,
                &format!("[{r}, {c}] sum"),
            );
            s.free(out).unwrap();
            s.free(a).unwrap();
        }
    });
}

/// A row view reads exactly the rows it names, ragged last tile row included,
/// and works as a matmul operand.
#[test]
fn a_row_view_is_the_rows_it_names() {
    with_session(|s| {
        let [r, c] = [200, 70];
        let v = floats(77, r * c);
        let a = s.upload(&v, r, c).unwrap();
        for (first, n) in [(0, 64), (64, 64), (192, 8), (32, 168)] {
            let view = a.rows_view(first, n).unwrap();
            assert_eq!(
                s.download(&view).unwrap(),
                v[first * c..(first + n) * c],
                "rows {first}..{}",
                first + n
            );
            s.free(view).unwrap();
        }
        assert!(a.rows_view(16, 32).is_err(), "not a tile row");
        assert!(a.rows_view(0, 40).is_err(), "ends inside a tile row");
        // As a matmul operand: rows 64..128 against the host product.
        let w = floats(78, c * 32);
        let bw = s.upload(&w, c, 32).unwrap();
        let view = a.rows_view(64, 64).unwrap();
        let got = s
            .matmul_dram(
                &view,
                false,
                &bw,
                false,
                tt_kernels::matmul::SrcRoute::Tf32FromFp32,
                tt_kernels::matmul::Fidelity::HiFi4,
                tt_tests::harness::BUDGET,
            )
            .unwrap();
        let want = s
            .matmul(
                &v[64 * c..128 * c],
                &w,
                [64, c, 32],
                tt_kernels::matmul::SrcRoute::Tf32FromFp32,
                tt_kernels::matmul::Fidelity::HiFi4,
                tt_tests::harness::BUDGET,
            )
            .unwrap();
        let got = s.download(&got).unwrap();
        assert!(got
            .iter()
            .zip(&want)
            .all(|(g, w)| g.to_bits() == w.to_bits()));
        // Freeing the view gave nothing back; the parent is intact.
        s.free(view).unwrap();
        assert_eq!(s.download(&a).unwrap(), v);
    });
}
