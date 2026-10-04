//! Phase 10 gate (R2's groundwork): binary ops with a broadcast operand.
//!
//! `a (op) b` with `b` one row (`[1, cols]`, laid into `Dst` by sub-run
//! unpacks) or one column (`[rows, 1]`, made into a whole tile by the mover's
//! `READ_BROADCAST_COL`), for `ADD`, `SUB`, `MUL` -- `burn-flex` bit for bit --
//! and `DIV`, within the one ulp `step28_division` derives. The device is held
//! bit for bit to the program (`ops::reference`), and each output's padding
//! claim to its raw tiles: a broadcast lands in the padding along the
//! dimension it is broadcast in.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_kernels::kind;
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::{kind_sfpu, reference};
use tt_kernels::tensor::Eltwise;
use tt_kernels::tensor::Pad;
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

fn operands(seed: u64, n: usize) -> Vec<f32> {
    let specials = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.0,
        -1.0,
        3.0,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if i % 13 == 0 {
                return specials[(i / 13) % specials.len()];
            }
            // Normal operands whose reciprocals and quotients stay normal.
            let e = ((s >> 33) % 80) as u32 + 87;
            f32::from_bits(((s >> 63) as u32) << 31 | e << 23 | (s >> 40) as u32 & 0x7f_ffff)
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
fn row_and_column_broadcasts_match_flex() {
    with_session(|s| {
        for [r, c] in [[37, 70], [64, 96]] {
            let av = operands(r as u64, r * c);
            let (rowv, colv) = (operands(7, c), operands(9, r));
            let a = s.upload(&av, r, c).unwrap();
            let row = s.upload(&rowv, 1, c).unwrap();
            let col = s.upload(&colv, r, 1).unwrap();
            let fa = flex(&av, r, c);
            let (frow, fcol) = (flex(&rowv, 1, c), flex(&colv, r, 1));
            let cases = [
                (
                    "add row",
                    kind::ADD,
                    &row,
                    &rowv,
                    values(fa.clone() + frow.clone()),
                ),
                (
                    "sub row",
                    kind::SUB,
                    &row,
                    &rowv,
                    values(fa.clone() - frow.clone()),
                ),
                (
                    "mul row",
                    kind::MUL,
                    &row,
                    &rowv,
                    values(fa.clone() * frow.clone()),
                ),
                (
                    "div row",
                    kind_sfpu::DIV,
                    &row,
                    &rowv,
                    values(fa.clone() / frow.clone()),
                ),
                (
                    "add col",
                    kind::ADD,
                    &col,
                    &colv,
                    values(fa.clone() + fcol.clone()),
                ),
                (
                    "sub col",
                    kind::SUB,
                    &col,
                    &colv,
                    values(fa.clone() - fcol.clone()),
                ),
                (
                    "mul col",
                    kind::MUL,
                    &col,
                    &colv,
                    values(fa.clone() * fcol.clone()),
                ),
                (
                    "div col",
                    kind_sfpu::DIV,
                    &col,
                    &colv,
                    values(fa.clone() / fcol.clone()),
                ),
            ];
            for (name, k, b, bv, want) in cases {
                let out = s
                    .eltwise(
                        Eltwise {
                            scalar2: 0.0,
                            kind: k,
                            scalar: 0.0,
                        },
                        &a,
                        Some(b),
                    )
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                let got = s.download(&out).unwrap();
                let model = reference(k, 0.0, &av, Some(bv), r, c);
                for (i, (g, m)) in got.iter().zip(&model).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        m.to_bits(),
                        "[{r}, {c}] {name}: element {i}: device {g:e}, program {m:e}"
                    );
                }
                for (i, (m, w)) in model.iter().zip(&want).enumerate() {
                    if w.is_nan() {
                        assert!(m.is_nan(), "[{r}, {c}] {name}: element {i}");
                    } else if k == kind_sfpu::DIV
                        && w.is_finite()
                        && *w != 0.0
                        && w.abs() >= f32::MIN_POSITIVE
                    {
                        let d = (m.to_bits() as i64 - w.to_bits() as i64).unsigned_abs();
                        assert!(d <= 1, "[{r}, {c}] {name}: element {i}: {m:e} vs {w:e}");
                    } else if k == kind_sfpu::DIV && w.abs() < f32::MIN_POSITIVE {
                        assert_eq!(
                            m.abs(),
                            0.0,
                            "[{r}, {c}] {name}: element {i}: a flushed denormal"
                        );
                    } else {
                        assert_eq!(
                            m.to_bits(),
                            w.to_bits(),
                            "[{r}, {c}] {name}: element {i}: {m:e} vs {w:e}"
                        );
                    }
                }
                // What the output's padding claims, it holds.
                if out.pad() == Pad::Zero {
                    let [rt, ct] = out.grid();
                    let raw = s.download_padded(&out).unwrap();
                    for rr in 0..32 * rt {
                        for cc in 0..32 * ct {
                            if rr >= r || cc >= c {
                                assert!(
                                    raw[rr * 32 * ct + cc] == 0.0,
                                    "{name}: padding ({rr}, {cc})"
                                );
                            }
                        }
                    }
                }
                s.free(out).unwrap();
            }
            for t in [a, row, col] {
                s.free(t).unwrap();
            }
        }
    });
}
