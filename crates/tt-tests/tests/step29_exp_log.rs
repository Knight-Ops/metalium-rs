//! Phase 10 gate (S4): `exp` and `log` on the SFPU.
//!
//! Approximations, so held twice: the device **bit for bit** to its program
//! run by the interpreter (`tt_kernels::sfpu::ops::reference`), and the
//! program to `burn-flex` within the bound derived on it
//! (`ops::EXP_BOUND`, `ops::LOG_BOUND`, relative to the exact value) plus
//! Flex's own error, under one ulp (`2^-23` relative). Special values exactly;
//! denormal results flush to zero.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::{kind_sfpu, reference, EXP_BOUND, LOG_BOUND};
use tt_kernels::tensor::Eltwise;
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
            // Across the range `exp` is finite over, and positive values of
            // every magnitude for `log`.
            let u = (s >> 40) as f32 / (1u64 << 24) as f32;
            if i % 2 == 0 {
                u * 170.0 - 85.0
            } else {
                f32::from_bits(0x0080_0000 + ((s >> 34) as u32 % 0x7e80_0000))
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

/// `got` within `rel` of `want`, relative; a NaN by class; an infinity, a zero
/// and a flushed denormal exactly.
fn close(got: f32, want: f32, rel: f64, what: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: {got} vs NaN");
    } else if want.is_infinite() || want.abs() < f32::MIN_POSITIVE {
        let want = if want.is_infinite() { want } else { 0.0 };
        assert_eq!(got.abs(), want.abs(), "{what}: {got} vs {want}");
        assert_eq!(
            got.is_sign_negative() && got != 0.0,
            want.is_sign_negative() && want != 0.0,
            "{what}: sign"
        );
    } else {
        let d = (got as f64 - want as f64).abs() / (want as f64).abs();
        assert!(d <= rel, "{what}: {got:e} vs {want:e}, {d:e} relative");
    }
}

#[test]
fn exp_and_log_are_the_program_and_within_the_bound_of_flex() {
    with_session(|s| {
        for [r, c] in [[37, 70], [64, 128]] {
            let av = operands(r as u64 * 7, r * c);
            let a = s.upload(&av, r, c).unwrap();
            let fa = flex(&av, r, c);
            let flex_error = 1.0 / 8_388_608.0;
            let cases = [
                ("exp", kind_sfpu::EXP, values(fa.clone().exp()), EXP_BOUND),
                ("log", kind_sfpu::LOG, values(fa.clone().log()), LOG_BOUND),
            ];
            for (name, kind, want, bound) in cases {
                let out = s
                    .eltwise(
                        Eltwise {
                            scalar2: 0.0,
                            kind,
                            scalar: 0.0,
                        },
                        &a,
                        None,
                    )
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                let got = s.download(&out).unwrap();
                let model = reference(kind, 0.0, &av, None, r, c);
                for (i, (g, m)) in got.iter().zip(&model).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        m.to_bits(),
                        "[{r}, {c}] {name}: element {i}: device {g:e}, program {m:e}"
                    );
                }
                for (i, (m, w)) in model.iter().zip(&want).enumerate() {
                    close(
                        *m,
                        *w,
                        bound + flex_error,
                        &format!("[{r}, {c}] {name}({})", av[i]),
                    );
                }
                s.free(out).unwrap();
            }
            s.free(a).unwrap();
        }
    });
}
