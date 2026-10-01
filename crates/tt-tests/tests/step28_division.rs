//! Phase 10 gate (S3): reciprocal and division on the SFPU.
//!
//! An approximation, so the Tolerance policy applies twice
//! (`hardware-coverage.md`, Definition of done): the device is held **bit for
//! bit** to its own program run by the interpreter
//! (`tt_kernels::sfpu::ops::reference`), and that is held to `burn-flex`'s
//! correctly rounded `1/x` and `a/b` within **one ulp** -- the bound derived on
//! `Program::recip` and `ops::divide` from `SFPARECIP`'s documented seed
//! accuracy and two Newton steps. Special values match IEEE exactly; a
//! denormal operand or result flushes to a zero of the right sign, as all of
//! the SFPU's arithmetic does (numerics row D), and is the one documented
//! difference.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::{kind_sfpu, reference};
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

/// `got` within `ulps` of `want` (a zero of either sign for a flushed
/// denormal), NaN by class, infinities and zeros exactly.
fn close(got: f32, want: f32, ulps: u32, what: &str) {
    let want = if want != 0.0 && want.abs() < f32::MIN_POSITIVE {
        0.0f32.copysign(want)
    } else {
        want
    };
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: {got} vs NaN");
    } else if want.is_infinite() || want == 0.0 {
        assert_eq!(got.to_bits(), want.to_bits(), "{what}: {got} vs {want}");
    } else {
        let d = (got.to_bits() as i64 - want.to_bits() as i64).unsigned_abs();
        assert!(d <= ulps as u64, "{what}: {got:e} vs {want:e}, {d} ulps");
    }
}

#[test]
fn reciprocal_and_division_are_the_program_and_within_one_ulp_of_flex() {
    with_session(|s| {
        for [r, c] in [[37, 70], [96, 128]] {
            let (av, bv) = (operands(r as u64, r * c), operands(c as u64 + 5, r * c));
            let a = s.upload(&av, r, c).unwrap();
            let b = s.upload(&bv, r, c).unwrap();
            let (fa, fb) = (flex(&av, r, c), flex(&bv, r, c));
            let cases = [
                (
                    "recip",
                    kind_sfpu::RECIP,
                    0.0,
                    None,
                    values(fa.clone().recip()),
                ),
                (
                    "div",
                    kind_sfpu::DIV,
                    0.0,
                    Some(&b),
                    values(fa.clone() / fb.clone()),
                ),
                (
                    "div 3.7",
                    kind_sfpu::DIV_SCALAR,
                    3.7,
                    None,
                    values(fa.clone() / 3.7),
                ),
            ];
            for (name, kind, scalar, other, want) in cases {
                let out = s
                    .eltwise(
                        Eltwise {
                            scalar2: 0.0,
                            kind,
                            scalar,
                        },
                        &a,
                        other,
                    )
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                let got = s.download(&out).unwrap();
                let model = reference(kind, scalar, &av, other.map(|_| bv.as_slice()), r, c);
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
                        1,
                        &format!("[{r}, {c}] {name}: element {i} (a {}, b {})", av[i], bv[i]),
                    );
                }
                s.free(out).unwrap();
            }
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
    });
}
