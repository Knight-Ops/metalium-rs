//! Phase 10 gate (S4, milestone 10.2f): trigonometry on the SFPU.
//!
//! As `step45_exp_family`: the device **bit for bit** to its programs
//! (`tt_kernels::sfpu::ops::reference_op`), and the programs to `burn-flex`
//! within the bounds derived on them (`SIN_BOUND`, `COS_BOUND`, `TAN_BOUND`,
//! `ATAN_BOUND`, `ATAN2_BOUND`, `ASIN_BOUND`, `ACOS_BOUND`) plus Flex's
//! own ulp -- for every finite input, the largest and the floats nearest a
//! multiple of `pi/2` included (`trig_reduce`'s exact Payne-Hanek reduction).
//! A zero-padding claim is held to the raw tiles. `atan2` reads a denormal
//! operand as a zero of its sign (numerics row D), so those lanes are held to
//! `f32::atan2` of the flushed pair.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::{
    kind_sfpu::*, reference_op, Broadcast, ACOS_BOUND, ASIN_BOUND, ATAN2_BOUND, ATAN_BOUND,
    COS_BOUND, SIN_BOUND, TAN_BOUND,
};
use tt_kernels::tensor::{DramTensor, Eltwise, Pad};
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

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

/// The floats nearest a multiple of `pi/2` (every float scanned:
/// `ops::transcendental::no_float_reduces_closer_than_the_hardest`).
const HARDEST: [u32; 6] = [
    0x6f79_be45,
    0x50a3_e87f,
    0x6ff9_be45,
    0x5123_e87f,
    0x437c_e5f1,
    0x7079_be45,
];

/// Values over a few periods, near zero, in every binade, the hardest
/// reductions and the specials (NaNs of both signs, both zeros, infinities,
/// denormals, the switches at `2^-12` and `pi/4`).
fn values(seed: u64, n: usize) -> Vec<f32> {
    let specials = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        1.0e-40,
        -1.0e-40,
        f32::MIN_POSITIVE,
        f32::from_bits(0x397f_ffff),
        f32::from_bits(0x3980_0000),
        std::f32::consts::FRAC_PI_4,
        std::f32::consts::FRAC_PI_2,
        -std::f32::consts::PI,
        16_777_216.0,
        f32::MAX,
        -f32::MAX,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 40) as f32 / (1u64 << 24) as f32;
            let sign = if s >> 63 == 0 { 1.0 } else { -1.0 };
            match i % 6 {
                0 => specials[(s >> 33) as usize % specials.len()],
                1 => u * 25.2 - 12.6,
                2 => u * 2.0 - 1.0,
                3 => sign * f32::from_bits(((s >> 20) as u32 % 254 + 1) << 23 | (s as u32 >> 9)),
                4 => sign * f32::from_bits(HARDEST[(s >> 33) as usize % HARDEST.len()]),
                _ => u * 2.0e4 - 1.0e4,
            }
        })
        .collect()
}

fn flex(v: &[f32], r: usize, c: usize) -> Tensor<Flex, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), &FlexDevice)
}

fn host(t: Tensor<Flex, 2>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

/// `got` within `rel` (relative) plus `abs` of Flex's `want`, plus Flex's ulp;
/// a NaN by class; an infinity exactly; a denormal or zero answer as a zero.
fn close(got: f32, want: f32, rel: f64, abs: f64, what: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: {got:e} vs NaN");
        return;
    }
    assert!(!got.is_nan(), "{what}: NaN vs {want:e}");
    if want.is_infinite() {
        assert_eq!(got, want, "{what}");
        return;
    }
    let (g, w) = (got as f64, want as f64);
    let tol = (rel + 1.2e-7) * w.abs() + abs + f32::MIN_POSITIVE as f64;
    assert!(
        (g - w).abs() <= tol,
        "{what}: {got:e} vs Flex {want:e}: {:e} > {tol:e}",
        (g - w).abs()
    );
}

fn op(kind: u32) -> Eltwise {
    Eltwise {
        kind,
        scalar: 0.0,
        scalar2: 0.0,
    }
}

/// Run `kind` on the device and hold it to its program and its padding
/// claim; return its values.
fn run<T: tt_device::Transport>(
    s: &mut Session<T>,
    kind: u32,
    t: &[&DramTensor],
    v: &[&[f32]],
    bcast: Broadcast,
    (r, c): (usize, usize),
    what: &str,
) -> Vec<f32> {
    let out = s
        .eltwise3(op(kind), t[0], t.get(1).copied(), t.get(2).copied())
        .unwrap();
    let got = s.download_bits(&out).unwrap();
    let model = reference_op(kind, [0.0; 2], bcast, v, r, c);
    for (i, (g, m)) in got.iter().zip(&model).enumerate() {
        assert_eq!(
            *g,
            m.to_bits(),
            "{what}: element {i}: device {:e}, program {m:e}",
            f32::from_bits(*g)
        );
    }
    if out.pad() == Pad::Zero {
        let raw = s.download_padded(&out).unwrap();
        let w = 32 * c.div_ceil(32);
        for (k, x) in raw.iter().enumerate() {
            let (i, j) = (k / w, k % w);
            assert!(
                (i < r && j < c) || x.to_bits() & 0x7fff_ffff == 0,
                "{what}: claimed zero padding holds {:#010x} at ({i}, {j})",
                x.to_bits()
            );
        }
    }
    s.free(out).unwrap();
    model
}

#[test]
fn sin_cos_and_tan_are_their_programs_within_their_bounds() {
    with_session(|s| {
        for (r, c) in [(37, 70), (64, 128)] {
            let av = values(1, r * c);
            let a = s.upload(&av, r, c).unwrap();
            let fa = flex(&av, r, c);
            for (kind, want, rel, what) in [
                (SIN, host(fa.clone().sin()), SIN_BOUND, "sin"),
                (COS, host(fa.clone().cos()), COS_BOUND, "cos"),
                (TAN, host(fa.clone().tan()), TAN_BOUND, "tan"),
            ] {
                let model = run(s, kind, &[&a], &[&av], Broadcast::None, (r, c), what);
                for i in 0..r * c {
                    // A denormal `sin x` is `x` itself on the device and in
                    // Flex; `close` takes it as a zero either way.
                    close(
                        model[i],
                        want[i],
                        rel,
                        0.0,
                        &format!("{what}({:e}) [{r}, {c}]", av[i]),
                    );
                }
            }
            s.free(a).unwrap();
        }
    });
}

#[test]
fn atan_and_atan2_are_their_programs_within_their_bounds() {
    let ftz = |x: f32| {
        if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
            f32::from_bits(x.to_bits() & 0x8000_0000)
        } else {
            x
        }
    };
    with_session(|s| {
        for (r, c) in [(37, 70), (64, 128)] {
            let (yv, xv) = (values(2, r * c), values(3, r * c));
            let (y, x) = (s.upload(&yv, r, c).unwrap(), s.upload(&xv, r, c).unwrap());
            let (fy, fx) = (flex(&yv, r, c), flex(&xv, r, c));
            let want = host(fy.clone().atan());
            let model = run(s, ATAN, &[&y], &[&yv], Broadcast::None, (r, c), "atan");
            for i in 0..r * c {
                close(
                    model[i],
                    want[i],
                    ATAN_BOUND,
                    0.0,
                    &format!("atan({:e}) [{r}, {c}]", yv[i]),
                );
            }
            let want = host(fy.atan2(fx));
            let model = run(
                s,
                ATAN2,
                &[&y, &x],
                &[&yv, &xv],
                Broadcast::None,
                (r, c),
                "atan2",
            );
            for i in 0..r * c {
                let (a, b) = (yv[i], xv[i]);
                let w = if ftz(a) != a || ftz(b) != b {
                    ftz(a).atan2(ftz(b))
                } else {
                    want[i]
                };
                let what = format!("atan2({a:e}, {b:e}) [{r}, {c}]");
                if w == 0.0 {
                    assert_eq!(model[i].to_bits(), w.to_bits(), "{what}");
                }
                close(model[i], w, ATAN2_BOUND, 0.0, &what);
            }
            s.free(y).unwrap();
            s.free(x).unwrap();
        }
    });
}

#[test]
fn asin_and_acos_are_their_programs_within_their_bounds() {
    with_session(|s| {
        for (r, c) in [(37, 70), (64, 128)] {
            // `values` folded into the domain, but for its specials and every
            // tenth value, which stay beyond it.
            let av: Vec<f32> = values(4, r * c)
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if !v.is_finite() || i % 10 == 0 {
                        *v
                    } else {
                        (v / 12.6).clamp(-1.0, 1.0)
                    }
                })
                .collect();
            let a = s.upload(&av, r, c).unwrap();
            let fa = flex(&av, r, c);
            for (kind, want, rel, what) in [
                (ASIN, host(fa.clone().asin()), ASIN_BOUND, "asin"),
                (ACOS, host(fa.clone().acos()), ACOS_BOUND, "acos"),
            ] {
                let model = run(s, kind, &[&a], &[&av], Broadcast::None, (r, c), what);
                for i in 0..r * c {
                    close(
                        model[i],
                        want[i],
                        rel,
                        0.0,
                        &format!("{what}({:e}) [{r}, {c}]", av[i]),
                    );
                }
            }
            s.free(a).unwrap();
        }
    });
}
