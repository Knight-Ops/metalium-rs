//! Phase 10 gate (S4, milestone 10.2d): `sqrt`, `1/sqrt`, `log1p`, `pow` and
//! the integer-to-float cast on the SFPU.
//!
//! Approximations, held twice as `step29_exp_log` holds `exp` and `log`: the
//! device **bit for bit** to its programs run by the interpreter
//! (`tt_kernels::sfpu::ops::reference_op`, `pow_reference` for `pow`'s
//! four-stage chain), and the programs to `burn-flex` within the bounds
//! derived on them (`sqrt` one ulp of the correct rounding; `RSQRT_BOUND`,
//! `LOG1P_BOUND`, `pow_bound(x, y)`) plus Flex's own ulp. Special values
//! exactly; a denormal input flushes (numerics row D). The cast is exact.
//! Also here, the device run of `recip` and division across every binade, since
//! 10.2d's sweeps found -- and fixed -- their Newton steps flushing for `|x| >
//! 2^111`.

use burn::tensor::{Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_kernels::session::{PowExponent, Session, TileChoice};
use tt_kernels::sfpu::ops::{
    kind_sfpu::*, pow_bound, pow_reference, reference_op, Broadcast, LOG1P_BOUND, RSQRT_BOUND,
};
use tt_kernels::tensor::{DramTensor, Elem, Eltwise};
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

/// Positive values of every magnitude, values around 1 and 0, negatives, and
/// the specials.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        f32::MIN_POSITIVE,
        2.0,
        0.5,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            match i % 4 {
                0 => specials[(s >> 33) as usize % specials.len()],
                1 => f32::from_bits(0x0080_0000 + ((s >> 34) as u32 % 0x7e80_0000)),
                2 => ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 1.0,
                _ => ((s >> 40) as f32 / (1u64 << 24) as f32) * 1.0e-3,
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

fn up<T: tt_device::Transport>(s: &mut Session<T>, v: &[f32], r: usize, c: usize) -> DramTensor {
    s.upload(v, r, c).unwrap()
}

/// `got` (the program's) within `rel` of Flex's `want` plus Flex's own ulp; a
/// NaN by class; an infinity, a zero and a flushed denormal exactly (sign
/// included where `signed`).
fn close(got: f32, want: f32, rel: f64, what: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{what}: {got:e} vs NaN");
    } else if want.is_infinite() || want.abs() < f32::MIN_POSITIVE {
        let w = if want.is_infinite() { want } else { 0.0 };
        assert_eq!(got.abs(), w.abs(), "{what}: {got:e} vs {want:e}");
        assert_eq!(
            got.is_sign_negative(),
            want.is_sign_negative(),
            "{what}: sign of {got:e}"
        );
    } else {
        let r = (got as f64 - want as f64).abs() / want.abs() as f64;
        assert!(r <= rel + 1.2e-7, "{what}: {got:e} vs Flex {want:e}: {r:e}");
    }
}

/// The device's bits are its program's.
fn same(got: &[f32], model: &[f32], what: &str) {
    for (i, (g, m)) in got.iter().zip(model).enumerate() {
        assert_eq!(
            g.to_bits(),
            m.to_bits(),
            "{what}: element {i}: device {g:e}, program {m:e}"
        );
    }
}

fn unary<T: tt_device::Transport>(s: &mut Session<T>, kind: u32, a: &DramTensor) -> Vec<f32> {
    let op = Eltwise {
        kind,
        scalar: 0.0,
        scalar2: 0.0,
    };
    let out = s.eltwise(op, a, None).unwrap();
    let v = s.download(&out).unwrap();
    s.free(out).unwrap();
    v
}

#[test]
fn sqrt_rsqrt_and_log1p_are_their_programs_within_their_bounds() {
    with_session(|s| {
        for (r, c) in [(37, 70), (64, 128)] {
            let av: Vec<f32> = values(1, r * c)
                .iter()
                .map(|x| {
                    if x.abs() < f32::MIN_POSITIVE && *x != 0.0 {
                        0.0
                    } else {
                        *x
                    }
                })
                .collect();
            let a = up(s, &av, r, c);
            let fa = flex(&av, r, c);
            let none = Broadcast::None;
            for (kind, want, rel, what) in [
                (SQRT, host(fa.clone().sqrt()), 1.2e-7, "sqrt"),
                (
                    RSQRT,
                    host(fa.clone().sqrt().recip()),
                    RSQRT_BOUND + 1.2e-7,
                    "rsqrt",
                ),
                (LOG1P, host(fa.clone().log1p()), LOG1P_BOUND, "log1p"),
            ] {
                let got = unary(s, kind, &a);
                let model = reference_op(kind, [0.0; 2], none, &[&av], r, c);
                same(&got, &model, what);
                for i in 0..r * c {
                    close(
                        model[i],
                        want[i],
                        rel,
                        &format!("{what}({:e}) [{r}, {c}]", av[i]),
                    );
                }
            }
            s.free(a).unwrap();
        }
    });
}

#[test]
fn pow_is_its_chain_within_its_bound() {
    with_session(|s| {
        for (r, c) in [(37, 70), (64, 64)] {
            let n = r * c;
            // Bases of every sign and size; exponents integral (odd and even),
            // fractional, and special.
            let xv: Vec<f32> = values(2, n)
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    if x.abs() < f32::MIN_POSITIVE && *x != 0.0 {
                        0.0
                    } else if i % 3 == 0 {
                        -*x
                    } else {
                        *x
                    }
                })
                .collect();
            let ys = [
                0.0f32,
                -0.0,
                1.0,
                2.0,
                3.0,
                -1.0,
                -3.0,
                0.5,
                -0.5,
                2.5,
                f32::INFINITY,
                f32::NAN,
                7.0,
                -2.25,
            ];
            let yv: Vec<f32> = (0..n).map(|i| ys[(i * 7 + i / 13) % ys.len()]).collect();
            let (x, y) = (up(s, &xv, r, c), up(s, &yv, r, c));
            let want = host(flex(&xv, r, c).powf(flex(&yv, r, c)));
            let out = s.pow(&x, PowExponent::Tensor(&y)).unwrap();
            let got = s.download(&out).unwrap();
            let model = pow_reference(&xv, Some(&yv), 0.0, r, c);
            same(&got, &model, "pow");
            for i in 0..n {
                let what = format!("pow({:e}, {:e}) [{r}, {c}]", xv[i], yv[i]);
                // Within the bound of the range's ends either side may round
                // over (`ops::transcendental` holds those lanes).
                let w = want[i] as f64;
                let b2 = 1.0 + 2.0 * pow_bound(xv[i], yv[i]);
                if w.is_finite()
                    && (w.abs() * b2 > f32::MAX as f64
                        || (w != 0.0 && w.abs() < f32::MIN_POSITIVE as f64 * b2))
                {
                    assert!(!model[i].is_nan(), "{what}: NaN at the range's end");
                    continue;
                }
                close(model[i], want[i], pow_bound(xv[i], yv[i]), &what);
            }
            s.free(out).unwrap();
            // A scalar exponent: the chain's scalar form, the same bits as a
            // tensor of it.
            for e in [2.5f32, -0.5, 3.0] {
                let out = s.pow(&x, PowExponent::Scalar(e)).unwrap();
                let got = s.download(&out).unwrap();
                same(
                    &got,
                    &pow_reference(&xv, None, e, r, c),
                    &format!("pow by {e}"),
                );
                s.free(out).unwrap();
            }
            // An `I32` exponent: `as f32`, then the chain.
            let iv: Vec<i32> = (0..n).map(|i| (i as i32 % 9) - 4).collect();
            let ib: Vec<u32> = iv.iter().map(|&v| v as u32).collect();
            let it = s.upload_bits(&ib, r, c, Elem::I32).unwrap();
            let out = s.pow(&x, PowExponent::Int(&it)).unwrap();
            let got = s.download(&out).unwrap();
            let yf: Vec<f32> = iv.iter().map(|&v| v as f32).collect();
            same(
                &got,
                &pow_reference(&xv, Some(&yf), 0.0, r, c),
                "pow by an I32 tensor",
            );
            s.free(out).unwrap();
        }
    });
}

#[test]
fn the_integer_cast_is_as_f32() {
    with_session(|s| {
        let mut st = 0x1234_5678u32;
        let mut iv: Vec<i32> = vec![
            0,
            1,
            -1,
            i32::MIN,
            i32::MAX,
            16_777_217,
            -16_777_217,
            0x7f80_0001,
        ];
        while iv.len() < 37 * 70 {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            iv.push((st as i32) >> (st % 31));
        }
        let ib: Vec<u32> = iv.iter().map(|&v| v as u32).collect();
        let t = s.upload_bits(&ib, 37, 70, Elem::I32).unwrap();
        let op = Eltwise {
            kind: I32_TO_F32,
            scalar: 0.0,
            scalar2: 0.0,
        };
        let out = s.eltwise(op, &t, None).unwrap();
        assert_eq!(out.elem, Elem::F32);
        let got = s.download(&out).unwrap();
        let flex: Vec<f32> =
            Tensor::<Flex, 2, Int>::from_data(TensorData::new(iv.clone(), [37, 70]), &FlexDevice)
                .float()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
        for i in 0..iv.len() {
            assert_eq!(got[i].to_bits(), flex[i].to_bits(), "{} as f32", iv[i]);
            assert_eq!(
                got[i].to_bits(),
                (iv[i] as f32).to_bits(),
                "{} as f32",
                iv[i]
            );
        }
    });
}

#[test]
fn recip_and_division_hold_in_every_binade() {
    with_session(|s| {
        // Every binade, positive and negative, as the reciprocal's input and
        // as a divisor of `1` and of large and small dividends.
        let bv: Vec<f32> = (0..64 * 64)
            .map(|i| {
                let e = 1 + (i as u32 * 7) % 254;
                let sign = if i % 2 == 0 { 0 } else { 0x8000_0000 };
                f32::from_bits(sign | (e << 23) | ((i as u32 * 104_729) & 0x7f_ffff))
            })
            .collect();
        let av: Vec<f32> = (0..64 * 64)
            .map(|i| [1.0f32, 3.0e30, -2.0e-20, 7.0][i % 4])
            .collect();
        let (a, b) = (up(s, &av, 64, 64), up(s, &bv, 64, 64));
        let got = unary(s, RECIP, &b);
        same(
            &got,
            &reference_op(RECIP, [0.0; 2], Broadcast::None, &[&bv], 64, 64),
            "recip",
        );
        let fr = host(flex(&bv, 64, 64).recip());
        for i in 0..bv.len() {
            close(got[i], fr[i], 1.2e-7, &format!("1/{:e}", bv[i]));
        }
        let op = Eltwise {
            kind: DIV,
            scalar: 0.0,
            scalar2: 0.0,
        };
        let out = s.eltwise(op, &a, Some(&b)).unwrap();
        let got = s.download(&out).unwrap();
        same(
            &got,
            &reference_op(DIV, [0.0; 2], Broadcast::None, &[&av, &bv], 64, 64),
            "div",
        );
        let fd = host(flex(&av, 64, 64).div(flex(&bv, 64, 64)));
        for i in 0..bv.len() {
            close(got[i], fd[i], 1.2e-7, &format!("{:e}/{:e}", av[i], bv[i]));
        }
    });
}
