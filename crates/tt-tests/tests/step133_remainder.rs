//! Gate (`float_remainder`, `float_remainder_scalar`): the exact
//! remainder on the SFPU.
//!
//! `burn-flex` 0.21 computes `((a % b) + b) % b` in `f32`: an exact `fmod`, an
//! IEEE add, an exact `fmod` (`ops/float.rs`). The program
//! (`tt_kernels::sfpu::ops::rem`) is held to **that, bit for bit, on every
//! input** -- denormal operands and results included, because it does its
//! arithmetic on integers and only feeds the device's flushing `SFPADD`
//! operands scaled out of the denormal range -- except a NaN's payload, which
//! is the canonical `0x7fc00000` where the host's libm picks its own (so NaN
//! is compared by class).
//!
//! Three independent oracles, none of them the code under test:
//!
//! - [`HAND`]: expected bits computed with Python's exact rationals
//!   (`fractions.Fraction`, correct rounding by integer arithmetic), not by any
//!   floating-point `fmod` -- `1e30 % 7`, signed zeros, the overflow of `a + b`,
//!   denormal operands, ...;
//! - [`oracle`]: exact `fmod` through `f64` (the result is representable in
//!   `f32`), then the add rounded by `f32`'s IEEE add, then `fmod` again --
//!   a different evaluation path from Flex's `f32 %` (cross-checked against it
//!   below);
//! - `burn-flex` itself, in `step134_burn_remainder`.
//!
//! The device is held bit for bit to its own program run by the interpreter
//! (`ops::reference`) and the interpreter to the oracles. The negative controls
//! run the same checker over deliberately wrong programs
//! ([`Variant`]): `a - b * trunc(a / b)`, no `+ b` correction, swapped
//! operands -- each must be caught.

use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::interp::Vector;
use tt_kernels::sfpu::kernel;
use tt_kernels::sfpu::ops::rem::{self, Divisor, Variant};
use tt_kernels::sfpu::ops::{kind_sfpu, reference};
use tt_kernels::tensor::Eltwise;
use tt_tests::backend::GATE_TILE;
use tt_tests::data::xorshift;

/// `(a, b, a % b)` as bits, from exact rational arithmetic.
const HAND: &[(u32, u32, u32)] = &[
    (0x40a00000, 0x40400000, 0x40000000), // 5.0 % 3.0
    (0xc0a00000, 0x40400000, 0x3f800000), // -5.0 % 3.0
    (0x40a00000, 0xc0400000, 0xbf800000), // 5.0 % -3.0
    (0xc0a00000, 0xc0400000, 0xc0000000), // -5.0 % -3.0
    (0xc0800000, 0x40000000, 0x00000000), // -4.0 % 2.0
    (0x40800000, 0xc0000000, 0x80000000), // 4.0 % -2.0
    (0x00000000, 0xc0400000, 0x80000000), // 0.0 % -3.0
    (0x80000000, 0x40400000, 0x00000000), // -0.0 % 3.0
    (0x80000000, 0xc0400000, 0x80000000), // -0.0 % -3.0
    (0x00000000, 0x40400000, 0x00000000), // 0.0 % 3.0
    (0x40f00000, 0x40000000, 0x3fc00000), // 7.5 % 2.0
    (0xc0f00000, 0x40000000, 0x3f000000), // -7.5 % 2.0
    (0x7149f2ca, 0x40400000, 0x00000000), // 1e+30 % 3.0 (1e30f is a multiple of 3)
    (0xf149f2ca, 0x40400000, 0x00000000), // -1e+30 % 3.0
    (0x7149f2ca, 0xc0400000, 0x80000000), // 1e+30 % -3.0
    (0x1e3ce508, 0xbf800000, 0x80000000), // 1e-20 % -1.0 (r + b rounds to b: -0)
    (0x9e3ce508, 0x3f800000, 0x00000000), // -1e-20 % 1.0
    (0x1e3ce508, 0x3f800000, 0x00000000), // 1e-20 % 1.0
    (0x9e3ce508, 0xbf800000, 0x80000000), // -1e-20 % -1.0
    (0x7f61b1e6, 0x7f7843b0, 0x7fc00000), // 3e+38 % 3.3e+38 (a + b overflows: NaN)
    (0x7f7fc99e, 0x7effc99e, 0x00000000), // 3.4e+38 % 1.7e+38
    (0xff7fc99e, 0x7effc99e, 0x00000000), // -3.4e+38 % 1.7e+38
    (0x7f7fffff, 0x40400000, 0x00000000), // f32::MAX % 3.0
    (0x7f7fffff, 0x0da24260, 0x0da11ba0), // f32::MAX % 1e-30
    (0x7f7fffff, 0x00000001, 0x00000000), // f32::MAX % the least denormal
    (0x3f800000, 0x00000001, 0x00000000), // 1.0 % the least denormal
    (0xbf800000, 0x00000002, 0x00000000), // -1.0 % 3e-45
    (0x007ffffd, 0x000116c2, 0x00009953), // 1.175494e-38 % 1e-40 (denormal result)
    (0x003671f7, 0x0020aac8, 0x0015c72f), // 5e-39 % 3e-39
    (0x803671f7, 0x0020aac8, 0x000ae399), // -5e-39 % 3e-39
    (0x3f800000, 0x3dcccccd, 0x3dcccccb), // 1.0 % 0.1
    (0xbf800000, 0x3dcccccd, 0x32800000), // -1.0 % 0.1
    (0x3e99999a, 0x3dcccccd, 0x32000000), // 0.3 % 0.1
    (0x501502f9, 0x3f333333, 0x3f32f0e1), // 1e10 % 0.7
    (0x47f12065, 0xba83126f, 0xba5213e0), // 123456.789 % -0.001
    (0x7149f2ca, 0x40e00000, 0x3f800000), // 1e+30 % 7.0
    (0x7149f2ca, 0x40a00000, 0x00000000), // 1e+30 % 5.0
    (0xf149f2ca, 0x447a0000, 0x44700000), // -1e+30 % 1000.0
    (0x7e967699, 0x3e99999a, 0x3ceb9200), // 1e+38 % 0.3
    (0xfe967699, 0x3b449ba6, 0x3b0d4846), // -1e+38 % 0.003
    (0x7e7fc99e, 0x7e348e52, 0x7d967698), // 8.5e+37 % 6e+37
    (0x7149f2ca, 0xc0e00000, 0xc0c00000), // 1e+30 % -7.0
    (0xf149f2ca, 0xc0a00000, 0x80000000), // -1e+30 % -5.0
    (0x3f7fffff, 0x3f800000, 0x00000000), // 0.99999994 % 1.0
    (0xbf7fffff, 0x3f800000, 0x33800000), // -0.99999994 % 1.0
    (0x3f000000, 0x40400000, 0x3f000000), // 0.5 % 3.0
    (0xbf000000, 0x40400000, 0x40200000), // -0.5 % 3.0
    (0x4b800000, 0x40400000, 0x3f800000), // 16777216.0 % 3.0
    (0x4b800000, 0x40000000, 0x00000000), // 16777216.0 % 2.0
];

/// `fmod` exact: through `f64`, where every `f32` is a number and the result is
/// representable.
fn fmod32(a: f32, b: f32) -> f32 {
    ((a as f64) % (b as f64)) as f32
}

/// Flex's formula by another route.
fn oracle(a: f32, b: f32) -> f32 {
    fmod32(fmod32(a, b) + b, b)
}

/// `got` is `want`'s bits, a NaN by class.
fn same(got: u32, want: f32) -> bool {
    if want.is_nan() {
        f32::from_bits(got).is_nan()
    } else {
        got == want.to_bits()
    }
}

/// Operands from every region the algorithm treats differently: arbitrary
/// bits, denormals, the exponents around 1, every normal exponent, the
/// smallest and the largest.
fn corpus(seed: u64, n: usize) -> Vec<u32> {
    let mut next = xorshift(seed);
    (0..n)
        .map(|_| {
            let (x, k) = (next(), next() >> 40);
            let sign = (x as u32) & 0x8000_0000;
            let frac = (x as u32) & 0x7f_ffff;
            let e = |lo: u32, span: u32| (((x >> 40) as u32) % span + lo) << 23;
            match k % 6 {
                0 => x as u32,
                1 => sign | frac,
                2 => sign | e(100, 40) | frac,
                3 => sign | e(1, 254) | frac,
                4 => sign | e(0, 12) | frac,
                _ => sign | e(243, 12) | frac,
            }
        })
        .collect()
}

/// The special values and the hand table's operands, crossed, then the corpus:
/// `n` pairs.
fn operand_pairs(seed: u64, n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut specials: Vec<u32> = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.0,
        3.0,
        -3.0,
        0.5,
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x7f_ffff),
        f32::from_bits(0x0040_0000),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xffc0_0000),
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xff80_0001),
        1e30,
        -1e30,
        1e-20,
        3.0e38,
        3.3e38,
        0.99999994,
        16777216.0,
    ]
    .iter()
    .map(|v| v.to_bits())
    .collect();
    specials.extend(HAND.iter().flat_map(|&(a, b, _)| [a, b]));
    let (mut a, mut b) = (Vec::new(), Vec::new());
    'cross: for &x in &specials {
        for &y in &specials {
            if a.len() == n {
                break 'cross;
            }
            a.push(x);
            b.push(y);
        }
    }
    let rest = n - a.len();
    a.extend(corpus(seed, rest));
    b.extend(corpus(seed + 1000, rest));
    (a, b)
}

fn floats(v: &[u32]) -> Vec<f32> {
    v.iter().map(|&b| f32::from_bits(b)).collect()
}

#[test]
fn the_oracle_is_flexs_formula_and_the_hand_table() {
    for &(a, b, want) in HAND {
        let (x, y) = (f32::from_bits(a), f32::from_bits(b));
        let flex = ((x % y) + y) % y;
        assert!(
            same(oracle(x, y).to_bits(), f32::from_bits(want)),
            "oracle {x:e} % {y:e}"
        );
        assert!(
            same(flex.to_bits(), f32::from_bits(want)),
            "flex {x:e} % {y:e}"
        );
    }
    for seed in 1..5 {
        let (a, b) = operand_pairs(seed, 20_000);
        for (&a, &b) in a.iter().zip(&b) {
            let (x, y) = (f32::from_bits(a), f32::from_bits(b));
            let flex = ((x % y) + y) % y;
            assert!(same(oracle(x, y).to_bits(), flex), "{x:e} % {y:e}");
        }
    }
}

/// The program `variant` run by the interpreter over `a`, `b` (`a` the first
/// operand), a tile at a time.
fn run_program(variant: Variant, a: &[u32], b: &[u32]) -> Vec<u32> {
    let code = rem::code(variant, Divisor::Tile).expand();
    let mut out = Vec::new();
    for (ta, tb) in a.chunks(1024).zip(b.chunks(1024)) {
        let (mut ta, mut tb) = (ta.to_vec(), tb.to_vec());
        ta.resize(1024, 0);
        tb.resize(1024, 0);
        let mut v = Vector::new();
        v.put_tile(kernel::A_ROW as usize, &ta);
        v.put_tile(kernel::B_ROW as usize, &tb);
        v.run(&code).unwrap();
        out.extend(v.tile(kernel::OUT_ROW as usize));
    }
    out.truncate(a.len());
    out
}

/// The indices where `variant`'s program is not the oracle.
fn mismatches(variant: Variant, a: &[u32], b: &[u32]) -> Vec<usize> {
    let got = run_program(variant, a, b);
    (0..a.len())
        .filter(|&i| !same(got[i], oracle(f32::from_bits(a[i]), f32::from_bits(b[i]))))
        .collect()
}

fn hand_operands() -> (Vec<u32>, Vec<u32>) {
    (
        HAND.iter().map(|h| h.0).collect(),
        HAND.iter().map(|h| h.1).collect(),
    )
}

#[test]
fn the_program_is_the_oracle_on_the_hand_table_and_the_corpus() {
    let (a, b) = hand_operands();
    let got = run_program(Variant::Exact, &a, &b);
    for (i, &(x, y, want)) in HAND.iter().enumerate() {
        assert!(
            same(got[i], f32::from_bits(want)),
            "{:e} % {:e}: program {:08x}, exact {want:08x}",
            f32::from_bits(x),
            f32::from_bits(y),
            got[i]
        );
    }
    for seed in 1..=6 {
        let (a, b) = operand_pairs(seed, 4096);
        let bad = mismatches(Variant::Exact, &a, &b);
        assert!(
            bad.is_empty(),
            "seed {seed}: {} mismatches, first {:08x} % {:08x}",
            bad.len(),
            a[bad[0]],
            b[bad[0]]
        );
    }
}

/// A program that is not the remainder fails the gate: each plausible mistake,
/// run through the same checker, is caught -- on the case the brief names where
/// it names one -- and the real program is not.
#[test]
fn the_negative_controls_are_caught() {
    let (a, b) = hand_operands();
    let index = |x: f32, y: f32| {
        HAND.iter()
            .position(|h| (h.0, h.1) == (x.to_bits(), y.to_bits()))
            .unwrap_or_else(|| panic!("{x} % {y} is not in the table"))
    };
    assert!(mismatches(Variant::Exact, &a, &b).is_empty());

    // `a - b * trunc(a / b)`: wrong where the quotient is not exact (`1e30 % 7`,
    // `-1e30 % 1000`), and in the sign of a zero (`4 % -2` is `-0`: Flex's
    // outer `%` takes the sign of `y = -2`).
    let trunc = mismatches(Variant::TruncReduction, &a, &b);
    for (x, y) in [(1e30f32, 7.0f32), (-1e30, 1000.0), (4.0, -2.0), (1e38, 0.3)] {
        assert!(
            trunc.contains(&index(x, y)),
            "trunc reduction passed {x} % {y}"
        );
    }
    // The same on the corpus: many, not a stray one.
    let (ca, cb) = operand_pairs(7, 2048);
    assert!(mismatches(Variant::TruncReduction, &ca, &cb).len() > 100);

    // Without the `+ b` correction the sign is `a`'s: `-5 % 3` is `-2`.
    let plain = mismatches(Variant::NoCorrection, &a, &b);
    // (`-4 % 2` is `-0` without it, Flex's `+0`: the brief's bare
    // `a - b * trunc(a / b)` fails there; with Flex's correction applied to
    // it, as `TruncReduction` does, it does not -- `1e30` is a multiple of 3
    // as an `f32` too, so `1e30 % 3 = 0` is no discriminator either.)
    for (x, y) in [
        (-5.0f32, 3.0f32),
        (5.0, -3.0),
        (-7.5, 2.0),
        (4.0, -2.0),
        (-4.0, 2.0),
    ] {
        assert!(
            plain.contains(&index(x, y)),
            "no correction passed {x} % {y}"
        );
    }
    assert!(mismatches(Variant::NoCorrection, &ca, &cb).len() > 100);

    // `b % a`: `5 % 3` is `2`, `3 % 5` is `3`.
    let swapped = mismatches(Variant::Swapped, &a, &b);
    for (x, y) in [(5.0f32, 3.0f32), (-5.0, 3.0), (7.5, 2.0), (0.5, 3.0)] {
        assert!(swapped.contains(&index(x, y)), "swapped passed {x} % {y}");
    }
    assert!(mismatches(Variant::Swapped, &ca, &cb).len() > 100);
}

/// The size of the program in a role's slot and the instructions a tile runs
/// (interpreter counts: the stream the runner pushes), for the performance
/// notes; not a timing.
#[test]
fn the_program_fits_a_role_slot() {
    for divisor in [Divisor::Tile, Divisor::Scalar(3.5)] {
        let code = rem::code(Variant::Exact, divisor);
        let (words, _) = code.stored().unwrap();
        eprintln!(
            "remainder {divisor:?}: {} stored words (slot 8192), loops {:?}, {} instructions per tile",
            words.len(),
            code.loops,
            code.expand().len()
        );
        assert!(words.len() <= 8192);
        assert_eq!(code.loops.len(), 1, "the row loop, stored once");
    }
}

fn with_session(f: impl FnOnce(&mut tt_tests::backend::Sess<'_>)) {
    tt_tests::backend::with_session(TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1), f);
}

/// The scalars of the scalar form: a normal, negative, inexact, denormal,
/// huge, a zero of either sign, an infinity, a NaN, and the least denormal.
const SCALARS: [f32; 13] = [
    3.0,
    -2.0,
    0.1,
    1.0e-40,
    f32::MAX,
    -f32::MIN_POSITIVE,
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NAN,
    7.0,
    f32::from_bits(1),
    -1.0e-5,
];

/// The device, on a ragged `[37, 35]` matrix of 4 tiles: bit for bit the
/// program, and the program the oracle. Edge tiles hold zero padding, which for
/// the tensor form is `0 % 0 = NaN`: the tensor's padding is declared
/// `Undefined` (`OpPadding`), which `step134` chains into a reduction.
#[test]
fn the_device_is_the_program_and_the_oracle() {
    let (rows, cols) = (37, 35);
    let (a, b) = operand_pairs(11, rows * cols);
    let (av, bv) = (floats(&a), floats(&b));
    with_session(|s| {
        let ta = s.upload(&av, rows, cols).unwrap();
        let tb = s.upload(&bv, rows, cols).unwrap();
        let op = |kind, scalar| Eltwise {
            kind,
            scalar,
            scalar2: 0.0,
        };
        let out = s
            .eltwise(op(kind_sfpu::REM, 0.0), &ta, Some(&tb))
            .unwrap_or_else(|e| panic!("rem: {e}"));
        let got = s.download_bits(&out).unwrap();
        let model = reference(kind_sfpu::REM, 0.0, &av, Some(&bv), rows, cols);
        for i in 0..rows * cols {
            assert_eq!(got[i], model[i].to_bits(), "element {i}: device vs program");
            let want = oracle(av[i], bv[i]);
            assert!(
                same(got[i], want),
                "{:e} % {:e}: device {:08x}, oracle {:08x}",
                av[i],
                bv[i],
                got[i],
                want.to_bits()
            );
        }
        s.free(out).unwrap();
        for scalar in SCALARS {
            let out = s
                .eltwise(op(kind_sfpu::REM_S, scalar), &ta, None)
                .unwrap_or_else(|e| panic!("rem scalar {scalar}: {e}"));
            let got = s.download_bits(&out).unwrap();
            let model = reference(kind_sfpu::REM_S, scalar, &av, None, rows, cols);
            for i in 0..rows * cols {
                assert_eq!(
                    got[i],
                    model[i].to_bits(),
                    "{scalar:e}: element {i}: device vs program"
                );
                assert!(
                    same(got[i], oracle(av[i], scalar)),
                    "{:e} % {scalar:e}: device {:08x}, oracle {:08x}",
                    av[i],
                    got[i],
                    oracle(av[i], scalar).to_bits()
                );
            }
            s.free(out).unwrap();
        }
        s.free(ta).unwrap();
        s.free(tb).unwrap();
    });
}

/// What `t`'s raw tiles hold past its last row or column.
fn padding<T: tt_device::Transport>(
    s: &mut Session<T>,
    t: &tt_kernels::tensor::DramTensor,
) -> Vec<f32> {
    let [rt, ct] = t.grid();
    let raw = s.download_padded(t).unwrap();
    let mut out = Vec::new();
    for r in 0..32 * rt {
        for c in 0..32 * ct {
            if r >= t.rows || c >= t.cols {
                out.push(raw[r * 32 * ct + c]);
            }
        }
    }
    out
}

/// What `pad()` claims is what the raw tiles hold, and a producer's dirty
/// padding never reaches an accumulation: the tensor form's padding is
/// `0 % 0 = NaN` (claimed `Undefined`, and it is NaN), the scalar form's
/// `0 % 3 = 0` (claimed `Zero`, and every datum is a zero), a scalar that is a
/// zero, an infinity or a NaN leaves NaN (`Undefined`). Each producer is chained
/// into a matmul with a ragged `K`, held to the same values uploaded fresh
/// (zero padding) through the same matmul.
#[test]
fn padding_claims_are_honest_and_reach_no_accumulation() {
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_kernels::tensor::Pad;
    let (rows, cols) = (37, 35);
    let finite = |v: Vec<u32>, lo: f32, hi: f32| -> Vec<f32> {
        floats(&v)
            .into_iter()
            .map(|x| if x.is_finite() { x.clamp(lo, hi) } else { 1.0 })
            .collect()
    };
    let av = finite(corpus(21, rows * cols), -1e6, 1e6);
    let bv: Vec<f32> = finite(corpus(22, rows * cols), -1e3, 1e3)
        .into_iter()
        .map(|x| if x == 0.0 { 1.5 } else { x })
        .collect();
    // The matmul's other operand is itself a remainder of ragged shape `[35,
    // 24]`: its padding rows (along `K`) are NaN too, so a zero there cannot
    // hide the dirt of this side (`NaN * 0`); the FPU's `0 * NaN` is not
    // relied on.
    let wa: Vec<f32> = (0..cols * 24)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.5)
        .collect();
    let wb: Vec<f32> = (0..cols * 24)
        .map(|i| ((i % 5) as f32 + 1.0) * if i % 3 == 0 { -1.5 } else { 1.0 })
        .collect();
    let wv: Vec<f32> = wa.iter().zip(&wb).map(|(&x, &y)| oracle(x, y)).collect();
    with_session(|s| {
        let a = s.upload(&av, rows, cols).unwrap();
        let b = s.upload(&bv, rows, cols).unwrap();
        let (wa, wb) = (
            s.upload(&wa, cols, 24).unwrap(),
            s.upload(&wb, cols, 24).unwrap(),
        );
        let w = s
            .eltwise(
                Eltwise {
                    kind: kind_sfpu::REM,
                    scalar: 0.0,
                    scalar2: 0.0,
                },
                &wa,
                Some(&wb),
            )
            .unwrap();
        assert_eq!(w.pad(), Pad::Undefined, "the partner is dirty on `K`");
        let w_fresh = s.upload(&wv, cols, 24).unwrap();
        let op = |kind, scalar| Eltwise {
            kind,
            scalar,
            scalar2: 0.0,
        };
        let mut producers = vec![(
            "a % b".to_string(),
            s.eltwise(op(kind_sfpu::REM, 0.0), &a, Some(&b)).unwrap(),
            Pad::Undefined,
            None,
        )];
        for (scalar, claim) in [
            (3.0f32, Pad::Zero),
            (-0.1, Pad::Zero),
            (0.0, Pad::Undefined),
            (f32::INFINITY, Pad::Undefined),
            (f32::NAN, Pad::Undefined),
        ] {
            producers.push((
                format!("a % {scalar}"),
                s.eltwise(op(kind_sfpu::REM_S, scalar), &a, None).unwrap(),
                claim,
                Some(scalar),
            ));
        }
        for (what, t, claim, scalar) in producers {
            assert_eq!(t.pad(), claim, "{what}: the claim");
            let held = padding(s, &t);
            if claim == Pad::Zero {
                assert!(
                    held.iter().all(|&v| v == 0.0),
                    "{what}: claims zero padding, holds {:?}",
                    &held[..4]
                );
            } else {
                // `Undefined` is the truth here, not a conservative claim.
                assert!(
                    held.iter().all(|v| v.is_nan()),
                    "{what}: undefined padding is NaN, holds {:?}",
                    &held[..4]
                );
            }
            let got = s.download(&t).unwrap();
            for i in 0..rows * cols {
                let want = oracle(av[i], scalar.unwrap_or(bv[i]));
                assert!(same(got[i].to_bits(), want), "{what}: element {i}");
            }
            let matmul = |s: &mut Session<_>,
                          x: &tt_kernels::tensor::DramTensor,
                          y: &tt_kernels::tensor::DramTensor| {
                s.matmul_dram(
                    x,
                    false,
                    y,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    tt_tests::harness::BUDGET,
                )
                .unwrap()
            };
            let p = matmul(s, &t, &w);
            let product = s.download(&p).unwrap();
            let fresh = s.upload(&got, rows, cols).unwrap();
            let q = matmul(s, &fresh, &w_fresh);
            let reference = s.download(&q).unwrap();
            for i in 0..rows * 24 {
                assert!(
                    same(product[i].to_bits(), reference[i]),
                    "{what}: matmul element {i}: {} vs fresh upload {}",
                    product[i],
                    reference[i]
                );
            }
            for x in [p, q, fresh, t] {
                s.free(x).unwrap();
            }
        }
        for x in [a, b, wa, wb, w, w_fresh] {
            s.free(x).unwrap();
        }
    });
}
