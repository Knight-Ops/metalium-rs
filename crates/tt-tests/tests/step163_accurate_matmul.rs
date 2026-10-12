//! `MatmulMode::Accurate`: a matmul as integer slices with shared exponents, on the Matrix Unit.
//!
//! The fast matmul rounds the products of each group of eight to a quantum set by the group's
//! largest term (`step158_matmul_accumulation`, the measured rule). Accurate removes that by
//! arithmetic: each row of the left operand and column of the right one is scaled by its own power
//! of two and cut into integer slices of 7 and 5 bits, which at `Fidelity::Lo` the unit multiplies
//! and adds without any rounding (step158 holds that: `lofi_is_exact_on_seven_by_five_bit_integers`);
//! about a dozen slice pairs are multiplied and added from the smallest (`ops/accurate_matmul.rs`).
//!
//! # The bound
//!
//! Derived from the construction, per entry, with `va` the power of two at or below the row's
//! largest magnitude (at least `2^-64`) and `vb` the column's:
//!
//! * the left operand's four slices leave a remainder below `2^-21` of the last slice's unit, so an
//!   operand is off by under `2^-27 * 2 va`... (`|dA| < va 2^-27`); the right one's five leave
//!   `|dB| < vb 2^-24`;
//! * the slice pairs not multiplied (total weight 24 and over: `(2,5) (3,3) (3,4) (3,5) (4,2) (4,3)
//!   (4,4) (4,5)`) are at most `127 * 31 / 2^10 va vb 2^-w` each, in all under `va vb 2^-21.4`;
//! * together `k va vb 2^-21` per entry, with `|a| < 2 va` and `|b| < 2 vb`;
//! * the F32 adds into `Dst` and between partial products round, at most `2^-24` of the running
//!   value each, and there are `k / 16 + 12` of them: `(k / 16 + 12) 2^-24 sum |a b|`.
//!
//! The first two terms scale with the row and column maxima, not with the entry: a row of very
//! different magnitudes keeps its small entries only to that scale, which is what the bound says.
//!
//! # What the gate establishes
//!
//! Random operands of several shapes and exponent spreads stay within the bound, where the fast mode
//! leaves it; the same bound fails when the slicing rule is broken (a scale one bit too large, or
//! slice pairs dropped); integer operands whose sums stay below 2^24 are exact, `arange` data of
//! Burn's `conv3d` tests included; NaN and infinity follow IEEE against an f64 oracle; views, batch
//! broadcast and rank mismatch, BF16 and F16 operands, and gradients come out the same; and the
//! domain's edge (a subnormal, a value past `2^64`) does what the module says.

use burn::backend::Autodiff;
use burn::tensor::{FloatDType, Tensor, TensorData};
use burn_tt::{
    accurate_matmul, set_accurate_matmul_slicing, with_matmul_mode, MatmulMode, TtBackend, TtDevice,
};
use tt_tests::burn_device::{with_device, Config};

mod fp16_support;
use fp16_support::{oracle_to_f16, oracle_to_f32};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
    /// `unit()` times a power of two drawn from `-spread / 2 ..= spread / 2`.
    fn spread(&mut self, spread: u64) -> f32 {
        let e = if spread == 0 {
            0
        } else {
            (self.next() % (spread + 1)) as i32 - spread as i32 / 2
        };
        self.unit() * 2f32.powi(e)
    }
}

const FAST: MatmulMode = MatmulMode::Fast;
const ACCURATE: MatmulMode = MatmulMode::Accurate;

fn tensor2(d: &TtDevice, v: &[f32], shape: [usize; 2]) -> Tensor<TtBackend, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), d)
}

fn matmul(d: &TtDevice, mode: MatmulMode, a: &[f32], b: &[f32], [m, k, n]: [usize; 3]) -> Vec<f32> {
    with_matmul_mode(mode, || {
        tensor2(d, a, [m, k])
            .matmul(tensor2(d, b, [k, n]))
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    })
}

/// The exact products' sum, IEEE in f64 (a NaN or an infinity in an operand reaches the sum the same
/// way in any order).
fn reference(a: &[f32], b: &[f32], [m, k, n]: [usize; 3]) -> Vec<f64> {
    let mut c = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0f64;
            for p in 0..k {
                sum += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// The power of two at or below `x`, at least `2^-64` (the module's `float_pow2_floor` of the clamped
/// row maximum).
fn floor_pow2(x: f64) -> f64 {
    let x = x.max(2f64.powi(-64));
    2f64.powi(x.log2().floor() as i32)
}

/// The derived bound of every entry (see the module docs). Non-finite entries count as zero: they are
/// taken out of the slicing.
fn bound(a: &[f32], b: &[f32], [m, k, n]: [usize; 3]) -> Vec<f64> {
    let fin = |x: f32| {
        if x.is_finite() {
            f64::from(x.abs())
        } else {
            0.0
        }
    };
    let va: Vec<f64> = (0..m)
        .map(|i| floor_pow2((0..k).map(|p| fin(a[i * k + p])).fold(0.0, f64::max)))
        .collect();
    let vb: Vec<f64> = (0..n)
        .map(|j| floor_pow2((0..k).map(|p| fin(b[p * n + j])).fold(0.0, f64::max)))
        .collect();
    let mut out = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let abs_sum: f64 = (0..k).map(|p| fin(a[i * k + p]) * fin(b[p * n + j])).sum();
            out[i * n + j] = k as f64 * va[i] * vb[j] * 2f64.powi(-21)
                + (k as f64 / 16.0 + 12.0) * 2f64.powi(-24) * abs_sum;
        }
    }
    out
}

/// Entries beyond their bound.
fn violations(got: &[f32], want: &[f64], bound: &[f64]) -> usize {
    got.iter()
        .zip(want)
        .zip(bound)
        .filter(|((g, w), b)| (f64::from(**g) - **w).abs() > **b)
        .count()
}

fn operands(rng: &mut Rng, [m, k, n]: [usize; 3], spread: u64) -> (Vec<f32>, Vec<f32>) {
    (
        (0..m * k).map(|_| rng.spread(spread)).collect(),
        (0..k * n).map(|_| rng.spread(spread)).collect(),
    )
}

/// The shapes: single elements, ragged faces and tiles, more than one K-block, a single row and
/// column, a long `k`.
const SHAPES: [[usize; 3]; 8] = [
    [1, 1, 1],
    [5, 7, 3],
    [32, 32, 32],
    [33, 65, 31],
    [64, 96, 48],
    [1, 33, 1],
    [40, 300, 24],
    [3, 2000, 5],
];

/// Random operands, every shape and four exponent spreads, within the derived bound; the fast mode on
/// the same data leaves it (so the bound is not vacuous), and a zero row and column come out zero.
#[test]
fn random_operands_are_within_the_derived_bound() {
    with_device(Config::default(), |d| {
        let (mut fast_over, mut worst_ratio) = (0usize, 0f64);
        for (case, shape) in SHAPES.iter().enumerate() {
            for spread in [0u64, 6, 12, 20] {
                let mut rng = Rng(0x9e37_79b9_7f4a_7c15
                    ^ (case as u64 * 4 + spread + 1).wrapping_mul(0x1000_0000_01b3));
                let [m, k, n] = *shape;
                let (mut a, mut b) = operands(&mut rng, *shape, spread);
                // A zero row of the left operand and a zero column of the right one.
                a[..k].fill(0.0);
                for p in 0..k {
                    b[p * n + n - 1] = 0.0;
                }
                let (want, limit) = (reference(&a, &b, *shape), bound(&a, &b, *shape));
                let got = matmul(&d, ACCURATE, &a, &b, *shape);
                assert_eq!(
                    violations(&got, &want, &limit),
                    0,
                    "{shape:?} spread {spread}: an entry is beyond its bound"
                );
                assert!(got[..n].iter().all(|x| *x == 0.0), "a zero row");
                assert!((0..m).all(|i| got[i * n + n - 1] == 0.0), "a zero column");
                for (g, (w, l)) in got.iter().zip(want.iter().zip(&limit)) {
                    if *l > 0.0 {
                        worst_ratio = worst_ratio.max((f64::from(*g) - w).abs() / l);
                    }
                }
                let fast = matmul(&d, FAST, &a, &b, *shape);
                fast_over += violations(&fast, &want, &limit);
            }
        }
        // The control: the fast mode is outside the bound on this data, the accurate one well inside.
        assert!(
            fast_over > 100,
            "the fast matmul is not outside the bound: {fast_over}"
        );
        assert!(worst_ratio <= 1.0 && worst_ratio > 1e-6, "{worst_ratio}");
        eprintln!("worst error over its bound: {worst_ratio:.3e}; the fast matmul left it {fast_over} times");
    });
}

/// The slicing rules are needed: the same data and bound fail when a scale is one bit too large (the
/// leading slices then have 8 and 6 bits, which the unit rounds), and when slice pairs are dropped.
#[test]
fn the_slicing_rules_are_needed() {
    with_device(Config::default(), |d| {
        let shape = [64usize, 96, 48];
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let (a, b) = operands(&mut rng, shape, 6);
        let (want, limit) = (reference(&a, &b, shape), bound(&a, &b, shape));
        assert_eq!(
            violations(&matmul(&d, ACCURATE, &a, &b, shape), &want, &limit),
            0
        );
        for (what, offset, shift) in [
            ("scale one bit too large", 23, 1),
            ("only the leading pair", 0, 0),
            ("pairs of weight 12 and under", 12, 0),
        ] {
            set_accurate_matmul_slicing(offset, shift);
            let over = violations(&matmul(&d, ACCURATE, &a, &b, shape), &want, &limit);
            set_accurate_matmul_slicing(23, 0);
            assert!(over > 0, "{what}: the gate cannot tell");
        }
        assert_eq!(
            violations(&matmul(&d, ACCURATE, &a, &b, shape), &want, &limit),
            0
        );
    });
}

/// Integer operands whose every sum stays below 2^24 give the exact result, bit for bit: every bit
/// lands in a slice and nothing is rounded. Burn's `conv3d` `arange` data is of this kind and is what
/// the fast mode gets wrong (step158); the control checks that it does here.
#[test]
fn integer_operands_are_exact() {
    with_device(Config::default(), |d| {
        let mut fast_wrong = 0;
        for (case, (k, max_a, max_b)) in [
            (16usize, 127u64, 31u64),
            (27, 720, 288),
            (64, 255, 255),
            (300, 90, 90),
        ]
        .into_iter()
        .enumerate()
        {
            let mut rng = Rng(0xdead_beef ^ ((case as u64 + 1) * 0x1000_0000_01b3));
            let (m, n) = (96usize, 40usize);
            let sign = |rng: &mut Rng, v: u64| {
                if rng.next() % 2 == 0 {
                    v as f32
                } else {
                    -(v as f32)
                }
            };
            let a: Vec<f32> = (0..m * k)
                .map(|_| {
                    let v = 1 + rng.next() % max_a;
                    sign(&mut rng, v)
                })
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|_| {
                    let v = 1 + rng.next() % max_b;
                    sign(&mut rng, v)
                })
                .collect();
            let shape = [m, k, n];
            let want = reference(&a, &b, shape);
            assert!(
                want.iter().all(|x| x.abs() < 16_777_216.0),
                "case {case}: sums past 2^24"
            );
            let got = matmul(&d, ACCURATE, &a, &b, shape);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(f64::from(*g), *w, "k {k}, entry {i}");
            }
            let fast = matmul(&d, FAST, &a, &b, shape);
            fast_wrong += fast
                .iter()
                .zip(&want)
                .filter(|(g, w)| f64::from(**g) != **w)
                .count();
        }
        // `arange` rows against `arange` columns, as Burn's conv3d tests have them.
        let (m, k, n) = (8usize, 27usize, 4usize);
        let a: Vec<f32> = (0..m * k).map(|v| (v * 3 % 720) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|v| (v % 288) as f32).collect();
        let want = reference(&a, &b, [m, k, n]);
        let got = matmul(&d, ACCURATE, &a, &b, [m, k, n]);
        assert!(got.iter().zip(&want).all(|(g, w)| f64::from(*g) == *w));
        assert!(
            fast_wrong > 0,
            "the fast matmul is exact on this data: nothing was shown"
        );
    });
}

/// NaN and infinity are IEEE's, entry by entry, against the f64 oracle: a NaN reaches its row and its
/// column, an infinity times a zero is NaN, an infinity times anything else nonzero is an infinity of
/// the product's sign, both signs are NaN, and an entry with none is finite and within the bound.
#[test]
fn nan_and_infinity_follow_ieee() {
    with_device(Config::default(), |d| {
        let (mut nans, mut pos, mut neg, mut finite) = (0, 0, 0, 0);
        for seed in 0..12u64 {
            let mut rng = Rng(0x1234_5678_9abc_def1 ^ (seed + 1).wrapping_mul(0x1000_0000_01b3));
            let (m, k, n) = (12usize, 9usize, 11usize);
            // The share of special values varies with the seed, so some entries stay finite.
            let special = 3 + seed % 6;
            let draw = |rng: &mut Rng| {
                let r = rng.next() % 100;
                match r {
                    r if r < special => f32::INFINITY,
                    r if r < 2 * special => f32::NEG_INFINITY,
                    r if r < 2 * special + 2 => f32::NAN,
                    r if r < 2 * special + 12 => 0.0,
                    _ => rng.spread(4),
                }
            };
            let a: Vec<f32> = (0..m * k).map(|_| draw(&mut rng)).collect();
            let b: Vec<f32> = (0..k * n).map(|_| draw(&mut rng)).collect();
            let shape = [m, k, n];
            let (want, limit) = (reference(&a, &b, shape), bound(&a, &b, shape));
            let got = matmul(&d, ACCURATE, &a, &b, shape);
            for i in 0..m * n {
                let (g, w) = (f64::from(got[i]), want[i]);
                if w.is_nan() {
                    nans += 1;
                    assert!(g.is_nan(), "seed {seed} entry {i}: {g}, want NaN");
                } else if w.is_infinite() {
                    if w > 0.0 {
                        pos += 1
                    } else {
                        neg += 1
                    }
                    assert_eq!(g, w, "seed {seed} entry {i}");
                } else {
                    finite += 1;
                    assert!(
                        (g - w).abs() <= limit[i],
                        "seed {seed} entry {i}: {g} against {w}"
                    );
                }
            }
        }
        // Every class occurred.
        assert!(
            nans > 20 && pos > 5 && neg > 5 && finite > 100,
            "{nans} {pos} {neg} {finite}"
        );
    });
}

/// Views, batch broadcast and rank mismatch go through the same path: a transposed operand on either
/// side, a slice, `[2, 1, m, k] @ [1, 3, k, n]`, `[m, k] @ [3, k, n]` and `[3, m, k] @ [k, n]`, each
/// within the bound of its own batch's matrices.
#[test]
fn views_batches_and_ranks() {
    with_device(Config::default(), |d| {
        let (m, k, n) = (13usize, 40usize, 9usize);
        let shape = [m, k, n];
        let mut rng = Rng(0x7777_1111_3333_5555);
        let (a, b) = operands(&mut rng, shape, 8);
        let within = |got: &[f32], a: &[f32], b: &[f32], what: &str| {
            assert_eq!(
                violations(got, &reference(a, b, shape), &bound(a, b, shape)),
                0,
                "{what}"
            );
        };
        let host = |t: Tensor<TtBackend, _>| t.into_data().to_vec::<f32>().unwrap();
        with_matmul_mode(ACCURATE, || {
            // A transposed left operand and a transposed right one.
            let at: Vec<f32> = (0..k * m).map(|i| a[(i % m) * k + i / m]).collect();
            let bt: Vec<f32> = (0..n * k).map(|i| b[(i % k) * n + i / k]).collect();
            within(
                &host(
                    tensor2(&d, &at, [k, m])
                        .transpose()
                        .matmul(tensor2(&d, &b, [k, n])),
                ),
                &a,
                &b,
                "lhs^T",
            );
            within(
                &host(tensor2(&d, &a, [m, k]).matmul(tensor2(&d, &bt, [n, k]).transpose())),
                &a,
                &b,
                "rhs^T",
            );
            // A slice of larger matrices.
            let (big_a, big_b) = operands(&mut rng, [m + 3, k + 5, n + 2], 8);
            let sa = tensor2(&d, &big_a, [m + 3, k + 5]).slice([1..m + 1, 2..k + 2]);
            let sb = tensor2(&d, &big_b, [k + 5, n + 2]).slice([3..k + 3, 1..n + 1]);
            let (ca, cb): (Vec<f32>, Vec<f32>) = (
                (0..m * k)
                    .map(|i| big_a[(i / k + 1) * (k + 5) + i % k + 2])
                    .collect(),
                (0..k * n)
                    .map(|i| big_b[(i / n + 3) * (n + 2) + i % n + 1])
                    .collect(),
            );
            within(&host(sa.matmul(sb)), &ca, &cb, "slices");
            // Batches: [2, 1, m, k] @ [1, 3, k, n] gives [2, 3, m, n].
            let (ba, bb) = (
                (0..2 * m * k).map(|_| rng.spread(6)).collect::<Vec<f32>>(),
                (0..3 * k * n).map(|_| rng.spread(6)).collect::<Vec<f32>>(),
            );
            let t4a =
                Tensor::<TtBackend, 4>::from_data(TensorData::new(ba.clone(), [2, 1, m, k]), &d);
            let t4b =
                Tensor::<TtBackend, 4>::from_data(TensorData::new(bb.clone(), [1, 3, k, n]), &d);
            let got = host4(t4a.matmul(t4b));
            assert_eq!(got.len(), 2 * 3 * m * n);
            for i in 0..2 {
                for j in 0..3 {
                    within(
                        &got[(i * 3 + j) * m * n..(i * 3 + j + 1) * m * n],
                        &ba[i * m * k..(i + 1) * m * k],
                        &bb[j * k * n..(j + 1) * k * n],
                        "batch broadcast",
                    );
                }
            }
            // A matrix against a batch, and a batch against a matrix.
            let t3 = Tensor::<TtBackend, 3>::from_data(TensorData::new(bb.clone(), [3, k, n]), &d);
            let got = host3(tensor2(&d, &a, [m, k]).unsqueeze::<3>().matmul(t3));
            for j in 0..3 {
                within(
                    &got[j * m * n..(j + 1) * m * n],
                    &a,
                    &bb[j * k * n..(j + 1) * k * n],
                    "matrix @ batch",
                );
            }
            let t3a = Tensor::<TtBackend, 3>::from_data(
                TensorData::new(
                    (0..3 * m * k).map(|i| ba[i % (2 * m * k)]).collect(),
                    [3, m, k],
                ),
                &d,
            );
            let a3: Vec<f32> = (0..3 * m * k).map(|i| ba[i % (2 * m * k)]).collect();
            let got = host3(t3a.matmul(tensor2(&d, &b, [k, n]).unsqueeze::<3>()));
            for j in 0..3 {
                within(
                    &got[j * m * n..(j + 1) * m * n],
                    &a3[j * m * k..(j + 1) * m * k],
                    &b,
                    "batch @ matrix",
                );
            }
        });
    });
}

fn host3(t: Tensor<TtBackend, 3>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}
fn host4(t: Tensor<TtBackend, 4>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[cfg(feature = "silicon")]
fn round_bf16(x: f64) -> f32 {
    let bits = (x as f32).to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) << 16)
}

fn round_f16(x: f64) -> f32 {
    f32::from_bits(oracle_to_f32(oracle_to_f16((x as f32).to_bits())))
}

/// A half type's operands widen to F32 on the card, take the accurate path and round once: on operands
/// that are exact in the half type, the result is the exact sum rounded once to nearest even.
fn half_rounds_the_exact_sum_once(
    d: &TtDevice,
    name: &str,
    dtype: FloatDType,
    round: fn(f64) -> f32,
) {
    let (m, k, n) = (20usize, 64usize, 12usize);
    let mut rng = Rng(0x0bad_cafe_0bad_cafe);
    // Multiples of 1/8 in [-8, 8]: exact in both half types, products and sums exact in F32.
    let mut eighths = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| (rng.next() % 129) as f32 / 8.0 - 8.0)
            .collect()
    };
    let (a, b) = (eighths(m * k), eighths(k * n));
    let want = reference(&a, &b, [m, k, n]);
    let got = with_matmul_mode(ACCURATE, || {
        let (ta, tb) = (
            tensor2(d, &a, [m, k]).cast(dtype),
            tensor2(d, &b, [k, n]).cast(dtype),
        );
        ta.matmul(tb)
            .cast(FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    });
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(*g, round(*w), "{name} entry {i}: exact sum {w}");
    }
}

#[test]
fn f16_operands_round_the_exact_sum_once() {
    with_device(Config::default(), |d| {
        half_rounds_the_exact_sum_once(&d, "f16", FloatDType::F16, round_f16);
    });
}

/// BF16 is the same on silicon; ttsim refuses the BF16 narrowing mode (`0x105`, divergence row 29),
/// so there is no simulator run of this one.
#[cfg(feature = "silicon")]
#[test]
fn bf16_operands_round_the_exact_sum_once() {
    with_device(Config::default(), |d| {
        half_rounds_the_exact_sum_once(&d, "bf16", FloatDType::BF16, round_bf16);
    });
}

/// Gradients: `backward()` inside the mode makes the two matmuls of `x @ w` accurate, each within the
/// bound of its own operands (`dx = 1 @ w^T`, `dw = x^T @ 1`).
#[test]
fn gradients_are_accurate() {
    with_device(Config::default(), |d| {
        type Ad = Autodiff<TtBackend>;
        let (m, k, n) = (17usize, 50usize, 10usize);
        let mut rng = Rng(0x5555_aaaa_5555_aaaa);
        let (x, w) = operands(&mut rng, [m, k, n], 10);
        let (dx, dw) = with_matmul_mode(ACCURATE, || {
            let xt =
                Tensor::<Ad, 2>::from_data(TensorData::new(x.clone(), [m, k]), &d).require_grad();
            let wt =
                Tensor::<Ad, 2>::from_data(TensorData::new(w.clone(), [k, n]), &d).require_grad();
            let grads = xt.clone().matmul(wt.clone()).sum().backward();
            (
                xt.grad(&grads)
                    .unwrap()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
                wt.grad(&grads)
                    .unwrap()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            )
        });
        let ones_mn = vec![1f32; m * n];
        let w_t: Vec<f32> = (0..n * k).map(|i| w[(i % k) * n + i / k]).collect();
        let x_t: Vec<f32> = (0..k * m).map(|i| x[(i % m) * k + i / m]).collect();
        assert_eq!(
            violations(
                &dx,
                &reference(&ones_mn, &w_t, [m, n, k]),
                &bound(&ones_mn, &w_t, [m, n, k])
            ),
            0,
            "dx"
        );
        assert_eq!(
            violations(
                &dw,
                &reference(&x_t, &ones_mn, [k, m, n]),
                &bound(&x_t, &ones_mn, [k, m, n])
            ),
            0,
            "dw"
        );
    });
}

/// The domain's edge, as `ops/accurate_matmul.rs` states it: a subnormal operand is zero (the
/// SFPU's and the unit's flush), an operand past `2^64` is taken as `2^64`, and a row below `2^-64`
/// is accurate only to the scale of `2^-64` (within the bound, which says that).
#[test]
fn the_domain_edge_does_what_the_module_says() {
    with_device(Config::default(), |d| {
        let one = [1.0f32];
        let run = |a: f32, b: f32| matmul(&d, ACCURATE, &[a], &[b], [1, 1, 1])[0];
        assert_eq!(run(f32::from_bits(0x0000_1234), 3.0), 0.0, "a subnormal");
        assert_eq!(run(f32::MAX, 1.0), 2f32.powi(64), "f32::MAX");
        assert_eq!(run(-(2f32.powi(70)), 1.0), -(2f32.powi(64)), "-2^70");
        assert_eq!(run(2f32.powi(64), 1.0), 2f32.powi(64));
        assert_eq!(
            run(1.5 * 2f32.powi(-20), 1.0),
            1.5 * 2f32.powi(-20),
            "an ordinary small value"
        );
        // Below the domain: within the bound.
        let (a, b) = ([1.5 * 2f32.powi(-80)], one);
        let got = matmul(&d, ACCURATE, &a, &b, [1, 1, 1]);
        assert_eq!(
            violations(
                &got,
                &reference(&a, &b, [1, 1, 1]),
                &bound(&a, &b, [1, 1, 1])
            ),
            0
        );
        // A row of mixed scales keeps its largest entry exactly and its small ones to the row's scale.
        let (a, b) = ([1024.0f32, 1.0, 0.5, 2f32.powi(-10)], [1.0f32; 4]);
        let got = matmul(&d, ACCURATE, &a, &b, [1, 4, 1])[0];
        assert_eq!(
            f64::from(got),
            1024.0 + 1.0 + 0.5 + 2f64.powi(-10),
            "a spread of 2^20 inside one row"
        );
    });
}

fn accurate_device() -> Config {
    Config {
        accurate_matmul: true,
        ..Config::default()
    }
}

/// The mode is chosen by the attachment and by the scope, as the exact mode is: 125..249 times 1..53
/// summed over 64 is exact in F32 (every product and sum), so the truth is the oracle and the fast
/// matmul misses it (step158); a device attached accurate is accurate unless the closure says
/// otherwise, scopes nest, a panic inside one does not leave it on, and the `Fidelity::Lo` the mode
/// runs its slices at does not outlast it (a fast matmul after it gives the bits it gave before).
#[test]
fn the_mode_is_chosen_by_the_attachment_and_the_scope() {
    let (m, k) = (32usize, 64usize);
    let a: Vec<f32> = (0..m * k).map(|i| (125 + i % 125) as f32).collect();
    let b: Vec<f32> = (0..k).map(|i| (1 + i % 53) as f32).collect();
    let truth: Vec<f32> = reference(&a, &b, [m, k, 1])
        .into_iter()
        .map(|x| x as f32)
        .collect();
    let run = |d: &TtDevice| {
        tensor2(d, &a, [m, k])
            .matmul(tensor2(d, &b, [k, 1]))
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    };
    with_device(Config::default(), |d| {
        let fast = run(&d);
        assert_ne!(fast, truth, "the default is the fast path");
        assert_eq!(accurate_matmul(|| run(&d)), truth);
        assert_eq!(with_matmul_mode(ACCURATE, || run(&d)), truth);
        assert_eq!(run(&d), fast, "the scope ended, and the fidelity with it");
        accurate_matmul(|| {
            assert_eq!(
                with_matmul_mode(FAST, || run(&d)),
                fast,
                "the innermost wins"
            );
            assert_eq!(run(&d), truth);
        });
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            accurate_matmul(|| panic!("inside the scope"))
        }));
        assert!(caught.is_err());
        assert_eq!(run(&d), fast, "a panic must not leak the override");
    });
    with_device(accurate_device(), |d| {
        assert_eq!(run(&d), truth, "an accurate device is accurate");
        let fast = with_matmul_mode(FAST, || run(&d));
        assert_ne!(fast, truth);
        assert_eq!(run(&d), truth);
    });
}

/// A trace freezes the mode in effect when it was captured, and replays what it captured: new inputs
/// through the replay stay within the bound.
#[test]
fn a_trace_captured_accurate_replays_accurate() {
    with_device(accurate_device(), |d| {
        use burn::tensor::TensorPrimitive;
        let mut rng = Rng(0x5555_aaaa_1111_2222);
        let (m, k, n) = (6usize, 40usize, 3usize);
        let shape = [m, k, n];
        let (xv, wv) = operands(&mut rng, shape, 8);
        let x = tensor2(&d, &xv, [m, k]).mul_scalar(1.0);
        let TensorPrimitive::Float(input) = x.clone().into_primitive() else {
            unreachable!()
        };
        let w = tensor2(&d, &wv, [k, n]).to_device(&d);
        let (trace, first) = burn_tt::Trace::capture(&input, || {
            let TensorPrimitive::Float(out) = x.clone().matmul(w.clone()).into_primitive() else {
                unreachable!()
            };
            out
        })
        .unwrap();
        let within = |got: &[f32], xs: &[f32], what: &str| {
            assert_eq!(
                violations(got, &reference(xs, &wv, shape), &bound(xs, &wv, shape)),
                0,
                "{what}"
            );
        };
        within(&first, &xv, "the capture's own run");
        for _ in 0..2 {
            let again: Vec<f32> = (0..m * k).map(|_| rng.spread(8)).collect();
            within(&trace.run(again.clone()).unwrap(), &again, "a replay");
        }
    });
}

/// What the accurate mode costs, on silicon: host-timed medians of the fast Matrix Unit matmul and of
/// the accurate one, from the shapes of the exact mode's benchmark (`step159`, where the exact one
/// took 0.5 us a product) up to a 1024 cube, the first sixteen rows of every sample validated (the
/// accurate one within the derived bound, the fast one within a loose sanity bound that says only
/// that it is a matmul). The host time includes launching the dozens of device operations the mode
/// is built from and reading the result back.
/// `cargo xtask bench --device 0 --filter step163_accurate_matmul::benchmark`.
#[cfg(feature = "silicon")]
#[test]
#[ignore = "release benchmark"]
fn benchmark_accurate_against_fast_matmul() {
    use std::time::Instant;
    use tt_tests::bench::{report, Stats, REPS};
    with_device(Config::default(), |d| {
        let mut rng = Rng(0x5eed_0000_0000_0163);
        for [m, k, n] in [
            [64usize, 32, 64],
            [1024, 27, 32],
            [128, 256, 128],
            [256, 256, 256],
            [32, 784, 128],
            [512, 512, 512],
            [1024, 1024, 1024],
        ] {
            let a: Vec<f32> = (0..m * k).map(|_| rng.unit()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| rng.unit()).collect();
            let (lhs, rhs) = (tensor2(&d, &a, [m, k]), tensor2(&d, &b, [k, n]));
            // The first sixteen rows: the reference and the bound for them alone.
            let rows = m.min(16);
            let want = reference(&a[..rows * k], &b, [rows, k, n]);
            let limit = bound(&a[..rows * k], &b, [rows, k, n]);
            for (name, mode) in [("fast", FAST), ("accurate", ACCURATE)] {
                let mut micros = Vec::new();
                for i in 0..REPS + 2 {
                    let t = Instant::now();
                    let got = with_matmul_mode(mode, || {
                        lhs.clone()
                            .matmul(rhs.clone())
                            .into_data()
                            .to_vec::<f32>()
                            .unwrap()
                    });
                    let elapsed = t.elapsed();
                    if mode == ACCURATE {
                        assert_eq!(
                            violations(&got[..rows * n], &want, &limit),
                            0,
                            "{m}x{k}x{n}: an accurate entry is beyond its bound"
                        );
                    } else {
                        for (x, (&g, w)) in got[..rows * n].iter().zip(&want).enumerate() {
                            assert!(
                                (f64::from(g) - w).abs() <= 1e-2 * k as f64,
                                "fast, entry {x}"
                            );
                        }
                    }
                    if i >= 2 {
                        micros.push(elapsed.as_secs_f64() * 1e6);
                    }
                }
                report(
                    &format!("matmul_{name}_{m}x{k}x{n}"),
                    "us",
                    "host",
                    Stats::of(micros),
                );
            }
        }
    });
}
