//! How a tile matmul accumulates, measured: the Matrix Unit's group adder.
//!
//! Found by two of Burn's own `conv3d` tests (`burn-tt-conformance`), whose operands are exact in TF32
//! and whose sums are exact in F32, yet whose results are off by whole units. The matmul keeps about
//! 16 to 17 significant bits of a large sum. This gate holds the mechanism, fitted to the device and
//! checked against it bit for bit, and the error bound that follows from it.
//!
//! # The model (measured; `MVMUL.md` calls its own float model "a rough guide")
//!
//! One `MVMUL` is a 16-term dot product, run once per fidelity phase (four at HiFi4: the SrcB slice
//! `hi`/`lo` times the SrcA slice `hi`/`lo`). Its terms form **two groups of eight** (k = 0..7 and
//! 8..15). In a group:
//!
//! * a term's exponent is `exp(a) + exp(b)`, the sum of the *operands'* exponents (not the product's),
//!   with a low slice's exponent taken from its position (SrcB: 7 below the operand's, SrcA: 5 below)
//!   even when the slice is zero;
//! * the group is aligned once, to a quantum of `2^(max term exponent - 10)`;
//! * every product is rounded half-up on its magnitude to a multiple of that quantum, and the products
//!   are summed exactly.
//!
//! The two group sums are then added to the F32 `Dst`, one add each, to nearest with ties toward
//! +infinity. Across groups, across `MVMUL`s and across faces nothing else is lost.
//!
//! A term below the group's quantum therefore loses its low bits: `P` and `Q` in one group give
//! `P + round(Q)`, in different groups `P + Q`. Where operands are exact and the terms of a group are
//! of one size, nothing is lost, which is why small integer data (step85, step157) matches exact
//! arithmetic and `arange` data (Burn's conv3d tests) does not.
//!
//! The same quantum is the one step90 measured for `ELWADD` (10 fraction bits, half-up).
//!
//! # What the gate establishes
//!
//! The model equals the device on every output of every case here (a pair grid, random operands of
//! every slice combination, sums past 2^24, Burn's conv3d cases), where exact arithmetic does not; a
//! bound derived from it holds; and each of its rules is watched failing when changed.
//! LoFi is covered only for what `MatmulMode::Accurate` needs: integers of 7 (`SrcB`) and 5 (`SrcA`) significant bits
//! are exact (`lofi_*` below). Not covered: HiFi2 and HiFi3, BF16 `Src`, `Dst` as BF16, subnormals, NaN and infinity.

use burn::tensor::module::conv3d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData};
use burn_tt::TtBackend;
use tt_isa::numerics::{src_a_fidelity_bits, src_b_fidelity_bits};
use tt_tests::burn_device::{with_device, Config};

/// The measured rules, with the three knobs the negative controls turn.
#[derive(Clone, Copy)]
struct Rules {
    fraction_bits: i32,
    zero_slice_sets_exponent: bool,
    ties_toward_zero: bool,
}
const MEASURED: Rules = Rules {
    fraction_bits: 10,
    zero_slice_sets_exponent: true,
    ties_toward_zero: false,
};

fn exp2(x: f32) -> i32 {
    f64::from(x.abs()).log2().floor() as i32
}

/// `x` to a multiple of `q`, half-up on the magnitude (or toward zero on a tie, as a control).
fn align(x: f64, q: f64, rules: Rules) -> f64 {
    let r = x.abs() / q;
    let n = if rules.ties_toward_zero {
        (r - 0.5).ceil().max(0.0)
    } else {
        (r + 0.5).floor()
    };
    x.signum() * n * q
}

/// The group quantum of one phase, and the products, of eight terms; `None` if every term is zero.
fn group(a: &[f32], b: &[f32], phase: u32, rules: Rules) -> Option<(f64, Vec<f64>)> {
    let mut e_max = i32::MIN;
    let mut products = Vec::new();
    for k in 0..8 {
        let (x, y) = (
            src_b_fidelity_bits(a[k], phase),
            src_a_fidelity_bits(b[k], phase),
        );
        products.push(f64::from(x) * f64::from(y));
        let live = if rules.zero_slice_sets_exponent {
            a[k] != 0.0 && b[k] != 0.0
        } else {
            x != 0.0 && y != 0.0
        };
        if live {
            let ea = exp2(a[k]) - if phase & 2 != 0 { 7 } else { 0 };
            let eb = exp2(b[k]) - if phase & 1 != 0 { 5 } else { 0 };
            e_max = e_max.max(ea + eb);
        }
    }
    (e_max != i32::MIN).then(|| (2f64.powi(e_max - rules.fraction_bits), products))
}

/// The next F32 above `x` (finite, not the largest).
fn above(x: f32) -> f32 {
    if x == 0.0 {
        f32::from_bits(1)
    } else if x > 0.0 {
        f32::from_bits(x.to_bits() + 1)
    } else {
        f32::from_bits(x.to_bits() - 1)
    }
}

/// The next F32 below `x`.
fn below(x: f32) -> f32 {
    -above(-x)
}

/// Round an exact value to F32: nearest, ties toward +infinity.
fn f32_ties_up(x: f64) -> f32 {
    let near = x as f32;
    let (lo, hi) = if f64::from(near) <= x {
        (near, above(near))
    } else {
        (below(near), near)
    };
    if x - f64::from(lo) == f64::from(hi) - x {
        hi
    } else {
        near
    }
}

fn ulp(x: f32) -> f64 {
    2f64.powi(exp2(x.max(f32::MIN_POSITIVE)) - 23)
}

/// `a . b` as the tile computes it, and the error bound the rules imply (against the exact dot of the
/// operands, valid where they are exact in the slices): half a quantum per rounded term, plus an ulp of
/// `Dst` per add.
fn dot(a: &[f32], b: &[f32], rules: Rules) -> (f32, f64) {
    let k = a.len().div_ceil(16) * 16;
    let (mut a, mut b) = (a.to_vec(), b.to_vec());
    a.resize(k, 0.0);
    b.resize(k, 0.0);
    let (mut dst, mut bound) = (0f32, 0f64);
    for face in 0..k / 16 {
        for phase in 0..4 {
            for g in 0..2 {
                let at = face * 16 + g * 8;
                let sum =
                    group(&a[at..at + 8], &b[at..at + 8], phase, rules).map_or(0.0, |(q, p)| {
                        bound += p.iter().filter(|&&x| x != 0.0).count() as f64 * q / 2.0;
                        p.iter().map(|&x| align(x, q, rules)).sum::<f64>()
                    });
                let next = f32_ties_up(f64::from(dst) + sum);
                bound += ulp(next).max(ulp(dst));
                dst = next;
            }
        }
    }
    (dst, bound)
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

fn matmul(d: &burn_tt::TtDevice, rows: &[f32], b: &[f32], k: usize) -> Vec<f32> {
    let m = rows.len() / k;
    let a = Tensor::<TtBackend, 2>::from_data(TensorData::new(rows.to_vec(), [m, k]), d);
    let b = Tensor::<TtBackend, 2>::from_data(TensorData::new(b.to_vec(), [k, 1]), d);
    a.matmul(b).into_data().to_vec::<f32>().unwrap()
}

fn exact(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// Operands drawn uniformly from `1..=max` with random signs when `signed`.
fn operands(rng: &mut Rng, n: usize, max: u64, signed: bool) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let v = (1 + rng.below(max)) as f32;
            if signed && rng.below(2) == 0 {
                -v
            } else {
                v
            }
        })
        .collect()
}

/// One big term `P` and one small `Q`, at every pair of columns of two tiles. Held to an oracle that is
/// not the model: in the same group of eight `Q` rounds to `P`'s quantum (64), in any other group, face
/// or tile it is kept whole.
#[test]
fn a_small_term_is_rounded_only_by_its_own_group_of_eight() {
    with_device(Config::default(), |d| {
        let p = 127.0f32 * 1024.0; // exponent 16: a ten-bit quantum of 64
        let qs = [
            1.0f32, 3.0, 16.0, 31.0, 32.0, 33.0, 64.0, 96.0, 127.0, -32.0, -33.0, -96.0,
        ];
        let (mut rows, mut keys) = (Vec::new(), Vec::new());
        for pc in [0usize, 5, 7, 8, 12, 15, 16, 23, 31, 32, 40, 63] {
            for qc in 0..64usize {
                for &q in qs.iter().filter(|_| pc != qc) {
                    let mut r = vec![0.0f32; 64];
                    r[pc] = p;
                    r[qc] = q;
                    rows.extend(r);
                    keys.push((pc, qc, q));
                }
            }
        }
        let got = matmul(&d, &rows, &[1.0; 64], 64);
        let mut lost = 0;
        for (i, &(pc, qc, q)) in keys.iter().enumerate() {
            // groups are k / 8 within a tile; the two tiles' MVMULs are separate additions
            let same_group = pc / 8 == qc / 8;
            let want = if same_group {
                lost += 1;
                f64::from(p)
                    + (f64::from(q).signum() * (f64::from(q).abs() / 64.0 + 0.5).floor() * 64.0)
            } else {
                f64::from(p) + f64::from(q)
            };
            assert_eq!(f64::from(got[i]), want, "P at {pc}, Q = {q} at {qc}");
            assert_eq!(
                f64::from(dot(&rows[i * 64..(i + 1) * 64], &[1.0; 64], MEASURED).0),
                want,
                "the model, P at {pc}, Q = {q} at {qc}"
            );
        }
        // The control: some of the rows were rounded, and some were not.
        assert!(lost > 0 && lost < keys.len());
        assert!((0..keys.len()).any(|i| f64::from(got[i]) != f64::from(p) + f64::from(keys[i].2)));
    });
}

/// Random operands of every slice combination and sums either side of 2^24, equal to the model bit for
/// bit and within its bound, where exact arithmetic is wrong on most rows.
#[test]
fn random_operands_equal_the_model_bit_for_bit() {
    with_device(Config::default(), |d| {
        // (k, largest A, largest B, signed): A is the SrcB side (7-bit hi slice), B the SrcA side (5-bit).
        let cases = [
            (16usize, 127, 31, true), // one phase
            (16, 2047, 31, true),     // + SrcB lo
            (16, 127, 1023, true),    // + SrcA lo
            (16, 255, 255, true),     // all four phases
            (16, 2047, 1023, true),   // all four, full mantissa
            (27, 249, 53, false),     // Burn's conv3d shapes
            (33, 255, 255, true),     // a ragged face
            (64, 255, 255, true),     // four faces
            (64, 2047, 1023, false),  // sums well past 2^24: every add into Dst rounds
        ];
        let mut inexact_exact_arithmetic = 0;
        for (n, &(k, max_a, max_b, signed)) in cases.iter().enumerate() {
            let mut rng =
                Rng(0x9e37_79b9_7f4a_7c15 ^ (n as u64 + 1).wrapping_mul(0x1000_0000_01b3));
            let rows_n = 256;
            let a = operands(&mut rng, rows_n * k, max_a, signed);
            let b = operands(&mut rng, k, max_b, signed);
            let got = matmul(&d, &a, &b, k);
            for r in 0..rows_n {
                let (row, (want, bound)) = (
                    &a[r * k..(r + 1) * k],
                    dot(&a[r * k..(r + 1) * k], &b, MEASURED),
                );
                assert_eq!(got[r], want, "k {k}, A <= {max_a}, B <= {max_b}, row {r}");
                let e = exact(row, &b);
                assert!(
                    (f64::from(got[r]) - e).abs() <= bound,
                    "k {k} row {r}: error {} over the bound {bound}",
                    f64::from(got[r]) - e
                );
                inexact_exact_arithmetic += (e as f32 != got[r]) as usize;
            }
        }
        // The control: the effect is large here, so equality with the model is a real finding.
        assert!(inexact_exact_arithmetic > 256 * 5);
    });
}

/// Burn's `conv3d` tests, whose outputs miss the vendored 1e-5: the device equals the model, with and
/// without the bias, on every output; exact arithmetic is wrong where they fail and nowhere in the case
/// that passes (`simple`, whose sums stay below 2^16).
#[test]
fn burn_conv3d_cases_are_this_rounding() {
    type Case = (
        &'static str,
        usize,
        usize,
        usize,
        [usize; 3],
        [usize; 3],
        [usize; 3],
        [usize; 3],
        usize,
        [usize; 3],
    );
    with_device(Config::default(), |d| {
        let cases: [Case; 3] = [
            (
                "groups",
                1,
                2,
                2,
                [3, 3, 3],
                [0; 3],
                [1; 3],
                [1; 3],
                2,
                [5, 5, 5],
            ),
            (
                "complex",
                2,
                3,
                4,
                [4, 3, 2],
                [1, 2, 3],
                [2, 3, 4],
                [1, 2, 3],
                1,
                [4, 5, 6],
            ),
            (
                "simple",
                1,
                2,
                2,
                [3, 3, 3],
                [1; 3],
                [1; 3],
                [1; 3],
                1,
                [4, 4, 4],
            ),
        ];
        for (name, n, ci, co, k, p, s, dl, g, sz) in cases {
            let x: Vec<f32> = (0..n * ci * sz[0] * sz[1] * sz[2])
                .map(|v| v as f32)
                .collect();
            let w: Vec<f32> = (0..co * (ci / g) * k[0] * k[1] * k[2])
                .map(|v| v as f32)
                .collect();
            let bias: Vec<f32> = (0..co).map(|v| v as f32).collect();
            let out: Vec<usize> = (0..3)
                .map(|i| (sz[i] + 2 * p[i] - dl[i] * (k[i] - 1) - 1) / s[i] + 1)
                .collect();
            let xt = Tensor::<TtBackend, 5>::from_data(
                TensorData::new(x.clone(), [n, ci, sz[0], sz[1], sz[2]]),
                &d,
            );
            let wt = Tensor::<TtBackend, 5>::from_data(
                TensorData::new(w.clone(), [co, ci / g, k[0], k[1], k[2]]),
                &d,
            );
            let bt = Tensor::<TtBackend, 1>::from_data(TensorData::new(bias.clone(), [co]), &d);
            let opts = || ConvOptions::new(s, p, dl, g);
            let plain = conv3d(xt.clone(), wt.clone(), None, opts())
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let biased = conv3d(xt, wt, Some(bt), opts())
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let (mut i, mut inexact) = (0, 0);
            for nn in 0..n {
                for oc in 0..co {
                    for od in 0..out[0] {
                        for oh in 0..out[1] {
                            for ow in 0..out[2] {
                                let grp = oc / (co / g);
                                let (mut row, mut col) = (Vec::new(), Vec::new());
                                for ic in 0..ci / g {
                                    for a in 0..k[0] {
                                        for bb in 0..k[1] {
                                            for c in 0..k[2] {
                                                let at = [(od, a, 0), (oh, bb, 1), (ow, c, 2)].map(
                                                    |(o, kk, ax)| {
                                                        (o * s[ax] + kk * dl[ax]) as isize
                                                            - p[ax] as isize
                                                    },
                                                );
                                                let inside = (0..3).all(|ax| {
                                                    at[ax] >= 0 && at[ax] < sz[ax] as isize
                                                });
                                                col.push(
                                                    w[(((oc * (ci / g) + ic) * k[0] + a) * k[1]
                                                        + bb)
                                                        * k[2]
                                                        + c],
                                                );
                                                row.push(if inside {
                                                    x[(((nn * ci + grp * (ci / g) + ic) * sz[0]
                                                        + at[0] as usize)
                                                        * sz[1]
                                                        + at[1] as usize)
                                                        * sz[2]
                                                        + at[2] as usize]
                                                } else {
                                                    0.0
                                                });
                                            }
                                        }
                                    }
                                }
                                let (want, bound) = dot(&row, &col, MEASURED);
                                assert_eq!(plain[i], want, "{name} output {i}");
                                assert_eq!(
                                    biased[i],
                                    want + bias[oc],
                                    "{name} output {i} with the bias"
                                );
                                assert!(
                                    (f64::from(plain[i]) - exact(&row, &col)).abs() <= bound,
                                    "{name} output {i}: over the bound"
                                );
                                inexact += (exact(&row, &col) as f32 != plain[i]) as usize;
                                i += 1;
                            }
                        }
                    }
                }
            }
            match name {
                "simple" => assert_eq!(inexact, 0),
                _ => assert!(inexact > 0, "{name}: exact arithmetic would have passed"),
            }
        }
    });
}

/// Each rule is watched failing: the same random operands, the model with one rule changed.
#[test]
fn each_rule_of_the_model_is_needed() {
    with_device(Config::default(), |d| {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ 0x1000_0000_01b3);
        let (k, rows_n) = (16usize, 2048);
        // Full-mantissa operands exercise every rule: all four phases, ties, and zero slices.
        let a = operands(&mut rng, rows_n * k, 2047, true);
        let b = operands(&mut rng, k, 1023, true);
        let got = matmul(&d, &a, &b, k);
        let differs = |rules: Rules| {
            (0..rows_n)
                .filter(|&r| dot(&a[r * k..(r + 1) * k], &b, rules).0 != got[r])
                .count()
        };
        assert_eq!(differs(MEASURED), 0);
        for (what, rules) in [
            (
                "nine fraction bits",
                Rules {
                    fraction_bits: 9,
                    ..MEASURED
                },
            ),
            (
                "eleven fraction bits",
                Rules {
                    fraction_bits: 11,
                    ..MEASURED
                },
            ),
            (
                "a zero slice sets no exponent",
                Rules {
                    zero_slice_sets_exponent: false,
                    ..MEASURED
                },
            ),
            (
                "ties toward zero",
                Rules {
                    ties_toward_zero: true,
                    ..MEASURED
                },
            ),
        ] {
            assert!(differs(rules) > 0, "{what}: the gate cannot tell");
        }
    });
}

/// The group alignment is an error the older derived bounds do not charge. `step68` and `step85`
/// allow `2^-9 sum|ab|` for TF32 truncation plus a few F32 additions; here every operand is exact in
/// TF32 and every sum is exact in F32, yet seven terms of 0.5 beside a 1024 in one group each round
/// up to the quantum of 1 (half-up), an error of 3.5 on 1027.5, which is `2^-8.2`.
#[test]
fn the_alignment_error_can_exceed_the_bound_the_older_gates_charge() {
    with_device(Config::default(), |d| {
        let mut row = vec![0.0f32; 16];
        row[0] = 1024.0;
        row[1..8].fill(0.5);
        let got = matmul(&d, &row, &[1.0; 16], 16)[0];
        assert_eq!(f64::from(got), 1031.0);
        assert_eq!(dot(&row, &[1.0; 16], MEASURED).0, got);
        let (exact, charged) = (1027.5f64, 2f64.powi(-9) * 1027.5);
        assert_eq!(f64::from(got) - exact, 3.5);
        assert!(3.5 > charged, "{charged}");
        // In another group the 0.5s are kept whole.
        let mut apart = row.clone();
        apart.swap(0, 8);
        assert_eq!(matmul(&d, &apart, &[1.0; 16], 16)[0], 1027.5);
    });
}

/// Integers of mixed magnitude: each has a random bit length up to `bits`, so one group of eight holds
/// terms whose exponents differ by up to `bits - 1`, and zeros occur. Signed.
fn mixed(rng: &mut Rng, n: usize, bits: u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let width = 1 + rng.below(u64::from(bits));
            let v = rng.below(1 << width) as f32;
            if rng.below(2) == 0 {
                -v
            } else {
                v
            }
        })
        .collect()
}

/// The shared-exponent slicing the accurate matmul is built on: at `Fidelity::Lo` (one phase, the SrcB
/// slice of 7 significant bits times the SrcA slice of 5) a matmul of integers `|a| <= 127`, `|b| <= 31`
/// is **exact**. Every term is then an integer, the largest operand-exponent sum is 6 + 4 = 10, so the
/// group quantum `2^(10 - 10)` is 1 and the alignment rounds nothing; every sum stays below 2^24, so
/// `Dst` loses nothing either. That is `Rules`'s quantum rule, run in the one place it had not been:
/// LoFi, with the exponent spread inside a group at its widest.
#[test]
fn lofi_is_exact_on_seven_by_five_bit_integers() {
    with_device(
        Config {
            fidelity: burn_tt::Fidelity::Lo,
            ..Config::default()
        },
        |d| {
            // (k, rows): ragged faces, one tile, and far more than one K-block (4096 x 127 x 31 < 2^24).
            for (n, &(k, rows_n)) in [
                (16usize, 128usize),
                (27, 128),
                (33, 128),
                (64, 128),
                (1024, 32),
                (4096, 32),
            ]
            .iter()
            .enumerate()
            {
                let mut rng =
                    Rng(0x2545_f491_4f6c_dd1d ^ (n as u64 + 1).wrapping_mul(0x1000_0000_01b3));
                let mut a = mixed(&mut rng, rows_n * k, 7);
                let mut b = mixed(&mut rng, k, 5);
                // The worst group: the largest product beside the smallest, in the first rows.
                for r in 0..4.min(rows_n) {
                    a[r * k] = 127.0;
                    b[0] = 31.0;
                    a[r * k + 1] = 1.0;
                    b[1] = 1.0;
                }
                // Full-size operands of one sign in the last rows: the largest sum the tests above allow.
                for r in rows_n - 2..rows_n {
                    a[r * k..(r + 1) * k].fill(127.0);
                }
                if rows_n > 2 {
                    b[2..].fill(31.0);
                }
                let got = matmul(&d, &a, &b, k);
                for r in 0..rows_n {
                    let e = exact(&a[r * k..(r + 1) * k], &b);
                    assert_eq!(f64::from(got[r]), e, "k {k}, row {r}");
                }
            }
        },
    );
}

/// The controls for the test above, each past the slice: an eighth bit in A and a sixth in B are cut by
/// LoFi's masks (7 and 5 bits), so the same check must fail where the operand has the bit.
#[test]
fn lofi_loses_a_bit_past_the_slice() {
    with_device(
        Config {
            fidelity: burn_tt::Fidelity::Lo,
            ..Config::default()
        },
        |d| {
            let k = 64;
            let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ 0x2545_f491_4f6c_dd1d);
            for (bits_a, bits_b) in [(8u32, 5u32), (7, 6)] {
                let a = mixed(&mut rng, 256 * k, bits_a);
                let b = mixed(&mut rng, k, bits_b);
                let got = matmul(&d, &a, &b, k);
                let wrong = (0..256)
                    .filter(|&r| f64::from(got[r]) != exact(&a[r * k..(r + 1) * k], &b))
                    .count();
                assert!(
                    wrong > 128,
                    "A {bits_a} bits, B {bits_b} bits: only {wrong} of 256 rows differ"
                );
            }
        },
    );
}
