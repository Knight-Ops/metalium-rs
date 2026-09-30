//! Phase 7, step 11: `burn-tt`'s matmul on the device, against `burn-flex`.
//!
//! Every other op is Flex's own (`burn-tt/tests/delegation.rs` checks that
//! they are byte-identical); this is the one that is not, so this is where the
//! two can disagree, and the question is by how much.
//!
//! **Where the arithmetic is exact, bit for bit.** Small-integer operands make
//! every product and every partial sum exact in FP32, so any summation order --
//! the device's, Flex's, a split `K`'s -- gives the same bits.
//!
//! **Elsewhere, within a bound derived from the formats, never an epsilon.**
//! For `C = A @ B` with inner dimension `k`, each element of the device's
//! answer is within
//!
//! ```text
//! |C_tt - C_flex| <= S * (2^-9 + 5k * 2^-23 + k * 2^-24),   S = sum_q |a_iq| |b_qj|
//! ```
//!
//! of Flex's, where the terms are, in order:
//!
//! * **TF32 operands.** The unpacker truncates each FP32 operand to TF32's ten
//!   mantissa bits (`tile::fp32_to_tf32`, gated in `probe_src`), so each loses
//!   less than `2^-10` of itself and a product less than `2^-10 + 2^-10 +
//!   2^-20 < 2^-9` of `|a||b|`.
//! * **The device's accumulation.** HiFi4 forms each product exactly from the
//!   `Src` values (`step10_matmul_tile::a_32x32_tile_at_hifi4_recovers_the_exact_product`)
//!   but adds it to `Dst` once per phase, so `4k` additions, plus up to `k`
//!   more on the host when `K` is split (`matmul::matmul_chunked`). `Dst`'s
//!   rounding mode is not documented for Blackhole, so each addition is charged
//!   the worst case, truncation: `2^-23` of the running magnitude, which `S`
//!   bounds.
//! * **Flex's accumulation.** `k` round-to-nearest additions, `2^-24` each.
//!
//! The control, which the bound must reject, is the same product at
//! `Fidelity::Lo`: phase 0 alone keeps about seven bits of one operand and five
//! of the other.

use burn_flex::{Flex, FlexDevice};
use burn_tensor::{backend::Backend, Tensor, TensorData};
use burn_tt::{Fidelity, TtBackend};
use tt_tests::burn_device::{with_device, Config};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn int(&mut self, bound: i64) -> f32 {
        (self.next() as i64 % (2 * bound + 1) - bound) as f32
    }
    /// Uniform in `[-1, 1)`, with every mantissa bit in play.
    fn unit(&mut self) -> f32 {
        (self.next() as f64 / (1u64 << 30) as f64 - 1.0) as f32
    }
}

fn ints(rng: &mut Lcg, shape: &[usize], bound: i64) -> TensorData {
    let n: usize = shape.iter().product();
    TensorData::new(
        (0..n).map(|_| rng.int(bound)).collect::<Vec<f32>>(),
        shape.to_vec(),
    )
}

fn units(rng: &mut Lcg, shape: &[usize]) -> TensorData {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
    // A generator that only ever returned [-1, 0) once made these gates
    // weaker than they read -- no cancellation anywhere -- and killed every
    // ReLU in step 12.
    assert!(
        n < 16 || (v.iter().any(|x| *x > 0.5) && v.iter().any(|x| *x < -0.5)),
        "the generator must cover [-1, 1)"
    );
    TensorData::new(v, shape.to_vec())
}

fn values(d: TensorData) -> Vec<f32> {
    d.to_vec::<f32>().unwrap()
}

/// `a @ b` on backend `B`, for rank-`D` operands.
fn product<B: Backend, const D: usize>(
    a: &TensorData,
    b: &TensorData,
    device: &B::Device,
) -> TensorData {
    Tensor::<B, D>::from_data(a.clone(), device)
        .matmul(Tensor::<B, D>::from_data(b.clone(), device))
        .into_data()
}

/// The module documentation's bound, per element of `A[m, k] @ B[k, n]`.
fn bound(a: &[f32], b: &[f32], [m, k, n]: [usize; 3]) -> Vec<f64> {
    let rel = 2f64.powi(-9) + 5.0 * k as f64 * 2f64.powi(-23) + k as f64 * 2f64.powi(-24);
    (0..m * n)
        .map(|x| {
            let s: f64 = (0..k)
                .map(|q| f64::from(a[x / n * k + q].abs()) * f64::from(b[q * n + x % n].abs()))
                .sum();
            s * rel
        })
        .collect()
}

/// How many elements of `got` fall outside `bound` of `want`, and the worst
/// excess as a multiple of the bound.
fn outside(got: &[f32], want: &[f32], bound: &[f64]) -> (usize, f64) {
    let mut n = 0;
    let mut worst = 0f64;
    for ((g, w), b) in got.iter().zip(want).zip(bound) {
        let e = (f64::from(*g) - f64::from(*w)).abs();
        if e > *b {
            n += 1;
        }
        worst = worst.max(e / b);
    }
    (n, worst)
}

/// Small-integer operands, so every order of summation is exact: the device's
/// answer is Flex's, bit for bit, across plain, padded, batched, broadcast and
/// transposed shapes.
#[test]
fn small_integer_matmuls_are_bit_exact_against_flex() {
    with_device(Config::default(), |d| {
        let mut rng = Lcg(0xb11);
        let cases2: [([usize; 2], [usize; 2]); 3] = [
            ([32, 32], [32, 32]),
            ([13, 47], [47, 29]),
            ([1, 64], [64, 96]),
        ];
        for (sa, sb) in cases2 {
            let (a, b) = (ints(&mut rng, &sa, 7), ints(&mut rng, &sb, 7));
            let want = product::<Flex, 2>(&a, &b, &FlexDevice);
            let got = product::<TtBackend, 2>(&a, &b, &d);
            assert_eq!(want.shape, got.shape, "{sa:?} @ {sb:?}");
            assert_eq!(want.as_bytes(), got.as_bytes(), "{sa:?} @ {sb:?}");
        }
        let cases4: [([usize; 4], [usize; 4]); 2] = [
            ([1, 3, 20, 40], [1, 3, 40, 33]),
            ([2, 1, 17, 32], [1, 3, 32, 20]),
        ];
        for (sa, sb) in cases4 {
            let (a, b) = (ints(&mut rng, &sa, 7), ints(&mut rng, &sb, 7));
            let want = product::<Flex, 4>(&a, &b, &FlexDevice);
            let got = product::<TtBackend, 4>(&a, &b, &d);
            assert_eq!(want.shape, got.shape, "{sa:?} @ {sb:?}");
            assert_eq!(want.as_bytes(), got.as_bytes(), "{sa:?} @ {sb:?}");
        }
        // A transposed right operand: a strided view in Flex.
        let (a, b) = (ints(&mut rng, &[24, 40], 7), ints(&mut rng, &[30, 40], 7));
        let want = Tensor::<Flex, 2>::from_data(a.clone(), &FlexDevice)
            .matmul(Tensor::<Flex, 2>::from_data(b.clone(), &FlexDevice).transpose())
            .into_data();
        let got = Tensor::<TtBackend, 2>::from_data(a, &d)
            .matmul(Tensor::<TtBackend, 2>::from_data(b, &d).transpose())
            .into_data();
        assert_eq!(want.as_bytes(), got.as_bytes(), "A @ B^T");
    });
}

/// Random floats, where the orders disagree: every element within the derived
/// bound at HiFi4, including a shape large enough to split `K`.
#[test]
fn random_matmuls_are_within_the_derived_bound() {
    for (sa, sb) in [([40, 70], [70, 33]), ([64, 784], [784, 128])] {
        let mut rng = Lcg(0xf10a7 + sa[1] as u64);
        let (a, b) = (units(&mut rng, &sa), units(&mut rng, &sb));
        let want = values(product::<Flex, 2>(&a, &b, &FlexDevice));
        let limit = bound(
            &values(a.clone()),
            &values(b.clone()),
            [sa[0], sa[1], sb[1]],
        );
        with_device(Config::default(), |d| {
            let got = values(product::<TtBackend, 2>(&a, &b, &d));
            let (n, worst) = outside(&got, &want, &limit);
            eprintln!("{sa:?} @ {sb:?} HiFi4: worst error {worst:.3} of the bound");
            assert_eq!(
                n, 0,
                "{sa:?} @ {sb:?}: {n} elements outside the bound, worst {worst:.2}x"
            );
        });
    }
}

/// The control: the bound is tight enough to see the precision HiFi4 buys.
/// At `Fidelity::Lo` the same product falls outside it.
#[test]
fn at_lo_fidelity_the_same_product_breaks_the_bound() {
    let (sa, sb) = ([40, 70], [70, 33]);
    let mut rng = Lcg(0xf10a7 + sa[1] as u64);
    let (a, b) = (units(&mut rng, &sa), units(&mut rng, &sb));
    let want = values(product::<Flex, 2>(&a, &b, &FlexDevice));
    let limit = bound(&values(a.clone()), &values(b.clone()), [40, 70, 33]);
    let lo = Config {
        fidelity: Fidelity::Lo,
        ..Config::default()
    };
    with_device(lo, |d| {
        let got = values(product::<TtBackend, 2>(&a, &b, &d));
        let (n, worst) = outside(&got, &want, &limit);
        eprintln!(
            "Lo: {n} of {} outside, worst {worst:.1}x the bound",
            got.len()
        );
        assert!(
            n > got.len() / 2,
            "Lo must break the bound broadly, broke {n}"
        );
    });
}

/// Through autodiff: `y = x @ w + b`, `loss = sum(y * g)` for a fixed `g`, so
/// `dL/dw = x^T @ g` and `dL/dx = g @ w^T` -- two more matmuls, both on the
/// device -- and each gradient within its own product's bound of
/// `Autodiff<Flex>`'s. The bias gradient is `sum_rows(g)` on both backends,
/// computed by Flex both times, and identical.
#[test]
fn a_linear_layers_gradients_are_within_the_bound() {
    use burn::backend::Autodiff;
    let [m, k, n] = [48, 100, 40];
    let mut rng = Lcg(0x11ea);
    let x = units(&mut rng, &[m, k]);
    let w = units(&mut rng, &[k, n]);
    let bias = units(&mut rng, &[n]);
    let g = units(&mut rng, &[m, n]);

    fn grads<B: burn_tensor::backend::AutodiffBackend>(
        x: &TensorData,
        w: &TensorData,
        bias: &TensorData,
        g: &TensorData,
        device: &B::Device,
    ) -> [Vec<f32>; 3] {
        let x = Tensor::<B, 2>::from_data(x.clone(), device).require_grad();
        let w = Tensor::<B, 2>::from_data(w.clone(), device).require_grad();
        let bias = Tensor::<B, 1>::from_data(bias.clone(), device).require_grad();
        let g = Tensor::<B, 2>::from_data(g.clone(), device);
        let y = x.clone().matmul(w.clone()) + bias.clone().unsqueeze();
        let grads = (y * g).sum().backward();
        [
            values(x.grad(&grads).unwrap().into_data()),
            values(w.grad(&grads).unwrap().into_data()),
            values(bias.grad(&grads).unwrap().into_data()),
        ]
    }
    let [dx0, dw0, db0] = grads::<Autodiff<Flex>>(&x, &w, &bias, &g, &FlexDevice);

    let t = |d: &TensorData, r: usize, c: usize| -> Vec<f32> {
        let v = values(d.clone());
        (0..r * c).map(|i| v[(i % r) * c + i / r]).collect()
    };
    // dL/dx = g @ w^T: [m, n] @ [n, k].
    let dx_bound = bound(&values(g.clone()), &t(&w, k, n), [m, n, k]);
    // dL/dw = x^T @ g: [k, m] @ [m, n].
    let dw_bound = bound(&t(&x, m, k), &values(g.clone()), [k, m, n]);

    with_device(Config::default(), |d| {
        let [dx, dw, db] = grads::<Autodiff<TtBackend>>(&x, &w, &bias, &g, &d);
        let (n_dx, worst_dx) = outside(&dx, &dx0, &dx_bound);
        let (n_dw, worst_dw) = outside(&dw, &dw0, &dw_bound);
        eprintln!("dx worst {worst_dx:.3}, dw worst {worst_dw:.3} of the bound");
        assert_eq!(n_dx, 0, "dL/dx: {n_dx} outside, worst {worst_dx:.2}x");
        assert_eq!(n_dw, 0, "dL/dw: {n_dw} outside, worst {worst_dw:.2}x");
        assert_eq!(db, db0, "the bias gradient never touches the device");
        // The device was used: exact agreement on random floats would mean the
        // matmuls ran on the host.
        assert_ne!(dw, dw0, "dL/dw must come from the device");
    });
}
