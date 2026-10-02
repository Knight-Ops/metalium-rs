//! Phase 10 gate (S1, Burn routing): every element-wise method `burn-tt`
//! overrides, on a device-resident tensor, against `burn-flex` bit for bit,
//! and downloading nothing.
//!
//! The ops run where the session puts them -- the SFPU or the data mover,
//! whichever is cheaper for the size (`tt_kernels::tensor::sfpu_is_cheaper`),
//! both bit-identical (`step19_eltwise`) -- so the claim here is the routing:
//! a resident operand stays resident through the op, the result is the
//! device's, and the scalar is converted as Flex converts it.

use burn::tensor::{activation, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend};

/// Was `t` computed on the device (`TtTensor::computed_on_device`)?
fn computed_on_device(t: &Tensor<TtBackend, 2>) -> bool {
    match t.clone().into_primitive() {
        burn::tensor::TensorPrimitive::Float(p) => p.computed_on_device(),
        _ => false,
    }
}
use tt_tests::burn_device::{with_device, Config};

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            match i % 17 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::INFINITY,
                _ => ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0,
            }
        })
        .collect()
}

fn bits<const D: usize, B: burn::tensor::backend::Backend>(t: Tensor<B, D>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

#[test]
fn every_overridden_element_wise_method_matches_flex_and_stays_resident() {
    with_device(Config::default(), |d| {
        // Ragged, and big enough that the SFPU takes some of them.
        for [r, c] in [[37, 70], [512, 128]] {
            let (av, bv, rowv) = (values(1, r * c), values(2, r * c), values(3, c));
            let ta = |v: &[f32], s: [usize; 2]| {
                Tensor::<TtBackend, 2>::from_data(TensorData::new(v.to_vec(), s), &d).to_device(&d)
            };
            let fl = |v: &[f32], s: [usize; 2]| {
                Tensor::<Flex, 2>::from_data(TensorData::new(v.to_vec(), s), &FlexDevice)
            };
            let (a, b, row) = (ta(&av, [r, c]), ta(&bv, [r, c]), ta(&rowv, [1, c]));
            let (fa, fb, frow) = (fl(&av, [r, c]), fl(&bv, [r, c]), fl(&rowv, [1, c]));
            let colv = values(6, r);
            let (col, fcol) = (ta(&colv, [r, 1]), fl(&colv, [r, 1]));
            type Case = (
                &'static str,
                Box<dyn Fn() -> Tensor<TtBackend, 2>>,
                Vec<u32>,
            );
            let cases: Vec<Case> = vec![
                (
                    "add",
                    Box::new({
                        let (a, b) = (a.clone(), b.clone());
                        move || a.clone() + b.clone()
                    }),
                    bits(fa.clone() + fb.clone()),
                ),
                (
                    "sub",
                    Box::new({
                        let (a, b) = (a.clone(), b.clone());
                        move || a.clone() - b.clone()
                    }),
                    bits(fa.clone() - fb.clone()),
                ),
                (
                    "mul",
                    Box::new({
                        let (a, b) = (a.clone(), b.clone());
                        move || a.clone() * b.clone()
                    }),
                    bits(fa.clone() * fb.clone()),
                ),
                (
                    "add a row",
                    Box::new({
                        let (a, row) = (a.clone(), row.clone());
                        move || a.clone() + row.clone()
                    }),
                    bits(fa.clone() + frow.clone()),
                ),
                (
                    "subtract a row",
                    Box::new({
                        let (a, row) = (a.clone(), row.clone());
                        move || a.clone() - row.clone()
                    }),
                    bits(fa.clone() - frow.clone()),
                ),
                (
                    "a row times (on the left)",
                    Box::new({
                        let (a, row) = (a.clone(), row.clone());
                        move || row.clone() * a.clone()
                    }),
                    bits(frow.clone() * fa.clone()),
                ),
                (
                    "subtract a column",
                    Box::new({
                        let (a, col) = (a.clone(), col.clone());
                        move || a.clone() - col.clone()
                    }),
                    bits(fa.clone() - fcol.clone()),
                ),
                (
                    "a column plus (on the left)",
                    Box::new({
                        let (a, col) = (a.clone(), col.clone());
                        move || col.clone() + a.clone()
                    }),
                    bits(fcol.clone() + fa.clone()),
                ),
                (
                    "mul_scalar",
                    Box::new({
                        let a = a.clone();
                        move || a.clone() * 0.37
                    }),
                    bits(fa.clone() * 0.37),
                ),
                (
                    "add_scalar",
                    Box::new({
                        let a = a.clone();
                        move || a.clone() + 1.5
                    }),
                    bits(fa.clone() + 1.5),
                ),
                (
                    "sub_scalar",
                    Box::new({
                        let a = a.clone();
                        move || a.clone() - 0.25
                    }),
                    bits(fa.clone() - 0.25),
                ),
                (
                    "relu",
                    Box::new({
                        let a = a.clone();
                        move || activation::relu(a.clone())
                    }),
                    bits(activation::relu(fa.clone())),
                ),
            ];
            for (name, op, want) in cases {
                let before = tensor_traffic();
                let out = op();
                let during = tensor_traffic() - before;
                assert_eq!(
                    during.downloads, 0,
                    "[{r}, {c}] {name}: downloaded {during:?}"
                );
                assert_eq!(during.uploads, 0, "[{r}, {c}] {name}: uploaded {during:?}");
                let got = bits(out);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let wf = f32::from_bits(*w);
                    if wf.is_nan() {
                        assert!(
                            f32::from_bits(*g).is_nan(),
                            "[{r}, {c}] {name}: element {i}"
                        );
                    } else {
                        assert_eq!(
                            g, w,
                            "[{r}, {c}] {name}: element {i}: {g:#010x} vs {w:#010x}"
                        );
                    }
                }
            }
        }
    });
}

/// The SFPU's approximations through Burn: within the one ulp their gates
/// derive (`step28_division`), resident throughout.
#[test]
fn division_through_burn_is_within_one_ulp_and_stays_resident() {
    with_device(Config::default(), |d| {
        // Eight tiles: the size from which these approximations run on the
        // device (`burn-tt`'s `APPROX_MIN_TILES`).
        let [r, c] = [64, 128];
        let (av, bv) = (values(4, r * c), values(5, r * c));
        let ta = |v: &[f32]| {
            Tensor::<TtBackend, 2>::from_data(TensorData::new(v.to_vec(), [r, c]), &d).to_device(&d)
        };
        let fl = |v: &[f32]| {
            Tensor::<Flex, 2>::from_data(TensorData::new(v.to_vec(), [r, c]), &FlexDevice)
        };
        let (a, b, fa, fb) = (ta(&av), ta(&bv), fl(&av), fl(&bv));
        // Positive operands for `log`, uploaded as they are: an `abs` first
        // would run on the host copy, and `log` after it too.
        let posv: Vec<f32> = av.iter().map(|x| x.abs()).collect();
        let (pos, fpos) = (ta(&posv), fl(&posv));
        type Case = (
            &'static str,
            Box<dyn Fn() -> Tensor<TtBackend, 2>>,
            Vec<u32>,
        );
        let cases: Vec<Case> = vec![
            (
                "div",
                Box::new({
                    let (a, b) = (a.clone(), b.clone());
                    move || a.clone() / b.clone()
                }),
                bits(fa.clone() / fb.clone()),
            ),
            (
                "recip",
                Box::new({
                    let a = a.clone();
                    move || a.clone().recip()
                }),
                bits(fa.clone().recip()),
            ),
            (
                "div_scalar",
                Box::new({
                    let a = a.clone();
                    move || a.clone() / 0.3
                }),
                bits(fa.clone() / 0.3),
            ),
            (
                "exp",
                Box::new({
                    let a = a.clone();
                    move || a.clone().exp()
                }),
                bits(fa.clone().exp()),
            ),
            (
                "log",
                Box::new({
                    let p = pos.clone();
                    move || p.clone().log()
                }),
                bits(fpos.clone().log()),
            ),
        ];
        for (name, op, want) in cases {
            let before = tensor_traffic();
            let out = op();
            let during = tensor_traffic() - before;
            assert!(
                computed_on_device(&out),
                "{name}: the result was computed on the host"
            );
            assert_eq!(
                (during.downloads, during.uploads),
                (0, 0),
                "{name}: {during:?}"
            );
            for (i, (g, w)) in bits(out).iter().zip(&want).enumerate() {
                let (gf, wf) = (f32::from_bits(*g), f32::from_bits(*w));
                if wf.is_nan() {
                    assert!(gf.is_nan(), "{name}: element {i}");
                } else if wf.is_infinite() || wf == 0.0 || wf.abs() < f32::MIN_POSITIVE {
                    assert_eq!(
                        gf.abs(),
                        if wf.abs() < f32::MIN_POSITIVE {
                            0.0
                        } else {
                            wf.abs()
                        },
                        "{name}: element {i}"
                    );
                } else if name == "exp" || name == "log" {
                    // Their derived relative bounds, plus Flex's own ulp.
                    let bound = if name == "exp" {
                        tt_kernels::sfpu::ops::EXP_BOUND
                    } else {
                        tt_kernels::sfpu::ops::LOG_BOUND
                    };
                    let rel = (gf as f64 - wf as f64).abs() / (wf as f64).abs();
                    assert!(
                        rel <= bound + 1.2e-7,
                        "{name}: element {i}: {gf:e} vs {wf:e}"
                    );
                } else {
                    let ulps = (*g as i64 - *w as i64).unsigned_abs();
                    assert!(
                        ulps <= 1,
                        "{name}: element {i}: {gf:e} vs {wf:e}, {ulps} ulps"
                    );
                }
            }
        }
    });
}
