//! Phase 10 gate (milestone 10.2, Burn routing): what 10.2 puts on the device,
//! through Burn, against `burn-flex`, on resident tensors, downloading
//! nothing. In the silicon smoke tier (`xtask/src/silicon.rs`, `SMOKE`).
//!
//! - D3: `Int` (`i32`) and `Bool` tensors live on the card. `to_device`
//!   uploads them; reshapes that keep the stored matrix, whole-tile-row
//!   slices and transposes are views; the logic ops run there.

use burn::tensor::{Bool, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{with_device, Config};

/// Was `p` computed on the device: a device copy and no host one?
fn on_device(p: &burn_tt::TtTensor) -> bool {
    p.computed_on_device()
}

fn ints(seed: u64, n: usize) -> Vec<i32> {
    let mut s = seed | 1;
    let specials = [0, 1, -1, i32::MIN, i32::MAX, 0x7f80_0001, -0x0080_0000];
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if i % 11 == 0 {
                specials[(i / 11) % specials.len()]
            } else {
                (s >> 32) as i32 >> (s % 31)
            }
        })
        .collect()
}

fn bools(seed: u64, n: usize) -> Vec<bool> {
    ints(seed, n)
        .iter()
        .map(|x| x.count_ones() % 2 == 1)
        .collect()
}

/// `f` moves nothing across PCIe.
fn resident<T>(what: &str, f: impl FnOnce() -> T) -> T {
    let before = tensor_traffic();
    let out = f();
    let during = tensor_traffic() - before;
    assert_eq!(
        (during.uploads, during.downloads),
        (0, 0),
        "{what}: {during:?}"
    );
    out
}

#[test]
fn integers_and_booleans_stay_on_the_card() {
    with_device(Config::default(), |d| {
        for [r, c] in [[37, 70], [96, 64]] {
            let iv = ints(1, r * c);
            let ti =
                Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(iv.clone(), [r, c]), &d)
                    .to_device(&d);
            let fi =
                Tensor::<Flex, 2, Int>::from_data(TensorData::new(iv.clone(), [r, c]), &FlexDevice);
            let int_eq = |t: Tensor<TtBackend, 2, Int>, f: Tensor<Flex, 2, Int>, what: &str| {
                assert!(
                    on_device(&t.clone().into_primitive()),
                    "{what}: not on the device"
                );
                assert_eq!(
                    t.into_data().to_vec::<i32>().unwrap(),
                    f.into_data().to_vec::<i32>().unwrap(),
                    "{what} [{r}, {c}]"
                );
            };
            let t = resident("int transpose", || ti.clone().transpose());
            int_eq(t, fi.clone().transpose(), "an Int transposed");
            let t = resident("int reshape", || {
                ti.clone().reshape([1, r, c]).reshape([r, c])
            });
            int_eq(t, fi.clone(), "an Int reshaped there and back");
            if r >= 64 {
                let t = resident("int slice", || ti.clone().slice([32..64, 0..c]));
                int_eq(t, fi.clone().slice([32..64, 0..c]), "an Int's tile rows");
            }

            let (av, bv, rowv) = (bools(2, r * c), bools(3, r * c), bools(4, c));
            let tb = |v: &[bool], s: [usize; 2]| {
                Tensor::<TtBackend, 2, Bool>::from_data(TensorData::new(v.to_vec(), s), &d)
                    .to_device(&d)
            };
            let fb = |v: &[bool], s: [usize; 2]| {
                Tensor::<Flex, 2, Bool>::from_data(TensorData::new(v.to_vec(), s), &FlexDevice)
            };
            let (a, b, row) = (tb(&av, [r, c]), tb(&bv, [r, c]), tb(&rowv, [1, c]));
            let (fa, fbb, frow) = (fb(&av, [r, c]), fb(&bv, [r, c]), fb(&rowv, [1, c]));
            let bool_eq = |t: Tensor<TtBackend, 2, Bool>, f: Tensor<Flex, 2, Bool>, what: &str| {
                assert!(
                    on_device(&t.clone().into_primitive()),
                    "{what}: not on the device"
                );
                let got = t.into_data();
                assert_eq!(got.dtype, f.dtype(), "{what}: dtype");
                assert_eq!(
                    got.to_vec::<bool>().unwrap(),
                    f.into_data().to_vec::<bool>().unwrap(),
                    "{what} [{r}, {c}]"
                );
            };
            let t = resident("not", || a.clone().bool_not());
            bool_eq(t, fa.clone().bool_not(), "!a");
            let t = resident("and", || a.clone().bool_and(b.clone()));
            bool_eq(t, fa.clone().bool_and(fbb.clone()), "a && b");
            let t = resident("or", || a.clone().bool_or(b.clone()));
            bool_eq(t, fa.clone().bool_or(fbb.clone()), "a || b");
            let t = resident("xor", || a.clone().bool_xor(b.clone()));
            bool_eq(t, fa.clone().bool_xor(fbb.clone()), "a ^ b");
            let t = resident("and, a row broadcast", || a.clone().bool_and(row.clone()));
            bool_eq(t, fa.clone().bool_and(frow.clone()), "a && row");
            // A transpose is a view; element-wise ops do not read one yet (M3),
            // so the claim is the view's, not a logic op's on it.
            let t = resident("bool transpose", || a.clone().transpose());
            bool_eq(t, fa.clone().transpose(), "a^T");
        }
    });
}

/// Values with every special of either sign often enough that the ops' edge
/// lanes all occur: both zeros, infinities, NaNs, the extremes; no denormals,
/// which the products flush (numerics row D; `step43` holds those lanes).
fn floats(seed: u64, n: usize) -> Vec<f32> {
    let specials = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        -1.0,
        1.0,
        0.5,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if i % 5 == 0 {
                specials[(s >> 33) as usize % specials.len()]
            } else {
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0
            }
        })
        .collect()
}

fn fbits<B: burn::tensor::backend::Backend>(t: Tensor<B, 2>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn same_bits(got: &[u32], want: &[u32], what: &str) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let nan = f32::from_bits(*w).is_nan() && f32::from_bits(*g).is_nan();
        assert!(
            g == w || nan,
            "{what}: element {i}: {g:#010x} vs Flex {w:#010x}"
        );
    }
}

/// S2 (10.2c) through Burn: every compare, select and sign method, and the
/// three activations, on resident operands, against Flex bit for bit (a NaN
/// by class, which a product of one canonicalises), downloading nothing, the
/// result -- a `Bool` for a comparison -- on the device. Exact ops, so they
/// run on the device whatever the size, exact mode included.
#[test]
fn compare_select_and_sign_stay_on_the_card() {
    use burn::tensor::activation;
    let config = Config {
        exact: true,
        ..Config::default()
    };
    with_device(config, |d| {
        for [r, c] in [[37, 70], [64, 32]] {
            let (av, bv, rowv) = (floats(1, r * c), floats(2, r * c), floats(3, c));
            let tt = |v: &[f32], s: [usize; 2]| {
                Tensor::<TtBackend, 2>::from_data(TensorData::new(v.to_vec(), s), &d).to_device(&d)
            };
            let fl = |v: &[f32], s: [usize; 2]| {
                Tensor::<Flex, 2>::from_data(TensorData::new(v.to_vec(), s), &FlexDevice)
            };
            let (a, b, row) = (tt(&av, [r, c]), tt(&bv, [r, c]), tt(&rowv, [1, c]));
            let (fa, fb, frow) = (fl(&av, [r, c]), fl(&bv, [r, c]), fl(&rowv, [1, c]));
            let float = |t: Tensor<TtBackend, 2>, f: Tensor<Flex, 2>, what: &str| {
                assert!(
                    on_device(&t.clone().into_primitive().tensor()),
                    "{what}: not on the device"
                );
                same_bits(&fbits(t), &fbits(f), &format!("{what} [{r}, {c}]"));
            };
            let boolean = |t: Tensor<TtBackend, 2, Bool>, f: Tensor<Flex, 2, Bool>, what: &str| {
                assert!(
                    on_device(&t.clone().into_primitive()),
                    "{what}: not on the device"
                );
                assert_eq!(
                    t.into_data().to_vec::<bool>().unwrap(),
                    f.into_data().to_vec::<bool>().unwrap(),
                    "{what} [{r}, {c}]"
                );
            };
            // A cast to the dtype a tensor has is the tensor, kept resident.
            float(
                resident("cast to F32", || {
                    a.clone().neg().cast(burn::tensor::FloatDType::F32)
                }),
                fa.clone().neg(),
                "a no-op cast",
            );
            float(resident("neg", || a.clone().neg()), fa.clone().neg(), "neg");
            float(resident("abs", || a.clone().abs()), fa.clone().abs(), "abs");
            float(
                resident("sign", || a.clone().sign()),
                fa.clone().sign(),
                "sign",
            );
            float(
                resident("clamp", || a.clone().clamp(-1.0, 1.0)),
                fa.clone().clamp(-1.0, 1.0),
                "clamp",
            );
            float(
                resident("clamp_min", || a.clone().clamp_min(0.0)),
                fa.clone().clamp_min(0.0),
                "clamp_min",
            );
            float(
                resident("clamp_max", || a.clone().clamp_max(0.5)),
                fa.clone().clamp_max(0.5),
                "clamp_max",
            );
            boolean(
                resident("equal", || a.clone().equal(b.clone())),
                fa.clone().equal(fb.clone()),
                "equal",
            );
            boolean(
                resident("greater", || a.clone().greater(b.clone())),
                fa.clone().greater(fb.clone()),
                "greater",
            );
            boolean(
                resident("lower_equal", || a.clone().lower_equal(b.clone())),
                fa.clone().lower_equal(fb.clone()),
                "lower_equal",
            );
            boolean(
                resident("not_equal by a row", || a.clone().not_equal(row.clone())),
                fa.clone().not_equal(frow.clone()),
                "not_equal by a row",
            );
            boolean(
                resident("lower_elem", || a.clone().lower_elem(0.5)),
                fa.clone().lower_elem(0.5),
                "lower_elem",
            );
            boolean(
                resident("greater_equal_elem", || a.clone().greater_equal_elem(-0.0)),
                fa.clone().greater_equal_elem(-0.0),
                "greater_equal_elem",
            );
            boolean(
                resident("is_nan", || a.clone().is_nan()),
                fa.clone().is_nan(),
                "is_nan",
            );
            boolean(
                resident("is_inf", || a.clone().is_inf()),
                fa.clone().is_inf(),
                "is_inf",
            );
            // A mask made on the device, used there.
            float(
                resident("mask_fill", || {
                    let m = a.clone().greater(b.clone());
                    a.clone().mask_fill(m, 7.5)
                }),
                fa.clone().mask_fill(fa.clone().greater(fb.clone()), 7.5),
                "mask_fill",
            );
            float(
                resident("mask_where", || {
                    let m = a.clone().lower_elem(0.0);
                    a.clone().mask_where(m, b.clone())
                }),
                fa.clone()
                    .mask_where(fa.clone().lower_elem(0.0), fb.clone()),
                "mask_where",
            );
            float(
                resident("leaky_relu", || activation::leaky_relu(a.clone(), 0.01)),
                activation::leaky_relu(fa.clone(), 0.01),
                "leaky_relu",
            );
            float(
                resident("hard_sigmoid", || {
                    activation::hard_sigmoid(a.clone(), 0.2, 0.5)
                }),
                activation::hard_sigmoid(fa.clone(), 0.2, 0.5),
                "hard_sigmoid",
            );
            let alpha = floats(4, c);
            let ta = Tensor::<TtBackend, 1>::from_data(TensorData::new(alpha.clone(), [c]), &d)
                .to_device(&d);
            let fal = Tensor::<Flex, 1>::from_data(TensorData::new(alpha, [c]), &FlexDevice);
            float(
                resident("prelu", || activation::prelu(a.clone(), ta.clone())),
                activation::prelu(fa.clone(), fal),
                "prelu",
            );
        }
    });
}

/// `got` within `rel` of Flex's `want` (plus Flex's own ulp); a NaN by class;
/// infinities and zeros exactly.
fn within(got: &[f32], want: &[f32], rel: impl Fn(usize) -> f64, what: &str) {
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "{what}: element {i}: {g:e} vs NaN");
        } else if w.is_infinite() || w == 0.0 {
            assert_eq!(g, w, "{what}: element {i}");
        } else {
            // Within the bound of the range's ends the device may round over:
            // an infinity for a near-`MAX`, a zero for a near-`MIN_POSITIVE`.
            let b2 = 1.0 + 2.0 * (rel(i) + 1.2e-7);
            let wa = (w as f64).abs();
            if (wa * b2 > f32::MAX as f64 && g == w.signum() * f32::INFINITY)
                || (wa < f32::MIN_POSITIVE as f64 * b2 && g == 0.0)
            {
                continue;
            }
            let r = (g as f64 - w as f64).abs() / (w as f64).abs();
            assert!(
                r <= rel(i) + 1.2e-7,
                "{what}: element {i}: {g:e} vs Flex {w:e}: {r:e}"
            );
        }
    }
}

/// S4's algebraic ops (10.2d) through Burn, on tensors big enough
/// (`APPROX_MIN_TILES`) for an approximation to run on the device: `sqrt`,
/// `log1p`, `powf` by a tensor, by an integer tensor and by scalars --
/// integral ones by Flex's own dispatch (`ones`, the tensor, a product, a
/// reciprocal) -- and `int_into_float`, exact. Resident, nothing downloaded,
/// each within its derived bound of Flex.
#[test]
fn algebraic_ops_stay_on_the_card_within_their_bounds() {
    use tt_kernels::sfpu::ops::{pow_bound, LOG1P_BOUND};
    with_device(Config::default(), |d| {
        let [r, c] = [64, 128];
        let xv: Vec<f32> = floats(5, r * c).iter().map(|x| x.abs() + 0.25).collect();
        let yv: Vec<f32> = floats(6, r * c)
            .iter()
            .map(|y| {
                if y.is_finite() {
                    y.clamp(-4.0, 4.0)
                } else {
                    *y
                }
            })
            .collect();
        let iv: Vec<i32> = (0..r * c).map(|i| (i as i32 % 7) - 3).collect();
        let tt = |v: &[f32]| {
            Tensor::<TtBackend, 2>::from_data(TensorData::new(v.to_vec(), [r, c]), &d).to_device(&d)
        };
        let fl = |v: &[f32]| {
            Tensor::<Flex, 2>::from_data(TensorData::new(v.to_vec(), [r, c]), &FlexDevice)
        };
        let (x, y, fx, fy) = (tt(&xv), tt(&yv), fl(&xv), fl(&yv));
        let ti = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(iv.clone(), [r, c]), &d)
            .to_device(&d);
        let fi =
            Tensor::<Flex, 2, Int>::from_data(TensorData::new(iv.clone(), [r, c]), &FlexDevice);
        let vals = |t: Tensor<TtBackend, 2>, what: &str| {
            assert!(
                on_device(&t.clone().into_primitive().tensor()),
                "{what}: not on the device"
            );
            t.into_data().to_vec::<f32>().unwrap()
        };
        let host = |t: Tensor<Flex, 2>| t.into_data().to_vec::<f32>().unwrap();
        let g = vals(resident("sqrt", || x.clone().sqrt()), "sqrt");
        within(&g, &host(fx.clone().sqrt()), |_| 1.2e-7, "sqrt");
        let g = vals(resident("log1p", || x.clone().log1p()), "log1p");
        within(&g, &host(fx.clone().log1p()), |_| LOG1P_BOUND, "log1p");
        let g = vals(resident("powf", || x.clone().powf(y.clone())), "powf");
        within(
            &g,
            &host(fx.clone().powf(fy.clone())),
            |i| pow_bound(xv[i], yv[i]),
            "powf",
        );
        // `float_powi` by an `I32` tensor, which Burn's tensor API does not
        // reach (its `powi` takes floats): through the backend's trait.
        let g = vals(
            resident("powi", || {
                use burn::tensor::backend::ops::FloatTensorOps;
                let p = <TtBackend as FloatTensorOps<TtBackend>>::float_powi(
                    x.clone().into_primitive().tensor(),
                    ti.clone().into_primitive(),
                );
                Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(p))
            }),
            "powi",
        );
        within(
            &g,
            &host(fx.clone().powf(fi.clone().float())),
            |i| pow_bound(xv[i], iv[i] as f32),
            "powi",
        );
        // `x^1` is `x`, the tensor itself (Flex's dispatch): nothing moves.
        let one = resident("powf_scalar(1)", || x.clone().powf_scalar(1.0));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&one.into_data().to_vec::<f32>().unwrap()),
            bits(&xv),
            "powf_scalar(1)"
        );
        for e in [2.5f32, -0.5, 0.0, 2.0, -1.0, -2.0, 3.0] {
            let what = format!("powf_scalar({e})");
            let g = vals(resident(&what, || x.clone().powf_scalar(e)), &what);
            within(
                &g,
                &host(fx.clone().powf_scalar(e)),
                |i| pow_bound(xv[i], e),
                &what,
            );
        }
        let g = resident("int_into_float", || ti.clone().float());
        assert!(
            on_device(&g.clone().into_primitive().tensor()),
            "int_into_float: not on the device"
        );
        let gb: Vec<u32> = g
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let fb: Vec<u32> = fi
            .float()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(gb, fb, "int_into_float: as f32");
    });
}
