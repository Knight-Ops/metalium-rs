//! Flash-decode (`docs/plans/long-context.md` Part 2): decode attention past the fused op's L1
//! limit, a head's positions split over units and joined by a second op.
//!
//! P0, the oracle first. Before any split is built, the host oracle's `unsplit`
//! (`tt_tests::flash_oracle`: the Matrix Unit's measured group adder for the products, the SFPU
//! interpreter for the scale and mask, `SUB_EXP`, `DIV` and the reductions' own programs) must be
//! **today's op, bit for bit** (`Session::attention_rows`, through Burn's `attention`): the split's
//! oracle is built from the same parts, so an oracle that does not predict the unsplit op exactly
//! would prove nothing about the split one. The data has full mantissas (every bit TF32 drops is
//! live) and no subnormals, NaN or infinity, which the Matrix Unit model does not cover; a masked
//! tail is set in every head.

use burn::tensor::{module::attention, Bool, Tensor, TensorData};
use burn_tensor::ops::AttentionModuleOptions;
use burn_tt::{TileChoice, TtBackend};
use tt_tests::burn_device::{with_device, Config};
use tt_tests::flash_oracle::{unsplit, Head};

/// Full-mantissa values in `[-2, -2^-8] u [2^-8, 2]`: no zeros, subnormals or extremes.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let mag = 2f32.powi(-((x >> 60) as i32 % 9))
                * (1.0 + (x >> 37 & 0x7f_ffff) as f32 / 8_388_608.0);
            if x & 1 == 0 {
                mag
            } else {
                -mag
            }
        })
        .collect()
}

fn config() -> Config {
    Config {
        tiles: Some(TileChoice::Count(8)),
        ..Config::default()
    }
}

/// `heads` query heads over `kv` key/value heads of width `dim`, attending `len` positions of
/// which the first `live` are not masked: the device's result through today's op, and the
/// oracle's, as bits, head by head.
fn check(
    d: &burn_tt::TtDevice,
    heads: usize,
    kv: usize,
    dim: usize,
    len: usize,
    live: usize,
    seed: u64,
) {
    let groups = heads / kv;
    let qd = values(seed, heads * dim);
    let (kd, vd) = (
        values(seed + 1, len * kv * dim),
        values(seed + 2, len * kv * dim),
    );
    let q = Tensor::<TtBackend, 3>::from_data(TensorData::new(qd.clone(), [1, 1, heads * dim]), d)
        .reshape([1, 1, heads, dim])
        .swap_dims(1, 2)
        .reshape([1, kv, groups, dim]);
    let cache = |data: &[f32]| {
        Tensor::<TtBackend, 3>::from_data(TensorData::new(data.to_vec(), [1, len, kv * dim]), d)
            .reshape([1, len, kv, dim])
            .swap_dims(1, 2)
    };
    let masked: Vec<bool> = (0..len).map(|p| p >= live).collect();
    let mask =
        Tensor::<TtBackend, 4, Bool>::from_data(TensorData::new(masked.clone(), [1, 1, 1, len]), d);
    burn_tt::set_attention_rows(true);
    let opts = AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: false,
    };
    let (out, report) = burn_tt::with_report(|| {
        attention(q, cache(&kd), cache(&vd), Some(mask), None, opts)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    });
    let what = format!("{heads} heads over {kv}, d {dim}, {len} positions, {live} live");
    assert!(
        report.op("attention/rows").is_some_and(|s| s.jobs > 0),
        "{what}: today's op ran"
    );
    let scale = 1.0 / (dim as f32).sqrt();
    for h in 0..kv {
        let q: Vec<f32> = (0..groups)
            .flat_map(|g| qd[(h * groups + g) * dim..][..dim].to_vec())
            .collect();
        let head = |data: &[f32]| -> Vec<f32> {
            (0..len)
                .flat_map(|p| data[p * kv * dim + h * dim..][..dim].to_vec())
                .collect()
        };
        let (k, v) = (head(&kd), head(&vd));
        let want = unsplit(&Head {
            q: &q,
            k: &k,
            v: &v,
            masked: &masked,
            m: groups,
            d: dim,
            dv: dim,
            scale,
        });
        let got = &out[h * groups * dim..][..groups * dim];
        let wrong = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        assert_eq!(
            wrong,
            0,
            "{what}, head {h}: {wrong} of {} outputs differ from the oracle; first (device, \
             oracle): {:?}",
            want.len(),
            got.iter()
                .zip(&want)
                .find(|(g, w)| g.to_bits() != w.to_bits()),
        );
    }
}

#[test]
#[ignore = "parked (docs/plans/long-context.md Part 2): the Matrix Unit model is not bit for bit on full-mantissa operands"]
fn the_oracle_is_today_s_op_bit_for_bit() {
    with_device(config(), |d| {
        // Multi-head, and grouped-query rows folded a key/value head.
        check(&d, 4, 4, 64, 128, 100, 7);
        check(&d, 8, 2, 128, 256, 230, 11);
    });
}

/// Diagnosis: a plain product of full-mantissa F32 operands on the TF32 route against the Matrix
/// Unit model.
#[test]
#[ignore = "parked (docs/plans/long-context.md Part 2): the Matrix Unit model is not bit for bit on full-mantissa operands"]
fn probe_the_model_on_a_full_mantissa_product() {
    with_device(config(), |d| {
        let (m, k, n) = (4, 64, 32);
        let (a, b) = (values(3, m * k), values(4, k * n));
        let ta = Tensor::<TtBackend, 2>::from_data(TensorData::new(a.clone(), [m, k]), &d);
        let tb = Tensor::<TtBackend, 2>::from_data(TensorData::new(b.clone(), [k, n]), &d);
        let got = ta.matmul(tb).into_data().to_vec::<f32>().unwrap();
        let mut wrong = 0;
        for i in 0..m {
            for j in 0..n {
                let col: Vec<f32> = (0..k).map(|p| b[p * n + j]).collect();
                let want = tt_tests::matrix_unit::dot(
                    &a[i * k..][..k],
                    &col,
                    tt_tests::matrix_unit::MEASURED,
                )
                .0;
                if want.to_bits() != got[i * n + j].to_bits() {
                    let ulps = got[i * n + j].to_bits() as i64 - want.to_bits() as i64;
                    eprintln!(
                        "MISS ({i},{j}): device {} model {} ({ulps:+} ulp of the magnitude)",
                        got[i * n + j],
                        want
                    );
                    wrong += 1;
                }
            }
        }
        eprintln!("{wrong} of {} differ", m * n);
    });
}

/// Diagnosis: the same product against variants of the group rule (quantum's fraction bits,
/// rounding of each product).
#[test]
#[ignore = "parked (docs/plans/long-context.md Part 2): the Matrix Unit model is not bit for bit on full-mantissa operands"]
fn probe_group_rule_variants() {
    use tt_isa::numerics::{src_a_fidelity_bits, src_b_fidelity_bits};
    fn exp2(x: f32) -> i32 {
        f64::from(x.abs()).log2().floor() as i32
    }
    fn ties_up(x: f64) -> f32 {
        let near = x as f32;
        let up = |v: f32| {
            f32::from_bits(if v >= 0.0 {
                v.to_bits() + 1
            } else {
                v.to_bits() - 1
            })
        };
        let dn = |v: f32| -up(-v);
        let (lo, hi) = if f64::from(near) <= x {
            (near, up(near))
        } else {
            (dn(near), near)
        };
        if x - f64::from(lo) == f64::from(hi) - x {
            hi
        } else {
            near
        }
    }
    #[derive(Clone, Copy, Debug)]
    enum Mode {
        HalfUp,
        Trunc,
        Floor,
    }
    let dotv = |a: &[f32], b: &[f32], fb: i32, mode: Mode| -> f32 {
        let mut dst = 0f32;
        for face in 0..a.len() / 16 {
            for phase in 0..4u32 {
                for g in 0..2 {
                    let at = face * 16 + g * 8;
                    let mut e_max = i32::MIN;
                    let mut ps = Vec::new();
                    for k in at..at + 8 {
                        let (x, y) = (
                            src_b_fidelity_bits(a[k], phase),
                            src_a_fidelity_bits(b[k], phase),
                        );
                        ps.push(f64::from(x) * f64::from(y));
                        if a[k] != 0.0 && b[k] != 0.0 {
                            let ea = exp2(a[k]) - if phase & 2 != 0 { 7 } else { 0 };
                            let eb = exp2(b[k]) - if phase & 1 != 0 { 5 } else { 0 };
                            e_max = e_max.max(ea + eb);
                        }
                    }
                    if e_max == i32::MIN {
                        continue;
                    }
                    let q = 2f64.powi(e_max - fb);
                    let sum: f64 = ps
                        .iter()
                        .map(|&p| match mode {
                            Mode::HalfUp => p.signum() * (p.abs() / q + 0.5).floor() * q,
                            Mode::Trunc => p.signum() * (p.abs() / q).floor() * q,
                            Mode::Floor => (p / q).floor() * q,
                        })
                        .sum();
                    dst = ties_up(f64::from(dst) + sum);
                }
            }
        }
        dst
    };
    with_device(config(), |d| {
        let (m, k, n) = (4, 64, 32);
        let (a, b) = (values(3, m * k), values(4, k * n));
        let ta = Tensor::<TtBackend, 2>::from_data(TensorData::new(a.clone(), [m, k]), &d);
        let tb = Tensor::<TtBackend, 2>::from_data(TensorData::new(b.clone(), [k, n]), &d);
        let got = ta.matmul(tb).into_data().to_vec::<f32>().unwrap();
        for mode in [Mode::HalfUp, Mode::Trunc, Mode::Floor] {
            for fb in 8..=16 {
                let mut wrong = 0;
                for i in 0..m {
                    for j in 0..n {
                        let col: Vec<f32> = (0..k).map(|p| b[p * n + j]).collect();
                        if dotv(&a[i * k..][..k], &col, fb, mode).to_bits()
                            != got[i * n + j].to_bits()
                        {
                            wrong += 1;
                        }
                    }
                }
                eprintln!(
                    "VARIANT {mode:?} fraction bits {fb}: {wrong} of {} differ",
                    m * n
                );
            }
        }
    });
}

/// Diagnosis: the group rule as measured, its sums reaching `Dst` in other groupings and roundings.
#[test]
#[ignore = "parked (docs/plans/long-context.md Part 2): the Matrix Unit model is not bit for bit on full-mantissa operands"]
fn probe_accumulation_variants() {
    use tt_isa::numerics::{src_a_fidelity_bits, src_b_fidelity_bits};
    fn exp2(x: f32) -> i32 {
        f64::from(x.abs()).log2().floor() as i32
    }
    fn round(x: f64, mode: u8) -> f32 {
        let near = x as f32; // ties to even
        let up = |v: f32| {
            f32::from_bits(if v >= 0.0 {
                v.to_bits() + 1
            } else {
                v.to_bits() - 1
            })
        };
        let dn = |v: f32| -up(-v);
        let (lo, hi) = if f64::from(near) <= x {
            (near, up(near))
        } else {
            (dn(near), near)
        };
        match mode {
            0 => {
                if x - f64::from(lo) == f64::from(hi) - x {
                    hi
                } else {
                    near
                }
            }
            1 => near,
            _ => {
                if x >= 0.0 {
                    lo
                } else {
                    hi
                }
            }
        }
    }
    // grouping: 0 = a `Dst` add a group, 1 = a phase's two groups together, 2 = a face's whole.
    let dotv = |a: &[f32], b: &[f32], grouping: u8, mode: u8| -> f32 {
        let mut dst = 0f32;
        for face in 0..a.len() / 16 {
            let mut face_sum = 0f64;
            for phase in 0..4u32 {
                let mut phase_sum = 0f64;
                for g in 0..2 {
                    let at = face * 16 + g * 8;
                    let mut e_max = i32::MIN;
                    let mut ps = Vec::new();
                    for k in at..at + 8 {
                        let (x, y) = (
                            src_b_fidelity_bits(a[k], phase),
                            src_a_fidelity_bits(b[k], phase),
                        );
                        ps.push(f64::from(x) * f64::from(y));
                        if a[k] != 0.0 && b[k] != 0.0 {
                            let ea = exp2(a[k]) - if phase & 2 != 0 { 7 } else { 0 };
                            let eb = exp2(b[k]) - if phase & 1 != 0 { 5 } else { 0 };
                            e_max = e_max.max(ea + eb);
                        }
                    }
                    let sum = if e_max == i32::MIN {
                        0.0
                    } else {
                        let q = 2f64.powi(e_max - 10);
                        ps.iter()
                            .map(|&p| p.signum() * (p.abs() / q + 0.5).floor() * q)
                            .sum()
                    };
                    match grouping {
                        0 => dst = round(f64::from(dst) + sum, mode),
                        _ => phase_sum += sum,
                    }
                }
                match grouping {
                    1 => dst = round(f64::from(dst) + phase_sum, mode),
                    2 => face_sum += phase_sum,
                    _ => {}
                }
            }
            if grouping == 2 {
                dst = round(f64::from(dst) + face_sum, mode);
            }
        }
        dst
    };
    with_device(config(), |d| {
        let (m, k, n) = (4, 64, 32);
        let (a, b) = (values(3, m * k), values(4, k * n));
        let ta = Tensor::<TtBackend, 2>::from_data(TensorData::new(a.clone(), [m, k]), &d);
        let tb = Tensor::<TtBackend, 2>::from_data(TensorData::new(b.clone(), [k, n]), &d);
        let got = ta.matmul(tb).into_data().to_vec::<f32>().unwrap();
        for grouping in 0..3 {
            for mode in 0..3 {
                let mut wrong = 0;
                for i in 0..m {
                    for j in 0..n {
                        let col: Vec<f32> = (0..k).map(|p| b[p * n + j]).collect();
                        if dotv(&a[i * k..][..k], &col, grouping, mode).to_bits()
                            != got[i * n + j].to_bits()
                        {
                            wrong += 1;
                        }
                    }
                }
                eprintln!(
                    "ACCUM grouping {grouping} rounding {mode}: {wrong} of {} differ",
                    m * n
                );
            }
        }
    });
}

/// Diagnosis: the measured rule with the fidelity phases in every order.
#[test]
#[ignore = "parked (docs/plans/long-context.md Part 2): the Matrix Unit model is not bit for bit on full-mantissa operands"]
fn probe_phase_orders() {
    use tt_isa::numerics::{src_a_fidelity_bits, src_b_fidelity_bits};
    fn exp2(x: f32) -> i32 {
        f64::from(x.abs()).log2().floor() as i32
    }
    fn ties_up(x: f64) -> f32 {
        let near = x as f32;
        let up = |v: f32| {
            f32::from_bits(if v >= 0.0 {
                v.to_bits() + 1
            } else {
                v.to_bits() - 1
            })
        };
        let dn = |v: f32| -up(-v);
        let (lo, hi) = if f64::from(near) <= x {
            (near, up(near))
        } else {
            (dn(near), near)
        };
        if x - f64::from(lo) == f64::from(hi) - x {
            hi
        } else {
            near
        }
    }
    let dotv = |a: &[f32], b: &[f32], order: [u32; 4], groups_outer: bool| -> f32 {
        let mut dst = 0f32;
        for face in 0..a.len() / 16 {
            let mut steps = Vec::new();
            for &phase in &order {
                for g in 0..2 {
                    steps.push((phase, g));
                }
            }
            if groups_outer {
                steps.sort_by_key(|&(_, g)| g);
            }
            for (phase, g) in steps {
                let at = face * 16 + g * 8;
                let mut e_max = i32::MIN;
                let mut ps = Vec::new();
                for k in at..at + 8 {
                    let (x, y) = (
                        src_b_fidelity_bits(a[k], phase),
                        src_a_fidelity_bits(b[k], phase),
                    );
                    ps.push(f64::from(x) * f64::from(y));
                    if a[k] != 0.0 && b[k] != 0.0 {
                        let ea = exp2(a[k]) - if phase & 2 != 0 { 7 } else { 0 };
                        let eb = exp2(b[k]) - if phase & 1 != 0 { 5 } else { 0 };
                        e_max = e_max.max(ea + eb);
                    }
                }
                if e_max == i32::MIN {
                    continue;
                }
                let q = 2f64.powi(e_max - 10);
                let sum: f64 = ps
                    .iter()
                    .map(|&p| p.signum() * (p.abs() / q + 0.5).floor() * q)
                    .sum();
                dst = ties_up(f64::from(dst) + sum);
            }
        }
        dst
    };
    with_device(config(), |d| {
        let (m, k, n) = (4, 64, 32);
        let (a, b) = (values(3, m * k), values(4, k * n));
        let ta = Tensor::<TtBackend, 2>::from_data(TensorData::new(a.clone(), [m, k]), &d);
        let tb = Tensor::<TtBackend, 2>::from_data(TensorData::new(b.clone(), [k, n]), &d);
        let got = ta.matmul(tb).into_data().to_vec::<f32>().unwrap();
        let mut orders = Vec::new();
        for a0 in 0..4u32 {
            for a1 in 0..4u32 {
                for a2 in 0..4u32 {
                    for a3 in 0..4u32 {
                        let o = [a0, a1, a2, a3];
                        let mut s = o;
                        s.sort();
                        if s == [0, 1, 2, 3] {
                            orders.push(o);
                        }
                    }
                }
            }
        }
        for groups_outer in [false, true] {
            for &order in &orders {
                let mut wrong = 0;
                for i in 0..m {
                    for j in 0..n {
                        let col: Vec<f32> = (0..k).map(|p| b[p * n + j]).collect();
                        if dotv(&a[i * k..][..k], &col, order, groups_outer).to_bits()
                            != got[i * n + j].to_bits()
                        {
                            wrong += 1;
                        }
                    }
                }
                eprintln!(
                    "ORDER {order:?} groups outer {groups_outer}: {wrong} of {} differ",
                    m * n
                );
            }
        }
    });
}

/// Diagnosis: each product's rounding to the group's quantum, signed and with guard bits.
#[test]
#[ignore = "parked (docs/plans/long-context.md Part 2): the Matrix Unit model is not bit for bit on full-mantissa operands"]
fn probe_product_rounding() {
    use tt_isa::numerics::{src_a_fidelity_bits, src_b_fidelity_bits};
    fn exp2(x: f32) -> i32 {
        f64::from(x.abs()).log2().floor() as i32
    }
    fn ties_up(x: f64) -> f32 {
        let near = x as f32;
        let up = |v: f32| {
            f32::from_bits(if v >= 0.0 {
                v.to_bits() + 1
            } else {
                v.to_bits() - 1
            })
        };
        let dn = |v: f32| -up(-v);
        let (lo, hi) = if f64::from(near) <= x {
            (near, up(near))
        } else {
            (dn(near), near)
        };
        if x - f64::from(lo) == f64::from(hi) - x {
            hi
        } else {
            near
        }
    }
    // style 0: magnitude half-up (measured); 1: signed half toward +inf; 2: products truncated
    // (magnitude) to q 2^-g, the sum rounded half-up to q; 3: products floored (signed) to q 2^-g,
    // the sum floored to q; 4: as 3, the sum rounded half toward +inf.
    let dotv = |a: &[f32], b: &[f32], style: u8, g_bits: i32| -> f32 {
        let mut dst = 0f32;
        for face in 0..a.len() / 16 {
            for phase in 0..4u32 {
                for g in 0..2 {
                    let at = face * 16 + g * 8;
                    let mut e_max = i32::MIN;
                    let mut ps = Vec::new();
                    for k in at..at + 8 {
                        let (x, y) = (
                            src_b_fidelity_bits(a[k], phase),
                            src_a_fidelity_bits(b[k], phase),
                        );
                        ps.push(f64::from(x) * f64::from(y));
                        if a[k] != 0.0 && b[k] != 0.0 {
                            let ea = exp2(a[k]) - if phase & 2 != 0 { 7 } else { 0 };
                            let eb = exp2(b[k]) - if phase & 1 != 0 { 5 } else { 0 };
                            e_max = e_max.max(ea + eb);
                        }
                    }
                    if e_max == i32::MIN {
                        continue;
                    }
                    let q = 2f64.powi(e_max - 10);
                    let f = q * 2f64.powi(-g_bits);
                    let sum: f64 = match style {
                        0 => ps
                            .iter()
                            .map(|&p| p.signum() * (p.abs() / q + 0.5).floor() * q)
                            .sum(),
                        1 => ps.iter().map(|&p| (p / q + 0.5).floor() * q).sum(),
                        2 => {
                            let s: f64 = ps
                                .iter()
                                .map(|&p| p.signum() * (p.abs() / f).floor() * f)
                                .sum();
                            s.signum() * (s.abs() / q + 0.5).floor() * q
                        }
                        3 => {
                            let s: f64 = ps.iter().map(|&p| (p / f).floor() * f).sum();
                            (s / q).floor() * q
                        }
                        _ => {
                            let s: f64 = ps.iter().map(|&p| (p / f).floor() * f).sum();
                            (s / q + 0.5).floor() * q
                        }
                    };
                    dst = ties_up(f64::from(dst) + sum);
                }
            }
        }
        dst
    };
    with_device(config(), |d| {
        let (m, k, n) = (4, 64, 32);
        let (a, b) = (values(3, m * k), values(4, k * n));
        let ta = Tensor::<TtBackend, 2>::from_data(TensorData::new(a.clone(), [m, k]), &d);
        let tb = Tensor::<TtBackend, 2>::from_data(TensorData::new(b.clone(), [k, n]), &d);
        let got = ta.matmul(tb).into_data().to_vec::<f32>().unwrap();
        for style in 0..5u8 {
            for g_bits in 0..5 {
                if style < 2 && g_bits > 0 {
                    continue;
                }
                let mut wrong = 0;
                for i in 0..m {
                    for j in 0..n {
                        let col: Vec<f32> = (0..k).map(|p| b[p * n + j]).collect();
                        if dotv(&a[i * k..][..k], &col, style, g_bits).to_bits()
                            != got[i * n + j].to_bits()
                        {
                            wrong += 1;
                        }
                    }
                }
                eprintln!(
                    "ROUNDING style {style} guard {g_bits}: {wrong} of {} differ",
                    m * n
                );
            }
        }
    });
}

/// Diagnosis: a product rounding chosen per fidelity phase.
#[test]
#[ignore = "parked (docs/plans/long-context.md Part 2): the Matrix Unit model is not bit for bit on full-mantissa operands"]
fn probe_per_phase_rounding() {
    use tt_isa::numerics::{src_a_fidelity_bits, src_b_fidelity_bits};
    fn exp2(x: f32) -> i32 {
        f64::from(x.abs()).log2().floor() as i32
    }
    fn ties_up(x: f64) -> f32 {
        let near = x as f32;
        let up = |v: f32| {
            f32::from_bits(if v >= 0.0 {
                v.to_bits() + 1
            } else {
                v.to_bits() - 1
            })
        };
        let dn = |v: f32| -up(-v);
        let (lo, hi) = if f64::from(near) <= x {
            (near, up(near))
        } else {
            (dn(near), near)
        };
        if x - f64::from(lo) == f64::from(hi) - x {
            hi
        } else {
            near
        }
    }
    fn rnd(p: f64, q: f64, mode: u8) -> f64 {
        match mode {
            0 => p.signum() * (p.abs() / q + 0.5).floor() * q, // half away from zero
            1 => (p / q + 0.5).floor() * q,                    // half toward +inf
            2 => (p / q).floor() * q,                          // toward -inf
            _ => p.signum() * (p.abs() / q).floor() * q,       // toward zero
        }
    }
    let dotv = |a: &[f32], b: &[f32], modes: [u8; 4]| -> f32 {
        let mut dst = 0f32;
        for face in 0..a.len() / 16 {
            for phase in 0..4u32 {
                for g in 0..2 {
                    let at = face * 16 + g * 8;
                    let mut e_max = i32::MIN;
                    let mut ps = Vec::new();
                    for k in at..at + 8 {
                        let (x, y) = (
                            src_b_fidelity_bits(a[k], phase),
                            src_a_fidelity_bits(b[k], phase),
                        );
                        ps.push(f64::from(x) * f64::from(y));
                        if a[k] != 0.0 && b[k] != 0.0 {
                            let ea = exp2(a[k]) - if phase & 2 != 0 { 7 } else { 0 };
                            let eb = exp2(b[k]) - if phase & 1 != 0 { 5 } else { 0 };
                            e_max = e_max.max(ea + eb);
                        }
                    }
                    if e_max == i32::MIN {
                        continue;
                    }
                    let q = 2f64.powi(e_max - 10);
                    let sum: f64 = ps.iter().map(|&p| rnd(p, q, modes[phase as usize])).sum();
                    dst = ties_up(f64::from(dst) + sum);
                }
            }
        }
        dst
    };
    with_device(config(), |d| {
        let (m, k, n) = (4, 64, 32);
        let (a, b) = (values(3, m * k), values(4, k * n));
        let ta = Tensor::<TtBackend, 2>::from_data(TensorData::new(a.clone(), [m, k]), &d);
        let tb = Tensor::<TtBackend, 2>::from_data(TensorData::new(b.clone(), [k, n]), &d);
        let got = ta.matmul(tb).into_data().to_vec::<f32>().unwrap();
        let mut results = Vec::new();
        for c in 0..256u32 {
            let modes = [
                (c & 3) as u8,
                (c >> 2 & 3) as u8,
                (c >> 4 & 3) as u8,
                (c >> 6 & 3) as u8,
            ];
            let mut wrong = 0;
            for i in 0..m {
                for j in 0..n {
                    let col: Vec<f32> = (0..k).map(|p| b[p * n + j]).collect();
                    if dotv(&a[i * k..][..k], &col, modes).to_bits() != got[i * n + j].to_bits() {
                        wrong += 1;
                    }
                }
            }
            results.push((wrong, modes));
        }
        results.sort();
        for (wrong, modes) in &results[..6] {
            eprintln!("PHASE modes {modes:?}: {wrong} of {} differ", m * n);
        }
    });
}

/// Diagnosis: the measured group rule, its sums added to `Dst` with guard bits floored.
#[test]
#[ignore = "parked (docs/plans/long-context.md Part 2): the Matrix Unit model is not bit for bit on full-mantissa operands"]
fn probe_dst_alignment() {
    use tt_isa::numerics::{src_a_fidelity_bits, src_b_fidelity_bits};
    fn exp2(x: f32) -> i32 {
        f64::from(x.abs()).log2().floor() as i32
    }
    fn ties_up(x: f64) -> f32 {
        let near = x as f32;
        let up = |v: f32| {
            f32::from_bits(if v >= 0.0 {
                v.to_bits() + 1
            } else {
                v.to_bits() - 1
            })
        };
        let dn = |v: f32| -up(-v);
        let (lo, hi) = if f64::from(near) <= x {
            (near, up(near))
        } else {
            (dn(near), near)
        };
        if x - f64::from(lo) == f64::from(hi) - x {
            hi
        } else {
            near
        }
    }
    fn floor32(x: f64) -> f32 {
        let near = x as f32;
        if f64::from(near) <= x {
            near
        } else {
            -f32::from_bits((-near).to_bits() + 1)
        }
    }
    // The ulp of the larger of `dst` and `x`'s magnitudes, in the result's binade.
    fn ulp_of(v: f64) -> f64 {
        if v == 0.0 {
            f64::MIN_POSITIVE
        } else {
            2f64.powi(v.abs().log2().floor() as i32 - 23)
        }
    }
    // mode 0: ties-up (measured). 1: toward -inf. 2..: the sum floored to `g` bits below the ulp of
    // the larger operand's binade, then ties-up.
    let dotv = |a: &[f32], b: &[f32], mode: i32| -> f32 {
        let mut dst = 0f32;
        for face in 0..a.len() / 16 {
            for phase in 0..4u32 {
                for g in 0..2 {
                    let at = face * 16 + g * 8;
                    let mut e_max = i32::MIN;
                    let mut ps = Vec::new();
                    for k in at..at + 8 {
                        let (x, y) = (
                            src_b_fidelity_bits(a[k], phase),
                            src_a_fidelity_bits(b[k], phase),
                        );
                        ps.push(f64::from(x) * f64::from(y));
                        if a[k] != 0.0 && b[k] != 0.0 {
                            let ea = exp2(a[k]) - if phase & 2 != 0 { 7 } else { 0 };
                            let eb = exp2(b[k]) - if phase & 1 != 0 { 5 } else { 0 };
                            e_max = e_max.max(ea + eb);
                        }
                    }
                    if e_max == i32::MIN {
                        continue;
                    }
                    let q = 2f64.powi(e_max - 10);
                    let sum: f64 = ps
                        .iter()
                        .map(|&p| p.signum() * (p.abs() / q + 0.5).floor() * q)
                        .sum();
                    dst = match mode {
                        0 => ties_up(f64::from(dst) + sum),
                        1 => floor32(f64::from(dst) + sum),
                        g => {
                            let big = f64::from(dst).abs().max(sum.abs());
                            let f = ulp_of(big) * 2f64.powi(-(g - 1));
                            let small_floored = |x: f64| (x / f).floor() * f;
                            let (d, s) = if f64::from(dst).abs() >= sum.abs() {
                                (f64::from(dst), small_floored(sum))
                            } else {
                                (small_floored(f64::from(dst)), sum)
                            };
                            ties_up(d + s)
                        }
                    };
                }
            }
        }
        dst
    };
    with_device(config(), |d| {
        let (m, k, n) = (4, 64, 32);
        let (a, b) = (values(3, m * k), values(4, k * n));
        let ta = Tensor::<TtBackend, 2>::from_data(TensorData::new(a.clone(), [m, k]), &d);
        let tb = Tensor::<TtBackend, 2>::from_data(TensorData::new(b.clone(), [k, n]), &d);
        let got = ta.matmul(tb).into_data().to_vec::<f32>().unwrap();
        for mode in 0..8 {
            let mut wrong = 0;
            for i in 0..m {
                for j in 0..n {
                    let col: Vec<f32> = (0..k).map(|p| b[p * n + j]).collect();
                    if dotv(&a[i * k..][..k], &col, mode).to_bits() != got[i * n + j].to_bits() {
                        wrong += 1;
                    }
                }
            }
            eprintln!("DST mode {mode}: {wrong} of {} differ", m * n);
        }
    });
}
