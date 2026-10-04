//! Phase 10 gate (S2, milestone 10.2c): compare, select, sign on the SFPU.
//!
//! Exact ops, so held to `burn-flex`'s own ops **bit for bit** -- NaN payloads,
//! both zeros and denormals included, since they move raw bits and never pass
//! through arithmetic -- except where an op multiplies (`leaky_relu`'s negative
//! side, `hard_sigmoid`), whose operands and results flush denormals as all of
//! the SFPU's arithmetic does (numerics row D) and whose NaNs are compared by
//! class. And the device to its program in the interpreter
//! (`tt_kernels::sfpu::ops::reference_op`) bit for bit, and each output's
//! padding claim to the raw tiles.

use burn::tensor::{activation, Bool, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::{kind_sfpu::*, reference_op, Broadcast};
use tt_kernels::tensor::{DramTensor, Elem, Eltwise, Pad};
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
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
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
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Every special of either sign often, so each pair of them meets in some
/// lane: both zeros, infinities, NaNs with payloads, denormals, extremes, and
/// values around the scalars the cases use.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let specials: [u32; 18] = [
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0xffc0_0001,
        0x0000_0001,
        0x8040_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f00_0000,
        0xbf00_0000,
        0x4000_0000,
        0x3c23_d70a,
        0xc000_0000,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if i % 3 == 0 {
                f32::from_bits(specials[(s >> 33) as usize % specials.len()])
            } else {
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0
            }
        })
        .collect()
}

fn flex(v: &[f32], r: usize, c: usize) -> Tensor<Flex, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), &FlexDevice)
}

fn flex_mask(v: &[u32], r: usize, c: usize) -> Tensor<Flex, 2, Bool> {
    let b: Vec<bool> = v.iter().map(|&x| x != 0).collect();
    Tensor::from_data(TensorData::new(b, [r, c]), &FlexDevice)
}

fn bits(t: Tensor<Flex, 2>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn bool_bits(t: Tensor<Flex, 2, Bool>) -> Vec<u32> {
    t.into_data()
        .to_vec::<bool>()
        .unwrap()
        .iter()
        .map(|&x| u32::from(x))
        .collect()
}

/// The device's bits against Flex's: the same, or for an op that multiplies
/// a NaN by class and a denormal flushed to its sign's zero.
fn assert_flex(got: &[u32], want: &[u32], products: bool, what: &str) {
    assert_flex_or(got, want, want, products, what)
}

/// [`assert_flex`] for an op that may multiply: its lanes whose input `x` is
/// a denormal are left to the interpreter oracle (`sfpu::ops::s2`, which runs
/// the host's semantics on a flushed operand -- the SFPU flushes a denormal
/// before its arithmetic, numerics row D -- and which the device equals bit for
/// bit, `run`); no Flex run states them, since the device decides the branch on
/// the raw `x` and computes on the flushed one. Elsewhere Flex's answer, or its
/// answer on flushed operands (`flushed`) where an intermediate is a denormal.
fn assert_flex_or(got: &[u32], want: &[u32], flushed: &[u32], products: bool, what: &str) {
    assert_flex_on(got, want, flushed, products, None, what)
}

fn assert_flex_on(
    got: &[u32],
    want: &[u32],
    flushed: &[u32],
    products: bool,
    x: Option<&[f32]>,
    what: &str,
) {
    for (i, (&g, (&w, &w2))) in got.iter().zip(want.iter().zip(flushed)).enumerate() {
        if products && x.is_some_and(|x| x[i] != 0.0 && x[i].abs() < f32::MIN_POSITIVE) {
            continue;
        }
        let gf = f32::from_bits(g);
        let like = |w: u32| {
            let wf = f32::from_bits(w);
            g == w
                || (products && wf.is_nan() && gf.is_nan())
                || (products && wf != 0.0 && wf.abs() < f32::MIN_POSITIVE && g == w & 0x8000_0000)
        };
        assert!(
            like(w) || (products && like(w2)),
            "{what}: element {i}: device {g:#010x}, Flex {w:#010x} (on flushed operands {w2:#010x})"
        );
    }
}

/// Denormals flushed to their sign's zero.
fn ftz(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|&x| {
            if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
                f32::from_bits(x.to_bits() & 0x8000_0000)
            } else {
                x
            }
        })
        .collect()
}

/// Run `op` on the device and hold it to its program and its padding claim;
/// return its bits.
#[allow(clippy::too_many_arguments)]
fn run<T: tt_device::Transport>(
    s: &mut Session<T>,
    op: Eltwise,
    bcast: Broadcast,
    t: &[&DramTensor],
    v: &[&[f32]],
    r: usize,
    c: usize,
    what: &str,
) -> Vec<u32> {
    let out = s
        .eltwise3(op, t[0], t.get(1).copied(), t.get(2).copied())
        .unwrap();
    let got = s.download_bits(&out).unwrap();
    let model: Vec<u32> = reference_op(op.kind, [op.scalar, op.scalar2], bcast, v, r, c)
        .iter()
        .map(|x| x.to_bits())
        .collect();
    assert_eq!(got, model, "{what}: the device is its program");
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
    got
}

fn op(kind: u32, scalar: f32, scalar2: f32) -> Eltwise {
    Eltwise {
        kind,
        scalar,
        scalar2,
    }
}

#[test]
fn the_unary_ops_are_flex_s() {
    with_session(|s| {
        for (r, c) in [(37, 70), (64, 128)] {
            let av = values(1, r * c);
            let a = s
                .upload_bits(
                    &av.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    r,
                    c,
                    Elem::F32,
                )
                .unwrap();
            let fa = flex(&av, r, c);
            let fz = flex(&ftz(&av), r, c);
            let none = Broadcast::None;
            // `flushed` empty: nothing multiplies, Flex's bits alone.
            let mut case =
                |o: Eltwise, want: Vec<u32>, flushed: Vec<u32>, products: bool, what: &str| {
                    let got = run(s, o, none, &[&a], &[&av], r, c, what);
                    let flushed = if flushed.is_empty() {
                        want.clone()
                    } else {
                        flushed
                    };
                    let what = format!("{what} [{r}, {c}]");
                    assert_flex_on(&got, &want, &flushed, products, Some(&av), &what);
                };
            case(
                op(NEG, 0.0, 0.0),
                bits(fa.clone().neg()),
                vec![],
                false,
                "neg",
            );
            case(
                op(ABS, 0.0, 0.0),
                bits(fa.clone().abs()),
                vec![],
                false,
                "abs",
            );
            case(
                op(SIGN, 0.0, 0.0),
                bits(fa.clone().sign()),
                vec![],
                false,
                "sign",
            );
            for (lo, hi) in [(-1.0f32, 1.0f32), (-0.0, 0.0), (0.5, 2.0)] {
                let w = bits(fa.clone().clamp(lo, hi));
                case(
                    op(CLAMP, lo, hi),
                    w,
                    vec![],
                    false,
                    &format!("clamp({lo}, {hi})"),
                );
            }
            for v in [0.0f32, -0.0, -0.5, f32::NAN, 1.0] {
                let w = bits(fa.clone().clamp_min(v));
                case(
                    op(CLAMP_MIN, v, 0.0),
                    w,
                    vec![],
                    false,
                    &format!("clamp_min({v})"),
                );
                let w = bits(fa.clone().clamp_max(v));
                case(
                    op(CLAMP_MAX, v, 0.0),
                    w,
                    vec![],
                    false,
                    &format!("clamp_max({v})"),
                );
            }
            for ns in [0.01f32, -2.0] {
                let w = bits(activation::leaky_relu(fa.clone(), ns as f64));
                let z = bits(activation::leaky_relu(fz.clone(), ns as f64));
                case(
                    op(LEAKY_RELU, ns, 0.0),
                    w,
                    z,
                    true,
                    &format!("leaky_relu({ns})"),
                );
            }
            let w = bits(activation::hard_sigmoid(fa.clone(), 0.2, 0.5));
            let z = bits(activation::hard_sigmoid(fz.clone(), 0.2, 0.5));
            case(
                op(HARD_SIGMOID, 0.2, 0.5),
                w,
                z,
                true,
                "hard_sigmoid(0.2, 0.5)",
            );
            case(
                op(IS_NAN, 0.0, 0.0),
                bool_bits(fa.clone().is_nan()),
                vec![],
                false,
                "is_nan",
            );
            case(
                op(IS_INF, 0.0, 0.0),
                bool_bits(fa.clone().is_inf()),
                vec![],
                false,
                "is_inf",
            );
            for v in [0.5f32, -0.0, 0.0, f32::NAN, f32::from_bits(1)] {
                let f = &fa;
                let cases: [(u32, Vec<u32>); 6] = [
                    (EQ_S, bool_bits(f.clone().equal_elem(v))),
                    (NE_S, bool_bits(f.clone().not_equal_elem(v))),
                    (GT_S, bool_bits(f.clone().greater_elem(v))),
                    (GE_S, bool_bits(f.clone().greater_equal_elem(v))),
                    (LT_S, bool_bits(f.clone().lower_elem(v))),
                    (LE_S, bool_bits(f.clone().lower_equal_elem(v))),
                ];
                for (k, w) in cases {
                    case(
                        op(k, v, 0.0),
                        w,
                        vec![],
                        false,
                        &format!("compare {k:#x} with {v}"),
                    );
                }
            }
            s.free(a).unwrap();
        }
    });
}

#[test]
fn comparisons_and_masks_are_flex_s_with_broadcasts() {
    with_session(|s| {
        for (r, c) in [(37, 70), (64, 96)] {
            let (av, bv, rowv, colv, vv) = (
                values(2, r * c),
                values(3, r * c),
                values(4, c),
                values(5, r),
                values(6, r * c),
            );
            let mv: Vec<u32> = values(7, r * c)
                .iter()
                .map(|x| (x.to_bits() >> 3) & 1)
                .collect();
            let mrow: Vec<u32> = values(8, c)
                .iter()
                .map(|x| (x.to_bits() >> 5) & 1)
                .collect();
            let up = |s: &mut Session<_>, v: &[f32], rr, cc| {
                s.upload_bits(
                    &v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    rr,
                    cc,
                    Elem::F32,
                )
                .unwrap()
            };
            let (a, b) = (up(s, &av, r, c), up(s, &bv, r, c));
            let (row, col, v) = (up(s, &rowv, 1, c), up(s, &colv, r, 1), up(s, &vv, r, c));
            let m = s.upload_bits(&mv, r, c, Elem::Bool).unwrap();
            let mr = s.upload_bits(&mrow, 1, c, Elem::Bool).unwrap();
            let (fa, fb) = (flex(&av, r, c), flex(&bv, r, c));
            let (frow, fcol) = (flex(&rowv, 1, c), flex(&colv, r, 1));
            type Cmp = fn(Tensor<Flex, 2>, Tensor<Flex, 2>) -> Tensor<Flex, 2, Bool>;
            let cmps: [(u32, Cmp); 6] = [
                (EQ, |x, y| x.equal(y)),
                (NE, |x, y| x.not_equal(y)),
                (GT, |x, y| x.greater(y)),
                (GE, |x, y| x.greater_equal(y)),
                (LT, |x, y| x.lower(y)),
                (LE, |x, y| x.lower_equal(y)),
            ];
            for (k, f) in cmps {
                for (bt, bf, bvals, bc, how) in [
                    (&b, fb.clone(), &bv, Broadcast::None, "same shape"),
                    (&row, frow.clone(), &rowv, Broadcast::Row, "a row"),
                    (&col, fcol.clone(), &colv, Broadcast::Col, "a column"),
                ] {
                    let what = format!("compare {k:#x} with {how} [{r}, {c}]");
                    let got = run(
                        s,
                        op(k, 0.0, 0.0),
                        bc,
                        &[&a, bt],
                        &[&av, bvals],
                        r,
                        c,
                        &what,
                    );
                    assert_flex(&got, &bool_bits(f(fa.clone(), bf)), false, &what);
                }
            }
            let mvf: Vec<f32> = mv.iter().map(|&x| f32::from_bits(x)).collect();
            let mrf: Vec<f32> = mrow.iter().map(|&x| f32::from_bits(x)).collect();
            for value in [2.5f32, -0.0, f32::NAN] {
                let what = format!("mask_fill({value}) [{r}, {c}]");
                let got = run(
                    s,
                    op(MASK_FILL, value, 0.0),
                    Broadcast::None,
                    &[&a, &m],
                    &[&av, &mvf],
                    r,
                    c,
                    &what,
                );
                assert_flex(
                    &got,
                    &bits(fa.clone().mask_fill(flex_mask(&mv, r, c), value)),
                    false,
                    &what,
                );
                let what = format!("mask_fill({value}) by a row mask [{r}, {c}]");
                let got = run(
                    s,
                    op(MASK_FILL, value, 0.0),
                    Broadcast::Row,
                    &[&a, &mr],
                    &[&av, &mrf],
                    r,
                    c,
                    &what,
                );
                assert_flex(
                    &got,
                    &bits(fa.clone().mask_fill(flex_mask(&mrow, 1, c), value)),
                    false,
                    &what,
                );
            }
            let what = format!("mask_where [{r}, {c}]");
            let got = run(
                s,
                op(MASK_WHERE, 0.0, 0.0),
                Broadcast::None,
                &[&a, &m, &v],
                &[&av, &mvf, &vv],
                r,
                c,
                &what,
            );
            let want = bits(fa.clone().mask_where(flex_mask(&mv, r, c), flex(&vv, r, c)));
            assert_flex(&got, &want, false, &what);
        }
    });
}
