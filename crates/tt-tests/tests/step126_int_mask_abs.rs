//! Gate: `int/bool_mask_{where,fill}`, `int_abs`,
//! `int_cast`, and the Burn defaults they unblock (`int_clamp*`, `int_sign`, `int_max_abs*`).
//!
//! The selects and `int_abs` are raw-word SFPU programs (`kind_sfpu::INT_MASK_WHERE`,
//! `INT_MASK_FILL`, `BOOL_*`, `INT_ABS`): no integer passes through an `f32`. Held here against
//! independent host oracles (explicit wrapping `i32` arithmetic, exact bit equality), with Flex
//! as a second reference, and at ragged shapes whose results feed `int_sum_dim`, which reads
//! through the result's declared padding:
//!
//! - fills of `2^24 + 1` and `i32::MIN`, which an `f32` scalar cannot carry (`16777217` rounds
//!   to `16777216`; `i32::MIN` as a float is `0xcf000000`, and a `-0.0` zero test would call its
//!   bits zero);
//! - selects that keep NaN payloads (`0x7fc00001`), the sign bit alone and the denormal one;
//! - masks that are true in the padding (`x >= 0` of zero padding), whose results must not claim
//!   zero padding;
//! - row and column broadcasts of the mask and of the tensor;
//! - `int_abs` of `i32::MIN` stays `i32::MIN` (Flex's `wrapping_abs`);
//! - nothing downloads, and changed-input trace replay (one and two tiles, and ragged) matches
//!   the host.
//!
//! Negative controls (mutate, run, watch fail, restore): the integer fill through the `f32`
//! scalar path; swapped select branches.

use burn::tensor::{Bool, Int, IntDType, Scalar, Tensor, TensorData};
use burn_tensor::ops::IntTensorOps;
use burn_tt::{tensor_traffic, InputPayload, TracedInference, TtBackend, TtDevice};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

const SPECIALS: [i32; 12] = [
    i32::MIN,
    i32::MAX,
    -1,
    0,
    1,
    16_777_217,
    -16_777_217,
    0x7fc0_0001,
    0x8000_0001u32 as i32,
    0x007f_ffff,
    0x7f80_0000,
    i32::MIN + 1,
];

fn ints(n: usize, seed: u32) -> Vec<i32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if i % 3 == 0 {
                SPECIALS[(i / 3 + seed as usize) % SPECIALS.len()]
            } else {
                s as i32
            }
        })
        .collect()
}

fn bools(n: usize, seed: u32) -> Vec<bool> {
    ints(n, seed ^ 0x55)
        .iter()
        .map(|x| x.count_ones() & 1 == 1)
        .collect()
}

fn int_tensor<const D: usize>(
    d: &TtDevice,
    v: &[i32],
    shape: [usize; D],
) -> Tensor<TtBackend, D, Int> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), d).to_device(d)
}

fn bool_tensor<const D: usize>(
    d: &TtDevice,
    v: &[bool],
    shape: [usize; D],
) -> Tensor<TtBackend, D, Bool> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), d).to_device(d)
}

fn ints_of<const D: usize>(t: Tensor<TtBackend, D, Int>) -> Vec<i32> {
    t.into_data().to_vec::<i32>().unwrap()
}

fn resident<const D: usize>(t: &Tensor<TtBackend, D, Int>) -> bool {
    t.clone().into_primitive().computed_on_device()
}

/// Wrapping sums along one axis of a `[rows, cols]` matrix.
fn sum_rows(v: &[i32], rows: usize, cols: usize) -> Vec<i32> {
    (0..cols)
        .map(|c| (0..rows).fold(0i32, |a, r| a.wrapping_add(v[r * cols + c])))
        .collect()
}

fn sum_cols(v: &[i32], rows: usize, cols: usize) -> Vec<i32> {
    (0..rows)
        .map(|r| (0..cols).fold(0i32, |a, c| a.wrapping_add(v[r * cols + c])))
        .collect()
}

/// A result of shape `[rows, cols]`, read three ways that each go through its padding: the
/// whole sum and the sums along each axis, against the host's wrapping sums.
#[track_caller]
fn check_ints(got: Tensor<TtBackend, 2, Int>, want: &[i32], rows: usize, cols: usize, what: &str) {
    assert!(resident(&got), "{what}: not on the device");
    assert_eq!(
        ints_of(got.clone().sum_dim(0)),
        sum_rows(want, rows, cols),
        "{what}: sum_dim(0)"
    );
    assert_eq!(
        ints_of(got.clone().sum_dim(1)),
        sum_cols(want, rows, cols),
        "{what}: sum_dim(1)"
    );
    assert_eq!(
        ints_of(got.clone().sum()),
        vec![want.iter().fold(0i32, |a, &b| a.wrapping_add(b))],
        "{what}: sum"
    );
    assert_eq!(ints_of(got), want, "{what}: values");
}

#[test]
fn int_mask_fill_carries_every_32_bit_word() {
    with_device(Config::default(), |d| {
        for [rows, cols] in [[33usize, 65], [1, 5], [32, 32], [64, 33]] {
            let n = rows * cols;
            let data = ints(n, 3 + n as u32);
            let x = int_tensor(&d, &data, [rows, cols]);
            let random = bools(n, 7);
            let masks: [(&str, Vec<bool>); 4] = [
                ("random", random),
                ("all true", vec![true; n]),
                ("all false", vec![false; n]),
                // True in the padding too: `0 >= 0` of the zero padding.
                ("x >= 0", data.iter().map(|&v| v >= 0).collect()),
            ];
            for value in SPECIALS {
                for (name, mask) in &masks {
                    let m = if *name == "x >= 0" {
                        x.clone().greater_equal_elem(0)
                    } else {
                        bool_tensor(&d, mask, [rows, cols])
                    };
                    let before = tensor_traffic();
                    let (out, report) =
                        burn_tt::with_report(|| x.clone().mask_fill(m.clone(), value));
                    assert_native_model(&report);
                    assert_eq!(tensor_traffic().downloads, before.downloads);
                    let want: Vec<i32> = data
                        .iter()
                        .zip(mask)
                        .map(|(&v, &m)| if m { value } else { v })
                        .collect();
                    check_ints(
                        out,
                        &want,
                        rows,
                        cols,
                        &format!("fill {value:#x} mask {name} [{rows}, {cols}]"),
                    );
                }
            }
        }
    });
}

/// The mask broadcasts to the tensor (a row or a column) and the tensor to the mask.
#[test]
fn int_mask_fill_and_where_broadcast() {
    with_device(Config::default(), |d| {
        let (rows, cols) = (33usize, 65usize);
        let data = ints(rows * cols, 11);
        let x = int_tensor(&d, &data, [rows, cols]).into_primitive();
        let value: Scalar = 16_777_217i32.into();
        let wrap = |p| Tensor::<TtBackend, 2, Int>::from_primitive(p);
        let mrow = bools(cols, 13);
        let mcol = bools(rows, 17);
        let row = bool_tensor(&d, &mrow, [1, cols]).into_primitive();
        let col = bool_tensor(&d, &mcol, [rows, 1]).into_primitive();
        let out = wrap(TtBackend::int_mask_fill(x.clone(), row, value));
        let want: Vec<i32> = (0..rows * cols)
            .map(|i| if mrow[i % cols] { 16_777_217 } else { data[i] })
            .collect();
        check_ints(out, &want, rows, cols, "row-broadcast mask");
        let out = wrap(TtBackend::int_mask_fill(x.clone(), col, value));
        let want: Vec<i32> = (0..rows * cols)
            .map(|i| if mcol[i / cols] { 16_777_217 } else { data[i] })
            .collect();
        check_ints(out, &want, rows, cols, "column-broadcast mask");
        // The tensor is the column; the mask the whole matrix.
        let column = ints(rows, 19);
        let full = bools(rows * cols, 23);
        let t = int_tensor(&d, &column, [rows, 1]).into_primitive();
        let m = bool_tensor(&d, &full, [rows, cols]).into_primitive();
        let out = wrap(TtBackend::int_mask_fill(t, m, i32::MIN.into()));
        let want: Vec<i32> = (0..rows * cols)
            .map(|i| if full[i] { i32::MIN } else { column[i / cols] })
            .collect();
        check_ints(out, &want, rows, cols, "tensor broadcast to the mask");
    });
}

#[test]
fn int_mask_where_selects_raw_words() {
    with_device(Config::default(), |d| {
        for [rows, cols] in [[33usize, 65], [32, 32], [1, 7], [64, 33]] {
            let n = rows * cols;
            let (a, b) = (ints(n, 29), ints(n, 31));
            let (x, v) = (
                int_tensor(&d, &a, [rows, cols]),
                int_tensor(&d, &b, [rows, cols]),
            );
            for (name, mask) in [
                ("random", bools(n, 37)),
                ("all true", vec![true; n]),
                ("all false", vec![false; n]),
                ("x >= 0", a.iter().map(|&v| v >= 0).collect()),
            ] {
                let m = if name == "x >= 0" {
                    x.clone().greater_equal_elem(0)
                } else {
                    bool_tensor(&d, &mask, [rows, cols])
                };
                let before = tensor_traffic();
                let (out, report) =
                    burn_tt::with_report(|| x.clone().mask_where(m.clone(), v.clone()));
                assert_native_model(&report);
                assert_eq!(tensor_traffic().downloads, before.downloads);
                let want: Vec<i32> = (0..n).map(|i| if mask[i] { b[i] } else { a[i] }).collect();
                check_ints(
                    out,
                    &want,
                    rows,
                    cols,
                    &format!("where {name} [{rows}, {cols}]"),
                );
            }
        }
        // Operands that broadcast to one shape, through the primitive.
        let (rows, cols) = (33usize, 65usize);
        let a = ints(rows * cols, 41);
        let row = ints(cols, 43);
        let mask = bools(rows, 47);
        let out = TtBackend::int_mask_where(
            int_tensor(&d, &a, [rows, cols]).into_primitive(),
            bool_tensor(&d, &mask, [rows, 1]).into_primitive(),
            int_tensor(&d, &row, [1, cols]).into_primitive(),
        );
        let want: Vec<i32> = (0..rows * cols)
            .map(|i| if mask[i / cols] { row[i % cols] } else { a[i] })
            .collect();
        check_ints(
            Tensor::<TtBackend, 2, Int>::from_primitive(out),
            &want,
            rows,
            cols,
            "broadcast where",
        );
    });
}

#[test]
fn bool_mask_fill_and_where_stay_canonical() {
    with_device(Config::default(), |d| {
        for [rows, cols] in [[33usize, 65], [32, 32], [1, 5]] {
            let n = rows * cols;
            let (a, b) = (bools(n, 53), bools(n, 59));
            let x = bool_tensor(&d, &a, [rows, cols]);
            let v = bool_tensor(&d, &b, [rows, cols]);
            for (name, mask) in [
                ("random", bools(n, 61)),
                ("all true", vec![true; n]),
                ("all false", vec![false; n]),
                ("x", a.clone()),
            ] {
                let m = bool_tensor(&d, &mask, [rows, cols]);
                let before = tensor_traffic();
                let ((filled_t, filled_f, selected), report) = burn_tt::with_report(|| {
                    (
                        x.clone().mask_fill(m.clone(), true),
                        x.clone().mask_fill(m.clone(), false),
                        x.clone().mask_where(m.clone(), v.clone()),
                    )
                });
                assert_native_model(&report);
                assert_eq!(tensor_traffic().downloads, before.downloads);
                let host = |f: &dyn Fn(usize) -> bool| -> Vec<bool> { (0..n).map(f).collect() };
                for (got, want, what) in [
                    (filled_t, host(&|i| mask[i] || a[i]), "fill true"),
                    (filled_f, host(&|i| !mask[i] && a[i]), "fill false"),
                    (
                        selected,
                        host(&|i| if mask[i] { b[i] } else { a[i] }),
                        "where",
                    ),
                ] {
                    assert!(got.clone().into_primitive().computed_on_device());
                    // The device word is canonical 0/1: its integer sums are the counts, and
                    // a reduction reads the padding.
                    let as_int: Vec<i32> = want.iter().map(|&b| i32::from(b)).collect();
                    let ctx = format!("{what} {name} [{rows}, {cols}]");
                    assert_eq!(
                        ints_of(got.clone().int().sum_dim(1)),
                        sum_cols(&as_int, rows, cols),
                        "{ctx}: sum_dim(1)"
                    );
                    assert_eq!(
                        ints_of(got.clone().int().sum_dim(0)),
                        sum_rows(&as_int, rows, cols),
                        "{ctx}: sum_dim(0)"
                    );
                    assert_eq!(got.into_data().to_vec::<bool>().unwrap(), want, "{ctx}");
                }
            }
        }
    });
}

#[test]
fn int_abs_wraps_and_feeds_reductions() {
    with_device(Config::default(), |d| {
        for [rows, cols] in [[33usize, 65], [32, 32], [1, 1], [64, 33]] {
            let n = rows * cols;
            let data = ints(n, 67 + n as u32);
            let x = int_tensor(&d, &data, [rows, cols]);
            let before = tensor_traffic();
            let (out, report) = burn_tt::with_report(|| x.clone().abs());
            assert_native_model(&report);
            assert_eq!(tensor_traffic().downloads, before.downloads);
            let want: Vec<i32> = data.iter().map(|v| v.wrapping_abs()).collect();
            check_ints(out, &want, rows, cols, &format!("abs [{rows}, {cols}]"));
        }
        // i32::MIN alone, and a parent view: |permute|.
        let data = [i32::MIN; 40 * 33];
        let x = int_tensor(&d, &data, [40, 33]);
        assert!(ints_of(x.clone().abs()).iter().all(|&v| v == i32::MIN));
        let data = ints(3 * 33 * 65, 71);
        let x = int_tensor(&d, &data, [3, 33, 65]);
        let out = x.permute([2, 0, 1]).abs();
        let want: Vec<i32> = (0..65 * 3 * 33)
            .map(|i| {
                let (a, b, c) = (i / (3 * 33), i / 33 % 3, i % 33);
                data[(b * 33 + c) * 65 + a].wrapping_abs()
            })
            .collect();
        check_ints(
            out.reshape([65 * 3, 33]),
            &want,
            65 * 3,
            33,
            "abs of a permuted view",
        );
    });
}

/// Burn's default compositions are now native: they fail today only because `int_mask_fill`
/// and `int_abs` were stubs. Each against the host, nothing downloaded.
#[test]
fn burn_defaults_over_the_new_primitives_are_resident() {
    with_device(Config::default(), |d| {
        let (rows, cols) = (33usize, 65usize);
        let data = ints(rows * cols, 73);
        let x = int_tensor(&d, &data, [rows, cols]);
        let before = tensor_traffic();
        let (results, report) = burn_tt::with_report(|| {
            (
                x.clone().clamp(-1000, 1_000_000),
                x.clone().clamp_min(-5),
                x.clone().clamp_max(16_777_217),
                x.clone().sign(),
                x.clone().max_abs(),
                x.clone().max_abs_dim(1),
                x.clone().max_abs_dim(0),
            )
        });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let (clamp, cmin, cmax, sign, max_abs, mad1, mad0) = results;
        let host = |f: &dyn Fn(i32) -> i32| -> Vec<i32> { data.iter().map(|&v| f(v)).collect() };
        check_ints(
            clamp,
            &host(&|v| v.clamp(-1000, 1_000_000)),
            rows,
            cols,
            "clamp",
        );
        check_ints(cmin, &host(&|v| v.max(-5)), rows, cols, "clamp_min");
        check_ints(cmax, &host(&|v| v.min(16_777_217)), rows, cols, "clamp_max");
        check_ints(sign, &host(&|v| v.signum()), rows, cols, "sign");
        let abs: Vec<i32> = data.iter().map(|v| v.wrapping_abs()).collect();
        assert_eq!(
            ints_of(max_abs),
            vec![*abs.iter().max().unwrap()],
            "max_abs"
        );
        assert_eq!(
            ints_of(mad1),
            (0..rows)
                .map(|r| *abs[r * cols..(r + 1) * cols].iter().max().unwrap())
                .collect::<Vec<_>>(),
            "max_abs_dim(1)"
        );
        assert_eq!(
            ints_of(mad0),
            (0..cols)
                .map(|c| (0..rows).map(|r| abs[r * cols + c]).max().unwrap())
                .collect::<Vec<_>>(),
            "max_abs_dim(0)"
        );
    });
}

/// The same Burn programs on Flex, bit for bit: a second reference for what each op means.
#[test]
fn the_new_ops_match_flex() {
    use burn::tensor::backend::Backend;
    use burn_flex::{Flex, FlexDevice};
    fn program<B: Backend>(
        d: &B::Device,
        a: &[i32],
        b: &[i32],
        m: &[bool],
        shape: [usize; 2],
    ) -> Vec<Vec<i32>> {
        let x = Tensor::<B, 2, Int>::from_data(TensorData::new(a.to_vec(), shape), d);
        let v = Tensor::<B, 2, Int>::from_data(TensorData::new(b.to_vec(), shape), d);
        let mask = Tensor::<B, 2, Bool>::from_data(TensorData::new(m.to_vec(), shape), d);
        let flat = |t: Tensor<B, 2, Int>| t.into_data().to_vec::<i32>().unwrap();
        let bits = |t: Tensor<B, 2, Bool>| {
            t.into_data()
                .to_vec::<bool>()
                .unwrap()
                .into_iter()
                .map(i32::from)
                .collect::<Vec<_>>()
        };
        vec![
            flat(x.clone().mask_fill(mask.clone(), 16_777_217)),
            flat(x.clone().mask_fill(mask.clone(), i32::MIN)),
            flat(x.clone().mask_where(mask.clone(), v.clone())),
            flat(x.clone().abs()),
            flat(x.clone().sign()),
            flat(x.clone().clamp(-1000, 1_000_000)),
            flat(x.clone().clamp_min(-5)),
            flat(x.clone().clamp_max(16_777_217)),
            flat(x.clone().flip([0isize, 1])),
            flat(x.clone().permute([1, 0])),
            bits(mask.clone().mask_fill(mask.clone().bool_not(), true)),
            bits(mask.clone().mask_fill(mask.clone().bool_not(), false)),
            bits(
                mask.clone()
                    .mask_where(mask.clone().bool_not(), mask.clone()),
            ),
        ]
    }
    with_device(Config::default(), |d| {
        for shape in [[33usize, 65], [32, 32], [5, 7]] {
            let n = shape[0] * shape[1];
            let (a, b, m) = (ints(n, 79), ints(n, 83), bools(n, 89));
            let got = program::<TtBackend>(&d, &a, &b, &m, shape);
            let want = program::<Flex>(&FlexDevice, &a, &b, &m, shape);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "op {i} at {shape:?}");
            }
        }
    });
}

#[test]
fn int_cast_is_identity_for_i32_and_refuses_other_widths_by_name() {
    // Run in the forked child with the other gates: a plain in-process test would hold locks
    // across their forks. Refusal inspects metadata only.
    with_device(Config::default(), |d| {
        let t = || {
            Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3, 4], [2, 2]), &d)
                .into_primitive()
        };
        let same = TtBackend::int_cast(t(), IntDType::I32);
        assert_eq!(
            Tensor::<TtBackend, 2, Int>::from_primitive(same)
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            vec![1, 2, 3, 4]
        );
        for dtype in [IntDType::I64, IntDType::I16, IntDType::I8, IntDType::U8] {
            let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = TtBackend::int_cast(t(), dtype);
            }))
            .unwrap_err();
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .expect("a refusal carries a message");
            assert!(
                message.contains("unsupported operation int_cast")
                    && message.contains("I32")
                    && message.contains(&format!("{:?}", burn::tensor::DType::from(dtype))),
                "{message}"
            );
        }
    });
}

fn bits(v: &[i32]) -> Vec<u32> {
    v.iter().map(|&x| x as u32).collect()
}

fn bool_bits(v: &[bool]) -> Vec<u32> {
    v.iter().map(|&b| u32::from(b)).collect()
}

/// A trace of `mask_fill`, `mask_where` and `abs` over int and bool inputs, replayed with new
/// inputs at one tile, two tiles and a ragged shape: each replay is the host's answer on the new
/// data (and the same ops run fresh on the card).
#[test]
fn traced_selects_and_abs_replay_on_changed_inputs() {
    with_device(Config::default(), |d| {
        for [rows, cols] in [[32usize, 32], [64, 32], [33, 17]] {
            let n = rows * cols;
            let x = int_tensor(&d, &ints(n, 97), [rows, cols]);
            let v = int_tensor(&d, &ints(n, 101), [rows, cols]);
            let m = bool_tensor(&d, &bools(n, 103), [rows, cols]);
            let (xp, vp, mp) = (
                x.clone().into_primitive(),
                v.clone().into_primitive(),
                m.clone().into_primitive(),
            );
            let (trace, first) = TracedInference::capture(&[&xp, &vp, &mp], || {
                vec![
                    x.clone().mask_fill(m.clone(), 16_777_217).into_primitive(),
                    x.clone().mask_fill(m.clone(), i32::MIN).into_primitive(),
                    x.clone().mask_where(m.clone(), v.clone()).into_primitive(),
                    x.clone().abs().into_primitive(),
                    x.clone()
                        .greater_equal_elem(0)
                        .mask_fill(m.clone(), false)
                        .into_primitive(),
                ]
            })
            .unwrap();
            assert_eq!(first.len(), 5);
            let uploads = tensor_traffic().uploads;
            for round in 0..3u32 {
                let (a, b, mask) = (
                    ints(n, 200 + round),
                    ints(n, 300 + round),
                    bools(n, 400 + round),
                );
                let out = trace
                    .run(vec![
                        InputPayload::Bits(bits(&a)),
                        InputPayload::Bits(bits(&b)),
                        InputPayload::Bits(bool_bits(&mask)),
                    ])
                    .unwrap();
                let pick = |k: usize| out[k].as_bits().unwrap().to_vec();
                let host = |f: &dyn Fn(usize) -> i32| bits(&(0..n).map(f).collect::<Vec<_>>());
                assert_eq!(
                    pick(0),
                    host(&|i| if mask[i] { 16_777_217 } else { a[i] }),
                    "fill 2^24+1, [{rows}, {cols}] round {round}"
                );
                assert_eq!(
                    pick(1),
                    host(&|i| if mask[i] { i32::MIN } else { a[i] }),
                    "fill MIN, [{rows}, {cols}] round {round}"
                );
                assert_eq!(
                    pick(2),
                    host(&|i| if mask[i] { b[i] } else { a[i] }),
                    "where, [{rows}, {cols}] round {round}"
                );
                assert_eq!(
                    pick(3),
                    host(&|i| a[i].wrapping_abs()),
                    "abs, [{rows}, {cols}] round {round}"
                );
                assert_eq!(
                    pick(4),
                    host(&|i| i32::from(!mask[i] && a[i] >= 0)),
                    "bool fill, [{rows}, {cols}] round {round}"
                );
                // The same ops run fresh on the new inputs agree with the replay.
                let (fa, fb, fm) = (
                    int_tensor(&d, &a, [rows, cols]),
                    int_tensor(&d, &b, [rows, cols]),
                    bool_tensor(&d, &mask, [rows, cols]),
                );
                assert_eq!(
                    bits(&ints_of(fa.clone().mask_fill(fm.clone(), 16_777_217))),
                    pick(0)
                );
                assert_eq!(bits(&ints_of(fa.clone().mask_where(fm, fb))), pick(2));
                assert_eq!(bits(&ints_of(fa.abs())), pick(3));
            }
            let _ = uploads;
        }
    });
}

fn with_session(f: impl FnOnce(&mut tt_tests::backend::Sess<'_>)) {
    tt_tests::backend::with_session(
        tt_kernels::session::TileChoice::Exactly(
            tt_tests::backend::GATE_TILE.0,
            tt_tests::backend::GATE_TILE.1,
        ),
        f,
    );
}

/// What each op says its padding holds, it holds: a claimed `Pad::Zero` is checked against the
/// raw tiles. The masks include one that is true in the padding (`0 >= 0`), where a fill of a
/// nonzero word -- `i32::MIN` in particular, which a float zero test (`-0.0`) would call zero --
/// must not claim zero.
#[test]
fn padding_claims_hold_against_the_raw_tiles() {
    use tt_kernels::sfpu::ops::kind_sfpu::*;
    use tt_kernels::tensor::{Elem, Eltwise, Pad};
    with_session(|s| {
        let (r, c) = (37usize, 70usize);
        let n = r * c;
        let a: Vec<u32> = ints(n, 5).iter().map(|&x| x as u32).collect();
        let v: Vec<u32> = ints(n, 9).iter().map(|&x| x as u32).collect();
        let m: Vec<u32> = bools(n, 13).iter().map(|&b| u32::from(b)).collect();
        let x = s.upload_bits(&a, r, c, Elem::I32).unwrap();
        let y = s.upload_bits(&v, r, c, Elem::I32).unwrap();
        let mask = s.upload_bits(&m, r, c, Elem::Bool).unwrap();
        let ba = s.upload_bits(&m, r, c, Elem::Bool).unwrap();
        let reversed: Vec<u32> = m.iter().rev().copied().collect();
        let bv = s.upload_bits(&reversed, r, c, Elem::Bool).unwrap();
        // True in the padding: the zero padding's `0 >= 0`.
        let ge = Eltwise {
            kind: INT_GE_S,
            scalar: f32::from_bits(0),
            scalar2: 0.0,
        };
        let wide = s.eltwise(ge, &x, None).unwrap();
        let cols = 32 * c.div_ceil(32);
        let in_padding = |k: usize| k / cols >= r || k % cols >= c;
        let raw = s.download_padded(&wide).unwrap();
        assert!(
            raw.iter()
                .enumerate()
                .any(|(k, w)| in_padding(k) && w.to_bits() != 0),
            "the mask is not true in the padding: the control has nothing to test"
        );
        let op = |kind, scalar: u32| Eltwise {
            kind,
            scalar: f32::from_bits(scalar),
            scalar2: 0.0,
        };
        let mut claims = Vec::new();
        for (name, m) in [("zero-padded mask", &mask), ("padding-true mask", &wide)] {
            for fill in [0, 1, i32::MIN as u32, 16_777_217, 0x7fc0_0001] {
                let out = s.eltwise(op(INT_MASK_FILL, fill), &x, Some(m)).unwrap();
                claims.push((format!("INT_MASK_FILL {fill:#x} {name}"), out));
            }
            let out = s
                .eltwise3(op(INT_MASK_WHERE, 0), &x, Some(m), Some(&y))
                .unwrap();
            claims.push((format!("INT_MASK_WHERE {name}"), out));
            for fill in [0, 1] {
                let out = s.eltwise(op(BOOL_MASK_FILL, fill), &ba, Some(m)).unwrap();
                claims.push((format!("BOOL_MASK_FILL {fill} {name}"), out));
            }
            let out = s
                .eltwise3(op(BOOL_MASK_WHERE, 0), &ba, Some(m), Some(&bv))
                .unwrap();
            claims.push((format!("BOOL_MASK_WHERE {name}"), out));
        }
        claims.push((
            "INT_ABS".into(),
            s.eltwise(op(INT_ABS, 0), &x, None).unwrap(),
        ));
        let mut zero_claims = 0;
        for (name, out) in claims {
            if out.pad() == Pad::Zero {
                zero_claims += 1;
                let raw = s.download_padded(&out).unwrap();
                assert!(
                    raw.iter()
                        .enumerate()
                        .all(|(k, w)| !in_padding(k) || w.to_bits() == 0),
                    "{name}: claimed zero padding, and it is not"
                );
            }
        }
        assert!(
            zero_claims > 0,
            "no op claimed zero padding: nothing was checked"
        );
    });
}
