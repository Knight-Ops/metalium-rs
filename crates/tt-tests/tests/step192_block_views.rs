//! Element-wise ops on block views, read where they lie (`docs/plans/decode-small-ops.md`, S2a).
//!
//! A view that is a block of its buffer -- a gate's columns of a fused `gate|up` projection, the
//! heads of `q` along a projection's row -- was copied into a buffer of its own (`copy_blocks`, or
//! an element repack) before any element-wise op read it: SmolLM2-1.7B's decode step spent 48 block
//! copies a step on SiLU's `gate` and `up` alone. Now an element-wise op on such a view runs on the
//! block itself (`Strided::as_block`, `Session::eltwise_blocks`): each operand's tiles read at its
//! parent's row stride, the result a view of a new buffer with the view's strides.
//!
//! The claims:
//! - unary, binary (both operands blocks, a block and a plain tensor on either side, a row
//!   broadcast) and scalar ops on column blocks -- ragged at the parent's edge or not, more than one
//!   tile row -- queue **no copy and no repack**, and equal the op on the whole parent, sliced, bit
//!   for bit on raw bits (every special value), and burn-flex bit for bit on small integers;
//! - a block of a projection's row laid out as heads (`[1, h, 1, d]`) is one as well;
//! - a block feeding a reduction and a matmul keeps its padding claims (ragged at the parent's
//!   edge);
//! - a block that starts inside a tile is still materialised first (the negative control), and is
//!   still exact.

use burn::tensor::{Slice, Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{device_time, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// `n` datums of raw F32 bits: every fifth one a special value.
fn raw_bits(seed: u64, n: usize) -> Vec<f32> {
    const SPECIAL: [u32; 8] = [
        0x8000_0000, // -0.0
        0x0000_0001, // the least subnormal
        0x7fc1_2345, // a quiet NaN with a payload
        0xff81_0000, // a negative signalling NaN
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7f7f_ffff, // f32::MAX
        0x007f_ffff, // the greatest subnormal
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f32::from_bits(if i % 5 == 0 {
                SPECIAL[(i / 5) % SPECIAL.len()]
            } else {
                // Moderate magnitudes, so products and sums are mostly finite.
                (((s >> 40) as i32 - (1 << 23)) as f32 / 4096.0).to_bits()
            })
        })
        .collect()
}

/// Small integers, so sums and products are exact in any order.
fn ints(seed: u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as u64 * seed + 3) % 9) as f32 - 4.0)
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn on_card<const D: usize>(x: Tensor<TtBackend, D>) -> Tensor<TtBackend, D> {
    let TensorPrimitive::Float(p) = x.clone().into_primitive() else {
        unreachable!()
    };
    p.ensure_resident();
    x
}

/// How many times the server ran a kind of job in this process so far.
fn ran(kind: &str) -> u64 {
    device_time()
        .iter()
        .find(|(k, _, _)| *k == kind)
        .map_or(0, |(_, n, _)| *n)
}

/// Jobs that materialise a view.
fn copies() -> u64 {
    ran("copy_blocks") + ran("repack") + ran("repack_many")
}

fn slices(r: &[(usize, usize)]) -> Vec<Slice> {
    r.iter()
        .map(|&(a, b)| Slice::new(a as isize, Some(b as isize), 1))
        .collect()
}

fn read<const D: usize>(t: Tensor<TtBackend, D>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

/// The attachment's first op asks its engine once whether it keeps tensors on the device: ask
/// before counting jobs.
fn settle(d: &burn_tt::TtDevice) {
    let x = on_card(Tensor::<TtBackend, 2>::from_data([[1.0f32, 2.0]], d));
    let _ = (x.clone() + x).into_data();
}

type Op =
    fn(Tensor<TtBackend, 3>, Tensor<TtBackend, 3>, Tensor<TtBackend, 1>) -> Tensor<TtBackend, 3>;

/// Each op on column blocks `left` and `right` of one `[1, rows, cols]` buffer of raw bits (and on
/// a plain tensor, and a row of the block's width), against the same op on plain tensors holding the
/// same datums -- the blocks read back and uploaded -- so that only where the datums are read from
/// differs: the same bits, and no copy unless `copied` (a block the card cannot read in place).
/// `right` `None` is `left` again.
fn check_blocks(
    d: &burn_tt::TtDevice,
    [rows, cols]: [usize; 2],
    left: [usize; 2],
    right: Option<[usize; 2]>,
    copied: bool,
) {
    let right = right.unwrap_or(left);
    let w = left[1] - left[0];
    assert_eq!(right[1] - right[0], w);
    let x = on_card(Tensor::<TtBackend, 3>::from_data(
        TensorData::new(raw_bits(31, rows * cols), [1, rows, cols]),
        d,
    ));
    let plain = on_card(Tensor::<TtBackend, 3>::from_data(
        TensorData::new(raw_bits(32, rows * w), [1, rows, w]),
        d,
    ));
    let row = on_card(Tensor::<TtBackend, 1>::from_data(
        TensorData::new(raw_bits(33, w), [w]),
        d,
    ));
    let block = |b: [usize; 2]| x.clone().slice(&slices(&[(0, 1), (0, rows), (b[0], b[1])]));
    let plain_of =
        |b: [usize; 2]| on_card(Tensor::<TtBackend, 3>::from_data(block(b).into_data(), d));
    let ops: [(&str, Op); 6] = [
        ("sigmoid", |a, _, _| burn::tensor::activation::sigmoid(a)),
        ("mul_scalar", |a, _, _| a.mul_scalar(-3.0)),
        ("block * block", |a, b, _| a * b),
        ("block - plain", |a, _, _| a.clone() - a.mul_scalar(0.5)),
        ("plain / block", |a, b, _| b / a),
        ("block + row", |a, _, r| a + r.unsqueeze::<3>()),
    ];
    for (what, op) in ops {
        let second = |b: Tensor<TtBackend, 3>| {
            if what == "plain / block" {
                plain.clone()
            } else {
                b
            }
        };
        let want = read(op(plain_of(left), second(plain_of(right)), row.clone()));
        let before = copies();
        let (y, report) =
            burn_tt::with_report(|| op(block(left), second(block(right)), row.clone()));
        assert_native_model(&report);
        let got = read(y);
        let n = copies() - before;
        if copied {
            assert!(
                n > 0,
                "{what} {left:?}: a block inside a tile is materialised"
            );
        } else {
            assert_eq!(
                n, 0,
                "{what} {left:?} {right:?} of [{rows}, {cols}]: no copy\n{report}"
            );
        }
        assert_eq!(
            bits(&got),
            bits(&want),
            "{what} {left:?} {right:?} of [{rows}, {cols}]"
        );
    }
}

#[test]
fn element_wise_ops_read_column_blocks_in_place() {
    with_device(Config::default(), |d| {
        settle(&d);
        // SiLU's gate and up: the two halves of a fused product, one row and many (40, ragged
        // where the buffer is).
        check_blocks(&d, [1, 256], [0, 128], Some([128, 256]), false);
        check_blocks(&d, [40, 192], [0, 96], Some([96, 192]), false);
        // Ragged columns at the buffer's own edge.
        check_blocks(&d, [70, 150], [96, 150], None, false);
        // The negative control: a block starting inside a tile is made plain first.
        check_blocks(&d, [40, 160], [16, 80], Some([80, 144]), true);
    });
}

/// The heads of `q` and `k` along one row of a fused projection (`[1, h, 1, d]` at a stride of `d`
/// columns), as a decode step's RoPE gets them: blocks of the projection, combined in place; and the
/// same against burn-flex on small integers.
#[test]
fn heads_along_a_row_are_blocks() {
    with_device(Config::default(), |d| {
        settle(&d);
        for (h, dh) in [(9, 64), (4, 128), (3, 32)] {
            let width = 3 * h * dh;
            let x = ints(7, width);
            let f = Tensor::<Flex, 3>::from_data(
                TensorData::new(x.clone(), [1, 1, width]),
                &FlexDevice,
            );
            let t = on_card(Tensor::<TtBackend, 3>::from_data(
                TensorData::new(x, [1, 1, width]),
                &d,
            ));
            fn heads<B: burn::tensor::backend::Backend>(
                p: &Tensor<B, 3>,
                i: usize,
                [h, dh]: [usize; 2],
            ) -> Tensor<B, 4> {
                p.clone()
                    .slice([0..1, 0..1, i * h * dh..(i + 1) * h * dh])
                    .reshape([1, 1, h, dh])
                    .swap_dims(1, 2)
            }
            let (fq, fk) = (heads(&f, 0, [h, dh]), heads(&f, 1, [h, dh]));
            let want: Vec<f32> = (fq.clone() * fk.clone() + fq.mul_scalar(2.0))
                .into_data()
                .to_vec()
                .unwrap();
            let before = copies();
            let (y, report) = burn_tt::with_report(|| {
                let (q, k) = (heads(&t, 0, [h, dh]), heads(&t, 1, [h, dh]));
                q.clone() * k + q.mul_scalar(2.0)
            });
            assert_native_model(&report);
            assert_eq!(copies() - before, 0, "h {h} d {dh}: no copy\n{report}");
            let got: Vec<f32> = y.into_data().to_vec().unwrap();
            assert_eq!(bits(&got), bits(&want), "h {h} d {dh}");
        }
    });
}

/// A block ragged at its buffer's edge feeds a row sum and a matmul (whose `K` padding must be
/// zero): the block's padding is its parent's, and the chain equals burn-flex bit for bit on small
/// integers.
#[test]
fn a_ragged_block_feeds_a_reduction_and_a_matmul() {
    with_device(Config::default(), |d| {
        settle(&d);
        let ([rows, cols], at) = ([45, 150], 96);
        let w = cols - at;
        let x = ints(5, rows * cols);
        let m = ints(11, w * 7);
        let f =
            Tensor::<Flex, 3>::from_data(TensorData::new(x.clone(), [1, rows, cols]), &FlexDevice);
        let fm = Tensor::<Flex, 2>::from_data(TensorData::new(m.clone(), [w, 7]), &FlexDevice);
        let t = on_card(Tensor::<TtBackend, 3>::from_data(
            TensorData::new(x, [1, rows, cols]),
            &d,
        ));
        let tm = on_card(Tensor::<TtBackend, 2>::from_data(
            TensorData::new(m, [w, 7]),
            &d,
        ));
        let region = [0..1, 0..rows, at..cols];
        let (fy, ty) = (
            f.slice(region.clone()).mul_scalar(2.0) + 1.0,
            t.slice(region).mul_scalar(2.0) + 1.0,
        );
        let check = |a: Tensor<Flex, 3>, b: Tensor<TtBackend, 3>, what: &str| {
            let want: Vec<f32> = a.into_data().to_vec().unwrap();
            let got: Vec<f32> = b.into_data().to_vec().unwrap();
            assert_eq!(bits(&got), bits(&want), "{what}");
        };
        check(
            fy.clone().sum_dim(2),
            ty.clone().sum_dim(2),
            "sum over the block's columns",
        );
        check(
            fy.clone().sum_dim(1),
            ty.clone().sum_dim(1),
            "sum over its rows",
        );
        check(
            fy.matmul(fm.unsqueeze()),
            ty.matmul(tm.unsqueeze()),
            "a matmul over its columns",
        );
    });
}

/// RoPE's `apply` (`tt_models::rope::Rope::apply`), on tables given as tensors.
fn rope<B: burn::tensor::backend::Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [n, h, t, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..n, 0..h, 0..t, 0..half]);
    let x2 = x.clone().slice([0..n, 0..h, 0..t, half..d]);
    x * cos + Tensor::cat(vec![x2, x1], 3) * sin
}

/// `q` and `k` of a fused `[1, t, (h + 2 kv) d]` projection, split into heads as a Llama layer
/// splits them: views of the projection.
fn qk<B: burn::tensor::backend::Backend>(
    p: &Tensor<B, 3>,
    [t, h, kv, d]: [usize; 4],
) -> [Tensor<B, 4>; 2] {
    let heads = |at: usize, count: usize| {
        p.clone()
            .slice([0..1, 0..t, at..at + count * d])
            .reshape([1, t, count, d])
            .swap_dims(1, 2)
    };
    [heads(0, h), heads(h * d, kv)]
}

fn repacks() -> u64 {
    ran("repack") + ran("repack_many")
}

/// RoPE on the heads of a fused projection, in decode (`t = 1`, the heads along one row) and
/// prefill (`t` rows, ragged), against burn-flex bit for bit on small integers: no element repack
/// where a half is whole tiles (`head_dim` 64 and 128), the table laid out over the heads once and
/// the halves swapped by tile copies; `head_dim` 32 (halves of 16 columns) still repacks.
#[test]
fn rope_on_projection_heads_moves_whole_tiles() {
    with_device(Config::default(), |d| {
        settle(&d);
        for (t, h, kv, dh) in [
            (1, 9, 3, 64),
            (1, 32, 32, 64),
            (1, 4, 1, 128),
            (40, 3, 1, 64),
            (32, 2, 2, 128),
            (1, 5, 5, 32),
        ] {
            let width = (h + 2 * kv) * dh;
            let x = ints(13, t * width);
            let c = ints(17, t * dh);
            let s = ints(19, t * dh);
            let flex = |v: &[f32], shape: Vec<usize>| {
                Tensor::<Flex, 3>::from_data(TensorData::new(v.to_vec(), shape), &FlexDevice)
            };
            let tt = |v: &[f32], shape: Vec<usize>| {
                on_card(Tensor::<TtBackend, 3>::from_data(
                    TensorData::new(v.to_vec(), shape),
                    &d,
                ))
            };
            let table = |v: Tensor<Flex, 3>| v.reshape([1, 1, t, dh]);
            let fp = flex(&x, vec![1, t, width]);
            let (fc, fs) = (
                table(flex(&c, vec![1, t, dh])),
                table(flex(&s, vec![1, t, dh])),
            );
            let [fq, fk] = qk(&fp, [t, h, kv, dh]);
            let want = [rope(fq, fc.clone(), fs.clone()), rope(fk, fc, fs)]
                .map(|y| y.into_data().to_vec::<f32>().unwrap());

            let tp = tt(&x, vec![1, t, width]);
            let ttable = |v: Tensor<TtBackend, 3>| v.reshape([1, 1, t, dh]);
            let (tc, ts) = (
                ttable(tt(&c, vec![1, t, dh])),
                ttable(tt(&s, vec![1, t, dh])),
            );
            let before = repacks();
            let (got, report) = burn_tt::with_report(|| {
                let [tq, tk] = qk(&tp, [t, h, kv, dh]);
                [
                    rope(tq, tc.clone(), ts.clone()),
                    rope(tk, tc.clone(), ts.clone()),
                ]
            });
            assert_native_model(&report);
            let n = repacks() - before;
            let got = got.map(|y| y.into_data().to_vec::<f32>().unwrap());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(bits(g), bits(w), "t {t} h {h} kv {kv} d {dh}");
            }
            if dh % 64 == 0 {
                assert_eq!(n, 0, "t {t} h {h} kv {kv} d {dh}: no repack\n{report}");
            } else {
                assert!(n > 0, "d {dh}: halves of 16 columns are repacked");
            }
        }
    });
}

/// The table laid out over the heads is made once and kept beside it: a second layer's RoPE with
/// the same table queues only its halves' tile copy, not the table's.
#[test]
fn the_laid_out_table_is_kept_for_the_next_layer() {
    with_device(Config::default(), |d| {
        settle(&d);
        let (h, dh) = (4, 64);
        let width = 3 * h * dh;
        let table = |seed| {
            on_card(Tensor::<TtBackend, 3>::from_data(
                TensorData::new(ints(seed, dh), [1, 1, dh]),
                &d,
            ))
            .reshape([1, 1, 1, dh])
        };
        let (cos, sin) = (table(3), table(5));
        let layer = |seed| {
            let p = on_card(Tensor::<TtBackend, 3>::from_data(
                TensorData::new(ints(seed, width), [1, 1, width]),
                &d,
            ));
            let [q, _] = qk(&p, [1, h, h, dh]);
            let before = ran("copy_blocks");
            let y = rope(q, cos.clone(), sin.clone());
            let _ = y.into_data();
            ran("copy_blocks") - before
        };
        assert_eq!(layer(7), 3, "the first layer: the halves and both tables");
        assert_eq!(layer(11), 1, "the next: the halves alone");
    });
}

/// Grouped-query attention with `q` in the projection's layout, regrouped `[kv, groups, d]` as a
/// Llama step folds it (the groups' rows along the buffer's columns, not blocks of it): `q` is made
/// plain once and attention keeps its batched products, the same bits as the same attention on a
/// plain `q` holding the same datums. Watched failing (SmolLM2-135M's step, 23 -> 40 ms, profile
/// `1792172763`): without it, `q` is not a batched operand and attention falls to Burn's
/// composition, an element repack of the keys a group.
#[test]
fn grouped_query_attention_reads_regrouped_heads() {
    use burn::tensor::module::attention;
    use burn::tensor::ops::AttentionModuleOptions;
    with_device(Config::default(), |d| {
        settle(&d);
        let (h, kv, dh, len) = (9, 3, 64, 40);
        let card = |v: Vec<f32>, shape: [usize; 4]| {
            on_card(Tensor::<TtBackend, 4>::from_data(
                TensorData::new(v, shape),
                &d,
            ))
        };
        let x = ints(3, h * dh);
        let (k, v) = (
            card(ints(5, kv * len * dh), [1, kv, len, dh]),
            card(ints(7, kv * len * dh), [1, kv, len, dh]),
        );
        let attend = |q: Tensor<TtBackend, 4>| {
            attention(
                q,
                k.clone(),
                v.clone(),
                None,
                None,
                AttentionModuleOptions {
                    scale: None,
                    softcap: None,
                    is_causal: false,
                },
            )
        };
        // Query head `i g + j` is row `j` of group `i`: the projection's order, as data.
        let want: Vec<f32> = attend(card(x.clone(), [1, kv, h / kv, dh]))
            .into_data()
            .to_vec()
            .unwrap();
        let p = on_card(Tensor::<TtBackend, 3>::from_data(
            TensorData::new(x, [1, 1, h * dh]),
            &d,
        ));
        let before = repacks();
        let (y, report) = burn_tt::with_report(|| {
            let q = p
                .clone()
                .reshape([1, 1, h, dh])
                .swap_dims(1, 2)
                .mul_scalar(2.0)
                .mul_scalar(0.5);
            attend(q.reshape([1, kv, h / kv, dh]))
        });
        let got: Vec<f32> = y.into_data().to_vec().unwrap();
        assert_native_model(&report);
        assert_eq!(
            repacks() - before,
            1,
            "q made plain once, nothing else\n{report}"
        );
        assert_eq!(bits(&got), bits(&want));
    });
}
