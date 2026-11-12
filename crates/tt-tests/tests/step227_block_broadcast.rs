//! A grouped-query mask read as a block (`docs/plans/long-context.md` D3): an element-wise op
//! whose second operand is `[1, 1, R, C]` against `[1, kv_heads, R, C]` (each key/value head's
//! scores, the query heads of a group folded into their rows), `R` whole tiles, reads that
//! operand's tiles again for each head (`tt_kernels::sfpu::ops::Broadcast::Block`: plain runs over
//! its own grid), where it used to be expanded first by a row gather of `kv_heads R` rows
//! (`burn-tt`'s `expanded`), once a layer.
//!
//! The claims:
//! - **Same bits.** `mask_fill`, an addition and Burn's `attention` (its composition's scale and
//!   mask, `SCALE_MASK_FILL`) with the block operand equal the same ops with the operand expanded
//!   to the full shape first, bit for bit, at 2, 4 and 8 heads, `R` 32 and 64, widths 40, 64 and
//!   80 (ragged); on small integers `mask_fill` and the addition also equal burn-flex.
//! - **No row list.** The block runs no `broadcast/expand` gather.
//! - **Only a block is read as one.** A per-head mask row (`[1, kv_heads, 1, C]`, a row of each
//!   head repeated, not a block) and a block of 16 rows (not whole tiles) still take the gather and
//!   still equal burn-flex: read as a block, they would be wrong.
//!
//! Watched failing: the host gate `a_block_broadcast_reads_b_s_tile_rows_again_down_a`
//! (`tt-kernels`) with the wrap one tile row late.

use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Bool, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{TtBackend, TtDevice};
use tt_tests::burn_device::{with_device, Config};

/// `n` floats in `[-scale, scale)`, integers if `whole`.
fn floats(seed: u64, n: usize, scale: f32, whole: bool) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale;
            if whole {
                x.round()
            } else {
                x
            }
        })
        .collect()
}

fn bools(seed: u64, n: usize) -> Vec<bool> {
    floats(seed, n, 1.0, false)
        .into_iter()
        .map(|x| x > 0.2)
        .collect()
}

fn bits<B: burn::tensor::backend::Backend>(t: Tensor<B, 4>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn tt(v: Vec<f32>, shape: [usize; 4], d: &TtDevice) -> Tensor<TtBackend, 4> {
    Tensor::from_data(TensorData::new(v, shape), d)
}

fn tt_mask(v: Vec<bool>, shape: [usize; 4], d: &TtDevice) -> Tensor<TtBackend, 4, Bool> {
    Tensor::from_data(TensorData::new(v, shape), d)
}

fn flex(v: Vec<f32>, shape: [usize; 4]) -> Tensor<Flex, 4> {
    Tensor::from_data(TensorData::new(v, shape), &FlexDevice)
}

fn flex_mask(v: Vec<bool>, shape: [usize; 4]) -> Tensor<Flex, 4, Bool> {
    Tensor::from_data(TensorData::new(v, shape), &FlexDevice)
}

/// Whether `f` ran a broadcast's row gather.
fn expands<T>(f: impl FnOnce() -> T) -> (T, bool) {
    let (out, report) = burn_tt::with_report(f);
    let expanded = report.op("broadcast/expand").is_some_and(|s| s.jobs > 0);
    (out, expanded)
}

const SHAPES: [(usize, usize, usize); 3] = [(2, 32, 80), (4, 64, 64), (8, 32, 40)];

#[test]
fn a_grouped_query_mask_is_read_as_a_block_with_the_same_bits() {
    with_device(Config::default(), |d| {
        for (i, (kv, r, c)) in SHAPES.into_iter().enumerate() {
            let what = format!("{kv} heads of [{r}, {c}]");
            let seed = 17 * i as u64;
            let full = [1, kv, r, c];
            let block = [1, 1, r, c];
            for whole in [true, false] {
                let x = floats(seed, kv * r * c, 37.0, whole);
                let y = floats(seed + 1, r * c, 37.0, whole);
                let m = bools(seed + 2, r * c);
                let fill = -1.0e9;

                let (got, gathered) = expands(|| {
                    bits(tt(x.clone(), full, &d).mask_fill(tt_mask(m.clone(), block, &d), fill))
                });
                assert!(!gathered, "{what}: mask_fill gathered its mask");
                let want = bits(
                    tt(x.clone(), full, &d)
                        .mask_fill(tt_mask(m.clone(), block, &d).expand(full), fill),
                );
                assert_eq!(got, want, "{what}: mask_fill against the expanded mask");

                let (sum, gathered) =
                    expands(|| bits(tt(x.clone(), full, &d) + tt(y.clone(), block, &d)));
                assert!(!gathered, "{what}: the addition gathered its operand");
                let expanded =
                    bits(tt(x.clone(), full, &d) + tt(y.clone(), block, &d).expand(full));
                assert_eq!(
                    sum, expanded,
                    "{what}: addition against the expanded operand"
                );

                if whole {
                    let host =
                        bits(flex(x.clone(), full).mask_fill(flex_mask(m.clone(), block), fill));
                    assert_eq!(got, host, "{what}: mask_fill against burn-flex");
                    let host = bits(flex(x.clone(), full) + flex(y.clone(), block));
                    assert_eq!(sum, host, "{what}: addition against burn-flex");
                }
            }

            // Burn's attention over each key/value head, the group's query rows folded in: its
            // composition's scale and mask reads the mask as a block.
            let d_head = 64;
            let q = floats(seed + 3, kv * r * d_head, 1.0, false);
            let k = floats(seed + 4, kv * c * d_head, 1.0, false);
            let v = floats(seed + 5, kv * c * d_head, 1.0, false);
            let m = bools(seed + 6, r * c);
            let options = AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: false,
            };
            let run = |mask: Tensor<TtBackend, 4, Bool>| {
                bits(attention(
                    tt(q.clone(), [1, kv, r, d_head], &d),
                    tt(k.clone(), [1, kv, c, d_head], &d),
                    tt(v.clone(), [1, kv, c, d_head], &d),
                    Some(mask),
                    None,
                    options,
                ))
            };
            // The composition (its scale and mask one op), then the scale, mask and softmax as
            // one op (`Session::softmax_rows`): each reads the mask as a block, and they agree.
            burn_tt::set_softmax_rows(false);
            let (got, report) = burn_tt::with_report(|| run(tt_mask(m.clone(), block, &d)));
            assert!(
                report.op("attention/scale").is_some_and(|s| s.jobs > 0),
                "{what}: attention ran as its composition"
            );
            assert!(
                report.op("broadcast/expand").is_none(),
                "{what}: attention gathered its mask"
            );
            let want = run(tt_mask(m.clone(), block, &d).expand(full));
            assert_eq!(got, want, "{what}: attention against the expanded mask");
            burn_tt::set_softmax_rows(true);
            let (fused, report) = burn_tt::with_report(|| run(tt_mask(m.clone(), block, &d)));
            // Whole tiles, the scale, mask and softmax are one op; ragged, the composition.
            assert_eq!(
                report.op("attention/scale").is_none(),
                c % 32 == 0,
                "{what}: the softmax as one op where the scores are whole tiles"
            );
            assert!(
                report.op("broadcast/expand").is_none(),
                "{what}: attention gathered its mask"
            );
            assert_eq!(
                fused, want,
                "{what}: the softmax as one op against the composition"
            );
        }
    });
}

/// Operands that broadcast but are not a block of whole tile rows: still expanded by the gather,
/// and still right.
#[test]
fn operands_that_are_not_blocks_are_still_expanded() {
    with_device(Config::default(), |d| {
        let (kv, r, c) = (4, 32, 48);
        let full = [1, kv, r, c];
        let fill = -7.0;
        let x = floats(5, kv * r * c, 37.0, true);
        for (what, shape) in [
            ("a row of each head", [1, kv, 1, c]),
            ("a block of 16 rows", [1, 1, 16, c]),
        ] {
            let full = if shape[2] == 16 { [1, kv, 16, c] } else { full };
            let x = x[..full.iter().product::<usize>()].to_vec();
            let m = bools(9, shape.iter().product());
            let (got, gathered) = expands(|| {
                bits(tt(x.clone(), full, &d).mask_fill(tt_mask(m.clone(), shape, &d), fill))
            });
            assert!(gathered, "{what}: read without the gather");
            let host = bits(flex(x.clone(), full).mask_fill(flex_mask(m.clone(), shape), fill));
            assert_eq!(got, host, "{what}: against burn-flex");
        }
    });
}
