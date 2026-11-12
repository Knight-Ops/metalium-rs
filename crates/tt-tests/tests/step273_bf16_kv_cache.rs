//! A BF16 KV cache (roadmap W1, ADR 0006's performance contract): its keys and values stored BF16,
//! half the bytes an attention step reads, written and read where they lie.
//!
//! **Attention over BF16 keys and values is the attention of their widened values, bit for bit.**
//! An F32 query over BF16 keys and values gives an F32 result (`burn_tt::attention_wide`), each
//! product reading the cache as it lies (`SrcRoute::Edges`: a BF16 edge in `Src` is its widened
//! FP32 edge, gate `step270`), the keys' transpose made by the unpacker (`step272`). Against the
//! same attention of F32 keys and values holding the BF16 values widened on the host, on the
//! shapes a decode step attends with (`step209`'s: a query row a head read from a projection's
//! row, grouped query rows folded a key/value head, keys and values read from a position-major
//! cache, head_dim 32, 64 and 128) and on both paths: `attention_rows` (one op a step) and the
//! composition (`q k^T`, the softmax, `P V`), with a causal prompt's composition too. 1,024
//! positions at head_dim 64 take `attention_rows` with BF16 keys and values and not with F32
//! ones (L1), which the comparison spans. The masked positions hold NaN, infinities and the
//! largest finite value. Every output is compared as bits.
//!
//! **Nothing widens the cache:** the report charges no job to `attention/widen`, and the BF16 run
//! takes the path the F32 one does (`attention/rows`, or the composition's parts).
//!
//! **A cache write moves BF16 rows whole** (`scatter_nd` with `Assign`, `record::SCATTER_ROWS_BF16`):
//! one row scatter, no cast, the host's writes bit for bit, at positions of both face-row
//! parities, every head's row.
//!
//! **Negative control:** the attention of the keys and values before they were rounded to BF16
//! (full F32 mantissas) differs from the BF16 cache's, so equal bits above say the products read
//! the BF16 values, not that any two attentions agree.

use burn::tensor::{module::attention, Bool, DType, IndexingUpdateOp, Int, Tensor, TensorData};
use burn_tensor::ops::AttentionModuleOptions;
use burn_tt::{device_time, TileChoice, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// Full-mantissa F32 values in about `[-2, 2]`, with signed zeros and a subnormal.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            match i % 97 {
                5 => -0.0,
                11 => 0.0,
                23 => f32::from_bits(0x0000_1234),
                _ => ((x >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0,
            }
        })
        .collect()
}

/// `x` rounded to BF16 (nearest, ties to even; no NaN among them).
fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + (b >> 16 & 1)) >> 16) as u16
}

fn widened(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

fn bf16_data(bits: Vec<u16>, shape: impl Into<burn::tensor::Shape>) -> TensorData {
    let mut data = TensorData::new(bits, shape);
    data.dtype = DType::BF16;
    data
}

fn bf16_bits(data: TensorData) -> Vec<u16> {
    assert_eq!(data.dtype, DType::BF16);
    let mut data = data;
    data.dtype = DType::U16;
    data.to_vec().unwrap()
}

fn f32_bits(t: Tensor<TtBackend, 4>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// How many times the server ran a kind of op in this process so far.
fn ran(kind: &str) -> u64 {
    device_time()
        .iter()
        .find(|(k, _, _)| *k == kind)
        .map_or(0, |(_, n, _)| *n)
}

/// One step's attention operands: the query, and the keys and values three ways -- BF16, their
/// BF16 values as F32, and the F32 values they were rounded from.
struct Case {
    what: String,
    q: Tensor<TtBackend, 4>,
    kv: [[Tensor<TtBackend, 4>; 2]; 3],
    mask: Option<Tensor<TtBackend, 4, Bool>>,
    /// Whether `attention_rows` takes it with BF16 keys and values.
    rows: bool,
}

const BF16: usize = 0;
const WIDENED: usize = 1;
const UNROUNDED: usize = 2;

/// `heads` query heads over `kv` key/value heads of `dim`, `t` query rows a head (one: a decode
/// step under a mask; more: a causal prompt), attending `len` positions of which the first
/// `pos + 1` are live.
#[allow(clippy::too_many_arguments)]
fn case(
    d: &burn_tt::TtDevice,
    heads: usize,
    kv: usize,
    dim: usize,
    len: usize,
    pos: usize,
    t: usize,
    seed: u64,
) -> Case {
    let groups = heads / kv;
    let row = Tensor::<TtBackend, 3>::from_data(
        TensorData::new(values(seed, t * heads * dim), [1, t, heads * dim]),
        d,
    )
    .to_device(d);
    let q = row.reshape([1, t, heads, dim]).swap_dims(1, 2);
    let q = if groups > 1 && t == 1 {
        q.reshape([1, kv, groups, dim])
    } else {
        q
    };
    // Position-major caches `[1, len, kv dim]`, each head a column block; past `pos`, NaN,
    // infinities and the largest finite BF16, which only the mask keeps out.
    let cache = |s: u64| {
        let wide = values(s, len * kv * dim);
        let bits: Vec<u16> = wide
            .iter()
            .enumerate()
            .map(|(i, &x)| match (i / (kv * dim) > pos && t == 1, i % 4) {
                (true, 0) => 0x7fc0,
                (true, 1) => 0x7f80,
                (true, 2) => 0xff80,
                (true, _) => 0x7f7f,
                (false, _) => to_bf16(x),
            })
            .collect();
        let heads_of = |x: Tensor<TtBackend, 3>| x.reshape([1, len, kv, dim]).swap_dims(1, 2);
        let narrow = Tensor::<TtBackend, 3>::from_data(
            bf16_data(bits.clone(), [1, len, kv * dim]),
            (d, DType::BF16),
        )
        .to_device(d);
        let wide_bits = bits.iter().map(|&b| widened(b)).collect::<Vec<_>>();
        let as_f32 =
            Tensor::<TtBackend, 3>::from_data(TensorData::new(wide_bits, [1, len, kv * dim]), d)
                .to_device(d);
        let unrounded = wide
            .iter()
            .zip(&bits)
            .map(|(&x, &b)| if b & 0x7f80 == 0x7f80 { widened(b) } else { x })
            .collect::<Vec<_>>();
        let unrounded =
            Tensor::<TtBackend, 3>::from_data(TensorData::new(unrounded, [1, len, kv * dim]), d)
                .to_device(d);
        [heads_of(narrow), heads_of(as_f32), heads_of(unrounded)]
    };
    let [k0, k1, k2] = cache(seed + 1);
    let [v0, v1, v2] = cache(seed + 2);
    let mask = (t == 1).then(|| {
        let later: Vec<bool> = (0..len).map(|p| p > pos).collect();
        Tensor::<TtBackend, 4, Bool>::from_data(TensorData::new(later, [1, 1, 1, len]), d)
    });
    let m = if groups > 1 && t == 1 { groups } else { t };
    let rows = t == 1
        && tt_kernels::attention::fits_as([m, dim, len, dim], 0, tt_kernels::tensor::Elem::BF16);
    Case {
        what: format!("{heads} heads over {kv}, d {dim}, {t} rows over {len} positions"),
        q,
        kv: [[k0, v0], [k1, v1], [k2, v2]],
        mask,
        rows,
    }
}

/// The attention of `c`'s query over its keys and values `which`, its bits and the jobs charged
/// to `attention/rows`, to the composition's parts and to `attention/widen`.
fn run(c: &Case, which: usize, rows: bool) -> (Vec<u32>, [u64; 3]) {
    burn_tt::set_attention_rows(rows);
    let [k, v] = c.kv[which].clone();
    let call = || {
        attention(
            c.q.clone(),
            k.clone(),
            v.clone(),
            c.mask.clone(),
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: c.mask.is_none(),
            },
        )
    };
    // A first call makes the query's view plain, once.
    let _ = f32_bits(call());
    let (out, report) = burn_tt::with_report(call);
    burn_tt::set_attention_rows(true);
    assert_native_model(&report);
    assert_eq!(out.dtype(), DType::F32, "{}: an F32 result", c.what);
    let jobs = |p: &str| report.op(p).map_or(0, |s| s.jobs);
    let composed = ["attention/qk", "attention/softmax", "attention/pv"]
        .iter()
        .map(|p| jobs(p))
        .sum();
    (
        f32_bits(out),
        [jobs("attention/rows"), composed, jobs("attention/widen")],
    )
}

fn config() -> Config {
    Config {
        tiles: Some(TileChoice::Count(8)),
        ..Config::default()
    }
}

#[test]
fn attention_over_bf16_keys_and_values_is_their_widened_values_bit_for_bit() {
    with_device(config(), |d| {
        let cases = [
            case(&d, 8, 8, 64, 64, 40, 1, 3),
            case(&d, 6, 2, 64, 96, 70, 1, 11),
            case(&d, 4, 4, 32, 64, 20, 1, 17),
            case(&d, 2, 2, 128, 128, 100, 1, 19),
            case(&d, 4, 4, 64, 1024, 900, 1, 23),
            case(&d, 2, 2, 64, 2048, 2000, 1, 29),
            case(&d, 4, 4, 64, 64, 0, 40, 31),
        ];
        for c in &cases {
            for rows in [true, false] {
                let (got, [rows_jobs, composed, widen]) = run(c, BF16, rows);
                let (want, ..) = run(c, WIDENED, rows);
                let what = format!("{} (attention rows {rows})", c.what);
                assert_eq!(widen, 0, "{what}: nothing widened the cache");
                if rows && c.rows {
                    assert_eq!(
                        (rows_jobs, composed),
                        (1, 0),
                        "{what}: one op, widen {widen}"
                    );
                } else {
                    assert_eq!(rows_jobs, 0, "{what}: the composition");
                    assert!(composed > 0, "{what}: the composition ran");
                }
                let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differing,
                    0,
                    "{what}: {differing} of {} outputs differ from the widened cache's",
                    want.len()
                );
                println!("{what}: {} outputs equal", want.len());
            }
            // The negative control: the values before rounding give other bits.
            let (bf16, _) = run(c, BF16, true);
            let (unrounded, _) = run(c, UNROUNDED, true);
            let differing = bf16.iter().zip(&unrounded).filter(|(a, b)| a != b).count();
            assert!(
                differing > bf16.len() / 2,
                "{}: only {differing} of {} outputs moved with the rounding",
                c.what,
                bf16.len()
            );
        }
    });
}

/// A fixed cache's write in BF16: `[1, heads, len, d]` as rows, each head's row at
/// `h len + position`, the position on the card (what `FixedCache` writes), at an even and an
/// odd position, and the same through a position-major cache's row.
#[test]
fn a_bf16_cache_write_moves_each_head_s_row_whole() {
    with_device(Config::default(), |d| {
        let (heads, len, dim) = (5usize, 64usize, 96usize);
        let bits: Vec<u16> = (0..heads * len * dim)
            .map(|i| ((i as u32).wrapping_mul(2_654_435_761) >> 16) as u16)
            .collect();
        let cache = Tensor::<TtBackend, 2>::from_data(
            bf16_data(bits.clone(), [heads * len, dim]),
            (&d, DType::BF16),
        )
        .to_device(&d);
        for position in [36usize, 37] {
            let new_bits: Vec<u16> = (0..heads * dim)
                .map(|i| (i as u16).wrapping_mul(40_503) ^ position as u16)
                .collect();
            let new = Tensor::<TtBackend, 2>::from_data(
                bf16_data(new_bits.clone(), [heads, dim]),
                (&d, DType::BF16),
            )
            .to_device(&d);
            let at = Tensor::<TtBackend, 1, Int>::from_data([position as i32], &d).add_scalar(0);
            let rows = (Tensor::<TtBackend, 1, Int>::arange(0..heads as i64, &d)
                .mul_scalar(len as i64)
                + at.expand([heads]))
            .reshape([heads, 1]);
            let _ = rows.clone().into_data();
            let scatters = ran("scatter_rows");
            let (written, report) = burn_tt::with_report(|| {
                cache
                    .clone()
                    .scatter_nd(rows.clone(), new.clone(), IndexingUpdateOp::Assign)
            });
            assert_native_model(&report);
            let queued: u64 = report.0.iter().map(|s| s.jobs).sum();
            assert_eq!(queued, 1, "one row scatter, no cast: {report:?}");
            assert_eq!(ran("scatter_rows"), scatters + 1, "the row scatter ran");
            assert_eq!(written.dtype(), DType::BF16);
            let mut want = bits.clone();
            for h in 0..heads {
                let r = h * len + position;
                want[r * dim..(r + 1) * dim].copy_from_slice(&new_bits[h * dim..(h + 1) * dim]);
            }
            assert_eq!(
                bf16_bits(written.into_data()),
                want,
                "the write at position {position}"
            );
        }
        assert_eq!(bf16_bits(cache.into_data()), bits, "the cache itself");
    });
}

/// A cast of a view keeps the view (`ops::cast_view_in_place`): the values, a column block of
/// one projection row viewed as heads, narrow in one job (that block alone,
/// `Session::cast_block`) and stay one row, so the cache write's reshape is a view again, not a
/// repack. Bit for bit the cast of the same view made plain first. The narrowing is silicon's
/// (ttsim does not convert FP32 to BF16 at the pack).
#[cfg(feature = "silicon")]
#[test]
fn a_view_narrows_in_one_job_and_keeps_its_layout() {
    with_device(config(), |d| {
        let (kv, dim) = (4usize, 64usize);
        // A `q|k|v` projection row: `v` its last column block.
        let row = Tensor::<TtBackend, 3>::from_data(
            TensorData::new(values(41, 3 * kv * dim), [1, 1, 3 * kv * dim]),
            &d,
        )
        .to_device(&d);
        let heads = |x: Tensor<TtBackend, 3>| x.reshape([1, 1, kv, dim]).swap_dims(1, 2);
        let v = heads(row.clone().slice([0..1, 0..1, 2 * kv * dim..3 * kv * dim]));
        let (narrow, report) = burn_tt::with_report(|| v.clone().cast(DType::BF16));
        assert_native_model(&report);
        let queued: u64 = report.0.iter().map(|s| s.jobs).sum();
        assert_eq!(queued, 1, "one cast, no repack: {report:?}");
        // The cache write's view of it: one row again, with no job.
        let (as_row, report) =
            burn_tt::with_report(|| narrow.clone().swap_dims(1, 2).reshape([1, kv * dim]));
        let queued: u64 = report.0.iter().map(|s| s.jobs).sum();
        assert_eq!(queued, 0, "the write's row is a view: {report:?}");
        let want = (v * 1.0).cast(DType::BF16);
        assert_eq!(
            bf16_bits(narrow.into_data()),
            bf16_bits(want.clone().into_data()),
            "the view's cast"
        );
        assert_eq!(
            bf16_bits(as_row.into_data()),
            bf16_bits(want.swap_dims(1, 2).reshape([1, kv * dim]).into_data()),
            "its row"
        );
    });
}

/// RoPE rounding its own result (`burn_tt::rope_as`, `Rope::apply_as`): a decode step's keys,
/// heads as column blocks of a projection row, rotated and narrowed to BF16 in one job, bit for
/// bit RoPE then the cast, so a BF16 cache's keys need no cast op. Silicon only (the pack's
/// narrowing).
#[cfg(feature = "silicon")]
#[test]
fn rope_rounds_its_own_result_in_one_job() {
    use tt_models::rope::Rope;
    with_device(config(), |d| {
        let (kv, dim, max_len) = (4usize, 64usize, 64usize);
        let at = Tensor::<TtBackend, 1, Int>::from_data([37], &d);
        let rope = Rope::<TtBackend>::table(max_len, dim, 10_000.0, DType::F32, &d).at(at);
        let row = Tensor::<TtBackend, 3>::from_data(
            TensorData::new(values(43, 3 * kv * dim), [1, 1, 3 * kv * dim]),
            &d,
        )
        .to_device(&d);
        let k = row
            .slice([0..1, 0..1, kv * dim..2 * kv * dim])
            .reshape([1, 1, kv, dim])
            .swap_dims(1, 2);
        let _ = f32_bits(rope.apply(k.clone()));
        let (narrow, report) = burn_tt::with_report(|| rope.apply_as(k.clone(), DType::BF16));
        assert_native_model(&report);
        let queued: u64 = report.0.iter().map(|s| s.jobs).sum();
        assert_eq!(queued, 1, "RoPE alone: {report:?}");
        assert_eq!(narrow.dtype(), DType::BF16);
        let want = rope.apply(k).cast(DType::BF16);
        assert_eq!(
            bf16_bits(narrow.into_data()),
            bf16_bits(want.into_data()),
            "RoPE rounded once against RoPE then the cast"
        );
    });
}
