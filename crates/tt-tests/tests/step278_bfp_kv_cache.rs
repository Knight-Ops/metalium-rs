//! A block-float KV cache (`docs/plans/block-float-kv.md` K3): keys and values stored BFP8 or BFP4
//! (logical F32, `burn_tt::storage`), a quarter or an eighth of an F32 cache's bytes, written and
//! read where they lie.
//!
//! **Attention over block-float keys and values is the attention of their decoded values, bit for
//! bit.** Each product reads the cache as it lies (`SrcRoute::Edges`: a block float decoded into
//! `Src` is exact in TF32), the keys' transpose made by the unpacker (`step277`). Against the
//! same attention of F32 keys and values holding the decoded values (`with_storage(F32)`), on a
//! decode step's shapes (`step273`'s: a query row a head read from a projection's row, grouped
//! query rows folded a key/value head, a position-major cache, head_dim 32 to 256) and a causal
//! prompt's, on both paths: `attention_rows` (one op a step) and the composition. Every output
//! is compared as bits.
//!
//! **Nothing widens the cache:** no job is charged to `attention/widen` or to a decode
//! (`with_storage`: the generated adapter's, or a view made plain), and the block-float run takes
//! the path the F32 one does.
//!
//! **A cache write moves encoded rows** (`scatter_nd` with `Assign`, `record::SCATTER_ROWS` with a
//! block-float format, `step276`): the rows cast to the cache's format, then one row scatter;
//! the result is the F32 cache written alike and cast, every value.
//!
//! **Negative control:** the attention of the values before they were stored block-float differs
//! from the cache's, so equal bits above say the products read the stored values.

use burn::tensor::{module::attention, Bool, DType, IndexingUpdateOp, Int, Tensor, TensorData};
use burn_tensor::ops::AttentionModuleOptions;
use burn_tt::storage::{StorageFormat, TensorStorageExt};
use burn_tt::{device_time, TileChoice, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// Values of several magnitudes in about `[-4, 4]`, zeros among them.
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
                _ => {
                    ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0)
                        * f32::powi(2.0, (x >> 12) as i32 % 5 - 2)
                }
            }
        })
        .collect()
}

fn f32_bits(t: Tensor<TtBackend, 4>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn ran(kind: &str) -> u64 {
    device_time()
        .iter()
        .find(|(k, _, _)| *k == kind)
        .map_or(0, |(_, n, _)| *n)
}

const STORED: usize = 0;
const DECODED: usize = 1;
const UNROUNDED: usize = 2;

struct Case {
    what: String,
    q: Tensor<TtBackend, 4>,
    kv: [[Tensor<TtBackend, 4>; 2]; 3],
    mask: Option<Tensor<TtBackend, 4, Bool>>,
    rows: bool,
}

/// As `step273`'s: `heads` query heads over `kv` heads of `dim`, `t` query rows a head, over
/// `len` positions of which the first `pos + 1` are live; past them, large values only the mask
/// keeps out.
#[allow(clippy::too_many_arguments)]
fn case(
    d: &burn_tt::TtDevice,
    format: StorageFormat,
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
    let cache = |s: u64| {
        let wide: Vec<f32> = values(s, len * kv * dim)
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                if i / (kv * dim) > pos && t == 1 {
                    3.0e30 * if i % 2 == 0 { 1.0 } else { -1.0 }
                } else {
                    x
                }
            })
            .collect();
        let heads_of = |x: Tensor<TtBackend, 3>| x.reshape([1, len, kv, dim]).swap_dims(1, 2);
        let unrounded =
            Tensor::<TtBackend, 3>::from_data(TensorData::new(wide, [1, len, kv * dim]), d)
                .to_device(d);
        let stored = unrounded.clone().with_storage(format);
        let decoded = stored.clone().with_storage(StorageFormat::F32);
        [heads_of(stored), heads_of(decoded), heads_of(unrounded)]
    };
    let [k0, k1, k2] = cache(seed + 1);
    let [v0, v1, v2] = cache(seed + 2);
    let mask = (t == 1).then(|| {
        let later: Vec<bool> = (0..len).map(|p| p > pos).collect();
        Tensor::<TtBackend, 4, Bool>::from_data(TensorData::new(later, [1, 1, 1, len]), d)
    });
    let m = if groups > 1 && t == 1 { groups } else { t };
    let elem = tt_kernels::tensor::Elem::Packed(tt_isa::format::TileFormat::new(match format {
        StorageFormat::Bfp8 => tt_isa::tile::L1Format::Bfp8,
        _ => tt_isa::tile::L1Format::Bfp4,
    }));
    let rows = t == 1 && tt_kernels::attention::fits_as([m, dim, len, dim], 0, elem);
    Case {
        what: format!(
            "{format:?}: {heads} heads over {kv}, d {dim}, {t} rows over {len} positions"
        ),
        q,
        kv: [[k0, v0], [k1, v1], [k2, v2]],
        mask,
        rows,
    }
}

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
    let _ = f32_bits(call());
    let (out, report) = burn_tt::with_report(call);
    burn_tt::set_attention_rows(true);
    assert_native_model(&report);
    let jobs = |p: &str| report.op(p).map_or(0, |s| s.jobs);
    let composed = ["attention/qk", "attention/softmax", "attention/pv"]
        .iter()
        .map(|p| jobs(p))
        .sum();
    // A decode anywhere (the adapter's, or a view made plain) is a `with_storage` job.
    (
        f32_bits(out),
        [
            jobs("attention/rows"),
            composed,
            jobs("attention/widen") + jobs("with_storage"),
        ],
    )
}

fn config() -> Config {
    Config {
        tiles: Some(TileChoice::Count(8)),
        ..Config::default()
    }
}

#[test]
fn attention_over_block_float_keys_and_values_is_their_decoded_values_bit_for_bit() {
    with_device(config(), |d| {
        for format in [StorageFormat::Bfp8, StorageFormat::Bfp4] {
            let cases = [
                case(&d, format, 8, 8, 64, 64, 40, 1, 3),
                case(&d, format, 6, 2, 64, 96, 70, 1, 11),
                case(&d, format, 4, 4, 32, 64, 20, 1, 17),
                case(&d, format, 2, 2, 128, 128, 100, 1, 19),
                case(&d, format, 4, 1, 256, 256, 200, 1, 21),
                case(&d, format, 4, 4, 64, 1024, 900, 1, 23),
                case(&d, format, 4, 4, 64, 64, 0, 40, 31),
            ];
            for c in &cases {
                for rows in [true, false] {
                    let (got, [rows_jobs, composed, widen]) = run(c, STORED, rows);
                    let (want, ..) = run(c, DECODED, rows);
                    let what = format!("{} (attention rows {rows})", c.what);
                    assert_eq!(widen, 0, "{what}: nothing widened the cache");
                    if rows && c.rows {
                        assert_eq!((rows_jobs, composed), (1, 0), "{what}: one op");
                    } else {
                        assert_eq!(rows_jobs, 0, "{what}: the composition");
                        assert!(composed > 0, "{what}: the composition ran");
                    }
                    let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        differing,
                        0,
                        "{what}: {differing} of {} outputs differ from the decoded cache's",
                        want.len()
                    );
                    println!("{what}: {} outputs equal", want.len());
                }
                let (stored, _) = run(c, STORED, true);
                let (unrounded, _) = run(c, UNROUNDED, true);
                let differing = stored
                    .iter()
                    .zip(&unrounded)
                    .filter(|(a, b)| a != b)
                    .count();
                assert!(
                    differing > stored.len() / 2,
                    "{}: only {differing} of {} outputs moved with the rounding",
                    c.what,
                    stored.len()
                );
            }
        }
    });
}

/// A fixed cache's write into a block-float cache: `[heads len, dim]` rows, head `h`'s row at
/// `h len + position` (what `FixedCache` writes head-major), at positions of every kind (a face's
/// first row, its last, the next tile's), and a chunk's rows.
#[test]
fn a_block_float_cache_write_is_the_f32_write_cast() {
    with_device(Config::default(), |d| {
        let (heads, len, dim) = (3usize, 96usize, 64usize);
        for format in [StorageFormat::Bfp8, StorageFormat::Bfp4] {
            let start = values(5, heads * len * dim);
            let f32_cache =
                Tensor::<TtBackend, 2>::from_data(TensorData::new(start, [heads * len, dim]), &d)
                    .to_device(&d);
            let mut cache = f32_cache.clone().with_storage(format);
            let mut want = f32_cache;
            for (k, positions) in [
                vec![0usize],
                vec![15],
                vec![16],
                vec![33],
                (40..72).collect(),
            ]
            .into_iter()
            .enumerate()
            {
                let n = positions.len();
                let new = Tensor::<TtBackend, 2>::from_data(
                    TensorData::new(values(50 + k as u64, heads * n * dim), [heads * n, dim]),
                    &d,
                )
                .to_device(&d)
                .mul_scalar(100.0);
                let rows: Vec<i32> = (0..heads)
                    .flat_map(|h| positions.iter().map(move |&p| (h * len + p) as i32))
                    .collect();
                let rows = Tensor::<TtBackend, 1, Int>::from_data(rows.as_slice(), &d)
                    .reshape([heads * n, 1]);
                let scatters = ran("scatter_rows");
                let (written, report) = burn_tt::with_report(|| {
                    cache
                        .clone()
                        .scatter_nd(rows.clone(), new.clone(), IndexingUpdateOp::Assign)
                });
                assert_native_model(&report);
                assert_eq!(
                    ran("scatter_rows"),
                    scatters + 1,
                    "one row scatter: {report:?}"
                );
                assert_eq!(
                    written.storage_format(),
                    format,
                    "the cache stays {format:?}"
                );
                cache = written;
                want = want.scatter_nd(rows, new, IndexingUpdateOp::Assign);
            }
            let got: Vec<u32> = cache
                .with_storage(StorageFormat::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let want: Vec<u32> = want
                .with_storage(format)
                .with_storage(StorageFormat::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(
                differing,
                0,
                "{format:?}: {differing} of {} values differ",
                want.len()
            );
            println!("{format:?}: {} values equal the F32 write cast", want.len());
        }
    });
}
