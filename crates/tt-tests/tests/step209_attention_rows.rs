//! Attention for a decode step's rows as one op (`Session::attention_rows`, decode-to-bandwidth
//! Phase 1): each key/value head's scores, softmax and `P V` on one unit, the intermediates in
//! L1, where the composition ran ten ops (`Q K^T`, its stacking gather, the scale and mask, the
//! softmax's four, `P`'s gather, `P V` and its stacking gather) with a barrier after each.
//!
//! **Bit for bit the composition** (`burn_tt::set_attention_rows(false)`) on the shapes a decode
//! step attends with: a query row a head read in place from a projection's row (multi-head),
//! grouped query rows folded a key/value head (`[1, kv, groups, d]`), keys and values read in
//! place from a position-major cache, `head_dim` 32, 64 and 128, attended lengths of one to
//! sixteen tiles, and one mask row for every head. The data has full mantissas, signed zeros,
//! subnormals and extremes; the masked positions' keys and values hold NaN and infinities, which
//! the mask takes out of the scores and `P`'s zeros carry into the output as the composition's
//! products do. Every output is compared as bits, NaN payloads included.
//!
//! **The context's row and its packed twin** feed the projection after attention: its product
//! equals the same product of a copy of the row (an element-wise op's twin) bit for bit.
//!
//! **The op is the one that ran:** the report charges each call one job to `attention/rows` and
//! none to the composition's parts, and the other way round with it turned off.
//!
//! **Negative control:** the composition with the keys' and values' positions permuted (and the
//! mask with them) is the same attention summed in another order; its bits differ, so equal bits
//! here say the order was kept, not only the arithmetic. A reversal is no control: two `K`
//! tiles add commutatively and the lanes' fold is a butterfly, so it gave the same bits.

use burn::tensor::{module::attention, Bool, Tensor, TensorData};
use burn_tensor::ops::AttentionModuleOptions;
use burn_tt::{TileChoice, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// Full-mantissa values in about `[-2, 2]`, with a few signed zeros and subnormals, and with
/// `extremes` a few large values: only in the values, since a large query or key makes the
/// softmax one-hot, whose `P` of zeros and ones sums the same in any order.
fn values(seed: u64, n: usize, extremes: bool) -> Vec<f32> {
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
                41 if extremes => 37.5,
                67 if extremes => -52.25,
                _ => ((x >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0,
            }
        })
        .collect()
}

fn resident<const D: usize>(
    d: &burn_tt::TtDevice,
    shape: [usize; D],
    data: Vec<f32>,
) -> Tensor<TtBackend, D> {
    let x = Tensor::<TtBackend, D>::from_data(TensorData::new(data, shape), d);
    let _ = (x.clone() * 1.0).sum().into_data();
    x
}

fn bits(t: Tensor<TtBackend, 4>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// One decode step's attention operands, laid out as `Llama`'s are.
struct Case {
    what: String,
    q: Tensor<TtBackend, 4>,
    k: Tensor<TtBackend, 4>,
    v: Tensor<TtBackend, 4>,
    mask: Tensor<TtBackend, 4, Bool>,
}

/// `heads` query heads over `kv` key/value heads of `dim`, attending `len` positions of which
/// the first `pos + 1` are live: `q` is column blocks of one projection row (folded `[1, kv,
/// groups, d]` when grouped), the keys and values column blocks of position-major caches.
#[allow(clippy::too_many_arguments)]
fn case(
    d: &burn_tt::TtDevice,
    heads: usize,
    kv: usize,
    dim: usize,
    len: usize,
    pos: usize,
    seed: u64,
    stacked: bool,
) -> Case {
    let groups = heads / kv;
    let q = if stacked {
        // Each head a row of one stored `[heads, d]` matrix: head `h` at row `h` of its tile.
        resident(d, [1, heads, 1, dim], values(seed, heads * dim, false))
    } else {
        let row = resident(d, [1, 1, heads * dim], values(seed, heads * dim, false));
        row.reshape([1, 1, heads, dim]).swap_dims(1, 2)
    };
    let q = if groups > 1 {
        q.reshape([1, kv, groups, dim])
    } else {
        q
    };
    // The masked positions' keys and values: NaN, infinities and huge values, which only the mask
    // keeps out of the scores.
    let cache = |s: u64, extremes: bool| {
        let mut data = values(s, len * kv * dim, extremes);
        for (i, x) in data.iter_mut().enumerate() {
            if i / (kv * dim) > pos {
                *x = match i % 4 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    _ => 3.0e38,
                };
            }
        }
        resident(d, [1, len, kv * dim], data)
            .reshape([1, len, kv, dim])
            .swap_dims(1, 2)
    };
    let (k, v) = (cache(seed + 1, false), cache(seed + 2, true));
    let later: Vec<bool> = (0..len).map(|p| p > pos).collect();
    let mask = Tensor::<TtBackend, 4, Bool>::from_data(TensorData::new(later, [1, 1, 1, len]), d);
    Case {
        what: format!(
            "{heads} heads over {kv}, d {dim}, {len} positions, {} live",
            pos + 1
        ),
        q,
        k,
        v,
        mask,
    }
}

fn options() -> AttentionModuleOptions {
    AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: false,
    }
}

/// The attention's bits and the jobs charged to `attention/rows` and to the composition's parts.
fn run(c: &Case, rows: bool) -> (Vec<u32>, u64, u64) {
    burn_tt::set_attention_rows(rows);
    let call = || {
        attention(
            c.q.clone(),
            c.k.clone(),
            c.v.clone(),
            Some(c.mask.clone()),
            None,
            options(),
        )
    };
    let _ = bits(call());
    let (out, report) = burn_tt::with_report(call);
    burn_tt::set_attention_rows(true);
    assert_native_model(&report);
    let jobs = |p: &str| report.op(p).map_or(0, |s| s.jobs);
    let composed = [
        "attention/qk",
        "attention/scale",
        "attention/softmax",
        "attention/pv",
    ]
    .iter()
    .map(|p| jobs(p))
    .sum();
    (bits(out), jobs("attention/rows"), composed)
}

fn config() -> Config {
    Config {
        tiles: Some(TileChoice::Count(8)),
        ..Config::default()
    }
}

#[test]
fn attention_rows_are_the_composition_bit_for_bit() {
    with_device(config(), |d| {
        let cases = [
            case(&d, 8, 8, 64, 64, 40, 3, false),
            case(&d, 4, 4, 64, 32, 0, 5, false),
            case(&d, 4, 4, 64, 32, 31, 7, false),
            case(&d, 6, 2, 64, 96, 70, 11, false),
            case(&d, 9, 3, 64, 64, 33, 13, false),
            case(&d, 4, 4, 32, 64, 20, 17, false),
            case(&d, 2, 2, 128, 128, 100, 19, false),
            case(&d, 2, 1, 64, 512, 300, 23, false),
            case(&d, 32, 32, 64, 64, 50, 31, true),
            case(&d, 32, 32, 64, 512, 400, 37, true),
        ];
        for c in &cases {
            let (fused, rows_jobs, composed_jobs) = run(c, true);
            assert_eq!(rows_jobs, 1, "{}: attention as one op", c.what);
            assert_eq!(composed_jobs, 0, "{}: no composed part ran", c.what);
            let (want, rows_off, composed_off) = run(c, false);
            assert_eq!(rows_off, 0, "{}: turned off", c.what);
            assert!(composed_off > 0, "{}: the composition ran", c.what);
            let differing = fused.iter().zip(&want).filter(|(a, b)| a != b).count();
            assert_eq!(
                differing,
                0,
                "{}: {differing} of {} outputs differ from the composition",
                c.what,
                want.len()
            );
            println!("{}: {} outputs equal", c.what, want.len());
            // The context's row and its packed twin, read by the projection after attention: its
            // product equals the same product of a copy (`x * 1`, an element-wise op's twin,
            // `step207`), bit for bit, where the row is long enough to have one.
            let [_, h, _, dim] = c.q.dims();
            if h * dim >= 1024 {
                let ctx = attention(
                    c.q.clone(),
                    c.k.clone(),
                    c.v.clone(),
                    Some(c.mask.clone()),
                    None,
                    options(),
                )
                .swap_dims(1, 2)
                .reshape([1, h * dim]);
                let w = Tensor::<TtBackend, 2>::from_data(
                    TensorData::new(values(41, h * dim * 64, false), [h * dim, 64]),
                    &d,
                );
                let via_twin = ctx.clone().matmul(w.clone());
                let via_copy = (ctx * 1.0).matmul(w);
                let b = |t: Tensor<TtBackend, 2>| -> Vec<u32> {
                    t.into_data()
                        .to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect()
                };
                assert_eq!(b(via_twin), b(via_copy), "{}: the context's twin", c.what);
            }
        }
    });
}

#[test]
fn positions_in_another_order_give_other_bits() {
    with_device(config(), |d| {
        let len = 64;
        let c = case(&d, 4, 4, 64, len, len - 1, 29, false);
        let (fused, _, _) = run(&c, true);
        // Position `p` to `(5 p + 3) mod len`: not a reversal, which a two-tile chain's
        // commutative add and the lanes' butterfly fold would sum to the same bits.
        let order: Vec<i64> = (0..len).map(|p| ((5 * p + 3) % len) as i64).collect();
        let order =
            Tensor::<TtBackend, 1, burn::tensor::Int>::from_data(TensorData::new(order, [len]), &d);
        let permuted = Case {
            what: "permuted".into(),
            q: c.q.clone(),
            k: c.k.clone().select(2, order.clone()),
            v: c.v.clone().select(2, order.clone()),
            mask: c.mask.clone().select(3, order),
        };
        let (other, _, _) = run(&permuted, false);
        let differing = fused.iter().zip(&other).filter(|(a, b)| a != b).count();
        println!(
            "permuted positions: {differing} of {} outputs differ",
            other.len()
        );
        assert!(
            differing > 0,
            "the same attention summed in another order gave the same bits: the gate cannot see \
             the order"
        );
    });
}

/// **What a job reaches of L1 is what its op records** (`trace::OpRecord::l1_top`): the job lays its
/// slots from `DATA`'s base (`q`, `K`, the mask, the scores twice, the statistics, `V`, the
/// output), and a busy unit's weight prefetch lands above the highest top the ops before its
/// product recorded (`Session::busy_start`). Watched failing: the top recorded was the job's
/// products' own layouts', far below its `K` and `V` slots, and the 1.7B's step at 512 positions
/// read its next weights over them (`silicon_bench_models::traced_generation`, 256-token prompt).
#[test]
fn a_captured_job_records_the_l1_its_slots_reach() {
    use tt_kernels::attention::Item;
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_kernels::session::{Session, TileChoice as Tiles};
    use tt_kernels::tensor::Elem;
    tt_ttsim::fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(dev, tt_firmware_images::ROLES, Tiles::Count(2), |_, _| {
            Ok(None)
        })
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        let (heads, d) = (2usize, 64usize);
        for l in [64usize, 512] {
            let q = s.upload(&values(1, heads * d, false), heads, d).unwrap();
            let k = s
                .upload(&values(2, l * heads * d, false), l, heads * d)
                .unwrap();
            let v = s
                .upload(&values(3, l * heads * d, false), l, heads * d)
                .unwrap();
            let last: Vec<u32> = (0..l).map(|p| u32::from(p + 1 == l)).collect();
            let mask = s.upload_bits(&last, 1, l, Elem::Bool).unwrap();
            let items: Vec<Item> = (0..heads)
                .map(|h| Item {
                    q: [h, 0],
                    k: [0, h * d],
                    v: [0, h * d],
                })
                .collect();
            s.begin_trace().unwrap();
            let out = s
                .attention_rows(
                    &q,
                    &k,
                    &v,
                    &mask,
                    &items,
                    [1, d, l, d],
                    0.125,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    tt_tests::harness::BUDGET,
                )
                .unwrap();
            let id = s.end_trace().unwrap();
            let (kt, nt, dvt) = (d / 32, l / 32, d / 32);
            let slots = kt + kt * nt + nt + 2 * nt + 2 + nt * dvt + dvt;
            let reach = tt_isa::l1::DATA.base + slots as u64 * tt_isa::dm::TILE_SLOT;
            let top = s
                .trace_ops(id)
                .unwrap()
                .iter()
                .map(|op| op.l1_top)
                .max()
                .unwrap();
            assert!(
                top >= reach,
                "l {l}: the op recorded L1 to {top:#x}, its slots reach {reach:#x}"
            );
            s.release_trace(id).unwrap();
            for t in [q, k, v, mask, out] {
                s.free(t).unwrap();
            }
        }
    })
    .unwrap();
}
