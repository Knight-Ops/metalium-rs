//! B0 of `docs/plans/decode-step-bandwidth.md`: what each captured op moves through GDDR, counted
//! when its trace is stored (`tt_kernels::profile_mode::OpBytes`; the profile mode counts, and
//! `burn_tt::set_count_trace_bytes` counts without it).
//!
//! The claims, over a traced decode step of a synthetic Llama on `Tt`:
//! - **The products read their weights once.** The `B` bytes of the step's `float_matmul` ops are
//!   the model's weight tiles, each once, the head once: computed here from the configuration
//!   (an F32 weight of `[k, n]` is `ceil(k/32) ceil(n/32)` tile slots of 4,160 bytes), not from the
//!   counter.
//! - **`A` and `B` are told apart.** Fusing the projections (`Llama::fused`) leaves the `B` bytes
//!   unchanged and reads `A` fewer times: three products read the norm's output where one does.
//! - **Every op moves something,** and with counting off nothing is counted.

use burn_tt::{Tt, TtDevice};
use tt_models::generate::{Shapes, TracedGenerator};
use tt_models::llama::{Llama, LlamaConfig};
use tt_tests::burn_device::{with_device, Config};

const SLOT: u64 = tt_isa::dm::TILE_SLOT;

/// step193's two synthetic Llamas: multi-head with a tied head, and grouped-query 4:1 untied.
fn configs() -> Vec<(&'static str, LlamaConfig)> {
    let base = LlamaConfig {
        hidden: 64,
        heads: 2,
        kv_heads: 2,
        head_dim: 32,
        ff: 96,
        vocab: 100,
        layers: 2,
        rope_theta: 10_000.0,
        eps: 1e-5,
        tied_head: true,
    };
    vec![
        ("multi-head, tied", base),
        (
            "grouped-query 4:1, untied",
            LlamaConfig {
                hidden: 128,
                heads: 4,
                kv_heads: 1,
                ff: 160,
                vocab: 64,
                rope_theta: 100_000.0,
                tied_head: false,
                ..base
            },
        ),
    ]
}

fn tiles(k: usize, n: usize) -> u64 {
    (k.div_ceil(32) * n.div_ceil(32)) as u64 * SLOT
}

/// The bytes of every weight a decode step's products read, once each: the projections (fused
/// as `Llama::fused` fuses them, or not), the MLP and the head.
fn weight_bytes(c: &LlamaConfig, fused: bool) -> u64 {
    let (h, q, kv) = (c.hidden, c.heads * c.head_dim, c.kv_heads * c.head_dim);
    let layer = if fused {
        tiles(h, q + 2 * kv) + tiles(h, 2 * c.ff)
    } else {
        tiles(h, q) + 2 * tiles(h, kv) + 2 * tiles(h, c.ff)
    } + tiles(q, h)
        + tiles(c.ff, h);
    // A tied head reads the `[vocab, hidden]` table transposed: as many tiles.
    c.layers as u64 * layer + tiles(h, c.vocab)
}

/// A decode step's ops, as its trace recorded them, for each step length.
fn step_ops(model: &Llama<Tt>, d: &TtDevice) -> Vec<(String, Vec<burn_tt::OpRecord>)> {
    let g = TracedGenerator::<_, Tt>::new(model, &Shapes::new(16, &[4]), d).unwrap();
    g.trace_ops()
        .unwrap()
        .into_iter()
        .filter(|(k, _)| k.starts_with("Step"))
        .collect()
}

#[test]
fn a_decode_step_reads_each_weight_once() {
    with_device(Config::default(), |d| {
        burn_tt::set_count_trace_bytes(d, true).unwrap();
        for (what, config) in configs() {
            let unfused = Llama::<Tt>::new(&config, &d);
            let mut a_reads = Vec::new();
            for fused in [false, true] {
                let model = if fused {
                    unfused.clone().fused()
                } else {
                    unfused.clone()
                };
                let what = format!("{what}{}", if fused { ", fused" } else { "" });
                let want = weight_bytes(&config, fused);
                for (key, ops) in step_ops(&model, &d) {
                    let products: Vec<_> =
                        ops.iter().filter(|o| o.label == "float_matmul").collect();
                    let b: u64 = products.iter().map(|o| o.bytes.b).sum();
                    let a: u64 = products.iter().map(|o| o.bytes.a).sum();
                    let read: u64 = ops.iter().map(|o| o.bytes.read).sum();
                    let write: u64 = ops.iter().map(|o| o.bytes.write).sum();
                    eprintln!(
                        "{what} {key}: {} ops, {} products; read {read} B, written {write} B; \
                         products' A {a} B, B {b} B (weights {want} B)",
                        ops.len(),
                        products.len()
                    );
                    assert_eq!(b, want, "{what} {key}: the products' B against the weights");
                    let layers = config.layers;
                    let per_layer = if fused { 4 } else { 7 };
                    assert_eq!(products.len(), per_layer * layers + 1, "{what} {key}");
                    for (i, o) in ops.iter().enumerate() {
                        if o.entries.iter().any(|r| r.len() > o.barrier as usize) {
                            assert!(
                                o.bytes.read + o.bytes.write > 0,
                                "{what} {key}: op {i} ({}, {}) moved nothing",
                                o.label,
                                o.what
                            );
                        }
                    }
                    a_reads.push((fused, key, a));
                }
            }
            for (key, unfused_a) in a_reads.iter().filter(|r| !r.0).map(|r| (&r.1, r.2)) {
                let fused_a = a_reads.iter().find(|r| r.0 && &r.1 == key).unwrap().2;
                assert!(
                    fused_a < unfused_a,
                    "{what} {key}: fused products read A {fused_a} B, unfused {unfused_a} B"
                );
            }
        }
        // The control: counting off counts nothing.
        burn_tt::set_count_trace_bytes(d, false).unwrap();
        let (_, config) = configs().remove(0);
        let model = Llama::<Tt>::new(&config, &d);
        for (key, ops) in step_ops(&model, &d) {
            assert!(
                ops.iter().all(|o| o.bytes == Default::default()),
                "{key}: counted with counting off"
            );
        }
    });
}
