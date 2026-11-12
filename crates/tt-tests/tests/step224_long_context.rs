//! Long context (`docs/plans/long-context.md`): a decoder's traces build, fit and generate at
//! real context lengths, not only the 512 positions everything ran at before.
//!
//! What failed (2026-10-23): a `TracedGenerator` holds every trace's programs in one unit's
//! 240 KiB program cache at once (`burn_tt::SharedStateTraces`), and an element-wise op's and a
//! reduction's programs were unrolled a tile, so each longer step bucket held more and the
//! SmolLM2-1.7B's build was refused at `max_len` 2048 and 8192 ("a list's programs do not fit").
//!
//! The claims:
//! - **The traces build at 8,192 and 32,768 positions** for two-layer grouped-query Llamas of
//!   head widths 128 and 256 (shapes SmolLM2 does not have), with decode steps at every doubling
//!   from 64 to the whole cache.
//! - **A traced generation at those lengths equals the eager one, token for token and bit for
//!   bit** (the same schedule run op by op, `generate_fixed`): a prompt that nearly fills the
//!   cache (fed in chunks of 64 and up to 512, each written as its rows alone,
//!   `FixedCache::chunk_write`),
//!   and one whose decode steps cross the 4,096-position bucket's edge.
//! - **Each trace fits a unit's program cache alone:** its programs on its fullest unit are
//!   within [`TRACE_BUDGET`]. The traces together need not fit: a replay puts back the programs
//!   another trace's took the room of (`step226_paged_programs`).
//!
//! Silicon only: the simulator's one tile is not what runs a long context, and a build at these
//! lengths is minutes of simulation.

#![cfg(feature = "silicon")]

mod ladder_support;

use burn_tt::{TileChoice, TtBackend};
use tt_models::generate::{Shapes, TracedGenerator};
use tt_models::llama::{Llama, LlamaConfig};
use tt_tests::burn_device::{with_device, Config};

/// What one trace's programs may take of a unit's program cache: room for the eager ops between
/// replays beside them, so a replay does not evict what the next one needs.
const TRACE_BUDGET: u64 = tt_isa::l1::PROGRAM_CACHE.len() / 2;

fn configs() -> Vec<(&'static str, LlamaConfig)> {
    let base = LlamaConfig {
        hidden: 256,
        heads: 8,
        kv_heads: 2,
        head_dim: 128,
        ff: 512,
        vocab: 512,
        layers: 2,
        rope_theta: 500_000.0,
        eps: 1e-5,
        tied_head: true,
    };
    vec![
        ("grouped-query 4:1, head width 128", base),
        (
            "grouped-query 4:1, head width 256",
            LlamaConfig {
                heads: 4,
                kv_heads: 1,
                head_dim: 256,
                ..base
            },
        ),
    ]
}

fn every_tile() -> Config {
    Config {
        tiles: Some(TileChoice::All),
        ..Config::default()
    }
}

#[test]
fn traces_build_and_fit_at_long_context() {
    // What each captured op holds, by op (`Session::space_report`): `TT_SPACE_REPORT=1`.
    if std::env::var("TT_SPACE_REPORT").as_deref() == Ok("1") {
        tt_kernels::session::probe_space_report(true);
    }
    // Each build in its own forked child (`with_device`), which panics with what it found over
    // budget: every build is reported, not only the first.
    let mut failed = Vec::new();
    for max_len in [8192, 32768] {
        for (what, config) in configs() {
            let run = std::panic::catch_unwind(|| {
                with_device(every_tile(), |d| {
                    let mut over = Vec::new();
                    let model = Llama::<TtBackend>::new(&config, &d);
                    let shapes = Shapes::new(max_len, &[64]);
                    let t0 = std::time::Instant::now();
                    let g = TracedGenerator::new(&model, &shapes, &d).unwrap_or_else(|e| {
                        panic!("{what}, {max_len} positions: the traces do not build: {e}")
                    });
                    eprintln!(
                        "{what}, {max_len} positions: built in {:.1} s, steps {:?}",
                        t0.elapsed().as_secs_f64(),
                        shapes.steps
                    );
                    for set in g.footprint().unwrap() {
                        let (held, size) = set.cache_held_bytes();
                        eprintln!(
                            "  program cache: {held} of {size} bytes held on the fullest unit"
                        );
                        let mut by_trace = Vec::new();
                        for (key, units) in &set.traces {
                            let most = units
                                .iter()
                                .flatten()
                                .map(|u| u.program_bytes)
                                .max()
                                .unwrap_or(0);
                            let gddr: u64 = units.iter().flatten().map(|u| u.gddr_bytes).sum();
                            eprintln!("  {key}: {most} program bytes on its fullest unit, {gddr} stream bytes");
                            by_trace.push((key.clone(), most));
                        }
                        for (key, most) in &by_trace {
                            if *most > TRACE_BUDGET {
                                over.push(format!(
                                    "{what}, {max_len} positions: {key} takes {most} program \
                                     bytes of a unit's cache, past {TRACE_BUDGET}"
                                ));
                            }
                        }
                    }
                    // What each trace gathers by rows, for diagnosis: a chunk's grouped-query
                    // mask is read as a block (`Broadcast::Block`, D3; asserted by `step227`), so
                    // what is left is made once a chunk (its mask's own positions expanded), not
                    // once a layer.
                    for (key, ops) in g.trace_ops().unwrap() {
                        let labels: Vec<_> = ops
                            .iter()
                            .filter(|o| o.what == "row gather list")
                            .map(|o| o.label)
                            .collect();
                        eprintln!("  {key}: row gathers {labels:?}");
                    }
                    assert!(over.is_empty(), "over budget:\n{}", over.join("\n"));
                })
            });
            if run.is_err() {
                failed.push(format!("{what}, {max_len} positions"));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "failed (each child's message is above): {failed:?}"
    );
}

/// A Llama's traced generation against the eager fixed schedule, bit for bit, at long context.
/// Not against the growing cache, as `step187` also checks at short lengths: its prefill's
/// causal scores would be gigabytes a layer here.
#[test]
fn traced_generation_equals_the_eager_one_at_long_context() {
    use ladder_support::{assert_same, generate, prompt, traced};
    use tt_models::generate::generate_fixed;
    for max_len in [8192, 32768] {
        for (what, config) in configs() {
            with_device(every_tile(), |d| {
                let model = Llama::<TtBackend>::new(&config, &d);
                // A chunk's attention holds its scores `[rows, max_len]` several times over (the
                // composition's softmax): the largest chunk shrinks as the cache grows, 512 at
                // 8,192 positions, 128 at 32,768, so they take the same GDDR.
                let largest = (512 * 8192 / max_len).max(64);
                let shapes = Shapes::new(max_len, &[64, largest]);
                let g = traced(&model, &shapes, &d);
                let gb = |(fullest, all): (u64, u64)| {
                    format!(
                        "{:.2} GB on the fullest channel, {:.2} in all",
                        fullest as f64 / 1e9,
                        all as f64 / 1e9
                    )
                };
                eprintln!(
                    "{what}, {max_len} positions: built; GDDR free {}",
                    gb(burn_tt::gddr_free_bytes(d).unwrap())
                );
                // Nearly the whole cache, then a few steps; and decode steps across 4,096.
                for (len, steps) in [(max_len - 200, 4), (4090, 12)] {
                    let p = prompt(len, config.vocab);
                    let t0 = std::time::Instant::now();
                    let got = generate(&g, &p, steps);
                    let traced_s = t0.elapsed().as_secs_f64();
                    eprintln!(
                        "{what}, {max_len} positions, prompt {len}: traced in {traced_s:.2} s; GDDR free {}",
                        gb(burn_tt::gddr_free_bytes(d).unwrap())
                    );
                    let want = generate_fixed(&model, &p, steps, &shapes, &d).unwrap();
                    eprintln!(
                        "{what}, {max_len} positions, prompt {len}: eager done; GDDR free {}",
                        gb(burn_tt::gddr_free_bytes(d).unwrap())
                    );
                    eprintln!(
                        "{what}, {max_len} positions, prompt {len}: traced {traced_s:.2} s, \
                         tokens {:?}",
                        got.tokens
                    );
                    assert_same(&got, &want, &format!("{what}, {max_len}, prompt {len}"));
                }
            });
        }
    }
}

/// Diagnosis: the long-context gate's sequence by hand, GDDR free as the eager schedule runs.
#[test]
#[ignore = "a probe: run explicitly"]
fn probe_eager_chunks_gddr() {
    use tt_models::generate::{schedule, FixedShapeDecoder};
    let (_, config) = configs()[0];
    with_device(every_tile(), |d| {
        let model = Llama::<TtBackend>::new(&config, &d);
        let max_len = 32768;
        let shapes = Shapes::new(max_len, &[64, 128]);
        let g = TracedGenerator::new(&model, &shapes, &d).unwrap();
        let free = |what: &str| {
            let (fullest, all) = burn_tt::gddr_free_bytes(d).unwrap();
            eprintln!(
                "PROBE {what}: GDDR free {:.3} GB fullest, {:.3} all",
                fullest as f64 / 1e9,
                all as f64 / 1e9
            );
        };
        free("built");
        let p = ladder_support::prompt(max_len - 200, config.vocab);
        let _ = g.generate(&p, 4).unwrap();
        free("traced generation");
        let ids = |t: &[i32]| {
            burn::tensor::Tensor::<TtBackend, 2, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::new(t.to_vec(), [1, t.len()]),
                &d,
            )
        };
        let at = |v: usize| {
            burn::tensor::Tensor::<TtBackend, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::new(vec![v as i32], [1]),
                &d,
            )
        };
        let mut cache = model.new_cache(max_len, &d);
        for (i, chunk) in schedule(p.len(), &shapes.chunks).into_iter().enumerate() {
            let mut t = p[chunk.start..chunk.start + chunk.n].to_vec();
            t.resize(chunk.c, 0);
            let len = shapes.chunk_len(chunk);
            let (l, state) = model.chunk(ids(&t), at(chunk.start), at(chunk.n - 1), len, &cache);
            let _ = l.into_data();
            cache.set_state(state);
            if i % 32 == 0 {
                free(&format!("eager chunk {i} at {}", chunk.start));
            }
        }
        free("eager prefill");
        for step in 0..3 {
            let position = p.len() + step;
            let (l, state) = model.step(ids(&[3]), at(position), shapes.step_len(position), &cache);
            let _ = l.into_data();
            cache.set_state(state);
            free(&format!("eager step {step}"));
        }
    });
}
