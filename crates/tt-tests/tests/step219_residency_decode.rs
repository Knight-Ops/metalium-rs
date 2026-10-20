//! Phase 2 of `docs/plans/planned-l1-residency.md`: a planned capture keeps its graph's tensors
//! in the units' L1 between the ops that make and read them (`Session::set_residency`,
//! `Session::begin_trace_planned`, `tt_kernels::residency::Applied`).
//!
//! The claims, on a session, over a decode row's residual stream (an add, RMSNorm, `gamma`'s
//! multiply, the next add):
//! - **Bit for bit.** The planned trace's capture and replays over new inputs equal the ops run
//!   fresh, as the plain trace's do.
//! - **Through L1, not GDDR.** The planned trace moves fewer GDDR bytes than the plain one, by what
//!   its resident tensors moved: none of them is read or written in GDDR.
//! - **A capture that is not its plan's is refused by name**, keeps nothing, and the session runs
//!   on: a planned capture of another graph.
//! - **The negative control:** the plan with one resident tensor moved onto another still to be
//!   read is refused by `Plan::check`, and, applied anyway, gives other bits than the ops run fresh.
//!
//! And through Burn: a traced generation of a synthetic four-layer Llama on `Tt` (step197's), its
//! decode steps and prefill chunk captured twice with residency on (`burn_tt::set_residency`,
//! `SharedStateTraces::capture`), gives the tokens and the logits of the generation without, bit
//! for bit, and each of its traces moves fewer GDDR bytes than the same trace without.

use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_ttsim::fork_scope;

const K: usize = 2048;
const EPS: f32 = 1e-5;

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(8),
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
            TileChoice::Count(8),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn op(kind: u32) -> Eltwise {
    Eltwise {
        kind,
        scalar: 0.0,
        scalar2: 0.0,
    }
}

/// `x + a`, its norm times `gamma`, plus `x + a` again: what a layer's residual stream does
/// between two products. Frees what it made but the result.
fn step<T: tt_device::Transport>(
    s: &mut Session<T>,
    x: &DramTensor,
    a: &DramTensor,
    gamma: &DramTensor,
) -> DramTensor {
    use tt_kernels::kind;
    let h = s.eltwise(op(kind::ADD), x, Some(a)).unwrap();
    let n = s.rms_norm(&h, EPS).unwrap();
    let g = s.eltwise(op(kind::MUL), &n, Some(gamma)).unwrap();
    let out = s.eltwise(op(kind::ADD), &g, Some(&h)).unwrap();
    for t in [h, n, g] {
        s.free(t).unwrap();
    }
    out
}

/// The capture's own result and its replays' over each seed after the first, and the GDDR bytes
/// its ops move.
fn run_trace<T: tt_device::Transport>(
    s: &mut Session<T>,
    id: tt_kernels::trace::TraceId,
    x: &DramTensor,
    out: &DramTensor,
    seeds: &[u64],
) -> (Vec<Vec<u32>>, u64) {
    let mut got = vec![bits(&s.download(out).unwrap())];
    for &seed in &seeds[1..] {
        s.write(x, &values(seed, K)).unwrap();
        s.replay(id).unwrap();
        s.sync().unwrap();
        got.push(bits(&s.download(out).unwrap()));
    }
    let bytes = s
        .trace_ops(id)
        .unwrap()
        .iter()
        .map(|o| o.bytes.read + o.bytes.write)
        .sum();
    (got, bytes)
}

#[test]
fn a_planned_capture_keeps_the_residual_stream_in_l1_bit_for_bit() {
    with_session(|s| {
        s.set_count_trace_bytes(true);
        let seeds = [10u64, 20, 21];
        let a = s.upload(&values(2, K), 1, K).unwrap();
        let gamma = s.upload(&values(3, K), 1, K).unwrap();
        let x = s.upload(&values(seeds[0], K), 1, K).unwrap();
        // Fresh.
        let mut fresh = Vec::new();
        for &seed in &seeds {
            s.write(&x, &values(seed, K)).unwrap();
            let out = step(s, &x, &a, &gamma);
            fresh.push(bits(&s.download(&out).unwrap()));
            s.free(out).unwrap();
        }
        // The plain capture, which plans.
        s.set_residency(tt_kernels::session::RESIDENCY_REGION);
        s.write(&x, &values(seeds[0], K)).unwrap();
        s.begin_trace().unwrap();
        let out = step(s, &x, &a, &gamma);
        let plain = s.end_trace().unwrap();
        let (got, plain_bytes) = run_trace(s, plain, &x, &out, &seeds);
        assert_eq!(got, fresh, "the plain trace");
        // The planned capture of the same step.
        s.write(&x, &values(seeds[0], K)).unwrap();
        assert!(
            s.begin_trace_planned().unwrap(),
            "a plan with something resident"
        );
        let planned_out = step(s, &x, &a, &gamma);
        let planned = s.end_trace().unwrap();
        let (got, planned_bytes) = run_trace(s, planned, &x, &planned_out, &seeds);
        eprintln!("GDDR bytes: plain {plain_bytes}, planned {planned_bytes}");
        assert_eq!(got, fresh, "the planned trace");
        assert!(
            planned_bytes < plain_bytes,
            "the planned trace moves {planned_bytes} B of GDDR, the plain {plain_bytes}"
        );
        for id in [plain, planned] {
            s.release_trace(id).unwrap();
        }
        for t in [out, planned_out, x, a, gamma] {
            s.free(t).unwrap();
        }
    });
}

/// `y = x + a`, `z = y * gamma`, `z + y`: `y` waits in L1 while `z` is written.
fn chain<T: tt_device::Transport>(
    s: &mut Session<T>,
    x: &DramTensor,
    a: &DramTensor,
    gamma: &DramTensor,
) -> DramTensor {
    use tt_kernels::kind;
    let y = s.eltwise(op(kind::ADD), x, Some(a)).unwrap();
    let z = s.eltwise(op(kind::MUL), &y, Some(gamma)).unwrap();
    let out = s.eltwise(op(kind::ADD), &z, Some(&y)).unwrap();
    for t in [y, z] {
        s.free(t).unwrap();
    }
    out
}

#[test]
fn a_slot_reused_before_its_last_reader_gives_other_bits() {
    use tt_kernels::residency::Place;
    with_session(|s| {
        let a = s.upload(&values(2, K), 1, K).unwrap();
        let gamma = s.upload(&values(3, K), 1, K).unwrap();
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let out = chain(s, &x, &a, &gamma);
        let fresh = bits(&s.download(&out).unwrap());
        s.free(out).unwrap();
        // Rows read whole on many units too, so more is resident (read as the plan is made).
        std::env::set_var("TT_RESIDENCY_REPLICATED", "1");
        s.set_residency(tt_kernels::session::RESIDENCY_REGION);
        s.begin_trace().unwrap();
        let plain_out = chain(s, &x, &a, &gamma);
        let plain = s.end_trace().unwrap();
        // A tensor written while another, already resident, is still to be read: put it there.
        let mut moved = None;
        assert!(s.probe_mutate_plan(|g, plan| {
            let resident: Vec<usize> = plan.resident().collect();
            let pair = resident.iter().find_map(|&b| {
                let (sb, _) = g.tensors[b].lifetime()?;
                resident
                    .iter()
                    .copied()
                    .find(|&a| {
                        let (sa, ea) = g.tensors[a].lifetime().unwrap();
                        a != b && sa < sb && sb < ea
                    })
                    .map(|a| (a, b))
            });
            let (a, b) = pair.expect("two resident tensors, one written while the other waits");
            let Place::Resident { offset: to, .. } = plan.place[a] else {
                unreachable!()
            };
            if let Place::Resident { offset, .. } = &mut plan.place[b] {
                *offset = to;
            }
            assert!(
                plan.check(g).is_err(),
                "the check took a slot reused before its reader"
            );
            moved = Some((a, b));
        }));
        eprintln!("tensor {:?} moved onto the other", moved);
        assert!(s.begin_trace_planned().unwrap());
        let planned_out = chain(s, &x, &a, &gamma);
        let planned = s.end_trace().unwrap();
        let got = bits(&s.download(&planned_out).unwrap());
        assert_ne!(
            got, fresh,
            "a slot overwritten before its reader changed nothing"
        );
        for id in [plain, planned] {
            s.release_trace(id).unwrap();
        }
        for t in [plain_out, planned_out, x, a, gamma] {
            s.free(t).unwrap();
        }
    });
}

#[test]
fn a_planned_capture_of_another_graph_is_refused() {
    with_session(|s| {
        use tt_kernels::kind;
        let a = s.upload(&values(2, K), 1, K).unwrap();
        let gamma = s.upload(&values(3, K), 1, K).unwrap();
        let x = s.upload(&values(10, K), 1, K).unwrap();
        s.set_residency(tt_kernels::session::RESIDENCY_REGION);
        s.begin_trace().unwrap();
        let out = step(s, &x, &a, &gamma);
        let plain = s.end_trace().unwrap();
        // Planned, but one op more at the start: every op after it is out of place.
        assert!(s.begin_trace_planned().unwrap());
        let extra = s.eltwise(op(kind::MUL), &x, Some(&a)).unwrap();
        let other = step(s, &x, &a, &gamma);
        let refused = s.end_trace().expect_err("a capture of another graph");
        let why = refused.to_string();
        eprintln!("refused: {why}");
        assert!(why.contains("not its plan's"), "{why}");
        // The session runs on: the plain trace replays, and an op runs fresh.
        s.replay(plain).unwrap();
        s.sync().unwrap();
        let fresh = step(s, &x, &a, &gamma);
        assert_eq!(
            bits(&s.download(&out).unwrap()),
            bits(&s.download(&fresh).unwrap())
        );
        s.release_trace(plain).unwrap();
        for t in [out, extra, other, fresh, x, a, gamma] {
            s.free(t).unwrap();
        }
    });
}

mod ladder_support;

#[test]
fn a_traced_decoder_keeps_its_planned_traces_bit_for_bit() {
    use burn_tt::{Tt, TtDevice};
    use tt_models::generate::{Shapes, TracedGenerator};
    use tt_models::llama::{Llama, LlamaConfig};
    use tt_tests::burn_device::{with_device, Config};
    let config = LlamaConfig {
        hidden: 64,
        heads: 2,
        kv_heads: 2,
        head_dim: 32,
        ff: 96,
        vocab: 100,
        layers: 4,
        rope_theta: 10_000.0,
        eps: 1e-5,
        tied_head: true,
    };
    let bytes = |g: &TracedGenerator<'_, Llama<Tt>, Tt>| -> Vec<(String, u64)> {
        g.trace_ops()
            .unwrap()
            .into_iter()
            .map(|(k, ops)| (k, ops.iter().map(|o| o.bytes.read + o.bytes.write).sum()))
            .collect()
    };
    with_device(Config::default(), |d: TtDevice| {
        burn_tt::set_count_trace_bytes(d, true).unwrap();
        let model = Llama::<Tt>::new(&config, &d).fused();
        let (shapes, prompt) = (Shapes::new(16, &[4]), [3, 14, 15, 92, 65]);
        burn_tt::set_residency(d, 0).unwrap();
        let plain = TracedGenerator::<_, Tt>::new(&model, &shapes, &d).unwrap();
        let want = plain.generate(&prompt, 4).unwrap();
        let plain_bytes = bytes(&plain);
        drop(plain);
        burn_tt::set_residency(d, tt_kernels::session::RESIDENCY_REGION).unwrap();
        let planned = TracedGenerator::<_, Tt>::new(&model, &shapes, &d).unwrap();
        let got = planned.generate(&prompt, 4).unwrap();
        ladder_support::assert_same(&got, &want, "residency-planned traces");
        let planned_bytes = bytes(&planned);
        for ((key, p), (k2, q)) in plain_bytes.iter().zip(&planned_bytes) {
            assert_eq!(key, k2);
            eprintln!("{key}: GDDR {p} B plain, {q} B planned");
            assert!(
                q < p,
                "{key}: the planned trace moves {q} B, the plain {p} B"
            );
        }
        burn_tt::set_residency(d, 0).unwrap();
        burn_tt::set_count_trace_bytes(d, false).unwrap();
    });
}
