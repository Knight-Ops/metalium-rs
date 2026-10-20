//! Phase 1 of `docs/plans/planned-l1-residency.md`: a captured graph's residency plan, made and
//! reported as each capture ends (`tt_kernels::residency`), with nothing made resident.
//!
//! The claims, over the traced decode steps and prefill chunk of a synthetic four-layer Llama on
//! `Tt` (step197's), fused and not:
//! - **The graph accounts for every byte.** Each op's GDDR bytes read off its streams, the capture's
//!   own tensors' and the rest (weights, the cache, the inputs), equal what the trace counts it moving
//!   (`OpBytes`, counted independently at the store); nothing is left unattributed.
//! - **The plan keeps its rules** (`Plan::check`): no two resident tensors share bytes while both
//!   live, the region fits, and every repeat of the layer sees the same addresses.
//! - **A layer is the period,** and the residual stream crosses it resident.
//! - **Only what the graph makes and uses up is resident:** the logits and the cache are not the
//!   capture's to plan, and every resident tensor was freed inside it.
//! - **The plan is the same each time:** a second build plans every trace identically.
//! - **The negative control:** each plan with one resident tensor moved onto another live with it is
//!   refused.
//!
//! And Phase 2a's region, still empty (`burn_tt::set_l1_region`): with 256 KB kept at the top of
//! each unit's arena, every captured op plans below it (the products' `B` rings, at the arena's end
//! without it, move down), and the traced generation is bit for bit the one without. The negative
//! control, on a session: a region that leaves an op too little fails the op by name ("does not
//! fit"), no trace is kept, and the op run after it, outside a capture, has the whole arena again.

use burn_tt::{Tt, TtDevice};
use tt_kernels::residency::{self, Graph, Hazard, Place, Plan};
use tt_models::generate::{Shapes, TracedGenerator};
use tt_models::llama::{Llama, LlamaConfig};
use tt_tests::burn_device::{with_device, Config};

mod ladder_support;

fn config() -> LlamaConfig {
    LlamaConfig {
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
    }
}

/// A trace's name and ops.
type Traced = (String, Vec<burn_tt::OpRecord>);

/// Each trace's ops, and the plans its captures made, in capture order.
fn build(model: &Llama<Tt>, d: &TtDevice) -> (Vec<Traced>, Vec<(Graph, Plan)>) {
    residency::take_plans();
    residency::probe_residency_plan(true);
    let g = TracedGenerator::<_, Tt>::new(model, &Shapes::new(16, &[4]), d).unwrap();
    residency::probe_residency_plan(false);
    (g.trace_ops().unwrap(), residency::take_plans())
}

/// The trace whose ops are the graph's.
fn trace_of<'a>(traces: &'a [Traced], g: &Graph) -> &'a Traced {
    traces
        .iter()
        .find(|(_, ops)| {
            ops.len() == g.ops.len()
                && ops
                    .iter()
                    .zip(&g.ops)
                    .all(|(o, p)| o.label == p.label && o.what == p.what && o.l1_top == p.l1_top)
        })
        .expect("a trace for every plan")
}

#[test]
fn a_captured_graph_is_planned_and_reported() {
    with_device(Config::default(), |d| {
        burn_tt::set_count_trace_bytes(d, true).unwrap();
        let unfused = Llama::<Tt>::new(&config(), &d);
        for fused in [false, true] {
            let model = if fused {
                unfused.clone().fused()
            } else {
                unfused.clone()
            };
            let (traces, plans) = build(&model, &d);
            assert_eq!(plans.len(), traces.len(), "a plan for every capture");
            for (g, plan) in &plans {
                let (key, ops) = trace_of(&traces, g);
                let what = format!("fused {fused}, {key}");
                eprint!("{what}:\n{}", plan.report(g));
                // Every byte, the capture's own tensors' and the rest.
                assert_eq!(g.unattributed, 0, "{what}");
                let mut local = vec![[0u64; 2]; g.ops.len()];
                for t in &g.tensors {
                    for u in &t.uses {
                        local[u.op][0] += u.bytes[0];
                        local[u.op][1] += u.bytes[1];
                    }
                }
                for (k, (o, rec)) in g.ops.iter().zip(ops).enumerate() {
                    assert_eq!(
                        [local[k][0] + o.external[0], local[k][1] + o.external[1]],
                        [rec.bytes.read, rec.bytes.write],
                        "{what}: op {k} ({}/{})",
                        o.label,
                        o.what
                    );
                }
                // The rules.
                plan.check(g).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(plan.resident().count() > 0, "{what}: nothing resident");
                for t in plan.resident() {
                    assert!(
                        g.tensors[t].internal,
                        "{what}: tensor {t} outlives the capture"
                    );
                }
                // The weights and the cache are external: every product's `B` among them.
                let external: u64 = g.ops.iter().map(|o| o.external[0]).sum();
                let b: u64 = ops.iter().map(|o| o.bytes.b).sum();
                assert!(
                    external >= b,
                    "{what}: external {external} B, products' B {b} B"
                );
                // A layer is the period, and the residual stream crosses it: some resident tensor
                // is read in a later repeat than it was written in.
                if key.starts_with("Step") {
                    let per = plan.period.unwrap_or_else(|| panic!("{what}: no period"));
                    assert!(
                        per.repeats >= config().layers - 1,
                        "{what}: {per:?} against {} layers",
                        config().layers
                    );
                    let crosses = plan.resident().any(|t| {
                        let (a, b) = g.tensors[t].lifetime().unwrap();
                        let (pa, pb) = (per.phase(a), per.phase(b));
                        pa.is_some()
                            && pb.is_some()
                            && (b - per.start) / per.p > (a - per.start) / per.p
                    });
                    assert!(crosses, "{what}: no resident tensor crosses a layer's end");
                }
                // The negative control: a resident tensor moved onto another live at once.
                let residents: Vec<usize> = plan.resident().collect();
                let pair = residents.iter().find_map(|&a| {
                    let (sa, ea) = g.tensors[a].lifetime().unwrap();
                    residents
                        .iter()
                        .copied()
                        .find(|&b| {
                            let (sb, eb) = g.tensors[b].lifetime().unwrap();
                            b != a && sa <= eb && sb <= ea
                        })
                        .map(|b| (a, b))
                });
                if let Some((a, b)) = pair {
                    let mut bad = plan.clone();
                    let Place::Resident { offset: to, .. } = plan.place[b] else {
                        unreachable!()
                    };
                    if let Place::Resident { offset, .. } = &mut bad.place[a] {
                        *offset = to;
                    }
                    assert!(
                        matches!(
                            bad.check(g),
                            Err(Hazard::Overlap { .. } | Hazard::NotPeriodic { .. })
                        ),
                        "{what}: a slot shared by two live tensors was not refused: {:?}",
                        bad.check(g)
                    );
                }
            }
            // The same plans again.
            let (_, again) = build(&model, &d);
            let sigs = |p: &[(Graph, Plan)]| -> Vec<u64> {
                p.iter().map(|(g, p)| p.signature(g)).collect()
            };
            assert_eq!(
                sigs(&plans),
                sigs(&again),
                "fused {fused}: the plans differ between builds"
            );
        }
        burn_tt::set_count_trace_bytes(d, false).unwrap();
    });
}

#[test]
fn a_capture_keeps_its_region_below_every_op() {
    with_device(Config::default(), |d| {
        let model = Llama::<Tt>::new(&config(), &d).fused();
        let (shapes, prompt) = (Shapes::new(16, &[4]), [3, 14, 15, 92, 65]);
        let tops = |g: &TracedGenerator<'_, Llama<Tt>, Tt>| -> Vec<u64> {
            g.trace_ops()
                .unwrap()
                .into_iter()
                .flat_map(|(_, ops)| ops.into_iter().map(|o| o.l1_top))
                .collect()
        };
        // None to begin with, whatever `TT_L1_REGION` set.
        burn_tt::set_l1_region(d, 0).unwrap();
        let plain = TracedGenerator::<_, Tt>::new(&model, &shapes, &d).unwrap();
        let want = plain.generate(&prompt, 4).unwrap();
        let region = 256 * 1024;
        let base = tt_isa::l1::DATA.end - region;
        assert!(
            tops(&plain).iter().any(|&t| t > base),
            "without a region some op plans above {base:#x}, or the region moves nothing"
        );
        burn_tt::set_l1_region(d, region).unwrap();
        let kept = TracedGenerator::<_, Tt>::new(&model, &shapes, &d).unwrap();
        for top in tops(&kept) {
            assert!(
                top != 0 && top <= base,
                "an op planned to {top:#x}, past {base:#x}"
            );
        }
        let got = kept.generate(&prompt, 4).unwrap();
        ladder_support::assert_same(&got, &want, "with a 256 KB region");
        burn_tt::set_l1_region(d, 0).unwrap();
    });
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut tt_kernels::session::Session<tt_ttsim::LibTtsim<'_>>)) {
    use tt_kernels::session::{Session, TileChoice};
    if let Err(e) = tt_ttsim::fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
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
fn with_session(f: impl FnOnce(&mut tt_kernels::session::Session<tt_kmd::Kmd>)) {
    use tt_kernels::session::{Session, TileChoice};
    if let Err(e) = tt_ttsim::fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[test]
fn a_capture_with_too_little_below_its_region_is_refused() {
    with_session(|s| {
        let add = tt_kernels::tensor::Eltwise {
            kind: tt_kernels::kind::ADD,
            scalar: 0.0,
            scalar2: 0.0,
        };
        let (rows, cols) = (64, 1024);
        let a: Vec<f32> = (0..rows * cols).map(|i| (i % 97) as f32).collect();
        let x = s.upload(&a, rows, cols).unwrap();
        // One tile slot left below the region: an element-wise op's three rings do not fit.
        s.set_l1_region(tt_isa::l1::DATA.len() - tt_isa::dm::TILE_SLOT);
        s.begin_trace().unwrap();
        let failed = s.eltwise(add, &x, Some(&x));
        let refused = s.end_trace();
        eprintln!(
            "op: {:?}; capture: {:?}",
            failed.as_ref().err(),
            refused.as_ref().err()
        );
        let why = failed
            .expect_err("an op planned with one slot of arena")
            .to_string();
        assert!(why.contains("does not fit"), "{why}");
        assert!(refused.is_err(), "a capture whose op did not fit was kept");
        // Outside a capture the whole arena is the op's again.
        s.set_l1_region(0);
        let y = s.eltwise(add, &x, Some(&x)).unwrap();
        let want: Vec<u32> = a.iter().map(|v| (v + v).to_bits()).collect();
        let got: Vec<u32> = s
            .download(&y)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, want);
        for t in [y, x] {
            s.free(t).unwrap();
        }
    });
}
