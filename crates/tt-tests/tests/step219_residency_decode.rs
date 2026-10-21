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
//! - **Never-read writes dropped** (2g): the tensors the step makes and never reads are written
//!   nowhere, bit for bit, fewer GDDR bytes written; a plan dropping a read tensor is refused.
//! - **A row read whole, held by two of its readers** (2f, `Rows::Held(2)`): a multiply's packed
//!   twin, read by the one-row product on each of its units, is held by two of them; the planned
//!   trace equals the ops run fresh bit for bit and moves fewer GDDR bytes. A reader sent to a unit
//!   that holds nothing is refused by `Plan::check` and, applied anyway, by the rewrite.
//! - **The region sized by its plan** (2h): the plan pass keeps no region, the planned capture
//!   what its plan uses, under `residency::cap` (what leaves every op laid out as in the plan
//!   pass). Bit for bit at the plan's region and at exactly the cap; 64 bytes past the cap, a
//!   product's blocks narrow and the capture is refused by name.
//! - **Plans cached by key** (2h): a plan kept under its graph's key is applied in one capture,
//!   bit for bit; another graph under the key is refused by name and the key forgotten. Through
//!   Burn, a second build captures each trace once.
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
        s.begin_trace_plan_pass().unwrap();
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
        // Rows read whole on many units too (held by two of them), so more is resident.
        s.set_residency_rows(tt_kernels::residency::Rows::Held(2));
        s.set_residency(tt_kernels::session::RESIDENCY_REGION);
        s.begin_trace_plan_pass().unwrap();
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
        s.begin_trace_plan_pass().unwrap();
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

/// `y = x + a`, `z = y * gamma`, `z @ w`: the multiply writes `z` and its packed twin, which the
/// one-row product reads whole on each of its units (2f's row).
fn into_product<T: tt_device::Transport>(
    s: &mut Session<T>,
    x: &DramTensor,
    a: &DramTensor,
    gamma: &DramTensor,
    w: &DramTensor,
) -> DramTensor {
    use tt_kernels::kind;
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    let y = s.eltwise(op(kind::ADD), x, Some(a)).unwrap();
    let z = s.eltwise(op(kind::MUL), &y, Some(gamma)).unwrap();
    let out = s
        .matmul_dram(
            &z,
            false,
            w,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            tt_tests::harness::BUDGET,
        )
        .unwrap();
    for t in [y, z] {
        s.free(t).unwrap();
    }
    out
}

/// 2f: the packed twin the product reads whole on its units is held by two of them
/// (`Rows::Held(2)`): bit for bit, fewer GDDR bytes than the plain trace, and the negative control
/// -- a reader sent to a unit that holds nothing -- refused by the check and, applied anyway, by
/// the rewrite, which finds no copy there to read: the planned capture is refused by name.
#[test]
fn a_row_read_whole_is_held_by_two_units_bit_for_bit() {
    use tt_kernels::residency::{Hazard, Rows};
    with_session(|s| {
        s.set_count_trace_bytes(true);
        const N: usize = 256;
        let seeds = [30u64, 31, 32];
        let a = s.upload(&values(2, K), 1, K).unwrap();
        let gamma = s.upload(&values(3, K), 1, K).unwrap();
        let w = s.upload(&values(4, K * N), K, N).unwrap();
        let x = s.upload(&values(seeds[0], K), 1, K).unwrap();
        let mut fresh = Vec::new();
        for &seed in &seeds {
            s.write(&x, &values(seed, K)).unwrap();
            let out = into_product(s, &x, &a, &gamma, &w);
            fresh.push(bits(&s.download(&out).unwrap()));
            s.free(out).unwrap();
        }
        s.set_residency_rows(Rows::Held(2));
        s.set_residency(tt_kernels::session::RESIDENCY_REGION);
        s.write(&x, &values(seeds[0], K)).unwrap();
        s.begin_trace_plan_pass().unwrap();
        let out = into_product(s, &x, &a, &gamma, &w);
        let plain = s.end_trace().unwrap();
        let (got, plain_bytes) = run_trace(s, plain, &x, &out, &seeds);
        assert_eq!(got, fresh, "the plain trace");
        // The plan holds a row on two of its readers: the twin.
        let mut held = 0;
        assert!(s.probe_mutate_plan(|g, plan| {
            for (&t, h) in &plan.held {
                assert_eq!(h.holders.len(), 2, "tensor {t}");
                assert!(g.tensors[t].replicated() && g.tensors[t].read_alike());
                assert!(h.of.len() > 2, "tensor {t} is read by {} units", h.of.len());
            }
            held = plan.held.len();
        }));
        assert!(held > 0, "no row held");
        s.write(&x, &values(seeds[0], K)).unwrap();
        assert!(s.begin_trace_planned().unwrap());
        let planned_out = into_product(s, &x, &a, &gamma, &w);
        let planned = s.end_trace().unwrap();
        let (got, planned_bytes) = run_trace(s, planned, &x, &planned_out, &seeds);
        eprintln!("GDDR bytes: plain {plain_bytes}, planned {planned_bytes}; {held} rows held");
        assert_eq!(got, fresh, "the planned trace");
        assert!(planned_bytes < plain_bytes);
        for id in [plain, planned] {
            s.release_trace(id).unwrap();
        }
        // The negative control: plan again, then send one reader to a unit that holds nothing.
        s.write(&x, &values(seeds[0], K)).unwrap();
        s.begin_trace_plan_pass().unwrap();
        let again = into_product(s, &x, &a, &gamma, &w);
        let plain = s.end_trace().unwrap();
        assert!(s.probe_mutate_plan(|g, plan| {
            let (&t, h) = plan.held.iter().next().expect("a held row");
            let reader = *h
                .of
                .keys()
                .find(|r| !h.holders.contains(r))
                .expect("a reader that holds nothing");
            let elsewhere = *h
                .of
                .keys()
                .find(|&&r| r != reader && !h.holders.contains(&r))
                .expect("another");
            plan.held.get_mut(&t).unwrap().of.insert(reader, elsewhere);
            assert_eq!(plan.check(g), Err(Hazard::NotHeld { tensor: t }));
        }));
        // A unit with no copy has no address to read from: the rewrite stops, and the planned
        // capture is refused by name.
        assert!(s.begin_trace_planned().unwrap());
        let wrong_out = into_product(s, &x, &a, &gamma, &w);
        let why = s
            .end_trace()
            .expect_err("a reader sent to no holder")
            .to_string();
        eprintln!("refused: {why}");
        assert!(
            why.contains("not its plan's") && why.contains("not hold"),
            "{why}"
        );
        s.release_trace(plain).unwrap();
        for t in [out, planned_out, again, wrong_out, x, a, gamma, w] {
            s.free(t).unwrap();
        }
    });
}

/// 2h: the region is sized by the plan pass, never more than leaves every op laid out as in it
/// (`residency::cap`), nor than the 96 KiB ceiling. Here a product whose `K` = 2048 and whose
/// columns outnumber its units twice over, so a narrower block is a different deal: its widest
/// block's probe leaves 110,464 bytes of `DATA`, which is the graph's cap. The planned capture of
/// the plan's own region is bit for bit; so is one keeping exactly the cap (the rule is tight);
/// **the negative control:** one keeping 64 bytes past it narrows the product, and the capture is
/// refused by name, keeping the plain trace.
#[test]
fn the_region_is_sized_by_its_plan_and_never_past_its_cap() {
    use tt_isa::l1::DATA;
    use tt_kernels::session::RESIDENCY_REGION;
    with_session(|s| {
        s.set_count_trace_bytes(true);
        const N: usize = 1024;
        let seeds = [50u64, 51];
        let a = s.upload(&values(2, K), 1, K).unwrap();
        let gamma = s.upload(&values(3, K), 1, K).unwrap();
        let w = s.upload(&values(4, K * N), K, N).unwrap();
        let x = s.upload(&values(seeds[0], K), 1, K).unwrap();
        let mut fresh = Vec::new();
        for &seed in &seeds {
            s.write(&x, &values(seed, K)).unwrap();
            let out = into_product(s, &x, &a, &gamma, &w);
            fresh.push(bits(&s.download(&out).unwrap()));
            s.free(out).unwrap();
        }
        s.set_residency(RESIDENCY_REGION);
        // A plan pass, then its planned capture keeping `region` (`None`: the plan's own); the
        // planned trace's results, or why it was refused.
        let capture = |s: &mut Session<_>, region: Option<u64>| {
            s.probe_planned_region(region);
            s.write(&x, &values(seeds[0], K)).unwrap();
            s.begin_trace_plan_pass().unwrap();
            let out = into_product(s, &x, &a, &gamma, &w);
            let plain = s.end_trace().unwrap();
            // The plan pass kept no region: its product took the widest block.
            assert!(s
                .trace_ops(plain)
                .unwrap()
                .iter()
                .any(|o| o.l1_top > DATA.end - 4096));
            let applied = s.probe_last_plan().expect("a plan");
            let cap = tt_kernels::residency::cap(&applied.graph, DATA.end);
            s.write(&x, &values(seeds[0], K)).unwrap();
            assert!(s.begin_trace_planned().unwrap());
            let planned_out = into_product(s, &x, &a, &gamma, &w);
            let result = s.end_trace().map(|planned| {
                let ops = s.trace_ops(planned).unwrap().to_vec();
                let (got, _) = run_trace(s, planned, &x, &planned_out, &seeds);
                s.release_trace(planned).unwrap();
                (got, ops)
            });
            s.release_trace(plain).unwrap();
            for t in [out, planned_out] {
                s.free(t).unwrap();
            }
            s.probe_planned_region(None);
            (applied, cap, result)
        };
        // The plan's own region: the cap, or the 96 KiB ceiling under it, what the plan uses
        // inside it.
        let (applied, cap, result) = capture(s, None);
        assert_eq!(
            cap, 110_464,
            "the K = 2048 product's widest block sets the cap"
        );
        let region = DATA.end - applied.region_base;
        eprintln!(
            "region {region} bytes (plan {}), cap {cap}",
            applied.plan.region
        );
        assert_eq!(region, cap.min(RESIDENCY_REGION));
        assert!(applied.plan.region > 0 && applied.plan.region <= region);
        let (got, ops) = result.expect("the plan's own region");
        assert_eq!(got, fresh, "the planned trace, in its plan's region");
        assert!(ops.iter().all(|o| o.l1_top <= applied.region_base));
        // Exactly the cap: every op laid out as in the plan pass.
        let (applied, _, result) = capture(s, Some(cap));
        assert_eq!(applied.region_base, DATA.end - cap);
        let (got, _) = result.expect("a region of exactly the cap");
        assert_eq!(got, fresh, "the planned trace, in a region of the cap");
        // The negative control: 64 bytes past it, the product's blocks narrow.
        let (_, _, result) = capture(s, Some(cap + 64));
        let why = result.expect_err("a region past the cap").to_string();
        eprintln!("refused: {why}");
        assert!(why.contains("laid out unlike the plan pass's"), "{why}");
        // The session runs on.
        let out = into_product(s, &x, &a, &gamma, &w);
        s.free(out).unwrap();
        for t in [x, a, gamma, w] {
            s.free(t).unwrap();
        }
    });
}

/// 2h: a plan kept under a key (`Session::begin_trace_planned_as`) is applied with no plan pass
/// (`Session::begin_trace_cached`), bit for bit. **The negative controls:** another graph captured
/// under the same key is refused by name, and the key forgotten; a graph rewired at its fourth op
/// is refused before that op is queued, and the session runs on bit for bit.
#[test]
fn a_cached_plan_is_applied_once_and_forgotten_for_another_graph() {
    use tt_kernels::session::RESIDENCY_REGION;
    with_session(|s| {
        let seeds = [60u64, 61];
        let a = s.upload(&values(2, K), 1, K).unwrap();
        let gamma = s.upload(&values(3, K), 1, K).unwrap();
        let x = s.upload(&values(seeds[0], K), 1, K).unwrap();
        let mut fresh = Vec::new();
        for &seed in &seeds {
            s.write(&x, &values(seed, K)).unwrap();
            let out = step(s, &x, &a, &gamma);
            fresh.push(bits(&s.download(&out).unwrap()));
            s.free(out).unwrap();
        }
        const KEY: u64 = 0x2b;
        s.set_residency(RESIDENCY_REGION);
        assert!(!s.begin_trace_cached(KEY).unwrap(), "nothing kept yet");
        s.write(&x, &values(seeds[0], K)).unwrap();
        s.begin_trace_plan_pass().unwrap();
        let out = step(s, &x, &a, &gamma);
        let plain = s.end_trace().unwrap();
        s.write(&x, &values(seeds[0], K)).unwrap();
        assert!(s.begin_trace_planned_as(KEY).unwrap());
        let planned_out = step(s, &x, &a, &gamma);
        let planned = s.end_trace().unwrap();
        // Kept: one capture, under the plan.
        let before = s.probe_captures();
        s.write(&x, &values(seeds[0], K)).unwrap();
        assert!(s.begin_trace_cached(KEY).unwrap());
        let cached_out = step(s, &x, &a, &gamma);
        let cached = s.end_trace().unwrap();
        assert_eq!(s.probe_captures() - before, 1);
        let (got, _) = run_trace(s, cached, &x, &cached_out, &seeds);
        assert_eq!(got, fresh, "the trace captured under the cached plan");
        // The negative control: another graph under the key.
        s.write(&x, &values(seeds[0], K)).unwrap();
        assert!(s.begin_trace_cached(KEY).unwrap());
        let other = s
            .eltwise(op(tt_kernels::kind::ADD), &x, Some(&gamma))
            .unwrap();
        let why = s
            .end_trace()
            .expect_err("another graph under the key")
            .to_string();
        eprintln!("refused: {why}");
        assert!(why.contains("not its plan's"), "{why}");
        assert!(!s.begin_trace_cached(KEY).unwrap(), "the key forgotten");
        // A graph the plan's for three ops, its last add reading the norm where the plan's reads
        // the sum: refused before that op is queued, which would read from GDDR what the plan put
        // in L1, and nothing after it runs. The session runs on, bit for bit.
        let rewired = |s: &mut Session<_>| {
            use tt_kernels::kind;
            let h = s.eltwise(op(kind::ADD), &x, Some(&a)).unwrap();
            let n = s.rms_norm(&h, EPS).unwrap();
            let g = s.eltwise(op(kind::MUL), &n, Some(&gamma)).unwrap();
            let out = s.eltwise(op(kind::ADD), &g, Some(&n)).unwrap();
            for t in [h, n, g] {
                s.free(t).unwrap();
            }
            out
        };
        s.write(&x, &values(seeds[0], K)).unwrap();
        s.begin_trace_plan_pass().unwrap();
        let again = step(s, &x, &a, &gamma);
        let plain_again = s.end_trace().unwrap();
        s.write(&x, &values(seeds[0], K)).unwrap();
        assert!(s.begin_trace_planned_as(KEY).unwrap());
        let wrong = rewired(s);
        let why = s
            .end_trace()
            .expect_err("a graph rewired at op 3")
            .to_string();
        eprintln!("refused: {why}");
        assert!(why.contains("op 3 moves other tensors"), "{why}");
        s.write(&x, &values(seeds[1], K)).unwrap();
        let after = step(s, &x, &a, &gamma);
        assert_eq!(
            bits(&s.download(&after).unwrap()),
            fresh[1],
            "the session after"
        );
        s.release_trace(plain_again).unwrap();
        for t in [again, wrong, after] {
            s.free(t).unwrap();
        }
        // A settings change forgets every plan.
        s.set_residency(RESIDENCY_REGION);
        for id in [plain, planned, cached] {
            s.release_trace(id).unwrap();
        }
        for t in [out, planned_out, cached_out, other, x, a, gamma] {
            s.free(t).unwrap();
        }
    });
}

/// The GDDR bytes a trace's ops write.
fn written<T: tt_device::Transport>(s: &mut Session<T>, id: tt_kernels::trace::TraceId) -> u64 {
    s.trace_ops(id).unwrap().iter().map(|o| o.bytes.write).sum()
}

/// 2g: the step's never-read tensors (a row's packed twin no op reads, beside its tiles) are
/// written nowhere in the planned trace: bit for bit, fewer GDDR bytes written than the same plan
/// with the drops off. The negative control: a plan dropping a tensor that is read is refused by
/// its check and, applied anyway, by the rewrite, which finds the read; the planned capture is
/// refused by name.
#[test]
fn never_read_writes_are_dropped_bit_for_bit() {
    use tt_kernels::residency::Hazard;
    with_session(|s| {
        s.set_count_trace_bytes(true);
        let seeds = [40u64, 41, 42];
        let a = s.upload(&values(2, K), 1, K).unwrap();
        let gamma = s.upload(&values(3, K), 1, K).unwrap();
        let x = s.upload(&values(seeds[0], K), 1, K).unwrap();
        let mut fresh = Vec::new();
        for &seed in &seeds {
            s.write(&x, &values(seed, K)).unwrap();
            let out = step(s, &x, &a, &gamma);
            fresh.push(bits(&s.download(&out).unwrap()));
            s.free(out).unwrap();
        }
        s.set_residency(tt_kernels::session::RESIDENCY_REGION);
        let mut writes = Vec::new();
        for drop in [false, true] {
            s.set_residency_drop(drop);
            s.write(&x, &values(seeds[0], K)).unwrap();
            s.begin_trace_plan_pass().unwrap();
            let out = step(s, &x, &a, &gamma);
            let plain = s.end_trace().unwrap();
            let mut dropped = 0;
            assert!(s.probe_mutate_plan(|_, plan| dropped = plan.dropped.len()));
            assert_eq!(dropped > 0, drop, "tensors dropped: {dropped}");
            s.write(&x, &values(seeds[0], K)).unwrap();
            assert!(s.begin_trace_planned().unwrap());
            let planned_out = step(s, &x, &a, &gamma);
            let planned = s.end_trace().unwrap();
            let (got, _) = run_trace(s, planned, &x, &planned_out, &seeds);
            assert_eq!(got, fresh, "drops {drop}");
            writes.push(written(s, planned));
            for id in [plain, planned] {
                s.release_trace(id).unwrap();
            }
            for t in [out, planned_out] {
                s.free(t).unwrap();
            }
        }
        eprintln!(
            "GDDR bytes written: drops off {}, on {}",
            writes[0], writes[1]
        );
        assert!(writes[1] < writes[0]);
        // The negative control: drop a tensor some op reads.
        s.write(&x, &values(seeds[0], K)).unwrap();
        s.begin_trace_plan_pass().unwrap();
        let out = step(s, &x, &a, &gamma);
        let plain = s.end_trace().unwrap();
        assert!(s.probe_mutate_plan(|g, plan| {
            let t = (0..g.tensors.len())
                .find(|&t| {
                    g.tensors[t].internal
                        && g.tensors[t].uses.iter().any(|u| !u.reads.is_empty())
                        && g.tensors[t].uses.iter().all(|u| u.plain)
                        && !plan.resident().any(|r| r == t)
                })
                .expect("a tensor read in GDDR");
            plan.dropped.insert(t);
            assert_eq!(plan.check(g), Err(Hazard::ReadDropped { tensor: t }));
        }));
        assert!(s.begin_trace_planned().unwrap());
        let wrong = step(s, &x, &a, &gamma);
        let why = s
            .end_trace()
            .expect_err("a dropped tensor that is read")
            .to_string();
        eprintln!("refused: {why}");
        assert!(
            why.contains("not its plan's") && why.contains("dropped"),
            "{why}"
        );
        s.release_trace(plain).unwrap();
        for t in [out, wrong, x, a, gamma] {
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
        let before = burn_tt::probe_captures(d).unwrap();
        let planned = TracedGenerator::<_, Tt>::new(&model, &shapes, &d).unwrap();
        let first_build = burn_tt::probe_captures(d).unwrap() - before;
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
        drop(planned);
        // 2h: the plans are kept by their steps' keys, so a second build captures each trace
        // once, under its plan, and generates the same, bit for bit, moving the same bytes.
        let traces = planned_bytes.len() as u64;
        assert_eq!(
            first_build,
            2 * traces,
            "a plan pass and a planned capture each"
        );
        let before = burn_tt::probe_captures(d).unwrap();
        let again = TracedGenerator::<_, Tt>::new(&model, &shapes, &d).unwrap();
        assert_eq!(
            burn_tt::probe_captures(d).unwrap() - before,
            traces,
            "one capture a trace, its plan cached"
        );
        let got = again.generate(&prompt, 4).unwrap();
        ladder_support::assert_same(&got, &want, "traces captured under cached plans");
        assert_eq!(bytes(&again), planned_bytes);
        drop(again);
        burn_tt::set_residency(d, 0).unwrap();
        burn_tt::set_count_trace_bytes(d, false).unwrap();
    });
}
