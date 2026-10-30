//! A measured block choice (`docs/plans/dataflow-compiler.md` F4b.2, H9;
//! `tensor::BlockStream::Measured`): the first time a product's question is planned
//! (`tensor::BlockQuestion`), the session times its candidates (`tensor::block_stream_candidates`:
//! `Auto`'s block, unpulled and pulled, and the blocks measured fastest at prefill shapes), climbs
//! from the fastest one step at a time while a step gains (`tensor::block_stream_neighbours`), each
//! timed as a trace of the product alone (`Session::time_product`), and keeps the fastest for the
//! process. In a capture, the capture is paused for it (`Session::paused`).
//!
//! The claims, each product captured alone and replayed unless it says otherwise:
//! - **The choice is what was timed fastest, and the product is `Auto`'s bit for bit**: every
//!   candidate is the same product (ADR 0005). The gate scores the candidates itself
//!   (`Session::probe_block_scores`): the last listed is fastest and nothing off the list gains,
//!   so the session must time exactly the list and one round of its last's neighbours, each once,
//!   and choose the last, never `Auto`'s unpulled block: what runs is the measured choice.
//! - **The question the session asks is the one it says it asks** (`Session::block_question_of`).
//! - **A question is timed once a process**: a second product of the same shape scores nothing.
//! - **Only a trace's products are measured**: an eager product under `Measured` runs `Auto`'s
//!   block and times nothing (what is timed is a product inside a trace; an eager one also pays
//!   its planning every call).
//! - **A capture paused to measure is the capture unpaused**: a residency plan pass that measures
//!   inside it records the same ops and barriers, and on every unit the same records as the same
//!   plan pass once the choice is kept, but for one `op::KERNEL` setup run a pause on a unit
//!   whose tile state the unpaused pass carries from an earlier op (what the pause ran leaves
//!   that state unknown); its trace replays as the eager chain, bit for bit; and the planned
//!   capture applying its plan holds (it refuses a pass whose ops or tensors are not its own).
//! - **A measurement times its copies stored whole**, under a capture stored as layer sub-traces
//!   (silicon only).
//! - **The negative control:** a pause that does not move the capture's barrier targets and
//!   generations past what ran in it (`session::probe_pause_unrebased`) makes the paused plan
//!   pass's replay fail or differ.
//!
//! On ttsim the candidates' times say nothing, so the gate's scores choose; on silicon the times
//! do (`silicon_probe_pair_cost`, the bench of H9).
use std::sync::Mutex;

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, MEASURE_MOST};
use tt_kernels::tensor::{self, BlockChoice, BlockStream, DramTensor, Reuse};
use tt_kernels::tile_op::reuse::AUTO_HOLDERS;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const UNITS: usize = 8;

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            tt_kernels::session::TileChoice::Count(UNITS),
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
            tt_kernels::session::TileChoice::Count(UNITS),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Values whose sums round, so a product's terms and order show in its bits, with the special
/// values sprinkled in.
fn values(n: usize, seed: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n)
        .map(|i| ((i * 73 + seed) % 257) as f32 / 131.0 - 1.0 + ((i * 7 + seed) % 5) as f32 * 1e-3)
        .collect();
    let special = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(0x007f_ffff),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
    ];
    for (i, s) in special.into_iter().enumerate() {
        v[(i * 977 + seed * 31) % n] = s;
    }
    v
}

fn same(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: element {i}: {x} vs {y}");
    }
}

/// What the gate's scores have seen, in order.
static SCORED: Mutex<Vec<BlockChoice>> = Mutex::new(Vec::new());
/// The candidates of the question being measured: the gate's order, the last fastest.
static LISTED: Mutex<Vec<BlockChoice>> = Mutex::new(Vec::new());

fn pulled() -> Reuse {
    Reuse::Pull {
        holders: AUTO_HOLDERS,
        b_holders: None,
    }
}

/// The gate's order, least first: the listed candidates last first, anything else after them.
fn score(block: [usize; 3], reuse: Reuse, _us: f64) -> f64 {
    let c = BlockChoice { block, reuse };
    SCORED.lock().unwrap().push(c);
    let listed = LISTED.lock().unwrap();
    listed
        .iter()
        .position(|&l| l == c)
        .map_or(1e9, |p| (listed.len() - p) as f64)
}

fn scored() -> Vec<BlockChoice> {
    SCORED.lock().unwrap().clone()
}

fn product<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
) -> Vec<f32> {
    let out = s
        .matmul_dram(
            a,
            false,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET * 40,
        )
        .unwrap();
    let v = s.download(&out).unwrap();
    s.free(out).unwrap();
    v
}

/// `a @ b` captured alone and replayed once: what the replay leaves in the output.
fn traced_product<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
) -> Vec<f32> {
    s.begin_trace().unwrap();
    let out = s
        .matmul_dram(
            a,
            false,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET * 40,
        )
        .unwrap();
    let id = s.end_trace().unwrap();
    s.replay(id).unwrap();
    let v = s.download(&out).unwrap();
    s.free(out).unwrap();
    s.release_trace(id).unwrap();
    v
}

/// The modes a measured session runs in, its scores the gate's, for `a @ b` (or a product of
/// their shapes and formats): its question and candidates.
fn measuring<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
) -> Option<(tensor::BlockQuestion, Vec<BlockChoice>)> {
    tensor::forget_measured();
    s.set_block_stream(BlockStream::Measured);
    s.set_operand_reuse(Reuse::Auto).unwrap();
    s.probe_block_scores(Some(score));
    let q = s
        .block_question_of((a, false), (b, false), Fidelity::HiFi4)
        .unwrap()?;
    let listed = tensor::block_stream_candidates(&q, Some(pulled()));
    *LISTED.lock().unwrap() = listed.clone();
    Some((q, listed))
}

/// The model's own choice, nothing shared.
fn modelled<T: tt_device::Transport>(s: &mut Session<T>) {
    s.set_block_stream(BlockStream::Auto);
    s.set_operand_reuse(Reuse::Off).unwrap();
    s.probe_block_scores(None);
}

/// Products `Auto` blocks with several candidates on eight units, ragged ones among them.
const SHAPES: [[usize; 3]; 3] = [[256, 512, 512], [300, 320, 416], [128, 768, 512]];

/// [`chain`]'s product: blocked with four candidates on eight units, and the residency planner
/// keeps some of the chain in L1 (it keeps nothing of `SHAPES[0]`'s).
const CHAIN: [usize; 3] = [128, 768, 512];

#[test]
fn a_measured_product_runs_its_chosen_candidate_bit_for_bit_and_times_once() {
    with_session(|s| {
        let mut tested = 0;
        for [m, k, n] in SHAPES {
            let what = format!("[{m},{k}]@[{k},{n}]");
            let a = s.upload(&values(m * k, m), m, k).unwrap();
            let b = s.upload(&values(k * n, n), k, n).unwrap();
            modelled(s);
            let want = product(s, &a, &b);
            let (q, listed) = measuring(s, &a, &b).expect("a blocked product");
            let before = scored().len();
            // Eager: `Auto`'s block, nothing timed.
            same(&product(s, &a, &b), &want, &format!("{what}, eager"));
            assert_eq!(scored().len(), before, "{what}: an eager product was timed");
            assert!(tensor::measured().is_empty(), "{what}: measured eagerly");
            same(
                &traced_product(s, &a, &b),
                &want,
                &format!("{what}, measured"),
            );
            let measured = tensor::measured();
            assert_eq!(measured.len(), 1, "{what}: one question");
            assert_eq!(measured[0].0, q, "{what}: the question it says it asks");
            let choice = measured[0].1;
            let timed = scored()[before..].to_vec();
            eprintln!(
                "{what}: {q:?}, candidates {listed:?}, timed {}, chose {choice:?}",
                timed.len()
            );
            if let [auto, .., last] = listed[..] {
                assert_eq!(choice, Some(last), "{what}: the scores' choice");
                assert_ne!(choice, Some(auto), "{what}: Auto's unpulled block");
                let mut want_timed = listed.clone();
                for n in tensor::block_stream_neighbours(&q, last, Some(pulled())) {
                    if !want_timed.contains(&n) {
                        want_timed.push(n);
                    }
                }
                want_timed.truncate(MEASURE_MOST);
                assert_eq!(
                    timed, want_timed,
                    "{what}: the list, then one climb, each once"
                );
                tested += 1;
            }
            let before = scored().len();
            same(
                &traced_product(s, &a, &b),
                &want,
                &format!("{what}, measured again"),
            );
            assert_eq!(scored().len(), before, "{what}: timed again");
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
        assert!(tested > 0, "no shape had two candidates");
    });
}

/// `|A| @ B`, then `|.|` of the product: a chain the residency planner keeps something of.
fn chain<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
) -> DramTensor {
    let abs = tensor::Eltwise {
        kind: tt_kernels::sfpu::ops::kind_sfpu::ABS,
        scalar: 0.0,
        scalar2: 0.0,
    };
    let e = s.eltwise(abs, a, None).unwrap();
    let y = s
        .matmul_dram(
            &e,
            false,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET * 40,
        )
        .unwrap();
    s.free(e).unwrap();
    let z = s.eltwise(abs, &y, None).unwrap();
    s.free(y).unwrap();
    z
}

/// What a trace recorded: each op's kind, entries on every unit and barrier, and each unit's
/// stream sizes.
type Recorded = (
    Vec<(
        &'static str,
        &'static str,
        Vec<std::ops::Range<usize>>,
        bool,
    )>,
    Vec<Option<(u32, tt_kernels::trace::Composition)>>,
);

fn recorded<T: tt_device::Transport>(s: &Session<T>, id: tt_kernels::trace::TraceId) -> Recorded {
    let ops = s
        .trace_ops(id)
        .unwrap()
        .iter()
        .map(|o| (o.what, o.label, o.entries.clone(), o.barrier))
        .collect();
    let units = s
        .trace_footprint(id)
        .unwrap()
        .into_iter()
        .map(|u| u.map(|u| (u.top_level_entries, u.stream)))
        .collect();
    (ops, units)
}

/// A plan pass of [`chain`]: what it recorded, and whether its own run, then each of two replays,
/// equals `want`.
fn plan_pass<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
    want: &[f32],
) -> (Recorded, Vec<bool>) {
    s.begin_trace_plan_pass().unwrap();
    let z = chain(s, a, b);
    let id = s.end_trace().unwrap();
    let rec = recorded(s, id);
    let equal = |v: Vec<f32>| v.iter().zip(want).all(|(x, y)| x.to_bits() == y.to_bits());
    let mut same = vec![s.download(&z).is_ok_and(equal)];
    for _ in 0..2 {
        same.push(s.replay(id).is_ok() && s.download(&z).is_ok_and(equal));
    }
    s.free(z).unwrap();
    s.release_trace(id).unwrap();
    (rec, same)
}

#[test]
fn a_capture_paused_to_measure_is_the_capture_unpaused() {
    with_session(|s| {
        let [m, k, n] = CHAIN;
        let a = s.upload(&values(m * k, 1), m, k).unwrap();
        let b = s.upload(&values(k * n, 2), k, n).unwrap();
        modelled(s);
        let z = chain(s, &a, &b);
        let want = s.download(&z).unwrap();
        s.free(z).unwrap();
        measuring(s, &a, &b).expect("a blocked product");
        let before = scored().len();
        let (paused, replays) = plan_pass(s, &a, &b, &want);
        assert!(scored().len() > before, "the plan pass measured nothing");
        assert!(
            replays.iter().all(|&x| x),
            "the paused pass's run, then its replays: {replays:?}"
        );
        let (_, Some(choice), _) = tensor::measured()[0] else {
            panic!("the product is not blocked");
        };
        eprintln!("chose {choice:?}");
        // The planned capture applying the paused pass's plan: refused if its ops or tensors
        // were not the pass's own.
        assert!(s.begin_trace_planned().unwrap(), "the chain is planned");
        let z = chain(s, &a, &b);
        let id = s.end_trace().unwrap();
        for i in 0..2 {
            s.replay(id).unwrap();
            same(
                &s.download(&z).unwrap(),
                &want,
                &format!("planned replay {i}"),
            );
        }
        s.free(z).unwrap();
        s.release_trace(id).unwrap();
        // The same pass with the choice kept: nothing paused, the same record.
        let before = scored().len();
        let (unpaused, replays) = plan_pass(s, &a, &b, &want);
        assert_eq!(scored().len(), before, "measured again");
        assert!(
            replays.iter().all(|&x| x),
            "the unpaused pass's run, then its replays: {replays:?}"
        );
        same_but_setup(&paused, &unpaused);
    });
}

/// The paused pass's record is the unpaused one's, but for the setup runs a pause leaves: what
/// it ran on a unit leaves the tile's descriptors and semaphores unknown (`Session::paused`), so
/// a unit whose state the unpaused pass carries into its next op from an earlier one (a unit the
/// paused product left idle) sets them again there, one `op::KERNEL` setup run a pause. Every
/// op, barrier and other record is the same, on every unit.
fn same_but_setup(paused: &Recorded, unpaused: &Recorded) {
    const PAUSES: u32 = 1;
    let kernel = tt_isa::dm::op::KERNEL;
    let (ops, units) = paused;
    let (ops_u, units_u) = unpaused;
    assert_eq!(ops.len(), ops_u.len(), "the paused pass recorded other ops");
    for (o, u) in ops.iter().zip(ops_u) {
        assert_eq!((o.0, o.1, o.3), (u.0, u.1, u.3), "an op recorded otherwise");
        assert_eq!(o.2.len(), u.2.len(), "{}: other units", o.0);
    }
    assert_eq!(units.len(), units_u.len());
    for (i, (p, u)) in units.iter().zip(units_u).enumerate() {
        let (Some((top, p)), Some((top_u, u))) = (p, u) else {
            assert_eq!(p.is_some(), u.is_some(), "unit {i} used in one pass only");
            continue;
        };
        let count = |c: &tt_kernels::trace::Composition| c.heads.get(&kernel).copied().unwrap_or(0);
        let extra = count(p)
            .checked_sub(count(u))
            .unwrap_or_else(|| panic!("unit {i}: the paused pass has fewer setup runs"));
        assert!(
            extra <= PAUSES,
            "unit {i}: {extra} setup runs for {PAUSES} pause"
        );
        assert_eq!(
            (top - top_u, p.entries - u.entries),
            (extra, extra),
            "unit {i}: other entries"
        );
        assert_eq!(p.padding, u.padding, "unit {i}: other padding");
        let others = |c: &tt_kernels::trace::Composition| {
            let mut h = c.heads.clone();
            h.remove(&kernel);
            h
        };
        assert_eq!(others(p), others(u), "unit {i}: other records");
    }
}

/// The negative control: a pause that leaves the capture's barrier targets and generations where
/// they were before it makes the paused pass's replay fail or differ. On ttsim only: the replay
/// waits for what it never reaches until its deadline, which proves nothing more on a card.
#[cfg(not(feature = "silicon"))]
#[test]
fn a_pause_that_does_not_rebase_the_capture_is_caught() {
    with_session(|s| {
        let [m, k, n] = CHAIN;
        let a = s.upload(&values(m * k, 1), m, k).unwrap();
        let b = s.upload(&values(k * n, 2), k, n).unwrap();
        modelled(s);
        let z = chain(s, &a, &b);
        let want = s.download(&z).unwrap();
        s.free(z).unwrap();
        measuring(s, &a, &b).expect("a blocked product");
        tt_kernels::session::probe_pause_unrebased(true);
        s.begin_trace_plan_pass().unwrap();
        let z = chain(s, &a, &b);
        let id = s.end_trace().unwrap();
        tt_kernels::session::probe_pause_unrebased(false);
        let replayed = s.replay(id).and_then(|()| s.download(&z));
        let caught = match replayed {
            Err(_) => true,
            Ok(v) => v.iter().zip(&want).any(|(x, y)| x.to_bits() != y.to_bits()),
        };
        assert!(caught, "an unrebased pause replayed as the chain");
    });
}

/// A measurement times its copies stored whole, whatever the capture it pauses stores as layer
/// sub-traces (`Session::set_sub_traces`): its trace of two copies of a product repeats, and as
/// a body called twice it would be timed as no model's product is. Under sub-traces whose tables
/// are cut short (`session::probe_sub_trace_tables`, which makes any body's replay fail), the
/// measurement still times its candidates (it would fail on its first body's replay), the
/// capture ends, and its replay is the eager product bit for bit. (The smoke found it on card 0:
/// the failed replay recovered the session there, and the capture it paused ended stale.) Silicon
/// only: on ttsim the measurement passes with or without the fix (checked), so it would prove
/// nothing there; on card 0 it fails without it.
#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "not reached on ttsim: cargo xtask silicon --release --filter step245_measured_block"
)]
fn a_measurement_under_sub_traces_times_its_copies_whole() {
    use tt_kernels::session::{probe_sub_trace_tables, ProbeTables};
    with_session(|s| {
        let [m, k, n] = CHAIN;
        let a = s.upload(&values(m * k, 3), m, k).unwrap();
        let b = s.upload(&values(k * n, 4), k, n).unwrap();
        modelled(s);
        let want = product(s, &a, &b);
        measuring(s, &a, &b).expect("a blocked product");
        probe_sub_trace_tables(ProbeTables::Short);
        s.set_sub_traces(true);
        let got = traced_product(s, &a, &b);
        s.set_sub_traces(false);
        probe_sub_trace_tables(ProbeTables::Off);
        same(&got, &want, "measured under sub-traces");
        let measured = tensor::measured();
        assert_eq!(measured.len(), 1, "measured once");
        // A measurement that fails keeps `Auto`'s block and no times: its traces' replays did.
        assert!(
            measured[0].2.is_some(),
            "the measurement failed: {:?}",
            measured[0]
        );
        s.free(a).unwrap();
        s.free(b).unwrap();
    });
}
