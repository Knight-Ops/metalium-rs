//! Layer sub-traces (`docs/plans/layer-sub-traces.md`, `Session::set_sub_traces`): a trace whose
//! ops repeat is stored as one layer body a unit and a call a layer (`dm::op::CALL_BODY`), each
//! call's tensors in its argument table (`dm::ARG_REF`, `dm::ARG_WORD`), and its replays run what
//! the plain trace runs.
//!
//! - **A chain of distinct layers replays as the ops run fresh**: six layers of `relu(h W + b)`,
//!   each its own weights and activations, captured with sub-traces on: the footprint has a body
//!   and calls on every unit, fewer GDDR bytes than the plain trace, and every replay over a new
//!   input equals the ops run fresh, bit for bit.
//! - **A call given another layer's table diverges** (the negative control,
//!   `session::probe_rotate_sub_trace_tables`): the same capture with each call given the next
//!   call's table replays to other values.
//! - **A table shorter than its body takes is refused** (`ProbeTables::Short`): the mover reports
//!   `RANGE` for the argument past it, and reads nothing there.

use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{probe_sub_trace_tables, ProbeTables, Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_tiles(count: usize, test: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    fork_scope(|| {
        let mut simulator = tt_ttsim::Simulator::open().unwrap();
        let device = tt_device::Device::open(simulator.transport()).unwrap();
        let mut session = Session::open(
            device,
            tt_firmware_images::ROLES,
            TileChoice::Count(count),
            |_, _| Ok(None),
        )
        .unwrap();
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        test(&mut session);
    })
    .unwrap();
}

#[cfg(feature = "silicon")]
fn with_tiles(count: usize, test: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    fork_scope(|| {
        let mut session = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(count),
        )
        .unwrap();
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        test(&mut session);
    })
    .unwrap();
}

/// Rows and width of the chain's activations.
const ROWS: usize = 64;
const WIDTH: usize = 128;
const LAYERS: usize = 6;
const TILES: usize = 2;

/// One layer: `relu(h W + b)`.
fn layer<T: tt_device::Transport>(
    s: &mut Session<T>,
    h: &DramTensor,
    w: &DramTensor,
    b: &DramTensor,
) -> DramTensor {
    let p = s
        .matmul_dram(
            h,
            false,
            w,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET,
        )
        .unwrap();
    let ew = |s: &mut Session<T>, kind, a: &DramTensor, b: Option<&DramTensor>| {
        s.eltwise(
            Eltwise {
                scalar2: 0.0,
                kind,
                scalar: 0.0,
            },
            a,
            b,
        )
        .unwrap()
    };
    let pb = ew(s, kind::ADD_ROW, &p, Some(b));
    let y = ew(s, kind::RELU, &pb, None);
    s.free(p).unwrap();
    s.free(pb).unwrap();
    y
}

/// The chain over `x`, its intermediate activations freed.
fn chain<T: tt_device::Transport>(
    s: &mut Session<T>,
    x: &DramTensor,
    weights: &[(DramTensor, DramTensor)],
) -> DramTensor {
    let mut h: Option<DramTensor> = None;
    for (w, b) in weights {
        let y = layer(s, h.as_ref().unwrap_or(x), w, b);
        if let Some(old) = h.replace(y) {
            s.free(old).unwrap();
        }
    }
    h.unwrap()
}

/// Small exact-ish values, distinct per layer and per element.
fn values(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed * 13) % 17) as f32 - 8.0) * scale)
        .collect()
}

/// Capture the chain with sub-traces on, then replay it over new inputs: each replay's output
/// against the ops run fresh on that input. Returns how many replays differed from them.
fn replays_against_fresh<T: tt_device::Transport>(
    s: &mut Session<T>,
    check_footprint: bool,
) -> usize {
    let weights: Vec<(DramTensor, DramTensor)> = (0..LAYERS)
        .map(|l| {
            (
                s.upload(&values(WIDTH * WIDTH, l, 1.0 / 64.0), WIDTH, WIDTH)
                    .unwrap(),
                s.upload(&values(WIDTH, l + 50, 1.0 / 8.0), 1, WIDTH)
                    .unwrap(),
            )
        })
        .collect();
    let x = s
        .upload(&values(ROWS * WIDTH, 99, 0.25), ROWS, WIDTH)
        .unwrap();
    // The plain trace's footprint, for its GDDR bytes.
    s.set_sub_traces(false);
    s.begin_trace().unwrap();
    let y_plain = chain(s, &x, &weights);
    let plain = s.end_trace().unwrap();
    let plain_bytes: u64 = s
        .trace_footprint(plain)
        .unwrap()
        .iter()
        .flatten()
        .map(|f| f.gddr_bytes)
        .sum();
    s.release_trace(plain).unwrap();
    s.free(y_plain).unwrap();

    s.set_sub_traces(true);
    s.begin_trace().unwrap();
    let y = chain(s, &x, &weights);
    let id = s.end_trace().unwrap();
    s.set_sub_traces(false);
    let footprint = s.trace_footprint(id).unwrap();
    if check_footprint {
        for (u, f) in footprint.iter().enumerate() {
            let f = f.as_ref().expect("every unit runs the chain");
            assert!(
                f.called_layers >= 2 && f.body_entries > 0,
                "unit {u} makes no body: {f:?}"
            );
            println!(
                "MEASURE sub-trace unit {u}: {} layers called, body {} entries, tables {} entries, top level {}, {} B of GDDR",
                f.called_layers, f.body_entries, f.table_entries, f.top_level_entries, f.gddr_bytes
            );
        }
        let bytes: u64 = footprint.iter().flatten().map(|f| f.gddr_bytes).sum();
        println!("MEASURE sub-trace GDDR {bytes} B against {plain_bytes} B plain");
        assert!(
            bytes < plain_bytes,
            "{bytes} B against {plain_bytes} B plain"
        );
    }
    let mut differing = 0;
    for round in 0..3 {
        let input = values(ROWS * WIDTH, 200 + round, 0.25);
        s.write(&x, &input).unwrap();
        s.replay(id).unwrap();
        let replayed = s.download_bits(&y).unwrap();
        let fresh = chain(s, &x, &weights);
        let expected = s.download_bits(&fresh).unwrap();
        s.free(fresh).unwrap();
        if replayed != expected {
            differing += 1;
        }
    }
    s.release_trace(id).unwrap();
    s.free(y).unwrap();
    s.free(x).unwrap();
    for (w, b) in weights {
        s.free(w).unwrap();
        s.free(b).unwrap();
    }
    differing
}

#[test]
fn a_chain_of_distinct_layers_replays_as_the_ops_run_fresh() {
    with_tiles(TILES, |s| {
        assert_eq!(replays_against_fresh(s, true), 0);
    });
}

#[test]
fn a_call_given_another_layer_s_table_diverges() {
    with_tiles(TILES, |s| {
        probe_sub_trace_tables(ProbeTables::Rotate);
        let differing = replays_against_fresh(s, false);
        probe_sub_trace_tables(ProbeTables::Off);
        assert_eq!(
            differing, 3,
            "a call given the next layer's table must not replay its own layer"
        );
    });
}

#[test]
fn a_table_shorter_than_its_body_takes_is_refused() {
    with_tiles(TILES, |s| {
        let weights: Vec<_> = (0..LAYERS)
            .map(|l| {
                (
                    s.upload(&values(WIDTH * WIDTH, l, 1.0 / 64.0), WIDTH, WIDTH)
                        .unwrap(),
                    s.upload(&values(WIDTH, l + 50, 1.0 / 8.0), 1, WIDTH)
                        .unwrap(),
                )
            })
            .collect();
        let x = s
            .upload(&values(ROWS * WIDTH, 99, 0.25), ROWS, WIDTH)
            .unwrap();
        probe_sub_trace_tables(ProbeTables::Short);
        s.set_sub_traces(true);
        s.begin_trace().unwrap();
        let y = chain(s, &x, &weights);
        let id = s.end_trace().unwrap();
        s.set_sub_traces(false);
        probe_sub_trace_tables(ProbeTables::Off);
        let called: u32 = s
            .trace_footprint(id)
            .unwrap()
            .iter()
            .flatten()
            .map(|f| f.called_layers)
            .sum();
        assert!(called > 0, "the capture made no body to refuse");
        let refused = s.replay(id).and_then(|()| s.download(&y).map(|_| ()));
        let message = format!("{refused:?}");
        println!("refused: {message}");
        assert!(refused.is_err(), "a short table replayed");
        let code = format!("(code {})", tt_isa::dm::error::RANGE);
        assert!(message.contains(&code), "{message}");
    });
}
