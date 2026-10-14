//! What a trace costs a tile: the two limits that bound it, the program cache (240 KiB of L1)
//! and the stream in GDDR, and what the stream is made of (`Session::trace_footprint`).
//!
//! A baseline, not a speed gate: the oracle for each count is another count (the footprint's
//! entries against `trace_sections` and the stream's own heads), and the one external fact it
//! holds is that capturing the same pass again adds entries and no programs, which is what
//! "traces share programs by hash" means. Run with `--nocapture` for the `MEASURE trace` lines
//! the performance log quotes.

use tt_isa::dm::{op, record};
use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
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

fn name(word: u32) -> String {
    match word {
        op::PAIR_CALL => "PAIR_CALL".into(),
        op::POKE => "POKE".into(),
        op::KERNEL => "KERNEL".into(),
        op::LAUNCH => "LAUNCH".into(),
        op::KERNEL_WAIT => "KERNEL_WAIT".into(),
        op::WAIT => "WAIT".into(),
        op::BARRIER => "BARRIER".into(),
        op::READ => "READ".into(),
        op::WRITE => "WRITE".into(),
        op::CB => "CB".into(),
        op::RELEASED => "RELEASED".into(),
        record::GATHER => "GATHER".into(),
        record::SCATTER => "SCATTER".into(),
        record::READ_RUN => "READ_RUN".into(),
        record::WRITE_RUN => "WRITE_RUN".into(),
        other => format!("{other:#x}"),
    }
}

/// A layer's forward pass at MNIST's size: matmul, add a row, relu, matmul.
fn forward<T: tt_device::Transport>(
    s: &mut Session<T>,
    x: &DramTensor,
    w1: &DramTensor,
    b1: &DramTensor,
    w2: &DramTensor,
) -> DramTensor {
    let mm = |s: &mut Session<T>, a, b| {
        s.matmul_dram(
            a,
            false,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET,
        )
        .unwrap()
    };
    let ew = |s: &mut Session<T>, kind, a, b| {
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
    let h = mm(s, x, w1);
    let hb = ew(s, kind::ADD_ROW, &h, Some(b1));
    let r = ew(s, kind::RELU, &hb, None);
    let y = mm(s, &r, w2);
    for t in [h, hb, r] {
        s.free(t).unwrap();
    }
    y
}

/// MNIST's layer: batch, input, hidden, output.
const MNIST: (usize, usize, usize, usize) = (64, 784, 128, 10);

/// One capture of `passes` forward passes of the shapes in `shapes` (each pass uses the next,
/// cycling): its footprint per unit.
fn footprint_of_passes<T: tt_device::Transport>(
    s: &mut Session<T>,
    passes: usize,
    tiles: usize,
    shapes: &[(usize, usize, usize, usize)],
) -> Vec<tt_kernels::trace::UnitFootprint> {
    let inputs: Vec<_> = shapes
        .iter()
        .map(|&(batch, input, hidden, out)| {
            (
                s.upload(&vec![0.5; batch * input], batch, input).unwrap(),
                s.upload(&vec![0.01; input * hidden], input, hidden)
                    .unwrap(),
                s.upload(&vec![0.1; hidden], 1, hidden).unwrap(),
                s.upload(&vec![0.02; hidden * out], hidden, out).unwrap(),
            )
        })
        .collect();
    s.begin_trace().unwrap();
    let ys: Vec<_> = (0..passes)
        .map(|k| {
            let (x, w1, b1, w2) = &inputs[k % inputs.len()];
            forward(s, x, w1, b1, w2)
        })
        .collect();
    let trace = s.end_trace().unwrap();
    let units: Vec<_> = s
        .trace_footprint(trace)
        .unwrap()
        .into_iter()
        .map(|u| u.expect("every unit of a layer forward is used"))
        .collect();
    let sections = s.trace_sections(trace).unwrap();
    assert_eq!(units.len(), sections.len());
    for (u, (fp, sec)) in units.iter().zip(&sections).enumerate() {
        let stored: u32 = sec.iter().sum();
        // The stream's entries are its own plus every section's, padding included.
        assert_eq!(
            fp.stream.entries,
            fp.top_level_entries + stored,
            "unit {u}: {fp:?} against sections {sec:?}"
        );
        // One PAIR_CALL names one reader and one writer section.
        assert_eq!(
            fp.stream.heads.get(&op::PAIR_CALL).copied().unwrap_or(0) as usize * 2,
            sec.len(),
            "unit {u}"
        );
        assert!(fp.gddr_bytes >= fp.stream.entries as u64 * 32, "unit {u}");
        assert!(fp.top_level_entries <= fp.stream.entries, "unit {u}");
        assert!(fp.program_bytes <= fp.cache_held_bytes, "unit {u}: {fp:?}");
        assert!(fp.cache_held_bytes <= fp.cache_bytes, "unit {u}: {fp:?}");
        assert!(fp.programs > 0 && fp.program_bytes > 0, "unit {u}");
    }
    let what = if shapes.len() > 1 {
        "layer-forward (mixed shapes)"
    } else {
        "layer-forward"
    };
    // Per-unit lines, then the mix of unit 0 (the units differ only in their share of tiles).
    for (u, fp) in units.iter().enumerate().take(2) {
        println!(
            "MEASURE trace {what} x{passes} on {tiles} tiles, unit {u}: {} entries ({} top level, {} padding = {:.1}%), \
             {} B of entries in {} KiB of GDDR, {} programs {} KiB held of {} KiB ({:.1}%)",
            fp.stream.entries,
            fp.top_level_entries,
            fp.stream.padding,
            100.0 * fp.stream.padding as f64 / fp.stream.entries as f64,
            fp.stream.entries as u64 * 32,
            fp.gddr_bytes / 1024,
            fp.programs,
            fp.program_bytes / 1024,
            fp.cache_bytes / 1024,
            100.0 * fp.program_bytes as f64 / fp.cache_bytes as f64,
        );
    }
    let mix: Vec<String> = units[0]
        .stream
        .heads
        .iter()
        .map(|(&w, n)| format!("{} {n}", name(w)))
        .collect();
    println!(
        "MEASURE trace {what} x{passes} on {tiles} tiles, unit 0 records: {}",
        mix.join(", ")
    );
    s.release_trace(trace).unwrap();
    for y in ys {
        s.free(y).unwrap();
    }
    for (x, w1, b1, w2) in inputs {
        for t in [x, w1, b1, w2] {
            s.free(t).unwrap();
        }
    }
    units
}

/// The same pass captured once, twice and four times in one trace: the stream grows with the
/// ops, the pinned programs do not.
#[test]
fn a_repeated_pass_grows_the_stream_and_pins_no_new_programs() {
    for tiles in [1, 2] {
        with_tiles(tiles, |s| {
            let one = footprint_of_passes(s, 1, tiles, &[MNIST]);
            let two = footprint_of_passes(s, 2, tiles, &[MNIST]);
            let four = footprint_of_passes(s, 4, tiles, &[MNIST]);
            for u in 0..tiles {
                assert!(
                    two[u].stream.entries > one[u].stream.entries,
                    "unit {u}: {} then {}",
                    one[u].stream.entries,
                    two[u].stream.entries
                );
                assert!(four[u].stream.entries > two[u].stream.entries, "unit {u}");
                assert!(four[u].gddr_bytes >= one[u].gddr_bytes, "unit {u}");
                // The same programs, named again: held once each.
                assert_eq!(one[u].programs, two[u].programs, "unit {u}");
                assert_eq!(one[u].programs, four[u].programs, "unit {u}");
                assert_eq!(one[u].program_bytes, four[u].program_bytes, "unit {u}");
            }
        });
    }
}

/// The control for the test above: a pass of another shape does pin more programs, so equal
/// counts there mean shared programs, not a count that cannot move.
#[test]
fn a_pass_of_another_shape_pins_more_programs() {
    with_tiles(1, |s| {
        let same = footprint_of_passes(s, 2, 1, &[MNIST]);
        let other = footprint_of_passes(s, 2, 1, &[MNIST, (64, 784, 256, 32)]);
        assert!(
            other[0].programs > same[0].programs,
            "{} programs for two shapes, {} for one",
            other[0].programs,
            same[0].programs
        );
        assert!(other[0].program_bytes > same[0].program_bytes);
    });
}

/// A stream long enough to cross chunks pays padding; the small passes above do not.
#[test]
fn a_long_section_pays_chunk_padding_in_entries_and_bytes() {
    with_tiles(1, |s| {
        let chunk = tt_isa::dm::TRACE_CHUNK_ENTRIES;
        let input = s.upload(&vec![0.5; 2048 * 2048], 2048, 2048).unwrap();
        s.begin_trace().unwrap();
        let out = s
            .eltwise(
                Eltwise {
                    scalar2: 0.0,
                    kind: kind::ADD,
                    scalar: 0.0,
                },
                &input,
                Some(&input),
            )
            .unwrap();
        let trace = s.end_trace().unwrap();
        let fp = s.trace_footprint(trace).unwrap().remove(0).unwrap();
        let sections = s.trace_sections(trace).unwrap().remove(0);
        assert!(sections.iter().any(|&n| n > 2 * chunk), "{sections:?}");
        assert_eq!(
            fp.stream.entries,
            fp.top_level_entries + sections.iter().sum::<u32>()
        );
        println!(
            "MEASURE trace add 2048x2048 on 1 tiles: {} entries ({} padding = {:.1}%), {} KiB of entries in {} KiB of GDDR, \
             {} programs {} KiB held; {}",
            fp.stream.entries,
            fp.stream.padding,
            100.0 * fp.stream.padding as f64 / fp.stream.entries as f64,
            fp.stream.entries as u64 * 32 / 1024,
            fp.gddr_bytes / 1024,
            fp.programs,
            fp.program_bytes / 1024,
            fp.stream
                .heads
                .iter()
                .map(|(&w, n)| format!("{} {n}", name(w)))
                .collect::<Vec<_>>()
                .join(", ")
        );
        // Padding is what chunking added: the records alone, laid out again, take the
        // entries minus it.
        assert!(fp.stream.padding < fp.stream.entries);
        s.release_trace(trace).unwrap();
        s.free(out).unwrap();
        s.free(input).unwrap();
    });
}
