//! The profile mode (`burn_tt::profile`, `tt_kernels::profile_mode`): one switch that attributes
//! the card's time to Burn ops, eager and traced.
//!
//! The claims, on silicon (the simulator does not model the timestamps, divergence row 54):
//!
//! * **Eager ops are attributed to their Burn ops.** An add, a matmul and an attention each come
//!   back under their labels (the attention by its parts), with positive spans.
//! * **A trace too long for one stamp buffer is covered window by window.** A chain of 600 captured
//!   ops (whose launch stamps alone outrun a buffer), replayed, gets every op attributed after a few
//!   replays, none twice in one replay, and no replay overflows. B's fetches of the trace and its
//!   waits for its writer core are attributed too, within each op's span; a chain reads nothing
//!   ahead, so no time is a prefetch's (the trace's chunk-padding `WAIT`s are not).
//! * **Every unit's own stamps are attributed op by op.** Each unit's barriers line up with its
//!   stream in every replay, every covered op has its units' times, and unit 0's work and wait
//!   at each op are, sample by sample, that op's span.
//! * **The windows stitch to the whole.** On a trace short enough to stamp whole, the per-op spans
//!   stitched from windows sum to what whole-trace profiles measure, within the spread of those.
//! * **A measured block choice times what an unprofiled build runs.** A captured product's
//!   candidates (`TT_MATMUL_BLOCK=measured`) are timed with the profile mode suspended
//!   (`Session::unprofiled`): `Auto`'s block times within 1.5x of the same measurement with the
//!   mode off. Its negative control (`session::probe_profiled_measurement`) times them stamped, as
//!   before 2026-10-31, and must miss by more than 3x (it chose 8 units for a 64-row product
//!   that unprofiled ran on 64, `1793473164`).
//! * **Negative control:** with the window off, the long trace overflows the stamp buffer. The run
//!   goes on (profiling never fails one), and the trace is reported as overflowed and uncovered. That
//!   is what the window exists to avoid.

// Firmware that can stamp: `--features tt-tests/profile` (`cargo xtask silicon --profile`).
#![cfg(all(feature = "silicon", feature = "profile"))]

use burn::tensor::{Tensor, TensorData, TensorPrimitive};
use burn_tt::profile::{profile_mode, profile_report, Profile, ProfileOptions};
use burn_tt::{TracedInference, TtBackend, TtDevice, TtTensor};
use tt_tests::burn_device::{with_device, Config};

fn every_tile() -> Config {
    Config {
        tiles: Some(burn_tt::TileChoice::All),
        ..Config::default()
    }
}

fn float(t: Tensor<TtBackend, 2>) -> TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(t) => t,
        _ => unreachable!(),
    }
}

fn input(d: &TtDevice) -> Tensor<TtBackend, 2> {
    let v: Vec<f32> = (0..128 * 256).map(|i| (i % 97) as f32 / 97.0).collect();
    Tensor::from_data(TensorData::new(v, [128, 256]), d)
}

/// `n` pairs of a multiply and an add by scalars: `2 n` ops, each on several units, each ending in a
/// barrier.
fn chain(x: Tensor<TtBackend, 2>, n: usize) -> Tensor<TtBackend, 2> {
    let mut y = x;
    for _ in 0..n {
        y = y.mul_scalar(1.0001).add_scalar(0.5);
    }
    y
}

/// Keep a captured chain's barriers. Its ops run on the same units and read only what their own
/// unit wrote, so a capture would drop every barrier between them (`Session::set_barrier_prune`,
/// on by default since 2026-10-26) and store the chain as one op; these gates are about a trace of
/// many ops. Set before the device opens, which reads it; `cargo xtask silicon` runs each gate in
/// a process of its own.
fn keep_barriers() {
    std::env::set_var("TT_BARRIER_PRUNE", "0");
}

/// A captured chain of `2 n` ops over `x`.
fn traced_chain(d: &TtDevice, n: usize) -> TracedInference {
    let x = input(d);
    let xp = float(x.clone());
    xp.ensure_resident();
    let (trace, _) = TracedInference::capture(&[&xp], || vec![float(chain(x.clone(), n))]).unwrap();
    trace.set_name(format!("chain of {}", 2 * n));
    trace
}

fn inputs() -> Vec<burn_tt::InputPayload> {
    vec![burn_tt::InputPayload::F32(
        (0..128 * 256).map(|i| (i % 89) as f32 / 89.0).collect(),
    )]
}

fn report(d: &TtDevice) -> Profile {
    profile_report(*d).unwrap().expect("the mode is on")
}

#[test]
fn eager_ops_are_attributed_to_their_burn_ops() {
    with_device(every_tile(), |d| {
        profile_mode(d, Some(ProfileOptions::default())).unwrap();
        let x = input(&d);
        let y = x.clone() + x.clone();
        let z = y.matmul(x.clone().transpose());
        let q = z.reshape([1, 2, 64, 128]);
        let a = burn::tensor::module::attention(
            q.clone(),
            q.clone(),
            q,
            None,
            None,
            Default::default(),
        );
        let _ = a.into_data();
        let p = profile_mode(d, None)
            .unwrap()
            .expect("what the mode recorded");
        let labels: Vec<&str> = p.device.eager.iter().map(|s| s.label).collect();
        eprintln!("{}", p.summary());
        for want in ["float_add", "float_matmul", "attention/qk", "attention/pv"] {
            assert!(labels.contains(&want), "no `{want}` among {labels:?}");
        }
        assert!(p.device.eager.iter().all(|s| s.time.span > 0));
        assert!(p.device.ticks_per_us > 100.0, "{}", p.device.ticks_per_us);
        // The files the environment switch writes.
        let dir = std::env::temp_dir().join(format!("tt-profile-{}", std::process::id()));
        p.write(&dir).unwrap();
        for f in [
            "summary.md",
            "ops.tsv",
            "skew.tsv",
            "tails.tsv",
            "trace.json",
        ] {
            assert!(dir.join(f).metadata().unwrap().len() > 0, "{f}");
        }
        let _ = std::fs::remove_dir_all(dir);
    });
}

#[test]
fn a_long_trace_is_covered_window_by_window() {
    keep_barriers();
    with_device(every_tile(), |d| {
        // 600 ops: their launches' and waits' stamps alone (two an op) outrun one buffer, so every
        // stamp the mover makes must obey the window, not only the entries'.
        // On before the capture: a trace stored in the profile mode keeps each unit's barriers,
        // which its units' own stamps are told apart by.
        profile_mode(d, Some(ProfileOptions::default())).unwrap();
        let trace = traced_chain(&d, 300);
        let ops = trace.ops().unwrap().len();
        assert!(ops >= 600, "{ops}");
        let mut replays = 0;
        loop {
            trace.run(inputs()).unwrap();
            replays += 1;
            let p = report(&d);
            let (_, t) = p.device.traces.iter().next().expect("the trace is known");
            assert_eq!(t.ops.len(), ops);
            assert_eq!(t.overflowed, 0, "a window outran the buffer");
            if t.covered() == ops {
                // No op is stamped twice before every op has been once.
                assert!(t.samples.iter().all(|s| s.len() <= replays));
                eprintln!("{}", p.summary());
                assert!(p.summary().contains("chain of 600"), "named");
                // B's fetches of the trace and its waits for its writer are stamped too, inside
                // each op's span.
                let medians: Vec<_> = t.medians().into_iter().flatten().collect();
                let sum = |f: fn(&burn_tt::profile::OpTime) -> u64| -> u64 {
                    medians.iter().map(f).sum()
                };
                eprintln!(
                    "fetch {} writer {} span {} cycles",
                    sum(|m| m.fetch),
                    sum(|m| m.writer),
                    sum(|m| m.span)
                );
                assert!(sum(|m| m.fetch) > 0, "the replay's fetches are stamped");
                assert!(sum(|m| m.writer) > 0, "B's waits for NC are stamped");
                assert_eq!(sum(|m| m.prefetch), 0, "a plain `WAIT` is no prefetch's");
                for m in &medians {
                    assert!(
                        m.barrier + m.kernels + m.control + m.fetch + m.writer + m.prefetch
                            <= m.span
                    );
                }
                // Each unit's own stamps, op by op.
                assert_eq!(t.unit_mismatches, 0, "a unit's stamps did not line up");
                for (k, (spans, units)) in t.samples.iter().zip(&t.units).enumerate() {
                    let unit0: Vec<u64> = units
                        .iter()
                        .filter(|u| u.unit == 0)
                        .map(|u| u.work + u.wait)
                        .collect();
                    assert!(units.len() > unit0.len(), "op {k}: only unit 0's times");
                    assert_eq!(
                        unit0,
                        spans.iter().map(|s| s.span).collect::<Vec<_>>(),
                        "op {k}: unit 0's work and wait against its span"
                    );
                }
                let skew = p.device.skew_tsv(&|id| id.to_string());
                assert!(skew.lines().count() > ops, "a line a unit an op");
                break;
            }
            assert!(
                replays < 64,
                "{} of {ops} covered after {replays} replays",
                t.covered()
            );
        }
        assert!(replays > 1, "a 600-op trace needs more than one window");
        profile_mode(d, None).unwrap();
    });
}

#[test]
fn windows_stitch_to_the_whole_trace() {
    keep_barriers();
    with_device(every_tile(), |d| {
        // 24 ops: short enough to stamp whole.
        let trace = traced_chain(&d, 12);
        let whole = |options: ProfileOptions, replays: usize| -> Vec<u64> {
            profile_mode(d, Some(options)).unwrap();
            let mut sums = Vec::new();
            for _ in 0..replays {
                trace.run(inputs()).unwrap();
            }
            let p = profile_mode(d, None).unwrap().unwrap();
            let (_, t) = p.device.traces.iter().next().unwrap();
            for r in 0..replays {
                sums.push(
                    t.samples
                        .iter()
                        .map(|s| s.get(r).map_or(0, |o| o.span))
                        .sum(),
                );
            }
            sums
        };
        // Whole replays: every op in each (no window needed).
        let mut all = whole(
            ProfileOptions {
                window: false,
                ..ProfileOptions::default()
            },
            9,
        );
        all.sort_unstable();
        let (lo, hi) = (all[0], all[all.len() - 1]);
        let spread = hi - lo;
        // Windowed: the medians of each op, stitched.
        profile_mode(d, Some(ProfileOptions::default())).unwrap();
        for _ in 0..9 {
            trace.run(inputs()).unwrap();
        }
        let p = profile_mode(d, None).unwrap().unwrap();
        let (_, t) = p.device.traces.iter().next().unwrap();
        assert_eq!(t.covered(), t.ops.len());
        let stitched: u64 = t.medians().iter().flatten().map(|m| m.span).sum();
        eprintln!("whole replays {all:?} cycles; stitched medians {stitched}");
        assert!(
            stitched + spread >= lo && stitched <= hi + spread,
            "stitched {stitched} against whole {lo}..{hi}"
        );
    });
}

/// Negative control: the window off, a 120-op replay's stamps overflow the buffer. The profile mode
/// never fails a run: the replay runs, and the trace is reported as overflowed and uncovered.
#[test]
fn without_the_window_a_long_trace_overflows() {
    keep_barriers();
    with_device(every_tile(), |d| {
        let trace = traced_chain(&d, 60);
        profile_mode(
            d,
            Some(ProfileOptions {
                window: false,
                ..ProfileOptions::default()
            }),
        )
        .unwrap();
        for _ in 0..3 {
            trace.run(inputs()).expect("profiling never fails a run");
        }
        let p = profile_mode(d, None).unwrap().unwrap();
        let (_, t) = p.device.traces.iter().next().unwrap();
        assert_eq!((t.replays, t.overflowed), (3, 3), "every replay overflowed");
        assert_eq!(t.covered(), 0, "and none was attributed");
        assert!(p.summary().contains("(3 overflowed)"));
    });
}

/// `Auto`'s block's measured time, us, for a 64-row product (two tile rows, as a short prefill
/// chunk's) captured alone, with the profile mode on or off around the capture.
fn measured_auto_us(
    s: &mut tt_kernels::session::Session<tt_kmd::Kmd>,
    a: &tt_kernels::tensor::DramTensor,
    b: &tt_kernels::tensor::DramTensor,
    profiled: bool,
) -> f64 {
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_kernels::tensor::{self, BlockStream, Reuse};
    tensor::forget_measured();
    s.set_block_stream(BlockStream::Measured);
    s.set_operand_reuse(Reuse::Auto).unwrap();
    if profiled {
        s.profile_mode(Some(tt_kernels::profile_mode::ProfileOptions::default()))
            .unwrap();
    }
    s.begin_trace().unwrap();
    let out = s
        .matmul_dram(
            a,
            false,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            tt_tests::harness::BUDGET * 40,
        )
        .unwrap();
    let id = s.end_trace().unwrap();
    if profiled {
        s.profile_mode(None).unwrap();
    }
    s.release_trace(id).unwrap();
    s.free(out).unwrap();
    let measured = tensor::measured();
    assert_eq!(measured.len(), 1, "one question: {measured:?}");
    let (q, choice, times) = measured[0];
    let [auto, chosen] = times.unwrap_or_else(|| panic!("{q:?}: not measured"));
    eprintln!(
        "profiled {profiled}: {q:?} chose {choice:?}, Auto {auto:.1} us, chosen {chosen:.1} us"
    );
    auto
}

#[test]
fn a_measured_block_choice_times_what_an_unprofiled_build_runs() {
    use tt_kernels::session::{probe_profiled_measurement, Session, TileChoice};
    let mut s = Session::open_card(
        tt_tests::backend::device_index(),
        tt_firmware_images::ROLES,
        TileChoice::All,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
        .unwrap();
    let [m, k, n] = [64, 2048, 2048];
    let a: Vec<f32> = (0..m * k).map(|i| (i % 89) as f32 / 89.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 97) as f32 / 97.0).collect();
    let (a, b) = (s.upload(&a, m, k).unwrap(), s.upload(&b, k, n).unwrap());
    let plain = measured_auto_us(&mut s, &a, &b, false);
    let profiled = measured_auto_us(&mut s, &a, &b, true);
    let ratio = profiled / plain;
    assert!(
        (1.0 / 1.5..=1.5).contains(&ratio),
        "Auto's block: {profiled:.1} us profiled against {plain:.1} unprofiled ({ratio:.2}x)"
    );
    probe_profiled_measurement(true);
    let stamped = measured_auto_us(&mut s, &a, &b, true);
    probe_profiled_measurement(false);
    assert!(
        stamped / plain > 3.0,
        "the control: timed stamped, {stamped:.1} us against {plain:.1} unprofiled"
    );
}
