//! Phase 10 gate (X3): a device-side profile through the debug timestamper.
//!
//! Every core of a tile stamps its events with the tile's one cycle counter
//! (`DebugTimestamper.md`), so the claims are about order, not about time: on
//! each unit every list, entry and record the mover runs is bracketed, every
//! `KERNEL` entry contains exactly one run of each role, each role's run lies
//! inside it, and a profile outlives the 1024-event buffer because the session
//! drains it after every wave. The clock is measured, never assumed; the gate
//! prints it, and the export writes Chrome trace JSON for a person to look at.
//!
//! ttsim does not model the event stream (divergence row 54): there the claim
//! is that profiling is refused with a reason, and that the refusal leaves the
//! session usable.

use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::Eltwise;
use tt_ttsim::fork_scope;

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[cfg(not(feature = "silicon"))]
#[test]
fn profiling_is_refused_on_the_simulator_and_the_session_carries_on() {
    use tt_tests::backend::GATE_TILE;
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        let err = s.profile_start().expect_err("ttsim has no event stream");
        assert!(err.to_string().contains("row 54"), "{err}");
        assert!(s.profile_stop().is_err(), "nothing was started");
        // Nothing was armed: an op still runs, and no firmware stored to the
        // timestamper (which ttsim would have refused, fatally).
        let v = floats(1, 64 * 64);
        let a = s.upload(&v, 64, 64).unwrap();
        let k = Eltwise {
            scalar2: 0.0,
            kind: tt_isa::dm::kind::MUL_SCALAR,
            scalar: 2.0,
        };
        let b = s.eltwise(k, &a, None).unwrap();
        let got = s.download(&b).unwrap();
        assert!(got.iter().zip(&v).all(|(g, x)| *g == 2.0 * x));
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_tiles(n: usize, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(n),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
#[test]
fn a_profile_brackets_every_entry_and_nests_every_kernel() {
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_kernels::profile::Span;
    with_tiles(2, |s| {
        let (m, k, n) = (100, 70, 90);
        let a = s.upload(&floats(1, m * k), m, k).unwrap();
        let b = s.upload(&floats(2, k * n), k, n).unwrap();
        s.profile_start().unwrap();
        let add = Eltwise {
            scalar2: 0.0,
            kind: tt_isa::dm::kind::ADD,
            scalar: 0.0,
        };
        let sum = s.eltwise(add, &a, Some(&a)).unwrap();
        let prod = s
            .matmul_dram(
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                tt_tests::harness::BUDGET,
            )
            .unwrap();
        let p = s.profile_stop().unwrap();
        println!("timestamper clock: {:.1} ticks/us", p.ticks_per_us);
        // A Blackhole Tensix clock is in the hundreds of MHz to low GHz; this
        // only catches a counter read as the wrong word.
        assert!(
            (100.0..5000.0).contains(&p.ticks_per_us),
            "{}",
            p.ticks_per_us
        );

        let mut kernels = 0;
        for u in &p.units {
            let spans = u.spans().unwrap_or_else(|e| panic!("{e}"));
            assert!(
                spans.iter().any(|s| s.name == "list"),
                "every unit ran a list"
            );
            let in_ =
                |outer: &Span, inner: &Span| outer.begin <= inner.begin && inner.end <= outer.end;
            for k in spans.iter().filter(|s| s.name == "kernel") {
                kernels += 1;
                for t in ["T0", "T1", "T2"] {
                    let runs: Vec<_> = spans.iter().filter(|s| s.track == t && in_(k, s)).collect();
                    assert_eq!(runs.len(), 1, "{t}: one run inside each KERNEL entry");
                }
            }
            for e in spans
                .iter()
                .filter(|s| s.track == "mover" && s.name != "list")
            {
                assert!(
                    spans.iter().any(|l| l.name == "list" && in_(l, e)),
                    "{} outside any list",
                    e.name
                );
            }
            let roles = spans.iter().filter(|s| s.track != "mover").count();
            let ks = spans.iter().filter(|s| s.name == "kernel").count();
            assert_eq!(roles, 3 * ks, "every role run belongs to a KERNEL entry");
        }
        assert!(kernels > 0, "the matmul ran kernels");
        // On the mover (an `ELTWISE` record) or the SFPU (a `READ_RUN`, its
        // kernel, a `WRITE_RUN`), whichever the session found cheaper.
        assert!(
            p.units.iter().all(|u| u
                .spans()
                .unwrap()
                .iter()
                .any(|s| s.name == "eltwise" || s.name == "read run")),
            "the add was dealt to both units"
        );
        let json = p.to_chrome_trace().unwrap();
        let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("step23_profile.json");
        std::fs::write(&path, &json).unwrap();
        println!("chrome trace: {}", path.display());
        for t in [a, b, sum, prod] {
            s.free(t).unwrap();
        }
    });
}

#[cfg(feature = "silicon")]
#[test]
fn a_profile_outlives_the_event_buffer() {
    with_tiles(1, |s| {
        let a = s.upload(&floats(3, 64 * 64), 64, 64).unwrap();
        let relu = Eltwise {
            scalar2: 0.0,
            kind: tt_isa::dm::kind::RELU,
            scalar: 0.0,
        };
        s.profile_start().unwrap();
        // Each op is at least one list: four events for a one-record list.
        // 300 of them is more than the 1024-event buffer holds at once.
        for _ in 0..300 {
            let r = s.eltwise(relu, &a, None).unwrap();
            s.free(r).unwrap();
        }
        let p = s.profile_stop().unwrap();
        let events = p.units[0].events.len();
        assert!(events > 1024, "{events} events: the buffer was drained");
        let lists = p.units[0]
            .spans()
            .unwrap()
            .iter()
            .filter(|s| s.name == "list")
            .count();
        assert_eq!(lists, 300, "one list per op, none lost between drains");
        s.free(a).unwrap();
    });
}
