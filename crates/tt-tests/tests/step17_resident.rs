//! Phase 9 gate: resident role firmware computes what reset-per-run does.
//!
//! `session::matmul_on` resets the tile and loads three images before every
//! chunk -- the path every gate so far has trusted. `Session::matmul` resets
//! once, when the session opens, and leaves the images running
//! (`runtime::Resident`, `mailbox::GENERATION`). The claim is that nothing a
//! kernel reads is inherited from the one before, so the two are bit-identical:
//! across shapes, with a different kind of kernel in between, and after a
//! kernel that fails. Random floats in `[-1, 1)`, not small integers, so the
//! comparison sees every rounding.

use tt_device::Transport;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::runtime::{Kernel, Schedule};
use tt_kernels::session::{matmul_on, Session, TileChoice};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
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
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
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
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;

/// The reference: reset before every chunk.
fn reference<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &[f32],
    b: &[f32],
    mkn: [usize; 3],
    fid: Fidelity,
) -> Vec<f32> {
    let tile = s.tile();
    let images = *s.images();
    let c = matmul_on(s.device(), tile, &images, a, b, mkn, ROUTE, fid, BUDGET)
        .unwrap_or_else(|e| panic!("{e}"));
    // `matmul_on` leaves every core held; bring the resident roles back.
    s.prepare().unwrap_or_else(|e| panic!("{e}"));
    c
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
}

/// MNIST's shapes, forward and backward, one after another on the resident
/// roles, against the reset-per-chunk path.
#[test]
fn resident_matmuls_are_bit_identical_to_reset_per_run() {
    with_session(|s| {
        let shapes = [
            ([64, 784, 128], Fidelity::HiFi4),
            ([64, 128, 10], Fidelity::HiFi4),
            ([128, 64, 10], Fidelity::Lo),
            ([64, 10, 128], Fidelity::HiFi2),
        ];
        let mut want = Vec::new();
        for (i, &(mkn, fid)) in shapes.iter().enumerate() {
            let [m, k, n] = mkn;
            let (a, b) = (floats(i as u64, m * k), floats(100 + i as u64, k * n));
            want.push((a.clone(), b.clone(), reference(s, &a, &b, mkn, fid)));
        }
        // Twice round, so every shape also follows a different one.
        for round in 0..2 {
            for (i, &(mkn, fid)) in shapes.iter().enumerate() {
                let (a, b, c) = &want[i];
                let before = s.device().traffic();
                let got = s
                    .matmul(a, b, mkn, ROUTE, fid, BUDGET)
                    .unwrap_or_else(|e| panic!("{e}"));
                let t = s.device().traffic() - before;
                assert_bits(&got, c, &format!("round {round}, {mkn:?} {fid:?}"));
                println!(
                    "MEASURE resident {mkn:?} {fid:?}: {} B written, {} B read",
                    t.bytes_written, t.bytes_read
                );
            }
        }
    });
}

/// A kernel of another kind -- the whole per-thread state reset, in order --
/// between two matmuls changes neither.
#[test]
fn a_different_kernel_in_between_changes_nothing() {
    with_session(|s| {
        let mkn = [64, 96, 64];
        let (a, b) = (floats(7, 64 * 96), floats(8, 96 * 64));
        let want = reference(s, &a, &b, mkn, Fidelity::HiFi4);
        let first = s
            .matmul(&a, &b, mkn, ROUTE, Fidelity::HiFi4, BUDGET)
            .unwrap();
        let reset = tt_kernels::datapath::thread_state_reset();
        let k = Kernel {
            dump_rows: 0,
            ..Kernel::new([&reset, &reset, &reset], Schedule::InOrder)
        };
        // Not on ttsim, which refuses parts of the reset program (rows 36, 49).
        if !s.device().transport().is_simulated() {
            s.run(&k, BUDGET).unwrap_or_else(|e| panic!("{e}"));
        }
        let second = s
            .matmul(&a, &b, mkn, ROUTE, Fidelity::HiFi4, BUDGET)
            .unwrap();
        assert_bits(&first, &want, "before");
        assert_bits(&second, &want, "after");
    });
}

/// A kernel that never finishes -- math waits on a semaphore nothing posts --
/// fails as an error, and (on silicon) the session recovers and computes
/// correctly after.
#[test]
fn a_failed_kernel_leaves_the_session_usable() {
    with_session(|s| {
        let mkn = [32, 64, 32];
        let (a, b) = (floats(9, 32 * 64), floats(10, 64 * 32));
        let want = reference(s, &a, &b, mkn, Fidelity::HiFi4);
        // Any semaphore will do: math waits on one that starts at zero and
        // that nothing posts.
        let never = tt_isa::sync::Semaphore::new(0).unwrap();
        let stuck = tt_isa::sync::take(never, tt_isa::backend::Before::EVERYTHING).to_vec();
        let nothing: Vec<tt_isa::isa::Instruction> = Vec::new();
        let sems = [(never, 0, 1)];
        let k = Kernel {
            dump_rows: 0,
            ..Kernel::new([&nothing, &stuck, &nothing], Schedule::Concurrent(&sems))
        };
        assert!(s.run(&k, 50_000).is_err(), "the stuck kernel must fail");
        // Recovery is the backend pulse (`session::reset_tile`), which ttsim
        // refuses (divergence row 16): there, T1's `SEMWAIT` outlives the
        // reload and nothing can clear it. Silicon only.
        if s.device().transport().is_simulated() {
            return;
        }
        s.prepare().unwrap_or_else(|e| panic!("recovery: {e}"));
        let got = s
            .matmul(&a, &b, mkn, ROUTE, Fidelity::HiFi4, BUDGET)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_bits(&got, &want, "after recovery");
    });
}
