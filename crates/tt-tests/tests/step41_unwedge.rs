//! Phase 10 gate (X5b): a tile wedged by a math instruction starved of `Src`
//! is recovered in software.
//!
//! The wedge is made on purpose, with encodings every matmul uses: a math role
//! whose `MVMUL` no unpacker ever feeds. Such an instruction waits by itself
//! for its banks (`STALLWAIT.md`, C7/C8), and the backend pulse hands every
//! bank to the unpackers (`SoftReset.md`, bits 15-16), so on silicon it
//! outlives the pulse: the thread reset after it is `RunError::Wedged`.
//! `datapath::src_feeder` -- four plain `UNPACR`s from thread 0 -- gives it
//! its banks; then the tile computes correctly again.

use tt_device::Transport;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::runtime::{self, Kernel, RunError, Schedule};
use tt_kernels::session::{reset_tile, restart_roles, unwedge_tile, Session, TileChoice};
use tt_ttsim::fork_scope;

const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;
const BUDGET: u64 = 200_000;

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// A tile away from the gate tile, so a recovery that fails costs only it.
fn wedge_tile() -> TileChoice {
    TileChoice::Exactly(2, 3)
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(dev, tt_firmware_images::ROLES, wedge_tile(), |_, _| {
            Ok(None)
        })
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
            wedge_tile(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// A math role of one `MVMUL` and nothing to feed it.
fn starved() -> Kernel<'static> {
    static MVMUL: std::sync::OnceLock<Vec<tt_isa::isa::Instruction>> = std::sync::OnceLock::new();
    let mvmul = MVMUL.get_or_init(|| {
        vec![tt_isa::isa::generated::encode::Mvmul::ZERO
            .encode()
            .unwrap()]
    });
    Kernel {
        dump_rows: 0,
        ..Kernel::new([&[], mvmul, &[]], Schedule::Concurrent(&[]))
    }
}

fn product<T: tt_device::Transport>(s: &mut Session<T>) -> Vec<f32> {
    let mkn = [32, 64, 32];
    let (a, b) = (floats(9, 32 * 64), floats(10, 64 * 32));
    s.matmul(&a, &b, mkn, ROUTE, Fidelity::HiFi4, BUDGET)
        .unwrap_or_else(|e| panic!("{e}"))
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}, element {i}: {g:e} against {w:e}"
        );
    }
}

/// The recovery on a tile that needs none -- the isolated gate before any
/// wedge is made on purpose: fed, pulsed, reset, and computing bit for bit.
/// Silicon only: ttsim has no pulse to take the fed banks back (row 16).
#[test]
fn the_recovery_on_a_healthy_tile_leaves_it_healthy() {
    with_session(|s| {
        if s.device().transport().is_simulated() {
            return;
        }
        let want = product(s);
        let tile = s.tile();
        let images = tt_firmware_images::ROLES;
        let dev = s.device();
        assert!(
            unwedge_tile(dev, tile, &images).unwrap(),
            "the feeding run did not finish"
        );
        let r = restart_roles(dev, tile, &images).unwrap_or_else(|e| panic!("{e}"));
        r.stop(dev, &images).unwrap();
        s.prepare().unwrap_or_else(|e| panic!("{e}"));
        assert_bits(&product(s), &want, "after an unneeded recovery");
    });
}

/// Step by step: the starved instruction wedges the tile through the reset,
/// the feeder releases it, and the roles then start and compute.
#[test]
fn a_starved_mvmul_wedges_the_tile_and_the_feeder_releases_it() {
    with_session(|s| {
        let want = product(s);
        let tile = s.tile();
        let images = tt_firmware_images::ROLES;
        let dev = s.device();
        // Run behind the session's back, so its own recovery stays out of it.
        let starved = runtime::run(dev, tile, &images, &starved(), 50_000);
        assert!(starved.is_err(), "an MVMUL with no unpack must not finish");

        reset_tile(dev, tile).unwrap();
        match restart_roles(dev, tile, &images) {
            Err(RunError::Wedged { roles, .. }) => {
                assert!(
                    roles.contains(&tt_isa::tensix::Core::T1),
                    "the math role: {roles:?}"
                )
            }
            Err(e) => panic!("not the wedge: {e}"),
            Ok(_) => panic!("vacuous: the reset alone released the starved MVMUL"),
        }
        assert!(
            unwedge_tile(dev, tile, &images).unwrap(),
            "the feeding run did not finish"
        );
        let r = restart_roles(dev, tile, &images).unwrap_or_else(|e| panic!("after feeding: {e}"));
        r.stop(dev, &images).unwrap();
        // The rest of the recovery is the pulse that takes the fed banks back,
        // which ttsim does not have (row 16): there the tile would compute
        // from them. Silicon only from here.
        if dev.transport().is_simulated() {
            return;
        }

        s.prepare().unwrap_or_else(|e| panic!("{e}"));
        assert_bits(&product(s), &want, "after recovery");
    });
}

/// The session's own path: a kernel that wedges the tile fails as an error,
/// and the next op finds the tile recovered.
#[test]
fn a_session_recovers_a_wedged_tile_by_itself() {
    with_session(|s| {
        let want = product(s);
        let failed = s.run(&starved(), 50_000);
        assert!(failed.is_err(), "the starved kernel must fail");
        if s.device().transport().is_simulated() {
            // No pulse, no recovery (above): the tile says it is wedged.
            let e = s.prepare().expect_err("ttsim cannot finish the recovery");
            assert!(matches!(e, RunError::Wedged { .. }), "{e}");
            return;
        }
        assert_bits(&product(s), &want, "after the session's recovery");
    });
}
