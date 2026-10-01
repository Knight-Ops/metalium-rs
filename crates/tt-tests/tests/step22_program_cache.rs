//! Phase 9.7c gate: kernel programs resident in L1.
//!
//! Each tile keeps the programs it has run in `tt_isa::l1::PROGRAM_CACHE`,
//! mirrored on the host by `tt_kernels::program_cache::ProgramCache`, and a
//! `KERNEL` list entry names which one each role runs. The claims:
//!
//! * a matmul whose blocks have two shapes is still one list per tile, and
//!   the same bits as the host-staged path;
//! * run again, it uploads no program at all;
//! * past the region's capacity the cache evicts, and what it evicted runs
//!   correctly when it is needed again.

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;

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

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// `A @ B` in GDDR against the host-staged path, with the lists, cache
/// counters and result it took.
fn checked_matmul<S>(
    s: &mut S,
    [m, k, n]: [usize; 3],
    fidelity: Fidelity,
    seed: u64,
) -> (u64, tt_kernels::program_cache::CacheStats)
where
    S: SessionLike,
{
    let (av, bv) = (floats(seed, m * k), floats(seed + 1, k * n));
    let want = s.host_matmul(&av, &bv, [m, k, n], fidelity);
    let (lists, stats, got) = s.dram_matmul(&av, &bv, [m, k, n], fidelity);
    assert_bits(
        &got,
        &want,
        &format!("[{m}, {k}] @ [{k}, {n}] {fidelity:?}"),
    );
    (lists, stats)
}

/// What the gates need of a session, on either target.
trait SessionLike {
    fn host_matmul(&mut self, a: &[f32], b: &[f32], mkn: [usize; 3], f: Fidelity) -> Vec<f32>;
    /// Lists submitted, the change in cache counters, and the result.
    fn dram_matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        mkn: [usize; 3],
        f: Fidelity,
    ) -> (u64, tt_kernels::program_cache::CacheStats, Vec<f32>);
}

impl<T: tt_device::Transport> SessionLike for Session<T> {
    fn host_matmul(&mut self, a: &[f32], b: &[f32], mkn: [usize; 3], f: Fidelity) -> Vec<f32> {
        self.matmul(a, b, mkn, ROUTE, f, BUDGET)
            .unwrap_or_else(|e| panic!("{e}"))
    }
    fn dram_matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        [m, k, n]: [usize; 3],
        f: Fidelity,
    ) -> (u64, tt_kernels::program_cache::CacheStats, Vec<f32>) {
        let ta = self.upload(a, m, k).unwrap();
        let tb = self.upload(b, k, n).unwrap();
        let (l0, c0) = (self.lists_per_tile()[0], self.program_cache_stats()[0]);
        let c = self
            .matmul_dram(&ta, false, &tb, false, ROUTE, f, BUDGET)
            .unwrap_or_else(|e| panic!("{e}"));
        let (l1, c1) = (self.lists_per_tile()[0], self.program_cache_stats()[0]);
        let got = self.download(&c).unwrap();
        for t in [ta, tb, c] {
            self.free(t).unwrap();
        }
        let delta = tt_kernels::program_cache::CacheStats {
            hits: c1.hits - c0.hits,
            misses: c1.misses - c0.misses,
            bytes_uploaded: c1.bytes_uploaded - c0.bytes_uploaded,
            evictions: c1.evictions - c0.evictions,
            bypassed: c1.bypassed - c0.bypassed,
        };
        (l1 - l0, delta, got)
    }
}

/// `[512, 512] @ [512, 512]` on one tile plans blocks of two shapes
/// (`[2, 16, 3]` and a ragged `[2, 16, 1]`): with one program slot per role
/// that was two lists at least, and every change of shape a program upload.
#[test]
fn a_matmul_of_two_block_shapes_is_one_list_and_then_no_uploads() {
    with_session(|s| {
        let mkn = [512, 512, 512];
        let (lists, first) = checked_matmul(s, mkn, Fidelity::HiFi4, 1);
        assert_eq!(lists, 1, "one list for every block, of both shapes");
        assert_eq!(first.misses, 2 * 3, "two shapes, three roles each");
        assert!(first.bytes_uploaded > 0);
        let (lists, again) = checked_matmul(s, mkn, Fidelity::HiFi4, 1);
        assert_eq!(lists, 1);
        assert_eq!(
            (again.misses, again.bytes_uploaded),
            (0, 0),
            "the second run uploads no program: {again:?}"
        );
        assert!(again.hits > 0);
    });
}

/// More distinct kernels than the region holds: the cache evicts, and every
/// kernel it evicted is right when it comes back. Each shape is checked
/// against the host-staged path every time it runs.
#[test]
fn evicted_programs_run_correctly_when_they_return() {
    with_session(|s| {
        // The cache cut to 128 KB (`Session::limit_program_cache`), so the
        // kernels below -- about 230 KB of programs together, each role's
        // well under half the cut -- evict whatever shrinks the programs next
        // (X2b's loops took them from ~300 KB to this): the gate no longer
        // depends on how big a matmul's programs happen to be.
        s.limit_program_cache(128 * 1024).unwrap();
        let ops = [
            ([512, 512, 512], Fidelity::HiFi4),
            ([512, 512, 512], Fidelity::Lo),
            ([784, 64, 128], Fidelity::Lo),
            ([64, 784, 128], Fidelity::HiFi4),
            ([256, 1024, 128], Fidelity::HiFi4),
            ([128, 640, 320], Fidelity::HiFi4),
        ];
        let mut evictions = 0;
        for round in 0..2 {
            for (i, (mkn, f)) in ops.iter().enumerate() {
                let (_, c) = checked_matmul(s, *mkn, *f, 10 + i as u64);
                evictions += c.evictions;
                eprintln!("round {round} {mkn:?} {f:?}: {c:?}");
            }
        }
        assert!(
            evictions > 0,
            "the region never filled: the gate tests nothing"
        );
    });
}
