//! GDDR reused within a capture (`docs/plans/long-context.md` 1.G): with
//! `Session::set_capture_reuse`, a tensor a capture makes and frees gives its slots to the
//! capture's later allocations, instead of the trace holding every intermediate until it is
//! released. On more than one unit every captured op ends in a barrier that counts every earlier
//! op's, so an op after a tensor's last reader runs after that reader on every unit, on replay as
//! in the capture's own run.
//!
//! The claims, on a session of four units:
//! - **Same bits.** A chain whose intermediates are freed as it goes replays, input after input,
//!   as it runs eagerly, bit for bit; and the trace holds less GDDR than the same capture without
//!   reuse.
//! - **Every byte comes back** when the trace is released.
//! - **Only the capture's own tensors are reused.** A tensor made before the capture and freed in
//!   it (the next replay reads it again), and a constant uploaded during the capture (the host
//!   writes it once; a replay does not), keep their slots: replayed twice, the second replay still
//!   equals the eager run.
//!
//! Watched failing: with `DramAlloc::give_to_pool` taking a tensor from before the capture, and an
//! upload, the second replay of their tests diverges.

use tt_kernels::kind;
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::kind_sfpu;
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_ttsim::fork_scope;

/// A `[32, N]` tensor: a few tiles a unit.
const N: usize = 1024;
const UNITS: usize = 4;

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Away from zero: `RECIP` and `LOG` are taken of them.
            let x = ((s >> 40) as f32 / (1u64 << 24) as f32) * 3.0 + 0.25;
            if s & 1 == 0 {
                x
            } else {
                -x
            }
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn op(kind: u32) -> Eltwise {
    Eltwise {
        kind,
        scalar: 0.0,
        scalar2: 0.0,
    }
}

#[cfg(not(feature = "silicon"))]
type Transport<'a> = tt_ttsim::LibTtsim<'a>;
#[cfg(feature = "silicon")]
type Transport<'a> = tt_kmd::Kmd;

#[cfg(not(feature = "silicon"))]
fn open(f: impl FnOnce(&mut Session<Transport<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(UNITS),
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
fn open(f: impl FnOnce(&mut Session<Transport<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(UNITS),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

type S<'a> = Session<Transport<'a>>;

/// `x`'s chain, each intermediate freed once its last reader is queued: `exp`, `recip`, times
/// `x`, `abs`, `log`, divided by `x`, times `x` again. Seven tensors a run, one kept.
fn chain(s: &mut S<'_>, x: &DramTensor) -> DramTensor {
    let e = s.eltwise(op(kind_sfpu::EXP), x, None).unwrap();
    let r = s.eltwise(op(kind_sfpu::RECIP), &e, None).unwrap();
    s.free(e).unwrap();
    let m = s.eltwise(op(kind::MUL), &r, Some(x)).unwrap();
    s.free(r).unwrap();
    let a = s.eltwise(op(kind_sfpu::ABS), &m, None).unwrap();
    s.free(m).unwrap();
    let l = s.eltwise(op(kind_sfpu::LOG), &a, None).unwrap();
    s.free(a).unwrap();
    let d = s.eltwise(op(kind_sfpu::DIV), &l, Some(x)).unwrap();
    s.free(l).unwrap();
    let y = s.eltwise(op(kind::MUL), &d, Some(x)).unwrap();
    s.free(d).unwrap();
    y
}

fn free_total(s: &S<'_>) -> u64 {
    s.dram_free_bytes_by_channel().unwrap().1
}

const SEEDS: [u64; 4] = [2, 3, 4, 5];

/// Capture `chain` over an input `x`, with reuse `on`: the GDDR the trace holds, and each replay
/// (fresh inputs, `SEEDS`) against the eager chain. Released before returning, its GDDR checked.
fn capture_chain(s: &mut S<'_>, on: bool) -> (u64, Vec<bool>) {
    let want: Vec<Vec<u32>> = SEEDS
        .iter()
        .map(|&seed| {
            let x = s.upload(&values(seed, 32 * N), 32, N).unwrap();
            let y = chain(s, &x);
            let out = bits(&s.download(&y).unwrap());
            s.free(y).unwrap();
            s.free(x).unwrap();
            out
        })
        .collect();
    let x = s.upload(&values(1, 32 * N), 32, N).unwrap();
    s.sync().unwrap();
    let before = free_total(s);
    s.set_capture_reuse(on);
    s.begin_trace().unwrap();
    let y = chain(s, &x);
    let id = s.end_trace().unwrap();
    s.set_capture_reuse(false);
    let held = before - free_total(s);
    let mut same = Vec::new();
    for (i, &seed) in SEEDS.iter().enumerate() {
        let fresh = s.upload(&values(seed, 32 * N), 32, N).unwrap();
        s.copy_into(&fresh, &x).unwrap();
        s.free(fresh).unwrap();
        s.replay(id).unwrap();
        same.push(bits(&s.download(&y).unwrap()) == want[i]);
    }
    s.free(y).unwrap();
    s.release_trace(id).unwrap();
    s.sync().unwrap();
    assert_eq!(
        free_total(s),
        before,
        "reuse {on}: every byte back after the release"
    );
    s.free(x).unwrap();
    (held, same)
}

#[test]
fn a_capture_reuses_its_own_slots_and_replays_as_it_ran() {
    open(|s| {
        let (plain, same_plain) = capture_chain(s, false);
        let (reused, same) = capture_chain(s, true);
        eprintln!("the trace holds {plain} bytes of GDDR without reuse, {reused} with it");
        assert!(
            same_plain.iter().all(|&x| x),
            "without reuse: {same_plain:?}"
        );
        assert!(
            same.iter().all(|&x| x),
            "with reuse, replays against eager: {same:?}"
        );
        assert!(
            reused < plain,
            "reuse holds {reused} bytes, not fewer than {plain}"
        );
    });
}

/// A capture that reads `w` (made before it, or uploaded in it by `uploaded`) and frees it, then
/// makes as many tensors of its size as the chain does: replayed twice, as it runs eagerly.
fn keeps_its_slots(s: &mut S<'_>, uploaded: bool) -> Vec<bool> {
    let wv = values(9, 32 * N);
    let run = |s: &mut S<'_>, x: &DramTensor, w: DramTensor| {
        let z = s.eltwise(op(kind::MUL), x, Some(&w)).unwrap();
        s.free(w).unwrap();
        let y = chain(s, &z);
        s.free(z).unwrap();
        y
    };
    let want: Vec<Vec<u32>> = SEEDS[..2]
        .iter()
        .map(|&seed| {
            let x = s.upload(&values(seed, 32 * N), 32, N).unwrap();
            let w = s.upload(&wv, 32, N).unwrap();
            let y = run(s, &x, w);
            let out = bits(&s.download(&y).unwrap());
            s.free(y).unwrap();
            s.free(x).unwrap();
            out
        })
        .collect();
    let x = s.upload(&values(1, 32 * N), 32, N).unwrap();
    let before = (!uploaded).then(|| s.upload(&wv, 32, N).unwrap());
    s.set_capture_reuse(true);
    s.begin_trace().unwrap();
    let w = before.unwrap_or_else(|| s.upload(&wv, 32, N).unwrap());
    let y = run(s, &x, w);
    let id = s.end_trace().unwrap();
    s.set_capture_reuse(false);
    let mut same = Vec::new();
    for (i, &seed) in SEEDS[..2].iter().enumerate() {
        let fresh = s.upload(&values(seed, 32 * N), 32, N).unwrap();
        s.copy_into(&fresh, &x).unwrap();
        s.free(fresh).unwrap();
        s.replay(id).unwrap();
        same.push(bits(&s.download(&y).unwrap()) == want[i]);
    }
    s.free(y).unwrap();
    s.release_trace(id).unwrap();
    s.free(x).unwrap();
    same
}

#[test]
fn a_tensor_from_before_the_capture_keeps_its_slots() {
    open(|s| {
        let same = keeps_its_slots(s, false);
        assert!(same.iter().all(|&x| x), "replays against eager: {same:?}");
    });
}

#[test]
fn a_constant_uploaded_in_the_capture_keeps_its_slots() {
    open(|s| {
        let same = keeps_its_slots(s, true);
        assert!(same.iter().all(|&x| x), "replays against eager: {same:?}");
    });
}
