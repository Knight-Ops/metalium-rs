//! A trace's programs paged into the program cache (`docs/plans/long-context.md`): the traces of
//! a generator need not fit a unit's 240 KiB cache together, only each alone. A trace keeps its
//! programs' words; where another trace's programs took the room since, its replay puts its own
//! back at the addresses its stream names (`ProgramCache::restore`), after the unit drains if
//! what it evicts may still be queued.
//!
//! The claims, on one unit whose cache is cut (`Session::set_profile_buffer`) to the larger
//! trace's programs and a kilobyte:
//! - **Both traces capture and replay**, alternately, with new inputs each time, and every
//!   replay's output is the same ops run eagerly, bit for bit.
//! - **They were paged:** replays put program bytes back (`Session::restored_bytes`).
//!
//! Watched failing (the simulator only: on silicon a stale program can hang a core): with
//! restoring switched off (`probe_skip_restore`), a replay runs the other trace's programs and
//! its output is not the eager one, or the simulator refuses it.

use tt_kernels::kind;
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::kind_sfpu;
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_ttsim::fork_scope;

const N: usize = 4096;

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Away from zero: `LOG` and `RECIP` are taken of them.
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
fn open<R>(f: impl FnOnce(&mut Session<Transport<'_>>) -> R) -> Result<(), String> {
    fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(dev, tt_firmware_images::ROLES, TileChoice::First, |_, _| {
            Ok(None)
        })
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    })
    .map_err(|e| e.to_string())
}

#[cfg(feature = "silicon")]
fn open<R>(f: impl FnOnce(&mut Session<Transport<'_>>) -> R) -> Result<(), String> {
    fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::First,
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    })
    .map_err(|e| e.to_string())
}

/// Trace `a`'s ops: `x / exp(x)`'s reciprocal form, `recip(exp(x)) * x`.
fn ops_a<T: tt_device::Transport>(s: &mut Session<T>, x: &DramTensor) -> DramTensor {
    let e = s.eltwise(op(kind_sfpu::EXP), x, None).unwrap();
    let r = s.eltwise(op(kind_sfpu::RECIP), &e, None).unwrap();
    s.eltwise(op(kind::MUL), &r, Some(x)).unwrap()
}

/// Trace `b`'s ops: `log(|x|) / x`.
fn ops_b<T: tt_device::Transport>(s: &mut Session<T>, x: &DramTensor) -> DramTensor {
    let a = s.eltwise(op(kind_sfpu::ABS), x, None).unwrap();
    let l = s.eltwise(op(kind_sfpu::LOG), &a, None).unwrap();
    s.eltwise(op(kind_sfpu::DIV), &l, Some(x)).unwrap()
}

type Ops<T> = fn(&mut Session<T>, &DramTensor) -> DramTensor;

/// What a trace's programs take of unit 0's cache, captured alone in the whole cache.
fn program_bytes<T: tt_device::Transport>(s: &mut Session<T>, ops: Ops<T>) -> u64 {
    let x = s.upload(&values(1, N), 1, N).unwrap();
    s.begin_trace().unwrap();
    let _ = ops(s, &x);
    let id = s.end_trace().unwrap();
    let bytes = s.trace_footprint(id).unwrap()[0]
        .as_ref()
        .unwrap()
        .program_bytes;
    s.release_trace(id).unwrap();
    bytes
}

/// The two traces in a cache that holds either but not both, replayed in turn with new inputs:
/// each replay's output against the eager one's, and the bytes restored.
fn alternate<T: tt_device::Transport>(s: &mut Session<T>) -> (Vec<bool>, u64) {
    let (pa, pb) = (program_bytes(s, ops_a), program_bytes(s, ops_b));
    let full = tt_isa::l1::PROGRAM_CACHE.len();
    // Room for the larger trace and a little: programs the two share (an unary op's unpack and
    // pack roles are the same words) are stored once, so half the smaller would still hold both.
    let cache = (pa.max(pb) + 1024).next_multiple_of(16);
    assert!(
        cache < pa + pb,
        "a {pa} and b {pb} bytes fit {cache} together"
    );
    eprintln!("a {pa} and b {pb} program bytes in a cache of {cache}");
    s.set_profile_buffer(Some(full - cache)).unwrap();
    let seeds = [2u64, 3, 4, 5];
    // The eager results, each input fresh.
    let mut want = Vec::new();
    for &seed in &seeds {
        let x = s.upload(&values(seed, N), 1, N).unwrap();
        let (ya, yb) = (ops_a(s, &x), ops_b(s, &x));
        want.push((
            bits(&s.download(&ya).unwrap()),
            bits(&s.download(&yb).unwrap()),
        ));
    }
    let (xa, xb) = (
        s.upload(&values(1, N), 1, N).unwrap(),
        s.upload(&values(1, N), 1, N).unwrap(),
    );
    s.begin_trace().unwrap();
    let ya = ops_a(s, &xa);
    let a = s.end_trace().unwrap();
    s.begin_trace().unwrap();
    let yb = ops_b(s, &xb);
    let b = s.end_trace().unwrap();
    let before = s.restored_bytes();
    let mut same = Vec::new();
    for (i, &seed) in seeds.iter().enumerate() {
        let x = s.upload(&values(seed, N), 1, N).unwrap();
        s.copy_into(&x, &xa).unwrap();
        s.copy_into(&x, &xb).unwrap();
        s.replay(a).unwrap();
        same.push(bits(&s.download(&ya).unwrap()) == want[i].0);
        s.replay(b).unwrap();
        same.push(bits(&s.download(&yb).unwrap()) == want[i].1);
    }
    (same, s.restored_bytes() - before)
}

#[test]
fn traces_that_do_not_fit_together_replay_by_turns() {
    open(|s| {
        let (same, restored) = alternate(s);
        assert!(same.iter().all(|&x| x), "replays against eager: {same:?}");
        assert!(restored > 0, "nothing was paged");
        eprintln!(
            "{restored} program bytes restored over {} replays",
            same.len()
        );
    })
    .unwrap_or_else(|e| panic!("{e}"));
}

#[cfg(not(feature = "silicon"))]
#[test]
fn without_restoring_a_replay_runs_the_other_trace_s_programs() {
    // The child panics only if the control passes (a panic's exit code); a wrong output or a
    // failed replay ends it normally, and the simulator refusing a stale program kills it with
    // another code. Either of those is the control failing, as it must.
    let r = open(|s| {
        tt_kernels::session::probe_skip_restore(true);
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| alternate(s)));
        if let Ok((same, _)) = run {
            if same.iter().all(|&x| x) {
                panic!("the control passed: {same:?}");
            }
            eprintln!("replays against eager without restoring: {same:?}");
        }
    });
    if let Err(e) = r {
        assert!(!e.contains("panicked"), "the control passed: {e}");
        eprintln!("the simulator refused the stale programs: {e}");
    }
}
