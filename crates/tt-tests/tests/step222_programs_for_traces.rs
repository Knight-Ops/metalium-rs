//! The program slots are trace data while a trace runs (`docs/plans/decode-to-bandwidth.md` Phase 8,
//! "Room for a second job's `B`", P2): a trace's capture plans into `tt_isa::l1::TRACE_ARENA`,
//! which takes in `PROGRAMS`, since a traced kernel runs from the program cache.
//!
//! The claims:
//! - **A capture's own run leaves the program slots alone.** Its setup runs (each unit's first
//!   kernel after `begin_capture` forgets the semaphores) run from the program cache
//!   (`runtime::Resident::setup_from_cache`), not role 0's fixed slot: with every unit's slots
//!   poisoned before a plan pass and a planned capture of a layer, every poisoned word is intact
//!   after both, and the replays equal the layer run fresh, bit for bit.
//! - **A capture in the trace arena replays as the ops run fresh** (P3, `Session::set_trace_arena`):
//!   its persistent region at the top of the trace arena, above `DATA`, its products' `B` rings
//!   reaching up to it past `DATA`, a plan pass and a planned capture of the layer replay bit for
//!   bit as it runs fresh and as the same captures outside the trace arena.
//! - **After a capture the eager runtime writes its fixed slots again.** A fixed-slot kernel
//!   (`Session::matmul`) whose programs the runtime last wrote, run again after its slots were
//!   overwritten and a capture ended, gives its own result bit for bit: the capture made the
//!   runtime forget what the slots hold (`Resident::forget_program_slots`), so it wrote them.
//!
//! Watched failing (ttsim only: on silicon the second would hang a core): with a capture's setup
//! placed in role 0's slot again, the poison there is gone; with the slots never forgotten, the
//! kernel after the capture runs the overwritten words, a shorter program then zeros, which ttsim
//! refuses (`tensix_decode_and_execute: opcode=0x0`).

use tt_device::tlb::WindowKind;
use tt_isa::l1::PROGRAMS;
use tt_isa::noc::{Noc0, NocCoord};
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const K: usize = 1024;
const N: usize = 512;
const EPS: f32 = 1e-5;
const POISON: u8 = 0xC3;
const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
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
fn with_session<R>(tiles: TileChoice, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>) -> R) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(dev, tt_firmware_images::ROLES, tiles, |_, _| Ok(None)).unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session<R>(tiles: TileChoice, f: impl FnOnce(&mut Session<tt_kmd::Kmd>) -> R) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            tiles,
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn fill<T: tt_device::Transport>(s: &mut Session<T>, tiles: &[NocCoord<Noc0>], bytes: &[u8]) {
    let d = s.device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    for &t in tiles {
        d.write(&w, t, PROGRAMS.base, bytes).unwrap();
    }
    d.free_window(w);
}

fn slots<T: tt_device::Transport>(s: &mut Session<T>, t: NocCoord<Noc0>) -> Vec<u8> {
    let d = s.device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let mut out = vec![0u8; PROGRAMS.len() as usize];
    d.read(&w, t, PROGRAMS.base, &mut out).unwrap();
    d.free_window(w);
    out
}

/// `x + a`, its norm times `gamma`, plus `x + a` again, then a one-row product.
fn layer<T: tt_device::Transport>(
    s: &mut Session<T>,
    x: &DramTensor,
    a: &DramTensor,
    gamma: &DramTensor,
    w: &DramTensor,
) -> DramTensor {
    use tt_kernels::kind;
    let h = s.eltwise(op(kind::ADD), x, Some(a)).unwrap();
    let n = s.rms_norm(&h, None, EPS).unwrap();
    let g = s.eltwise(op(kind::MUL), &n, Some(gamma)).unwrap();
    let r = s.eltwise(op(kind::ADD), &g, Some(&h)).unwrap();
    let out = s
        .matmul_dram(&r, false, w, false, ROUTE, Fidelity::HiFi4, BUDGET)
        .unwrap();
    for t in [h, n, g, r] {
        s.free(t).unwrap();
    }
    out
}

#[test]
fn a_captures_own_run_leaves_the_program_slots_alone() {
    with_session(TileChoice::Count(8), |s| {
        let seeds = [10u64, 20, 21];
        let a = s.upload(&values(2, K), 1, K).unwrap();
        let gamma = s.upload(&values(3, K), 1, K).unwrap();
        let w = s.upload(&values(4, K * N), K, N).unwrap();
        let x = s.upload(&values(seeds[0], K), 1, K).unwrap();
        let mut fresh = Vec::new();
        for &seed in &seeds {
            s.write(&x, &values(seed, K)).unwrap();
            let out = layer(s, &x, &a, &gamma, &w);
            fresh.push(bits(&s.download(&out).unwrap()));
            s.free(out).unwrap();
        }
        let tiles = s.tiles();
        let poison = vec![POISON; PROGRAMS.len() as usize];
        fill(s, &tiles, &poison);
        s.set_residency(tt_kernels::session::RESIDENCY_REGION);
        // In the trace arena the capture plans into the slots: off here, whatever the environment.
        s.set_trace_arena(false);
        s.write(&x, &values(seeds[0], K)).unwrap();
        s.begin_trace_plan_pass().unwrap();
        let plain_out = layer(s, &x, &a, &gamma, &w);
        let plain = s.end_trace().unwrap();
        s.write(&x, &values(seeds[0], K)).unwrap();
        s.begin_trace_planned().unwrap();
        let out = layer(s, &x, &a, &gamma, &w);
        let planned = s.end_trace().unwrap();
        for &t in &tiles {
            let got = slots(s, t);
            let stray = got.iter().position(|&b| b != POISON);
            assert_eq!(
                stray, None,
                "{t:?}: a program slot written at byte {stray:?}"
            );
        }
        let mut got = vec![bits(&s.download(&out).unwrap())];
        for &seed in &seeds[1..] {
            s.write(&x, &values(seed, K)).unwrap();
            s.replay(planned).unwrap();
            s.sync().unwrap();
            got.push(bits(&s.download(&out).unwrap()));
        }
        assert_eq!(got, fresh, "the planned capture and its replays");
        for id in [plain, planned] {
            s.release_trace(id).unwrap();
        }
        for t in [plain_out, out, x, w, gamma, a] {
            s.free(t).unwrap();
        }
    });
}

/// A plan pass and a planned capture of the layer, replayed over `seeds` after the first: each
/// result's bits, the region's base and the highest `l1_top` of the planned capture's ops.
fn captured<T: tt_device::Transport>(
    s: &mut Session<T>,
    tensors: [&DramTensor; 4],
    seeds: &[u64],
    what: &str,
) -> (Vec<Vec<u32>>, u64, u64) {
    let [x, a, gamma, w] = tensors;
    s.write(x, &values(seeds[0], K)).unwrap();
    s.begin_trace_plan_pass().unwrap();
    let plain_out = layer(s, x, a, gamma, w);
    let plain = s.end_trace().unwrap();
    let base = s
        .probe_last_plan()
        .unwrap_or_else(|| panic!("{what}: no plan"))
        .region_base;
    s.write(x, &values(seeds[0], K)).unwrap();
    assert!(s.begin_trace_planned().unwrap(), "{what}: nothing resident");
    let out = layer(s, x, a, gamma, w);
    let planned = s.end_trace().unwrap();
    let top = s
        .trace_ops(planned)
        .unwrap()
        .iter()
        .map(|o| o.l1_top)
        .max()
        .unwrap();
    let mut got = vec![bits(&s.download(&out).unwrap())];
    for &seed in &seeds[1..] {
        s.write(x, &values(seed, K)).unwrap();
        s.replay(planned).unwrap();
        s.sync().unwrap();
        got.push(bits(&s.download(&out).unwrap()));
    }
    for id in [plain, planned] {
        s.release_trace(id).unwrap();
    }
    for t in [plain_out, out] {
        s.free(t).unwrap();
    }
    (got, base, top)
}

#[test]
fn a_capture_in_the_trace_arena_replays_as_the_ops_run_fresh() {
    use tt_isa::l1::{DATA, TRACE_ARENA};
    with_session(TileChoice::Count(8), |s| {
        let seeds = [10u64, 20, 21];
        let a = s.upload(&values(2, K), 1, K).unwrap();
        let gamma = s.upload(&values(3, K), 1, K).unwrap();
        let w = s.upload(&values(4, K * N), K, N).unwrap();
        let x = s.upload(&values(seeds[0], K), 1, K).unwrap();
        let mut fresh = Vec::new();
        for &seed in &seeds {
            s.write(&x, &values(seed, K)).unwrap();
            let out = layer(s, &x, &a, &gamma, &w);
            fresh.push(bits(&s.download(&out).unwrap()));
            s.free(out).unwrap();
        }
        s.set_residency(tt_kernels::session::RESIDENCY_REGION);
        s.set_trace_arena(false);
        let (outside, base, top) = captured(s, [&x, &a, &gamma, &w], &seeds, "outside");
        assert_eq!(outside, fresh, "outside the trace arena");
        assert!(base < DATA.end && top <= base, "{base:#x} {top:#x}");
        s.set_trace_arena(true);
        let (inside, base, top) = captured(s, [&x, &a, &gamma, &w], &seeds, "inside");
        assert_eq!(inside, fresh, "in the trace arena");
        assert!(
            (DATA.end..TRACE_ARENA.end).contains(&base),
            "the region at {base:#x}"
        );
        assert!(
            top > DATA.end && top <= base,
            "the products' rings reach {top:#x}, the region at {base:#x}"
        );
        s.set_trace_arena(false);
        for t in [x, w, gamma, a] {
            s.free(t).unwrap();
        }
    });
}

#[test]
fn after_a_capture_a_fixed_slot_kernel_writes_its_programs_again() {
    with_session(TileChoice::First, |s| {
        let shapes = [[32, 64, 32], [64, 32, 96]];
        let inputs: Vec<(Vec<f32>, Vec<f32>)> = shapes
            .iter()
            .enumerate()
            .map(|(i, &[m, k, n])| (values(30 + i as u64, m * k), values(40 + i as u64, k * n)))
            .collect();
        let run = |s: &mut Session<_>, i: usize| {
            let (a, b) = &inputs[i];
            bits(
                &s.matmul(a, b, shapes[i], ROUTE, Fidelity::HiFi4, BUDGET)
                    .unwrap(),
            )
        };
        let tile = s.tile();
        run(s, 0);
        let first = slots(s, tile);
        let want = run(s, 1);
        // The second kernel's programs are what the runtime last wrote; put the first's back, as
        // a trace's data would land there, then capture.
        fill(s, &[tile], &first);
        let x = s.upload(&values(5, K), 1, K).unwrap();
        s.begin_trace().unwrap();
        let y = s.eltwise(op(tt_kernels::kind::ADD), &x, Some(&x)).unwrap();
        let id = s.end_trace().unwrap();
        assert_eq!(run(s, 1), want, "the kernel after the capture");
        s.release_trace(id).unwrap();
        for t in [x, y] {
            s.free(t).unwrap();
        }
    });
}
