//! The L1 map after the mailboxes were packed above the program slots
//! (`docs/plans/decode-to-bandwidth.md` Phase 8, "Room for a second job's `B`", M2): `DATA`, then
//! the capture-only arena where the mailboxes were, then `PROGRAMS`, then the 40 KiB of
//! `MAILBOXES`, each block of which is in `tt_isa::l1::FIXED`.
//!
//! The claims:
//! - **Nothing writes the old mailboxes, nor the new ones outside their fixed blocks.** The
//!   capture-only arena (the old mailboxes' range) and every word of `MAILBOXES` that no block of
//!   `FIXED` holds are poisoned on every unit, a session runs eager element-wise ops, a norm, a
//!   one-row product, and a planned capture with its replays, and every poisoned word is still the
//!   poison. On ttsim the poison goes in before the session opens, so its bring-up and the movers'
//!   starts are covered too; on silicon, after (the card session opens the device itself). A host
//!   or firmware word left at an address of the old layout lands in the capture-only arena; one at
//!   an offset of it from the new base lands in a gap, or breaks the block it lands in, which the
//!   workload's results show. **The control:** a word the test writes into a gap is reported, and
//!   only it.
//! - **A session opened on another session's L1 computes as the first** (ttsim; on silicon every
//!   process after the first is this case): the same workload, run by a second session on the
//!   device the first handed back (`Session::into_device`), every result bit for bit. ttsim starts
//!   a process from zeroed L1, so this is how it sees the mailboxes, the program slots and the
//!   movers' state as an earlier process left them.

use tt_device::tlb::WindowKind;
use tt_isa::l1::{CAPTURE_ONLY, FIXED, MAILBOXES};
use tt_isa::noc::{Noc0, NocCoord};
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const K: usize = 512;
const N: usize = 256;
const EPS: f32 = 1e-5;
const POISON: u32 = 0x5A17_C0DE;

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

/// The capture-only arena, then the ranges of `MAILBOXES` no block of `FIXED` holds.
fn gaps() -> Vec<std::ops::Range<u64>> {
    let mut out = Vec::new();
    out.push(CAPTURE_ONLY.base..CAPTURE_ONLY.end);
    let mut at = MAILBOXES.base;
    for b in FIXED {
        if b.base > at {
            out.push(at..b.base);
        }
        at = at.max(b.end);
    }
    if MAILBOXES.end > at {
        out.push(at..MAILBOXES.end);
    }
    out
}

fn poison<T: tt_device::Transport>(d: &mut tt_device::Device<T>, tiles: &[NocCoord<Noc0>]) {
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    for &t in tiles {
        for g in gaps() {
            let bytes: Vec<u8> = (g.start..g.end)
                .step_by(4)
                .flat_map(|_| POISON.to_le_bytes())
                .collect();
            d.write(&w, t, g.start, &bytes).unwrap();
        }
    }
    d.free_window(w);
}

/// Every word of a gap that is not the poison, by tile and address.
fn touched<T: tt_device::Transport>(
    d: &mut tt_device::Device<T>,
    tiles: &[NocCoord<Noc0>],
) -> Vec<(NocCoord<Noc0>, u64, u32)> {
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let mut out = Vec::new();
    for &t in tiles {
        for g in gaps() {
            let mut bytes = vec![0u8; (g.end - g.start) as usize];
            d.read(&w, t, g.start, &mut bytes).unwrap();
            for (i, word) in bytes.chunks_exact(4).enumerate() {
                let v = u32::from_le_bytes(word.try_into().unwrap());
                if v != POISON {
                    out.push((t, g.start + 4 * i as u64, v));
                }
            }
        }
    }
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
        .matmul_dram(
            &r,
            false,
            w,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET,
        )
        .unwrap();
    for t in [h, n, g, r] {
        s.free(t).unwrap();
    }
    out
}

/// The layer run fresh, then captured (a plan pass and a planned capture) and replayed over the
/// seeds: every result's bits, in order.
fn workload<T: tt_device::Transport>(s: &mut Session<T>) -> Vec<Vec<u32>> {
    let seeds = [10u64, 20, 21];
    let a = s.upload(&values(2, K), 1, K).unwrap();
    let gamma = s.upload(&values(3, K), 1, K).unwrap();
    let w = s.upload(&values(4, K * N), K, N).unwrap();
    let x = s.upload(&values(seeds[0], K), 1, K).unwrap();
    let mut got = Vec::new();
    for &seed in &seeds {
        s.write(&x, &values(seed, K)).unwrap();
        let out = layer(s, &x, &a, &gamma, &w);
        got.push(bits(&s.download(&out).unwrap()));
        s.free(out).unwrap();
    }
    s.set_residency(tt_kernels::session::RESIDENCY_REGION);
    // A capture in the trace arena writes the capture-only arena: off here, whatever the
    // environment (`step222` covers it on).
    s.set_trace_arena(false);
    s.write(&x, &values(seeds[0], K)).unwrap();
    s.begin_trace_plan_pass().unwrap();
    let plain_out = layer(s, &x, &a, &gamma, &w);
    let plain = s.end_trace().unwrap();
    s.write(&x, &values(seeds[0], K)).unwrap();
    s.begin_trace_planned().unwrap();
    let out = layer(s, &x, &a, &gamma, &w);
    let planned = s.end_trace().unwrap();
    got.push(bits(&s.download(&out).unwrap()));
    for &seed in &seeds[1..] {
        s.write(&x, &values(seed, K)).unwrap();
        s.replay(planned).unwrap();
        s.sync().unwrap();
        got.push(bits(&s.download(&out).unwrap()));
    }
    for id in [plain, planned] {
        s.release_trace(id).unwrap();
    }
    for t in [plain_out, out, x, w, gamma, a] {
        s.free(t).unwrap();
    }
    got
}

/// The fresh results are the replays': the workload computed what it should.
fn assert_consistent(got: &[Vec<u32>]) {
    assert_eq!(got.len(), 6);
    assert_eq!(got[3], got[0], "the planned capture's own run");
    assert_eq!(got[4..], got[1..3], "the replays");
}

fn check_gaps<T: tt_device::Transport>(s: &mut Session<T>, tiles: &[NocCoord<Noc0>]) {
    let got = workload(s);
    assert_consistent(&got);
    let d = s.device();
    let stray = touched(d, tiles);
    assert!(
        stray.is_empty(),
        "words written outside the mailboxes' fixed blocks: {:x?}",
        &stray[..stray.len().min(16)]
    );
    // The control: a word written into a gap is reported, and only it. The first gap's first word
    // is the old layout's single-core `STATUS`.
    let at = gaps()[0].start;
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    d.write32(&w, tiles[0], at, !POISON).unwrap();
    d.free_window(w);
    assert_eq!(touched(d, tiles), vec![(tiles[0], at, !POISON)]);
}

#[test]
fn the_gaps_of_the_l1_map_are_what_fixed_does_not_hold() {
    let g = gaps();
    let free: u64 = g.iter().map(|r| r.end - r.start).sum();
    let held: u64 = FIXED.iter().map(|b| b.len()).sum();
    assert_eq!(free + held, CAPTURE_ONLY.len() + MAILBOXES.len());
    assert!(g.iter().all(|r| r.start % 4 == 0 && r.end % 4 == 0));
    assert!(free > 0, "something to poison");
}

#[cfg(not(feature = "silicon"))]
#[test]
fn nothing_writes_the_mailboxes_outside_their_fixed_blocks() {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let mut dev = tt_device::Device::open(sim.transport()).unwrap();
        let grid = tt_kernels::session::tensix_grid(&mut dev).unwrap();
        let all: Vec<NocCoord<Noc0>> = grid.tiles::<Noc0>().collect();
        poison(&mut dev, &all);
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(8),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        check_gaps(&mut s, &all);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
#[test]
fn nothing_writes_the_mailboxes_outside_their_fixed_blocks() {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(8),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        let tiles = s.tiles();
        poison(s.device(), &tiles);
        check_gaps(&mut s, &tiles);
    }) {
        panic!("{e}");
    }
}

#[cfg(not(feature = "silicon"))]
#[test]
fn a_session_opened_on_another_sessions_l1_computes_as_the_first() {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let mut dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut first = Vec::new();
        for round in 0..2 {
            let mut s = Session::open(
                dev,
                tt_firmware_images::ROLES,
                TileChoice::Count(8),
                |_, _| Ok(None),
            )
            .unwrap();
            s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
            let got = workload(&mut s);
            assert_consistent(&got);
            if round == 0 {
                first = got;
            } else {
                assert_eq!(got, first, "the second session's results");
            }
            dev = s.into_device();
        }
    }) {
        panic!("{e}");
    }
}
