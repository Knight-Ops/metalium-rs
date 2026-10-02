//! Phase 10 gate (X4c): ops queued by a batching session run as if each had
//! been waited for.
//!
//! The case that broke on silicon: a program cache too small for every
//! queued list's programs. Making room by evicting would overwrite programs
//! lists still queued run, so the session drains the queue first. Here the
//! cache holds about two ops' programs and a chain of forty alternates four
//! kinds, so room is made over and over while lists are queued; the result is
//! the chain run on the host by the interpreter, bit for bit.

use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::{kind_sfpu, reference};
use tt_kernels::tensor::Eltwise;
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    with_tiles(TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1), f)
}

#[cfg(not(feature = "silicon"))]
fn with_tiles(choice: TileChoice, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(dev, tt_firmware_images::ROLES, choice, |_, _| Ok(None)).unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    with_tiles(TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1), f)
}

#[cfg(feature = "silicon")]
fn with_tiles(choice: TileChoice, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            choice,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[test]
fn a_full_program_cache_never_evicts_what_queued_lists_run() {
    with_session(|s| {
        let (r, c) = (32, 32);
        let kinds = [
            (kind_sfpu::RECIP, 0.0),
            (kind_sfpu::EXP, 0.0),
            (kind_sfpu::LOG, 0.0),
            (kind_sfpu::DIV_SCALAR, 3.0),
        ];
        let x: Vec<f32> = (0..r * c).map(|i| 0.5 + (i % 97) as f32 / 64.0).collect();
        let a = s.upload(&x, r, c).unwrap();
        // Each kind's programs, measured: the cache is then cut to twice the
        // largest kind's, so any one program is admitted (at most half the
        // cache) and about two kinds are resident at a time.
        s.set_batching(false).unwrap();
        let mut sizes = Vec::new();
        for &(kind, scalar) in &kinds {
            let before = s.program_cache_stats()[0].bytes_uploaded;
            let out = s
                .eltwise(
                    Eltwise {
                        scalar2: 0.0,
                        kind,
                        scalar,
                    },
                    &a,
                    None,
                )
                .unwrap();
            sizes.push(s.program_cache_stats()[0].bytes_uploaded - before);
            s.free(out).unwrap();
        }
        let limit = 2 * sizes.iter().max().unwrap();
        s.limit_program_cache(limit).unwrap();
        s.set_batching(true).unwrap();

        let mut want = x.clone();
        let mut cur = a;
        for step in 0..40 {
            let (kind, scalar) = kinds[step % kinds.len()];
            let out = s
                .eltwise(
                    Eltwise {
                        scalar2: 0.0,
                        kind,
                        scalar,
                    },
                    &cur,
                    None,
                )
                .unwrap();
            want = reference(kind, scalar, &want, None, r, c);
            s.free(cur).unwrap();
            cur = out;
        }
        let got = s.download(&cur).unwrap();
        assert!(
            s.program_cache_stats()[0].evictions > 0,
            "vacuous: the cache never had to make room"
        );
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "element {i}: device {g:e}, host {w:e}"
            );
        }
        s.free(cur).unwrap();
    });
}

/// A layer's forward pass at a size that gives every one of four units
/// several lists per op -- `[m, 784] @ [784, 128]`, a bias row, `relu`,
/// `@ [128, 10]` -- is the same bits queued as run op by op.
#[test]
fn several_lists_per_unit_queued_are_the_unbatched_bits() {
    use tt_kernels::kind;
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    let m = if cfg!(feature = "silicon") { 1000 } else { 200 };
    let values = |seed: u64, n: usize| -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    };
    let (x, w1, b1, w2) = (
        values(1, m * 784),
        values(2, 784 * 128),
        values(3, 128),
        values(4, 128 * 10),
    );
    with_tiles(TileChoice::Count(4), |s| {
        let forward = |s: &mut Session<_>, batching: bool| {
            s.set_batching(batching).unwrap();
            let t = |s: &mut Session<_>, v: &[f32], r, c| s.upload(v, r, c).unwrap();
            let (x, w1, b1, w2) = (
                t(s, &x, m, 784),
                t(s, &w1, 784, 128),
                t(s, &b1, 1, 128),
                t(s, &w2, 128, 10),
            );
            let mm = |s: &mut Session<_>, a, b| {
                s.matmul_dram(
                    a,
                    false,
                    b,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    1 << 40,
                )
                .unwrap()
            };
            let h = mm(s, &x, &w1);
            let hb = s
                .eltwise(
                    Eltwise {
                        scalar2: 0.0,
                        kind: kind::ADD_ROW,
                        scalar: 0.0,
                    },
                    &h,
                    Some(&b1),
                )
                .unwrap();
            let r = s
                .eltwise(
                    Eltwise {
                        scalar2: 0.0,
                        kind: kind::RELU,
                        scalar: 0.0,
                    },
                    &hb,
                    None,
                )
                .unwrap();
            let y = mm(s, &r, &w2);
            let out = s.download(&y).unwrap();
            for t in [x, w1, b1, w2, h, hb, r, y] {
                s.free(t).unwrap();
            }
            out
        };
        let direct = forward(s, false);
        let lists = s.lists_per_tile();
        assert!(
            lists.iter().all(|&l| l > 0),
            "a unit did nothing: {lists:?}"
        );
        // Several passes: a missing order between queued lists is a race,
        // which one pass can win.
        for pass in 0..8 {
            let queued = forward(s, true);
            for (i, (q, d)) in queued.iter().zip(&direct).enumerate() {
                assert_eq!(
                    q.to_bits(),
                    d.to_bits(),
                    "pass {pass}, element {i}: queued {q:e}, direct {d:e}"
                );
            }
        }
    });
}

/// An upload is in GDDR before the op queued after it reads it, the movers
/// busy or idle: `Device::dram_write` reads its last word back. A guard, not a
/// reproduction: the race it closes -- the host's posted writes losing to a
/// list that reaches the mover over its tile's L1 -- showed only in a batched
/// `tt-mnist` on four tiles (the first test batch after training read part
/// stale images, and later ones did with the queue drained first; wrong with
/// the read-back disabled, right with it, every run). This test passes either
/// way on card 0.
#[test]
fn an_upload_lands_before_the_queued_op_that_reads_it() {
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    let m = if cfg!(feature = "silicon") { 1000 } else { 64 };
    let (k, n) = (784, 128);
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
        .collect();
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i * 104_729) % 997) as f32 / 997.0 - 0.5)
        .collect();
    with_tiles(TileChoice::Count(4), |s| {
        let mm = |s: &mut Session<_>, a: &_, b: &_| {
            s.matmul_dram(
                a,
                false,
                b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                1 << 40,
            )
            .unwrap()
        };
        let w = s.upload(&w, k, n).unwrap();
        // Two inputs, alternated: each upload lands on slots the other's
        // just gave back, so bytes that have not arrived read as the other's.
        let xs = [x.clone(), x.iter().map(|v| 1.0 - v).collect::<Vec<f32>>()];
        // The bits with nothing in flight.
        s.set_batching(false).unwrap();
        let mut want = Vec::new();
        for x in &xs {
            let t = s.upload(x, m, k).unwrap();
            let y = mm(s, &t, &w);
            want.push(s.download(&y).unwrap());
            s.free(y).unwrap();
            s.free(t).unwrap();
        }
        let t = s.upload(&xs[0], m, k).unwrap();
        s.set_batching(true).unwrap();
        for round in 0..32 {
            // Odd rounds with the movers busy, even ones idle and polling.
            let busy: Vec<_> = (0..4 * (round % 2)).map(|_| mm(s, &t, &w)).collect();
            let which = (round / 2) % 2;
            let fresh = s.upload(&xs[which], m, k).unwrap();
            let y = mm(s, &fresh, &w);
            let got = s.download(&y).unwrap();
            for (i, (g, w)) in got.iter().zip(&want[which]).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "round {round}, element {i}");
            }
            for b in busy.into_iter().chain([fresh, y]) {
                s.free(b).unwrap();
            }
        }
        s.free(t).unwrap();
        s.free(w).unwrap();
    });
}

/// The barrier counter starts from zero in every session. It lives in unit
/// 0's L1, which keeps what an earlier process left: a stale count past every
/// target lets each barrier through at once (silicon: a batched four-tile
/// MNIST diverged). A count left behind here, then one multi-unit op: the
/// counter holds exactly that op's arrivals.
#[test]
fn barriers_count_from_zero_whatever_an_earlier_session_left() {
    use tt_device::tlb::WindowKind;
    use tt_kernels::kind;
    with_tiles(TileChoice::Count(4), |s| {
        s.set_batching(true).unwrap();
        let coordinator = s.tile();
        let stale = 0x00DE_AD00;
        {
            let d = s.device();
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            d.write32(&w, coordinator, tt_isa::dm::BARRIER_COUNTER, stale)
                .unwrap();
        }
        let v: Vec<f32> = (0..256 * 128).map(|i| i as f32).collect();
        let a = s.upload(&v, 256, 128).unwrap();
        let y = s
            .eltwise(
                Eltwise {
                    scalar2: 0.0,
                    kind: kind::RELU,
                    scalar: 0.0,
                },
                &a,
                None,
            )
            .unwrap();
        s.sync().unwrap();
        let d = s.device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let count = d
            .read32(&w, coordinator, tt_isa::dm::BARRIER_COUNTER)
            .unwrap();
        assert_eq!(
            count, 4,
            "one barrier, four arrivals, from zero (stale {stale:#x})"
        );
        s.free(y).unwrap();
        s.free(a).unwrap();
    });
}
