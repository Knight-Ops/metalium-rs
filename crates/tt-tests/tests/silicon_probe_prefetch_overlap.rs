//! B2 of `docs/plans/decode-step-bandwidth.md`: can weights stream into idle units' L1 while a
//! decode layer's small ops run, and what does that cost the small ops?
//!
//! A decode step's small ops (18.3 of 42.3 ms, profile `1792202359`) mostly run on units 0-63 and
//! move almost nothing through GDDR, while units 64-119 sit idle. The probe, on every tile of the
//! card, release, host-timed with a sync, warm-ups and medians:
//! - **small ops alone:** a chain of element-wise adds on a `[1, 2048]` activation (64 units, the
//!   shape of a decode step's residual adds and RoPE products);
//! - **the stream alone:** `Session::probe_prefetch`, every idle unit reading `per_unit` weight
//!   tiles into the top of its `DATA` arena, queued without a barrier; and four back to back;
//! - **both:** the stream queued first, then the chain;
//! - how long the chain's calls take to return, against its work's end.
//!
//! The chain's result is checked bit for bit against the chain alone, and every tile the stream
//! landed against its GDDR image, on the first and last idle unit.
//!
//! **What it found (card 0, `1792210511` and after):** eagerly, everything here is the host's time.
//! The adds' calls return in ~290 us each (queueing an op on 64 units) and the work ends ~0.3 ms
//! after the last; the stream's ~300 us is its own queueing on 56 units. "Both" is the sum of the
//! two host costs, so the eager path cannot show whether the card overlaps them. That needs a
//! replay, and a captured op's barrier involves every unit (`Session::enqueue_work`), so a unit
//! streaming between two ops holds both ops' barriers: the prefetch first needs barriers on the
//! units an op ran on (`docs/plans/replay-fixed-cost.md` P1). Kept as the harness for that.
#![cfg(feature = "silicon")]

use std::time::{Duration, Instant};

use tt_device::tlb::WindowKind;
use tt_isa::dm::TILE_SLOT;
use tt_kernels::kind;
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_tests::backend::device_index;
use tt_ttsim::fork_scope;

type Card = Session<tt_kmd::Kmd>;

const WARMUPS: usize = 2;
const SAMPLES: usize = 9;
/// The units a decode step's 64-tile ops run on.
const BUSY: usize = 64;

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

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

/// `y = y + x`, `ops` times, from `x`: the chain's result (the caller frees it).
fn chain(s: &mut Card, x: &DramTensor, ops: usize) -> DramTensor {
    let add = Eltwise {
        kind: kind::ADD,
        scalar: 0.0,
        scalar2: 0.0,
    };
    let mut y = s.eltwise(add, x, Some(x)).unwrap();
    for _ in 1..ops {
        let next = s.eltwise(add, &y, Some(x)).unwrap();
        s.free(y).unwrap();
        y = next;
    }
    y
}

/// Every tile the stream left on unit `u` (the `k`-th idle unit) equals `w`'s in GDDR.
fn landed(s: &mut Card, w: &DramTensor, u: usize, k: usize, at: u64, per_unit: usize) {
    let tile = s.tiles()[u];
    let window = s.device().alloc_window(WindowKind::FourGib).unwrap();
    let ct = w.grid()[1];
    let mut l1 = vec![0u8; per_unit * TILE_SLOT as usize];
    s.device().l1_read(&window, tile, at, &mut l1).unwrap();
    for i in 0..per_unit {
        let n = k * per_unit + i;
        let mut gddr = vec![0u8; TILE_SLOT as usize];
        s.device()
            .dram_read(&window, w.tile(n / ct, n % ct), &mut gddr)
            .unwrap();
        let slot = &l1[i * TILE_SLOT as usize..(i + 1) * TILE_SLOT as usize];
        assert!(slot == &gddr[..], "unit {u}: slot {i} is not tile {n}");
    }
}

#[test]
#[ignore = "benchmark"]
fn weights_stream_into_idle_units_under_small_ops() {
    let card = device_index();
    fork_scope(|| {
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::All)
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        // Queued without a host wait per op, as a replay's ops are: otherwise each submission
        // syncs, and the stream ends before the chain starts.
        s.set_batching(true).unwrap();
        let n = s.tiles().len();
        assert!(n > BUSY, "{n} tiles: none idle beside {BUSY}");
        let idle = BUSY..n;
        let x = s.upload(&floats(1, 2048), 1, 2048).unwrap();
        // `o`'s share a unit (64 tiles, 266 KB) and `gate|up`'s first round's (128, 532 KB).
        for per_unit in [64usize, 128] {
            let tiles = idle.len() * per_unit;
            let cols = tiles.div_ceil(64) * 32;
            let w = s.upload(&floats(2, 2048 * cols), 2048, cols).unwrap();
            let at = tt_isa::l1::DATA.end - per_unit as u64 * TILE_SLOT;
            let bytes = (tiles as u64 * TILE_SLOT) as f64;
            for ops in [8usize, 32] {
                let mut times = [Vec::new(), Vec::new(), Vec::new()];
                let reference = chain(&mut s, &x, ops);
                let want = s.download(&reference).unwrap();
                s.free(reference).unwrap();
                for i in 0..WARMUPS + SAMPLES {
                    for (k, (stream, small)) in [(false, true), (true, false), (true, true)]
                        .into_iter()
                        .enumerate()
                    {
                        s.sync().unwrap();
                        let t0 = Instant::now();
                        if stream {
                            s.probe_prefetch(&w, idle.clone(), at, per_unit).unwrap();
                        }
                        let y = small.then(|| chain(&mut s, &x, ops));
                        s.sync().unwrap();
                        let dt = t0.elapsed();
                        if i >= WARMUPS {
                            times[k].push(dt);
                        }
                        if let Some(y) = y {
                            let got = s.download(&y).unwrap();
                            s.free(y).unwrap();
                            let same = got
                                .iter()
                                .zip(&want)
                                .all(|(a, b)| a.to_bits() == b.to_bits());
                            assert!(same, "the chain's result moved under the stream");
                        }
                        if stream && i == 0 {
                            landed(&mut s, &w, idle.start, 0, at, per_unit);
                            landed(&mut s, &w, idle.end - 1, idle.len() - 1, at, per_unit);
                        }
                    }
                }
                let [small, stream, both] = times.map(median);
                // The stream's own cost against its fixed one: four back to back.
                let mut four = Vec::new();
                for i in 0..WARMUPS + SAMPLES {
                    s.sync().unwrap();
                    let t0 = Instant::now();
                    for _ in 0..4 {
                        s.probe_prefetch(&w, idle.clone(), at, per_unit).unwrap();
                    }
                    s.sync().unwrap();
                    if i >= WARMUPS {
                        four.push(t0.elapsed());
                    }
                }
                let four = median(four);
                // Whether the chain's calls return before its work is done.
                s.sync().unwrap();
                let t0 = Instant::now();
                let y = chain(&mut s, &x, ops);
                let queued = t0.elapsed();
                s.sync().unwrap();
                let done = t0.elapsed();
                s.free(y).unwrap();
                println!(
                    "{ops} adds: the calls return in {:.1} us, the work is done in {:.1} us",
                    queued.as_secs_f64() * 1e6,
                    done.as_secs_f64() * 1e6
                );
                let us = |d: Duration| d.as_secs_f64() * 1e6;
                let hidden = (us(small) + us(stream) - us(both)) / us(small).min(us(stream));
                println!(
                    "{} idle units x {per_unit} tiles ({:.1} MB), {ops} adds on {BUSY}: small ops \
                     {:.1} us, stream {:.1} us ({:.0} GB/s; four {:.1} us, {:.0} GB/s at the \
                     margin), both {:.1} us; {:.0}% of the shorter hidden",
                    idle.len(),
                    bytes / 1e6,
                    us(small),
                    us(stream),
                    bytes / us(stream) / 1e3,
                    us(four),
                    3.0 * bytes / (us(four) - us(stream)) / 1e3,
                    us(both),
                    100.0 * hidden
                );
            }
            s.free(w).unwrap();
        }
        s.free(x).unwrap();
    })
    .unwrap_or_else(|e| panic!("{e}"));
}
