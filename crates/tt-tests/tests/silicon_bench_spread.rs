//! What a per-op spread (`Session::set_spread`) buys on a session that opens every tile: small ops
//! want few tiles (each costs the host ~5 us an op), a large op wants many (its device time divides).
//!
//! `cargo xtask bench --device 1 --filter silicon_bench_spread`
//!
//! On one session of every tile, for each spread: 64 chained relus of a 64x64 tensor (small ops;
//! host-timed per op, queued and then synced), and one add of two 2048x2048 tensors (a large op; host
//! time to its end). Medians of `REPS` after a warm-up; every result is checked against the host.

#![cfg(feature = "silicon")]

use std::time::Instant;

use tt_kernels::kind;
use tt_kernels::session::{Session, Spread, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_tests::backend::device_index;
use tt_ttsim::fork_scope;

const REPS: usize = 7;
const CHAIN: usize = 64;

fn ew(
    s: &mut Session<tt_kmd::Kmd>,
    kind: u32,
    a: &DramTensor,
    b: Option<&DramTensor>,
) -> DramTensor {
    s.eltwise(
        Eltwise {
            scalar2: 0.0,
            kind,
            scalar: 0.0,
        },
        a,
        b,
    )
    .unwrap()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "benchmark"]
fn small_and_large_ops_by_spread() {
    let card = device_index();
    fork_scope(|| {
        let mut s = Session::open_card(card, tt_firmware_images::ROLES, TileChoice::All)
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        let tiles = s.lists_per_tile().len();
        let small_in: Vec<f32> = (0..64 * 64).map(|i| (i % 13) as f32 - 6.0).collect();
        let big_in: Vec<f32> = (0..2048 * 2048).map(|i| (i % 7) as f32 - 3.0).collect();
        let small = s.upload(&small_in, 64, 64).unwrap();
        let big = s.upload(&big_in, 2048, 2048).unwrap();
        let want_small: Vec<f32> = small_in.iter().map(|v| v.max(0.0)).collect();
        let want_big: Vec<f32> = big_in.iter().map(|v| v + v).collect();
        for spread in [
            Spread::All,
            Spread::Fixed(4),
            Spread::Work { share: 64, min: 4 },
            Spread::Work { share: 16, min: 4 },
        ] {
            s.set_spread(spread);
            let mut small_us = Vec::new();
            let mut big_us = Vec::new();
            for rep in 0..REPS + 1 {
                // Small ops: CHAIN relus queued, then synced; per op.
                let t = Instant::now();
                let mut cur = ew(&mut s, kind::RELU, &small, None);
                for _ in 1..CHAIN {
                    let next = ew(&mut s, kind::RELU, &cur, None);
                    s.free(cur).unwrap();
                    cur = next;
                }
                s.sync().unwrap();
                let per_op = t.elapsed().as_secs_f64() * 1e6 / CHAIN as f64;
                let got = s.download(&cur).unwrap();
                assert_eq!(got, want_small, "{spread:?}: the relu chain");
                s.free(cur).unwrap();
                // One large op, to its end.
                let t = Instant::now();
                let sum = ew(&mut s, kind::ADD, &big, Some(&big));
                s.sync().unwrap();
                let large = t.elapsed().as_secs_f64() * 1e6;
                if rep == 0 {
                    let got = s.download(&sum).unwrap();
                    assert!(got == want_big, "{spread:?}: the large add");
                }
                s.free(sum).unwrap();
                if rep > 0 {
                    small_us.push(per_op);
                    big_us.push(large);
                }
            }
            println!(
                "MEASURE spread {spread:?} on {tiles} tiles: small op {:.1} us (a 64x64 relu, {CHAIN} chained), large op {:.0} us (a 2048x2048 add)",
                median(small_us),
                median(big_us)
            );
        }
    })
    .unwrap_or_else(|e| panic!("{e}"));
}
