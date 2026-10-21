//! 2f of `docs/plans/planned-l1-residency.md`: what each way of delivering a row that many units
//! read whole costs on the card (`tt_kernels::residency::RowScheme`,
//! `Session::probe_row_delivery`).
//!
//! `cargo xtask silicon --release --profile --device 0 --filter silicon_probe_row_delivery`
//!
//! On every tile of the card, release, the profile mode's per-op spans (medians over replays):
//! - **the rows:** a `[1, 2048]` row (64 output tiles, a 2-tile twin), written by 64 producers a
//!   tile each, as `gamma`'s multiply writes it, read whole by 32 (RMSNorm's units), 64 and 104 (a
//!   product's, prefetching idle units included); and a `[1, 8192]` row (256 tiles, an 8-tile
//!   twin), 4 tiles a producer, read by 64, as `silu_mul`'s is by `down`;
//! - **the schemes:** through GDDR (the baseline), Phase 2's multicast from every producer, the
//!   same with a producer's acknowledgements counted once (2f (c)), 1, 2, 4 and 8 holders pulled
//!   from (2f (a)), and a relay op multicasting the row whole from one gatherer (2f (b));
//! - **two conditions:** the card otherwise idle, and under a GDDR stream on every unit the
//!   delivery leaves idle (`probe_background_stream`), what a weight stream is to the small ops.
//!
//! Each trace holds `REPS` deliveries. Each delivery is checked first, eagerly: every reader's twin
//! equals the producers' rows, byte for byte.
//!
//! Prints a `ROWDELIVERY` line per case: the producers' op, the relay op and the readers' op, in
//! microseconds, and their sum.
#![cfg(all(feature = "silicon", feature = "profile"))]

use tt_device::tlb::WindowKind;
use tt_isa::dm::{TILE_DATA, TILE_SLOT};
use tt_kernels::profile_mode::ProfileOptions;
use tt_kernels::residency::RowScheme;
use tt_kernels::session::{
    probe_background_stream, ProbeBackground, RowLayout, Session, TileChoice,
};
use tt_kernels::tensor::DramTensor;
use tt_tests::backend::device_index;
use tt_ttsim::fork_scope;

type Card = Session<tt_kmd::Kmd>;

const WARMUPS: usize = 2;
const SAMPLES: usize = 9;
/// Deliveries in one trace.
const REPS: usize = 8;
/// Tiles each idle unit streams a replay: long enough to outlast the slowest trace (~2.4 ms of
/// multicast at ~5 GB/s a unit).
const BACKGROUND_TILES: usize = 4096;
const POISON: u8 = 0xA5;
const FACE: usize = 1024;

const SCHEMES: [RowScheme; 8] = [
    RowScheme::Gddr,
    RowScheme::Multicast,
    RowScheme::MulticastRunAcks,
    RowScheme::Held(1),
    RowScheme::Held(2),
    RowScheme::Held(4),
    RowScheme::Held(8),
    RowScheme::Relay,
];

fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as u8
        })
        .collect()
}

fn src_for(layout: RowLayout, rows: usize, producers: usize) -> u64 {
    let most = tt_kernels::tensor::runs(rows, producers, usize::MAX)
        .iter()
        .map(|r| r.len())
        .max()
        .unwrap() as u64;
    (layout.dst - most * TILE_SLOT) & !63
}

/// Deliver once, eagerly, and check every reader's twin against the producers' rows.
fn check(s: &mut Card, scheme: RowScheme, rows: usize, p: usize, r: usize, twin: &DramTensor) {
    let tiles = s.tiles();
    let w = s.device().alloc_window(WindowKind::FourGib).unwrap();
    let layout = s
        .probe_row_delivery(RowScheme::Gddr, 0..p, 0..r, twin)
        .unwrap();
    for &t in &tiles {
        let fill = vec![POISON; (tt_isa::l1::DATA.end - layout.dst) as usize];
        s.device().l1_write(&w, t, layout.dst, &fill).unwrap();
    }
    let src = src_for(layout, rows, p);
    let data = 4 * FACE;
    let mut want = vec![0u8; rows / 32 * data];
    for (k, run) in tt_kernels::tensor::runs(rows, p, usize::MAX)
        .into_iter()
        .enumerate()
    {
        let bytes = pattern(run.len() * TILE_SLOT as usize, 0x2f + k as u32 * 31);
        s.device().l1_write(&w, tiles[k], src, &bytes).unwrap();
        for (m, kk) in run.enumerate() {
            let row = kk % 32;
            for half in 0..2 {
                let from = m * TILE_SLOT as usize + TILE_DATA as usize + half * FACE;
                let to = kk / 32 * data + (2 * (row / 16) + half) * FACE + (row % 16) * 64;
                want[to..to + 64].copy_from_slice(&bytes[from..from + 64]);
            }
        }
    }
    s.probe_row_delivery(scheme, 0..p, 0..r, twin).unwrap();
    s.sync().unwrap();
    for (u, &t) in tiles.iter().enumerate().take(r) {
        let mut got = vec![0u8; layout.bytes as usize];
        s.device().l1_read(&w, t, layout.dst, &mut got).unwrap();
        for k in 0..rows / 32 {
            let slot = &got[k * TILE_SLOT as usize + TILE_DATA as usize..][..data];
            assert!(
                slot == &want[k * data..(k + 1) * data],
                "{scheme:?}, {rows} rows to {r} readers: reader {u}'s twin tile {k} differs"
            );
        }
    }
}

/// The medians of each kind of op over a trace's replays, in microseconds: the producers', the
/// relay's and the readers'.
fn spans(s: &mut Card, scheme: RowScheme, p: usize, r: usize, twin: &DramTensor) -> [f64; 3] {
    s.begin_trace().unwrap();
    for _ in 0..REPS {
        s.probe_row_delivery(scheme, 0..p, 0..r, twin).unwrap();
    }
    let id = s.end_trace().unwrap();
    for _ in 0..WARMUPS {
        s.replay(id).unwrap();
    }
    s.sync().unwrap();
    s.profile_mode(Some(ProfileOptions::default())).unwrap();
    for _ in 0..SAMPLES {
        s.replay(id).unwrap();
        s.sync().unwrap();
    }
    let report = s.profile_mode(None).unwrap().expect("the mode was on");
    s.release_trace(id).unwrap();
    let t = &report.traces[&id.raw()];
    let mut by = [Vec::new(), Vec::new(), Vec::new()];
    for (op, samples) in t.ops.iter().zip(&t.samples) {
        let k = match op.what {
            "probe row writes" => 0,
            "probe row relay" => 1,
            "probe row reads" => 2,
            other => panic!("an op of the probe's named {other}"),
        };
        by[k].extend(samples.iter().map(|o| o.span as f64 / report.ticks_per_us));
    }
    by.map(|mut v| {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    })
}

#[test]
#[ignore = "benchmark"]
fn row_delivery_schemes() {
    let card = device_index();
    fork_scope(|| {
        let mut s = Card::open_card(card, tt_firmware_images::ROLES, TileChoice::All)
            .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_residency(0);
        s.set_l1_region(0);
        let n = s.tiles().len();
        println!(
            "BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"tiles\":{n},\"warmups\":{WARMUPS},\"samples\":{SAMPLES},\"reps\":{REPS},\"background_tiles\":{BACKGROUND_TILES},\"timing\":\"profile mode, per-op span medians\"}}",
            tt_tests::bench::git_sha(),
            !cfg!(debug_assertions),
        );
        let cases: Vec<(usize, usize, usize)> = [(64, 64, 32), (64, 64, 64), (64, 64, 104), (256, 64, 64)]
            .into_iter()
            .filter(|&(_, _, r)| r <= n)
            .collect();
        for (rows, p, r) in cases {
            let twin = s.upload(&vec![0.0; rows * 32], rows, 32).unwrap();
            for scheme in SCHEMES {
                check(&mut s, scheme, rows, p, r, &twin);
                for stream in [false, true] {
                    // The stream: every unit the delivery leaves idle.
                    let idle = n - p.max(r);
                    if stream && idle == 0 {
                        continue;
                    }
                    probe_background_stream(stream.then_some(ProbeBackground {
                        first: p.max(r),
                        units: idle,
                        tiles_per_unit: BACKGROUND_TILES,
                    }));
                    let [write, relay, read] = spans(&mut s, scheme, p, r, &twin);
                    probe_background_stream(None);
                    println!(
                        "ROWDELIVERY {{\"rows\":{rows},\"producers\":{p},\"readers\":{r},\"scheme\":\"{scheme:?}\",\"stream_units\":{},\"write_us\":{write:.2},\"relay_us\":{relay:.2},\"read_us\":{read:.2},\"total_us\":{:.2}}}",
                        if stream { idle } else { 0 },
                        write + relay + read
                    );
                }
            }
            s.free(twin).unwrap();
        }
    })
    .unwrap_or_else(|e| panic!("{e}"));
}
