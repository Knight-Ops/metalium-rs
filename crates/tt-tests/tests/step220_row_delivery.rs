//! 2f of `docs/plans/planned-l1-residency.md`: a row many units read whole, delivered by each of
//! the candidate schemes (`tt_kernels::residency::RowScheme`, `Session::probe_row_delivery`),
//! lands whole on every reader. `silicon_probe_row_delivery` times them; this gate holds them to
//! their bytes.
//!
//! The claims, on 8 units, for a 64-row twin (a `[1, 2048]` row packed) dealt over the producers
//! as an element-wise op deals its output tiles, two face rows a tile:
//! - **Every reader ends with the twin** (each tile's 1,024 datums; the header and padding no
//!   scheme writes), in every scheme -- through GDDR, Phase 2's multicast from
//!   every producer, the same with each producer's acknowledgements counted once, 1, 2 and 8 holders pulled from, and a relay op multicasting from one gatherer
//!   -- with the readers the producers, a subset of them, and units that produce nothing.
//! - **A replay delivers again:** captured once, replayed over new source rows with every reader's
//!   slots and every unit's region poisoned, each reader ends with the new twin.
//! - **The negative control:** with each holder's readers pointed at the reader after it
//!   (`probe_row_misroute`), a reader that is no holder reads poison, and the check sees it.

use tt_device::tlb::WindowKind;
use tt_isa::dm::{TILE_DATA, TILE_SLOT};
use tt_kernels::residency::RowScheme;
use tt_kernels::session::{probe_row_misroute, RowLayout, Session, TileChoice};
use tt_kernels::tensor::DramTensor;
use tt_ttsim::fork_scope;

/// The twin's rows: a `[1, 2048]` row's 64 output tiles.
const ROWS: usize = 64;
const POISON: u8 = 0xA5;
const FACE: usize = 1024;

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(8),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        // The probe's row takes the top of `DATA`: no region kept, whatever `TT_RESIDENCY` says.
        s.set_residency(0);
        s.set_l1_region(0);
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
            TileChoice::Count(8),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        // The probe's row takes the top of `DATA`: no region kept, whatever `TT_RESIDENCY` says.
        s.set_residency(0);
        s.set_l1_region(0);
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

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

/// Each producer's output tiles, as `tensor::runs` deals them.
fn runs(producers: usize) -> Vec<std::ops::Range<usize>> {
    tt_kernels::tensor::runs(ROWS, producers, usize::MAX)
}

/// Fill each producer's source slots with its own pattern; return the twin's data bytes (each
/// twin tile's 1,024 datums from `TILE_DATA`, in tile order) those slots make.
fn sources<T: tt_device::Transport>(
    s: &mut Session<T>,
    producers: std::ops::Range<usize>,
    layout: RowLayout,
    seed: u32,
) -> Vec<u8> {
    let tiles = s.tiles();
    let w = s.device().alloc_window(WindowKind::FourGib).unwrap();
    let data = 4 * FACE;
    let mut twin = vec![0u8; ROWS / 32 * data];
    for (k, run) in runs(producers.len()).into_iter().enumerate() {
        let u = producers.start + k;
        let src = pattern(run.len() * TILE_SLOT as usize, seed + u as u32 * 31);
        s.device()
            .l1_write(&w, tiles[u], src_for(layout, producers.len()), &src)
            .unwrap();
        for (m, kk) in run.enumerate() {
            let r = kk % 32;
            for half in 0..2 {
                let from = m * TILE_SLOT as usize + TILE_DATA as usize + half * FACE;
                let to = kk / 32 * data + (2 * (r / 16) + half) * FACE + (r % 16) * 64;
                twin[to..to + 64].copy_from_slice(&src[from..from + 64]);
            }
        }
    }
    twin
}

/// Where the probe puts the producers' source slots: below the twin's, room for the most tiles
/// a producer holds.
fn src_for(layout: RowLayout, producers: usize) -> u64 {
    let most = runs(producers).iter().map(|r| r.len()).max().unwrap() as u64;
    (layout.dst - most * TILE_SLOT) & !63
}

/// Poison every reader's twin slots and every unit's region.
fn poison<T: tt_device::Transport>(s: &mut Session<T>, layout: RowLayout) {
    let w = s.device().alloc_window(WindowKind::FourGib).unwrap();
    for t in s.tiles() {
        let fill = vec![POISON; (tt_isa::l1::DATA.end - layout.dst) as usize];
        s.device().l1_write(&w, t, layout.dst, &fill).unwrap();
    }
}

/// The first reader whose twin differs from `want`, and where.
fn first_wrong<T: tt_device::Transport>(
    s: &mut Session<T>,
    readers: std::ops::Range<usize>,
    layout: RowLayout,
    want: &[u8],
) -> Option<(usize, usize)> {
    let tiles = s.tiles();
    let w = s.device().alloc_window(WindowKind::FourGib).unwrap();
    let data = 4 * FACE;
    for u in readers {
        let mut got = vec![0u8; layout.bytes as usize];
        s.device()
            .l1_read(&w, tiles[u], layout.dst, &mut got)
            .unwrap();
        for t in 0..ROWS / 32 {
            let slot = &got[t * TILE_SLOT as usize + TILE_DATA as usize..][..data];
            if let Some(b) = slot
                .iter()
                .zip(&want[t * data..(t + 1) * data])
                .position(|(a, b)| a != b)
            {
                return Some((u, t * data + b));
            }
        }
    }
    None
}

const SCHEMES: [RowScheme; 7] = [
    RowScheme::Gddr,
    RowScheme::Multicast,
    RowScheme::MulticastRunAcks,
    RowScheme::Held(1),
    RowScheme::Held(2),
    RowScheme::Held(8),
    RowScheme::Relay,
];

fn twin<T: tt_device::Transport>(s: &mut Session<T>) -> DramTensor {
    s.upload(&vec![0.0; ROWS * 32], ROWS, 32).unwrap()
}

/// The schemes' layouts are the same: the probe's addresses depend only on the row and the
/// producers.
fn layout_of<T: tt_device::Transport>(s: &mut Session<T>, twin: &DramTensor) -> RowLayout {
    s.probe_row_delivery(RowScheme::Gddr, 0..8, 0..8, twin)
        .unwrap()
}

#[test]
fn every_reader_ends_with_the_twin_in_every_scheme() {
    with_session(|s| {
        let twin = twin(s);
        let layout = layout_of(s, &twin);
        for (producers, readers) in [(0..8, 0..8), (0..8, 2..7), (0..3, 3..8)] {
            for (n, scheme) in SCHEMES.into_iter().enumerate() {
                poison(s, layout);
                let want = sources(s, producers.clone(), layout, 0x220 + n as u32);
                let got = s
                    .probe_row_delivery(scheme, producers.clone(), readers.clone(), &twin)
                    .unwrap();
                assert_eq!(
                    (got.region, got.dst, got.src),
                    (layout.region, layout.dst, src_for(layout, producers.len())),
                );
                s.sync().unwrap();
                assert_eq!(
                    first_wrong(s, readers.clone(), got, &want),
                    None,
                    "{scheme:?}, producers {producers:?}, readers {readers:?}: (reader, byte)"
                );
            }
        }
    });
}

#[test]
fn a_replay_delivers_again() {
    with_session(|s| {
        let twin = twin(s);
        let (producers, readers) = (0..8, 1..8);
        for (n, scheme) in SCHEMES.into_iter().enumerate() {
            s.begin_trace().unwrap();
            let layout = s
                .probe_row_delivery(scheme, producers.clone(), readers.clone(), &twin)
                .unwrap();
            let id = s.end_trace().unwrap();
            for round in 0..2u32 {
                poison(s, layout);
                let want = sources(s, producers.clone(), layout, 0x2200 + n as u32 * 8 + round);
                s.replay(id).unwrap();
                s.sync().unwrap();
                assert_eq!(
                    first_wrong(s, readers.clone(), layout, &want),
                    None,
                    "{scheme:?}, replay {round}: (reader, byte)"
                );
            }
            s.release_trace(id).unwrap();
        }
    });
}

#[test]
fn a_reader_pointed_at_no_holder_reads_poison() {
    with_session(|s| {
        let twin = twin(s);
        let layout = layout_of(s, &twin);
        poison(s, layout);
        let want = sources(s, 0..8, layout, 0x2201);
        probe_row_misroute(true);
        let r = s.probe_row_delivery(RowScheme::Held(2), 0..8, 0..8, &twin);
        probe_row_misroute(false);
        r.unwrap();
        s.sync().unwrap();
        // Holders 2 and 6 (`holders`' middles of 0..4 and 4..8): their readers read 3 and 7,
        // which hold nothing.
        let wrong = first_wrong(s, 0..8, layout, &want);
        assert_eq!(wrong.map(|(u, _)| u), Some(0), "{wrong:?}");
        // And the same probe routed as it should be delivers.
        poison(s, layout);
        let want = sources(s, 0..8, layout, 0x2202);
        s.probe_row_delivery(RowScheme::Held(2), 0..8, 0..8, &twin)
            .unwrap();
        s.sync().unwrap();
        assert_eq!(first_wrong(s, 0..8, layout, &want), None);
    });
}
