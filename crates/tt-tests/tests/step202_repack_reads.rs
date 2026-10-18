//! A native repack's reads in flight together (`docs/plans/decode-to-bandwidth.md` Phase 1):
//! `tensor::repack_many` reads up to `REPACK_SLOTS` source tiles into slots of their own before
//! it copies any of their words (the mover waits for the preceding reads before a `COPY_WORDS`),
//! where it read each into one slot and copied it before reading the next; and each read is only
//! the 64-byte span of the slot that holds the words copied from it (a row's two face rows,
//! 1,152 bytes of 4,160). A decode step's key and value rows, each head's row a column block of
//! the projection's one row, took 38 us a repack on two units, 32 whole-tile reads one after the
//! other.
//!
//! The claim: every datum the mapping names is its source's bits, on data with every special
//! value, for
//! - **the decode repack:** 32 heads' rows of 64 from a `[1, 2048]` row into `[32, 64]`, and 9
//!   heads (135M's `[1, 576]`);
//! - **a transpose:** each tile's 1,024 words scattered one by one, so a tile's copies span
//!   batches and the tile is read again for each part;
//! - **more source tiles than slots:** an output tile drawn from 48 source tiles, so it takes two
//!   batches and the second reads the first's output back before it copies;
//! - **two sources:** rows of one tensor and columns of another in one output.
//!
//! Before each repack, a repack over NaN fills the slots, so a copy from the wrong slot shows.
//! Watched failing with each tile's copies reading slot 0, and with each read's span 64 bytes
//! short.

use tt_kernels::session::{Session, TileChoice};
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        use tt_tests::backend::GATE_TILE;
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let tiles = TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1);
        let mut s = Session::open(dev, tt_firmware_images::ROLES, tiles, |_, _| Ok(None)).unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
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
            TileChoice::Count(4),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Distinct words, with every special value among them (moved as raw bits).
fn words(seed: u32, n: usize) -> Vec<f32> {
    const SPECIAL: [u32; 8] = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x807f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0x7f7f_ffff,
    ];
    (0..n)
        .map(|i| {
            if i % 29 == 3 {
                f32::from_bits(SPECIAL[(i / 29) % SPECIAL.len()])
            } else {
                f32::from_bits(0x3f80_0000 ^ (seed << 20) ^ i as u32)
            }
        })
        .collect()
}

/// `inputs` (each `(values, rows, cols)`) repacked by `sources` into `[rows, cols]`, after a
/// repack of NaN through the same mapping primes the unit's slots; checked datum by datum.
fn check<T: tt_device::Transport>(
    s: &mut Session<T>,
    what: &str,
    inputs: &[(Vec<f32>, usize, usize)],
    sources: &[(usize, [usize; 2])],
    [rows, cols]: [usize; 2],
) {
    let nan: Vec<_> = inputs
        .iter()
        .map(|(_, r, c)| s.upload(&vec![f32::NAN; r * c], *r, *c).unwrap())
        .collect();
    let primed = s
        .repack_many(&nan.iter().collect::<Vec<_>>(), sources, [rows, cols])
        .unwrap();
    let up: Vec<_> = inputs
        .iter()
        .map(|(v, r, c)| s.upload(v, *r, *c).unwrap())
        .collect();
    let out = s
        .repack_many(&up.iter().collect::<Vec<_>>(), sources, [rows, cols])
        .unwrap();
    let got = s.download(&out).unwrap();
    for (k, &(i, [r, c])) in sources.iter().enumerate() {
        let (v, _, vc) = &inputs[i];
        assert_eq!(
            got[k].to_bits(),
            v[r * vc + c].to_bits(),
            "{what}: datum {k} ({}, {}) from input {i} ({r}, {c})",
            k / cols,
            k % cols
        );
    }
    for t in nan.into_iter().chain(up).chain([primed, out]) {
        s.free(t).unwrap();
    }
}

#[test]
fn a_decode_rows_repack_is_its_sources_bit_for_bit() {
    with_session(|s| {
        for heads in [32usize, 9] {
            let d = 64;
            let row = words(heads as u32, heads * d);
            let sources: Vec<_> = (0..heads * d)
                .map(|k| (0, [0, (k / d) * d + k % d]))
                .collect();
            check(
                s,
                &format!("{heads} heads"),
                &[(row, 1, heads * d)],
                &sources,
                [heads, d],
            );
        }
    });
}

#[test]
fn a_transpose_and_a_wide_gather_are_their_sources_bit_for_bit() {
    with_session(|s| {
        // A transpose: every word its own copy, a tile's copies over several batches.
        let (r, c) = (64usize, 70usize);
        let x = words(1, r * c);
        let t: Vec<_> = (0..c * r).map(|k| (0, [k % r, k / r])).collect();
        check(s, "a transpose", &[(x, r, c)], &t, [c, r]);

        // One output tile from 48 source tiles: output row i is row 0 of tile column i, for 48
        // tile columns.
        let (r, c) = (1usize, 48 * 32);
        let x = words(2, r * c);
        let wide: Vec<_> = (0..48 * 32)
            .map(|k| (0, [0, (k / 32) * 32 + k % 32]))
            .collect();
        check(s, "48 source tiles", &[(x, r, c)], &wide, [48, 32]);

        // Two sources: the left half of each output row from `a`'s row, the right from `b`'s
        // column.
        let (a, b) = (words(3, 40 * 50), words(4, 50 * 40));
        let two: Vec<_> = (0..40 * 50)
            .map(|k| {
                let (i, j) = (k / 50, k % 50);
                if j < 25 {
                    (0, [i, j])
                } else {
                    (1, [j, i])
                }
            })
            .collect();
        check(
            s,
            "two sources",
            &[(a, 40, 50), (b, 50, 40)],
            &two,
            [40, 50],
        );
    });
}
