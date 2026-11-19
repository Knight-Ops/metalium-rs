//! A row scatter into a block-float tensor (`docs/plans/block-float-kv.md` K1): a BFP8 or BFP4 KV
//! cache written in place, a decode step's one position or a chunk's rows at a time.
//!
//! `record::SCATTER_ROWS` with a block-float format writes each staged row's two face-rows of
//! datums and their two exponent bytes to the row its resident index names. A row of a
//! block-float tile shares nothing with the tile's other rows, so **the cache written row by row
//! in its format is, byte for byte, the F32 cache written the same way and then cast**
//! (`Session::bfp_from_f32`). That is the claim, on every exponent and datum byte of every tile:
//! - decode steps at every position of two tile rows (each residue mod 32: a staged row 0 lands
//!   at every offset, so exponents are written straight and through their aside);
//! - chunks: 40 rows across a tile's edge from an unaligned start, 32 rows tile-aligned;
//! - repeated indices (the last one's row is written);
//! - one unit, and four (a batch's indices split between units).
//!
//! **Control:** every row written changes the cache's image (the values are of another
//! magnitude than the cache's, so their exponents differ): equal images say the rows were
//! written, not that nothing moved.

use tt_kernels::bfp::BfpFormat;
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Elem};
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_units(units: usize, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(units),
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
fn with_units(units: usize, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(units),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// `n` values in about `[-scale, scale]`, a few zeros among them.
fn values(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if i % 41 == 7 {
                0.0
            } else {
                ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            }
        })
        .collect()
}

const ROWS: usize = 128;
const COLS: usize = 96;

/// The writes: each a list of positions, one row of values each.
fn writes() -> Vec<Vec<u32>> {
    let mut w: Vec<Vec<u32>> = (0..64).map(|p| vec![p]).collect();
    w.push((7..47).collect());
    w.push((64..96).collect());
    w.push(vec![100, 9, 100, 33, 9, 127]);
    w
}

/// Every tile's image past its header: the exponent section and the datums.
fn image<T: tt_device::Transport>(s: &mut Session<T>, t: &DramTensor) -> Vec<Vec<u8>> {
    let b = t.as_bfp().expect("a block-float tensor");
    s.download_bfp_raw(&b)
        .unwrap()
        .into_iter()
        .map(|tile| tile[16..].to_vec())
        .collect()
}

fn check(units: usize, format: BfpFormat) {
    with_units(units, |s| {
        let start = values(1, ROWS * COLS, 2.0);
        let f32_cache = s.upload(&start, ROWS, COLS).unwrap();
        let bfp = |s: &mut Session<_>, t: &DramTensor| {
            DramTensor::from_bfp(s.bfp_from_f32(t, format).unwrap())
        };
        let cache = bfp(s, &f32_cache);
        let before = image(s, &cache);
        for (k, positions) in writes().iter().enumerate() {
            let n = positions.len();
            // Another magnitude than the cache's: each written row's exponents change.
            let rows = values(100 + k as u64, n * COLS, 300.0);
            let dv = s.upload(&rows, n, COLS).unwrap();
            let bits: Vec<u32> = positions.clone();
            let di = s.upload_bits(&bits, 1, n, Elem::I32).unwrap();
            s.scatter_rows_into(&f32_cache, &di, &dv, false).unwrap();
            let dv_bfp = bfp(s, &dv);
            s.scatter_rows_into(&cache, &di, &dv_bfp, false).unwrap();
        }
        let got = image(s, &cache);
        let want_t = bfp(s, &f32_cache);
        let want = image(s, &want_t);
        let differing: Vec<usize> = (0..got.len()).filter(|&t| got[t] != want[t]).collect();
        if let Some(&t) = differing.first() {
            let at = (0..got[t].len())
                .find(|&b| got[t][b] != want[t][b])
                .unwrap();
            panic!(
                "{format:?} on {units} units: {} of {} tiles differ from the F32 cache cast; tile \
                 {t} first at byte {} past the header (got {:#04x}, want {:#04x})",
                differing.len(),
                got.len(),
                at,
                got[t][at],
                want[t][at]
            );
        }
        // Control: every tile row was written, so every tile changed.
        let unchanged = (0..got.len()).filter(|&t| got[t] == before[t]).count();
        assert_eq!(
            unchanged, 0,
            "{format:?}: {unchanged} tiles unchanged by the writes"
        );
        println!(
            "{format:?} on {units} units: {} tiles equal to the F32 cache cast, every one written",
            got.len()
        );
    });
}

#[test]
fn a_bfp8_cache_written_by_rows_is_the_f32_cache_cast() {
    check(1, BfpFormat::Bfp8);
}

#[test]
fn a_bfp4_cache_written_by_rows_is_the_f32_cache_cast() {
    check(1, BfpFormat::Bfp4);
}

#[test]
fn rows_split_between_units_write_the_same_bytes() {
    check(4, BfpFormat::Bfp8);
    check(4, BfpFormat::Bfp4);
}
