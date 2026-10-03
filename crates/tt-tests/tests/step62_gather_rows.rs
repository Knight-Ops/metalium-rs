//! D4 gate: rows gathered on the card -- an embedding's lookup.
//!
//! `Session::gather_rows` builds each output tile from 64-byte face-row reads
//! (any source row into any output row: every face-row is congruent to
//! `TILE_DATA` mod 64) and writes it out after a `WAIT`. The claim: bit for
//! bit the host's row selection -- repeated rows, rows from two sources,
//! ragged widths, a row count off the tile grid -- and a row that does not
//! exist refused before anything runs. Watched failing with the face halves
//! swapped.

use tt_kernels::session::{Session, TileChoice};
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

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

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
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
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[test]
fn gathered_rows_are_the_host_s_selection() {
    with_session(|s| {
        for (vocab, cols, n) in [(64, 64, 128), (100, 70, 37), (40, 32, 96)] {
            let w = floats(vocab as u64, vocab * cols);
            let z = floats(1, 3 * cols);
            let dw = s.upload(&w, vocab, cols).unwrap();
            let dz = s.upload(&z, 3, cols).unwrap();
            // Rows of `w` in a scrambled order with repeats, every 7th from `z`.
            let rows: Vec<(usize, usize)> = (0..n)
                .map(|i| {
                    if i % 7 == 6 {
                        (1, i % 3)
                    } else {
                        (0, (i * 37 + 11) % vocab)
                    }
                })
                .collect();
            let got = s.gather_rows(&[&dw, &dz], &rows, cols).unwrap();
            assert_eq!([got.rows, got.cols], [n, cols]);
            let want: Vec<f32> = rows
                .iter()
                .flat_map(|&(k, r)| {
                    let src = if k == 0 { &w } else { &z };
                    src[r * cols..(r + 1) * cols].to_vec()
                })
                .collect();
            assert_bits(
                &s.download(&got).unwrap(),
                &want,
                &format!("{n} rows of [{vocab}, {cols}]"),
            );
        }
        let w = floats(9, 64 * 32);
        let dw = s.upload(&w, 64, 32).unwrap();
        let e = s.gather_rows(&[&dw], &[(0, 3), (0, 64)], 32).unwrap_err();
        assert!(e.to_string().contains("which has none"), "{e}");
    });
}

/// `Session::rows_add` -- an embedding's gradient: `value`'s rows added to
/// the rows the indices name, each row's additions in index order -- bit for
/// bit the host's sequential `+=`, with rows repeated up to five times,
/// untouched rows kept to the bit, and ragged widths. Watched failing with
/// the rounds' occurrences reversed.
#[test]
fn rows_added_by_index_are_the_host_s_sequential_sums() {
    with_session(|s| {
        for (vocab, cols, n) in [(64, 64, 128), (100, 70, 37)] {
            let t = floats(vocab as u64 + 1, vocab * cols);
            let v = floats(n as u64 + 2, n * cols);
            // Small indices repeat: row 3 appears every fifth index.
            let idx: Vec<usize> = (0..n)
                .map(|i| if i % 5 == 0 { 3 } else { (i * 13 + 7) % vocab })
                .collect();
            let dt = s.upload(&t, vocab, cols).unwrap();
            let dv = s.upload(&v, n, cols).unwrap();
            let got = s.rows_add(&dt, &idx, &dv).unwrap();
            let mut want = t.clone();
            for (i, &r) in idx.iter().enumerate() {
                for c in 0..cols {
                    want[r * cols + c] += v[i * cols + c];
                }
            }
            assert_bits(
                &s.download(&got).unwrap(),
                &want,
                &format!("{n} rows into [{vocab}, {cols}]"),
            );
            // The input is untouched.
            assert_bits(&s.download(&dt).unwrap(), &t, "the input");
        }
        let t = floats(5, 64 * 32);
        let dt = s.upload(&t, 64, 32).unwrap();
        let src = s.upload(&t[..32 * 32], 32, 32).unwrap();
        let e = s.write_rows(&dt, &src, &[(1, 0), (1, 2)]).unwrap_err();
        assert!(e.to_string().contains("twice"), "{e}");
    });
}
