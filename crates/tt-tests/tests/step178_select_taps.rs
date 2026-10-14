//! Multi-tap row select: `Session::select_taps`, the im2col of a chunk-free
//! convolution (roadmap Track 2 item 2).
//!
//! A convolution over an NHWC table `[N*H*W, C]` is one product once its patch
//! matrix `[P, taps x C]` exists: row `p` holds, for each tap `t`, the table's
//! row under that tap. `select_taps` builds it from a resident `[taps, P]` I32
//! index with B's `SELECT_ROWS` record (one per tap and output tile row), so no
//! host address is built for a datum; tap `t`'s index is row `t % 32` of its
//! index tile, reached by moving the record's `idx_at` to that row.
//!
//! The claims, each against the host indexing a `Vec` (a select is a raw copy:
//! bit for bit, never a tolerance):
//! - every datum comes through unchanged (signed zero, subnormals, NaN
//!   payloads, infinities, `f32::MAX`), for 1, 2, 9, 25 and 49 taps: tap counts
//!   that split a batch's 24 staged tiles, and that cross the 32 taps of an
//!   index tile row inside a batch;
//! - a ragged table width puts each tap at a tile column, with zeros between;
//! - the zero row: an index at `table.rows` reads zeros, from the first
//!   padding row (also when the padding held something else) or, for a table
//!   of whole tile rows, from a copy a row longer;
//! - an index past the bound is `DOMAIN` (code 11) from the mover.
//!
//! Watched failing (2026-10-14, ttsim): with tap `t`'s `idx_at` left at row 0
//! of its index tile (every tap reads tap 0's indices, or tap 32's), and with
//! the second index tile read into the first's slot.

use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{Elem, Eltwise, Pad};
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

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
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// `n` datums of raw bits: every fifth one of a pattern a copy must not touch,
/// the rest anything at all.
fn raw_bits(seed: u64, n: usize) -> Vec<u32> {
    const SPECIAL: [u32; 8] = [
        0x8000_0000, // -0.0
        0x0000_0001, // the least subnormal
        0x7fc1_2345, // a quiet NaN with a payload
        0xff81_0000, // a negative signalling NaN
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7f7f_ffff, // f32::MAX
        0x007f_ffff, // the greatest subnormal
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if i % 5 == 0 {
                SPECIAL[(i / 5) % SPECIAL.len()]
            } else {
                (s >> 32) as u32
            }
        })
        .collect()
}

/// `[taps, n]` indices into `bound` rows, row-major: each tap its own
/// scramble, with the first and last row and repeats.
fn indices(bound: usize, taps: usize, n: usize) -> Vec<u32> {
    (0..taps * n)
        .map(|k| {
            let (t, i) = (k / n, k % n);
            match i {
                0 => 0,
                1 => bound as u32 - 1,
                _ if i % 6 == 5 => (t % bound) as u32,
                _ => ((i * 37 + t * 101 + 11) % bound) as u32,
            }
        })
        .collect()
}

/// The host's patch matrix: row `i` is tap `t`'s table row from column
/// `t x ct x 32`, zeros between taps; an index at `rows` is the zero row.
fn expected(table: &[u32], rows: usize, cols: usize, idx: &[u32], taps: usize) -> Vec<u32> {
    let n = idx.len() / taps;
    let stride = cols.div_ceil(32) * 32;
    let width = (taps - 1) * stride + cols;
    let mut out = vec![0u32; n * width];
    for t in 0..taps {
        for i in 0..n {
            let r = idx[t * n + i] as usize;
            if r < rows {
                out[i * width + t * stride..][..cols]
                    .copy_from_slice(&table[r * cols..(r + 1) * cols]);
            }
        }
    }
    out
}

fn assert_bits(got: &[u32], want: &[u32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g, w, "{what}: datum {i}: {g:#010x} vs {w:#010x}");
    }
}

#[test]
fn tapped_rows_are_the_host_s_indexing_bit_for_bit() {
    with_session(|s| {
        // (rows, cols, taps, n): one tap and two; a 3x3 window's 9 over two
        // tile columns (18 tiles a row, one batch); 25 over one column (a
        // batch of 24 and a batch of 1); 49 (a 7x7 window) over a ragged
        // width of 3, whose second batch spans taps 24..48, two index tile
        // rows; and 9 over 25 tile columns, which a batch's staging splits
        // mid-tap.
        for (rows, cols, taps, n) in [
            (100, 70, 1, 33),
            (64, 64, 2, 70),
            (97, 64, 9, 45),
            (40, 32, 25, 33),
            (150, 3, 49, 70),
            (33, 800, 9, 5),
        ] {
            let bits = raw_bits((rows * 1000 + cols * 10 + taps) as u64, rows * cols);
            let table = s.upload_bits(&bits, rows, cols, Elem::F32).unwrap();
            let idx = indices(rows, taps, n);
            let di = s.upload_bits(&idx, taps, n, Elem::I32).unwrap();
            let out = s.select_taps(&table, &di, false).unwrap();
            let what = format!("{taps} taps of {n} rows of [{rows}, {cols}]");
            let width = (taps - 1) * cols.div_ceil(32) * 32 + cols;
            assert_eq!([out.rows, out.cols], [n, width], "{what}");
            assert_eq!(out.pad(), Pad::Zero, "{what}");
            assert_bits(
                &s.download_bits(&out).unwrap(),
                &expected(&bits, rows, cols, &idx, taps),
                &what,
            );
            // The table is unchanged.
            assert_bits(&s.download_bits(&table).unwrap(), &bits, "the table");
            for t in [out, di, table] {
                s.free(t).unwrap();
            }
        }
    });
}

/// A padding tap names `table.rows` and reads zeros: the first padding row of
/// an uploaded table, of one whose padding held the bias of an `x + b`
/// (`Pad::Undefined`), which the select makes zero before it reads, and of one
/// with whole tile rows (64), which has no padding row and is first copied
/// into a zeroed one a row longer.
#[test]
fn the_zero_row_is_the_first_padding_row_made_zero() {
    with_session(|s| {
        for rows in [50, 64] {
            zero_row_case(s, rows);
        }
    });
}

fn zero_row_case<T: tt_device::Transport>(s: &mut Session<T>, rows: usize) {
    {
        let (cols, taps, n) = (40, 9, 37);
        let xv: Vec<f32> = (0..rows * cols).map(|i| (i % 23) as f32 - 11.0).collect();
        let x = s.upload(&xv, rows, cols).unwrap();
        let b = s.upload(&vec![1.0; cols], 1, cols).unwrap();
        let add_row = Eltwise {
            scalar2: 0.0,
            kind: tt_kernels::kind::ADD_ROW,
            scalar: 0.0,
        };
        let dirty = s.eltwise(add_row, &x, Some(&b)).unwrap();
        if rows % 32 != 0 {
            assert_eq!(
                dirty.pad(),
                Pad::Undefined,
                "the bias is in the padding rows"
            );
        }
        let dirty_bits: Vec<u32> = xv.iter().map(|v| (v + 1.0).to_bits()).collect();
        let clean_bits: Vec<u32> = xv.iter().map(|v| v.to_bits()).collect();
        // Every fourth index is the zero row.
        let idx: Vec<u32> = indices(rows, taps, n)
            .into_iter()
            .enumerate()
            .map(|(k, v)| if k % 4 == 3 { rows as u32 } else { v })
            .collect();
        let di = s.upload_bits(&idx, taps, n, Elem::I32).unwrap();
        for (table, bits, what) in [
            (&x, &clean_bits, "uploaded"),
            (&dirty, &dirty_bits, "x + b"),
        ] {
            // Whatever the select allocates next reuses slots that held NaNs:
            // a zero row that was never written is not zero by luck.
            let nan = s
                .upload_bits(
                    &vec![0x7fc1_2345; (rows + 1) * cols],
                    rows + 1,
                    cols,
                    Elem::F32,
                )
                .unwrap();
            s.free(nan).unwrap();
            s.sync().unwrap();
            let out = s.select_taps(table, &di, true).unwrap();
            assert_bits(
                &s.download_bits(&out).unwrap(),
                &expected(bits, rows, cols, &idx, taps),
                &format!("the zero row of a table {what}"),
            );
            s.free(out).unwrap();
        }
    }
}

/// Negative controls: past the bound is `DOMAIN` (code 11) -- `table.rows`
/// without the zero row, `table.rows + 1` with it (also for a table of whole
/// tile rows, whose zero row is a copy's), and a negative index.
#[test]
fn an_index_past_the_bound_is_domain() {
    for (rows, invalid, zero_row) in [
        (50, 50u32, false),
        (50, 51, true),
        (50, u32::MAX, true),
        (64, 65, true),
    ] {
        with_session(|s| {
            let bits = raw_bits(5, rows * 40);
            let table = s.upload_bits(&bits, rows, 40, Elem::F32).unwrap();
            let mut idx = indices(rows, 9, 20);
            let valid = s.upload_bits(&idx, 9, 20, Elem::I32).unwrap();
            let out = s.select_taps(&table, &valid, zero_row).unwrap();
            assert_bits(
                &s.download_bits(&out).unwrap(),
                &expected(&bits, rows, 40, &idx, 9),
                "the valid indices",
            );
            // In the last tap, past the first index tile's first rows.
            idx[8 * 20 + 7] = invalid;
            let bad = s.upload_bits(&idx, 9, 20, Elem::I32).unwrap();
            let _pending = s.select_taps(&table, &bad, zero_row).unwrap();
            let err = s.sync().unwrap_err();
            assert!(
                err.to_string().contains("code 11"),
                "index {invalid:#x} of {rows} rows (zero row {zero_row}) DOMAIN: {err}"
            );
        });
    }
}
