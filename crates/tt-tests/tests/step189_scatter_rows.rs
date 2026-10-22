//! Row scatter at resident indices: `Session::scatter_rows` and `burn-tt`'s
//! `scatter_nd` with `Assign`.
//!
//! A traced decode step wrote its key and value into a fixed cache by expanding
//! them to the cache's whole shape and a `mask_where`: a quarter of a
//! SmolLM2-1.7B step (roadmap Track 2, item 6, step 6; `decode-length-cost.md`
//! B3). `Session::scatter_rows` copies the table and NC's `SCATTER_ROWS` record
//! writes each staged row at the row a resident index names: two 64-byte
//! face-row writes a tile column, no mask over the table, no host address.
//!
//! The claims, each against an oracle that is the host writing rows of a `Vec`
//! in index order (a scatter is a raw copy: bit for bit, never a tolerance):
//! - every datum comes through unchanged (signed zero, subnormals, NaN payloads,
//!   infinities, `f32::MAX`; I32 and Bool tables), every row not named is the
//!   table's, and the table itself is not written;
//! - any index count against the 32-row tile (1, 31, 32, 33, 70), ragged rows
//!   and columns, a table wider than a batch's staging, indices as a row or a
//!   column of their tensor;
//! - a repeated index writes its last row, within a batch and across batches,
//!   on one tile and spread over several;
//! - an index at the row count, or negative, stops the mover with `DOMAIN`;
//! - through Burn, `scatter_nd(.., Assign)` is that one op (nothing
//!   downloaded, no mask loop), a trace replays it at the indices it finds,
//!   and a ragged result feeds reductions and matmuls exactly.
//!
//! Sixteen indices or fewer are read as their tile's face 0 alone (a decode step's one
//! position is 64 bytes, not the tile): 16 and 17 indices of each form are among the cases.
//!
//! Watched failing: with the tile copy skipped (the table is written), with the
//! column flag dropped (the wrong words read as indices), with a column of
//! sixteen indices read as 64 bytes (the rest stale), and, in the record's
//! unit tests (`tt_isa::dm::record`), with the face row off by one and with
//! the repeated-row check removed.

use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::Elem;
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

/// A session on `units` Tensix units (the scatter splits one batch of indices between them).
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

/// `n` indices into `rows` rows: the first and last row, a row repeated within
/// a batch and (past 32) across batches, and a scramble that crosses tile rows.
fn indices(rows: usize, n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| match i {
            0 => 0,
            1 => rows as u32 - 1,
            _ if i % 6 == 5 => 3 % rows as u32,
            _ => ((i * 37 + 11) % rows) as u32,
        })
        .collect()
}

/// The host's scatter: `table` with row `indices[i]` replaced by `values` row
/// `i`, in index order, so a repeated index's last row stays.
fn expected(table: &[u32], cols: usize, indices: &[u32], values: &[u32]) -> Vec<u32> {
    let mut out = table.to_vec();
    for (i, &r) in indices.iter().enumerate() {
        let r = r as usize;
        out[r * cols..(r + 1) * cols].copy_from_slice(&values[i * cols..(i + 1) * cols]);
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
fn scattered_rows_are_the_host_s_writes_bit_for_bit() {
    with_session(|s| {
        // (rows, cols): ragged both ways, exact tiles, and 25 tile columns,
        // which a batch's staging (24) splits in two.
        for (rows, cols) in [(100, 70), (64, 64), (33, 32), (40, 800), (97, 33)] {
            let bits = raw_bits(rows as u64 * 1000 + cols as u64, rows * cols);
            let table = s.upload_bits(&bits, rows, cols, Elem::F32).unwrap();
            for n in [1usize, 16, 17, 31, 32, 33, 70] {
                let idx = indices(rows, n);
                let values = raw_bits(n as u64 * 7 + 3, n * cols);
                let dv = s.upload_bits(&values, n, cols, Elem::F32).unwrap();
                let want = expected(&bits, cols, &idx, &values);
                for column in [false, true] {
                    let shape = if column { (n, 1) } else { (1, n) };
                    let di = s.upload_bits(&idx, shape.0, shape.1, Elem::I32).unwrap();
                    let out = s.scatter_rows(&table, &di, &dv, column).unwrap();
                    assert_eq!([out.rows, out.cols], [rows, cols]);
                    assert_bits(
                        &s.download_bits(&out).unwrap(),
                        &want,
                        &format!("{n} rows into [{rows}, {cols}], column {column}"),
                    );
                    s.free(out).unwrap();
                    s.free(di).unwrap();
                }
                s.free(dv).unwrap();
            }
            // The table is unchanged: the scatter wrote a copy.
            assert_bits(&s.download_bits(&table).unwrap(), &bits, "the table");
            s.free(table).unwrap();
        }
    });
}

#[test]
fn integer_and_bool_tables_scatter_as_raw_words() {
    with_session(|s| {
        let ints: Vec<u32> = (0..70 * 9)
            .map(|i| match i % 7 {
                0 => i32::MIN as u32,
                1 => i32::MAX as u32,
                2 => -1i32 as u32,
                3 => 0,
                _ => (i * 12345 - 99999) as u32,
            })
            .collect();
        let bools: Vec<u32> = (0..70 * 9).map(|i| u32::from(i % 3 == 0)).collect();
        for (bits, elem) in [(&ints, Elem::I32), (&bools, Elem::Bool)] {
            let table = s.upload_bits(bits, 70, 9, elem).unwrap();
            let idx = indices(70, 45);
            let values: Vec<u32> = (0..45 * 9).map(|i| bits[(i * 7) % bits.len()]).collect();
            let dv = s.upload_bits(&values, 45, 9, elem).unwrap();
            let di = s.upload_bits(&idx, 45, 1, Elem::I32).unwrap();
            let out = s.scatter_rows(&table, &di, &dv, true).unwrap();
            assert_eq!(out.elem, elem);
            assert_bits(
                &s.download_bits(&out).unwrap(),
                &expected(bits, 9, &idx, &values),
                &format!("{elem:?}"),
            );
        }
    });
}

/// In place (`scatter_rows_into`): only the named rows change, so a second
/// scatter over the first's result keeps both, the later one winning where
/// they meet.
#[test]
fn scattering_in_place_writes_only_the_named_rows() {
    with_session(|s| {
        let (rows, cols) = (96, 64);
        let bits = raw_bits(9, rows * cols);
        let t = s.upload_bits(&bits, rows, cols, Elem::F32).unwrap();
        let (a, b) = (vec![5u32, 40, 95], vec![40u32, 0]);
        let (va, vb) = (raw_bits(1, 3 * cols), raw_bits(2, 2 * cols));
        let da = s.upload_bits(&a, 3, 1, Elem::I32).unwrap();
        let db = s.upload_bits(&b, 2, 1, Elem::I32).unwrap();
        let dva = s.upload_bits(&va, 3, cols, Elem::F32).unwrap();
        let dvb = s.upload_bits(&vb, 2, cols, Elem::F32).unwrap();
        s.scatter_rows_into(&t, &da, &dva, true).unwrap();
        s.scatter_rows_into(&t, &db, &dvb, true).unwrap();
        let want = expected(&expected(&bits, cols, &a, &va), cols, &b, &vb);
        assert_bits(
            &s.download_bits(&t).unwrap(),
            &want,
            "two scatters in place",
        );
    });
}

/// One batch of indices (32 or fewer, a decode step's row a head) split between units
/// (`record::scatter_rows_range`), in place: each named row is written once, from its last index,
/// whichever unit's range holds it, and no other row changes. Repeats inside and across the
/// ranges; 32, 9 and 3 indices; two and three tile columns; row and column indices; and 33
/// indices (two batches, not split) beside them. Watched failing with each range deduplicated
/// only within itself (the record unit gate `scatter_rows_ranges_split_one_records_writes`
/// fails the same way).
#[test]
fn one_batch_split_between_units_writes_each_row_once_from_its_last_index() {
    with_units(16, |s| {
        for cols in [64usize, 96] {
            let rows = 32 * 40;
            let bits = raw_bits(cols as u64, rows * cols);
            let t = s.upload_bits(&bits, rows, cols, Elem::F32).unwrap();
            let mut want = bits.clone();
            let mut idx32: Vec<u32> = (0..32).map(|h| h * 40 + 7).collect();
            // Repeats: across the ranges (1 and 30), and three of one row (5, 6, 20).
            idx32[30] = idx32[1];
            idx32[6] = idx32[5];
            idx32[20] = idx32[5];
            let cases: [(&str, Vec<u32>); 4] = [
                ("32 heads", idx32),
                ("9 heads", (0..9).map(|h| h * 40 + 39).collect()),
                ("3 heads", vec![1, 41, 1]),
                ("33 rows", (0..33).map(|i| (i * 37) % 1280).collect()),
            ];
            for (k, (what, idx)) in cases.into_iter().enumerate() {
                let n = idx.len();
                let values = raw_bits(100 + k as u64, n * cols);
                let column = k % 2 == 0;
                let di = if column {
                    s.upload_bits(&idx, n, 1, Elem::I32).unwrap()
                } else {
                    s.upload_bits(&idx, 1, n, Elem::I32).unwrap()
                };
                let dv = s.upload_bits(&values, n, cols, Elem::F32).unwrap();
                s.scatter_rows_into(&t, &di, &dv, column).unwrap();
                want = expected(&want, cols, &idx, &values);
                assert_bits(
                    &s.download_bits(&t).unwrap(),
                    &want,
                    &format!("{what}, {cols} columns"),
                );
                s.free(di).unwrap();
                s.free(dv).unwrap();
            }
            s.free(t).unwrap();
        }
    });
}

/// Negative controls: an index at the row count, or negative, is `DOMAIN`
/// from the mover (code 11), and the valid indices beside it do not rescue it.
#[test]
fn an_index_outside_the_table_is_domain() {
    for invalid in [100u32, u32::MAX, 0x8000_0000] {
        with_session(|s| {
            let bits = raw_bits(5, 100 * 40);
            let table = s.upload_bits(&bits, 100, 40, Elem::F32).unwrap();
            let values = raw_bits(6, 3 * 40);
            let dv = s.upload_bits(&values, 3, 40, Elem::F32).unwrap();
            let valid = s.upload_bits(&[0, 17, 99], 3, 1, Elem::I32).unwrap();
            let out = s.scatter_rows(&table, &valid, &dv, true).unwrap();
            assert_bits(
                &s.download_bits(&out).unwrap(),
                &expected(&bits, 40, &[0, 17, 99], &values),
                "the valid indices",
            );
            let bad = s.upload_bits(&[0, invalid, 99], 3, 1, Elem::I32).unwrap();
            let _pending = s.scatter_rows(&table, &bad, &dv, true).unwrap();
            let err = s.sync().unwrap_err();
            assert!(
                err.to_string().contains("code 11"),
                "index {invalid:#x} DOMAIN: {err}"
            );
        });
    }
}

#[test]
fn shapes_the_scatter_does_not_take_are_refused_by_name() {
    with_session(|s| {
        let table = s
            .upload_bits(&raw_bits(1, 64 * 8), 64, 8, Elem::F32)
            .unwrap();
        let values = s.upload_bits(&raw_bits(2, 3 * 8), 3, 8, Elem::F32).unwrap();
        let idx = s.upload_bits(&[1, 2, 3], 3, 1, Elem::I32).unwrap();
        // A column of indices read as a row: three of them, not one.
        assert!(s.scatter_rows(&table, &idx, &values, false).is_err());
        // Values of another width.
        let narrow = s.upload_bits(&raw_bits(3, 3 * 7), 3, 7, Elem::F32).unwrap();
        assert!(s.scatter_rows(&table, &idx, &narrow, true).is_err());
        // Float indices.
        let float = s.upload_bits(&[1, 2, 3], 3, 1, Elem::F32).unwrap();
        let e = s.scatter_rows(&table, &float, &values, true).unwrap_err();
        assert!(e.to_string().contains("I32"), "{e}");
        // A view is another tensor's slots: written in place, refused.
        let view = table.rows_view(32, 32).unwrap();
        assert!(s.scatter_rows_into(&view, &idx, &values, true).is_err());
    });
}

// ---------------------------------------------------------------------------
// Burn: `scatter_nd` with `Assign` on the card.

use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData, TensorPrimitive};
use burn_tt::{device_time, tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// How many times the server ran a kind of op in this process so far
/// (`burn_tt::device_time`): each gate's fork is its own process.
fn ran(kind: &str) -> u64 {
    device_time()
        .iter()
        .find(|(k, _, _)| *k == kind)
        .map_or(0, |(_, n, _)| *n)
}

fn floats_of(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
}

fn bits_of<const D: usize>(t: Tensor<TtBackend, D>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// `scatter_nd(.., Assign)` with one coordinate is the row scatter, once, and
/// nothing else: no download, no mask loop, the result on the card and the
/// host's writes bit for bit, repeats and indices the card computed included.
/// Spread over several tiles, the column groups of a 25-tile-wide table and
/// repeats across batches still give the last row.
#[test]
fn scatter_nd_assign_is_one_row_scatter_and_the_host_s_writes() {
    for tiles in [1usize, 4] {
        with_device(
            Config {
                tiles: Some(burn_tt::TileChoice::Count(tiles)),
                ..Config::default()
            },
            |d| {
                for (rows, cols, n) in [(100usize, 70usize, 33usize), (40, 800, 70), (7, 3, 5)] {
                    let bits = raw_bits(rows as u64 * 31 + n as u64, rows * cols);
                    let data = Tensor::<TtBackend, 2>::from_data(
                        TensorData::new(floats_of(&bits), [rows, cols]),
                        &d,
                    )
                    .to_device(&d);
                    let idx = indices(rows, n);
                    let ids = Tensor::<TtBackend, 2, Int>::from_data(
                        TensorData::new(idx.iter().map(|&i| i as i32).collect::<Vec<_>>(), [n, 1]),
                        &d,
                    )
                    .add_scalar(0);
                    let values = raw_bits(n as u64, n * cols);
                    let v = Tensor::<TtBackend, 2>::from_data(
                        TensorData::new(floats_of(&values), [n, cols]),
                        &d,
                    )
                    .to_device(&d);
                    let (scatters, eltwise, checks) =
                        (ran("scatter_rows"), ran("eltwise"), ran("gather_indexed"));
                    let before = tensor_traffic();
                    let (out, report) = burn_tt::with_report(|| {
                        data.clone().scatter_nd(ids, v, IndexingUpdateOp::Assign)
                    });
                    assert_native_model(&report);
                    assert_eq!(tensor_traffic().downloads, before.downloads);
                    assert_eq!(
                        ran("scatter_rows"),
                        scatters + 1,
                        "the row scatter ran once"
                    );
                    assert_eq!(ran("eltwise"), eltwise, "and no mask select");
                    assert_eq!(ran("gather_indexed"), checks, "nor a coordinate check");
                    let TensorPrimitive::Float(p) = out.clone().into_primitive() else {
                        unreachable!()
                    };
                    assert!(p.computed_on_device());
                    assert_bits(
                        &bits_of(out),
                        &expected(&bits, cols, &idx, &values),
                        &format!("{n} rows into [{rows}, {cols}] on {tiles} tiles"),
                    );
                    assert_bits(&bits_of(data), &bits, "the data");
                }
            },
        );
    }
}

/// A fixed cache's write: `[1, heads, len, d]` as rows, each head's row at
/// `h len + position`, the position on the card -- what `FixedCache` does --
/// and two coordinates (`K = 2`, the checked flat rows) giving the same.
#[test]
fn a_cache_write_lands_on_each_head_s_row() {
    with_device(Config::default(), |d| {
        let (heads, len, dim, position) = (5usize, 64usize, 48usize, 37usize);
        let bits = raw_bits(11, heads * len * dim);
        let cache = Tensor::<TtBackend, 4>::from_data(
            TensorData::new(floats_of(&bits), [1, heads, len, dim]),
            &d,
        )
        .to_device(&d);
        let new_bits = raw_bits(12, heads * dim);
        let new = Tensor::<TtBackend, 4>::from_data(
            TensorData::new(floats_of(&new_bits), [1, heads, 1, dim]),
            &d,
        )
        .to_device(&d);
        let at = Tensor::<TtBackend, 1, Int>::from_data([position as i32], &d).add_scalar(0);
        let rows = (Tensor::<TtBackend, 1, Int>::arange(0..heads as i64, &d)
            .mul_scalar(len as i64)
            + at.clone().expand([heads]))
        .reshape([heads, 1]);
        let written = cache
            .clone()
            .reshape([heads * len, dim])
            .scatter_nd(
                rows,
                new.clone().reshape([heads, dim]),
                IndexingUpdateOp::Assign,
            )
            .reshape([1, heads, len, dim]);
        let idx: Vec<u32> = (0..heads).map(|h| (h * len + position) as u32).collect();
        let want = expected(&bits, dim, &idx, &new_bits);
        assert_bits(&bits_of(written), &want, "the cache write");
        // As `[heads, len]` coordinates: the same rows.
        let coords = Tensor::cat(
            vec![
                Tensor::<TtBackend, 1, Int>::arange(0..heads as i64, &d).reshape([heads, 1]),
                at.expand([heads]).reshape([heads, 1]),
            ],
            1,
        );
        let by_coords = cache.reshape([heads, len, dim]).scatter_nd(
            coords,
            new.reshape([heads, dim]),
            IndexingUpdateOp::Assign,
        );
        assert_bits(&bits_of(by_coords), &want, "the cache write by coordinates");
    });
}

/// A trace captured with a cache write replays the *device's* position: the
/// row written is the argmax of the replayed input.
#[test]
fn a_trace_replays_a_row_scatter_at_the_indices_it_finds() {
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            let cache = Tensor::<TtBackend, 2>::zeros([4, 5], &d).add_scalar(1.0);
            let row: Vec<f32> = (0..5).map(|i| i as f32 * 3.0 + 2.0).collect();
            let v = Tensor::<TtBackend, 2>::from_data(TensorData::new(row.clone(), [1, 5]), &d)
                .mul_scalar(1.0);
            let x = Tensor::<TtBackend, 2>::from_data([[0.1, 0.9, 0.2, 0.3]], &d).mul_scalar(1.0);
            let TensorPrimitive::Float(input) = x.clone().into_primitive() else {
                unreachable!()
            };
            let scatters = ran("scatter_rows");
            let (trace, first) = burn_tt::Trace::capture(&input, || {
                let at = x.clone().argmax(1).reshape([1, 1]);
                let out = cache
                    .clone()
                    .scatter_nd(at, v.clone(), IndexingUpdateOp::Assign);
                let TensorPrimitive::Float(out) = out.into_primitive() else {
                    unreachable!()
                };
                out
            })
            .unwrap();
            assert_eq!(
                ran("scatter_rows"),
                scatters + 1,
                "the capture ran the scatter"
            );
            let written = |r: usize| {
                let mut out = vec![1.0f32; 4 * 5];
                out[r * 5..(r + 1) * 5].copy_from_slice(&row);
                out
            };
            assert_eq!(first, written(1));
            assert_eq!(trace.run(vec![5.0, 1.0, 2.0, 3.0]).unwrap(), written(0));
            assert_eq!(trace.run(vec![0.0, 0.0, 0.0, 7.0]).unwrap(), written(3));
            assert_eq!(trace.run(vec![1.0, 2.0, 9.0, 0.5]).unwrap(), written(2));
        },
    );
}

/// Padding: a scatter into a table whose rows and columns are ragged (33 x 40)
/// is followed by a column sum and a matmul with a row of ones, which read the
/// tile's rows past the 33rd and columns past the 40th -- exact on small
/// integers whatever the order, so nonzero padding changes the result.
#[test]
fn a_ragged_scatter_feeds_reductions_and_matmuls_exactly() {
    with_device(Config::default(), |d| {
        let (rows, cols, n) = (33usize, 40usize, 9usize);
        let table: Vec<f32> = (0..rows * cols).map(|i| (i % 7) as f32 + 1.0).collect();
        let data =
            Tensor::<TtBackend, 2>::from_data(TensorData::new(table.clone(), [rows, cols]), &d)
                .to_device(&d);
        let values: Vec<f32> = (0..n * cols).map(|i| (i % 5) as f32 - 2.0).collect();
        // Values made on the card, ragged themselves.
        let v = Tensor::<TtBackend, 2>::from_data(TensorData::new(values.clone(), [n, cols]), &d)
            .add_scalar(0.0);
        let idx = indices(rows, n);
        let ids = Tensor::<TtBackend, 2, Int>::from_data(
            TensorData::new(idx.iter().map(|&i| i as i32).collect::<Vec<_>>(), [n, 1]),
            &d,
        )
        .add_scalar(0);
        let out = data.scatter_nd(ids, v, IndexingUpdateOp::Assign);
        let mut host = table.clone();
        for (i, &r) in idx.iter().enumerate() {
            host[r as usize * cols..(r as usize + 1) * cols]
                .copy_from_slice(&values[i * cols..(i + 1) * cols]);
        }
        let mut want = vec![0f32; cols];
        for r in 0..rows {
            for (c, w) in want.iter_mut().enumerate() {
                *w += host[r * cols + c];
            }
        }
        let sum = out.clone().sum_dim(0).into_data().to_vec::<f32>().unwrap();
        assert_eq!(sum, want, "the column sums");
        let ones = Tensor::<TtBackend, 2>::ones([1, rows], &d);
        let product = ones
            .matmul(out.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(product, want, "ones @ the scatter");
        let row_sums = out.sum_dim(1).into_data().to_vec::<f32>().unwrap();
        let want: Vec<f32> = (0..rows)
            .map(|r| host[r * cols..(r + 1) * cols].iter().sum())
            .collect();
        assert_eq!(row_sums, want, "the row sums");
    });
}

/// An index outside the table is the card's error through Burn, not a write
/// elsewhere: the same `DOMAIN` the primitive reports, with no flat-row check
/// before it for one coordinate.
#[test]
fn an_index_outside_the_table_is_an_error_through_burn() {
    for bad in [10, -1] {
        with_device(Config::default(), |d| {
            let data = Tensor::<TtBackend, 2>::from_data(
                TensorData::new(vec![1.0f32; 10 * 4], [10, 4]),
                &d,
            )
            .to_device(&d);
            let v = Tensor::<TtBackend, 2>::zeros([3, 4], &d);
            let ids = Tensor::<TtBackend, 2, Int>::from_data(
                TensorData::new(vec![2, bad, 3], [3, 1]),
                &d,
            )
            .add_scalar(0);
            let err = data
                .clone()
                .scatter_nd(ids, v.clone(), IndexingUpdateOp::Assign)
                .try_into_data()
                .expect_err("a write to a row outside the table failed")
                .to_string();
            assert!(err.contains("code 11"), "index {bad}: {err}");
        });
    }
}
