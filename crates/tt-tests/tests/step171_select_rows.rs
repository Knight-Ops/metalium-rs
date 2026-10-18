//! Row select without staging: `Session::select_rows` and `burn-tt`'s `select`.
//!
//! One token of SmolLM2's embedding cost 15.9 s because `select` staged every
//! row of the table for every element it returned (host addresses,
//! roadmap Track 2, "Gather without staging"). `Session::select_rows` reads a
//! tile of resident indices and B's `SELECT_ROWS` record turns each into the
//! table's two 64-byte face-row reads a tile column, so a lookup costs reads
//! in the rows it returns and no host address.
//!
//! The claims, each against an oracle that is the host indexing a `Vec` (a
//! select is a raw copy: bit for bit, never a tolerance):
//! - every datum a table holds comes through unchanged: signed zero,
//!   subnormals, NaN payloads, infinities, `f32::MAX`, and I32 and Bool
//!   tables;
//! - any row count against the 32-row tile (1, 31, 32, 33, 70), a table whose
//!   rows and columns are ragged, one wider than a batch's staging, one that
//!   is a view starting mid-grid, repeated and first and last rows;
//! - an index at the table's row count, or negative, stops the mover with
//!   `DOMAIN` before a datum is copied.
//!
//! Watched failing: with the face halves swapped, with the source row taken
//! mod 31, and with `bound` one too large (the last two caught by the
//! `tt_isa::dm::record` unit tests as well).

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

/// A session on `units` Tensix units (the select deals its tile groups between them).
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

/// `n` indices into `rows` rows: the first and last row, repeats, and a
/// scramble that crosses tile rows.
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

fn expected(table: &[u32], cols: usize, indices: &[u32]) -> Vec<u32> {
    indices
        .iter()
        .flat_map(|&r| {
            table[r as usize * cols..(r as usize + 1) * cols]
                .iter()
                .copied()
        })
        .collect()
}

fn assert_bits(got: &[u32], want: &[u32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g, w,
            "{what}: datum {i} ({} of a row of the output): {g:#010x} vs {w:#010x}",
            i
        );
    }
}

#[test]
fn selected_rows_are_the_host_s_indexing_bit_for_bit() {
    with_session(|s| {
        // (rows, cols): ragged both ways, exact tiles, one row, and 25 tile
        // columns, which a batch's staging (24) splits in two.
        for (rows, cols) in [(100, 70), (64, 64), (33, 32), (1, 5), (40, 800), (97, 33)] {
            let bits = raw_bits(rows as u64 * 1000 + cols as u64, rows * cols);
            let table = s.upload_bits(&bits, rows, cols, Elem::F32).unwrap();
            for n in [1usize, 31, 32, 33, 70] {
                let idx = indices(rows, n);
                let di = s.upload_bits(&idx, 1, n, Elem::I32).unwrap();
                let out = s.select_rows(&table, &di).unwrap();
                assert_eq!([out.rows, out.cols], [n, cols]);
                assert_eq!(out.pad(), tt_kernels::tensor::Pad::Zero);
                assert_bits(
                    &s.download_bits(&out).unwrap(),
                    &expected(&bits, cols, &idx),
                    &format!("{n} of the rows of [{rows}, {cols}]"),
                );
                s.free(out).unwrap();
                s.free(di).unwrap();
            }
            // The table is unchanged.
            assert_bits(&s.download_bits(&table).unwrap(), &bits, "the table");
            s.free(table).unwrap();
        }
    });
}

/// On 16 units, a select's tile groups are narrowed so every unit has one (a decode step's
/// embedding row of 64 tiles: 64 groups of one tile, not 3 of 24): the rows are the host's
/// indexing, and the output's padding rows are still zero (`Pad::Zero`), each unit filling its
/// own tiles'. One row of 2048 columns (the 1.7B embedding), of 576 (the 135M's), 9 rows, and
/// 33 (two output tile rows).
#[test]
fn a_select_over_many_units_is_the_host_s_indexing_with_zero_padding() {
    with_units(16, |s| {
        for (rows, cols) in [(300usize, 2048usize), (100, 576), (97, 70)] {
            let bits = raw_bits(rows as u64 * 7 + cols as u64, rows * cols);
            let table = s.upload_bits(&bits, rows, cols, Elem::F32).unwrap();
            for n in [1usize, 9, 33] {
                let idx = indices(rows, n);
                let di = s.upload_bits(&idx, 1, n, Elem::I32).unwrap();
                let out = s.select_rows(&table, &di).unwrap();
                let what = format!("{n} of the rows of [{rows}, {cols}]");
                assert_eq!(out.pad(), tt_kernels::tensor::Pad::Zero, "{what}");
                assert_bits(
                    &s.download_bits(&out).unwrap(),
                    &expected(&bits, cols, &idx),
                    &what,
                );
                let padded = s.download_padded(&out).unwrap();
                let pc = 32 * cols.div_ceil(32);
                for (k, v) in padded.iter().enumerate() {
                    if k / pc >= n || k % pc >= cols {
                        assert_eq!(v.to_bits(), 0, "{what}: padding datum {k}");
                    }
                }
                s.free(out).unwrap();
                s.free(di).unwrap();
            }
            s.free(table).unwrap();
        }
    });
}

#[test]
fn integer_and_bool_tables_select_as_raw_words() {
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
            let di = s.upload_bits(&idx, 1, 45, Elem::I32).unwrap();
            let out = s.select_rows(&table, &di).unwrap();
            assert_eq!(out.elem, elem);
            assert_bits(
                &s.download_bits(&out).unwrap(),
                &expected(bits, 9, &idx),
                &format!("{elem:?}"),
            );
        }
    });
}

/// A table that starts mid-grid (`rows_view`: its tiles are another tensor's,
/// from a tile row on) selects the view's rows, not the parent's: one that ends
/// on a tile row, and one that ends ragged at the parent's last row.
#[test]
fn a_view_selects_its_own_rows() {
    with_session(|s| {
        let (rows, cols) = (150, 70);
        let bits = raw_bits(77, rows * cols);
        let parent = s.upload_bits(&bits, rows, cols, Elem::F32).unwrap();
        for (first, count) in [(32, 96), (64, 86)] {
            let view = parent.rows_view(first, count).unwrap();
            let idx = indices(count, 41);
            let di = s.upload_bits(&idx, 1, 41, Elem::I32).unwrap();
            let out = s.select_rows(&view, &di).unwrap();
            assert_bits(
                &s.download_bits(&out).unwrap(),
                &expected(&bits[first * cols..(first + count) * cols], cols, &idx),
                &format!("rows {first}..{} of [150, 70]", first + count),
            );
        }
        // An index past the view's end is the view's DOMAIN, though the
        // parent has the row.
        let di = s.upload_bits(&[0, 96], 1, 2, Elem::I32).unwrap();
        let view = parent.rows_view(32, 96).unwrap();
        let _pending = s.select_rows(&view, &di).unwrap();
        let err = s.sync().unwrap_err();
        assert!(err.to_string().contains("code 11"), "{err}");
    });
}

/// Negative controls: an index at the row count, or negative, is `DOMAIN`
/// from the mover (code 11), and the valid index before it does not rescue it.
#[test]
fn an_index_outside_the_table_is_domain_before_any_copy() {
    for invalid in [100u32, u32::MAX, 0x8000_0000] {
        with_session(|s| {
            let bits = raw_bits(5, 100 * 40);
            let table = s.upload_bits(&bits, 100, 40, Elem::F32).unwrap();
            let valid = s.upload_bits(&[0, 17, 99], 1, 3, Elem::I32).unwrap();
            let out = s.select_rows(&table, &valid).unwrap();
            assert_bits(
                &s.download_bits(&out).unwrap(),
                &expected(&bits, 40, &[0, 17, 99]),
                "the valid indices",
            );
            let bad = s.upload_bits(&[0, invalid, 99], 1, 3, Elem::I32).unwrap();
            let _pending = s.select_rows(&table, &bad).unwrap();
            let err = s.sync().unwrap_err();
            assert!(
                err.to_string().contains("code 11"),
                "index {invalid:#x} DOMAIN: {err}"
            );
        });
    }
}

#[test]
fn shapes_the_select_does_not_take_are_refused_by_name() {
    with_session(|s| {
        let table = s
            .upload_bits(&raw_bits(1, 64 * 8), 64, 8, Elem::F32)
            .unwrap();
        // Indices as a column, not a row.
        let col = s.upload_bits(&[1, 2, 3], 3, 1, Elem::I32).unwrap();
        assert!(s.select_rows(&table, &col).is_err());
        // Float indices.
        let float = s.upload_bits(&[1, 2, 3], 1, 3, Elem::F32).unwrap();
        let e = s.select_rows(&table, &float).unwrap_err();
        assert!(e.to_string().contains("I32"), "{e}");
    });
}

// ---------------------------------------------------------------------------
// Burn: `select` and `embedding` on the card.

use burn::tensor::{module, Bool, Int, Tensor, TensorData, TensorPrimitive};
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

/// `Tensor::select` on a table (the lookup `nn::Embedding` makes) takes the
/// row select, not the staged gather, and returns the host's rows bit for bit:
/// specials, repeats, a row count off the tile grid, indices the card computed
/// (`add_scalar`), nothing downloaded and nothing computed on the host.
#[test]
fn select_takes_the_row_select_and_is_the_host_s_indexing() {
    with_device(Config::default(), |d| {
        for (vocab, cols, n) in [(100usize, 70usize, 33usize), (64, 64, 70), (7, 3, 5)] {
            let bits = raw_bits(vocab as u64 * 31 + n as u64, vocab * cols);
            let table = Tensor::<TtBackend, 2>::from_data(
                TensorData::new(floats_of(&bits), [vocab, cols]),
                &d,
            )
            .to_device(&d);
            let idx = indices(vocab, n);
            let ids = Tensor::<TtBackend, 1, Int>::from_data(
                TensorData::new(idx.iter().map(|&i| i as i32).collect::<Vec<_>>(), [n]),
                &d,
            )
            .add_scalar(0);
            let (selected, rows_before, gathers_before) = (
                ran("select_rows"),
                ran("select_rows"),
                ran("gather_indexed"),
            );
            let _ = selected;
            let before = tensor_traffic();
            let (out, report) = burn_tt::with_report(|| table.clone().select(0, ids));
            assert_native_model(&report);
            assert_eq!(tensor_traffic().downloads, before.downloads);
            assert_eq!(
                ran("select_rows"),
                rows_before + 1,
                "the row select ran once"
            );
            assert_eq!(
                ran("gather_indexed"),
                gathers_before,
                "and the staged gather did not"
            );
            let TensorPrimitive::Float(p) = out.clone().into_primitive() else {
                unreachable!()
            };
            assert!(p.computed_on_device());
            let got: Vec<u32> = out
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_bits(
                &got,
                &expected(&bits, cols, &idx),
                &format!("{n} of [{vocab}, {cols}]"),
            );
        }
        // `module::embedding` of [batch, seq] ids: the same select, reshaped.
        let bits = raw_bits(3, 50 * 12);
        let table =
            Tensor::<TtBackend, 2>::from_data(TensorData::new(floats_of(&bits), [50, 12]), &d)
                .to_device(&d);
        let ids2: Vec<i32> = (0..6).map(|i| [49, 0, 7, 7, 31, 32][i]).collect();
        let ids = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(ids2.clone(), [2, 3]), &d)
            .add_scalar(0);
        let rows_before = ran("select_rows");
        let out = module::embedding(table, ids);
        assert_eq!(out.dims(), [2, 3, 12]);
        assert_eq!(ran("select_rows"), rows_before + 1);
        let want = expected(
            &bits,
            12,
            &ids2.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        );
        let got: Vec<u32> = out
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_bits(&got, &want, "an embedding of [2, 3] ids");
    });
}

/// Padding: a selection whose row count is ragged (33 of 64) is followed by a
/// column sum and a matmul with a row of ones, which read the tile's rows past
/// the 33rd -- exact on small integers whatever the order, so a nonzero
/// padding row changes the result.
#[test]
fn a_ragged_selection_feeds_reductions_and_matmuls_exactly() {
    with_device(Config::default(), |d| {
        let (vocab, cols, n) = (90usize, 40usize, 33usize);
        let values: Vec<f32> = (0..vocab * cols).map(|i| (i % 7) as f32 + 1.0).collect();
        let table =
            Tensor::<TtBackend, 2>::from_data(TensorData::new(values.clone(), [vocab, cols]), &d)
                .to_device(&d);
        let idx = indices(vocab, n);
        let ids = Tensor::<TtBackend, 1, Int>::from_data(
            TensorData::new(idx.iter().map(|&i| i as i32).collect::<Vec<_>>(), [n]),
            &d,
        )
        .add_scalar(0);
        let rows = table.select(0, ids);
        let mut want = vec![0f32; cols];
        for &r in &idx {
            for (c, w) in want.iter_mut().enumerate() {
                *w += values[r as usize * cols + c];
            }
        }
        let sum = rows.clone().sum_dim(0).into_data().to_vec::<f32>().unwrap();
        assert_eq!(sum, want, "the column sums");
        let ones = Tensor::<TtBackend, 2>::ones([1, n], &d);
        let product = ones.matmul(rows).into_data().to_vec::<f32>().unwrap();
        assert_eq!(product, want, "ones @ the selection");
    });
}

/// Integer and boolean tables through Burn: the same select, raw words.
#[test]
fn integer_and_boolean_tables_select_through_burn() {
    with_device(Config::default(), |d| {
        let (vocab, cols) = (45usize, 6usize);
        let ints: Vec<i32> = (0..vocab * cols)
            .map(|i| [i32::MIN, i32::MAX, -1, 0, 123_456_789, -987_654][i % 6])
            .collect();
        let table = Tensor::<TtBackend, 2, Int>::from_data(
            TensorData::new(ints.clone(), [vocab, cols]),
            &d,
        )
        .to_device(&d);
        let flags: Vec<bool> = (0..vocab * cols).map(|i| i % 4 == 1).collect();
        let bools = Tensor::<TtBackend, 2, Bool>::from_data(
            TensorData::new(flags.clone(), [vocab, cols]),
            &d,
        )
        .to_device(&d);
        let idx = indices(vocab, 37);
        let ids = Tensor::<TtBackend, 1, Int>::from_data(
            TensorData::new(idx.iter().map(|&i| i as i32).collect::<Vec<_>>(), [37]),
            &d,
        )
        .add_scalar(0);
        let rows_before = ran("select_rows");
        let got = table
            .select(0, ids.clone())
            .into_data()
            .to_vec::<i32>()
            .unwrap();
        let want: Vec<i32> = idx
            .iter()
            .flat_map(|&r| ints[r as usize * cols..(r as usize + 1) * cols].to_vec())
            .collect();
        assert_eq!(got, want);
        let got = bools.select(0, ids).into_data().to_vec::<bool>().unwrap();
        let want: Vec<bool> = idx
            .iter()
            .flat_map(|&r| flags[r as usize * cols..(r as usize + 1) * cols].to_vec())
            .collect();
        assert_eq!(got, want);
        assert_eq!(ran("select_rows"), rows_before + 2);
    });
}

/// A trace captured with a lookup replays the *device's* indices: the row is
/// the argmax of the replayed input, which a cache of host indices could not
/// follow (`traced-execution.md` 5 left whether changed ids take effect on
/// replay "not asserted").
#[test]
fn a_trace_replays_a_row_select_with_the_indices_it_finds() {
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            let table: Vec<f32> = (0..4 * 5).map(|i| i as f32 * 3.0 + 1.0).collect();
            let w = Tensor::<TtBackend, 2>::from_data(TensorData::new(table.clone(), [4, 5]), &d)
                .mul_scalar(1.0);
            let x = Tensor::<TtBackend, 2>::from_data([[0.1, 0.9, 0.2, 0.3]], &d).mul_scalar(1.0);
            let TensorPrimitive::Float(input) = x.clone().into_primitive() else {
                unreachable!()
            };
            let rows_before = ran("select_rows");
            let (trace, first) = burn_tt::Trace::capture(&input, || {
                let row = x.clone().argmax(1).reshape([1]);
                let out = w.clone().select(0, row);
                let TensorPrimitive::Float(out) = out.into_primitive() else {
                    unreachable!()
                };
                out
            })
            .unwrap();
            assert_eq!(
                ran("select_rows"),
                rows_before + 1,
                "the capture ran the row select"
            );
            let row = |r: usize| table[r * 5..(r + 1) * 5].to_vec();
            assert_eq!(first, row(1));
            assert_eq!(trace.run(vec![5.0, 1.0, 2.0, 3.0]).unwrap(), row(0));
            assert_eq!(trace.run(vec![0.0, 0.0, 0.0, 7.0]).unwrap(), row(3));
            assert_eq!(trace.run(vec![1.0, 2.0, 9.0, 0.5]).unwrap(), row(2));
        },
    );
}

/// An index outside the table is the card's error through Burn, not a wrong
/// row: the same `DOMAIN` the primitive reports.
#[test]
fn an_index_outside_the_table_is_an_error_through_burn() {
    with_device(Config::default(), |d| {
        let table =
            Tensor::<TtBackend, 2>::from_data(TensorData::new(vec![1.0f32; 10 * 4], [10, 4]), &d)
                .to_device(&d);
        let ids = Tensor::<TtBackend, 1, Int>::from_data([2, 10, 3], &d).add_scalar(0);
        let err = table
            .select(0, ids)
            .try_into_data()
            .expect_err("the lookup of row 10 of 10 failed")
            .to_string();
        assert!(err.contains("code 11"), "{err}");
    });
}
