//! Rows added at resident indices, in order (`Session::select_add_rows`): Burn's `select_add`
//! along rows, an embedding's gradient, with no index read by the host.
//!
//! Since `6d979e7` (2026-10-06) `float_select_add` kept its indices on the card by running
//! `scatter_add_native`'s mask loop, about twenty ops an index: 90% of the tt-mnist
//! transformer's ~2,860 ops a step (92 ms/step on card 1 against 9.2 before). It is now one op
//! (`tensor::select_add_rows_jobs`): a job a group of tile columns, and in it a block an index
//! (the row selected, its value row added, the sum scattered back), each block's reads waiting
//! for the last block's writes.
//!
//! The claims:
//! - **Bit for bit the host's sequential `+=`** over rows repeated up to every fifth index,
//!   untouched rows kept to the bit, ragged widths and row counts, more indices than a tile row,
//!   and the input left as it was.
//! - **The same bits as `Session::rows_add`** (the host-index route, gated by `step62`) over
//!   signed zeros, subnormals, infinities and NaN, where the host's `+=` is not the device's
//!   (subnormals).
//! - **In index order:** a row given `1.0` and twice `2^-24` stays `1.0` (each addition rounds to
//!   even); adding the values first would give `1 + 2^-23` (the negative control, computed).
//! - **Every index naming one row** (the longest chain of reads after writes): the host's
//!   sequential sum, which a read that overtook a write would lose an addition from.
//! - **An index outside the table** is the row select's `DOMAIN` (code 11).
//! - **Through Burn:** `select_assign(.., Add)` on `TtBackend` runs this op once, downloads
//!   nothing, and equals the host's loop.
//! - **In a trace,** captured and replayed over other indices, it is the op run fresh.

use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Elem};
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(4),
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
            TileChoice::Count(4),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

/// `floats` with signed zeros, subnormals, infinities and NaN mixed in.
fn specials(seed: u64, n: usize) -> Vec<f32> {
    floats(seed, n)
        .into_iter()
        .enumerate()
        .map(|(i, x)| match i % 37 {
            2 => -0.0,
            5 => 0.0,
            11 => f32::from_bits(0x0000_0042),
            17 => -f32::from_bits(0x0070_0000),
            23 => f32::INFINITY,
            29 => f32::NEG_INFINITY,
            31 => f32::NAN,
            33 => f32::MAX,
            _ => x,
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    let (g, w) = (bits(got), bits(want));
    if let Some(i) = g.iter().zip(&w).position(|(a, b)| a != b) {
        panic!(
            "{what}: datum {i} is {:#010x} ({}), want {:#010x} ({})",
            g[i], got[i], w[i], want[i]
        );
    }
    assert_eq!(g.len(), w.len(), "{what}: length");
}

/// Small indices repeat: row 3 appears every fifth index.
fn repeating(vocab: usize, n: usize) -> Vec<usize> {
    (0..n)
        .map(|i| if i % 5 == 0 { 3 } else { (i * 13 + 7) % vocab })
        .collect()
}

fn index_row<T: tt_device::Transport>(s: &mut Session<T>, idx: &[usize]) -> DramTensor {
    let words: Vec<u32> = idx.iter().map(|&i| i as u32).collect();
    s.upload_bits(&words, 1, idx.len(), Elem::I32).unwrap()
}

fn sequential(t: &[f32], v: &[f32], idx: &[usize], cols: usize) -> Vec<f32> {
    let mut want = t.to_vec();
    for (i, &r) in idx.iter().enumerate() {
        for c in 0..cols {
            want[r * cols + c] += v[i * cols + c];
        }
    }
    want
}

#[test]
fn rows_added_at_resident_indices_are_the_host_s_sequential_sums() {
    with_session(|s| {
        for (vocab, cols, n) in [(64usize, 64usize, 70usize), (100, 70, 37), (7, 3, 1)] {
            let t = floats(vocab as u64 + 1, vocab * cols);
            let v = floats(n as u64 + 2, n * cols);
            let idx = repeating(vocab, n);
            let (dt, dv, di) = (
                s.upload(&t, vocab, cols).unwrap(),
                s.upload(&v, n, cols).unwrap(),
                index_row(s, &idx),
            );
            let got = s.select_add_rows(&dt, &di, &dv).unwrap();
            assert_bits(
                &s.download(&got).unwrap(),
                &sequential(&t, &v, &idx, cols),
                &format!("{n} rows into [{vocab}, {cols}]"),
            );
            assert_bits(&s.download(&dt).unwrap(), &t, "the input");
            for x in [got, dt, dv, di] {
                s.free(x).unwrap();
            }
        }
    });
}

#[test]
fn specials_are_the_host_index_route_s_bits() {
    with_session(|s| {
        let (vocab, cols, n) = (40usize, 64usize, 45usize);
        let t = specials(3, vocab * cols);
        let v = specials(4, n * cols);
        let idx = repeating(vocab, n);
        let (dt, dv, di) = (
            s.upload(&t, vocab, cols).unwrap(),
            s.upload(&v, n, cols).unwrap(),
            index_row(s, &idx),
        );
        let got = s.select_add_rows(&dt, &di, &dv).unwrap();
        let want = s.rows_add(&dt, &idx, &dv).unwrap();
        assert_bits(
            &s.download(&got).unwrap(),
            &s.download(&want).unwrap(),
            "against Session::rows_add",
        );
    });
}

#[test]
fn a_repeated_row_takes_its_additions_in_index_order() {
    with_session(|s| {
        let cols = 32;
        let tiny = f32::powi(2.0, -24);
        // Row 2 starts at zero and takes 1, 2^-24 and -1, then 2^-24 and 2^-24 again from 1.
        let t = vec![0.0f32; 4 * cols];
        let mut v = Vec::new();
        for x in [1.0, tiny, -1.0, 1.0, tiny, tiny] {
            v.extend(std::iter::repeat_n(x, cols));
        }
        let idx = [2usize, 2, 2, 1, 1, 1];
        let (dt, dv, di) = (
            s.upload(&t, 4, cols).unwrap(),
            s.upload(&v, 6, cols).unwrap(),
            index_row(s, &idx),
        );
        let sum = s.select_add_rows(&dt, &di, &dv).unwrap();
        let got = s.download(&sum).unwrap();
        // In order: 0 + 1 + 2^-24 rounds to 1, and - 1 leaves 0.
        assert_bits(&got[2 * cols..3 * cols], &vec![0.0; cols], "row 2 in order");
        // In order: 1 + 2^-24 rounds to 1, twice.
        assert_bits(&got[cols..2 * cols], &vec![1.0; cols], "row 1 in order");
        // The controls, computed: the reverse order, and the values summed first, differ.
        assert_eq!(((0.0f32 - 1.0) + tiny) + 1.0, tiny, "the reverse order");
        assert_ne!(1.0f32 + (tiny + tiny), 1.0, "the values first");
    });
}

#[test]
fn every_index_naming_one_row_is_the_host_s_sequential_sum() {
    with_session(|s| {
        let (vocab, cols, n) = (8usize, 96usize, 100usize);
        let t = floats(11, vocab * cols);
        let v = floats(12, n * cols);
        let idx = vec![5usize; n];
        let (dt, dv, di) = (
            s.upload(&t, vocab, cols).unwrap(),
            s.upload(&v, n, cols).unwrap(),
            index_row(s, &idx),
        );
        let got = s.select_add_rows(&dt, &di, &dv).unwrap();
        assert_bits(
            &s.download(&got).unwrap(),
            &sequential(&t, &v, &idx, cols),
            "row 5, 100 times",
        );
    });
}

#[test]
fn an_index_outside_the_table_is_domain() {
    with_session(|s| {
        let t = floats(5, 10 * 32);
        let dt = s.upload(&t, 10, 32).unwrap();
        let dv = s.upload(&floats(6, 3 * 32), 3, 32).unwrap();
        let di = index_row(s, &[1, 10, 2]);
        let _pending = s.select_add_rows(&dt, &di, &dv).unwrap();
        let err = s.sync().unwrap_err();
        assert!(err.to_string().contains("code 11"), "index 10 of 10: {err}");
    });
}

#[test]
fn a_captured_select_add_replays_with_the_indices_it_finds() {
    with_session(|s| {
        let (vocab, cols, n) = (48usize, 32usize, 9usize);
        let t = floats(7, vocab * cols);
        let v = floats(8, n * cols);
        let (dt, dv) = (
            s.upload(&t, vocab, cols).unwrap(),
            s.upload(&v, n, cols).unwrap(),
        );
        let first = repeating(vocab, n);
        let di = index_row(s, &first);
        s.begin_trace().unwrap();
        let out = s.select_add_rows(&dt, &di, &dv).unwrap();
        let id = s.end_trace().unwrap();
        assert_bits(
            &s.download(&out).unwrap(),
            &sequential(&t, &v, &first, cols),
            "the capture",
        );
        for seed in [3usize, 11] {
            let idx: Vec<usize> = (0..n).map(|i| (i * seed + seed) % vocab).collect();
            let words: Vec<u32> = idx.iter().map(|&i| i as u32).collect();
            s.write_bits(&di, &words).unwrap();
            s.replay(id).unwrap();
            s.sync().unwrap();
            assert_bits(
                &s.download(&out).unwrap(),
                &sequential(&t, &v, &idx, cols),
                &format!("a replay over indices x{seed}"),
            );
        }
        s.release_trace(id).unwrap();
    });
}

// ---------------------------------------------------------------------------
// Burn: `select_assign(.., Add)` on the card.

use burn::tensor::{IndexingUpdateOp, Int, Tensor, TensorData};
use burn_tt::{device_time, tensor_traffic, TtBackend};
use tt_tests::burn_device::{with_device, Config};

fn ran(kind: &str) -> u64 {
    device_time()
        .iter()
        .find(|(k, _, _)| *k == kind)
        .map_or(0, |(_, n, _)| *n)
}

#[test]
fn select_add_through_burn_is_one_op_and_the_host_s_loop() {
    with_device(Config::default(), |d| {
        let (vocab, cols, n) = (50usize, 40usize, 33usize);
        let t = floats(9, vocab * cols);
        let v = floats(10, n * cols);
        let idx = repeating(vocab, n);
        let table =
            Tensor::<TtBackend, 2>::from_data(TensorData::new(t.clone(), [vocab, cols]), &d)
                .to_device(&d);
        let values = Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), [n, cols]), &d)
            .to_device(&d);
        let ids = Tensor::<TtBackend, 1, Int>::from_data(
            TensorData::new(idx.iter().map(|&i| i as i32).collect::<Vec<_>>(), [n]),
            &d,
        )
        .add_scalar(0);
        let before = (ran("select_add_rows"), tensor_traffic().downloads);
        let out = table.select_assign(0, ids, values, IndexingUpdateOp::Add);
        let got = out.into_data().to_vec::<f32>().unwrap();
        assert_eq!(ran("select_add_rows"), before.0 + 1, "the op ran once");
        assert_eq!(
            tensor_traffic().downloads,
            before.1 + 1,
            "only the result came down"
        );
        assert_bits(&got, &sequential(&t, &v, &idx, cols), "select_assign Add");
    });
}
