//! A gather and a scatter-add of one index a row along the last dimension (cross-entropy's pick
//! of each row's target and its gradient) as a few ops: `Session::pick_columns` (the mover's
//! `INDEX_PICK` over every tile row, one op) and `where(column == index, x + value, x)` over the
//! row's column numbers, kept for the attachment.
//!
//! Since `c4255c7` (2026-10-05) these ran as staged repacks and a mask loop: 8 and 18 device jobs
//! a call and 9 broadcast copies, MNIST's step 1.4 -> 2.3 ms/step (1.9-2.0 since); now 2 and 6.
//!
//! The claims, each through Burn on `TtBackend` against the host's indexing (raw datums for the
//! gather; for the scatter-add one addition at the indexed datum, the device's element-wise `ADD`
//! of the same operands -- subnormals flushed as the SFPU's add flushes them -- and every other
//! datum the input's):
//! - **Bit for bit** at 1, 31, 32, 33 and 70 rows and 1, 10, 37 and 100 columns, rank 2 and 3,
//!   over signed zeros, subnormals, infinities, NaN payloads and `f32::MAX` in both operands,
//!   every column chosen somewhere; the datums the scatter-add does not index are kept to the bit.
//! - **The ops it names ran:** one `pick_columns` for the gather; the scatter-add's
//!   check, and no staged gather.
//! - **An index outside the row**, or negative, is the mover's `DOMAIN` (code 11), for both.
//! - **In a trace,** captured and replayed over other indices, both are the ops run fresh.

use burn::tensor::{Int, Tensor, TensorData, TensorPrimitive};
use burn_tt::{device_time, TtBackend};
use tt_tests::burn_device::{with_device, Config};

fn floats(seed: u64, n: usize, specials: bool) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0;
            if !specials {
                return x;
            }
            match i % 23 {
                2 => -0.0,
                5 => f32::from_bits(0x0000_0042),
                7 => f32::INFINITY,
                11 => f32::from_bits(0x7fc1_2345),
                13 => f32::MAX,
                17 => -f32::from_bits(0x0070_0000),
                _ => x,
            }
        })
        .collect()
}

/// One index a row, every column chosen somewhere.
fn picks(rows: usize, cols: usize) -> Vec<i32> {
    (0..rows).map(|r| ((r * 7 + 3) % cols) as i32).collect()
}

fn first_diff(got: &[f32], want: &[f32]) -> Option<(usize, u32, u32)> {
    got.iter()
        .zip(want)
        .position(|(a, b)| a.to_bits() != b.to_bits())
        .map(|i| (i, got[i].to_bits(), want[i].to_bits()))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn ran(kind: &str) -> u64 {
    device_time()
        .iter()
        .find(|(k, _, _)| *k == kind)
        .map_or(0, |(_, n, _)| *n)
}

fn host_gather(x: &[f32], cols: usize, idx: &[i32]) -> Vec<f32> {
    idx.iter()
        .enumerate()
        .map(|(r, &i)| x[r * cols + i as usize])
        .collect()
}

/// `x` with `sum`'s datum at each row's index: `sum` is `x + v` as the device's element-wise
/// `ADD` makes it (subnormals flushed, as the SFPU's add flushes them), every other datum `x`'s.
fn host_scatter_add(x: &[f32], cols: usize, idx: &[i32], sum: &[f32]) -> Vec<f32> {
    let mut out = x.to_vec();
    for (r, &i) in idx.iter().enumerate() {
        let at = r * cols + i as usize;
        out[at] = sum[at];
    }
    out
}

/// `x + v` (`v` one a row, broadcast along it) by the device's element-wise `ADD`.
fn device_sum(x: &[f32], v: &[f32], rows: usize, cols: usize, d: &burn_tt::TtDevice) -> Vec<f32> {
    data(tensor(x.to_vec(), [rows, cols], d) + tensor(v.to_vec(), [rows, 1], d))
}

fn tensor<const D: usize>(
    v: Vec<f32>,
    shape: [usize; D],
    d: &burn_tt::TtDevice,
) -> Tensor<TtBackend, D> {
    Tensor::<TtBackend, D>::from_data(TensorData::new(v, shape), d).to_device(d)
}

fn ints<const D: usize>(
    v: Vec<i32>,
    shape: [usize; D],
    d: &burn_tt::TtDevice,
) -> Tensor<TtBackend, D, Int> {
    Tensor::<TtBackend, D, Int>::from_data(TensorData::new(v, shape), d).add_scalar(0)
}

fn data<const D: usize>(t: Tensor<TtBackend, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn one_index_a_row_is_the_host_s_indexing_bit_for_bit() {
    with_device(Config::default(), |d| {
        for (rows, cols) in [(1, 10), (31, 37), (32, 1), (33, 100), (70, 10)] {
            let what = format!("[{rows}, {cols}]");
            let x = floats((rows * 100 + cols) as u64, rows * cols, true);
            let v = floats((rows + cols) as u64, rows, true);
            let idx = picks(rows, cols);
            let (gathers, picks_before) = (ran("gather_indexed"), ran("pick_columns"));
            let got = data(
                tensor(x.clone(), [rows, cols], &d).gather(1, ints(idx.clone(), [rows, 1], &d)),
            );
            assert_eq!(
                bits(&got),
                bits(&host_gather(&x, cols, &idx)),
                "{what}: gather"
            );
            assert_eq!(ran("pick_columns"), picks_before + 1, "{what}: one pick");
            assert_eq!(ran("gather_indexed"), gathers, "{what}: no staged gather");
            let got = data(tensor(x.clone(), [rows, cols], &d).scatter(
                1,
                ints(idx.clone(), [rows, 1], &d),
                tensor(v.clone(), [rows, 1], &d),
                burn::tensor::IndexingUpdateOp::Add,
            ));
            let want = host_scatter_add(&x, cols, &idx, &device_sum(&x, &v, rows, cols, &d));
            if let Some((i, g, w)) = first_diff(&got, &want) {
                let (r, c) = (i / cols, i % cols);
                panic!(
                    "{what}: scatter-add at ({r}, {c}): {g:#010x} for {w:#010x}; x {:#010x}, v \
                     {:#010x}, index {}",
                    x[i].to_bits(),
                    v[r].to_bits(),
                    idx[r]
                );
            }
            assert_eq!(ran("gather_indexed"), gathers, "{what}: no staged gather");
        }
        // Rank 3: the rows are the leading dimensions.
        let (b, t, c) = (2, 17, 12);
        let x = floats(9, b * t * c, true);
        let v = floats(10, b * t, true);
        let idx = picks(b * t, c);
        let got =
            data(tensor(x.clone(), [b, t, c], &d).gather(2, ints(idx.clone(), [b, t, 1], &d)));
        assert_eq!(
            bits(&got),
            bits(&host_gather(&x, c, &idx)),
            "rank 3: gather"
        );
        let got = data(tensor(x.clone(), [b, t, c], &d).scatter(
            2,
            ints(idx.clone(), [b, t, 1], &d),
            tensor(v.clone(), [b, t, 1], &d),
            burn::tensor::IndexingUpdateOp::Add,
        ));
        assert_eq!(
            bits(&got),
            bits(&host_scatter_add(
                &x,
                c,
                &idx,
                &device_sum(&x, &v, b * t, c, &d)
            )),
            "rank 3: scatter-add"
        );
    });
}

#[test]
fn an_index_outside_the_row_is_domain() {
    for bad in [10, -1] {
        with_device(Config::default(), |d| {
            let x = tensor(floats(3, 4 * 10, false), [4, 10], &d);
            let idx = ints(vec![1, bad, 2, 3], [4, 1], &d);
            let err = x
                .clone()
                .gather(1, idx)
                .try_into_data()
                .expect_err("a gather past the row")
                .to_string();
            assert!(err.contains("code 11"), "gather at {bad}: {err}");
        });
        with_device(Config::default(), |d| {
            let x = tensor(floats(3, 4 * 10, false), [4, 10], &d);
            let idx = ints(vec![1, bad, 2, 3], [4, 1], &d);
            let err = x
                .scatter(
                    1,
                    idx,
                    tensor(vec![1.0; 4], [4, 1], &d),
                    burn::tensor::IndexingUpdateOp::Add,
                )
                .try_into_data()
                .expect_err("a scatter-add past the row")
                .to_string();
            assert!(err.contains("code 11"), "scatter-add at {bad}: {err}");
        });
    }
}

/// The indices the card finds (an argmax), captured once and replayed over other inputs.
#[test]
fn a_trace_replays_both_with_the_indices_it_finds() {
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            let (rows, cols) = (40, 10);
            let table = floats(5, rows * cols, false);
            let w = tensor(table.clone(), [rows, cols], &d);
            let v = floats(6, rows, false);
            let vt = tensor(v.clone(), [rows, 1], &d);
            let x0 = floats(7, rows * cols, false);
            let sum = device_sum(&table, &v, rows, cols, &d);
            let x = tensor(x0.clone(), [rows, cols], &d);
            let TensorPrimitive::Float(input) = x.clone().into_primitive() else {
                unreachable!()
            };
            let (trace, first) = burn_tt::Trace::capture(&input, || {
                let idx = x.clone().argmax(1);
                let picked = w.clone().gather(1, idx.clone());
                let added =
                    w.clone()
                        .scatter(1, idx, vt.clone(), burn::tensor::IndexingUpdateOp::Add);
                let TensorPrimitive::Float(out) =
                    Tensor::cat(vec![picked, added], 1).into_primitive()
                else {
                    unreachable!()
                };
                out
            })
            .unwrap();
            let want = |x: &[f32]| {
                let idx: Vec<i32> = (0..rows)
                    .map(|r| {
                        let row = &x[r * cols..(r + 1) * cols];
                        (0..cols).fold(0, |m, c| if row[c] > row[m] { c } else { m }) as i32
                    })
                    .collect();
                let g = host_gather(&table, cols, &idx);
                let s = host_scatter_add(&table, cols, &idx, &sum);
                (0..rows)
                    .flat_map(|r| {
                        std::iter::once(g[r]).chain(s[r * cols..(r + 1) * cols].iter().copied())
                    })
                    .collect::<Vec<f32>>()
            };
            assert_eq!(bits(&first), bits(&want(&x0)), "the capture");
            for seed in [8u64, 9] {
                let x = floats(seed, rows * cols, false);
                assert_eq!(
                    bits(&trace.run(x.clone()).unwrap()),
                    bits(&want(&x)),
                    "replay {seed}"
                );
            }
        },
    );
}
