//! Wrapping and signed integer scans (`ScanOp::{ISum, IProd, IMin, IMax}`), against
//! plain Rust, and `burn-tt`'s `int_cumsum`/`int_cumprod`/`int_cummin`/`int_cummax` over
//! every axis and rank.
//!
//! Sums and products wrap modulo 2^32 (Flex's `acc + val` wraps in a release build and
//! panics in a debug one, so Flex is compared only where nothing overflows); the minimum
//! and maximum are signed, so `i32::MIN` is the smallest word. The identities are
//! 0, 1, `i32::MAX` and `i32::MIN`: a column holding exactly the identity must scan to
//! itself.
use burn::tensor::{Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, InputPayload, TracedInference, TtBackend, TtTensor};
use tt_kernels::{
    session::{Session, TileChoice},
    sfpu::scan::{reference_bits, ScanOp},
    tensor::Elem,
};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Kind {
    Sum,
    Prod,
    Min,
    Max,
}

const KINDS: [Kind; 4] = [Kind::Sum, Kind::Prod, Kind::Min, Kind::Max];

fn scan_op(k: Kind) -> ScanOp {
    match k {
        Kind::Sum => ScanOp::ISum,
        Kind::Prod => ScanOp::IProd,
        Kind::Min => ScanOp::IMin,
        Kind::Max => ScanOp::IMax,
    }
}

/// The oracle: wrapping two's-complement `i32` arithmetic and `Ord`.
fn scan1(kind: Kind, column: &[i32]) -> Vec<i32> {
    let mut acc = match kind {
        Kind::Sum => 0,
        Kind::Prod => 1,
        Kind::Min => i32::MAX,
        Kind::Max => i32::MIN,
    };
    column
        .iter()
        .map(|&v| {
            acc = match kind {
                Kind::Sum => acc.wrapping_add(v),
                Kind::Prod => acc.wrapping_mul(v),
                Kind::Min => acc.min(v),
                Kind::Max => acc.max(v),
            };
            acc
        })
        .collect()
}

fn along(kind: Kind, values: &[i32], shape: &[usize], dim: usize) -> Vec<i32> {
    let inner: usize = shape[dim + 1..].iter().product();
    let n = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let mut out = vec![0; values.len()];
    for o in 0..outer {
        for i in 0..inner {
            let idx = |k: usize| (o * n + k) * inner + i;
            let column: Vec<i32> = (0..n).map(|k| values[idx(k)]).collect();
            for (k, w) in scan1(kind, &column).into_iter().enumerate() {
                out[idx(k)] = w;
            }
        }
    }
    out
}

/// Words that stress a 32-bit integer ALU: extremes, around the 16-bit limbs `IProd`
/// multiplies, and a pseudo-random word stream.
fn wild(n: usize, seed: u64) -> Vec<i32> {
    let table = [
        0,
        1,
        -1,
        2,
        -2,
        i32::MIN,
        i32::MAX,
        65535,
        65536,
        -65536,
        0x1234_5678,
        -0x1234_5678,
        0x7fff_0001,
        3,
        -3,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let w = (s >> 32) as u32;
            match i % 3 {
                0 => table[w as usize % table.len()],
                1 => w as i32,
                _ => (w | 1) as i32,
            }
        })
        .collect()
}

/// Small words: no sum or product of eleven of them overflows.
fn small(n: usize, seed: u64) -> Vec<i32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) % 7) as i32 - 3
        })
        .collect()
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    tt_ttsim::fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    })
    .unwrap();
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    tt_ttsim::fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    })
    .unwrap();
}

fn matrix_oracle(kind: Kind, values: &[i32], rows: usize, cols: usize) -> Vec<u32> {
    along(kind, values, &[rows, cols], 0)
        .into_iter()
        .map(|v| v as u32)
        .collect()
}

#[test]
fn integer_scans_wrap_compare_signed_and_carry_across_tiles() {
    // 67 rows cross two tile boundaries and leave a ragged last tile; 35 columns
    // leave a ragged column tile.
    let (rows, cols) = (67, 35);
    let mut values = wild(rows * cols, 0x131);
    // A column of exactly each identity, and one that starts at the other extreme.
    for r in 0..rows {
        values[r * cols] = 0; // sum identity
        values[r * cols + 1] = 1; // product identity
        values[r * cols + 2] = i32::MAX; // min identity
        values[r * cols + 3] = i32::MIN; // max identity
        values[r * cols + 4] = if r < 40 { -1 } else { 1 }; // sign change across a tile
    }
    let bits: Vec<u32> = values.iter().map(|&v| v as u32).collect();
    with_session(|s| {
        let input = s.upload_bits(&bits, rows, cols, Elem::I32).unwrap();
        for kind in KINDS {
            let want = matrix_oracle(kind, &values, rows, cols);
            // The interpreter-level prediction and the oracle agree.
            assert_eq!(reference_bits(scan_op(kind), &bits, rows, cols), want);
            let out = s.scan(&input, scan_op(kind)).unwrap();
            let got = s.download_bits(&out).unwrap();
            let diff = (0..want.len()).find(|&i| got[i] != want[i]);
            assert_eq!(
                diff.map(|i| (i / cols, i % cols, got[i] as i32, want[i] as i32)),
                None,
                "{kind:?}"
            );
            assert_ne!(want, bits, "{kind:?}: not an identity copy");
            s.free(out).unwrap();
        }
        assert_eq!(s.download_bits(&input).unwrap(), bits);
        // A float scan refuses an I32 tensor and the reverse: no reinterpretation.
        assert!(s.scan(&input, ScanOp::MinNan).is_err());
        let f = s.upload_bits(&bits, rows, cols, Elem::F32).unwrap();
        assert!(s.scan(&f, ScanOp::ISum).is_err());
        s.free(f).unwrap();
        s.free(input).unwrap();
    });
}

fn burn_check<const D: usize>(d: &burn_tt::TtDevice, shape: [usize; D], seed: u64) {
    let n: usize = shape.iter().product();
    for (wrapping, values) in [(true, wild(n, seed)), (false, small(n, seed))] {
        let data = TensorData::new(values.clone(), shape.to_vec());
        for dim in 0..D {
            for kind in KINDS {
                let want = along(kind, &values, &shape, dim);
                // Flex is the oracle where it cannot overflow (and for min/max always).
                if !wrapping || matches!(kind, Kind::Min | Kind::Max) {
                    let t = Tensor::<Flex, D, Int>::from_data(data.clone(), &FlexDevice);
                    let flex = match kind {
                        Kind::Sum => t.cumsum(dim),
                        Kind::Prod => t.cumprod(dim),
                        Kind::Min => t.cummin(dim),
                        Kind::Max => t.cummax(dim),
                    };
                    assert_eq!(
                        flex.into_data().to_vec::<i32>().unwrap(),
                        want,
                        "Flex is the oracle: {kind:?} {shape:?} dim {dim}"
                    );
                }
                let x = Tensor::<TtBackend, D, Int>::from_data(data.clone(), d);
                let before = tensor_traffic();
                let (y, report) = burn_tt::with_report(|| match kind {
                    Kind::Sum => x.clone().cumsum(dim),
                    Kind::Prod => x.clone().cumprod(dim),
                    Kind::Min => x.clone().cummin(dim),
                    Kind::Max => x.clone().cummax(dim),
                });
                assert_native_model(&report);
                let name = match kind {
                    Kind::Sum => "int_cumsum",
                    Kind::Prod => "int_cumprod",
                    Kind::Min => "int_cummin",
                    Kind::Max => "int_cummax",
                };
                let op = report.op(name).unwrap();
                assert_eq!((op.downloads, op.staged, op.on_host), (0, 0, 0), "{name}");
                assert_eq!(tensor_traffic().downloads, before.downloads);
                assert!(
                    y.clone().into_primitive().computed_on_device(),
                    "{name} {shape:?} dim {dim} stayed on the device"
                );
                let got = y.into_data().to_vec::<i32>().unwrap();
                assert_eq!(
                    got.iter()
                        .zip(&want)
                        .position(|(g, w)| g != w)
                        .map(|i| (i, got[i], want[i])),
                    None,
                    "{name} {shape:?} dim {dim} wrapping={wrapping}"
                );
            }
        }
    }
}

#[test]
fn burn_integer_scans_match_the_oracle_on_every_axis_and_rank() {
    with_device(Config::default(), |d| {
        burn_check::<1>(&d, [67], 1);
        burn_check::<2>(&d, [67, 35], 2);
        burn_check::<2>(&d, [11, 33], 3);
        burn_check::<3>(&d, [3, 35, 5], 4);
        burn_check::<4>(&d, [2, 3, 11, 4], 5);
    });
}

fn primitive<const D: usize>(t: Tensor<TtBackend, D, Int>) -> TtTensor {
    t.into_primitive()
}

#[test]
fn ragged_integer_scan_feeds_reductions_and_replays_a_trace_on_new_inputs() {
    with_device(Config::default(), |d| {
        for (rows, cols) in [(32, 32), (40, 33)] {
            let forward = |x: Tensor<TtBackend, 2, Int>| {
                // Ragged scans into integer reductions: the sum of a cumulative
                // maximum (all-negative inputs make a leaked zero pad row visible)
                // and the maximum of a wrapping cumulative sum.
                (
                    x.clone().cummax(0).sum_dim(0),
                    x.clone().cumsum(0).max_dim(1),
                    x.cummin(0).min_dim(0),
                )
            };
            let want = |values: &[i32]| -> [Vec<i32>; 3] {
                let hi = along(Kind::Max, values, &[rows, cols], 0);
                let sums = along(Kind::Sum, values, &[rows, cols], 0);
                let lo = along(Kind::Min, values, &[rows, cols], 0);
                [
                    (0..cols)
                        .map(|c| (0..rows).fold(0i32, |a, r| a.wrapping_add(hi[r * cols + c])))
                        .collect(),
                    (0..rows)
                        .map(|r| *sums[r * cols..(r + 1) * cols].iter().max().unwrap())
                        .collect(),
                    (0..cols)
                        .map(|c| (0..rows).map(|r| lo[r * cols + c]).min().unwrap())
                        .collect(),
                ]
            };
            let negative = |seed: u64| -> Vec<i32> {
                wild(rows * cols, seed)
                    .into_iter()
                    .map(|v| -(v & 0x3fff_ffff) - 1)
                    .collect()
            };
            let first = negative(1);
            let x = Tensor::<TtBackend, 2, Int>::from_data(
                TensorData::new(first.clone(), [rows, cols]),
                &d,
            );
            let xp = primitive(x.clone());
            let (outs, report) = burn_tt::with_report(|| forward(x.clone()));
            assert_native_model(&report);
            let read = |t: Tensor<TtBackend, 2, Int>| t.into_data().to_vec::<i32>().unwrap();
            let w = want(&first);
            assert_eq!(read(outs.0), w[0], "{rows}x{cols} sum of cummax");
            assert_eq!(read(outs.1), w[1], "{rows}x{cols} max of cumsum");
            assert_eq!(read(outs.2), w[2], "{rows}x{cols} min of cummin");

            let (trace, captured) = TracedInference::capture(&[&xp], || {
                let (a, b, c) = forward(x.clone());
                vec![primitive(a), primitive(b), primitive(c)]
            })
            .unwrap();
            for k in 0..3 {
                let got: Vec<i32> = captured[k]
                    .as_bits()
                    .unwrap()
                    .iter()
                    .map(|&b| b as i32)
                    .collect();
                assert_eq!(got, w[k], "{rows}x{cols} capture output {k}");
            }
            for seed in [2u64, 3] {
                let next = negative(seed);
                let payload = next.iter().map(|&v| v as u32).collect();
                let out = trace.run(vec![InputPayload::Bits(payload)]).unwrap();
                let w = want(&next);
                for k in 0..3 {
                    let got: Vec<i32> = out[k]
                        .as_bits()
                        .unwrap()
                        .iter()
                        .map(|&b| b as i32)
                        .collect();
                    assert_eq!(got, w[k], "{rows}x{cols} replay {seed} output {k}");
                }
            }
        }
    });
}
