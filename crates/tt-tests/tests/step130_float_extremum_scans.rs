//! Flex's `cummin`/`cummax` on the device (`ScanOp::{MinNan, MaxNan}`), bit for bit.
//!
//! The oracle below is a transcription of `burn-flex`'s `cummin_f32`/`cummax_f32`
//! (`ops/cumulative.rs`): the accumulator starts at `+inf` (`-inf`) and
//! `if val.is_nan() || val < acc { val } else { acc }` (`val > acc`). So a NaN
//! replaces the accumulator, a NaN accumulator is replaced only by another NaN
//! (the later one's bits win), and on equal values -- `+0` against `-0`
//! included -- the earlier element is kept. The older total-order `ScanOp::{Min,
//! Max}` (step79) differs on exactly the inputs the crafted columns hold, which
//! is the gate's negative control: it must fail them.
use burn::tensor::{Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, InputPayload, TracedInference, TtBackend, TtTensor};
use tt_kernels::{
    session::{Session, TileChoice},
    sfpu::scan::{reference_bits, ScanOp},
    tensor::Elem,
};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// Flex's per-lane scan, from its source.
fn flex_scan(minimum: bool, column: &[u32]) -> Vec<u32> {
    let mut acc = if minimum {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };
    column
        .iter()
        .map(|&bits| {
            let val = f32::from_bits(bits);
            acc = if val.is_nan() || if minimum { val < acc } else { val > acc } {
                val
            } else {
                acc
            };
            acc.to_bits()
        })
        .collect()
}

const NAN_A: u32 = 0x7fc1_2345;
const NAN_B: u32 = 0xffc5_4321;
const SNAN: u32 = 0x7f80_0001;
const NEG_SNAN: u32 = 0xff80_0001;
const SUBNORMAL: u32 = 0x0000_0001;
const NEG_SUBNORMAL: u32 = 0x8000_0001;

/// The words the random columns draw from.
fn specials() -> Vec<u32> {
    let mut v = vec![
        0,
        0x8000_0000,
        SUBNORMAL,
        NEG_SUBNORMAL,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x7f80_0000,
        0xff80_0000,
        NAN_A,
        NAN_B,
        SNAN,
        NEG_SNAN,
        0x7fff_ffff,
        0xffff_ffff,
        0x7f7f_ffff,
        0xff7f_ffff,
    ];
    v.extend([1.0f32, -1.0, 2.5, -0.25, 3.0, -7.0, 100.0].map(f32::to_bits));
    v
}

/// Columns the total-order scan gets wrong, then pseudo-random special words.
/// Column `c` of a `rows`-row matrix is `crafted[c]` followed by random words;
/// the crafted prefixes sit at rows 0.. and again straddling the 32-row tile
/// boundary, so the carry matters.
fn matrix(rows: usize, cols: usize, seed: u64) -> Vec<u32> {
    let f = f32::to_bits;
    let crafted: Vec<Vec<u32>> = vec![
        vec![f(0.0), f(-0.0)],
        vec![f(-0.0), f(0.0)],
        vec![f(1.0), NAN_A, f(0.0)],
        vec![NAN_B, f(1.0)],
        vec![f(5.0), NAN_A, f(-3.0), NAN_B, f(9.0)],
        vec![f(f32::INFINITY), f(f32::NEG_INFINITY), NAN_A],
        vec![SUBNORMAL, NEG_SUBNORMAL, f(0.0), f(-0.0)],
        vec![f(-0.0), f(0.0), f(-0.0), f(-1.0), f(0.0)],
    ];
    let table = specials();
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let mut m: Vec<u32> = (0..rows * cols)
        .map(|_| table[next() % table.len()])
        .collect();
    for (c, prefix) in crafted.iter().enumerate().filter(|(c, _)| *c < cols) {
        for (k, &w) in prefix.iter().enumerate() {
            m[k * cols + c] = w;
            // Again across the first tile boundary.
            if rows > 33 + prefix.len() {
                m[(30 + k) * cols + c] = w;
            }
        }
    }
    m
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

/// The device results of `ops` over `bits`, a `rows x cols` matrix, as words.
fn scans<T: tt_device::Transport>(
    s: &mut Session<T>,
    ops: &[ScanOp],
    bits: &[u32],
    rows: usize,
    cols: usize,
) -> Vec<Vec<u32>> {
    let input = s.upload_bits(bits, rows, cols, Elem::F32).unwrap();
    let mut results = Vec::new();
    for &op in ops {
        let t = s.scan(&input, op).unwrap();
        results.push(s.download_bits(&t).unwrap());
        s.free(t).unwrap();
    }
    assert_eq!(s.download_bits(&input).unwrap(), bits, "input untouched");
    s.free(input).unwrap();
    results
}

fn columns(m: &[u32], rows: usize, cols: usize, c: usize) -> Vec<u32> {
    (0..rows).map(|r| m[r * cols + c]).collect()
}

fn scan_by_column(minimum: bool, m: &[u32], rows: usize, cols: usize) -> Vec<u32> {
    let mut out = vec![0; rows * cols];
    for c in 0..cols {
        for (r, w) in flex_scan(minimum, &columns(m, rows, cols, c))
            .into_iter()
            .enumerate()
        {
            out[r * cols + c] = w;
        }
    }
    out
}

/// First differing column of `got` against `want`, if any.
fn first_difference(got: &[u32], want: &[u32], cols: usize) -> Option<String> {
    let i = (0..want.len()).find(|&i| got[i] != want[i])?;
    Some(format!(
        "row {} col {}: got {:#010x}, want {:#010x}",
        i / cols,
        i % cols,
        got[i],
        want[i]
    ))
}

#[test]
fn flex_order_scans_match_the_oracle_across_tiles_and_the_total_order_scan_does_not() {
    let (rows, cols) = (67, 35);
    let bits = matrix(rows, cols, 0x130);
    let want_min = scan_by_column(true, &bits, rows, cols);
    let want_max = scan_by_column(false, &bits, rows, cols);
    // The kernel crate's own reference (the interpreter's prediction) agrees
    // with the transcription of Flex's rule.
    assert_eq!(reference_bits(ScanOp::MinNan, &bits, rows, cols), want_min);
    assert_eq!(reference_bits(ScanOp::MaxNan, &bits, rows, cols), want_max);
    with_session(|s| {
        let got = scans(
            s,
            &[ScanOp::MinNan, ScanOp::MaxNan, ScanOp::Min, ScanOp::Max],
            &bits,
            rows,
            cols,
        );
        assert_eq!(
            first_difference(&got[0], &want_min, cols),
            None,
            "MinNan against Flex's cummin"
        );
        assert_eq!(
            first_difference(&got[1], &want_max, cols),
            None,
            "MaxNan against Flex's cummax"
        );
        // Negative control: the total-order scans fail the same gate, on
        // each crafted column.
        for (name, total, want) in [("Min", &got[2], &want_min), ("Max", &got[3], &want_max)] {
            for c in [0usize, 1, 2, 3, 6] {
                assert_ne!(
                    columns(total, rows, cols, c),
                    columns(want, rows, cols, c),
                    "total-order {name} must differ from Flex on crafted column {c}"
                );
            }
        }
    });
    // The inputs are not trivially fixed points of the scan.
    assert_ne!(want_min, bits);
    assert_ne!(want_max, bits);
}

#[test]
fn hand_written_columns_follow_flex() {
    // The crafted columns, evaluated by hand from Flex's rule.
    let f = f32::to_bits;
    assert_eq!(flex_scan(true, &[f(0.0), f(-0.0)]), [f(0.0), f(0.0)]);
    assert_eq!(flex_scan(true, &[f(-0.0), f(0.0)]), [f(-0.0), f(-0.0)]);
    assert_eq!(flex_scan(false, &[f(-0.0), f(0.0)]), [f(-0.0), f(-0.0)]);
    assert_eq!(
        flex_scan(true, &[f(1.0), NAN_A, f(0.0)]),
        [f(1.0), NAN_A, NAN_A]
    );
    assert_eq!(flex_scan(true, &[NAN_B, f(1.0)]), [NAN_B, NAN_B]);
    assert_eq!(
        flex_scan(false, &[f(5.0), NAN_A, f(-3.0), NAN_B, f(9.0)]),
        [f(5.0), NAN_A, NAN_A, NAN_B, NAN_B]
    );
    // The first element against the identity: +inf is not below +inf.
    assert_eq!(flex_scan(true, &[f(f32::INFINITY)]), [f(f32::INFINITY)]);
}

/// A scan's output claims no padding (`Pad::Undefined`): its ragged rows continue the
/// scan over the input's zeros. A reduction after it must not see them.
#[test]
fn scan_padding_is_undefined_and_a_following_reduction_ignores_it() {
    use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
    use tt_kernels::tensor::Pad;
    let (rows, cols) = (67, 35);
    // Every value negative: a leaked zero pad row would win the maximum.
    let bits: Vec<u32> = (0..rows * cols)
        .map(|i| (-1.0f32 - (i % 13) as f32).to_bits())
        .collect();
    with_session(|s| {
        let input = s.upload_bits(&bits, rows, cols, Elem::F32).unwrap();
        let scanned = s.scan(&input, ScanOp::MaxNan).unwrap();
        let max = s.reduce(&scanned, ReduceOp::Max, Axis::Rows).unwrap();
        let last = scan_by_column(false, &bits, rows, cols)[(rows - 1) * cols..].to_vec();
        let got = s.download_bits(&max).unwrap();
        assert_eq!(got[..cols], last[..], "the maximum over a ragged scan");
        assert!(last.iter().all(|&w| f32::from_bits(w) < 0.0));
        assert_eq!(scanned.pad(), Pad::Undefined);
        s.free(max).unwrap();
        s.free(scanned).unwrap();
        s.free(input).unwrap();
    });
}

fn cumulative_oracle(minimum: bool, bits: &[u32], shape: &[usize], dim: usize) -> Vec<u32> {
    let inner: usize = shape[dim + 1..].iter().product();
    let n = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let mut out = vec![0; bits.len()];
    for o in 0..outer {
        for i in 0..inner {
            let idx = |k: usize| (o * n + k) * inner + i;
            let col: Vec<u32> = (0..n).map(|k| bits[idx(k)]).collect();
            for (k, w) in flex_scan(minimum, &col).into_iter().enumerate() {
                out[idx(k)] = w;
            }
        }
    }
    out
}

fn to_bits_vec(data: TensorData) -> Vec<u32> {
    data.to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn float_data(bits: &[u32], shape: &[usize]) -> TensorData {
    TensorData::new(
        bits.iter().map(|&b| f32::from_bits(b)).collect::<Vec<_>>(),
        shape.to_vec(),
    )
}

fn check_rank<const D: usize>(d: &burn_tt::TtDevice, shape: [usize; D], seed: u64) {
    let n: usize = shape.iter().product();
    let bits = matrix(n, 1, seed);
    let data = float_data(&bits, &shape);
    for dim in 0..D {
        for minimum in [true, false] {
            let want = cumulative_oracle(minimum, &bits, &shape, dim);
            let flex = {
                let t = Tensor::<Flex, D>::from_data(data.clone(), &FlexDevice);
                to_bits_vec(
                    if minimum {
                        t.cummin(dim)
                    } else {
                        t.cummax(dim)
                    }
                    .into_data(),
                )
            };
            assert_eq!(
                flex, want,
                "the oracle is Flex's: {shape:?} dim {dim} min={minimum}"
            );
            let x = Tensor::<TtBackend, D>::from_data(data.clone(), d);
            let before = tensor_traffic();
            let (y, report) = burn_tt::with_report(|| {
                if minimum {
                    x.clone().cummin(dim)
                } else {
                    x.clone().cummax(dim)
                }
            });
            assert_native_model(&report);
            let name = if minimum {
                "float_cummin"
            } else {
                "float_cummax"
            };
            let op = report.op(name).unwrap();
            assert_eq!((op.downloads, op.staged, op.on_host), (0, 0, 0), "{name}");
            assert_eq!(tensor_traffic().downloads, before.downloads);
            assert!(
                primitive(y.clone()).computed_on_device(),
                "{name} {shape:?} dim {dim} stayed on the device"
            );
            let got = to_bits_vec(y.into_data());
            assert_eq!(
                got.iter()
                    .zip(&want)
                    .position(|(g, w)| g != w)
                    .map(|i| (i, got[i], want[i])),
                None,
                "{name} {shape:?} dim {dim}"
            );
        }
    }
}

#[test]
fn burn_cummin_cummax_match_flex_on_every_axis_and_rank() {
    with_device(Config::default(), |d| {
        check_rank::<1>(&d, [67], 1);
        check_rank::<2>(&d, [67, 35], 2);
        check_rank::<2>(&d, [31, 33], 3);
        check_rank::<3>(&d, [3, 35, 5], 4);
        check_rank::<4>(&d, [2, 3, 34, 4], 5);
    });
}

fn primitive<const D: usize>(t: Tensor<TtBackend, D>) -> TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!("a float tensor"),
    }
}

#[test]
fn ragged_scan_feeds_a_following_reduction_and_replays_a_trace_on_new_inputs() {
    with_device(Config::default(), |d| {
        // Finite integer-valued data: the following sum is exact, so the
        // comparison is against plain Rust.
        let make = |seed: u64, rows: usize, cols: usize| -> Vec<f32> {
            let mut s = seed | 1;
            (0..rows * cols)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 40) % 21) as f32 - 10.0
                })
                .collect()
        };
        for (rows, cols) in [(32, 32), (40, 33)] {
            let forward = |x: Tensor<TtBackend, 2>| {
                // A ragged producer into a reduction and into a matmul.
                let lo = x.clone().cummin(0);
                let hi = x.cummax(0);
                (
                    lo.clone().sum_dim(0),
                    hi.clone().max_dim(1),
                    lo.matmul(hi.transpose()),
                )
            };
            let first = make(1, rows, cols);
            let x =
                Tensor::<TtBackend, 2>::from_data(TensorData::new(first.clone(), [rows, cols]), &d);
            let xp = primitive(x.clone());
            let want = |values: &[f32]| -> [Vec<f32>; 3] {
                let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
                let lo: Vec<f32> = cumulative_oracle(true, &bits, &[rows, cols], 0)
                    .into_iter()
                    .map(f32::from_bits)
                    .collect();
                let hi: Vec<f32> = cumulative_oracle(false, &bits, &[rows, cols], 0)
                    .into_iter()
                    .map(f32::from_bits)
                    .collect();
                let sum: Vec<f32> = (0..cols)
                    .map(|c| (0..rows).map(|r| lo[r * cols + c]).sum())
                    .collect();
                let max: Vec<f32> = (0..rows)
                    .map(|r| {
                        hi[r * cols..(r + 1) * cols]
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .collect();
                let mut prod = vec![0.0f32; rows * rows];
                for i in 0..rows {
                    for j in 0..rows {
                        prod[i * rows + j] =
                            (0..cols).map(|k| lo[i * cols + k] * hi[j * cols + k]).sum();
                    }
                }
                [sum, max, prod]
            };
            let (outs, report) = burn_tt::with_report(|| forward(x.clone()));
            assert_native_model(&report);
            let got = [
                outs.0.clone().into_data().to_vec::<f32>().unwrap(),
                outs.1.clone().into_data().to_vec::<f32>().unwrap(),
                outs.2.clone().into_data().to_vec::<f32>().unwrap(),
            ];
            assert_eq!(got, want(&first), "{rows}x{cols} fresh chain");

            // Capture once, replay on changed inputs, no uploads in between.
            let (trace, captured) = TracedInference::capture(&[&xp], || {
                let (a, b, c) = forward(x.clone());
                vec![primitive(a), primitive(b), primitive(c)]
            })
            .unwrap();
            assert_eq!(captured[0].as_f32().unwrap(), &want(&first)[0][..]);
            for seed in [2u64, 3] {
                let next = make(seed, rows, cols);
                let out = trace.run(vec![InputPayload::F32(next.clone())]).unwrap();
                let w = want(&next);
                for k in 0..3 {
                    assert_eq!(
                        out[k].as_f32().unwrap(),
                        &w[k][..],
                        "{rows}x{cols} replay {seed} output {k}"
                    );
                }
            }
        }
    });
}
