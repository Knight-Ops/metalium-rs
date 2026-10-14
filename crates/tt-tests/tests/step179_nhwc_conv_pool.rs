//! Chunk-free convolution and max pooling (roadmap Track 2 item 2).
//!
//! `conv2d` was one product a group and 32 output positions, each with a
//! host-built patch gather (a datum a mover entry), and `max_pool2d` 32
//! windows at a time; ResNet-18 took 16 s. Now a convolution is one tap select
//! (`Session::select_taps`, B's `SELECT_ROWS` over the NHWC table, indices that
//! depend on the shapes alone and are kept) and one product, and its output is
//! an NCHW view of the `[P, O]` result that element-wise ops, the next
//! convolution and max pooling read in place; a max pool is one select and a
//! fold over its taps.
//!
//! The claims:
//! - each convolution equals a scalar oracle exactly, on small integers (exact
//!   in any summation order; `step85`'s `permuted_fractional` case bounds
//!   fractional data with the Matrix Unit's derived error): a 7x7 stride-2
//!   first layer over 3 channels (a ragged tap, padding, the zero row), 3x3
//!   over 64 channels, a table of whole tile rows (its zero row a copy's), a
//!   strided 1x1, dilation, a batch of two and a bias;
//! - max pooling equals the host's window scan bit for bit -- the first NaN,
//!   else the first maximum (`-0` and `+0` tie, so the first stays), payloads,
//!   infinities and subnormals kept -- with padding before a window's first
//!   valid datum, dilation and ceil mode;
//! - a ResNet block (conv, BatchNorm, relu, max pool, conv, add) runs with no
//!   repack once its weights and geometry are kept (the input's relayout aside),
//!   nothing downloaded and nothing computed on the host;
//! - a trace of convolution and max pooling replays changed inputs.
//!
//! Watched failing (2026-10-14, ttsim): a tap index off by one column, the
//! weights in (channel, tap) order, a padding tap reading row 0 instead of the
//! zero row, `>=` for `>` in the fold, and a fold that ignores NaNs.

use burn::tensor::ops::ConvOptions;
use burn::tensor::{module, Tensor, TensorData, TensorPrimitive};
use burn_tt::{device_time, tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// How many times the server ran a kind of op in this process so far.
fn ran(kind: &str) -> u64 {
    device_time()
        .iter()
        .find(|(k, _, _)| *k == kind)
        .map_or(0, |(_, n, _)| *n)
}

fn resident<const D: usize>(x: &Tensor<TtBackend, D>) {
    let TensorPrimitive::Float(x) = x.clone().into_primitive() else {
        unreachable!()
    };
    assert!(x.computed_on_device());
}

/// The scalar convolution, in `f64` (exact for these integers).
fn oracle(
    x: &[f32],
    xs: [usize; 4],
    w: &[f32],
    ws: [usize; 4],
    bias: Option<&[f32]>,
    o: &ConvOptions<2>,
) -> (Vec<f32>, [usize; 4]) {
    let out = |a: usize| {
        (xs[a + 2] + 2 * o.padding[a] - o.dilation[a] * (ws[a + 2] - 1) - 1) / o.stride[a] + 1
    };
    let ys = [xs[0], ws[0], out(0), out(1)];
    let mut y = vec![0.0; ys.iter().product()];
    for n in 0..ys[0] {
        for oc in 0..ys[1] {
            for r in 0..ys[2] {
                for c in 0..ys[3] {
                    let mut s = bias.map_or(0.0, |b| b[oc] as f64);
                    for ic in 0..xs[1] {
                        for kr in 0..ws[2] {
                            for kc in 0..ws[3] {
                                let ir = (r * o.stride[0] + kr * o.dilation[0]) as isize
                                    - o.padding[0] as isize;
                                let icl = (c * o.stride[1] + kc * o.dilation[1]) as isize
                                    - o.padding[1] as isize;
                                if ir < 0
                                    || icl < 0
                                    || ir >= xs[2] as isize
                                    || icl >= xs[3] as isize
                                {
                                    continue;
                                }
                                let xi =
                                    ((n * xs[1] + ic) * xs[2] + ir as usize) * xs[3] + icl as usize;
                                let wi = ((oc * ws[1] + ic) * ws[2] + kr) * ws[3] + kc;
                                s += x[xi] as f64 * w[wi] as f64;
                            }
                        }
                    }
                    y[((n * ys[1] + oc) * ys[2] + r) * ys[3] + c] = s as f32;
                }
            }
        }
    }
    (y, ys)
}

#[test]
fn convolutions_are_the_scalar_oracle_in_one_select_and_one_product() {
    with_device(Config::default(), |d| {
        let cases: [([usize; 4], [usize; 4], ConvOptions<2>, bool); 6] = [
            // ResNet's first layer, small: 3 channels (a ragged tap of a tile
            // column), 49 taps over two index tile rows, padding.
            (
                [1, 3, 20, 22],
                [8, 3, 7, 7],
                ConvOptions::new([2, 2], [3, 3], [1, 1], 1),
                false,
            ),
            // 3x3 over 64 channels: two tile columns a tap.
            (
                [1, 64, 9, 10],
                [16, 64, 3, 3],
                ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
                false,
            ),
            // 64 table rows: no padding row, so the zero row is a copy's.
            (
                [1, 32, 8, 8],
                [8, 32, 3, 3],
                ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
                true,
            ),
            // A strided 1x1 (a block's shortcut) over a batch of two.
            (
                [2, 40, 9, 7],
                [24, 40, 1, 1],
                ConvOptions::new([2, 2], [0, 0], [1, 1], 1),
                false,
            ),
            // Dilation, unequal strides and padding, and a bias.
            (
                [2, 5, 11, 9],
                [7, 5, 3, 2],
                ConvOptions::new([2, 1], [2, 1], [2, 3], 1),
                true,
            ),
            // No padding at all: no zero row.
            (
                [1, 33, 6, 6],
                [4, 33, 2, 2],
                ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
                false,
            ),
        ];
        for (xs, ws, options, with_bias) in cases {
            let what = format!("x {xs:?}, w {ws:?}, {options:?}");
            let xv: Vec<f32> = (0..xs.iter().product())
                .map(|i: usize| (i * 7 % 9) as f32 - 4.0)
                .collect();
            let wv: Vec<f32> = (0..ws.iter().product())
                .map(|i: usize| (i * 5 % 7) as f32 - 3.0)
                .collect();
            let bv: Vec<f32> = (0..ws[0]).map(|i| i as f32 - 3.0).collect();
            let x = Tensor::<TtBackend, 4>::from_data(TensorData::new(xv.clone(), xs), &d);
            let w = Tensor::<TtBackend, 4>::from_data(TensorData::new(wv.clone(), ws), &d);
            let b = with_bias.then(|| {
                Tensor::<TtBackend, 1>::from_data(TensorData::new(bv.clone(), [ws[0]]), &d)
            });
            let (selects, before) = (ran("select_taps"), tensor_traffic());
            let (y, report) = burn_tt::with_report(|| module::conv2d(x, w, b, options.clone()));
            assert_native_model(&report);
            resident(&y);
            assert_eq!(tensor_traffic().downloads, before.downloads, "{what}");
            assert_eq!(ran("select_taps"), selects + 1, "{what}: one select");
            let (want, ys) = oracle(&xv, xs, &wv, ws, with_bias.then_some(&bv[..]), &options);
            assert_eq!(y.dims(), ys, "{what}");
            assert_eq!(y.into_data().to_vec::<f32>().unwrap(), want, "{what}");
        }
    });
}

/// The host's window scan: the first NaN, else the first datum greater than
/// every one before it (so of `-0` and `+0` the first stays).
fn pool_oracle(
    x: &[u32],
    xs: [usize; 4],
    k: [usize; 2],
    s: [usize; 2],
    p: [usize; 2],
    dl: [usize; 2],
    [oh, ow]: [usize; 2],
) -> Vec<u32> {
    let mut y = Vec::new();
    for plane in 0..xs[0] * xs[1] {
        for r in 0..oh {
            for c in 0..ow {
                let mut best: Option<u32> = None;
                for kr in 0..k[0] {
                    for kc in 0..k[1] {
                        let ir = (r * s[0] + kr * dl[0]) as isize - p[0] as isize;
                        let ic = (c * s[1] + kc * dl[1]) as isize - p[1] as isize;
                        if ir < 0 || ic < 0 || ir >= xs[2] as isize || ic >= xs[3] as isize {
                            continue;
                        }
                        let v = x[(plane * xs[2] + ir as usize) * xs[3] + ic as usize];
                        best = Some(match best {
                            None => v,
                            Some(b) if f32::from_bits(b).is_nan() => b,
                            Some(_) if f32::from_bits(v).is_nan() => v,
                            Some(b) if f32::from_bits(v) > f32::from_bits(b) => v,
                            Some(b) => b,
                        });
                    }
                }
                y.push(best.expect("a window with a datum"));
            }
        }
    }
    y
}

/// A max pool's input shape, kernel, stride, padding, dilation and ceil mode.
type PoolCase = (
    [usize; 4],
    [usize; 2],
    [usize; 2],
    [usize; 2],
    [usize; 2],
    bool,
);

#[test]
fn max_pooling_is_the_host_s_window_scan_bit_for_bit() {
    const SPECIAL: [u32; 10] = [
        0x8000_0000, // -0.0
        0x0000_0000, // +0.0
        0x0000_0001, // the least subnormal
        0x8000_0001, // its negative
        0x7fc1_2345, // a quiet NaN with a payload
        0xffc0_0001, // a negative NaN with a payload
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7f7f_ffff, // f32::MAX
        0x807f_ffff, // the most negative subnormal
    ];
    with_device(Config::default(), |d| {
        // (shape, kernel, stride, padding, dilation, ceil): ResNet's 3x3 s2 p1
        // (padding before a window's first valid datum), dilation, ceil mode
        // with a partial last window, and 64 channels over a batch of two.
        let cases: [PoolCase; 4] = [
            ([1, 32, 9, 11], [3, 3], [2, 2], [1, 1], [1, 1], false),
            ([1, 32, 10, 9], [2, 3], [1, 2], [1, 1], [2, 1], false),
            ([2, 32, 7, 8], [3, 2], [2, 3], [0, 1], [1, 1], true),
            ([2, 64, 6, 6], [3, 3], [2, 2], [1, 1], [1, 1], false),
        ];
        for (seed, (xs, k, s, p, dl, ceil)) in cases.into_iter().enumerate() {
            let what = format!("{xs:?} k{k:?} s{s:?} p{p:?} d{dl:?} ceil {ceil}");
            let n: usize = xs.iter().product();
            let mut state = seed as u64 * 7919 + 1;
            let bits: Vec<u32> = (0..n)
                .map(|i| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    match i % 4 {
                        // Ties and specials often, the rest small integers.
                        0 => SPECIAL[(state >> 40) as usize % SPECIAL.len()],
                        _ => (((state >> 33) % 7) as f32 - 3.0).to_bits(),
                    }
                })
                .collect();
            let x = Tensor::<TtBackend, 4>::from_data(
                TensorData::new(bits.iter().map(|&b| f32::from_bits(b)).collect(), xs),
                &d,
            );
            let (selects, before) = (ran("select_taps"), tensor_traffic());
            let (y, report) = burn_tt::with_report(|| module::max_pool2d(x, k, s, p, dl, ceil));
            assert_native_model(&report);
            resident(&y);
            assert_eq!(tensor_traffic().downloads, before.downloads, "{what}");
            assert_eq!(ran("select_taps"), selects + 1, "{what}: one select");
            let [_, _, oh, ow] = y.dims();
            let got: Vec<u32> = y
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let want = pool_oracle(&bits, xs, k, s, p, dl, [oh, ow]);
            assert_eq!(got.len(), want.len(), "{what}");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "{what}: output {i}: {g:#010x} vs {w:#010x}");
            }
        }
    });
}

/// A ResNet block on NHWC views: conv, BatchNorm, relu, max pool, conv, the
/// shortcut's add. The second run repacks only the input (its relayout into
/// the first table): the weights' relayouts and the tap indices are kept, and
/// every other op reads the views where they lie. Its values equal the first
/// run's.
#[test]
fn a_resnet_block_runs_on_views_with_no_repack_once_warm() {
    use burn::module::Module;
    use burn::nn::{BatchNorm, BatchNormConfig};
    with_device(Config::default(), |d| {
        let xs = [1, 3, 18, 18];
        let xv: Vec<f32> = (0..xs.iter().product())
            .map(|i: usize| (i * 7 % 9) as f32 - 4.0)
            .collect();
        let weights = |o: usize, c: usize, k: usize| {
            Tensor::<TtBackend, 4>::from_data(
                TensorData::new(
                    (0..o * c * k * k)
                        .map(|i| ((i * 5 % 7) as f32 - 3.0) / 8.0)
                        .collect::<Vec<_>>(),
                    [o, c, k, k],
                ),
                &d,
            )
        };
        let (w1, w2) = (weights(32, 3, 3), weights(32, 32, 3));
        let bn: BatchNorm<TtBackend> = BatchNormConfig::new(32).init(&d);
        let bn = bn.no_grad();
        let block = |x: Tensor<TtBackend, 4>| {
            let y = module::conv2d(
                x,
                w1.clone(),
                None,
                ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
            );
            let y = burn::tensor::activation::relu(bn.forward(y));
            let y = module::max_pool2d(y, [3, 3], [2, 2], [1, 1], [1, 1], false);
            let z = module::conv2d(
                y.clone(),
                w2.clone(),
                None,
                ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
            );
            burn::tensor::activation::relu(z + y)
        };
        let x = || Tensor::<TtBackend, 4>::from_data(TensorData::new(xv.clone(), xs), &d);
        let first = block(x()).into_data().to_vec::<f32>().unwrap();
        let input = x().to_device(&d);
        let (repacks, many, downloads) = (
            ran("repack"),
            ran("repack_many"),
            tensor_traffic().downloads,
        );
        let (y, report) = burn_tt::with_report(|| block(input));
        assert_native_model(&report);
        resident(&y);
        assert_eq!(tensor_traffic().downloads, downloads);
        assert_eq!(ran("repack"), repacks, "nothing repacked");
        assert_eq!(ran("repack_many"), many + 1, "but the input's relayout");
        assert_eq!(y.into_data().to_vec::<f32>().unwrap(), first);
    });
}

/// A trace of a convolution and a max pool over 32 channels replays changed
/// inputs: the tap indices and relaid weights are kept from an eager forward
/// before it, and the capture does not make them again.
#[test]
fn a_trace_of_conv_and_max_pool_replays_changed_inputs() {
    with_device(Config::default(), |d| {
        let xs = [1, 1, 10, 9];
        let x = Tensor::<TtBackend, 4>::ones(xs, &d).mul_scalar(1.0);
        let TensorPrimitive::Float(input) = x.clone().into_primitive() else {
            unreachable!()
        };
        let w = Tensor::<TtBackend, 4>::ones([32, 1, 3, 3], &d).to_device(&d);
        let forward = || {
            let y = module::conv2d(
                x.clone(),
                w.clone(),
                None,
                ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
            );
            module::max_pool2d(y, [3, 3], [2, 2], [1, 1], [1, 1], false)
        };
        // An eager forward first makes the relaid weights, which the capture then reads: it
        // repacks only its input (a 1-channel NCHW tensor relaid into its table).
        let eager = forward().into_data().to_vec::<f32>().unwrap();
        let (before, repacks) = (tensor_traffic(), ran("repack_many"));
        let ((trace, first), report) = burn_tt::with_report(|| {
            burn_tt::Trace::capture(&input, || {
                let TensorPrimitive::Float(y) = forward().into_primitive() else {
                    unreachable!()
                };
                y
            })
            .unwrap()
        });
        assert_eq!(
            ran("repack_many"),
            repacks + 1,
            "the capture relays its input, not the weights"
        );
        assert_eq!(first, eager);
        assert_native_model(&report);
        assert_eq!(tensor_traffic().uploads, before.uploads);
        // A 3x3 sum of a constant plane is 9 inside, 6 on an edge, 4 in a
        // corner; a 3x3 window's maximum over it touches the inside wherever
        // the window reaches row and column 1.
        let pooled = |v: f32| {
            let (h, w) = (10usize, 9usize);
            let conv = |r: usize, c: usize| {
                let n = |i: usize, len: usize| if i == 0 || i == len - 1 { 2.0 } else { 3.0 };
                n(r, h) * n(c, w) * v
            };
            let mut out = Vec::new();
            for _ in 0..32 {
                for r in 0..5 {
                    for c in 0..5 {
                        let mut best = f32::NEG_INFINITY;
                        for kr in 0..3 {
                            for kc in 0..3 {
                                let (ir, ic) =
                                    ((r * 2 + kr) as isize - 1, (c * 2 + kc) as isize - 1);
                                if ir >= 0 && ic >= 0 && (ir as usize) < h && (ic as usize) < w {
                                    best = best.max(conv(ir as usize, ic as usize));
                                }
                            }
                        }
                        out.push(best);
                    }
                }
            }
            out
        };
        assert_eq!(first, pooled(1.0));
        for v in [2.0, -1.0] {
            assert_eq!(trace.run(vec![v; 90]).unwrap(), pooled(v));
        }
    });
}
