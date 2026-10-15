//! Chunk-free convolution and max pooling over NHWC tables (roadmap Track 2,
//! item 2).
//!
//! A 2-D convolution's patch matrix `[P, taps x C]` -- row `p` an output
//! position, each tap's `C` channels side by side, Flex's im2col order -- is
//! made on the card by `Session::select_taps` from the activation's NHWC table
//! `[N H W, C]` and a resident `[taps, P]` index of table rows that depends on
//! the shapes alone (cached, [`geometry`]). The convolution is then one
//! product with the weights relaid to `[taps x C, O]` (kept on the weight,
//! `TtTensor::derived`), and its `[P, O]` result is the next layer's table:
//! the output is returned as an NCHW view of it ([`nhwc_strides`]), which
//! element-wise ops run on in place (`ops::view_op`). A padding tap reads the
//! select's zero row.
//!
//! Max pooling selects the same way, a padding tap reading its window's first
//! valid datum (the same bits, so neither the maximum nor which one is first
//! changes), and folds the taps' column blocks in window order with exact
//! compares and selects: the first NaN, else the first maximum, bit for bit.
use super::conv::Geometry;
use super::*;
use burn_backend::ops::ConvOptions;
use std::collections::HashMap;
use std::sync::Mutex;

/// The strides of the NCHW view `[n, c, h, w]` of an NHWC matrix `[n h w,
/// c]`: element `(b, k, y, x)` is at row `(b h + y) w + x`, column `k`.
pub(super) fn nhwc_strides([_, _, h, w]: [usize; 4]) -> Vec<[usize; 2]> {
    vec![[h * w, 0], [0, 1], [w, 0], [1, 0]]
}

fn shape4(x: &TtTensor) -> Option<[usize; 4]> {
    x.shape().to_vec().try_into().ok()
}

/// The NCHW tensor of `shape` that is a view of `matrix`, `[n h w, c]`.
fn nhwc_view(matrix: &TtTensor, shape: [usize; 4]) -> TtTensor {
    let src = plain_dram(matrix);
    TtTensor::view(
        crate::views::Strided {
            src,
            base: [0, 0],
            strides: nhwc_strides(shape),
        },
        shape.to_vec().into(),
        matrix.dtype(),
        matrix.device,
    )
}

/// `x`'s NHWC table `[n h w, c]`: the buffer under `x` when `x` is an NHWC
/// view of it (a convolution's output: nothing moves), else `x` relaid on the
/// card (one repack, a datum an entry: a network's input).
fn table(x: &TtTensor, [n, c, h, w]: [usize; 4]) -> TtTensor {
    if let Some(v) = x.as_strided() {
        let strides = nhwc_strides([n, c, h, w]);
        let nhwc = v.base == [0, 0]
            && !v.src.transposed
            && v.src.buffer.storage == crate::storage::StorageFormat::F32
            && [v.src.buffer.rows, v.src.buffer.cols] == [n * h * w, c]
            && [n, c, h, w]
                .iter()
                .zip(v.strides.iter().zip(&strides))
                .all(|(&k, (s, t))| k == 1 || s == t);
        if nhwc {
            return TtTensor::on_device(v.src, vec![n * h * w, c].into(), x.dtype(), x.device);
        }
    }
    let plane = h * w;
    let sources = (0..n * plane * c)
        .map(|i| {
            let (row, k) = (i / c, i % c);
            (0, ((row / plane) * c + k) * plane + row % plane)
        })
        .collect();
    mapped_native(&[x], sources, vec![n * plane, c])
}

type Key = (TtDevice, &'static str, Vec<usize>);

/// Geometry tensors made on each attachment, with the attachment's number
/// (`server::attachment`): a later attachment's lookups drop them.
static GEOMETRY: Mutex<Option<HashMap<Key, (u64, TtTensor)>>> = Mutex::new(None);

/// The `[rows, cols]` I32 tensor `values` makes, which depends on the shapes in
/// `key` alone: made once on the card (descriptor immediates,
/// `server::metadata`, which a trace replays without an upload) and kept for
/// the device's attachment.
fn geometry(
    device: TtDevice,
    what: &'static str,
    key: Vec<usize>,
    dims: [usize; 2],
    values: impl FnOnce() -> Vec<u32>,
) -> TtTensor {
    cached_metadata(
        device,
        what,
        key,
        dims,
        burn_backend::Shape::new(dims),
        DType::I32,
        values,
    )
}

/// The tensor of `shape` (stored `dims`, of `dtype`: I32 or Bool) that `values`
/// makes, which depends on the shapes in `key` alone: made once on the card
/// (descriptor immediates, `server::metadata`, which a trace replays without
/// an upload) and kept for the device's attachment -- a convolution's tap
/// index, attention's causal mask.
pub(crate) fn cached_metadata(
    device: TtDevice,
    what: &'static str,
    key: Vec<usize>,
    dims: [usize; 2],
    shape: burn_backend::Shape,
    dtype: DType,
    values: impl FnOnce() -> Vec<u32>,
) -> TtTensor {
    let serial = crate::server::attachment(device);
    let key = (device, what, key);
    let mut stale = Vec::new();
    if let Some(serial) = serial {
        let mut guard = GEOMETRY.lock().unwrap_or_else(|p| p.into_inner());
        let map = guard.get_or_insert_with(HashMap::new);
        if let Some((s, t)) = map.get(&key) {
            if *s == serial {
                return t.clone();
            }
        }
        let old: Vec<Key> = map
            .iter()
            .filter(|((d, ..), (s, _))| *d == device && *s != serial)
            .map(|(k, _)| k.clone())
            .collect();
        stale.extend(old.into_iter().filter_map(|k| map.remove(&k)));
    }
    // Outside the lock: a dropped buffer is freed through the server.
    drop(stale);
    let elem = if matches!(dtype, DType::Bool(_)) {
        Elem::Bool
    } else {
        Elem::I32
    };
    let id = crate::server::metadata(device, values(), dims, elem);
    let t = device_result_shaped(device, id, dims, shape, dtype);
    if let Some(serial) = serial {
        GEOMETRY
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get_or_insert_with(HashMap::new)
            .insert(key, (serial, t.clone()));
    }
    t
}

/// The input row under tap `(ky, kx)` of output position `(y, x)` along each
/// axis, or `None` in the padding.
fn tap_at(
    out: usize,
    tap: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    len: usize,
) -> Option<usize> {
    (out * stride + tap * dilation)
        .checked_sub(padding)
        .filter(|&i| i < len)
}

/// `table`'s rows for each tap of `indices`, side by side
/// (`Session::select_taps`).
fn select_taps(table: &TtTensor, indices: &TtTensor, zero_row: bool) -> TtTensor {
    let (t, i) = (plain_dram(table), plain_dram(indices));
    let [taps, n] = [i.buffer.rows, i.buffer.cols];
    let dims = [n, tt_kernels::index::select_taps_cols(t.buffer.cols, taps)];
    let (id, dims) =
        crate::server::select_taps(table.device, t.buffer.id, i.buffer.id, zero_row, dims);
    device_result_shaped(
        table.device,
        id,
        dims,
        burn_backend::Shape::new(dims),
        table.dtype(),
    )
}

/// `conv2d` as one product over the NHWC table, its output an NCHW view of
/// the `[P, O]` result. BF16 and F16 operands are widened, as the chunked path
/// widens them, and the result rounded once (then no longer a view), so every
/// dtype sums in the same order. `None` where it does not apply, for the
/// chunked path: grouped, not on a device that keeps tensors, a matmul mode
/// other than [`MatmulMode::Fast`](crate::MatmulMode) (whose products are
/// refused past a size the chunks stay under), or a table too long for I32
/// indices.
pub(super) fn conv2d(
    x: &TtTensor,
    weight: &TtTensor,
    bias: Option<&TtTensor>,
    options: &ConvOptions<2>,
) -> Option<TtTensor> {
    let device = x.device;
    let float = |t: &TtTensor| {
        matches!(t.dtype(), DType::F32 | DType::BF16 | DType::F16)
            && t.is_storable()
            && t.device == device
    };
    if options.groups != 1
        || !float(x)
        || !float(weight)
        || bias.is_some_and(|b| !float(b))
        || !crate::server::supports_dram(device)
        || crate::matmul_mode::effective(device) != crate::MatmulMode::Fast
    {
        return None;
    }
    let dtype = float_compute_dtype(&[x, weight]);
    let y = conv2d_f32(x, weight, bias, options)?;
    Some(cast_native(y, dtype))
}

fn conv2d_f32(
    x: &TtTensor,
    weight: &TtTensor,
    bias: Option<&TtTensor>,
    options: &ConvOptions<2>,
) -> Option<TtTensor> {
    let device = x.device;
    let [n, c, h, w] = shape4(x)?;
    let g = Geometry::<2>::new(x, weight, options.clone());
    let (o, [kh, kw], [oh, ow]) = (g.w[0], [g.w[2], g.w[3]], [g.y[2], g.y[3]]);
    let (taps, positions, rows) = (kh * kw, n * oh * ow, n * h * w);
    if rows >= i32::MAX as usize {
        return None;
    }
    let ConvOptions {
        stride: s,
        padding: p,
        dilation: d,
        ..
    } = options.clone();
    let indices = geometry(
        device,
        "conv2d taps",
        vec![n, h, w, kh, kw, s[0], s[1], p[0], p[1], d[0], d[1]],
        [taps, positions],
        || {
            let mut v = Vec::with_capacity(taps * positions);
            for t in 0..taps {
                let (ky, kx) = (t / kw, t % kw);
                for q in 0..positions {
                    let (b, y, xo) = (q / (oh * ow), q / ow % oh, q % ow);
                    let at = tap_at(y, ky, s[0], d[0], p[0], h)
                        .zip(tap_at(xo, kx, s[1], d[1], p[1], w))
                        .map_or(rows, |(iy, ix)| (b * h + iy) * w + ix);
                    v.push(at as u32);
                }
            }
            v
        },
    );
    let zero_row = p.iter().any(|&p| p > 0);
    let x = float_compute_input(x.clone());
    let patches = select_taps(&table(&x, [n, c, h, w]), &indices, zero_row);
    // `[taps x C, O]` in the patches' order: row `t ct32 + k` is channel `k` of
    // tap `t`, zero in the columns a tap's channels do not fill.
    let k = tt_kernels::index::select_taps_cols(c, taps);
    let stride = c.div_ceil(32) * 32;
    let weights = weight.derived(format!("conv2d nhwc [{k}, {o}]"), |weight| {
        let weight = float_compute_input(weight.clone());
        let zero = super::conv::zero(device);
        let sources = (0..k * o)
            .map(|i| {
                let (r, oc) = (i / o, i % o);
                let (t, ch) = (r / stride, r % stride);
                if ch < c {
                    (0, (oc * c + ch) * taps + t)
                } else {
                    (1, 0)
                }
            })
            .collect();
        mapped_native(&[&weight, &zero], sources, vec![k, o])
    });
    let mut out = float::float_matmul(patches, weights);
    if let Some(bias) = bias {
        let bias = float_compute_input(bias.clone());
        out = float::float_add(out, reshaped(bias, [1, o].into()));
    }
    Some(nhwc_view(&out, [n, o, oh, ow]))
}

/// `max_pool2d` as one select and a fold over the taps, its output an NCHW
/// view of the `[P, C]` result. `None` where it does not apply: not F32, not on
/// a device that keeps tensors, or `C` not a whole number of tile columns (a
/// tap's block is then not whole tiles).
pub(super) fn max_pool2d(
    x: &TtTensor,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil: bool,
) -> Option<TtTensor> {
    let device = x.device;
    if x.dtype() != DType::F32 || !x.is_storable() || !crate::server::supports_dram(device) {
        return None;
    }
    let [n, c, h, w] = shape4(x)?;
    if c % 32 != 0 || n * h * w >= i32::MAX as usize {
        return None;
    }
    let [oh, ow] = [0, 1].map(|a| {
        super::pool::pooled_len(
            [h, w][a],
            kernel[a],
            stride[a],
            padding[a],
            dilation[a],
            ceil,
        )
    });
    let (taps, positions) = (kernel[0] * kernel[1], n * oh * ow);
    let indices = geometry(
        device,
        "max_pool2d taps",
        vec![
            n,
            h,
            w,
            kernel[0],
            kernel[1],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            oh,
            ow,
        ],
        [taps, positions],
        || {
            let mut v = vec![0u32; taps * positions];
            for q in 0..positions {
                let (b, y, xo) = (q / (oh * ow), q / ow % oh, q % ow);
                let at = |t: usize| {
                    tap_at(y, t / kernel[1], stride[0], dilation[0], padding[0], h)
                        .zip(tap_at(
                            xo,
                            t % kernel[1],
                            stride[1],
                            dilation[1],
                            padding[1],
                            w,
                        ))
                        .map(|(iy, ix)| (b * h + iy) * w + ix)
                };
                let first = (0..taps)
                    .find_map(at)
                    .expect("pooling window contains only padding");
                for t in 0..taps {
                    v[t * positions + q] = at(t).unwrap_or(first) as u32;
                }
            }
            v
        },
    );
    let patches = select_taps(&table(x, [n, c, h, w]), &indices, false);
    let src = plain_dram(&patches);
    let block = |t: usize| {
        let (id, dims) = crate::server::copy_blocks(
            device,
            src.buffer.id,
            vec![crate::BlockMove {
                from: [0, t * c],
                to: [0, 0],
                extent: [positions, c],
                transposed: false,
            }],
            [positions, c],
        );
        device_result(device, id, dims)
    };
    let bools = burn_backend::BoolDType::Native;
    let mut best = block(0);
    let mut nan = float::float_is_nan(best.clone(), bools);
    for t in 1..taps {
        let b = block(t);
        let b_nan = float::float_is_nan(b.clone(), bools);
        // The first NaN, else the first maximum: `b` replaces the best so far
        // only if that is not a NaN and `b` is one or is greater.
        let take = super::bool::bool_and(
            super::bool::bool_not(nan.clone()),
            super::bool::bool_or(
                float::float_greater(b.clone(), best.clone(), bools),
                b_nan.clone(),
            ),
        );
        best = float::float_mask_where(best, take, b);
        nan = super::bool::bool_or(nan, b_nan);
    }
    Some(nhwc_view(&best, [n, c, oh, ow]))
}
