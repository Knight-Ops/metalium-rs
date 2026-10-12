//! `deform_conv2d` and its gradients, composed from resident primitives.
//!
//! The taps of a deformable convolution depend on the data: each output cell and kernel tap samples
//! `x` at `base + offset`, a fractional position read from the `offset` tensor, with bilinear
//! interpolation and an optional per-tap `mask`. Unlike `interpolate` the host therefore cannot
//! tabulate the taps from the shapes; it builds only what the shapes fix (the position of each
//! tap without its offset, and which plane it reads) and the card does the rest, in burn-flex's
//! arithmetic (`deform_im2col_f32`, `bilinear_interpolate`, `deform_conv2d_backward_f32`):
//!
//! 1. a sample is one `(batch, offset group, tap, cell)`, `P` of them; its position is
//!    `base + offset` (SFPU add), its corner fractions come from `floor`, and each of the four
//!    corners is valid when the sample is (`-1 < h < H`, `-1 < w < W`, as Flex) and the corner
//!    is inside the plane. An invalid corner reads an appended all-zero row, so no index the
//!    device uses is out of range and no value outside the plane is ever read;
//! 2. the corner values are four resident gathers (`gather_native`) of whole channel rows of `x`
//!    laid out `[batch * groups * H * W + 1, channels per offset group]`; bits are moved, so a
//!    NaN or infinity in `x` reaches only the samples that read it;
//! 3. `((v1 * w1 + v2 * w2) + v3 * w3) + v4 * w4`, left to right with `w1 = (1 - lh) * (1 - lw)`
//!    and so on, then the mask: SFPU multiplies and adds in Flex's order;
//! 4. the columns are the patches of an ordinary grouped convolution, so the product with the
//!    weights, the bias, the join and the weight gradient are `conv`'s (`conv_products`,
//!    `weight_gradient`), the Matrix Unit's matmul with its documented rounding.
//!
//! The gradient of `x` is the transpose of step 2: a one-hot matrix `[P, rows + 1]` (each corner's
//! row, scaled by its bilinear weight, built by comparing the row index with a constant iota) times
//! the gradient of the columns, one matmul. `select_add` would fold in index order, as Flex's
//! sequential `+=` does, but it launches several kernels per index; the matrix form is a fixed
//! number of launches, at the price of Flex's summation order (the Matrix Unit's, here).
//!
//! **Cost.** A gather stages every row of its table for every element it returns, so the host
//! holds `4 * P * channels_per_offset_group * (rows + 1)` addresses; more than [`MAX_SOURCES`] is
//! refused by name. The one-hot matrix holds `P * (rows + 1)` elements on the card. Row indices
//! are integers carried in `f32` (exact below 2^24), which that limit also guarantees.
//!
//! **Where it differs from Flex.** A NaN offset gives Flex a position that fails neither range test
//! and then indexes through a saturated cast; here it is an invalid sample (value 0, gradient 0).
//! Infinite offsets are invalid in both. No test depends on either.
use super::conv::{conv_bias_backward, conv_products, weight_gradient, zero, Geometry, PATCH_ROWS};
use super::pool::constant;
use super::*;
use burn_backend::ops::{ConvOptions, DeformConv2dBackward, DeformConvOptions};

/// The most gathered-row addresses one call stages on the host (see the module documentation).
const MAX_SOURCES: usize = 1 << 24;

/// Extents of one call. A sample `p` is `((n * groups + og) * taps + tap) * cells + cell`: the
/// layout of `mask` (`[n, groups * taps, oh, ow]`) flattened.
struct Dims {
    n: usize,
    channels: usize,
    h: usize,
    w: usize,
    /// Offset groups.
    groups: usize,
    /// Channels per offset group.
    cg: usize,
    /// Kernel taps.
    taps: usize,
    kw: usize,
    ow: usize,
    /// Output cells per batch element.
    cells: usize,
    /// Samples.
    p: usize,
    /// Rows of the table the gathers read, before its zero row.
    rows: usize,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
}

impl Dims {
    fn new(
        op: &str,
        x: &TtTensor,
        offset: &TtTensor,
        mask: Option<&TtTensor>,
        g: &Geometry<2>,
        options: &DeformConvOptions<2>,
    ) -> Self {
        let [n, channels, h, w] = [g.x[0], g.x[1], g.x[2], g.x[3]];
        let groups = options.offset_groups;
        assert!(groups > 0 && channels % groups == 0, "{op}: offset groups");
        let taps = g.kernel();
        let cells = g.spatial();
        assert_eq!(
            offset.shape().to_vec(),
            vec![n, groups * taps * 2, g.y[2], g.y[3]],
            "{op}: offset shape"
        );
        if let Some(mask) = mask {
            assert_eq!(
                mask.shape().to_vec(),
                vec![n, groups * taps, g.y[2], g.y[3]],
                "{op}: mask shape"
            );
        }
        let cg = channels / groups;
        let checked = |factors: &[usize]| {
            factors
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .filter(|&v| v <= MAX_SOURCES)
        };
        let rows = checked(&[n, groups, h, w]);
        let p = checked(&[n, groups, taps, cells]);
        let staged = rows
            .zip(p)
            .and_then(|(rows, p)| checked(&[4, p, cg, rows + 1]));
        if staged.is_none() {
            fail(
                op,
                format_args!(
                    "stages 4 gathers x {n}x{groups}x{taps}x{cells} samples x {cg} channels x \
                     {n}x{groups}x{h}x{w}+1 rows of host addresses, over the {MAX_SOURCES} limit \
                     (a gather stages its whole table per element); {}",
                    context(x)
                ),
            );
        }
        Self {
            n,
            channels,
            h,
            w,
            groups,
            cg,
            taps,
            kw: g.w[3],
            ow: g.y[3],
            cells,
            p: p.unwrap(),
            rows: rows.unwrap(),
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
        }
    }

    /// `(batch, offset group, tap, cell)` of sample `p`.
    fn sample(&self, p: usize) -> (usize, usize, usize, usize) {
        let (q, cell) = (p / self.cells, p % self.cells);
        let (q, tap) = (q / self.taps, q % self.taps);
        (q / self.groups, q % self.groups, tap, cell)
    }
}

/// Everything the columns and the gradients are made of, as `[P, 1]` or `[P, cg]` tensors.
struct Taps {
    /// Fractions below and above the sample, per axis: `(1 - lh)`, `lh`, `(1 - lw)`, `lw`.
    hh: TtTensor,
    lh: TtTensor,
    hw: TtTensor,
    lw: TtTensor,
    /// The row of the table each corner reads (`rows` is the zero row), in `f32`, corners in the
    /// order `(h, w)`, `(h, w + 1)`, `(h + 1, w)`, `(h + 1, w + 1)`.
    index: [TtTensor; 4],
    /// The weight of each corner: `hh * hw`, `hh * lw`, `lh * hw`, `lh * lw`.
    weight: [TtTensor; 4],
    /// What each corner read.
    value: [TtTensor; 4],
    /// The bilinear value of each sample and channel, before the mask.
    interp: TtTensor,
}

fn scalar(v: usize) -> burn_backend::Scalar {
    (v as f32).into()
}

fn bool_dtype(device: TtDevice) -> burn_backend::BoolDType {
    burn_backend::get_device_settings::<TtBackend>(&device).bool_dtype
}

/// `0 <= c < extent`.
fn inside(c: &TtTensor, extent: usize, bool_dtype: burn_backend::BoolDType) -> TtTensor {
    bool::bool_and(
        float::float_greater_equal_elem(c.clone(), 0.0f32.into(), bool_dtype),
        float::float_lower_elem(c.clone(), scalar(extent), bool_dtype),
    )
}

fn taps(d: &Dims, x: &TtTensor, offset: &TtTensor) -> Taps {
    let device = x.device;
    let bd = bool_dtype(device);
    let column = |values: Vec<f32>| constant(device, values, [d.p, 1]);

    // The offsets, one h and one w per sample: [P, 1].
    let offsets = |t: usize| {
        let sources = (0..d.p)
            .map(|p| {
                let (n, og, tap, cell) = d.sample(p);
                let channel = og * d.taps * 2 + tap * 2 + t;
                (0, (n * d.groups * d.taps * 2 + channel) * d.cells + cell)
            })
            .collect();
        mapped_native(&[offset], sources, vec![d.p, 1])
    };
    // Where the tap would sample with no offset, and where its plane starts in the table.
    let base = |axis: usize| {
        column(
            (0..d.p)
                .map(|p| {
                    let (_, _, tap, cell) = d.sample(p);
                    let (o, k) = if axis == 0 {
                        (cell / d.ow, tap / d.kw)
                    } else {
                        (cell % d.ow, tap % d.kw)
                    };
                    (o * d.stride[axis] + k * d.dilation[axis]) as f32 - d.padding[axis] as f32
                })
                .collect(),
        )
    };
    let plane = column(
        (0..d.p)
            .map(|p| {
                let (n, og, _, _) = d.sample(p);
                ((n * d.groups + og) * d.h * d.w) as f32
            })
            .collect(),
    );
    let sh = float::float_add(offsets(0), base(0));
    let sw = float::float_add(offsets(1), base(1));

    // Flex: a sample on or past the plane's border, in either axis, is zero.
    let valid = bool::bool_and(
        bool::bool_and(
            float::float_greater_elem(sh.clone(), (-1.0f32).into(), bd),
            float::float_lower_elem(sh.clone(), scalar(d.h), bd),
        ),
        bool::bool_and(
            float::float_greater_elem(sw.clone(), (-1.0f32).into(), bd),
            float::float_lower_elem(sw.clone(), scalar(d.w), bd),
        ),
    );
    let invalid = bool::bool_not(valid.clone());

    let (hl, wl) = (
        float::float_floor(sh.clone()),
        float::float_floor(sw.clone()),
    );
    // The fractions of an invalid sample are cleared so that no NaN or infinity reaches a product.
    let fraction = |s: TtTensor, low: &TtTensor| {
        float::float_mask_fill(
            float::float_sub(s, low.clone()),
            invalid.clone(),
            0.0f32.into(),
        )
    };
    let (lh, lw) = (fraction(sh, &hl), fraction(sw, &wl));
    let complement = |l: &TtTensor| {
        float::float_add_scalar(
            float::float_mul_scalar(l.clone(), (-1.0f32).into()),
            1.0f32.into(),
        )
    };
    let (hh, hw) = (complement(&lh), complement(&lw));

    let (hh_up, wl_up) = (
        float::float_add_scalar(hl.clone(), 1.0f32.into()),
        float::float_add_scalar(wl.clone(), 1.0f32.into()),
    );
    let rows_h = [hl, hh_up];
    let cols_w = [wl, wl_up];
    let inside_h = rows_h.each_ref().map(|c| inside(c, d.h, bd));
    let inside_w = cols_w.each_ref().map(|c| inside(c, d.w, bd));

    let corner = |a: usize, b: usize| {
        let ok = bool::bool_and(
            bool::bool_and(inside_h[a].clone(), inside_w[b].clone()),
            valid.clone(),
        );
        let flat = float::float_add(
            float::float_add(
                plane.clone(),
                float::float_mul_scalar(rows_h[a].clone(), scalar(d.w)),
            ),
            cols_w[b].clone(),
        );
        // Row indices are exact in f32 (below 2^24); anything not a valid corner reads the zero row.
        float::float_mask_fill(flat, bool::bool_not(ok), scalar(d.rows))
    };
    let index = [corner(0, 0), corner(0, 1), corner(1, 0), corner(1, 1)];

    // x as `[n, group, h, w]` rows of one offset group's channels, then a row of zeros.
    let z = zero(device);
    let sources = (0..=d.rows)
        .flat_map(|r| {
            (0..d.cg).map(move |ch| {
                if r == d.rows {
                    return (1, 0);
                }
                let (q, w) = (r / d.w, r % d.w);
                let (q, h) = (q / d.h, q % d.h);
                let (n, og) = (q / d.groups, q % d.groups);
                (0, ((n * d.channels + og * d.cg + ch) * d.h + h) * d.w + w)
            })
        })
        .collect();
    let table = mapped_native(&[x, &z], sources, vec![d.rows + 1, d.cg]);
    let value = index.each_ref().map(|i| {
        let at = float::float_into_int(i.clone(), IntDType::I32);
        let at = expanded(&at, [d.p, d.cg].into(), "deform_conv2d gather indices");
        gather_native(0, table.clone(), at)
    });

    let weight = [
        float::float_mul(hh.clone(), hw.clone()),
        float::float_mul(hh.clone(), lw.clone()),
        float::float_mul(lh.clone(), hw.clone()),
        float::float_mul(lh.clone(), lw.clone()),
    ];
    let term = |c: usize| float::float_mul(value[c].clone(), weight[c].clone());
    let interp = float::float_add(
        float::float_add(float::float_add(term(0), term(1)), term(2)),
        term(3),
    );
    Taps {
        hh,
        lh,
        hw,
        lw,
        index,
        weight,
        value,
        interp,
    }
}

/// `[P, 1]` view of a mask of `[n, groups * taps, oh, ow]`: the same order.
fn mask_column(d: &Dims, mask: TtTensor) -> TtTensor {
    float::float_reshape(mask, [d.p, 1].into())
}

/// The patch matrix `[count, g.k()]` of rows `first..first + count` that `conv_products` and
/// `weight_gradient` ask for, read out of `columns` (`[P, cg]`).
fn patches(
    d: &Dims,
    g: &Geometry<2>,
    columns: &TtTensor,
    group: usize,
    first: usize,
    count: usize,
) -> TtTensor {
    let sources = (first..first + count)
        .flat_map(|pos| {
            (0..g.k()).map(move |k| {
                let (n, cell) = (pos / d.cells, pos % d.cells);
                let (ic, tap) = (k / d.taps, k % d.taps);
                let c = group * g.w[1] + ic;
                let p = ((n * d.groups + c / d.cg) * d.taps + tap) * d.cells + cell;
                (0, p * d.cg + c % d.cg)
            })
        })
        .collect();
    mapped_native(&[columns], sources, vec![count, g.k()])
}

fn geometry(x: &TtTensor, weight: &TtTensor, options: &DeformConvOptions<2>) -> Geometry<2> {
    assert!(
        options.offset_groups > 0,
        "deform_conv2d: offset groups must be nonzero"
    );
    Geometry::<2>::new(
        x,
        weight,
        ConvOptions::new(
            options.stride,
            options.padding,
            options.dilation,
            options.weight_groups,
        ),
    )
}

/// The columns of the convolution (`[P, cg]`, mask applied) and the taps they came from.
fn columns(d: &Dims, t: &Taps, mask: Option<TtTensor>) -> TtTensor {
    match mask {
        Some(mask) => float::float_mul(t.interp.clone(), mask_column(d, mask)),
        None => t.interp.clone(),
    }
}

pub fn deform_conv2d(
    x: TtTensor,
    offset: TtTensor,
    weight: TtTensor,
    mask: Option<TtTensor>,
    bias: Option<TtTensor>,
    options: DeformConvOptions<2>,
) -> TtTensor {
    let mut floats = vec![&x, &offset, &weight];
    floats.extend(mask.as_ref());
    let dtype = float_compute_dtype(&floats);
    let g = geometry(&x, &weight, &options);
    let x = float_compute_input(x);
    let offset = float_compute_input(offset);
    let weight = float_compute_input(weight);
    let mask = mask.map(float_compute_input);
    let d = Dims::new("deform_conv2d", &x, &offset, mask.as_ref(), &g, &options);
    let t = taps(&d, &x, &offset);
    let cols = columns(&d, &t, mask);
    conv_products(&g, &weight, bias, dtype, |group, first, count| {
        patches(&d, &g, &cols, group, first, count)
    })
}

pub fn deform_conv2d_backward(
    x: TtTensor,
    offset: TtTensor,
    weight: TtTensor,
    mask: Option<TtTensor>,
    bias: Option<TtTensor>,
    output_grad: TtTensor,
    options: DeformConvOptions<2>,
) -> DeformConv2dBackward<TtBackend> {
    let mut floats = vec![&x, &offset, &weight, &output_grad];
    floats.extend(mask.as_ref());
    let dtype = float_compute_dtype(&floats);
    let g = geometry(&x, &weight, &options);
    assert_eq!(
        output_grad.shape().to_vec(),
        g.y,
        "deform_conv2d: output gradient shape"
    );
    let x_shape = x.shape().to_vec();
    let bias_grad = bias.map(|bias| conv_bias_backward(bias, output_grad.clone()));
    let x = float_compute_input(x);
    let offset = float_compute_input(offset);
    let weight = float_compute_input(weight);
    let mask = mask.map(float_compute_input);
    let grad = float_compute_input(output_grad);
    let d = Dims::new(
        "deform_conv2d_backward",
        &x,
        &offset,
        mask.as_ref(),
        &g,
        &options,
    );
    let device = x.device;
    let t = taps(&d, &x, &offset);
    let cols = columns(&d, &t, mask.clone());

    let weight_grad = weight_gradient(&g, &grad, dtype, |group, first, count| {
        patches(&d, &g, &cols, group, first, count)
    });

    // Gradient of the columns, `grad * weights^T` per group and chunk, laid back out as [P, cg].
    let chunks = g.positions().div_ceil(PATCH_ROWS);
    let mut products = Vec::new();
    for group in 0..g.options.groups {
        let weights = float::float_transpose(g.weights(&weight, group));
        for first in (0..g.positions()).step_by(PATCH_ROWS) {
            let count = PATCH_ROWS.min(g.positions() - first);
            products.push(float::float_matmul(
                g.gradients(&grad, group, first, count),
                weights.clone(),
            ));
        }
    }
    let sources = (0..d.p)
        .flat_map(|p| {
            let (n, og, tap, cell) = d.sample(p);
            let (d, g) = (&d, &g);
            (0..d.cg).map(move |ch| {
                let c = og * d.cg + ch;
                let (group, ic) = (c / g.w[1], c % g.w[1]);
                let pos = n * d.cells + cell;
                (
                    group * chunks + pos / PATCH_ROWS,
                    (pos % PATCH_ROWS) * g.k() + ic * d.taps + tap,
                )
            })
        })
        .collect();
    let refs: Vec<_> = products.iter().collect();
    let column_grad = mapped_native(&refs, sources, vec![d.p, d.cg]);

    // The mask scales the columns, so it also takes the gradient of the unscaled value.
    let (mask_grad, value_grad) = match mask {
        Some(mask) => {
            let shape = mask.shape();
            let sum =
                float::float_sum_dim(float::float_mul(column_grad.clone(), t.interp.clone()), 1);
            (
                Some(float::float_reshape(sum, shape)),
                float::float_mul(column_grad, mask_column(&d, mask)),
            )
        }
        None => (None, column_grad),
    };

    // Offsets: the derivative of the bilinear value with respect to each coordinate, as Flex.
    let [v1, v2, v3, v4] = t.value.clone();
    let diff = |a: &TtTensor, b: &TtTensor| float::float_sub(a.clone(), b.clone());
    let along = |lhs: TtTensor, rhs: TtTensor, first: &TtTensor, second: &TtTensor| {
        float::float_add(
            float::float_mul(lhs, first.clone()),
            float::float_mul(rhs, second.clone()),
        )
    };
    let grad_h = along(diff(&v3, &v1), diff(&v4, &v2), &t.hw, &t.lw);
    let grad_w = along(diff(&v2, &v1), diff(&v4, &v3), &t.hh, &t.lh);
    let reduced = |g: TtTensor| float::float_sum_dim(float::float_mul(value_grad.clone(), g), 1);
    let (gh, gw) = (reduced(grad_h), reduced(grad_w));
    let sources = (0..d.n)
        .flat_map(|n| {
            let d = &d;
            (0..d.groups * d.taps * 2).flat_map(move |channel| {
                let (og, rest) = (channel / (d.taps * 2), channel % (d.taps * 2));
                let (tap, axis) = (rest / 2, rest % 2);
                (0..d.cells)
                    .map(move |cell| (axis, ((n * d.groups + og) * d.taps + tap) * d.cells + cell))
            })
        })
        .collect();
    let offset_grad = mapped_native(
        &[&gh, &gw],
        sources,
        vec![d.n, d.groups * d.taps * 2, g.y[2], g.y[3]],
    );

    // x: each corner's weight at the row it read, summed over samples by one matmul.
    let width = d.rows + 1;
    let iota = constant(device, (0..width).map(|r| r as f32).collect(), [1, width]);
    let zeros = native_zeros([d.p, width].into(), &device, DType::F32)
        .expect("native deform_conv2d one-hot zeros");
    let bd = bool_dtype(device);
    let hot = |c: usize| {
        let at = expanded(
            &t.index[c],
            [d.p, width].into(),
            "deform_conv2d one-hot rows",
        );
        let hit = float::float_equal(at, iota.clone(), bd);
        float::float_mul(
            float::float_mask_fill(zeros.clone(), hit, 1.0f32.into()),
            t.weight[c].clone(),
        )
    };
    let matrix = float::float_add(
        float::float_add(float::float_add(hot(0), hot(1)), hot(2)),
        hot(3),
    );
    let table = float::float_matmul(float::float_transpose(matrix), value_grad);
    let sources = (0..x_shape.iter().product::<usize>())
        .map(|flat| {
            let (q, w) = (flat / d.w, flat % d.w);
            let (q, h) = (q / d.h, q % d.h);
            let (n, c) = (q / d.channels, q % d.channels);
            let row = ((n * d.groups + c / d.cg) * d.h + h) * d.w + w;
            (0, row * d.cg + c % d.cg)
        })
        .collect();
    let x_grad = mapped_native(&[&table], sources, x_shape);

    DeformConv2dBackward {
        x_grad: cast_native(x_grad, dtype),
        offset_grad: cast_native(offset_grad, dtype),
        weight_grad,
        mask_grad: mask_grad.map(|m| cast_native(m, dtype)),
        bias_grad,
    }
}
