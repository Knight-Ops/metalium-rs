//! NCHW patch geometry is metadata. Products and overlap sums stay on Tensix.
use super::*;
use burn_backend::ops::{ConvOptions, ConvTransposeOptions};

pub(super) const PATCH_ROWS: usize = 32;

/// Shapes are `[batch, channels, spatial...]` with `D` spatial dimensions.
pub(super) struct Geometry<const D: usize> {
    pub(super) x: Vec<usize>,
    pub(super) w: Vec<usize>,
    pub(super) y: Vec<usize>,
    pub(super) options: ConvOptions<D>,
}

impl<const D: usize> Geometry<D> {
    pub(super) fn new(x: &TtTensor, weight: &TtTensor, options: ConvOptions<D>) -> Self {
        let x = x.shape().to_vec();
        let w = weight.shape().to_vec();
        assert_eq!(x.len(), D + 2, "convolution input requires N, C, spatial");
        assert_eq!(w.len(), D + 2, "convolution weights require O, I, spatial");
        assert!(!x.contains(&0) && !w.contains(&0));
        assert!(
            options.groups > 0
                && options
                    .stride
                    .iter()
                    .chain(&options.dilation)
                    .all(|&n| n > 0)
        );
        assert_eq!(x[1] % options.groups, 0);
        assert_eq!(w[0] % options.groups, 0);
        assert_eq!(w[1].checked_mul(options.groups), Some(x[1]));
        let mut y = vec![x[0], w[0]];
        for d in 0..D {
            let effective = (w[d + 2] - 1)
                .checked_mul(options.dilation[d])
                .unwrap()
                .checked_add(1)
                .unwrap();
            let padded = x[d + 2]
                .checked_add(options.padding[d].checked_mul(2).unwrap())
                .unwrap();
            assert!(padded >= effective, "empty convolution output");
            y.push((padded - effective) / options.stride[d] + 1);
        }
        for shape in [&x, &w, &y] {
            shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .expect("convolution shape overflow");
        }
        Self { x, w, y, options }
    }

    /// Output cells per batch element.
    pub(super) fn spatial(&self) -> usize {
        self.y[2..].iter().product()
    }
    pub(super) fn kernel(&self) -> usize {
        self.w[2..].iter().product()
    }
    pub(super) fn positions(&self) -> usize {
        self.y[0] * self.spatial()
    }
    pub(super) fn k(&self) -> usize {
        self.w[1] * self.kernel()
    }
    pub(super) fn channels(&self) -> usize {
        self.w[0] / self.options.groups
    }

    /// Logical input coordinate of one patch datum, or explicit zero padding.
    fn input(&self, group: usize, position: usize, k: usize) -> Option<usize> {
        let batch = position / self.spatial();
        let channel = group * self.w[1] + k / self.kernel();
        let mut out = position % self.spatial();
        let mut tap = k % self.kernel();
        let mut at = [0i128; D];
        // Row-major, last dimension fastest.
        for d in (0..D).rev() {
            let (o, kd) = (self.y[d + 2], self.w[d + 2]);
            let (r, kr) = (out % o, tap % kd);
            (out, tap) = (out / o, tap / kd);
            let i = r as i128 * self.options.stride[d] as i128
                + kr as i128 * self.options.dilation[d] as i128
                - self.options.padding[d] as i128;
            if i < 0 || i >= self.x[d + 2] as i128 {
                return None;
            }
            at[d] = i;
        }
        let mut flat = batch * self.x[1] + channel;
        for (&i, &size) in at.iter().zip(&self.x[2..]) {
            flat = flat * size + i as usize;
        }
        Some(flat)
    }

    fn patches(
        &self,
        x: &TtTensor,
        zero: &TtTensor,
        group: usize,
        first: usize,
        count: usize,
    ) -> TtTensor {
        let sources = (first..first + count)
            .flat_map(|p| {
                (0..self.k()).map(move |k| self.input(group, p, k).map_or((1, 0), |at| (0, at)))
            })
            .collect();
        mapped_native(&[x, zero], sources, vec![count, self.k()])
    }

    pub(super) fn weights(&self, weight: &TtTensor, group: usize) -> TtTensor {
        // [patch K, output channels], so output-column mesh partitioning can
        // execute the same native product on both cards.
        let sources = (0..self.k())
            .flat_map(|k| {
                (0..self.channels()).map(move |c| (0, (group * self.channels() + c) * self.k() + k))
            })
            .collect();
        mapped_native(&[weight], sources, vec![self.k(), self.channels()])
    }

    pub(super) fn gradients(
        &self,
        grad: &TtTensor,
        group: usize,
        first: usize,
        count: usize,
    ) -> TtTensor {
        let spatial = self.spatial();
        let sources = (first..first + count)
            .flat_map(|p| {
                (0..self.channels()).map(move |c| {
                    (
                        0,
                        (p / spatial * self.y[1] + group * self.channels() + c) * spatial
                            + p % spatial,
                    )
                })
            })
            .collect();
        mapped_native(&[grad], sources, vec![count, self.channels()])
    }
}

pub(super) fn zero(device: TtDevice) -> TtTensor {
    native_zeros([1, 1].into(), &device, DType::F32).expect("native convolution zeros")
}

fn conv<const D: usize>(
    x: TtTensor,
    weight: TtTensor,
    bias: Option<TtTensor>,
    options: ConvOptions<D>,
) -> TtTensor {
    let dtype = float_compute_dtype(&[&x, &weight]);
    let g = Geometry::<D>::new(&x, &weight, options);
    let x = float_compute_input(x);
    let weight = float_compute_input(weight);
    let z = zero(x.device);
    conv_products(&g, &weight, bias, dtype, |group, first, count| {
        g.patches(&x, &z, group, first, count)
    })
}

/// Patch rows times the weights, one native product per group and chunk of [`PATCH_ROWS`]
/// positions, joined into `g.y` with the bias added. `patches(group, first, count)` is the
/// `[count, g.k()]` matrix of rows `first..first + count` (position `batch * spatial + cell`,
/// patch element `channel * kernel + tap`), so a convolution whose taps are not fixed by the
/// shapes supplies its own. `weight` is already in compute dtype; the result is cast to `dtype`.
pub(super) fn conv_products<const D: usize>(
    g: &Geometry<D>,
    weight: &TtTensor,
    bias: Option<TtTensor>,
    dtype: DType,
    mut patches: impl FnMut(usize, usize, usize) -> TtTensor,
) -> TtTensor {
    let chunks = g.positions().div_ceil(PATCH_ROWS);
    let spatial = g.spatial();
    let mut products = Vec::new();
    for group in 0..g.options.groups {
        let weights = g.weights(weight, group);
        for first in (0..g.positions()).step_by(PATCH_ROWS) {
            let count = PATCH_ROWS.min(g.positions() - first);
            products.push(float::float_matmul(
                patches(group, first, count),
                weights.clone(),
            ));
        }
    }
    let sources = (0..g.y[0])
        .flat_map(|n| {
            (0..g.y[1]).flat_map(move |c| {
                (0..spatial).map(move |s| {
                    let p = n * spatial + s;
                    (
                        c / g.channels() * chunks + p / PATCH_ROWS,
                        p % PATCH_ROWS * g.channels() + c % g.channels(),
                    )
                })
            })
        })
        .collect();
    let refs: Vec<_> = products.iter().collect();
    let mut output = mapped_native(&refs, sources, g.y.clone());
    if let Some(bias) = bias {
        assert_eq!(bias.dtype(), dtype);
        assert_eq!(bias.shape().to_vec(), vec![g.y[1]]);
        let bias = float::float_reshape(float_compute_input(bias), channel_shape(g.y[1], D));
        output = float::float_add(output, bias);
    }
    cast_native(output, dtype)
}

/// `[1, channels, 1, ...]` with `spatial` trailing ones, the shape a bias broadcasts from.
fn channel_shape(channels: usize, spatial: usize) -> burn_backend::Shape {
    let mut shape = vec![1, channels];
    shape.extend(std::iter::repeat_n(1, spatial));
    shape.into()
}

fn conv_weight_backward<const D: usize>(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvOptions<D>,
) -> TtTensor {
    let g = Geometry::<D>::new(&x, &weight, options);
    assert_eq!(output_grad.shape().to_vec(), g.y);
    conv_weight_backward_geometry(x, weight, output_grad, g)
}

fn conv_weight_backward_geometry<const D: usize>(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    g: Geometry<D>,
) -> TtTensor {
    let dtype = float_compute_dtype(&[&x, &weight, &output_grad]);
    let x = float_compute_input(x);
    let grad = float_compute_input(output_grad);
    let z = zero(x.device);
    weight_gradient(&g, &grad, dtype, |group, first, count| {
        g.patches(&x, &z, group, first, count)
    })
}

/// `patches^T * output_grad` per group, summed over chunks of [`PATCH_ROWS`] positions, joined
/// into the weight's shape. `grad` is in compute dtype and `patches` is as in [`conv_products`].
pub(super) fn weight_gradient<const D: usize>(
    g: &Geometry<D>,
    grad: &TtTensor,
    dtype: DType,
    mut patches: impl FnMut(usize, usize, usize) -> TtTensor,
) -> TtTensor {
    let mut outputs = Vec::new();
    for group in 0..g.options.groups {
        let mut sum = None;
        for first in (0..g.positions()).step_by(PATCH_ROWS) {
            let count = PATCH_ROWS.min(g.positions() - first);
            let patches = float::float_transpose(patches(group, first, count));
            let partial = float::float_matmul(patches, g.gradients(grad, group, first, count));
            sum = Some(if let Some(prior) = sum {
                float::float_add(prior, partial)
            } else {
                partial
            });
        }
        outputs.push(sum.unwrap());
    }
    let channels = g.channels();
    let sources = (0..g.w[0])
        .flat_map(|c| (0..g.k()).map(move |k| (c / channels, k * channels + c % channels)))
        .collect();
    let refs: Vec<_> = outputs.iter().collect();
    cast_native(mapped_native(&refs, sources, g.w.clone()), dtype)
}

fn conv_x_backward<const D: usize>(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvOptions<D>,
) -> TtTensor {
    let g = Geometry::<D>::new(&x, &weight, options);
    assert_eq!(output_grad.shape().to_vec(), g.y);
    conv_x_backward_geometry(x, weight, output_grad, g)
}

fn conv_x_backward_geometry<const D: usize>(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    g: Geometry<D>,
) -> TtTensor {
    let dtype = float_compute_dtype(&[&x, &weight, &output_grad]);
    let weight = float_compute_input(weight);
    let grad = float_compute_input(output_grad);
    let z = zero(x.device);
    let mut products = Vec::new();
    // Reverse geometry visits output positions, then patch coordinates, in
    // logical order. Duplicate overlap additions have a deterministic fold.
    let mut contributions = vec![Vec::new(); x.shape().num_elements()];
    for group in 0..g.options.groups {
        let weights = float::float_transpose(g.weights(&weight, group));
        for first in (0..g.positions()).step_by(PATCH_ROWS) {
            let count = PATCH_ROWS.min(g.positions() - first);
            let product = products.len();
            products.push(float::float_matmul(
                g.gradients(&grad, group, first, count),
                weights.clone(),
            ));
            for p in first..first + count {
                for k in 0..g.k() {
                    if let Some(at) = g.input(group, p, k) {
                        contributions[at].push((product, (p - first) * g.k() + k));
                    }
                }
            }
        }
    }
    let zero_index = products.len();
    products.push(z);
    let refs: Vec<_> = products.iter().collect();
    let mut outputs = Vec::new();
    for rows in contributions.chunks(PATCH_ROWS) {
        let width = rows.iter().map(Vec::len).max().unwrap().max(1);
        let sources = rows
            .iter()
            .flat_map(|r| (0..width).map(move |c| r.get(c).copied().unwrap_or((zero_index, 0))))
            .collect();
        let packed = mapped_native(&refs, sources, vec![rows.len(), width]);
        outputs.push(float::float_sum_dim(packed, 1));
    }
    let sources = (0..contributions.len())
        .map(|i| (i / PATCH_ROWS, i % PATCH_ROWS))
        .collect();
    let refs: Vec<_> = outputs.iter().collect();
    cast_native(mapped_native(&refs, sources, g.x.clone()), dtype)
}

pub(super) fn conv_bias_backward(bias: TtTensor, output_grad: TtTensor) -> TtTensor {
    let dtype = output_grad.dtype();
    assert_eq!(bias.dtype(), dtype);
    let shape = output_grad.shape().to_vec();
    assert!(shape.len() >= 3, "bias gradient requires N, C, spatial");
    assert_eq!(bias.shape().to_vec(), vec![shape[1]]);
    let grad = float_compute_input(output_grad);
    let grad = pack_axis(&grad, 1).expect("native bias gradient");
    cast_native(
        float::float_reshape(float::float_sum_dim(grad, 0), [shape[1]].into()),
        dtype,
    )
}

fn regular<const D: usize>(options: &ConvTransposeOptions<D>) -> ConvOptions<D> {
    ConvOptions::new(
        options.stride,
        options.padding,
        options.dilation,
        options.groups,
    )
}

fn conv_transpose<const D: usize>(
    x: TtTensor,
    weight: TtTensor,
    bias: Option<TtTensor>,
    options: ConvTransposeOptions<D>,
) -> TtTensor {
    let dtype = float_compute_dtype(&[&x, &weight]);
    let xs = x.shape().to_vec();
    let ws = weight.shape().to_vec();
    assert_eq!(xs.len(), D + 2);
    assert_eq!(ws.len(), D + 2);
    assert_eq!(xs[1], ws[0]);
    let channels = ws[1].checked_mul(options.groups).unwrap();
    let mut shape = vec![xs[0], channels];
    for d in 0..D {
        // Burn defines any output padding: the trailing cells no input reaches are
        // zero (plus the bias), and the backward pass drops `padding_out / stride`
        // of them. PyTorch limits it below `max(stride, dilation)`; Burn's own tests
        // (`conv_transpose1d_padding_out_stride1_backward_shape`) use more.
        assert!(
            options.stride[d] > 0 && options.dilation[d] > 0,
            "stride and dilation must be positive"
        );
        shape.push(
            (xs[d + 2] - 1)
                .checked_mul(options.stride[d])
                .unwrap()
                .checked_add((ws[d + 2] - 1).checked_mul(options.dilation[d]).unwrap())
                .unwrap()
                .checked_add(options.padding_out[d])
                .unwrap()
                .checked_add(1)
                .unwrap()
                .checked_sub(options.padding[d].checked_mul(2).unwrap())
                .filter(|&n| n > 0)
                .expect("empty transposed convolution"),
        );
    }
    let template = native_zeros(shape.into(), &x.device, DType::F32).unwrap();
    let mut g = Geometry::<D>::new(&template, &weight, regular(&options));
    g.y = xs;
    let mut output = conv_x_backward_geometry(
        template,
        float_compute_input(weight),
        float_compute_input(x),
        g,
    );
    if let Some(bias) = bias {
        assert_eq!(bias.dtype(), dtype);
        assert_eq!(bias.shape().to_vec(), vec![channels]);
        output = float::float_add(
            output,
            float::float_reshape(float_compute_input(bias), channel_shape(channels, D)),
        );
    }
    cast_native(output, dtype)
}

fn conv_transpose_x_backward<const D: usize>(
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvTransposeOptions<D>,
) -> TtTensor {
    let output = conv(output_grad, weight, None, regular(&options));
    let full = output.shape().to_vec();
    let mut shape = full.clone();
    for d in 0..D {
        shape[d + 2] -= options.padding_out[d] / options.stride[d];
    }
    if shape == full {
        return output;
    }
    let sources = (0..shape.iter().product::<usize>())
        .map(|i| {
            let mut rest = i;
            let mut flat = 0;
            let mut stride = 1;
            for d in (0..shape.len()).rev() {
                flat += rest % shape[d] * stride;
                rest /= shape[d];
                stride *= full[d];
            }
            (0, flat)
        })
        .collect();
    mapped_native(&[&output], sources, shape)
}

fn conv_transpose_weight_backward<const D: usize>(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvTransposeOptions<D>,
) -> TtTensor {
    let mut g = Geometry::<D>::new(&output_grad, &weight, regular(&options));
    g.y = x.shape().to_vec();
    conv_weight_backward_geometry(output_grad, weight, x, g)
}

pub fn conv2d(
    x: TtTensor,
    weight: TtTensor,
    bias: Option<TtTensor>,
    options: ConvOptions<2>,
) -> TtTensor {
    if let Some(y) = super::nhwc::conv2d(&x, &weight, bias.as_ref(), &options) {
        return y;
    }
    conv(x, weight, bias, options)
}

pub fn conv2d_weight_backward(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvOptions<2>,
) -> TtTensor {
    conv_weight_backward(x, weight, output_grad, options)
}

pub fn conv2d_x_backward(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvOptions<2>,
) -> TtTensor {
    conv_x_backward(x, weight, output_grad, options)
}

pub fn conv2d_bias_backward(_x: TtTensor, bias: TtTensor, output_grad: TtTensor) -> TtTensor {
    conv_bias_backward(bias, output_grad)
}

pub fn conv_transpose2d(
    x: TtTensor,
    weight: TtTensor,
    bias: Option<TtTensor>,
    options: ConvTransposeOptions<2>,
) -> TtTensor {
    conv_transpose(x, weight, bias, options)
}

pub fn conv_transpose2d_x_backward(
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvTransposeOptions<2>,
) -> TtTensor {
    conv_transpose_x_backward(weight, output_grad, options)
}

pub fn conv_transpose2d_weight_backward(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvTransposeOptions<2>,
) -> TtTensor {
    conv_transpose_weight_backward(x, weight, output_grad, options)
}

pub fn conv_transpose2d_bias_backward(
    x: TtTensor,
    bias: TtTensor,
    output_grad: TtTensor,
) -> TtTensor {
    conv2d_bias_backward(x, bias, output_grad)
}

pub fn conv3d(
    x: TtTensor,
    weight: TtTensor,
    bias: Option<TtTensor>,
    options: ConvOptions<3>,
) -> TtTensor {
    conv(x, weight, bias, options)
}

pub fn conv3d_weight_backward(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvOptions<3>,
) -> TtTensor {
    conv_weight_backward(x, weight, output_grad, options)
}

pub fn conv3d_x_backward(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvOptions<3>,
) -> TtTensor {
    conv_x_backward(x, weight, output_grad, options)
}

pub fn conv3d_bias_backward(_x: TtTensor, bias: TtTensor, output_grad: TtTensor) -> TtTensor {
    conv_bias_backward(bias, output_grad)
}

pub fn conv_transpose3d(
    x: TtTensor,
    weight: TtTensor,
    bias: Option<TtTensor>,
    options: ConvTransposeOptions<3>,
) -> TtTensor {
    conv_transpose(x, weight, bias, options)
}

pub fn conv_transpose3d_x_backward(
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvTransposeOptions<3>,
) -> TtTensor {
    conv_transpose_x_backward(weight, output_grad, options)
}

pub fn conv_transpose3d_weight_backward(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvTransposeOptions<3>,
) -> TtTensor {
    conv_transpose_weight_backward(x, weight, output_grad, options)
}

pub fn conv_transpose3d_bias_backward(
    x: TtTensor,
    bias: TtTensor,
    output_grad: TtTensor,
) -> TtTensor {
    conv2d_bias_backward(x, bias, output_grad)
}

pub fn unfold4d(
    x: TtTensor,
    kernel_size: [usize; 2],
    options: burn_backend::ops::UnfoldOptions,
) -> TtTensor {
    let shape = x.shape().to_vec();
    assert_eq!(shape.len(), 4);
    let device = x.device;
    let dtype = x.dtype();
    let weights = native_zeros(
        [1, shape[1], kernel_size[0], kernel_size[1]].into(),
        &device,
        dtype,
    )
    .unwrap();
    let g = Geometry::<2>::new(
        &x,
        &weights,
        ConvOptions::new(options.stride, options.padding, options.dilation, 1),
    );
    let zero = native_zeros([1].into(), &device, dtype).unwrap();
    let positions = g.spatial();
    let mut sources = Vec::new();
    for n in 0..g.x[0] {
        for k in 0..g.k() {
            for p in 0..positions {
                sources.push(
                    g.input(0, n * positions + p, k)
                        .map_or((1, 0), |at| (0, at)),
                );
            }
        }
    }
    mapped_native(&[&x, &zero], sources, vec![g.x[0], g.k(), positions])
}
