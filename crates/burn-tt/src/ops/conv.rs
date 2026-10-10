//! NCHW patch geometry is metadata. Products and overlap sums stay on Tensix.
use super::*;
use burn_backend::ops::{ConvOptions, ConvTransposeOptions};

const PATCH_ROWS: usize = 32;

struct Geometry {
    x: [usize; 4],
    w: [usize; 4],
    y: [usize; 4],
    options: ConvOptions<2>,
}

impl Geometry {
    fn new(x: &TtTensor, weight: &TtTensor, options: ConvOptions<2>) -> Self {
        let xs = x.shape().to_vec();
        let ws = weight.shape().to_vec();
        assert_eq!(xs.len(), 4, "convolution input requires NCHW");
        assert_eq!(ws.len(), 4, "convolution weights require OIHW");
        assert!(!xs.contains(&0) && !ws.contains(&0));
        assert!(
            options.groups > 0
                && options
                    .stride
                    .iter()
                    .chain(&options.dilation)
                    .all(|&n| n > 0)
        );
        let x: [usize; 4] = xs.try_into().unwrap();
        let w: [usize; 4] = ws.try_into().unwrap();
        assert_eq!(x[1] % options.groups, 0);
        assert_eq!(w[0] % options.groups, 0);
        assert_eq!(w[1].checked_mul(options.groups), Some(x[1]));
        let spatial = std::array::from_fn::<_, 2, _>(|d| {
            let effective = (w[d + 2] - 1)
                .checked_mul(options.dilation[d])
                .unwrap()
                .checked_add(1)
                .unwrap();
            let padded = x[d + 2]
                .checked_add(options.padding[d].checked_mul(2).unwrap())
                .unwrap();
            assert!(padded >= effective, "empty convolution output");
            (padded - effective) / options.stride[d] + 1
        });
        let y = [x[0], w[0], spatial[0], spatial[1]];
        for shape in [x, w, y] {
            shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .expect("convolution shape overflow");
        }
        Self { x, w, y, options }
    }

    fn positions(&self) -> usize {
        self.y[0] * self.y[2] * self.y[3]
    }
    fn k(&self) -> usize {
        self.w[1] * self.w[2] * self.w[3]
    }
    fn channels(&self) -> usize {
        self.w[0] / self.options.groups
    }

    /// Logical input coordinate of one patch datum, or explicit zero padding.
    fn input(&self, group: usize, position: usize, k: usize) -> Option<usize> {
        let [_, channels, h, w] = self.x;
        let [_, _, oh, ow] = self.y;
        let batch = position / (oh * ow);
        let r = position / ow % oh;
        let c = position % ow;
        let channel = group * self.w[1] + k / (self.w[2] * self.w[3]);
        let kr = k / self.w[3] % self.w[2];
        let kc = k % self.w[3];
        let ir = (r as i128) * self.options.stride[0] as i128
            + kr as i128 * self.options.dilation[0] as i128
            - self.options.padding[0] as i128;
        let ic = (c as i128) * self.options.stride[1] as i128
            + kc as i128 * self.options.dilation[1] as i128
            - self.options.padding[1] as i128;
        (ir >= 0 && ir < h as i128 && ic >= 0 && ic < w as i128)
            .then(|| ((batch * channels + channel) * h + ir as usize) * w + ic as usize)
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

    fn weights(&self, weight: &TtTensor, group: usize) -> TtTensor {
        // [patch K, output channels], so output-column mesh partitioning can
        // execute the same native product on both cards.
        let sources = (0..self.k())
            .flat_map(|k| {
                (0..self.channels()).map(move |c| (0, (group * self.channels() + c) * self.k() + k))
            })
            .collect();
        mapped_native(&[weight], sources, vec![self.k(), self.channels()])
    }

    fn gradients(&self, grad: &TtTensor, group: usize, first: usize, count: usize) -> TtTensor {
        let spatial = self.y[2] * self.y[3];
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

fn zero(device: TtDevice) -> TtTensor {
    native_zeros([1, 1].into(), &device, DType::F32).expect("native convolution zeros")
}

pub fn conv2d(
    x: TtTensor,
    weight: TtTensor,
    bias: Option<TtTensor>,
    options: ConvOptions<2>,
) -> TtTensor {
    let dtype = float_compute_dtype(&[&x, &weight]);
    let g = Geometry::new(&x, &weight, options);
    let x = float_compute_input(x);
    let weight = float_compute_input(weight);
    let z = zero(x.device);
    let chunks = g.positions().div_ceil(PATCH_ROWS);
    let mut products = Vec::new();
    for group in 0..g.options.groups {
        let weights = g.weights(&weight, group);
        for first in (0..g.positions()).step_by(PATCH_ROWS) {
            let count = PATCH_ROWS.min(g.positions() - first);
            products.push(float::float_matmul(
                g.patches(&x, &z, group, first, count),
                weights.clone(),
            ));
        }
    }
    let sources = (0..g.y[0])
        .flat_map(|n| {
            let g = &g;
            (0..g.y[1]).flat_map(move |c| {
                (0..g.y[2] * g.y[3]).map(move |s| {
                    let p = n * g.y[2] * g.y[3] + s;
                    (
                        c / g.channels() * chunks + p / PATCH_ROWS,
                        p % PATCH_ROWS * g.channels() + c % g.channels(),
                    )
                })
            })
        })
        .collect();
    let refs: Vec<_> = products.iter().collect();
    let mut output = mapped_native(&refs, sources, g.y.to_vec());
    if let Some(bias) = bias {
        assert_eq!(bias.dtype(), dtype);
        assert_eq!(bias.shape().to_vec(), vec![g.y[1]]);
        let bias = float::float_reshape(float_compute_input(bias), [1, g.y[1], 1, 1].into());
        output = float::float_add(output, bias);
    }
    cast_native(output, dtype)
}

pub fn conv2d_weight_backward(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvOptions<2>,
) -> TtTensor {
    let g = Geometry::new(&x, &weight, options);
    assert_eq!(output_grad.shape().to_vec(), g.y);
    conv2d_weight_backward_geometry(x, weight, output_grad, g)
}

fn conv2d_weight_backward_geometry(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    g: Geometry,
) -> TtTensor {
    let dtype = float_compute_dtype(&[&x, &weight, &output_grad]);
    let x = float_compute_input(x);
    let grad = float_compute_input(output_grad);
    let z = zero(x.device);
    let mut outputs = Vec::new();
    for group in 0..g.options.groups {
        let mut sum = None;
        for first in (0..g.positions()).step_by(PATCH_ROWS) {
            let count = PATCH_ROWS.min(g.positions() - first);
            let patches = float::float_transpose(g.patches(&x, &z, group, first, count));
            let partial = float::float_matmul(patches, g.gradients(&grad, group, first, count));
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
    cast_native(mapped_native(&refs, sources, g.w.to_vec()), dtype)
}

pub fn conv2d_x_backward(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvOptions<2>,
) -> TtTensor {
    let g = Geometry::new(&x, &weight, options);
    assert_eq!(output_grad.shape().to_vec(), g.y);
    conv2d_x_backward_geometry(x, weight, output_grad, g)
}

fn conv2d_x_backward_geometry(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    g: Geometry,
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
    cast_native(mapped_native(&refs, sources, g.x.to_vec()), dtype)
}

pub fn conv2d_bias_backward(_x: TtTensor, bias: TtTensor, output_grad: TtTensor) -> TtTensor {
    let dtype = output_grad.dtype();
    assert_eq!(bias.dtype(), dtype);
    let shape = output_grad.shape().to_vec();
    assert_eq!(shape.len(), 4);
    assert_eq!(bias.shape().to_vec(), vec![shape[1]]);
    let grad = float_compute_input(output_grad);
    let grad = pack_axis(&grad, 1).expect("native bias gradient");
    cast_native(
        float::float_reshape(float::float_sum_dim(grad, 0), [shape[1]].into()),
        dtype,
    )
}

fn regular(options: &ConvTransposeOptions<2>) -> ConvOptions<2> {
    ConvOptions::new(
        options.stride,
        options.padding,
        options.dilation,
        options.groups,
    )
}

pub fn conv_transpose2d(
    x: TtTensor,
    weight: TtTensor,
    bias: Option<TtTensor>,
    options: ConvTransposeOptions<2>,
) -> TtTensor {
    let dtype = float_compute_dtype(&[&x, &weight]);
    let xs = x.shape().to_vec();
    let ws = weight.shape().to_vec();
    assert_eq!(xs.len(), 4);
    assert_eq!(ws.len(), 4);
    assert_eq!(xs[1], ws[0]);
    let spatial = std::array::from_fn::<_, 2, _>(|d| {
        assert!(
            options.stride[d] > 0
                && options.dilation[d] > 0
                && options.padding_out[d] < options.stride[d].max(options.dilation[d]),
            "output padding must be less than stride or dilation"
        );
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
            .expect("empty transposed convolution")
    });
    let channels = ws[1].checked_mul(options.groups).unwrap();
    let shape = [xs[0], channels, spatial[0], spatial[1]];
    let template = native_zeros(shape.into(), &x.device, DType::F32).unwrap();
    let mut g = Geometry::new(&template, &weight, regular(&options));
    g.y = xs.try_into().unwrap();
    let mut output = conv2d_x_backward_geometry(
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
            float::float_reshape(float_compute_input(bias), [1, channels, 1, 1].into()),
        );
    }
    cast_native(output, dtype)
}

pub fn conv_transpose2d_x_backward(
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvTransposeOptions<2>,
) -> TtTensor {
    let output = conv2d(output_grad, weight, None, regular(&options));
    let full = output.shape().to_vec();
    let mut shape = full.clone();
    for d in 0..2 {
        shape[d + 2] -= options.padding_out[d] / options.stride[d];
    }
    if shape == full {
        return output;
    }
    let sources = (0..shape.iter().product::<usize>())
        .map(|i| {
            let c = i % shape[3];
            let r = i / shape[3] % shape[2];
            let nc = i / (shape[2] * shape[3]);
            (0, (nc * full[2] + r) * full[3] + c)
        })
        .collect();
    mapped_native(&[&output], sources, shape)
}

pub fn conv_transpose2d_weight_backward(
    x: TtTensor,
    weight: TtTensor,
    output_grad: TtTensor,
    options: ConvTransposeOptions<2>,
) -> TtTensor {
    let mut g = Geometry::new(&output_grad, &weight, regular(&options));
    g.y = x.shape().to_vec().try_into().unwrap();
    conv2d_weight_backward_geometry(output_grad, weight, x, g)
}

pub fn conv_transpose2d_bias_backward(
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
    let g = Geometry::new(
        &x,
        &weights,
        ConvOptions::new(options.stride, options.padding, options.dilation, 1),
    );
    let zero = native_zeros([1].into(), &device, dtype).unwrap();
    let positions = g.y[2] * g.y[3];
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
