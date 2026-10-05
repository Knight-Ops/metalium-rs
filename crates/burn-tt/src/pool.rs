//! NCHW pooling geometry is host metadata; values and gradients stay resident.
use super::*;
use burn_backend::ops::{MaxPool2dBackward, MaxPool2dWithIndices};

struct Geometry {
    input: [usize; 4],
    output: [usize; 4],
    windows: Vec<Vec<[usize; 2]>>,
    divisors: Vec<usize>,
}

fn shape(x: &TtTensor) -> [usize; 4] {
    let shape = x.shape().to_vec();
    assert!(
        shape.len() == 4
            && x.is_storable()
            && matches!(x.dtype(), DType::F32 | DType::BF16)
            && !shape.contains(&0),
        "pooling requires a nonempty resident NCHW tensor: {}",
        context(x)
    );
    checked_elements(&shape);
    shape.try_into().unwrap()
}

fn checked_elements(dims: &[usize]) {
    assert!(
        dims.iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .is_some(),
        "pooling shape overflow"
    );
}

fn adaptive(x: &TtTensor, [oh, ow]: [usize; 2]) -> Geometry {
    let [n, c, h, w] = shape(x);
    assert!(oh > 0 && ow > 0, "adaptive pooling output must be nonempty");
    checked_elements(&[n, c, oh, ow]);
    let mut windows = Vec::new();
    for plane in 0..n * c {
        for r in 0..oh {
            for col in 0..ow {
                let r0 = r.checked_mul(h).unwrap() / oh;
                let r1 = (r + 1).checked_mul(h).unwrap().div_ceil(oh);
                let c0 = col.checked_mul(w).unwrap() / ow;
                let c1 = (col + 1).checked_mul(w).unwrap().div_ceil(ow);
                windows.push(
                    (r0..r1)
                        .flat_map(|i| (c0..c1).map(move |j| [plane * h + i, j]))
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    let divisors = windows.iter().map(Vec::len).collect();
    Geometry {
        input: [n, c, h, w],
        output: [n, c, oh, ow],
        windows,
        divisors,
    }
}

fn regular(
    x: &TtTensor,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    include_pad: bool,
    ceil: bool,
) -> Geometry {
    let [n, c, h, w] = shape(x);
    assert!(
        kernel
            .iter()
            .chain(&stride)
            .chain(&dilation)
            .all(|&n| n > 0),
        "pooling kernel, stride and dilation must be positive"
    );
    let output_dim = |len: usize, axis: usize| {
        let effective = (kernel[axis] - 1)
            .checked_mul(dilation[axis])
            .unwrap()
            .checked_add(1)
            .unwrap();
        let padded = len
            .checked_add(padding[axis].checked_mul(2).unwrap())
            .unwrap();
        let span = padded as i128 - effective as i128;
        let mut count = if ceil {
            -(-span).div_euclid(stride[axis] as i128)
        } else {
            span.div_euclid(stride[axis] as i128)
        } + 1;
        if ceil && (count - 1) * stride[axis] as i128 >= (len + padding[axis]) as i128 {
            count -= 1;
        }
        assert!(count > 0, "empty pooling output");
        usize::try_from(count).expect("pooling output overflow")
    };
    let [oh, ow] = [output_dim(h, 0), output_dim(w, 1)];
    checked_elements(&[n, c, oh, ow]);
    let mut windows = Vec::new();
    let mut divisors = Vec::new();
    for plane in 0..n * c {
        for r in 0..oh {
            for col in 0..ow {
                let mut window = Vec::new();
                let mut padded_count = 0;
                for kr in 0..kernel[0] {
                    for kc in 0..kernel[1] {
                        let ir = r as i128 * stride[0] as i128 + kr as i128 * dilation[0] as i128
                            - padding[0] as i128;
                        let ic = col as i128 * stride[1] as i128 + kc as i128 * dilation[1] as i128
                            - padding[1] as i128;
                        if ir < h as i128 + padding[0] as i128
                            && ic < w as i128 + padding[1] as i128
                        {
                            padded_count += 1;
                        }
                        if ir >= 0 && ir < h as i128 && ic >= 0 && ic < w as i128 {
                            window.push([plane * h + ir as usize, ic as usize]);
                        }
                    }
                }
                assert!(!window.is_empty(), "pooling window contains only padding");
                divisors.push(if include_pad {
                    padded_count
                } else {
                    window.len()
                });
                windows.push(window);
            }
        }
    }
    Geometry {
        input: [n, c, h, w],
        output: [n, c, oh, ow],
        windows,
        divisors,
    }
}

fn constant(device: TtDevice, values: Vec<f32>, dims: [usize; 2]) -> TtTensor {
    let bits = values.into_iter().map(f32::to_bits).collect();
    let id = crate::server::metadata(device, bits, dims, Elem::F32);
    device_result(device, id, dims)
}

/// Group equal-sized windows so ragged padding never enters the SFPU fold.
fn sfpu_forward(
    x: &TtTensor,
    g: &Geometry,
    maximum: bool,
    indices: bool,
) -> (TtTensor, Option<TtTensor>) {
    let source = plain_dram(x);
    let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for (i, w) in g.windows.iter().enumerate() {
        groups.entry(w.len()).or_default().push(i);
    }
    let mut values = Vec::new();
    let mut indexes = Vec::new();
    let mut order = vec![(0, 0); g.windows.len()];
    for (width, ids) in groups {
        let mapping = ids
            .iter()
            .flat_map(|&i| g.windows[i].iter().copied())
            .collect();
        let (id, dims) =
            crate::server::repack(x.device, source.buffer.id, mapping, [ids.len(), width]);
        let packed = device_result(x.device, id, dims);
        // Select the value through the same first-tie/first-NaN index as the
        // backward path. SFPU's total-order max differs for negative NaNs.
        let local = maximum.then(|| float::float_argmax(packed.clone(), 1, IntDType::I32));
        let mut out = if maximum {
            float::float_gather_bits(1, packed.clone(), local.as_ref().unwrap().clone())
        } else {
            float::float_sum_dim(packed.clone(), 1)
        };
        if !maximum {
            let count = constant(
                x.device,
                ids.iter().map(|&i| g.divisors[i] as f32).collect(),
                [ids.len(), 1],
            );
            out = float::float_div(out, count);
        }
        if indices {
            assert!(
                g.input[2].checked_mul(g.input[3]).unwrap() <= 1 << 23,
                "pooling spatial indices exceed exact F32 metadata range"
            );
            let positions = ids
                .iter()
                .flat_map(|&i| {
                    g.windows[i]
                        .iter()
                        .map(|&[r, c]| ((r % g.input[2]) * g.input[3] + c) as f32)
                })
                .collect();
            let positions = constant(x.device, positions, [ids.len(), width]);
            let selected = float::float_gather(1, positions, local.unwrap());
            indexes.push(float::float_into_int(selected, IntDType::I32));
        }
        let group = values.len();
        for (row, &i) in ids.iter().enumerate() {
            order[i] = (group, row);
        }
        values.push(out);
    }
    let join = |inputs: &[TtTensor]| {
        let ids = inputs.iter().map(|t| t.to_dram().buffer.id).collect();
        let (id, dims) = crate::server::gather_rows(x.device, ids, order.clone(), 1);
        device_result_shaped(
            x.device,
            id,
            dims,
            burn_backend::Shape::new(dims),
            inputs[0].dtype(),
        )
    };
    let values = repack_shape(&join(&values), g.output.to_vec());
    let indexes = indices.then(|| repack_shape(&join(&indexes), g.output.to_vec()));
    (values, indexes)
}

fn mean_forward(x: TtTensor, g: Geometry) -> TtTensor {
    assert!(
        g.divisors.iter().all(|&n| n > 0 && n <= 1 << 24),
        "pooling divisor exceeds exact F32 metadata range"
    );
    if x.dtype() == DType::BF16 {
        let source = plain_dram(&x);
        let dims = [g.output[0] * g.output[1] * g.output[2], g.output[3]];
        let (id, dims) =
            crate::server::pool_bf16(x.device, source.buffer.id, g.windows, g.divisors, dims);
        let output = device_result_shaped(
            x.device,
            id,
            dims,
            burn_backend::Shape::new(g.output),
            DType::F32,
        );
        cast_native(output, DType::BF16)
    } else {
        sfpu_forward(&x, &g, false, false).0
    }
}

fn backward(x: &TtTensor, grad: TtTensor, indices: Option<TtTensor>, g: Geometry) -> TtTensor {
    if indices.is_none() {
        assert!(
            g.divisors.iter().all(|&n| n > 0 && n <= 1 << 24),
            "pooling divisor exceeds exact F32 metadata range"
        );
    }
    assert_eq!(
        grad.shape().to_vec(),
        g.output,
        "pool gradient shape mismatch"
    );
    assert_eq!(
        x.device, grad.device,
        "pool gradients must share the input device"
    );
    let source = plain_dram(&grad);
    let [n, c, h, w] = g.input;
    let mut mapping = Vec::new();
    let mut targets = Vec::new();
    let mut divisors = Vec::new();
    let mut candidates = Vec::new();
    for (i, window) in g.windows.iter().enumerate() {
        for &[r, col] in window {
            mapping.push([i / g.output[3], i % g.output[3]]);
            targets.push(r * w + col);
            divisors.push(g.divisors[i] as f32);
            candidates.push(((r % h) * w + col) as u32);
        }
    }
    let count = mapping.len();
    let (id, dims) = crate::server::repack(x.device, source.buffer.id, mapping.clone(), [count, 1]);
    let mut values = device_result(x.device, id, dims);
    if let Some(indices) = indices {
        assert_eq!(
            indices.shape().to_vec(),
            g.output,
            "pool index shape mismatch"
        );
        assert_eq!(
            indices.device, x.device,
            "pool indices must share the input device"
        );
        let source = plain_dram(&indices);
        let (id, dims) = crate::server::repack(x.device, source.buffer.id, mapping, [count, 1]);
        let actual = device_result_shaped(
            x.device,
            id,
            dims,
            burn_backend::Shape::new(dims),
            DType::I32,
        );
        let id = crate::server::metadata(x.device, candidates, [count, 1], Elem::I32);
        let expected = device_result_shaped(
            x.device,
            id,
            [count, 1],
            burn_backend::Shape::new([count, 1]),
            DType::I32,
        );
        let ne = super::bool::bool_not(super::int::int_equal(
            actual,
            expected,
            burn_backend::BoolDType::Native,
        ));
        values = float::float_mask_fill(values, ne, 0.0.into());
    } else {
        values = float::float_div(values, constant(x.device, divisors, [count, 1]));
    }
    let zero = constant(x.device, vec![0.0; n * c * h * w], [n * c * h * w, 1]);
    let (id, dims) = crate::server::rows_add(
        x.device,
        zero.to_dram().buffer.id,
        targets,
        values.to_dram().buffer.id,
    );
    repack_shape(&device_result(x.device, id, dims), g.input.to_vec())
}

pub fn adaptive_avg_pool2d(x: TtTensor, output_size: [usize; 2]) -> TtTensor {
    let g = adaptive(&x, output_size);
    mean_forward(x, g)
}
pub fn adaptive_avg_pool2d_backward(x: TtTensor, grad: TtTensor) -> TtTensor {
    let s = shape(&grad);
    let g = adaptive(&x, [s[2], s[3]]);
    backward(&x, grad, None, g)
}
pub fn avg_pool2d(
    x: TtTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> TtTensor {
    let g = regular(
        &x,
        kernel_size,
        stride,
        padding,
        [1, 1],
        count_include_pad,
        ceil_mode,
    );
    mean_forward(x, g)
}
pub fn avg_pool2d_backward(
    x: TtTensor,
    grad: TtTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> TtTensor {
    let g = regular(
        &x,
        kernel_size,
        stride,
        padding,
        [1, 1],
        count_include_pad,
        ceil_mode,
    );
    backward(&x, grad, None, g)
}
pub fn max_pool2d(
    x: TtTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> TtTensor {
    let g = regular(&x, kernel_size, stride, padding, dilation, false, ceil_mode);
    sfpu_forward(&x, &g, true, false).0
}
pub fn max_pool2d_with_indices(
    x: TtTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> MaxPool2dWithIndices<TtBackend> {
    let g = regular(&x, kernel_size, stride, padding, dilation, false, ceil_mode);
    let (output, indices) = sfpu_forward(&x, &g, true, true);
    MaxPool2dWithIndices {
        output,
        indices: indices.unwrap(),
    }
}
// Matches Burn's ModuleOps signature.
#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_with_indices_backward(
    x: TtTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    output_grad: TtTensor,
    indices: TtTensor,
) -> MaxPool2dBackward<TtBackend> {
    let g = regular(&x, kernel_size, stride, padding, dilation, false, ceil_mode);
    MaxPool2dBackward {
        x_grad: backward(&x, output_grad, Some(indices), g),
    }
}
