//! Device sort family: `sort`, `sort_with_indices`, `argsort` and `argtopk` of `F32` and `I32`
//! tensors along any axis run as one SFPU program (`tt_kernels::sfpu::sort`),
//! stable and deterministic: ties break by ascending original index, in either
//! direction, and `F32` follows `f32::total_cmp` (NaNs of both signs, `-0 < +0`).
//! Burn's own sorts are unstable `total_cmp` sorts, so the results agree with
//! Burn wherever Burn's order is determined and pick the one stable order where
//! it is not. Nothing is read back: the axis is laid out on the card by native
//! repacks (host-built addresses only) and the results are repacked into the
//! output shapes. `topk` follows through Burn's default (`sort` then `select`).
//!
//! The axis length is bounded (`tt_kernels::sfpu::sort::max_axis`): longer axes
//! fail explicitly, naming the operation and the input.

use super::*;
use crate::server::Elem;
use crate::unsupported::{context, fail};
use crate::views::Strided;
use burn_backend::{DType, IntDType, TensorMetadata};
use core::primitive::bool;
use tt_kernels::sfpu::sort::{self as device_sort, plane_coord, plane_dims, plane_element, Spec};

/// What one sort call is about: the input's axis in problems and positions.
struct Geometry {
    shape: Vec<usize>,
    dim: usize,
    n: usize,
    problems: usize,
}

impl Geometry {
    /// `shape` with the axis length replaced.
    fn with_axis(&self, n: usize) -> Vec<usize> {
        let mut s = self.shape.clone();
        s[self.dim] = n;
        s
    }
}

/// The problems an axis splits a tensor into: every combination of the
/// other coordinates, row-major.
fn problems(shape: &[usize], dim: usize) -> usize {
    shape
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != dim)
        .map(|(_, &d)| d)
        .product()
}

/// Refuse what the device sort cannot do, naming the operation and the
/// input, then describe the axis.
fn geometry(
    op: &str,
    tensor: &TtTensor,
    dim: usize,
    elem: DType,
    indices: bool,
    detail: &str,
) -> Geometry {
    let shape = tensor.shape().to_vec();
    let refuse = |why: String| -> ! {
        fail(
            op,
            format_args!("{why}: {}, dim={dim}{detail}", context(tensor)),
        )
    };
    if tensor.dtype() != elem {
        refuse(format!("the device sort takes {elem:?} elements only"));
    }
    if shape.is_empty() || dim >= shape.len() {
        refuse("axis out of range".into());
    }
    if shape.contains(&0) {
        refuse("empty tensors have nothing to sort".into());
    }
    if !tensor.is_storable() || !crate::server::supports_dram(tensor.device) {
        refuse("the tensor cannot be held on this device".into());
    }
    let n = shape[dim];
    let max = device_sort::max_axis(indices);
    if n > max {
        refuse(format!(
            "axis length {n} exceeds the device sort's limit of {max}{}",
            if indices { " with indices" } else { "" }
        ));
    }
    Geometry {
        problems: problems(&shape, dim),
        shape,
        dim,
        n,
    }
}

/// Flat row-major position of the problem `q`'s first element (axis
/// coordinate zero), and the flat stride of the axis.
fn flat_bases(g: &Geometry) -> (Vec<usize>, usize) {
    let rank = g.shape.len();
    let mut stride = vec![1usize; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        stride[k] = stride[k + 1] * g.shape[k + 1];
    }
    let bases = (0..g.problems)
        .map(|q| {
            let mut rest = q;
            let mut flat = 0;
            for k in (0..rank).rev() {
                if k == g.dim {
                    continue;
                }
                flat += (rest % g.shape[k]) * stride[k];
                rest /= g.shape[k];
            }
            flat
        })
        .collect();
    (bases, stride[g.dim])
}

/// The axis laid out as planes on the device: a native repack whose only
/// host work is address metadata. Padding positions and lanes copy an
/// arbitrary element; the kernel never reads them as data.
fn to_planes(tensor: &TtTensor, g: &Geometry) -> TtTensor {
    let view = tensor
        .as_strided()
        .unwrap_or_else(|| Strided::of(tensor.to_dram(), &g.shape));
    let [rows, cols] = plane_dims(g.problems, g.n);
    let (bases, stride) = flat_bases(g);
    let filler = view.at(&g.shape, 0);
    let mut sources = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let (q, e) = plane_element([r, c]);
            sources.push(if q < g.problems && e < g.n {
                view.at(&g.shape, bases[q] + e * stride)
            } else {
                filler
            });
        }
    }
    let (id, dims) =
        crate::server::repack(tensor.device, view.src.buffer.id, sources, [rows, cols]);
    device_view(tensor.device, id, dims, None, tensor.dtype())
}

/// The first `m` positions of every problem of `planes`, as a tensor of
/// the input's shape with the axis `m` long.
fn from_planes(planes: &TtTensor, g: &Geometry, m: usize, dtype: DType) -> TtTensor {
    let shape = g.with_axis(m);
    let rank = shape.len();
    let dims = crate::tensor::stored_dims(&shape).expect("nonempty output shape");
    let numel: usize = shape.iter().product();
    let mut sources = Vec::with_capacity(numel);
    for flat in 0..numel {
        let mut rest = flat;
        let (mut q, mut stride_q, mut e) = (0, 1, 0);
        for k in (0..rank).rev() {
            let i = rest % shape[k];
            rest /= shape[k];
            if k == g.dim {
                e = i;
            } else {
                q += i * stride_q;
                stride_q *= shape[k];
            }
        }
        sources.push(plane_coord(q, e));
    }
    let source = plain_dram(planes);
    let (id, dims) = crate::server::repack(planes.device, source.buffer.id, sources, dims);
    device_result_shaped(
        planes.device,
        id,
        dims,
        burn_backend::Shape::from(shape),
        dtype,
    )
}

struct Sorted {
    g: Geometry,
    keys: TtTensor,
    indices: Option<TtTensor>,
}

/// Sort every problem on the device: the planes of the sorted keys and,
/// if asked, of the original indices.
fn sort(tensor: &TtTensor, g: Geometry, elem: Elem, descending: bool, indices: bool) -> Sorted {
    let spec = Spec {
        elem,
        n: g.n,
        descending,
        indices,
    };
    let planes = to_planes(tensor, &g);
    let dims = planes.stored().expect("planes are a matrix");
    let made = crate::server::sort_planes(tensor.device, planes.to_dram().buffer.id, spec, dims);
    Sorted {
        keys: device_view(
            tensor.device,
            made.keys.0,
            made.keys.1,
            None,
            tensor.dtype(),
        ),
        indices: made
            .indices
            .map(|i| device_view(tensor.device, i.0, i.1, None, DType::I32)),
        g,
    }
}

impl Sorted {
    fn values(&self, m: usize) -> TtTensor {
        from_planes(&self.keys, &self.g, m, self.keys.dtype())
    }

    fn index_tensor(&self, m: usize) -> TtTensor {
        from_planes(
            self.indices.as_ref().expect("sorted with indices"),
            &self.g,
            m,
            DType::I32,
        )
    }
}

/// Indices are `I32` on this device.
fn require_i32(op: &str, tensor: &TtTensor, dtype: IntDType) {
    if dtype != IntDType::I32 {
        fail(
            op,
            format_args!(
                "index dtype {dtype:?} (the device produces I32): {}",
                context(tensor)
            ),
        );
    }
}

/// `total_cmp` order, ascending or descending, ties by original index.
pub fn float_sort(tensor: TtTensor, dim: usize, descending: bool) -> TtTensor {
    let g = geometry(
        "float_sort",
        &tensor,
        dim,
        DType::F32,
        false,
        &format!(", descending={descending}"),
    );
    let n = g.n;
    sort(&tensor, g, Elem::F32, descending, false).values(n)
}

/// The sorted values and the original index of each.
pub fn float_sort_with_indices(
    tensor: TtTensor,
    dim: usize,
    descending: bool,
    indices_dtype: IntDType,
) -> (TtTensor, TtTensor) {
    require_i32("float_sort_with_indices", &tensor, indices_dtype);
    let g = geometry(
        "float_sort_with_indices",
        &tensor,
        dim,
        DType::F32,
        true,
        &format!(", descending={descending}"),
    );
    let n = g.n;
    let sorted = sort(&tensor, g, Elem::F32, descending, true);
    (sorted.values(n), sorted.index_tensor(n))
}

/// The original index of each sorted element.
pub fn float_argsort(
    tensor: TtTensor,
    dim: usize,
    descending: bool,
    out_dtype: IntDType,
) -> TtTensor {
    require_i32("float_argsort", &tensor, out_dtype);
    let g = geometry(
        "float_argsort",
        &tensor,
        dim,
        DType::F32,
        true,
        &format!(", descending={descending}"),
    );
    let n = g.n;
    sort(&tensor, g, Elem::F32, descending, true).index_tensor(n)
}

/// The `k` largest values, largest first: the first `k` of the descending
/// sort, with no host-built index tensor to select by (Burn's default
/// would upload an `arange`).
pub fn float_topk(tensor: TtTensor, dim: usize, k: usize) -> TtTensor {
    let g = geometry(
        "float_topk",
        &tensor,
        dim,
        DType::F32,
        false,
        &format!(", k={k}"),
    );
    if k == 0 || k > g.n {
        fail(
            "float_topk",
            format_args!("k={k} outside 1..={}: {}", g.n, context(&tensor)),
        );
    }
    sort(&tensor, g, Elem::F32, true, false).values(k)
}

/// The indices of the `k` largest elements, largest first (ties by
/// original index): the first `k` of the descending argsort.
pub fn float_argtopk(tensor: TtTensor, dim: usize, k: usize, out_dtype: IntDType) -> TtTensor {
    require_i32("float_argtopk", &tensor, out_dtype);
    let g = geometry(
        "float_argtopk",
        &tensor,
        dim,
        DType::F32,
        true,
        &format!(", k={k}"),
    );
    if k == 0 || k > g.n {
        fail(
            "float_argtopk",
            format_args!("k={k} outside 1..={}: {}", g.n, context(&tensor)),
        );
    }
    sort(&tensor, g, Elem::F32, true, true).index_tensor(k)
}

/// Two's-complement order, ties by original index.
pub fn int_sort(tensor: TtTensor, dim: usize, descending: bool) -> TtTensor {
    let g = geometry(
        "int_sort",
        &tensor,
        dim,
        DType::I32,
        false,
        &format!(", descending={descending}"),
    );
    let n = g.n;
    sort(&tensor, g, Elem::I32, descending, false).values(n)
}

pub fn int_sort_with_indices(
    tensor: TtTensor,
    dim: usize,
    descending: bool,
) -> (TtTensor, TtTensor) {
    let g = geometry(
        "int_sort_with_indices",
        &tensor,
        dim,
        DType::I32,
        true,
        &format!(", descending={descending}"),
    );
    let n = g.n;
    let sorted = sort(&tensor, g, Elem::I32, descending, true);
    (sorted.values(n), sorted.index_tensor(n))
}

pub fn int_argsort(tensor: TtTensor, dim: usize, descending: bool) -> TtTensor {
    let g = geometry(
        "int_argsort",
        &tensor,
        dim,
        DType::I32,
        true,
        &format!(", descending={descending}"),
    );
    let n = g.n;
    sort(&tensor, g, Elem::I32, descending, true).index_tensor(n)
}

/// The `k` largest values, largest first.
pub fn int_topk(tensor: TtTensor, dim: usize, k: usize) -> TtTensor {
    let g = geometry(
        "int_topk",
        &tensor,
        dim,
        DType::I32,
        false,
        &format!(", k={k}"),
    );
    if k == 0 || k > g.n {
        fail(
            "int_topk",
            format_args!("k={k} outside 1..={}: {}", g.n, context(&tensor)),
        );
    }
    sort(&tensor, g, Elem::I32, true, false).values(k)
}

/// The indices of the `k` largest elements, largest first.
pub fn int_argtopk(tensor: TtTensor, dim: usize, k: usize) -> TtTensor {
    let g = geometry(
        "int_argtopk",
        &tensor,
        dim,
        DType::I32,
        true,
        &format!(", k={k}"),
    );
    if k == 0 || k > g.n {
        fail(
            "int_argtopk",
            format_args!("k={k} outside 1..={}: {}", g.n, context(&tensor)),
        );
    }
    sort(&tensor, g, Elem::I32, true, true).index_tensor(k)
}
