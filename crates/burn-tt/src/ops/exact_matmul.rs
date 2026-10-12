//! `MatmulMode::Exact`: a matmul as F32 products summed in order, composed from resident SFPU ops.
//!
//! `C[.., m, n] = sum_k A[.., m, k] B[.., k, n]` where every product is an F32 SFPU multiply (to
//! nearest, subnormals flushed) and the sum runs over increasing `k`, sequentially, in F32, from
//! `+0.0`. No FPU and no TF32: the operands keep all 24 bits, and none of the Matrix Unit's group
//! alignment (`docs/learnings/silicon-operating-notes.md`) applies. A sum that is `-0.0` reads `+0.0`,
//! as every SFPU reduction starts from `+0.0`.
//!
//! No new kernel. Per chunk of output elements, two gathers lay the operands out as `[k, x]` (one
//! column per output element, the products' `k` index down the rows); an SFPU multiply forms the
//! products; `float_sum_dim` over dimension 0 of that rank-2 layout is the in-order reduction
//! (`accumulate_in_order`; any other axis is a tree). A final gather joins the chunks. The host builds
//! only addresses, from the *logical* shapes, so transposed and sliced views, batch broadcast and
//! ragged producers need no special case, and nothing is downloaded.
//!
//! Chunking is over output elements, never over `k`: another order across `k` chunks would not be the
//! same sum.
use super::*;

/// Products per chunk (`k` times the chunk's output elements) unless [`set_chunk_products`] says
/// otherwise: the host builds two address tables this long.
const DEFAULT_CHUNK_PRODUCTS: usize = 1 << 20;
static CHUNK_PRODUCTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(DEFAULT_CHUNK_PRODUCTS);

/// Set the products per chunk (at least 1) for the process: the memory the host spends on address
/// tables per chunk, and the largest `k` the exact path accepts. A tuning setting, and how a gate makes
/// small shapes span several chunks.
pub fn set_chunk_products(products: usize) {
    CHUNK_PRODUCTS.store(products.max(1), std::sync::atomic::Ordering::Relaxed);
}
/// Chunks per matmul; each costs a handful of device launches.
const MAX_CHUNKS: usize = 1024;

/// The exact product if the matmul is to run exactly, `None` if it is not (the FPU path runs).
pub(crate) fn exact_matmul(lhs: &TtTensor, rhs: &TtTensor) -> Option<TtTensor> {
    use crate::MatmulMode::Exact;
    let device = lhs.device;
    let mode = crate::matmul_mode::effective(device);
    if mode != Exact {
        return None;
    }
    let refuse = |why: &str| -> ! {
        fail(
            "float_matmul",
            format_args!(
                "exact matmul: {why}: lhs=({}), rhs=({})",
                context(lhs),
                context(rhs)
            ),
        )
    };
    // A device whose own mode is fast and whose engine cannot compose exactly (a mesh) is refused
    // here; a config-exact mesh was refused at attach.
    if !crate::server::supports_exact_matmul(device) {
        refuse("this engine does not support the exact mode");
    }
    let (l, r) = (lhs.shape().to_vec(), rhs.shape().to_vec());
    if l.len() < 2 || r.len() < 2 {
        refuse("operands need rank >= 2");
    }
    let (m, k) = (l[l.len() - 2], l[l.len() - 1]);
    let (k2, n) = (r[r.len() - 2], r[r.len() - 1]);
    if k != k2 {
        refuse("inner dimensions differ");
    }
    // Broadcast the batch axes, right-aligned, as `float_matmul` does.
    let rank = l.len().max(r.len());
    let pad = |s: &[usize]| -> Vec<usize> {
        let mut v = vec![1; rank - s.len()];
        v.extend_from_slice(&s[..s.len() - 2]);
        v
    };
    let (lb, rb) = (pad(&l), pad(&r));
    let mut batch = Vec::new();
    for d in 0..rank - 2 {
        match (lb[d], rb[d]) {
            (a, b) if a == b || b == 1 => batch.push(a),
            (1, b) => batch.push(b),
            _ => refuse("batch axes do not broadcast"),
        }
    }
    let batches: usize = batch.iter().product();
    // No elements, no arithmetic: the ordinary path answers with the right shape.
    if batches == 0 || m == 0 || n == 0 || k == 0 {
        return None;
    }
    if !lhs.is_storable() || !rhs.is_storable() || !crate::server::supports_dram(device) {
        refuse("operands must be resident-storable with GDDR");
    }
    let chunk_products = CHUNK_PRODUCTS.load(std::sync::atomic::Ordering::Relaxed);
    if k > chunk_products {
        refuse(&format!(
            "k = {k} exceeds the {chunk_products}-product chunk, and a sum is never split across k"
        ));
    }
    let outputs = batches
        .checked_mul(m)
        .and_then(|x| x.checked_mul(n))
        .unwrap_or_else(|| refuse("output size overflows"));
    let per_chunk = (chunk_products / k).max(1);
    let chunks = outputs.div_ceil(per_chunk);
    if chunks > MAX_CHUNKS {
        refuse(&format!(
            "{batches}x{m}x{k} @ {k}x{n} needs {chunks} chunks (limit {MAX_CHUNKS}); split the operands"
        ));
    }

    // The flat batch of each operand that broadcasts into each output batch.
    let operand_batch = |dims: &[usize]| -> Vec<usize> {
        (0..batches)
            .map(|mut flat| {
                let (mut at, mut stride) = (0, 1);
                for d in (0..batch.len()).rev() {
                    let coordinate = flat % batch[d];
                    flat /= batch[d];
                    if dims[d] != 1 {
                        at += coordinate * stride;
                    }
                    stride *= dims[d];
                }
                at
            })
            .collect()
    };
    let (lbatch, rbatch) = (operand_batch(&lb), operand_batch(&rb));

    let mut sums = Vec::with_capacity(chunks);
    for chunk in 0..chunks {
        let first = chunk * per_chunk;
        let count = per_chunk.min(outputs - first);
        let mut a_sources = Vec::with_capacity(k * count);
        let mut b_sources = Vec::with_capacity(k * count);
        for p in 0..k {
            for x in first..first + count {
                let (j, i, b) = (x % n, x / n % m, x / (n * m));
                a_sources.push((0, lbatch[b] * m * k + i * k + p));
                b_sources.push((0, rbatch[b] * k * n + p * n + j));
            }
        }
        let a = mapped_native(&[lhs], a_sources, vec![k, count]);
        let b = mapped_native(&[rhs], b_sources, vec![k, count]);
        let product = float::float_mul(a, b);
        sums.push(float::float_sum_dim(product, 0));
    }
    let mut shape = batch;
    shape.extend([m, n]);
    let refs: Vec<_> = sums.iter().collect();
    let sources = (0..outputs)
        .map(|x| (x / per_chunk, x % per_chunk))
        .collect();
    Some(mapped_native(&refs, sources, shape))
}
