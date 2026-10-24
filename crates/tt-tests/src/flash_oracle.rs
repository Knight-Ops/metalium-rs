//! The host oracle of decode attention (`docs/plans/long-context.md` Part 2): what the card
//! computes for one key/value head, exactly, in today's order ([`unsplit`],
//! `tt_kernels::attention`'s seven kernels and burn-tt's composition alike).
//!
//! Each step is the program the card runs, run on the host: the products by the Matrix Unit's
//! measured group adder ([`crate::matrix_unit`]), the scale and mask, `SUB_EXP` and `DIV` by the
//! SFPU interpreter (`sfpu::ops::reference_op`), the row maximum and sum by the reduction's own
//! programs (`sfpu::reduce::reference`). Nothing is rounded the way IEEE arithmetic would round
//! it, and no tolerance is involved: a gate requires the device's bits.
//!
//! The Matrix Unit model is measured for finite operands without subnormals; a head whose
//! attended keys or values hold NaN, infinity or a subnormal is outside it.

use tt_kernels::sfpu::ops::kind_sfpu::{CLAMP_MIN, DIV, SCALE_MASK_FILL, SUB_EXP};
use tt_kernels::sfpu::ops::{reference_op, Broadcast};
use tt_kernels::sfpu::reduce::{reference as reduce, Axis, ReduceOp};

use crate::matrix_unit::{dot, MEASURED};

/// One key/value head's operands: `m` query rows of `d` (`q`, row-major), `l` keys of `d` and
/// values of `dv` (`k`, `v`, row-major by position), the positions not attended (`masked`), and
/// the scores' scale.
pub struct Head<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub masked: &'a [bool],
    pub m: usize,
    pub d: usize,
    pub dv: usize,
    pub scale: f32,
}

impl Head<'_> {
    pub fn l(&self) -> usize {
        self.masked.len()
    }
}

/// `x` (`[rows, cols]`, row-major) times `y`'s columns (`[k, cols_y]`, row-major), each output a
/// `Dst` chain over `k` in order as the Matrix Unit adds it.
fn product(x: &[f32], rows: usize, k: usize, y: &[f32], cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let col: Vec<f32> = (0..k).map(|p| y[p * cols + j]).collect();
            out[i * cols + j] = dot(&x[i * k..(i + 1) * k], &col, MEASURED).0;
        }
    }
    out
}

/// The masked, scaled scores `[m, l]`: `q K^T` (a key's `d` in order), then `SCALE_MASK_FILL`.
pub fn scores(h: &Head) -> Vec<f32> {
    let (m, d, l) = (h.m, h.d, h.l());
    // `K^T`: `[d, l]`.
    let kt: Vec<f32> = (0..d * l).map(|i| h.k[(i % l) * d + i / l]).collect();
    let s = product(h.q, m, d, &kt, l);
    let mask: Vec<f32> = h
        .masked
        .iter()
        .map(|&x| if x { 1.0 } else { 0.0 })
        .collect();
    reference_op(
        SCALE_MASK_FILL,
        [h.scale, f32::NEG_INFINITY],
        Broadcast::Row,
        &[&s, &mask],
        m,
        l,
    )
}

/// A row reduction over `[m, n]` with its clamp, as the card finishes one: `[m]`.
fn reduce_clamped(op: ReduceOp, clamp: f32, x: &[f32], m: usize, n: usize) -> Vec<f32> {
    let r = reduce(op, Axis::Cols, x, m, n);
    reference_op(CLAMP_MIN, [clamp, 0.0], Broadcast::None, &[&r], m, 1)
}

/// `softmax(scores) V` for one head, as today's op and the composition compute it: the row
/// maximum (clamped at `-f32::MAX`), `SUB_EXP` against it, the row sum (clamped at the least
/// normal), `DIV`, then `P V` (a value column's positions in order). `[m, dv]`.
pub fn unsplit(h: &Head) -> Vec<f32> {
    let (m, l) = (h.m, h.l());
    let t = scores(h);
    let max = reduce_clamped(ReduceOp::Max, -f32::MAX, &t, m, l);
    let e = reference_op(SUB_EXP, [0.0, 0.0], Broadcast::Col, &[&t, &max], m, l);
    let sum = reduce_clamped(ReduceOp::Sum, f32::MIN_POSITIVE, &e, m, l);
    let p = reference_op(DIV, [0.0, 0.0], Broadcast::Col, &[&e, &sum], m, l);
    product(&p, m, l, h.v, h.dv)
}
