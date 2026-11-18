//! The gated delta rule of a Gated DeltaNet ([`crate::qwen35`]), token by token or a chunk at a
//! time.
//!
//! Per head, with the state `S` (`[k_dim, v_dim]`), a token's decay `γ = exp(g)`, write strength
//! `β`, unit key `k`, value `v` and query `q`:
//!
//! ```text
//! S <- γ S;   S <- S + k (β (v - kᵀ S))ᵀ;   o = qᵀ S
//! ```
//!
//! [`by_token`] runs that as written, a few ops a token: bit for bit the recurrence, which the
//! reference contract keeps. Its op count grows with the tokens, which makes a long prompt's
//! chunk (and its capture) slow.
//!
//! [`chunked`] computes the same recurrence for `C` tokens at once in its WY form (as `fla`'s
//! `chunk_gated_delta_rule` and the tt-metal prefill do), with `G` the decays' cumulative sum in
//! log space and `Γ = exp(G)`:
//!
//! ```text
//! A = strictly_lower(diag(β) K Kᵀ ⊙ exp(Gᵢ - Gⱼ))     T = (I + A)⁻¹
//! W = T (diag(β Γ) K)                                 U = T (diag(β) V)
//! V' = U - W S
//! O  = diag(Γ) Q S + lower(Q Kᵀ ⊙ exp(Gᵢ - Gⱼ)) V'
//! S' = exp(G_C) S + (diag(exp(G_C - G)) K)ᵀ V'
//! ```
//!
//! `T` is the Neumann series `Σ (-A)ʲ` (exact: `A` is strictly lower, so `Aᶜ = 0`), summed by
//! Horner's rule `R <- I - A R`, `C - 2` products. The doubling form (`(I - A)(I + A²)(I + A⁴)…`)
//! has fewer products but overflows F32 on Qwen3.5's undamped `A` (`‖A‖ ≈ 19`, tt-metal's
//! `_solve_lower_triangular_ttnn`, PROTOKOLL #93). Every exponent is of a difference that is at
//! most zero (`g ≤ 0`), so nothing overflows. The state is read and written once a chunk instead
//! of once a token. The result is the recurrence's to rounding, not bit for bit: the performance
//! contract's choice.
//!
//! A padded chunk's padding rows have `g = 0` and `β = 0`: they add nothing to `A`, `W`, `U`,
//! `V'` or `S'`, and `G_C` is the last real token's, so the state is the prompt's.

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

/// The longest chunk [`chunked`] solves at once; a longer one is split. 64 keeps `T`'s Horner
/// sum at 62 products of `[heads, 64, 64]`.
pub const MAX_CHUNK: usize = 64;

/// The token-by-token recurrence over `t` tokens: `q`, `k` `[1, heads, t, k_dim]`, `v`
/// `[1, heads, t, v_dim]`, `decay` (`γ`, as a column of the state) `[1, t, heads, k_dim]`, `beta`
/// `[1, heads, t, 1]`, all F32; `s` `[1, heads, k_dim, v_dim]` carried. The outputs
/// `[1, heads, t, v_dim]`.
pub fn by_token<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    decay: Tensor<B, 4>,
    beta: Tensor<B, 4>,
    s: &mut Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [n, vh, t, kd] = k.dims();
    let vd = v.dims()[3];
    let mut outs = Vec::with_capacity(t);
    for i in 0..t {
        let row =
            |x: &Tensor<B, 4>, width: usize| x.clone().slice([0..n, 0..vh, i..i + 1, 0..width]);
        let (q_i, k_i, v_i) = (row(&q, kd), row(&k, kd), row(&v, vd));
        let decay_i = decay
            .clone()
            .slice([0..n, i..i + 1, 0..vh, 0..kd])
            .reshape([n, vh, kd, 1]);
        let state = s.clone() * decay_i;
        // What the state already predicts for this key, and the correction written for it
        // (the key as a column: `k^T`, the same values).
        let predicted = k_i.clone().matmul(state.clone());
        let delta = (v_i - predicted) * row(&beta, 1);
        let state = state + k_i.reshape([n, vh, kd, 1]).matmul(delta);
        outs.push(q_i.matmul(state.clone()));
        *s = state;
    }
    Tensor::cat(outs, 2)
}

/// The masks of a `c x c` chunk, `[1, heads, c, c]`: made from positions on the device (a
/// compare, no upload, so a capture may make them).
pub struct Triangles<B: Backend> {
    /// `i < j`: what neither `exp(Gᵢ - Gⱼ)` nor the output's `Q Kᵀ` keeps.
    later: Tensor<B, 4, Bool>,
    /// `i <= j`: what `A` (strictly lower) does not keep.
    not_below: Tensor<B, 4, Bool>,
    /// The identity, F32.
    eye: Tensor<B, 4>,
}

impl<B: Backend> Triangles<B> {
    /// From `positions` (`[≥ c]`, `0, 1, 2, …` on the device: a cache's), for `heads` heads.
    pub fn new(positions: &Tensor<B, 1, Int>, c: usize, heads: usize) -> Self {
        let idx = positions.clone().narrow(0, 0, c);
        let rows = idx.clone().reshape([c, 1]).expand([c, c]);
        let cols = idx.reshape([1, c]).expand([c, c]);
        let wide = |m: Tensor<B, 2, Bool>| m.unsqueeze::<4>().expand([1, heads, c, c]);
        Self {
            later: wide(rows.clone().lower(cols.clone())),
            not_below: wide(rows.clone().lower_equal(cols.clone())),
            eye: rows.equal(cols).float().unsqueeze::<4>().expand([1, heads, c, c]),
        }
    }
}

/// [`by_token`]'s recurrence a chunk at a time (the module docs): the same inputs but the decay
/// in log space, `g` `[1, heads, t, 1]` (`0` on padding rows, as `beta` is `0` there).
/// `positions` gives the masks ([`Triangles`]); chunks of [`MAX_CHUNK`], the last shorter.
pub fn chunked<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    g: Tensor<B, 4>,
    beta: Tensor<B, 4>,
    s: &mut Tensor<B, 4>,
    positions: &Tensor<B, 1, Int>,
) -> Tensor<B, 4> {
    let [n, vh, t, kd] = k.dims();
    let vd = v.dims()[3];
    let mut outs = Vec::with_capacity(t.div_ceil(MAX_CHUNK));
    let mut full: Option<Triangles<B>> = None;
    let mut a = 0;
    while a < t {
        let c = MAX_CHUNK.min(t - a);
        let rows = |x: &Tensor<B, 4>, w: usize| x.clone().slice([0..n, 0..vh, a..a + c, 0..w]);
        // The full chunks share their masks; a shorter last one makes its own.
        let rest;
        let tri = if c == MAX_CHUNK {
            &*full.get_or_insert_with(|| Triangles::new(positions, c, vh))
        } else {
            rest = Triangles::new(positions, c, vh);
            &rest
        };
        let out = chunk(
            rows(&q, kd),
            rows(&k, kd),
            rows(&v, vd),
            rows(&g, 1),
            rows(&beta, 1),
            s,
            tri,
        );
        outs.push(out);
        a += c;
    }
    if outs.len() == 1 {
        outs.pop().expect("one chunk")
    } else {
        Tensor::cat(outs, 2)
    }
}

/// One chunk of `c` tokens ([`chunked`]).
fn chunk<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    g: Tensor<B, 4>,
    beta: Tensor<B, 4>,
    s: &mut Tensor<B, 4>,
    tri: &Triangles<B>,
) -> Tensor<B, 4> {
    let [n, vh, c, kd] = k.dims();
    let vd = v.dims()[3];
    let square = [n, vh, c, c];
    // The cumulative log decay `G` `[.., c, 1]`, and `exp(Gᵢ - Gⱼ)` on and below the diagonal
    // (zero above, where the difference would be positive).
    let big_g = g.cumsum(2);
    let gaps = big_g.clone().expand(square) - big_g.clone().swap_dims(2, 3).expand(square);
    let decays = gaps
        .mask_fill(tri.later.clone(), 0.0)
        .exp()
        .mask_fill(tri.later.clone(), 0.0);
    let kt = k.clone().swap_dims(2, 3);
    // `A`: row `i` scaled by `βᵢ`, strictly below the diagonal.
    let a = (k.clone().matmul(kt.clone()) * decays.clone() * beta.clone().expand(square))
        .mask_fill(tri.not_below.clone(), 0.0);
    // `T = (I + A)⁻¹ = Σ (-A)ʲ` by Horner: `R <- I - A R`, from `R = I - A`.
    let eye = tri.eye.clone();
    let mut t_inv = eye.clone() - a.clone();
    for _ in 2..c {
        t_inv = eye.clone() - a.clone().matmul(t_inv);
    }
    let gamma = big_g.clone().exp();
    let w = t_inv
        .clone()
        .matmul(k.clone() * (beta.clone() * gamma.clone()).expand([n, vh, c, kd]));
    let u = t_inv.matmul(v * beta.expand([n, vh, c, vd]));
    let state = s.clone();
    let v_new = u - w.matmul(state.clone());
    let out = (q.clone() * gamma.expand([n, vh, c, kd])).matmul(state.clone())
        + (q.matmul(kt) * decays).matmul(v_new.clone());
    // The state after the chunk: decayed by the whole chunk, each token's write by what follows.
    let last = big_g.clone().slice([0..n, 0..vh, c - 1..c, 0..1]);
    let tail = (last.clone().expand([n, vh, c, 1]) - big_g).exp();
    *s = state * last.exp().expand([n, vh, kd, vd])
        + (k * tail.expand([n, vh, c, kd]))
            .swap_dims(2, 3)
            .matmul(v_new);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, TensorData};
    use burn_flex::{Flex, FlexDevice};

    fn unit(x: Tensor<Flex, 4>) -> Tensor<Flex, 4> {
        let norm = (x.clone() * x.clone()).sum_dim(3).sqrt();
        x / norm
    }

    /// Random inputs as a Gated DeltaNet makes them: unit keys and queries, `β` in (0, 1), the
    /// decay's log in (-1, 0), and the last `pad` rows padding (`g = 0`, `β = 0`).
    fn inputs(
        heads: usize,
        t: usize,
        kd: usize,
        vd: usize,
        pad: usize,
    ) -> [Tensor<Flex, 4>; 5] {
        let d = FlexDevice;
        let r = |s: [usize; 4]| Tensor::<Flex, 4>::random(s, Distribution::Normal(0.0, 1.0), &d);
        let q = unit(r([1, heads, t, kd])).mul_scalar((kd as f32).powf(-0.5));
        let k = unit(r([1, heads, t, kd]));
        let v = r([1, heads, t, vd]);
        let keep: Vec<f32> = (0..t).map(|i| if i < t - pad { 1.0 } else { 0.0 }).collect();
        let keep = Tensor::<Flex, 4>::from_data(TensorData::new(keep, [1, 1, t, 1]), &d)
            .expand([1, heads, t, 1]);
        let u = |lo, hi| Tensor::<Flex, 4>::random([1, heads, t, 1], Distribution::Uniform(lo, hi), &d);
        let beta = u(0.0, 1.0) * keep.clone();
        let g = u(-1.0, 0.0) * keep;
        [q, k, v, g, beta]
    }

    fn worst(a: Tensor<Flex, 4>, b: Tensor<Flex, 4>) -> f32 {
        (a - b).abs().max().into_scalar()
    }

    /// The chunked form against the recurrence: outputs and the final state, for chunks of one
    /// size, split ones (past [`MAX_CHUNK`]) and padded ones.
    #[test]
    fn chunks_compute_the_recurrence() {
        let (heads, kd, vd) = (3, 16, 8);
        for (t, pad) in [(2, 0), (7, 0), (16, 5), (64, 0), (100, 13)] {
            let [q, k, v, g, beta] = inputs(heads, t, kd, vd, pad);
            let s0 = Tensor::<Flex, 4>::random(
                [1, heads, kd, vd],
                Distribution::Normal(0.0, 1.0),
                &FlexDevice,
            );
            let decay = g
                .clone()
                .exp()
                .swap_dims(1, 2)
                .expand([1, t, heads, kd]);
            let mut s_loop = s0.clone();
            let want = by_token(
                q.clone(),
                k.clone(),
                v.clone(),
                decay,
                beta.clone(),
                &mut s_loop,
            );
            let mut s_chunk = s0;
            let positions = Tensor::<Flex, 1, Int>::arange(0..t as i64, &FlexDevice);
            let got = chunked(q, k, v, g, beta, &mut s_chunk, &positions);
            let (o, st) = (worst(got, want), worst(s_chunk, s_loop));
            eprintln!("t {t}, pad {pad}: outputs {o:e}, state {st:e}");
            assert!(o < 1e-4 && st < 1e-4, "t {t}: outputs {o}, state {st}");
        }
    }
}
