//! Rotary position embeddings (RoPE) in the rotate-half layout, for any model that uses them.

use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData};

/// Rotary tables for a run of positions: `[1, 1, t, head_dim]` each. `sin` holds its first half
/// negated, the sign `rotate_half` would put on `x`'s second half ([`Rope::apply`]).
#[derive(Clone, Debug)]
pub struct Rope<B: Backend> {
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
}

impl<B: Backend> Rope<B> {
    /// `angle = position * theta^(-2i / head_dim)`, the first half of the row repeated as the
    /// second (the rotate-half layout), `sin`'s first half negated. The positions and the
    /// frequencies are host data that the shapes fix; the products and the `sin` and `cos` run on
    /// the device.
    pub fn new(
        start: usize,
        t: usize,
        head_dim: usize,
        theta: f64,
        dtype: DType,
        device: &B::Device,
    ) -> Self {
        let positions: Vec<f32> = (start..start + t).map(|p| p as f32).collect();
        let freqs: Vec<f32> = (0..head_dim / 2)
            .map(|i| theta.powf(-(2.0 * i as f64) / head_dim as f64) as f32)
            .collect();
        let pos = Tensor::<B, 2>::from_data(TensorData::new(positions, [t, 1]), device);
        let freq = Tensor::<B, 2>::from_data(TensorData::new(freqs, [1, head_dim / 2]), device);
        let angles = pos * freq;
        let sin = angles.clone().sin();
        let sin = Tensor::cat(vec![sin.clone().neg(), sin], 1).reshape([1, 1, t, head_dim]);
        let angles = Tensor::cat(vec![angles.clone(), angles], 1).reshape([1, 1, t, head_dim]);
        Self {
            cos: angles.cos().cast(dtype),
            sin: sin.cast(dtype),
        }
    }

    /// The tables for every position below `max_len`: what [`Rope::at`] selects a position's row
    /// from. Each row is [`Rope::new`]'s for that position, the same products in the same order.
    pub fn table(
        max_len: usize,
        head_dim: usize,
        theta: f64,
        dtype: DType,
        device: &B::Device,
    ) -> Self {
        Self::new(0, max_len, head_dim, theta, dtype, device)
    }

    /// The rows of a [`Rope::table`] at `positions` (`[n]`, on the device): a row select, so a
    /// traced step replays it with new positions.
    pub fn at(&self, positions: Tensor<B, 1, Int>) -> Self {
        let [_, _, max_len, d] = self.cos.dims();
        let [n] = positions.dims();
        let row = |t: &Tensor<B, 4>| {
            t.clone()
                .reshape([max_len, d])
                .select(0, positions.clone())
                .reshape([1, 1, n, d])
        };
        Self {
            cos: row(&self.cos),
            sin: row(&self.sin),
        }
    }

    /// The tables, `cos` and `sin` (`sin`'s first half negated).
    pub fn tables(&self) -> (Tensor<B, 4>, Tensor<B, 4>) {
        (self.cos.clone(), self.sin.clone())
    }

    /// `x` (`[n, heads, t, head_dim]`) rotated: `x cos + rotate_half(x) sin`. On the card one op
    /// where it can be (`burn_tt::rope`, bit for bit this composition), else the composition.
    pub fn apply(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        if let Some(y) = burn_tt::rope(&x, &self.cos, &self.sin) {
            return y;
        }
        let [n, h, t, d] = x.dims();
        let half = d / 2;
        let x1 = x.clone().slice([0..n, 0..h, 0..t, 0..half]);
        let x2 = x.clone().slice([0..n, 0..h, 0..t, half..d]);
        // `rotate_half(x) * sin`, `cat(-x2, x1) * [s, s]`, as `cat(x2, x1) * [-s, s]`: the same bits
        // (a product's sign is its operands' product, its magnitude theirs), without the
        // negation an application.
        let swapped = Tensor::cat(vec![x2, x1], 3);
        x * self.cos.clone() + swapped * self.sin.clone()
    }
}
