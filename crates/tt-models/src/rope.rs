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
        Self::with_frequencies(
            start,
            t,
            &frequencies(head_dim, head_dim / 2, theta),
            dtype,
            device,
        )
    }

    /// [`Rope::new`] with each rotated pair's frequency given: `freqs[i]` turns dimensions `i` and
    /// `i + head_dim / 2`, `head_dim` being twice `freqs.len()`. A zero frequency leaves its pair
    /// as it is (`cos 0 = 1`, `sin 0 = 0`), which is how a partial rotation laid over the whole
    /// head is written (Gemma 4's "proportional" embedding, [`frequencies`]).
    pub fn with_frequencies(
        start: usize,
        t: usize,
        freqs: &[f32],
        dtype: DType,
        device: &B::Device,
    ) -> Self {
        Self::scaled(start, t, freqs, 1.0, dtype, device)
    }

    /// [`Rope::with_frequencies`] with both tables multiplied by `scale` before they are rounded
    /// to `dtype`: YaRN's attention factor ([`Yarn::attention_factor`]), which scales `q k` by
    /// its square. A scale of one is [`Rope::with_frequencies`] bit for bit (no product).
    pub fn scaled(
        start: usize,
        t: usize,
        freqs: &[f32],
        scale: f32,
        dtype: DType,
        device: &B::Device,
    ) -> Self {
        let half = freqs.len();
        let positions: Vec<f32> = (start..start + t).map(|p| p as f32).collect();
        let pos = Tensor::<B, 2>::from_data(TensorData::new(positions, [t, 1]), device);
        let freq = Tensor::<B, 2>::from_data(TensorData::new(freqs.to_vec(), [1, half]), device);
        Self::from_angle_rows(pos * freq, scale, dtype)
    }

    /// Tables from each position's angles, `angles[p half + i]` turning pair `i` (dimensions `i`
    /// and `i + half`) of position `p`: for an embedding whose angle is not one position times one
    /// frequency, such as one over several axes (an image's rows and columns). The angles are host
    /// data; the `sin` and `cos` run on the device.
    pub fn from_angles(
        angles: Vec<f32>,
        t: usize,
        half: usize,
        dtype: DType,
        device: &B::Device,
    ) -> Self {
        assert_eq!(angles.len(), t * half, "{t} positions of {half} pairs");
        let angles = Tensor::<B, 2>::from_data(TensorData::new(angles, [t, half]), device);
        Self::from_angle_rows(angles, 1.0, dtype)
    }

    /// The tables from `angles` `[t, half]` (on the device): each row's half repeated as the
    /// second (the rotate-half layout), `sin`'s first half negated, both times `scale` (skipped
    /// at one) and then rounded to `dtype`.
    fn from_angle_rows(angles: Tensor<B, 2>, scale: f32, dtype: DType) -> Self {
        let [t, half] = angles.dims();
        let shape = [1, 1, t, 2 * half];
        let sin = angles.clone().sin();
        let sin = Tensor::cat(vec![sin.clone().neg(), sin], 1).reshape(shape);
        let cos = Tensor::cat(vec![angles.clone(), angles], 1)
            .reshape(shape)
            .cos();
        let (cos, sin) = if scale == 1.0 {
            (cos, sin)
        } else {
            (cos.mul_scalar(scale), sin.mul_scalar(scale))
        };
        Self {
            cos: cos.cast(dtype),
            sin: sin.cast(dtype),
        }
    }

    /// The tables of positions `from..to` of these.
    pub fn rows(&self, from: usize, to: usize) -> Self {
        let [a, b, _, d] = self.cos.dims();
        Self {
            cos: self.cos.clone().slice([0..a, 0..b, from..to, 0..d]),
            sin: self.sin.clone().slice([0..a, 0..b, from..to, 0..d]),
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
        self.composed(x)
    }

    /// [`Self::apply`] with its result in `dtype`: rounded once where it is made where the backend
    /// can (`burn_tt::rope_as`: a BF16 KV cache's keys with no cast op after), else the result cast.
    /// Bit for bit `self.apply(x).cast(dtype)`.
    pub fn apply_as(&self, x: Tensor<B, 4>, dtype: burn::tensor::DType) -> Tensor<B, 4> {
        if let Some(y) = burn_tt::rope_as(&x, &self.cos, &self.sin, dtype) {
            return y;
        }
        let y = self.apply(x);
        if y.dtype() == dtype {
            y
        } else {
            y.cast(dtype)
        }
    }

    fn composed(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
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

/// The rotary frequencies of a head of `head_dim` whose first `rotated` pairs turn:
/// `theta^(-2i / head_dim)` for `i < rotated`, zero after. `rotated = head_dim / 2` is the ordinary
/// embedding; fewer is a partial rotation spread over the whole head, the exponent's denominator
/// still the head's width (`transformers`' "proportional" RoPE).
pub fn frequencies(head_dim: usize, rotated: usize, theta: f64) -> Vec<f32> {
    assert!(
        head_dim % 2 == 0 && rotated <= head_dim / 2,
        "{rotated} rotated pairs of a head of {head_dim}"
    );
    (0..head_dim / 2)
        .map(|i| {
            if i < rotated {
                theta.powf(-(2.0 * i as f64) / head_dim as f64) as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// YaRN context extension ("NTK-by-parts" with an attention factor, `transformers`'
/// `rope_type: "yarn"`): a pair that turns many times over the original context keeps its
/// frequency, one that turns less than once is divided by `factor`, and those between ramp
/// linearly from one to the other. Both tables are scaled by [`Yarn::attention_factor`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    /// How much longer the context is than the one the model was trained at.
    pub factor: f64,
    /// The trained context (`original_max_position_embeddings`).
    pub original_max: usize,
    /// Pairs turning at least `beta_fast` times over `original_max` are left as they are, those
    /// turning at most `beta_slow` times are interpolated.
    pub beta_fast: f64,
    pub beta_slow: f64,
}

impl Yarn {
    /// `factor` over `original_max` positions with `transformers`' defaults (`beta_fast` 32,
    /// `beta_slow` 1).
    pub fn new(factor: f64, original_max: usize) -> Self {
        Self {
            factor,
            original_max,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }

    /// `0.1 ln(factor) + 1` (one at a factor of one or less): what both tables are scaled by.
    pub fn attention_factor(&self) -> f32 {
        if self.factor <= 1.0 {
            1.0
        } else {
            (0.1 * self.factor.ln() + 1.0) as f32
        }
    }

    /// The frequencies of a rotation `dim` wide (`dim / 2` pairs) at `theta`, as
    /// `transformers`' `_compute_yarn_parameters` makes them (`truncate` on).
    pub fn frequencies(&self, dim: usize, theta: f64) -> Vec<f32> {
        // The pair index at which a pair turns `turns` times over the original context.
        let pair_at = |turns: f64| {
            dim as f64 * (self.original_max as f64 / (turns * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * theta.ln())
        };
        let low = pair_at(self.beta_fast).floor().max(0.0);
        let high = pair_at(self.beta_slow).ceil().min(dim as f64 - 1.0);
        let high = if low == high { high + 0.001 } else { high };
        (0..dim / 2)
            .map(|i| {
                let extrapolated = theta.powf(-(2.0 * i as f64) / dim as f64);
                let interpolated = extrapolated / self.factor;
                // One below `low` (keep), zero above `high` (interpolate).
                let keep = 1.0 - ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                (interpolated * (1.0 - keep) + extrapolated * keep) as f32
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `transformers` 5.23's `_compute_yarn_parameters` for Qwen3.5's rotation (64 wide, theta
    /// 1e7) at factor 2 over 262,144 positions (computed in F32 by torch, so to 1e-6 relative):
    /// the first pairs kept, the last halved, the attention factor `0.1 ln 2 + 1`.
    #[test]
    fn yarn_matches_transformers() {
        const REFERENCE: [f32; 32] = [
            1.0,
            0.604_296_4,
            0.365_174_14,
            0.220_673_4,
            0.133_352_15,
            0.080_584_21,
            0.048_696_753,
            0.029_427_273,
            0.017_782_794,
            0.010_746_079,
            0.006_493_816_6,
            0.003_924_189_6,
            0.002_371_373_8,
            0.001_433_012_7,
            0.000_865_964_3,
            0.000_490_592_9,
            0.000_276_699_3,
            0.000_155_264_93,
            8.660_865e-5,
            4.797_585_3e-5,
            2.635_603_4e-5,
            1.433_416_9e-5,
            7.699_632e-6,
            4.652_860_3e-6,
            2.811_706_7e-6,
            1.699_104_1e-6,
            1.026_762_4e-6,
            6.204_689e-7,
            3.749_471_2e-7,
            2.265_791_9e-7,
            1.369_209_9e-7,
            8.274_085_4e-8,
        ];
        let yarn = Yarn::new(2.0, 262_144);
        for (i, (got, want)) in yarn.frequencies(64, 1e7).iter().zip(REFERENCE).enumerate() {
            assert!(
                (got - want).abs() <= want * 1e-6,
                "pair {i}: {got} vs {want}"
            );
        }
        assert!((yarn.attention_factor() - 1.069_314_7).abs() < 1e-7);
    }

    /// Scaled tables are the scale times the plain ones (F32, one product each), and a scale of
    /// one is the unscaled table bit for bit.
    #[test]
    fn scaled_tables_are_the_scale_times_the_plain_ones() {
        use burn_flex::{Flex, FlexDevice};
        let freqs = Yarn::new(4.0, 64).frequencies(16, 1e4);
        let plain = Rope::<Flex>::with_frequencies(3, 5, &freqs, DType::F32, &FlexDevice);
        let one = Rope::<Flex>::scaled(3, 5, &freqs, 1.0, DType::F32, &FlexDevice);
        let m = 1.25;
        let scaled = Rope::<Flex>::scaled(3, 5, &freqs, m, DType::F32, &FlexDevice);
        let values = |t: Tensor<Flex, 4>| t.into_data().to_vec::<f32>().unwrap();
        for (p, o, s) in [
            (plain.cos.clone(), one.cos, scaled.cos),
            (plain.sin.clone(), one.sin, scaled.sin),
        ] {
            let p = values(p);
            assert_eq!(p, values(o));
            let want: Vec<f32> = p.iter().map(|x| x * m).collect();
            assert_eq!(values(s), want);
        }
    }

    #[test]
    fn yarn_at_factor_one_is_the_plain_rotation() {
        let yarn = Yarn::new(1.0, 4096);
        assert_eq!(yarn.frequencies(64, 1e7), frequencies(64, 32, 1e7));
        assert_eq!(yarn.attention_factor(), 1.0);
    }
}
