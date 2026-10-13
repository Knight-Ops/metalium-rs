//! Mixed precision: a different number format for each matrix of a model.
//!
//! Six formats meet in one network: F32, BF16 and F16 (logical dtypes, the matrix is computed in
//! that dtype) and BFP8, BFP4 and BFP2 (block floating point, physical storage of a logical F32
//! tensor that the device decodes as it computes, `burn_tt::storage`). A [`Plan`] names the format
//! of each matrix by a substring of its parameter path (`layers.3.mlp.down_proj.weight`), and
//! [`apply`] rewrites a model's matrices accordingly.
//!
//! Two quantisers apply a plan. [`Native`] is for burn-tt: BFP becomes resident BFP storage and the
//! half types become logical casts. [`Emulated`] is the reference for any backend, burn-flex in
//! particular: BFP is rounded on the host by [`bfp_round_trip`] (an independent model of the packer
//! and the unpacker) and the result kept as ordinary F32, which is what the device computes with
//! after decoding. Both run the same model code, so a comparison between them measures the device's
//! arithmetic on identical quantised values.

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor, TensorData};
use burn_tt::storage::{StorageBackend, StorageFormat, TensorStorageExt};

/// A number format a matrix or an activation can be held in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    F32,
    Bf16,
    F16,
    Bfp8,
    Bfp4,
    Bfp2,
}

impl Format {
    /// Bits of a block-floating-point datum, `None` for the ordinary floats.
    fn bfp_width(self) -> Option<u32> {
        match self {
            Format::Bfp8 => Some(8),
            Format::Bfp4 => Some(4),
            Format::Bfp2 => Some(2),
            _ => None,
        }
    }

    fn half(self) -> Option<DType> {
        match self {
            Format::Bf16 => Some(DType::BF16),
            Format::F16 => Some(DType::F16),
            _ => None,
        }
    }

    fn storage(self) -> StorageFormat {
        match self {
            Format::Bfp8 => StorageFormat::Bfp8,
            Format::Bfp4 => StorageFormat::Bfp4,
            Format::Bfp2 => StorageFormat::Bfp2,
            _ => StorageFormat::F32,
        }
    }
}

/// The format of each named tensor: the first rule whose pattern is a substring of the name wins,
/// and a name no rule matches stays F32.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    rules: Vec<(String, Format)>,
}

impl Plan {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn rule(mut self, pattern: &str, format: Format) -> Self {
        self.rules.push((pattern.to_string(), format));
        self
    }

    pub fn format(&self, name: &str) -> Format {
        self.rules
            .iter()
            .find(|(pattern, _)| name.contains(pattern.as_str()))
            .map_or(Format::F32, |(_, f)| *f)
    }
}

/// What `values` become when held in a block-floating-point `format` and read back: rows of
/// `cols` values in blocks of 16 consecutive columns that share the largest biased exponent.
///
/// The rule is the packer's, written independently of the kernels: truncate to BF16, scale by the
/// block's quantum `2^(exponent - 127 - 6)`, round to nearest, clamp to 127, drop the low bits the
/// format has no room for (7, 3 or 1 magnitude bits), and decode with `tt_isa`'s model of the
/// unpacker. It is the oracle of `step92_bfp_formats`, as a function. Finite values only.
pub fn bfp_round_trip(values: &[f32], cols: usize, format: Format) -> Vec<f32> {
    let width = format.bfp_width().expect("a block floating point format");
    assert!(cols > 0 && values.len() % cols == 0);
    let mut out = vec![0f32; values.len()];
    for (row, dst) in values.chunks(cols).zip(out.chunks_mut(cols)) {
        for (block, dst) in row.chunks(16).zip(dst.chunks_mut(16)) {
            assert!(
                block.iter().all(|v| v.is_finite()),
                "a non-finite value cannot be held in {format:?}"
            );
            let exponent = block
                .iter()
                .map(|v| (v.to_bits() >> 23) as u8)
                .max()
                .expect("a block has values");
            let quantum = 2.0f64.powi(exponent as i32 - 127 - 6);
            for (value, dst) in block.iter().zip(dst) {
                let bits = value.to_bits() & 0xffff_0000;
                let scaled = if bits & 0x7f80_0000 == 0 {
                    0.0
                } else {
                    f32::from_bits(bits).abs() as f64 / quantum
                };
                let magnitude = (scaled.round().min(127.0) as u8) >> (8 - width);
                let datum = if magnitude == 0 {
                    0
                } else {
                    magnitude | (((bits >> 31) as u8) << (width - 1))
                };
                let decoded = match format {
                    Format::Bfp8 => tt_isa::tile::bfp8_to_bf16(datum, exponent),
                    Format::Bfp4 => tt_isa::tile::bfp4_to_bf16(datum, exponent),
                    _ => tt_isa::tile::bfp2_to_bf16(datum, exponent),
                };
                *dst = if decoded & 0x7f80 == 0 {
                    0.0
                } else {
                    f32::from_bits((decoded as u32) << 16)
                };
            }
        }
    }
    out
}

/// Puts tensors into a [`Format`], keeping them logical F32 for the block formats.
pub trait Quantizer<B: Backend> {
    /// A matrix held in `format`: a half format changes its logical dtype (so it is computed in it),
    /// a block format keeps it F32 and changes how it is stored.
    fn matrix(&self, format: Format, t: Tensor<B, 2>) -> Tensor<B, 2>;

    /// An activation passed through `format` and left in its own dtype: it has lost what the format
    /// cannot hold.
    fn activation<const D: usize>(&self, format: Format, t: Tensor<B, D>) -> Tensor<B, D>;
}

/// burn-tt's own storage: resident block floating point and logical half types.
pub struct Native;

impl<B: StorageBackend> Quantizer<B> for Native {
    fn matrix(&self, format: Format, t: Tensor<B, 2>) -> Tensor<B, 2> {
        if let Some(dtype) = format.half() {
            t.cast(dtype)
        } else if format.bfp_width().is_some() {
            // Block formats take a logical F32 tensor (`StorageBackend`).
            t.cast(DType::F32).with_storage(format.storage())
        } else {
            t
        }
    }

    fn activation<const D: usize>(&self, format: Format, t: Tensor<B, D>) -> Tensor<B, D> {
        let own = t.dtype();
        if let Some(dtype) = format.half() {
            t.cast(dtype).cast(own)
        } else if format.bfp_width().is_some() {
            t.cast(DType::F32).with_storage(format.storage())
        } else {
            t
        }
    }
}

/// The host reference: block formats rounded by [`bfp_round_trip`], held as plain F32.
pub struct Emulated;

impl Emulated {
    fn round<B: Backend, const D: usize>(format: Format, t: Tensor<B, D>) -> Tensor<B, D> {
        let device = t.device();
        let dims = t.dims();
        let values = t
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("F32 values");
        let rounded = bfp_round_trip(&values, dims[D - 1], format);
        Tensor::from_data(TensorData::new(rounded, dims), &device)
    }
}

impl<B: Backend> Quantizer<B> for Emulated {
    fn matrix(&self, format: Format, t: Tensor<B, 2>) -> Tensor<B, 2> {
        if let Some(dtype) = format.half() {
            t.cast(dtype)
        } else if format.bfp_width().is_some() {
            Self::round(format, t.cast(DType::F32))
        } else {
            t
        }
    }

    fn activation<const D: usize>(&self, format: Format, t: Tensor<B, D>) -> Tensor<B, D> {
        let own = t.dtype();
        if let Some(dtype) = format.half() {
            t.cast(dtype).cast(own)
        } else if format.bfp_width().is_some() {
            Self::round(format, t.cast(DType::F32))
        } else {
            t
        }
    }
}

struct Apply<'a, Q> {
    path: Vec<String>,
    plan: &'a Plan,
    quantizer: &'a Q,
}

impl<B: Backend, Q: Quantizer<B>> ModuleMapper<B> for Apply<'_, Q> {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        // Matrices only: vectors (norm scales, biases) and the position tables stay as they are.
        if D != 2 {
            return param;
        }
        let format = self.plan.format(&self.path.join("."));
        if format == Format::F32 {
            return param;
        }
        param.map(|t| {
            let dims = t.dims();
            let matrix = t.reshape([dims[0], dims[1]]);
            self.quantizer.matrix(format, matrix).reshape(dims)
        })
    }
}

/// `model` with every matrix put into the format `plan` names for it.
pub fn apply<B: Backend, M: Module<B>>(model: M, plan: &Plan, quantizer: &impl Quantizer<B>) -> M {
    model.map(&mut Apply {
        path: Vec::new(),
        plan,
        quantizer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Vec<f32> {
        // Two rows of 32: a spread of magnitudes inside each block, both signs, zeros.
        (0..64)
            .map(|i| {
                let v = ((i * 37 % 29) as f32 - 14.0)
                    * (1.0 + 0.0137 * (i % 7) as f32)
                    * 2.0f32.powi(i / 16 - 2);
                if i % 11 == 0 {
                    0.0
                } else {
                    v
                }
            })
            .collect()
    }

    #[test]
    fn a_round_trip_is_idempotent_and_within_the_compression_bound() {
        for format in [Format::Bfp8, Format::Bfp4, Format::Bfp2] {
            let v = data();
            let once = bfp_round_trip(&v, 32, format);
            assert_eq!(bfp_round_trip(&once, 32, format), once, "{format:?}");
            let width = format.bfp_width().unwrap() as i32;
            for (block, (src, dst)) in v.chunks(16).zip(once.chunks(16)).enumerate() {
                let exp = src.iter().map(|x| (x.to_bits() >> 23) as u8).max().unwrap();
                let quantum = 2.0f64.powi(exp as i32 - 127 - (width - 2));
                for (a, b) in src.iter().zip(dst) {
                    // step92's finite-input bound, the same one the packer is held to.
                    let bound = (*a as f64).abs() / 128.0 + quantum + 2.0f64.powi(-126);
                    assert!(
                        (*a as f64 - *b as f64).abs() <= bound,
                        "{format:?} block {block}: {a} became {b}, bound {bound}"
                    );
                }
            }
        }
    }

    #[test]
    fn fewer_bits_lose_more() {
        let v = data();
        let err = |f| -> f64 {
            let r = bfp_round_trip(&v, 32, f);
            v.iter()
                .zip(&r)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum()
        };
        let (e8, e4, e2) = (err(Format::Bfp8), err(Format::Bfp4), err(Format::Bfp2));
        assert!(e8 < e4 && e4 < e2, "{e8} {e4} {e2}");
        assert!(
            e8 > 0.0,
            "control: BFP8 is lossy on this data, or the test proves nothing"
        );
    }

    #[test]
    fn a_plan_matches_by_substring_and_first_rule_wins() {
        let plan = Plan::new()
            .rule("layers.0.", Format::Bfp2)
            .rule("mlp.", Format::Bfp4)
            .rule("self_attn", Format::Bf16);
        assert_eq!(plan.format("layers.0.mlp.up_proj.weight"), Format::Bfp2);
        assert_eq!(plan.format("layers.1.mlp.up_proj.weight"), Format::Bfp4);
        assert_eq!(
            plan.format("layers.1.self_attn.q_proj.weight"),
            Format::Bf16
        );
        assert_eq!(plan.format("embed_tokens.weight"), Format::F32);
    }
}
