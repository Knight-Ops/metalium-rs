//! MOSS-Image (`OpenMOSS-Team/MOSS-Image-Base`): text to image by flow matching. A Qwen3.5 text
//! encoder ([`crate::qwen35::MOSS_TEXT_ENCODER`]) turns the chat-templated prompt into hidden
//! states; a diffusion transformer ([`dit::Dit`]) predicts a velocity for latent tokens at each
//! step of an Euler schedule ([`Schedule`]), with classifier-free guidance against the empty
//! prompt; the FLUX.2 VAE's decoder ([`vae::Vae`]) turns the final latents into pixels.
//!
//! The reference is `transformer/pipeline_moss_image.py` at the pinned revision (`diffusers`
//! 0.38.0 underneath), whose defaults these are: 50 steps, shift 5, guidance 3, and the negative
//! branch run with block 22 left out (`uncond_skip_layers`). The predictions and the Euler update
//! are F32.
//!
//! The three parts are loaded in turn ([`text_encoder`], [`transformer`], [`vae::Vae::pretrained`])
//! so the 9-billion-parameter encoder and the 5-billion-parameter transformer are never in memory
//! together: the prompt is encoded, the encoder dropped, then the image denoised.

pub mod dit;
pub mod vae;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData};

use crate::llama::Precision;
use crate::load::{load_safetensors_shards, weights_files, Layout, LoadError, Loaded};
use crate::qwen35::{Qwen35, MOSS_TEXT_ENCODER};
use dit::{Dit, DitConfig, MOSS_IMAGE_BASE};

/// Where `cargo xtask fetch-weights --model moss-image-base` puts the checkpoint.
pub const WEIGHTS: &str = "moss-image-base";

const ENCODER_FILES: [&str; 4] = [
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
];

const TRANSFORMER_FILES: [&str; 3] = [
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
];

/// The pipeline's default guidance: the scale, and the blocks the negative branch leaves out.
pub const GUIDANCE: f32 = 3.0;
pub const UNCOND_SKIP: [usize; 1] = [22];

/// The text encoder, `text_encoder/` (its vision tower left in the file).
pub fn text_encoder<B: Backend>(
    precision: Precision,
    device: &B::Device,
) -> Result<(Qwen35<B>, Loaded), LoadError> {
    Qwen35::text_encoder(
        &MOSS_TEXT_ENCODER,
        WEIGHTS,
        &ENCODER_FILES,
        precision,
        device,
    )
}

/// The prompt's conditioning `[1, S, 4096]`: the encoder's last hidden states (after its final
/// norm) over the chat-templated prompt's token ids.
pub fn encode<B: Backend>(encoder: &Qwen35<B>, ids: &[i32], dtype: DType) -> Tensor<B, 3> {
    let device = encoder.device();
    let tokens =
        Tensor::<B, 2, Int>::from_data(TensorData::new(ids.to_vec(), [1, ids.len()]), &device);
    let mut state = encoder.state(ids.len(), dtype, &device);
    encoder.hidden(encoder.embed(tokens), &mut state, dtype)
}

/// The diffusion transformer, `transformer/`, in `precision` (its derived `rope.bands` left in the
/// file).
pub fn transformer<B: Backend>(
    precision: Precision,
    device: &B::Device,
) -> Result<(Dit<B>, Loaded), LoadError> {
    transformer_of(&MOSS_IMAGE_BASE, precision, device)
}

fn transformer_of<B: Backend>(
    cfg: &DitConfig,
    precision: Precision,
    device: &B::Device,
) -> Result<(Dit<B>, Loaded), LoadError> {
    let mut model = Dit::new(cfg, device);
    let ignore = |name: &str| name == "rope.bands";
    let loaded = load_safetensors_shards(
        &mut model,
        &weights_files(WEIGHTS, &TRANSFORMER_FILES),
        &Layout {
            remaps: &[
                (r"^time_embedding\.0\.", "time_in."),
                (r"^time_embedding\.2\.", "time_out."),
            ],
            pytorch: true,
            ignore: &ignore,
        },
    )?;
    let model = match precision {
        Precision::Bf16 => model,
        Precision::F32 => model.map(&mut ToF32),
    };
    Ok((model, loaded))
}

/// Widens every float parameter to F32.
struct ToF32;

impl<B: Backend> burn::module::ModuleMapper<B> for ToF32 {
    fn map_float<const D: usize>(
        &mut self,
        param: burn::module::Param<Tensor<B, D>>,
    ) -> burn::module::Param<Tensor<B, D>> {
        param.map(|t| t.cast(DType::F32))
    }
}

/// The Euler schedule: `sigmas[i]` for each step and a terminal zero, and the transformer's
/// timestep `sigma * 1000` for each step.
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
}

/// `torch.linspace(start, end, steps)` in F32, as ATen fills it: `start + step i` for the first
/// half and `end - step (steps - 1 - i)` for the rest, each one fused multiply-add (measured
/// against torch 2.14.1's CPU kernel: a separate product and sum differ in the last bit at 239 of
/// 1,000 points).
fn linspace(start: f64, end: f64, steps: usize) -> Vec<f32> {
    let (start, end) = (start as f32, end as f32);
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps)
        .map(|i| {
            if i < half {
                step.mul_add(i as f32, start)
            } else {
                (-step).mul_add((steps - i - 1) as f32, end)
            }
        })
        .collect()
}

impl Schedule {
    /// `MOSSImageScheduler.set_timesteps(steps)` with shift `shift` over 1000 training steps
    /// (training shift 1): `steps` shifted sigmas from 1 down, each snapped to the nearest of the
    /// training grid's, repeats removed.
    pub fn new(steps: usize, shift: f32) -> Self {
        assert!(steps >= 1 && shift > 0.0);
        const TOTAL: usize = 1000;
        let grid = |count: usize, shift: f32| -> Vec<f32> {
            let lower = shift as f64 / (TOTAL as f64 + shift as f64 - 1.0);
            linspace(1.0, lower, count)
                .into_iter()
                .map(|v| shift * v / (1.0 + (shift - 1.0) * v))
                .collect()
        };
        let wanted = grid(steps, shift);
        let mut training = grid(TOTAL, 1.0);
        training.reverse();
        let mut sigmas: Vec<f32> = Vec::with_capacity(steps + 1);
        for s in wanted {
            let upper = training.partition_point(|&t| t < s).min(TOTAL - 1);
            let lower = upper.saturating_sub(1);
            let nearest = if (s - training[lower]).abs() <= (training[upper] - s).abs() {
                lower
            } else {
                upper
            };
            let snapped = training[nearest];
            if sigmas.last() != Some(&snapped) {
                sigmas.push(snapped);
            }
        }
        let timesteps = sigmas.iter().map(|s| s * TOTAL as f32).collect();
        sigmas.push(0.0);
        Self { sigmas, timesteps }
    }

    pub fn steps(&self) -> usize {
        self.timesteps.len()
    }
}

/// Denoise `noise` (`[1, 128, H, W]`, F32) over `schedule`: at each step the conditioned
/// prediction and the negative one (with `skip` left out) combined as `uncond + guidance (cond -
/// uncond)` in F32, then the Euler update `x + (sigma_next - sigma) v`. The final latents.
pub fn denoise<B: Backend>(
    dit: &Dit<B>,
    noise: Tensor<B, 4>,
    cond: Tensor<B, 3>,
    uncond: Tensor<B, 3>,
    schedule: &Schedule,
    guidance: f32,
    skip: &[usize],
) -> Tensor<B, 4> {
    let mut x = noise.cast(DType::F32);
    for (i, &t) in schedule.timesteps.iter().enumerate() {
        let c = dit
            .forward(x.clone(), t, cond.clone(), &[])
            .cast(DType::F32);
        let u = dit
            .forward(x.clone(), t, uncond.clone(), skip)
            .cast(DType::F32);
        let v = u.clone() + (c - u).mul_scalar(guidance);
        let dt = schedule.sigmas[i + 1] - schedule.sigmas[i];
        x = x + v.mul_scalar(dt);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_fills_both_ends_exactly() {
        let v = linspace(1.0, 0.001, 1000);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[999], 0.001f32);
        assert_eq!(v.len(), 1000);
        assert!(v.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn a_schedule_runs_from_one_down_to_zero() {
        let s = Schedule::new(50, 5.0);
        assert_eq!(s.sigmas[0], 1.0);
        assert_eq!(*s.sigmas.last().unwrap(), 0.0);
        assert_eq!(s.sigmas.len(), s.steps() + 1);
        assert!(s.sigmas.windows(2).all(|w| w[0] > w[1]));
        assert_eq!(s.timesteps[0], 1000.0);
    }
}
