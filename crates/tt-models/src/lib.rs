//! The model ladder: pretrained models on burn-tt, checked against burn-flex.
//!
//! Dev-only (`xtask/src/ship.rs`): it depends on burn-flex as the host reference.

pub mod delta_rule;
pub mod gemma4;
pub mod generate;
pub mod host_embed;
pub mod image_input;
pub mod kv_cache;
pub mod llama;
pub mod load;
pub mod moss_image;
pub mod progress;
pub mod prompt_sets;
pub mod pytorch_adapter;
pub mod quant;
pub mod qwen35;
pub mod resnet;
pub mod rope;
pub mod vit;
pub mod weight_cache;
