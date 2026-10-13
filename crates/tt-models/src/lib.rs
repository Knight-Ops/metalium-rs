//! The model ladder: pretrained models on burn-tt, checked against burn-flex.
//!
//! Dev-only (`xtask/src/ship.rs`): it depends on burn-flex as the host reference.

pub mod image_input;
pub mod llama;
pub mod load;
pub mod quant;
pub mod resnet;
pub mod vit;
