//! SmolLM2 with a different number format per matrix, on the host (`Emulated`).
//!
//! Block floating point is rounded by `bfp_round_trip` and kept as F32; the half types are logical
//! casts. This is the reference a device run with the same plan is compared against
//! (`step168_smollm2_mixed_precision`), and here it only has to behave: a plan that keeps every
//! matrix in F32 changes nothing, and fewer bits change the answer more.
//! Needs `cargo xtask fetch-weights`.

use burn_flex::{Flex, FlexDevice};
use tt_models::llama::{Llama, Precision, FULL_LAYERS};
use tt_models::quant::{apply, Emulated, Format, Plan};

const PROMPT: [i32; 5] = [504, 3575, 282, 4649, 314];

fn logits(plan: &Plan) -> Vec<f32> {
    let (model, _) = Llama::<Flex>::pretrained(FULL_LAYERS, Precision::F32, &FlexDevice).unwrap();
    let model = apply(model, plan, &Emulated);
    model
        .generate(&PROMPT, 1, Precision::F32, &FlexDevice)
        .logits
        .remove(0)
}

fn top(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx.truncate(k);
    idx
}

fn report(name: &str, base: &[f32], got: &[f32]) {
    let worst = base
        .iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("{name:28} top-5 {:?}  max |dlogit| {worst:.3}", top(got, 5));
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-models/weights"
)]
fn how_much_each_format_costs() {
    let base = logits(&Plan::new());
    report("f32", &base, &base);
    for (name, plan) in [
        ("all bf16", Plan::new().rule("weight", Format::Bf16)),
        ("all f16", Plan::new().rule("weight", Format::F16)),
        ("all bfp8", Plan::new().rule("weight", Format::Bfp8)),
        ("all bfp4", Plan::new().rule("weight", Format::Bfp4)),
        ("all bfp2", Plan::new().rule("weight", Format::Bfp2)),
        (
            "mixed (all six)",
            Plan::new()
                .rule("embed_tokens", Format::Bfp8)
                .rule("q_proj", Format::Bfp8)
                .rule("k_proj", Format::Bfp8)
                .rule("v_proj", Format::Bf16)
                .rule("o_proj", Format::F16)
                .rule("gate_proj", Format::Bfp4)
                .rule("up_proj", Format::Bfp4)
                .rule("down_proj", Format::Bfp8),
        ),
    ] {
        let got = logits(&plan);
        report(name, &base, &got);
    }
}
