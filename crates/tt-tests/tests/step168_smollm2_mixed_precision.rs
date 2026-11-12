//! SmolLM2 with a different number format per matrix, on burn-tt against burn-flex: F32, BF16,
//! F16 and the three block floating point formats BFP8, BFP4 and BFP2, alone and all in one model.
//!
//! The device applies a plan natively (`quant::Native`: resident BFP storage, logical half casts).
//! The reference applies the same plan on the host (`quant::Emulated`: block formats rounded by
//! `bfp_round_trip`, the independent model of the packer and unpacker, then kept as F32). Both run
//! the same model code over the same strictly loaded weights and the same teacher-forced tokens, so
//! the difference between them is the device's arithmetic on identical quantised values, and
//! nothing else.
//!
//! What is claimed, per plan:
//!
//! * **Native**: `assert_native_model` over the whole run: no staged compute, no host arithmetic,
//!   no downloaded operands, so every format is computed by the card.
//! * **Close to the reference**: the largest logit difference is printed against the device's own
//!   baseline (an F32 plan, run first), and the top-1 prediction at each position agrees wherever
//!   the reference's margin exceeds twice the measured difference (which is what a bounded
//!   difference allows; it is a consequence, not a tolerance).
//! * **It really quantised** (block formats): the device's logits are closer to the quantised
//!   reference than to the unquantised one. A device that skipped the compression, or served a
//!   host copy of the original weights, would be closer to the unquantised model.
//!
//! How much each format costs the model is printed too; a plan's quality is not asserted, only
//! that the device computes the plan the way the reference does.
//!
//! Silicon only: the simulator cannot afford the full model, and BF16 results need packer mode
//! `0x105`, which it refuses (`ttsim-divergence.md`, "BF16 late narrowing").

#![cfg(feature = "silicon")]

use std::sync::Arc;

use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_models::llama::{ActFn, Generation, Llama, Precision};
use tt_models::quant::{apply, Emulated, Format, Native, Plan, Quantizer};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

/// "The capital of France is", then HuggingFace's greedy continuation of it.
const PROMPT: [i32; 5] = [504, 3575, 282, 4649, 314];
const CONTINUATION: [i32; 3] = [260, 3575, 282];
/// Layers run: the whole model is 30, and each plan runs it twice on the card.
const LAYERS: usize = 6;

fn flex_run(plan: &Plan, acts: &Plan) -> Generation {
    let (model, _) = Llama::<Flex>::smollm2_135m(LAYERS, Precision::F32, &FlexDevice).unwrap();
    let model = apply(model, plan, &Emulated);
    let acts = acts.clone();
    let hook: ActFn<Flex> = Arc::new(move |name, t| Emulated.activation(acts.format(name), t));
    model.forced_with(
        &PROMPT,
        &CONTINUATION,
        Precision::F32,
        Some(hook),
        &FlexDevice,
    )
}

fn gap(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| b.total_cmp(a));
    s[0] - s[1]
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

fn max_diff(a: &Generation, b: &Generation) -> f32 {
    a.logits
        .iter()
        .flatten()
        .zip(b.logits.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Run `plan` on the device and on the host and check the claims above.
fn check(name: &str, plan: Plan, acts: Plan, quantises: bool) {
    let reference = flex_run(&plan, &acts);
    let unquantised = flex_run(&Plan::new(), &Plan::new());
    with_device(Config::default(), |d| {
        let (model, _) = Llama::<TtBackend>::smollm2_135m(LAYERS, Precision::F32, &d).unwrap();
        let model = apply(model, &plan, &Native);
        let acts = acts.clone();
        let hook: ActFn<TtBackend> = Arc::new(move |n, t| Native.activation(acts.format(n), t));
        let (device, report) = burn_tt::with_report(|| {
            model.forced_with(&PROMPT, &CONTINUATION, Precision::F32, Some(hook), &d)
        });
        eprintln!("{report}");
        assert_native_model(&report);

        assert!(
            device.logits.iter().flatten().all(|v| v.is_finite()),
            "{name}: a non-finite logit"
        );
        let to_reference = max_diff(&device, &reference);
        let to_unquantised = max_diff(&device, &unquantised);
        let cost = max_diff(&reference, &unquantised);
        eprintln!(
            "{name}: device vs reference {to_reference:.4}, device vs unquantised {to_unquantised:.4}; \
             the plan itself moves the host's logits by {cost:.4}"
        );
        for (i, (r, dv)) in reference.logits.iter().zip(&device.logits).enumerate() {
            if gap(r) > 2.0 * to_reference {
                assert_eq!(
                    argmax(dv),
                    argmax(r),
                    "{name}: position {i}, reference margin {:.4} exceeds twice the difference",
                    gap(r)
                );
            }
        }
        if quantises {
            assert!(
                to_reference < to_unquantised,
                "{name}: the device is no closer to the quantised model ({to_reference:.4}) than \
                 to the unquantised one ({to_unquantised:.4}); did it compress?"
            );
        }
    });
}

fn all(format: Format) -> Plan {
    Plan::new().rule("weight", format)
}

#[test]
fn f32_baseline() {
    check("f32", Plan::new(), Plan::new(), false);
}

#[test]
fn bf16_weights() {
    check("bf16", all(Format::Bf16), Plan::new(), false);
}

#[test]
fn f16_weights() {
    check("f16", all(Format::F16), Plan::new(), false);
}

#[test]
fn bfp8_weights() {
    check("bfp8", all(Format::Bfp8), Plan::new(), true);
}

#[test]
fn bfp4_weights() {
    check("bfp4", all(Format::Bfp4), Plan::new(), true);
}

#[test]
fn bfp2_weights() {
    check("bfp2", all(Format::Bfp2), Plan::new(), true);
}

/// All six formats in one model, weights and activations.
#[test]
fn all_six_formats_in_one_model() {
    let weights = Plan::new()
        .rule("embed_tokens", Format::Bfp8)
        .rule("q_proj", Format::Bfp8)
        .rule("k_proj", Format::Bfp4)
        .rule("v_proj", Format::Bf16)
        .rule("o_proj", Format::F16)
        .rule("gate_proj", Format::Bfp2)
        .rule("up_proj", Format::Bfp8)
        .rule("down_proj", Format::Bfp4);
    let acts = Plan::new()
        .rule("attn_ctx", Format::Bf16)
        .rule("mlp_hidden", Format::Bfp8);
    check("mixed", weights, acts, true);
}
