//! The performance contract's model gate (ADR 0006; roadmap W1, "BFP8 weights by default under
//! the contract"): SmolLM2-1.7B-Instruct, traced on the card as inference runs it
//! (`Llama::for_inference`), teacher-forced over a fixed prompt set against HuggingFace.
//!
//! The reference is `tt_models::prompt_sets::smollm2_1_7b`: eight chat prompts, HuggingFace's
//! greedy continuation of each (at most 32 tokens), and its five largest logits at every position
//! that predicts a continuation token. Each run feeds the reference's tokens (teacher forcing,
//! `TracedGenerator::prefill` then `decode`), so every position is scored on the same context.
//!
//! - **Top-1 and top-5** are against HuggingFace's stored top-1 (top-5: HuggingFace's top-1 among
//!   the run's five largest).
//! - **PCC** is the full logits' correlation with burn-flex's F32 model at the same position.
//!   burn-flex is HuggingFace's stand-in there, since the full logits are not stored:
//!   `tt-models`' `llama_1_7b_host::flex_reproduces_the_prompt_set` holds it to HuggingFace over
//!   the same set.
//!
//! The runs, in one attachment:
//!
//! 1. **The reference contract** (BF16-stored weights, bit for bit the F32 model): its top-1
//!    equals burn-flex's at every position (the ladder's claim, `step188`), and its agreement is
//!    the **baseline** the thresholds are derived from.
//! 2. **The performance contract** (BFP8 weights): top-1 and top-5 at least the baseline's less
//!    2 points, and mean PCC at least the baseline's less 0.005 (the margin the owner chose,
//!    2026-11-10). That its logits are those of the F32 model holding the rounded weights, bit for
//!    bit, is `step240`'s claim (the contract's specification model); this gate is its quality.
//! 3. **Its named option** (BFP4 `gate|up`): its agreement recorded, not held to a threshold. On
//!    this model it misses tt-metal's acceptance floors (top-1 90%, PCC 0.95), and the owner kept
//!    it as a recorded option (2026-11-10), to be judged again on the 8B-class model (W5).
//!
//! Negative control: the correlation falls strictly from BF16 to BFP8 to BFP4 weights, so each
//! run really computed with its contract's formats (a `for_inference` that ignored its contract
//! would tie two of them).
//!
//! **Measured** (2026-11-10, card 0, release, every tile; all eight prompts, 232 positions,
//! `1794287910`; burn-flex F32 against HuggingFace: 100% / 100%, largest logit difference 1.1e-4,
//! `llama_1_7b_host::flex_reproduces_the_prompt_set`):
//!
//! | run | top-1 | top-5 | PCC mean (least) |
//! |---|--:|--:|--:|
//! | reference contract (BF16 weights): the baseline | 100% | 100% | 0.999998 (0.999994) |
//! | performance contract (BFP8 weights) | 98.28% | 100% | 0.999678 (0.999352) |
//! | BFP4 `gate\|up` | 86.64% | 96.55% | 0.932528 (0.773370) |
//!
//! The thresholds that follow: BFP8 top-1 and top-5 at least 98%, mean PCC at least 0.994998.
//! BFP8's top-1 clears its floor by one position (228 of 232); the contract is deterministic, so
//! the margin moves only with the code (a later W1 box that flips one more near-tie is reviewed).
//! The smoke run's two prompts (40 positions, `1794287596`): 100% / 100% / 0.999645 for BFP8.
//!
//! Model workload: silicon only (`docs/learnings/test-tiers.md`).

use burn_flex::{Flex, FlexDevice};
use burn_tt::Numerics;
use tt_models::generate::{Shapes, TracedGenerator};
use tt_models::llama::{Llama, Precision, SMOLLM2_1_7B_INSTRUCT};
use tt_models::prompt_sets::{agreement, argmax, smollm2_1_7b::CASES, Agreement, Case};
use tt_models::quant::WeightOptions;
use tt_tests::burn_device::{with_device, Config};

/// Each case's logits at every position that predicts a continuation token.
type Logits = Vec<Vec<Vec<f32>>>;

/// burn-flex's F32 model, teacher-forced over `cases` on the host.
fn flex_logits(cases: &[Case]) -> Logits {
    let (model, _) = Llama::<Flex>::smollm2_1_7b_instruct(
        SMOLLM2_1_7B_INSTRUCT.layers,
        Precision::F32,
        &FlexDevice,
    )
    .unwrap();
    cases
        .iter()
        .map(|c| {
            let g = model.forced(c.prompt, c.continuation, Precision::F32, &FlexDevice);
            g.logits[..c.continuation.len()].to_vec()
        })
        .collect()
}

/// The prompt set's traces: one chunk size, the cache the longest case needs rounded up to it.
fn shapes(cases: &[Case]) -> Shapes {
    const CHUNK: usize = 64;
    let longest = cases
        .iter()
        .map(|c| c.prompt.len() + c.continuation.len())
        .max()
        .unwrap();
    Shapes::new(longest.div_ceil(CHUNK) * CHUNK, &[CHUNK])
}

/// `model` (the F32 checkpoint on the card) as inference runs it under `numerics`, traced, and
/// teacher-forced over `cases`.
fn device_logits(
    model: &Llama<burn_tt::Tt>,
    numerics: Numerics,
    options: WeightOptions,
    cases: &[Case],
    d: &burn_tt::TtDevice,
) -> Logits {
    let model = model.clone().for_inference(numerics, options);
    let g = TracedGenerator::<_, burn_tt::Tt>::new(&model, &shapes(cases), d).unwrap();
    assert_eq!(g.numerics(), numerics, "the traces' contract");
    cases
        .iter()
        .map(|c| {
            let mut rows = vec![g.prefill(c.prompt).unwrap()];
            let fed = &c.continuation[..c.continuation.len() - 1];
            for (i, &token) in fed.iter().enumerate() {
                rows.push(g.decode(token, c.prompt.len() + i).unwrap());
            }
            rows
        })
        .collect()
}

fn report(what: &str, a: &Agreement) {
    eprintln!("{what}: {a}");
    println!(
        "BENCH {{\"kind\":\"agreement\",\"model\":\"smollm2-1.7b-instruct\",\"run\":\"{what}\",\
         \"positions\":{},\"top1\":{},\"top5\":{},\"pcc_mean\":{},\"pcc_min\":{}}}",
        a.positions, a.top1, a.top5, a.pcc_mean, a.pcc_min
    );
}

fn run(cases: &[Case]) {
    let flex = flex_logits(cases);
    report(
        "burn-flex F32 against HuggingFace",
        &agreement(cases, &flex, &flex),
    );
    let config = Config {
        tiles: Some(burn_tt::TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        let (model, _) = Llama::<burn_tt::Tt>::smollm2_1_7b_instruct(
            SMOLLM2_1_7B_INSTRUCT.layers,
            Precision::F32,
            &d,
        )
        .unwrap();

        let reference = device_logits(
            &model,
            Numerics::Reference,
            WeightOptions::default(),
            cases,
            &d,
        );
        for (c, (got, want)) in cases.iter().zip(reference.iter().zip(&flex)) {
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert_eq!(
                    argmax(g),
                    argmax(w),
                    "the reference contract's top-1 differs from burn-flex's: {:?}, position {i}",
                    c.chat
                );
            }
        }
        let baseline = agreement(cases, &reference, &flex);
        report("reference contract (BF16 weights)", &baseline);

        let bfp8 = device_logits(
            &model,
            Numerics::Performance,
            WeightOptions::default(),
            cases,
            &d,
        );
        let bfp8 = agreement(cases, &bfp8, &flex);
        report("performance contract (BFP8 weights)", &bfp8);

        let bfp4 = WeightOptions { mlp_bfp4: true };
        let bfp4 = device_logits(&model, Numerics::Performance, bfp4, cases, &d);
        let bfp4 = agreement(cases, &bfp4, &flex);
        report("performance contract, BFP4 gate|up", &bfp4);

        // The thresholds: the baseline less the owner's margin (2 points of top-1 and top-5, 0.005
        // of mean PCC). The named option is recorded only.
        assert!(
            bfp8.top1 >= baseline.top1 - 0.02
                && bfp8.top5 >= baseline.top5 - 0.02
                && bfp8.pcc_mean >= baseline.pcc_mean - 0.005,
            "BFP8 weights: {bfp8}, against the reference contract's {baseline} less 2 points and 0.005"
        );
        // The negative control: each format's rounding shows in the correlation.
        assert!(
            baseline.pcc_mean > bfp8.pcc_mean && bfp8.pcc_mean > bfp4.pcc_mean,
            "the correlation does not fall with the weights' bits: {baseline} / {bfp8} / {bfp4}"
        );
    });
}

/// The whole prompt set.
#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask fetch-weights --model smollm2-1.7b-instruct, then cargo xtask silicon --release --filter step269_performance_contract"
)]
fn smollm2_1_7b_agrees_with_huggingface_under_each_contract() {
    run(&CASES);
}

/// Its first two prompts: the routine (smoke) run.
#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask fetch-weights --model smollm2-1.7b-instruct, then cargo xtask silicon --release --filter step269_performance_contract"
)]
fn smollm2_1_7b_agrees_with_huggingface_smoke() {
    run(&CASES[..2]);
}
