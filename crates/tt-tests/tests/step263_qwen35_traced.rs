//! Qwen3.5-4B's traced generation (`tt_models::generate::TracedGenerator` over
//! `tt_models::qwen35::HybridCache`), against the eager model on the card and against burn-flex.
//!
//! The model is `step260_qwen35`'s, its matrices stored BF16 (the shipped 1.7B's configuration,
//! F32 arithmetic on the checkpoint's own values). Its state is fixed-shape: the eight
//! full-attention layers' keys and values over the cache's positions, and each of the 24 Gated
//! DeltaNets' carried convolution inputs and recurrent state. A recurrent state, unlike a key, is
//! never overwritten, so a padded chunk's padding rows must leave it as the prompt's last token did
//! (decay 1, write strength 0, the convolution inputs that end at that token).
//!
//! What is claimed, on silicon:
//!
//! * **Traced generation is the eager one**: the traced greedy logits and tokens, bit for bit, are
//!   those of the eager model with its growing state (`EagerDecoder::generate`), for prompts that
//!   pad a chunk by 3 and by 7 rows, fill one exactly, and decode across a step length's end, all
//!   through one generator (a prompt from position 0 forgets the last one's state).
//! * **It agrees with burn-flex** as `step260` claims it: the top-1 prediction equal at the prompt
//!   and at every decode step (the host model is checked against `transformers` in
//!   `qwen35_host`). The traced generation built no Burn op and staged nothing.
//! * **The negative control**: the same schedule without the padding mask
//!   (`qwen35::UnmaskedPadding`, run eagerly on the card, what its traces would compute) gives the
//!   same prompt logits and different decode logits, so the bit-for-bit claim can fail.
//!
//! A model workload: silicon only, and out of `--smoke` (as `step260`-`step262`, by owner decision).

mod ladder_support;

use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_models::generate::{generate_fixed, EagerDecoder, Generation, Shapes};
use tt_models::llama::Precision;
use tt_models::qwen35::{Qwen35, UnmaskedPadding, QWEN3_5_4B};
use tt_tests::burn_device::{with_device, Config};

/// "What is the capital of France?" in the model's chat template (`step260`'s): 17 tokens, fed as
/// a chunk of 16 and a chunk of 8 holding one token and 7 padding rows.
const CHAT: [i32; 17] = [
    248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068,
    198,
];

/// Greedy tokens after each prompt.
const STEPS: usize = 6;

/// The prompts: the chat prompt; 5 tokens (a chunk of 8, 3 padding rows); 16 (one chunk, none);
/// 61 (three chunks of 16 and one of 16 holding 13; its decode steps cross from the step attending
/// to 64 positions to the one attending to 128).
fn prompts() -> Vec<Vec<i32>> {
    let vocab = QWEN3_5_4B.vocab;
    vec![
        CHAT.to_vec(),
        ladder_support::prompt(5, vocab),
        ladder_support::prompt(16, vocab),
        ladder_support::prompt(61, vocab),
    ]
}

/// The process's resident memory, in GB (`VmRSS`): the gate runs inside a host-memory cap.
fn rss_gb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
        .map_or(0.0, |kb| kb / 1e6)
}

fn decode_ms(g: &Generation) -> Vec<f64> {
    g.times[1..].iter().map(|t| t.as_secs_f64() * 1e3).collect()
}

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask fetch-weights --model qwen3.5-4b, then cargo xtask silicon --release --filter step263_qwen35_traced"
)]
fn traced_generation_equals_the_eager_one_and_agrees_with_flex() {
    let prompts = prompts();
    // burn-flex, eager, growing state, F32 (the checkpoint's BF16 values widened, which is what
    // the card's stored weights hold).
    let host: Vec<Generation> = tt_ttsim::outside_fork(|| {
        let (model, _) =
            Qwen35::<Flex>::qwen3_5_4b(QWEN3_5_4B.layers, Precision::F32, &FlexDevice).unwrap();
        prompts
            .iter()
            .map(|p| {
                let g = model.generate(p, STEPS, Precision::F32, &FlexDevice);
                eprintln!("burn-flex, prompt {}: tokens {:?}", p.len(), g.tokens);
                g
            })
            .collect()
    });
    eprintln!("after burn-flex: {:.1} GB resident", rss_gb());

    // Chunks of 8 and 16 attending to the whole cache; decode steps attending to 64, then 128.
    let shapes = Shapes::new(128, &[8, 16]).with_prefixes(&[]);
    assert_eq!(shapes.steps, [64, 128]);
    let config = Config {
        tiles: Some(burn_tt::TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        let (model, _) =
            Qwen35::<TtBackend>::qwen3_5_4b(QWEN3_5_4B.layers, Precision::F32, &d).unwrap();
        eprintln!("loaded: {:.1} GB resident", rss_gb());
        let model = model.with_bf16_weights();
        eprintln!("weights stored BF16: {:.1} GB resident", rss_gb());
        let t0 = std::time::Instant::now();
        let g = ladder_support::traced(&model, &shapes, &d);
        eprintln!(
            "built in {:.1} s; {:.1} GB resident",
            t0.elapsed().as_secs_f64(),
            rss_gb()
        );

        let mut traced = Vec::new();
        for (p, host) in prompts.iter().zip(&host) {
            let got = ladder_support::generate(&g, p, STEPS);
            eprintln!(
                "traced, prompt {}: tokens {:?}, prompt {:.1} ms, decode steps {:.1?} ms",
                p.len(),
                got.tokens,
                got.times[0].as_secs_f64() * 1e3,
                decode_ms(&got)
            );
            ladder_support::compare(&format!("traced, prompt {}", p.len()), host, &got);
            assert_eq!(got.tokens, host.tokens, "prompt {}: greedy tokens", p.len());
            traced.push(got);
        }

        eager_and_control(&model, &shapes, &prompts, &traced, &d);
    });
}

/// `traced` (the traced generations of `prompts`) bit for bit against the eager model's growing
/// state on the card, then the negative control: the first prompt (which pads a chunk) without the
/// padding mask leaves its own logits and moves the decode steps'.
fn eager_and_control(
    model: &Qwen35<TtBackend>,
    shapes: &Shapes,
    prompts: &[Vec<i32>],
    traced: &[Generation],
    d: &burn_tt::TtDevice,
) {
    {
        for (p, got) in prompts.iter().zip(traced) {
            let t0 = std::time::Instant::now();
            let eager = model.generate(p, STEPS, Precision::F32, d);
            eprintln!(
                "eager, prompt {}: {:.1} s, decode steps {:.0?} ms",
                p.len(),
                t0.elapsed().as_secs_f64(),
                decode_ms(&eager)
            );
            ladder_support::assert_same(
                got,
                &eager,
                &format!("prompt {}, traced against the eager growing state", p.len()),
            );
        }

        // The negative control: the chat prompt's padded chunk without the mask.
        let unmasked =
            generate_fixed(&UnmaskedPadding(model), &prompts[0], STEPS, shapes, d).unwrap();
        let want = &traced[0];
        assert_eq!(
            ladder_support::bits(&unmasked.logits[0]),
            ladder_support::bits(&want.logits[0]),
            "without the mask, the prompt's own logits are unchanged"
        );
        let moved = unmasked.logits[1..]
            .iter()
            .flatten()
            .zip(want.logits[1..].iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "without the padding mask: decode logits move by up to {moved:.4e}, tokens {:?}",
            unmasked.tokens
        );
        assert!(
            moved > 1e-2,
            "the padding rows' mask changes nothing ({moved}): the control cannot fail"
        );
    }
}

/// The first four layers (three DeltaNets and a full-attention layer), traced against eager on the
/// card with the negative control, without burn-flex: what the whole gate claims of the traces, in
/// a minute rather than ten.
#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "model workload, silicon tier: cargo xtask fetch-weights --model qwen3.5-4b, then cargo xtask silicon --release --filter step263_qwen35_traced"
)]
fn the_first_layers_trace_as_they_run_eagerly() {
    let prompts = vec![CHAT.to_vec(), ladder_support::prompt(5, QWEN3_5_4B.vocab)];
    let shapes = Shapes::new(64, &[8, 16]).with_prefixes(&[]);
    let config = Config {
        tiles: Some(burn_tt::TileChoice::All),
        ..Config::default()
    };
    with_device(config, |d| {
        let (model, _) = Qwen35::<TtBackend>::qwen3_5_4b(4, Precision::F32, &d).unwrap();
        let model = model.with_bf16_weights();
        let g = ladder_support::traced(&model, &shapes, &d);
        let traced: Vec<Generation> = prompts
            .iter()
            .map(|p| ladder_support::generate(&g, p, STEPS))
            .collect();
        eager_and_control(&model, &shapes, &prompts, &traced, &d);
    });
}
