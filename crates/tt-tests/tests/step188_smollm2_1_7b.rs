//! The first rung past 1B parameters: SmolLM2-1.7B-Instruct (24 layers of width 2048, 32 heads,
//! F32) on burn-tt against burn-flex, and its traced generation against the eager one.
//!
//! The host model is checked against HuggingFace `transformers` in `tt-models`'s
//! `llama_1_7b_host` (identical top-10 on a raw and a chat-templated prompt, and the same greedy
//! answer), so agreement here is agreement on a model that is right.
//!
//! What is claimed, on silicon (all 24 layers, every tile):
//!
//! * **The model runs on the device and agrees with Flex**: the chat prompt, 16 greedy tokens,
//!   `assert_native_model` over the whole generation, and the teacher-forced top-1 equal at every
//!   position (`ladder_support::flex_then_device`).
//! * **Traced generation is the eager one**: `TracedGenerator` with chunks of 32 and 64 and decode
//!   steps attending to 64 and then 128 positions, bit for bit against the eager schedule and the
//!   growing cache, at prompt lengths on both sides of a chunk and of a step length.
//!
//! On the simulator the Flex comparison is opt-in (`-- --ignored`), at one layer: the 2048 x 49,152
//! output projection makes it the slowest test of the suite (about 6 minutes). The traced test is
//! silicon only: the simulator runs on one tile, whose program cache cannot hold a 2048-wide
//! capture's programs ("a list's programs do not fit an empty program cache", the limit trace
//! segmentation lifts); `step187` covers the traced path on the simulator.

mod ladder_support;

use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_models::llama::{Llama, Precision, SMOLLM2_1_7B_INSTRUCT};
use tt_tests::burn_device::Config;

/// "What is the capital of France?" in the model's chat template (`llama_1_7b_host`'s `CHAT`).
const CHAT: [i32; 37] = [
    1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407,
    19712, 8182, 2, 198, 1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47, 2, 198, 1, 520, 9531,
    198,
];

fn layers() -> usize {
    if cfg!(feature = "silicon") {
        SMOLLM2_1_7B_INSTRUCT.layers
    } else {
        1
    }
}

/// Every tile on silicon (a 1.7B model on one tile is not what anyone runs, and its traces would
/// not fit one tile's program cache); the gate tile on the simulator.
fn config() -> Config {
    if cfg!(feature = "silicon") {
        Config {
            tiles: Some(burn_tt::TileChoice::All),
            ..Config::default()
        }
    } else {
        Config::default()
    }
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-tests/weights"
)]
#[cfg_attr(
    all(feature = "weights", not(feature = "silicon")),
    ignore = "slow on the simulator (minutes): run explicitly with -- --ignored"
)]
fn device_agrees_with_flex() {
    let (layers, steps) = (layers(), if cfg!(feature = "silicon") { 16 } else { 2 });
    ladder_support::flex_then_device(
        config(),
        Precision::F32,
        &CHAT,
        steps,
        || {
            Llama::<Flex>::smollm2_1_7b_instruct(layers, Precision::F32, &FlexDevice)
                .unwrap()
                .0
        },
        |d| {
            Llama::<TtBackend>::smollm2_1_7b_instruct(layers, Precision::F32, d)
                .unwrap()
                .0
        },
    );
}

/// Silicon only: the simulator runs on one tile, whose program cache cannot hold the programs of a
/// 2048-wide capture ("a list's programs do not fit an empty program cache"; trace segmentation
/// lifts it). `step187` covers the traced path on the simulator.
#[cfg(feature = "silicon")]
#[test]
fn traced_generation_equals_the_eager_one() {
    use tt_models::generate::Shapes;
    use tt_tests::burn_device::with_device;

    // Chunks of 32 and 64 over 128 positions; decode steps attending to 64, then all 128.
    let shapes = Shapes::new(128, &[32, 64]);
    assert_eq!(shapes.steps, [64, 128]);
    // One chunk padded (5), one chunk of 64 (40), and two chunks (70); 62 decodes across the
    // 64-position step's end.
    let (lengths, steps): (&[usize], usize) = (&[5, 40, 62, 70], 4);
    with_device(config(), |d| {
        let (model, _) =
            Llama::<TtBackend>::smollm2_1_7b_instruct(layers(), Precision::F32, &d).unwrap();
        ladder_support::traced_equals_eager(&model, &shapes, lengths, steps, "SmolLM2-1.7B", &d);
    });
}
