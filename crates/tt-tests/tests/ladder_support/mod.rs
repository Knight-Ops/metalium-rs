//! What the language-model gates share: a pretrained model on the device against the same model on
//! burn-flex, and a traced generation against the eager one (`step167`, `step187`, `step188`).
//! Each gate includes it with `mod ladder_support;` and uses what it needs.
#![allow(dead_code)]

use burn_flex::{Flex, FlexDevice};
use burn_tt::{TtBackend, TtDevice};
use tt_models::generate::{generate_fixed, FixedShapeDecoder, Generation, Shapes, TracedGenerator};
use tt_models::llama::{Llama, Precision};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

pub fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

/// The top-1 logit's margin over the second.
pub fn gap(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| b.total_cmp(a));
    s[0] - s[1]
}

pub fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The device's predictions against the host's on the same inputs: finite, and the top-1 equal at
/// every position. The largest difference and the closest margin are printed, not bounded: two
/// backends that round differently agree on the answer, which is the claim.
pub fn compare(label: &str, host: &Generation, device: &Generation) {
    let mut worst = 0f32;
    let mut scale = 0f32;
    for (h, d) in host.logits.iter().zip(&device.logits) {
        for (a, b) in h.iter().zip(d) {
            worst = worst.max((a - b).abs());
            scale = scale.max(a.abs());
        }
    }
    eprintln!(
        "{label}: max |flex - device| {worst:.4e} against a largest logit of {scale:.4}; \
         top-1 margin of the closest position {:.4}",
        host.logits.iter().map(|l| gap(l)).fold(f32::MAX, f32::min)
    );
    assert!(
        device.logits.iter().flatten().all(|v| v.is_finite()),
        "a non-finite logit"
    );
    for (i, (h, d)) in host.logits.iter().zip(&device.logits).enumerate() {
        assert_eq!(
            argmax(d),
            argmax(h),
            "{label}: the top-1 prediction at position {i} differs (flex margin {:.4})",
            gap(h)
        );
    }
}

/// A pretrained model on the device against the same one on burn-flex: `steps` greedy tokens after
/// `prompt` on the host, then on the device both free-running and teacher-forced with the host's
/// tokens, under `assert_native_model`. The host model is dropped before the device one loads, so a
/// large model is in memory once at a time. The claim is the teacher-forced top-1 at every
/// position; where free-running decoding parts is printed.
pub fn flex_then_device(
    config: Config,
    precision: Precision,
    prompt: &[i32],
    steps: usize,
    load_flex: impl FnOnce() -> Llama<Flex>,
    load_tt: impl FnOnce(&TtDevice) -> Llama<TtBackend>,
) {
    let (free_host, forced_host) = {
        let host_model = load_flex();
        let free = host_model.generate(prompt, steps, precision, &FlexDevice);
        let forced = host_model.forced(prompt, &free.tokens, precision, &FlexDevice);
        (free, forced)
    };

    with_device(config, |d| {
        let model = load_tt(&d);
        let ((free, forced), report) = burn_tt::with_report(|| {
            (
                model.generate(prompt, steps, precision, &d),
                model.forced(prompt, &free_host.tokens, precision, &d),
            )
        });
        eprintln!("{report}");
        for (label, g) in [("free-running", &free), ("teacher-forced", &forced)] {
            eprintln!(
                "{label}: prompt {:.2?}, then per token {:?}",
                g.times[0],
                g.times[1..]
                    .iter()
                    .map(|t| format!("{:.0?}", t))
                    .collect::<Vec<_>>()
            );
        }
        assert_native_model(&report);

        compare("teacher-forced", &forced_host, &forced);
        match free
            .tokens
            .iter()
            .zip(&free_host.tokens)
            .position(|(a, b)| a != b)
        {
            None => eprintln!(
                "free-running: {} tokens, identical: {:?}",
                free.tokens.len(),
                free.tokens
            ),
            Some(i) => eprintln!(
                "free-running: first divergence at token {i}: flex {:?}, device {:?}",
                free_host.tokens, free.tokens
            ),
        }
    });
}

/// A prompt of `len` ids below `vocab`, not all alike.
pub fn prompt(len: usize, vocab: usize) -> Vec<i32> {
    (0..len)
        .map(|i| ((7 + 31 * i * i + 13 * i) % vocab) as i32)
        .collect()
}

pub fn assert_same(got: &Generation, want: &Generation, what: &str) {
    assert_eq!(got.tokens, want.tokens, "{what}: tokens");
    assert_eq!(got.logits.len(), want.logits.len(), "{what}: predictions");
    for (i, (g, w)) in got.logits.iter().zip(&want.logits).enumerate() {
        assert_eq!(bits(g), bits(w), "{what}: logits {i}");
    }
}

/// Build a generator, checking that its captures and warm-up ran natively: the trace inputs it
/// makes resident are the only transfers outside an op (uploads, nothing read back or staged).
pub fn traced<'m, M: FixedShapeDecoder<TtBackend>>(
    model: &'m M,
    shapes: &Shapes,
    d: &TtDevice,
) -> TracedGenerator<'m, M, TtBackend> {
    let (g, report) = burn_tt::with_report(|| TracedGenerator::new(model, shapes, d));
    let report = burn_tt::Report(
        report
            .0
            .into_iter()
            .filter(|s| !(s.op == "(outside an op)" && s.downloaded == 0 && s.staged == 0))
            .collect(),
    );
    assert_native_model(&report);
    g.unwrap()
}

/// A traced generation, checking that the host built no Burn op for it: what runs is replays,
/// their inputs written and their logits read.
pub fn generate(
    g: &TracedGenerator<'_, impl FixedShapeDecoder<TtBackend>, TtBackend>,
    p: &[i32],
    steps: usize,
) -> Generation {
    let (out, report) = burn_tt::with_report(|| g.generate(p, steps));
    for s in &report.0 {
        assert_eq!(s.op, "(outside an op)", "a traced generation ran a Burn op");
        assert_eq!(s.staged, 0, "a traced generation staged");
    }
    out.unwrap()
}

/// A Llama `model`'s traced generation (`shapes`) of each prompt length in `lengths`, `steps`
/// tokens, bit for bit against the eager fixed-shape schedule and the growing cache.
pub fn traced_equals_eager(
    model: &Llama<TtBackend>,
    shapes: &Shapes,
    lengths: &[usize],
    steps: usize,
    what: &str,
    d: &TtDevice,
) {
    let vocab = model.config().vocab;
    let g = traced(model, shapes, d);
    for &len in lengths {
        let p = prompt(len, vocab);
        let got = generate(&g, &p, steps);
        let fixed = generate_fixed(model, &p, steps, shapes, d).unwrap();
        assert_same(
            &got,
            &fixed,
            &format!("{what}, prompt {len}, eager fixed schedule"),
        );
        let growing = model.generate(&p, steps, Precision::F32, d);
        assert_same(
            &got,
            &growing,
            &format!("{what}, prompt {len}, growing cache"),
        );
    }
}
