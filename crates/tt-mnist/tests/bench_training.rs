//! `tt-mnist`'s training benchmarks on a card, for `cargo xtask bench` (ignored: a card, and
//! `cargo xtask bench` runs the ignored tests): the binary run as a user runs it, three times each, its card's and burn-flex's
//! milliseconds a step printed as `MEASURE` and `BENCH` lines that `cargo xtask bench` collects.
//!
//! The scoreboard's tt-mnist rows were carried forward unmeasured for three weeks while two
//! commits made untraced training up to ten times slower (`docs/learnings/performance-log.md`,
//! "The tt-mnist regression found"); these are its rows' command, run with the other benchmarks.
//!
//! Each run's output is validated: MNIST's test accuracy on the card at least 90% and its loss
//! curve agreeing with burn-flex's (the binary checks ten checkpoints), the transformer's final
//! loss on the card within 0.01 of burn-flex's.

use std::process::Command;

/// The card `cargo xtask silicon` names (`TT_SILICON_DEVICE`), else 0.
fn card() -> String {
    std::env::var("TT_SILICON_DEVICE").unwrap_or_else(|_| "0".into())
}

fn tt_mnist(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_tt-mnist"))
        .args(["--card", &card()])
        .args(args)
        .output()
        .expect("running tt-mnist");
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "tt-mnist {args:?} failed:\n{text}");
    text
}

/// `(median, p10, p90)`.
fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).expect("a sample is NaN"));
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    (at(0.5), at(0.1), at(0.9))
}

/// A `MEASURE` line and its `BENCH` record, in `tt_tests::bench`'s format.
fn report(key: &str, samples: Vec<f64>) {
    let n = samples.len();
    let (median, p10, p90) = stats(samples);
    println!("MEASURE [host] {key:<44} {median:>10.3} ms (p10 {p10:.3}, p90 {p90:.3}, n {n})");
    println!(
        "BENCH {{\"kind\":\"result\",\"key\":\"{key}\",\"unit\":\"ms\",\"timed\":\"host\",\
         \"median\":{median},\"p10\":{p10},\"p90\":{p90},\"n\":{n}}}"
    );
}

/// Milliseconds a step from a `937 training steps   1.31s  (1.4 ms/step)` line: the total over
/// the steps, which keeps the digits the binary's rounding drops.
fn ms_per_step(line: &str) -> f64 {
    let words: Vec<&str> = line.split_whitespace().collect();
    let steps: f64 = words[0].parse().expect("a step count");
    let total = words[3];
    let ms = if let Some(ms) = total.strip_suffix("ms") {
        ms.parse::<f64>().expect("milliseconds")
    } else {
        total
            .strip_suffix('s')
            .expect("seconds")
            .parse::<f64>()
            .expect("seconds")
            * 1e3
    };
    ms / steps
}

const RUNS: usize = 3;

fn mnist(key: &str, extra: &[&str]) {
    let (mut card, mut host) = (Vec::new(), Vec::new());
    for _ in 0..RUNS {
        let mut args = vec!["--host"];
        args.extend_from_slice(extra);
        let out = tt_mnist(&args);
        let steps: Vec<&str> = out
            .lines()
            .filter(|l| l.contains("training steps"))
            .collect();
        assert_eq!(steps.len(), 2, "the card's and burn-flex's runs:\n{out}");
        card.push(ms_per_step(steps[0]));
        host.push(ms_per_step(steps[1]));
        let accuracy: f64 = out
            .lines()
            .find(|l| l.contains("test accuracy"))
            .and_then(|l| l.split_whitespace().last())
            .and_then(|a| a.trim_end_matches('%').parse().ok())
            .expect("the card's test accuracy");
        assert!(accuracy >= 90.0, "{key}: test accuracy {accuracy}%\n{out}");
        assert!(
            out.contains("loss curves agree"),
            "{key}: the loss curves disagree\n{out}"
        );
    }
    report(&format!("tt-mnist {key} training, card"), card);
    report(&format!("tt-mnist {key} training, burn-flex"), host);
}

#[test]
#[ignore = "a card benchmark: cargo xtask bench --filter bench_training"]
fn mnist_784_128_10() {
    mnist("MNIST 784-128-10", &["--hidden", "128"]);
}

#[test]
#[ignore = "a card benchmark: cargo xtask bench --filter bench_training"]
fn mnist_default() {
    mnist("MNIST 784-512-256-10", &[]);
}

#[test]
#[ignore = "a card benchmark: cargo xtask bench --filter bench_training"]
fn transformer() {
    let (mut card, mut host) = (Vec::new(), Vec::new());
    for _ in 0..RUNS {
        let out = tt_mnist(&["--model", "transformer"]);
        let ms = |who: &str| -> f64 {
            out.lines()
                .find(|l| l.contains(who) && l.contains("ms/step"))
                .and_then(|l| l.split_whitespace().nth(2))
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{who}'s time a step:\n{out}"))
        };
        card.push(ms("burn-tt (card)"));
        host.push(ms("burn-flex (host)"));
        // `loss  card a -> b, host c -> d`
        let finals: Vec<f64> = out
            .lines()
            .find(|l| l.trim_start().starts_with("loss") && l.contains("host"))
            .map(|l| {
                l.split(',')
                    .filter_map(|part| part.split("->").nth(1)?.trim().parse().ok())
                    .collect()
            })
            .unwrap_or_default();
        assert_eq!(finals.len(), 2, "the card's and burn-flex's losses:\n{out}");
        assert!(
            (finals[0] - finals[1]).abs() < 0.01,
            "the card's final loss {} against burn-flex's {}",
            finals[0],
            finals[1]
        );
    }
    report("tt-mnist transformer training, card", card);
    report("tt-mnist transformer training, burn-flex", host);
}
