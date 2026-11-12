//! Probe: where Nemotron-3.5-Bolt-E4B's load puts the BF16 per-layer table on the card.
//! `step261_gemma4` fails with no GDDR room for a BF16 `[262144, 10752]` tensor; this loads the
//! model stage by stage and prints each stage's op report and the host's resident set.

#![cfg(feature = "silicon")]

use burn_tt::{TtBackend, TtDevice};
use tt_models::gemma4::Gemma4;
use tt_models::llama::Precision;
use tt_tests::burn_device::{with_device, Config};

fn rss_gb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    status
        .lines()
        .find(|l| l.starts_with("VmRSS"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap()
        / 1048576.0
}

#[test]
fn where_the_load_puts_the_tables() {
    with_device(Config::default(), |d: TtDevice| {
        let (model, report) = burn_tt::with_report(|| {
            Gemma4::<TtBackend>::nemotron_3_5_bolt_e4b(Precision::F32, &d)
                .unwrap()
                .0
        });
        eprintln!("=== loaded, rss {:.2} GB\n{report}", rss_gb());
        let (model, report) = burn_tt::with_report(|| model.with_f32_tables());
        eprintln!("=== tables widened, rss {:.2} GB\n{report}", rss_gb());
        let (_model, report) = burn_tt::with_report(|| model.with_bf16_weights());
        eprintln!("=== weights stored, rss {:.2} GB\n{report}", rss_gb());
    });
}
