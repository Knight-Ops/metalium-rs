//! The general-model gate: a small transformer, built from Burn's own modules,
//! trains on `burn-tt`, and the fallback report says which of its ops still
//! run on the host.
//!
//! MNIST's MLP exercises matmul, element-wise ops and a loss. A transformer
//! exercises what a model past MNIST is made of (`tt_mnist::transformer`,
//! which `tt-mnist --model transformer` times against burn-flex on silicon).
//!
//! What is claimed:
//!
//! * **The report is a ratchet** (`a_transformer_trains_and_the_report_only_shrinks`):
//!   every op that made a result on the host or moved bytes during a training
//!   step is listed in `tests/golden/transformer_off_device.txt`. A new name
//!   there is a regression and fails; a name that has left is reported, so the
//!   list is shortened in the commit that moved the op to the card. The report
//!   is printed either way: it is the worklist.
//! * **The first loss agrees with Flex's.** Not a derived bound: the per-op
//!   bounds live in the op gates (`step11_burn`, `step32_burn_softmax`, ...);
//!   this checks that their composition through Burn's modules is not wrong,
//!   at a relative 1e-3 that TF32 matmuls through two layers comfortably meet
//!   and a wrong op does not.
//! * **Training works**: the loss falls on both backends over a few steps.
//!
//! Weights are drawn by Flex and copied into the `burn-tt` model in visiting
//! order (`transformer::init`), so both runs start from the same bits.

use burn::backend::Autodiff;
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use tt_mnist::transformer::{init, train};
use tt_tests::burn_device::{with_device, Config};

const STEPS: usize = 3;

/// The names in the baseline: one op per line, `#` comments.
fn baseline() -> std::collections::BTreeSet<String> {
    include_str!("golden/transformer_off_device.txt")
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

#[test]
fn a_transformer_trains_and_the_report_only_shrinks() {
    let weights = init(59);
    let host = train::<Autodiff<Flex>>(&weights, STEPS, &FlexDevice).losses;
    with_device(Config::default(), |d| {
        let (tt, report) =
            burn_tt::with_report(|| train::<Autodiff<TtBackend>>(&weights, STEPS, &d).losses);
        eprintln!("losses: flex {host:?}\n        tt   {tt:?}\n{report}");

        let rel = ((tt[0] - host[0]) / host[0]).abs();
        assert!(
            rel < 1e-3,
            "first loss {} against Flex's {}: {rel:e} relative",
            tt[0],
            host[0]
        );
        for (what, l) in [("flex", &host), ("tt", &tt)] {
            assert!(
                l[STEPS - 1] < l[0],
                "{what}: the loss did not fall over {STEPS} steps: {l:?}"
            );
        }

        let off: std::collections::BTreeSet<String> =
            report.off_device().map(|s| s.op.to_string()).collect();
        let known = baseline();
        let new: Vec<_> = off.difference(&known).collect();
        let gone: Vec<_> = known.difference(&off).collect();
        if !gone.is_empty() {
            eprintln!(
                "now on the device -- remove from tests/golden/transformer_off_device.txt: {gone:?}"
            );
        }
        assert!(
            new.is_empty(),
            "ops newly off the device (a regression, or add them to the baseline \
             with a reason): {new:?}\n{report}"
        );
    });
}
