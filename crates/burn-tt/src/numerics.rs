//! The numeric contract a device's work runs under (ADR 0006,
//! `docs/adr/0006-two-numeric-contracts-reference-and-performance.md`).
//!
//! There are two, both deterministic: what a result is depends on the contract, never on the
//! grid, the card or the schedule.
//!
//! - [`Numerics::Reference`], the default: ADR 0005's contract, bit for bit against the
//!   specification models (HiFi4, FP32 `Dst`, `K` ascending, F32 activations). Conformance and
//!   the training gates run under it.
//! - [`Numerics::Performance`], the inference contract. Its tokens are not the reference's; a
//!   model is gated on teacher-forced top-1/top-5 agreement and logit PCC, with thresholds
//!   derived from a reference run (`crates/tt-tests/tests/step269_performance_contract.rs`).
//!
//! **What the performance contract is today (roadmap W1, its first box):** BFP8 weights,
//! chosen where a model is loaded (`tt_models::quant::Plan::for_numerics`); activations stay
//! F32, products HiFi4. No op of this crate computes differently under it yet, so its
//! specification model is exact: the F32 model holding the BFP8-rounded weights. Later W1 boxes
//! (BF16 activations, derived fidelity, packer L1 accumulation) make ops read
//! [`numerics`] and extend that model.
//!
//! The contract is chosen two ways, as [`MatmulMode`](crate::MatmulMode) is:
//! [`TtConfig::numerics`](crate::TtConfig::numerics) (or `TT_NUMERICS`) fixes it for an
//! attachment, and [`with_numerics`] overrides it for the work a closure builds **on the calling
//! thread** (it cannot reach `Fusion<TtBackend>`'s ops, which run later on another thread). A
//! trace records the contract in force while it was captured
//! ([`TracedInference::numerics`](crate::TracedInference::numerics)). A graph is captured under one
//! contract: a [`with_numerics`] to another while a capture is open on the thread panics by name
//! (ADR 0006 allows a mix only at a named boundary, and none is defined yet).

use std::cell::Cell;

use crate::error::TtError;
use crate::TtDevice;

/// A numeric contract. See the [module](self) docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Numerics {
    /// Bit for bit against the specification models: the default.
    #[default]
    Reference,
    /// The inference contract: block-float weights, gated on token agreement and PCC.
    Performance,
}

impl Numerics {
    /// The contract a `TT_NUMERICS` value names: `reference` or `performance`. Anything else is
    /// an error naming the value.
    pub fn parse(value: &str) -> Result<Self, TtError> {
        match value {
            "reference" => Ok(Numerics::Reference),
            "performance" => Ok(Numerics::Performance),
            other => Err(TtError::Device(format!(
                "TT_NUMERICS={other}: expected reference or performance"
            ))),
        }
    }

    /// `TT_NUMERICS`, if set.
    pub fn from_env() -> Option<Result<Self, TtError>> {
        std::env::var("TT_NUMERICS").ok().map(|v| Self::parse(&v))
    }

    /// Its name, as `TT_NUMERICS` spells it.
    pub fn name(self) -> &'static str {
        match self {
            Numerics::Reference => "reference",
            Numerics::Performance => "performance",
        }
    }
}

thread_local! {
    static SCOPED: Cell<Option<Numerics>> = const { Cell::new(None) };
    /// The contract of the capture open on this thread, if any ([`Capture`]).
    static CAPTURING: Cell<Option<Numerics>> = const { Cell::new(None) };
}

/// Run `f` with the work built on this thread under `numerics`, whatever the device's own, then
/// restore what was in force (also if `f` panics). Scopes nest.
///
/// # Panics
///
/// Inside a trace's capture on this thread that is under another contract: a graph is captured
/// under one.
pub fn with_numerics<R>(numerics: Numerics, f: impl FnOnce() -> R) -> R {
    if let Some(capturing) = CAPTURING.with(Cell::get) {
        assert!(
            capturing == numerics,
            "with_numerics({}) inside a capture under the {} contract: a trace is captured under \
             one numeric contract",
            numerics.name(),
            capturing.name()
        );
    }
    struct Restore(Option<Numerics>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SCOPED.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(SCOPED.with(|s| s.replace(Some(numerics))));
    f()
}

/// The contract work on `device` is built under now: this thread's [`with_numerics`], else the
/// attachment's ([`TtConfig::numerics`](crate::TtConfig::numerics)). Attaches the device if
/// nothing has.
pub fn numerics(device: TtDevice) -> Result<Numerics, TtError> {
    match SCOPED.with(Cell::get) {
        Some(n) => Ok(n),
        None => crate::server::numerics(device),
    }
}

/// Marks a capture open on this thread under `numerics` until dropped (also on an unwind).
pub(crate) struct Capture(Option<Numerics>);

impl Capture {
    pub(crate) fn open(numerics: Numerics) -> Self {
        Capture(CAPTURING.with(|c| c.replace(Some(numerics))))
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        CAPTURING.with(|c| c.set(self.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_parses_or_is_refused_by_name() {
        assert_eq!(Numerics::parse("reference").unwrap(), Numerics::Reference);
        assert_eq!(
            Numerics::parse("performance").unwrap(),
            Numerics::Performance
        );
        let e = Numerics::parse("Performance").unwrap_err().to_string();
        assert!(e.contains("TT_NUMERICS=Performance"), "{e}");
        for n in [Numerics::Reference, Numerics::Performance] {
            assert_eq!(Numerics::parse(n.name()).unwrap(), n);
        }
    }

    #[test]
    fn scopes_nest_and_restore_even_after_a_panic() {
        let scoped = || SCOPED.with(Cell::get);
        assert_eq!(scoped(), None);
        with_numerics(Numerics::Performance, || {
            assert_eq!(scoped(), Some(Numerics::Performance));
            with_numerics(Numerics::Reference, || {
                assert_eq!(scoped(), Some(Numerics::Reference));
            });
            assert_eq!(scoped(), Some(Numerics::Performance));
        });
        let _ = std::panic::catch_unwind(|| {
            with_numerics(Numerics::Performance, || panic!("inside"));
        });
        assert_eq!(scoped(), None);
    }

    #[test]
    fn a_capture_refuses_another_contract_but_not_its_own() {
        {
            let _capture = Capture::open(Numerics::Reference);
            with_numerics(Numerics::Reference, || {});
            let other = std::panic::catch_unwind(|| with_numerics(Numerics::Performance, || {}));
            let message = other.unwrap_err();
            let message = message.downcast_ref::<String>().unwrap();
            assert!(message.contains("inside a capture"), "{message}");
        }
        // Closed: any contract again.
        with_numerics(Numerics::Performance, || {});
    }
}
