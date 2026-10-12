//! How a matmul accumulates: the Matrix Unit's fast path, or exact F32.
//!
//! The default, [`MatmulMode::Fast`], is the tile matmul on the FPU (`MVMUL`). It is the fastest
//! path and not an F32 accumulation: each group of eight products is aligned once and rounded to a
//! quantum of `2^(largest operand-exponent sum - 10)`, and the operands are TF32. Its error against
//! the exact value is bounded by the measured rule of `step158_matmul_accumulation`
//! (`docs/learnings/silicon-operating-notes.md`, "Matmul accumulation: the Matrix Unit's group
//! adder"), at most half a quantum per rounded term plus an ulp of `Dst` per add.
//!
//! [`MatmulMode::Exact`] computes every product as an F32 SFPU multiply (to nearest, subnormals
//! flushed) and sums the products in increasing `k` order, sequentially, in F32, from `+0.0`. It uses
//! neither the FPU nor TF32. It is composed from resident SFPU ops (`ops/exact_matmul.rs`), so it is
//! far slower; it is for validation and for work that needs CPU-order accuracy, not for training.
//!
//! The mode is chosen in two ways. [`TtConfig::matmul`](crate::TtConfig::matmul) (or
//! `TT_EXACT_MATMUL=1`) fixes it for a device's attachment. [`with_matmul_mode`] overrides it for the
//! ops a closure builds **on the calling thread**. The override cannot reach `Fusion<TtBackend>`
//! ([`crate::Tt`]), whose ops run later on another thread: there only the config applies. A trace
//! freezes the mode in effect while it was captured.

use std::cell::Cell;

use crate::TtDevice;

/// How a matmul accumulates. See the [module](self) docs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MatmulMode {
    /// The tile matmul on the Matrix Unit: fastest, with the documented rounding.
    #[default]
    Fast,
    /// F32 products summed in order on the SFPU: slow, exact to F32 rounding.
    Exact,
}

thread_local! {
    static SCOPED: Cell<Option<MatmulMode>> = const { Cell::new(None) };
}

/// Run `f` with matmuls built on this thread in `mode`, whatever the device's own, then restore what
/// was in force (also if `f` panics). Scopes nest. Does not reach ops that run on another thread
/// (`Fusion<TtBackend>`); wrap `backward()` too to get exact gradients.
pub fn with_matmul_mode<R>(mode: MatmulMode, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<MatmulMode>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SCOPED.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(SCOPED.with(|s| s.replace(Some(mode))));
    f()
}

/// [`with_matmul_mode`] in [`MatmulMode::Exact`].
pub fn exact_matmul<R>(f: impl FnOnce() -> R) -> R {
    with_matmul_mode(MatmulMode::Exact, f)
}

/// The mode a matmul on `device` runs in now: this thread's override, else the device's.
pub(crate) fn effective(device: TtDevice) -> MatmulMode {
    SCOPED
        .with(Cell::get)
        .unwrap_or_else(|| crate::server::matmul_mode(device))
}

/// Was the mode forced by [`with_matmul_mode`] on this thread?
#[cfg(test)]
pub(crate) fn scoped() -> Option<MatmulMode> {
    SCOPED.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_nest_and_restore_even_after_a_panic() {
        assert_eq!(scoped(), None);
        exact_matmul(|| {
            assert_eq!(scoped(), Some(MatmulMode::Exact));
            with_matmul_mode(MatmulMode::Fast, || {
                assert_eq!(scoped(), Some(MatmulMode::Fast));
            });
            assert_eq!(scoped(), Some(MatmulMode::Exact));
        });
        assert_eq!(scoped(), None);
        let caught = std::panic::catch_unwind(|| exact_matmul(|| panic!("inside")));
        assert!(caught.is_err());
        assert_eq!(scoped(), None, "a panic must not leak the override");
    }

    #[test]
    fn the_default_is_fast() {
        assert_eq!(MatmulMode::default(), MatmulMode::Fast);
    }
}
