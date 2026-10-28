//! Allocation (pass 5 of `docs/learnings/dataflow-compiler.md` §5): a run's L1, declared by its
//! body as `l1::Requirements` ([`Microkernel::needs`]), placed here and nowhere else.
//!
//! A run is placed in the arena (`l1::arena`), or, staged in half `h` of it where its unit's runs
//! are pipelined, planned in the first half (`matmul::half_arena`) and moved `h` halves up; a
//! buffer declared against the arena's end ([`l1::Requirements::against_end`]) lies against the
//! ring's end (`l1::ring_end`). Each placement is checked independently of the planner
//! ([`l1::check`]) and noted as the op's (`l1::note_planned`), the same on every run whether or not
//! it was planned before.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::l1::{self, Placed, Planned, Requirements};
use crate::tensor::{Result, TensorError};

use super::{At, Microkernel, RunCode};

/// `body`'s run `at`: its requirements placed ([`place`]), then its code.
pub fn run(body: &dyn Microkernel, at: &At<'_>, half: Option<u8>) -> Result<RunCode> {
    let req = body.needs(at)?;
    let placed = place(&req, half)?;
    body.code(at, &placed)
}

/// `req` placed: in the arena, or staged in `half` of it; noted as the op's.
pub fn place(req: &Arc<Requirements>, half: Option<u8>) -> Result<Placed> {
    type Key = (u64, Option<u8>, u64, u64);
    static MEMO: OnceLock<Mutex<HashMap<Key, (Placed, Planned)>>> = OnceLock::new();
    let key = (
        req.id(),
        half,
        l1::arena().end,
        if req.has_end() { l1::ring_end() } else { 0 },
    );
    let memo = MEMO.get_or_init(Default::default);
    if let Some((placed, planned)) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        l1::note_planned(*planned);
        return Ok(placed.clone());
    }
    let (region, by) = match half {
        Some(h) => (
            crate::matmul::half_arena(),
            h as u64 * crate::matmul::half(),
        ),
        None if req.has_end() => (
            tt_isa::l1::Region {
                end: l1::ring_end(),
                ..l1::arena()
            },
            0,
        ),
        None => (l1::arena(), 0),
    };
    let before = l1::take_planned();
    let plan = req.plan(region);
    let planned = l1::take_planned();
    l1::note_planned(before);
    let plan = plan.map_err(|e| TensorError::Shape(e.to_string()))?;
    l1::check(req, &plan, region).map_err(TensorError::Shape)?;
    // Noted where the run lies: `by` up.
    let up = |v: u64| if v == 0 { 0 } else { v + by };
    let planned = Planned {
        top: up(planned.top),
        need: up(planned.need),
    };
    let placed = Placed::new(req.clone(), plan, by);
    l1::note_planned(planned);
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key, (placed.clone(), planned));
    Ok(placed)
}

/// A body's requirements by shape ([`declared`]).
pub(crate) type Declared<K> = OnceLock<Mutex<HashMap<K, Arc<Requirements>>>>;

/// A memo of requirements by a body's shape: built once a process, so that a run's placement is
/// found by its requirements' identity ([`place`]).
pub(crate) fn declared<K: std::hash::Hash + Eq + Send + 'static>(
    memo: &'static Declared<K>,
    key: K,
    declare: impl FnOnce() -> Requirements,
) -> Arc<Requirements> {
    let memo = memo.get_or_init(Default::default);
    if let Some(r) = memo.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return r.clone();
    }
    let r = Arc::new(declare());
    memo.lock()
        .unwrap_or_else(|p| p.into_inner())
        .entry(key)
        .or_insert(r)
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run staged in the arena's second half is its first-half placement moved by half the
    /// arena, semaphores unmoved, and noted there; placed again, it is the same and noted the
    /// same.
    #[test]
    fn a_half_is_the_first_moved_and_noted_where_it_lies() {
        let req = Arc::new(crate::sfpu::kernel::requirements(
            4,
            crate::sfpu::kernel::Operands::Binary,
        ));
        let _ = l1::take_planned();
        let first = place(&req, Some(0)).unwrap();
        let p0 = l1::take_planned();
        let second = place(&req, Some(1)).unwrap();
        let p1 = l1::take_planned();
        let by = crate::matmul::half();
        for name in ["sfpu A slots", "sfpu B slots", "sfpu output slots"] {
            assert_eq!(second.addr(name), first.addr(name) + by, "{name}");
        }
        assert_eq!(
            second.semaphore("sfpu computed"),
            first.semaphore("sfpu computed")
        );
        assert_eq!(p1.top, p0.top + by);
        assert_eq!(p1.need, p0.need + by);
        let again = place(&req, Some(1)).unwrap();
        assert_eq!(l1::take_planned(), p1);
        assert_eq!(again.addr("sfpu A slots"), second.addr("sfpu A slots"));
        // Not one item fits: refused, not placed past the arena.
        let big = Arc::new(crate::sfpu::kernel::requirements(
            1 << 12,
            crate::sfpu::kernel::Operands::Binary,
        ));
        assert!(place(&big, None).is_err());
    }
}
