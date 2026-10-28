//! Reads ahead (X8 of `docs/learnings/dataflow-compiler.md` §6): a lowered job's contraction
//! reads, which depend on no earlier op, moved out of its gathers so that a list placed before
//! the job (or on another unit, or before a barrier) can read them while other work runs. Each
//! read lands in the slots its gather named, so the job computes what it did, bit for bit.
//!
//! A read ahead is recognised by the mover's record alone (`record::GATHER` and its four
//! reference words); which reads may go ahead, and where the list goes, is the schedule's.

use tt_isa::dm::record;

use crate::tensor::{Job, Step};

/// A `GATHER` record and its two tensor references, two words each.
const GATHER_ENTRIES: usize = 5;

/// The first `n` gathers' contraction records split out of `job`'s lists, in order, each list
/// keeping what came before its record (a packed operand's reads). With `prefetch`, each record
/// reads ahead into its slots (`GATHER_PREFETCH` for `GATHER_STREAM`) for a later `WAIT_PREFETCH`.
pub fn take_reads(job: &mut Job, n: usize, prefetch: bool) -> Vec<[u32; 8]> {
    let mut out = Vec::new();
    let gathers = job.iter_mut().filter_map(|step| match step {
        Step::List { entries, .. } => entries
            .iter()
            .position(|e| e[0] == record::GATHER)
            .map(|at| (entries, at)),
        _ => None,
    });
    for (entries, at) in gathers.take(n) {
        let mut reads = entries.split_off(at);
        debug_assert!(reads.len() >= GATHER_ENTRIES);
        if prefetch {
            reads[0][1] = reads[0][1] & !record::GATHER_STREAM | record::GATHER_PREFETCH;
        }
        out.extend(reads);
    }
    out
}
