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
    job.iter_mut()
        .filter_map(|step| match step {
            Step::List { entries, .. } => split(entries, prefetch),
            _ => None,
        })
        .take(n)
        .flatten()
        .collect()
}

/// [`take_reads`] with `prefetch`, pumped: the first `n` gathers' contraction records split out of
/// `job`'s lists as `GATHER_PREFETCH` reads, and the first list that gave one ending with the
/// `WAIT` that drains them (`dm::WAIT_PREFETCH`) before its kernel, as a unit's first run reads
/// ahead (`tile_op::lower`). The mover queues a prefetch and issues its reads in its wait loops, so
/// a list of these may run before a barrier without holding it (F4a.2, the idle units' lists).
/// Each read lands in the slots its gather named, so the job computes what it did, bit for bit.
pub fn take_reads_pumped(job: &mut Job, n: usize) -> Vec<[u32; 8]> {
    let mut reads = Vec::new();
    let mut first = None;
    let mut taken = 0;
    for (i, step) in job.iter_mut().enumerate() {
        if taken == n {
            break;
        }
        if let Step::List { entries, .. } = step {
            if let Some(r) = split(entries, true) {
                reads.extend(r);
                taken += 1;
                first.get_or_insert(i);
            }
        }
    }
    if let Some(Step::List { entries, .. }) = first.map(|i| &mut job[i]) {
        entries.push([
            tt_isa::dm::op::WAIT,
            tt_isa::dm::WAIT_PREFETCH,
            0,
            0,
            0,
            0,
            0,
            0,
        ]);
    }
    reads
}

/// One gather list's contraction record and what follows it, split out of `entries`, which keeps
/// what came before; `None` where it has none. With `prefetch`, the record reads ahead
/// ([`take_reads`]).
pub fn split(entries: &mut Vec<[u32; 8]>, prefetch: bool) -> Option<Vec<[u32; 8]>> {
    let at = entries.iter().position(|e| e[0] == record::GATHER)?;
    let mut reads = entries.split_off(at);
    debug_assert!(reads.len() >= GATHER_ENTRIES);
    if prefetch {
        reads[0][1] = reads[0][1] & !record::GATHER_STREAM | record::GATHER_PREFETCH;
    }
    Some(reads)
}
