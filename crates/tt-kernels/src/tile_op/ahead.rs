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
/// ([`take_reads`]), and `None` where it may not: a `B` laid by column but read a tile at a time
/// (no `record::REF_L1_BY_COLUMN`), which the mover reads ahead only by column, stays in place.
pub fn split(entries: &mut Vec<[u32; 8]>, prefetch: bool) -> Option<Vec<[u32; 8]>> {
    let at = entries.iter().position(|e| e[0] == record::GATHER)?;
    let b = entries.get(at + 3).map_or(0, |e| e[4]);
    if prefetch && b & record::REF_BY_COLUMN != 0 && b & record::REF_L1_BY_COLUMN == 0 {
        return None;
    }
    let mut reads = entries.split_off(at);
    debug_assert!(reads.len() >= GATHER_ENTRIES);
    if prefetch {
        reads[0][1] = reads[0][1] & !record::GATHER_STREAM | record::GATHER_PREFETCH;
    }
    Some(reads)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A list: a packed operand's read, then a `GATHER` whose `B` reference's word 4 is `b_marks`.
    fn list(b_marks: u32) -> Vec<[u32; 8]> {
        let mut e = vec![[0u32; 8]; 1 + GATHER_ENTRIES];
        e[0][0] = tt_isa::dm::op::READ;
        e[1][0] = record::GATHER;
        e[1][1] = record::GATHER_STREAM;
        e[4][4] = b_marks;
        e
    }

    /// A `B` laid by column is read ahead only by column; read a tile at a time, its gather stays
    /// in its list (the mover refuses that read ahead, `record::gather_span`).
    #[test]
    fn a_weight_laid_by_column_is_read_ahead_only_by_column() {
        let by_column = record::REF_BY_COLUMN;
        let in_l1_by_column = by_column | record::REF_L1_BY_COLUMN;
        for (marks, ahead) in [(0, true), (in_l1_by_column, true), (by_column, false)] {
            let mut entries = list(marks);
            let early = split(&mut entries, true);
            assert_eq!(early.is_some(), ahead, "marks {marks:#x}");
            if let Some(early) = early {
                assert_eq!(entries.len(), 1, "the operand's read stays");
                assert_eq!(early[0][1], record::GATHER_PREFETCH, "marks {marks:#x}");
            } else {
                assert_eq!(entries, list(marks), "left as it was");
            }
            // Split to run elsewhere in its turn (no read ahead), any layout moves.
            assert!(split(&mut list(marks), false).is_some(), "marks {marks:#x}");
        }
    }
}
