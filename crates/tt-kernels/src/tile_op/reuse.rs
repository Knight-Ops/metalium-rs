//! Operand reuse across cores (X4 of `docs/learnings/dataflow-compiler.md` §6, H1): the units of
//! a shared deal ([`super::distribute::Shared`]) that read the same strip of a contraction's
//! first operand read it from GDDR once a holder, and copy it from the holder's L1 otherwise.
//!
//! A lowered job's `GATHER` records name each chunk's strip (its tensor reference, first term,
//! rows, terms and slots), so the pass works on records alone, as [`super::ahead`] does. A
//! reader's strips are its holder's in order: its `t`-th job of a row is its holder's `t`-th
//! (the deal's promise), so each reader gather is paired with the holder's next gather of the
//! same strip, and an op whose gathers do not pair is refused.
//!
//! Each copy is credited both ways through the tile's op-level credit words
//! ([`tt_isa::dm::CREDITS`]):
//! - The holder gathers a strip, then credits each of its readers' [`READY`] word
//!   ([`op::CREDIT`] waits for the gather's reads first, so the credit follows the data).
//! - A reader waits for its [`READY`] count, copies the strip into the same slots
//!   ([`op::READ_L1`]), credits its holder's [`FREED`] word for those slots, and gathers only its
//!   second operand ([`record::GATHER_A_RESIDENT`]).
//! - Before a holder gathers into slots again, it waits for every copy of the strips it gathered
//!   there before. One count a slot range keeps that sound: a copy of a later strip in the same
//!   slots needs a credit the holder sends only after that wait.
//! - A reader's credits come from its one holder, in order; a holder's frees in a slot range come
//!   only from copies it granted. Every wait is for a credit sent at the same or an earlier
//!   strip, so the waits cannot cycle.
//!
//! Each unit leaves its words at zero: once its last wait has counted every credit aimed at a
//! word, its last list (the writer's) credits the word minus its total. The barrier after the op
//! orders that before any credit of the next.
//!
//! The copies move bytes only, into the slots the gather would have filled, so every output
//! keeps its terms, groups and order (S2): the product is bit for bit the unshared one.

use std::collections::HashMap;

use tt_isa::dm::{credit_word, op, record};

use super::distribute::Shared;
use crate::tensor::{Job, Result, Step, TensorError};

/// The credit word in which a reader counts the strips its holder has gathered for it.
pub const READY: u32 = 0;
/// The credit words in which a holder counts its readers' copies, one for each slot range its
/// strips land in (a chain's two halves).
pub const FREED: [u32; 2] = [1, 2];

/// A `GATHER` record and its two tensor references, two words each.
const GATHER_ENTRIES: usize = 5;

/// Strips the pass has made copies of, in this process: what a gate reads to know an op shared.
static COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many strips [`pull`] has made copies of in this process.
pub fn copies() -> u64 {
    COPIES.load(std::sync::atomic::Ordering::Relaxed)
}

/// A gate's negative control: each copy reads its holder's strip 16 bytes past where it is.
static MISROUTE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether [`pull`]'s copies read 16 bytes past their strip (a gate's negative control: the
/// product must then differ).
pub fn probe_misroute(on: bool) {
    MISROUTE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The flags that say which tiles of the first operand a `GATHER` reads and how: transposed,
/// live rows, terms and format.
const A_FLAGS: u32 = 1 | 0x1F << 3 | 0x3FFF << 8 | record::GATHER_A_FORMAT_MASK;

/// The strip a gather reads: its first operand's flags, slots, rows and tensor reference.
type Strip = (u32, u32, u32, u32, [[u32; 8]; 2]);

/// One `GATHER` of a unit: where it is, and the strip it reads.
struct Gather {
    job: usize,
    step: usize,
    at: usize,
    strip: Strip,
    /// The first operand's slots, and how many bytes the strip fills from there.
    slots: u32,
    bytes: u32,
}

/// Each unit's gathers, in the order it runs them: jobs `j` with `(first + j) % units == u`.
fn gathers(jobs: &[Job], units: usize, first: usize) -> Result<Vec<Vec<Gather>>> {
    let mut out: Vec<Vec<Gather>> = (0..units).map(|_| Vec::new()).collect();
    for (j, job) in jobs.iter().enumerate() {
        let unit = (first + j) % units;
        for (s, step) in job.iter().enumerate() {
            let Step::List { entries, .. } = step else {
                continue;
            };
            let mut at = 0;
            while at < entries.len() {
                let head = entries[at];
                if head[0] == record::GATHER {
                    let flags = head[1];
                    if flags & (record::GATHER_PREFETCH | record::GATHER_STREAM) != 0 {
                        return Err(TensorError::Shape(
                            "operand reuse: a gather read ahead or streamed is not shared".into(),
                        ));
                    }
                    if flags & record::GATHER_A_RESIDENT == 0 {
                        let slot = record::gather_a_tile_format(flags)
                            .map_err(|code| {
                                TensorError::Shape(format!(
                                    "operand reuse: a gather's format code ({code})"
                                ))
                            })?
                            .slot_bytes();
                        let bytes = head[5] as u64 * record::gather_kt(flags) as u64 * slot;
                        out[unit].push(Gather {
                            job: j,
                            step: s,
                            at,
                            strip: (
                                flags & A_FLAGS,
                                head[2],
                                head[4],
                                head[5],
                                [entries[at + 1], entries[at + 2]],
                            ),
                            slots: head[2],
                            bytes: u32::try_from(bytes).map_err(|_| {
                                TensorError::Shape("operand reuse: a strip past L1".into())
                            })?,
                        });
                    }
                }
                at += record::len(head[0]).max(1);
            }
        }
    }
    Ok(out)
}

/// What the pass adds around one gather: entries before its record and after it, and whether
/// the record no longer reads its first operand.
#[derive(Default)]
struct Edit {
    before: Vec<[u32; 8]>,
    after: Vec<[u32; 8]>,
    resident: bool,
}

/// Rewrite `jobs` (job `j` on unit `(first + j) % units`) so that each reader of `shared` copies
/// its strips from its holder's L1 instead of reading them from GDDR. `tiles` names each unit's
/// tile. Refused where a reader's strips are not its holder's, in order, or a holder's strips
/// land in more than two slot ranges.
pub fn pull(
    jobs: &mut [Job],
    units: usize,
    first: usize,
    shared: &Shared,
    tiles: &[(u8, u8)],
) -> Result<()> {
    let units = units.max(1);
    if tiles.len() < units || shared.holder_of.len() < units {
        return Err(TensorError::Shape(format!(
            "operand reuse: {} tiles and {} holders for {units} units",
            tiles.len(),
            shared.holder_of.len()
        )));
    }
    let gathers = gathers(jobs, units, first)?;
    // Each holder's gathers' readers, by pairing each reader's strips with its holder's next.
    let mut readers: Vec<Vec<Vec<usize>>> =
        gathers.iter().map(|g| vec![Vec::new(); g.len()]).collect();
    let mut paired: Vec<Vec<usize>> = gathers
        .iter()
        .map(|g| Vec::with_capacity(g.len()))
        .collect();
    for r in 0..units {
        let Some(h) = shared.holder_of[r] else {
            continue;
        };
        let mut next = 0;
        for g in &gathers[r] {
            let held = &gathers[h];
            while next < held.len() && held[next].strip != g.strip {
                next += 1;
            }
            if next == held.len() {
                return Err(TensorError::Shape(format!(
                    "operand reuse: unit {r} reads a strip its holder {h} does not, in order"
                )));
            }
            readers[h][next].push(r);
            paired[r].push(next);
            next += 1;
        }
    }
    let mut edits: HashMap<(usize, usize, usize), Edit> = HashMap::new();
    // Per unit, its count in each word, for the return to zero.
    let mut totals = vec![[0u32; 3]; units];
    for h in 0..units {
        if shared.holder_of[h].is_some() || readers[h].iter().all(Vec::is_empty) {
            continue;
        }
        // Each slot range the holder's strips land in, a word of its own.
        let mut ranges: Vec<u32> = Vec::new();
        for g in &gathers[h] {
            if !ranges.contains(&g.slots) {
                ranges.push(g.slots);
            }
        }
        if ranges.len() > FREED.len() {
            return Err(TensorError::Shape(format!(
                "operand reuse: a holder's strips land in {} slot ranges",
                ranges.len()
            )));
        }
        let mut freed = [0u32; 2];
        for (i, g) in gathers[h].iter().enumerate() {
            let q = ranges.iter().position(|&s| s == g.slots).unwrap();
            let edit = edits.entry((g.job, g.step, g.at)).or_default();
            if freed[q] > 0 {
                edit.before.push([
                    op::WAIT_CREDIT,
                    credit_word(FREED[q]),
                    freed[q],
                    0,
                    0,
                    0,
                    0,
                    0,
                ]);
            }
            for &r in &readers[h][i] {
                let (x, y) = tiles[r];
                edit.after.push([
                    op::CREDIT,
                    x as u32,
                    y as u32,
                    credit_word(READY),
                    1,
                    0,
                    0,
                    0,
                ]);
            }
            freed[q] += readers[h][i].len() as u32;
        }
        totals[h][1] = freed[0];
        totals[h][2] = freed[1];
        // Each reader's copies, in its own order.
        for &r in readers[h].iter().flatten() {
            if !totals[r].iter().all(|&t| t == 0) {
                continue;
            }
            let (hx, hy) = tiles[h];
            for (n, (g, &i)) in gathers[r].iter().zip(&paired[r]).enumerate() {
                let q = ranges
                    .iter()
                    .position(|&s| s == gathers[h][i].slots)
                    .unwrap();
                let edit = edits.entry((g.job, g.step, g.at)).or_default();
                edit.before.push([
                    op::WAIT_CREDIT,
                    credit_word(READY),
                    n as u32 + 1,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]);
                let max = tt_isa::noc::niu::MAX_REQUEST_BYTES;
                let skew = 16 * u32::from(MISROUTE.load(std::sync::atomic::Ordering::Relaxed));
                for off in (0..g.bytes).step_by(max as usize) {
                    let at = g.slots + off;
                    edit.before.push([
                        op::READ_L1,
                        hx as u32,
                        hy as u32,
                        at + skew,
                        at,
                        (g.bytes - off).min(max),
                        0,
                        0,
                    ]);
                }
                edit.before.push([
                    op::CREDIT,
                    hx as u32,
                    hy as u32,
                    credit_word(FREED[q]),
                    1,
                    0,
                    0,
                    0,
                ]);
                edit.resident = true;
            }
            totals[r][0] = gathers[r].len() as u32;
            COPIES.fetch_add(
                gathers[r].len() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }
    for ((j, s, at), edit) in sorted(edits) {
        let Step::List { entries, .. } = &mut jobs[j][s] else {
            unreachable!("a gather is in a list");
        };
        let mut record: Vec<[u32; 8]> = entries[at..at + GATHER_ENTRIES].to_vec();
        if edit.resident {
            record[0][1] |= record::GATHER_A_RESIDENT;
        }
        let replacement = edit.before.into_iter().chain(record).chain(edit.after);
        entries.splice(at..at + GATHER_ENTRIES, replacement);
    }
    // Each unit's words back to zero, in its last list: the writer's, after its last kernel and
    // so after its last wait for a strip.
    for (u, total) in totals.iter().enumerate() {
        if total.iter().all(|&t| t == 0) {
            continue;
        }
        let last = (0..jobs.len())
            .rev()
            .find(|&j| (first + j) % units == u && !jobs[j].is_empty())
            .and_then(|j| {
                let s = jobs[j]
                    .iter()
                    .rposition(|step| matches!(step, Step::List { .. }))?;
                Some((j, s))
            })
            .ok_or_else(|| TensorError::Shape(format!("operand reuse: unit {u} has no list")))?;
        let (x, y) = tiles[u];
        let mut restore = Vec::new();
        for (w, &t) in total.iter().enumerate() {
            if t == 0 {
                continue;
            }
            let word = credit_word(w as u32);
            // A holder's frees come from other tiles: all of them counted before it returns.
            if w != READY as usize {
                restore.push([op::WAIT_CREDIT, word, t, 0, 0, 0, 0, 0]);
            }
            restore.push([
                op::CREDIT,
                x as u32,
                y as u32,
                word,
                t.wrapping_neg(),
                0,
                0,
                0,
            ]);
        }
        let Step::List { entries, .. } = &mut jobs[last.0][last.1] else {
            unreachable!("found as a list");
        };
        entries.extend(restore);
    }
    Ok(())
}

/// `edits` by position, the last of a list first, so each splice leaves the earlier positions
/// where they were.
fn sorted(edits: HashMap<(usize, usize, usize), Edit>) -> Vec<((usize, usize, usize), Edit)> {
    let mut v: Vec<_> = edits.into_iter().collect();
    v.sort_by_key(|e| std::cmp::Reverse(e.0));
    v
}

#[cfg(test)]
mod tests {
    use crate::tile_op::distribute::shared;

    /// Every unit of a group but its holders has a holder in its group, holders have none, and
    /// each block runs once.
    #[test]
    fn a_shared_deal_runs_every_block_once_in_groups() {
        for (rows, cols, units, holders) in [
            (8, 57, 120, 1),
            (8, 57, 120, 4),
            (3, 5, 8, 2),
            (16, 9, 120, 2),
        ] {
            let s = shared(rows, cols, units, holders).unwrap();
            let mut seen = vec![0; rows * cols];
            for b in s.order.iter().flatten() {
                seen[*b] += 1;
            }
            assert!(seen.iter().all(|&n| n == 1), "{rows}x{cols} on {units}");
            for (u, h) in s.holder_of.iter().enumerate() {
                if let Some(h) = *h {
                    assert!(h < u && s.holder_of[h].is_none());
                    // In every round a reader has a block, its holder has one of the same row.
                    for t in (0..s.order.len()).step_by(units) {
                        if let Some(b) = s.order.get(t + u).copied().flatten() {
                            let held = s.order.get(t + h).copied().flatten();
                            assert_eq!(held.map(|x| x / cols), Some(b / cols), "round {t}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nothing_is_shared_where_a_row_has_one_unit_or_one_block() {
        assert_eq!(shared(120, 57, 120, 1), None);
        assert_eq!(shared(8, 1, 120, 1), None);
    }

    use super::{A_FLAGS, GATHER_ENTRIES};
    use crate::matmul::{Fidelity, SrcRoute};
    use crate::tensor::{self, BlockStream, DramAlloc, DramTensor, Elem, Reads, Reuse, Step};
    use tt_isa::dm::{credit_word, op, record, CREDIT_WORDS};
    use tt_isa::dram::Dram;

    /// `[m, k, n]` on `units`, blocks `[mb, kq, nb]`, its units sharing with `reuse`: the work.
    fn product(
        [m, k, n]: [usize; 3],
        units: usize,
        block: [usize; 3],
        reuse: Reuse,
    ) -> (Vec<tensor::Job>, Vec<(u8, u8)>) {
        let mut alloc = DramAlloc::new(&Dram::FULL);
        let a = DramTensor::alloc_elem(&mut alloc, m, k, Elem::F32).unwrap();
        let b = DramTensor::alloc_elem(&mut alloc, k, n, Elem::F32).unwrap();
        let tiles: Vec<(u8, u8)> = (0..units)
            .map(|u| (1 + (u % 7) as u8, 2 + (u / 7) as u8))
            .collect();
        let reads = Reads {
            thin: true,
            resident_a: true,
            unpack_b: true,
            packed_a: None,
            stream_k: true,
            stream_prefetch: true,
            prefetch_units: [crate::session::PREFETCH_UNITS; 2],
            handoff: None,
            second_job: false,
            balanced_deal: false,
            block_stream: BlockStream::Fixed(block),
            reuse,
        };
        let (work, ..) = tensor::matmul_dram_prefetching(
            &mut alloc,
            &a,
            false,
            &b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            units,
            false,
            true,
            None,
            reads,
            &[],
            units,
            &[],
            &tiles,
        )
        .unwrap();
        (work.jobs, tiles)
    }

    /// The strip a `GATHER` at `e[at..]` reads: its first operand's flags, slots, rows and
    /// tensor reference.
    fn strip(e: &[[u32; 8]], at: usize) -> Vec<u32> {
        let h = e[at];
        let mut s = vec![h[1] & A_FLAGS, h[2], h[4], h[5]];
        s.extend(e[at + 1]);
        s.extend(e[at + 2]);
        s
    }

    /// Run every unit's lists as one program each, interleaved at random by `seed`: credits add
    /// to their tile's words, a credit wait holds its unit until its word reaches the target, a
    /// gather fills its first operand's slots with its strip (each 16 KiB piece apart), and a
    /// copy takes the holder's pieces as they are when it runs. Each gather the pass made
    /// resident must find, in every piece, the strip it would have read; nothing may stall; every
    /// word ends at zero.
    fn run(jobs: &[tensor::Job], tiles: &[(u8, u8)], seed: u64) -> Result<(), String> {
        let units = tiles.len();
        let programs: Vec<Vec<[u32; 8]>> = (0..units)
            .map(|u| {
                jobs.iter()
                    .enumerate()
                    .filter(|(j, _)| j % units == u)
                    .flat_map(|(_, job)| job.iter())
                    .filter_map(|s| match s {
                        Step::List { entries, .. } => Some(entries.clone()),
                        _ => None,
                    })
                    .flatten()
                    .collect()
            })
            .collect();
        let unit_of = |x: u32, y: u32| {
            tiles
                .iter()
                .position(|&t| t == (x as u8, y as u8))
                .expect("a credit or copy to one of the op's tiles")
        };
        let word = |addr: u32| {
            (0..CREDIT_WORDS)
                .position(|i| credit_word(i) == addr)
                .unwrap()
        };
        let mut words = vec![[0u32; CREDIT_WORDS as usize]; units];
        let mut slots: Vec<std::collections::HashMap<u32, Vec<u32>>> =
            vec![Default::default(); units];
        let mut pc = vec![0usize; units];
        let mut rng = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let piece = tt_isa::noc::niu::MAX_REQUEST_BYTES;
        loop {
            let runnable: Vec<usize> = (0..units)
                .filter(|&u| match programs[u].get(pc[u]) {
                    None => false,
                    Some(e) if e[0] == op::WAIT_CREDIT => {
                        (words[u][word(e[1])].wrapping_sub(e[2]) as i32) >= 0
                    }
                    Some(_) => true,
                })
                .collect();
            if runnable.is_empty() {
                if let Some(u) = (0..units).find(|&u| pc[u] < programs[u].len()) {
                    return Err(format!("stalled: unit {u} at {:?}", programs[u][pc[u]]));
                }
                break;
            }
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = runnable[(rng >> 33) as usize % runnable.len()];
            let p = &programs[u];
            let e = p[pc[u]];
            match e[0] {
                op::CREDIT => {
                    let w = &mut words[unit_of(e[1], e[2])][word(e[3])];
                    *w = w.wrapping_add(e[4]);
                }
                op::READ_L1 => {
                    let h = unit_of(e[1], e[2]);
                    let got = slots[h].get(&e[3]).cloned().unwrap_or_default();
                    slots[u].insert(e[4], got);
                }
                record::GATHER => {
                    let (flags, at) = (e[1], e[2]);
                    let s = strip(p, pc[u]);
                    let slot = record::gather_a_tile_format(flags).unwrap().slot_bytes();
                    let bytes = (e[5] as u64 * record::gather_kt(flags) as u64 * slot) as u32;
                    for off in (0..bytes).step_by(piece as usize) {
                        if flags & record::GATHER_A_RESIDENT == 0 {
                            slots[u].insert(at + off, s.clone());
                        } else if slots[u].get(&(at + off)) != Some(&s) {
                            return Err(format!(
                                "unit {u}: a copied strip is not the one it reads"
                            ));
                        }
                    }
                }
                _ => {}
            }
            pc[u] += match e[0] {
                record::GATHER => GATHER_ENTRIES,
                h => record::len(h).max(1),
            };
        }
        match words.iter().position(|w| w.iter().any(|&x| x != 0)) {
            Some(u) => Err(format!("unit {u}'s words end at {:?}", words[u])),
            None => Ok(()),
        }
    }

    const CASES: [([usize; 3], usize, [usize; 3], usize); 4] = [
        ([256, 256, 576], 8, [2, 2, 3], 1),
        ([256, 256, 576], 8, [4, 2, 3], 2),
        ([200, 256, 600], 6, [2, 2, 3], 1),
        ([2048, 2048, 16384], 120, [8, 2, 9], 4),
    ];

    /// The shared product's copies land what its gathers would have read, under any
    /// interleaving, with no stall and every word back at zero; its kernels are the unshared
    /// product's, word for word, and each reader reads no `A` from GDDR.
    #[test]
    fn copies_land_what_the_gathers_read_under_any_interleaving() {
        for (shape, units, block, holders) in CASES {
            let (plain, _) = product(shape, units, block, Reuse::Off);
            let (shared, tiles) = product(shape, units, block, Reuse::Pull { holders });
            let kernels = |jobs: &[tensor::Job]| {
                let mut k: Vec<_> = jobs
                    .iter()
                    .flatten()
                    .filter_map(|s| match s {
                        Step::Kernel { roles, half, .. } => Some((roles.clone(), *half)),
                        _ => None,
                    })
                    .collect();
                k.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
                k
            };
            assert_eq!(kernels(&plain), kernels(&shared), "{shape:?}");
            let resident = shared
                .iter()
                .flatten()
                .filter_map(|s| match s {
                    Step::List { entries, .. } => Some(entries),
                    _ => None,
                })
                .flatten()
                .filter(|e| e[0] == record::GATHER && e[1] & record::GATHER_A_RESIDENT != 0)
                .count();
            assert!(resident > 0, "{shape:?}: something is shared");
            let seeds = if units > 16 { 1 } else { 64 };
            for seed in 0..seeds {
                run(&shared, &tiles, seed).unwrap_or_else(|e| panic!("{shape:?} seed {seed}: {e}"));
            }
        }
    }

    /// The negative control: a holder that gathers into its slots without waiting for its
    /// readers' copies lets some interleaving copy a later strip.
    #[test]
    fn a_holder_that_does_not_wait_for_its_copies_is_caught() {
        let (mut jobs, tiles) = product([256, 256, 576], 8, [2, 2, 3], Reuse::Pull { holders: 1 });
        let freed = [credit_word(super::FREED[0]), credit_word(super::FREED[1])];
        let mut dropped = 0;
        for step in jobs.iter_mut().flatten() {
            if let Step::List { entries, .. } = step {
                let before = entries.len();
                // Only the waits before a gather, not the last list's before it returns.
                let gathers = entries.iter().any(|e| e[0] == record::GATHER);
                entries.retain(|e| !(gathers && e[0] == op::WAIT_CREDIT && freed.contains(&e[1])));
                dropped += before - entries.len();
            }
        }
        assert!(dropped > 0);
        let caught = (0..256)
            .filter(|&seed| run(&jobs, &tiles, seed).is_err())
            .count();
        assert!(caught > 0, "no interleaving saw the race");
    }
}
