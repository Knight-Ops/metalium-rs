//! Operand reuse across cores (X4 of `docs/learnings/dataflow-compiler.md` §6, H1): the units of
//! a shared deal ([`super::distribute::Shared`]) that read the same strip of a contraction's
//! first operand read it from GDDR once a holder, and copy it from the holder's L1 otherwise; and
//! with the deal's column groups, the same for the second operand.
//!
//! A lowered job's `GATHER` records name each chunk's strips (each operand's tensor reference,
//! first tile, extent, terms and slots), so the pass works on records alone, as [`super::ahead`]
//! does. A reader's strips of an operand are its holder's in order: its `t`-th job of a row is
//! its holder's `t`-th (the deal's promise), so each reader gather is paired with the holder's
//! next gather of the same strip, and an op whose gathers do not pair is refused.
//!
//! Each copy is credited both ways through the tile's op-level credit words
//! ([`tt_isa::dm::CREDITS`]), three an operand ([`READY`] and [`FREED`] for `A`, [`READY_B`]
//! and [`FREED_B`] for `B`):
//! - The holder gathers a strip, then credits each of its readers' ready word ([`op::CREDIT`]
//!   waits for the gather's reads first, so the credit follows the data).
//! - A reader waits for its ready count, copies the strip into the same slots ([`op::READ_L1`]),
//!   credits its holder's freed word for those slots, and gathers only what it does not copy
//!   ([`record::GATHER_A_RESIDENT`], [`record::GATHER_B_RESIDENT`]; both: nothing).
//! - Before a holder gathers into slots again, it waits for every copy of the strips it gathered
//!   there before. One count a slot range keeps that sound: a copy of a later strip in the same
//!   slots needs a credit the holder sends only after that wait.
//! - A reader's credits of an operand come from its one holder, in order; a holder's frees in a
//!   slot range come only from copies it granted. A unit may hold one operand and read the other.
//!   Unit `(k, m)` (member `m` of group `k`) waits for ready credits only from `(k, m')` with
//!   `m' < m` (its holder of `A`) and `(k', m)` with `k' < k` (of `B`), and a free only for a
//!   strip before: the waits cannot cycle.
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
use super::Share;
use crate::tensor::{Job, Result, Reuse, Step, TensorError};

/// The credit word in which a reader counts the strips of `A` its holder has gathered for it.
pub const READY: u32 = 0;
/// The credit words in which a holder of `A` counts its readers' copies, one for each slot range
/// its strips land in (a chain's two halves).
pub const FREED: [u32; 2] = [1, 2];
/// [`READY`] for `B`.
pub const READY_B: u32 = 3;
/// [`FREED`] for `B`.
pub const FREED_B: [u32; 2] = [4, 5];
/// The credit words the pass counts in.
const WORDS: usize = 6;

/// A `GATHER` record and its two tensor references, two words each.
const GATHER_ENTRIES: usize = 5;

/// Strips the pass has made copies of, in this process: what a gate reads to know an op shared.
static COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many strips [`pull`] has made copies of in this process.
pub fn copies() -> u64 {
    COPIES.load(std::sync::atomic::Ordering::Relaxed)
}

/// A gate's negative control, by operand (bit 0 `A`, bit 1 `B`): each copy of that operand reads
/// its holder's strip 16 bytes past where it is.
static MISROUTE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn set_misroute(bit: u8, on: bool) {
    if on {
        MISROUTE.fetch_or(bit, std::sync::atomic::Ordering::Relaxed);
    } else {
        MISROUTE.fetch_and(!bit, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Whether [`pull`]'s copies of `A` read 16 bytes past their strip (a gate's negative control:
/// the product must then differ).
pub fn probe_misroute(on: bool) {
    set_misroute(1, on);
}

/// [`probe_misroute`] for the copies of `B`.
pub fn probe_misroute_b(on: bool) {
    set_misroute(2, on);
}

/// A gate's negative control: each reader's last wait asks for one credit more than its holder
/// sends, so the op stalls until its deadline.
static OVERWAIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether [`pull`]'s readers wait, at their last copy, for a credit that never comes (a gate's
/// negative control: the op must fail, and the session must recover from it).
pub fn probe_overwait(on: bool) {
    OVERWAIT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Holders a group when a measured choice pulls `A` ([`Reuse::Auto`],
/// `tensor::BlockStream::Measured`): D5's scheme, `A` pulled from four
/// (`docs/learnings/performance-log.md`, "D5 decided").
pub const AUTO_HOLDERS: usize = 4;

/// How an op's units share their reads under `reuse` ([`super::TileOp::share`]), each unit's
/// tile in `tiles`; `None` where they read their own, or where `tiles` is empty (the session does
/// not let its movers copy). [`Reuse::Auto`] shares only what a measurement chose to pull
/// (`tensor::BlockStream::Measured` passes its choice's [`Reuse::Pull`] here): unmeasured, it
/// shares nothing. No rule by format or shape: D5's "pull with an FP32 `B`", from prefill
/// shapes, ran ResNet-18 and ViT-Tiny 10-11% slower (F4b.2). The deal decides whether two
/// units share a strip at all ([`super::distribute::shared`]).
pub fn share(reuse: Reuse, tiles: &[(u8, u8)]) -> Option<Share> {
    let Reuse::Pull { holders, b_holders } = reuse else {
        return None;
    };
    (!tiles.is_empty()).then(|| Share {
        holders,
        b_holders,
        tiles: tiles.into(),
    })
}

/// The flags that say which tiles of the first operand a `GATHER` reads and how: transposed,
/// live rows, terms and format.
const A_FLAGS: u32 = 1 | 0x1F << 3 | 0x3FFF << 8 | record::GATHER_A_FORMAT_MASK;
/// [`A_FLAGS`] for the second operand: transposed, transposed by the unpacker, terms and format.
const B_FLAGS: u32 =
    2 | record::GATHER_B_UNPACK_TRANSPOSE | 0x3FFF << 8 | record::GATHER_B_FORMAT_MASK;

/// One of a contraction's two operands, as a `GATHER` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operand {
    A,
    B,
}

impl Operand {
    /// The flags that say which of its tiles a gather reads and how.
    fn flags(self) -> u32 {
        match self {
            Operand::A => A_FLAGS,
            Operand::B => B_FLAGS,
        }
    }

    /// The flag that says a gather does not read it.
    fn resident(self) -> u32 {
        match self {
            Operand::A => record::GATHER_A_RESIDENT,
            Operand::B => record::GATHER_B_RESIDENT,
        }
    }

    /// Its ready word and its freed words.
    fn words(self) -> (u32, [u32; 2]) {
        match self {
            Operand::A => (READY, FREED),
            Operand::B => (READY_B, FREED_B),
        }
    }

    /// The header words of its slots, first tile and extent (rows of `A`, columns of `B`), and
    /// the entry of its tensor reference.
    fn fields(self) -> ([usize; 3], usize) {
        match self {
            Operand::A => ([2, 4, 5], 1),
            Operand::B => ([3, 6, 7], 3),
        }
    }

    /// Its slot's bytes under `flags`.
    fn slot(self, flags: u32) -> std::result::Result<u64, u32> {
        match self {
            Operand::A => record::gather_a_tile_format(flags).map(|f| f.slot_bytes()),
            Operand::B => record::gather_b_tile_format(flags).map(|f| f.slot_bytes()),
        }
    }

    fn misrouted(self) -> bool {
        let bit = match self {
            Operand::A => 1,
            Operand::B => 2,
        };
        MISROUTE.load(std::sync::atomic::Ordering::Relaxed) & bit != 0
    }
}

/// The strip a gather reads of an operand: its flags, slots, first tile, extent and tensor
/// reference.
type Strip = (u32, u32, u32, u32, [[u32; 8]; 2]);

/// One `GATHER` of a unit: where it is, and the strip it reads.
struct Gather {
    job: usize,
    step: usize,
    at: usize,
    strip: Strip,
    /// The operand's slots, and how many bytes the strip fills from there.
    slots: u32,
    bytes: u32,
}

/// Each unit's gathers that read `operand`, in the order it runs them: jobs `j` with
/// `(first + j) % units == u`.
fn gathers(jobs: &[Job], units: usize, first: usize, operand: Operand) -> Result<Vec<Vec<Gather>>> {
    let ([slots, from, extent], reference) = operand.fields();
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
                    if flags & operand.resident() == 0 {
                        let slot = operand.slot(flags).map_err(|code| {
                            TensorError::Shape(format!(
                                "operand reuse: a gather's format code ({code})"
                            ))
                        })?;
                        let bytes = head[extent] as u64 * record::gather_kt(flags) as u64 * slot;
                        out[unit].push(Gather {
                            job: j,
                            step: s,
                            at,
                            strip: (
                                flags & operand.flags(),
                                head[slots],
                                head[from],
                                head[extent],
                                [entries[at + reference], entries[at + reference + 1]],
                            ),
                            slots: head[slots],
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

/// What the pass adds around one gather: entries before its record and after it, and the
/// operands the record no longer reads (its resident flags).
#[derive(Default)]
struct Edit {
    before: Vec<[u32; 8]>,
    after: Vec<[u32; 8]>,
    resident: u32,
}

/// Rewrite `jobs` (job `j` on unit `(first + j) % units`) so that each reader of `shared` copies
/// its strips from its holders' L1 instead of reading them from GDDR: of the first operand, and
/// of the second where the deal has column groups. `tiles` names each unit's tile. Refused where
/// a reader's strips are not its holder's, in order, or a holder's strips land in more than two
/// slot ranges.
pub fn pull(
    jobs: &mut [Job],
    units: usize,
    first: usize,
    shared: &Shared,
    tiles: &[(u8, u8)],
) -> Result<()> {
    let units = units.max(1);
    let b_shared = shared.b_holder_of.iter().any(Option::is_some);
    if tiles.len() < units
        || shared.holder_of.len() < units
        || (b_shared && shared.b_holder_of.len() < units)
    {
        return Err(TensorError::Shape(format!(
            "operand reuse: {} tiles and {} holders for {units} units",
            tiles.len(),
            shared.holder_of.len()
        )));
    }
    let mut edits: HashMap<(usize, usize, usize), Edit> = HashMap::new();
    // Per unit, its count in each word, for the return to zero.
    let mut totals = vec![[0u32; WORDS]; units];
    plan(
        jobs,
        units,
        first,
        &shared.holder_of,
        tiles,
        Operand::A,
        &mut edits,
        &mut totals,
    )?;
    if b_shared {
        plan(
            jobs,
            units,
            first,
            &shared.b_holder_of,
            tiles,
            Operand::B,
            &mut edits,
            &mut totals,
        )?;
    }
    for ((j, s, at), edit) in sorted(edits) {
        let Step::List { entries, .. } = &mut jobs[j][s] else {
            unreachable!("a gather is in a list");
        };
        let mut record: Vec<[u32; 8]> = entries[at..at + GATHER_ENTRIES].to_vec();
        record[0][1] |= edit.resident;
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
            if w as u32 != READY && w as u32 != READY_B {
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

/// The edits that share `operand`'s strips among the units `holder_of` pairs, added to `edits`,
/// and each unit's counts in its words to `totals`.
#[allow(clippy::too_many_arguments)]
fn plan(
    jobs: &[Job],
    units: usize,
    first: usize,
    holder_of: &[Option<usize>],
    tiles: &[(u8, u8)],
    operand: Operand,
    edits: &mut HashMap<(usize, usize, usize), Edit>,
    totals: &mut [[u32; WORDS]],
) -> Result<()> {
    let (ready, freed_words) = operand.words();
    let gathers = gathers(jobs, units, first, operand)?;
    // Each holder's gathers' readers, by pairing each reader's strips with its holder's next.
    let mut readers: Vec<Vec<Vec<usize>>> =
        gathers.iter().map(|g| vec![Vec::new(); g.len()]).collect();
    let mut paired: Vec<Vec<usize>> = gathers
        .iter()
        .map(|g| Vec::with_capacity(g.len()))
        .collect();
    for r in 0..units {
        let Some(h) = holder_of[r] else {
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
                    "operand reuse: unit {r} reads a strip of {operand:?} its holder {h} does \
                     not, in order"
                )));
            }
            readers[h][next].push(r);
            paired[r].push(next);
            next += 1;
        }
    }
    let mut copied = vec![false; units];
    for h in 0..units {
        if holder_of[h].is_some() || readers[h].iter().all(Vec::is_empty) {
            continue;
        }
        // Each slot range the holder's strips land in, a word of its own.
        let mut ranges: Vec<u32> = Vec::new();
        for g in &gathers[h] {
            if !ranges.contains(&g.slots) {
                ranges.push(g.slots);
            }
        }
        if ranges.len() > freed_words.len() {
            return Err(TensorError::Shape(format!(
                "operand reuse: a holder's strips of {operand:?} land in {} slot ranges",
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
                    credit_word(freed_words[q]),
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
                    credit_word(ready),
                    1,
                    0,
                    0,
                    0,
                ]);
            }
            freed[q] += readers[h][i].len() as u32;
        }
        totals[h][freed_words[0] as usize] = freed[0];
        totals[h][freed_words[1] as usize] = freed[1];
        // Each reader's copies, in its own order.
        for &r in readers[h].iter().flatten() {
            if std::mem::replace(&mut copied[r], true) {
                continue;
            }
            let (hx, hy) = tiles[h];
            for (n, (g, &i)) in gathers[r].iter().zip(&paired[r]).enumerate() {
                let q = ranges
                    .iter()
                    .position(|&s| s == gathers[h][i].slots)
                    .unwrap();
                let edit = edits.entry((g.job, g.step, g.at)).or_default();
                let last = n + 1 == gathers[r].len()
                    && OVERWAIT.load(std::sync::atomic::Ordering::Relaxed);
                edit.before.push([
                    op::WAIT_CREDIT,
                    credit_word(ready),
                    n as u32 + 1 + u32::from(last),
                    0,
                    0,
                    0,
                    0,
                    0,
                ]);
                let max = tt_isa::noc::niu::MAX_REQUEST_BYTES;
                let skew = 16 * u32::from(operand.misrouted());
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
                    credit_word(freed_words[q]),
                    1,
                    0,
                    0,
                    0,
                ]);
                edit.resident |= operand.resident();
            }
            totals[r][ready as usize] = gathers[r].len() as u32;
            COPIES.fetch_add(
                gathers[r].len() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
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
        for (rows, cols, units, holders, b_holders) in [
            (8, 57, 120, 1, None),
            (8, 57, 120, 4, None),
            (3, 5, 8, 2, None),
            (16, 9, 120, 2, None),
            (8, 57, 120, 2, Some(1)),
            (3, 5, 8, 2, Some(2)),
            (16, 9, 120, 2, Some(3)),
        ] {
            let s = shared(rows, cols, units, holders, b_holders).unwrap();
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
            // A column reader's holder of `B`: a unit of an earlier group that holds `B` itself
            // and, in every round the reader has a block, has one of the same column.
            let b_shared = s.b_holder_of.iter().any(Option::is_some);
            assert_eq!(b_shared, b_holders.is_some_and(|_| rows > 1));
            for (u, h) in s.b_holder_of.iter().enumerate() {
                if let Some(h) = *h {
                    assert!(h < u && s.b_holder_of[h].is_none());
                    for t in (0..s.order.len()).step_by(units) {
                        if let Some(b) = s.order.get(t + u).copied().flatten() {
                            let held = s.order.get(t + h).copied().flatten();
                            assert_eq!(held.map(|x| x % cols), Some(b % cols), "round {t}");
                        }
                    }
                }
            }
            // With `B` shared the groups are one size, and the remainder runs nothing.
            if b_holders.is_some() {
                let used = units / rows.min(units) * rows.min(units);
                for u in used..units {
                    assert!((u..s.order.len())
                        .step_by(units)
                        .all(|j| s.order[j].is_none()));
                }
            }
        }
    }

    #[test]
    fn nothing_is_shared_where_a_row_has_one_unit_or_one_block() {
        assert_eq!(shared(120, 57, 120, 1, None), None);
        assert_eq!(shared(8, 1, 120, 1, None), None);
        assert_eq!(shared(120, 57, 120, 1, Some(1)), None);
    }

    use super::{Operand, GATHER_ENTRIES};
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

    /// The strip of `operand` a `GATHER` at `e[at..]` reads: its flags, slots, first tile,
    /// extent and tensor reference.
    fn strip(e: &[[u32; 8]], at: usize, operand: Operand) -> Vec<u32> {
        let h = e[at];
        let ([slots, from, extent], reference) = operand.fields();
        let mut s = vec![h[1] & operand.flags(), h[slots], h[from], h[extent]];
        s.extend(e[at + reference]);
        s.extend(e[at + reference + 1]);
        s
    }

    /// Run every unit's lists as one program each, interleaved at random by `seed`: credits add
    /// to their tile's words, a credit wait holds its unit until its word reaches the target, a
    /// gather fills each operand's slots with its strip (each 16 KiB piece apart), and a
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
                    let flags = e[1];
                    for operand in [Operand::A, Operand::B] {
                        let ([at, _, extent], _) = operand.fields();
                        let (at, s) = (e[at], strip(p, pc[u], operand));
                        let bytes = (e[extent] as u64
                            * record::gather_kt(flags) as u64
                            * operand.slot(flags).unwrap())
                            as u32;
                        for off in (0..bytes).step_by(piece as usize) {
                            if flags & operand.resident() == 0 {
                                slots[u].insert(at + off, s.clone());
                            } else if slots[u].get(&(at + off)) != Some(&s) {
                                return Err(format!(
                                    "unit {u}: a copied strip of {operand:?} is not the one it reads"
                                ));
                            }
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

    /// Shape, units, block, and the holders of `A` a group and of `B` a column group.
    #[allow(clippy::type_complexity)]
    const CASES: [([usize; 3], usize, [usize; 3], usize, Option<usize>); 9] = [
        ([256, 256, 576], 8, [2, 2, 3], 1, None),
        ([256, 256, 576], 8, [4, 2, 3], 2, None),
        ([200, 256, 600], 6, [2, 2, 3], 1, None),
        ([2048, 2048, 16384], 120, [8, 2, 9], 4, None),
        // `B` shared too: groups of two, two groups of four, an idle remainder with two holders
        // a column group, groups of one (only `B` shared), and H1's product.
        ([256, 256, 576], 8, [2, 2, 3], 1, Some(1)),
        ([256, 256, 576], 8, [4, 2, 3], 2, Some(1)),
        ([256, 256, 576], 9, [2, 2, 3], 1, Some(2)),
        ([200, 256, 600], 6, [2, 2, 3], 1, Some(1)),
        ([2048, 2048, 16384], 120, [8, 2, 9], 2, Some(1)),
    ];

    /// The shared product's copies land what its gathers would have read, under any
    /// interleaving, with no stall and every word back at zero; its kernels are the unshared
    /// product's, word for word, and the readers of each shared operand read none of it from
    /// GDDR.
    #[test]
    fn copies_land_what_the_gathers_read_under_any_interleaving() {
        for (shape, units, block, holders, b_holders) in CASES {
            let (plain, _) = product(shape, units, block, Reuse::Off);
            let reuse = Reuse::Pull { holders, b_holders };
            let (shared, tiles) = product(shape, units, block, reuse);
            let kernels = |jobs: &[tensor::Job]| {
                let mut k: Vec<_> = jobs
                    .iter()
                    .flatten()
                    .filter_map(|s| match s {
                        Step::Kernel { roles, half, .. } => Some((roles.clone(), *half)),
                        _ => None,
                    })
                    .collect();
                k.sort_by_cached_key(|x| format!("{x:?}"));
                k
            };
            assert_eq!(kernels(&plain), kernels(&shared), "{shape:?}");
            let resident = |flag: u32| {
                shared
                    .iter()
                    .flatten()
                    .filter_map(|s| match s {
                        Step::List { entries, .. } => Some(entries),
                        _ => None,
                    })
                    .flatten()
                    .filter(|e| e[0] == record::GATHER && e[1] & flag != 0)
                    .count()
            };
            assert!(
                resident(record::GATHER_A_RESIDENT | record::GATHER_B_RESIDENT) > 0,
                "{shape:?}: something is shared"
            );
            assert_eq!(
                resident(record::GATHER_B_RESIDENT) > 0,
                b_holders.is_some(),
                "{shape:?}: B shared as asked"
            );
            let seeds = if units > 16 { 1 } else { 64 };
            for seed in 0..seeds {
                run(&shared, &tiles, seed).unwrap_or_else(|e| panic!("{shape:?} seed {seed}: {e}"));
            }
        }
    }

    /// Unmeasured, `Auto` shares nothing; an explicit scheme is taken as asked; no tiles, no
    /// sharing.
    #[test]
    fn auto_shares_nothing_unmeasured_and_a_scheme_as_asked() {
        let tiles = [(1, 2), (2, 2)];
        assert_eq!(super::share(Reuse::Auto, &tiles), None);
        assert_eq!(super::share(Reuse::Off, &tiles), None);
        let asked = Reuse::Pull {
            holders: 2,
            b_holders: Some(1),
        };
        let s = super::share(asked, &tiles).expect("an explicit scheme");
        assert_eq!((s.holders, s.b_holders), (2, Some(1)));
        assert_eq!(super::share(asked, &[]), None);
    }

    /// The negative control: a holder that gathers into its slots without waiting for its
    /// readers' copies lets some interleaving copy a later strip.
    #[test]
    fn a_holder_that_does_not_wait_for_its_copies_is_caught() {
        let reuse = Reuse::Pull {
            holders: 1,
            b_holders: None,
        };
        caught_without_waits(product([256, 256, 576], 8, [2, 2, 3], reuse), super::FREED);
    }

    /// [`a_holder_that_does_not_wait_for_its_copies_is_caught`] for a holder of `B`.
    #[test]
    fn a_holder_of_b_that_does_not_wait_for_its_copies_is_caught() {
        let reuse = Reuse::Pull {
            holders: 1,
            b_holders: Some(1),
        };
        caught_without_waits(
            product([256, 256, 576], 8, [2, 2, 3], reuse),
            super::FREED_B,
        );
    }

    /// Drop every wait on `freed` before a gather; some interleaving must then fail.
    fn caught_without_waits((mut jobs, tiles): (Vec<tensor::Job>, Vec<(u8, u8)>), freed: [u32; 2]) {
        let freed = [credit_word(freed[0]), credit_word(freed[1])];
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
