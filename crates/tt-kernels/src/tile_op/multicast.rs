//! Operand multicast across a grid of units (roadmap W2, X4 of
//! `docs/learnings/dataflow-compiler.md` §6, H1): the units of a grid deal
//! ([`super::distribute::grid`]) that read the same strip of a contraction's operand read it from
//! GDDR once, on its sender, whose RISCV NC writes it into every other reader's L1 with one
//! multicast a rectangle of readers ([`op::WRITE_L1_MCAST`]) -- `A` along a grid row, `B` down a
//! grid column, as tt-metal's 2-D multicast products do. [`super::reuse`] pulls the same strips;
//! this pushes them, so no unit's L1 serves more than its own kernel.
//!
//! The pass works on a lowered product's `GATHER` records, as [`super::reuse::pull`] does, and
//! pairs each reader's strips with its sender's in the same way ([`super::reuse::pair`]). Each
//! strip is credited both ways through the op-level credit words ([`tt_isa::dm::CREDITS`]):
//! [`super::reuse::READY`] and [`super::reuse::FREED`] for `A`, [`super::reuse::READY_B`] and
//! [`super::reuse::FREED_B`] for `B` (an op shares one way or the other, never both).
//!
//! - **A reader**, at its gather of a strip (its input half free: the session's `CB` reserve
//!   comes first), credits its sender's freed word, gathers only what it does not receive
//!   ([`record::GATHER_A_RESIDENT`], [`record::GATHER_B_RESIDENT`]), and then waits for its ready
//!   count before the kernel may read the half.
//! - **A sender**'s RISCV B gathers each chunk's strips into its own half as it would unshared,
//!   inside a reserve and a push of the tile's transfer stream
//!   ([`tt_isa::dataflow::Stream::Transfer`]), whether or not the chunk has readers: the push
//!   waits for the gather's reads, so the token follows the data, and the reserve keeps B off a
//!   half NC may still be multicasting from. Its RISCV NC, at the
//!   head of the chunk's writer section (before it waits for the chunk's results;
//!   `session::streaming_segments` moves everything up to the transfer pop there), waits for the
//!   token, waits until every reader of every strip it has sent so far has credited its freed word
//!   (each reader's half is free for this strip), multicasts the strip from its slots to the same
//!   slots in each rectangle of readers ([`crate::residency::rectangles`]; the rectangle never
//!   holds the sender, and splits where the grid does), raises its own ready word by one (a
//!   credit settles the multicasts' acknowledgements first, so it follows the data) and
//!   multicasts that word's 16 bytes into each rectangle's ready words, and pops the token. B
//!   gathers into that half again only after the pop (the stream is two deep). A reader waits
//!   for its ready word to reach the strip's rank among the sender's sends: the word is set, not
//!   counted, and only ever rises, so one multicast a rectangle stands for a credit a reader (one
//!   credit a reader cost ~1 us of the sender's NC each, and bound the product: roadmap W2's
//!   probe, `docs/learnings/performance-log.md`).
//!
//! A reader credits a sender's strips in order and waits for one before crediting the next. It
//! may sit a block out (a round with no block of its own: [`super::distribute::grid`]), so it can
//! free the next block's first strip before the sender has sent the block it sat out; never a
//! strip two blocks on, since it never sits out two in a row (refused otherwise). So the frees are
//! counted on two words by the parity of the strip's block among the sender's
//! ([`super::reuse::FREED`], [`super::reuse::FREED_B`]): on the word of strip `i`'s block's
//! parity no reader counts past its own strips up to `i`, and the sender's count of every
//! reader's strips of that parity up to `i` means each reader of strip `i` has freed its half. The waits cannot cycle: a reader of `B` in grid row `k` waits for row 0, and a
//! sender of `A` waits for its own row's readers, which never wait for it to receive.
//!
//! Each unit leaves its words at zero, in its last list ([`super::reuse::restore`]).

use std::collections::HashMap;

use tt_isa::dataflow::{Action, Stream};
use tt_isa::dm::{self, credit_word, op, record};

use super::distribute::Shared;
use super::reuse::{self, Edit, Operand, WORDS};
use crate::channel::Channel;
use crate::tensor::{Job, Result, Step, TensorError};

/// Strips this process has multicast: what a gate reads to know an op shared.
static STRIPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many strips [`push`] has multicast in this process.
pub fn strips() -> u64 {
    STRIPS.load(std::sync::atomic::Ordering::Relaxed)
}

/// A gate's negative control, by operand (bit 0 `A`, bit 1 `B`): each multicast of that operand
/// writes 16 bytes past where its strip goes.
static MISROUTE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether [`push`]'s multicasts of `operand` (0 `A`, 1 `B`) land 16 bytes past their readers'
/// slots (a gate's negative control: the product must then differ). Read as an op is built.
#[doc(hidden)]
pub fn probe_misroute(operand: u8, on: bool) {
    let bit = 1 << operand;
    if on {
        MISROUTE.fetch_or(bit, std::sync::atomic::Ordering::Relaxed);
    } else {
        MISROUTE.fetch_and(!bit, std::sync::atomic::Ordering::Relaxed);
    }
}

/// A timing probe, by operand (bit 0 `A`, bit 1 `B`): each sender of that operand gathers its
/// strips [`record::GATHER_A_RESIDENT`] (or `B_RESIDENT`), reading nothing of them from GDDR, and
/// still multicasts them, credits and all. **The products are wrong**; only their time means
/// anything: the bound on making a sender's reads free with its multicasts still sent.
static SKIP_SENDER_READS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether [`push`]'s senders of `operand` (0 `A`, 1 `B`) skip their strips' GDDR reads, sending
/// whatever their slots hold ([`SKIP_SENDER_READS`]). Read as an op is built. Dev-facing: a timing
/// bound, never a result.
#[doc(hidden)]
pub fn probe_skip_sender_reads(operand: u8, on: bool) {
    let bit = 1 << operand;
    if on {
        SKIP_SENDER_READS.fetch_or(bit, std::sync::atomic::Ordering::Relaxed);
    } else {
        SKIP_SENDER_READS.fetch_and(!bit, std::sync::atomic::Ordering::Relaxed);
    }
}

/// A gate's negative control: each reader's last wait asks for one credit more than its sender
/// sends, so the op stalls until its deadline.
static OVERWAIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether [`push`]'s readers wait, at their last strip, for a credit that never comes (a gate's
/// negative control: the op must fail, and the session must recover). Read as an op is built.
#[doc(hidden)]
pub fn probe_overwait(on: bool) {
    OVERWAIT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The transfer stream's credit `action`, two deep: a block-stream product's packet's capacity
/// (`session::streaming_segments`; its kernels are staged in halves).
fn transfer(action: Action) -> [u32; 8] {
    Channel::local(0, Stream::Transfer, 2).credit(action)
}

/// Whether `e` is the transfer stream's pop: where a sender's writer prefix ends
/// (`session::streaming_segments`).
pub fn is_transfer_pop(e: &[u32; 8]) -> bool {
    e[0] == op::CB && e[1] == Stream::Transfer.word() && e[2] == Action::Pop.word()
}

/// RISCV NC's list for one ownership region's chunks, from each chunk's writer section: a
/// multicast sender's prefix (its entries up to its transfer pop, [`push`]) runs one chunk ahead
/// of the chunk's results, so NC sends strip `c + 1` before it waits for chunk `c`'s output and
/// scatters it: `lead 0, lead 1, wait 0, scatter 0, pop 0, lead 2, wait 1, ...`. A prefix that
/// followed its own chunk's output wait serialised each strip's send behind the sender's own
/// kernel (the sender-side meter, `docs/learnings/performance-log.md`). The transfer stream is two
/// deep, so B gathers strip `c + 2` while NC sends `c + 1`; a sender's prefix waits only for its
/// own B and its readers' frees, neither of which waits for this unit's output past chunk `c - 1`,
/// so running it ahead adds no cycle (the interleaving test). `wait` and `pop` are the output
/// stream's credits.
pub fn writer_list(sections: Vec<Vec<[u32; 8]>>, wait: [u32; 8], pop: [u32; 8]) -> Vec<[u32; 8]> {
    let split = |s: &[[u32; 8]]| s.iter().rposition(is_transfer_pop).map_or(0, |p| p + 1);
    let mut out = Vec::with_capacity(sections.iter().map(|s| s.len() + 2).sum());
    if let Some(first) = sections.first() {
        out.extend_from_slice(&first[..split(first)]);
    }
    for (c, section) in sections.iter().enumerate() {
        if let Some(next) = sections.get(c + 1) {
            out.extend_from_slice(&next[..split(next)]);
        }
        out.push(wait);
        out.extend_from_slice(&section[split(section)..]);
        out.push(pop);
    }
    out
}

/// Rewrite `jobs` (job `j` on unit `(first + j) % units`, dealt by
/// [`super::distribute::grid`]) so that each strip with readers is read from GDDR by its sender
/// only and multicast into its readers' slots. `tiles` names each unit's tile. Refused where a
/// reader's strips are not its sender's, in order, a reader sits out two of its sender's blocks
/// in a row, or a sender's chunk does not gather once.
pub fn push(
    jobs: &mut [Job],
    units: usize,
    first: usize,
    shared: &Shared,
    tiles: &[(u8, u8)],
) -> Result<()> {
    let units = units.max(1);
    #[cfg(test)]
    if tests::SKIP_PUSH.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(());
    }
    if tiles.len() < units || shared.holder_of.len() < units || shared.b_holder_of.len() < units {
        return Err(TensorError::Shape(format!(
            "operand multicast: {} tiles and {} senders for {units} units",
            tiles.len(),
            shared.holder_of.len()
        )));
    }
    let mut edits: HashMap<(usize, usize, usize), Edit> = HashMap::new();
    // A sender's NC entries for a chunk, by the job and step of the gather that reads it.
    let mut sends: HashMap<(usize, usize), Vec<[u32; 8]>> = HashMap::new();
    // The units that multicast any strip: every gather of theirs is wrapped in the transfer
    // token, and every chunk of theirs has a writer prefix, empty or not (module doc).
    let mut senders = vec![false; units];
    let mut totals = vec![[0u32; WORDS]; units];
    let max = tt_isa::noc::niu::MAX_REQUEST_BYTES;
    for (operand, holder_of) in [
        (Operand::A, &shared.holder_of),
        (Operand::B, &shared.b_holder_of),
    ] {
        if holder_of.iter().all(Option::is_none) {
            continue;
        }
        let (ready, freed) = operand.words();
        let gathers = reuse::gathers(jobs, units, first, operand)?;
        // Each sender's strips sent, and each strip's rank among them (0: not sent).
        let mut sent = vec![0u32; units];
        let mut rank: Vec<Vec<u32>> = gathers.iter().map(|g| vec![0; g.len()]).collect();
        let (readers, paired) = reuse::pair(&gathers, holder_of, operand)?;
        let bit = match operand {
            Operand::A => 1,
            Operand::B => 2,
        };
        let skew = 16 * u32::from(MISROUTE.load(std::sync::atomic::Ordering::Relaxed) & bit != 0);
        let skip_reads = SKIP_SENDER_READS.load(std::sync::atomic::Ordering::Relaxed) & bit != 0;
        // Each sender gather's block: its job's ordinal among the sender's jobs.
        let blocks: Vec<Vec<usize>> = gathers
            .iter()
            .map(|g| {
                let mut ordinal = 0;
                let mut last = None;
                g.iter()
                    .map(|g| {
                        if last.is_some_and(|j| j != g.job) {
                            ordinal += 1;
                        }
                        last = Some(g.job);
                        ordinal
                    })
                    .collect()
            })
            .collect();
        for h in 0..units {
            if holder_of[h].is_some() {
                continue;
            }
            // Frees counted by the parity of the strip's block among the sender's; each strip
            // with readers ranked among the sender's sends, from 1.
            let mut granted = [0u32; 2];
            let (sx, sy) = tiles[h];
            for (i, g) in gathers[h].iter().enumerate() {
                let to = &readers[h][i];
                if to.is_empty() {
                    continue;
                }
                let (from, bytes) = (g.slots, g.bytes);
                let q = blocks[h][i] % 2;
                granted[q] += to.len() as u32;
                let nc = sends.entry((g.job, g.step)).or_default();
                nc.push([
                    op::WAIT_CREDIT,
                    credit_word(freed[q]),
                    granted[q],
                    0,
                    0,
                    0,
                    0,
                    0,
                ]);
                let receivers: Vec<(u8, u8)> = to.iter().map(|&r| tiles[r]).collect();
                for (start, end) in crate::residency::rectangles(&receivers) {
                    for off in (0..bytes).step_by(max as usize) {
                        let at = from + off;
                        nc.push([
                            op::WRITE_L1_MCAST,
                            dm::mcast_corners(start, end),
                            0,
                            at + skew,
                            at,
                            (bytes - off).min(max),
                            dm::MCAST_RUN_ACKS,
                            0,
                        ]);
                    }
                }
                // The rank in the sender's own ready word (the credit settles the multicasts
                // first, so it follows the data), then that word into each reader's.
                rank[h][i] = sent[h] + 1;
                sent[h] += 1;
                nc.push([
                    op::CREDIT,
                    sx as u32,
                    sy as u32,
                    credit_word(ready),
                    1,
                    0,
                    0,
                    0,
                ]);
                for (start, end) in crate::residency::rectangles(&receivers) {
                    nc.push([
                        op::WRITE_L1_MCAST,
                        dm::mcast_corners(start, end),
                        0,
                        credit_word(ready),
                        credit_word(ready),
                        16,
                        0,
                        0,
                    ]);
                }
                senders[h] = true;
                if skip_reads {
                    edits.entry((g.job, g.step, g.at)).or_default().resident |= operand.resident();
                }
                STRIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            totals[h][freed[0] as usize] = granted[0];
            totals[h][freed[1] as usize] = granted[1];
            totals[h][ready as usize] = sent[h];
        }
        for r in 0..units {
            let Some(h) = holder_of[r] else {
                continue;
            };
            let (hx, hy) = tiles[h];
            // Its strips, each with its index among its sender's.
            let got: Vec<(usize, usize)> = paired[r].iter().copied().enumerate().collect();
            // A reader never sits out two of its sender's blocks in a row, so it never frees a
            // strip two blocks past the last it received: each parity's count is exact (module
            // doc).
            let own: Vec<usize> = got.iter().map(|&(_, i)| blocks[h][i]).collect();
            if let Some(w) = own.windows(2).find(|w| w[1] > w[0] + 2) {
                return Err(TensorError::Shape(format!(
                    "operand multicast: unit {r} sits out its sender's blocks {} to {} of \
                     {operand:?}",
                    w[0] + 1,
                    w[1] - 1
                )));
            }
            if own.first().is_some_and(|&b| b > 1) {
                return Err(TensorError::Shape(format!(
                    "operand multicast: unit {r} sits out its sender's first blocks of \
                     {operand:?}"
                )));
            }
            let mut last = 0;
            for (k, (&(n, i), &b)) in got.iter().zip(&own).enumerate() {
                let g = &gathers[r][n];
                let edit = edits.entry((g.job, g.step, g.at)).or_default();
                edit.before.push([
                    op::CREDIT,
                    hx as u32,
                    hy as u32,
                    credit_word(freed[b % 2]),
                    1,
                    0,
                    0,
                    0,
                ]);
                let over =
                    k + 1 == got.len() && OVERWAIT.load(std::sync::atomic::Ordering::Relaxed);
                last = rank[h][i];
                edit.after.push([
                    op::WAIT_CREDIT,
                    credit_word(ready),
                    rank[h][i] + u32::from(over),
                    0,
                    0,
                    0,
                    0,
                    0,
                ]);
            }
            // What it receives it does not read.
            for g in &gathers[r] {
                edits.entry((g.job, g.step, g.at)).or_default().resident |= operand.resident();
            }
            // Its ready word ends at the rank of the last strip it received.
            totals[r][ready as usize] = last;
        }
    }
    // Each of a sender's gathers inside the transfer token: reserved before (the stream two deep,
    // so NC is done with the half it multicast two chunks back), pushed after (the push waits for
    // the gather's reads). A chunk it sends nothing from takes the token too: its gather may land
    // on a half NC is still multicasting from.
    for (j, job) in jobs.iter().enumerate() {
        if !senders[(first + j) % units] {
            continue;
        }
        for (s, step) in job.iter().enumerate() {
            let Step::List { entries, .. } = step else {
                continue;
            };
            if !matches!(job.get(s + 1), Some(Step::Kernel { .. })) {
                continue;
            }
            let records: Vec<usize> = (0..entries.len())
                .filter(|&at| entries[at][0] == record::GATHER)
                .collect();
            let &[at] = &records[..] else {
                return Err(TensorError::Shape(format!(
                    "operand multicast: a sender's chunk with {} gathers",
                    records.len()
                )));
            };
            let edit = edits.entry((j, s, at)).or_default();
            edit.before.insert(0, transfer(Action::Reserve));
            edit.after.insert(0, transfer(Action::Push));
            sends.entry((j, s)).or_default();
        }
    }
    reuse::apply(jobs, edits);
    // Each chunk's sends at the head of its writer list, inside the token.
    for ((j, s), entries) in sends {
        let scatter = (s + 1..jobs[j].len())
            .find(|&i| matches!(jobs[j][i], Step::List { .. }))
            .filter(|&i| matches!(jobs[j][i - 1], Step::Kernel { .. }))
            .ok_or_else(|| {
                TensorError::Shape("operand multicast: a chunk without a writer list".into())
            })?;
        let Step::List { entries: list, .. } = &mut jobs[j][scatter] else {
            unreachable!("found as a list");
        };
        let head = std::iter::once(transfer(Action::Wait))
            .chain(entries)
            .chain(std::iter::once(transfer(Action::Pop)));
        list.splice(0..0, head);
    }
    reuse::restore(jobs, units, first, tiles, &totals)
}

#[cfg(test)]
mod tests {
    use super::super::distribute::{grid, grid_of};
    use super::super::reuse::Operand;
    use crate::matmul::{Fidelity, SrcRoute};
    use crate::tensor::{self, BlockStream, DramAlloc, DramTensor, Elem, Reads, Reuse, Step};
    use tt_isa::dataflow::{Action, Stream};
    use tt_isa::dm::{credit_word, op, record, CREDIT_WORDS};
    use tt_isa::dram::Dram;

    /// Held by every test that builds a product: the negative controls' hooks are process-wide
    /// ([`super::probe_misroute`], [`super::probe_overwait`]).
    static PROBES: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn probes() -> std::sync::MutexGuard<'static, ()> {
        PROBES.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `[m, k, n]` on `units` tiles in rows of `width`, blocks `[mb, kq, nb]`, shared with
    /// `reuse`: the work and the tiles.
    fn product(
        [m, k, n]: [usize; 3],
        units: usize,
        width: usize,
        block: [usize; 3],
        reuse: Reuse,
    ) -> (Vec<tensor::Job>, Vec<(u8, u8)>) {
        let mut alloc = DramAlloc::new(&Dram::FULL);
        let a = DramTensor::alloc_elem(&mut alloc, m, k, Elem::F32).unwrap();
        let b = DramTensor::alloc_elem(&mut alloc, k, n, Elem::F32).unwrap();
        // Columns 1..=7 then 10.., as a card's grid splits at 8 and 9.
        let column = |c: usize| if c < 7 { 1 + c } else { 3 + c } as u8;
        let tiles: Vec<(u8, u8)> = (0..units)
            .map(|u| (column(u % width), 2 + (u / width) as u8))
            .collect();
        let reads = Reads {
            thin: true,
            resident_a: true,
            unpack_b: true,
            packed_a: None,
            stream_k: true,
            stream_prefetch: true,
            prefetch_units: [crate::session::PREFETCH_UNITS; 2],
            block_stream: BlockStream::Fixed(block),
            reuse,
            out: crate::matmul::Out::Fp32,
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
            None,
        )
        .unwrap();
        (work.jobs, tiles)
    }

    /// Set while a test builds a product's jobs as the deal lays them out, before [`super::push`]
    /// rewrites them: what each kernel must find in its slots.
    pub(super) static SKIP_PUSH: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// The tiles the `GATHER` at `e[at..]` reads, as `(L1 address, what lands there)`: each
    /// operand's tile named by the operand, its tensor reference, and its row (`A`) or column
    /// (`B`) and `K` tile, unless the record leaves that operand resident.
    fn tiles_read(e: &[[u32; 8]], at: usize, all: bool) -> Vec<(u32, Vec<u32>)> {
        let h = e[at];
        let flags = h[1];
        let kt = record::gather_kt(flags);
        let mut out = Vec::new();
        for (x, operand) in [Operand::A, Operand::B].into_iter().enumerate() {
            if !all && flags & operand.resident() != 0 {
                continue;
            }
            let slot = operand.slot(flags).unwrap() as u32;
            let (_, reference) = operand.fields();
            let mut name = vec![x as u32];
            name.extend(e[at + reference]);
            name.extend(e[at + reference + 1]);
            match operand {
                Operand::A => {
                    for i in 0..h[5] {
                        for kk in 0..kt {
                            let mut v = name.clone();
                            v.extend([h[4] + i, kk]);
                            out.push((h[2] + (i * kt + kk) * slot, v));
                        }
                    }
                }
                Operand::B => {
                    // By column where its reference asks for that order in L1.
                    let by_column = e[at + reference][4] & record::REF_L1_BY_COLUMN != 0;
                    for kk in 0..kt {
                        for j in 0..h[7] {
                            let mut v = name.clone();
                            v.extend([h[6] + j, kk]);
                            let index = if by_column {
                                j * kt + kk
                            } else {
                                kk * h[7] + j
                            };
                            out.push((h[3] + index * slot, v));
                        }
                    }
                }
            }
        }
        out
    }

    /// What each unit's kernels read, chunk by chunk, from the jobs before the pass.
    type Expected = Vec<Vec<Vec<(u32, Vec<u32>)>>>;

    fn expected(jobs: &[tensor::Job], units: usize) -> Expected {
        (0..units)
            .map(|u| {
                jobs.iter()
                    .skip(u)
                    .step_by(units)
                    .flatten()
                    .filter_map(|s| match s {
                        Step::List { entries, .. } => entries
                            .iter()
                            .position(|e| e[0] == record::GATHER)
                            .map(|at| tiles_read(entries, at, true)),
                        _ => None,
                    })
                    .collect()
            })
            .collect()
    }

    /// One unit's actors: RISCV B's list (the gathers inside their input credits), its kernel's
    /// chunks (what each must find in its slots), and RISCV NC's list (the writer sections, a
    /// sender's prefix a chunk ahead, [`super::writer_list`]), as `session::streaming_segments` lays them out.
    struct Unit {
        b: Vec<[u32; 8]>,
        chunks: Vec<Vec<(u32, Vec<u32>)>>,
        nc: Vec<[u32; 8]>,
    }

    fn units_of(jobs: &[tensor::Job], tiles: &[(u8, u8)], expect: &Expected) -> Vec<Unit> {
        let n = tiles.len();
        let cb = |s: Stream, a: Action| crate::channel::Channel::local(0, s, 2).credit(a);
        (0..n)
            .map(|u| {
                let mut unit = Unit {
                    b: Vec::new(),
                    chunks: expect[u].clone(),
                    nc: Vec::new(),
                };
                for job in jobs.iter().skip(u).step_by(n) {
                    let mut sections = Vec::new();
                    let mut steps = job.iter();
                    while let Some(step) = steps.next() {
                        let Step::List {
                            entries: gather, ..
                        } = step
                        else {
                            panic!("a block opens with its gather list")
                        };
                        assert!(matches!(steps.next(), Some(Step::Kernel { .. })));
                        let Some(Step::List {
                            entries: scatter, ..
                        }) = steps.next()
                        else {
                            panic!("a block closes with its writer list")
                        };
                        unit.b.push(cb(Stream::Input, Action::Reserve));
                        unit.b.extend(gather);
                        unit.b.push(cb(Stream::Input, Action::Push));
                        sections.push(scatter.clone());
                    }
                    // A job taken as one ownership region (`session::streaming_segments`; a
                    // longer job splits into regions, and NC sends no further ahead than its
                    // region's end, as the layout before sending ahead did everywhere).
                    unit.nc.extend(super::writer_list(
                        sections,
                        cb(Stream::Output, Action::Wait),
                        cb(Stream::Output, Action::Pop),
                    ));
                }
                assert_eq!(unit.chunks.len(), {
                    let b = &unit.b;
                    b.iter().filter(|e| e[0] == record::GATHER).count()
                });
                unit
            })
            .collect()
    }

    /// Run every unit's three actors interleaved at random by `seed`: credits add to their
    /// tile's words, a credit wait holds until its word reaches the target, the streams hold two
    /// batches, a gather writes each tile it reads, a multicast copies the sender's bytes as they
    /// are when it runs into every tile of its rectangle (a credit word's value into a credit
    /// word), and a kernel checks that its slots hold every tile `expect` says the unpassed
    /// product's gather read there. Nothing may stall; every word ends at zero.
    fn run(
        jobs: &[tensor::Job],
        tiles: &[(u8, u8)],
        expect: &Expected,
        seed: u64,
    ) -> Result<(), String> {
        let units = units_of(jobs, tiles, expect);
        let n = units.len();
        let unit_of = |x: u32, y: u32| {
            tiles
                .iter()
                .position(|&t| t == (x as u8, y as u8))
                .expect("a credit or multicast to one of the op's tiles")
        };
        let word = |addr: u32| {
            (0..CREDIT_WORDS)
                .position(|i| credit_word(i) == addr)
                .unwrap()
        };
        let mut words = vec![[0u32; CREDIT_WORDS as usize]; n];
        let mut slots: Vec<std::collections::BTreeMap<u32, Vec<u32>>> = vec![Default::default(); n];
        // Per unit, each stream's received and consumed batches: input, output, transfer.
        let mut streams = vec![[[0u32; 2]; 3]; n];
        let index = |s: u32| match s {
            0 => 0,
            1 => 1,
            _ => 2,
        };
        let mut pc = vec![[0usize; 3]; n];
        let mut rng = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        loop {
            // Each actor that can take its next step: `(unit, 0 B | 1 kernel | 2 NC)`.
            let mut runnable = Vec::new();
            for (u, unit) in units.iter().enumerate() {
                let ready = |e: &[u32; 8], st: &[[u32; 2]; 3], w: &[u32]| match e[0] {
                    op::WAIT_CREDIT => (w[word(e[1])].wrapping_sub(e[2]) as i32) >= 0,
                    op::CB => {
                        let [rcv, cons] = st[index(e[1])];
                        match e[2] {
                            a if a == Action::Reserve.word() => rcv - cons < 2,
                            a if a == Action::Wait.word() => rcv - cons >= 1,
                            _ => true,
                        }
                    }
                    _ => true,
                };
                if let Some(e) = unit.b.get(pc[u][0]) {
                    if ready(e, &streams[u], &words[u]) {
                        runnable.push((u, 0));
                    }
                }
                if pc[u][1] < unit.chunks.len() {
                    let [[ir, ic], [or, oc], _] = streams[u];
                    if ir - ic >= 1 && or - oc < 2 {
                        runnable.push((u, 1));
                    }
                }
                if let Some(e) = unit.nc.get(pc[u][2]) {
                    if ready(e, &streams[u], &words[u]) {
                        runnable.push((u, 2));
                    }
                }
            }
            if runnable.is_empty() {
                for (u, unit) in units.iter().enumerate() {
                    if pc[u][0] < unit.b.len() {
                        return Err(format!("stalled: unit {u}'s B at {:?}", unit.b[pc[u][0]]));
                    }
                    if pc[u][2] < unit.nc.len() {
                        return Err(format!("stalled: unit {u}'s NC at {:?}", unit.nc[pc[u][2]]));
                    }
                    if pc[u][1] < unit.chunks.len() {
                        return Err(format!("stalled: unit {u}'s kernel at {}", pc[u][1]));
                    }
                }
                break;
            }
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let (u, actor) = runnable[(rng >> 33) as usize % runnable.len()];
            if actor == 1 {
                let c = pc[u][1];
                for (at, s) in &units[u].chunks[c] {
                    if slots[u].get(at) != Some(s) {
                        return Err(format!(
                            "unit {u}'s chunk {c}: the slot at {at:#x} holds {:?}, not {s:?}",
                            slots[u].get(at)
                        ));
                    }
                }
                streams[u][0][1] += 1;
                streams[u][1][0] += 1;
                pc[u][1] += 1;
                continue;
            }
            let list = if actor == 0 {
                &units[u].b
            } else {
                &units[u].nc
            };
            let p = pc[u][actor];
            let e = list[p];
            match e[0] {
                op::CREDIT => {
                    let w = &mut words[unit_of(e[1], e[2])][word(e[3])];
                    *w = w.wrapping_add(e[4]);
                }
                op::CB => {
                    let s = &mut streams[u][index(e[1])];
                    match e[2] {
                        a if a == Action::Push.word() => s[0] += 1,
                        a if a == Action::Pop.word() => s[1] += 1,
                        _ => {}
                    }
                }
                op::WRITE_L1_MCAST if (0..CREDIT_WORDS).any(|i| credit_word(i) == e[3]) => {
                    let [x0, y0, x1, y1] = e[1].to_le_bytes();
                    let value = words[u][word(e[4])];
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            words[unit_of(x as u32, y as u32)][word(e[3])] = value;
                        }
                    }
                }
                op::WRITE_L1_MCAST => {
                    let [x0, y0, x1, y1] = e[1].to_le_bytes();
                    let (to, from, len) = (e[3], e[4], e[5]);
                    let got: Vec<(u32, Vec<u32>)> = slots[u]
                        .range(from..from + len)
                        .map(|(&a, v)| (a, v.clone()))
                        .collect();
                    for y in y0..=y1 {
                        for x in x0..=x1 {
                            let t = unit_of(x as u32, y as u32);
                            assert_ne!(t, u, "a sender multicasts into its own rectangle");
                            for (a, v) in &got {
                                slots[t].insert(a - from + to, v.clone());
                            }
                        }
                    }
                }
                record::GATHER => {
                    for (a, v) in tiles_read(list, p, false) {
                        slots[u].insert(a, v);
                    }
                }
                _ => {}
            }
            pc[u][actor] += match e[0] {
                record::GATHER => super::super::reuse::GATHER_ENTRIES,
                h => record::len(h).max(1),
            };
        }
        match words.iter().position(|w| w.iter().any(|&x| x != 0)) {
            Some(u) => Err(format!("unit {u}'s words end at {:?}", words[u])),
            None => Ok(()),
        }
    }

    /// The product before the pass (the deal's jobs: what its kernels read) and after it.
    fn shared_product(
        shape: [usize; 3],
        units: usize,
        width: usize,
        block: [usize; 3],
        reuse: Reuse,
    ) -> (Vec<tensor::Job>, Vec<(u8, u8)>, Expected) {
        SKIP_PUSH.store(true, std::sync::atomic::Ordering::Relaxed);
        let (unpassed, _) = product(shape, units, width, block, reuse);
        SKIP_PUSH.store(false, std::sync::atomic::Ordering::Relaxed);
        let (jobs, tiles) = product(shape, units, width, block, reuse);
        let expect = expected(&unpassed, units);
        (jobs, tiles, expect)
    }

    /// Shape, units, grid width, block, and which operands are multicast.
    #[allow(clippy::type_complexity)]
    const CASES: [([usize; 3], usize, usize, [usize; 3], bool, bool); 8] = [
        // 2 x 4: two block rows a grid row, column blocks 2, 2, 1, 1 a unit.
        ([256, 256, 576], 8, 4, [2, 2, 3], true, true),
        ([256, 256, 576], 8, 4, [2, 2, 3], true, false),
        ([256, 256, 576], 8, 4, [2, 2, 3], false, true),
        // 3 x 3 with a unit past the grid; a grid row split at the card's middle columns.
        ([200, 256, 600], 10, 3, [2, 2, 3], true, true),
        ([256, 256, 576], 18, 9, [2, 2, 1], true, true),
        // More grid rows than block rows: the rows past them run nothing.
        ([128, 256, 576], 12, 3, [2, 2, 3], true, true),
        // H1's product on 8 x 12 and 10 x 12.
        ([2048, 2048, 16384], 96, 12, [8, 2, 8], true, true),
        ([2048, 2048, 16384], 120, 12, [4, 2, 8], true, true),
    ];

    /// Every block runs once; a reader of `A` is in its sender's grid row and, in every round it
    /// has a block, its sender has one of the same block row; of `B`, likewise down a column.
    #[test]
    fn a_grid_deal_runs_every_block_once_and_pairs_rows_and_columns() {
        for (rows, cols, width, height, units) in [
            (8, 64, 12, 10, 120),
            (16, 64, 12, 10, 120),
            (8, 64, 8, 8, 64),
            (3, 5, 4, 2, 9),
        ] {
            let tiles: Vec<(u8, u8)> = (0..units)
                .map(|u| ((1 + u % width) as u8, (2 + u / width) as u8))
                .collect();
            assert_eq!(grid_of(&tiles, units), (width, units / width));
            let s = grid(rows, cols, &tiles, units, true, true).unwrap();
            // No unit sends both operands where another choice exists: more than one unit with
            // every round of a row, or more than one grid row with every block row.
            let choice = cols % width == 0 || rows % rows.min(height) == 0;
            let a_senders: std::collections::BTreeSet<usize> =
                s.holder_of.iter().flatten().copied().collect();
            let b_senders: std::collections::BTreeSet<usize> =
                s.b_holder_of.iter().flatten().copied().collect();
            assert!(
                !choice || a_senders.is_disjoint(&b_senders),
                "{rows}x{cols} on {width}x{height}: a unit sends both"
            );
            let mut seen = vec![0; rows * cols];
            for b in s.order.iter().flatten() {
                seen[*b] += 1;
            }
            assert!(
                seen.iter().all(|&n| n == 1),
                "{rows}x{cols} on {width}x{height}"
            );
            for u in 0..units {
                let rounds = (u..s.order.len()).step_by(units);
                if let Some(h) = s.holder_of[u] {
                    assert_eq!(h / width, u / width, "A's sender in the reader's grid row");
                    for t in rounds.clone() {
                        if let Some(b) = s.order[t] {
                            let held = s.order[t - u + h];
                            assert_eq!(held.map(|x| x / cols), Some(b / cols));
                        }
                    }
                }
                if let Some(h) = s.b_holder_of[u] {
                    assert_eq!(
                        h % width,
                        u % width,
                        "B's sender in the reader's grid column"
                    );
                    for t in rounds {
                        if let Some(b) = s.order[t] {
                            let held = s.order[t - u + h];
                            assert_eq!(held.map(|x| x % cols), Some(b % cols));
                        }
                    }
                }
            }
        }
    }

    /// The multicast product's strips land what its gathers would have read, before each
    /// kernel reads them, under any interleaving of every unit's B, kernel and NC, with no stall
    /// and every word back at zero; its kernels are the unshared product's, word for word; the
    /// readers of each multicast operand read none of it from GDDR.
    #[test]
    fn multicasts_land_what_the_gathers_read_under_any_interleaving() {
        let _probes = probes();
        for (shape, units, width, block, a, b) in CASES {
            let (plain, _) = product(shape, units, width, block, Reuse::Off);
            let (shared, tiles, expect) =
                shared_product(shape, units, width, block, Reuse::Multicast { a, b });
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
            assert_eq!(kernels(&plain), kernels(&shared), "{shape:?} {width}");
            let count = |flag: u32| {
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
            assert_eq!(
                count(record::GATHER_A_RESIDENT) > 0,
                a,
                "{shape:?}: A as asked"
            );
            assert_eq!(
                count(record::GATHER_B_RESIDENT) > 0,
                b,
                "{shape:?}: B as asked"
            );
            let seeds = if units > 24 { 1 } else { 48 };
            for seed in 0..seeds {
                run(&shared, &tiles, &expect, seed)
                    .unwrap_or_else(|e| panic!("{shape:?} on {units}/{width} seed {seed}: {e}"));
            }
        }
    }

    /// Drop every entry `drop` matches from the multicast product; some interleaving must then
    /// fail.
    fn caught_without(reuse: Reuse, drop: impl Fn(&[u32; 8]) -> bool, what: &str) {
        let _probes = probes();
        // One grid row of eight multicasting `A`, or one column of eight `B`, sixteen chunks a
        // block: room for a sender to run two chunks ahead of a reader.
        let (mut jobs, tiles, expect) = match reuse {
            Reuse::Multicast { b: false, .. } => {
                shared_product([64, 1024, 576], 8, 8, [2, 2, 3], reuse)
            }
            _ => shared_product([576, 1024, 64], 8, 1, [2, 2, 2], reuse),
        };
        let mut dropped = 0;
        for step in jobs.iter_mut().flatten() {
            if let Step::List { entries, .. } = step {
                let before = entries.len();
                entries.retain(|e| !drop(e));
                dropped += before - entries.len();
            }
        }
        assert!(dropped > 0, "{what}: nothing to drop");
        assert!(
            (0..1024).any(|seed| run(&jobs, &tiles, &expect, seed).is_err()),
            "{what}: no interleaving caught it"
        );
    }

    /// The negative controls: a sender that multicasts without waiting for its readers' frees,
    /// a sender's B that gathers into a half its NC may still be multicasting, a reader that
    /// does not wait for its strip.
    #[test]
    fn a_missing_wait_is_caught() {
        let freed = [
            credit_word(super::super::reuse::FREED[0]),
            credit_word(super::super::reuse::FREED[1]),
            credit_word(super::super::reuse::FREED_B[0]),
            credit_word(super::super::reuse::FREED_B[1]),
        ];
        let ready = [
            credit_word(super::super::reuse::READY),
            credit_word(super::super::reuse::READY_B),
        ];
        // Each guard alone, on an operand multicast without the other (with both, each sender
        // also waits for the other operand's strips, which can hide one missing guard): the
        // freed waits and the ready waits for either operand; the transfer reserves for `B`'s
        // column, where a sender's B runs ahead of its NC soonest (along a row split by chunk
        // parity, a sender's B also waits for the other sender's strips, so its NC must stall
        // far longer than a uniform interleaving makes likely; the reserve guards it all the
        // same).
        let (a, b) = (
            Reuse::Multicast { a: true, b: false },
            Reuse::Multicast { a: false, b: true },
        );
        let freed_wait = |e: &[u32; 8]| {
            e[0] == op::WAIT_CREDIT && freed.contains(&e[1]) && e[3] == 0 && e[2] > 0
        };
        let reserve = |e: &[u32; 8]| {
            e[0] == op::CB && e[1] == Stream::Transfer.word() && e[2] == Action::Reserve.word()
        };
        let ready_wait = |e: &[u32; 8]| e[0] == op::WAIT_CREDIT && ready.contains(&e[1]);
        caught_without(a, freed_wait, "A: the senders' freed waits");
        caught_without(b, freed_wait, "B: the senders' freed waits");
        caught_without(b, reserve, "B: the senders' transfer reserves");
        caught_without(a, ready_wait, "A: the readers' ready waits");
        caught_without(b, ready_wait, "B: the readers' ready waits");
    }

    /// A misrouted multicast (16 bytes past the strip) leaves a reader's slots without its
    /// strip; a reader that waits for a credit too many stalls.
    #[test]
    fn a_misroute_and_an_overwait_are_caught() {
        let _probes = probes();
        let reuse = Reuse::Multicast { a: true, b: true };
        for operand in [0, 1] {
            super::probe_misroute(operand, true);
            let (jobs, tiles, expect) = shared_product([256, 256, 576], 8, 4, [2, 2, 3], reuse);
            super::probe_misroute(operand, false);
            assert!(
                run(&jobs, &tiles, &expect, 0).is_err(),
                "a misrouted operand {operand}"
            );
        }
        super::probe_overwait(true);
        let (jobs, tiles, expect) = shared_product([256, 256, 576], 8, 4, [2, 2, 3], reuse);
        super::probe_overwait(false);
        let e = run(&jobs, &tiles, &expect, 0).unwrap_err();
        assert!(e.contains("stalled"), "{e}");
    }

    /// NC's list sends each chunk's strip before it waits for the chunk before: the prefix of
    /// chunk `c + 1` sits between chunk `c - 1`'s output pop and chunk `c`'s output wait, and a
    /// section with no prefix (an unshared chunk) keeps its place.
    #[test]
    fn a_sender_sends_a_chunk_ahead_of_its_results() {
        let entry = |k: u32| [op::WRITE_L1, k, 0, 0, 0, 0, 0, 0];
        let (wait, pop) = (entry(100), entry(101));
        let tp = super::transfer(Action::Pop);
        let section = |c: u32, sends: bool| {
            let mut s = Vec::new();
            if sends {
                s.extend([entry(10 * c + 1), tp]);
            }
            s.push(entry(10 * c + 5));
            s
        };
        let list = super::writer_list(
            vec![
                section(1, true),
                section(2, true),
                section(3, false),
                section(4, true),
            ],
            wait,
            pop,
        );
        #[rustfmt::skip]
        let expect = vec![
            entry(11), tp, entry(21), tp, wait, entry(15), pop,
            wait, entry(25), pop,
            entry(41), tp, wait, entry(35), pop,
            wait, entry(45), pop,
        ];
        assert_eq!(list, expect);
    }
}
