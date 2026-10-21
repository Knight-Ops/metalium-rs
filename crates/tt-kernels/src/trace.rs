//! Traces: a stretch of device ops captured once and replayed without the host
//! (`hardware-coverage.md` X4d; tt-metal's `BeginTraceCapture`/`ReplayTrace`
//! the model).
//!
//! `Session::begin_trace` ... `Session::end_trace` around any ops captures what
//! the session enqueued on every unit -- and runs it, so the capture is also
//! the first run. `Session::replay` runs it again: one list per unit, a single
//! `CALL` entry (`tt_isa::dm::op::CALL`), and each data mover streams the
//! captured entries from GDDR. Between replays, `Session::write` puts new
//! values into the tensors the trace reads (an inference's input), and the
//! tensors it wrote hold the results.
//!
//! What a replay cannot change, the capture makes position-independent: each
//! kernel's generation and each barrier's target are offsets the `CALL`
//! supplies; the roles' descriptors are `POKE` entries in the stream, since
//! the host writes none during a replay; and the semaphore setups the host ran
//! between kernels are kernels of their own in it.
//!
//! And what tt-metal leaves to its users is enforced here (each a
//! [`TraceError`], never a silent corruption):
//!
//! * a trace holds every allocation that existed when its capture ended: a
//!   free of one is deferred until the trace is released, so nothing else
//!   lands in slots it reads or writes. (Precise without tracking each op's
//!   tensors: frees during the capture are deferred too, so anything
//!   allocated after it lies in what was free then -- [`FreeSnapshot`].)
//! * the programs its kernels name stay in the program cache while it lives;
//! * a tile reset or a mover restart since the capture makes it
//!   [`TraceError::Stale`];
//! * a download, a [`Session::write`] or a host-run kernel during a capture is
//!   [`TraceError::HostTransfer`]: a replay cannot repeat host work between
//!   device ops. (An upload is allowed: a constant, which the replay reads
//!   where the capture left it.)
//! * the tile state the host keeps track of -- descriptors, semaphores -- is
//!   forgotten at a capture's start, so the capture records everything its
//!   kernels need, and after a replay, so nothing queued later trusts it.
//!
//! The capture keeps each op's record ([`OpRecord`]) beside the words: the
//! seam for optimizing a captured graph later (X4e).
//!
//! [`Session::write`]: crate::session::Session::write
//! [`FreeSnapshot`]: crate::tensor::FreeSnapshot

use tt_isa::dm::{self, op, record};

use tt_isa::mailbox::{self, role::Mailbox, DESCRIPTOR_WORDS};

use crate::tensor::{FreeSnapshot, Placement};

/// Why a trace operation was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceError {
    /// A capture is already open.
    Capturing,
    /// No capture is open.
    NotCapturing,
    /// No trace by this id (never made, or released).
    Unknown(u64),
    /// The tiles have been reset or a mover restarted since the capture: the
    /// programs, descriptors and counters it was made against are gone.
    Stale,
    /// A host transfer during the capture -- `what` -- which a replay could
    /// not repeat.
    HostTransfer(&'static str),
    /// Traces queue on the movers: the session must be batching
    /// (`Session::set_batching`).
    NotBatching,
    /// A kernel whose programs the program cache does not admit: the host
    /// stages those itself, which a replay cannot.
    NotResident,
    /// A capture with nothing in it.
    Empty,
    /// A planned capture that is not its plan's (`Session::begin_trace_planned`): why. Nothing
    /// was kept; what its run wrote to the state is the plain trace's to restore.
    Planned(String),
    /// Captured op `op` (`label`, `what`) planned L1 up to `top` (0: its layout was not planned
    /// while captured, so its need is unknown), past the persistent region's base `base`
    /// (`Session::set_l1_region`).
    IntoRegion {
        op: usize,
        label: &'static str,
        what: &'static str,
        top: u64,
        base: u64,
    },
}

impl core::fmt::Display for TraceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TraceError::Capturing => write!(f, "a trace capture is already open"),
            TraceError::NotCapturing => write!(f, "no trace capture is open"),
            TraceError::Unknown(id) => write!(f, "no trace {id}: never made, or released"),
            TraceError::Stale => write!(
                f,
                "the trace is stale: a tile was reset or a data mover restarted since it was \
                 captured, so the programs and counters it names are gone; capture it again"
            ),
            TraceError::HostTransfer(what) => write!(
                f,
                "a {what} during a trace capture: a replay runs device work only and could \
                 not repeat it; end the capture before reading results"
            ),
            TraceError::NotBatching => write!(
                f,
                "traces run from the data movers' queues: turn batching on (Session::set_batching)"
            ),
            TraceError::NotResident => write!(
                f,
                "a kernel too large for the program cache: the host stages it itself, which \
                 a replay cannot do"
            ),
            TraceError::Empty => write!(f, "the capture ran no device work"),
            TraceError::Planned(why) => write!(f, "the planned capture is not its plan's: {why}"),
            TraceError::IntoRegion {
                op,
                label,
                what,
                top,
                base,
            } => match top {
                0 => write!(
                    f,
                    "captured op {op} ({label}, {what}) did not plan its L1 layout while captured, \
                     so whether it stays below the persistent region at {base:#x} is unknown"
                ),
                _ => write!(
                    f,
                    "captured op {op} ({label}, {what}) plans L1 up to {top:#x}, into the \
                     persistent region at {base:#x}"
                ),
            },
        }
    }
}

/// A trace, by number.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraceId(pub(crate) u64);

impl TraceId {
    /// The number, as the profile mode reports traces by it (`crate::profile_mode`).
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// What one captured op was: the graph a later pass can optimize over.
#[derive(Clone, Debug)]
pub struct OpRecord {
    /// Its first list's name (`tensor::stats`).
    pub what: &'static str,
    /// The caller's name for it when it was queued ([`Session::set_op_label`]);
    /// empty if none was set.
    ///
    /// [`Session::set_op_label`]: crate::session::Session::set_op_label
    pub label: &'static str,
    /// Per unit, the range of that unit's stream its lists took.
    pub entries: Vec<core::ops::Range<usize>>,
    /// Whether a barrier followed it, on the units it ran on.
    pub barrier: bool,
    /// The events it could stamp in the profile mode on its busiest unit, at most:
    /// `[Detail::Ops, Detail::Full]` (`crate::profile_mode`). Set when the trace is stored.
    pub stamps: [u32; 2],
    /// What it moves through GDDR on all its units. Counted when the trace is stored, in the
    /// profile mode only; zero otherwise.
    pub bytes: crate::profile_mode::OpBytes,
    /// The highest `DATA` address its L1 plans reached ([`crate::l1::take_planned_top`]): what it
    /// needs of a unit's arena. Zero where its layout was not planned while it was captured.
    pub l1_top: u64,
    /// The lowest arena end at which it is laid out as it is ([`crate::l1::Planned::need`]): what
    /// a smaller arena must still hold. Below `l1_top` where a plan left room on purpose (a
    /// product's `B` ring against the arena's end); above it where a choice sized from the arena
    /// tested more than it runs (a product's block width, probed with `A` unpacked). Zero where
    /// nothing was planned.
    pub l1_need: u64,
    /// A hash of how its lists were laid out before a planned capture rewrote them: each unit's
    /// steps, their kinds and entry counts. A planned capture's ops must have their plan pass's
    /// (`crate::residency::Applied`).
    pub shape: u64,
}

/// What a stored stream is made of: how many entries it holds, how many of those
/// are chunk padding, and how many records or entries there are of each op word.
/// Counted as the stream is laid out for GDDR (nothing keeps the words on the
/// host), so a size change can be seen entry by entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Composition {
    /// Entries stored, padding included.
    pub entries: u32,
    /// `WAIT`s [`chunked`] added so no record crosses a chunk.
    pub padding: u32,
    /// Records (or single entries) by op word, padding not counted.
    pub heads: std::collections::BTreeMap<u32, u32>,
}

impl Composition {
    /// The stream `entries` as [`chunked`] lays it out into `laid` entries.
    pub(crate) fn of(entries: &[[u32; 8]], laid: usize) -> Self {
        let mut heads = std::collections::BTreeMap::new();
        let mut i = 0;
        while i < entries.len() {
            *heads.entry(entries[i][0]).or_insert(0) += 1;
            i += record::len(entries[i][0]).max(1);
        }
        Composition {
            entries: laid as u32,
            padding: (laid - entries.len()) as u32,
            heads,
        }
    }

    pub(crate) fn add(&mut self, other: &Composition) {
        self.entries += other.entries;
        self.padding += other.padding;
        for (&op, &n) in &other.heads {
            *self.heads.entry(op).or_insert(0) += n;
        }
    }
}

/// What a finished trace costs one unit: the two things that bound a trace
/// (the program cache, 240 KiB of L1, and the stream in GDDR) and what the
/// stream is made of. See [`crate::session::Session::trace_footprint`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UnitFootprint {
    /// The unit's stream and every section its `PAIR_CALL`s name, together.
    pub stream: Composition,
    /// Entries in the unit's own stream, padding included.
    pub top_level_entries: u32,
    /// GDDR bytes of the stream and the sections, as allocated (whole slots).
    pub gddr_bytes: u64,
    /// Distinct programs this trace holds in the unit's program cache, and their bytes.
    pub programs: u32,
    pub program_bytes: u64,
    /// Bytes held in the cache by every trace on the unit, and the cache's size.
    pub cache_held_bytes: u64,
    pub cache_bytes: u64,
}

/// One unit's share of a capture in progress.
#[derive(Default)]
pub(crate) struct UnitCapture {
    /// The captured barriers this unit's stream accounts for so far: those it arrived at, and
    /// those a wait in it stands for (`tt_isa::dm::BARRIER_STANDS_FOR`).
    pub(crate) barriers_accounted: u32,
    /// What the sections so far are made of.
    pub(crate) sections_composition: Composition,
    /// Each packet's reader and writer sections, laid out for streaming
    /// ([`chunked`]), kept on the host until the capture ends: they are stored
    /// with the unit's stream, in one region (`Session::store_stream`).
    pub(crate) pending_sections: Vec<Vec<[u32; 8]>>,
    /// Where each `PAIR_CALL` sits in `stream`, in order: the `i`th names
    /// `pending_sections[2 i]` and `[2 i + 1]`, its offsets filled in when they
    /// are stored.
    pub(crate) pair_calls: Vec<usize>,
    /// The unit's entries, in order, generations and barrier targets relative.
    pub(crate) stream: Vec<[u32; 8]>,
    /// The resident's last generation when the capture began: the stream's
    /// generations are counted from it.
    pub(crate) generation_base: u32,
    /// What the stream has set each role's descriptor to so far.
    pub(crate) descriptors: [Option<[u32; DESCRIPTOR_WORDS]>; 3],
    /// Program-cache holds for the trace, one per hold: the cache's
    /// generation when taken ([`crate::program_cache::ProgramCache::generation`])
    /// and the address.
    pub(crate) held_programs: Vec<(u64, u64)>,
    /// Entries in each reader and writer section stored for the stream's
    /// `PAIR_CALL`s, chunk padding included: reader, then writer, per packet.
    pub(crate) sections: Vec<u32>,
    /// The entries a development probe put at the front of `stream`
    /// (`Session`'s `ProbeBackground`); 0 without one.
    pub(crate) probe_entries: usize,
    /// Each program-cache hold's address and the op that took it (`OpRecord::label` and
    /// `what`), for the space report (`Session`'s `probe_space_report`).
    pub(crate) program_owners: Vec<(u64, &'static str, &'static str)>,
}

impl UnitCapture {
    /// Entries setting role `thread`'s descriptor to `d`, where the stream has not
    /// already: all but the program's address and length, which each `KERNEL` entry
    /// writes itself.
    ///
    /// A descriptor's words are written in runs of consecutive addresses
    /// (`record::POKE_RUN`), not one `op::POKE` each: a run costs the mover one record
    /// and one fence instead of one entry and one fence a word, and the prologue of a
    /// trace was 51 `POKE`s, 7.2 us of every replay
    /// (`docs/learnings/performance-log.md`, "Trace replay"). The words the program's
    /// address and length divide the runs, as the live handshake words
    /// (`generation`, `ack`) do; a run of one word stays a `POKE`, which is the
    /// smaller of the two.
    pub(crate) fn poke_descriptor(&mut self, thread: usize, d: &mailbox::Descriptor) {
        let mb = Mailbox::of(thread as u32);
        // Address order, which `Descriptor::writes` is not in, so a run is consecutive
        // entries of this array.
        let mut words = d.writes(mb);
        words.sort_unstable_by_key(|&(at, _)| at);
        let ours = |at: u64| at != mb.program_len() && at != mb.program_addr();
        let have = self.descriptors[thread];
        let values = words.map(|(_, v)| v);
        let mut k = 0;
        while k < words.len() {
            if !ours(words[k].0) {
                k += 1;
                continue;
            }
            // The longest run of consecutive words this record can carry.
            let mut end = k + 1;
            while end < words.len()
                && ours(words[end].0)
                && words[end].0 == words[end - 1].0 + 4
                && end - k < record::POKE_RUN_MAX
            {
                end += 1;
            }
            if have.is_none_or(|h| h[k..end] != values[k..end]) {
                let at = words[k].0 as u32;
                if end - k == 1 {
                    self.stream.push([op::POKE, at, values[k], 0, 0, 0, 0, 0]);
                } else {
                    self.stream
                        .extend_from_slice(&record::poke_run(at, &values[k..end]));
                }
            }
            k = end;
        }
        self.descriptors[thread] = Some(values);
    }
}

/// A capture in progress.
pub(crate) struct Capture {
    /// The session's epoch ([`TraceError::Stale`]) when it began.
    pub(crate) epoch: u64,
    /// The session's barrier count when it began.
    pub(crate) barriers_base: u32,
    /// Arrivals at the barriers captured: each barrier's units, the ones its op ran on.
    pub(crate) barriers: u32,
    /// The barriers captured, one an op that ended in one.
    pub(crate) barrier_ops: u32,
    pub(crate) units: Vec<UnitCapture>,
    /// Frees during the capture, deferred to the trace's release.
    pub(crate) freed: Vec<Placement>,
    pub(crate) ops: Vec<OpRecord>,
    /// An op failed part-way through being captured.
    pub(crate) failed: bool,
    /// The last weight-streaming product's barrier target (relative, as every captured target
    /// is) and each unit's stream length after it: a unit whose stream has not grown since ran
    /// nothing after it, so a list appended to it runs as soon as the product is done
    /// (`Session::matmul_dram`'s weight prefetch).
    pub(crate) weight_anchor: Option<(u32, Vec<usize>)>,
    /// The units that product ran on: the rest are where the next one's weights stream.
    pub(crate) weight_units: usize,
    /// Where the persistent region begins, if the capture keeps one (`Session::set_l1_region`):
    /// every op it holds planned below it ([`crate::l1::arena`]).
    pub(crate) region_base: Option<u64>,
    /// The residency plan it applies (`Session::begin_trace_planned`), while it does.
    pub(crate) applied: Option<std::sync::Arc<crate::residency::Applied>>,
    /// Why the plan stopped applying, if it did: the capture is then refused.
    pub(crate) plan_mismatch: Option<String>,
    /// Whether it began planned: it ends verified against its plan, or refused.
    pub(crate) planned: bool,
    /// Whether it is a residency plan pass (`Session::begin_trace_plan_pass`): it ends by planning
    /// the next planned capture.
    pub(crate) plan_pass: bool,
    /// The key its plan is cached under (`Session::begin_trace_planned_as`,
    /// `Session::begin_trace_cached`): kept there if the capture is its plan's, forgotten if not.
    pub(crate) plan_key: Option<u64>,
    /// While it applies a plan: its tensors' GDDR slots so far, grown as it allocates.
    pub(crate) index: crate::residency::Index,
}

impl Capture {
    /// Move unit `u`'s entries `block` of its stream to start at `to`, earlier (`to <=
    /// block.start`: the entries from `to` move up behind it) or later (`to >= block.end`, `to`
    /// counted before the move: the block ends where `to` was, the entries between move down).
    /// Every op's range on `u` and every `PAIR_CALL`'s position follow their entries; a range
    /// that held the block keeps the rest of its entries, which are contiguous again. The block
    /// is no op's own afterwards: a prefetch moved to before the ops it runs under, or the
    /// probe's background moved after them.
    pub(crate) fn move_block(&mut self, u: usize, block: std::ops::Range<usize>, to: usize) {
        let len = block.len();
        assert!(
            to <= block.start || to >= block.end,
            "a block moves out of its own place"
        );
        if len == 0 || to == block.start || to == block.end {
            return;
        }
        debug_assert!(
            self.ops
                .iter()
                .all(|op| !(op.entries[u].start < to && to < op.entries[u].end)),
            "a block moves to the middle of an op"
        );
        let earlier = to < block.start;
        // Where the entry at `i` goes.
        let at = |i: usize| -> usize {
            if block.contains(&i) {
                i - block.start + if earlier { to } else { to - len }
            } else if earlier && (to..block.start).contains(&i) {
                i + len
            } else if !earlier && (block.end..to).contains(&i) {
                i - len
            } else {
                i
            }
        };
        let uc = &mut self.units[u];
        for p in uc.pair_calls.iter_mut() {
            *p = at(*p);
        }
        if earlier {
            uc.stream[to..block.end].rotate_right(len);
        } else {
            uc.stream[block.start..to].rotate_left(len);
        }
        for op in &mut self.ops {
            let r = &mut op.entries[u];
            if r.start >= r.end {
                continue;
            }
            let holds = r.start <= block.start && r.end >= block.end;
            debug_assert!(
                holds || r.end <= block.start || r.start >= block.end,
                "an op's range cuts the block"
            );
            // The range's own entries, the block's left out where it held them.
            let (first, last) = match holds {
                true if r.start == block.start && r.end == block.end => {
                    *r = 0..0;
                    continue;
                }
                true if r.start == block.start => (block.end, r.end - 1),
                true if r.end == block.end => (r.start, block.start - 1),
                _ => (r.start, r.end - 1),
            };
            let (a, b) = (at(first), at(last));
            debug_assert!(b >= a && b - a + 1 == (r.len() - if holds { len } else { 0 }));
            *r = a..b + 1;
        }
    }
}

/// One unit's share of a finished trace.
pub(crate) struct UnitTrace {
    /// Its stream in GDDR, its sections after it in the same region, and where
    /// the stream is: the chip's channel index, the offset, the entry count.
    pub(crate) stream: Placement,
    pub(crate) channel: u32,
    pub(crate) offset: u32,
    pub(crate) count: u32,
    /// Generations a replay takes on the unit.
    pub(crate) generations: u32,
    /// As [`UnitCapture::held_programs`].
    pub(crate) held_programs: Vec<(u64, u64)>,
    /// As [`UnitCapture::sections`].
    pub(crate) sections: Vec<u32>,
    /// What the unit's stream and its sections are made of, and their GDDR bytes.
    pub(crate) composition: Composition,
    pub(crate) top_level_entries: u32,
    pub(crate) gddr_bytes: u64,
    /// Its stream's `BARRIER` entries: `[arrivals, wait-only, the barriers both stand for]`.
    pub(crate) barriers: [u32; 3],
}

/// A finished trace: see the module documentation.
pub(crate) struct Trace {
    pub(crate) epoch: u64,
    pub(crate) units: Vec<Option<UnitTrace>>,
    /// Arrivals a replay adds to the barrier counter ([`Capture::barriers`]).
    pub(crate) barriers: u32,
    /// GDDR's free space when the capture ended: whatever was not free, the
    /// trace holds.
    pub(crate) free_at_end: FreeSnapshot,
    /// Frees of what it holds, deferred to its release.
    pub(crate) freed: Vec<Placement>,
    pub(crate) ops: Vec<OpRecord>,
}

impl Trace {
    /// Must a free of `p` wait for this trace's release?
    pub(crate) fn holds(&self, p: &Placement) -> bool {
        !self.free_at_end.was_free(p)
    }
}

/// Lay a unit's entries out for streaming: a `WAIT` wherever a record would
/// otherwise cross a chunk of [`dm::TRACE_CHUNK_ENTRIES`], since the mover
/// runs a chunk at a time (`dm::op::CALL`).
pub(crate) fn chunked(entries: &[[u32; 8]]) -> Vec<[u32; 8]> {
    let chunk = dm::TRACE_CHUNK_ENTRIES as usize;
    let wait = [op::WAIT, 0, 0, 0, 0, 0, 0, 0];
    let mut out = Vec::with_capacity(entries.len() + entries.len() / chunk + 1);
    let mut i = 0;
    while i < entries.len() {
        let n = record::len(entries[i][0]).max(1);
        if out.len() % chunk + n > chunk {
            while out.len() % chunk != 0 {
                out.push(wait);
            }
        }
        out.extend_from_slice(&entries[i..i + n]);
        i += n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capture of one unit whose entries are numbered (word 1), with `PAIR_CALL`s at `calls`
    /// and one op a range of `ranges`.
    fn numbered(n: usize, calls: &[usize], ranges: &[std::ops::Range<usize>]) -> Capture {
        let stream = (0..n)
            .map(|i| {
                let op = if calls.contains(&i) {
                    op::PAIR_CALL
                } else {
                    op::WAIT
                };
                [op, i as u32, 0, 0, 0, 0, 0, 0]
            })
            .collect();
        let uc = UnitCapture {
            stream,
            pair_calls: calls.to_vec(),
            ..Default::default()
        };
        Capture {
            epoch: 0,
            barriers_base: 0,
            barriers: 0,
            barrier_ops: 0,
            units: vec![uc],
            freed: Vec::new(),
            ops: ranges
                .iter()
                .map(|r| OpRecord {
                    what: "",
                    label: "",
                    entries: vec![r.clone()],
                    barrier: true,
                    stamps: [0; 2],
                    bytes: Default::default(),
                    l1_top: 0,
                    l1_need: 0,
                    shape: 0,
                })
                .collect(),
            failed: false,
            weight_anchor: None,
            weight_units: 0,
            region_base: None,
            applied: None,
            plan_mismatch: None,
            planned: false,
            plan_pass: false,
            plan_key: None,
            index: Default::default(),
        }
    }

    /// Each op's entries (by number) and the numbers at the `PAIR_CALL` positions.
    fn contents(c: &Capture) -> (Vec<Vec<u32>>, Vec<u32>) {
        let s = &c.units[0].stream;
        let ops = c
            .ops
            .iter()
            .map(|o| s[o.entries[0].clone()].iter().map(|e| e[1]).collect())
            .collect();
        let calls = c.units[0]
            .pair_calls
            .iter()
            .map(|&p| {
                assert_eq!(
                    s[p][0],
                    op::PAIR_CALL,
                    "a PAIR_CALL position follows its entry"
                );
                s[p][1]
            })
            .collect();
        (ops, calls)
    }

    #[test]
    fn a_moved_block_leaves_every_op_its_own_entries() {
        // Ops 0..4, 4..10 (a PAIR_CALL at 6), 10..20 (the last holds a block 12..15 with a
        // PAIR_CALL at 13): the block moves back to 4, before the second op.
        let mut c = numbered(20, &[6, 13], &[0..4, 4..10, 10..20]);
        c.move_block(0, 12..15, 4);
        let (ops, calls) = contents(&c);
        assert_eq!(ops[0], (0..4).collect::<Vec<_>>());
        assert_eq!(ops[1], (4..10).collect::<Vec<_>>());
        assert_eq!(ops[2], [10, 11, 15, 16, 17, 18, 19]);
        assert_eq!(calls, [6, 13]);
        let order: Vec<u32> = c.units[0].stream.iter().map(|e| e[1]).collect();
        assert_eq!(&order[..10], [0, 1, 2, 3, 12, 13, 14, 4, 5, 6]);
        // A block at the front of the stream (the probe's background) moved after the ops.
        let mut c = numbered(12, &[5], &[3..6, 6..9, 9..12]);
        c.move_block(0, 0..3, 9);
        let (ops, calls) = contents(&c);
        assert_eq!(ops[0], [3, 4, 5]);
        assert_eq!(ops[1], [6, 7, 8]);
        assert_eq!(ops[2], [9, 10, 11]);
        assert_eq!(calls, [5]);
        let order: Vec<u32> = c.units[0].stream.iter().map(|e| e[1]).collect();
        assert_eq!(order, [3, 4, 5, 6, 7, 8, 0, 1, 2, 9, 10, 11]);
        // A block that starts an op's range, moved back past another op.
        let mut c = numbered(10, &[], &[0..5, 5..10]);
        c.move_block(0, 5..7, 0);
        let (ops, _) = contents(&c);
        assert_eq!(ops[0], [0, 1, 2, 3, 4]);
        assert_eq!(ops[1], [7, 8, 9]);
        let order: Vec<u32> = c.units[0].stream.iter().map(|e| e[1]).collect();
        assert_eq!(&order[..2], [5, 6]);
    }

    #[test]
    fn no_record_crosses_a_chunk() {
        let chunk = dm::TRACE_CHUNK_ENTRIES as usize;
        // A stream of WAITs with a five-entry record placed to straddle the
        // first chunk's end.
        let mut v = vec![[op::WAIT, 0, 0, 0, 0, 0, 0, 0]; chunk - 2];
        let rec_len = record::len(record::GATHER);
        let mut rec = vec![[0u32; 8]; rec_len];
        rec[0][0] = record::GATHER;
        v.extend_from_slice(&rec);
        let out = chunked(&v);
        let at = out.iter().position(|e| e[0] == record::GATHER).unwrap();
        assert_eq!(at, chunk, "the record moved to the next chunk");
        assert!(out[chunk - 2..chunk].iter().all(|e| e[0] == op::WAIT));
        assert_eq!(out.len(), chunk + rec_len);
    }

    /// A region's `LAUNCH` and the `KERNEL_WAIT` that joins it, with more than a
    /// chunk of mover entries between: they fall in different chunks, in order,
    /// with their pairing (word 1, the generation) and every entry between
    /// untouched. The firmware keeps the active generation across chunks.
    #[test]
    fn a_launch_and_its_wait_can_fall_in_different_chunks() {
        let chunk = dm::TRACE_CHUNK_ENTRIES as usize;
        let launch = [op::LAUNCH, 7, 16, 1, 32, 1, 48, 1];
        let wait = [op::KERNEL_WAIT, 7, 0, 0, 0, 0, 0, 0];
        let rec_len = record::len(record::GATHER);
        let mut stream = vec![launch];
        // Enough five-entry records to fill two chunks and a part.
        for tag in 0..(2 * chunk / rec_len + 3) {
            let mut rec = vec![[0u32; 8]; rec_len];
            rec[0] = [record::GATHER, tag as u32, 0, 0, 0, 0, 0, 0];
            stream.extend_from_slice(&rec);
        }
        stream.push(wait);
        let out = chunked(&stream);
        let at = |entry: [u32; 8]| out.iter().position(|e| *e == entry).unwrap();
        let (launched, waited) = (at(launch), at(wait));
        assert!(launched / chunk < waited / chunk, "{launched} {waited}");
        assert!(waited - launched > chunk);
        // Nothing was dropped or reordered: what is not padding is the stream.
        let kept: Vec<_> = out
            .iter()
            .filter(|e| **e != [op::WAIT, 0, 0, 0, 0, 0, 0, 0])
            .copied()
            .collect();
        assert_eq!(kept, stream);
        // No record starts in one chunk and ends in the next.
        let mut i = 0;
        while i < out.len() {
            let n = record::len(out[i][0]).max(1);
            assert_eq!(i / chunk, (i + n - 1) / chunk, "record at {i}");
            i += n;
        }
    }

    #[test]
    fn a_composition_counts_records_once_and_padding_apart() {
        let chunk = dm::TRACE_CHUNK_ENTRIES as usize;
        let rec_len = record::len(record::GATHER);
        let poke = [op::POKE, 1, 2, 0, 0, 0, 0, 0];
        let mut stream = vec![poke, poke];
        // Records until one would straddle the first chunk's end, so there is padding.
        let records = chunk / rec_len + 1;
        for tag in 0..records {
            let mut rec = vec![[0u32; 8]; rec_len];
            rec[0] = [record::GATHER, tag as u32, 0, 0, 0, 0, 0, 0];
            stream.extend_from_slice(&rec);
        }
        let laid = chunked(&stream);
        let mix = Composition::of(&stream, laid.len());
        assert_eq!(mix.entries as usize, laid.len());
        assert_eq!(mix.padding as usize, laid.len() - stream.len());
        assert!(mix.padding > 0, "the stream was built to pad");
        // A five-entry record is one record, not five entries.
        assert_eq!(mix.heads[&record::GATHER] as usize, records);
        assert_eq!(mix.heads[&op::POKE], 2);
        assert_eq!(mix.heads.len(), 2);
        // Adding is a sum of every field.
        let mut total = mix.clone();
        total.add(&mix);
        assert_eq!(total.entries, 2 * mix.entries);
        assert_eq!(total.padding, 2 * mix.padding);
        assert_eq!(total.heads[&record::GATHER], 2 * mix.heads[&record::GATHER]);
    }
}
