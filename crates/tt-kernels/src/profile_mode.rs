//! The profile mode: where the device's time goes, op by op, for any workload, eager or traced
//! (`Session::profile_mode`). One switch, no per-workload analysis.
//!
//! What it records, from the data movers' timestamps (`crate::profile`):
//!
//! * **Every eager op** (a `Session` submission, labelled by `Session::set_op_label`, which burn-tt
//!   sets to the Burn op or op part): its span on the card, from its first list's start to its last
//!   list's end on the unit where it ran longest. A profile syncs after each op, so ops are timed one
//!   at a time, without overlap.
//! * **Every captured op of every trace**, replay by replay. A long trace's stamps do not fit a tile's
//!   16 KiB stamp buffer (a 1.7B-parameter decode step is ~1,900 ops), so each replay stamps a
//!   *window* of its ops (`tt_isa::dm::TRACE_WINDOW`) and the next replay the next window, until every
//!   op has been seen, then round again. Nothing is replayed twice and nothing is added to a replay,
//!   so a trace that carries state is profiled as it runs. An op's span runs from the previous op's
//!   barrier to its own on unit 0, which arrives at every op's barrier: its own work there and its
//!   wait for the slowest unit.
//!
//! Each span is split by the kind of entry it spent its time in (`OpTime`): the barrier, the kernels'
//! wait, the control entries (launches, kernels, circular buffers, descriptor pokes), a replay's
//! fetches of its own stream and sections, B's waits for its writer core and for a prefetch's last
//! reads, and data movement. In the
//! default [`Detail::Ops`] the data entries are not stamped (they keep the mover's untraced fast
//! paths); their time is the span less the rest.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use tt_device::trace::TraceEvent;
use tt_isa::dm::op;
use tt_isa::mailbox::trace as ev;

use crate::profile::{Event, RoleTotal};

/// How much of an op the movers stamp.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Detail {
    /// Only what bounds an op's phases (`tt_isa::dm::trace_mode::OPS`): ~8 events an op.
    #[default]
    Ops,
    /// Every entry (`trace_mode::FULL`): data entries timed one by one, ~25 events an op, so a
    /// window holds about a third as many ops.
    Full,
}

/// The profile mode's settings ([`crate::session::Session::profile_mode`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileOptions {
    pub detail: Detail,
    /// Also stamp the role runners (`START`, `PUSHED`, `RETIRED`). Off by default: role and mover
    /// stamps arriving together have been seen to lose events (`performance-log.md`).
    pub roles: bool,
    /// Stamp a replay a window at a time. Off, a replay is stamped whole, which overflows the stamp
    /// buffer for any trace of more than ~100 ops (the negative control of the window).
    pub window: bool,
    /// Have the math role total each kernel's pushing and its waits for the coprocessor
    /// (`tt_isa::mailbox::trace_word::PUSH`), which the mover stamps after the kernel's
    /// `ROLES_DONE`: [`UnitTime::push`]. Three mover stamps a kernel, no role stamps.
    pub push: bool,
}

impl Default for ProfileOptions {
    fn default() -> Self {
        Self {
            detail: Detail::Ops,
            roles: false,
            window: true,
            push: true,
        }
    }
}

impl ProfileOptions {
    /// The mover's `tt_isa::dm::TRACE` word for this detail.
    pub fn trace_mode(&self) -> u32 {
        match self.detail {
            Detail::Ops => tt_isa::dm::trace_mode::OPS,
            Detail::Full => tt_isa::dm::trace_mode::FULL,
        }
    }

    /// Events an entry headed by `op` may stamp, at most: its begin and end, the one event a
    /// launch (`KICK`) or a kernels' wait (`ROLES_DONE`) adds inside it, and after a wait the math
    /// role's two totals ([`ProfileOptions::push`]).
    pub fn stamps(&self, op: u32) -> u32 {
        let waits = matches!(op, op::KERNEL | op::KERNEL_WAIT);
        match self.detail {
            Detail::Ops if !tt_isa::dm::trace_mode::ops_stamped(op) => 0,
            _ if waits && self.push => 6,
            _ => 3,
        }
    }
}

/// Where one op's time went on the card, in tile cycles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpTime {
    /// From the previous op's end (its barrier, or its list's end) to this op's.
    pub span: u64,
    /// In `BARRIER` entries: arriving, and waiting for the slowest unit.
    pub barrier: u64,
    /// In `KERNEL_WAIT`: waiting for the op's kernels to retire.
    pub kernels: u64,
    /// In `LAUNCH`, `KERNEL`, `CB` and `POKE` entries.
    pub control: u64,
    /// B fetching the trace from GDDR: a chunk of its stream or of a `PAIR_CALL`'s reader section,
    /// read and waited for (`tt_isa::dm::trace_mode::FETCH`). Replays, [`Detail::Ops`] only.
    pub fetch: u64,
    /// B waiting for NC to finish a pair's writer section: NC's fetch of it and its writes
    /// (`tt_isa::dm::trace_mode::WRITER_WAIT`). [`Detail::Ops`] only: in `Full` it is data.
    pub writer: u64,
    /// B issuing what is left of a prefetch and waiting for it to land, before the kernel that
    /// reads it (`tt_isa::dm::trace_mode::PREFETCH_WAIT`): `B` read ahead that had not arrived.
    /// [`Detail::Ops`] only: in `Full` it is data.
    pub prefetch: u64,
    /// Moving data: the reads, writes, gathers and scatters ([`Detail::Full`]), or, in
    /// [`Detail::Ops`], the span less everything above.
    pub data: u64,
    /// [`Detail::Full`] only: the time between entries (the mover's own decode and dispatch).
    pub other: u64,
    /// The math role's pushing in the op's kernels on this unit ([`ProfileOptions::push`]): inside
    /// `kernels` and `data`, not a share of the span.
    pub push: u64,
    /// The math role's waits, after pushing, for the coprocessor to retire what it pushed.
    pub drain: u64,
    /// The instruction words the math role read and pushed (fused code's not counted). `push`
    /// near ~3.3 cycles a word is the runner's own rate: the kernel is bound by its pushing;
    /// far more, the pushes stalled on a full FIFO behind the backend or a semaphore.
    pub words: u64,
}

/// The most cycles a word an op's pushing may take and still count as bound by the pushing
/// (`ProfileReport`'s push table): the runner pushes at ~3.3 (`.ttinsn` experiment, card 0);
/// a push that waited on a full FIFO takes far more.
pub const PUSH_BOUND_CYCLES_A_WORD: f64 = 4.0;

/// The card's measured GDDR read rate, 120 tiles (performance scoreboard, "GDDR6 reads, card,
/// 120 tiles"): what a step's bytes are held against.
pub const CARD_READ_GBPS: f64 = 469.0;

/// What one captured op moves through GDDR, summed over its units: counted from its entries when
/// the trace is stored, with records expanded as the movers expand them. Bytes are requested
/// lengths; host and L1-only entries are not counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpBytes {
    /// Bytes read from GDDR.
    pub read: u64,
    /// Bytes written to GDDR.
    pub write: u64,
    /// Read entries: the mover's cost is per entry as much as per byte.
    pub reads: u64,
    /// Write entries.
    pub writes: u64,
    /// Of `read`, what `GATHER`s read into their `A` operand's slots (a product's activation).
    pub a: u64,
    /// Of `read`, what `GATHER`s read into their `B` operand's slots (its weights, or a cache).
    pub b: u64,
    /// Of `read`, what lists read ahead of the op, under the ops before it: an idle unit's
    /// `PREFETCH` list, a busy unit's `BUSY_PREFETCH` (`trace::OpRecord::ahead`). Not moved in
    /// the op's own span.
    pub ahead: u64,
}

impl OpBytes {
    pub fn add(&mut self, o: &OpBytes) {
        self.read += o.read;
        self.write += o.write;
        self.reads += o.reads;
        self.writes += o.writes;
        self.a += o.a;
        self.b += o.b;
        self.ahead += o.ahead;
    }

    /// Count `entries` (a list, a stream or a section; `PAIR` headers, credits and control entries
    /// move nothing) into `self`. A record that does not expand is counted as nothing: the capture
    /// validated it, so that does not happen.
    pub fn count(&mut self, entries: &[[u32; 8]]) {
        use tt_isa::dm::{record, TILE_SLOT};
        let mut i = 0;
        while i < entries.len() {
            let head = entries[i];
            let n = record::len(head[0]).max(1);
            if record::is_record(head[0]) {
                let Some(rec) = entries.get(i..i + n) else {
                    return;
                };
                // A gather's `A` slots, from its header: `rows x kt` slots from `a_at`. One with
                // `A` resident reads only `B` (a packed `A`'s tiles are a `READ_RUN` of their
                // own, counted as reads, not as `A`).
                let a_slots = (head[0] == record::GATHER).then(|| {
                    let (a_at, rows, kt) = (
                        head[2] as u64,
                        head[5] as u64,
                        record::gather_kt(head[1]) as u64,
                    );
                    if head[1] & record::GATHER_A_RESIDENT != 0 {
                        return 0..0;
                    }
                    a_at..a_at + rows * kt * TILE_SLOT
                });
                let _ = record::expand(rec, |e| {
                    let len = self.entry(e);
                    if len > 0 {
                        match &a_slots {
                            Some(a) if a.contains(&(e[4] as u64)) => self.a += len,
                            Some(_) => self.b += len,
                            None => {}
                        }
                    }
                    Ok(())
                });
            } else {
                self.entry(head);
            }
            i += n;
        }
    }

    /// Count one plain entry: the bytes it reads from GDDR (returned), or writes.
    fn entry(&mut self, e: [u32; 8]) -> u64 {
        match e[0] {
            // A transposed or broadcast read is one tile slot, its length in the same word.
            op::READ | op::READ_TRANSPOSED | op::READ_BROADCAST_COL => {
                self.read += e[5] as u64;
                self.reads += 1;
                e[5] as u64
            }
            op::READ_ENCODED => {
                self.read += e[7] as u64;
                self.reads += 1;
                e[7] as u64
            }
            op::WRITE => {
                self.write += e[5] as u64;
                self.writes += 1;
                0
            }
            _ => 0,
        }
    }
}

impl OpTime {
    fn finish(mut self, detail: Detail, mut data: u64) -> Self {
        if detail == Detail::Full {
            // `Full` stamps every `WAIT` as itself: moving data, not a prefetch's own stamp.
            data += std::mem::take(&mut self.prefetch);
        }
        let known =
            self.barrier + self.kernels + self.control + self.fetch + self.writer + self.prefetch;
        match detail {
            Detail::Ops => self.data = self.span.saturating_sub(known),
            Detail::Full => {
                self.data = data;
                self.other = self.span.saturating_sub(known + data);
            }
        }
        self
    }
}

/// Adds each stamped entry's time to its class.
#[derive(Default)]
struct Classes {
    open: Vec<(u32, u64)>,
    time: OpTime,
    data: u64,
}

impl Classes {
    fn see(&mut self, event: Event, t: u64) {
        match event {
            Event::Entry { begin: true, op } => self.open.push((op, t)),
            Event::Entry {
                begin: false,
                op: end,
            } => {
                if let Some(at) = self.open.iter().rposition(|&(o, _)| o == end) {
                    let (_, t0) = self.open.remove(at);
                    let d = t.saturating_sub(t0);
                    match end {
                        op::BARRIER => self.time.barrier += d,
                        op::KERNEL_WAIT => self.time.kernels += d,
                        op::LAUNCH | op::KERNEL | op::CB | op::POKE => self.time.control += d,
                        tt_isa::dm::trace_mode::FETCH => self.time.fetch += d,
                        tt_isa::dm::trace_mode::WRITER_WAIT => self.time.writer += d,
                        tt_isa::dm::trace_mode::PREFETCH_WAIT => self.time.prefetch += d,
                        _ => self.data += d,
                    }
                }
            }
            Event::RolePush { what, amount } => match what {
                RoleTotal::Push => self.time.push += amount,
                RoleTotal::Drain => self.time.drain += amount,
                RoleTotal::Words => self.time.words += amount,
            },
            _ => {}
        }
    }

    fn take(&mut self, span: u64, detail: Detail) -> OpTime {
        let time = OpTime {
            span,
            ..std::mem::take(&mut self.time)
        };
        let data = std::mem::take(&mut self.data);
        self.open.clear();
        time.finish(detail, data)
    }
}

fn mover(e: &TraceEvent) -> Option<(Event, u64)> {
    let event = Event::decode(e.token);
    let from_mover = (e.token >> 8) & 0xF == ev::MOVER;
    from_mover.then_some((event, e.cycles))
}

/// One eager op on one unit: its span from its first list's start to its last list's end, split by
/// class. `None` if the unit ran none of it.
pub fn attribute_eager(events: &[TraceEvent], detail: Detail) -> Option<OpTime> {
    let mut first = None;
    let mut last = None;
    let mut classes = Classes::default();
    for (event, t) in events.iter().filter_map(mover) {
        match event {
            Event::List { begin: true, .. } => {
                first.get_or_insert(t);
            }
            Event::List { begin: false, .. } => last = Some(t),
            other => classes.see(other, t),
        }
    }
    let (first, last) = (first?, last?);
    Some(classes.take(last.saturating_sub(first), detail))
}

/// One replay's stamps on unit 0 (from its last outermost list on: the replay's `CALL`), stamped
/// within the window `lo..` (`TRACE_WINDOW`): the ops whose span lies wholly inside it, in order.
/// With `lo == 0` the first is op 0, timed from the list's start; otherwise op `lo` is only partly
/// stamped and the first returned is op `lo + 1`.
pub fn attribute_replay(events: &[TraceEvent], lo: u32, detail: Detail) -> Vec<OpTime> {
    let stamps: Vec<(Event, u64)> = events.iter().filter_map(mover).collect();
    let start = stamps
        .iter()
        .rposition(|(e, _)| matches!(e, Event::List { begin: true, .. }))
        .unwrap_or(0);
    let mut since = match (lo, stamps.get(start)) {
        (0, Some((Event::List { begin: true, .. }, t))) => Some(*t),
        _ => None,
    };
    let mut classes = Classes::default();
    let mut out = Vec::new();
    for &(event, t) in &stamps[start..] {
        classes.see(event, t);
        if let Event::Entry {
            begin: false,
            op: op::BARRIER,
        } = event
        {
            match since {
                Some(s) => out.push(classes.take(t.saturating_sub(s), detail)),
                None => {
                    classes.take(0, detail);
                }
            }
            since = Some(t);
        }
    }
    out
}

/// One entry kind's share of an op on unit 0 ([`attribute_entries`]), in tile cycles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EntryCost {
    /// Entries of the kind the op ran.
    pub count: u64,
    /// Inside them, from each one's begin stamp to its end.
    pub cycles: u64,
    /// Before them: from the end of the entry before (or the op's start) to each one's begin --
    /// the mover's decode and dispatch, and what runs unstamped there: a pair's handshake with
    /// its writer core, a stream chunk's or a reader section's fetch.
    pub gap: u64,
}

/// Each op's entries by kind (the head's first word), as [`attribute_replay`] splits a replay.
pub type EntryCosts = BTreeMap<u32, EntryCost>;

/// [`Detail::Full`]: one replay's ops on unit 0, as [`attribute_replay`] returns them (the same ops,
/// in the same order), each split by the kind of entry it ran. A `PAIR_CALL` itself is never
/// stamped: its reader section's entries are, and its handshake and its wait for the writer core
/// are the gaps before the entries that follow them.
pub fn attribute_entries(events: &[TraceEvent], lo: u32) -> Vec<EntryCosts> {
    let stamps: Vec<(Event, u64)> = events.iter().filter_map(mover).collect();
    let start = stamps
        .iter()
        .rposition(|(e, _)| matches!(e, Event::List { begin: true, .. }))
        .unwrap_or(0);
    let mut timing = matches!(
        (lo, stamps.get(start)),
        (0, Some((Event::List { begin: true, .. }, _)))
    );
    let mut last = stamps.get(start).map_or(0, |s| s.1);
    let mut open: Vec<(u32, u64)> = Vec::new();
    let mut costs = EntryCosts::new();
    let mut out = Vec::new();
    for &(event, t) in &stamps[start..] {
        match event {
            Event::Entry { begin: true, op } => {
                if open.is_empty() {
                    costs.entry(op).or_default().gap += t.saturating_sub(last);
                }
                open.push((op, t));
            }
            Event::Entry { begin: false, op } => {
                if let Some(at) = open.iter().rposition(|&(o, _)| o == op) {
                    let (_, t0) = open.remove(at);
                    let c = costs.entry(op).or_default();
                    c.count += 1;
                    c.cycles += t.saturating_sub(t0);
                }
                last = t;
                if op == op::BARRIER {
                    // An entry whose end fell outside the window never closes: the op ends here.
                    open.clear();
                    let done = std::mem::take(&mut costs);
                    if timing {
                        out.push(done);
                    }
                    timing = true;
                }
            }
            _ => {}
        }
    }
    out
}

/// One replay's stamps on unit `unit`, whose barriers are `seq` in stream order, stamped within the
/// window `lo..hi` (`hi` 0 for none): each op it arrived at whose work is wholly stamped, as `(op,
/// time)`. The mover stamps a barrier when its count of the trace's barriers so far lies in the
/// window, counting an arrival as one and a wait as the barriers it stands for; this counts the
/// same way to tell which stamp is which. `None` if the stamps and `seq` do not line up.
pub fn unit_times(
    events: &[TraceEvent],
    seq: &[UnitBarrier],
    unit: u16,
    lo: u32,
    hi: u32,
) -> Option<Vec<(usize, UnitTime)>> {
    let stamps: Vec<(Event, u64)> = events.iter().filter_map(mover).collect();
    let Some(start) = stamps
        .iter()
        .rposition(|(e, _)| matches!(e, Event::List { begin: true, .. }))
    else {
        return Some(Vec::new());
    };
    let mut pairs = Vec::new();
    let mut open = None;
    // The math role's totals since the last barrier: the work before the next arrival.
    let mut totals = [0u64; 3];
    for &(event, t) in &stamps[start..] {
        match event {
            Event::Entry {
                begin: true,
                op: op::BARRIER,
            } => open = Some((t, std::mem::take(&mut totals))),
            Event::Entry {
                begin: false,
                op: op::BARRIER,
            } => {
                let (arrive, totals) = open.take()?;
                pairs.push((arrive, t, totals));
            }
            Event::RolePush { what, amount } => totals[what as usize] += amount,
            _ => {}
        }
    }
    let mut pairs = pairs.into_iter();
    let mut out = Vec::new();
    let mut since = (lo == 0).then_some(stamps[start].1);
    let mut count = 0u32;
    for b in seq {
        if hi == 0 || (lo <= count && count < hi) {
            let (arrive, release, [push, drain, words]) = pairs.next()?;
            if let (Some(k), Some(s)) = (b.arrival_of, since) {
                out.push((
                    k as usize,
                    UnitTime {
                        unit,
                        work: arrive.saturating_sub(s),
                        wait: release.saturating_sub(arrive),
                        push,
                        drain,
                        words,
                    },
                ));
            }
            since = Some(release);
        } else {
            since = None;
        }
        count += b.counts;
    }
    pairs.next().is_none().then_some(out)
}

/// One eager op's profile.
#[derive(Clone, Debug)]
pub struct EagerSample {
    pub label: &'static str,
    pub what: &'static str,
    pub time: OpTime,
}

/// A captured op, as the profile names it.
#[derive(Clone, Debug)]
pub struct OpInfo {
    pub label: &'static str,
    pub what: &'static str,
    /// Units that ran a share of it.
    pub units: usize,
    /// What it moves through GDDR.
    pub bytes: OpBytes,
    /// What `bytes` is made of, unit by unit (`trace::OpRecord::unit_bytes`).
    pub unit_bytes: Vec<(u16, OpBytes)>,
}

impl OpInfo {
    /// What unit `u` moved of it, if anything.
    pub fn unit_bytes(&self, u: u16) -> Option<&OpBytes> {
        self.unit_bytes
            .iter()
            .find(|(v, _)| *v == u)
            .map(|(_, b)| b)
    }

    /// What unit `u` moved in the op's own span: its reads and writes less what it read ahead.
    pub fn unit_span_bytes(&self, u: u16) -> u64 {
        self.unit_bytes(u).map_or(0, |b| b.read + b.write - b.ahead)
    }
}

/// One of a unit's `BARRIER` entries in a trace's stream, in the order the unit runs them: what
/// tells which of its stamped barriers is which op's ([`unit_times`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitBarrier {
    /// The op it is the unit's arrival at; `None` for a wait-only barrier (those the unit sat
    /// out, before its next op or at the trace's end).
    pub arrival_of: Option<u32>,
    /// How many of the trace's barriers the window counts it as (the mover's `TRACE_BARRIERS`):
    /// one for an arrival, `dm::BARRIER_STANDS_FOR` for a wait.
    pub counts: u32,
}

/// One unit's time on one op of a replay, from its own stamps, so no two tiles' clocks are
/// compared: its work, from its previous barrier's release to its arrival at the op's, and its
/// wait there for the slowest unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitTime {
    pub unit: u16,
    pub work: u64,
    pub wait: u64,
    /// Of `work`, the math role's pushing ([`OpTime::push`]), 0 unless measured.
    pub push: u64,
    /// Of `work`, the math role's waits for the coprocessor after pushing ([`OpTime::drain`]).
    pub drain: u64,
    /// The words the math role pushed ([`OpTime::words`]).
    pub words: u64,
}

/// One unit's medians on one op over the replays that stamped it ([`TraceSamples::unit_medians`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitMedian {
    pub unit: u16,
    pub work: u64,
    pub wait: u64,
    pub push: u64,
    pub drain: u64,
    pub words: u64,
    pub samples: usize,
}

/// A trace's ops and their samples, one per replay that stamped them.
#[derive(Clone, Debug, Default)]
pub struct TraceSamples {
    pub ops: Vec<OpInfo>,
    pub samples: Vec<Vec<OpTime>>,
    /// Per op, each unit that arrived at its barrier, once per replay that stamped it.
    pub units: Vec<Vec<UnitTime>>,
    /// Units' replays whose barrier stamps did not line up with their streams: not attributed.
    pub unit_mismatches: u32,
    /// [`Detail::Full`]: per op, unit 0's entries by kind ([`attribute_entries`]), summed over the
    /// replays that stamped it (`samples[k].len()` of them).
    pub entries: Vec<EntryCosts>,
    /// Per unit, its barriers in stream order (empty where unknown).
    unit_barriers: Vec<Vec<UnitBarrier>>,
    pub replays: u32,
    /// Replays whose stamps outran a unit's buffer: none of their ops were attributed.
    pub overflowed: u32,
    /// Each op's stamps on its busiest unit, at most ([`ProfileOptions::stamps`]): what sizes a window.
    stamps: Vec<u32>,
    /// The first op of the next replay's window.
    cursor: usize,
    /// Its ops end in barriers on unit 0, so a replay can be split into ops.
    segmented: bool,
}

/// What the profile is waiting to attribute when the next drain comes.
#[derive(Clone, Copy, Debug)]
enum Pending {
    Eager {
        label: &'static str,
        what: &'static str,
    },
    Replay {
        trace: u64,
        lo: u32,
        /// The window's end, 0 for none.
        hi: u32,
        first: usize,
    },
}

/// Events a unit's stamp buffer holds (`tt_isa::mailbox::TRACE_BUFFER_BYTES`), less room for the
/// catch-up list before a replay and the replay's own list.
const WINDOW_EVENTS: u32 = (tt_isa::mailbox::TRACE_BUFFER_BYTES / 16) as u32 - 96;

/// The profile mode's state in a session: see the module documentation.
#[derive(Debug)]
pub struct Profiler {
    pub options: ProfileOptions,
    eager: Vec<EagerSample>,
    /// Eager ops whose stamps outran a unit's buffer, by label: not attributed.
    eager_overflowed: BTreeMap<&'static str, u32>,
    /// The stamps since the last attribution outran a buffer.
    overflow: bool,
    traces: BTreeMap<u64, TraceSamples>,
    pending: Option<Pending>,
    counter_at_start: u64,
    host_at_start: std::time::Instant,
}

impl Profiler {
    pub(crate) fn new(options: ProfileOptions, counter_at_start: u64) -> Self {
        Self {
            options,
            eager: Vec::new(),
            eager_overflowed: BTreeMap::new(),
            overflow: false,
            traces: BTreeMap::new(),
            pending: None,
            counter_at_start,
            host_at_start: std::time::Instant::now(),
        }
    }

    /// A trace's ops, each op's stamps on its busiest unit, and each unit's barriers, when it is
    /// first replayed.
    pub(crate) fn know_trace(
        &mut self,
        trace: u64,
        ops: impl FnOnce() -> (Vec<OpInfo>, Vec<u32>, bool, Vec<Vec<UnitBarrier>>),
    ) {
        self.traces.entry(trace).or_insert_with(|| {
            let (ops, stamps, segmented, unit_barriers) = ops();
            TraceSamples {
                samples: vec![Vec::new(); ops.len()],
                units: vec![Vec::new(); ops.len()],
                entries: vec![EntryCosts::new(); ops.len()],
                ops,
                stamps,
                segmented,
                unit_barriers,
                ..Default::default()
            }
        });
    }

    /// The window the next replay of `trace` stamps (`lo | hi << 16`, 0 for all), and the op it
    /// starts timing at. Sized so the stamps of the busiest unit fit its buffer.
    pub(crate) fn next_window(&mut self, trace: u64) -> u32 {
        let windowed = self.options.window;
        let t = self
            .traces
            .get_mut(&trace)
            .expect("known before its replay");
        let n = t.ops.len();
        if !windowed || !t.segmented || n == 0 {
            self.pending = Some(Pending::Replay {
                trace,
                lo: 0,
                hi: 0,
                first: 0,
            });
            return 0;
        }
        let first = t.cursor;
        let lo = first.saturating_sub(1);
        let mut events = if first == 0 { 4 } else { t.stamps[lo] };
        let mut end = first;
        while end < n && events + t.stamps[end] <= WINDOW_EVENTS {
            events += t.stamps[end];
            end += 1;
        }
        let end = end.max(first + 1).min(n);
        t.cursor = if end >= n { 0 } else { end };
        self.pending = Some(Pending::Replay {
            trace,
            lo: lo as u32,
            hi: end as u32,
            first,
        });
        (lo as u32) | ((end as u32) << 16)
    }

    pub(crate) fn expect_eager(&mut self, label: &'static str, what: &'static str) {
        self.pending = Some(Pending::Eager { label, what });
    }

    /// A unit's stamps since the last attribution outran its buffer: what was pending is counted as
    /// overflowed, not attributed.
    pub(crate) fn overflowed(&mut self) {
        self.overflow = true;
    }

    pub(crate) fn expecting_replay(&self) -> bool {
        matches!(self.pending, Some(Pending::Replay { .. }))
    }

    /// The stamps the units recorded since the last drain (one stream a unit, unit 0 first),
    /// attributed to what was pending.
    pub(crate) fn drained(&mut self, units: &[Vec<TraceEvent>]) {
        let detail = self.options.detail;
        if std::mem::take(&mut self.overflow) {
            match self.pending.take() {
                Some(Pending::Eager { label, .. }) => {
                    *self.eager_overflowed.entry(label).or_default() += 1;
                }
                Some(Pending::Replay { trace, .. }) => {
                    if let Some(t) = self.traces.get_mut(&trace) {
                        t.replays += 1;
                        t.overflowed += 1;
                    }
                }
                None => {}
            }
            return;
        }
        match self.pending.take() {
            Some(Pending::Eager { label, what }) => {
                let longest = units
                    .iter()
                    .filter_map(|e| attribute_eager(e, detail))
                    .max_by_key(|t| t.span);
                if let Some(time) = longest {
                    self.eager.push(EagerSample { label, what, time });
                }
            }
            Some(Pending::Replay {
                trace,
                lo,
                hi,
                first,
            }) => {
                let Some(t) = self.traces.get_mut(&trace) else {
                    return;
                };
                t.replays += 1;
                let unit0 = units.first().map(Vec::as_slice).unwrap_or(&[]);
                if t.segmented {
                    for (k, time) in attribute_replay(unit0, lo, detail).into_iter().enumerate() {
                        if let Some(s) = t.samples.get_mut(first + k) {
                            s.push(time);
                        }
                    }
                    if detail == Detail::Full {
                        for (k, costs) in attribute_entries(unit0, lo).into_iter().enumerate() {
                            if let Some(sum) = t.entries.get_mut(first + k) {
                                for (op, c) in costs {
                                    let s = sum.entry(op).or_default();
                                    s.count += c.count;
                                    s.cycles += c.cycles;
                                    s.gap += c.gap;
                                }
                            }
                        }
                    }
                    for (u, events) in units.iter().enumerate() {
                        let Some(seq) = t.unit_barriers.get(u).filter(|s| !s.is_empty()) else {
                            continue;
                        };
                        match unit_times(events, seq, u as u16, lo, hi) {
                            Some(times) => {
                                for (k, time) in times {
                                    if let Some(s) = t.units.get_mut(k) {
                                        s.push(time);
                                    }
                                }
                            }
                            None => t.unit_mismatches += 1,
                        }
                    }
                } else if let (Some(time), Some(s)) =
                    (attribute_eager(unit0, detail), t.samples.first_mut())
                {
                    // One unit and no barriers: the replay as a whole, charged to its first op.
                    s.push(time);
                }
            }
            None => {}
        }
    }

    /// Everything recorded, with the tiles' clock measured since the mode began.
    pub(crate) fn report(&self, counter_now: u64) -> ProfileReport {
        let us = self.host_at_start.elapsed().as_secs_f64() * 1e6;
        ProfileReport {
            ticks_per_us: (counter_now.saturating_sub(self.counter_at_start)) as f64 / us.max(1.0),
            eager: self.eager.clone(),
            eager_overflowed: self.eager_overflowed.clone(),
            traces: self.traces.clone(),
        }
    }
}

/// What a profile recorded: see the module documentation. Times are tile cycles;
/// `ticks_per_us` converts them.
#[derive(Clone, Debug, Default)]
pub struct ProfileReport {
    pub ticks_per_us: f64,
    pub eager: Vec<EagerSample>,
    /// Eager ops whose stamps outran a unit's buffer (16 KiB, ~1,000 events), by label: their time
    /// is not attributed. A single op of that size (a cast of a whole embedding table, say) is
    /// rarely what a profile is for; [`Detail::Ops`] keeps them few.
    pub eager_overflowed: BTreeMap<&'static str, u32>,
    /// By the session's trace id.
    pub traces: BTreeMap<u64, TraceSamples>,
}

/// `bytes` moved in `us` microseconds, in GB/s (`0` for no time).
fn gbps(bytes: u64, us: f64) -> f64 {
    if us > 0.0 {
        bytes as f64 / us / 1e3
    } else {
        0.0
    }
}

fn median(mut v: Vec<u64>) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

impl TraceSamples {
    /// Each op's median time over the replays that stamped it (component by component), or `None`
    /// for an op no replay covered yet.
    pub fn medians(&self) -> Vec<Option<OpTime>> {
        self.samples
            .iter()
            .map(|s| {
                (!s.is_empty()).then(|| {
                    let m = |f: fn(&OpTime) -> u64| median(s.iter().map(f).collect());
                    OpTime {
                        span: m(|t| t.span),
                        barrier: m(|t| t.barrier),
                        kernels: m(|t| t.kernels),
                        control: m(|t| t.control),
                        fetch: m(|t| t.fetch),
                        writer: m(|t| t.writer),
                        prefetch: m(|t| t.prefetch),
                        data: m(|t| t.data),
                        other: m(|t| t.other),
                        push: m(|t| t.push),
                        drain: m(|t| t.drain),
                        words: m(|t| t.words),
                    }
                })
            })
            .collect()
    }

    /// Per op, each unit's median work, wait and math role's totals over the replays that stamped
    /// it, by unit.
    pub fn unit_medians(&self) -> Vec<Vec<UnitMedian>> {
        self.units
            .iter()
            .map(|times| {
                let mut by: BTreeMap<u16, Vec<&UnitTime>> = BTreeMap::new();
                for t in times {
                    by.entry(t.unit).or_default().push(t);
                }
                by.into_iter()
                    .map(|(unit, ts)| {
                        let m = |f: fn(&UnitTime) -> u64| median(ts.iter().map(|t| f(t)).collect());
                        UnitMedian {
                            unit,
                            work: m(|t| t.work),
                            wait: m(|t| t.wait),
                            push: m(|t| t.push),
                            drain: m(|t| t.drain),
                            words: m(|t| t.words),
                            samples: ts.len(),
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Ops that at least one replay stamped.
    pub fn covered(&self) -> usize {
        self.samples.iter().filter(|s| !s.is_empty()).count()
    }
}

/// Times summed under one label.
#[derive(Clone, Copy, Debug, Default)]
struct Total {
    count: u64,
    time: OpTime,
}

impl Total {
    fn add(&mut self, t: &OpTime) {
        self.count += 1;
        self.time.span += t.span;
        self.time.barrier += t.barrier;
        self.time.kernels += t.kernels;
        self.time.control += t.control;
        self.time.fetch += t.fetch;
        self.time.writer += t.writer;
        self.time.prefetch += t.prefetch;
        self.time.data += t.data;
        self.time.other += t.other;
        self.time.push += t.push;
        self.time.drain += t.drain;
        self.time.words += t.words;
    }
}

fn by_label<'a>(items: impl Iterator<Item = (&'a str, &'a OpTime)>) -> Vec<(&'a str, Total)> {
    let mut m: BTreeMap<&str, Total> = BTreeMap::new();
    for (label, t) in items {
        m.entry(label).or_default().add(t);
    }
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort_by_key(|(_, t)| std::cmp::Reverse(t.time.span));
    v
}

impl ProfileReport {
    fn us(&self, cycles: u64) -> f64 {
        if self.ticks_per_us > 0.0 {
            cycles as f64 / self.ticks_per_us
        } else {
            0.0
        }
    }

    fn label_table(&self, out: &mut String, rows: &[(&str, Total)], count: &str) {
        let total: u64 = rows.iter().map(|(_, t)| t.time.span).sum();
        let _ = writeln!(
            out,
            "| label | {count} | us | % | us each | barrier | kernels | control | fetch | writer | prefetch | data | other |"
        );
        let _ = writeln!(out, "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|");
        for (label, t) in rows {
            let pct = |c: u64| 100.0 * c as f64 / t.time.span.max(1) as f64;
            let _ =
                writeln!(
                out,
                "| {} | {} | {:.1} | {:.1} | {:.2} | {:.0}% | {:.0}% | {:.0}% | {:.0}% | {:.0}% | {:.0}% | {:.0}% | {:.0}% |",
                if label.is_empty() { "(unlabelled)" } else { label },
                t.count,
                self.us(t.time.span),
                100.0 * t.time.span as f64 / total.max(1) as f64,
                self.us(t.time.span) / t.count.max(1) as f64,
                pct(t.time.barrier),
                pct(t.time.kernels),
                pct(t.time.control),
                pct(t.time.fetch),
                pct(t.time.writer),
                pct(t.time.prefetch),
                pct(t.time.data),
                pct(t.time.other),
            );
        }
    }

    /// A trace's GDDR traffic (counted when it was stored, in the profile mode) against its covered
    /// ops' median spans: by label, then the step budget -- its bytes at [`CARD_READ_GBPS`] against
    /// its time, the products (ops that gathered a `B` operand) apart from everything else.
    fn bytes_tables(&self, out: &mut String, t: &TraceSamples, medians: &[Option<OpTime>]) {
        let covered: Vec<(&OpInfo, &OpTime)> = t
            .ops
            .iter()
            .zip(medians)
            .filter_map(|(o, m)| m.as_ref().map(|m| (o, m)))
            .collect();
        if covered.iter().all(|(o, _)| o.bytes == OpBytes::default()) {
            return;
        }
        #[derive(Default)]
        struct Sum {
            ops: u64,
            span: u64,
            barrier: u64,
            bytes: OpBytes,
        }
        impl Sum {
            fn add(&mut self, o: &OpInfo, m: &OpTime) {
                self.ops += 1;
                self.span += m.span;
                self.barrier += m.barrier;
                self.bytes.add(&o.bytes);
            }
        }
        let mb = |b: u64| b as f64 / 1e6;
        let mut labels: BTreeMap<&str, Sum> = BTreeMap::new();
        for (o, m) in &covered {
            labels.entry(o.label).or_default().add(o, m);
        }
        let mut labels: Vec<_> = labels.into_iter().collect();
        labels.sort_by_key(|(_, s)| std::cmp::Reverse(s.bytes.read + s.bytes.write));
        let _ = writeln!(
            out,
            "\nGDDR traffic by label (all units; `A`/`B`: what gathers read into a product's operands):\n\n\
             | label | ops | us | read MB | write MB | reads | writes | A MB | B MB | ahead MB | GB/s in its span |\n\
             |---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
        );
        for (label, s) in &labels {
            let b = &s.bytes;
            let _ = writeln!(
                out,
                "| {} | {} | {:.1} | {:.1} | {:.1} | {} | {} | {:.1} | {:.1} | {:.1} | {:.0} |",
                if label.is_empty() {
                    "(unlabelled)"
                } else {
                    label
                },
                s.ops,
                self.us(s.span),
                mb(b.read),
                mb(b.write),
                b.reads,
                b.writes,
                mb(b.a),
                mb(b.b),
                mb(b.ahead),
                gbps(b.read + b.write - b.ahead, self.us(s.span)),
            );
        }
        let (mut products, mut rest, mut all) = (Sum::default(), Sum::default(), Sum::default());
        for (o, m) in &covered {
            all.add(o, m);
            if o.bytes.b > 0 {
                products.add(o, m)
            } else {
                rest.add(o, m)
            }
        }
        let floor = |bytes: u64| bytes as f64 / CARD_READ_GBPS / 1e3;
        let _ = writeln!(
            out,
            "\nStep budget ({} of {} ops covered): {:.1} us, {:.1} MB read and {:.1} MB written; \
             at {CARD_READ_GBPS} GB/s those bytes take {:.1} us, the products' `B` alone {:.1} us. \
             Barriers {:.1} us. The products' `B` is {:.1} MB, {:.1} of it read ahead of its \
             product: over the covered spans the weights stream at {:.0} GB/s.\n\n\
             | part | ops | us | MB moved | entries | GB/s | us at {CARD_READ_GBPS} GB/s | us over it |\n\
             |---|--:|--:|--:|--:|--:|--:|--:|",
            covered.len(),
            t.ops.len(),
            self.us(all.span),
            mb(all.bytes.read),
            mb(all.bytes.write),
            floor(all.bytes.read + all.bytes.write),
            floor(all.bytes.b),
            self.us(all.barrier),
            mb(products.bytes.b),
            mb(products.bytes.ahead),
            gbps(products.bytes.b, self.us(all.span)),
        );
        for (part, s) in [
            ("products", &products),
            ("everything else", &rest),
            ("all", &all),
        ] {
            let moved = s.bytes.read + s.bytes.write;
            let _ = writeln!(
                out,
                "| {part} | {} | {:.1} | {:.1} | {} | {:.0} | {:.1} | {:.1} |",
                s.ops,
                self.us(s.span),
                mb(moved),
                s.bytes.reads + s.bytes.writes,
                gbps(moved, self.us(s.span)),
                floor(moved),
                self.us(s.span) - floor(moved),
            );
        }
    }

    /// A trace's units against each other, by label: per op, each unit's median work (from its
    /// previous barrier's release to its arrival at the op's), the fastest, the median and the
    /// slowest unit's summed over the label's ops, and the units slowest most often.
    fn skew_table(&self, out: &mut String, t: &TraceSamples) {
        let per_op = t.unit_medians();
        if per_op.iter().all(Vec::is_empty) {
            return;
        }
        #[derive(Default)]
        struct Skew {
            ops: u64,
            units: u64,
            fastest: u64,
            median: u64,
            slowest: u64,
            slowest_units: BTreeMap<u16, u32>,
            /// What the slowest unit moved in its own span, bytes.
            slowest_bytes: u64,
            /// What every unit moved in the op's span, bytes.
            span_bytes: u64,
        }
        let mut labels: BTreeMap<&str, Skew> = BTreeMap::new();
        for (o, units) in t.ops.iter().zip(&per_op) {
            if units.is_empty() {
                continue;
            }
            let mut work: Vec<(u64, u16)> = units.iter().map(|m| (m.work, m.unit)).collect();
            work.sort_unstable();
            let s = labels.entry(o.label).or_default();
            s.ops += 1;
            s.units += work.len() as u64;
            s.fastest += work[0].0;
            s.median += work[work.len() / 2].0;
            let last = work[work.len() - 1];
            s.slowest += last.0;
            *s.slowest_units.entry(last.1).or_default() += 1;
            s.slowest_bytes += o.unit_span_bytes(last.1);
            s.span_bytes += o.bytes.read + o.bytes.write - o.bytes.ahead;
        }
        let mut labels: Vec<_> = labels.into_iter().collect();
        labels.sort_by_key(|(_, s)| std::cmp::Reverse(s.slowest - s.fastest));
        let _ = writeln!(
            out,
            "\nUnits against each other (each unit's own stamps: its work from its last barrier's \
             release to its arrival; {} unit replays did not line up and are left out):\n\n\
             | label | ops | units an op | fastest us | median us | slowest us | slowest - median us | \
             slowest's MB in its span | slowest's GB/s | all units' MB in the span at {CARD_READ_GBPS} GB/s, us | \
             slowest most often |\n\
             |---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|",
            t.unit_mismatches
        );
        for (label, s) in labels {
            let mut often: Vec<_> = s.slowest_units.into_iter().collect();
            often.sort_by_key(|&(u, n)| (std::cmp::Reverse(n), u));
            let often: Vec<String> = often
                .iter()
                .take(4)
                .map(|(u, n)| format!("{u} x{n}"))
                .collect();
            let _ = writeln!(
                out,
                "| {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} | {:.1} | {:.1} | {} |",
                if label.is_empty() {
                    "(unlabelled)"
                } else {
                    label
                },
                s.ops,
                s.units as f64 / s.ops as f64,
                self.us(s.fastest),
                self.us(s.median),
                self.us(s.slowest),
                self.us(s.slowest - s.median),
                s.slowest_bytes as f64 / 1e6,
                gbps(s.slowest_bytes, self.us(s.slowest)),
                s.span_bytes as f64 / CARD_READ_GBPS / 1e3,
                often.join(", ")
            );
        }
    }

    /// A trace's math role against its units' work, by label: per op, the slowest unit's work, and
    /// its math role's pushing and its waits after pushing for the coprocessor
    /// ([`ProfileOptions::push`]). A label whose pushing is most of its work, with little drain, is
    /// bound by its pushing: what a faster push (`.ttinsn`) would take from.
    fn push_table(&self, out: &mut String, t: &TraceSamples) {
        let per_op = t.unit_medians();
        if per_op.iter().flatten().all(|m| m.push == 0 && m.drain == 0) {
            return;
        }
        #[derive(Default)]
        struct Push {
            ops: u64,
            work: u64,
            push: u64,
            drain: u64,
            words: u64,
            /// Of `push`, the ops' whose pushing ran at the runner's own rate.
            bound: u64,
        }
        let mut labels: BTreeMap<&str, Push> = BTreeMap::new();
        let mut all = Push::default();
        for (o, units) in t.ops.iter().zip(&per_op) {
            let Some(slowest) = units.iter().max_by_key(|m| (m.work, m.unit)) else {
                continue;
            };
            let bound = slowest.words > 0
                && slowest.push as f64 <= PUSH_BOUND_CYCLES_A_WORD * slowest.words as f64;
            for p in [labels.entry(o.label).or_default(), &mut all] {
                p.ops += 1;
                p.work += slowest.work;
                p.push += slowest.push;
                p.drain += slowest.drain;
                p.words += slowest.words;
                if bound {
                    p.bound += slowest.push;
                }
            }
        }
        let mut labels: Vec<_> = labels.into_iter().collect();
        labels.sort_by_key(|(_, p)| std::cmp::Reverse(p.bound));
        let pct = |a: u64, b: u64| 100.0 * a as f64 / b.max(1) as f64;
        let rate = |p: &Push| p.push as f64 / p.words.max(1) as f64;
        // A fused push takes 1.19 cycles a word where the runner takes 3.31 (`.ttinsn` experiment).
        let saved = |p: &Push| self.us(p.bound) * (1.0 - 1.19 / 3.31);
        let _ = writeln!(
            out,
            "\nThe math role on each op's slowest unit (its own totals, stamped by the mover after each \
             kernel): pushing {:.1} us of {:.1} us of work, {:.1} cycles a word; waiting after pushing \
             {:.1} us. Pushing near the runner's ~3.3 cycles a word is bound by the pushing (at most \
             {PUSH_BOUND_CYCLES_A_WORD} counts here: {:.1} us of it, {:.1}% of the work, and a fused push \
             would save ~{:.1} us); far more, the pushes waited on a full FIFO behind the backend.\n\n\
             | label | ops | slowest's work us | push us | words | cycles a word | push-bound us | \
             fused saves ~us | drain us |\n\
             |---|--:|--:|--:|--:|--:|--:|--:|--:|",
            self.us(all.push),
            self.us(all.work),
            rate(&all),
            self.us(all.drain),
            self.us(all.bound),
            pct(all.bound, all.work),
            saved(&all),
        );
        for (label, p) in labels {
            if p.push == 0 && p.drain == 0 {
                continue;
            }
            let _ = writeln!(
                out,
                "| {} | {} | {:.1} | {:.1} | {} | {:.1} | {:.1} | {:.1} | {:.1} |",
                if label.is_empty() {
                    "(unlabelled)"
                } else {
                    label
                },
                p.ops,
                self.us(p.work),
                self.us(p.push),
                p.words,
                rate(&p),
                self.us(p.bound),
                saved(&p),
                self.us(p.drain),
            );
        }
    }

    /// Every trace op's units, one line each, tab-separated: each unit's median work (from its last
    /// barrier's release to its arrival at the op's) and wait there, in microseconds.
    pub fn skew_tsv(&self, name: &dyn Fn(u64) -> String) -> String {
        let mut out = String::from(
            "source\tindex\tlabel\twhat\tunit\tsamples\twork_us\twait_us\tspan_mb\tahead_mb\tspan_gbps\tpush_us\tdrain_us\tpush_words\n",
        );
        for (&id, t) in &self.traces {
            let source = name(id);
            for (i, (o, units)) in t.ops.iter().zip(t.unit_medians()).enumerate() {
                for m in units {
                    let u = m.unit;
                    let span = o.unit_span_bytes(u);
                    let ahead = o.unit_bytes(u).map_or(0, |b| b.ahead);
                    let _ = writeln!(
                        out,
                        "{source}\t{i}\t{}\t{}\t{u}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{:.3}\t{:.3}\t{}",
                        o.label,
                        o.what,
                        m.samples,
                        self.us(m.work),
                        self.us(m.wait),
                        span as f64 / 1e6,
                        ahead as f64 / 1e6,
                        gbps(span, self.us(m.work)),
                        self.us(m.push),
                        self.us(m.drain),
                        m.words,
                    );
                }
            }
        }
        out
    }

    /// [`Detail::Full`]: every trace op's entries on unit 0 by kind, one line a kind, averaged over
    /// the replays that stamped the op ([`attribute_entries`]). Microseconds a replay. Empty
    /// without full detail.
    pub fn entries_tsv(&self, name: &dyn Fn(u64) -> String) -> String {
        let mut out = String::from(
            "source\tindex\tlabel\twhat\tunits\tsamples\tentry\tname\tcount\tus\tgap_us\n",
        );
        for (&id, t) in &self.traces {
            let source = name(id);
            for (i, (o, costs)) in t.ops.iter().zip(&t.entries).enumerate() {
                let n = t.samples[i].len();
                if n == 0 {
                    continue;
                }
                for (&op, c) in costs {
                    let _ = writeln!(
                        out,
                        "{source}\t{i}\t{}\t{}\t{}\t{n}\t{op:#x}\t{}\t{:.2}\t{:.3}\t{:.3}",
                        o.label,
                        o.what,
                        o.units,
                        crate::profile::op_name(op),
                        c.count as f64 / n as f64,
                        self.us(c.cycles) / n as f64,
                        self.us(c.gap) / n as f64,
                    );
                }
            }
        }
        out
    }

    /// The report as markdown: eager ops by label, then each trace by label and its costliest ops.
    /// `name` names a trace by its session id.
    pub fn summary(&self, name: &dyn Fn(u64) -> String) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "# Device profile\n\nTile clock {:.1} ticks/us. Times are on the card, from the data \
             movers' timestamps on unit 0 (`tt_kernels::profile_mode`).\n",
            self.ticks_per_us
        );
        if !self.eager.is_empty() {
            let rows = by_label(self.eager.iter().map(|s| (s.label, &s.time)));
            let total: u64 = self.eager.iter().map(|s| s.time.span).sum();
            let _ = writeln!(
                out,
                "## Eager ops\n\n{} ops, {:.1} ms on the card, each timed alone.\n",
                self.eager.len(),
                self.us(total) / 1e3
            );
            self.label_table(&mut out, &rows, "ops");
            out.push('\n');
        }
        if !self.eager_overflowed.is_empty() {
            let _ = writeln!(
                out,
                "Eager ops whose stamps outran a unit's buffer, not attributed: {}.\n",
                self.eager_overflowed
                    .iter()
                    .map(|(l, n)| format!("`{l}` x{n}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        for (&id, t) in &self.traces {
            let medians = t.medians();
            let sum: u64 = medians.iter().flatten().map(|m| m.span).sum();
            let _ = writeln!(
                out,
                "## Trace {}\n\n{} ops, {} replays ({} overflowed), {} ops covered; {:.1} us a \
                 replay (the covered ops' median spans, summed).\n",
                name(id),
                t.ops.len(),
                t.replays,
                t.overflowed,
                t.covered(),
                self.us(sum)
            );
            let _ = writeln!(
                out,
                "Windows sized from at most {} stamps for the whole trace on its busiest unit \
                 (largest op {}), {WINDOW_EVENTS} a window.\n",
                t.stamps.iter().sum::<u32>(),
                t.stamps.iter().max().copied().unwrap_or(0)
            );
            let rows = by_label(
                t.ops
                    .iter()
                    .zip(&medians)
                    .filter_map(|(o, m)| m.as_ref().map(|m| (o.label, m))),
            );
            self.label_table(&mut out, &rows, "ops");
            self.bytes_tables(&mut out, t, &medians);
            self.skew_table(&mut out, t);
            self.push_table(&mut out, t);
            let mut top: Vec<(usize, &OpTime)> = medians
                .iter()
                .enumerate()
                .filter_map(|(i, m)| m.as_ref().map(|m| (i, m)))
                .collect();
            top.sort_by_key(|(_, m)| std::cmp::Reverse(m.span));
            let _ = writeln!(
                out,
                "\nCostliest ops:\n\n| op | label | what | units | us | barrier | kernels | control | fetch | writer | prefetch | data |\n|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
            );
            for (i, m) in top.into_iter().take(25) {
                let o = &t.ops[i];
                let _ = writeln!(
                    out,
                    "| {i} | {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
                    o.label,
                    o.what,
                    o.units,
                    self.us(m.span),
                    self.us(m.barrier),
                    self.us(m.kernels),
                    self.us(m.control),
                    self.us(m.fetch),
                    self.us(m.writer),
                    self.us(m.prefetch),
                    self.us(m.data)
                );
            }
            out.push('\n');
        }
        out
    }

    /// Every op, one line each, tab-separated: eager ops in order, then each trace's ops by index
    /// (medians over their samples). Microseconds.
    pub fn tsv(&self, name: &dyn Fn(u64) -> String) -> String {
        let mut out = String::from(
            "source\tindex\tlabel\twhat\tunits\tsamples\tspan_us\tbarrier_us\tkernels_us\tcontrol_us\tdata_us\tother_us\tread_bytes\twrite_bytes\treads\twrites\ta_bytes\tb_bytes\tgbps\tfetch_us\twriter_us\tahead_bytes\tprefetch_us\tpush_us\tdrain_us\tpush_words\n",
        );
        let none = OpBytes::default();
        // The math role's totals: an eager op's on its longest unit, a trace op's on its slowest.
        let mut row = |source: &str,
                       i: usize,
                       o: (&str, &str, usize, &OpBytes),
                       n: usize,
                       t: &OpTime,
                       (push, drain, words): (u64, u64, u64)| {
            let b = o.3;
            let _ = writeln!(
                out,
                "{source}\t{i}\t{}\t{}\t{}\t{n}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.3}\t{:.3}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}",
                o.0,
                o.1,
                o.2,
                self.us(t.span),
                self.us(t.barrier),
                self.us(t.kernels),
                self.us(t.control),
                self.us(t.data),
                self.us(t.other),
                b.read,
                b.write,
                b.reads,
                b.writes,
                b.a,
                b.b,
                gbps(b.read + b.write - b.ahead, self.us(t.span)),
                self.us(t.fetch),
                self.us(t.writer),
                b.ahead,
                self.us(t.prefetch),
                self.us(push),
                self.us(drain),
                words,
            );
        };
        for (i, s) in self.eager.iter().enumerate() {
            let totals = (s.time.push, s.time.drain, s.time.words);
            row("eager", i, (s.label, s.what, 0, &none), 1, &s.time, totals);
        }
        for (&id, t) in &self.traces {
            let source = name(id);
            let units = t.unit_medians();
            for (i, (o, m)) in t.ops.iter().zip(t.medians()).enumerate() {
                if let Some(m) = m {
                    let totals = units[i]
                        .iter()
                        .max_by_key(|u| (u.work, u.unit))
                        .map_or((m.push, m.drain, m.words), |u| (u.push, u.drain, u.words));
                    row(
                        &source,
                        i,
                        (o.label, o.what, o.units, &o.bytes),
                        t.samples[i].len(),
                        &m,
                        totals,
                    );
                }
            }
        }
        out
    }

    /// A Perfetto/Chrome trace: eager ops one after another on one track, and each trace's ops one
    /// after another at their median spans on a track of its own, each span named by its label.
    pub fn chrome_trace(&self, name: &dyn Fn(u64) -> String) -> String {
        let mut events = Vec::new();
        let mut track = |pid: usize, title: &str, ops: Vec<(&str, &str, &OpTime)>| {
            events.push(format!(
                "{{\"ph\":\"M\",\"pid\":{pid},\"name\":\"process_name\",\"args\":{{\"name\":{title:?}}}}}"
            ));
            let mut at = 0.0;
            for (label, what, t) in ops {
                let dur = self.us(t.span);
                events.push(format!(
                    "{{\"ph\":\"X\",\"pid\":{pid},\"tid\":0,\"name\":{label:?},\"ts\":{at:.3},\"dur\":{dur:.3},\"args\":{{\"what\":{what:?},\"barrier_us\":{:.3},\"kernels_us\":{:.3},\"control_us\":{:.3},\"fetch_us\":{:.3},\"writer_us\":{:.3},\"prefetch_us\":{:.3},\"data_us\":{:.3}}}}}",
                    self.us(t.barrier),
                    self.us(t.kernels),
                    self.us(t.control),
                    self.us(t.fetch),
                    self.us(t.writer),
                    self.us(t.prefetch),
                    self.us(t.data)
                ));
                at += dur;
            }
        };
        track(
            0,
            "eager ops",
            self.eager
                .iter()
                .map(|s| (s.label, s.what, &s.time))
                .collect(),
        );
        for (k, (&id, t)) in self.traces.iter().enumerate() {
            let medians = t.medians();
            let ops = t
                .ops
                .iter()
                .zip(&medians)
                .filter_map(|(o, m)| m.as_ref().map(|m| (o.label, o.what, m)))
                .collect();
            track(k + 1, &format!("trace {}", name(id)), ops);
        }
        format!(
            "{{\"displayTimeUnit\":\"ns\",\"traceEvents\":[{}]}}",
            events.join(",")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tt_isa::dm::record;

    fn e(event: u32, detail: u32, cycles: u64) -> TraceEvent {
        TraceEvent {
            token: ev::token_with(ev::MOVER, event, detail),
            cycles,
        }
    }

    fn entry(op: u32, t0: u64, t1: u64) -> [TraceEvent; 2] {
        [e(ev::ENTRY_BEGIN, op, t0), e(ev::ENTRY_END, op, t1)]
    }

    /// Three ops of a replay, each a launch, a kernels' wait and a barrier, after a catch-up list.
    fn replay() -> Vec<TraceEvent> {
        let mut v = vec![e(ev::LIST_BEGIN, 1, 0)];
        v.extend(entry(op::BARRIER, 1, 3));
        v.push(e(ev::LIST_END, 1, 4));
        v.push(e(ev::LIST_BEGIN, 1, 100));
        let mut t = 100;
        for _ in 0..3 {
            v.extend(entry(op::LAUNCH, t + 2, t + 4));
            v.extend(entry(op::KERNEL_WAIT, t + 10, t + 40));
            v.extend(entry(op::BARRIER, t + 41, t + 50));
            t += 50;
        }
        v.push(e(ev::LIST_END, 1, t + 1));
        v
    }

    #[test]
    fn a_replay_splits_at_its_barriers_after_the_catch_up() {
        let ops = attribute_replay(&replay(), 0, Detail::Ops);
        assert_eq!(ops.len(), 3);
        for o in &ops {
            assert_eq!(o.span, 50);
            assert_eq!((o.control, o.kernels, o.barrier), (2, 30, 9));
            assert_eq!(o.data, 50 - 2 - 30 - 9, "the rest is data in Ops detail");
        }
    }

    #[test]
    fn a_window_past_the_start_drops_its_partial_first_op() {
        // Stamping began inside op 1 (lo = 1): its barrier opens op 2's span.
        let v: Vec<TraceEvent> = replay().into_iter().skip(5 + 6).collect();
        let ops = attribute_replay(&v, 1, Detail::Ops);
        assert_eq!(ops.len(), 1, "{ops:?}");
        assert_eq!(ops[0].span, 50);
    }

    #[test]
    fn entries_are_split_by_kind_with_the_gaps_before_them() {
        let ops = attribute_entries(&replay(), 0);
        assert_eq!(ops.len(), 3, "the ops attribute_replay returns");
        for (k, o) in ops.iter().enumerate() {
            let launch = o[&op::LAUNCH];
            // The first op's launch follows the list's start by 2, a later one its barrier by 2.
            assert_eq!(
                (launch.count, launch.cycles, launch.gap),
                (1, 2, 2),
                "op {k}"
            );
            assert_eq!(
                o[&op::KERNEL_WAIT],
                EntryCost {
                    count: 1,
                    cycles: 30,
                    gap: 6
                }
            );
            assert_eq!(
                o[&op::BARRIER],
                EntryCost {
                    count: 1,
                    cycles: 9,
                    gap: 1
                }
            );
            let all: u64 = o.values().map(|c| c.cycles + c.gap).sum();
            assert_eq!(all, 50, "the op's span, entries and gaps");
        }
        let v: Vec<TraceEvent> = replay().into_iter().skip(5 + 6).collect();
        assert_eq!(
            attribute_entries(&v, 1).len(),
            1,
            "a partial first op is dropped"
        );
    }

    /// A unit that runs ops 0 and 3 of five: its arrivals at 0 and 3, a wait standing for 1 and 2
    /// before 3, and one standing for 4 at the trace's end.
    fn sat_out() -> Vec<UnitBarrier> {
        let arrive = |k| UnitBarrier {
            arrival_of: Some(k),
            counts: 1,
        };
        let wait = |n| UnitBarrier {
            arrival_of: None,
            counts: n,
        };
        vec![arrive(0), wait(2), arrive(3), wait(1)]
    }

    #[test]
    fn a_units_stamps_are_told_apart_by_its_barriers() {
        // Whole: every barrier stamped, from the replay's list start at 100.
        let mut v = vec![e(ev::LIST_BEGIN, 1, 0)];
        v.extend(entry(op::BARRIER, 1, 3));
        v.push(e(ev::LIST_END, 1, 4));
        v.push(e(ev::LIST_BEGIN, 1, 100));
        v.extend(entry(op::BARRIER, 130, 150)); // arrives at 0
        v.extend(entry(op::BARRIER, 200, 201)); // waits for 1 and 2
        v.extend(entry(op::KERNEL_WAIT, 210, 240));
        v.extend(entry(op::BARRIER, 260, 300)); // arrives at 3
        v.extend(entry(op::BARRIER, 400, 401)); // waits for 4
        v.push(e(ev::LIST_END, 1, 402));
        let t = |work, wait| UnitTime {
            unit: 5,
            work,
            wait,
            push: 0,
            drain: 0,
            words: 0,
        };
        assert_eq!(
            unit_times(&v, &sat_out(), 5, 0, 0),
            Some(vec![(0, t(30, 20)), (3, t(59, 40))])
        );
        // A window of ops 2..4: the wait (counted 1 and 2) starts outside it, so op 3's work has
        // no start; the trace's last wait (count 4) is stamped.
        let mut w = vec![e(ev::LIST_BEGIN, 1, 100)];
        w.extend(entry(op::BARRIER, 260, 300));
        w.extend(entry(op::BARRIER, 400, 401));
        assert_eq!(unit_times(&w, &sat_out(), 5, 2, 5), Some(Vec::new()));
        // A window of ops 1..4: the wait is stamped, so op 3 is.
        let mut x = vec![e(ev::LIST_BEGIN, 1, 100)];
        x.extend(entry(op::BARRIER, 200, 201));
        x.extend(entry(op::BARRIER, 260, 300));
        x.extend(entry(op::BARRIER, 400, 401));
        assert_eq!(
            unit_times(&x, &sat_out(), 5, 1, 5),
            Some(vec![(3, t(59, 40))])
        );
        // Stamps that do not line up with the stream are refused, not misattributed.
        assert_eq!(unit_times(&w, &sat_out(), 5, 1, 5), None, "one stamp short");
        assert_eq!(unit_times(&x, &sat_out(), 5, 2, 5), None, "one stamp over");
    }

    #[test]
    fn the_math_roles_totals_go_to_the_op_whose_kernel_they_follow() {
        // A kernel's three totals, in units of 64.
        let totals = |push: u32, drain: u32, words: u32| {
            [
                (ev::ROLE_PUSH, push),
                (ev::ROLE_DRAIN, drain),
                (ev::ROLE_WORDS, words),
            ]
            .map(|(event, units)| TraceEvent {
                token: ev::token_with(ev::MOVER, event, units),
                cycles: 0,
            })
        };
        let mut v = vec![e(ev::LIST_BEGIN, 1, 100)];
        v.extend(entry(op::KERNEL_WAIT, 110, 140));
        v.extend(totals(3, 1, 1));
        v.extend(entry(op::BARRIER, 150, 160)); // arrives at 0
        v.extend(entry(op::BARRIER, 200, 201)); // waits for 1 and 2
        v.extend(entry(op::KERNEL_WAIT, 210, 240));
        v.extend(totals(5, 0, 2));
        v.extend(entry(op::KERNEL_WAIT, 245, 250));
        v.extend(totals(2, 2, 1));
        v.extend(entry(op::BARRIER, 260, 300)); // arrives at 3
        v.extend(entry(op::BARRIER, 400, 401)); // waits for 4
        let got = unit_times(&v, &sat_out(), 5, 0, 0).unwrap();
        let sums: Vec<_> = got
            .iter()
            .map(|(k, t)| (*k, t.push, t.drain, t.words))
            .collect();
        assert_eq!(sums, vec![(0, 3 * 64, 64, 64), (3, 7 * 64, 2 * 64, 3 * 64)]);
        // Unit 0's own op times carry them too.
        let ops = attribute_replay(&v, 0, Detail::Ops);
        assert_eq!((ops[0].push, ops[0].drain, ops[0].words), (3 * 64, 64, 64));
        // A wait now stamps three more events when the totals are on.
        let on = ProfileOptions::default();
        let off = ProfileOptions { push: false, ..on };
        assert_eq!(
            (on.stamps(op::KERNEL_WAIT), off.stamps(op::KERNEL_WAIT)),
            (6, 3)
        );
        assert_eq!(on.stamps(op::LAUNCH), 3);
    }

    #[test]
    fn full_detail_times_data_entries_and_keeps_the_gaps_apart() {
        let mut v = vec![e(ev::LIST_BEGIN, 3, 0)];
        v.extend(entry(record::GATHER, 1, 21));
        v.extend(entry(op::KERNEL_WAIT, 22, 52));
        v.push(e(ev::LIST_END, 3, 60));
        let t = attribute_eager(&v, Detail::Full).unwrap();
        assert_eq!((t.span, t.data, t.kernels, t.other), (60, 20, 30, 10));
        let t = attribute_eager(&v, Detail::Ops).unwrap();
        assert_eq!(t.data, 30, "Ops detail charges the unstamped rest to data");
    }

    #[test]
    fn windows_cover_a_trace_and_fit_the_buffer() {
        let mut p = Profiler::new(ProfileOptions::default(), 0);
        let n = 500;
        p.know_trace(7, || {
            (
                (0..n)
                    .map(|_| OpInfo {
                        label: "x",
                        what: "y",
                        units: 1,
                        bytes: OpBytes::default(),
                        unit_bytes: Vec::new(),
                    })
                    .collect(),
                vec![9; n],
                true,
                Vec::new(),
            )
        });
        let mut seen = vec![false; n];
        let mut replays = 0;
        loop {
            let w = p.next_window(7);
            let (lo, hi) = ((w & 0xFFFF) as usize, (w >> 16) as usize);
            let first = if lo == 0 && seen.iter().all(|s| !s) {
                0
            } else {
                lo + 1
            };
            assert!(
                9 * (hi - lo) as u32 <= WINDOW_EVENTS,
                "window {lo}..{hi} overflows"
            );
            for s in &mut seen[first..hi] {
                *s = true;
            }
            replays += 1;
            p.pending = None;
            if hi == n {
                break;
            }
        }
        assert!(seen.iter().all(|&s| s), "every op covered");
        assert!(replays >= n * 9 / WINDOW_EVENTS as usize);
    }
}
