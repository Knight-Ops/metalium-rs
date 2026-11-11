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
//!
//! **The card's GDDR reads over a replay** ([`unit_reads`], [`card_reads`]): every unit's B marks
//! the GDDR bytes it has issued (`tt_isa::mailbox::trace::READ_BYTES`) at each barrier's arrival and
//! release, and when a queued prefetch starts and when it has issued its last read. Between two
//! marks a unit's bytes are spread evenly; each unit's clock is put on unit 0's by the barrier
//! releases they share; the sum over units is the card's read rate in each op's span on unit 0,
//! in [`READ_BINS`] parts of it. Counted at issue, not landing, so a read is charged up to its
//! latency early (an op's first part can read above the card's rate); what B fetches of the
//! trace's own stream is counted too, and reads of other tiles' L1 are not.
//!
//! **A streamed pair's batches, unit by unit** ([`Detail::Pipeline`], [`unit_pipes`]): B's input
//! credits, which [`Detail::Ops`] already stamps (a pair's reader runs them as Reserve, Push,
//! Reserve, Push), give each batch's wait for a slot, its gather's issue and its wait for the reads
//! to land; T0's and T2's batch tables (`tt_isa::mailbox::BATCH_TIMES`), which B stamps after the
//! kernel, give when T0 waited for the batch, saw it and popped it, and when T2 waited for an
//! output slot, got it and pushed. Every time is on the unit's own clock, so no two tiles' clocks
//! are compared ([`PipeStats`]).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use tt_device::trace::TraceEvent;
use tt_isa::dm::op;
use tt_isa::mailbox::trace as ev;

use crate::profile::{Event, RoleTotal};

/// The `READ_BYTES` marks a prefetch `GATHER` (`tt_isa::dm::record::GATHER_PREFETCH`) stamps: when
/// it is queued and when its last read is issued, which may be under a later op.
pub const PREFETCH_MARKS: u32 = 2;

/// Parts of an op's span the card's read rate is told in ([`TraceSamples::reads`]).
pub const READ_BINS: usize = 8;

/// How much of an op the movers stamp.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Detail {
    /// Only what bounds an op's phases (`tt_isa::dm::trace_mode::OPS`): ~8 events an op.
    #[default]
    Ops = 0,
    /// Every entry (`trace_mode::FULL`): data entries timed one by one, ~25 events an op, so a
    /// window holds about a third as many ops.
    Full = 1,
    /// [`Detail::Ops`], and each streamed pair's batches on every unit ([`unit_pipes`]): T0 and T2
    /// keep batch tables (`tt_isa::mailbox::trace_word::BATCHES`) that B stamps after the kernel,
    /// six events a batch, so a window holds fewer ops. The movers stamp as in `Ops`.
    Pipeline = 2,
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
            Detail::Ops | Detail::Pipeline => tt_isa::dm::trace_mode::OPS,
            Detail::Full => tt_isa::dm::trace_mode::FULL,
        }
    }

    /// Events an entry headed by `op` may stamp, at most: its begin and end, the one event a
    /// launch (`KICK`) or a kernels' wait (`ROLES_DONE`) adds inside it, and after a wait the math
    /// role's two totals ([`ProfileOptions::push`]).
    /// A `BARRIER` adds its two `READ_BYTES` marks (as does a prefetch `GATHER`, which the
    /// caller counts: [`PREFETCH_MARKS`]). In [`Detail::Pipeline`] a wait adds its batch tables'
    /// base (`BATCH_BASE`); the caller counts their times, six a batch.
    pub fn stamps(&self, op: u32) -> u32 {
        let waits = matches!(op, op::KERNEL | op::KERNEL_WAIT);
        let marks = if op == op::BARRIER { 2 } else { 0 };
        let base = u32::from(waits && self.detail == Detail::Pipeline);
        marks
            + base
            + match self.detail {
                Detail::Ops | Detail::Pipeline if !tt_isa::dm::trace_mode::ops_stamped(op) => 0,
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

/// The card's GDDR6 read rate, what a step's bytes are held against: its peak, 512 GB/s, which
/// reads reach on our movers (511.2 with one reader beside each channel and long entries,
/// performance scoreboard, tt-metal's adjacent gate). Not the 469 of every tile reading its own
/// share (the scoreboard's "120 tiles" row): that is a limit of how we read, not of the card.
pub const CARD_READ_GBPS: f64 = 512.0;

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
    /// Of `read`, the bytes by GDDR channel and the NoC #0 endpoint (port) the read goes to, as
    /// the mover maps its rotation (`DramChannel::port_for`): where an op's reads land
    /// (`decode-to-bandwidth.md` Phase 10, A0). A host-encoded read is not placed.
    pub endpoint: [[u64; PORTS]; CHANNELS],
    /// Of `endpoint`, what lists read ahead of the op (as `ahead` is of `read`).
    pub ahead_endpoint: [[u64; PORTS]; CHANNELS],
}

/// GDDR channels and the endpoints (ports) a channel has.
const CHANNELS: usize = tt_isa::dram::CHANNELS as usize;
const PORTS: usize = tt_isa::dram::PORTS as usize;

impl OpBytes {
    pub fn add(&mut self, o: &OpBytes) {
        self.read += o.read;
        self.write += o.write;
        self.reads += o.reads;
        self.writes += o.writes;
        self.a += o.a;
        self.b += o.b;
        self.ahead += o.ahead;
        for c in 0..CHANNELS {
            for p in 0..PORTS {
                self.endpoint[c][p] += o.endpoint[c][p];
                self.ahead_endpoint[c][p] += o.ahead_endpoint[c][p];
            }
        }
    }

    /// Of `endpoint`, what was read in the op's own span: less what was read ahead.
    pub fn span_endpoint(&self) -> [[u64; PORTS]; CHANNELS] {
        let mut out = self.endpoint;
        for (o, a) in out.iter_mut().zip(&self.ahead_endpoint) {
            for (o, a) in o.iter_mut().zip(a) {
                *o -= a;
            }
        }
        out
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
                    let slot =
                        record::gather_a_tile_format(head[1]).map_or(TILE_SLOT, |f| f.slot_bytes());
                    a_at..a_at + rows * kt * slot
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
                let all = tt_isa::dram::Dram::from_usable_mask(u8::MAX);
                if let Some(ch) = u8::try_from(e[1]).ok().and_then(|c| all.channel(c)) {
                    let port =
                        ch.port_for(tt_isa::noc::niu::Niu::Noc0, (e[2] % PORTS as u32) as u8);
                    self.endpoint[ch.index() as usize][port as usize] += e[5] as u64;
                }
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
            Detail::Ops | Detail::Pipeline => self.data = self.span.saturating_sub(known),
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

/// One unit's GDDR reads in a replay ([`unit_reads`]), on its own clock.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UnitReads {
    /// `(from, to, bytes)`: what B issued between two of its `READ_BYTES` marks. The first mark
    /// of the replay counts from one before it that was not stamped, and is left out.
    pub spans: Vec<(u64, u64, u64)>,
    /// Its barriers' releases: `(the trace's barrier it waited for, when)`. An arrival waits for
    /// its own; a wait standing for `n` waits for the last of them.
    pub releases: Vec<(u32, u64)>,
    /// Its arrivals: `(the op it ran a share of, when it arrived at the op's barrier)`, the end of
    /// its work there.
    pub arrivals: Vec<(u32, u64)>,
    /// When the replay began (its outermost list), for its first op.
    pub begin: u64,
}

/// One replay's stamps on a unit whose barriers are `seq`, stamped within the window `lo..hi` (as
/// [`unit_times`]): its `READ_BYTES` marks as spans of time and its barrier releases. `None` if the
/// stamps and `seq` do not line up.
pub fn unit_reads(
    events: &[TraceEvent],
    seq: &[UnitBarrier],
    lo: u32,
    hi: u32,
) -> Option<UnitReads> {
    let stamps: Vec<(Event, u64)> = events.iter().filter_map(mover).collect();
    let Some(start) = stamps
        .iter()
        .rposition(|(e, _)| matches!(e, Event::List { begin: true, .. }))
    else {
        return Some(UnitReads::default());
    };
    let mut out = UnitReads {
        begin: stamps[start].1,
        ..Default::default()
    };
    let mut last_mark = None;
    let (mut begins, mut ends) = (Vec::new(), Vec::new());
    for &(event, t) in &stamps[start..] {
        match event {
            Event::ReadBytes { bytes } => {
                if let Some(from) = last_mark {
                    out.spans.push((from, t, bytes));
                }
                last_mark = Some(t);
            }
            Event::Entry {
                begin,
                op: op::BARRIER,
            } => {
                if begin {
                    begins.push(t);
                } else {
                    ends.push(t);
                }
            }
            _ => {}
        }
    }
    let (mut begins, mut ends) = (begins.into_iter(), ends.into_iter());
    let mut count = 0u32;
    for b in seq {
        if hi == 0 || (lo <= count && count < hi) {
            let arrive = begins.next()?;
            out.releases.push((count + b.counts - 1, ends.next()?));
            if let Some(k) = b.arrival_of {
                out.arrivals.push((k, arrive));
            }
        }
        count += b.counts;
    }
    (begins.next().is_none() && ends.next().is_none()).then_some(out)
}

/// The card's GDDR reads in one op's span on unit 0, in [`READ_BINS`] equal parts of it
/// ([`card_reads`]): who issued them, and how many of the op's own units were still working.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadBins {
    /// Every unit's bytes.
    pub all: [u64; READ_BINS],
    /// Of `all`, the bytes of units that ran a share of the op (arrived at its barrier): its own
    /// reads. The rest is units reading ahead for later ops.
    pub own: [u64; READ_BINS],
    /// The op's own units whose arrival lies past the part's start: still at work in it. A
    /// product's tail is the parts where few are.
    pub working: [u16; READ_BINS],
}

/// The card's GDDR reads in one replay: each unit's [`UnitReads`] (by unit, `None` where unknown)
/// put on unit 0's clock and summed into [`READ_BINS`] parts of each op's span there. Unit 0's
/// barriers are `seq0`; the ops are [`attribute_replay`]'s, from `first` (op 0 with `lo == 0`,
/// timed from the replay's start). A unit's clock is unit 0's plus the median of its releases'
/// offsets from unit 0's of the same barrier (a release reaches the tiles within a multicast's
/// flight).
pub fn card_reads(
    units: &[Option<UnitReads>],
    seq0: &[UnitBarrier],
    first: usize,
    lo: u32,
) -> Vec<(usize, ReadBins)> {
    let Some(Some(u0)) = units.first() else {
        return Vec::new();
    };
    // Unit 0's release of each barrier it waited for, and each op's end there.
    let release0: BTreeMap<u32, u64> = u0.releases.iter().copied().collect();
    let mut ends: BTreeMap<usize, u64> = BTreeMap::new();
    let mut count = 0u32;
    for b in seq0 {
        if let (Some(k), Some(&t)) = (b.arrival_of, release0.get(&(count + b.counts - 1))) {
            ends.insert(k as usize, t);
        }
        count += b.counts;
    }
    // Each op's span on unit 0: from the op before's end (the replay's start for op 0).
    let mut spans: Vec<(usize, i64, i64)> = Vec::new();
    for (&k, &end) in &ends {
        if k < first {
            continue;
        }
        let begin = match k.checked_sub(1) {
            Some(prev) => ends.get(&prev).copied(),
            None if lo == 0 => Some(u0.begin),
            None => None,
        };
        if let Some(begin) = begin.filter(|&b| b <= end) {
            spans.push((k, begin as i64, end as i64));
        }
    }
    let mut out: Vec<(usize, ReadBins)> = spans
        .iter()
        .map(|&(k, ..)| (k, ReadBins::default()))
        .collect();
    for u in units.iter().flatten() {
        let mut offsets: Vec<i64> = u
            .releases
            .iter()
            .filter_map(|&(n, t)| release0.get(&n).map(|&t0| t as i64 - t0 as i64))
            .collect();
        if offsets.is_empty() {
            continue;
        }
        offsets.sort_unstable();
        let offset = offsets[offsets.len() / 2];
        let arrivals: BTreeMap<usize, i64> = u
            .arrivals
            .iter()
            .map(|&(k, t)| (k as usize, t as i64 - offset))
            .collect();
        for (&(k, begin, end), (_, bins)) in spans.iter().zip(out.iter_mut()) {
            let width = (end - begin).max(1) as f64 / READ_BINS as f64;
            let own = arrivals.get(&k).copied();
            if let Some(arrive) = own {
                for (i, n) in bins.working.iter_mut().enumerate() {
                    if arrive as f64 > begin as f64 + i as f64 * width {
                        *n += 1;
                    }
                }
            }
            for &(from, to, bytes) in &u.spans {
                let (a, b) = ((from as i64 - offset) as f64, (to as i64 - offset) as f64);
                if b < begin as f64 || a > end as f64 {
                    continue;
                }
                for i in 0..READ_BINS {
                    let lo = begin as f64 + i as f64 * width;
                    let hi = lo + width;
                    let share = if b > a {
                        (hi.min(b) - lo.max(a)).max(0.0) / (b - a)
                    } else if (lo..hi).contains(&b) {
                        1.0
                    } else {
                        0.0
                    };
                    let part = (bytes as f64 * share).round() as u64;
                    bins.all[i] += part;
                    if own.is_some() {
                        bins.own[i] += part;
                    }
                }
            }
        }
    }
    out
}

/// One batch of a streamed pair on one unit ([`unit_pipes`]): when each step of it happened, in
/// tile cycles on the unit's own clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Batch {
    /// B began its input reserve: it wanted a slot for the batch.
    pub reserve: u64,
    /// The reserve returned: a slot was free (T0 had popped the batch that held it).
    pub granted: u64,
    /// B began its input push: the gather had issued the batch's last read (each waited for room
    /// under the stream's in-flight cap, so the earlier ones have landed).
    pub issued: u64,
    /// The push returned: the batch's reads had landed, and T0 may see it.
    pub landed: u64,
    /// T0 entering its input wait, the wait returning, and its pop (the batch unpacked).
    pub t0: [u64; 3],
    /// T2 entering its output reserve, the reserve returning, and its push (the batch packed).
    pub t2: [u64; 3],
}

impl Batch {
    /// B's reads of the batch: from its grant to their landing.
    pub fn read(&self) -> (u64, u64) {
        (self.granted, self.landed)
    }

    /// The batch's kernel: from T0 seeing it to T2 pushing it.
    pub fn kernel(&self) -> (u64, u64) {
        (self.t0[1], self.t2[2])
    }
}

/// One unit's share of one op in a replay, from its streamed batches ([`unit_pipes`]), in tile
/// cycles. B's figures (`ring`, `issue`, `land`) and the roles' (`starve`, `kernel`, `out_wait`)
/// run on different cores at once, so the shares of `work` do not sum to one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipeStats {
    pub batches: u64,
    /// From the unit's previous barrier's release to its arrival at the op's.
    pub work: u64,
    /// Before the first batch's reserve: the op's control and anything before its pair.
    pub head: u64,
    /// B waiting for a slot ([`Batch::reserve`] to `granted`): its next reads not issued, the
    /// ring full of batches T0 had not popped.
    pub ring: u64,
    /// B running the gather (`granted` to `issued`): issuing the batch's reads, each waiting for
    /// room under the stream's in-flight cap (`dm::STREAM_CAP`), so most of the reads' flight is
    /// here, not in `land`.
    pub issue: u64,
    /// B waiting for the batch's last reads to land (`issued` to `landed`).
    pub land: u64,
    /// T0 waiting for a batch's data (its input wait).
    pub starve: u64,
    /// The batches' kernels, T0 seeing each to T2 pushing it ([`Batch::kernel`]).
    pub kernel: u64,
    /// T2 waiting for an output slot: NC behind on its writes.
    pub out_wait: u64,
    /// After the last batch's push to the barrier: the kernels' wait, NC's last writes.
    pub tail: u64,
    /// Reads in flight while a kernel ran: the overlap of the batches' [`Batch::read`] and
    /// [`Batch::kernel`] intervals.
    pub overlap: u64,
    /// From the first batch's grant to the last batch's push.
    pub pipe: u64,
    /// `pipe` had every batch's reads overlapped the kernel before (`r0 + Σ max(r_i, k_(i-1)) +
    /// k_last`): what a deeper ring could reach at the same read and kernel times.
    pub ideal: u64,
}

impl PipeStats {
    /// A unit's batches for an op it worked on from `since` to `arrive`.
    pub fn of(since: u64, arrive: u64, batches: &[Batch]) -> PipeStats {
        let (Some(first), Some(last)) = (batches.first(), batches.last()) else {
            return PipeStats {
                work: arrive.saturating_sub(since),
                ..Default::default()
            };
        };
        let len = |(a, b): (u64, u64)| b.saturating_sub(a);
        let mut s = PipeStats {
            batches: batches.len() as u64,
            work: arrive.saturating_sub(since),
            head: first.reserve.saturating_sub(since),
            tail: arrive.saturating_sub(last.t2[2]),
            pipe: last.t2[2].saturating_sub(first.granted),
            ..Default::default()
        };
        let mut ideal = len(first.read());
        for (i, b) in batches.iter().enumerate() {
            s.ring += b.granted.saturating_sub(b.reserve);
            s.issue += b.issued.saturating_sub(b.granted);
            s.land += b.landed.saturating_sub(b.issued);
            s.starve += b.t0[1].saturating_sub(b.t0[0]);
            s.kernel += len(b.kernel());
            s.out_wait += b.t2[1].saturating_sub(b.t2[0]);
            if i > 0 {
                ideal += len(b.read()).max(len(batches[i - 1].kernel()));
            }
            for k in batches {
                let (r, q) = (b.read(), k.kernel());
                s.overlap += r.1.min(q.1).saturating_sub(r.0.max(q.0));
            }
        }
        s.ideal = ideal + len(last.kernel());
        s
    }

    pub fn add(&mut self, o: &PipeStats) {
        self.batches += o.batches;
        self.work += o.work;
        self.head += o.head;
        self.ring += o.ring;
        self.issue += o.issue;
        self.land += o.land;
        self.starve += o.starve;
        self.kernel += o.kernel;
        self.out_wait += o.out_wait;
        self.tail += o.tail;
        self.overlap += o.overlap;
        self.pipe += o.pipe;
        self.ideal += o.ideal;
    }
}

/// One unit's batches on one op: `(unit, since, arrive, batches)`, on the unit's clock (`since`
/// its previous barrier's release, `arrive` its arrival at the op's).
pub type UnitBatches = (u16, u64, u64, Vec<Batch>);

/// One op's [`PipeStats`] in one replay: its slowest unit's (most `work`), and the sum over the
/// `units` that streamed batches for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipeSample {
    pub slowest: PipeStats,
    pub sum: PipeStats,
    pub units: u64,
}

/// Why [`unit_pipes`] could not tell a pair's batches, an index into [`UnitPipes::dropped`]: its
/// reader stamped fewer input credits than two a batch of its table, or more; its table was
/// truncated (more batches than `mailbox::BATCH_TIMES_MAX`) or its roles' counts differed; it was
/// not in B's batch log (the log was full); another table began before its times were all read;
/// or the join broke the credit protocol's order ([`in_protocol_order`]).
pub const DROP_REASONS: [&str; 7] = [
    "fewer credits",
    "more credits",
    "truncated",
    "mismatched",
    "not in the log",
    "interrupted",
    "disordered",
];

/// How far apart two cores' times may stand against the credit protocol's order and still be
/// taken as in order, in cycles. B stamps a credit's end after publishing it, so the role it
/// frees may act before that stamp: by 6 cycles on card 0 (`1793635382`); the roles' own times
/// are exact reads of the same counter. 64 cycles (~47 ns) is a tenth of a stamp's spacing.
pub const CROSS_CORE_SLACK: u64 = 64;

/// The host's half of the batch log (`tt_isa::dm::BATCH_LOG`): unit `u`'s stamps since its last
/// drain, each `BATCH_BASE` replaced by its record from `log` (the log's words after its count)
/// as a `BATCH_TABLE` and its `BATCH_TIME`s. A record's clock reads are the low word of the
/// counter the base was stamped with, so each is placed on the full counter exactly, counting back
/// from the base. A base whose record `log` lacks stays, and is dropped as not in the log. A
/// base naming a record already spliced is a copy the timestamper repeated (each record is
/// named once a drain): it becomes a `BATCH_REPEAT`, which [`unit_pipes`] counts.
pub fn splice_batch_log(events: Vec<TraceEvent>, log: &[u32]) -> Vec<TraceEvent> {
    use tt_isa::mailbox::{trace as ev, BATCH_TIME_WORDS};
    let mut out = Vec::with_capacity(events.len());
    let mut named = std::collections::BTreeSet::new();
    for e in events {
        let record = match Event::decode(e.token) {
            Event::BatchBase {
                record,
                log_full: false,
            } if (e.token >> 8) & 0xF == ev::MOVER => record as usize,
            _ => {
                out.push(e);
                continue;
            }
        };
        if !named.insert(record) {
            out.push(TraceEvent {
                token: ev::token_with(ev::MOVER, ev::BATCH_REPEAT, 0),
                cycles: e.cycles,
            });
            continue;
        }
        let Some(&header) = log.get(record) else {
            out.push(e);
            continue;
        };
        let n = (header & (ev::BATCH_TRUNCATED - 1)) as usize;
        let Some(times) = log.get(record + 1..record + 1 + 2 * BATCH_TIME_WORDS as usize * n)
        else {
            out.push(e);
            continue;
        };
        out.push(TraceEvent {
            token: ev::token_with(ev::MOVER, ev::BATCH_TABLE, header),
            cycles: e.cycles,
        });
        let low = e.cycles as u32;
        out.extend(times.iter().map(|&t| TraceEvent {
            token: ev::token_with(ev::MOVER, ev::BATCH_TIME, 0),
            cycles: e.cycles.saturating_sub(u64::from(low.wrapping_sub(t))),
        }));
    }
    out
}

/// What a pair's batch log record costs a window, in stamps: its `words`, scaled so a window's
/// records fit the log as its stamps fit the buffer (`tt_isa::dm::BATCH_LOG_WORDS` against
/// [`WINDOW_EVENTS`]).
pub fn batch_log_cost(words: u32) -> u32 {
    (u64::from(words) * u64::from(WINDOW_EVENTS))
        .div_ceil(u64::from(tt_isa::dm::BATCH_LOG_WORDS - 1)) as u32
}

/// Whether a pair's joined batches keep the credit protocol's order, which they always do if the
/// join is right ([`CROSS_CORE_SLACK`] across cores): each core's own steps in turn; a batch
/// landed before T0 saw it, and T0 saw it before T2 pushed it; each batch's T0 and T2 began after
/// they finished the one before (every role retires a batch before the next).
pub fn in_protocol_order(batches: &[Batch]) -> bool {
    let slack = CROSS_CORE_SLACK;
    batches.iter().all(|b| {
        b.reserve <= b.granted
            && b.granted <= b.issued
            && b.issued <= b.landed
            && b.t0[0] <= b.t0[1]
            && b.t0[1] <= b.t0[2]
            && b.t2[0] <= b.t2[1]
            && b.t2[1] <= b.t2[2]
            && b.landed <= b.t0[1] + slack
            && b.t0[1] <= b.t2[2] + slack
    }) && batches.windows(2).all(|w| {
        w[0].t0[2] <= w[1].t0[0] && w[0].t2[2] <= w[1].t2[0] && w[0].landed <= w[1].reserve
    })
}

/// What [`unit_pipes`] found on one unit in one replay: each op it streamed batches for, as `(op,
/// since, arrive, batches)` on the unit's clock; and the pairs whose batches it could not tell,
/// by reason ([`DROP_REASONS`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UnitPipes {
    pub ops: Vec<(usize, u64, u64, Vec<Batch>)>,
    /// Pairs told: their batches are in `ops`.
    pub told: u32,
    /// Bases the timestamper repeated ([`splice_batch_log`]), not pairs.
    pub repeats: u32,
    pub dropped: [u32; DROP_REASONS.len()],
    /// Of `dropped`, those within an op's work on the unit: `(op, pairs)`.
    pub dropped_ops: Vec<(usize, u32)>,
}

/// [`Detail::Pipeline`]: one replay's stamps on a unit whose barriers are `seq`, stamped within the
/// window `lo..hi` (as [`unit_times`]), as each op's streamed batches. A pair's reader stamps its
/// input credits as `CB` entries, a reserve then a push for each batch, after its `LAUNCH`; after
/// its kernel B logs the roles' batch tables, which [`splice_batch_log`] has put in the stream
/// (`Event::BatchTable`, then six `Event::BatchTime`s a batch: T0's three, then T2's) and which
/// pair with those credits in order. `None` if the barrier stamps and `seq` do not line up.
pub fn unit_pipes(
    events: &[TraceEvent],
    seq: &[UnitBarrier],
    lo: u32,
    hi: u32,
) -> Option<UnitPipes> {
    let stamps: Vec<(Event, u64)> = events.iter().filter_map(mover).collect();
    let Some(start) = stamps
        .iter()
        .rposition(|(e, _)| matches!(e, Event::List { begin: true, .. }))
    else {
        return Some(UnitPipes::default());
    };
    let words = tt_isa::mailbox::BATCH_TIME_WORDS as usize;
    let mut out = UnitPipes::default();
    // The current pair's credits, `(begin, end)`, and the batches since the last barrier.
    let mut credits: Vec<(u64, u64)> = Vec::new();
    let mut open_credit = None;
    let mut batches: Vec<Batch> = Vec::new();
    let mut drops = 0u32;
    // A table being read: its batches, why it cannot be used (if it cannot), and its times so far.
    let mut table: Option<(usize, Option<usize>, Vec<u64>)> = None;
    let mut pairs = Vec::new();
    let mut open = None;
    for &(event, t) in &stamps[start..] {
        match event {
            Event::Entry {
                begin: true,
                op: op::LAUNCH,
            } => credits.clear(),
            Event::Entry {
                begin: true,
                op: op::CB,
            } => open_credit = Some(t),
            Event::Entry {
                begin: false,
                op: op::CB,
            } => {
                if let Some(b) = open_credit.take() {
                    credits.push((b, t));
                }
            }
            Event::BatchBase { .. } => {
                // Not replaced by its record: not in the log.
                out.dropped[4] += 1;
                drops += 1;
            }
            Event::BatchRepeat => out.repeats += 1,
            Event::BatchTable {
                batches: n,
                truncated,
                mismatched,
            } => {
                if table.take().is_some() {
                    out.dropped[5] += 1;
                    drops += 1;
                }
                let want = 2 * n as usize;
                if n == 0 {
                    // No times follow: one role logged none (its count was zero).
                    out.dropped[3] += 1;
                    drops += 1;
                    credits.clear();
                    continue;
                }
                let why = if truncated {
                    Some(2)
                } else if mismatched {
                    Some(3)
                } else if credits.len() < want {
                    Some(0)
                } else if credits.len() > want {
                    Some(1)
                } else {
                    None
                };
                table = Some((n as usize, why, Vec::new()));
            }
            Event::BatchTime => {
                let Some((n, _, times)) = table.as_mut() else {
                    continue;
                };
                times.push(t);
                if times.len() == 2 * words * *n {
                    let (n, why, times) = table.take().expect("just matched");
                    let mut pair = Vec::with_capacity(n);
                    if why.is_none() {
                        for i in 0..n {
                            let (r, p) = (credits[2 * i], credits[2 * i + 1]);
                            let at = |k: usize| times[2 * words * i + k];
                            pair.push(Batch {
                                reserve: r.0,
                                granted: r.1,
                                issued: p.0,
                                landed: p.1,
                                t0: [at(0), at(1), at(2)],
                                t2: [at(3), at(4), at(5)],
                            });
                        }
                    }
                    let why = why.or_else(|| (!in_protocol_order(&pair)).then_some(6));
                    if let Some(why) = why {
                        out.dropped[why] += 1;
                        drops += 1;
                    } else {
                        out.told += 1;
                        batches.extend(pair);
                    }
                    credits.clear();
                }
            }
            Event::Entry {
                begin: true,
                op: op::BARRIER,
            } => open = Some((t, std::mem::take(&mut batches), std::mem::take(&mut drops))),
            Event::Entry {
                begin: false,
                op: op::BARRIER,
            } => {
                let (arrive, b, d) = open.take()?;
                pairs.push((arrive, t, b, d));
            }
            _ => {}
        }
    }
    let mut pairs = pairs.into_iter();
    let mut since = (lo == 0).then_some(stamps[start].1);
    let mut count = 0u32;
    for b in seq {
        if hi == 0 || (lo <= count && count < hi) {
            let (arrive, release, batches, drops) = pairs.next()?;
            if let (Some(k), Some(s)) = (b.arrival_of, since) {
                if !batches.is_empty() {
                    out.ops.push((k as usize, s, arrive, batches));
                }
                if drops > 0 {
                    out.dropped_ops.push((k as usize, drops));
                }
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
    /// The 10th and 90th percentiles of `work` over the replays ([`quantile`]): how much a unit's
    /// own time moves from replay to replay, beside how far units sit from each other.
    pub work_p10: u64,
    pub work_p90: u64,
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
    /// Per op, once per replay that stamped it, the GDDR bytes every unit's B issued in each of
    /// [`READ_BINS`] equal parts of the op's span on unit 0, and whose ([`card_reads`]).
    pub reads: Vec<Vec<ReadBins>>,
    /// [`Detail::Pipeline`]: per op, once per replay that stamped it, its units' streamed batches
    /// ([`unit_pipes`]).
    pub pipes: Vec<Vec<PipeSample>>,
    /// [`Detail::Pipeline`]: per op, the first replay's batches on every unit that streamed some,
    /// `(unit, since, arrive, batches)`.
    pub pipe_batches: Vec<Vec<UnitBatches>>,
    /// [`Detail::Pipeline`]: pairs whose batches could not be told, by reason
    /// ([`UnitPipes::dropped`], [`DROP_REASONS`]).
    pub pipe_dropped: [u32; DROP_REASONS.len()],
    /// [`Detail::Pipeline`]: per op, of `pipe_dropped`, the pairs within its units' work.
    pub pipe_dropped_ops: Vec<u32>,
    /// [`Detail::Pipeline`]: pairs told, summed over units and replays ([`UnitPipes::told`]).
    pub pipe_told: u64,
    /// [`Detail::Pipeline`]: batch bases the timestamper repeated ([`UnitPipes::repeats`]).
    pub pipe_repeats: u64,
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
    /// Every replay, overflowed or not, in order ([`ReplayRecord`]).
    pub replay_records: Vec<ReplayRecord>,
}

/// One replay of a trace as a whole: its wall time on the card beside the spans it attributed, and
/// the host's time around it. What tells the card's time from the host's when a trace's covered
/// spans fall short of its host-timed replay.
#[derive(Clone, Debug, Default)]
pub struct ReplayRecord {
    /// The window it stamped (`lo..hi`, `hi == 0` for all).
    pub lo: u32,
    pub hi: u32,
    /// The window's estimated stamps on its busiest unit ([`ProfileOptions::stamps`]).
    pub estimated: u32,
    /// Units whose stamps outran their buffer: nothing of this replay was attributed.
    pub overflowed: Vec<u16>,
    /// Unit 0's replay list, its begin to its end (cycles): the trace on the card. `None` when
    /// unit 0's stamps were lost.
    pub wall: Option<u64>,
    /// The longest replay list over every unit whose stamps were kept, `(unit, cycles)`.
    pub wall_max: Option<(u16, u64)>,
    /// Unit 0's lists queued before the replay's (its catch-up and its programs put back), from
    /// the first one's begin to the replay's begin (cycles).
    pub lead: Option<u64>,
    /// The ops this replay attributed, and their spans summed (cycles).
    pub ops: u32,
    pub spans: u64,
    /// Bytes of programs put back before it: by the B movers from GDDR, and by the host.
    pub restore_mover_bytes: u64,
    pub restore_host_bytes: u64,
    /// The host's time from queueing the replay to its stamps attributed, and the part of it
    /// spent reading the stamps back (microseconds).
    pub host_us: f64,
    pub drain_us: f64,
}

/// A unit's replay list in its drained stamps: `(lead, wall)` in cycles, the replay's list (the
/// last outermost one) from its begin to its end, and the lists before it from the first's begin
/// to the replay's. `None` without a whole replay list.
pub fn replay_wall(events: &[TraceEvent]) -> Option<(u64, u64)> {
    let stamps: Vec<(Event, u64)> = events.iter().filter_map(mover).collect();
    let begin = stamps
        .iter()
        .rposition(|(e, _)| matches!(e, Event::List { begin: true, .. }))?;
    let end = stamps[begin..]
        .iter()
        .rposition(|(e, _)| matches!(e, Event::List { begin: false, .. }))?;
    let first = stamps
        .iter()
        .find(|(e, _)| matches!(e, Event::List { begin: true, .. }))?
        .1;
    let (t0, t1) = (stamps[begin].1, stamps[begin + end].1);
    Some((t0.saturating_sub(first), t1.saturating_sub(t0)))
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
    /// The units whose stamps since the last attribution outran their buffer.
    overflow: Vec<u16>,
    traces: BTreeMap<u64, TraceSamples>,
    pending: Option<Pending>,
    /// The trace whose replay was last attributed, for the host's side of its record.
    last_replay: Option<u64>,
    counter_at_start: u64,
    host_at_start: std::time::Instant,
}

impl Profiler {
    pub(crate) fn new(options: ProfileOptions, counter_at_start: u64) -> Self {
        Self {
            options,
            eager: Vec::new(),
            eager_overflowed: BTreeMap::new(),
            overflow: Vec::new(),
            traces: BTreeMap::new(),
            pending: None,
            last_replay: None,
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
                reads: vec![Vec::new(); ops.len()],
                pipes: vec![Vec::new(); ops.len()],
                pipe_batches: vec![Vec::new(); ops.len()],
                pipe_dropped_ops: vec![0; ops.len()],
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
    pub(crate) fn overflowed(&mut self, unit: usize) {
        self.overflow.push(unit as u16);
    }

    /// The host's side of the replay just attributed ([`ReplayRecord`]): its time from queueing to
    /// attribution, the stamps' read-back within it, and the programs put back before it.
    pub(crate) fn note_replay_host(
        &mut self,
        host: std::time::Duration,
        drain: std::time::Duration,
        restore_mover_bytes: u64,
        restore_host_bytes: u64,
    ) {
        let Some(trace) = self.last_replay.take() else {
            return;
        };
        if let Some(r) = self
            .traces
            .get_mut(&trace)
            .and_then(|t| t.replay_records.last_mut())
        {
            r.host_us = host.as_secs_f64() * 1e6;
            r.drain_us = drain.as_secs_f64() * 1e6;
            r.restore_mover_bytes = restore_mover_bytes;
            r.restore_host_bytes = restore_host_bytes;
        }
    }

    pub(crate) fn expecting_replay(&self) -> bool {
        matches!(self.pending, Some(Pending::Replay { .. }))
    }

    /// The stamps the units recorded since the last drain (one stream a unit, unit 0 first),
    /// attributed to what was pending.
    pub(crate) fn drained(&mut self, units: &[Vec<TraceEvent>]) {
        let detail = self.options.detail;
        let mut overflow = std::mem::take(&mut self.overflow);
        // Every replay is recorded whole, overflowed or not: an overflowed unit's stream is empty,
        // so the others still give their walls.
        if let Some(Pending::Replay { trace, lo, hi, .. }) = self.pending {
            if let Some(t) = self.traces.get_mut(&trace) {
                overflow.sort_unstable();
                overflow.dedup();
                let estimated = match (lo, hi) {
                    (0, 0) => t.stamps.iter().sum(),
                    _ => t.stamps[lo as usize..(hi as usize).min(t.stamps.len())]
                        .iter()
                        .sum(),
                };
                let walls: Vec<(u16, (u64, u64))> = units
                    .iter()
                    .enumerate()
                    .filter_map(|(u, e)| Some((u as u16, replay_wall(e)?)))
                    .collect();
                let unit0 = walls.iter().find(|(u, _)| *u == 0).map(|&(_, w)| w);
                t.replay_records.push(ReplayRecord {
                    lo,
                    hi,
                    estimated,
                    overflowed: overflow.clone(),
                    wall: unit0.map(|(_, w)| w),
                    wall_max: walls
                        .iter()
                        .map(|&(u, (_, w))| (u, w))
                        .max_by_key(|&(_, w)| w),
                    lead: unit0.map(|(l, _)| l),
                    ..Default::default()
                });
                self.last_replay = Some(trace);
            }
        }
        if !overflow.is_empty() {
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
                    let (mut ops, mut spans) = (0u32, 0u64);
                    for (k, time) in attribute_replay(unit0, lo, detail).into_iter().enumerate() {
                        if let Some(s) = t.samples.get_mut(first + k) {
                            ops += 1;
                            spans += time.span;
                            s.push(time);
                        }
                    }
                    if let Some(r) = t.replay_records.last_mut() {
                        r.ops = ops;
                        r.spans = spans;
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
                    let marks: Vec<Option<UnitReads>> = units
                        .iter()
                        .enumerate()
                        .map(|(u, events)| {
                            let seq = t.unit_barriers.get(u).filter(|s| !s.is_empty())?;
                            unit_reads(events, seq, lo, hi)
                        })
                        .collect();
                    if let Some(seq0) = t.unit_barriers.first() {
                        for (k, bins) in card_reads(&marks, seq0, first, lo) {
                            if let Some(s) = t.reads.get_mut(k) {
                                s.push(bins);
                            }
                        }
                    }
                    if detail == Detail::Pipeline {
                        t.pipelines(units, lo, hi);
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
                    if let Some(r) = t.replay_records.last_mut() {
                        r.ops = 1;
                        r.spans = time.span;
                    }
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

/// The `q` quantile of `v` (the sample nearest `q` of the way from the least to the greatest, as
/// the benches' `Stats` take it), 0 for none.
fn quantile(mut v: Vec<u64>, q: f64) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[((v.len() - 1) as f64 * q).round() as usize]
}

impl TraceSamples {
    /// [`Detail::Pipeline`]: one replay's streamed batches on every unit, by op ([`unit_pipes`]).
    fn pipelines(&mut self, units: &[Vec<TraceEvent>], lo: u32, hi: u32) {
        let mut by_op: BTreeMap<usize, (PipeSample, Vec<UnitBatches>)> = BTreeMap::new();
        for (u, events) in units.iter().enumerate() {
            let Some(seq) = self.unit_barriers.get(u).filter(|s| !s.is_empty()) else {
                continue;
            };
            // A unit whose barriers do not line up is counted by `unit_times`.
            let Some(found) = unit_pipes(events, seq, lo, hi) else {
                continue;
            };
            self.pipe_told += u64::from(found.told);
            self.pipe_repeats += u64::from(found.repeats);
            for (d, n) in self.pipe_dropped.iter_mut().zip(found.dropped) {
                *d += n;
            }
            for (k, n) in found.dropped_ops {
                if let Some(d) = self.pipe_dropped_ops.get_mut(k) {
                    *d += n;
                }
            }
            for (k, since, arrive, batches) in found.ops {
                let stats = PipeStats::of(since, arrive, &batches);
                let (sample, kept) = by_op.entry(k).or_default();
                sample.sum.add(&stats);
                sample.units += 1;
                if stats.work >= sample.slowest.work {
                    sample.slowest = stats;
                }
                kept.push((u as u16, since, arrive, batches));
            }
        }
        for (k, (sample, kept)) in by_op {
            if let Some(s) = self.pipes.get_mut(k) {
                s.push(sample);
            }
            if let Some(b) = self.pipe_batches.get_mut(k).filter(|b| b.is_empty()) {
                *b = kept;
            }
        }
    }

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
                        let work: Vec<u64> = ts.iter().map(|t| t.work).collect();
                        UnitMedian {
                            unit,
                            work: m(|t| t.work),
                            work_p10: quantile(work.clone(), 0.1),
                            work_p90: quantile(work, 0.9),
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

    /// Per op, the card's reads in its span ([`TraceSamples::reads`]) from the replay with the
    /// median total, or `None` for an op none measured.
    pub fn read_medians(&self) -> Vec<Option<[u64; READ_BINS]>> {
        self.read_bin_medians()
            .into_iter()
            .map(|r| r.map(|r| r.all))
            .collect()
    }

    /// [`TraceSamples::read_medians`] with who issued the reads and the op's units at work.
    pub fn read_bin_medians(&self) -> Vec<Option<ReadBins>> {
        self.reads
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s.sort_by_key(|b| b.all.iter().sum::<u64>());
                s.get(s.len() / 2).copied()
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
        // Where the products' reads land: by label, their in-span reads by channel and NoC #0
        // endpoint, each against an even share (1.00: every channel, or each of the 16 endpoints
        // NoC #0 owns, read alike).
        let mut by_label: BTreeMap<&str, (u64, OpBytes)> = BTreeMap::new();
        for (o, m) in covered.iter().filter(|(o, _)| o.bytes.b > 0) {
            let e = by_label.entry(o.label).or_default();
            e.0 += m.span;
            e.1.add(&o.bytes);
        }
        if !by_label.is_empty() {
            let _ = writeln!(
                out,
                "\nWhere the products read in their spans (by channel and NoC #0 endpoint;                  busiest against an even share, 1.00 even):\n\n                 | label | us | span MB | busiest channel | its share | busiest endpoint | its share |\n\
                 |---|--:|--:|--:|--:|--:|--:|"
            );
            for (label, (span, b)) in &by_label {
                let e = b.span_endpoint();
                let total: u64 = e.iter().flatten().sum();
                if total == 0 {
                    continue;
                }
                let (ch, ch_bytes) = e
                    .iter()
                    .enumerate()
                    .map(|(c, p)| (c, p.iter().sum::<u64>()))
                    .max_by_key(|&(c, b)| (b, std::cmp::Reverse(c)))
                    .unwrap_or_default();
                let ((ec, ep), ep_bytes) = e
                    .iter()
                    .enumerate()
                    .flat_map(|(c, p)| p.iter().enumerate().map(move |(q, &b)| ((c, q), b)))
                    .max_by_key(|&(at, b)| (b, std::cmp::Reverse(at)))
                    .unwrap_or_default();
                let endpoints = (CHANNELS * (PORTS - 1)) as f64;
                let _ = writeln!(
                    out,
                    "| {label} | {:.1} | {:.1} | {ch} | {:.2} | {ec}.{ep} | {:.2} |",
                    self.us(*span),
                    mb(total),
                    ch_bytes as f64 * CHANNELS as f64 / total as f64,
                    ep_bytes as f64 * endpoints / total as f64,
                );
            }
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

    /// A trace's GDDR reads card-wide over its replays ([`TraceSamples::reads`]): in all, how long
    /// the card read at what share of [`CARD_READ_GBPS`] (each op's span in [`READ_BINS`] parts),
    /// and by label, against what the label's ops read in their own spans.
    fn reads_table(&self, out: &mut String, t: &TraceSamples, medians: &[Option<OpTime>]) {
        let reads = t.read_medians();
        let measured: Vec<(&OpInfo, &OpTime, &[u64; READ_BINS])> = t
            .ops
            .iter()
            .zip(medians)
            .zip(&reads)
            .filter_map(|((o, m), r)| Some((o, m.as_ref()?, r.as_ref()?)))
            .collect();
        if measured.is_empty() {
            return;
        }
        // Time at each share of the card's rate: under 25%, 25-50%, 50-75%, 75-90%, 90% and over.
        const SHARES: [f64; 4] = [0.25, 0.5, 0.75, 0.9];
        let mut at = [0f64; 5];
        let (mut bytes, mut span) = (0u64, 0u64);
        #[derive(Default)]
        struct Label {
            ops: u64,
            span: u64,
            issued: u64,
            own: u64,
            bins: [u64; READ_BINS],
        }
        let mut labels: BTreeMap<&str, Label> = BTreeMap::new();
        for &(o, m, r) in &measured {
            let part = self.us(m.span) / READ_BINS as f64;
            for &b in r {
                let share = gbps(b, part) / CARD_READ_GBPS;
                at[SHARES.iter().take_while(|&&s| share >= s).count()] += part;
            }
            let issued: u64 = r.iter().sum();
            bytes += issued;
            span += m.span;
            let l = labels.entry(o.label).or_default();
            l.ops += 1;
            l.span += m.span;
            l.issued += issued;
            l.own += o.bytes.read - o.bytes.ahead;
            for (s, b) in l.bins.iter_mut().zip(r) {
                *s += b;
            }
        }
        let _ = writeln!(
            out,
            "\nThe card's GDDR reads (every unit's B, as it issued them, by its `READ_BYTES` marks, \
             on unit 0's clock; {} of the trace's ops measured): {:.1} MB in {:.1} us, {:.0} GB/s. \
             Time at under 25% / 25-50% / 50-75% / 75-90% / 90% and over of {CARD_READ_GBPS} GB/s \
             (by eighths of each op's span): {:.1} / {:.1} / {:.1} / {:.1} / {:.1} us.\n\n\
             | label | ops | us | MB read in the span | GB/s | its own MB in the span | \
             GB/s by eighth of the span |\n|---|--:|--:|--:|--:|--:|---|",
            measured.len(),
            bytes as f64 / 1e6,
            self.us(span),
            gbps(bytes, self.us(span)),
            at[0],
            at[1],
            at[2],
            at[3],
            at[4],
        );
        let mut labels: Vec<_> = labels.into_iter().collect();
        labels.sort_by_key(|(_, l)| std::cmp::Reverse(l.span));
        for (label, l) in labels {
            let part = self.us(l.span) / READ_BINS as f64;
            let bins: Vec<String> = l
                .bins
                .iter()
                .map(|&b| format!("{:.0}", gbps(b, part)))
                .collect();
            let _ = writeln!(
                out,
                "| {} | {} | {:.1} | {:.2} | {:.0} | {:.2} | {} |",
                if label.is_empty() {
                    "(unlabelled)"
                } else {
                    label
                },
                l.ops,
                self.us(l.span),
                l.issued as f64 / 1e6,
                gbps(l.issued, self.us(l.span)),
                l.own as f64 / 1e6,
                bins.join(" ")
            );
        }
    }

    /// Each measured op's median span, its read bins and its own units, with the bins' width in
    /// microseconds: what [`ProfileReport::tails_table`] and [`ProfileReport::tails_tsv`] read.
    fn read_bins<'a>(
        &self,
        t: &'a TraceSamples,
        medians: &[Option<OpTime>],
    ) -> Vec<(usize, &'a OpInfo, f64, ReadBins)> {
        t.ops
            .iter()
            .zip(medians)
            .zip(t.read_bin_medians())
            .enumerate()
            .filter_map(|(k, ((o, m), r))| {
                Some((k, o, self.us(m.as_ref()?.span) / READ_BINS as f64, r?))
            })
            .collect()
    }

    /// The card's read capacity left unused, in microseconds of [`CARD_READ_GBPS`], in a part of an
    /// op's span where some of its own units have already arrived at its barrier (its tail): the
    /// most a unit freed there could buy by reading later ops' bytes ahead, if nothing else bound
    /// them. Zero outside the tail.
    fn tail_slack(r: &ReadBins, i: usize, part: f64) -> f64 {
        if r.working[i] >= r.working[0] {
            return 0.0;
        }
        part * (1.0 - gbps(r.all[i], part) / CARD_READ_GBPS).max(0.0)
    }

    /// A trace's product tails ([`ReadBins`]): by label, the GDDR rate in eighths of the span split
    /// into the op's own units and units reading ahead, its own units still at work, and the
    /// capacity unused while some of them wait at its barrier ([`ProfileReport::tail_slack`]).
    /// Their sum is an upper bound on what starting freed units' reads early could save (H4).
    fn tails_table(&self, out: &mut String, t: &TraceSamples, medians: &[Option<OpTime>]) {
        let measured = self.read_bins(t, medians);
        if measured.is_empty() {
            return;
        }
        #[derive(Default)]
        struct Label {
            ops: u64,
            part: f64,
            all: [u64; READ_BINS],
            own: [u64; READ_BINS],
            working: [u64; READ_BINS],
            slack: f64,
        }
        let mut labels: BTreeMap<&str, Label> = BTreeMap::new();
        let mut slack = 0.0;
        for (_, o, part, r) in &measured {
            let l = labels.entry(o.label).or_default();
            l.ops += 1;
            l.part += part;
            for i in 0..READ_BINS {
                l.all[i] += r.all[i];
                l.own[i] += r.own[i];
                l.working[i] += u64::from(r.working[i]);
                let s = Self::tail_slack(r, i, *part);
                l.slack += s;
                slack += s;
            }
        }
        let _ = writeln!(
            out,
            "\nTails (each op's eighths: GB/s of its own units, the units that ran a share of it, and \
             of units reading ahead for later ops; its own units still at work, a mean an op; and the \
             card's capacity at {CARD_READ_GBPS} GB/s left unused in the eighths after its first unit \
             arrived): **{:.1} us unused in tails a replay**, an upper bound on what reading ahead \
             from freed units could save.\n\n\
             | label | ops | tail slack us | own GB/s by eighth | ahead GB/s by eighth | \
             own units at work by eighth |\n|---|--:|--:|---|---|---|",
            slack
        );
        let mut labels: Vec<_> = labels.into_iter().collect();
        labels.sort_by(|a, b| b.1.slack.total_cmp(&a.1.slack));
        for (label, l) in labels {
            let row = |v: &[u64; READ_BINS], f: &dyn Fn(u64) -> String| {
                v.iter().map(|&x| f(x)).collect::<Vec<_>>().join(" ")
            };
            let _ = writeln!(
                out,
                "| {} | {} | {:.1} | {} | {} | {} |",
                if label.is_empty() {
                    "(unlabelled)"
                } else {
                    label
                },
                l.ops,
                l.slack,
                row(&l.own, &|b| format!("{:.0}", gbps(b, l.part))),
                row(&std::array::from_fn(|i| l.all[i] - l.own[i]), &|b| format!(
                    "{:.0}",
                    gbps(b, l.part)
                )),
                row(&l.working, &|n| format!("{:.1}", n as f64 / l.ops as f64)),
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

    /// [`Detail::Pipeline`]: a trace's streamed batches by label. Per op, its slowest unit (most work)
    /// in each replay, its time split by the pipeline's steps as shares of its work; and its units'
    /// mean. B's columns (ring, issue, land) and the roles' (starve, kernel, out wait) run at once
    /// on different cores, so a row does not sum to 100%. *What-if saves* is the slowest unit's
    /// pipe less its batches perfectly overlapped ([`PipeStats::ideal`]), a share of its work.
    fn pipeline_table(&self, out: &mut String, t: &TraceSamples) {
        if t.pipes.iter().all(Vec::is_empty) {
            return;
        }
        #[derive(Default)]
        struct Row {
            ops: u64,
            samples: u64,
            units: u64,
            slowest: PipeStats,
            sum: PipeStats,
        }
        let mut rows: Vec<(&str, Row)> = Vec::new();
        for (o, samples) in t.ops.iter().zip(&t.pipes) {
            if samples.is_empty() {
                continue;
            }
            let at = match rows.iter().position(|(l, _)| *l == o.label) {
                Some(i) => i,
                None => {
                    rows.push((o.label, Row::default()));
                    rows.len() - 1
                }
            };
            let r = &mut rows[at].1;
            r.ops += 1;
            for s in samples {
                r.samples += 1;
                r.units += s.units;
                r.slowest.add(&s.slowest);
                r.sum.add(&s.sum);
            }
        }
        rows.sort_by_key(|(_, r)| std::cmp::Reverse(r.slowest.work / r.samples.max(1) * r.ops));
        let mut dropped: Vec<String> = DROP_REASONS
            .iter()
            .zip(t.pipe_dropped)
            .filter(|(_, n)| *n > 0)
            .map(|(why, n)| format!("{n} {why}"))
            .collect();
        let mut by_label: Vec<(&str, u32)> = Vec::new();
        for (o, &n) in t.ops.iter().zip(&t.pipe_dropped_ops) {
            if n == 0 {
                continue;
            }
            match by_label.iter_mut().find(|(l, _)| *l == o.label) {
                Some((_, m)) => *m += n,
                None => by_label.push((o.label, n)),
            }
        }
        by_label.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        if !by_label.is_empty() {
            dropped.push(format!(
                "by label {}",
                by_label
                    .iter()
                    .take(8)
                    .map(|(l, n)| format!("`{l}` {n}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let dropped = if dropped.is_empty() {
            "none".to_string()
        } else {
            dropped.join("; ")
        };
        let _ = writeln!(
            out,
            "\nStreamed batches by label (`Detail::Pipeline`): per op, its slowest unit's work and its \
             shares of it; B's columns and the roles' overlap in time. Pairs told: {}; not told: {}; \
             bases the timestamper repeated: {}.\n\n\
             | label | ops | units | batches a unit | slowest unit us | head | ring | issue | land | \
             starve | kernel | out wait | tail | overlap | what-if saves | mean unit us | its ring | \
             its starve |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
            t.pipe_told,
            dropped,
            t.pipe_repeats
        );
        let share = |part: u64, whole: u64| 100.0 * part as f64 / whole.max(1) as f64;
        for (label, r) in rows {
            let (w, m) = (&r.slowest, &r.sum);
            let _ = writeln!(
                out,
                "| {label} | {} | {:.0} | {:.1} | {:.2} | {:.0}% | {:.0}% | {:.0}% | {:.0}% | {:.0}% | \
                 {:.0}% | {:.0}% | {:.0}% | {:.0}% | {:.0}% | {:.2} | {:.0}% | {:.0}% |",
                r.ops,
                r.units as f64 / r.samples.max(1) as f64,
                m.batches as f64 / r.units.max(1) as f64,
                self.us(w.work) / r.samples.max(1) as f64,
                share(w.head, w.work),
                share(w.ring, w.work),
                share(w.issue, w.work),
                share(w.land, w.work),
                share(w.starve, w.work),
                share(w.kernel, w.work),
                share(w.out_wait, w.work),
                share(w.tail, w.work),
                share(w.overlap, w.work),
                share(w.pipe.saturating_sub(w.ideal), w.work),
                self.us(m.work) / r.units.max(1) as f64,
                share(m.ring, m.work),
                share(m.starve, m.work),
            );
        }
    }

    /// [`Detail::Pipeline`]: every trace op's streamed batches from the first replay that stamped
    /// it, one line a batch on each unit, tab-separated: each step's time in microseconds from the
    /// unit's previous barrier's release ([`Batch`]), and the unit's work on the op.
    pub fn pipeline_tsv(&self, name: &dyn Fn(u64) -> String) -> String {
        let mut out = String::from(
            "source\tindex\tlabel\twhat\tunit\tbatch\twork_us\treserve_us\tgranted_us\tissued_us\t\
             landed_us\tt0_wait_us\tt0_ready_us\tt0_pop_us\tt2_wait_us\tt2_reserved_us\tt2_push_us\n",
        );
        for (&id, t) in &self.traces {
            let source = name(id);
            for (i, (o, units)) in t.ops.iter().zip(&t.pipe_batches).enumerate() {
                for (u, since, arrive, batches) in units {
                    let at = |c: u64| self.us(c) - self.us(*since);
                    for (b, x) in batches.iter().enumerate() {
                        let _ = write!(
                            out,
                            "{source}\t{i}\t{}\t{}\t{u}\t{b}\t{:.3}",
                            o.label,
                            o.what,
                            self.us(arrive.saturating_sub(*since)),
                        );
                        for c in [x.reserve, x.granted, x.issued, x.landed]
                            .into_iter()
                            .chain(x.t0)
                            .chain(x.t2)
                        {
                            let _ = write!(out, "\t{:.3}", at(c));
                        }
                        out.push('\n');
                    }
                }
            }
        }
        out
    }

    /// Every trace op's units, one line each, tab-separated: each unit's median work (from its last
    /// barrier's release to its arrival at the op's) and wait there, in microseconds, and its work's
    /// 10th and 90th percentiles over the replays (last, so readers by position are unchanged).
    pub fn skew_tsv(&self, name: &dyn Fn(u64) -> String) -> String {
        let mut out = String::from(
            "source\tindex\tlabel\twhat\tunit\tsamples\twork_us\twait_us\tspan_mb\tahead_mb\tspan_gbps\tpush_us\tdrain_us\tpush_words\twork_p10_us\twork_p90_us\n",
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
                        "{source}\t{i}\t{}\t{}\t{u}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.1}\t{:.3}\t{:.3}\t{}\t{:.3}\t{:.3}",
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
                        self.us(m.work_p10),
                        self.us(m.work_p90),
                    );
                }
            }
        }
        out
    }

    /// Every measured trace op's reads in each eighth of its span, one line an eighth, tab-separated
    /// ([`ReadBins`]): the rate of its own units and of units reading ahead, its own units still at
    /// work, and the capacity left unused in its tail ([`ProfileReport::tail_slack`]).
    pub fn tails_tsv(&self, name: &dyn Fn(u64) -> String) -> String {
        let mut out = String::from(
            "source\tindex\tlabel\twhat\tunits\teighth\tpart_us\tall_gbps\town_gbps\tahead_gbps\tworking\tslack_us\n",
        );
        for (&id, t) in &self.traces {
            let source = name(id);
            for (k, o, part, r) in self.read_bins(t, &t.medians()) {
                for i in 0..READ_BINS {
                    let _ = writeln!(
                        out,
                        "{source}\t{k}\t{}\t{}\t{}\t{i}\t{part:.3}\t{:.1}\t{:.1}\t{:.1}\t{}\t{:.3}",
                        o.label,
                        o.what,
                        o.units,
                        gbps(r.all[i], part),
                        gbps(r.own[i], part),
                        gbps(r.all[i] - r.own[i], part),
                        r.working[i],
                        Self::tail_slack(&r, i, part),
                    );
                }
            }
        }
        out
    }

    /// Every trace op's GDDR reads by channel and NoC #0 endpoint, one line a (channel, endpoint)
    /// it reads: in its own span and ahead of it, megabytes, beside the op's median span.
    pub fn channels_tsv(&self, name: &dyn Fn(u64) -> String) -> String {
        let mut out =
            String::from("source\tindex\tlabel\twhat\tspan_us\tchannel\tport\tspan_mb\tahead_mb\n");
        for (&id, t) in &self.traces {
            let source = name(id);
            for (i, (o, m)) in t.ops.iter().zip(t.medians()).enumerate() {
                let span = m.map_or(0.0, |m| self.us(m.span));
                let (all, ahead) = (&o.bytes.endpoint, &o.bytes.ahead_endpoint);
                for c in 0..CHANNELS {
                    for p in 0..PORTS {
                        if all[c][p] == 0 {
                            continue;
                        }
                        let _ = writeln!(
                            out,
                            "{source}\t{i}\t{}\t{}\t{span:.3}\t{c}\t{p}\t{:.4}\t{:.4}",
                            o.label,
                            o.what,
                            (all[c][p] - ahead[c][p]) as f64 / 1e6,
                            ahead[c][p] as f64 / 1e6,
                        );
                    }
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

    /// A trace's wall time on the card against its covered spans (`covered`, cycles), and the ops
    /// no replay stamped: what tells the card's time from the host's ([`ReplayRecord`]).
    fn wall_lines(&self, out: &mut String, t: &TraceSamples, covered: u64) {
        let r = &t.replay_records;
        let med = |f: &dyn Fn(&ReplayRecord) -> Option<u64>| {
            let v: Vec<u64> = r.iter().filter_map(f).collect();
            (!v.is_empty()).then(|| median(v))
        };
        if let Some(wall) = med(&|r| r.wall) {
            let max = med(&|r| r.wall_max.map(|(_, w)| w)).unwrap_or(0);
            let lead = med(&|r| r.lead).unwrap_or(0);
            let host: Vec<u64> = r.iter().map(|r| (r.host_us * 1e3) as u64).collect();
            let drain: Vec<u64> = r.iter().map(|r| (r.drain_us * 1e3) as u64).collect();
            let _ = writeln!(
                out,
                "On the card, a replay (median of {} with unit 0's stamps): **{:.1} us** on unit 0, \
                 {:.1} us on its longest unit, after {:.1} us of lists queued before it (catch-up, \
                 programs put back); the covered ops' spans {:.1} us, so **{:.1} us** is in no \
                 covered span (ops never stamped, and between spans). Host, profiled: {:.1} us from \
                 queueing to attribution, {:.1} us of it reading the stamps back. Programs put back \
                 a replay (mean): {:.0} bytes by the B movers, {:.0} by the host.\n",
                r.iter().filter(|r| r.wall.is_some()).count(),
                self.us(wall),
                self.us(max),
                self.us(lead),
                self.us(covered),
                self.us(wall.saturating_sub(covered)),
                median(host) as f64 / 1e3,
                median(drain) as f64 / 1e3,
                r.iter().map(|r| r.restore_mover_bytes).sum::<u64>() as f64
                    / r.len().max(1) as f64,
                r.iter().map(|r| r.restore_host_bytes).sum::<u64>() as f64 / r.len().max(1) as f64,
            );
        }
        let never: Vec<usize> = (0..t.ops.len())
            .filter(|&i| t.samples[i].is_empty())
            .collect();
        if !never.is_empty() && t.segmented {
            let _ = writeln!(
                out,
                "Ops no replay stamped ({}): {}.\n",
                never.len(),
                never
                    .iter()
                    .take(40)
                    .map(|&i| {
                        let o = &t.ops[i];
                        let units: Vec<String> = r
                            .iter()
                            .filter(|r| {
                                !r.overflowed.is_empty()
                                    && (r.lo as usize..r.hi as usize).contains(&i)
                            })
                            .flat_map(|r| r.overflowed.iter().map(|u| u.to_string()))
                            .collect();
                        format!(
                            "{i} `{}` ({}, {} stamps est.{})",
                            o.label,
                            o.what,
                            t.stamps.get(i).copied().unwrap_or(0),
                            if units.is_empty() {
                                String::new()
                            } else {
                                format!(", overflowed on unit {}", units.join("/"))
                            }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
    }

    /// Every replay of every trace, one line each, tab-separated ([`ReplayRecord`]): its window,
    /// the units that overflowed, its wall time on the card and the spans it attributed, the
    /// programs put back before it, and the host's time around it (microseconds).
    pub fn replays_tsv(&self, name: &dyn Fn(u64) -> String) -> String {
        let mut out = String::from(
            "source\treplay\tlo\thi\testimated\toverflowed\twall_us\twall_max_us\twall_max_unit\tlead_us\tops\tspans_us\trestore_mover_bytes\trestore_host_bytes\thost_us\tdrain_us\n",
        );
        let opt = |c: Option<u64>| c.map_or(String::new(), |c| format!("{:.3}", self.us(c)));
        for (&id, t) in &self.traces {
            let source = name(id);
            for (n, r) in t.replay_records.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "{source}\t{n}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{:.1}\t{:.1}",
                    r.lo,
                    r.hi,
                    r.estimated,
                    r.overflowed
                        .iter()
                        .map(u16::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                    opt(r.wall),
                    opt(r.wall_max.map(|(_, w)| w)),
                    r.wall_max.map_or(String::new(), |(u, _)| u.to_string()),
                    opt(r.lead),
                    r.ops,
                    self.us(r.spans),
                    r.restore_mover_bytes,
                    r.restore_host_bytes,
                    r.host_us,
                    r.drain_us,
                );
            }
        }
        out
    }

    /// A trace's median wall time on the card a replay, unit 0's (microseconds), if any replay kept
    /// unit 0's stamps.
    pub fn replay_wall_us(&self, trace: u64) -> Option<f64> {
        let v: Vec<u64> = self
            .traces
            .get(&trace)?
            .replay_records
            .iter()
            .filter_map(|r| r.wall)
            .collect();
        (!v.is_empty()).then(|| self.us(median(v)))
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
                 (largest op {}), {WINDOW_EVENTS} a window. Units' replays whose barriers did not \
                 line up: {}.\n",
                t.stamps.iter().sum::<u32>(),
                t.stamps.iter().max().copied().unwrap_or(0),
                t.unit_mismatches,
            );
            self.wall_lines(&mut out, t, sum);
            let rows = by_label(
                t.ops
                    .iter()
                    .zip(&medians)
                    .filter_map(|(o, m)| m.as_ref().map(|m| (o.label, m))),
            );
            self.label_table(&mut out, &rows, "ops");
            self.bytes_tables(&mut out, t, &medians);
            self.reads_table(&mut out, t, &medians);
            self.tails_table(&mut out, t, &medians);
            self.skew_table(&mut out, t);
            self.push_table(&mut out, t);
            self.pipeline_table(&mut out, t);
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
            "source\tindex\tlabel\twhat\tunits\tsamples\tspan_us\tbarrier_us\tkernels_us\tcontrol_us\tdata_us\tother_us\tread_bytes\twrite_bytes\treads\twrites\ta_bytes\tb_bytes\tgbps\tfetch_us\twriter_us\tahead_bytes\tprefetch_us\tpush_us\tdrain_us\tpush_words\tcard_read_bytes\tcard_read_gbps\n",
        );
        let none = OpBytes::default();
        // The math role's totals: an eager op's on its longest unit, a trace op's on its slowest.
        let mut row = |source: &str,
                       i: usize,
                       o: (&str, &str, usize, &OpBytes),
                       n: usize,
                       t: &OpTime,
                       (push, drain, words): (u64, u64, u64),
                       issued: Option<u64>| {
            let b = o.3;
            let (issued, issued_gbps) = match issued {
                Some(i) => (i.to_string(), format!("{:.1}", gbps(i, self.us(t.span)))),
                None => (String::new(), String::new()),
            };
            let _ = writeln!(
                out,
                "{source}\t{i}\t{}\t{}\t{}\t{n}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.3}\t{:.3}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{issued}\t{issued_gbps}",
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
            row(
                "eager",
                i,
                (s.label, s.what, 0, &none),
                1,
                &s.time,
                totals,
                None,
            );
        }
        for (&id, t) in &self.traces {
            let source = name(id);
            let units = t.unit_medians();
            let reads = t.read_medians();
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
                        reads[i].map(|r| r.iter().sum()),
                    );
                }
            }
        }
        out
    }

    /// A Perfetto/Chrome trace: eager ops one after another on one track, and each trace's ops one
    /// after another at their median spans on a track of its own, each span named by its label,
    /// with the card's GDDR read rate as a counter beside them ([`TraceSamples::reads`]).
    pub fn chrome_trace(&self, name: &dyn Fn(u64) -> String) -> String {
        type Op<'a> = (&'a str, &'a str, &'a OpTime, Option<[u64; READ_BINS]>);
        let mut events = Vec::new();
        let mut track = |pid: usize, title: &str, ops: Vec<Op>| {
            events.push(format!(
                "{{\"ph\":\"M\",\"pid\":{pid},\"name\":\"process_name\",\"args\":{{\"name\":{title:?}}}}}"
            ));
            let mut at = 0.0;
            for (label, what, t, reads) in ops {
                let dur = self.us(t.span);
                for (i, &b) in reads.iter().flatten().enumerate() {
                    let part = dur / READ_BINS as f64;
                    events.push(format!(
                        "{{\"ph\":\"C\",\"pid\":{pid},\"name\":\"GDDR reads\",\"ts\":{:.3},\"args\":{{\"GB/s\":{:.1}}}}}",
                        at + i as f64 * part,
                        gbps(b, part)
                    ));
                }
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
                .map(|s| (s.label, s.what, &s.time, None))
                .collect(),
        );
        for (k, (&id, t)) in self.traces.iter().enumerate() {
            let medians = t.medians();
            let reads = t.read_medians();
            let ops = t
                .ops
                .iter()
                .zip(&medians)
                .zip(reads)
                .filter_map(|((o, m), r)| m.as_ref().map(|m| (o.label, o.what, m, r)))
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

    /// A product's `B` reads land by channel and NoC #0 endpoint as its tensor and the mover's
    /// port rotation place them: a `[4, 16]`-tile weight over eight channels puts its column `j`
    /// on channel `j % 8`, two columns a channel, each read on the port its rotation names, or
    /// CMFW's where that is NoC #1's (the rule written out here, not `port_for`).
    #[test]
    fn a_gathers_reads_are_counted_by_channel_and_endpoint() {
        let (kt, ct) = (4u32, 16u32);
        let b = record::TensorRef {
            n: 8,
            channels: [0, 1, 2, 3, 4, 5, 6, 7],
            base: [0x10_0000; 8],
            first: 0,
            ct,
            col_tiles: 0,
            slot: 0,
        };
        let [a0, a1] = record::TensorRef::default().encode();
        let [b0, b1] = b.encode();
        let head = [
            record::GATHER,
            record::GATHER_A_RESIDENT | kt << 8,
            0x2_0000,
            0x6_0000,
            0,
            1,
            0,
            ct,
        ];
        let mut got = OpBytes::default();
        got.count(&[head, a0, a1, b0, b1]);
        let tile = tt_isa::dm::TILE_SLOT;
        let mut want = [[0u64; PORTS]; CHANNELS];
        for kk in 0..kt as usize {
            for j in 0..ct as usize {
                let ch = j % 8;
                let (noc1, cmfw) = (
                    if ch < 4 { 1 } else { 0 },
                    if (1..=3).contains(&ch) { 0 } else { 2 },
                );
                let port = match (kk + j) % 3 {
                    p if p == noc1 => cmfw,
                    p => p,
                };
                want[ch][port] += tile;
            }
        }
        assert_eq!(got.endpoint, want);
        for (c, ports) in got.endpoint.iter().enumerate() {
            assert_eq!(
                ports.iter().sum::<u64>(),
                2 * kt as u64 * tile,
                "channel {c}"
            );
            assert_eq!(
                ports[if c < 4 { 1 } else { 0 }],
                0,
                "NoC #1's endpoint, channel {c}"
            );
        }
        assert_eq!(got.read, got.endpoint.iter().flatten().sum::<u64>());
        // Read ahead, the same bytes are the op's own and none in its span.
        let mut op = OpBytes::default();
        op.add(&got);
        op.ahead_endpoint = got.endpoint;
        assert_eq!(op.span_endpoint(), [[0; PORTS]; CHANNELS]);
    }

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

    /// One op on a unit: a pair of two batches through a one-batch ring, its credits stamped as
    /// `CB` entries, and after its kernel the roles' tables as the host splices them from the log
    /// (`BATCH_TABLE` with `header`, a `BATCH_TIME` a clock read); then the unit's arrival at 430.
    fn pair_of_two(header: u32, times: &[u64]) -> Vec<TraceEvent> {
        let mut v = vec![e(ev::LIST_BEGIN, 1, 100)];
        v.extend(entry(op::LAUNCH, 101, 103));
        v.extend(entry(op::CB, 104, 108));
        v.extend(entry(op::CB, 140, 170));
        v.extend(entry(op::CB, 172, 224));
        v.extend(entry(op::CB, 260, 290));
        v.push(e(ev::ENTRY_BEGIN, op::KERNEL_WAIT, 291));
        v.push(e(ev::ROLES_DONE, 7, 410));
        v.push(e(ev::BATCH_TABLE, header, 416));
        v.extend(times.iter().map(|&t| e(ev::BATCH_TIME, 0, t)));
        v.push(e(ev::ENTRY_END, op::KERNEL_WAIT, 420));
        v.extend(entry(op::BARRIER, 430, 440));
        v.push(e(ev::LIST_END, 1, 441));
        v
    }

    /// The two batches' role times: T0's wait, ready and pop, then T2's wait, reserved and push.
    const ROLE_TIMES: [u64; 12] = [128, 192, 224, 128, 160, 256, 256, 320, 352, 256, 288, 384];

    /// A pair's credits (reserve, then push, per batch) join the roles' tables in order, each
    /// table time counted back from its base; the op's stats follow from the batches.
    #[test]
    fn a_pairs_credits_join_the_roles_batch_tables() {
        let seq = [UnitBarrier {
            arrival_of: Some(0),
            counts: 1,
        }];
        let found = unit_pipes(&pair_of_two(2, &ROLE_TIMES), &seq, 0, 0).unwrap();
        assert_eq!(found.dropped, [0; DROP_REASONS.len()]);
        assert_eq!((found.told, found.repeats), (1, 0));
        let batches = vec![
            Batch {
                reserve: 104,
                granted: 108,
                issued: 140,
                landed: 170,
                t0: [128, 192, 224],
                t2: [128, 160, 256],
            },
            Batch {
                reserve: 172,
                granted: 224,
                issued: 260,
                landed: 290,
                t0: [256, 320, 352],
                t2: [256, 288, 384],
            },
        ];
        assert_eq!(found.ops, vec![(0, 100, 430, batches.clone())]);
        // The credit protocol's own order holds in them: a batch lands before T0 sees it, and with
        // a ring of one batch, the next batch's grant comes after T0's pop of this one.
        assert!(batches.iter().all(|b| b.landed <= b.t0[1]));
        assert!(batches[1].granted >= batches[0].t0[2]);
        assert_eq!(
            PipeStats::of(100, 430, &batches),
            PipeStats {
                batches: 2,
                work: 330,
                head: 4,
                ring: 4 + 52,
                issue: 32 + 36,
                land: 30 + 30,
                starve: 64 + 64,
                kernel: 64 + 64,
                out_wait: 32 + 32,
                tail: 430 - 384,
                // Batch 1's reads, 224..290, beside batch 0's kernel, 192..256.
                overlap: 32,
                pipe: 384 - 108,
                // 62 for batch 0's reads, then max(66, 64), then batch 1's kernel.
                ideal: 62 + 66 + 64,
            }
        );
    }

    /// The host puts each logged table in place of its base: the header as a `BATCH_TABLE`, each
    /// clock read on the full counter, counting back from the base's (across a wrap of the low
    /// word too); a base without a record, or with the log full, stays.
    #[test]
    fn a_logged_table_takes_its_bases_place() {
        let base = (7u64 << 32) + 100;
        let log = [
            0xdead,
            1 | ev::BATCH_MISMATCHED,
            50,
            80,
            90,
            u32::MAX - 9, // 110 cycles before the base, across the low word's wrap
            60,
            95,
        ];
        let stream = vec![
            e(ev::KICK, 1, base - 500),
            e(ev::BATCH_BASE, 1, base),
            e(ev::BATCH_BASE, 1, base + 2), // the timestamper's repeat
            e(ev::BATCH_BASE, ev::BATCH_LOG_FULL, base + 10),
            e(ev::BATCH_BASE, 40, base + 20),
        ];
        let out = splice_batch_log(stream.clone(), &log);
        let at = |t: u64| e(ev::BATCH_TIME, 0, t);
        assert_eq!(
            out,
            vec![
                stream[0],
                e(ev::BATCH_TABLE, 1 | ev::BATCH_MISMATCHED, base),
                at((7 << 32) + 50),
                at((7 << 32) + 80),
                at((7 << 32) + 90),
                at(base - 110),
                at((7 << 32) + 60),
                at((7 << 32) + 95),
                e(ev::BATCH_REPEAT, 0, base + 2),
                stream[3],
                stream[4],
            ]
        );
    }

    /// A table that cannot be joined is dropped and counted, not misattributed: its count against
    /// the pair's credits, a time past what a detail can say, the roles' counts differing, or more
    /// batches than the table held.
    #[test]
    fn a_batch_table_that_does_not_join_is_dropped() {
        let seq = [UnitBarrier {
            arrival_of: Some(0),
            counts: 1,
        }];
        let dropped = |v: Vec<TraceEvent>| {
            let found = unit_pipes(&v, &seq, 0, 0).unwrap();
            assert!(found.ops.is_empty());
            assert_eq!(
                found.dropped_ops.iter().map(|d| d.1).sum::<u32>(),
                1,
                "op 0's"
            );
            found
                .dropped
                .iter()
                .position(|&n| n == 1)
                .filter(|_| found.dropped.iter().sum::<u32>() == 1)
                .map(|why| DROP_REASONS[why])
        };
        assert_eq!(
            dropped(pair_of_two(1, &ROLE_TIMES[..6])),
            Some("more credits"),
            "one batch for two"
        );
        // A base the host found no record for stays in the stream.
        let mut unlogged = pair_of_two(2, &ROLE_TIMES);
        let at = unlogged
            .iter()
            .position(|x| ev::split(x.token).1 == ev::BATCH_TABLE)
            .unwrap();
        unlogged.truncate(at);
        unlogged.push(e(ev::BATCH_BASE, ev::BATCH_LOG_FULL, 416));
        unlogged.push(e(ev::ENTRY_END, op::KERNEL_WAIT, 420));
        unlogged.extend(entry(op::BARRIER, 430, 440));
        assert_eq!(dropped(unlogged), Some("not in the log"));
        assert_eq!(
            dropped(pair_of_two(2 | ev::BATCH_MISMATCHED, &ROLE_TIMES)),
            Some("mismatched")
        );
        assert_eq!(
            dropped(pair_of_two(2 | ev::BATCH_TRUNCATED, &ROLE_TIMES)),
            Some("truncated")
        );
        // A join out of the protocol's order: T2 pushing batch 0 (at 96) before T0 saw it (192),
        // further apart than the stamps' resolution.
        let mut early = ROLE_TIMES;
        early[3..6].copy_from_slice(&[64, 96, 96]);
        assert_eq!(dropped(pair_of_two(2, &early)), Some("disordered"));
        // Three batches' tables against the pair's two batches of credits.
        let three: Vec<u64> = ROLE_TIMES.iter().chain(&ROLE_TIMES[..6]).copied().collect();
        assert_eq!(dropped(pair_of_two(3, &three)), Some("fewer credits"));
        // And barriers that do not line up with the stream refuse the replay.
        let two = [seq[0], seq[0]];
        assert_eq!(unit_pipes(&pair_of_two(2, &ROLE_TIMES), &two, 0, 0), None);
    }

    /// The pipeline detail stamps as `Ops` does, and one base more after each kernel's wait.
    #[test]
    fn the_pipeline_detail_adds_a_base_to_each_wait() {
        let (ops, pipeline) = (
            ProfileOptions::default(),
            ProfileOptions {
                detail: Detail::Pipeline,
                ..ProfileOptions::default()
            },
        );
        assert_eq!(pipeline.trace_mode(), tt_isa::dm::trace_mode::OPS);
        for o in [op::KERNEL, op::KERNEL_WAIT] {
            assert_eq!(pipeline.stamps(o), ops.stamps(o) + 1);
        }
        for o in [op::BARRIER, op::CB, op::LAUNCH, op::READ] {
            assert_eq!(pipeline.stamps(o), ops.stamps(o));
        }
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
    fn a_units_work_over_replays_gives_its_median_and_its_10th_and_90th_percentiles() {
        // Eleven replays of one unit, in no order: 10..=110 us of work.
        let replays = [70, 10, 110, 40, 20, 100, 30, 60, 90, 50, 80].map(|work| UnitTime {
            unit: 3,
            work,
            wait: 0,
            push: 0,
            drain: 0,
            words: 0,
        });
        let mut t = TraceSamples {
            units: vec![replays.to_vec()],
            ..TraceSamples::default()
        };
        let m = t.unit_medians()[0][0];
        assert_eq!(
            (m.unit, m.samples, m.work_p10, m.work, m.work_p90),
            (3, 11, 20, 60, 100)
        );
        // One replay: every quantile is that replay.
        t.units[0].truncate(1);
        let m = t.unit_medians()[0][0];
        assert_eq!((m.work_p10, m.work, m.work_p90), (70, 70, 70));
    }

    #[test]
    fn every_units_reads_land_in_the_ops_spans_on_unit_0s_clock() {
        let arrive = |k| UnitBarrier {
            arrival_of: Some(k),
            counts: 1,
        };
        let marks = |t, bytes: u32| e(ev::READ_BYTES, bytes >> ev::READ_UNIT_SHIFT, t);
        // Unit 0 runs ops 0 and 1, spans 100..150 and 150..250, and reads 4,000 B before its
        // first arrival.
        let seq0 = vec![arrive(0), arrive(1)];
        let mut u0 = vec![e(ev::LIST_BEGIN, 1, 100), marks(100, 64 * 3)];
        u0.extend([marks(140, 4000), e(ev::ENTRY_BEGIN, op::BARRIER, 140)]);
        u0.extend([marks(150, 0), e(ev::ENTRY_END, op::BARRIER, 150)]);
        u0.extend([marks(240, 0), e(ev::ENTRY_BEGIN, op::BARRIER, 240)]);
        u0.extend([marks(250, 0), e(ev::ENTRY_END, op::BARRIER, 250)]);
        // Unit 1, its clock 1,000 ahead, waits out both ops and meanwhile issues an 8,000 B
        // prefetch from 150 to 200 on unit 0's clock.
        let seq1 = vec![UnitBarrier {
            arrival_of: None,
            counts: 2,
        }];
        let mut u1 = vec![e(ev::LIST_BEGIN, 1, 1100), marks(1100, 0)];
        u1.extend([marks(1100, 0), e(ev::ENTRY_BEGIN, op::BARRIER, 1100)]);
        u1.extend([marks(1150, 0), marks(1200, 8000)]);
        u1.extend([marks(1250, 0), e(ev::ENTRY_END, op::BARRIER, 1250)]);

        let r0 = unit_reads(&u0, &seq0, 0, 0).unwrap();
        // The first mark counts from before the replay: left out.
        assert_eq!(r0.spans[0], (100, 140, 4000 / 64 * 64));
        assert_eq!(r0.releases, vec![(0, 150), (1, 250)]);
        let r1 = unit_reads(&u1, &seq1, 0, 0).unwrap();
        assert_eq!(
            r1.releases,
            vec![(1, 1250)],
            "a wait releases at its last barrier"
        );

        assert_eq!(r0.arrivals, vec![(0, 140), (1, 240)]);
        assert!(r1.arrivals.is_empty(), "a wait is no arrival");

        let got = card_reads(&[Some(r0), Some(r1)], &seq0, 0, 0);
        let b = 4000 / 64 * 64;
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].0, 0);
        // 6.25 cycles an eighth of op 0: six whole eighths at 100 B a cycle, then 2.5 cycles.
        let part = b as f64 * 6.25 / 40.0;
        assert_eq!(
            got[0].1.own, got[0].1.all,
            "op 0's reads are its own unit's"
        );
        // Unit 0 arrives at 140, in op 0's seventh eighth (137.5..143.75).
        assert_eq!(got[0].1.working, [1, 1, 1, 1, 1, 1, 1, 0]);
        for (i, &v) in got[0].1.all.iter().enumerate() {
            let want = match i {
                0..=5 => part,
                6 => b as f64 * 2.5 / 40.0,
                _ => 0.0,
            };
            assert!(
                (v as f64 - want).abs() <= 1.0,
                "op 0 eighth {i}: {v} vs {want}"
            );
        }
        // Unit 1's prefetch, moved onto unit 0's clock, is the first half of op 1.
        assert_eq!(got[1].0, 1);
        assert_eq!(got[1].1.all, [2000, 2000, 2000, 2000, 0, 0, 0, 0]);
        assert_eq!(got[1].1.own, [0; READ_BINS], "a prefetch is not op 1's own");
        // Unit 0 arrives at 240, in op 1's last eighth (237.5..250).
        assert_eq!(got[1].1.working, [1; READ_BINS]);

        // Stamps that do not line up with the stream are refused.
        assert_eq!(unit_reads(&u0, &seq0[..1], 0, 0), None);
    }

    #[test]
    fn a_tails_slack_is_the_capacity_unused_after_the_first_unit_arrives() {
        // Four units: all at work for four eighths, then two, then one. An eighth of 10 us.
        let mut r = ReadBins {
            working: [4, 4, 4, 4, 2, 2, 1, 1],
            ..Default::default()
        };
        let full = (CARD_READ_GBPS * 10.0 * 1e3) as u64;
        r.all = [full, full, 0, 0, full / 2, full, 0, full * 2];
        let slack: Vec<f64> = (0..READ_BINS)
            .map(|i| ProfileReport::tail_slack(&r, i, 10.0))
            .collect();
        // Nothing before a unit has arrived, however little is read; half an eighth at half the
        // rate; none at or above it.
        assert_eq!(slack, [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 10.0, 0.0]);
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

    /// A profiler that knows trace 7: three ops of nine stamps, ending in barriers on unit 0.
    fn three_ops() -> Profiler {
        let mut p = Profiler::new(ProfileOptions::default(), 0);
        p.know_trace(7, || {
            (
                (0..3)
                    .map(|_| OpInfo {
                        label: "x",
                        what: "y",
                        units: 2,
                        bytes: OpBytes::default(),
                        unit_bytes: Vec::new(),
                    })
                    .collect(),
                vec![9; 3],
                true,
                Vec::new(),
            )
        });
        p
    }

    /// `replay()` on another unit, its replay list `extra` cycles longer.
    fn longer(extra: u64) -> Vec<TraceEvent> {
        let mut v = replay();
        v.last_mut().unwrap().cycles += extra;
        v
    }

    #[test]
    fn a_replays_wall_is_its_list_after_the_lists_before_it() {
        assert_eq!(replay_wall(&replay()), Some((100, 151)));
        assert_eq!(replay_wall(&[]), None);
        // A replay list that never ended (its unit stopped) has no wall.
        let mut open = replay();
        open.pop();
        assert_eq!(replay_wall(&open), None);
    }

    #[test]
    fn every_replay_is_recorded_with_its_wall_beside_its_spans() {
        let mut p = three_ops();
        assert_eq!(p.next_window(7), 3 << 16, "one window holds the trace");
        p.drained(&[replay(), longer(30)]);
        p.note_replay_host(
            std::time::Duration::from_micros(500),
            std::time::Duration::from_micros(200),
            64,
            0,
        );
        let t = &p.traces[&7];
        assert_eq!(t.covered(), 3);
        let r = &t.replay_records[0];
        assert_eq!((r.lo, r.hi, r.estimated), (0, 3, 27));
        assert!(r.overflowed.is_empty());
        assert_eq!(
            (r.wall, r.lead, r.wall_max),
            (Some(151), Some(100), Some((1, 181)))
        );
        assert_eq!((r.ops, r.spans), (3, 150));
        assert!(r.spans <= r.wall.unwrap(), "spans lie within the wall");
        assert_eq!(
            (r.host_us, r.drain_us, r.restore_mover_bytes),
            (500.0, 200.0, 64)
        );
        let report = ProfileReport {
            ticks_per_us: 1.0,
            traces: p.traces.clone(),
            ..Default::default()
        };
        assert_eq!(report.replay_wall_us(7), Some(151.0));
        let s = report.summary(&|_| "T".into());
        assert!(s.contains("**151.0 us** on unit 0"), "{s}");
        assert!(s.contains("**1.0 us** is in no covered span"), "{s}");
        assert!(!s.contains("Ops no replay stamped"), "{s}");
        let tsv = report.replays_tsv(&|_| "T".into());
        assert_eq!(tsv.lines().count(), 2, "{tsv}");
        assert!(
            tsv.lines().nth(1).unwrap().starts_with(
                "T\t0\t0\t3\t27\t\t151.000\t181.000\t1\t100.000\t3\t150.000\t64\t0\t500.0\t200.0"
            ),
            "{tsv}"
        );
    }

    /// The negative control: a replay whose stamps outran unit 1's buffer attributes nothing, but
    /// its wall is kept from unit 0, so its time shows as in no covered span instead of vanishing,
    /// and its ops are named with the unit that overflowed.
    #[test]
    fn an_overflowed_replays_time_shows_as_uncovered_and_its_ops_are_named() {
        let mut p = three_ops();
        p.next_window(7);
        p.overflowed(1);
        p.drained(&[replay(), Vec::new()]);
        let t = &p.traces[&7];
        assert_eq!((t.replays, t.overflowed, t.covered()), (1, 1, 0));
        let r = &t.replay_records[0];
        assert_eq!(r.overflowed, vec![1]);
        assert_eq!((r.wall, r.wall_max, r.ops), (Some(151), Some((0, 151)), 0));
        let report = ProfileReport {
            ticks_per_us: 1.0,
            traces: p.traces.clone(),
            ..Default::default()
        };
        let s = report.summary(&|_| "T".into());
        assert!(s.contains("**151.0 us** is in no covered span"), "{s}");
        assert!(s.contains("Ops no replay stamped (3)"), "{s}");
        assert!(
            s.contains("0 `x` (y, 9 stamps est., overflowed on unit 1)"),
            "{s}"
        );
        // The next replay starts clean: no overflow carried over.
        p.next_window(7);
        p.drained(&[replay(), replay()]);
        let t = &p.traces[&7];
        assert!(t.replay_records[1].overflowed.is_empty());
        assert_eq!(t.covered(), 3);
    }
}
