//! Which of a capture's barriers its data needs (`docs/plans/decode-small-ops.md` S5): a host-only
//! analysis of the captured streams, made as a capture ends.
//!
//! Every captured op ends in a barrier its units arrive at, and every unit that runs the next op
//! waits for it first, so a trace's ops run totally ordered. What actually orders them is data:
//! an op on one unit that reads (or overwrites) what an op on another unit wrote (or still reads).
//! This reads each op's moves off the streams, as the movers expand them, and finds for each
//! (op, unit) the other units' ops it must follow:
//!
//! - **GDDR**, byte ranges a channel: a plain or expanded `READ`, `READ_TRANSPOSED`,
//!   `READ_BROADCAST_COL` or `WRITE`. A record whose rows the card picks from indices in L1
//!   (`SELECT_ROWS`, `SCATTER_ROWS`) touches every row its index could name. A read the host
//!   cannot place (`READ_ENCODED`) reads all of GDDR.
//! - **Another tile's L1**: `WRITE_L1`, `WRITE_L1_MCAST` (each of the session's units in its
//!   rectangle) and `READ_L1`, the planned residency's moves.
//! - **Anything else** that is not one of the mover's tile-local entries (a host transfer, an op
//!   this analysis does not know) is a dependency on every unit, both ways.
//!
//! A unit's own L1 outside those ranges (its slots, rings, descriptors, the program cache) is its
//! own: one unit's entries run in stream order, so it needs no barrier. The lists a product reads
//! ahead (`OpRecord::ahead`) are its op's.
//!
//! From the dependencies, each barrier is classed ([`Class`]): **unneeded** where no unit of the
//! next op depends on another unit's op since the last barrier kept (the barriers are kept
//! greedily, which keeps the fewest: if ops `g..k` need none between them, any run inside does
//! not either); **sparse** where each unit of the next op depends on some, not all, of the units
//! since; **full** otherwise.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;

use tt_isa::dm::{op, record};

use crate::trace::Capture;

/// An address space a move touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Space {
    /// A GDDR channel.
    Gddr(u8),
    /// A unit's L1.
    L1(u16),
}

/// One move's footprint.
#[derive(Clone, Copy, Debug)]
enum Access {
    Range {
        space: Space,
        lo: u64,
        hi: u64,
        write: bool,
    },
    /// Reads all of GDDR (a read the host cannot place).
    AllGddr,
    /// Anything: a dependency on every unit both ways.
    Unknown,
}

/// Entries that touch nothing another unit can see.
fn local(head: u32) -> bool {
    matches!(
        head,
        op::PAIR
            | op::CB
            | op::SHARED
            | op::RELEASED
            | op::COPY_WORDS
            | op::COPY_HALFWORDS
            | op::FILL_HALFWORDS
            | op::CHECK_FLAGS
            | op::INDEX_PICK
            | op::FILL
            | op::KERNEL
            | op::WAIT
            | op::BARRIER
            | op::POKE
            | op::LAUNCH
            | op::KERNEL_WAIT
            | op::TILIZE
            | op::UNTILIZE
    ) || head == record::POKE_RUN
}

/// The footprint of one plain (or expanded) move `w` by a unit, `units` the session's coordinates.
fn access(w: [u32; 8], units: &[(u8, u8)], out: &mut Vec<Access>) {
    let unit_at = |x: u32, y: u32| {
        units
            .iter()
            .position(|&(ux, uy)| (ux as u32, uy as u32) == (x, y))
            .map(|u| u as u16)
    };
    match w[0] {
        op::READ | op::READ_TRANSPOSED | op::READ_BROADCAST_COL | op::WRITE => {
            let (lo, len) = (u64::from(w[3]), u64::from(w[5]));
            out.push(Access::Range {
                space: Space::Gddr(w[1] as u8),
                lo,
                hi: lo + len,
                write: w[0] == op::WRITE,
            });
        }
        op::READ_L1 | op::WRITE_L1 => {
            let Some(u) = unit_at(w[1], w[2]) else {
                out.push(Access::Unknown);
                return;
            };
            let (lo, len) = (u64::from(w[3]), u64::from(w[5]));
            out.push(Access::Range {
                space: Space::L1(u),
                lo,
                hi: lo + len,
                write: w[0] == op::WRITE_L1,
            });
        }
        op::WRITE_L1_MCAST => {
            let c = w[1];
            let (x0, y0, x1, y1) = (c & 0xff, (c >> 8) & 0xff, (c >> 16) & 0xff, c >> 24);
            let (lo, len) = (u64::from(w[3]), u64::from(w[5]));
            for (u, &(x, y)) in units.iter().enumerate() {
                let (x, y) = (x as u32, y as u32);
                if (x0..=x1).contains(&x) && (y0..=y1).contains(&y) {
                    out.push(Access::Range {
                        space: Space::L1(u as u16),
                        lo,
                        hi: lo + len,
                        write: true,
                    });
                }
            }
        }
        op::READ_ENCODED => out.push(Access::AllGddr),
        h if local(h) => {}
        _ => out.push(Access::Unknown),
    }
}

/// The footprint of the entries `entries` (a list, a stream range or a section).
fn accesses(entries: &[[u32; 8]], units: &[(u8, u8)], out: &mut Vec<Access>) {
    let mut i = 0;
    while i < entries.len() {
        let head = entries[i];
        let n = record::len(head[0]).max(1);
        if record::is_record(head[0]) {
            let Some(rec) = entries.get(i..i + n) else {
                out.push(Access::Unknown);
                return;
            };
            if matches!(head[0], record::SELECT_ROWS | record::SCATTER_ROWS) {
                indexed(rec, out);
            } else if head[0] != record::POKE_RUN
                && record::expand(rec, |w| {
                    access(w, units, out);
                    Ok(())
                })
                .is_err()
            {
                out.push(Access::Unknown);
            }
        } else {
            access(head, units, out);
        }
        i += n;
    }
}

/// A record whose rows come from indices in L1: every row its index could name, as one range a
/// channel (from its lowest byte to its highest).
fn indexed(rec: &[[u32; 8]], out: &mut Vec<Access>) {
    let bound = rec[0][3];
    let mut extent: BTreeMap<u8, (u64, u64, bool)> = BTreeMap::new();
    let mut ok = true;
    for v in 0..bound.max(1) {
        let mut emit = |w: [u32; 8]| {
            if matches!(w[0], op::READ | op::WRITE) {
                let (lo, hi) = (u64::from(w[3]), u64::from(w[3]) + u64::from(w[5]));
                let e = extent
                    .entry(w[1] as u8)
                    .or_insert((lo, hi, w[0] == op::WRITE));
                e.0 = e.0.min(lo);
                e.1 = e.1.max(hi);
            }
            Ok(())
        };
        let r = if rec[0][0] == record::SELECT_ROWS {
            record::expand_reads(rec, |_| v, &mut emit)
        } else {
            record::expand_writes(rec, |_| v, &mut emit)
        };
        if r.is_err() {
            ok = false;
            break;
        }
    }
    if !ok || extent.is_empty() {
        out.push(Access::Unknown);
        return;
    }
    for (c, (lo, hi, write)) in extent {
        out.push(Access::Range {
            space: Space::Gddr(c),
            lo,
            hi,
            write,
        });
    }
}

/// A barrier's class ([`Analysis::classes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// No unit of the next op depends on another unit's op since the last barrier kept.
    Unneeded,
    /// Each unit of the next op depends on some of the units since the last barrier kept.
    Sparse,
    /// Some unit of the next op depends on every unit since the last barrier kept, or on what
    /// cannot be placed.
    Full,
}

/// What [`analyse`] found for one capture.
#[derive(Clone, Debug, Default)]
pub struct Analysis {
    /// Per op: its label, its first list's name and the units it ran on.
    pub ops: Vec<(&'static str, &'static str, Vec<u16>)>,
    /// Per op `k`, per unit `u` it ran on: the latest op of each other unit `p` it must follow.
    pub deps: Vec<BTreeMap<u16, BTreeMap<u16, u32>>>,
    /// Per op `k > 0`, the class of the barrier between op `k - 1` and op `k` (index `k - 1`).
    pub classes: Vec<Class>,
    /// Moves that could not be placed (each a dependency on everything).
    pub unknown: u64,
}

impl Analysis {
    /// Barriers of each class: `[unneeded, sparse, full]`.
    pub fn counts(&self) -> [usize; 3] {
        let mut n = [0; 3];
        for c in &self.classes {
            n[match c {
                Class::Unneeded => 0,
                Class::Sparse => 1,
                Class::Full => 2,
            }] += 1;
        }
        n
    }

    /// One line an op: index, label, what, units, the class of the barrier before it, and each
    /// unit's dependencies (`u<-p:j,p:j`).
    pub fn tsv(&self) -> String {
        let mut out = String::from("index\tlabel\twhat\tunits\tbefore\tdeps\n");
        for (k, ((label, what, units), deps)) in self.ops.iter().zip(&self.deps).enumerate() {
            let class = match k.checked_sub(1).map(|i| self.classes[i]) {
                None => "-",
                Some(Class::Unneeded) => "unneeded",
                Some(Class::Sparse) => "sparse",
                Some(Class::Full) => "full",
            };
            let units: Vec<String> = units.iter().map(|u| u.to_string()).collect();
            let deps: Vec<String> = deps
                .iter()
                .map(|(u, d)| {
                    let d: Vec<String> = d.iter().map(|(p, j)| format!("{p}:{j}")).collect();
                    format!("{u}<-{}", d.join(","))
                })
                .collect();
            let _ = writeln!(
                out,
                "{k}\t{label}\t{what}\t{}\t{class}\t{}",
                units.join(","),
                deps.join(" ")
            );
        }
        out
    }
}

/// A segment of an address space: the last writer and each unit's latest reader since.
#[derive(Clone, Debug, Default)]
struct Seg {
    end: u64,
    writer: Option<(u16, u32)>,
    readers: Vec<(u16, u32)>,
}

/// Per space, segments by start; only the ranges some op writes are tracked (a range nothing
/// writes orders nothing).
#[derive(Default)]
struct Memory {
    spaces: HashMap<Space, BTreeMap<u64, Seg>>,
}

impl Memory {
    /// Split so that a segment starts at `at`, if one covers it.
    fn split(map: &mut BTreeMap<u64, Seg>, at: u64) {
        let Some((&start, seg)) = map.range(..at).next_back() else {
            return;
        };
        if seg.end > at {
            let right = seg.clone();
            map.get_mut(&start).unwrap().end = at;
            map.insert(at, right);
        }
    }

    /// Op `k` on unit `u` touches `lo..hi`: the other units' earlier ops it must follow go to
    /// `deps`; with `apply`, the access is recorded instead (after every unit of the op has found
    /// its dependencies: an op's units run together, and none follows another's share of it).
    fn touch(
        &mut self,
        space: Space,
        range: std::ops::Range<u64>,
        write: bool,
        (u, k): (u16, u32),
        deps: &mut BTreeMap<u16, u32>,
        apply: bool,
    ) {
        let (lo, hi) = (range.start, range.end);
        let Some(map) = self.spaces.get_mut(&space) else {
            return;
        };
        Self::split(map, lo);
        Self::split(map, hi);
        let mut need = |p: u16, j: u32| {
            if p != u {
                let e = deps.entry(p).or_insert(j);
                *e = (*e).max(j);
            }
        };
        for (_, seg) in map.range_mut(lo..hi) {
            if !apply {
                if let Some((p, j)) = seg.writer {
                    need(p, j);
                }
                if write {
                    for &(p, j) in &seg.readers {
                        need(p, j);
                    }
                }
                continue;
            }
            if write {
                seg.writer = Some((u, k));
                seg.readers.clear();
            } else {
                match seg.readers.iter_mut().find(|r| r.0 == u) {
                    Some(r) => r.1 = k,
                    None => seg.readers.push((u, k)),
                }
            }
        }
    }
}

/// The dependencies and barrier classes of capture `c`, whose units are at `coords`.
pub(crate) fn analyse(c: &Capture, coords: &[(u8, u8)]) -> Analysis {
    // Each (op, unit)'s footprint.
    let mut feet: Vec<BTreeMap<u16, Vec<Access>>> = Vec::with_capacity(c.ops.len());
    let mut ops = Vec::with_capacity(c.ops.len());
    let mut unknown = 0u64;
    for rec in &c.ops {
        let mut per: BTreeMap<u16, Vec<Access>> = BTreeMap::new();
        let mut units = Vec::new();
        for (u, (uc, range)) in c.units.iter().zip(&rec.entries).enumerate() {
            if range.is_empty() {
                continue;
            }
            units.push(u as u16);
            let out = per.entry(u as u16).or_default();
            let stream = &uc.stream[range.clone()];
            let mut i = 0;
            while i < stream.len() {
                if stream[i][0] == op::PAIR_CALL {
                    match uc.pair_calls.iter().position(|&at| at == range.start + i) {
                        Some(p) => {
                            for section in uc.pending_sections.iter().skip(2 * p).take(2) {
                                accesses(section, coords, out);
                            }
                        }
                        None => out.push(Access::Unknown),
                    }
                    i += 1;
                    continue;
                }
                let n = record::len(stream[i][0]).max(1);
                accesses(&stream[i..(i + n).min(stream.len())], coords, out);
                i += n;
            }
        }
        for (u, r) in &rec.ahead {
            if !rec.entries[*u].contains(&r.start) {
                let out = per.entry(*u as u16).or_default();
                accesses(&c.units[*u].stream[r.clone()], coords, out);
            }
        }
        unknown += per
            .values()
            .flatten()
            .filter(|a| !matches!(a, Access::Range { .. }))
            .count() as u64;
        feet.push(per);
        ops.push((rec.label, rec.what, units));
    }
    // What is written anywhere: only those ranges are tracked.
    let mut memory = Memory::default();
    for per in &feet {
        for a in per.values().flatten() {
            if let Access::Range {
                space,
                lo,
                hi,
                write: true,
            } = *a
            {
                let map = memory.spaces.entry(space).or_default();
                Memory::split(map, lo);
                Memory::split(map, hi);
                // Fill the gaps in `lo..hi` with fresh segments.
                let mut at = lo;
                let starts: Vec<(u64, u64)> = map.range(lo..hi).map(|(&s, g)| (s, g.end)).collect();
                for (s, e) in starts {
                    if s > at {
                        map.insert(
                            at,
                            Seg {
                                end: s,
                                ..Seg::default()
                            },
                        );
                    }
                    at = at.max(e);
                }
                if at < hi {
                    map.insert(
                        at,
                        Seg {
                            end: hi,
                            ..Seg::default()
                        },
                    );
                }
            }
        }
    }
    // The latest op each unit ran so far: what "everything" means for an unplaced move.
    let mut latest: BTreeMap<u16, u32> = BTreeMap::new();
    // The ops since an unplaced move or an all-GDDR read, which every later write must follow.
    let mut deps_all: Vec<BTreeMap<u16, BTreeMap<u16, u32>>> = Vec::with_capacity(feet.len());
    let mut last_unknown: Option<BTreeMap<u16, u32>> = None;
    let mut gddr_readers_all: BTreeMap<u16, u32> = BTreeMap::new();
    for (k, per) in feet.iter().enumerate() {
        let k32 = k as u32;
        let mut op_deps: BTreeMap<u16, BTreeMap<u16, u32>> = BTreeMap::new();
        for (&u, accs) in per {
            let mut d: BTreeMap<u16, u32> = BTreeMap::new();
            let add = |d: &mut BTreeMap<u16, u32>, from: &BTreeMap<u16, u32>| {
                for (&p, &j) in from {
                    if p != u {
                        let e = d.entry(p).or_insert(j);
                        *e = (*e).max(j);
                    }
                }
            };
            if let Some(all) = &last_unknown {
                add(&mut d, all);
            }
            for a in accs {
                match *a {
                    Access::Range {
                        space,
                        lo,
                        hi,
                        write,
                    } => {
                        if write && matches!(space, Space::Gddr(_)) {
                            add(&mut d, &gddr_readers_all.clone());
                        }
                        memory.touch(space, lo..hi, write, (u, k32), &mut d, false);
                    }
                    Access::AllGddr | Access::Unknown => {
                        add(&mut d, &latest.clone());
                    }
                }
            }
            op_deps.insert(u, d);
        }
        for (&u, accs) in per {
            let mut none = BTreeMap::new();
            for a in accs {
                if let Access::Range {
                    space,
                    lo,
                    hi,
                    write,
                } = *a
                {
                    memory.touch(space, lo..hi, write, (u, k32), &mut none, true);
                }
            }
            if accs.iter().any(|a| matches!(a, Access::Unknown)) {
                let mut all = latest.clone();
                all.insert(u, k32);
                last_unknown = Some(all);
            }
            if accs.iter().any(|a| matches!(a, Access::AllGddr)) {
                gddr_readers_all.insert(u, k32);
            }
        }
        for &u in per.keys() {
            latest.insert(u, k32);
        }
        deps_all.push(op_deps);
    }
    // Keep barriers greedily: op `k` joins the run since the last kept barrier unless one of its
    // units depends on another unit's op in that run.
    let mut classes = Vec::with_capacity(feet.len().saturating_sub(1));
    let mut start = 0u32;
    for (k, deps) in deps_all.iter().enumerate().skip(1) {
        let since: BTreeSet<u16> = feet[start as usize..k]
            .iter()
            .flat_map(|p| p.keys().copied())
            .collect();
        let needs = deps.values().any(|d| d.values().any(|&j| j >= start));
        if !needs {
            classes.push(Class::Unneeded);
            continue;
        }
        let full = deps.iter().any(|(u, d)| {
            let from: BTreeSet<u16> = d
                .iter()
                .filter(|&(_, &j)| j >= start)
                .map(|(&p, _)| p)
                .collect();
            since.iter().all(|p| p == u || from.contains(p))
        });
        classes.push(if full { Class::Full } else { Class::Sparse });
        start = k as u32;
    }
    Analysis {
        ops,
        deps: deps_all,
        classes,
        unknown,
    }
}

static PROBE: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();

/// A development probe: `TT_PROBE_BARRIERS=<dir>` analyses every stored trace's capture as it ends
/// and writes `<dir>/barriers-<trace>.tsv` ([`Analysis::tsv`]), printing a `BARRIERS` line.
pub(crate) fn probe_dir() -> Option<&'static std::path::Path> {
    PROBE
        .get_or_init(|| std::env::var_os("TT_PROBE_BARRIERS").map(Into::into))
        .as_deref()
}

/// Write `a` for trace `id` into the probe's directory.
pub(crate) fn write_probe(id: u64, a: &Analysis) {
    let Some(dir) = probe_dir() else {
        return;
    };
    let [unneeded, sparse, full] = a.counts();
    eprintln!(
        "BARRIERS trace {id}: {} ops, barriers unneeded {unneeded}, sparse {sparse}, full {full}, \
         unplaced moves {}",
        a.ops.len(),
        a.unknown
    );
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(dir.join(format!("barriers-{id}.tsv")), a.tsv());
}

/// The barriers [`analyse`] found unneeded, as `keep` (one per op: is the barrier after it kept?
/// the last op's always is). With [`probe_prune_every_barrier`] on (a negative control), every
/// barrier but the last is dropped, needed or not.
pub(crate) fn keep_from(a: &Analysis) -> Vec<bool> {
    let all = PRUNE_ALL.load(std::sync::atomic::Ordering::Relaxed);
    let n = a.ops.len();
    (0..n)
        .map(|k| k + 1 == n || (!all && a.classes[k] != Class::Unneeded))
        .collect()
}

static PRUNE_ALL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A negative control for the barrier pruning's gate: prune every captured barrier but a trace's
/// last, whatever its data needs. Process-wide; never for real work.
#[doc(hidden)]
pub fn probe_prune_every_barrier(on: bool) {
    PRUNE_ALL.store(on, std::sync::atomic::Ordering::SeqCst);
}

/// Labels of merged ops ("a + b"), interned: an `OpRecord` names its op by a `&'static str`.
fn intern(s: String) -> &'static str {
    static NAMES: std::sync::Mutex<BTreeSet<&'static str>> = std::sync::Mutex::new(BTreeSet::new());
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&n) = names.get(s.as_str()) {
        return n;
    }
    let n: &'static str = Box::leak(s.into_boxed_str());
    names.insert(n);
    n
}

/// What [`prune`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pruned {
    /// Ops before, and after merging the runs between kept barriers.
    pub ops: [usize; 2],
    /// Arrivals a replay adds, before and after.
    pub arrivals: [u32; 2],
    /// `BARRIER` entries removed from the streams.
    pub entries: usize,
}

/// Drop the barriers `keep` does not keep from capture `c`'s streams (`docs/plans/decode-small-ops.md`
/// S5, A1). The ops between two kept barriers become one op (a run): every unit that ran any of
/// them arrives once, after its last, at the run's barrier; its earlier arrivals in the run go. A
/// wait-only entry waits for the last kept barrier at or before the one it waited for (none: it
/// goes), and stands for the runs since the unit's last count; one that stands for none goes. The
/// cumulative targets, the trace's arrivals, the ops' stream ranges, the `PAIR_CALL` positions and
/// the lists read ahead follow. Nothing changes unless every check holds: each arrival's target is
/// its op's, each wait's names an op's, no wait waits for a run its unit has still to arrive at,
/// and every unit's count ends at the number of runs.
pub(crate) fn prune(c: &mut Capture, keep: &[bool]) -> Result<Pruned, String> {
    let n = c.ops.len();
    if n == 0 || keep.len() != n || !keep[n - 1] {
        return Err("no ops, or the last barrier not kept".into());
    }
    if c.ops.iter().any(|o| !o.barrier) {
        return Err("an op without a barrier".into());
    }
    let is_barrier = |e: &[u32; 8]| e[0] == op::BARRIER;
    let arrival = |e: &[u32; 8]| is_barrier(e) && e[4] == 0;
    // Each op's arrivals, from the streams: the units with an arrival in its range.
    let mut arrivals: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (k, o) in c.ops.iter().enumerate() {
        for (u, r) in o.entries.iter().enumerate() {
            let stream = &c.units[u].stream;
            let found = r.clone().filter(|&i| arrival(&stream[i])).count();
            match found {
                0 => {}
                1 => arrivals[k].push(u),
                _ => return Err(format!("op {k}: unit {u} arrives twice")),
            }
        }
        if arrivals[k].is_empty() {
            return Err(format!("op {k}: no unit arrives"));
        }
    }
    let mut target = Vec::with_capacity(n);
    let mut sum = 0u32;
    for a in &arrivals {
        sum += a.len() as u32;
        target.push(sum);
    }
    if sum != c.barriers {
        return Err(format!(
            "{sum} arrivals in the streams, {} counted",
            c.barriers
        ));
    }
    let op_of = |t: u32| target.binary_search(&t).ok();
    // Runs.
    let mut run = Vec::with_capacity(n);
    let mut g = 0usize;
    for &kept in keep {
        run.push(g);
        g += kept as usize;
    }
    let runs = g;
    let mut run_units: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); runs];
    // Each unit's last op with an arrival, per run.
    let mut last_in_run: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (k, a) in arrivals.iter().enumerate() {
        for &u in a {
            run_units[run[k]].insert(u);
            last_in_run.insert((u, run[k]), k);
        }
    }
    let mut new_target = Vec::with_capacity(runs);
    let mut sum = 0u32;
    for us in &run_units {
        sum += us.len() as u32;
        new_target.push(sum);
    }
    let new_arrivals = sum;
    // The kept barrier at or before op `b`'s: its run.
    let kept_at_or_before = |b: usize| -> Option<usize> {
        if keep[b] {
            Some(run[b])
        } else {
            run[b].checked_sub(1)
        }
    };
    // Each op's owner on each unit, to tell which op an arrival belongs to.
    let mut streams: Vec<Vec<[u32; 8]>> = Vec::with_capacity(c.units.len());
    let mut maps: Vec<Vec<usize>> = Vec::with_capacity(c.units.len());
    let mut removed = 0usize;
    for (u, uc) in c.units.iter().enumerate() {
        let mut owner: Vec<Option<usize>> = vec![None; uc.stream.len()];
        for (k, o) in c.ops.iter().enumerate() {
            for i in o.entries[u].clone() {
                owner[i] = Some(k);
            }
        }
        let mut out = Vec::with_capacity(uc.stream.len());
        let mut map = Vec::with_capacity(uc.stream.len() + 1);
        // (index in `out`, run waited for) of each wait kept so far, and of each arrival.
        let mut marks: Vec<(usize, bool, usize)> = Vec::new();
        let mut i = 0;
        while i < uc.stream.len() {
            let e = uc.stream[i];
            let len = record::len(e[0]).max(1);
            map.push(out.len());
            if is_barrier(&e) {
                if e[4] == 0 {
                    let k =
                        owner[i].ok_or_else(|| format!("unit {u}: an arrival outside every op"))?;
                    if target[k] != e[1] {
                        return Err(format!(
                            "unit {u}, op {k}: arrival target {} is not {}",
                            e[1], target[k]
                        ));
                    }
                    if last_in_run.get(&(u, run[k])) == Some(&k) {
                        let mut a = e;
                        a[1] = new_target[run[k]];
                        marks.push((out.len(), true, run[k]));
                        out.push(a);
                    } else {
                        removed += 1;
                    }
                } else {
                    let b = op_of(e[1])
                        .ok_or_else(|| format!("unit {u}: a wait for {}, no op's target", e[1]))?;
                    match kept_at_or_before(b) {
                        Some(h) => {
                            let mut w = e;
                            w[1] = new_target[h];
                            marks.push((out.len(), false, h));
                            out.push(w);
                        }
                        None => removed += 1,
                    }
                }
            } else {
                let at = out.len();
                let end = (i + len).min(uc.stream.len());
                out.extend_from_slice(&uc.stream[i..end]);
                for j in 1..end - i {
                    map.push(at + j);
                }
            }
            i += len;
        }
        map.push(out.len());
        // No wait for a run the unit arrives at later; counts: arrivals one, waits what they stand for.
        let mut accounted = 0u32;
        let mut drop = BTreeSet::new();
        for (m, &(at, is_arrival, h)) in marks.iter().enumerate() {
            if is_arrival {
                if (h as u32) < accounted {
                    return Err(format!(
                        "unit {u}: arrives at run {h} after counting {accounted}"
                    ));
                }
                if h as u32 != accounted {
                    return Err(format!(
                        "unit {u}: arrives at run {h} with {accounted} counted, no wait between"
                    ));
                }
                accounted = h as u32 + 1;
            } else {
                if let Some(&(_, _, later)) = marks[m + 1..].iter().find(|x| x.1) {
                    if h >= later {
                        return Err(format!(
                            "unit {u}: waits for run {h} before arriving at run {later}"
                        ));
                    }
                }
                let stands = (h as u32 + 1).saturating_sub(accounted);
                if stands == 0 {
                    drop.insert(at);
                } else {
                    out[at][tt_isa::dm::BARRIER_STANDS_FOR] = stands;
                    accounted = h as u32 + 1;
                }
            }
        }
        if !uc.stream.is_empty() && accounted as usize != runs {
            return Err(format!("unit {u}: counts {accounted} runs of {runs}"));
        }
        // Drop the waits that stand for nothing, and shift the map past them.
        if !drop.is_empty() {
            removed += drop.len();
            let mut shift = vec![0usize; out.len() + 1];
            let mut kept = Vec::with_capacity(out.len());
            for (j, e) in out.iter().enumerate() {
                shift[j] = kept.len();
                if !drop.contains(&j) {
                    kept.push(*e);
                }
            }
            shift[out.len()] = kept.len();
            for m in map.iter_mut() {
                *m = shift[*m];
            }
            out = kept;
        }
        streams.push(out);
        maps.push(map);
    }
    // Every check held: rewrite.
    let remap = |map: &Vec<usize>, r: &std::ops::Range<usize>| map[r.start]..map[r.end];
    let mut merged: Vec<crate::trace::OpRecord> = Vec::with_capacity(runs);
    for (k, o) in c.ops.iter().enumerate() {
        let mut o = o.clone();
        for (u, r) in o.entries.iter_mut().enumerate() {
            *r = remap(&maps[u], r);
        }
        for (u, r) in o.ahead.iter_mut() {
            *r = remap(&maps[*u], r);
        }
        for (_, t) in o.ahead_waits.iter_mut() {
            if let Some(h) = op_of(*t).and_then(kept_at_or_before) {
                *t = new_target[h];
            }
        }
        if k > 0 && run[k] == run[k - 1] {
            let m = merged.last_mut().expect("a run's first op");
            m.label = intern(format!("{} + {}", m.label, o.label));
            m.what = intern(format!("{} + {}", m.what, o.what));
            for (r, s) in m.entries.iter_mut().zip(&o.entries) {
                if r.start >= r.end {
                    *r = s.clone();
                } else if s.start < s.end {
                    *r = r.start.min(s.start)..r.end.max(s.end);
                }
            }
            m.ahead.extend(o.ahead);
            m.ahead_waits.extend(o.ahead_waits);
            m.l1_top = m.l1_top.max(o.l1_top);
            m.l1_need = m.l1_need.max(o.l1_need);
            m.allocs = m.allocs.max(o.allocs);
        } else {
            merged.push(o);
        }
    }
    for (uc, (stream, map)) in c.units.iter_mut().zip(streams.into_iter().zip(&maps)) {
        for p in uc.pair_calls.iter_mut() {
            *p = map[*p];
        }
        uc.stream = stream;
        uc.barriers_accounted = runs as u32;
    }
    let out = Pruned {
        ops: [n, runs],
        arrivals: [c.barriers, new_arrivals],
        entries: removed,
    };
    c.ops = merged;
    c.barriers = new_arrivals;
    c.barrier_ops = runs as u32;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{OpRecord, UnitCapture};

    /// The units' coordinates: unit `u` at `(u + 1, 2)`.
    fn coords(n: usize) -> Vec<(u8, u8)> {
        (0..n).map(|u| (u as u8 + 1, 2)).collect()
    }

    /// A capture of `ops` (per op, per unit, the unit's entries; none where it sat the op out) on
    /// `n` units, laid out as a capture lays them out: a unit that sat ops out waits for the latest
    /// barrier before its next op and at the end, each op ends in an arrival on every unit it ran
    /// on, the targets cumulative.
    fn capture(n: usize, ops: &[Vec<Option<Vec<[u32; 8]>>>]) -> Capture {
        let mut units: Vec<UnitCapture> = (0..n).map(|_| UnitCapture::default()).collect();
        let mut records = Vec::new();
        let mut target = 0u32;
        let mut seen = vec![0u32; n];
        let barrier =
            |t: u32, wait: bool, stands: u32| [op::BARRIER, t, 1, 2, wait as u32, stands, 1, 0];
        for (k, per) in ops.iter().enumerate() {
            let ran: Vec<usize> = (0..n)
                .filter(|&u| per.get(u).is_some_and(Option::is_some))
                .collect();
            let starts: Vec<usize> = units.iter().map(|u| u.stream.len()).collect();
            for &u in &ran {
                if seen[u] != k as u32 {
                    units[u]
                        .stream
                        .push(barrier(target, true, k as u32 - seen[u]));
                }
                units[u].stream.extend(per[u].clone().unwrap());
            }
            target += ran.len() as u32;
            for &u in &ran {
                units[u].stream.push(barrier(target, false, 0));
                seen[u] = k as u32 + 1;
            }
            records.push(OpRecord {
                what: "w",
                label: "l",
                entries: units
                    .iter()
                    .zip(starts)
                    .map(|(u, s)| s..u.stream.len())
                    .collect(),
                barrier: true,
                stamps: [0; 2],
                bytes: Default::default(),
                unit_bytes: Vec::new(),
                ahead: Vec::new(),
                ahead_waits: Vec::new(),
                l1_top: 0,
                l1_need: 0,
                shape: 0,
                allocs: 0,
            });
        }
        let total = ops.len() as u32;
        for (u, uc) in units.iter_mut().enumerate() {
            if !uc.stream.is_empty() && seen[u] != total {
                uc.stream.push(barrier(target, true, total - seen[u]));
            }
        }
        Capture {
            epoch: 0,
            barriers_base: 0,
            barriers: target,
            barrier_ops: total,
            units,
            freed: Vec::new(),
            ops: records,
            failed: false,
            weight_anchor: None,
            weight_units: 0,
            unit_anchors: Vec::new(),
            written: Vec::new(),
            region_base: None,
            applied: None,
            plan_mismatch: None,
            planned: false,
            plan_pass: false,
            plan_key: None,
            index: Default::default(),
            reuse: false,
            pooled: Vec::new(),
            pool: Vec::new(),
        }
    }

    fn read(at: u32) -> [u32; 8] {
        [op::READ, 0, 0, at, 0x1000, 64, 0, 0]
    }
    fn write(at: u32) -> [u32; 8] {
        [op::WRITE, 0, 0, at, 0x1000, 64, 0, 0]
    }
    fn local() -> [u32; 8] {
        [op::LAUNCH, 1, 0, 0, 0, 0, 0, 0]
    }

    /// The classes of a capture of two or three ops.
    fn classes(n: usize, ops: &[Vec<Option<Vec<[u32; 8]>>>]) -> Vec<Class> {
        analyse(&capture(n, ops), &coords(n)).classes
    }

    #[test]
    fn a_dependency_of_each_kind_keeps_its_barrier() {
        use Class::*;
        let x = 0x40_0000;
        // Read after write, on another unit, and on the same.
        assert_eq!(
            classes(
                2,
                &[
                    vec![None, Some(vec![write(x)])],
                    vec![Some(vec![read(x)]), None]
                ]
            ),
            [Full]
        );
        assert_eq!(
            classes(
                2,
                &[
                    vec![None, Some(vec![write(x)])],
                    vec![None, Some(vec![read(x)])]
                ]
            ),
            [Unneeded]
        );
        // Write after read, and after write.
        assert_eq!(
            classes(
                2,
                &[
                    vec![None, Some(vec![read(x)])],
                    vec![Some(vec![write(x)]), None]
                ]
            ),
            [Full]
        );
        assert_eq!(
            classes(
                2,
                &[
                    vec![None, Some(vec![write(x)])],
                    vec![Some(vec![write(x)]), None]
                ]
            ),
            [Full]
        );
        // Reads after reads, and accesses that do not overlap, need nothing.
        assert_eq!(
            classes(
                2,
                &[
                    vec![None, Some(vec![read(x)])],
                    vec![Some(vec![read(x)]), None]
                ]
            ),
            [Unneeded]
        );
        assert_eq!(
            classes(
                2,
                &[
                    vec![None, Some(vec![write(x)])],
                    vec![Some(vec![read(x + 64)]), None]
                ]
            ),
            [Unneeded]
        );
        // Another tile's L1: a unicast and a multicast write, read by the tile itself.
        let c = coords(3);
        let wl1 = [
            op::WRITE_L1,
            c[0].0 as u32,
            c[0].1 as u32,
            0x8000,
            0x1000,
            64,
            0,
            0,
        ];
        let rl1 = [
            op::READ_L1,
            c[0].0 as u32,
            c[0].1 as u32,
            0x8000,
            0x2000,
            64,
            0,
            0,
        ];
        let corners = tt_isa::dm::mcast_corners(c[0], c[1]);
        let mcast = [op::WRITE_L1_MCAST, corners, 0, 0x8000, 0x1000, 64, 0, 0];
        assert_eq!(
            classes(
                3,
                &[
                    vec![None, None, Some(vec![wl1])],
                    vec![Some(vec![rl1]), None, None]
                ]
            ),
            [Full]
        );
        assert_eq!(
            classes(
                3,
                &[
                    vec![None, None, Some(vec![mcast])],
                    vec![Some(vec![rl1]), None, None]
                ]
            ),
            [Full]
        );
        // The multicast's rectangle stops at unit 1: unit 2's L1 is not written.
        let rl1_2 = [
            op::READ_L1,
            c[2].0 as u32,
            c[2].1 as u32,
            0x8000,
            0x2000,
            64,
            0,
            0,
        ];
        let mut w2 = vec![None, None, None];
        w2[2] = Some(vec![rl1_2]);
        assert_eq!(
            classes(3, &[vec![Some(vec![mcast]), None, None], w2]),
            [Unneeded]
        );
        // Whatever the analysis cannot place depends on everything.
        let host = [op::HOST_READ, 0, 0, 0, 0x1000, 64, 0, 0];
        assert_eq!(
            classes(
                2,
                &[
                    vec![None, Some(vec![local()])],
                    vec![Some(vec![host]), None]
                ]
            ),
            [Full]
        );
    }

    #[test]
    fn an_op_s_units_find_their_dependencies_before_any_records_its_own() {
        use Class::*;
        let x = 0x40_0000;
        // Unit 2 writes in op 0; in op 1 unit 1 writes the same bytes and unit 0 reads them: unit
        // 0 follows unit 2's op 0, not unit 1's share of its own op.
        let a = analyse(
            &capture(
                3,
                &[
                    vec![None, None, Some(vec![write(x)])],
                    vec![Some(vec![read(x)]), Some(vec![write(x)]), None],
                ],
            ),
            &coords(3),
        );
        assert_eq!(a.deps[1][&0], BTreeMap::from([(2, 0)]));
        assert_eq!(a.classes, [Full]);
        // Some of the units since: sparse.
        assert_eq!(
            classes(
                3,
                &[
                    vec![None, Some(vec![write(x)]), Some(vec![write(x + 64)])],
                    vec![Some(vec![read(x)]), None, None]
                ]
            ),
            [Sparse]
        );
        // A run: op 2 depends on op 0 across the unneeded barrier after op 0, so the barrier
        // after op 1 is kept.
        assert_eq!(
            classes(
                2,
                &[
                    vec![None, Some(vec![write(x)])],
                    vec![Some(vec![local()]), None],
                    vec![Some(vec![read(x)]), None],
                ]
            ),
            [Unneeded, Full]
        );
    }

    /// Each unit's barrier entries: `(target, wait, stands for)`.
    fn barriers_of(c: &Capture, u: usize) -> Vec<(u32, bool, u32)> {
        c.units[u]
            .stream
            .iter()
            .filter(|e| e[0] == op::BARRIER)
            .map(|e| (e[1], e[4] == 1, e[tt_isa::dm::BARRIER_STANDS_FOR]))
            .collect()
    }

    #[test]
    fn pruning_merges_a_run_and_renumbers_its_barriers() {
        let x = 0x40_0000;
        // Op 0 on units 0 and 1, op 1 on unit 1 alone (its own tiles), op 2 on unit 0 reading
        // what unit 1 wrote in op 1, op 3 on unit 2 alone.
        let ops = vec![
            vec![Some(vec![write(x)]), Some(vec![write(x + 64)]), None],
            vec![None, Some(vec![read(x + 64), write(x + 128)]), None],
            vec![Some(vec![read(x + 128)]), None, None],
            vec![None, None, Some(vec![local()])],
        ];
        let mut c = capture(3, &ops);
        let a = analyse(&c, &coords(3));
        assert_eq!(a.classes, [Class::Unneeded, Class::Full, Class::Unneeded]);
        let keep = keep_from(&a);
        assert_eq!(keep, [false, true, false, true]);
        let p = prune(&mut c, &keep).unwrap();
        assert_eq!(p.ops, [4, 2]);
        // Run 0 (ops 0-1): units 0 and 1 arrive, target 2; run 1 (ops 2-3): units 0 and 2, 4.
        assert_eq!(p.arrivals, [5, 4]);
        assert_eq!(c.barriers, 4);
        assert_eq!(barriers_of(&c, 0), [(2, false, 0), (4, false, 0)]);
        // Unit 1: its arrival at op 0 goes, its wait before op 1 (for op 0) stands for nothing.
        assert_eq!(barriers_of(&c, 1), [(2, false, 0), (4, true, 1)]);
        // Unit 2: its wait before op 3 is for run 0, then it arrives at run 1.
        assert_eq!(barriers_of(&c, 2), [(2, true, 1), (4, false, 0)]);
        // The ops: two, each unit's range its entries.
        assert_eq!(c.ops.len(), 2);
        assert_eq!(c.ops[0].label, "l + l");
        for (u, uc) in c.units.iter().enumerate() {
            let covered: usize = c.ops.iter().map(|o| o.entries[u].len()).sum();
            let trailing = uc.stream.len() - c.ops.iter().map(|o| o.entries[u].end).max().unwrap();
            assert_eq!(
                covered + trailing,
                uc.stream.len(),
                "unit {u}: every entry is an op's or the end's"
            );
        }
    }

    #[test]
    fn a_wait_for_a_run_its_unit_arrives_at_later_is_refused() {
        // A stream whose arrival at op 0 is missing, and whose wait for op 1 sits before op 1's
        // own arrival: the checks refuse it, and nothing changes.
        let mut c = capture(
            2,
            &[
                vec![Some(vec![local()]), Some(vec![local()])],
                vec![Some(vec![local()]), Some(vec![local()])],
            ],
        );
        c.units[1]
            .stream
            .insert(1, [op::BARRIER, 4, 1, 2, 1, 1, 1, 0]);
        for o in &mut c.ops {
            let r = &mut o.entries[1];
            *r = if r.start == 0 {
                0..r.end + 1
            } else {
                r.start + 1..r.end + 1
            };
        }
        let before = c.units[1].stream.clone();
        assert!(prune(&mut c, &[false, true]).is_err());
        assert_eq!(c.units[1].stream, before);
    }
}
