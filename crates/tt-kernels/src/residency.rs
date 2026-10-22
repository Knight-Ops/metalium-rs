//! Planned L1 residency (`docs/plans/planned-l1-residency.md`): which of a captured graph's tensors
//! could live in a persistent region at the top of each unit's `DATA` arena, from the op that makes
//! them to the last op that reads them, instead of going through GDDR in between.
//!
//! Phase 1 plans and reports; nothing is made resident yet. A plan is made over a capture as it
//! ends ([`Graph::from_capture`]):
//!
//! - **The graph** is read off the captured streams, not off the ops' builders: every record and
//!   plain move of every op on every unit is expanded as the movers would expand it, and each GDDR
//!   access is charged to the tensor whose slots it touches. The tensors a capture can plan are the
//!   ones allocated during it (its allocator kept them, `DramAlloc::start_log`); the rest -- weights,
//!   the cache, the inputs -- are external. A capture defers its frees, so no two of its tensors
//!   share an address, and one freed before it ended was made and used up inside it.
//! - **The period** is the one the space report finds: the ops' signatures repeat every layer
//!   ([`period`]). A tensor read or written at the same place of two repeats (the same op of the
//!   layer, the same of its tensors) gets the same address in both, so every repeat of a layer runs
//!   at the same L1 addresses: a tensor that crosses a layer's end (the residual stream) is pinned
//!   at both ends, and one the head makes for the first layer where the last layer made the next's.
//! - **Addresses** are assigned once, first fit, over lifetimes counted in ops and inclusive of the
//!   last reader, so an op never writes the slot of a tensor it reads. What does not fit leaves the
//!   plan largest `bytes x lifetime` first.
//! - **The check** ([`Plan::check`]) is independent of the assignment: no two resident tensors live
//!   at once in overlapping bytes, and the ops one period apart see the same addresses.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use tt_isa::dm::{op, record};

use crate::tensor::Placement;
use crate::trace::{Capture, UnitCapture};

/// Addresses in the region are multiples of this: a GDDR read's congruence unit, which a tile slot
/// read into L1 keeps.
pub const ALIGN: u64 = tt_isa::dram::ALIGN;

static PROBE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static PLANS: std::sync::Mutex<Vec<(Graph, Plan)>> = std::sync::Mutex::new(Vec::new());

/// A development probe (`docs/plans/planned-l1-residency.md` Phase 1): every capture plans its
/// graph as it ends, with an L1 path assumed for every op, prints the plan (`RESIDENCY ...`) and
/// keeps it for [`take_plans`]. Nothing is made resident. Process-wide; dev-facing.
#[doc(hidden)]
pub fn probe_residency_plan(on: bool) {
    PROBE.store(on, std::sync::atomic::Ordering::SeqCst);
}

pub(crate) fn probing() -> bool {
    PROBE.load(std::sync::atomic::Ordering::Relaxed)
}

/// The graphs and plans [`probe_residency_plan`] made since the last call, in capture order.
#[doc(hidden)]
pub fn take_plans() -> Vec<(Graph, Plan)> {
    std::mem::take(&mut *PLANS.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Plan capture `c`, ending ([`probe_residency_plan`]), print it and keep it.
pub(crate) fn probe(c: &Capture, allocs: &[Placement]) {
    let g = Graph::from_capture(c, allocs);
    let plan = Plan::make(&g, tt_isa::l1::DATA.end, |_, _| true);
    keep(g, plan);
}

/// Print a plan of `g` ([`probe_residency_plan`]) and keep it for [`take_plans`].
pub(crate) fn keep(g: Graph, plan: Plan) {
    eprint!("{}", plan.report(&g));
    PLANS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((g, plan));
}

/// One op of a captured graph, as the plan sees it.
#[derive(Clone, Debug)]
pub struct Op {
    pub label: &'static str,
    pub what: &'static str,
    /// Equal for the ops a repeated layer holds at the same place ([`signatures`]).
    pub signature: String,
    /// [`crate::trace::OpRecord::l1_top`] and [`crate::trace::OpRecord::l1_need`].
    pub l1_top: u64,
    pub l1_need: u64,
    /// The capture's own tensors it touches, in the order its streams first touch them (unit by
    /// unit): the `j`th is what the same op of another layer touches `j`th.
    pub tensors: Vec<usize>,
    /// GDDR bytes it reads and writes of tensors from outside the capture: `[read, written]`.
    pub external: [u64; 2],
}

/// One op's use of a tensor: per unit, the tiles it read and wrote.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Use {
    pub op: usize,
    pub reads: BTreeMap<usize, BTreeSet<u32>>,
    pub writes: BTreeMap<usize, BTreeSet<u32>>,
    /// GDDR bytes: `[read, written]`.
    pub bytes: [u64; 2],
    /// Every access was a whole move the planned capture can redirect ([`Applied::rewrite`]): a
    /// plain `READ` or `WRITE`, or one a `READ_RUN` without flags, a `WRITE_RUN` or a product's
    /// `SCATTER` expands to. Not a product's `GATHER`, a padding fill, a row select or scatter, a
    /// transposing or broadcasting read.
    pub plain: bool,
}

impl Use {
    fn reads(&self) -> bool {
        !self.reads.is_empty()
    }
    fn writes(&self) -> bool {
        !self.writes.is_empty()
    }
}

/// How one op's units read a tensor.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Layout {
    /// Every unit that reads it reads all of what any reads: a row each unit needs whole (a
    /// product's packed `A`, a norm's statistic).
    Replicated,
    /// Each tile on one unit only: tiles dealt in runs (element-wise ops), a head a unit.
    Sharded,
    /// Neither.
    Mixed,
}

impl Use {
    /// How this use's units read.
    pub fn layout(&self) -> Layout {
        let sets: Vec<&BTreeSet<u32>> = self.reads.values().collect();
        if sets.len() <= 1 {
            return Layout::Sharded;
        }
        if sets.iter().all(|s| *s == sets[0]) {
            return Layout::Replicated;
        }
        let total: usize = sets.iter().map(|s| s.len()).sum();
        let union: BTreeSet<u32> = sets.iter().flat_map(|s| s.iter().copied()).collect();
        if union.len() == total {
            Layout::Sharded
        } else {
            Layout::Mixed
        }
    }
}

/// One of a capture's own tensors.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub tiles: usize,
    pub slot_bytes: u64,
    /// Freed before the capture ended: made and used up inside it -- not its output or state.
    pub internal: bool,
    /// Its uses, in op order.
    pub uses: Vec<Use>,
}

impl Tensor {
    /// The ops it lives across, inclusive: from its first use to its last.
    pub fn lifetime(&self) -> Option<(usize, usize)> {
        Some((self.uses.first()?.op, self.uses.last()?.op))
    }

    /// The units that read it, over all its readers, and the tiles each reads.
    pub fn read_by(&self) -> BTreeMap<usize, BTreeSet<u32>> {
        let mut per_unit: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
        for u in &self.uses {
            for (&unit, tiles) in &u.reads {
                per_unit.entry(unit).or_default().extend(tiles);
            }
        }
        per_unit
    }

    /// What one unit holds of it in L1 to serve every op that reads it: the most tiles any one
    /// unit reads, over all its readers, in slots of its own size.
    pub fn bytes_a_unit(&self) -> u64 {
        let tiles = self.read_by().values().map(|s| s.len()).max().unwrap_or(0) as u64;
        (tiles * self.slot_bytes).div_ceil(ALIGN) * ALIGN
    }

    /// Whether an op reads it whole on more than one unit ([`Layout::Replicated`]): a row many
    /// units need, which [`Rows`] decides how to deliver.
    pub fn replicated(&self) -> bool {
        self.uses
            .iter()
            .any(|u| u.reads.len() > 1 && u.layout() == Layout::Replicated)
    }

    /// Whether every unit that reads it reads the same tiles, over all its readers: what one
    /// holder holds then serves any of them.
    pub fn read_alike(&self) -> bool {
        let by = self.read_by();
        let mut sets = by.values();
        sets.next().is_some_and(|first| sets.all(|s| s == first))
    }

    /// GDDR bytes its uses move: `[read, written]`.
    pub fn gddr_bytes(&self) -> [u64; 2] {
        self.uses
            .iter()
            .fold([0, 0], |[r, w], u| [r + u.bytes[0], w + u.bytes[1]])
    }
}

/// A captured graph: its ops and the tensors it made.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub ops: Vec<Op>,
    pub tensors: Vec<Tensor>,
    /// Moves that could not be read for what they touch (an encoded read, a record the host's
    /// expansion refused): the graph is incomplete by these.
    pub unattributed: u64,
}

/// Each op's signature: its label, its first list's name and its entries on each unit, sections
/// included -- equal for the same op of two layers ([`period`]).
pub(crate) fn signatures(c: &Capture) -> Vec<String> {
    c.ops
        .iter()
        .map(|op| {
            let per_unit: Vec<u64> = c
                .units
                .iter()
                .zip(&op.entries)
                .map(|(uc, r)| entries_with_sections(uc, r))
                .collect();
            format!("{}|{}|{per_unit:?}", op.label, op.what)
        })
        .collect()
}

/// Unit stream `uc`'s entries `r`, with the sections of the `PAIR_CALL`s among them.
pub(crate) fn entries_with_sections(uc: &UnitCapture, r: &std::ops::Range<usize>) -> u64 {
    let sections: u64 = uc
        .pair_calls
        .iter()
        .enumerate()
        .filter(|&(_, at)| r.contains(at))
        .map(|(i, _)| u64::from(uc.sections[2 * i] + uc.sections[2 * i + 1]))
        .sum();
    r.len() as u64 + sections
}

/// Where a graph's ops repeat: every `p` ops, `repeats` times from op `start`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Period {
    pub p: usize,
    pub start: usize,
    pub repeats: usize,
    /// How many ops equal the op `p` later, in the whole graph.
    pub matches: usize,
}

impl Period {
    /// The ops it covers.
    pub fn span(&self) -> std::ops::Range<usize> {
        self.start..self.start + self.p * self.repeats
    }

    /// Op `k`'s place in a repeat, if `k` is in one.
    pub fn phase(&self, k: usize) -> Option<usize> {
        self.span().contains(&k).then(|| (k - self.start) % self.p)
    }
}

/// The period at which `signatures` repeat most (the shortest of equals), and the longest run of
/// whole repeats at it; `None` for fewer than two repeats.
pub fn period(signatures: &[impl PartialEq]) -> Option<Period> {
    let n = signatures.len();
    let matches = |p: usize| {
        (0..n - p)
            .filter(|&k| signatures[k] == signatures[k + p])
            .count()
    };
    let p = (1..n / 2).max_by_key(|&p| (matches(p), std::cmp::Reverse(p)))?;
    let m = matches(p);
    if m == 0 {
        return None;
    }
    // The longest run of ops each equal to the one `p` later: `a..b` repeat into `a + p..b + p`.
    let (mut best, mut run) = ((0, 0), None);
    for k in 0..=n - p {
        let equal = k < n - p && signatures[k] == signatures[k + p];
        match (equal, run) {
            (true, None) => run = Some(k),
            (false, Some(a)) => {
                if k - a > best.1 - best.0 {
                    best = (a, k);
                }
                run = None;
            }
            _ => {}
        }
    }
    let (a, b) = best;
    let repeats = (b + p - a) / p;
    (repeats >= 2).then_some(Period {
        p,
        start: a,
        repeats,
        matches: m,
    })
}

/// One channel's share of a tensor's slots.
#[derive(Copy, Clone)]
struct Extent {
    end: u64,
    tensor: usize,
    /// The channel's place among the tensor's, of how many.
    pos: usize,
    n: usize,
    slot_bytes: u64,
}

/// Which GDDR slots belong to which of a capture's tensors.
#[derive(Default)]
pub(crate) struct Index {
    /// Per chip channel index, by start.
    channels: BTreeMap<u8, BTreeMap<u64, Extent>>,
    /// How many of the allocations it holds.
    len: usize,
}

impl Index {
    pub(crate) fn new(allocs: &[Placement]) -> Self {
        let mut index = Index::default();
        index.extend(allocs);
        index
    }

    /// Add the allocations of `allocs` past those it holds (`allocs` the same log, grown): what
    /// [`Index::new`] of the whole log would hold, a later tensor's slots over an earlier one's.
    pub(crate) fn extend(&mut self, allocs: &[Placement]) {
        for (tensor, p) in allocs.iter().enumerate().skip(self.len) {
            let n = p.extents().count();
            for (pos, (c, base, bytes)) in p.extents().enumerate() {
                self.channels.entry(c).or_default().insert(
                    base,
                    Extent {
                        end: base + bytes,
                        tensor,
                        pos,
                        n,
                        slot_bytes: p.slot_bytes(),
                    },
                );
            }
        }
        self.len = self.len.max(allocs.len());
    }

    /// The tensor and tile GDDR `(channel, offset)` is in.
    fn find(&self, channel: u32, offset: u32) -> Option<(usize, u32)> {
        self.locate(channel, offset).map(|(t, tile, _)| (t, tile))
    }

    /// [`Index::find`], and the byte of the tile's slot.
    pub(crate) fn locate(&self, channel: u32, offset: u32) -> Option<(usize, u32, u64)> {
        let at = u64::from(offset);
        let (&start, e) = self
            .channels
            .get(&u8::try_from(channel).ok()?)?
            .range(..=at)
            .next_back()?;
        let slot = ((at - start) / e.slot_bytes) as usize;
        let byte = (at - start) % e.slot_bytes;
        (at < e.end).then(|| (e.tensor, (slot * e.n + e.pos) as u32, byte))
    }
}

/// Hand each move list `entries` holds to `f`, records expanded as the host expands them, with
/// the plainness of what holds it ([`plain`]); the count of what could not be read is returned.
fn walk(entries: &[[u32; 8]], f: &mut impl FnMut([u32; 8], bool)) -> u64 {
    let mut unread = 0;
    let mut i = 0;
    while i < entries.len() {
        let head = entries[i][0];
        let n = record::len(head);
        if record::is_record(head) {
            let whole = plain(Some(entries[i]), entries[i]);
            let ok = i + n <= entries.len()
                && record::expand(&entries[i..i + n], |w| {
                    f(w, whole);
                    Ok(())
                })
                .is_ok();
            unread += u64::from(!ok);
        } else if head == op::READ_ENCODED {
            unread += 1;
        } else {
            f(entries[i], plain(None, entries[i]));
        }
        i += n.max(1);
    }
    unread
}

/// Whether move `w`, plain or expanded from the record `head`, is one [`Applied::rewrite`]
/// redirects: a plain `READ` or `WRITE`, or a `READ_RUN` without flags, a `WRITE_RUN` or a
/// `SCATTER`. A product's `SCATTER` expands to whole-tile writes as a `WRITE_RUN` does, so its
/// output can go straight into its consumers' L1 (`docs/plans/planned-l1-residency.md` Phase 3).
fn plain(head: Option<[u32; 8]>, w: [u32; 8]) -> bool {
    match head {
        None => matches!(w[0], op::READ | op::WRITE),
        Some(h) => {
            (h[0] == record::READ_RUN && h[4] == 0)
                || matches!(h[0], record::WRITE_RUN | record::SCATTER)
        }
    }
}

impl Graph {
    /// The graph of capture `c`, ending: `allocs` are the tensors allocated during it, in order
    /// ([`crate::tensor::DramAlloc::take_log`]).
    pub(crate) fn from_capture(c: &Capture, allocs: &[Placement]) -> Graph {
        let index = Index::new(allocs);
        let mut tensors: Vec<Tensor> = allocs
            .iter()
            .map(|p| Tensor {
                tiles: p.tiles(),
                slot_bytes: p.slot_bytes(),
                internal: c.freed.iter().any(|f| f.owned() && f.same_allocation(p)),
                uses: Vec::new(),
            })
            .collect();
        let mut unattributed = 0;
        let signatures = signatures(c);
        let mut ops = Vec::with_capacity(c.ops.len());
        for (k, rec) in c.ops.iter().enumerate() {
            let mut uses: BTreeMap<usize, Use> = BTreeMap::new();
            let mut order: Vec<usize> = Vec::new();
            let mut external = [0u64; 2];
            for (u, (uc, range)) in c.units.iter().zip(&rec.entries).enumerate() {
                let mut access = |w: [u32; 8], whole: bool| {
                    let write = match w[0] {
                        op::READ | op::READ_TRANSPOSED | op::READ_BROADCAST_COL => false,
                        op::WRITE => true,
                        _ => return,
                    };
                    let (channel, offset, len) = (w[1], w[3], u64::from(w[5]));
                    match index.find(channel, offset) {
                        Some((t, tile)) => {
                            if !uses.contains_key(&t) {
                                order.push(t);
                            }
                            let use_ = uses.entry(t).or_insert_with(|| Use {
                                op: k,
                                plain: true,
                                ..Use::default()
                            });
                            use_.plain &= whole;
                            let side = if write {
                                &mut use_.writes
                            } else {
                                &mut use_.reads
                            };
                            side.entry(u).or_default().insert(tile);
                            use_.bytes[usize::from(write)] += len;
                        }
                        None => external[usize::from(write)] += len,
                    }
                };
                let stream = &uc.stream[range.clone()];
                let mut i = 0;
                while i < stream.len() {
                    if stream[i][0] == op::PAIR_CALL {
                        if let Some(p) = uc.pair_calls.iter().position(|&at| at == range.start + i)
                        {
                            for section in uc.pending_sections.iter().skip(2 * p).take(2) {
                                unattributed += walk(section, &mut access);
                            }
                        }
                        i += 1;
                        continue;
                    }
                    let n = record::len(stream[i][0]).max(1);
                    unattributed += walk(&stream[i..(i + n).min(stream.len())], &mut access);
                    i += n;
                }
            }
            for (t, use_) in uses {
                tensors[t].uses.push(use_);
            }
            ops.push(Op {
                label: rec.label,
                what: rec.what,
                signature: signatures[k].clone(),
                l1_top: rec.l1_top,
                l1_need: rec.l1_need,
                tensors: order,
                external,
            });
        }
        Graph {
            ops,
            tensors,
            unattributed,
        }
    }

    /// Ops the plan has no `DATA` top for ([`crate::trace::OpRecord::l1_top`] zero): what they
    /// need of a unit's arena is unknown.
    pub fn unplanned(&self) -> impl Iterator<Item = usize> + '_ {
        self.ops
            .iter()
            .enumerate()
            .filter(|(_, o)| o.l1_top == 0)
            .map(|(k, _)| k)
    }
}

/// Why a tensor stays in GDDR.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Reason {
    /// Still live when the capture ended: its output or state.
    Output,
    /// No op reads it.
    NeverRead,
    /// Read before any op of the capture wrote it.
    ReadFirst,
    /// One op both reads and writes it (it would read and write one slot).
    ReadAndWritten,
    /// An op that uses it has no L1 path for it ([`Plan::make`]'s `capable`).
    NoPath,
    /// A tensor the same place of another layer uses stays in GDDR: the layer's addresses would
    /// differ.
    RepeatInGddr,
    /// It and the tensor at its place in another layer live at once: one address cannot serve both.
    OutlivesPeriod,
    /// Left out so the rest fit the region.
    Room,
}

/// Which class a region that is short leaves out first ([`Plan::make_rows`] tries both).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum LeaveOut {
    /// The largest `bytes x lifetime`: what holds the most region for longest.
    Longest,
    /// The fewest GDDR bytes spared a byte of its address.
    LeastSpared,
}

/// How a plan delivers a resident row that an op reads whole on many units ([`Tensor::replicated`]:
/// a product's packed `A`, RMSNorm's twin), `docs/plans/planned-l1-residency.md` 2f.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Rows {
    /// Not at all: such a row stays in GDDR.
    Gddr,
    /// Phase 2's: a copy on every unit that reads it, written by each producer to itself and by
    /// multicast to every rectangle of the others. ~250 us a multiply on the 1.7B (2e).
    Multicast,
    /// 2f (a): a copy on `h` of its readers ([`holders`]), each producer writing its tiles to each
    /// by unicast, each reader pulling what it reads from its holder (`op::READ_L1`). Only for a
    /// row every reader reads alike ([`Tensor::read_alike`]), so a holder's copy serves each.
    Held(usize),
}

/// What 2f measured best on the 1.7B's decode step (card 0, `1792537293`: 20.06 ms, against 20.18
/// with 4 and 21.03 without residency). Delivering one row alone, 4 were best
/// (`silicon_probe_row_delivery`, `1792533708`: 8-10 us a `[1, 2048]` row to 32-104 readers
/// against 9-17 through GDDR); in a step the other rows have fewer readers, and 2 holders cost
/// each producer fewer writes.
pub const HOLDERS: usize = 2;

/// A resident tile that at most this many units besides its writer read is written to each by
/// unicast (`op::WRITE_L1`), not by a multicast over their rectangles: a product's output tile is
/// read by one unit (the residual add, `silu_mul`) or two (RoPE's tile and its partner).
pub const UNICAST_READERS: usize = 4;

/// The holders of a [`Rows::Held`] tensor ([`holders`]).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Held {
    /// The units that hold it, ascending.
    pub holders: Vec<usize>,
    /// Each unit that reads it, and the holder it reads from.
    pub of: BTreeMap<usize, usize>,
}

/// A tensor's decision.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Place {
    /// At `offset` from the region's base, in `class` (tensors that share the address).
    Resident {
        class: usize,
        offset: u64,
        bytes: u64,
    },
    Gddr(Reason),
}

/// A static plan of a graph's resident tensors.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Per tensor of the graph.
    pub place: Vec<Place>,
    /// The region's bytes on each unit: the highest offset any resident tensor reaches.
    pub region: u64,
    /// What a unit's arena has for the region: `DATA`'s end less the most any planned op needs.
    pub budget: u64,
    /// The op that needs the most of the arena.
    pub neediest: Option<usize>,
    pub period: Option<Period>,
    /// Whether every repeat of the period sees the same addresses ([`Plan::make_within`]).
    pub pinned: bool,
    /// The resident tensors held by a few of their readers ([`Rows::Held`]); every other resident
    /// tensor is held by each unit that reads it.
    pub held: BTreeMap<usize, Held>,
    /// Tensors made inside the graph that nothing reads ([`Reason::NeverRead`]), every write of
    /// them a plain move: a planned capture writes them nowhere (2g; a row written both as tiles
    /// and as its packed twin, where the next op reads one).
    pub dropped: BTreeSet<usize>,
}

/// A plan that breaks a rule ([`Plan::check`]).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Hazard {
    /// Resident tensors `a` and `b` overlap in the region and live at once: op `op` would find
    /// one where it wrote the other.
    Overlap { a: usize, b: usize, op: usize },
    /// Op `op` and the op one period later see different addresses.
    NotPeriodic { op: usize },
    /// The region does not fit the budget.
    DoesNotFit { region: u64, budget: u64 },
    /// A resident tensor is placed past the region.
    OutOfRegion { tensor: usize },
    /// A held tensor's holders do not serve its readers: one reads from no holder, or from a unit
    /// that is not one, or the readers do not read it alike.
    NotHeld { tensor: usize },
    /// A tensor whose writes are dropped is read, or is not the graph's own, or is resident, or is
    /// written other than by plain moves.
    ReadDropped { tensor: usize },
}

impl std::fmt::Display for Hazard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hazard::Overlap { a, b, op } => write!(
                f,
                "tensors {a} and {b} share bytes of the region and are both live at op {op}"
            ),
            Hazard::NotPeriodic { op } => {
                write!(
                    f,
                    "op {op} and the op a period later see different addresses"
                )
            }
            Hazard::DoesNotFit { region, budget } => {
                write!(f, "a region of {region} bytes does not fit {budget}")
            }
            Hazard::OutOfRegion { tensor } => {
                write!(f, "tensor {tensor} is placed past the region")
            }
            Hazard::NotHeld { tensor } => {
                write!(f, "tensor {tensor}'s holders do not serve its readers")
            }
            Hazard::ReadDropped { tensor } => {
                write!(f, "tensor {tensor}'s writes are dropped, but it is needed")
            }
        }
    }
}

/// The op that needs most of a unit's arena, and what it needs (`DATA.base` if none planned).
fn neediest(g: &Graph) -> (Option<usize>, u64) {
    g.ops
        .iter()
        .enumerate()
        .filter(|(_, o)| o.l1_top != 0)
        .map(|(k, o)| (Some(k), o.l1_need))
        .max_by_key(|&(k, need)| (need, std::cmp::Reverse(k)))
        .unwrap_or((None, tt_isa::l1::DATA.base))
}

/// Union-find over tensors.
fn root(parent: &mut [usize], mut t: usize) -> usize {
    while parent[t] != t {
        parent[t] = parent[parent[t]];
        t = parent[t];
    }
    t
}

/// The most a region at the top of an arena ending at `data_end` can take with every op of `g`
/// laid out as it was: `data_end` less the highest of the ops' needs
/// ([`crate::trace::OpRecord::l1_need`], the lowest arena end each op's layouts stand at), in whole
/// [`ALIGN`]s (`docs/plans/planned-l1-residency.md` 2h). At the 1.7B, the `K` = 2048 products'
/// widest blocks set it: 110,464 bytes.
pub fn cap(g: &Graph, data_end: u64) -> u64 {
    let (_, need) = neediest(g);
    data_end.saturating_sub(need) / ALIGN * ALIGN
}

impl Plan {
    /// Plan `g` for a unit arena ending at `data_end`: the tensors whose every use `capable` gives an
    /// L1 path (the op and its use), placed so each layer of `g`'s period runs at the same
    /// addresses, in a region of [`cap`].
    pub fn make(g: &Graph, data_end: u64, capable: impl Fn(&Op, &Use) -> bool) -> Plan {
        Self::make_within(g, cap(g, data_end), capable)
    }

    /// [`Plan::make`] for a region of `budget` bytes a unit: what a capture that kept one has
    /// (`Session::set_residency`), every op planned below it.
    ///
    /// The period's pinning (one address for the tensor at one place of every repeat) is what
    /// lets a layer's repeats share a body (`docs/proposals/layer-sub-traces.md`), not what makes
    /// a plan correct. Where it costs tensors (a repeat's two tensors live at once, one stays in
    /// GDDR, or a class finds no room), the graph is planned without it too, and the plan that
    /// spares more GDDR bytes is kept (the periodic one on a tie). Where a class finds no room,
    /// each is planned again leaving out first what spares least a byte, not what holds the most
    /// region longest, and the same rule keeps one.
    pub fn make_within(g: &Graph, budget: u64, capable: impl Fn(&Op, &Use) -> bool) -> Plan {
        Self::make_rows(g, budget, Rows::Multicast, capable)
    }

    /// [`Plan::make_within`], with the rows many units read whole delivered as `rows` says.
    pub fn make_rows(
        g: &Graph,
        budget: u64,
        rows: Rows,
        capable: impl Fn(&Op, &Use) -> bool,
    ) -> Plan {
        let period = period(&g.ops.iter().map(|o| &o.signature).collect::<Vec<_>>());
        let make = |period, leave| Self::make_inner(g, budget, rows, &capable, period, leave);
        let pinned = make(period, LeaveOut::Longest);
        let room = |p: &Plan| p.place.contains(&Place::Gddr(Reason::Room));
        // Pinning costs tensors where a repeat's two live at once, one stays in GDDR, or a class
        // takes its room in every repeat while the tensors themselves would share it (a layer's
        // products' outputs beside its residual stream, Phase 3).
        let costs = period.is_some()
            && pinned.place.iter().any(|p| {
                matches!(
                    p,
                    Place::Gddr(Reason::OutlivesPeriod | Reason::RepeatInGddr | Reason::Room)
                )
            });
        // The candidates, the first kept on a tie: pinned, then not, each leaving out first what
        // holds the most region longest, then what spares the least a byte (Phase 4: a mask or a
        // table every layer reads lives the whole step and spares the most).
        let mut candidates = vec![(pinned, true)];
        if costs {
            candidates.push((make(None, LeaveOut::Longest), false));
        }
        if room(&candidates[0].0) {
            candidates.push((make(period, LeaveOut::LeastSpared), true));
        }
        if period.is_some() && candidates.iter().any(|(p, pinned)| !pinned && room(p)) {
            candidates.push((make(None, LeaveOut::LeastSpared), false));
        }
        let spared = |p: &Plan| p.gddr_avoided(g).iter().sum::<u64>();
        let mut best = 0;
        for (i, (p, _)) in candidates.iter().enumerate() {
            if spared(p) > spared(&candidates[best].0) {
                best = i;
            }
        }
        let (plan, is_pinned) = candidates.swap_remove(best);
        if is_pinned {
            plan
        } else {
            Plan { period, ..plan }.unpinned()
        }
    }

    /// A plan made without its period's pinning keeps the period for its report, not its check.
    fn unpinned(self) -> Plan {
        Plan {
            pinned: false,
            ..self
        }
    }

    fn make_inner(
        g: &Graph,
        budget: u64,
        rows: Rows,
        capable: &impl Fn(&Op, &Use) -> bool,
        period: Option<Period>,
        leave: LeaveOut,
    ) -> Plan {
        let (neediest, _) = neediest(g);
        // Each tensor on its own first.
        let mut reason: Vec<Option<Reason>> = g
            .tensors
            .iter()
            .map(|t| {
                if !t.internal {
                    Some(Reason::Output)
                } else if !t.uses.iter().any(Use::reads) {
                    Some(Reason::NeverRead)
                } else if !t.uses[0].writes() {
                    Some(Reason::ReadFirst)
                } else if t.uses.iter().any(|u| u.reads() && u.writes()) {
                    Some(Reason::ReadAndWritten)
                } else if t.uses.iter().any(|u| !capable(&g.ops[u.op], u))
                    || t.replicated()
                        && match rows {
                            Rows::Gddr => true,
                            Rows::Multicast => false,
                            Rows::Held(_) => !t.read_alike(),
                        }
                {
                    Some(Reason::NoPath)
                } else {
                    None
                }
            })
            .collect();
        let dropped: BTreeSet<usize> = (0..g.tensors.len())
            .filter(|&t| {
                reason[t] == Some(Reason::NeverRead) && g.tensors[t].uses.iter().all(|u| u.plain)
            })
            .collect();
        // Tensors at one place of two layers share an address: the `j`th tensor op `k` touches,
        // and the `j`th the op a period later does.
        let n = g.tensors.len();
        let mut parent: Vec<usize> = (0..n).collect();
        if let Some(per) = period {
            for k in per.span() {
                let next = k + per.p;
                if !per.span().contains(&next) {
                    continue;
                }
                for (&a, &b) in g.ops[k].tensors.iter().zip(&g.ops[next].tensors) {
                    let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                    if ra != rb {
                        parent[ra.max(rb)] = ra.min(rb);
                    }
                }
            }
        }
        let mut classes: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for t in 0..n {
            let r = root(&mut parent, t);
            classes.entry(r).or_default().push(t);
        }
        let mut classes: Vec<Vec<usize>> = classes.into_values().collect();
        for members in &classes {
            if members.len() > 1 && members.iter().any(|&t| reason[t].is_some()) {
                for &t in members {
                    reason[t].get_or_insert(Reason::RepeatInGddr);
                }
            }
        }
        // A class's members must live one after another.
        for members in &classes {
            let mut spans: Vec<(usize, usize)> = members
                .iter()
                .filter(|&&t| reason[t].is_none())
                .filter_map(|&t| g.tensors[t].lifetime())
                .collect();
            spans.sort_unstable();
            if spans.windows(2).any(|w| w[1].0 <= w[0].1) {
                for &t in members {
                    reason[t].get_or_insert(Reason::OutlivesPeriod);
                }
            }
        }
        classes.retain(|m| reason[m[0]].is_none());
        // Place, and drop the costliest class until the rest fit.
        loop {
            let (offsets, region) = assign(g, &classes);
            if region <= budget || classes.is_empty() {
                let mut place: Vec<Place> = reason
                    .iter()
                    .map(|r| Place::Gddr(r.unwrap_or(Reason::Room)))
                    .collect();
                for (c, members) in classes.iter().enumerate() {
                    let bytes = class_bytes(g, members);
                    for &t in members {
                        place[t] = Place::Resident {
                            class: c,
                            offset: offsets[c],
                            bytes,
                        };
                    }
                }
                let held = match rows {
                    Rows::Held(h) => place
                        .iter()
                        .enumerate()
                        .filter(|&(t, p)| {
                            matches!(p, Place::Resident { .. }) && g.tensors[t].replicated()
                        })
                        .map(|(t, _)| {
                            let readers: Vec<usize> = g.tensors[t].read_by().into_keys().collect();
                            let (holders, of) = holders(&readers, h);
                            (t, Held { holders, of })
                        })
                        .collect(),
                    _ => BTreeMap::new(),
                };
                return Plan {
                    place,
                    region,
                    budget,
                    neediest,
                    period,
                    pinned: true,
                    held,
                    dropped,
                };
            }
            let worst = match leave {
                LeaveOut::Longest => {
                    let cost = |m: &Vec<usize>| -> u64 {
                        let life: usize = m
                            .iter()
                            .filter_map(|&t| g.tensors[t].lifetime())
                            .map(|(a, b)| b - a + 1)
                            .sum();
                        class_bytes(g, m) * life as u64
                    };
                    let (worst, _) = classes
                        .iter()
                        .enumerate()
                        .max_by_key(|&(i, m)| (cost(m), i))
                        .expect("not empty");
                    worst
                }
                LeaveOut::LeastSpared => {
                    // `spared / bytes`, compared across: `spared_a * bytes_b` against the other.
                    let of = |m: &Vec<usize>| -> (u128, u128) {
                        let spared: u64 = m
                            .iter()
                            .map(|&t| g.tensors[t].gddr_bytes().iter().sum::<u64>())
                            .sum();
                        (spared as u128, class_bytes(g, m).max(1) as u128)
                    };
                    let (worst, _) = classes
                        .iter()
                        .enumerate()
                        .map(|(i, m)| (i, of(m)))
                        .min_by(|(i, (sa, ba)), (j, (sb, bb))| {
                            (sa * bb).cmp(&(sb * ba)).then(j.cmp(i))
                        })
                        .expect("not empty");
                    worst
                }
            };
            for &t in &classes[worst] {
                reason[t] = Some(Reason::Room);
            }
            classes.remove(worst);
        }
    }

    /// Check the plan against `g`'s rules, independently of how it was made.
    pub fn check(&self, g: &Graph) -> Result<(), Hazard> {
        if self.region > self.budget {
            return Err(Hazard::DoesNotFit {
                region: self.region,
                budget: self.budget,
            });
        }
        let resident: Vec<(usize, u64, u64, (usize, usize))> = self
            .place
            .iter()
            .enumerate()
            .filter_map(|(t, p)| match *p {
                Place::Resident { offset, bytes, .. } => {
                    Some((t, offset, bytes, g.tensors[t].lifetime()?))
                }
                Place::Gddr(_) => None,
            })
            .collect();
        for &(t, offset, bytes, _) in &resident {
            if offset + bytes > self.region || offset % ALIGN != 0 {
                return Err(Hazard::OutOfRegion { tensor: t });
            }
        }
        for (i, &(a, oa, ba, (sa, ea))) in resident.iter().enumerate() {
            for &(b, ob, bb, (sb, eb)) in &resident[i + 1..] {
                let bytes_meet = oa < ob + bb && ob < oa + ba;
                let lives_meet = sa <= eb && sb <= ea;
                if bytes_meet && lives_meet {
                    return Err(Hazard::Overlap {
                        a,
                        b,
                        op: sa.max(sb),
                    });
                }
            }
        }
        for &t in &self.dropped {
            let tensor = &g.tensors[t];
            let needed = !tensor.internal
                || tensor.uses.iter().any(|u| u.reads() || !u.plain)
                || matches!(self.place.get(t), Some(Place::Resident { .. }));
            if needed {
                return Err(Hazard::ReadDropped { tensor: t });
            }
        }
        for (&t, h) in &self.held {
            let tensor = &g.tensors[t];
            let readers: BTreeSet<usize> = tensor.read_by().into_keys().collect();
            let serves = matches!(self.place.get(t), Some(Place::Resident { .. }))
                && tensor.read_alike()
                && !h.holders.is_empty()
                && h.holders.iter().all(|v| readers.contains(v))
                && h.of.keys().copied().eq(readers.iter().copied())
                && h.of.values().all(|v| h.holders.contains(v));
            if !serves {
                return Err(Hazard::NotHeld { tensor: t });
            }
        }
        if let Some(per) = self.period.filter(|_| self.pinned) {
            let seen = |k: usize| -> Vec<Option<u64>> {
                g.ops[k]
                    .tensors
                    .iter()
                    .map(|&t| match self.place[t] {
                        Place::Resident { offset, .. } => Some(offset),
                        Place::Gddr(_) => None,
                    })
                    .collect()
            };
            for k in per.span() {
                if per.span().contains(&(k + per.p)) && seen(k) != seen(k + per.p) {
                    return Err(Hazard::NotPeriodic { op: k });
                }
            }
        }
        Ok(())
    }

    /// The resident tensors.
    pub fn resident(&self) -> impl Iterator<Item = usize> + '_ {
        self.place
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p, Place::Resident { .. }))
            .map(|(t, _)| t)
    }

    /// A stable hash of what the plan decided over `g`'s ops: equal plans of equal graphs are
    /// equal, whatever run made them.
    pub fn signature(&self, g: &Graph) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for o in &g.ops {
            o.signature.hash(&mut h);
            for &t in &o.tensors {
                match self.place[t] {
                    Place::Resident { offset, bytes, .. } => (1u8, offset, bytes).hash(&mut h),
                    Place::Gddr(r) => (0u8, r as u64, 0u64).hash(&mut h),
                }
                if let Some(held) = self.held.get(&t) {
                    held.holders.hash(&mut h);
                }
                self.dropped.contains(&t).hash(&mut h);
            }
        }
        self.region.hash(&mut h);
        h.finish()
    }

    /// The region the repeats alone need: the highest byte of the addresses some tensor made in a
    /// repeat holds. What the head and tail add beyond it (a step's logits, read whole by one unit)
    /// could leave the plan without changing a layer.
    pub fn region_within_repeats(&self, g: &Graph) -> u64 {
        let Some(per) = self.period else {
            return self.region;
        };
        self.place
            .iter()
            .enumerate()
            .filter_map(|(t, p)| match *p {
                Place::Resident { offset, bytes, .. }
                    if per.span().contains(&g.tensors[t].lifetime()?.0) =>
                {
                    Some(offset + bytes)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    /// What the plan's resident tensors would no longer move through GDDR: `[read, written]`.
    pub fn gddr_avoided(&self, g: &Graph) -> [u64; 2] {
        self.resident().fold([0, 0], |[r, w], t| {
            let [a, b] = g.tensors[t].gddr_bytes();
            [r + a, w + b]
        })
    }

    /// The plan, as the lines a probe prints (`RESIDENCY ...`).
    pub fn report(&self, g: &Graph) -> String {
        let mut out = String::new();
        let local: [u64; 2] = g.tensors.iter().fold([0, 0], |[r, w], t| {
            let [a, b] = t.gddr_bytes();
            [r + a, w + b]
        });
        let external = g
            .ops
            .iter()
            .fold([0, 0], |[r, w], o| [r + o.external[0], w + o.external[1]]);
        let unplanned: BTreeSet<(&str, &str)> = g
            .unplanned()
            .map(|k| (g.ops[k].label, g.ops[k].what))
            .collect();
        let _ = writeln!(
            out,
            "RESIDENCY graph: {} ops, {} capture tensors ({} internal); GDDR {} B read / {} B written of \
             them, {} B / {} B external; {} moves unattributed; {} ops unplanned {:?}",
            g.ops.len(),
            g.tensors.len(),
            g.tensors.iter().filter(|t| t.internal).count(),
            local[0],
            local[1],
            external[0],
            external[1],
            g.unattributed,
            g.unplanned().count(),
            unplanned,
        );
        match self.period {
            Some(p) => {
                let _ = writeln!(
                    out,
                    "RESIDENCY period: {} ops x {} from op {} ({} of {} ops repeat at it); addresses {}",
                    p.p,
                    p.repeats,
                    p.start,
                    p.matches,
                    g.ops.len() - p.p,
                    if self.pinned {
                        "the same in every repeat"
                    } else {
                        "not pinned: pinning spared less"
                    }
                );
            }
            None => {
                let _ = writeln!(out, "RESIDENCY period: none");
            }
        }
        let avoided = self.gddr_avoided(g);
        let classes: BTreeSet<usize> = self
            .place
            .iter()
            .filter_map(|p| match p {
                Place::Resident { class, .. } => Some(*class),
                Place::Gddr(_) => None,
            })
            .collect();
        let neediest = self
            .neediest
            .map(|k| format!("{}/{}", g.ops[k].label, g.ops[k].what))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "RESIDENCY plan: {} tensors resident in {} addresses; region {} B a unit ({} B within the \
             repeats) of a budget of {} B (the neediest op {neediest}); {} B a unit left for prefetched \
             B; GDDR avoided {} B read / {} B written; check: {}; signature {:016x}",
            self.resident().count(),
            classes.len(),
            self.region,
            self.region_within_repeats(g),
            self.budget,
            self.budget.saturating_sub(self.region),
            avoided[0],
            avoided[1],
            match self.check(g) {
                Ok(()) => "ok".to_string(),
                Err(e) => e.to_string(),
            },
            self.signature(g),
        );
        // Each address, by its first member.
        let mut seen = BTreeSet::new();
        for (t, p) in self.place.iter().enumerate() {
            let Place::Resident {
                class,
                offset,
                bytes,
            } = *p
            else {
                continue;
            };
            if !seen.insert(class) {
                continue;
            }
            let members = self
                .place
                .iter()
                .filter(|q| matches!(q, Place::Resident { class: c, .. } if *c == class))
                .count();
            let _ = writeln!(
                out,
                "RESIDENCY   @{offset:>7} {bytes:>7} B x{members:<3} {}",
                describe(g, t)
            );
        }
        let mut by_reason: BTreeMap<Reason, (usize, u64)> = BTreeMap::new();
        for (t, p) in self.place.iter().enumerate() {
            if let Place::Gddr(r) = p {
                let e = by_reason.entry(*r).or_default();
                e.0 += 1;
                let [a, b] = g.tensors[t].gddr_bytes();
                e.1 += a + b;
            }
        }
        for (r, (count, bytes)) in by_reason {
            let _ = writeln!(
                out,
                "RESIDENCY   in GDDR, {r:?}: {count} tensors, {bytes} B moved"
            );
        }
        // The internal tensors left out, by what made them; those left out for room with what
        // they would take a unit.
        let mut left: BTreeMap<String, usize> = BTreeMap::new();
        for (t, p) in self.place.iter().enumerate() {
            if let Place::Gddr(r) = p {
                let what = match r {
                    Reason::Output => continue,
                    Reason::Room => format!(
                        "Room ({} B a unit): {}",
                        g.tensors[t].bytes_a_unit(),
                        describe(g, t)
                    ),
                    r => format!("{r:?}: {}", describe(g, t)),
                };
                *left.entry(what).or_default() += 1;
            }
        }
        for (what, count) in left.iter().take(32) {
            let _ = writeln!(out, "RESIDENCY   left x{count:<3} {what}");
        }
        out
    }
}

/// A plan applied to a capture of the same graph (`docs/plans/planned-l1-residency.md` Phase 2):
/// what the planned capture's ops' lists become, op by op, so its resident tensors move between
/// units' L1 and never through GDDR.
///
/// The planned capture allocates the same tensors in the same order as the plan pass, so a
/// tensor is known by its place in the capture's allocations. Each resident tensor `t` holds, on
/// every unit that reads it, the tiles that unit reads (over all its readers), in tile order,
/// from `region_base + offset(t)`, a slot each. So:
/// - a whole move that **reads** a resident tensor on unit `u` (a plain `READ`, or one a
///   `READ_RUN` without flags expands to) becomes a read of `u`'s own region over the NoC
///   (`op::READ_L1` of its own tile);
/// - a whole move that **writes** one (a plain `WRITE`, or one a `WRITE_RUN` expands to) becomes,
///   for each address the tile has on the units that read it, an `op::WRITE_L1` to the writing
///   unit itself and one `op::WRITE_L1_MCAST` a rectangle of the others (a single tile, a
///   `WRITE_L1`).
///
/// A dropped tensor's writes ([`Plan::dropped`]) go nowhere: removed from the list.
///
/// A held tensor ([`Plan::held`], [`Rows::Held`]) is held by its holders alone, each holding
/// every tile its readers read: a read becomes an `op::READ_L1` of the reader's holder's region,
/// and a write one `op::WRITE_L1` to each holder, no multicast. Reads of consecutive slots from
/// one tile into consecutive slots are one `READ_L1`, up to a request.
///
/// A record that touches a resident tensor is rewritten whole, so all its moves must; anything
/// else is a planned capture that is not its plan's, refused by name.
#[derive(Clone, Debug)]
pub struct Applied {
    pub graph: Graph,
    pub plan: Plan,
    /// Each op's shape in the plan pass ([`crate::trace::OpRecord::shape`]): the planned capture's
    /// must equal it, op for op.
    pub shapes: Vec<u64>,
    /// Where the region begins on every unit.
    pub region_base: u64,
    /// The session's units' coordinates, by unit.
    pub coords: Vec<(u8, u8)>,
    /// Per tensor, per unit that reads it: the tiles it holds, in address order.
    holds: Vec<BTreeMap<usize, Vec<u32>>>,
    /// The (op, unit) lists that move a resident or dropped tensor in the plan pass, made when
    /// first asked ([`Applied::touches`]; forgotten when the plan changes).
    touched: std::sync::OnceLock<BTreeSet<(usize, usize)>>,
}

impl Applied {
    /// `plan` of `graph`, made with its region below `region_base`'s end, for a session of units at
    /// `coords`; `None` if nothing is resident.
    pub fn new(
        graph: Graph,
        plan: Plan,
        shapes: Vec<u64>,
        region_base: u64,
        coords: Vec<(u8, u8)>,
    ) -> Option<Self> {
        if plan.resident().next().is_none() && plan.dropped.is_empty() {
            return None;
        }
        let holds = graph
            .tensors
            .iter()
            .enumerate()
            .map(|(t, tensor)| {
                let per_unit = tensor.read_by();
                match plan.held.get(&t) {
                    Some(h) => {
                        let all: BTreeSet<u32> = per_unit.into_values().flatten().collect();
                        h.holders
                            .iter()
                            .map(|&v| (v, all.iter().copied().collect()))
                            .collect()
                    }
                    None => per_unit
                        .into_iter()
                        .map(|(unit, tiles)| (unit, tiles.into_iter().collect()))
                        .collect(),
                }
            })
            .collect();
        Some(Applied {
            graph,
            plan,
            shapes,
            region_base,
            coords,
            holds,
            touched: Default::default(),
        })
    }

    /// Whether op `k`'s lists on unit `u` moved a resident or dropped tensor in the plan pass: the
    /// only lists [`Applied::rewrite`] changes. The others are kept as they are, unread: a
    /// product's gathers expand to every weight tile it reads. A planned capture whose other lists
    /// move one is refused all the same, by [`Applied::verify`].
    pub(crate) fn touches(&self, k: usize, u: usize) -> bool {
        self.touched
            .get_or_init(|| {
                self.plan
                    .resident()
                    .chain(self.plan.dropped.iter().copied())
                    .flat_map(|t| &self.graph.tensors[t].uses)
                    .flat_map(|w| w.reads.keys().chain(w.writes.keys()).map(|&u| (w.op, u)))
                    .collect()
            })
            .contains(&(k, u))
    }

    /// Forget what [`Applied::touches`] found: the plan changed.
    pub(crate) fn forget_touches(&mut self) {
        self.touched = Default::default();
    }

    /// Whether op `k`, about to be queued as `queues` (each unit's steps, before
    /// [`Applied::rewrite`]) with `allocs` the capture's tensors so far, is the plan pass's op `k`:
    /// the same tensors allocated so far, and every move of every list on the same tensor's same
    /// tiles, read or written alike, as [`Graph::from_capture`] attributes them. Checked before
    /// the op runs, so a capture of another graph is stopped before any of its ops reads what the
    /// plan put in L1 for this one (an op given other data can stop the queue: a division by
    /// zero, `dm::error::DOMAIN`).
    pub(crate) fn check_op(
        &self,
        index: &Index,
        allocs: &[Placement],
        k: usize,
        queues: &[Vec<crate::tensor::Step>],
    ) -> Result<(), String> {
        use crate::tensor::Step;
        if allocs.len() > self.graph.tensors.len() {
            return Err(format!(
                "op {k}: {} tensors allocated against the plan's {}",
                allocs.len(),
                self.graph.tensors.len()
            ));
        }
        if let Some(t) = (0..allocs.len()).find(|&t| {
            let want = &self.graph.tensors[t];
            allocs[t].tiles() != want.tiles || allocs[t].slot_bytes() != want.slot_bytes
        }) {
            return Err(format!("op {k}: tensor {t} is not the plan's"));
        }
        let mut uses: BTreeMap<usize, Use> = BTreeMap::new();
        let mut unread = 0;
        for (u, steps) in queues.iter().enumerate() {
            let mut access = |w: [u32; 8], whole: bool| {
                let write = match w[0] {
                    op::READ | op::READ_TRANSPOSED | op::READ_BROADCAST_COL => false,
                    op::WRITE => true,
                    _ => return,
                };
                if let Some((t, tile)) = index.find(w[1], w[3]) {
                    let use_ = uses.entry(t).or_insert_with(|| Use {
                        op: k,
                        plain: true,
                        ..Use::default()
                    });
                    use_.plain &= whole;
                    let side = if write {
                        &mut use_.writes
                    } else {
                        &mut use_.reads
                    };
                    side.entry(u).or_default().insert(tile);
                    use_.bytes[usize::from(write)] += u64::from(w[5]);
                }
            };
            for step in steps {
                match step {
                    Step::List { entries, .. } => unread += walk(entries, &mut access),
                    Step::Transfer { batches, .. } => {
                        for b in batches {
                            unread += walk(&b.read, &mut access);
                            unread += walk(&b.write, &mut access);
                        }
                    }
                    Step::Kernel { .. } => {}
                }
            }
        }
        let want = self.graph.ops.get(k).map_or(&[][..], |o| &o.tensors[..]);
        let same = unread == 0
            && uses.len() == want.len()
            && want.iter().all(|&t| {
                let planned = self.graph.tensors[t].uses.iter().find(|w| w.op == k);
                planned.is_some_and(|w| uses.get(&t).is_some_and(|got| got == w))
            });
        if same {
            Ok(())
        } else {
            Err(format!("op {k} moves other tensors than the plan pass's"))
        }
    }

    /// Whether capture `c`, ending, with `allocs` its tensors, is the plan applied: the plan pass's
    /// ops laid out alike, the same tensors allocated in order, nothing stopped the rewrite, and no
    /// resident tensor moved through GDDR (its graph, read off the streams as the plan pass's was).
    pub(crate) fn verify(&self, c: &Capture, allocs: &[Placement]) -> Result<(), String> {
        if let Some(why) = &c.plan_mismatch {
            return Err(why.clone());
        }
        if c.ops.len() != self.shapes.len() {
            return Err(format!(
                "{} ops against the plan's {}",
                c.ops.len(),
                self.shapes.len()
            ));
        }
        if let Some(k) = (0..c.ops.len()).find(|&k| c.ops[k].shape != self.shapes[k]) {
            return Err(format!("op {k} is laid out unlike the plan pass's"));
        }
        if allocs.len() != self.graph.tensors.len() {
            return Err(format!(
                "{} tensors allocated against the plan's {}",
                allocs.len(),
                self.graph.tensors.len()
            ));
        }
        if let Some(t) = (0..allocs.len()).find(|&t| {
            let want = &self.graph.tensors[t];
            allocs[t].tiles() != want.tiles || allocs[t].slot_bytes() != want.slot_bytes
        }) {
            return Err(format!("tensor {t} is not the plan's"));
        }
        let g = Graph::from_capture(c, allocs);
        if let Some(t) = self
            .plan
            .resident()
            .chain(self.plan.dropped.iter().copied())
            .find(|&t| !g.tensors[t].uses.is_empty())
        {
            return Err(format!(
                "resident tensor {t} still moves through GDDR, at op {}",
                g.tensors[t].uses[0].op
            ));
        }
        Ok(())
    }

    /// Whether tensor `t` is resident.
    pub fn resident(&self, t: usize) -> bool {
        matches!(self.plan.place.get(t), Some(Place::Resident { .. }))
    }

    /// Whether tensor `t`'s writes are dropped ([`Plan::dropped`]).
    pub fn dropped(&self, t: usize) -> bool {
        self.plan.dropped.contains(&t)
    }

    /// Where tile `tile` of resident tensor `t` is on unit `u`, if `u` holds it.
    fn address(&self, t: usize, u: usize, tile: u32) -> Option<u64> {
        let Place::Resident { offset, .. } = self.plan.place[t] else {
            return None;
        };
        let pos = self.holds[t].get(&u)?.iter().position(|&x| x == tile)?;
        Some(self.region_base + offset + pos as u64 * self.graph.tensors[t].slot_bytes)
    }

    /// The units that hold tile `tile` of `t`, by the address they hold it at.
    fn holders(&self, t: usize, tile: u32) -> BTreeMap<u64, Vec<usize>> {
        let mut by: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for &u in self.holds[t].keys() {
            if let Some(at) = self.address(t, u, tile) {
                by.entry(at).or_default().push(u);
            }
        }
        by
    }

    /// Unit `u`'s list `entries`, of op `k`, rewritten: see [`Applied`]. `index` names the
    /// capture's tensors so far.
    pub(crate) fn rewrite(
        &self,
        index: &Index,
        k: usize,
        u: usize,
        entries: &[[u32; 8]],
    ) -> Result<Vec<[u32; 8]>, String> {
        let mut out = Vec::with_capacity(entries.len());
        let mut i = 0;
        while i < entries.len() {
            let head = entries[i];
            let n = record::len(head[0]).max(1);
            let chunk = &entries[i..(i + n).min(entries.len())];
            i += n;
            if !(record::is_record(head[0]) || matches!(head[0], op::READ | op::WRITE)) {
                out.extend_from_slice(chunk);
                continue;
            }
            let mut moves = Vec::new();
            let mut other = false;
            let ok = if record::is_record(head[0]) {
                record::expand(chunk, |w| {
                    match w[0] {
                        op::READ | op::WRITE | op::READ_TRANSPOSED | op::READ_BROADCAST_COL => {
                            moves.push(w)
                        }
                        _ => other = true,
                    }
                    Ok(())
                })
                .is_ok()
            } else {
                moves.push(head);
                true
            };
            let located: Vec<Option<(usize, u32, u64)>> = moves
                .iter()
                .map(|w| {
                    index
                        .locate(w[1], w[3])
                        .filter(|&(t, ..)| self.resident(t) || self.dropped(t))
                })
                .collect();
            if located.iter().all(Option::is_none) {
                out.extend_from_slice(chunk);
                continue;
            }
            let whole = record::is_record(head[0]) && plain(Some(head), head) || plain(None, head);
            if !ok || other || !whole || located.iter().any(Option::is_none) {
                return Err(format!(
                    "op {k} on unit {u}: a move of a resident tensor in an entry {:#x} that is not \
                     wholly a plain move of resident tensors",
                    head[0]
                ));
            }
            for (w, at) in moves.iter().zip(located) {
                let (t, tile, byte) = at.expect("checked");
                let len = w[5];
                if self.dropped(t) {
                    if w[0] != op::WRITE {
                        return Err(format!(
                            "op {k} on unit {u} reads tensor {t}, whose writes are dropped"
                        ));
                    }
                    continue;
                }
                if w[0] == op::READ {
                    // A held tensor is read from the reader's holder; any other from its own copy.
                    let src = match self.plan.held.get(&t) {
                        Some(h) => *h.of.get(&u).ok_or_else(|| {
                            format!("op {k}: unit {u} reads held tensor {t} from no holder")
                        })?,
                        None => u,
                    };
                    let from = self
                        .address(t, src, tile)
                        .ok_or_else(|| format!("op {k}: unit {u} reads tile {tile} of tensor {t}, which the plan has unit {src} not hold"))?
                        + byte;
                    let (x, y) = self.coords[src];
                    let mut done = 0;
                    while done < len {
                        let n = (len - done).min(tt_isa::noc::niu::MAX_REQUEST_BYTES);
                        push_read(
                            &mut out,
                            [x as u32, y as u32, from as u32 + done, w[4] + done, n],
                        );
                        done += n;
                    }
                } else if let Some(h) = self.plan.held.get(&t) {
                    let to = self.address(t, h.holders[0], tile).ok_or_else(|| {
                        format!("op {k}: tile {tile} of held tensor {t} has no holder")
                    })? + byte;
                    for &v in &h.holders {
                        let (x, y) = self.coords[v];
                        out.push([op::WRITE_L1, x as u32, y as u32, to as u32, w[4], len, 0, 0]);
                    }
                } else {
                    for (at, units) in self.holders(t, tile) {
                        let to = (at + byte) as u32;
                        let me = units.contains(&u).then_some(u);
                        if me.is_some() {
                            let (x, y) = self.coords[u];
                            out.push([op::WRITE_L1, x as u32, y as u32, to, w[4], len, 0, 0]);
                        }
                        let others: Vec<(u8, u8)> = units
                            .iter()
                            .filter(|&&v| v != u)
                            .map(|&v| self.coords[v])
                            .collect();
                        // A tile few units read goes to each by unicast: many producers'
                        // path-reserved multicasts serialize (2e), and a unicast to a neighbour
                        // costs no more than one to a rectangle of two.
                        let rects = if others.len() <= UNICAST_READERS {
                            others.iter().map(|&t| (t, t)).collect()
                        } else {
                            rectangles(&others)
                        };
                        for (s, e) in rects {
                            out.push(if s == e {
                                [op::WRITE_L1, s.0 as u32, s.1 as u32, to, w[4], len, 0, 0]
                            } else {
                                [
                                    op::WRITE_L1_MCAST,
                                    tt_isa::dm::mcast_corners(s, e),
                                    0,
                                    to,
                                    w[4],
                                    len,
                                    0,
                                    0,
                                ]
                            });
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Push `READ_L1` `[x, y, from, to, len]` onto `out`, or lengthen the `READ_L1` before it if this
/// one continues it: the same tile, the next bytes on both sides, within one request.
fn push_read(out: &mut Vec<[u32; 8]>, [x, y, from, to, len]: [u32; 5]) {
    if let Some(last) = out.last_mut() {
        if last[0] == op::READ_L1
            && (last[1], last[2]) == (x, y)
            && last[3] + last[5] == from
            && last[4] + last[5] == to
            && last[5] + len <= tt_isa::noc::niu::MAX_REQUEST_BYTES
        {
            last[5] += len;
            return;
        }
    }
    out.push([op::READ_L1, x, y, from, to, len, 0, 0]);
}

/// Rectangles covering exactly `tiles`: each row's runs of adjacent columns, runs of the same
/// columns on adjacent rows merged. Deterministic.
pub fn rectangles(tiles: &[(u8, u8)]) -> Vec<((u8, u8), (u8, u8))> {
    let mut rows: BTreeMap<u8, BTreeSet<u8>> = BTreeMap::new();
    for &(x, y) in tiles {
        rows.entry(y).or_default().insert(x);
    }
    let mut runs: Vec<(u8, u8, u8)> = Vec::new();
    for (&y, xs) in &rows {
        let mut it = xs.iter().copied();
        let Some(mut a) = it.next() else { continue };
        let mut b = a;
        for x in it {
            if x == b + 1 {
                b = x;
            } else {
                runs.push((a, b, y));
                a = x;
                b = x;
            }
        }
        runs.push((a, b, y));
    }
    runs.sort_by_key(|&(a, b, y)| (a, b, y));
    let mut out: Vec<((u8, u8), (u8, u8))> = Vec::new();
    for (a, b, y) in runs {
        match out.last_mut() {
            Some((s, e)) if s.0 == a && e.0 == b && e.1 + 1 == y => e.1 = y,
            _ => out.push(((a, y), (b, y))),
        }
    }
    out
}

/// How a row that many units read whole reaches them (`docs/plans/planned-l1-residency.md` 2f):
/// a product's packed `A`, RMSNorm's twin. Its producers each hold a few of its face rows; every
/// reader needs all of them. [`crate::session::Session::probe_row_delivery`] measures each.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowScheme {
    /// Through GDDR, as without residency: each producer writes its face rows into the twin, each
    /// reader reads the twin whole.
    Gddr,
    /// Phase 2's: each producer writes each face row into its own region if it reads the row, and
    /// multicasts it to every rectangle of the other readers (2e: ~250 us a multiply, the
    /// rectangles' multicasts serialized).
    Multicast,
    /// 2f (c): [`RowScheme::Multicast`], its multicasts marked as runs
    /// ([`tt_isa::dm::MCAST_RUN_ACKS`]): a producer's acknowledgements counted once, after its last.
    MulticastRunAcks,
    /// 2f (a): each producer writes each face row to `h` holders among the readers ([`holders`]);
    /// each reader pulls the row whole from its holder, one `READ_L1` a request.
    Held(usize),
    /// 2f (b): each producer writes each face row to one gatherer (the middle reader); an op of its
    /// own then multicasts the row whole from there, a request at a time, to every rectangle of the
    /// other readers, which read their own copy.
    Relay,
}

/// The holders of a row read whole by `readers` (ascending unit indices) when `h` of them hold it
/// ([`RowScheme::Held`]): the readers split into `h` contiguous runs ([`crate::tensor::runs`]),
/// each held by its middle reader. Returns the holders, ascending, and each reader's holder. Pure
/// in `readers`, so every repeat of a layer, whose readers are the same units, picks the same.
pub fn holders(readers: &[usize], h: usize) -> (Vec<usize>, BTreeMap<usize, usize>) {
    let h = h.clamp(1, readers.len().max(1));
    let mut held = Vec::with_capacity(h);
    let mut of = BTreeMap::new();
    for run in crate::tensor::runs(readers.len(), h, usize::MAX) {
        let holder = readers[run.start + run.len() / 2];
        held.push(holder);
        for &r in &readers[run] {
            of.insert(r, holder);
        }
    }
    (held, of)
}

/// A tensor's ops: who makes it and who reads it, how.
fn describe(g: &Graph, t: usize) -> String {
    let tensor = &g.tensors[t];
    let name = |k: usize| format!("{}/{}", g.ops[k].label, g.ops[k].what);
    let mut s = String::new();
    for u in &tensor.uses {
        let role = match (u.reads(), u.writes()) {
            (true, true) => "rw",
            (true, false) => "r",
            (false, true) => "w",
            (false, false) => "-",
        };
        let units = u.reads.len().max(u.writes.len());
        let layout = if u.reads() {
            format!(" {:?}", u.layout())
        } else {
            String::new()
        };
        let _ = write!(s, "{role} {}[{units}u{layout}] ", name(u.op));
    }
    let (a, b) = tensor.lifetime().unwrap_or((0, 0));
    let _ = write!(s, "lives {} ops", b - a + 1);
    s
}

/// Bytes a class's address holds: its largest member.
fn class_bytes(g: &Graph, members: &[usize]) -> u64 {
    members
        .iter()
        .map(|&t| g.tensors[t].bytes_a_unit())
        .max()
        .unwrap_or(0)
}

/// First fit of `classes` over their members' lifetimes, earliest first: each class's offset, and
/// the region's bytes.
fn assign(g: &Graph, classes: &[Vec<usize>]) -> (Vec<u64>, u64) {
    let lives: Vec<Vec<(usize, usize)>> = classes
        .iter()
        .map(|m| m.iter().filter_map(|&t| g.tensors[t].lifetime()).collect())
        .collect();
    let bytes: Vec<u64> = classes.iter().map(|m| class_bytes(g, m)).collect();
    let mut order: Vec<usize> = (0..classes.len()).collect();
    order.sort_by_key(|&c| {
        let start = lives[c].iter().map(|l| l.0).min().unwrap_or(0);
        (start, std::cmp::Reverse(bytes[c]), c)
    });
    let meet = |a: &[(usize, usize)], b: &[(usize, usize)]| {
        a.iter()
            .any(|&(sa, ea)| b.iter().any(|&(sb, eb)| sa <= eb && sb <= ea))
    };
    let mut offsets = vec![0u64; classes.len()];
    let mut placed: Vec<usize> = Vec::new();
    let mut region = 0;
    for c in order {
        let conflicts: Vec<(u64, u64)> = placed
            .iter()
            .filter(|&&d| meet(&lives[c], &lives[d]))
            .map(|&d| (offsets[d], offsets[d] + bytes[d]))
            .collect();
        let mut candidates: Vec<u64> = std::iter::once(0)
            .chain(
                conflicts
                    .iter()
                    .map(|&(_, end)| end.div_ceil(ALIGN) * ALIGN),
            )
            .collect();
        candidates.sort_unstable();
        let at = candidates
            .into_iter()
            .find(|&at| {
                conflicts
                    .iter()
                    .all(|&(s, e)| at + bytes[c] <= s || at >= e)
            })
            .expect("past every conflict fits");
        offsets[c] = at;
        region = region.max(at + bytes[c]);
        placed.push(c);
    }
    (offsets, region)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SLOT: u64 = tt_isa::dm::TILE_SLOT;

    /// `(tensor, unit, writes, tiles)`.
    type Access<'a> = (usize, usize, bool, &'a [u32]);

    /// A graph from ops given as `(signature, [(tensor, unit, writes, tiles)])`, each tensor
    /// internal, of `tiles` tiles.
    fn graph(tiles: &[usize], ops: &[(&str, &[Access])]) -> Graph {
        let mut tensors: Vec<Tensor> = tiles
            .iter()
            .map(|&n| Tensor {
                tiles: n,
                slot_bytes: SLOT,
                internal: true,
                uses: Vec::new(),
            })
            .collect();
        let mut out = Vec::new();
        for (k, (sig, accesses)) in ops.iter().enumerate() {
            let mut uses: BTreeMap<usize, Use> = BTreeMap::new();
            let mut order = Vec::new();
            for &(t, unit, write, ts) in accesses.iter() {
                if !uses.contains_key(&t) {
                    order.push(t);
                }
                let u = uses.entry(t).or_insert_with(|| Use {
                    op: k,
                    plain: true,
                    ..Use::default()
                });
                let side = if write { &mut u.writes } else { &mut u.reads };
                side.entry(unit).or_default().extend(ts.iter().copied());
                u.bytes[usize::from(write)] += ts.len() as u64 * SLOT;
            }
            for (t, u) in uses {
                tensors[t].uses.push(u);
            }
            out.push(Op {
                label: "",
                what: "op",
                signature: sig.to_string(),
                l1_top: tt_isa::l1::DATA.base + 4096,
                l1_need: tt_isa::l1::DATA.base + 4096,
                tensors: order,
                external: [0, 0],
            });
        }
        Graph {
            ops: out,
            tensors,
            unattributed: 0,
        }
    }

    /// `layers` layers of a pre-norm block: `n = norm(x)`, `y = f(n)`, `h = add(x, y)`,
    /// `m = g(h)`, `x' = add2(h, m)`. Layer `i`'s tensors are `5 i..5 i + 5` (`x`, `n`, `y`, `h`,
    /// `m`), its `x` the head's for `i = 0`; the last `x'` is `5 layers`, which the tail reads.
    fn residual(layers: usize) -> Graph {
        let tiles = vec![1; 5 * layers + 1];
        type Owned = (usize, usize, bool, Vec<u32>);
        let r = |t: usize| (t, 0, false, vec![0]);
        let w = |t: usize| (t, 0, true, vec![0]);
        let mut ops: Vec<(&str, Vec<Owned>)> = vec![("head", vec![w(0)])];
        for i in 0..layers {
            let [x, n, y, h, m, next] = std::array::from_fn(|k| 5 * i + k);
            ops.push(("norm", vec![r(x), w(n)]));
            ops.push(("f", vec![r(n), w(y)]));
            ops.push(("add", vec![r(x), r(y), w(h)]));
            ops.push(("g", vec![r(h), w(m)]));
            ops.push(("add2", vec![r(h), r(m), w(next)]));
        }
        ops.push(("tail", vec![r(5 * layers)]));
        let ops: Vec<(&str, Vec<Access>)> = ops
            .iter()
            .map(|(s, a)| {
                (
                    *s,
                    a.iter()
                        .map(|(t, u, w, ts)| (*t, *u, *w, ts.as_slice()))
                        .collect(),
                )
            })
            .collect();
        let ops: Vec<(&str, &[Access])> = ops.iter().map(|(s, a)| (*s, a.as_slice())).collect();
        graph(&tiles, &ops)
    }

    fn offset(p: &Plan, t: usize) -> u64 {
        match p.place[t] {
            Place::Resident { offset, .. } => offset,
            Place::Gddr(r) => panic!("tensor {t} in GDDR: {r:?}"),
        }
    }

    /// A tile slot, as the region aligns it.
    const ONE: u64 = SLOT.div_ceil(ALIGN) * ALIGN;

    #[test]
    fn the_period_of_repeated_layers_is_found() {
        let g = residual(4);
        let sigs: Vec<&String> = g.ops.iter().map(|o| &o.signature).collect();
        let p = period(&sigs).expect("four layers repeat");
        assert_eq!((p.p, p.start, p.repeats), (5, 1, 4));
        assert_eq!(period(&["a", "b", "c", "d"]), None);
    }

    /// The residual stream takes two slots, the same in every layer: the first add cannot write
    /// where it reads `x`, the second writes the next `x` where this one was, so the next layer
    /// finds its `x` where this one found its own -- the head's `x` included.
    #[test]
    fn the_residual_stream_alternates_two_slots_the_same_in_every_layer() {
        let g = residual(4);
        let p = Plan::make(&g, tt_isa::l1::DATA.end, |_, _| true);
        p.check(&g).expect("the plan keeps its rules");
        assert_eq!(p.resident().count(), g.tensors.len());
        let xs: Vec<u64> = (0..=4).map(|i| offset(&p, 5 * i)).collect();
        assert!(
            xs.windows(2).all(|w| w[0] == w[1]),
            "every layer's x at one address: {xs:?}"
        );
        let hs: Vec<u64> = (0..4).map(|i| offset(&p, 5 * i + 3)).collect();
        assert!(
            hs.windows(2).all(|w| w[0] == w[1]),
            "every layer's h at one address: {hs:?}"
        );
        assert_ne!(xs[0], hs[0]);
        // At most three live at once (x, y and h at the first add).
        assert_eq!(p.region, 3 * ONE);
    }

    #[test]
    fn a_plan_is_the_same_each_time() {
        let (a, b) = (residual(5), residual(5));
        let (pa, pb) = (
            Plan::make(&a, tt_isa::l1::DATA.end, |_, _| true),
            Plan::make(&b, tt_isa::l1::DATA.end, |_, _| true),
        );
        assert_eq!(pa.place, pb.place);
        assert_eq!(pa.signature(&a), pb.signature(&b));
    }

    /// The negative controls: a slot reused before its last reader, and an op that would read and
    /// write one slot, are refused; so is an address that differs a period later.
    #[test]
    fn a_mutated_plan_is_refused() {
        let g = residual(4);
        let good = Plan::make(&g, tt_isa::l1::DATA.end, |_, _| true);
        good.check(&g).unwrap();
        let x = offset(&good, 5);
        let moved = |t: usize, to: u64| {
            let mut bad = good.clone();
            if let Place::Resident { offset, .. } = &mut bad.place[t] {
                *offset = to;
            }
            bad
        };
        // Layer 1's `y` where its `x` is: `f` writes it while `x` waits for the add.
        let bad = moved(7, x);
        assert!(
            matches!(bad.check(&g), Err(Hazard::Overlap { .. })),
            "{:?}",
            bad.check(&g)
        );
        // Layer 1's `h` where its `x` is: the add (op 8) would read and write one slot.
        let bad = moved(8, x);
        assert_eq!(bad.check(&g), Err(Hazard::Overlap { a: 5, b: 8, op: 8 }));
        // Layer 2's `n` somewhere free: the layer no longer matches the one before it.
        let mut bad = moved(11, good.region);
        bad.region += ONE;
        assert!(
            matches!(bad.check(&g), Err(Hazard::NotPeriodic { .. })),
            "{:?}",
            bad.check(&g)
        );
    }

    /// With room for less than the plan, the costliest address (largest bytes x lifetime) leaves
    /// first, with every tensor that shares it; what stays still keeps the rules.
    #[test]
    fn what_does_not_fit_leaves_costliest_first() {
        let g = residual(4);
        let full = Plan::make(&g, tt_isa::l1::DATA.end, |_, _| true);
        let need = tt_isa::l1::DATA.base + 4096;
        let tight = Plan::make(&g, need + full.region - 1, |_, _| true);
        tight.check(&g).unwrap();
        assert!(tight.region < full.region);
        // The `x` address lives longest: it goes first.
        assert_eq!(tight.place[0], Place::Gddr(Reason::Room));
        assert_eq!(tight.place[5], Place::Gddr(Reason::Room));
        assert!(matches!(tight.place[1], Place::Resident { .. }));
    }

    /// Pinned, an address holds its class's largest member in every repeat: where a layer's large
    /// tensors are different ones in different layers (`a` 4 tiles in the first, `b` in the second),
    /// the classes take 8 slots where the tensors need 5. With room for 5, the pinned plan leaves a
    /// class out (`Room`), so the plan without the pinning, which keeps everything, is the one kept.
    #[test]
    fn a_pinning_that_costs_room_is_not_kept() {
        // Tensors: per layer `a`, then `b`; then the tail's input.
        let sizes = [(4usize, 1usize), (1, 4), (1, 1)];
        let mut tiles = Vec::new();
        for &(a, b) in &sizes {
            tiles.extend([a, b]);
        }
        tiles.push(1);
        let all: Vec<Vec<u32>> = tiles.iter().map(|&n| (0..n as u32).collect()).collect();
        let mut ops: Vec<(&str, Vec<Access>)> = vec![("head", vec![(6, 0, true, &all[6][..])])];
        for i in 0..sizes.len() {
            let (a, b) = (2 * i, 2 * i + 1);
            ops.push(("p", vec![(a, 0, true, &all[a][..])]));
            ops.push((
                "q",
                vec![(a, 0, false, &all[a][..]), (b, 0, true, &all[b][..])],
            ));
            ops.push(("s", vec![(b, 0, false, &all[b][..])]));
        }
        ops.push(("tail", vec![(6, 0, false, &all[6][..])]));
        let ops: Vec<(&str, &[Access])> = ops.iter().map(|(s, a)| (*s, a.as_slice())).collect();
        let g = graph(&tiles, &ops);
        let budget = 6 * ONE;
        let pinned = Plan::make_inner(
            &g,
            budget,
            Rows::Multicast,
            &|_, _| true,
            period(&g.ops.iter().map(|o| &o.signature).collect::<Vec<_>>()),
            LeaveOut::Longest,
        );
        assert!(
            pinned.place.contains(&Place::Gddr(Reason::Room)),
            "pinned, the classes do not fit: {:?}",
            pinned.place
        );
        let p = Plan::make_within(&g, budget, |_, _| true);
        p.check(&g).unwrap();
        assert!(!p.pinned);
        assert!(
            p.place.iter().all(|p| matches!(p, Place::Resident { .. })),
            "{:?}",
            p.place
        );
    }

    /// Phase 4: a small tensor read by many ops over a long life (a mask, a table every layer
    /// reads) spares more a byte than a large one read once beside it, so where only one fits it
    /// is the one kept, though it holds its room longer.
    #[test]
    fn a_short_region_keeps_what_spares_most_a_byte() {
        // Tensor 0: one tile, written by op 0, read by ops 1-8. Tensor 1: three tiles, written
        // by op 4, read by op 5. At op 4 both live: four slots against a budget of three.
        let one: &[u32] = &[0];
        let three: &[u32] = &[0, 1, 2];
        let sigs: Vec<String> = (0..9).map(|k| format!("op{k}")).collect();
        let mut ops: Vec<(&str, Vec<Access>)> = vec![(&sigs[0], vec![(0, 0, true, one)])];
        for (k, sig) in sigs.iter().enumerate().skip(1) {
            let mut a = vec![(0, 0, false, one)];
            match k {
                4 => a.push((1, 0, true, three)),
                5 => a.push((1, 0, false, three)),
                _ => {}
            }
            ops.push((sig, a));
        }
        let ops: Vec<(&str, &[Access])> = ops.iter().map(|(s, a)| (*s, a.as_slice())).collect();
        let g = graph(&[1, 3], &ops);
        let budget = 3 * ONE;
        let longest = Plan::make_inner(
            &g,
            budget,
            Rows::Multicast,
            &|_, _| true,
            None,
            LeaveOut::Longest,
        );
        assert_eq!(longest.place[0], Place::Gddr(Reason::Room));
        assert!(matches!(longest.place[1], Place::Resident { .. }));
        let p = Plan::make_within(&g, budget, |_, _| true);
        p.check(&g).unwrap();
        assert!(
            matches!(p.place[0], Place::Resident { .. }),
            "{:?}",
            p.place
        );
        assert_eq!(p.place[1], Place::Gddr(Reason::Room));
        assert!(
            p.gddr_avoided(&g).iter().sum::<u64>() > longest.gddr_avoided(&g).iter().sum::<u64>()
        );
        // Deterministic.
        assert_eq!(p.place, Plan::make_within(&g, budget, |_, _| true).place);
    }

    #[test]
    fn outputs_and_ops_without_a_path_stay_in_gddr() {
        let mut g = residual(2);
        g.tensors[10].internal = false;
        let p = Plan::make(&g, tt_isa::l1::DATA.end, |_, _| true);
        // The last `x` is the graph's output. Pinned, every `x` would share its address and stay
        // in GDDR with it; the plan without the pinning spares more, so it is the one kept.
        assert_eq!(p.place[10], Place::Gddr(Reason::Output));
        assert!(!p.pinned);
        assert!(matches!(p.place[5], Place::Resident { .. }));
        assert!(matches!(p.place[3], Place::Resident { .. }));
        p.check(&g).unwrap();
        // No L1 path for `f`: what it touches stays.
        let p = Plan::make(&g, tt_isa::l1::DATA.end, |o, _| o.signature != "f");
        assert_eq!(p.place[2], Place::Gddr(Reason::NoPath));
        assert_eq!(p.place[1], Place::Gddr(Reason::NoPath));
        assert!(matches!(p.place[4], Place::Resident { .. }));
        p.check(&g).unwrap();
    }

    /// A tensor that one op both reads and writes, or that is read before it is written, stays.
    #[test]
    fn in_place_and_read_first_tensors_stay_in_gddr() {
        let g = graph(
            &[1, 1],
            &[
                ("a", &[(0, 0, false, &[0]), (1, 0, true, &[0])]),
                ("b", &[(1, 0, false, &[0]), (1, 0, true, &[0])]),
                ("c", &[(1, 0, false, &[0])]),
            ],
        );
        let p = Plan::make(&g, tt_isa::l1::DATA.end, |_, _| true);
        assert_eq!(p.place[0], Place::Gddr(Reason::ReadFirst));
        assert_eq!(p.place[1], Place::Gddr(Reason::ReadAndWritten));
    }

    #[test]
    fn rectangles_cover_exactly_the_tiles() {
        assert_eq!(rectangles(&[(1, 2)]), [((1, 2), (1, 2))]);
        // Two full rows of 1..=3: one rectangle; a gap at column 8 splits a row.
        let two_rows: Vec<(u8, u8)> = (2..=3).flat_map(|y| (1..=3).map(move |x| (x, y))).collect();
        assert_eq!(rectangles(&two_rows), [((1, 2), (3, 3))]);
        assert_eq!(
            rectangles(&[(7, 2), (10, 2)]),
            [((7, 2), (7, 2)), ((10, 2), (10, 2))]
        );
        // A ragged last row: the full rows, then the rest.
        let mut ragged = two_rows.clone();
        ragged.push((1, 4));
        let r = rectangles(&ragged);
        let covered: BTreeSet<(u8, u8)> = r
            .iter()
            .flat_map(|&(s, e)| (s.1..=e.1).flat_map(move |y| (s.0..=e.0).map(move |x| (x, y))))
            .collect();
        assert_eq!(covered, ragged.iter().copied().collect());
        assert_eq!(
            covered.len(),
            r.iter()
                .map(|&(s, e)| (e.0 - s.0 + 1) as usize * (e.1 - s.1 + 1) as usize)
                .sum()
        );
    }

    /// `h` holders split the readers into contiguous runs, each held by its middle reader; every
    /// reader has a holder among them; `h` past the readers is every reader; the same readers
    /// give the same holders.
    #[test]
    fn holders_split_the_readers_evenly() {
        let readers: Vec<usize> = (0..104).collect();
        let (held, of) = holders(&readers, 4);
        assert_eq!(held, [13, 39, 65, 91]);
        assert_eq!(of.len(), readers.len());
        assert!(of.values().all(|h| held.contains(h)));
        assert_eq!((of[&0], of[&25], of[&26], of[&103]), (13, 13, 39, 91));
        let per: Vec<usize> = held
            .iter()
            .map(|h| of.values().filter(|&v| v == h).count())
            .collect();
        assert_eq!(per, [26, 26, 26, 26]);
        // Not contiguous unit indices: the readers' own, in order.
        let (held, of) = holders(&[3, 5, 8, 40], 2);
        assert_eq!(held, [5, 40]);
        assert_eq!((of[&3], of[&8]), (5, 40));
        let (held, _) = holders(&[3, 5], 8);
        assert_eq!(held, [3, 5], "more holders than readers: every reader");
        assert_eq!(holders(&readers, 1).0, [52]);
        assert_eq!(holders(&readers, 4), holders(&readers, 4));
    }

    /// A packed twin (2 tiles, tensor 0) written by units 0-3 a tile each and read whole by units
    /// 0-7, its rewrite applied with `rows`: the `Applied`, the twin and the `Index` of its GDDR.
    fn twin_applied(rows: Rows) -> (Applied, crate::tensor::DramTensor, Index) {
        let mut alloc = crate::tensor::DramAlloc::new(&tt_isa::dram::Dram::FULL);
        alloc.start_log();
        let twin = crate::tensor::DramTensor::alloc(&mut alloc, 64, 32).unwrap();
        let index = Index::new(alloc.logged());
        let writes: Vec<Access> = (0..4)
            .map(|u| (0, u, true, if u < 2 { &[0u32][..] } else { &[1u32][..] }))
            .collect();
        let reads: Vec<Access> = (0..8).map(|u| (0, u, false, &[0u32, 1][..])).collect();
        let g = graph(&[2], &[("mul", &writes), ("matmul", &reads)]);
        let plan = Plan::make_rows(&g, 96 << 10, rows, |_, _| true);
        plan.check(&g).unwrap();
        let coords = [1u8, 2, 3, 4, 5, 6, 7, 10].map(|x| (x, 2)).to_vec();
        let base = tt_isa::l1::DATA.end - (96 << 10);
        let applied = Applied::new(g, plan, vec![0, 0], base, coords).expect("the twin resident");
        (applied, twin, index)
    }

    /// The reads of the whole twin into slots from `at`: the packed `A` read a product queues.
    fn twin_reads(twin: &crate::tensor::DramTensor, at: u32) -> Vec<[u32; 8]> {
        let mut e = vec![[record::READ_RUN, 0, 2, at, 0, 1, 0, 0]];
        e.extend(twin.tensor_ref().encode());
        e
    }

    /// `Rows::Held`: a twin read whole by eight units is held by two of them; each producer's
    /// face-row writes become one `WRITE_L1` to each holder and no multicast; each reader's two
    /// tile reads become one `READ_L1` of its holder's region.
    #[test]
    fn a_held_row_is_written_to_its_holders_and_read_from_them() {
        let (a, twin, index) = twin_applied(Rows::Held(2));
        let held = &a.plan.held[&0];
        assert_eq!(held.holders, [2, 6]);
        let at = |u: usize| a.address(0, u, 0).unwrap();
        assert_eq!(at(2), at(6), "every holder holds the row at one address");
        assert!(
            a.address(0, 1, 0).is_none(),
            "a reader that is no holder holds nothing"
        );
        // Unit 1 writes output tiles 16..32: twin tile 0's rows 16-31, two face rows each.
        let writes = crate::tensor::packed_twin_writes(&twin, 16, 16, 0x4_0000);
        let out = a.rewrite(&index, 0, 1, &writes).unwrap();
        assert_eq!(out.len(), 2 * writes.len());
        assert!(out.iter().all(|e| e[0] == op::WRITE_L1));
        for (w, pair) in writes.iter().zip(out.chunks(2)) {
            let (_, tile, byte) = index.locate(w[1], w[3]).unwrap();
            assert_eq!(tile, 0);
            for (e, v) in pair.iter().zip([2usize, 6]) {
                let (x, y) = a.coords[v];
                assert_eq!(
                    *e,
                    [
                        op::WRITE_L1,
                        x as u32,
                        y as u32,
                        (at(v) + byte) as u32,
                        w[4],
                        64,
                        0,
                        0
                    ]
                );
            }
        }
        // Each reader reads both tiles from its holder, as one request.
        for (u, holder) in [(0, 2), (3, 2), (4, 6), (7, 6), (6, 6)] {
            let out = a
                .rewrite(&index, 1, u, &twin_reads(&twin, 0x5_0000))
                .unwrap();
            let (x, y) = a.coords[holder];
            assert_eq!(
                out,
                [[
                    op::READ_L1,
                    x as u32,
                    y as u32,
                    at(holder) as u32,
                    0x5_0000,
                    2 * SLOT as u32,
                    0,
                    0
                ]],
                "unit {u}"
            );
        }
    }

    /// Phase 3: a product's output (`[1, 4]` tiles, tensor 0) written by `SCATTER`s, units 0 and 1
    /// two columns each, and read by the next op sharded: tile 0 by its writer, tile 1 by unit 2,
    /// tile 2 by unit 3, tile 3 by units 4 and 5 (a RoPE partner). Each expanded write is plain
    /// and becomes a `WRITE_L1` to each reader, by unicast even where two readers are neighbours;
    /// each reader reads its tile from its own region.
    #[test]
    fn a_scatter_writes_into_its_readers_l1() {
        let mut alloc = crate::tensor::DramAlloc::new(&tt_isa::dram::Dram::FULL);
        alloc.start_log();
        let c = crate::tensor::DramTensor::alloc(&mut alloc, 32, 128).unwrap();
        let index = Index::new(alloc.logged());
        let scatter = |j0: u32| {
            let mut e = vec![[record::SCATTER, 0x4_0010, SLOT as u32, 0, 1, j0, 2, 0]];
            e.extend(c.tensor_ref().encode());
            e
        };
        let mut whole = Vec::new();
        assert_eq!(walk(&scatter(0), &mut |w, p| whole.push((w[0], p))), 0);
        assert_eq!(
            whole,
            [(op::WRITE, true); 2],
            "a SCATTER's writes are plain"
        );
        let writes: Vec<Access> = vec![(0, 0, true, &[0, 1]), (0, 1, true, &[2, 3])];
        let reads: Vec<Access> = vec![
            (0, 0, false, &[0]),
            (0, 2, false, &[1]),
            (0, 3, false, &[2]),
            (0, 4, false, &[3]),
            (0, 5, false, &[3]),
        ];
        let g = graph(&[4], &[("matmul", &writes), ("add", &reads)]);
        let plan = Plan::make_rows(&g, 96 << 10, Rows::Held(HOLDERS), |_, u| u.plain);
        plan.check(&g).unwrap();
        assert!(plan.held.is_empty(), "a sharded output is not held");
        let coords = [1u8, 2, 3, 4, 5, 6].map(|x| (x, 2)).to_vec();
        let base = tt_isa::l1::DATA.end - (96 << 10);
        let a = Applied::new(g, plan, vec![0, 0], base, coords).expect("the output resident");
        let at = |u: usize, tile: u32| a.address(0, u, tile).unwrap() as u32;
        let to = |u: usize, tile: u32, from: u32| {
            let (x, y) = a.coords[u];
            [
                op::WRITE_L1,
                x as u32,
                y as u32,
                at(u, tile) + 16,
                from,
                4096,
                0,
                0,
            ]
        };
        assert_eq!(
            a.rewrite(&index, 0, 0, &scatter(0)).unwrap(),
            [to(0, 0, 0x4_0010), to(2, 1, 0x4_0010 + SLOT as u32)]
        );
        assert_eq!(
            a.rewrite(&index, 0, 1, &scatter(2)).unwrap(),
            [
                to(3, 2, 0x4_0010),
                to(4, 3, 0x4_0010 + SLOT as u32),
                to(5, 3, 0x4_0010 + SLOT as u32)
            ]
        );
        // Unit 5 reads tile 3 (a `READ_RUN` of one tile) from its own region.
        let mut read = vec![[record::READ_RUN, 3, 1, 0x5_0000, 0, 4, 0, 0]];
        read.extend(c.tensor_ref().encode());
        let (x, y) = a.coords[5];
        assert_eq!(
            a.rewrite(&index, 1, 5, &read).unwrap(),
            [[
                op::READ_L1,
                x as u32,
                y as u32,
                at(5, 3),
                0x5_0000,
                SLOT as u32,
                0,
                0
            ]]
        );
    }

    /// `Rows::Multicast` keeps Phase 2's copies (a multicast a write), and `Rows::Gddr` keeps the
    /// row in GDDR; a held plan whose reader reads from a unit that is not a holder is refused.
    #[test]
    fn rows_choose_the_delivery_and_a_misrouted_holder_is_refused() {
        let (a, twin, index) = twin_applied(Rows::Multicast);
        assert!(a.plan.held.is_empty());
        let writes = crate::tensor::packed_twin_writes(&twin, 0, 1, 0x4_0000);
        let out = a.rewrite(&index, 0, 0, &writes).unwrap();
        assert!(out.iter().any(|e| e[0] == op::WRITE_L1_MCAST), "{out:x?}");
        let held = twin_applied(Rows::Held(2)).0;
        let reads: Vec<Access> = (0..8).map(|u| (0, u, false, &[0u32, 1][..])).collect();
        let writes: Vec<Access> = vec![(0, 0, true, &[0u32, 1][..])];
        let g = graph(&[2], &[("mul", &writes), ("matmul", &reads)]);
        let p = Plan::make_rows(&g, 96 << 10, Rows::Gddr, |_, _| true);
        assert_eq!(p.place[0], Place::Gddr(Reason::NoPath));
        // The negative control: a reader pointed at a unit that holds nothing.
        let mut bad = held.plan.clone();
        bad.held.get_mut(&0).unwrap().of.insert(0, 1);
        assert_eq!(bad.check(&held.graph), Err(Hazard::NotHeld { tensor: 0 }));
        let mut bad = held.plan.clone();
        bad.held.get_mut(&0).unwrap().of.remove(&5);
        assert_eq!(bad.check(&held.graph), Err(Hazard::NotHeld { tensor: 0 }));
        assert_eq!(held.plan.check(&held.graph), Ok(()));
    }

    /// 2g: a tensor the graph makes and never reads (tensor 1, beside a held twin) has its writes
    /// dropped: they leave the list, the twin's are rewritten as before. A read of it is refused
    /// by the rewrite, and a plan dropping a tensor that is read by the check.
    #[test]
    fn a_never_read_tensor_s_writes_are_dropped() {
        let mut alloc = crate::tensor::DramAlloc::new(&tt_isa::dram::Dram::FULL);
        alloc.start_log();
        let twin = crate::tensor::DramTensor::alloc(&mut alloc, 64, 32).unwrap();
        let tiles = crate::tensor::DramTensor::alloc(&mut alloc, 1, 64).unwrap();
        let index = Index::new(alloc.logged());
        let writes: Vec<Access> = vec![(0, 0, true, &[0u32, 1][..]), (1, 0, true, &[0u32, 1][..])];
        let reads: Vec<Access> = (0..8).map(|u| (0, u, false, &[0u32, 1][..])).collect();
        let g = graph(&[2, 2], &[("mul", &writes), ("matmul", &reads)]);
        let plan = Plan::make_rows(&g, 96 << 10, Rows::Held(2), |_, _| true);
        assert_eq!(plan.place[1], Place::Gddr(Reason::NeverRead));
        assert_eq!(plan.dropped, BTreeSet::from([1]));
        plan.check(&g).unwrap();
        let coords = [1u8, 2, 3, 4, 5, 6, 7, 10].map(|x| (x, 2)).to_vec();
        let base = tt_isa::l1::DATA.end - (96 << 10);
        let a = Applied::new(g.clone(), plan.clone(), vec![0, 0], base, coords).unwrap();
        // The tile writes of tensor 1 (a `WRITE_RUN`) vanish; the twin's face rows do not.
        let mut list = vec![[record::WRITE_RUN, 0, 2, 0x4_0000, 0, 2, 0, 0]];
        list.extend(tiles.tensor_ref().encode());
        assert_eq!(
            a.rewrite(&index, 0, 0, &list).unwrap(),
            Vec::<[u32; 8]>::new()
        );
        let twin_writes = crate::tensor::packed_twin_writes(&twin, 0, 2, 0x4_0000);
        let out = a.rewrite(&index, 0, 0, &twin_writes).unwrap();
        assert_eq!(out.len(), 2 * twin_writes.len(), "each to two holders");
        // A read of it: refused.
        let mut read = vec![[record::READ_RUN, 0, 2, 0x5_0000, 0, 2, 0, 0]];
        read.extend(tiles.tensor_ref().encode());
        let why = a.rewrite(&index, 1, 3, &read).unwrap_err();
        assert!(why.contains("dropped"), "{why}");
        // The negative control: the plan drops the twin, which is read.
        let mut bad = plan.clone();
        bad.dropped.insert(0);
        assert_eq!(bad.check(&g), Err(Hazard::ReadDropped { tensor: 0 }));
    }

    /// A row read whole by some units and in part by another is not read alike: no one holder's
    /// copy is what each reads, so `Rows::Held` leaves it in GDDR.
    #[test]
    fn a_row_not_read_alike_is_not_held() {
        let writes: Vec<Access> = vec![(0, 0, true, &[0u32, 1][..])];
        let whole: Vec<Access> = (0..4).map(|u| (0, u, false, &[0u32, 1][..])).collect();
        let part: Vec<Access> = vec![(0, 5, false, &[1u32][..])];
        let g = graph(
            &[2],
            &[("mul", &writes), ("matmul", &whole), ("add", &part)],
        );
        assert!(g.tensors[0].replicated() && !g.tensors[0].read_alike());
        let p = Plan::make_rows(&g, 96 << 10, Rows::Held(2), |_, _| true);
        assert_eq!(p.place[0], Place::Gddr(Reason::NoPath));
        let p = Plan::make_rows(&g, 96 << 10, Rows::Multicast, |_, _| true);
        assert!(matches!(p.place[0], Place::Resident { .. }));
    }

    #[test]
    fn layouts_are_read_off_the_units() {
        let one = |unit: usize, ts: &[u32]| (unit, ts.iter().copied().collect::<BTreeSet<u32>>());
        let u = |reads: Vec<(usize, BTreeSet<u32>)>| Use {
            reads: reads.into_iter().collect(),
            ..Use::default()
        };
        assert_eq!(
            u(vec![one(0, &[0, 1]), one(1, &[0, 1])]).layout(),
            Layout::Replicated
        );
        assert_eq!(
            u(vec![one(0, &[0]), one(1, &[1])]).layout(),
            Layout::Sharded
        );
        assert_eq!(
            u(vec![one(0, &[0, 1]), one(1, &[1, 2])]).layout(),
            Layout::Mixed
        );
        // Bytes a unit: the most any unit holds, over every reader.
        let t = Tensor {
            tiles: 2,
            slot_bytes: SLOT,
            internal: true,
            uses: vec![u(vec![one(0, &[0]), one(1, &[1])]), u(vec![one(0, &[1])])],
        };
        assert_eq!(t.bytes_a_unit(), (2 * SLOT).div_ceil(ALIGN) * ALIGN);
    }
}
