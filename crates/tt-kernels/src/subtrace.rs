//! Layer sub-traces: a captured stream whose ops repeat, stored as one layer body and a call a
//! layer (`docs/plans/layer-sub-traces.md`).
//!
//! A decoder's layers capture the same entries layer after layer, but for the tensors they
//! touch and two counters (the kernels' generations, the barriers' targets) that advance by a
//! constant a layer. So one unit's repeat of the ops, its *layer*, can be stored once as a
//! *body*: each tensor reference that differs between layers replaced by an argument
//! ([`dm::ARG_REF`]), each plain move whose GDDR address differs marked to take one
//! ([`dm::ARG_WORD`]); and every layer that equals the body but for those, as a *call*: the
//! layer's arguments and its two counter offsets ([`dm::op::CALL_BODY`]).
//!
//! What a layer is, and which ops repeat, is found from the capture alone
//! ([`crate::residency::period`]): nothing here knows a model. Any other word that differs keeps
//! that layer as it was captured, and every call is expanded back and compared with its layer
//! word for word before anything is stored ([`check`]).

use std::collections::BTreeMap;
use std::ops::Range;

use tt_isa::dm::{self, op, record};

use crate::residency::{self, Period};
use crate::trace::{Capture, OpRecord, UnitCapture};

type Entry = [u32; 8];

/// One repeat of a unit's stream: its own entries and, in order, the reader and writer sections
/// of every `PAIR_CALL` among them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Layer {
    pub(crate) stream: Vec<Entry>,
    pub(crate) sections: Vec<Vec<Entry>>,
}

impl Layer {
    /// Unit stream `uc`'s entries `r`, with their `PAIR_CALL`s' sections.
    pub(crate) fn of(uc: &UnitCapture, r: Range<usize>) -> Layer {
        let mut sections = Vec::new();
        for (i, &at) in uc.pair_calls.iter().enumerate() {
            if r.contains(&at) {
                sections.push(uc.pending_sections[2 * i].clone());
                sections.push(uc.pending_sections[2 * i + 1].clone());
            }
        }
        Layer {
            stream: uc.stream[r].to_vec(),
            sections,
        }
    }

    /// Its parts: the stream (0), then each pair's reader (odd) and writer (even) section.
    fn parts(&self) -> usize {
        1 + self.sections.len()
    }

    fn part(&self, p: usize) -> &[Entry] {
        if p == 0 {
            &self.stream
        } else {
            &self.sections[p - 1]
        }
    }

    fn part_mut(&mut self, p: usize) -> &mut Vec<Entry> {
        if p == 0 {
            &mut self.stream
        } else {
            &mut self.sections[p - 1]
        }
    }

    /// Entries in the stream and the sections.
    pub(crate) fn entries(&self) -> usize {
        self.stream.len() + self.sections.iter().map(Vec::len).sum::<usize>()
    }
}

/// Whether part `p` runs with the call's generation base: the stream and a pair's reader section
/// (B passes its base to the reader, `dm::op::PAIR_CALL`); NC runs a writer section with none.
fn takes_generations(p: usize) -> bool {
    p == 0 || p % 2 == 1
}

/// Whether part `p` runs with the call's barrier base: the stream only.
fn takes_barriers(p: usize) -> bool {
    p == 0
}

/// What a layer takes as an argument: a tensor reference (two entries) or a plain move's channel,
/// port and offset (one), at entry `at` of part `part`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Slot {
    part: usize,
    at: usize,
    reference: bool,
}

impl Slot {
    fn entries(self) -> usize {
        if self.reference {
            2
        } else {
            1
        }
    }
}

/// Why a layer cannot be called with a body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Unlike {
    /// Their parts' lengths or their entries' op words differ.
    Shape,
    /// A generation or a barrier target differs by other than the layer's one offset.
    Counter { part: usize, at: usize },
    /// Some other word differs: word `word` of entry `at` in part `part`, of a record or entry
    /// whose op is `head`.
    Word {
        part: usize,
        at: usize,
        word: usize,
        head: u32,
    },
}

/// How `other` differs from `body`, when that is only arguments and counters.
struct Diff {
    generation: Option<u32>,
    barrier: Option<u32>,
    slots: Vec<Slot>,
}

fn offset(
    o: &mut Option<u32>,
    body: u32,
    other: u32,
    part: usize,
    at: usize,
) -> Result<(), Unlike> {
    let d = other.wrapping_sub(body);
    match *o {
        None => {
            *o = Some(d);
            Ok(())
        }
        Some(x) if x == d => Ok(()),
        Some(_) => Err(Unlike::Counter { part, at }),
    }
}

fn diff(body: &Layer, other: &Layer) -> Result<Diff, Unlike> {
    if body.sections.len() != other.sections.len() {
        return Err(Unlike::Shape);
    }
    let mut d = Diff {
        generation: None,
        barrier: None,
        slots: Vec::new(),
    };
    for p in 0..body.parts() {
        let (a, b) = (body.part(p), other.part(p));
        if a.len() != b.len() {
            return Err(Unlike::Shape);
        }
        let mut i = 0;
        while i < a.len() {
            let head = a[i][0];
            let n = record::len(head).max(1);
            if b[i][0] != head || i + n > a.len() {
                return Err(Unlike::Shape);
            }
            let (x, y) = (&a[i], &b[i]);
            let same = |words: &[usize]| match words.iter().find(|&&w| x[w] != y[w]) {
                Some(&word) => Err(Unlike::Word {
                    part: p,
                    at: i,
                    word,
                    head,
                }),
                None => Ok(()),
            };
            match head {
                op::KERNEL | op::LAUNCH | op::KERNEL_WAIT if takes_generations(p) => {
                    offset(&mut d.generation, x[1], y[1], p, i)?;
                    same(&[2, 3, 4, 5, 6, 7])?;
                }
                op::BARRIER if takes_barriers(p) => {
                    offset(&mut d.barrier, x[1], y[1], p, i)?;
                    same(&[2, 3, 4, 5, 6, 7])?;
                }
                op::READ | op::WRITE if x[1..4] != y[1..4] => {
                    same(&[4, 5, 6, 7])?;
                    if x[6] != 0 {
                        return Err(Unlike::Word {
                            part: p,
                            at: i,
                            word: 6,
                            head,
                        });
                    }
                    d.slots.push(Slot {
                        part: p,
                        at: i,
                        reference: false,
                    });
                }
                _ => same(&[1, 2, 3, 4, 5, 6, 7])?,
            }
            for &r in record::references(head) {
                if a[i + r..i + r + 2] != b[i + r..i + r + 2] {
                    d.slots.push(Slot {
                        part: p,
                        at: i + r,
                        reference: true,
                    });
                }
            }
            if head == record::POKE_RUN && a[i + 1] != b[i + 1] {
                let word = (0..8).find(|&w| a[i + 1][w] != b[i + 1][w]).unwrap_or(0);
                return Err(Unlike::Word {
                    part: p,
                    at: i + 1,
                    word,
                    head,
                });
            }
            i += n;
        }
    }
    Ok(d)
}

/// One layer's call: its arguments and its counters' offsets from the body's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Call {
    pub(crate) generation_offset: u32,
    pub(crate) barrier_offset: u32,
    pub(crate) table: Vec<Entry>,
}

/// One unit's layers as a body and calls.
#[derive(Clone, Debug)]
pub(crate) struct UnitBody {
    /// The body: the layer it was made from, its arguments marked.
    pub(crate) body: Layer,
    /// Which layer that was.
    pub(crate) from: usize,
    /// Per layer, its call, or `None` where it stays as captured.
    pub(crate) calls: Vec<Option<Call>>,
}

impl UnitBody {
    pub(crate) fn called(&self) -> usize {
        self.calls.iter().flatten().count()
    }
}

/// Why a unit's layers make no body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum NoBody {
    /// No layer equals the next but for arguments and counters: the first one's difference.
    Unlike(Unlike),
    /// Fewer than two layers would be called.
    Fewer,
    /// The layers hold no entries: the unit runs none of the repeated ops.
    Empty,
    /// A call would take this many argument entries, more than the table holds.
    Table(usize),
}

/// `layers` as one body and a call each where they allow, with at most `table_entries`
/// arguments a call. The body is the first layer that equals the next but for arguments and
/// counters; the arguments are every slot where a called layer differs from it.
pub(crate) fn make(layers: &[Layer], table_entries: usize) -> Result<UnitBody, NoBody> {
    let mut first = None;
    let from =
        (0..layers.len().saturating_sub(1)).find(|&b| match diff(&layers[b], &layers[b + 1]) {
            Ok(_) => true,
            Err(e) => {
                first.get_or_insert(e);
                false
            }
        });
    let Some(from) = from else {
        return Err(first.map_or(NoBody::Fewer, NoBody::Unlike));
    };
    if layers[from].stream.is_empty() {
        return Err(NoBody::Empty);
    }
    let diffs: Vec<Option<Diff>> = layers.iter().map(|l| diff(&layers[from], l).ok()).collect();
    let mut slots: Vec<Slot> = diffs
        .iter()
        .flatten()
        .flat_map(|d| d.slots.iter().copied())
        .collect();
    slots.sort();
    slots.dedup();
    // What each slot takes in each called layer: slots that take the same in every layer (a
    // tensor several records read) share one argument.
    let what = |l: &Layer, s: &Slot| -> Vec<Entry> {
        let part = l.part(s.part);
        if s.reference {
            part[s.at..s.at + 2].to_vec()
        } else {
            let e = part[s.at];
            vec![[0, e[1], e[2], e[3], 0, 0, 0, 0]]
        }
    };
    let called: Vec<&Layer> = layers
        .iter()
        .zip(&diffs)
        .filter_map(|(l, d)| d.as_ref().map(|_| l))
        .collect();
    let mut args: BTreeMap<Vec<Entry>, u32> = BTreeMap::new();
    let mut unique: Vec<Slot> = Vec::new();
    let mut entries = 0u32;
    let at: Vec<u32> = slots
        .iter()
        .map(|s| {
            let key: Vec<Entry> = called.iter().flat_map(|l| what(l, s)).collect();
            *args.entry(key).or_insert_with(|| {
                unique.push(*s);
                entries += s.entries() as u32;
                entries - s.entries() as u32
            })
        })
        .collect();
    let entries = entries as usize;
    if entries > table_entries {
        return Err(NoBody::Table(entries));
    }
    if called.len() < 2 {
        return Err(NoBody::Fewer);
    }
    let mut body = layers[from].clone();
    for (s, &k) in slots.iter().zip(&at) {
        let part = body.part_mut(s.part);
        if s.reference {
            part[s.at] = [dm::ARG_REF, k, 0, 0, 0, 0, 0, 0];
            part[s.at + 1] = [0; 8];
        } else {
            let e = &mut part[s.at];
            e[1..4].copy_from_slice(&[u32::MAX, 0, 0]);
            e[6] = dm::ARG_WORD | k;
        }
    }
    let calls = layers
        .iter()
        .zip(&diffs)
        .map(|(l, d)| {
            let d = d.as_ref()?;
            Some(Call {
                generation_offset: d.generation.unwrap_or(0),
                barrier_offset: d.barrier.unwrap_or(0),
                table: unique.iter().flat_map(|s| what(l, s)).collect(),
            })
        })
        .collect();
    Ok(UnitBody { body, from, calls })
}

/// Why a body cannot be expanded with a call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Bad {
    /// An argument past the call's table: (part, entry, argument).
    Argument(usize, usize, u32),
}

/// What the mover runs for `body` called with `call` (`dm::op::CALL_BODY`): each generation in
/// the stream and the reader sections offset by the call's, each barrier target in the stream by
/// its own, and every argument taken from its table.
pub(crate) fn expand(body: &Layer, call: &Call) -> Result<Layer, Bad> {
    let mut out = body.clone();
    for p in 0..out.parts() {
        let part = out.part_mut(p);
        let mut i = 0;
        while i < part.len() {
            let head = part[i][0];
            match head {
                op::KERNEL | op::LAUNCH | op::KERNEL_WAIT if takes_generations(p) => {
                    part[i][1] = part[i][1].wrapping_add(call.generation_offset);
                }
                op::BARRIER if takes_barriers(p) => {
                    part[i][1] = part[i][1].wrapping_add(call.barrier_offset);
                }
                op::READ | op::WRITE if part[i][6] & dm::ARG_WORD != 0 => {
                    let k = part[i][6] & !dm::ARG_WORD;
                    let t = call.table.get(k as usize).ok_or(Bad::Argument(p, i, k))?;
                    part[i][1..4].copy_from_slice(&t[1..4]);
                    part[i][6] = 0;
                }
                _ => {}
            }
            for &r in record::references(head) {
                if part[i + r][0] == dm::ARG_REF {
                    let k = part[i + r][1];
                    let t = call
                        .table
                        .get(k as usize..k as usize + 2)
                        .ok_or(Bad::Argument(p, i + r, k))?;
                    part[i + r] = t[0];
                    part[i + r + 1] = t[1];
                }
            }
            i += record::len(head).max(1);
        }
    }
    Ok(out)
}

/// Whether every call of `b` expands to its layer of `layers`, word for word: the check made
/// before a body is stored.
pub(crate) fn check(layers: &[Layer], b: &UnitBody) -> Result<(), String> {
    for (i, call) in b.calls.iter().enumerate() {
        let Some(call) = call else { continue };
        let e = expand(&b.body, call).map_err(|e| format!("layer {i}: {e:?}"))?;
        if e != layers[i] {
            return Err(format!("layer {i} does not expand to what was captured"));
        }
    }
    Ok(())
}

/// Unit `u`'s stream ranges of the repeats `period` finds in `ops`: repeat `i` from where op
/// `start + i p` begins on the unit to where the next repeat (or the op after the last, or the
/// stream's end, at `len`) begins, so what was moved between two ops (a busy unit's prefetch
/// list) is the repeat's it sits in.
pub(crate) fn bounds(ops: &[OpRecord], len: usize, period: &Period, u: usize) -> Vec<Range<usize>> {
    let start = |k: usize| ops.get(k).map_or(len, |op| op.entries[u].start);
    (0..period.repeats)
        .map(|i| start(period.start + i * period.p)..start(period.start + (i + 1) * period.p))
        .collect()
}

/// Per unit of `c`, its layers' stream ranges and its body, where its layers make one whose
/// every call expands to its layer ([`check`]); `None` for a unit that makes none, and for every
/// unit of a capture whose ops do not repeat.
pub(crate) fn plan(
    c: &Capture,
    table_entries: usize,
) -> Vec<Option<(Vec<Range<usize>>, UnitBody)>> {
    let Some(period) = residency::period(&residency::signatures(c)) else {
        return c.units.iter().map(|_| None).collect();
    };
    (0..c.units.len())
        .map(|u| {
            let ranges = bounds(&c.ops, c.units[u].stream.len(), &period, u);
            let layers: Vec<Layer> = ranges
                .iter()
                .map(|r| Layer::of(&c.units[u], r.clone()))
                .collect();
            let b = make(&layers, table_entries).ok()?;
            check(&layers, &b).ok()?;
            Some((ranges, b))
        })
        .collect()
}

/// What a unit's mover runs from a stored region, as the card runs it (`dm::op::CALL`,
/// `op::CALL_BODY`, `op::PAIR_CALL`): `words` the region's entries (entry 0 at GDDR offset `base`),
/// the call's stream at `offset`, `count` entries. Chunks of [`dm::TRACE_CHUNK_ENTRIES`] entries
/// on the stream's grid; a layer body's call runs the body's chunks with its table and offsets,
/// then the rest of its chunk read again; each pair's reader section then its writer section
/// follow its `PAIR_CALL`, the reader with the call's generation base and the body's arguments,
/// the writer with the arguments only. Chunk padding (`chunked`'s `WAIT`s) is dropped, and each
/// `PAIR_CALL` given zero offsets, so the result compares with a captured stream.
pub(crate) fn emulate(
    words: &[Entry],
    base: u64,
    offset: u32,
    count: u32,
    generation_base: u32,
    barrier_base: u32,
) -> Result<Vec<Entry>, String> {
    let at = |offset: u32, n: u32| -> Result<&[Entry], String> {
        let i = (offset as u64)
            .checked_sub(base)
            .filter(|d| d % dm::ENTRY_BYTES == 0)
            .ok_or_else(|| format!("offset {offset:#x} outside the region or not an entry's"))?
            / dm::ENTRY_BYTES;
        words
            .get(i as usize..i as usize + n as usize)
            .ok_or_else(|| format!("{n} entries at {offset:#x} past the region"))
    };
    struct Run<'a> {
        at: &'a dyn Fn(u32, u32) -> Result<&'a [Entry], String>,
        out: Vec<Entry>,
    }
    fn patch(
        e: &mut [Entry],
        i: usize,
        g: u32,
        b: u32,
        args: Option<&[Entry]>,
    ) -> Result<(), String> {
        let head = e[i][0];
        match head {
            op::KERNEL | op::LAUNCH | op::KERNEL_WAIT => e[i][1] = e[i][1].wrapping_add(g),
            op::BARRIER => e[i][1] = e[i][1].wrapping_add(b),
            _ => {}
        }
        let Some(t) = args else { return Ok(()) };
        if (head == op::READ || head == op::WRITE) && e[i][6] & dm::ARG_WORD != 0 {
            let k = (e[i][6] & !dm::ARG_WORD) as usize;
            let a = t.get(k).ok_or(format!("argument {k} past the table"))?;
            e[i][1..4].copy_from_slice(&a[1..4]);
            e[i][6] = 0;
        }
        for &r in record::references(head) {
            if e[i + r][0] == dm::ARG_REF {
                let k = e[i + r][1] as usize;
                let a = t
                    .get(k..k + 2)
                    .ok_or(format!("argument {k} past the table"))?;
                e[i + r] = a[0];
                e[i + r + 1] = a[1];
            }
        }
        Ok(())
    }
    /// A stream's chunks; `bodies` whether it may call one, `args` its body's table.
    #[allow(clippy::too_many_arguments)]
    fn stream(
        r: &mut Run<'_>,
        offset: u32,
        count: u32,
        g: u32,
        b: u32,
        bodies: bool,
        args: Option<&[Entry]>,
        writer: bool,
    ) -> Result<(), String> {
        let chunk = dm::TRACE_CHUNK_ENTRIES;
        let mut done = 0;
        while done < count {
            let n = (count - done).min(chunk - done % chunk);
            let mut e = (r.at)(offset + done * dm::ENTRY_BYTES as u32, n)?.to_vec();
            let mut i = 0;
            let mut body = None;
            while i < e.len() {
                let head = e[i][0];
                if head == op::CALL_BODY {
                    if !bodies {
                        return Err("a CALL_BODY where none may be".into());
                    }
                    body = Some((i, e[i]));
                    break;
                }
                let len = record::len(head).max(1);
                if i + len > e.len() {
                    return Err(format!("a record crosses the chunk at {}", done + i as u32));
                }
                patch(&mut e, i, g, b, args)?;
                i += len;
            }
            let mut k = 0;
            while k < i {
                let entry = e[k];
                let len = record::len(entry[0]).max(1);
                if entry[0] == op::PAIR_CALL && !writer {
                    let mut header = entry;
                    for w in [1, 2, 4, 5] {
                        header[w] = 0;
                    }
                    r.out.push(header);
                    stream(r, entry[2], entry[3], g, 0, false, args, false)?;
                    stream(r, entry[5], entry[6], 0, 0, false, args, true)?;
                } else if entry != [op::WAIT, 0, 0, 0, 0, 0, 0, 0] {
                    r.out.extend_from_slice(&e[k..k + len]);
                }
                k += len;
            }
            match body {
                Some((at, c)) => {
                    let [_, _, o, cnt, go, bo, to, tn] = c;
                    // What the mover refuses (`call_body`).
                    if cnt == 0 || tn > tt_isa::dataflow::ARG_TABLE_ENTRIES {
                        return Err(format!("a CALL_BODY of {cnt} entries and a table of {tn}"));
                    }
                    let table = (r.at)(to, tn)?.to_vec();
                    stream(
                        r,
                        o,
                        cnt,
                        g.wrapping_add(go),
                        b.wrapping_add(bo),
                        false,
                        Some(&table),
                        false,
                    )?;
                    done += at as u32 + 1;
                    if done % 2 != 0 {
                        return Err("the entry after a CALL_BODY is not 64-byte aligned".into());
                    }
                }
                None => done += n,
            }
        }
        Ok(())
    }
    let mut r = Run {
        at: &at,
        out: Vec::new(),
    };
    stream(
        &mut r,
        offset,
        count,
        generation_base,
        barrier_base,
        true,
        None,
        false,
    )?;
    Ok(r.out)
}

/// `uc`'s stream as [`emulate`] gives a stored one: each pair's sections after its `PAIR_CALL`,
/// chunk padding dropped.
pub(crate) fn flattened(uc: &UnitCapture) -> Vec<Entry> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < uc.stream.len() {
        let e = uc.stream[i];
        let len = record::len(e[0]).max(1);
        if e[0] == op::PAIR_CALL {
            out.push(e);
            if let Some(k) = uc.pair_calls.iter().position(|&at| at == i) {
                for part in &uc.pending_sections[2 * k..2 * k + 2] {
                    out.extend(
                        part.iter()
                            .copied()
                            .filter(|e| *e != [op::WAIT, 0, 0, 0, 0, 0, 0, 0]),
                    );
                }
            }
        } else if e != [op::WAIT, 0, 0, 0, 0, 0, 0, 0] {
            out.extend_from_slice(&uc.stream[i..i + len]);
        }
        i += len;
    }
    out
}

/// What the capture `c` would become as sub-traces, unit by unit.
pub(crate) struct Report {
    pub(crate) period: Option<Period>,
    /// Per unit: its layers' entries, and its body and calls or why it has none.
    pub(crate) units: Vec<(usize, Result<UnitBody, NoBody>)>,
    /// Calls whose expansion did not equal their layer, by unit: none, or a bug.
    pub(crate) mismatches: Vec<(usize, String)>,
}

/// [`make`] for every unit of `c`, each checked ([`check`]).
pub(crate) fn report(c: &Capture, table_entries: usize) -> Report {
    let period = residency::period(&residency::signatures(c));
    let mut units = Vec::new();
    let mut mismatches = Vec::new();
    for u in 0..c.units.len() {
        let Some(p) = period else { break };
        let layers: Vec<Layer> = bounds(&c.ops, c.units[u].stream.len(), &p, u)
            .into_iter()
            .map(|r| Layer::of(&c.units[u], r))
            .collect();
        let entries: usize = layers.iter().map(Layer::entries).sum();
        let made = make(&layers, table_entries);
        if let Ok(b) = &made {
            if let Err(e) = check(&layers, b) {
                mismatches.push((u, e));
            }
        }
        units.push((entries, made));
    }
    Report {
        period,
        units,
        mismatches,
    }
}

impl Report {
    /// Lines for the space report: how many units make a body, how many layers they call, the
    /// entries the layers hold and what a body and the calls' tables would, and why the other
    /// units make none.
    pub(crate) fn lines(&self) -> Vec<String> {
        let Some(p) = self.period else {
            return vec!["SUBTRACE no period".into()];
        };
        let mut out = Vec::new();
        let (mut layered, mut kept, mut tables, mut bodies, mut most_table) = (0, 0, 0, 0, 0);
        let mut why: BTreeMap<String, usize> = BTreeMap::new();
        let mut called_hist: BTreeMap<usize, usize> = BTreeMap::new();
        for (entries, made) in &self.units {
            layered += entries;
            match made {
                Ok(b) => {
                    bodies += b.body.entries();
                    *called_hist.entry(b.called()).or_default() += 1;
                    for c in b.calls.iter().flatten() {
                        tables += c.table.len();
                        most_table = most_table.max(c.table.len());
                    }
                    // A layer kept as captured keeps its entries (about the body's).
                    kept += b.calls.iter().filter(|c| c.is_none()).count() * b.body.entries();
                }
                Err(e) => {
                    kept += entries;
                    let key = match e {
                        NoBody::Unlike(Unlike::Word {
                            word, head, part, ..
                        }) => {
                            format!(
                                "op {head:#x} word {word} ({})",
                                if *part == 0 {
                                    "stream"
                                } else if part % 2 == 1 {
                                    "reader"
                                } else {
                                    "writer"
                                }
                            )
                        }
                        NoBody::Unlike(u) => format!("{u:?}")
                            .split_whitespace()
                            .next()
                            .unwrap_or("")
                            .to_string(),
                        NoBody::Fewer => "fewer than two layers".into(),
                        NoBody::Empty => "no entries in its layers".into(),
                        NoBody::Table(n) => format!("table of {}+ entries", n / 16 * 16),
                    };
                    *why.entry(key).or_default() += 1;
                }
            }
        }
        let made = self.units.iter().filter(|(_, m)| m.is_ok()).count();
        let from = self
            .units
            .iter()
            .filter_map(|(_, m)| m.as_ref().ok().map(|b| b.from))
            .max()
            .unwrap_or(0);
        out.push(format!(
            "SUBTRACE period {} ops x {} repeats from op {}: {made} of {} units make a body (from layer {from} at the \
             latest); layers {layered} entries -> bodies {bodies} + tables {tables} + kept {kept} ({:.1}x); largest table \
             {most_table} entries; {} mismatches",
            p.p,
            p.repeats,
            p.start,
            self.units.len(),
            layered as f64 / (bodies + tables + kept).max(1) as f64,
            self.mismatches.len(),
        ));
        for (called, units) in called_hist {
            out.push(format!(
                "SUBTRACE   {units} units call {called} of {} layers",
                p.repeats
            ));
        }
        let mut why: Vec<_> = why.into_iter().collect();
        why.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        for (key, n) in why.into_iter().take(12) {
            out.push(format!("SUBTRACE   no body on {n} units: {key}"));
        }
        for (u, e) in self.mismatches.iter().take(4) {
            out.push(format!("SUBTRACE   MISMATCH unit {u}: {e}"));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATHER_FLAGS: u32 = 0;

    /// A layer whose stream is a launch at generation `g`, a gather of tensors `a` and `b`, a
    /// read from `offset`, a barrier at `t` and a pair whose reader gathers `b` too.
    fn layer(g: u32, a: u32, b: u32, offset: u32, t: u32) -> Layer {
        let reference = |base: u32| [[1, 0, 0, 4, 0, 0, 0, 0], [base, 0, 0, 0, 0, 0, 0, 0]];
        let mut stream = vec![[op::LAUNCH, g, 0x14_4000, 64, 0, 0, 0, 0]];
        stream.push([record::GATHER, GATHER_FLAGS, 0x2_0000, 0x3_0000, 0, 1, 0, 1]);
        stream.extend(reference(a));
        stream.extend(reference(b));
        stream.push([op::READ, 0, 1, offset, 0x2_0000, 64, 0, 0]);
        stream.push([op::KERNEL_WAIT, g, 0, 0, 0, 0, 0, 0]);
        stream.push([op::BARRIER, t, 1, 2, 0, 0, 0, 0]);
        stream.push([op::PAIR_CALL, 0, 0, 6, 0, 0, 1, 2]);
        let mut reader = vec![[op::KERNEL, g + 1, 0x14_4000, 64, 0, 0, 0, 0]];
        reader.push([record::READ_RUN, 0, 1, 0x2_0000, 0, 1, 0, 0]);
        reader.extend(reference(b));
        let writer = vec![[op::WRITE, 0, 1, offset + 64, 0x2_0000, 64, 0, 0]];
        Layer {
            stream,
            sections: vec![reader, writer],
        }
    }

    fn layers() -> Vec<Layer> {
        (0..4)
            .map(|i| layer(10 + 3 * i, 0x100, 0x1000 + 0x100 * i, 0x40 * i, 7 + 2 * i))
            .collect()
    }

    #[test]
    fn layers_that_differ_by_tensors_and_counters_make_one_body_and_a_call_each() {
        let l = layers();
        let b = make(&l, 64).unwrap();
        assert_eq!((b.from, b.called()), (0, 4));
        // `b`, read in the stream and the reader, as one argument; the read's and the write's
        // offsets; not `a`.
        assert_eq!(b.calls[0].as_ref().unwrap().table.len(), 2 + 1 + 1);
        assert_eq!(
            b.body.sections[0][2], b.body.stream[4],
            "b's two reads share it"
        );
        assert_eq!(b.body.stream[4][0], dm::ARG_REF, "b is an argument");
        assert_eq!(b.body.stream[1..4], l[0].stream[1..4], "a is not");
        let c = b.calls[2].as_ref().unwrap();
        assert_eq!((c.generation_offset, c.barrier_offset), (6, 4));
        check(&l, &b).unwrap();
    }

    #[test]
    fn a_call_with_another_layer_s_table_does_not_expand_to_its_layer() {
        let l = layers();
        let b = make(&l, 64).unwrap();
        let mut wrong = b.calls[1].clone().unwrap();
        wrong.table = b.calls[2].as_ref().unwrap().table.clone();
        assert_ne!(expand(&b.body, &wrong).unwrap(), l[1]);
        let mut corrupt = b.clone();
        corrupt.calls[3].as_mut().unwrap().table[1][0] ^= 0x40;
        assert!(check(&l, &corrupt).is_err());
        let mut short = b.calls[1].clone().unwrap();
        short.table.pop();
        assert!(matches!(expand(&b.body, &short), Err(Bad::Argument(..))));
    }

    #[test]
    fn a_layer_whose_counters_do_not_move_together_or_whose_other_words_differ_stays() {
        let mut l = layers();
        // Layer 2's reader's kernel advances by one more than its stream's generations; layer 3's
        // launch names another program.
        l[2].sections[0][0][1] += 1;
        l[3].stream[0][2] += 32;
        let b = make(&l, 64).unwrap();
        assert!(b.calls[2].is_none() && b.calls[3].is_none());
        assert_eq!(
            diff(&l[0], &l[2]).err(),
            Some(Unlike::Counter { part: 1, at: 0 })
        );
        assert!(matches!(
            diff(&l[0], &l[3]).err(),
            Some(Unlike::Word {
                part: 0,
                at: 0,
                word: 2,
                head: op::LAUNCH
            })
        ));
        check(&l, &b).unwrap();
    }

    #[test]
    fn the_body_is_the_first_layer_equal_to_the_next_and_a_short_table_makes_none() {
        let mut l = layers();
        // Layer 0 is longer (a first descriptor poke).
        l[0].stream
            .insert(0, [op::POKE, 0x13_6000, 1, 0, 0, 0, 0, 0]);
        let b = make(&l, 64).unwrap();
        assert_eq!((b.from, b.called()), (1, 3));
        assert!(b.calls[0].is_none());
        check(&l, &b).unwrap();
        assert_eq!(make(&l, 3).err(), Some(NoBody::Table(4)));
        assert_eq!(make(&l[..2], 64).err(), Some(NoBody::Unlike(Unlike::Shape)));
    }

    #[test]
    fn a_unit_that_runs_none_of_the_repeats_makes_no_body() {
        let empty = vec![Layer::default(); 4];
        assert_eq!(make(&empty, 64).err(), Some(NoBody::Empty));
    }

    #[test]
    fn a_writer_section_takes_no_generation_offset() {
        let mut l = layers();
        for (i, layer) in l.iter_mut().enumerate() {
            layer.sections[1].push([op::KERNEL_WAIT, 5 + i as u32, 0, 0, 0, 0, 0, 0]);
        }
        assert!(matches!(
            make(&l, 64).err(),
            Some(NoBody::Unlike(Unlike::Word {
                part: 2,
                head: op::KERNEL_WAIT,
                ..
            }))
        ));
    }
}
