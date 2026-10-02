//! A device-side profile: what every unit's data mover and role runners did,
//! stamped by the tile's own cycle counter (`hardware-coverage.md` X3).
//!
//! The debug timestamper (`TensixTile/DebugTimestamper.md`) appends one
//! `{token, counter}` event per store, and every core of a tile stamps with
//! the same counter, so a tile's events order exactly: the mover's `KERNEL`
//! entry contains its roles' runs, a role's `RETIRED` follows its `PUSHED`.
//! Different tiles' counters are not synchronised. Each tile's counter is read
//! against the host's clock when profiling starts, and the export places its
//! events on the host's time line through that, to within the MMIO round trip
//! of the read (microseconds, never cycles). Cycles convert to time at a clock
//! *measured* over the profile (two counter reads and the host time between
//! them), not quoted.
//!
//! Silicon only: ttsim does not model the event stream (divergence row 54),
//! and a store to a register it does not model is fatal there, so
//! [`crate::session::Session::profile_start`] refuses on the simulator.

use std::fmt::Write as _;
use std::time::Instant;

use tt_device::trace::TraceEvent;
use tt_isa::mailbox::trace as ev;
use tt_isa::noc::{Noc0, NocCoord};

/// One unit's stream: every event drained from its tile since profiling
/// started, in order, and how its counter relates to the host's clock.
#[derive(Clone, Debug)]
pub struct UnitProfile {
    pub tile: NocCoord<Noc0>,
    pub events: Vec<TraceEvent>,
    /// The tile's counter when the host read it at `host`.
    pub counter_at_start: u64,
    pub host_at_start: Instant,
}

/// Every unit's stream, and the clock the counters ran at.
#[derive(Clone, Debug)]
pub struct DeviceProfile {
    pub units: Vec<UnitProfile>,
    /// Counter ticks per microsecond, measured on the first unit between
    /// `Session::profile_start` and `Session::profile_stop`.
    pub ticks_per_us: f64,
}

/// What an event says, decoded from its token.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Event {
    /// A role runner (`thread` 0..3): read its descriptor, pushed its last
    /// word, or saw the program retire.
    Role { thread: u32, what: RoleEvent },
    /// The data mover began or ended a list of `entries` entries.
    List { begin: bool, entries: u32 },
    /// The data mover began or ended a list entry or op record whose first
    /// word is `op` (`tt_isa::dm::op` or `tt_isa::dm::record`).
    Entry { begin: bool, op: u32 },
    /// The data mover posted `generation` to the roles (`kick`), or saw all
    /// three acknowledge it.
    Kick { done: bool, generation: u32 },
    /// The list just ended had requests wait `cycles` in all for room under
    /// the in-flight cap.
    Throttle { cycles: u32 },
    /// Anything else: a token no firmware of this build writes.
    Unknown(u32),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RoleEvent {
    /// A resident runner saw a new generation.
    Woke,
    Start,
    Pushed,
    Retired,
    /// A resident runner acknowledged its generation.
    Acked,
}

impl Event {
    pub fn decode(token: u32) -> Event {
        let (source, event) = ev::split(token);
        let detail = ev::detail(token);
        match (source, event) {
            (0..=2, ev::START) => Event::Role {
                thread: source,
                what: RoleEvent::Start,
            },
            (0..=2, ev::PUSHED) => Event::Role {
                thread: source,
                what: RoleEvent::Pushed,
            },
            (0..=2, ev::RETIRED) => Event::Role {
                thread: source,
                what: RoleEvent::Retired,
            },
            (0..=2, ev::WOKE) => Event::Role {
                thread: source,
                what: RoleEvent::Woke,
            },
            (0..=2, ev::ACKED) => Event::Role {
                thread: source,
                what: RoleEvent::Acked,
            },
            (ev::MOVER, ev::THROTTLE) => Event::Throttle { cycles: detail },
            (ev::MOVER, ev::KICK | ev::ROLES_DONE) => Event::Kick {
                done: event == ev::ROLES_DONE,
                generation: detail,
            },
            (ev::MOVER, ev::LIST_BEGIN | ev::LIST_END) => Event::List {
                begin: event == ev::LIST_BEGIN,
                entries: detail,
            },
            (ev::MOVER, ev::ENTRY_BEGIN | ev::ENTRY_END) => Event::Entry {
                begin: event == ev::ENTRY_BEGIN,
                op: detail,
            },
            _ => Event::Unknown(token),
        }
    }
}

/// The name a mover entry's first word goes by.
pub fn op_name(op: u32) -> &'static str {
    use tt_isa::dm::{op, record};
    match op {
        op::READ => "read",
        op::WRITE => "write",
        op::READ_TRANSPOSED => "read transposed",
        op::FILL => "fill",
        op::KERNEL => "kernel",
        op::WAIT => "wait",
        record::GATHER => "gather",
        record::SCATTER => "scatter",
        record::FILL_PAD => "fill pad",
        record::READ_RUN => "read run",
        record::WRITE_RUN => "write run",
        _ => "entry",
    }
}

/// A begin event without its end, an end without its begin, or a role whose
/// events are out of order: a stream that does not mean what the firmware
/// says it writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unbalanced {
    pub tile: NocCoord<Noc0>,
    pub at: usize,
    pub what: String,
}

impl std::fmt::Display for Unbalanced {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tile ({}, {}): event {}: {}",
            self.tile.x(),
            self.tile.y(),
            self.at,
            self.what
        )
    }
}

impl std::error::Error for Unbalanced {}

/// A span: something that began and ended on one unit, in counter ticks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub name: &'static str,
    /// `"mover"` or `"T0"`..`"T2"`.
    pub track: &'static str,
    pub begin: u64,
    pub end: u64,
}

impl UnitProfile {
    /// Pair each unit's begins with their ends: lists, entries, a kernel's
    /// kick (`KICK` to `ROLES_DONE`, on the mover's track) and role runs
    /// (`START` to `RETIRED`, `"program"`), with a resident role's wake
    /// (`WOKE` to `START`) and acknowledgement (`RETIRED` to `ACKED`) either
    /// side of it; and a list's waits for room under the in-flight cap
    /// (`"throttle"`, on the mover's track: their total, drawn as ending where
    /// the list did). Refuses a stream that does not nest.
    pub fn spans(&self) -> Result<Vec<Span>, Unbalanced> {
        let bad = |at: usize, what: String| Unbalanced {
            tile: self.tile,
            at,
            what,
        };
        let mut out = Vec::new();
        let mut list: Option<u64> = None;
        let mut entry: Option<(u32, u64)> = None;
        let mut role: [Option<(u64, bool)>; 3] = [None; 3];
        let mut kick: Option<(u32, u64)> = None;
        let mut woke: [Option<u64>; 3] = [None; 3];
        let mut retired: [Option<u64>; 3] = [None; 3];
        for (i, e) in self.events.iter().enumerate() {
            match Event::decode(e.token) {
                Event::List { begin: true, .. } => {
                    if list.replace(e.cycles).is_some() {
                        return Err(bad(i, "a list began inside a list".into()));
                    }
                }
                Event::List { begin: false, .. } => {
                    let b = list
                        .take()
                        .ok_or_else(|| bad(i, "a list ended that never began".into()))?;
                    if entry.is_some() {
                        return Err(bad(i, "a list ended inside an entry".into()));
                    }
                    out.push(Span {
                        name: "list",
                        track: "mover",
                        begin: b,
                        end: e.cycles,
                    });
                }
                Event::Entry { begin: true, op } => {
                    if list.is_none() || entry.replace((op, e.cycles)).is_some() {
                        return Err(bad(i, "an entry began outside a list or inside one".into()));
                    }
                }
                Event::Entry { begin: false, op } => match entry.take() {
                    Some((o, b)) if o == op => out.push(Span {
                        name: op_name(op),
                        track: "mover",
                        begin: b,
                        end: e.cycles,
                    }),
                    _ => return Err(bad(i, format!("entry {op:#x} ended without beginning"))),
                },
                Event::Kick {
                    done: false,
                    generation,
                } => {
                    if kick.replace((generation, e.cycles)).is_some() {
                        return Err(bad(i, "a kick inside a kick".into()));
                    }
                }
                Event::Kick {
                    done: true,
                    generation,
                } => match kick.take() {
                    Some((g, b)) if g == generation => out.push(Span {
                        name: "kick",
                        track: "mover",
                        begin: b,
                        end: e.cycles,
                    }),
                    _ => {
                        return Err(bad(
                            i,
                            format!("generation {generation} done without a kick"),
                        ))
                    }
                },
                Event::Throttle { cycles } => out.push(Span {
                    name: "throttle",
                    track: "mover",
                    begin: e.cycles.saturating_sub(cycles as u64),
                    end: e.cycles,
                }),
                Event::Role {
                    thread,
                    what: RoleEvent::Woke,
                } => {
                    let t = thread as usize;
                    if role[t].is_some() || woke[t].replace(e.cycles).is_some() {
                        return Err(bad(i, format!("T{thread} woke mid-run")));
                    }
                }
                Event::Role {
                    thread,
                    what: RoleEvent::Acked,
                } => {
                    let t = thread as usize;
                    let b = retired[t].take().ok_or_else(|| {
                        bad(i, format!("T{thread} acknowledged without retiring"))
                    })?;
                    out.push(Span {
                        name: "ack",
                        track: ["T0", "T1", "T2"][t],
                        begin: b,
                        end: e.cycles,
                    });
                }
                Event::Role { thread, what } => {
                    let t = thread as usize;
                    let r = &mut role[t];
                    match (what, *r) {
                        (RoleEvent::Start, None) => {
                            if let Some(w) = woke[t].take() {
                                out.push(Span {
                                    name: "wake",
                                    track: ["T0", "T1", "T2"][t],
                                    begin: w,
                                    end: e.cycles,
                                });
                            }
                            *r = Some((e.cycles, false));
                        }
                        (RoleEvent::Pushed, Some((b, false))) => *r = Some((b, true)),
                        (RoleEvent::Retired, Some((b, true))) => {
                            *r = None;
                            retired[t] = Some(e.cycles);
                            out.push(Span {
                                name: "program",
                                track: ["T0", "T1", "T2"][thread as usize],
                                begin: b,
                                end: e.cycles,
                            });
                        }
                        _ => return Err(bad(i, format!("T{thread} {what:?} out of order"))),
                    }
                }
                Event::Unknown(t) => return Err(bad(i, format!("unknown token {t:#x}"))),
            }
        }
        if list.is_some()
            || entry.is_some()
            || kick.is_some()
            || role.iter().any(Option::is_some)
            || woke.iter().any(Option::is_some)
        {
            return Err(bad(self.events.len(), "the stream ends mid-span".into()));
        }
        Ok(out)
    }
}

impl DeviceProfile {
    /// The profile as Chrome trace JSON (`chrome://tracing`, Perfetto): one
    /// process per tile, one thread per track, a complete event per span, in
    /// microseconds on the host's time line.
    pub fn to_chrome_trace(&self) -> Result<String, Unbalanced> {
        let Some(origin) = self.units.iter().map(|u| u.host_at_start).min() else {
            return Ok("{\"traceEvents\":[]}".into());
        };
        let mut s = String::from("{\"displayTimeUnit\":\"ns\",\"traceEvents\":[");
        let mut first = true;
        let mut push = |s: &mut String, item: String| {
            if !first {
                s.push(',');
            }
            first = false;
            s.push_str(&item);
        };
        for (pid, u) in self.units.iter().enumerate() {
            push(
                &mut s,
                format!(
                    "{{\"ph\":\"M\",\"pid\":{pid},\"name\":\"process_name\",\"args\":{{\"name\":\"tile ({}, {})\"}}}}",
                    u.tile.x(),
                    u.tile.y()
                ),
            );
            for (tid, track) in ["mover", "T0", "T1", "T2"].iter().enumerate() {
                push(
                    &mut s,
                    format!(
                        "{{\"ph\":\"M\",\"pid\":{pid},\"tid\":{tid},\"name\":\"thread_name\",\"args\":{{\"name\":\"{track}\"}}}}"
                    ),
                );
            }
            let base_us = u.host_at_start.duration_since(origin).as_secs_f64() * 1e6;
            let us = |c: u64| base_us + (c as f64 - u.counter_at_start as f64) / self.ticks_per_us;
            for sp in u.spans()? {
                let tid = ["mover", "T0", "T1", "T2"]
                    .iter()
                    .position(|t| *t == sp.track)
                    .unwrap_or(0);
                let mut item = String::new();
                let _ = write!(
                    item,
                    "{{\"ph\":\"X\",\"pid\":{pid},\"tid\":{tid},\"name\":\"{}\",\"ts\":{:.3},\"dur\":{:.3},\"args\":{{\"cycles\":{}}}}}",
                    sp.name,
                    us(sp.begin),
                    (sp.end - sp.begin) as f64 / self.ticks_per_us,
                    sp.end - sp.begin
                );
                push(&mut s, item);
            }
        }
        s.push_str("]}");
        Ok(s)
    }

    /// Total ticks spent in spans called `name` on `track`, over every unit.
    pub fn ticks_in(&self, track: &str, name: &str) -> Result<u64, Unbalanced> {
        let mut t = 0;
        for u in &self.units {
            t += u
                .spans()?
                .iter()
                .filter(|s| s.track == track && s.name == name)
                .map(|s| s.end - s.begin)
                .sum::<u64>();
        }
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tt_isa::dm::{op, record};

    fn e(source: u32, event: u32, detail: u32, cycles: u64) -> TraceEvent {
        TraceEvent {
            token: ev::token_with(source, event, detail),
            cycles,
        }
    }

    fn unit(events: Vec<TraceEvent>) -> UnitProfile {
        UnitProfile {
            tile: NocCoord::new(1, 2).unwrap(),
            events,
            counter_at_start: 0,
            host_at_start: Instant::now(),
        }
    }

    fn kernel_list() -> Vec<TraceEvent> {
        let m = ev::MOVER;
        let mut v = vec![
            e(m, ev::LIST_BEGIN, 3, 10),
            e(m, ev::ENTRY_BEGIN, record::GATHER, 11),
            e(m, ev::ENTRY_END, record::GATHER, 20),
            e(m, ev::ENTRY_BEGIN, op::KERNEL, 21),
        ];
        for t in 0..3 {
            v.push(e(t, ev::START, 0, 30 + t as u64));
        }
        for t in 0..3 {
            v.push(e(t, ev::PUSHED, 0, 40 + t as u64));
            v.push(e(t, ev::RETIRED, 0, 50 + t as u64));
        }
        v.extend([
            e(m, ev::ENTRY_END, op::KERNEL, 60),
            e(m, ev::ENTRY_BEGIN, record::SCATTER, 61),
            e(m, ev::ENTRY_END, record::SCATTER, 70),
            e(m, ev::LIST_END, 3, 80),
        ]);
        v
    }

    #[test]
    fn a_kernel_list_pairs_into_spans() {
        let spans = unit(kernel_list()).spans().unwrap();
        let names: Vec<_> = spans.iter().map(|s| (s.track, s.name)).collect();
        assert!(names.contains(&("mover", "gather")));
        assert!(names.contains(&("mover", "kernel")));
        assert!(names.contains(&("T2", "program")));
        assert_eq!(spans.last().unwrap().name, "list");
        let k = spans.iter().find(|s| s.name == "kernel").unwrap();
        assert_eq!((k.begin, k.end), (21, 60));
    }

    /// A resident kernel's whole hand-off: the mover's kick, each role
    /// waking, running and acknowledging, the mover seeing the last ack.
    #[test]
    fn a_resident_kernel_pairs_its_hand_offs() {
        let m = ev::MOVER;
        let mut v = vec![
            e(m, ev::LIST_BEGIN, 1, 10),
            e(m, ev::ENTRY_BEGIN, op::KERNEL, 11),
            e(m, ev::KICK, 7, 12),
        ];
        for t in 0..3 {
            v.push(e(t, ev::WOKE, 0, 20 + t as u64));
            v.push(e(t, ev::START, 0, 30 + t as u64));
            v.push(e(t, ev::PUSHED, 0, 40 + t as u64));
            v.push(e(t, ev::RETIRED, 0, 50 + t as u64));
            v.push(e(t, ev::ACKED, 0, 55 + t as u64));
        }
        v.extend([
            e(m, ev::ROLES_DONE, 7, 60),
            e(m, ev::ENTRY_END, op::KERNEL, 61),
            e(m, ev::LIST_END, 1, 62),
        ]);
        let spans = unit(v).spans().unwrap();
        let find = |track, name| {
            spans
                .iter()
                .find(|s| s.track == track && s.name == name)
                .unwrap()
        };
        assert_eq!(
            (find("mover", "kick").begin, find("mover", "kick").end),
            (12, 60)
        );
        assert_eq!((find("T1", "wake").begin, find("T1", "wake").end), (21, 31));
        assert_eq!(
            (find("T2", "program").begin, find("T2", "program").end),
            (32, 52)
        );
        assert_eq!((find("T0", "ack").begin, find("T0", "ack").end), (50, 55));
    }

    #[test]
    fn a_lists_throttle_is_a_span_ending_where_it_was_stamped() {
        let m = ev::MOVER;
        let v = vec![
            e(m, ev::LIST_BEGIN, 1, 10),
            e(m, ev::ENTRY_BEGIN, op::READ, 11),
            e(m, ev::ENTRY_END, op::READ, 95),
            e(m, ev::LIST_END, 1, 100),
            e(m, ev::THROTTLE, 40, 101),
        ];
        let spans = unit(v).spans().unwrap();
        let t = spans.iter().find(|s| s.name == "throttle").unwrap();
        assert_eq!((t.track, t.begin, t.end), ("mover", 61, 101));
    }

    #[test]
    fn a_stream_that_does_not_nest_is_refused() {
        let mut v = kernel_list();
        v.remove(2); // the gather's end
        assert!(unit(v).spans().is_err());
        let mut v = kernel_list();
        v.pop(); // the list's end
        assert!(unit(v).spans().is_err());
        let mut v = kernel_list();
        v.swap(7, 8); // T0 retires before it has pushed
        assert!(unit(v).spans().is_err());
    }

    #[test]
    fn the_chrome_trace_has_one_complete_event_per_span() {
        let p = DeviceProfile {
            units: vec![unit(kernel_list())],
            ticks_per_us: 1000.0,
        };
        let json = p.to_chrome_trace().unwrap();
        let spans = unit(kernel_list()).spans().unwrap().len();
        assert_eq!(json.matches("\"ph\":\"X\"").count(), spans);
        assert_eq!(json.matches('{').count(), json.matches('}').count());
        assert!(json.starts_with('{') && json.ends_with("]}"));
        assert_eq!(p.ticks_in("mover", "kernel").unwrap(), 39);
    }
}
