//! Channels: the edges of a dataflow graph (L2 of `docs/learnings/dataflow-compiler.md`, §5 and §7).
//!
//! A channel has one producer and any number of consumers, a depth (the pages it holds at once,
//! which is the credits its reverse edge starts with) and a page. Its kind says how its data and
//! its credits move:
//!
//! - [`Kind::Local`]: within one tile, the in-core streams of [`tt_isa::dataflow`]. Its counters
//!   are the stream's fixed words, and a credit is a `CB` entry (`dm::op::CB`) of the role that owns
//!   the action.
//! - [`Kind::Remote`]: from one tile's L1 to another's. The producer writes a page into the
//!   consumer's L1 and then credits it with a NoC atomic increment, sent after the page's write
//!   acknowledgements (T7); the consumer waits on its own word.
//! - [`Kind::Broadcast`]: one page multicast to a rectangle of consumers, each credited.
//! - [`Kind::Gddr`]: a page through GDDR, credited to the tile that reads it back.
//! - [`Kind::Roles`]: within one tile, from one of its roles to another through `Dst` or L1,
//!   credited by the body's own semaphores: it has no words of its own.
//!
//! The local kind is what every op emits today; this module is where its words are made
//! ([`Channel::credit`]), so a stream's credits are written in one place. Each op's channels are
//! checked as a [`Graph`] before its jobs are lowered (`tile_op::channels`).

use tt_isa::dataflow::{Action, Endpoint, Stream};
use tt_isa::format::TileFormat;

/// What one token of a channel carries.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Page {
    /// One compute-sized batch: the operands or results of one kernel launch, whatever their
    /// tiles. The in-core streams count these today.
    Batch,
    /// `tiles` tiles of `format`, each in a slot of [`TileFormat::slot_bytes`].
    Tiles { format: TileFormat, tiles: u16 },
}

impl Page {
    /// The L1 bytes one page takes, if its size is known.
    pub fn bytes(self) -> Option<u64> {
        match self {
            Page::Batch => None,
            Page::Tiles { format, tiles } => Some(format.slot_bytes() * tiles as u64),
        }
    }
}

/// How a channel's data and credits move.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Kind {
    /// Within one tile: one of the in-core streams.
    Local(Stream),
    /// From one tile's L1 to another's, credited by a NoC atomic increment.
    Remote,
    /// Multicast to a rectangle of tiles, each consumer credited.
    Broadcast,
    /// Through GDDR, credited to the reader.
    Gddr,
    /// Within one tile, between two of its roles (an unpacked batch to the packer, a partial
    /// packed for the next chunk's unpack), credited by the body's own semaphores.
    Roles,
}

/// One end of a channel: the role on a unit that produces or consumes it, and how many tokens it
/// moves each time it fires.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct End {
    /// The session unit (an index into its tiles).
    pub unit: usize,
    pub endpoint: Endpoint,
    /// Tokens produced or consumed a firing.
    pub tokens: u32,
}

/// A bounded channel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Channel {
    pub producer: End,
    pub consumers: Vec<End>,
    /// Pages held at once: the credits the reverse edge starts with.
    pub depth: u16,
    /// Pages held before the first firing (at most `depth`): a chain's first link, which takes
    /// none from the last (synchronous dataflow's delay tokens).
    pub initial: u16,
    pub page: Page,
    pub kind: Kind,
}

/// The most pages a channel holds at once: the in-core streams' counters are 16 bits, compared
/// by signed difference.
pub const MAX_DEPTH: u16 = 0x7FFF;

impl Channel {
    /// The in-core stream `stream` on `unit`, `depth` batches deep, one batch a firing.
    pub fn local(unit: usize, stream: Stream, depth: u16) -> Self {
        assert!(
            (1..=MAX_DEPTH).contains(&depth),
            "a {stream:?} stream of depth {depth}"
        );
        let end = |endpoint| End {
            unit,
            endpoint,
            tokens: 1,
        };
        Channel {
            producer: end(stream.producer()),
            consumers: vec![end(stream.consumer())],
            depth,
            initial: 0,
            page: Page::Batch,
            kind: Kind::Local(stream),
        }
    }

    /// The mover entry that performs `action` on a local channel: `[CB, stream, action, depth]`.
    /// Which role may run it is the stream's ([`tt_isa::dataflow::Channel::permits`]).
    ///
    /// # Panics
    ///
    /// On a channel that is not [`Kind::Local`]: its credits are not `CB` entries.
    pub fn credit(&self, action: Action) -> [u32; 8] {
        let Kind::Local(stream) = self.kind else {
            panic!("a {:?} channel's credits are not CB entries", self.kind)
        };
        [
            tt_isa::dm::op::CB,
            stream.word(),
            action.word(),
            self.depth as u32,
            0,
            0,
            0,
            0,
        ]
    }

    /// The local counters' address on the tile: received, then consumed.
    pub fn counters(&self) -> Option<u64> {
        match self.kind {
            Kind::Local(stream) => Some(stream.address()),
            _ => None,
        }
    }
}

/// Where a cross-core channel's pages wait for their consumers: a ring of `depth` slots.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Slots {
    /// In each consumer's L1, at the same address on every one.
    L1 { base: u32 },
    /// In GDDR: `channel`'s bytes from `offset`, written through `port`.
    Gddr {
        channel: u32,
        port: u32,
        offset: u32,
    },
}

/// A cross-core channel lowered to mover entries: a [`Kind::Remote`] channel (one consumer and
/// [`Slots::L1`]), a [`Kind::Broadcast`] one (several consumers, a rectangle, [`Slots::L1`]) or a
/// [`Kind::Gddr`] one ([`Slots::Gddr`]).
///
/// The producer's NC writes page `k` into slot `k % depth` and then credits every consumer's
/// `received` word ([`tt_isa::dm::op::CREDIT`], which waits for the writes' acknowledgements
/// first). A consumer waits for `received` to reach `k + 1` ([`tt_isa::dm::op::WAIT_CREDIT`]), uses
/// the page, and credits the producer's `freed` word. Before it writes page `k >= depth` over page
/// `k - depth`, the producer waits for `freed` to reach `(k + 1 - depth) x consumers`. Every count
/// starts at zero, written by the host before either side runs ([`prepare`]), and none is reset
/// by the card.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ring {
    /// The producer's tile, as the session names it.
    pub producer: (u8, u8),
    /// The consumers' tiles: one, or every tile of a rectangle.
    pub consumers: Vec<(u8, u8)>,
    pub depth: u16,
    /// Bytes a page, a multiple of 16.
    pub page_bytes: u32,
    pub slots: Slots,
    /// The credit word on every consumer that counts the pages it has been sent.
    pub received: u32,
    /// The credit word on the producer that counts the pages consumers have freed.
    pub freed: u32,
}

/// Why a channel or its lowering is refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChannelError {
    /// A ring that cannot be lowered: no consumers, depth 0, a page that is not whole 16-byte
    /// units, broadcast consumers that are not one rectangle.
    Invalid(String),
    /// A producer write that lands outside the slot of the page it is writing.
    OutsideSlot { page: u32, entry: usize },
    /// A write into the slot of `page` before its consumers have freed the page the slot held.
    NotFreed { page: u32, entry: usize },
    /// A write into the slot of `page` after a consumer has already been credited for it: the
    /// consumer may be reading what is being written.
    WriteAfterCredit { page: u32, entry: usize },
    /// A credit for `page` before all of its bytes were written.
    Unwritten {
        page: u32,
        written: u64,
        bytes: u64,
        entry: usize,
    },
    /// A graph whose rates admit no repeating schedule: channel `channel` (an index into the
    /// graph's) cannot carry what its producer makes as often as its consumer takes it.
    Unbalanced { channel: usize },
    /// A graph that stops before one iteration completes: every process in `blocked` waits for
    /// data or room no other process can give.
    Deadlock { blocked: Vec<(usize, Endpoint)> },
}

impl std::fmt::Display for ChannelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelError::Invalid(why) => f.write_str(why),
            ChannelError::OutsideSlot { page, entry } => {
                write!(f, "entry {entry} writes page {page} outside its slot")
            }
            ChannelError::NotFreed { page, entry } => write!(
                f,
                "entry {entry} writes page {page} before its slot's last page is freed"
            ),
            ChannelError::WriteAfterCredit { page, entry } => write!(
                f,
                "entry {entry} writes page {page} after its consumers were credited for it"
            ),
            ChannelError::Unwritten {
                page,
                written,
                bytes,
                entry,
            } => write!(
                f,
                "entry {entry} credits page {page} with {written} of its {bytes} bytes written"
            ),
            ChannelError::Unbalanced { channel } => {
                write!(f, "channel {channel}'s rates admit no repeating schedule")
            }
            ChannelError::Deadlock { blocked } => write!(
                f,
                "the graph deadlocks: {blocked:?} each wait for data or room"
            ),
        }
    }
}

impl std::error::Error for ChannelError {}

const MAX_REQUEST: u32 = tt_isa::noc::niu::MAX_REQUEST_BYTES;

impl Ring {
    fn validate(&self) -> Result<(), ChannelError> {
        let invalid = |why: &str| Err(ChannelError::Invalid(why.into()));
        if self.consumers.is_empty() || self.depth == 0 || self.depth > MAX_DEPTH {
            return invalid("a ring needs consumers and a depth of 1 to 0x7FFF");
        }
        if self.page_bytes == 0 || self.page_bytes % 16 != 0 {
            return invalid("a page is whole 16-byte units");
        }
        if self.received % 4 != 0 || self.freed % 4 != 0 {
            return invalid("a credit word is a word");
        }
        if self.consumers.len() > 1 {
            if matches!(self.slots, Slots::Gddr { .. }) {
                return invalid("a GDDR ring has one consumer");
            }
            let (start, end) = self.rectangle();
            let tiles = (end.0 - start.0 + 1) as usize * (end.1 - start.1 + 1) as usize;
            let mut distinct = self.consumers.clone();
            distinct.sort_unstable();
            distinct.dedup();
            if distinct.len() != self.consumers.len() || tiles != distinct.len() {
                return invalid("broadcast consumers are every tile of one rectangle, once");
            }
            if self.consumers.contains(&self.producer) {
                return invalid("a multicast never reaches its sender");
            }
        }
        Ok(())
    }

    /// The consumers' bounding rectangle.
    fn rectangle(&self) -> ((u8, u8), (u8, u8)) {
        let xs = self.consumers.iter().map(|c| c.0);
        let ys = self.consumers.iter().map(|c| c.1);
        (
            (xs.clone().min().unwrap(), ys.clone().min().unwrap()),
            (xs.max().unwrap(), ys.max().unwrap()),
        )
    }

    /// Slot `page % depth`'s first byte: an L1 address, or a GDDR offset.
    pub fn slot(&self, page: u32) -> u32 {
        let base = match self.slots {
            Slots::L1 { base } => base,
            Slots::Gddr { offset, .. } => offset,
        };
        base + (page % self.depth as u32) * self.page_bytes
    }

    fn credit_entry(at: (u8, u8), word: u32, value: u32) -> [u32; 8] {
        [
            tt_isa::dm::op::CREDIT,
            at.0 as u32,
            at.1 as u32,
            word,
            value,
            0,
            0,
            0,
        ]
    }

    fn wait_entry(word: u32, target: u32) -> [u32; 8] {
        [tt_isa::dm::op::WAIT_CREDIT, word, target, 0, 0, 0, 0, 0]
    }

    /// The producer's (NC's) entries that send page `page` from `from` in its own L1: the wait for
    /// its slot to be freed (from the ring's second lap), the page's writes, then a credit to each
    /// consumer.
    pub fn send(&self, page: u32, from: u32) -> Result<Vec<[u32; 8]>, ChannelError> {
        use tt_isa::dm::op;
        self.validate()?;
        let mut out = Vec::new();
        let consumers = self.consumers.len() as u32;
        if page >= self.depth as u32 {
            out.push(Self::wait_entry(
                self.freed,
                (page + 1 - self.depth as u32) * consumers,
            ));
        }
        let slot = self.slot(page);
        let mut done = 0;
        while done < self.page_bytes {
            let len = (self.page_bytes - done).min(MAX_REQUEST);
            let (to, from) = (slot + done, from + done);
            out.push(match (self.slots, self.consumers.as_slice()) {
                (Slots::Gddr { channel, port, .. }, _) => {
                    [op::WRITE, channel, port, to, from, len, 0, 0]
                }
                (Slots::L1 { .. }, [one]) => [
                    op::WRITE_L1,
                    one.0 as u32,
                    one.1 as u32,
                    to,
                    from,
                    len,
                    0,
                    0,
                ],
                (Slots::L1 { .. }, _) => {
                    let (start, end) = self.rectangle();
                    let corners = tt_isa::dm::mcast_corners(start, end);
                    [op::WRITE_L1_MCAST, corners, 0, to, from, len, 0, 0]
                }
            });
            done += len;
        }
        for &consumer in &self.consumers {
            out.push(Self::credit_entry(consumer, self.received, 1));
        }
        Ok(out)
    }

    /// A consumer's wait for page `page`.
    pub fn receive(&self, page: u32) -> [u32; 8] {
        Self::wait_entry(self.received, page + 1)
    }

    /// A GDDR ring's consumer (B): the wait for page `page`, then its reads into `to`.
    pub fn fetch(&self, page: u32, to: u32) -> Result<Vec<[u32; 8]>, ChannelError> {
        self.validate()?;
        let Slots::Gddr { channel, port, .. } = self.slots else {
            return Err(ChannelError::Invalid(
                "only a GDDR ring's page is fetched".into(),
            ));
        };
        let mut out = vec![self.receive(page)];
        let slot = self.slot(page);
        let mut done = 0;
        while done < self.page_bytes {
            let len = (self.page_bytes - done).min(MAX_REQUEST);
            out.push([
                tt_isa::dm::op::READ,
                channel,
                port,
                slot + done,
                to + done,
                len,
                0,
                0,
            ]);
            done += len;
        }
        Ok(out)
    }

    /// A consumer's credit to the producer once it has finished with a page.
    pub fn free(&self) -> [u32; 8] {
        Self::credit_entry(self.producer, self.freed, 1)
    }

    /// Check a producer's entries against the ring's protocol, page by page from page 0: each
    /// page's writes land in its slot, after the slot's last page was freed and before any consumer
    /// is credited for it, and every byte of a page is written before its first credit. Entries
    /// that neither write the ring nor credit or wait on its words are other work and pass.
    pub fn check_send(&self, entries: &[[u32; 8]]) -> Result<(), ChannelError> {
        use tt_isa::dm::op;
        self.validate()?;
        let consumers = self.consumers.len() as u32;
        let depth = self.depth as u32;
        let ring = {
            let base = self.slot(0) as u64;
            base..base + depth as u64 * self.page_bytes as u64
        };
        // The page being written: the first not yet credited to every consumer.
        let mut page = 0u32;
        // Consumers credited for `page` so far.
        let mut credited = 0u32;
        let mut written = 0u64;
        // Pages freed by every consumer, as far as the producer has waited.
        let mut freed = 0u32;
        for (entry, e) in entries.iter().enumerate() {
            let target = match (e[0], self.slots) {
                (op::WRITE, Slots::Gddr { channel, .. }) if e[1] == channel => {
                    Some((e[3] as u64, e[5] as u64))
                }
                (op::WRITE_L1, Slots::L1 { .. })
                    if self.consumers.contains(&(e[1] as u8, e[2] as u8)) =>
                {
                    Some((e[3] as u64, e[5] as u64))
                }
                (op::WRITE_L1_MCAST, Slots::L1 { .. }) => Some((e[3] as u64, e[5] as u64)),
                _ => None,
            };
            if let Some((at, len)) = target {
                if at + len <= ring.start || at >= ring.end {
                    continue;
                }
                let bytes = self.page_bytes as u64;
                let index = ((at.max(ring.start) - ring.start) / bytes) as u32;
                let slot = ring.start + index as u64 * bytes;
                if at < slot || at + len > slot + bytes {
                    return Err(ChannelError::OutsideSlot { page, entry });
                }
                if index != page % depth {
                    // The slot of an earlier page, credited already: unless its consumers freed
                    // it, they may be reading it.
                    let back = (page % depth + depth - index) % depth;
                    return Err(match page.checked_sub(back) {
                        Some(earlier) if earlier >= freed => ChannelError::WriteAfterCredit {
                            page: earlier,
                            entry,
                        },
                        _ => ChannelError::OutsideSlot { page, entry },
                    });
                }
                if page >= depth && freed < page + 1 - depth {
                    return Err(ChannelError::NotFreed { page, entry });
                }
                if credited > 0 {
                    return Err(ChannelError::WriteAfterCredit { page, entry });
                }
                written += len;
                continue;
            }
            match e[0] {
                op::WAIT_CREDIT if e[1] == self.freed => {
                    freed = freed.max(e[2] / consumers);
                }
                op::CREDIT
                    if e[3] == self.received
                        && self.consumers.contains(&(e[1] as u8, e[2] as u8)) =>
                {
                    if written < self.page_bytes as u64 {
                        return Err(ChannelError::Unwritten {
                            page,
                            written,
                            bytes: self.page_bytes as u64,
                            entry,
                        });
                    }
                    credited += e[4];
                    if credited >= consumers {
                        page += 1;
                        credited = 0;
                        written = 0;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Channels as a synchronous dataflow graph, checked before anything is emitted
/// (`docs/learnings/dataflow-compiler.md` §7). A process is a role on a unit (an [`End`]'s `unit`
/// and `endpoint`); each consumer of a channel is an edge from its producer, carrying the
/// producer's tokens a firing in and the consumer's out, with room for `depth` tokens: its reverse
/// edge starts with `depth` credits, less the `initial` pages the forward edge starts with.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    channels: Vec<Channel>,
}

/// How often each process fires in one iteration of a balanced graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Schedule {
    pub processes: Vec<(usize, Endpoint)>,
    pub firings: Vec<u64>,
}

impl Graph {
    pub fn new(channels: Vec<Channel>) -> Self {
        Graph { channels }
    }

    /// The graph's repetition vector, once its rates balance (Lee and Messerschmitt 1987) and one
    /// iteration runs to the end from each channel's initial pages, with its depth as its room.
    pub fn check(&self) -> Result<Schedule, ChannelError> {
        let key = |e: &End| (e.unit, e.endpoint);
        // Linear in the graph throughout: a trace's credits make a process of every unit's every
        // run (Qwen3.5's prefill chunk: hundreds of thousands).
        let mut processes: Vec<(usize, Endpoint)> = Vec::new();
        let mut seen: std::collections::HashMap<(usize, u8), usize> = Default::default();
        let mut index = |p: (usize, Endpoint)| {
            *seen.entry((p.0, p.1 as u8)).or_insert_with(|| {
                processes.push(p);
                processes.len() - 1
            })
        };
        // (channel, from, to, produced a firing, consumed a firing, depth), and the initial pages.
        let mut edges = Vec::new();
        let mut initial = Vec::new();
        for (c, channel) in self.channels.iter().enumerate() {
            if channel.depth == 0 || channel.producer.tokens == 0 {
                return Err(ChannelError::Invalid(format!(
                    "channel {c} holds nothing or carries nothing"
                )));
            }
            if channel.initial > channel.depth {
                return Err(ChannelError::Invalid(format!(
                    "channel {c} starts with {} pages and holds {}",
                    channel.initial, channel.depth
                )));
            }
            let from = index(key(&channel.producer));
            for consumer in &channel.consumers {
                if consumer.tokens == 0 {
                    return Err(ChannelError::Invalid(format!(
                        "channel {c}'s consumer takes nothing"
                    )));
                }
                let to = index(key(consumer));
                edges.push((
                    c,
                    from,
                    to,
                    channel.producer.tokens as u64,
                    consumer.tokens as u64,
                    channel.depth as u64,
                ));
                initial.push(channel.initial as u64);
            }
        }
        // Each process's edges in, out, and either way (in edge order, a self-loop once).
        let mut ins = vec![Vec::new(); processes.len()];
        let mut outs = vec![Vec::new(); processes.len()];
        let mut adjacent = vec![Vec::new(); processes.len()];
        for (e, &(_, from, to, ..)) in edges.iter().enumerate() {
            outs[from].push(e);
            ins[to].push(e);
            adjacent[from].push(e);
            if to != from {
                adjacent[to].push(e);
            }
        }
        let firings = repetitions(&adjacent, &edges)?;
        // One iteration, self-timed: any process with its inputs' tokens and its outputs' room
        // fires, until each has fired its count or none can. Each edge has one producer and one
        // consumer, so a firing never disables another process and the order cannot change the
        // outcome: a process is looked at again only when a neighbour's firing gave it tokens or
        // room.
        let mut tokens = initial;
        let mut fired = vec![0u64; processes.len()];
        let mut queued = vec![true; processes.len()];
        let mut queue: std::collections::VecDeque<usize> = (0..processes.len()).collect();
        while let Some(p) = queue.pop_front() {
            queued[p] = false;
            while fired[p] < firings[p]
                && ins[p].iter().all(|&e| tokens[e] >= edges[e].4)
                && outs[p]
                    .iter()
                    .all(|&e| edges[e].5 - tokens[e] >= edges[e].3)
            {
                for &e in &ins[p] {
                    tokens[e] -= edges[e].4;
                }
                for &e in &outs[p] {
                    tokens[e] += edges[e].3;
                }
                fired[p] += 1;
                let woken = outs[p].iter().map(|&e| edges[e].2);
                for q in woken.chain(ins[p].iter().map(|&e| edges[e].1)) {
                    if q != p && !queued[q] {
                        queued[q] = true;
                        queue.push_back(q);
                    }
                }
            }
        }
        if fired == firings {
            return Ok(Schedule { processes, firings });
        }
        let blocked = (0..processes.len())
            .filter(|&p| fired[p] < firings[p])
            .map(|p| processes[p])
            .collect();
        Err(ChannelError::Deadlock { blocked })
    }
}

/// The least positive firings `q` with `q[from] x made == q[to] x taken` on every edge.
fn repetitions(
    adjacent: &[Vec<usize>],
    edges: &[(usize, usize, usize, u64, u64, u64)],
) -> Result<Vec<u64>, ChannelError> {
    let processes = adjacent.len();
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    // Each process's rate as a fraction (numerator, denominator), spread from a first one in each
    // connected part.
    let mut rate: Vec<Option<(u64, u64)>> = vec![None; processes];
    for start in 0..processes {
        if rate[start].is_some() {
            continue;
        }
        rate[start] = Some((1, 1));
        let mut queue = vec![start];
        while let Some(p) = queue.pop() {
            let (n, d) = rate[p].unwrap();
            for &(c, from, to, made, taken, _) in adjacent[p].iter().map(|&e| &edges[e]) {
                // q[to] = q[from] x made / taken, and the reverse.
                let (other, num, den) = if from == p {
                    (to, n * made, d * taken)
                } else if to == p {
                    (from, n * taken, d * made)
                } else {
                    continue;
                };
                let g = gcd(num, den);
                let r = (num / g, den / g);
                match rate[other] {
                    None => {
                        rate[other] = Some(r);
                        queue.push(other);
                    }
                    Some(q) if q.0 * r.1 == r.0 * q.1 => {}
                    Some(_) => return Err(ChannelError::Unbalanced { channel: c }),
                }
            }
        }
    }
    let rate: Vec<(u64, u64)> = rate.into_iter().map(Option::unwrap).collect();
    let lcm = rate.iter().fold(1, |l, &(_, d)| l / gcd(l, d) * d);
    let whole: Vec<u64> = rate.iter().map(|&(n, d)| n * (lcm / d)).collect();
    let g = whole.iter().fold(0, |g, &q| gcd(g, q));
    Ok(whole.into_iter().map(|q| q / g.max(1)).collect())
}

/// A mover held at a credit wait ([`tt_isa::dm::op::WAIT_CREDIT`]): the word's count and the target
/// it waits for. Credits that never come stall a consumer; they never let it race.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Stall {
    pub word: u32,
    pub have: u32,
    pub want: u32,
}

/// Whether `mover` on `tile` is held at the credit wait `wait` (the entry the host queued), as its
/// wait reason ([`tt_isa::dataflow::WAIT_REASON`]) and the word say: `Some` with the count it has.
pub fn stalled<T: tt_device::Transport, N: tt_isa::noc::NocId>(
    d: &mut tt_device::Device<T>,
    w: &tt_device::Window,
    tile: tt_isa::noc::NocCoord<N>,
    mover: tt_isa::dm::Mover,
    wait: [u32; 8],
) -> Result<Option<Stall>, tt_device::TransportError> {
    use tt_isa::dataflow::{CREDIT_WAIT, WAIT_REASON};
    assert_eq!(wait[0], tt_isa::dm::op::WAIT_CREDIT, "not a credit wait");
    let actor = if mover == tt_isa::dm::Mover::B { 0 } else { 4 };
    let reason = d.read32(w, tile, WAIT_REASON + actor * 4)?;
    let have = d.read32(w, tile, wait[1] as u64)?;
    let reached = (have.wrapping_sub(wait[2]) as i32) >= 0;
    Ok((reason == CREDIT_WAIT && !reached).then_some(Stall {
        word: wait[1],
        have,
        want: wait[2],
    }))
}

/// Make `tile` ready for a cross-core channel's entries: each of its credit `words` zero, and its
/// abort word ([`tt_isa::dataflow::ABORT`]) clear, since a credit wait outside a packet reads it and
/// silicon keeps L1 between programs and processes. Before either side of the channel runs.
pub fn prepare<T: tt_device::Transport, N: tt_isa::noc::NocId>(
    d: &mut tt_device::Device<T>,
    w: &tt_device::Window,
    tile: tt_isa::noc::NocCoord<N>,
    words: &[u32],
) -> Result<(), tt_device::TransportError> {
    for &word in words {
        d.write32(w, tile, word as u64, 0)?;
    }
    d.write32(w, tile, tt_isa::dataflow::ABORT, 0)
}

/// Cancel what waits on `tile`'s credits and streams: its abort word ([`tt_isa::dataflow::ABORT`]),
/// which every credit poll reads. The waiting list fails with `code`; a packet that starts later
/// clears the word.
pub fn cancel<T: tt_device::Transport, N: tt_isa::noc::NocId>(
    d: &mut tt_device::Device<T>,
    w: &tt_device::Window,
    tile: tt_isa::noc::NocCoord<N>,
    code: u32,
) -> Result<(), tt_device::TransportError> {
    assert_ne!(code, 0, "an abort carries a code");
    d.write32(w, tile, tt_isa::dataflow::ABORT, code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_local_credit_is_the_streams_cb_entry_and_passes_its_ownership_rule() {
        for stream in [Stream::Input, Stream::Output, Stream::Transfer] {
            for depth in [1u16, 2, MAX_DEPTH] {
                let channel = Channel::local(3, stream, depth);
                let rule = tt_isa::dataflow::Channel::of(stream, depth).unwrap();
                for action in [Action::Reserve, Action::Push, Action::Wait, Action::Pop] {
                    let entry = channel.credit(action);
                    assert_eq!(
                        entry,
                        [
                            tt_isa::dm::op::CB,
                            stream.word(),
                            action.word(),
                            depth as u32,
                            0,
                            0,
                            0,
                            0
                        ]
                    );
                    let owner = match action {
                        Action::Reserve | Action::Push => channel.producer.endpoint,
                        Action::Wait | Action::Pop => channel.consumers[0].endpoint,
                    };
                    assert!(rule.permits(owner, action), "{stream:?} {action:?}");
                }
                assert_eq!(channel.counters(), Some(stream.address()));
            }
        }
    }

    #[test]
    #[should_panic(expected = "depth 0")]
    fn a_local_channel_holds_a_page() {
        Channel::local(0, Stream::Input, 0);
    }

    #[test]
    fn a_cross_core_credit_word_is_planned_alone_in_its_sixteen_bytes() {
        use crate::l1::{Kind as Buffer, Requirements};
        let remote = Channel {
            producer: End {
                unit: 0,
                endpoint: Endpoint::Writer,
                tokens: 1,
            },
            consumers: vec![End {
                unit: 1,
                endpoint: Endpoint::Reader,
                tokens: 1,
            }],
            depth: 2,
            initial: 0,
            page: Page::Tiles {
                format: TileFormat::BF16,
                tiles: 4,
            },
            kind: Kind::Remote,
        };
        let mut requirements = Requirements::new(1);
        let odd = requirements.scratch("odd", 4, 4, 0..1);
        let data = requirements.channel("received", &remote, 0..1);
        let freed = requirements.channel("freed", &remote, 0..1);
        requirements.channel("input", &Channel::local(0, Stream::Input, 2), 0..1);
        let arena = tt_isa::l1::DATA;
        let plan = requirements.plan(arena).unwrap();
        crate::l1::check(&requirements, &plan, arena).unwrap();
        for word in [data, freed] {
            assert_eq!(plan.addr(word) % 16, 0);
        }
        assert!(plan.addr(data).abs_diff(plan.addr(freed)) >= 16);
        assert!(plan.addr(odd).abs_diff(plan.addr(data)) >= 4);
        let kinds: Vec<Buffer> = requirements.buffers().iter().map(|b| b.kind).collect();
        assert_eq!(
            kinds[1..],
            [Buffer::Credit, Buffer::Credit, Buffer::Counters]
        );
        assert_eq!(requirements.buffers()[3].bytes, 8);
        // A credit word sharing its 16 bytes, or counters of another size, are refused.
        for (bytes, align, kind) in [
            (4, 4, Buffer::Credit),
            (16, 4, Buffer::Credit),
            (4, 4, Buffer::Counters),
        ] {
            let mut bad = Requirements::new(1);
            bad.buffer(crate::l1::BufferSpec {
                name: "bad",
                bytes,
                align,
                live: 0..1,
                kind,
            });
            assert!(bad.plan(arena).is_err(), "{bytes} {align} {kind:?}");
        }
    }

    fn ring(consumers: Vec<(u8, u8)>, slots: Slots) -> Ring {
        Ring {
            producer: (1, 2),
            consumers,
            depth: 2,
            page_bytes: 20 * 1024,
            slots,
            received: tt_isa::l1::DATA.base as u32,
            freed: tt_isa::l1::DATA.base as u32 + 16,
        }
    }

    fn rings() -> [Ring; 3] {
        let base = tt_isa::l1::DATA.base as u32 + 0x1000;
        [
            ring(vec![(3, 4)], Slots::L1 { base }),
            ring(vec![(3, 4), (4, 4), (3, 5), (4, 5)], Slots::L1 { base }),
            ring(
                vec![(3, 4)],
                Slots::Gddr {
                    channel: 2,
                    port: 1,
                    offset: 0x40_0000,
                },
            ),
        ]
    }

    /// Five pages over a depth-2 ring: what the producer's NC sends, entry by entry.
    fn producer_list(r: &Ring) -> Vec<[u32; 8]> {
        let from = tt_isa::l1::DATA.base as u32 + 0x2_0000;
        (0..5)
            .flat_map(|page| r.send(page, from).unwrap())
            .collect()
    }

    #[test]
    fn a_ring_lowers_to_entries_its_movers_run_and_its_protocol_admits() {
        use tt_isa::dm::{op, Entry, Mover};
        for r in rings() {
            let list = producer_list(&r);
            r.check_send(&list).unwrap();
            for e in &list {
                assert!(Mover::NC.permits(e[0]), "{e:x?}");
                assert!(Entry::decode(u32::MAX, *e).is_ok(), "{e:x?}");
            }
            let consumers = r.consumers.len();
            // A page of 20 KiB is two requests; each consumer credited once a page; from the
            // third page on, a wait for the first lap's frees, counted over every consumer.
            let waits: Vec<u32> = list
                .iter()
                .filter(|e| e[0] == op::WAIT_CREDIT)
                .map(|e| e[2])
                .collect();
            assert_eq!(
                waits,
                [1, 2, 3].map(|freed| freed * consumers as u32),
                "{:?}",
                r.slots
            );
            assert_eq!(
                list.iter().filter(|e| e[0] == op::CREDIT).count(),
                5 * consumers
            );
            let data = list
                .iter()
                .filter(|e| matches!(e[0], op::WRITE | op::WRITE_L1 | op::WRITE_L1_MCAST));
            assert_eq!(data.count(), 10);
            // The consumer's side: waits for page k at k + 1, and frees to the producer.
            assert_eq!(
                r.receive(4),
                [op::WAIT_CREDIT, r.received, 5, 0, 0, 0, 0, 0]
            );
            assert_eq!(r.free(), [op::CREDIT, 1, 2, r.freed, 1, 0, 0, 0]);
            assert!(Mover::B.permits(r.receive(0)[0]) && Mover::B.permits(r.free()[0]));
        }
        let gddr = &rings()[2];
        let fetch = gddr.fetch(3, 0x4_0000).unwrap();
        assert_eq!(fetch[0], gddr.receive(3));
        assert_eq!(
            fetch[1][..6],
            [op::READ, 2, 1, gddr.slot(3), 0x4_0000, 16384]
        );
        assert!(rings()[0].fetch(0, 0x4_0000).is_err());
    }

    /// The planted faults the protocol check exists for: each one, in each kind of ring, is
    /// refused by name.
    #[test]
    fn a_producer_that_breaks_the_protocol_is_refused() {
        use tt_isa::dm::op;
        for r in rings() {
            let good = producer_list(&r);
            let at = |kind: u32, nth: usize| {
                good.iter()
                    .enumerate()
                    .filter(|(_, e)| e[0] == kind)
                    .nth(nth)
                    .unwrap()
                    .0
            };
            let data = good[1][0];
            // A write after its page's credit: page 0's first write moved past its credit.
            let mut late = good.clone();
            let write = late.remove(at(data, 0));
            late.insert(at(op::CREDIT, r.consumers.len() - 1), write);
            assert!(
                matches!(
                    r.check_send(&late),
                    Err(ChannelError::Unwritten { page: 0, .. })
                        | Err(ChannelError::WriteAfterCredit { page: 0, .. })
                ),
                "{:?}",
                r.check_send(&late)
            );
            // Page 0 written again once credited, before it is freed.
            let mut again = good.clone();
            again.insert(at(op::CREDIT, r.consumers.len() - 1) + 1, good[0]);
            assert_eq!(
                r.check_send(&again),
                Err(ChannelError::WriteAfterCredit {
                    page: 0,
                    entry: at(op::CREDIT, r.consumers.len() - 1) + 1
                })
            );
            // The second lap without waiting for the first's frees.
            let mut eager = good.clone();
            eager.remove(at(op::WAIT_CREDIT, 0));
            assert!(matches!(
                r.check_send(&eager),
                Err(ChannelError::NotFreed { page: 2, .. })
            ));
            // A wait for one free too few.
            let mut short = good.clone();
            short[at(op::WAIT_CREDIT, 0)][2] -= 1;
            assert!(matches!(
                r.check_send(&short),
                Err(ChannelError::NotFreed { page: 2, .. })
            ));
            // A page credited with a request's bytes missing.
            let mut dropped = good.clone();
            dropped.remove(at(data, 1));
            assert!(matches!(
                r.check_send(&dropped),
                Err(ChannelError::Unwritten { page: 0, .. })
            ));
            // A write into the wrong slot.
            let mut stray = good.clone();
            stray[at(data, 0)][3] += r.page_bytes;
            assert!(r.check_send(&stray).is_err());
        }
    }

    #[test]
    fn a_ring_that_cannot_be_lowered_is_refused() {
        let base = tt_isa::l1::DATA.base as u32;
        let mut r = ring(vec![(3, 4), (5, 4)], Slots::L1 { base });
        assert!(
            matches!(r.send(0, base), Err(ChannelError::Invalid(_))),
            "not a rectangle"
        );
        r.consumers = vec![(1, 2), (2, 2)];
        assert!(r.send(0, base).is_err(), "the sender inside its multicast");
        r.consumers = vec![(3, 4)];
        r.depth = 0;
        assert!(r.send(0, base).is_err());
        r.depth = 2;
        r.page_bytes = 24;
        assert!(r.send(0, base).is_err());
    }

    fn edge(from: (usize, Endpoint, u32), to: (usize, Endpoint, u32), depth: u16) -> Channel {
        let end = |(unit, endpoint, tokens)| End {
            unit,
            endpoint,
            tokens,
        };
        Channel {
            producer: end(from),
            consumers: vec![end(to)],
            depth,
            initial: 0,
            page: Page::Batch,
            kind: Kind::Remote,
        }
    }

    #[test]
    fn a_double_buffered_pipeline_across_two_units_is_scheduled() {
        use Endpoint::*;
        // Unit 0 reads, unpacks, packs and hands its writer's pages to unit 1's reader.
        let graph = Graph::new(vec![
            Channel::local(0, Stream::Input, 2),
            edge((0, Unpack, 1), (0, Pack, 1), 1),
            Channel::local(0, Stream::Output, 2),
            edge((0, Writer, 1), (1, Reader, 1), 2),
            Channel::local(1, Stream::Input, 2),
        ]);
        let schedule = graph.check().unwrap();
        assert_eq!(schedule.processes.len(), 6);
        assert!(schedule.firings.iter().all(|&q| q == 1));
    }

    /// A chain's partial, packed by one firing for the next one's unpack: a cycle that runs only
    /// from its one initial page; without it, nothing fires. More initial pages than room is
    /// refused.
    #[test]
    fn a_chain_runs_from_its_initial_page_and_deadlocks_without_it() {
        use Endpoint::*;
        let chain = |initial| {
            let mut partial = edge((0, Pack, 1), (0, Unpack, 1), 1);
            partial.kind = Kind::Roles;
            partial.initial = initial;
            let mut batch = edge((0, Unpack, 1), (0, Pack, 1), 1);
            batch.kind = Kind::Roles;
            Graph::new(vec![
                Channel::local(0, Stream::Input, 2),
                batch,
                partial,
                Channel::local(0, Stream::Output, 2),
            ])
        };
        assert!(chain(1).check().is_ok());
        assert!(matches!(
            chain(0).check(),
            Err(ChannelError::Deadlock { .. })
        ));
        assert!(matches!(chain(2).check(), Err(ChannelError::Invalid(_))));
    }

    #[test]
    fn rates_give_each_process_its_firings() {
        use Endpoint::*;
        // A reader making two pages a firing, an unpacker taking one, a packer three to a writer
        // taking two.
        let graph = Graph::new(vec![
            edge((0, Reader, 2), (0, Unpack, 1), 2),
            edge((0, Unpack, 3), (0, Pack, 2), 6),
        ]);
        let schedule = graph.check().unwrap();
        let q = |p| schedule.firings[schedule.processes.iter().position(|&x| x == p).unwrap()];
        // The least firings: two pages unpacked a read, three packed every two unpacks.
        assert_eq!([q((0, Reader)), q((0, Unpack)), q((0, Pack))], [1, 2, 3]);
    }

    #[test]
    fn unbalanced_rates_are_refused() {
        use Endpoint::*;
        // Two paths from the reader to the writer that disagree on how often the writer fires.
        let graph = Graph::new(vec![
            edge((0, Reader, 1), (0, Unpack, 1), 2),
            edge((0, Unpack, 1), (0, Writer, 1), 2),
            edge((0, Reader, 1), (0, Writer, 2), 2),
        ]);
        // Which channel shows it depends on the order the rates are spread in.
        assert!(matches!(
            graph.check(),
            Err(ChannelError::Unbalanced { .. })
        ));
    }

    #[test]
    fn a_cycle_without_initial_tokens_is_refused() {
        use Endpoint::*;
        // Unit 0's writer feeds unit 1's, which feeds unit 0's back: each waits for the other's
        // first page.
        let graph = Graph::new(vec![
            edge((0, Writer, 1), (1, Writer, 1), 2),
            edge((1, Writer, 1), (0, Writer, 1), 2),
        ]);
        assert!(matches!(
            graph.check(),
            Err(ChannelError::Deadlock { blocked }) if blocked.len() == 2
        ));
    }

    #[test]
    fn a_channel_shallower_than_a_firing_is_refused_and_one_deep_enough_is_not() {
        use Endpoint::*;
        let pages = |depth| Graph::new(vec![edge((0, Writer, 2), (1, Reader, 1), depth)]);
        assert_eq!(
            pages(1).check(),
            Err(ChannelError::Deadlock {
                blocked: vec![(0, Writer), (1, Reader)]
            })
        );
        assert_eq!(pages(2).check().unwrap().firings, [1, 2]);
    }

    #[test]
    fn a_traces_worth_of_runs_checks_in_linear_time() {
        use Endpoint::*;
        // `barriers::credit`'s shape: each of 120 units' 1,000 runs a process, in order on its unit,
        // and a credit from each run to the next unit's next run. Checked by scanning every edge
        // for every process this took hours (Qwen3.5's prefill chunk).
        let (units, runs) = (120, 1000);
        let id = |u: usize, r: usize| u * runs + r;
        let mut channels = Vec::new();
        for u in 0..units {
            for r in 0..runs - 1 {
                channels.push(edge((id(u, r), Writer, 1), (id(u, r + 1), Writer, 1), 1));
                channels.push(edge(
                    (id(u, r), Writer, 1),
                    (id((u + 1) % units, r + 1), Writer, 1),
                    1,
                ));
            }
        }
        let schedule = Graph::new(channels.clone()).check().unwrap();
        assert_eq!(schedule.firings.len(), units * runs);
        assert!(schedule.firings.iter().all(|&f| f == 1));
        // One credit back from a unit's last run to its first deadlocks the lot.
        channels.push(edge((id(7, runs - 1), Writer, 1), (id(7, 0), Writer, 1), 1));
        assert!(matches!(
            Graph::new(channels).check(),
            Err(ChannelError::Deadlock { blocked }) if blocked.len() > runs
        ));
    }

    #[test]
    fn a_page_of_tiles_takes_their_slots() {
        let page = Page::Tiles {
            format: TileFormat::BFP8,
            tiles: 3,
        };
        assert_eq!(page.bytes(), Some(3 * TileFormat::BFP8.slot_bytes()));
        assert_eq!(Page::Batch.bytes(), None);
    }
}
