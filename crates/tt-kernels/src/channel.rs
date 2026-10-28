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
//!
//! The local kind is what every op emits today; this module is where its words are made
//! ([`Channel::credit`]), so a stream's credits are written in one place.

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
