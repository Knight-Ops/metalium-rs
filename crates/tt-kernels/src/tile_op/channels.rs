//! Channels (pass 6 of `docs/learnings/dataflow-compiler.md` §5): an op's channels on a unit as a
//! synchronous dataflow graph ([`channel::Graph`]), checked before its jobs are lowered.
//!
//! On each unit, a run (or a chunk of a reduction's terms) is one firing of every role:
//! - the mover's gather fills the `Input` stream, which the unpacker drains;
//! - the unpacker hands the batch to the packer through `Dst` ([`Kind::Roles`]);
//! - the packer fills the `Output` stream, which the mover's scatter drains.
//!
//! The streams are as deep as the runs are staged: two where a unit's runs alternate halves of
//! the arena (pipelined, or a chain continued in L1's halves), one otherwise. A reduction whose
//! contract chains its chunks adds the partial's edge, with one page before the first firing (the
//! first chunk reloads none): from the packer to the next chunk's unpack in L1
//! ([`Combine::L1Halves`]), or from the mover's write to its next read through GDDR
//! ([`Combine::GddrPartials`]). Until an op's units hand each other data (F4), their graphs are
//! disjoint and alike, so one unit's is checked, once a process for each shape of graph.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use tt_isa::dataflow::{Endpoint, Stream};

use crate::channel::{self, Channel, End, Graph, Kind, Page};
use crate::tensor::{Result, TensorError};

use super::distribute::Deal;
use super::{Combine, IterKind, TileOp};

/// How a reduction's contract chains an item's chunks: the partial's channel kind.
fn chain(op: &TileOp) -> Option<Kind> {
    op.domain.iter().find_map(|i| match i.kind {
        IterKind::Reduction(c) => match c.combine {
            Combine::InDst => None,
            Combine::L1Halves { .. } => Some(Kind::Roles),
            Combine::GddrPartials { .. } => Some(Kind::Gddr),
        },
        IterKind::Parallel => None,
    })
}

/// The pages a unit's streams hold: two where its runs (or chunks) alternate halves of the arena.
pub fn depth(op: &TileOp, deal: &Deal) -> u16 {
    match chain(op) {
        Some(Kind::Roles) => 2,
        Some(_) => 1,
        None if deal.pipelined => 2,
        None => 1,
    }
}

/// One unit's channels: its streams `depth` pages deep, and a chained reduction's partial with
/// `initial` pages before the first firing.
pub fn unit_graph(unit: usize, depth: u16, chain: Option<Kind>, initial: u16) -> Graph {
    let edge = |from, to, kind, depth, initial| Channel {
        producer: End {
            unit,
            endpoint: from,
            tokens: 1,
        },
        consumers: vec![End {
            unit,
            endpoint: to,
            tokens: 1,
        }],
        depth,
        initial,
        page: Page::Batch,
        kind,
    };
    let mut channels = vec![
        Channel::local(unit, Stream::Input, depth),
        edge(Endpoint::Unpack, Endpoint::Pack, Kind::Roles, 1, 0),
        Channel::local(unit, Stream::Output, depth),
    ];
    match chain {
        Some(Kind::Roles) => channels.push(edge(
            Endpoint::Pack,
            Endpoint::Unpack,
            Kind::Roles,
            1,
            initial,
        )),
        Some(kind) => channels.push(edge(Endpoint::Writer, Endpoint::Reader, kind, 1, initial)),
        None => {}
    }
    Graph::new(channels)
}

/// `op`'s channels on a unit, dealt as `deal` says, checked ([`channel::Graph::check`]): the
/// depth its streams take.
pub fn check(op: &TileOp, deal: &Deal) -> Result<u16> {
    /// The shapes of graph checked: the streams' depth, and the chain's (`true`: in L1).
    type Checked = OnceLock<Mutex<HashSet<(u16, Option<bool>)>>>;
    static CHECKED: Checked = OnceLock::new();
    let (depth, chain) = (depth(op, deal), chain(op));
    let key = (depth, chain.map(|k| k == Kind::Roles));
    let checked = CHECKED.get_or_init(Default::default);
    if checked
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .contains(&key)
    {
        return Ok(depth);
    }
    unit_graph(0, depth, chain, 1)
        .check()
        .map_err(|e: channel::ChannelError| {
            TensorError::Shape(format!("the op's channels are refused: {e}"))
        })?;
    checked
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(key);
    Ok(depth)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shape of graph an op makes passes; a chain without its initial page is refused (the
    /// planted fault: its first chunk would wait for a partial no chunk has made).
    #[test]
    fn each_ops_graph_passes_and_a_chain_without_its_first_page_deadlocks() {
        for (depth, chain) in [
            (1, None),
            (2, None),
            (2, Some(Kind::Roles)),
            (1, Some(Kind::Gddr)),
        ] {
            let schedule = unit_graph(3, depth, chain, 1).check().unwrap();
            assert!(schedule.firings.iter().all(|&q| q == 1));
            if chain.is_some() {
                assert!(matches!(
                    unit_graph(3, depth, chain, 0).check(),
                    Err(channel::ChannelError::Deadlock { .. })
                ));
            }
        }
    }
}
