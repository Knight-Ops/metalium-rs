//! Which cards a Burn device computes on, chosen in one place.
//!
//! Training code attaches one [`TtDevice`] and never learns how many chips are
//! behind it: both engines implement [`crate::Engine`]. So switching between one
//! card and several -- to benchmark a single card, say -- is this one call, or
//! the `TT_TOPOLOGY` environment variable through [`Topology::from_env`].
//!
//! What each gives today:
//! * [`Topology::Single`]: one card, tensors resident in its GDDR (Phase 9),
//!   every supported op on the device.
//! * [`Topology::Cards`]: matmuls split along `N` across the cabled cards over
//!   Ethernet (Phase 8), bit-identical to one card. Tensors are staged from the
//!   host for each matmul: the mesh does not yet keep them in GDDR.

use crate::server::{attach, kmd_engine, kmd_mesh_engine, AttachGuard, EngineError};
use crate::{Fidelity, SrcRoute, TileChoice, TtDevice};

/// The cards behind one [`TtDevice`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Topology {
    /// `/dev/tenstorrent/{card}`, computing on `tile`.
    Single { card: u16, tile: TileChoice },
    /// Every card in `cards`, the first being the host's way in and out,
    /// computing on `compute` and relaying through `relay` on each.
    Cards {
        cards: Vec<u16>,
        compute: (u8, u8),
        relay: (u8, u8),
    },
}

impl Topology {
    /// The tiles every gate in this workspace uses: inside the surviving Tensix
    /// columns of both p150a cards here (`tt-tests/src/backend.rs`).
    pub const COMPUTE: (u8, u8) = (3, 4);
    pub const RELAY: (u8, u8) = (4, 4);

    /// One card.
    pub fn single(card: u16) -> Self {
        Topology::Single {
            card,
            tile: TileChoice::Exactly(Self::COMPUTE.0, Self::COMPUTE.1),
        }
    }

    /// Several cards, sharded.
    pub fn cards(cards: &[u16]) -> Self {
        match cards {
            [card] => Self::single(*card),
            _ => Topology::Cards {
                cards: cards.to_vec(),
                compute: Self::COMPUTE,
                relay: Self::RELAY,
            },
        }
    }

    /// From `TT_TOPOLOGY`: a comma-separated card list, `"0"` or `"0,1"`; one
    /// card is [`Topology::Single`]. `None` when unset.
    pub fn from_env() -> Option<Result<Self, EngineError>> {
        let v = std::env::var("TT_TOPOLOGY").ok()?;
        Some(Self::parse(&v))
    }

    /// See [`Topology::from_env`].
    pub fn parse(s: &str) -> Result<Self, EngineError> {
        let cards: Result<Vec<u16>, _> = s.split(',').map(|c| c.trim().parse()).collect();
        match cards {
            Ok(c) if !c.is_empty() => Ok(Self::cards(&c)),
            _ => Err(EngineError(format!(
                "TT_TOPOLOGY={s:?}: expected card numbers, e.g. \"0\" or \"0,1\""
            ))),
        }
    }

    /// How many cards.
    pub fn len(&self) -> usize {
        match self {
            Topology::Single { .. } => 1,
            Topology::Cards { cards, .. } => cards.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Attach `device` on `topology`'s silicon. The one call training code makes.
pub fn attach_topology(
    device: TtDevice,
    topology: Topology,
    route: SrcRoute,
    fidelity: Fidelity,
) -> Result<AttachGuard, EngineError> {
    match topology {
        Topology::Single { card, tile } => attach(
            device,
            kmd_engine(TtDevice::new(card), tile, route, fidelity),
        ),
        Topology::Cards {
            cards,
            compute,
            relay,
        } => attach(
            device,
            kmd_mesh_engine(cards, compute, relay, route, fidelity),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_card_is_single_and_several_are_sharded() {
        assert_eq!(Topology::parse("1").unwrap(), Topology::single(1));
        assert!(
            matches!(Topology::parse("0, 1").unwrap(), Topology::Cards { ref cards, .. } if cards == &[0, 1])
        );
        assert!(Topology::parse("").is_err());
        assert!(Topology::parse("zero").is_err());
    }
}
