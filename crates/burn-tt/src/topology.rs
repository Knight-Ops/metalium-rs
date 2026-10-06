//! Which cards a Burn device computes on, chosen in one place.
//!
//! Training code attaches one [`TtDevice`] and never learns how many chips are
//! behind it: both engines implement [`crate::Engine`]. So switching between one
//! card and several -- to benchmark a single card, say -- is this one call, or
//! the `TT_TOPOLOGY` environment variable through [`Topology::from_env`].
//!
//! What each gives today:
//! * [`Topology::Single`]: one card, tensors resident in its GDDR (Phase 9),
//!   every supported op on the device, on one Tensix tile or many
//!   ([`TileChoice`]; `TT_TILES` through [`tiles_from_env`]).
//! * [`Topology::Cards`]: matmuls split along `N` across the cabled cards over
//!   Ethernet, bit-identical to one card, with tensor slots retained in GDDR.
//!   Other primitives and batched matmuls compute on chip 0.

use crate::server::{
    attach, kmd_engine_with_elementwise, kmd_mesh_engine, AttachGuard, EngineError,
};
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

    /// This topology computing on `tile` on each card it has; only
    /// [`Topology::Single`] can spread over several tiles yet.
    pub fn on_tiles(self, tile: TileChoice) -> Result<Self, EngineError> {
        match self {
            Topology::Single { card, .. } => Ok(Topology::Single { card, tile }),
            Topology::Cards { .. }
                if tile == TileChoice::Exactly(Self::COMPUTE.0, Self::COMPUTE.1) =>
            {
                Ok(self)
            }
            Topology::Cards { .. } => Err(EngineError(format!(
                "{tile:?}: several cards compute on one tile each, for now"
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

/// From `TT_TILES`: how many Tensix tiles one card computes on, a count or
/// `"all"` ([`parse_tiles`]). `None` when unset.
pub fn tiles_from_env() -> Option<Result<TileChoice, EngineError>> {
    let v = std::env::var("TT_TILES").ok()?;
    Some(parse_tiles(&v))
}

/// See [`tiles_from_env`].
pub fn parse_tiles(s: &str) -> Result<TileChoice, EngineError> {
    match s.trim() {
        "all" => Ok(TileChoice::All),
        n => match n.parse::<usize>() {
            Ok(n) if n > 0 => Ok(TileChoice::Count(n)),
            _ => Err(EngineError(format!(
                "TT_TILES={s:?}: expected a tile count, e.g. \"8\", or \"all\""
            ))),
        },
    }
}

/// Attach `device` on `topology`'s silicon. The one call training code makes.
pub fn attach_topology(
    device: TtDevice,
    topology: Topology,
    route: SrcRoute,
    fidelity: Fidelity,
) -> Result<AttachGuard, EngineError> {
    attach_topology_with_elementwise(
        device,
        topology,
        route,
        fidelity,
        crate::ElementwiseMode::Sfpu,
    )
}

pub fn attach_topology_with_elementwise(
    device: TtDevice,
    topology: Topology,
    route: SrcRoute,
    fidelity: Fidelity,
    mode: crate::ElementwiseMode,
) -> Result<AttachGuard, EngineError> {
    if matches!(topology, Topology::Cards { .. }) && mode != crate::ElementwiseMode::Sfpu {
        return Err(EngineError(
            "matrix elementwise mode is unsupported on mesh engines".into(),
        ));
    }
    match topology {
        Topology::Single { card, tile } => attach(
            device,
            kmd_engine_with_elementwise(TtDevice::new(card), tile, route, fidelity, mode),
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
    fn matrix_mesh_opt_in_is_refused_before_attachment() {
        let result = attach_topology_with_elementwise(
            TtDevice::new(0),
            Topology::cards(&[0, 1]),
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            crate::ElementwiseMode::Matrix {
                precision: crate::SrcPrecision::Tf32,
                fidelity: Fidelity::HiFi4,
            },
        );
        assert!(matches!(result,Err(e) if e.0.contains("unsupported on mesh")));
    }

    #[test]
    fn one_card_is_single_and_several_are_sharded() {
        assert_eq!(Topology::parse("1").unwrap(), Topology::single(1));
        assert!(
            matches!(Topology::parse("0, 1").unwrap(), Topology::Cards { ref cards, .. } if cards == &[0, 1])
        );
        assert!(Topology::parse("").is_err());
        assert!(Topology::parse("zero").is_err());
    }

    #[test]
    fn tiles_are_a_count_or_all() {
        assert_eq!(parse_tiles("8").unwrap(), TileChoice::Count(8));
        assert_eq!(parse_tiles(" all ").unwrap(), TileChoice::All);
        assert!(parse_tiles("0").is_err());
        assert!(parse_tiles("many").is_err());
        let one = Topology::single(0);
        assert_eq!(
            one.on_tiles(TileChoice::All).unwrap(),
            Topology::Single {
                card: 0,
                tile: TileChoice::All
            }
        );
        assert!(Topology::cards(&[0, 1])
            .on_tiles(TileChoice::Count(4))
            .is_err());
    }
}
