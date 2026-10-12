//! What opens a device nobody attached explicitly, and how it is configured.
//!
//! The first op that needs a [`TtDevice`] attaches it: the registry's default
//! engine opens `/dev/tenstorrent/{chip}` with the configuration the environment
//! gives (`TT_TOPOLOGY`, `TT_TILES`, `TT_MATH`, ...). [`init`] does the same
//! eagerly and fallibly, with a [`TtConfig`] the caller chose; [`attach`] and
//! [`attach_with`] keep the scoped, explicit forms for tests and benchmarks.
//! [`set_default_engine`] replaces what the registry opens -- how the
//! simulator, which `burn-tt` never depends on, serves a device lazily.
//!
//! A failed attach is remembered per device, so the next op fails at once with
//! the same [`TtError`] instead of opening the card again; [`init`] and
//! [`detach`] clear it.

use std::sync::{Arc, Mutex};

use crate::error::TtError;
use crate::server::{self, AttachGuard, Serve};
use crate::topology::{tiles_from_env, Topology};
use crate::{ElementwiseMode, Fidelity, MatmulMode, SrcRoute, TileChoice, TtDevice};

/// How a device is opened.
///
/// The default is one card -- the device's own, `/dev/tenstorrent/{chip}` --
/// computing on its first surviving tile, through `Tf32FromFp32` at `HiFi4`,
/// with SFPU element-wise arithmetic and the fast (FPU) matmul ([`MatmulMode`]). That tile is the one choice that exists
/// on every card, however it is harvested; a tile *count* is the caller's
/// ([`TtConfig::tiles`], `TT_TILES`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TtConfig {
    topology: Option<Topology>,
    tiles: Option<TileChoice>,
    route: SrcRoute,
    fidelity: Fidelity,
    elementwise: ElementwiseMode,
    matmul: MatmulMode,
}

impl Default for TtConfig {
    fn default() -> Self {
        TtConfig {
            topology: None,
            tiles: None,
            route: SrcRoute::Tf32FromFp32,
            fidelity: Fidelity::HiFi4,
            elementwise: ElementwiseMode::Sfpu,
            matmul: MatmulMode::Fast,
        }
    }
}

impl TtConfig {
    /// The defaults, with `TT_TOPOLOGY`, `TT_TILES` and `TT_EXACT_MATMUL` applied if set.
    pub fn from_env() -> Result<Self, TtError> {
        let mut config = Self::default();
        match std::env::var("TT_EXACT_MATMUL").as_deref() {
            Err(_) | Ok("0") => {}
            Ok("1") => config.matmul = MatmulMode::Exact,
            Ok(v) => {
                return Err(TtError::Device(format!(
                    "TT_EXACT_MATMUL={v}: expected 0 or 1"
                )))
            }
        }
        if let Some(topology) = Topology::from_env() {
            config.topology = Some(topology?);
        }
        if let Some(tiles) = tiles_from_env() {
            config.tiles = Some(tiles?);
        }
        Ok(config)
    }

    /// Which card(s) to use, instead of the device's own.
    pub fn topology(mut self, topology: Topology) -> Self {
        self.topology = Some(topology);
        self
    }

    /// Which Tensix tiles each card computes on ([`Topology::on_tiles`]).
    pub fn tiles(mut self, tiles: TileChoice) -> Self {
        self.tiles = Some(tiles);
        self
    }

    pub fn route(mut self, route: SrcRoute) -> Self {
        self.route = route;
        self
    }

    pub fn fidelity(mut self, fidelity: Fidelity) -> Self {
        self.fidelity = fidelity;
        self
    }

    /// The element-wise arithmetic engine, fixed for the attachment.
    pub fn elementwise(mut self, mode: ElementwiseMode) -> Self {
        self.elementwise = mode;
        self
    }

    /// How matmuls accumulate on this attachment: the fast FPU path (the default) or exact F32
    /// ([`MatmulMode`]). Fixed for the attachment; [`with_matmul_mode`](crate::with_matmul_mode)
    /// overrides it for the ops a closure builds.
    pub fn matmul(mut self, mode: MatmulMode) -> Self {
        self.matmul = mode;
        self
    }

    /// The cards and tiles this configuration opens for `device`.
    pub fn resolve(&self, device: TtDevice) -> Result<Topology, TtError> {
        let base = self.topology.clone().unwrap_or(Topology::Single {
            card: device.chip,
            tile: TileChoice::First,
        });
        match self.tiles {
            Some(tiles) => base.on_tiles(tiles),
            None => Ok(base),
        }
    }
}

/// What starts an engine's server: handed to [`attach`](crate::attach).
pub type EngineFactory = Box<dyn FnOnce(Serve) -> Result<(), TtError> + Send>;

type DefaultEngine =
    Arc<dyn Fn(TtDevice, &TtConfig) -> Result<EngineFactory, TtError> + Send + Sync>;

static DEFAULT: Mutex<Option<DefaultEngine>> = Mutex::new(None);

/// The silicon engine for `device` as `config` asks: one card on a `Session`,
/// or a `Fabric` of cards.
pub(crate) fn kmd_factory(device: TtDevice, config: &TtConfig) -> Result<EngineFactory, TtError> {
    match config.resolve(device)? {
        Topology::Single { card, tile } => Ok(Box::new(server::kmd_engine_with_elementwise(
            TtDevice::new(card),
            tile,
            config.route,
            config.fidelity,
            config.elementwise,
            config.matmul,
        ))),
        Topology::Cards {
            cards,
            compute,
            relay,
        } => {
            if config.elementwise != ElementwiseMode::Sfpu {
                return Err(TtError::unsupported(
                    "matrix elementwise mode is unsupported on mesh engines",
                ));
            }
            if config.matmul != MatmulMode::Fast {
                return Err(TtError::unsupported(
                    "exact matmul mode is unsupported on mesh engines",
                ));
            }
            Ok(Box::new(server::kmd_mesh_engine(
                cards,
                compute,
                relay,
                config.route,
                config.fidelity,
            )))
        }
    }
}

/// What the registry opens for `device`: the engine [`set_default_engine`]
/// chose, or silicon.
pub(crate) fn default_factory(
    device: TtDevice,
    config: &TtConfig,
) -> Result<EngineFactory, TtError> {
    let custom = DEFAULT.lock().unwrap_or_else(|p| p.into_inner()).clone();
    match custom {
        Some(make) => make(device, config),
        None => kmd_factory(device, config),
    }
}

/// Choose what the registry opens for a device nobody attached: `make` builds
/// the engine's factory for the device and the configuration. For the process;
/// replaces an earlier choice and forgets remembered attach failures. This is
/// how `tt-tests` serves devices from the simulator.
pub fn set_default_engine(
    make: impl Fn(TtDevice, &TtConfig) -> Result<EngineFactory, TtError> + Send + Sync + 'static,
) {
    *DEFAULT.lock().unwrap_or_else(|p| p.into_inner()) = Some(Arc::new(make));
    server::forget_failures();
}

/// The registry attaches nothing: an op on a device nobody attached panics
/// with [`TtError::NotAttached`].
pub fn refuse_default_engine() {
    set_default_engine(|device, _| Err(TtError::NotAttached { device }));
}

/// Back to silicon.
pub fn reset_default_engine() {
    *DEFAULT.lock().unwrap_or_else(|p| p.into_inner()) = None;
    server::forget_failures();
}

/// Attach `device` now, with `config`, and keep it attached for the process;
/// the explicit, fallible form of the attach the first op would do.
///
/// Idempotent for an equal `config`. A different one on a device that is
/// already attached is [`TtError::AlreadyAttached`], since an engine's
/// configuration cannot change once it is serving: [`detach`] it first.
/// Clears the remembered failure of an earlier attempt.
pub fn init(device: TtDevice, config: TtConfig) -> Result<(), TtError> {
    server::init(device, config)
}

/// Detach a device [`init`] or the first op attached, dropping its hardware
/// (a card returns to idle). The next op attaches it again; tensors made
/// before are [`TtError::StaleTensor`]. Does nothing for a device that is not
/// attached.
pub fn detach(device: TtDevice) -> Result<(), TtError> {
    server::detach(device)
}

/// Attach `device` on the silicon `config` describes, for as long as the
/// guard lives: the scoped form tests and benchmarks use. Unlike [`init`] it
/// does not use [`set_default_engine`].
pub fn attach_with(device: TtDevice, config: TtConfig) -> Result<AttachGuard, TtError> {
    server::attach(device, kmd_factory(device, &config)?)
}

impl TtDevice {
    /// The devices this process can use: every `/dev/tenstorrent/{n}` node,
    /// plus any device attached some other way (the simulator's). Opens
    /// nothing.
    pub fn enumerate() -> Vec<TtDevice> {
        let mut chips: Vec<u16> = tt_kmd::Kmd::enumerate().unwrap_or_default();
        chips.extend(server::attached_devices().into_iter().map(|d| d.chip));
        chips.sort_unstable();
        chips.dedup();
        chips.into_iter().map(TtDevice::new).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_one_card_on_its_first_tile() {
        let config = TtConfig::default();
        assert_eq!(
            config.resolve(TtDevice::new(3)).unwrap(),
            Topology::Single {
                card: 3,
                tile: TileChoice::First
            }
        );
    }

    #[test]
    fn tiles_and_topology_override_the_default() {
        let config = TtConfig::default().tiles(TileChoice::Count(8));
        assert_eq!(
            config.resolve(TtDevice::new(1)).unwrap(),
            Topology::Single {
                card: 1,
                tile: TileChoice::Count(8)
            }
        );
        let config = TtConfig::default().topology(Topology::single(0));
        assert_eq!(
            config.resolve(TtDevice::new(1)).unwrap(),
            Topology::single(0)
        );
        // Several cards compute on one tile each, for now.
        let several = TtConfig::default()
            .topology(Topology::cards(&[0, 1]))
            .tiles(TileChoice::Count(4));
        assert!(several.resolve(TtDevice::new(0)).is_err());
    }

    #[test]
    fn matrix_arithmetic_on_a_mesh_is_refused_before_anything_opens() {
        let config = TtConfig::default()
            .topology(Topology::cards(&[0, 1]))
            .elementwise(ElementwiseMode::Matrix {
                precision: crate::SrcPrecision::Tf32,
                fidelity: Fidelity::HiFi4,
            });
        let refused = attach_with(TtDevice::new(0), config.clone())
            .err()
            .expect("refused");
        assert!(refused.to_string().contains("unsupported on mesh"));
        assert!(kmd_factory(TtDevice::new(0), &config).is_err());
    }

    #[test]
    fn exact_matmul_on_a_mesh_is_refused_before_anything_opens() {
        let config = TtConfig::default()
            .topology(Topology::cards(&[0, 1]))
            .matmul(MatmulMode::Exact);
        let refused = attach_with(TtDevice::new(0), config.clone())
            .err()
            .expect("refused");
        assert!(refused.to_string().contains("exact matmul"));
        assert!(refused.to_string().contains("unsupported on mesh"));
        assert!(kmd_factory(TtDevice::new(0), &config).is_err());
        // Fast matmul on the same mesh is not refused for that reason.
        let fast = TtConfig::default().topology(Topology::cards(&[0, 1]));
        assert_eq!(fast, fast.clone().matmul(MatmulMode::Fast));
    }

    #[test]
    fn the_matmul_mode_is_part_of_the_configuration() {
        assert_eq!(
            TtConfig::default(),
            TtConfig::default().matmul(MatmulMode::Fast)
        );
        assert_ne!(
            TtConfig::default(),
            TtConfig::default().matmul(MatmulMode::Exact)
        );
    }

    #[test]
    fn enumerate_lists_nodes_without_opening_anything() {
        let found = TtDevice::enumerate();
        let mut sorted = found.iter().map(|d| d.chip).collect::<Vec<_>>();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(found.len(), sorted.len(), "sorted and without repeats");
    }
}
