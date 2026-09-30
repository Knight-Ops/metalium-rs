//! A `burn_tt::TtDevice` attached to whatever the suite runs against: ttsim by
//! default, the card `TT_SILICON_DEVICE` names with `--features silicon`.
//!
//! The simulator engine lives here rather than in `burn-tt` so the backend never
//! depends on the simulator (`check-no-sim-in-ship`). Like every gate, a Burn
//! gate runs in a fork: the simulator is a once-per-process singleton, and on
//! silicon the child's closing file descriptor fires the driver's cleanup write.

use burn_tt::{attach, Fidelity, SrcRoute, TtDevice};

/// How the device multiplies.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub route: SrcRoute,
    pub fidelity: Fidelity,
    /// Simulated cycles per role per run (`tt_kernels::runtime::run`).
    pub budget: u64,
}

impl Default for Config {
    /// What a training backend runs: TF32 operands, all four fidelity phases.
    fn default() -> Self {
        Config {
            route: SrcRoute::Tf32FromFp32,
            fidelity: Fidelity::HiFi4,
            budget: 4_000_000,
        }
    }
}

/// Run `f` in a fork with a device attached, as `f`'s argument.
///
/// Panics in `f` fail the calling test, with the child's message above.
#[track_caller]
pub fn with_device(config: Config, f: impl FnOnce(TtDevice)) {
    if let Err(e) = tt_ttsim::fork_scope(|| {
        let device = device();
        let _guard = attach_engine(device, config)
            .unwrap_or_else(|e| panic!("could not attach {device}: {e}"));
        f(device);
    }) {
        panic!("{e}");
    }
}

/// [`with_device`], but the device computes across `chips` chips joined by
/// Ethernet (`burn_tt::MeshEngine`): on the simulator the `bh_x2` or `bh_x4`
/// build, on silicon cards `0..chips`.
#[track_caller]
pub fn with_mesh_device(config: Config, chips: usize, f: impl FnOnce(TtDevice)) {
    if let Err(e) = tt_ttsim::fork_scope(|| {
        let device = TtDevice::new(0);
        let _guard = attach_mesh(device, config, chips)
            .unwrap_or_else(|e| panic!("could not attach a {chips}-chip {device}: {e}"));
        f(device);
    }) {
        panic!("{e}");
    }
}

#[cfg(not(feature = "silicon"))]
fn attach_mesh(
    device: TtDevice,
    config: Config,
    chips: usize,
) -> Result<burn_tt::AttachGuard, burn_tt::EngineError> {
    use burn_tt::{EngineError, MeshEngine};
    use tt_device::Device;
    use tt_kernels::shard::{Chip, Fabric};
    let (lib, table): (_, &[_]) = match chips {
        2 => (tt_ttsim::x2_lib_path(), &crate::topology::BH_X2),
        4 => (tt_ttsim::x4_lib_path(), &crate::topology::BH_X4),
        n => return Err(EngineError(format!("no {n}-chip simulator build"))),
    };
    attach(device, move |serve| {
        let err = |e: &dyn std::fmt::Display| EngineError(e.to_string());
        let mut sim = tt_ttsim::Simulator::open_path(lib).map_err(|e| err(&e))?;
        let (compute, relay) = (crate::harness::tensix_tile(), crate::harness::relay_tile());
        let mut list = Vec::new();
        for t in sim.transports() {
            let dev = Device::open(t).map_err(|e| err(&e))?;
            list.push(Chip::new(dev, compute, relay).map_err(|e| err(&e))?);
        }
        let links = crate::topology::links(table);
        let fabric = Fabric::new(
            list,
            &links,
            tt_firmware_images::ROLES,
            tt_firmware_images::ETH_E1,
        )
        .map_err(|e| err(&e))?;
        let mut engine = MeshEngine {
            fabric,
            route: config.route,
            fidelity: config.fidelity,
            budget: config.budget,
        };
        serve.serve(&mut engine);
        Ok(())
    })
}

#[cfg(feature = "silicon")]
fn attach_mesh(
    device: TtDevice,
    config: Config,
    chips: usize,
) -> Result<burn_tt::AttachGuard, burn_tt::EngineError> {
    attach(
        device,
        burn_tt::kmd_mesh_engine(
            (0..chips as u16).collect(),
            crate::backend::GATE_TILE,
            crate::backend::RELAY_TILE,
            config.route,
            config.fidelity,
        ),
    )
}

#[cfg(not(feature = "silicon"))]
fn device() -> TtDevice {
    TtDevice::new(0)
}

#[cfg(feature = "silicon")]
fn device() -> TtDevice {
    TtDevice::new(crate::backend::device_index())
}

#[cfg(not(feature = "silicon"))]
fn attach_engine(
    device: TtDevice,
    config: Config,
) -> Result<burn_tt::AttachGuard, burn_tt::EngineError> {
    use burn_tt::{Engine, EngineError};
    use tt_device::Device;

    // The same `Session` the silicon engine uses (`burn_tt::kmd_engine`), so
    // the reduced run's golden exercises the resident path on both targets.
    struct Sim<'a> {
        session: tt_kernels::session::Session<tt_ttsim::LibTtsim<'a>>,
        config: Config,
        buffers: burn_tt::DramBuffers,
    }
    impl Engine for Sim<'_> {
        fn supports_dram(&self) -> bool {
            true
        }
        fn upload(
            &mut self,
            v: &[f32],
            r: usize,
            c: usize,
        ) -> Result<burn_tt::BufferId, EngineError> {
            self.buffers.upload(&mut self.session, v, r, c)
        }
        fn download(&mut self, id: burn_tt::BufferId) -> Result<Vec<f32>, EngineError> {
            self.buffers.download(&mut self.session, id)
        }
        fn free(&mut self, id: burn_tt::BufferId) {
            self.buffers.free(&mut self.session, id)
        }
        fn matmul_dram(
            &mut self,
            a: burn_tt::BufferId,
            ta: bool,
            b: burn_tt::BufferId,
            tb: bool,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), EngineError> {
            let c = &self.config;
            self.buffers.matmul(
                &mut self.session,
                a,
                ta,
                b,
                tb,
                c.route,
                c.fidelity,
                c.budget,
            )
        }
        fn eltwise(
            &mut self,
            kind: u32,
            scalar: f32,
            a: burn_tt::BufferId,
            b: Option<burn_tt::BufferId>,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), EngineError> {
            self.buffers.eltwise(&mut self.session, kind, scalar, a, b)
        }
        fn matmul(
            &mut self,
            a: &[f32],
            b: &[f32],
            mkn: [usize; 3],
        ) -> Result<Vec<f32>, EngineError> {
            Ok(self.session.matmul(
                a,
                b,
                mkn,
                self.config.route,
                self.config.fidelity,
                self.config.budget,
            )?)
        }
    }

    attach(device, move |serve| {
        let mut sim = tt_ttsim::Simulator::open()
            .map_err(|e| EngineError(format!("could not open the simulator: {e}")))?;
        let dev = Device::open(sim.transport()).map_err(|e| EngineError(e.to_string()))?;
        let t = crate::harness::tensix_tile();
        let session = tt_kernels::session::Session::open(
            dev,
            tt_firmware_images::ROLES,
            tt_kernels::session::TileChoice::Exactly(t.x(), t.y()),
            |_, _| Ok(()),
        )
        .map_err(|e| EngineError(e.to_string()))?;
        let mut session = session;
        session
            .enable_dram(tt_firmware_images::DM_B.1)
            .map_err(|e| EngineError(e.to_string()))?;
        serve.serve(&mut Sim {
            session,
            config,
            buffers: Default::default(),
        });
        Ok(())
    })
}

#[cfg(feature = "silicon")]
fn attach_engine(
    device: TtDevice,
    config: Config,
) -> Result<burn_tt::AttachGuard, burn_tt::EngineError> {
    // `TT_TOPOLOGY` ("0", "0,1") picks the cards, so one benchmark runs on one
    // card or several unchanged (`burn_tt::Topology`); unset, the card this
    // gate was pointed at, on the gate tile.
    let topology = match burn_tt::Topology::from_env() {
        Some(t) => t?,
        None => {
            let (x, y) = crate::backend::GATE_TILE;
            burn_tt::Topology::Single {
                card: device.chip,
                tile: burn_tt::TileChoice::Exactly(x, y),
            }
        }
    };
    eprintln!("topology: {topology:?}");
    burn_tt::attach_topology(device, topology, config.route, config.fidelity)
}
