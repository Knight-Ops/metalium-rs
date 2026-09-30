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

    struct Sim<'a> {
        dev: Device<tt_ttsim::LibTtsim<'a>>,
        config: Config,
    }
    impl Engine for Sim<'_> {
        fn matmul(
            &mut self,
            a: &[f32],
            b: &[f32],
            mkn: [usize; 3],
        ) -> Result<Vec<f32>, EngineError> {
            Ok(tt_kernels::session::matmul_on(
                &mut self.dev,
                crate::harness::tensix_tile(),
                &tt_firmware_images::ROLES,
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
        serve.serve(&mut Sim { dev, config });
        Ok(())
    })
}

#[cfg(feature = "silicon")]
fn attach_engine(
    device: TtDevice,
    config: Config,
) -> Result<burn_tt::AttachGuard, burn_tt::EngineError> {
    let (x, y) = crate::backend::GATE_TILE;
    attach(
        device,
        burn_tt::kmd_engine(
            device,
            burn_tt::TileChoice::Exactly(x, y),
            config.route,
            config.fidelity,
        ),
    )
}
