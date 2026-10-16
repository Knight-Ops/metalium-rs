//! A `burn_tt::TtDevice` attached to whatever the suite runs against: ttsim by
//! default, the card `TT_SILICON_DEVICE` names with `--features silicon`.
//!
//! The simulator engine lives here rather than in `burn-tt` so the backend never
//! depends on the simulator (`check-no-sim-in-ship`). Like every gate, a Burn
//! gate runs in a fork: the simulator is a once-per-process singleton, and on
//! silicon the child's closing file descriptor fires the driver's cleanup write.

#[cfg(not(feature = "silicon"))]
use burn_tt::attach;
use burn_tt::{Fidelity, SrcRoute, TtDevice};

/// Core models may stage creation data and layouts, but all arithmetic stays resident.
pub fn assert_native_model(report: &burn_tt::Report) {
    assert!(
        report.0.iter().any(|s| s.on_device > 0),
        "model did not execute on the device"
    );
    for op in &report.0 {
        assert!(op.hand_written, "unsupported method was called: {}", op.op);
        assert_eq!(op.staged, 0, "{} staged model computation", op.op);
        assert!(
            op.on_host == 0
                || matches!(
                    op.op,
                    "float_from_data"
                        | "float_random"
                        | "int_from_data"
                        | "int_random"
                        | "bool_from_data"
                        | "bool_zeros"
                        | "bool_ones"
                        | "float_reshape"
                        | "int_reshape"
                        | "bool_reshape"
                ),
            "{} computed a model result on the host",
            op.op
        );
        assert!(
            op.downloads == 0 || op.op.ends_with("_into_data") || op.op == "tr_execute",
            "{} downloaded a model operand",
            op.op
        );
    }
}

/// How the device multiplies.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub elementwise: burn_tt::ElementwiseMode,
    pub route: SrcRoute,
    pub fidelity: Fidelity,
    /// Simulated cycles per role per run (`tt_kernels::runtime::run`).
    pub budget: u64,
    /// The Tensix tiles the device computes on; `None` is the gate tile
    /// (`harness::tensix_tile`), or on silicon what `TT_TILES` says.
    pub tiles: Option<burn_tt::TileChoice>,
    /// Matmuls accumulate in exact F32 (`burn_tt::MatmulMode::Exact`) instead of on the FPU.
    /// The default follows `TT_EXACT_MATMUL=1`, which is how `cargo xtask conformance --exact`
    /// selects its profile.
    pub exact_matmul: bool,
    /// Matmuls run as integer slices with shared exponents on the FPU at `Fidelity::Lo`
    /// (`burn_tt::MatmulMode::Accurate`). The default follows `TT_ACCURATE_MATMUL=1`
    /// (`cargo xtask conformance --accurate`). Setting both this and `exact_matmul` is refused.
    pub accurate_matmul: bool,
}

impl Default for Config {
    /// What a training backend runs: TF32 operands, all four fidelity phases.
    fn default() -> Self {
        Config {
            elementwise: burn_tt::ElementwiseMode::Sfpu,
            route: SrcRoute::Tf32FromFp32,
            fidelity: Fidelity::HiFi4,
            budget: 4_000_000,
            tiles: None,
            exact_matmul: std::env::var("TT_EXACT_MATMUL").as_deref() == Ok("1"),
            accurate_matmul: std::env::var("TT_ACCURATE_MATMUL").as_deref() == Ok("1"),
        }
    }
}

impl Config {
    fn matmul_mode(&self) -> burn_tt::MatmulMode {
        assert!(
            !(self.exact_matmul && self.accurate_matmul),
            "exact_matmul and accurate_matmul are both set: choose one matmul mode"
        );
        if self.exact_matmul {
            burn_tt::MatmulMode::Exact
        } else if self.accurate_matmul {
            burn_tt::MatmulMode::Accurate
        } else {
            burn_tt::MatmulMode::Fast
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
    assert_eq!(
        config.elementwise,
        burn_tt::ElementwiseMode::Sfpu,
        "matrix mode is unsupported on mesh engines"
    );
    assert!(
        !config.exact_matmul && !config.accurate_matmul,
        "exact and accurate matmul modes are unsupported on mesh engines"
    );
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
) -> Result<burn_tt::AttachGuard, burn_tt::TtError> {
    use burn_tt::{MeshEngine, TtError};
    use tt_device::Device;
    use tt_kernels::shard::{Chip, Fabric};
    let (lib, table): (_, &[_]) = match chips {
        2 => (tt_ttsim::x2_lib_path(), &crate::topology::BH_X2),
        4 => (tt_ttsim::x4_lib_path(), &crate::topology::BH_X4),
        n => return Err(TtError::Device(format!("no {n}-chip simulator build"))),
    };
    attach(device, move |serve| {
        let err = |e: &dyn std::fmt::Display| TtError::Device(e.to_string());
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
        let mut engine = MeshEngine::new(fabric, config.route, config.fidelity, config.budget)?;
        serve.serve(&mut engine);
        Ok(())
    })
}

#[cfg(feature = "silicon")]
fn attach_mesh(
    device: TtDevice,
    config: Config,
    chips: usize,
) -> Result<burn_tt::AttachGuard, burn_tt::TtError> {
    burn_tt::attach_with(
        device,
        burn_tt::TtConfig::default()
            .topology(burn_tt::Topology::Cards {
                cards: (0..chips as u16).collect(),
                compute: crate::backend::GATE_TILE,
                relay: crate::backend::RELAY_TILE,
            })
            .route(config.route)
            .fidelity(config.fidelity),
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
) -> Result<burn_tt::AttachGuard, burn_tt::TtError> {
    attach(device, sim_factory(config))
}

/// Make the simulator what `burn_tt` opens for a device nobody attached
/// (`burn_tt::set_default_engine`), so a gate can use `TtDevice::default()`
/// with no `attach`, as a user's code does. Call it inside the
/// `tt_ttsim::fork_scope` child that runs the gate: the simulator is a
/// once-per-process singleton, and a parent that attached it could not fork.
#[cfg(not(feature = "silicon"))]
pub fn register_simulator(config: Config) {
    burn_tt::set_default_engine(move |_device, _burn_config| Ok(Box::new(sim_factory(config))));
}

/// The simulator engine: the same `Session` the silicon engine uses, served
/// by [`burn_tt::attach`] or the registry.
#[cfg(not(feature = "silicon"))]
fn sim_factory(
    config: Config,
) -> impl FnOnce(burn_tt::Serve) -> Result<(), burn_tt::TtError> + Send + 'static {
    use burn_tt::{Engine, TtError};
    use tt_device::Device;

    // The same `Session` the silicon engine uses (`burn_tt::attach_with`), so
    // the reduced run's golden exercises the resident path on both targets.
    struct Sim<'a> {
        session: tt_kernels::session::Session<tt_ttsim::LibTtsim<'a>>,
        config: Config,
        buffers: burn_tt::DramBuffers,
    }
    impl Engine for Sim<'_> {
        fn elementwise_mode(&self) -> burn_tt::ElementwiseMode {
            self.config.elementwise
        }
        fn matmul_mode(&self) -> burn_tt::MatmulMode {
            self.config.matmul_mode()
        }
        fn math_mode(&self) -> burn_tt::MathMode {
            self.session.math_mode()
        }
        fn set_math_mode(&mut self, mode: burn_tt::MathMode) -> Result<(), TtError> {
            self.session.set_math_mode(mode);
            Ok(())
        }
        fn pool_bf16(
            &mut self,
            a: burn_tt::BufferId,
            windows: &[Vec<[usize; 2]>],
            divisors: &[usize],
            dims: [usize; 2],
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers
                .pool_bf16(&mut self.session, a, windows, divisors, dims)
        }
        fn matmul_bf16(
            &mut self,
            a: burn_tt::BufferId,
            b: burn_tt::BufferId,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.matmul_bf16(
                &mut self.session,
                a,
                b,
                self.config.fidelity,
                self.config.budget,
            )
        }
        fn upload_bf16(
            &mut self,
            bits: &[u16],
            rows: usize,
            cols: usize,
        ) -> Result<burn_tt::BufferId, TtError> {
            self.buffers
                .upload_bf16(&mut self.session, bits, rows, cols)
        }
        fn download_bf16(&mut self, id: burn_tt::BufferId) -> Result<Vec<u16>, TtError> {
            self.buffers.download_bf16(&mut self.session, id)
        }
        fn cast_float(
            &mut self,
            id: burn_tt::BufferId,
            bf16: bool,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.cast_float(&mut self.session, id, bf16)
        }
        fn cast_f16(
            &mut self,
            id: burn_tt::BufferId,
            to_f16: bool,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.cast_f16(&mut self.session, id, to_f16)
        }
        fn cast_bfp(
            &mut self,
            id: burn_tt::BufferId,
            format: Option<tt_kernels::bfp::BfpFormat>,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.cast_bfp(&mut self.session, id, format)
        }
        fn matmul_bfp(
            &mut self,
            a: burn_tt::BufferId,
            b: burn_tt::BufferId,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers
                .matmul_bfp(&mut self.session, a, b, self.config.fidelity)
        }
        fn supports_dram(&self) -> bool {
            true
        }
        fn upload(&mut self, v: &[f32], r: usize, c: usize) -> Result<burn_tt::BufferId, TtError> {
            self.buffers.upload(&mut self.session, v, r, c)
        }
        fn metadata(
            &mut self,
            bits: &[u32],
            dims: [usize; 2],
            elem: burn_tt::Elem,
        ) -> Result<burn_tt::BufferId, TtError> {
            self.buffers.metadata(&mut self.session, bits, dims, elem)
        }
        fn download(&mut self, id: burn_tt::BufferId) -> Result<Vec<f32>, TtError> {
            self.buffers.download(&mut self.session, id)
        }
        fn upload_bits(
            &mut self,
            v: &[u32],
            r: usize,
            c: usize,
            elem: burn_tt::Elem,
        ) -> Result<burn_tt::BufferId, TtError> {
            self.buffers.upload_bits(&mut self.session, v, r, c, elem)
        }
        fn download_bits(&mut self, id: burn_tt::BufferId) -> Result<Vec<u32>, TtError> {
            self.buffers.download_bits(&mut self.session, id)
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
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
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
        fn matmul_dram_at(
            &mut self,
            a: burn_tt::BufferId,
            ta: bool,
            b: burn_tt::BufferId,
            tb: bool,
            fidelity: burn_tt::Fidelity,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            let c = &self.config;
            self.buffers
                .matmul(&mut self.session, a, ta, b, tb, c.route, fidelity, c.budget)
        }
        fn matmul_dram_batched_at(
            &mut self,
            a: burn_tt::BufferId,
            b: burn_tt::BufferId,
            items: &[(burn_tt::Block, burn_tt::Block)],
            mkn: [usize; 3],
            fidelity: burn_tt::Fidelity,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            let c = &self.config;
            self.buffers.matmul_batched(
                &mut self.session,
                a,
                b,
                items,
                mkn,
                c.route,
                fidelity,
                c.budget,
                tt_kernels::session::BatchedLayout::STACKED,
            )
        }
        fn eltwise(
            &mut self,
            kind: u32,
            scalar: f32,
            a: burn_tt::BufferId,
            b: Option<burn_tt::BufferId>,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.eltwise(&mut self.session, kind, scalar, a, b)
        }
        fn eltwise_op(
            &mut self,
            op: tt_kernels::tensor::Eltwise,
            a: burn_tt::BufferId,
            b: Option<burn_tt::BufferId>,
            c: Option<burn_tt::BufferId>,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.eltwise_op(&mut self.session, op, a, b, c)
        }
        fn eltwise_blocks(
            &mut self,
            op: tt_kernels::tensor::Eltwise,
            a: burn_tt::BlockArg,
            b: Option<burn_tt::BlockArg>,
            c: Option<burn_tt::BlockArg>,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.eltwise_blocks(&mut self.session, op, a, b, c)
        }
        fn pow(
            &mut self,
            x: burn_tt::BufferId,
            y: burn_tt::PowArg,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.pow(&mut self.session, x, y)
        }
        fn sum_rows(
            &mut self,
            a: burn_tt::BufferId,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.sum_rows(&mut self.session, a)
        }
        fn reduce(
            &mut self,
            a: burn_tt::BufferId,
            op: tt_kernels::sfpu::reduce::ReduceOp,
            axis: tt_kernels::sfpu::reduce::Axis,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.reduce(&mut self.session, a, op, axis)
        }
        fn rms(
            &mut self,
            a: burn_tt::BufferId,
            eps: f32,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.rms(&mut self.session, a, eps)
        }
        fn scan(
            &mut self,
            a: burn_tt::BufferId,
            op: tt_kernels::sfpu::scan::ScanOp,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.scan(&mut self.session, a, op)
        }
        fn sort_planes(
            &mut self,
            a: burn_tt::BufferId,
            spec: tt_kernels::sfpu::sort::Spec,
        ) -> Result<burn_tt::SortedBuffers, TtError> {
            self.buffers.sort_planes(&mut self.session, a, spec)
        }
        fn slice_rows(
            &mut self,
            a: burn_tt::BufferId,
            first: usize,
            rows: usize,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.slice_rows(a, first, rows)
        }
        fn matmul_dram_batched(
            &mut self,
            a: burn_tt::BufferId,
            b: burn_tt::BufferId,
            items: &[(burn_tt::Block, burn_tt::Block)],
            mkn: [usize; 3],
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            let c = &self.config;
            self.buffers.matmul_batched(
                &mut self.session,
                a,
                b,
                items,
                mkn,
                c.route,
                c.fidelity,
                c.budget,
                tt_kernels::session::BatchedLayout::STACKED,
            )
        }
        fn supports_batched_layout(&self) -> bool {
            true
        }
        fn matmul_dram_batched_layout(
            &mut self,
            a: burn_tt::BufferId,
            b: burn_tt::BufferId,
            items: &[(burn_tt::Block, burn_tt::Block)],
            mkn: [usize; 3],
            layout: tt_kernels::session::BatchedLayout,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            let c = &self.config;
            self.buffers.matmul_batched(
                &mut self.session,
                a,
                b,
                items,
                mkn,
                c.route,
                c.fidelity,
                c.budget,
                layout,
            )
        }
        fn repack(
            &mut self,
            a: burn_tt::BufferId,
            sources: &[[usize; 2]],
            dims: [usize; 2],
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.repack(&mut self.session, a, sources, dims)
        }
        fn copy_blocks(
            &mut self,
            a: burn_tt::BufferId,
            moves: &[burn_tt::BlockMove],
            dims: [usize; 2],
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.copy_blocks(&mut self.session, a, moves, dims)
        }
        fn zeros_dram(
            &mut self,
            dims: [usize; 2],
            elem: tt_kernels::tensor::Elem,
            bf16: bool,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.zeros(&mut self.session, dims, elem, bf16)
        }
        fn random_dram(
            &mut self,
            dims: [usize; 2],
            draw: burn_tt::Draw,
            base: u64,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.random(&mut self.session, dims, draw, base)
        }
        fn gather_indexed(
            &mut self,
            input: burn_tt::BufferId,
            indices: burn_tt::BufferId,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers
                .gather_indexed(&mut self.session, input, indices)
        }
        fn select_rows(
            &mut self,
            table: burn_tt::BufferId,
            indices: burn_tt::BufferId,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.select_rows(&mut self.session, table, indices)
        }
        fn select_taps(
            &mut self,
            table: burn_tt::BufferId,
            indices: burn_tt::BufferId,
            zero_row: bool,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers
                .select_taps(&mut self.session, table, indices, zero_row)
        }
        fn repack_many(
            &mut self,
            inputs: &[burn_tt::BufferId],
            sources: &[(usize, [usize; 2])],
            dims: [usize; 2],
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers
                .repack_many(&mut self.session, inputs, sources, dims)
        }
        fn gather_rows(
            &mut self,
            sources: &[burn_tt::BufferId],
            rows: &[(usize, usize)],
            cols: usize,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers
                .gather_rows(&mut self.session, sources, rows, cols)
        }
        fn rows_add(
            &mut self,
            t: burn_tt::BufferId,
            indices: &[usize],
            value: burn_tt::BufferId,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.rows_add(&mut self.session, t, indices, value)
        }
        fn assign_rows(
            &mut self,
            t: burn_tt::BufferId,
            value: burn_tt::BufferId,
            rows: &[(usize, usize)],
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers.assign_rows(&mut self.session, t, value, rows)
        }
        fn scatter_rows(
            &mut self,
            table: burn_tt::BufferId,
            indices: burn_tt::BufferId,
            values: burn_tt::BufferId,
            column: bool,
        ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
            self.buffers
                .scatter_rows(&mut self.session, table, indices, values, column)
        }
        fn device_traffic(&mut self) -> Option<tt_device::Traffic> {
            Some(self.session.device().traffic())
        }
        fn begin_trace(&mut self) -> Result<(), TtError> {
            self.buffers.begin_trace(&mut self.session)
        }
        fn end_trace(&mut self) -> Result<u64, TtError> {
            self.buffers.end_trace(&mut self.session)
        }
        fn run_trace(
            &mut self,
            trace: u64,
            input: burn_tt::BufferId,
            values: &[f32],
            output: burn_tt::BufferId,
        ) -> Result<burn_tt::TraceRun, TtError> {
            self.buffers
                .run_trace(&mut self.session, trace, input, values, output)
        }
        fn release_trace(&mut self, trace: u64) {
            self.buffers.release_trace(&mut self.session, trace)
        }
        fn set_op_label(&mut self, label: &'static str) {
            self.session.set_op_label(label);
        }
        fn trace_ops(&mut self, trace: u64) -> Result<Vec<tt_kernels::trace::OpRecord>, TtError> {
            self.buffers.trace_ops(&self.session, trace)
        }
        fn trace_footprint(
            &mut self,
            trace: u64,
        ) -> Result<Vec<Option<tt_kernels::trace::UnitFootprint>>, TtError> {
            self.buffers.trace_footprint(&self.session, trace)
        }
        fn copy_into(
            &mut self,
            src: burn_tt::BufferId,
            dst: burn_tt::BufferId,
        ) -> Result<(), TtError> {
            self.buffers.copy_into(&mut self.session, src, dst)
        }
        fn run_generic_trace(
            &mut self,
            trace: u64,
            inputs: &[(burn_tt::BufferId, burn_tt::InputPayload)],
            outputs: &[(burn_tt::BufferId, burn_tt::OutputKind)],
        ) -> Result<burn_tt::GenericTraceRun, TtError> {
            self.buffers
                .run_generic_trace(&mut self.session, trace, inputs, outputs)
        }
        fn matmul(&mut self, a: &[f32], b: &[f32], mkn: [usize; 3]) -> Result<Vec<f32>, TtError> {
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

    move |serve| {
        let mut sim = tt_ttsim::Simulator::open()
            .map_err(|e| TtError::Device(format!("could not open the simulator: {e}")))?;
        let dev = Device::open(sim.transport()).map_err(|e| TtError::Device(e.to_string()))?;
        let t = crate::harness::tensix_tile();
        let session = tt_kernels::session::Session::open(
            dev,
            tt_firmware_images::ROLES,
            config
                .tiles
                .unwrap_or(tt_kernels::session::TileChoice::Exactly(t.x(), t.y())),
            |_, _| Ok(None),
        )
        .map_err(|e| TtError::Device(e.to_string()))?;
        let mut session = session;
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .map_err(|e| TtError::Device(e.to_string()))?;
        serve.serve(&mut Sim {
            session,
            config,
            buffers: burn_tt::DramBuffers::default().with_elementwise_mode(config.elementwise),
        });
        Ok(())
    }
}

#[cfg(feature = "silicon")]
fn attach_engine(
    device: TtDevice,
    config: Config,
) -> Result<burn_tt::AttachGuard, burn_tt::TtError> {
    // `TT_TOPOLOGY` ("0", "0,1") picks the cards, so one benchmark runs on one
    // card or several unchanged (`burn_tt::Topology`); unset, the card this
    // gate was pointed at, on the gate tile. The gate's own `tiles`, else
    // `TT_TILES` ("8", "all"), spreads one card's work over more tiles.
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
    let topology = match config.tiles.map(Ok).or_else(burn_tt::tiles_from_env) {
        Some(tiles) => topology.on_tiles(tiles?)?,
        None => topology,
    };
    eprintln!("topology: {topology:?}");
    burn_tt::attach_with(
        device,
        burn_tt::TtConfig::default()
            .topology(topology)
            .route(config.route)
            .fidelity(config.fidelity)
            .elementwise(config.elementwise)
            .matmul(config.matmul_mode()),
    )
}
