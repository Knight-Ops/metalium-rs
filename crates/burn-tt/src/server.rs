//! One server thread per attached device, owning its hardware.
//!
//! Burn requires tensors and devices to be `Send + Sync + 'static` and calls
//! ops from whatever thread it likes. The hardware is neither: a
//! `tt_device::Device` takes `&mut self` for everything, `Device<Kmd>` is not
//! `Sync`, and the simulator is a `!Send` process-wide singleton. So each
//! device's hardware lives on a thread of its own, created by [`attach`], and
//! ops send it jobs and block on the answer. The thread is also where the
//! hardware is *created*: [`attach`] runs the caller's factory there, which is
//! what lets a `!Send` simulator serve at all.
//!
//! Dropping the [`AttachGuard`] closes the job queue, waits for the thread, and
//! so drops the hardware -- which for silicon returns the chip to idle
//! (`tt_device::Device`'s `Drop`). One `Device` per chip applies here as
//! everywhere: attaching a chip twice is refused.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;
use std::thread::JoinHandle;

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, SessionError, TileChoice};

use crate::TtDevice;

/// What a device can do for the backend, on its server thread.
pub trait Engine {
    /// `A[m, k] @ B[k, n]`, row-major.
    fn matmul(&mut self, a: &[f32], b: &[f32], mkn: [usize; 3]) -> Result<Vec<f32>, EngineError>;

    /// Can this engine keep tensors on the device (Phase 9)? If not, every
    /// tensor stays on the host and only `matmul` runs on the device.
    fn supports_dram(&self) -> bool {
        false
    }
    /// Put a row-major `[rows, cols]` matrix on the device.
    fn upload(
        &mut self,
        _values: &[f32],
        _rows: usize,
        _cols: usize,
    ) -> Result<BufferId, EngineError> {
        Err(unsupported())
    }
    /// Read one back, row-major.
    fn download(&mut self, _id: BufferId) -> Result<Vec<f32>, EngineError> {
        Err(unsupported())
    }
    /// Forget one.
    fn free(&mut self, _id: BufferId) {}
    /// `op(A) @ op(B)`, each operand transposed if asked, result left on the
    /// device, as `(id, [rows, cols])`.
    fn matmul_dram(
        &mut self,
        _a: BufferId,
        _a_transposed: bool,
        _b: BufferId,
        _b_transposed: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
    /// Element-wise `a (kind) b` or `a (kind) scalar` (`tt_isa::dm::kind`),
    /// result left on the device.
    fn eltwise(
        &mut self,
        _kind: u32,
        _scalar: f32,
        _a: BufferId,
        _b: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        Err(unsupported())
    }
}

fn unsupported() -> EngineError {
    EngineError("this engine keeps no tensors on the device".into())
}

/// A tensor kept on the device, by the engine's own numbering.
pub type BufferId = u64;

/// The device-resident tensors of one engine: a [`Session`]'s `DramTensor`s by
/// id. The engines built on a `Session` -- [`KmdEngine`], and the simulator's in
/// `tt-tests` -- forward their DRAM methods here.
#[derive(Default)]
pub struct DramBuffers {
    next: BufferId,
    live: HashMap<BufferId, tt_kernels::tensor::DramTensor>,
}

impl DramBuffers {
    pub fn upload<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<BufferId, EngineError> {
        let t = s
            .upload(values, rows, cols)
            .map_err(|e| EngineError(e.to_string()))?;
        self.next += 1;
        self.live.insert(self.next, t);
        Ok(self.next)
    }

    pub fn download<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        id: BufferId,
    ) -> Result<Vec<f32>, EngineError> {
        let t = self.get(id)?;
        s.download(t).map_err(|e| EngineError(e.to_string()))
    }

    pub fn free<T: tt_device::Transport>(&mut self, s: &mut Session<T>, id: BufferId) {
        if let Some(t) = self.live.remove(&id) {
            let _ = s.free(t);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        a: BufferId,
        a_transposed: bool,
        b: BufferId,
        b_transposed: bool,
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let (ta, tb) = (self.get(a)?.clone(), self.get(b)?.clone());
        let c = s
            .matmul_dram(
                &ta,
                a_transposed,
                &tb,
                b_transposed,
                route,
                fidelity,
                budget,
            )
            .map_err(|e| EngineError(e.to_string()))?;
        let dims = [c.rows, c.cols];
        self.next += 1;
        self.live.insert(self.next, c);
        Ok((self.next, dims))
    }

    pub fn eltwise<T: tt_device::Transport>(
        &mut self,
        s: &mut Session<T>,
        kind: u32,
        scalar: f32,
        a: BufferId,
        b: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let ta = self.get(a)?.clone();
        let tb = b.map(|b| self.get(b).cloned()).transpose()?;
        let op = tt_kernels::tensor::Eltwise { kind, scalar };
        let c = s
            .eltwise(op, &ta, tb.as_ref())
            .map_err(|e| EngineError(e.to_string()))?;
        let dims = [c.rows, c.cols];
        self.next += 1;
        self.live.insert(self.next, c);
        Ok((self.next, dims))
    }

    /// How many are live.
    pub fn len(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    fn get(&self, id: BufferId) -> Result<&tt_kernels::tensor::DramTensor, EngineError> {
        self.live
            .get(&id)
            .ok_or_else(|| EngineError(format!("no device buffer {id}")))
    }
}

/// Why an engine could not start or finish a job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineError(pub String);

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EngineError {}

impl From<tt_kernels::runtime::RunError> for EngineError {
    fn from(e: tt_kernels::runtime::RunError) -> Self {
        EngineError(e.to_string())
    }
}

impl From<SessionError> for EngineError {
    fn from(e: SessionError) -> Self {
        EngineError(e.to_string())
    }
}

type Job = Box<dyn FnOnce(&mut dyn Engine) + Send>;

/// Handed to an [`attach`] factory: call [`Serve::serve`] with the engine once
/// it exists, and it runs the device's jobs until the device is detached.
pub struct Serve {
    ready: Sender<Result<(), EngineError>>,
    jobs: Receiver<Job>,
}

impl Serve {
    /// Serve jobs on `engine` until the [`AttachGuard`] is dropped.
    pub fn serve(self, engine: &mut dyn Engine) {
        let _ = self.ready.send(Ok(()));
        while let Ok(job) = self.jobs.recv() {
            job(engine);
        }
    }
}

struct Attached {
    jobs: Sender<Job>,
}

static ATTACHED: Mutex<Option<HashMap<TtDevice, Attached>>> = Mutex::new(None);

fn with_attached<R>(f: impl FnOnce(&mut HashMap<TtDevice, Attached>) -> R) -> R {
    let mut guard = ATTACHED.lock().unwrap_or_else(|p| p.into_inner());
    f(guard.get_or_insert_with(HashMap::new))
}

/// How many devices are attached.
pub(crate) fn attached_count() -> usize {
    with_attached(|a| a.len())
}

/// Keeps `device` attached; dropping it detaches the device and drops its
/// hardware.
#[must_use = "dropping the guard detaches the device at once"]
pub struct AttachGuard {
    device: TtDevice,
    thread: Option<JoinHandle<()>>,
}

impl Drop for AttachGuard {
    fn drop(&mut self) {
        // Removing the sender closes the queue; the server's `recv` fails and
        // `serve` returns, and so does the factory, dropping the hardware.
        with_attached(|a| a.remove(&self.device));
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Attach `device`: start its server thread and run `factory` on it.
///
/// `factory` builds the engine -- opening the hardware -- and calls
/// [`Serve::serve`] with it. It returns an error, rather than serving, if the
/// hardware cannot be opened; `attach` returns that error. Attaching a device
/// that is already attached is refused.
pub fn attach<F>(device: TtDevice, factory: F) -> Result<AttachGuard, EngineError>
where
    F: FnOnce(Serve) -> Result<(), EngineError> + Send + 'static,
{
    let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = mpsc::channel();
    let already = with_attached(|a| match a.entry(device) {
        std::collections::hash_map::Entry::Occupied(_) => true,
        std::collections::hash_map::Entry::Vacant(v) => {
            v.insert(Attached { jobs: jobs_tx });
            false
        }
    });
    if already {
        return Err(EngineError(format!("{device} is already attached")));
    }
    let failed = ready_tx.clone();
    let thread = std::thread::Builder::new()
        .name(format!("burn-tt {device}"))
        .spawn(move || {
            let serve = Serve {
                ready: ready_tx,
                jobs: jobs_rx,
            };
            match factory(serve) {
                Ok(()) => {
                    let _ = failed.send(Err(EngineError(
                        "the engine factory returned without serving".into(),
                    )));
                }
                Err(e) => {
                    let _ = failed.send(Err(e));
                }
            }
        })
        .map_err(|e| EngineError(format!("could not start the server thread: {e}")))?;
    let guard = AttachGuard {
        device,
        thread: Some(thread),
    };
    match ready_rx.recv() {
        Ok(Ok(())) => Ok(guard),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(EngineError("the server thread died before serving".into())),
    }
}

/// Run `job` on `device`'s engine and wait for it.
///
/// # Panics
///
/// If `device` is not attached, or its server thread has gone: Burn's ops
/// cannot return an error, and quietly computing on the host instead would
/// make the device path unobservable.
pub(crate) fn run<R: Send + 'static>(
    device: TtDevice,
    job: impl FnOnce(&mut dyn Engine) -> R + Send + 'static,
) -> R {
    let (tx, rx) = mpsc::channel();
    let sender = with_attached(|a| a.get(&device).map(|d| d.jobs.clone()));
    let Some(sender) = sender else {
        panic!("{device} is not attached: call burn_tt::attach before running device ops on it");
    };
    sender
        .send(Box::new(move |engine| {
            let _ = tx.send(job(engine));
        }))
        .unwrap_or_else(|_| panic!("{device}'s server thread has stopped"));
    rx.recv()
        .unwrap_or_else(|_| panic!("{device}'s server thread stopped during a job"))
}

/// [`run`], timed by `kind` (`crate::traffic::device_time`).
fn timed_run<R: Send + 'static>(
    kind: &'static str,
    device: TtDevice,
    job: impl FnOnce(&mut dyn Engine) -> R + Send + 'static,
) -> R {
    crate::traffic::timed(kind, || run(device, job))
}

/// `A[m, k] @ B[k, n]` on `device`, panicking on a device error.
pub(crate) fn matmul(device: TtDevice, a: &[f32], b: &[f32], mkn: [usize; 3]) -> Vec<f32> {
    let (a, b) = (a.to_vec(), b.to_vec());
    timed_run("matmul_host", device, move |engine| {
        engine.matmul(&a, &b, mkn)
    })
    .unwrap_or_else(|e| panic!("matmul {mkn:?} on {device}: {e}"))
}

/// Does `device`'s engine keep tensors on the device? Asked once per device.
pub(crate) fn supports_dram(device: TtDevice) -> bool {
    static KNOWN: Mutex<Option<HashMap<TtDevice, bool>>> = Mutex::new(None);
    if let Some(&k) = KNOWN
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_or_insert_with(HashMap::new)
        .get(&device)
    {
        return k;
    }
    let k = run(device, |engine| engine.supports_dram());
    KNOWN
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(device, k);
    k
}

/// Upload, panicking on a device error.
pub(crate) fn upload(device: TtDevice, values: Vec<f32>, rows: usize, cols: usize) -> BufferId {
    crate::traffic::uploaded(values.len() * 4);
    timed_run("upload", device, move |engine| {
        engine.upload(&values, rows, cols)
    })
    .unwrap_or_else(|e| panic!("upload [{rows}, {cols}] to {device}: {e}"))
}

/// Download, panicking on a device error.
pub(crate) fn download(device: TtDevice, id: BufferId) -> Vec<f32> {
    let v = timed_run("download", device, move |engine| engine.download(id))
        .unwrap_or_else(|e| panic!("download {id} from {device}: {e}"));
    crate::traffic::downloaded(v.len() * 4);
    v
}

/// Free, without waiting; nothing to do if the device has gone.
pub(crate) fn free(device: TtDevice, id: BufferId) {
    if let Some(sender) = with_attached(|a| a.get(&device).map(|d| d.jobs.clone())) {
        let _ = sender.send(Box::new(move |engine| engine.free(id)));
    }
}

/// Element-wise on the device, panicking on a device error.
pub(crate) fn eltwise(
    device: TtDevice,
    kind: u32,
    scalar: f32,
    a: BufferId,
    b: Option<BufferId>,
) -> (BufferId, [usize; 2]) {
    timed_run("eltwise", device, move |engine| {
        engine.eltwise(kind, scalar, a, b)
    })
    .unwrap_or_else(|e| panic!("element-wise {kind} on {device}: {e}"))
}

/// `op(A) @ op(B)` on the device, panicking on a device error.
pub(crate) fn matmul_dram(
    device: TtDevice,
    a: BufferId,
    a_transposed: bool,
    b: BufferId,
    b_transposed: bool,
) -> (BufferId, [usize; 2]) {
    timed_run("matmul_dram", device, move |engine| {
        engine.matmul_dram(a, a_transposed, b, b_transposed)
    })
    .unwrap_or_else(|e| panic!("matmul on {device}: {e}"))
}

// --- Silicon ------------------------------------------------------------------

/// The silicon engine: a [`Session`] on `/dev/tenstorrent/N`.
pub struct KmdEngine {
    pub session: Session<tt_kmd::Kmd>,
    pub route: SrcRoute,
    pub fidelity: Fidelity,
    /// Per-role budget for each run; on silicon, a floor of one second applies
    /// (`tt_kernels::runtime::run`).
    pub budget: u64,
    /// Tensors kept in GDDR, if the session has it enabled.
    pub buffers: Option<DramBuffers>,
}

impl Engine for KmdEngine {
    fn matmul(&mut self, a: &[f32], b: &[f32], mkn: [usize; 3]) -> Result<Vec<f32>, EngineError> {
        Ok(self
            .session
            .matmul(a, b, mkn, self.route, self.fidelity, self.budget)?)
    }
    fn supports_dram(&self) -> bool {
        self.buffers.is_some()
    }
    fn upload(&mut self, v: &[f32], rows: usize, cols: usize) -> Result<BufferId, EngineError> {
        let b = self.buffers.as_mut().ok_or_else(unsupported)?;
        b.upload(&mut self.session, v, rows, cols)
    }
    fn download(&mut self, id: BufferId) -> Result<Vec<f32>, EngineError> {
        let b = self.buffers.as_mut().ok_or_else(unsupported)?;
        b.download(&mut self.session, id)
    }
    fn free(&mut self, id: BufferId) {
        if let Some(b) = self.buffers.as_mut() {
            b.free(&mut self.session, id);
        }
    }
    fn matmul_dram(
        &mut self,
        a: BufferId,
        ta: bool,
        b: BufferId,
        tb: bool,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.matmul(
            &mut self.session,
            a,
            ta,
            b,
            tb,
            self.route,
            self.fidelity,
            self.budget,
        )
    }
    fn eltwise(
        &mut self,
        kind: u32,
        scalar: f32,
        a: BufferId,
        b: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        let bufs = self.buffers.as_mut().ok_or_else(unsupported)?;
        bufs.eltwise(&mut self.session, kind, scalar, a, b)
    }
}

/// A factory for [`attach`] that opens `/dev/tenstorrent/{device.chip}` as a
/// [`Session`] computing on `tile` through `route` at `fidelity`.
pub fn kmd_engine(
    device: TtDevice,
    tile: TileChoice,
    route: SrcRoute,
    fidelity: Fidelity,
) -> impl FnOnce(Serve) -> Result<(), EngineError> + Send + 'static {
    move |serve| {
        let mut session = Session::open_card(device.chip, tt_firmware_images::ROLES, tile)?;
        // Tensors live in GDDR (Phase 9). Bit-identical to the host-staged path
        // (`step18_dram_matmul`), so on by default.
        session
            .enable_dram(tt_firmware_images::DM_B.1)
            .map_err(|e| EngineError(e.to_string()))?;
        let mut engine = KmdEngine {
            session,
            route,
            fidelity,
            budget: 400_000,
            buffers: Some(DramBuffers::default()),
        };
        serve.serve(&mut engine);
        Ok(())
    }
}

// --- Several chips --------------------------------------------------------------

/// An engine over a [`Fabric`](tt_kernels::shard::Fabric): every matmul split
/// along `N` across the fabric's chips, bit-identical to the single-chip
/// product. Attached as *one* Burn device -- the chips are how it computes, not
/// something Burn sees.
pub struct MeshEngine<T: tt_device::Transport> {
    pub fabric: tt_kernels::shard::Fabric<T>,
    pub route: SrcRoute,
    pub fidelity: Fidelity,
    pub budget: u64,
}

impl<T: tt_device::Transport> Engine for MeshEngine<T> {
    fn matmul(&mut self, a: &[f32], b: &[f32], mkn: [usize; 3]) -> Result<Vec<f32>, EngineError> {
        self.fabric
            .matmul(a, b, mkn, self.route, self.fidelity, self.budget)
            .map_err(|e| EngineError(e.to_string()))
    }
}

/// A factory for [`attach`] that opens every card in `cards` (the first is chip
/// 0, the host's way in and out), computes on `compute` and relays through
/// `relay` on each, finds the cabled links between them from the chips
/// (`tt_kernels::link::discover`), and serves a [`MeshEngine`].
pub fn kmd_mesh_engine(
    cards: Vec<u16>,
    compute: (u8, u8),
    relay: (u8, u8),
    route: SrcRoute,
    fidelity: Fidelity,
) -> impl FnOnce(Serve) -> Result<(), EngineError> + Send + 'static {
    use tt_device::tlb::WindowKind;
    use tt_kernels::shard::{Chip, Fabric};
    move |serve| {
        let e = |e: tt_device::TransportError| EngineError(e.to_string());
        let mut chips = Vec::new();
        for &card in &cards {
            let s = Session::open_card(
                card,
                tt_firmware_images::ROLES,
                TileChoice::Exactly(compute.0, compute.1),
            )?;
            if !s.grid().contains(relay.0, relay.1) {
                return Err(EngineError(format!(
                    "relay tile {relay:?} is fused off on card {card}"
                )));
            }
            let tile = s.tile();
            let relay = tt_isa::noc::NocCoord::new(relay.0, relay.1).expect("a Tensix coordinate");
            chips.push(Chip::new(s.into_device(), tile, relay).map_err(e)?);
        }
        let mut links = Vec::new();
        for p in 0..chips.len() {
            for q in p + 1..chips.len() {
                let (l, r) = chips.split_at_mut(q);
                let (a, b) = (&mut l[p].dev, &mut r[0].dev);
                let wa = a.alloc_window(WindowKind::TwoMib).map_err(e)?;
                let wb = b.alloc_window(WindowKind::TwoMib).map_err(e)?;
                let (ga, gb) = (
                    a.ethernet_grid(&wa).map_err(e)?,
                    b.ethernet_grid(&wb).map_err(e)?,
                );
                let found = tt_kernels::link::discover(a, &wa, &ga, b, &wb, &gb)
                    .map_err(|x| EngineError(x.to_string()))?;
                if let Some(link) = found.first() {
                    links.push((p, q, *link));
                }
            }
        }
        let fabric = Fabric::new(
            chips,
            &links,
            tt_firmware_images::ROLES,
            tt_firmware_images::ETH_E1,
        )
        .map_err(|x| EngineError(x.to_string()))?;
        serve.serve(&mut MeshEngine {
            fabric,
            route,
            fidelity,
            budget: 400_000,
        });
        Ok(())
    }
}
