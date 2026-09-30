//! Owning a chip for compute: the bring-up order, and the reset between runs.
//!
//! Established by the silicon campaign in `tt-tests/src/backend.rs` and moved
//! here so a backend gets the same sequence the gates do. Every step exists
//! because its absence cost a run, and in two cases the host:
//!
//! 1. **Read the Tensix grid from the ARC before touching any Tensix tile.** A
//!    fused-off tile does not reject an access; nothing answers, the NoC hangs,
//!    and the chip reset that recovers it drops the PCIe link (divergence row 35).
//! 2. **Refuse a tile this chip does not have**, as an error.
//! 3. **Register the driver's crash-cleanup write**, now that its target is known
//!    to exist -- registered against a fused-off tile it would fire on every close.
//! 4. **Reset the tile**: hold every core, pulse the Tensix backend, and put the
//!    per-thread state (`Config`, `ThreadConfig`, RWCs, ADCs) back to zero, which
//!    nothing outside the tile can do and which otherwise outlives the program
//!    that set it (rows 47 and 49).
//!
//! On the simulator steps 1, 3 and 4 have nothing to do: ttsim models an
//! unharvested chip, has no driver, starts every tile at zero, and refuses both
//! the backend reset bits (row 16) and `STATE_RESET_EN` (row 49).

use tt_device::tlb::WindowKind;
use tt_device::{Device, Transport, TransportError};
use tt_isa::noc::grid::Tensix;
use tt_isa::noc::{Noc0, NocCoord};
use tt_isa::tensix::{self, Core};

use crate::datapath;
use crate::dm::DataMover;
use crate::matmul::{self, Fidelity, SrcRoute};
use crate::runtime::{self, Kernel, Resident, RoleImages, RunError, Schedule};
use crate::tensor::{self, DramAlloc, DramTensor, TensorError};

/// Every baby RISC-V held in reset: what the cleanup write leaves behind, and
/// the resting state between runs.
///
/// Assembled from `Core`'s own bits so it cannot drift from `set_core_reset`.
pub const ALL_BABIES_HELD: u32 = Core::B.soft_reset_mask()
    | Core::T0.soft_reset_mask()
    | Core::T1.soft_reset_mask()
    | Core::T2.soft_reset_mask()
    | Core::NC.soft_reset_mask();

/// Simulated cycles each role may take in the reset program. On silicon the
/// runner's one-second floor applies.
const RESET_BUDGET: u64 = 400_000;

/// Which of this chip's Tensix tiles a session computes on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileChoice {
    /// This one, or an error if the chip does not have it.
    Exactly(u8, u8),
    /// The first surviving tile in translated order (lowest column, then row).
    First,
}

/// Why a session could not be opened.
#[derive(Debug)]
pub enum SessionError {
    Transport(TransportError),
    /// The requested tile is fused off (or never existed) on this chip.
    NoSuchTile {
        x: u8,
        y: u8,
        columns: Vec<u8>,
    },
    /// The reset program did not run.
    Reset(RunError),
}

impl From<TransportError> for SessionError {
    fn from(e: TransportError) -> Self {
        SessionError::Transport(e)
    }
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Transport(e) => write!(f, "{e}"),
            SessionError::NoSuchTile { x, y, columns } => write!(
                f,
                "({x},{y}) is not a Tensix tile on this chip: it has {} Tensix columns, \
                 so X must be one of {columns:?}",
                columns.len()
            ),
            SessionError::Reset(e) => write!(f, "resetting the tile: {e}"),
        }
    }
}

impl std::error::Error for SessionError {}

/// This chip's Tensix grid, asked of the ARC.
///
/// Touches only the ARC tile, whose coordinate means the same thing whether or
/// not translation is on, so it is safe before anything else is known. On the
/// simulator, [`Tensix::FULL`]: ttsim models an unharvested chip (row 35) and no
/// telemetry.
pub fn tensix_grid<T: Transport>(dev: &mut Device<T>) -> Result<Tensix, TransportError> {
    if dev.transport().is_simulated() {
        return Ok(Tensix::FULL);
    }
    let w = dev.alloc_window(WindowKind::TwoMib)?;
    dev.tensix_grid(&w)
}

/// Hold every core on `tile` and pulse its Tensix backend, leaving the backend
/// released and the cores held.
///
/// Holding the cores alone is not enough: a program that hangs the coprocessor
/// leaves the backend stuck after its core is stopped. Entering reset aborts
/// in-flight unpacker, packer, Matrix Unit and Vector Unit work, resets `Src`
/// bank ownership and zeroes the THCON configuration (`SoftReset.md`). `Dst`
/// survives it. A no-op on the simulator (row 16).
pub fn reset_tile<T: Transport>(
    dev: &mut Device<T>,
    tile: NocCoord<Noc0>,
) -> Result<(), TransportError> {
    if dev.transport().is_simulated() {
        return Ok(());
    }
    let w = dev.alloc_window(WindowKind::TwoMib)?;
    dev.write32(
        &w,
        tile,
        tensix::SOFT_RESET_0,
        ALL_BABIES_HELD | tensix::BACKEND_RESET_MASK,
    )?;
    dev.write32(&w, tile, tensix::SOFT_RESET_0, ALL_BABIES_HELD)
}

/// Zero `Config` and all three threads' per-thread state on `tile`
/// ([`datapath::thread_state_reset`]). A no-op on the simulator, which starts
/// every run from zero and refuses parts of the program (rows 36 and 49).
pub fn reset_thread_state<T: Transport>(
    dev: &mut Device<T>,
    tile: NocCoord<Noc0>,
    images: &RoleImages<'_>,
) -> Result<(), RunError> {
    if dev.transport().is_simulated() {
        return Ok(());
    }
    let program = datapath::thread_state_reset();
    // Thread 0 first releases anything a failed kernel left blocked on a
    // semaphore, which the backend pulse does not (row 65); otherwise thread 1
    // or 2's reset would queue behind it forever.
    let first = [datapath::release_semaphores(), program.clone()].concat();
    let kernel = Kernel {
        dump_rows: 0,
        ..Kernel::new([&first, &program, &program], Schedule::InOrder)
    };
    runtime::run(dev, tile, images, &kernel, RESET_BUDGET).map(|_| ())
}

/// `A[m,k] @ B[k,n]`, row-major, on `tile`, in as many runs as it takes
/// ([`matmul::matmul_chunked`]), each preceded by the tile reset
/// ([`reset_tile`], [`reset_thread_state`]) so no run inherits another's state.
#[allow(clippy::too_many_arguments)]
pub fn matmul_on<T: Transport>(
    dev: &mut Device<T>,
    tile: NocCoord<Noc0>,
    images: &RoleImages<'_>,
    a: &[f32],
    b: &[f32],
    mkn: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    budget: u64,
) -> Result<Vec<f32>, RunError> {
    matmul::matmul_chunked(a, b, mkn, route, fidelity, |a, b, mkn| {
        reset_tile(dev, tile)?;
        reset_thread_state(dev, tile, images)?;
        matmul::matmul(dev, tile, images, a, b, mkn, route, fidelity, budget)
    })
}

/// A chip opened for compute on one Tensix tile, in the order the module
/// documentation gives.
pub struct Session<T: Transport> {
    dev: Device<T>,
    tile: NocCoord<Noc0>,
    grid: Tensix,
    images: RoleImages<'static>,
    /// The role images, resident since the last [`Session::prepare`]
    /// (`runtime::Resident`): one reset and one load per session rather than per
    /// run. `None` only between a failed run and the next `prepare`.
    resident: Option<Resident<Noc0>>,
    /// Every run's phases since the last [`Session::take_profile`].
    profile: runtime::Profile,
    /// GDDR, once [`Session::enable_dram`] has been called.
    dram: Option<DramState>,
}

/// What a session needs to keep tensors in GDDR: the chip's channels, an
/// allocator, the data mover on this session's tile, and a window for each
/// size of access.
struct DramState {
    alloc: DramAlloc,
    dram: tt_isa::dram::Dram,
    image: &'static [u8],
    mover: Option<DataMover<Noc0>>,
    w: tt_device::Window,
    w4: tt_device::Window,
}

impl<T: Transport> Session<T> {
    /// Bring `dev` up for compute on the tile `choice` names.
    ///
    /// `register_cleanup` is step 3: given the transport and the chosen tile, it
    /// arranges for [`ALL_BABIES_HELD`] to be written to the tile's
    /// `SOFT_RESET_0` however the process ends. It is called only once the tile
    /// is known to exist. [`Session::open_card`] supplies the driver's.
    pub fn open(
        mut dev: Device<T>,
        images: RoleImages<'static>,
        choice: TileChoice,
        register_cleanup: impl FnOnce(&mut T, NocCoord<Noc0>) -> Result<(), TransportError>,
    ) -> Result<Self, SessionError> {
        let grid = tensix_grid(&mut dev)?;
        let (x, y) = match choice {
            TileChoice::Exactly(x, y) => (x, y),
            TileChoice::First => {
                let t = grid
                    .tiles::<Noc0>()
                    .min_by_key(|t| (t.x(), t.y()))
                    .expect("a chip with no Tensix tiles would have failed telemetry");
                (t.x(), t.y())
            }
        };
        if !grid.contains(x, y) {
            return Err(SessionError::NoSuchTile {
                x,
                y,
                columns: grid.columns().collect(),
            });
        }
        let tile = NocCoord::new(x, y).expect("a grid tile is a NoC coordinate");
        register_cleanup(dev.transport(), tile)?;
        let mut session = Session {
            dev,
            tile,
            grid,
            images,
            resident: None,
            profile: runtime::Profile::default(),
            dram: None,
        };
        session.prepare().map_err(SessionError::Reset)?;
        Ok(session)
    }

    /// Put the tile back to a known state: step 4 again, then the role images
    /// loaded and left resident.
    pub fn prepare(&mut self) -> Result<(), RunError> {
        if let Some(r) = self.resident.take() {
            r.stop(&mut self.dev, &self.images)?;
        }
        reset_tile(&mut self.dev, self.tile)?;
        reset_thread_state(&mut self.dev, self.tile, &self.images)?;
        self.resident = Some(Resident::start(
            &mut self.dev,
            self.tile,
            &self.images,
            RESET_BUDGET,
        )?);
        // The reset held RISCV B too; GDDR contents survive, the mover does not.
        if let Some(d) = &mut self.dram {
            d.mover = None;
        }
        Ok(())
    }

    /// Keep tensors in GDDR from now on: read the chip's channels and set up an
    /// allocator. `dm_image` is `tt_firmware_images::DM_B`'s bytes, started on
    /// this session's tile's RISCV B when first needed.
    pub fn enable_dram(&mut self, dm_image: &'static [u8]) -> Result<(), TransportError> {
        if self.dram.is_some() {
            return Ok(());
        }
        let w = self.dev.alloc_window(WindowKind::TwoMib)?;
        let w4 = self.dev.alloc_window(WindowKind::FourGib)?;
        let dram = self.dev.dram_grid(&w)?;
        self.dram = Some(DramState {
            alloc: DramAlloc::new(&dram),
            dram,
            image: dm_image,
            mover: None,
            w,
            w4,
        });
        Ok(())
    }

    fn dram_state(&mut self) -> Result<&mut DramState, TensorError> {
        self.dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled on this session".into()))
    }

    /// Upload a row-major `[rows, cols]` matrix to GDDR.
    pub fn upload(
        &mut self,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<DramTensor, TensorError> {
        let Session { dev, dram, .. } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        DramTensor::upload(dev, &d.w4, &mut d.alloc, values, rows, cols)
    }

    /// Download a tensor to row-major values.
    pub fn download(&mut self, t: &DramTensor) -> Result<Vec<f32>, TensorError> {
        let Session { dev, dram, .. } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        t.download(dev, &d.w4)
    }

    /// Give a tensor's slots back.
    pub fn free(&mut self, t: DramTensor) -> Result<(), TensorError> {
        self.dram_state()?.alloc.free(&t.placement);
        Ok(())
    }

    /// Free GDDR bytes on the fullest channel.
    pub fn dram_free_bytes(&self) -> u64 {
        self.dram.as_ref().map_or(0, |d| d.alloc.free_bytes())
    }

    /// Element-wise `a (op) b` in GDDR ([`tensor::eltwise`]), on the data
    /// mover's FP32 unit. No Tensix run, so the resident roles are untouched.
    pub fn eltwise(
        &mut self,
        op: tensor::Eltwise,
        a: &DramTensor,
        b: Option<&DramTensor>,
    ) -> Result<DramTensor, TensorError> {
        let tile = self.tile;
        let Session { dev, dram, .. } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        if d.mover.is_none() {
            d.mover = Some(DataMover::start(dev, &d.w, tile, &d.dram, d.image)?);
        }
        let mover = d.mover.as_mut().expect("started above");
        let out = tensor::eltwise(dev, &d.w, mover, &mut d.alloc, op, a, b);
        if out.is_err() {
            d.mover = None;
        }
        out
    }

    /// `op(A) @ op(B)` with every operand and the result in GDDR
    /// ([`tensor::matmul_dram`]), on the resident roles.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_dram(
        &mut self,
        a: &DramTensor,
        a_transposed: bool,
        b: &DramTensor,
        b_transposed: bool,
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<DramTensor, TensorError> {
        if self.resident.is_none() {
            self.prepare()?;
        }
        let tile = self.tile;
        let Session {
            dev,
            dram,
            resident,
            images,
            profile,
            ..
        } = self;
        let d = dram
            .as_mut()
            .ok_or_else(|| TensorError::Shape("GDDR is not enabled".into()))?;
        if d.mover.is_none() {
            d.mover = Some(DataMover::start(dev, &d.w, tile, &d.dram, d.image)?);
        }
        let mover = d.mover.as_mut().expect("started above");
        let r = resident.as_mut().expect("prepared above");
        let out = tensor::matmul_dram(
            dev,
            &d.w,
            mover,
            &mut d.alloc,
            a,
            a_transposed,
            b,
            b_transposed,
            route,
            fidelity,
            |dev, k| {
                let o = r.run(dev, images, k, budget)?;
                profile.phases.extend_from_slice(&o.profile.phases);
                Ok(o)
            },
        );
        if out.is_err() {
            // As `run`: leave the session usable.
            self.resident = None;
            if let Some(d) = &mut self.dram {
                d.mover = None;
            }
            if let Err(e) = self.prepare() {
                eprintln!("session: recovery after a failed kernel failed as well: {e}");
            }
        }
        out
    }

    /// Run `kernel` on the resident roles. A failure resets the tile and
    /// restarts them before it is returned, so the session stays usable.
    pub fn run(&mut self, kernel: &Kernel<'_>, budget: u64) -> Result<runtime::Outcome, RunError> {
        if self.resident.is_none() {
            self.prepare()?;
        }
        let r = self.resident.as_mut().expect("prepared above");
        let out = r.run(&mut self.dev, &self.images, kernel, budget);
        if let Ok(o) = &out {
            self.profile.phases.extend_from_slice(&o.profile.phases);
        }
        if out.is_err() {
            self.resident = None;
            // The kernel's error is the one to report; a recovery that fails
            // too leaves `resident` empty, and the next run tries again.
            if let Err(e) = self.prepare() {
                eprintln!("session: recovery after a failed kernel failed as well: {e}");
            }
        }
        out
    }

    /// `A[m,k] @ B[k,n]`, row-major, on this session's tile, chunked as
    /// [`matmul_on`] chunks it but on the resident roles: no reset or image load
    /// between chunks. Bit-identical to `matmul_on` (`step17_resident`).
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        mkn: [usize; 3],
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<Vec<f32>, RunError> {
        matmul::matmul_chunked(a, b, mkn, route, fidelity, |a, b, mkn| {
            matmul::matmul_with(a, b, mkn, route, fidelity, |k| self.run(k, budget))
        })
    }

    /// The phases of every run since the last call, and forget them.
    pub fn take_profile(&mut self) -> runtime::Profile {
        std::mem::take(&mut self.profile)
    }

    pub fn device(&mut self) -> &mut Device<T> {
        &mut self.dev
    }

    pub fn tile(&self) -> NocCoord<Noc0> {
        self.tile
    }

    pub fn grid(&self) -> &Tensix {
        &self.grid
    }

    pub fn images(&self) -> &RoleImages<'static> {
        &self.images
    }

    /// The device, with the role cores held again.
    pub fn into_device(mut self) -> Device<T> {
        if let Some(r) = self.resident.take() {
            let _ = r.stop(&mut self.dev, &self.images);
        }
        self.dev
    }
}

impl Session<tt_kmd::Kmd> {
    /// Open `/dev/tenstorrent/{index}` for compute, with the driver's cleanup
    /// write as step 3.
    pub fn open_card(
        index: u16,
        images: RoleImages<'static>,
        choice: TileChoice,
    ) -> Result<Self, SessionError> {
        let kmd = tt_kmd::Kmd::open(index)?;
        let dev = Device::open(kmd)?;
        Self::open(dev, images, choice, |kmd, tile| {
            kmd.set_cleanup_write(tile.x(), tile.y(), 0, tensix::SOFT_RESET_0, ALL_BABIES_HELD)
        })
    }
}
