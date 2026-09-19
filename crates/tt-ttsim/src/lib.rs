//! A safe, singleton-enforcing wrapper over `libttsim.so`.
//!
//! **This crate is a development and test dependency. It must never appear in the
//! dependency graph of a shipped artifact** — CI enforces that with a `cargo tree`
//! check. Binding the simulator does not compromise the native-Rust goal the way
//! an FFI dependency on a production library would, but only as long as it stays
//! out of shipped builds.
//!
//! # Why this wrapper exists
//!
//! `libttsim.so` has two properties that make direct use hazardous:
//!
//! 1. **It is a process-wide singleton** with no context handle. Its state lives in
//!    file-scope globals, it is single-threaded and non-reentrant, and `init` may be
//!    called exactly once per process. `cargo test`'s thread-per-test default would
//!    otherwise corrupt it in ways that look exactly like simulator bugs.
//! 2. **Every contract violation terminates the process** via `_Exit` — no return
//!    code, no unwinding, no destructors, no panic hook. A misaligned access does
//!    not fail, it ends the test run.
//!
//! The answers are [`Simulator`], which can only be obtained once and is neither
//! `Send` nor `Sync`, and [`fork_scope`], which runs a closure in a forked child so
//! that a fatal error becomes a legible test failure rather than a vanished runner.
//!
//! Property (2) is also why [`transport`] validates every access against the
//! simulator's decode map before making the call. By the time a bad address reaches
//! `libttsim`, there is no one left to report it to.

pub mod fork;
pub mod transport;

pub use fork::{fork_scope, ForkError};
pub use transport::LibTtsim;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use tt_ttsim_sys::{Lib, LoadError};

/// Set once `Simulator::open` has succeeded. Never cleared: `libttsim_init` has no
/// re-init path, so a second open would be a fatal error inside the library.
static OPENED: AtomicBool = AtomicBool::new(false);

/// Holds the loaded library for the life of the process.
///
/// Leaked deliberately rather than stored in a `OnceLock<Lib>`: the DMA callbacks
/// below are plain `extern "C"` functions that may run at any point after `init`,
/// and a `'static` reference is the simplest way to guarantee the library outlives
/// them. There is exactly one per process, so this leaks a fixed, bounded amount.
static mut LIB: Option<&'static Lib> = None;

/// The environment variable that overrides which `libttsim*.so` is loaded.
pub const LIB_PATH_ENV: &str = "TT_TTSIM_LIB";

#[derive(Debug)]
pub enum OpenError {
    /// [`Simulator::open`] was already called in this process.
    AlreadyOpen,
    /// The `.so` could not be found on disk.
    NotFound(PathBuf),
    /// The `.so` could not be loaded or was missing an entry point.
    Load(LoadError),
    /// The simulator came up, but did not report a Blackhole.
    NotBlackhole { vendor: u16, device: u16 },
    /// A BAR base in configuration space did not match the address the library
    /// actually decodes. See [`expected_bar_base`].
    BarMismatch { bar: tt_device::Bar, expected: u64, found: u64 },
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::AlreadyOpen => write!(
                f,
                "libttsim was already initialized in this process; it is a singleton \
                 with no re-init path. Use fork_scope() to isolate test cases."
            ),
            OpenError::NotFound(p) => write!(
                f,
                "libttsim not found at {}. Run `cargo xtask fetch-ttsim`, or set {LIB_PATH_ENV}.",
                p.display()
            ),
            OpenError::Load(e) => write!(f, "{e}"),
            OpenError::NotBlackhole { vendor, device } => write!(
                f,
                "simulator reported {vendor:#06x}:{device:#06x}, not Blackhole. \
                 Is this libttsim_wh.so rather than libttsim_bh.so?"
            ),
            OpenError::BarMismatch { bar, expected, found } => write!(
                f,
                "{bar:?} base is {found:#x} in config space but this build expects \
                 {expected:#x}; libttsim's hardcoded BAR map has changed"
            ),
        }
    }
}

impl std::error::Error for OpenError {}

impl From<LoadError> for OpenError {
    fn from(e: LoadError) -> Self {
        OpenError::Load(e)
    }
}

/// A running simulator.
///
/// There is at most one of these per process, and it is neither `Send` nor `Sync`:
/// the library is single-threaded and non-reentrant, so the token stays on the
/// thread that created it.
pub struct Simulator {
    lib: &'static Lib,
    /// Makes `Simulator` `!Send + !Sync` without an unstable negative impl.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Simulator {
    /// Load and initialize the simulator.
    ///
    /// Succeeds at most once per process; the second call returns
    /// [`OpenError::AlreadyOpen`] rather than letting `libttsim_init` abort.
    ///
    /// The path comes from `$TT_TTSIM_LIB` if set, otherwise `vendor/libttsim_bh.so`
    /// beside the workspace root.
    pub fn open() -> Result<Self, OpenError> {
        match std::env::var_os(LIB_PATH_ENV) {
            Some(p) => Self::open_path(PathBuf::from(p)),
            None => Self::open_path(default_lib_path()),
        }
    }

    /// Load and initialize the simulator from a specific `.so`.
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self, OpenError> {
        let path = path.as_ref();

        // Claim the singleton before touching the library, so a losing racer never
        // reaches `libttsim_init`.
        if OPENED.swap(true, Ordering::SeqCst) {
            return Err(OpenError::AlreadyOpen);
        }
        if !path.exists() {
            return Err(OpenError::NotFound(path.to_path_buf()));
        }

        // SAFETY: the path names a libttsim.so, which we are about to verify by
        // resolving all ten entry points and reading a known device ID from it.
        let lib: &'static Lib = Box::leak(Box::new(unsafe { Lib::load(path) }?));

        // SAFETY: single-threaded, and this is the one point in the process where
        // LIB is written — before `init`, and therefore before any callback can run.
        unsafe { LIB = Some(lib) };

        // Callbacks must be installed *before* init; the library rejects them
        // afterwards. Install them even though the baseline performs no DMA:
        // DMA with no callbacks installed is itself a fatal error, and ours at
        // least says what happened.
        //
        // SAFETY: both pointers are `extern "C"` functions with the required
        // signatures, and the simulator is not yet running.
        unsafe { (lib.set_pci_dma_mem_callbacks)(dma_read_unexpected, dma_write_unexpected) };

        // SAFETY: callbacks are installed and this is the first and only init.
        unsafe { (lib.init)() };

        let sim = Simulator { lib, _not_send: std::marker::PhantomData };
        sim.verify()?;
        Ok(sim)
    }

    /// Check that the simulator is the Blackhole build, and that its BAR map is the
    /// one this crate's validation layer was written against.
    fn verify(&self) -> Result<(), OpenError> {
        use tt_device::{Bar, ConfigOffset, DEVICE_ID_BLACKHOLE, VENDOR_ID_TENSTORRENT};

        // SAFETY: offset 0 is always decoded; the simulator is initialized.
        let id = unsafe { (self.lib.pci_config_rd32)(BDF_CHIP0, ConfigOffset::VendorDevice as u32) };
        let vendor = (id & 0xFFFF) as u16;
        let device = (id >> 16) as u16;
        if vendor != VENDOR_ID_TENSTORRENT || device != DEVICE_ID_BLACKHOLE {
            return Err(OpenError::NotBlackhole { vendor, device });
        }

        // The validation layer in `transport` decides whether an access is legal by
        // comparing against libttsim's *hardcoded* decode map, not against whatever
        // config space reports. Confirm the two still agree: if a future libttsim
        // moves a BAR, every subsequent access would be validated against the wrong
        // map and then terminate the process. Better to refuse to start.
        for bar in [Bar::Bar0, Bar::Bar2, Bar::Bar4] {
            let lo_off = bar.config_offset() as u32;
            // SAFETY: BAR offsets are decoded; the simulator is initialized.
            let lo = unsafe { (self.lib.pci_config_rd32)(BDF_CHIP0, lo_off) };
            let hi = unsafe { (self.lib.pci_config_rd32)(BDF_CHIP0, lo_off + 4) };
            let found = ((hi as u64) << 32) | ((lo & !0xF) as u64);
            let expected = expected_bar_base(bar);
            if found != expected {
                return Err(OpenError::BarMismatch { bar, expected, found });
            }
        }
        Ok(())
    }

    /// Borrow this simulator as a [`Transport`](tt_device::Transport).
    pub fn transport(&mut self) -> LibTtsim<'_> {
        LibTtsim::new(self.lib)
    }

    /// Advance simulated time.
    ///
    /// Nothing else does. The simulator has no free-running clock: outside a call
    /// to this function, the device is frozen.
    pub fn clock(&mut self, n_clocks: u32) {
        // SAFETY: initialized, single-threaded (enforced by `!Send`).
        unsafe { (self.lib.clock)(n_clocks) }
    }
}

/// The bus/device/function of the first (and, in the `bh` build, only) chip.
///
/// `function = bdf & 7`, `device = (bdf >> 3) & 0x1F`, `bus = (bdf >> 8) & 0xFF`.
/// Reading an absent BDF returns all-ones; *writing* one is fatal.
pub const BDF_CHIP0: u32 = 0;

/// The address at which `libttsim` decodes each BAR.
///
/// These are compile-time constants inside the library, not values derived from
/// configuration space. `libttsim_init` pre-programs config space to match, and
/// config-space BAR writes are accepted but change nothing — so relocating a BAR
/// the way a BIOS would makes config space lie while every memory access continues
/// to decode against the constants below, and the first access outside them
/// terminates the process.
pub const fn expected_bar_base(bar: tt_device::Bar) -> u64 {
    match bar {
        tt_device::Bar::Bar0 => 0x1_0000_0000,
        tt_device::Bar::Bar2 => 0x1_2000_0000,
        tt_device::Bar::Bar4 => 0x8_0000_0000,
    }
}

/// Where `cargo xtask fetch-ttsim` puts the library.
pub fn default_lib_path() -> PathBuf {
    workspace_root().join("vendor").join("libttsim_bh.so")
}

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <root>/crates/tt-ttsim.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate manifest is two levels below the workspace root")
        .to_path_buf()
}

/// Installed so that unexpected DMA produces a diagnosis rather than libttsim's
/// generic fatal error. The baseline performs no DMA; reaching here means the
/// device tried to touch host memory, which is a bug in the code that programmed it.
unsafe extern "C" fn dma_read_unexpected(paddr: u64, _dst: *mut core::ffi::c_void, size: u32) {
    fatal_unexpected_dma("read", paddr, size)
}

unsafe extern "C" fn dma_write_unexpected(paddr: u64, _src: *const core::ffi::c_void, size: u32) {
    fatal_unexpected_dma("write", paddr, size)
}

fn fatal_unexpected_dma(kind: &str, paddr: u64, size: u32) -> ! {
    // Cannot unwind out of an `extern "C"` callback into C, and there is no way to
    // tell libttsim "no". Report and stop.
    eprintln!(
        "tt-ttsim: device attempted an unexpected DMA {kind} of {size} bytes at host \
         physical address {paddr:#x}.\n\
         The baseline registers no host memory for DMA, so this means a NoC or PCIe \
         descriptor was programmed with a host address by mistake."
    );
    std::process::abort()
}
