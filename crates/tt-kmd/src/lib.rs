//! The silicon transport: `/dev/tenstorrent/N` via tt-kmd.
//!
//! The counterpart to `tt-ttsim`. `tt-device` is written entirely against the
//! [`Transport`] trait, so everything above it — the TLB window allocator, the
//! 96-bit configuration encoder, the shadow table, `load_and_start` — is the same
//! code on silicon as on the simulator, with no `#[cfg]` anywhere. This crate
//! supplies four methods and changes nothing else.
//!
//! # What is different about silicon, and where it is handled
//!
//! * **Time runs on its own.** [`Transport::tick`] is a no-op here, which the
//!   trait already anticipates. The consequence is not free: a poll loop with a
//!   budget denominated in simulated cycles spins for a few hundred microseconds
//!   of wall clock and then declares a timeout. Callers need a wall-clock budget.
//! * **Nothing is isolated.** The simulator hands out a fresh chip per test. Here
//!   the chip keeps whatever the last run left in it — `Dst` in particular has no
//!   power-on reset value (`Dst.md:15`) — so a harness must scrub deliberately.
//! * **Other processes exist.** TLB window ownership is arbitrated by the driver
//!   ([`Kmd::allocate_tlb`]), and the soft-reset register's read-modify-write has
//!   no cross-process protection at all ([`Kmd::lock`]).
//! * **A crash leaves the chip running.** [`Kmd::set_cleanup_write`] registers a
//!   NoC write the *driver* performs when this file descriptor closes, however it
//!   closes.

pub mod abi;
pub mod ioctl;
pub mod mapping;

use std::fs::File;
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use mapping::Mapping;
use tt_device::{Bar, ConfigOffset, Transport, TransportError};
use tt_isa::noc::ChipId;

/// `TENSTORRENT_DRIVER_VERSION` at the pinned tag, mirrored from `PINS.toml`.
///
/// Checked against the loaded module at [`Kmd::open`]. A matching header is no
/// evidence about the module that is actually running.
pub const PINNED_API_VERSION: u32 = 2;

/// Where the device nodes live.
pub const DEV_DIR: &str = "/dev/tenstorrent";

/// One Blackhole, reached through one `/dev/tenstorrent/N`.
pub struct Kmd {
    fd: File,
    /// `/sys/bus/pci/devices/<bdf>/config`, the only readable path to PCI
    /// configuration space: tt-kmd exposes no config-space ioctl, and
    /// `/sys/.../resourceN` is root-only. The first 64 bytes are world-readable,
    /// which is exactly the range [`ConfigOffset`] is closed over.
    config: File,
    bdf: String,
    chip: ChipId,
    bar0: Mapping,
    bar2: Mapping,
    bar4: Mapping,
}

/// A TLB window index reserved from the driver.
///
/// Held so the driver will not hand the same index to another process. The
/// *configuration* of the window is still written by `tt_device::tlb`, through
/// BAR0, so the encoder that the simulator gates cover is the one silicon runs.
pub struct TlbReservation {
    pub index: u32,
    pub mmap_offset_uc: u64,
    pub mmap_offset_wc: u64,
}

impl Kmd {
    /// Open `/dev/tenstorrent/<index>`.
    pub fn open(index: u16) -> Result<Self, TransportError> {
        Self::open_path(Path::new(DEV_DIR).join(index.to_string()), ChipId(index))
    }

    /// Every Blackhole this host can see, in device-node order.
    pub fn enumerate() -> Result<Vec<u16>, TransportError> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(DEV_DIR)? {
            let entry = entry?;
            if let Some(n) = entry.file_name().to_str().and_then(|s| s.parse().ok()) {
                out.push(n);
            }
        }
        out.sort_unstable();
        Ok(out)
    }

    fn open_path(path: PathBuf, chip: ChipId) -> Result<Self, TransportError> {
        // Deliberately *without* O_APPEND. tt-kmd reads the open flags as a power
        // policy (`ioctl.h:363-380`): without O_APPEND the driver requests high
        // power immediately, with it the state starts at zero and every power
        // feature -- including TT_POWER_FLAG_TENSIX_ENABLE, whose zero means the
        // Tensix array is clock gated -- must be asked for explicitly. Opening
        // with O_APPEND and forgetting that produces a chip on which nothing runs
        // and nothing says why.
        let fd = File::options().read(true).write(true).open(&path)?;

        let info = ioctl::get_driver_info(fd.as_fd())?;
        if info.driver_version != PINNED_API_VERSION {
            return Err(TransportError::Io(std::io::Error::other(format!(
                "{} speaks tt-kmd ioctl API version {}, but tt-kmd is built against {}. \
                 The structs in `tt_kmd::abi` describe a different ABI than the loaded \
                 module; re-pin [tt-kmd] in PINS.toml against `modinfo tenstorrent`.",
                path.display(),
                info.driver_version,
                PINNED_API_VERSION,
            ))));
        }

        let dev = ioctl::get_device_info(fd.as_fd())?;
        let bdf = format!(
            "{:04x}:{:02x}:{:02x}.{}",
            dev.pci_domain,
            dev.bus_dev_fn >> 8,
            (dev.bus_dev_fn >> 3) & 0x1F,
            dev.bus_dev_fn & 0x7,
        );
        let config = File::open(format!("/sys/bus/pci/devices/{bdf}/config"))?;

        let mappings = ioctl::query_mappings(fd.as_fd(), 6)?;
        let find = |id: u32, bar: Bar| -> Result<Mapping, TransportError> {
            let m = mappings
                .iter()
                .find(|m| m.mapping_id == id)
                .ok_or_else(|| {
                    std::io::Error::other(format!("{} reports no mapping {id}", path.display()))
                })?;
            // The driver reports the BAR's true length. If it disagrees with the
            // size this workspace has compiled in, every offset computed above
            // this trait is measured against the wrong aperture -- so refuse
            // rather than truncate.
            if m.mapping_size != bar.size() {
                return Err(TransportError::Io(std::io::Error::other(format!(
                    "{} reports {bar:?} as {} bytes; tt-device is built for {}",
                    path.display(),
                    m.mapping_size,
                    bar.size(),
                ))));
            }
            Ok(Mapping::new(
                fd.as_fd(),
                m.mapping_base,
                m.mapping_size as usize,
            )?)
        };

        // Uncached, not write-combining. The TLB configuration array and the NIU
        // registers both live in BAR0 and are documented as dword-only registers;
        // write-combining is free to merge and reorder stores, which for a
        // write-only configuration register is silent corruption. A WC aperture
        // for bulk L1 staging is a Phase 9 question, not a bring-up one.
        let bar0 = find(abi::mapping_id::RESOURCE0_UC, Bar::Bar0)?;
        let bar2 = find(abi::mapping_id::RESOURCE1_UC, Bar::Bar2)?;
        let bar4 = find(abi::mapping_id::RESOURCE2_UC, Bar::Bar4)?;

        Ok(Kmd {
            fd,
            config,
            bdf,
            chip,
            bar0,
            bar2,
            bar4,
        })
    }

    /// The PCI address of this card, as sysfs spells it.
    pub fn bdf(&self) -> &str {
        &self.bdf
    }

    fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    fn bar(&self, bar: Bar) -> &Mapping {
        match bar {
            Bar::Bar0 => &self.bar0,
            Bar::Bar2 => &self.bar2,
            Bar::Bar4 => &self.bar4,
        }
    }

    /// Reserve a TLB window index from the driver.
    pub fn allocate_tlb(&self, size: u64) -> Result<TlbReservation, TransportError> {
        let a = ioctl::allocate_tlb(self.fd(), size)?;
        Ok(TlbReservation {
            index: a.out_id,
            mmap_offset_uc: a.out_mmap_offset_uc,
            mmap_offset_wc: a.out_mmap_offset_wc,
        })
    }

    pub fn free_tlb(&self, index: u32) -> Result<(), TransportError> {
        Ok(ioctl::free_tlb(self.fd(), index)?)
    }

    /// Program a window through the driver rather than through
    /// `tt_device::tlb::write_config`.
    ///
    /// Present as an oracle, not as the runtime path: configuring the same window
    /// both ways and comparing is how the hand-written 96-bit encoder earns
    /// silicon evidence.
    pub fn configure_tlb_via_driver(
        &self,
        index: u32,
        addr: u64,
        x_end: u16,
        y_end: u16,
        noc: u8,
        ordering: u8,
    ) -> Result<(), TransportError> {
        Ok(ioctl::configure_tlb(
            self.fd(),
            index,
            addr,
            x_end,
            y_end,
            noc,
            ordering,
        )?)
    }

    /// Ask the driver to perform `data -> (x, y, addr)` when this fd closes.
    ///
    /// Intended to hold every baby RISC-V in reset, so that a test which segfaults
    /// or is killed cannot leave a core running against the next one.
    pub fn set_cleanup_write(
        &self,
        x: u8,
        y: u8,
        noc: u8,
        addr: u64,
        data: u32,
    ) -> Result<(), TransportError> {
        Ok(ioctl::set_noc_cleanup(
            self.fd(),
            true,
            x,
            y,
            noc,
            addr,
            data,
        )?)
    }

    /// Take one of the driver's arbitrated locks; `false` if another process holds it.
    ///
    /// `Device::set_core_reset` is a read-modify-write on a register with no
    /// atomic bit operations, and its own documentation says `&mut self` "is not
    /// enough ... if anything else on the host can touch the same tile". On
    /// silicon something else can.
    pub fn lock(&self, index: u8) -> Result<bool, TransportError> {
        Ok(ioctl::lock_ctl(self.fd(), index, abi::lock::ACQUIRE)?)
    }

    pub fn unlock(&self, index: u8) -> Result<bool, TransportError> {
        Ok(ioctl::lock_ctl(self.fd(), index, abi::lock::RELEASE)?)
    }

    /// Read one of the driver's sysfs telemetry attributes, e.g. `tt_heartbeat`.
    ///
    /// The cheap health check: no ioctl, no NoC traffic, world-readable.
    pub fn telemetry(&self, name: &str) -> Result<String, TransportError> {
        let p = format!("/sys/bus/pci/devices/{}/tenstorrent", self.bdf);
        let dir = std::fs::read_dir(&p)?
            .next()
            .ok_or_else(|| std::io::Error::other(format!("{p} is empty")))??;
        Ok(std::fs::read_to_string(dir.path().join(name))?
            .trim()
            .to_string())
    }
}

impl Transport for Kmd {
    fn bar_read(&mut self, bar: Bar, offset: u64, dst: &mut [u8]) -> tt_device::Result<()> {
        let end = offset
            .checked_add(dst.len() as u64)
            .ok_or(TransportError::OutOfBounds {
                bar,
                offset,
                len: dst.len() as u64,
            })?;
        if end > bar.size() {
            return Err(TransportError::OutOfBounds {
                bar,
                offset,
                len: dst.len() as u64,
            });
        }
        self.bar(bar).read(offset as usize, dst);
        Ok(())
    }

    fn bar_write(&mut self, bar: Bar, offset: u64, src: &[u8]) -> tt_device::Result<()> {
        let end = offset
            .checked_add(src.len() as u64)
            .ok_or(TransportError::OutOfBounds {
                bar,
                offset,
                len: src.len() as u64,
            })?;
        if end > bar.size() {
            return Err(TransportError::OutOfBounds {
                bar,
                offset,
                len: src.len() as u64,
            });
        }
        self.bar(bar).write(offset as usize, src);
        Ok(())
    }

    fn config_read32(&mut self, offset: ConfigOffset) -> tt_device::Result<u32> {
        let mut buf = [0u8; 4];
        self.config.read_exact_at(&mut buf, offset as u64)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Nothing to do: silicon advances time by itself.
    fn tick(&mut self, _n: u32) {}

    fn chip(&self) -> ChipId {
        self.chip
    }
}
