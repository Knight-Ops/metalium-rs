//! The tt-kmd userspace ABI, transcribed from the pinned `ioctl.h`.
//!
//! Hand-written rather than bindgen'd, for one reason: this is a dozen small
//! structs, and `crates/tt-kmd/tests/abi_layout.rs` compiles the *real header*
//! and checks every size and field offset against what is written here. That is
//! a stronger check than a diff of generated code — it compares against the C
//! compiler's own opinion of the layout — and it costs no build-time tool.
//!
//! Every item cites its line in `vendor/ioctl.h` at the pinned tag
//! (`PINS.toml` `[tt-kmd]`), which is byte-identical to the DKMS source of the
//! module this host has loaded.

/// `TENSTORRENT_IOCTL_MAGIC` (`ioctl.h:12`).
const MAGIC: u64 = 0xFA;

/// `_IO(type, nr)` for a Linux ioctl with no argument-size encoding.
///
/// Every tt-kmd ioctl is declared with the bare `_IO` macro rather than
/// `_IOR`/`_IOW`, so direction and size are both zero and the request number is
/// just `(type << 8) | nr`. Spelled out rather than hardcoded so the numbers in
/// the specification (`0xFA0B` and friends) are derived, not transcribed.
const fn io(nr: u64) -> u64 {
    (MAGIC << 8) | nr
}

// `ioctl.h:14-31`.
pub const GET_DEVICE_INFO: u64 = io(0);
pub const QUERY_MAPPINGS: u64 = io(2);
pub const GET_DRIVER_INFO: u64 = io(5);
pub const PIN_PAGES: u64 = io(7);
pub const LOCK_CTL: u64 = io(8);
pub const UNPIN_PAGES: u64 = io(10);
pub const ALLOCATE_TLB: u64 = io(11);
pub const FREE_TLB: u64 = io(12);
pub const CONFIGURE_TLB: u64 = io(13);
pub const SET_NOC_CLEANUP: u64 = io(14);

// Deliberately absent: RESET_DEVICE (6). Resets go through `tt-smi -r` so that
// there is one path, one log, and one place where the fact that every open file
// descriptor becomes permanently invalid is handled. See `docs/hardware-access.md`.

/// `tenstorrent_mapping.mapping_id` values (`ioctl.h:35-40`).
///
/// "These are not array indices" — the header says so explicitly, and the
/// resource numbering is PCI BAR 0, 2, 4 rather than 0, 1, 2.
pub mod mapping_id {
    pub const RESOURCE0_UC: u32 = 1;
    pub const RESOURCE0_WC: u32 = 2;
    pub const RESOURCE1_UC: u32 = 3;
    pub const RESOURCE1_WC: u32 = 4;
    pub const RESOURCE2_UC: u32 = 5;
    pub const RESOURCE2_WC: u32 = 6;
}

/// `tenstorrent_lock_ctl_in.flags` (`ioctl.h:216-219`).
pub mod lock {
    pub const ACQUIRE: u32 = 0;
    pub const RELEASE: u32 = 1;
}

#[repr(C)]
#[derive(Default)]
pub struct GetDeviceInfo {
    pub in_output_size_bytes: u32,
    pub out_output_size_bytes: u32,
    pub vendor_id: u16,
    pub device_id: u16,
    pub subsystem_vendor_id: u16,
    pub subsystem_id: u16,
    /// `[0:2]` function, `[3:7]` device, `[8:15]` bus (`ioctl.h:57`).
    pub bus_dev_fn: u16,
    pub max_dma_buf_size_log2: u16,
    pub pci_domain: u16,
    pub reserved: u16,
}

#[repr(C)]
#[derive(Default)]
pub struct GetDriverInfo {
    pub in_output_size_bytes: u32,
    pub out_output_size_bytes: u32,
    /// The IOCTL API version — `TENSTORRENT_DRIVER_VERSION`, not the module's
    /// release number. This is the field that decides whether the structs below
    /// describe the module that is actually loaded.
    pub driver_version: u32,
    pub driver_version_major: u8,
    pub driver_version_minor: u8,
    pub driver_version_patch: u8,
    pub reserved0: u8,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct Mapping {
    pub mapping_id: u32,
    pub reserved: u32,
    pub mapping_base: u64,
    pub mapping_size: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct QueryMappingsIn {
    pub output_mapping_count: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Default)]
pub struct AllocateTlb {
    pub in_size: u64,
    pub in_reserved: u64,
    pub out_id: u32,
    pub out_reserved0: u32,
    pub out_mmap_offset_uc: u64,
    pub out_mmap_offset_wc: u64,
    pub out_reserved1: u64,
}

/// `tenstorrent_pin_pages_in` followed by `tenstorrent_pin_pages_out_extended`
/// (`ioctl.h:175-189`): the driver writes `in_output_size_bytes` of output,
/// so asking for 16 gets the NoC address as well as the IOVA.
#[repr(C)]
#[derive(Default)]
pub struct PinPages {
    pub in_output_size_bytes: u32,
    pub in_flags: u32,
    pub in_virtual_address: u64,
    pub in_size: u64,
    pub out_physical_address: u64,
    pub out_noc_address: u64,
}

/// `tenstorrent_pin_pages_in.flags` (`ioctl.h:169-173`).
pub mod pin {
    /// The caller attests the pages are physically contiguous.
    pub const CONTIGUOUS: u32 = 1;
    /// Map the pages for the card's NoC (through the outbound iATU) and
    /// return the NoC address that reaches them.
    pub const NOC_DMA: u32 = 2;
    pub const NOC_TOP_DOWN: u32 = 4;
    /// The card only reads them; the IOMMU enforces it.
    pub const READ_ONLY: u32 = 8;
}

/// `tenstorrent_unpin_pages` (`ioctl.h:192-204`).
#[repr(C)]
#[derive(Default)]
pub struct UnpinPages {
    pub in_virtual_address: u64,
    pub in_size: u64,
    pub in_reserved: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct FreeTlb {
    pub in_id: u32,
}

/// `tenstorrent_noc_tlb_config` (`ioctl.h:300-313`).
#[repr(C)]
#[derive(Default)]
pub struct NocTlbConfig {
    pub addr: u64,
    pub x_end: u16,
    pub y_end: u16,
    pub x_start: u16,
    pub y_start: u16,
    pub noc: u8,
    pub mcast: u8,
    pub ordering: u8,
    pub linked: u8,
    pub static_vc: u8,
    pub reserved0: [u8; 3],
    pub reserved1: [u32; 2],
}

#[repr(C)]
#[derive(Default)]
pub struct ConfigureTlb {
    pub in_id: u32,
    pub in_reserved: u32,
    pub in_config: NocTlbConfig,
    pub out_reserved: u64,
}

/// `tenstorrent_set_noc_cleanup` (`ioctl.h:344-353`).
///
/// A NoC write the *driver* performs when this file descriptor is closed, however
/// it is closed. It is the only crash-safety mechanism that survives a segfault
/// or the OOM killer, because it does not rely on our process running any code.
#[repr(C)]
#[derive(Default)]
pub struct SetNocCleanup {
    pub argsz: u32,
    pub flags: u32,
    pub enabled: u8,
    pub x: u8,
    pub y: u8,
    pub noc: u8,
    pub reserved0: u32,
    pub addr: u64,
    pub data: u64,
}

#[repr(C)]
#[derive(Default)]
pub struct LockCtl {
    pub in_output_size_bytes: u32,
    pub in_flags: u32,
    pub in_index: u8,
    pub in_reserved: [u8; 3],
    pub out_value: u8,
    pub out_reserved: [u8; 3],
}
