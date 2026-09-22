//! Typed wrappers over the tt-kmd ioctls.
//!
//! Each returns `io::Error` rather than a bespoke type: every failure here is a
//! syscall failure, and `TransportError::Io` already exists to carry one.

use std::os::fd::{AsRawFd, BorrowedFd};

use crate::abi;

/// Issue an ioctl whose argument is a single `#[repr(C)]` struct.
fn call<T>(fd: BorrowedFd<'_>, request: u64, arg: &mut T) -> std::io::Result<()> {
    // SAFETY: `arg` is a live, correctly-sized, correctly-aligned instance of the
    // struct this request number expects — which is what
    // `crates/tt-kmd/tests/abi_layout.rs` checks against the real header.
    let rc = unsafe { libc::ioctl(fd.as_raw_fd(), request as libc::c_ulong, arg as *mut T) };
    if rc < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub fn get_device_info(fd: BorrowedFd<'_>) -> std::io::Result<abi::GetDeviceInfo> {
    let mut arg = abi::GetDeviceInfo {
        in_output_size_bytes: (std::mem::size_of::<abi::GetDeviceInfo>()
            - std::mem::size_of::<u32>()) as u32,
        ..Default::default()
    };
    call(fd, abi::GET_DEVICE_INFO, &mut arg)?;
    Ok(arg)
}

pub fn get_driver_info(fd: BorrowedFd<'_>) -> std::io::Result<abi::GetDriverInfo> {
    let mut arg = abi::GetDriverInfo {
        in_output_size_bytes: (std::mem::size_of::<abi::GetDriverInfo>()
            - std::mem::size_of::<u32>()) as u32,
        ..Default::default()
    };
    call(fd, abi::GET_DRIVER_INFO, &mut arg)?;
    Ok(arg)
}

/// `QUERY_MAPPINGS` — the BAR apertures and the mmap keys that reach them.
///
/// The argument is a flexible-array struct: a count, then that many entries
/// written back by the driver (`memory.c:376-379`). Built as one byte buffer
/// because Rust has no flexible array member.
pub fn query_mappings(fd: BorrowedFd<'_>, count: u32) -> std::io::Result<Vec<abi::Mapping>> {
    let head = std::mem::size_of::<abi::QueryMappingsIn>();
    let entry = std::mem::size_of::<abi::Mapping>();
    let mut buf = vec![0u8; head + entry * count as usize];
    buf[0..4].copy_from_slice(&count.to_le_bytes());

    // SAFETY: the buffer is at least as large as the driver will write, and is
    // aligned to 8 by `Vec<u8>`'s allocator for this size class. The layout of
    // the region the driver writes is checked in `tests/abi_layout.rs`.
    let rc = unsafe {
        libc::ioctl(
            fd.as_raw_fd(),
            abi::QUERY_MAPPINGS as libc::c_ulong,
            buf.as_mut_ptr(),
        )
    };
    if rc < 0 {
        return Err(std::io::Error::last_os_error());
    }

    let mut out = Vec::new();
    for i in 0..count as usize {
        let at = head + entry * i;
        let field = |o: usize, n: usize| &buf[at + o..at + o + n];
        let id = u32::from_le_bytes(field(0, 4).try_into().unwrap());
        if id == 0 {
            continue;
        }
        out.push(abi::Mapping {
            mapping_id: id,
            reserved: 0,
            mapping_base: u64::from_le_bytes(field(8, 8).try_into().unwrap()),
            mapping_size: u64::from_le_bytes(field(16, 8).try_into().unwrap()),
        });
    }
    Ok(out)
}

/// Reserve a TLB window of `size` bytes, returning its index and mmap keys.
///
/// The index is the same number that indexes the 96-bit configuration array in
/// BAR0, which is what lets `tt_device::tlb` keep configuring windows itself
/// while the driver still arbitrates who owns which one.
pub fn allocate_tlb(fd: BorrowedFd<'_>, size: u64) -> std::io::Result<abi::AllocateTlb> {
    let mut arg = abi::AllocateTlb {
        in_size: size,
        ..Default::default()
    };
    call(fd, abi::ALLOCATE_TLB, &mut arg)?;
    Ok(arg)
}

pub fn free_tlb(fd: BorrowedFd<'_>, id: u32) -> std::io::Result<()> {
    let mut arg = abi::FreeTlb { in_id: id };
    call(fd, abi::FREE_TLB, &mut arg)
}

/// Program a window through the driver.
///
/// Not the path the runtime takes — `tt_device::tlb::write_config` writes the
/// configuration words directly, so that the encoder the simulator gates cover is
/// also the encoder silicon runs. This exists so the two can be compared, which
/// is the only way that encoder gets silicon evidence.
#[allow(clippy::too_many_arguments)]
pub fn configure_tlb(
    fd: BorrowedFd<'_>,
    id: u32,
    addr: u64,
    x_end: u16,
    y_end: u16,
    noc: u8,
    ordering: u8,
) -> std::io::Result<()> {
    let mut arg = abi::ConfigureTlb {
        in_id: id,
        in_config: abi::NocTlbConfig {
            addr,
            x_end,
            y_end,
            noc,
            ordering,
            ..Default::default()
        },
        ..Default::default()
    };
    call(fd, abi::CONFIGURE_TLB, &mut arg)
}

/// Register a NoC write for the driver to perform when this fd is closed.
///
/// The only cleanup that survives a segfault or the OOM killer, because it does
/// not depend on this process running any more code.
pub fn set_noc_cleanup(
    fd: BorrowedFd<'_>,
    enabled: bool,
    x: u8,
    y: u8,
    noc: u8,
    addr: u64,
    data: u32,
) -> std::io::Result<()> {
    let mut arg = abi::SetNocCleanup {
        argsz: std::mem::size_of::<abi::SetNocCleanup>() as u32,
        enabled: enabled as u8,
        x,
        y,
        noc,
        addr,
        data: data as u64,
        ..Default::default()
    };
    call(fd, abi::SET_NOC_CLEANUP, &mut arg)
}

/// Take or release one of the driver's 64 arbitrated locks.
///
/// Returns whether the operation succeeded — `ACQUIRE` reports 0 when another
/// process holds it rather than failing.
pub fn lock_ctl(fd: BorrowedFd<'_>, index: u8, flags: u32) -> std::io::Result<bool> {
    let mut arg = abi::LockCtl {
        in_output_size_bytes: std::mem::size_of::<abi::LockCtl>() as u32,
        in_flags: flags,
        in_index: index,
        ..Default::default()
    };
    call(fd, abi::LOCK_CTL, &mut arg)?;
    Ok(arg.out_value == 1)
}
