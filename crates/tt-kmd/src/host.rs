//! Host memory the card reads and writes itself (`PIN_PAGES` with
//! `NOC_DMA`): the card's movers reach it through the PCIe tile at the NoC
//! address the driver returns, as DMA. Host-to-card copies through a BAR are
//! CPU stores, and run uncached under this machine's VM passthrough
//! (`docs/learnings/ttsim-divergence.md` row M); a copy the card makes is a PCIe
//! transaction of the card's, which no CPU mapping slows.

use std::fs::File;
use std::os::fd::AsFd;

use crate::ioctl;

/// Page-aligned host memory, pinned for the card's NoC. Unpinned and freed on
/// drop -- and by the driver if the process dies first.
pub struct HostBuffer {
    fd: File,
    ptr: *mut u8,
    len: usize,
    iova: u64,
    noc_address: u64,
}

// The buffer is plain memory owned by this value; the raw pointer is only
// how it is held.
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    /// `len` bytes, zeroed, pinned for the card's NoC. A page or less is an
    /// ordinary page; anything larger is one 1 GiB hugepage (`len` at most
    /// 1 GiB), pinned as physically contiguous. A guest whose IOMMU does not
    /// translate (`iommu=pt`, as this workstation's VM boots) cannot pin
    /// scattered pages for DMA, so a hugepage is how a buffer of more than one
    /// page stays contiguous -- tt-metal's host memory is 1 GiB hugepages for
    /// the same reason. `fd` is a clone of the device's: a pin belongs to the
    /// open file, which a clone shares.
    pub(crate) fn pin(fd: File, len: usize) -> std::io::Result<Self> {
        const PAGE: usize = 4096;
        const HUGE: usize = 1 << 30;
        let huge = len > PAGE;
        if len > HUGE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{len} bytes of pinned host memory: one buffer holds at most 1 GiB"),
            ));
        }
        let (len, extra, flags) = if huge {
            // MAP_HUGE_1GB: log2(1 GiB) in the flags' size field.
            (
                HUGE,
                libc::MAP_HUGETLB | (30 << libc::MAP_HUGE_SHIFT),
                crate::abi::pin::CONTIGUOUS,
            )
        } else {
            (PAGE, 0, 0)
        };
        // SAFETY: an anonymous private mapping; nothing else refers to it.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE | extra,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let e = std::io::Error::last_os_error();
            return Err(if huge {
                std::io::Error::new(
                    e.kind(),
                    format!(
                        "no free 1 GiB hugepage for pinned host memory ({e}); reserve some \
                         (/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages)"
                    ),
                )
            } else {
                e
            });
        }
        let ptr = ptr as *mut u8;
        match ioctl::pin_pages(
            fd.as_fd(),
            ptr as u64,
            len as u64,
            flags | crate::abi::pin::NOC_DMA,
        ) {
            Ok((iova, noc_address)) => Ok(HostBuffer {
                fd,
                ptr,
                len,
                iova,
                noc_address,
            }),
            Err(e) => {
                // SAFETY: mapped above, not yet shared.
                unsafe { libc::munmap(ptr as *mut libc::c_void, len) };
                Err(e)
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Where the card's NoC reaches the buffer's first byte, at the PCIe tile.
    pub fn noc_address(&self) -> u64 {
        self.noc_address
    }

    /// The IOVA the PCIe tile's outbound requests carry.
    pub fn iova(&self) -> u64 {
        self.iova
    }

    /// The buffer's bytes. Only while nothing the card runs writes them.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `len` mapped bytes, owned by this value.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The buffer's bytes, to fill. Only while nothing the card runs reads
    /// or writes them.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` is the only borrow.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        let _ = ioctl::unpin_pages(self.fd.as_fd(), self.ptr as u64, self.len as u64);
        // SAFETY: mapped in `pin`, and the card no longer reaches it.
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
    }
}
