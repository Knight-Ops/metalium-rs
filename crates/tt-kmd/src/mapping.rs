//! An mmap'd PCI BAR.
//!
//! The only `unsafe` in this crate lives here and in [`crate::ioctl`]. `tt-device`
//! is `#![forbid(unsafe_code)]`, which is why the driver path is a separate crate
//! rather than a module inside it.

use std::os::fd::BorrowedFd;

/// A mapped BAR aperture, unmapped on drop.
pub struct Mapping {
    ptr: *mut u8,
    len: usize,
}

// The pointer is an MMIO aperture, not shared heap state; sending one between
// threads is fine. It is deliberately not `Sync`: concurrent access to a TLB
// window whose configuration another thread can retarget is exactly the race
// `Device`'s `&mut self` exists to prevent.
unsafe impl Send for Mapping {}

impl Mapping {
    /// Map `len` bytes of `fd` at `offset`.
    ///
    /// `offset` is one of the `mapping_base` values `QUERY_MAPPINGS` reports —
    /// a multiplexing key the driver decodes (`memory.c:258-263`), not a byte
    /// offset into anything.
    pub fn new(fd: BorrowedFd<'_>, offset: u64, len: usize) -> std::io::Result<Self> {
        // SAFETY: a fresh anonymous-style mapping of a device file; the kernel
        // validates `offset` against its own table of mappable entities and
        // fails the call rather than handing back a bad pointer.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                std::os::fd::AsRawFd::as_raw_fd(&fd),
                offset as libc::off_t,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Mapping {
            ptr: ptr.cast(),
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy `dst.len()` bytes out of the aperture at `offset`.
    ///
    /// Volatile and, wherever alignment allows, dword-at-a-time. Neither is an
    /// optimisation: these are device registers, so the compiler may not elide,
    /// reorder or merge the accesses, and several regions behind this aperture
    /// (the TLB configuration array, the NIU registers) are documented as
    /// dword-only. A `memcpy` would be free to split a dword into bytes.
    pub fn read(&self, offset: usize, dst: &mut [u8]) {
        assert!(offset + dst.len() <= self.len);
        let mut i = 0;
        while i < dst.len() {
            let at = offset + i;
            if at % 4 == 0 && dst.len() - i >= 4 {
                // SAFETY: bounds checked above; `at` is 4-aligned within the map.
                let v = unsafe { std::ptr::read_volatile(self.ptr.add(at).cast::<u32>()) };
                dst[i..i + 4].copy_from_slice(&v.to_le_bytes());
                i += 4;
            } else {
                // SAFETY: bounds checked above.
                dst[i] = unsafe { std::ptr::read_volatile(self.ptr.add(at)) };
                i += 1;
            }
        }
    }

    /// Copy `src` into the aperture at `offset`. See [`Mapping::read`].
    pub fn write(&self, offset: usize, src: &[u8]) {
        assert!(offset + src.len() <= self.len);
        let mut i = 0;
        while i < src.len() {
            let at = offset + i;
            if at % 4 == 0 && src.len() - i >= 4 {
                let v = u32::from_le_bytes(src[i..i + 4].try_into().unwrap());
                // SAFETY: bounds checked above; `at` is 4-aligned within the map.
                unsafe { std::ptr::write_volatile(self.ptr.add(at).cast::<u32>(), v) };
                i += 4;
            } else {
                // SAFETY: bounds checked above.
                unsafe { std::ptr::write_volatile(self.ptr.add(at), src[i]) };
                i += 1;
            }
        }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this pointer and length came from a successful `mmap` above and
        // are unmapped exactly once.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}
