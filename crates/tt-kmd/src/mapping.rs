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

/// Bulk copies, for apertures mapped write-combining over device *memory* --
/// L1 or GDDR, never registers. The caller ([`crate::Kmd`], asked by
/// `tt-device`, which decides what is memory) owns that distinction; these
/// routines only make the copy fast.
///
/// The body moves 16 bytes per instruction with non-temporal accesses, which is
/// what a WC aperture is for (`PCIExpressTile/README.md:126-154`: 22.6 GB/s
/// `memcpy` to a WC mapping against 7 GB/s UC). Head and tail bytes that do not
/// fill an aligned 16 go through the dword path above.
impl Mapping {
    pub fn write_bulk(&self, offset: usize, src: &[u8]) {
        assert!(offset + src.len() <= self.len);
        let head = (16 - offset % 16) % 16;
        let head = head.min(src.len());
        self.write(offset, &src[..head]);
        let body = (src.len() - head) / 16 * 16;
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // Twice the 16-byte path where the aperture is effectively UC --
            // `MEASURED` 152 against 76 MB/s -- since each store then costs one
            // bus transaction whatever its width.
            // SAFETY: as below; the feature was detected just above.
            unsafe { stream_store_256(self.ptr.add(offset + head), &src[head..head + body]) };
            self.write(offset + head + body, &src[head + body..]);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: bounds checked above; `offset + head` is 16-aligned within the
        // map, and `_mm_loadu_si128` takes an unaligned source. SSE2 is baseline
        // on x86_64.
        unsafe {
            use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_sfence, _mm_stream_si128};
            let dst = self.ptr.add(offset + head).cast::<__m128i>();
            let from = src.as_ptr().add(head).cast::<__m128i>();
            for k in 0..body / 16 {
                _mm_stream_si128(dst.add(k), _mm_loadu_si128(from.add(k)));
            }
            // Non-temporal stores are weakly ordered even against later UC
            // stores; nothing that follows (a core release, a mailbox word) may
            // overtake them.
            _mm_sfence();
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.write(offset + head, &src[head..head + body]);
        self.write(offset + head + body, &src[head + body..]);
    }

    pub fn read_bulk(&self, offset: usize, dst: &mut [u8]) {
        assert!(offset + dst.len() <= self.len);
        let head = (16 - offset % 16) % 16;
        let head = head.min(dst.len());
        self.read(offset, &mut dst[..head]);
        let body = (dst.len() - head) / 16 * 16;
        let mut done = false;
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: as in `write_bulk`; the feature was detected just above.
            unsafe { stream_load_256(self.ptr.add(offset + head), &mut dst[head..head + body]) };
            done = true;
        } else if std::arch::is_x86_feature_detected!("sse4.1") {
            // SAFETY: as in `write_bulk`; the feature was detected just above.
            unsafe { stream_load(self.ptr.add(offset + head), &mut dst[head..head + body]) };
            done = true;
        }
        if !done {
            self.read(offset + head, &mut dst[head..head + body]);
        }
        self.read(offset + head + body, &mut dst[head + body..]);
    }
}

/// The body of [`Mapping::write_bulk`] in 32-byte stores.
///
/// # Safety
/// `dst` is 16-aligned and writable for `src.len()` bytes (a multiple of 16).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn stream_store_256(dst: *mut u8, src: &[u8]) {
    use std::arch::x86_64::*;
    let mut i = 0;
    if (dst as usize) % 32 != 0 && src.len() >= 16 {
        _mm_stream_si128(dst.cast(), _mm_loadu_si128(src.as_ptr().cast()));
        i = 16;
    }
    while i + 32 <= src.len() {
        _mm256_stream_si256(
            dst.add(i).cast(),
            _mm256_loadu_si256(src.as_ptr().add(i).cast()),
        );
        i += 32;
    }
    if i < src.len() {
        _mm_stream_si128(
            dst.add(i).cast(),
            _mm_loadu_si128(src.as_ptr().add(i).cast()),
        );
    }
    _mm_sfence();
}

/// `_mm_stream_load_si128`: the one load that reads a WC line without a round
/// trip per access (1.59 GB/s against 0.1 GB/s for `memcpy`, same table).
///
/// # Safety
/// `src` is 16-aligned and readable for `dst.len()` bytes, which is a multiple
/// of 16; the CPU has SSE4.1.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn stream_load(src: *const u8, dst: &mut [u8]) {
    use std::arch::x86_64::{__m128i, _mm_storeu_si128, _mm_stream_load_si128};
    let src = src.cast::<__m128i>();
    let out = dst.as_mut_ptr().cast::<__m128i>();
    for k in 0..dst.len() / 16 {
        _mm_storeu_si128(out.add(k), _mm_stream_load_si128(src.add(k)));
    }
}

/// [`stream_load`] in 32-byte loads where the source allows.
///
/// # Safety
/// As [`stream_load`], with AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn stream_load_256(src: *const u8, dst: &mut [u8]) {
    use std::arch::x86_64::*;
    let mut i = 0;
    if (src as usize) % 32 != 0 && dst.len() >= 16 {
        _mm_storeu_si128(dst.as_mut_ptr().cast(), _mm_stream_load_si128(src.cast()));
        i = 16;
    }
    while i + 32 <= dst.len() {
        let v = _mm256_stream_load_si256(src.add(i).cast());
        _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), v);
        i += 32;
    }
    if i < dst.len() {
        _mm_storeu_si128(
            dst.as_mut_ptr().add(i).cast(),
            _mm_stream_load_si128(src.add(i).cast()),
        );
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this pointer and length came from a successful `mmap` above and
        // are unmapped exactly once.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}
