//! Evidence that the validation layer is load-bearing.
//!
//! `transport::validate` rejects accesses that libttsim would refuse. That is only
//! worth its complexity if the refusal is genuinely fatal rather than, say, a
//! silently ignored write. These tests bypass the safe wrapper, make the bad access
//! against the raw library inside a forked child, and assert the child dies.
//!
//! If one of these ever starts *passing* the access through, the corresponding rule
//! in `validate` has become unnecessary and should be deleted rather than kept as
//! folklore.

use tt_ttsim::{fork_scope, ForkError};
use tt_ttsim_sys::Lib;

/// Bring up the raw library in this (already forked) process and hand it over.
///
/// # Safety
///
/// The caller is inside a forked child that is expected to die.
unsafe fn raw_simulator() -> Lib {
    let path = std::env::var_os(tt_ttsim::LIB_PATH_ENV)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(tt_ttsim::default_lib_path);
    let lib = unsafe { Lib::load(&path) }.expect("load libttsim");
    unsafe { (lib.set_pci_dma_mem_callbacks)(noop_dma_rd, noop_dma_wr) };
    unsafe { (lib.init)() };
    lib
}

unsafe extern "C" fn noop_dma_rd(_p: u64, _d: *mut core::ffi::c_void, _s: u32) {}
unsafe extern "C" fn noop_dma_wr(_p: u64, _s: *const core::ffi::c_void, _n: u32) {}

/// Assert that the given raw access terminates the process.
#[track_caller]
fn is_fatal(what: &str, body: impl FnOnce(&Lib)) {
    let outcome = fork_scope(|| {
        let lib = unsafe { raw_simulator() };
        body(&lib);
        // Reaching here means libttsim accepted the access.
        eprintln!("ACCESS WAS ACCEPTED");
    });
    match outcome {
        Err(ForkError::Exited(code)) if code != 0 => { /* died as expected */ }
        Err(ForkError::Signalled(_)) => { /* also fatal */ }
        Ok(()) => panic!(
            "{what}: libttsim accepted this access, so the matching rule in \
             transport::validate is no longer needed and should be removed"
        ),
        Err(e) => panic!("{what}: unexpected outcome: {e}"),
    }
}

#[test]
fn reading_an_undecoded_bar0_offset_is_fatal() {
    is_fatal("undecoded BAR0 offset", |lib| {
        let mut buf = [0u8; 4];
        // 0x1A00_0000 is the reserved span between the TLB windows and the config
        // array -- a plausible-looking offset that is not decoded.
        unsafe { (lib.pci_mem_rd_bytes)(0x1_0000_0000 + 0x1A00_0000, buf.as_mut_ptr().cast(), 4) };
        std::hint::black_box(buf);
    });
}

#[test]
fn reading_a_write_only_tlb_config_register_is_fatal() {
    is_fatal("TLB config read-back", |lib| {
        let mut buf = [0u8; 4];
        unsafe { (lib.pci_mem_rd_bytes)(0x1_0000_0000 + 0x1FC0_0000, buf.as_mut_ptr().cast(), 4) };
        std::hint::black_box(buf);
    });
}

#[test]
fn straddling_a_window_boundary_is_fatal() {
    is_fatal("2 MiB window straddle", |lib| {
        let mut buf = [0u8; 8];
        unsafe {
            (lib.pci_mem_rd_bytes)(
                0x1_0000_0000 + (2 * 1024 * 1024 - 4),
                buf.as_mut_ptr().cast(),
                8,
            )
        };
        std::hint::black_box(buf);
    });
}

#[test]
fn reading_config_space_past_0x3c_is_fatal() {
    // This is why `ConfigOffset` is a closed enum: the obvious "walk the capability
    // list" or "dump config space" routine would land here.
    is_fatal("config space beyond 0x3C", |lib| {
        let v = unsafe { (lib.pci_config_rd32)(0, 0x40) };
        std::hint::black_box(v);
    });
}

#[test]
fn tile_access_backdoor_is_unsupported_on_blackhole() {
    // libttsim_tile_rd_bytes would be a convenient shortcut into L1 that skips TLB
    // windows entirely. It is UnsupportedFunctionality on Blackhole, which is why
    // the TLB layer is on the critical path for every later step rather than an
    // optimisation.
    is_fatal("libttsim_tile_rd_bytes", |lib| {
        let mut buf = [0u8; 4];
        unsafe { (lib.tile_rd_bytes)(1, 2, 0, buf.as_mut_ptr().cast(), 4) };
        std::hint::black_box(buf);
    });
}
