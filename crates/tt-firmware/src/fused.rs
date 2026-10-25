//! Calling fused `.ttinsn` code (`tt_isa::ttinsn`): a role program the host wrote to the program
//! cache as RISC-V code that pushes its Tensix words by executing them, its length flagged
//! `tt_isa::ttinsn::FUSED` (`docs/plans/ttinsn-push.md`).
//!
//! The build-time instruction gate cannot see code written at run time: the host validated it
//! (`tt_isa::ttinsn::validate`) and ran it against the program's own expansion before writing it,
//! and the runner checks only where it lies (the program cache, aligned) before calling it.
//!
//! A core's instruction cache is not coherent with L1: the program cache reuses its addresses for
//! other programs, so code is called only after the cache is invalidated
//! (`InstructionCache.md`, "Cache invalidation"). Without it, a rewritten program runs as it was
//! (`docs/learnings/silicon-operating-notes.md`, the `.ttinsn` experiment's stale-code control).

use crate::publish;
use tt_isa::cfg::{generated::global::RISCV_IC_INVALIDATE_InvalidateAll, ConfigBank};

/// Invalidate the instruction caches `mask` names (bit 0 RISCV B, 1 T0, 2 T1, 3 T2, 4 NC), after
/// code was written to L1 ending at `last_written`: the sequence the page gives -- a read back of
/// the last write, consumed, then the mask store, so the store is processed after the writes --
/// followed by the 24 instructions it says the pipeline may need to drain. ~330 cycles.
///
/// # Safety
///
/// `last_written` is an aligned L1 word; only a core's own cache has been exercised
/// (`mask` its own bit).
pub unsafe fn invalidate_icache(last_written: u64, mask: u32) {
    publish();
    let register = RISCV_IC_INVALIDATE_InvalidateAll.riscv_address(ConfigBank::Bank0);
    // SAFETY: the documented register, a whole-word store (`sw`: RISC-V cannot write `Config`
    // any narrower); the word holds only this field.
    unsafe {
        core::arch::asm!(
            "lw   {t}, 0({last})",
            "andi {t}, {t}, 0",
            "sw   {mask}, 0({register})",
            last = in(reg) last_written as u32,
            mask = in(reg) mask,
            register = in(reg) register as u32,
            t = out(reg) _,
            options(nostack),
        );
    }
    for _ in 0..24 {
        // SAFETY: a no-op; not `spin_loop`, whose `pause` Blackhole lacks.
        unsafe { core::arch::asm!("nop") };
    }
}

/// Call the code at `address`: a function taking nothing, returning `a0`.
///
/// # Safety
///
/// `address` holds valid RV32IM code ending in a return, and this core's instruction cache is
/// current for it ([`invalidate_icache`]).
pub unsafe fn call(address: u64) -> u32 {
    // SAFETY: the caller's contract.
    let f: extern "C" fn() -> u32 = unsafe { core::mem::transmute(address as usize) };
    f()
}
