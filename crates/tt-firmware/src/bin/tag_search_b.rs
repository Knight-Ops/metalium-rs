//! L1 Cache Tag Search Accelerator probe on RISCV B (`tt_isa::tag_search`).
//!
//! Runs the script the host staged at `tag_search::probe::SCRIPT`. For each step
//! it applies the step's masked `Config` read-modify-writes, fences, reads the
//! eight configuration words back, waits a few cycles, fences (which flushes the
//! L0 data cache, so the load below cannot be an L0 hit) and performs one `lw` at
//! the step's address. The loaded word and the readback go to the results.
//!
//! Script word 1 selects blind mode (diagnostic, for the simulator): `Config` is
//! never loaded, each store carries a whole word merged in a local shadow, and no
//! readback is taken. It exists to tell a simulator that refuses the *loads* of
//! these registers from one that refuses the stores.
//!
//! Script word 2 non-zero skips the disarm before the first step and after the
//! last (a control that touches no `Config` word, and per-word simulator probes).
//!
//! The block is disarmed (all five trigger bits cleared) before the first step
//! and again at the end, so no later load by this core is intercepted; `Config`
//! outlives a process on silicon. The probe writes nowhere but `Config` words
//! 212..=219 and the result records.

#![no_std]
#![no_main]

use tt_firmware::{fail, finish, l1_read32, l1_write32, publish, spin};
use tt_isa::mailbox::panic_code;
use tt_isa::tag_search::{self as ts, probe, DISARM};

/// Masked read-modify-write of one `Config` word with plain `lw`/`sw`: RISC-V can
/// write `Config` only with whole-word stores.
///
/// # Safety
///
/// The Tensix backend must be out of soft reset, and `step.word` in 212..=219.
unsafe fn apply(step: ts::WriteStep) {
    let at = ts::config_address(step.word) as *mut u32;
    // SAFETY: a word of the generated `Config` table, per the caller.
    unsafe {
        let old = core::ptr::read_volatile(at);
        core::ptr::write_volatile(at, (old & !step.mask) | (step.value & step.mask));
    }
}

/// Blind-mode counterpart of [`apply`]: one whole-word store of the merge of
/// `step` into `shadow`, with no load of `Config`.
///
/// # Safety
///
/// As [`apply`].
unsafe fn apply_blind(shadow: &mut [u32; 8], step: ts::WriteStep) {
    let i = (step.word - 212) as usize;
    shadow[i] = (shadow[i] & !step.mask) | (step.value & step.mask);
    // SAFETY: a word of the generated `Config` table, per the caller.
    unsafe { core::ptr::write_volatile(ts::config_address(step.word) as *mut u32, shadow[i]) };
}

fn disarm(blind: bool, shadow: &mut [u32; 8]) {
    for s in DISARM {
        // SAFETY: words 212, 218 and 219 of the generated table; the host
        // released the backend before starting this image.
        unsafe {
            if blind {
                apply_blind(shadow, s)
            } else {
                apply(s)
            }
        };
    }
    publish();
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    // SAFETY: the script header is an aligned word staged before release.
    let steps = unsafe { l1_read32(probe::SCRIPT) };
    if steps == 0 || steps > probe::MAX_STEPS {
        fail(panic_code::EXPLICIT);
    }
    // SAFETY: an aligned word of the staged header.
    let blind = unsafe { l1_read32(probe::SCRIPT + 4) } != 0;
    // SAFETY: an aligned word of the staged header.
    let keep = unsafe { l1_read32(probe::SCRIPT + 8) } != 0;
    let mut shadow = [0u32; 8];
    if !keep {
        disarm(blind, &mut shadow);
    }
    for k in 0..steps {
        let step = probe::STEPS + u64::from(k) * probe::STEP_STRIDE;
        let result = probe::RESULTS + u64::from(k) * probe::RESULT_STRIDE;
        // SAFETY: aligned words of the staged step.
        let (load_at, writes) = unsafe { (l1_read32(step), l1_read32(step + 4)) };
        if writes as usize > ts::PROGRAM_STEPS {
            fail(panic_code::EXPLICIT);
        }
        for i in 0..u64::from(writes) {
            let at = step + 8 + i * 12;
            // SAFETY: bounded by `writes`, inside the step's 256 bytes.
            let (word, mask, value) =
                unsafe { (l1_read32(at), l1_read32(at + 4), l1_read32(at + 8)) };
            if !(212..=219).contains(&word) {
                fail(panic_code::EXPLICIT);
            }
            let write = ts::WriteStep {
                word: word as u16,
                mask,
                value,
            };
            // SAFETY: as in `disarm`, and `word` was range-checked.
            unsafe {
                if blind {
                    apply_blind(&mut shadow, write)
                } else {
                    apply(write)
                }
            };
        }
        // The stores must retire before the load that depends on them.
        publish();
        for w in 0..8u64 {
            if blind {
                break;
            }
            // SAFETY: Config words 212..=219; result words in the data arena.
            unsafe {
                let v = core::ptr::read_volatile(ts::config_address(212 + w as u16) as *const u32);
                l1_write32(result + 4 + w * 4, v);
            }
        }
        publish();
        for _ in 0..probe::SETTLE_NOPS {
            // SAFETY: ordinary RISC-V NOP, supported by the instruction gate.
            unsafe { core::arch::asm!("nop", options(nomem, nostack)) };
        }
        // `fence` flushes the L0 data cache: the load below cannot hit in it.
        publish();
        // SAFETY: an aligned L1 word; the accelerator may answer it instead.
        let got = unsafe { l1_read32(u64::from(load_at)) };
        // SAFETY: the step's result record.
        unsafe { l1_write32(result, got) };
        publish();
    }
    if !keep {
        disarm(blind, &mut shadow);
    }
    finish(steps);
    spin()
}
