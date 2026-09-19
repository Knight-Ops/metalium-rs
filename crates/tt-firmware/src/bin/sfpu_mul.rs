//! The step 4 gate: drive the Tensix coprocessor and read the result back.
//!
//! Computes 3.0 × 2.0 on the Vector Unit and publishes the FP32 bit pattern to the
//! L1 mailbox, where the host can read it. Expected: `0x40C0_0000`.
//!
//! This core has to exist in the middle of the chain. `INSTRN_BUF_BASE` and `Dst`
//! are both unmapped to the NoC, so the host can neither push the instructions nor
//! read the answer.

#![no_std]
#![no_main]

use tt_firmware::cfg::write_config_field;
use tt_firmware::tensix::{push, read_dst32, wait_for_coprocessor};
use tt_firmware::{fail, finish, l1_read32, publish, spin};
use tt_isa::cfg::generated::alu;
use tt_isa::cfg::ConfigBank;
use tt_isa::mailbox::{self, panic_code};
use tt_isa::sfpu::{self, dst32_address, store_format};

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    // Operands come from the host, through the L1 mailbox, rather than being
    // baked in. That makes the gate a test of arithmetic rather than of one
    // constant: the host can ask for several products and check each, which a
    // firmware returning a fixed value could not satisfy.
    //
    // It also means the instruction encoder runs on the device. `SFPLOADI`
    // carries its operand as a 16-bit immediate, so a host-supplied value has to
    // be encoded into the instruction word here, at run time.
    //
    // SAFETY: fixed, aligned mailbox locations written by the host before this
    // core left reset.
    let (a, b) = unsafe { (l1_read32(mailbox::OPERAND_A), l1_read32(mailbox::OPERAND_B)) };
    let thread = unsafe { l1_read32(mailbox::THREAD_INDEX) };
    let fmt = unsafe { l1_read32(mailbox::DST_ACCESS_FMT) };
    if thread > 2 {
        fail(panic_code::EXPLICIT);
    }

    // Set the shape of the Dst mapping deliberately rather than inheriting
    // whatever reset left behind. `fmt = 0` is `float Dst32b[512][16]`, the shape
    // a 32-bit `lw` below is valid for; the reset default happens to be 0 too, but
    // depending on that is a bug waiting for the first kernel that changes it.
    //
    // SEC1 because this firmware runs on the core driving Tensix thread 1 -- the
    // only one whose Dst mapping ttsim models. The field is indexed per section,
    // so the thread the core drives selects which one to write.
    //
    // SAFETY: the host released the Tensix backend before releasing this core, and
    // nothing has been pushed yet, so no Tensix instruction is in flight.
    unsafe {
        // Bank 0, not `cfg::active_bank(thread)`. The live bank is selected by
        // `ThreadConfig[thread].CFG_STATE_ID_StateID`, but ttsim maps the whole
        // configuration aperture to a flat `Config` array with the bank hardcoded
        // to zero and no `ThreadConfig` region at all, so reading the state ID
        // there is fatal. Nothing in this firmware changes the state ID, and its
        // reset value is 0, so bank 0 is correct either way -- but on silicon,
        // where a kernel may have switched banks, `active_bank` is the right call.
        let bank = ConfigBank::Bank0;
        let field = match thread {
            0 => alu::RISC_DEST_ACCESS_CTRL_SEC0_fmt,
            1 => alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt,
            _ => alu::RISC_DEST_ACCESS_CTRL_SEC2_fmt,
        };
        if !field.fits(fmt) {
            fail(panic_code::EXPLICIT);
        }
        write_config_field(field, bank, fmt);
    }
    // Configuration is written by RISC-V but read by the coprocessor. Auto TTSync
    // covers the push-then-load direction, not this one, so drain the store queue
    // before pushing anything that depends on the new value.
    publish();

    // Build the program first. Every encoding is checked, and a bad one fails
    // loudly here rather than being pushed and silently misexecuting.
    let Ok(load_a) = sfpu::load_f32(0, a) else { fail(panic_code::EXPLICIT) };
    let Ok(load_b) = sfpu::load_f32(1, b) else { fail(panic_code::EXPLICIT) };
    let Ok(multiply) = sfpu::mul(0, 1, 2) else { fail(panic_code::EXPLICIT) };
    let Ok(store) = sfpu::store(2, store_format::FP32, 0, 0) else {
        fail(panic_code::EXPLICIT)
    };

    // SAFETY: this is RISCV T0, which may push; the host released the Tensix
    // backend from soft reset before releasing this core.
    unsafe {
        push(load_a[0]);
        push(load_a[1]);
        push(load_b[0]);
        push(load_b[1]);
        push(multiply);
        // No SFPNOP between the multiply and the store: on Blackhole the two-cycle
        // SFPMAD latency is covered by automatic stalling, and SFPSTORE is not one
        // of the documented cases that stalling fails to detect. See
        // `Instruction::stalls_automatically_after_mad`.
        push(store);
    }

    // The pushes above have only reached a FIFO. Wait for them to retire before
    // looking at Dst.
    wait_for_coprocessor();

    // Lane 0 of the store lands at Dst[0][0].
    // SAFETY: the coprocessor has retired the store, and fmt is a 32-bit shape.
    let result = unsafe { read_dst32(dst32_address(0, 0)) };

    finish(result);
    spin()
}
