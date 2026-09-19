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

use tt_firmware::tensix::{push, read_dst32, wait_for_coprocessor};
use tt_firmware::{fail, finish, l1_read32, spin};
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
