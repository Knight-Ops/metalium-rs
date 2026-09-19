//! A generic Tensix program runner: push what the host staged, dump what it asks
//! for.
//!
//! The step 4 firmware computes one thing. This computes whatever the host encoded,
//! which is what turns "does this instruction execute correctly" from a firmware
//! change into a test case. The host has the generated instruction table; the device
//! side only has to push words and copy `Dst` back.
//!
//! Both halves of that are forced by the hardware, not chosen. `INSTRN_BUF_BASE` is
//! unmapped to the NoC, so the host cannot push; `Dst` is unmapped too, so the host
//! cannot read a result. A baby RISC-V has to sit in the middle, and this is the
//! smallest thing that can.

#![no_std]
#![no_main]

use tt_firmware::cfg::write_config_field;
use tt_firmware::tensix::{push_word, read_dst32, wait_for_coprocessor};
use tt_firmware::{fail, finish, l1_read32, l1_write32, publish, spin};
use tt_isa::cfg::generated::alu;
use tt_isa::cfg::ConfigBank;
use tt_isa::mailbox::{self, panic_code};
use tt_isa::sfpu::dst32_address;

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    // SAFETY: fixed, aligned mailbox locations written by the host before this core
    // left reset.
    let thread = unsafe { l1_read32(mailbox::THREAD_INDEX) };
    let fmt = unsafe { l1_read32(mailbox::DST_ACCESS_FMT) };
    let program_len = unsafe { l1_read32(mailbox::PROGRAM_LEN) };
    let dump_first = unsafe { l1_read32(mailbox::DUMP_ROW_FIRST) };
    let dump_rows = unsafe { l1_read32(mailbox::DUMP_ROW_COUNT) };

    // Bounds are checked here rather than trusted, because a runaway length would
    // push whatever happens to be in L1 into the coprocessor.
    if thread > 2 || program_len > mailbox::PROGRAM_MAX || dump_rows > mailbox::DUMP_MAX_ROWS {
        fail(panic_code::EXPLICIT);
    }

    // Set the shape of the Dst mapping deliberately rather than inheriting whatever
    // reset left behind, exactly as the step 4 firmware does.
    //
    // SAFETY: the host released the Tensix backend before releasing this core, and
    // nothing has been pushed yet, so no Tensix instruction is in flight.
    unsafe {
        // Bank 0, not `cfg::active_bank(thread)`: ttsim maps the whole configuration
        // aperture to a flat `Config` array with the bank hardcoded to zero and no
        // `ThreadConfig` region, so reading the state ID there is fatal. Nothing
        // here changes it and its reset value is 0.
        let field = match thread {
            0 => alu::RISC_DEST_ACCESS_CTRL_SEC0_fmt,
            1 => alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt,
            _ => alu::RISC_DEST_ACCESS_CTRL_SEC2_fmt,
        };
        if !field.fits(fmt) {
            fail(panic_code::EXPLICIT);
        }
        write_config_field(field, ConfigBank::Bank0, fmt);
    }
    // Configuration is written by RISC-V but read by the coprocessor. Auto TTSync
    // covers the push-then-load direction, not this one, so drain the store queue
    // before pushing anything that depends on the new value.
    publish();

    let mut i = 0;
    while i < program_len {
        // SAFETY: the word is inside the staged program, whose length was checked
        // above; this is RISCV T0/T1/T2, which may push; the backend is out of
        // reset.
        unsafe { push_word(l1_read32(mailbox::PROGRAM + (i as u64) * 4)) }
        i += 1;
    }

    // The pushes above have only reached a FIFO. Wait for them to retire before
    // looking at Dst.
    wait_for_coprocessor();

    let mut row = 0;
    while row < dump_rows {
        let mut column = 0;
        while column < mailbox::DUMP_ROW_WORDS {
            // SAFETY: the coprocessor has retired the program, fmt is a 32-bit
            // shape, and the destination is inside the dump region.
            unsafe {
                let value = read_dst32(dst32_address(dump_first + row, column));
                l1_write32(mailbox::dump_offset(row, column), value);
            }
            column += 1;
        }
        row += 1;
    }

    // The dump is what carries the result; this only says it is complete.
    finish(program_len);
    spin()
}
