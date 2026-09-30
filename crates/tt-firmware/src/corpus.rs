//! The body of the generic Tensix program runner, shared by the `corpus` (T1)
//! and `corpus_t0` (T0) images.
//!
//! Generic over the pushing core and its Tensix thread because which one is right
//! differs by target. ttsim models the RISC-V view of `Dst` only for T1
//! (divergence row 12). Silicon needs the program to run on thread 0, because
//! `UNPACR` in multi-context mode reads `ADCs[ContextADC]` -- thread 0 -- whichever
//! thread issues it, while the `SETADC*` instructions that program it write the
//! issuing thread's (divergence row 45). The core and thread stay type
//! parameters, so `PushesTo` still rules out the pairings that hang.

use crate::cfg::write_config_field;
use crate::tensix::{push_word, read_dst32, wait_for_coprocessor};
use crate::{fail, finish, l1_read32, l1_write32, publish, spin};
use tt_isa::cfg::ConfigBank;
use tt_isa::mailbox::role::Mailbox;
use tt_isa::mailbox::{self, panic_code, status};
use tt_isa::sfpu::dst32_address;
use tt_isa::tensix::{PushesTo, TensixThread};

pub fn run<Riscv, Thread>() -> !
where
    Thread: TensixThread,
    Riscv: PushesTo<Thread>,
{
    run_in::<Riscv, Thread>(Mailbox::single_core())
}

/// The same runner, reporting through a role mailbox: one of three cores each
/// running its own part of a kernel (`tt_isa::mailbox::role`).
pub fn run_role<Riscv, Thread>() -> !
where
    Thread: TensixThread,
    Riscv: PushesTo<Thread>,
{
    run_in::<Riscv, Thread>(Mailbox::of(Thread::INDEX))
}

/// Stop, having said why in `mb` as well as in the single-core mailbox the
/// panic path always uses.
fn fail_in(mb: Mailbox, code: u32) -> ! {
    // SAFETY: fixed aligned mailbox locations inside L1.
    unsafe {
        l1_write32(mb.panic_code(), code);
        l1_write32(mb.status(), status::PANICKED);
    }
    fail(code)
}

fn run_in<Riscv, Thread>(mb: Mailbox) -> !
where
    Thread: TensixThread,
    Riscv: PushesTo<Thread>,
{
    // SAFETY: fixed aligned mailbox locations inside L1.
    unsafe { l1_write32(mb.status(), status::RUNNING) };
    publish();
    // SAFETY: fixed, aligned mailbox locations written by the host before this core
    // left reset.
    let thread = unsafe { l1_read32(mb.thread_index()) };
    let fmt = unsafe { l1_read32(mb.dst_access_fmt()) };
    let program_len = unsafe { l1_read32(mb.program_len()) };
    let dump_first = unsafe { l1_read32(mb.dump_row_first()) };
    let dump_rows = unsafe { l1_read32(mb.dump_row_count()) };

    // Bounds are checked here rather than trusted, because a runaway length would
    // push whatever happens to be in L1 into the coprocessor.
    // `thread` is a cross-check rather than a choice: which core this is, is fixed
    // at build time by `Riscv`/`Thread` above. A mismatch means the host started
    // this image somewhere it did not intend, which would otherwise show up as a
    // wrong result from a correctly-executed program.
    if thread != Thread::INDEX
        || program_len > mailbox::PROGRAM_MAX
        || dump_rows > mailbox::DUMP_MAX_ROWS
    {
        fail_in(mb, panic_code::EXPLICIT);
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
        let field = Thread::DST_ACCESS_FMT;
        if !field.fits(fmt) {
            fail_in(mb, panic_code::EXPLICIT);
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
        // above; `Riscv` may push to `Thread`, which the type system checked; the
        // backend is out of reset.
        unsafe { push_word::<Riscv, Thread>(l1_read32(mb.program() + (i as u64) * 4)) }
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
                l1_write32(mb.dump_offset(row, column), value);
            }
            column += 1;
        }
        row += 1;
    }

    // The dump is what carries the result; this only says it is complete.
    if mb == Mailbox::single_core() {
        finish(program_len);
    } else {
        // SAFETY: fixed aligned mailbox location inside L1.
        unsafe { l1_write32(mb.status(), status::DONE) };
        publish();
    }
    spin()
}
