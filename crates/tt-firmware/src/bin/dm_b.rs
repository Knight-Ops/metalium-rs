//! The RISCV B data mover: GDDR <-> this tile's L1, one descriptor at a time.
//!
//! Protocol and layout are `tt_isa::dm`, which the host shares. Resident: loaded
//! once and left running, so a kernel's operands and results move without the
//! host touching either. Keeps the heartbeat, so the host can tell a stuck
//! mover from a dead core.

#![no_std]
#![no_main]

use tt_firmware::{l1_read32, l1_write32, mailbox_word, noc, publish};
use tt_isa::dm::{self, op, Descriptor};
use tt_isa::mailbox::offset;
use tt_isa::noc::niu::{Command, TxnId, MAX_REQUEST_BYTES};

const TXN: TxnId = match TxnId::new(2) {
    Some(t) => t,
    None => panic!(),
};

fn rd(addr: u64) -> u32 {
    // SAFETY: every address used is a fixed, aligned word of the mover's mailbox.
    unsafe { l1_read32(addr) }
}

fn wr(addr: u64, v: u32) {
    // SAFETY: as `rd`.
    unsafe { l1_write32(addr, v) }
}

/// Move one descriptor's bytes, in requests of at most 16 KiB, and wait for all
/// of them. Every request's range is a sub-range of the checked descriptor, so
/// it is inside the channel and inside L1, with the congruence preserved.
fn run(me: (u8, u8), d: Descriptor) -> Result<(), u32> {
    let len = d.range.len() as u32;
    let mut done = 0u32;
    while done < len {
        let n = (len - done).min(MAX_REQUEST_BYTES);
        let part = d
            .range
            .channel()
            .range(d.range.offset() + done as u64, n as u64)
            .ok_or(dm::error::RANGE)?;
        let l1 = d.l1 + done;
        let cmd = if d.op == op::READ {
            Command::ReadDram { from: part, port: d.port, to_local: l1 }
        } else {
            Command::WriteDram { from_local: l1, to: part, port: d.port }
        };
        noc::issue(&cmd, me, TXN).map_err(|_| dm::error::ALIGNMENT)?;
        done += n;
    }
    noc::wait(TXN);
    Ok(())
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    let me = (rd(dm::MY_X) as u8, rd(dm::MY_Y) as u8);
    let usable = rd(dm::USABLE);
    let mut beat: u32 = 0;
    loop {
        beat = beat.wrapping_add(1);
        wr(mailbox_word(offset::HEARTBEAT), beat);
        // The host's descriptor arrives over the NoC, which does not invalidate
        // the L0 data cache (MemoryOrdering.md:59): every poll through a fence.
        publish();
        // Compared against DONE in L1 rather than a local copy, so the whole
        // of the mover's state is the mailbox, which the host can read and reset.
        let seq = rd(dm::SEQ);
        if seq == 0 || seq == rd(dm::DONE) {
            continue;
        }
        let d = Descriptor::decode(
            usable,
            rd(dm::OP),
            rd(dm::CHANNEL),
            rd(dm::PORT),
            rd(dm::DRAM_OFFSET),
            rd(dm::L1_ADDR),
            rd(dm::LEN),
        );
        let result = d.and_then(|d| run(me, d));
        wr(dm::ERROR, result.err().unwrap_or(dm::error::NONE));
        // The data is in L1 (a read) or acknowledged by the DRAM tile (a write,
        // response-marked) before DONE is published.
        publish();
        wr(dm::DONE, seq);
        publish();
    }
}
