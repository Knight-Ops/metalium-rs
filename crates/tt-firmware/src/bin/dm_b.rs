//! The RISCV B data mover: GDDR <-> this tile's L1, one descriptor at a time.
//!
//! Protocol and layout are `tt_isa::dm`, which the host shares. Resident: loaded
//! once and left running, so a kernel's operands and results move without the
//! host touching either. Keeps the heartbeat, so the host can tell a stuck
//! mover from a dead core.

#![no_std]
#![no_main]

use tt_firmware::{float, l1_read32, l1_write32, mailbox_word, noc, publish};
use tt_isa::dm::{self, op, record, Descriptor, Entry};
use tt_isa::mailbox::role::Mailbox;
use tt_isa::mailbox::{offset, status};
use tt_isa::noc::niu::{Command, TxnId, MAX_REQUEST_BYTES};

const TXN: TxnId = match TxnId::new(2) {
    Some(t) => t,
    None => panic!(),
};

fn rd(addr: u64) -> u32 {
    // SAFETY: every address used is an aligned word of the mover's mailbox, its
    // list, its scratch slot, or a destination slot `Entry::decode` placed
    // inside L1.
    unsafe { l1_read32(addr) }
}

fn wr(addr: u64, v: u32) {
    // SAFETY: as `rd`, or the timestamper's `TIMESTAMP` register, where a
    // store only appends an event.
    unsafe { l1_write32(addr, v) }
}

/// Issue one descriptor's bytes as NIU requests of at most 16 KiB, without
/// waiting for them. Every request's range is a sub-range of the checked
/// descriptor, so it is inside the channel and inside L1, with the congruence
/// preserved.
fn issue(me: (u8, u8), d: Descriptor) -> Result<(), u32> {
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
    Ok(())
}

/// Transpose the tile in the scratch slot into the slot at `dst`: header
/// copied, datum `(r, c)` from `(c, r)`, both in face order -- face `(fr, fc)`
/// of the result is face `(fc, fr)` of the source, transposed.
fn transpose_from_scratch(dst: u64) {
    for w in 0..(dm::TILE_DATA / 4) {
        wr(dst + w * 4, rd(dm::SCRATCH + w * 4));
    }
    let src = (dm::SCRATCH + dm::TILE_DATA) as *const u32;
    let out = (dst + dm::TILE_DATA) as *mut u32;
    for fr in 0..2usize {
        for fc in 0..2usize {
            let (from, to) = ((fc * 2 + fr) * 256, (fr * 2 + fc) * 256);
            for i in 0..16usize {
                for j in 0..16usize {
                    // SAFETY: both slots are inside L1 (`Entry::decode`), and
                    // indices stay inside their 1024 datums. Plain accesses: the
                    // NoC's writes to the scratch were fenced before this, and
                    // the caller fences these stores before anything reads them.
                    unsafe { *out.add(to + i * 16 + j) = *src.add(from + j * 16 + i) };
                }
            }
        }
    }
}

/// One compute entry over the 1024 datums of three tile slots (`dm::kind`).
fn compute(kind: u32, s: u32, dst: u64, a: u64, b: u64) {
    let (dst, a, b) = (dst + dm::TILE_DATA, a + dm::TILE_DATA, b + dm::TILE_DATA);
    // Same-shape kinds pair datum i with datum i whatever the face order, so
    // they walk the tile straight through (`tt_firmware::float`'s loops).
    let (pd, pa, pb) = (dst as *mut u32, a as *const u32, b as *const u32);
    // SAFETY: three tile slots `Entry::decode` placed inside L1, 1024 words
    // of datums each.
    unsafe {
        match kind {
            dm::kind::ADD => float::add_n(pd, pa, pb, 1024),
            dm::kind::SUB => float::sub_n(pd, pa, pb, 1024),
            dm::kind::MUL => float::mul_n(pd, pa, pb, 1024),
            dm::kind::MUL_SCALAR => float::mul_scalar_n(pd, pa, s, 1024),
            dm::kind::COL_SUM => col_sum(s != 0, dst, a),
            _ => per_datum(kind, dst, a, b),
        }
    }
    // The stores must reach L1 before the mover's next NoC write reads it.
    publish();
}

/// `dm::kind::COL_SUM`: row 0 of `dst` accumulates each column of `a`, rows
/// in order. The addresses are the datums' (past the header).
fn col_sum(first: bool, dst: u64, a: u64) {
    for c in 0..32usize {
        let at = dst + dm::face_index(0, c) as u64 * 4;
        let mut acc = if first { 0 } else { rd(at) };
        for r in 0..32usize {
            acc = float::add(acc, rd(a + dm::face_index(r, c) as u64 * 4));
        }
        wr(at, acc);
    }
}

/// The kinds that need a datum's position, or integer tests: one at a time.
fn per_datum(kind: u32, dst: u64, a: u64, b: u64) {
    for r in 0..32usize {
        for c in 0..32usize {
            let i = dm::face_index(r, c) as u64 * 4;
            let x = rd(a + i);
            let v = match kind {
                // `max(x, 0)`: x itself if positive (as a signed integer, which
                // excludes both zeros, negatives and negative NaNs), +0 for
                // every negative and zero; a positive NaN is `max`'s other
                // argument, +0, too.
                dm::kind::RELU => {
                    if (x as i32) > 0 && x <= 0x7F80_0000 {
                        x
                    } else {
                        0
                    }
                }
                // `out > 0 ? g : 0` in IEEE terms: positive, not zero, not NaN.
                dm::kind::RELU_BACKWARD => {
                    if (x as i32) > 0 && x <= 0x7F80_0000 {
                        rd(b + i)
                    } else {
                        0
                    }
                }
                dm::kind::ADD_ROW => float::add(x, rd(b + dm::face_index(0, c) as u64 * 4)),
                _ => x,
            };
            wr(dst + i, v);
        }
    }
}

/// Post `generation` to the three resident roles and wait for each to
/// acknowledge it (`dm::op::KERNEL`). The roles' programs and descriptors were
/// staged by the host; everything this list moved before is already in L1.
///
/// The roles' acknowledgements are stores by other cores, which do not
/// invalidate this core's L0 data cache (`MemoryOrdering.md:59`): every poll
/// goes through a fence.
fn kernel(generation: u32, programs: [(u32, u32); 3]) -> Result<(), u32> {
    // Each role's resident program first, if named (checked by
    // `Entry::decode`), so the generation that starts the run finds it.
    for (t, (at, len)) in programs.into_iter().enumerate() {
        if at != 0 {
            let mb = Mailbox::of(t as u32);
            wr(mb.program_addr(), at);
            wr(mb.program_len(), len);
        }
    }
    publish();
    for t in 0..3 {
        wr(Mailbox::of(t).generation(), generation);
    }
    publish();
    for t in 0..3 {
        let mb = Mailbox::of(t);
        loop {
            publish();
            if rd(mb.ack()) == generation {
                break;
            }
            if rd(mb.status()) == status::PANICKED {
                return Err(dm::error::ROLE);
            }
        }
    }
    Ok(())
}

/// Record `event` through the tile's timestamper, if this list is traced
/// (`dm::TRACE`). One 128-bit event per store, so the mover's events and the
/// role runners' share one stream without interleaving.
fn trace(on: bool, event: u32, detail: u32) {
    use tt_isa::mailbox::trace as ev;
    use tt_isa::tensix::timestamper as ts;
    if on {
        wr(ts::TIMESTAMP, ts::event_128(ev::token_with(ev::MOVER, event, detail)));
    }
}

/// Run one descriptor to completion.
fn run(me: (u8, u8), d: Descriptor) -> Result<(), u32> {
    issue(me, d)?;
    noc::wait(TXN);
    Ok(())
}

/// Run one list entry: plain entries are issued without waiting; a transposed
/// read waits for its own tile before rearranging it; a kernel or a wait entry
/// first waits for everything before it.
fn exec(me: (u8, u8), usable: u32, w: [u32; 8]) -> Result<(), u32> {
    match Entry::decode(usable, w)? {
        Entry::Move { descriptor, transpose: true } => {
            // Everything before it has landed, and the scratch is free.
            noc::wait(TXN);
            run(me, Descriptor { l1: dm::SCRATCH as u32, ..descriptor })?;
            publish();
            transpose_from_scratch(descriptor.l1 as u64);
            // Visible in L1 before anything else reads the slot.
            publish();
        }
        Entry::Move { descriptor, .. } => issue(me, descriptor)?,
        Entry::Kernel { generation, programs } => {
            // The operands it computes on must have landed.
            noc::wait(TXN);
            publish();
            kernel(generation, programs)?;
        }
        Entry::Wait => {
            noc::wait(TXN);
            publish();
        }
        Entry::Compute { kind, scalar, dst, a, b } => {
            // Its operands may still be arriving.
            noc::wait(TXN);
            publish();
            compute(kind, scalar, dst as u64, a as u64, b as u64);
        }
    }
    Ok(())
}

/// Read list entry `i`.
fn entry(i: u64) -> [u32; 8] {
    let at = dm::LIST + i * dm::ENTRY_BYTES;
    let mut w = [0u32; 8];
    for (k, word) in w.iter_mut().enumerate() {
        *word = rd(at + k as u64 * 4);
    }
    w
}

/// Run the list at `dm::LIST`, entry by entry ([`exec`]). An op record
/// (`dm::record`) is expanded here, on the tile, into the entries the host
/// would otherwise have sent, and each runs exactly as a sent one would --
/// through `Entry::decode` and every refusal in it.
///
/// With `dm::TRACE` set, the list, and each entry or record in it, is
/// bracketed by timestamper events (`tt_isa::mailbox::trace`); a record's
/// expanded moves are not, so the events per list stay bounded by its length.
fn run_list(me: (u8, u8), usable: u32, count: u32) -> Result<(), u32> {
    use tt_isa::mailbox::trace as ev;
    if count > dm::LIST_MAX {
        return Err(dm::error::LENGTH);
    }
    let traced = rd(dm::TRACE) != 0;
    trace(traced, ev::LIST_BEGIN, count);
    let mut i = 0u64;
    while i < count as u64 {
        let head = entry(i);
        let n = record::len(head[0]) as u64;
        trace(traced, ev::ENTRY_BEGIN, head[0]);
        if n == 1 {
            exec(me, usable, head)?;
        } else {
            if i + n > count as u64 {
                return Err(dm::error::LENGTH);
            }
            let mut rec = [[0u32; 8]; 7];
            for k in 0..n {
                rec[k as usize] = entry(i + k);
            }
            record::expand(&rec[..n as usize], |e| exec(me, usable, e))?;
        }
        trace(traced, ev::ENTRY_END, head[0]);
        i += n;
    }
    noc::wait(TXN);
    trace(traced, ev::LIST_END, count);
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
        let result = if rd(dm::OP) == op::LIST {
            run_list(me, usable, rd(dm::LEN))
        } else {
            Descriptor::decode(
                usable,
                rd(dm::OP),
                rd(dm::CHANNEL),
                rd(dm::PORT),
                rd(dm::DRAM_OFFSET),
                rd(dm::L1_ADDR),
                rd(dm::LEN),
            )
            .and_then(|d| run(me, d))
        };
        wr(dm::ERROR, result.err().unwrap_or(dm::error::NONE));
        // The data is in L1 (a read) or acknowledged by the DRAM tile (a write,
        // response-marked) before DONE is published.
        publish();
        wr(dm::DONE, seq);
        publish();
    }
}
