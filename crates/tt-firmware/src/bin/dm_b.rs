//! The RISCV B data mover: GDDR <-> this tile's L1, one descriptor at a time.
//!
//! Protocol and layout are `tt_isa::dm`, which the host shares. Resident: loaded
//! once and left running, so a kernel's operands and results move without the
//! host touching either. Keeps the heartbeat, so the host can tell a stuck
//! mover from a dead core.

#![no_std]
#![no_main]

use tt_firmware::{float, l1_read32, l1_write32, mailbox_word, noc, publish};
use tt_isa::dm::{self, op, record, Descriptor, Entry, Transform};
use tt_isa::mailbox::role::Mailbox;
use tt_isa::mailbox::{offset, status};
use tt_isa::noc::niu::{Command, TxnId, MAX_REQUEST_BYTES};

const TXN: TxnId = match TxnId::new(2) {
    Some(t) => t,
    None => panic!(),
};
/// The barrier's own requests, apart from the moves'.
const BARRIER_TXN: TxnId = match TxnId::new(3) {
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

/// The tile in the scratch slot into the slot at `dst`, its column 0 copied
/// into every column (`dm::op::READ_BROADCAST_COL`): header copied, datum
/// `(r, c)` from `(r, 0)`.
fn broadcast_col0_from_scratch(dst: u64) {
    for w in 0..(dm::TILE_DATA / 4) {
        wr(dst + w * 4, rd(dm::SCRATCH + w * 4));
    }
    let (src, out) = (dm::SCRATCH + dm::TILE_DATA, dst + dm::TILE_DATA);
    for r in 0..32usize {
        let v = rd(src + dm::face_index(r, 0) as u64 * 4);
        for c in 0..32usize {
            wr(out + dm::face_index(r, c) as u64 * 4, v);
        }
    }
}

/// One compute entry over the 1024 datums of three tile slots (`dm::kind`).
fn compute(kind: u32, s: u32, param: u32, dst: u64, a: u64, b: u64) {
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
            dm::kind::COL_SUM => col_sum(s != 0, dm::kind::extent(param) as usize, dst, a),
            dm::kind::FILL_PAD => fill_pad(s, param, dst),
            dm::kind::ADD_SCALAR => {
                for i in 0..1024 {
                    *pd.add(i) = float::add(*pa.add(i), s);
                }
            }
            dm::kind::COPY => {
                for i in 0..1024 {
                    *pd.add(i) = *pa.add(i);
                }
            }
            _ => per_datum(kind, dst, a, b),
        }
    }
    // The stores must reach L1 before the mover's next NoC write reads it.
    publish();
}

/// `dm::kind::COL_SUM`: row 0 of `dst` accumulates each column of `a`'s
/// first `rows` rows, in order; the first tile of a column also zeroes `dst`'s
/// other rows. The addresses are the datums' (past the header).
fn col_sum(first: bool, rows: usize, dst: u64, a: u64) {
    for c in 0..32usize {
        let at = dst + dm::face_index(0, c) as u64 * 4;
        let mut acc = if first { 0 } else { rd(at) };
        for r in 0..rows {
            acc = float::add(acc, rd(a + dm::face_index(r, c) as u64 * 4));
        }
        wr(at, acc);
        if first {
            for r in 1..32usize {
                wr(dst + dm::face_index(r, c) as u64 * 4, 0);
            }
        }
    }
}

/// `dm::kind::FILL_PAD`: `v` at every datum of `dst` outside its first
/// `param & 0xff` rows and `param >> 8` columns (`0` meaning 32).
fn fill_pad(v: u32, param: u32, dst: u64) {
    let rows = dm::kind::extent(param & 0xff) as usize;
    let cols = dm::kind::extent(param >> 8) as usize;
    for r in 0..32usize {
        for c in 0..32usize {
            if r >= rows || c >= cols {
                wr(dst + dm::face_index(r, c) as u64 * 4, v);
            }
        }
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

/// `dm::op::BARRIER`: one NoC atomic increment of the coordinator's counter,
/// then NoC reads of it until it reaches `target`. The reads land in this
/// tile's L1, which the NoC writes without invalidating the L0 cache: every
/// check goes through a fence (`publish`).
fn barrier(me: (u8, u8), target: u32, x: u8, y: u8) -> Result<(), u32> {
    use tt_isa::noc::niu::Endpoint;
    let counter = Endpoint { x, y, addr: dm::BARRIER_COUNTER as u32 };
    noc::issue(
        &Command::AtomicIncrement { to: counter, value: 1, ret_local: dm::BARRIER_RET as u32 },
        me,
        BARRIER_TXN,
    )
    .map_err(|_| dm::error::ALIGNMENT)?;
    noc::wait(BARRIER_TXN);
    loop {
        noc::issue(
            &Command::Read { from: counter, to_local: dm::BARRIER_POLL as u32, len: 4 },
            me,
            BARRIER_TXN,
        )
        .map_err(|_| dm::error::ALIGNMENT)?;
        noc::wait(BARRIER_TXN);
        publish();
        if (rd(dm::BARRIER_POLL).wrapping_sub(target) as i32) >= 0 {
            return Ok(());
        }
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
        Entry::Move {
            descriptor,
            transform: transform @ (Transform::Transpose | Transform::BroadcastCol0),
        } => {
            // Everything before it has landed, and the scratch is free.
            noc::wait(TXN);
            run(me, Descriptor { l1: dm::SCRATCH as u32, ..descriptor })?;
            publish();
            if transform == Transform::Transpose {
                transpose_from_scratch(descriptor.l1 as u64);
            } else {
                broadcast_col0_from_scratch(descriptor.l1 as u64);
            }
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
        Entry::Barrier { target, x, y } => {
            // Everything this unit moved before it has landed.
            noc::wait(TXN);
            publish();
            barrier(me, target, x, y)?;
        }
        Entry::Compute {
            kind,
            param,
            scalar,
            dst,
            a,
            b,
        } => {
            // Its operands may still be arriving.
            noc::wait(TXN);
            publish();
            compute(kind, scalar, param, dst as u64, a as u64, b as u64);
        }
    }
    Ok(())
}

/// Read list entry `i` of the ring.
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
    run_list_at(me, usable, 0, count)
}

/// [`run_list`] of the `count` entries from ring entry `first`.
fn run_list_at(me: (u8, u8), usable: u32, first: u32, count: u32) -> Result<(), u32> {
    use tt_isa::mailbox::trace as ev;
    if count > dm::LIST_MAX || first + count > dm::LIST_MAX {
        return Err(dm::error::LENGTH);
    }
    let traced = rd(dm::TRACE) != 0;
    trace(traced, ev::LIST_BEGIN, count);
    let mut i = 0u64;
    while i < count as u64 {
        let head = entry(first as u64 + i);
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
                rec[k as usize] = entry(first as u64 + i + k);
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
        // The queue first: lists the host enqueued, in order, until one
        // fails (which stops the queue until the host restarts the mover).
        let done = rd(dm::QUEUE_DONE);
        if done != rd(dm::QUEUE_HEAD) && rd(dm::QUEUE_ERROR) == dm::error::NONE {
            let slot = rd(dm::QUEUE_SLOTS + (done % dm::QUEUE_LEN) as u64 * 4);
            let result = run_list_at(me, usable, slot & 0xffff, slot >> 16);
            // Everything the list moved has landed before it is reported.
            publish();
            match result {
                Ok(()) => wr(dm::QUEUE_DONE, done.wrapping_add(1)),
                Err(code) => {
                    wr(dm::QUEUE_ERROR_AT, done.wrapping_add(1));
                    wr(dm::QUEUE_ERROR, code);
                }
            }
            publish();
            continue;
        }
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
