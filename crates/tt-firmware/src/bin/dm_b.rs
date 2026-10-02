//! The RISCV B data mover: GDDR <-> this tile's L1, one descriptor at a time.
//!
//! Protocol and layout are `tt_isa::dm`, which the host shares. Resident: loaded
//! once and left running, so a kernel's operands and results move without the
//! host touching either. Keeps the heartbeat, so the host can tell a stuck
//! mover from a dead core.

#![no_std]
#![no_main]

use tt_firmware::{l1_read32, l1_write32, mailbox_word, noc, publish};
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
#[link_section = ".text.hot"]
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

/// The tile counter's low word, which ttsim models (divergence row 54). The
/// clock `noc` times waits for room under the in-flight cap with: read only
/// when a request has to wait.
fn wall_clock() -> u32 {
    rd(tt_isa::tensix::timestamper::WALL_CLOCK_L)
}

/// After a list: if any of its requests waited for room under the in-flight
/// cap, add them where the host reads them (`dm::THROTTLE_STALLS`,
/// `THROTTLE_CYCLES`) and, in a traced list, record one `THROTTLE` event with
/// the cycles. `seen` is what was published last. One compare per list when
/// nothing waited.
fn publish_stalls(seen: &mut noc::Stalls, traced: bool) {
    let now = noc::stalls();
    if now.count != seen.count {
        stalled(*seen, now, traced);
        *seen = now;
    }
}

#[cold]
#[inline(never)]
fn stalled(seen: noc::Stalls, now: noc::Stalls, traced: bool) {
    use tt_isa::mailbox::trace as ev;
    let cycles = now.cycles.wrapping_sub(seen.cycles);
    wr(dm::THROTTLE_STALLS, rd(dm::THROTTLE_STALLS).wrapping_add(now.count.wrapping_sub(seen.count)));
    wr(dm::THROTTLE_CYCLES, rd(dm::THROTTLE_CYCLES).wrapping_add(cycles));
    trace(traced, ev::THROTTLE, cycles.min(ev::DETAIL_MAX));
}

/// Transpose the tile in the scratch slot into the slot at `dst`: header
/// copied, datum `(r, c)` from `(c, r)`, both in face order -- face `(fr, fc)`
/// of the result is face `(fc, fr)` of the source, transposed.
#[inline(never)]
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
#[inline(never)]
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

/// `dm::op::FILL`: `v` at every datum of the tile slot `dst` outside its
/// first `param & 0xff` rows and `param >> 8` columns (`0` meaning 32).
/// Stores only: the mover moves data and does no arithmetic.
#[inline(never)]
fn fill(v: u32, param: u32, dst: u64) {
    let dst = dst + dm::TILE_DATA;
    let rows = dm::fill::extent(param & 0xff) as usize;
    let cols = dm::fill::extent(param >> 8) as usize;
    for r in 0..32usize {
        for c in 0..32usize {
            if r >= rows || c >= cols {
                wr(dst + dm::face_index(r, c) as u64 * 4, v);
            }
        }
    }
    // The stores must reach L1 before the mover's next NoC write reads it.
    publish();
}

/// Post `generation` to the three resident roles and wait for each to
/// acknowledge it (`dm::op::KERNEL`). The roles' programs and descriptors were
/// staged by the host; everything this list moved before is already in L1.
///
/// The roles' acknowledgements are stores by other cores, which do not
/// invalidate this core's L0 data cache (`MemoryOrdering.md:59`): every poll
/// goes through a fence.
#[inline(never)]
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
    let traced = rd(dm::TRACE) != 0;
    trace(traced, tt_isa::mailbox::trace::KICK, generation);
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
    trace(traced, tt_isa::mailbox::trace::ROLES_DONE, generation);
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
#[inline(never)]
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
#[link_section = ".text.hot"]
fn run(me: (u8, u8), d: Descriptor) -> Result<(), u32> {
    issue(me, d)?;
    noc::wait(TXN);
    Ok(())
}

/// Run one list entry: plain entries are issued without waiting; a transposed
/// read waits for its own tile before rearranging it; a kernel or a wait entry
/// first waits for everything before it.
#[link_section = ".text.hot"]
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
        Entry::Poke { address, value } => {
            // The roles read the word only once a later `KERNEL` posts their
            // generation.
            wr(address as u64, value);
            publish();
        }
        // Only as a list of its own, which `run_list_at` runs.
        Entry::Call { .. } => return Err(IS_CALL),
        Entry::Fill { value, param, dst } => {
            // The tile it fills may still be arriving.
            noc::wait(TXN);
            publish();
            fill(value, param, dst as u64);
        }
    }
    Ok(())
}

/// Read list entry `i` of the ring.
fn entry_at(at: u64) -> [u32; 8] {
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
    if count > dm::LIST_MAX || first + count > dm::LIST_MAX {
        return Err(dm::error::LENGTH);
    }
    let base = dm::LIST + first as u64 * dm::ENTRY_BYTES;
    // A `CALL` is a list of its own (a replay's), run here rather than from
    // the list loop: there it makes the loop recursive. `exec` checks it, as
    // it does every entry -- `Entry::decode` with a second caller is no longer
    // inlined into `exec`, which costs every entry ~0.04 us
    // (`silicon_perf::mover_read_shapes`).
    let head = entry_at(base);
    let result = if count == 1 && head[0] == op::CALL {
        match exec(me, usable, head) {
            Err(IS_CALL) => call(me, usable, head[1], head[2], head[3], head[4], head[5]),
            r => r,
        }
    } else {
        run_entries(me, usable, base, count)
    };
    // One anywhere else.
    match result {
        Err(IS_CALL) => Err(dm::error::OP),
        r => r,
    }
}

/// What [`exec`] returns for a valid `CALL`, which only [`run_list_at`] runs:
/// no `dm::error` code.
const IS_CALL: u32 = u32::MAX;

/// The `count` entries at `base` in L1, in order: a list's, or a trace chunk's
/// ([`call`]).
#[link_section = ".text.hot"]
fn run_entries(me: (u8, u8), usable: u32, base: u64, count: u32) -> Result<(), u32> {
    use tt_isa::mailbox::trace as ev;
    let traced = rd(dm::TRACE) != 0;
    trace(traced, ev::LIST_BEGIN, count);
    let at = |i: u64| entry_at(base + i * dm::ENTRY_BYTES);
    let mut i = 0u64;
    while i < count as u64 {
        let head = at(i);
        let n = record::len(head[0]) as u64;
        trace(traced, ev::ENTRY_BEGIN, head[0]);
        if n == 1 {
            exec(me, usable, head)?;
        } else {
            if i + n > count as u64 {
                return Err(dm::error::LENGTH);
            }
            run_record(me, usable, base + i * dm::ENTRY_BYTES, n as usize)?;
        }
        trace(traced, ev::ENTRY_END, head[0]);
        i += n;
    }
    noc::wait(TXN);
    trace(traced, ev::LIST_END, count);
    Ok(())
}

/// The `n`-entry op record at `at`, expanded ([`record::expand`]). Out of the
/// list loop, as the rest of the cold paths are, so the loop and [`exec`]'s
/// move path sit together in `.text.hot` (`sections.x`).
#[inline(never)]
fn run_record(me: (u8, u8), usable: u32, at: u64, n: usize) -> Result<(), u32> {
    let mut rec = [[0u32; 8]; 7];
    for (k, e) in rec[..n].iter_mut().enumerate() {
        *e = entry_at(at + k as u64 * dm::ENTRY_BYTES);
    }
    record::expand(&rec[..n], |e| exec(me, usable, e))
}

/// [`dm::op::CALL`]: a trace's entries from GDDR, a chunk at a time into
/// `dm::TRACE_CHUNK`, each chunk run by the list's own loop ([`run_entries`])
/// once it has landed. What a replay supplies is patched into the chunk first:
/// a top-level `KERNEL`'s generation plus `generation_base`, a `BARRIER`'s
/// target plus `barrier_base`. A `CALL` in a trace is refused, as is a record
/// the capture let cross a chunk.
#[cold]
#[inline(never)]
fn call(
    me: (u8, u8),
    usable: u32,
    channel: u32,
    offset: u32,
    count: u32,
    generation_base: u32,
    barrier_base: u32,
) -> Result<(), u32> {
    let mut done = 0u32;
    while done < count {
        let n = (count - done).min(dm::TRACE_CHUNK_ENTRIES);
        let read = [
            op::READ,
            channel,
            0,
            offset + done * dm::ENTRY_BYTES as u32,
            dm::TRACE_CHUNK as u32,
            n * dm::ENTRY_BYTES as u32,
            0,
            0,
        ];
        exec(me, usable, read)?;
        noc::wait(TXN);
        // The chunk is in L1, written by the NoC past the L0 cache.
        publish();
        let mut i = 0u32;
        while i < n {
            let a = dm::TRACE_CHUNK + i as u64 * dm::ENTRY_BYTES;
            let head = rd(a);
            let base = match head {
                op::KERNEL => generation_base,
                op::BARRIER => barrier_base,
                op::CALL => return Err(dm::error::OP),
                _ => 0,
            };
            if base != 0 {
                wr(a + 4, rd(a + 4).wrapping_add(base));
            }
            i += record::len(head) as u32;
        }
        if i != n {
            return Err(dm::error::LENGTH);
        }
        publish();
        run_entries(me, usable, dm::TRACE_CHUNK, n)?;
        done += n;
    }
    Ok(())
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    let me = (rd(dm::MY_X) as u8, rd(dm::MY_Y) as u8);
    let usable = rd(dm::USABLE);
    let mut beat: u32 = 0;
    noc::set_clock(wall_clock);
    let mut stalls = noc::stalls();
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
            noc::set_cap(TXN, rd(dm::IN_FLIGHT_CAP));
            let slot = rd(dm::QUEUE_SLOTS + (done % dm::QUEUE_LEN) as u64 * 4);
            let result = run_list_at(me, usable, slot & 0xffff, slot >> 16);
            publish_stalls(&mut stalls, rd(dm::TRACE) != 0);
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
        noc::set_cap(TXN, rd(dm::IN_FLIGHT_CAP));
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
        publish_stalls(&mut stalls, rd(dm::TRACE) != 0);
        wr(dm::ERROR, result.err().unwrap_or(dm::error::NONE));
        // The data is in L1 (a read) or acknowledged by the DRAM tile (a write,
        // response-marked) before DONE is published.
        publish();
        wr(dm::DONE, seq);
        publish();
    }
}
