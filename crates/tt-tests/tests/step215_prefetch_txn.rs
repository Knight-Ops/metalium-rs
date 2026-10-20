//! decode-to-bandwidth.md Phase 8, Step 3: RISCV B's prefetch reads on a transaction ID of their own
//! (`dm::TXN_PREFETCH`). A `GATHER` with `record::GATHER_PREFETCH` issues its reads there, so the
//! entries that drain the mover's own ID -- every kernel launch, barrier, `WAIT` and chunk end --
//! leave them in flight; only a `WAIT` with `dm::WAIT_PREFETCH` drains them, and so does the end of
//! a list (a list is reported done when everything it read has landed).
//!
//! One list on one tile: a prefetch `GATHER` of `B` (`A` resident: only `B` is read), 128 tiles
//! into L1; a plain `WAIT` (what every draining entry does); a snapshot of the last eight `B`
//! slots by `COPY_WORDS` (which waits for the mover's own ID only), the last first; a prefetch
//! `WAIT`; a second snapshot. The claims:
//! - **Same reads.** The flag changes no read: the host's expansion of the record with and without
//!   it is the same entries, and every `B` slot holds what those entries name, from the DRAM
//!   written beforehand (the oracle is the host's own expansion and the mirror).
//! - **Drained by its wait.** The snapshot after the prefetch `WAIT` is the slots' bytes.
//! - **And by the list's end.** The same list without the prefetch `WAIT`: once it is done, every
//!   slot holds its bytes.
//! - **Pumped, not issued at once.** A resident-`A` prefetch is queued (`record::PrefetchCursor`):
//!   the mover issues at most its ID's cap of reads when it arrives and the rest while it waits
//!   (barriers, the roles, the writer) or at its `WAIT`. With no wait between, the snapshot after
//!   the plain `WAIT` misses the last slots, every round, on both targets; without the flag (the
//!   control) it is exact.
//! - **On an ID of its own** (silicon only: ttsim completes each request before the next is issued,
//!   divergence row 72). With the mover's own in-flight cap at one, a plain gather stalls on every
//!   read after the first (the control, as `step49`), a prefetch on fewer than half as many: its
//!   reads are not under that ID's cap, so not under its drains either.

use tt_device::tlb::WindowKind;
use tt_device::Window;
use tt_isa::dm::{self, op, record, TILE_SLOT};
use tt_isa::noc::Noc0;
use tt_kernels::dm::DataMover;
use tt_tests::backend::{GATE_TILE, ON_SILICON};
use tt_tests::harness::{in_device, tile, Dev};

/// Where `B`'s tiles lie in every channel.
const DRAM_AT: u64 = (300 << 20) + 0x200_0000;
const CHANNELS: usize = 8;
/// `B`: `K` 32 tiles by 4 columns, 128 tiles (532 KB), ~75 us at one unit's ~7 GB/s.
const KT: u32 = 32;
const COLS: u32 = 4;
/// Where the gather writes `B`, and where the snapshots go: after the last slot.
const B_AT: u32 = 0x2_0000;
const SNAP_AT: u32 = B_AT + KT * COLS * TILE_SLOT as u32;
/// Slots each snapshot copies (the last ones the gather reads), 1,024 words each.
const SNAP_SLOTS: u32 = 8;
const POISON: u8 = 0xA5;

fn pattern(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u8
        })
        .collect()
}

/// `B`'s reference: tile `t` in channel `t % 8`, slot `t / 8` from [`DRAM_AT`].
fn b_ref() -> record::TensorRef {
    record::TensorRef {
        n: CHANNELS as u8,
        channels: std::array::from_fn(|c| c as u8),
        base: [DRAM_AT as u32; 8],
        first: 0,
        ct: COLS,
    }
}

/// The `GATHER` of `B` alone (`A` resident), with `extra` flags.
fn gather(extra: u32) -> Vec<[u32; 8]> {
    let b = b_ref().encode();
    let flags = record::GATHER_A_RESIDENT | KT << 8 | extra;
    vec![
        [record::GATHER, flags, B_AT, B_AT, 0, 1, 0, COLS],
        b[0],
        b[1],
        b[0],
        b[1],
    ]
}

/// What the gather leaves in L1 from [`B_AT`] (the slots its reads name), from the host's own
/// expansion of the record and the DRAM mirror.
fn expected(mirror: &[Vec<u8>]) -> Vec<u8> {
    let span = (KT * COLS) as usize * TILE_SLOT as usize;
    let mut want = vec![POISON; span];
    record::expand_reads(
        &gather(0),
        |_| 0,
        |e| {
            assert_eq!(e[0], op::READ, "a plain read");
            let (c, off, l1, len) = (e[1] as usize, e[3] as u64, e[4] as usize, e[5] as usize);
            let from = (off - DRAM_AT) as usize;
            want[l1 - B_AT as usize..][..len].copy_from_slice(&mirror[c][from..from + len]);
            Ok(())
        },
    )
    .unwrap();
    want
}

/// The snapshot entries: the last [`SNAP_SLOTS`] slots' 1,024 words each, to `to`, the last slot
/// (the last read issued) first.
fn snapshot(to: u32) -> Vec<[u32; 8]> {
    (0..SNAP_SLOTS)
        .rev()
        .map(|k| {
            let slot = KT * COLS - SNAP_SLOTS + k;
            [
                op::COPY_WORDS,
                B_AT + slot * TILE_SLOT as u32,
                to + k * 4096,
                1024,
                4,
                4,
                0,
                0,
            ]
        })
        .collect()
}

struct Run {
    slots: Vec<u8>,
    before: Vec<u8>,
    after: Vec<u8>,
}

/// Run the list (the prefetch `WAIT` left out if `wait` is false) from poisoned L1.
fn run(d: &mut Dev<'_>, w: &Window, m: &mut DataMover<Noc0>, flag: u32, wait: bool) -> Run {
    let t = m.tile();
    let span = (KT * COLS) as usize * TILE_SLOT as usize + 2 * SNAP_SLOTS as usize * 4096;
    d.l1_write(w, t, B_AT as u64, &vec![POISON; span]).unwrap();
    let before_at = SNAP_AT;
    let after_at = SNAP_AT + SNAP_SLOTS * 4096;
    let mut list = gather(flag);
    // What every draining entry does: wait for the mover's own ID. (A read of its own before it
    // would take a round trip, long enough for the last prefetch reads to land.)
    list.push([op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
    list.extend(snapshot(before_at));
    if wait {
        list.push([op::WAIT, dm::WAIT_PREFETCH, 0, 0, 0, 0, 0, 0]);
        list.extend(snapshot(after_at));
    }
    m.run_list(d, w, &list).unwrap();
    let mut all = vec![0u8; span];
    d.l1_read(w, t, B_AT as u64, &mut all).unwrap();
    let slots_len = (KT * COLS) as usize * TILE_SLOT as usize;
    let snap = SNAP_SLOTS as usize * 4096;
    Run {
        slots: all[..slots_len].to_vec(),
        before: all[slots_len..][..snap].to_vec(),
        after: all[slots_len + snap..][..snap].to_vec(),
    }
}

/// The last [`SNAP_SLOTS`] slots' first 4,096 bytes of `slots`, as the snapshots copy them.
fn last_slots(slots: &[u8]) -> Vec<u8> {
    (0..SNAP_SLOTS as usize)
        .flat_map(|k| {
            let slot = (KT * COLS) as usize - SNAP_SLOTS as usize + k;
            slots[slot * TILE_SLOT as usize..][..4096].to_vec()
        })
        .collect()
}

#[test]
fn a_prefetch_stays_in_flight_past_the_movers_drains_until_its_wait() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let per = (KT * COLS) as usize / CHANNELS * TILE_SLOT as usize;
        let mirror: Vec<Vec<u8>> = dram
            .channels()
            .take(CHANNELS)
            .map(|ch| {
                let p = pattern(per, 0x9F + ch.index() as u64);
                d.dram_write(&w4, ch.range(DRAM_AT, per as u64).unwrap(), &p)
                    .unwrap();
                p
            })
            .collect();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let mut m = DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap();
        let want = expected(&mirror);
        let want_last = last_slots(&want);

        // Same reads: the flag changes no entry the record expands to.
        let entries = |flags| {
            let mut v = Vec::new();
            record::expand_reads(
                &gather(flags),
                |_| 0,
                |e| {
                    v.push(e);
                    Ok(())
                },
            )
            .unwrap();
            v
        };
        assert_eq!(entries(record::GATHER_PREFETCH), entries(0));

        // The control: without the flag the plain `WAIT` drains the gather.
        let plain = run(d, &w, &mut m, 0, true);
        assert!(plain.slots == want, "without the flag: the slots");
        assert!(
            plain.before == want_last,
            "without the flag: drained by the plain WAIT"
        );

        let mut in_flight = 0;
        for round in 0..4 {
            let r = run(d, &w, &mut m, record::GATHER_PREFETCH, true);
            assert!(r.slots == want, "round {round}: the slots");
            assert!(
                r.after == want_last,
                "round {round}: drained by the prefetch WAIT"
            );
            if r.before != want_last {
                in_flight += 1;
            }
        }
        // Queued, not issued: a resident-`A` prefetch issues at most its ID's cap of reads when it
        // arrives and the rest from the mover's waits or its own `WAIT` (`record::PrefetchCursor`,
        // the pump); this list has no wait between, so the last slots are not read yet when the
        // first snapshot copies them, on both targets.
        println!("rounds whose last slots were not read by the plain WAIT: {in_flight} of 4");
        assert_eq!(in_flight, 4, "the prefetch's last reads wait for its WAIT");

        // On another ID: the mover's own cap of one throttles a plain gather on every read after
        // the first, but not a prefetch, whose ID has a cap of its own (`TILE_IN_FLIGHT_CAP`).
        // The stalls are counted on silicon only (row 72).
        m.set_in_flight_cap(d, &w, 1).unwrap();
        let mut stalls = [0u32; 2];
        for (k, flag) in [0, record::GATHER_PREFETCH].into_iter().enumerate() {
            let before = m.throttle(d, &w).unwrap();
            let r = run(d, &w, &mut m, flag, true);
            assert!(
                r.slots == want,
                "under a cap of one, flag {flag:#x}: the slots"
            );
            stalls[k] = m.throttle(d, &w).unwrap().stalls - before.stalls;
        }
        m.set_in_flight_cap(d, &w, dm::TILE_IN_FLIGHT_CAP).unwrap();
        println!(
            "under the mover's cap of one: a plain gather stalled {}, a prefetch {}",
            stalls[0], stalls[1]
        );
        if ON_SILICON {
            assert!(
                stalls[0] >= KT * COLS - 1,
                "the control: a plain gather waits on every read after the first"
            );
            assert!(
                2 * stalls[1] < stalls[0],
                "a prefetch's reads are not on the mover's own ID"
            );
        }

        // The list's end drains it too.
        let r = run(d, &w, &mut m, record::GATHER_PREFETCH, false);
        assert!(
            r.slots == want,
            "without its WAIT: landed when the list is done"
        );
    });
}
