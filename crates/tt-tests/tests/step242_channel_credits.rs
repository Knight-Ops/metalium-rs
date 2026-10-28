//! Cross-core channels: a producer tile hands pages to consumer tiles through a ring, each page
//! granted by a NoC atomic credit (`op::CREDIT`) and each slot freed by one back, every wait a
//! fenced poll of a credit word (`op::WAIT_CREDIT`) (`docs/plans/dataflow-compiler.md` F2,
//! `tt_kernels::channel::Ring`).
//!
//! The claims:
//! - **A remote ring hands every page over in order, whole.** RISCV NC on the producer writes
//!   pages into a depth-2 ring in the consumer's L1 and credits it; RISCV B on the consumer waits for
//!   each page, copies it out and frees its slot. Every page copied out equals the producer's source
//!   byte for byte, though the ring's slots are reused lap after lap: a consumer that read before its
//!   credit, or a producer that wrote before its slot was freed, would leave poison or a later page
//!   in the copy. Both credit words end at the page count.
//! - **A broadcast ring does the same for every tile of a rectangle**: one multicast a request,
//!   one credit to each consumer, and the producer reuses a slot only once all four have freed it.
//! - **A GDDR ring does the same through GDDR**: NC writes the page to GDDR and credits the
//!   consumer, whose B fetches it after its credit.
//! - **One credit too few stalls; it never races.** A consumer waiting for one page more than the
//!   producer sends is held at its credit wait, which the host reads as a typed stall (have, want);
//!   the page it waits for is never copied (its copy is still poison), and an abort cancels the wait,
//!   failing the list with the abort's code.
//! - **The tile refuses a malformed credit by itself** (sent past the host's check,
//!   `enqueue_raw`): a credit of nothing, a credit word in a mover's mailbox or misaligned, a wait on
//!   the streams' counters; and the host refuses a credit to a tile its session does not name.
//!
//! What ttsim cannot show: its NoC requests complete before the next is issued (divergence row 72),
//! so a credit sent before its data's acknowledgement cannot race there. That order is held by the
//! firmware (a credit waits for the mover's requests first) and by the producer's list
//! (`Ring::check_send`, whose negative controls are `tt_kernels::channel`'s unit tests); silicon
//! runs the same rings with more pages and larger ones, where a race would show.

use tt_device::tlb::WindowKind;
use tt_device::Window;
use tt_isa::dm::{self, op, Mover};
use tt_isa::noc::niu::Niu;
use tt_isa::noc::Noc0;
use tt_kernels::channel::{self, Ring, Slots, Stall};
use tt_kernels::dm::{DataMover, DmError, L1Writes};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile, Dev, ON_SILICON};

const POISON: u8 = 0xA5;
const PRODUCER: (u8, u8) = GATE_TILE;
const CONSUMER: (u8, u8) = (GATE_TILE.0 + 1, GATE_TILE.1);
/// The broadcast's rectangle, beside the producer and not holding it.
const RECT: [(u8, u8); 4] = [
    (GATE_TILE.0 + 1, GATE_TILE.1),
    (GATE_TILE.0 + 2, GATE_TILE.1),
    (GATE_TILE.0 + 1, GATE_TILE.1 + 1),
    (GATE_TILE.0 + 2, GATE_TILE.1 + 1),
];
/// The producer's distinct source pages, one after another; page `k` is sent from source page
/// `k % SOURCES`, so a page that arrives two laps stale (or as poison) differs from the one due.
const SRC: u32 = 0x2_0000;
const SOURCES: u32 = 12;
/// Where a consumer copies the pages it is handed: below the window, past the source.
const OUT: u32 = 0x6_0000;
/// The window peers may write: the top of every tile's arena. The ring's slots first, then the
/// credit words.
const WINDOW_LEN: u64 = 128 << 10;

fn window() -> std::ops::Range<u64> {
    tt_isa::l1::DATA.end - WINDOW_LEN..tt_isa::l1::DATA.end
}

/// The credit words: `received` on every consumer, `freed` on the producer, each alone in its 16
/// bytes, past the ring.
fn received() -> u32 {
    (window().end - 64) as u32
}
fn freed() -> u32 {
    (window().end - 32) as u32
}

/// Pages a ring hands over, and their bytes: on silicon more laps and pages past one request.
fn pages() -> (u32, u32) {
    if ON_SILICON {
        (24, 20 << 10)
    } else {
        (7, 8 << 10)
    }
}
const _: () = assert!(SRC + SOURCES * (20 << 10) <= OUT);
const _: () = assert!(OUT as u64 + 24 * (20 << 10) <= tt_isa::l1::DATA.end - WINDOW_LEN);

/// The bytes page `k` carries.
fn page(source: &[u8], k: u32, page_bytes: u32) -> &[u8] {
    let at = ((k % SOURCES) * page_bytes) as usize;
    &source[at..at + page_bytes as usize]
}

fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as u8
        })
        .collect()
}

fn start(d: &mut Dev<'_>, w: &Window, at: (u8, u8), mover: Mover) -> DataMover<Noc0> {
    let dram = d.dram_grid(w).unwrap();
    let t = tile(d, at.0, at.1);
    let image = match mover {
        Mover::B => tt_firmware_images::DM_B.1,
        _ => tt_firmware_images::DM_NC.1,
    };
    DataMover::start_on(d, w, t, &dram, mover, image).unwrap()
}

fn poison(d: &mut Dev<'_>, w: &Window, at: (u8, u8), from: u64, len: usize) {
    let t = tile(d, at.0, at.1);
    d.l1_write(w, t, from, &vec![POISON; len]).unwrap();
}

fn read(d: &mut Dev<'_>, w: &Window, at: (u8, u8), from: u64, len: usize) -> Vec<u8> {
    let t = tile(d, at.0, at.1);
    let mut back = vec![0u8; len];
    d.l1_read(w, t, from, &mut back).unwrap();
    back
}

/// A consumer B's copy of `bytes` at `from` to `to`, a word at a time, at most 1024 an entry.
fn copy(from: u32, to: u32, bytes: u32) -> Vec<[u32; 8]> {
    (0..bytes)
        .step_by(4096)
        .map(|at| {
            let words = (bytes - at).min(4096) / 4;
            [op::COPY_WORDS, from + at, to + at, words, 4, 4, 0, 0]
        })
        .collect()
}

/// A ring to `consumers` in their L1 (or, with `gddr`, through GDDR).
fn ring(consumers: Vec<(u8, u8)>, gddr: Option<u32>) -> Ring {
    let (_, page_bytes) = pages();
    Ring {
        producer: PRODUCER,
        consumers,
        depth: 2,
        page_bytes,
        slots: match gddr {
            Some(channel) => Slots::Gddr {
                channel,
                port: 0,
                offset: 0x100_0000,
            },
            None => Slots::L1 {
                base: window().start as u32,
            },
        },
        received: received(),
        freed: freed(),
    }
}

/// Each tile's movers started, B and NC as a session has them, the consumers' credit words and the
/// producer's zeroed and their abort words clear (`channel::prepare`), the producer's source
/// written and every consumer's ring and copy poisoned. The producer's NC writes the consumers'
/// window through NoC #1; each consumer's B credits the producer's.
struct Rig {
    producer: DataMover<Noc0>,
    consumers: Vec<DataMover<Noc0>>,
    others: Vec<DataMover<Noc0>>,
    source: Vec<u8>,
}

fn rig(d: &mut Dev<'_>, w: &Window, r: &Ring, sent: u32) -> Rig {
    let (count, page_bytes) = pages();
    let mut producer = start(d, w, PRODUCER, Mover::NC);
    producer
        .set_l1_writes(L1Writes {
            tiles: r.consumers.clone(),
            window: window(),
        })
        .unwrap();
    producer.set_write_noc(d, w, Niu::Noc1).unwrap();
    let mut others = vec![start(d, w, PRODUCER, Mover::B)];
    let mut consumers = Vec::new();
    for &c in &r.consumers {
        let mut b = start(d, w, c, Mover::B);
        b.set_l1_writes(L1Writes {
            tiles: vec![PRODUCER],
            window: window(),
        })
        .unwrap();
        consumers.push(b);
        others.push(start(d, w, c, Mover::NC));
        let t = tile(d, c.0, c.1);
        channel::prepare(d, w, t, &[r.received]).unwrap();
        poison(d, w, c, window().start, 2 * page_bytes as usize);
        poison(d, w, c, OUT as u64, (count * page_bytes) as usize);
    }
    let t = tile(d, PRODUCER.0, PRODUCER.1);
    channel::prepare(d, w, t, &[r.freed]).unwrap();
    let source = pattern((sent.min(SOURCES) * page_bytes) as usize, 0x242 + sent);
    d.l1_write(w, t, SRC as u64, &source).unwrap();
    Rig {
        producer,
        consumers,
        others,
        source,
    }
}

/// The producer's list for `sent` pages, checked against the ring's protocol.
fn producer_list(r: &Ring, sent: u32) -> Vec<[u32; 8]> {
    let list: Vec<[u32; 8]> = (0..sent)
        .flat_map(|k| r.send(k, SRC + (k % SOURCES) * r.page_bytes).unwrap())
        .collect();
    r.check_send(&list).unwrap();
    list
}

/// A consumer B's list for `wanted` pages: wait, take the page out (copy it, or fetch it from
/// GDDR), free its slot.
fn consumer_list(r: &Ring, wanted: u32) -> Vec<[u32; 8]> {
    let mut list = Vec::new();
    for k in 0..wanted {
        let to = OUT + k * r.page_bytes;
        match r.slots {
            Slots::Gddr { .. } => list.extend(r.fetch(k, to).unwrap()),
            Slots::L1 { .. } => {
                list.push(r.receive(k));
                list.extend(copy(r.slot(k), to, r.page_bytes));
            }
        }
        list.push(r.free());
    }
    list
}

/// Queue at most a list's entries at a time, in order.
fn enqueue_all(d: &mut Dev<'_>, w: &Window, m: &mut DataMover<Noc0>, list: &[[u32; 8]]) -> u32 {
    let mut last = 0;
    for chunk in list.chunks(dm::LIST_MAX as usize / 2) {
        last = m.enqueue(d, w, chunk).unwrap();
    }
    last
}

fn stop(d: &mut Dev<'_>, w: &Window, rig: Rig) {
    rig.producer.stop(d, w).unwrap();
    for m in rig.consumers.into_iter().chain(rig.others) {
        m.stop(d, w).unwrap();
    }
}

/// Run a ring of `pages()` pages end to end and check every consumer's copies and both counts.
fn hand_over(d: &mut Dev<'_>, w: &Window, r: &Ring) {
    let (count, page_bytes) = pages();
    let mut rig = rig(d, w, r, count);
    let consumer = consumer_list(r, count);
    let mut waits = Vec::new();
    for m in &mut rig.consumers {
        waits.push(enqueue_all(d, w, m, &consumer));
    }
    let last = enqueue_all(d, w, &mut rig.producer, &producer_list(r, count));
    for (m, n) in rig.consumers.iter_mut().zip(waits) {
        m.wait_for(d, w, n).unwrap();
    }
    rig.producer.wait_for(d, w, last).unwrap();
    let bytes = (count * page_bytes) as usize;
    let want: Vec<u8> = (0..count)
        .flat_map(|k| page(&rig.source, k, page_bytes).to_vec())
        .collect();
    for &c in &r.consumers {
        let got = read(d, w, c, OUT as u64, bytes);
        let wrong = got.iter().zip(&want).position(|(a, b)| a != b);
        assert!(
            wrong.is_none(),
            "{:?}: consumer {c:?}'s copy differs at page {:?}",
            r.slots,
            wrong.map(|at| at / page_bytes as usize)
        );
        let t = tile(d, c.0, c.1);
        assert_eq!(d.read32(w, t, r.received as u64).unwrap(), count);
    }
    let t = tile(d, PRODUCER.0, PRODUCER.1);
    assert_eq!(
        d.read32(w, t, r.freed as u64).unwrap(),
        count * r.consumers.len() as u32
    );
    stop(d, w, rig);
}

#[test]
fn a_remote_ring_hands_every_page_over_in_order() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        hand_over(d, &w, &ring(vec![CONSUMER], None));
    });
}

#[test]
fn a_broadcast_ring_hands_every_page_to_every_consumer() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        hand_over(d, &w, &ring(RECT.to_vec(), None));
    });
}

#[test]
fn a_gddr_ring_hands_every_page_over_through_gddr() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        hand_over(d, &w, &ring(vec![CONSUMER], Some(1)));
    });
}

/// The consumer waits for one page more than the producer sends.
#[test]
fn one_credit_too_few_stalls_and_never_races() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let r = ring(vec![CONSUMER], None);
        let (count, page_bytes) = pages();
        let sent = count - 1;
        let mut rig = rig(d, &w, &r, sent);
        let consumer = consumer_list(&r, count);
        let held = enqueue_all(d, &w, &mut rig.consumers[0], &consumer);
        let last = enqueue_all(d, &w, &mut rig.producer, &producer_list(&r, sent));
        rig.producer.wait_for(d, &w, last).unwrap();
        // Long enough for the consumer to take every page sent and reach its last wait.
        let t = tile(d, CONSUMER.0, CONSUMER.1);
        let want = r.receive(sent);
        let mut stall = None;
        for _ in 0..2000 {
            stall = channel::stalled(d, &w, t, Mover::B, want).unwrap();
            if stall.is_some() {
                break;
            }
            d.tick(10_000);
            if ON_SILICON {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        assert_eq!(
            stall,
            Some(Stall {
                word: r.received,
                have: sent,
                want: sent + 1
            })
        );
        // Held, not raced: still held a while later, the list not done, the page never copied.
        d.tick(200_000);
        assert!(channel::stalled(d, &w, t, Mover::B, want)
            .unwrap()
            .is_some());
        let done = rig.consumers[0].refresh(d, &w).unwrap();
        assert!(
            (done.wrapping_sub(held) as i32) < 0,
            "the consumer finished"
        );
        let bytes = (sent * page_bytes) as usize;
        let got = read(d, &w, CONSUMER, OUT as u64, bytes + page_bytes as usize);
        let want: Vec<u8> = (0..sent)
            .flat_map(|k| page(&rig.source, k, page_bytes).to_vec())
            .collect();
        assert_eq!(got[..bytes], want[..], "the pages sent");
        assert!(
            got[bytes..].iter().all(|&b| b == POISON),
            "the page never sent was copied"
        );
        // The abort cancels the wait; the list fails with its code.
        channel::cancel(d, &w, t, dm::error::PEER).unwrap();
        match rig.consumers[0].wait_for(d, &w, held) {
            Err(DmError::Queued { code, .. }) => assert_eq!(code, dm::error::PEER),
            other => panic!("the cancelled wait gave {other:?}"),
        }
        channel::prepare(d, &w, t, &[]).unwrap();
        stop(d, &w, rig);
    });
}

#[test]
fn a_malformed_credit_is_refused() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let r = ring(vec![CONSUMER], None);
        let mut rig = rig(d, &w, &r, 1);
        let (x, y) = (CONSUMER.0 as u32, CONSUMER.1 as u32);
        // The host: a credit to a tile its session does not name, or outside its window.
        let not_ours = [op::CREDIT, x + 3, y, received(), 1, 0, 0, 0];
        let outside = [op::CREDIT, x, y, window().start as u32 - 16, 1, 0, 0, 0];
        for e in [not_ours, outside] {
            match rig.producer.enqueue(d, &w, &[e]) {
                Err(DmError::Invalid(code)) => assert_eq!(code, dm::error::RANGE, "{e:x?}"),
                other => panic!("the host took {e:x?}: {other:?}"),
            }
        }
        // The tile, sent raw: each refused with its code, the word untouched. A fresh producer
        // each time: a refused list stops the mover it was sent to.
        let Rig {
            producer,
            others,
            consumers,
            ..
        } = rig;
        producer.stop(d, &w).unwrap();
        let raw: [(&str, [u32; 8], u32); 5] = [
            (
                "a credit of nothing",
                [op::CREDIT, x, y, received(), 0, 0, 0, 0],
                dm::error::LENGTH,
            ),
            (
                "a credit into a mover's mailbox",
                [op::CREDIT, x, y, dm::BARRIER_COUNTER as u32, 1, 0, 0, 0],
                dm::error::ALIGNMENT,
            ),
            (
                "a misaligned credit word",
                [op::CREDIT, x, y, received() + 2, 1, 0, 0, 0],
                dm::error::ALIGNMENT,
            ),
            (
                "the non-memory column",
                [op::CREDIT, 8, y, received(), 1, 0, 0, 0],
                dm::error::RANGE,
            ),
            (
                "a wait on the streams' counters",
                [
                    op::WAIT_CREDIT,
                    tt_isa::dataflow::INPUT as u32,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dm::error::ALIGNMENT,
            ),
        ];
        for (what, e, code) in raw {
            let mut m = start(d, &w, PRODUCER, Mover::NC);
            let n = m.enqueue_raw(d, &w, &[e]).unwrap();
            match m.wait_for(d, &w, n) {
                Err(DmError::Queued { code: got, .. }) => assert_eq!(got, code, "{what}"),
                other => panic!("the tile took {what}: {other:?}"),
            }
            m.stop(d, &w).unwrap();
        }
        let t = tile(d, CONSUMER.0, CONSUMER.1);
        assert_eq!(d.read32(&w, t, received() as u64).unwrap(), 0);
        for m in consumers.into_iter().chain(others) {
            m.stop(d, &w).unwrap();
        }
    });
}
