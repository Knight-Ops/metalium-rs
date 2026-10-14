//! `record::POKE_RUN`: a run of consecutive role-mailbox words in one record, which the
//! mover writes with one fence instead of [`op::POKE`]'s one a word.
//!
//! The oracle is the entry it replaces: the same words written by one `POKE` each must
//! leave the three mailboxes bit for bit as the records do, so the firmware's own loop
//! (`mover::poke_run`) is held to `record::expand_reads`, which is also what the host
//! checks a list with. The controls are the refusals: a count the record cannot carry, a
//! value past the count, an address outside a role mailbox, a run that starts inside one
//! and ends past it, and NC, which runs no poke at all.

use tt_device::tlb::WindowKind;
use tt_isa::dm::{self, op, record, Mover};
use tt_isa::mailbox::role::Mailbox;
use tt_isa::noc::Noc0;
use tt_kernels::dm::{DataMover, DmError};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile, Dev};

/// The words of role `thread`'s mailbox a descriptor covers, as the trace pokes them.
fn words_of(thread: u32, seed: u32) -> (u32, Vec<u32>) {
    let base = Mailbox::of(thread).base() as u32 + 0x1C;
    (base, (0..7).map(|k| 0xC0DE_0000 + seed * 16 + k).collect())
}

fn start(d: &mut Dev<'_>, mover: Mover) -> DataMover<Noc0> {
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let dram = d.dram_grid(&w).unwrap();
    let t = tile(d, GATE_TILE.0, GATE_TILE.1);
    let image = match mover.core {
        tt_isa::tensix::Core::NC => tt_firmware_images::DM_NC.1,
        _ => tt_firmware_images::DM_B.1,
    };
    DataMover::start_on(d, &w, t, &dram, mover, image).unwrap()
}

/// A record writes the three mailboxes exactly as the `POKE`s it expands to do.
#[test]
fn a_run_writes_what_the_pokes_it_expands_to_write() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let mut m = start(d, Mover::B);
        let t = m.tile();
        // Every length one record carries, on each role, at a few offsets into its
        // mailbox: what the host's runs are (2, 4 and 11 words) and the extremes.
        let mut cases = Vec::new();
        for thread in 0..3 {
            for count in [1usize, 2, 4, 11, record::POKE_RUN_MAX] {
                let base = Mailbox::of(thread).base() as u32 + 0x1C;
                let values: Vec<u32> = (0..count as u32)
                    .map(|k| 0x5EED_0000 + thread * 0x100 + k)
                    .collect();
                cases.push((base, values));
            }
        }
        for (at, values) in &cases {
            // Poison, so a word the mover does not write is not last run's value.
            let poison: Vec<u8> = (0..values.len() * 4).map(|k| 0xA5 ^ k as u8).collect();
            d.l1_write(&w, t, *at as u64, &poison).unwrap();
            // The entries the record stands for, run one at a time.
            let pokes: Vec<[u32; 8]> = values
                .iter()
                .enumerate()
                .map(|(k, &v)| [op::POKE, at + 4 * k as u32, v, 0, 0, 0, 0, 0])
                .collect();
            let n = m.enqueue(d, &w, &pokes).unwrap();
            m.wait_for(d, &w, n).unwrap();
            let mut by_pokes = vec![0u8; values.len() * 4];
            d.l1_read(&w, t, *at as u64, &mut by_pokes).unwrap();

            d.l1_write(&w, t, *at as u64, &poison).unwrap();
            if values.len() == 1 {
                // A one-word run is a `POKE`: the record would be the larger of the two,
                // and `poke_run` is not built for it.
                continue;
            }
            let rec = record::poke_run(*at, values);
            let n = m.enqueue(d, &w, &rec).unwrap();
            m.wait_for(d, &w, n).unwrap();
            let mut by_record = vec![0u8; values.len() * 4];
            d.l1_read(&w, t, *at as u64, &mut by_record).unwrap();

            let want: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(by_pokes, want, "the pokes at {at:#x}");
            assert_eq!(by_record, want, "the record at {at:#x}");
        }
        // Nothing past the run: the word after the last is the poison it was given.
        let (at, values) = words_of(1, 3);
        let after = at + 4 * values.len() as u32;
        d.write32(&w, t, after as u64, 0xDEAD_BEEF).unwrap();
        let n = m.enqueue(d, &w, &record::poke_run(at, &values)).unwrap();
        m.wait_for(d, &w, n).unwrap();
        assert_eq!(d.read32(&w, t, after as u64).unwrap(), 0xDEAD_BEEF);
        m.stop(d, &w).unwrap();
    });
}

/// What the host refuses before the card sees it, and what the card refuses if the
/// host's check is bypassed (`enqueue_raw`): the two agree, and nothing is written.
#[test]
fn a_malformed_run_is_refused_by_the_host_and_by_the_tile() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let (at, values) = words_of(0, 1);
        let last = (Mailbox::of(2).base() + tt_isa::mailbox::MAILBOX_SIZE - 4) as u32;
        let mut stray = record::poke_run(at, &values);
        stray[1][7] = 1;
        let mut count0 = record::poke_run(at, &values);
        count0[0][2] = 0;
        let mut count_big = record::poke_run(at, &values);
        count_big[0][2] = record::POKE_RUN_MAX as u32 + 1;
        let bad: [(&str, [[u32; 8]; 2], u32); 6] = [
            ("a value past the count", stray, dm::error::OP),
            ("a count of none", count0, dm::error::RANGE),
            ("a count past the record", count_big, dm::error::RANGE),
            (
                "the mover's own L1",
                record::poke_run(0x2_0000, &[1, 2]),
                dm::error::OP,
            ),
            (
                "past the last mailbox word",
                record::poke_run(last + 4, &[1, 2]),
                dm::error::OP,
            ),
            (
                "a run that ends past its mailbox",
                record::poke_run(last, &[1, 2]),
                dm::error::OP,
            ),
        ];
        for (what, rec, code) in bad {
            // A fresh mover each time: a refused list stops the one it was sent to.
            let mut m = start(d, Mover::B);
            let t = m.tile();
            let r = m.enqueue(d, &w, &rec);
            assert!(
                matches!(r, Err(DmError::Invalid(c)) if c == code),
                "the host took {what}: {r:?}"
            );
            // Past the host's check, with the words it names poisoned: the tile refuses
            // it with the same code and writes none of them.
            let poison: Vec<u8> = (0..4 * record::POKE_RUN_MAX)
                .map(|k| 0xA5 ^ k as u8)
                .collect();
            let poisoned = rec[0][1] as u64;
            let inside = dm::is_role_mailbox_word(rec[0][1]);
            if inside {
                d.l1_write(&w, t, poisoned, &poison).unwrap();
            }
            let n = m.enqueue_raw(d, &w, &rec).unwrap();
            let got = m.wait_for(d, &w, n);
            assert!(
                matches!(got, Err(DmError::Queued { code: c, .. }) if c == code),
                "the tile took {what}: {got:?}"
            );
            if inside {
                let mut back = vec![0u8; poison.len()];
                d.l1_read(&w, t, poisoned, &mut back).unwrap();
                assert_eq!(back, poison, "the tile wrote something for {what}");
            }
            m.stop(d, &w).unwrap();
        }
    });
}

/// NC is the writer: it runs no poke, as a record no more than as an entry.
#[test]
fn the_writer_refuses_a_run() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let mut m = start(d, Mover::NC);
        let (at, values) = words_of(0, 5);
        let rec = record::poke_run(at, &values);
        assert!(!Mover::NC.permits(record::POKE_RUN));
        let e = m.enqueue(d, &w, &rec).unwrap_err();
        assert!(matches!(e, DmError::Invalid(dm::error::DIRECTION)), "{e}");
        let n = m.enqueue_raw(d, &w, &rec).unwrap();
        let got = m.wait_for(d, &w, n);
        assert!(
            matches!(got, Err(DmError::Queued { code, .. }) if code == dm::error::DIRECTION),
            "{got:?}"
        );
        m.stop(d, &w).unwrap();
    });
}
