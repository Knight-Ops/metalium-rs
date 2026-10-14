//! Differential gate for B's read modes: a list of reads lands exactly what a host `memcpy`
//! from a mirror of the DRAM says, in the default (safe) build, and in a `fast-reads` build with
//! the host sending plain entries and with it sending host-encoded ones.
//!
//! `step172` holds the encoded entry's own properties (what is refused, and by whom). This holds
//! the property that matters to everything above it: whichever way a read travels, the bytes in L1
//! are the bytes of the oracle. Seeded random lists of reads of mixed sizes (one request, and
//! several), on every channel and port, with `WAIT`s between, into disjoint destinations; the
//! window is poisoned before every list and compared whole afterwards.
//!
//! **Negative control** (fast builds): an encoded entry whose destination is moved by 64 bytes, its
//! checksum made good, is another valid read: nothing on the host (`validate`) or the card (checksum,
//! guard) can call it wrong, because only the oracle knows where it was meant to land. It lands the
//! bytes 64 late, and the comparison used above must report that as a mismatch, or it would not
//! catch a mover that reads or writes the wrong place.
//!
//! `cargo test -p tt-tests --test step173_mover_differential`, and with `--features
//! tt-tests/fast-reads` and `tt-tests/encoded-guard`; on silicon through `cargo xtask silicon`.

use tt_device::tlb::WindowKind;
use tt_device::Window;
use tt_isa::dm::{op, Mover};
use tt_isa::dram::PORTS;
use tt_isa::noc::Noc0;
use tt_kernels::dm::DataMover;
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile, Dev};

const L1_AT: u32 = 0x2_0000;
/// The destination window in L1: 192 KiB, the slots of one list.
const WINDOW: usize = 192 << 10;
const POISON: u8 = 0xA5;
/// Where the source data lies in every channel, and how much of each is used.
const DRAM_AT: u64 = (300 << 20) + 0x100_0000;
const REGION: usize = 512 << 10;
const CHANNELS: usize = 8;
/// Request sizes: one request each up to 16 KiB, then two and a bit.
const LENS: [u32; 9] = [
    64,
    1024,
    2048,
    4096,
    4160,
    8192,
    16384,
    16384 + 64,
    16384 + 4096,
];
const LISTS: u64 = 24;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn pattern(len: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    (0..len).map(|_| r.next() as u8).collect()
}

fn start(d: &mut Dev<'_>, w: &Window) -> DataMover<Noc0> {
    let dram = d.dram_grid(w).unwrap();
    let t = tile(d, GATE_TILE.0, GATE_TILE.1);
    DataMover::start_on(d, w, t, &dram, Mover::B, tt_firmware_images::DM_B.1).unwrap()
}

/// The DRAM the lists read, channel by channel, written to the card and kept as the oracle's mirror.
fn fill_dram(d: &mut Dev<'_>, w: &Window, w4: &Window) -> Vec<Vec<u8>> {
    let dram = d.dram_grid(w).unwrap();
    let mirror: Vec<Vec<u8>> = (0..CHANNELS)
        .map(|c| pattern(REGION, 0xD1FF + c as u64))
        .collect();
    for (c, ch) in dram.channels().enumerate() {
        d.dram_write(w4, ch.range(DRAM_AT, REGION as u64).unwrap(), &mirror[c])
            .unwrap();
    }
    mirror
}

/// A random list and the L1 window it must leave: reads of mixed sizes into disjoint slots,
/// `WAIT`s between some of them.
fn random_list(r: &mut Rng, mirror: &[Vec<u8>]) -> (Vec<[u32; 8]>, Vec<u8>) {
    let mut list = Vec::new();
    let mut want = vec![POISON; WINDOW];
    let mut at = 0usize;
    while list.len() < 24 {
        if r.below(100) < 15 {
            list.push([op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
            continue;
        }
        let len = LENS[r.below(LENS.len() as u64) as usize] as usize;
        if at + len > WINDOW {
            break;
        }
        let (c, port) = (r.below(CHANNELS as u64) as usize, r.below(PORTS as u64));
        let off = 64 * r.below(((REGION - len) / 64 + 1) as u64) as usize;
        list.push([
            op::READ,
            c as u32,
            port as u32,
            (DRAM_AT + off as u64) as u32,
            L1_AT + at as u32,
            len as u32,
            0,
            0,
        ]);
        want[at..at + len].copy_from_slice(&mirror[c][off..off + len]);
        at += len.next_multiple_of(64);
    }
    (list, want)
}

/// The first byte at which `got` and `want` differ.
fn first_difference(got: &[u8], want: &[u8]) -> Option<usize> {
    got.iter().zip(want).position(|(a, b)| a != b)
}

/// Run `list` on `m` from a poisoned window and return the window.
fn run(d: &mut Dev<'_>, w: &Window, m: &mut DataMover<Noc0>, list: &[[u32; 8]]) -> Vec<u8> {
    let t = m.tile();
    d.l1_write(w, t, L1_AT as u64, &vec![POISON; WINDOW])
        .unwrap();
    m.run_list(d, w, list).unwrap();
    let mut back = vec![0u8; WINDOW];
    d.l1_read(w, t, L1_AT as u64, &mut back).unwrap();
    back
}

#[test]
fn every_way_a_read_travels_lands_the_oracles_bytes() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let mirror = fill_dram(d, &w, &w4);
        let mut m = start(d, &w);
        // The modes this build has: a safe one has only plain entries.
        let modes: &[bool] = if m.fast_reads() {
            &[false, true]
        } else {
            &[false]
        };
        let mut reads = 0;
        for &encode in modes {
            m.set_encode_reads(encode).unwrap();
            let mut r = Rng(0x173);
            for i in 0..LISTS {
                let (list, want) = random_list(&mut r, &mirror);
                let got = run(d, &w, &mut m, &list);
                assert!(
                    first_difference(&got, &want).is_none(),
                    "list {i}, encoded reads {encode}: first difference at byte {:?}",
                    first_difference(&got, &want)
                );
                reads += list.iter().filter(|e| e[0] == op::READ).count();
            }
        }
        assert!(reads > 200 * modes.len(), "{reads} reads");
        m.stop(d, &w).unwrap();
    });
}

/// The comparison has teeth: see the module docs.
#[cfg(feature = "fast-reads")]
#[test]
fn a_consistent_but_wrong_entry_is_a_mismatch() {
    use tt_isa::dm::encoded;
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let mirror = fill_dram(d, &w, &w4);
        let mut m = start(d, &w);
        let t = m.tile();
        let plain = [op::READ, 3, 0, DRAM_AT as u32 + 0x1000, L1_AT, 4096, 0, 0];
        let want = {
            let mut v = vec![POISON; WINDOW];
            v[..4096].copy_from_slice(&mirror[3][0x1000..0x2000]);
            v
        };
        // The control: the entry as the host makes it lands the oracle's bytes.
        assert!(first_difference(&run(d, &w, &mut m, &[plain]), &want).is_none());
        let good = encoded::encode(m.encode_context(), plain).unwrap().unwrap();
        // The same, destination moved by 64 bytes, checksum made good. It is another valid read
        // (a congruent destination inside L1), so neither `validate` nor the guard can call it
        // wrong: only the oracle knows where it was meant to land. The queue's trusting entry
        // point takes it, and the card runs it.
        let mut wrong = good;
        wrong[3] += 64;
        wrong[encoded::CHECK_WORD] =
            encoded::check_with(&wrong, m.encode_context().seed(op::READ_ENCODED));
        d.l1_write(&w, t, L1_AT as u64, &vec![POISON; WINDOW])
            .unwrap();
        let n = m.enqueue_checked(d, &w, &[wrong]).unwrap();
        m.wait_for(d, &w, n).unwrap();
        let mut back = vec![0u8; WINDOW];
        d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
        assert!(
            first_difference(&back, &want).is_some(),
            "a read 64 bytes late compared equal to the oracle"
        );
        assert_eq!(
            &back[64..64 + 4096],
            &want[..4096],
            "it landed whole, 64 late"
        );
        m.stop(d, &w).unwrap();
    });
}
