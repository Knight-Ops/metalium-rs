//! Unit tests for the fenced writes, against the windowing fake transport
//! (`device::tests::FakeTransport`), which logs every data access in order.

use super::super::tests::{c, device};
use super::*;
use crate::tlb::WindowKind;

fn tile() -> NocCoord<tt_isa::noc::Noc0> {
    c(3, 4)
}

/// `(is_write, tile address, len)` of every data access so far.
type Log = Vec<(bool, u64, usize)>;

fn reads(log: &Log) -> Log {
    log.iter().copied().filter(|e| !e.0).collect()
}

#[test]
fn write32_fenced_is_one_write_then_one_read_of_the_same_dword() {
    let mut d = device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    d.write32_fenced(&w, tile(), 0x1004, 0xDEAD_BEEF).unwrap();
    assert_eq!(
        d.transport().log,
        vec![(true, 0x1004, 4), (false, 0x1004, 4)],
        "the write, then exactly one read of the word just written"
    );
    assert_eq!(d.read32(&w, tile(), 0x1004).unwrap(), 0xDEAD_BEEF);
}

#[test]
fn write_fenced_reads_back_the_last_dword_after_the_last_write() {
    let mut d = device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let data: Vec<u8> = (0..16).collect();
    d.write_fenced(&w, tile(), 0x2000, &data).unwrap();
    assert_eq!(
        d.transport().log,
        vec![(true, 0x2000, 16), (false, 0x200C, 4)]
    );
    let mut back = [0u8; 16];
    d.read(&w, tile(), 0x2000, &mut back).unwrap();
    assert_eq!(&back[..], &data[..]);
}

#[test]
fn a_transfer_split_across_windows_still_reads_back_once() {
    let mut d = device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let size = WindowKind::TwoMib.size();
    d.write_fenced(&w, tile(), size - 8, &[0x5A; 16]).unwrap();
    assert_eq!(
        d.transport().log,
        vec![(true, size - 8, 8), (true, size, 8), (false, size + 4, 4)],
        "two write pieces, one read of the last dword"
    );
}

#[test]
fn l1_write_fenced_takes_ragged_ranges_and_reads_the_dword_with_the_last_byte() {
    let mut d = device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    // Bytes 0x3002..0x3008: the last byte is in the dword at 0x3004.
    d.l1_write_fenced(&w, tile(), 0x3002, &[1, 2, 3, 4, 5, 6])
        .unwrap();
    assert_eq!(
        d.transport().log,
        vec![(true, 0x3002, 6), (false, 0x3004, 4)]
    );
    let mut back = [0u8; 6];
    d.l1_read(&w, tile(), 0x3002, &mut back).unwrap();
    assert_eq!(back, [1, 2, 3, 4, 5, 6]);
}

#[test]
fn the_bulk_fence_refuses_what_the_bulk_path_refuses_without_traffic() {
    let mut d = device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let before = d.traffic();
    // The ARC tile, a Tensix register, and a range crossing the end of L1.
    let end = tt_isa::tensix::L1_SIZE - 2;
    for (coord, addr) in [(c(8, 0), 0x1000), (tile(), 0xFFB1_21B0), (tile(), end)] {
        let e = d.l1_write_fenced(&w, coord, addr, &[0; 4]).unwrap_err();
        assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
    }
    assert_eq!(d.traffic(), before, "nothing may be sent");
    assert!(d.transport().log.is_empty());
}

#[test]
fn misaligned_fenced_writes_are_refused_before_any_traffic() {
    let mut d = device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let before = d.traffic();
    // Unaligned address, a length that is not whole dwords, a ragged fence
    // word, and the second write of a batch (the first is fine).
    let results: Vec<Result<()>> = vec![
        d.write32_fenced(&w, tile(), 0x1002, 1),
        d.write_fenced(&w, tile(), 0x1000, &[1, 2, 3, 4, 5, 6]),
        d.write_fenced(&w, tile(), 0x1001, &[1, 2, 3, 4]),
        FencedWrite::new(&w, tile())
            .word(0x1000, 1)
            .fence_at(0x1002)
            .commit(&mut d),
        FencedWrite::new(&w, tile())
            .word(0x1000, 1)
            .bytes(0x1010, &[9; 3])
            .commit(&mut d),
    ];
    for (i, r) in results.into_iter().enumerate() {
        let e = r.unwrap_err();
        assert!(
            matches!(e, TransportError::Misaligned { .. }),
            "case {i}: {e}"
        );
    }
    assert_eq!(d.traffic(), before, "a refused batch changes nothing");
    assert!(d.transport().log.is_empty());
}

#[test]
fn the_local_ram_aperture_is_refused_before_any_traffic() {
    let mut d = device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let before = d.traffic();
    let e = d.write32_fenced(&w, tile(), 0xFFB1_4000, 1).unwrap_err();
    assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
    let e = FencedWrite::new(&w, tile())
        .word(0x1000, 1)
        .fence_at(0xFFB1_4000)
        .commit(&mut d)
        .unwrap_err();
    assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
    assert_eq!(d.traffic(), before);
}

#[test]
fn a_failed_write_reads_nothing_back() {
    let mut d = device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    d.transport().fail_data_writes = true;
    let before = d.traffic();
    assert!(d.write_fenced(&w, tile(), 0x1000, &[0; 8]).is_err());
    assert!(d.l1_write_fenced(&w, tile(), 0x1000, &[0; 8]).is_err());
    assert!(d.write_range_fenced(&w, tile(), 0x1001, &[0; 3]).is_err());
    assert_eq!(d.traffic().read_calls, before.read_calls, "no read-back");
    assert!(reads(&d.transport().log).is_empty());
}

#[test]
fn a_batch_is_ordered_and_fenced_once_at_the_last_word_or_the_named_one() {
    let mut d = device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    FencedWrite::new(&w, tile())
        .word(0x4000, 1)
        .bytes(0x4010, &[7; 8])
        .word(0x4004, 2)
        .commit(&mut d)
        .unwrap();
    assert_eq!(
        d.transport().log,
        vec![
            (true, 0x4000, 4),
            (true, 0x4010, 8),
            (true, 0x4004, 4),
            (false, 0x4004, 4)
        ],
        "writes in call order, then one read of the last one's word"
    );
    d.transport().log.clear();
    // A doorbell that cannot be read: fence on a readable word instead.
    FencedWrite::new(&w, tile())
        .word(0x4000, 3)
        .word(0x4100, 1)
        .fence_at(0x4000)
        .commit(&mut d)
        .unwrap();
    assert_eq!(
        d.transport().log,
        vec![(true, 0x4000, 4), (true, 0x4100, 4), (false, 0x4000, 4)]
    );
}

#[test]
fn an_empty_batch_is_no_traffic() {
    let mut d = device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let before = d.traffic();
    FencedWrite::new(&w, tile()).commit(&mut d).unwrap();
    d.write_fenced(&w, tile(), 0x1000, &[]).unwrap();
    d.l1_write_fenced(&w, tile(), 0x1001, &[]).unwrap();
    assert_eq!(d.traffic(), before);
}

#[test]
fn a_window_from_another_device_is_refused_before_any_write() {
    let (mut a, mut b) = (device(), device());
    let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
    let _wb = b.alloc_window(WindowKind::TwoMib).unwrap();
    let before = b.traffic();
    let e = b.write32_fenced(&wa, tile(), 0x1000, 1).unwrap_err();
    assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
    assert_eq!(b.traffic(), before);
    assert!(b.transport().log.is_empty());
}
