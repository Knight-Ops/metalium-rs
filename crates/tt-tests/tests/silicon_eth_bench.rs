//! Throwaway: where does the Ethernet path's time go?
//!
//! `cargo xtask silicon --include-ignored --filter silicon_eth_bench`

#![cfg(feature = "silicon")]

use std::time::{Duration, Instant};

use tt_device::tlb::WindowKind;
use tt_isa::eth;
use tt_kernels::link::{discover, Dest, Dir, Mover, Source};
use tt_tests::bench::{mbps, median, pattern, with_cards};
use tt_tests::harness::{tile, Dev};

const SIZES: [usize; 5] = [4 << 10, 16 << 10, 32 << 10, 64 << 10, 128 << 10];
const REPS: usize = 20;

/// Host-driven TT-link, Ethernet L1 to Ethernet L1, no firmware: the link and
/// the TX queue alone, plus one PCIe poll loop on the receiver.
#[test]
#[ignore = "benchmark"]
fn raw_tt_link() {
    with_cards(|a, b| {
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let (ga, gb) = (a.ethernet_grid(&wa).unwrap(), b.ethernet_grid(&wb).unwrap());
        let l = discover(a, &wa, &ga, b, &wb, &gb).unwrap()[0];
        let (src, dst) = (eth::BUFFERS.start, eth::BUFFERS.start + 0x2_0000);
        let q = eth::txq_base(eth::DATA_QUEUE);
        let c = l.a.coord();
        for cmd_bytes in [4usize << 10, 16 << 10, 64 << 10, 128 << 10] {
            for &size in SIZES.iter().filter(|&&s| s >= cmd_bytes) {
                let mut times = Vec::new();
                for rep in 0..REPS {
                    let data = pattern(size, rep as u32 + 1);
                    a.eth_write(&wa, l.a, src, &data).unwrap();
                    b.eth_write(&wb, l.b, dst, &vec![0u8; size]).unwrap();
                    if std::env::var_os("NO_FENCE").is_none() {
                        // Posted writes: read the last word back so the zero-fill has
                        // landed before the link starts writing the same bytes.
                        let mut w = [0u8; 4];
                        b.eth_read(&wb, l.b, dst + size as u64 - 4, &mut w).unwrap();
                    }
                    let tail = &data[size - 16..];
                    let t0 = Instant::now();
                    for off in (0..size).step_by(cmd_bytes) {
                        while a.read32(&wa, c, q + eth::txq::STATUS).unwrap()
                            & eth::txq::STATUS_CMD_ONGOING
                            != 0
                        {}
                        a.write32(
                            &wa,
                            c,
                            q + eth::txq::TRANSFER_START_ADDR,
                            (src + off as u64) as u32,
                        )
                        .unwrap();
                        a.write32(&wa, c, q + eth::txq::TRANSFER_SIZE_BYTES, cmd_bytes as u32)
                            .unwrap();
                        a.write32(&wa, c, q + eth::txq::DEST_ADDR, (dst + off as u64) as u32)
                            .unwrap();
                        a.write32(&wa, c, q + eth::txq::CMD, 2).unwrap();
                        let _ = a.read32(&wa, c, q + eth::txq::CMD).unwrap();
                    }
                    let mut got = [0u8; 16];
                    loop {
                        b.eth_read(&wb, l.b, dst + size as u64 - 16, &mut got)
                            .unwrap();
                        if got[..] == tail[..] {
                            break;
                        }
                        if t0.elapsed() > Duration::from_secs(1) {
                            let mut all = vec![0u8; size];
                            b.eth_read(&wb, l.b, dst, &mut all).unwrap();
                            let bad: Vec<usize> = (0..size / 4096)
                                .filter(|&i| all[i * 4096..][..4096] != data[i * 4096..][..4096])
                                .collect();
                            let first = (0..size).find(|&i| all[i] != data[i]);
                            let rq = eth::rxq_base(eth::DATA_QUEUE);
                            let cb = l.b.coord();
                            let rx: Vec<u32> = [0x04u64, 0x08, 0x28, 0x4C, 0x50]
                                .iter()
                                .map(|o| b.read32(&wb, cb, rq + o).unwrap())
                                .collect();
                            println!("MEASURE stuck: rx bytes/buf_ptr/pkt_end/drops/outstanding = {rx:x?}");
                            // A flush: one more small packet, to an unrelated address.
                            a.write32(&wa, c, q + eth::txq::TRANSFER_START_ADDR, src as u32)
                                .unwrap();
                            a.write32(&wa, c, q + eth::txq::TRANSFER_SIZE_BYTES, 16)
                                .unwrap();
                            a.write32(&wa, c, q + eth::txq::DEST_ADDR, (dst + 0x2_0000) as u32)
                                .unwrap();
                            a.write32(&wa, c, q + eth::txq::CMD, 2).unwrap();
                            std::thread::sleep(Duration::from_millis(10));
                            b.eth_read(&wb, l.b, dst + size as u64 - 16, &mut got)
                                .unwrap();
                            let rx: Vec<u32> = [0x04u64, 0x08, 0x28, 0x4C, 0x50]
                                .iter()
                                .map(|o| b.read32(&wb, cb, rq + o).unwrap())
                                .collect();
                            println!(
                                "MEASURE after a 16-byte flush: tail arrived = {}, rx = {rx:x?}",
                                got[..] == tail[..]
                            );
                            let tx: Vec<u32> = [0x08u64, 0x30, 0x34, 0x3C, 0x48, 0x4C]
                                .iter()
                                .map(|o| a.read32(&wa, c, q + o).unwrap())
                                .collect();
                            println!("MEASURE tx status/xfer/pkt_start/pkt_end/resend_timeout/seq_update_timeout = {tx:x?}");
                            for wait in [100u64, 1000, 3000] {
                                std::thread::sleep(Duration::from_millis(wait));
                                b.eth_read(&wb, l.b, dst + size as u64 - 16, &mut got)
                                    .unwrap();
                                println!(
                                    "MEASURE +{wait} ms: tail arrived = {}",
                                    got[..] == tail[..]
                                );
                            }
                            panic!("rep {rep} cmd {cmd_bytes} size {size}: chunks wrong {bad:?}, first bad byte {first:?}, tail {got:02x?} want {tail:02x?}");
                        }
                    }
                    times.push(t0.elapsed());
                    let mut all = vec![0u8; size];
                    b.eth_read(&wb, l.b, dst, &mut all).unwrap();
                    assert!(all == data, "payload differs");
                }
                let t = median(times);
                println!(
                    "MEASURE raw  cmd {:>3} KiB size {:>3} KiB: {:>8.1?} {:>7.0} MB/s",
                    cmd_bytes >> 10,
                    size >> 10,
                    t,
                    mbps(size, t)
                );
            }
        }
    });
}

/// The E1 mover: staged -> landed (checksums and handshake, no NoC hops), then
/// Tensix -> Tensix (plus a NoC hop at each end).
#[test]
#[ignore = "benchmark"]
fn mover() {
    with_cards(|a, b| {
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let (ga, gb) = (a.ethernet_grid(&wa).unwrap(), b.ethernet_grid(&wb).unwrap());
        let l = discover(a, &wa, &ga, b, &wb, &gb).unwrap()[0];
        let (ta, tb) = (tile(a, 3, 4), tile(b, 4, 4));
        let mut m = Mover::start(a, &wa, b, &wb, l, tt_firmware_images::ETH_E1).unwrap();
        for &size in &SIZES {
            let data = pattern(size, 9);
            m.stage(a, &wa, Dir::AToB, &data).unwrap();
            let mut staged = Vec::new();
            for _ in 0..REPS {
                let t0 = Instant::now();
                m.send(a, &wa, Dir::AToB, Source::Staged, Dest::Landed, size as u32)
                    .unwrap();
                staged.push(t0.elapsed());
            }
            a.write(&wa, ta, 0x8_0000, &data).unwrap();
            let _ = a.read32(&wa, ta, 0x8_0000 + size as u64 - 4).unwrap();
            let mut t2t = Vec::new();
            for _ in 0..REPS {
                let t0 = Instant::now();
                m.send(
                    a,
                    &wa,
                    Dir::AToB,
                    Source::Tensix(ta, 0x8_0000),
                    Dest::Tensix(tb, 0x8_0000),
                    size as u32,
                )
                .unwrap();
                t2t.push(t0.elapsed());
            }
            let mut got = vec![0u8; size];
            b.read(&wb, tb, 0x8_0000, &mut got).unwrap();
            assert!(got == data);
            let (s, t) = (median(staged), median(t2t));
            println!("MEASURE mover size {:>3} KiB: staged {:>8.1?} {:>6.0} MB/s | tensix->tensix {:>8.1?} {:>6.0} MB/s",
                size >> 10, s, mbps(size, s), t, mbps(size, t));
        }
        a.park_e1(&wa, l.a).unwrap();
        b.park_e1(&wb, l.b).unwrap();
    });
}

/// Baseline: the host writing and reading chip 1's Tensix L1 over PCIe.
#[test]
#[ignore = "benchmark"]
fn host_pcie() {
    with_cards(|_, b| {
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let tb = tile(b, 4, 4);
        for &size in &SIZES {
            let data = pattern(size, 3);
            let mut buf = vec![0u8; size];
            let (mut w, mut r) = (Vec::new(), Vec::new());
            for _ in 0..REPS {
                let t0 = Instant::now();
                b.write(&wb, tb, 0x8_0000, &data).unwrap();
                // A read of the last word, so the posted writes have landed.
                let _ = b.read32(&wb, tb, 0x8_0000 + size as u64 - 4).unwrap();
                w.push(t0.elapsed());
                let t0 = Instant::now();
                b.read(&wb, tb, 0x8_0000, &mut buf).unwrap();
                r.push(t0.elapsed());
            }
            let (w, r) = (median(w), median(r));
            println!("MEASURE pcie size {:>3} KiB: write {:>8.1?} {:>6.0} MB/s | read {:>8.1?} {:>6.0} MB/s",
                size >> 10, w, mbps(size, w), r, mbps(size, r));
        }
    });
}

/// Why does a host-driven 128 KiB burst of 4 KiB commands not arrive? Sweep
/// the command count and dump the queue counters on both ends when it fails.
#[test]
#[ignore = "benchmark"]
fn raw_burst_threshold() {
    with_cards(|a, b| {
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let (ga, gb) = (a.ethernet_grid(&wa).unwrap(), b.ethernet_grid(&wb).unwrap());
        let l = discover(a, &wa, &ga, b, &wb, &gb).unwrap()[0];
        let (src, dst) = (eth::BUFFERS.start, eth::BUFFERS.start + 0x2_0000);
        let q = eth::txq_base(eth::DATA_QUEUE);
        let rq = eth::rxq_base(eth::DATA_QUEUE);
        let (ca, cb) = (l.a.coord(), l.b.coord());
        let dump = |a: &mut Dev<'_>, b: &mut Dev<'_>, tag: &str| {
            let tx: Vec<u32> = [0x08u64, 0x30, 0x34, 0x3C, 0x40]
                .iter()
                .map(|o| a.read32(&wa, ca, q + o).unwrap())
                .collect();
            let rx: Vec<u32> = [0x04u64, 0x24, 0x28, 0x40, 0x44, 0x4C, 0x50]
                .iter()
                .map(|o| b.read32(&wb, cb, rq + o).unwrap())
                .collect();
            println!("MEASURE {tag}: tx status/xfer/pkt_start/pkt_end/words={tx:x?} rx bytes/pkt_start/pkt_end/local_seq/remote_seq/drops/outstanding={rx:x?}");
        };
        dump(a, b, "before");
        for cmds in [16usize, 24, 28, 30, 31, 32, 33, 40] {
            let size = cmds * 4096;
            let data = pattern(size, cmds as u32);
            a.eth_write(&wa, l.a, src, &data).unwrap();
            b.eth_write(&wb, l.b, dst, &vec![0u8; size]).unwrap();
            if std::env::var_os("NO_FENCE").is_none() {
                // Posted writes: read the last word back so the zero-fill has
                // landed before the link starts writing the same bytes.
                let mut w = [0u8; 4];
                b.eth_read(&wb, l.b, dst + size as u64 - 4, &mut w).unwrap();
            }
            for i in 0..cmds {
                while a.read32(&wa, ca, q + eth::txq::STATUS).unwrap()
                    & eth::txq::STATUS_CMD_ONGOING
                    != 0
                {}
                a.write32(
                    &wa,
                    ca,
                    q + eth::txq::TRANSFER_START_ADDR,
                    (src + i as u64 * 4096) as u32,
                )
                .unwrap();
                a.write32(&wa, ca, q + eth::txq::TRANSFER_SIZE_BYTES, 4096)
                    .unwrap();
                a.write32(
                    &wa,
                    ca,
                    q + eth::txq::DEST_ADDR,
                    (dst + i as u64 * 4096) as u32,
                )
                .unwrap();
                a.write32(&wa, ca, q + eth::txq::CMD, 2).unwrap();
                let _ = a.read32(&wa, ca, q + eth::txq::CMD).unwrap();
            }
            std::thread::sleep(Duration::from_millis(50));
            let mut got = vec![0u8; size];
            b.eth_read(&wb, l.b, dst, &mut got).unwrap();
            let bad: Vec<usize> = (0..cmds)
                .filter(|&i| got[i * 4096..][..4096] != data[i * 4096..][..4096])
                .collect();
            println!(
                "MEASURE {cmds} cmds ({} KiB): {} chunks wrong {:?}",
                size >> 10,
                bad.len(),
                bad
            );
            dump(a, b, &format!("after {cmds}"));
        }
    });
}

/// The same bursts, but the receiver's L1 is not touched until the sender says
/// it is done: wait on the sender's `TRANSFER_CNT`, then a fixed settle, then read.
#[test]
#[ignore = "benchmark"]
fn raw_no_receiver_polling() {
    with_cards(|a, b| {
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let (ga, gb) = (a.ethernet_grid(&wa).unwrap(), b.ethernet_grid(&wb).unwrap());
        let l = discover(a, &wa, &ga, b, &wb, &gb).unwrap()[0];
        let (src, dst) = (eth::BUFFERS.start, eth::BUFFERS.start + 0x2_0000);
        let q = eth::txq_base(eth::DATA_QUEUE);
        let c = l.a.coord();
        let mut failures = 0;
        for cmd_bytes in [4usize << 10, 16 << 10, 128 << 10] {
            for rep in 0..40 {
                let size = 128 << 10;
                let data = pattern(size, rep as u32 + 100);
                a.eth_write(&wa, l.a, src, &data).unwrap();
                b.eth_write(&wb, l.b, dst, &vec![0u8; size]).unwrap();
                if std::env::var_os("NO_FENCE").is_none() {
                    // Posted writes: read the last word back so the zero-fill has
                    // landed before the link starts writing the same bytes.
                    let mut w = [0u8; 4];
                    b.eth_read(&wb, l.b, dst + size as u64 - 4, &mut w).unwrap();
                }
                for off in (0..size).step_by(cmd_bytes) {
                    while a.read32(&wa, c, q + eth::txq::STATUS).unwrap()
                        & eth::txq::STATUS_CMD_ONGOING
                        != 0
                    {}
                    a.write32(
                        &wa,
                        c,
                        q + eth::txq::TRANSFER_START_ADDR,
                        (src + off as u64) as u32,
                    )
                    .unwrap();
                    a.write32(&wa, c, q + eth::txq::TRANSFER_SIZE_BYTES, cmd_bytes as u32)
                        .unwrap();
                    a.write32(&wa, c, q + eth::txq::DEST_ADDR, (dst + off as u64) as u32)
                        .unwrap();
                    a.write32(&wa, c, q + eth::txq::CMD, 2).unwrap();
                    let _ = a.read32(&wa, c, q + eth::txq::CMD).unwrap();
                }
                std::thread::sleep(Duration::from_millis(5));
                let mut all = vec![0u8; size];
                b.eth_read(&wb, l.b, dst, &mut all).unwrap();
                if all != data {
                    failures += 1;
                    let first = (0..size).find(|&i| all[i] != data[i]);
                    println!("MEASURE no-poll cmd {cmd_bytes} rep {rep}: first bad byte {first:?}");
                    if failures == 1 {
                        // Did the missing bytes land somewhere else?
                        let f = first.unwrap();
                        let needle = &data[f..(f + 16).min(size)];
                        let (lo, hi) = (eth::E1_IMAGE, eth::BUFFERS.end);
                        let mut l1 = vec![0u8; (hi - lo) as usize];
                        b.eth_read(&wb, l.b, lo, &mut l1).unwrap();
                        let hits: Vec<u64> = (0..l1.len() - needle.len())
                            .filter(|&i| &l1[i..i + needle.len()] == needle)
                            .map(|i| lo + i as u64)
                            .collect();
                        println!(
                            "MEASURE missing bytes (want at {:#x}) found at {hits:x?}",
                            dst + f as u64
                        );
                        // And does the sender still have them?
                        let mut s = vec![0u8; 16];
                        a.eth_read(&wa, l.a, src + f as u64, &mut s).unwrap();
                        println!(
                            "MEASURE sender's source still holds them: {}",
                            s[..needle.len()] == *needle
                        );
                    }
                }
            }
        }
        println!("MEASURE no-poll: {failures} of 120 transfers incomplete after 5 ms");
    });
}

/// 200 Tensix -> Tensix transfers, each with fresh data and a fenced sentinel,
/// each verified. `NO_LANDING_WAIT=1` turns the receiver's wait off on silicon,
/// to see whether it is load-bearing.
#[test]
#[ignore = "benchmark"]
fn mover_integrity() {
    with_cards(|a, b| {
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let (ga, gb) = (a.ethernet_grid(&wa).unwrap(), b.ethernet_grid(&wb).unwrap());
        let l = discover(a, &wa, &ga, b, &wb, &gb).unwrap()[0];
        let (ta, tb) = (tile(a, 3, 4), tile(b, 4, 4));
        let mut m = Mover::start(a, &wa, b, &wb, l, tt_firmware_images::ETH_E1).unwrap();
        if std::env::var_os("NO_LANDING_WAIT").is_some() {
            b.eth_write(&wb, l.b, eth::mover::LANDING_WAIT, &[0; 4])
                .unwrap();
            println!("MEASURE landing wait OFF");
        }
        let mut bad = 0;
        for rep in 0..200u32 {
            let size = [16usize, 4096, 4112, 65536, 131072][rep as usize % 5];
            let data = pattern(size, rep * 7 + 1);
            a.write(&wa, ta, 0x8_0000, &data).unwrap();
            b.write(&wb, tb, 0x8_0000, &vec![0xEE; size + 64]).unwrap();
            let _ = a.read32(&wa, ta, 0x8_0000 + size as u64 - 4).unwrap();
            let _ = b.read32(&wb, tb, 0x8_0000 + size as u64 + 60).unwrap();
            m.send(
                a,
                &wa,
                Dir::AToB,
                Source::Tensix(ta, 0x8_0000),
                Dest::Tensix(tb, 0x8_0000),
                size as u32,
            )
            .unwrap_or_else(|e| panic!("rep {rep}: {e}"));
            let mut got = vec![0u8; size + 64];
            b.read(&wb, tb, 0x8_0000, &mut got).unwrap();
            if got[..size] != data[..] || got[size..].iter().any(|&v| v != 0xEE) {
                bad += 1;
                let first = (0..size).find(|&i| got[i] != data[i]);
                println!("MEASURE rep {rep} size {size}: wrong, first bad byte {first:?}");
            }
        }
        println!("MEASURE integrity: {bad} of 200 transfers wrong");
        a.park_e1(&wa, l.a).unwrap();
        b.park_e1(&wb, l.b).unwrap();
    });
}
