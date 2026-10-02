//! Phase 9 gate: the host reaches every usable GDDR6 channel.
//!
//! Tensors are to live on the card (Phase 9), so this is the first thing that
//! has to be true: each channel the chip reports trained is reachable through a
//! 4 GiB window, holds what was written, and its three endpoints are one memory.
//!
//! Ordered as a survey -- each test touches strictly more than the one before --
//! so `cargo xtask silicon --filter step15_dram::a --filter step15_dram::b ...`
//! attributes a hang to the first access that caused it (Silicon operating
//! notes). Every address comes from [`Dram`], which is built from the chip's own
//! telemetry, and every endpoint is a translated coordinate: `X = 0` names
//! nothing in the Tensix rows under translation (`NoC/Coordinates.md:35-42`).

use tt_device::tlb::WindowKind;
use tt_isa::dram::{Dram, CHANNEL_BYTES, PORTS};
use tt_isa::noc::niu::Niu;
use tt_tests::harness::{in_device, Dev};

fn grid(dev: &mut Dev<'_>) -> Dram {
    let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
    let d = dev.dram_grid(&w).unwrap();
    println!("MEASURE dram channels usable: {}", d.channel_count());
    assert_eq!(
        d.channel_count(),
        8,
        "a p150 (and ttsim) has eight channels"
    );
    d
}

/// Deterministic, distinct per seed, not self-similar.
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

/// Step 1: one dword read from each channel, nothing written.
#[test]
fn a_single_read_of_each_channel_completes() {
    in_device(|dev| {
        let d = grid(dev);
        let w = dev.alloc_window(WindowKind::FourGib).unwrap();
        for ch in d.channels() {
            let mut b = [0u8; 4];
            dev.dram_read(&w, ch.range(0x1000, 4).unwrap(), &mut b)
                .unwrap();
            println!(
                "MEASURE dram ch{} [0x1000] = {:#010x}",
                ch.index(),
                u32::from_le_bytes(b)
            );
        }
    });
}

/// Step 2: a pattern round-trips in every channel, low, middle and near the top
/// of the extent, and one channel's write is not visible in another.
#[test]
fn b_patterns_round_trip_in_every_channel() {
    in_device(|dev| {
        let d = grid(dev);
        let w = dev.alloc_window(WindowKind::FourGib).unwrap();
        const LEN: usize = 64 * 1024;
        let offsets = [0x0, 0x8000_0000, CHANNEL_BYTES - LEN as u64];
        for &off in &offsets {
            for ch in d.channels() {
                let data = pattern(LEN, (off as u32) ^ (ch.index() as u32 * 0x9E37));
                dev.dram_write(&w, ch.range(off, LEN as u64).unwrap(), &data)
                    .unwrap();
            }
            // Read only after every channel was written, so a write landing in
            // the wrong channel is caught as a mismatch there.
            for ch in d.channels() {
                let data = pattern(LEN, (off as u32) ^ (ch.index() as u32 * 0x9E37));
                let mut back = vec![0u8; LEN];
                dev.dram_read(&w, ch.range(off, LEN as u64).unwrap(), &mut back)
                    .unwrap();
                assert!(back == data, "channel {} at {off:#x}", ch.index());
            }
        }
    });
}

/// Step 3: the endpoints of a channel are one memory
/// (`BlackholeA0/README.md:5`), and those of different channels are not --
/// through the endpoints the host's NoC #0 owns (`DramChannel::owns`; port 1
/// is NoC #1's, and a host on NoC #0 there is half of the SYS-1419 hang).
#[test]
fn c_the_three_endpoints_of_a_channel_alias() {
    in_device(|dev| {
        let d = grid(dev);
        let w = dev.alloc_window(WindowKind::FourGib).unwrap();
        const AT: u64 = 0x4000;
        for ch in d.channels() {
            for p in 0..PORTS {
                let Some(e) = ch.endpoint(Niu::Noc0, p) else {
                    continue;
                };
                let tag = 0xA11A_0000 | (ch.index() as u32) << 8 | p as u32;
                dev.write32(&w, e, AT, tag).unwrap();
                // Every other channel's last value is untouched, and every port
                // of this one sees the write.
                for other in d.channels() {
                    for q in 0..PORTS {
                        let Some(via) = other.endpoint(Niu::Noc0, q) else {
                            continue;
                        };
                        let v = dev.read32(&w, via, AT).unwrap();
                        if other == ch {
                            assert_eq!(v, tag, "ch{} port {q} after a write via {p}", ch.index());
                        } else if other.index() < ch.index() {
                            assert_eq!(
                                v >> 8 & 0xFF,
                                other.index() as u32,
                                "ch{} disturbed",
                                other.index()
                            );
                        }
                    }
                }
            }
        }
    });
}

/// Refusals: nothing past the extent, and a buffer must match its range.
#[test]
fn d_out_of_range_access_is_refused_before_the_transport() {
    in_device(|dev| {
        let d = grid(dev);
        let ch = d.channel(0).unwrap();
        assert!(ch.range(CHANNEL_BYTES, 1).is_none());
        assert!(ch.range(CHANNEL_BYTES - 1, 2).is_none());
        let w = dev.alloc_window(WindowKind::FourGib).unwrap();
        let before = dev.traffic();
        let r = ch.range(0, 8).unwrap();
        assert!(dev.dram_write(&w, r, &[0; 4]).is_err());
        assert!(dev.dram_read(&w, r, &mut [0; 16]).is_err());
        assert_eq!(dev.traffic(), before, "nothing may be sent");
    });
}
