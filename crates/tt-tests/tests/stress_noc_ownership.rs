//! Stress: every tile's mover reading and writing GDDR at once, for as long
//! as `STRESS_SECS` says (default 60 s on silicon, two rounds on the
//! simulator), with every byte checked.
//!
//! The shapes Blackhole's DRAM endpoints are known to hang under, made safe by
//! construction and then driven hard (`DramChannel::owns`):
//! * SYS-1419, one endpoint fed by both NoCs: here reads go out on NoC #0
//!   (ports 0 and 2) while writes go out on NoC #1 (port 1) -- both NoCs on
//!   every channel, never on one endpoint. The host's own GDDR traffic, on
//!   NoC #0, runs between the rounds' submit and wait.
//! * BH-76, every tile driving all of a channel's ports on different virtual
//!   channels: here everything is on static VC 1 (`Command::registers`), and
//!   the all-NoC-#0 rounds drive both of NoC #0's ports from every tile.
//!
//! Each round moves the writes to the next of NoC #1, NoC #0, and the two
//! in turn. A round is: every tile reads its
//! own pattern from GDDR (64 KiB entries, rotating channel and port) and
//! writes a fresh pattern back to its own region, interleaved in one list
//! after a barrier; then the host checks a rotating sample of tiles' reads and
//! writes.

use std::time::{Duration, Instant};

use tt_device::tlb::WindowKind;
use tt_isa::dm::{self, op};
use tt_isa::noc::niu::Niu;
use tt_isa::noc::Noc0;
use tt_kernels::dm::{DataMover, WriteNoc};
use tt_tests::harness::{in_device, tensix_grid, tile};

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

const LEN: u32 = 65536;
/// Reads and writes per tile per round: 384 KiB each way, both halves inside
/// the data arena.
const PAIRS: u32 = 6;
/// Reads land in L1 from here, writes leave from `L1_WRITE`.
const L1_READ: u32 = 0x2_0000;
const L1_WRITE: u32 = L1_READ + PAIRS * LEN;
const _: () = assert!(L1_WRITE as u64 + (PAIRS * LEN) as u64 <= tt_isa::l1::DATA.end);
/// Each tile's 1 MiB to read and 1 MiB to write, per channel.
const READ_BASE: u64 = 256 << 20;
const WRITE_BASE: u64 = 512 << 20;
/// The host's own region, apart from every tile's.
const HOST_BASE: u64 = 1 << 30;
/// Tiles whose bytes are checked each round.
const SAMPLE: usize = 4;

fn secs() -> Duration {
    let default = if tt_tests::backend::ON_SILICON { 60 } else { 0 };
    Duration::from_secs(
        std::env::var("STRESS_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default),
    )
}

#[test]
#[ignore = "stress: long-running, every tile"]
fn every_tile_reads_on_noc0_and_writes_on_either_noc_and_nothing_hangs() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let coords: Vec<_> = tensix_grid(d).tiles::<Noc0>().collect();
        // All of them on silicon; four on the simulator, which is slow.
        let n = if tt_tests::backend::ON_SILICON {
            coords.len()
        } else {
            4
        };
        let mut movers: Vec<DataMover<Noc0>> = coords[..n]
            .iter()
            .map(|c| {
                let t = tile(d, c.x(), c.y());
                DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap()
            })
            .collect();
        let chans: Vec<_> = dram.channels().collect();
        let nc = chans.len() as u32;
        let coord = movers[0].tile();
        // Entry i of tile t uses channel (i + t) % nc, its (i / nc)-th 64 KiB
        // of the tile's MiB there.
        let at = |base: u64, t: usize, i: u32| base + ((t as u64) << 20) + ((i / nc) * LEN) as u64;
        let ch_of = |t: usize, i: u32| chans[((i + t as u32) % nc) as usize];
        // Every tile's read source, written once.
        for t in 0..n {
            let p = pattern((PAIRS * LEN) as usize, t as u32 + 1);
            for i in 0..PAIRS {
                let r = ch_of(t, i).range(at(READ_BASE, t, i), LEN as u64).unwrap();
                d.dram_write(&w4, r, &p[(i * LEN) as usize..][..LEN as usize])
                    .unwrap();
            }
        }
        let list = |t: usize, round: u32| {
            let mut l = vec![[
                op::BARRIER,
                n as u32 * (round + 1),
                coord.x() as u32,
                coord.y() as u32,
                0,
                0,
                0,
                0,
            ]];
            for i in 0..PAIRS {
                let ch = ch_of(t, i).index() as u32;
                // Port hints rotate all three; the mover maps each onto one
                // its NIU owns.
                let port = (i + t as u32) % 3;
                l.push([
                    op::READ,
                    ch,
                    port,
                    at(READ_BASE, t, i) as u32,
                    L1_READ + i * LEN,
                    LEN,
                    0,
                    0,
                ]);
                l.push([
                    op::WRITE,
                    ch,
                    port,
                    at(WRITE_BASE, t, i) as u32,
                    L1_WRITE + i * LEN,
                    LEN,
                    0,
                    0,
                ]);
            }
            l
        };
        d.write32(&w, coord, dm::BARRIER_COUNTER, 0).unwrap();
        let host_data = pattern(1 << 20, 0xC0FFEE);
        let limit = secs();
        let started = Instant::now();
        let mut round = 0u32;
        loop {
            let noc = [WriteNoc::Noc1, WriteNoc::Noc0, WriteNoc::Alternate][round as usize % 3];
            // A fresh pattern for each sampled tile's writes this round.
            let sample: Vec<usize> = (0..SAMPLE.min(n))
                .map(|k| (round as usize * SAMPLE + k * 7) % n)
                .collect();
            for &t in &sample {
                let p = pattern((PAIRS * LEN) as usize, round * 1000 + t as u32);
                d.l1_write(&w, movers[t].tile(), L1_WRITE as u64, &p)
                    .unwrap();
            }
            for m in &movers {
                m.set_write_noc(d, &w, noc).unwrap();
            }
            for (t, m) in movers.iter_mut().enumerate() {
                m.submit_list(d, &w, &list(t, round)).unwrap();
            }
            // The host on NoC #0 while the movers run.
            for ch in &chans {
                let r = ch.range(HOST_BASE, host_data.len() as u64).unwrap();
                d.dram_write(&w4, r, &host_data).unwrap();
            }
            for (t, m) in movers.iter().enumerate() {
                m.wait(d, &w).unwrap_or_else(|e| {
                    panic!("round {round} ({noc:?} writes): tile {t} did not finish: {e}")
                });
            }
            for ch in &chans {
                let mut back = vec![0u8; host_data.len()];
                d.dram_read(
                    &w4,
                    ch.range(HOST_BASE, back.len() as u64).unwrap(),
                    &mut back,
                )
                .unwrap();
                assert!(
                    back == host_data,
                    "round {round}: the host's bytes in ch {} changed",
                    ch.index()
                );
            }
            for &t in &sample {
                let tile = movers[t].tile();
                let want_read = pattern((PAIRS * LEN) as usize, t as u32 + 1);
                let mut got = vec![0u8; (PAIRS * LEN) as usize];
                d.l1_read(&w, tile, L1_READ as u64, &mut got).unwrap();
                assert!(
                    got == want_read,
                    "round {round}: tile {t}'s reads did not land"
                );
                let want = pattern((PAIRS * LEN) as usize, round * 1000 + t as u32);
                for i in 0..PAIRS {
                    let mut back = vec![0u8; LEN as usize];
                    let r = ch_of(t, i).range(at(WRITE_BASE, t, i), LEN as u64).unwrap();
                    d.dram_read(&w4, r, &mut back).unwrap();
                    assert!(
                        back[..] == want[(i * LEN) as usize..][..LEN as usize],
                        "round {round} ({noc:?} writes): tile {t}'s write {i} did not land"
                    );
                }
            }
            round += 1;
            if round >= 3 && started.elapsed() >= limit {
                break;
            }
        }
        let gib = (round as f64 * n as f64 * 2.0 * (PAIRS * LEN) as f64) / (1u64 << 30) as f64;
        println!(
            "MEASURE stress: {round} rounds on {n} tiles in {:.1?}, {gib:.1} GiB moved, every sampled byte intact",
            started.elapsed()
        );
        for m in movers {
            m.set_write_noc(d, &w, Niu::Noc0).unwrap();
            m.stop(d, &w).unwrap();
        }
    });
}
