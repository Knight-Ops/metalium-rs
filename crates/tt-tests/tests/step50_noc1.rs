//! The data mover writes GDDR through NoC #1 (`tt_isa::dm::WRITE_NOC`) while
//! it reads through NoC #0, and every byte lands as it does with both on
//! NoC #0.
//!
//! Each GDDR endpoint belongs to one NoC (`DramChannel::owns`): NoC #1 writes
//! go to port 1 of every channel, and NoC #0 traffic -- the mover's reads, its
//! NoC #0 writes, the host -- to ports 0 and 2. One endpoint fed by both NoCs
//! is Blackhole's SYS-1419 arbiter hang, which took card 0 down when tiles on
//! both NoCs shared ports (2026-10-02). The entries name ports as before; the
//! mover maps each onto one its NIU owns, so the lists here rotate all three.
//!
//! The entries and the tile's coordinate are the same on both NoCs only if
//! NoC #1's NIU translates coordinates as NoC #0's does
//! (`NoC/Coordinates.md`, "Coordinate Translation"): checked first.

use tt_device::tlb::WindowKind;
use tt_isa::dm::op;
use tt_isa::dram::PORTS;
use tt_isa::noc::niu::{self, Niu};
use tt_kernels::dm::{DataMover, WriteNoc};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile};

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

const L1_AT: u32 = 0x2_0000;
/// Where each pattern sits in its channel.
const DRAM_AT: u64 = 96 << 20;
/// Three NoC requests per entry, the last one short; 24 of them (8 channels,
/// 3 ports) fit the data arena below the mailboxes (`tt_isa::l1::DATA`).
const LEN: usize = 32 * 1024 + 64;
const _: () = assert!(L1_AT as u64 + 24 * LEN as u64 <= tt_isa::l1::DATA.end);

/// NoC #1 first, then back to NoC #0, then taking turns: every switch.
const WRITE_NOCS: [WriteNoc; 4] = [
    WriteNoc::Noc1,
    WriteNoc::Noc0,
    WriteNoc::Alternate,
    WriteNoc::Noc1,
];

#[test]
fn noc1_translates_coordinates_as_noc0_does() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let cfg0 = d.read32(&w, t, niu::NOC0_BASE + niu::NIU_CFG_0).unwrap();
        let cfg1 = d.read32(&w, t, niu::NOC1_BASE + niu::NIU_CFG_0).unwrap();
        let on = |c: u32| c & niu::NIU_CFG_0_TRANSLATION_ENABLED != 0;
        println!("NIU_CFG_0: NoC #0 {cfg0:#x}, NoC #1 {cfg1:#x}");
        assert_eq!(on(cfg0), on(cfg1), "the NIUs disagree on translation");
        let id0 = d
            .read32(&w, t, niu::NOC0_BASE + niu::NOC_ID_LOGICAL)
            .unwrap();
        let id1 = d
            .read32(&w, t, niu::NOC1_BASE + niu::NOC_ID_LOGICAL)
            .unwrap();
        println!("NOC_ID_LOGICAL: NoC #0 {id0:#x}, NoC #1 {id1:#x}");
        assert_eq!(id0, id1, "the NIUs name this tile differently");
    });
}

/// One entry per channel and port: each its own slot of L1 and of GDDR.
fn list(dram: &tt_isa::dram::Dram, kind: u32) -> Vec<[u32; 8]> {
    let mut out = Vec::new();
    for ch in dram.channels() {
        for port in 0..PORTS {
            let i = out.len() as u32;
            out.push([
                kind,
                ch.index() as u32,
                port as u32,
                DRAM_AT as u32 + port as u32 * LEN as u32,
                L1_AT + i * LEN as u32,
                LEN as u32,
                0,
                0,
            ]);
        }
    }
    out
}

#[test]
fn writes_through_either_noc_land_through_every_channel_and_port() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let mut m = DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap();
        let reads = list(&dram, op::READ);
        let writes = list(&dram, op::WRITE);
        let range = |e: &[u32; 8]| {
            dram.channel(e[1] as u8)
                .unwrap()
                .range(e[3] as u64, LEN as u64)
                .unwrap()
        };
        for (k, noc) in WRITE_NOCS.into_iter().enumerate() {
            m.set_write_noc(d, &w, noc).unwrap();
            // L1 -> DRAM through `noc`, from fresh patterns.
            let data = pattern(writes.len() * LEN, k as u32 + 77);
            d.l1_write(&w, t, L1_AT as u64, &data).unwrap();
            m.run_list(d, &w, &writes).unwrap();
            for (i, e) in writes.iter().enumerate() {
                let mut back = vec![0u8; LEN];
                d.dram_read(&w4, range(e), &mut back).unwrap();
                assert!(
                    back == data[i * LEN..][..LEN],
                    "{noc:?}: write {i} did not land"
                );
            }
            // And back through NoC #0's reads, into a zeroed L1.
            d.l1_write(&w, t, L1_AT as u64, &vec![0u8; reads.len() * LEN])
                .unwrap();
            m.run_list(d, &w, &reads).unwrap();
            let mut back = vec![0u8; reads.len() * LEN];
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            assert!(back == data, "{noc:?}: what was written did not read back");
        }
        m.set_write_noc(d, &w, Niu::Noc0).unwrap();
        m.stop(d, &w).unwrap();
    });
}

/// Reads on NoC #0 and writes on NoC #1, then writes taking turns between
/// the NoCs, interleaved in one list, so both NIUs have requests in flight at
/// once: each half lands, and the list's end waits for both.
#[test]
fn reads_and_writes_on_both_nocs_at_once() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let mut m = DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap();
        let chans: Vec<_> = dram.channels().collect();
        const N: usize = 24;
        // Even entries read slot i from GDDR; odd ones write slot i to GDDR
        // at a second region, so the halves touch disjoint bytes.
        let src = pattern(N * LEN, 3);
        let up = pattern(N * LEN, 4);
        let at = |i: usize, region: u64| DRAM_AT + region + (i * LEN) as u64;
        let mut list = Vec::new();
        for i in 0..N {
            let ch = chans[i % chans.len()];
            let l1 = L1_AT + (i * LEN) as u32;
            if i % 2 == 0 {
                d.dram_write(
                    &w4,
                    ch.range(at(i, 0), LEN as u64).unwrap(),
                    &src[i * LEN..][..LEN],
                )
                .unwrap();
                list.push([
                    op::READ,
                    ch.index() as u32,
                    (i % 3) as u32,
                    at(i, 0) as u32,
                    l1,
                    LEN as u32,
                    0,
                    0,
                ]);
            } else {
                d.l1_write(&w, t, l1 as u64, &up[i * LEN..][..LEN]).unwrap();
                list.push([
                    op::WRITE,
                    ch.index() as u32,
                    (i % 3) as u32,
                    at(i, 64 << 20) as u32,
                    l1,
                    LEN as u32,
                    0,
                    0,
                ]);
            }
        }
        for noc in [WriteNoc::Noc1, WriteNoc::Alternate] {
            for i in (1..N).step_by(2) {
                let ch = chans[i % chans.len()];
                d.dram_write(
                    &w4,
                    ch.range(at(i, 64 << 20), LEN as u64).unwrap(),
                    &vec![0u8; LEN],
                )
                .unwrap();
            }
            // The reads' slots zeroed, so each pass shows its own reads.
            for i in (0..N).step_by(2) {
                d.l1_write(&w, t, L1_AT as u64 + (i * LEN) as u64, &vec![0u8; LEN])
                    .unwrap();
            }
            m.set_write_noc(d, &w, noc).unwrap();
            m.run_list(d, &w, &list).unwrap();
            for i in 0..N {
                let ch = chans[i % chans.len()];
                if i % 2 == 0 {
                    let mut back = vec![0u8; LEN];
                    d.l1_read(&w, t, L1_AT as u64 + (i * LEN) as u64, &mut back)
                        .unwrap();
                    assert!(
                        back == src[i * LEN..][..LEN],
                        "{noc:?}: read {i} did not land"
                    );
                } else {
                    let mut back = vec![0u8; LEN];
                    d.dram_read(
                        &w4,
                        ch.range(at(i, 64 << 20), LEN as u64).unwrap(),
                        &mut back,
                    )
                    .unwrap();
                    assert!(
                        back == up[i * LEN..][..LEN],
                        "{noc:?}: write {i} did not land"
                    );
                }
            }
        }
        m.set_write_noc(d, &w, Niu::Noc0).unwrap();
        m.stop(d, &w).unwrap();
    });
}

/// As `step49_in_flight`, for writes on NoC #1: more requests than the 8-bit
/// counter holds land whole with the default cap and with a cap of one, and
/// at one they wait on NoC #1's own counter. 16 KiB each, so a write is still
/// in flight when the next is issued: on card 0, 1 KiB writes were all
/// acknowledged within one issue period and the cap never bound.
#[test]
fn noc1s_cap_holds_on_its_own_counter() {
    const ENTRIES: u32 = 300;
    const BIG: u32 = 16384;
    /// L1 source slots, reused round robin: entry i writes slot i % SLOTS.
    const SLOTS: u32 = 32;
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let mut m = DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap();
        let ch = dram.channels().next().unwrap();
        let src = pattern((SLOTS * BIG) as usize, 5);
        d.l1_write(&w, t, L1_AT as u64, &src).unwrap();
        let list: Vec<[u32; 8]> = (0..ENTRIES)
            .map(|i| {
                let at = DRAM_AT as u32 + i * BIG;
                let l1 = L1_AT + (i % SLOTS) * BIG;
                [op::WRITE, ch.index() as u32, i % 3, at, l1, BIG, 0, 0]
            })
            .collect();
        let want: Vec<u8> = (0..ENTRIES)
            .flat_map(|i| {
                src[((i % SLOTS) * BIG) as usize..][..BIG as usize]
                    .iter()
                    .copied()
            })
            .collect();
        let region = ch.range(DRAM_AT, want.len() as u64).unwrap();
        m.set_write_noc(d, &w, Niu::Noc1).unwrap();
        for (cap, what) in [
            (0u32, "the maximum cap"),
            (tt_isa::dm::TILE_IN_FLIGHT_CAP, "the tile movers' cap"),
            (1, "a cap of one"),
        ] {
            d.dram_write(&w4, region, &vec![0u8; want.len()]).unwrap();
            m.set_in_flight_cap(d, &w, cap).unwrap();
            let before = m.throttle(d, &w).unwrap();
            m.run_list(d, &w, &list).unwrap();
            let mut back = vec![0u8; want.len()];
            d.dram_read(&w4, region, &mut back).unwrap();
            assert!(back == want, "{what}: the list did not land whole");
            let th = m.throttle(d, &w).unwrap();
            println!("{what}: {} stalls", th.stalls - before.stalls);
            // ttsim completes each request before the next (divergence row 72).
            if cap == 1 && tt_tests::backend::ON_SILICON {
                assert!(
                    th.stalls > before.stalls,
                    "{what}: the throttle never engaged"
                );
            }
        }
        m.set_in_flight_cap(d, &w, tt_isa::dm::TILE_IN_FLIGHT_CAP)
            .unwrap();
        m.set_write_noc(d, &w, Niu::Noc0).unwrap();
        m.stop(d, &w).unwrap();
    });
}
