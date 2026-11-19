//! Probe: whether an L1 -> GDDR write shorter than 16 bytes lands as asked, for a block-float KV
//! cache's row write (`docs/plans/block-float-kv.md` K0).
//!
//! `cargo xtask silicon --device 0 --include-ignored --filter silicon_probe_small_writes`
//!
//! A block-float tile's row is two face-rows of datums (16 bytes each at BFP8, 8 at BFP4) and two
//! shared exponents, **one byte each**, at `TILE_DATA + 16 face + row` (`TileImage::exponent_byte_offset`).
//! Writing one position of a cache in place therefore needs single-byte writes, whose GDDR
//! neighbours (other rows' exponents) must stay as they were. `Descriptor::decode` takes any
//! nonzero length whose addresses are congruent mod 16; whether the card writes exactly those
//! bytes is what this measures.
//!
//! Every case writes from an L1 slot holding a pattern to a GDDR region poisoned with `0xA5`, then
//! reads the whole region back: the written bytes must be the pattern's, every other byte poison.
//! - lengths 1, 2, 4, 8 and 16 at each offset 0..32 inside a 64-byte burst, each in its own region;
//! - one tile's whole exponent section, 64 single-byte writes, each its own entry through the
//!   ports in turn (a row write's exponents, every row of a tile), in one region;
//! - both NoCs NC writes on.

#![cfg(feature = "silicon")]

use tt_device::tlb::WindowKind;
use tt_isa::dm::{op, Mover};
use tt_kernels::dm::{DataMover, WriteNoc};
use tt_tests::bench::on_card;

/// Source slot in L1, 64-byte aligned.
const L1: u32 = 0x2_0000;
/// Bytes a case's GDDR region spans.
const REGION: u32 = 256;
/// Where in a region (and in the L1 slot) the written bytes start, before their offset.
const AT: u32 = 64;

fn pattern(i: usize) -> u8 {
    // No byte equals the poison.
    let b = (i as u32).wrapping_mul(0x9E37_79B1).rotate_left(7) as u8;
    if b == 0xA5 {
        0x5A
    } else {
        b
    }
}

#[test]
#[ignore = "probe"]
fn short_writes_land_exactly() {
    on_card(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let w4 = d.alloc_window(WindowKind::FourGib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tt_tests::harness::tensix_tile();
        let chans: Vec<_> = dram.channels().collect();
        let src: Vec<u8> = (0..REGION as usize).map(pattern).collect();
        d.l1_write(&w, t, L1 as u64, &src).unwrap();
        let base = 160u32 << 20;
        // (region, writes as (offset from AT, length)) -- each write's L1 source is the same offset of
        // the slot, so both ends are congruent mod 16 (mod 256, in fact).
        let mut cases: Vec<(String, Vec<(u32, u32)>)> = Vec::new();
        for len in [1u32, 2, 4, 8, 16] {
            for o in 0..32u32 {
                cases.push((format!("{len} B at +{o}"), vec![(o, len)]));
            }
        }
        // A BFP tile's exponent section: header 16 bytes, then one byte per face-row, 64 of them.
        // From AT - 64, so the section sits at region bytes 16..80 as it does in a slot.
        cases.push((
            "an exponent section a byte at a time".into(),
            (0..64u32).map(|e| ((16 + e).wrapping_sub(AT), 1)).collect(),
        ));
        let image = tt_firmware_images::DM_NC.1;
        for noc in [WriteNoc::Noc0, WriteNoc::Noc1] {
            let list: Vec<[u32; 8]> = cases
                .iter()
                .enumerate()
                .flat_map(|(k, (_, writes))| {
                    let ch = chans[k % chans.len()].index() as u32;
                    let region = base + (k / chans.len()) as u32 * REGION;
                    writes.iter().enumerate().map(move |(n, &(o, len))| {
                        let off = AT.wrapping_add(o);
                        [
                            op::WRITE,
                            ch,
                            (n % tt_isa::dram::PORTS as usize) as u32,
                            region + off,
                            L1 + off,
                            len,
                            0,
                            0,
                        ]
                    })
                })
                .collect();
            let regions = cases.len().div_ceil(chans.len()) as u64 * REGION as u64;
            let poison = vec![0xA5u8; regions as usize];
            for ch in &chans {
                d.dram_write(&w4, ch.range(base as u64, regions).unwrap(), &poison)
                    .unwrap();
            }
            let mut m = DataMover::start_on(d, &w, t, &dram, Mover::NC, image).unwrap();
            m.set_write_noc(d, &w, noc).unwrap();
            for part in list.chunks(256) {
                m.submit_list(d, &w, part).unwrap();
                m.wait(d, &w).unwrap();
            }
            m.stop(d, &w).unwrap();
            let mut back: Vec<Vec<u8>> = Vec::new();
            for ch in &chans {
                let mut b = vec![0u8; regions as usize];
                d.dram_read(&w4, ch.range(base as u64, regions).unwrap(), &mut b)
                    .unwrap();
                back.push(b);
            }
            let mut bad = Vec::new();
            for (k, (name, writes)) in cases.iter().enumerate() {
                let at = (k / chans.len()) * REGION as usize;
                let got = &back[k % chans.len()][at..at + REGION as usize];
                let mut want = vec![0xA5u8; REGION as usize];
                for &(o, len) in writes {
                    let off = AT.wrapping_add(o) as usize;
                    want[off..off + len as usize].copy_from_slice(&src[off..off + len as usize]);
                }
                if got != want.as_slice() {
                    let diff: Vec<usize> = (0..REGION as usize)
                        .filter(|&i| got[i] != want[i])
                        .collect();
                    bad.push(format!(
                        "{name}: {} bytes differ, at {:?}",
                        diff.len(),
                        &diff[..diff.len().min(12)]
                    ));
                }
            }
            println!(
                "{noc:?}: {} cases, {} writes, {} wrong",
                cases.len(),
                list.len(),
                bad.len()
            );
            for b in &bad {
                println!("  {b}");
            }
            assert!(bad.is_empty(), "{noc:?}: {} cases wrong", bad.len());
        }
    });
}
