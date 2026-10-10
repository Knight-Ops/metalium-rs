//! X6: NIU request-initiator register persistence, the premise of a
//! `noc_async_read_set_state` / `noc_async_read_with_state` fast path in the
//! mover.
//!
//! What the pinned page says (`BlackholeA0/NoC/MemoryMap.md`, "NIU Request
//! Initiators" and `NOC_CMD_CTRL`):
//!
//! * a request is "the fields of this initiator" at the moment `NOC_CMD_CTRL`'s
//!   low bit is written; the fields are ordinary read/write registers and the
//!   page never says an issue consumes or clears any of them;
//! * coordinate translation is applied to the request, and "Wormhole would also
//!   write the translated coordinates back to the MMIO registers, but Blackhole
//!   does not: the registers retain the original values provided by software";
//! * hardware does write "some of the reserved fields within `NOC_CTRL`"
//!   (bits 10-12 and 18-26, "software should always write 0 to these bits, but
//!   hardware might subsequently change them");
//! * "software must not write to any fields of the initiator" until hardware has
//!   cleared `NOC_CMD_CTRL`'s low bit; "recommended that software always checks
//!   the low bit of `NOC_CMD_CTRL` before writing to any fields".
//!
//! The page therefore does not contradict persistence, and says outright that
//! the address and coordinate words keep their values. It says nothing about
//! `NOC_CTRL`'s reserved bits after an issue, so a fast path must not rely on
//! `NOC_CTRL` reading back as written.
//!
//! This file gates the premise on ttsim and holds the silicon-only arm; the
//! mover's fast path gates follow in the same file.

mod noc_support;

use tt_isa::noc::niu::{initiator, Command, Endpoint, Niu, TxnId};
use tt_isa::noc::probe;
use tt_tests::harness::{self, Dev};

use noc_support::{partial, read_bytes, run_probe, snapshot, unicast, write_bytes};

/// The neighbour whose L1 is read, claimed through the grid-checked constructor.
const SRC_TILE: (u8, u8) = (4, 4);
/// Source data in the neighbour's L1, then destinations in the gate tile's.
const SRC_A: u32 = 0x3_0000;
const SRC_B: u32 = 0x3_1000;
const DEST: u32 = 0x3_2000;
const LEN: u32 = 2048;
const POISON: u8 = 0xEE;

fn pattern(seed: u8) -> Vec<u8> {
    (0..LEN as usize)
        .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
        .collect()
}

/// Bit `i` of a [`partial`] mask is `probe::REGISTERS[i]`.
const fn bit(register: u64) -> u32 {
    let mut i = 0;
    while probe::REGISTERS[i] != register {
        i += 1;
    }
    1 << i
}
const TARG_LO: u32 = bit(initiator::TARG_ADDR_LO);
const RET_LO: u32 = bit(initiator::RET_ADDR_LO);
const LEN_BE: u32 = bit(initiator::AT_LEN_BE);

/// The ten register values of a read of `len` bytes from the neighbour's `src`
/// into the gate tile's `dest`, as the typed encoder lays them out.
fn read_regs(me: (u8, u8), src: u32, dest: u32, len: u32) -> Vec<(u64, u32)> {
    Command::Read {
        from: Endpoint {
            x: SRC_TILE.0,
            y: SRC_TILE.1,
            addr: src,
        },
        to_local: dest,
        len,
    }
    .registers(me, TxnId::new(1).unwrap(), Niu::Noc0)
    .unwrap()
    .to_vec()
}

/// What the second and later requests leave unwritten.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Skip {
    /// Nothing wrongly skipped: each later request writes exactly the words
    /// that differ from the one before.
    None,
    /// The second request forgets `RET_ADDR_LO`, the destination: the negative
    /// control.
    SecondDestination,
}

/// Four reads on one initiator, each writing only the registers that changed
/// since the previous one:
///
/// 0. everything: `A` into `DEST`;
/// 1. `TARG_ADDR_LO` and `RET_ADDR_LO`: `B` into `DEST + 0x1000`;
/// 2. `RET_ADDR_LO` alone: `A` again into `DEST + 0x2000` (so the source address
///    written by request 1 must also have persisted: it reads `B`);
/// 3. `TARG_ADDR_LO`, `RET_ADDR_LO` and `AT_LEN_BE`: 1024 bytes of `A` into
///    `DEST + 0x3000`, so the length written afterwards is not the length
///    before.
///
/// Every destination is poisoned first and checked afterwards against the
/// expected bytes; a request that went to the wrong place or with the wrong
/// length shows in the data.
fn persistence_case(dev: &mut Dev<'_>, skip: Skip, initiator1: bool) -> Result<(), String> {
    let extra = if initiator1 {
        probe::flag::INITIATOR_1
    } else {
        0
    };
    let me = harness::tensix_tile();
    let src = harness::tile(dev, SRC_TILE.0, SRC_TILE.1);
    let a = pattern(0x11);
    let b = pattern(0x93);
    write_bytes(dev, src, u64::from(SRC_A), &a);
    write_bytes(dev, src, u64::from(SRC_B), &b);
    let dests = [DEST, DEST + 0x1000, DEST + 0x2000, DEST + 0x3000];
    for d in dests {
        write_bytes(dev, me, u64::from(d), &vec![POISON; LEN as usize]);
    }
    let m = (me.x(), me.y());
    let r0 = read_regs(m, SRC_A, dests[0], LEN);
    let r1 = read_regs(m, SRC_B, dests[1], LEN);
    let r2 = read_regs(m, SRC_A, dests[2], LEN);
    let r3 = read_regs(m, SRC_A, dests[3], 1024);
    let second = if skip == Skip::SecondDestination {
        TARG_LO
    } else {
        TARG_LO | RET_LO
    };
    // Each request has its own transaction ID's counter to quiesce on, so the
    // probe's per-request check of "nothing in flight" holds.
    let reqs = [
        unicast(&r0, 1, probe::flag::PARTIAL | extra),
        partial(&r1, second, 1, extra),
        partial(&r2, RET_LO, 1, extra),
        partial(&r3, TARG_LO | RET_LO | LEN_BE, 1, extra),
    ];
    // The first request writes every register: `unicast` sets mask all-ones and
    // the image only masks under `PARTIAL`, which `unicast` above also carries.
    let res = run_probe(dev, me, &reqs, &[])?;
    for (k, r) in res.iter().enumerate() {
        if r.status != probe::status::OK {
            return Err(format!("request {k}: probe status {} ({r:?})", r.status));
        }
    }
    let want = [&a[..], &b[..], &b[..], &a[..1024]];
    for (k, d) in dests.iter().enumerate() {
        let got = read_bytes(dev, me, u64::from(*d), LEN as usize);
        let n = want[k].len();
        if got[..n] != *want[k] {
            return Err(format!(
                "request {k}: destination {d:#x} holds {:02x?}..., expected {:02x?}...",
                &got[..8],
                &want[k][..8]
            ));
        }
        if got[n..].iter().any(|&x| x != POISON) {
            return Err(format!("request {k}: wrote beyond its {n} bytes"));
        }
    }
    // The registers as the probe read them after each request: the words software
    // did not write must read as the previous request left them (all but `CTRL`,
    // whose reserved bits hardware may change).
    let snaps: Vec<[u32; 10]> = (0..reqs.len()).map(|k| snapshot(dev, me, k)).collect();
    eprintln!(
        "initiator snapshots after each request, {:?}:",
        probe::REGISTERS
    );
    for (k, s) in snaps.iter().enumerate() {
        eprintln!("  {k}: {s:#x?}");
    }
    let written = [r0, r1, r2, r3];
    for (k, s) in snaps.iter().enumerate() {
        for (i, &reg) in probe::REGISTERS.iter().take(10).enumerate() {
            if reg == initiator::CTRL {
                continue;
            }
            let expect = if k == 0 || reqs[k].mask & (1 << i) != 0 {
                written[k][i].1
            } else {
                snaps[k - 1][i]
            };
            if s[i] != expect && skip == Skip::None {
                return Err(format!(
                    "request {k}: register {reg:#x} reads {:#x}, expected {expect:#x}",
                    s[i]
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn a_request_in_the_model_writes_only_what_changed() {
    // Host check of the test's own premise: between the requests above, exactly
    // the registers named in each mask differ.
    let m = (3u8, 4u8);
    let r0 = read_regs(m, SRC_A, DEST, LEN);
    let r1 = read_regs(m, SRC_B, DEST + 0x1000, LEN);
    let r2 = read_regs(m, SRC_A, DEST + 0x2000, LEN);
    let r3 = read_regs(m, SRC_A, DEST + 0x3000, 1024);
    let diff = |a: &[(u64, u32)], b: &[(u64, u32)]| -> u32 {
        (0..10)
            .filter(|&i| a[i] != b[i])
            .fold(0, |m, i| m | (1 << i))
    };
    assert_eq!(diff(&r0, &r1), TARG_LO | RET_LO);
    // Request 2 re-reads `A` into a new place: the source differs from request
    // 1's, so a probe writing only `RET_ADDR_LO` really reads `B` again.
    assert_eq!(diff(&r1, &r2), TARG_LO | RET_LO);
    assert_eq!(diff(&r0, &r3), RET_LO | LEN_BE);
}

// ---------------------------------------------------------------------------
// The mover's fast read path against the slow one.
// ---------------------------------------------------------------------------

use tt_device::tlb::WindowKind;
use tt_isa::dm::op;
use tt_kernels::dm::DataMover;

/// Each channel's pattern, written at `DRAM_AT`, a MiB of it.
const DRAM_AT: u64 = 64 << 20;
const CHANNEL_SPAN: usize = 1 << 20;
/// Destinations: a list's entries are laid out from here, 64-aligned, below the
/// mover's list ring and mailbox.
const L1_AT: u32 = 0x2_0000;
const L1_ROOM: u32 = 0x7_0000;

fn bytes(len: usize, seed: u32) -> Vec<u8> {
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

/// A list of reads and what each must deliver, from the host's own copy of the
/// channels (the oracle: not the mover, not the slow path).
struct Case {
    name: String,
    list: Vec<[u32; 8]>,
    /// `(destination, expected bytes)` per entry.
    want: Vec<(u32, Vec<u8>)>,
}

type Chans = [(u32, Vec<u8>)];

/// Entries of the given `lens`, each from channel `chan(i)` through port
/// `port(i)` at the 64-aligned offset `off(i)`, laid out in L1 from `L1_AT`.
fn case(
    name: &str,
    chans: &Chans,
    lens: &[u32],
    chan: &dyn Fn(usize) -> usize,
    port: &dyn Fn(usize) -> u32,
    off: &dyn Fn(usize) -> u32,
) -> Case {
    let mut list = vec![];
    let mut want = vec![];
    let mut at = L1_AT;
    for (i, &len) in lens.iter().enumerate() {
        let (index, data) = &chans[chan(i) % chans.len()];
        let o = off(i) & !63;
        assert!(
            (o + len) as usize <= CHANNEL_SPAN && at + len <= L1_AT + L1_ROOM,
            "{name}: entry {i} does not fit"
        );
        list.push([op::READ, *index, port(i), DRAM_AT as u32 + o, at, len, 0, 0]);
        want.push((at, data[o as usize..(o + len) as usize].to_vec()));
        at = (at + len + 63) & !63;
    }
    Case {
        name: name.into(),
        list,
        want,
    }
}

#[allow(clippy::vec_init_then_push)]
fn cases(chans: &Chans) -> Vec<Case> {
    let mut rng = 0x1234_5678u32;
    let mut next = move |n: u32| {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng % n
    };
    let mut out = vec![];
    // One size on one channel and port: nothing but the addresses change.
    out.push(case(
        "4 KiB, one port",
        chans,
        &[4096; 64],
        &|_| 0,
        &|_| 0,
        &|i| i as u32 * 4096,
    ));
    out.push(case(
        "16 KiB, one port",
        chans,
        &[16384; 16],
        &|_| 0,
        &|_| 0,
        &|i| i as u32 * 16384,
    ));
    // The target coordinate changes every entry, the length never does.
    out.push(case(
        "4 KiB, every channel",
        chans,
        &[4096; 64],
        &|i| i,
        &|_| 0,
        &|i| i as u32 * 4096,
    ));
    out.push(case(
        "4 KiB, three ports",
        chans,
        &[4096; 63],
        &|_| 1,
        &|i| i as u32 % 3,
        &|i| i as u32 * 4096,
    ));
    // Many small entries.
    out.push(case(
        "64 B x 300",
        chans,
        &[64; 300],
        &|i| i / 7,
        &|i| i as u32 % 3,
        &|i| i as u32 * 64,
    ));
    let ragged: Vec<u32> = (0..150).map(|i| 64 + (i * 37) % 900).collect();
    out.push(case(
        "ragged lengths",
        chans,
        &ragged,
        &|i| i,
        &|i| i as u32 % 3,
        &|i| i as u32 * 1024,
    ));
    // An entry of more than one NIU request: the length changes inside it
    // (16384, then the remainder), and again at the next entry.
    out.push(case(
        "multi-request entries",
        chans,
        &[40_000, 16384, 16385, 65_000, 100],
        &|i| i,
        &|_| 2,
        &|i| i as u32 * 65_536,
    ));
    // Random shapes, many entries: channel, port, length and offset all free.
    for k in 0..4 {
        let lens: Vec<u32> = (0..180)
            .map(|_| match next(5) {
                0 => 64 * (1 + next(8)),
                1 => 1088,
                2 => 2048 * (1 + next(2)),
                3 => 4096,
                _ => 1 + next(6000),
            })
            .collect();
        let c: Vec<u32> = (0..180).map(|_| next(8)).collect();
        let p: Vec<u32> = (0..180).map(|_| next(3)).collect();
        let o: Vec<u32> = (0..180)
            .map(|_| next(CHANNEL_SPAN as u32 - 70_000))
            .collect();
        out.push(case(
            &format!("random {k}"),
            chans,
            &lens,
            &|i| c[i] as usize,
            &|i| p[i],
            &|i| o[i],
        ));
    }
    out
}

/// Run every case through the mover on both paths -- alternating which comes
/// first and switching paths between lists on one mover, so the initiator's
/// state crosses lists and switches -- and compare each delivery with the
/// oracle. With `corrupt_oracle`, one expected bit is flipped: the oracle's own
/// negative control.
fn fast_matches_slow(dev: &mut Dev<'_>, corrupt_oracle: bool) -> Result<(), String> {
    let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
    let w4 = dev.alloc_window(WindowKind::FourGib).unwrap();
    let dram = dev.dram_grid(&w).unwrap();
    let gate = tt_tests::backend::GATE_TILE;
    let t = harness::tile(dev, gate.0, gate.1);
    let mut m = DataMover::start(dev, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap();
    let chans: Vec<(u32, Vec<u8>)> = dram
        .channels()
        .map(|c| {
            let data = bytes(CHANNEL_SPAN, c.index() as u32 + 11);
            dev.dram_write(&w4, c.range(DRAM_AT, CHANNEL_SPAN as u64).unwrap(), &data)
                .unwrap();
            (c.index() as u32, data)
        })
        .collect();
    let mut result = Ok(());
    'cases: for (k, c) in cases(&chans).iter().enumerate() {
        let order = if k % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let mut got: Vec<Vec<u8>> = vec![];
        for fast in order {
            dev.write(&w, t, L1_AT as u64, &vec![0xEE; L1_ROOM as usize])
                .unwrap();
            m.set_read_fast(dev, &w, fast).unwrap();
            m.run_list(dev, &w, &c.list).unwrap();
            let mut back = vec![0u8; L1_ROOM as usize];
            dev.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            for (e, (at, want)) in c.want.iter().enumerate() {
                let mut want = want.clone();
                if corrupt_oracle && e == c.want.len() / 2 {
                    want[0] ^= 1;
                }
                if back[(*at - L1_AT) as usize..][..want.len()] != want[..] {
                    result = Err(format!(
                        "{}: entry {e} ({} bytes at {at:#x}) on the {} path is not what the channel holds",
                        c.name,
                        want.len(),
                        if fast { "fast" } else { "slow" }
                    ));
                    break 'cases;
                }
            }
            got.push(back);
        }
        // The whole destination region, delivered bytes and untouched poison alike.
        if got[0] != got[1] {
            result = Err(format!("{}: the paths left different L1", c.name));
            break;
        }
    }
    m.set_read_fast(dev, &w, false).unwrap();
    m.stop(dev, &w).unwrap();
    result
}

#[cfg(not(feature = "silicon"))]
mod simulator {
    use super::*;

    /// ttsim: the registers persist between requests. The four requests above
    /// land their bytes exactly where and as long as the changed registers say,
    /// and the unwritten registers read back as they were.
    #[test]
    fn simulator_initiator_registers_persist_between_requests() {
        harness::in_device(|dev| {
            for initiator1 in [false, true] {
                persistence_case(dev, Skip::None, initiator1)
                    .unwrap_or_else(|e| panic!("initiator {}: {e}", u8::from(initiator1)));
            }
        });
    }

    /// The two initiators are separate register files: a request on initiator 1
    /// that writes only the two address words, after a full request on
    /// initiator 0, does not inherit initiator 0's coordinates, length or
    /// control word -- it either never lands its bytes or ttsim refuses it.
    /// Control: the same two requests on one initiator land (the case above).
    #[test]
    fn initiator_1_does_not_inherit_initiator_0s_registers() {
        let lands = harness::survives(|dev| {
            let me = harness::tensix_tile();
            let src = harness::tile(dev, SRC_TILE.0, SRC_TILE.1);
            write_bytes(dev, src, u64::from(SRC_A), &pattern(0x11));
            write_bytes(dev, src, u64::from(SRC_B), &pattern(0x93));
            for d in [DEST, DEST + 0x1000] {
                write_bytes(dev, me, u64::from(d), &vec![POISON; LEN as usize]);
            }
            let m = (me.x(), me.y());
            let r0 = read_regs(m, SRC_A, DEST, LEN);
            let r1 = read_regs(m, SRC_B, DEST + 0x1000, LEN);
            let reqs = [
                unicast(&r0, 1, probe::flag::PARTIAL),
                partial(&r1, TARG_LO | RET_LO, 1, probe::flag::INITIATOR_1),
            ];
            let res = run_probe(dev, me, &reqs, &[]).expect("probe");
            assert!(res.iter().all(|r| r.status == probe::status::OK));
            let got = read_bytes(dev, me, u64::from(DEST + 0x1000), LEN as usize);
            assert_eq!(got, pattern(0x93), "request 1 did not land");
        });
        assert!(
            !lands,
            "a request on initiator 1 landed with only its address words written"
        );
    }

    /// The fast and slow read paths deliver byte-identical L1, equal to what the
    /// channels hold, over entries of every shape (`cases`), on one mover that
    /// switches path between lists.
    #[test]
    fn simulator_fast_read_path_is_byte_identical_to_the_slow_one() {
        harness::in_device(|dev| {
            fast_matches_slow(dev, false).unwrap_or_else(|e| panic!("{e}"));
        });
    }

    /// The oracle's own control: an expectation with one flipped bit is caught.
    #[test]
    fn the_byte_identity_oracle_rejects_a_flipped_bit() {
        harness::in_device(|dev| {
            let e = fast_matches_slow(dev, true).expect_err("a corrupted expectation passed");
            assert!(e.contains("not what the channel holds"), "{e}");
        });
    }

    /// The negative control: a second request that does not write its new
    /// destination reads into the first request's, and the oracle says so.
    #[test]
    fn a_request_that_skips_a_changed_register_is_caught() {
        harness::in_device(|dev| {
            for initiator1 in [false, true] {
                let e = persistence_case(dev, Skip::SecondDestination, initiator1)
                    .expect_err("the oracle accepted a request that skipped RET_ADDR_LO");
                assert!(e.contains("destination"), "{e}");
            }
        });
    }
}

/// Silicon: written, NOT run. Risk class: documented (BlackholeA0
/// `MemoryMap.md`), UNVERIFIED on silicon that the registers keep their values
/// between requests, **NoC-hang class** (unicast L1 reads from the surviving
/// neighbour (4, 4), claimed through `harness::tile`, which checks it against the
/// chip's ARC grid; a request with a stale or half-written register could in
/// principle address a tile the grid does not have, so the probe reads L1 only
/// and every address is below 1.5 MiB).
///
/// Order: `silicon_x6_minimal_persistence_probe` alone (inside `survives`),
/// then `silicon_x6_initiator1_persistence_probe`, then the mover gates of this
/// file. Expected: all four destinations hold the bytes
/// the changed registers named, and the printed snapshots show every address,
/// coordinate, tag and length word keeping the value last written (`NOC_CTRL`
/// excepted). The mutant `silicon_x6_rejects_a_skipped_destination` must fail the
/// oracle on silicon too.
#[cfg(feature = "silicon")]
mod silicon {
    use super::*;

    /// Run alone, first. Initiator 0, the one the mover already uses; the first
    /// request writes every register, the later ones only the changed words.
    #[test]
    fn silicon_x6_minimal_persistence_probe() {
        harness::assert_on_silicon();
        assert!(harness::survives(|dev| {
            persistence_case(dev, Skip::None, false).unwrap();
        }));
    }

    /// Run alone, second: the same on initiator 1 (`NIU_BASE + 0x800`), which
    /// the fast path would use. Documented as functionally identical to
    /// initiator 0; not yet driven by any code in this repository on silicon.
    #[test]
    fn silicon_x6_initiator1_persistence_probe() {
        harness::assert_on_silicon();
        assert!(harness::survives(|dev| {
            persistence_case(dev, Skip::None, true).unwrap();
        }));
    }

    /// Run third, alone, after both persistence probes passed: the mover's fast
    /// read path against the slow one over every shape of `cases`, inside
    /// `survives`. Risk class: the same NoC #0 reads, endpoints and ports as the
    /// established path (documented, measured); new only in that the requests go
    /// through initiator 1.
    #[test]
    fn silicon_x6_fast_read_path_matches_the_slow_one() {
        harness::assert_on_silicon();
        assert!(harness::survives(|dev| {
            fast_matches_slow(dev, false).unwrap();
        }));
    }

    #[test]
    fn silicon_x6_oracle_rejects_a_flipped_bit() {
        harness::assert_on_silicon();
        harness::in_device(|dev| {
            assert!(fast_matches_slow(dev, true).is_err());
        });
    }

    #[test]
    fn silicon_x6_rejects_a_skipped_destination() {
        harness::assert_on_silicon();
        harness::in_device(|dev| {
            assert!(persistence_case(dev, Skip::SecondDestination, true).is_err());
        });
    }
}
