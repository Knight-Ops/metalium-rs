//! Exploratory, Phase 8 spike S1-S3: what does each Ethernet tile of `bh_x2` say
//! about itself, its link, and who owns its queues?
//!
//! Every read is one `fork_scope`, because an access ttsim does not decode is
//! fatal and we want the rest of the table. Nothing here writes.
//!
//! Run with `cargo test -p tt-tests --test probe_eth -- --ignored --nocapture`.

// Drives the dual-chip simulator directly.
#![cfg(not(feature = "silicon"))]

use std::fs;
use std::path::PathBuf;

use tt_device::{tlb::WindowKind, Device};
use tt_isa::noc::{niu, Noc0, NocCoord};
use tt_ttsim::{fork_scope, Simulator};

/// Candidate Ethernet coordinates: row 1, skipping the non-memory and DRAM
/// columns (`ethdump.c:462-463`).
fn candidates() -> impl Iterator<Item = u8> {
    (1..=16u8).filter(|x| *x != 8 && *x != 9)
}

/// One 32-bit read on `chip` at `(x, 1)`, in its own process.
fn read(chip: usize, x: u8, addr: u64, dir: &std::path::Path) -> Option<u32> {
    let out = dir.join("r");
    let _ = fs::remove_file(&out);
    let _ = fork_scope(|| {
        let mut sim = Simulator::open_path(tt_ttsim::x2_lib_path()).unwrap();
        let t = sim.transports().into_iter().nth(chip).unwrap();
        let mut dev = Device::open(t).unwrap();
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let c = NocCoord::<Noc0>::new(x, 1).unwrap();
        if let Ok(v) = dev.read32(&w, c, addr) {
            fs::write(&out, v.to_string()).unwrap();
        }
    });
    fs::read_to_string(&out).ok().and_then(|s| s.parse().ok())
}

fn show(v: Option<u32>) -> String {
    v.map_or("  (fatal) ".into(), |v| format!("{v:#010x}"))
}

const BOOT_RESULTS: u64 = 0x7_CC00;
const TXQ: u64 = 0xFFB9_0000;
const RXQ: u64 = 0xFFB9_4000;

#[test]
#[ignore = "exploratory -- forks per register"]
fn survey_ethernet_tiles() {
    let dir = PathBuf::from(std::env::var_os("CARGO_TARGET_TMPDIR").unwrap_or("/tmp".into()))
        .join("eth-survey");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let regs: &[(&str, u64)] = &[
        ("NIU_CFG_0", niu::NOC0_BASE + niu::NIU_CFG_0),
        ("NOC_ID_LOGICAL", niu::NOC0_BASE + niu::NOC_ID_LOGICAL),
        ("boot[0]", BOOT_RESULTS),
        ("port_status", BOOT_RESULTS + 4),
        ("train_status", BOOT_RESULTS + 8),
        ("boot[243]", BOOT_RESULTS + 243 * 4),
        ("boot[244]", BOOT_RESULTS + 244 * 4),
        ("boot[245]", BOOT_RESULTS + 245 * 4),
        ("boot[246]", BOOT_RESULTS + 246 * 4),
        ("boot[247]", BOOT_RESULTS + 247 * 4),
        ("TXQ0_CTRL", TXQ),
        ("TXQ1_CTRL", TXQ + 0x1000),
        ("TXQ2_CTRL", TXQ + 0x2000),
        ("RXQ0_CTRL", RXQ),
        ("RXQ1_CTRL", RXQ + 0x1000),
        ("RXQ2_CTRL", RXQ + 0x2000),
        ("SOFT_RESET_0", 0xFFB1_21B0),
    ];
    for chip in 0..2 {
        for x in candidates() {
            let row: Vec<String> = regs
                .iter()
                .map(|(n, a)| format!("{n}={}", show(read(chip, x, *a, &dir))))
                .collect();
            println!("chip {chip} x={x:2}: {}", row.join(" "));
        }
    }
}

#[test]
#[ignore = "exploratory -- forks per register"]
fn survey_txq_config_on_live_links() {
    let dir = PathBuf::from(std::env::var_os("CARGO_TARGET_TMPDIR").unwrap_or("/tmp".into()))
        .join("eth-txq");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mut regs: Vec<(String, u64)> = [
        ("MAX_PKT", 0x0C),
        ("SEL_SW", 0x80),
        ("SEL_HW", 0x84),
        ("SEQ_TIMEOUT", 0x48),
        ("SEQ_UPD_TIMEOUT", 0x4C),
        ("STATUS", 0x08),
        ("TRANSFER_CNT", 0x30),
    ]
    .iter()
    .flat_map(|(n, o)| (0..2u64).map(move |q| (format!("TXQ{q}_{n}"), TXQ + q * 0x1000 + o)))
    .collect();
    for e in 0..10u64 {
        let b = 0xFFB9_8200 + e * 0x80;
        regs.push((format!("HDR{e}_DA"), b + 0x18));
        regs.push((format!("HDR{e}_DA_HI"), b + 0x1C));
        regs.push((format!("HDR{e}_SA"), b + 0x10));
        regs.push((format!("HDR{e}_SA_HI"), b + 0x14));
    }
    for (chip, x) in [(0usize, 2u8), (0, 15), (1, 5), (1, 12)] {
        let row: Vec<String> = regs
            .iter()
            .map(|(n, a)| format!("{n}={}", show(read(chip, x, *a, &dir))))
            .collect();
        println!("chip {chip} x={x:2}:\n  {}", row.join("\n  "));
    }
}

/// S4: the host drives a TT-link L1 write on chip 0's TXQ0 with no firmware of
/// ours anywhere, then looks for the bytes on each live chip-1 tile.
#[test]
#[ignore = "exploratory"]
fn host_driven_tt_link_l1_write() {
    const BUF: u64 = 0x6_0000;
    const LEN: usize = 4096;
    let (src_x, dsts) = (
        std::env::var("SRC_X").map_or(2u8, |s| s.parse().unwrap()),
        [5u8, 12],
    );
    let src_chip = std::env::var("SRC_CHIP").map_or(0usize, |s| s.parse().unwrap());
    let dsts = if src_chip == 0 { dsts } else { [2u8, 15] };
    let r = fork_scope(|| {
        let mut sim = Simulator::open_path(tt_ttsim::x2_lib_path()).unwrap();
        let mut ts = sim.transports().into_iter();
        let mut a = Device::open(ts.next().unwrap()).unwrap();
        let mut b = Device::open(ts.next().unwrap()).unwrap();
        if src_chip == 1 {
            std::mem::swap(&mut a, &mut b);
        }
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let src = NocCoord::<Noc0>::new(src_x, 1).unwrap();
        let data: Vec<u8> = (0..LEN).map(|i| (i * 7 + 3) as u8).collect();
        a.write(&wa, src, BUF, &data).unwrap();
        for x in dsts {
            b.write(
                &wb,
                NocCoord::<Noc0>::new(x, 1).unwrap(),
                BUF,
                &vec![0xEE; LEN],
            )
            .unwrap();
        }
        let q = TXQ + 0x1000 * std::env::var("QUEUE").map_or(0u64, |s| s.parse().unwrap());
        a.write32(&wa, src, q + 0x14, BUF as u32).unwrap();
        a.write32(&wa, src, q + 0x18, LEN as u32).unwrap();
        a.write32(&wa, src, q + 0x1C, BUF as u32).unwrap();
        println!("wrote addr/size/dest");
        a.write32(&wa, src, q + 0x04, 2).unwrap();
        println!("wrote CMD=2");
        for i in 0..100 {
            a.tick(1000);
            let st = a.read32(&wa, src, q + 0x08).unwrap();
            if st & (1 << 16) == 0 {
                println!("STATUS clear after {i} polls: {st:#x}");
                break;
            }
        }
        a.tick(100_000);
        for x in dsts {
            let mut got = vec![0u8; LEN];
            b.read(&wb, NocCoord::<Noc0>::new(x, 1).unwrap(), BUF, &mut got)
                .unwrap();
            let same = got.iter().zip(&data).filter(|(g, d)| g == d).count();
            println!(
                "peer x={x}: {same}/{LEN} bytes match, first {:02x?}",
                &got[..8]
            );
        }
    });
    println!("child: {r:?}");
}

/// `bh_x4`'s link map: every Up tile of every chip, and where a TT-link write
/// from it lands. Prints `(chip, x) -> (chip, x)` pairs.
#[test]
#[ignore = "exploratory"]
fn bh_x4_link_map() {
    use tt_isa::eth::{self, Ethernet, PortStatus};
    let r = fork_scope(|| {
        let mut sim = Simulator::open_path(tt_ttsim::x4_lib_path()).unwrap();
        println!("chips: {}", sim.chip_count());
        let mut devs: Vec<_> = sim
            .transports()
            .into_iter()
            .map(|t| Device::open(t).unwrap())
            .collect();
        let ws: Vec<_> = devs
            .iter_mut()
            .map(|d| d.alloc_window(WindowKind::TwoMib).unwrap())
            .collect();
        let g = Ethernet::FULL;
        let mut up = Vec::new();
        for (c, d) in devs.iter_mut().enumerate() {
            for x in g.tiles() {
                if d.eth_link_state(&ws[c], g.tile(x).unwrap()).unwrap().port == PortStatus::Up {
                    up.push((c, x));
                }
            }
        }
        println!("up: {up:?}");
        let buf = eth::BUFFERS.start;
        for &(c, x) in &up {
            for &(c2, x2) in &up {
                devs[c2]
                    .eth_write(&ws[c2], g.tile(x2).unwrap(), buf, &[0u8; 64])
                    .unwrap();
            }
            let marker: Vec<u8> = (0..64).map(|i| i as u8 ^ (c as u8 * 16 + x)).collect();
            devs[c]
                .eth_write(&ws[c], g.tile(x).unwrap(), buf + 0x1000, &marker)
                .unwrap();
            devs[c]
                .eth_tt_link_write(&ws[c], g.tile(x).unwrap(), buf + 0x1000, buf, 64)
                .unwrap()
                .unwrap();
            devs[0].tick(100_000);
            for &(c2, x2) in &up {
                let mut got = [0u8; 64];
                devs[c2]
                    .eth_read(&ws[c2], g.tile(x2).unwrap(), buf, &mut got)
                    .unwrap();
                if got[..] == marker[..] {
                    println!("link ({c},{x}) -> ({c2},{x2})");
                }
            }
        }
    });
    println!("child: {r:?}");
}

/// Does ttsim decode the RX queue counters the mover's landing check reads?
#[test]
#[ignore = "exploratory"]
fn rxq_counters_on_ttsim() {
    let dir = PathBuf::from(std::env::var_os("CARGO_TARGET_TMPDIR").unwrap_or("/tmp".into()))
        .join("eth-rxq");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    for (n, o) in [
        ("BYTE_CNT", 0x04u64),
        ("PKT_START", 0x24),
        ("PKT_END", 0x28),
        ("DROP", 0x4C),
        ("OUTSTANDING_WR_CNT", 0x50),
    ] {
        println!("RXQ2_{n} = {}", show(read(1, 12, RXQ + 0x2000 + o, &dir)));
    }
}
