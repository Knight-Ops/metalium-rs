//! Ethernet throughput between the two cards, device-timed by each E1's own
//! cycle counter (`eth::mover::TRACE`; gated alone by `silicon_eth_clock`).
//!
//! `cargo xtask bench --filter silicon_bench_eth`
//!
//! The cards share one QSFP-DD cable: two links, each a 400 GbE Ethernet tile
//! pair, so 800 GbE between them is both links at once. One link's numbers
//! are held against 400 GbE, both links' against 800 GbE -- raw line rates,
//! before any framing.
//!
//! The two ends' counters are not one clock, so each span is on one end:
//! the sender's from pickup to the acknowledgement arriving, the receiver's
//! from the record arriving to its acknowledgement leaving. Their difference
//! is the two flights across the cable.

#![cfg(feature = "silicon")]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use tt_device::tlb::WindowKind;
use tt_device::Window;
use tt_isa::eth::mover::{event as ev, MAX_LEN};
use tt_kernels::link::{discover, Dest, Dir, EthEvent, Link, Mover, Source};
use tt_tests::bench::{pattern, report, report_rate, with_cards, Conditions, Peak, Stats, REPS};
use tt_tests::harness::{tile, Dev};

const SIZES: [u32; 6] = [16, 4 << 10, 16 << 10, 32 << 10, 64 << 10, 128 << 10];
/// Sends per stream measurement.
const STREAM: usize = 32;

/// The cycle of `event` for `seq` in `trace`.
fn at(trace: &[EthEvent], event: u32, seq: u32) -> Option<u32> {
    trace
        .iter()
        .find(|e| e.event == event && e.seq == seq)
        .map(|e| e.cycles)
}

fn span(trace: &[EthEvent], from: u32, to: u32, seq: u32) -> Option<f64> {
    Some(at(trace, to, seq)?.wrapping_sub(at(trace, from, seq)?) as f64)
}

struct Pair<'a> {
    a: &'a mut Dev<'static>,
    b: &'a mut Dev<'static>,
    wa: Window,
    wb: Window,
    links: Vec<Link>,
    /// Each E1's counter rate, MHz (measured, `silicon_eth_clock`).
    mhz: f64,
}

fn pair(f: impl FnOnce(&mut Pair<'_>)) {
    with_cards(|a, b| {
        let c = Conditions::measure(a, 0, tt_tests::harness::tensix_tile());
        c.print();
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        let (ga, gb) = (a.ethernet_grid(&wa).unwrap(), b.ethernet_grid(&wb).unwrap());
        let links = discover(a, &wa, &ga, b, &wb, &gb).unwrap();
        for l in &links {
            let (sa, sb) = (
                a.eth_link_state(&wa, l.a).unwrap(),
                b.eth_link_state(&wb, l.b).unwrap(),
            );
            println!(
                "MEASURE link X {} <-> X {}: port {:?} / {:?}, train status {:#x} / {:#x}",
                l.a.x(),
                l.b.x(),
                sa.port,
                sb.port,
                sa.train_status,
                sb.train_status
            );
        }
        println!("MEASURE {} links between the cards", links.len());
        let mut p = Pair {
            a,
            b,
            wa,
            wb,
            links,
            mhz: 0.0,
        };
        p.mhz = e1_mhz(&mut p);
        report("E1 cycle counter rate", "MHz", "host", Stats::of([p.mhz]));
        f(&mut p);
    });
}

/// The sending E1's counter against the host, over two sends 50 ms apart.
fn e1_mhz(p: &mut Pair<'_>) -> f64 {
    let l = p.links[0];
    let mut m = Mover::start(p.a, &p.wa, p.b, &p.wb, l, tt_firmware_images::ETH_E1).unwrap();
    m.set_trace(p.a, &p.wa, l.a, true).unwrap();
    m.stage(p.a, &p.wa, Dir::AToB, &[0u8; 16]).unwrap();
    let h0 = Instant::now();
    m.send(p.a, &p.wa, Dir::AToB, Source::Staged, Dest::Landed, 16)
        .unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let h1 = Instant::now();
    m.send(p.a, &p.wa, Dir::AToB, Source::Staged, Dest::Landed, 16)
        .unwrap();
    let t = m.take_trace(p.a, &p.wa, l.a).unwrap();
    m.set_trace(p.a, &p.wa, l.a, false).unwrap();
    park(p, l);
    let c = at(&t, ev::SEND_PICKUP, 2)
        .unwrap()
        .wrapping_sub(at(&t, ev::SEND_PICKUP, 1).unwrap());
    c as f64 / ((h1 - h0).as_secs_f64() * 1e6)
}

fn park(p: &mut Pair<'_>, l: Link) {
    p.a.park_e1(&p.wa, l.a).unwrap();
    p.b.park_e1(&p.wb, l.b).unwrap();
}

/// One send's steps on each end, from both ends' traces: cycles, by name.
fn steps(sent: &[EthEvent], got: &[EthEvent], seq: u32, tensix: bool) -> Vec<(&'static str, f64)> {
    let s = |a, b| span(sent, a, b, seq).expect("a sender step is missing");
    let r = |a, b| span(got, a, b, seq).expect("a receiver step is missing");
    let data_from = if tensix {
        ev::NOC_IN_DONE
    } else {
        ev::SEND_PICKUP
    };
    let ack_from = if tensix { ev::NOC_OUT_DONE } else { ev::LANDED };
    let mut v = Vec::new();
    if tensix {
        v.push((
            "1 send: NoC read Tensix -> TX stage",
            s(ev::SEND_PICKUP, ev::NOC_IN_DONE),
        ));
    }
    v.extend([
        // Accepted, not delivered: the TX queue takes commands faster than
        // the wire carries them. Delivery is in step 4, behind the record.
        (
            "2 send: data commands taken by the TX queue",
            s(data_from, ev::DATA_SENT),
        ),
        (
            "3 send: record onto the link",
            s(ev::DATA_SENT, ev::RECORD_SENT),
        ),
        (
            "4 send: record sent -> ack seen",
            s(ev::RECORD_SENT, ev::ACK_SEEN),
        ),
        (
            "5 recv: record seen -> landed",
            r(ev::RECORD_SEEN, ev::LANDED),
        ),
    ]);
    if tensix {
        v.push((
            "6 recv: NoC write RX land -> Tensix",
            r(ev::LANDED, ev::NOC_OUT_DONE),
        ));
    }
    let recv = r(ev::RECORD_SEEN, ev::ACK_SENT);
    v.extend([
        ("7 recv: ack onto the link", r(ack_from, ev::ACK_SENT)),
        ("8 recv: record seen -> ack sent", recv),
        (
            "9 both flights: record + ack across the cable",
            s(ev::RECORD_SENT, ev::ACK_SEEN) - recv,
        ),
        (
            "A send: whole, pickup -> ack seen",
            s(ev::SEND_PICKUP, ev::ACK_SEEN),
        ),
    ]);
    v
}

/// One link, one transfer at a time: each step of a send, by size, staged ->
/// landed (the link and the mover's protocol alone) and Tensix -> Tensix
/// (with a NoC hop at each end). The data step is the link's own rate.
#[test]
#[ignore = "benchmark"]
fn mover_breakdown() {
    pair(|p| {
        let l = p.links[0];
        let mut m = Mover::start(p.a, &p.wa, p.b, &p.wb, l, tt_firmware_images::ETH_E1).unwrap();
        m.set_trace(p.a, &p.wa, l.a, true).unwrap();
        m.set_trace(p.b, &p.wb, l.b, true).unwrap();
        let (ta, tb) = (tile(p.a, 3, 4), tile(p.b, 4, 4));
        let link = Conditions::eth_link();
        for tensix in [false, true] {
            let path = if tensix {
                "tensix->tensix"
            } else {
                "staged->landed"
            };
            let mut whole: Vec<(u32, f64)> = Vec::new();
            for size in SIZES {
                let data = pattern(size as usize, size);
                if tensix {
                    p.a.write(&p.wa, ta, 0x8_0000, &data).unwrap();
                    let _ = p.a.read32(&p.wa, ta, 0x8_0000 + size as u64 - 4).unwrap();
                } else {
                    m.stage(p.a, &p.wa, Dir::AToB, &data).unwrap();
                }
                let (src, dst) = if tensix {
                    (Source::Tensix(ta, 0x8_0000), Dest::Tensix(tb, 0x8_0000))
                } else {
                    (Source::Staged, Dest::Landed)
                };
                let mut by: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
                let mut host = Vec::new();
                for rep in 0..=REPS {
                    let h0 = Instant::now();
                    let seq = m.post(p.a, &p.wa, Dir::AToB, src, dst, size).unwrap();
                    m.wait(p.a, &p.wa, Dir::AToB, seq).unwrap();
                    let h = h0.elapsed();
                    let sent = m.take_trace(p.a, &p.wa, l.a).unwrap();
                    let got = m.take_trace(p.b, &p.wb, l.b).unwrap();
                    if rep == 0 {
                        continue;
                    }
                    host.push(h);
                    for (k, v) in steps(&sent, &got, seq, tensix) {
                        by.entry(k).or_default().push(v);
                    }
                }
                let us = |v: &Vec<f64>| Stats::of(v.iter().copied()).map(|c| c / p.mhz);
                whole.push((size, us(&by["A send: whole, pickup -> ack seen"]).median));
                for (k, v) in &by {
                    let key = format!("eth 1 link {path} {size:>6} B: {}", &k[2..]);
                    if k.starts_with('A') {
                        report_rate(&key, "device", size as f64, us(v), Some(&link));
                    } else {
                        report(&key, "us", "device", us(v));
                    }
                }
                report_rate(
                    &format!("eth 1 link {path} {size:>6} B: host send -> ack"),
                    "host",
                    size as f64,
                    Stats::of_durations(host),
                    Some(&link),
                );
                // The bytes arrived.
                let mut back = vec![0u8; size as usize];
                if tensix {
                    p.b.read(&p.wb, tb, 0x8_0000, &mut back).unwrap();
                } else {
                    m.landed(p.b, &p.wb, Dir::AToB, &mut back).unwrap();
                }
                assert!(back == data, "{path} {size} B did not arrive intact");
            }
            // What each further byte costs: the slope of the whole send
            // between sizes, the fixed per-transfer cost cancelled. Staged,
            // that is the link's own rate.
            for w in whole.windows(2).filter(|w| w[0].0 >= 16 << 10) {
                let (b, us) = ((w[1].0 - w[0].0) as f64, w[1].1 - w[0].1);
                report_rate(
                    &format!(
                        "eth 1 link {path}: per-byte rate, {} -> {} B",
                        w[0].0, w[1].0
                    ),
                    "device",
                    b,
                    Stats::of([us]),
                    Some(&link),
                );
            }
            // And the fixed cost: the whole send extrapolated to no bytes.
            let (s1, s2) = (whole[whole.len() - 2], whole[whole.len() - 1]);
            let per_byte = (s2.1 - s1.1) / (s2.0 - s1.0) as f64;
            report(
                &format!("eth 1 link {path}: fixed cost per transfer (intercept)"),
                "us",
                "device",
                Stats::of([s2.1 - per_byte * s2.0 as f64]),
            );
        }
        park(p, l);
    });
}

/// `n` streams of [`STREAM`] back-to-back `MAX_LEN` sends at once, each the
/// host posting the next as soon as the last was acknowledged. Each stream's
/// rate from its sender's trace (first pickup to last acknowledgement), and
/// all of them together from the host's clock.
fn streams(p: &mut Pair<'_>, plan: &[(usize, Dir)], label: &str) {
    let len = MAX_LEN;
    let data = pattern(len as usize, 11);
    let mut movers: Vec<Mover> = Vec::new();
    let mut used: Vec<usize> = plan.iter().map(|s| s.0).collect();
    used.dedup();
    for &li in &used {
        let l = p.links[li];
        let m = Mover::start(p.a, &p.wa, p.b, &p.wb, l, tt_firmware_images::ETH_E1).unwrap();
        m.set_trace(p.a, &p.wa, l.a, true).unwrap();
        m.set_trace(p.b, &p.wb, l.b, true).unwrap();
        m.stage(p.a, &p.wa, Dir::AToB, &data).unwrap();
        m.stage(p.b, &p.wb, Dir::BToA, &data).unwrap();
        movers.push(m);
    }
    let mi = |li: usize| used.iter().position(|&u| u == li).unwrap();
    let mut host = Vec::new();
    let mut per_stream: Vec<Vec<f64>> = vec![Vec::new(); plan.len()];
    for rep in 0..=REPS {
        let h0 = Instant::now();
        let mut seqs = vec![0u32; plan.len()];
        for _ in 0..STREAM {
            for (k, &(li, dir)) in plan.iter().enumerate() {
                let m = &mut movers[mi(li)];
                let (d, w) = match dir {
                    Dir::AToB => (&mut *p.a, &p.wa),
                    Dir::BToA => (&mut *p.b, &p.wb),
                };
                seqs[k] = m
                    .post(d, w, dir, Source::Staged, Dest::Landed, len)
                    .unwrap();
            }
            for (k, &(li, dir)) in plan.iter().enumerate() {
                let m = &movers[mi(li)];
                let (d, w) = match dir {
                    Dir::AToB => (&mut *p.a, &p.wa),
                    Dir::BToA => (&mut *p.b, &p.wb),
                };
                m.wait(d, w, dir, seqs[k]).unwrap();
            }
        }
        let wall = h0.elapsed();
        // Each sender's trace: its stream's sends are the last STREAM seqs.
        for (k, &(li, dir)) in plan.iter().enumerate() {
            let l = p.links[li];
            let m = &movers[mi(li)];
            let (d, w, end) = match dir {
                Dir::AToB => (&mut *p.a, &p.wa, l.a),
                Dir::BToA => (&mut *p.b, &p.wb, l.b),
            };
            let t = m.take_trace(d, w, end).unwrap();
            let last = seqs[k];
            let first = last + 1 - STREAM as u32;
            let c = at(&t, ev::ACK_SEEN, last)
                .unwrap()
                .wrapping_sub(at(&t, ev::SEND_PICKUP, first).unwrap());
            if rep > 0 {
                per_stream[k].push(c as f64 / p.mhz);
            }
        }
        // Each receiver's ring too, so it never fills.
        for (k, &(li, dir)) in plan.iter().enumerate() {
            let _ = k;
            let l = p.links[li];
            let m = &movers[mi(li)];
            let (d, w, end) = match dir {
                Dir::AToB => (&mut *p.b, &p.wb, l.b),
                Dir::BToA => (&mut *p.a, &p.wa, l.a),
            };
            m.take_trace(d, w, end).unwrap();
        }
        if rep > 0 {
            host.push(wall);
        }
    }
    let bytes = (len as usize * STREAM) as f64;
    let link = Conditions::eth_link();
    let port = Conditions::eth_port();
    let mut sum = 0.0;
    for (k, &(li, dir)) in plan.iter().enumerate() {
        let s = Stats::of(per_stream[k].iter().copied());
        sum += bytes / (s.median * 1e-6);
        let l = p.links[li];
        report_rate(
            &format!(
                "eth {label}: link X {} {dir:?} stream of {STREAM} x {len} B",
                l.a.x()
            ),
            "device",
            bytes,
            s,
            Some(&link),
        );
    }
    // A port is full duplex: 800 GbE each way. The ceiling is one link's
    // rate for every (link, direction) a stream uses.
    let mut lanes: Vec<(usize, bool)> = plan.iter().map(|&(l, d)| (l, d == Dir::AToB)).collect();
    lanes.sort_unstable();
    lanes.dedup();
    let both = Peak {
        name: if used.len() > 1 {
            "eth_800g x dirs"
        } else {
            "eth_link x dirs"
        },
        bytes_per_sec: link.bytes_per_sec * lanes.len() as f64,
        source: format!("{} (link, direction) pairs x {}", lanes.len(), link.source),
    };
    let peak = if lanes.len() == 1 {
        &link
    } else if lanes.len() == 2 && used.len() == 2 {
        &port
    } else {
        &both
    };
    let total = bytes * plan.len() as f64;
    report_rate(
        &format!("eth {label}: all streams (host-timed)"),
        "host",
        total,
        Stats::of_durations(host),
        Some(peak),
    );
    println!(
        "MEASURE [device] eth {label}: sum of stream rates {:.2} GB/s = {:.1}% of {} {:.1} GB/s",
        sum / 1e9,
        100.0 * sum / peak.bytes_per_sec,
        peak.name,
        peak.bytes_per_sec / 1e9
    );
    // The bytes arrived, on every receiving end.
    for &(li, dir) in plan {
        let m = &movers[mi(li)];
        let mut back = vec![0u8; len as usize];
        match dir {
            Dir::AToB => m.landed(p.b, &p.wb, dir, &mut back).unwrap(),
            Dir::BToA => m.landed(p.a, &p.wa, dir, &mut back).unwrap(),
        }
        assert!(
            back == data,
            "{label}: link {li} {dir:?} did not arrive intact"
        );
    }
    for &li in &used {
        park(p, p.links[li]);
    }
}

/// One link, one direction, back to back: today's protocol's steady state.
#[test]
#[ignore = "benchmark"]
fn mover_stream() {
    pair(|p| streams(p, &[(0, Dir::AToB)], "1 link 1 way"));
}

/// The 800G question: both links at once, one way and then both ways.
#[test]
#[ignore = "benchmark"]
fn dual_link_800g() {
    pair(|p| {
        assert!(
            p.links.len() >= 2,
            "800 GbE is two links; found {}",
            p.links.len()
        );
        streams(p, &[(0, Dir::AToB), (0, Dir::BToA)], "1 link both ways");
        streams(p, &[(0, Dir::AToB), (1, Dir::AToB)], "2 links 1 way");
        streams(
            p,
            &[
                (0, Dir::AToB),
                (0, Dir::BToA),
                (1, Dir::AToB),
                (1, Dir::BToA),
            ],
            "2 links both ways",
        );
    });
}
