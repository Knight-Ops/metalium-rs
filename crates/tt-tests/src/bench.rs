//! What the benchmarks share: statistics, the peaks a number is held against,
//! the conditions it was measured under, and one output format.
//!
//! Every benchmark prints a human `MEASURE` line, as the Phase 9 ones always
//! have, and a `BENCH {json}` line beside it that `cargo xtask bench` collects
//! into `target/silicon/bench/`. A throughput carries the peak it is measured
//! against and the fraction of it reached; a peak carries where it came from.
//!
//! Device-timed numbers are counter cycles. They are converted with the rate
//! [`Conditions::measure`] measured for the counter against the host's clock,
//! which is what the counter actually did, not with the AICLK telemetry claims
//! (printed beside it, as a cross-check).

use std::fmt::Write as _;
use std::time::Duration;

/// Repetitions per measurement, after one warm-up.
pub const REPS: usize = 9;

pub fn mbps(bytes: usize, t: Duration) -> f64 {
    bytes as f64 / t.as_secs_f64() / 1e6
}

pub fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

/// Xorshift bytes: no two 4-byte words of a buffer alike in practice, so a move
/// that lands at the wrong offset does not compare equal.
pub fn pattern(len: usize, seed: u32) -> Vec<u8> {
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

/// The median and spread of a set of samples.
#[derive(Copy, Clone, Debug)]
pub struct Stats {
    pub n: usize,
    pub median: f64,
    pub p10: f64,
    pub p90: f64,
}

impl Stats {
    pub fn of(samples: impl IntoIterator<Item = f64>) -> Stats {
        let mut v: Vec<f64> = samples.into_iter().collect();
        assert!(!v.is_empty(), "no samples");
        v.sort_by(|a, b| a.partial_cmp(b).expect("a sample is NaN"));
        let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
        Stats {
            n: v.len(),
            median: at(0.5),
            p10: at(0.1),
            p90: at(0.9),
        }
    }

    pub fn of_durations(samples: impl IntoIterator<Item = Duration>) -> Stats {
        Stats::of(samples.into_iter().map(|d| d.as_secs_f64() * 1e6))
    }

    /// Each sample mapped through `f` (cycles to microseconds, say). Order is
    /// kept for a monotonic `f`; a decreasing one swaps the percentiles.
    pub fn map(self, f: impl Fn(f64) -> f64) -> Stats {
        let (a, b) = (f(self.p10), f(self.p90));
        Stats {
            n: self.n,
            median: f(self.median),
            p10: a.min(b),
            p90: a.max(b),
        }
    }
}

/// A theoretical ceiling, in bytes per second, and where the figure comes from.
#[derive(Clone, Debug)]
pub struct Peak {
    pub name: &'static str,
    pub bytes_per_sec: f64,
    pub source: String,
}

/// What a run measured under. Printed once per benchmark, and attached to every
/// `BENCH` line so a number is never separated from its clock.
#[derive(Clone, Debug, Default)]
pub struct Conditions {
    pub card: u16,
    /// AICLK as ARC telemetry reports it (tag 14), MHz.
    pub aiclk_mhz: Option<u32>,
    /// The tile counter's rate, measured against the host's clock, MHz.
    pub counter_mhz: f64,
    /// GDDR data rate (tag 23), MT/s.
    pub gddr_mts: Option<u32>,
    /// Enabled GDDR channels (tag 36's set bits).
    pub gddr_channels: Option<u32>,
    pub sha: String,
    pub release: bool,
}

impl Conditions {
    /// The tile counter's rate in cycles per microsecond.
    pub fn cycles_per_us(&self) -> f64 {
        self.counter_mhz
    }

    pub fn cycles_to_us(&self, cycles: f64) -> f64 {
        cycles / self.counter_mhz
    }

    /// Bytes per second for `bytes` moved in `cycles` of the tile counter.
    pub fn rate(&self, bytes: f64, cycles: f64) -> f64 {
        bytes / (cycles / (self.counter_mhz * 1e6))
    }

    /// One NIU's link into its router: one 512-bit flit per NoC cycle
    /// (`BlackholeA0/NoC/README.md:64`), at AICLK -- the NoC runs at 1.35 GHz
    /// (`NoC/README.md:44`), and 1350 MHz is the busy AICLK here (divergence
    /// row N). Raw: a 16 KiB packet's header flit costs 1/257 of it.
    pub fn noc_link(&self) -> Peak {
        let mhz = self.aiclk_mhz.map(f64::from).unwrap_or(self.counter_mhz);
        Peak {
            name: "noc_link",
            bytes_per_sec: 64.0 * mhz * 1e6,
            source: format!("64 B/flit x 1 flit/cycle x {mhz:.0} MHz (NoC/README.md:64)"),
        }
    }

    /// [`Conditions::noc_link`] in both directions at once: an NIU's link to
    /// its router carries a flit a cycle each way, so a tile reading and
    /// writing together can move twice what either alone does.
    pub fn noc_link_both(&self) -> Peak {
        let one = self.noc_link();
        Peak {
            name: "noc_link_both",
            bytes_per_sec: 2.0 * one.bytes_per_sec,
            source: format!("2 directions x {}", one.source),
        }
    }

    /// One GDDR6 channel. 32 GiB over 8 channels is one 32 Gb device each,
    /// and a GDDR6 device is x32 (two x16 channels): 32 pins at the data rate.
    /// The x32 is reasoned, not published for Blackhole.
    pub fn gddr_channel(&self) -> Option<Peak> {
        let mts = f64::from(self.gddr_mts?);
        Some(Peak {
            name: "gddr_channel",
            bytes_per_sec: mts * 1e6 * 32.0 / 8.0,
            source: format!(
                "{mts:.0} MT/s (tag 23) x 32 pins / 8 (one x32 device per channel, reasoned)"
            ),
        })
    }

    /// Every enabled channel at once.
    pub fn gddr_card(&self) -> Option<Peak> {
        let ch = self.gddr_channel()?;
        let n = f64::from(self.gddr_channels?);
        Some(Peak {
            name: "gddr_card",
            bytes_per_sec: ch.bytes_per_sec * n,
            source: format!("{n:.0} channels (tag 36) x {}", ch.source),
        })
    }

    /// One Ethernet tile's port: 400 GbE (`BlackholeA0/README.md:7`), raw --
    /// before any framing or TT-link protocol overhead.
    pub fn eth_link() -> Peak {
        Peak {
            name: "eth_link",
            bytes_per_sec: 400e9 / 8.0,
            source: "400 GbE per Ethernet tile (BlackholeA0/README.md:7)".into(),
        }
    }

    /// One QSFP-DD port: its pair of Ethernet tiles, 800 GbE.
    pub fn eth_port() -> Peak {
        Peak {
            name: "eth_800g",
            bytes_per_sec: 800e9 / 8.0,
            source: "2 Ethernet tiles per QSFP-DD port (BlackholeA0/README.md:7)".into(),
        }
    }

    pub fn print(&self) {
        println!(
            "MEASURE conditions: card {} | AICLK {:?} MHz (telemetry) | counter {:.1} MHz (measured) | GDDR {:?} MT/s x {:?} ch | {} | {}",
            self.card,
            self.aiclk_mhz,
            self.counter_mhz,
            self.gddr_mts,
            self.gddr_channels,
            if self.release { "release" } else { "DEBUG -- not a number to quote" },
            self.sha
        );
        let mut j = String::from("{\"kind\":\"conditions\"");
        let _ = write!(
            j,
            ",\"card\":{},\"aiclk_mhz\":{},\"counter_mhz\":{:.3},\"gddr_mts\":{},\"gddr_channels\":{},\"sha\":\"{}\",\"release\":{}}}",
            self.card,
            opt(self.aiclk_mhz),
            self.counter_mhz,
            opt(self.gddr_mts),
            opt(self.gddr_channels),
            escape(&self.sha),
            self.release
        );
        println!("BENCH {j}");
    }
}

fn opt(v: Option<u32>) -> String {
    v.map_or("null".into(), |v| v.to_string())
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// The source tree's commit, and whether it has local changes.
pub fn git_sha() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let sha = run(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty =
        run(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    if dirty {
        format!("{sha}+dirty")
    } else {
        sha
    }
}

/// One result. `value` is in `unit`; with a `peak`, `bytes_per_sec` is what is
/// held against it (for a throughput, the median's).
pub struct Record<'a> {
    pub key: &'a str,
    pub unit: &'a str,
    pub stats: Stats,
    pub bytes_per_sec: Option<f64>,
    pub peak: Option<&'a Peak>,
    /// How it was timed: `device` (tile counter) or `host` (`Instant`).
    pub timed: &'a str,
}

impl Record<'_> {
    pub fn emit(&self) {
        let s = &self.stats;
        let mut line = format!(
            "MEASURE [{}] {:<44} {:>10.3} {} (p10 {:.3}, p90 {:.3}, n {})",
            self.timed, self.key, s.median, self.unit, s.p10, s.p90, s.n
        );
        if let Some(bps) = self.bytes_per_sec {
            let _ = write!(line, " | {:.2} GB/s", bps / 1e9);
            if let Some(p) = self.peak {
                let _ = write!(
                    line,
                    " = {:.1}% of {} {:.1} GB/s",
                    100.0 * bps / p.bytes_per_sec,
                    p.name,
                    p.bytes_per_sec / 1e9
                );
            }
        }
        println!("{line}");
        let mut j = format!(
            "{{\"kind\":\"result\",\"key\":\"{}\",\"unit\":\"{}\",\"timed\":\"{}\",\"median\":{},\"p10\":{},\"p90\":{},\"n\":{}",
            escape(self.key),
            escape(self.unit),
            self.timed,
            s.median,
            s.p10,
            s.p90,
            s.n
        );
        if let Some(bps) = self.bytes_per_sec {
            let _ = write!(j, ",\"gbps\":{:.4}", bps / 1e9);
        }
        if let Some(p) = self.peak {
            let _ = write!(
                j,
                ",\"peak\":\"{}\",\"peak_gbps\":{:.4},\"peak_source\":\"{}\"",
                p.name,
                p.bytes_per_sec / 1e9,
                escape(&p.source)
            );
            if let Some(bps) = self.bytes_per_sec {
                let _ = write!(j, ",\"pct_of_peak\":{:.2}", 100.0 * bps / p.bytes_per_sec);
            }
        }
        j.push('}');
        println!("BENCH {j}");
    }
}

/// A latency or count, with no peak.
pub fn report(key: &str, unit: &str, timed: &str, stats: Stats) {
    Record {
        key,
        unit,
        stats,
        bytes_per_sec: None,
        peak: None,
        timed,
    }
    .emit();
}

/// A throughput from `bytes` moved per sample of `us` microseconds, against
/// `peak`. The stats printed are the microseconds; the rate is the median's.
pub fn report_rate(key: &str, timed: &str, bytes: f64, us: Stats, peak: Option<&Peak>) {
    Record {
        key,
        unit: "us",
        stats: us,
        bytes_per_sec: Some(bytes / (us.median * 1e-6)),
        peak,
        timed,
    }
    .emit();
}

#[cfg(feature = "silicon")]
pub use silicon::*;

#[cfg(feature = "silicon")]
mod silicon {
    use super::*;
    use crate::backend::{open_card, scrub};
    use crate::harness::Dev;
    use tt_device::telemetry::TelemetryTable;
    use tt_device::tlb::WindowKind;
    use tt_isa::noc::{Noc0, NocCoord};
    use tt_ttsim::fork_scope;

    /// One card (`TT_SILICON_DEVICE`, default 0), opened and scrubbed after, in
    /// a child process so a panic cannot leave the parent holding it.
    pub fn on_card(f: impl FnOnce(&mut Dev<'_>)) {
        let card = crate::backend::device_index();
        if let Err(e) = fork_scope(|| {
            let mut d = open_card(card);
            f(&mut d);
            scrub(&mut d);
        }) {
            panic!("{e}");
        }
    }

    /// Both cards, for the Ethernet benchmarks.
    pub fn with_cards(f: impl FnOnce(&mut Dev<'_>, &mut Dev<'_>)) {
        if let Err(e) = fork_scope(|| {
            let mut a = open_card(0);
            let mut b = open_card(1);
            f(&mut a, &mut b);
            scrub(&mut a);
            scrub(&mut b);
        }) {
            panic!("{e}");
        }
    }

    impl Conditions {
        /// Read the chip's clocks and memory from its ARC, and time `tile`'s
        /// counter against the host over 100 ms.
        pub fn measure(d: &mut Dev<'_>, card: u16, tile: NocCoord<Noc0>) -> Conditions {
            use tt_isa::arc::tag;
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let table = TelemetryTable::read(d, &w).unwrap();
            let mut read = |t| table.read_tag(d, &w, t).unwrap();
            let aiclk_mhz = read(tag::AICLK);
            let gddr_mts = read(tag::GDDR_SPEED);
            let gddr_channels = read(tag::ENABLED_GDDR).map(u32::count_ones);
            // Bracketed by host reads either side, so the PCIe round trip is
            // in both ends and cancels.
            let over = Duration::from_millis(100);
            let c0 = d.wall_clock(&w, tile).unwrap();
            let h0 = std::time::Instant::now();
            std::thread::sleep(over);
            let c1 = d.wall_clock(&w, tile).unwrap();
            let host_us = h0.elapsed().as_secs_f64() * 1e6;
            Conditions {
                card,
                aiclk_mhz,
                counter_mhz: (c1 - c0) as f64 / host_us,
                gddr_mts,
                gddr_channels,
                sha: git_sha(),
                release: !cfg!(debug_assertions),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_percentiles() {
        let s = Stats::of((1..=11).map(f64::from));
        assert_eq!((s.p10, s.median, s.p90, s.n), (2.0, 6.0, 10.0, 11));
        let one = Stats::of([3.0]);
        assert_eq!((one.p10, one.median, one.p90), (3.0, 3.0, 3.0));
    }

    #[test]
    fn peaks_from_conditions() {
        let c = Conditions {
            aiclk_mhz: Some(1350),
            counter_mhz: 1350.0,
            gddr_mts: Some(16000),
            gddr_channels: Some(8),
            ..Default::default()
        };
        assert!((c.noc_link().bytes_per_sec - 86.4e9).abs() < 1e6);
        assert!((c.gddr_channel().unwrap().bytes_per_sec - 64e9).abs() < 1e6);
        assert!((c.gddr_card().unwrap().bytes_per_sec - 512e9).abs() < 1e6);
        // 1350 bytes in one microsecond of a 1350 MHz counter.
        assert!((c.rate(1350.0, 1350.0) - 1.35e9).abs() < 1.0);
    }
}
