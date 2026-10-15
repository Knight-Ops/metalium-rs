//! The profile mode: one switch for where the time goes, on the card and on the host, for
//! anything run through burn-tt.
//!
//! ```text
//! TT_PROFILE=target/profile/run cargo run --release ...      # or: cargo xtask bench --profile ...
//! ```
//!
//! On attach the card's session enters `tt_kernels::profile_mode`; on detach this writes
//! `<dir>/chip<N>/`:
//!
//! * `summary.md`:
//!   - every Burn op's device time (eager ops each timed alone), split into barrier, kernels,
//!     control and data;
//!   - every trace's ops the same way, by name (`SharedStateTraces` keys, `TracedInference::set_name`),
//!     with how many replays covered them;
//!   - each trace's replays (input write, replay, output read);
//!   - the host's side: server-thread time and calls per Burn op, the session's host stages,
//!     dataflow counts and lists per tile.
//! * `ops.tsv`: every attributed op, one per line.
//! * `trace.json`: the same as a Perfetto/Chrome trace, spans named by Burn op.
//!
//! `TT_PROFILE_DETAIL=full` stamps every data entry too (fewer ops per replay window). A
//! `TT_PROFILE` ending in `.json` keeps the raw timestamp trace of every entry instead
//! (`tt_kernels::profile`). Programs can turn the mode on and read it without the environment:
//! [`profile_mode`], [`profile_report`].

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

pub use tt_kernels::profile_mode::{Detail, OpTime, ProfileOptions, ProfileReport};

use crate::error::TtError;
use crate::{server, TtDevice};

/// What the profile mode recorded on one card: see the module documentation.
#[derive(Clone, Debug)]
pub struct Profile {
    /// The card's side, from the data movers' timestamps.
    pub device: ProfileReport,
    /// Traces' names, by the session's trace number.
    pub names: BTreeMap<u64, String>,
    /// Each replay's input write, replay and output read, by the session's trace number.
    pub replays: BTreeMap<u64, Vec<[Duration; 3]>>,
    /// The session's own counters, as text: host stages, dataflow, lists per tile.
    pub host: String,
}

fn median_ms(mut v: Vec<Duration>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort();
    v[v.len() / 2].as_secs_f64() * 1e3
}

impl Profile {
    /// A trace's name, or its number.
    pub fn name(&self, id: u64) -> String {
        self.names
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("#{id}"))
    }

    /// Everything as markdown: the card's side, then the replays, then the host's.
    pub fn summary(&self) -> String {
        let name = |id: u64| self.name(id);
        let mut out = self.device.summary(&name);
        if !self.replays.is_empty() {
            let _ = writeln!(
                out,
                "## Replays\n\nMedians, host-timed.\n\n| trace | replays | input write ms | replay ms | output read ms |\n|---|--:|--:|--:|--:|"
            );
            for (&id, runs) in &self.replays {
                let col = |k: usize| median_ms(runs.iter().map(|r| r[k]).collect());
                let _ = writeln!(
                    out,
                    "| {} | {} | {:.3} | {:.3} | {:.3} |",
                    self.name(id),
                    runs.len(),
                    col(0),
                    col(1),
                    col(2)
                );
            }
            out.push('\n');
        }
        let server = crate::report::server_report();
        if !server.is_empty() {
            let busy: Duration = server.iter().map(|s| s.time).sum();
            let _ = writeln!(
                out,
                "## Host: the server thread by Burn op\n\n{:.1} ms busy in all (it builds and queues \
                 device work; a replay is one job, \"(outside an op)\").\n\n| op | jobs | ms | us a job |\n|---|--:|--:|--:|",
                busy.as_secs_f64() * 1e3
            );
            for s in &server {
                let ms = s.time.as_secs_f64() * 1e3;
                let _ = writeln!(
                    out,
                    "| {} | {} | {:.2} | {:.1} |",
                    s.op,
                    s.jobs,
                    ms,
                    ms * 1e3 / s.jobs.max(1) as f64
                );
            }
            out.push('\n');
        }
        let _ = writeln!(
            out,
            "## Host: Burn ops (calls, jobs, transfers)\n\n```text\n{}```\n\n## Host: the session\n\n```text\n{}```",
            crate::report::report(),
            self.host
        );
        out
    }

    /// Write `summary.md`, `ops.tsv` and `trace.json` into `dir`, creating it.
    pub fn write(&self, dir: &Path) -> std::io::Result<()> {
        let name = |id: u64| self.name(id);
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("summary.md"), self.summary())?;
        std::fs::write(dir.join("ops.tsv"), self.device.tsv(&name))?;
        std::fs::write(dir.join("trace.json"), self.device.chrome_trace(&name))
    }
}

/// The session's counters as text: where the server thread's time went queueing ops, the
/// dataflow counts, and how many lists each tile took.
pub(crate) fn session_text<T: tt_device::Transport>(s: &tt_kernels::session::Session<T>) -> String {
    let mut out = String::new();
    let h = s.host_times();
    let total: Duration = h.stages.iter().map(|(_, t)| t.time).sum();
    let _ = writeln!(
        out,
        "host stages: {} ops queued in {:.1} ms",
        h.ops,
        total.as_secs_f64() * 1e3
    );
    for (stage, t) in h.stages.iter().filter(|(_, t)| t.count > 0) {
        let _ = writeln!(
            out,
            "  {:<10} {:>9.1} ms  x{:<8} reads {:<8} writes {}",
            format!("{stage:?}"),
            t.time.as_secs_f64() * 1e3,
            t.count,
            t.traffic.read_calls,
            t.traffic.write_calls
        );
    }
    let d = s.dataflow_stats();
    let _ = writeln!(
        out,
        "dataflow: {} regions, {} batches, {} pack waits, {} release waits, {} transfer packets",
        d.regions, d.batches, d.pack_waits, d.release_waits, d.transfer_packets
    );
    let lists = s.lists_per_tile();
    let _ = writeln!(
        out,
        "lists: {} on {} of {} tiles; most on one tile {}",
        lists.iter().sum::<u64>(),
        lists.iter().filter(|&&l| l > 0).count(),
        lists.len(),
        lists.iter().max().copied().unwrap_or(0)
    );
    out
}

/// The profile mode's settings from the environment: `TT_PROFILE_DETAIL=full` for every entry.
pub fn options_from_env() -> ProfileOptions {
    ProfileOptions {
        detail: match std::env::var("TT_PROFILE_DETAIL").as_deref() {
            Ok("full") => Detail::Full,
            _ => Detail::Ops,
        },
        ..ProfileOptions::default()
    }
}

/// Turn `device`'s profile mode on (`Some`) or off (`None`, returning what it recorded), without
/// the environment. Silicon only: the simulator does not model the timestamps.
pub fn profile_mode(
    device: TtDevice,
    options: Option<ProfileOptions>,
) -> Result<Option<Profile>, TtError> {
    server::run(device, move |e, _| e.profile_mode(options))
}

/// What `device`'s profile mode has recorded so far, if it is on.
pub fn profile_report(device: TtDevice) -> Result<Option<Profile>, TtError> {
    server::run(device, |e, _| e.profile_report())
}
