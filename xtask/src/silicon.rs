//! `cargo xtask silicon`: run the silicon suite one test per process, with a log
//! that survives the host dying.
//!
//! Every rule in here was paid for during Phase 1's bring-up
//! (`docs/implementation-checklist.md`, "Silicon operating notes"):
//!
//! * **One test per process, one thread.** All gates share one physical card, and
//!   window allocation is global card state. Running them one at a time is also
//!   what identified the failing access when the host went down — the output of a
//!   whole-suite run died with the session.
//! * **A log `fsync`'d after every line, carrying `boot_id`.** A hard kill loses
//!   the journal's last minutes and unflushed file data, so the last `START` line
//!   without a matching `END` names the test that took the machine down, and a
//!   changed `boot_id` on the next run makes a reboot evidence rather than
//!   inference.
//! * **Stop at the first failure** unless told otherwise. A failing silicon gate
//!   may have left the card in a state the next gate should not inherit.
//!
//! The test binaries are built once and run directly, not through `cargo test`,
//! so a run does not rebuild between tests and the per-test timeout measures the
//! test rather than cargo.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::util::workspace_root;

struct Opts {
    devices: Vec<u16>,
    filters: Vec<String>,
    include_ignored: bool,
    keep_going: bool,
    allow_armed_watchdog: bool,
    list_only: bool,
    timeout: Duration,
}

fn parse(args: impl Iterator<Item = String>) -> Result<Opts, String> {
    let mut o = Opts {
        devices: vec![0],
        filters: Vec::new(),
        include_ignored: false,
        keep_going: false,
        allow_armed_watchdog: false,
        list_only: false,
        timeout: Duration::from_secs(120),
    };
    let mut args = args.peekable();
    while let Some(a) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .ok_or_else(|| format!("`{name}` needs a value\n\n{USAGE}"))
        };
        match a.as_str() {
            "--device" => {
                let v = value("--device")?;
                o.devices = if v == "all" {
                    all_devices()?
                } else {
                    vec![v
                        .parse()
                        .map_err(|_| format!("`--device {v}`: expected a number or `all`"))?]
                };
            }
            "--filter" => o.filters.push(value("--filter")?),
            "--timeout-secs" => {
                let v = value("--timeout-secs")?;
                o.timeout = Duration::from_secs(
                    v.parse()
                        .map_err(|_| format!("`--timeout-secs {v}`: expected a number"))?,
                );
            }
            "--include-ignored" => o.include_ignored = true,
            "--keep-going" => o.keep_going = true,
            "--allow-armed-watchdog" => o.allow_armed_watchdog = true,
            "--list" => o.list_only = true,
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown option `{other}`\n\n{USAGE}")),
        }
    }
    Ok(o)
}

pub const USAGE: &str = "\
usage: cargo xtask silicon [options]

  --device N|all        which /dev/tenstorrent/N to run against (default 0);
                        `all` runs the whole selection on each card in turn
  --filter S            run only tests whose `binary::test` name contains S;
                        repeatable, and the selection runs in filter order
  --include-ignored     also run #[ignore] tests (the exploratory probes)
  --keep-going          do not stop at the first failure
  --allow-armed-watchdog  run even though auto_reset_timeout is not 0; a hung
                        NoC then resets the chip and can take the host down
  --timeout-secs N      per-test wall-clock limit (default 120)
  --list                print the selection and exit without touching a card";

/// Every card the driver has enumerated.
fn all_devices() -> Result<Vec<u16>, String> {
    let mut v: Vec<u16> = std::fs::read_dir("/dev/tenstorrent")
        .map_err(|e| format!("/dev/tenstorrent: {e}"))?
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
        .collect();
    v.sort();
    if v.is_empty() {
        return Err("no cards under /dev/tenstorrent".into());
    }
    Ok(v)
}

pub fn run(args: impl Iterator<Item = String>) -> Result<(), String> {
    let o = parse(args)?;
    let root = workspace_root();
    let binaries = build(&root)?;

    let mut selection = Vec::new();
    for (name, exe) in &binaries {
        for test in list(exe, o.include_ignored)? {
            let full = format!("{name}::{test}");
            selection.push((full, exe.clone(), test));
        }
    }
    if !o.filters.is_empty() {
        // Filter order is run order, so a caller can say "Phase 2, then Phase 3"
        // and have the riskier group gate the next one.
        let mut ordered = Vec::new();
        for f in &o.filters {
            for s in &selection {
                if s.0.contains(f.as_str()) && !ordered.iter().any(|o: &(String, _, _)| o.0 == s.0)
                {
                    ordered.push(s.clone());
                }
            }
        }
        selection = ordered;
    }
    if selection.is_empty() {
        return Err("no tests selected".into());
    }
    if o.list_only {
        for (full, _, _) in &selection {
            println!("{full}");
        }
        return Ok(());
    }

    preflight(&o.devices, o.allow_armed_watchdog)?;

    let dir = root.join("target/silicon");
    std::fs::create_dir_all(dir.join("out")).map_err(|e| format!("{}: {e}", dir.display()))?;
    let stamp = unix_secs();
    let log_path = dir.join(format!("{stamp}.log"));
    let mut log = Log::open(&log_path)?;
    let boot = boot_id();
    println!("logging to {}", log_path.display());

    let mut results = Vec::new();
    'cards: for &dev in &o.devices {
        for (full, exe, test) in &selection {
            log.line(&format!(
                "{} boot={boot} dev={dev} START {full}",
                unix_secs()
            ))?;
            let out_path = dir
                .join("out")
                .join(format!("{stamp}-dev{dev}-{}.txt", full.replace("::", "__")));
            let started = Instant::now();
            let verdict = run_one(
                o.allow_armed_watchdog,
                &root,
                exe,
                test,
                dev,
                o.timeout,
                &out_path,
            )?;
            let secs = started.elapsed().as_secs_f64();
            log.line(&format!(
                "{} boot={boot} dev={dev} END {full} {verdict} {secs:.2}s",
                unix_secs()
            ))?;
            println!("dev{dev} {verdict:<7} {secs:>7.2}s  {full}");
            let failed = verdict != Verdict::Pass;
            results.push((dev, full.clone(), verdict, out_path));
            if failed && !o.keep_going {
                break 'cards;
            }
        }
    }

    let failures: Vec<_> = results.iter().filter(|r| r.2 != Verdict::Pass).collect();
    println!(
        "\n{} run, {} passed, {} failed",
        results.len(),
        results.len() - failures.len(),
        failures.len()
    );
    for (dev, full, verdict, out) in &failures {
        println!("  dev{dev} {verdict} {full}\n    output: {}", out.display());
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err("silicon suite failed".into())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Pass,
    Fail,
    Timeout,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Timeout => "TIMEOUT",
        })
    }
}

/// Build the silicon test binaries and return `(target name, executable)`.
///
/// Parses cargo's JSON messages by hand, which is enough here: the two fields
/// wanted are simple strings, and `xtask` stays dependency-free.
fn build(root: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let out = Command::new(env!("CARGO"))
        .args([
            "test",
            "-p",
            "tt-tests",
            "--features",
            "silicon",
            "--no-run",
            "--message-format=json",
        ])
        .current_dir(root)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("could not run cargo: {e}"))?;
    if !out.status.success() {
        return Err("building the silicon suite failed".into());
    }
    let mut v = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if !line.contains("\"reason\":\"compiler-artifact\"") {
            continue;
        }
        let Some(exe) = json_str(line, "\"executable\":\"") else {
            continue;
        };
        let Some(target) = line
            .find("\"target\":{")
            .and_then(|i| json_str(&line[i..], "\"name\":\""))
        else {
            continue;
        };
        v.push((target, PathBuf::from(exe)));
    }
    v.sort();
    Ok(v)
}

fn json_str(s: &str, key: &str) -> Option<String> {
    let start = s.find(key)? + key.len();
    let end = s[start..].find('"')?;
    Some(s[start..start + end].to_string())
}

fn list(exe: &Path, include_ignored: bool) -> Result<Vec<String>, String> {
    let mut all = run_list(exe, false)?;
    if !include_ignored {
        let ignored = run_list(exe, true)?;
        all.retain(|t| !ignored.contains(t));
    }
    Ok(all)
}

fn run_list(exe: &Path, ignored_only: bool) -> Result<Vec<String>, String> {
    let mut cmd = Command::new(exe);
    cmd.args(["--list", "--format", "terse"]);
    if ignored_only {
        cmd.arg("--ignored");
    }
    let out = cmd
        .output()
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.strip_suffix(": test"))
        .map(str::to_string)
        .collect())
}

fn run_one(
    allow_armed_watchdog: bool,
    root: &Path,
    exe: &Path,
    test: &str,
    dev: u16,
    timeout: Duration,
    out_path: &Path,
) -> Result<Verdict, String> {
    let out = File::create(out_path).map_err(|e| format!("{}: {e}", out_path.display()))?;
    let err = out.try_clone().map_err(|e| e.to_string())?;
    let mut child = Command::new(exe)
        .args([
            test,
            "--exact",
            "--test-threads=1",
            "--include-ignored",
            "--nocapture",
        ])
        .env("TT_SILICON_DEVICE", dev.to_string())
        // The harness refuses an armed watchdog on its own; `--allow-armed-watchdog`
        // is what lifts that, so it has to reach the child.
        .env(
            "TT_ALLOW_ARMED_WATCHDOG",
            if allow_armed_watchdog { "1" } else { "0" },
        )
        // `cargo test` runs a package's tests from its own directory.
        .current_dir(root.join("crates/tt-tests"))
        .stdout(out)
        .stderr(err)
        .spawn()
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    let deadline = Instant::now() + timeout;
    let verdict = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break if status.success() {
                Verdict::Pass
            } else {
                Verdict::Fail
            };
        }
        if Instant::now() >= deadline {
            // A process blocked in an MMIO read against a hung NoC may not die;
            // the verdict is recorded either way, and the log already says START.
            let _ = child.kill();
            break Verdict::Timeout;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    if let Ok(f) = File::open(out_path) {
        let _ = f.sync_all();
    }
    Ok(verdict)
}

/// Refuse to start against a card that is not there, and say out loud whether the
/// ARC watchdog is armed.
fn preflight(devices: &[u16], allow_armed_watchdog: bool) -> Result<(), String> {
    for d in devices {
        let p = format!("/dev/tenstorrent/{d}");
        if !Path::new(&p).exists() {
            return Err(format!("{p} does not exist"));
        }
    }
    let param = "/sys/module/tenstorrent/parameters/auto_reset_timeout";
    match std::fs::read_to_string(param).map(|v| v.trim().to_string()) {
        Ok(v) if v == "0" => {
            println!("auto_reset_timeout=0: a hung NoC wedges the card, not the host");
            Ok(())
        }
        _ if allow_armed_watchdog => {
            println!("WARNING: running with the ARC watchdog armed, as asked");
            Ok(())
        }
        Ok(v) => Err(format!(
            "the ARC watchdog is armed (auto_reset_timeout={v}): a hung NoC escalates \
             to a chip reset that drops the PCIe link and, with the card passed \
             through, the host. Reload the driver with auto_reset_timeout=0, or pass \
             --allow-armed-watchdog to accept that."
        )),
        Err(e) => Err(format!(
            "cannot read {param} ({e}); is the tenstorrent module loaded? \
             --allow-armed-watchdog proceeds without knowing."
        )),
    }
}

struct Log(File);

impl Log {
    fn open(path: &Path) -> Result<Self, String> {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map(Log)
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Append and `fsync`: a line that is not on disk when the host dies is a line
    /// that was never written.
    fn line(&mut self, s: &str) -> Result<(), String> {
        writeln!(self.0, "{s}").map_err(|e| e.to_string())?;
        self.0.sync_all().map_err(|e| e.to_string())
    }
}

fn boot_id() -> String {
    std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
