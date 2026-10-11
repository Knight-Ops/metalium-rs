//! Building, listing and running test binaries one test per process: what
//! `cargo xtask silicon` and `cargo xtask conformance` share.
//!
//! The binaries are built once and run directly, not through `cargo test`, so a
//! run does not rebuild between tests and a per-test timeout measures the test
//! rather than cargo. One test per process because a card is one physical
//! resource and the simulator is a once-per-process singleton.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// A test binary `cargo test --no-run` produced.
#[derive(Clone)]
pub struct Built {
    /// The cargo target name (`step11_burn`, `tensor`).
    pub name: String,
    pub exe: PathBuf,
    /// The directory `cargo test` would run it from: its package's.
    pub dir: PathBuf,
}

/// Build test binaries and return what cargo reports.
///
/// `select` is the package and feature selection, as `cargo test` arguments
/// (`-p tt-tests --features tt-tests/silicon`). Parses cargo's JSON messages by
/// hand, which is enough here: the fields wanted are simple strings, and `xtask`
/// stays dependency-free.
pub fn build(root: &Path, release: bool, select: &[String]) -> Result<Vec<Built>, String> {
    let out = Command::new(env!("CARGO"))
        .arg("test")
        .args(select)
        .args(["--no-run", "--message-format=json"])
        .args(release.then_some("--release"))
        .current_dir(root)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("could not run cargo: {e}"))?;
    if !out.status.success() {
        return Err("building the test binaries failed".into());
    }
    let mut v = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        if !line.contains("\"reason\":\"compiler-artifact\"") {
            continue;
        }
        let Some(exe) = json_str(line, "\"executable\":\"") else {
            continue;
        };
        let Some(name) = line
            .find("\"target\":{")
            .and_then(|i| json_str(&line[i..], "\"name\":\""))
        else {
            continue;
        };
        let Some(dir) = json_str(line, "\"manifest_path\":\"")
            .and_then(|m| Path::new(&m).parent().map(Path::to_path_buf))
        else {
            continue;
        };
        v.push(Built {
            name,
            exe: PathBuf::from(exe),
            dir,
        });
    }
    v.sort_by(|a, b| (&a.name, &a.exe).cmp(&(&b.name, &b.exe)));
    Ok(v)
}

fn json_str(s: &str, key: &str) -> Option<String> {
    let start = s.find(key)? + key.len();
    let end = s[start..].find('"')?;
    Some(s[start..start + end].to_string())
}

/// The tests in a binary, without the `#[ignore]` ones unless `include_ignored`.
pub fn list(exe: &Path, include_ignored: bool) -> Result<Vec<String>, String> {
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
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

/// Run one test of `exe` in its own process, on one thread, from `dir`, with
/// `env` added; stdout and stderr go to `out_path`.
pub fn run_one(
    exe: &Path,
    test: &str,
    dir: &Path,
    env: &[(&str, String)],
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
        .envs(env.iter().map(|(k, v)| (*k, v)))
        .current_dir(dir)
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
            let _ = child.wait();
            break Verdict::Timeout;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    if let Ok(f) = File::open(out_path) {
        let _ = f.sync_all();
    }
    Ok(verdict)
}
