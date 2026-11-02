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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Mutex;
use std::time::Duration;

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

/// [`list`] for every binary, several at a time (each is two process starts, and a
/// selection is a few hundred binaries); the result is in `bins`' order.
pub fn list_all(bins: &[Built], include_ignored: bool) -> Result<Vec<Vec<String>>, String> {
    let next = AtomicUsize::new(0);
    type Listed = Option<Result<Vec<String>, String>>;
    let lists: Mutex<Vec<Listed>> = Mutex::new(bins.iter().map(|_| None).collect());
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|scope| {
        for _ in 0..jobs.min(bins.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some(b) = bins.get(i) else {
                    return;
                };
                let l = list(&b.exe, include_ignored);
                lists.lock().unwrap()[i] = Some(l);
            });
        }
    });
    lists
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|l| l.expect("every binary was listed"))
        .collect()
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
    // Its own process group: the gates do their device work in a forked child
    // (`tt_ttsim::fork_scope`), and a timeout that killed only the process it spawned left that
    // child running, still holding the card, until the next run found it "in use".
    use std::os::unix::process::CommandExt;
    let mut child = Command::new(exe)
        .process_group(0)
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
    // Waited for on another thread, so the verdict comes the moment the test exits
    // rather than at the next poll.
    let pid = child.id();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(child.wait());
    });
    let verdict = match rx.recv_timeout(timeout) {
        Ok(status) => {
            if status.map_err(|e| e.to_string())?.success() {
                Verdict::Pass
            } else {
                Verdict::Fail
            }
        }
        Err(RecvTimeoutError::Timeout) => {
            // A process blocked in an MMIO read against a hung NoC may not die;
            // the verdict is recorded either way, and the log already says START.
            // The test leads its own group, so this kills it too.
            kill_group(pid);
            let _ = rx.recv();
            // The leader's exit is not the group's: a forked child still tearing down its
            // mappings holds the card, and the next gate's attach found it "in use"
            // (`1793550655`).
            wait_group_gone(pid, Duration::from_secs(30));
            Verdict::Timeout
        }
        Err(RecvTimeoutError::Disconnected) => return Err("the test's waiter thread died".into()),
    };
    if let Ok(f) = File::open(out_path) {
        let _ = f.sync_all();
    }
    Ok(verdict)
}

/// SIGKILL every process in the group `pgid` leads (the test and any child it forked).
fn kill_group(pgid: u32) {
    let _ = Command::new("kill")
        .args(["-KILL", "--"])
        .arg(format!("-{pgid}"))
        .status();
}

/// Wait until no process of group `pgid` is left (`kill -0` on the group fails), or `most`.
fn wait_group_gone(pgid: u32, most: Duration) {
    let start = std::time::Instant::now();
    while start.elapsed() < most {
        let alive = Command::new("kill")
            .args(["-0", "--"])
            .arg(format!("-{pgid}"))
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if !alive {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A timeout must take the test's forked children with it: the gates fork their device work, and
    /// a survivor keeps the card "in use" for the next run.
    #[test]
    fn a_timeout_kills_the_forked_child_too() {
        let dir = std::env::temp_dir().join(format!("xtask-timeout-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fake-test.sh");
        let pid_file = dir.join("child.pid");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nsleep 300 &\necho $! > {}\nsleep 300\n",
                pid_file.display()
            ),
        )
        .unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        let verdict = run_one(
            &script,
            "ignored",
            &dir,
            &[],
            Duration::from_millis(1500),
            &dir.join("out.txt"),
        )
        .unwrap();
        assert!(matches!(verdict, Verdict::Timeout));

        let child: u32 = std::fs::read_to_string(&pid_file)
            .expect("the script started its child before the deadline")
            .trim()
            .parse()
            .unwrap();
        // `kill -0` succeeds only for a live process (a reaped one has no entry; a zombie of ours
        // would, but the child's parent was the script, which is gone, so init reaps it). No
        // grace period: `run_one` returns only once the group is gone.
        let alive = Command::new("kill")
            .args(["-0", &child.to_string()])
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(!alive, "the forked child {child} survived the timeout");
    }
}
