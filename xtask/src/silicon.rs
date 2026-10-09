//! `cargo xtask silicon`: run the silicon suite one test per process, with a log
//! that survives the host dying.
//!
//! Every rule in here was paid for during Phase 1's bring-up
//! (`docs/learnings/silicon-operating-notes.md`):
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
    list_only: bool,
    release: bool,
    timeout: Duration,
}

fn parse(args: impl Iterator<Item = String>) -> Result<Opts, String> {
    let mut o = Opts {
        devices: vec![0],
        filters: Vec::new(),
        include_ignored: false,
        keep_going: false,
        list_only: false,
        release: false,
        timeout: Duration::from_secs(120),
    };
    let mut smoke = false;
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
            "--smoke" => smoke = true,
            "--include-ignored" => o.include_ignored = true,
            "--keep-going" => o.keep_going = true,
            "--list" => o.list_only = true,
            "--release" => o.release = true,
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown option `{other}`\n\n{USAGE}")),
        }
    }
    if smoke {
        o.filters.extend(SMOKE.iter().map(|s| s.to_string()));
    }
    Ok(o)
}

/// What `--smoke` runs, in order: the device backend against `burn-flex`,
/// cheapest claim first, so a broken op fails before a training run spends a
/// minute finding it. Simulator validation is plain `cargo test`; this is the
/// tier that says burn-tt still computes what Burn's CPU backend computes, on
/// the hardware it is for.
pub const SMOKE: &[&str] = &[
    "step19_eltwise::",
    "step27_burn_eltwise::",
    "step32_burn_softmax::",
    "step35_burn_rank_n::",
    "step47_burn_activations::",
    "step11_burn::",
    "step20_many_tiles::eltwise",
    "step20_many_tiles::column_sums",
    "step12_mnist::the_first_forward_pass",
    "step12_mnist::the_mlp_trains_on_a_reduced_dataset",
    "step12_mnist::the_mlp_trains_on_four_tiles",
    "step59_burn_transformer::",
    "step60_batched_blocks::",
    "step61_burn_gather::",
    "step62_gather_rows::",
    "step63_burn_full_reduce::",
    "step66_burn_small_ops::",
    "step67_general_reduce::",
    "step68_k_block_matmul::",
    "step69_reduction_primitives::",
    "step70_native_norms::",
    "step71_integer_alu::",
    "step72_round_cast::",
    "step73_fpu_transpose::",
    "step74_bf16_storage::",
    "step75_fpu_pooling::",
    "step76_burn_bf16::",
    "step77_bf16_matmul::",
    "step78_burn_pooling::",
    "step79_extremum_scans::",
    "step80_integer_reductions::",
    "step81_hardware_rounding::",
    "step82_integer_division::",
    "step83_attention::",
    "step84_slice_assignment::",
    "step85_convolution::",
    "step86_resident_indices::",
    "step87_src_transpose::",
    "step88_mesh_module_reference::",
    "step89_mnist_cnn::",
    "step90_matrix_eltwise::",
    "step91_seeded_prng::",
    "step92_bfp_formats::",
    "step93_exponent_history::",
    "step94_bfp_storage::",
    "step95_burn_bfp::",
    "step96_mixed_bfp_mnist::",
    "step97_matrix_register_moves::",
    "step98_adc_copy::",
    "step99_adc_planes::",
    "step100_scalar_config::",
    "step101_l1_movement::",
    "step102_xmov_tensor::",
    "step103_source_banks::",
    "step104_unpacker_handover::",
    "step130_float_extremum_scans::",
    "step131_int_scans::",
    "step132_int_arg::",
    "step125_int_bool_views::",
    "step126_int_mask_abs::",
    "step127_nd_indexing::",
    "step128_cross_int_matmul::",
    "step129_prelu_residency::",
    "step133_remainder::",
    "step134_burn_remainder::",
    "step135_sort::",
    "step136_burn_sort::",
    "step137_sort_residency::",
    "step138_sort_traces::",
    "step139_sort_long_axes::",
    "step9_matmul::elw_broadcast_assignment_and_destination_fields",
    "step9_matmul::elw_oracles_reject_safe_instruction_mutants",
];

pub const USAGE: &str = "\
usage: cargo xtask silicon [options]

  --device N|all        which /dev/tenstorrent/N to run against (default 0);
                        `all` runs the whole selection on each card in turn
  --filter S            run only tests whose `binary::test` name contains S;
                        repeatable, and the selection runs in filter order
  --smoke               the everyday smoke test: burn-tt against burn-flex,
                        from single ops up to the reduced training runs
                        (adds the filters in `SMOKE`, after any given)
  --include-ignored     also run #[ignore] tests (the exploratory probes)
  --keep-going          do not stop at the first failure
  --timeout-secs N      per-test wall-clock limit (default 120)
  --release             build the suite optimised; every timing worth quoting
                        comes from this, since the dev build is not a baseline
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
    let (_, results) = run_opts(&o)?;
    if results.iter().all(|r| r.2 == Verdict::Pass) {
        Ok(())
    } else {
        Err("silicon suite failed".into())
    }
}

/// What `cargo xtask bench` runs by default, in order: device-timed memory,
/// the core path, the Ethernet counter's own gate, then Ethernet -- which
/// needs that gate to have passed -- and the host link last.
pub const BENCH: &[&str] = &[
    "silicon_bench_memory::",
    "silicon_bench_path::",
    "silicon_eth_clock::",
    "silicon_bench_eth::",
    "silicon_perf::pcie_l1",
    "silicon_perf::pcie_dram",
];

pub const BENCH_USAGE: &str = "\
usage: cargo xtask bench [silicon options]

  Runs the firmware benchmarks (`BENCH` in xtask/src/silicon.rs, or the
  given --filter selection) as `cargo xtask silicon --release
  --include-ignored`, with a 900 s per-test limit unless --timeout-secs says
  otherwise, then collects every `BENCH {json}` line the tests printed into

    target/silicon/bench/<stamp>.jsonl   one record per line, with its test
    target/silicon/bench/<stamp>.md      the same as a table

  Takes every `cargo xtask silicon` option; see `cargo xtask silicon --help`.";

/// `cargo xtask bench`: the benchmark preset of the silicon runner, plus the
/// collection of what the benchmarks reported.
pub fn bench(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args: Vec<String> = args.collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return Err(BENCH_USAGE.into());
    }
    let mut o = parse(args.iter().cloned())?;
    o.release = true;
    o.include_ignored = true;
    if !args.iter().any(|a| a == "--timeout-secs") {
        o.timeout = Duration::from_secs(900);
    }
    if o.filters.is_empty() {
        o.filters.extend(BENCH.iter().map(|s| s.to_string()));
    }
    let (stamp, results) = run_opts(&o)?;
    if o.list_only {
        return Ok(());
    }
    let dir = workspace_root().join("target/silicon/bench");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut jsonl = String::new();
    let mut md = String::from(
        "| test | dev | result | timed | median | unit | p10 | p90 | GB/s | % of peak | peak |\n\
         |---|--:|---|---|--:|---|--:|--:|--:|--:|---|\n",
    );
    let mut conditions = String::new();
    for (dev, full, verdict, out) in &results {
        let text = std::fs::read_to_string(out).unwrap_or_default();
        for rec in text.lines().filter_map(|l| l.strip_prefix("BENCH ")) {
            jsonl += &format!("{{\"test\":\"{full}\",\"dev\":{dev},\"verdict\":\"{verdict}\",\"record\":{rec}}}\n");
            if rec.contains("\"kind\":\"conditions\"") {
                if conditions.is_empty() {
                    conditions = rec.to_string();
                }
                continue;
            }
            let f = |k: &str| json_field(rec, k).unwrap_or_default();
            md += &format!(
                "| {} | {dev} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                full.split("::").last().unwrap_or(full),
                f("key"),
                f("timed"),
                num(&f("median")),
                f("unit"),
                num(&f("p10")),
                num(&f("p90")),
                f("gbps"),
                f("pct_of_peak"),
                f("peak"),
            );
        }
    }
    let jsonl_path = dir.join(format!("{stamp}.jsonl"));
    let md_path = dir.join(format!("{stamp}.md"));
    std::fs::write(&jsonl_path, jsonl).map_err(|e| format!("{}: {e}", jsonl_path.display()))?;
    std::fs::write(&md_path, format!("conditions: `{conditions}`\n\n{md}"))
        .map_err(|e| format!("{}: {e}", md_path.display()))?;
    println!(
        "benchmarks collected into {} and {}",
        jsonl_path.display(),
        md_path.display()
    );
    if results.iter().all(|r| r.2 == Verdict::Pass) {
        Ok(())
    } else {
        Err("a benchmark failed; what the others reported was still collected".into())
    }
}

/// A value from one flat JSON object, as its text, quotes removed.
fn json_field(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let start = obj.find(&pat)? + pat.len();
    let rest = &obj[start..];
    if let Some(r) = rest.strip_prefix('"') {
        return Some(r[..r.find('"')?].to_string());
    }
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

fn num(s: &str) -> String {
    s.parse::<f64>()
        .map_or(s.to_string(), |v| format!("{v:.3}"))
}

type Results = Vec<(u16, String, Verdict, PathBuf)>;

fn run_opts(o: &Opts) -> Result<(u64, Results), String> {
    let root = workspace_root();
    let binaries = build(&root, o.release)?;

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
        // A filter that matches nothing is a typo or a renamed test, and with
        // `--smoke` a stale preset entry; either way, say so rather than run less.
        let unmatched: Vec<&String> = o
            .filters
            .iter()
            .filter(|f| !selection.iter().any(|s| s.0.contains(f.as_str())))
            .collect();
        if !unmatched.is_empty() {
            return Err(format!("no test matches the filter(s) {unmatched:?}"));
        }
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
        return Ok((0, Vec::new()));
    }

    preflight(&o.devices)?;

    let dir = root.join("target/silicon");
    std::fs::create_dir_all(dir.join("out")).map_err(|e| format!("{}: {e}", dir.display()))?;
    let stamp = unix_secs();
    let log_path = dir.join(format!("{stamp}.log"));
    let mut log = Log::open(&log_path)?;
    let boot = boot_id();
    println!("logging to {}", log_path.display());
    // A timing in the log is only comparable with one from the same profile.
    let profile = if o.release { "release" } else { "dev" };
    log.line(&format!("{} boot={boot} PROFILE {profile}", unix_secs()))?;

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
            let verdict = run_one(&root, exe, test, dev, o.timeout, &out_path)?;
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
    Ok((stamp, results))
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
fn build(root: &Path, release: bool) -> Result<Vec<(String, PathBuf)>, String> {
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
        .args(release.then_some("--release"))
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
fn preflight(devices: &[u16]) -> Result<(), String> {
    for d in devices {
        let p = format!("/dev/tenstorrent/{d}");
        if !Path::new(&p).exists() {
            return Err(format!("{p} does not exist"));
        }
    }
    let param = "/sys/module/tenstorrent/parameters/auto_reset_timeout";
    match std::fs::read_to_string(param).map(|v| v.trim().to_string()) {
        Ok(v) if v == "0" => {
            println!("auto_reset_timeout=0: a hung NoC wedges the card, not the host")
        }
        Ok(v) => println!(
            "note: auto_reset_timeout={v} -- the ARC watchdog is armed, so a hung NoC \
             resets the chip and drops the PCIe link"
        ),
        Err(e) => println!("note: cannot read {param} ({e})"),
    }
    Ok(())
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
