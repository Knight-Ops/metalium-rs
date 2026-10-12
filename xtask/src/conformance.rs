//! `cargo xtask conformance`: Burn's own backend tests, run against `burn-tt`.
//!
//! The tests are upstream's (`vendor/burn-backend-tests`, `cargo xtask
//! fetch-burn-tests`), compiled by `crates/burn-tt-conformance`. This runs every
//! test in its own process, because the simulator is a once-per-process singleton
//! and one failed test must not poison the next, and compares the outcome with
//! `crates/burn-tt-conformance/expected-failures.toml`:
//!
//! * a test that fails and is not listed is a regression or a new finding;
//! * a test that passes and is listed is stale, so the list only shrinks;
//! * an entry whose test no longer exists is stale too;
//! * an entry in the `todo` category is untriaged.
//!
//! Every one of those fails the run. Passing is not the claim: matching the list
//! is, so the list is the record of what burn-tt does not do yet, with a reason.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::burn_tests;
use crate::pin;
use crate::silicon::{boot_id, preflight, unix_secs, Log};
use crate::testrun::{build, list, run_one, Built, Verdict};
use crate::util::workspace_root;

pub const USAGE: &str = "\
usage: cargo xtask conformance [options]

  Builds crates/burn-tt-conformance and runs Burn's own backend tests against
  burn-tt on the simulator, one test per process, then compares the outcome with
  crates/burn-tt-conformance/expected-failures.toml. Needs `cargo xtask
  fetch-burn-tests` and `cargo xtask fetch-ttsim`.

  --suite NAME          run only this test binary (`tensor`, `autodiff`,
                        `autodiff_checkpointing`, `tensor_f16`, `autodiff_f16`,
                        `tensor_bf16`, `autodiff_bf16`); repeatable. The two
                        bf16 binaries run only with --silicon or --reference-flex
                        (ttsim refuses the BF16 narrowing)
  --filter S            run only tests whose `binary::test` name contains S;
                        repeatable. A filtered run does not check for stale entries
  --fusion              run on Fusion<TtBackend> (expected-failures-fusion.toml)
  --exact               matmuls accumulate in exact F32, in k order, on the SFPU
                        (TT_EXACT_MATMUL=1, `burn_tt::MatmulMode::Exact`) rather
                        than on the Matrix Unit: expected-failures-exact.toml,
                        and `-fusion-exact` with --fusion
  --accurate            matmuls run as integer slices with shared exponents on the
                        Matrix Unit at LoFi (TT_ACCURATE_MATMUL=1,
                        `burn_tt::MatmulMode::Accurate`): expected-failures-accurate.toml,
                        and `-fusion-accurate` with --fusion. Not with --exact
  --silicon             run on card 0 instead of ttsim: one test at a time (the card
                        lock refuses a second process), with the fsync'd log
                        target/silicon/<stamp>.log, against
                        expected-failures-silicon{-fusion}{-exact}{-accurate}.toml.
                        Does not need ttsim. Includes the bf16 suites
  --reference-flex      run the suite on burn-flex, on the host, instead of burn-tt:
                        the second implementation that tells a test a dtype cannot
                        pass from a burn-tt defect. Compares with no list; prints
                        the failures and exits 0
  --jobs N              processes at a time (default: the CPU count; 1 with --silicon)
  --timeout-secs N      per-test wall-clock limit (default 120; 600 with --silicon,
                        where the 300 to 500 LU tests take up to 355 s)
  --strict              TT_STRICT=1: a download or a staged matmul panics the test
  --release             build optimised; the simulator runs the same program
                        faster
  --update              rewrite the expected-failures file: drop entries that now
                        pass, and add new failures under a `todo` category to be
                        triaged by hand
  --list                print the selection and exit";

struct Opts {
    suites: Vec<String>,
    filters: Vec<String>,
    fusion: bool,
    exact: bool,
    accurate: bool,
    jobs: usize,
    timeout: Duration,
    strict: bool,
    release: bool,
    update: bool,
    list_only: bool,
    silicon: bool,
    reference_flex: bool,
}

/// Test binaries that need the BF16 narrowing (PACR mode `0x105`), which ttsim refuses
/// (`docs/learnings/ttsim-divergence.md`, "BF16 late narrowing"). burn-tt runs them on
/// silicon only; burn-flex runs them anywhere.
const SILICON_ONLY_SUITES: &[&str] = &["tensor_bf16", "autodiff_bf16"];

fn parse(args: impl Iterator<Item = String>) -> Result<Opts, String> {
    let mut o = Opts {
        suites: Vec::new(),
        filters: Vec::new(),
        fusion: false,
        exact: false,
        accurate: false,
        jobs: std::thread::available_parallelism().map_or(4, |n| n.get()),
        timeout: Duration::from_secs(120),
        strict: false,
        release: false,
        update: false,
        list_only: false,
        silicon: false,
        reference_flex: false,
    };
    let (mut jobs_given, mut timeout_given) = (false, false);
    let mut args = args;
    while let Some(a) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .ok_or_else(|| format!("`{name}` needs a value\n\n{USAGE}"))
        };
        let number = |name: &str, v: String| {
            v.parse::<usize>()
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(|| format!("`{name} {v}`: expected a positive number"))
        };
        match a.as_str() {
            "--suite" => o.suites.push(value("--suite")?),
            "--filter" => o.filters.push(value("--filter")?),
            "--jobs" => {
                o.jobs = number("--jobs", value("--jobs")?)?;
                jobs_given = true;
            }
            "--timeout-secs" => {
                o.timeout =
                    Duration::from_secs(number("--timeout-secs", value("--timeout-secs")?)? as u64);
                timeout_given = true;
            }
            "--silicon" => o.silicon = true,
            "--reference-flex" => o.reference_flex = true,
            "--fusion" => o.fusion = true,
            "--exact" => o.exact = true,
            "--accurate" => o.accurate = true,
            "--strict" => o.strict = true,
            "--release" => o.release = true,
            "--update" => o.update = true,
            "--list" => o.list_only = true,
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown option `{other}`\n\n{USAGE}")),
        }
    }
    if o.exact && o.accurate {
        return Err(format!(
            "`--exact` and `--accurate` are two matmul modes: choose one\n\n{USAGE}"
        ));
    }
    if o.silicon && o.reference_flex {
        return Err(format!(
            "`--silicon` runs burn-tt on the card and `--reference-flex` runs burn-flex on the host: choose one\n\n{USAGE}"
        ));
    }
    if o.reference_flex && (o.fusion || o.exact || o.accurate || o.update) {
        return Err(format!(
            "`--reference-flex` takes no burn-tt profile and keeps no list: not with `--fusion`, `--exact`, `--accurate` or `--update`\n\n{USAGE}"
        ));
    }
    if o.silicon {
        if jobs_given && o.jobs != 1 {
            return Err(format!(
                "`--silicon` runs one test at a time: the card lock refuses a second process\n\n{USAGE}"
            ));
        }
        o.jobs = 1;
        if !timeout_given {
            o.timeout = Duration::from_secs(600);
        }
    }
    Ok(o)
}

/// Why a test is expected to fail. Every entry needs one, and `todo` is only a
/// place to put a new failure until it has been looked at.
const CATEGORIES: &[&str] = &[
    "unsupported-dtype",
    "unsupported-op",
    "precision",
    "bug",
    "sim-divergence",
    "timeout",
    "todo",
];

/// One reason and the tests it explains.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Group {
    category: String,
    reason: String,
    tests: BTreeSet<String>,
}

fn expected_path(silicon: bool, fusion: bool, exact: bool, accurate: bool) -> PathBuf {
    let name = format!(
        "expected-failures{}{}{}{}.toml",
        if silicon { "-silicon" } else { "" },
        if fusion { "-fusion" } else { "" },
        if exact { "-exact" } else { "" },
        if accurate { "-accurate" } else { "" }
    );
    workspace_root()
        .join("crates/burn-tt-conformance")
        .join(name)
}

/// Read the expected-failures file.
///
/// A hand parser for the one shape the file has (`[[group]]` tables of a
/// `category`, a `reason` and a `tests` array of strings), for the reason the rest
/// of xtask is dependency-free. Anything else is an error, so a typo cannot
/// quietly drop an entry.
fn read_expected(path: &Path) -> Result<Vec<Group>, String> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let bad = |n: usize, what: &str| format!("{}:{}: {what}", path.display(), n + 1);
    let mut groups: Vec<Group> = Vec::new();
    let mut in_tests = false;
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if in_tests {
            if line == "]" {
                in_tests = false;
                continue;
            }
            let name = unquote(line.trim_end_matches(','))
                .ok_or_else(|| bad(n, "expected a quoted test name"))?;
            let group = groups.last_mut().ok_or_else(|| bad(n, "no group"))?;
            if !group.tests.insert(name.clone()) {
                return Err(bad(n, &format!("`{name}` is listed twice in the group")));
            }
            continue;
        }
        if line == "[[group]]" {
            groups.push(Group {
                category: String::new(),
                reason: String::new(),
                tests: BTreeSet::new(),
            });
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| bad(n, "expected `key = value`"))?;
        let (key, value) = (key.trim(), value.trim());
        let group = groups
            .last_mut()
            .ok_or_else(|| bad(n, "a key before the first [[group]]"))?;
        match key {
            "category" => {
                let c = unquote(value).ok_or_else(|| bad(n, "expected a quoted string"))?;
                if !CATEGORIES.contains(&c.as_str()) {
                    return Err(bad(
                        n,
                        &format!("unknown category `{c}`; one of {CATEGORIES:?}"),
                    ));
                }
                group.category = c;
            }
            "reason" => {
                group.reason = unquote(value).ok_or_else(|| bad(n, "expected a quoted string"))?
            }
            "tests" if value == "[" => in_tests = true,
            _ => return Err(bad(n, &format!("unexpected `{key}`"))),
        }
    }
    if in_tests {
        return Err(format!("{}: unterminated `tests` array", path.display()));
    }
    for g in &groups {
        if g.category.is_empty() || g.reason.trim().is_empty() {
            return Err(format!(
                "{}: a group needs a category and a reason (tests: {:?})",
                path.display(),
                g.tests.iter().next()
            ));
        }
    }
    // A test explained twice has two reasons, and one of them is wrong.
    let mut seen = BTreeMap::new();
    for g in &groups {
        for t in &g.tests {
            if let Some(other) = seen.insert(t.clone(), &g.reason) {
                return Err(format!(
                    "{}: `{t}` is in two groups ({other:?} and {:?})",
                    path.display(),
                    g.reason
                ));
            }
        }
    }
    Ok(groups)
}

fn unquote(s: &str) -> Option<String> {
    let inner = s.trim().strip_prefix('"')?.strip_suffix('"')?;
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                _ => return None,
            }
        } else {
            out.push(c);
        }
    }
    Some(out)
}

fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

const HEADER: &str = "\
# Tests of Burn's own backend suite (vendor/burn-backend-tests, pinned in PINS.toml)
# that burn-tt is expected to fail, and why. `cargo xtask conformance` compares
# every run with this file: an unlisted failure is a regression or a new finding,
# and a listed test that passes is stale, so the list only shrinks. The goal is an
# empty file.
#
# Never loosen a vendored tolerance to make a test pass; a `precision` entry
# names the mechanism, and says whether a bound has been derived.
#
# Test names are `<binary>::<test path>`, as `cargo xtask conformance --list`
# prints them.
";

fn write_expected(path: &Path, groups: &[Group]) -> Result<(), String> {
    let mut out = String::from(HEADER);
    for g in groups.iter().filter(|g| !g.tests.is_empty()) {
        out.push_str(&format!(
            "\n[[group]]\ncategory = {}\nreason = {}\ntests = [\n",
            quote(&g.category),
            quote(&g.reason)
        ));
        for t in &g.tests {
            out.push_str(&format!("    {},\n", quote(t)));
        }
        out.push_str("]\n");
    }
    fs::write(path, out).map_err(|e| format!("{}: {e}", path.display()))
}

/// What happened to one test, against what the file says.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Pass,
    ExpectedFailure,
    UnexpectedFailure,
    UnexpectedPass,
}

struct Ran {
    full: String,
    verdict: Verdict,
    secs: f64,
    out: PathBuf,
}

pub fn run(args: impl Iterator<Item = String>) -> Result<(), String> {
    let o = parse(args)?;
    let root = workspace_root();

    let vendored = burn_tests::vendored_root();
    let got = burn_tests::digest(&vendored.join("tests")).map_err(|e| {
        format!("{e}\nThe Burn test suite is not in vendor/: run `cargo xtask fetch-burn-tests`.")
    })?;
    if got != pin::BURN_TESTS_CONTENT_SHA256 {
        return Err(format!(
            "{} hashes {got}, not the pinned {}. Run `cargo xtask fetch-burn-tests --force`.",
            vendored.display(),
            pin::BURN_TESTS_CONTENT_SHA256
        ));
    }
    if !o.silicon && !o.reference_flex && !root.join("vendor/libttsim_bh.so").exists() {
        return Err("vendor/libttsim_bh.so is missing: run `cargo xtask fetch-ttsim`.".into());
    }
    if !o.silicon && !o.reference_flex {
        if let Some(s) = o
            .suites
            .iter()
            .find(|s| SILICON_ONLY_SUITES.contains(&s.as_str()))
        {
            return Err(format!(
                "`{s}` needs the BF16 narrowing (PACR mode 0x105), which ttsim refuses \
                 (docs/learnings/ttsim-divergence.md, \"BF16 late narrowing\"): \
                 run it with `--silicon`, or on burn-flex with `--reference-flex`"
            ));
        }
    }

    let mut features = vec![if o.silicon {
        "burn-tt-conformance/silicon"
    } else if o.reference_flex {
        "burn-tt-conformance/reference-flex"
    } else {
        "burn-tt-conformance/run"
    }];
    if o.fusion {
        features.push("burn-tt-conformance/fusion");
    }
    let select = vec![
        "-p".to_string(),
        "burn-tt-conformance".to_string(),
        "--features".to_string(),
        features.join(","),
    ];
    let binaries: Vec<Built> = build(&root, o.release, &select)?
        .into_iter()
        .filter(|b| o.suites.is_empty() || o.suites.contains(&b.name))
        // The default selection is the ttsim lane: the BF16 binaries wait for silicon.
        .filter(|b| {
            o.silicon
                || o.reference_flex
                || !o.suites.is_empty()
                || !SILICON_ONLY_SUITES.contains(&b.name.as_str())
        })
        .collect();
    if let Some(unknown) = o
        .suites
        .iter()
        .find(|s| !binaries.iter().any(|b| &b.name == *s))
    {
        return Err(format!("no test binary named `{unknown}`"));
    }

    let mut selection: Vec<(String, Built, String)> = Vec::new();
    // Every test of the binaries built, selected or not, to tell an entry whose
    // test is gone from one a filter left out.
    let mut every: BTreeSet<String> = BTreeSet::new();
    for b in &binaries {
        for test in list(&b.exe, false)? {
            let full = format!("{}::{test}", b.name);
            every.insert(full.clone());
            if o.filters.is_empty() || o.filters.iter().any(|f| full.contains(f.as_str())) {
                selection.push((full, b.clone(), test));
            }
        }
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

    // Preflight only now: `--list` above never touches a card.
    if o.silicon {
        preflight(&[0])?;
    }
    let expected_file = expected_path(o.silicon, o.fusion, o.exact, o.accurate);
    let mut groups = read_expected(&expected_file)?;
    let reasons: BTreeMap<String, (String, String)> = groups
        .iter()
        .flat_map(|g| {
            g.tests
                .iter()
                .map(|t| (t.clone(), (g.category.clone(), g.reason.clone())))
        })
        .collect();

    let stamp = unix_secs();
    let dir = root.join("target/conformance").join(stamp.to_string());
    fs::create_dir_all(dir.join("out")).map_err(|e| format!("{}: {e}", dir.display()))?;
    println!(
        "{} tests, {} at a time; outputs in {}",
        selection.len(),
        o.jobs,
        dir.display()
    );

    // The silicon lane keeps the runner's log, fsync'd per line: a line not on disk when
    // the host dies was never written. Its stamp is the run ID the docs cite.
    let log = if o.silicon {
        let silicon_dir = root.join("target/silicon");
        fs::create_dir_all(&silicon_dir).map_err(|e| format!("{}: {e}", silicon_dir.display()))?;
        let path = silicon_dir.join(format!("{stamp}.log"));
        let mut log = Log::open(&path)?;
        println!("silicon run {stamp}, logging to {}", path.display());
        let profile = if o.release { "release" } else { "dev" };
        log.line(&format!(
            "{stamp} boot={} PROFILE {profile} conformance",
            boot_id()
        ))?;
        Some(Mutex::new(log))
    } else {
        None
    };

    let started = Instant::now();
    let ran = run_all(&o, &selection, &dir.join("out"), log.as_ref());
    let elapsed = started.elapsed();

    if o.reference_flex {
        let failed: Vec<&Ran> = ran.iter().filter(|r| r.verdict != Verdict::Pass).collect();
        let mut summary = format!(
            "burn-flex reference: {} run, {} passed, {} failed\n",
            ran.len(),
            ran.len() - failed.len(),
            failed.len()
        );
        for r in &failed {
            summary.push_str(&format!("FAIL {} {}\n", r.verdict, r.full));
        }
        print!("{summary}");
        let _ = fs::write(dir.join("summary.txt"), &summary);
        println!("summary: {}", dir.join("summary.txt").display());
        return Ok(());
    }
    if o.silicon {
        println!("silicon run {stamp}");
    }

    let mut outcomes = Vec::new();
    for r in &ran {
        let listed = reasons.contains_key(&r.full);
        outcomes.push(match (r.verdict == Verdict::Pass, listed) {
            (true, false) => Outcome::Pass,
            (true, true) => Outcome::UnexpectedPass,
            (false, true) => Outcome::ExpectedFailure,
            (false, false) => Outcome::UnexpectedFailure,
        });
    }
    let count = |want: Outcome| outcomes.iter().filter(|o| **o == want).count();

    // An entry whose test is gone: its binary was built and does not list it.
    let stale: Vec<&String> = reasons
        .keys()
        .filter(|t| {
            binaries
                .iter()
                .any(|b| t.strip_prefix(&format!("{}::", b.name)).is_some())
                && !every.contains(*t)
        })
        .collect();
    let todo: Vec<&String> = reasons
        .iter()
        .filter(|(_, (c, _))| c == "todo")
        .map(|(t, _)| t)
        .collect();

    let mut summary = String::new();
    let mut say = |s: String| {
        println!("{s}");
        summary.push_str(&s);
        summary.push('\n');
    };
    say(format!(
        "\n{} run in {:.0}s: {} passed, {} failed as expected, {} unexpected failures, {} unexpected passes",
        ran.len(),
        elapsed.as_secs_f64(),
        count(Outcome::Pass),
        count(Outcome::ExpectedFailure),
        count(Outcome::UnexpectedFailure),
        count(Outcome::UnexpectedPass),
    ));
    let mut by_category: BTreeMap<&str, usize> = BTreeMap::new();
    for (r, oc) in ran.iter().zip(&outcomes) {
        if *oc == Outcome::ExpectedFailure {
            *by_category.entry(reasons[&r.full].0.as_str()).or_default() += 1;
        }
    }
    for (c, n) in &by_category {
        say(format!("  expected failures, {c}: {n}"));
    }
    for (r, oc) in ran.iter().zip(&outcomes) {
        match oc {
            Outcome::UnexpectedFailure => say(format!(
                "UNEXPECTED {} {}\n    output: {}",
                r.verdict,
                r.full,
                r.out.display()
            )),
            Outcome::UnexpectedPass => say(format!(
                "UNEXPECTED PASS {} (listed as: {}); remove it from the expected-failures file",
                r.full, reasons[&r.full].1
            )),
            _ => {}
        }
    }
    for t in &stale {
        say(format!("STALE ENTRY {t}: no such test any more"));
    }
    if !todo.is_empty() {
        say(format!(
            "{} untriaged entries (category `todo`): give each a category and a reason",
            todo.len()
        ));
    }
    let _ = fs::write(dir.join("summary.txt"), &summary);

    if o.update {
        let now_passing: BTreeSet<&String> = ran
            .iter()
            .zip(&outcomes)
            .filter(|(_, oc)| **oc == Outcome::UnexpectedPass)
            .map(|(r, _)| &r.full)
            .collect();
        for g in &mut groups {
            g.tests
                .retain(|t| !now_passing.contains(t) && !stale.contains(&t));
        }
        let new: BTreeSet<String> = ran
            .iter()
            .zip(&outcomes)
            .filter(|(_, oc)| **oc == Outcome::UnexpectedFailure)
            .map(|(r, _)| r.full.clone())
            .collect();
        if !new.is_empty() {
            match groups.iter_mut().find(|g| g.category == "todo") {
                Some(g) => g.tests.extend(new),
                None => groups.push(Group {
                    category: "todo".into(),
                    reason: "TODO: not triaged yet".into(),
                    tests: new,
                }),
            }
        }
        write_expected(&expected_file, &groups)?;
        println!("rewrote {}", expected_file.display());
        return Ok(());
    }

    if count(Outcome::UnexpectedFailure) + count(Outcome::UnexpectedPass) + stale.len() > 0
        || !todo.is_empty()
    {
        return Err("the run does not match the expected-failures file".into());
    }
    Ok(())
}

/// Run the selection `o.jobs` at a time, each test in its own process.
fn run_all(
    o: &Opts,
    selection: &[(String, Built, String)],
    out_dir: &Path,
    log: Option<&Mutex<Log>>,
) -> Vec<Ran> {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Ran>>> = Mutex::new((0..selection.len()).map(|_| None).collect());
    let mut env: Vec<(&str, String)> = vec![("TT_REPORT", "1".into())];
    if o.strict {
        env.push(("TT_STRICT", "1".into()));
    }
    if o.exact {
        env.push(("TT_EXACT_MATMUL", "1".into()));
    }
    if o.accurate {
        env.push(("TT_ACCURATE_MATMUL", "1".into()));
    }
    std::thread::scope(|scope| {
        for _ in 0..o.jobs.min(selection.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some((full, bin, test)) = selection.get(i) else {
                    return;
                };
                // Long names are cut, and the index keeps them distinct.
                let stem: String = full.replace("::", "__").chars().take(150).collect();
                let out = out_dir.join(format!("{i:04}-{stem}.txt"));
                let started = Instant::now();
                if let Some(log) = log {
                    let _ = log
                        .lock()
                        .unwrap()
                        .line(&format!("{} START {full}", unix_secs()));
                }
                let verdict = run_one(&bin.exe, test, &bin.dir, &env, o.timeout, &out)
                    // A harness failure is not a pass.
                    .unwrap_or(Verdict::Fail);
                if let Some(log) = log {
                    let _ = log.lock().unwrap().line(&format!(
                        "{} END {full} {verdict} {:.2}s",
                        unix_secs(),
                        started.elapsed().as_secs_f64()
                    ));
                }
                let ran = Ran {
                    full: full.clone(),
                    verdict,
                    secs: started.elapsed().as_secs_f64(),
                    out,
                };
                results.lock().unwrap()[i] = Some(ran);
                let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                if n % 200 == 0 || n == selection.len() {
                    println!("  {n}/{}", selection.len());
                }
            });
        }
    });
    let ran: Vec<Ran> = results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|r| r.expect("every test ran"))
        .collect();
    // Suite order is kept, so a failure is found where it ran; name the slowest.
    let slowest = ran
        .iter()
        .max_by(|a, b| a.secs.total_cmp(&b.secs))
        .map(|r| format!("slowest: {:.1}s {}", r.secs, r.full));
    if let Some(s) = slowest {
        println!("{s}");
    }
    ran
}
