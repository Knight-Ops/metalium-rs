//! `cargo xtask fw-size`: each shipped firmware image's bytes against its slot, in
//! every shipped feature set, and where the mover's hot path sits.
//!
//! The firmware workspace is built here as `tt-firmware-images/build.rs` builds it
//! (release, the host's `CARGO_*` and wrapper variables stripped, so the bytes are
//! a build's and not a clippy run's), once a feature set, into
//! `target/fw-size/<set>`. An image's span comes from the absolute symbols
//! `sections.x` defines (`__image_base`, `__image_limit`) and the end of its last
//! L1 section (`__bss_end`), so the slot is whatever the link script says and this
//! tool holds no copy of it.
//!
//! Sizes move by tens of bytes with unrelated changes (inlining decisions shift):
//! quote this tool's output for a change, never an estimate
//! (`docs/learnings/silicon-operating-notes.md`, "RISCV B's image budget").

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::util::workspace_root;

const TARGET: &str = "riscv32im-unknown-none-elf";

/// The feature sets a shipped image is built in: the read modes, and each with
/// the profile mode's stamping. `profile,encoded-guard` is the largest B image.
const SETS: &[(&str, &str)] = &[
    ("safe", ""),
    ("fast-reads", "fast-reads"),
    ("encoded-guard", "encoded-guard"),
    ("profile", "profile"),
    ("profile+fast-reads", "profile,fast-reads"),
    ("profile+encoded-guard", "profile,encoded-guard"),
];

/// The images loaded together on a Tensix tile in ordinary runs.
const IMAGES: &[&str] = &["dm_b", "dm_nc", "role_t0", "role_t1", "role_t2"];

/// The B mover's per-entry and per-tile path, in `sections.x`'s order, then the
/// cold paths the C1 placement lesson keeps out of it. Matched against the
/// demangled name's last path segment.
const HOT: &[&str] = &[
    "run_entries",
    "exec",
    "record_entry",
    "issue",
    "run",
    "issue_dram_on",
    "issue_on",
    "issue_write",
    "issue_l1_write",
    "credit",
    "multicast_l1",
    "issue_read",
    "run_record",
    "pump",
    "gather_by_column",
    "walk_by_column",
    "gather_span",
];

const USAGE: &str = "\
usage: cargo xtask fw-size [--set <name>]... [--symbols] [--top <n>]

  Builds crates/tt-firmware in each shipped feature set and prints each loaded
  image's bytes, its slot and the room left; then the B mover's hot-path
  functions (address and size) in the safe build.

  --set <name>   only these sets (safe, fast-reads, encoded-guard, profile,
                 profile+fast-reads, profile+encoded-guard); repeatable
  --symbols      also list the largest functions of dm_b and of role_t1
  --top <n>      how many functions --symbols lists (default 30)";

pub fn run(args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut sets: Vec<&(&str, &str)> = Vec::new();
    let mut symbols = false;
    let mut top = 30usize;
    let mut args = args.peekable();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--set" => {
                let name = args.next().ok_or("--set needs a name")?;
                let set = SETS
                    .iter()
                    .find(|(n, _)| *n == name)
                    .ok_or_else(|| format!("unknown set `{name}`\n\n{USAGE}"))?;
                sets.push(set);
            }
            "--symbols" => symbols = true,
            "--top" => {
                top = args
                    .next()
                    .ok_or("--top needs a number")?
                    .parse()
                    .map_err(|e| format!("--top: {e}"))?;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown option `{other}`\n\n{USAGE}")),
        }
    }
    if sets.is_empty() {
        sets = SETS.iter().collect();
    }

    let root = workspace_root();
    let nm = llvm_tool("llvm-nm")?;
    let mut built = BTreeMap::new();
    for (name, features) in &sets {
        built.insert(*name, build(&root, name, features)?);
    }

    println!(
        "{:<24} {:<8} {:>8} {:>8} {:>7}",
        "set", "image", "bytes", "slot", "room"
    );
    for (name, _) in &sets {
        let dir = &built[name];
        for image in IMAGES {
            let span = span(&nm, &dir.join(image))?;
            println!(
                "{:<24} {:<8} {:>8} {:>8} {:>7}",
                name,
                image,
                span.used(),
                span.slot(),
                span.room()
            );
        }
    }

    // The hot path in the first set asked for (the safe build by default).
    let first = sets[0].0;
    let dm_b = built[first].join("dm_b");
    let funcs = functions(&nm, &dm_b)?;
    println!("\ndm_b ({first}): the hot path and its cold neighbours, by address");
    println!("{:>10} {:>7}  name", "address", "bytes");
    let mut hot: Vec<&Func> = funcs
        .iter()
        .filter(|f| HOT.contains(&last_segment(&f.name)))
        .collect();
    hot.sort_by_key(|f| f.addr);
    for f in hot {
        println!("{:>#10x} {:>7}  {}", f.addr, f.size, f.name);
    }

    if symbols {
        for image in ["dm_b", "role_t1"] {
            let mut funcs = functions(&nm, &built[first].join(image))?;
            funcs.sort_by_key(|f| std::cmp::Reverse(f.size));
            println!("\n{image} ({first}): the {top} largest functions");
            println!("{:>10} {:>7}  name", "address", "bytes");
            for f in funcs.iter().take(top) {
                println!("{:>#10x} {:>7}  {}", f.addr, f.size, f.name);
            }
        }
    }
    Ok(())
}

/// Build the firmware with `features` into its own target directory, and return
/// the directory holding the ELFs.
fn build(root: &Path, set: &str, features: &str) -> Result<PathBuf, String> {
    let firmware = root.join("crates/tt-firmware");
    let target_dir = root.join("target/fw-size").join(set);
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(&firmware)
        .args(["build", "--release", "--quiet", "--target", TARGET])
        .arg("--target-dir")
        .arg(&target_dir);
    if !features.is_empty() {
        cmd.args(["--features", features]);
    }
    // As `tt-firmware-images/build.rs` does: the nested build must not inherit
    // the host's target directory, profile or wrappers (a clippy wrapper made
    // every image ~900 bytes larger).
    for (key, _) in std::env::vars() {
        if key.starts_with("CARGO_")
            || key == "RUSTFLAGS"
            || key == "RUSTC_WRAPPER"
            || key == "RUSTC_WORKSPACE_WRAPPER"
        {
            cmd.env_remove(key);
        }
    }
    let status = cmd
        .status()
        .map_err(|e| format!("could not run cargo for the firmware ({set}): {e}"))?;
    if !status.success() {
        return Err(format!("the firmware build failed for set `{set}`"));
    }
    Ok(target_dir.join(TARGET).join("release"))
}

struct Span {
    base: u64,
    limit: u64,
    end: u64,
}

impl Span {
    fn used(&self) -> u64 {
        self.end - self.base
    }
    fn slot(&self) -> u64 {
        self.limit - self.base
    }
    fn room(&self) -> i64 {
        self.limit as i64 - self.end as i64
    }
}

fn span(nm: &Path, elf: &Path) -> Result<Span, String> {
    let out = run_nm(nm, elf, &[])?;
    let mut found = BTreeMap::new();
    for line in out.lines() {
        let mut parts = line.split_whitespace();
        let (Some(addr), Some(_kind), Some(name)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if matches!(name, "__image_base" | "__image_limit" | "__bss_end") {
            let v = u64::from_str_radix(addr, 16)
                .map_err(|e| format!("{}: bad address for {name}: {e}", elf.display()))?;
            found.insert(name.to_string(), v);
        }
    }
    let get = |n: &str| {
        found.get(n).copied().ok_or_else(|| {
            format!(
                "{} has no `{n}` symbol (sections.x defines it; is this an old build?)",
                elf.display()
            )
        })
    };
    Ok(Span {
        base: get("__image_base")?,
        limit: get("__image_limit")?,
        end: get("__bss_end")?,
    })
}

struct Func {
    addr: u64,
    size: u64,
    name: String,
}

/// Every sized text symbol of `elf`, demangled.
fn functions(nm: &Path, elf: &Path) -> Result<Vec<Func>, String> {
    let out = run_nm(nm, elf, &["-S", "-C"])?;
    let mut funcs = Vec::new();
    for line in out.lines() {
        // "<addr> <size> <kind> <name with spaces>"
        let mut parts = line.splitn(4, ' ');
        let (Some(addr), Some(size), Some(kind), Some(name)) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        if !matches!(kind, "t" | "T") {
            continue;
        }
        let (Ok(addr), Ok(size)) = (u64::from_str_radix(addr, 16), u64::from_str_radix(size, 16))
        else {
            continue;
        };
        funcs.push(Func {
            addr,
            size,
            name: name.to_string(),
        });
    }
    Ok(funcs)
}

/// The last path segment of a demangled name, generics and closures aside:
/// `tt_isa::dm::record::gather_by_column` -> `gather_by_column`.
fn last_segment(name: &str) -> &str {
    let name = name
        .split('<')
        .next()
        .unwrap_or(name)
        .trim_end_matches("::");
    name.rsplit("::").next().unwrap_or(name)
}

fn run_nm(nm: &Path, elf: &Path, flags: &[&str]) -> Result<String, String> {
    let out = Command::new(nm)
        .args(flags)
        .arg(elf)
        .output()
        .map_err(|e| format!("could not run llvm-nm: {e}"))?;
    if !out.status.success() {
        return Err(format!("llvm-nm failed on {}", elf.display()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// One of the `llvm-tools` binaries shipped with the active toolchain.
fn llvm_tool(name: &str) -> Result<PathBuf, String> {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let out = Command::new(&rustc)
        .args(["--print", "sysroot"])
        .output()
        .map_err(|e| format!("could not ask rustc for its sysroot: {e}"))?;
    let sysroot = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let vv = Command::new(&rustc)
        .arg("-vV")
        .output()
        .map_err(|e| format!("could not ask rustc for its host: {e}"))?;
    let host = String::from_utf8_lossy(&vv.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("host: ").map(str::to_string))
        .ok_or("rustc -vV named no host")?;
    let path = Path::new(&sysroot)
        .join("lib/rustlib")
        .join(host)
        .join("bin")
        .join(name);
    if path.exists() {
        Ok(path)
    } else {
        Err(format!(
            "{name} not found at {}; install it with `rustup component add llvm-tools`",
            path.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::last_segment;

    #[test]
    fn last_segment_drops_paths_and_generics() {
        assert_eq!(
            last_segment("tt_isa::dm::record::gather_by_column"),
            "gather_by_column"
        );
        assert_eq!(last_segment("dm_b::run_record"), "run_record");
        assert_eq!(
            last_segment("tt_isa::dm::record::expand_for::<true, false, dm_b::run_record::{closure#2}>::{closure#0}"),
            "expand_for"
        );
    }
}
