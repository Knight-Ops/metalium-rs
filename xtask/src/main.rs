//! Build orchestration: fetching the simulator, building firmware images.
//!
//! Deliberately dependency-free — it shells out to `curl` and `sha256sum` rather
//! than pulling an HTTP stack and a hash crate into the workspace for two calls.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// Kept in sync with `PINS.toml`, which is the human-readable record.
mod pin {
    pub const TTSIM_TAG: &str = "v1.10.9";
    pub const TTSIM_ASSET: &str = "libttsim_bh.so";
    pub const TTSIM_SHA256: &str =
        "e6ed2da11718683738d43f14a0bf4f13285b8621697b3165c1eaa240d36cfad5";

    /// The tt-metal commit `BackendConfiguration.md:17` cites for `cfg_defines.h`.
    pub const TT_METAL_REV: &str = "81989dcdb8f9b340c932ae7a71a346f4f08703eb";
    pub const CFG_DEFINES_SHA256: &str =
        "bc2636abc3ea04e6ca322923e6f2713b242857d021d8ef39b9180a14bcb1a5b8";
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let cmd = args.next();
    let result = match cmd.as_deref() {
        Some("fetch-ttsim") => fetch_ttsim(args.any(|a| a == "--force")),
        Some("check-no-sim-in-ship") => check_no_sim_in_ship(),
        Some("gen-cfg") => gen_cfg(args.any(|a| a == "--check")),
        Some(other) => Err(format!("unknown task `{other}`\n\n{USAGE}")),
        None => Err(USAGE.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("xtask: {msg}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "\
usage: cargo xtask <task>

tasks:
  fetch-ttsim [--force]   download the pinned libttsim_bh.so into vendor/
  gen-cfg [--check]       regenerate tt-isa's backend-configuration field table
                          from the pinned cfg_defines.h; --check fails if the
                          committed file is out of date rather than rewriting it
  check-no-sim-in-ship    assert tt-ttsim is absent from shippable dependency graphs";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives directly below the workspace root")
        .to_path_buf()
}

/// Download the pinned simulator build, verifying its hash.
///
/// The `.so` is not committed — it is a 244 KiB binary blob with its own release
/// cadence — but it is *pinned*, because it is the oracle every gate is measured
/// against. An unnoticed change to it silently changes what the test suite proves.
fn fetch_ttsim(force: bool) -> Result<(), String> {
    let dest = workspace_root().join("vendor").join(pin::TTSIM_ASSET);

    if dest.exists() && !force {
        match sha256(&dest) {
            Ok(h) if h == pin::TTSIM_SHA256 => {
                println!("{} is already present and matches the pin", dest.display());
                return Ok(());
            }
            Ok(h) => {
                return Err(format!(
                    "{} exists but hashes {h}, not the pinned {}.\n\
                     Re-run with --force to replace it, or update PINS.toml if the bump \
                     is intentional — note that doing so invalidates every gate until \
                     they are re-run.",
                    dest.display(),
                    pin::TTSIM_SHA256
                ))
            }
            Err(e) => return Err(e),
        }
    }

    let url = format!(
        "https://github.com/tenstorrent/ttsim/releases/download/{}/{}",
        pin::TTSIM_TAG,
        pin::TTSIM_ASSET
    );
    println!("fetching {url}");

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }

    // Download beside the destination, then rename: a failed or interrupted fetch
    // must not leave a truncated .so that later hashes as "wrong version".
    let tmp = dest.with_extension("so.partial");
    let status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--max-time",
            "300",
            "-o",
        ])
        .arg(&tmp)
        .arg(&url)
        .status()
        .map_err(|e| format!("could not run curl: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("curl failed downloading {url}"));
    }

    let got = sha256(&tmp)?;
    if got != pin::TTSIM_SHA256 {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!(
            "hash mismatch for {}:\n  expected {}\n  got      {got}\n\
             The pinned release asset should be immutable, so this means either the \
             download was corrupted or the tag was moved.",
            pin::TTSIM_ASSET,
            pin::TTSIM_SHA256
        ));
    }

    std::fs::rename(&tmp, &dest).map_err(|e| format!("renaming into place: {e}"))?;
    println!("wrote {} ({})", dest.display(), pin::TTSIM_TAG);
    Ok(())
}

fn sha256(path: &Path) -> Result<String, String> {
    let out = Command::new("sha256sum")
        .arg(path)
        .output()
        .map_err(|e| format!("could not run sha256sum: {e}"))?;
    if !out.status.success() {
        return Err(format!("sha256sum failed on {}", path.display()));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .split_whitespace()
        .next()
        .map(str::to_string)
        .ok_or_else(|| "sha256sum produced no output".to_string())
}

/// Assert that the simulator binding cannot reach a shipped artifact.
///
/// `tt-ttsim` binds a third-party `.so`. Binding it for development does not
/// compromise the native-Rust goal; shipping it would. The constraint is only real
/// if something checks, so this is that check.
fn check_no_sim_in_ship() -> Result<(), String> {
    const SHIPPABLE: &[&str] = &["tt-isa", "tt-device"];
    let root = workspace_root();
    let mut violations = Vec::new();

    for crate_name in SHIPPABLE {
        let out = Command::new("cargo")
            .args([
                "tree",
                "--package",
                crate_name,
                "--edges",
                "normal",
                "--prefix",
                "none",
            ])
            .current_dir(&root)
            .output()
            .map_err(|e| format!("could not run cargo tree: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "cargo tree failed for {crate_name}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        let tree = String::from_utf8_lossy(&out.stdout);
        for line in tree.lines() {
            let name = line.split_whitespace().next().unwrap_or_default();
            if name == "tt-ttsim" || name == "tt-ttsim-sys" {
                violations.push(format!("{crate_name} depends on {name}"));
            }
        }
    }

    if violations.is_empty() {
        println!("ok: no shippable crate depends on the simulator binding");
        Ok(())
    } else {
        Err(format!(
            "the simulator binding leaked into a shippable crate:\n  {}",
            violations.join("\n  ")
        ))
    }
}

/// Regenerate `tt-isa`'s backend-configuration field table.
///
/// `cfg_defines.h` holds 827 fields across seven sections. The plan's rule is
/// "generate, never transcribe", and this is that generator.
///
/// The output is **committed** rather than produced by a build script. That keeps
/// `tt-isa` free of build dependencies and buildable for `riscv32im` with nothing
/// but rustc, makes the table reviewable, and turns a pin bump into a diff someone
/// can read. `--check` regenerates into memory and compares, so CI can prove the
/// committed file matches the pinned header.
fn gen_cfg(check_only: bool) -> Result<(), String> {
    let root = workspace_root();
    let header = root.join("vendor").join("cfg_defines.h");
    if !header.exists() {
        fetch_cfg_defines(&header)?;
    }
    let got = sha256(&header)?;
    if got != pin::CFG_DEFINES_SHA256 {
        return Err(format!(
            "{} hashes {got}, not the pinned {}.\n\
             Delete it to re-fetch, or update PINS.toml and xtask's pin module if \
             the bump is intentional.",
            header.display(),
            pin::CFG_DEFINES_SHA256
        ));
    }

    let source = std::fs::read_to_string(&header)
        .map_err(|e| format!("reading {}: {e}", header.display()))?;
    let rendered = render_cfg(&parse_cfg_defines(&source)?)?;

    let dest = root.join("crates/tt-isa/src/cfg/generated.rs");

    // Format the output rather than trying to emit rustfmt-clean code by hand.
    // Long field names push some lines past the width limit, and guessing where
    // rustfmt would wrap them is a losing game -- it also makes `--check` compare
    // two things produced the same way, so it cannot fail for cosmetic reasons.
    let generated = rustfmt(&rendered, &root)?;

    if check_only {
        let current = std::fs::read_to_string(&dest)
            .map_err(|e| format!("reading {}: {e}", dest.display()))?;
        if current == generated {
            println!(
                "ok: {} is up to date with the pinned header",
                dest.display()
            );
            Ok(())
        } else {
            Err(format!(
                "{} is out of date with the pinned cfg_defines.h.\n\
                 Run `cargo xtask gen-cfg` and commit the result.",
                dest.display()
            ))
        }
    } else {
        std::fs::write(&dest, &generated)
            .map_err(|e| format!("writing {}: {e}", dest.display()))?;
        println!("wrote {} ({} bytes)", dest.display(), generated.len());
        Ok(())
    }
}

/// Run the generated source through rustfmt.
fn rustfmt(source: &str, root: &Path) -> Result<String, String> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new("rustfmt")
        .args(["--edition", "2021", "--emit", "stdout", "--quiet"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("could not run rustfmt: {e}"))?;
    child
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(source.as_bytes())
        .map_err(|e| format!("writing to rustfmt: {e}"))?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("waiting for rustfmt: {e}"))?;
    if !out.status.success() {
        return Err("rustfmt rejected the generated source".into());
    }
    String::from_utf8(out.stdout).map_err(|e| format!("rustfmt produced invalid UTF-8: {e}"))
}

fn fetch_cfg_defines(dest: &Path) -> Result<(), String> {
    let url = format!(
        "https://raw.githubusercontent.com/tenstorrent/tt-metal/{}/tt_metal/hw/inc/blackhole/cfg_defines.h",
        pin::TT_METAL_REV
    );
    println!("fetching {url}");
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    let tmp = dest.with_extension("h.partial");
    let status = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--max-time",
            "120",
            "-o",
        ])
        .arg(&tmp)
        .arg(&url)
        .status()
        .map_err(|e| format!("could not run curl: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("curl failed downloading {url}"));
    }
    std::fs::rename(&tmp, dest).map_err(|e| format!("renaming into place: {e}"))
}

use std::collections::BTreeMap;

/// One parsed field.
struct Field {
    name: String,
    section: String,
    addr32: u32,
    shamt: u32,
    /// Hex digits of the mask, unparsed — two fields have 128-bit masks.
    mask: String,
}

/// Parse `#define NAME_{ADDR32,SHAMT,MASK} VALUE`, tracking the current section.
///
/// The file is machine-generated by Tenstorrent's `genCfgRegs.py`, so the format
/// is regular: every field has exactly those three defines plus an `_RMW`
/// convenience macro, which is skipped because it carries no information the
/// other three do not.
/// A field under construction: its section, then `ADDR32`, `SHAMT` and `MASK` as
/// each is encountered. The header emits them on consecutive lines, but nothing
/// in the format guarantees that, so they are collected independently.
type PartialField = (String, Option<u32>, Option<u32>, Option<String>);

fn parse_cfg_defines(source: &str) -> Result<CfgTable, String> {
    let mut section = String::new();
    let mut parts: BTreeMap<String, PartialField> = BTreeMap::new();
    let mut cfg_state_size = None;
    let mut thd_state_size = None;
    let mut section_bases: BTreeMap<String, u32> = BTreeMap::new();

    for line in source.lines() {
        if let Some(rest) = line.strip_prefix("// Registers for ") {
            section = rest.trim().to_string();
            continue;
        }
        let Some(rest) = line.strip_prefix("#define ") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let (Some(name), Some(value)) = (it.next(), it.next()) else {
            continue;
        };

        match name {
            "CFG_STATE_SIZE" => {
                cfg_state_size = parse_int(value);
                continue;
            }
            "THD_STATE_SIZE" => {
                thd_state_size = parse_int(value);
                continue;
            }
            _ => {}
        }

        // `_RMW` expands to the other three; nothing to learn from it.
        let Some((stem, kind)) = name.rsplit_once('_') else {
            continue;
        };
        if !matches!(kind, "ADDR32" | "SHAMT" | "MASK") {
            continue;
        }
        // `<SECTION>_CFGREG_BASE_ADDR32` is the first word index of a section,
        // not a field: it has no SHAMT or MASK. There is exactly one per section,
        // and together they partition the Config space.
        if let Some(sec) = stem.strip_suffix("_CFGREG_BASE") {
            if kind == "ADDR32" {
                if let Some(v) = parse_int(value) {
                    section_bases.insert(sec.to_string(), v);
                }
            }
            continue;
        }
        let entry = parts
            .entry(stem.to_string())
            .or_insert_with(|| (section.clone(), None, None, None));
        match kind {
            "ADDR32" => entry.1 = parse_int(value),
            "SHAMT" => entry.2 = parse_int(value),
            "MASK" => entry.3 = Some(value.to_string()),
            _ => unreachable!(),
        }
    }

    let cfg_state_size = cfg_state_size.ok_or("CFG_STATE_SIZE not found")?;
    let thd_state_size = thd_state_size.ok_or("THD_STATE_SIZE not found")?;

    let mut fields = Vec::new();
    for (name, (section, addr32, shamt, mask)) in parts {
        let (Some(addr32), Some(shamt), Some(mask)) = (addr32, shamt, mask) else {
            return Err(format!(
                "field `{name}` is missing one of ADDR32/SHAMT/MASK"
            ));
        };
        if section.is_empty() {
            return Err(format!("field `{name}` appears before any section header"));
        }
        fields.push(Field {
            name,
            section,
            addr32,
            shamt,
            mask,
        });
    }
    if fields.is_empty() {
        return Err("parsed no fields; has the header format changed?".into());
    }
    for f in &fields {
        if !section_bases.contains_key(&f.section) {
            return Err(format!(
                "section `{}` has no _CFGREG_BASE_ADDR32",
                f.section
            ));
        }
    }
    Ok(CfgTable {
        fields,
        cfg_state_size,
        thd_state_size,
        section_bases,
    })
}

struct CfgTable {
    fields: Vec<Field>,
    cfg_state_size: u32,
    thd_state_size: u32,
    /// First `Config` word index of each section, from `<SECTION>_CFGREG_BASE_ADDR32`.
    section_bases: BTreeMap<String, u32>,
}

fn parse_int(s: &str) -> Option<u32> {
    let s = s.trim();
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u32::from_str_radix(hex, 16).ok(),
        None => s.parse().ok(),
    }
}

/// Emit the Rust module.
fn render_cfg(table: &CfgTable) -> Result<String, String> {
    use std::fmt::Write;

    // Section name -> (module name, is ThreadConfig).
    let module_of =
        |section: &str| -> (String, bool) { (section.to_ascii_lowercase(), section == "THREAD") };

    let mut by_module: BTreeMap<String, Vec<&Field>> = BTreeMap::new();
    for f in &table.fields {
        by_module
            .entry(module_of(&f.section).0)
            .or_default()
            .push(f);
    }

    let mut out = String::new();
    writeln!(
        out,
        "//! Tensix backend-configuration fields, generated from `cfg_defines.h`.\n\
         //!\n\
         //! **Do not edit.** Regenerate with `cargo xtask gen-cfg`; CI checks that this\n\
         //! file matches the header pinned in `PINS.toml`.\n\
         //!\n\
         //! Source: tt-metal `{}`, `tt_metal/hw/inc/blackhole/cfg_defines.h`\n\
         //! ({} fields across {} sections).\n\
         //!\n\
         //! Field names are kept exactly as `cfg_defines.h` spells them, so that a\n\
         //! name in the specification can be searched for here without translation.\n\
         //! That is why this module allows non-upper-case globals.\n\
         #![allow(non_upper_case_globals)]\n\
         #![allow(clippy::unreadable_literal)]\n",
        pin::TT_METAL_REV,
        table.fields.len(),
        by_module.len(),
    )
    .unwrap();

    writeln!(
        out,
        "use super::{{ConfigField, ConfigSpan, ThreadConfigField}};\n"
    )
    .unwrap();
    writeln!(
        out,
        "/// `uint32_t Config[2][CFG_STATE_SIZE * 4]` — the multiplier is in the\n\
         /// declaration, so one bank holds `CFG_STATE_SIZE * 4` words.\n\
         pub const CFG_STATE_SIZE: u32 = {};\n",
        table.cfg_state_size
    )
    .unwrap();
    writeln!(
        out,
        "/// `struct {{uint16_t Value, Padding;}} ThreadConfig[3][THD_STATE_SIZE]`.\n\
         pub const THD_STATE_SIZE: u32 = {};\n",
        table.thd_state_size
    )
    .unwrap();

    let mut all_config = Vec::new();
    let mut all_thread = Vec::new();
    let mut all_spans = Vec::new();
    let mut per_section: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();

    for (module, fields) in &by_module {
        let is_thread = fields[0].section == "THREAD";
        let section = &fields[0].section;
        writeln!(
            out,
            "/// `// Registers for {section}` — {} fields, indexing {}.",
            fields.len(),
            if is_thread {
                "`ThreadConfig` (write with `SETC16`)"
            } else {
                "`Config` (write with `WRCFG`)"
            }
        )
        .unwrap();
        writeln!(out, "pub mod {module} {{").unwrap();
        writeln!(out, "    use super::*;\n").unwrap();
        let base = table.section_bases[section];
        writeln!(
            out,
            "    /// First word index of this section, from `{section}_CFGREG_BASE_ADDR32`.\n    \
             pub const CFGREG_BASE: u16 = {base};\n"
        )
        .unwrap();

        for f in fields {
            let addr32 = f.addr32;
            let shamt = f.shamt;
            let hex = f.mask.trim_start_matches("0x").trim_start_matches("0X");
            let wide = hex.len() > 8;

            if wide {
                // A whole-register aggregate rather than a bitfield: mask is all
                // ones and the shift is zero. Anything else would be a parse bug.
                if shamt != 0 || hex.chars().any(|c| c != 'f' && c != 'F') {
                    return Err(format!(
                        "{}: mask {} is wider than 32 bits but is not a whole-word run",
                        f.name, f.mask
                    ));
                }
                let words = (hex.len() / 8) as u32;
                writeln!(
                    out,
                    "    /// `Config[{addr32}..{}]` — a {}-bit aggregate, not a bitfield.",
                    addr32 + words,
                    words * 32
                )
                .unwrap();
                writeln!(
                    out,
                    "    pub const {}: ConfigSpan = ConfigSpan::new({addr32}, {words});",
                    f.name
                )
                .unwrap();
                all_spans.push((module.clone(), f.name.clone()));
            } else if is_thread {
                writeln!(
                    out,
                    "    pub const {}: ThreadConfigField = ThreadConfigField::new({addr32}, {shamt}, 0x{hex});",
                    f.name
                )
                .unwrap();
                all_thread.push((module.clone(), f.name.clone()));
            } else {
                writeln!(
                    out,
                    "    pub const {}: ConfigField = ConfigField::new({addr32}, {shamt}, 0x{hex});",
                    f.name
                )
                .unwrap();
                all_config.push((module.clone(), f.name.clone()));
                per_section
                    .entry(module.clone())
                    .or_default()
                    .push((module.clone(), f.name.clone()));
            }
        }
        writeln!(out, "}}\n").unwrap();
    }

    // Flat tables, so invariants can be asserted over every field at once rather
    // than a sample. This is what catches a misclassified section.
    let emit_table =
        |out: &mut String, name: &str, ty: &str, items: &[(String, String)], doc: &str| {
            writeln!(out, "/// {doc}").unwrap();
            writeln!(out, "pub static {name}: &[(&str, {ty})] = &[").unwrap();
            for (module, field) in items {
                writeln!(out, "    (\"{field}\", {module}::{field}),").unwrap();
            }
            writeln!(out, "];\n").unwrap();
        };
    emit_table(
        &mut out,
        "ALL_CONFIG_FIELDS",
        "ConfigField",
        &all_config,
        "Every `Config` bitfield, for table-wide invariant checks.",
    );
    emit_table(
        &mut out,
        "ALL_THREAD_CONFIG_FIELDS",
        "ThreadConfigField",
        &all_thread,
        "Every `ThreadConfig` bitfield, for table-wide invariant checks.",
    );
    emit_table(
        &mut out,
        "ALL_CONFIG_SPANS",
        "ConfigSpan",
        &all_spans,
        "Every multi-word `Config` aggregate.",
    );

    // Per-section tables, so the section-partitioning invariant can be checked.
    // A field attributed to the wrong section is still in bounds for `Config` as
    // a whole, so only a per-section check catches it.
    for (module, items) in &per_section {
        emit_table(
            &mut out,
            &format!("{}_FIELDS", module.to_ascii_uppercase()),
            "ConfigField",
            items,
            &format!("Every `Config` bitfield in the `{module}` section."),
        );
    }

    Ok(out)
}
