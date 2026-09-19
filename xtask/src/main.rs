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
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let cmd = args.next();
    let result = match cmd.as_deref() {
        Some("fetch-ttsim") => fetch_ttsim(args.any(|a| a == "--force")),
        Some("check-no-sim-in-ship") => check_no_sim_in_ship(),
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
        std::fs::create_dir_all(parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }

    // Download beside the destination, then rename: a failed or interrupted fetch
    // must not leave a truncated .so that later hashes as "wrong version".
    let tmp = dest.with_extension("so.partial");
    let status = Command::new("curl")
        .args(["--fail", "--location", "--silent", "--show-error", "--max-time", "300", "-o"])
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
            .args(["tree", "--package", crate_name, "--edges", "normal", "--prefix", "none"])
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
