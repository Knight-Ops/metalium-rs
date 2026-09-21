//! The check that keeps the simulator binding out of shippable crates.

use std::process::Command;

use crate::util::workspace_root;

/// Crates whose dependency tree must stay free of the simulator binding.
const SHIPPABLE: &[&str] = &["tt-isa", "tt-device", "tt-layout"];

/// Crates that exist only for development, and so are exempt.
///
/// `tt-ttsim` and `tt-ttsim-sys` *are* the binding; `tt-tests` holds the gates that
/// drive it; `xtask` is build tooling.
const DEV_ONLY: &[&str] = &["tt-ttsim", "tt-ttsim-sys", "tt-tests", "xtask"];

/// Every workspace member is classified as shippable or dev-only.
///
/// Without this, [`SHIPPABLE`] is a list someone has to remember to extend, and a
/// new crate would be silently unchecked -- which is the failure mode this whole
/// check exists to prevent, one level up. A new member fails here until it is named.
fn check_every_member_is_classified(root: &std::path::Path) -> Result<(), String> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml"))
        .map_err(|e| format!("could not read the workspace manifest: {e}"))?;

    // The `members` array of the `[workspace]` table. Hand-parsed for the same
    // reason the rest of xtask is: this crate has no dependencies.
    let start = manifest
        .find("members = [")
        .ok_or("the workspace manifest has no `members` list")?;
    let rest = &manifest[start..];
    let end = rest.find(']').ok_or("`members` is not terminated")?;
    let members: Vec<&str> = rest[..end]
        .split('"')
        .skip(1)
        .step_by(2)
        .map(|path| path.rsplit('/').next().unwrap_or(path))
        .collect();

    if members.is_empty() {
        return Err("parsed no workspace members, so this check proves nothing".into());
    }

    let unclassified: Vec<&str> = members
        .iter()
        .copied()
        .filter(|m| !SHIPPABLE.contains(m) && !DEV_ONLY.contains(m))
        .collect();
    if !unclassified.is_empty() {
        return Err(format!(
            "workspace members are neither shippable nor dev-only, so nothing \
             checks them: {}\nAdd each to SHIPPABLE or DEV_ONLY in xtask/src/ship.rs.",
            unclassified.join(", ")
        ));
    }
    Ok(())
}

/// Assert that the simulator binding cannot reach a shipped artifact.
///
/// `tt-ttsim` binds a third-party `.so`. Binding it for development does not
/// compromise the native-Rust goal; shipping it would. The constraint is only real
/// if something checks, so this is that check.
pub fn check_no_sim_in_ship() -> Result<(), String> {
    let root = workspace_root();
    let mut violations = Vec::new();

    check_every_member_is_classified(&root)?;

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
