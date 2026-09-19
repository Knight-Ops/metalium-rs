//! The check that keeps the simulator binding out of shippable crates.

use std::process::Command;

use crate::util::workspace_root;

/// Assert that the simulator binding cannot reach a shipped artifact.
///
/// `tt-ttsim` binds a third-party `.so`. Binding it for development does not
/// compromise the native-Rust goal; shipping it would. The constraint is only real
/// if something checks, so this is that check.
pub fn check_no_sim_in_ship() -> Result<(), String> {
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
