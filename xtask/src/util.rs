//! Small helpers shared by the tasks: paths, hashing, and formatting generated code.
//!
//! Hashing shells out to `sha256sum` rather than taking a hash crate, for the same
//! reason the rest of this tool shells out: `xtask` stays dependency-free.

use std::path::{Path, PathBuf};
use std::process::Command;

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives directly below the workspace root")
        .to_path_buf()
}

pub fn sha256(path: &Path) -> Result<String, String> {
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

/// Run the generated source through rustfmt.
pub fn rustfmt(source: &str, root: &Path) -> Result<String, String> {
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
