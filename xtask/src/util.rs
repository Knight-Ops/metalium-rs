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

/// SHA-256 of a byte string, for digests that are computed rather than read from
/// a file.
pub fn sha256_stdin(bytes: &[u8]) -> Result<String, String> {
    use std::io::Write;
    use std::process::Stdio;

    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("could not run sha256sum: {e}"))?;
    child
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(bytes)
        .map_err(|e| format!("writing to sha256sum: {e}"))?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("waiting for sha256sum: {e}"))?;
    if !out.status.success() {
        return Err("sha256sum failed".into());
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
        .ok_or_else(|| "sha256sum produced no output".to_string())
}

/// Download a URL to a path, via a `.partial` file so an interrupted fetch cannot
/// leave a truncated result that later hashes as "wrong version".
pub fn download(url: &str, dest: &Path) -> Result<(), String> {
    println!("fetching {url}");
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    let tmp = dest.with_extension("partial");
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
        .arg(url)
        .status()
        .map_err(|e| format!("could not run curl: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("curl failed downloading {url}"));
    }
    std::fs::rename(&tmp, dest).map_err(|e| format!("renaming into place: {e}"))
}
