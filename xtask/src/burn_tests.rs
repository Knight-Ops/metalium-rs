//! Locating and pinning Burn's own backend test suite.
//!
//! `burn-backend-tests` is not published to crates.io, so the suite that
//! `burn-tt-conformance` runs against `TtBackend` is fetched from the Burn
//! repository at the tag the workspace pins (`PINS.toml`, `[burn-backend-tests]`),
//! and kept in `vendor/burn-backend-tests/` (gitignored, like every pinned input).
//!
//! Like the ISA specification (see [`crate::spec`]) it is pinned by the digest of
//! its extracted content rather than by the tarball's hash, which GitHub does not
//! promise is byte-stable. Unlike the specification, every file under `tests/` is
//! hashed: the suite is only a meaningful oracle if no test can change under it.

use std::path::{Path, PathBuf};

use crate::util::{sha256_stdin, workspace_root};

/// Where the verified copy lives: `tests/` of the upstream `burn-backend-tests` crate.
pub fn vendored_root() -> PathBuf {
    workspace_root().join("vendor").join("burn-backend-tests")
}

/// Every file under `root`, sorted, as `/`-separated paths relative to `root`.
///
/// Sorted because the digest is order-sensitive and directory iteration order is
/// not. An empty tree is an error rather than an empty digest: hashing nothing
/// would make the pin vacuous.
pub fn files(root: &Path) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    walk(root, "", &mut out)?;
    if out.is_empty() {
        return Err(format!("{} holds no files", root.display()));
    }
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, rel: &str, out: &mut Vec<String>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("reading {}: {e}", dir.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let child = if rel.is_empty() {
            name
        } else {
            format!("{rel}/{name}")
        };
        let kind = entry
            .file_type()
            .map_err(|e| format!("{}: {e}", entry.path().display()))?;
        if kind.is_dir() {
            walk(&entry.path(), &child, out)?;
        } else if kind.is_file() {
            out.push(child);
        } else {
            // A symlink could point outside the pinned tree, so none is accepted.
            return Err(format!(
                "{} is neither a file nor a directory",
                entry.path().display()
            ));
        }
    }
    Ok(())
}

/// SHA-256 over every file: its relative path, its length, and its bytes, in
/// sorted path order. Lengths are included so that no rearrangement of content
/// between two files can produce the same digest. Same framing as
/// [`crate::spec::digest`].
pub fn digest(root: &Path) -> Result<String, String> {
    let mut buf: Vec<u8> = Vec::new();
    for rel in files(root)? {
        let path = root.join(&rel);
        let contents =
            std::fs::read(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        buf.extend_from_slice(rel.as_bytes());
        buf.push(0);
        buf.extend_from_slice(contents.len().to_string().as_bytes());
        buf.push(0);
        buf.extend_from_slice(&contents);
        buf.push(0);
    }
    sha256_stdin(&buf)
}
