//! Locating and pinning the ISA specification tree.
//!
//! The specification is a *pinned input*, exactly like the simulator and
//! `cfg_defines.h`: bumping it invalidates every gate until they are re-run. Until
//! now it was pinned in `PINS.toml` by revision alone, with nothing in the build
//! reading it and nothing checking it. The instruction generator changes that, so
//! the pin has to become enforceable.
//!
//! # Why the content is hashed rather than the tarball
//!
//! The obvious thing to pin is the SHA-256 of
//! `github.com/.../archive/<rev>.tar.gz`. GitHub does not promise those bytes are
//! stable — the archive is regenerated on demand and its gzip framing has changed
//! before — so a tarball hash can fail for a reason that has nothing to do with the
//! specification. Hashing the extracted content instead is stable by construction,
//! and it can say *which file* moved, which is what the standing "re-sync the spec
//! quarterly" task actually needs.

use std::path::{Path, PathBuf};

use crate::util::{sha256_stdin, workspace_root};

/// Where the verified copy lives. Gitignored, like `vendor/cfg_defines.h`.
pub fn vendored_root() -> PathBuf {
    workspace_root().join("vendor").join("tt-isa-documentation")
}

/// Directories whose `.md` files the generator reads, relative to the tree root.
///
/// Every instruction page lives in a `TensixCoprocessor` directory — Wormhole's has
/// `Packers/` and `Unpackers/` subdirectories, Blackhole's is flat — and the two
/// `Atomics.md` pages carry the `NOC_AT_LEN_BE_*` encodings.
const MARKDOWN_ROOTS: &[&str] = &[
    "BlackholeA0/TensixTile/TensixCoprocessor",
    "WormholeB0/TensixTile/TensixCoprocessor",
];

const MARKDOWN_FILES: &[&str] = &["BlackholeA0/NoC/Atomics.md", "WormholeB0/NoC/Atomics.md"];

/// The machine-readable bit-layout source every encoding diagram is rendered from.
pub const BITS32: &str = "Diagrams/Src/Bits32.lua";

/// Every file the generator reads, sorted, as paths relative to `root`.
///
/// Sorted because the digest is order-sensitive and directory iteration order is
/// not. A missing directory is an error rather than an empty result: silently
/// hashing nothing would make the pin vacuous.
pub fn consumed_files(root: &Path) -> Result<Vec<String>, String> {
    let mut out = vec![BITS32.to_string()];
    for f in MARKDOWN_FILES {
        out.push((*f).to_string());
    }
    for dir in MARKDOWN_ROOTS {
        let base = root.join(dir);
        if !base.is_dir() {
            return Err(format!(
                "{} is not a directory — is {} really a tt-isa-documentation checkout?",
                base.display(),
                root.display()
            ));
        }
        collect_markdown(&base, dir, &mut out)?;
    }
    for rel in &out {
        let p = root.join(rel);
        if !p.is_file() {
            return Err(format!(
                "{} is missing from the specification tree",
                p.display()
            ));
        }
    }
    out.sort();
    Ok(out)
}

fn collect_markdown(dir: &Path, rel: &str, out: &mut Vec<String>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("reading {}: {e}", dir.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let child_rel = format!("{rel}/{name}");
        let path = entry.path();
        if path.is_dir() {
            collect_markdown(&path, &child_rel, out)?;
        } else if name.ends_with(".md") {
            out.push(child_rel);
        }
    }
    Ok(())
}

/// SHA-256 over every consumed file: for each, its relative path, its length, and
/// its bytes, in sorted path order.
///
/// Lengths are included so that no rearrangement of content between two files can
/// produce the same digest.
pub fn digest(root: &Path) -> Result<String, String> {
    let files = consumed_files(root)?;
    let mut buf: Vec<u8> = Vec::new();
    for rel in &files {
        let contents = std::fs::read(root.join(rel))
            .map_err(|e| format!("reading {}: {e}", root.join(rel).display()))?;
        buf.extend_from_slice(rel.as_bytes());
        buf.push(0);
        buf.extend_from_slice(contents.len().to_string().as_bytes());
        buf.push(0);
        buf.extend_from_slice(&contents);
        buf.push(0);
    }
    sha256_stdin(&buf)
}
