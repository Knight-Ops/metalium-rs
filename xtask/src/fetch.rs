//! Fetching pinned specification inputs into `vendor/`.
//!
//! Everything here is *pinned* and hash-verified on arrival. These are
//! specification inputs, not dependencies: an unnoticed change to one silently
//! changes what the test suite proves, which is why nothing is fetched without a
//! hash to check it against.

use std::path::Path;
use std::process::Command;

use crate::pin;
use crate::spec;
use crate::util::{download, sha256, workspace_root};

///
/// The `.so` is not committed — it is a 244 KiB binary blob with its own release
/// cadence — but it is *pinned*, because it is the oracle every gate is measured
/// against. An unnoticed change to it silently changes what the test suite proves.
pub fn fetch_ttsim(force: bool) -> Result<(), String> {
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

pub fn fetch_cfg_defines(dest: &Path) -> Result<(), String> {
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

/// Download and verify the pinned ISA specification tree.
///
/// Unlike the other two pinned inputs this is a whole repository rather than one
/// file, so it is fetched as a source tarball and verified by content digest —
/// see [`crate::spec`] for why the tarball's own hash will not do.
pub fn fetch_spec(force: bool) -> Result<(), String> {
    let dest = spec::vendored_root();

    if dest.exists() && !force {
        let got = spec::digest(&dest)?;
        if got == pin::SPEC_CONTENT_SHA256 {
            println!("{} is already present and matches the pin", dest.display());
            return Ok(());
        }
        return Err(format!(
            "{} hashes {got}, not the pinned {}.\n\
             Re-run with --force to replace it, or update PINS.toml and xtask's pin \
             module if the bump is intentional — note that doing so invalidates every \
             gate until they are re-run.",
            dest.display(),
            pin::SPEC_CONTENT_SHA256
        ));
    }

    let url = format!(
        "https://github.com/tenstorrent/tt-isa-documentation/archive/{}.tar.gz",
        pin::SPEC_REV
    );
    let tarball = workspace_root()
        .join("vendor")
        .join("tt-isa-documentation.tar.gz");
    download(&url, &tarball)?;

    // Extract into a staging directory and swap it in, so a failed extraction
    // cannot leave a half-populated tree that the digest check would then have to
    // diagnose.
    let staging = workspace_root()
        .join("vendor")
        .join("tt-isa-documentation.staging");
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)
        .map_err(|e| format!("creating {}: {e}", staging.display()))?;
    let status = Command::new("tar")
        .args(["-xzf"])
        .arg(&tarball)
        .args(["-C"])
        .arg(&staging)
        // The archive has a single `tt-isa-documentation-<rev>/` top level.
        .args(["--strip-components", "1"])
        .status()
        .map_err(|e| format!("could not run tar: {e}"))?;
    if !status.success() {
        return Err(format!("tar failed extracting {}", tarball.display()));
    }

    let got = spec::digest(&staging)?;
    if got != pin::SPEC_CONTENT_SHA256 {
        return Err(format!(
            "content digest mismatch for the specification tree:\n  \
             expected {}\n  got      {got}\n\
             The pinned revision is immutable, so this means either the download was \
             corrupted or PINS.toml and xtask's pin module disagree about the digest.",
            pin::SPEC_CONTENT_SHA256
        ));
    }

    let _ = std::fs::remove_dir_all(&dest);
    std::fs::rename(&staging, &dest).map_err(|e| format!("renaming into place: {e}"))?;
    let _ = std::fs::remove_file(&tarball);
    println!("wrote {} ({})", dest.display(), pin::SPEC_REV);
    Ok(())
}
