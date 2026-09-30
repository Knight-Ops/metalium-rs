//! Compiles the specification's FMA model for the differential tests.
//!
//! The device firmware used to be built here too; it is now built by
//! `tt-firmware-images`, which ships, and which this crate re-exports.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    build_fma_oracle(&out_dir);
}

/// Compile the specification's own FMA model and link it into the test binary.
///
/// The tolerance policy is "never a guessed epsilon", and for multiply-add the
/// documented model is `Miscellaneous/FMA/fma.c`. `tt_isa::numerics::fma_bh` is a
/// hand port of it, which is the kind of transcription this workspace otherwise
/// generates its way out of -- so the port is checked against the C itself rather
/// than against a copy of the C's answers.
///
/// Shelled out to `cc` for the same reason `xtask` shells out to `curl` and
/// `sha256sum`: adding a build-dependency to compile eighty lines of C would be a
/// larger commitment than the job needs. If `cc` is missing the differential test
/// is skipped rather than failing, and says so -- a missing compiler is an
/// environment gap, not a finding about the port.
fn build_fma_oracle(out_dir: &Path) {
    // `have_fma_oracle` is set below when the C actually compiled; declare it so a
    // clone without a C compiler does not warn about an unexpected `cfg`.
    println!("cargo::rustc-check-cfg=cfg(have_fma_oracle)");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.parent().unwrap().parent().unwrap();
    let source = root.join("vendor/tt-isa-documentation/Miscellaneous/FMA/fma.c");
    println!("cargo:rerun-if-changed={}", source.display());
    if !source.exists() {
        // The specification tree is fetched by `cargo xtask fetch-spec`; a clone
        // that has not run it yet should still build.
        return;
    }

    let obj = out_dir.join("fma.o");
    let lib = out_dir.join("libfmaoracle.a");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let compiled = Command::new(&cc)
        .args(["-c", "-O2", "-fPIC", "-o"])
        .arg(&obj)
        .arg(&source)
        .status();
    let Ok(status) = compiled else {
        println!("cargo:warning=no C compiler found; the FMA differential test will be skipped");
        return;
    };
    assert!(
        status.success(),
        "compiling the specification's fma.c failed"
    );

    let ar = std::env::var("AR").unwrap_or_else(|_| "ar".into());
    let archived = Command::new(&ar).arg("crs").arg(&lib).arg(&obj).status();
    let Ok(status) = archived else {
        println!("cargo:warning=no `ar` found; the FMA differential test will be skipped");
        return;
    };
    assert!(status.success(), "archiving the FMA oracle failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=fmaoracle");
    println!("cargo:rustc-cfg=have_fma_oracle");
}
