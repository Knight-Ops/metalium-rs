//! Compiles the specification's FMA model for the differential tests.
//!
//! The device firmware used to be built here too; it is now built by
//! `tt-firmware-images`, which ships, and which this crate re-exports.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    build_fma_oracle(&out_dir);
    build_sfpu_models(&out_dir);
}

/// The C functions of the Vector Unit's pages that `tt_isa::numerics::sfpu`
/// ports, extracted from the pinned tree: `(page, a line the block holds)`.
/// Each block is a page's "Supporting definitions", self-contained C -- or C++
/// where a page reaches for `std::bit_cast`, which is why they are compiled as
/// C++20 inside `extern "C"`.
const SFPU_MODELS: &[(&str, &str)] = &[
    ("SFPARECIP.md", "uint32_t ApproxRecip("),
    ("SFPLE.md", "bool SignMagIsSmaller("),
    ("SFPLUT.md", "uint32_t Lut8ToFp32("),
    ("SFPLUTFP32.md", "float Lut16ToFp32("),
];

/// Extract and compile [`SFPU_MODELS`] into `libsfpumodels.a`, for
/// `sfpu_models_oracle`: the ports are held to the page's own C, as `fma_bh` is
/// to `fma.c`, so a transcription slip in a lookup table cannot hide. Skipped,
/// with a warning, without the tree or a C compiler.
fn build_sfpu_models(out_dir: &Path) {
    println!("cargo::rustc-check-cfg=cfg(have_sfpu_models)");
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let pages = manifest
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("vendor/tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor");
    let mut c = String::from("#include <bit>\n#include <cstdint>\nextern \"C\" {\n");
    for (page, marker) in SFPU_MODELS {
        let path = pages.join(page);
        println!("cargo:rerun-if-changed={}", path.display());
        let Ok(text) = std::fs::read_to_string(&path) else {
            return;
        };
        let block = text
            .split("```c")
            .skip(1)
            .map(|b| b.split("```").next().unwrap_or(""))
            .find(|b| b.contains(marker))
            .unwrap_or_else(|| panic!("{page}: no C block holds `{marker}`"));
        c += &format!("/* {page} */\n{block}\n");
    }
    c += "}\n";
    let src = out_dir.join("sfpu_models.cc");
    std::fs::write(&src, c).unwrap();
    let obj = out_dir.join("sfpu_models.o");
    let lib = out_dir.join("libsfpumodels.a");
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    let Ok(status) = Command::new(&cxx)
        .args(["-std=c++20", "-fno-exceptions", "-c", "-O2", "-fPIC", "-o"])
        .arg(&obj)
        .arg(&src)
        .status()
    else {
        println!(
            "cargo:warning=no C++ compiler found; the SFPU model differential test will be skipped"
        );
        return;
    };
    assert!(
        status.success(),
        "compiling the pages' SFPU models failed: {}",
        src.display()
    );
    let ar = std::env::var("AR").unwrap_or_else(|_| "ar".into());
    let Ok(status) = Command::new(&ar).arg("crs").arg(&lib).arg(&obj).status() else {
        println!("cargo:warning=no `ar` found; the SFPU model differential test will be skipped");
        return;
    };
    assert!(status.success(), "archiving the SFPU models failed");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-cfg=have_sfpu_models");
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
