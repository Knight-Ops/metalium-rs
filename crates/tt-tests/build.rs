//! Builds the device firmware and checks its instruction encoding.
//!
//! Runs as part of `cargo test`, so a fresh clone needs no separate step.
//!
//! The firmware is a separate workspace targeting `riscv32im-unknown-none-elf`.
//! It is built into `OUT_DIR` with its own `--target-dir`, and the host's
//! `CARGO_*` environment is stripped from the child so the nested invocation does
//! not inherit the parent's target directory or profile.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Binaries in `tt-firmware/src/bin` to build and expose.
const BINARIES: &[&str] = &[
    "heartbeat",
    "sfpu_mul",
    "corpus",
    "corpus_t0",
    "role_t0",
    "role_t1",
    "role_t2",
];

const TARGET: &str = "riscv32im-unknown-none-elf";

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let firmware_dir = manifest.parent().unwrap().join("tt-firmware");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Track individual files rather than directories. A directory-level
    // `rerun-if-changed` does not reliably notice an edit to a file nested inside
    // it, which showed up here as firmware changes that never rebuilt.
    track_sources(&firmware_dir);
    track_sources(&manifest.parent().unwrap().join("tt-isa"));

    let target_dir = out_dir.join("firmware-target");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());

    let mut cmd = Command::new(&cargo);
    cmd.current_dir(&firmware_dir)
        .args(["build", "--release", "--target", TARGET])
        .arg("--target-dir")
        .arg(&target_dir);
    // A nested cargo inherits variables that redirect it back at the parent
    // workspace; strip them so the firmware build is genuinely independent.
    for (key, _) in std::env::vars() {
        if key.starts_with("CARGO_") || key == "RUSTFLAGS" || key == "RUSTC_WRAPPER" {
            cmd.env_remove(key);
        }
    }
    let status = cmd
        .status()
        .expect("could not run cargo for the firmware build");
    assert!(status.success(), "firmware build failed");

    build_fma_oracle(&out_dir);

    let objcopy = tool("llvm-objcopy");
    let objdump = tool("llvm-objdump");

    for name in BINARIES {
        let elf = target_dir.join(TARGET).join("release").join(name);
        assert!(
            elf.exists(),
            "firmware binary {name} was not produced at {}",
            elf.display()
        );

        let bin = out_dir.join(format!("{name}.bin"));
        let status = Command::new(&objcopy)
            .args(["-O", "binary"])
            .arg(&elf)
            .arg(&bin)
            .status()
            .expect("could not run llvm-objcopy");
        assert!(status.success(), "objcopy failed for {name}");

        check_instruction_set(&objdump, &elf, name);

        println!(
            "cargo:rustc-env=FIRMWARE_{}={}",
            name.to_uppercase(),
            bin.display()
        );
    }
}

/// Emit `rerun-if-changed` for every source file under `dir`, skipping build
/// output.
fn track_sources(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if name == "target" || name == ".git" {
            continue;
        }
        if path.is_dir() {
            track_sources(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Locate one of the `llvm-tools` binaries shipped with the active toolchain.
fn tool(name: &str) -> PathBuf {
    let sysroot = Command::new(std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()))
        .arg("--print")
        .arg("sysroot")
        .output()
        .expect("could not ask rustc for its sysroot");
    let sysroot = String::from_utf8(sysroot.stdout).unwrap();
    let host = std::env::var("HOST").unwrap_or_else(|_| "x86_64-unknown-linux-gnu".into());
    let path = Path::new(sysroot.trim())
        .join("lib/rustlib")
        .join(host)
        .join("bin")
        .join(name);
    assert!(
        path.exists(),
        "{name} not found at {}. Install it with `rustup component add llvm-tools`.",
        path.display()
    );
    path
}

/// Assert the image contains nothing Blackhole cannot execute.
///
/// This is a correctness gate, not hygiene. Each of these is a documented gap in
/// the baby RISC-V implementation, and executing an unimplemented instruction is
/// `UndefinedBehavior` — which, per the glossary, includes possible physical
/// damage. There is no illegal-instruction trap to fall back on.
fn check_instruction_set(objdump: &Path, elf: &Path, name: &str) {
    let out = Command::new(objdump)
        .args(["-d", "--no-show-raw-insn"])
        .arg(elf)
        .output()
        .expect("could not run llvm-objdump");
    assert!(out.status.success(), "objdump failed for {name}");
    let text = String::from_utf8_lossy(&out.stdout);

    let mut checked = 0usize;
    for line in text.lines() {
        // Instruction lines are "<hex address>:<whitespace><mnemonic><whitespace><operands>".
        // Label lines ("00006000 <_start>:") end in ':' and have no mnemonic after it.
        let Some((addr, rest)) = line.split_once(':') else {
            continue;
        };
        if addr.trim().is_empty() || !addr.trim().chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let Some(mnemonic) = rest.split_whitespace().next() else {
            continue;
        };
        checked += 1;
        let operands = rest.trim().strip_prefix(mnemonic).unwrap_or("").trim();
        if let Some(why) = forbidden_reason(mnemonic).or_else(|| fence_reason(mnemonic, operands)) {
            panic!(
                "firmware `{name}` contains `{mnemonic}`, which Blackhole does not \
                 implement: {why}\n  in:{line}"
            );
        }
    }

    // A parser that matches nothing passes everything. This check exists because
    // an earlier version of it did exactly that: llvm-objdump puts spaces between
    // the address colon and the mnemonic, and the first parser looked for ":\t".
    assert!(
        checked > 8,
        "the instruction-set check only recognised {checked} instructions in `{name}`, \
         which means it is not actually reading the disassembly"
    );
}

/// Why this `fence` must not appear, if it must not.
///
/// `pause` (Zihintpause) shares an encoding with `fence w, 0`, and a disassembler
/// that does not know the extension prints the latter — so checking the mnemonic
/// alone misses it. Blackhole does not list Zihintpause among the extensions it
/// implements, and `core::hint::spin_loop()` emits it, which is an easy way to
/// put an unimplemented instruction into a spin loop without noticing.
///
/// The permitted forms are the ordering fences the specification's own recipes
/// use. Anything else is flagged rather than assumed safe.
fn fence_reason(mnemonic: &str, operands: &str) -> Option<&'static str> {
    if mnemonic != "fence" {
        return None;
    }
    const ALLOWED: &[&str] = &["", "iorw, iorw", "rw, rw", "rw, w", "r, rw"];
    if ALLOWED.contains(&operands) {
        return None;
    }
    if operands == "w, 0" {
        return Some(
            "this is the `pause` encoding (Zihintpause), which Blackhole does not implement. \
             `core::hint::spin_loop()` emits it -- use an empty asm block instead",
        );
    }
    Some("an unrecognised fence ordering; only the forms the specification's recipes use are allowed")
}

/// Why this mnemonic must not appear in a firmware image, if it must not.
///
/// Each is a documented gap in the baby RISC-V implementation, and executing an
/// unimplemented instruction is `UndefinedBehavior` — which per `Glossary.md`
/// includes possible physical damage. There is no illegal-instruction trap to
/// fall back on, so this is a correctness gate rather than hygiene.
fn forbidden_reason(mnemonic: &str) -> Option<&'static str> {
    // Strip the ordering suffixes AMOs may carry, so `lr.w.aq` is caught too.
    let base = mnemonic.trim_end_matches(".aq").trim_end_matches(".rl");
    if mnemonic == "<unknown>" {
        return Some(
            "the disassembler could not decode it, so neither this gate nor a reader can \
             tell what it is. Usually a missing `.riscv.attributes` section or an \
             instruction outside the target's extension set",
        );
    }
    if mnemonic == "fence.i" {
        return Some(
            "Zifencei is not implemented. `fence.i` executes as a `nop`, which \
             `InstructionCache.md:23` classes as NonContractualBehavior, and ttsim refuses it \
             outright: \"fence.i does not flush the instruction cache on babyrisc and should \
             not be used\". Write the 5-bit mask to `RISCV_IC_INVALIDATE_InvalidateAll` instead",
        );
    }
    if mnemonic.starts_with("c.") {
        return Some(
            "the C extension is not implemented, and its encoding space is reused by `.ttinsn`",
        );
    }
    if base.starts_with("lr.") || base.starts_with("sc.") {
        return Some("Zalrsc is not implemented; there is no load-reserved/store-conditional");
    }
    if mnemonic.starts_with("fdiv") {
        return Some("floating-point divide is not implemented");
    }
    if mnemonic.starts_with("fsqrt") {
        return Some("floating-point square root is not implemented");
    }
    if matches!(mnemonic, "div" | "divu" | "rem" | "remu") {
        return Some(
            "RISCV T2 has no integer divide or remainder; avoided everywhere so that one \
             image can run on any core",
        );
    }
    None
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
