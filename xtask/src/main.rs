//! Build orchestration: fetching pinned specification inputs, generating the
//! tables `tt-isa` is built from, and the checks that keep those pins honest.
//!
//! Deliberately dependency-free — it shells out to `curl`, `sha256sum` and
//! `rustfmt` rather than pulling an HTTP stack, a hash crate and a formatter into
//! the workspace for a handful of calls.

use std::process::ExitCode;

mod fetch;
mod gen_burn;
mod gen_cfg;
mod gen_isa;
mod pin;
mod ship;
mod silicon;
mod spec;
mod util;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let cmd = args.next();
    let result = match cmd.as_deref() {
        Some("fetch-ttsim") => fetch::fetch_ttsim(args.any(|a| a == "--force")),
        Some("check-no-sim-in-ship") => ship::check_no_sim_in_ship(),
        Some("gen-cfg") => gen_cfg::gen_cfg(args.any(|a| a == "--check")),
        Some("fetch-spec") => fetch::fetch_spec(args.any(|a| a == "--force")),
        Some("fetch-kmd") => fetch::fetch_kmd(args.any(|a| a == "--force")),
        Some("fetch-mnist") => fetch::fetch_mnist(args.any(|a| a == "--force")),
        Some("check-isa-sources") => gen_isa::check_sources(),
        Some("gen-isa") => gen_isa::generate(args.any(|a| a == "--check")),
        Some("gen-burn-delegate") => gen_burn::generate(args.any(|a| a == "--check")),
        Some("silicon") => silicon::run(args),
        Some("bench") => silicon::bench(args),
        Some(other) => Err(format!("unknown task `{other}`\n\n{USAGE}")),
        None => Err(USAGE.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("xtask: {msg}");
            ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "\
usage: cargo xtask <task>

tasks:
  fetch-ttsim [--force]   download the pinned libttsim builds into vendor/
  fetch-spec [--force]    download the pinned ISA specification tree into vendor/
                          and verify it against the pinned content digest
  fetch-kmd [--force]     download the pinned tt-kmd ioctl.h into vendor/; this is
                          the driver ABI tt-kmd binds, not a specification
  fetch-mnist [--force]   download MNIST into vendor/mnist/, decompressed and
                          verified against the pinned hashes; the Phase 7
                          training gate reads it
  gen-cfg [--check]       regenerate tt-isa's backend-configuration field table
                          from the pinned cfg_defines.h; --check fails if the
                          committed file is out of date rather than rewriting it
  gen-isa [--check]       regenerate tt-isa's instruction table from the pinned
                          Bits32.lua, cross-checked against the specification's
                          syntax blocks; --check fails if the committed file is
                          out of date rather than rewriting it
  gen-burn-delegate [--check]
                          regenerate burn-tt's forwarding of every burn-backend
                          op to burn-flex from the pinned burn-backend source;
                          --check fails if the committed file is out of date
  check-isa-sources       parse the pinned Bits32.lua and report what it holds,
                          without generating anything
  check-no-sim-in-ship    assert tt-ttsim is absent from shippable dependency graphs
  silicon [options]       run the silicon suite one test per process, with an
                          fsync'd log; `cargo xtask silicon --help` for options
  bench [options]         the firmware benchmarks on silicon (release, ignored
                          tests included), collected into target/silicon/bench/;
                          `cargo xtask bench --help`";
