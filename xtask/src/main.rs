//! Build orchestration: fetching pinned specification inputs, generating the
//! tables `tt-isa` is built from, and the checks that keep those pins honest.
//!
//! Deliberately dependency-free — it shells out to `curl`, `sha256sum` and
//! `rustfmt` rather than pulling an HTTP stack, a hash crate and a formatter into
//! the workspace for a handful of calls.

use std::process::ExitCode;

mod fetch;
mod gen_cfg;
mod gen_isa;
mod pin;
mod ship;
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
        Some("check-isa-sources") => gen_isa::check_sources(),
        Some("gen-isa") => gen_isa::generate(args.any(|a| a == "--check")),
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
  gen-cfg [--check]       regenerate tt-isa's backend-configuration field table
                          from the pinned cfg_defines.h; --check fails if the
                          committed file is out of date rather than rewriting it
  gen-isa [--check]       regenerate tt-isa's instruction table from the pinned
                          Bits32.lua, cross-checked against the specification's
                          syntax blocks; --check fails if the committed file is
                          out of date rather than rewriting it
  check-isa-sources       parse the pinned Bits32.lua and report what it holds,
                          without generating anything
  check-no-sim-in-ship    assert tt-ttsim is absent from shippable dependency graphs";
