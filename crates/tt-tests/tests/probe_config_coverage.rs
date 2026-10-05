//! Which `Config` registers does ttsim actually model?
//!
//! `docs/learnings/ttsim-divergence.md` row 21 records that ttsim implements `Config` as a
//! `switch` over specific registers rather than as a backing array, so a legitimate
//! word no workload has exercised is *fatal* rather than merely uninitialised. That
//! row says "topping out around register 220", which is not precise enough to plan
//! against: driving the unpacker means writing about twenty specific registers, and
//! finding out which of them are unreachable one failed run at a time is slow and
//! tells you nothing about the ones you have not tried yet.
//!
//! This walks all 224 of them and prints the map. Ignored, because it is a survey
//! rather than a claim; the claims it supports live in the tests that use the
//! registers.
//!
//! Run with:
//! `cargo test -p tt-tests --test probe_config_coverage -- --ignored --nocapture`

// Simulator only, permanently: it writes every `Config` register blind, which on silicon is a whole-backend reconfiguration.
#![cfg(not(feature = "silicon"))]

use tt_isa::backend;
use tt_isa::isa::Instruction;
use tt_isa::sfpu;
use tt_tests::harness::{self, survives, Run};

/// A scratch GPR well clear of anything the firmware touches.
const SCRATCH_GPR: u32 = 8;

/// Write `value` to `Config` word `addr32` through the instruction path, with a
/// tail that computes so a surviving run has also done work.
fn write_raw(dev: &mut harness::Dev<'_>, addr32: u16, value: u32) {
    let mut program: Vec<Instruction> = Vec::new();
    program.extend(backend::set_gpr(SCRATCH_GPR, value).unwrap());
    program.push(backend::write_word(SCRATCH_GPR, addr32).unwrap());
    program.push(backend::nop());
    program.extend(sfpu::load_f32(0, 1.5f32.to_bits()).unwrap());
    program.push(sfpu::store(0, sfpu::store_format::FP32, 0, 0).unwrap());
    harness::run(dev, &Run::new(&program));
}

#[test]
#[ignore]
fn map_which_config_registers_ttsim_models() {
    let limit = backend::CONFIG_INDEX_LIMIT as u16;
    let mut modelled = Vec::new();
    let mut refused = Vec::new();

    for addr32 in 0..limit {
        // `STATE_RESET_EN` would zero most of `Config`; `backend::write_word`
        // refuses it, and so does this survey.
        if addr32 == backend::STATE_RESET_EN_ADDR32 {
            continue;
        }
        // Write zero: the least disruptive value, and the one most likely to be
        // what the register already holds.
        let ok = survives(|dev| write_raw(dev, addr32, 0));
        if ok {
            modelled.push(addr32);
        } else {
            refused.push(addr32);
        }
    }

    println!("\n=== Config registers ttsim models (write of 0 survives) ===");
    println!("modelled: {} of {}", modelled.len(), limit - 1);
    println!("{modelled:?}");
    println!("\nrefused: {}", refused.len());
    println!("{refused:?}");

    // Name the refused ones, so the map is readable rather than a list of numbers.
    println!("\n=== refused registers, by the fields that live in them ===");
    for addr32 in &refused {
        let mut names: Vec<&str> = tt_isa::cfg::generated::ALL_CONFIG_FIELDS
            .iter()
            .filter(|(_, f)| f.addr32() == *addr32)
            .map(|(n, _)| *n)
            .collect();
        names.sort_unstable();
        names.truncate(4);
        println!("  {addr32:3}: {}", names.join(", "));
    }
}
