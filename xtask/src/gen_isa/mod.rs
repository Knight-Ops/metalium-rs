//! `Diagrams/Src/Bits32.lua` -> `tt_isa::isa`.
//!
//! The Tensix instruction set is described twice in the specification, in two
//! notations written for two different readers:
//!
//! * `Diagrams/Src/Bits32.lua` — a drawing script. Each entry places labelled
//!   rectangles over a word, and the rectangles are the bit fields.
//! * Each instruction page's `## Syntax` block — a hand-written C macro call
//!   template, `TT_PACR(/* u2 */ CfgContext, …)`, for a human to copy.
//!
//! Neither is generated from the other. They are produced by different tooling for
//! different purposes, so a slip in one does not propagate to the other. **That is
//! what makes checking them against each other worth anything**, and it is the
//! standard this repository already applies elsewhere: `gen-cfg` checks a
//! machine-generated header against the prose that documents it.
//!
//! The weakness is worth stating plainly: same repository, same revision, often the
//! same author. A shared *misunderstanding* survives both sources, and only silicon
//! settles that. What this catches is transcription error, which is the failure mode
//! that actually bites when a human copies 818 bit positions.

pub mod check;
pub mod lua;
pub mod model;
pub mod syntax;

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use crate::spec;

/// Parse and cross-check the specification sources without generating anything.
///
/// Split out from generation deliberately: it is the step where the two sources are
/// compared, and it is useful on its own after a specification re-sync.
pub fn check_sources() -> Result<(), String> {
    let root = spec::root()?;
    let bits32 = std::fs::read_to_string(root.join(spec::BITS32))
        .map_err(|e| format!("reading {}: {e}", root.join(spec::BITS32).display()))?;

    let unused = lua::unused(&bits32);
    if !unused.is_empty() {
        return Err(format!(
            "SPECIAL_LABELS has entries nothing uses: {}.\n\
             An exception that is no longer needed means the specification changed \
             shape somewhere nobody looked. Remove it deliberately.",
            unused.join(", ")
        ));
    }

    let diagrams = lua::parse(&bits32)?;
    let diagrams = check::dedupe(diagrams)?;
    check::structure(&diagrams)?;

    let instructions = diagrams
        .iter()
        .filter(|d| d.opcode_field().is_some())
        .count();
    let mut pages = Vec::new();
    for rel in spec::consumed_files(&root)? {
        if !rel.ends_with(".md") {
            continue;
        }
        let text =
            std::fs::read_to_string(root.join(&rel)).map_err(|e| format!("reading {rel}: {e}"))?;
        if let Some(page) = syntax::parse_page(&rel, &text)? {
            pages.push(page);
        }
    }
    let crossed = check::cross_check(&diagrams, &pages)?;

    let stale = check::stale_exceptions(&crossed);
    if !stale.is_empty() {
        return Err(format!(
            "these exceptions are no longer needed: {}.\n\
             Remove them deliberately — an exception that stopped applying means the \
             specification moved somewhere nobody looked.",
            stale.join(", ")
        ));
    }

    if !crossed.unembedded.is_empty() {
        return Err(format!(
            "Bits32.lua defines diagrams no page embeds: {}.\n\
             The page is what says which chip the encoding applies to, so a diagram \
             without one has no provenance.",
            crossed.unembedded.join(", ")
        ));
    }

    // Two independent ways of asking "is this an instruction?" -- does the diagram
    // carry an opcode at bits 24..31, and does its page document a macro call --
    // and they must give the same answer. Neither is derived from the other, so
    // this is free evidence that the split between instructions and the `Src`/`Dst`
    // datum layouts is real rather than a naming convention.
    let no_opcode: BTreeSet<&str> = diagrams
        .iter()
        .filter(|d| d.opcode_field().is_none())
        .map(|d| d.key.as_str())
        .collect();
    let no_macro: BTreeSet<&str> = crossed.unchecked.iter().map(String::as_str).collect();
    if no_opcode != no_macro {
        return Err(format!(
            "`has an opcode` and `has a macro` disagree about which diagrams are \
             instructions.\n  opcode but no macro: {:?}\n  macro but no opcode: {:?}",
            no_opcode.difference(&no_macro).collect::<Vec<_>>(),
            no_macro.difference(&no_opcode).collect::<Vec<_>>(),
        ));
    }

    // Reported because it is the surprising number: the diagrams label only the
    // bits that carry meaning, so most of them leave gaps. Anything that assumed a
    // diagram partitioned its word would be wrong about nearly all of them.
    let complete = diagrams.iter().filter(|d| d.undrawn() == 0).count();
    println!(
        "ok: {} diagrams ({} with an opcode, {} layouts), {} fields, \
         {complete} covering every bit of their word",
        diagrams.len(),
        instructions,
        diagrams.len() - instructions,
        diagrams.iter().map(|d| d.fields.len()).sum::<usize>(),
    );
    println!(
        "ok: {} pages, {} (diagram, macro) pairs agree on names, widths, signedness \
         and slot positions; {} diagram(s) have no syntax block to check against: {}",
        pages.len(),
        crossed.pairs,
        crossed.unchecked.len(),
        crossed.unchecked.join(", "),
    );
    Ok(())
}
