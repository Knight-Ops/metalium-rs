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
pub mod provenance;
pub mod render;
pub mod syntax;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};

use crate::spec;

/// Everything the generator needs, parsed and checked.
pub struct Sources {
    pub diagrams: Vec<model::Diagram>,
    pub provenance: BTreeMap<String, (provenance::Provenance, String)>,
    /// Diagram key -> the `TT_*` macro name its page documents.
    pub mnemonics: BTreeMap<String, String>,
    /// Diagram key -> every macro name on its page.
    ///
    /// Almost always one. `RMWCIB` has four, `TT_RMWCIB0` through `TT_RMWCIB3`,
    /// because the digit is part of its opcode.
    pub variants: BTreeMap<String, Vec<String>>,
    pub report: Report,
}

/// What the checks found, for `check-isa-sources` to print.
pub struct Report {
    pub instructions: usize,
    pub layouts: usize,
    pub fields: usize,
    pub complete: usize,
    pub pages: usize,
    pub pairs: usize,
    pub unchecked: Vec<String>,
    pub provenance_counts: BTreeMap<&'static str, usize>,
}

/// Read the pinned specification, parse both descriptions of the instruction set,
/// and check them against each other.
///
/// Shared by `check-isa-sources` and `gen-isa`, so that generating cannot happen
/// against sources the checks would have rejected.
pub fn load() -> Result<Sources, String> {
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

    let mut pages = Vec::new();
    let mut bare_paths = Vec::new();
    for rel in spec::consumed_files(&root)? {
        if !rel.ends_with(".md") {
            continue;
        }
        let text =
            std::fs::read_to_string(root.join(&rel)).map_err(|e| format!("reading {rel}: {e}"))?;
        match syntax::parse_page(&rel, &text)? {
            Some(page) => pages.push(page),
            // A Blackhole page with no encoding of its own. Whether it delegates to
            // the Wormhole tree is read from the phrase it uses, not inferred from
            // its silence.
            None if rel.starts_with("BlackholeA0/") => {
                let delegates = text.contains("This document is shared with the Wormhole tree");
                bare_paths.push((rel, delegates));
            }
            None => {}
        }
    }
    let bare: Vec<provenance::BlackholePage<'_>> = bare_paths
        .iter()
        .map(|(rel, delegates)| provenance::BlackholePage {
            stem: rel
                .rsplit('/')
                .next()
                .and_then(|f| f.strip_suffix(".md"))
                .unwrap_or(rel),
            delegates: *delegates,
        })
        .collect();

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

    let provenance = provenance::classify(&pages, &bare)?;
    let mut mnemonics = BTreeMap::new();
    let mut variants: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for page in &pages {
        for (key, call) in page.keys.iter().zip(&page.calls) {
            mnemonics
                .entry(key.clone())
                .or_insert_with(|| call.name.clone());
        }
        // A page with one diagram and several macros is documenting several
        // encodings of it. Only `RMWCIB` does this, and the check in `render`
        // insists on that.
        if page.keys.len() == 1 && page.calls.len() > 1 {
            variants
                .entry(page.keys[0].clone())
                .or_insert_with(|| page.calls.iter().map(|c| c.name.clone()).collect());
        }
    }

    let mut provenance_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (p, _) in provenance.values() {
        *provenance_counts.entry(p.variant_name()).or_default() += 1;
    }

    let instructions = diagrams
        .iter()
        .filter(|d| d.opcode_field().is_some())
        .count();
    let report = Report {
        instructions,
        layouts: diagrams.len() - instructions,
        fields: diagrams.iter().map(|d| d.fields.len()).sum(),
        // Reported because it is the surprising number: the diagrams label only the
        // bits that carry meaning, so most of them leave gaps. Anything that assumed
        // a diagram partitioned its word would be wrong about nearly all of them.
        complete: diagrams.iter().filter(|d| d.undrawn() == 0).count(),
        pages: pages.len(),
        pairs: crossed.pairs,
        unchecked: crossed.unchecked.clone(),
        provenance_counts,
    };

    Ok(Sources {
        diagrams,
        provenance,
        mnemonics,
        variants,
        report,
    })
}

/// Parse and cross-check the specification sources without generating anything.
///
/// Split out from generation deliberately: it is the step where the two descriptions
/// of the instruction set are compared, and it is useful on its own after a
/// specification re-sync.
pub fn check_sources() -> Result<(), String> {
    let r = load()?.report;
    println!(
        "ok: {} diagrams ({} with an opcode, {} layouts), {} fields, {} covering every bit of their word",
        r.instructions + r.layouts,
        r.instructions,
        r.layouts,
        r.fields,
        r.complete,
    );
    println!(
        "ok: {} pages, {} (diagram, macro) pairs agree on names, widths, signedness and slot positions; \
         {} diagram(s) have no syntax block to check against: {}",
        r.pages,
        r.pairs,
        r.unchecked.len(),
        r.unchecked.join(", "),
    );
    println!("ok: provenance {:?}", r.provenance_counts);
    Ok(())
}

/// Regenerate `crates/tt-isa/src/isa/generated.rs`.
///
/// Committed rather than produced by a build script, for the same reasons as the
/// configuration table next door: it keeps `tt-isa` free of build dependencies and
/// buildable for `riscv32im` with nothing but rustc, it makes 145 encodings
/// reviewable, and it turns a specification bump into a diff someone can read.
pub fn generate(check_only: bool) -> Result<(), String> {
    let sources = load()?;
    let rendered = render::render(&render::Input {
        diagrams: &sources.diagrams,
        provenance: &sources.provenance,
        mnemonics: &sources.mnemonics,
        variants: &sources.variants,
        spec_rev: crate::pin::SPEC_REV,
    })?;
    let root = crate::util::workspace_root();
    let generated = crate::util::rustfmt(&rendered, &root)?;
    let dest = root.join("crates/tt-isa/src/isa/generated.rs");

    if check_only {
        let current = std::fs::read_to_string(&dest)
            .map_err(|e| format!("reading {}: {e}", dest.display()))?;
        if current == generated {
            println!(
                "ok: {} is up to date with the pinned specification",
                dest.display()
            );
            Ok(())
        } else {
            Err(format!(
                "{} is out of date with the pinned Bits32.lua.\n\
                 Run `cargo xtask gen-isa` and commit the result.",
                dest.display()
            ))
        }
    } else {
        std::fs::write(&dest, &generated)
            .map_err(|e| format!("writing {}: {e}", dest.display()))?;
        println!("wrote {} ({} bytes)", dest.display(), generated.len());
        Ok(())
    }
}
