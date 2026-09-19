//! Which chip a bit layout is actually evidence for.
//!
//! The Blackhole documentation is a delta over Wormhole's. `tt-isa`'s module doc
//! states the consequence — *a fact sourced from a Wormhole page is a hypothesis
//! until verified* — and the repository marks such facts `UNVERIFIED` by hand. For
//! the instruction set that is 79 of 145 encodings, which is far too many to track
//! in comments.
//!
//! It does not have to be tracked by hand, because the specification already says
//! it. The page that embeds a diagram is in one tree or the other, and that is a
//! fact rather than a naming convention. Classifying by a `_BH` suffix would be a
//! guess; classifying by which tree embeds the diagram is a measurement.

use std::collections::{BTreeMap, BTreeSet};

use super::syntax::Page;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provenance {
    /// A Blackhole page embeds this diagram. Authoritative.
    Blackhole,
    /// Only a Wormhole page embeds it, but the Blackhole tree has the same page
    /// stating the document is shared and the behaviour identical. Documented to
    /// apply, rather than assumed to.
    SharedWithWormhole,
    /// Only a Wormhole page embeds it, and the Blackhole page of the same name
    /// embeds a *different* diagram instead. This layout is Wormhole's and must
    /// never be used on Blackhole.
    SupersededOnBlackhole { by: String },
    /// Only a Wormhole page embeds it, and Blackhole has no such page at all.
    /// A hypothesis: `UNVERIFIED` until silicon says otherwise.
    WormholeOnly,
}

impl Provenance {
    pub fn variant_name(&self) -> &'static str {
        match self {
            Provenance::Blackhole => "Blackhole",
            Provenance::SharedWithWormhole => "SharedWithWormhole",
            Provenance::SupersededOnBlackhole { .. } => "SupersededOnBlackhole",
            Provenance::WormholeOnly => "WormholeOnly",
        }
    }
}

const BLACKHOLE: &str = "BlackholeA0/";

/// A Blackhole page that documents no encoding of its own.
///
/// Either it delegates to the Wormhole tree -- "This document is shared with the
/// Wormhole tree; see ...", 28 pages say exactly that -- or it is not about an
/// encoding at all.
pub struct BlackholePage<'a> {
    pub stem: &'a str,
    pub delegates: bool,
}

/// Classify every diagram key, and name the page it is best documented on.
///
/// `bare` carries the Blackhole pages that embed no diagram, which is how a
/// delegating page looks from the outside: the encoding lives in the Wormhole tree
/// and the Blackhole page says so.
pub fn classify(
    pages: &[Page],
    bare: &[BlackholePage<'_>],
) -> Result<BTreeMap<String, (Provenance, String)>, String> {
    let mut bh_page_of: BTreeMap<&str, &Page> = BTreeMap::new();
    let mut wh_page_of: BTreeMap<&str, &Page> = BTreeMap::new();
    // Blackhole pages that embed a diagram, by stem: these are the ones that can
    // supersede a Wormhole encoding.
    let mut bh_by_stem: BTreeMap<&str, &Page> = BTreeMap::new();

    for page in pages {
        let blackhole = page.path.starts_with(BLACKHOLE);
        if blackhole {
            bh_by_stem.insert(stem(&page.path), page);
        }
        for key in &page.keys {
            let table = if blackhole {
                &mut bh_page_of
            } else {
                &mut wh_page_of
            };
            table.entry(key.as_str()).or_insert(page);
        }
    }

    let delegating: BTreeSet<&str> = bare
        .iter()
        .filter(|p| p.delegates)
        .map(|p| p.stem)
        .collect();

    let mut out = BTreeMap::new();
    let keys: BTreeSet<&str> = bh_page_of
        .keys()
        .chain(wh_page_of.keys())
        .copied()
        .collect();
    for key in keys {
        if let Some(page) = bh_page_of.get(key) {
            out.insert(key.to_string(), (Provenance::Blackhole, page.path.clone()));
            continue;
        }
        let page = wh_page_of[key];
        let s = stem(&page.path);
        // The Blackhole page with the same name as the Wormhole one that embeds
        // this diagram is what decides between the three remaining cases.
        let provenance = if let Some(bh) = bh_by_stem.get(s) {
            Provenance::SupersededOnBlackhole {
                by: bh.keys[0].clone(),
            }
        } else if delegating.contains(s) {
            Provenance::SharedWithWormhole
        } else {
            Provenance::WormholeOnly
        };
        out.insert(key.to_string(), (provenance, page.path.clone()));
    }
    Ok(out)
}

fn stem(path: &str) -> &str {
    path.rsplit('/')
        .next()
        .and_then(|f| f.strip_suffix(".md"))
        .unwrap_or(path)
}
