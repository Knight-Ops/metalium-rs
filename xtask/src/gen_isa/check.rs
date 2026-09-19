//! Invariants every parsed diagram must satisfy.
//!
//! These need no second source to be worth running: they catch parser bugs and
//! upstream changes that are syntactically fine but semantically new. Each one is
//! exercised by a mutation test in [`super::tests`] — an invariant nobody has
//! watched reject something is not yet evidence.

use std::collections::{BTreeMap, BTreeSet};

use super::model::{Diagram, DrawnField, Label};
use super::syntax::{self, MacroCall, Page, Term};

/// Collapse keys defined more than once, insisting the definitions agree.
///
/// `SEMINIT` is defined twice in `Bits32.lua`, byte-identically; Lua silently keeps
/// the second. Doing the same would hide the day the two copies diverge, so this
/// compares them and keeps one.
pub fn dedupe(diagrams: Vec<Diagram>) -> Result<Vec<Diagram>, String> {
    let mut seen: BTreeMap<String, Diagram> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();

    for d in diagrams {
        match seen.get(&d.key) {
            None => {
                order.push(d.key.clone());
                seen.insert(d.key.clone(), d);
            }
            Some(first) => {
                // Compared on content, not on `DrawnField::line`: the two copies
                // are at different lines by definition, and that is not a
                // disagreement about the encoding.
                let same = |a: &Diagram, b: &Diagram| {
                    a.nbits == b.nbits
                        && a.fields.len() == b.fields.len()
                        && a.fields.iter().zip(&b.fields).all(|(x, y)| {
                            x.first_bit == y.first_bit && x.width == y.width && x.label == y.label
                        })
                };
                if !same(first, &d) {
                    return Err(format!(
                        "`{}` is defined twice, at lines {} and {}, and the two \
                         definitions differ. Lua keeps the second silently; decide \
                         which is right rather than inheriting that.",
                        d.key, first.line, d.line
                    ));
                }
            }
        }
    }

    Ok(order
        .into_iter()
        .map(|k| seen.remove(&k).unwrap())
        .collect())
}

/// Per-diagram structural invariants.
pub fn structure(diagrams: &[Diagram]) -> Result<(), String> {
    let mut computed: Vec<&str> = Vec::new();

    for d in diagrams {
        for (i, f) in d.fields.iter().enumerate() {
            if f.last_bit() >= d.nbits as u32 {
                return Err(format!(
                    "line {}: `{}` field at bits {}..={} runs past the {}-bit word",
                    f.line,
                    d.key,
                    f.first_bit,
                    f.last_bit(),
                    d.nbits
                ));
            }
            for g in &d.fields[i + 1..] {
                if f.mask() & g.mask() != 0 {
                    return Err(format!(
                        "line {}: `{}` fields at bits {}..={} and {}..={} overlap",
                        f.line,
                        d.key,
                        f.first_bit,
                        f.last_bit(),
                        g.first_bit,
                        g.last_bit()
                    ));
                }
            }
            if let Label::Fixed { value } = &f.label {
                let max = if f.width >= 32 {
                    u32::MAX
                } else {
                    (1u32 << f.width) - 1
                };
                if *value > max {
                    return Err(format!(
                        "line {}: `{}` fixed value {value:#x} does not fit the {}-bit \
                         field at bit {}",
                        f.line, d.key, f.width, f.first_bit
                    ));
                }
            }
            if matches!(f.label, Label::Computed { .. }) {
                computed.push(&d.key);
            }
        }
    }

    // `RMWCIB`'s `0xB3 + Index1` is the only opcode in the instruction set that
    // carries an operand. A second one would need its own decision about how to
    // split the opcode from the field, so it stops the build rather than being
    // handled by a rule invented for one case.
    match computed.as_slice() {
        ["RMWCIB"] => Ok(()),
        [] => Err("`RMWCIB`'s computed opcode has disappeared from Bits32.lua".into()),
        other => Err(format!(
            "expected exactly one computed opcode (`RMWCIB`), found {}: {}",
            other.len(),
            other.join(", ")
        )),
    }
}

/// Diagram fields the documented macro form does not expose.
///
/// A macro that takes fewer operands than the diagram draws is not a
/// disagreement: it is the documented calling convention pinning a field. `ZEROACC`
/// draws `Revert`, but `TT_ZEROACC` does not offer it; the six ADC instructions all
/// draw `ThreadOverride`, which ttsim reports as unsupported anyway.
///
/// Each row is `(diagram key, field name, why)`. Kept as data rather than a rule so
/// that a *new* omission is a finding: the specification growing a pinned field is
/// something to notice, not something to absorb.
const PINNED_BY_MACRO: &[(&str, &str, &str)] = &[
    (
        "SFPLUTFP32_BH",
        "Mod1Mirror",
        "a copy of Mod1 the macro fills in",
    ),
    ("ZEROACC", "Revert", "not offered by TT_ZEROACC"),
    ("ATSWAP", "SingleDataReg", "not offered by TT_ATSWAP"),
    (
        "SETADCXY",
        "ThreadOverride",
        "not offered; ttsim reports it unsupported",
    ),
    (
        "SETADCZW",
        "ThreadOverride",
        "not offered; ttsim reports it unsupported",
    ),
    (
        "INCADCXY",
        "ThreadOverride",
        "not offered; ttsim reports it unsupported",
    ),
    (
        "INCADCZW",
        "ThreadOverride",
        "not offered; ttsim reports it unsupported",
    ),
    (
        "ADDRCRXY",
        "ThreadOverride",
        "not offered; ttsim reports it unsupported",
    ),
    (
        "ADDRCRZW",
        "ThreadOverride",
        "not offered; ttsim reports it unsupported",
    ),
];

/// Slots where the macro passes an operand the diagram names differently.
///
/// One case: the `SFPSTOCHRND` pages document forms that feed `VC` into the slot
/// the diagram calls `VB`, because those variants use one register for both. The
/// bit positions agree; only the name the caller is told to pass differs.
///
/// Each row is `(diagram key, diagram name, macro name, why)`.
const RENAMED_BY_MACRO: &[(&str, &str, &str, &str)] = &[(
    "SFPSTOCHRND_BH",
    "VB",
    "VC",
    "the documented variants pass one register into both slots",
)];

/// What the cross-check found, for reporting.
pub struct CrossCheck {
    /// (key, macro) pairs compared.
    pub pairs: usize,
    /// Diagram keys no page embeds — none today.
    pub unembedded: Vec<String>,
    /// Keys whose page carries no `## Syntax` block, so only the diagram speaks.
    pub unchecked: Vec<String>,
    used_pinned: BTreeSet<(&'static str, &'static str)>,
    used_renamed: BTreeSet<(&'static str, &'static str, &'static str)>,
}

/// Check every diagram against the macro template on the page that embeds it.
///
/// The three comparisons, in increasing order of how hard they are to satisfy by
/// accident: field names, field widths and signedness, and — the one that matters —
/// **slot base agreement**. A macro argument is one slot of the word and `<< k`
/// places a term inside it, so each term implies `first_bit - k` for its slot's
/// base. Every term in a slot must imply the same base. Three terms agreeing on a
/// base is not something a transcription error produces.
pub fn cross_check(diagrams: &[Diagram], pages: &[Page]) -> Result<CrossCheck, String> {
    let by_key: BTreeMap<&str, &Diagram> = diagrams.iter().map(|d| (d.key.as_str(), d)).collect();
    let mut used_pinned = BTreeSet::new();
    let mut used_renamed = BTreeSet::new();
    let mut embedded = BTreeSet::new();
    let mut unchecked = BTreeSet::new();
    let mut pairs = 0usize;

    for page in pages {
        for key in &page.keys {
            embedded.insert(key.clone());
            if !by_key.contains_key(key.as_str()) {
                return Err(format!(
                    "{} embeds Bits32_{key}.svg, which Bits32.lua does not define",
                    page.path
                ));
            }
        }
        if page.calls.is_empty() {
            unchecked.extend(page.keys.iter().cloned());
            continue;
        }

        // `RMWCIB` is one diagram with four macros -- `TT_RMWCIB0` through
        // `TT_RMWCIB3` -- because the digit is part of the opcode. They differ only
        // in that digit, so checking the first checks all four.
        let calls: Vec<&MacroCall> = if page.keys.len() == page.calls.len() {
            page.calls.iter().collect()
        } else if page.keys.len() == 1 && page.calls.iter().all(|c| c.slots == page.calls[0].slots)
        {
            vec![&page.calls[0]]
        } else {
            return Err(format!(
                "{}: {} diagram(s) but {} macro(s), and they are not all the same shape",
                page.path,
                page.keys.len(),
                page.calls.len()
            ));
        };

        for (key, call) in page.keys.iter().zip(calls) {
            let diagram = by_key[key.as_str()];
            compare(page, diagram, call, &mut used_pinned, &mut used_renamed)?;
            pairs += 1;
        }
    }

    Ok(CrossCheck {
        pairs,
        unembedded: diagrams
            .iter()
            .map(|d| d.key.clone())
            .filter(|k| !embedded.contains(k))
            .collect(),
        unchecked: unchecked.into_iter().collect(),
        used_pinned,
        used_renamed,
    })
}

/// Exceptions the cross-check did not need.
///
/// Only meaningful over the whole specification -- a single page naturally uses
/// almost none of them -- so it is a separate call rather than part of
/// [`cross_check`]. An exception nobody needs is as much a finding as a missing
/// one: it means the specification changed shape somewhere nobody looked.
pub fn stale_exceptions(report: &CrossCheck) -> Vec<String> {
    PINNED_BY_MACRO
        .iter()
        .map(|(k, f, _)| (*k, *f))
        .filter(|kf| !report.used_pinned.contains(kf))
        .map(|(k, f)| format!("PINNED_BY_MACRO {k}.{f}"))
        .chain(
            RENAMED_BY_MACRO
                .iter()
                .map(|(k, a, b, _)| (*k, *a, *b))
                .filter(|kab| !report.used_renamed.contains(kab))
                .map(|(k, a, b)| format!("RENAMED_BY_MACRO {k}.{a}->{b}")),
        )
        .collect()
}

fn compare(
    page: &Page,
    diagram: &Diagram,
    call: &MacroCall,
    used_pinned: &mut BTreeSet<(&'static str, &'static str)>,
    used_renamed: &mut BTreeSet<(&'static str, &'static str, &'static str)>,
) -> Result<(), String> {
    let where_ = format!("{} / Bits32_{}", page.path, diagram.key);

    // Slots run from the high bits down, and so do the diagram's named fields.
    let mut drawn: Vec<&DrawnField> = diagram
        .fields
        .iter()
        .filter(|f| matches!(f.label, Label::Named { .. }))
        .collect();
    drawn.sort_by_key(|f| std::cmp::Reverse(f.first_bit));

    let pinned: Vec<&DrawnField> = drawn
        .iter()
        .copied()
        .filter(|f| {
            let name = field_name(f);
            PINNED_BY_MACRO
                .iter()
                .find(|(k, n, _)| *k == diagram.key && *n == name)
                .inspect(|(k, n, _)| {
                    used_pinned.insert((*k, *n));
                })
                .is_some()
        })
        .collect();
    drawn.retain(|f| !pinned.iter().any(|p| std::ptr::eq(*p, *f)));

    let named = call.named();
    if named.len() != drawn.len() {
        return Err(format!(
            "{where_}: the diagram draws {} operand field(s) {:?}, the macro takes {} {:?}",
            drawn.len(),
            drawn.iter().map(field_name).collect::<Vec<_>>(),
            named.len(),
            named.iter().map(|(_, t)| term_name(t)).collect::<Vec<_>>(),
        ));
    }

    let mut slot_bases: BTreeMap<usize, BTreeSet<i32>> = BTreeMap::new();
    for (field, (slot, term)) in drawn.iter().zip(&named) {
        let (name, width, signed, shift) = match term {
            Term::Named {
                name,
                width,
                signed,
                shift,
            } => (name.as_str(), *width, *signed, *shift),
            Term::Literal { .. } => unreachable!("named() filters to Term::Named"),
        };
        let drawn_name = field_name(field);

        if drawn_name != name {
            let known = RENAMED_BY_MACRO
                .iter()
                .find(|(k, a, b, _)| *k == diagram.key && *a == drawn_name && *b == name);
            match known {
                Some((k, a, b, _)) => {
                    used_renamed.insert((*k, *a, *b));
                }
                None => {
                    return Err(format!(
                        "{where_}: the diagram calls bits {}..={} `{drawn_name}`, the macro \
                         calls it `{name}`",
                        field.first_bit,
                        field.last_bit()
                    ))
                }
            }
        }
        if field.width != width {
            return Err(format!(
                "{where_}: `{drawn_name}` is {} bits in the diagram and {width} in the macro",
                field.width
            ));
        }
        if syntax::label_is_signed(&field.label) != signed {
            return Err(format!(
                "{where_}: the two sources disagree about whether `{drawn_name}` is signed \
                 (diagram: {}, macro: {signed})",
                syntax::label_is_signed(&field.label)
            ));
        }

        slot_bases
            .entry(*slot)
            .or_default()
            .insert(field.first_bit as i32 - shift as i32);
    }

    // The check worth having: within one macro argument, every term's shift must
    // place it at the position the diagram gives, relative to a single slot base.
    for (slot, bases) in slot_bases {
        if bases.len() > 1 {
            return Err(format!(
                "{where_}: argument {slot} packs terms whose shifts imply different slot \
                 bases {bases:?} — the two sources disagree about where those fields sit"
            ));
        }
    }
    Ok(())
}

fn field_name<'a>(f: &&'a DrawnField) -> &'a str {
    match &f.label {
        Label::Named { name, .. } => name,
        _ => unreachable!("filtered to Label::Named"),
    }
}

fn term_name<'a>(t: &&'a Term) -> &'a str {
    match t {
        Term::Named { name, .. } => name,
        Term::Literal { .. } => "<literal>",
    }
}
