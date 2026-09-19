//! Invariants every parsed diagram must satisfy.
//!
//! These need no second source to be worth running: they catch parser bugs and
//! upstream changes that are syntactically fine but semantically new. Each one is
//! exercised by a mutation test in [`super::tests`] — an invariant nobody has
//! watched reject something is not yet evidence.

use std::collections::BTreeMap;

use super::model::{Diagram, Label};

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
