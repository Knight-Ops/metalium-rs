//! Emit `crates/tt-isa/src/isa/generated.rs`.
//!
//! Two layers, because they are not interchangeable. The **table** is what makes
//! table-wide invariants assertable — no two mode variants ambiguous, every field
//! inside its word — and what any future tooling would read. The per-instruction
//! **`const fn` encoders** are the call-site API, and they must be `const fn`
//! because the firmware encodes instructions at run time on the device, where there
//! is no allocator and no lookup by name.
//!
//! Encoders take their operands from the high bits of the word down. That is the
//! order the diagrams list fields in and the order the syntax blocks pass them, so
//! a call site reads like the specification.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use super::model::{Diagram, Label};
use super::provenance::Provenance;

/// Above this many operands, a positional call is a bug farm and the encoder gets
/// a builder instead. `UNPACR_Regular` has thirteen.
const POSITIONAL_LIMIT: usize = 4;

/// One emitted encoding. Usually one per diagram; `RMWCIB` is four.
struct Entry<'a> {
    diagram: &'a Diagram,
    /// The `Bits32.lua` key, or the variant name where one diagram is several
    /// encodings.
    key: String,
    /// The `TT_*` macro name for this encoding specifically.
    mnemonic: String,
    opcode: u8,
}

pub struct Input<'a> {
    pub diagrams: &'a [Diagram],
    pub provenance: &'a BTreeMap<String, (Provenance, String)>,
    pub mnemonics: &'a BTreeMap<String, String>,
    pub variants: &'a BTreeMap<String, Vec<String>>,
    pub spec_rev: &'a str,
}

pub fn render(input: &Input<'_>) -> Result<String, String> {
    let mut instructions: Vec<Entry<'_>> = Vec::new();
    let mut layouts = Vec::new();
    for d in input.diagrams {
        let Some(opcode_field) = d.opcode_field() else {
            layouts.push(d);
            continue;
        };
        match &opcode_field.label {
            Label::Fixed { value } => instructions.push(Entry {
                diagram: d,
                key: d.key.clone(),
                mnemonic: input
                    .mnemonics
                    .get(&d.key)
                    .cloned()
                    .unwrap_or_else(|| d.key.clone()),
                opcode: *value as u8,
            }),
            // `0xB3 + Index1` is an *addition*, not a bitfield: the four opcodes
            // are 0xB3 through 0xB6, and 0xB0, 0xB1 and 0xB2 belong to other
            // instructions entirely. Treating the digit as bits of the opcode byte
            // would have produced 0xB0..0xB3 and collided with three of them.
            //
            // The specification documents this as four macros, `TT_RMWCIB0`
            // through `TT_RMWCIB3`, so it is emitted as four encodings.
            Label::Computed { base, addend } => {
                let names = input.variants.get(&d.key).ok_or_else(|| {
                    format!(
                        "`{}` has the computed opcode `{base:#x} + {addend}` but its page \
                         documents no macro variants, so there is nothing to say how many \
                         encodings it is",
                        d.key
                    )
                })?;
                for (i, name) in names.iter().enumerate() {
                    let opcode = u8::try_from(*base as usize + i).map_err(|_| {
                        format!("`{}` variant {i} overflows the opcode byte", d.key)
                    })?;
                    instructions.push(Entry {
                        diagram: d,
                        key: name.clone(),
                        mnemonic: name.clone(),
                        opcode,
                    });
                }
            }
            Label::Named { .. } => return Err(format!("`{}`'s opcode field is an operand", d.key)),
        }
    }

    // Blackhole's form of an instruction takes the plain name; the Wormhole form it
    // supersedes is reachable, but only through `wormhole::`. That way `sfpstore`
    // cannot accidentally mean the encoding with `AddrMod` in the wrong place.
    let mut names: BTreeMap<&str, (String, bool)> = BTreeMap::new();
    let mut taken: BTreeMap<String, &str> = BTreeMap::new();
    for e in &instructions {
        let (prov, _) = &input.provenance[&e.diagram.key];
        let superseded = matches!(prov, Provenance::SupersededOnBlackhole { .. });
        let base = if superseded {
            e.key.as_str()
        } else {
            e.key.strip_suffix("_BH").unwrap_or(&e.key)
        };
        let name = base.to_string();
        let scope = if superseded { "wormhole::" } else { "" };
        let scoped = format!("{scope}{name}");
        if let Some(other) = taken.insert(scoped.clone(), &e.key) {
            return Err(format!(
                "`{other}` and `{}` both want the name `{scoped}`",
                e.key
            ));
        }
        names.insert(&e.key, (name, superseded));
    }

    let mut out = String::new();
    header(&mut out, input, instructions.len(), layouts.len());

    writeln!(out, "use super::{{DatumLayout, EncodeError, Field, Instruction, InstructionDef, Provenance}};\n").unwrap();

    // --- definitions -------------------------------------------------------
    writeln!(
        out,
        "/// Every instruction encoding, by name. Blackhole's form takes the plain\n\
         /// name; the Wormhole form it supersedes is under [`defs::wormhole`].\n\
         pub mod defs {{\n    use super::*;\n"
    )
    .unwrap();
    for e in &instructions {
        let (name, superseded) = &names[e.key.as_str()];
        if *superseded {
            continue;
        }
        emit_def(&mut out, input, e, name, "    ");
    }
    writeln!(
        out,
        "    /// Encodings Blackhole replaces. Present so that the difference is\n\
         \x20   /// visible and testable, not so that they can be used here.\n\
         \x20   pub mod wormhole {{\n        use super::super::*;\n"
    )
    .unwrap();
    for e in &instructions {
        let (name, superseded) = &names[e.key.as_str()];
        if !*superseded {
            continue;
        }
        emit_def(&mut out, input, e, name, "        ");
    }
    writeln!(out, "    }}\n}}\n").unwrap();

    // --- flat table --------------------------------------------------------
    writeln!(
        out,
        "/// Every instruction encoding, for table-wide checks.\n\
         pub static ALL: &[&InstructionDef] = &["
    )
    .unwrap();
    for e in &instructions {
        let (name, superseded) = &names[e.key.as_str()];
        let path = if *superseded {
            format!("defs::wormhole::{name}")
        } else {
            format!("defs::{name}")
        };
        writeln!(out, "    &{path},").unwrap();
    }
    writeln!(out, "];\n").unwrap();

    // --- datum layouts -----------------------------------------------------
    writeln!(
        out,
        "/// The documented bit layout of each datum type in `Src` and `Dst`.\n\
         ///\n\
         /// The coprocessor does not entirely follow IEEE 754, so conversions are\n\
         /// built against these rather than against assumptions.\n\
         pub mod datum {{\n    use super::*;\n"
    )
    .unwrap();
    for d in &layouts {
        emit_layout(&mut out, input, d);
    }
    writeln!(out, "}}\n").unwrap();
    writeln!(out, "/// Every datum layout, for table-wide checks.\npub static ALL_LAYOUTS: &[&DatumLayout] = &[").unwrap();
    for d in &layouts {
        writeln!(out, "    &datum::{},", d.key.to_uppercase()).unwrap();
    }
    writeln!(out, "];\n").unwrap();

    // --- encoders ----------------------------------------------------------
    writeln!(
        out,
        "/// One encoder per instruction, taking its operands from the high bits of\n\
         /// the word down — the order the specification lists them in.\n\
         pub mod encode {{\n    use super::*;\n"
    )
    .unwrap();
    for e in &instructions {
        let (name, superseded) = &names[e.key.as_str()];
        if *superseded {
            continue;
        }
        emit_encoder(&mut out, e, name, "defs", "    ");
    }
    writeln!(
        out,
        "    /// Encoders for the forms Blackhole replaces. Reaching one has to be\n\
         \x20   /// deliberate.\n\
         \x20   pub mod wormhole {{\n        use super::super::*;\n"
    )
    .unwrap();
    for e in &instructions {
        let (name, superseded) = &names[e.key.as_str()];
        if !*superseded {
            continue;
        }
        emit_encoder(&mut out, e, name, "defs::wormhole", "        ");
    }
    writeln!(out, "    }}\n}}").unwrap();

    Ok(out)
}

fn header(out: &mut String, input: &Input<'_>, instructions: usize, layouts: usize) {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (p, _) in input.provenance.values() {
        *counts.entry(p.variant_name()).or_default() += 1;
    }
    writeln!(
        out,
        "//! The Tensix instruction set, generated from `Diagrams/Src/Bits32.lua`.\n\
         //!\n\
         //! **Do not edit.** Regenerate with `cargo xtask gen-isa`; CI checks that this\n\
         //! file matches the specification revision pinned in `PINS.toml`.\n\
         //!\n\
         //! Source: tt-isa-documentation `{}`,\n\
         //! {instructions} instruction encodings and {layouts} datum layouts, each\n\
         //! cross-checked against the `TT_*(…)` syntax block on the page that embeds\n\
         //! its diagram — an independently written description of the same bits.\n\
         //!\n\
         //! Provenance: {} documented for Blackhole, {} shared with Wormhole and stated\n\
         //! to be identical, {} superseded on Blackhole, {} Wormhole-only and therefore\n\
         //! **`UNVERIFIED`**, {} **`MEASURED`** on ttsim or silicon where the specification\n\
         //! draws only Wormhole's layout (`xtask/src/gen_isa/Bits32_BH.lua`), and {}\n\
         //! Wormhole-only layouts **`CONFIRMED`** unchanged on Blackhole by a gate\n\
         //! (`xtask/src/gen_isa/measured.rs`, `CONFIRMED`).\n\
         //!\n\
         //! Names are the `Bits32.lua` diagram keys, so a name in the specification can\n\
         //! be found here without translation — except that a Blackhole-specific form\n\
         //! drops its `_BH` suffix and the Wormhole form it replaces moves into the\n\
         //! `wormhole` submodule, so the plain name is always the one to use here.\n\
         //! Names keep the specification's spelling, so a name in the documentation can\n\
         //! be searched for here without translation. That is why this module allows\n\
         //! globals that are not upper case.\n\
         #![allow(non_upper_case_globals)]\n\
         #![allow(clippy::unreadable_literal)]\n",
        input.spec_rev,
        counts.get("Blackhole").copied().unwrap_or(0),
        counts.get("SharedWithWormhole").copied().unwrap_or(0),
        counts.get("SupersededOnBlackhole").copied().unwrap_or(0),
        counts.get("WormholeOnly").copied().unwrap_or(0),
        counts.get("Measured").copied().unwrap_or(0),
        counts.get("Confirmed").copied().unwrap_or(0),
    )
    .unwrap();
}

fn operands(d: &Diagram) -> Vec<(&str, &super::model::DrawnField)> {
    let mut v: Vec<_> = d
        .fields
        .iter()
        .filter_map(|f| match &f.label {
            Label::Named { name, .. } => Some((name.as_str(), f)),
            _ => None,
        })
        .collect();
    v.sort_by_key(|(_, f)| std::cmp::Reverse(f.first_bit));
    v
}

/// `Field::new(name, first_bit, width, signed, part)`.
///
/// `signed` stays its own parameter rather than being derived from `part` at
/// runtime: `Field::signed` is `const fn` on the encode path, and a string compare
/// there would be a needless risk.
///
/// Every other parenthesised qualifier becomes `part`. Dropping it is what made
/// `Dst32_FP32` carry two fields both called `Mantissa`, with nothing to say which
/// run was which -- and the runs do not reassemble in bit order, so the qualifier
/// is the only record of how they fit together.
fn field_expr(name: &str, f: &super::model::DrawnField) -> String {
    let (signed, part) = match &f.label {
        Label::Named { note: Some(n), .. } if n == "signed" => (true, "None".to_string()),
        Label::Named { note: Some(n), .. } => (false, format!("Some({n:?})")),
        _ => (false, "None".to_string()),
    };
    format!(
        "Field::new(\"{name}\", {}, {}, {signed}, {part})",
        f.first_bit, f.width
    )
}

fn emit_def(out: &mut String, input: &Input<'_>, e: &Entry<'_>, name: &str, pad: &str) {
    let d = e.diagram;
    let (prov, page) = &input.provenance[&d.key];

    let ops = operands(d);
    let fields: Vec<String> = ops.iter().map(|(n, f)| field_expr(n, f)).collect();
    let fixed: Vec<String> = d
        .fields
        .iter()
        .filter_map(|f| match &f.label {
            // The opcode is carried separately.
            Label::Fixed { .. } if f.first_bit == 24 && f.width == 8 => None,
            Label::Fixed { value } => Some(format!(
                "(Field::new(\"\", {}, {}, false, None), {value})",
                f.first_bit, f.width
            )),
            _ => None,
        })
        .collect();

    let prov_expr = match prov {
        Provenance::SupersededOnBlackhole { by } => {
            format!("Provenance::SupersededOnBlackhole {{ by: \"{by}\" }}")
        }
        Provenance::Measured {
            evidence,
            moved,
            dropped,
            widened,
        } => {
            let quoted = |v: &Vec<String>| {
                v.iter()
                    .map(|m| format!("\"{m}\""))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            format!(
                "Provenance::Measured {{ evidence: \"{evidence}\", moved: &[{}], dropped: &[{}], widened: &[{}] }}",
                quoted(moved),
                quoted(dropped),
                quoted(widened)
            )
        }
        Provenance::Confirmed { evidence } => {
            format!("Provenance::Confirmed {{ evidence: \"{evidence}\" }}")
        }
        other => format!("Provenance::{}", other.variant_name()),
    };
    let mnemonic = &e.mnemonic;

    let doc = match prov {
        Provenance::Blackhole => format!("Documented for Blackhole in `{page}`."),
        Provenance::SharedWithWormhole => {
            format!("`{page}`, which the Blackhole tree states is shared and identical.")
        }
        Provenance::SupersededOnBlackhole { by } => format!(
            "**Wormhole's encoding**, from `{page}`. Blackhole replaces it with `{by}`; \
             use that instead."
        ),
        Provenance::WormholeOnly => format!(
            "**`UNVERIFIED`.** `{page}` is a Wormhole page and Blackhole has none, so this \
             layout is a hypothesis until silicon or the simulator confirms it."
        ),
        Provenance::Confirmed { evidence } => format!(
            "**`CONFIRMED`** on Blackhole, ttsim and silicon, by `{evidence}`: the layout of \
             `{page}` (a Wormhole page; Blackhole has none), every field exercised."
        ),
        Provenance::Measured {
            evidence,
            moved,
            dropped,
            widened,
        } => {
            let ticked = |v: &Vec<String>| {
                v.iter()
                    .map(|m| format!("`{m}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let dropped = if dropped.is_empty() {
                String::new()
            } else {
                format!(
                    " {} {} not carried: {} bits hold something else on Blackhole and \
                     {} own position is unknown.",
                    ticked(dropped),
                    if dropped.len() == 1 { "is" } else { "are" },
                    if dropped.len() == 1 { "its" } else { "their" },
                    if dropped.len() == 1 { "its" } else { "their" },
                )
            };
            let widened = if widened.is_empty() {
                String::new()
            } else {
                format!(
                    " {} {} a different width on Blackhole.",
                    ticked(widened),
                    if widened.len() == 1 { "has" } else { "have" },
                )
            };
            let remaining = if matches!(e.key.as_str(), "ELWADD_BH" | "ELWSUB_BH" | "ELWMUL_BH") {
                "Other fields are carried from that diagram. Step9 and step90 validate broadcast, assignment, Dst addressing and repeated bank release on ttsim and both Blackhole cards; see silicon run 1791254100 and silicon-operating-notes.md. Floating arithmetic is not IEEE754."
            } else {
                "Every other field is carried from that diagram and is as unverified as it was. Re-derive on silicon."
            };
            format!(
                "**`MEASURED`** on ttsim or silicon by `{evidence}`, not documented: the only \
                 diagram is Wormhole's (`{page}`), and on Blackhole {} sit{} elsewhere.{widened}{dropped} \
                 {remaining}",
                ticked(moved),
                if moved.len() == 1 { "s" } else { "" }
            )
        }
    };

    let doc = match (prov, e.key.as_str()) {
        (Provenance::Measured { evidence, .. }, "UNPACR_NOP_SETDVALID_BH") => format!(
            "**`MEASURED`** on Blackhole silicon by `{evidence}`. Replaces Wormhole fixed mode 7 with the bounded non-clearing publication profile 0x1e9; only WhichUnpacker varies. C1/C2 retirement and C5/C6 ownership waits remain required. ttsim refuses this format-selector flavor."
        ),
        _ => doc,
    };

    writeln!(out, "{pad}/// `{}`. {doc}", e.key).unwrap();
    writeln!(
        out,
        "{pad}pub static {name}: InstructionDef = InstructionDef::new(\n\
         {pad}    \"{}\",\n{pad}    \"{mnemonic}\",\n{pad}    {:#04x},\n\
         {pad}    &[{}],\n{pad}    &[{}],\n{pad}    {:#010x},\n{pad}    {prov_expr},\n\
         {pad}    \"{page}\",\n{pad});",
        e.key,
        e.opcode,
        fields.join(", "),
        fixed.join(", "),
        unspecified(d),
    )
    .unwrap();
    out.push('\n');
}

/// Bits no field claims, excluding the opcode byte.
fn unspecified(d: &Diagram) -> u32 {
    d.undrawn() & 0x00FF_FFFF
}

fn emit_layout(out: &mut String, input: &Input<'_>, d: &Diagram) {
    let (_, page) = &input.provenance[&d.key];
    let fields: Vec<String> = operands(d).iter().map(|(n, f)| field_expr(n, f)).collect();
    // The must-be-zero padding `Src` formats carry between mantissa and exponent.
    // Without it the table describes 16 of a 19-bit datum, and code building a
    // datum has no way to learn which bits to leave alone. Datums have no opcode,
    // so unlike `emit_def` there is nothing to exclude here.
    let fixed: Vec<String> = d
        .fields
        .iter()
        .filter_map(|f| match &f.label {
            Label::Fixed { value } => Some(format!(
                "(Field::new(\"\", {}, {}, false, None), {value})",
                f.first_bit, f.width
            )),
            _ => None,
        })
        .collect();
    writeln!(out, "    /// `{}`, from `{page}`.", d.key).unwrap();
    writeln!(
        out,
        "    pub static {}: DatumLayout = DatumLayout::new(\"{}\", {}, &[{}], &[{}], \"{page}\");\n",
        d.key.to_uppercase(),
        d.key,
        d.nbits,
        fields.join(", "),
        fixed.join(", "),
    )
    .unwrap();
}

fn emit_encoder(out: &mut String, e: &Entry<'_>, name: &str, defs: &str, pad: &str) {
    let d = e.diagram;
    let args: Vec<(String, &super::model::DrawnField)> =
        operands(d).iter().map(|(n, f)| (snake(n), *f)).collect();
    let fn_name = snake(name);

    if args.len() <= POSITIONAL_LIMIT {
        let params: Vec<String> = args.iter().map(|(n, _)| format!("{n}: u32")).collect();
        writeln!(out, "{pad}/// `{}`.", e.key).unwrap();
        writeln!(
            out,
            "{pad}pub const fn {fn_name}({}) -> Result<Instruction, EncodeError> {{",
            params.join(", ")
        )
        .unwrap();
        writeln!(out, "{pad}    let def = &{defs}::{name};").unwrap();
        let binding = if args.is_empty() { "let" } else { "let mut" };
        writeln!(out, "{pad}    {binding} word = def.skeleton();").unwrap();
        for (i, (arg, _)) in args.iter().enumerate() {
            emit_place(out, pad, "    ", i, arg, arg);
        }
        writeln!(out, "{pad}    Ok(Instruction::new(word, def))\n{pad}}}\n").unwrap();
        return;
    }

    // Too many operands to pass positionally without inviting a transposition.
    let ty = camel(name);
    writeln!(
        out,
        "{pad}/// `{}`, built field by field.\n\
         {pad}///\n\
         {pad}/// {} operands is too many to pass positionally without inviting a\n\
         {pad}/// transposition, so each is named: `{ty}::ZERO.{}(1).encode()`.",
        e.key,
        args.len(),
        args[0].0
    )
    .unwrap();
    writeln!(out, "{pad}#[derive(Copy, Clone, Debug, Eq, PartialEq)]").unwrap();
    writeln!(out, "{pad}pub struct {ty} {{").unwrap();
    for (arg, _) in &args {
        writeln!(out, "{pad}    {arg}: u32,").unwrap();
    }
    writeln!(out, "{pad}}}\n").unwrap();
    writeln!(out, "{pad}impl {ty} {{").unwrap();
    writeln!(
        out,
        "{pad}    /// Every operand zero. Fixed bits are added by [`Self::encode`]."
    )
    .unwrap();
    writeln!(out, "{pad}    pub const ZERO: Self = {ty} {{").unwrap();
    for (arg, _) in &args {
        writeln!(out, "{pad}        {arg}: 0,").unwrap();
    }
    writeln!(out, "{pad}    }};\n").unwrap();
    for (arg, _) in &args {
        writeln!(
            out,
            "{pad}    pub const fn {arg}(mut self, value: u32) -> Self {{\n\
             {pad}        self.{arg} = value;\n\
             {pad}        self\n\
             {pad}    }}\n"
        )
        .unwrap();
    }
    writeln!(
        out,
        "{pad}    pub const fn encode(self) -> Result<Instruction, EncodeError> {{"
    )
    .unwrap();
    writeln!(out, "{pad}        let def = &{defs}::{name};").unwrap();
    writeln!(out, "{pad}        let mut word = def.skeleton();").unwrap();
    for (i, (arg, _)) in args.iter().enumerate() {
        emit_place(out, pad, "        ", i, arg, &format!("self.{arg}"));
    }
    writeln!(
        out,
        "{pad}        Ok(Instruction::new(word, def))\n{pad}    }}\n{pad}}}\n"
    )
    .unwrap();
}

/// Check one operand fits its field, then place it.
fn emit_place(out: &mut String, pad: &str, indent: &str, i: usize, _name: &str, value: &str) {
    writeln!(
        out,
        "{pad}{indent}let f = def.fields()[{i}];\n\
         {pad}{indent}if !f.fits({value}) {{\n\
         {pad}{indent}    return Err(EncodeError::FieldTooLarge {{\n\
         {pad}{indent}        instruction: def.key(),\n\
         {pad}{indent}        field: f.name(),\n\
         {pad}{indent}        value: {value},\n\
         {pad}{indent}        width: f.width(),\n\
         {pad}{indent}    }});\n\
         {pad}{indent}}}\n\
         {pad}{indent}word |= f.place({value});"
    )
    .unwrap();
}

/// `SFPSTORE_BH` -> `sfpstore_bh`, `WhichPackers` -> `which_packers`.
fn snake(s: &str) -> String {
    let mut out = String::new();
    let chars: Vec<char> = s.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if c.is_ascii_uppercase() {
            let prev_lower = i > 0 && chars[i - 1].is_ascii_lowercase();
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_lower = i + 1 < chars.len() && chars[i + 1].is_ascii_lowercase();
            let prev_upper = i > 0 && chars[i - 1].is_ascii_uppercase();
            if (prev_lower || prev_digit || (prev_upper && next_lower)) && !out.ends_with('_') {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(*c);
        }
    }
    let out = out.to_ascii_lowercase();
    // `type`, `mod` and friends cannot be identifiers.
    if matches!(out.as_str(), "type" | "mod" | "ref" | "move" | "box" | "in") {
        format!("{out}_")
    } else {
        out
    }
}

fn camel(s: &str) -> String {
    snake(s)
        .split('_')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut c = p.chars();
            match c.next() {
                Some(f) => f.to_ascii_uppercase().to_string() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// Keys that must not collide once `_BH` suffixes are dropped.
#[allow(dead_code)]
pub fn name_collisions(diagrams: &[Diagram]) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut dup = BTreeSet::new();
    for d in diagrams {
        let base = d.key.strip_suffix("_BH").unwrap_or(&d.key);
        if !seen.insert(base.to_string()) {
            dup.insert(base.to_string());
        }
    }
    dup
}
