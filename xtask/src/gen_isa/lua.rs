//! A parser for the `diagrams` table of `Diagrams/Src/Bits32.lua`.
//!
//! # Why this is hand-rolled rather than run through LuaJIT
//!
//! `Bits32.lua` is a real Lua program — it renders SVG — so the obvious thing is to
//! run it and read the table out. Three reasons not to:
//!
//! * It needs **LuaJIT specifically** (`require"string.buffer"`), which is a fourth
//!   toolchain to pin and install in CI for one file.
//! * A Lua interpreter would **silently absorb** exactly the upstream changes this
//!   wants to catch. If a new key appeared, or a label took a new shape, `eval`
//!   would evaluate it happily and the generator would quietly emit something
//!   different. A parser that refuses to guess is the point.
//! * It only solves the easy half. The cross-check against the specification's
//!   `TT_*(…)` syntax blocks needs a C-expression parser regardless, and that is
//!   the harder one.
//!
//! So: same approach as `crate::gen_cfg`'s `cfg_defines.h` parser, with one
//! deliberate difference. That one `continue`s past lines it does not recognise;
//! this one **fails**. Skipping is what makes a parser survive an upstream format
//! change by quietly dropping data, and dropped instructions are precisely what
//! this generator must never do.
//!
//! # The grammar, in full
//!
//! ```lua
//! local diagrams = {
//!   KEY = function()
//!     return Bits32{                       -- optionally `{nbits = N,` or `{extra_w = N,`
//!       {first_bit, width, "label"},       -- optionally `, y = N` and/or `, edge = "…"`
//!       ...
//!     }
//!   end,
//! }
//! ```
//!
//! `y` and `edge` place the label text in the rendered SVG — a leader line and a
//! text anchor. They carry no encoding information and are discarded. `extra_w`
//! widens the drawing. Any *other* key is an error, so that a semantically
//! meaningful addition upstream cannot be discarded along with the cosmetics.

use super::model::{Diagram, DrawnField, Label};

const TABLE_OPEN: &str = "local diagrams = {";

/// Parse every entry of the `diagrams` table.
///
/// Fails on anything it does not recognise, including a key it has no grammar for
/// and a table entirely empty of diagrams.
pub fn parse(source: &str) -> Result<Vec<Diagram>, String> {
    let lines: Vec<&str> = source.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.trim_end() == TABLE_OPEN)
        .ok_or_else(|| format!("`{TABLE_OPEN}` not found — is this Bits32.lua?"))?;

    let mut diagrams: Vec<Diagram> = Vec::new();
    let mut i = start + 1;
    loop {
        let line = *lines
            .get(i)
            .ok_or_else(|| "the diagrams table is never closed".to_string())?;
        // A `}` at the left margin closes the table; everything inside is indented.
        if line.starts_with('}') {
            break;
        }
        if line.trim().is_empty() {
            i += 1;
            continue;
        }
        let (diagram, next) = parse_entry(&lines, i)?;
        diagrams.push(diagram);
        i = next;
    }

    if diagrams.is_empty() {
        return Err("the diagrams table is empty".into());
    }
    Ok(diagrams)
}

/// Parse one `KEY = function() … end,`, returning it and the line after it.
fn parse_entry(lines: &[&str], start: usize) -> Result<(Diagram, usize), String> {
    let at = |n: usize| n + 1; // human line numbers

    let head = lines[start].trim();
    let key = head
        .strip_suffix(" = function()")
        .filter(|k| !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
        .ok_or_else(|| {
            format!(
                "line {}: expected `KEY = function()`, found `{head}`",
                at(start)
            )
        })?
        .to_string();

    let open = lines
        .get(start + 1)
        .map(|l| l.trim())
        .ok_or_else(|| format!("line {}: `{key}` is never closed", at(start)))?;
    let rest = open.strip_prefix("return Bits32{").ok_or_else(|| {
        format!(
            "line {}: expected `return Bits32{{`, found `{open}`",
            at(start + 1)
        )
    })?;

    // `nbits` and `extra_w` appear on the opening line when present.
    let mut nbits = 32u8;
    for assignment in rest.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, value) = assignment
            .split_once('=')
            .map(|(a, b)| (a.trim(), b.trim()))
            .ok_or_else(|| {
                format!(
                    "line {}: expected `name = value` in the Bits32 table, found `{assignment}`",
                    at(start + 1)
                )
            })?;
        match name {
            "nbits" => {
                nbits = parse_u32(value)
                    .and_then(|v| u8::try_from(v).ok())
                    .ok_or_else(|| format!("line {}: bad nbits `{value}`", at(start + 1)))?;
            }
            // Widens the drawing to fit long labels. Cosmetic.
            "extra_w" => {}
            other => {
                return Err(format!(
                    "line {}: unknown Bits32 table key `{other}` — \
                     it may be meaningful rather than cosmetic, so this refuses to \
                     discard it",
                    at(start + 1)
                ))
            }
        }
    }
    if !matches!(nbits, 16 | 19 | 32) {
        return Err(format!(
            "line {}: `{key}` has nbits = {nbits}; only 16, 19 and 32 are known",
            at(start + 1)
        ));
    }

    let mut fields = Vec::new();
    let mut i = start + 2;
    loop {
        let line = lines
            .get(i)
            .ok_or_else(|| format!("line {}: `{key}` is never closed", at(start)))?;
        let trimmed = line.trim();
        if trimmed == "}" {
            i += 1;
            break;
        }
        if trimmed.is_empty() {
            i += 1;
            continue;
        }
        fields.push(parse_field(trimmed, at(i))?);
        i += 1;
    }

    let end = lines.get(i).map(|l| l.trim()).unwrap_or_default();
    if end != "end," {
        return Err(format!(
            "line {}: expected `end,` closing `{key}`, found `{end}`",
            at(i)
        ));
    }

    Ok((
        Diagram {
            key,
            nbits,
            fields,
            line: at(start),
        },
        i + 1,
    ))
}

/// Parse `{first_bit, width, "label"}` plus optional cosmetic keys.
fn parse_field(text: &str, line: usize) -> Result<DrawnField, String> {
    let body = text
        .strip_prefix('{')
        .and_then(|t| t.strip_suffix(','))
        .and_then(|t| t.strip_suffix('}'))
        .ok_or_else(|| format!("line {line}: expected `{{a, b, \"label\"}},`, found `{text}`"))?;

    // The label is the only quoted part, so splitting on the quotes separates the
    // two numbers from it without having to worry about commas inside the label.
    let (before, rest) = body
        .split_once('"')
        .ok_or_else(|| format!("line {line}: field has no quoted label: `{text}`"))?;
    let (label_text, after) = rest
        .split_once('"')
        .ok_or_else(|| format!("line {line}: unterminated label: `{text}`"))?;

    let nums: Vec<&str> = before
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if nums.len() != 2 {
        return Err(format!(
            "line {line}: expected two numbers before the label, found {}: `{text}`",
            nums.len()
        ));
    }
    let first_bit = parse_u32(nums[0])
        .and_then(|v| u8::try_from(v).ok())
        .ok_or_else(|| format!("line {line}: bad first_bit `{}`", nums[0]))?;
    let width = parse_u32(nums[1])
        .and_then(|v| u8::try_from(v).ok())
        .ok_or_else(|| format!("line {line}: bad width `{}`", nums[1]))?;
    if width == 0 {
        return Err(format!("line {line}: zero-width field: `{text}`"));
    }

    for assignment in after.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, _) = assignment
            .split_once('=')
            .map(|(a, b)| (a.trim(), b.trim()))
            .ok_or_else(|| format!("line {line}: expected `name = value`, found `{assignment}`"))?;
        match name {
            // A leader line dropping the label below the diagram, and the text
            // anchor. Both purely how the SVG looks.
            "y" | "edge" => {}
            other => {
                return Err(format!(
                    "line {line}: unknown field key `{other}` — it may be meaningful \
                     rather than cosmetic, so this refuses to discard it"
                ))
            }
        }
    }

    Ok(DrawnField {
        first_bit,
        width,
        label: parse_label(label_text, line)?,
        line,
    })
}

/// Labels that are none of the three general shapes, with what each one means and
/// where the specification says so.
///
/// This table is deliberately short, explicit and hard to grow. Every entry is a
/// judgement about what a piece of the instruction set encodes, so adding one is a
/// reviewed act rather than a regex quietly widening. Two rules keep it honest, both
/// exercised by mutation tests:
///
/// * remove an entry and the parser refuses the label, so no entry is decorative;
/// * add an entry nothing uses and [`unused`] refuses it, so the table cannot
///   accumulate leftovers from a specification bump.
///
/// Each row is `(label text, field name, note, why)`.
const SPECIAL_LABELS: &[(&str, &str, &str, &str)] = &[(
    "16 or 0",
    "Exponent",
    "16 or 0",
    // `Dst.md`: integer \"8\" is \"overlaid onto FP16, using a fixed raw exponent of
    // 16 (or sometimes a raw exponent of 0 when the magnitude is zero)\". The other
    // 16-bit layouts label this same bit range `Exponent`, so naming it that makes
    // the layouts comparable rather than inventing a name.
    "Dst.md, `Data type bit layout`: the raw exponent of an integer datum",
)];

/// Which [`SPECIAL_LABELS`] entries no longer appear in the source.
///
/// An exception that has stopped being needed is as much a finding as a missing
/// one: it means the specification changed shape somewhere nobody looked.
pub fn unused(source: &str) -> Vec<&'static str> {
    SPECIAL_LABELS
        .iter()
        .filter(|(text, ..)| !source.contains(&format!("\"{text}\"")))
        .map(|(text, ..)| *text)
        .collect()
}

/// Classify a diagram label.
///
/// The four shapes are described on [`Label`]. Anything else is an error: a label
/// this does not understand is a fact about the instruction set that would
/// otherwise be silently dropped.
fn parse_label(text: &str, line: usize) -> Result<Label, String> {
    let text = text.trim();
    if text.is_empty() {
        return Err(format!(
            "line {line}: empty label. No diagram uses one today; if upstream \
             introduces one, decide what it means rather than guessing here."
        ));
    }

    if let Some(value) = parse_u32(text) {
        return Ok(Label::Fixed { value });
    }

    // `0xB3 + Index1` — an opcode that carries an operand.
    if let Some((base, addend)) = text.split_once('+') {
        if let Some(base) = parse_u32(base.trim()) {
            let addend = addend.trim();
            if is_identifier(addend) {
                return Ok(Label::Computed {
                    base,
                    addend: addend.to_string(),
                });
            }
        }
    }

    // `Imm12 (signed)`, `Magnitude (low)`.
    let (name, note) = match text.split_once(" (") {
        Some((name, note)) => match note.strip_suffix(')') {
            Some(note) => (name, Some(note.to_string())),
            None => (text, None),
        },
        None => (text, None),
    };
    if is_identifier(name) {
        return Ok(Label::Named {
            name: name.to_string(),
            note,
        });
    }

    if let Some((_, name, note, _)) = SPECIAL_LABELS.iter().find(|(t, ..)| *t == text) {
        return Ok(Label::Named {
            name: (*name).to_string(),
            note: Some((*note).to_string()),
        });
    }

    Err(format!(
        "line {line}: unrecognised label `{text}`. It is neither an identifier, a \
         number, nor `0xNN + Name`, and SPECIAL_LABELS has no entry for it, so what \
         it encodes has to be decided rather than guessed."
    ))
}

fn is_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn parse_u32(s: &str) -> Option<u32> {
    let s = s.trim();
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) if !hex.is_empty() => u32::from_str_radix(hex, 16).ok(),
        Some(_) => None,
        None if s.chars().all(|c| c.is_ascii_digit()) && !s.is_empty() => s.parse().ok(),
        None => None,
    }
}
