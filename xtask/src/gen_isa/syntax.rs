//! The specification's second description of the instruction set.
//!
//! Every page that documents an encoding embeds its diagram —
//! `![](../../../Diagrams/Out/Bits32_<KEY>.svg)` — which is what maps a diagram key
//! to a page, mechanically and without guessing. Most of those pages also carry a
//! `## Syntax` block: a hand-written C macro call template for a human to copy.
//!
//! ```c
//! TT_SETDMAREG(/* u2 */ ResultSize,
//!            ((/* u4 */ WhichPackers) << 7) +
//!            ((/* u4 */ InputSource ) << 3) +
//!              /* u3 */ InputHalfReg,
//!              1,
//!              /* u7 */ ResultHalfReg)
//! ```
//!
//! # What this is worth checking against
//!
//! Field names and widths are the obvious content, and they are the weaker half.
//! The valuable part is the **shifts**. Each argument is one slot of the
//! instruction word, slots run from high bits to low, and `<< k` places a term
//! within its slot. So a slot with several terms states their *relative* positions
//! — and the diagram states their absolute ones. Subtracting gives the slot's base,
//! independently, once per term; every term in a slot must imply the same base.
//!
//! `SETDMAREG_Special` above is the example: the diagram puts `WhichPackers` at bit
//! 15, `InputSource` at 11 and `InputHalfReg` at 8, and the macro's shifts of 7, 3
//! and 0 make all three imply a slot base of 8. Nothing but agreement produces that.
//!
//! Two further things fall out for free: literal arguments (`TT_ATCAS(0, …, 0, …)`)
//! name bits the diagram leaves undrawn, and `/* iN */` states signedness, which the
//! diagram spells as an `(signed)` annotation on the label.

use super::model::Label;

/// One page of the specification that documents an encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    /// Path relative to the specification root, e.g.
    /// `BlackholeA0/TensixTile/TensixCoprocessor/PACR.md`.
    pub path: String,
    /// Diagram keys the page embeds, in the order it embeds them.
    pub keys: Vec<String>,
    /// Macro calls from the `## Syntax` block, in order. Empty if the page has no
    /// such block.
    pub calls: Vec<MacroCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroCall {
    /// Without the `TT_`, `TTI_` or `TT_OP_` prefix.
    pub name: String,
    /// One per argument, from the high bits of the word down.
    pub slots: Vec<Slot>,
}

pub type Slot = Vec<Term>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    /// `/* u4 */ WhichPackers`, possibly shifted within its slot.
    Named {
        name: String,
        width: u8,
        signed: bool,
        shift: u8,
    },
    /// A constant the caller is told to pass. These name bits the diagram leaves
    /// undrawn, which is the only place that information exists.
    Literal { value: u32, shift: u8 },
}

impl MacroCall {
    /// Named terms in order, paired with the index of the slot they came from.
    pub fn named(&self) -> Vec<(usize, &Term)> {
        self.slots
            .iter()
            .enumerate()
            .flat_map(|(i, slot)| {
                slot.iter()
                    .filter(|t| matches!(t, Term::Named { .. }))
                    .map(move |t| (i, t))
            })
            .collect()
    }
}

/// Parse one markdown page, returning `None` if it documents no encoding.
pub fn parse_page(path: &str, source: &str) -> Result<Option<Page>, String> {
    let keys = embedded_keys(source);
    if keys.is_empty() {
        return Ok(None);
    }
    let calls = match syntax_block(source) {
        Some(block) => parse_block(path, block)?,
        None => Vec::new(),
    };
    Ok(Some(Page {
        path: path.to_string(),
        keys,
        calls,
    }))
}

/// Diagram keys the page embeds, in order.
fn embedded_keys(source: &str) -> Vec<String> {
    const OPEN: &str = "Diagrams/Out/Bits32_";
    let mut keys = Vec::new();
    let mut rest = source;
    while let Some(i) = rest.find(OPEN) {
        rest = &rest[i + OPEN.len()..];
        if let Some(j) = rest.find(".svg") {
            keys.push(rest[..j].to_string());
            rest = &rest[j..];
        }
    }
    keys
}

/// The fenced block directly under `## Syntax`.
///
/// Scoped to that heading on purpose: `TT_*(…)` also appears in ordinary prose —
/// `SETC16.md` discusses `TT_SETC16(CFG_STATE_ID_StateID_ADDR32, x)` in its notes —
/// and a whole-file search would take those for encodings.
fn syntax_block(source: &str) -> Option<&str> {
    let heading = source
        .match_indices("\n## Syntax")
        .map(|(i, _)| i + 1)
        .next()?;
    let after = &source[heading..];
    let fence = after.find("```")?;
    let body = &after[fence..];
    let start = body.find('\n')? + 1;
    let end = body[start..].find("```")?;
    Some(&body[start..start + end])
}

/// Split a syntax block into macro calls and parse each.
fn parse_block(path: &str, block: &str) -> Result<Vec<MacroCall>, String> {
    let mut calls = Vec::new();
    let mut current = String::new();
    for line in block.lines() {
        if is_call_start(line) && !current.trim().is_empty() {
            calls.push(parse_call(path, &current)?);
            current.clear();
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        calls.push(parse_call(path, &current)?);
    }
    Ok(calls)
}

fn is_call_start(line: &str) -> bool {
    ["TT_", "TTI_", "TT_OP_"]
        .iter()
        .any(|p| line.starts_with(p))
}

fn parse_call(path: &str, text: &str) -> Result<MacroCall, String> {
    let text = text.trim().trim_end_matches(';').trim();
    let name_end = text.find('(').unwrap_or(text.len());
    let head = text[..name_end].trim();
    let name = ["TT_OP_", "TTI_", "TT_"]
        .iter()
        .find_map(|p| head.strip_prefix(p))
        .ok_or_else(|| format!("{path}: `{head}` is not a TT_/TTI_/TT_OP_ macro"))?;
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || name.is_empty() {
        return Err(format!("{path}: `{head}` is not a macro name"));
    }

    // An operandless instruction is written `TTI_SFPNOP`, with no argument list.
    if name_end == text.len() {
        return Ok(MacroCall {
            name: name.to_string(),
            slots: Vec::new(),
        });
    }
    let body = text[name_end..]
        .trim()
        .strip_prefix('(')
        .and_then(|b| b.strip_suffix(')'))
        .ok_or_else(|| format!("{path}: `{name}`'s argument list is not parenthesised"))?;

    let mut slots = Vec::new();
    for arg in split_top_level(body, ',') {
        let mut terms = Vec::new();
        for part in split_top_level(&arg, '+') {
            terms.push(parse_term(path, name, &part)?);
        }
        slots.push(terms);
    }
    Ok(MacroCall {
        name: name.to_string(),
        slots,
    })
}

/// Split on a separator that is not inside parentheses.
fn split_top_level(s: &str, sep: char) -> Vec<String> {
    let mut depth = 0i32;
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth -= 1;
                cur.push(c);
            }
            c if c == sep && depth == 0 => {
                out.push(std::mem::take(&mut cur));
            }
            c => cur.push(c),
        }
    }
    out.push(cur);
    out
}

fn parse_term(path: &str, macro_name: &str, text: &str) -> Result<Term, String> {
    let text = strip_parens(text.trim());

    // `(… ) << k` places the term within its slot.
    let (body, shift) = match find_top_level_shift(text) {
        Some(i) => {
            let n = text[i + 2..].trim();
            let shift = n
                .parse::<u8>()
                .map_err(|_| format!("{path}: `{macro_name}` has a non-numeric shift `{n}`"))?;
            (&text[..i], shift)
        }
        None => (text, 0u8),
    };
    let body = strip_parens(body.trim());

    // `/* u4 */ Name`, `/* i12 */ (Imm12 & 0xfff)`.
    if let Some(rest) = body.strip_prefix("/*") {
        let (ty, rest) = rest
            .split_once("*/")
            .ok_or_else(|| format!("{path}: `{macro_name}` has an unterminated comment"))?;
        let ty = ty.trim();
        let (signed, width) = match ty {
            "bool" => (false, 1u8),
            _ => {
                let (signed, digits) = match ty.split_at(1) {
                    ("u", d) => (false, d),
                    ("i", d) => (true, d),
                    _ => {
                        return Err(format!(
                            "{path}: `{macro_name}` has an unknown operand type `{ty}`"
                        ))
                    }
                };
                let width = digits.parse::<u8>().map_err(|_| {
                    format!("{path}: `{macro_name}` has an unknown operand type `{ty}`")
                })?;
                (signed, width)
            }
        };
        // A signed immediate is passed masked: `(Imm12 & 0xfff)`. The mask is how
        // the macro converts it, not part of the encoding.
        let rest = strip_parens(rest.trim());
        let name = rest.split('&').next().unwrap_or(rest).trim();
        if !is_identifier(name) {
            return Err(format!(
                "{path}: `{macro_name}` names its operand `{name}`, which is not an identifier"
            ));
        }
        return Ok(Term::Named {
            name: name.to_string(),
            width,
            signed,
            shift,
        });
    }

    let value = match body.trim() {
        "false" => 0,
        "true" => 1,
        n => parse_u32(n).ok_or_else(|| {
            format!("{path}: `{macro_name}` has an argument this does not understand: `{text}`")
        })?,
    };
    Ok(Term::Literal { value, shift })
}

/// Byte index of a `<<` that is not inside parentheses.
fn find_top_level_shift(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    for i in 0..bytes.len().saturating_sub(1) {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b'<' if depth == 0 && bytes[i + 1] == b'<' => return Some(i),
            _ => {}
        }
    }
    None
}

fn strip_parens(s: &str) -> &str {
    let mut s = s.trim();
    while let Some(inner) = s.strip_prefix('(').and_then(|i| i.strip_suffix(')')) {
        // Only if the parentheses actually match each other, so `(a) + (b)` is left
        // alone.
        let mut depth = 0i32;
        let balanced = inner.chars().all(|c| {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            depth >= 0
        }) && depth == 0;
        if !balanced {
            break;
        }
        s = inner.trim();
    }
    s
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
        None if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) => s.parse().ok(),
        None => None,
    }
}

/// Is this diagram label annotated as signed?
pub fn label_is_signed(label: &Label) -> bool {
    matches!(label, Label::Named { note: Some(n), .. } if n == "signed")
}
