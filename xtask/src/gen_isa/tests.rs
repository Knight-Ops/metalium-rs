//! Mutation tests for the specification parsers.
//!
//! The checklist's standing rule is that a gate nobody has watched reject something
//! is not yet evidence — this repository has shipped two vacuous gates already. A
//! perturbation performed once by hand and described in a commit message satisfies
//! that rule today and stops satisfying it the moment the code changes underneath.
//!
//! So each rejection the parser is supposed to make has a test that makes it. They
//! run against small embedded fixtures rather than the vendored tree, so they need
//! no network and no `fetch-spec`, and they run under the `cargo test --workspace`
//! CI already has.

use super::model::Label;
use super::{check, lua};

/// A fixture in the shape of the real file, small enough to mutate in one line.
///
/// Deliberately exercises every syntactic feature the real file uses: a bare
/// diagram, `nbits`, `extra_w`, the cosmetic `y` and `edge` keys, a fixed opcode, a
/// must-be-zero bit, an annotated label, and the computed `RMWCIB` opcode.
const FIXTURE: &str = r#"#!/usr/bin/env luajit
local function Bits32(fields) end

local diagrams = {
  SFPSTORE_BH = function()
    return Bits32{
      {0, 10, "Imm10"},
      {13, 3, "AddrMod", y = 1},
      {16, 4, "Mod0"},
      {20, 4, "VD"},
      {24, 8, "0x72"},
    }
  end,
  SFPNOP = function()
    return Bits32{
      {7, 1, "0"},
      {24, 8, "0x8F"},
    }
  end,
  SFPIADD = function()
    return Bits32{extra_w = 70,
      {0, 4, "Mod1"},
      {12, 12, "Imm12 (signed)", y = 2, edge = "right"},
      {24, 8, "0x79"},
    }
  end,
  RMWCIB = function()
    return Bits32{
      {0, 8, "Index4"},
      {24, 8, "0xB3 + Index1"},
    }
  end,
  Dst16_FP16 = function()
    return Bits32{nbits = 16,
      {0, 5, "Exponent"},
      {5, 10, "Mantissa"},
      {15, 1, "Sign", y = 1, edge = "left"},
    }
  end,
}

do_diagram(...)
"#;

/// Parse the fixture, or fail with the parser's message.
fn parse(source: &str) -> Vec<super::model::Diagram> {
    let diagrams = lua::parse(source).expect("fixture should parse");
    let diagrams = check::dedupe(diagrams).expect("fixture should have no divergent duplicates");
    check::structure(&diagrams).expect("fixture should satisfy the structural invariants");
    diagrams
}

/// Assert that a mutated fixture is rejected, and that the message says why.
#[track_caller]
fn rejects(source: &str, expected: &str) {
    let result = lua::parse(source)
        .and_then(check::dedupe)
        .and_then(|d| check::structure(&d).map(|()| d));
    match result {
        Ok(_) => panic!("expected rejection mentioning {expected:?}, but it parsed"),
        Err(msg) => assert!(
            msg.contains(expected),
            "expected a message mentioning {expected:?}, got: {msg}"
        ),
    }
}

#[test]
fn the_fixture_parses_and_is_what_it_claims_to_be() {
    let diagrams = parse(FIXTURE);
    assert_eq!(diagrams.len(), 5);

    // Order is preserved, so a diff of the generated table follows the source.
    let keys: Vec<&str> = diagrams.iter().map(|d| d.key.as_str()).collect();
    assert_eq!(
        keys,
        ["SFPSTORE_BH", "SFPNOP", "SFPIADD", "RMWCIB", "Dst16_FP16"]
    );

    let store = &diagrams[0];
    assert_eq!(store.nbits, 32);
    assert_eq!(
        store.opcode_field().map(|f| &f.label),
        Some(&Label::Fixed { value: 0x72 })
    );

    // The Blackhole AddrMod position: three bits at 13, not Wormhole's two at 14.
    // The single most consequential encoding difference in this instruction set.
    let addr_mod = store
        .fields
        .iter()
        .find(|f| matches!(&f.label, Label::Named { name, .. } if name == "AddrMod"))
        .expect("SFPSTORE_BH has an AddrMod field");
    assert_eq!((addr_mod.first_bit, addr_mod.width), (13, 3));

    // A datum layout: narrower than a word, and no opcode.
    let dst16 = diagrams.last().unwrap();
    assert_eq!(dst16.nbits, 16);
    assert!(dst16.opcode_field().is_none());
}

#[test]
fn cosmetic_keys_are_discarded_but_nothing_else_is() {
    // `y` and `edge` place text in the rendered SVG and carry no encoding meaning.
    let with = parse(FIXTURE);
    let without = parse(
        &FIXTURE
            .replace(", y = 1", "")
            .replace(", edge = \"right\"", ""),
    );
    let strip = |ds: Vec<super::model::Diagram>| {
        ds.into_iter()
            .map(|d| {
                d.fields
                    .into_iter()
                    .map(|f| (f.first_bit, f.width, f.label))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(strip(with), strip(without));
}

#[test]
fn an_unknown_table_key_is_refused_rather_than_ignored() {
    // The cosmetic keys are skipped by name. A key nobody has classified might be
    // meaningful, so it stops the build instead of joining them.
    rejects(
        &FIXTURE.replace("Bits32{nbits = 16,", "Bits32{nbits = 16, lanes = 4,"),
        "unknown Bits32 table key `lanes`",
    );
}

#[test]
fn an_unknown_field_key_is_refused_rather_than_ignored() {
    rejects(
        &FIXTURE.replace(r#""AddrMod", y = 1"#, r#""AddrMod", y = 1, role = "sub""#),
        "unknown field key `role`",
    );
}

#[test]
fn a_stray_line_inside_the_table_is_refused() {
    // The cfg_defines.h parser `continue`s past lines it does not recognise. That
    // is the one thing not copied here: skipping is how a parser survives an
    // upstream format change by silently dropping instructions.
    rejects(
        &FIXTURE.replace(
            "  SFPNOP = function()",
            "  -- a new comment\n  SFPNOP = function()",
        ),
        "expected `KEY = function()`",
    );
}

#[test]
fn an_unrecognised_label_is_refused() {
    rejects(
        &FIXTURE.replace(r#""Mod0""#, r#""Mod0 or Mod1""#),
        "unrecognised label",
    );
}

#[test]
fn the_exception_table_is_load_bearing_in_both_directions() {
    // Removing an entry must break the label it explains...
    assert!(lua::unused(FIXTURE).contains(&"16 or 0"));

    // ...and a label only the table explains must parse only because of it.
    let with_special = FIXTURE.replace(r#"{0, 5, "Exponent"}"#, r#"{0, 5, "16 or 0"}"#);
    let parsed = parse(&with_special);
    let exponent = &parsed.last().unwrap().fields[0];
    assert_eq!(
        exponent.label,
        Label::Named {
            name: "Exponent".into(),
            note: Some("16 or 0".into()),
        },
        "the exception should name the field and keep the original text as a note"
    );
}

#[test]
fn overlapping_fields_are_refused() {
    rejects(
        &FIXTURE.replace(r#"{16, 4, "Mod0"}"#, r#"{15, 4, "Mod0"}"#),
        "overlap",
    );
}

#[test]
fn a_field_past_the_end_of_the_word_is_refused() {
    rejects(
        &FIXTURE.replace(r#"{15, 1, "Sign""#, r#"{15, 4, "Sign""#),
        "runs past the 16-bit word",
    );
}

#[test]
fn a_fixed_value_too_large_for_its_field_is_refused() {
    rejects(
        &FIXTURE.replace(r#"{7, 1, "0"}"#, r#"{7, 1, "3"}"#),
        "does not fit",
    );
}

#[test]
fn an_unknown_word_width_is_refused() {
    rejects(&FIXTURE.replace("nbits = 16", "nbits = 24"), "nbits = 24");
}

#[test]
fn identical_duplicates_are_collapsed_but_divergent_ones_are_not() {
    // `SEMINIT` really is defined twice in Bits32.lua, byte-identically, and Lua
    // silently keeps the second. Matching that would hide the day they diverge.
    let duplicate = "  SFPNOP = function()\n    return Bits32{\n      {7, 1, \"0\"},\n      {24, 8, \"0x8F\"},\n    }\n  end,\n";
    let twice = FIXTURE.replace(duplicate, &format!("{duplicate}{duplicate}"));
    assert_eq!(
        parse(&twice).len(),
        5,
        "an identical duplicate is collapsed"
    );

    let divergent = duplicate.replace("{7, 1, \"0\"}", "{6, 1, \"0\"}");
    rejects(
        &FIXTURE.replace(duplicate, &format!("{duplicate}{divergent}")),
        "defined twice",
    );
}

#[test]
fn the_computed_opcode_stays_unique() {
    // `RMWCIB`'s `0xB3 + Index1` is the only opcode that carries an operand. A
    // second would need its own decision about how to split opcode from field.
    rejects(
        &FIXTURE.replace(r#"{24, 8, "0x79"}"#, r#"{24, 8, "0x79 + Index1"}"#),
        "expected exactly one computed opcode",
    );
    rejects(
        &FIXTURE.replace(r#""0xB3 + Index1""#, r#""0xB3""#),
        "computed opcode has disappeared",
    );
}

#[test]
fn a_truncated_file_is_refused() {
    let open = FIXTURE.find("  SFPNOP").unwrap();
    rejects(&FIXTURE[..open], "never closed");
    rejects("-- no table here\n", "not found");
    rejects("local diagrams = {\n}\n", "empty");
}

#[test]
fn undrawn_bits_are_reported_rather_than_assumed_zero() {
    // Only a small minority of diagrams cover their whole word, so the gaps are
    // the norm and have to be recorded rather than silently zero-filled.
    let diagrams = parse(FIXTURE);
    let nop = &diagrams[1];
    assert_eq!(nop.undrawn(), !(1 << 7) & 0x00FF_FFFF);
    let dst16 = diagrams.last().unwrap();
    assert_eq!(dst16.undrawn(), 0, "this layout does cover its whole word");
}
