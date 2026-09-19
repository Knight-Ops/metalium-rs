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

// ---------------------------------------------------------------------------
// The cross-check: the diagram against the macro template on its page.
// ---------------------------------------------------------------------------

use super::syntax::{self, Term};

/// A page in the shape of the real ones.
///
/// Modelled on `SETDMAREG_Special.md`, because its three-term argument is the case
/// the slot-base check exists for: the macro's shifts of 7, 3 and 0 and the
/// diagram's bits 15, 11 and 8 all have to imply the same slot base of 8.
const PAGE: &str = r#"# `SETDMAREG` (Special)

**Backend execution unit:** Scalar Unit (ThCon)

## Syntax

```c
TT_SETDMAREG(/* u2 */ ResultSize,
           ((/* u4 */ WhichPackers) << 7) +
           ((/* u4 */ InputSource ) << 3) +
             /* u3 */ InputHalfReg,
             1,
             /* u7 */ ResultHalfReg)
```

## Encoding

![](../../../Diagrams/Out/Bits32_SETDMAREG_Special.svg)

## Functional model

Prose that mentions `TT_SETDMAREG(x, y)` and must not be mistaken for the syntax.
"#;

const PAGE_DIAGRAM: &str = r#"local diagrams = {
  SETDMAREG_Special = function()
    return Bits32{
      {0, 7, "ResultHalfReg"},
      {7, 1, "1"},
      {8, 3, "InputHalfReg", y = 1},
      {11, 4, "InputSource", y = 2},
      {15, 4, "WhichPackers", y = 1},
      {22, 2, "ResultSize", y = 1},
      {24, 8, "0x45"},
    }
  end,
}
"#;

fn cross(diagram_src: &str, page_src: &str) -> Result<super::check::CrossCheck, String> {
    let diagrams = lua::parse(diagram_src)?;
    // Not `check::structure`: that holds over the whole instruction set (there is
    // exactly one computed opcode, and it is RMWCIB's), which a two-diagram fixture
    // cannot satisfy. The cross-check does not depend on it.
    let diagrams = check::dedupe(diagrams)?;
    let page =
        syntax::parse_page("Fixture.md", page_src)?.expect("the fixture documents an encoding");
    check::cross_check(&diagrams, std::slice::from_ref(&page))
}

#[track_caller]
fn cross_rejects(diagram_src: &str, page_src: &str, expected: &str) {
    match cross(diagram_src, page_src) {
        Ok(_) => panic!("expected rejection mentioning {expected:?}, but it agreed"),
        Err(msg) => assert!(
            msg.contains(expected),
            "expected a message mentioning {expected:?}, got: {msg}"
        ),
    }
}

#[test]
fn the_macro_parses_into_slots_and_shifts() {
    let page = syntax::parse_page("Fixture.md", PAGE).unwrap().unwrap();
    assert_eq!(page.keys, ["SETDMAREG_Special"]);
    assert_eq!(
        page.calls.len(),
        1,
        "prose mentioning TT_SETDMAREG is not a syntax block"
    );

    let call = &page.calls[0];
    assert_eq!(call.name, "SETDMAREG");
    assert_eq!(call.slots.len(), 4, "four arguments, high bits first");

    // The packed argument: three terms, shifted within one slot.
    assert_eq!(
        call.slots[1],
        vec![
            Term::Named {
                name: "WhichPackers".into(),
                width: 4,
                signed: false,
                shift: 7
            },
            Term::Named {
                name: "InputSource".into(),
                width: 4,
                signed: false,
                shift: 3
            },
            Term::Named {
                name: "InputHalfReg".into(),
                width: 3,
                signed: false,
                shift: 0
            },
        ]
    );
    // A literal argument names bits the diagram draws as a fixed value.
    assert_eq!(call.slots[2], vec![Term::Literal { value: 1, shift: 0 }]);
}

#[test]
fn the_two_sources_agree_about_the_fixture() {
    let report = cross(PAGE_DIAGRAM, PAGE).expect("the fixture should agree");
    assert_eq!(report.pairs, 1);
    assert!(report.unembedded.is_empty());
    assert!(report.unchecked.is_empty());
}

#[test]
fn a_moved_field_is_caught_by_the_slot_base_even_though_its_width_is_right() {
    // This is the check the whole cross-check exists for. `InputSource` keeps its
    // name and its width; only its position moves, by one bit. Nothing in the
    // diagram alone notices, and nothing in the macro alone notices. Together they
    // do, because the slot no longer has a single base.
    cross_rejects(
        &PAGE_DIAGRAM.replace(r#"{11, 4, "InputSource""#, r#"{10, 4, "InputSource""#),
        PAGE,
        "imply different slot bases",
    );
    // The same mutation applied to the other source, for symmetry.
    cross_rejects(
        PAGE_DIAGRAM,
        &PAGE.replace("InputSource ) << 3", "InputSource ) << 2"),
        "imply different slot bases",
    );
}

#[test]
fn a_width_disagreement_is_caught_from_either_side() {
    cross_rejects(
        &PAGE_DIAGRAM.replace(r#"{22, 2, "ResultSize""#, r#"{22, 1, "ResultSize""#),
        PAGE,
        "bits in the diagram and",
    );
    cross_rejects(
        PAGE_DIAGRAM,
        &PAGE.replace("/* u2 */ ResultSize", "/* u3 */ ResultSize"),
        "bits in the diagram and",
    );
}

#[test]
fn a_renamed_field_is_caught() {
    cross_rejects(
        PAGE_DIAGRAM,
        &PAGE.replace("/* u7 */ ResultHalfReg", "/* u7 */ ResultReg"),
        "the macro calls it",
    );
}

#[test]
fn a_disagreement_about_signedness_is_caught() {
    cross_rejects(
        PAGE_DIAGRAM,
        &PAGE.replace("/* u7 */ ResultHalfReg", "/* i7 */ ResultHalfReg"),
        "whether `ResultHalfReg` is signed",
    );
}

#[test]
fn a_field_the_macro_does_not_take_is_caught_unless_it_is_a_known_pinning() {
    cross_rejects(
        PAGE_DIAGRAM,
        &PAGE.replace("/* u2 */ ResultSize,\n", "0,\n"),
        "the macro takes",
    );
}

#[test]
fn a_diagram_no_page_embeds_is_reported() {
    let extra = PAGE_DIAGRAM.replace(
        "\n}\n",
        "\n  SFPNOP = function()\n    return Bits32{\n      {24, 8, \"0x8F\"},\n    }\n  end,\n}\n",
    );
    assert_ne!(extra, PAGE_DIAGRAM, "the mutation must actually apply");
    let report = cross(&extra, PAGE).expect("the pairs that exist still agree");
    assert_eq!(report.unembedded, ["SFPNOP"]);
}

#[test]
fn a_page_embedding_an_unknown_diagram_is_refused() {
    cross_rejects(
        PAGE_DIAGRAM,
        &PAGE.replace(
            "Bits32_SETDMAREG_Special.svg",
            "Bits32_SETDMAREG_Imaginary.svg",
        ),
        "which Bits32.lua does not define",
    );
}

#[test]
fn prose_outside_the_syntax_block_is_not_mistaken_for_one() {
    // `SETC16.md` really does discuss `TT_SETC16(CFG_STATE_ID_StateID_ADDR32, x)` in
    // its notes. A whole-file search would take that for an encoding.
    let page = syntax::parse_page("Fixture.md", PAGE).unwrap().unwrap();
    assert_eq!(page.calls.len(), 1);
}

#[test]
fn an_operandless_instruction_has_a_macro_with_no_arguments() {
    // `TTI_SFPNOP` takes no arguments at all. It still counts as documented, which
    // is what keeps the "has an opcode iff it has a macro" invariant true.
    let src = PAGE
        .replace("Bits32_SETDMAREG_Special.svg", "Bits32_SFPNOP.svg")
        .replace(
            "TT_SETDMAREG(/* u2 */ ResultSize,\n           ((/* u4 */ WhichPackers) << 7) +\n           ((/* u4 */ InputSource ) << 3) +\n             /* u3 */ InputHalfReg,\n             1,\n             /* u7 */ ResultHalfReg)",
            "TTI_SFPNOP",
        );
    let page = syntax::parse_page("Fixture.md", &src).unwrap().unwrap();
    assert_eq!(page.calls.len(), 1);
    assert_eq!(page.calls[0].name, "SFPNOP");
    assert!(page.calls[0].slots.is_empty());
}

/// `ZEROACC`, whose macro really does omit the `Revert` field the diagram draws.
const ZEROACC_DIAGRAM: &str = r#"local diagrams = {
  ZEROACC = function()
    return Bits32{
      {0, 10, "Imm10"},
      {12, 2, "AddrMod"},
      {14, 1, "Revert", y = 1},
      {15, 2, "Mode"},
      {19, 1, "UseDst32b", y = 1},
      {24, 8, "0x48"},
    }
  end,
}
"#;

const ZEROACC_PAGE: &str = r#"# `ZEROACC`

## Syntax

```c
TT_ZEROACC(/* bool */ UseDst32b, /* u2 */ Mode, /* u2 */ AddrMod, /* u10 */ Imm10)
```

## Encoding

![](../../../Diagrams/Out/Bits32_ZEROACC.svg)
"#;

#[test]
fn a_pinned_field_lets_the_macro_take_fewer_operands_than_the_diagram_draws() {
    // The exception is what makes this agree; without it the arity check fires,
    // which `a_field_the_macro_does_not_take_is_caught_unless_it_is_a_known_pinning`
    // covers from the other side.
    let report = cross(ZEROACC_DIAGRAM, ZEROACC_PAGE)
        .expect("ZEROACC's macro omits `Revert`, and the exception table says so");
    assert_eq!(report.pairs, 1);

    // ...and the exception is recorded as used, which is what stops it going stale.
    assert!(
        !check::stale_exceptions(&report)
            .iter()
            .any(|s| s.contains("ZEROACC.Revert")),
        "the ZEROACC exception should count as used once it applies"
    );
}

#[test]
fn an_exception_that_never_applies_is_reported_as_stale() {
    // Over a single page almost every exception is unused, which is exactly why
    // this is a whole-corpus rule rather than part of the per-page comparison.
    let report = cross(ZEROACC_DIAGRAM, ZEROACC_PAGE).unwrap();
    let stale = check::stale_exceptions(&report);
    assert!(
        stale.iter().any(|s| s.contains("ATSWAP.SingleDataReg")),
        "an exception no page needed should be named: {stale:?}"
    );
}
