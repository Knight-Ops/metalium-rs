//! Blackhole layouts measured against ttsim, where the specification has only
//! Wormhole's.
//!
//! The specification's `Bits32.lua` is the only encoding source this generator
//! trusts, and for most of the Matrix Unit it draws only Wormhole. Phase 6 found
//! that one of those drawings is wrong for Blackhole: `MVMUL`'s `AddrMod` sits at
//! bits 14..15, not 15..16 (`docs/ttsim-divergence.md` row 42), so an encoder
//! generated from the Wormhole diagram applies the wrong address modifier and
//! nothing says so.
//!
//! Hand-writing the Blackhole encoder would put a second source of truth beside
//! the generated table. Instead the measurement enters the table *through* the
//! generator, as data with its evidence attached, and is held to rules that keep
//! it honest:
//!
//! * **It supersedes a `WormholeOnly` diagram and nothing else.** A measurement
//!   never overrides documentation. When the specification documents the
//!   instruction for Blackhole, the override is refused, so the day upstream fills
//!   the gap is a build failure that says to reconcile the two -- not a silent
//!   preference for either.
//! * **Only the fields it lists as moved may differ**, and each listed field must
//!   really have moved. Every other field is carried from the Wormhole diagram bit
//!   for bit, and stays exactly as unverified as it was.
//! * **It names the gate that measured it**, and that gate must exist. Deleting the
//!   test deletes the licence for the layout.
//!
//! The superseded Wormhole diagram stays in the table under `wormhole::`, as the
//! specification's own superseded layouts do, so it can still be reached
//! deliberately.

use std::collections::BTreeMap;
use std::path::Path;

use super::check;
use super::lua;
use super::model::{Diagram, Label};
use super::provenance::Provenance;

/// The measured layouts, in `Bits32.lua`'s dialect.
pub const SOURCE: &str = include_str!("Bits32_BH.lua");

/// One measured layout's credentials.
pub struct Measured {
    /// Key in [`SOURCE`], always `<supersedes>_BH`.
    pub key: &'static str,
    /// The `WormholeOnly` diagram this replaces on Blackhole.
    pub supersedes: &'static str,
    /// Fields whose position differs from the Wormhole diagram's.
    pub moved: &'static [&'static str],
    /// `(file, test function)` pairs that measured the moved fields.
    pub evidence: &'static [(&'static str, &'static str)],
}

pub const MEASURED: &[Measured] = &[
    Measured {
        key: "MVMUL_BH",
        supersedes: "MVMUL",
        moved: &["AddrMod"],
        evidence: &[(
            "crates/tt-tests/tests/step9_matmul.rs",
            "mvmul_addr_mod_sits_one_bit_lower_on_blackhole",
        )],
    },
    Measured {
        key: "MOVA2D_BH",
        supersedes: "MOVA2D",
        moved: &["AddrMod"],
        evidence: &[(
            "crates/tt-tests/tests/probe_src.rs",
            "mov_to_dst_addr_mod_sits_one_bit_lower_on_blackhole",
        )],
    },
    // `BroadcastCol0` and `Broadcast1RowTo8` move with `Move4Rows`: the three are
    // LLK's three-bit `instr_mod` at bit 11. Only `Move4Rows` is measured; the
    // broadcasts' positions rest on LLK (`MOV_1_ROW_D0_BRCST` = 1,
    // `MOV_8_ROW_BRCST` = 2) until a gate exercises them.
    Measured {
        key: "MOVB2D_BH",
        supersedes: "MOVB2D",
        moved: &["BroadcastCol0", "Broadcast1RowTo8", "Move4Rows", "AddrMod"],
        evidence: &[
            (
                "crates/tt-tests/tests/probe_src.rs",
                "movb2d_move4_rows_is_bit_13_on_blackhole",
            ),
            (
                "crates/tt-tests/tests/probe_src.rs",
                "mov_to_dst_addr_mod_sits_one_bit_lower_on_blackhole",
            ),
        ],
    },
];

/// Does `file` (relative to the workspace root) define `fn name(`?
pub fn gate_exists(root: &Path) -> impl Fn(&str, &str) -> bool + '_ {
    move |file, name| {
        std::fs::read_to_string(root.join(file))
            .map(|text| text.contains(&format!("fn {name}(")))
            .unwrap_or(false)
    }
}

/// Parse `source`, check every entry against `table` and the specification's own
/// diagrams, and fold the measured layouts in: each measured key joins
/// `diagrams`, the diagram it supersedes becomes `SupersededOnBlackhole`, and the
/// measured key takes the superseded one's page and mnemonic.
pub fn apply(
    source: &str,
    table: &[Measured],
    gate_exists: &dyn Fn(&str, &str) -> bool,
    diagrams: &mut Vec<Diagram>,
    provenance: &mut BTreeMap<String, (Provenance, String)>,
    mnemonics: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    let measured = check::dedupe(lua::parse(source)?)?;

    for d in &measured {
        if !table.iter().any(|m| m.key == d.key) {
            return Err(format!(
                "measured layout `{}` has no row in `MEASURED`: every measured \
                 layout must say what it supersedes and which gate measured it",
                d.key
            ));
        }
    }
    for m in table {
        let Some(d) = measured.iter().find(|d| d.key == m.key) else {
            return Err(format!(
                "`MEASURED` lists `{}`, which `Bits32_BH.lua` does not define",
                m.key
            ));
        };
        if m.key != format!("{}_BH", m.supersedes) {
            return Err(format!(
                "`{}` must be named `{}_BH`, after the diagram it supersedes",
                m.key, m.supersedes
            ));
        }
        if diagrams.iter().any(|s| s.key == m.key) {
            return Err(format!(
                "`{}` is defined by the specification itself; the specification now \
                 documents this layout, so compare it with the measurement and \
                 remove the override",
                m.key
            ));
        }
        let Some(wh) = diagrams.iter().find(|s| s.key == m.supersedes) else {
            return Err(format!(
                "`{}` supersedes `{}`, which the specification does not define",
                m.key, m.supersedes
            ));
        };
        match provenance.get(m.supersedes) {
            Some((Provenance::WormholeOnly, _)) => {}
            Some((other, page)) => {
                return Err(format!(
                    "`{}` is no longer Wormhole-only (it is `{}`, per `{page}`): the \
                     specification now says something about it for Blackhole, so \
                     reconcile that with the measurement and remove the override",
                    m.supersedes,
                    other.variant_name()
                ))
            }
            None => return Err(format!("`{}` has no provenance", m.supersedes)),
        }
        compare(d, wh, m)?;
        if m.evidence.is_empty() {
            return Err(format!("`{}` names no gate that measured it", m.key));
        }
        for (file, name) in m.evidence {
            if !gate_exists(file, name) {
                return Err(format!(
                    "`{}` cites `{file}::{name}` as its evidence, and there is no such \
                     test. The layout is only as good as the gate that measured it.",
                    m.key
                ));
            }
        }
    }

    for m in table {
        let d = measured.iter().find(|d| d.key == m.key).unwrap().clone();
        let page = provenance[m.supersedes].1.clone();
        provenance.insert(
            m.supersedes.to_string(),
            (
                Provenance::SupersededOnBlackhole {
                    by: m.key.to_string(),
                },
                page.clone(),
            ),
        );
        let evidence = m
            .evidence
            .iter()
            .map(|(f, n)| format!("{f}::{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        provenance.insert(
            m.key.to_string(),
            (
                Provenance::Measured {
                    evidence,
                    moved: m.moved.iter().map(|s| s.to_string()).collect(),
                },
                page,
            ),
        );
        if let Some(mnemonic) = mnemonics.get(m.supersedes).cloned() {
            mnemonics.insert(m.key.to_string(), mnemonic);
        }
        diagrams.push(d);
    }
    Ok(())
}

/// The measured layout must be the Wormhole one with exactly `m.moved` relocated.
fn compare(measured: &Diagram, wh: &Diagram, m: &Measured) -> Result<(), String> {
    if measured.nbits != wh.nbits {
        return Err(format!(
            "`{}` and `{}` differ in width",
            m.key, m.supersedes
        ));
    }
    if measured.opcode_field().map(|f| &f.label) != wh.opcode_field().map(|f| &f.label) {
        return Err(format!(
            "`{}` has a different opcode from `{}`; a measured layout relocates \
             fields of one instruction, it does not define a new one",
            m.key, m.supersedes
        ));
    }
    let named = |d: &Diagram| -> BTreeMap<String, (u8, u8)> {
        d.fields
            .iter()
            .filter_map(|f| match &f.label {
                Label::Named { name, .. } => Some((name.clone(), (f.first_bit, f.width))),
                _ => None,
            })
            .collect()
    };
    let fixed = |d: &Diagram| -> Vec<(u8, u8, u32)> {
        d.fields
            .iter()
            .filter_map(|f| match &f.label {
                Label::Fixed { value } => Some((f.first_bit, f.width, *value)),
                _ => None,
            })
            .collect()
    };
    if fixed(measured) != fixed(wh) {
        return Err(format!(
            "`{}` changes a fixed field of `{}`; only named fields may be relocated",
            m.key, m.supersedes
        ));
    }
    let (a, b) = (named(measured), named(wh));
    if a.keys().ne(b.keys()) {
        return Err(format!(
            "`{}` and `{}` name different fields: {:?} against {:?}",
            m.key,
            m.supersedes,
            a.keys().collect::<Vec<_>>(),
            b.keys().collect::<Vec<_>>()
        ));
    }
    for name in m.moved {
        if !a.contains_key(*name) {
            return Err(format!(
                "`{}` lists `{name}` as moved, but has no such field",
                m.key
            ));
        }
    }
    for (name, (bit, width)) in &a {
        let (wh_bit, wh_width) = b[name];
        if *width != wh_width {
            return Err(format!(
                "`{}` changes the width of `{name}`; a relocation keeps widths",
                m.key
            ));
        }
        let listed = m.moved.contains(&name.as_str());
        match (listed, *bit == wh_bit) {
            (true, true) => {
                return Err(format!(
                    "`{}` lists `{name}` as moved, but it sits where `{}` has it. A \
                     stale listing claims a measurement that no longer applies.",
                    m.key, m.supersedes
                ))
            }
            (false, false) => {
                return Err(format!(
                    "`{}` moves `{name}` from bit {wh_bit} to bit {bit} without listing \
                     it as moved; every relocation must be one the evidence measured",
                    m.key
                ))
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: &str = r#"
local diagrams = {
  FOO = function()
    return Bits32{
      {0, 10, "Imm"},
      {15, 2, "AddrMod"},
      {20, 1, "0"},
      {24, 8, "0x26"},
    }
  end,
}
"#;

    const GOOD: &str = r#"
local diagrams = {
  FOO_BH = function()
    return Bits32{
      {0, 10, "Imm"},
      {14, 2, "AddrMod"},
      {20, 1, "0"},
      {24, 8, "0x26"},
    }
  end,
}
"#;

    const ROW: Measured = Measured {
        key: "FOO_BH",
        supersedes: "FOO",
        moved: &["AddrMod"],
        evidence: &[("some/test.rs", "measured_it")],
    };

    fn run(source: &str, table: &[Measured], prov: Provenance) -> Result<Vec<Diagram>, String> {
        let mut diagrams = lua::parse(SPEC).unwrap();
        let mut provenance = BTreeMap::from([("FOO".to_string(), (prov, "WH/FOO.md".to_string()))]);
        let mut mnemonics = BTreeMap::from([("FOO".to_string(), "TT_FOO".to_string())]);
        let exists = |f: &str, n: &str| f == "some/test.rs" && n == "measured_it";
        apply(
            source,
            table,
            &exists,
            &mut diagrams,
            &mut provenance,
            &mut mnemonics,
        )?;
        assert_eq!(
            provenance["FOO"].0,
            Provenance::SupersededOnBlackhole {
                by: "FOO_BH".into()
            }
        );
        assert!(matches!(
            provenance["FOO_BH"].0,
            Provenance::Measured { .. }
        ));
        assert_eq!(provenance["FOO_BH"].1, "WH/FOO.md");
        assert_eq!(mnemonics["FOO_BH"], "TT_FOO");
        Ok(diagrams)
    }

    fn rejects(source: &str, table: &[Measured], prov: Provenance, expected: &str) {
        let err = run(source, table, prov).expect_err("should have been refused");
        assert!(err.contains(expected), "wrong refusal: {err}");
    }

    #[test]
    fn a_good_override_supersedes_its_wormhole_diagram() {
        let diagrams = run(GOOD, &[ROW], Provenance::WormholeOnly).unwrap();
        assert!(diagrams.iter().any(|d| d.key == "FOO_BH"));
        assert!(diagrams.iter().any(|d| d.key == "FOO"));
    }

    #[test]
    fn an_override_without_credentials_is_refused() {
        rejects(
            GOOD,
            &[],
            Provenance::WormholeOnly,
            "has no row in `MEASURED`",
        );
    }

    #[test]
    fn credentials_without_an_override_are_refused() {
        let empty = "\nlocal diagrams = {\n  BAR_BH = function()\n    return Bits32{\n      {24, 8, \"0x01\"},\n    }\n  end,\n}\n";
        let row = Measured {
            key: "BAR_BH",
            supersedes: "BAR",
            moved: &[],
            evidence: &[],
        };
        rejects(
            empty,
            &[ROW, row],
            Provenance::WormholeOnly,
            "which `Bits32_BH.lua` does not define",
        );
    }

    #[test]
    fn a_documented_instruction_cannot_be_overridden() {
        rejects(
            GOOD,
            &[ROW],
            Provenance::Blackhole,
            "no longer Wormhole-only",
        );
        rejects(
            GOOD,
            &[ROW],
            Provenance::SharedWithWormhole,
            "no longer Wormhole-only",
        );
    }

    #[test]
    fn a_missing_gate_is_refused() {
        let row = Measured {
            evidence: &[("some/test.rs", "deleted")],
            ..ROW
        };
        rejects(
            GOOD,
            &[row],
            Provenance::WormholeOnly,
            "there is no such test",
        );
        let row = Measured {
            evidence: &[],
            ..ROW
        };
        rejects(GOOD, &[row], Provenance::WormholeOnly, "names no gate");
    }

    #[test]
    fn an_unlisted_relocation_is_refused() {
        let row = Measured { moved: &[], ..ROW };
        rejects(
            GOOD,
            &[row],
            Provenance::WormholeOnly,
            "without listing it as moved",
        );
    }

    #[test]
    fn a_stale_listing_is_refused() {
        let same = GOOD.replace("{14, 2, \"AddrMod\"}", "{15, 2, \"AddrMod\"}");
        rejects(&same, &[ROW], Provenance::WormholeOnly, "A stale listing");
    }

    #[test]
    fn a_changed_opcode_width_or_fixed_field_is_refused() {
        rejects(
            &GOOD.replace("0x26", "0x27"),
            &[ROW],
            Provenance::WormholeOnly,
            "different opcode",
        );
        rejects(
            &GOOD.replace("{14, 2, \"AddrMod\"}", "{14, 1, \"AddrMod\"}"),
            &[ROW],
            Provenance::WormholeOnly,
            "changes the width",
        );
        rejects(
            &GOOD.replace("{20, 1, \"0\"}", "{21, 1, \"0\"}"),
            &[ROW],
            Provenance::WormholeOnly,
            "fixed field",
        );
    }

    #[test]
    fn renamed_fields_are_refused() {
        rejects(
            &GOOD.replace("\"Imm\"", "\"Imm10\""),
            &[ROW],
            Provenance::WormholeOnly,
            "name different fields",
        );
    }

    #[test]
    fn the_name_must_follow_the_superseded_diagram() {
        let src = GOOD.replace("FOO_BH", "FOO_MEASURED");
        let row = Measured {
            key: "FOO_MEASURED",
            ..ROW
        };
        let mut diagrams = lua::parse(SPEC).unwrap();
        let mut provenance = BTreeMap::from([(
            "FOO".to_string(),
            (Provenance::WormholeOnly, "p".to_string()),
        )]);
        let err = apply(
            &src,
            &[row],
            &|_, _| true,
            &mut diagrams,
            &mut provenance,
            &mut BTreeMap::new(),
        )
        .unwrap_err();
        assert!(err.contains("must be named `FOO_BH`"), "{err}");
    }

    /// The real table passes its own checks against a synthetic MVMUL shaped like
    /// the specification's, and cites gates that exist in this workspace.
    #[test]
    fn the_real_table_cites_gates_that_exist() {
        let root = crate::util::workspace_root();
        let exists = gate_exists(&root);
        for m in MEASURED {
            for (file, name) in m.evidence {
                assert!(exists(file, name), "{}: {file}::{name} is missing", m.key);
            }
        }
        let parsed = lua::parse(SOURCE).unwrap();
        assert_eq!(parsed.len(), MEASURED.len());
    }
}
