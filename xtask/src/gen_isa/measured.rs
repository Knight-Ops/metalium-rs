//! Blackhole layouts measured on ttsim or silicon, where the specification has only
//! Wormhole's.
//!
//! The specification's `Bits32.lua` is the only encoding source this generator
//! trusts, and for most of the Matrix Unit it draws only Wormhole. Phase 6 found
//! that one of those drawings is wrong for Blackhole: `MVMUL`'s `AddrMod` sits at
//! bits 14..15, not 15..16 (`docs/learnings/ttsim-divergence.md` row 42), so an encoder
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
type FixedBits = (u8, u8, u32);

pub struct Measured {
    /// Key in [`SOURCE`], always `<supersedes>_BH`.
    pub key: &'static str,
    /// The `WormholeOnly` diagram this replaces on Blackhole.
    pub supersedes: &'static str,
    /// Fields whose position differs from the Wormhole diagram's.
    pub moved: &'static [&'static str],
    /// Exact (documented, measured) fixed-mode replacements, with gate evidence.
    pub replaced_fixed: &'static [(FixedBits, FixedBits)],
    /// Required fixed bits measured in positions the Wormhole diagram leaves undrawn.
    pub added_fixed: &'static [(u8, u8, u32)],
    /// Wormhole fields the Blackhole layout does not carry, each with the reason.
    /// A field whose Wormhole bits the evidence showed to hold something else on
    /// Blackhole, and whose own Blackhole position is unknown, is dropped rather
    /// than guessed: an encoder that cannot set it is honest; one that sets a
    /// guessed bit is not.
    pub dropped: &'static [(&'static str, &'static str)],
    /// Fields whose width differs from the Wormhole diagram's. Each must really
    /// have changed width, and the evidence must exercise the bits it gained.
    pub widened: &'static [&'static str],
    /// `(file, test function)` pairs that measured the moved fields.
    pub evidence: &'static [(&'static str, &'static str)],
}

pub const MEASURED: &[Measured] = &[
    Measured {
        key: "UNPACR_NOP_SETDVALID_BH",
        supersedes: "UNPACR_NOP_SETDVALID",
        moved: &[],
        // BH uses UNP_CLR_SRC, clear-to-one selection and repurposed format 3
        // to preserve data. The WH fixed mode 7 is a stream-pop on BH.
        replaced_fixed: &[((0, 3, 7), (0, 9, 0x1e9))],
        added_fixed: &[],
        dropped: &[],
        widened: &[],
        evidence: &[
            (
                "crates/tt-tests/tests/step104_unpacker_handover.rs",
                "llk_nonclearing_dvalid_a",
            ),
            (
                "crates/tt-tests/tests/step104_unpacker_handover.rs",
                "llk_nonclearing_dvalid_b",
            ),
        ],
    },
    Measured {
        key: "UNPACR_NOP_ZEROSRC_BH",
        supersedes: "UNPACR_NOP_ZEROSRC",
        moved: &["BothBanks", "WaitLikeUnpacr"],
        replaced_fixed: &[],
        added_fixed: &[],
        dropped: &[],
        // The inherited name is historical: BH uses a two-bit clear-value
        // code, not a negative-infinity flag. Checked APIs expose only zero.
        widened: &["NegativeInfSrcA"],
        evidence: &[
            (
                "crates/tt-tests/tests/step103_source_banks.rs",
                "unpacr_zero_blackhole_bank_and_clear_value_fields",
            ),
            (
                "crates/tt-tests/tests/step103_source_banks.rs",
                "unpacr_zero_waits_on_current_unpacker_bank_with_matrix_bank_held",
            ),
        ],
    },
    Measured {
        key: "GMPOOL_BH",
        supersedes: "GMPOOL",
        moved: &[],
        replaced_fixed: &[],
        added_fixed: &[(19, 1, 1)],
        dropped: &[],
        widened: &[],
        evidence: &[(
            "crates/tt-tests/tests/step75_fpu_pooling.rs",
            "matrix_pooling_uses_explicit_weights_and_releases_banks",
        )],
    },
    Measured {
        key: "GAPOOL_BH",
        supersedes: "GAPOOL",
        moved: &[],
        replaced_fixed: &[],
        added_fixed: &[(19, 1, 1)],
        dropped: &[],
        widened: &[],
        evidence: &[(
            "crates/tt-tests/tests/step75_fpu_pooling.rs",
            "matrix_pooling_uses_explicit_weights_and_releases_banks",
        )],
    },
    Measured {
        key: "MVMUL_BH",
        supersedes: "MVMUL",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
        evidence: &[(
            "crates/tt-tests/tests/step9_matmul.rs",
            "mvmul_addr_mod_sits_one_bit_lower_on_blackhole",
        )],
    },
    Measured {
        key: "MOVA2D_BH",
        supersedes: "MOVA2D",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
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
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["BroadcastCol0", "Broadcast1RowTo8", "Move4Rows", "AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
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
    Measured {
        key: "MOVD2A_BH",
        supersedes: "MOVD2A",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
        evidence: &[(
            "crates/tt-tests/tests/step9_matmul.rs",
            "mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole",
        )],
    },
    Measured {
        key: "MOVD2B_BH",
        supersedes: "MOVD2B",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
        evidence: &[(
            "crates/tt-tests/tests/step9_matmul.rs",
            "mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole",
        )],
    },
    Measured {
        key: "MOVB2A_BH",
        supersedes: "MOVB2A",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
        evidence: &[(
            "crates/tt-tests/tests/step9_matmul.rs",
            "mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole",
        )],
    },
    Measured {
        key: "ELWADD_BH",
        supersedes: "ELWADD",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
        evidence: &[
            (
                "crates/tt-tests/tests/step9_matmul.rs",
                "matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole",
            ),
            (
                "crates/tt-tests/tests/step9_matmul.rs",
                "elw_broadcast_assignment_and_destination_fields",
            ),
            (
                "crates/tt-tests/tests/step90_matrix_eltwise.rs",
                "resident_matrix_arithmetic_and_rhs_broadcasts",
            ),
        ],
    },
    Measured {
        key: "ELWSUB_BH",
        supersedes: "ELWSUB",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
        evidence: &[
            (
                "crates/tt-tests/tests/step9_matmul.rs",
                "matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole",
            ),
            (
                "crates/tt-tests/tests/step9_matmul.rs",
                "elw_broadcast_assignment_and_destination_fields",
            ),
            (
                "crates/tt-tests/tests/step90_matrix_eltwise.rs",
                "resident_matrix_arithmetic_and_rhs_broadcasts",
            ),
        ],
    },
    Measured {
        key: "ELWMUL_BH",
        supersedes: "ELWMUL",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
        evidence: &[
            (
                "crates/tt-tests/tests/step9_matmul.rs",
                "matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole",
            ),
            (
                "crates/tt-tests/tests/step9_matmul.rs",
                "elw_broadcast_assignment_and_destination_fields",
            ),
            (
                "crates/tt-tests/tests/step90_matrix_eltwise.rs",
                "resident_matrix_arithmetic_and_rhs_broadcasts",
            ),
        ],
    },
    Measured {
        key: "DOTPV_BH",
        supersedes: "DOTPV",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
        evidence: &[(
            "crates/tt-tests/tests/step9_matmul.rs",
            "matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole",
        )],
    },
    Measured {
        key: "MOVDBGA2D_BH",
        supersedes: "MOVDBGA2D",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
        evidence: &[(
            "crates/tt-tests/tests/step9_matmul.rs",
            "matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole",
        )],
    },
    Measured {
        key: "SHIFTXB_BH",
        supersedes: "SHIFTXB",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &["AddrMod"],
        evidence: &[(
            "crates/tt-tests/tests/step9_matmul.rs",
            "shiftxb_addr_mod_sits_one_bit_lower_on_blackhole",
        )],
    },
    // LLK's Blackhole `ZEROACC` is `clear_mode << 19`, `use_32_bit_mode << 18`,
    // `clear_zero_flags << 17`, `addr_mode << 14`. `clear_zero_flags` is not drawn:
    // nothing has measured it, so the encoder cannot set it.
    Measured {
        key: "ZEROACC_BH",
        supersedes: "ZEROACC",
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod", "UseDst32b"],
        dropped: &[(
            "Revert",
            "its Wormhole bit 18 is `UseDst32b` on Blackhole (measured, and LLK's \
             `use_32_bit_mode << 18`); LLK has no such field, and Wormhole's \
             `ZEROACC.md` makes any use of it `UndefinedBehavior`",
        )],
        widened: &["AddrMod"],
        evidence: &[(
            "crates/tt-tests/tests/step9_matmul.rs",
            "zeroacc_addr_mod_and_use_dst32b_on_blackhole",
        )],
    },
];

/// Does `file` (relative to the workspace root) define `fn name(`?
/// A `WormholeOnly` layout a gate ran on Blackhole and found unchanged.
pub struct Confirmed {
    /// The diagram key, as the specification draws it.
    pub key: &'static str,
    /// Every field of the diagram, each set to more than one value by the
    /// evidence: a field the gate never moved is not confirmed, so the list
    /// must name them all.
    pub exercised: &'static [&'static str],
    /// `(file, test function)` pairs: ttsim and silicon, both.
    pub evidence: &'static [(&'static str, &'static str)],
}

pub const CONFIRMED: &[Confirmed] = &[
    Confirmed {
        // Immediate multiplication and the other new scalar families refuse
        // ttsim; they retain WormholeOnly provenance despite silicon coverage.
        key: "MULDMAREG",
        exercised: &["ResultReg", "RightReg", "LeftReg"],
        evidence: &[(
            "crates/tt-tests/tests/step100_scalar_config.rs",
            "scalar_register_and_immediate_semantics",
        )],
    },
    Confirmed {
        // The register form; the six-bit immediate form (`ADDDMAREGi`) runs
        // on silicon (`adddmareg_adds_an_immediate`) but not on ttsim
        // (divergence row 67), so it is not confirmed here.
        key: "ADDDMAREG",
        exercised: &["ResultReg", "RightReg", "LeftReg"],
        evidence: &[(
            "crates/tt-tests/tests/step38_gpr_add.rs",
            "adddmareg_adds_two_registers",
        )],
    },
    Confirmed {
        key: "MOP",
        exercised: &["Template", "Count1", "MaskLo"],
        evidence: &[(
            "crates/tt-tests/tests/step36_mop.rs",
            "a_mop_expands_as_the_page_models_it",
        )],
    },
    Confirmed {
        key: "MOP_CFG",
        exercised: &["MaskHi"],
        evidence: &[(
            "crates/tt-tests/tests/step36_mop.rs",
            "a_mop_expands_as_the_page_models_it",
        )],
    },
];

/// Mark each [`CONFIRMED`] layout so. Refused unless it is `WormholeOnly` --
/// a confirmation never overrides documentation or a measurement, so the day
/// upstream documents it is a build failure that says to drop the row -- and
/// unless its gate exists and it names every field of the diagram.
pub fn confirm(
    table: &[Confirmed],
    gate_exists: &dyn Fn(&str, &str) -> bool,
    diagrams: &[Diagram],
    provenance: &mut BTreeMap<String, (Provenance, String)>,
) -> Result<(), String> {
    for c in table {
        let Some((p, page)) = provenance.get(c.key).cloned() else {
            return Err(format!("confirmed layout `{}` is not in the table", c.key));
        };
        if p != Provenance::WormholeOnly {
            return Err(format!(
                "`{}` is {}, not WormholeOnly: a confirmation only stands in for missing \
                 documentation; drop its row from `CONFIRMED`",
                c.key,
                p.variant_name()
            ));
        }
        if c.evidence.is_empty() {
            return Err(format!("confirmed layout `{}` names no gate", c.key));
        }
        for (file, name) in c.evidence {
            if !gate_exists(file, name) {
                return Err(format!(
                    "confirmed layout `{}` cites `{file}::{name}`, which does not exist",
                    c.key
                ));
            }
        }
        let d = diagrams
            .iter()
            .find(|d| d.key == c.key)
            .ok_or_else(|| format!("confirmed layout `{}` has no diagram", c.key))?;
        let mut fields: Vec<&str> = d
            .fields
            .iter()
            .filter_map(|f| match &f.label {
                Label::Named { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        let mut named: Vec<&str> = c.exercised.to_vec();
        fields.sort_unstable();
        named.sort_unstable();
        if fields != named {
            return Err(format!(
                "confirmed layout `{}` exercises {named:?}, but its fields are {fields:?}: \
                 every field must be exercised",
                c.key
            ));
        }
        let evidence = c
            .evidence
            .iter()
            .map(|(f, n)| format!("{f}::{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        provenance.insert(
            c.key.to_string(),
            (Provenance::Confirmed { evidence }, page),
        );
    }
    Ok(())
}

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
                    dropped: m.dropped.iter().map(|(s, _)| s.to_string()).collect(),
                    widened: m.widened.iter().map(|s| s.to_string()).collect(),
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
    let mut expected_fixed = fixed(wh);
    // Fixed modes can also differ between architectures. Each replacement
    // must name an exact documented field and may occupy only its old bits or
    // undrawn bits; carried named fields and opcode bits remain protected.
    for &(before, after) in m.replaced_fixed {
        let Some(index) = expected_fixed.iter().position(|f| *f == before) else {
            return Err(format!("`{}` replaces an absent fixed field", m.key));
        };
        let (first, width, value) = after;
        if width == 0 || width >= 32 || first as u32 + width as u32 > 32 || value >= (1u32 << width)
        {
            return Err(format!(
                "`{}` has an invalid replacement fixed field",
                m.key
            ));
        }
        let old_mask = ((1u32 << before.1) - 1) << before.0;
        let new_mask = ((1u32 << width) - 1) << first;
        if before.0 >= 24 || new_mask & !(wh.undrawn() | old_mask) != 0 {
            return Err(format!(
                "`{}` replaces a fixed field over protected bits",
                m.key
            ));
        }
        expected_fixed[index] = after;
    }
    for &(first, width, value) in m.added_fixed {
        if width == 0 || width >= 32 || first as u32 + width as u32 > 32 || value >= (1u32 << width)
        {
            return Err(format!("`{}` has an invalid added fixed field", m.key));
        }
        let mask = ((1u32 << width) - 1) << first;
        if wh.undrawn() & mask != mask {
            return Err(format!(
                "`{}` adds a fixed field over documented bits",
                m.key
            ));
        }
        expected_fixed.push((first, width, value));
    }
    expected_fixed.sort_unstable();
    let mut actual_fixed = fixed(measured);
    actual_fixed.sort_unstable();
    if actual_fixed != expected_fixed {
        return Err(format!(
            "`{}` changes a fixed field of `{}` without an exact measurement credential",
            m.key, m.supersedes
        ));
    }
    let (a, mut b) = (named(measured), named(wh));
    for (name, why) in m.dropped {
        if why.trim().is_empty() {
            return Err(format!("`{}` drops `{name}` without saying why", m.key));
        }
        if a.contains_key(*name) {
            return Err(format!(
                "`{}` lists `{name}` as dropped, but still draws it",
                m.key
            ));
        }
        if b.remove(*name).is_none() {
            return Err(format!(
                "`{}` drops `{name}`, which `{}` does not have",
                m.key, m.supersedes
            ));
        }
    }
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
    for name in m.widened {
        if !a.contains_key(*name) {
            return Err(format!(
                "`{}` lists `{name}` as widened, but has no such field",
                m.key
            ));
        }
    }
    for (name, (bit, width)) in &a {
        let (wh_bit, wh_width) = b[name];
        match (m.widened.contains(&name.as_str()), *width == wh_width) {
            (false, false) => {
                return Err(format!(
                    "`{}` changes the width of `{name}` without listing it as widened",
                    m.key
                ))
            }
            (true, true) => {
                return Err(format!(
                    "`{}` lists `{name}` as widened, but it is as wide as in `{}`",
                    m.key, m.supersedes
                ))
            }
            _ => {}
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
        replaced_fixed: &[],
        added_fixed: &[],
        moved: &["AddrMod"],
        dropped: &[],
        widened: &[],
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
    fn required_fixed_bits_need_credentials_and_undrawn_positions() {
        let source = GOOD.replace("{20, 1, \"0\"}", "{19, 1, \"1\"},\n      {20, 1, \"0\"}");
        rejects(
            &source,
            &[ROW],
            Provenance::WormholeOnly,
            "changes a fixed field",
        );
        let row = Measured {
            added_fixed: &[(19, 1, 1)],
            ..ROW
        };
        run(&source, &[row], Provenance::WormholeOnly).unwrap();
        let row = Measured {
            added_fixed: &[(20, 1, 1)],
            ..ROW
        };
        rejects(
            &source,
            &[row],
            Provenance::WormholeOnly,
            "over documented bits",
        );
        let row = Measured {
            added_fixed: &[(19, 1, 2)],
            ..ROW
        };
        rejects(
            &source,
            &[row],
            Provenance::WormholeOnly,
            "invalid added fixed field",
        );
    }

    #[test]
    fn fixed_mode_replacements_require_exact_credentials_and_protected_bits() {
        let source = GOOD.replace("{20, 1, \"0\"}", "{20, 2, \"3\"}");
        rejects(
            &source,
            &[ROW],
            Provenance::WormholeOnly,
            "changes a fixed field",
        );
        let row = Measured {
            replaced_fixed: &[((20, 1, 0), (20, 2, 3))],
            ..ROW
        };
        run(&source, &[row], Provenance::WormholeOnly).unwrap();
        let absent = Measured {
            replaced_fixed: &[((20, 1, 1), (20, 2, 3))],
            ..ROW
        };
        rejects(
            &source,
            &[absent],
            Provenance::WormholeOnly,
            "absent fixed field",
        );
        let protected = Measured {
            replaced_fixed: &[((20, 1, 0), (8, 2, 3))],
            ..ROW
        };
        rejects(
            &source,
            &[protected],
            Provenance::WormholeOnly,
            "protected bits",
        );
        let too_wide = Measured {
            replaced_fixed: &[((20, 1, 0), (20, 1, 3))],
            ..ROW
        };
        rejects(
            &source,
            &[too_wide],
            Provenance::WormholeOnly,
            "invalid replacement fixed field",
        );
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
            replaced_fixed: &[],
            added_fixed: &[],
            moved: &[],
            dropped: &[],
            widened: &[],
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

    /// `FOO` with a `Revert` bit at 18, which the Blackhole layout below drops.
    const SPEC_WITH_REVERT: &str = r#"
local diagrams = {
  FOO = function()
    return Bits32{
      {0, 10, "Imm"},
      {15, 2, "AddrMod"},
      {18, 1, "Revert"},
      {20, 1, "0"},
      {24, 8, "0x26"},
    }
  end,
}
"#;

    fn run_with_revert(source: &str, table: &[Measured]) -> Result<(), String> {
        let mut diagrams = lua::parse(SPEC_WITH_REVERT).unwrap();
        let mut provenance = BTreeMap::from([(
            "FOO".to_string(),
            (Provenance::WormholeOnly, "WH/FOO.md".to_string()),
        )]);
        let mut mnemonics = BTreeMap::new();
        let exists = |f: &str, n: &str| f == "some/test.rs" && n == "measured_it";
        apply(
            source,
            table,
            &exists,
            &mut diagrams,
            &mut provenance,
            &mut mnemonics,
        )
    }

    #[test]
    fn a_dropped_field_must_be_listed_with_a_reason() {
        let dropped =
            |dropped: &'static [(&'static str, &'static str)]| Measured { dropped, ..ROW };
        // GOOD omits `Revert`: accepted only when the row says so, and why.
        run_with_revert(
            GOOD,
            &[dropped(&[("Revert", "its bit is another field's")])],
        )
        .unwrap();
        let err = run_with_revert(GOOD, &[ROW]).unwrap_err();
        assert!(err.contains("name different fields"), "{err}");
        let err = run_with_revert(GOOD, &[dropped(&[("Revert", " ")])]).unwrap_err();
        assert!(err.contains("without saying why"), "{err}");
        let err = run_with_revert(GOOD, &[dropped(&[("Nope", "x")])]).unwrap_err();
        assert!(err.contains("does not have"), "{err}");
    }

    #[test]
    fn a_field_still_drawn_cannot_be_dropped() {
        let row = Measured {
            dropped: &[("Imm", "x")],
            ..ROW
        };
        rejects(GOOD, &[row], Provenance::WormholeOnly, "still draws it");
    }

    #[test]
    fn a_width_change_must_be_listed_and_real() {
        const WIDE: &str = r#"
local diagrams = {
  FOO_BH = function()
    return Bits32{
      {0, 10, "Imm"},
      {14, 3, "AddrMod"},
      {20, 1, "0"},
      {24, 8, "0x26"},
    }
  end,
}
"#;
        let widened = Measured {
            widened: &["AddrMod"],
            ..ROW
        };
        run(WIDE, &[widened], Provenance::WormholeOnly).unwrap();
        rejects(
            WIDE,
            &[ROW],
            Provenance::WormholeOnly,
            "without listing it as widened",
        );
        let stale = Measured {
            widened: &["AddrMod"],
            ..ROW
        };
        rejects(GOOD, &[stale], Provenance::WormholeOnly, "as wide as");
        let missing = Measured {
            widened: &["Nope"],
            ..ROW
        };
        rejects(WIDE, &[missing], Provenance::WormholeOnly, "no such field");
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

    const CONFIRM: Confirmed = Confirmed {
        key: "FOO",
        exercised: &["Imm", "AddrMod"],
        evidence: &[("some/test.rs", "measured_it")],
    };

    fn confirming(row: Confirmed, prov: Provenance) -> Result<Provenance, String> {
        let diagrams = lua::parse(SPEC).unwrap();
        let mut provenance = BTreeMap::from([("FOO".to_string(), (prov, "WH/FOO.md".to_string()))]);
        let exists = |f: &str, n: &str| f == "some/test.rs" && n == "measured_it";
        confirm(&[row], &exists, &diagrams, &mut provenance)?;
        Ok(provenance["FOO"].0.clone())
    }

    #[test]
    fn a_confirmation_marks_a_wormhole_only_layout_and_nothing_else() {
        assert_eq!(
            confirming(CONFIRM, Provenance::WormholeOnly),
            Ok(Provenance::Confirmed {
                evidence: "some/test.rs::measured_it".into()
            })
        );
        for documented in [Provenance::Blackhole, Provenance::SharedWithWormhole] {
            let err = confirming(CONFIRM, documented).unwrap_err();
            assert!(err.contains("not WormholeOnly"), "{err}");
        }
    }

    #[test]
    fn a_confirmation_needs_its_gate_and_every_field() {
        let err = confirming(
            Confirmed {
                evidence: &[("some/test.rs", "deleted")],
                ..CONFIRM
            },
            Provenance::WormholeOnly,
        )
        .unwrap_err();
        assert!(err.contains("does not exist"), "{err}");
        let err = confirming(
            Confirmed {
                exercised: &["Imm"],
                ..CONFIRM
            },
            Provenance::WormholeOnly,
        )
        .unwrap_err();
        assert!(err.contains("every field must be exercised"), "{err}");
        let err = confirming(
            Confirmed {
                evidence: &[],
                ..CONFIRM
            },
            Provenance::WormholeOnly,
        )
        .unwrap_err();
        assert!(err.contains("names no gate"), "{err}");
    }
}
