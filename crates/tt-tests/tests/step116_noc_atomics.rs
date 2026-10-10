//! NoC atomics beyond the increment (`BlackholeA0/NoC/Atomics.md`).
//!
//! Three layers of evidence:
//!
//! * **Model and layout (host, always).** [`model`] is a literal transcription of
//!   the page's pseudocode. It does not use `tt_isa::noc::atomic`: it *decodes*
//!   the `NOC_AT_LEN_BE` word the encoder produced, with field positions parsed
//!   from the vendored diagram source (`Diagrams/Src/Bits32.lua`,
//!   `NOC_AT_LEN_BE_*`) rather than copied from the encoder, so an encoder that
//!   puts a field in the wrong place is decoded as a different operation and
//!   fails. The scenario expectations are hand-computed literals.
//! * **ttsim (not silicon).** ttsim executes only the full-width increment (every
//!   other `NOC_AT_LEN_BE` opcode, and a partial-width increment, stops the
//!   process with `UnimplementedFunctionality`); `simulator_models_only_the_full_
//!   width_increment` runs the increment through the probe image against the
//!   model and records each refusal (divergence row 88, proposed). The other
//!   forms therefore have silicon-only gates.
//! * **Silicon (`--features silicon`, written, NOT run).** Probe order: the
//!   full-width increment to a neighbouring tile (precedent: the mover barrier),
//!   then one test per form, each its own process. Documented, UNVERIFIED on
//!   silicon, NoC-hang class: an L1-only atomic to a surviving neighbour.
//!
//! Negative control: [`Mutant`] changes the model (one per command) and
//! `mutants_are_caught` requires the expectation table to reject each.

mod noc_support;

use tt_isa::noc::atomic::{AccFormat, AtomicOp, AtomicRequest, SwapForm, Zaamo};

// ---------------------------------------------------------------------------
// Layouts, parsed from the vendored diagram source.
// ---------------------------------------------------------------------------

/// `(low bit, width, name)` of each field of `NOC_AT_LEN_BE_<which>`.
fn layout(which: &str) -> Vec<(u32, u32, String)> {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vendor/tt-isa-documentation/Diagrams/Src/Bits32.lua"
    ))
    .expect("vendor/tt-isa-documentation is fetched");
    let head = format!("NOC_AT_LEN_BE_{which} = function()");
    let at = src.find(&head).unwrap_or_else(|| panic!("no {head}"));
    let body = &src[at..];
    let end = body.find("\n  end,").unwrap();
    let mut out = vec![];
    for line in body[..end].lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix('{') {
            let rest = rest.trim_end_matches(',').trim_end_matches('}');
            let parts: Vec<&str> = rest.split(',').map(str::trim).collect();
            if parts.len() == 3 && parts[2].starts_with('"') {
                out.push((
                    parts[0].parse().unwrap(),
                    parts[1].parse().unwrap(),
                    parts[2].trim_matches('"').to_string(),
                ));
            }
        }
    }
    assert!(!out.is_empty(), "{which} has no fields");
    out
}

fn constant(name: &str) -> Option<u32> {
    match name.strip_prefix("0x") {
        Some(h) => u32::from_str_radix(h, 16).ok(),
        None => name.parse().ok(),
    }
}

/// Pack `fields` (by name) into a `NOC_AT_LEN_BE` word of layout `which`; the
/// layout's literal fields (the opcode and fixed bits) are filled in.
fn pack(which: &str, fields: &[(&str, u32)]) -> u32 {
    let mut word = 0u32;
    for (lo, width, name) in layout(which) {
        let mask = if width == 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        let v = match constant(&name) {
            Some(c) => c,
            None => {
                fields
                    .iter()
                    .find(|(n, _)| *n == name)
                    .unwrap_or_else(|| panic!("{which}: no value for {name}"))
                    .1
            }
        };
        assert!(
            v <= mask,
            "{which}.{name} = {v:#x} does not fit {width} bits"
        );
        word |= v << lo;
    }
    word
}

/// The value of `name` in `word` under layout `which`.
fn field(which: &str, word: u32, name: &str) -> u32 {
    let (lo, width, _) = layout(which)
        .into_iter()
        .find(|(_, _, n)| n == name)
        .unwrap_or_else(|| panic!("{which} has no {name}"));
    (word >> lo) & ((1u32 << width) - 1)
}

/// Does `word` carry every literal field of layout `which`?
fn matches(which: &str, word: u32) -> bool {
    layout(which)
        .into_iter()
        .all(|(lo, width, name)| match constant(&name) {
            Some(c) => (word >> lo) & ((1u32 << width) - 1) == c,
            None => true,
        })
}

// ---------------------------------------------------------------------------
// The model: the page, transcribed.
// ---------------------------------------------------------------------------

/// The deliberately wrong models the control runs, one per command.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Mutant {
    None,
    /// Increment: the mask is `(1 << IntWidth) - 1`, one bit short.
    IncrementMaskOffByOne,
    /// CAS: only the low four bits of the word are compared.
    CasComparesLowNibble,
    /// Swap, mask: the halves of `NOC_AT_DATA` are used the wrong way round.
    SwapMaskHalvesSwapped,
    /// Swap, index: `Ofs` is ignored.
    SwapIndexIgnoresOfs,
    /// Zaamo: `amomin.w` compares unsigned.
    ZaamoMinUnsigned,
    /// Parallel add, `u8`: code 7 wraps instead of saturating.
    AccU8Wraps,
}

/// What an atomic does to one 16-byte unit of L1: the new bytes and the word the
/// response carries (`None`: the page says `UndefinedValue()`).
#[derive(Debug, PartialEq)]
struct Outcome {
    unit: [u8; 16],
    result: Option<u32>,
}

fn word(unit: &[u8; 16], i: usize) -> u32 {
    u32::from_le_bytes(unit[i * 4..i * 4 + 4].try_into().unwrap())
}

fn set_word(unit: &mut [u8; 16], i: usize, v: u32) {
    unit[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

fn flush32(x: f32) -> f32 {
    if x != 0.0 && x.abs() < f32::MIN_POSITIVE {
        0.0
    } else {
        x
    }
}

/// A half format's two conversions.
type Cvt = (fn(u16) -> f32, fn(f32) -> u16);

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

fn f32_to_bf16(x: f32) -> u16 {
    // Round to nearest even.
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    ((b.wrapping_add(0x7FFF + lsb)) >> 16) as u16
}

fn fp16_to_f32(h: u16) -> f32 {
    let (s, e, m) = (
        u32::from(h >> 15),
        u32::from((h >> 10) & 0x1F),
        u32::from(h & 0x3FF),
    );
    let v = match e {
        0 => (m as f32) * 2f32.powi(-24),
        31 => f32::NAN,
        _ => (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15),
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

fn f32_to_fp16(x: f32) -> u16 {
    // Exhaustive search over the 65536 patterns for the nearest (ties to even
    // mantissa): slow, obviously right, and independent of any bit trick.
    let mut best = (f32::INFINITY, 0u16);
    for h in 0..=u16::MAX {
        let v = fp16_to_f32(h);
        if !v.is_finite() {
            continue;
        }
        let d = (v - x).abs();
        if d < best.0 || (d == best.0 && h & 1 == 0 && best.1 & 1 == 1) {
            best = (d, h);
        }
    }
    best.1
}

/// The model. `targ` is `NOC_TARG_ADDR_LO`, `unit` the 16-byte-aligned unit of L1
/// holding it. `None` when the request is outside what the page defines (a
/// layout that matches no form, a non-finite floating-point operand or result).
fn model(unit: &[u8; 16], targ: u32, len_be: u32, data: u32, mutant: Mutant) -> Option<Outcome> {
    let mut u = *unit;
    let in_unit = (targ & 0xF) as usize;
    let result = word(unit, in_unit / 4);
    let opcode = (len_be >> 12) & 0xF;
    match opcode {
        1 if matches("Increment", len_be) => {
            let ofs = field("Increment", len_be, "Ofs") as usize;
            let w = field("Increment", len_be, "IntWidth");
            let orig = word(&u, ofs);
            let inc = orig.wrapping_add(data);
            let mask = match mutant {
                Mutant::IncrementMaskOffByOne => ((1u64 << w) - 1) as u32,
                _ => ((2u64 << w) - 1) as u32,
            };
            set_word(&mut u, ofs, (inc & mask) | (orig & !mask));
            Some(Outcome {
                unit: u,
                result: Some(result),
            })
        }
        4 if matches("CAS", len_be) => {
            let ofs = field("CAS", len_be, "Ofs") as usize;
            let (cmp, set) = (
                field("CAS", len_be, "CmpVal"),
                field("CAS", len_be, "SetVal"),
            );
            let orig = word(&u, ofs);
            let equal = match mutant {
                Mutant::CasComparesLowNibble => orig & 0xF == cmp,
                _ => orig == cmp,
            };
            if equal {
                set_word(&mut u, ofs, set);
            }
            Some(Outcome {
                unit: u,
                result: Some(result),
            })
        }
        3 if matches("SwapMask", len_be) => {
            let mask = field("SwapMask", len_be, "Mask");
            let to_write = [data & 0xFFFF, data >> 16];
            for i in 0..8 {
                if mask & (1 << i) != 0 {
                    let h = match mutant {
                        Mutant::SwapMaskHalvesSwapped => to_write[(i & 1) ^ 1],
                        _ => to_write[i & 1],
                    };
                    u[i * 2..i * 2 + 2].copy_from_slice(&(h as u16).to_le_bytes());
                }
            }
            Some(Outcome {
                unit: u,
                result: Some(result),
            })
        }
        6 | 7 | 10 => {
            let which = match (
                opcode,
                matches("Zaamo", len_be),
                matches("SwapIndex10", len_be),
            ) {
                (6, _, _) if matches("SwapIndex6", len_be) => "SwapIndex6",
                (7, _, _) if matches("SwapIndex7", len_be) => "SwapIndex7",
                (10, true, _) => "Zaamo",
                (10, false, true) => "SwapIndex10",
                _ => return None,
            };
            let ofs = field(which, len_be, "Ofs") as usize;
            if which == "Zaamo" {
                let orig = word(&u, ofs);
                let op = field("Zaamo", len_be, "Op");
                let (a, b) = (orig, data);
                let operated = match op {
                    0 => a.wrapping_add(b),
                    1 => a ^ b,
                    2 => a | b,
                    3 => a & b,
                    4 => {
                        let lt = if mutant == Mutant::ZaamoMinUnsigned {
                            a < b
                        } else {
                            (a as i32) < (b as i32)
                        };
                        if lt {
                            a
                        } else {
                            b
                        }
                    }
                    5 => {
                        if (a as i32) > (b as i32) {
                            a
                        } else {
                            b
                        }
                    }
                    6 => {
                        if a < b {
                            a
                        } else {
                            b
                        }
                    }
                    _ => {
                        if a > b {
                            a
                        } else {
                            b
                        }
                    }
                };
                set_word(&mut u, ofs, operated);
            } else {
                let at = if mutant == Mutant::SwapIndexIgnoresOfs {
                    0
                } else {
                    ofs
                };
                set_word(&mut u, at, data);
            }
            Some(Outcome {
                unit: u,
                result: Some(result),
            })
        }
        9 if matches("Acc", len_be) => {
            let fmt = field("Acc", len_be, "Fmt");
            let mut new = u;
            match fmt {
                4 | 12 | 13 => {
                    for i in 0..4 {
                        set_word(&mut new, i, word(&u, i).wrapping_add(data));
                    }
                }
                7 | 15 => {
                    let d = data.to_le_bytes();
                    for i in 0..16 {
                        let (a, b) = (u[i], d[i & 3]);
                        new[i] = if fmt == 7 && mutant != Mutant::AccU8Wraps {
                            a.saturating_add(b)
                        } else {
                            a.wrapping_add(b)
                        };
                    }
                }
                0 | 8 => {
                    for i in 0..4 {
                        let (a, b) = (
                            flush32(f32::from_bits(word(&u, i))),
                            flush32(f32::from_bits(data)),
                        );
                        let r = flush32(a + b);
                        if !a.is_finite() || !b.is_finite() || !r.is_finite() {
                            return None;
                        }
                        set_word(&mut new, i, r.to_bits());
                    }
                }
                1 | 9 | 2 | 10 => {
                    let (to, from): Cvt = if fmt & 3 == 1 {
                        (fp16_to_f32, f32_to_fp16)
                    } else {
                        (bf16_to_f32, f32_to_bf16)
                    };
                    let d = [data as u16, (data >> 16) as u16];
                    for i in 0..8 {
                        let a = flush32(to(u16::from_le_bytes([u[i * 2], u[i * 2 + 1]])));
                        let b = flush32(to(d[i & 1]));
                        let r = flush32(a + b);
                        if !a.is_finite() || !b.is_finite() || !r.is_finite() {
                            return None;
                        }
                        let h = from(r);
                        if to(h).is_infinite() {
                            return None;
                        }
                        new[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
                    }
                }
                _ => return None,
            }
            Some(Outcome {
                unit: new,
                result: None,
            })
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Scenarios: hand-computed expectations.
// ---------------------------------------------------------------------------

fn unit_of(words: [u32; 4]) -> [u8; 16] {
    let mut u = [0u8; 16];
    for (i, w) in words.iter().enumerate() {
        set_word(&mut u, i, *w);
    }
    u
}

fn bytes_0_to_15() -> [u8; 16] {
    core::array::from_fn(|i| i as u8)
}

struct Scenario {
    name: &'static str,
    unit: [u8; 16],
    /// The word of the unit the request targets (`NOC_TARG_ADDR_LO & 0xC`).
    target_word: u32,
    op: AtomicOp,
    expect_unit: [u8; 16],
    expect_result: Option<u32>,
}

fn scenarios() -> Vec<Scenario> {
    let zaamo = |name, op, orig: u32, data: u32, new: u32| Scenario {
        name,
        unit: unit_of([7, orig, 9, 11]),
        target_word: 1,
        op: AtomicOp::Zaamo { op, data },
        expect_unit: unit_of([7, new, 9, 11]),
        expect_result: Some(orig),
    };
    let swap = |name, form| Scenario {
        name,
        unit: unit_of([0x1000_0001, 0x2000_0002, 0x3000_0003, 0x4000_0004]),
        target_word: 2,
        op: AtomicOp::SwapIndex {
            form,
            data: 0xDEAD_BEEF,
        },
        expect_unit: unit_of([0x1000_0001, 0x2000_0002, 0xDEAD_BEEF, 0x4000_0004]),
        expect_result: Some(0x3000_0003),
    };
    let v = vec![
        Scenario {
            name: "increment, full width, wraps",
            unit: unit_of([1, 0xFFFF_FFF0, 3, 4]),
            target_word: 1,
            op: AtomicOp::Increment {
                value: 0x20,
                int_width: 31,
            },
            expect_unit: unit_of([1, 0x10, 3, 4]),
            expect_result: Some(0xFFFF_FFF0),
        },
        Scenario {
            name: "increment, 8 bits: carry stays in the low byte",
            unit: unit_of([1, 2, 0x1234_56F0, 4]),
            target_word: 2,
            op: AtomicOp::Increment {
                value: 0x20,
                int_width: 7,
            },
            expect_unit: unit_of([1, 2, 0x1234_5610, 4]),
            expect_result: Some(0x1234_56F0),
        },
        Scenario {
            name: "cas, equal: swaps",
            unit: unit_of([1, 5, 3, 4]),
            target_word: 1,
            op: AtomicOp::CompareSwap { cmp: 5, set: 9 },
            expect_unit: unit_of([1, 9, 3, 4]),
            expect_result: Some(5),
        },
        Scenario {
            name: "cas, different: leaves it (high bits matter)",
            unit: unit_of([1, 0x15, 3, 4]),
            target_word: 1,
            op: AtomicOp::CompareSwap { cmp: 5, set: 9 },
            expect_unit: unit_of([1, 0x15, 3, 4]),
            expect_result: Some(0x15),
        },
        Scenario {
            name: "swap, mask 0b0000_0110: halfwords 1 and 2",
            unit: bytes_0_to_15(),
            target_word: 0,
            op: AtomicOp::SwapMask {
                mask: 0b0000_0110,
                data: 0xAABB_CCDD,
            },
            // Halfword 1 (bytes 2, 3) takes the HIGH half 0xAABB, halfword 2
            // (bytes 4, 5) the LOW half 0xCCDD.
            expect_unit: [
                0, 1, 0xBB, 0xAA, 0xDD, 0xCC, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            ],
            expect_result: Some(0x0302_0100),
        },
        swap("swap, index form 6", SwapForm::Six),
        swap("swap, index form 7", SwapForm::Seven),
        swap("swap, index form 10", SwapForm::Ten),
        zaamo("zaamo add wraps", Zaamo::Add, 0xFFFF_FFFE, 3, 1),
        zaamo("zaamo xor", Zaamo::Xor, 0xFFFF_FFFE, 3, 0xFFFF_FFFD),
        zaamo("zaamo or", Zaamo::Or, 0xFFFF_FFFE, 3, 0xFFFF_FFFF),
        zaamo("zaamo and", Zaamo::And, 0xFFFF_FFFE, 3, 2),
        zaamo(
            "zaamo min is signed",
            Zaamo::Min,
            0xFFFF_FFFE,
            3,
            0xFFFF_FFFE,
        ),
        zaamo("zaamo max is signed", Zaamo::Max, 0xFFFF_FFFE, 3, 3),
        zaamo("zaamo minu is unsigned", Zaamo::MinU, 0xFFFF_FFFE, 3, 3),
        zaamo(
            "zaamo maxu is unsigned",
            Zaamo::MaxU,
            0xFFFF_FFFE,
            3,
            0xFFFF_FFFE,
        ),
        Scenario {
            name: "parallel add, u32 wraps",
            unit: unit_of([1, 2, 3, 0xFFFF_FFFF]),
            target_word: 0,
            op: AtomicOp::Accumulate {
                fmt: AccFormat::U32,
                data: 5,
            },
            expect_unit: unit_of([6, 7, 8, 4]),
            expect_result: None,
        },
        Scenario {
            name: "parallel add, u8 saturates (code 7)",
            unit: [
                0xFE, 1, 0xFF, 0x80, 0xFE, 1, 0xFF, 0x80, 0xFE, 1, 0xFF, 0x80, 0xFE, 1, 0xFF, 0x80,
            ],
            target_word: 0,
            op: AtomicOp::Accumulate {
                fmt: AccFormat::U8Saturating,
                data: 0x7F05_0505,
            },
            // Lane i adds byte i & 3 of the data: 5, 5, 5, 0x7F.
            expect_unit: [
                0xFF, 6, 0xFF, 0xFF, 0xFF, 6, 0xFF, 0xFF, 0xFF, 6, 0xFF, 0xFF, 0xFF, 6, 0xFF, 0xFF,
            ],
            expect_result: None,
        },
        Scenario {
            name: "parallel add, u8 wraps (code 15)",
            unit: [
                0xFE, 1, 0xFF, 0x80, 0xFE, 1, 0xFF, 0x80, 0xFE, 1, 0xFF, 0x80, 0xFE, 1, 0xFF, 0x80,
            ],
            target_word: 0,
            op: AtomicOp::Accumulate {
                fmt: AccFormat::U8Wrapping,
                data: 0x7F05_0505,
            },
            expect_unit: [3, 6, 4, 0xFF, 3, 6, 4, 0xFF, 3, 6, 4, 0xFF, 3, 6, 4, 0xFF],
            expect_result: None,
        },
        Scenario {
            name: "parallel add, fp32, denormal input flushed",
            unit: unit_of([
                1.5f32.to_bits(),
                (-3.0f32).to_bits(),
                0x0000_0001,
                0x0080_0000,
            ]),
            target_word: 0,
            op: AtomicOp::Accumulate {
                fmt: AccFormat::Fp32,
                data: 0.25f32.to_bits(),
            },
            // 0x0000_0001 is a denormal input: flushed, so 0 + 0.25. The last
            // word is the smallest normal plus 0.25, which rounds to 0.25.
            expect_unit: unit_of([
                1.75f32.to_bits(),
                (-2.75f32).to_bits(),
                0.25f32.to_bits(),
                0.25f32.to_bits(),
            ]),
            expect_result: None,
        },
        Scenario {
            name: "parallel add, bf16, lanes alternate halves of the data",
            unit: unit_of([0x3F80_3F80, 0x3F80_3F80, 0x3F80_3F80, 0x3F80_3F80]),
            target_word: 0,
            // Even lanes add 0x3F00 (0.5), odd lanes 0x4000 (2.0).
            op: AtomicOp::Accumulate {
                fmt: AccFormat::Bf16,
                data: 0x4000_3F00,
            },
            expect_unit: unit_of([0x4040_3FC0, 0x4040_3FC0, 0x4040_3FC0, 0x4040_3FC0]),
            expect_result: None,
        },
        Scenario {
            name: "parallel add, fp16",
            unit: unit_of([0x3C00_3C00, 0x3C00_3C00, 0x3C00_3C00, 0x3C00_3C00]),
            target_word: 0,
            // 1.0 + 0.5 = 1.5 (0x3E00) on even lanes, 1.0 + 2.0 = 3.0 (0x4200).
            op: AtomicOp::Accumulate {
                fmt: AccFormat::Fp16,
                data: 0x4000_3800,
            },
            expect_unit: unit_of([0x4200_3E00, 0x4200_3E00, 0x4200_3E00, 0x4200_3E00]),
            expect_result: None,
        },
    ];
    v
}

const TARGET_BASE: u32 = 0x2_2000;

fn request_of(s: &Scenario) -> AtomicRequest {
    AtomicRequest {
        to: tt_isa::noc::niu::Endpoint {
            x: 4,
            y: 4,
            addr: TARGET_BASE + 4 * s.target_word,
        },
        ret_local: TARGET_BASE + 0x800,
        op: s.op,
    }
}

fn run_model(s: &Scenario, mutant: Mutant) -> Option<Outcome> {
    let r = request_of(s);
    let (len_be, data) = r.len_be_and_data().unwrap();
    model(&s.unit, r.to.addr, len_be, data, mutant)
}

fn check(s: &Scenario, got: &Option<Outcome>) -> Result<(), String> {
    let got = got
        .as_ref()
        .ok_or_else(|| format!("{}: model refused", s.name))?;
    if got.unit != s.expect_unit {
        return Err(format!(
            "{}: unit {:02x?}, expected {:02x?}",
            s.name, got.unit, s.expect_unit
        ));
    }
    if got.result != s.expect_result {
        return Err(format!(
            "{}: result {:x?}, expected {:x?}",
            s.name, got.result, s.expect_result
        ));
    }
    Ok(())
}

#[test]
fn model_reproduces_the_hand_computed_expectations() {
    for s in scenarios() {
        check(&s, &run_model(&s, Mutant::None)).unwrap();
    }
}

#[test]
fn mutants_are_caught() {
    for m in [
        Mutant::IncrementMaskOffByOne,
        Mutant::CasComparesLowNibble,
        Mutant::SwapMaskHalvesSwapped,
        Mutant::SwapIndexIgnoresOfs,
        Mutant::ZaamoMinUnsigned,
        Mutant::AccU8Wraps,
    ] {
        let caught = scenarios()
            .iter()
            .any(|s| check(s, &run_model(s, m)).is_err());
        assert!(caught, "{m:?} survived every expectation");
    }
}

// ---------------------------------------------------------------------------
// Encoder against the vendored layouts.
// ---------------------------------------------------------------------------

#[test]
fn layouts_match_the_vendored_diagrams() {
    let enc = |op, addr: u32| {
        AtomicRequest {
            to: tt_isa::noc::niu::Endpoint { x: 1, y: 2, addr },
            ret_local: 0x100,
            op,
        }
        .len_be_and_data()
        .unwrap()
        .0
    };
    for word_index in 0..4u32 {
        let a = 0x1_0000 + 4 * word_index;
        let ofs = word_index;
        for w in [0u32, 7, 15, 31] {
            assert_eq!(
                enc(
                    AtomicOp::Increment {
                        value: 1,
                        int_width: w as u8
                    },
                    a
                ),
                pack("Increment", &[("Ofs", ofs), ("IntWidth", w)])
            );
        }
        for (c, s) in [(0u32, 15u32), (5, 9), (15, 0)] {
            assert_eq!(
                enc(
                    AtomicOp::CompareSwap {
                        cmp: c as u8,
                        set: s as u8
                    },
                    a
                ),
                pack("CAS", &[("Ofs", ofs), ("CmpVal", c), ("SetVal", s)])
            );
        }
        for (form, which) in [
            (SwapForm::Six, "SwapIndex6"),
            (SwapForm::Seven, "SwapIndex7"),
            (SwapForm::Ten, "SwapIndex10"),
        ] {
            assert_eq!(
                enc(AtomicOp::SwapIndex { form, data: 0 }, a),
                pack(which, &[("Ofs", ofs)])
            );
        }
        for (op, n) in [
            (Zaamo::Add, 0),
            (Zaamo::Xor, 1),
            (Zaamo::Or, 2),
            (Zaamo::And, 3),
            (Zaamo::Min, 4),
            (Zaamo::Max, 5),
            (Zaamo::MinU, 6),
            (Zaamo::MaxU, 7),
        ] {
            assert_eq!(
                enc(AtomicOp::Zaamo { op, data: 0 }, a),
                pack("Zaamo", &[("Ofs", ofs), ("Op", n)])
            );
        }
    }
    for mask in [0u32, 1, 0xA5, 0xFF] {
        assert_eq!(
            enc(
                AtomicOp::SwapMask {
                    mask: mask as u8,
                    data: 0
                },
                0x1_0000
            ),
            pack("SwapMask", &[("Mask", mask)])
        );
    }
    for code in 0..=15u8 {
        if let Some(fmt) = AccFormat::from_code(code) {
            assert_eq!(
                enc(AtomicOp::Accumulate { fmt, data: 0 }, 0x1_0010),
                pack("Acc", &[("Fmt", u32::from(code))])
            );
        }
    }
    // Every distinct form decodes as itself and as no other layout.
    let forms = [
        (
            "Increment",
            pack("Increment", &[("Ofs", 1), ("IntWidth", 31)]),
        ),
        (
            "CAS",
            pack("CAS", &[("Ofs", 1), ("CmpVal", 3), ("SetVal", 4)]),
        ),
        ("SwapMask", pack("SwapMask", &[("Mask", 0x5A)])),
        ("SwapIndex6", pack("SwapIndex6", &[("Ofs", 2)])),
        ("SwapIndex7", pack("SwapIndex7", &[("Ofs", 2)])),
        ("SwapIndex10", pack("SwapIndex10", &[("Ofs", 2)])),
        ("Zaamo", pack("Zaamo", &[("Ofs", 2), ("Op", 5)])),
        ("Acc", pack("Acc", &[("Fmt", 2)])),
    ];
    for (name, word) in forms {
        let hits: Vec<&str> = forms
            .iter()
            .map(|(n, _)| *n)
            .filter(|n| matches(n, word))
            .collect();
        assert_eq!(hits, [name], "{word:#x} is ambiguous");
    }
}

// ---------------------------------------------------------------------------
// Device runs (ttsim and silicon).
// ---------------------------------------------------------------------------

use noc_support::{read_bytes, run_probe, unicast, write_bytes};
use tt_isa::noc::niu::TxnId;
use tt_isa::noc::probe;
use tt_tests::harness::{self, Dev};

/// Run `s` on a device: the target tile's unit is seeded, the probe on the gate
/// tile issues the atomic, and the unit and the response word are read back.
fn run_on_device(dev: &mut Dev<'_>, s: &Scenario) -> Result<(Outcome, noc_support::Res), String> {
    let me = harness::tensix_tile();
    let target = harness::tile(dev, 4, 4);
    write_bytes(dev, target, u64::from(TARGET_BASE), &s.unit);
    // Poison the response word so an absent response is not a zero.
    write_bytes(
        dev,
        me,
        u64::from(TARGET_BASE) + 0x800,
        &0xC0FF_EE00u32.to_le_bytes(),
    );
    let regs = request_of(s)
        .registers((me.x(), me.y()), TxnId::new(1).unwrap())
        .unwrap();
    let res = run_probe(dev, me, &[unicast(&regs, 1, 0)], &[])?;
    let unit: [u8; 16] = read_bytes(dev, target, u64::from(TARGET_BASE), 16)
        .try_into()
        .unwrap();
    let ret = u32::from_le_bytes(
        read_bytes(dev, me, u64::from(TARGET_BASE) + 0x800, 4)
            .try_into()
            .unwrap(),
    );
    Ok((
        Outcome {
            unit,
            result: Some(ret),
        },
        res[0],
    ))
}

/// The device agrees with the model; the response word is compared only where
/// the page defines it.
fn device_matches(dev: &mut Dev<'_>, s: &Scenario, mutant: Mutant) -> Result<(), String> {
    let (got, res) = run_on_device(dev, s)?;
    if res.status != probe::status::OK {
        return Err(format!("{}: probe status {} ({res:?})", s.name, res.status));
    }
    let want = run_model(s, mutant).ok_or_else(|| format!("{}: model refused", s.name))?;
    if got.unit != want.unit {
        return Err(format!(
            "{}: unit {:02x?}, model {:02x?}",
            s.name, got.unit, want.unit
        ));
    }
    if want.result.is_some() && got.result != want.result {
        return Err(format!(
            "{}: response {:x?}, model {:x?}",
            s.name, got.result, want.result
        ));
    }
    Ok(())
}

/// What ttsim does (divergence row 88, proposed): it executes the full-width
/// increment, with the response and the counters, and stops the process on every
/// other `NOC_AT_LEN_BE` opcode and on a partial-width increment.
#[cfg(not(feature = "silicon"))]
#[test]
fn simulator_models_only_the_full_width_increment() {
    let all = scenarios();
    let by = |n: &str| all.iter().find(|s| s.name.starts_with(n)).unwrap();
    // The control: the full-width increment runs and matches the model, and the
    // mutated model does not.
    let inc = by("increment, full width");
    assert!(harness::survives(|dev| device_matches(
        dev,
        inc,
        Mutant::None
    )
    .unwrap()));
    assert!(
        !harness::survives(|dev| device_matches(dev, inc, Mutant::IncrementMaskOffByOne).unwrap()),
        "the negative control must diverge from the device"
    );
    // Every other form is refused, each in its own fork.
    for s in &all {
        if s.name.starts_with("increment, full width") {
            continue;
        }
        let ran = harness::survives(|dev| {
            let _ = run_on_device(dev, s);
        });
        assert!(!ran, "ttsim now executes {}: write its gate", s.name);
    }
}

/// Silicon: written, not run. Risk class: documented (BlackholeA0 page),
/// UNVERIFIED on silicon, NoC-hang class (an L1-only unicast atomic to a
/// surviving neighbour, tile (4, 4); `harness::tile` checks it against the chip's
/// ARC grid first). One test per form, each its own process; the first, the
/// full-width increment, has a measured precedent (the mover barrier).
#[cfg(feature = "silicon")]
mod silicon {
    use super::*;

    fn scenario(prefix: &str) -> Scenario {
        scenarios()
            .into_iter()
            .find(|s| s.name.starts_with(prefix))
            .unwrap_or_else(|| panic!("no scenario {prefix}"))
    }

    fn gate(prefix: &str) {
        harness::assert_on_silicon();
        let s = scenario(prefix);
        harness::in_device(|dev| device_matches(dev, &s, Mutant::None).unwrap());
    }

    fn mutant_diverges(prefix: &str, mutant: Mutant) {
        harness::assert_on_silicon();
        let s = scenario(prefix);
        harness::in_device(|dev| {
            assert!(
                device_matches(dev, &s, mutant).is_err(),
                "{mutant:?} agrees with the device on {}",
                s.name
            );
        });
    }

    /// Run this first, alone.
    #[test]
    fn silicon_noc_atomic_minimal_probe() {
        harness::assert_on_silicon();
        let s = scenario("increment, full width");
        assert!(harness::survives(|dev| device_matches(
            dev,
            &s,
            Mutant::None
        )
        .unwrap()));
    }

    macro_rules! gates {
        ($($test:ident => $prefix:expr),* $(,)?) => {
            $( #[test] fn $test() { gate($prefix); } )*
        };
    }
    gates! {
        silicon_noc_atomic_increment_8_bits => "increment, 8 bits",
        silicon_noc_atomic_cas_equal => "cas, equal",
        silicon_noc_atomic_cas_different => "cas, different",
        silicon_noc_atomic_swap_mask => "swap, mask",
        silicon_noc_atomic_swap_index_6 => "swap, index form 6",
        silicon_noc_atomic_swap_index_7 => "swap, index form 7",
        silicon_noc_atomic_swap_index_10 => "swap, index form 10",
        silicon_noc_atomic_zaamo_add => "zaamo add",
        silicon_noc_atomic_zaamo_xor => "zaamo xor",
        silicon_noc_atomic_zaamo_or => "zaamo or",
        silicon_noc_atomic_zaamo_and => "zaamo and",
        silicon_noc_atomic_zaamo_min => "zaamo min is",
        silicon_noc_atomic_zaamo_max => "zaamo max is",
        silicon_noc_atomic_zaamo_minu => "zaamo minu",
        silicon_noc_atomic_zaamo_maxu => "zaamo maxu",
        silicon_noc_atomic_acc_u32 => "parallel add, u32",
        silicon_noc_atomic_acc_u8_saturating => "parallel add, u8 saturates",
        silicon_noc_atomic_acc_u8_wrapping => "parallel add, u8 wraps",
        silicon_noc_atomic_acc_fp32 => "parallel add, fp32",
        silicon_noc_atomic_acc_bf16 => "parallel add, bf16",
        silicon_noc_atomic_acc_fp16 => "parallel add, fp16",
    }

    #[test]
    fn silicon_rejects_the_swap_mask_mutant() {
        mutant_diverges("swap, mask", Mutant::SwapMaskHalvesSwapped);
    }

    #[test]
    fn silicon_rejects_the_cas_nibble_mutant() {
        mutant_diverges("cas, different", Mutant::CasComparesLowNibble);
    }
}
