//! Diagnostics behind the `MOVDBGA2D` / `MOVA2D` increment-1 `AddrMod` finding
//! (`hardware-coverage.md`, `MOVDBGA2D`), which is now explained and encoded in
//! `tt_isa::matrix::debug::model`; kept as the evidence. **These assert nothing.**
//! Each silicon test runs a handful of tiny programs and prints what the hardware
//! did, decoded as "row r holds source row s", so the evidence can be read without
//! a model.
//!
//! # What step111 already shows (card 0, `target/silicon/out/17915924*`)
//!
//! Decoding the failing cases' device rows by source row (`a_bits()` gives every
//! source row 0..16 a distinct value):
//!
//! * `single_move_names_entry_1` -- one move, `SrcRow 2`, `DstRow 4`, `AddrMod 1`:
//!   device rows 4..8 hold source rows 0..4. That is a **four-row aligned block**
//!   (`DstRow & !3`, `SrcRow & !3`), not the one row the model writes. It is the
//!   move's *own* write, before any increment applies.
//! * `two_rows_*`: the second move (`AddrMod 0`, same operands) lands exactly where
//!   the model says (`+1` source row, `+1` Dst row, per entry 1's increments); the
//!   rows the model leaves empty are the leftovers of the first move's four rows.
//!   So the table writes, the entry index, the counter units and the "advance after
//!   the move" order all behave as modelled; only the first move's *width* differs.
//! * The `AddrMod 0` cases and the two-block case (`Move8Rows` set, bits 13 and 14
//!   both set) are exact, which is why they never saw it.
//!
//! The finding is that on Blackhole bit 14 of `MOVA2D` and `MOVDBGA2D`, with
//! `Move8Rows` (bit 13) clear, is not a pure `AddrMod` bit: it also widens the move
//! to four rows. The sweeps below separated that from the other hypotheses (a)
//! increment units, (b) a shifted entry, (c) misplaced table words, (d) a leftover
//! RWC base.
//!
//! # How to read the output
//!
//! Every run builds the same shape: reset the RWCs (unless the case says not to),
//! write the whole address-modifier table, one *first* move carrying the bits under
//! test, then (optionally) a plain one-row *second* move (`AddrMod 0`) whose landing
//! row exposes the counters the first move's entry applied. Entry `k` is configured
//! as `SrcA += k`, `Dst += 7 - k` in the entry-selection sweeps, so the pair
//! `(s, d)` the second move shows names the entry the hardware used (`s == k`,
//! `d == 7 - k`).
//!
//! Output lines start with the case label, then `r0=- r1=s3 ...` for the sixteen
//! `Dst` rows (`-` zero, `sN` holds source row N, `?` unrecognised).
//!
//! Silicon only (ttsim implements neither the one-row `MOVA2D` nor `MOVDBGA2D`,
//! divergence rows 37 and 83); the one simulator test checks the host side
//! (program builder, decoder) with the eight-row `MOVA2D` ttsim does run.
#![cfg_attr(not(feature = "silicon"), allow(dead_code))]
mod matrix_debug_support;
use matrix_debug_support::{a_bits, eight, entry1, one, rows16};
use tt_isa::{
    backend::{self, Before},
    isa::{generated::encode, Instruction},
    matrix::debug::{
        AddrModEntry, AddrModTable, DebugFormat, DebugMode, DebugMove, Rows, SrcAFormat,
    },
    matrix::{Banks, Loaded},
};
use tt_tests::datapath::{set_adc_x, Unpacker, STAGE};
use tt_tests::harness::{self, Roles, Run};
use tt_tests::matmul::{self, stage_operand, ROW, SRC_A_ROW, SRC_B_ROW, TF32_CODE};

const STAGE_A: u64 = STAGE;
const STAGE_B: u64 = STAGE + 0x2000;

type MatB = [[f32; 16]; 8];
type Pair = (Vec<Instruction>, Vec<Instruction>);

// ---------------------------------------------------------------- the data

/// The IEEE bits a TF32 `Src` datum moves to FP32 `Dst` as, denormals flushed.
fn direct(bits: u32) -> u32 {
    if (bits >> 23) & 0xff == 0 {
        0
    } else {
        bits & 0xffff_e000
    }
}

/// What each source row looks like in `Dst` after a move.
fn moved_rows(bits: &[u32]) -> Vec<Vec<u32>> {
    (0..16)
        .map(|r| (0..16).map(|c| direct(bits[r * 16 + c])).collect())
        .collect()
}

fn fp32_format() -> DebugFormat {
    DebugFormat::new(DebugMode::Dst32Tf32, SrcAFormat::Tf32, true).unwrap()
}

// ---------------------------------------------------------------- programs

fn operands(na: u32, nb: u32) -> matmul::Operands {
    matmul::Operands {
        a_addr: STAGE_A,
        na,
        b_addr: STAGE_B,
        nb,
        out: TF32_CODE,
    }
}

/// Both operands unpacked and handed to the Matrix Unit, then `body` on the math thread
/// (the step111 shape).
fn loaded_program(
    na: u32,
    nb: u32,
    body: impl FnOnce(Banks<Loaded, Loaded>, &mut Vec<Instruction>),
) -> Pair {
    let mut up = matmul::unpack_prelude(operands(na, nb));
    let unpack = encode::UnpacrRegular::ZERO.multi_context_mode(1);
    up.push(set_adc_x(Unpacker::SrcA, 0, na - 1));
    let (i, banks) = Banks::after_reset().unpack_a(unpack).unwrap();
    up.push(i);
    up.push(set_adc_x(Unpacker::SrcB, 0, nb - 1));
    let (i, banks) = banks.unpack_b(unpack).unwrap();
    up.push(i);
    up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    up.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());
    let mut math = matmul::math_prelude();
    body(banks, &mut math);
    math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
    (up, math)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Mover {
    Mova2d,
    Movdbga2d,
}

/// What to do to the RWCs before the table is written.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Rwc {
    /// Leave whatever the previous program on the thread left (silicon keeps it).
    Untouched,
    /// `SETRWC` `SrcA`, `SrcB`, `Dst` and `FidelityPhase` to zero (`Cr` too).
    Reset,
    /// Positive control for the instrument: `SETRWC` `Dst = dst`, `SrcA = src_a`.
    Set { dst: u32, src_a: u32 },
}

fn setrwc(dst: u32, src_a: u32) -> Instruction {
    encode::Setrwc::ZERO
        .src_a(1)
        .src_a_val(src_a)
        .src_b(1)
        .src_b_val(0)
        .dst(1)
        .dst_val(dst)
        .fidelity(1)
        .encode()
        .unwrap()
}

/// One diagnostic program.
#[derive(Clone)]
struct Probe {
    mover: Mover,
    table: AddrModTable,
    rwc: Rwc,
    /// The first move, built with `AddrMod 0` by the checked helper.
    first: Rows,
    /// Raw bits ORed onto the first move's word: `AddrMod k` is `k << 14`.
    first_bits: u32,
    /// A plain `AddrMod 0` move after it, if any.
    second: Option<Rows>,
}

fn emit(
    banks: Banks<Loaded, Loaded>,
    p: &mut Vec<Instruction>,
    mover: Mover,
    rows: Rows,
    bits: u32,
) -> Banks<Loaded, Loaded> {
    match mover {
        Mover::Mova2d => {
            let (src, dst, eight) = match rows {
                Rows::One { src_row, dst_row } => (src_row, dst_row, 0),
                Rows::Eight { src_row, dst_row } => (src_row, dst_row, 1),
            };
            let base = encode::Mova2D::ZERO
                .src_row(src)
                .dst_row(dst)
                .addr_mod(0)
                .move8_rows(eight);
            let (i, next) = banks.mova2d(base).unwrap();
            p.push(Instruction::new(i.word() | bits, i.def()));
            next
        }
        Mover::Movdbga2d => {
            let mv = DebugMove::new(rows, 0, fp32_format()).unwrap();
            let ([wait, i], next) = banks.debug_move_a(mv).unwrap();
            p.push(wait);
            p.push(Instruction::new(i.word() | bits, i.def()));
            next
        }
    }
}

fn program(na: u32, nb: u32, probe: Probe) -> Pair {
    loaded_program(na, nb, move |banks, math| {
        match probe.rwc {
            Rwc::Untouched => {}
            Rwc::Reset => math.push(setrwc(0, 0)),
            Rwc::Set { dst, src_a } => math.push(setrwc(dst, src_a)),
        }
        math.extend(probe.table.setup().unwrap());
        let mut banks = emit(banks, math, probe.mover, probe.first, probe.first_bits);
        if let Some(second) = probe.second {
            banks = emit(banks, math, probe.mover, second, 0);
        }
        // Hand both banks back, as step111 does.
        let (i, banks) = banks.release_a().unwrap();
        math.push(i);
        let (i, _) = banks.release_b().unwrap();
        math.push(i);
    })
}

/// Run a probe and return `Dst` rows 0..16.
fn run_probe(dev: &mut harness::Dev<'_>, probe: &Probe, bits: &[u32]) -> Vec<Vec<u32>> {
    let (sa, na) = stage_operand(SRC_A_ROW, &rows16(bits));
    let zero_b: MatB = [[0f32; 16]; 8];
    let (sb, nb) = stage_operand(SRC_B_ROW, &zero_b);
    let programs = program(na, nb, probe.clone());
    let out = harness::run(
        dev,
        &Run::roles(Roles {
            unpack: &programs.0,
            math: &programs.1,
            pack: &[],
        })
        .stage(&[(STAGE_A, &sa), (STAGE_B, &sb)])
        .dump_rows(16),
    );
    (0..16)
        .map(|r| (0..ROW).map(|c| out.dst_at(r, c)).collect())
        .collect()
}

// ---------------------------------------------------------------- decoding

/// `-` for an all-zero row, `sN` if it equals source row `N`, else `?`.
fn label(row: &[u32], src: &[Vec<u32>]) -> String {
    if row.iter().all(|&v| v == 0) {
        return "-".to_string();
    }
    match src.iter().position(|s| s == row) {
        Some(n) => format!("s{n}"),
        None => "?".to_string(),
    }
}

/// `Some(source row)` for a row that holds exactly one source row.
fn source_of(row: &[u32], src: &[Vec<u32>]) -> Option<usize> {
    if row.iter().all(|&v| v == 0) {
        None
    } else {
        src.iter().position(|s| s == row)
    }
}

fn map_line(dst: &[Vec<u32>], src: &[Vec<u32>]) -> String {
    (0..16)
        .map(|r| format!("r{r}={}", label(&dst[r], src)))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The rows the first move wrote (below row 8) as `dst<-src` pairs, then the row
/// the second move wrote (the one holding a source row `>= 8`, in rows `>= 8`).
fn summary(dst: &[Vec<u32>], src: &[Vec<u32>]) -> String {
    let first: Vec<String> = (0..8)
        .filter_map(|r| source_of(&dst[r], src).map(|s| format!("{r}<-s{s}")))
        .collect();
    let second: Vec<String> = (8..16)
        .filter_map(|r| source_of(&dst[r], src).map(|s| format!("{r}<-s{s}")))
        .collect();
    format!(
        "first move wrote [{}]; rows 8.. [{}]",
        first.join(" "),
        second.join(" ")
    )
}

/// Entry `k` of the selection sweeps: `SrcA += k`, `Dst += 7 - k`.
fn entry_table() -> AddrModTable {
    let mut t = AddrModTable::ZERO;
    for k in 0..8u32 {
        t = t
            .with(
                k as usize,
                AddrModEntry {
                    src_a_incr: k,
                    dst_incr: 7 - k,
                },
            )
            .unwrap();
    }
    t
}

// ---------------------------------------------------------------- the sweeps

/// A single move (`first`) with raw `bits`, then a plain one-row move at
/// `src 8 -> dst 8` that exposes the counters: print the map and the entry it
/// implies (`s == k`, `d == 7 - k` for a table from [`entry_table`]).
fn entry_case(
    dev: &mut harness::Dev<'_>,
    mover: Mover,
    first: Rows,
    bits: u32,
    rwc: Rwc,
    tag: &str,
) {
    let data = a_bits();
    let src = moved_rows(&data);
    let probe = Probe {
        mover,
        table: entry_table(),
        rwc,
        first,
        first_bits: bits,
        second: Some(one(8, 8)),
    };
    let dst = run_probe(dev, &probe, &data);
    let second = (8..16).find_map(|r| source_of(&dst[r], &src).map(|s| (r, s)));
    let implied = match second {
        Some((r, s)) if s >= 8 => {
            let (s_adv, d_adv) = (s - 8, r - 8);
            format!(
                "second move read src {s} wrote row {r}: SrcA +{s_adv}, Dst +{d_adv} => entry {s_adv}{}",
                if d_adv + s_adv == 7 { "" } else { " (src and dst disagree!)" }
            )
        }
        Some((r, s)) => format!("rows >= 8 hold s{s} at r{r}: unexpected"),
        None => "second move wrote nothing visible".to_string(),
    };
    println!("[{tag}] {mover:?} raw 0x{bits:05x} {rwc:?}");
    println!("    {}", map_line(&dst, &src));
    println!("    {}; {implied}", summary(&dst, &src));
}

/// `AddrMod` 0..8 on one-row moves (`Move8Rows` clear): which rows each writes and
/// which entry the counters show.
fn entry_select_one_row(mover: Mover) {
    harness::assert_on_silicon();
    harness::in_device(|dev| {
        for k in 0..8u32 {
            entry_case(
                dev,
                mover,
                one(2, 4),
                k << 14,
                Rwc::Reset,
                &format!("one-row AddrMod {k}"),
            );
        }
    });
}

/// The same with `Move8Rows` set (bit 13): the first move's mask hides row-unit
/// increments, but the second move still shows which entry applied.
fn entry_select_eight_rows(mover: Mover) {
    harness::assert_on_silicon();
    harness::in_device(|dev| {
        for k in 0..8u32 {
            entry_case(
                dev,
                mover,
                eight(0, 0),
                k << 14,
                Rwc::Reset,
                &format!("eight-row AddrMod {k}"),
            );
        }
    });
}

/// Sweep entry 1's `Dst` increment 0..=9 with `SrcA` 0. First move at `dst 0`, second
/// `src 8 -> dst 6`: the second lands on row `6 + d`.
fn dst_incr_sweep(mover: Mover) {
    harness::assert_on_silicon();
    let data = a_bits();
    let src = moved_rows(&data);
    harness::in_device(|dev| {
        for d in 0..=9u32 {
            let probe = Probe {
                mover,
                table: entry1(0, d),
                rwc: Rwc::Reset,
                first: one(2, 0),
                first_bits: 1 << 14,
                second: Some(one(8, 6)),
            };
            let dst = run_probe(dev, &probe, &data);
            let landed = (6..16).find(|&r| source_of(&dst[r], &src) == Some(8));
            println!("[dst incr {d}] {mover:?} entry 1 = (SrcA +0, Dst +{d})");
            println!("    {}", map_line(&dst, &src));
            println!(
                "    second move (src 8 -> row 6) landed at row {landed:?} => observed Dst advance {:?} (rows)",
                landed.map(|r| r as i64 - 6)
            );
        }
    });
}

/// Sweep entry 1's `SrcA` increment 0..=9 with `Dst` 0. First move at `dst 0`, second
/// `src 4 -> dst 15`: the second reads source row `4 + s`.
fn src_incr_sweep(mover: Mover) {
    harness::assert_on_silicon();
    let data = a_bits();
    let src = moved_rows(&data);
    harness::in_device(|dev| {
        for s in 0..=9u32 {
            let probe = Probe {
                mover,
                table: entry1(s, 0),
                rwc: Rwc::Reset,
                first: one(2, 0),
                first_bits: 1 << 14,
                second: Some(one(4, 15)),
            };
            let dst = run_probe(dev, &probe, &data);
            let read = source_of(&dst[15], &src);
            println!("[src incr {s}] {mover:?} entry 1 = (SrcA +{s}, Dst +0)");
            println!("    {}", map_line(&dst, &src));
            println!(
                "    second move (src 4 -> row 15) read source row {read:?} => observed SrcA advance {:?} (rows)",
                read.map(|r| r as i64 - 4)
            );
        }
    });
}

// -------------------------------------------------------------- silicon tests

/// Run 1: which rows does a one-row move write when it names entry k, and which
/// entry do the counters show? Separates (b) from the "width" finding.
#[test]
#[cfg(feature = "silicon")]
fn entry_select_one_row_movdbga2d() {
    entry_select_one_row(Mover::Movdbga2d);
}

#[test]
#[cfg(feature = "silicon")]
fn entry_select_one_row_mova2d() {
    entry_select_one_row(Mover::Mova2d);
}

/// Run 2: the same with `Move8Rows` set.
#[test]
#[cfg(feature = "silicon")]
fn entry_select_eight_rows_movdbga2d() {
    entry_select_eight_rows(Mover::Movdbga2d);
}

#[test]
#[cfg(feature = "silicon")]
fn entry_select_eight_rows_mova2d() {
    entry_select_eight_rows(Mover::Mova2d);
}

/// Run 3: `Dst` increment 0..=9 alone (units: hypothesis (a)).
#[test]
#[cfg(feature = "silicon")]
fn dst_incr_sweep_movdbga2d() {
    dst_incr_sweep(Mover::Movdbga2d);
}

#[test]
#[cfg(feature = "silicon")]
fn dst_incr_sweep_mova2d() {
    dst_incr_sweep(Mover::Mova2d);
}

/// Run 4: `SrcA` increment 0..=9 alone (units / width: hypothesis (a)).
#[test]
#[cfg(feature = "silicon")]
fn src_incr_sweep_movdbga2d() {
    src_incr_sweep(Mover::Movdbga2d);
}

#[test]
#[cfg(feature = "silicon")]
fn src_incr_sweep_mova2d() {
    src_incr_sweep(Mover::Mova2d);
}

/// Run 5: hypothesis (d). With and without an explicit RWC reset, `AddrMod 0`
/// and `AddrMod 1`, plus a positive control that the instrument sees the RWCs
/// (`SETRWC Dst = 3, SrcA = 5` must move a plain one-row `2 -> 4` to
/// `row 7 <- src 7`). A previous program's counters survive, so the untouched
/// arms run after the reset arms' dirtying (the table's entry-selected
/// increments are left in the RWCs on purpose).
#[test]
#[cfg(feature = "silicon")]
fn rwc_reset_and_leftover_mova2d() {
    harness::assert_on_silicon();
    harness::in_device(|dev| {
        for (rwc, tag) in [
            (Rwc::Reset, "reset"),
            (Rwc::Set { dst: 3, src_a: 5 }, "poked Dst 3, SrcA 5"),
            (Rwc::Untouched, "untouched after the poke"),
            (Rwc::Reset, "reset again"),
        ] {
            for k in [0u32, 1] {
                entry_case(
                    dev,
                    Mover::Mova2d,
                    one(2, 4),
                    k << 14,
                    rwc,
                    &format!("rwc {tag}, AddrMod {k}"),
                );
            }
        }
    });
}

/// The reset case for `MOVDBGA2D` itself.
#[test]
#[cfg(feature = "silicon")]
fn rwc_reset_and_leftover_movdbga2d() {
    harness::assert_on_silicon();
    harness::in_device(|dev| {
        for (rwc, tag) in [
            (Rwc::Reset, "reset"),
            (Rwc::Set { dst: 3, src_a: 5 }, "poked Dst 3, SrcA 5"),
            (Rwc::Untouched, "untouched after the poke"),
            (Rwc::Reset, "reset again"),
        ] {
            for k in [0u32, 1] {
                entry_case(
                    dev,
                    Mover::Movdbga2d,
                    one(2, 4),
                    k << 14,
                    rwc,
                    &format!("rwc {tag}, AddrMod {k}"),
                );
            }
        }
    });
}

/// Run 6, last, `UNVERIFIED` reserved bit: bit 12 (LLK's `instr_mod` is `<< 12`
/// with `MOV_8_ROWS = 2`, so bit 12 is the unused low bit of that field) on a
/// one-row move with `AddrMod 0` and `AddrMod 1`, to see whether bit 12 or bit
/// 14 carries the four-row behaviour. Risk class: UNVERIFIED encoding of a
/// documented-reserved bit; a `MOVA2D` word with a reserved bit set has never
/// run on these cards. `MOVA2D` only.
#[ignore = "UNVERIFIED: sets a reserved instruction bit (bit 12); run only deliberately, alone, after the documented sweeps"]
#[test]
#[cfg(feature = "silicon")]
fn probe_reserved_bit_12_mova2d() {
    harness::assert_on_silicon();
    assert!(harness::survives(|dev| {
        for (bits, tag) in [
            (1u32 << 12, "bit 12"),
            ((1 << 12) | (1 << 14), "bits 12 and 14"),
        ] {
            entry_case(
                dev,
                Mover::Mova2d,
                one(2, 4),
                bits,
                Rwc::Reset,
                &format!("reserved {tag}"),
            );
        }
    }));
}

// ------------------------------------------------------------ host mechanics

/// Host side of the diagnostics on ttsim: the same program builder, `SETRWC` reset
/// prelude, table write and decoder, with the only form ttsim runs (eight-row
/// `MOVA2D`). Entry 1 advances both counters by 8; the second move is eight-row
/// too. Expect rows 0..8 = source rows 0..8 and rows 8..16 = source rows 8..16,
/// decoded by [`label`]; a table that did not advance (negative control) must not
/// produce the second block.
#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_checks_the_diagnostics_host_side() {
    let data = a_bits();
    let src = moved_rows(&data);
    harness::in_device(|dev| {
        let run = |dev: &mut harness::Dev<'_>, table: AddrModTable| {
            let probe = Probe {
                mover: Mover::Mova2d,
                table,
                rwc: Rwc::Reset,
                first: eight(0, 0),
                first_bits: 1 << 14,
                second: Some(eight(0, 0)),
            };
            run_probe(dev, &probe, &data)
        };
        let dst = run(dev, entry1(8, 8));
        println!("{}", map_line(&dst, &src));
        for (r, row) in dst.iter().enumerate() {
            assert_eq!(label(row, &src), format!("s{r}"), "row {r}");
        }
        // Negative control: no advance, the second block never appears.
        let flat = run(dev, entry1(0, 0));
        for (r, row) in flat.iter().enumerate().skip(8) {
            assert_eq!(label(row, &src), "-", "row {r} without an advance");
        }
        // The decoder names a mis-ordered image as such.
        let mut swapped = dst.clone();
        swapped.swap(0, 1);
        assert_eq!(label(&swapped[0], &src), "s1");
    });
}
