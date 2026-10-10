//! Matrix diagnostics: `MOVDBGA2D` and `GATESRCRST` (`tt_isa::matrix::debug`).
//!
//! Both are diagnostic surfaces, so there is no Burn routing, gradient or tensor
//! padding to establish: the gates below move `SrcA` rows through the Matrix Unit
//! on one tile and read `Dst` back, and nothing reaches a tensor API.
//!
//! # Oracles
//!
//! `MOVDBGA2D` is `MOVA2D` without the bank-ownership wait. The expected `Dst`
//! comes from two routes that must agree: `matrix_debug::model` (a port of the
//! page's functional model, over physical `Src` datums) and a direct IEEE bit
//! formula (`bits & 0xffffe000`, flushed when the exponent is zero), which never
//! touches the physical `Src`/`Dst` layouts. ttsim refuses `MOVDBGA2D`
//! (divergence row 50), but it runs the eight-row `MOVA2D` the model also
//! describes, so that control validates the model's TF32 path and row masking on
//! the simulator; the debug forms themselves are silicon-only gates.
//!
//! `GATESRCRST` has no observable-state oracle on the page: the cache is "one
//! slot", invalidated by hardware, and the instruction "should only be required if
//! there are hardware bugs". The experiment is therefore stated before it runs
//! (`matrix_debug::cache_verdict`): rewrite `SrcB` through `MOVD2B` (not an unpack),
//! run `MVMUL` with and without `GATESRCRST`, and require the *no-gate* arm to be
//! stale and the gate arm fresh for the instruction to count as having an observable
//! effect. Both arms fresh is the `[-]` "no observable oracle" outcome; a no-effect
//! comparison alone proves nothing about invalidation.
// The silicon-only gates share helpers the simulator build does not call.
#![cfg_attr(not(feature = "silicon"), allow(dead_code))]
use tt_isa::{
    backend::{self, Before},
    isa::{generated::encode, Instruction},
    matrix::debug::{
        self as matrix_debug, model, AddrModEntry, AddrModTable, CacheArm, CacheOutcome,
        CacheVerdict, DebugFormat, DebugMode, DebugMove, Rows, SrcAFormat,
    },
    matrix::{Banks, Empty, Filling, Loaded},
    numerics::mvmul_reference,
};
use tt_tests::datapath::{config_program, set_adc_x, Unpacker, STAGE};
use tt_tests::harness::{self, Roles, Run};
use tt_tests::matmul::{self, stage_operand, ROW, SRC_A_ROW, SRC_B_ROW, TF32_CODE};

const STAGE_A: u64 = STAGE;
const STAGE_B: u64 = STAGE + 0x2000;

type MatA = [[f32; 16]; 16];
type MatB = [[f32; 16]; 8];
type LoadedBody = Box<dyn FnOnce(Banks<Loaded, Loaded>, &mut Vec<Instruction>)>;
type FillingBody = Box<dyn FnOnce(Banks<Filling, Empty>, &mut Vec<Instruction>)>;

fn operands(na: u32, nb: u32) -> matmul::Operands {
    matmul::Operands {
        a_addr: STAGE_A,
        na,
        b_addr: STAGE_B,
        nb,
        out: TF32_CODE,
    }
}

/// Both operands unpacked and handed to the Matrix Unit, then `body` on the math thread.
fn loaded_program(na: u32, nb: u32, body: LoadedBody) -> (Vec<Instruction>, Vec<Instruction>) {
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

/// `SrcA` unpacked *without* a flip: the unpacker still owns the bank.
fn filling_program(na: u32, nb: u32, body: FillingBody) -> (Vec<Instruction>, Vec<Instruction>) {
    let mut up = matmul::unpack_prelude(operands(na, nb));
    let unpack = encode::UnpacrRegular::ZERO.multi_context_mode(1);
    up.push(set_adc_x(Unpacker::SrcA, 0, na - 1));
    let (i, banks) = Banks::after_reset().unpack_a_partial(unpack).unwrap();
    up.push(i);
    up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    let mut math = matmul::math_prelude();
    body(banks, &mut math);
    math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
    (up, math)
}

/// Run one program pair and return `Dst` rows 0..16 (the harness's maximum dump).
fn run_on(
    dev: &mut harness::Dev<'_>,
    programs: &(Vec<Instruction>, Vec<Instruction>),
    stage: &[(u64, &[u8])],
) -> Vec<u32> {
    let out = harness::run(
        dev,
        &Run::roles(Roles {
            unpack: &programs.0,
            math: &programs.1,
            pack: &[],
        })
        .stage(stage)
        .dump_rows(16),
    );
    (0..16 * ROW)
        .map(|f| out.dst_at(f / ROW, f % ROW))
        .collect()
}

// ---------------------------------------------------------------- the data

/// `SrcA` as FP32 bits, rows 0..16: specials first, then a deterministic spread.
/// Every value is finite (the unpacker's NaN/infinity behavior on silicon is not what
/// this gate measures; the host model test covers them).
fn a_bits() -> Vec<u32> {
    let mut bits = vec![
        0x0000_0000, // +0
        0x8000_0000, // -0: flushed to +0 by the move
        0x0000_0001, // smallest subnormal
        0x807f_ffff, // largest negative subnormal
        0x0080_0000, // smallest normal
        0x8080_0000,
        0x7f7f_ffff, // largest finite: truncation keeps it finite
        0xff7f_ffff,
        0x3f80_1fff, // low 13 mantissa bits set: truncated, not rounded
        0xbf80_1fff,
        0x3fff_ffff,
        0xbfff_ffff,
        0x3f80_0000,
        0xbf80_0000,
        0x4048_f5c3,
        0xc0c9_0fdb,
    ];
    let mut x = 0x1234_5678u32;
    while bits.len() < 256 {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let exp = 100 + (x >> 24) % 40;
        bits.push((x & 0x8000_0000) ^ (exp << 23) ^ (x.rotate_left(7) & 0x007f_ffff));
    }
    bits
}

fn rows16(bits: &[u32]) -> MatA {
    let mut m = [[0f32; 16]; 16];
    for (i, &b) in bits.iter().enumerate() {
        m[i / 16][i % 16] = f32::from_bits(b);
    }
    m
}

fn zero_b() -> MatB {
    [[0f32; 16]; 8]
}

/// The IEEE bits a TF32 `Src` datum moves to FP32 `Dst` as: no layout involved.
fn direct(bits: u32, flush: bool) -> u32 {
    if flush && (bits >> 23) & 0xff == 0 {
        0
    } else {
        bits & 0xffff_e000
    }
}

fn src_bank(bits: &[u32]) -> [[u32; 16]; 64] {
    let mut bank = [[0u32; 16]; 64];
    for (i, &b) in bits.iter().enumerate() {
        bank[i / 16][i % 16] = model::tf32_src_datum(b);
    }
    bank
}

fn fp32_format(src: SrcAFormat, flush: bool) -> DebugFormat {
    DebugFormat::new(DebugMode::Dst32Tf32, src, flush).unwrap()
}

/// One move sequence and the address-modifier table it runs under.
#[derive(Clone)]
struct Case {
    moves: Vec<(Rows, u32)>,
    table: AddrModTable,
}

/// Entry 1 advances `SrcA` by `src` rows and `Dst` by `dst`; every other entry
/// (and every other field of every entry) is zero.
fn entry1(src: u32, dst: u32) -> AddrModTable {
    AddrModTable::ZERO
        .with(
            1,
            AddrModEntry {
                src_a_incr: src,
                dst_incr: dst,
            },
        )
        .unwrap()
}

fn one(src_row: u32, dst_row: u32) -> Rows {
    Rows::One { src_row, dst_row }
}

fn eight(src_row: u32, dst_row: u32) -> Rows {
    Rows::Eight { src_row, dst_row }
}

/// `Dst` rows 0..16 after `case`, from the model, cross-checked against [`direct`]
/// with a separate counter walk. The model takes the table as input.
fn expected(bits: &[u32], case: &Case, format: DebugFormat) -> Vec<u32> {
    let bank = src_bank(bits);
    let mut dst = vec![0u32; 16 * ROW];
    let state = model::State {
        format,
        src_rwc: 0,
        dst_base: 0,
        block_columns: [0; 8],
    };
    model::run_moves(&bank, state, &case.table, &case.moves, false, |w| {
        let model::Write::Dst32 { row, column, value } = w else {
            panic!("Fp32 mode writes 32-bit datums: {w:?}")
        };
        dst[row * ROW + column] = model::dst32_to_ieee(value);
    })
    .unwrap();
    // Independent route: the same rows by IEEE bit arithmetic and a plain counter walk.
    let mut check = vec![0u32; 16 * ROW];
    let (mut src_rwc, mut dst_rwc) = (0u32, 0u32);
    for &(rows, addr_mod) in &case.moves {
        let (s, d, n) = match rows {
            // Measured on card 0 (step111b): instruction bit 14, the low bit of the entry
            // number, widens a one-row move to a four-row aligned block.
            Rows::One { src_row, dst_row } if addr_mod & 1 == 1 => {
                ((src_row + src_rwc) & 0x3c, (dst_row + dst_rwc) & 0x3fc, 4)
            }
            Rows::One { src_row, dst_row } => {
                ((src_row + src_rwc) & 0x3f, (dst_row + dst_rwc) & 0x3ff, 1)
            }
            Rows::Eight { src_row, dst_row } => {
                ((src_row + src_rwc) & 0x38, (dst_row + dst_rwc) & 0x3f8, 8)
            }
        };
        for k in 0..n {
            for c in 0..16 {
                check[(d + k) as usize * ROW + c] =
                    direct(bits[(s + k) as usize * 16 + c], format.flush_denormals());
            }
        }
        let e = case.table.entry(addr_mod as usize);
        src_rwc += e.src_a_incr;
        dst_rwc += e.dst_incr;
    }
    assert_eq!(dst, check, "model and direct formula disagree");
    dst
}

/// Compare two `Dst` images; on a mismatch print every differing row in full,
/// device then model, then panic with the first difference.
fn equal(got: &[u32], want: &[u32], context: &str) {
    assert_eq!(got.len(), want.len());
    let mut first = None;
    for r in 0..got.len() / ROW {
        let (g, w) = (&got[r * ROW..(r + 1) * ROW], &want[r * ROW..(r + 1) * ROW]);
        if g != w {
            first.get_or_insert(r);
            let hex = |row: &[u32]| {
                row.iter()
                    .map(|v| format!("{v:08x}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            eprintln!(
                "{context}: Dst row {r}\n  device {}\n  model  {}",
                hex(g),
                hex(w)
            );
        }
    }
    if let Some(r) = first {
        let c = (0..ROW)
            .find(|&c| got[r * ROW + c] != want[r * ROW + c])
            .unwrap();
        panic!(
            "{context}: first difference at Dst row {r}, column {c}: device {:08x}, model {:08x}",
            got[r * ROW + c],
            want[r * ROW + c]
        );
    }
}

fn stage_a(bits: &[u32]) -> (Vec<u8>, u32) {
    stage_operand(SRC_A_ROW, &rows16(bits))
}

fn stage_b() -> (Vec<u8>, u32) {
    stage_operand(SRC_B_ROW, &zero_b())
}

/// The configuration of `format`, emitted on the math thread.
fn set_format(p: &mut Vec<Instruction>, format: DebugFormat) {
    let (disable, words) = format.setup().unwrap();
    p.push(disable);
    p.extend(config_program(&words));
}

/// Which instruction moves, in a gate that compares `MOVA2D` with `MOVDBGA2D`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Mover {
    Mova2d,
    Movdbga2d,
}

/// The `MOVA2D` with `rows` and `addr_mod`: same operands as `MOVDBGA2D`.
fn mova2d_base(rows: Rows, addr_mod: u32) -> encode::Mova2D {
    let (src, dst, eight) = match rows {
        Rows::One { src_row, dst_row } => (src_row, dst_row, 0),
        Rows::Eight { src_row, dst_row } => (src_row, dst_row, 1),
    };
    encode::Mova2D::ZERO
        .src_row(src)
        .dst_row(dst)
        .addr_mod(addr_mod)
        .move8_rows(eight)
}

/// A Loaded-`SrcA` program of `case` by `mover`. The address-modifier table is
/// written in full first; `with_format` establishes `format` (the checked-helper
/// path), otherwise the unpack prelude's own configuration stands.
fn loaded_moves(
    na: u32,
    nb: u32,
    mover: Mover,
    case: Case,
    format: DebugFormat,
    with_format: bool,
    configure_table: bool,
) -> (Vec<Instruction>, Vec<Instruction>) {
    loaded_program(
        na,
        nb,
        Box::new(move |banks, p| {
            if with_format {
                set_format(p, format);
            }
            if configure_table {
                p.extend(case.table.setup().unwrap());
            }
            let mut banks = banks;
            for (rows, addr_mod) in case.moves {
                match mover {
                    Mover::Mova2d => {
                        let (i, next) = banks.mova2d(mova2d_base(rows, addr_mod)).unwrap();
                        p.push(i);
                        banks = next;
                    }
                    Mover::Movdbga2d => {
                        let mv = DebugMove::new(rows, addr_mod, format).unwrap();
                        let (words, next) = banks.debug_move_a(mv).unwrap();
                        p.extend(words);
                        banks = next;
                    }
                }
            }
            // Hand both banks back: the state outlives the program, so the next
            // run's reset-state assumption holds.
            let (i, banks) = banks.release_a().unwrap();
            p.push(i);
            let (i, _) = banks.release_b().unwrap();
            p.push(i);
        }),
    )
}

/// Run `case` with `mover` over `bits` and compare with the model. One case per
/// call, so each silicon test reports its own result.
fn check_with(
    mover: Mover,
    bits: &[u32],
    case: Case,
    format: DebugFormat,
    with_format: bool,
    label: &str,
) {
    let (sa, na) = stage_a(bits);
    let (sb, nb) = stage_b();
    let want = expected(bits, &case, format);
    harness::in_device(|dev| {
        let programs = loaded_moves(na, nb, mover, case, format, with_format, true);
        let got = run_on(dev, &programs, &[(STAGE_A, &sa), (STAGE_B, &sb)]);
        equal(&got, &want, &format!("{mover:?} {label}"));
    });
}

#[cfg(feature = "silicon")]
fn check_case(mover: Mover, case: Case, label: &str) {
    let format = fp32_format(SrcAFormat::Tf32, true);
    check_with(mover, &a_bits(), case, format, false, label);
}

fn case_eight_0_to_0() -> Case {
    Case {
        moves: vec![(eight(0, 0), 0)],
        table: AddrModTable::ZERO,
    }
}

fn case_eight_8_to_8() -> Case {
    Case {
        moves: vec![(eight(8, 8), 0)],
        table: AddrModTable::ZERO,
    }
}

fn case_one_3_to_5() -> Case {
    Case {
        moves: vec![(one(3, 5), 0)],
        table: AddrModTable::ZERO,
    }
}

/// A single move naming entry 1: its own write must be unaffected by the entry.
fn case_single_move_naming_entry_1() -> Case {
    Case {
        moves: vec![(one(2, 4), 1)],
        table: entry1(1, 1),
    }
}

/// Two one-row moves; the first names entry 1, which advances `SrcA` by `src`
/// and `Dst` by `dst`, so the second (entry 0) lands there.
fn case_two_rows_entry_1(src: u32, dst: u32) -> Case {
    Case {
        moves: vec![(one(2, 4), 1), (one(2, 4), 0)],
        table: entry1(src, dst),
    }
}

/// The two eight-row moves ttsim can run as `MOVA2D`: entry 1 advances both by 8.
fn case_two_blocks_entry_1() -> Case {
    Case {
        moves: vec![(eight(0, 0), 1), (eight(0, 0), 0)],
        table: entry1(8, 8),
    }
}

// ------------------------------------------------------------ host tests

/// The model equals the IEEE bit formula for every special value, in every mode.
#[test]
fn model_matches_ieee_bit_formulas_for_special_values_and_every_mode() {
    let specials = [
        0u32,
        0x8000_0000,
        1,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2000,
        0xffc1_2000,
        0x3f80_1fff,
        0xbfff_ffff,
    ];
    for flush in [true, false] {
        let mut bank = [[0u32; 16]; 64];
        for (i, &b) in specials.iter().enumerate() {
            bank[0][i] = model::tf32_src_datum(b);
        }
        let state = |format| model::State {
            format,
            src_rwc: 0,
            dst_base: 0,
            block_columns: [0; 8],
        };
        // Fp32 Dst: forced TF32 whatever the override says.
        for src in [
            SrcAFormat::Tf32,
            SrcAFormat::Bf16,
            SrcAFormat::Fp16,
            SrcAFormat::Fp32,
        ] {
            let mut got = vec![];
            model::move_rows(
                &bank,
                state(fp32_format(src, flush)),
                0,
                0,
                false,
                false,
                |w| got.push(w),
            )
            .unwrap();
            for (i, &b) in specials.iter().enumerate() {
                let model::Write::Dst32 { value, .. } = got[i] else {
                    panic!("{:?}", got[i])
                };
                assert_eq!(
                    model::dst32_to_ieee(value),
                    direct(b, flush),
                    "Fp32 {src:?} flush {flush}: {b:08x}"
                );
            }
        }
        // Dst16 BF16: sign, 7 mantissa bits, 8 exponent bits, built from IEEE fields.
        let bf16 = DebugFormat::new(DebugMode::Dst16, SrcAFormat::Bf16, flush).unwrap();
        let mut got = vec![];
        model::move_rows(&bank, state(bf16), 0, 0, false, false, |w| got.push(w)).unwrap();
        for (i, &b) in specials.iter().enumerate() {
            let model::Write::Dst16 { value, .. } = got[i] else {
                panic!("{:?}", got[i])
            };
            let flushed = flush && (b >> 23) & 0xff == 0;
            let want = if flushed {
                0
            } else {
                (((b >> 31) << 15) | (((b >> 16) & 0x7f) << 8) | ((b >> 23) & 0xff)) as u16
            };
            assert_eq!(value, want, "BF16 flush {flush}: {b:08x}");
        }
    }
}

/// The two predictions of the cache experiment are exact and separable, and the
/// classifier names each.
#[test]
fn cache_experiment_predictions_are_exact_and_separable() {
    let (m, b0) = cache_operands();
    let first = mvmul_reference(&[[0f32; 16]; 8], &b0, &m, &[0]).unwrap();
    let b1: MatB = std::array::from_fn(|r| m[8 + r]);
    let fresh = mvmul_reference(&first, &b1, &m, &[0]).unwrap();
    let stale = mvmul_reference(&first, &b0, &m, &[0]).unwrap();
    assert_ne!(fresh, stale, "the rewritten SrcB must change the product");
    assert_ne!(b0, b1);
    let dump = |top: &[[f32; 16]; 8]| -> Vec<u32> {
        let mut d = vec![0u32; 16 * ROW];
        for r in 0..8 {
            for c in 0..16 {
                d[r * ROW + c] = top[r][c].to_bits();
                d[(8 + r) * ROW + c] = m[8 + r][c].to_bits();
            }
        }
        d
    };
    assert_eq!(classify(&dump(&fresh)), CacheOutcome::Fresh);
    assert_eq!(classify(&dump(&stale)), CacheOutcome::Stale);
    assert_eq!(classify(&dump(&first)), CacheOutcome::Neither);
    let mut wrong_stage = dump(&fresh);
    wrong_stage[8 * ROW] ^= 1;
    assert_eq!(classify(&wrong_stage), CacheOutcome::Neither);
    assert_eq!(
        matrix_debug::cache_verdict(CacheOutcome::Fresh, CacheOutcome::Fresh),
        CacheVerdict::NoObservableOracle
    );
}

/// Small integers (exact in TF32 and under the phase-0 fidelity model) for `SrcA`
/// `M` and the first `SrcB`.
fn cache_operands() -> (MatA, MatB) {
    let m: MatA =
        std::array::from_fn(|r| std::array::from_fn(|c| ((r * 7 + c * 3) % 11) as f32 - 5.0));
    let b0: MatB =
        std::array::from_fn(|r| std::array::from_fn(|c| ((r * 5 + c * 2 + 1) % 9) as f32 - 4.0));
    (m, b0)
}

fn classify(dst: &[u32]) -> CacheOutcome {
    let (m, b0) = cache_operands();
    let first = mvmul_reference(&[[0f32; 16]; 8], &b0, &m, &[0]).unwrap();
    let b1: MatB = std::array::from_fn(|r| m[8 + r]);
    let fresh = mvmul_reference(&first, &b1, &m, &[0]).unwrap();
    let stale = mvmul_reference(&first, &b0, &m, &[0]).unwrap();
    let rows = |want: &[[f32; 16]; 8]| {
        (0..8).all(|r| (0..16).all(|c| dst[r * ROW + c] == want[r][c].to_bits()))
    };
    let staged = (0..8).all(|r| (0..16).all(|c| dst[(8 + r) * ROW + c] == m[8 + r][c].to_bits()));
    match (staged, rows(&fresh), rows(&stale)) {
        (true, true, false) => CacheOutcome::Fresh,
        (true, false, true) => CacheOutcome::Stale,
        _ => CacheOutcome::Neither,
    }
}

fn cache_arm(dev: &mut harness::Dev<'_>, arm: CacheArm) -> Vec<u32> {
    let (m, b0) = cache_operands();
    let (sa, na) = stage_operand(SRC_A_ROW, &m);
    let (sb, nb) = stage_operand(SRC_B_ROW, &b0);
    let programs = loaded_program(
        na,
        nb,
        Box::new(move |banks, p| {
            p.extend_from_slice(
                matrix_debug::src_b_cache_probe(banks, arm)
                    .unwrap()
                    .as_slice(),
            )
        }),
    );
    run_on(dev, &programs, &[(STAGE_A, &sa), (STAGE_B, &sb)])
}

// ---------------------------------------------------------- simulator gates

/// `MOVA2D` eight-row forms run on ttsim; their `Dst` is the model's, which is
/// `MOVDBGA2D`'s. The negative control moves a different source row.
#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_mova2d_control_matches_the_debug_model() {
    let bits = a_bits();
    let (sa, na) = stage_a(&bits);
    let (sb, nb) = stage_b();
    let format = fp32_format(SrcAFormat::Tf32, true);
    harness::in_device(|dev| {
        // ttsim implements the eight-row `MOVA2D` only (divergence row 37). The
        // last case runs the explicit address-modifier table and advances both
        // counters by eight between the two moves.
        for (label, case) in [
            ("eight rows 0 -> 0", case_eight_0_to_0()),
            ("eight rows 8 -> 8", case_eight_8_to_8()),
            (
                "two blocks, entry 1 advancing both by 8",
                case_two_blocks_entry_1(),
            ),
        ] {
            let want = expected(&bits, &case, format);
            let programs = loaded_moves(na, nb, Mover::Mova2d, case, format, false, true);
            let got = run_on(dev, &programs, &[(STAGE_A, &sa), (STAGE_B, &sb)]);
            equal(&got, &want, label);
        }
        // Negative control: the expectation of another source row must not match.
        let programs = loaded_moves(
            na,
            nb,
            Mover::Mova2d,
            case_eight_8_to_8(),
            format,
            false,
            true,
        );
        let got = run_on(dev, &programs, &[(STAGE_A, &sa), (STAGE_B, &sb)]);
        let wrong_row = Case {
            moves: vec![(eight(0, 8), 0)],
            table: AddrModTable::ZERO,
        };
        assert_ne!(
            got,
            expected(&bits, &wrong_row, format),
            "a mis-targeted expectation must differ"
        );
        // And the same sequence with the entry not advancing differs from the
        // advancing expectation (the table is doing the work).
        let flat = Case {
            moves: vec![(eight(0, 0), 1), (eight(0, 0), 0)],
            table: entry1(0, 0),
        };
        let programs = loaded_moves(na, nb, Mover::Mova2d, flat, format, false, true);
        let got = run_on(dev, &programs, &[(STAGE_A, &sa), (STAGE_B, &sb)]);
        assert_ne!(got, expected(&bits, &case_two_blocks_entry_1(), format));
    });
}

/// ttsim refuses every `MOVDBGA2D` form this module builds (divergence row 83),
/// while an eight-row `MOVA2D` of the same shape survives each time.
#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_refuses_every_movdbga2d_form_with_surviving_controls() {
    let bits = a_bits();
    let (sa, na) = stage_a(&bits);
    let (sb, nb) = stage_b();
    let format = fp32_format(SrcAFormat::Tf32, true);
    let tries = |mover: Mover, rows: Rows, with_format: bool, format: DebugFormat| {
        let case = Case {
            moves: vec![(rows, 0)],
            table: AddrModTable::ZERO,
        };
        let programs = loaded_moves(na, nb, mover, case, format, with_format, false);
        harness::survives(|dev| {
            run_on(dev, &programs, &[(STAGE_A, &sa), (STAGE_B, &sb)]);
        })
    };
    let eight = Rows::Eight {
        src_row: 0,
        dst_row: 0,
    };
    let one = Rows::One {
        src_row: 3,
        dst_row: 5,
    };
    // Surviving controls: the same shape with MOVA2D, with and without format setup.
    assert!(tries(Mover::Mova2d, eight, false, format));
    assert!(tries(Mover::Mova2d, eight, true, format));
    // The debug forms.
    assert!(!tries(Mover::Movdbga2d, eight, false, format));
    assert!(!tries(Mover::Movdbga2d, one, false, format));
    for src in [SrcAFormat::Tf32, SrcAFormat::Bf16, SrcAFormat::Fp16] {
        assert!(!tries(
            Mover::Movdbga2d,
            eight,
            true,
            fp32_format(src, true)
        ));
    }
    // The unpacker-owned (Filling) form; the control is the same program with
    // only the unpacker drain, which survives.
    let mv = DebugMove::new(
        Rows::Eight {
            src_row: 0,
            dst_row: 0,
        },
        0,
        format,
    )
    .unwrap();
    let debug = filling_program(
        na,
        nb,
        Box::new(move |banks, p| {
            let (words, _) = banks.debug_move_a(mv).unwrap();
            p.extend(words);
        }),
    );
    let control = filling_program(
        na,
        nb,
        Box::new(|_banks, p| p.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap())),
    );
    assert!(harness::survives(|dev| {
        run_on(dev, &control, &[(STAGE_A, &sa), (STAGE_B, &sb)]);
    }));
    assert!(!harness::survives(|dev| {
        run_on(dev, &debug, &[(STAGE_A, &sa), (STAGE_B, &sb)]);
    }));
}

/// What ttsim does with `GATESRCRST`, and the experiment's arms on the
/// simulator, whose `SrcB` has no operand cache to go stale.
#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_runs_the_cache_experiment_arms() {
    harness::in_device(|dev| {
        let outcome = |dev: &mut harness::Dev<'_>, arm| classify(&cache_arm(dev, arm));
        // The detector control: no rewrite is the stale prediction.
        assert_eq!(outcome(dev, CacheArm::NoRewrite), CacheOutcome::Stale);
        // A rewrite through MOVD2B reaches the next MVMUL on the simulator.
        assert_eq!(outcome(dev, CacheArm::NoGate), CacheOutcome::Fresh);
    });
}

/// `GATESRCRST` on ttsim: see the divergence row 84 text in the close-out record.
#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_gatesrcrst_probe() {
    let (m, b0) = cache_operands();
    let (sa, na) = stage_operand(SRC_A_ROW, &m);
    let (sb, nb) = stage_operand(SRC_B_ROW, &b0);
    let with = |arm| {
        loaded_program(
            na,
            nb,
            Box::new(move |banks, p| {
                p.extend_from_slice(
                    matrix_debug::src_b_cache_probe(banks, arm)
                        .unwrap()
                        .as_slice(),
                )
            }),
        )
    };
    for arm in [CacheArm::GateOperandClear, CacheArm::GateInvalidate] {
        let programs = with(arm);
        let survived = harness::survives(|dev| {
            run_on(dev, &programs, &[(STAGE_A, &sa), (STAGE_B, &sb)]);
        });
        eprintln!("ttsim GATESRCRST arm {arm:?} survives: {survived}");
        assert!(survived, "{arm:?}");
        harness::in_device(|dev| {
            assert_eq!(
                classify(&cache_arm(dev, arm)),
                CacheOutcome::Fresh,
                "{arm:?}"
            );
        });
    }
}

// ------------------------------------------------------------ silicon gates

/// Isolated minimal probe, first: one eight-row `MOVDBGA2D` over loaded `SrcA`.
/// Its `AddrMod` position is measured (step9); every other field is the Wormhole
/// page's. Risk class: measured encoding, unmeasured semantics.
#[test]
#[cfg(feature = "silicon")]
fn probe_movdbga2d_minimal_survives() {
    harness::assert_on_silicon();
    let bits = a_bits();
    let (sa, na) = stage_a(&bits);
    let (sb, nb) = stage_b();
    let format = fp32_format(SrcAFormat::Tf32, true);
    let programs = loaded_moves(
        na,
        nb,
        Mover::Movdbga2d,
        case_eight_0_to_0(),
        format,
        false,
        false,
    );
    assert!(harness::survives(|dev| {
        run_on(dev, &programs, &[(STAGE_A, &sa), (STAGE_B, &sb)]);
    }));
}

/// Isolated minimal probe, `UNVERIFIED`: a lone `GATESRCRST` (operand set), after
/// loaded banks, then a drain. Encoding is the Wormhole page's; Blackhole has none.
#[test]
#[cfg(feature = "silicon")]
fn probe_gatesrcrst_minimal_survives() {
    harness::assert_on_silicon();
    let (m, b0) = cache_operands();
    let (sa, na) = stage_operand(SRC_A_ROW, &m);
    let (sb, nb) = stage_operand(SRC_B_ROW, &b0);
    let programs = loaded_program(
        na,
        nb,
        Box::new(|banks, p| {
            let (i, banks) = banks.gatesrcrst().unwrap();
            p.push(i);
            p.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
            let (i, banks) = banks.release_a().unwrap();
            p.push(i);
            let (i, _) = banks.release_b().unwrap();
            p.push(i);
        }),
    );
    assert!(harness::survives(|dev| {
        run_on(dev, &programs, &[(STAGE_A, &sa), (STAGE_B, &sb)]);
    }));
}

/// One silicon test per case below, so each reports its own result. Each runs
/// `MOVDBGA2D` through the checked helper over loaded `SrcA`, under an address-modifier
/// table written in full, and compares all sixteen dumped `Dst` rows with the model.
macro_rules! debug_case_open {
    ($name:ident, $case:expr) => {
        #[test]
        #[cfg(feature = "silicon")]
        fn $name() {
            harness::assert_on_silicon();
            check_case(Mover::Movdbga2d, $case, stringify!($name));
        }
    };
}

macro_rules! debug_case {
    ($name:ident, $case:expr) => {
        #[test]
        #[cfg(feature = "silicon")]
        fn $name() {
            harness::assert_on_silicon();
            check_case(Mover::Movdbga2d, $case, stringify!($name));
        }
    };
}

debug_case!(movdbga2d_eight_rows_0_to_0, case_eight_0_to_0());
debug_case!(movdbga2d_eight_rows_8_to_8, case_eight_8_to_8());
debug_case!(movdbga2d_one_row_3_to_5, case_one_3_to_5());
// The AddrMod ladder: a single move naming entry 1, then two moves with the
// entry advancing both counters, `Dst` only, and `SrcA` only.
debug_case_open!(
    movdbga2d_addr_mod_single_move_names_entry_1,
    case_single_move_naming_entry_1()
);
debug_case_open!(
    movdbga2d_addr_mod_1_two_rows_src_and_dst_advance,
    case_two_rows_entry_1(1, 1)
);
debug_case_open!(
    movdbga2d_addr_mod_1_two_rows_dst_advance_only,
    case_two_rows_entry_1(0, 1)
);
debug_case_open!(
    movdbga2d_addr_mod_1_two_rows_src_advance_only,
    case_two_rows_entry_1(1, 0)
);
debug_case!(
    movdbga2d_addr_mod_1_two_blocks_advance_by_8,
    case_two_blocks_entry_1()
);

/// The same two-row sequence with entries 4..8 copying entries 0..4. If
/// `movdbga2d_addr_mod_1_two_rows_src_and_dst_advance` fails and this passes, the
/// index is being shifted up by four (the Wormhole page's `ExtraAddrModBit`), and
/// the entry the instruction reaches is not entry 1.
#[test]
#[cfg(feature = "silicon")]
fn movdbga2d_addr_mod_1_two_rows_with_mirrored_upper_entries() {
    harness::assert_on_silicon();
    let mut case = case_two_rows_entry_1(1, 1);
    case.table = case.table.mirrored();
    check_case(Mover::Movdbga2d, case, "mirrored upper entries");
}

/// `MOVA2D` controls for the two cases that tell an `AddrMod` problem common to
/// both movers from one specific to `MOVDBGA2D`.
#[test]
#[cfg(feature = "silicon")]
fn mova2d_control_one_row_3_to_5() {
    harness::assert_on_silicon();
    check_case(Mover::Mova2d, case_one_3_to_5(), "control one row 3 -> 5");
}

#[test]
#[cfg(feature = "silicon")]
fn mova2d_control_addr_mod_1_two_rows_src_and_dst_advance() {
    harness::assert_on_silicon();
    check_case(
        Mover::Mova2d,
        case_two_rows_entry_1(1, 1),
        "control two rows, entry 1 advancing both",
    );
}

/// Format selection: with Fp32 accumulation the override value does not select the
/// format (TF32 is forced), so every value gives the same `Dst`.
fn format_override(src: SrcAFormat) {
    harness::assert_on_silicon();
    let bits = a_bits();
    let case = case_eight_8_to_8();
    let format = fp32_format(src, true);
    check_with(
        Mover::Movdbga2d,
        &bits,
        case,
        format,
        true,
        &format!("override {src:?}"),
    );
}

#[test]
#[cfg(feature = "silicon")]
fn movdbga2d_format_override_tf32() {
    format_override(SrcAFormat::Tf32);
}

#[test]
#[cfg(feature = "silicon")]
fn movdbga2d_format_override_bf16_is_forced_to_tf32() {
    format_override(SrcAFormat::Bf16);
}

#[test]
#[cfg(feature = "silicon")]
fn movdbga2d_format_override_fp16_is_forced_to_tf32() {
    format_override(SrcAFormat::Fp16);
}

#[test]
#[cfg(feature = "silicon")]
fn movdbga2d_format_override_fp32_is_forced_to_tf32() {
    format_override(SrcAFormat::Fp32);
}

/// Denormal flush off: normal data only, so the unpacker's own handling of subnormals
/// and signed zero (not what this arm measures) cannot differ.
#[test]
#[cfg(feature = "silicon")]
fn movdbga2d_denormal_flush_off_on_normal_data() {
    harness::assert_on_silicon();
    let normal: Vec<u32> = a_bits()
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            if i < 16 {
                0x3f80_0000 | ((i as u32) << 14)
            } else {
                b
            }
        })
        .collect();
    let keep = fp32_format(SrcAFormat::Tf32, false);
    check_with(
        Mover::Movdbga2d,
        &normal,
        case_eight_8_to_8(),
        keep,
        true,
        "flush off",
    );
}

/// Data mutants: the expectation of a different source row, and of an `AddrMod`-0
/// sequence (which does not advance), must each differ from what the device does.
#[test]
#[cfg(feature = "silicon")]
fn movdbga2d_data_mutants_differ() {
    harness::assert_on_silicon();
    let bits = a_bits();
    let (sa, na) = stage_a(&bits);
    let (sb, nb) = stage_b();
    let stage = [(STAGE_A, sa.as_slice()), (STAGE_B, sb.as_slice())];
    let format = fp32_format(SrcAFormat::Tf32, true);
    harness::in_device(|dev| {
        let programs = loaded_moves(
            na,
            nb,
            Mover::Movdbga2d,
            case_eight_8_to_8(),
            format,
            false,
            true,
        );
        let wrong_row = Case {
            moves: vec![(eight(0, 8), 0)],
            table: AddrModTable::ZERO,
        };
        assert_ne!(
            run_on(dev, &programs, &stage),
            expected(&bits, &wrong_row, format),
            "source-row mutant"
        );
        let no_advance = Case {
            moves: vec![(one(2, 4), 0), (one(2, 4), 0)],
            table: entry1(1, 1),
        };
        let programs = loaded_moves(na, nb, Mover::Movdbga2d, no_advance, format, false, true);
        assert_ne!(
            run_on(dev, &programs, &stage),
            expected(&bits, &case_two_rows_entry_1(1, 1), format),
            "AddrMod-0 mutant must not advance"
        );
    });
}

/// The diagnostic's reason to exist: `MOVDBGA2D` reads a `SrcA` bank the
/// unpacker still owns (an `UNPACR` without `FlipSrc`), after the helper's
/// explicit unpacker drain. `MOVA2D` would wait on the gate forever, so no
/// `MOVA2D` control is run on this arm.
#[test]
#[cfg(feature = "silicon")]
fn movdbga2d_reads_a_bank_the_unpacker_still_owns() {
    harness::assert_on_silicon();
    let bits = a_bits();
    let (sa, na) = stage_a(&bits);
    let (sb, nb) = stage_b();
    let format = fp32_format(SrcAFormat::Tf32, true);
    harness::in_device(|dev| {
        let programs = filling_program(
            na,
            nb,
            Box::new(move |banks, p| {
                set_format(p, format);
                let mv = DebugMove::new(
                    Rows::Eight {
                        src_row: 0,
                        dst_row: 0,
                    },
                    0,
                    format,
                )
                .unwrap();
                let (words, _) = banks.debug_move_a(mv).unwrap();
                p.extend(words);
            }),
        );
        let got = run_on(dev, &programs, &[(STAGE_A, &sa), (STAGE_B, &sb)]);
        equal(
            &got,
            &expected(&bits, &case_eight_0_to_0(), format),
            "staged rows read through MOVDBGA2D",
        );
    });
}

/// The `GATESRCRST` experiment. Decision rule fixed in `matrix_debug::cache_verdict`:
/// `NoGate` stale and `GateInvalidate` fresh is an observable effect; both fresh is
/// `[-]` "no observable oracle". The control arms must hold either way: the
/// operand-clear form behaves as no gate, and no rewrite is the stale prediction.
#[test]
#[cfg(feature = "silicon")]
fn gatesrcrst_stale_versus_fresh_experiment() {
    harness::assert_on_silicon();
    harness::in_device(|dev| {
        let no_rewrite = classify(&cache_arm(dev, CacheArm::NoRewrite));
        assert_eq!(no_rewrite, CacheOutcome::Stale, "detector control");
        let no_gate = classify(&cache_arm(dev, CacheArm::NoGate));
        let clear = classify(&cache_arm(dev, CacheArm::GateOperandClear));
        let gate = classify(&cache_arm(dev, CacheArm::GateInvalidate));
        let verdict = matrix_debug::cache_verdict(no_gate, gate);
        eprintln!(
            "GATESRCRST experiment: no_gate {no_gate:?}, operand_clear {clear:?}, \
             gate {gate:?}, no_rewrite {no_rewrite:?} => {verdict:?}"
        );
        assert_ne!(no_gate, CacheOutcome::Neither, "the experiment is broken");
        assert_eq!(
            clear, no_gate,
            "the operand-clear form must behave as no gate"
        );
        assert_ne!(
            verdict,
            CacheVerdict::Inconclusive,
            "inconclusive: GATESRCRST changed the result in a way the model does not name"
        );
    });
}
