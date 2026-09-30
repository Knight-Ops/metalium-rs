//! The `SrcA`/`SrcB` path: what ttsim will let us configure, and what it does.
//!
//! Phase 6 needs the unpackers to write `SrcA` and `SrcB` rather than `Dst`, which
//! means unpacker 1 for the first time, the ALU format fields, and the `ThreadConfig`
//! address modifiers `MVMUL` advances the RWCs with. `probe_config_coverage.rs`
//! answered "which registers survive a zero write" (divergence row 28), which is not
//! enough here: rows 32 and 33 are refusals that depend on the *field* and on the
//! *value*, on registers whose zero write is accepted. So this survey is per field,
//! and writes each one twice -- zero, and a non-zero value -- with every other bit of
//! the word left zero.
//!
//! Run with:
//! `cargo test -p tt-tests --test probe_src -- --ignored --nocapture`

// The surveys and refusal probes are simulator-only, so their helpers are dead in
// the silicon build.
#![cfg_attr(feature = "silicon", allow(dead_code, unused_imports))]

use tt_isa::backend;
use tt_isa::cfg::generated::{ALL_CONFIG_FIELDS, ALL_THREAD_CONFIG_FIELDS};
use tt_isa::isa::Instruction;
use tt_isa::sfpu;
use tt_tests::harness::{self, survives, Run};

const SCRATCH_GPR: u32 = 8;

/// `Config` fields the `Src` path and `MVMUL` read.
const CONFIG_PREFIXES: &[&str] = &[
    "THCON_SEC1_REG0_",
    "THCON_SEC1_REG2_",
    "THCON_SEC1_REG3_",
    "THCON_SEC1_REG5_",
    "THCON_SEC1_REG7_",
    "UNP1_ADDR_",
    "ALU_FORMAT_SPEC_",
    "ALU_ACC_CTRL_",
    "DEST_REGW_BASE_",
    // Unpacker 0 fields the `Dst` path never needed but the `Src` path reads.
    "THCON_SEC0_REG2_Unpack_Src_Reg_Set_Upd",
    "THCON_SEC0_REG2_Unpack_if_sel",
];

/// `ThreadConfig` fields the `Src` path and `MVMUL` read.
const THREAD_PREFIXES: &[&str] = &[
    "ADDR_MOD_AB_SEC",
    "ADDR_MOD_DST_SEC",
    "SRCA_SET_",
    "SRCB_SET_",
    "DISABLE_IMPLIED_SRC",
    "CLR_DVALID_",
    "FIDELITY_BASE_",
    "DEST_TARGET_REG_CFG_MATH_",
];

/// A tail that computes, so a surviving run has also done work after the write.
fn tail(program: &mut Vec<Instruction>) {
    program.push(backend::nop());
    program.extend(sfpu::load_f32(0, 1.5f32.to_bits()).unwrap());
    program.push(sfpu::store(0, sfpu::store_format::FP32, 0, 0).unwrap());
}

fn write_config(dev: &mut harness::Dev<'_>, addr32: u16, word: u32) {
    let mut program = Vec::new();
    program.extend(backend::set_gpr(SCRATCH_GPR, word).unwrap());
    program.push(backend::write_word(SCRATCH_GPR, addr32).unwrap());
    tail(&mut program);
    harness::run(dev, &Run::new(&program));
}

fn write_thread(dev: &mut harness::Dev<'_>, addr32: u16, entry: u16) {
    let mut program = vec![backend::set_thread_entry(addr32, entry).unwrap()];
    tail(&mut program);
    harness::run(dev, &Run::new(&program));
}

fn verdict(zero: bool, one: bool) -> &'static str {
    match (zero, one) {
        (true, true) => "ok",
        (true, false) => "ZERO ONLY",
        (false, true) => "NONZERO ONLY",
        (false, false) => "REFUSED",
    }
}

// Writes configuration fields blind at zero and one; simulator only.
#[cfg(not(feature = "silicon"))]
#[test]
#[ignore]
fn map_the_src_path_configuration_surface() {
    println!("\n=== Config fields (word = field alone; values 0 and 1) ===");
    for (name, f) in ALL_CONFIG_FIELDS {
        if !CONFIG_PREFIXES.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        if f.addr32() == backend::STATE_RESET_EN_ADDR32 {
            continue;
        }
        let zero = survives(|dev| write_config(dev, f.addr32(), f.insert(0, 0)));
        let one = survives(|dev| write_config(dev, f.addr32(), f.insert(0, 1)));
        println!("  {:3} {:<48} {}", f.addr32(), name, verdict(zero, one));
    }

    println!("\n=== ThreadConfig fields (entry = field alone; values 0 and 1) ===");
    for (name, f) in ALL_THREAD_CONFIG_FIELDS {
        if !THREAD_PREFIXES.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        let zero = survives(|dev| write_thread(dev, f.addr32(), f.insert(0, 0)));
        let one = survives(|dev| write_thread(dev, f.addr32(), f.insert(0, 1)));
        println!("  {:3} {:<48} {}", f.addr32(), name, verdict(zero, one));
    }
}

// ---------------------------------------------------------------------------
// Unpacking into `SrcA`/`SrcB`, observed through `MOVA2D`/`MOVB2D`.
//
// There is no RISC-V path into `SrcA`/`SrcB`, so every claim here goes through a
// Matrix Unit move into `Dst` and the corpus firmware's `Dst` dump. `MOVA2D` and
// `MOVB2D` are `TTArchitecture`-conditionalized pages, so the Blackhole behaviour
// they document is authoritative: with `ALU_ACC_CTRL_Fp32_enabled` set they
// "pretend that SrcAFmt is TF32" and write the 19-bit `Src` datum out as the FP32
// bit pattern of the TF32 value. That is what makes a `Src` datum readable at all.
// ---------------------------------------------------------------------------

use tt_isa::backend::ConfigWords;
use tt_isa::cfg::generated::alu;
use tt_isa::isa::generated::encode;
use tt_isa::tile::{bf16_to_fp32, fp32_to_bf16_truncate, fp32_to_tf32, L1Format, TileImage};
use tt_tests::datapath::{
    flat_descriptor, set_adc_x, src_thread_config, unpack_src_config, unpack_src_instruction,
    Unpacker, SCRATCH_GPR as GPR, STAGE,
};

/// Datums per `Src` row.
const ROW: usize = 16;

/// Datums staged by the gates: two `Src` rows.
const N: usize = 32;

const FP32_CODE: u32 = 0;
const TF32_CODE: u32 = 4;
const BF16_CODE: u32 = 5;

/// FP32 operands with mantissa bits on both sides of the TF32 cut (bit 13) and of
/// the BF16 cut (bit 16), so either truncation is observed rather than assumed.
fn operand_bits() -> Vec<u32> {
    (0..N)
        .map(|i| (1.0f32 + i as f32).to_bits() | 0x3FFF)
        .collect()
}

fn stage(format: L1Format, code: u32, datums: &[u32]) -> Vec<u8> {
    let descriptor = flat_descriptor(datums.len() as u32).with_in_data_format_raw(code);
    let image = TileImage::new(descriptor, format).unwrap();
    let bytes = (format.datum_bits() / 8) as usize;
    let mut staged = vec![0u8; image.total_bytes()];
    for (i, d) in datums.iter().enumerate() {
        let off = image.datum_bit_offset(i) / 8;
        staged[off..off + bytes].copy_from_slice(&d.to_le_bytes()[..bytes]);
    }
    staged
}

/// Configure, unpack `N` datums into `unpacker`'s `Src`, hand the bank to the
/// Matrix Unit, and move `Src` rows 0..8 into `Dst` rows 0..8 -- split the way
/// LLK splits it (`harness::Roles`): the unpack on thread 0, the move on thread 1.
fn src_program(unpacker: Unpacker, in_code: u32, out: u32, flip: bool) -> SrcProgram {
    let mut unpack = src_thread_config();
    let mut words = ConfigWords::new();
    let descriptor = flat_descriptor(N as u32).with_in_data_format_raw(in_code);
    unpack_src_config(&mut words, unpacker, descriptor, STAGE, out);
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    let mut buf = vec![sfpu::nop(); words.program_len()];
    let k = words.program(GPR, &mut buf).unwrap();
    unpack.extend_from_slice(&buf[..k]);
    unpack.push(set_adc_x(unpacker, 0, N as u32 - 1));
    unpack.push(unpack_src_instruction(unpacker, flip));
    unpack.push(match unpacker {
        Unpacker::SrcA => backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap(),
        Unpacker::SrcB => backend::wait_for_unpacker1(backend::Before::EVERYTHING).unwrap(),
    });

    // The Matrix Unit waits for the bank by itself (`MOVA2D.md`: the Wait Gate
    // holds it until `AllowedClient == MatrixUnit`).
    let mut math = vec![tt_tests::datapath::state_id()];
    match unpacker {
        Unpacker::SrcA => {
            // ttsim implements only the eight-row form (`tensix_mova2d:
            // instr_mod=0` is `UnsupportedFunctionality`).
            math.push(
                encode::Mova2D::ZERO
                    .move8_rows(1)
                    .src_row(0)
                    .dst_row(0)
                    .encode()
                    .unwrap(),
            );
        }
        Unpacker::SrcB => {
            for r in 0..8 {
                math.push(encode::Movb2D::ZERO.src_row(r).dst_row(r).encode().unwrap());
            }
        }
    }
    math.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
    SrcProgram { unpack, math }
}

/// The two role programs of a `Src` probe.
struct SrcProgram {
    unpack: Vec<Instruction>,
    math: Vec<Instruction>,
}

/// Run and return `Dst` rows 0..8, flattened.
fn run_src(dev: &mut harness::Dev<'_>, staged: &[u8], program: &SrcProgram) -> Vec<u32> {
    let roles = harness::Roles {
        unpack: &program.unpack,
        math: &program.math,
        pack: &[],
    };
    let out = harness::run(
        dev,
        &Run::roles(roles).stage(&[(STAGE, staged)]).dump_rows(8),
    );
    (0..8 * ROW).map(|f| out.dst_at(f / ROW, f % ROW)).collect()
}

/// Assert that every staged datum `i` landed at flat `Src` position `i`, as
/// `expect(i)`.
///
/// There was once a leading offset here -- four datums for a 4-byte input, eight
/// for a 2-byte one -- with as many dropped from the end, recorded as divergence
/// row 35. It was the unpacker reading our tile header as datums, because
/// `REG3_Base_address` pointed one unit early (`datapath::tile_base_units`).
fn assert_landed(dst: &[u32], expect: impl Fn(usize) -> u32) {
    for (i, &got) in dst.iter().enumerate().take(N) {
        assert_eq!(
            got,
            expect(i),
            "staged datum {i} should be at Src flat {i} (row {}, col {}); got {got:08x}",
            i / ROW,
            i % ROW,
        );
    }
}

#[test]
fn fp32_unpacks_into_srca_as_truncated_tf32() {
    let bits = operand_bits();
    let staged = stage(L1Format::Fp32, FP32_CODE, &bits);
    let program = src_program(Unpacker::SrcA, FP32_CODE, TF32_CODE, true);
    harness::in_device(|dev| {
        let dst = run_src(dev, &staged, &program);
        assert_landed(&dst, |i| fp32_to_tf32(bits[i]));
        // And the truncation is real: the staged bits differ from what landed.
        assert_ne!(dst[0], bits[0]);
    });
}

#[test]
fn fp32_unpacks_into_srcb_as_truncated_tf32() {
    let bits = operand_bits();
    let staged = stage(L1Format::Fp32, FP32_CODE, &bits);
    let program = src_program(Unpacker::SrcB, FP32_CODE, TF32_CODE, true);
    harness::in_device(|dev| {
        let dst = run_src(dev, &staged, &program);
        assert_landed(&dst, |i| fp32_to_tf32(bits[i]));
    });
}

/// The BF16 route the `Dst` path lacks (row 31): FP32 in L1, BF16 in `Src`, by
/// truncation (`UNPACR_Regular.md:509-513`).
#[test]
fn fp32_unpacks_into_src_as_truncated_bf16() {
    let bits = operand_bits();
    let staged = stage(L1Format::Fp32, FP32_CODE, &bits);
    for unpacker in [Unpacker::SrcA, Unpacker::SrcB] {
        let program = src_program(unpacker, FP32_CODE, BF16_CODE, true);
        harness::in_device(|dev| {
            let dst = run_src(dev, &staged, &program);
            assert_landed(&dst, |i| bf16_to_fp32(fp32_to_bf16_truncate(bits[i])));
        });
    }
}

/// BF16 staged in L1 reaches `Src` unchanged: the first 16-bit *input* format any
/// path has accepted.
#[test]
fn bf16_in_l1_unpacks_into_src_unchanged() {
    // BF16 1 + i with the lowest mantissa bit set, so a lost bit would show.
    let halves: Vec<u32> = (0..N)
        .map(|i| u32::from(fp32_to_bf16_truncate((1.0f32 + i as f32).to_bits()) | 1))
        .collect();
    let staged = stage(L1Format::Bf16, BF16_CODE, &halves);
    for unpacker in [Unpacker::SrcA, Unpacker::SrcB] {
        let program = src_program(unpacker, BF16_CODE, BF16_CODE, true);
        harness::in_device(|dev| {
            let dst = run_src(dev, &staged, &program);
            assert_landed(&dst, |i| bf16_to_fp32(halves[i] as u16));
        });
    }
}

/// `UNPACR_Regular.md:580`: FP32 into `Src` is `UndefinedBehavior`, because a
/// 19-bit slot cannot hold it. ttsim refuses it; the control with `TF32` out and
/// everything else identical survives, so the refusal is about the format.
// `UndefinedBehavior` per `UNPACR_Regular.md:580`; never executed on silicon.
#[cfg(not(feature = "silicon"))]
#[test]
fn fp32_into_src_is_refused() {
    let staged = stage(L1Format::Fp32, FP32_CODE, &operand_bits());
    for unpacker in [Unpacker::SrcA, Unpacker::SrcB] {
        let refused = src_program(unpacker, FP32_CODE, 0, true);
        let control = src_program(unpacker, FP32_CODE, TF32_CODE, true);
        assert!(!survives(|dev| {
            run_src(dev, &staged, &refused);
        }));
        assert!(survives(|dev| {
            run_src(dev, &staged, &control);
        }));
    }
}

/// Negative control: corrupting one staged datum moves exactly one landed element,
/// and it is the one `assert_landed` puts it at.
///
/// Each run gets a fresh simulator. Two runs in one session read the *first*
/// run's data the second time: the first `UNPACR` flipped the unpacker onto bank
/// 1, `MOVA2D` does not flip the Matrix Unit's bank, so the second unpack writes
/// bank 1 while the move reads bank 0 again. That is the bank-ownership hazard
/// `tt_isa::matrix` exists to make unrepresentable, found here by accident.
#[test]
fn a_corrupted_datum_moves_exactly_one_src_element() {
    let bits = operand_bits();
    let mut corrupted = bits.clone();
    corrupted[7] ^= 0x0040_0000;
    let program = src_program(Unpacker::SrcA, FP32_CODE, TF32_CODE, true);
    let dump = |datums: &[u32]| {
        let staged = stage(L1Format::Fp32, FP32_CODE, datums);
        let path = std::env::temp_dir().join(format!(
            "ttsrc-corrupt-{}-{:08x}.bin",
            std::process::id(),
            datums[7]
        ));
        harness::in_device(|dev| {
            let dst = run_src(dev, &staged, &program);
            let bytes: Vec<u8> = dst.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, bytes).unwrap();
        });
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<u32>>()
    };
    let a = dump(&bits);
    let b = dump(&corrupted);
    let differ: Vec<usize> = (0..a.len()).filter(|&f| a[f] != b[f]).collect();
    assert_eq!(differ, vec![7]);
}

/// `MOVB2D.md` says `Move4Rows` moves four rows; ttsim moves one, silently (row
/// 38). The silicon gate asserts the documented four.
#[test]
#[cfg(feature = "silicon")]
fn on_silicon_movb2d_move4_rows_moves_four() {
    let bits = operand_bits();
    let staged = stage(L1Format::Fp32, FP32_CODE, &bits);
    let mut program = src_program(Unpacker::SrcB, FP32_CODE, TF32_CODE, true);
    // Replace the single-row moves with one four-row move of rows 0..4.
    program.math.retain(|i| i.def().mnemonic() != "MOVB2D");
    let wait = program.math.pop().unwrap();
    program.math.push(
        encode::Movb2D::ZERO
            .move4_rows(1)
            .src_row(0)
            .dst_row(0)
            .encode()
            .unwrap(),
    );
    program.math.push(wait);
    harness::in_device(|dev| {
        let dst = run_src(dev, &staged, &program);
        // Row 1 is fully written by the unpacker whatever the base turns out to
        // be, so it tells four rows from one.
        assert!(dst[ROW..2 * ROW].iter().all(|&v| v != 0));
    });
}
