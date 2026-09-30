//! Phase 6, first milestone: `MVMUL`, `Dst += SrcB @ SrcA`, one 8x16 by 16x16 block.
//!
//! # Two oracles, two claims -- as in step 8
//!
//! `tt_isa::numerics::mvmul_reference` ports the documented fidelity-phase model
//! (`MVMUL.md`, `MatrixUnit.md:143-165`) and says what each output datum is, bit
//! for bit. It refuses to answer unless every product and sum is exact, because
//! `MVMUL.md` calls its float model "a rough guide" to order and precision; the
//! operands here are chosen so it always answers. `burn-flex` says which datums
//! meet which -- that it is `SrcB @ SrcA` and not a transpose -- for small integers
//! where IEEE and the Matrix Unit cannot disagree. No epsilon anywhere.
//!
//! # Where the operands sit, and why
//!
//! Each operand is staged behind zero rows so that it lands at the row the RWCs
//! point `MVMUL` at. `SrcA` goes at row 16 rather than 8 because ttsim refuses
//! `src_a_row=8` (row 41), which `MVMUL.md`'s `& 0x38` allows.

use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::cfg::generated::{alu, thread};
use tt_isa::isa::generated::encode;
use tt_isa::isa::Instruction;
use tt_isa::matrix::{Banks, Loaded};
use tt_isa::numerics::mvmul_reference;
use tt_isa::sfpu;
use tt_isa::tile::{fp32_to_tf32, L1Format, TileImage};
use tt_tests::datapath::{
    self, config_program, flat_descriptor, pack_config, set_adc_x, src_thread_config, thread_entry,
    unpack_src_config, Unpacker, OUT, SCRATCH_GPR, STAGE,
};
use tt_tests::harness::{self, Run};

const ROW: usize = 16;
const SRC_A_ROW: usize = 16;
const SRC_B_ROW: usize = 8;
const TF32_CODE: u32 = 4;

const STAGE_A: u64 = STAGE;
const STAGE_B: u64 = STAGE + 0x2000;

type MatA = [[f32; 16]; 16];
type MatB = [[f32; 16]; 8];
type Body = Box<dyn FnOnce(Banks<Loaded, Loaded>, &mut Vec<Instruction>)>;

const ZERO_DST: MatB = [[0f32; 16]; 8];

/// Stage `rows` as a flat FP32 run that lands at `Src` row `src_row`, column 0.
fn stage_operand(src_row: usize, rows: &[[f32; 16]]) -> (Vec<u8>, u32) {
    let mut datums = vec![0u32; src_row * ROW];
    datums.extend(rows.iter().flatten().map(|v| v.to_bits()));
    let n = datums.len() as u32;
    let image = TileImage::new(flat_descriptor(n), L1Format::Fp32).unwrap();
    let mut staged = vec![0u8; image.total_bytes()];
    for (i, d) in datums.iter().enumerate() {
        let off = image.datum_bit_offset(i) / 8;
        staged[off..off + 4].copy_from_slice(&d.to_le_bytes());
    }
    (staged, n)
}

/// `value` placed in `i`'s `AddrMod` field, as its (measured) definition draws it.
/// Entry 4 is the third bit, which only the Blackhole layouts have.
fn addr_mod_bits(i: Instruction, value: u32) -> u32 {
    let f = i.def().field("AddrMod").unwrap();
    assert!(
        f.fits(value),
        "{}: AddrMod {value} does not fit",
        i.def().key()
    );
    let bits = f.place(value);
    if value == 4 {
        assert_eq!(bits, 1 << 16, "{}: the third AddrMod bit", i.def().key());
    }
    bits
}

fn fidelity_base(phase: u16) -> Instruction {
    thread_entry(thread::FIDELITY_BASE_Phase, phase)
}

/// Configure, stage both operands, zero `Dst`, point the RWCs at the operands,
/// then hand the loaded banks to `body` for the Matrix Unit work.
fn program(na: u32, nb: u32, body: Body) -> (Vec<Instruction>, Vec<Instruction>) {
    // Split the way LLK splits it (`harness::Roles`): the unpacks on thread 0,
    // everything the Matrix Unit does on thread 1. The address modifiers, the
    // fidelity base and the RWCs are the *math* thread's `ThreadConfig` and
    // counters, so they are set there.
    let mut up = src_thread_config();
    let mut words = ConfigWords::new();
    unpack_src_config(
        &mut words,
        Unpacker::SrcA,
        flat_descriptor(na),
        STAGE_A,
        TF32_CODE,
    );
    unpack_src_config(
        &mut words,
        Unpacker::SrcB,
        flat_descriptor(nb),
        STAGE_B,
        TF32_CODE,
    );
    // FP32 `Dst`, and `Zero_Flag_disabled_src` clear: the other setting is the
    // "keep SrcB denormals" mode `MVMUL.md` says must not be used.
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    words
        .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 0)
        .unwrap();
    let mut buf = vec![sfpu::nop(); words.program_len()];
    let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
    up.extend_from_slice(&buf[..k]);

    let unpack = encode::UnpacrRegular::ZERO.multi_context_mode(1);
    let banks = Banks::after_reset();
    up.push(set_adc_x(Unpacker::SrcA, 0, na - 1));
    let (i, banks) = banks.unpack_a(unpack).unwrap();
    up.push(i);
    up.push(set_adc_x(Unpacker::SrcB, 0, nb - 1));
    let (i, banks) = banks.unpack_b(unpack).unwrap();
    up.push(i);
    up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    up.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());

    let mut math = vec![tt_tests::datapath::state_id()];
    // Address modifier 0 moves nothing; 1 advances the fidelity phase only (with
    // the measured Blackhole `AddrMod` position; divergence row 42).
    math.push(thread_entry(thread::ADDR_MOD_AB_SEC0_SrcAIncr, 0));
    math.push(thread_entry(thread::ADDR_MOD_DST_SEC0_DestIncr, 0));
    math.push(thread_entry(thread::ADDR_MOD_AB_SEC1_SrcAIncr, 0));
    math.push(thread_entry(thread::ADDR_MOD_DST_SEC1_FidelityIncr, 1));
    math.push(fidelity_base(0));
    // `SrcAVal` is four bits, so 16 takes two steps: set 8, then add the carried
    // 8 (`SETRWC.md`: `if (SrcACr) SrcAVal += RWC.SrcA_Cr`).
    math.push(encode::Setrwc::ZERO.src_a(1).src_a_val(8).encode().unwrap());
    math.push(
        encode::Setrwc::ZERO
            .src_a(1)
            .src_a_cr(1)
            .src_a_val(SRC_A_ROW as u32 - 8)
            .src_b(1)
            .src_b_val(SRC_B_ROW as u32)
            .dst(1)
            .dst_val(0)
            .fidelity(1)
            .encode()
            .unwrap(),
    );
    // All of `Dst`: mode 3 is `CLR_ALL` in LLK's encoding too (row 40).
    // (mode, use_dst32b, addr_mod, imm10)
    math.push(encode::zeroacc(3, 0, 0, 0).unwrap());
    body(banks, &mut math);
    math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
    (up, math)
}

/// One `MVMUL` at `dst_row`, releasing both operands.
fn once(dst_row: u32) -> Body {
    Box::new(move |banks, p| {
        let (i, _) = banks
            .mvmul_release_both(encode::Mvmul::ZERO.dst_row(dst_row))
            .unwrap();
        p.push(i);
    })
}

/// `Dst` rows 0..16 after running `body` over `a` and `b`.
fn run(a: &MatA, b: &MatB, body: Body) -> Vec<u32> {
    run_and_pack(a, b, body, 0).0
}

/// A sentinel pre-written over the packer's output, so a word it never wrote is
/// distinguishable from one it wrote as zero.
const L1_SENTINEL: u32 = 0xA5A5_5A5A;

/// Datums of L1 read back after a pack: sixteen rows' worth, so a pack that
/// runs past what it was asked for lands on sentinel the gate can see.
const PACK_READBACK: usize = 16 * ROW;

/// [`run`], then pack `pack_rows` rows of `Dst` from row 0 to [`OUT`] on the
/// pack thread (`datapath::pack_rows`). Returns `(Dst rows 0..16, the
/// PACK_READBACK words at OUT)`; with `pack_rows == 0` there is no pack role
/// and the second is empty.
fn run_and_pack(a: &MatA, b: &MatB, body: Body, pack_rows: u32) -> (Vec<u32>, Vec<u32>) {
    let (sa, na) = stage_operand(SRC_A_ROW, a);
    let (sb, nb) = stage_operand(SRC_B_ROW, b);
    let (unpack, math) = program(na, nb, body);
    let pack = if pack_rows == 0 {
        Vec::new()
    } else {
        let mut words = ConfigWords::new();
        pack_config(&mut words, OUT);
        let mut p = config_program(&words);
        p.extend(datapath::pack_rows(pack_rows));
        p.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
        p
    };
    let sentinel: Vec<u8> = L1_SENTINEL
        .to_le_bytes()
        .iter()
        .copied()
        .cycle()
        .take(PACK_READBACK * 4)
        .collect();
    // `fork_scope` gives the child no return channel.
    let path = std::env::temp_dir().join(format!(
        "ttmvmul-{}-{:?}.bin",
        std::process::id(),
        std::thread::current().id()
    ));
    harness::in_device(|dev| {
        let roles = harness::Roles {
            unpack: &unpack,
            math: &math,
            pack: &pack,
        };
        let readback = [(OUT, PACK_READBACK * 4)];
        let out = harness::run(
            dev,
            &Run::roles(roles)
                .stage(&[(STAGE_A, &sa), (STAGE_B, &sb), (OUT, &sentinel)])
                .dump_rows(16)
                .read_back(if pack_rows == 0 { &[] } else { &readback }),
        );
        let mut bytes: Vec<u8> = (0..16 * ROW)
            .flat_map(|f| out.dst_at(f / ROW, f % ROW).to_le_bytes())
            .collect();
        if let Some(l1) = out.l1.first() {
            bytes.extend_from_slice(l1);
        }
        std::fs::write(&path, bytes).unwrap();
    });
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (dst, l1) = words.split_at(16 * ROW);
    (dst.to_vec(), l1.to_vec())
}

fn assert_block(dst: &[u32], first_row: usize, want: &MatB, what: &str) {
    for (i, want_row) in want.iter().enumerate() {
        for (j, w) in want_row.iter().enumerate() {
            let got = dst[(first_row + i) * ROW + j];
            assert_eq!(
                got,
                w.to_bits(),
                "{what}: Dst[{}][{j}] is {got:#010x} = {}, want {w}",
                first_row + i,
                f32::from_bits(got),
            );
        }
    }
}

/// A deterministic generator; CI is flake-free on purpose.
struct Lcg(u64);

impl Lcg {
    /// A uniformly chosen integer in `-bound..=bound`, as an `f32`.
    fn int(&mut self, bound: i32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as i32 % (2 * bound + 1) - bound) as f32
    }
}

fn identity() -> MatA {
    let mut a = [[0f32; 16]; 16];
    for (i, row) in a.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    a
}

fn truncated<const R: usize>(m: &[[f32; 16]; R]) -> [[f32; 16]; R] {
    let mut out = *m;
    for v in out.iter_mut().flatten() {
        *v = f32::from_bits(fp32_to_tf32(v.to_bits()));
    }
    out
}

#[test]
fn an_identity_src_a_returns_src_b() {
    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = 1.0 + k as f32;
    }
    let dst = run(&identity(), &b, once(0));
    assert_block(&dst, 0, &b, "SrcB @ I");
}

/// Small integers, where the device, the documented model and Burn must all agree
/// exactly: phase 0 takes five significant bits of `SrcA` and seven of `SrcB`, so
/// `|A| <= 31` and `|B| <= 127` lose nothing, and every sum stays under 2^24.
#[test]
fn small_integer_products_match_the_model_and_burn() {
    use burn_tensor::{Tensor, TensorData};
    type B = burn_flex::Flex;
    let device = burn_flex::FlexDevice;

    let mut rng = Lcg(0x5eed);
    let mut a = [[0f32; 16]; 16];
    let mut b = [[0f32; 16]; 8];
    a.iter_mut().flatten().for_each(|v| *v = rng.int(31));
    b.iter_mut().flatten().for_each(|v| *v = rng.int(127));

    let model = mvmul_reference(&ZERO_DST, &b, &a, &[0]).expect("small integers are exact");
    let flat = |m: &[[f32; 16]]| m.iter().flatten().copied().collect::<Vec<f32>>();
    let tb = Tensor::<B, 2>::from_data(TensorData::new(flat(&b), [8, 16]), &device);
    let ta = Tensor::<B, 2>::from_data(TensorData::new(flat(&a), [16, 16]), &device);
    let burn: Vec<f32> = tb.clone().matmul(ta.clone()).into_data().to_vec().unwrap();
    let mut burn_block = ZERO_DST;
    for (k, v) in burn.iter().enumerate() {
        burn_block[k / 16][k % 16] = *v;
    }
    assert_eq!(
        model, burn_block,
        "the model and Burn must agree before either judges the device"
    );

    let dst = run(&a, &b, once(0));
    assert_block(&dst, 0, &burn_block, "SrcB @ SrcA");

    // Control: the gate distinguishes `SrcB @ SrcA` from `SrcB @ SrcA^T`.
    let transposed: Vec<f32> = tb.matmul(ta.transpose()).into_data().to_vec().unwrap();
    assert!((0..128).any(|k| dst[k] != transposed[k].to_bits()));
}

/// The packer carries a Matrix Unit result out to L1: the 8x16 product of
/// `MVMUL`, packed by the pack thread in two `PACR`s of four rows, lands in L1
/// datum for datum as `Dst` holds it and as the model predicts, and nothing is
/// written past it.
///
/// This is the first gate in which all three roles do work, and the path every
/// later matmul result leaves by: the `Dst` dump holds one face at most, a tile
/// is four.
#[test]
fn the_packer_writes_the_matmul_result_to_l1() {
    let (a, b) = small_integer_operands(0xfeed);
    let model = mvmul_reference(&ZERO_DST, &b, &a, &[0]).expect("small integers are exact");
    let (dst, l1) = run_and_pack(&a, &b, once(0), 8);
    assert_block(&dst, 0, &model, "Dst");
    assert_block(&l1, 0, &model, "L1");
    assert!(
        l1[8 * ROW..].iter().all(|&w| w == L1_SENTINEL),
        "the packer wrote past the eight rows it was asked for"
    );
    // Control: the result is the product, not either operand or zero.
    assert!(l1[..8 * ROW].iter().any(|&w| w != 0));
    assert!((0..8 * ROW).any(|k| l1[k] != b[k / ROW][k % ROW].to_bits()));
}

/// A final partial group of rows is packed with the mask `PACR.md` gives for
/// it, `(1 << remaining) - 1`, and stops there: six rows are 96 datums, the
/// seventh row's L1 is untouched.
#[test]
fn a_partial_final_group_packs_only_its_rows() {
    let (a, b) = small_integer_operands(0xfeed);
    let model = mvmul_reference(&ZERO_DST, &b, &a, &[0]).expect("small integers are exact");
    let (_, l1) = run_and_pack(&a, &b, once(0), 6);
    let mut six = ZERO_DST;
    six[..6].copy_from_slice(&model[..6]);
    for (k, &w) in l1.iter().enumerate().take(6 * ROW) {
        assert_eq!(w, six[k / ROW][k % ROW].to_bits(), "L1 datum {k}");
    }
    assert!(
        l1[6 * ROW..].iter().all(|&w| w == L1_SENTINEL),
        "a six-row pack wrote past row 6"
    );
}

/// A math program far longer than the 256 words a mailbox once held runs to the
/// end: two thousand `SFPNOP`s and then the `MVMUL`, whose product is exact
/// only if every word before it was pushed from the program slot
/// (`mailbox::PROGRAM_REGION`) in order.
#[test]
fn a_program_longer_than_the_old_mailbox_limit_runs_to_the_end() {
    let (a, b) = small_integer_operands(0xbeef);
    let model = mvmul_reference(&ZERO_DST, &b, &a, &[0]).expect("small integers are exact");
    let body: Body = Box::new(|banks, p| {
        p.extend(std::iter::repeat_n(sfpu::nop(), 2000));
        let (i, _) = banks
            .mvmul_release_both(encode::Mvmul::ZERO.dst_row(0))
            .unwrap();
        p.push(i);
    });
    let dst = run(&a, &b, body);
    assert_block(&dst, 0, &model, "after 2000 SFPNOPs");
}

fn small_integer_operands(seed: u64) -> (MatA, MatB) {
    let mut rng = Lcg(seed);
    let mut a = [[0f32; 16]; 16];
    let mut b = [[0f32; 16]; 8];
    a.iter_mut().flatten().for_each(|v| *v = rng.int(31));
    b.iter_mut().flatten().for_each(|v| *v = rng.int(127));
    (a, b)
}

/// `MVMUL` is `+=`: without a `ZEROACC` between them, two give twice the product.
#[test]
fn mvmul_accumulates_into_dst() {
    let mut rng = Lcg(7);
    let mut a = [[0f32; 16]; 16];
    let mut b = [[0f32; 16]; 8];
    a.iter_mut().flatten().for_each(|v| *v = rng.int(15));
    b.iter_mut().flatten().for_each(|v| *v = rng.int(63));
    let first = mvmul_reference(&ZERO_DST, &b, &a, &[0]).unwrap();
    let twice = mvmul_reference(&first, &b, &a, &[0]).unwrap();
    assert_ne!(first, twice);
    let dst = run(
        &a,
        &b,
        Box::new(|banks, p| {
            let (i, banks) = banks.mvmul(encode::Mvmul::ZERO).unwrap();
            p.push(i);
            let (i, _) = banks.mvmul_release_both(encode::Mvmul::ZERO).unwrap();
            p.push(i);
        }),
    );
    assert_block(&dst, 0, &twice, "two MVMULs");
}

/// `DstRow` places the eight output rows; the eight it does not name stay zeroed.
#[test]
fn dst_row_places_the_block() {
    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = -(k as f32) - 1.0;
    }
    let dst = run(&identity(), &b, once(8));
    assert_block(&dst, 8, &b, "DstRow = 8");
    assert_block(
        &dst,
        0,
        &ZERO_DST,
        "rows 0..8, which DstRow = 8 must not touch",
    );
}

/// Operands whose mantissas use bits in every phase's share, so each phase's
/// contribution is visible.
fn fidelity_operands() -> (MatA, MatB) {
    let mut rng = Lcg(0xf1de);
    let mut a = [[0f32; 16]; 16];
    let mut b = [[0f32; 16]; 8];
    // A: an integer part phase 0 sees, plus 2^-5, which only the odd phases see.
    // B: likewise plus 2^-7, which only phases 2 and 3 see. Kept that narrow so
    // every sum of sixteen products fits FP32's 24 bits -- with a 2^-9 term too
    // the sums need 27, and `mvmul_reference` rightly refuses to answer.
    for v in a.iter_mut().flatten() {
        *v = 1.0 + rng.int(3).abs() + 2f32.powi(-5);
    }
    for v in b.iter_mut().flatten() {
        *v = 1.0 + rng.int(3).abs() + 2f32.powi(-7);
    }
    (truncated(&a), truncated(&b))
}

#[test]
fn phase_zero_alone_drops_exactly_the_bits_the_model_says() {
    let (a, b) = fidelity_operands();
    let model = mvmul_reference(&ZERO_DST, &b, &a, &[0]).unwrap();
    let full = mvmul_reference(&ZERO_DST, &b, &a, &[0, 1, 2, 3]).unwrap();
    assert_ne!(model, full, "the operands must exercise the later phases");
    let dst = run(&a, &b, once(0));
    assert_block(&dst, 0, &model, "phase 0 only");
}

/// All four phases, selected with `FIDELITY_BASE_Phase` between `MVMUL`s.
#[test]
fn four_phases_through_fidelity_base_recover_the_exact_product() {
    let (a, b) = fidelity_operands();
    let full = mvmul_reference(&ZERO_DST, &b, &a, &[0, 1, 2, 3]).unwrap();
    // The four phases are the whole product, computed independently.
    for (i, row) in full.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let exact: f64 = (0..16)
                .map(|k| f64::from(b[i][k]) * f64::from(a[k][j]))
                .sum();
            assert_eq!(f64::from(*v), exact);
        }
    }
    let dst = run(
        &a,
        &b,
        Box::new(|mut banks, p| {
            for phase in 0..3 {
                p.push(fidelity_base(phase));
                let (i, next) = banks.mvmul(encode::Mvmul::ZERO).unwrap();
                p.push(i);
                banks = next;
            }
            p.push(fidelity_base(3));
            let (i, _) = banks.mvmul_release_both(encode::Mvmul::ZERO).unwrap();
            p.push(i);
        }),
    );
    assert_block(&dst, 0, &full, "phases 0..4 via FIDELITY_BASE");
}

/// All four phases the way a kernel steps them: address modifier 1 increments the
/// RWC `FidelityPhase` after each `MVMUL` (`RWCs.md`, `ApplyAddrMod`). This is the
/// gate that found row 42: with the Wormhole encoding of `AddrMod`, it summed phase
/// 0 four times.
#[test]
fn four_phases_through_the_rwc_recover_the_exact_product() {
    let (a, b) = fidelity_operands();
    let full = mvmul_reference(&ZERO_DST, &b, &a, &[0, 1, 2, 3]).unwrap();
    let dst = run(
        &a,
        &b,
        Box::new(|mut banks, p| {
            for _ in 0..3 {
                let (i, next) = banks.mvmul(encode::Mvmul::ZERO.addr_mod(1)).unwrap();
                p.push(i);
                banks = next;
            }
            let (i, _) = banks
                .mvmul_release_both(encode::Mvmul::ZERO.addr_mod(1))
                .unwrap();
            p.push(i);
        }),
    );
    assert_block(&dst, 0, &full, "phases 0..4 via ADDR_MOD FidelityIncr");
}

/// **`MVMUL`'s `AddrMod` is bits 14..16 on Blackhole, not the 15..16 of
/// `Bits32.lua`.** The only diagram in the pinned specification is Wormhole's, and
/// it is wrong here: its `addr_mod(1)` sets bit 15, which Blackhole reads as index
/// **2**, so it applies the wrong modifier and says nothing (divergence row 42).
///
/// This gate is the evidence `xtask/src/gen_isa/Bits32_BH.lua` cites for the
/// measured `MVMUL_BH` layout, which is what `encode::Mvmul` now generates; the
/// Wormhole diagram lives on as `encode::wormhole::Mvmul`. `gen-isa` refuses the
/// layout if this test disappears.
///
/// Measured with a modifier that advances `Dst` by 8 in entry 1 and nothing in
/// entry 2: a second `MVMUL` lands at row 8 only when the first selected entry 1.
#[test]
fn mvmul_addr_mod_sits_one_bit_lower_on_blackhole() {
    let blackhole = encode::Mvmul::ZERO.addr_mod(1).encode().unwrap().word();
    let wormhole = encode::wormhole::Mvmul::ZERO
        .addr_mod(1)
        .encode()
        .unwrap()
        .word();
    assert_eq!(blackhole & 0x0001_c000, 1 << 14, "the measured layout");
    assert_eq!(wormhole & 0x0001_c000, 1 << 15, "the Wormhole diagram");

    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = 1.0 + k as f32;
    }
    // The first `MVMUL` carries `addr_mod_bits` on top of the typestate's word.
    let body = |addr_mod_bits: u32| -> Body {
        Box::new(move |banks, p| {
            p.push(thread_entry(thread::ADDR_MOD_DST_SEC1_DestIncr, 8));
            p.push(thread_entry(thread::ADDR_MOD_DST_SEC2_DestIncr, 0));
            p.push(thread_entry(thread::ADDR_MOD_DST_SEC4_DestIncr, 8));
            let (i, banks) = banks.mvmul(encode::Mvmul::ZERO).unwrap();
            p.push(Instruction::new(i.word() | addr_mod_bits, i.def()));
            let (i, _) = banks.mvmul_release_both(encode::Mvmul::ZERO).unwrap();
            p.push(i);
        })
    };
    let doubled = {
        let mut d = b;
        d.iter_mut().flatten().for_each(|v| *v *= 2.0);
        d
    };

    // The measured encoding: modifier 1, so the second `MVMUL` goes eight rows down.
    let dst = run(&identity(), &b, body(blackhole & 0x0001_c000));
    assert_block(&dst, 0, &b, "first MVMUL, Blackhole addr_mod(1)");
    assert_block(&dst, 8, &b, "second MVMUL after modifier 1");

    // The third bit alone: entry 4, set up as entry 1.
    let four = addr_mod_bits(encode::Mvmul::ZERO.encode().unwrap(), 4);
    let dst = run(&identity(), &b, body(four));
    assert_block(&dst, 0, &b, "first MVMUL, Blackhole addr_mod(4)");
    assert_block(&dst, 8, &b, "second MVMUL after modifier 4");

    // The Wormhole encoding of the same request: modifier 2, which moves nothing,
    // so both land on row 0.
    let dst = run(&identity(), &b, body(wormhole & 0x0001_c000));
    assert_block(&dst, 0, &doubled, "both MVMULs, Wormhole addr_mod(1)");
}

/// The whole identity result, printed next to what it should be. Silicon only:
/// a diagnostic for the open `Src`-path losses, not a claim.
#[cfg(feature = "silicon")]
#[test]
fn dump_the_identity_block() {
    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = 1.0 + k as f32;
    }
    let dst = run(&identity(), &b, once(0));
    for (r, want_row) in b.iter().enumerate() {
        let got: Vec<String> = (0..16)
            .map(|c| format!("{:5}", f32::from_bits(dst[r * ROW + c])))
            .collect();
        let want: Vec<String> = want_row.iter().map(|w| format!("{w:5}")).collect();
        println!("row {r} got  {}", got.join(" "));
        println!("row {r} want {}", want.join(" "));
    }
}

/// `MOVD2A`'s, `MOVD2B`'s and `MOVB2A`'s `AddrMod` sit at bits 14..16 on
/// Blackhole, one bit lower than the Wormhole diagrams draw them -- as `MVMUL`'s,
/// `MOVA2D`'s and `MOVB2D`'s do (row 42), and as LLK's `addr_mode << 14` has them.
///
/// Each case moves four rows into `Src` twice, both moves carrying `addr_mod(1)`,
/// where entry 1 advances the counter of the rows being *read* by four and entry
/// 2 advances nothing. So the second move reads the next four rows with the
/// measured encoding and the same four again with the Wormhole one (modifier 2).
/// The written `Src` rows are then copied into `Dst` with `MOVA2D`/`MOVB2D`.
/// Four-row moves because ttsim implements no other form of these (as for
/// `MOVA2D`, row 37: `tensix_movd2a: instr_mod=0` is `UnsupportedFunctionality`).
///
/// At the body's start the RWCs are `SrcA` 16, `SrcB` 8, `Dst` 0 (`program`), so
/// `Src` row arguments here are offsets from the staged operands.
#[test]
fn mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole() {
    let words = [
        (
            "MOVD2A",
            encode::Movd2A::ZERO.addr_mod(1).encode().unwrap().word(),
            encode::wormhole::Movd2A::ZERO
                .addr_mod(1)
                .encode()
                .unwrap()
                .word(),
        ),
        (
            "MOVD2B",
            encode::Movd2B::ZERO.addr_mod(1).encode().unwrap().word(),
            encode::wormhole::Movd2B::ZERO
                .addr_mod(1)
                .encode()
                .unwrap()
                .word(),
        ),
        (
            "MOVB2A",
            // (src_a_row, addr_mod, move4_rows, src_b_row)
            encode::movb2_a(0, 1, 0, 0).unwrap().word(),
            encode::wormhole::movb2_a(0, 1, 0, 0).unwrap().word(),
        ),
    ];
    for (name, bh, wh) in words {
        assert_eq!(bh & 0x00ff_ffff, 1 << 14, "{name}: the measured layout");
        assert_eq!(wh & 0x00ff_ffff, 1 << 15, "{name}: the Wormhole diagram");
    }

    // Every datum distinct and exact in TF32, so a row names where it came from.
    let mut a = [[0f32; 16]; 16];
    for (k, v) in a.iter_mut().flatten().enumerate() {
        *v = 1.0 + k as f32;
    }
    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = 1000.0 + k as f32;
    }
    let row_bits = |r: &[f32; 16]| -> Vec<u32> { r.iter().map(|v| v.to_bits()).collect() };
    let dst_row = |dst: &[u32], r: usize| dst[r * ROW..(r + 1) * ROW].to_vec();

    // Entry 1 advances `SrcB` by `b_incr` and `Dst` by `d_incr`; entry 2, nothing.
    let modifiers = |p: &mut Vec<Instruction>, b_incr: u16, d_incr: u16| {
        p.push(thread_entry(thread::ADDR_MOD_AB_SEC1_SrcBIncr, b_incr));
        p.push(thread_entry(thread::ADDR_MOD_DST_SEC1_DestIncr, d_incr));
        p.push(thread_entry(thread::ADDR_MOD_AB_SEC2_SrcBIncr, 0));
        p.push(thread_entry(thread::ADDR_MOD_DST_SEC2_DestIncr, 0));
        // Entry 4, reached by the third bit alone, as entry 1.
        p.push(thread_entry(thread::ADDR_MOD_AB_SEC4_SrcBIncr, b_incr));
        p.push(thread_entry(thread::ADDR_MOD_DST_SEC4_DestIncr, d_incr));
    };
    let raw = |i: Instruction, bits: u32| Instruction::new(i.word() | bits, i.def());
    let dst_rwc_zero = || encode::Setrwc::ZERO.dst(1).dst_val(0).encode().unwrap();

    for (addr_mod, encoding) in [
        (words[0].1, "measured"),
        (
            addr_mod_bits(encode::Movd2A::ZERO.encode().unwrap(), 4),
            "measured entry 4",
        ),
        (words[0].2, "Wormhole"),
    ] {
        let bits = addr_mod & 0x00ff_ffff;
        let second = if encoding == "Wormhole" { 0 } else { 4 };

        // MOVD2A: B rows 0..8 into `Dst` 0..8; `Dst` 0..4, then `Dst` RWC..+4,
        // into `SrcA` 24..28 and 28..32; `SrcA` 24..32 back into `Dst` 8..16.
        let dst = run(
            &a,
            &b,
            Box::new(move |_banks, p| {
                modifiers(p, 0, 4);
                for row in [0, 4] {
                    p.push(
                        encode::Movb2D::ZERO
                            .move4_rows(1)
                            .src_row(row)
                            .dst_row(row)
                            .encode()
                            .unwrap(),
                    );
                }
                for src_row in [8, 12] {
                    let i = encode::Movd2A::ZERO
                        .move4_rows(1)
                        .src_row(src_row)
                        .encode()
                        .unwrap();
                    p.push(raw(i, bits));
                }
                p.push(dst_rwc_zero());
                p.push(
                    encode::Mova2D::ZERO
                        .move8_rows(1)
                        .src_row(8)
                        .dst_row(8)
                        .encode()
                        .unwrap(),
                );
            }),
        );
        assert_eq!(
            dst_row(&dst, 8),
            row_bits(&b[0]),
            "MOVD2A {encoding}: first"
        );
        assert_eq!(
            dst_row(&dst, 12),
            row_bits(&b[second]),
            "MOVD2A {encoding}: second"
        );
    }

    for (addr_mod, encoding) in [
        (words[1].1, "measured"),
        (
            addr_mod_bits(encode::Movd2B::ZERO.encode().unwrap(), 4),
            "measured entry 4",
        ),
        (words[1].2, "Wormhole"),
    ] {
        let bits = addr_mod & 0x00ff_ffff;
        let second = if encoding == "Wormhole" { 0 } else { 4 };

        // MOVD2B: A rows 0..8 into `Dst` 0..8; `Dst` 0..4, then `Dst` RWC..+4,
        // into `SrcB` 16..20 and 20..24; those back into `Dst` 8..16.
        let dst = run(
            &a,
            &b,
            Box::new(move |_banks, p| {
                modifiers(p, 0, 4);
                p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
                for src_row in [8, 12] {
                    let i = encode::Movd2B::ZERO
                        .move4_rows(1)
                        .src_row(src_row)
                        .encode()
                        .unwrap();
                    p.push(raw(i, bits));
                }
                p.push(dst_rwc_zero());
                for row in [8, 12] {
                    p.push(
                        encode::Movb2D::ZERO
                            .move4_rows(1)
                            .src_row(row)
                            .dst_row(row)
                            .encode()
                            .unwrap(),
                    );
                }
            }),
        );
        assert_eq!(
            dst_row(&dst, 8),
            row_bits(&a[0]),
            "MOVD2B {encoding}: first"
        );
        assert_eq!(
            dst_row(&dst, 12),
            row_bits(&a[second]),
            "MOVD2B {encoding}: second"
        );
    }

    for (addr_mod, encoding) in [
        (words[2].1, "measured"),
        (
            addr_mod_bits(encode::movb2_a(0, 0, 0, 0).unwrap(), 4),
            "measured entry 4",
        ),
        (words[2].2, "Wormhole"),
    ] {
        let bits = addr_mod & 0x00ff_ffff;
        let second = if encoding == "Wormhole" { 0 } else { 4 };

        // MOVB2A: `SrcB` 8..12, then `SrcB` RWC..+4 (B rows 0..4, then 4..8 or
        // 0..4 again), into `SrcA` 24..28 and 28..32; `SrcA` 24..32 back into
        // `Dst` 0..8.
        let dst = run(
            &a,
            &b,
            Box::new(move |_banks, p| {
                modifiers(p, 4, 0);
                for src_a_row in [8, 12] {
                    // (src_a_row, addr_mod, move4_rows, src_b_row)
                    let i = encode::movb2_a(src_a_row, 0, 1, 0).unwrap();
                    p.push(raw(i, bits));
                }
                p.push(
                    encode::Mova2D::ZERO
                        .move8_rows(1)
                        .src_row(8)
                        .encode()
                        .unwrap(),
                );
            }),
        );
        assert_eq!(
            dst_row(&dst, 0),
            row_bits(&b[0]),
            "MOVB2A {encoding}: first"
        );
        assert_eq!(
            dst_row(&dst, 4),
            row_bits(&b[second]),
            "MOVB2A {encoding}: second"
        );
    }
}

/// The other Matrix Unit instructions that write `Dst` -- `ELWADD`, `ELWSUB`,
/// `ELWMUL`, `DOTPV`, `MOVDBGA2D` -- have their `AddrMod` at bits 14..16 on
/// Blackhole too, where LLK's `addr_mode << 14` has it; the Wormhole diagrams
/// draw 15..16.
///
/// Semantics-free, so it holds whatever each instruction computes: a reference
/// run of one instruction gives the eight rows it writes at `Dst` row 0. Then two
/// copies, each carrying `addr_mod(1)`, where entry 1 advances `Dst` by 8 and
/// entry 2 by nothing: with the measured encoding the second copy writes the same
/// rows again 8 rows lower; with the Wormhole one (modifier 2) nothing reaches row
/// 8. The `Src` counters stay put, so both copies read the same operands.
#[test]
fn matrix_unit_addr_mod_sits_one_bit_lower_on_blackhole() {
    // (src_a_row, addr_mod, move8_rows, ...) orders follow the generated encoders.
    let cases: [(&str, Instruction, u32, u32); 5] = [
        (
            "ELWADD",
            encode::Elwadd::ZERO.encode().unwrap(),
            encode::Elwadd::ZERO.addr_mod(1).encode().unwrap().word(),
            encode::wormhole::Elwadd::ZERO
                .addr_mod(1)
                .encode()
                .unwrap()
                .word(),
        ),
        (
            "ELWSUB",
            encode::Elwsub::ZERO.encode().unwrap(),
            encode::Elwsub::ZERO.addr_mod(1).encode().unwrap().word(),
            encode::wormhole::Elwsub::ZERO
                .addr_mod(1)
                .encode()
                .unwrap()
                .word(),
        ),
        (
            "ELWMUL",
            encode::Elwmul::ZERO.encode().unwrap(),
            encode::Elwmul::ZERO.addr_mod(1).encode().unwrap().word(),
            encode::wormhole::Elwmul::ZERO
                .addr_mod(1)
                .encode()
                .unwrap()
                .word(),
        ),
        (
            "DOTPV",
            // (flip_src_b, flip_src_a, addr_mod, dst_row)
            encode::dotpv(0, 0, 0, 0).unwrap(),
            encode::dotpv(0, 0, 1, 0).unwrap().word(),
            encode::wormhole::dotpv(0, 0, 1, 0).unwrap().word(),
        ),
        (
            "MOVDBGA2D",
            // ttsim implements only the eight-row form of `MOVA2D` (row 37).
            encode::Movdbga2D::ZERO.move8_rows(1).encode().unwrap(),
            encode::Movdbga2D::ZERO.addr_mod(1).encode().unwrap().word(),
            encode::wormhole::Movdbga2D::ZERO
                .addr_mod(1)
                .encode()
                .unwrap()
                .word(),
        ),
    ];
    for (name, _, bh, wh) in &cases {
        assert_eq!(bh & 0x00ff_ffff, 1 << 14, "{name}: the measured layout");
        assert_eq!(wh & 0x00ff_ffff, 1 << 15, "{name}: the Wormhole diagram");
    }

    // Distinct, nonzero, and far enough apart that neither `A - B` nor `A * B`
    // is zero anywhere.
    let mut a = [[0f32; 16]; 16];
    for (k, v) in a.iter_mut().flatten().enumerate() {
        *v = 1.0 + k as f32;
    }
    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = 1000.0 + k as f32;
    }
    let block = |dst: &[u32], r: usize| dst[r * ROW..(r + 8) * ROW].to_vec();

    for (name, one, bh, wh) in cases {
        if matches!(name, "DOTPV" | "MOVDBGA2D") && !cfg!(feature = "silicon") {
            // `tensix_decode_dotpv`/`_movdbga2d` are `UnsupportedFunctionality`;
            // `ttsim_implements_no_dotpv_shiftxb_or_movdbga2d` pins that.
            continue;
        }
        // Entry 1 advances `Dst` by `step`, entry 2 by nothing, neither touches `Src`.
        let twice = move |step: u16, bits: u32| -> Body {
            Box::new(move |_banks, p| {
                p.push(thread_entry(thread::ADDR_MOD_AB_SEC1_SrcAIncr, 0));
                p.push(thread_entry(thread::ADDR_MOD_DST_SEC1_DestIncr, step));
                p.push(thread_entry(thread::ADDR_MOD_AB_SEC2_SrcAIncr, 0));
                p.push(thread_entry(thread::ADDR_MOD_DST_SEC2_DestIncr, 0));
                p.push(thread_entry(thread::ADDR_MOD_AB_SEC4_SrcAIncr, 0));
                p.push(thread_entry(thread::ADDR_MOD_DST_SEC4_DestIncr, step));
                for _ in 0..2 {
                    p.push(Instruction::new(one.word() | bits, one.def()));
                }
            })
        };
        let reference = run(&a, &b, Box::new(move |_banks, p| p.push(one)));
        let written = block(&reference, 0);
        assert!(
            written.iter().any(|&v| v != 0),
            "{name}: the reference run wrote nothing"
        );
        assert!(
            block(&reference, 8).iter().all(|&v| v == 0),
            "{name}: the reference run wrote past eight rows"
        );

        for bits in [bh & 0x00ff_ffff, addr_mod_bits(one, 4)] {
            let dst = run(&a, &b, twice(8, bits));
            assert_eq!(
                block(&dst, 8),
                written,
                "{name} measured, AddrMod bits {bits:#x}: second copy 8 rows down"
            );
        }

        let dst = run(&a, &b, twice(8, wh & 0x00ff_ffff));
        assert!(
            block(&dst, 8).iter().all(|&v| v == 0),
            "{name} Wormhole: nothing should reach row 8"
        );
    }
}

/// `SHIFTXB`'s `AddrMod` sits at bits 14..16 on Blackhole, as LLK's
/// `addr_mode << 14` has it; the Wormhole diagram draws 15..16.
///
/// `SHIFTXB` rotates one `SrcB` row left by a column (`ShiftInZero` clear). Two of
/// them at `SrcB` row 8 (offset 0 from the RWC), each carrying `addr_mod(1)`,
/// where entry 1 advances `SrcB` by one: the measured encoding rotates rows 8 and
/// 9 once each; the Wormhole one (modifier 2) rotates row 8 twice and leaves 9.
/// Read back with `MOVB2D`.
///
/// Silicon only: ttsim does not implement `SHIFTXB`
/// (`ttsim_implements_no_dotpv_shiftxb_or_movdbga2d`).
#[test]
#[cfg(feature = "silicon")]
fn shiftxb_addr_mod_sits_one_bit_lower_on_blackhole() {
    // (addr_mod, shift_in_zero, src_row)
    let bh = encode::shiftxb(1, 0, 0).unwrap().word();
    let wh = encode::wormhole::shiftxb(1, 0, 0).unwrap().word();
    assert_eq!(bh & 0x00ff_ffff, 1 << 14, "the measured layout");
    assert_eq!(wh & 0x00ff_ffff, 1 << 15, "the Wormhole diagram");

    let a = identity();
    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = 1000.0 + k as f32;
    }
    let rotated = |row: &[f32; 16], by: usize| -> Vec<u32> {
        (0..16).map(|c| row[(c + by) % 16].to_bits()).collect()
    };
    let dst_row = |dst: &[u32], r: usize| dst[r * ROW..(r + 1) * ROW].to_vec();
    let body = |bits: u32| -> Body {
        Box::new(move |_banks, p| {
            p.push(thread_entry(thread::ADDR_MOD_AB_SEC1_SrcBIncr, 1));
            p.push(thread_entry(thread::ADDR_MOD_DST_SEC1_DestIncr, 0));
            p.push(thread_entry(thread::ADDR_MOD_AB_SEC2_SrcBIncr, 0));
            p.push(thread_entry(thread::ADDR_MOD_DST_SEC2_DestIncr, 0));
            p.push(thread_entry(thread::ADDR_MOD_AB_SEC4_SrcBIncr, 1));
            p.push(thread_entry(thread::ADDR_MOD_DST_SEC4_DestIncr, 0));
            let i = encode::shiftxb(0, 0, 0).unwrap();
            for _ in 0..2 {
                p.push(Instruction::new(i.word() | bits, i.def()));
            }
            // Back to row 8 and copy `SrcB` 8..12 into `Dst` 0..4.
            p.push(
                encode::Setrwc::ZERO
                    .src_b(1)
                    .src_b_val(SRC_B_ROW as u32)
                    .encode()
                    .unwrap(),
            );
            p.push(encode::Movb2D::ZERO.move4_rows(1).encode().unwrap());
        })
    };

    for bits in [
        bh & 0x00ff_ffff,
        addr_mod_bits(encode::shiftxb(0, 0, 0).unwrap(), 4),
    ] {
        let dst = run(&a, &b, body(bits));
        assert_eq!(
            dst_row(&dst, 0),
            rotated(&b[0], 1),
            "measured: row 8 rotated once"
        );
        assert_eq!(
            dst_row(&dst, 1),
            rotated(&b[1], 1),
            "measured: row 9 rotated once"
        );
    }

    let dst = run(&a, &b, body(wh & 0x00ff_ffff));
    assert_eq!(
        dst_row(&dst, 0),
        rotated(&b[0], 2),
        "Wormhole: row 8 rotated twice"
    );
    assert_eq!(
        dst_row(&dst, 1),
        rotated(&b[1], 0),
        "Wormhole: row 9 untouched"
    );
}

/// Does ttsim run `body` over the step's operands to completion?
#[cfg(not(feature = "silicon"))]
fn survives_body(body: Body) -> bool {
    let (sa, na) = stage_operand(SRC_A_ROW, &identity());
    let (sb, nb) = stage_operand(SRC_B_ROW, &ZERO_DST);
    let (unpack, math) = program(na, nb, body);
    harness::survives(|dev| {
        let roles = harness::Roles {
            unpack: &unpack,
            math: &math,
            pack: &[],
        };
        harness::run(
            dev,
            &Run::roles(roles)
                .stage(&[(STAGE_A, &sa), (STAGE_B, &sb)])
                .dump_rows(16),
        );
    })
}

/// ttsim implements none of `DOTPV`, `SHIFTXB` and `MOVDBGA2D`
/// (`tensix_decode_dotpv`, `_shiftxb`, `_movdbga2d`: `UnsupportedFunctionality`),
/// so their `AddrMod` gates are silicon-only. Pinned so that ttsim learning any
/// of them shows up here.
#[test]
#[cfg(not(feature = "silicon"))]
fn ttsim_implements_no_dotpv_shiftxb_or_movdbga2d() {
    // A body that runs, so the refusals below are the instructions' own.
    assert!(survives_body(Box::new(|_banks, p| {
        p.push(encode::Elwadd::ZERO.encode().unwrap())
    })));
    assert!(!survives_body(Box::new(|_banks, p| {
        p.push(encode::dotpv(0, 0, 0, 0).unwrap())
    })));
    assert!(!survives_body(Box::new(|_banks, p| {
        p.push(encode::shiftxb(0, 0, 0).unwrap())
    })));
    assert!(!survives_body(Box::new(|_banks, p| {
        p.push(encode::Movdbga2D::ZERO.move8_rows(1).encode().unwrap())
    })));
}

/// `ZEROACC` on Blackhole, measured: `AddrMod` at bits 14..16 and `UseDst32b` at
/// bit 18, where LLK's `addr_mode << 14` and `use_32_bit_mode << 18` have them.
/// The Wormhole diagram draws 15..16 and 21, and puts `Revert` at 18.
///
/// Bit 21 is not tried: ttsim calls it undefined (row 40), and on Blackhole it
/// is the top of LLK's three-bit `clear_mode`, whose `_32B` modes nothing uses.
///
/// `Dst` is FP32 here (`program`), so a 32-bit row `r` consults the valid bit of
/// physical row `Adj32(r) = 2 * (r & !7) + (r & 7)` (`Dst.md`), and reads zero once
/// that bit is cleared.
#[test]
fn zeroacc_addr_mod_and_use_dst32b_on_blackhole() {
    // (mode, use_dst32b, addr_mod, imm10); the Wormhole one also takes `revert`.
    let bh = encode::zeroacc(0, 0, 1, 0).unwrap().word();
    let wh = encode::wormhole::Zeroacc::ZERO
        .addr_mod(1)
        .encode()
        .unwrap()
        .word();
    assert_eq!(bh & 0x00ff_ffff, 1 << 14, "AddrMod: the measured layout");
    assert_eq!(wh & 0x00ff_ffff, 1 << 15, "AddrMod: the Wormhole diagram");
    let wide = encode::zeroacc(0, 1, 0, 0).unwrap().word();
    let wh_wide = encode::wormhole::Zeroacc::ZERO
        .use_dst32b(1)
        .encode()
        .unwrap()
        .word();
    assert_eq!(
        wide & 0x00ff_ffff,
        1 << 18,
        "UseDst32b: the measured layout"
    );
    assert_eq!(
        wh_wide & 0x00ff_ffff,
        1 << 21,
        "UseDst32b: the Wormhole diagram"
    );

    let a = identity();
    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = 1000.0 + k as f32;
    }
    // `Dst` rows 0..16 filled from `SrcA` rows 16..32 (the identity).
    let fill = |p: &mut Vec<Instruction>| {
        for (src, dst) in [(0, 0), (8, 8)] {
            p.push(
                encode::Mova2D::ZERO
                    .move8_rows(1)
                    .src_row(src)
                    .dst_row(dst)
                    .encode()
                    .unwrap(),
            );
        }
    };
    let filled: Vec<u32> = truncated(&a)
        .iter()
        .flatten()
        .map(|v| v.to_bits())
        .collect();
    let rows_read_zero = |dst: &[u32]| -> Vec<usize> {
        (0..16)
            .filter(|&r| dst[r * ROW..(r + 1) * ROW].iter().all(|&v| v == 0))
            .collect()
    };
    let check = |dst: &[u32], cleared: &[usize], what: &str| {
        assert_eq!(rows_read_zero(dst), cleared, "{what}: rows reading zero");
        for r in (0..16).filter(|r| !cleared.contains(r)) {
            assert_eq!(
                &dst[r * ROW..(r + 1) * ROW],
                &filled[r * ROW..(r + 1) * ROW],
                "{what}: row {r} should be untouched"
            );
        }
    };

    // AddrMod, read back through the RWC it advances: a sixteen-row clear of
    // block 1 (32-bit rows 8..16) carrying the modifier, where entry 1 advances
    // `Dst` by 2, entry 2 by 3 and entry 4 by 6, then a plain one-row clear at the RWC. Not
    // one-row mode for the first clear: on silicon an odd modifier changes what
    // one-row mode clears (row 52).
    let advanced_by = move |bits: u32| -> Body {
        Box::new(move |_banks, p| {
            fill(p);
            p.push(thread_entry(thread::ADDR_MOD_AB_SEC1_SrcAIncr, 0));
            p.push(thread_entry(thread::ADDR_MOD_DST_SEC1_DestIncr, 2));
            p.push(thread_entry(thread::ADDR_MOD_AB_SEC2_SrcAIncr, 0));
            p.push(thread_entry(thread::ADDR_MOD_DST_SEC2_DestIncr, 3));
            p.push(thread_entry(thread::ADDR_MOD_AB_SEC4_SrcAIncr, 0));
            p.push(thread_entry(thread::ADDR_MOD_DST_SEC4_DestIncr, 6));
            let i = encode::zeroacc(1, 0, 0, 1).unwrap();
            p.push(Instruction::new(i.word() | bits, i.def()));
            p.push(encode::zeroacc(0, 0, 0, 0).unwrap());
        })
    };
    let block_1: Vec<usize> = (8..16).collect();
    let dst = run(&a, &b, advanced_by(bh & 0x00ff_ffff));
    check(
        &dst,
        &[&[2][..], &block_1].concat(),
        "AddrMod measured: entry 1",
    );
    let four = addr_mod_bits(encode::zeroacc(0, 0, 0, 0).unwrap(), 4);
    let dst = run(&a, &b, advanced_by(four));
    check(
        &dst,
        &[&[6][..], &block_1].concat(),
        "AddrMod measured: entry 4, by the third bit",
    );
    let dst = run(&a, &b, advanced_by(wh & 0x00ff_ffff));
    check(
        &dst,
        &[&[3][..], &block_1].concat(),
        "AddrMod Wormhole: entry 2",
    );

    // UseDst32b: sixteen-row mode, block 0. 16-bit, it clears physical rows
    // 0..16 -- 32-bit rows 0..8; 32-bit, physical 0..8 and 16..24 -- rows 0..16.
    let block_0 = move |bits: u32| -> Body {
        Box::new(move |_banks, p| {
            fill(p);
            let i = encode::zeroacc(1, 0, 0, 0).unwrap();
            p.push(Instruction::new(i.word() | bits, i.def()));
        })
    };
    let dst = run(&a, &b, block_0(0));
    check(&dst, &(0..8).collect::<Vec<_>>(), "UseDst32b clear");
    let dst = run(&a, &b, block_0(wide & 0x00ff_ffff));
    check(&dst, &(0..16).collect::<Vec<_>>(), "UseDst32b at bit 18");
}

/// `Dst` rows reading zero after `body` runs over a `Dst` whose rows 0..16 hold
/// the identity.
fn zeroacc_leaves_zero(body: impl FnOnce(&mut Vec<Instruction>) + 'static) -> Vec<usize> {
    let dst = run(
        &identity(),
        &ZERO_DST,
        Box::new(move |_banks, p| {
            for (src, dst) in [(0, 0), (8, 8)] {
                p.push(
                    encode::Mova2D::ZERO
                        .move8_rows(1)
                        .src_row(src)
                        .dst_row(dst)
                        .encode()
                        .unwrap(),
                );
            }
            body(p);
        }),
    );
    (0..16)
        .filter(|&r| dst[r * ROW..(r + 1) * ROW].iter().all(|&v| v == 0))
        .collect()
}

/// `ZEROACC.md` says one-row mode clears `DstRowValid[Adj32(Row)]` when `Dst` is
/// 32-bit, and ttsim does: `Imm10 = 9` empties 32-bit row 9. Silicon clears
/// physical row 9 instead -- the low half of 32-bit row 1, whose valid bit nothing
/// consults -- so no row reads zero; `Imm10 = 3` (physical 3 is `Adj32(3)`) looks
/// the same on both (row 51).
#[test]
fn zeroacc_one_row_in_32_bit_dst() {
    // (mode, use_dst32b, addr_mod, imm10)
    let three = zeroacc_leaves_zero(|p| p.push(encode::zeroacc(0, 0, 0, 3).unwrap()));
    assert_eq!(three, [3]);
    let nine = zeroacc_leaves_zero(|p| p.push(encode::zeroacc(0, 0, 0, 9).unwrap()));
    if cfg!(feature = "silicon") {
        assert_eq!(nine, [] as [usize; 0], "silicon: physical row 9");
    } else {
        assert_eq!(nine, [9], "ttsim: Adj32(9)");
    }
}

/// On silicon, one-row `ZEROACC` with an odd `AddrMod` also clears the eight
/// physical rows from its target -- entries 1, 3 and 5 alike, whatever the entry
/// holds -- while still applying the entry. Even modifiers clear one row. ttsim
/// clears one row either way, and neither `ZEROACC.md` nor LLK (which issues
/// one-row `ZEROACC` only with `ADDR_MOD_0`) says anything about it (row 52).
#[test]
fn zeroacc_one_row_with_an_odd_addr_mod() {
    for addr_mod in [1, 2, 3] {
        let zero = zeroacc_leaves_zero(move |p| {
            for f in [
                thread::ADDR_MOD_DST_SEC1_DestIncr,
                thread::ADDR_MOD_DST_SEC2_DestIncr,
                thread::ADDR_MOD_DST_SEC3_DestIncr,
            ] {
                p.push(thread_entry(f, 0));
            }
            p.push(encode::zeroacc(0, 0, addr_mod, 0).unwrap());
        });
        let want: Vec<usize> = if cfg!(feature = "silicon") && addr_mod % 2 == 1 {
            (0..8).collect()
        } else {
            vec![0]
        };
        assert_eq!(zero, want, "one-row ZEROACC with addr_mod {addr_mod}");
    }
}
